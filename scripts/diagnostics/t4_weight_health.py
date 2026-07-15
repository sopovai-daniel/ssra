"""T4 — weight-health snapshot (docs/cc/M2-spike-diagnostics.md §3).

Per checkpoint (e.g. 5000/6000/7000) and per module: ||W||_F, max |w|,
LN gain min/max, unembedding row-norm distribution (top-10 outlier rows +
summary stats), embedding norm. Flags monotone pre-spike drift (visible
already A->B) vs discontinuity only at B->C. Read-only (AP-20).

Usage:
  python scripts/diagnostics/t4_weight_health.py CKPT_A CKPT_B CKPT_C \
      [--csv results/M2-spike-diag-T4-params.csv]
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from ckpt_common import (ROOT, block_of, canonical_params, fmt_table,
                         layer_of, load_blob, sci)

DRIFT_X = 1.05  # flag threshold: >5 % blockwise norm move within one interval


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs=3, metavar="CKPT")
    ap.add_argument("--csv", default="results/M2-spike-diag-T4-params.csv")
    args = ap.parse_args()

    blobs = [load_blob(p) for p in args.ckpts]
    steps = [b["step"] for b in blobs]
    print(f"# T4 weight health: steps {steps}, run {blobs[0]['run_name']}")
    params = [canonical_params(b) for b in blobs]

    per_param = []
    for name in params[0]:
        row = {"param": name, "block": block_of(name),
               "layer": layer_of(name), "numel": params[0][name].numel()}
        for s, p in zip(steps, params):
            t = p[name].double()
            row[f"fro_{s}"] = t.norm().item()
            row[f"absmax_{s}"] = t.abs().max().item()
        per_param.append(row)

    csv_path = ROOT / args.csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_param[0].keys()))
        w.writeheader()
        w.writerows(per_param)
    print(f"full per-parameter table -> {csv_path}\n")

    # -- per-block snapshot with drift/discontinuity flags --------------------
    blocks: dict[str, dict] = {}
    for r in per_param:
        b = blocks.setdefault(r["block"], {f"fro2_{s}": 0.0 for s in steps}
                              | {f"absmax_{s}": 0.0 for s in steps})
        for s in steps:
            b[f"fro2_{s}"] += r[f"fro_{s}"] ** 2
            b[f"absmax_{s}"] = max(b[f"absmax_{s}"], r[f"absmax_{s}"])
    rows = []
    for name, b in sorted(blocks.items()):
        fro = [math.sqrt(b[f"fro2_{s}"]) for s in steps]
        x_ab = fro[1] / fro[0] if fro[0] else math.nan
        x_bc = fro[2] / fro[1] if fro[1] else math.nan
        flag = ""
        if x_ab > DRIFT_X and x_bc > DRIFT_X:
            flag = "MONOTONE-GROWTH"
        elif x_bc > DRIFT_X >= x_ab:
            flag = "DISCONTINUITY-BC"
        elif x_ab > DRIFT_X >= x_bc:
            flag = "PRE-SPIKE-DRIFT"
        rows.append({"block": name,
                     **{f"fro_{s}": sci(f) for s, f in zip(steps, fro)},
                     "x_ab": f"{x_ab:.4f}", "x_bc": f"{x_bc:.4f}",
                     **{f"absmax_{s}": sci(b[f"absmax_{s}"]) for s in steps},
                     "flag": flag})
    print(f"\n## Per-block ||W||_F and max|w| across steps {steps} "
          f"(flag threshold {DRIFT_X}x per interval)\n")
    print(fmt_table(rows, ["block"] + [f"fro_{s}" for s in steps]
                    + ["x_ab", "x_bc"] + [f"absmax_{s}" for s in steps]
                    + ["flag"]))

    # -- LN gain min/max per site ---------------------------------------------
    ln_rows = []
    for r in per_param:
        if not (r["block"].startswith("ln.") and r["block"].endswith(".gain")):
            continue
        row = {"param": r["param"]}
        for s, p in zip(steps, params):
            t = p[r["param"]]
            row[f"min_{s}"] = f"{t.min().item():.4f}"
            row[f"max_{s}"] = f"{t.max().item():.4f}"
        ln_rows.append(row)
    print(f"\n## LN gain min/max per site\n")
    print(fmt_table(ln_rows, ["param"] + [c for s in steps
                                          for c in (f"min_{s}", f"max_{s}")]))

    # -- embedding / tied-unembedding row norms --------------------------------
    print("\n## Token-embedding (tied unembedding) row norms\n")
    for s, p in zip(steps, params):
        rn = p["emb.weight"].double().norm(dim=1)
        top = rn.topk(10)
        stats = (f"step {s}: mean {rn.mean():.4f}, median {rn.median():.4f}, "
                 f"p99 {rn.quantile(0.99):.4f}, max {rn.max():.4f}")
        outliers = ", ".join(f"row {i}: {v:.3f}"
                             for v, i in zip(top.values.tolist(),
                                             top.indices.tolist()))
        print(f"- {stats}\n  top-10 rows: {outliers}")


if __name__ == "__main__":
    main()
