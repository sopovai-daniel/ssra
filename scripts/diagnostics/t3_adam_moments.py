"""T3 — Adam-moment localization (docs/cc/M2-spike-diagnostics.md §3).

At two checkpoints (e.g. 6000 and 7000): per-module mean and max of
`exp_avg_sq` plus ||exp_avg||_F, and the B->C growth ratios — a substitute
for the unlogged gradient norms (report §xi C3). Read-only (AP-20).

Usage:
  python scripts/diagnostics/t3_adam_moments.py CKPT_B CKPT_C \
      [--csv results/M2-spike-diag-T3-params.csv]
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from ckpt_common import (ROOT, block_of, fmt_table, layer_of, load_blob,
                         optimizer_param_names, sci)


def moment_stats(blob: dict) -> dict[str, dict]:
    """{canonical param name -> {m2_mean, m2_max, m1_norm2, numel}}."""
    names = optimizer_param_names(blob)
    state = blob["optimizer"]["state"]
    out = {}
    for i, name in enumerate(names):
        st = state.get(i)
        if st is None:  # param never updated (should not happen mid-run)
            continue
        m2 = st["exp_avg_sq"].double()
        m1 = st["exp_avg"].double()
        out[name] = {"m2_mean": m2.mean().item(), "m2_max": m2.max().item(),
                     "m1_norm2": float(m1.pow(2).sum().item()),
                     "numel": m2.numel()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_b")
    ap.add_argument("ckpt_c")
    ap.add_argument("--csv", default="results/M2-spike-diag-T3-params.csv")
    args = ap.parse_args()

    blob_b, blob_c = load_blob(args.ckpt_b), load_blob(args.ckpt_c)
    sb, sc = blob_b["step"], blob_c["step"]
    print(f"# T3 Adam moments: step {sb} vs {sc}, run {blob_b['run_name']}")
    mb, mc = moment_stats(blob_b), moment_stats(blob_c)
    assert mb.keys() == mc.keys()

    def ratio(a: float, b: float) -> float:
        return b / a if a else math.nan

    per_param = []
    for name in mb:
        b, c = mb[name], mc[name]
        per_param.append({
            "param": name, "block": block_of(name), "layer": layer_of(name),
            "numel": b["numel"],
            "m2_mean_b": b["m2_mean"], "m2_mean_c": c["m2_mean"],
            "m2_max_b": b["m2_max"], "m2_max_c": c["m2_max"],
            "m1_norm_b": math.sqrt(b["m1_norm2"]),
            "m1_norm_c": math.sqrt(c["m1_norm2"]),
            "m2_mean_ratio": ratio(b["m2_mean"], c["m2_mean"]),
            "m2_max_ratio": ratio(b["m2_max"], c["m2_max"]),
            "m1_norm_ratio": ratio(math.sqrt(b["m1_norm2"]),
                                   math.sqrt(c["m1_norm2"])),
        })

    csv_path = ROOT / args.csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_param[0].keys()))
        w.writeheader()
        w.writerows(sorted(per_param, key=lambda r: -r["m2_mean_ratio"]))
    print(f"full per-parameter table -> {csv_path}\n")

    def table(key, title, col):
        agg = defaultdict(lambda: {"m2_sum_b": 0.0, "m2_sum_c": 0.0,
                                   "m2_max_b": 0.0, "m2_max_c": 0.0,
                                   "m1_b2": 0.0, "m1_c2": 0.0, "numel": 0})
        for r in per_param:
            k = key(r)
            if k is None:
                continue
            a = agg[k]
            a["m2_sum_b"] += r["m2_mean_b"] * r["numel"]
            a["m2_sum_c"] += r["m2_mean_c"] * r["numel"]
            a["m2_max_b"] = max(a["m2_max_b"], r["m2_max_b"])
            a["m2_max_c"] = max(a["m2_max_c"], r["m2_max_c"])
            a["m1_b2"] += r["m1_norm_b"] ** 2
            a["m1_c2"] += r["m1_norm_c"] ** 2
            a["numel"] += r["numel"]
        rows = []
        for k, a in agg.items():
            mean_b = a["m2_sum_b"] / a["numel"]
            mean_c = a["m2_sum_c"] / a["numel"]
            rows.append({
                col: k,
                "m2_mean_b": sci(mean_b), "m2_mean_c": sci(mean_c),
                "m2_mean_x": f"{ratio(mean_b, mean_c):.3f}",
                "m2_max_b": sci(a["m2_max_b"]), "m2_max_c": sci(a["m2_max_c"]),
                "m2_max_x": f"{ratio(a['m2_max_b'], a['m2_max_c']):.3f}",
                "m1_norm_b": sci(math.sqrt(a["m1_b2"])),
                "m1_norm_c": sci(math.sqrt(a["m1_c2"])),
                "m1_norm_x": f"{ratio(math.sqrt(a['m1_b2']), math.sqrt(a['m1_c2'])):.3f}",
                "_r": ratio(mean_b, mean_c),
            })
        rows.sort(key=lambda r: -r["_r"])
        for r in rows:
            r.pop("_r")
        print(f"\n## {title} (b = step {sb}, c = step {sc}, "
              f"ranked by m2_mean growth)\n")
        print(fmt_table(rows, [col, "m2_mean_b", "m2_mean_c", "m2_mean_x",
                               "m2_max_b", "m2_max_c", "m2_max_x",
                               "m1_norm_b", "m1_norm_c", "m1_norm_x"]))

    table(lambda r: r["block"], "Per logical block", "block")
    table(lambda r: r["layer"], "Per layer", "layer")


if __name__ == "__main__":
    main()
