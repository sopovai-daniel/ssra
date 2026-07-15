"""T2 — weight-delta localization (docs/cc/M2-spike-diagnostics.md §3).

Per-module relative deltas ||dW||_F / ||W||_F for two checkpoint pairs:
A->B (pre-spike baseline, e.g. 5000->6000) and B->C (spike-crossing, e.g.
6000->7000), ranked by the ratio (spike-crossing) / (baseline). Read-only
(AP-20): loads checkpoints, writes one CSV under results/, prints markdown.

Usage:
  python scripts/diagnostics/t2_weight_delta.py CKPT_A CKPT_B CKPT_C \
      [--csv results/M2-spike-diag-T2-params.csv]
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from ckpt_common import (ROOT, block_of, canonical_params, fmt_table,
                         layer_of, load_blob, sci)


def rel_delta(a, b) -> tuple[float, float]:
    """(||b-a||_F, ||a||_F) in fp64 accumulation."""
    d = (b.double() - a.double()).norm().item()
    base = a.double().norm().item()
    return d, base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_a")
    ap.add_argument("ckpt_b")
    ap.add_argument("ckpt_c")
    ap.add_argument("--csv", default="results/M2-spike-diag-T2-params.csv")
    args = ap.parse_args()

    blobs = [load_blob(p) for p in (args.ckpt_a, args.ckpt_b, args.ckpt_c)]
    steps = [b["step"] for b in blobs]
    print(f"# T2 weight deltas: steps {steps[0]}->{steps[1]} (baseline) vs "
          f"{steps[1]}->{steps[2]} (spike-crossing), run {blobs[0]['run_name']}")
    pa, pb, pc = (canonical_params(b) for b in blobs)
    assert pa.keys() == pb.keys() == pc.keys()

    per_param = []
    for name in pa:
        d_ab, n_a = rel_delta(pa[name], pb[name])
        d_bc, n_b = rel_delta(pb[name], pc[name])
        rel_ab = d_ab / n_a if n_a else math.nan
        rel_bc = d_bc / n_b if n_b else math.nan
        per_param.append({
            "param": name, "block": block_of(name), "layer": layer_of(name),
            "numel": pa[name].numel(),
            "norm_a": n_a, "delta_ab": d_ab, "delta_bc": d_bc,
            "rel_ab": rel_ab, "rel_bc": rel_bc,
            "ratio": rel_bc / rel_ab if rel_ab else math.nan,
        })

    csv_path = ROOT / args.csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_param[0].keys()))
        w.writeheader()
        w.writerows(sorted(per_param, key=lambda r: -r["ratio"]
                           if math.isfinite(r["ratio"]) else -math.inf))
    print(f"full per-parameter table -> {csv_path}\n")

    def aggregate(key) -> list[dict]:
        agg = defaultdict(lambda: {"d_ab2": 0.0, "d_bc2": 0.0,
                                   "n_a2": 0.0, "n_b2": 0.0, "numel": 0})
        for r in per_param:
            k = key(r)
            if k is None:
                continue
            a = agg[k]
            a["d_ab2"] += r["delta_ab"] ** 2
            a["d_bc2"] += r["delta_bc"] ** 2
            a["n_a2"] += r["norm_a"] ** 2
            # ||W_b||^2 recoverable from stored fields only per-param; keep
            # blockwise base at ckpt B via delta identity is not exact -> use
            # the direct tensor norms captured below instead.
            a["numel"] += r["numel"]
        return agg, None

    # blockwise/basewise norms at B need the tensors; compute directly.
    base_b2 = defaultdict(float)
    base_b2_layer = defaultdict(float)
    for name, tens in pb.items():
        n2 = float(tens.double().norm().item() ** 2)
        base_b2[block_of(name)] += n2
        if layer_of(name) is not None:
            base_b2_layer[layer_of(name)] += n2

    def table(key, base_b2_map, title, col):
        agg, _ = aggregate(key)
        rows = []
        for k, a in agg.items():
            rel_ab = math.sqrt(a["d_ab2"]) / math.sqrt(a["n_a2"])
            rel_bc = math.sqrt(a["d_bc2"]) / math.sqrt(base_b2_map[k])
            rows.append({col: k, "numel": a["numel"],
                         "rel_ab": sci(rel_ab), "rel_bc": sci(rel_bc),
                         "ratio": f"{rel_bc / rel_ab:.3f}" if rel_ab else "inf",
                         "_r": rel_bc / rel_ab if rel_ab else math.inf})
        rows.sort(key=lambda r: -r["_r"])
        for r in rows:
            r.pop("_r")
        print(f"\n## {title} (rel_ab = {steps[0]}->{steps[1]}, "
              f"rel_bc = {steps[1]}->{steps[2]}, ranked by ratio)\n")
        print(fmt_table(rows, [col, "numel", "rel_ab", "rel_bc", "ratio"]))

    table(lambda r: r["block"], base_b2, "Per logical block", "block")
    table(lambda r: r["layer"], base_b2_layer, "Per layer (all params of layer)",
          "layer")


if __name__ == "__main__":
    main()
