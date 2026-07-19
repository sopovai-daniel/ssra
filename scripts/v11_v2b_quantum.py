"""V11 K3-b: bf16 position-quantum figure (assignment
docs/cc/V11-data-exploitation.md §3 K3-b).

Renders the empirical bf16 integer-position quantum table from
`results/M2-g2lite.md` (V2b erratum, governing) as a step plot, and
re-evaluates the quantum numerically in this standalone script (CPU
`torch.bfloat16` round-trip — the same dtype semantics as the eval path's
explicit `pos.to(bf16)` cast; no model code touched). The committed table
is pinned below as the expectation; any numeric disagreement is printed
and recorded, never silently patched.

Outputs (AP-21):
  results/figures/v11/v11-v2b-quantum.png   quantum vs position (log2/log2)
                                            + ULP >= w = 64 annotation
  results/v11/v11-v2b-quantum.json          per-binade numeric re-evaluation
                                            vs the committed table

Usage: .venv/bin/python scripts/v11_v2b_quantum.py
Observations only; no architecture conclusions (spec §16).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
W = 64          # read-out window (spec v1.2); annotation target
N_MAX = 32768   # trained n_max (e_l table headroom)

# Committed expectation — results/M2-g2lite.md V2b table (empirical, governing):
# binade_lo: (quantum, max rounding err, distinct bf16 positions in the last
# 65-token window at N = 2*binade_lo; None where the report has "—").
COMMITTED = {
    128:   (1,   0,  65),
    256:   (2,   1,  None),
    512:   (4,   2,  17),
    1024:  (8,   4,  9),
    2048:  (16,  8,  5),
    4096:  (32,  16, 3),
    8192:  (64,  32, 2),
    16384: (128, 64, 1),
}


def bf16_roundtrip(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.bfloat16).to(torch.float64)


def main() -> None:
    pos = torch.arange(1, N_MAX + 1, dtype=torch.float64)
    q = bf16_roundtrip(pos)

    rows, all_match = [], True
    for lo in COMMITTED:
        hi = lo * 2
        sel = (pos >= lo) & (pos < hi)
        vals = torch.unique(q[sel])
        diffs = torch.unique(vals[1:] - vals[:-1])
        quantum = int(diffs[0].item()) if len(diffs) == 1 else None
        max_err = float((pos[sel] - q[sel]).abs().max().item())
        n = hi
        # 1-indexed last read-out window of query t = N: positions [N-64, N].
        win = torch.arange(n - W, n + 1, dtype=torch.float64)
        distinct = len(torch.unique(bf16_roundtrip(win)))
        exp_q, exp_err, exp_d = COMMITTED[lo]
        match = (quantum == exp_q and max_err == exp_err
                 and (exp_d is None or distinct == exp_d))
        all_match &= match
        rows.append({"binade_lo": lo, "binade_hi": hi,
                     "quantum_empirical": quantum,
                     "max_rounding_err": max_err,
                     "distinct_positions_last_window": distinct,
                     "committed": {"quantum": exp_q, "max_err": exp_err,
                                   "distinct_last_window": exp_d},
                     "match_committed": match})
        print(f"[{lo}, {hi}): quantum={quantum} err={max_err:.0f} "
              f"distinct@N={n}={distinct} match={match}")
    print("ALL BINADES MATCH COMMITTED TABLE" if all_match
          else "MISMATCH vs committed table — see JSON")

    out_json = ROOT / "results" / "v11" / "v11-v2b-quantum.json"
    out_json.write_text(json.dumps(
        {"source_table": "results/M2-g2lite.md V2b erratum (empirical, "
                         "governing)",
         "method": "CPU torch.bfloat16 round-trip of integer positions "
                   "1..32768 (standalone; no model code)",
         "torch": torch.__version__,
         "all_match": all_match, "binades": rows}, indent=2) + "\n")
    print(f"-> {out_json}")

    # Step plot: quantum vs position. Below 256 integers are bf16-exact.
    xs, ys = [1], [1]
    for lo in COMMITTED:
        xs += [lo, lo * 2]
        ys += [COMMITTED[lo][0]] * 2
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.step(xs, ys, where="post", lw=1.8, color="#0072B2")
    ax.axhline(W, color="gray", ls=":", lw=1)
    ax.axvline(8192, color="crimson", ls="--", lw=1.2)
    ax.annotate("ULP $\\geq$ w = 64 from t = 8192 (N $\\approx$ 8k):\n"
                "read-out window positions collapse",
                xy=(8192, W), xytext=(8, 24), fontsize=8, color="crimson",
                arrowprops=dict(arrowstyle="->", color="crimson", lw=1))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("integer position t (log2)")
    ax.set_ylabel("bf16 position quantum $2^{\\lfloor\\log_2 t\\rfloor-7}$ "
                  "(log2)")
    ax.set_title("V11 K3-b: bf16 position quantum vs position "
                 "(V2b, both models' shared eval path)\n"
                 "quantum re-evaluated numerically; table = "
                 "results/M2-g2lite.md", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = ROOT / "results" / "figures" / "v11" / "v11-v2b-quantum.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
