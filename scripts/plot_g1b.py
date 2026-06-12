"""Plot the G1b-D3 loss curves (matched P1/P3 smoke pair, AP-3) plus any
additional smoke runs, and print the relative final-val-loss gap.

Usage: .venv/bin/python scripts/plot_g1b.py [run-names...] \
           [--out results/M1-g1b-curves.png]
Defaults to the M1 smoke set. Reads logs/<run>.log (JSONL from train.py).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS = ["M1-smoke-p1", "M1-smoke-p3", "M1-smoke-p2",
                "M1-smoke-flat", "M1-smoke-megabyte"]


def read_log(run: str):
    train, val, summary = [], [], None
    for line in (ROOT / "logs" / f"{run}.log").read_text().splitlines():
        rec = json.loads(line)
        if "train_loss" in rec:
            train.append((rec["step"], rec["train_loss"]))
        if "val_loss" in rec:
            val.append((rec["step"], rec["val_loss"]))
        if "summary" in rec:
            summary = rec["summary"]
    return train, val, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", default=DEFAULT_RUNS)
    ap.add_argument("--out", default="results/M1-g1b-curves.png")
    args = ap.parse_args()
    runs = args.runs or DEFAULT_RUNS

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    finals = {}
    for run in runs:
        try:
            train, val, summary = read_log(run)
        except FileNotFoundError:
            print(f"skip {run}: no log")
            continue
        label = run.replace("M1-smoke-", "")
        ax1.plot(*zip(*train), label=label, alpha=0.8)
        ax2.plot(*zip(*val), "o-", label=label, ms=3)
        if summary:
            finals[label] = summary["final_val_loss"]
    ax1.set(xlabel="step", ylabel="train CE loss [nats]", title="train loss")
    ax2.set(xlabel="step", ylabel="val CE loss [nats]", title="validation loss")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / args.out, dpi=150)
    print(f"wrote {args.out}")

    print("final val loss:", json.dumps(finals, indent=2))
    if "p1" in finals and "p3" in finals:
        gap = (finals["p3"] - finals["p1"]) / finals["p1"] * 100
        print(f"G1b-D3 relative gap (p3 vs p1): {gap:+.2f}% "
              f"(threshold X = 5%; gate decision: Daniel)")


if __name__ == "__main__":
    main()
