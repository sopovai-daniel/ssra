"""Plot the M2 Phase 2 sweep loss curves (train + val, all cells) and print
the per-model final_eval_loss table (selection input, M2-phase2-sweep §3).

Usage: .venv/bin/python scripts/plot_sweep.py [run-names...] \
           [--out-prefix results/M2-sweep-curves]
Defaults to the six stage-1 cells; pass all 8 once stage 2 exists. Writes one
PNG per model ({out-prefix}-{flat,ssra}.png). Reads logs/<run>.log (JSONL
from train.py). Loss values carry no quality meaning beyond the mechanical
selection rule (spec §16).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGE1 = [f"m2-sweep-{arch}-lr{lr}-do00"
          for arch in ("flat", "ssra") for lr in ("1e3", "6e4", "3e4")]


def read_log(run: str):
    train, val, final_eval, summary = [], [], None, None
    for line in (ROOT / "logs" / f"{run}.log").read_text().splitlines():
        rec = json.loads(line)
        if "train_loss" in rec:
            train.append((rec["step"], rec["train_loss"]))
        if "val_loss" in rec:
            val.append((rec["step"], rec["val_loss"]))
        if "final_eval" in rec:
            final_eval = rec["final_eval"]
        if "summary" in rec:
            summary = rec["summary"]
    return train, val, final_eval, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", default=STAGE1)
    ap.add_argument("--out-prefix", default="results/M2-sweep-curves")
    ap.add_argument("--title-prefix", default="M2 sweep",
                    help="plot title prefix (e.g. 'M2 core pair' for Phase 3)")
    args = ap.parse_args()
    runs = args.runs or STAGE1

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_model: dict[str, list[str]] = {}
    for run in runs:
        arch = "flat" if "-flat-" in run else "ssra"
        by_model.setdefault(arch, []).append(run)

    print(f"{'run':40s} {'status':10s} {'final_eval_loss':>15s}")
    for arch, model_runs in sorted(by_model.items()):
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = plt.cm.viridis([i / max(1, len(model_runs) - 1)
                                 for i in range(len(model_runs))])
        for run, color in zip(model_runs, colors):
            train, val, fe, _ = read_log(run)
            cell = run.replace(f"m2-sweep-{arch}-", "")
            if train:
                ax.plot(*zip(*train), color=color, alpha=0.35, lw=0.8)
            if val:
                ax.plot(*zip(*val), color=color, lw=1.8, marker=".",
                        ms=3, label=f"{cell} (val)")
            status = "DONE" if fe else "no final_eval"
            fel = fe["eval_loss"] if fe else float("nan")
            print(f"{run:40s} {status:10s} {fel:15.5f}")
        ax.set_xlabel("step")
        ax.set_ylabel("loss (nats/token)")
        ax.set_title(f"{args.title_prefix} — {arch} (train faint, val marked; "
                     f"metric = final_eval on val-eval-2M)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        out = ROOT / f"{args.out_prefix}-{arch}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
