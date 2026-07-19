"""V11 K3-a: P-C pooling-attention entropy across ALL SSRA runs (assignment
docs/cc/V11-data-exploitation.md §3 K3-a).

Pinned inputs: every `logs/**/*.log` file in the repo whose JSONL records
contain `p1_attn_entropy` (the P1 pooling head logs it; flat runs have no
such field). Every qualifying FILE is one series — files that are session
variants of the same run (resume / re-measurement) are plotted as-is and
enumerated; no selection or merging.

Outputs (AP-21):
  results/figures/v11/v11-pc-entropy.png        entropy vs step overlay
                                                + ln(32) reference line
  results/figures/v11/v11-pc-participation.png  companion: p1 participation
                                                [min, max] band vs step
  results/v11/v11-pc-entropy-summary.json       per-file enumeration + stats

Usage: .venv/bin/python scripts/v11_pc_entropy.py
X axis is symlog (linear below step 10) so 120-step smokes and the 51,880-step
core run share one frame without dropping step-0 records.
Observations only; no architecture conclusions (spec §16).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LN32 = math.log(32.0)


def load_series(path: Path) -> dict | None:
    steps, ent, pmin, pmax = [], [], [], []
    run = ""
    with open(path, errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            if "meta" in rec and not run:
                run = rec["meta"].get("run", "")
            if "p1_attn_entropy" in rec:
                steps.append(rec["step"])
                ent.append(rec["p1_attn_entropy"])
                pmin.append(rec.get("p1_participation_min"))
                pmax.append(rec.get("p1_participation_max"))
    if not ent:
        return None
    return {"file": str(path.relative_to(ROOT)), "meta_run": run,
            "steps": steps, "entropy": ent,
            "participation_min": pmin, "participation_max": pmax}


def main() -> None:
    series = []
    for path in sorted(ROOT.glob("logs/**/*.log")):
        s = load_series(path)
        if s is not None:
            series.append(s)

    fig_dir = ROOT / "results" / "figures" / "v11"
    fig_dir.mkdir(parents=True, exist_ok=True)
    colors = [cm.viridis(x) for x in
              [i / max(len(series) - 1, 1) for i in range(len(series))]]

    # Overlay: entropy vs step, ln(32) reference.
    fig, ax = plt.subplots(figsize=(9, 6))
    for s, c in zip(series, colors):
        label = Path(s["file"]).stem
        ax.plot(s["steps"], s["entropy"], color=c, lw=1.2, label=label)
    ax.axhline(LN32, color="crimson", ls="--", lw=1.2)
    ax.text(1.5, LN32 + 0.004, "ln(32) = uniform over 2m = 32 slots",
            color="crimson", fontsize=8, va="bottom")
    ax.set_xscale("symlog", linthresh=10, base=10)
    ax.set_xlim(left=0)
    ax.set_xlabel("training step (symlog)")
    ax.set_ylabel("p1_attn_entropy (nats)")
    ax.set_title("V11 K3-a: P1 pooling-attention entropy vs step — all SSRA "
                 "runs with JSONL logs\n(one series per log file; observations "
                 "only, spec §16)", fontsize=10)
    ax.legend(fontsize=6, ncol=2, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = fig_dir / "v11-pc-entropy.png"
    fig.savefig(out, dpi=150)
    print(f"-> {out}")

    # Companion: participation [min, max] band vs step.
    fig, ax = plt.subplots(figsize=(9, 6))
    for s, c in zip(series, colors):
        if any(v is None for v in s["participation_min"]):
            continue
        label = Path(s["file"]).stem
        ax.fill_between(s["steps"], s["participation_min"],
                        s["participation_max"], color=c, alpha=0.25, lw=0)
        ax.plot(s["steps"], s["participation_max"], color=c, lw=1.0,
                label=label)
        ax.plot(s["steps"], s["participation_min"], color=c, lw=1.0, ls=":")
    ax.set_xscale("symlog", linthresh=10, base=10)
    ax.set_xlim(left=0)
    ax.set_xlabel("training step (symlog)")
    ax.set_ylabel("p1_participation_min/max as logged "
                  "(solid = max, dotted = min, band = range)")
    ax.set_title("V11 K3-a companion: P1 participation range vs step",
                 fontsize=10)
    ax.legend(fontsize=6, ncol=2, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = fig_dir / "v11-pc-participation.png"
    fig.savefig(out, dpi=150)
    print(f"-> {out}")

    # Enumeration + summary stats for the report.
    summary = []
    for s in series:
        summary.append({
            "file": s["file"],
            "meta_run": s["meta_run"],
            "n_entropy_records": len(s["entropy"]),
            "step_min": min(s["steps"]),
            "step_max": max(s["steps"]),
            "entropy_first": s["entropy"][0],
            "entropy_last": s["entropy"][-1],
            "entropy_min": min(s["entropy"]),
            "entropy_max": max(s["entropy"]),
            "abs_dev_from_ln32_max": max(abs(e - LN32) for e in s["entropy"]),
            "participation_last_min": s["participation_min"][-1],
            "participation_last_max": s["participation_max"][-1],
        })
    out = ROOT / "results" / "v11" / "v11-pc-entropy-summary.json"
    out.write_text(json.dumps({"ln32": LN32, "series": summary}, indent=2)
                   + "\n")
    print(f"-> {out} ({len(summary)} series)")


if __name__ == "__main__":
    main()
