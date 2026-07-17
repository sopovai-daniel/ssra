"""Plot the M2 G2-lite measurement results (assignment §6.2): ppl vs N
(log-x), needle accuracy heatmaps, and per-position bucket curves.

Usage: .venv/bin/python scripts/plot_g2lite.py \
           [--in-dir results/g2lite] [--out-prefix results/M2-g2lite]
Reads the committed per-mode JSONs written by scripts/g2lite_eval.py.
Numbers carry no architecture meaning (spec §16); the O1-O7 labels are
applied in results/M2-g2lite.md, not here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ARCHES = ("flat", "ssra")
DEPTHS = (0.1, 0.5, 0.9)


def load(in_dir: Path, arch: str, mode: str) -> dict:
    return json.loads((in_dir / f"m2-g2lite-{arch}-{mode}.json").read_text())


def plot_ppl(in_dir: Path, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for arch, marker in zip(ARCHES, "os"):
        cells = load(in_dir, arch, "m1")["cells"]
        ns = [c["n"] for c in cells]
        ppl = [c["ppl"] for c in cells]
        ax.plot(ns, ppl, marker=marker, label=f"{arch} lr6e4")
        for c in cells:
            ax.annotate(f"{c['ppl']:.0f}", (c["n"], c["ppl"]),
                        textcoords="offset points", xytext=(0, 6),
                        fontsize=7, ha="center")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("context length N (tokens, log2)")
    ax.set_ylabel("ppl on region E (log)")
    ax.axvline(1024, color="gray", ls=":", lw=1)
    ax.text(1024, ax.get_ylim()[0] * 1.2, " train seq_len", fontsize=7,
            color="gray")
    ax.set_title("M2 G2-lite M1: ppl vs context length (region E)\n"
                 "packed short docs — degradation robustness, "
                 "not long-range benefit", fontsize=10)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"-> {out}")


def plot_needle(in_dir: Path, out_prefix: Path) -> None:
    for arch in ARCHES:
        cells = load(in_dir, arch, "m2")["cells"]
        ns = sorted({c["n"] for c in cells})
        acc = [[next(c["accuracy"] for c in cells
                     if c["n"] == n and c["depth"] == d) for n in ns]
               for d in DEPTHS]
        fig, ax = plt.subplots(figsize=(7, 3))
        im = ax.imshow(acc, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(ns)), [str(n) for n in ns])
        ax.set_yticks(range(len(DEPTHS)), [str(d) for d in DEPTHS])
        ax.set_xlabel("context length N")
        ax.set_ylabel("needle depth")
        ax.set_title(f"M2 G2-lite needle-lite accuracy — {arch} lr6e4 "
                     "(20 trials/cell, greedy)")
        for i in range(len(DEPTHS)):
            for j in range(len(ns)):
                ax.text(j, i, f"{acc[i][j]:.2f}", ha="center", va="center",
                        color="white" if acc[i][j] < 0.5 else "black",
                        fontsize=8)
        fig.colorbar(im, label="exact-match accuracy")
        fig.tight_layout()
        out = Path(f"{out_prefix}-needle-{arch}.png")
        fig.savefig(out, dpi=150)
        print(f"-> {out}")


def plot_buckets(in_dir: Path, out_prefix: Path) -> None:
    for arch in ARCHES:
        cells = load(in_dir, arch, "m1")["cells"]
        fig, ax = plt.subplots(figsize=(7, 5))
        for c in cells:
            edges = [int(k.split("-")[1]) for k in c["bucket_mean_nll"]]
            ax.plot(edges, list(c["bucket_mean_nll"].values()),
                    marker=".", label=f"N={c['n']}")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("target position bucket upper edge (log2)")
        ax.set_ylabel("bucket mean NLL (nats)")
        ax.set_title(f"M2 G2-lite M1 per-position NLL — {arch} lr6e4")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out = Path(f"{out_prefix}-buckets-{arch}.png")
        fig.savefig(out, dpi=150)
        print(f"-> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="results/g2lite")
    ap.add_argument("--out-prefix", default="results/M2-g2lite")
    args = ap.parse_args()
    in_dir = ROOT / args.in_dir
    prefix = ROOT / args.out_prefix
    plot_ppl(in_dir, Path(f"{prefix}-ppl-vs-n.png"))
    plot_needle(in_dir, prefix)
    plot_buckets(in_dir, prefix)


if __name__ == "__main__":
    main()
