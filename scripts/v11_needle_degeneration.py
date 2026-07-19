"""V11 K3-c: needle degeneration characterization from the 720 raw G2-lite
trials (assignment docs/cc/V11-data-exploitation.md §3 K3-c).

Pinned inputs: results/g2lite/m2-g2lite-{flat,ssra}-m2.json (360 trials per
model: 6 N x 3 depths x 20 trials, per-trial verbatim `generated` output).
Passkey length is READ from scripts/needle_gen.py (KEY_RE literal), not
assumed.

Pre-registered mechanical categories per trial output (assignment verbatim;
digit strings = non-overlapping re.findall of the generator's own KEY_RE):
  cat1  exact gold passkey match — gold key is among the KEY_RE matches
  cat2  contains a digit string of the passkey length that is != gold
        (KEY_RE matches exist, none equals gold)
  cat3  contains no such digit string
No post-hoc categories; every one of the 720 trials is categorized.

Outputs (AP-21):
  results/v11/v11-needle-categorized.csv       full 720-row raw appendix
  results/v11/v11-needle-category-counts.csv   counts per model x N x depth
  results/v11/v11-needle-top10.json            verbatim top-10 most-frequent
                                               outputs per model (+ counts)
  results/figures/v11/v11-needle-categories.png stacked-bar figure

Usage: .venv/bin/python scripts/v11_needle_degeneration.py
Observations only; no architecture conclusions (spec §16).
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODELS = ("flat", "ssra")
CATS = ("cat1", "cat2", "cat3")
# Fixed category colors (identity job; colorblind-safe, fixed order).
CAT_COLORS = {"cat1": "#0072B2", "cat2": "#E69F00", "cat3": "#999999"}
CAT_LABELS = {"cat1": "cat1: exact gold passkey",
              "cat2": "cat2: other same-length digit string",
              "cat3": "cat3: no such digit string"}


def read_key_re() -> re.Pattern:
    """Read the passkey regex literal from scripts/needle_gen.py."""
    src = (ROOT / "scripts" / "needle_gen.py").read_text()
    m = re.search(r'KEY_RE\s*=\s*re\.compile\(r"([^"]+)"\)', src)
    if not m:
        raise SystemExit("KEY_RE literal not found in scripts/needle_gen.py")
    return re.compile(m.group(1))


def categorize(generated: str, key: str | int, key_re: re.Pattern) -> str:
    # `key` is stored as int in the m2 JSONs; findall returns strings.
    matches = key_re.findall(generated)
    if str(key) in matches:
        return "cat1"
    return "cat2" if matches else "cat3"


def main() -> None:
    key_re = read_key_re()
    print(f"passkey pattern from scripts/needle_gen.py: {key_re.pattern}")

    rows = []
    for model in MODELS:
        data = json.loads(
            (ROOT / "results" / "g2lite" / f"m2-g2lite-{model}-m2.json")
            .read_text())
        for cell in data["cells"]:
            for r in cell["results"]:
                rows.append({
                    "model": model, "n": cell["n"], "depth": cell["depth"],
                    "trial": r["trial"], "key": r["key"],
                    "generated": r["generated"],
                    "extracted": r.get("extracted"),
                    "correct": r["correct"],
                    "category": categorize(r["generated"], r["key"], key_re),
                })
    assert len(rows) == 720, f"expected 720 trials, got {len(rows)}"

    out_dir = ROOT / "results" / "v11"
    with open(out_dir / "v11-needle-categorized.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]),
                           quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)
    print(f"-> {out_dir / 'v11-needle-categorized.csv'} ({len(rows)} rows)")

    # Cross-check vs the stored first-match correctness metric: `correct`
    # implies cat1 (the converse need not hold — gold may not be first).
    viol = [r for r in rows if r["correct"] and r["category"] != "cat1"]
    cat1_not_correct = sum(1 for r in rows
                           if r["category"] == "cat1" and not r["correct"])
    print(f"cross-check: correct-but-not-cat1 = {len(viol)} (must be 0); "
          f"cat1-but-not-first-match = {cat1_not_correct} (observation)")

    # Counts per model x N x depth.
    ns = sorted({r["n"] for r in rows})
    depths = sorted({r["depth"] for r in rows})
    counts = {}
    for r in rows:
        cell = counts.setdefault((r["model"], r["n"], r["depth"]),
                                 dict.fromkeys(CATS, 0))
        cell[r["category"]] += 1
    with open(out_dir / "v11-needle-category-counts.csv", "w",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "n", "depth", *CATS, "trials"])
        for model in MODELS:
            for n in ns:
                for d in depths:
                    c = counts[(model, n, d)]
                    w.writerow([model, n, d, *(c[k] for k in CATS),
                                sum(c.values())])
    print(f"-> {out_dir / 'v11-needle-category-counts.csv'}")

    # Verbatim top-10 most-frequent outputs per model.
    top10 = {}
    for model in MODELS:
        freq = Counter(r["generated"] for r in rows if r["model"] == model)
        top10[model] = [{"count": c, "output": o}
                        for o, c in freq.most_common(10)]
    (out_dir / "v11-needle-top10.json").write_text(
        json.dumps({"note": "verbatim greedy outputs, most frequent first, "
                            "of 360 trials per model",
                    "top10": top10}, indent=2) + "\n")
    print(f"-> {out_dir / 'v11-needle-top10.json'}")

    # Figure: per-model panel, stacked bars (cat1/cat2/cat3) per N x depth.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    width = 0.27
    for ax, model in zip(axes, MODELS):
        for di, d in enumerate(depths):
            xs = [i + (di - 1) * width for i in range(len(ns))]
            bottom = [0] * len(ns)
            for cat in CATS:
                ys = [counts[(model, n, d)][cat] for n in ns]
                ax.bar(xs, ys, width * 0.93, bottom=bottom,
                       color=CAT_COLORS[cat],
                       label=CAT_LABELS[cat] if di == 0 else None)
                bottom = [b + y for b, y in zip(bottom, ys)]
        ax.set_xticks(range(len(ns)), [str(n) for n in ns], fontsize=8)
        ax.set_xlabel("context length N (3 bars per N = depth 0.1/0.5/0.9)")
        ax.set_title(f"{model} lr6e4")
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("trials (of 20 per cell)")
    axes[0].legend(fontsize=8, loc="center left")
    fig.suptitle("V11 K3-c: needle output categories per model x N x depth "
                 "(720 raw G2-lite trials; pre-registered categories)",
                 fontsize=10)
    fig.tight_layout()
    out = ROOT / "results" / "figures" / "v11" / "v11-needle-categories.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
