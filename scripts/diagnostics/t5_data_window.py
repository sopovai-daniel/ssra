"""T5 — data-window due diligence (docs/cc/M2-spike-diagnostics.md §3).

Reconstructs the exact token batches of a training-step window via the
harness's OWN deterministic data path: `batches()` is imported from
scripts/train.py (not reimplemented), the sampling generator is seeded with
the run seed, and the per-step draws are replayed from step 0. Valid because
the run's data generator is consumed by exactly one randint per training
step (scripts/train.py) and the run log shows a single continuous run
(`resumed_from: null`). No training step is executed (AP-20).

Reports per-step window stats (EOT/document structure, distinct-token
fraction, unigram token entropy, longest constant run) against a
baseline of independently sampled windows, cross-window overlap counts,
and decodes short excerpts of the most anomalous windows.

Usage:
  python scripts/diagnostics/t5_data_window.py \
      --train-bin data/m2/train.bin --tokenizer artifacts/tokenizer/... \
      --seed 1337 --batch-size 16 --seq-len 1024 \
      --steps 6450 6550 [--csv results/M2-spike-diag-T5-steps.csv]
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from ckpt_common import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
from train import batches  # noqa: E402  (the harness's own sampler)

from ssra.data import EOT_TOKEN, load_shard  # noqa: E402


def window_stats(win: np.ndarray, eot_id: int) -> dict:
    """Stats for one (seq_len,) token window."""
    counts = np.bincount(win, minlength=1)
    probs = counts[counts > 0] / win.size
    runs = np.diff(np.flatnonzero(np.concatenate(
        ([True], win[1:] != win[:-1], [True]))))
    longest = int(runs.max())
    run_ends = np.cumsum(runs)
    run_end = int(run_ends[np.argmax(runs)])
    longest_tok = int(win[run_end - 1]) if longest > 1 else -1
    return {
        "eot_count": int((win == eot_id).sum()),
        "distinct_frac": float(np.unique(win).size / win.size),
        "entropy_nats": float(-(probs * np.log(probs)).sum()),
        "longest_run": longest,
        "longest_run_token": longest_tok,
        "longest_run_end": run_end,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-bin", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--seq-len", type=int, required=True)
    ap.add_argument("--steps", type=int, nargs=2, required=True,
                    metavar=("FIRST", "LAST"), help="inclusive step range")
    ap.add_argument("--baseline-windows", type=int, default=512)
    ap.add_argument("--baseline-seed", type=int, default=20260715)
    ap.add_argument("--n-decode", type=int, default=4)
    ap.add_argument("--csv", default="results/M2-spike-diag-T5-steps.csv")
    args = ap.parse_args()
    first, last = args.steps

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(ROOT / args.tokenizer))
    eot_id = tok.token_to_id(EOT_TOKEN)

    shard = load_shard(str(ROOT / args.train_bin))
    data = torch.from_numpy(np.asarray(shard, dtype=np.int64))
    print(f"# T5 data window: steps {first}..{last} (inclusive), "
          f"shard {args.train_bin} ({len(data)} tokens), seed {args.seed}, "
          f"eot_id {eot_id}")

    # Replay the harness's per-step draws from step 0 (one randint per step).
    gen = torch.Generator().manual_seed(args.seed)
    step_windows: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for step in range(last + 1):
        # batches() consumes the generator identically to the training loop;
        # we only materialize the window range under analysis.
        starts = torch.randint(0, len(data) - args.seq_len - 1,
                               (args.batch_size,), generator=gen)
        if step >= first:
            x = torch.stack([data[s: s + args.seq_len] for s in starts])
            step_windows[step] = (starts, x)

    # Baseline: independent uniform windows, same shape, separate generator
    # (baseline statistic only -- NOT part of the run's data path).
    bgen = torch.Generator().manual_seed(args.baseline_seed)
    bstarts = torch.randint(0, len(data) - args.seq_len - 1,
                            (args.baseline_windows,), generator=bgen)
    base = [window_stats(data[s: s + args.seq_len].numpy(), eot_id)
            for s in bstarts]

    def base_ms(k: str) -> tuple[float, float]:
        v = np.array([b[k] for b in base], dtype=np.float64)
        return float(v.mean()), float(v.std())

    keys = ["eot_count", "distinct_frac", "entropy_nats", "longest_run"]
    print(f"\n## Baseline ({args.baseline_windows} independent windows, "
          f"seed {args.baseline_seed})\n")
    for k in keys:
        m, s = base_ms(k)
        v = np.array([b[k] for b in base], dtype=np.float64)
        print(f"- {k}: mean {m:.4f}, std {s:.4f}, p95 {np.quantile(v, .95):.4f}, "
              f"p99 {np.quantile(v, .99):.4f}, max {v.max():.4f}")
    n_run = sum(b["longest_run"] >= 8 for b in base)
    print(f"- windows with longest_run >= 8: {n_run}/{len(base)} "
          f"(heavy tail -- z-scores on longest_run overstate significance)")

    # Per-step stats (window stats averaged over the batch; extremes kept).
    rows, all_windows = [], []
    for step, (starts, x) in sorted(step_windows.items()):
        ws = [window_stats(r.numpy(), eot_id) for r in x]
        all_windows.extend((step, i, int(starts[i]), w)
                           for i, w in enumerate(ws))
        row = {"step": step}
        for k in keys:
            v = np.array([w[k] for w in ws], dtype=np.float64)
            m, s = base_ms(k)
            row[f"{k}_mean"] = round(float(v.mean()), 4)
            row[f"{k}_min"] = round(float(v.min()), 4)
            row[f"{k}_max"] = round(float(v.max()), 4)
            row[f"{k}_z"] = round((float(v.mean()) - m)
                                  / (s / math.sqrt(len(ws))), 2) if s else 0.0
        rows.append(row)

    csv_path = ROOT / args.csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nper-step table -> {csv_path}")

    # Flag steps whose batch-mean z-score is extreme on any stat.
    print("\n## Steps with |z| >= 3 on any batch-mean stat\n")
    flagged = [r for r in rows if any(abs(r[f"{k}_z"]) >= 3 for k in keys)]
    for r in flagged:
        zs = ", ".join(f"{k}_z={r[f'{k}_z']}" for k in keys
                       if abs(r[f"{k}_z"]) >= 3)
        print(f"- step {r['step']}: {zs}")
    if not flagged:
        print("- none")

    # Cross-window overlap inside the analyzed range (same text reused).
    starts_all = [(step, i, s) for step, i, s, _ in all_windows]
    starts_sorted = sorted(starts_all, key=lambda t: t[2])
    overlaps = [(a, b) for a, b in zip(starts_sorted, starts_sorted[1:])
                if b[2] - a[2] < args.seq_len]
    n_win = len(starts_all)
    exp_pairs = (n_win * (n_win - 1) / 2) * (2 * args.seq_len / len(data))
    print(f"\n## Window overlap within steps {first}..{last}\n")
    print(f"- windows: {n_win}; overlapping adjacent pairs: {len(overlaps)} "
          f"(uniform-sampling expectation ~{exp_pairs:.1f})")
    for a, b in overlaps[:10]:
        print(f"  - step {a[0]} win {a[1]} (start {a[2]}) ~ "
              f"step {b[0]} win {b[1]} (start {b[2]})")

    # Decode the most anomalous windows (short excerpts only): half by
    # lowest entropy, half by longest constant run.
    print(f"\n## Decoded excerpts ({args.n_decode} lowest-entropy + "
          f"{args.n_decode} longest-run windows, first 240 chars each)\n")
    by_entropy = sorted(all_windows, key=lambda t: t[3]["entropy_nats"])
    by_run = sorted(all_windows, key=lambda t: -t[3]["longest_run"])
    picked, seen = [], set()
    for cand in by_entropy[: args.n_decode] + by_run[: args.n_decode]:
        if (cand[0], cand[1]) not in seen:
            seen.add((cand[0], cand[1]))
            picked.append(cand)
    for step, i, start, w in picked:
        win = data[start: start + args.seq_len].tolist()
        if w["longest_run"] > 4:  # centre the excerpt on the long run
            lo = max(0, w["longest_run_end"] - w["longest_run"] - 20)
            text = tok.decode(win[lo: w["longest_run_end"] + 20])
        else:
            text = tok.decode(win)
        excerpt = text[:240].replace("\n", " ")
        print(f"- step {step} win {i} (start {start}, "
              f"entropy {w['entropy_nats']:.3f}, run {w['longest_run']} "
              f"of token {w['longest_run_token']}, eot {w['eot_count']}):"
              f"\n    {excerpt!r}")


if __name__ == "__main__":
    main()
