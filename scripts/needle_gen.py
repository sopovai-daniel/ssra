"""Passkey-retrieval suite generator for M2 G2-lite (docs/cc/M2-g2lite.md
§2 V5, §3 M2).

Canonical passkey format per Mohtashami & Jaggi, Landmark Attention
(arXiv:2305.16300, retrieved 2026-07-17 — the task's defining source):
a short preamble, repeated filler sentences, ONE needle sentence
"The pass key is <5-digit>. Remember it. <5-digit> is the pass key."
placed at a controlled depth, and a final question so the prompt ends
"... What is the pass key? The pass key is". Multi-needle (k > 1) is M3
scope and deliberately NOT implemented here (assignment §2 V5).

Prompts are defined AS token-id sequences (the eval feeds ids, never
re-tokenized text): segments are tokenized independently with the frozen
tokenizer (sha256 gate) and assembled by token count, so the target
sequence length is met EXACTLY (prompt token budget = N - 16, §3 M2).
Filler is cycled and cut at token granularity; the needle and question
segments are never cut.

Determinism: the 5-digit key of trial (N, depth, trial) is drawn from
`random.Random(f"{seed}:{N}:{depth_str}:{trial}")` — platform-independent
(CPython seeds str via sha512) and independent of generation order.

Outputs (committed BEFORE any pod deploy — pre-registration):
  <out>/suite.jsonl.gz   one JSON record per trial (token ids inline);
                         gzip mtime=0 -> byte-deterministic archive
  <out>/manifest.json    suite params + per-trial sha256 of the uint16
                         token bytes + keys + suite-level sha256

Usage:
  .venv/bin/python scripts/needle_gen.py \
      --tokenizer artifacts/tokenizer/fineweb-edu-bpe-16384.json \
      --tokenizer-sha256 019568a2... \
      --seed 20260717 --grid 1024 2048 4096 8192 16384 32768 \
      --depths 0.1 0.5 0.9 --trials 20 --headroom 16 \
      --out artifacts/needles/m2-g2lite
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np

# Canonical strings (Mohtashami & Jaggi, arXiv:2305.16300, Appendix prompt).
PREAMBLE = ("There is an important info hidden inside a lot of irrelevant "
            "text. Find it and memorize them. I will quiz you about the "
            "important information there.")
FILLER = (" The grass is green. The sky is blue. The sun is yellow. "
          "Here we go. There and back again.")
NEEDLE_TEMPLATE = (" The pass key is {key}. Remember it. {key} is the "
                   "pass key.")
QUESTION = " What is the pass key? The pass key is"

KEY_RE = re.compile(r"\d{5}")  # metric: first 5-digit match (assignment §3 M2)


def _load_tokenizer(path: Path, expected_sha256: str | None):
    """Frozen-tokenizer gate: sha256 must match before any suite is built."""
    blob = path.read_bytes()
    actual = hashlib.sha256(blob).hexdigest()
    if expected_sha256 and actual != expected_sha256:
        raise SystemExit(f"[gate] tokenizer sha256 mismatch: {path}\n"
                         f"  actual   {actual}\n  expected {expected_sha256}")
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(path)), actual


def draw_key(seed: int, n: int, depth: float, trial: int) -> int:
    """Deterministic per-trial 5-digit key, independent of build order."""
    rng = random.Random(f"{seed}:{n}:{depth:.4f}:{trial}")
    return rng.randint(10000, 99999)


def build_prompt(tok, n: int, depth: float, key: int, headroom: int,
                 preamble: bool = True) -> dict:
    """Assemble one prompt of EXACTLY n - headroom tokens.

    Layout: [preamble][filler_a][needle][filler_b][question]; the needle
    start index is round(depth * budget) clamped to the feasible range.
    """
    budget = n - headroom
    pre_ids = tok.encode(PREAMBLE).ids if preamble else []
    needle_ids = tok.encode(NEEDLE_TEMPLATE.format(key=key)).ids
    question_ids = tok.encode(QUESTION).ids
    filler_ids = tok.encode(FILLER).ids

    n_filler = budget - len(pre_ids) - len(needle_ids) - len(question_ids)
    if n_filler < 0:
        raise ValueError(f"budget {budget} too small for fixed segments "
                         f"({len(pre_ids)}+{len(needle_ids)}+"
                         f"{len(question_ids)} tokens)")

    def cycled(k: int, offset: int = 0) -> list[int]:
        return [filler_ids[(offset + i) % len(filler_ids)] for i in range(k)]

    target = round(depth * budget)
    lo, hi = len(pre_ids), len(pre_ids) + n_filler
    needle_start = min(max(target, lo), hi)
    k_a = needle_start - len(pre_ids)
    ids = (pre_ids + cycled(k_a) + needle_ids
           + cycled(n_filler - k_a, offset=k_a) + question_ids)
    assert len(ids) == budget, (len(ids), budget)
    return {"ids": ids, "needle_start": needle_start,
            "needle_len": len(needle_ids)}


def generate_suite(tok, seed: int, grid: list[int], depths: list[float],
                   trials: int, headroom: int, preamble: bool = True):
    """Yield trial records in a fixed (N, depth, trial) order."""
    for n in grid:
        for depth in depths:
            for trial in range(trials):
                key = draw_key(seed, n, depth, trial)
                p = build_prompt(tok, n, depth, key, headroom, preamble)
                # round-trip gate: the key must be recoverable from the
                # decoded prompt by the SAME first-5-digit rule the metric
                # uses on generated text (frozen tokenizer, V5)
                decoded = tok.decode(p["ids"])
                m = KEY_RE.search(decoded)
                if m is None or m.group(0) != str(key):
                    raise SystemExit(f"[gate] key round-trip failed for "
                                     f"(N={n}, depth={depth}, trial={trial}):"
                                     f" got {m.group(0) if m else None},"
                                     f" want {key}")
                yield {"n": n, "depth": depth, "trial": trial, "key": key,
                       "prompt_len": len(p["ids"]),
                       "needle_start": p["needle_start"],
                       "needle_len": p["needle_len"], "ids": p["ids"]}


def trial_sha256(ids: list[int]) -> str:
    return hashlib.sha256(np.asarray(ids, dtype=np.uint16).tobytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--tokenizer-sha256", default=None,
                    help="frozen tokenizer gate (assignment V5)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--grid", type=int, nargs="+", required=True)
    ap.add_argument("--depths", type=float, nargs="+", required=True)
    ap.add_argument("--trials", type=int, required=True)
    ap.add_argument("--headroom", type=int, default=16,
                    help="prompt budget = N - headroom (§3 M2)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok, tok_sha = _load_tokenizer(Path(args.tokenizer), args.tokenizer_sha256)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lines, entries = [], []
    for rec in generate_suite(tok, args.seed, args.grid, args.depths,
                              args.trials, args.headroom):
        lines.append(json.dumps(rec, sort_keys=True))
        entries.append({k: rec[k] for k in
                        ("n", "depth", "trial", "key", "prompt_len",
                         "needle_start", "needle_len")}
                       | {"sha256_uint16": trial_sha256(rec["ids"])})

    payload = ("\n".join(lines) + "\n").encode()
    suite_path = out / "suite.jsonl.gz"
    with open(suite_path, "wb") as f:  # mtime=0 -> deterministic bytes
        with gzip.GzipFile(fileobj=f, mode="wb", mtime=0) as gz:
            gz.write(payload)

    manifest = {
        "assignment": "docs/cc/M2-g2lite.md §2 V5 + §3 M2",
        "format": "Mohtashami & Jaggi, Landmark Attention, arXiv:2305.16300 "
                  "(retrieved 2026-07-17); single needle (multi-needle = M3)",
        "seed": args.seed, "grid": args.grid, "depths": args.depths,
        "trials_per_cell": args.trials, "headroom": args.headroom,
        "prompt_budget": "N - headroom, exact",
        "metric": r"exact match of the first regex \d{5} match in the "
                  "decoded greedy continuation (<= 12 tokens, stop at eot)",
        "tokenizer": args.tokenizer, "tokenizer_sha256": tok_sha,
        "segments": {"preamble": PREAMBLE, "filler": FILLER,
                     "needle_template": NEEDLE_TEMPLATE, "question": QUESTION},
        "n_trials": len(entries),
        "suite_sha256_uncompressed_jsonl": hashlib.sha256(payload).hexdigest(),
        "suite_sha256_gz": hashlib.sha256(suite_path.read_bytes()).hexdigest(),
        "trials": entries,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"suite": str(suite_path), "n_trials": len(entries),
                      "sha256_gz": manifest["suite_sha256_gz"],
                      "sha256_jsonl": manifest[
                          "suite_sha256_uncompressed_jsonl"]}, indent=2))


if __name__ == "__main__":
    main()
