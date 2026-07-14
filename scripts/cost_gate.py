"""Early cost gate — supervisory read-out (M2 Phase 3, M2-phase3-core-pair §3).

Read-only tool: parses a JSONL training log written by scripts/train.py,
computes the windowed steady-state TRAINING tok/s between two diagnostic
records, projects the full-run cost at the booked hourly rate + carried ECB
rate, and prints PASS / STOP against the scoped cap (30 EUR for
m2-core-ssra-s2-850m, D-log 2026-07-13). No training-code changes (AP-20
boundary): the harness already logs cumulative pure-train throughput
(`tok_per_s`, val/checkpoint time excluded, first MEAS_SKIP=10 steps
skipped); this script only differences two of those records:

  train_time_i = tokens_per_step * meas_steps_i / tok_per_s_i,
  meas_steps_i = step_i - start_step - MEAS_SKIP + 1
  windowed tok/s = tokens_per_step * (step_b - step_a)
                   / (train_time_b - train_time_a)

Usage (gate values per M2-phase3-core-pair §3, rate re-checked on deploy day):
  python scripts/cost_gate.py logs/m2-core-ssra-s2-850m.log \
      --rate-usd-hr 1.497 --ecb 1.1430 --cap-eur 30 \
      --window 1000 1500 [--min-window 200] [--anchor-tok-s 12335]

Exit codes: 0 = PASS, 2 = STOP (projected cost > cap), 1 = usage/log error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MEAS_SKIP = 10  # scripts/train.py throughput accounting constant


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="logs/<run>.log JSONL from scripts/train.py")
    ap.add_argument("--rate-usd-hr", type=float, required=True,
                    help="booked hourly rate (console value of the day)")
    ap.add_argument("--ecb", type=float, required=True,
                    help="carried ECB EUR/USD rate (AP-12 ledger rate)")
    ap.add_argument("--cap-eur", type=float, default=30.0,
                    help="scoped hard cap in EUR (default 30)")
    ap.add_argument("--window", type=int, nargs=2, default=(1000, 1500),
                    metavar=("START", "END"),
                    help="step window for the steady-state measurement")
    ap.add_argument("--min-window", type=int, default=200,
                    help="minimum steps between the two records (gate: >= 200)")
    ap.add_argument("--total-tokens", type=int, default=None,
                    help="override; default steps*batch_size*seq_len from meta")
    ap.add_argument("--anchor-tok-s", type=float, default=None,
                    help="informative-only sanity anchor (+/-10%%, non-gating)")
    a = ap.parse_args()

    meta, recs = None, []
    for line in Path(a.log).read_text().splitlines():
        rec = json.loads(line)
        if "meta" in rec:
            meta = rec["meta"]  # last meta wins (resume appends a new one)
        elif "tok_per_s" in rec and "train_loss" in rec:
            recs.append(rec)
    if meta is None:
        print("[cost-gate] ERROR: no meta record in log", file=sys.stderr)
        return 1

    tokens_per_step = meta["batch_size"] * meta["seq_len"]
    total_tokens = a.total_tokens or meta["steps"] * tokens_per_step
    start_step = meta.get("resumed_from") or 0

    lo, hi = a.window
    window = [r for r in recs if lo <= r["step"] <= hi]
    if len(window) < 2:
        print(f"[cost-gate] ERROR: <2 diagnostic records in step window "
              f"[{lo}, {hi}] (have {len(window)})", file=sys.stderr)
        return 1
    ra, rb = window[0], window[-1]
    span = rb["step"] - ra["step"]
    if span < a.min_window:
        print(f"[cost-gate] ERROR: window spans {span} steps "
              f"< required {a.min_window}", file=sys.stderr)
        return 1

    def train_time(r: dict) -> float:
        meas = r["step"] - start_step - MEAS_SKIP + 1
        return tokens_per_step * meas / r["tok_per_s"]

    tok_s = tokens_per_step * span / (train_time(rb) - train_time(ra))
    hours = total_tokens / tok_s / 3600.0
    usd = hours * a.rate_usd_hr
    eur = usd / a.ecb
    verdict = "PASS" if eur <= a.cap_eur else "STOP"

    out = {"cost_gate": {
        "log": a.log, "run": meta["run"],
        "window_steps": [ra["step"], rb["step"]], "window_span": span,
        "steady_tok_s": round(tok_s, 1),
        "total_tokens": total_tokens,
        "projected_hours": round(hours, 3),
        "rate_usd_hr": a.rate_usd_hr, "ecb": a.ecb,
        "projected_usd": round(usd, 2), "projected_eur": round(eur, 2),
        "cap_eur": a.cap_eur, "verdict": verdict,
    }}
    if a.anchor_tok_s:  # informative only, never gates (assignment §3)
        dev = tok_s / a.anchor_tok_s - 1.0
        out["cost_gate"]["anchor_tok_s"] = a.anchor_tok_s
        out["cost_gate"]["anchor_deviation_pct"] = round(100 * dev, 2)
        out["cost_gate"]["anchor_within_10pct"] = abs(dev) <= 0.10
    print(json.dumps(out, indent=2))
    if verdict == "STOP":
        print(f"[cost-gate] STOP: projected {eur:.2f} EUR > cap "
              f"{a.cap_eur} EUR — abort the run, status ABORTED-cost-gate, "
              f"upload logs, report (M2-phase3-core-pair §3).")
        return 2
    print(f"[cost-gate] PASS: projected {eur:.2f} EUR <= cap {a.cap_eur} EUR")
    return 0


if __name__ == "__main__":
    sys.exit(main())
