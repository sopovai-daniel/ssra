# Run ledger
One row per run. A run without a committed config in `experiments/` does not exist.

| run id | date | config | model / variant | scale | HW | status | key metric | log artifact | notes |
|---|---|---|---|---|---|---|---|---|---|
| M1-smoke-p1 | 2026-06-12 | `experiments/M1-smoke-p1.yaml` (commit 21ceeab) | SSRA P1 (latent-query pool) | 4.00M params, char, 2000×4×512 tok | MacBook M1 16GB, MPS fp32 | DONE | val loss 1.56295 (2.2549 bpc) | `logs/M1-smoke-p1.log` | G1b-D3 pair member; corpus Tiny Shakespeare (karpathy/char-rnn, 1,115,394 B, vocab 65); P-C: Q_φ attention ~uniform throughout |
| M1-smoke-p3 | 2026-06-12 | `experiments/M1-smoke-p3.yaml` (commit 21ceeab) | SSRA P3 (top-k select, STE) | 3.98M params, ditto | ditto | DONE | val loss 1.56980 (2.2647 bpc) | `logs/M1-smoke-p3.log` | G1b-D3 pair member: gap vs P1 +0.44% (X=5%); τ 2.0→0.5, λ_lb=0.01; no divergence |
| M1-smoke-p2 | 2026-06-12 | `experiments/M1-smoke-p2.yaml` (commit 21ceeab) | SSRA P2 (strided merge) | 4.63M params, ditto | ditto | DONE | val loss 1.56045 (2.2513 bpc) | `logs/M1-smoke-p2.log` | control pool; no divergence |
| M1-smoke-flat | 2026-06-12 | `experiments/M1-smoke-flat.yaml` (commit 21ceeab) | flat pre-norm Transformer | 3.96M params, ditto | ditto | DONE | val loss 1.57880 (2.2777 bpc) | `logs/M1-smoke-flat.log` | baseline (a), same d/h/L |
| M1-smoke-megabyte | 2026-06-12 | `experiments/M1-smoke-megabyte.yaml` (commit 21ceeab) | MEGABYTE-style 2-level | 4.39M params, ditto | ditto | DONE | val loss 2.00817 (2.8972 bpc) | `logs/M1-smoke-megabyte.log` | baseline (c), patch=8; AP-5 short smoke |
| M1-bench | 2026-06-12 | `scripts/bench_throughput.py` defaults (commit 184d9bd) | SSRA vs flat, d=192/h=8/L=2 | N 1k–16k, B=1 | ditto | DONE | G1a slopes: SSRA 0.983, flat 1.923 | `logs/M1-throughput.log`, `results/M1-throughput.{json,png}` | swap-aware AP-6 fit; superseded runs kept in `logs/M1-throughput-*.log` |

Smoke chain provenance: runs executed back-to-back (`logs/M1-smoke-chain.log`);
`meta.commit` is 184d9bd for p1 and f06263f for the rest solely because the
docs-only HO-04 commit landed 13 s after the chain started — model code
identical for the whole chain.
