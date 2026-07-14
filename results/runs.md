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
| M2-phase0-cpu-smoke | 2026-06-14 | `experiments/M2-phase0-cpu-smoke.yaml` (commit e4fa16f) | SSRA P1, tokenized path | 2.70M params, FineWeb-Edu BPE-16k, 60×4×256 tok | MacBook M1, CPU fp32 | DONE | val loss 8.71173 (12.568 bits/tok); train 9.737→8.779; no divergence | `logs/M2-phase0-cpu-smoke.log` | functionality only (no quality conclusion). Phase-0 harness smoke: token shards + AP-11 checkpoint/resume (kill@30+resume reproduced the uninterrupted curve bit-for-bit, single-thread). Corpus FineWeb-Edu sample-10BT (odc-by, sha 87f09149), tokenizer sha256 019568a2; P-C p1_attn_entropy≈ln(32)=3.466 (M1-consistent, informative) |

| M2-cal-s1-ssra-b16 | 2026-07-12 | `experiments/M2-cal-s1-ssra-b16.yaml` (commit 2b22848) | SSRA P1 | 24.2M params, FineWeb-Edu BPE-16k, 120x16x1024 tok | RunPod A100 PCIe 80GB Secure $1.39/hr, CA-MTL-3, bf16 | DONE | 9,457 tok/s, peak VRAM 54.67 GiB; wall 211.6s (~0.07 EUR) | `logs/M2-cal-s1-ssra-b16.log` | Phase 1 calibration measurement only (no quality conclusions); launch subset depends on booked GPU VRAM; also AP-11 GPU kill+resume vehicle after b32 OOM (config amended in 4a842a9, committed pre-launch): SIGKILL@step~72, resume from ckpt@60, bit-for-bit identical curve; exposed+fixed GPU resume RNG bug (124ee72) |
| M2-cal-s1-ssra-b32 | 2026-07-12 | `experiments/M2-cal-s1-ssra-b32.yaml` (commit 2b22848) | SSRA P1 | 24.2M params, FineWeb-Edu BPE-16k, 200x32x1024 tok | RunPod A100 PCIe 80GB Secure $1.39/hr, CA-MTL-3, bf16 | FAILED-OOM | CUDA OOM on 80GB at readout gather (last alloc 2.23 GiB); <0.01 EUR | `logs/M2-cal-s1-ssra-b32.log` | Phase 1 calibration measurement only (no quality conclusions); launch subset depends on booked GPU VRAM; AP-11 GPU kill+resume vehicle (ckpt@50, GCS mirror) |
| M2-cal-s1-ssra-b64 | 2026-07-12 | `experiments/M2-cal-s1-ssra-b64.yaml` (commit 2b22848) | SSRA P1 | 24.2M params, FineWeb-Edu BPE-16k, 120x64x1024 tok | RunPod A100 PCIe 80GB Secure $1.39/hr, CA-MTL-3, bf16 | FAILED-OOM | CUDA OOM on 80GB at readout gather (last alloc 4.45 GiB); <0.01 EUR | `logs/M2-cal-s1-ssra-b64.log` | Phase 1 calibration measurement only (no quality conclusions); launch subset depends on booked GPU VRAM |
| M2-cal-s1-flat-b16 | 2026-07-12 | `experiments/M2-cal-s1-flat-b16.yaml` (commit 2b22848) | flat pre-norm Transformer | 24.0M params, FineWeb-Edu BPE-16k, 120x16x1024 tok | RunPod A100 PCIe 80GB Secure $1.39/hr, CA-MTL-3, bf16 | DONE | 300,978 tok/s, peak VRAM 6.34 GiB; wall 7.7s (~0.00 EUR) | `logs/M2-cal-s1-flat-b16.log` | Phase 1 calibration measurement only (no quality conclusions); launch subset depends on booked GPU VRAM |
| M2-cal-s1-flat-b32 | 2026-07-12 | `experiments/M2-cal-s1-flat-b32.yaml` (commit 2b22848) | flat pre-norm Transformer | 24.0M params, FineWeb-Edu BPE-16k, 120x32x1024 tok | RunPod A100 PCIe 80GB Secure $1.39/hr, CA-MTL-3, bf16 | DONE | 325,414 tok/s, peak VRAM 12.34 GiB; wall 13.9s (~0.00 EUR) | `logs/M2-cal-s1-flat-b32.log` | Phase 1 calibration measurement only (no quality conclusions); launch subset depends on booked GPU VRAM |
| M2-cal-s1-flat-b64 | 2026-07-12 | `experiments/M2-cal-s1-flat-b64.yaml` (commit 2b22848) | flat pre-norm Transformer | 24.0M params, FineWeb-Edu BPE-16k, 120x64x1024 tok | RunPod A100 PCIe 80GB Secure $1.39/hr, CA-MTL-3, bf16 | DONE | 341,948 tok/s, peak VRAM 24.34 GiB; wall 26.1s (~0.01 EUR) | `logs/M2-cal-s1-flat-b64.log` | Phase 1 calibration measurement only (no quality conclusions); launch subset depends on booked GPU VRAM |
| M2-cal-s2-ssra-b16 | 2026-07-12 | `experiments/M2-cal-s2-ssra-b16.yaml` (commit 2b22848) | SSRA P1 | 84.6M params, FineWeb-Edu BPE-16k, 120x16x1024 tok | RunPod A100 PCIe 80GB Secure $1.39/hr, CA-MTL-3, bf16 | FAILED-OOM | CUDA OOM on 80GB at readout gather (last alloc 1.27 GiB); <0.01 EUR | `logs/M2-cal-s2-ssra-b16.log` | Phase 1 calibration measurement only (no quality conclusions); launch subset depends on booked GPU VRAM |
| M2-cal-s2-ssra-b32 | 2026-07-12 | `experiments/M2-cal-s2-ssra-b32.yaml` (commit 2b22848) | SSRA P1 | 84.6M params, FineWeb-Edu BPE-16k, 120x32x1024 tok | RunPod A100 PCIe 80GB Secure $1.39/hr, CA-MTL-3, bf16 | FAILED-OOM | CUDA OOM on 80GB at readout gather (last alloc 3.71 GiB); <0.01 EUR | `logs/M2-cal-s2-ssra-b32.log` | Phase 1 calibration measurement only (no quality conclusions); launch subset depends on booked GPU VRAM |
| M2-cal-s2-ssra-b64 | 2026-07-12 | `experiments/M2-cal-s2-ssra-b64.yaml` (commit 2b22848) | SSRA P1 | 84.6M params, FineWeb-Edu BPE-16k, 120x64x1024 tok | RunPod A100 PCIe 80GB Secure $1.39/hr, CA-MTL-3, bf16 | FAILED-OOM | CUDA OOM on 80GB at readout gather (last alloc 5.08 GiB); <0.01 EUR | `logs/M2-cal-s2-ssra-b64.log` | Phase 1 calibration measurement only (no quality conclusions); launch subset depends on booked GPU VRAM |
| M2-cal-s2-flat-b16 | 2026-07-12 | `experiments/M2-cal-s2-flat-b16.yaml` (commit 2b22848) | flat pre-norm Transformer | 84.3M params, FineWeb-Edu BPE-16k, 120x16x1024 tok | RunPod A100 PCIe 80GB Secure $1.39/hr, CA-MTL-3, bf16 | DONE | 128,730 tok/s, peak VRAM 10.85 GiB; wall 17.0s (~0.01 EUR) | `logs/M2-cal-s2-flat-b16.log` | Phase 1 calibration measurement only (no quality conclusions); launch subset depends on booked GPU VRAM |
| M2-cal-s2-flat-b32 | 2026-07-12 | `experiments/M2-cal-s2-flat-b32.yaml` (commit 2b22848) | flat pre-norm Transformer | 84.3M params, FineWeb-Edu BPE-16k, 120x32x1024 tok | RunPod A100 PCIe 80GB Secure $1.39/hr, CA-MTL-3, bf16 | DONE | 135,493 tok/s, peak VRAM 20.58 GiB; wall 31.6s (~0.01 EUR) | `logs/M2-cal-s2-flat-b32.log` | Phase 1 calibration measurement only (no quality conclusions); launch subset depends on booked GPU VRAM |
| M2-cal-s2-flat-b64 | 2026-07-12 | `experiments/M2-cal-s2-flat-b64.yaml` (commit 2b22848) | flat pre-norm Transformer | 84.3M params, FineWeb-Edu BPE-16k, 120x64x1024 tok | RunPod A100 PCIe 80GB Secure $1.39/hr, CA-MTL-3, bf16 | DONE | 140,444 tok/s, peak VRAM 40.01 GiB; wall 60.5s (~0.02 EUR) | `logs/M2-cal-s2-flat-b64.log` | Phase 1 calibration measurement only (no quality conclusions); launch subset depends on booked GPU VRAM |

| M2-recal-s1-flat-b16 | 2026-07-13 | `experiments/M2-cal-s1-flat-b16.yaml` (commit 2b22848) | flat pre-norm Transformer | 24.0M params, FineWeb-Edu BPE-16k, 120x16x1024 tok | RunPod A100 SXM 80GB Secure $1.49/hr, US-MD-1, bf16 | DONE | 319,945 tok/s, peak VRAM 6.345 GiB; wall 7.1s (~0.00 EUR) | `logs/M2-recal-s1-flat-b16.log` | M2 re-calibration (D-log 2026-07-13 GO) env anchor; code 4abf3b4 (model code 9417399); measurement only (no quality conclusions) |
| M2-recal-s1-ssra-b16 | 2026-07-13 | `experiments/M2-cal-s1-ssra-b16.yaml` (commit 4a842a9) | SSRA P1 | 24.2M params, ditto | ditto | DONE | 27,079 tok/s, peak VRAM 18.557 GiB; wall 124.6s (~0.05 EUR) | `logs/M2-recal-s1-ssra-b16.log` | re-calibration regression point vs calib 9,457 tok/s / 54.67 GiB (2.86x faster, 2.95x less mem); D5 proj 15.40 GiB -> +20.5% error; ckpt@30+GCS per committed config (ckpt time excluded from tok/s) |
| M2-recal-s2-ssra-b16 | 2026-07-13 | `experiments/M2-cal-s2-ssra-b16.yaml` (commit 2b22848) | SSRA P1 | 84.6M params, ditto | ditto | DONE | **12,335 tok/s, peak VRAM 41.207 GiB**; wall 164.6s (~0.06 EUR) | `logs/M2-recal-s2-ssra-b16.log` | **GATE — measurable-target input** (calib: OOM at any batch); D5 proj 36.59 GiB -> +12.6% error; 850M-tok projection 24.95 EUR @ $1.49 SXM (see report) |
| M2-recal-s1-ssra-b32 | 2026-07-13 | `experiments/M2-cal-s1-ssra-b32.yaml` (commit 2b22848) | SSRA P1 | 24.2M params, 200x32x1024 tok | ditto | DONE | 31,759 tok/s, peak VRAM 36.584 GiB; wall 261.5s (~0.09 EUR) | `logs/M2-recal-s1-ssra-b32.log` | scaling point (calib: OOM); D5 proj 30.44 GiB -> +20.2% error; 200 steps + ckpt@50+GCS per committed config |
| M2-recal-s1-ssra-b64 | 2026-07-13 | `experiments/M2-cal-s1-ssra-b64.yaml` (commit 2b22848) | SSRA P1 | 24.2M params, 120x64x1024 tok | ditto | DONE | 33,357 tok/s, peak VRAM 72.624 GiB; wall 243.4s (~0.09 EUR) | `logs/M2-recal-s1-ssra-b64.log` | optional marginal probe (calib: OOM); D5 proj 60.51 GiB -> +20.0% error |
| M2-recal-s2-ssra-b32 | 2026-07-13 | `experiments/M2-cal-s2-ssra-b32.yaml` (commit 2b22848) | SSRA P1 | 84.6M params, 120x32x1024 tok | ditto | FAILED-OOM | CUDA OOM in first backward (2.00 GiB req @ 78.22 GiB allocated); <0.02 EUR | `logs/M2-recal-s2-ssra-b32.log` | optional marginal probe, single attempt per assignment (OOM = valid outcome, no retry); D5 proj 71.92 GiB + ~20% model error ~= 86 GiB > 80 — consistent |

| m2-data-900m | 2026-07-13 | `experiments/M2-data-900m.yaml` (commit 7597ff4; code at execution 7bb2d1a) | data pipeline (no model) | 913,605,620 train + 48,050,671 val tok packed, FineWeb-Edu BPE-16k frozen | RunPod CPU pod `ssra-m2-data`, 16 vCPU EPYC 4564P / 32 GB, runpod-ubuntu, $0.568/hr (console 2026-07-13) | DONE | pack 195.9 s (~4.7M train-tok/s); val-eval-2M sha256 bde526d2 | `logs/m2-data-900m.log` | M2 Task A (M2-phase2-sweep §2); shards + manifests at `gs://ssra-poc-ew3/m2/data/m2-data-900m/`; corpus sample-10BT (odc-by, hub sha 87f09149, retrieved 2026-07-13); tokenizer FROZEN sha 019568a2; 2 pre-run import aborts preserved as `logs/m2-data-900m-import-abort*.log` (see `results/M2-sweep.md` §A.6/A.8) |

| m2-sweep-localsmoke-r0 | 2026-07-14 | `experiments/m2-sweep-localsmoke-r0.yaml` (commit 482bdb5) | SSRA P1 (tiny, d64/L2) | 1.15M params, Phase-0 shards, 12x4x256 tok | MacBook M1, CPU fp32 | DONE | harness plumbing only — sha256 gates 4/4, eval_bin final_eval pass (1306 win), wall 13.7 s, 0 EUR | `logs/m2-sweep-localsmoke-r0.log` | M2 Task B local prep (M2-phase2-sweep §3 prep; see `results/M2-sweep.md` §B.0); throwaway name per task brief; NO conclusions of any kind from its loss (spec §16); no GCS access |

Ledger note (2026-07-13, Task A CPU pod): wall-clock estimate ≈ 0.61 h x $0.568/hr
≈ $0.35 ≈ 0.30 EUR (ECB 1.1430 carried); billed console total for pod
`ssra-m2-data`: **$0.5567 (0.98 h billed, region EUR-IS-1) ≈ 0.49 EUR** (console
2026-07-13 evening; **provisional** pending T+1 re-check per the D-log 2026-07-13
correction row).

Ledger correction (console 2026-07-13 evening, appended 2026-07-14 — historical
rows and notes below are left unchanged, append-only): pod `ssra-m2-cal`
(Phase 1 calibration) final console total settled at **$4.6293 ≈ 4.05 EUR**
(ECB 1.1430), superseding the $3.4786 ≈ 3.04 EUR read on 2026-07-12
(late-settling charges; the final figure matches the original CC timestamp
estimate, resolving the "billed below estimate" delta noted for Phase 1).
Per the new rule (D-log 2026-07-13 correction row) console figures are
provisional until a T+1 re-check: `ssra-m2-recal` **$0.9700 ≈ 0.85 EUR** —
unchanged on 2026-07-13, provisional, re-check due 2026-07-14; `ssra-m2-data`
**$0.5567 ≈ 0.49 EUR** — provisional, re-check due 2026-07-14. Cumulative M2
spend: **≈ 5.39 EUR** (4.05 final + 0.85 provisional + 0.49 provisional) of the
300 EUR envelope.

Ledger confirmation (T+1 re-check, console data supplied by Daniel 2026-07-14;
appended per the D-log 2026-07-13 correction row — prior rows and notes left
unchanged, append-only):
- pod `ssra-m2-recal`: **$0.9700 ≈ 0.85 EUR CONFIRMED** at T+1 (2026-07-14) —
  did NOT grow to the CC wall-clock estimate $1.30; provisional flag CLOSED.
- pod `ssra-m2-data`: **$0.5567 ≈ 0.49 EUR CONFIRMED** at T+1 (2026-07-14)
  (region EUR-IS-1, ~0.98 h billed); provisional flag CLOSED. The billed-total
  figures already filled in `results/M2-sweep.md` §A.5 and the Task A ledger
  note above match this confirmed value — verified, not modified.
- Cumulative M2 spend after confirmation: **≈ 5.39 EUR ≈ 1.8 %** of the
  300 EUR cap (4.05 final + 0.85 confirmed + 0.49 confirmed; ECB 1.1430).

Ledger note (2026-07-13, re-calibration): per-run EUR above are wall-clock x $1.49/hr
estimates; the console-authoritative billed total for pod `ssra-m2-recal`
(`1u7wmoy6l71ull`) is **$0.9700 ~ 0.85 EUR** (console 2026-07-13, ECB 1.1430;
region US-MD-1) — ~25 % below the CC wall-clock estimate ($1.30), same pattern as
Phase 1; the delta is not reconstructed from CC-side timestamps (Pravidlo W); see
`results/M2-recalibration.md` §6.

Ledger note (2026-07-13, per D-log 2026-07-12 / HO-10 open item #2): the per-run EUR
figures above are wall-clock x $1.39/hr estimates (they sum to ~0.13 EUR); the
console-authoritative billed total for the whole calibration pod is $3.4786 ~ 3.04 EUR
(pod lifetime incl. the ~2 h idle incident). Per-run attribution within the billed
total is **not derivable** from CC-side timestamps and is not reconstructed
(Pravidlo W) — see `results/M2-calibration.md` §7.
Smoke chain provenance: runs executed back-to-back (`logs/M1-smoke-chain.log`);
`meta.commit` is 184d9bd for p1 and f06263f for the rest solely because the
docs-only HO-04 commit landed 13 s after the chain started — model code
identical for the whole chain.
