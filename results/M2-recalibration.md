# M2 — re-calibration report (post read-out optimization R1+R4+R5)

**Date:** 2026-07-13 · **Assignment:** `docs/cc/M2-recalibration.md` v1 (D-log
2026-07-13 GO; step 3 of D-log 2026-07-12 option (a)) · **Spec:** v1.2 ·
**Pod:** RunPod `ssra-m2-recal` (`1u7wmoy6l71ull`), 1× **A100 SXM 80 GB**,
**Secure on-demand $1.49/hr GPU + $0.007/hr disk = $1.497/hr** (console
2026-07-13, booked by Daniel), datacenter **US-MD-1**, pod start 17:18 local.
**Code:** commit `4abf3b4` working tree (model code = `9417399`; later commits
docs/pin-comments only). No code changes on the box.

**Headline result: the S2 b16 gate measurement passes both measurable-target
inputs.** SSRA S2 b16 **trains** on the A100 80 GB (Phase 1: OOM at every
batch) at **12,334.7 tok/s ≥ ~11.5k** with peak 41.21 GiB, projecting
**24.95 EUR @ 850M tokens at the booked $1.49/hr** — under the 25 EUR line,
with a 0.2 % margin (§4). The D5 memory model under-projects consistently by
**+12.6…+20.5 %** (§3). Verdict = Daniel (D-log); no Phase 2 work started.

## 1. Pre-flight record (assignment §2)

| item | value |
|---|---|
| AP-19 Secure on-demand | **$1.49/hr** A100 SXM 80 GB (+$0.007/hr disk), RunPod deploy console, 2026-07-13 |
| AP-19 Community | **not capturable in console 2026-07-13** — deploy flow showed Secure only; recorded as such per Pravidlo W, no backfill. Tier comparison remains open (carried since Phase 1) |
| credit check | confirmed by Daniel pre-deploy (shared account; ≥ 1 h on-demand minimum + ~2 EUR envelope); no top-up today |
| ECB EUR/USD | **1.1430** (fixing 2026-07-10, latest published; no top-up today) |
| HW ladder | step 2: **A100 PCIe 80 GB unavailable 2026-07-13 → A100 SXM** (same 80 GB class; no AP-12 re-projection needed beyond §4) |
| AP-17 | secret flow exactly as calibration: env var `GCP_SA_KEY_B64` on pod, image-default start command, bootstrap `/proc/1/environ` fallback decoded the key; **sanity gate `gsutil ls gs://ssra-poc-ew3` PASSED before any billable work** |
| thread limits | cgroup quota 13.6 vCPU (of 128 host) → `OMP_NUM_THREADS=MKL_NUM_THREADS=13` for every process (thread-thrash lesson) |

## 2. Environment snapshot + pytest

| | |
|---|---|
| GPU / driver | NVIDIA A100-SXM4-80GB, driver 580.126.16 |
| pod | Secure, US-MD-1, container disk 50 GB, no network volume; image = image-default start command; base env had torch 2.4.1+cu124 (consistent with the Phase-1 `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` image; tag not verifiable from inside the pod — see §7.6) |
| OS / python | Ubuntu 22.04.5 LTS · Python 3.11.10 |
| torch | **2.12.0+cu126** (bootstrap pin) · triton 3.7.0 · numpy 2.4.6 |
| gcloud | Google Cloud SDK 575.0.1 (installed by bootstrap) |
| stale pkgs removed | torchvision 0.19.1+cu124, torchaudio 2.4.1+cu124 (same as Phase 1) |
| code | `4abf3b4` (model code `9417399`); configs `experiments/M2-cal-*.yaml` unchanged (`2b22848`, b16 vehicle amendment `4a842a9`) |

**pytest on the box (`logs/M2-recal-pytest.log`, also in GCS): 64 passed,
1 failed, 28.17 s.** The single failure is the known
`test_baselines.py::test_loglinear_integration` (fla 0.5.0 × transformers
tied-weights API, Phase-4-only) — identical failure mode to Phase 1, status
unchanged. All spec §14 gate tests pass on GPU, **including the new
read-out A/B equivalence tests — no CUDA-fp32 atol excess observed** (the
CPU fp32 referee remains authoritative per AP-2; already certified green).
Suite grew 37 → 65 tests vs Phase 1 (read-out optimization tests, R1+R4+R5).

## 3. Measured vs D5 projected (assignment §4, model-error calibration)

bf16 autocast (AP-16), ctx 1024, seed 1337, steady-state tok/s excluding the
first 10 steps; `peak_vram_gib` = `torch.cuda.max_memory_allocated`. Logs:
`logs/M2-recal-*.log` + `gs://ssra-poc-ew3/m2/recalibration/logs/`.

| # | run | tok/s | peak GiB measured | D5 projected GiB | model error | Phase 1 (PCIe) |
|---|---|---|---|---|---|---|
| 1 | flat S1 b16 (anchor) | 319,945 | 6.345 | — | — | 300,978 / 6.35 GiB |
| 2 | SSRA S1 b16 | 27,079 | 18.557 | 15.40 | **+20.5 %** | 9,457 / 54.67 GiB |
| 3 | **SSRA S2 b16 (GATE)** | **12,335** | **41.207** | 36.59 | **+12.6 %** | OOM |
| 4 | SSRA S1 b32 | 31,759 | 36.584 | 30.44 | **+20.2 %** | OOM |
| 5 | SSRA S1 b64 | 33,357 | 72.624 | 60.51 | **+20.0 %** | OOM |
| 6 | SSRA S2 b32 | OOM (first backward: 2.00 GiB req @ 78.22 GiB alloc) | > 79 (capacity 79.25) | 71.92 | OOM consistent with +20 % (≈ 86 GiB) | OOM |

**Model error, explicit:** the D5 analytic memory model under-projects
measured peak VRAM by **+20.0…+20.5 % at S1 (all batches) and +12.6 % at S2
b16**; the S2 b32 OOM is consistent with the same band. For Phase 2 planning,
treat D5 projections with a **×1.20 error bar** (S1-calibrated; S2 ran
lighter). Anchor comparability: flat S1 b16 is +6.3 % vs Phase 1 (SXM vs
PCIe) — SSRA gains below are architecture, not hardware: **SSRA S1 b16 is
2.86× faster and uses 2.95× less memory than Phase 1**; the SSRA-vs-flat
throughput gap at S1 b16 narrowed from ~32× to **11.8×**.

P-C diagnostics: `p1_attn_entropy ≈ ln(32) = 3.4657` throughout, participation
min/max ~0.055/0.072 — consistent with M1/Phase 0/Phase 1. Losses recorded in
logs; **no quality conclusions from any loss (spec §16)**.

## 4. Gate arithmetic + EUR projection, S2 @ 850M tokens (assignment §1, §4)

Measured S2 b16: 12,334.7 tok/s → 850M tokens = 68,911 s = **19.14 GPU-h**.
ECB EUR/USD 1.1430.

| price basis | USD | EUR | vs 25 EUR line |
|---|---|---|---|
| **booked: Secure A100 SXM $1.49/hr** (console 2026-07-13) | $28.52 | **24.95** | **under, margin 0.05 EUR (0.2 %)** |
| Community A100 SXM | not capturable 2026-07-13 (Pravidlo W) | — | — |
| reference: $1.39/hr A100 PCIe (D-log target basis; 2026-07-12 price, PCIe unavailable today) | $26.61 | 23.28 | under (informative) |

**Measurable-target inputs: S2 b16 trains ✔ · 12,334.7 ≥ ~11.5k tok/s ✔ ·
projected 24.95 EUR ≤ 25 EUR at the booked price ✔.** Notes for the verdict
(Daniel's): (i) the margin at $1.49 is 0.2 % — any price/availability drift
flips it; the $1.39 PCIe basis gives 6.9 % margin; (ii) disk $0.007/hr adds
≈ 0.12 EUR over 19.14 h if included: 25.07 EUR — the target was defined on
GPU rate; both numbers stated to keep the verdict honest; (iii) the projection
is per-run steady-state, excludes checkpoint/restart overhead (AP-11 bounds
loss at ≤ 1 ckpt interval).

## 5. AP-12 status

Single-run projection 24.95 EUR < 25 EUR gate (marginal — see §4). This
session's envelope: measured runs sum to ≈ 0.31 EUR wall-clock estimate;
pod total well inside the ~1–2 EUR envelope (billed total §6). Cumulative
project spend ≈ 3.04 EUR (Phase 1) + this pod ≈ **< 5 EUR ≈ 1.6 % of the
300 EUR cap** — no 50 %/80 % triggers.

## 6. Cost ledger

| item | value |
|---|---|
| billed (console, authoritative) | **$0.9700 ≈ 0.85 EUR** (console 2026-07-13, ECB 1.1430) — pod `ssra-m2-recal` (`1u7wmoy6l71ull`), $1.497/hr total rate; ≈ 25 % below the CC estimate, same pattern as Phase 1, delta not reconstructed (Pravidlo W) |
| CC-observed window | pod start 17:18 local (15:18Z) → terminate signal ≈ 16:10Z; ≈ 52 min wall ⇒ ≈ $1.30 ≈ 1.14 EUR estimate (informative only) |
| wall-clock composition | bootstrap + env ≈ 8 min, pytest 28 s, 6 runs ≈ 18 min, uploads/log handling ≈ 5 min, CC orchestration gaps remainder |
| GCS | logs ≈ 35 KiB new under `m2/recalibration/logs/`; two `latest.pt` ckpt objects overwritten in place (§7.5) — negligible |
| per-run estimates | `results/runs.md` rows (2026-07-13) |

## 7. Deviations + incidents (all explicit, none silent)

1. **HW ladder step 2:** A100 PCIe 80 GB unavailable in console 2026-07-13 →
   A100 SXM $1.49/hr Secure booked (ladder-conform). Same 80 GB class; flat
   anchor +6.3 % vs PCIe recorded for comparability (§3).
2. **AP-19 Community price not capturable** (deploy flow showed Secure only,
   2026-07-13). Recorded as such, no backfill (Pravidlo W). The
   Secure-vs-Community comparison remains open for the next launch.
3. **`rsync` absent** on the pod image → repo copied via tar-over-ssh
   (content identical, `git status` clean at `4abf3b4` on the box).
4. **Log filename collision (working tree only):** pulling today's logs
   clobbered six committed Phase 1 `logs/M2-cal-*.log` files locally (same
   harness `run_name` → same filename). Recovered immediately: today's logs
   renamed to `logs/M2-recal-*.log`, Phase 1 files restored byte-identical
   from git. Committed history was never at risk; GCS objects renamed to
   match. Lesson: re-run of a committed config needs a distinct log name —
   flag for the next assignment.
5. **Phase 1 GCS checkpoint objects overwritten:** the committed
   `s1-ssra-b16` (ckpt@30) and `s1-ssra-b32` (ckpt@50) configs carry Phase 1
   `gcs_ckpt_dir` paths, so today's runs replaced the Phase 1 `latest.pt`
   objects (incl. the kill+resume evidence object of 2026-07-12). The
   kill+resume evidence itself lives in committed logs and the Phase 1
   report and is unaffected; checkpoints are transient artifacts by design
   (AP-11). Recorded, not reconstructed.
6. **Image tag not verifiable from inside the pod** (no image env var;
   image-default start command). Observed base env (torch 2.4.1+cu124,
   Ubuntu 22.04.5, Python 3.11.10) is consistent with the Phase 1 image;
   the console value at deploy (Daniel-side) is authoritative if needed.
7. **Pod-scoped `RUNPOD_API_KEY` surfaced in CC session output** while
   probing PID-1 env for the image tag (it sits in `/proc/1/environ`
   alongside the RunPod metadata). It appears nowhere in the repo, logs, or
   GCS; pod-scoped keys are invalidated at pod termination (imminent per
   AP-18). Flagged for Daniel's awareness; no action believed needed
   post-terminate.
8. **`s1-ssra-b32` ran its committed 200-step kill+resume-vehicle shape**
   (200×32×1024, ckpt@50) rather than the 120-step probe shape — configs
   were bound "unchanged" by the assignment; checkpoint time is excluded
   from the harness tok/s accounting, so the steady-state measurement is
   comparable.
9. **Region US-MD-1** (not EU; permitted per AP-18, data = public
   FineWeb-Edu).

## 8. Definition-of-done check

Pre-flight recorded (§1) ✔ · AP-17 gate PASSED before billable work ✔ ·
pytest once, known-failure-only ✔ · run set #1–#4 captured, optional #5–#6
single attempts (one DONE, one OOM-recorded) ✔ · measured-vs-D5 table with
explicit model error ✔ · EUR projection at booked price, Community recorded
as not capturable ✔ · logs in repo + GCS ✔ · runs.md rows appended ✔ ·
billed console total ✔ (**$0.9700 ≈ 0.85 EUR**, filled post-terminate,
console-authoritative) · no code/config/allocator changes, no OOM retries,
no Phase 2 work ✔.

**re-calibration report ready for verdict**
