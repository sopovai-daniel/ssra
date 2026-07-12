# M2 Phase 1 — calibration report (RunPod launch flow)

**Date:** 2026-07-12 · **Milestone:** M2 Phase 1 (`docs/cc/M2-assignment.md` v1.1 §4;
launch-flow assignment 2026-07-12) · **Spec:** v1.2 · **Pod:** RunPod `e8r68jb8fduz8n`,
1× A100 PCIe 80 GB, **Secure on-demand $1.39/hr** (booked by Daniel, console
2026-07-12), datacenter **CA-MTL-3** (EU preferred but not required per AP-18; noted).

**Headline result:** infrastructure works end-to-end (AP-17 secret flow, GCS, AP-11
GPU kill+resume bit-for-bit), the §14 gate tests pass on GPU — but **the S2 core
pair (Gate G1 as specced) is not executable with the current SSRA implementation**:
SSRA OOMs on 80 GB at S2 for every batch tried, and its measured S1 throughput is
~32× below flat, projecting ≈ 253 EUR for a single hypothetical S2 SSRA run (>10×
the AP-12 single-run gate). Details in §5–§6; decision needed (§9, D-log proposals).
Per the STOP gate, no Phase 2 work has been started.

---

## 1. Environment snapshot (§7 i)

| | |
|---|---|
| GPU / driver | NVIDIA A100 80GB PCIe, driver 550.127.05 |
| pod | RunPod Secure, image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (digest `sha256:61a4aafb0094c…41fb5`, Docker Hub 2026-07-12), container disk 50 GB, no network volume |
| OS / python | Ubuntu 22.04.5 LTS · Python 3.11.10 |
| torch | **2.12.0+cu126** (see §8.2 pin correction) · triton 3.7.0 |
| key pins | numpy 2.4.6, PyYAML 6.0.3, pytest 9.0.3, datasets 5.0.0, pyarrow 24.0.0, tokenizers 0.22.2, google-cloud-storage 3.12.1, fla-core/flash-linear-attention 0.5.0, transformers 5.13.1 (unpinned — see §8.5) |
| gcloud | Google Cloud SDK 575.0.1 (installed by bootstrap; recorded per client-path check) |
| CPU quota | cgroup 26.35 vCPU (of 252 host cores) — `OMP_NUM_THREADS=MKL_NUM_THREADS=26` set for all runs (see §8.4) |
| code | measurement runs: working tree of `accdb1b` (model code = config commit `2b22848`); kill+resume: + `4a842a9` (vehicle YAML) and `124ee72` (GPU resume fix, §4) |
| stale pkgs removed | torchvision 0.19.1+cu124, torchaudio 2.4.1+cu124 (base-image leftovers built for torch 2.4.1; project uses neither) |

AP-18 image deviation (accepted by Daniel by executing the runbook): the project
Dockerfile image could not be built/pushed from the prep machine (no running Docker
daemon, 25 GB free disk, private repo, no registry write scope). The official RunPod
cu124 image + bootstrap-pinned environment was used instead; full resolved
environment above. AP-17 flow: RunPod Secret `gcp_ssra_runpod_sa` → env var →
key file; sanity gate `gsutil ls gs://ssra-poc-ew3` **PASSED before any billable
work**. No secret material appears in any log, config, or this report.

## 2. pytest on the box (§7 ii)

`logs/M2-phase1-pytest.log` (full output; also in GCS): **36 passed, 1 failed, 28–31 s**.

- **All blocking gate tests pass on GPU** (spec §14.1 shift/causality, §14.2
  completion, §14.3 gradient flow, §14.7 P3 determinism — plus checkpoint/resume,
  config validation, baseline flat/megabyte).
- The 1 failure is `test_baselines.py::test_loglinear_integration` (baseline (b),
  Phase 4 only). The M1 "Triton skip" did **not** reproduce as a skip: Triton exists
  here, the test ran for the first time and failed —
  `transformers/modeling_utils.py:2648: AttributeError: 'list' object has no
  attribute 'keys'` (tied-weights handling), an **fla 0.5.0 ↔ transformers 5.13.1
  incompatibility** (transformers is unpinned and resolved to 5.13.1 today).
  Phase 4 blocker only; D-log proposal in §9.2.

## 3. Measured tok/s + peak VRAM (§7 iii)

bf16 autocast (AP-16), ctx 1024, seed 1337, steady-state tok/s excluding the first
10 steps (harness `tok_per_s`; `peak_vram_gib` = `torch.cuda.max_memory_allocated`).
Configs `experiments/M2-cal-*.yaml` @ `2b22848`; per-run rows + costs in
`results/runs.md`; JSONL logs in `logs/` and GCS.

| config | SSRA-P1 tok/s | SSRA peak GiB | flat tok/s | flat peak GiB |
|---|---|---|---|---|
| S1 (24.2M/24.0M) b16 | **9,457** | **54.67** | 300,978 | 6.35 |
| S1 b32 | OOM (readout gather, +2.23 GiB @ 78.7 used) | — | 325,414 | 12.34 |
| S1 b64 | OOM | — | 341,948 | 24.34 |
| S2 (84.6M/84.3M) b16 | OOM | — | 128,730 | 10.85 |
| S2 b32 | OOM | — | 135,493 | 20.58 |
| S2 b64 | OOM | — | 140,444 | 40.01 |

All five SSRA failures are `torch.OutOfMemoryError` in the read-out
(`src/ssra/model.py:114` gather / einsum): the read-out materializes
B×h×N×cover×d_head tensors (≈2.2 GiB per layer per tensor at S1 b32) which autograd
retains. **SSRA vs flat at identical d/h/L: ~32× slower, ~9× more memory (S1 b16).**
The M1 G1a result (slope 0.983 vs 1.923) measured asymptotic *scaling*, which
remains valid; calibration exposes the *constant factor* at real scale.
No quality conclusions from any loss above (spec §16). P-C diagnostics:
`p1_attn_entropy ≈ ln(32) = 3.4657` throughout — consistent with M1/Phase 0.

## 4. AP-11 GPU kill+resume (§7 iv)

Vehicle: `M2-cal-s1-ssra-b16` (amended + committed `4a842a9` **before** launch,
after the planned vehicle `s1-ssra-b32` proved OOM; measurement logs preserved as
`logs/M2-cal-s1-ssra-b16-measurement.*`). ckpt every 30 steps, GCS mirror on.

1. First resume attempt **crashed and exposed a real GPU-only harness bug**:
   `load_checkpoint(map_location="cuda")` moved the saved RNG-state ByteTensors to
   the device; `Generator.set_state()` requires CPU. Invisible to the Phase-0 CPU
   unit test. **Fixed in `124ee72`** (`.cpu()` on both RNG states); CPU suite still
   green. This finding alone justifies the Phase-1 GPU verification step.
2. Verification after fix: SIGKILL at step ~72 (after ckpt@60; GCS object
   `…/ckpt/M2-cal-s1-ssra-b16/latest.pt`, 276.72 MiB, 20:15:44Z — checkpoint GCS
   upload path verified), `--resume` → `[resume] from step 60`. Resumed steps
   70/80/110 train losses **7.41796 / 7.33734 / 7.23545** and final val **7.24613**
   are **bit-for-bit identical** to the uninterrupted reference run. Continuous
   curve: PASS. Evidence: `logs/M2-cal-s1-ssra-b16{.log,-measurement.log,-killrun.stdout,-resume.stdout}`.

## 5. EUR projection, Phases 2–4 (§7 v)

Basis: measured tok/s (§3); $1.39/hr Secure; **ECB EUR/USD 1.1430 (fixing
2026-07-10, latest published before this Sunday session)** → 1.216 EUR/GPU-h.
Ladder alternatives at D-log-2026-07-12-verified list prices (re-verify on any
deploy day, Pravidlo W): A100 SXM $1.49, H100 PCIe $2.89, RTX 4090 $0.69, L4 $0.39.

| phase | run | GPU-h @ A100 PCIe | EUR | AP-12 flag |
|---|---|---|---|---|
| 2 (sweep, 4 SSRA runs @ 50M tok) | SSRA S1 b16 | 4×1.47 = 5.9 | 7.15 | exceeds the 2–4 GPU-h sweep target on its own |
| 2 (sweep, 4 flat runs @ 50M tok) | flat S1 b64 | 4×0.04 = 0.16 | 0.20 | — |
| 2b (confirmation pair @ 500M) | SSRA S1 b16 | 14.7 | **17.9** | under 25 EUR gate, but see §6 |
| 2b | flat S1 b64 | 0.41 | 0.50 | — |
| 3 (S2 pair @ 1.7B) | flat S2 b64 | 3.4 | 4.1 | — |
| 3 | **SSRA S2** | **not executable (OOM)**; hypothetical @ ~2.3k tok/s [ODHAD: S1 tok/s ÷ 4.17 FLOP ratio] ≈ 208 | **≈ 253** | **>10× the 25 EUR single-run gate; ≈84 % of the whole cap. STOP.** |
| 3 (S2 floor 850M, AP-12) | SSRA S2 | ≈ 104 | ≈ 127 | still >5× gate; >42 % cap. STOP. |
| 4 (b) GatedDeltaNet | — | blocked on transformers pin (§9.2) | — | defer decision |
| 4 (c) MEGABYTE | — | not measured (no committed config; M3-order cost ≈ flat) | — | — |

Ladder alternatives do not change the verdict: H100/A100-SXM have the same 80 GB
(SSRA S2 still OOMs; H100 ≈ +2.1× price for ≲2× speed), and the 24 GB fallbacks
(4090/L4) cannot even hold SSRA S1 b16 (54.7 GiB). **The SSRA blocker is
implementation memory/throughput, not hardware choice or budget arithmetic.**

## 6. AP-19 tier comparison + HW recommendation (§7 vi)

- **Secure on-demand: $1.39/hr** (booked; console 2026-07-12).
- **Community: not captured on deploy day** (confirmed by Daniel, 2026-07-12).
  Per Pravidlo W it is not reconstructed from memory or a later page-load; the
  Secure-vs-Community comparison is carried over to the pre-flight of the next
  launch (Phase 2+), where both console values of that day will be recorded.
  Preemption risk is mitigated either way:
  AP-11 is now GPU-verified bit-for-bit, max loss = 1 checkpoint interval
  (≤30 min wall-clock per AP-11; calibration used 30-step intervals).
- **Recommendation (numbers only, decision Daniel's):** A100 PCIe 80 GB is the
  right class for all *flat* M2 runs and any SSRA S1-b16-shaped work; Community
  admissible given AP-11. **However: do not launch Phase 2 as specced** until the
  §5 SSRA finding is decided (§9.1) — the sweep is symmetric by construction
  (AP-14), so the SSRA side dominates cost/feasibility of every subsequent phase.

## 7. Cost ledger (§7 vii)

| item | value |
|---|---|
| **billed (console, authoritative)** | pod `ssra-m2-cal` (`e8r68jb8fduz8n`), terminated 2026-07-12: **$3.4785989448 ≈ 3.04 EUR** (ECB 1.1430) — implies 2.5026 h ≈ 2 h 30 m 09 s at $1.39/hr |
| CC-observed window | 17:23:49Z (first SSH) → 20:39Z terminate signal; ≈ 3 h 15 m wall + Daniel's termination after the signal. **Delta vs billed (~45 min) not derivable from CC-side timestamps; recorded as-is, no reconstructed explanation** (earlier estimate ≈ $4.6 was wall-clock-based and is superseded) |
| wall-clock composition (CC window, informative) | productive ≈ 35 min (bootstrap+env 15, pytest 2×0.5, 12 runs ≈ 8, kill+resume ≈ 8, uploads/snapshot ≈ 3); incident idle ≈ 2 h 05 min — pytest thread-thrash (§8.4), mitigated for all subsequent work. Monetary attribution within the billed total not derivable (see delta note) |
| GCS | ≈ 290 MB new (ckpt 277 MB + logs) — negligible |
| **cumulative project spend vs 300 EUR** | **≈ 3.04 EUR ≈ 1.0 %** (M0–M1: 0; Phase 0: negligible GCS) |

Per-run wall-clock + EUR: `results/runs.md` rows (config commit `2b22848`).

## 8. Deviations + incidents (all recorded, none silent)

1. **AP-18 image** — official RunPod image + bootstrap pins instead of the project
   Dockerfile image (§1; accepted pre-launch via runbook).
2. **torch pin correction** — `torch==2.12.0` was never published for cu124 (cu124
   index ends at 2.6.0; verified on download.pytorch.org 2026-07-12). Installed
   **2.12.0+cu126** (same torch version as the dev/test environment; runs on driver
   550 via CUDA minor-version compatibility). §14 gate tests green on the result.
   `requirements-gpu.txt`/`docker/Dockerfile` still say cu124 — D-log §9.3.
3. **RunPod start-command decode did not execute** (dir `/root/.gcp` never created)
   even though the secret env var was present in PID-1 env and `/start.sh` ran.
   Worked around by extracting the var from `/proc/1/environ` (value never printed);
   `scripts/pod_bootstrap.sh` now does this natively as a fallback. Future launches
   need no start-command decode at all — bootstrap self-serves.
4. **Thread-thrash incident (~2 h idle, ≈2.6 EUR):** torch on CPU defaulted to 252
   threads (host cores) against a 26-vCPU cgroup quota → pytest crawled at ~1
   test/hour and looked alive. Fix: `OMP_NUM_THREADS=MKL_NUM_THREADS=26` for every
   process (baked into driver env; `.thread_env` on pod). Lesson recorded: export
   thread limits in bootstrap for all future pods.
5. **test_loglinear_integration failure** (§2) — fla 0.5.0 vs transformers 5.13.1;
   Phase 4 blocker only. Also removed stale base-image torchvision/torchaudio
   (broke transformers import under torch 2.12).
6. **Kill+resume vehicle switch** b32→b16 (§4; committed `4a842a9` pre-launch).
7. **GPU resume RNG bug found + fixed** (`124ee72`, §4).
8. **Region** CA-MTL-3, not EU (availability on deploy day; permitted by AP-18).

## 9. Open questions → proposed D-log entries (§7 viii)

1. **[BLOCKING for Phase 2+] SSRA implementation memory/throughput.** Proposed
   D-log: *"Calibration 2026-07-12: SSRA-P1 read-out materializes O(B·h·N·cover·d_h)
   tensors; S2 OOMs on 80 GB at any batch; S1 tok/s 9.5k vs flat 301k (≈32×). G1 at
   S2 as specced is infeasible within AP-12 (≈253 EUR/run hypothetical). Options:
   (a) read-out optimization work (chunked gather / recomputation / fused kernel) —
   new zadanie, must preserve spec §14 test behavior, complexity claims in docs/01
   §7 to be re-examined for the memory constant; (b) symmetric scale-down of the G1
   comparison (e.g. S2→S1-class, per 03's 'scale down, never drop the flat
   baseline'); (c) document as negative/engineering result for the publication.
   Decision: Daniel + Claude.ai project."*
2. **Phase 4 baseline (b):** *"Pin transformers to an fla-0.5.0-compatible version
   before Phase 4 (5.13.1 breaks GatedDeltaNet build: tied-weights API); verify on
   the box; record pin in requirements-gpu.txt."*
3. **Ratify environment pins:** *"torch 2.12.0+cu126 replaces the unsatisfiable
   cu124 pin (verified 2026-07-12); update requirements-gpu.txt + Dockerfile
   comments; RunPod official image + pod_bootstrap.sh is the standing launch path
   for M2 (AP-18 image line amended)."*
4. **AP-19 Community price** — not captured on deploy day (§6); proposed D-log:
   *"AP-19 tier comparison deferred to the next launch pre-flight: record both
   Secure and Community console prices of that day before deploying Phase 2+."*

## 10. Definition-of-done check (launch-flow §9)

AP-17 executed, sanity gate PASSED ✔ · Phase 1 steps 1–7 ✔ (12/12 configs
launched: 7 DONE, 5 recorded OOM) · kill+resume verified ✔ (after `124ee72`) ·
logs/artifacts in GCS (`gs://ssra-poc-ew3/m2/calibration/{logs,ckpt}/`) ✔ ·
YAMLs + runs.md + ledger ✔ · pod terminated, billed $3.4786 ≈ 3.04 EUR recorded ✔ ·
**STOP: CC stops here. No Phase 2 work. Daniel confirms HW + plan (D-log) — the
§9.1 decision is the gating item.**
