# Zadanie pre CC — M2 Phase 2: symmetric S1 lr/dropout sweep (AP-14)

**Version:** v1.1 (2026-07-13; Task A moved to a dedicated CPU pod per Daniel's decision — §2, §6) · originally v1 (2026-07-13) · **Milestone:** M2, Phase 2 per `docs/cc/M2-assignment.md` v1.1 §4 · **Mode:** veto-based — AP items introduced here stand unless Daniel vetoes them. AP-1…AP-20 remain in force where applicable.

**Authority chain (binding):** `docs/spec.md` v1.2 > `docs/cc/M2-assignment.md` v1.1 > `docs/cc/M2-runpod-launch.md` v1 (launch/lifecycle) > this document. Project state: `docs/00` D-log (2026-07-13: recalibration verdict PASSED, GO Phase 2). Closed decisions (D1–D6, Q1–Q5, MD-1…MD-13, AP-1…AP-20, gates G0/G1a/G1b-D3) must not be reopened. On ambiguity or conflict: STOP, write the question + a proposed D-log entry into the report. Never improvise design changes.

**Context anchors (from `results/M2-recalibration.md`, commit `819fcb2` — measured, do not re-derive):**
flat S1 b16 = 319,945 tok/s / 6.35 GiB · SSRA S1 b16 = 27,079 tok/s / 18.56 GiB · booked HW class = A100 80 GB (SXM $1.49/hr Secure on 2026-07-13; re-check on deploy day). Code = `9417399` model code (R1+R4+R5 read-out); no code changes since certified.

## 1. Goal

Execute the AP-14 two-stage symmetric S1 hyperparameter sweep for SSRA-P1 and flat, producing the per-model (lr, dropout) selections that Phase 3 (S2 core pair) will use. Secondary deliverable: scale the data pipeline to the full Phase 2+3 token budget (Task A, dedicated cheap CPU pod — Daniel's decision 2026-07-13). Output: `results/M2-sweep.md` + committed configs/logs/plots + `results/runs.md` rows. **No S2 runs, no Phase 2b, no G1 work in this assignment.**

## 2. Task A — data scale-up (dedicated CPU pod; must complete and be verified in GCS BEFORE the Task B GPU pod deploy)

Current packed shards (Phase 0) hold 6.53M train tokens — insufficient. Scale the existing pipeline, changing nothing methodological:

- **Tokenizer is frozen:** reuse the Phase 0 artifact (byte-level BPE, vocab 16,384, sha256 `019568a2…`). Retraining the tokenizer is an anti-goal.
- Same corpus + provenance rules (AP-9): `HuggingFaceFW/fineweb-edu` config `sample-10BT`; re-record hub revision + license + retrieval date at integration time (Pravidlo W).
- Same deterministic document-disjoint split: `sha1(doc.id) % 1000 < 50` → val.
- Target: packed train ≥ **900M tokens** (covers 850M S2 budget per D-log 2026-07-13 — planning is 850M, NOT 1.7B — plus sweep consumption and margin); packed val ≥ 5M tokens.
- **Fixed eval slice [ODHAD, veto]:** define `val-eval-2M` = first 2.0M tokens of packed val, deterministic, sha256 recorded in the report. This slice is the sweep selection metric input and the candidate G1 eval set for Phase 3 (Daniel confirms in the Phase 3 assignment).
- Upload shards to `gs://ssra-poc-ew3` with paths + hashes recorded.
- **Execution environment (Daniel's decision 2026-07-13):** a dedicated cheap **CPU-only pod** (`ssra-m2-data`), never a billed GPU pod and not the local machine. Pod flow mirrors the GPU launch: existing secret `gcp_ssra_runpod_sa` via env var `GCP_SA_KEY_B64` (AP-17), blocking sanity gate `gsutil ls gs://ssra-poc-ew3` before any work, thread limits per cgroup vCPU quota, AP-21 naming for all new GCS paths, terminate — not stop — on CC's explicit completion signal (AP-18 lifecycle rule). Container disk ≈ 60 GB, no network volume. Config target [ODHAD]: 16 vCPU / 32 GB RAM (min 8/16). **CPU hourly rate is console-only** — record the deploy-console rate + date in the ledger (Pravidlo W; per-second billing confirmed at docs.runpod.io/pods/pricing, retrieved 2026-07-13); if the console shows a Community/Secure split for CPU, record both. Projected cost < 1–1.5 EUR [ODHAD, 1–3 h wall-clock].

## 3. Run plan — sweep (Task B, on pod)

All cells: S1 config (d 384 / h 6 / L 10, ≈24M params), ctx 1024, **b16** (matches the S2 gate batch; grad-accum 1), bf16 autocast (AP-16), AdamW + cosine decay, warmup ≈ 1.5 % of steps, seed **1337** every cell (AP-13: 1 seed per cell). Identical token stream, document order, step count, batch schedule, precision, and eval protocol for both models — any asymmetry invalidates the sweep (M2-assignment §3).

**Token budget per run [ODHAD, veto]: 60M tokens** = 3,662 steps @ 16,384 tok/step. Rationale: 4 SSRA runs @ 27.1k tok/s ≈ 2.46 GPU-h + 4 flat runs ≈ 0.21 GPU-h ⇒ ≈ 2.7 GPU-h measured runtime — inside the AP-14 target of 2–4 GPU-h for the whole sweep.

| stage | cells per model | runs |
|---|---|---|
| 1 | lr ∈ {1e-3, 6e-4, 3e-4} @ dropout 0.0 | 3 flat + 3 SSRA |
| 2 | dropout 0.1 @ stage-1 winner lr (the 0.0 cell already exists) | ≤ 1 flat + ≤ 1 SSRA |

**Selection rule (mechanical, per model):** winner = min **final val loss on `val-eval-2M`** (fp32 accumulation, identical eval batching for both models). Stage-1 winner lr → stage 2; final per-model (lr, dropout) = min final val loss among {winner @ do 0.0, winner @ do 0.1}. Record both selections explicitly.

**Divergence handling:** NaN/inf, or final val loss above the initial value ⇒ run status DIVERGED — a valid sweep outcome, recorded, never retried or re-tuned. If all 3 SSRA stage-1 cells diverge: STOP, deliver diagnostics.

**Execution order:** pytest → flat stage 1 (cheap) → SSRA stage 1 → compute winners → stage 2 pair → final evals → uploads → terminate signal (AP-18: terminate, not stop).

**Monitoring (standing, informative, non-gating):** `p1_attn_entropy` + per-query participation logged at every log interval on all SSRA runs (D-log 2026-06-12 M1 closure; carried per HO-12 §4.7). Loss values carry no quality meaning beyond the selection rule (spec §16) — in particular, SSRA-vs-flat loss comparisons are NEVER evidence of anything in this phase.

**Throughput sanity (informative):** first SSRA run's steady-state tok/s should be within ≈ ±10 % of 27,079 (recal, same config class, new data shards). Larger deviation: record in the report, do not gate.

## 4. Pre-flight checklist (extends `M2-runpod-launch.md`; order binding)

- **Step 0 — AP-19 Community price, BEFORE deploy (3rd carry, HO-12 §4.3):** in the deploy console, capture the Community price for the selected GPU **before booking anything** — value + date into the report. If the flow shows Secure only (as on 2026-07-12 and 2026-07-13): record "not capturable" + date, no backfill (Pravidlo W). If captured: add an informative Secure-vs-Community cost line for the sweep; Secure stays the booked default unless Daniel decides otherwise in the console.
- HW ladder per AP-18 / D-log 2026-07-12: A100 PCIe 80 GB → A100 SXM → H100 PCIe; low-cost fallback rung stays admissible but is NOT recommended here — S1 b16 SSRA measured 18.56 GiB fits a 24 GB card only with ≈ 23 % headroom and breaks tok/s comparability with recalibration numbers. Recommendation: stay in the A100 80 GB class; Daniel decides live in the console.
- Credit check (≥ 1 h on-demand minimum + ≈ 5 EUR envelope); ECB EUR/USD fixed on top-up day — if no top-up, carry 1.1430 (fixing 2026-07-10) with a note.
- AP-17 secret flow exactly as recalibration (env var `GCP_SA_KEY_B64`, bootstrap `/proc/1/environ` fallback); **blocking sanity gate `gsutil ls gs://ssra-poc-ew3` before any billable work.**
- Thread limits: `OMP_NUM_THREADS = MKL_NUM_THREADS = floor(cgroup vCPU quota)` for every process (thread-thrash lesson).
- `pytest tests/` once on the box; expected 64 pass / 1 known fail (`test_loglinear_integration`, fla × transformers, Phase-4-only). Any OTHER failure: STOP.

## 5. Protocol additions (AP — new in this document, veto applies)

- **AP-21 Run identity (standing rule, all future phases):** every run YAML defines a globally unique `run_name`; log file = `logs/{run_name}.log`; GCS checkpoint dir = `gs://ssra-poc-ew3/m2/sweep/{run_name}/`. Naming here: `m2-sweep-{ssra|flat}-lr{1e3|6e4|3e4}-do{00|01}`. **Any re-execution of any config — including a committed one — gets a new `run_name` (suffix `-r1`, `-r2`, …) and therefore new log + GCS paths.** Committed log files and prior GCS objects are never overwritten. (Root cause: recalibration deviations §7.4 log clobber + §7.5 checkpoint overwrite.)
- **AP-22 D5 planning error bar:** all VRAM planning for configs without a direct measurement uses **D5 projection × 1.20** (systematic under-projection measured 2026-07-13: +20.0…+20.5 % at S1, +12.6 % at S2 b16). Launch gate: D5 × 1.20 must be ≤ 76 GiB on an 80 GB card, else STOP. In this sweep every cell sits at a measured point (S1 b16; dropout adds no material VRAM) — the rule binds Phase 3+ planning and any deviation here.

## 6. Budget (AP-12; scoping of the 30 EUR cap)

- Projection [ODHAD]: Task A CPU pod < 1–1.5 EUR (console rate of the day, per-second billing) + Task B ≈ 2.7 GPU-h runs + ≈ 0.5 h overhead ≈ 3.2 pod-h ≈ $4.8 ≈ **4.2 EUR** @ SXM $1.497/hr total ⇒ Phase 2 whole ≈ **5–6 EUR**. Every run row in `results/runs.md` with machine, wall-clock, EUR estimate; billed console total filled post-terminate (console authoritative).
- Every individual run is far below the AP-12 single-run 25 EUR gate; cumulative spend (3.89 EUR pre-sweep ≈ 1.3 %) stays far from the 50/80 % thresholds.
- **The pre-approved 30 EUR cap (D-log 2026-07-13) is scoped EXCLUSIVELY to the Phase 3 S2 850M SSRA run. It does not apply to any Phase 2 run.** All Phase 2 work is governed by unchanged AP-12.

## 7. Reporting (mandatory) — `results/M2-sweep.md`

(i) pre-flight record incl. AP-19 step-0 outcome; (ii) environment snapshot (GPU, driver, torch, commit) + pytest result; (iii) data scale-up provenance: hub revision, license, retrieval date, shard sizes, tokenizer sha (unchanged), `val-eval-2M` sha256, CPU pod rate + billed total; (iv) run table — all 8 runs with `run_name`, config commit, status, tok/s, peak GiB, final val loss, EUR; (v) **selection table: per-model (lr, dropout) winners** + one-line mechanical justification; (vi) committed loss-curve plots (train + val, all cells); (vii) `p1_attn_entropy` + participation summary (informative); (viii) cost ledger vs 300 EUR; (ix) deviations — all explicit, none silent; (x) open questions with proposed D-log entries. Plus `results/runs.md` rows and raw logs in `logs/` + GCS.

## 8. Anti-goals

No S2 runs and no Phase 2b (Daniel decides after the sweep report). No tokenizer retraining. No quality or architecture conclusions from any loss (spec §16) — selection is mechanical and within-model only. No asymmetric treatment of any kind. No config edits after commit; a changed config is a new run under AP-21. No OOM retries, no allocator tweaks, no contingency flags (`summary_pos_override`, `pool_own_proj`, `p1_diversity_loss`). No edits to `docs/*` or `paper/*`. No prices from memory (Pravidlo W). No spend without a ledger row. No runs without a pre-committed YAML.

## 9. Definition of done

Task A shards ≥ 900M train tokens in GCS with full provenance and frozen tokenizer, produced on the dedicated CPU pod, pod terminated with billed total recorded ✔ · pre-flight incl. AP-19 step 0 recorded ✔ · AP-17 gate PASSED before billable work on every pod ✔ · pytest known-failure-only ✔ · 6 stage-1 + 2 stage-2 runs completed or DIVERGED-recorded, all under AP-21 naming ✔ · per-model (lr, dropout) selections in `results/M2-sweep.md` ✔ · plots + ledger + runs.md rows committed ✔ · GPU pod terminated on explicit signal, billed total filled post-terminate ✔ · no Phase 3 work ✔. Phase 3 (S2 core pair, 30 EUR scoped cap) starts only from a new assignment after Daniel reviews the sweep report.
