# Zadanie pre CC — M2 Phase 3: S2 core pair (Gate G1 inputs)

**Version:** v1 (2026-07-14) · **Milestone:** M2, Phase 3 per `docs/cc/M2-assignment.md` v1.1 §4 · **Mode:** veto-based — AP items introduced here stand unless Daniel vetoes them. AP-1…AP-22 remain in force where applicable (incl. the AP-19 amendment and the billing read-out rule, D-log 2026-07-14).

**Authority chain (binding):** `docs/spec.md` v1.2 > `docs/cc/M2-assignment.md` v1.1 > `docs/cc/M2-runpod-launch.md` v1 (launch/lifecycle) > this document. Project state: `docs/00` D-log (2026-07-14: Phase 2 closed, sweep selections final). Closed decisions (D1–D6, Q1–Q5, MD-1…MD-13, AP-1…AP-22, gates G0/G1a/G1b-D3, sweep selections §B.4) must not be reopened. On ambiguity or conflict: STOP, write the question + a proposed D-log entry into the report. Never improvise design changes.

**Context anchors (measured — do not re-derive, do not re-measure as a goal):**
- SSRA-P1 S2 b16 = **12,335 tok/s / 41.21 GiB peak** (recalibration, A100 SXM, `results/M2-recalibration.md`).
- flat S2 b16 = 129–140k tok/s / ≤ 40 GiB (Phase 1 calibration, A100 PCIe); SXM ≈ +6 % [ODHAD].
- Sweep selections (`results/M2-sweep.md` §B.4, mechanical, final): **flat (lr 1e-3, dropout 0.0)** · **SSRA (lr 1e-3, dropout 0.0)**.
- Model code `9417399` (R1+R4+R5 read-out) unchanged since AP-20 certification; harness as of `482bdb5` (sha256 hard gates, distinct `eval_bin`, deterministic `final_eval`, `--dry-run`).
- Data (final, frozen): `gs://ssra-poc-ew3/m2/data/m2-data-900m/` — train 913,605,620 tok (`6d0e47cd…`), val 48,050,671 (`03e0dd1a…`), val-eval-2M 2,000,000 (`bde526d2…`), tokenizer frozen (`019568a2…`). Full sha256 values: committed Task A manifest `results/M2-data-900m-manifest.json` (copy into configs verbatim, as in the sweep).

## 1. Goal

Train the **S2 core pair** — SSRA-P1 vs flat Transformer — under matched-parameters + matched-tokens discipline (AP-8): identical token stream, document order, step count, batch schedule, precision, eval protocol; seed 1337 (AP-13); selected hyperparameters (lr 1e-3, dropout 0.0) for both models. Token budget **850M** (AP-12 floor per D-log 2026-07-13; 1.7B is out of scope). Produce the **Gate G1 inputs**: final val ppl @ ctx 1024 for both models on the fixed eval set + stability evidence. **G1 verdict (stable training AND val ppl within ±5 % of flat) is Daniel's decision** — CC reports numbers and a recommendation only. On a G1-fail signal: STOP, deliver diagnostics; no autonomous redesign.

**G1 eval set — CONFIRMED by this assignment (closes the Task A open point):** `val-eval-2M` (first 2.0M tokens of packed val, sha256 `bde526d2…`) is the G1 eval set. G1 metric = **ppl = exp(final_eval_loss)**, where `final_eval_loss` is the deterministic full-pass token-weighted mean CE (nats/token) exactly as defined and implemented for the sweep (`results/M2-sweep.md` §B.0: non-overlapping windows seq_len+1 @ stride seq_len, bf16 forward + fp32 accumulation, batch 16, identical code path for both models). No re-derivation, no new eval code.

## 2. Run definitions

Two runs, each with a YAML in `experiments/` committed BEFORE launch + a `results/runs.md` row (run discipline). AP-21 names:

- `m2-core-flat-s2-850m`
- `m2-core-ssra-s2-850m`

GCS checkpoint dir = `gs://ssra-poc-ew3/m2/core/{run_name}/` (AP-21 naming extended to the `m2/core/` prefix); cross-run plots → `gs://ssra-poc-ew3/m2/core/plots/` (explicit this time — sweep deviation B.8-4 precedent).

**Config (identical except `arch` and AP-21 name/paths; generate from one template, diff-verify):**
S2 = d 640 / h 10 / L 15 (≈ 84M params, tied emb, vocab 16,384, `n_max: 32768`) · ctx 1024 · **b16, grad-accum 1** · bf16 autocast (AP-16) · AdamW + cosine decay, `lr_min_frac` 0.1 · **lr 1e-3, dropout 0.0** (sweep selections, both models) · warmup **778 steps** (= same ≈ 1.5 % fraction as the sweep, scaled to this run length) · seed **1337** · **51,880 steps = 850,001,920 tok** ≥ 850M (verify params + total tokens via `--dry-run` before commit) · SSRA = P1 defaults, no contingency flags (`summary_pos_override`, `pool_own_proj`, `p1_diversity_loss` all off).

**Intervals [proposal, veto]:** `val_every` 200 (fixed-seed val batches, `val_batches` 8, on val.bin — loss-curve only), `log_every` 25 (P-C diagnostics on SSRA: `p1_attn_entropy` + per-query participation, standing rule), `ckpt_every` **1,000** (≈ 22 min SSRA wall-clock — inside the AP-11 ≤ 30 min bound; S2 checkpoint ≈ 1 GB [ODHAD, fp32 params + Adam moments] ⇒ 1,000 halves upload overhead vs the sweep's 500 with no AP-11 violation).

**Harness gates:** the four sha256 hard gates (train/val/val-eval-2M/tokenizer) in both configs, values verbatim from the committed manifest; `data.eval_bin` = `val-eval-2M.bin` as a distinct shard. `scripts/data_scale.py` is NOT run — data is final.

## 3. Execution order + in-flight gates (order binding)

1. `pytest tests/` on the box — expected 64 pass / 1 known fail (`test_loglinear_integration`, box text may vary, §B.2 precedent). Any OTHER failure: STOP.
2. **flat run** (`m2-core-flat-s2-850m`) first: cheap (≈ 1.6–1.8 h [ODHAD]), doubles as the environment anchor and produces the G1 denominator early.
3. **SSRA run** (`m2-core-ssra-s2-850m`) with the **early cost gate (hard, scoped-cap enforcement):** at steps ≈ 1,000–1,500 (post-warmup) measure steady-state tok/s over ≥ 200 steps; projected run cost = (total_tokens / steady_tok_s) × booked hourly rate / 3600, converted at the carried ECB rate. **If projected total run cost > 30 EUR ⇒ STOP: abort the run, record status ABORTED-cost-gate, upload logs, report.** Equivalent threshold at SXM $1.497/hr + ECB 1.1430: steady-state < ≈ 10.3k tok/s. Informative (non-gating): sanity vs the 12,335 tok/s anchor, ±10 %.
4. Immediately after the early gate passes: **post the run ETA (UTC + Bratislava)** into the report/session output (terminate-window input, §4 last item).
5. Final evals (`final_eval` full pass on val-eval-2M, both runs) → loss-curve plots → uploads to GCS → commit + push results → **explicit completion signal** → **self-terminate (AP-23, §5)**.

**Divergence:** NaN/inf ⇒ abort the run immediately (no burn-through of the remaining budget), status DIVERGED — a valid outcome, never retried or re-tuned. STOP + diagnostics; any next step is Daniel's.

**Preemption/crash mid-run:** AP-11 resume (`--resume` from the run's GCS `latest.pt`) **continues the SAME `run_name`** — a continuation is not a re-execution. A fresh restart from step 0 IS a re-execution ⇒ `-r1` (AP-21). Preemption events + resume points logged per run.

## 4. Pre-flight checklist (extends `M2-runpod-launch.md`; order binding)

- **Step 0 — AP-19 (as amended 2026-07-14):** attempt to capture the Community price for the selected GPU BEFORE booking; record value or "not capturable" + date, no backfill. The Secure-vs-Community comparison is NOT an expected deliverable while the tier stays absent from the deploy flow.
- **HW:** A100 80 GB class only — 41.21 GiB peak rules out 24 GB cards entirely; anchors are SXM. A100 PCIe admissible if available (cheaper: projection 23.28 EUR @ $1.39; the recal projection base). Rate re-check in the console on deploy day (Pravidlo W); the early cost gate threshold is recomputed at the booked rate before launch.
- **Credit check: ≥ $40 headroom** (pod-session projection §6 + on-demand minimum) — a pod without network volume terminates at balance 0 mid-run (AP-11 covers the loss, but do not invite it).
- ECB 1.1430 carried unless a top-up happens (then fix the new ECB rate, record it).
- **AP-17** secret flow exactly as prior pods (`GCP_SA_KEY_B64`, `/proc/1/environ` fallback); **blocking sanity gate `gsutil ls gs://ssra-poc-ew3` before any billable work.**
- Thread limits: `OMP_NUM_THREADS = MKL_NUM_THREADS = floor(cgroup vCPU quota)` for every process.
- Repo delivery: git bundle at the launch commit (unauthenticated HTTPS clone fails — known); any commit executed on the pod is pushed to origin BEFORE the run it governs starts.
- **AP-23 verification (zero cost, in bootstrap):** confirm `runpodctl` presence + pod-scoped API key + `RUNPOD_POD_ID` (via `/proc/1/environ` if absent in the SSH session env — known RunPod behavior). If verification fails: record it; the manual terminate window (below) becomes the primary path.
- **Terminate window (idle pattern, 3rd occurrence in M2 — this item is mandatory):** BEFORE deploy, CC states the projected full pod-session wall-clock (§6) so Daniel can pick a deploy time whose completion ETA lands inside his console availability; the in-flight ETA post (§3.4) refines it. AP-23 self-terminate is the primary idle eliminator; the pre-agreed manual window is the fallback.

## 5. Protocol addition (AP — new in this document, veto applies)

- **AP-23 Pod self-termination on completion (extends AP-18):** the last action of a completed pod session is `runpodctl remove pod $RUNPOD_POD_ID`, executed from the pod itself. Basis [OVERENÉ, Pravidlo W]: every RunPod pod ships with `runpodctl` preinstalled and a pod-scoped API key (github.com/runpod/runpodctl README, retrieved 2026-07-14); `runpodctl remove pod [podId]` terminates the pod (docs.runpod.io/runpodctl/reference/runpodctl-remove-pod, retrieved 2026-07-14). **Strict sequence:** all GCS uploads verified (listing check) → results committed AND pushed to origin (confirmed) → explicit completion signal sent → only then self-terminate. Never self-terminate with unpushed results or unverified uploads. If the command fails or verification (§4) failed: fall back to AP-18 manual terminate in the pre-agreed window, record the failure verbatim. Billing read-out per the D-log 2026-07-14 rule: console value is final ≥ 2 h after termination; later corrections append-only.

## 6. Budget (AP-12 + the scoped 30 EUR cap)

| item | wall [ODHAD] | ≈ EUR @ SXM $1.497/hr, ECB 1.1430 |
|---|---|---|
| flat S2 850M | 1.6–1.8 h | 2.2–2.4 |
| SSRA S2 850M | ≈ 19.1 h @ 12,335 tok/s | ≈ **25.1** (23.3 @ PCIe $1.39) |
| bootstrap + pytest + final evals + uploads | ≈ 0.5 h | ≈ 0.7 |
| **pod session total** | **≈ 21.5 h** | **≈ 28–29** |

- **The pre-approved 30 EUR cap (D-log 2026-07-13) applies EXCLUSIVELY to the `m2-core-ssra-s2-850m` run** and is hard — the §3 early cost gate enforces it in-flight; the AP-12 single-run 25 EUR gate is waived for this one run only.
- The flat run and all overhead sit under unchanged AP-12 (each far below 25 EUR).
- Cumulative after Phase 3 ≈ 11.75 + 28.5 ≈ **40 EUR ≈ 13 %** of 300 — the 50/80 % thresholds are not approached.
- Ledger: `results/runs.md` row per run pre-launch; billed console total per the 2 h rule (append-only corrections); any decomposition estimate flagged [ODHAD], console total authoritative.

## 7. Reporting (mandatory) — `results/M2-core-pair.md`

(i) pre-flight record incl. AP-19 step 0 + AP-23 verification outcome; (ii) environment snapshot (GPU, driver, torch, commit) + pytest result; (iii) run table — `run_name`, config commit, status, tok/s, peak GiB, final_eval_loss, wall, EUR; (iv) **G1 input table: val ppl @ ctx 1024 pair (ppl = exp(final_eval_loss) on val-eval-2M), relative gap in %, stability evidence (full loss curves, explicit no-divergence statement) + CC recommendation — the verdict itself is Daniel's**; (v) committed loss-curve plots (train + val, both runs); (vi) P-C summary — does `p1_attn_entropy` ≈ ln(32) uniformity persist at 850M tokens? (informative, non-gating); (vii) cost ledger vs 300 EUR + explicit scoped-cap accounting for the SSRA run; (viii) **M3 handoff: final checkpoint GCS paths + config hashes for both models**; (ix) deviations — all explicit, none silent; (x) open questions, each with a proposed D-log entry. Plus `results/runs.md` rows and raw logs in `logs/` + GCS.

## 8. Anti-goals

No Phase 4 baselines ((b)/(c) are a separate assignment on remaining budget). No M3 evaluations (needle, multi-needle, length extrapolation, per-position loss) — checkpoint collection only. No deviation from the selected hyperparameters and no re-tuning on any mid-run signal. No OOM or divergence retries. **No quality or architecture conclusions beyond the G1 metric (spec §16): the SSRA-vs-flat final ppl gap is the G1 INPUT; its interpretation is Daniel's verdict, not CC's.** No asymmetric treatment of any kind. No config edits after commit (AP-21). No contingency flags. No edits to `docs/*` or `paper/*`. No prices from memory (Pravidlo W). No runs without a pre-committed YAML. No spend without a ledger row. No autonomous redesign on a G1-fail signal — STOP + diagnostics only.

## 9. Definition of done

Both configs committed pre-launch with dry-run-verified params + token totals ✔ · AP-17 gate PASSED before billable work ✔ · pytest known-fail-only ✔ · flat DONE ✔ · SSRA DONE, or DIVERGED / ABORTED-cost-gate recorded ✔ · `final_eval` on val-eval-2M for both models, G1 input table in `results/M2-core-pair.md` ✔ · plots + ledger + runs.md rows committed ✔ · final checkpoints archived in GCS with recorded paths + config hashes ✔ · pod self-terminated (AP-23) or manually terminated in the pre-agreed window, billed total per the 2 h rule ✔ · `m2-core-ssra-s2-850m` spend ≤ 30 EUR ✔ · no Phase 4 / M3 work ✔. Gate G1 verdict + D-log entries then happen on Daniel's side; interpretation against predictions (P-A…P-C) happens in the Claude.ai project.
