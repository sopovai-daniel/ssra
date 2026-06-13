# Zadanie pre CC — M2: SSRA v2 training runs (S1/S2) on GCP

**Version:** v1 (2026-06-12) · **Milestone:** M2 (`docs/03-poc-plan.md`) · **Mode:** veto-based — items marked AP-x are protocol details introduced by this assignment and stand unless Daniel vetoes them. AP numbering continues from M1 (AP-1…AP-7); AP-1…AP-7 remain in force where applicable.

**Authority chain (binding):** `docs/spec.md` **v1.2** = single source of truth for WHAT the model is; this assignment schedules and operationalizes M2 — **on any conflict, spec wins**. Project state/decisions: `docs/00` D-log. Behavior rules: `CLAUDE.md`. Closed decisions (D1–D6, Q1–Q5, MD-1…MD-13, gates G0/G1a/G1b-D3) must not be reopened. If anything looks ambiguous, contradictory, or unimplementable: STOP, write the question + a proposed D-log entry into the report. Never improvise design changes.

**Environment (fixed by D-log 2026-06-12):** GCP project `ssra-poc` @ **europe-west3 only** (org policy; verified live), HW plan **1× NVIDIA L4 24 GB**, budget cap **300 EUR cumulative**. GPU quota is currently NOT granted (fallback in progress on Daniel's side) — Phase 0 below requires no GPU and starts immediately; Phases 1+ are blocked on quota.

## 1. Goal

Produce the Gate G1 inputs: train SSRA-P1 and the flat baseline at S2 scale under matched-parameters + matched-tokens discipline (AP-8), preceded by a calibration run and a symmetric S1 hyperparameter sweep; execute baselines (b)/(c) as budget allows; deliver `results/M2-report.md` with the G1 numbers, full cost ledger, and M3-ready checkpoints. **G1 verdict (per `03`: stable training AND val ppl @ ctx 1k within ±5 % of flat) is Daniel's decision** — CC reports numbers and a recommendation only. On a G1-fail signal: stop, deliver diagnostics; any redesign iteration (max 1 per `03`) is Daniel's call, never autonomous.

## 2. Deliverables

| # | deliverable | where |
|---|---|---|
| 1 | Data pipeline: FineWeb-Edu sample → trained tokenizer → packed token shards in a GCS bucket (europe-west3); deterministic, document-disjoint train/val split; full provenance (AP-9) | `scripts/`, GCS, report |
| 2 | GPU training harness: extends the M1 training loop; config-driven per spec §13; bf16 autocast (AP-16); checkpoint/resume to GCS, Spot-preemption-safe (AP-11); cost+throughput logging; P-C hooks (`p1_attn_entropy`, per-query participation) wired from M1 | `src/ssra/`, `scripts/` |
| 3 | Calibration report (first paid GPU step): measured tok/s + VRAM for S1 and S2-candidate configs (both models), GPU resume verification, EUR projection table for the whole run plan, GCE-Spot-vs-Vertex price comparison (AP-10), HW recommendation (stay L4 / H100 ew3 / w4 exception) | `results/M2-calibration.md` |
| 4 | S1 sweep: symmetric two-stage lr/dropout sweep for SSRA-P1 and flat (AP-14); per-model selections recorded | `experiments/`, `results/runs.md` |
| 5 | S2 core pair: SSRA-P1 vs flat, matched tokens, seed 1337 (AP-8, AP-13); loss curves + final val ppl @ ctx 1024 | `experiments/`, `results/`, GCS |
| 6 | Baselines per remaining budget (AP-15): (b) GatedDeltaNet (fla-org/flash-linear-attention v0.5.0, MIT — provenance fixed in M1) GPU execution; (c) MEGABYTE-style | `experiments/`, `results/` |
| 7 | `results/M2-report.md` (§7) + committed plots + complete cost ledger | `results/` |
| 8 | M3 handoff: final checkpoints of every trained model archived in GCS, paths + config hashes recorded in the report | GCS, report |

## 3. Binding constraints (reminders — spec text wins over this list)

- **Run discipline (CLAUDE.md):** 1 run = 1 YAML in `experiments/` committed BEFORE launch + a row in `results/runs.md`. A run without a committed config does not exist. This includes calibration and sweep runs.
- **Pool = P1 only in M2.** P2/P3/hybrid, sharing-off, w/m/k variations are M3 ablations. Defaults stand: m=16, w=64, k=2, level_emb on, summary_pos none, readout_params shared.
- **Symmetric treatment:** identical sweep grid, selection rule, token stream, batch schedule, precision, and eval protocol for SSRA and flat — any asymmetry invalidates G1.
- **Budget:** 300 EUR hard cap, launch gates per AP-12. Never drop the flat baseline to save budget (`03`: scale S2 down instead).
- **Pravidlo W:** no prices, quotas, or API behavior from memory — verify in official GCP docs/pricing pages on the day of use, record URL + retrieval date.
- **Secrets hygiene:** no credentials, keys, or tokens in the repo — ever. GCS paths and project IDs in configs are fine.
- Contingency flags stay off: `summary_pos_override`, `pool_own_proj`, `p1_diversity_loss > 0` (NotImplementedError stands). NoPE summary keys: watch and report (spec §6 [HYPOTÉZA]); never flip `summary_pos` yourself.
- fp32 correctness tests (spec §14.1–.3, .7) must pass on the GPU environment once before Phase 2 — run `pytest tests/` on the box, report.

## 4. Run plan (phases; each gated on the previous)

- **Phase 0 — no-GPU groundwork (starts now, quota-independent):** GCS bucket (ew3), environment image/pins, data pipeline end-to-end on a small shard, tokenizer trained, harness CPU smoke, checkpoint+resume unit test (AP-11).
- **Phase 1 — calibration (first paid step, target ≈ 1 GPU-hour):** measured tok/s + peak VRAM for S1 and S2-candidate configs (both models), one kill+resume verification on GPU, `pytest tests/` on the box, price check (AP-10), EUR projection for Phases 2–4. **STOP → Daniel confirms plan + HW before Phase 2.**
- **Phase 2 — S1 sweep (AP-14).** Optional **Phase 2b** (Daniel decides on calibration numbers): one S1 confirmation pair (SSRA+flat, chosen hyperparams, full or reduced S1 token budget) as a cheap G1 dress rehearsal before committing the most expensive runs.
- **Phase 3 — S2 core pair:** SSRA-P1 + flat, matched tokens (AP-8), seed 1337, chosen hyperparams.
- **Phase 4 — baselines per remaining budget (AP-15):** (b), then (c); explicit defer with budget evidence if it does not fit.

## 5. Proposed run configs [ODHAD — Phase 1 calibration confirms; veto applies]

| | S1 (tuning) | S2 (main comparison) |
|---|---|---|
| d / h / L | 384 / 6 / 10 | 640 / 10 / 15 |
| params (tied emb, 16k vocab) | ≈ 24M | ≈ 84M |
| train ctx | 1024 | 1024 |
| token budget | 500M (≈ 20×, Chinchilla heuristic per `03`) | 1.7B (≈ 20×) — AP-12 scaling applies |
| optimizer | AdamW, cosine decay, warmup 1–2 % of steps | ditto |
| batch × grad-accum | set by calibration (VRAM fit) | ditto |

Param arithmetic: 12·d²·L + vocab·d (tied). `n_max: 32768` (e_ℓ table headroom for M3 length extrapolation). `vocab: 16384` (AP-9). Eval: val loss on a fixed held-out token set at ctx 1024, identical batches for both models; **ppl = exp(mean CE nats/token)** — this number is the G1 input. Any token-budget reduction forced by AP-12 is symmetric and pre-declared in the configs before launch.

## 6. Assignment-level protocol (AP — new in this document, veto applies)

- **AP-8 Matched compute, G1 operationalization:** matched **parameters** (same d/h/L; SSRA overhead < 1 % per spec §12) + matched **data** (identical tokenized stream, document order, step count, batch schedule, seed 1337). Analytic FLOPs/token estimates and measured wall-clock + tok/s are **reported for both models but NOT matched**; this accounting choice is stated in the report and carried into the paper (spec §10 honesty note). Rationale: isolates mechanism quality; efficiency scaling was measured separately (G1a).
- **AP-9 Data + tokenizer:** corpus = **FineWeb-Edu** (`HuggingFaceFW/fineweb-edu`) sample subset sized for the largest token budget + val; license and dataset version verified at integration time (Pravidlo W; record URL + date). Tokenizer = byte-level BPE, **vocab 16,384**, trained on a corpus sample document-disjoint from the val split; record tokenizer training sample size + artifact hash. Val split deterministic and document-disjoint from train. Fallback if FineWeb-Edu access fails: SlimPajama sample (record the reason).
- **AP-10 GCE Spot vs Vertex custom training:** decided at calibration day on verified prices — fetch current L4 (and H100 @ ew3) pricing from official GCP pages, record URL + retrieval date, project EUR for the full run plan under both options, recommend; **Daniel decides**. Default lean: GCE Spot + own checkpoint/resume (no Vertex premium, Vertex AI API stays disabled) unless the evidence says otherwise.
- **AP-11 Checkpoint/resume:** checkpoints to GCS every ≤ 30 min wall-clock and at run end; resume must produce a continuous loss curve (unit-tested in Phase 0, verified once by kill+resume on GPU in Phase 1). Max acceptable loss on Spot preemption = 1 checkpoint interval. Preemption events + resume points logged per run.
- **AP-12 Cost ledger + launch gates:** every run row in `results/runs.md` carries machine type, wall-clock, and measured (billing) or estimated EUR; cumulative spend tracked against 300 EUR in every report section. **STOP and get Daniel's explicit OK before:** any single run projected > 25 EUR, and any launch that would push cumulative spend past 50 % or 80 % of the cap. If the ≈ 20× token budgets do not fit the cap, scale them down symmetrically (S2 floor: 10× ≈ 850M tokens) and/or defer Phase 4 — never drop the flat baseline.
- **AP-13 Seeds:** single seed **1337** for the S2 core pair (budget-bound); flagged as a limitation in the report. S1 sweep: 1 seed per cell.
- **AP-14 S1 sweep (two-stage, symmetric):** stage 1 = lr ∈ {1e-3, 6e-4, 3e-4} at dropout 0.0 (3 short runs per model); stage 2 = dropout ∈ {0.0, 0.1} at the stage-1 winner (≤ 1 extra run per model). Short-run token budget set in Phase 1 from measured tok/s, target ≈ 2–4 GPU-hours for the whole sweep. Selection = final val loss; chosen (lr, dropout) per model carried to S2. (D5: dropout strength is tuned here.)
- **AP-15 Baseline order in Phase 4:** (b) GatedDeltaNet at S2 scale, matched tokens, first (sharpest M3 comparison); then (c) MEGABYTE-style. If the remaining budget is below projection: run at S1 scale or defer to an M3 budget decision — never silently skip (mirror of AP-5). Baseline configs follow the same run discipline and parameter-match reporting.
- **AP-16 Training precision:** bf16 autocast on GPU for all M2 training runs, identical for SSRA, flat, and baselines; loss/eval accumulation in fp32. Correctness-test tolerances remain defined in fp32 (MD-7) and are judged on the §3 `pytest` pass. Any bf16 instability: report it; fall back to fp32 training only with Daniel's OK (cost impact must be projected).

## 7. Reporting (mandatory)

`results/M2-report.md` must contain: (i) run table — all runs, config commit, status, cost; (ii) **G1 input table**: S2 val ppl pair, relative gap, stability evidence (full loss curves, no-divergence statement) + CC recommendation; (iii) committed loss-curve plots; (iv) calibration findings + the AP-10 price comparison + HW decision outcome; (v) S1 sweep table + per-model selections; (vi) cost ledger summary vs 300 EUR; (vii) environment snapshot (image, driver/CUDA, torch, commit hash); (viii) data/tokenizer provenance (AP-9) + baseline (b) GPU execution evidence; (ix) open questions / spec ambiguities, each with a proposed D-log entry; (x) GCS checkpoint paths for M3. Plus: one row per run in `results/runs.md`; raw logs under `logs/` or a recorded GCS path.

## 8. Anti-goals (spec §16 + M2 scope)

No M3 evaluations (needle/multi-needle, length extrapolation, per-position loss) — only checkpoint collection for them; no ablations of any axis (§15); no design or spec changes; **no edits to `docs/*` or `paper/*`** (report conflicts instead); no asymmetric tuning or eval; no runs without a pre-committed YAML; no spend without a ledger entry; no prices or quota claims from memory; no Vertex AI API enablement before the AP-10 decision; no quality conclusions beyond the G1 metric (in particular none from S1 sweep losses); no contingency flags enabled silently; k=4 stays stubbed (AP-7 — required before M3 ablation (g), not an M2 task).

## 9. Definition of done

Phase 0 complete; Phase 1 calibration report delivered and plan confirmed by Daniel; S1 sweep done with selections recorded; S2 core pair trained to the declared (possibly AP-12-scaled) token budget with G1 inputs in `results/M2-report.md`; Phase 4 executed or explicitly deferred with budget evidence; all runs ledgered with pre-committed configs; checkpoints archived in GCS with recorded paths; cumulative spend ≤ 300 EUR documented. Gate G1 verdict and D-log entries then happen on Daniel's side; interpretation against predictions (P-A…P-C) happens in the Claude.ai project.
