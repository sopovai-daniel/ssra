# CC Assignment: M2 spike diagnostics (local, zero spend)

**Version:** v1 (2026-07-15) · **Author:** Claude (Fable 5), approved by Daniel
**Authority chain:** `docs/spec.md` v1.2 > `docs/cc/M2-assignment.md` v1.1 > this doc
**Inputs (binding):** `results/M2-core-pair.md` incl. §xi corrections; `logs/m2-core-ssra-s2-850m.log`; `logs/m2-core-flat-s2-850m.log`; GCS run dir `gs://ssra-poc-ew3/m2/core/m2-core-ssra-s2-850m/`

## §1 Goal and boundaries

Localize the `m2-core-ssra-s2-850m` training instability (primary spike steps
6,475→6,500; secondary episode 16,675–22,100 per report §xi C1) from already
existing artifacts — checkpoints, logs, data. Output = an **informative
diagnostics report** that serves as input for (a) Daniel's G1 verdict and
(b) the decision on the single permitted retune iteration per
`docs/03-poc-plan.md`.

Hard boundaries:

- **No architecture conclusions** (spec §16). Localization ≠ mechanism claims.
  Ranked tables + observations only; interpretation belongs to the design
  analysis and to Daniel.
- **No model/training code changes.** Read-only analysis scripts only, placed
  under `scripts/diagnostics/` (new directory). AP-20 must not be activated;
  model code stays on the `9417399`-certified lineage untouched.
- **Zero GPU spend, zero GCS writes** (read-only `gsutil ls` / `gsutil cp`).
- **No retraining, no re-tuning, no config edits, no fixes.** Diagnosis only.
- `results/runs.md` untouched (no run happens here).
- Ambiguity rule: if anything material is ambiguous, STOP and surface to
  Daniel rather than guess.

## §2 Binding facts (from report §iv/§vi/§xi — do NOT re-measure)

- Primary spike: train 3.9703 → 7.45049 across steps 6,475→6,500 (largest
  25-step jump in the run, +3.4802); lr 9.73e-4; grad-clip 1.0 active; no
  coincident val/ckpt boundary; **no NaN/inf anywhere in the run**.
- Best val 3.92983 @ step 6,400; first post-spike val 7.39642 @ 6,600.
- Secondary episode: 155 train records > 9 nats within steps 16,675–22,100
  (71 % density, ~5 sub-blocks), train max 10.351 @ 19,775, val max
  10.088 @ 19,800; last val > 8.0 @ 25,800; tight 7.52–7.59 band from ~35k.
- P-C: `p1_attn_entropy` ≈ ln(32) uniform THROUGH the spike (3.4645 @ 6,500),
  de-uniformization only afterwards — symptom, not precursor.
- Gradient norms are NOT in the logs (§xi C3). Adam moments inside the AP-11
  checkpoints are the only gradient-scale source.
- Flat arm on the identical token stream: mild elevation in the same window
  (z ≈ +2.3 @ 6,525), shown chance-consistent by whole-run scan (§xi C2) —
  data-window cause is a LOW-expectation hypothesis, checked as due diligence.

## §3 Tasks (in order)

**T0 — Checkpoint availability gate.**
`gsutil ls -l gs://ssra-poc-ew3/m2/core/m2-core-ssra-s2-850m/` and record the
listing verbatim in the report. Required for T1–T4: step-5000, step-6000,
step-7000 checkpoints (ckpt_every 1000). If per-step checkpoints are absent
(e.g. only `latest.pt` retained): record the fact, mark T1–T4 NOT EXECUTABLE,
and proceed with T5 only. Do not improvise substitutes.

**T1 — Local staging (outside git history).**
Download ckpt 5000 / 6000 / 7000 (~3 GB total) plus `train.bin` if T5 needs it
(1.83 GB). Destination: prefer `checkpoints/diag/` if
`git check-ignore checkpoints/` confirms it is ignored; otherwise
`~/ssra-scratch/m2-diag/`. Checkpoints and shards must NOT be committed.
Load with `torch.load(path, map_location="cpu")`; if torch 2.12's
`weights_only` default rejects the optimizer payload, `weights_only=False` is
acceptable (trusted self-produced artifact — record the flag used).

**T2 — Weight-delta localization.**
Per-module relative deltas ‖ΔW‖_F / ‖W‖_F for the pairs **5000→6000
(pre-spike baseline)** and **6000→7000 (spike-crossing)**. Module granularity:
every named parameter, aggregated per logical block — shared attention θ
(Q/K/V/O per matrix), Pool_φ / Q_φ parameters, token embedding, unembedding,
all LN gains/biases, any scale/residual parameters. Deliverable: one table
ranked by the ratio (spike-crossing delta) / (baseline delta), full table in
an appendix or CSV artifact under `results/`.

**T3 — Adam-moment localization.**
At ckpt 6000 and 7000: per-module mean and max of `exp_avg_sq`, plus
‖`exp_avg`‖ per module; ranked table of 6000→7000 growth. Purpose: localize
where gradient energy exploded (substitute for the unlogged grad norms).

**T4 — Weight-health snapshot.**
Per checkpoint (5000/6000/7000) and per module: ‖W‖_F, max |w|, LN gain
min/max, unembedding row-norm distribution (report top-10 outlier rows +
summary stats), embedding norm. Flag any monotone blow-up already visible
5000→6000 (pre-spike drift) vs discontinuity only at 6000→7000.

**T5 — Data-window due diligence (low expectation, cheap).**
Reconstruct the exact token batches for steps 6,450–6,550 via the harness's
OWN deterministic data path (same config, seed 1337) — a small read-only
script importing the harness sampler; **no training step executed**. Report:
window-level stats (repetition rate, byte/token entropy vs corpus baseline,
unusually long or degenerate documents) + a few decoded samples described
qualitatively (no long verbatim dumps). If reconstruction requires touching
harness code rather than importing it: STOP and surface (do not modify).

**T6 — OPTIONAL, only if T2–T5 leave the localization ambiguous.**
Single CPU forward pass of ckpt-6000 on the reconstructed step-6,487–6,500
batches, logging per-layer activation stats (mean/max magnitude, logit
scale). Standalone script, no backward, no optimizer, no model-code edits.
Skip freely if T2–T4 already localize.

## §4 Acceptance criteria

1. `results/M2-spike-diagnostics.md` containing: T0 listing verbatim; T2 and
   T3 ranked tables; T4 snapshot table; T5 findings; every claim marked
   [OVERENÉ]/[HYPOTÉZA]/[ODHAD]; an explicit "what this does and does not
   establish" closing section with **no architecture conclusions and no
   redesign proposals**.
2. All new scripts under `scripts/diagnostics/`, read-only w.r.t. model and
   training code (AP-20 statement included in the report, same form as
   report §0.5).
3. Zero GPU spend; zero GCS writes; checkpoints/shards not committed;
   `results/runs.md` untouched.
4. Anything ambiguous surfaced to Daniel instead of guessed.

## §5 Anti-goals

No retraining, no hyperparameter experiments, no config edits, no spec or
test modifications, no "quick fixes", no quality conclusions from loss
values beyond localization of the instability, no conclusions about P1/P2/P3
ranking, no edits to committed logs or GCS objects (AP-21).
