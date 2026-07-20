# Assignment for CC: V11 K1-analysis (T-A…T-E + C-T1 verdict)

**Version:** v1 (2026-07-20). **Budget:** 0 EUR, local CPU only.
**Authority chain:** spec v1.2 > `docs/cc/V11-data-exploitation.md` v1 (§5
pre-registered metrics and criteria — GOVERNING, unchanged) > this doc.
**Approval:** veto regime — handover to CC = approved.
**Role of this doc:** execution supplement only. It operationalizes §5 for
the now-known NPZ inputs (checkpoints are deleted; the NPZ extracts are the
only trajectory data). It adds NO new metrics and changes NO thresholds.
Every rule marked [OP] below is an operationalization fixed here BEFORE any
result is computed; disclose the [OP] list verbatim in the report section.

## 0. Context

K1 extraction closed with oversight PASS (D-log 2026-07-20); the Phase 3b
core-pair checkpoints were deleted afterwards. This is the last open V11
step: compute T-A…T-E and the C-T1 verdict from the committed NPZ extracts,
append §K1-analysis to the report, hand over to oversight review, V11 close.

## 1. Inputs (verified 2026-07-20; assert, do not re-derive)

- `results/v11/v11-k1-extract-ssra.npz`, `results/v11/v11-k1-extract-flat.npz`.
- Common arrays: `steps` (52: 1000…51000 stride 1000, then 51880),
  `param_names` (P), `numel` (P), `l2` (52,P), `delta_ref_l2` (52,P),
  `cos_ref` (52,P), `upd_l2` (51,P), `timings_s`, `init_l2` (P), `meta_json`.
  P = 393 (ssra) / 183 (flat). Reference for `*_ref` = **S_min = step-1000**
  (row 0; `delta_ref_l2[0] == 0.0` exactly — assert).
- ssra only: `full/<key>` (52,…) and `full_init/<key>` for the 60 phi/e_l
  state-dict keys (15× `layers.{i}.pool.latent_q`, 30× `pool.ln_pool.{weight,
  bias}`, 15× `layers.{i}.level_emb`); schema per `scripts/v11_ckpt_extract.py`.
- `meta_json`: `alias_groups` (ssra 60 groups × 3 names, sorted; flat 0),
  `trainable` (alias-deduped `named_parameters()` list: ssra 273, flat 183),
  `s_min_step` = 1000, `init_validated` = false (both arms).

## 2. Binding analysis rules (D-log 2026-07-20 / HO-29 §2)

1. **Dedup by `alias_groups`:** canonical name = first member of each sorted
   group; non-aliased keys are their own canonical. Analysis population =
   unique canonical tensors ∩ `trainable`. **Assert |population| = 273
   (ssra) / 183 (flat); mismatch = STOP, report.** Per-key raw series stay
   untouched; all statistics (incl. the C-T1 median) run over TENSORS.
2. **Trainable filter** from `meta_json.trainable` — no other source.
3. **Reference exclusively S_min.** Init is DROPPED (pre-registered branch
   fired): no init-relative delta/cos anywhere in §K1-analysis. `init_l2` +
   `full_init/*` may be cited only when restating the recorded drop verdict.

## 3. Metrics (per §5 of the governing assignment)

**T-A (both arms):** per-tensor `l2` and `delta_ref_l2` trajectories vs
step over the population. Figures aggregate by a module-class mapping
(e.g. emb, attn, pool.latent_q, pool.ln, level_emb, ffn, ln, ln_f/head)
defined as an explicit regex table printed in the report. [OP] Class
aggregate across tensors = sqrt of the sum of squared per-tensor values
(norm of concatenation); per-tensor CSV keeps full fidelity.

**T-B + C-T1 (ssra):** phi latent queries and phi LayerNorm (minor series):
Frobenius norm, delta-to-S_min, cosine-to-S_min vs step, computed from
`full/*` tensors. **Cross-check gate:** the same three series derived from
the `l2`/`delta_ref_l2`/`cos_ref` columns must match the full-tensor path
(rel tol 1e-9); mismatch = STOP.
C-T1 exactly as pre-registered: rho(m) = `delta_ref_l2[-1, m]` /
`l2[0, m]` over the population; median over ALL population tensors
(latent_q included — they are trainable). [OP] Aggregation over the 15
per-layer latent_q tensors: *supported* iff max_i rho(latent_q_i) <
0.1 × median; *refuted* iff min_i rho(latent_q_i) ≥ median; otherwise
*inconclusive*. Report the per-layer rho table, the median, and the
mechanical verdict — no interpretation beyond the criterion. [OP] Guard:
any tensor with `l2[0, m] == 0` is excluded from the median and reported
verbatim (expected: none).

**T-C formal (ssra):** from `full/layers.{i}.level_emb` (52,16,d) and
`full_init/...`: per-row L2 norms; **rows 11–15 must equal 0.0 exactly**
(`== 0.0`, no tolerance) for all 15 layers, all 52 steps, and init. Any
nonzero value = flag, report verbatim with provenance. Figure: rows 0–10
norm trajectories. The oversight preliminary check does not substitute
this scripted deliverable.

**T-D (both arms; mandatory per Daniel):** relative update rate
r_k(m) = `upd_l2[k, m]` / `l2[k, m]`. [OP] Denominator = start-of-interval
norm (row k); x-axis = end step of the interval; the final 880-step
interval (51000→51880) is annotated as shorter than the 1000-step stride.
Overlay the lr schedule: primary source = the per-step lr field in the
committed run JSONL logs (verify the actual field name from the log,
cite it); fallback = reconstruction from the committed YAML +
`scripts/train.py` schedule code, with the source documented. No guessed
schedule shapes. Panels by the T-A module classes.

**T-E (flat control):** T-A and T-D for the flat arm through the identical
pipeline and plot layouts — no SSRA-specific styling differences beyond
the absent phi/e_l panels.

## 4. Deliverables

- Script(s) `scripts/v11_k1_analysis*.py` (prefix `v11_k1_`), headless,
  pinned inputs, one command reproduces all figures + CSVs.
- Figures `results/figures/v11/v11-k1-*.png` (AP-21 naming, no overwrites).
- Raw appendix `results/v11/v11-k1-rho.csv`: arm, canonical name, alias
  group members, numel, l2_ref, l2_final, delta_ref_final, rho,
  is_latent_q — full population, both arms (oversight recount input).
- Report: insert section `## §K1-analysis` into
  `results/V11-exploitation.md` between §K1 and §Deviations, containing:
  the [OP] list, module-class table, T-A/T-B/T-C/T-D/T-E figures with
  one-paragraph observational captions, the C-T1 per-layer table +
  mechanical verdict, and pointers to scripts/CSVs. Deviations (if any)
  append to §Deviations; §Ledger gets a 0-EUR line.

## 5. Acceptance criteria (gates)

1. Input asserts pass: steps list, P, alias-group count, |population|
   273/183, `s_min_step` 1000, `delta_ref_l2[0] == 0.0` exactly.
2. T-B cross-check gate passes (columns vs full tensors, rel tol 1e-9).
3. T-C exact-zero check passes (or the flag is reported verbatim).
4. C-T1 verdict stated mechanically with all inputs (per-layer rhos,
   median) visible in the report and reproducible from the CSV.
5. **G-V11-4:** zero diffs under `src/ssra/` and `baselines/` (verify via
   commit file lists). **G-V11-5:** section committed regardless of the
   verdict direction.
6. Every number in §K1-analysis traceable to a script + committed input.

## 6. Anti-goals

No forward passes, no gradients, no tuning, no GPU, no GCS access (all
inputs are in the repo). No architecture verdicts, no causal claims about
the G1 gap or the lr-stability finding (spec §16). No init-reference
analysis. No post-hoc metric additions or threshold changes; if a result
suggests a follow-up metric, note it as a proposal for Daniel — do not
compute-and-report it inside §K1-analysis.

## 7. Handover

§K1-analysis committed → oversight review (Claude, independent recount
from the NPZ/CSV) → V11 close (Daniel). Checkpoint deletion is already
done; nothing in this assignment touches GCS.
