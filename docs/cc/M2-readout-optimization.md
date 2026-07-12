# Zadanie pre CC — M2: read-out optimization (D-log 2026-07-12, option (a))

**Version:** v1 (2026-07-12) · **Milestone:** M2, between Phase 1 (closed) and re-calibration · **Mode:** veto-based — items marked AP-x stand unless Daniel vetoes them.

**Authority chain (binding):** `docs/spec.md` v1.2 = single source of truth for WHAT the computation means — **on any conflict, spec wins**; `docs/cc/M2-assignment.md` v1.1 = milestone contract (phases, gates, APs unchanged); this document operationalizes D-log 2026-07-12 decision (a). Project state: `docs/00` D-log. Behavior rules: `CLAUDE.md`. Closed decisions (D1–D6, Q1–Q5, MD-1…MD-13, AP-1…AP-19) must not be reopened. **This is a pure implementation change: spec §8 explicitly grants realization freedom — "any batched realization is acceptable iff it is logit-equivalent to the per-position definition"; the §14 tests are the referee.** If anything looks unimplementable without a normative change: STOP, write the question + proposed D-log entry into the report. Never improvise design changes.

**Context (read first):** `results/M2-calibration.md` (Phase 1 findings), `docs/handover/HO-10-…`, design analysis summary in §2 below.

## 1. Goal

Restructure the SSRA read-out implementation (`src/ssra/model.py::SSRALayer.readout` + window path) so that the batched training forward/backward no longer materializes per-token gathered key/value copies, with **identical logits** (spec §8/§14 semantics untouched).

**Measurable target (D-log 2026-07-12; verdict at re-calibration, not in this assignment):** SSRA S2 (d=640, h=10, L=15, ctx 1024) trains on 1× 80 GB at batch ≥ 16, **and** projected S2 cost @ 850M tokens ≤ 25 EUR (≈ ≥ 11.5k tok/s at S2 at $1.39/hr A100 PCIe, ECB 1.1430; full 1.7B budget ⇒ ≈ 10× current). CC's job here: make the analytic memory budget and CPU-verified implementation credible enough to spend the ~1–2 EUR re-calibration.

**Stop-loss (binding):** this assignment = the ONE implementation iteration granted by the D-log. One re-calibration follows and decides. Do not open a second redesign track inside this assignment; if the primary restructure (§3 R1) hits a wall, STOP and report.

## 2. Root-cause analysis (input, Claude.ai 2026-07-12 — binding context, not to re-derive)

Measured baseline (calibration, A100 80 GB, bf16 autocast, ctx 1024): SSRA-P1 S1 b16 = 9,457 tok/s / 54.67 GiB peak vs flat 300,978 tok/s / 6.35 GiB; SSRA S1 b32/b64 and all S2 batches OOM at `model.py:113–114`.

1. **[OVERENÉ arithmetically against the measured OOM]** `k_g = k_sum[:, :, idx]` and `v_g = v_sum[:, :, idx]` materialize two tensors `[B, h, N, k_max, d_h]`. For N = 1024, w = 64, m = 16 the Fenwick worst-case row count is **k_max = 95** (levels 0–3 contribute 1+2+4+8 = 15 rows; up to 5 blocks at levels ≥ 4 contribute 16 each). At S1 b32 (B=32, h=6, d_h=64, bf16): 32·6·1024·95·64·2 B = **2.226 GiB — exactly the failed allocation in the OOM trace.** Both tensors are saved by autograd for the einsum backwards, per layer, so all L layers' copies coexist during forward.
2. **The calibration red flag "cover ≈ 90–180 vs theory ≈ 40" is resolved — no implementation excess.** The HO-10 back-of-envelope used d_h = 24 (an M1-era throughput config, d=192/h=8) instead of the actual S1 d_h = 384/6 = 64 — root cause: stale dims reused. With correct d_h, measured cover = 95 = the exact theoretical worst-case **row** count. The earlier "≈ log₂N + w ≈ 40" estimate conflated Fenwick **blocks** (≈ log₂N ≈ 10) with **rows** (m per block above the lossless threshold). Spec's own bound |K_t| ≤ (w+1) + m·(⌊log₂t⌋+1) already says this. The implementation gathers exactly the spec §8 key set; the flaw is the **materialization strategy** (per-token padded copies), not the key set.
3. **Structural OOM at S2 b16 [OVERENÉ by arithmetic]:** gathers alone = 2 × [16,10,1024,95,64] bf16 = 3.71 GiB/layer × 15 layers = **55.7 GiB**, before window path, up-pass activations, logits.
4. **Padding waste:** `build_readout_index` pads every position to k_max = 95; the mean row count over positions is ≈ 50 ⇒ ≈ 1.9× redundant gather traffic and score compute on masked slots.
5. **[HYPOTÉZA — verify by profiling, §5 D2]** Window path: `einsum` over the `unfold` views (`k_win`, `v_win`) very likely forces `.contiguous()` copies `[B, h, N, d_h, w+1]` (overlapping strides are not bmm-consumable) ⇒ additional ≈ 2 × 0.76 GiB/layer at S1 b16, ≈ 2 × 1.27 GiB/layer at S2 b16 (≈ 38 GiB total at S2), also autograd-retained.
6. **Throughput [HYPOTÉZA]:** per-token score-op FLOPs of the read-out (~160 keys) are *below* flat's (~512 avg keys at N=1024), so the 32× gap is dominated by overhead: ~45 GiB/step of gather write+read traffic (S1 b32 scale), ~1.9× padding waste, 5-D einsums, and many small up-pass kernels. Removing the materialization attacks memory and speed with one change. No speed promise is made — re-calibration decides.

## 3. Required restructure

### R1 (primary, mandatory) — level-grouped summary attention, zero gather

**Structural lemma [OVERENÉ — direct corollary of the spec §9 v1.1 retention derivation, completion-test-certified in M1]:** a node u = (ℓ, j) with end e = j·2^ℓ appears in `fenwick_blocks(p)` **iff** lowbit(e) = 2^ℓ (⇔ j odd) **and** p ∈ [e, e + 2^ℓ − 1]. Each decomposition contains **at most one block per level**. Therefore the tokens consuming node (ℓ, j odd) in the read-out (t = p + w + 1) are exactly the **contiguous run t ∈ [j·2^ℓ + w + 1, (j+1)·2^ℓ + w]** of length 2^ℓ.

Consequence: per level ℓ, shift the token axis by w+1 (p = t − w − 1, positions p ≤ 0 have no summaries), partition p into blocks of 2^ℓ; block index g = ⌊p / 2^ℓ⌋; **odd g attends node (ℓ, g)'s s_ℓ rows; even g attends nothing at this level.** This is a perfectly regular block-local cross-attention per level:

- queries: `[B, h, G, 2^ℓ, d_h]` grouped views of q (edge blocks truncated/padded; per-level q copies are ≤ 12 MiB each at S1 b16 — acceptable),
- keys/values: `[B, h, G_odd, s_ℓ, d_h]` — slices of the already-projected `k_sum`/`v_sum` level segments (odd-node selection is a regular stride; a `.contiguous()` copy here is ≤ half the level's rows, i.e. tiny),
- per-level bmm → scores written into a per-token score buffer `[B, h, N, Σ_ℓ s_ℓ]` (≈ 111 slots at N=1024; inactive/even-g and p ≤ 0 slots masked −inf).

Then, unchanged §8 semantics: concatenate window scores + summary scores → **ONE softmax** → split probs → window AV + per-level grouped AV bmms (same grouping, probs slices) → sum → `w_o`. No `[·,·,N,k_max,d_h]` tensor exists anywhere in the training path. Skip levels with no consumers (e.g. the top level at N = 1024, since p ≤ N − w − 1). Ragged N works natively (runs truncate at N; every consumed node satisfies span ⊆ [1, N] per the §8 alignment lemma — no new materialization condition).

Implementation freedom on packing/layout is CC's, **equivalence is not** (§4, §6). Level 0 participates in the same scheme (s_0 = 1, runs of length 1, odd p) — rows come from `levels[0]`, e_0 key tag as today.

**`summary_pos: virtual` contingency (never enabled to date, must stay functional):** the current path rotates per-token key copies at vpos = t − w − 1, which conflicts with shared keys. Two admissible options — (i) keep a gathered fallback branch used **only** when `summary_pos == "virtual"` (default `none` takes the grouped path), or (ii) implement the constant-phase identity: with q rotated at t and keys at t − w − 1, the relative RoPE phase is the constant w + 1, so an equivalent realization rotates shared summary keys once at a fixed offset relative to the query rotation. If (ii) is chosen it needs its own unit test against the current virtual-mode output (fp32, atol 1e−5) — otherwise choose (i). Do not spend more than trivial effort here; (i) is fine.

### R4 (mandatory, co-scoped) — window path without materialized per-token K/V copies

Replace `unfold` + einsum with a block-local band realization: pad N to a multiple of w'; reshape tokens into blocks; each block attends its own + previous block's keys (`[B, h, N/w', w', 2w']` scores), mask to the exact window [t−w, t] ∩ [1, t] (MD-6), producing **raw scores that join the common softmax**. Memory: O(B·h·N·w) scores, no `[·,·,N,d_h,w+1]` K/V copies.

**Explicit prohibition:** `F.scaled_dot_product_attention` may NOT be used inside the read-out for the window part — SDPA performs its own softmax and would break the single-softmax-over-heterogeneous-keys semantics (§8). (SDPA elsewhere: see R5.)

### R2 (conditional knob) — selective recomputation

If the §5 analytic budget still projects S2 b16 total peak > ~60 GiB after R1+R4, add `torch.utils.checkpoint` around the read-out segment and/or per up-pass level (recompute in backward). Internal implementation detail — **no §13 config-surface addition**; record the choice + measured CPU/MPS effect in the report. Do not apply blindly if the budget already clears — checkpointing costs an extra forward.

### R3 (fallback of last resort) — token-chunked gather under checkpoint

Only if R1 hits an unforeseen wall (STOP + report first): chunk the read-out over tokens (chunks of N_c), gather per chunk under checkpoint. Trivially correct, bounded memory, but keeps padding waste and gather traffic — expect a much weaker throughput effect. Choosing R3 = an explicit deviation to record, not a silent swap.

### R5 (optional, strictly timeboxed) — node_attn via SDPA

`SharedAttention.node_attn` is a plain bidirectional softmax attention (RoPE pre-applied, dropout on probs) and IS SDPA-expressible. Allowed only if: §14 stays green on CPU fp32, the A/B test (§4) stays green, and the CPU microbench shows no regression. Skip entirely at the first sign of friction — R5 must not consume iteration budget.

## 4. Binding constraints

- `docs/spec.md` v1.2 UNCHANGED — no edits. Read-out semantics = §8 per-position definition; §6 positional treatment (RoPE window keys/query at absolute positions; NoPE + e_ℓ tag on summary **keys only**, values untagged, MD-5); window [t−w, t] inclusive (MD-6); ONE softmax over heterogeneous keys.
- Unchanged files/behavior: up-pass (`up_pass`, `node_step`), `pool.py` operators, `decode.py` (incremental decoder + retention rule), `fenwick.py` public API may be extended but `build_readout_index` must remain (the decoder and any fallback branch may use it), config surface §13 (no new keys), `checkpoint.py`, data path, seeds, AP-16 autocast policy, loss/optimizer.
- `tests/`: existing test files may **not** be modified; new test files may be added. The known-red `test_baselines.py::test_loglinear_integration` (fla × transformers, Phase-4-only) stays as is.
- Diagnostics preserved: `aux["capture_summaries"]`, P-C hooks (`p1_attn_entropy`), `lb_terms` — byte-identical behavior.
- `readout_params: separate` (ψ) and `level_emb: off` modes must work on the new path (projections are applied to the summary table before grouping — structure unchanged).
- Development is CPU-only (no paid GPU in this assignment); MPS numbers informative (AP-2 discipline carries over).
- **AP-20 (new, veto):** any performance rework of a spec-governed computation ships with (i) a frozen-reference A/B test — a verbatim copy of the replaced implementation kept test-local, compared to the new path on randomized inputs, (ii) an analytic peak-memory model evaluated at the target configs, and (iii) the unchanged spec §14 suite green. Applies to this assignment and any future one.

## 5. Deliverables

| # | deliverable | where |
|---|---|---|
| D1 | Restructured read-out per §3 (R1 + R4; R2 if budget demands; R3 only after STOP+report; R5 optional) | `src/ssra/` |
| D2 | Memory-behavior verification of claims §2.1/§2.5: a small profiling script (CPU allocator stats or MPS `current_allocated_memory`) demonstrating (a) the old path's gather + window materialization, (b) the new path's absence of both | `scripts/`, numbers in report |
| D3 | **Frozen-reference A/B test (AP-20):** old `readout` copied verbatim into the test module; new vs old logits on randomized configs — ragged N ∈ {257, 1000, 1024}, `level_emb` on/off, `readout_params` shared/separate, `summary_pos` none (+ virtual if identity path chosen), several (m, w) — fp32 atol 1e−5 | `tests/test_readout_equiv.py` |
| D4 | Full suite green: `pytest tests/` — all previously-green tests pass unmodified; §14.2 completion test (N ∈ {257, 1000}, atol 1e−4 fp32) is the batched-vs-incremental certificate and must pass against the **unchanged** decoder | `logs/` |
| D5 | **Analytic peak-memory model:** closed-form per-layer read-out residency (old vs new) as a function of (B, h, N, d_h, w, m, L); evaluated table for S1 b16/b32/b64 and S2 b16/b32/b64 + a stated-assumptions total-model estimate for S2 b16. This is the go/no-go input for spending re-calibration EUR | report |
| D6 | CPU (+ MPS informative) microbenchmark: read-out fwd+bwd wall-clock, old vs new, S1 shape and an S2-shaped slice; report the ratio — informative only, the real number is re-calibration's | `scripts/`, report |
| D7 | Housekeeping (already D-log-accepted 2026-07-12): `requirements-gpu.txt` + `docker/Dockerfile` comments cu124 → cu126 | repo |
| D8 | Ledger delta (HO-10 open item #2, bounded): annotate `results/runs.md` so per-run EUR figures reconcile to the console-authoritative $3.4786 total (proportional rescale with a one-line note), or record "not derivable" exactly as `results/M2-calibration.md` §7 already does — **no reconstructed/backfilled explanation** (Pravidlo W) | `results/runs.md` |
| D9 | Report `results/M2-readout-optimization.md`: what changed and why (map to R1/R4/R2/R5), A/B + §14 evidence, D2 profiling numbers, D5 table, D6 ratios, deviations (each explicit), open questions with proposed D-log entries | `results/` |

Suggested order: D2 old-path profile → R1 → D3 A/B green → R4 → D3 re-green + D4 → D5 budget → (R2 if needed → D5 re-run) → D6 → (R5 optional) → D7/D8 → D9.

## 6. Acceptance criteria

| # | criterion | pass |
|---|---|---|
| 1 | Spec §14 suite | all previously-green tests green on CPU fp32, zero modifications to existing test files |
| 2 | A/B equivalence (D3) | new ≡ frozen old readout, fp32 atol 1e−5, all listed config axes |
| 3 | No per-token K/V materialization | no `[·,·,N,k_max,d_h]`-shaped tensor and no contiguous `[·,·,N,d_h,w+1]` window copy in the default training path (D2 evidence; the only admissible exception = the `summary_pos: virtual` fallback branch, off by default) |
| 4 | Analytic budget (D5) | projected S2 b16 total peak ≤ ~60 GiB (headroom under 80); if not reachable with R1+R4+R2 → STOP, report, no further redesign |
| 5 | No-regression microbench (D6) | new read-out fwd+bwd not slower than old on CPU at S1 shape (informative gate; large CPU wins do **not** promise GPU wins — say so in the report) |
| 6 | Anti-goals (§7) respected | reviewer pass in report |

A documented fail on #4/#5 is a valid outcome — report it; the fallback ladder (b)/(c) is Daniel's D-log decision, not CC's.

## 7. Anti-goals

- No spec/docs edits (the report + D7 comment lines are the only doc changes).
- No custom CUDA/Triton kernels, no third-party attention libraries (spec §16 boundary; plain PyTorch ops only).
- No §13 config-surface additions; no silent contingency flips (`summary_pos`, `pool_own_proj`, …).
- No changes to loss, optimizer, LR, data pipeline, seeds, YAML configs of Phase 1; no new training runs beyond microbench-scale; **no quality conclusions from any loss** (spec §16).
- No autonomous scope growth: R1 blocked ⇒ STOP + report + proposed D-log entry. Stop-loss = this one iteration.
- No GPU spend; re-calibration is a separate, Daniel-gated launch (standing launch path per HO-10 §5: RunPod official image + `pod_bootstrap.sh`, secret exists, AP-19 tier check in its pre-flight).

## 8. Reporting & handover

Report at `results/M2-readout-optimization.md` per D9. Commit discipline per `CLAUDE.md` (CC prepares, Daniel commits/pushes; substantive vs housekeeping commits separated — D7/D8 go in their own commit). On completion, signal explicitly: **"readout optimization ready for re-calibration decision"** — the re-calibration launch and its EUR are Daniel's call.
