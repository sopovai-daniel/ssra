# M2 — read-out optimization report (assignment D9)

**Date:** 2026-07-13 · **Assignment:** `docs/cc/M2-readout-optimization.md` v1
(D-log 2026-07-12, option (a), step 2) · **Spec:** v1.2 (UNCHANGED) ·
**Development:** CPU fp32 (MacBook, torch 2.12.0); MPS numbers informative only
(AP-2). **No GPU spend.**

**Headline:** R1 + R4 implemented and certified. The training read-out no longer
materializes any per-token K/V copy — autograd-retained read-out memory drops
**~9.7×** (measured, CPU audit), read-out fwd+bwd wall-clock drops **~5.4–5.9×**
(measured, CPU). The analytic model (validated to ±1.2% against measurement and
back-predicting all five calibration OOMs) projects **SSRA S2 b16 ≈ 36.6 GiB
peak on the A100 80 GB — the ≤ ~60 GiB gate passes; R2 checkpointing is not
needed.** Spec §14 suite green with zero modifications to existing tests; the
frozen-reference A/B suite (AP-20) passes at fp32 atol 1e-5 on all required
axes. R5 (SDPA in `node_attn`) was implemented within its timebox and kept.
No throughput promise is made — the re-calibration decides (assignment §6 #5).

**Signal: readout optimization ready for re-calibration decision.**

---

## 1. What changed and why (map to R1/R4/R2/R3/R5)

### R1 — level-grouped summary attention, zero gather (mandatory) — DONE

`src/ssra/model.py::SSRALayer.readout` restructured per the assignment §3
structural lemma: node (ℓ, j odd) is consumed exactly by the contiguous token
run t ∈ [j·2^ℓ + w + 1, (j+1)·2^ℓ + w]. Per level ℓ the token axis is shifted
by w+1 (p = t − w − 1), partitioned into blocks of 2^ℓ, and the **odd blocks**
attend that node's s_ℓ rows as one regular grouped bmm
(`[B,h,G2,2^ℓ,d_h] × [B,h,G2,s_ℓ,d_h]`). Odd/even block selection is a
pair-chunk reshape (`unflatten` + first-half slice) — **no gather, no
per-token index table** in the default path. Per-level scores are padded
(−inf) into `[B,h,N,s_ℓ]` buffers, concatenated with the window scores →
**ONE softmax** (§8) → split → per-level grouped AV bmms → summed output.
Levels with no consumer (2^ℓ > N−w−1) are skipped. Ragged N is native (spans
truncate at N; every consumed node is materialized per the §8 alignment
lemma). Level 0 participates in the same scheme (s_0 = 1, odd p, e_0 key tag
as before). `level_emb: off` and `readout_params: separate` work unchanged
(tag/projection applied to the selected rows before grouping).

**`summary_pos: virtual` contingency — option (i)** (assignment §3): the old
gathered realization is kept verbatim as `SSRALayer._readout_gathered` and is
dispatched **only** when `summary_pos == "virtual"` (still gated by
`summary_pos_override`, off by default). `build_readout_index` and the cache
plumbing are unchanged and serve this branch. The constant-phase RoPE identity
(option ii) was not attempted, per the "do not spend more than trivial effort"
instruction.

### R4 — window path without materialized per-token K/V copies (mandatory) — DONE

The `unfold`+einsum window (which D2 profiling **confirmed** was forcing two
contiguous `[B,h,N,w+1,d_h]`-sized copies — hypothesis §2.5) is replaced by a
block-local band: block size c = w, each query block attends its own +
previous key block (`[B,h,N/c,c,2c]` raw scores, masked to exactly
[t−w, t] ∩ [1, t], MD-6), and the raw scores join the common softmax. Key/value
band layout costs 2 token-K/V copies total (O(B·h·N·d_h)), not O(w) of them.
No SDPA anywhere in the read-out (explicit prohibition respected — the single
softmax over heterogeneous keys is intact and literal in the code).

### R2 — selective recomputation (conditional) — NOT APPLIED

The D5 budget projects S2 b16 ≈ 36.6 GiB ≤ ~60 GiB, so per the assignment
("do not apply blindly if the budget already clears") no checkpointing was
added. No §13 config surface change anywhere.

### R3 — token-chunked gather fallback — NOT NEEDED

R1 hit no wall; R3 was never invoked. No STOP was triggered.

### R5 — node_attn via SDPA (optional, timeboxed) — DONE and kept

`SharedAttention.node_attn` now uses `F.scaled_dot_product_attention`
(bidirectional, RoPE pre-applied, dropout on probs — semantics identical; the
read-out and P1 `cross_attn` are untouched). All three gate conditions hold:
§14 green on CPU fp32, A/B suite green, CPU microbench improved — up-pass
fwd+bwd at S1 shape (B=4, N=1024): 300.1 ms → 237.4 ms (**−21%**). Side
effect recorded honestly: the **CPU** saved-tensor audit of the non-read-out
layer remainder grew from 59.7 to 64.3 × B·N·d fp32 elements (the CPU SDPA
kernel saves q/k/v + output + logsumexp). The D5 projection uses the larger
(64.3) figure, i.e. the budget below is conservative w.r.t. this choice; on
CUDA, flash-attention SDPA typically retains less, not more. R5 is a
self-contained one-hunk change and can be reverted independently if the
re-calibration shows a GPU-side regression.

## 2. Equivalence evidence (D3 + D4, acceptance #1/#2)

- **D3 — `tests/test_readout_equiv.py` (new file): 28 passed.** Contains the
  frozen reference — a verbatim copy of the replaced `readout` @ commit
  `576b927` (sole mechanical change: `self` → `layer`). Grid: ragged
  N ∈ {257, 1000, 1024} × level_emb {on, off} × readout_params
  {shared, separate} × (m, w) ∈ {(16,64), (4,8), (8,32)} (24 cases, fp32 atol
  1e-5, randomized inputs, level_emb tags randomized to nonzero so tag bugs
  can't hide behind zeros-init); `summary_pos: virtual` fallback vs reference
  (bit-for-bit, 2 cases); full-model logits new vs monkeypatched-old (atol
  1e-5); gradient-flow guard through the grouped plumbing.
- **D4 — full suite: 64 passed, 1 skipped**
  (`logs/M2-readout-optimization-pytest.log`). Zero modifications to existing
  test files (`git diff` on `tests/` shows only the added file). The skip is
  `test_baselines.py::test_loglinear_integration` — fla is not installed in
  the CPU dev env; on the GPU box it is the known-red fla × transformers
  Phase-4-only failure, status unchanged per assignment §4. The §14.2
  completion test (N ∈ {257, 1000}, atol 1e-4 fp32) passes against the
  **unchanged** decoder — the batched-vs-incremental certificate of §8
  realization freedom.

## 3. D2 — memory-behavior verification (acceptance #3)

Method: `scripts/profile_readout_memory.py` audits every tensor autograd
saves during one read-out forward via `torch.autograd.graph.saved_tensors_hooks`
(unique storages, parameters excluded) — the saved set is exactly what
coexists across all L layers and caused the calibration OOM. z/levels are
detached, so the audit is read-out-internal only. CPU fp32:

| config | path | saved total | gathers B·h·N·k_max·d_h | window copies B·h·N·(w+1)·d_h |
|---|---|---|---|---|
| S1 shape, B=4 (d=384, h=6, N=1024, k_max=95) | gathered (old) | 2,026.2 MiB | **2 × 570.0 MiB** | **2 × 390.0 MiB** |
| ditto | grouped (new) | **207.8 MiB (−9.75×)** | **0** | **0** |
| S2 shape, B=2 (d=640, h=10, N=1024) | gathered (old) | 1,688.7 MiB | 2 × 475.0 MiB | 2 × 325.0 MiB |
| ditto | grouped (new) | **173.3 MiB (−9.74×)** | **0** | **0** |

- Claim §2.1 **verified**: the gathers are the dominant term (56% of old-path
  residency) with exactly the predicted B·h·N·95·d_h size (einsum saves them
  bmm-flattened as `[B·h·N, 95, 64]`).
- Claim §2.5 (window `unfold` → contiguous copies) **verified**, hypothesis →
  measured: two `[B·h·N, {65,64}, {64,65}]` copies, 38% of old residency.
- New path: **no tensor of either family exists** in the training path
  (byte-exact size scan over all saved storages; the largest new-path tensor
  is the softmax probs `[B,h,N,239]`). The only admissible exception per
  acceptance #3 — the `summary_pos: virtual` fallback — is off by default.
- MPS informative (S1 shape, B=4): retained-after-forward 2,037.8 MiB →
  278.9 MiB (**−7.3×**), consistent with the CPU audit.

## 4. D5 — analytic peak-memory model (acceptance #4)

`scripts/readout_memory_model.py`: closed-form enumeration of autograd-retained
tensors per layer, old vs new, as a function of (B, h, N, d_h, d, w, m).

**Validation** (CPU fp32, vs the D2 audit): old path +0.7% (both shapes), new
path −1.1%/−1.2%. The old-path closed form also **reproduces the measured
failed allocation from the calibration OOM trace exactly** (S1 b32 gather term
= 2.226 GiB) and back-predicts all five calibration OOMs (totals 109–536 GiB
vs 80 GB card).

**GPU projection assumptions (stated per AP-20):** bf16 autocast (AP-16) —
matmul-saved activations 2 B/elt, softmax output fp32 4 B/elt + bf16 cast
copies of prob slices; non-read-out per-layer remainder measured by the same
audit (64.3 × B·N·d elements, identical across S1/S2) scaled to bf16; stack
terms = logits (bf16) + CE log-probs (fp32) + fp32 params/grads/AdamW;
forward/backward transients covered by a multiplicative margin on the layer
term **calibrated on the one measured point** (old S1 b16 = 54.67 GiB ⇒ margin
1.11). Model = end-of-forward residency; the margin is empirical, not derived.

| config | old GiB/layer | new GiB/layer | old total (model) | new total (model) | measured (old) |
|---|---|---|---|---|---|
| S1 b16 | 4.01 | **0.47** | 54.79 | **15.40** | 54.67 (calibration point) |
| S1 b32 | 8.03 | 0.93 | 109.21 | 30.44 | OOM ✓ |
| S1 b64 | 16.05 | 1.86 | 218.04 | 60.51 | OOM ✓ |
| S2 b16 | 6.69 | **0.78** | 135.06 | **36.59** | OOM ✓ |
| S2 b32 | 13.38 | 1.55 | 268.84 | 71.92 | OOM ✓ |
| S2 b64 | 26.76 | 3.10 | 536.40 | 142.58 | OOM ✓ |

**Gate verdict (acceptance #4): projected S2 b16 total ≈ 36.6 GiB ≤ ~60 GiB —
PASS, with ~43 GiB headroom on the 80 GB card.** Secondary read-outs for the
re-calibration plan: S1 b16 ≈ 15.4 GiB (regression run vs measured 54.67),
S1 b32 ≈ 30.4 GiB, S1 b64 ≈ 60.5 GiB (borderline), S2 b32 ≈ 71.9 GiB (likely
fits 80 GB but above the 60 GiB comfort bound — cheap to attempt, may OOM),
S2 b64 does not fit. After the fix the read-out is **no longer the dominant
activation term** (0.78 vs 1.29 GiB/layer non-read-out at S2 b16); any future
memory work should target the up-pass/FFN remainder, not the read-out.

## 5. D6 — microbenchmark (acceptance #5)

`scripts/bench_readout.py`, read-out fwd+bwd, median of 5 (fp32):

| shape | device | old | new | ratio |
|---|---|---|---|---|
| S1 (B=4, N=1024, d=384, h=6) | CPU | 969.1 ms | 164.9 ms | **×5.88** |
| S2 slice (B=2, N=1024, d=640, h=10) | CPU | 829.2 ms | 153.7 ms | **×5.39** |
| S1 | MPS (informative) | 932.7 ms | 168.8 ms | ×5.52 |
| S2 slice | MPS (informative) | 779.6 ms | 154.9 ms | ×5.03 |

No-regression gate: PASS (large improvement). **Explicitly: large CPU/MPS wins
do NOT promise GPU wins** — the gather-traffic bottleneck has different
relative cost on an A100, and the measurable target (S2 ≥ ~11.5k tok/s ⇒
projected ≤ 25 EUR @ 850M tok) can only be decided by the re-calibration.
R5 side measurement: up-pass fwd+bwd 300.1 → 237.4 ms (−21%) at S1 shape.

## 6. Acceptance criteria self-check

| # | criterion | status |
|---|---|---|
| 1 | §14 suite green, existing tests unmodified | **PASS** — 64 passed, 1 known skip; `tests/` diff = one added file |
| 2 | A/B ≡ frozen reference, fp32 atol 1e-5, all axes | **PASS** — 28/28 (§2) |
| 3 | no per-token K/V materialization in default path | **PASS** — D2 byte-exact scan; virtual fallback is the sole, off-by-default exception |
| 4 | analytic S2 b16 total ≤ ~60 GiB | **PASS** — 36.6 GiB (§4) |
| 5 | CPU no-regression | **PASS** — ×5.4–5.9 faster (§5); no GPU promise |
| 6 | anti-goals respected | reviewer pass below |

**Anti-goals reviewer pass (§7):** no spec/docs edits (only this report + D7
comment lines + the D8 ledger note); plain PyTorch ops only, no custom
kernels/third-party attention (SDPA is core PyTorch and used only where R5
explicitly permits it — never in the read-out); no §13 config additions and no
contingency flips (`summary_pos` default untouched, `pool_own_proj` untouched);
no changes to loss/optimizer/data/seeds/Phase-1 YAMLs; no training runs, no
quality conclusions from any loss; no GPU spend; stop-loss respected (one
iteration, no second redesign track, R3 never opened).

## 7. Deviations (all explicit, none silent)

1. **R5 exercised** (optional but permitted): kept after all three gate checks
   passed; CPU-side memory side effect (+4.6 × B·N·d units/layer in the fp32
   audit) measured, disclosed, and included in the D5 budget (§1/R5).
2. **Frozen reference** carries the unavoidable mechanical `self` → `layer`
   rename; otherwise verbatim (diffable against `git show 576b927:src/ssra/model.py`).
3. **D2 method**: saved-tensors audit instead of raw allocator statistics —
   strictly stronger evidence (per-tensor shapes + bytes, directly testing
   acceptance #3); process peak RSS and MPS allocator numbers reported alongside.
4. `scripts/profile_readout_memory.py` and `scripts/bench_readout.py` import
   the frozen reference from `tests/test_readout_equiv.py` (single source for
   the frozen copy; scripts add `tests/` to `sys.path`).
5. D5 margin factor (1.11) is calibrated on the single measured total
   (old S1 b16), not derived from first principles — stated in §4.

## 8. Open questions → proposed D-log entries

1. **Re-calibration go/no-go (Daniel + oversight review):** proposed D-log:
   *"Read-out optimization (2026-07-13): R1+R4 implemented, AP-20 satisfied
   (frozen A/B 1e-5 green, §14 green unmodified, analytic model ±1.2% vs
   audit); no per-token K/V materialization; projected S2 b16 ≈ 36.6 GiB ≤ 60
   GiB gate; CPU read-out fwd+bwd ×5.4–5.9. R2 not needed, R3 not used, R5
   (SDPA node_attn) kept. GO for re-calibration (~1–2 EUR): run set S1 b16
   (regression vs 9,457 tok/s / 54.67 GiB), S1 b32, S2 b16 = gate; optional
   cheap attempts S1 b64 / S2 b32 (projected 60.5 / 71.9 GiB — may OOM,
   informative). Verdict vs measurable target (S2 b16 trains ∧ ≥ ~11.5k tok/s
   ⇒ ≤ 25 EUR @ 850M) decided there."*
2. **No normative questions.** Spec §8/§6/§9 sufficed exactly as written; the
   restructure exercised only the §8 realization freedom. No ambiguity found
   that needs a spec change.
3. (Carry-over, unchanged: transformers pin before Phase 4; AP-19 both console
   prices at next launch pre-flight — already in the D-log.)

## 9. Artifacts

| what | where |
|---|---|
| restructured read-out (R1+R4) + virtual fallback | `src/ssra/model.py` |
| R5 SDPA node attention | `src/ssra/attention.py` |
| frozen-reference A/B suite (D3) | `tests/test_readout_equiv.py` |
| saved-tensor memory audit (D2) | `scripts/profile_readout_memory.py` |
| analytic memory model + projection table (D5) | `scripts/readout_memory_model.py` |
| read-out microbench (D6) | `scripts/bench_readout.py` |
| full-suite log (D4) | `logs/M2-readout-optimization-pytest.log` |
| cu126 pin comments (D7) | `requirements-gpu.txt`, `docker/Dockerfile` |
| ledger delta note (D8) | `results/runs.md` |

Reproduce: `pytest tests/` · `python scripts/profile_readout_memory.py --path
{gathered,grouped} --preset {s1,s2} --batch {4,2}` · `python
scripts/readout_memory_model.py --validate` and `--table --nonreadout-s1 64.26
--nonreadout-s2 64.25 --margin 1.11` · `python scripts/bench_readout.py`.
