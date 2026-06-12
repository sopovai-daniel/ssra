# M1 report — SSRA v2 implementation & verification

**Assignment:** `docs/cc/M1-assignment.md` v1 · **Spec:** `docs/spec.md` v1.0
(implemented), v1.1 retention correction accepted mid-M1 — see (v) item 1 ·
**Date:** 2026-06-12 · **Author:** Claude Code (CC) · **Gate decisions: Daniel**

> Status: FINAL. All 7 criteria executed; smoke chain completed 2026-06-12
> 14:46 (`logs/M1-smoke-chain.log`), all 5 runs DONE, none failed.

## (i) Pass/fail table (assignment §4 = spec §14)

| # | test | criterion | result | verdict |
|---|---|---|---|---|
| 1 | Shift (causality) | max Δlogits[1..t−1] ≤ 1e−4 fp32, position set §14.1 | all 4 pools pass on CPU fp32 (`tests/test_shift.py`); MPS informative: Δ = 0.0 | **PASS** |
| 2 | Completion | incremental ≡ full forward at every position, N ∈ {257, 1000}, atol 1e−4 fp32 | all 4 pools pass, max obs. Δ ≈ 3.6e−7 CPU (`tests/test_completion.py`); MPS informative ≈ 3.7e−7 | **PASS** |
| 3 | Gradient flow | nonzero grad in every parameter group + ∂loss/∂S_root ≠ 0 | all 4 pools pass at N=323 (deepest node consumed by read-out); e_ℓ checked per materialized level (`tests/test_gradient_flow.py`) | **PASS** |
| 4 | Throughput/VRAM | G1a: SSRA log-log slope ≤ 1.5 AND < flat slope, N 1k–8k | **SSRA slope 0.983, flat 1.923** (d=192/h=8/L=2, MPS fp32, B=1); plot committed | **PASS** (criterion; gate = Daniel) |
| 5 | P-C diagnostics | implemented + logged during smoke; informative | entropy of Q_φ attention maps + per-query participation logged every 50 steps; diagnostic verified live; P1 pooling attention stayed ~uniform at smoke scale — see (vi-c) | **PASS** |
| 6 | G1b-D3 | P3 within 5 % rel. of P1 final val loss, no divergence, stabilization active | P3 1.5698 vs P1 1.56295 → **gap +0.44 %**; no divergence; τ annealed 2.0→0.5, lb-loss active — see (vi-b) | **PASS** (criterion; gate = Daniel) |
| 7 | P3 determinism | two inference passes (different seeds) bitwise identical | full forward + incremental decode, ste & gumbel_topk configs (`tests/test_p3_determinism.py`); also bitwise on MPS | **PASS** |

`pytest tests/` = 34 passed, 1 skipped (the documented Triton skip, see (iv)).
Config-validation tests (assignment §3) included: P2×linear, P2×k4, hybrid
k_sel ≥ s_ℓ, summary_pos=virtual without override all rejected;
p1_diversity_loss > 0 raises NotImplementedError; k=4 stubbed per AP-7.

## (ii) Artifacts

| artifact | path |
|---|---|
| verification suite | `tests/` (one command: `pytest tests/`) |
| MPS informative log (AP-2) | `logs/M1-mps-informative.log` |
| throughput log / data / plot | `logs/M1-throughput.log`, `results/M1-throughput.json`, `results/M1-throughput.png` |
| superseded bench logs (kept as evidence) | `logs/M1-throughput-d256L4-swapbound.log`, `logs/M1-throughput-d128L2.log` |
| smoke chain log | `logs/M1-smoke-chain.log` |
| per-run JSONL logs | `logs/M1-smoke-{p1,p3,p2,flat,megabyte}.log` |
| G1b-D3 curves | `results/M1-g1b-curves.png` |
| run ledger | `results/runs.md` |

## (iii) Environment snapshot

- Python 3.11.15, PyTorch 2.12.0, macOS-26.5.1-arm64 (MacBook M1, 16 GB unified)
- devices: tests 1–3, 7 judged on **CPU fp32** (AP-2); MPS re-run informative;
  throughput + smoke on **MPS fp32**
- code commit at verification: `f06263f` (+ this report's commit)
- pinned deps: `requirements.txt` / `pyproject.toml` (torch 2.12.0, PyYAML 6.0.3,
  numpy 2.4.6, pytest 9.0.3, matplotlib 3.11.0, flash-linear-attention 0.5.0)

## (iv) Baseline provenance

| baseline | source | status |
|---|---|---|
| (a) flat pre-norm Transformer | `baselines/flat.py` (this repo); same d/h/L, RoPE, SDPA causal | smoke-run + bench ✓ |
| (b) GatedDeltaNet (Log-Linear family) | **fla-org/flash-linear-attention v0.5.0**, commit `3a9ce1c83a13994d824dbb3421e2989d330bb38b`, **MIT** (verified 2026-06-12); PyPI `flash-linear-attention==0.5.0` | **integration done, execution deferred to M2 GPU** (AP-5): `import fla` OK locally, every kernel path imports Triton at layer-import time and Triton has no macOS/arm64 build (`tests/test_baselines.py::test_loglinear_integration`, explicit skip). NVlabs/GatedDeltaNet rejected: NVIDIA Source Code License-NC is not Apache-2.0-compatible. |
| (c) MEGABYTE-style 2-level | `baselines/megabyte.py`, minimal faithful to arXiv:2305.07185 (patch=8, global-over-patches + local-within-patch, both causal) | fwd+bwd integration + causality tests ✓; short smoke DONE (val loss 2.008, no divergence) |

## (v) Open questions / spec ambiguities (proposed D-log entries; no docs edited)

1. **MD-2 retention rule has a re-entry gap (spec §9).** The pointwise set
   Frontier(t) ∪ Fenwick(t−w−1) evicts a node (ℓ, j odd, span end E) at
   t = E + 2^ℓ (parent forms) although the read-out still consumes it during
   t ∈ [E + w + 1, E + 2^ℓ + w]; for 2^ℓ ≤ w these intervals are disjoint.
   Concrete counterexample (w=2): node (1,3) spanning [5,6] leaves both sets at
   t=8 but is required by Fenwick(t−3) at t=9,10. Implemented instead: the
   minimal correct closure — retain (ℓ, j odd) for t ∈ [E, E + 2^ℓ + w]; level-0
   states live in a (w+2)-slot ring (window ∪ the level-0 Fenwick block at
   t−w−1); even-j nodes are consumed within their formation step and never
   stored. Same complexity class O((w + m·log N)·d); the exact constant gains a
   ~w·log₂(m) term over the spec's (w+1)d + 2md(⌊log₂N⌋+1). The completion test
   (test 2) certifies the implementation. **Status: RESOLVED same-day** —
   Daniel accepted the finding and published spec v1.1 (`08f8939`, docs-only:
   §9/§10/§18 MD-2 corrected to the interval closure, D-log entry 2026-06-12).
   The v1.1 wording matches this implementation exactly, including the
   optional immediate drop of lowbit(end) > 2^ℓ (even-j) nodes; no code change
   was needed.
2. **Stack-final LayerNorm.** Spec §3 fixes the sublayer pattern but is silent
   on a final LN before unembedding; standard pre-norm convention (ln_f) was
   used. **Proposed D-log entry:** confirm ln_f (or strike it).
3. **m_schedule=linear scope.** §13's validation list rejects only P2×linear,
   but the §13 YAML comment says "linear: P1 only". Enforced strictly: linear ⇒
   pool=p1 (P3/hybrid×linear also rejected). Additionally linear schedules with
   s_ℓ > k·s_{ℓ−1} (Pool would have to expand) are rejected. **Proposed D-log
   entry:** confirm both.
4. **§5.3 freedoms pinned by implementation** (within spec wording, flagged for
   the record): STE backward = rank-wise masked softmax over σ/τ aligned to the
   slot-ordered selection; load-balance p̄ from the soft relaxation (hard counts
   carry no gradient), KL collected per lossy node-batch and averaged; τ anneal
   = linear interpolation tau_start → tau_end over tau_anneal_steps.
5. **summary_pos=virtual override flag name.** Spec demands "an explicit
   override flag" without naming it; implemented as `summary_pos_override: true`.
6. **Spec §12 parameter inventory** does not list ln_f (see 2) nor the untied
   unembedding matrix when `tied_embeddings: false`.

7. **P1 pooling attention ~uniform at smoke scale (M3 relevance: P-A/P-B).**
   See (vi-c). If Q_φ stays near-uniform at M2/M3 scale and budgets, the
   latent queries add little addressing capacity over mean pooling, which
   bears on the cross-scale-sharing prediction (P-A) and on recall through
   compressed nodes (P-B needle suite) — the M3 ablations read on a mechanism
   that may effectively be "uniform read + residual" unless specialization
   emerges with scale/steps. Question for the Claude.ai project: whether to
   track entropy-vs-steps explicitly in M2 as a P-C early signal (no design
   change proposed; `p1_diversity_loss` stays unverified [K] and off,
   `pool_own_proj` stays off).

None of these blocked implementation; no design change was made silently.

## (vi) Gate numbers + CC recommendation (decisions are Daniel's)

### (vi-a) G1a — throughput scaling

Protocol: AP-6 with a swap-aware fit rule — a candidate (d,h,L) "fits" only if
the sampled peak stays under 70 % of physical RAM at N=8k, because on unified
memory both models otherwise page-thrash and the 8k point measures swap, not
compute (first run, d=256/L=4: SSRA 165.6 s and flat 36.0 s at N=8k with 17–19
GiB peaks; preserved in `logs/M1-throughput-d256L4-swapbound.log`). MPS has no
peak-memory API in torch 2.12, so peak = `driver_allocated_memory` sampled
after fwd and after bwd (documented lower bound), allocator cache reset
between models. Selected size: **d=192, h=8, L=2** (~0.95 M params each), B=1,
fp32, MPS; rejected: d=384/L=4 (OOM at 8k), d=256/L=4 and d=256/L=2 (swap).

| N | SSRA fwd+bwd | flat fwd+bwd | SSRA peak | flat peak |
|---|---|---|---|---|
| 1024 | 510.8 ms | 46.3 ms | 1.17 GiB | 1.12 GiB |
| 2048 | 1053.9 ms | 160.7 ms | 2.24 GiB | 1.21 GiB |
| 4096 | 1985.1 ms | 523.6 ms | 4.31 GiB | 3.08 GiB |
| 8192 | 4008.9 ms | 2654.1 ms | 7.77 GiB | 11.14 GiB |
| 16384 | swap-bound (excluded) | OOM | 16.7 GiB | >20 GiB |

**G1a numbers: SSRA log-log slope 0.983; flat 1.923** (criterion: ≤ 1.5 and
< flat) → criterion satisfied. Honest notes per spec §10: SSRA's absolute
wall-clock is 1.5–11× above flat in this range (QKVO/d² constants + gather
overheads); the crossover is just above N=8k on this hardware. At N=8k SSRA's
peak memory is already *below* flat's (7.8 vs 11.1 GiB — flat's quadratic
attention matrices dominate on the MPS SDPA backward path).

**CC recommendation: G1a = pass.** The slope criterion is met with margin on a
clean, swap-free measurement range; the sub-quadratic claim (D3/D6, spec §10)
is empirically supported. Slope 0.98 < the expected 1.0–1.3 band (MD-7)
because fixed per-level Python/dispatch overheads still dominate at N=1k.

### (vi-b) G1b-D3 — P3 stability screen (X = 5 %)

Protocol AP-3: matched pair `M1-smoke-p1` / `M1-smoke-p3`, identical config /
data / step budget / seed (1337) except `pool`; metric = final validation loss
on the held-out split (fixed val batches across runs); stabilization active
(λ_lb = 0.01 load-balance + τ anneal 2.0 → 0.5 over 1000 steps, STE).

| run | final val loss (nats) | bpc | divergence |
|---|---|---|---|
| M1-smoke-p1 | **1.56295** | 2.25486 | none |
| M1-smoke-p3 | **1.56980** | 2.26474 | none |

**Relative gap: (1.56980 − 1.56295) / 1.56295 = +0.44 %** — well inside
X = 5 % and outside the 4–6 % uncertainty band, so no extra seeds are
required per AP-3. Stabilization was demonstrably active: τ annealed 2.0 →
0.5 over the first 1000 steps (logged per step block), load-balance KL logged
throughout (early ≈ 0.046–0.109, settling to ≈ 0.032–0.036 by run end), STE
gradients, hard deterministic inference (test 7). Loss curves:
`results/M1-g1b-curves.png`.

**CC recommendation: G1b-D3 = pass; keep P3 as challenger** (hybrid stays a
config-only fallback). Reminder of scope: smoke runs verify functionality —
this is a stability screen, not a quality comparison (assignment §5).

### (vi-c) P-C diagnostics — observed behavior (informative, not gating)

`p1_attn_entropy` is pinned at 3.4657 ≈ ln(32) = max entropy over the 32 keys
of a lossy node in every logged step of `M1-smoke-p1`. Verified that this is a
real observation, not a dead metric — three independent checks:

1. the metric responds to non-uniform attention (scaling Q_φ ×100 at init
   drops it to ≈ 3.25);
2. per-query participation min/max from the same diagnostic call vary across
   logged smoke steps (0.0504–0.0530 / 0.0755–0.0806), so the pipeline emits
   live per-step values;
3. a tiny CPU training run printed at full precision shows the entropy is
   computed live and falls monotonically with training — but only in the
   4th–6th decimal (3.4657279 → 3.4644055 over 30 steps at lr 1e-2), far
   below the log's 4-decimal rounding.

**Statement:** at smoke scale, P1 latent-query pooling attention stayed
~uniform throughout training — *no* query collapse (the P-C failure mode),
but *no query specialization either*: with Q_φ ~ N(0, 0.02²) and LN'd keys,
pooling scores stay near zero and P1 acts ≈ as mean pooling with a residual.
Observation only; no config changed, no contingency flag enabled
(`pool_own_proj` stays off).

### Smoke runs (assignment §5)

Corpus (AP-4): Tiny Shakespeare,
`https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`,
1,115,394 bytes, char vocab 65; train/val split 90/10 (val = last 10 %).
Runs (configs committed before launch, commit `21ceeab`): SSRA-P1, SSRA-P3,
SSRA-P2, flat, MEGABYTE-style; d=256/h=8/L=5 (3,999,232 params SSRA), seq 512,
batch 4, 2000 steps, AdamW lr 3e-4 cosine, MPS fp32.

Provenance note: per-run `meta.commit` differs across the chain —
`M1-smoke-p1` records `184d9bd`, the later runs record `f06263f` — only
because the docs-only handover commit HO-04 (`f06263f`, single file under
`docs/handover/`) landed 13 s after the chain started, between p1's process
start and the rest. `git diff 184d9bd f06263f` touches no code path: **model
code was identical for the whole chain.**

| run | arch/pool | params | final val loss | bpc | wall-clock | status |
|---|---|---|---|---|---|---|
| M1-smoke-p1 | SSRA P1 | 3,999,232 | 1.56295 | 2.25486 | 4734 s | DONE |
| M1-smoke-p3 | SSRA P3 | 3,977,477 | 1.56980 | 2.26474 | 4498 s | DONE |
| M1-smoke-p2 | SSRA P2 | 4,632,832 | 1.56045 | 2.25126 | 4539 s | DONE |
| M1-smoke-flat | flat | 3,960,832 | 1.57880 | 2.27772 | 368 s | DONE |
| M1-smoke-megabyte | MEGABYTE-style | 4,391,968 | 2.00817 | 2.89717 | 112 s | DONE |

All three SSRA pools and both runnable baselines trained without divergence
or NaN; every run completed its full 2000-step budget (functionality
confirmed — no quality conclusions drawn, per assignment §5).

**NoPE summary-key watch (spec §6 [HYPOTÉZA], assignment §5):** no positional
pathology observed at smoke scale — all SSRA variants trained smoothly to
val loss ≈ 1.56 with monotone-decreasing curves and stable validation; no
loss spikes or divergence that would implicate the pure-NoPE summary keys.
`summary_pos` stays `none`; the [HYPOTÉZA] marker stands until larger-scale
M2 evidence (a smoke run cannot positively confirm the hypothesis, only fail
to falsify it).

**Wall-clock note (honest accounting):** SSRA smoke runs took ~75 min vs
6 min for flat at N=512 — at short sequence lengths SSRA's constants
(per-level Python dispatch, read-out gather) dominate; consistent with the
G1a picture where the crossover sits above N=8k on this hardware.

## (vii) Scope notes / deviations

- **k=4 tree path stubbed** (AP-7): config accepted, model constructor raises
  NotImplementedError with a pointer to AP-7; must exist before M3 ablation (g).
  All other §13 flags are implemented and tested (w, level_emb, readout_params
  separate, m_schedule linear, hybrid, pool_own_proj, summary_pos virtual
  behind override).
- **p1_diversity_loss** > 0 raises NotImplementedError ([K], assignment §3) —
  formulation not invented.
- No inference-time stochasticity anywhere (D5); verified bitwise by test 7.
- No edits under `docs/` or `paper/`.
