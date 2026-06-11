# Zadanie pre CC — M1: SSRA v2 implementation & verification

**Version:** v1 (2026-06-11) · **Milestone:** M1 (`docs/03-poc-plan.md`) · **Mode:** veto-based — items marked AP-x are protocol details introduced by this assignment and stand unless Daniel vetoes them.

**Authority chain (binding):** `docs/spec.md` v1.0 = single source of truth for WHAT to build; this assignment only schedules and operationalizes it — **on any conflict, spec wins**. Project state/decisions: `docs/00` D-log. Behavior rules: `CLAUDE.md`. Closed decisions (D1–D6, Q1–Q5, MD-1…MD-10) must not be reopened. If the spec looks ambiguous, contradictory, or unimplementable: STOP, write the question + a proposed D-log entry into the report. Never improvise design changes.

## 1. Goal

Implement SSRA v2 (variant A) exactly per spec, plus the baseline harness; run the full M1 verification suite (spec §14) and local smoke runs; produce `results/M1-report.md` with a pass/fail table. M1 output feeds two gate decisions made by **Daniel**: **G1a** (throughput scaling) and **G1b-D3** (P3 stability screen, **X = 5 %**). CC reports numbers and a recommendation; CC does not decide gates.

## 2. Deliverables

| # | deliverable | where |
|---|---|---|
| 1 | SSRA v2 model per spec §3–§9, §11–§13: level-wise batched up-pass, Fenwick read-out (variant A), incremental decoder with retention rule, P1/P2/P3/hybrid behind one Pool interface, YAML config + validation | `src/ssra/` |
| 2 | Baselines: (a) flat pre-norm Transformer (same d/h/L); (b) Log-Linear / GatedDeltaNet from a public implementation — pin URL + commit + license in the report; (c) MEGABYTE-style 2-level (minimal faithful) | `baselines/` |
| 3 | Automated test suite per spec §14.1–.3 + §14.7 + config-validation tests; one command (`pytest tests/`) | `tests/` |
| 4 | Throughput/VRAM benchmark + P-C diagnostics logging hooks | `scripts/`, `src/ssra/` |
| 5 | Smoke runs (§5) with committed configs + ledger rows | `experiments/`, `results/runs.md`, `logs/` |
| 6 | `results/M1-report.md` (§7) + committed throughput/VRAM plot artifact | `results/` |
| 7 | Pinned dependencies (requirements.txt or pyproject) + short "how to run tests/benchmarks" note | repo root / README |

Suggested order: core SSRA + tests 1–3, 7 → flat baseline + throughput (test 4) → smoke runs + diagnostics + G1b-D3 pair (tests 5–6) → baselines (b), (c) integration → report.

## 3. Binding constraints (reminders — spec text wins over this list)

- Level-wise batching, **no Python recursion in the training path** (D6, spec §4); ragged sequences: node materialized iff span ⊆ [1, N], no padding nodes.
- Pre-norm residual + LN_node in every node (hard requirement); bidirectional intra-node attention; slot-RoPE positions 1..n_in; e_ℓ broadcast-added (spec §4, §6).
- Read-out: window [t−w, t] inclusive (MD-6); summary keys NoPE + e_ℓ on **keys only** (MD-5); one softmax over heterogeneous keys; θ shared with nodes; `readout_params: separate` flag implemented (spec §8).
- Incremental decoder implements retention rule Frontier(t) ∪ Fenwick(t−w−1) (spec §9, MD-2) — the completion test depends on it.
- Pool operators per spec §5: P1 reuses the layer's θ projections (MD-3); P2 valid only with fixed-m × k=2; P3 = STE default, load-balance loss, τ anneal, hard deterministic inference; hybrid(k_sel).
- Config surface exactly per spec §13 incl. validation rules (reject P2×linear, P2×k=4, hybrid k_sel ≥ s_ℓ, summary_pos=virtual without override flag). Defaults: m=16, w=64, k=2, pool=p1, level_emb=on, summary_pos=none.
- `p1_diversity_loss`: plumb the config key; the formulation is unverified [K] ⇒ any value > 0 must raise NotImplementedError. Do not invent a formulation.
- No inference-time stochasticity of any kind (D5).
- fp32 for all correctness tests — tolerances are defined in fp32 (MD-7).

## 4. Acceptance criteria (= spec §14; report every row as pass/fail)

| # | test | pass criterion | artifact |
|---|---|---|---|
| 1 | Shift (causality) | max Δlogits[1..t−1] ≤ 1e−4 fp32 for the position set of spec §14.1 | `tests/test_shift.py` |
| 2 | Completion | incremental decoding ≡ full batched forward at **every** position, N ∈ {257, 1000}, atol 1e−4 fp32 | `tests/test_completion.py` |
| 3 | Gradient flow | nonzero grad in every parameter group (θ, φ, e_ℓ for materialized levels, embeddings, FFN, LNs); ∂loss/∂S_root path exists | `tests/test_gradient_flow.py` |
| 4 | Throughput/VRAM | fwd+bwd wall-clock + peak memory, N ∈ {1k, 2k, 4k, 8k} (+16k if it fits), SSRA vs flat at same d/h/L; **G1a:** SSRA log-log slope ≤ 1.5 AND strictly below flat's slope; plot committed | `scripts/bench_throughput.py` |
| 5 | P-C diagnostics | implemented + logged during smoke runs (entropy of Q_φ attention maps, per-query participation); informative, not gating | training-loop hooks + logs |
| 6 | G1b-D3 | P3 within **5 % relative** of P1 final validation loss on a matched 10M smoke pair, no divergence, stabilization active (spec §5.3) | smoke pair per AP-3 |
| 7 | P3 determinism | two inference passes (different seeds) → bitwise identical outputs | `tests/test_p3_determinism.py` |

A documented **fail is a valid M1 outcome** — report it; do not tune until it passes without flagging what was changed.

## 5. Smoke runs (local: MacBook M1 16 GB, MPS with CPU fallback)

Char-level, ≤10M params, short budget — **functionality only, no quality conclusions** (`03` M1). Minimum set: SSRA-P1, SSRA-P2, SSRA-P3 (P1/P3 = the matched G1b-D3 pair), flat baseline; baselines (b)/(c): forward+backward integration check + short smoke where the implementation permits (AP-5). Watch and report NoPE summary-key behavior (spec §6, [HYPOTÉZA]); if positional pathology appears, report it — **do not** flip `summary_pos: virtual` yourself (Daniel decides).

Run discipline (CLAUDE.md): 1 run = 1 YAML in `experiments/` committed BEFORE launch + a row in `results/runs.md`.

## 6. Assignment-level protocol (AP — new in this document, veto applies)

- **AP-1** This assignment lives at `docs/cc/M1-assignment.md`; future CC assignments go under `docs/cc/`.
- **AP-2** Device policy: tests 1, 2, 3, 7 are **judged on CPU fp32** (deterministic); additionally executed on MPS and reported informatively (MPS numerics do not gate). Throughput (test 4) measured on MPS; device + dtype recorded.
- **AP-3** G1b-D3 protocol: identical config / data / step budget / seed except `pool`; metric = final validation loss on a held-out split; report loss curves + the relative gap. If the gap lands in 4–6 % (±1 pp around the threshold), run 2 extra seeds and report mean ± range. CC gives a recommendation only; the gate verdict is Daniel's.
- **AP-4** Smoke corpus: small public char-level corpus (enwik8 subset or Tiny Shakespeare); exact source URL + size recorded in `results/runs.md`.
- **AP-5** Baseline (b): verify the chosen public implementation's hardware requirements and license at integration time. If it is CUDA/Triton-bound, local M1 verification is limited to integration (import + tiny CPU forward if supported); report "integration done, execution deferred to M2 GPU" explicitly — never silently skip. License must be Apache-2.0-compatible; record provenance in the report.
- **AP-6** Throughput config: largest (d, h, L) that fits **both** models at N = 8k locally; SSRA and flat always measured under identical settings; any precision/size deviation recorded. (The slope criterion measures scaling, not absolute speed.)
- **AP-7** The k = 4 tree path may be stubbed (NotImplementedError with a clear message) if it threatens M1 scope; flag it in the report; it must exist before M3 ablation (g). All other §13 flags (w, level_emb, readout_params, m_schedule linear, hybrid) are implemented now.

## 7. Reporting (HO-03 addendum — mandatory)

`results/M1-report.md` must contain: (i) the pass/fail table — test → result → criterion → verdict — for all 7 criteria; (ii) paths to logs and plots; (iii) environment snapshot (Python/PyTorch versions, device, commit hash); (iv) baseline (b) provenance (URL + commit + license); (v) open questions / spec ambiguities encountered, each with a proposed D-log entry; (vi) G1a and G1b-D3 numbers + CC recommendation. Plus: one row per smoke run in `results/runs.md`; throughput/VRAM plot committed; raw logs under `logs/`.

## 8. Anti-goals (spec §16 + M1 scope)

No variant B / down-pass; no content-based segmentation (axis C); no θ sharing across layers; no FFN inside nodes; no inference-time stochasticity; no KV-cache tricks, quantization, or custom kernels beyond what G1a needs; no complexity claims beyond spec §10; **no edits to `docs/*` or `paper/*`** (report conflicts instead); no training-quality conclusions from smoke runs; no contingency flags enabled silently (`summary_pos: virtual`, `pool_own_proj`, `p1_diversity_loss > 0`).

## 9. Definition of done

All 7 criteria executed and reported in `results/M1-report.md` (pass **or** documented fail), smoke runs ledgered with pre-committed configs, plot artifacts committed, dependencies pinned. Gate decisions (G1a, G1b-D3) and D-log entries then happen on Daniel's side; interpretation of the report against predictions (P-A…P-C) happens in the Claude.ai project.
