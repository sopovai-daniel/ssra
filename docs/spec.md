# SSRA v2 — Implementation Specification

**Status:** v1.0 — **APPROVED, Gate G0 passed 2026-06-11** (draft commit 7dd958f, G0 logged 7ed62aa). All §18 micro-decisions closed; **G1b-D3 threshold X = 5 %** (set by Daniel 2026-06-11).
**Authority:** Once approved, this file is the **single source of truth for implementation** (M1+). Design rationale and history live in `docs/00`–`03`; on conflict regarding *what to build*, this spec wins. On conflict regarding *project state/decisions*, `docs/00` D-log wins.
**Language:** English (feeds the Zenodo technical note and Claude Code implementation). Epistemic markers follow project convention: [OVERENÉ] / [HYPOTÉZA] / [ŠPEKULÁCIA].
**Traceability:** Every normative choice cites its D-log entry (D1–D6, Q1–Q5) in `docs/00`. Spec-level micro-decisions not covered by the D-log are consolidated in §18 (veto register).

---

## 1. Scope

In scope (v2 PoC): the SSRA-mix sublayer (up-pass tree + Fenwick read-out, variant A), three Pool_φ operators (P1 default, P2 control, P3 challenger) behind one interface, hybrid P1×P3 configuration, verification test suite (M1), ablation configuration surface (M3), complexity statements with derivations.

Out of scope (v2): variant B / down-pass (follow-up SSRA-TD, post-G2), content-based segmentation (axis C, post-PoC), θ sharing across layers (ablation post-G2), FFN inside nodes (contingency ablation only), any inference-time noise (D5). See §16.

## 2. Notation

| symbol | meaning |
|---|---|
| B, N, d, h | batch size, sequence length, model dim, attention heads (d_h = d/h) |
| t | token position, 1-indexed |
| k | tree arity; **k = 2 default** (Q2); k = 4 ablation |
| ℓ | tree level; tokens are level-0 nodes; L_tree = ⌈log₂ N⌉ |
| u = (ℓ, j) | node at level ℓ, index j ≥ 1; span(u) = [(j−1)·2^ℓ + 1, j·2^ℓ] (1-indexed, inclusive) |
| m | summary budget; **m = 16 default** (D3) |
| M(ℓ) | capacity schedule: fixed M(ℓ) = m (default) or linear M(ℓ) = m₀ + g·ℓ (ablation, D3) |
| s_ℓ | summaries emitted at level ℓ: **s_ℓ = min(2^ℓ, M(ℓ))**; s_0 = 1 |
| S_u ∈ R^{s_ℓ × d} | summary block of node u |
| w | local read-out window; **w = 64 default**, ablation {32, 128} (D4) |
| θ | shared attention parameters of a layer (W_Q, W_K, W_V, W_O; h heads) |
| φ | pooling parameters of a layer (operator-specific, §5) |
| e_ℓ ∈ R^d | learned level embedding, ℓ ∈ {0..L_max} (D2) |
| Z = LN₁(x) | normalized sublayer input (pre-norm), z_t its t-th row |

All sharing scopes are **per layer**: θ_i, φ_i, e_{i,·} are owned by layer i and shared across all tree levels, all nodes, and the read-out within that layer (D-log 2026-06-10, "maximal axis A"). Nothing is shared across layers in v2 (Q4).

## 3. Layer architecture

SSRA replaces the attention sublayer of a standard pre-norm Transformer block (Q4):

```
x ← x + SSRA_mix_i(LN₁(x))      # this spec
x ← x + FFN_i(LN₂(x))           # standard MLP, d → 4d → d, GELU; NOT inside nodes
```

Each layer builds its own tree over its own current activations Z = LN₁(x); summaries are per-layer activations, not persistent state (Q4). Token/positional embeddings at the stack input: token embedding only; **no global absolute positional embedding** — all positional information enters via §6 (slot-RoPE, window RoPE, e_ℓ). Unembedding: tied or untied = config flag.

## 4. Up-pass

```
# per layer; binary tree over Z; level-wise batched (D6)
leaf u = token t (level 0):
    S_u = z_t                                  # s_0 = 1; no e_0 added here

internal node u at level ℓ ≥ 1, children c1, c2:
    X_u  = concat(S_c1, S_c2)                  # n_in = s_{c1} + s_{c2} slots (= 2·s_{ℓ−1} for complete levels)
    X̃_u  = X_u + e_ℓ                           # e_ℓ broadcast-added to every slot (D2)
    H_u  = X̃_u + Attn_θ(LN_node(X̃_u))          # pre-norm residual; HARD REQUIREMENT (D-log 2026-06-09)
                                               # bidirectional MHA over slots (D1); slot-RoPE pos 1..n_in (D2)
    S_u  = Pool_φ(H_u, s_ℓ)                    # §5; identity iff s_ℓ = n_in
```

Normative details:
- **Lossless levels:** with fixed m = 2^a, levels ℓ ≤ a have s_ℓ = 2^ℓ = n_in ⇒ Pool = identity; the first lossy compression occurs at level a+1 (32 → 16 slots for m = 16). [OVERENÉ by arithmetic of the schedule]
- **Ragged sequences (N not a power of 2):** materialize node u **iff span(u) ⊆ [1, N]**. No padding, no partial nodes. The read-out never references a non-materialized node (proof in §8). Level ℓ holds ⌊N / 2^ℓ⌋ nodes.
- **Level-wise batching (D6, hard requirement):** all nodes of level ℓ across the batch are processed as one tensor `[B·n_nodes(ℓ), n_in, d]`. Python-level recursion is prohibited in the training path.
- **Bidirectional attention inside nodes is legal** — causality argument in §7.

## 5. Pool_φ operators

Unified interface: `Pool_φ : R^{n_in × d} × (s_ℓ) → R^{s_ℓ × d}`; identity when s_ℓ = n_in. Operator choice is a config enum `pool ∈ {p1, p2, p3, hybrid}` — one operator per run, same operator at every lossy level (per layer).

### 5.1 P1 — latent-query attention pool (default; anchor #16/#17)
- φ_P1 = { Q_φ ∈ R^{m_max × d} (learned latent queries, init N(0, 0.02)), LN_pool }.
- `S_u = Q[:s_ℓ] + Attn_θ(q = Q[:s_ℓ], kv = LN_pool(H_u))` — **reuses the layer's θ projections** (W_Q/K/V/O); φ adds only the queries + one LN. No RoPE on either side of this cross-attention (latents are position-free; addressing is content-based).
- Rationale: keeps φ small (m·d + 2d ≪ 4d²) — matches the Q4 record "parameter-match with flat is almost free, φ is small" and is the maximal form of axis A (one attention rule everywhere, including pooling). Contingency config `pool_own_proj: true` gives P1 its own W^p_{Q,K,V,O} (off by default; lever if P-C collapse diagnostics fire or attribution demands decoupling).
- Known risk: query collapse → monitored by P-C diagnostics (§14.5); optional diversity regularizer [K — verify formulation in T2 before relying on it] behind config `p1_diversity_loss`.

### 5.2 P2 — strided pairwise merge (control; anchor #9, #19)
- φ_P2 = { W_merge ∈ R^{d × 2d}, b ∈ R^d }.
- `S_u[i] = W_merge · [H_u[2i−1]; H_u[2i]] + b`, i = 1..n_in/2.
- Valid **only** with the fixed-m schedule and k = 2 (output is structurally n_in/2). Config validation must reject P2 × linear-growth schedule and P2 × k=4. The m-schedule ablation (c) therefore runs on P1.

### 5.3 P3 — learned top-(s_ℓ−1) selection + context residual (challenger; anchor #6)
- φ_P3 = { g_φ : R^d → R, a linear scorer (wₛ ∈ R^d, bₛ) }.
- Scores σ = g_φ(H_u) ∈ R^{n_in}. **Selected set:** indices of the top (s_ℓ − 1) scores (ties broken by lower slot index — deterministic). Selected slots are copied **verbatim**: pass-through preserves exact content.
- **Context residual slot (1 slot):** `c = Σ_{i ∉ sel} softmax(σ_rest / τ)_i · H_u[i]` — a temperature-controlled summary of the non-selected slots. No extra projection.
- Output order: selected slots in original slot order, context slot last.
- **Training-only stochasticity/gradients** (config `p3_grad ∈ {ste, gumbel_topk}`): STE default — hard top-k forward, softmax relaxation backward; Gumbel-top-k optional. **Inference is hard, deterministic, noise-free** (D5: no inference-time stochasticity of any kind).
- Stabilization (mandatory when P3 active): (i) load-balance auxiliary loss `λ_lb · KL(p̄ ∥ uniform)` where p̄ = batch-mean selection distribution over slot positions (counters rich-get-richer); (ii) temperature anneal schedule for τ (start high → anneal down; schedule = config). Subject to micro-gate **G1b-D3** (§14.6).

### 5.4 Hybrid P1×P3 (fallback challenger)
Config `hybrid(k_sel)`: k_sel slots via P3 selection (verbatim) + (s_ℓ − k_sel) slots via P1 latent queries. Activated as challenger iff P3 fails G1b-D3 (per `03-poc-plan.md`).

## 6. Positional scheme (D2)

"Shared geometry, coordinate as input":

| location | positional treatment |
|---|---|
| inside node (up-pass) | **relative slot-RoPE**: RoPE applied per-head to Q and K of Attn_θ with positions = slot indices 1..n_in; identical at every level (scale-transportable rule — the H1 bet). Left/right child encoded by slot order. RoPE base = config `rope_base` (default 10000). |
| node input | **e_ℓ** added to all slots (§4). Init **zeros** (ablation OFF ≈ init state). Default ON; ablation (f) OFF = pure scale equivariance test. |
| read-out: query + window keys | RoPE at **absolute token positions** (q at t, key j at j ⇒ relative offsets ≤ w by construction). |
| read-out: summary keys | **NoPE + e_ℓ tag**: K = W_K(S_u + e_{ℓ(u)}), values V = W_V(S_u) **without** e (coordinate modulates addressing, not content — §18 MD-5). e_0 is used here for level-0 Fenwick blocks (single tokens just left of the window) — its only use in the model. |

Fenwick property bonus (recorded in D2): block level ≈ log distance from query ⇒ e_ℓ doubles as a log-scaled distance code ("recent fine, old coarse").

Mixed key types (RoPE-rotated query attending NoPE summary keys): the sink/memory-token precedent (#20, **verified in T2 2026-06-11**) confirms heterogeneous persistent keys inside one softmax (including a learnable dedicated sink token). Nuance from T2: #20 assigns positions *within the cache*, i.e. the directly precedented treatment of special keys is a fixed virtual position, not pure NoPE. Pure-NoPE summary keys therefore carry the marker [HYPOTÉZA], checked in the M1 smoke run. Contingency config `summary_pos ∈ {none (default), virtual}` where `virtual` rotates every summary key at the fixed virtual position t−w−1 (constant phase wrt query); flip only if T2 verification or M1 smoke surfaces a problem — do not enable silently.

## 7. Causality (D1)

Three mechanisms, no causal mask inside nodes:

1. **Local window:** read-out query t attends tokens [max(1, t−w), t] only — banded causal by key-set construction (a band mask implements this in batched training).
2. **Structural gating of summaries:** node u is consumable (by read-out) only if span(u) ⊆ [1, t−w−1]; (by its parent) only when the sibling's span is also complete. Causality is an availability property, not a mask. [OVERENÉ by construction]
3. **Bidirectional attention inside nodes is legal.** Proof: S_u influences the logit at position t only via some ancestor-or-self a with a ∈ Fenwick(t) (§8). Then span(u) ⊆ span(a) ⊆ [1, t−w−1], so every token inside u precedes t. Hence no information flows from any position ≥ t into the prediction at t. ∎ [OVERENÉ by construction]

Verification: shift test + completion test (§14). Lesson encoded from v1.0: a *decreasing training loss is not evidence of causal correctness* (target leakage, `docs/00` T0); only the tests are.

## 8. Read-out (variant A, D4)

For query position t with prefix budget p = t − w − 1:

```
fenwick_blocks(p):                      # standard BIT decomposition; empty if p ≤ 0
    blocks = []
    while p > 0:
        b = p & (−p)                    # largest power of 2 dividing p
        blocks.append(node(level = log2(b), span = [p − b + 1, p]))
        p −= b
    return blocks                       # |blocks| = popcount ≤ ⌊log₂ p⌋ + 1
```

Alignment lemma: each emitted block ends at a multiple of its size and has length 2^ℓ ⇒ it **is** a tree node of §2; and span ⊆ [1, t−w−1] ⊆ [1, N] ⇒ it is always materialized (§4), including possible level-0 blocks (= single token states). [OVERENÉ: standard Fenwick/BIT property]

Key/value set for position t:

```
K_t = { z_j : j ∈ [max(1, t−w), t] }  ∪  { rows of S_u : u ∈ fenwick_blocks(t−w−1) }
y_t = Attn_θ(q = z_t, kv = K_t)        # ONE softmax over heterogeneous keys (#20)
```

- |K_t| ≤ (w+1) + m·(⌊log₂ t⌋ + 1). Window and Fenwick cover [1, t] disjointly and completely.
- **Read-out reuses θ** — the same Attn_θ as in nodes; a token is a level-0 node (maximal axis A). Ablation (e): separate read-out parameters ψ.
- Per-block capacity: m vectors **per block** (the A-vs-B differentiator recorded in D4).
- Training implementation: any batched realization (gather into padded `[B, N, k_max, d]` + mask, or block-sparse kernels) is acceptable **iff** it is logit-equivalent to the per-position definition above — equivalence is what the completion test certifies (§14.2).

## 9. Autoregressive decoding

Per layer, the decoder maintains:

- **W:** ring buffer of the last w+1 token states z (window keys).
- **Node store:** retained summary blocks under the rule below.

**Retention rule [OVERENÉ by derivation — §18 MD-2]:** at time t, retain node u iff
`u ∈ Frontier(t) := fenwick_blocks(t)` (needed to build future parents) **or** `u ∈ Fenwick(t) := fenwick_blocks(t − w − 1)` (needed by the read-out *now*).
The naive rule "free children when the parent forms" is **incorrect**: a parent completes at t = end(parent), but the read-out keeps consuming its children until t = end(parent) + w + 1, because the read-out frontier lags the tree frontier by w+1 positions. Both sets have ≤ ⌊log₂ t⌋ + 1 members ⇒ per-layer state ≤ (w+1)·d + 2·m·d·(⌊log₂ N⌋ + 1) = **O(m·d·log N)** — same class as recorded in `01` §7, with an explicit constant 2 on the log term.

**Per-token step (per layer):**
1. Compute z_t; read-out y_t over W ∪ Fenwick(t) (all retained by the rule).
2. Append leaf t. While the current rightmost node's sibling is complete: form the parent (node attention + Pool over the two retained child blocks). Amortized 1 internal completion per token ((N−1) internal nodes / N tokens); **worst case at t = 2^k: a chain of k sequential completions** — latency spike O(log N), accepted for PoC, flag stands (D4).
3. Evict per the retention rule.

Decode compute per token (amortized): O((w + m·log N + m²)·d) score-ops. [OVERENÉ: `01` §7]

## 10. Complexity (consolidated; per layer, ×L for the model) [OVERENÉ by derivation, `01` §7]

Assumptions: binary tree, pass-through schedule, score-ops accounting.

| quantity | bound | derivation sketch |
|---|---|---|
| up-pass, training | Θ(N·m·d) | below threshold Σ_ℓ c·N·2^ℓ·d < 2cNmd; above threshold ~N/m nodes à 4c·m²·d ⇒ < 4cNmd |
| read-out, training | Θ(N·(w + m·log N)·d) | position t: ≤ (w+1) + m(⌊log₂t⌋+1) keys; sum over t |
| **training total** | **Θ(N·(w + m·log N)·d)** | read-out dominates; with w, m, d constants: **Θ(N log N)** ⇒ G0 criterion met; same class as #2, **not better** |
| activation memory (tree) | O(N·d·log m) | ≈ 5× flat token activations at m=16; score matrices N·(w + m log N) vs N² flat |
| decode state | O(m·d·log N) | retention rule §9; explicit bound (w+1)·d + 2·m·d·(⌊log₂N⌋+1) |
| decode compute / token | O((w + m·log N + m²)·d) amortized | read-out + amortized 1 node completion |
| decode worst-case latency | O(log N) sequential completions at t = 2^k | accepted for PoC; documented limitation |

Honest accounting (carry into the paper): QKVO projections add Θ(N·d²) terms — class-neutral, but at d ≈ 512, N = 8k they are of the same order as score-ops; the wall-clock advantage vs flat is decided **only** by the M1 throughput curve (G1a). Do not promise more.

## 11. Regularization (D5)

- v1.0 additive noise: **removed entirely** (design + inference). "Stochastic resonance" stays in FIKCIA.
- Training-only dropout: attention-probability dropout + residual dropout; rates are config, tuned in M2.
- P3 auxiliary load-balance loss and temperature anneal per §5.3 (training only).

## 12. Parameter inventory (per layer)

| tensor | shape | scope / sharing |
|---|---|---|
| W_Q, W_K, W_V, W_O (θ) | 4 × d×d | node attention + read-out + P1 pooling cross-attention (ablation e: duplicate ψ for read-out) |
| LN₁, LN₂ | 2 × 2d | sublayer pre-norms (standard) |
| LN_node | 2d | node input norm, all levels |
| e_ℓ | (L_max+1) × d | up-pass node input (ℓ ≥ 1) + read-out summary-key tag (ℓ ≥ 0); init zeros |
| FFN | 8d² + bias | standard sublayer (not in nodes) |
| φ_P1 | m_max·d + 2d | latent queries + LN_pool |
| φ_P2 | 2d² + d | merge matrix |
| φ_P3 | d + 1 | scorer |

L_max = ⌈log₂ N_max⌉ from config. Parameter-match note for baselines: SSRA overhead vs flat at equal d, h, L is e_ℓ + LN_node + φ — at d = 512, m = 16, N_max = 32k: ≈ (16·512) + (15·512) + (2·512) + (2·512) ≈ 26k params/layer, ≪ 1% of a layer. [OVERENÉ by arithmetic]

## 13. Configuration surface

Fixed by this spec (changing them = D-log change): tree arity default k=2; schedule default fixed m=16; w default 64; pre-norm residual + LN_node in every node; level-wise batching; pool default P1; read-out shares θ; e_ℓ default ON; no global positional embedding; no inference-time stochasticity.

Tunable without spec change (M2/M3 territory): d, h, L, vocab, N (train ctx), dropout rates, lr/schedule/optimizer, p3 {τ schedule, λ_lb, ste|gumbel}, rope_base, tied embeddings, ablation enums below.

```yaml
model:
  d: int          h: int          n_layers: int     vocab: int
  n_max: int      # sizes e_ℓ table (L_max)
  m: 16           m_schedule: fixed | linear   # linear: {m0: int, g: int}; P1 only
  w: 64           # {32, 64, 128}
  k: 2            # {2, 4}
  pool: p1 | p2 | p3 | hybrid    # hybrid: {k_sel: int}
  pool_own_proj: false           # P1 contingency
  p1_diversity_loss: 0.0         # [K] verify before use
  summary_pos: none | virtual    # default none; see §6
  level_emb: on | off
  readout_params: shared | separate
  rope_base: 10000
  dropout_attn: float            dropout_resid: float
  tied_embeddings: bool
p3:
  grad: ste | gumbel_topk
  tau_start: float   tau_end: float   tau_anneal_steps: int
  lambda_lb: float
```

Config validation rules: reject (P2 ∧ m_schedule=linear), (P2 ∧ k=4), (hybrid ∧ k_sel ≥ s_ℓ at any lossy level), (summary_pos=virtual without explicit override flag).

## 14. Verification tests (M1 acceptance; run before any training conclusions)

1. **Shift test (causality).** Random x; for t ∈ {2, w−1, w, w+1, w+2} ∪ {2^j, 2^j ± 1 : 2^j ≤ N} ∪ {N−1}: perturb x_t, assert max |Δlogits[1..t−1]| ≤ atol. Proposed atol = 1e−4 (fp32). Rationale for position set: window edge + every Fenwick merge boundary.
2. **Completion test (Fenwick + causality, catches §9 retention bugs).** For N ∈ {257, 1000} (non-powers crossing several 2^k): logits from incremental decoding ≡ logits from the full batched forward at **every** position; atol = 1e−4 (fp32). This is the equivalence certificate of §8's implementation freedom.
3. **Gradient flow check.** One backward pass: every parameter group (θ, φ, e_ℓ for materialized levels, embeddings, FFN, LNs) has nonzero grad; additionally ∂loss/∂S_root ≠ 0 path exists (deep-summary reachability).
4. **Throughput/VRAM curves.** Forward+backward wall-clock and peak memory vs N ∈ {1k, 2k, 4k, 8k} (16k if VRAM allows), SSRA vs flat baseline, same d, h, L. **G1a criterion (proposal, §18 MD-7):** log-log slope of SSRA wall-clock ≤ 1.5 over the range, and strictly below flat's slope. Plot artifact committed to repo.
5. **P-C collapse diagnostics (P1).** Logged during any training: entropy of Q_φ attention maps + per-query participation rates. Thresholds informative, not gating (M3 prediction P-C).
6. **G1b-D3 (P3 stability micro-gate).** On the 10M smoke run: P3 loss within **X = 5%** of P1 (relative), without divergence, after standard stabilization (§5.3). X set by Daniel 2026-06-11 (D-log). Fail ⇒ P3 → appendix, hybrid takes the challenger slot.
7. **P3 determinism.** Two inference passes, identical seeds-independent outputs (bitwise).

## 15. Ablation registry (M3; axes fixed in `03`)

| id | axis | values | spec ref |
|---|---|---|---|
| a | cross-scale weight sharing | on / off (per-level θ copies) | §2 — core H1 test |
| b | Pool_φ | P1 / P2 / P3 / hybrid | §5 |
| c | m schedule | fixed 16 / m₀+g·ℓ (P1 only) | §5.2 constraint |
| d | window w | 32 / 64 / 128 | §8 |
| e | read-out params | shared θ / separate ψ | §8 |
| f | level embedding | on / off | §6 |
| g | arity k | 2 / 4 | §2; needle effect [HYPOTÉZA], see `01` §6.2 |
| h | depth per (g) | derived | §2 |

Post-G2 only (not in the main run): θ sharing across layers (UT × scale), FFN-in-node, `pool_own_proj`, SSRA-TD (variant B).

## 16. Explicit non-goals (v2)

No down-pass / top-down state (variant B → SSRA-TD post-G2; carries H3). No content-based boundaries (axis C). No claim of a better complexity class than Log-Linear Attention (#2) — the contest is mechanism + recall at matched compute, stated verbatim in the paper. No KV-cache tricks, no quantization, no kernel engineering beyond what G1a needs.

## 17. Novelty thesis v1 (publication wording; supersedes v0.2 in `02`)

**Claim.** SSRA is a causal language-modeling block in which a single weight-shared rule — one softmax attention block Attn_θ and one learned pooling operator Pool_φ per layer — generates the entire scale hierarchy of the sequence. The same (θ, φ) applies at every level of a binary tree over the sequence, and the same θ performs the token-level read-out: a token is a level-0 node. Two properties complete the mechanism: (i) learned, discrete-slot compression at every node (m summary vectors via latent-query attention or hard top-m selection), and (ii) fully bidirectional attention inside nodes that remains legal in a causal LM because causality is enforced structurally — a node becomes consumable only after its span is complete — rather than by masking.

**Delimitation.** The prefix decomposition used by the read-out is the standard Fenwick (binary indexed tree) structure shared with Log-Linear Attention [#2]; we claim no novelty for it. Relative to #2, the mechanism differs in all three remaining components: softmax attention vs. linear-attention kernels; learned discrete-slot pooling vs. structural matrix state with λ-decay; and cross-scale parameter sharing that includes the read-out. Relative to hierarchical and multiresolution models [#3, #8, #9], SSRA shares one rule across all levels instead of per-level parameters in fixed 2–3-level stacks; relative to weight sharing in depth [#10, #11], sharing here is across sequence scale. PMA/Perceiver-style pooling [#16, #17] anchors the P1 operator; its use as the node compressor of a causal scale hierarchy with queries shared across all nodes and scales is, to our knowledge, new. The nearest combinations found in the T2 verification and the 2026-06-11 novelty scan are delimited explicitly: GPST [#21] shares a learned composition function over an induced *syntactic* tree inside a generative LM, but composes constituents to single vectors, keeps the composition model separate from the generative model, and makes no long-context or retrieval claim; MANO [#22] shares convolution kernels and attention weights across scales, but for encoder-style vision/physics grids without causality, slot summaries, or a prefix read-out; PSMs [#23] obtain softmax-like operators with O(log N) decoding state via prefix scans over chunk encodings, without cross-scale rule sharing or learned per-node slot compression. No work among #1–#25 simultaneously combines: cross-scale weight sharing × learned m-slot node compression × causal LM over text.

**Falsifiability.** The thesis is decided by two pre-registered ablation axes at matched compute: if cross-scale sharing does not help (ablation a), SSRA degenerates into a slower relative of H-Transformer-1D [#3]; if learned node compression loses retrievable detail (multi-needle suite, prediction P-B), SSRA loses to Log-Linear Attention [#2] on recall. Either outcome answers H1–H2 and is publishable.

*(References #n = `docs/02-prior-art-mapa.md`. T2-subset verification completed 2026-06-11: rows #3–#5, #8–#13, #16–#25 carry primary-source URLs + retrieval dates; remaining [K]-level details — e.g. the P1 diversity-loss formulation in §5.1 — stay flagged until verified.)*

## 18. Spec-level micro-decisions (veto register)

Choices made while drafting this spec, inside the boundaries of closed D-log items. Veto-based regime applies: each stands unless Daniel vetoes it by next message.

| id | decision | rationale | alternative |
|---|---|---|---|
| MD-1 | spec.md in English | feeds Zenodo note + CC; avoids double translation | Slovak + translation pass pre-Zenodo |
| MD-2 | decode retention rule = Frontier(t) ∪ Fenwick(t−w−1); constant 2 on the log term | naive child-freeing breaks read-out for w+1 steps after each merge; [OVERENÉ] §9 | none correct found; refinement, class unchanged |
| MD-3 | P1 reuses Attn_θ projections; φ_P1 = {Q_φ, LN_pool} | "φ is small" (Q4 record) + maximal axis A | `pool_own_proj: true` (config exists) |
| MD-4 | e_ℓ parameters shared between up-pass input and read-out key tag; e_0 read-out-only; init zeros | one coordinate system, ablation-friendly init | separate read-out e'_ℓ table |
| MD-5 | read-out: e_ℓ added to summary **keys only**, not values | coordinate modulates addressing, not content | add to K and V |
| MD-6 | window = [t−w, t] inclusive ⇒ w+1 token keys; Fenwick covers [1, t−w−1] | exact disjoint cover of [1, t] | window excl. self (w keys) |
| MD-7 | test tolerances atol 1e−4 fp32; G1a slope ≤ 1.5 (N 1k–8k) | concrete, falsifiable; slope expected 1.0–1.3 given QKVO terms [HYPOTÉZA] | Daniel sets other numbers |
| MD-8 | P3 context slot = score-softmax-weighted sum, no extra projection; ties → lower index | minimal φ, deterministic | learned W_ctx projection |
| MD-9 | P2 restricted to fixed-m × k=2 | structural (pairwise halving) | generalized P2 (rejected: stops being a clean control) |
| MD-10 | G1b-D3 X = **5%** | needs a number to be falsifiable | **closed — confirmed by Daniel 2026-06-11** |

## 19. Gate G0 self-check

| G0 criterion (`03`) | status | where |
|---|---|---|
| causal design | ✔ band window + structural gating + proof | §7 |
| positional design | ✔ slot-RoPE + e_ℓ + heterogeneous keys | §6 |
| compression design | ✔ P1/P2/P3 + schedule, unified interface | §5 |
| derived complexity ≤ O(N log N) | ✔ Θ(N·(w + m·log N)·d) [OVERENÉ] | §10 |
| defined decode state | ✔ O(m·d·log N) with explicit retention rule | §9–§10 |

Formal G0 passage = Daniel approves this document. **Approved 2026-06-11** — D-log entry in `docs/00`; next: Zenodo stage-1 technical note per `03`.
