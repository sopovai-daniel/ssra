# SSRA: Scale-Shared Recursive Attention — Empirical Evaluation and Negative Results

**Full paper (stage 2 of 2).**
**Author:** Daniel Sopov (SopovAi). ORCID: [0009-0004-8584-5156](https://orcid.org/0009-0004-8584-5156).
**Status:** DRAFT v0.4, 2026-07-18 — full text drafted (B1–B3: Abstract, §1–§8, AI disclosure, captions); Sunday reviewer pass (B4) pending; do not cite.
**DOI:** TODO — reserve a **new** Zenodo record (own DOI) before PDF export; this is not a new version of the stage-1 record.
**Prior version:** SSRA technical note v1.0, DOI [10.5281/zenodo.20647034](https://doi.org/10.5281/zenodo.20647034) — cited as prior version; design and complexity analysis fixed there before any training run.
**License:** text CC BY 4.0; reference implementation Apache-2.0 (repo public flip on publication day).

---

## 0. INTERNAL — drafting rules (binding; delete this section before PDF export)

Framework: HO-20 §2–§4 (binding), D-log 2026-07-17 (GO entry), spec v1.2 §16, docs/00 forbidden vocabulary.

1. **Number provenance (Rule W):** every quantitative claim carries an inline tag `[src: <file> §<sec>]` during drafting. No number from memory. Tags are resolved (each number re-checked against the source report) and stripped during the Sunday reviewer pass. Sources of record per section are listed in the source map below.
2. **Binding formulations (HO-20 §4):**
   - Length extrapolation: **"flat prior confirmed; SSRA prior violated"** — never "both violated". SSRA prior (stable/mild) violated from N = 2,048.
   - H2: **no positive evidence; not tested in its M3 form.** Pre-registered caveat sentence applies: an 85M model at 850M tokens need not exhibit copy behavior; the needle suite is informative, not decisive. Beyond N = 1,024 the needle result is confounded by the positional collapse measured in §4.7.
   - Cross-model ppl ratio 21.35 → 19.06 at the top of the length grid is reported **without interpretation**.
3. **Forbidden vocabulary** (docs/00): "new paradigm", "cognitive system", "consciousness", "stream of thought", "tensor intelligence", "revolutionary". Mechanisms in neutral terms only (block, module, operator, complexity, memory).
4. **No architectural conclusions** from loss/ppl gaps (spec §16). The lr-stability finding is an empirical observation with **mechanism undetermined**; sweep losses are never read across models.
5. **No new measurements, no spend** (HO-19 §4 / HO-20 §2 — applies to 0-EUR local runs too). Figures = existing committed artifacts only; new composite figures are v1.1 scope (D-log 2026-07-17, post-publication exploitation). A figure gap escalates to the author; fallback is a table from existing raw data files.
6. **AP-8 honesty note** wherever "matched" appears: matched **parameters + tokens** (identical token stream, order, steps, seed); FLOPs and wall-clock are **reported, not matched**.
7. **Erratum (R5)** is in-paper: §2.8 explicitly corrects the point retention rule of note v1.0 §2.6/§3.
8. Language EN throughout; note v1.0 terminology and notation reused verbatim where possible.

### Source map (sections → artifacts of record)

| Paper § | Primary sources |
|---|---|
| §2 | `paper/technical-note.md` (§2–§5) + `docs/spec.md` v1.2 (§3–§13; corrected §9 retention rule) |
| §3 | `docs/cc/M2-assignment.md` (AP-8/AP-9), `results/M2-phase0-report.md` (data, tokenizer), `results/runs.md` (ledger), docs/00 |
| §4.1 | `results/M1-report.md` |
| §4.2 | `results/M2-sweep.md` |
| §4.3 | `results/M2-core-pair.md` (incl. §xi C1–C6) + `results/M2-spike-diagnostics.md` (T5) |
| §4.4–4.5 | `results/M2-core-pair-lr6e4.md` |
| §4.6 | `results/M2-recalibration.md` + `results/M2-calibration.md` — **repo read required (not in project files)** |
| §4.7–4.8, §5.2–5.5 | `results/M2-g2lite.md` (+ raw JSON/CSV in results/ and GCS mirror) |
| §5.1 | P-C rows in `results/M1-report.md`, `M2-sweep.md`, `M2-core-pair.md`, `M2-core-pair-lr6e4.md` |
| §7 | `docs/02-prior-art-mapa.md` (#1–#25) + note §6 |
| Abstract, §1, §8, captions | no independent sources — recap of values already tagged in §2–§6; B4 cross-checks them against the same reports |

---

## Abstract

SSRA (Scale-Shared Recursive Attention) is a causal language-modeling block in which a single weight-shared rule per layer — one softmax attention block and one learned pooling operator, applied at every level of a binary tree over the sequence and reused by the token-level read-out — replaces the attention sublayer, at Θ(N·(w + m·log N)·d) training cost per layer: the same complexity class as Log-Linear Attention, and explicitly not a better one [src: §2.7]. A technical note (the stage-1 record of this work) fixed the design, its causality proof, its derived complexity, and a falsification plan before any training run; this paper executes that pre-registered plan, and the outcome is negative on every measured axis. The comparison is SSRA in its P1 (latent-query pooling) configuration against a flat Transformer baseline at matched parameters (≈ 84–85M [src: results/M2-core-pair.md §0.2]) and matched tokens (850M; identical token stream, order, step count, and seed), with FLOPs and wall-clock reported, not matched [src: docs/cc/M2-assignment.md AP-8]. Parity: SSRA finishes +10.22 % above the baseline on validation perplexity at the training context length 1,024 (26.860 vs 24.369), outside the pre-registered ±5 % band [src: results/M2-core-pair-lr6e4.md §iv]. Stability: at the sweep-selected learning rate 1e-3 the SSRA arm suffered an unrecovered finite loss spike while the flat arm was clean on the identical token stream [src: results/M2-core-pair.md §iv]; the single permitted retune to 6e-4 — the only changed variable — trained cleanly in both arms, implicating the learning rate and leaving SSRA with an empirically narrower stable learning-rate range at this scale, mechanism undetermined [src: results/M2-core-pair-lr6e4.md §iv]. Length extrapolation (inference-only, N = 1,024 … 32,768): the flat baseline's pre-registered degradation prior is confirmed; SSRA's stable-to-mild prior is violated from N = 2,048; there is no crossover at any measured length [src: results/M2-g2lite.md §O2–O4]. Retrieval (needle-lite passkey): SSRA scores 0 % in every length × depth cell, including its own training length, while the flat baseline shows depth-local copy behavior at the training length only; a pre-registered caveat records that an 85M-parameter model trained on 850M tokens need not exhibit copy behavior at all [src: results/M2-g2lite.md §M2/§O5]. Measured training-throughput constants on A100-class hardware favor the baseline by 11.8× at the reduced sweep scale (S1) and 11.1× in the production runs (S2), so the equal-token protocol favored SSRA on the compute axis [src: results/M2-recalibration.md §3; results/M2-core-pair-lr6e4.md §iii]. The repository releases the frozen implementation specification, the append-only decision log, the pre-registered assignments, per-run configurations, raw logs, and the verification suite. Either outcome was declared publishable in advance; this is the negative one.

## 1. Introduction

Natural language exhibits self-similar, long-range-dependent statistics: Alabdulmohsin et al. estimate a Hurst parameter of H ≈ 0.7 for language and connect this fractal structure to next-token prediction [1]. Standard Transformers do not encode this property architecturally: every layer attends over a flat token sequence, and the hierarchical variants surveyed in §7 typically introduce *per-level* parameters over a small, fixed number of levels. SSRA tests a sharper inductive bias: **if language is self-similar, one rule should suffice at every scale.** Concretely, a single attention block and a single learned compressor per layer process pairs of tokens, pairs of summaries of pairs, and so on up a binary tree — and the very same attention block performs the final token-level read-out (§2).

The design operationalizes two hypotheses, restated from the technical note [26] §1:

- **H1 (scale-shared rule):** cross-scale weight sharing is an inductive bias matched to the self-similarity of language [1], yielding better length extrapolation at equal parameter budget than a flat baseline.
- **H2 (learned node compression):** m-slot learned compression at tree nodes preserves retrieval ability (measured by needle-in-a-haystack tests) at Θ(N·(w + m·log N)·d) training cost.

The work is published in two stages. The technical note [26] fixed the design, the causality proof, the derived complexity, and the falsification plan before any training run, and declared in advance that either outcome — positive or negative — answers the hypotheses and is intended for publication. This paper is the second stage: it reports the pre-registered evaluation. The outcome is negative on every measured axis (§4): H1's length-extrapolation prediction is contradicted by the measurements of §4.7 [src: results/M2-g2lite.md §O2–O4], and H2 obtains no positive evidence in the needle-lite form actually run (§4.8) and was not tested in its planned multi-needle M3 form, because the pre-registered gate before that stage failed (§6).

Contributions:

1. **Executed pre-registered plan** at matched parameters + matched tokens (identical token stream, order, steps, and seed; FLOPs and wall-clock reported, not matched — §3.2), with the falsification criteria fixed in [26] before any run.
2. **Negative headline results, stated plainly** (§4): a +10.22 % parity gap at the training context [src: results/M2-core-pair-lr6e4.md §iv]; a training instability at lr 1e-3 removed by a single-variable retune to 6e-4, leaving SSRA with an empirically narrower stable learning-rate range at this scale, mechanism undetermined [src: results/M2-core-pair-lr6e4.md §iv]; no length-extrapolation crossover at any measured N [src: results/M2-g2lite.md §O4]; 0 % needle-lite retrieval for SSRA in every cell [src: results/M2-g2lite.md §M2].
3. **Descriptive diagnostics** (§5), characterizing without mechanism claims: pooling-attention uniformity across all runs, exactly-zero level-embedding rows at extrapolation lengths, a bf16 position-quantization artifact shared by both models, and the positional locality of the degradation.
4. **Public methodology artifacts** (§3.6): the frozen implementation specification, the append-only decision log, the pre-registered assignments committed before each run, one committed configuration per run, raw training logs, and the verification suite.

The paper additionally corrects one error of [26] found during implementation: the decoder retention rule of the note's §2.6/§3 is amended by the in-paper erratum of §2.8; the headline decoding class is unaffected.

## 2. Mechanism

This section restates the design fixed before any training run in the technical note [26], together with the one correction found during implementation (§2.8). The implementation specification (v1.2, in the repository) is the normative reference; this summary is self-contained for reading §4–§5. The configuration described here is the one trained in §4 (P1 pooling, default hyperparameters); the registered alternatives — the P2/P3 pooling operators, the P1×P3 hybrid, and the ablation axes — are defined in [26] and the specification and were not exercised in the runs reported in this paper.

SSRA replaces the attention sublayer of a standard pre-norm Transformer block:

```
x ← x + SSRA_mix_i(LN1(x))      # this paper
x ← x + FFN_i(LN2(x))           # standard MLP (d → 4d → d, GELU); not used inside tree nodes
```

Each layer i owns one attention parameter set θ_i (W_Q, W_K, W_V, W_O; h heads), one pooling parameter set φ_i, and a learned level-embedding table e_ℓ. All three are **shared across all tree levels, all nodes, and the read-out within that layer**; nothing is shared across layers. Each layer builds its own binary tree over its own current activations; summaries are per-layer activations, not persistent state.

Notation: sequence length N; model width d; tree level ℓ (a token is a level-0 node); a node u at level ℓ spans 2^ℓ consecutive tokens; summary budget m = 16; per-level summary count s_ℓ = min(2^ℓ, m); local read-out window w = 64; tree arity k = 2.

### 2.1 Up-pass (tree construction)

```
leaf u = token t:                S_u = z_t                       # s_0 = 1
internal node u (children c1, c2, level ℓ ≥ 1):
    X_u  = concat(S_c1, S_c2)                                    # ≤ 2m input slots
    X̃_u  = X_u + e_ℓ                                             # level coordinate as input
    H_u  = X̃_u + Attn_θ(LN_node(X̃_u))                            # pre-norm residual;
                                                                 # bidirectional over slots;
                                                                 # relative slot-RoPE (positions 1..n_in)
    S_u  = Pool_φ(H_u, s_ℓ)                                      # identity while s_ℓ = n_in
```

Pre-norm residual paths and a LayerNorm in every node are hard requirements: without an identity path, log N successive rewrites destroy both verbatim signal transport and gradient flow. With fixed m = 2^a, levels ℓ ≤ a are lossless (Pool = identity); the first lossy compression occurs at level a+1 (32 → 16 slots for m = 16). For ragged N, a node is materialized iff its span lies inside [1, N]; no padding, no partial nodes. All nodes of a level are processed as one batched tensor (level-wise batching); host-language recursion is prohibited in the training path.

### 2.2 Pooling operator (P1, as trained)

All runs in this paper use **P1**, a latent-query attention pool (PMA / Set Transformer [16]; Perceiver [17]): s_ℓ learned latent queries cross-attend to the node's slots, reusing the layer's θ projections; φ adds only the queries and one LayerNorm. The latent queries are shared across **all** nodes and **all** scales of the tree — the maximal form of the scale-sharing bet. Two further operators (P2, a strided pairwise merge; P3, hard top-m selection with a context-residual slot) and a P1×P3 hybrid are registered behind the same interface in [26]; they were not part of the runs reported here.

### 2.3 Positional scheme: shared geometry, coordinate as input

| location | treatment |
|---|---|
| inside a node | relative slot-RoPE over slot indices 1..n_in — *identical geometry at every level*, which is what makes the rule scale-transportable (the H1 bet); left/right child encoded by slot order |
| node input | learned level embedding e_ℓ added to every slot (initialized at zero) |
| read-out, window keys | RoPE at absolute token positions (relative offsets ≤ w by construction) |
| read-out, summary keys | no rotary phase (NoPE) + e_ℓ added to keys only — the coordinate modulates addressing, not content |

In the Fenwick decomposition (§2.5) the level of a block grows with its distance from the query, so e_ℓ doubles as a log-scaled distance code: recent context is addressed finely, old context coarsely. Mixing rotated token keys with special non-rotated keys in a single softmax follows the attention-sink / memory-token precedent [20], with one recorded caveat: [20] assigns its persistent keys positions *within the cache*, i.e. the directly precedented treatment is a fixed virtual position rather than pure NoPE. Pure-NoPE summary keys are therefore recorded in [26] as a hypothesis with a specified fixed-virtual-position fallback; the fallback was never enabled in the runs of §4.

### 2.4 Causality without masks inside nodes

Three mechanisms, no causal mask inside nodes:

1. **Local window:** the read-out query at position t attends to tokens [max(1, t−w), t] only.
2. **Structural gating:** a node u is consumable by the read-out only once span(u) ⊆ [1, t−w−1], and by its parent only when its sibling's span is also complete. Causality is an availability property of summaries, not a mask.
3. **Bidirectional attention inside nodes is legal.** Proof sketch: a summary S_u can influence the logit at position t only through an ancestor-or-self a that the read-out at t consumes, which requires span(u) ⊆ span(a) ⊆ [1, t−w−1]; hence every token inside u precedes t, and no information flows from positions ≥ t into the prediction at t. ∎ The enumeration of paths is exhaustive: the only cross-node operation in the up-pass is parent composition — attention, normalization, and pooling all act within a single node.

A lesson encoded from an earlier prototype of this project: a decreasing training loss is *not* evidence of causal correctness (target leakage produces the same curve); only explicit tests are. Two are part of the verification suite and pass on the trained implementation: a **shift test** (perturbing token t must not change any logit at positions < t) and a **completion test** (incremental decoding must reproduce the full batched forward within tolerance at every position). [src: results/M1-report.md; spec §14]

### 2.5 Read-out (Fenwick prefix decomposition)

For query position t with prefix budget p = t − w − 1, the prefix [1, p] is decomposed by the standard binary-indexed-tree (Fenwick) rule into ≤ ⌊log₂ p⌋ + 1 disjoint blocks, each of which is exactly a tree node (block ends at a multiple of its size; always materialized). The key/value set for position t is

```
K_t = { z_j : j ∈ [max(1, t−w), t] }  ∪  { rows of S_u : u ∈ Fenwick(t−w−1) }
y_t = Attn_θ(q = z_t, kv = K_t)       # ONE softmax over heterogeneous keys
```

so |K_t| ≤ (w+1) + m·(⌊log₂ t⌋ + 1), and window + Fenwick blocks cover [1, t] disjointly and completely. The read-out **reuses θ** — the same attention block as in the nodes; a token is a level-0 node. Per-block capacity is m vectors *per block*, i.e. the macro context available to a query scales as m·log t vectors, not a single m-vector bottleneck. The Fenwick prefix structure itself is taken from Log-Linear Attention [2]; no novelty is claimed for it (§7). Any batched training realization is acceptable iff it is logit-equivalent to this per-position definition; the completion test certifies the equivalence.

### 2.6 Autoregressive decoding and the retention rule (corrected)

Per layer, the decoder maintains a ring buffer of the last w+1 token states and a node store governed by an explicit retention rule: at time t, retain node u (level ℓ, end e = end(u)) iff **t ∈ [e, e + 2^ℓ + w]**. This is the interval closure of the two pointwise demands: fenwick_blocks(t) needs u while future parents are being built (t ∈ [e, e + 2^ℓ − 1]), and the read-out needs u while its span sits just left of the window (t ∈ [e + w + 1, e + 2^ℓ + w]). The naive rule "free children when the parent forms" is incorrect: the read-out frontier lags the tree frontier by w+1 positions, so children remain consumable for w+1 steps after their parent completes. Per-level retained count under the closure is ≤ 2 + ⌈w/2^ℓ⌉, giving per-layer state ≤ (w+1)·d + Σ_ℓ (2 + ⌈w/2^ℓ⌉)·s_ℓ·d = O((m·log N + w·log m)·d) — at fixed w, m this is O(m·d·log N) in N. Amortized, one internal node completes per generated token; the worst case at t = 2^k is a chain of k sequential completions — an O(log N) latency spike, accepted for the proof of concept and documented as a limitation.

### 2.7 Complexity (derived; per layer, ×L for the model)

Score-operation accounting, binary tree, pass-through schedule s_ℓ = min(2^ℓ, m):

| quantity | bound | derivation sketch |
|---|---|---|
| up-pass, training | Θ(N·m·d) | below the lossy threshold Σ_ℓ c·N·2^ℓ·d < 2cNmd; above it ~N/m nodes at 4c·m²·d each ⇒ < 4cNmd |
| read-out, training | Θ(N·(w + m·log N)·d) | position t sees ≤ (w+1) + m(⌊log₂ t⌋+1) keys; sum over t |
| **training total** | **Θ(N·(w + m·log N)·d)** | read-out dominates; with w, m, d constant this is Θ(N log N) — the **same class** as Log-Linear Attention [2], not better |
| activation memory (tree) | O(N·d·log m) | ≈ 5× flat token activations at m = 16; score matrices are N·(w + m·log N) vs N² flat |
| decode state | O(m·d·log N) at fixed w, m; explicit O((m·log N + w·log m)·d) | corrected retention rule, §2.6 |
| decode compute / token | O((w + m·log N + m²)·d) amortized | read-out + amortized one node completion |
| decode worst-case latency | O(log N) sequential completions at t = 2^k | documented limitation |

**Honest accounting.** The QKVO projections add Θ(N·d²) terms. These are class-neutral, but at the widths used here they are of the same order as the score operations. Consequently, any wall-clock relation to a flat Transformer is an *empirical* question decided by measured throughput (§4.1, §4.6), not by the asymptotics above; no wall-clock advantage is claimed, and none was observed.

### 2.8 Erratum to the technical note (v1.0)

The note [26] §2.6 and §3 state the decoder retention rule in pointwise form — retain u iff u ∈ Frontier(t) ∪ Fenwick(t−w−1) — and give the decode-state bound with an explicit constant 2 on the log term. The pointwise rule is incorrect: u ∈ fenwick_blocks(p) holds exactly for p ∈ [e, e + 2^ℓ − 1], so for levels with 2^ℓ ≤ w the Frontier interval and the read-out interval are disjoint, and the rule evicts u inside the gap [e + 2^ℓ, e + w] although the read-out requires it again at t = e + w + 1 — at w = 64 this affects every level ℓ ≤ 6, including level-0 blocks one step after leaving the local window. The fault was found during the first implementation milestone and its correction is certified by the completion test; the corrected interval-closure rule is the one given in §2.6 above. The constant-2 form of the decode-state bound held only for levels with 2^ℓ > w; the corrected explicit bound is O((m·log N + w·log m)·d), ≈ 1.5× the note's estimate at m = 16, w = 64, N = 8k by direct count. The headline decoding class O(m·d·log N) at fixed w, m is unaffected (the w-dependent term is constant in N). This paragraph supersedes the corresponding statements of [26]; the published note file remains unmodified for provenance. [src: D-log 2026-06-12; spec §9 v1.1]

## 3. Experimental setup

### 3.1 Models

Two models at matched scale ("S2"): a **flat Transformer baseline** and **SSRA-P1**, both with d = 640, h = 10 heads, L = 15 layers, vocabulary 16,384, tied embeddings, and identical FFN shape (d → 4d → d, GELU). Parameter counts: flat **84,301,440**, SSRA **84,647,040** — a gap of 345,600 (+0.41 % SSRA), the e_ℓ table + node LayerNorm + φ overhead of §2. [src: experiments/m2-core-*-s2-850m-lr6e4.yaml; results/M2-core-pair.md §0.2]

The flat baseline is a standard pre-norm causal Transformer with RoPE at absolute token positions and no length gating or extrapolation-specific mechanism. [src: baselines/flat.py, verified results/M2-g2lite.md V1] SSRA runs with the §2 defaults: m = 16, w = 64, pool P1, level embeddings on, pure-NoPE summary keys, read-out sharing θ; the e_ℓ table is sized for n_max = 32,768 (levels above the training length are therefore initialized-but-untrained; see §5.2). [src: configs]

### 3.2 Matching protocol

"Matched" in this paper means **matched parameters + matched tokens**: identical token stream, identical batch order, identical step count, and a shared seed (1337) fixing both initialization draws and data order. FLOPs and wall-clock are **reported, not matched** — SSRA consumes substantially more compute per token than the flat baseline at these sizes (§4.6), so equal-token comparisons favor SSRA on the compute axis; this is stated rather than corrected. [src: docs/cc/M2-assignment.md AP-8]

### 3.3 Data and provenance

Corpus: **FineWeb-Edu**, config `sample-10BT`, Hugging Face hub revision `87f09149…`, license ODC-By v1.0 [29]. Documents are split deterministically and document-disjointly by hash (sha1(doc.id) mod 1000 < 50 → validation, ≈ 5 %). The tokenizer is a byte-level BPE with vocabulary 16,384, trained only on training-side documents (document-disjoint from validation) and frozen (sha256 `019568a2…`); shards are packed uint16 streams guarded by sha256 gates checked before any training step. Sizes: train 913,605,620 tokens; validation 48,050,671 tokens. The **evaluation set** `val-eval-2M` is the first 2,000,000 tokens of the validation shard (byte-verified prefix). The **extrapolation region E** used in §4.7 is `val.bin`[2,000,000, 2,000,000 + 2^21) — disjoint from the evaluation set by construction; E contains 1,840 complete documents, mean length 1,138.7 tokens (median 666, p90 2,152). [src: results/M2-phase0-report.md; D-log 2026-07-13; results/M2-g2lite.md V4]

### 3.4 Training protocol

Each arm trains for 51,880 steps × batch 16 × sequence length 1,024 = **850,001,920 tokens** (≈ 10× parameters). Optimizer: AdamW (β₁ = 0.9, β₂ = 0.95, weight decay 0.01) with global gradient-norm clipping at 1.0; learning-rate schedule: linear warmup 778 steps (≈ 1.5 % of steps, fraction carried over from the sweep) then cosine decay to a floor of 0.1× peak. Precision: bf16 autocast with fp32 loss/eval accumulation. Batches are drawn from the packed stream by a seeded generator; with seed and step count fixed, both arms consume the identical token stream in the identical order. Checkpoint/resume is bit-for-bit verified. [src: scripts/train.py; experiments/*.yaml; results/M2-phase0-report.md]

Validation loss is computed every 200 steps on fixed-seed batches from the validation shard (8 batches, seed fixed across the pair). The final metric is a deterministic full pass over `val-eval-2M` in non-overlapping 1,024-token windows: 1,953 windows / 1,999,872 tokens scored / 127 tail tokens dropped — byte-identical protocol for both arms. [src: results/M2-core-pair-lr6e4.md §iv]

Pre-registered decision criteria: (i) **parity gate** — SSRA within ±5 % of the flat baseline on validation perplexity at context length 1,024, plus stable training; (ii) at most **one retune iteration**, changing exactly one variable. Two enumerated in-flight abort rules exist in the harness: any non-finite train/val loss aborts immediately; and a sustained-instability rule (validation loss above the run's best by > 2.0 nats on ≥ 6 consecutive evaluations, i.e. 1,000 steps) checkpoints and aborts. The sustained-instability rule was introduced after the §4.3 run — which ran with the non-finite rule only — and was active in §4.4, where neither rule fired. [src: docs/cc/M2-phase3b-retune.md; scripts/train.py]

### 3.5 Hardware and cost

Training runs executed on a single rented A100 SXM 80 GB (on-demand); the inference-only length/needle evaluation of §4.7–4.8 on a single rented RTX A6000 48 GB. Total paid compute across all training, calibration, sweep, and evaluation phases reported here: **72.37 EUR** of a 300 EUR budget cap (itemized append-only ledger in the repository). [src: results/runs.md; D-log 2026-07-17]

### 3.6 Artifacts and reproducibility

The repository (code Apache-2.0) is public from the publication date and contains: the frozen implementation specification, an append-only decision log, the pre-registered assignments committed before each run, one committed YAML per run (committed before launch), raw JSONL training logs including per-step diagnostics, the verification test suite, and this paper's source. Checkpoints and raw evaluation outputs are mirrored in object storage. Repository: https://github.com/sopovai-daniel/ssra (tag TODO(B5)). The technical note [26] is the prior version of record for the design.

## 4. Results

Results are presented in execution order: the implementation-scale shape measurement (§4.1), the hyperparameter sweep that fixed the training configuration (§4.2), the core pair at the selected learning rate (§4.3), the single permitted retune (§4.4) and the stability observation it isolates (§4.5), the measured throughput and memory constants (§4.6), and the inference-only length-extrapolation (§4.7) and retrieval (§4.8) measurements on the final §4.4 checkpoints. Headline outcomes are negative on every pre-registered axis.

### 4.1 Scaling shape (implementation verification scale)

Before any paid training, the wall-clock scaling shape of the implementation was measured at a small verification scale: d = 192, h = 8, L = 2 (≈ 0.95 M parameters per model), batch 1, fp32, on an Apple-silicon laptop GPU (MPS backend), N ∈ {1,024 … 8,192} — the largest swap-free range on that machine; candidate sizes whose N = 8,192 point paged to swap were rejected, because a paging measurement measures swap, not compute [src: results/M1-report.md §(vi-a)].

**Table — forward+backward wall-clock and peak memory per step (B = 1, fp32, MPS).**

| N | SSRA fwd+bwd (ms) | flat fwd+bwd (ms) | SSRA peak (GiB) | flat peak (GiB) |
|---|---|---|---|---|
| 1,024 | 510.8 | 46.3 | 1.17 | 1.12 |
| 2,048 | 1,053.9 | 160.7 | 2.24 | 1.21 |
| 4,096 | 1,985.1 | 523.6 | 4.31 | 3.08 |
| 8,192 | 4,008.9 | 2,654.1 | 7.77 | 11.14 |

[src: results/M1-report.md §(vi-a)] (Figure F1: `results/M1-throughput.png`.)

The fitted log-log wall-clock slopes are **0.983 for SSRA and 1.923 for the flat baseline** [src: results/M1-report.md §(vi-a)]. This is strictly a shape measurement: it is consistent with the sub-quadratic Θ(N·(w + m·log N)·d) class of §2.7 against the baseline's quadratic behavior, and the sub-unit SSRA slope reflects fixed per-level dispatch overheads that still dominate at N = 1,024 — it must not be read as linear scaling. It is also not a speed claim: SSRA's absolute wall-clock is 1.5–11× above flat across the measured range, with a crossover just above N = 8,192 that is specific to this hardware and backend [src: results/M1-report.md §(vi-a)]. At N = 8,192 SSRA's peak memory is already below flat's (7.77 vs 11.14 GiB) on this backend, where flat's quadratic score matrices dominate the backward pass [src: results/M1-report.md §(vi-a)].

### 4.2 Hyperparameter sweep

At a reduced scale S1 (d = 384, h = 6, L = 10; SSRA 24,159,744 / flat 24,021,504 parameters) both models ran a symmetric two-stage sweep: stage 1 over lr ∈ {1e-3, 6e-4, 3e-4} at dropout 0.0, stage 2 over dropout 0.1 at the within-model stage-1 winner; each cell 3,662 steps × 16 × 1,024 = 59,998,208 tokens at seed 1337, otherwise the §3.4 protocol [src: results/M2-sweep.md §B.0/§B.3]. All 8 runs completed without divergence [src: results/M2-sweep.md §B.3]. Selection was mechanical and within-model: minimum final_eval_loss on the §3.4 evaluation set.

**Table — sweep final_eval_loss (nats/token, `val-eval-2M`); winners in bold.**

| model | lr 1e-3, do 0.0 | lr 6e-4, do 0.0 | lr 3e-4, do 0.0 | lr 1e-3, do 0.1 | selected |
|---|---|---|---|---|---|
| flat | **4.28121** | 4.42130 | 4.80148 | 4.36339 | (1e-3, 0.0) |
| SSRA-P1 | **4.23127** | 4.34882 | 4.69499 | 4.35232 | (1e-3, 0.0) |

[src: results/M2-sweep.md §B.3–B.4]

Both models select (lr 1e-3, dropout 0.0); the runner-up learning rate is 6e-4 for both [src: results/M2-sweep.md §B.4]. Sweep losses are read only within a model for selection; no cross-model comparison of sweep losses is made or used as evidence anywhere in this paper.

### 4.3 Core pair at lr 1e-3

Both S2 arms trained the full 51,880 steps (850,001,920 tokens) on the identical token stream under the §3 protocol [src: results/M2-core-pair.md §iii]. The flat arm was clean end-to-end (137,252 tok/s pure-train, peak 10.85 GiB), finishing at final_eval_loss **3.21201** (ppl **24.829**) [src: results/M2-core-pair.md §iii/§iv]. The SSRA arm destabilized: a finite loss spike in the 25-step window **6,475 → 6,500** (train loss 3.97 → 7.45 nats — the largest 25-step train jump of the run, +3.4802 nats), from which the run never recovered over the remaining 45,380 steps [src: results/M2-core-pair.md §iv/§xi]. The post-spike trajectory contains a second instability episode: 155 train records above 9 nats within steps 16,675–22,100 (≈ 71 % of the possible records in that span, in ≈ 5 oscillating sub-blocks), with train maximum 10.351 at step 19,775 and validation maximum 10.088 at step 19,800; the last validation value above 8.0 occurs at step 25,800, and a tight 7.52–7.59 validation band holds only from ≈ step 35,000 [src: results/M2-core-pair.md §xi C1]. No loss was ever non-finite, so the NaN/inf abort — the only in-flight trigger active in this run (§3.4) — never fired, and the run completed by protocol [src: results/M2-core-pair.md §iv]. Final: **7.55885** (ppl **1,917.6**); the cross-model ppl ratio is **77.23×**, outside the pre-registered ±5 % band by three orders of magnitude [src: results/M2-core-pair.md §iv]. (Figure F2: `results/M2-core-curves-flat.png`, `results/M2-core-curves-ssra.png`.)

Data due diligence: because the run was a single continuous process (`resumed_from: null`), the exact token batches of steps 6,450–6,550 are reconstructible by replaying the harness's own seeded sampler, and were reconstructed. Against a 2,048-window corpus baseline they are statistically and qualitatively corpus-typical on document-boundary density, distinct-token fraction, unigram entropy, longest-run, and window-reuse statistics: of 404 batch-mean z-tests, 4 flagged at |z| ≥ 3 (≈ the count expected by chance), all mundane on decode; the only flag inside the spike window proper is a mild document-boundary elevation (z = 3.27 at step 6,487) [src: results/M2-spike-diagnostics.md §3.2–3.3]. The flat arm, which consumed the identical stream, shows only a chance-consistent local wiggle in the same token window [src: results/M2-core-pair.md §xi C2]. A data-window cause is therefore disfavoured, not excluded [src: results/M2-spike-diagnostics.md §3.3]. Mechanism-level localization was not possible for this run: gradient norms were not logged, and per-step checkpoints were not retained (the checkpoint mirror kept a single, repeatedly overwritten object), leaving log-only evidence; both instrumentation gaps were closed for §4.4 [src: results/M2-core-pair.md §xi C3; results/M2-spike-diagnostics.md §1/§5].

### 4.4 Retune at lr 6e-4 (single permitted iteration)

The pre-registered plan allowed at most one retune, changing exactly one variable (§3.4). The retune set lr 1e-3 → 6e-4 — the sweep runner-up for both models, declared before launch — holding seed 1337 and therefore initialization draws and token stream identical to §4.3; a machine-verified diff of the parsed configurations shows the learning rate and run names as the only deltas [src: results/M2-core-pair-lr6e4.md §0.1]. Instrumentation added for this pair (pre-clip gradient-norm logging, step-tagged checkpoint retention, and the sustained-instability abort of §3.4) is observability and run-control only; a frozen-reference smoke run reproduced all per-step losses bit-identically across the instrumentation change [src: results/M2-core-pair-lr6e4.md §0.3–0.4].

Both arms were stable end-to-end over all 51,880 steps: zero divergence flags, no non-finite loss at any step, the sustained-instability counter never left 0 across 261 validation evaluations per arm, and the maximum single-evaluation validation regression against the running best was 0.00279 nats (flat) / 0.00334 nats (SSRA) against the 2.0-nat trigger margin; final validation loss equals the running best in both arms [src: results/M2-core-pair-lr6e4.md §iv]. Post-warmup pre-clip gradient norms peaked at 0.887 (flat) / 1.237 (SSRA), with late-run ranges 0.60–0.76 / 0.69–0.88 [src: results/M2-core-pair-lr6e4.md §iv]. **The §4.3 spike did not recur.** (Figure F3: `results/M2-core-lr6e4-curves-flat.png`, `results/M2-core-lr6e4-curves-ssra.png`.)

**Table — final quality of the retuned pair (Gate inputs).**

| model | final_eval_loss (nats/token, `val-eval-2M`) | val ppl @ ctx 1,024 |
|---|---|---|
| flat | 3.19333 | **24.369** |
| SSRA-P1 | 3.29065 | **26.860** |

[src: results/M2-core-pair-lr6e4.md §iv]

The gap is **+10.22 %** in SSRA's disfavor (ppl ratio 1.10223; Δ = +0.09732 nats/token) [src: results/M2-core-pair-lr6e4.md §iv]. The pre-registered parity criterion (±5 %) is **not met**; the stability criterion is met. This pair is the paper's headline parity result, and the retune budget of the plan is exhausted.

### 4.5 Learning-rate stability observation

Under the pre-registered single-variable logic of the retune — identical initialization, token stream, schedule shape, and step count, with the learning rate as the only change — the disappearance of the §4.3 instability at lr 6e-4 implicates the learning rate as the cause of that instability [src: results/M2-core-pair-lr6e4.md §1/§iv]. Stated as an empirical observation: at this scale the flat baseline trains cleanly at lr 1e-3 while SSRA does not, and both train cleanly at lr 6e-4 — SSRA exhibits an empirically narrower stable learning-rate range than the flat baseline. The mechanism is undetermined, and no architectural conclusion is drawn from this observation. One scoping fact: the sweep that selected lr 1e-3 for both models ran 3,662 steps (≈ 60M tokens) per cell without incident [src: results/M2-sweep.md §B.3], while the §4.3 instability appeared at step 6,475 (≈ 106M tokens into the run) [src: results/M2-core-pair.md §iv; token count derived as 6,475 × 16,384] — the selection procedure's horizon ended before the instability's onset, so the sweep could not have detected it.

### 4.6 Throughput and memory constants

The complexity class of §2.7 says nothing about constants; the constants below are measured properties of this implementation on the stated hardware, reported under the §3.2 rule (FLOPs and wall-clock reported, not matched).

An initial batched training realization of the §2.5 read-out materialized the per-query gathered key/value cover as dense [B, h, N, k_max, d_head] tensors (k_max = 95 rows at N = 1,024, the exact Fenwick worst-case count), which autograd retains across layers; at the S2 scale this ran out of memory on an 80 GB device at every batch size tried [src: results/M2-calibration.md §3; results/M2-g2lite.md §V2]. Restructuring the same read-out as per-level block cross-attention over the contiguous consumer intervals implied by the §2.6 retention rule — logit-equivalent by construction, certified by frozen-reference A/B equivalence tests against the prior realization and by the verification suite — removed the blow-up [src: results/M2-recalibration.md §2–3; D-log 2026-07-13; results/M2-readout-optimization.md].

**Table — measured pure-train throughput and peak VRAM (bf16 autocast, ctx 1,024, seed 1337).**

| configuration (device) | SSRA tok/s | flat tok/s | flat/SSRA | SSRA peak GiB | flat peak GiB |
|---|---|---|---|---|---|
| S1 b16, initial read-out realization (A100 PCIe 80 GB) | 9,457 | 300,978 | 31.8× | 54.67 | 6.35 |
| S1 b16, restructured read-out (A100 SXM 80 GB) | 27,079 | 319,945 | **11.8×** | 18.56 | 6.35 |
| S2 b16, training runs of §4.4 (A100 SXM 80 GB) | 12,383 | 137,500 | 11.1× | 41.2 | 10.85 |

[src: row 1 results/M2-calibration.md §3; row 2 results/M2-recalibration.md §3; row 3 results/M2-core-pair-lr6e4.md §iii]

After restructuring, SSRA at S1 b16 is 2.86× faster and uses 2.95× less memory than before on the same device class, and the SSRA-vs-flat training-throughput gap at S1 b16 narrows from ≈ 32× to **11.8×** (the flat anchor's +6.3 % between rows is the SXM-vs-PCIe hardware delta, not architecture) [src: results/M2-recalibration.md §3]. In the production S2 runs of §4.4 the realized pure-train ratio is **11.1×**, with peak memory 41.2 vs 10.85 GiB (≈ 3.8×) [src: results/M2-core-pair-lr6e4.md §iii]. SSRA S2 b32 goes out of memory on 80 GB (consistent with the analytic projection of ≈ 86 GiB), while the flat model runs S2 at batch 64 in 40.01 GiB [src: results/M2-recalibration.md §3; results/M2-calibration.md §3]. Consequently, the equal-token comparisons of §4.3–§4.4 favor SSRA on the compute axis by roughly an order of magnitude; this is stated, not corrected (§3.2). No wall-clock advantage is claimed at any measured size (§2.7).

### 4.7 Length extrapolation (inference-only)

The final §4.4 checkpoints were evaluated as-is (no weight, configuration, or tokenizer modification; no positional rescaling of any kind) on region E (§3.3), disjoint from the parity evaluation set, at N ∈ {1,024, 2,048, 4,096, 8,192, 16,384, 32,768}: per cell exactly 2^21/N disjoint N-token windows, cell metric = token-weighted mean NLL over targets at window positions 2..N [src: results/M2-g2lite.md §P/§M1]. A replication anchor gates the measurement: re-running the §3.4 final-metric protocol on the evaluation set reproduced §4.4 exactly (flat Δ = 0.0 nats; SSRA Δ = −1e-5 nats; tolerance 1e-3), certifying that the evaluation harness computes the identical function the models were trained and scored with [src: results/M2-g2lite.md §M0]. Because region E consists of packed short documents (mean 1,138.7 tokens; §3.3), at N ≫ 1,024 most of each window is earlier, unrelated documents: the grid measures **degradation robustness at unseen absolute lengths, not long-range modeling benefit** (§6) [src: results/M2-g2lite.md §P.1/§V4].

Pre-registered priors: the flat baseline (RoPE at absolute positions, no extrapolation mechanism) was expected to degrade beyond its training length — the known behavior of this positional family [27] — so flat degradation is not evidence of an SSRA advantage. SSRA's positional scheme is length-invariant by construction except for two trained inputs: the level-embedding rows for levels 11–15 (untrained and exactly zero at these lengths; §5.2) and a longer Fenwick key list in the single read-out softmax (≤ 321 keys at N = 32,768 vs 225 at N = 1,024); its pre-registered prior under the §2 structural account was stable-to-mild degradation [src: results/M2-g2lite.md §P.2].

**Table T1 — perplexity vs. context length N on region E (one measurement per cell); r(N) = ppl(N)/ppl(1,024) within model.**

| N | windows | flat ppl | flat r(N) | SSRA ppl | SSRA r(N) | SSRA/flat |
|---|---|---|---|---|---|---|
| 1,024 | 2,048 | 23.684 | 1.000 | 26.306 | 1.000 | 1.111 |
| 2,048 | 1,024 | 37.134 | 1.568 | 108.217 | 4.114 | 2.914 |
| 4,096 | 512 | 96.074 | 4.057 | 1,154.91 | 43.90 | 12.02 |
| 8,192 | 256 | 224.375 | 9.474 | 4,178.99 | 158.86 | 18.63 |
| 16,384 | 128 | 443.793 | 18.74 | 9,476.24 | 360.23 | 21.35 |
| 32,768 | 64 | 775.354 | 32.74 | 14,778.0 | 561.77 | 19.06 |

[src: results/M2-g2lite.md §M1; raw `results/g2lite/m2-g2lite-flat-m1.json`, `results/g2lite/m2-g2lite-ssra-m1.json`]

Under the pre-registered degradation labels (stable r ≤ 1.10 < mild ≤ 1.50 < marked ≤ 10 < collapse), flat reads *marked* at N = 2,048–8,192 and *collapse* at 16,384/32,768; SSRA reads *marked* at 2,048 and *collapse* from 4,096 onward [src: results/M2-g2lite.md §O2/O3]. The priors therefore resolve as: **the flat prior is confirmed; the SSRA prior is violated**, from N = 2,048 onward [src: results/M2-g2lite.md §O2/O3]. There is **no crossover at any N** — SSRA perplexity exceeds flat perplexity in every cell; the cross-model ratio rises 1.111 → 21.35 (N = 16,384) and reads 19.06 at N = 32,768, and the 21.35 → 19.06 movement at the top of the grid is reported without interpretation [src: results/M2-g2lite.md §O4]. Cross-region consistency: the ratio at the shared training length on E is 1.111 (+11.07 %), matching the +10.22 % parity-set gap of §4.4 [src: results/M2-g2lite.md §O4/§M0]. All SSRA cells at N ≥ 2,048 execute with the exactly-zero level-embedding rows described in §5.2 [src: results/M2-g2lite.md §O7]. (Figure F4a: `results/M2-g2lite-ppl-vs-n.png`.)

### 4.8 Needle-lite retrieval

A committed passkey suite in the canonical format [28] — filler sentences, one embedded 5-digit key, a closing retrieval query — was generated once (360 prompts: 6 lengths × depths {0.1, 0.5, 0.9} × 20 trials; byte-deterministic; prompt budget N − 16) and evaluated identically on both models: greedy argmax via repeated full forward (a single code path; the incremental decode path deliberately unused), at most 12 new tokens, success iff the first 5-digit group in the generation equals the key [src: results/M2-g2lite.md §P/§V5].

**Table T2 — needle-lite accuracy (20 trials per cell).**

| model | depth | 1,024 | 2,048 | 4,096 | 8,192 | 16,384 | 32,768 |
|---|---|---|---|---|---|---|---|
| flat | 0.1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| flat | 0.5 | **0.95** | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| flat | 0.9 | **0.85** | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| SSRA | 0.1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| SSRA | 0.5 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| SSRA | 0.9 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

[src: results/M2-g2lite.md §M2; per-trial raw `results/g2lite/m2-g2lite-flat-m2.json`, `results/g2lite/m2-g2lite-ssra-m2.json`]

The pooled flat accuracy at N = 1,024 is 60 %, above the pre-registered 20 % floor, so the grid is informative rather than floor-limited [src: results/M2-g2lite.md §O5]. The flat model's copy behavior is depth-local even at its training length (0.00 / 0.95 / 0.85 at depths 0.1 / 0.5 / 0.9) and none of it is retained at any N ≥ 2,048 [src: results/M2-g2lite.md §M2/§O5]. **SSRA scores 0 % in every cell, including its own training length**; a retrieval-retention label is therefore not applicable to SSRA [src: results/M2-g2lite.md §O5]. Pre-registered caveat (binding): an 85M-parameter model trained on 850M tokens need not exhibit copy behavior at all — the needle suite is informative, not decisive, and H2 was not tested in its planned M3 form (§6); beyond N = 1,024 the needle result is additionally confounded by the positional collapse measured in §4.7. Behavioral characterization of the failure modes is in §5.5. (Figures F4d/F4e: `results/M2-g2lite-needle-flat.png`, `results/M2-g2lite-needle-ssra.png`.)

## 5. Diagnostics (descriptive)

The observations in this section are descriptive; none carries a mechanism claim.

### 5.1 Pooling-attention uniformity

A standing diagnostic logs the entropy of the P1 latent-query (Q_φ) attention maps over the 32 slots of a lossy node (`p1_attn_entropy`; maximum = ln 32 ≈ 3.4657) together with per-query participation. The metric is live, not dead: it responds to artificially non-uniform attention (scaling Q_φ ×100 at initialization drops it to ≈ 3.25) and decreases measurably on a toy run, but only in the 4th–6th decimal [src: results/M1-report.md §(vi-c)]. Across every SSRA run of this paper the map stays near-uniform: in [3.4655, 3.4657] through the full 60M-token sweep in all cells [src: results/M2-sweep.md §B.6], and in the stable 850M-token run start 3.4657 → minimum 3.4287 (step 50,300) → final 3.4348, never leaving a 3.43–3.47 band, with collapse-free participation ending in [0.0471, 0.1003] around the uniform value 1/16 [src: results/M2-core-pair-lr6e4.md §vi]. In the unstable §4.3 run the entropy held textbook-uniform through the spike onset itself (3.4645 at step 6,500) and de-uniformized only afterwards (3.0–3.3 band; participation spreading to [0.03, 0.23]) — a post-spike symptom, not a precursor [src: results/M2-core-pair.md §vi]. Honest reading: with Q_φ initialized ≈ N(0, 0.02²) against LayerNormed keys, pooling scores stay near zero, and at this scale P1 operates ≈ as mean pooling with a residual — no query collapse, but no query specialization was observed at any point in any run [src: results/M1-report.md §(vi-c)].

### 5.2 Level embeddings at extrapolation lengths

The e_ℓ table is sized for n_max = 32,768 ([16, 640] per layer × 15 layers), but training at sequence length 1,024 exercises only levels ≤ 10. In the trained checkpoint, rows 11–15 are exactly 0.0 in every layer (fp32 exact-zero test, not a tolerance; trained rows have max |·| = 4.638), i.e. the documented ablation-OFF state at those levels [src: results/M2-g2lite.md §V2]. Every SSRA measurement at N ≥ 2,048 in §4.7–§4.8 therefore runs with zero level embeddings at the newly reached levels.

### 5.3 bf16 position quantization

On the bf16-autocast path used in training and evaluation, integer token positions are cast to bf16 inside the rotary embedding at all three call sites. The empirical quantum in binade [2^k, 2^(k+1)) is 2^(k−7) (integers exact through 256): 4 at N = 1,024, 64 at N = 16,384, 128 in the top binade; by N ≈ 8,192 the position ULP reaches the window size w = 64, and at N = 32,768 all 65 positions of the last read-out window cast to the single value 32,768.0 — window-relative offsets are effectively constant in the top binade [src: results/M2-g2lite.md §V2b]. The effect is device-independent (a CUDA re-probe shows the angle tensor in fp32 via autocast promotion, but positions are bf16-cast before the multiply) and is shared by both models: flat has all attention positions quantized above 256, SSRA its read-out window/query positions (intra-node slot positions ≤ 32 are exact at every N) [src: results/M2-g2lite.md §V2b]. The training function at sequence length 1,024 already contained this quantization (quantum ≤ 4); it is characterized, not modified — the §4.7 anchor replication certifies function identity with training.

### 5.4 Positional locality of the degradation

Per-position bucket means show that for both models every bucket at target positions ≤ 1,024 holds baseline NLL at every N (flat 3.087–3.393 nats; SSRA 3.220–3.410 nats): the entire §4.7 degradation lives at target positions > 1,024 [src: results/M2-g2lite.md §M1; raw m1 JSONs]. At N = 32,768 the buckets beyond 1,024 rise 4.08 → 5.52 → 6.30 → 6.78 → 7.20 nats (flat) and 6.09 → 9.42 → 9.63 → 9.97 → 10.05 nats (SSRA); the penalty against each model's own 513–1,024 bucket spans +0.99 … +4.11 nats (flat) and +2.87 … +6.83 nats (SSRA) [src: results/M2-g2lite.md §M1/§O6]. Under the pre-registered +0.10-nat rule, neither model is positionally graceful [src: results/M2-g2lite.md §O6]. (Figures F4b/F4c: `results/M2-g2lite-buckets-flat.png`, `results/M2-g2lite-buckets-ssra.png`.)

### 5.5 Needle behavioral observations (qualitative)

From the per-trial generation records: beyond N = 1,024 the flat model degenerates into template-shaped loops that contain no digits; SSRA at N = 1,024 continues the filler template fluently but produces no digits at any depth; SSRA beyond 1,024 degenerates into non-template token loops [src: per-trial generations in `results/g2lite/m2-g2lite-flat-m2.json` / `m2-g2lite-ssra-m2.json`; characterization per HO-20 §4]. These are qualitative observations on committed raw records; no mechanism is inferred.

## 6. Limitations

**Scale and seeds.** Every result is at one scale — ≈ 84–85M parameters, 850M training tokens, training context 1,024 — with a single pair per configuration at one seed (1337). The single seed is by design (budget, and the single-variable isolation between §4.3 and §4.4 requires a fixed seed), but it means this is not a seed study: seed sensitivity of the parity gap, of the lr-stability boundary, and of the extrapolation behavior is unquantified, and no conclusion extends beyond this scale.

**Unexecuted pre-registered comparisons.** The plan's later stages — the Log-Linear-family baseline (GatedDeltaNet) as the empirical comparator on the retrieval axis (§7, [2]), the MEGABYTE-style two-level baseline, and the registered ablations (P2/P3 pooling operators and the P1×P3 hybrid, window w, summary schedule, separate read-out parameters, level embeddings OFF, tree arity k = 4) — did not run, because the pre-registered gate before them (§4.4) failed and the plan's stop-loss ended the branch. Their absence is a consequence of the protocol, and it leaves the mechanism-level questions they were designed to answer open.

**Corpus and evaluation regime.** Region E consists of packed short documents (mean 1,138.7 tokens), so the §4.7 grid measures degradation robustness beyond the training length, not long-range dependency benefit. Two evaluation-regime artifacts additionally shape the extrapolation measurements: the bf16 position quantization (§5.3; shared by both models) and the untrained level-embedding rows (§5.2; SSRA only). Neither was corrected, per the pre-registered identical-function rule; their contributions are not separable from architecture within this design.

**Hardware-specific constants.** The §4.1 absolute wall-clock numbers are specific to the Apple-silicon MPS backend, and the §4.6 constants (32×, 11.8×, 11.1×) to one A100-class device and this implementation. They are properties of implementation-on-hardware, not of the asymptotics in §2.7.

**Instability evidence depth.** For the §4.3 run, gradient norms were not logged and per-step checkpoints were not retained, so the instability evidence is log-only and no module- or layer-level localization was possible; the §4.4 instrumentation closes both gaps prospectively but cannot deepen §4.3 retroactively.

**Untested design branches.** H3 (the top-down pass) and axis C (content-based hierarchy) were deferred by design [26] and remain untested. H2 was not tested in its planned M3 form: needle-lite (§4.8) is the only retrieval evidence in this paper, and beyond the training length it is confounded by the positional collapse measured in §4.7.

## 7. Related work

**Premise.** The motivating observation — that language exhibits self-similar, long-range-dependent statistics (Hurst H ≈ 0.7) connected to next-token prediction — is published [1]; SSRA's contribution was never the premise but a specific architectural bet on it.

**Same complexity class.** Log-Linear Attention [2] is the closest neighbor: SSRA's read-out uses the same Fenwick prefix decomposition, taken from [2] with no novelty claimed. Within that shared skeleton the mechanism differs in all three remaining components: softmax attention vs. linear-attention kernels; learned discrete-slot pooling vs. structural matrix state with λ-decay; and cross-scale parameter sharing that includes the read-out. [2] was also the planned empirical comparator on the retrieval axis; that comparison did not run because the pre-registered gate before it failed (§4, §6). Prefix-scannable models [23] obtain softmax-like operators with O(log N) decoding state via prefix scans over chunk encodings, without cross-scale rule sharing or learned per-node slot compression.

**Hierarchical and multiresolution approximations.** H-Transformer-1D [3] reaches linear time and memory via hierarchical low-rank structure with fixed inter-level averaging — a better complexity class than SSRA's; SSRA's claim was never a better class. Fast Multipole Attention [4] learns its group-compression weights but learns them *per level* and aggregates linearly rather than with a shared attention block; MRA attention [5] uses a fixed wavelet family. The Hierarchical Kernel Transformer [24] (verified at abstract level) computes independent score matrices per resolution from trainable causal downsampling and fuses them — again per-level processing rather than one shared rule. Adjacent efficiency mechanisms: Native Sparse Attention [6] (the anchor for the registered P3 operator) and Infini-attention [7] (compressive memory in vanilla attention).

**Multi-level language models.** MEGABYTE [8] stacks two fixed levels with distinct sub-models (the planned ablation opponent for the sharing axis); Hourglass [9] shortens and re-expands activations with per-level parameters (the anchor for the registered P2 operator). Learned segment boundaries — dynamic token pooling [25], H-Net [13] — form an orthogonal axis (content-based hierarchy) deliberately excluded from this design and deferred.

**Weight sharing.** Universal Transformers [10] and Mixture-of-Recursions [11] share weights in *depth*; SSRA shares across *sequence scale*. MANO [22] is, in the surveyed set, the one found case of attention-weight sharing across scales — but for encoder-style vision/physics grids, without causality, slot summaries, or a prefix read-out. GPST [21] is the nearest neighbor on the axis "shared rule × learned tree × generative LM": it shares a learned composition function over an induced *syntactic* per-sentence tree, composes constituents to single vectors, keeps the composition model separate from the generative model, and makes no long-context or retrieval claim — versus SSRA's positional whole-context tree, m-slot summaries, and one θ for nodes and token-level read-out alike. R2D2 [12] is the encoder/parsing precursor of that line. HRM [14] shares the "small recursive model" framing but operates recurrence over two timescales in puzzle domains, without a sequence tree.

**Compression anchors.** PMA / Set Transformer [16] and Perceiver [17] anchor the P1 operator; per the note's survey [26], using PMA-style pooling as the node compressor of a causal scale hierarchy with queries shared across all nodes and scales was new. Gist tokens [18] compress prompts into a small slot set in a causal LM (one level, no tree); ToMe [19] merges tokens training-free in ViTs. StreamingLLM [20] is the precedent for heterogeneous persistent keys inside one softmax, with the nuance recorded in §2.3: its persistent keys carry a fixed cache position, so pure-NoPE summary keys go one step beyond the direct precedent.

**Disambiguation.** SSRA is unrelated, beyond the word "fractal" in informal descriptions, to Fractal Generative Models for images [15].

The novelty claim of [26] rests on the combination — cross-scale weight sharing × learned m-slot node compression × causal LM over text — which none of the surveyed works [1–25] instantiates simultaneously. This paper reports the pre-registered empirical outcome of that claim at one small scale; the outcome (§4) is negative.

## 8. Conclusion and future work

The pre-registered questions of [26] are answered, and at this scale the answers are negative. At matched parameters and matched tokens, SSRA-P1 does not reach parity with the flat baseline at the training context (+10.22 %, outside the pre-registered ±5 % band; §4.4); it exhibits an empirically narrower stable learning-rate range, mechanism undetermined (§4.5); its length-extrapolation prior is violated from N = 2,048 with no crossover at any measured length (§4.7); and it shows no needle-lite copy behavior in any cell, including its own training length (§4.8). The measured throughput constants (§4.6) mean that the equal-token protocol favored SSRA on the compute axis by roughly an order of magnitude; per §3.2 this is stated, not corrected. No architectural conclusion beyond the measured configuration and scale is drawn (§6): what is established is that the specific pre-registered success criteria were not met by this implementation at ≈ 84–85M parameters and 850M training tokens.

The project's declared success definition, fixed before any run, was a clear publishable answer to H1–H2 in either direction at matched budget; this paper is that answer for the negative direction. What survives the outcome: the mechanism's verified causal correctness (§2.4), the corrected decoding-state analysis and the in-paper erratum to [26] (§2.6, §2.8), the restructured read-out with its measured constants (§4.6), the descriptive characterization of the trained models (§5), and the audit trail that lets each result be checked or re-run (§3.6).

Future work — proposals, not commitments; none is scheduled, and each would require its own pre-registered plan:

- **Stabilization and schedule ablations** targeting the §4.5 observation (warmup shape, schedule variants, per-module learning rates), turning "mechanism undetermined" into a testable question.
- **Scale.** A single point at ≈ 84–85M parameters and 850M tokens cannot separate a mechanism-level negative from an underscaled one; a larger matched pair would sharpen either reading.
- **The registered, unexecuted comparisons and ablations** listed in §6: the Log-Linear-family and MEGABYTE-style baselines, the P2/P3 pooling operators and the P1×P3 hybrid, window w, the summary schedule, separate read-out parameters, level embeddings OFF, and tree arity k = 4.
- **Deferred design branches:** SSRA-TD (the top-down pass, H3) and axis C (content-based hierarchy), both excluded from this version by design [26].
- **Trajectory analysis** on the retained step-tagged checkpoints of the §4.4 pair — executable without new training compute — as candidate material for a v1.1 of this record.

## AI Assistance Disclosure

This work used AI assistance throughout, under the author's direction. Design triage, formalization, drafting, and independent verification passes were assisted by Anthropic's Claude; the experiments were executed by Claude Code, an AI coding agent, on infrastructure operated and paid for by the author, under the author's supervision. The AI-executed work runs against machine-checkable guards: a frozen implementation specification as the normative reference for all code, one configuration committed before each launch, and a verification suite that includes causality (shift and completion), equivalence (frozen-reference A/B), and gradient-flow checks (§2.4, §3.6). All decisions, verdicts, and reviews are the author's. Consistent with the COPE position statement on authorship and AI tools, AI tools are not authors of this work; the author takes full responsibility for all content, including AI-generated portions. This section refines the record-level disclosure added to the stage-1 note's Zenodo record [26].

## References

URLs [1]–[25] retrieved 2026-06-09 / 2026-06-11 (verification records with per-entry retrieval dates in the repository, docs/02); [27]–[28] retrieved 2026-07-17; [29] retrieved 2026-06-14 / 2026-07-13.

1. Alabdulmohsin et al. *Fractal Patterns May Illuminate the Success of Next-Token Prediction.* NeurIPS 2024. https://arxiv.org/abs/2402.01825
2. Guo et al. *Log-Linear Attention.* ICLR 2026. https://arxiv.org/abs/2506.04761
3. Zhu, Soricut. *H-Transformer-1D: Fast One-Dimensional Hierarchical Attention for Sequences.* ACL-IJCNLP 2021. https://arxiv.org/abs/2107.11906
4. Kang, Tran, De Sterck. *Fast Multipole Attention: A Divide-and-Conquer Attention Mechanism for Long Sequences.* 2023. https://arxiv.org/abs/2310.11960
5. Zeng et al. *Multi Resolution Analysis (MRA) for Approximate Self-Attention.* ICML 2022. https://arxiv.org/abs/2207.10284
6. DeepSeek. *Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention.* ACL 2025. https://arxiv.org/abs/2502.11089
7. Munkhdalai et al. *Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention.* 2024. https://arxiv.org/abs/2404.07143
8. Yu et al. *MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers.* NeurIPS 2023. https://arxiv.org/abs/2305.07185
9. Nawrot et al. *Hierarchical Transformers Are More Efficient Language Models.* Findings of NAACL 2022. https://arxiv.org/abs/2110.13711
10. Dehghani et al. *Universal Transformers.* ICLR 2019. https://arxiv.org/abs/1807.03819
11. Bae et al. *Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation.* NeurIPS 2025. https://arxiv.org/abs/2507.10524
12. Hu et al. *R2D2: Recursive Transformer based on Differentiable Tree for Interpretable Hierarchical Language Modeling.* ACL-IJCNLP 2021. https://arxiv.org/abs/2107.00967
13. Hwang, Wang, Gu. *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling.* 2025. https://arxiv.org/abs/2507.07955
14. Wang et al. *Hierarchical Reasoning Model.* 2025. https://arxiv.org/abs/2506.21734
15. Li et al. *Fractal Generative Models.* 2025. https://arxiv.org/abs/2502.17437
16. Lee et al. *Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks.* ICML 2019. https://arxiv.org/abs/1810.00825
17. Jaegle et al. *Perceiver: General Perception with Iterative Attention.* ICML 2021. https://arxiv.org/abs/2103.03206
18. Mu, Li, Goodman. *Learning to Compress Prompts with Gist Tokens.* NeurIPS 2023. https://arxiv.org/abs/2304.08467
19. Bolya et al. *Token Merging: Your ViT But Faster.* ICLR 2023. https://arxiv.org/abs/2210.09461
20. Xiao et al. *Efficient Streaming Language Models with Attention Sinks.* ICLR 2024. https://arxiv.org/abs/2309.17453
21. Hu, Ji, Zhu, Wu, Tu. *Generative Pretrained Structured Transformers: Unsupervised Syntactic Language Models at Scale.* 2024. https://arxiv.org/abs/2403.08293
22. Colagrande, Caillon, Feillet, Allauzen. *Linear Attention with Global Context: A Multipole Attention Mechanism for Vision and Physics (MANO).* 2025. https://arxiv.org/abs/2507.02748
23. Yau et al. *Sequential-Parallel Duality in Prefix Scannable Models.* 2025. https://arxiv.org/abs/2506.10918
24. Cirrincione. *Hierarchical Kernel Transformer: Multi-Scale Attention with an Information-Theoretic Approximation Analysis.* 2026. https://arxiv.org/abs/2604.08829
25. Nawrot, Chorowski, Łańcucki, Ponti. *Efficient Transformers with Dynamic Token Pooling.* ACL 2023. https://arxiv.org/abs/2211.09761
26. Sopov, D. *SSRA: Scale-Shared Recursive Attention — Design and Complexity Analysis.* Technical note v1.0, Zenodo, 2026 (prior version of this work). DOI: https://doi.org/10.5281/zenodo.20647034
27. Press, O., Smith, N. A., Lewis, M. *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation.* arXiv:2108.12409, 2021 (v2 2022). https://arxiv.org/abs/2108.12409
28. Mohtashami, A., Jaggi, M. *Landmark Attention: Random-Access Infinite Context Length for Transformers.* NeurIPS 2023. https://arxiv.org/abs/2305.16300
29. HuggingFaceFW. *fineweb-edu* (dataset), config `sample-10BT`, hub revision `87f09149…`, license ODC-By v1.0. https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu — TODO(B4): optionally add the associated academic citation after primary-source verification.

---

## INTERNAL — figures & tables inventory + captions (existing artifacts only; inventory scaffolding deleted before export, captions move to final placement at PDF export)

Verified against `results/` on 2026-07-18 (B2): all referenced artifacts exist as committed files — no gap, no author escalation needed. Captions drafted in B3; final placement (two-panel composition for F2/F3, pairing for F4b/c and F4d/e) happens at PDF export (B5). Caption numbers are recaps of values already tagged in §4–§5; B4 cross-checks them against the same source reports.

- T1 ppl(N) grid + ratios — in-text table in §4.7, built from raw `results/g2lite/m2-g2lite-{flat,ssra}-m1.json` (values cross-checked against `results/M2-g2lite.md` §M1 at B2: 12/12 cells match).
  **Caption T1:** "Table T1 — Perplexity vs. evaluation context length N on region E. One measurement per cell over disjoint N-token windows (2^21 tokens per cell); r(N) = ppl(N)/ppl(1,024) is the within-model degradation ratio, SSRA/flat the cross-model ratio at each N. Final §4.4 checkpoints evaluated as-is. The 21.35 → 19.06 movement at the top of the grid is reported without interpretation (§4.7)."
- T2 needle 18-cell grid — in-text table in §4.8, from `results/M2-g2lite.md` §M2; per-trial raw `results/g2lite/m2-g2lite-{flat,ssra}-m2.json` (oversight full recount of 720 trials, D-log 2026-07-17).
  **Caption T2:** "Table T2 — Needle-lite passkey accuracy (fraction of 20 trials per cell). One embedded 5-digit key per prompt (prompt budget N − 16); success iff the first 5-digit group of a greedy generation of at most 12 new tokens equals the key; identical prompts for both models, single code path via repeated full forward (§4.8)."
- F1 scaling shape — `results/M1-throughput.png` ✔ exists.
  **Caption F1:** "Figure F1 — Wall-clock scaling shape at the implementation-verification scale. Forward+backward time per step vs. N ∈ {1,024 … 8,192} (d = 192, h = 8, L = 2; batch 1, fp32, Apple-silicon MPS backend), log-log; fitted slopes 0.983 (SSRA) vs. 1.923 (flat). A shape measurement only: the sub-unit SSRA slope reflects fixed per-level dispatch overheads still dominant at these N, not linear scaling, and absolute wall-clock here is not a speed claim (§4.1)."
- F2 loss curves Phase 3 — `results/M2-core-curves-flat.png`, `results/M2-core-curves-ssra.png` ✔ exist.
  **Caption F2:** "Figure F2 — Training and validation loss of the core pair at lr 1e-3 (§4.3); two panels: flat, SSRA. Identical token stream in both arms. The flat arm is clean end-to-end; the SSRA arm shows the finite spike in the 25-step window 6,475 → 6,500 without recovery, and a second instability episode within steps 16,675–22,100."
- F3 loss curves Phase 3b — `results/M2-core-lr6e4-curves-flat.png`, `results/M2-core-lr6e4-curves-ssra.png` ✔ exist.
  **Caption F3:** "Figure F3 — Training and validation loss of the retuned pair at lr 6e-4 (§4.4); two panels: flat, SSRA. The learning rate is the only change against F2 (seed, initialization, and token stream identical). Both arms are stable over all 51,880 steps; final validation loss equals the running best in both; the §4.3 spike does not recur."
- F4a ppl vs N — `results/M2-g2lite-ppl-vs-n.png` ✔; F4b/F4c per-position buckets — `results/M2-g2lite-buckets-flat.png`, `results/M2-g2lite-buckets-ssra.png` ✔; F4d/F4e needle heatmaps — `results/M2-g2lite-needle-flat.png`, `results/M2-g2lite-needle-ssra.png` ✔.
  **Caption F4a:** "Figure F4a — Perplexity vs. context length N on region E (log-log), final §4.4 checkpoints evaluated as-is. The flat baseline follows its pre-registered degradation prior [27]; SSRA violates its stable-to-mild prior from N = 2,048; no crossover at any measured N (§4.7; values in Table T1)."
  **Caption F4b/F4c:** "Figures F4b/F4c — Per-position bucket mean NLL at each evaluation length N (F4b flat, F4c SSRA). For both models, buckets at target positions ≤ 1,024 hold baseline NLL at every N; the entire §4.7 degradation lives at target positions > 1,024. Under the pre-registered +0.10-nat rule, neither model is positionally graceful (§5.4)."
  **Caption F4d/F4e:** "Figures F4d/F4e — Needle-lite accuracy over the length × depth grid (F4d flat, F4e SSRA), 20 trials per cell. Flat: depth-local copy behavior at the training length only (0.00 / 0.95 / 0.85 at depths 0.1 / 0.5 / 0.9), 0 % at every N ≥ 2,048. SSRA: 0 % in every cell including its training length; the pre-registered §4.8 caveat applies (values in Table T2)."
- Optional, not referenced in text (author's call whether to include): sweep curves `results/M2-sweep-curves-{flat,ssra}.png`, G1b micro-gate curves `results/M1-g1b-curves.png`.
- New composite figures remain v1.1 scope (D-log 2026-07-17).
