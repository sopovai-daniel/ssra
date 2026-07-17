# SSRA: Scale-Shared Recursive Attention — Empirical Evaluation and Negative Results

**Full paper (stage 2 of 2).**
**Author:** Daniel Sopov (SopovAi). ORCID: [0009-0004-8584-5156](https://orcid.org/0009-0004-8584-5156).
**Status:** DRAFT v0.2, 2026-07-17 — §2/§3/§7 + References drafted (block B1); Abstract/§1/§4–§6/§8 pending; do not cite.
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

---

## Abstract

TODO(draft, block B3). Must contain: one-sentence mechanism recap + complexity class (same as Log-Linear Attention, no better-class claim); pre-registered evaluation at matched parameters + tokens; **negative results stated plainly** — parity criterion (±5 % val ppl @ ctx 1,024) not met (+10.22 %); training instability at lr 1e-3 with a stable retune at 6e-4 (narrower empirical lr range, mechanism undetermined); length extrapolation: flat prior confirmed, SSRA prior violated, no crossover; needle-lite 0 % for SSRA in all cells; throughput constant 11.8×. Framing: pre-registered falsification plan of note v1.0 executed; either outcome was declared publishable.

## 1. Introduction

TODO(draft, block B3).
- Motivation recap from note §1: self-similar statistics of language (Hurst H ≈ 0.7, [1]); flat Transformers do not encode this architecturally.
- H1 / H2 restated, wording consistent with note §1.
- Two-stage publication: note v1.0 fixed design + complexity before any run; this paper reports the pre-registered evaluation.
- Contributions: (i) executed pre-registered plan at matched parameters + tokens; (ii) negative headline results (§4); (iii) descriptive diagnostics (§5); (iv) full methodology artifacts public with the repo (decision log, pre-registered assignments, configs, raw logs).

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

TODO(draft, block B2). Every number tagged per rule 1.
- 4.1 Scaling shape: log-log wall-clock slope SSRA 0.983 vs flat 1.923 (N 1k–8k); shape measurement, not a speed claim; absolutes hardware-specific (MPS). [src: M1-report]
- 4.2 Hyperparameter sweep: symmetric two-stage lr/dropout sweep, 8/8 runs completed without divergence; within-model selections: both models lr 1e-3, dropout 0.0. No cross-model reading of sweep losses. [src: M2-sweep]
- 4.3 Core pair @ lr 1e-3: flat final_eval_loss 3.21201 (ppl 24.829), clean end-to-end; SSRA finite loss spike steps 6,475→6,500 without recovery + second instability episode in band 16,675–22,100; final 7.55885 (ppl 1,917.6). Data-window cause disfavoured-not-excluded (exact token-window reconstruction; corpus-typical statistics). [src: M2-core-pair incl. §xi; M2-spike-diagnostics T5]
- 4.4 Retune @ lr 6e-4 (single permitted iteration; lr the only changed variable, seed + token stream identical): flat 3.19333 (ppl 24.369) vs SSRA 3.29065 (ppl 26.860); **gap +10.22 %**, outside the pre-registered ±5 % band. Both arms stable end-to-end: 0 divergence flags, 261 val evals/arm, max transient val regression 0.0033 nats, final val = running best in both. [src: M2-core-pair-lr6e4]
- 4.5 lr-stability observation: disappearance of the instability under single-variable isolation implicates lr as the cause of §4.3; SSRA shows an empirically narrower stable lr range than flat at this scale; **mechanism undetermined**. [src: M2-core-pair-lr6e4 §1 pre-registered logic]
- 4.6 Throughput/memory constant: SSRA ≈ 11.8× lower training throughput than flat at S2 b16 (from 32× before read-out restructuring); constant-factor architectural cost, reported per AP-8 honesty note; peak VRAM figures. [src: M2-recalibration, M2-calibration]
- 4.7 Length extrapolation (inference-only, region E, N ∈ {1k, 2k, 4k, 8k, 16k, 32k}): table ppl(N) + within-model ratios r(N) — flat 23.68 → 775.4 (r 32.7), SSRA 26.31 → 14,778.0 (r 561.8); binding formulation rule 2; no crossover at any N; cross-region consistency: gap @ 1,024 on E = 11.07 % vs +10.22 % on the parity set. [src: M2-g2lite §M1]
- 4.8 Needle-lite (passkey, depths 0.1/0.5/0.9 × 20 trials): flat 0.00/0.95/0.85 @ 1,024 (pooled 60 %), 0 % everywhere beyond; SSRA 0 % in all 18 cells including its training length; pre-registered caveat sentence (rule 2). [src: M2-g2lite §M2]

## 5. Diagnostics (descriptive; hard cap ~1.5–2 pages)

TODO(draft, block B2). Descriptive only — no mechanism claims.
- 5.1 Pooling attention uniformity (P-C): p1_attn_entropy ≈ ln(32) throughout training in all SSRA runs (smoke → 850M tokens, both lr); participation without collapse; in §4.3 de-uniformization occurs only after the spike (symptom, not precursor). Honest reading: Q_φ attention ≈ mean pooling + residual at this scale — no specialization observed. [src: P-C rows across reports]
- 5.2 Level embeddings at extrapolation lengths: e_ℓ rows 11–15 exactly 0.0 (never trained); extrapolation runs with zero level embeddings at new levels. [src: M2-g2lite V2/V3]
- 5.3 bf16 position quantization (V2b): quantum 2^(⌊log₂ t⌋−7); from N ≈ 8k the position ULP ≥ 64 ≈ w, and the terminal window @ 32k collapses to a single position value; artifact shared by both models; characterized, not modified (anchor replication guarantees function identity with training). [src: M2-g2lite V2b]
- 5.4 Positional locality of damage: both models degrade exclusively at positions > 1,024; positions ≤ 1,024 hold baseline NLL at every N (flat 3.087–3.393; SSRA 3.220–3.410); per-position penalty @ 32k: SSRA +2.87…+6.83 nats vs flat +0.99…+4.11. [src: M2-g2lite buckets]
- 5.5 Needle behavioral observations (qualitative): flat degenerates into template-shaped loops without digits; SSRA @ 1,024 continues the template fluently without digits at every depth; SSRA > 1,024 degenerates into non-template token loops. [src: M2-g2lite §M2]

## 6. Limitations

TODO(draft, block B2).
- Single scale (≈85M params / 850M tokens); conclusions scoped to it.
- Single pair per configuration @ seed 1337 — by design (budget + single-variable isolation), not a seed study.
- Packed short-document corpus (mean doc ≈ 1.1k tokens in E): the length grid measures **degradation robustness beyond training length**, not long-range dependency benefit.
- V2b quantization + untrained e_ℓ rows shape the extrapolation regime for both models.
- G1a absolute wall-clock MPS-specific; throughput constant hardware-specific.
- H3 (top-down path) and axis C (content hierarchy) not tested — deferred branches (note §7).

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

TODO(draft, block B3).
- Pre-registered questions answered negatively at this scale; per the project's declared success definition, a negative answer at matched parameters + tokens is the publishable outcome.
- Future work (proposals, not commitments): stabilization/schedule ablations, scale, SSRA-TD (top-down branch), axis C (content hierarchy), trajectory analysis on retained step-tagged checkpoints (possible note/paper v1.1 material).

## AI Assistance Disclosure

TODO(draft, block B3). Equivalent to the v1.0 record-level disclosure, refined: experiments were executed by Claude Code on infrastructure operated and paid for by the author, under the author's supervision; the verification suite includes causality, equivalence and gradient-flow checks; all decisions, verdicts and reviews are the author's. AI tools are not authors; the author takes full responsibility for all content (COPE position statement).

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

## INTERNAL — figures & tables inventory (existing artifacts only; delete before export)

- T1 ppl(N) grid + ratios — build as a table from `results/` G2-lite CSV (no new plot).
- T2 needle 18-cell grid — table from raw JSON.
- F1 scaling shape — `results/M1-throughput.png` [verify path at B2].
- F2/F3 loss curves Phase 3 / Phase 3b — committed plots in `results/` [verify exact filenames at B2].
- F4 G2-lite plots — committed in `results/` [verify at B2].
- Any gap → author decision; new composite figures are v1.1 scope.
