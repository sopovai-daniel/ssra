# SSRA: Scale-Shared Recursive Attention — Empirical Evaluation and Negative Results

**Full paper (stage 2 of 2).**
**Author:** Daniel Sopov (SopovAi). ORCID: [0009-0004-8584-5156](https://orcid.org/0009-0004-8584-5156).
**Status:** SKELETON v0.1, 2026-07-17 — structure only, no section drafted; do not cite.
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

## 2. Mechanism (assembled from note v1.0 + spec v1.2)

TODO(draft, block B1). Subsections:
- 2.1 Up-pass: binary tree over the sequence; one shared (Attn_θ, Pool_φ) per layer at every level; token = level-0 node. [note §2.1]
- 2.2 Pool_φ: P1 latent-query pool (default; deployed config), P2/P3 named as registered alternatives. [note §2.2, spec §5]
- 2.3 Positional scheme: slot-RoPE inside nodes, level embedding e_ℓ as input coordinate; read-out keys heterogeneous (window RoPE, summaries NoPE + e_ℓ). [note §2.3, spec §6]
- 2.4 Causality without masks inside nodes: structural gating (node consumable only after its span completes); shift + completion tests. [note §2.4, spec §7]
- 2.5 Read-out: Fenwick prefix decomposition, window w = 64, read-out shares θ. [note §2.5, spec §8]
- 2.6 Autoregressive decoding + **corrected retention rule**: interval hull — node u held for t ∈ [end(u), end(u)+2^ℓ+w]. [spec §9 v1.1+]
- 2.7 Complexity: training Θ(N·(w + m·log N)·d) per layer; decode state O(m·d·log N) per layer; same class as Log-Linear Attention — explicitly no better-class claim. [note §3, spec §10]
- 2.8 **Erratum to note v1.0:** note §2.6/§3 state the point rule Frontier(t) ∪ Fenwick(t−w−1) and an explicit constant-2 bound; the point rule has a re-entry gap for nodes with 2^ℓ ≤ w and is superseded by the interval hull; the constant-2 bound held only for 2^ℓ > w. The headline decoding class O(m·d·log N) is unaffected (w-term constant in N). [D-log 2026-06-12, spec v1.1]

## 3. Experimental setup

TODO(draft, block B1).
- 3.1 Models: SSRA-P1 vs flat Transformer baseline, matched parameter count at scale S2; exact parameter counts, d/h/L, from committed configs. [src: configs + results reports]
- 3.2 Matching protocol (AP-8): matched parameters + matched tokens — identical token stream, order, step count, seed 1337; FLOPs + wall-clock reported, not matched (honesty note).
- 3.3 Data + provenance (AP-9): FineWeb-Edu `sample-10BT`, hub revision `87f09149…`, license odc-by; document-disjoint split by sha1(doc.id); byte-level BPE tokenizer, vocab 16,384, sha256 `019568a2…` (frozen); 913.6M train tokens; eval set `val-eval-2M` (2,000,000 tokens); extrapolation region E disjoint from the eval set (byte-for-byte prefix check). [src: M2-phase0-report, M2-g2lite V4]
- 3.4 Training protocol: 850,001,920 tokens per arm, bf16 autocast with fp32 accumulation, checkpoint/resume verified bit-for-bit, pre-registered stop rule (val_loss > running best + 2.0 nats on ≥ 6 consecutive evals), pre-registered parity criterion ±5 % val ppl @ ctx 1,024, single permitted retune iteration with a single changed variable.
- 3.5 Hardware + cost: rented GPUs (A100 SXM 80 GB training; RTX A6000 48 GB inference-only eval); total project compute spend from ledger (`results/runs.md`), ~72.4 EUR of a 300 EUR cap [verify exact at draft].
- 3.6 Artifacts + reproducibility: public repo (tag at publication), decision log, pre-registered assignments committed before runs, raw JSONL logs, GCS artifact store.

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

TODO(draft, block B1). From `02` #1–#25 + note §6:
- Log-Linear Attention (#2): same complexity class, structural Fenwick family — acknowledged; mechanism differs (one softmax rule + learned Pool_φ + cross-scale sharing incl. read-out).
- FMA (#4): per-level learned downsampling weights — no cross-scale sharing (verified in primary source).
- StreamingLLM (#20): precedent for heterogeneous persistent keys in one softmax.
- GPST (#21), MANO (#22), PSM (#23): explicit delimitations per spec §17.
- MEGABYTE-style two-level models; HRM (#14).
- All citations verified in primary sources with URL + retrieval date (`02`); any new citation verified at draft time (Rule W).

## 8. Conclusion and future work

TODO(draft, block B3).
- Pre-registered questions answered negatively at this scale; per the project's declared success definition, a negative answer at matched parameters + tokens is the publishable outcome.
- Future work (proposals, not commitments): stabilization/schedule ablations, scale, SSRA-TD (top-down branch), axis C (content hierarchy), trajectory analysis on retained step-tagged checkpoints (possible note/paper v1.1 material).

## AI Assistance Disclosure

TODO(draft, block B3). Equivalent to the v1.0 record-level disclosure, refined: experiments were executed by Claude Code on infrastructure operated and paid for by the author, under the author's supervision; the verification suite includes causality, equivalence and gradient-flow checks; all decisions, verdicts and reviews are the author's. AI tools are not authors; the author takes full responsibility for all content (COPE position statement).

## References

TODO(draft, blocks B1–B3). Assemble from note references + `02`; must include: [1] Alabdulmohsin et al. 2024 (fractal language); Log-Linear Attention; Press et al. (arXiv:2108.12409, extrapolation prior); Mohtashami & Jaggi (arXiv:2305.16300, passkey format); FineWeb-Edu; prior version: technical note DOI 10.5281/zenodo.20647034.

---

## INTERNAL — figures & tables inventory (existing artifacts only; delete before export)

- T1 ppl(N) grid + ratios — build as a table from `results/` G2-lite CSV (no new plot).
- T2 needle 18-cell grid — table from raw JSON.
- F1 scaling shape — `results/M1-throughput.png` [verify path at B2].
- F2/F3 loss curves Phase 3 / Phase 3b — committed plots in `results/` [verify exact filenames at B2].
- F4 G2-lite plots — committed in `results/` [verify at B2].
- Any gap → author decision; new composite figures are v1.1 scope.
