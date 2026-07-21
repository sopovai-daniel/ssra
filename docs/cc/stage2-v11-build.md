# Assignment for CC: stage-2 v1.1 build (V11 material integration into the results paper)

**Version:** v1 (2026-07-21).
**Authority chain:** spec v1.2 > `docs/cc/V11-data-exploitation.md` §3/§5 (pre-registered metrics, categories, and criteria — UNCHANGED) > this assignment.
**Approval:** veto regime — handover to CC = approved (D-log 2026-07-21: scope B (#1–#8) accepted by the author).
**Scope of this build:** source edits to `paper/results-paper.md` plus one build report. **No PDF export, no Zenodo action, no DOI work** — publish happens in a separate later session (Zenodo "New version" on record 21439493, full AP-25 block; hard deadline 2026-08-31). The build leaves `TODO(v1.1-DOI)` / `TODO(v1.1-date)` placeholders resolved only at publish time.

## 0. Context

Stage-2 paper v1.0 is live (DOI 10.5281/zenodo.21439493). V11 post-publication data exploitation is complete and oversight-PASSed (report `results/V11-exploitation.md`; D-log 2026-07-20). The author decided (D-log 2026-07-21) that v1.1 = scope B: integrate the entire routed V11 material in one version. Governing inputs for every number and formulation in this build:

- `results/V11-exploitation.md` — §K1-analysis (C-T1, [OP] list, T-A…T-E observations), §K3 (entropy overlay, V2b, needle categories), §K1 (extraction provenance, init-validation drop).
- `docs/cc/V11-data-exploitation.md` — §5 (T-A…T-E metric definitions, H-T1, C-T1 thresholds), §3 K3-c (needle category definitions). Note: HO-32 located the category definitions in "§5"; the governing location is **§3 K3-c** (verified 2026-07-21) — cite that.
- Machine-readable artifacts listed in §2 below (the only permissible numeric sources).

## 1. Hard rules

1. **0 EUR.** Local CPU only. No GPU, no VM, no GCS access, no model forwards, no training, no tuning. K2 stays descoped; no substitute analysis.
2. **Zero diffs under `src/ssra/` and `baselines/`** (G-V11-4 pattern; verified from the commit file list). Edit #8 is a read-only code read.
3. **Files touched: exactly two** — `paper/results-paper.md` and `results/stage2-v11-build-report.md` (new). Nothing else. In particular, `paper/ssra-results-paper-v1.0.pdf` (md5 `a0177d2334b30adc552ed8d80f4a9509`) and `paper/ssra-results-paper-v1.0.md` (md5 `db038ffb4bf1a9df0f238bed8b51becf`) must be byte-identical before and after (assert md5 pre/post).
4. **§0 INTERNAL rules of the paper apply to every new sentence:** number-provenance tags during drafting (rule 1), binding formulations (rule 2), forbidden vocabulary (rule 3), no architectural conclusions (rule 4, spec §16), no new measurements (rule 5), AP-8 honesty note where "matched" appears (rule 6).
5. **Untouchable text (grep-verified pre/post, byte-identical contexts):** the Abstract; §1–§4 (all of them; exceptions: none — §2.8 and §8 edits below are outside §4.x and are the only sanctioned touches near them); §6; §7; the headline formulations "flat prior confirmed; SSRA prior violated", "+10.22 %", "11.8×", "11.1×"; every §4.x conclusion sentence.
6. **Mandatory changelog sentence, verbatim, in the v1.1 Version history entry:** "no v1.0 conclusion modified; one precision refinement (§5.2)".
7. AP-21: no overwrites of any committed artifact; the two files in rule 3 are the only writes.

## 2. Gate B0 — input asserts (hard; abort and report on any failure, no edits)

Assert before any edit; record all results in the build report:

| # | Artifact | Assert |
|---|---|---|
| 1 | `results/v11/v11-k1-extract-ssra.npz` | exists, **52,855,593 B** |
| 2 | `results/v11/v11-k1-extract-flat.npz` | exists, **293,600 B** |
| 3 | `results/v11/v11-k1-rho.csv` | parses; recomputed medians over the analysis populations = **2.763482** (ssra, 273 tensors) and **2.864481** (flat, 183) at 6 dp |
| 4 | `results/v11/v11-k1-analysis-summary.json` | parses; C-T1 verdict field = inconclusive; median 2.763482; per-layer ρ(latent_q) min **1.533105** (layer 13), max **7.836802** (layer 9); thresholds 0.276348 / 2.763482 |
| 5 | `results/v11/v11-needle-category-counts.csv` | totals per model (of 360): flat **36 / 4 / 320**, ssra **0 / 0 / 360** (cat1/cat2/cat3) |
| 6 | `results/v11/v11-pc-entropy-summary.json` | 17 series; 15 of 17 with max abs deviation from ln 32 ≤ 2.4e-4; lr6e4 core min 3.4287, final 3.4348 |
| 7 | `results/v11/v11-v2b-quantum.json` | `all_match: true`; [256, 512) cell = 33 |
| 8 | Figures (11 files) | all exist under `results/figures/v11/`: `v11-k1-{rho-ssra,ta-flat,ta-ssra,tb-phi-ssra,tc-levelemb-ssra,td-flat,td-ssra}.png`, `v11-{needle-categories,pc-entropy,pc-participation,v2b-quantum}.png` |

**Numeric sourcing rule (Gate B3):** every number written into the paper is read programmatically from artifacts 3–7 (or, for §K1 provenance scalars — drift 1.1359/1.1354, byte counts, step list — from the NPZ `meta_json` / committed listings), never typed from the prose report. Draft each new quantitative sentence with an inline `[src: <artifact>]` tag; resolve every tag against its artifact; strip tags before commit; record the full number → artifact map in the build report (§7).

## 3. Edit #8 — pre-registered 0-EUR code read (read-only; feeds §5.2)

**Binary question:** does the read-out planner consume a level-10 (root) summary at any query position when seq_len N = 1,024?

- Method: reading code under `src/ssra/` only (planner / Fenwick index construction / retention logic). No forwards, no execution of model code, no diffs anywhere. A tiny standalone throwaway enumeration script (pure Python re-implementation of the published Fenwick rule, no imports from `src/ssra/`) MAY be used as a cross-check but the answer of record is the code citation.
- Deliverable: YES/NO + file:line citation(s) + a one-sentence mechanical reason derived from the cited code.
- Context, not the answer ([ODHAD], HO-30 §2 — do NOT paste as fact): the expectation is NO, because a level-10 node ends at position 1,024 and the read-out prefix budget t − w − 1 never reaches 1,024 for t ≤ 1,024; verify what the code actually does.
- Consequence: if answered, §5.2 states the mechanical reason as verified from code with the citation; if the code read is inconclusive, §5.2 carries the bare measured fact only (no speculation). Either way the answer goes in the build report.

## 4. Edits — exact scope B list

Anchors below quote the current v1.0 text of `paper/results-paper.md` (verified 2026-07-21). Prose is CC's to write within the stated contracts; facts, numbers, thresholds and formulations are fixed.

### #1 New §5.6 "Weight-trajectory analysis (v1.1)" — inserted after §5.5, before §6

Compact (anti-dilution of a negative-results paper; target ≤ ~1 page incl. figure pointers). Required content, in order:

1. **Provenance:** 52 step-tagged checkpoints per arm of the §4.4 pair (steps 1,000 … 51,000 stride 1,000, plus 51,880), extracted to compressed per-arm archives on a transient same-region CPU VM and deleted after verification; the committed extracts (`results/v11/v11-k1-extract-{ssra,flat}.npz`, `v11-k1-rho.csv`, `v11-k1-analysis-summary.json`) are the permanent record. Init reconstruction failed its pre-registered validation in both arms (max per-tensor relative drift ≈ 1.136 at the embedding vs the 0.5 threshold), so per the pre-registered fallback **all deltas are referenced to the earliest checkpoint S_min = step 1,000**; no init-relative statistic is reported.
2. **Pre-registered criterion (the post-hoc defense — state the dates):** H-T1 and the C-T1 thresholds were committed 2026-07-19, before any checkpoint was read (`docs/cc/V11-data-exploitation.md` §5): ρ(m) = ‖W_m(final) − W_m(ref)‖_F / ‖W_m(ref)‖_F; *supported* iff ρ(latent queries) < 0.1 × median_m ρ(m); *refuted* iff ≥ median. The per-layer aggregation operationalization, fixed before computation and disclosed in the committed analysis report, quoted verbatim: "[OP] Aggregation over the 15 per-layer latent_q tensors: *supported* iff max_i rho(latent_q_i) < 0.1 × median; *refuted* iff min_i rho(latent_q_i) ≥ median; otherwise *inconclusive*."
3. **Mechanical verdict:** analysis population = 273 unique trainable tensors (alias-deduplicated); median ρ = **2.763482**; *supported* required max_i ρ(latent_q_i) < **0.276348** — not met (max **7.836802**, layer 9); *refuted* required min_i ≥ **2.763482** — not met (min **1.533105**, layer 13). **C-T1: inconclusive.** Exclusively mechanical; no softening, no strengthening.
4. **Observations (a)–(c) — labeled as observations, letters and content mapped verbatim from the §K1-analysis report (mapping recorded in D-log 2026-07-20 / HO-30):**
   - (a) from the C-T1 paragraph: the latent queries do NOT remain near the S_min reference — all 15 ρ values lie between 0.55× and 2.84× the population median — so the persistent ≈ ln 32 pooling entropy of §5.1 is not accompanied by immobile latent queries; H-T1's candidate "frozen queries" mechanism is unsupported by these data.
   - (b) from T-C: the trained e_ℓ rows are **0–9, not 0–10** — row 10 is exactly 0.0 (`== 0.0`, no tolerance) at all 52 checkpoints AND at init, in all 15 layers; rows 11–15 pass the pre-registered exact-zero check unchanged (→ §5.2 refinement, edit #2).
   - (c) from T-D: pool.latent_q is the **only** module class with a non-monotone relative update rate under the cosine decay — 0.0538 (interval ending 28,000) → local peak 0.0667 (33,000) → resumed decay; no flat-arm counterpart.
5. **Flat control:** identical pipeline on the flat arm; flat median ρ = **2.864481**, the same order as SSRA's — recorded as control, no criterion attached.
6. **Figures/artifacts:** F7a/F7b/F7c in body (edit #6); repo-referenced pointers to the remaining figures and to `v11-k1-rho.csv` (full per-tensor inputs, both arms) for the reproduction path (`scripts/v11_k1_analysis.py`, one command).
7. Closing sentence: descriptive only; no architectural conclusion (spec §16) — consistent with the §5 preamble.

### #2 §5.2 precision refinement (NOT an erratum; §4.7 and §3.1 are NOT edited)

Current §5.2 sentence 1 contains the imprecise implication "training at sequence length 1,024 exercises only levels ≤ 10". Refine: the v1.1 trajectory extraction (§5.6) shows row 10 is ALSO exactly 0.0 at every checkpoint and at init, so the trained rows are **0–9**. Keep intact and true: "rows 11–15 are exactly 0.0 in every layer (fp32 exact-zero test, not a tolerance…)" and "trained rows have max |·| = 4.638". If edit #8 answered YES/NO with a citation, add the one-sentence mechanical reason verified from code; else the bare fact only. Classification (for the report and Version history): **precision refinement** — no published number was false; only the levels-≤ 10 implication was imprecise. The §4.7 sentence about rows 11–15 and the §3.1 header note remain true and untouched (diff purity).

### #3 §5.1 — P-C entropy overlay figure (text unchanged)

Existing §5.1 prose stays as-is (consistent with the K3-a semantic finding). Add figure **F5** = `results/figures/v11/v11-pc-entropy.png` plus ONE new sentence referencing it: the overlay covers all 17 SSRA runs with committed JSONL logs; 15 of 17 series never depart from ln 32 by more than 2.4e-4 nats at any logged step; the only sustained departures are the two 51,880-step core runs already described. **Caption F5 must carry the axis note:** entropy is computed over the 32 pooled keys of a lossy node (maximum ln 32); the companion participation statistic is computed over the 16 latent queries (uniform value 1/16); participation figure repo-referenced (`v11-pc-participation.png`), not in body ([OP], veto-able).

### #4 §5.3 — V2b quantum figure

Add figure **F6** = `results/figures/v11/v11-v2b-quantum.png` plus ONE new sentence: the quantum table was independently re-derived by a standalone CPU bf16 round-trip of positions 1 … 32,768 — all 8 binades match, and the previously uncharacterized [256, 512) binade evaluates to 33 distinct last-window positions at N = 512. Caption annotates t = 8,192 where ULP = w = 64. Existing §5.3 text unchanged.

### #5 §5.5 — needle categorization table T3

Insert Table **T3** (category counts) into §5.5; keep the existing qualitative sentences (they stay, now anchored by the table). Category definitions **verbatim from `docs/cc/V11-data-exploitation.md` §3 K3-c** (pre-registered): "(cat1) exact gold passkey match; (cat2) contains a digit string of the generator's passkey length … that is ≠ gold; (cat3) contains no such digit string" — with the operational reading from the analysis: passkey length 5 (`KEY_RE = \d{5}` in `scripts/needle_gen.py`), non-overlapping `re.findall`, cat1 coincides with `correct` in both directions (first match = gold). Counts (of 360 per model): flat **36 / 4 / 320**, SSRA **0 / 0 / 360**. Distribution sentence: all flat cat1/cat2 sit at N = 1,024 (depth 0.5: 19 + 1; depth 0.9: 17 + 3; depth 0.1: 20× cat3); every flat cell at N ≥ 2,048 is 20× cat3; SSRA is 360× cat3 at every N including its training length. Repo-reference the categorized CSV, the counts CSV, the top-10 JSON, and `v11-needle-categories.png`.

### #6 Figure curation ([OP], fixed 2026-07-21 after visual inventory; flag conflicts, do not silently swap)

- **In body §5.6:** F7a `results/figures/v11/v11-k1-rho-ssra.png` (ρ strip, both thresholds — the C-T1 figure); F7b `results/figures/v11/v11-k1-tc-levelemb-ssra.png` (e_ℓ rows 0–10 — observation (b)); F7c `results/figures/v11/v11-k1-td-ssra.png` (relative update rates + lr overlay — observation (c)).
- **Repo-referenced only:** `v11-k1-ta-ssra.png`, `v11-k1-ta-flat.png`, `v11-k1-td-flat.png`, `v11-k1-tb-phi-ssra.png`, `v11-pc-participation.png`, `v11-needle-categories.png`.

### #7 Cross-ref hygiene, header, Version history, INTERNAL bookkeeping

1. **§2.8 append one sentence:** the erratum is additionally mirrored in-file in technical note v1.1 (inline DOI 10.5281/zenodo.21462145). **[26] remains the v1.0 citation** — the pre-registration record; do not add a new bibliography entry ([OP], veto-able).
2. **§8 last bullet:** append "(executed in v1.1; §5.6)" to the trajectory-analysis bullet; no other wording change.
3. **Header:** Status line → v1.1 pattern of note v1.1 — `**Status:** v1.1, TODO(v1.1-date) — Changes from v1.0 (2026-07-19): a trajectory-analysis section (§5.6) with the pre-registered C-T1 verdict, figure/table upgrades (§5.1, §5.3, §5.5), and cross-references to note v1.1; no v1.0 conclusion modified; one precision refinement (§5.2).` DOI line → `**DOI:** TODO(v1.1-DOI) (this version); v1.0: [10.5281/zenodo.21439493](…)`. "Prior version" line (technical note v1.0) unchanged.
4. **New `## Version history` section** between "AI Assistance Disclosure" and "References" (mirror of note v1.1): v1.0 bullet (2026-07-19, DOI 21439493, publication release) + v1.1 bullet (TODO(v1.1-date), TODO(v1.1-DOI)) itemizing the delta and containing the rule-1.6 changelog sentence verbatim.
5. **§0 INTERNAL:** source map += `§5.6 | results/V11-exploitation.md §K1-analysis + results/v11/{v11-k1-rho.csv, v11-k1-analysis-summary.json, v11-k1-extract-*.npz}`; extend the §5.1/§5.5 row and the §4.7–4.8/§5.2–5.5 row with the corresponding `results/v11/*` artifacts. Figures & tables inventory += F5, F6, F7a–c, T3 entries with drafted captions and existence checkmarks. §0 stays in the source (stripped only in the export copy at publish time, as in v1.0).
6. Zenodo related-works reciprocity check = publish session (AP-25 block), NOT this build.

## 5. Anti-goals (restated; violations void the build)

- No edits to Abstract, §1, §2 (except the single #7.1 sentence in §2.8), §3, §4, §6, §7; no changes to any §4.x conclusion or headline formulation; no reinterpretation of C-T1 beyond the mechanical verdict; observations stay observations.
- No new metrics, thresholds, categories, or figure types beyond the enumerated set; no post-hoc analysis of any artifact; no re-running of any V11 script to produce NEW outputs (re-running for verification of existing numbers is fine and read-only).
- No PDF export; no export-copy creation; no touching `paper/export/`, `paper/technical-note.md`, or any v1.0 artifact.
- No architectural conclusions, no causal claims about the G1 gap (spec §16); forbidden vocabulary per docs/00.
- No push (human checkpoint); no Zenodo/DOI actions.

## 6. Deliverables, commit, handoff

1. Edited `paper/results-paper.md` (all edits #1–#8).
2. **Build report `results/stage2-v11-build-report.md`:** (i) Gate B0 assert results incl. md5 pre/post of both v1.0 artifacts; (ii) edit #8 answer + file:line citation; (iii) the complete number → artifact map (Gate B3); (iv) per-edit anchor summary (#1–#8: what changed, where); (v) grep evidence for rule 1.5 (headline formulations present, contexts byte-identical); (vi) §Deviations (any deviation disclosed before results are used; silent deviations void the affected edit).
3. **One signed commit** containing exactly the two files; message: `Stage-2 v1.1 build: integrate V11 material into results paper (#1-#8)`. Clean `git status` before commit; substantive only, no housekeeping mixed in. **No push.**
4. Handoff to oversight review (Claude): every new number recomputed independently from artifacts 3–7 + NPZ; rule 1.5 grep re-verified; G-V11-4-style file-list check post-push. Reviewer-pass gate (Daniel) follows oversight PASS and is the publish precondition.
