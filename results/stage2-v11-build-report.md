# Stage-2 v1.1 build report (V11 material integration into the results paper)

Assignment: `docs/cc/stage2-v11-build.md` v1 (2026-07-21; pre-registered, committed
at `6bf7be9` before this build). Authority chain: spec v1.2 >
`docs/cc/V11-data-exploitation.md` §3/§5 (pre-registered, unchanged) > the assignment.
Executed 2026-07-21, local CPU only, 0 EUR. Files touched: exactly two —
`paper/results-paper.md` (edits #1–#8) and this report (new). Zero diffs under
`src/ssra/` and `baselines/` (edit #8 was a read-only code read; verify via the
commit file list, G-V11-4 pattern). No PDF export, no Zenodo/DOI action, no push;
`TODO(v1.1-DOI)` / `TODO(v1.1-date)` placeholders left for the publish session.

## 1. Gate B0 — input asserts (all PASS, no edit before completion)

| # | Artifact | Assert | Result |
|---|---|---|---|
| 1 | `results/v11/v11-k1-extract-ssra.npz` | exists, 52,855,593 B | **PASS** (52,855,593 B) |
| 2 | `results/v11/v11-k1-extract-flat.npz` | exists, 293,600 B | **PASS** (293,600 B) |
| 3 | `results/v11/v11-k1-rho.csv` | parses; recomputed medians over analysis populations | **PASS** — ssra median 2.763482 (273 tensors), flat 2.864481 (183), both recomputed with Python `statistics.median` over the `rho` column per arm, matched at 6 dp |
| 4 | `results/v11/v11-k1-analysis-summary.json` | verdict; median; latent_q min/max + layers; thresholds | **PASS** — `ct1.verdict = "inconclusive"`; `median_rho` = 2.7634817547553556 → 2.763482 (6 dp); min 1.5331051032226932 at `layers.13.pool.latent_q`, max 7.836802079668234 at `layers.9.pool.latent_q`; thresholds 0.2763481754755356 / 2.7634817547553556 → 0.276348 / 2.763482 (6 dp) |
| 5 | `results/v11/v11-needle-category-counts.csv` | totals per model (of 360) | **PASS** — flat [36, 4, 320], ssra [0, 0, 360] (cat1/cat2/cat3, summed over all cells) |
| 6 | `results/v11/v11-pc-entropy-summary.json` | 17 series; 15/17 within 2.4e-4 of ln 32; lr6e4 core min/final | **PASS** — 17 series; 15 with `abs_dev_from_ln32_max` ≤ 2.4e-4; `m2-core-ssra-s2-850m-lr6e4.log`: `entropy_min` 3.4287, `entropy_last` 3.4348 |
| 7 | `results/v11/v11-v2b-quantum.json` | `all_match: true`; [256, 512) cell = 33 | **PASS** — `all_match: true`, 8 binades all `match_committed: true`, [256, 512) `distinct_positions_last_window` = 33 |
| 8 | Figures (11 files) under `results/figures/v11/` | all exist | **PASS** — all 11 present (`v11-k1-{rho-ssra,ta-flat,ta-ssra,tb-phi-ssra,tc-levelemb-ssra,td-flat,td-ssra}.png`, `v11-{needle-categories,pc-entropy,pc-participation,v2b-quantum}.png`) |

Checker note (disclosed for transparency, not a deviation): the first run of the
B0 checker flagged assert 4 because it string-matched the 6-dp literal
"2.763482" against the JSON, which stores full-precision floats
(2.7634817547553556). The artifact was correct; the checker was corrected to
compare numerically at 6 dp (the assignment's stated precision) and all asserts
passed. No edit was made before the full PASS.

### v1.0 artifact integrity (rule 1.3)

| Artifact | md5 pre-edit | md5 post-edit |
|---|---|---|
| `paper/ssra-results-paper-v1.0.pdf` | `a0177d2334b30adc552ed8d80f4a9509` | `a0177d2334b30adc552ed8d80f4a9509` (byte-identical) |
| `paper/ssra-results-paper-v1.0.md` | `db038ffb4bf1a9df0f238bed8b51becf` | `db038ffb4bf1a9df0f238bed8b51becf` (byte-identical) |

## 2. Edit #8 — pre-registered 0-EUR code read (answer of record)

**Question:** does the read-out planner consume a level-10 (root) summary at any
query position when seq_len N = 1,024?

**Answer: NO.**

Citations (answer of record = the code):

- `src/ssra/fenwick.py:51` — the read-out key set for 1-indexed position t is
  `fenwick_blocks(t - w - 1)`: the Fenwick decomposition of the prefix
  [1, t − w − 1].
- `src/ssra/fenwick.py:16-19` — each emitted block's level is the bit-length of
  the lowest set bit of the remaining prefix p (`b = p & (-p)`), so a level-10
  block can be emitted only when p ≥ 2^10 = 1,024.
- `src/ssra/config.py:39` — `w: int = 64`.
- `src/ssra/model.py:262-267` — the training/eval read-out consumes exactly this
  planner via `build_readout_index(n, cfg.w, cfg.s_l)`.

One-sentence mechanical reason (as placed in §5.2): at sequence length 1,024 the
read-out prefix budget t − w − 1 never exceeds 1,024 − 64 − 1 = 959 < 2^10, so
`fenwick_blocks` never emits a level-10 block and the level-10 (root) summary is
never consumed by the read-out.

Cross-check (standalone throwaway enumeration, pure-Python re-implementation of
the published Fenwick rule, no imports from `src/ssra/`): over all t ∈ [1, 1,024]
at w = 64, the maximum block level ever emitted is 9 (first at t = 577); a
level-10 block is never emitted. The code citation, not the script, is the
answer of record. Consequence applied: §5.2 states the mechanical reason as
verified from code (edit #2).

## 3. Gate B3 — number → artifact map (every new number in the paper)

Every number below was read programmatically from the named artifact during this
build; inline `[src:]` tags were used during drafting and stripped before commit.

| Paper location | Number(s) | Artifact + programmatic derivation |
|---|---|---|
| §5.1 F5 sentence; caption F5 | 17 series; 15 of 17; 2.4e-4; ln 32 | `results/v11/v11-pc-entropy-summary.json`: `len(series)` = 17; count of series with `abs_dev_from_ln32_max` ≤ 2.4e-4 = 15; `ln32` field |
| §5.2 | trained rows 0–9; row 10 exactly 0.0 at every checkpoint and at init | `v11-k1-extract-ssra.npz`: max abs of `full/layers.{i}.level_emb[:, 10, :]` over all 52 steps × 15 layers = 0.0; `full_init/layers.{i}.level_emb[10, :]` = 0.0 all layers (also `v11-k1-analysis-summary.json` `tc.rows_exact_zero_all_steps_and_init` = [10, …, 15]) |
| §5.2 | 959; 2^10 = 1,024 | arithmetic from code constants: 1,024 − w − 1 with w = 64 (`src/ssra/config.py:39`); Fenwick lowest-set-bit rule (`src/ssra/fenwick.py:16-19,51`) |
| §5.2 (kept v1.0, intact) | rows 11–15 exactly 0.0; max \|·\| = 4.638 | v1.0 formulations kept verbatim; consistency re-verified from NPZ: rows 11–15 max abs = 0.0; rows 0–9 final-step max abs = 4.6381 |
| §5.3 F6 sentence; caption F6 | 8 binades all match; 33 at [256, 512), N = 512; positions 1 … 32,768; t = 8,192, ULP = w = 64 (caption) | `results/v11/v11-v2b-quantum.json`: 8 `binades` entries, all `match_committed: true`, `all_match: true`; [256, 512) cell `distinct_positions_last_window` = 33; `method` field; t = 8,192 annotation = committed [8192, 16384) binade quantum 64 = w (v1.0 §5.3 statement, unchanged) |
| §5.5 defs | passkey length 5; `KEY_RE = \d{5}`; 720 trials | `scripts/needle_gen.py` (KEY_RE, as pre-registered); 720 = 2 × 360 = sum of `trials` column, `v11-needle-category-counts.csv` |
| §5.5 Table T3 | flat 36 / 4 / 320; SSRA 0 / 0 / 360 | `v11-needle-category-counts.csv`: per-model sums of cat1/cat2/cat3 |
| §5.5 distribution sentence | depth 0.5: 19 + 1; depth 0.9: 17 + 3; depth 0.1: 20× cat3; every flat cell N ≥ 2,048 = 20× cat3; SSRA 360× cat3 | `v11-needle-category-counts.csv`: per-cell rows (flat/1024/0.5 = 19,1,0; flat/1024/0.9 = 17,3,0; flat/1024/0.1 = 0,0,20; all 15 flat cells N ≥ 2,048 = 0,0,20; all 18 ssra cells = 0,0,20) |
| §5.6 provenance | 52 checkpoints; steps 1,000 … 51,000 stride 1,000 + 51,880 | `v11-k1-extract-{ssra,flat}.npz` `steps` array: 52 entries, verified `list(range(1000, 52000, 1000)) + [51880]`, both arms |
| §5.6 provenance | drift ≈ 1.136 at the embedding; 0.5 threshold | NPZ `meta_json.init_rel_drift_at_smin`: max = 1.1359429 (ssra) / 1.1353811 (flat), argmax `emb.weight` both arms; `init_validated: false` both; threshold 0.5 in the committed `results/v11/v11-k1-extract.log` (lines 3, 58) per the pre-registered rule |
| §5.6 criterion | committed 2026-07-19, before any checkpoint was read | `git log -- docs/cc/V11-data-exploitation.md`: first commit `ee173e7` dated 2026-07-19; checkpoints read 2026-07-20 (`results/V11-exploitation.md` §K1) |
| §5.6 criterion | [OP] aggregation quote | quoted verbatim from `results/V11-exploitation.md` §K1-analysis [OP] 2 (sanctioned by the assignment as the quotation source) |
| §5.6 verdict | 273 tensors; median 2.763482; thresholds 0.276348 / 2.763482; max 7.836802 (layer 9); min 1.533105 (layer 13); inconclusive | `v11-k1-analysis-summary.json` `ct1.*` (6-dp match) + independent recompute of the median from `v11-k1-rho.csv` (273 ssra rows) |
| §5.6 obs (a) | 0.55× … 2.84× | computed from `v11-k1-analysis-summary.json`: `latent_q_min/median_rho` = 0.5548, `latent_q_max/median_rho` = 2.8358 |
| §5.6 obs (b); caption F7b | row 10 == 0.0, all 52 checkpoints + init, all 15 layers; rows 11–15 unchanged PASS | NPZ full tensors as for §5.2; `tc.rows_11_15_exact_zero: true`, `tc.flags: []` |
| §5.6 obs (c); caption F7c | 0.0538 (interval ending 28,000) → 0.0667 (33,000) → resumed decay | recomputed from `v11-k1-extract-ssra.npz`: class ‖Δθ‖₂/‖θ‖₂ for the 15 `pool.latent_q` tensors per [OP] 1 (norm of concatenation) and [OP] 4 (start-of-interval denominator, end-step x-axis): rate(28,000) = 0.0538, rate(33,000) = 0.0667, local peak (0.0667 > rate(32,000) = 0.0627 and > rate(34,000) = 0.0597) |
| §5.6 flat control | flat median ρ = 2.864481 | `v11-k1-rho.csv` recompute over the 183 flat rows (B0 assert 3) |
| §2.8 sentence; Version history | note v1.1 DOI 10.5281/zenodo.21462145 | assignment §4 #7.1 / D-log 2026-07-21 (commit `b4ded72`); bibliographic identifier, not a measurement |
| Header; Version history | v1.0 date 2026-07-19; DOI 10.5281/zenodo.21439493 | carried over from the v1.0 header lines (unchanged values) |

## 4. Per-edit anchor summary (#1–#8)

1. **#1 New §5.6 "Weight-trajectory analysis (v1.1)"** — inserted between the
   §5.5 closing sentence and `## 6. Limitations`. Content in the assignment's
   order: provenance (52 checkpoints/arm, S_min fallback), pre-registered
   criterion with dates and the verbatim [OP] quote, mechanical C-T1 verdict
   (inconclusive), observations (a)–(c) mapped verbatim from §K1-analysis, flat
   control, F7a–c in body + repo-referenced remainder + one-command
   reproduction, descriptive closing sentence (spec §16). ≈ 0.9 page.
2. **#2 §5.2 precision refinement** — sentence 1's imprecise "exercises only
   levels ≤ 10" implication refined to trained rows 0–9 (row 10 exactly 0.0 at
   every checkpoint and at init, per §5.6), plus the edit-#8 mechanical reason
   verified from code (`src/ssra/fenwick.py`). Kept intact and byte-identical:
   "rows 11–15 are exactly 0.0 in every layer (fp32 exact-zero test, not a
   tolerance; trained rows have max |·| = 4.638)" and the closing N ≥ 2,048
   sentence. §4.7 rows-11–15 sentence and §3.1 header note untouched
   (verified: §3 and §4 byte-identical, §5 below). Classification: precision
   refinement — no published number was false.
3. **#3 §5.1** — existing prose unchanged (byte-identical prefix); one new
   sentence + figure pointer F5 appended (17 series, 15/17 within 2.4e-4, the
   two 51,880-step core departures). Caption F5 with the required axis note
   (entropy over 32 pooled keys, max ln 32; participation over 16 latent
   queries, uniform 1/16; participation figure repo-referenced) added to the
   INTERNAL inventory.
4. **#4 §5.3** — existing text unchanged; one new sentence + figure pointer F6
   (standalone CPU bf16 round-trip 1 … 32,768; 8/8 binades match; [256, 512) →
   33 at N = 512). Caption F6 annotates t = 8,192 where ULP = w = 64.
5. **#5 §5.5** — Table T3 inserted with the category definitions quoted
   verbatim from `docs/cc/V11-data-exploitation.md` §3 K3-c (cited as §3 K3-c
   per the assignment's HO-32 correction), the operational reading, the
   distribution sentence, and the four repo references; the existing
   qualitative sentences kept byte-identical.
6. **#6 Figure curation** — followed exactly as fixed: body §5.6 = F7a
   (`v11-k1-rho-ssra.png`), F7b (`v11-k1-tc-levelemb-ssra.png`), F7c
   (`v11-k1-td-ssra.png`); repo-referenced only: `v11-k1-ta-ssra.png`,
   `v11-k1-ta-flat.png`, `v11-k1-td-flat.png`, `v11-k1-tb-phi-ssra.png`,
   `v11-pc-participation.png`, `v11-needle-categories.png`. No conflicts found;
   nothing swapped.
7. **#7 Cross-refs, header, Version history, INTERNAL bookkeeping** —
   §2.8: one sentence appended (note v1.1 in-file erratum mirror, inline DOI;
   [26] remains the v1.0 citation; no new bibliography entry — References
   byte-identical). §8: "(executed in v1.1; §5.6)" appended to the
   trajectory-analysis bullet, no other wording change. Header: Status/DOI
   lines set to the assignment's v1.1 pattern with TODO(v1.1-date)/
   TODO(v1.1-DOI); "Prior version" line unchanged. New `## Version history`
   between AI Assistance Disclosure and References, v1.0 + v1.1 bullets, with
   the rule-1.6 changelog sentence verbatim (grep count 2: header + version
   history). §0 INTERNAL: source map gained the §5.6 row and the §5.1 and
   §4.7–4.8/§5.2–5.5 rows were extended with the `results/v11/*` artifacts;
   inventory gained T3, F5, F6, F7a–c entries with drafted captions and
   existence checkmarks (all 11 v11 figures verified present in B0 assert 8).
   §0 stays in source (stripped only in the export copy at publish time).
8. **#8 Code read** — §2 of this report; answer NO, consumed by edit #2.

## 5. Rule 1.5 grep evidence (headline formulations + untouchable text)

Headline formulations — present, line contexts byte-identical vs HEAD
(`diff <(git show HEAD:paper/results-paper.md | grep -F "<pat>") <(grep -F "<pat>" paper/results-paper.md)`
empty for each):

| Formulation | Occurrences (pre = post) | Contexts |
|---|---|---|
| "flat prior confirmed; SSRA prior violated" | 1 | byte-identical |
| "+10.22 %" | 5 | byte-identical |
| "11.8×" | 4 | byte-identical |
| "11.1×" | 4 | byte-identical |

Untouchable sections — md5 of the extracted section text, HEAD vs working tree,
all IDENTICAL: Abstract; §1; §2 up to §2.8 (§2.8 itself verified as pure
one-sentence append, v1.0 text preserved as prefix); §3; §4 (all — includes
every §4.x conclusion sentence); §6; §7; References. Diff hunk audit: 8 hunks,
each in a sanctioned location (header; §0 source map; §2.8; §5.1–§5.2;
§5.3/§5.5/§5.6 insertion; §8 bullet; Version history; INTERNAL inventory).
Forbidden-vocabulary grep over added lines: clean. No new "matched" claim
introduced (AP-8 note not required in any new sentence).

## 6. Deviations

**None.** All Gate B0 asserts passed before any edit; edit #8 preceded edits
#1–#7 as ordered; no file outside the two deliverables was modified; no V11
script was re-run to produce new outputs (all verification reads were
read-only recomputations from committed artifacts); K2 remains descoped. The
B0 checker-precision note in §1 is disclosed above; it changed no artifact and
no number.

## 7. Handoff

Ready for oversight review (Claude): recompute every §3 map row independently
from artifacts 3–7 + NPZ; re-verify the rule 1.5 greps; G-V11-4-style file-list
check post-push. Reviewer-pass gate (Daniel) follows oversight PASS and is the
publish precondition (Zenodo "New version" on record 21439493, full AP-25
block; hard deadline 2026-08-31).
