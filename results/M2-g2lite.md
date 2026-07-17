# M2 G2-lite — inference-only length-extrapolation measurement

Assignment: `docs/cc/M2-g2lite.md` v1 (2026-07-17; pre-registered, veto
passed by handover). Authorization: D-log 2026-07-17 (2nd entry, formulation
B) — G1 = FAIL recorded and closed; this is the single authorized follow-up
measurement. Runs: `m2-g2lite-flat`, `m2-g2lite-ssra` (AP-21). Scoped cap
**10 EUR**; inference only; result published regardless of direction.
**No architecture conclusions anywhere in this report (spec §16)** — every
output is an input for Daniel's decision on the paper's shape.

Objects of study: the two trained artifacts exactly as checkpointed
(final lr6e4 pair = `step-51880.pt`, assignment §1 table). Eval config:
`experiments/m2-g2lite-eval.yaml`. Harness: `scripts/g2lite_eval.py`
(modes m0/m1/m2), suite generator `scripts/needle_gen.py`, verifications
`scripts/g2lite_verify.py`, projection `scripts/g2lite_memory_projection.py`.

## §P Protocol echo (binding; pre-registered before any measurement)

**Honesty statement (assignment §1.2, verbatim):**

1. The eval corpus is packed short FineWeb-Edu documents (mean doc length
   [ODHAD] ~1.1k tokens; V4 records the exact number). At window lengths
   N ≫ 1k most of the context is earlier, unrelated documents.
   **ppl-vs-length here therefore measures degradation robustness at unseen
   absolute lengths, not long-range modeling benefit.** The expected shape
   under graceful extrapolation is a roughly flat ppl(N) curve; an exploding
   curve indicates positional failure.
2. Positional facts established from code (re-verified as V1/V2 below):
   **flat** — RoPE at absolute token positions, same rope_base, no learned
   positional parameters, no length guard — mechanically runs at any N.
   Literature prior: existing position methods including rotary do not
   extrapolate efficiently beyond training length (Press, Smith & Lewis,
   arXiv:2108.12409, retrieved 2026-07-17) — flat degradation beyond 1,024
   is the expected outcome and is **not** evidence of an SSRA advantage; it
   is the known cost of the shared positional family both models were given.
   **SSRA** — intra-node slot-RoPE uses slot indices ≤ 2m = 32 (exactly
   representable at any N); read-out window RoPE uses absolute positions
   with relative offsets ≤ w = 64 by construction (RoPE relative-offset
   property, Su et al., arXiv:2104.09864); summary keys are NoPE + e_ℓ. The
   only length-dependent trained inputs are (a) e_ℓ rows for levels 11–15
   (never exercised at seq 1,024 — expected exactly zero-init = the
   documented ablation-OFF state, spec §6; V2 verifies) and (b) a longer
   Fenwick key list in the one softmax (≤ 321 keys at 32k vs 225 at 1k).
   This asymmetry is part of the H1 design bet; it is **reported, not
   exploited** — no positional rescaling of either model.
3. Both expected-shape statements are pre-registered priors. Either
   direction of every result is publishable; there is no "good" outcome to
   steer toward.

**Grid:** N ∈ {1,024, 2,048, 4,096, 8,192, 16,384, 32,768}, both models,
identical inputs per cell; 16k/32k subject to the §5 VRAM gate and the
de-scope ladder D1–D4, never to mid-run improvisation.

**M0 anchor (gates everything):** exact replication of the G1 final-eval
protocol at N=1,024 on `val-eval-2M` via `final_eval` **imported from
`scripts/train.py`** (identical code path by construction): 1,953 windows /
1,999,872 tokens / 127 dropped, deterministic full pass, bf16 autocast +
fp32 loss accumulation (AP-16), batch 16. Must reproduce final_eval_loss
**flat 3.19333 / SSRA 3.29065** within ≤ 1e-3 nats. Failure = STOP (harness
exit 5), report, no further measurement.

**M1 ppl vs length:** E = `val.bin` tokens [2,000,000, 2,000,000 + 2^21);
per cell exactly 2^21/N disjoint windows of N tokens (2,048 @ 1k … 64 @
32k). Operationalization (recorded in the YAML `m1.scoring` before launch):
targets exist at 1-indexed window positions 2..N; position 1 of each window
has no prediction; cell metric = token-weighted mean NLL over all targets
(fp32 CE, fp64 accumulation) → ppl; per-position bucket means by target
position (buckets 1–256, 257–512, 513–1,024, then doubling). One
measurement per cell; no re-runs.

**M2 needle-lite:** committed suite (seed 20260717, V5 below), cells
N × depth ∈ {0.1, 0.5, 0.9} × 20 trials, prompt budget N − 16, identical
prompts for both models; greedy argmax via repeated full forward (single
code path; decode path deliberately unused), ≤ 12 tokens, stop at eot;
metric = first `\d{5}` regex match == key.

**Interpretation criteria O1–O7 (assignment §4)** apply mechanically:
r_M(N) = ppl_M(N)/ppl_M(1,024); labels stable ≤ 1.10 < mild ≤ 1.50 <
marked ≤ 10 < collapse; O4 crossover; O5 needle floor rule (pooled 1k
accuracy < 20 % both models → `UNINFORMATIVE-FLOOR` for longer N); O6
positionally-graceful bucket rule (+0.10 nats vs own 513–1,024 bucket);
O7 e_ℓ zero-rows note wherever SSRA numbers at N ≥ 2,048 appear.

**Prohibitions (assignment §3):** no RoPE rescaling (PI/NTK/YaRN/
temperature); no sliding-window/chunked eval for flat; no weight, config or
tokenizer modification; no sampling; no per-cell retries; no decode-path
usage; no baselines (b)/(c); no multi-needle; `summary_pos_override` off;
model code untouched.

## §V Pre-launch verifications (assignment §2; local, 0 EUR)

Machine-readable records: `results/g2lite/verify-*.json`.

### V1 — flat positional encoding [VERIFIED from code]

`baselines/flat.py:35-37`: `pos = arange(1, n+1)`; `rope_rotate` applied to
q and k at absolute token positions with `cfg.rope_base` (= 10000, same
`ModelConfig` as SSRA). `FlatLM` (lines 60–81): token embedding only, no
learned positional parameters, no `n_max` check in `forward` — mechanically
runs at any N. Training/eval call path (`scripts/train.py` `BUILDERS` +
`final_eval`): same config object, bf16 autocast + fp32 accumulation.
The §1.2 description is exact.

### V2 — SSRA length mechanics [PARTIAL — checkpoint items pending GCS reauth]

- (a) `n_max` guard: `SSRALM.forward` raises `ValueError` for N > n_max
  (exercised on a tiny model: N = n_max admitted, N = n_max+1 raises);
  trained `n_max: 32768` admits the whole grid. [VERIFIED]
- (b) `level_emb` table shape [16, 640] in the SSRA checkpoint — **PENDING**
  (blob download blocked; see §Status).
- (c) rows 11–15 exactly 0.0 — **PENDING** (same; any nonzero value =
  STOP-and-report, not silently accepted).
- (d) `readout_cache` builds cleanly for every grid N on CPU [VERIFIED]:

| N | k_max (gather rows) | cache bytes | build s |
|---|---|---|---|
| 1,024 | 95 | 883,712 | 0.01 |
| 2,048 | 111 | 2,062,336 | 0.02 |
| 4,096 | 127 | 4,714,496 | 0.05 |
| 8,192 | 143 | 10,608,640 | 0.13 |
| 16,384 | 159 | 23,576,576 | 0.22 |
| 32,768 | 175 | 51,871,744 | 0.46 |

- Blob sha256 of both checkpoints (first-ever read) — **PENDING**.

### V2b — bf16 position quantization [CHARACTERIZED; no code change]

Probe (`verify-v2b.json`): all three RoPE call sites instrumented in-process
(wrapper functions in `scripts/g2lite_verify.py` only; model code untouched)
on the eval path (`model.eval()` + `no_grad` + bf16 autocast, tiny models,
CPU; pod pre-flight re-runs the probe on CUDA):

| site | x dtype | pos dtype | **angle dtype** |
|---|---|---|---|
| flat attention (absolute RoPE) | bf16 | int64 | **bf16** |
| SSRA read-out window/query RoPE | bf16 | int64 | **bf16** |
| SSRA node slot-RoPE (slots ≤ 32) | bf16 | int64 | **bf16** |

Empirical integer-position quantum per binade (positions cast via
`pos.to(bf16)` inside `rope_rotate`):

| binade [2^k, 2^(k+1)) | quantum | max rounding err | distinct bf16 positions in the last 65-token window at N = 2^(k+1) |
|---|---|---|---|
| [128, 256) | 1 (exact) | 0 | 65 (all) |
| [256, 512) | 2 | 1 | — |
| [512, 1024) | 4 | 2 | 17 @ N=1,024 |
| [1024, 2048) | 8 | 4 | 9 @ N=2,048 |
| [2048, 4096) | 16 | 8 | 5 @ N=4,096 |
| [4096, 8192) | 32 | 16 | 3 @ N=8,192 |
| [8192, 16384) | 64 | 32 | 2 @ N=16,384 |
| [16384, 32768) | 128 | 64 | **1 @ N=32,768** |

- Formula note: the empirical quantum in binade [2^k, 2^(k+1)) is
  **2^(k−7)** (bf16 stores 7 mantissa bits; integers exact up to 256, first
  rounding at 257). The assignment's parenthetical "quantum
  2^(⌊log2 t⌋−8)" is off by one against this table; the empirical table
  governs (Pravidlo W). The qualitative statement (rounding above 256,
  doubling per binade) is unchanged.
- Shared effect, characterized and reported, not suppressed or exploited:
  flat — ALL attention positions quantized above 256; SSRA — read-out
  window/query RoPE only (at N = 32,768 all 65 window positions collapse to
  one bf16 value ⇒ window relative offsets are effectively constant in the
  top binade); SSRA intra-node slot positions ≤ 32 are exact at every N.
  The training function at seq 1,024 already contained this quantization
  (quantum ≤ 4); the eval uses the identical code path — the M0 anchor
  certifies function identity. No "fix" is applied (binding handling rule,
  assignment §2 V2b).

### V3 — checkpoint integrity [PARTIAL]

- Config sha256/16 of the committed training YAMLs: at `3db45ef` AND at
  HEAD both equal the assignment §1 values `ed606161f99e713a` (flat) /
  `d35a628774f87d65` (SSRA). [VERIFIED, `verify-local.json`]
- Frozen tokenizer sha256 `019568a2…d669a0` matches. [VERIFIED]
- GCS object sizes vs §1 table; blob sha256; strict `load_state_dict` into
  the matching model class — **PENDING** (GCS reauth; `g2lite_verify.py
  ckpt` runs these the moment the blobs are local).

### V4 — eval-region statistics [PENDING]

Requires `data/m2/val.bin` (present only in GCS; local copy blocked on the
same reauth). `g2lite_verify.py eregion` is committed and will record:
token count, eot document count, mean/median/p90 doc length, and the
prefix-identity check (val.bin[:2,000,000] == val-eval-2M byte-for-byte ⇒
E at offset 2,000,000 has zero overlap).

### V5 — needle generator + suite [DONE]

- Confirmed absent from the repo before this assignment (no `*needle*`
  match); implemented as `scripts/needle_gen.py`, canonical passkey format
  per Mohtashami & Jaggi (arXiv:2305.16300, retrieved 2026-07-17): repeated
  filler sentences, one needle "The pass key is <5-digit>. Remember it.
  <5-digit> is the pass key.", prompt ends "… What is the pass key? The
  pass key is". Single needle only (multi-needle = M3, excluded).
- Unit tests `tests/test_needle_gen.py` (7 tests, green): exact
  target-length control (= N − 16 for every cell shape); depth placement
  (needle start = round(depth × budget), unclamped on the real grid);
  tokenizer round-trip of the key digits under the frozen tokenizer
  (sha 019568a2…) using the same first-`\d{5}` rule as the metric;
  determinism under the suite seed; 5-digit key range; too-small-budget
  rejection.
- Suite generated at seed **20260717**: 6 N × 3 depths × 20 trials = 360
  prompts; byte-deterministic archive (regeneration reproduces the hash):
  - `artifacts/needles/m2-g2lite/suite.jsonl.gz`
    sha256 `5876f3a46421a7252223cfaccf9d85e70190dde50f2b59bf91298f1dd0b9da89`
  - uncompressed JSONL sha256
    `fdb2ec5542ff6d5bc578340b5cb4f32fa917498cf81aa32fedec90bc33d5f440`
  - `artifacts/needles/m2-g2lite/manifest.json`: per-trial sha256 of the
    uint16 token bytes + keys + placement indices + segment texts.

### Harness smoke (assignment §6.1) [DONE]

`tests/test_g2lite_eval.py` (11 tests, green; CPU, bf16 autocast + fp32
accumulation, tiny d=64/L=2 models, both archs, N ∈ {64, 128}): M0 code
path via the imported `final_eval` + anchor pass/fail logic (within-vs-out
of 1e-3 tolerance); M1 window counts, scored-token counts, buckets, r(N)
field, and **batch-size invariance of the fp64-accumulated cell value**;
M2 greedy full-forward generation (deterministic across repeat runs, stops
at eot, ≤ max_new) and metric fields. Full repo test suite green
(includes causality tests `test_shift.py` / `test_completion.py`).

## §5 VRAM projection and GPU recommendation

`scripts/g2lite_memory_projection.py` — D5 analytic model extended to the
no-grad inference path (no autograd residency; peak = fp32 params + the
largest single-phase concurrent live set; assumptions in the script
docstring), **×1.20 (AP-22)**, gate ≤ 95 % of card VRAM (48 GB → 45.6 GiB;
80 GB → 76.0 GiB). Logits materialization [B, N, 16384] bf16 + chunked
fp32 CE is inside the projection (chunked loss = admissible plumbing).
Params (fp32): flat 84,301,440 (0.314 GiB), SSRA 84,647,040 (0.315 GiB).

| mode | model | N | B | projected GiB | ×1.20 GiB | 48 GB gate | 80 GB gate | dominant phase |
|---|---|---|---|---|---|---|---|---|
| m1 | flat | 1,024 | 64 | 10.62 | 12.74 | PASS | PASS | head |
| m1 | flat | 2,048 | 64 | 20.93 | 25.12 | PASS | PASS | head |
| m1 | flat | 4,096 | 64 | 25.57 | 30.68 | PASS | PASS | head |
| m1 | flat | 8,192 | 64 | 34.82 | 41.78 | PASS | PASS | head |
| m1 | flat | 16,384 | 32 | 26.82 | 32.18 | PASS | PASS | head |
| m1 | flat | 32,768 | 16 | 22.82 | 27.38 | PASS | PASS | head |
| m1 | ssra | 1,024 | 64 | 10.62 | 12.74 | PASS | PASS | head |
| m1 | ssra | 2,048 | 64 | 20.93 | 25.12 | PASS | PASS | head |
| m1 | ssra | 4,096 | 64 | 25.57 | 30.68 | PASS | PASS | head |
| m1 | ssra | 8,192 | 64 | 34.82 | 41.78 | PASS | PASS | head |
| m1 | ssra | 16,384 | 32 | 32.93 | 39.51 | PASS | PASS | attn/readout |
| m1 | ssra | 32,768 | 16 | 34.05 | 40.86 | PASS | PASS | attn/readout |
| m2 | flat | all ≤ 32,768 | 20 | ≤ 23.44 | ≤ 28.13 | PASS | PASS | head |
| m2 | ssra | ≤ 16,384 | 20 | ≤ 20.71 | ≤ 24.85 | PASS | PASS | attn/readout |
| m2 | ssra | 32,768 | 10 | 21.42 | 25.70 | PASS | PASS | attn/readout |

Flat SDPA math-fallback risk (extra GiB if neither flash nor mem-efficient
dispatches; the pod-start micro-benchmark checks measured VRAM against this
table at N=8,192 BEFORE any 16k/32k cell): N=4,096 +40 GiB; N ≥ 8,192
+160…+640 GiB (infeasible → the §5 gate/de-scope ladder applies, never
mid-run improvisation). Training dispatch on A100 (peak 10.846 GiB @
b16/N=1,024 incl. backward) is consistent with no N² materialization.

**GPU recommendation (§5 selection rule):** every cell passes the 48 GB
gate at usable batch sizes ⇒ book the **cheapest available** of
{RTX A6000 48 GB, L40S 48 GB, A100 80 GB} on deploy day (console prices,
Pravidlo W; assignment [ODHAD] ranks A6000 ≈ $0.49/hr cheapest). On-demand
Secure; AP-19 step 0 Community capture attempt first, verbatim record.
Batch tables as committed in `experiments/m2-g2lite-eval.yaml`.

## §D Deviations

None so far. Two protocol notes recorded before launch (not deviations):

1. M1 window operationalization (targets at positions 2..N of each
   N-token window) — the only reading consistent with the §3 window count
   2^21/N; fixed in the YAML `m1.scoring` before any measurement.
2. V2b quantum formula: empirical quantum is 2^(⌊log2 t⌋−7), not the
   assignment's parenthetical 2^(⌊log2 t⌋−8); characterization only, no
   protocol impact (the handling rule — identical code path, no fix — is
   unchanged).

## §Status (pre-launch)

Committed and green: needle suite + manifest (V5), harness + smoke tests,
verify records `verify-local.json` / `verify-v2b.json`, projection table,
eval YAML. **Blocked on GCS reauthentication (interactive `gcloud auth
login`; no service key on this machine):** checkpoint downloads → V2(b,c) +
blob sha256 + V3 strict-load; `val.bin` download → V4; then the two
`sha256: PENDING-GCS-REAUTH` fields in the eval YAML get filled and this
section is replaced by the "ready for pod" record. `runs.md` rows are
appended at launch (assignment §6.1). No pod, no spend, no model-code
changes in this phase.

## §M0 / §M1 / §M2 Results

*(filled during the pod session; one measurement per cell)*

## §O Interpretation criteria applied (O1–O7)

*(mechanical application post-run; no architecture conclusions, spec §16)*

## §L Ledger

*(console-final rows per the ≥ 2 h rule; ECB 1.1430 unless a top-up occurs)*
