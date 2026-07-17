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

### V2 — SSRA length mechanics [VERIFIED]

- (a) `n_max` guard: `SSRALM.forward` raises `ValueError` for N > n_max
  (exercised on a tiny model: N = n_max admitted, N = n_max+1 raises);
  trained `n_max: 32768` (confirmed from the checkpoint's stored
  `config_raw`) admits the whole grid. [VERIFIED]
- (b) `level_emb` table shape **[16, 640]** in the SSRA checkpoint, one
  table per layer × 15 layers. [VERIFIED, `verify-ckpt.json`]
- (c) rows 11–15 **exactly 0.0**: max |·| over all 15 layers = 0.0
  (fp32 exact-zero test, not a tolerance); trained rows 0–10 have
  max |·| = 4.638 — the zero rows are the documented ablation-OFF state,
  not an artifact of a dead table. [VERIFIED]
- (d) `readout_cache` builds cleanly for every grid N on CPU [VERIFIED]:

| N | k_max (gather rows) | cache bytes | build s |
|---|---|---|---|
| 1,024 | 95 | 883,712 | 0.01 |
| 2,048 | 111 | 2,062,336 | 0.02 |
| 4,096 | 127 | 4,714,496 | 0.05 |
| 8,192 | 143 | 10,608,640 | 0.13 |
| 16,384 | 159 | 23,576,576 | 0.22 |
| 32,768 | 175 | 51,871,744 | 0.46 |

- Blob sha256 (first-ever read, 2026-07-17, `verify-ckpt.json`; recorded
  in the eval YAML):
  - flat `559940e74039b5523a5fad3c7984ac5fccb18669f2041183cd9bb00ea6cfba7b`
  - SSRA `b0ba9b552fdb62e19b7c8bd0817d696107720b35a650de403b0da4cc11651e01`

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
- Terminal-window addendum (verbatim probe, `verify-v2b-window.json`):
  at N = 32,768 the eval-path positions are 1-indexed (`arange(1, n+1)`),
  so the last read-out window of query t = 32,768 is [32704, 32768]. Manual
  RNE arithmetic predicts the two representable neighbors {32640, 32768};
  the probe (governing) records: 32703 → 32640.0 (the 0-indexed reading's
  extra position; distance 63 < 65); 32704 → 32768.0 (exact tie 32640+64,
  RNE picks the even mantissa 32768); 32705…32767 → 32768.0; 32768 exact.
  ⇒ every position of the 1-indexed window casts to the single value
  **32768.0** (the "distinct = 1" row above); the {32640, 32768} pair
  appears only when 32703 is included.
- **CUDA re-probe (pod pre-flight, RTX A6000, torch 2.12.0+cu126;
  `results/g2lite/pod/verify-v2b.json`):** x dtype bf16, pos dtype int64 at
  all three sites — but the **angle tensor is fp32 on CUDA** (vs bf16 in
  the CPU probe): CUDA autocast's op policy promotes the `pow` in
  `inv_freq` to fp32, and the bf16-cast positions then multiply into an
  fp32 result (CPU autocast leaves `pow` in bf16). The explicit
  `pos.to(x.dtype)` cast happens BEFORE that multiply, so integer
  positions are still bf16-quantized on CUDA and the quantum table above
  is device-independent. This fp32-angle path is byte-identical to what
  training ran on the A100 (same code, same autocast policy) — the M0
  anchor (§M0: deltas 0.0 / −1e-5) certifies function identity. No code
  change (binding handling rule).
- Shared effect, characterized and reported, not suppressed or exploited:
  flat — ALL attention positions quantized above 256; SSRA — read-out
  window/query RoPE only (at N = 32,768 all 65 window positions collapse to
  one bf16 value ⇒ window relative offsets are effectively constant in the
  top binade); SSRA intra-node slot positions ≤ 32 are exact at every N.
  The training function at seq 1,024 already contained this quantization
  (quantum ≤ 4); the eval uses the identical code path — the M0 anchor
  certifies function identity. No "fix" is applied (binding handling rule,
  assignment §2 V2b).

### V3 — checkpoint integrity [VERIFIED]

- Config sha256/16 of the committed training YAMLs: at `3db45ef` AND at
  HEAD both equal the assignment §1 values `ed606161f99e713a` (flat) /
  `d35a628774f87d65` (SSRA). [VERIFIED, `verify-local.json`]
- Frozen tokenizer sha256 `019568a2…d669a0` matches. [VERIFIED]
- GCS object sizes == §1 table exactly (1,011,848,651 / 1,016,124,393
  bytes; console listing and downloaded files agree). [VERIFIED]
- Strict `load_state_dict` into the matching model class: **OK for both**
  (no missing/unexpected keys); `step` 51,880 and stored `run_name` match
  the §1 identity (= `step-51880.pt`); parameter counts 84,301,440 (flat) /
  84,647,040 (SSRA) equal the projection-script counts. [VERIFIED,
  `verify-ckpt.json`]

### V4 — eval-region statistics [VERIFIED]

`verify-eregion.json` (sha256 gates on both shards passed first):

| quantity | value |
|---|---|
| val.bin tokens | 48,050,671 |
| prefix identity | val.bin[:2,000,000] == val-eval-2M **byte-for-byte** |
| E region | val.bin[2,000,000, 4,097,152) = 2,097,152 tokens (2^21) |
| overlap with val-eval-2M | **0 tokens** (E starts exactly at the G1 set's end) |
| eot markers in E (id 0) | 1,841 |
| documents fully inside E | 1,840 |
| doc length mean / median / p90 / max | **1,138.7** / 666 / 2,152 / 33,513 tokens |

The §1.1 [ODHAD] "mean doc length ~1.1k tokens" is confirmed at 1,138.7;
the honesty statement's framing (at N ≫ 1k most context is earlier,
unrelated documents) stands.

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

Declared BEFORE launch (nothing silent):

- **D1 — M1 accumulation precision:** protocol §3 pre-registers "fp32
  accumulation"; the harness computes per-position cross-entropy in fp32
  and **accumulates the per-position sums in fp64**
  (`g2lite_eval.eval_cell`). Strictly more precise than the pre-registered
  wording, entirely outside the model forward (bf16-autocast forward is
  untouched), and certifiedly batch-invariant
  (`test_m1_batching_invariance`). The M0 anchor is unaffected: it runs
  the imported `final_eval` from `scripts/train.py`, which accumulates in
  fp32 exactly as trained.
- **D2 — SSRA batch tables revised at pod pre-flight (commit `e305ad0`,
  BEFORE any measurement):** the pre-registered YAML batch tables were
  modified after the pre-flight found that level-1 node SDPA flattens to
  B·N/2 batch items and CUDA kernel launches fail above ~65,535
  ("invalid configuration argument", RTX A6000 / torch 2.12.0+cu126);
  SSRA B capped so B·N/2 ≤ 32,768. Disclosed here (never silent);
  wall-clock-only — batch-invariance of cell values is test-certified
  (`test_m1_batching_invariance`); the M0 anchor is unaffected (its batch
  size 16 was never changed).

Protocol notes recorded before launch (not deviations):

1. M1 window operationalization (targets at positions 2..N of each
   N-token window) — the only reading consistent with the §3 window count
   2^21/N; fixed in the YAML `m1.scoring` before any measurement.
2. V2b quantum formula: empirical quantum is 2^(⌊log2 t⌋−7), not the
   assignment's parenthetical 2^(⌊log2 t⌋−8); characterization only, no
   protocol impact (the handling rule — identical code path, no fix — is
   unchanged).
3. SDPA backend gate (admissible plumbing, `g2lite_eval.sdpa_backend_gate`):
   the harness logs flash/mem-efficient/math availability at start and
   REFUSES to start a flat run with grid ≥ 8,192 if both flash and
   mem-efficient are disabled (math fallback = +160…+640 GiB, §5 risk
   table) — routing that state to the §5 de-scope ladder instead of a
   silent fallback. Dispatch itself is never altered (training default
   kept).

## §Status: EXECUTED (pre-launch record below; session record in §S)

All §6.1 pre-launch deliverables committed: V1–V5 verified (records
`results/g2lite/verify-{local,v2b,v2b-window,ckpt,eregion}.json`), needle
suite + manifest, harness + 18 harness/suite tests green (full repo suite
green), eval YAML complete including both blob sha256 values, projection
table + GPU rule. Zero pod time, zero spend, zero model-code diffs.
`runs.md` rows are appended at launch (assignment §6.1).

**Deploy parameters (AP-18 checklist inputs; console values authoritative
on deploy day, Pravidlo W):**

| item | value |
|---|---|
| GPU | cheapest available of {RTX A6000 48 GB, L40S 48 GB, A100 80 GB}, on-demand **Secure** (AP-19 step 0: Community capture attempt first, verbatim record) |
| image | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (= template "Runpod Pytorch 2.4.0 - SSRA"; bootstrap pins torch 2.12.0+cu126) |
| container disk | **40 GB** (Phase 3b measured ~9.6 GB used on the same env incl. m2 shards; + 2×0.95 GiB checkpoints ⇒ ~12 GB, >3× margin), no network volume |
| env | `GCP_SA_KEY_B64={{ RUNPOD_SECRET_gcp_ssra_runpod_sa }}` |
| start command | default (bootstrap decodes the key from PID-1 env; AP-17) |
| session | supervised end-to-end; wall cap 4.0 h → de-scope ladder D1–D4; AP-23 strict self-terminate; terminate-not-stop (AP-18); ECB 1.1430 |

**On-pod sequence:** clone/bundle repo @ the pre-launch commit → `bash
scripts/pod_bootstrap.sh` (AP-17 gate blocks before any billable work) →
pytest → `g2lite_verify.py v2b` on cuda (angle-dtype re-probe; new record
name) → pull both checkpoints from GCS (sha256 gates in the harness) →
append runs.md rows → micro-benchmark (tok/s + measured VRAM vs projection
@ N=1,024 and 8,192, both models; flash-dispatch check for flat) → per
model: m0 (anchor gate) → m1 → m2 → mirror raw JSON to
`gs://ssra-poc-ew3/m2/g2lite/` → AP-23 terminate.

## §S Session record (2026-07-17, supervised end-to-end)

- Pod `pktqlt4jys3uiz`, 1× RTX A6000 48 GB (49,140 MiB, driver 570.195.03),
  Secure on-demand, region EU-SE-1, **$0.49/hr GPU + $0.006/hr disk =
  $0.50/hr (console verbatim)**; AP-19 step 0: Community not shown in the
  deploy flow (7th occurrence). Image
  `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, container
  disk 40 GB; torch pinned 2.12.0+cu126 by bootstrap; commit `5bcbe39`
  (+ `fef2831` probe --device flag, `e305ad0` batch tables, shipped by scp
  and committed to main).
- Gates, in order, all PASSED: **AP-17** sanity gate before any workload
  (SA `ssra-runpod@…`, `gsutil ls` OK); env snapshot
  (`logs/g2lite/g2lite-env-snapshot.txt`); SDPA flags flash/mem-efficient/
  math all enabled; both checkpoint blobs re-hashed ON THE POD ==
  YAML/§1 sha256; **pytest 83 passed + the single standing known failure**
  (`test_loglinear_integration`, torchvision::nms pin conflict — identical
  to Phase 3b, `logs/g2lite/g2lite-pytest.log`); CUDA V2b probe (§V2b
  addendum above).
- **Micro-benchmark** (`results/g2lite/pod/microbench.json`; forward-only,
  measured): flat 228,472 tok/s @ N=1,024/B=64, 180,207 @ 8,192/B=64,
  98,023 @ 32,768/B=16 (peak 18.36 GiB — **no N² blowup ⇒ flash/
  mem-efficient dispatch confirmed empirically**; math fallback would need
  +640 GiB); SSRA 25,238 @ 1,024/B=64, 23,635 @ 8,192/B=8, 22,778 @
  32,768/B=2 (peak ≤ 5.38 GiB). Recomputed session wall ≈ 1.2 h ≪ 4.0 h
  cap ⇒ **full grid, no de-scope ladder rung applied**.
- Timeline (UTC): first SSH 15:32:42 → bootstrap+gates → M0 → M1 → M2 →
  GCS mirror verified by listing 16:44:41–46 (12 objects,
  `gs://ssra-poc-ew3/m2/g2lite/`) → **AP-23 self-terminate invoked
  16:44:59 (pod id + API key sourced from /proc/1/environ), `pod
  "pktqlt4jys3uiz" removed`, connection refused confirmed 16:45:10** —
  idle tail ≈ 11 s.

## §M0 Anchor (gate of everything; O1)

| model | measured | expected (G1) | Δ nats | tol | verdict |
|---|---|---|---|---|---|
| flat | 3.19333 | 3.19333 | **0.0** | 1e-3 | **PASS** |
| SSRA | 3.29064 | 3.29065 | **−1e-5** | 1e-3 | **PASS** |

Window identity exact in both: 1,953 windows / 1,999,872 tokens / 127
dropped (`final_eval` imported from `scripts/train.py`, batch 16, AP-16).
The −1e-5 SSRA delta is A6000-vs-A100 float reassociation, 100× inside
tolerance.

## §M1 ppl vs length (region E; one measurement per cell)

| N | windows | flat ppl | flat r(N) | SSRA ppl | SSRA r(N) | SSRA/flat |
|---|---|---|---|---|---|---|
| 1,024 | 2,048 | 23.684 | 1.000 | 26.3061 | 1.000 | 1.111 |
| 2,048 | 1,024 | 37.1343 | 1.568 | 108.2169 | 4.114 | 2.914 |
| 4,096 | 512 | 96.0737 | 4.057 | 1,154.91 | 43.90 | 12.02 |
| 8,192 | 256 | 224.3745 | 9.474 | 4,178.99 | 158.86 | 18.63 |
| 16,384 | 128 | 443.7932 | 18.74 | 9,476.24 | 360.23 | 21.35 |
| 32,768 | 64 | 775.3537 | 32.74 | 14,777.99 | 561.77 | 19.06 |

Per-position bucket means (nats; full values in the committed JSONs;
curves in `results/M2-g2lite-buckets-{flat,ssra}.png`): for BOTH models
every bucket ≤ 1,024 stays flat across all N (flat ≈ 3.09–3.39; SSRA ≈
3.22–3.41) — **the entire degradation lives at target positions >
1,024**. At N=32,768: flat buckets beyond 1,024 rise 4.08 → 5.52 → 6.30 →
6.78 → 7.20; SSRA 6.09 → 9.42 → 9.63 → 9.97 → 10.05. (O7: all SSRA cells
at N ≥ 2,048 run with exactly-zero e_ℓ rows 11–15 = spec ablation-OFF
state at those levels.)

Plot: `results/M2-g2lite-ppl-vs-n.png` (log-x/log-y).

## §M2 Needle-lite (accuracy, 20 trials/cell, greedy full-forward)

| model | depth | 1,024 | 2,048 | 4,096 | 8,192 | 16,384 | 32,768 |
|---|---|---|---|---|---|---|---|
| flat | 0.1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| flat | 0.5 | **0.95** | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| flat | 0.9 | **0.85** | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| SSRA | 0.1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| SSRA | 0.5 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| SSRA | 0.9 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

Heatmaps: `results/M2-g2lite-needle-{flat,ssra}.png`. Per-trial records
(keys, generated text, extraction) in the committed m2 JSONs. (O7 note as
above for all SSRA cells at N ≥ 2,048.)

## §O Interpretation criteria applied mechanically (assignment §4; no
architecture conclusions, spec §16 — all labels are measurement outcomes)

- **O1 anchor: PASS** (both models, §M0).
- **O2/O3 degradation labels** (r = ppl(N)/ppl(1,024)):
  - flat: 2,048 `marked` (1.568); 4,096 `marked` (4.057); 8,192 `marked`
    (9.474); 16,384 `collapse` (18.74); 32,768 `collapse` (32.74).
  - SSRA: 2,048 `marked` (4.114); 4,096–32,768 `collapse` (43.90 →
    561.77).
  - Pre-registered priors: flat was expected to leave `stable` beyond
    1,024 — confirmed. SSRA was expected `stable`/`mild` if the §1.2
    structural story holds — **violated at every N ≥ 2,048; reported as a
    finding** (per §1.3 both directions publishable).
- **O4 crossover: none.** ppl_SSRA(N) > ppl_flat(N) at every grid N;
  ratio SSRA/flat: 1.111 → 2.914 → 12.02 → 18.63 → 21.35 → 19.06 (the
  known +10.22 % G1 gap at the training protocol corresponds to the 1.111
  starting point on E).
- **O5 needle floor rule: not triggered** — pooled 1,024 accuracy: flat
  60 % ≥ 20 % (0.00/0.95/0.85 at depths 0.1/0.5/0.9), so full grids are
  reported. flat retains 0 % of its own 1,024 accuracy at every N ≥
  2,048 ⇒ `retrieval-retaining` at no N. SSRA is 0 % in every cell
  including 1,024 (its own baseline is 0 ⇒ the retention label is not
  applicable); the copy behavior exists in the flat artifact, so the
  suite is informative.
- **O6 per-position signature at N=32,768:** neither model is
  `positionally graceful` — every bucket > 1,024 exceeds the own
  513–1,024 bucket mean by ≫ +0.10 nats (flat: +0.99 … +4.11; SSRA:
  +2.87 … +6.83). Bucket profiles reported without label (§M1).
- **O7:** stated in §M1 and §M2 wherever SSRA numbers at N ≥ 2,048
  appear.

## §L Ledger

- Session window: pod created ≈ 15:29 UTC (console), first SSH 15:32:42,
  AP-23 terminate 16:44:59, confirmed dead 16:45:10 ⇒ ≈ 1.2–1.3 h ×
  $0.50/hr ≈ **$0.61–0.65 ≈ 0.53–0.57 EUR [ODHAD]** (ECB 1.1430; ≪ 10 EUR
  scoped cap, ≈ 18× margin). **Console total ≥ 2 h after termination is
  FINAL** (2026-07-14 rule) — append row pending Daniel's readout.
- Cumulative M2 after session [ODHAD]: 71.78 + ≈0.6 ≈ **72.4 EUR ≈ 24 %
  of 300** (final after console readout).
