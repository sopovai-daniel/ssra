# M2 spike diagnostics — task report

**Assignment:** `docs/cc/M2-spike-diagnostics.md` v1 (2026-07-15)
**Executed:** 2026-07-15, local CPU only (zero GPU spend, zero GCS writes)
**Run under diagnosis:** `m2-core-ssra-s2-850m` (primary spike 6,475→6,500;
secondary episode 16,675–22,100 per `results/M2-core-pair.md` §xi C1)
**Binding inputs:** report §iv/§vi/§xi facts (assignment §2) — not re-measured
here.

## §0 Scope, method, statements

### §0.1 AP-20 statement (same form as core-pair report §0.5)

No model or training code was modified. New files are supervisory read-only
diagnostics: `scripts/diagnostics/{ckpt_common,t2_weight_delta,
t3_adam_moments,t4_weight_health,t5_data_window}.py` (load checkpoints /
import the harness sampler; never edit it) plus the artifacts under
`results/` named below. Model code remains `9417399`-certified lineage;
harness semantics unchanged. No training step was executed. AP-21: no
committed log or GCS object was edited.

### §0.2 GCS access method

Bucket-scoped service account per-invocation
(`CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE`, established pattern per D-log
2026-07-12), read-only operations only (`ls`, `cp` to local). Daniel's user
credentials untouched. AP-17: key path and contents not printed, copied, or
committed. `results/runs.md` untouched (no run happened).

### §0.3 Claim markers

[OVERENÉ] = reproducible from a committed artifact or script output named in
this report. [ODHAD] = analytic estimate. [HYPOTÉZA] = interpretation, not
established.

## §1 T0 — checkpoint availability gate

Command (SA form per Daniel's 2026-07-15 instruction; `--all-versions` lists
every retained object generation and IS the versioning evidence):

```
gcloud storage ls --all-versions --long 'gs://ssra-poc-ew3/m2/core/m2-core-ssra-s2-850m/**'
```

Verbatim listing (2026-07-15):

```
1016124393  2026-07-15T15:24:37Z  gs://ssra-poc-ew3/m2/core/m2-core-ssra-s2-850m/latest.pt#1784129077139209  metageneration=1
    456860  2026-07-15T15:28:21Z  gs://ssra-poc-ew3/m2/core/m2-core-ssra-s2-850m/m2-core-ssra-s2-850m.log#1784129301521880  metageneration=1
    457322  2026-07-15T15:28:21Z  gs://ssra-poc-ew3/m2/core/m2-core-ssra-s2-850m/m2-core-ssra-s2-850m.stdout#1784129301984241  metageneration=1
TOTAL: 3 objects, 1017038575 bytes (969.92MiB)
```

- **Exactly one generation per object ⇒ bucket versioning OFF ⇒ the
  step-5000/6000/7000 checkpoints do not exist.** Only the final
  `latest.pt` (step 51,880, uploaded 15:24:37Z, matching the run-end
  timeline in §xi C4) survives. [OVERENÉ — listing above]
- `gsutil versioning get` was not run: the SA (objectAdmin) lacks
  `storage.buckets.get`, so it would fail with PermissionDenied (expected;
  recorded per instruction, not chased). The generation count above is the
  versioning evidence.
- Code-level corroboration: `ssra/checkpoint.py::save_checkpoint` writes
  only `<ckpt_dir>/latest.pt` and mirrors it to GCS under the same object
  name; each `ckpt_every`-1000 upload overwrote the previous one. Absence
  of per-step objects is the expected behaviour of the committed harness,
  not an upload failure. [OVERENÉ — `src/ssra/checkpoint.py` lines 44–48]

**Gate outcome: T1–T4 NOT EXECUTABLE** (assignment T0 rule; no substitutes
improvised). T5 proceeds.

## §2 T1–T4 — not executable; machinery status

- T1 executed only for the T5 input: `data/m2/train.bin` (1.83 GB) staged
  from `gs://ssra-poc-ew3/m2/data/m2-data-900m/train.bin`; sha256
  `6d0e47cd…9549ea0` matches `results/M2-data-900m-manifest.json` and the
  run config gate value. [OVERENÉ] `data/` is git-ignored; nothing large
  committed.
- T2/T3/T4 scripts are implemented and smoke-tested against the local
  `checkpoints/M2-phase0-cpu-smoke/latest.pt` (same AP-11 blob format):
  optimizer-state index ↔ parameter-name mapping is recovered via
  `model.named_parameters()` order and shape-verified against `exp_avg`
  (57/57 params, aliases of the shared attention θ — `readout_attn.*`,
  `pool.attn.*` — deduplicated to canonical `layers.N.attn.*`). [OVERENÉ —
  scripts run end-to-end] Loading used `torch.load(...,
  weights_only=False)` (trusted self-produced artifact, per assignment T1;
  the blob carries optimizer + RNG payloads the safe loader rejects).
- If step-adjacent checkpoints ever exist for a future run, T2–T4 execute
  in minutes on CPU with no further development.

## §3 T5 — data-window due diligence (executed)

### §3.1 Method validity

Reconstruction replays the harness's OWN data path: `batches()` is
**imported** from `scripts/train.py` (not reimplemented), the sampling
generator is `torch.Generator().manual_seed(1337)` exactly as in the run
config, and draws are replayed from step 0 (the training loop consumes
exactly one `randint` per step from this generator; validation uses a
separate generator). The run log contains a single `meta` record with
`"resumed_from": null` — one continuous run, so the replayed stream is the
stream the model saw. [OVERENÉ — `logs/m2-core-ssra-s2-850m.log`,
`scripts/train.py::batches`/`main`]

Full output: `results/M2-spike-diag-T5.log`; per-step table:
`results/M2-spike-diag-T5-steps.csv` (101 steps × 16 windows of 1,024
tokens). Invocation:

```
python scripts/diagnostics/t5_data_window.py \
  --train-bin data/m2/train.bin \
  --tokenizer artifacts/tokenizer/fineweb-edu-bpe-16384.json \
  --seed 1337 --batch-size 16 --seq-len 1024 \
  --steps 6450 6550 --baseline-windows 2048 --n-decode 5 \
  --csv results/M2-spike-diag-T5-steps.csv
```

### §3.2 Corpus baseline (2,048 independent windows, seed 20260715)

| stat (per 1,024-token window) | mean | std | p95 | p99 | max |
|---|---|---|---|---|---|
| eot_count (doc boundaries) | 0.938 | 0.917 | 3 | 3 | 5 |
| distinct_frac | 0.464 | 0.050 | 0.531 | 0.555 | 0.584 |
| entropy_nats (unigram) | 5.534 | 0.172 | 5.752 | 5.823 | 5.907 |
| longest_run | 1.229 | 1.170 | 2 | 3 | 48 |

`longest_run` is heavy-tailed (2/2048 baseline windows ≥ 8, max 48) —
z-scores on it overstate significance. [OVERENÉ]

### §3.3 Findings

With 101 steps × 4 stats ≈ 404 batch-mean z-tests, ~1 |z| ≥ 3 flag is
expected under normality [ODHAD]; 4 were observed, all mundane on
inspection [OVERENÉ]:

| step | flag | batch value vs baseline | content on decode |
|---|---|---|---|
| 6,473 | eot_count z = 3.00 | mean 1.63 vs 0.94 | more short docs; ordinary text |
| 6,487 | eot_count z = 3.27 | mean 1.69 vs 0.94; window max 4 (baseline max 5) | more short docs; ordinary text |
| 6,504 | longest_run z = 5.84 | one window with a run of 27 × token `=` (ASCII separator line); batch mean 2.94 | web separator `====…`; baseline max run is 48 |
| 6,532 | longest_run z = 3.06 | one run of 10 × `=` | same pattern |

- **Inside the primary spike interval 6,475→6,500 proper, the only flag is
  the mild doc-boundary elevation @ 6,487** (eot mean 1.69 vs 0.94; every
  per-window value within baseline range). No entropy, repetition, or
  degeneracy anomaly. [OVERENÉ]
- Window overlap (same text sampled twice) within 6,450–6,550: 2 adjacent
  overlapping pairs observed vs ~2.9 expected under uniform sampling
  [ODHAD for the expectation; observation OVERENÉ]. No reuse anomaly.
- Decoded lowest-entropy windows are ordinary FineWeb-Edu material:
  multilingual glossaries (Russian/English, Japanese romanization), an
  isnād chain, a tag-index page, a metric-units table, `=`-separator
  lines, a zooplankton data table (run of ` 9` digits), dot leaders `…`.
  Qualitatively unremarkable; no long verbatim dumps included (excerpts in
  `results/M2-spike-diag-T5.log`). [OVERENÉ]

**T5 outcome: the token window that produced the spike is statistically
and qualitatively corpus-typical on every measured statistic.** This
matches §xi C2 (flat arm saw the identical stream with only a
chance-consistent wiggle) and keeps the data-window cause where the
assignment placed it: low expectation, now with direct negative evidence
on the window itself. [OVERENÉ for the measured statistics; "the data
window did not cause the spike" remains [HYPOTÉZA] — unmeasured properties
of the window cannot be excluded by construction.]

## §4 T6 — moot

T6 (CPU forward of ckpt-6000 on the spike batches) requires the step-6000
checkpoint, which does not exist (§1). NOT EXECUTABLE.

## §5 What this does and does not establish

**Establishes:**

1. Per-step checkpoints for the spike region are unrecoverable from any
   existing artifact (single-generation `latest.pt` only, versioning OFF);
   weight-delta, Adam-moment, and weight-health localization (T2–T4) and
   the activation probe (T6) are impossible for THIS run. [OVERENÉ]
2. The exact token batches of steps 6,450–6,550 are reconstructible and
   were reconstructed; they are corpus-typical on doc-boundary density,
   distinct-token fraction, unigram entropy, longest-run, and window-reuse
   statistics. [OVERENÉ]
3. Evidence inventory for the G1 verdict and the retune decision is
   therefore log-only: loss trajectories (§iv, §xi C1), P-C attention
   entropy (symptom-not-precursor, §2 of the assignment), lr/schedule
   values at the spike, AP-24 retro-trigger quantification (§xi C5), plus
   this report's negative data-window result. Gradient-scale evidence does
   not exist anywhere (grad norms unlogged — §xi C3; Adam moments only in
   destroyed checkpoints). [OVERENÉ]

**Does not establish:** any mechanism of the instability; any module-,
layer-, or component-level localization; anything about P1/P2/P3 ranking or
the architecture (spec §16 respected — no architecture conclusions, no
redesign proposals). A data-window interaction is disfavoured, not
excluded.

**Surfaced to Daniel (decisions, not proposals):** (a) whether the retune
iteration proceeds on log-only evidence; (b) whether any future run should
retain step-tagged checkpoints and/or log grad norms — noted here strictly
as the two evidence gaps that made T2–T4/T6 impossible, instrumentation
choice is Daniel's; (c) the final `latest.pt` (step 51,880, post-settling)
is the only weight artifact if M3 needs one (usability caveat already in
core-pair report §xi).

## §6 Artifacts

| artifact | role |
|---|---|
| `scripts/diagnostics/ckpt_common.py` | blob loading, optimizer-index→name mapping (shape-verified), block taxonomy |
| `scripts/diagnostics/t2_weight_delta.py` | T2 (ready; not executable this run) |
| `scripts/diagnostics/t3_adam_moments.py` | T3 (ready; not executable this run) |
| `scripts/diagnostics/t4_weight_health.py` | T4 (ready; not executable this run) |
| `scripts/diagnostics/t5_data_window.py` | T5 (executed) |
| `results/M2-spike-diag-T5.log` | full T5 output (baseline, flags, overlaps, excerpts) |
| `results/M2-spike-diag-T5-steps.csv` | per-step window statistics, steps 6,450–6,550 |

Checkpoints/shards not committed (`checkpoints/`, `data/` git-ignored —
verified with `git check-ignore`). Zero GPU spend; zero GCS writes;
`results/runs.md` untouched.
