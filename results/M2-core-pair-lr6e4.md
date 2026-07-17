# M2 Phase 3b — S2 core pair retune @ lr 6e-4 (fresh Gate G1 inputs)

Assignment: `docs/cc/M2-phase3b-retune.md` v1 (2026-07-15); predecessor
protocol `docs/cc/M2-phase3-core-pair.md` v1 inherited verbatim unless
changed there. Runs: `m2-core-flat-s2-850m-lr6e4`,
`m2-core-ssra-s2-850m-lr6e4` (AP-21). This is the **single permitted retune
iteration**; seed 1337 held fixed — the only changed training variable vs
Phase 3 is lr 1e-3 → 6e-4 (single-variable isolation, assignment §1).
This report follows assignment §6 (Phase 3 report structure); sections
beyond §0 are filled during/after the pod session. No quality or
architecture conclusions beyond the G1 metric anywhere in this report
(spec §16); the G1 verdict itself is Daniel's.

## §0 Local prep (2026-07-16, no pod, no spend, no GCS writes)

### §0.1 Config generation (assignment §2)

Both configs generated from the committed Phase 3 configs (predecessor
§0.1 method: one template, scripted substitution + programmatic sha256
injection); committed at **`3db45ef`**.

- Pair-internal diff
  (`diff experiments/m2-core-flat-s2-850m-lr6e4.yaml experiments/m2-core-ssra-s2-850m-lr6e4.yaml`)
  → **exactly the same 4 differing lines as Phase 3**: `arch`, `run_name`,
  `training.ckpt_dir`, `training.gcs_ckpt_dir`. Nothing else differs
  (matched-parameters + matched-tokens, AP-8).
- vs the Phase 3 configs: **parsed-config delta = `training.lr`
  (1.0e-3 → 6.0e-4) + the AP-21 names** (`run_name`,
  `training.ckpt_dir`, `training.gcs_ckpt_dir`) — machine-verified by
  loading both YAMLs and diffing the flattened dicts (key sets identical).
  The textual diff additionally replaces the **header comment block**
  (8 lines → 11 lines referencing the retune assignment instead of
  Phase 3) and the inline comment on the `lr` line — documentation only,
  no parsed value involved; recorded here explicitly so no diff line is
  unaccounted for. Every other line, including every hyperparameter,
  interval, and data line, is byte-identical to Phase 3.
- lr 6e-4 (pre-declared sweep runner-up, `results/M2-sweep.md`);
  `lr_min_frac` 0.1 ⇒ cosine floor 6e-5; `warmup_steps` 778 unchanged;
  steps 51,880; batch 16; seq 1024; seed 1337; weight_decay 0.01;
  `val_every` 200; `val_batches` 8; `log_every` 25; `ckpt_every` 1000 —
  all inherited unchanged.
- sha256 hard-gate values (train/val/val-eval-2M/tokenizer) injected
  programmatically from `results/M2-data-900m-manifest.json` and
  re-verified by regex extraction against the manifest: **4/4 verbatim
  match in both files**.

### §0.2 Dry-run verification (assignment §2; harness `--dry-run`, zero steps)

| run | arch | params | steps | tok/step | total tokens |
|---|---|---|---|---|---|
| m2-core-flat-s2-850m-lr6e4 | flat | **84,301,440** | 51,880 | 16,384 | **850,001,920** |
| m2-core-ssra-s2-850m-lr6e4 | ssra (P1) | **84,647,040** | 51,880 | 16,384 | **850,001,920** |

Identical to the Phase 3 dry-run values (same model config, AP-8). Total
tokens 850,001,920 ≥ 850M (AP-12 floor). Path resolution: `val.bin` and
`val-eval-2M.bin` absent locally as expected (pod bootstrap pulls them;
`verify_data_gates` enforces the sha256 gates before any training step);
`train.bin` present locally (T5 diagnostics artifact, sha-gated on the pod
regardless); frozen tokenizer artifact present. Reproduce with
`.venv/bin/python scripts/train.py experiments/<run>.yaml --dry-run`
at commit `3db45ef`.

### §0.3 Instrumentation (assignment §3–§4; commit **`da6f828`**)

Three harness deltas, observability/run-control only — training semantics
unchanged (verified by the §0.4 gate):

1. **grad_norm logging:** the global pre-clip norm that
   `clip_grad_norm_` already computes is captured and written as
   `"grad_norm"` in every per-`log_every` JSONL train record. No change
   to clipping behavior, loss, optimizer, LR, or data paths. Closes the
   Phase 3 evidence gap (`results/M2-core-pair.md` §xi C3).
2. **Step-tagged checkpoint retention** (`src/ssra/checkpoint.py`): after
   the existing `latest.pt` write+mirror, the same local file is uploaded
   once more to `<gcs_ckpt_dir>/step-<N>.pt` (N = completed steps).
   Checkpoint blob format UNCHANGED (AP-11 test green, §0.4); local disk
   usage unchanged. ~52 objects ≈ 50 GiB per arm in GCS, temporary —
   deletion after post-run analysis is Daniel's decision; nothing is ever
   deleted by the harness. On recurrence + AP-24 stop, T2–T4 diagnostics
   (`scripts/diagnostics/`, smoke-tested in Phase 3 follow-up) become
   immediately executable on the bracketing checkpoints.
3. **AP-24 in-flight instability stop** (D-log 2026-07-15; symmetric for
   both arms, in-loop placement next to the Phase 3 §3 NaN/inf trigger):
   - Trigger: current val_loss > (running best val_loss of this run +
     **2.0 nats**) on **≥ 6 consecutive** val evaluations (`val_every`
     200 ⇒ 1,000 steps sustained, no recovery in between).
   - Action (no further confirmation): checkpoint (latest + step-tagged)
     → GCS → `ABORTED-instability` in the log summary →
     `scripts/ap24_abort.sh`: append-only runs.md mark → commit+push of
     results collected so far → AP-23 strict self-terminate (listing
     check → push confirmed → completion signal → `runpodctl remove pod`)
     with `RUNPOD_POD_ID` + `RUNPOD_API_KEY` sourced from
     `/proc/1/environ` **in the terminate step itself** (Phase 3 lesson,
     `results/M2-core-pair.md` §0.4). On any AP-23 precondition failure
     the script does NOT terminate and the AP-18 manual window becomes
     the primary path. Daniel decides resume-from-checkpoint vs close
     (AP-11 reversibility).
   - The NaN/inf trigger remains in force and takes precedence:
     non-finite train loss (checked at every `log_every` record, ≤ 25
     steps latency) or non-finite val_loss aborts immediately with
     status `DIVERGED` and **no checkpoint write** — a non-finite state
     must never overwrite the last good `latest.pt` (single
     generation/object, bucket versioning OFF; Phase 3 T0 lesson).
   - Trigger state is rebuilt from the run's own JSONL log on AP-11
     resume (checkpoint blob format untouched); a truncated tail line
     after a kill is skipped as expected JSONL damage.
   - **Retro-validation against the committed Phase 3 logs** (replaying
     every val record through the implemented update rule):
     `m2-core-ssra-s2-850m` first fire at **step 7,600**;
     `m2-core-flat-s2-850m` **0 false positives** over 261 val evals —
     exactly the accepted D-log 2026-07-15 retro-test.

### §0.4 Verification gate (assignment §3 (i)–(iv)) — AP-20-style statement

All three harness deltas (grad_norm, step-tagged mirror, AP-24 trigger)
live in one commit (`da6f828`) and are covered by this gate. Model code
remains `9417399`-certified lineage; no spec-governed math was touched; no
test was modified.

- **(i) §14 suite:** `pytest tests/` at `da6f828` → **64 passed,
  1 skipped in 10.24 s**. The single non-pass is exactly the known item
  `tests/test_baselines.py::test_loglinear_integration` — on this machine
  it manifests as SKIP ("Triton unavailable on this machine"; on the pod
  it manifests as the known allowed fail with box-specific text, §B.2 /
  Phase 3 §ii precedent). No other failure. **PASS**
- **(ii) AP-11 kill+resume unit test:**
  `pytest tests/test_checkpoint_resume.py -v` → **2 passed**
  (`test_resume_yields_continuous_loss_curve`,
  `test_checkpoint_is_atomic_and_reloadable`) — checkpoint blob format
  untouched. **PASS**
- **(iii) frozen-reference check:** 60-step CPU fp32 smoke at seed 1337
  (`experiments/m2-3b-frozenref-smoke.yaml`, committed pre-run at
  `3db45ef`), executed at the pre-instrumentation commit `3db45ef` and
  re-executed at `da6f828`: **all 60 per-step train losses and all 4 val
  losses BIT-IDENTICAL** (JSON-emitted values compared exactly; 0
  mismatches). Logs committed: `logs/m2-3b-frozenref-smoke-pre.log`
  (before) and `logs/m2-3b-frozenref-smoke.log` (after, with the new
  `grad_norm` field present as expected). **PASS**
- **(iv) exact diff scope of the instrumentation commit `da6f828`**
  (`git show --numstat`):
  `scripts/train.py` +77/−1 · `src/ssra/checkpoint.py` +6/−0 ·
  `scripts/ap24_abort.sh` +66/−0 (new file) — 3 files,
  149 insertions, 1 deletion. No other file touched.

### §0.5 Pravidlo W — GCS Standard storage pricing (step-tagged retention cost)

Verified on the official page on the day the retention code was written
(**2026-07-16**, https://cloud.google.com/storage/pricing, region table
for the bucket's region europe-west3 / Frankfurt, prices embedded in the
page's region dataset):

- Standard storage, europe-west3: **$0.023 / GiB-month** (hourly SKU
  $0.000031507 / GiB-hour on the same page; Nearline 0.013 / Coldline
  0.006 / Archive 0.0025 listed alongside — not used).
- ~100 GiB (both arms, ~52 step-tagged objects ≈ 50 GiB per arm) ⇒
  ≈ **$2.30 / month ≈ 2.01 EUR / month** at the carried ECB 1.1430.
  Storage is billed pro-rata (hourly SKU), so the realistic
  until-analysis-closes exposure is a fraction of that (≈ 0.07 $/day at
  full 100 GiB). Deletion after post-run analysis is Daniel's decision.
- RunPod egress $0 (verified 2026-07-12) and GCS ingress free — nothing
  observed that would require re-verification (assignment §3.2).

### §0.6 AP-20 statement

New/changed files in this prep: two run configs + one frozen-reference
smoke config (`3db45ef`); harness observability/run-control commit
`da6f828` exactly as scoped in §0.4(iv); this report + runs.md rows +
frozen-reference logs. No model code, no spec, no tests, no docs edits.
Configs and instrumentation are committed BEFORE any launch; no committed
log or GCS object was overwritten or deleted.

## §i Pre-flight record (2026-07-16)

- **AP-19 step 0 (verbatim, recorded by Daniel at deploy):** "Community
  price not shown in deploy flow for A100 SXM 80GB, 2026-07-16" —
  6th occurrence, no backfill.
- **Booked HW (console 2026-07-16, Pravidlo W):** pod `ne2w6airwb4401`,
  1× A100 SXM 80 GB Secure on-demand, region US (console "Location" = US;
  exact datacenter code not captured; EU preference not met — admissible
  per AP-18, public data / 0 PII). Rate **$1.49/hr GPU + $0.008/hr
  container disk (60 GB) = $1.50/hr total**; no network volume. Ladder
  record: A100 PCIe availability at booking not recorded (SXM booked).
  Console vCPU inconsistent (16 in listing / 24 in details, Xeon Platinum
  8470); **cgroup quota authoritative**: cpu.cfs_quota 2,040,000 /
  period 100,000 = 20.4 vCPU ⇒ `OMP_NUM_THREADS=MKL_NUM_THREADS=20`
  exported for every process.
- **Early cost gate threshold recomputed at the booked rate:** 30 EUR ×
  1.1430 / $1.50 = 22.86 h on 850,001,920 tok ⇒ break-even
  **≈ 10,329 tok/s** (steady-state below ⇒ STOP, ABORTED-cost-gate).
  ECB 1.1430 carried (no top-up).
- **Credit check:** $49.49 confirmed by Daniel 2026-07-16 (≥ $40 ✓;
  ≥ projected session total ≈ $33.6 with ≈ $16 margin).
- **AP-17:** secret injected as `GCP_SA_KEY_B64` (PID-1 env; SSH-session
  absence pattern confirmed again); bootstrap decoded key
  (SA `ssra-runpod@ssra-poc.iam.gserviceaccount.com`), **sanity gate
  `gsutil ls gs://ssra-poc-ew3` PASSED before any billable work**.
- **AP-23 capability check (bootstrap step 7): PASSED** — `runpodctl`
  present, `RUNPOD_POD_ID=ne2w6airwb4401` + pod-scoped `RUNPOD_API_KEY`
  in `/proc/1/environ` (values sourced from PID-1 env in the terminate
  step itself, Phase 3 lesson).
- **Repo:** git bundle at pre-launch commit `a7ea0f5`, cloned on pod,
  checked out `main` @ `a7ea0f5`.
- **Terminate window (posted pre-launch, before flat):** session total
  ≈ 22.4 h [ODHAD] ⇒ AP-23 self-terminate ETA ≈ 07:15–07:30 UTC =
  09:15–09:30 Bratislava, 2026-07-17; refined after the SSRA cost gate.
- **Note (git push from pod):** the pod has no git credentials (as in
  Phase 3) — an unattended AP-24 fire would complete checkpoint + GCS +
  runs.md mark + local commit, then stop before push and NOT
  self-terminate (designed AP-23-safe fallback); the supervising session
  then completes push + terminate. Recorded here, not a deviation.

## §ii Environment + pytest (2026-07-16, pod `ne2w6airwb4401`)

- Snapshot (`logs/m2-3b-env-snapshot.txt`): A100-SXM4-80GB 81,920 MiB,
  driver 570.172.08; template "Runpod Pytorch 2.4.0 - SSRA" shipped torch
  2.4.1+cu124 → bootstrap pinned **torch 2.12.0+cu126** (standing path,
  CUDA 12.6 on 570-series driver); Google Cloud SDK 576.0.0 (installed by
  bootstrap); commit `a7ea0f5`; container disk 60 GB (16 % used after
  shards).
- Data shards pulled from GCS; integrity enforced by the harness sha256
  hard gates at run start: **4/4 verified** (`train_bin`, `val_bin`,
  `eval_bin`, `tokenizer`) in both runs' meta records.
- **pytest (`logs/m2-3b-pytest.log`): 64 passed, 1 failed in 30.09 s** —
  the single failure is exactly the known
  `test_loglinear_integration` (`operator torchvision::nms does not
  exist`, box-specific text; §B.2 / Phase 3 §ii precedent). No other
  failure ⇒ gate passed.

## §iii Run table

| run | config commit | status | tok/s | peak GiB | final_eval_loss | wall | ≈ EUR [ODHAD] |
|---|---|---|---|---|---|---|---|
| m2-core-flat-s2-850m-lr6e4 | 3db45ef (`ed606161f99e713a`) | **DONE, stable** | 137,500.3 | 10.846 | **3.19333** | 10,542.6 s = 2.93 h | 3.84 |
| m2-core-ssra-s2-850m-lr6e4 | 3db45ef (`d35a628774f87d65`) | **DONE, stable** | 12,383.2 | 41.2 | **3.29065** | 73,794.8 s = 20.50 h | **26.90 ≤ 30 cap ✓** |

- Order per assignment §5: flat first (09:11:44→12:07:13 UTC 2026-07-16),
  then SSRA (≈ 12:14:30 UTC 2026-07-16 → 08:44:25 UTC 2026-07-17; times
  from log arithmetic, console billing authoritative).
- **Early cost gate (SSRA): PASS** — windowed steady-state
  **12,404.2 tok/s** over steps [1000, 1500] ⇒ projected 19.035 h pure
  train = 24.98 EUR ≤ 30 EUR cap at the booked $1.50/hr, ECB 1.1430.
  Margin 5.02 EUR > 1 EUR ⇒ no pre-continuation surface required.
  Anchor sanity: +0.56 % vs 12,335 tok/s (informative, within ±10 %).
  Cumulative-meter tok/s at run end 12,383.2 (+0.39 % vs anchor);
  peak 41.2 GiB = recalibration value exactly.
- Throughputs are pure-train (checkpoint/val time excluded by the meter).
  Step-tagged upload overhead observed ≈ 70 s per 1000-step interval
  (2 × ~0.95 GiB sequential uploads, US pod → EU bucket ≈ 29 MB/s) —
  above the ~O(10 s) [ODHAD]; absorbed by the cap margin as designed
  (assignment §5); see §ix.

## §iv G1 inputs (fresh pair @ lr 6e-4; verdict is Daniel's)

| model | final_eval_loss (nats/tok, val-eval-2M) | **val ppl @ ctx 1024** |
|---|---|---|
| flat lr6e4 | 3.19333 | **24.369** |
| SSRA-P1 lr6e4 | 3.29065 | **26.860** |

- **Relative gap: SSRA +10.22 % vs flat** (ppl ratio 1.10223;
  Δ = +0.09732 nats/token). The pre-registered ±5 % band (G1) is not met
  by these numbers.
- **Stability evidence (both arms, full 51,880 steps / 850,001,920 tok,
  identical token stream, seed 1337):** 0 divergence flags; no NaN/inf at
  any step (neither the §3 trigger nor AP-24 ever fired — **AP-24
  consecutive counter never left 0** across 261 val evals per arm);
  maximum single-eval val regression vs the running best:
  flat **0.00279 nats**, SSRA **0.00334 nats** (trigger margin is
  2.0 nats); post-warmup grad_norm max flat 0.887 / SSRA 1.237, late-run
  ranges 0.60–0.76 / 0.69–0.88; monotonically improving val curves to
  run end (final val = running best in both arms). **Explicit
  no-divergence statement: the Phase 3 spike did NOT recur at lr 6e-4.**
  Full evaluation protocol byte-identical: 1,953 windows / 1,999,872
  tokens / 127 dropped, both arms.
- Single-variable isolation held: `resumed_from: null`, seed 1337, sha256
  gates 4/4 in both meta records; configs differ from Phase 3 in lr and
  names only (§0.1).
- **CC recommendation (G1 input only, no architecture conclusion, spec
  §16):** on these inputs the stability criterion reads PASS (both arms
  stable end-to-end; per the assignment §1 pre-registered rationale, the
  spike vanishing under lr 6e-4 with everything else held fixed
  implicates lr for the Phase 3 instability) and the ±5 % ppl-band
  criterion reads FAIL (+10.22 %). The verdict and its interpretation
  are Daniel's; per assignment §1 this was the single permitted retune —
  no further tuning from CC.

## §v Loss-curve plots

`results/M2-core-lr6e4-curves-flat.png`,
`results/M2-core-lr6e4-curves-ssra.png` (train faint, val marked),
committed; mirrored to `gs://ssra-poc-ew3/m2/core/plots/`. Both curves
are smooth end-to-end; no discontinuity anywhere (contrast: Phase 3 SSRA
step-6,500 spike).

## §vi P-C summary (informative, non-gating)

**`p1_attn_entropy` ≈ ln(32) = 3.4657 uniformity effectively persists at
850M tokens at lr 6e-4:** start 3.4657, minimum 3.4287 (step 50,300),
final 3.4348 — a −0.9 % late drift, never leaving the 3.43–3.47 band
(contrast: Phase 3 lr 1e-3 drifted to 3.0–3.3 post-spike). Per-query
participation stayed collapse-free throughout, final band
[0.0471, 0.1003] around 1/16 = 0.0625. Sequencing input for P-C: at this
lr there is no spike and no de-uniformization episode.

## §vii Cost ledger (vs 300 EUR envelope)

| item | wall | ≈ EUR @ $1.50/hr, ECB 1.1430 [ODHAD] |
|---|---|---|
| bootstrap + pytest + inter-run gap + close-out | ≈ 1.1 h | ≈ 1.5 |
| flat lr6e4 | 2.93 h | 3.84 |
| SSRA lr6e4 (scoped 30 EUR cap) | 20.50 h | **26.90 ≤ 30 ✓** |
| **pod session total** (created 08:56 UTC 07-16 → terminate ≈ 09:2x UTC 07-17) | **≈ 24.5 h** | **≈ 32.1** |

- Scoped-cap accounting: the 30 EUR cap applies exclusively to
  `m2-core-ssra-s2-850m-lr6e4` (26.90 EUR ✓, incl. its checkpoint-upload
  overhead inside the run wall); flat + overhead sit under unchanged
  AP-12 (far below 25 EUR each).
- Completion→reconnect idle: **none** — supervision reconnected 07:59
  UTC, before the 08:44:25 UTC SSRA completion; the completion→terminate
  window is close-out work (ledgered in the overhead line, not the
  SSRA-arm cap line). The overnight supervision gap (laptop offline
  ≈ 21:00 UTC → 07:59 UTC) coincided with productive training — zero
  idle cost.
- Step-tagged GCS retention: 53 objects × ~0.94/0.95 GiB per arm ≈
  98 GiB total ≈ $2.25 ≈ 1.97 EUR/month pro-rata (§0.5 pricing);
  deletion post-analysis is Daniel's decision.
- Billed console total: to be read by Daniel ≥ 2 h after termination
  (D-log 2026-07-14 rule); decomposition above is [ODHAD], console total
  authoritative. Cumulative M2 after this session ≈ 40.53 + 32.1 ≈
  **72.6 EUR ≈ 24 % of 300** — the 50 % threshold is not approached.

## §viii M3 handoff

| model | final checkpoint (GCS) | config (commit `3db45ef`, sha256/16) |
|---|---|---|
| flat lr6e4 | `gs://ssra-poc-ew3/m2/core/m2-core-flat-s2-850m-lr6e4/latest.pt` (= `step-51880.pt`, 1,011,848,651 B) | `ed606161f99e713a` |
| SSRA-P1 lr6e4 | `gs://ssra-poc-ew3/m2/core/m2-core-ssra-s2-850m-lr6e4/latest.pt` (= `step-51880.pt`, 1,016,124,393 B) | `d35a628774f87d65` |

Plus 52 step-tagged intermediate checkpoints per arm
(`step-1000.pt` … `step-51000.pt`, `step-51880.pt`) for trajectory
analysis; retention decision Daniel's.

## §ix Deviations (all explicit, none silent)

1. Step-tagged upload overhead ≈ 70 s/interval, ~7× the ~O(10 s)
   [ODHAD] (US pod → EU bucket). Non-gating by design (pure-train gate);
   absorbed by the cap margin; flat wall +≈ 0.9 h vs the pre-deploy
   projection. Recorded, no action.
2. EU-region preference not met (US booked; admissible per AP-18,
   public data / 0 PII) — §i.
3. Supervision gap overnight (laptop offline): monitors died with the
   session; run continued detached (§i note); re-established 07:59 UTC
   before completion. No idle cost, no artifact effect.
4. AP-24 full pod-autonomy is limited by absent git credentials on the
   pod (push step; §i note) — designed AP-23-safe fallback, never
   exercised (AP-24 never fired).

## §x Open questions (with proposed D-log entries)

1. Step-tagged checkpoint retention (~98 GiB ≈ 2 EUR/month): keep until
   M3 trajectory analysis or delete after the G1 verdict? **Proposed
   D-log entry:** "2026-07-17: step-tagged ckpts of the lr6e4 pair
   [kept for M3 / deleted post-verdict] — decision Daniel."
2. AP-24 terminate tail on credential-less pods: accept the documented
   fallback (stop-without-terminate ⇒ idle-rate exposure until
   supervision) or provision a scoped push credential for future pods?
   **Proposed D-log entry:** "2026-07-17: AP-24 tail on pods without
   push credentials = [accepted fallback / scoped deploy credential
   added as AP-25]."
