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

## §iii Run table — pending pod session

Order binding: flat lr6e4 first (~2.0–2.2 h [ODHAD]), then SSRA lr6e4
with the early cost gate (`scripts/cost_gate.py`, window [1000, 1500],
cap 30 EUR scoped to the SSRA run; break-even recomputed at the booked
rate; anchor 12,335 tok/s informative, non-gating; step-tagged upload
overhead ~52 × O(10 s) [ODHAD] sits outside the pure-train gate — if the
projection lands within 1 EUR of the cap, surface to Daniel before
continuing).

## §iv G1 inputs — pending pod session

## §v Plots — pending pod session

## §vi P-C diagnostics (`p1_attn_entropy`) — pending pod session

## §vii Cost ledger — pending pod session

## §viii M3 handoff — pending pod session

## §ix Deviations — pending pod session

None in local prep. (The §0.1 header-comment-block diff vs Phase 3 is
recorded there explicitly, not silently.)

## §x Open questions — pending pod session
