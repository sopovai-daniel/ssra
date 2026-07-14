# M2 Phase 3 — S2 core pair (Gate G1 inputs)

Assignment: `docs/cc/M2-phase3-core-pair.md` v1 (2026-07-14). Runs:
`m2-core-flat-s2-850m`, `m2-core-ssra-s2-850m` (AP-21). This report follows
assignment §7; sections beyond §0 are filled during/after the pod session.
No quality or architecture conclusions beyond the G1 metric anywhere in this
report (spec §16); the G1 verdict itself is Daniel's.

## §0 Local prep (2026-07-14, no pod, no spend, no GCS writes)

### §0.1 Config generation (assignment §2)

Both configs generated from a single template
(scratchpad `m2-core-template.yaml`, substitution + verbatim sha256 injection
by script); committed at **`76fc814`**.

- `diff experiments/m2-core-flat-s2-850m.yaml experiments/m2-core-ssra-s2-850m.yaml`
  → exactly 4 differing lines: `arch`, `run_name`, `training.ckpt_dir`,
  `training.gcs_ckpt_dir`. Nothing else differs (matched-parameters +
  matched-tokens discipline, AP-8).
- sha256 hard-gate values (train/val/val-eval-2M/tokenizer) injected
  programmatically from `results/M2-data-900m-manifest.json` and re-verified
  by regex extraction against the manifest: 4/4 verbatim match in both files.
- Intervals per assignment §2 [proposal, veto]: `val_every` 200,
  `val_batches` 8, `log_every` 25, `ckpt_every` 1000.
- Warmup 778 steps = 1.4996 % of 51,880 (sweep fraction 55/3662 = 1.502 %,
  rescaled).

### §0.2 Dry-run verification (assignment §2; harness `--dry-run`, zero steps)

| run | arch | params | steps | tok/step | total tokens |
|---|---|---|---|---|---|
| m2-core-flat-s2-850m | flat | **84,301,440** | 51,880 | 16,384 | **850,001,920** |
| m2-core-ssra-s2-850m | ssra (P1) | **84,647,040** | 51,880 | 16,384 | **850,001,920** |

Both in the ~84M class and consistent with the Phase 1 calibration ledger
(84.3M / 84.6M); param gap 345,600 (+0.41 % SSRA). Total tokens
850,001,920 ≥ 850M (AP-12 floor). Dry-run path resolution: `data/m2/*.bin`
absent locally as expected (shards live only in GCS; the pod bootstrap pulls
them and `verify_data_gates` enforces the sha256 gates before any training
step); frozen tokenizer artifact present locally. Full dry-run JSON outputs
reproduced by re-running
`.venv/bin/python scripts/train.py experiments/<run>.yaml --dry-run`
at commit `76fc814`.

### §0.3 Early cost gate — executability (assignment §3 step 3; AP-20 respected)

Verified executable as a **supervisory procedure with the current harness —
no model/training code changes**. The harness already logs cumulative
pure-train throughput (`tok_per_s`; val/checkpoint time excluded,
MEAS_SKIP 10) every `log_every` 25 steps. New read-only tool
`scripts/cost_gate.py` (commit `76fc814`) differences two such records to get
windowed steady-state tok/s over ≥ 200 steps and projects
(total_tokens / tok_s) × booked rate / 3600 / ECB against the scoped cap.

Plumbing validation against the committed sweep log
`logs/m2-sweep-ssra-lr1e3-do00.log` (S1 model — numbers are plumbing
evidence only, not S2 predictions):

- window [1000, 1500] → windowed **27,067.0 tok/s** vs run summary 27,062
  and recal anchor 27,079 (−0.04 %) — reconstruction correct;
- PASS path exit 0; STOP path exercised with `--total-tokens 850001920
  --cap-eur 10` → projected 11.42 EUR > 10 EUR, exit 2, ABORTED-cost-gate
  instruction printed.

Gate invocation on the pod (rate re-checked in console on deploy day,
Pravidlo W; threshold recomputed at the booked rate before launch):

```
python3.11 scripts/cost_gate.py logs/m2-core-ssra-s2-850m.log \
    --rate-usd-hr <booked> --ecb 1.1430 --cap-eur 30 \
    --window 1000 1500 --anchor-tok-s 12335
```

At SXM $1.497/hr + ECB 1.1430 the 30 EUR cap ⇔ break-even ≈ 10,308 tok/s —
matches the assignment's ≈ 10.3k figure. Anchor sanity ±10 % vs 12,335 tok/s
is informative, non-gating.

### §0.4 AP-23 self-terminate — executability (assignment §4/§5)

Verified executable as session-side supervisory steps; no spec-governed code
touched. `scripts/pod_bootstrap.sh` step 7 (commit `76fc814`) now performs
the zero-cost AP-23 capability verification at bootstrap: `runpodctl`
presence, `RUNPOD_POD_ID` (session env, `/proc/1/environ` fallback — known
RunPod SSH behavior), pod-scoped `RUNPOD_API_KEY` presence (value never
printed, AP-17 hygiene). Non-blocking: on FAIL the AP-18 manual terminate
window becomes the primary path and the outcome is recorded here. The strict
AP-23 sequence itself (GCS listing check → results committed AND pushed →
explicit completion signal → `runpodctl remove pod $RUNPOD_POD_ID`) is a
session-side runbook executed only after all three preconditions are
confirmed; it cannot run before them by construction of the procedure.
Actual on-pod verification outcome: **pending pod session**.

### §0.5 AP-20 statement

No model or training code was modified. New/changed files are supervisory
only: two run configs, `scripts/cost_gate.py` (reads logs), bootstrap
verification block. Model code remains `9417399`-certified lineage; harness
semantics unchanged.

## §i Pre-flight record (2026-07-14)

- **AP-19 step 0 (verbatim, recorded by Daniel at deploy):** "Community price
  not capturable in deploy flow, 2026-07-14 (5th occurrence)". No backfill.
  Secure-vs-Community comparison not an expected deliverable (D-log
  2026-07-14 amendment).
- **Booked HW (console 2026-07-14, Pravidlo W):** pod `ssra-m2-core`
  (`bxwa0whm15v8mi`), A100 SXM 80 GB Secure, EUR-IS-1, 32 vCPU advertised
  (cgroup quota 27.2 → thread limits `OMP_NUM_THREADS=MKL_NUM_THREADS=27`
  exported for every process), 60 GB container disk, no network volume.
  Rate: **$1.49/hr GPU + $0.008/hr disk = $1.50/hr total**.
- **Early cost gate threshold recomputed at the booked rate:** 30 EUR ×
  1.1430 / $1.50 = 22.86 h ⇒ break-even **≈ 10,329 tok/s** on 850,001,920
  tokens (steady-state below ⇒ STOP, ABORTED-cost-gate). ECB 1.1430 carried
  (no top-up).
- **Credit check:** ≥ $40 headroom confirmed by Daniel at deploy (handoff).
- **AP-17:** secret injected at deploy as `GCP_SA_KEY_B64`; bootstrap decoded
  key (SA `ssra-runpod@ssra-poc.iam.gserviceaccount.com`), **sanity gate
  `gsutil ls gs://ssra-poc-ew3` PASSED before any billable work**.
- **AP-23 capability verification (bootstrap step 7): PASSED** — runpodctl
  present (`/usr/bin/runpodctl`), `RUNPOD_POD_ID=bxwa0whm15v8mi`, pod-scoped
  `RUNPOD_API_KEY` present (value not printed). Self-terminate is the primary
  idle eliminator; manual window is fallback.
- **Repo delivery:** git bundle at `73a783c` (HTTPS clone not attempted;
  bundle is the known path), cloned clean on pod at `/workspace/ssra`.
- **Terminate window:** projected session ≈ 21.5 h [ODHAD] posted to Daniel
  pre-deploy; refined ETA posted after the SSRA early cost gate (§3.4).

## §ii Environment snapshot + pytest

- GPU: NVIDIA A100-SXM4-80GB (81,920 MiB), driver 580.159.04.
- torch 2.12.0+cu126 (bootstrap pin; image shipped 2.4.1+cu124), CUDA
  runtime 12.6, python 3.11.10, gsutil 5.37.
- Image: project cu124 class (container `CUDA_VERSION=12.4.1`); image
  tag/digest not exposed in the pod env — Daniel's console deploy record is
  authoritative (same limitation as prior pods).
- Code: commit `73a783c` (model code `9417399` lineage unchanged).
- Data shards byte-exact: train.bin 1,827,211,240 B (= 913,605,620 tok ×
  2 B), val.bin 96,101,342 B, val-eval-2M.bin 4,000,000 B; harness sha256
  gates verified 4/4 in-run (meta `sha256_verified`).
- **pytest: 64 passed, 1 failed in 51.5 s — the single failure is exactly
  the known `test_loglinear_integration`** (§B.2 precedent; box-specific).
  Full output: `logs/m2-core-pytest.log`. Gate: PASS (no OTHER failure).
## §iii Run table — pending pod session
## §iv G1 input table — pending pod session
## §v Loss-curve plots — pending pod session
## §vi P-C summary (informative, non-gating) — pending pod session
## §vii Cost ledger — pending pod session
## §viii M3 handoff (checkpoints + config hashes) — pending pod session
## §ix Deviations — pending pod session
## §x Open questions — pending pod session
