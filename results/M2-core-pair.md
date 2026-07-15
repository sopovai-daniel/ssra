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
Actual on-pod verification outcome (bootstrap step 7): **PASSED**.

**AP-23 execution record (2026-07-15):** preconditions confirmed in order
(GCS listing verified → results pushed at `ae3b882` → completion signal
15:32:08Z). First `runpodctl remove pod` invocation FAILED (recorded
verbatim per §5): `Runpod config file not found` + `API key not found` —
`RUNPOD_POD_ID` and `RUNPOD_API_KEY` are absent in SSH sessions (the known
RunPod behavior; bootstrap's step-7 check passed because it reads PID-1
env). Immediate retry with both values sourced from `/proc/1/environ`
(values not printed) succeeded at ≈ 15:32:40Z: `pod "bxwa0whm15v8mi"
removed`; SSH connection-refused confirmed post-terminate. Manual fallback
window not needed. Lesson recorded for future pods: source both AP-23
values from PID-1 env in the terminate command itself.

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
## §iii Run table (final)

| run | config commit | status | tok/s | peak GiB | final_eval_loss | wall | EUR [ODHAD, console authoritative] |
|---|---|---|---|---|---|---|---|
| m2-core-flat-s2-850m | 76fc814 (`b213648cde31a045`) | DONE, stable | 137,251.9 | 10.846 | **3.21201** | 7,358.8 s ≈ 2.04 h | ≈ 2.68 |
| m2-core-ssra-s2-850m | 76fc814 (`d66fb5755983e7a7`) | DONE, **unstable (OQ-1 spike, unrecovered)** | 12,375.7 | 41.2 | **7.55885** | 70,467.2 s ≈ 19.57 h | ≈ 25.69 ≤ 30 cap ✓ |

- Launches: flat 2026-07-14 17:46:23Z, SSRA 19:50:19Z; seed 1337 both;
  identical token stream/step schedule by construction (same template,
  §0.1); final_eval protocol byte-identical (1953 windows / 1,999,872 tok /
  127 dropped, both).
- **Early cost gate (§3, steps 1000–1500, 500-step window): PASS** —
  steady 12,387.1 tok/s ⇒ projected $28.59 ⇒ **25.01 EUR ≤ 30 EUR scoped
  cap** (rate $1.50/hr, ECB 1.1430; break-even 10,329 tok/s). Anchor sanity
  +0.42 % vs 12,335 (informative). Actual run cost ≈ 25.69 EUR — 0.7 EUR
  over the projection (val/ckpt overhead), still ≤ 30 EUR cap.
- Flat wall 2.04 h vs 1.6–1.8 h [ODHAD] — delta = val passes + checkpoints
  + final_eval, not throughput (137.3k tok/s ≈ calibration +6.6 %).

## §iv G1 input table (verdict is Daniel's)

| model | final_eval_loss (nats/tok, val-eval-2M) | **val ppl @ ctx 1024** |
|---|---|---|
| flat | 3.21201 | **24.829** |
| SSRA-P1 | 7.55885 | **1,917.639** |

- **Relative gap: SSRA/flat ppl ratio 77.23× (+7,623 % — outside ±5 % by
  three orders of magnitude).**
- **Stability evidence:** flat — monotone descent, no anomalies (full curve
  committed, §v). SSRA — catastrophic finite loss spike in the 25-step
  window 6,475 → 6,500 (train 3.97 → 7.45 nats; details §x OQ-1),
  **never recovered** over the remaining 45,380 steps; val loss 3.93 →
  7.53–8.4 band to run end; final_eval 7.55885 vs pre-spike val ≈ 3.93.
  No NaN/inf at any step (the §3 divergence trigger never fired); both
  runs completed all 51,880 steps.
- **CC recommendation (input only, no architecture conclusion per spec
  §16/§8): both G1 arms read FAIL on these inputs — training not stable,
  gap not within ±5 %.** Interpretation and the verdict are Daniel's;
  per assignment §1, on a G1-fail signal CC stops after diagnostics —
  no autonomous redesign, no retry, no re-tuning.

## §v Loss-curve plots

`results/M2-core-curves-flat.png`, `results/M2-core-curves-ssra.png`
(train faint, val marked), committed; mirrored to
`gs://ssra-poc-ew3/m2/core/plots/`. The SSRA plot shows the step-6,500
discontinuity and the post-spike plateau explicitly.

## §vi P-C summary (informative, non-gating)

**`p1_attn_entropy` ≈ ln(32) = 3.4657 uniformity did NOT persist at 850M
tokens.** It held exactly (3.4655–3.4656) from step 0 through the spike
onset itself (3.4645 at step 6,500), then de-uniformized only AFTER the
loss spike — drifting to a 3.0–3.3 band with participation spreading from
[0.05, 0.11] to [0.03, 0.23] by run end. Sequencing note for P-C: the
de-uniformization is a post-spike symptom, not a precursor — entropy was
textbook-uniform in the exact window where the loss exploded.

## §vii Cost ledger (vs 300 EUR envelope)

| item | wall | ≈ EUR @ $1.50/hr, ECB 1.1430 [ODHAD] |
|---|---|---|
| flat run | 2.04 h | 2.68 |
| SSRA run (scoped 30 EUR cap) | 19.57 h | **25.69 ≤ 30 ✓** |
| bootstrap + pytest + evals + uploads + gaps | ≈ 0.9 h | ≈ 1.2 |
| **pod session total (create → terminate)** | ≈ 22.5 h | **≈ 29.5** |

Billed console total: pending (final ≥ 2 h post-termination per the D-log
2026-07-14 read-out rule; append-only corrections). Scoped-cap accounting:
the SSRA run alone ≈ 25.69 EUR ≤ 30 EUR ✓; flat + overhead under unchanged
AP-12 ✓. Cumulative M2 after Phase 3 ≈ 11.75 + 29.5 ≈ **41.3 EUR ≈ 13.8 %**
of 300 — the 50 % threshold is not approached.

## §viii M3 handoff

| model | final checkpoint (GCS) | config (commit 76fc814, sha256-16) |
|---|---|---|
| flat | `gs://ssra-poc-ew3/m2/core/m2-core-flat-s2-850m/latest.pt` (964.97 MiB, 2026-07-14T19:49:37Z) | `b213648cde31a045` |
| SSRA-P1 | `gs://ssra-poc-ew3/m2/core/m2-core-ssra-s2-850m/latest.pt` (969.05 MiB, 2026-07-15T15:24:37Z) | `d66fb5755983e7a7` |

Caveat for M3: the SSRA checkpoint is post-spike state (val ≈ 7.55) — its
usability for M3 evaluations is part of Daniel's G1 verdict. Raw logs:
repo `logs/m2-core-*` + GCS run dirs + `gs://ssra-poc-ew3/m2/core/`
(pytest, bootstrap, env snapshot).
## §iv G1 input table — pending pod session
## §v Loss-curve plots — pending pod session
## §vi P-C summary (informative, non-gating) — pending pod session
## §vii Cost ledger — pending pod session
## §viii M3 handoff (checkpoints + config hashes) — pending pod session
## §ix Deviations (all explicit, none silent)

1. Image tag/digest not recorded from inside the pod (no env key exposes
   it) — console deploy record is authoritative; same limitation as prior
   pods. No secret material involved.
2. Flat run wall 2.04 h exceeded the 1.6–1.8 h [ODHAD] band (val/ckpt/eval
   overhead); throughput itself was on-anchor. No cost impact of note.
3. SSRA run cost ≈ 25.69 EUR exceeded the gate-time projection 25.01 EUR
   (same overhead cause); within the 30 EUR scoped cap at all times.
4. OQ-1 in-flight decision: the assignment enumerates abort only for
   NaN/inf; on the finite unrecovered spike CC surfaced the decision to
   Daniel with diagnostics (2026-07-15 session + commit 81dfeb2) and, with
   no instruction received, continued per the assignment-literal default to
   completion. No re-tuning, no retry, no config edit (§8 respected).
5. AP-23 self-terminate executed by CC from the pod as the last action
   (per §5 strict sequence); manual window not needed.

## §x Open questions

- **OQ-1 (2026-07-15, in-flight): SSRA loss spike without NaN/inf.**
  `m2-core-ssra-s2-850m` train loss jumped 3.97 → 7.45 in the 25-step window
  6,475 → 6,500 (lr 9.73e-4, no coincident val/ckpt boundary; grad-clip 1.0
  active) and has not recovered for 4,500+ steps (train 7.2–8.4, val
  7.6–8.4 vs 3.93 at step 6,400). Loss finite throughout — the §3 NaN/inf
  abort trigger did not fire. P-C: `p1_attn_entropy` held ≈ ln(32) through
  the spike itself (3.4645 @ 6,500) and de-uniformized only afterwards
  (3.01 @ 11,000; participation [0.03, 0.17]) — symptom, not precursor.
  Assignment does not enumerate this case; per the ambiguity rule the
  decision (continue vs abort) was surfaced to Daniel in-session with
  diagnostics + recommendation (continue: pre-approved budget, no-retry rule
  makes this the one shot, abort forecloses a recovered result). Run left
  RUNNING pending his call; no instruction arrived; run completed all
  51,880 steps (deviation §ix-4). **Proposed D-log entry:** "2026-07-15:
  M2 Phase 3 SSRA 850M run exhibited a finite non-recovering loss spike at
  step ~6,500 (no NaN/inf; the §3 trigger never fired); CC surfaced the
  decision in-flight and continued per assignment default to completion;
  G1 inputs delivered (flat ppl 24.83, SSRA ppl 1917.64, ratio 77.2×,
  stability evidence report §iv/§v); G1 verdict: [Daniel]."
- **OQ-2: enumerate a finite-instability STOP trigger for Phase 4+?**
  Under current rules the SSRA run spent ≈ 20 EUR post-spike (abort is
  defined only for NaN/inf). **Proposed D-log entry:** "2026-07-15: AP-24
  [if accepted]: a val_loss regression > 2 nats vs the run's best,
  sustained ≥ 1,000 steps without recovery, is an enumerated in-flight
  STOP (status ABORTED-instability), symmetric for all models; CC executes
  without further confirmation." Daniel's call; not retroactive.
