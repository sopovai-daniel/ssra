# Zadanie pre CC — M2 RunPod launch flow + Phase 1 calibration

**Version:** v1 (2026-07-12) · **Milestone:** M2 Phase 1 (`docs/cc/M2-assignment.md` v1.1 §4)
**Mode:** veto-based — AP-17…AP-19 introduced here stand unless Daniel vetoes them.
AP numbering continues from M2 (AP-8…AP-16); all prior APs remain in force.

**Authority chain (binding):** `docs/spec.md` v1.2 > `docs/cc/M2-assignment.md` v1.1
(milestone contract: phases, gates, configs, AP-8…AP-16) > this document (operational
launch procedure + Phase 1 execution order). On any conflict, the higher document wins.
This document does NOT modify M2-assignment; AP-10 content is void per D-log 2026-06-17
and its replacement is defined here (AP-19).

**Division of labor (hard boundary):** Daniel executes all RunPod **console** actions
(account/credit, secret creation, pod deploy/terminate) — CC prepares exact instructions
and values for him. CC executes everything **scriptable**: local prep, SSH into the pod,
tests, calibration runs, verification, report. CC never asks Daniel to paste secret
values into chat or files visible to CC.

## 1. Goal

Bring up the first paid GPU pod on RunPod safely (SA key injection per AP-17, launch
checklist per AP-18), execute M2 Phase 1 calibration exactly per M2-assignment §4,
deliver `results/M2-calibration.md`, and STOP for Daniel's Phase 2 go/no-go.
Target ≈ 1 GPU-hour paid time.

## 2. Deliverables

| # | deliverable | where |
|---|---|---|
| 1 | Pod bootstrap script: decode SA key → file, activate credentials, sanity check | `scripts/pod_bootstrap.sh` |
| 2 | Operator runbook: exact console steps for Daniel (secret creation, pod deploy, terminate) | §4 of this doc executed; deviations recorded in report |
| 3 | Calibration YAML(s) committed BEFORE launch + rows in `results/runs.md` | `experiments/`, `results/runs.md` |
| 4 | Calibration report per M2-assignment §4 Phase 1 (contents in §7 below) | `results/M2-calibration.md` |
| 5 | Cost ledger entries: measured USD, fixed ECB EUR/USD rate, EUR; cumulative vs 300 EUR | `results/runs.md` + report |

## 3. AP-17 — SA key injection (binding; secrets hygiene)

Mechanism verified in primary source (docs.runpod.io/pods/templates/secrets +
/pods/templates/environment-variables, retrieved 2026-07-12): RunPod Secrets are
encrypted strings created in the console, referenced in pod environment variables as
`{{ RUNPOD_SECRET_name }}`, substituted at pod start. Secrets are strings — the SA
JSON key file must therefore travel as single-line base64.

Flow:
1. **Local (CC prepares command, Daniel runs it):** `base64 -i ~/ssra-secrets/ssra-runpod-key.json | pbcopy`
   (macOS: single line by default). Output goes to clipboard only — never into a file
   in the repo, never into chat.
2. **Console (Daniel):** create secret `gcp_ssra_runpod_sa`, paste clipboard as value.
3. **Pod template env var (Daniel, per runbook):**
   `GCP_SA_KEY_B64={{ RUNPOD_SECRET_gcp_ssra_runpod_sa }}`
4. **Container start command / bootstrap (CC-authored `scripts/pod_bootstrap.sh`):**
   decode `$GCP_SA_KEY_B64` → `/root/.gcp/sa-key.json` (dir 700, file 600), export
   `GOOGLE_APPLICATION_CREDENTIALS=/root/.gcp/sa-key.json` (persist to shell profile),
   and if `gcloud` is present additionally
   `gcloud auth activate-service-account --key-file=/root/.gcp/sa-key.json`.
   Decoding happens in the start command because [K, community-reported] secret env
   vars may be absent in SSH-over-TCP sessions; the file makes SSH sessions independent
   of the env var.
5. **Sanity gate (blocking):** `gsutil ls gs://ssra-poc-ew3` (or, if the image lacks
   gcloud SDK, an equivalent listing via the python `google-cloud-storage` client with
   the same key file) must succeed BEFORE any training or test run. On failure: STOP,
   diagnose, no billable work.

Prohibitions (absolute): the key — raw or base64 — never appears in the Docker image,
the repo, any YAML config, any log, any report, or chat. No `env` dumps into logs.
`printenv`/`env` output is never committed. The bootstrap script itself contains no
secret material and IS committed.

Client-path check: before Phase 1 launch, CC verifies which GCS client path the image
provides (gcloud SDK vs python client) and that the checkpoint/upload code path used
by `scripts/train.py` works with `GOOGLE_APPLICATION_CREDENTIALS`. If the image lacks
both, report — do not silently pip-install into a running pod without recording it in
the environment snapshot.

## 4. AP-18 — Pod launch + lifecycle checklist (binding for every launch, Phases 1–4)

Pre-launch (CC prepares, Daniel executes in console):
- [ ] GPU per fallback ladder (D-log 2026-07-12): A100 PCIe 80GB → A100 SXM → H100 PCIe
      (speed escalation; AP-12 EUR projection before any H100 launch) → L40S / RTX 4090 / L4.
      Live availability decides; record actual GPU + tier + hourly rate in the ledger.
- [ ] Tier: compare Community vs Secure price in the deploy console on the day
      (Pravidlo W — record both numbers + date in the report). Community is admissible
      (AP-11 covers preemption).
- [ ] Region: EU preferred, not required (training data = public FineWeb-Edu, 0 PII).
- [ ] Image: the project CUDA 12.4 (cu124) image. Record image tag + digest.
- [ ] Storage: NO network volume (checkpoints go to GCS per AP-11); container disk
      minimal but sufficient for image + data shard cache + one local checkpoint
      (CC computes and states the GB number in the runbook). Note: stopped pods still
      bill storage — the lifecycle rule below avoids stopped-pod states entirely.
- [ ] Env var with secret reference set (AP-17 step 3); start command wired to bootstrap.
- [ ] Calibration YAML committed + `runs.md` row exists (run discipline — a run without
      a pre-committed config does not exist; applies to calibration too).
- [ ] Credit topped up (on-demand deploy requires balance ≥ 1 h of the chosen rate,
      per docs.runpod.io/pods/pricing 2026-07-12); ECB EUR/USD reference rate of the
      top-up day recorded in the ledger (AP-12).

Lifecycle (hard rule): pod is terminated — not stopped — immediately after the last
verification step, by Daniel in the console on CC's explicit "calibration complete,
terminate now" signal. Per-second billing makes shutdown discipline the main cost
lever. CC includes pod start/end timestamps and the billed duration in the ledger.
At balance 0 a pod without a network volume is terminated with its data (verified
2026-07-12) — irrelevant if AP-11 discipline holds (everything durable is in GCS
before termination).

## 5. Phase 1 calibration — execution order (per M2-assignment §4; no scope changes)

All steps run inside the pod via SSH (CC-driven), in this order, each gated on the
previous:
1. AP-17 sanity gate (`gsutil ls`) — blocking.
2. `pytest tests/` on the box — fp32 correctness (spec §14.1–.3, .7) must pass;
   record full output. Known M1 Triton skip is acceptable if it reproduces identically.
3. Measured tok/s + peak VRAM for S1 AND S2-candidate configs (M2-assignment §5),
   BOTH models (SSRA-P1 and flat), bf16 autocast (AP-16), short measurement runs only —
   no quality conclusions (spec §16).
4. One kill+resume verification on GPU: kill mid-run after a checkpoint, `--resume`,
   verify loss-curve continuity per AP-11.
5. EUR projection table for Phases 2–4 from measured tok/s: per-phase GPU-hours and
   EUR at the actual booked rate, plus the same table at the ladder alternatives'
   verified rates; flag any single projected run > 25 EUR and any cumulative crossing
   of 50 % / 80 % of the 300 EUR cap (AP-12 STOP triggers).
6. Upload calibration artifacts/logs to GCS; record paths.
7. Signal Daniel: terminate pod (AP-18 lifecycle rule).

## 6. AP-19 — Tier comparison (replaces void AP-10 content)

The calibration report's price-comparison section = RunPod **on-demand (Secure) vs
Community** for the chosen GPU class: verified console prices of the day (URL/console
+ date), preemption-risk note (AP-11 mitigation), and a recommendation for Phases 2–4.
Daniel decides. GCE-Spot-vs-Vertex is void (D-log 2026-06-17) and is not reported.

## 7. Report + STOP gate

`results/M2-calibration.md` must contain: (i) environment snapshot (GPU, tier, region,
image tag/digest, driver/CUDA, torch, commit hash); (ii) pytest output summary;
(iii) tok/s + peak VRAM table (S1/S2 × SSRA/flat); (iv) kill+resume evidence;
(v) EUR projection table + AP-12 flags; (vi) AP-19 tier comparison + HW
recommendation for Phases 2+; (vii) cost ledger: billed seconds, USD, ECB rate, EUR,
cumulative vs 300 EUR; (viii) any open questions with proposed D-log entries.

**STOP:** after the report is committed, CC stops. No Phase 2 work, no sweep prep
launches, no further paid time. Daniel confirms HW + plan (D-log entry on his side)
before Phase 2.

## 8. Anti-goals

No training beyond calibration measurement runs; no quality conclusions from
calibration losses; no design/spec/docs/paper edits; no contingency flags; no secret
material anywhere in repo/logs/chat (AP-17); no stopped-pod idle states; no Vertex
or GCP GPU retry paths (dead end, D-log 2026-06-17); no network volume unless Daniel
approves a documented need; no prices from memory (Pravidlo W — console values of
the day, recorded).

## 9. Definition of done

AP-17 flow executed with sanity gate passed; Phase 1 steps 1–7 complete; pod
terminated with billed duration recorded; `results/M2-calibration.md` committed;
calibration YAML(s) + `runs.md` rows + ledger complete; cumulative spend documented;
CC stopped at the STOP gate awaiting Daniel's Phase 2 confirmation.
