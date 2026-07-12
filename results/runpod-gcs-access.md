# Task report — RunPod GCS access (service account + key): BLOCKED at org-policy gate

**Date:** 2026-07-12 · **Milestone:** M2 (compute pivot to RunPod, D-log 2026-06-17,
`docs/handover/HO-08-*`) · **gcloud:** Google Cloud SDK 572.0.0 ·
**Outcome:** HARD STOP per zadanie pre-flight gate. **No infrastructure was created
or modified.**

---

## 1. Result

**SA key creation blocked by org policy — Daniel decides (exempt ssra-poc vs
workload identity).**

The effective policy on `ssra-poc` enforces `iam.disableServiceAccountKeyCreation`:

```
$ gcloud resource-manager org-policies describe iam.disableServiceAccountKeyCreation \
    --project=ssra-poc --effective
booleanPolicy:
  enforced: true
constraint: constraints/iam.disableServiceAccountKeyCreation
```

Enforcement is inherited: `ssra-poc` belongs to organization `469897210871`
(`gcloud projects describe ssra-poc`), and orgs created on/after 2024-05-03 get
this constraint enforced by default.

## 2. What was (and was not) done

- Pre-flight auth OK: active account `daniel@sopovai.com`, configuration `ssra`,
  project `ssra-poc`.
- The primary check command from the zadanie
  (`gcloud org-policies describe … --project=ssra-poc --effective`) failed because
  the Org Policy API (`orgpolicy.googleapis.com`) is not enabled on `ssra-poc`.
  Enabling it was outside approved scope, so the effective policy was read via the
  legacy Resource Manager API instead (read-only, already enabled) — command and
  output in §1. The managed variant
  (`iam.managed.disableServiceAccountKeyCreation`) could not be checked without
  the Org Policy API, but the legacy constraint alone is enforced, which is
  sufficient to trigger the gate.
- Per the gate: **STOP before STEPS.** Not created: service account, bucket IAM
  binding, JSON key. Not touched: org policies, `.gitignore`, any file under
  `docs/*` or `paper/*`. No secret material exists anywhere.

## 3. Decision needed (proposed D-log entry)

> **D-xx (proposed):** `iam.disableServiceAccountKeyCreation` is enforced on
> `ssra-poc` (inherited from org `469897210871`). RunPod GCS access therefore
> cannot use a downloaded SA JSON key as specified. Options:
> **(a)** org-policy exemption for `ssra-poc` (Daniel, via console/org admin —
> explicitly out of CC scope), then re-run this task unchanged;
> **(b)** keyless auth: workload identity federation (or another keyless
> mechanism) for the RunPod box — requires a revised zadanie.
> Note: creating the service account itself and the bucket-scoped
> `roles/storage.objectAdmin` binding is *not* blocked by this policy and is
> needed under either option; it was deferred only because the gate mandates a
> full stop.

## 4. Verification of clean state

- `gcloud auth list`: active account `daniel@sopovai.com` (never changed).
- No key file at `$HOME/ssra-secrets/` (directory not created).
- `git status`: only this report added; no secret material in the repo.

---

# Follow-up (2026-07-12) — decision (a) executed: exemption + SA + bucket-scoped key

**Decision:** Daniel chose option (a) from §3 (D-log entry to be logged by the
Claude.ai side after this report). **gcloud:** Google Cloud SDK 572.0.0.

## 5. Org-policy exemption (project scope ONLY)

- `orgpolicy.googleapis.com` enabled on `ssra-poc` (management plane; no policy
  change by itself).
- `roles/orgpolicy.policyAdmin` was **already granted** to `daniel@sopovai.com`
  at org level — no IAM grant was made in this task.
- Effective state before override:
  - `iam.disableServiceAccountKeyCreation` → `enforce: true` (inherited)
  - `iam.managed.disableServiceAccountKeyCreation` → `enforce: false`
    (not enforced; recorded, no override needed)
- Override applied — project-level policy
  `projects/ssra-poc/policies/iam.disableServiceAccountKeyCreation` with
  `enforce: false`, via `gcloud org-policies set-policy` on a `/tmp` spec file
  (removed afterwards). Applied **by Daniel in a separate terminal** (the CC
  permission layer declined to execute the org-governance write itself);
  re-verified by CC.
- Effective state after override: `enforce: false` on `ssra-poc`.
  **Org-level policy untouched; no other project touched.**
- **TEMPORARY:** this override is to be REVERTED after M2/M3 — delete the SA key
  and restore enforcement (remove the project-level policy so the org default
  applies again).

## 6. Service account, binding, key

- **SA:** `ssra-runpod@ssra-poc.iam.gserviceaccount.com`
  (display name "SSRA RunPod GCS access", project `ssra-poc`).
- **IAM:** single bucket-scoped binding — `roles/storage.objectAdmin` on
  `gs://ssra-poc-ew3` only. No project-level roles, no other buckets.
- **Key:** JSON key at `$HOME/ssra-secrets/ssra-runpod-key.json`, mode `0600`.
  **Out-of-repo; never committed; never printed.** `.gitignore` extended with
  `*-key.json` and `ssra-secrets/` (belt-and-suspenders).

Commands run (Phase 4):

```
gcloud iam service-accounts create ssra-runpod --project=ssra-poc \
  --display-name="SSRA RunPod GCS access"
gcloud storage buckets add-iam-policy-binding gs://ssra-poc-ew3 \
  --member="serviceAccount:ssra-runpod@ssra-poc.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"
mkdir -p $HOME/ssra-secrets
gcloud iam service-accounts keys create $HOME/ssra-secrets/ssra-runpod-key.json \
  --iam-account=ssra-runpod@ssra-poc.iam.gserviceaccount.com
chmod 600 $HOME/ssra-secrets/ssra-runpod-key.json
```

## 7. Key verification (2026-07-12)

Authenticated **with the key only** via per-invocation
`CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE` (no `gcloud auth activate-service-account`,
so Daniel's stored credentials were never touched):

- `gcloud storage ls gs://ssra-poc-ew3/` → listed `phase0/` ✓
- Write+read+delete round-trip on throwaway object
  `gs://ssra-poc-ew3/_verify/cc-key-check.txt` ✓ (object removed)
- Afterwards `gcloud auth list` → active account `daniel@sopovai.com` ✓

## 8. Deferred

RunPod-side key injection (env var / RunPod Secret) is a **separate task** —
not started here. Reminder: revert of the org-policy override + key deletion is
scheduled for after M2/M3 (§5).
