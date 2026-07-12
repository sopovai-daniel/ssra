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
