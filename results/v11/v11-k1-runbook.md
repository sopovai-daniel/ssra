# V11 K1 runbook — VM window (path (ii), assignment §5)

Prepared 2026-07-20 (Phase A, 0 EUR). Gates already passed: G-V11-2
(0.727 ≤ 3.00 EUR, report §S0-D). Wall cap G-V11-3: **4.0 h from VM create**.
Scope: the EXPLICIT 104-object list — `results/v11/v11-k1-manifest-ssra.tsv`
+ `v11-k1-manifest-flat.tsv` (52 rows each, generated from the committed
S0-A listings by `scripts/v11_ckpt_extract.py manifest`; no prefix
globbing; `latest.pt` and the stray nested object excluded by
construction). Nothing is written to the bucket; no user-managed keys (R7).

## 0. Phase A status (local, 0 EUR)

- Extraction script `scripts/v11_ckpt_extract.py` smoke-tested end-to-end
  on dummy checkpoints built through the repo's own `save_checkpoint` code
  path (production blob structure, synthetic optimizer-state tensors, no
  gradients/forwards): `torch.load(map_location="cpu", weights_only=True)`
  loads the blob format cleanly; pairwise T-D path executed (upd_l2 > 0 on
  all perturbed pairs); φ/e_ℓ key census asserted; init-validation both
  branches exercised (correct seed → max rel drift 0.051 < 0.5 VALIDATED;
  wrong seed → 1.586 ≥ 0.5 DROPPED).
- Local torch 2.12.0 == pod torch 2.12.0+cu126 (run-log meta); model
  construction files unchanged since run commit `a7ea0f5` (`git diff
  --stat` empty) ⇒ init reconstruction is attempted on the VM with pinned
  `torch==2.12.0`, used ONLY if validated at S_min (pre-registered
  threshold in the script docstring; no silent switching).

## 1. IAM pre-flight (executed read-only 2026-07-20, local user auth)

- Project: `ssra-poc` (number 1045608942004). Default compute SA:
  `1045608942004-compute@developer.gserviceaccount.com` — exists, enabled
  (`gcloud iam service-accounts list`; `ssra-runpod@…` is disabled per R7).
- `compute.googleapis.com` is enabled.
- **Project IAM:** `gcloud projects get-iam-policy ssra-poc --flatten
  "bindings[].members" --filter "bindings.members:1045608942004-compute@…"`
  → **empty** (the automatic Editor grant is not present — consistent with
  the org-policy hardening).
- **Bucket IAM:** `gcloud storage buckets get-iam-policy gs://ssra-poc-ew3`
  → only legacy project-convenience roles + `ssra-runpod@…`
  `roles/storage.objectAdmin`. The compute SA holds **no** read access.
- **⇒ The bucket-level grant below is REQUIRED before the VM can read.**

## 2. Command blocks

Blocks marked **[Daniel]** are console actions (Daniel executes verbatim).
Blocks marked **[CC]** are driven by CC over `gcloud compute ssh`/`scp`
under Daniel's local auth. Zone: `europe-west3-c` (bucket region).

### D1 [Daniel] — bucket read grant (before create; minimal, bucket-level)

```
gcloud storage buckets add-iam-policy-binding gs://ssra-poc-ew3 \
  --member="serviceAccount:1045608942004-compute@developer.gserviceaccount.com" \
  --role="roles/storage.objectViewer"
```

Teardown pair (run in D4 below AND verify in the ≤ 2026-08-31 teardown
checklist):

```
gcloud storage buckets remove-iam-policy-binding gs://ssra-poc-ew3 \
  --member="serviceAccount:1045608942004-compute@developer.gserviceaccount.com" \
  --role="roles/storage.objectViewer"
```

### D2 [Daniel] — VM create (STARTS the 4.0 h wall cap, G-V11-3)

```
gcloud compute instances create ssra-v11-k1 \
  --project=ssra-poc \
  --zone=europe-west3-c \
  --machine-type=e2-standard-4 \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --boot-disk-size=30GB \
  --boot-disk-type=pd-balanced \
  --service-account=1045608942004-compute@developer.gserviceaccount.com \
  --scopes=storage-ro
```

(`--scopes=storage-ro` = `devstorage.read_only` via the metadata server —
read-only auth, no keys. Debian 12 images ship `google-cloud-cli`, used on
the VM for `gcloud storage cp` under the attached SA.)

### C1 [CC] — ship inputs (repo tarball; upload = free ingress)

```
git archive --format=tar.gz -o <scratch>/ssra-repo.tar.gz HEAD
gcloud compute scp <scratch>/ssra-repo.tar.gz ssra-v11-k1:~ \
  --project=ssra-poc --zone=europe-west3-c
```

### C2 [CC] — bootstrap (venv, CPU-only torch pinned to the pod version)

```
gcloud compute ssh ssra-v11-k1 --project=ssra-poc --zone=europe-west3-c --command='
  sudo apt-get -qq update && sudo apt-get -qq install -y python3-venv &&
  python3 -m venv ~/venv &&
  ~/venv/bin/pip -q install --index-url https://download.pytorch.org/whl/cpu torch==2.12.0 numpy &&
  mkdir -p ~/repo ~/extracts ~/ckpt-tmp &&
  tar xzf ~/ssra-repo.tar.gz -C ~/repo &&
  ~/venv/bin/python -c "import torch, numpy; print(torch.__version__, numpy.__version__)"'
```

### C3 [CC] — init reconstruction on the VM (both arms, seconds)

```
gcloud compute ssh ssra-v11-k1 --project=ssra-poc --zone=europe-west3-c --command='
  cd ~/repo &&
  ~/venv/bin/python scripts/v11_ckpt_extract.py reconstruct-init \
    --config experiments/m2-core-ssra-s2-850m-lr6e4.yaml --out ~/extracts/v11-k1-init-ssra.pt &&
  ~/venv/bin/python scripts/v11_ckpt_extract.py reconstruct-init \
    --config experiments/m2-core-flat-s2-850m-lr6e4.yaml --out ~/extracts/v11-k1-init-flat.pt'
```

### C4 [CC] — extraction, detached (nohup; ssh sessions are then short polls
only — dead-peer lesson from M2)

```
gcloud compute ssh ssra-v11-k1 --project=ssra-poc --zone=europe-west3-c --command='
  cd ~/repo && nohup bash -c "
    ~/venv/bin/python scripts/v11_ckpt_extract.py extract --arm ssra \
      --manifest results/v11/v11-k1-manifest-ssra.tsv --workdir ~/ckpt-tmp \
      --init ~/extracts/v11-k1-init-ssra.pt \
      --expect-run-name m2-core-ssra-s2-850m-lr6e4 \
      --out ~/extracts/v11-k1-extract-ssra.npz &&
    ~/venv/bin/python scripts/v11_ckpt_extract.py extract --arm flat \
      --manifest results/v11/v11-k1-manifest-flat.tsv --workdir ~/ckpt-tmp \
      --init ~/extracts/v11-k1-init-flat.pt \
      --expect-run-name m2-core-flat-s2-850m-lr6e4 \
      --out ~/extracts/v11-k1-extract-flat.npz
  " > ~/extract.log 2>&1 & echo detached'
```

Poll (repeat as needed):

```
gcloud compute ssh ssra-v11-k1 --project=ssra-poc --zone=europe-west3-c \
  --command='tail -n 3 ~/extract.log'
```

Streaming discipline is inside the script: one object at a time
(download → size check vs manifest → `weights_only=True` load → metrics →
delete local copy); previous state dict kept in RAM only (T-D pairwise);
NPZ rewritten atomically every 10 checkpoints (partial-scp safety).

### C5 [CC] — scp extracts to the Mac (expected ~65 MiB, hard bound 1 GiB)

```
gcloud compute scp 'ssra-v11-k1:~/extracts/v11-k1-extract-*.npz' \
  results/v11/ --project=ssra-poc --zone=europe-west3-c
gcloud compute scp ssra-v11-k1:~/extract.log \
  results/v11/v11-k1-extract.log --project=ssra-poc --zone=europe-west3-c
```

Size note: the S0-D egress line assumed ~21 MiB [ODHAD]; the correct
figure with all 15 layers' φ + e_ℓ tensors at 52 steps is ≈ 65 MiB fp32
(SSRA arm; flat arm ≲ 1 MiB — norms only). Still ≪ the 1 GiB bound the
projection already priced (€0.105) — no gate impact.

### D3 [Daniel] — VM delete IMMEDIATELY after C5

```
gcloud compute instances delete ssra-v11-k1 \
  --project=ssra-poc --zone=europe-west3-c --quiet
```

### D4 [Daniel] — IAM grant removal (right after D3; teardown pair of D1)

Run the `remove-iam-policy-binding` block from D1.

## 3. Time budget (wall cap 4.0 h from D2)

| # | Segment | Budget | Cumulative |
|---|---|---|---|
| 1 | VM create + ssh readiness | 0:05 | 0:05 |
| 2 | C2 bootstrap (apt, venv, torch 2.12.0 CPU wheel) | 0:15 | 0:20 |
| 3 | C1/C3 tarball scp + init reconstruction | 0:10 | 0:30 |
| 4 | SSRA arm: 52 × ≤ 90 s (download ≤ 20 s + load ≤ 40 s + metrics ≤ 30 s) | 1:18 | 1:48 |
| 5 | flat arm: 52 × ≤ 90 s | 1:18 | 3:06 |
| 6 | C5 scp extracts + log (~65 MiB) | 0:10 | 3:16 |
| — | slack to cap | 0:44 | 4:00 |

**Pre-registered abort rule:** at T+3.0 h, if fewer than 78 of 104
manifest objects are processed → stop extraction, scp the partial NPZs
(atomic 10-checkpoint flushes make them valid), Daniel deletes the VM,
and the shortfall goes to §Deviations.

## 4. After the window

CC writes report §K1 + §Deviations + §Ledger (scoped cap 3.00 EUR;
console-authoritative billing per the ≥ 2 h rule; ledger row in
`results/runs.md`) → oversight review (Claude). Checkpoint deletion is
OUT of this session (post-oversight, Daniel; pre-delete listing expects
107 objects).
