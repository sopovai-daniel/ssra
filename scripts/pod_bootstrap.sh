#!/usr/bin/env bash
# RunPod pod bootstrap for M2 Phase 1+ (launch-flow assignment AP-17/AP-18).
#
# Runs INSIDE the pod, from the repo root, after the repo has been rsynced in:
#   bash scripts/pod_bootstrap.sh
#
# What it does, in order (each step gates the next):
#   1. Materialize the GCS service-account key from $GCP_SA_KEY_B64 (RunPod
#      Secret, injected as {{ RUNPOD_SECRET_gcp_ssra_runpod_sa }}) unless the
#      container start command already wrote /root/.gcp/sa-key.json — secret
#      env vars may be absent in SSH sessions, so the start command is the
#      primary decode path and this script is the fallback + verifier.
#   2. Install the Google Cloud CLI if absent (checkpoint.py mirrors
#      checkpoints via `gcloud storage cp`), activate the SA.
#   3. AP-17 SANITY GATE (blocking): list gs://ssra-poc-ew3. On failure the
#      script aborts — no training, no tests, no billable work.
#   4. Pin the project environment: torch cu124 wheel + requirements-gpu.txt
#      + `pip install -e .` (recorded in the calibration report's environment
#      snapshot, per the launch-flow client-path check).
#   5. Pull the Phase-0 token shards from GCS into data/phase0/.
#
# This script contains NO secret material and never prints $GCP_SA_KEY_B64,
# the key file contents, or an environment dump (AP-17 prohibitions).
set -euo pipefail

KEY_DIR=/root/.gcp
KEY_FILE=$KEY_DIR/sa-key.json
BUCKET=gs://ssra-poc-ew3
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=python3.11
TORCH_PIN=2.12.0
# Pin-manifest correction (Phase 1, 2026-07-12, verified on download.pytorch.org):
# the cu124 wheel index ends at torch 2.6.0 - torch 2.12.0 was never published
# for cu124. The version pin (2.12.0, matching the dev/test environment) is the
# load-bearing part; cu126 is the lowest CUDA build that provides it and runs
# on 12.4-era drivers via CUDA minor-version compatibility. Verified by the
# spec SS14 pytest pass on the box (results/M2-calibration.md).
TORCH_CUDA=cu126
TORCH_INDEX=https://download.pytorch.org/whl/${TORCH_CUDA}

log() { echo "[bootstrap] $*"; }

# ---- 1. SA key file ---------------------------------------------------------
if [[ ! -s $KEY_FILE ]]; then
    # Secret env vars are absent in SSH sessions and (observed 2026-07-12) the
    # start-command decode may be skipped by RunPod's command handling, so the
    # authoritative fallback is the container's PID-1 environment.
    if [[ -z ${GCP_SA_KEY_B64:-} && -r /proc/1/environ ]]; then
        GCP_SA_KEY_B64=$(tr '\0' '\n' < /proc/1/environ \
            | sed -n 's/^GCP_SA_KEY_B64=//p')
    fi
    if [[ -z ${GCP_SA_KEY_B64:-} ]]; then
        log "FATAL: $KEY_FILE missing and GCP_SA_KEY_B64 not present in the"
        log "session env or /proc/1/environ - secret not wired to the pod."
        exit 1
    fi
    mkdir -p "$KEY_DIR"
    chmod 700 "$KEY_DIR"
    printf '%s' "$GCP_SA_KEY_B64" | base64 -d > "$KEY_FILE"
    chmod 600 "$KEY_FILE"
    unset GCP_SA_KEY_B64
    log "SA key decoded -> $KEY_FILE"
else
    chmod 700 "$KEY_DIR" && chmod 600 "$KEY_FILE"
    log "SA key file already present: $KEY_FILE"
fi
$PY - "$KEY_FILE" <<'EOF'
import json, sys
blob = json.load(open(sys.argv[1]))
assert blob.get("type") == "service_account", "key file is not an SA key"
print(f"[bootstrap] key OK: {blob['client_email']}")
EOF

export GOOGLE_APPLICATION_CREDENTIALS=$KEY_FILE
grep -q GOOGLE_APPLICATION_CREDENTIALS /root/.bashrc 2>/dev/null || \
    echo "export GOOGLE_APPLICATION_CREDENTIALS=$KEY_FILE" >> /root/.bashrc

# ---- 2. gcloud CLI + SA activation -----------------------------------------
if ! command -v gcloud >/dev/null 2>&1; then
    log "gcloud absent - installing google-cloud-cli (recorded in env snapshot)"
    apt-get update -qq
    apt-get install -y -qq apt-transport-https ca-certificates gnupg curl
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
        > /etc/apt/sources.list.d/google-cloud-sdk.list
    apt-get update -qq
    apt-get install -y -qq google-cloud-cli
fi
gcloud auth activate-service-account --key-file="$KEY_FILE" --quiet
gcloud config set project ssra-poc --quiet
log "gcloud $(gcloud version 2>/dev/null | head -1), SA activated"

# ---- 3. AP-17 sanity gate (BLOCKING) ----------------------------------------
log "AP-17 sanity gate: gsutil ls $BUCKET"
if ! gsutil ls "$BUCKET"; then
    log "FATAL: sanity gate FAILED - STOP, no billable work (AP-17 step 5)."
    exit 1
fi
log "AP-17 sanity gate PASSED"

# ---- 4. project environment (pins per requirements-gpu.txt) -----------------
cd "$REPO_ROOT"
have_torch=$($PY -c "import torch; print(torch.__version__)" 2>/dev/null || echo none)
if [[ $have_torch != ${TORCH_PIN}+${TORCH_CUDA} ]]; then
    log "torch is '$have_torch' - installing ${TORCH_PIN} ${TORCH_CUDA} wheels"
    $PY -m pip install --no-cache-dir "torch==${TORCH_PIN}" --index-url "$TORCH_INDEX"
fi
$PY -m pip install --no-cache-dir -r requirements-gpu.txt
$PY -m pip install --no-cache-dir --no-deps -e .
$PY - <<'EOF'
import torch
print(f"[bootstrap] torch {torch.__version__} cuda {torch.version.cuda} "
      f"available {torch.cuda.is_available()} "
      f"device {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a'}")
assert torch.cuda.is_available(), "CUDA not available"
EOF

# ---- 5. Phase-0 token shards -------------------------------------------------
mkdir -p data/phase0
for f in train.bin val.bin shards_meta.json; do
    [[ -s data/phase0/$f ]] || gcloud storage cp "$BUCKET/phase0/data/$f" data/phase0/
done
log "data shards in place: $(ls -l data/phase0 | tail -n +2 | awk '{print $9, $5}' | tr '\n' ' ')"

# ---- 6. M2 token shards (Task A output; M2-phase2-sweep Task B) --------------
# Download only — integrity is enforced by the harness sha256 hard gates
# (train.py verify_data_gates) before any training step.
mkdir -p data/m2
for f in train.bin val.bin val-eval-2M.bin shards_meta.json; do
    [[ -s data/m2/$f ]] || gcloud storage cp "$BUCKET/m2/data/m2-data-900m/$f" data/m2/
done
log "m2 shards in place: $(ls -l data/m2 | tail -n +2 | awk '{print $9, $5}' | tr '\n' ' ')"

log "DONE - pod ready (pytest, then the committed experiments/*.yaml for this task)"
