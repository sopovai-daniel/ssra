#!/usr/bin/env bash
# RunPod CPU-pod bootstrap for M2 Task A data scale-up (M2-phase2-sweep.md SS2).
#
# Runs INSIDE the pod, from the repo root, after the repo has been rsynced in:
#   bash scripts/pod_bootstrap_cpu.sh
#
# CPU sibling of scripts/pod_bootstrap.sh (AP-17/AP-18 flow), differences:
#   - NO torch / CUDA install (Task A needs none — recorded deviation: the pod
#     uses the runpod-ubuntu template instead of the pytorch image),
#   - installs only the data-pipeline pins from requirements.txt,
#   - thread limits from the cgroup vCPU quota (thread-thrash lesson): with an
#     unlimited quota (cfs_quota_us = -1 / cpu.max "max") falls back to nproc,
#   - does NOT pull Phase-0 shards (Task A produces new shards).
#
# This script contains NO secret material and never prints $GCP_SA_KEY_B64,
# the key file contents, or an environment dump (AP-17 prohibitions).
set -euo pipefail

KEY_DIR=/root/.gcp
KEY_FILE=$KEY_DIR/sa-key.json
BUCKET=gs://ssra-poc-ew3
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=python3.11

log() { echo "[bootstrap-cpu] $*"; }

# ---- 1. SA key file (fallback: PID-1 environment) ----------------------------
if [[ ! -s $KEY_FILE ]]; then
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
print(f"[bootstrap-cpu] key OK: {blob['client_email']}")
EOF

export GOOGLE_APPLICATION_CREDENTIALS=$KEY_FILE
grep -q GOOGLE_APPLICATION_CREDENTIALS /root/.bashrc 2>/dev/null || \
    echo "export GOOGLE_APPLICATION_CREDENTIALS=$KEY_FILE" >> /root/.bashrc

# ---- 2. gcloud CLI + SA activation -------------------------------------------
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

# ---- 3. AP-17 sanity gate (BLOCKING) ------------------------------------------
log "AP-17 sanity gate: gsutil ls $BUCKET"
if ! gsutil ls "$BUCKET"; then
    log "FATAL: sanity gate FAILED - STOP, no billable work (AP-17 step 5)."
    exit 1
fi
log "AP-17 sanity gate PASSED"

# ---- 4. thread limits from the cgroup vCPU quota ------------------------------
quota=""
if [[ -r /sys/fs/cgroup/cpu.max ]]; then                    # cgroup v2
    read -r q period < /sys/fs/cgroup/cpu.max
    [[ $q != max ]] && quota=$(( q / period ))
elif [[ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]]; then     # cgroup v1
    q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)
    period=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
    [[ $q -gt 0 ]] && quota=$(( q / period ))
fi
NTHREADS=${quota:-$(nproc)}
for var in OMP_NUM_THREADS MKL_NUM_THREADS RAYON_NUM_THREADS; do
    export $var=$NTHREADS
    grep -q "export $var=" /root/.bashrc 2>/dev/null || \
        echo "export $var=$NTHREADS" >> /root/.bashrc
done
log "thread limits: OMP/MKL/RAYON = $NTHREADS (cgroup quota '${quota:-unlimited}', nproc $(nproc))"

# ---- 5. python environment (data-pipeline pins only, NO torch) ----------------
cd "$REPO_ROOT"
$PY -m pip --version >/dev/null 2>&1 || $PY -m ensurepip --upgrade
$PY -m pip install --no-cache-dir -q \
    PyYAML==6.0.3 numpy==2.4.6 datasets==5.0.0 pyarrow==24.0.0 \
    tokenizers==0.22.2 huggingface_hub==1.19.0
log "python deps installed (pins from requirements.txt data section; no torch)"

# ---- 6. environment snapshot ---------------------------------------------------
log "ENV SNAPSHOT:"
echo "  os:       $(. /etc/os-release && echo "$PRETTY_NAME") / $(uname -r)"
echo "  cpu:      $(nproc) vCPU  $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-)"
echo "  mem:      $(free -h | awk '/^Mem:/{print $2}') total (cgroup limit: $(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo n/a))"
echo "  disk:     $(df -h / | awk 'NR==2{print $2" total, "$4" free"}')"
echo "  python:   $($PY --version 2>&1)"
echo "  gcloud:   $(gcloud version 2>/dev/null | head -1)"
$PY - <<'EOF'
import datasets, huggingface_hub, numpy, pyarrow, tokenizers, yaml
print(f"  pins:     datasets {datasets.__version__}, pyarrow {pyarrow.__version__}, "
      f"tokenizers {tokenizers.__version__}, huggingface_hub {huggingface_hub.__version__}, "
      f"numpy {numpy.__version__}, PyYAML {yaml.__version__}")
EOF
echo "  commit:   $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"

log "DONE - pod ready for Task A (scripts/data_scale.py experiments/M2-data-900m.yaml)"
