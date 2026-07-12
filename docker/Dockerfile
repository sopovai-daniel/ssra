# M2 training image for the L4 (24GB) box @ europe-west3.
#
# STATUS: pin manifest only. This image is BUILT AND VERIFIED ON THE GPU BOX in
# Phase 1 (calibration). The base-image digest and the resolved torch CUDA build
# hash are recorded in results/M2-calibration.md once the box is live — do not
# claim it as built/tested before then (no GPU in Phase 0).
#
# Build (on the box):
#   docker build -f docker/Dockerfile -t ssra-m2:phase1 .
# Run:
#   docker run --gpus all --rm -it -v $PWD:/workspace ssra-m2:phase1 \
#       pytest tests/        # spec §14 correctness on the GPU env (AP-16 §3)

# Phase 1 outcome (D-log 2026-07-12): this image was NOT used on the box —
# the standing launch path is the official RunPod image + pod_bootstrap.sh
# (AP-18 as amended). torch pin corrected to cu126: the cu124 wheel index
# ends at torch 2.6.0 (verified 2026-07-12), so 2.12.0 resolves only from
# cu126; the wheels ship their own CUDA runtime and run on driver 550 via
# minor-version compatibility (calibration report §8.2). Base image kept at
# 12.4.1 until this Dockerfile is next actually built.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# torch first, from the CUDA 12.6 wheel index (cu126 pin per D-log 2026-07-12;
# version matches the CPU/MPS dev pin so spec §14 tests run on the same torch
# minor on both environments).
RUN python3.11 -m pip install --upgrade pip \
    && python3.11 -m pip install torch==2.12.0 \
        --index-url https://download.pytorch.org/whl/cu126

COPY requirements-gpu.txt .
RUN python3.11 -m pip install -r requirements-gpu.txt

COPY . .
RUN python3.11 -m pip install -e .

# Default: report the environment snapshot (Phase 1 calibration records this).
CMD ["python3.11", "-c", "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"]
