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

# CUDA 12.4 runtime + cuDNN; matches the torch cu124 wheels pinned below.
# Pin to a digest on the box for reproducibility (recorded in calibration).
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# torch first, from the CUDA 12.4 wheel index (version matches the CPU/MPS dev
# pin so spec §14 tests run on the same torch minor on both environments).
RUN python3.11 -m pip install --upgrade pip \
    && python3.11 -m pip install torch==2.12.0 \
        --index-url https://download.pytorch.org/whl/cu124

COPY requirements-gpu.txt .
RUN python3.11 -m pip install -r requirements-gpu.txt

COPY . .
RUN python3.11 -m pip install -e .

# Default: report the environment snapshot (Phase 1 calibration records this).
CMD ["python3.11", "-c", "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"]
