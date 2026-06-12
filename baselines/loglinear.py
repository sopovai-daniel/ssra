"""Baseline (b): Log-Linear / GatedDeltaNet from a public implementation
(assignment §2, deliverable 2b; AP-5 protocol).

Provenance (verified 2026-06-12):
  package : flash-linear-attention 0.5.0 (PyPI)
  repo    : https://github.com/fla-org/flash-linear-attention
  tag     : v0.5.0 -> commit 3a9ce1c83a13994d824dbb3421e2989d330bb38b
  license : MIT (Apache-2.0-compatible), (c) 2023-2026 Songlin Yang,
            Yu Zhang, Zhiyuan Li
  note    : NVlabs/GatedDeltaNet (the paper authors' repo) is NVIDIA
            Source-Code-License-NC (non-commercial) and was therefore
            rejected per AP-5; the fla-org implementation is used instead.

Hardware status (AP-5, checked on this machine 2026-06-12): `import fla`
succeeds, but every kernel path (including mode="chunk" on CPU) imports
Triton at layer-import time, and Triton has no macOS/arm64 build. Local M1
verification is therefore limited to: pinned install + top-level import +
this wrapper. Execution (tiny forward + smoke run) is DEFERRED TO M2 GPU —
not silently skipped; see results/M1-report.md and tests/test_baselines.py.
"""

from __future__ import annotations

import torch.nn as nn

from ssra.config import ModelConfig

FLA_REPO = "https://github.com/fla-org/flash-linear-attention"
FLA_VERSION = "0.5.0"
FLA_COMMIT = "3a9ce1c83a13994d824dbb3421e2989d330bb38b"
FLA_LICENSE = "MIT"


def build_gated_deltanet_lm(cfg: ModelConfig) -> nn.Module:
    """GatedDeltaNet causal LM at the same d/h/L (lazy import: Triton-bound).

    Raises RuntimeError with the M2-deferral message on machines without
    Triton (e.g. this MacBook M1)."""
    try:
        from fla.models import GatedDeltaNetConfig, GatedDeltaNetForCausalLM
    except ModuleNotFoundError as e:  # pragma: no cover - GPU-only path
        raise RuntimeError(
            f"baseline (b) requires Triton ({e}); integration is pinned at "
            f"{FLA_REPO} v{FLA_VERSION} ({FLA_COMMIT}, {FLA_LICENSE}); "
            "execution deferred to M2 GPU per AP-5") from e
    fla_cfg = GatedDeltaNetConfig(
        hidden_size=cfg.d,
        num_heads=cfg.h,
        num_hidden_layers=cfg.n_layers,
        vocab_size=cfg.vocab,
        tie_word_embeddings=cfg.tied_embeddings,
    )
    return GatedDeltaNetForCausalLM(fla_cfg)
