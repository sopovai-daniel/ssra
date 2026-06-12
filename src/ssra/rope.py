"""Rotary position embedding (RoPE), used for slot positions inside nodes
(spec §6: positions 1..n_in) and absolute token positions in the read-out.
Summary keys are NoPE by default (spec §6, MD-5)."""

from __future__ import annotations

import torch


def rope_rotate(x: torch.Tensor, pos: torch.Tensor, base: float) -> torch.Tensor:
    """Rotate the last dim of x by RoPE angles pos * base^(-2i/d_h).

    x:   [..., d_h] with d_h even.
    pos: broadcastable to x.shape[:-1] (float or int tensor of positions).
    """
    d_h = x.shape[-1]
    assert d_h % 2 == 0, "head dim must be even for RoPE"
    inv_freq = base ** (
        -torch.arange(0, d_h, 2, dtype=x.dtype, device=x.device) / d_h
    )  # [d_h/2]
    angles = pos.to(x.dtype).unsqueeze(-1) * inv_freq  # [..., d_h/2]
    cos, sin = angles.cos(), angles.sin()
    x_even, x_odd = x[..., 0::2], x[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    return torch.stack((out_even, out_odd), dim=-1).flatten(-2)
