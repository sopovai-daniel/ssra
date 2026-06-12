"""Shared attention parameters theta = (W_Q, W_K, W_V, W_O) of one layer.

One instance serves node attention, P1 pooling cross-attention and the
read-out of its layer (spec §2: maximal axis A). With readout_params=separate
a second instance (psi) is created for the read-out only (ablation e)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import rope_rotate


class SharedAttention(nn.Module):
    def __init__(self, d: int, h: int, rope_base: float, dropout_attn: float = 0.0):
        super().__init__()
        assert d % h == 0
        self.d, self.h, self.d_h = d, h, d // h
        self.rope_base = rope_base
        self.w_q = nn.Linear(d, d, bias=False)
        self.w_k = nn.Linear(d, d, bias=False)
        self.w_v = nn.Linear(d, d, bias=False)
        self.w_o = nn.Linear(d, d, bias=False)
        self.dropout_attn = dropout_attn

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[..., L, d] -> [..., h, L, d_h]"""
        return x.unflatten(-1, (self.h, self.d_h)).transpose(-2, -3)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[..., h, L, d_h] -> [..., L, d]"""
        return x.transpose(-2, -3).flatten(-2)

    def _drop(self, probs: torch.Tensor) -> torch.Tensor:
        if self.dropout_attn > 0.0 and self.training:
            probs = F.dropout(probs, p=self.dropout_attn)
        return probs

    def node_attn(self, x: torch.Tensor) -> torch.Tensor:
        """Bidirectional MHA over slots with slot-RoPE positions 1..n_in.

        x: [B*, n_in, d] (already LN_node-normalized); returns [B*, n_in, d].
        """
        n_in = x.shape[-2]
        q = self.split_heads(self.w_q(x))
        k = self.split_heads(self.w_k(x))
        v = self.split_heads(self.w_v(x))
        pos = torch.arange(1, n_in + 1, device=x.device)
        q = rope_rotate(q, pos, self.rope_base)
        k = rope_rotate(k, pos, self.rope_base)
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.d_h)
        out = self._drop(scores.softmax(dim=-1)) @ v
        return self.w_o(self.merge_heads(out))

    def cross_attn(self, q_in: torch.Tensor, kv_in: torch.Tensor,
                   return_probs: bool = False):
        """Position-free cross-attention (P1 pooling): no RoPE on either side.

        q_in: [B*, s, d], kv_in: [B*, n_in, d]; returns [B*, s, d].
        """
        q = self.split_heads(self.w_q(q_in))
        k = self.split_heads(self.w_k(kv_in))
        v = self.split_heads(self.w_v(kv_in))
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.d_h)
        probs = scores.softmax(dim=-1)
        out = self.w_o(self.merge_heads(self._drop(probs) @ v))
        if return_probs:
            return out, probs
        return out
