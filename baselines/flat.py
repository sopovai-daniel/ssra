"""Baseline (a): flat pre-norm causal Transformer with the same d, h, L as
the SSRA model under test (assignment §2, deliverable 2a).

Positional scheme: RoPE at absolute token positions with the same rope_base
as SSRA — the same positional family, no extra parameters, which keeps the
parameter-match note of spec §12 exact. Attention uses the fused SDPA causal
kernel (full quadratic attention; this is the Theta(N^2) reference for G1a).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ssra.config import ModelConfig
from ssra.model import FFN
from ssra.rope import rope_rotate


class FlatAttention(nn.Module):
    def __init__(self, d: int, h: int, rope_base: float, dropout_attn: float):
        super().__init__()
        self.h, self.d_h = h, d // h
        self.rope_base = rope_base
        self.w_q = nn.Linear(d, d, bias=False)
        self.w_k = nn.Linear(d, d, bias=False)
        self.w_v = nn.Linear(d, d, bias=False)
        self.w_o = nn.Linear(d, d, bias=False)
        self.dropout_attn = dropout_attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        split = lambda t: t.unflatten(-1, (self.h, self.d_h)).transpose(1, 2)
        pos = torch.arange(1, n + 1, device=x.device)
        q = rope_rotate(split(self.w_q(x)), pos, self.rope_base)
        k = rope_rotate(split(self.w_k(x)), pos, self.rope_base)
        v = split(self.w_v(x))
        drop = self.dropout_attn if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                             dropout_p=drop)
        return self.w_o(out.transpose(1, 2).flatten(-2))


class FlatBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d)
        self.ln2 = nn.LayerNorm(cfg.d)
        self.attn = FlatAttention(cfg.d, cfg.h, cfg.rope_base, cfg.dropout_attn)
        self.ffn = FFN(cfg.d)
        self.drop = nn.Dropout(cfg.dropout_resid)

    def forward(self, x):
        x = x + self.drop(self.attn(self.ln1(x)))
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


class FlatLM(nn.Module):
    """Token embedding + L pre-norm blocks + final LN + (tied) unembedding;
    mirrors SSRALM with the attention sublayer swapped for full causal MHA."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab, cfg.d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.layers = nn.ModuleList(FlatBlock(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d)
        if not cfg.tied_embeddings:
            self.head = nn.Linear(cfg.d, cfg.vocab, bias=False)

    def forward(self, tokens: torch.Tensor):
        x = self.emb(tokens)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        if self.cfg.tied_embeddings:
            return F.linear(x, self.emb.weight), {}
        return self.head(x), {}
