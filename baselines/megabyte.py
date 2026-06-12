"""Baseline (c): minimal faithful MEGABYTE-style 2-level decoder
(assignment §2, deliverable 2c).

Faithful to Yu et al. 2023 (arXiv:2305.07185) in the elements that define the
architecture: fixed-size patches; a global causal Transformer over patch
embeddings formed by concatenating the per-byte embeddings of a patch; a
small local causal Transformer per patch that consumes the global output for
its patch plus the within-patch bytes shifted right by one. Minimal: no
cross-patch local attention, no fancy init; pre-norm blocks and RoPE reused
from the flat baseline for comparability.

Alignment convention: forward() returns logits where logits[:, t] predicts
token t+1 (same as SSRALM/FlatLM), so the shared training harness applies
unchanged; the final position is zero-filled and is dropped by the loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from baselines.flat import FlatBlock
from ssra.config import ModelConfig


class MegabyteLM(nn.Module):
    def __init__(self, cfg: ModelConfig, patch: int = 8,
                 d_local: int | None = None, n_local_layers: int = 2):
        super().__init__()
        self.cfg = cfg
        self.patch = patch
        d_g = cfg.d
        d_l = d_local or cfg.d // 2
        assert d_g % patch == 0, "global dim must split across patch bytes"
        h_l = max(1, cfg.h // 2)
        assert d_l % h_l == 0

        self.emb_global = nn.Embedding(cfg.vocab, d_g // patch)
        self.emb_local = nn.Embedding(cfg.vocab, d_l)
        for e in (self.emb_global, self.emb_local):
            nn.init.normal_(e.weight, std=0.02)
        # learned start-of-sequence embeddings for the two shifts
        self.global_start = nn.Parameter(torch.zeros(d_g))
        self.local_start = nn.Parameter(torch.zeros(d_l))

        g_cfg = ModelConfig(d=d_g, h=cfg.h, n_layers=cfg.n_layers,
                            vocab=cfg.vocab, n_max=cfg.n_max,
                            rope_base=cfg.rope_base,
                            dropout_attn=cfg.dropout_attn,
                            dropout_resid=cfg.dropout_resid)
        l_cfg = ModelConfig(d=d_l, h=h_l, n_layers=n_local_layers,
                            vocab=cfg.vocab, n_max=patch,
                            rope_base=cfg.rope_base,
                            dropout_attn=cfg.dropout_attn,
                            dropout_resid=cfg.dropout_resid)
        self.global_blocks = nn.ModuleList(
            FlatBlock(g_cfg) for _ in range(cfg.n_layers))
        self.local_blocks = nn.ModuleList(
            FlatBlock(l_cfg) for _ in range(n_local_layers))
        self.g2l = nn.Linear(d_g, d_l)
        self.ln_g = nn.LayerNorm(d_g)
        self.ln_l = nn.LayerNorm(d_l)
        self.head = nn.Linear(d_l, cfg.vocab, bias=False)

    def forward(self, tokens: torch.Tensor):
        b, n = tokens.shape
        p = self.patch
        assert n % p == 0, f"sequence length {n} must be a multiple of patch {p}"
        k = n // p

        # global stream over patches, shifted right by one patch
        g_in = self.emb_global(tokens).reshape(b, k, -1)  # concat patch bytes
        g_in = torch.cat([self.global_start.expand(b, 1, -1),
                          g_in[:, :-1]], dim=1)
        g = g_in
        for blk in self.global_blocks:
            g = blk(g)
        g_ctx = self.g2l(self.ln_g(g))  # [B, K, d_l]

        # local stream per patch: bytes shifted right by one within the patch
        l_in = self.emb_local(tokens).reshape(b, k, p, -1)
        l_in = torch.cat([self.local_start.expand(b, k, 1, -1),
                          l_in[:, :, :-1]], dim=2)
        x = (l_in + g_ctx.unsqueeze(2)).reshape(b * k, p, -1)
        for blk in self.local_blocks:
            x = blk(x)
        pred = self.head(self.ln_l(x)).reshape(b, n, -1)  # pred[t] = dist of t

        # re-align: logits[t] predicts t+1; last position zero-filled (unused)
        logits = torch.cat([pred[:, 1:], torch.zeros_like(pred[:, :1])], dim=1)
        return logits, {}
