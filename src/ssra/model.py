"""SSRA v2 model per docs/spec.md: level-wise batched up-pass (§4), Fenwick
read-out variant A (§8), pre-norm sublayers (§3)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import SharedAttention
from .config import ModelConfig
from .fenwick import build_readout_index
from .pool import make_pool
from .rope import rope_rotate


class FFN(nn.Module):
    """Standard sublayer MLP d -> 4d -> d, GELU (spec §3). NOT inside nodes."""

    def __init__(self, d: int):
        super().__init__()
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class SSRALayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.ln_node = nn.LayerNorm(d)
        self.attn = SharedAttention(d, cfg.h, cfg.rope_base, cfg.dropout_attn)
        # ablation (e): separate psi for the read-out (spec §8)
        self.readout_attn = (self.attn if cfg.readout_params == "shared"
                             else SharedAttention(d, cfg.h, cfg.rope_base,
                                                  cfg.dropout_attn))
        # e_l, l in 0..L_max; init zeros (D2); e_0 is read-out key tag only
        self.level_emb = nn.Parameter(torch.zeros(cfg.l_max + 1, d))
        self.pool = make_pool(cfg, self.attn)
        self.ffn = FFN(d)
        self.drop = nn.Dropout(cfg.dropout_resid)

    # ---- up-pass (spec §4) ---------------------------------------------------

    def node_step(self, x: torch.Tensor, level: int, aux: dict) -> torch.Tensor:
        """One batched node computation: [B*, n_in, d] -> [B*, s_l, d]."""
        if self.cfg.level_emb == "on":
            x = x + self.level_emb[level]
        h = x + self.attn.node_attn(self.ln_node(x))
        s_out = self.cfg.s_l(level)
        if s_out == h.shape[-2]:
            return h  # lossless level: Pool = identity
        return self.pool(h, s_out, aux)

    def up_pass(self, z: torch.Tensor, aux: dict) -> list[torch.Tensor]:
        """Level-wise batched tree build (D6). Returns per-level summaries,
        levels[l]: [B, floor(N/2^l), s_l, d]; node u=(l,j) at index j-1."""
        b, n, d = z.shape
        levels = [z.unsqueeze(2)]  # leaves: S_u = z_t, s_0 = 1
        level = 1
        while levels[-1].shape[1] >= 2:
            child = levels[-1]
            n_par = child.shape[1] // 2
            # ragged sequences: node materialized iff span within [1, N] (§4)
            child = child[:, : 2 * n_par]
            x = torch.cat([child[:, 0::2], child[:, 1::2]], dim=2)
            n_in = x.shape[2]
            h = self.node_step(x.reshape(b * n_par, n_in, d), level, aux)
            levels.append(h.reshape(b, n_par, -1, d))
            level += 1
        return levels

    # ---- read-out (spec §8, variant A) ---------------------------------------

    def readout(self, z: torch.Tensor, levels: list[torch.Tensor],
                cache) -> torch.Tensor:
        cfg, attn = self.cfg, self.readout_attn
        b, n, d = z.shape
        w = cfg.w
        idx, sum_mask, vpos = cache
        pos = torch.arange(1, n + 1, device=z.device)

        # query + window keys: RoPE at absolute token positions (§6)
        q = rope_rotate(attn.split_heads(attn.w_q(z)), pos, cfg.rope_base)
        k_tok = rope_rotate(attn.split_heads(attn.w_k(z)), pos, cfg.rope_base)
        v_tok = attn.split_heads(attn.w_v(z))

        # window [t-w, t] inclusive (MD-6) via left-pad + unfold
        k_win = F.pad(k_tok, (0, 0, w, 0)).unfold(2, w + 1, 1)  # [B,h,N,dh,w+1]
        v_win = F.pad(v_tok, (0, 0, w, 0)).unfold(2, w + 1, 1)
        scores_win = torch.einsum("bhnd,bhndk->bhnk", q, k_win) / math.sqrt(attn.d_h)
        win_valid = (pos.unsqueeze(1) - w
                     + torch.arange(w + 1, device=z.device)) >= 1  # [N, w+1]
        scores_win = scores_win.masked_fill(~win_valid, float("-inf"))

        # summary keys: NoPE + e_l tag on keys only (MD-5); values untagged
        key_rows, val_rows = [], []
        for level, s in enumerate(levels):
            flat = s.flatten(1, 2)  # [B, n_nodes*s_l, d]
            tagged = (flat + self.level_emb[level]
                      if cfg.level_emb == "on" else flat)
            key_rows.append(tagged)
            val_rows.append(flat)
        k_sum = attn.split_heads(attn.w_k(torch.cat(key_rows, 1)))  # [B,h,R,dh]
        v_sum = attn.split_heads(attn.w_v(torch.cat(val_rows, 1)))
        k_g = k_sum[:, :, idx]  # [B,h,N,k_max,dh]
        v_g = v_sum[:, :, idx]
        if cfg.summary_pos == "virtual":
            # contingency: rotate summary keys at virtual position t-w-1 (§6)
            k_g = rope_rotate(k_g, vpos.unsqueeze(1), cfg.rope_base)
        scores_sum = torch.einsum("bhnd,bhnkd->bhnk", q, k_g) / math.sqrt(attn.d_h)
        scores_sum = scores_sum.masked_fill(~sum_mask, float("-inf"))

        # ONE softmax over heterogeneous keys (§8)
        probs = torch.cat([scores_win, scores_sum], dim=-1).softmax(dim=-1)
        probs = attn._drop(probs)
        p_win, p_sum = probs.split([w + 1, scores_sum.shape[-1]], dim=-1)
        out = (torch.einsum("bhnk,bhndk->bhnd", p_win, v_win)
               + torch.einsum("bhnk,bhnkd->bhnd", p_sum, v_g))
        return attn.w_o(attn.merge_heads(out))

    def forward(self, x: torch.Tensor, aux: dict, cache) -> torch.Tensor:
        z = self.ln1(x)
        levels = self.up_pass(z, aux)
        if aux.get("capture_summaries"):
            aux.setdefault("summaries", []).append(levels)
        x = x + self.drop(self.readout(z, levels, cache))
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


class SSRALM(nn.Module):
    """Token embedding (no global positional embedding, §3) + n_layers SSRA
    blocks + final LN + (tied or untied) unembedding."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cfg.validate()
        if cfg.k == 4:
            raise NotImplementedError(
                "k=4 tree path is stubbed in M1 (assignment AP-7); "
                "must exist before the M3 arity ablation (g)")
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab, cfg.d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.layers = nn.ModuleList(SSRALayer(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d)
        if not cfg.tied_embeddings:
            self.head = nn.Linear(cfg.d, cfg.vocab, bias=False)
        self._readout_cache: dict = {}

    def readout_cache(self, n: int, device) -> tuple:
        key = (n, str(device))
        if key not in self._readout_cache:
            self._readout_cache[key] = build_readout_index(
                n, self.cfg.w, self.cfg.s_l, device=device)
        return self._readout_cache[key]

    def lm_head(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.tied_embeddings:
            return F.linear(x, self.emb.weight)
        return self.head(x)

    def forward(self, tokens: torch.Tensor, collect_diagnostics: bool = False,
                capture_summaries: bool = False):
        """tokens: [B, N] int64 -> (logits [B, N, vocab], aux dict).

        aux carries P3 load-balance terms ('lb_terms') and, when requested,
        P-C diagnostics and per-layer summary tensors."""
        b, n = tokens.shape
        if n > self.cfg.n_max:
            raise ValueError(f"N={n} exceeds n_max={self.cfg.n_max}")
        aux: dict = {"collect_diagnostics": collect_diagnostics,
                     "capture_summaries": capture_summaries}
        cache = self.readout_cache(n, tokens.device)
        x = self.emb(tokens)
        for layer in self.layers:
            x = layer(x, aux, cache)
        return self.lm_head(self.ln_f(x)), aux


def lb_loss(aux: dict) -> torch.Tensor | float:
    """Mean of collected P3 load-balance terms (training loop multiplies by
    lambda_lb)."""
    terms = aux.get("lb_terms")
    if not terms:
        return 0.0
    return torch.stack(terms).mean()
