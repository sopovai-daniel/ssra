"""Pool_phi operators (spec §5) behind one interface:

    Pool_phi : R^{n_in x d} x (s_l) -> R^{s_l x d}

The caller short-circuits to identity when s_l == n_in (lossless levels), so
operators here only see lossy calls. One operator per run (config enum)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import SharedAttention
from .config import ModelConfig


def make_pool(cfg: ModelConfig, attn: SharedAttention) -> "PoolBase":
    return {"p1": P1LatentQuery, "p2": P2StridedMerge,
            "p3": P3TopKSelect, "hybrid": HybridPool}[cfg.pool](cfg, attn)


class PoolBase(nn.Module):
    def __init__(self, cfg: ModelConfig, attn: SharedAttention):
        super().__init__()
        self.cfg = cfg
        self.attn = attn  # the layer's shared theta (not owned)

    def forward(self, h: torch.Tensor, s_out: int, aux: dict) -> torch.Tensor:
        raise NotImplementedError


class P1LatentQuery(PoolBase):
    """P1 (default): S_u = Q[:s_l] + Attn_theta(q=Q[:s_l], kv=LN_pool(H_u)).

    Reuses the layer's theta projections (MD-3); phi = {Q_phi, LN_pool}.
    Contingency pool_own_proj=true gives P1 its own projections (off by
    default; spec §5.1)."""

    def __init__(self, cfg: ModelConfig, attn: SharedAttention):
        super().__init__(cfg, attn)
        self.latent_q = nn.Parameter(torch.randn(cfg.m_max, cfg.d) * 0.02)
        self.ln_pool = nn.LayerNorm(cfg.d)
        if cfg.pool_own_proj:
            self.own_attn = SharedAttention(cfg.d, cfg.h, cfg.rope_base,
                                            cfg.dropout_attn)

    def forward(self, h, s_out, aux):
        attn = self.own_attn if self.cfg.pool_own_proj else self.attn
        q = self.latent_q[:s_out].expand(h.shape[0], s_out, self.cfg.d)
        out, probs = attn.cross_attn(q, self.ln_pool(h), return_probs=True)
        if aux.get("collect_diagnostics"):
            _p1_diagnostics(probs, aux)
        return q + out


def _p1_diagnostics(probs: torch.Tensor, aux: dict) -> None:
    """P-C collapse diagnostics (spec §14.5): entropy of Q_phi attention maps
    and per-query participation (max attention weight any key gives it)."""
    with torch.no_grad():
        p = probs.clamp_min(1e-12)
        entropy = -(p * p.log()).sum(-1).mean().item()  # nats, mean over q/heads
        # participation of query i: how often it is some key's argmax reader
        winner = probs.argmax(dim=-2)  # [B*, h, n_keys] -> winning query index
        part = torch.bincount(winner.flatten(), minlength=probs.shape[-2]).float()
        part = (part / part.sum()).cpu()
        aux.setdefault("p1_entropy", []).append(entropy)
        aux.setdefault("p1_participation", []).append(part)


class P2StridedMerge(PoolBase):
    """P2 (control): S[i] = W_merge [H[2i-1]; H[2i]] + b. Fixed-m, k=2 only."""

    def __init__(self, cfg: ModelConfig, attn: SharedAttention):
        super().__init__(cfg, attn)
        self.merge = nn.Linear(2 * cfg.d, cfg.d, bias=True)

    def forward(self, h, s_out, aux):
        n_in = h.shape[-2]
        assert n_in % 2 == 0 and s_out == n_in // 2, \
            "P2 is structurally pairwise halving (spec §5.2)"
        return self.merge(h.unflatten(-2, (s_out, 2)).flatten(-2))


class P3TopKSelect(PoolBase):
    """P3 (challenger): top-(s_l - 1) selection + context residual (spec §5.3).

    Forward is hard and deterministic (ties -> lower slot index, MD-8);
    training gradients via STE (default) or Gumbel-top-k. Output: selected
    slots in original slot order, context slot last."""

    def __init__(self, cfg: ModelConfig, attn: SharedAttention,
                 n_select: int | None = None, context_slot: bool = True):
        super().__init__(cfg, attn)
        self.scorer = nn.Linear(cfg.d, 1, bias=True)  # g_phi = (w_s, b_s)
        self.n_select = n_select  # None -> s_out - 1 (plain P3)
        self.context_slot = context_slot
        self.tau = cfg.p3.tau_start  # annealed by the training loop

    def forward(self, h, s_out, aux):
        n_in = h.shape[-2]
        k = self.n_select if self.n_select is not None else s_out - 1
        assert 0 < k < n_in
        sigma = self.scorer(h).squeeze(-1)  # [B*, n_in]

        if self.training and self.cfg.p3.grad == "gumbel_topk":
            # training-only stochasticity (spec §5.3); inference stays hard
            gumbel = -torch.log(-torch.log(
                torch.rand_like(sigma).clamp_min(1e-9)).clamp_min(1e-9))
            sigma_sel = sigma + gumbel
        else:
            sigma_sel = sigma

        # hard top-k of scores; stable sort => ties broken by lower slot index
        order = torch.argsort(sigma_sel, dim=-1, descending=True, stable=True)
        sel = order[..., :k].sort(dim=-1).values  # original slot order
        sel_mask = torch.zeros_like(sigma, dtype=torch.bool).scatter_(-1, sel, True)

        if self.training:
            p_hard = torch.zeros(*sigma.shape[:-1], k, sigma.shape[-1],
                                 dtype=h.dtype, device=h.device).scatter_(
                -1, sel.unsqueeze(-1), 1.0)
            p_soft = self._soft_relaxation(sigma_sel, sel)
            p = p_hard + p_soft - p_soft.detach()  # STE: hard fwd, soft bwd
            selected = p @ h  # exact verbatim copies in forward
            _lb_loss(p_soft, aux)
        else:
            selected = torch.gather(
                h, -2, sel.unsqueeze(-1).expand(*sel.shape, h.shape[-1]))

        if not self.context_slot:
            return selected

        # context residual over non-selected slots, softmax(sigma_rest / tau)
        rest_logits = (sigma / self.tau).masked_fill(sel_mask, float("-inf"))
        ctx = (rest_logits.softmax(-1).unsqueeze(-2) @ h)  # [B*, 1, d]
        return torch.cat([selected, ctx], dim=-2)

    def _soft_relaxation(self, sigma: torch.Tensor, sel: torch.Tensor):
        """Softmax relaxation of top-k for the STE backward: row r is a
        masked softmax over sigma/tau with the other selected slots removed,
        aligned with the slot-ordered selection in `sel`."""
        k = sel.shape[-1]
        rows = []
        all_sel = torch.zeros_like(sigma, dtype=torch.bool).scatter_(-1, sel, True)
        for r in range(k):
            keep_self = torch.zeros_like(all_sel).scatter_(
                -1, sel[..., r : r + 1], True)
            mask = all_sel & ~keep_self  # mask out the other selected slots
            rows.append((sigma / self.tau).masked_fill(mask, float("-inf"))
                        .softmax(-1))
        return torch.stack(rows, dim=-2)  # [B*, k, n_in]


def _lb_loss(p_soft: torch.Tensor, aux: dict) -> None:
    """Load-balance term KL(p_bar || uniform) over slot positions (spec §5.3);
    p_bar from the soft relaxation so the scorer receives gradient. The
    training loop multiplies the collected mean by lambda_lb."""
    n_in = p_soft.shape[-1]
    p_bar = p_soft.reshape(-1, n_in).mean(dim=0).clamp_min(1e-12)
    kl = (p_bar * (p_bar * n_in).log()).sum()
    aux.setdefault("lb_terms", []).append(kl)


class HybridPool(PoolBase):
    """Hybrid(k_sel): k_sel slots by P3 selection (verbatim, no context slot)
    + (s_l - k_sel) slots by P1 latent queries (spec §5.4)."""

    def __init__(self, cfg: ModelConfig, attn: SharedAttention):
        super().__init__(cfg, attn)
        self.p3 = P3TopKSelect(cfg, attn, n_select=cfg.k_sel, context_slot=False)
        self.p1 = P1LatentQuery(cfg, attn)

    def forward(self, h, s_out, aux):
        selected = self.p3(h, s_out, aux)
        latent = self.p1(h, s_out - self.cfg.k_sel, aux)
        return torch.cat([selected, latent], dim=-2)
