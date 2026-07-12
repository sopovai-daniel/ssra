"""Frozen-reference A/B test for the read-out restructure (AP-20, assignment
docs/cc/M2-readout-optimization.md D3).

`reference_readout` below is a VERBATIM copy of `SSRALayer.readout` as of
commit 576b927 (the gathered realization measured in M2 Phase 1 calibration),
with the sole mechanical change `self` -> `layer` (module method -> free
function). It must never be edited; it is the frozen semantic reference.

The new grouped realization (R1 + R4) must be logit-equivalent on randomized
inputs across: ragged N in {257, 1000, 1024}, level_emb on/off,
readout_params shared/separate, summary_pos none + virtual, several (m, w).
fp32, atol 1e-5.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from ssra import ModelConfig, SSRALM
from ssra.fenwick import build_readout_index
from ssra.model import SSRALayer
from ssra.rope import rope_rotate

ATOL = 1e-5


# ---- frozen reference (verbatim from src/ssra/model.py @ 576b927) -----------

def reference_readout(layer, z: torch.Tensor, levels: list[torch.Tensor],
                      cache) -> torch.Tensor:
    cfg, attn = layer.cfg, layer.readout_attn
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
        tagged = (flat + layer.level_emb[level]
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


# ---- helpers -----------------------------------------------------------------

def make_layer(seed: int = 0, **kw) -> SSRALayer:
    cfg = ModelConfig(**{**dict(d=64, h=4, n_layers=1, vocab=50, n_max=2048),
                         **kw})
    cfg.validate()
    torch.manual_seed(seed)
    return SSRALayer(cfg).eval()


def run_both(layer: SSRALayer, n: int, seed: int = 42):
    torch.manual_seed(seed)
    z = torch.randn(2, n, layer.cfg.d)
    with torch.no_grad():
        levels = layer.up_pass(z, {})
        cache = build_readout_index(n, layer.cfg.w, layer.cfg.s_l)
        old = reference_readout(layer, z, levels, cache)
        new = layer.readout(z, levels, cache)
    return old, new


GRID = [  # (n, m, w) — ragged + power-of-two N, several (m, w)
    (257, 16, 64),
    (1000, 16, 64),
    (1024, 16, 64),
    (257, 4, 8),
    (1000, 8, 32),
    (1024, 4, 8),
]


# ---- tests --------------------------------------------------------------------

@pytest.mark.parametrize("n,m,w", GRID)
@pytest.mark.parametrize("level_emb", ["on", "off"])
@pytest.mark.parametrize("readout_params", ["shared", "separate"])
def test_grouped_equals_reference(n, m, w, level_emb, readout_params):
    layer = make_layer(m=m, w=w, level_emb=level_emb,
                       readout_params=readout_params)
    if level_emb == "on":  # nonzero tags — zeros-init would mask tag bugs
        torch.nn.init.normal_(layer.level_emb, std=0.02)
    old, new = run_both(layer, n)
    delta = (old - new).abs().max().item()
    assert delta <= ATOL, f"grouped != reference at N={n}, m={m}, w={w}: {delta:.3e}"


@pytest.mark.parametrize("n", [257, 1024])
def test_virtual_fallback_equals_reference(n):
    """summary_pos=virtual takes the gathered fallback branch (assignment §3
    R1 option (i)); it must reproduce the reference bit-for-bit."""
    layer = make_layer(summary_pos="virtual", summary_pos_override=True)
    torch.nn.init.normal_(layer.level_emb, std=0.02)
    old, new = run_both(layer, n)
    assert torch.equal(old, new), "virtual fallback diverged from reference"


def test_full_model_logits_equal():
    """End-to-end: model logits with the grouped read-out vs the frozen
    reference monkeypatched in (fp32, atol 1e-5)."""
    torch.manual_seed(0)
    cfg = ModelConfig(d=64, h=4, n_layers=2, vocab=50, n_max=2048)
    model = SSRALM(cfg).eval()
    for layer in model.layers:
        torch.nn.init.normal_(layer.level_emb, std=0.02)
    torch.manual_seed(7)
    x = torch.randint(0, cfg.vocab, (2, 1000))
    with torch.no_grad():
        new_logits, _ = model(x)
    orig = SSRALayer.readout
    SSRALayer.readout = reference_readout
    try:
        with torch.no_grad():
            old_logits, _ = model(x)
    finally:
        SSRALayer.readout = orig
    delta = (old_logits - new_logits).abs().max().item()
    assert delta <= ATOL, f"model logits diverge: {delta:.3e}"


def test_grouped_gradients_flow():
    """Backward through the grouped path reaches z, the summary levels and
    the read-out projections (guards against detached scatter/pad plumbing)."""
    layer = make_layer(m=4, w=8)
    torch.manual_seed(3)
    z = torch.randn(2, 257, layer.cfg.d, requires_grad=True)
    levels = layer.up_pass(z, {})
    cache = build_readout_index(257, layer.cfg.w, layer.cfg.s_l)
    layer.readout(z, levels, cache).sum().backward()
    assert z.grad is not None and z.grad.abs().sum() > 0
    assert layer.readout_attn.w_k.weight.grad.abs().sum() > 0
    assert layer.readout_attn.w_v.weight.grad.abs().sum() > 0
