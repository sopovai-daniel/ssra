"""Gradient-flow check (spec §14.3): after one backward pass every parameter
group (theta, phi, e_l for materialized levels, embeddings, FFN, LNs) has a
nonzero gradient, and the deep-summary path d(loss)/d(S_root) is nonzero.

N = 323 so the deepest materialized node (level 8, span [1, 256]) is consumed
by the read-out at t = 322 = 256 + w + 1 and its logits enter the loss."""

import math

import pytest
import torch
import torch.nn.functional as F

from conftest import make_model, pool_variants
from ssra import lb_loss

N = 323


@pytest.mark.parametrize("kw", pool_variants())
def test_gradient_flow(kw):
    model = make_model(**kw).train()
    cfg = model.cfg
    torch.manual_seed(7)
    x = torch.randint(0, cfg.vocab, (2, N))
    logits, aux = model(x, capture_summaries=True)

    # retain grad on the deepest summary of every layer (S_root reachability)
    deepest = [levels[-1] for levels in aux["summaries"]]
    for s in deepest:
        s.retain_grad()

    loss = F.cross_entropy(logits[:, :-1].flatten(0, 1), x[:, 1:].flatten())
    lb = lb_loss(aux)
    if torch.is_tensor(lb):
        loss = loss + cfg.p3.lambda_lb * lb
    loss.backward()

    n_lvl_materialized = int(math.log2(N))  # floor: levels 0..8 for N=323
    hybrid_latents = cfg.m_max - cfg.k_sel  # hybrid uses only the first rows

    for name, p in model.named_parameters():
        g = p.grad
        if "level_emb" in name:
            assert g is not None
            used = g[: n_lvl_materialized + 1]
            assert used.abs().max() > 0, f"e_l grad zero on materialized levels"
            assert g[n_lvl_materialized + 1:].abs().max() == 0, \
                "e_l grad nonzero on non-materialized level"
            continue
        if "latent_q" in name and cfg.pool == "hybrid":
            assert g is not None and g[:hybrid_latents].abs().max() > 0, name
            continue
        assert g is not None, f"no grad for {name}"
        assert g.abs().max() > 0, f"zero grad for {name}"

    for i, s in enumerate(deepest):
        assert s.grad is not None and s.grad.abs().max() > 0, \
            f"d(loss)/d(S_root) is zero in layer {i}"
