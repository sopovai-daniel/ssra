"""Shift test (spec §14.1): perturbing token t must not change logits at any
position < t. Position set: window edges {2, w-1, w, w+1, w+2}, every Fenwick
merge boundary {2^j, 2^j +- 1 : 2^j <= N}, and N-1. atol = 1e-4 fp32."""

import pytest
import torch

from conftest import make_model, pool_variants

N = 300
ATOL = 1e-4


def shift_positions(n: int, w: int) -> list[int]:
    pos = {2, w - 1, w, w + 1, w + 2, n - 1}
    p = 2
    while p <= n:
        pos.update({p - 1, p, p + 1})
        p *= 2
    return sorted(t for t in pos if 2 <= t <= n)


@pytest.mark.parametrize("kw", pool_variants())
def test_shift(kw):
    model = make_model(**kw).eval()
    torch.manual_seed(123)
    x = torch.randint(0, model.cfg.vocab, (1, N))
    with torch.no_grad():
        base, _ = model(x)
        for t in shift_positions(N, model.cfg.w):
            xp = x.clone()
            xp[0, t - 1] = (xp[0, t - 1] + 7) % model.cfg.vocab
            pert, _ = model(xp)
            delta = (pert[:, : t - 1] - base[:, : t - 1]).abs().max().item()
            assert delta <= ATOL, f"causality violated at t={t}: {delta:.3e}"
            # sanity: the perturbation itself must be visible at position t
            assert (pert[:, t - 1] - base[:, t - 1]).abs().max() > 0
