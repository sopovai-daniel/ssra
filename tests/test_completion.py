"""Completion test (spec §14.2): incremental decoding logits must equal the
full batched forward at EVERY position, N in {257, 1000} (non-powers of two
crossing several 2^k boundaries). Certifies the read-out implementation and
the §9 retention rule. atol = 1e-4 fp32."""

import pytest
import torch

from conftest import make_model, pool_variants
from ssra import decode_logits

ATOL = 1e-4


@pytest.mark.parametrize("kw", pool_variants())
@pytest.mark.parametrize("n", [257, 1000])
def test_completion(kw, n):
    model = make_model(**kw).eval()
    torch.manual_seed(42)
    x = torch.randint(0, model.cfg.vocab, (1, n))
    with torch.no_grad():
        full, _ = model(x)
    inc = decode_logits(model, x)
    delta = (full - inc).abs().max().item()
    assert delta <= ATOL, f"incremental != full forward at N={n}: {delta:.3e}"
