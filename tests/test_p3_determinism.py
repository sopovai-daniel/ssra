"""P3 determinism (spec §14.7, D5): inference is hard, deterministic and
noise-free. Two inference passes under different global seeds must produce
bitwise identical outputs — for the full batched forward and for the
incremental decoder, including the gumbel_topk training-grad config."""

import pytest
import torch

from conftest import make_model
from ssra import decode_logits


@pytest.mark.parametrize("grad", ["ste", "gumbel_topk"])
def test_p3_inference_bitwise_deterministic(grad):
    model = make_model(pool="p3").eval()
    model.cfg.p3.grad = grad
    x = torch.randint(0, model.cfg.vocab, (2, 300),
                      generator=torch.Generator().manual_seed(99))

    torch.manual_seed(0)
    with torch.no_grad():
        a, _ = model(x)
    torch.manual_seed(12345)
    with torch.no_grad():
        b, _ = model(x)
    assert torch.equal(a, b), "P3 full forward is not bitwise deterministic"

    torch.manual_seed(0)
    da = decode_logits(model, x)
    torch.manual_seed(54321)
    db = decode_logits(model, x)
    assert torch.equal(da, db), "P3 decoding is not bitwise deterministic"
