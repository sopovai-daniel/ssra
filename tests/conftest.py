import pytest
import torch

from ssra import ModelConfig, SSRALM

# AP-2: tests 1, 2, 3, 7 are judged on CPU fp32. MPS is exercised separately
# (scripts/run_mps_informative.py) and reported informatively.
DEVICE = "cpu"


def make_model(seed: int = 0, **kw) -> SSRALM:
    defaults = dict(d=64, h=4, n_layers=2, vocab=50, n_max=2048)
    defaults.update(kw)
    torch.manual_seed(seed)
    return SSRALM(ModelConfig(**defaults)).to(DEVICE)


def pool_variants():
    return [
        pytest.param(dict(pool="p1"), id="p1"),
        pytest.param(dict(pool="p2"), id="p2"),
        pytest.param(dict(pool="p3"), id="p3"),
        pytest.param(dict(pool="hybrid", k_sel=4), id="hybrid"),
    ]
