"""Integration checks for baselines (a) flat and (c) MEGABYTE-style:
forward+backward runs, shapes are right, and the causal alignment matches the
shared harness convention (logits[t] predicts token t+1, so perturbing token
t must not change logits at positions < t-... strictly below t-1)."""

import pytest
import torch
import torch.nn.functional as F

from baselines.flat import FlatLM
from baselines.megabyte import MegabyteLM
from ssra import ModelConfig


def build(arch: str):
    torch.manual_seed(0)
    cfg = ModelConfig(d=64, h=4, n_layers=2, vocab=50, n_max=512)
    return FlatLM(cfg) if arch == "flat" else MegabyteLM(cfg, patch=8)


@pytest.mark.parametrize("arch", ["flat", "megabyte"])
def test_forward_backward(arch):
    model = build(arch).train()
    x = torch.randint(0, 50, (2, 256))
    logits, _ = model(x)
    assert logits.shape == (2, 256, 50)
    loss = F.cross_entropy(logits[:, :-1].flatten(0, 1), x[:, 1:].flatten())
    loss.backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None for g in grads)
    assert all(g.abs().max() > 0 for g in grads)


def test_loglinear_integration():
    """Baseline (b), AP-5: pinned fla package imports; the GatedDeltaNet
    LM builds where Triton exists, otherwise the wrapper must raise the
    explicit M2-deferral error (never a silent skip)."""
    import fla
    from baselines.loglinear import FLA_VERSION, build_gated_deltanet_lm
    assert fla.__version__ == FLA_VERSION
    cfg = ModelConfig(d=64, h=2, n_layers=2, vocab=50, n_max=512)
    try:
        model = build_gated_deltanet_lm(cfg)
        assert sum(p.numel() for p in model.parameters()) > 0
    except RuntimeError as e:
        assert "deferred to M2 GPU" in str(e)
        pytest.skip(f"Triton unavailable on this machine: {e}")


@pytest.mark.parametrize("arch", ["flat", "megabyte"])
def test_causal_alignment(arch):
    model = build(arch).eval()
    x = torch.randint(0, 50, (1, 256), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        base, _ = model(x)
        for t in [2, 8, 9, 64, 128, 129, 255]:  # incl. patch boundaries
            xp = x.clone()
            xp[0, t - 1] = (xp[0, t - 1] + 7) % 50
            pert, _ = model(xp)
            delta = (pert[:, : t - 1] - base[:, : t - 1]).abs().max().item()
            assert delta <= 1e-4, f"{arch} leaks at t={t}: {delta:.3e}"
