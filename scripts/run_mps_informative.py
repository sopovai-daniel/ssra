"""Informative MPS execution of tests 1-3 and 7 (assignment AP-2).

Tests are JUDGED on CPU fp32 (pytest suite); this script re-runs the same
checks on MPS and reports the observed numerics. MPS results do not gate.

Usage: .venv/bin/python scripts/run_mps_informative.py | tee logs/M1-mps-informative.log
"""

import math
import platform
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "tests")
from conftest import make_model  # noqa: E402
from test_shift import shift_positions  # noqa: E402

from ssra import decode_logits, lb_loss  # noqa: E402

DEV = "mps"
POOLS = [dict(pool="p1"), dict(pool="p2"), dict(pool="p3"),
         dict(pool="hybrid", k_sel=4)]
ATOL = 1e-4  # fp32 tolerance from the spec; informative only on MPS


def banner():
    print(f"platform : {platform.platform()}")
    print(f"python   : {platform.python_version()}  torch: {torch.__version__}")
    print(f"device   : {DEV} (available={torch.backends.mps.is_available()})")
    print(f"dtype    : {torch.get_default_dtype()}")
    print("judged on: CPU fp32 (AP-2); the numbers below are informative\n")


def run_shift(kw):
    model = make_model(**kw).to(DEV).eval()
    torch.manual_seed(123)
    x = torch.randint(0, model.cfg.vocab, (1, 300), device=DEV)
    worst = 0.0
    with torch.no_grad():
        base, _ = model(x)
        for t in shift_positions(300, model.cfg.w):
            xp = x.clone()
            xp[0, t - 1] = (xp[0, t - 1] + 7) % model.cfg.vocab
            pert, _ = model(xp)
            worst = max(worst, (pert[:, : t - 1] - base[:, : t - 1])
                        .abs().max().item())
    return worst


def run_completion(kw, n):
    model = make_model(**kw).to(DEV).eval()
    torch.manual_seed(42)
    x = torch.randint(0, model.cfg.vocab, (1, n), device=DEV)
    with torch.no_grad():
        full, _ = model(x)
    inc = decode_logits(model, x)
    return (full - inc).abs().max().item()


def run_gradient_flow(kw):
    model = make_model(**kw).to(DEV).train()
    torch.manual_seed(7)
    x = torch.randint(0, model.cfg.vocab, (2, 323), device=DEV)
    logits, aux = model(x, capture_summaries=True)
    loss = F.cross_entropy(logits[:, :-1].flatten(0, 1), x[:, 1:].flatten())
    lb = lb_loss(aux)
    if torch.is_tensor(lb):
        loss = loss + model.cfg.p3.lambda_lb * lb
    loss.backward()
    n_lvl = int(math.log2(323))
    zeros = []
    for name, p in model.named_parameters():
        g = p.grad
        if g is None:
            zeros.append(name + " (None)")
            continue
        if "level_emb" in name:
            g = g[: n_lvl + 1]
        if "latent_q" in name and model.cfg.pool == "hybrid":
            g = g[: model.cfg.m_max - model.cfg.k_sel]
        if g.abs().max() == 0:
            zeros.append(name)
    return zeros


def run_p3_determinism():
    model = make_model(pool="p3").to(DEV).eval()
    x = torch.randint(0, model.cfg.vocab, (2, 300),
                      generator=torch.Generator().manual_seed(99)).to(DEV)
    torch.manual_seed(0)
    with torch.no_grad():
        a, _ = model(x)
    torch.manual_seed(12345)
    with torch.no_grad():
        b, _ = model(x)
    return torch.equal(a, b)


def main():
    banner()
    for kw in POOLS:
        name = kw["pool"]
        s = run_shift(kw)
        c257 = run_completion(kw, 257)
        c1000 = run_completion(kw, 1000)
        zeros = run_gradient_flow(kw)
        flag = "OK " if max(s, c257, c1000) <= ATOL and not zeros else "DIFF"
        print(f"[{flag}] {name:7s} shift max-delta {s:.3e} | completion "
              f"max-delta N=257 {c257:.3e} N=1000 {c1000:.3e} | "
              f"zero-grad params: {zeros or 'none'}")
    det = run_p3_determinism()
    print(f"[{'OK ' if det else 'DIFF'}] p3 determinism (bitwise, MPS): {det}")


if __name__ == "__main__":
    main()
