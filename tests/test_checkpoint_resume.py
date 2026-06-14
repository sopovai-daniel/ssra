"""AP-11 checkpoint/resume: a run interrupted and resumed from a checkpoint must
produce the SAME loss curve as an uninterrupted run -- i.e. no discontinuity at
the resume point (max acceptable loss on Spot preemption = one checkpoint
interval). This mirrors the harness's continuity guarantee: it rests entirely on
ssra.checkpoint restoring model + optimizer + BOTH RNG streams (data sampler and
torch global RNG).

Judged on CPU fp32 (AP-2 convention), so the trajectory is reproducible to
tight tolerance.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ssra import ModelConfig, SSRALM
from ssra.checkpoint import latest_path, load_checkpoint, save_checkpoint

SEED = 1337
VOCAB = 37
SEQ_LEN = 16
BATCH = 4
STEPS = 12
INTERRUPT_AT = 6  # checkpoint here, then resume in a fresh process-like state


def _make_data() -> torch.Tensor:
    g = torch.Generator().manual_seed(99)
    return torch.randint(0, VOCAB, (2000,), generator=g)


def _make_model() -> SSRALM:
    torch.manual_seed(SEED)
    cfg = ModelConfig(d=32, h=2, n_layers=2, vocab=VOCAB, n_max=256, w=8, m=4)
    return SSRALM(cfg)


def _batch(data, gen):
    starts = torch.randint(0, len(data) - SEQ_LEN - 1, (BATCH,), generator=gen)
    return torch.stack([data[s : s + SEQ_LEN] for s in starts])


def _step(model, opt, data, gen) -> float:
    x = _batch(data, gen)
    logits, aux = model(x)
    loss = F.cross_entropy(logits[:, :-1].flatten(0, 1), x[:, 1:].flatten())
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return loss.item()


def _fresh():
    torch.manual_seed(SEED)
    model = _make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95))
    gen = torch.Generator().manual_seed(SEED)
    return model, opt, gen


def test_resume_yields_continuous_loss_curve(tmp_path):
    data = _make_data()

    # ---- uninterrupted reference run ---------------------------------------
    torch.manual_seed(SEED)
    model, opt, gen = _fresh()
    ref = [_step(model, opt, data, gen) for _ in range(STEPS)]

    # ---- interrupted run: train, checkpoint, resume from scratch -----------
    torch.manual_seed(SEED)
    model, opt, gen = _fresh()
    got = [_step(model, opt, data, gen) for _ in range(INTERRUPT_AT)]
    ckpt = latest_path(tmp_path)
    save_checkpoint(ckpt, step=INTERRUPT_AT, model=model, optimizer=opt,
                    data_gen=gen, config_raw={}, run_name="ut")

    # brand-new objects (simulates a Spot preemption + fresh worker)
    model2, opt2, gen2 = _fresh()
    resumed_step = load_checkpoint(ckpt, model=model2, optimizer=opt2,
                                   data_gen=gen2)
    assert resumed_step == INTERRUPT_AT
    got += [_step(model2, opt2, data, gen2) for _ in range(STEPS - INTERRUPT_AT)]

    # ---- the curves must coincide everywhere, incl. across the seam --------
    assert len(got) == len(ref) == STEPS
    for i, (a, b) in enumerate(zip(ref, got)):
        assert abs(a - b) < 1e-5, f"step {i}: ref={a} resumed={b} (discontinuity)"

    # the post-resume segment must match bit-for-tight, not merely trend down
    seam = abs(ref[INTERRUPT_AT] - got[INTERRUPT_AT])
    assert seam < 1e-5, f"discontinuity at resume seam: {seam}"


def test_checkpoint_is_atomic_and_reloadable(tmp_path):
    """latest.pt is written via tmp + os.replace; reload returns the step."""
    model, opt, gen = _fresh()
    data = _make_data()
    for _ in range(3):
        _step(model, opt, data, gen)
    ckpt = latest_path(tmp_path)
    save_checkpoint(ckpt, step=3, model=model, optimizer=opt, data_gen=gen,
                    config_raw={"model": {"d": 32}}, run_name="ut")
    assert ckpt.exists()
    assert not ckpt.with_suffix(ckpt.suffix + ".tmp").exists()
    m2, o2, g2 = _fresh()
    assert load_checkpoint(ckpt, model=m2, optimizer=o2, data_gen=g2) == 3
