"""CPU smoke test for scripts/g2lite_eval.py (M2 G2-lite §6.1: tiny model,
N in {64, 128}, proving the M0/M1/M2 code paths run and the anchor logic
works). Functionality only — no quality conclusions (spec §16).

Runs everything at bf16 autocast + fp32 accumulation (AP-16), the same
precision constants as the pod session.
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from baselines.flat import FlatLM  # noqa: E402
from g2lite_eval import (check_anchor, eval_cell, greedy_generate,  # noqa: E402
                         run_m0, run_m1, run_m2)
from needle_gen import generate_suite  # noqa: E402
from ssra import ModelConfig, SSRALM  # noqa: E402
from ssra.data import EOT_TOKEN  # noqa: E402

DEVICE, PRECISION = "cpu", "bf16"
VOCAB = 16384  # real tokenizer vocab so the needle suite round-trips
GRID = [64, 128]


def tiny_cfg():
    return ModelConfig(d=64, h=4, n_layers=2, vocab=VOCAB, n_max=128,
                       m=16, w=64)


@pytest.fixture(scope="module", params=["ssra", "flat"])
def model(request):
    torch.manual_seed(7)
    m = (SSRALM if request.param == "ssra" else FlatLM)(tiny_cfg())
    m.eval()
    m.arch = request.param
    return m


@pytest.fixture(scope="module")
def tok():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(
        str(ROOT / "artifacts/tokenizer/fineweb-edu-bpe-16384.json"))


def test_anchor_logic():
    assert check_anchor(3.19333, 3.19333, 1e-3)["pass"]
    assert check_anchor(3.19400, 3.19333, 1e-3)["pass"]  # within tolerance
    bad = check_anchor(3.19560, 3.19333, 1e-3)
    assert not bad["pass"] and bad["delta_nats"] == pytest.approx(0.00227)


def test_m0_code_path_and_anchor(model):
    """final_eval (imported training code path) runs; the anchor verdict is
    pass at the self-measured value and fail when the expectation shifts."""
    torch.manual_seed(11)
    ids = torch.randint(0, VOCAB, (64 * 64 + 65,))
    anchor = {"seq_len": 64, "batch_size": 4, "tol_nats": 1e-3,
              "expected": {model.arch: 0.0}}
    first = run_m0(model, ids, anchor, model.arch, DEVICE, PRECISION)
    assert first["final_eval"]["eval_windows"] == 65
    assert not model.training  # eval-only session restored after final_eval

    anchor["expected"][model.arch] = first["final_eval"]["eval_loss"]
    good = run_m0(model, ids, anchor, model.arch, DEVICE, PRECISION)
    assert good["anchor"]["pass"]
    anchor["expected"][model.arch] += 0.01
    assert not run_m0(model, ids, anchor, model.arch,
                      DEVICE, PRECISION)["anchor"]["pass"]


def test_m1_code_path(model):
    """Cells at N in {64, 128} over one shared 1,024-token region: window
    counts, scored-token counts, buckets, and the r(N) ratio field."""
    torch.manual_seed(13)
    region = torch.randint(0, VOCAB, (1024,))
    out = run_m1(model, region, GRID, {64: 8, 128: 4}, model.arch,
                 DEVICE, PRECISION)
    by_n = {c["n"]: c for c in out["cells"]}
    assert by_n[64]["windows"] == 16 and by_n[128]["windows"] == 8
    for n, c in by_n.items():
        assert c["tokens_scored"] == c["windows"] * (n - 1)
        assert c["ppl"] > 0 and c["r_vs_min_n"] > 0
        assert list(c["bucket_mean_nll"]) == ["1-256"]  # N <= 256
    assert by_n[64]["r_vs_min_n"] == 1.0  # ratio vs the smallest grid N


def test_m1_batching_invariance(model):
    """Batch size affects wall-clock only, never values (§3): same cell,
    different batch sizes, identical fp64-accumulated result."""
    torch.manual_seed(17)
    region = torch.randint(0, VOCAB, (512,))
    a = eval_cell(model, region.view(-1, 64), 8, DEVICE, PRECISION)
    b = eval_cell(model, region.view(-1, 64), 3, DEVICE, PRECISION)
    assert a["mean_nll"] == b["mean_nll"]


def test_m2_code_path(model, tok):
    """Needle cells at N in {64, 128} (smoke suite, no preamble — the fixed
    segments alone exceed the N=64 budget; the committed suite always
    includes the preamble): greedy full-forward generation runs, stops at
    <= max_new, and the metric fields are produced deterministically."""
    trials = list(generate_suite(tok, seed=20260717, grid=GRID,
                                 depths=[0.1, 0.5, 0.9], trials=2,
                                 headroom=16, preamble=False))
    eot_id = tok.token_to_id(EOT_TOKEN)
    out = run_m2(model, trials, GRID, {64: 4, 128: 4}, tok, eot_id,
                 max_new=12, arch=model.arch, device=DEVICE,
                 precision=PRECISION)
    assert len(out["cells"]) == 6  # 2 N x 3 depths
    for cell in out["cells"]:
        assert cell["trials"] == 2
        assert 0.0 <= cell["accuracy"] <= 1.0
        for r in cell["results"]:
            assert set(r) == {"trial", "key", "generated", "extracted",
                              "correct"}

    again = run_m2(model, trials, GRID, {64: 4, 128: 4}, tok, eot_id,
                   max_new=12, arch=model.arch, device=DEVICE,
                   precision=PRECISION)
    assert [r["generated"] for c in out["cells"] for r in c["results"]] == \
           [r["generated"] for c in again["cells"] for r in c["results"]]


def test_m2_greedy_respects_eot_and_cap(model):
    """Generation never exceeds max_new and truncates rows at eot."""
    torch.manual_seed(19)
    prompts = torch.randint(1, VOCAB, (3, 48))
    gen = greedy_generate(model, prompts, eot_id=0, max_new=5,
                          device=DEVICE, precision=PRECISION)
    for g in gen:
        assert 1 <= len(g) <= 5
        if 0 in g:
            assert g.index(0) == len(g) - 1  # nothing recorded past eot
