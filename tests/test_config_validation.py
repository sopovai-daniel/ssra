"""Config-surface validation per spec §13 + assignment §3."""

import pytest
import torch

from ssra import ConfigError, ModelConfig, SSRALM, config_from_dict


def cfg(**kw) -> ModelConfig:
    base = dict(d=64, h=4, n_layers=1, vocab=50, n_max=512)
    base.update(kw)
    c = ModelConfig(**base)
    c.validate()
    return c


def test_defaults_match_spec():
    c = cfg()
    assert (c.m, c.w, c.k, c.pool) == (16, 64, 2, "p1")
    assert c.level_emb == "on" and c.summary_pos == "none"
    assert c.readout_params == "shared" and c.rope_base == 10000


def test_schedule():
    c = cfg()
    assert [c.s_l(l) for l in range(7)] == [1, 2, 4, 8, 16, 16, 16]
    assert c.lossy_levels()[0] == 5  # first lossy compression: 32 -> 16
    lin = cfg(m_schedule="linear", m0=8, g=2)
    assert lin.s_l(3) == 8 and lin.s_l(4) == 16 and lin.s_l(5) == 18


def test_reject_p2_linear():
    with pytest.raises(ConfigError, match="fixed-m"):
        cfg(pool="p2", m_schedule="linear", m0=8, g=2)


def test_reject_p2_k4():
    with pytest.raises(ConfigError, match="k=2"):
        cfg(pool="p2", k=4)


def test_reject_linear_non_p1():
    with pytest.raises(ConfigError, match="P1 only"):
        cfg(pool="p3", m_schedule="linear", m0=8, g=2)


def test_reject_hybrid_k_sel_too_large():
    with pytest.raises(ConfigError, match="k_sel"):
        cfg(pool="hybrid", k_sel=16)  # s_l = 16 at every lossy level
    cfg(pool="hybrid", k_sel=15)  # largest legal value


def test_reject_virtual_summary_pos_without_override():
    with pytest.raises(ConfigError, match="override"):
        cfg(summary_pos="virtual")
    cfg(summary_pos="virtual", summary_pos_override=True)


def test_p1_diversity_loss_not_implemented():
    with pytest.raises(NotImplementedError, match="unverified"):
        cfg(p1_diversity_loss=0.1)


def test_k4_stubbed():
    c = cfg(k=4)  # config itself is legal (M3 ablation g)
    with pytest.raises(NotImplementedError, match="AP-7"):
        SSRALM(c)


def test_unknown_keys_rejected():
    with pytest.raises(ConfigError, match="unknown model config keys"):
        config_from_dict({"model": {"d": 64, "h": 4, "bogus": 1}})
    with pytest.raises(ConfigError, match="unknown top-level"):
        config_from_dict({"model": {}, "training": {}})


def test_nested_yaml_forms():
    c = config_from_dict({"model": {"d": 64, "h": 4, "vocab": 50, "n_max": 512,
                                    "pool": {"k_sel": 4}}})
    assert c.pool == "hybrid" and c.k_sel == 4
    c = config_from_dict({"model": {"d": 64, "h": 4, "vocab": 50, "n_max": 512,
                                    "m_schedule": {"m0": 8, "g": 2}}})
    assert c.m_schedule == "linear" and (c.m0, c.g) == (8, 2)


def test_yaml_file_roundtrip(tmp_path):
    from ssra import load_config
    p = tmp_path / "run.yaml"
    p.write_text("model:\n  d: 64\n  h: 4\n  vocab: 50\n  n_max: 512\n"
                 "  pool: p3\np3:\n  grad: gumbel_topk\n  lambda_lb: 0.05\n")
    c = load_config(str(p))
    assert c.pool == "p3" and c.p3.grad == "gumbel_topk"
    assert c.p3.lambda_lb == 0.05
