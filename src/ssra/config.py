"""Configuration surface per docs/spec.md §13, including validation rules.

Fixed-by-spec defaults: k=2, m=16 fixed schedule, w=64, pool=p1, level_emb=on,
summary_pos=none, readout_params=shared, rope_base=10000.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields

import yaml


class ConfigError(ValueError):
    """Raised when a config violates a spec §13 validation rule."""


@dataclass
class P3Config:
    grad: str = "ste"  # ste | gumbel_topk
    tau_start: float = 2.0
    tau_end: float = 0.5
    tau_anneal_steps: int = 1000
    lambda_lb: float = 0.01


@dataclass
class ModelConfig:
    d: int = 64
    h: int = 4
    n_layers: int = 2
    vocab: int = 256
    n_max: int = 1024          # sizes the e_l table (L_max = ceil(log2 n_max))
    m: int = 16
    m_schedule: str = "fixed"  # fixed | linear (linear: P1 only, spec §13)
    m0: int = 0                # linear schedule M(l) = m0 + g*l
    g: int = 0
    w: int = 64
    k: int = 2                 # {2, 4}; k=4 is an M1 stub (AP-7)
    pool: str = "p1"           # p1 | p2 | p3 | hybrid
    k_sel: int = 0             # hybrid only
    pool_own_proj: bool = False
    p1_diversity_loss: float = 0.0  # [K] unverified formulation -> >0 raises
    summary_pos: str = "none"  # none | virtual
    summary_pos_override: bool = False  # explicit flag required for virtual
    level_emb: str = "on"      # on | off
    readout_params: str = "shared"  # shared | separate
    rope_base: float = 10000.0
    dropout_attn: float = 0.0
    dropout_resid: float = 0.0
    tied_embeddings: bool = True
    p3: P3Config = field(default_factory=P3Config)

    # ---- derived quantities -------------------------------------------------

    @property
    def l_max(self) -> int:
        return max(1, math.ceil(math.log2(self.n_max)))

    def capacity(self, level: int) -> int:
        """M(l): summary capacity at level l."""
        if self.m_schedule == "fixed":
            return self.m
        return self.m0 + self.g * level

    def s_l(self, level: int) -> int:
        """Summaries emitted at level l: s_l = min(2^l, M(l)); s_0 = 1."""
        if level == 0:
            return 1
        return min(2 ** level, self.capacity(level))

    @property
    def m_max(self) -> int:
        """Max slots any level emits (sizes the P1 latent-query table)."""
        return max(self.s_l(l) for l in range(self.l_max + 1))

    def lossy_levels(self) -> list[int]:
        """Levels (up to L_max) where Pool is not the identity."""
        out = []
        for level in range(1, self.l_max + 1):
            n_in = 2 * self.s_l(level - 1) if self.k == 2 else 4 * self.s_l(level - 1)
            if self.s_l(level) < n_in:
                out.append(level)
        return out

    # ---- validation (spec §13) ----------------------------------------------

    def validate(self) -> None:
        if self.d <= 0 or self.h <= 0 or self.d % self.h != 0:
            raise ConfigError(f"d={self.d} must be a positive multiple of h={self.h}")
        if self.n_layers <= 0 or self.vocab <= 0 or self.n_max <= 0:
            raise ConfigError("n_layers, vocab, n_max must be positive")
        if self.m < 1 or self.w < 1:
            raise ConfigError("m >= 1 and w >= 1 required")
        if self.k not in (2, 4):
            raise ConfigError(f"k={self.k}: tree arity must be 2 or 4")
        if self.pool not in ("p1", "p2", "p3", "hybrid"):
            raise ConfigError(f"unknown pool '{self.pool}'")
        if self.m_schedule not in ("fixed", "linear"):
            raise ConfigError(f"unknown m_schedule '{self.m_schedule}'")
        if self.summary_pos not in ("none", "virtual"):
            raise ConfigError(f"unknown summary_pos '{self.summary_pos}'")
        if self.level_emb not in ("on", "off"):
            raise ConfigError(f"level_emb must be 'on' or 'off'")
        if self.readout_params not in ("shared", "separate"):
            raise ConfigError("readout_params must be 'shared' or 'separate'")
        if self.p3.grad not in ("ste", "gumbel_topk"):
            raise ConfigError(f"unknown p3.grad '{self.p3.grad}'")

        # spec §13 explicit rejection rules
        if self.pool == "p2" and self.m_schedule == "linear":
            raise ConfigError("P2 is valid only with the fixed-m schedule (spec §5.2)")
        if self.pool == "p2" and self.k == 4:
            raise ConfigError("P2 is valid only with k=2 (spec §5.2)")
        if self.summary_pos == "virtual" and not self.summary_pos_override:
            raise ConfigError(
                "summary_pos=virtual requires the explicit override flag "
                "summary_pos_override=true (spec §6: do not enable silently)")

        if self.m_schedule == "linear":
            if self.m0 < 1 or self.g < 0:
                raise ConfigError("linear schedule requires m0 >= 1, g >= 0")
            # spec §13 YAML comment: linear schedule is P1 only
            if self.pool != "p1":
                raise ConfigError("m_schedule=linear is P1 only (spec §13)")
            for level in range(1, self.l_max + 1):
                if self.s_l(level) > self.k * self.s_l(level - 1):
                    raise ConfigError(
                        f"linear schedule emits s_{level}={self.s_l(level)} > "
                        f"n_in={self.k * self.s_l(level - 1)} (Pool cannot expand)")

        if self.pool == "hybrid":
            if self.k_sel < 1:
                raise ConfigError("hybrid requires k_sel >= 1")
            for level in self.lossy_levels():
                if self.k_sel >= self.s_l(level):
                    raise ConfigError(
                        f"hybrid k_sel={self.k_sel} >= s_l={self.s_l(level)} "
                        f"at lossy level {level} (spec §13)")

        # [K] formulation unverified -> any value > 0 must raise (assignment §3)
        if self.p1_diversity_loss > 0.0:
            raise NotImplementedError(
                "p1_diversity_loss formulation is unverified [K] (spec §5.1); "
                "values > 0 are not implemented in M1")

        for name, val in (("dropout_attn", self.dropout_attn),
                          ("dropout_resid", self.dropout_resid)):
            if not 0.0 <= val < 1.0:
                raise ConfigError(f"{name}={val} must be in [0, 1)")


_MODEL_KEYS = {f.name for f in fields(ModelConfig)} - {"p3"}
_P3_KEYS = {f.name for f in fields(P3Config)}


def config_from_dict(raw: dict) -> ModelConfig:
    """Build + validate a ModelConfig from a {model: {...}, p3: {...}} dict."""
    raw = dict(raw or {})
    model_raw = dict(raw.pop("model", {}) or {})
    p3_raw = dict(raw.pop("p3", {}) or {})
    if raw:
        raise ConfigError(f"unknown top-level config sections: {sorted(raw)}")

    # spec §13 nests hybrid/linear parameters; accept both nested and flat
    pool = model_raw.get("pool")
    if isinstance(pool, dict):
        model_raw["pool"] = "hybrid"
        model_raw["k_sel"] = pool.get("k_sel", 0)
    sched = model_raw.get("m_schedule")
    if isinstance(sched, dict):
        model_raw["m_schedule"] = "linear"
        model_raw["m0"] = sched.get("m0", 0)
        model_raw["g"] = sched.get("g", 0)

    unknown = set(model_raw) - _MODEL_KEYS
    if unknown:
        raise ConfigError(f"unknown model config keys: {sorted(unknown)}")
    unknown = set(p3_raw) - _P3_KEYS
    if unknown:
        raise ConfigError(f"unknown p3 config keys: {sorted(unknown)}")

    cfg = ModelConfig(**model_raw, p3=P3Config(**p3_raw))
    cfg.validate()
    return cfg


def load_config(path: str) -> ModelConfig:
    with open(path) as f:
        return config_from_dict(yaml.safe_load(f))
