"""SSRA (Scale-Shared Recursive Attention) — PoC per docs/spec.md v1.0."""

from .config import ConfigError, ModelConfig, P3Config, config_from_dict, load_config
from .decode import IncrementalDecoder, decode_logits
from .fenwick import fenwick_blocks
from .model import SSRALM, lb_loss

__all__ = [
    "ConfigError", "ModelConfig", "P3Config", "config_from_dict", "load_config",
    "IncrementalDecoder", "decode_logits", "fenwick_blocks", "SSRALM", "lb_loss",
]
