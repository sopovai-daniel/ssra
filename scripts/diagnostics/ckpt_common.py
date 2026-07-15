"""Shared helpers for the M2 spike diagnostics (docs/cc/M2-spike-diagnostics.md).

Read-only w.r.t. model and training code (AP-20): this package only LOADS
AP-11 checkpoint blobs and IMPORTS the certified `ssra` package to recover
the optimizer-order parameter names. It never modifies model/harness code,
never runs a training step, never touches GCS.

AP-11 blob layout (ssra/checkpoint.py):
  {step, model: state_dict, optimizer: AdamW state_dict, data_gen_state,
   torch_rng_state, config_raw, run_name, extra: {arch, vocab}}

Weight sharing note (ssra/model.py): `readout_attn` and the P1 pool reuse the
node-block `attn` module, so `model.state_dict()` contains alias entries
(readout_attn.*, pool.attn.*) pointing at the same tensors. Everything here
works off `model.named_parameters()` order (duplicates removed, exactly the
order `AdamW(model.parameters())` saw), so each physical tensor is counted
once under its canonical name (`layers.N.attn.*`).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)


def load_blob(path: str | Path) -> dict:
    """Load an AP-11 checkpoint on CPU.

    weights_only=False: the blob carries optimizer + RNG payloads that the
    torch 2.x safe loader rejects; acceptable for this trusted self-produced
    artifact (assignment T1, flag recorded in the report)."""
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def build_model(blob: dict):
    """Rebuild the run's model architecture from the config embedded in the
    checkpoint (read-only import of the certified packages)."""
    from baselines.flat import FlatLM
    from ssra import SSRALM, config_from_dict

    builders = {"ssra": SSRALM, "flat": FlatLM}
    raw = dict(blob["config_raw"])
    raw.setdefault("model", {})["vocab"] = int(blob["extra"]["vocab"])
    return builders[blob["extra"]["arch"]](config_from_dict(raw))


def optimizer_param_names(blob: dict) -> list[str]:
    """Names of optimizer params, index-aligned with blob['optimizer']['state'].

    AdamW was constructed as AdamW(model.parameters()); named_parameters()
    yields the same (deduplicated) order. Every shape is cross-checked
    against the stored exp_avg — any mismatch aborts (no silent mislabeling)."""
    model = build_model(blob)
    names = [n for n, _ in model.named_parameters()]
    shapes = {n: p.shape for n, p in model.named_parameters()}
    pg = blob["optimizer"]["param_groups"]
    assert len(pg) == 1 and pg[0]["params"] == list(range(len(names))), \
        "unexpected optimizer param_groups layout"
    for i, n in enumerate(names):
        st = blob["optimizer"]["state"].get(i)
        if st is not None and st["exp_avg"].shape != shapes[n]:
            raise AssertionError(f"shape mismatch at idx {i} ({n}): "
                                 f"{st['exp_avg'].shape} vs {shapes[n]}")
    return names


def canonical_params(blob: dict) -> dict[str, torch.Tensor]:
    """{canonical name -> fp32 tensor} for each physical parameter (aliases
    from the shared-theta registration are excluded via named_parameters)."""
    sd = blob["model"]
    return {n: sd[n].float() for n in optimizer_param_names(blob)}


_LAYER_RE = re.compile(r"^layers\.(\d+)\.")


def layer_of(name: str) -> int | None:
    m = _LAYER_RE.match(name)
    return int(m.group(1)) if m else None


def block_of(name: str) -> str:
    """Map a canonical parameter name to its logical block (assignment T2):
    shared attention theta per matrix, Pool_phi params, token embedding
    (tied unembedding), level embedding, LN gains/biases per site, FFN."""
    n = _LAYER_RE.sub("", name)
    if n == "emb.weight":
        return "token_emb(tied_unemb)"
    if n == "level_emb":
        return "level_emb"
    m = re.match(r"^attn\.(w_[qkvo])\.weight$", n)
    if m:
        return f"attn_theta.{m.group(1)}"  # shared with readout + P1 pool
    m = re.match(r"^readout_attn\.(w_[qkvo])\.weight$", n)
    if m:
        return f"readout_attn.{m.group(1)}"  # only present if untied readout
    if n == "pool.latent_q":
        return "pool_phi.latent_q"
    m = re.match(r"^pool\.attn\.(w_[qkvo])\.weight$", n)
    if m:
        return f"pool_phi.attn.{m.group(1)}"  # only present if untied pool
    m = re.match(r"^(ln1|ln2|ln_node|ln_f|pool\.ln_pool)\.(weight|bias)$", n)
    if m:
        site = m.group(1).replace("pool.ln_pool", "ln_pool")
        kind = "gain" if m.group(2) == "weight" else "bias"
        return f"ln.{site}.{kind}"
    m = re.match(r"^ffn\.(fc[12])\.(weight|bias)$", n)
    if m:
        return f"ffn.{m.group(1)}.{m.group(2)}"
    return f"other.{n}"


def fmt_table(rows: list[dict], cols: list[str]) -> str:
    """Render rows as a GitHub-markdown table (values pre-formatted)."""
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |"
            for r in rows]
    return "\n".join([head, sep] + body)


def sci(x: float) -> str:
    return f"{x:.4e}"
