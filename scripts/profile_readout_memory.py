"""Memory-behavior verification of the read-out (assignment D2).

Audits every tensor autograd SAVES during one read-out forward (the saved
set is what coexists across all L layers during training and caused the
calibration OOM), via torch.autograd.graph.saved_tensors_hooks. Reports:

  * total unique saved storage (parameters excluded, counted separately),
  * the top saved tensors by size with shapes,
  * detection of (a) the gathered [B,h,N,k_max,d_h] tensors and (b) the
    contiguous [B,h,N,d_h,w+1] window copies (claims §2.1 / §2.5).

--path gathered  = frozen reference (verbatim old readout, tests/test_readout_equiv.py)
--path grouped   = current SSRALayer.readout (R1 + R4)

CPU fp32 by default; --device mps adds torch.mps allocator numbers
(informative only, AP-2). z and levels are detached so the audit contains
read-out-internal tensors only.

Usage: python scripts/profile_readout_memory.py --path gathered --preset s1 --batch 4
"""

from __future__ import annotations

import argparse
import resource
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ssra import ModelConfig  # noqa: E402
from ssra.fenwick import build_readout_index  # noqa: E402
from ssra.model import SSRALayer  # noqa: E402
from test_readout_equiv import reference_readout  # noqa: E402

PRESETS = {  # calibration shapes (experiments/M2-cal-*.yaml), single layer
    "s1": dict(d=384, h=6),
    "s2": dict(d=640, h=10),
    "small": dict(d=64, h=4),
}


class SavedTensorAudit:
    """Records unique storages saved by autograd during forward."""

    def __init__(self, skip_storages: set[int]):
        self.skip = skip_storages
        self.records: dict[int, tuple[int, tuple]] = {}  # ptr -> (nbytes, shape)

    def _pack(self, t):
        if isinstance(t, torch.Tensor) and t.device.type != "meta":
            st = t.untyped_storage()
            ptr = st.data_ptr()
            if ptr not in self.skip:
                prev = self.records.get(ptr, (0, ()))
                if st.nbytes() >= prev[0]:
                    self.records[ptr] = (st.nbytes(), tuple(t.shape))
        return t

    def __enter__(self):
        self.ctx = torch.autograd.graph.saved_tensors_hooks(self._pack, lambda x: x)
        self.ctx.__enter__()
        return self

    def __exit__(self, *exc):
        self.ctx.__exit__(*exc)

    @property
    def total_bytes(self) -> int:
        return sum(nb for nb, _ in self.records.values())

    def top(self, k: int = 12):
        return sorted(self.records.values(), reverse=True)[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", choices=["gathered", "grouped"], required=True)
    ap.add_argument("--preset", choices=PRESETS, default="s1")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = ModelConfig(n_layers=1, vocab=256, n_max=max(2048, args.n),
                      **PRESETS[args.preset])
    torch.manual_seed(0)
    layer = SSRALayer(cfg).to(args.device).train()
    b, n, w = args.batch, args.n, cfg.w

    torch.manual_seed(1)
    z = torch.randn(b, n, cfg.d, device=args.device)
    with torch.no_grad():
        levels = layer.up_pass(z, {})
    z = z.detach().requires_grad_(True)
    levels = [lv.detach().requires_grad_(True) for lv in levels]
    cache = build_readout_index(n, w, cfg.s_l, device=args.device)
    k_max = cache[0].shape[1]

    param_storages = {p.untyped_storage().data_ptr() for p in layer.parameters()}
    fn = (lambda: reference_readout(layer, z, levels, cache)) \
        if args.path == "gathered" else (lambda: layer.readout(z, levels, cache))

    if args.device == "mps":
        torch.mps.empty_cache()
        base_alloc = torch.mps.current_allocated_memory()

    with SavedTensorAudit(param_storages) as audit:
        out = fn()
    retained_alloc = None
    if args.device == "mps":
        torch.mps.synchronize()
        retained_alloc = torch.mps.current_allocated_memory() - base_alloc
    out.sum().backward()

    mib = 1024 ** 2
    print(f"path={args.path} preset={args.preset} B={b} N={n} h={cfg.h} "
          f"d_h={cfg.d // cfg.h} w={w} m={cfg.m} k_max={k_max} "
          f"device={args.device} dtype=fp32")
    print(f"autograd-saved (unique storages, params excluded): "
          f"{audit.total_bytes / mib:,.1f} MiB in {len(audit.records)} storages")
    print("top saved tensors:")
    gather_hits, window_hits = [], []
    for nb, shape in audit.top():
        print(f"  {nb / mib:9,.1f} MiB  shape={list(shape)}")
    # einsum saves its operands bmm-flattened (3-D), so detect the per-token
    # materializations by exact byte size, not by 5-D shape
    d_h, elt = cfg.d // cfg.h, 4
    gather_nb = b * cfg.h * n * k_max * d_h * elt
    window_nb = b * cfg.h * n * (w + 1) * d_h * elt
    for nb, shape in audit.records.values():
        if nb == gather_nb:
            gather_hits.append((nb, shape))
        elif nb == window_nb:
            window_hits.append((nb, shape))
    print(f"[check] gathered B*h*N*k_max*d_h tensors saved "
          f"({gather_nb / mib:,.1f} MiB each): {len(gather_hits)} "
          f"({sum(nb for nb, _ in gather_hits) / mib:,.1f} MiB)")
    print(f"[check] materialized B*h*N*(w+1)*d_h window copies saved "
          f"({window_nb / mib:,.1f} MiB each): {len(window_hits)} "
          f"({sum(nb for nb, _ in window_hits) / mib:,.1f} MiB)")
    if retained_alloc is not None:
        print(f"MPS retained after forward (informative): "
              f"{retained_alloc / mib:,.1f} MiB")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mib = rss / mib if sys.platform == "darwin" else rss / 1024
    print(f"process peak RSS: {rss_mib:,.1f} MiB")


if __name__ == "__main__":
    main()
