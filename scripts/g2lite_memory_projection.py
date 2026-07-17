"""Peak-VRAM projection for the M2 G2-lite inference session (assignment §5).

Extends the D5 analytic memory model (scripts/readout_memory_model.py,
validated against the saved-tensor audit for training) to the NO-GRAD
inference path: no autograd residency, no optimizer/grad state — the peak is
fp32 params + the largest concurrent live-tensor set of a single phase
(one layer's read-out softmax, one layer's FFN, or the lm_head/loss phase),
since each layer's transients are freed before the next layer runs and only
the residual stream persists.

Assumptions (stated per AP-20/AP-22; every projection is multiplied by the
AP-22 x1.20 error bar before the gate):
  * bf16 autocast (AP-16): matmul outputs 2 B/elt; softmax runs in fp32
    under autocast -> fp32 input cast + fp32 output (4 B/elt each) + bf16
    prob-slice casts for the AV einsums (upper-bounded as the full width).
  * flat SDPA: flash/mem-efficient backend assumed (same default dispatch as
    training; the training peak 10.846 GiB @ b16/N=1024 incl. backward is
    consistent with no N^2 materialization). The math-fallback N^2 term is
    reported as a separate risk column and is checked empirically by the
    pod-start micro-benchmark BEFORE any 16k/32k cell (§5).
  * M1 loss: full bf16 logits [B, N, V] + chunked fp32 CE (chunk 2048).
    M2 generation: full bf16 logits, no CE.
  * Launch gate: projected x 1.20 <= 95 % of card VRAM (AP-22 pro-rata).

Usage:
  .venv/bin/python scripts/g2lite_memory_projection.py [--markdown]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from readout_memory_model import grouped_levels, k_max, s_l  # noqa: E402

GIB = 1024 ** 3
D, H, L, VOCAB, W, M = 640, 10, 15, 16384, 64, 16
DH = D // H
GRID = [1024, 2048, 4096, 8192, 16384, 32768]
CE_CHUNK = 2048
CARDS = {"A6000/L40S 48GB": 48 * GIB, "A100 80GB": 80 * GIB}
GATE_FRAC = 0.95
ERROR_BAR = 1.20  # AP-22


def param_bytes() -> dict:
    """Exact fp32 parameter counts, from instantiated models (tied emb)."""
    import torch  # noqa: F401  (import cost only here)
    from baselines.flat import FlatLM
    from ssra import ModelConfig, SSRALM
    cfg = ModelConfig(d=D, h=H, n_layers=L, vocab=VOCAB, n_max=32768,
                      m=M, w=W, tied_embeddings=True)
    out = {}
    for name, cls in (("flat", FlatLM), ("ssra", SSRALM)):
        model = cls(cfg)
        out[name] = sum(p.numel() for p in model.parameters()) * 4
        del model
    return out


def idx_cache_bytes(n: int) -> int:
    """Fenwick gather cache built by SSRALM.readout_cache (int64 idx + bool
    mask + int64 vpos); sizes verified against verify-local.json."""
    km = k_max(n, W, M)
    return n * km * 8 + n * km + n * 8


def levels_bytes(b: int, n: int) -> int:
    """All per-level summary tensors held concurrently for the read-out."""
    rows = sum((n >> lvl) * s_l(lvl, M)
               for lvl in range(1, int(math.log2(n)) + 1))
    return b * rows * D * 2


def ssra_layer_terms(b: int, n: int) -> dict:
    c = W
    n_pad = math.ceil(n / c) * c
    lv = grouped_levels(n, W, M)
    s_tot = sum(s for *_, s in lv)
    width = 2 * c + s_tot
    bnd = b * n * D
    return {
        "residual x + z (bf16)": 2 * bnd * 2,
        "levels summaries (bf16, all levels)": levels_bytes(b, n),
        "q/k/v rotated (bf16)": 3 * bnd * 2,
        "rope pre-rotation transients (bf16)": 2 * bnd * 2,
        "window bands k,v (bf16)": 2 * (b * 2 * n_pad * D) * 2,
        "score parts + cat (bf16)": 2 * b * H * n * width * 2,
        "softmax fp32 in + out": 2 * b * H * n * width * 4,
        "prob-slice bf16 casts": b * H * n * width * 2,
        "read-out outputs (bf16)": 3 * bnd * 2,
        "fenwick idx cache (int64)": idx_cache_bytes(n),
    }


def flat_layer_terms(b: int, n: int) -> dict:
    bnd = b * n * D
    return {
        "residual x + ln1 (bf16)": 2 * bnd * 2,
        "q/k/v rotated (bf16)": 3 * bnd * 2,
        "rope pre-rotation transients (bf16)": 2 * bnd * 2,
        "sdpa out + softmax stats (flash)": bnd * 2 + b * H * n * 4,
        "w_o out (bf16)": bnd * 2,
    }


def flat_math_fallback_extra(b: int, n: int) -> int:
    """Risk column: math-backend SDPA materializes scores+probs [B,h,N,N]."""
    return 2 * b * H * n * n * 2


def ffn_phase(b: int, n: int, extra_persistent: int) -> int:
    """x + ln2 + fc1 + gelu + fc2 outputs, bf16."""
    return (1 + 1 + 4 + 4 + 1) * b * n * D * 2 + extra_persistent


def head_phase(b: int, n: int, with_ce: bool) -> int:
    """ln_f in/out + full bf16 logits (+ chunked fp32 CE for M1)."""
    total = 2 * b * n * D * 2 + b * n * VOCAB * 2
    if with_ce:
        total += b * min(CE_CHUNK, n - 1) * VOCAB * 8  # fp32 copy + logsoftmax
    return total


def project(arch: str, b: int, n: int, mode: str, params: dict) -> dict:
    layer = (ssra_layer_terms if arch == "ssra" else flat_layer_terms)(b, n)
    persistent = (levels_bytes(b, n) + idx_cache_bytes(n)
                  if arch == "ssra" else 0)
    phases = {
        "attn/readout": sum(layer.values()),
        "ffn": ffn_phase(b, n, persistent),
        "head": head_phase(b, n, with_ce=(mode == "m1")) + (
            2 * b * n * D * 2),  # residual still alive at ln_f
    }
    peak = params[arch] + b * n * 8 + max(phases.values())
    return {"arch": arch, "n": n, "b": b, "mode": mode,
            "peak_gib": peak / GIB, "gated_gib": peak * ERROR_BAR / GIB,
            "dominant_phase": max(phases, key=phases.get)}


def passes(gated_gib: float, card_bytes: int) -> bool:
    return gated_gib * GIB <= GATE_FRAC * card_bytes


def pick_batch(arch: str, n: int, mode: str, params: dict,
               card_bytes: int, candidates: list[int]) -> dict | None:
    for b in sorted(candidates, reverse=True):
        p = project(arch, b, n, mode, params)
        if passes(p["gated_gib"], card_bytes):
            return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()
    params = param_bytes()
    print(f"params fp32: flat {params['flat'] / GIB:.3f} GiB "
          f"({params['flat'] // 4:,}), ssra {params['ssra'] / GIB:.3f} GiB "
          f"({params['ssra'] // 4:,})")

    m1_cand = [1, 2, 4, 8, 16, 32, 64]
    m2_cand = [1, 2, 4, 5, 10, 20]
    sep = "|" if args.markdown else " "
    hdr = (f"{sep} mode {sep} model {sep} N {sep} B {sep} projected GiB "
           f"{sep} x1.20 GiB {sep} gate 48GB (<=45.6) {sep} gate 80GB "
           f"(<=76.0) {sep} dominant phase {sep}")
    print()
    print(hdr)
    if args.markdown:
        print("|---" * 9 + "|")
    chosen: dict = {}
    for mode, cand in (("m1", m1_cand), ("m2", m2_cand)):
        for arch in ("flat", "ssra"):
            for n in GRID:
                # M1: B also capped by the per-cell window count 2^21/N;
                # M2: by the 20 trials per cell.
                cap = (2 ** 21) // n if mode == "m1" else 20
                cands = [c for c in cand if c <= cap]
                p = pick_batch(arch, n, mode, params, CARDS["A6000/L40S 48GB"],
                               cands)
                if p is None:  # fall back to the 80 GB card
                    p = pick_batch(arch, n, mode, params, CARDS["A100 80GB"],
                                   cands)
                    if p is None:
                        raise SystemExit(f"no batch size fits: {arch} {mode} "
                                         f"N={n}")
                ok48 = passes(p["gated_gib"], CARDS["A6000/L40S 48GB"])
                ok80 = passes(p["gated_gib"], CARDS["A100 80GB"])
                chosen[(mode, arch, n)] = p
                print(f"{sep} {mode} {sep} {arch} {sep} {n} {sep} {p['b']} "
                      f"{sep} {p['peak_gib']:.2f} {sep} {p['gated_gib']:.2f} "
                      f"{sep} {'PASS' if ok48 else 'FAIL'} "
                      f"{sep} {'PASS' if ok80 else 'FAIL'} "
                      f"{sep} {p['dominant_phase']} {sep}")

    print()
    print("flat math-fallback SDPA risk (extra bytes if neither flash nor "
          "mem-efficient dispatches; checked by the pod-start "
          "micro-benchmark before any 16k/32k cell):")
    for n in GRID:
        b = chosen[("m1", "flat", n)]["b"]
        extra = flat_math_fallback_extra(b, n) / GIB
        print(f"  N={n:6d} B={b:3d}: +{extra:9.2f} GiB "
              f"{'(infeasible -> de-scope/STOP)' if extra > 40 else ''}")

    print()
    print("batch tables (largest B passing the 48 GB gate; identical "
          "values on 80 GB — batching never affects values, §3):")
    for mode in ("m1", "m2"):
        for arch in ("flat", "ssra"):
            tbl = {n: chosen[(mode, arch, n)]["b"] for n in GRID}
            print(f"  batch_table_{mode}.{arch}: {tbl}")


if __name__ == "__main__":
    main()
