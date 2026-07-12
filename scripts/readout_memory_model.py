"""Analytic peak-memory model of the read-out (assignment D5, AP-20 (ii)).

Closed-form enumeration of the tensors autograd retains per layer during one
training forward, for the gathered (pre-2026-07-12) and the grouped (R1+R4)
read-out realizations, as a function of (B, h, N, d_h, w, m, d).

Two modes:
  --validate   compare the closed form against the measured saved-tensor
               audit (scripts/profile_readout_memory.py machinery) on CPU
               fp32 at small batch, for both paths + the non-read-out layer
               remainder used by the total-model estimate.
  --table      evaluate the GPU bf16-autocast projection for the calibration
               configs S1/S2 x b16/b32/b64 and print the total-model estimate
               for S2 b16 (go/no-go input for re-calibration).

GPU projection assumptions (stated per AP-20; see results/M2-readout-optimization.md):
  * bf16 autocast (AP-16): matmul-saved activations 2 B/elt; softmax runs in
    fp32 under autocast -> probs 4 B/elt + a bf16 cast copy of the prob
    slices consumed by the AV matmuls.
  * The model counts end-of-forward autograd residency x L layers + stack
    terms (logits/CE, embeddings, params+grads+AdamW states). Forward/backward
    transients are covered by the empirical margin calibrated on the measured
    old-path S1 b16 peak (54.67 GiB, results/M2-calibration.md §3).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ssra.fenwick import fenwick_blocks  # noqa: E402

GIB = 1024 ** 3
MIB = 1024 ** 2


def s_l(level: int, m: int) -> int:
    return 1 if level == 0 else min(2 ** level, m)


def k_max(n: int, w: int, m: int) -> int:
    """Worst-case Fenwick row count over positions t = 1..n (p = t - w - 1)."""
    best = 1
    for p in range(1, n - w):
        rows = sum(s_l(level, m) for level, _ in fenwick_blocks(p))
        best = max(best, rows)
    return best


def summary_rows(n: int, m: int) -> int:
    """R: total rows of the flattened summary table (all levels)."""
    return sum((n >> level) * s_l(level, m)
               for level in range(int(math.log2(n)) + 1))


def grouped_levels(n: int, w: int, m: int):
    """Participating levels of the grouped read-out: (level, l2, g2, s)."""
    p_max = n - w - 1
    out = []
    level = 0
    while (1 << level) <= max(p_max, 0) and (1 << level) <= n:
        l2 = 1 << level
        g2 = (p_max // l2 + 1) // 2
        if g2 > 0:
            out.append((level, l2, g2, s_l(level, m)))
        level += 1
    return out


# ---- per-layer read-out residency (bytes) ------------------------------------
# ez = element size of matmul-saved activations; ez_sm = softmax output;
# cast = extra bf16 copy of prob slices under autocast (0 on CPU fp32)

def old_readout_bytes(B, h, N, d_h, d, w, m, ez, ez_sm, cast):
    km = k_max(N, w, m)
    R = summary_rows(N, m)
    width = (w + 1) + km
    return {
        "k_g/v_g gathers [B,h,N,k_max,d_h]": 2 * B * h * N * km * d_h * ez,
        "window K/V contiguous copies [B,h,N,w+1,d_h]":
            2 * B * h * N * (w + 1) * d_h * ez,
        "summary projection inputs (2 x B*R*d)": 2 * B * R * d * ez,
        "probs (softmax out, one softmax)": B * h * N * width * ez_sm,
        "prob-slice copies for AV matmuls": B * h * N * width * (cast or ez),
        "q (rotated)": B * h * N * d_h * ez,
        "z + w_o input": 2 * B * N * d * ez,
        "gather index table (int64)": N * km * 8,
    }


def new_readout_bytes(B, h, N, d_h, d, w, m, ez, ez_sm, cast, level_emb=True):
    c = w
    n_pad = math.ceil(N / c) * c
    lv = grouped_levels(N, w, m)
    s_tot = sum(s for _, _, _, s in lv)
    width = 2 * c + s_tot
    q_lvl = sum(min(2 * g2 * l2, 2 * g2 * l2) for _, l2, g2, _ in lv)  # padded spans
    rows_sel = sum(g2 * s for _, _, g2, s in lv)
    pl_elems = sum(g2 * l2 * s for _, l2, g2, s in lv)
    return {
        "band q_blk + k^T + v (5 x B*h*N'*d_h)": 5 * B * h * n_pad * d_h * ez,
        "band prob-block copy [B,h,N',2c]": B * h * n_pad * 2 * c * (cast or ez),
        "probs (softmax out, one softmax)": B * h * N * width * ez_sm,
        "summary prob-slice cast (autocast only)": B * h * N * s_tot * cast,
        "per-level q copies (sum 2*g2*2^l)": B * h * q_lvl * d_h * ez,
        "per-level k^T + v (2 x B*h*rows_sel*d_h)":
            2 * B * h * rows_sel * d_h * ez,
        "per-level projection inputs (rows + tagged)":
            (2 if level_emb else 1) * B * rows_sel * d * ez,
        "per-level prob copies for AV": B * h * pl_elems * (cast or ez),
        "z + w_o input": 2 * B * N * d * ez,
    }


PRESETS = {"s1": dict(d=384, h=6, L=10), "s2": dict(d=640, h=10, L=15)}
N, W, M, VOCAB = 1024, 64, 16, 16384
CPU, GPU = dict(ez=4, ez_sm=4, cast=0), dict(ez=2, ez_sm=4, cast=2)


def total(d): return sum(d.values())


# ---- validation against the measured audit -----------------------------------

def validate():
    import torch
    from profile_readout_memory import SavedTensorAudit
    from ssra import ModelConfig
    from ssra.fenwick import build_readout_index
    from ssra.model import SSRALayer
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
    from test_readout_equiv import reference_readout

    for preset, B in (("s1", 4), ("s2", 2)):
        p = PRESETS[preset]
        cfg = ModelConfig(d=p["d"], h=p["h"], n_layers=1, vocab=VOCAB,
                          n_max=2048)
        torch.manual_seed(0)
        layer = SSRALayer(cfg).train()
        torch.manual_seed(1)
        z0 = torch.randn(B, N, cfg.d)
        with torch.no_grad():
            levels = layer.up_pass(z0, {})
        z = z0.detach().requires_grad_(True)
        levels = [l.detach().requires_grad_(True) for l in levels]
        cache = build_readout_index(N, W, cfg.s_l)
        params = {q.untyped_storage().data_ptr() for q in layer.parameters()}

        args = (B, p["h"], N, p["d"] // p["h"], p["d"], W, M)
        for name, fn, model_fn in (
                ("gathered", lambda: reference_readout(layer, z, levels, cache),
                 old_readout_bytes),
                ("grouped", lambda: layer.readout(z, levels, cache),
                 new_readout_bytes)):
            with SavedTensorAudit(params) as audit:
                out = fn()
            out.sum().backward()
            z.grad = None
            model = total(model_fn(*args, **CPU))
            meas = audit.total_bytes
            print(f"{preset} B={B} {name:8s}: model {model / MIB:8.1f} MiB  "
                  f"audit {meas / MIB:8.1f} MiB  "
                  f"err {100 * (model - meas) / meas:+5.1f}%")

        # non-read-out remainder of the layer (up-pass, FFN, LNs, residuals)
        x = torch.randn(B, N, cfg.d, requires_grad=True)
        with SavedTensorAudit(params) as audit:
            out = layer(x, {}, cache)
        out.sum().backward()
        with SavedTensorAudit(params) as audit_ro:
            out = layer.readout(z, levels, cache)
        out.sum().backward()
        nonreadout = audit.total_bytes - audit_ro.total_bytes
        per_bn = nonreadout / (B * N * cfg.d * 4)
        print(f"{preset} B={B} non-readout layer remainder: "
              f"{nonreadout / MIB:8.1f} MiB = {per_bn:.2f} x B*N*d fp32")


# ---- GPU projection table ------------------------------------------------------

# non-read-out per-layer activation residency, in units of B*N*d elements
# (measured on CPU fp32 via --validate; see report). Includes up-pass node
# attention saves, Pool P1, FFN (d->4d->4d->d saves), LNs, residual streams.
NONREADOUT_UNITS = {"s1": None, "s2": None}  # filled from --validate output


def stack_bytes(preset, B, params_m):
    """Stack-level terms: logits+CE, embedding, params/grads/AdamW."""
    d = PRESETS[preset]["d"]
    logits = B * N * VOCAB * 2          # bf16 matmul output
    log_softmax = B * N * VOCAB * 4     # CE keeps fp32 log-probs
    emb = B * N * d * 2 + B * N * 8
    train_state = params_m * 1e6 * 16   # fp32 params + grads + AdamW m,v
    return logits + log_softmax + emb + train_state


def table(nonreadout_units, margin):
    params = {"s1": 24.2, "s2": 84.6}
    print(f"per-layer read-out residency, GPU bf16 autocast projection "
          f"(N={N}, w={W}, m={M}); k_max={k_max(N, W, M)}")
    print(f"{'config':14s} {'old GiB/layer':>14s} {'new GiB/layer':>14s} "
          f"{'old model total':>16s} {'new model total':>16s}")
    for preset in ("s1", "s2"):
        p = PRESETS[preset]
        for B in (16, 32, 64):
            a = (B, p["h"], N, p["d"] // p["h"], p["d"], W, M)
            old_l = total(old_readout_bytes(*a, **GPU))
            new_l = total(new_readout_bytes(*a, **GPU))
            nonro = nonreadout_units[preset] * B * N * p["d"] * 2  # bf16
            stack = stack_bytes(preset, B, params[preset])
            old_t = (margin * (p["L"] * (old_l + nonro)) + stack)
            new_t = (margin * (p["L"] * (new_l + nonro)) + stack)
            print(f"{preset.upper()} b{B:<4d}     {old_l / GIB:11.2f}    "
                  f"{new_l / GIB:11.2f}    {old_t / GIB:13.2f}    "
                  f"{new_t / GIB:13.2f}")
    print(f"(totals = margin {margin:.2f} x L x (read-out + non-read-out "
          f"layer) + logits/CE + params/grads/AdamW; margin calibrated on "
          f"measured old S1 b16 = 54.67 GiB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--nonreadout-s1", type=float, default=None,
                    help="non-readout units (x B*N*d) from --validate")
    ap.add_argument("--nonreadout-s2", type=float, default=None)
    ap.add_argument("--margin", type=float, default=1.0)
    args = ap.parse_args()
    if args.validate:
        validate()
    if args.table:
        units = {"s1": args.nonreadout_s1, "s2": args.nonreadout_s2}
        if None in units.values():
            sys.exit("--table needs --nonreadout-s1/--nonreadout-s2 "
                     "(run --validate first)")
        table(units, args.margin)
