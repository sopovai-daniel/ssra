"""Test 4 (spec §14.4): forward+backward wall-clock and peak memory vs N for
SSRA and the flat baseline at identical d, h, L, on MPS (AP-2).

G1a criterion: log-log slope of SSRA wall-clock <= 1.5 over the range AND
strictly below the flat baseline's slope. The script reports numbers and the
slope check; the gate decision is Daniel's.

AP-6: picks the largest candidate (d, h, L) that fits BOTH models at N=8k.

Usage:
  .venv/bin/python scripts/bench_throughput.py \
      [--device mps] [--out results/M1-throughput] [--max-n 16384]
Writes <out>.json, <out>.png and prints a table; raw stdout should be tee'd
into logs/.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from baselines.flat import FlatLM  # noqa: E402
from ssra import ModelConfig, SSRALM  # noqa: E402

# AP-6 candidates, largest first; same d/h/L for both models always.
# Probed 2026-06-12 on this machine (16 GiB unified memory):
#  - d=384/h=8/L=4 OOMs at N=8k (MPS allocator cap 20.13 GiB);
#  - d=256/h=8/L=4 allocates but page-thrashes at N=8k (SSRA 18.7 GiB, flat
#    17.0 GiB sampled peak; wall-clock jumps 25-28x from 4k -> 8k for BOTH
#    models — measuring swap, not compute; logs/M1-throughput-d256L4-
#    swapbound.log). "Fits" therefore additionally requires the sampled peak
#    to stay under MEM_FIT_FRACTION of physical RAM so the slope reflects
#    compute scaling.
CANDIDATES = [
    dict(d=256, h=8, n_layers=4),
    dict(d=256, h=8, n_layers=2),
    dict(d=192, h=8, n_layers=2),
    dict(d=128, h=4, n_layers=2),
    dict(d=64, h=4, n_layers=2),
]
VOCAB = 256
WARMUP, ITERS = 2, 5
MEM_FIT_FRACTION = 0.70


def physical_ram() -> int:
    import subprocess
    out = subprocess.run(["sysctl", "-n", "hw.memsize"],
                         capture_output=True, text=True).stdout.strip()
    return int(out) if out else 16 * 2**30


def sync(device: str):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def reset_peak(device: str):
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.reset_peak_memory_stats()


def mem_now(device: str) -> float:
    """Allocated bytes right now. MPS has no peak-tracking API (torch 2.12),
    so the benchmark samples driver_allocated_memory after the forward and
    after the backward and reports the max — a documented lower bound on the
    true peak."""
    if device == "mps":
        return torch.mps.driver_allocated_memory()
    if device == "cuda":
        return torch.cuda.max_memory_allocated()
    return 0.0


def fwd_bwd(model, tokens, device: str = "") -> float:
    logits, _ = model(tokens)
    loss = F.cross_entropy(logits[:, :-1].flatten(0, 1),
                           tokens[:, 1:].flatten())
    mem = mem_now(device) if device else 0.0
    loss.backward()
    if device:
        sync(device)
        mem = max(mem, mem_now(device))
    model.zero_grad(set_to_none=True)
    return mem


def bench_one(model, n: int, device: str) -> tuple[float, float]:
    """Returns (median seconds per fwd+bwd, sampled peak bytes)."""
    tokens = torch.randint(0, VOCAB, (1, n), device=device)
    for _ in range(WARMUP):
        fwd_bwd(model, tokens)
    sync(device)
    reset_peak(device)
    times = []
    for _ in range(ITERS):
        sync(device)
        t0 = time.perf_counter()
        fwd_bwd(model, tokens)
        sync(device)
        times.append(time.perf_counter() - t0)
    mem = fwd_bwd(model, tokens, device)  # separate pass: mem sampling syncs
    return statistics.median(times), mem


def build_models(size: dict, n_max: int, device: str):
    cfg = ModelConfig(vocab=VOCAB, n_max=n_max, **size)
    torch.manual_seed(0)
    ssra = SSRALM(cfg).to(device).train()
    torch.manual_seed(0)
    flat = FlatLM(cfg).to(device).train()
    return ssra, flat


def fits(size: dict, n: int, n_max: int, device: str) -> bool:
    limit = MEM_FIT_FRACTION * physical_ram()
    try:
        ssra, flat = build_models(size, n_max, device)
        for name, model in (("ssra", ssra), ("flat", flat)):
            reset_peak(device)  # drop the other model's allocator cache
            mem = fwd_bwd(model, torch.randint(0, VOCAB, (1, n), device=device),
                          device)
            if mem > limit:
                print(f"  candidate {size} rejected at N={n}: {name} sampled "
                      f"peak {mem/2**30:.1f} GiB > {limit/2**30:.1f} GiB "
                      f"(would swap)")
                return False
        sync(device)
        return True
    except RuntimeError as e:
        print(f"  candidate {size} does not fit at N={n}: "
              f"{str(e).splitlines()[0][:120]}")
        return False
    finally:
        reset_peak(device)


def slope(ns: list[int], ts: list[float]) -> float:
    xs = [math.log2(n) for n in ns]
    ys = [math.log2(t) for t in ts]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    return (sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            / sum((x - mx) ** 2 for x in xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default="results/M1-throughput")
    ap.add_argument("--max-n", type=int, default=16384)
    args = ap.parse_args()
    device = args.device

    base_ns = [1024, 2048, 4096, 8192]
    n_max = args.max_n
    print(f"platform: {platform.platform()}")
    print(f"torch {torch.__version__} | device {device} | dtype fp32 | "
          f"B=1, warmup {WARMUP}, iters {ITERS} (median)")

    size = next(s for s in CANDIDATES if fits(s, 8192, n_max, device))
    print(f"AP-6 size selection: {size} (largest candidate fitting both at 8k)\n")

    ssra, flat = build_models(size, n_max, device)
    params = {k: sum(p.numel() for p in m.parameters())
              for k, m in (("ssra", ssra), ("flat", flat))}
    print(f"params: ssra {params['ssra']:,} | flat {params['flat']:,}")

    results = {"meta": {"device": device, "dtype": "float32",
                        "torch": torch.__version__,
                        "platform": platform.platform(), "size": size,
                        "params": params, "batch": 1,
                        "warmup": WARMUP, "iters": ITERS},
               "rows": []}
    ns = list(base_ns)
    if args.max_n >= 16384:
        ns.append(16384)  # spec: +16k if it fits; OOM is caught below
    for n in ns:
        row = {"n": n}
        for name, model in (("ssra", ssra), ("flat", flat)):
            try:
                reset_peak(device)  # peaks must not include the other model
                t, mem = bench_one(model, n, device)
                swap = mem > MEM_FIT_FRACTION * physical_ram()
                row[name] = {"sec": t, "peak_bytes": mem, "swap_bound": swap}
                print(f"N={n:6d} {name:5s} {t*1000:9.1f} ms  "
                      f"peak {mem/2**20:8.1f} MiB"
                      f"{'  [swap-bound, excluded from slope]' if swap else ''}")
            except RuntimeError as e:
                row[name] = {"oom": str(e).splitlines()[0][:120]}
                print(f"N={n:6d} {name:5s} OOM/err: {row[name]['oom']}")
        results["rows"].append(row)

    ok_rows = [r for r in results["rows"]
               if "sec" in r.get("ssra", {}) and "sec" in r.get("flat", {})
               and not (r["ssra"]["swap_bound"] or r["flat"]["swap_bound"])]
    ns_ok = [r["n"] for r in ok_rows]
    s_ssra = slope(ns_ok, [r["ssra"]["sec"] for r in ok_rows])
    s_flat = slope(ns_ok, [r["flat"]["sec"] for r in ok_rows])
    g1a = s_ssra <= 1.5 and s_ssra < s_flat
    results["g1a"] = {"slope_ssra": s_ssra, "slope_flat": s_flat,
                      "range": [min(ns_ok), max(ns_ok)],
                      "criterion": "slope_ssra <= 1.5 and slope_ssra < slope_flat",
                      "passes": g1a}
    print(f"\nlog-log slope over N {min(ns_ok)}..{max(ns_ok)}: "
          f"SSRA {s_ssra:.3f} | flat {s_flat:.3f}")
    print(f"G1a criterion (slope <= 1.5 and < flat): "
          f"{'PASS' if g1a else 'FAIL'} (gate decision: Daniel)")

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(results, indent=2))
    plot(results, out.with_suffix(".png"))
    print(f"wrote {out.with_suffix('.json')} and {out.with_suffix('.png')}")


def plot(results: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, color in (("ssra", "tab:blue"), ("flat", "tab:orange")):
        rows = [(r["n"], r[name]) for r in results["rows"]
                if "sec" in r.get(name, {})]
        ns = [n for n, _ in rows]
        ax1.plot(ns, [v["sec"] * 1000 for _, v in rows], "o-",
                 color=color, label=name)
        ax2.plot(ns, [v["peak_bytes"] / 2**20 for _, v in rows], "o-",
                 color=color, label=name)
    g = results["g1a"]
    size = results["meta"]["size"]
    ax1.set(xscale="log", yscale="log", xlabel="N (tokens)",
            ylabel="fwd+bwd wall-clock [ms]",
            title=f"slopes: SSRA {g['slope_ssra']:.2f}, "
                  f"flat {g['slope_flat']:.2f} "
                  f"({'G1a PASS' if g['passes'] else 'G1a FAIL'})")
    ax2.set(xscale="log", xlabel="N (tokens)", ylabel="peak memory [MiB]",
            title=f"d={size['d']} h={size['h']} L={size['n_layers']}, "
                  f"{results['meta']['device']} fp32")
    for ax in (ax1, ax2):
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
