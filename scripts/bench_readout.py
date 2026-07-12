"""Read-out microbenchmark, old (gathered) vs new (grouped) path
(assignment D6). Forward+backward wall-clock of the read-out alone.

CPU fp32 numbers are the acceptance input (criterion #5, no-regression);
--device mps is informative only (AP-2). Neither promises GPU throughput —
the re-calibration decides (assignment §6).

Usage: python scripts/bench_readout.py [--device cpu] [--reps 5]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from ssra import ModelConfig  # noqa: E402
from ssra.fenwick import build_readout_index  # noqa: E402
from ssra.model import SSRALayer  # noqa: E402
from test_readout_equiv import reference_readout  # noqa: E402

SHAPES = {  # calibration layer shapes, CPU-feasible batch
    "s1-shape": dict(d=384, h=6, batch=4),
    "s2-slice": dict(d=640, h=10, batch=2),
}
N = 1024


def bench(fn, reps: int, sync) -> float:
    for _ in range(2):  # warmup
        fn()
        sync()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        sync()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    dev = args.device
    sync = torch.mps.synchronize if dev == "mps" else (lambda: None)

    for name, sh in SHAPES.items():
        cfg = ModelConfig(d=sh["d"], h=sh["h"], n_layers=1, vocab=256,
                          n_max=2048)
        torch.manual_seed(0)
        layer = SSRALayer(cfg).to(dev).train()
        torch.manual_seed(1)
        z0 = torch.randn(sh["batch"], N, cfg.d, device=dev)
        with torch.no_grad():
            levels0 = layer.up_pass(z0, {})
        cache = build_readout_index(N, cfg.w, cfg.s_l, device=dev)

        def run(fn):
            z = z0.detach().requires_grad_(True)
            levels = [lv.detach().requires_grad_(True) for lv in levels0]
            fn(layer, z, levels, cache).sum().backward()

        t_old = bench(lambda: run(reference_readout), args.reps, sync)
        t_new = bench(lambda: run(SSRALayer.readout), args.reps, sync)
        print(f"{name} (B={sh['batch']}, N={N}, d={sh['d']}, h={sh['h']}, "
              f"device={dev}): old {t_old * 1e3:8.1f} ms  "
              f"new {t_new * 1e3:8.1f} ms  speedup x{t_old / t_new:.2f}")


if __name__ == "__main__":
    main()
