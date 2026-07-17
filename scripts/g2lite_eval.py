"""M2 G2-lite inference-only eval harness (docs/cc/M2-g2lite.md §3).

Three measurement modes, all `model.eval()` + `torch.no_grad`, bf16 autocast
+ fp32 loss accumulation (AP-16) — byte-for-byte the training eval path:

  m0  anchor: exact replication of the G1 final-eval protocol at N=1,024 on
      val-eval-2M. The eval loop is `final_eval` IMPORTED from
      scripts/train.py (not reimplemented), so the code path is identical by
      construction. Must reproduce the recorded final_eval_loss within the
      configured tolerance, else the harness STOPs (exit 5): it would not be
      measuring the same function. M0 gates m1/m2 (run m0 first).

  m1  ppl vs length: region E = val.bin[offset, offset + tokens) split into
      exactly tokens/N disjoint windows of N tokens per cell (the §3 window
      count). Per window, the model forwards all N tokens; targets exist at
      window positions 2..N (1-indexed) — position 1 of each window has no
      prediction. Cell metric = token-weighted mean NLL over all targets of
      all windows (fp32 CE per position, fp64 accumulation) -> ppl. From the
      SAME forward passes, per-position NLL bucket means over target
      positions (buckets 1-256, 257-512, 513-1024, then doubling).

  m2  needle-lite: the pre-committed passkey suite (scripts/needle_gen.py,
      manifest sha256-gated). Greedy argmax via REPEATED FULL FORWARD for
      both models (single code path; the incremental decode path is
      deliberately not used, §3), <= max_new tokens, stop at eot; metric =
      the first \\d{5} regex match in the decoded continuation == key.

Batch sizes come from the config's batch tables (§5 projection); batching
affects wall-clock only, never values. Outputs are append-only (AP-21):
an existing output file aborts the run instead of being overwritten.

Usage:
  .venv/bin/python scripts/g2lite_eval.py experiments/m2-g2lite-eval.yaml \
      --arch flat|ssra --mode m0|m1|m2 [--device cuda]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from baselines.flat import FlatLM  # noqa: E402
from needle_gen import KEY_RE  # noqa: E402
from ssra import SSRALM, config_from_dict  # noqa: E402
from ssra.data import EOT_TOKEN, load_shard  # noqa: E402
from train import autocast_ctx, final_eval, sha256_file  # noqa: E402

BUILDERS = {"ssra": SSRALM, "flat": FlatLM}
EXIT_ANCHOR_FAIL = 5

# §3 M1 buckets, by 1-indexed target position within the window
BUCKETS = [(1, 256), (257, 512), (513, 1024), (1025, 2048), (2049, 4096),
           (4097, 8192), (8193, 16384), (16385, 32768)]


# ---- gates -------------------------------------------------------------------

def gate_sha256(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise SystemExit(f"[gate] sha256 FAILED for {label} ({path}):\n"
                         f"  actual   {actual}\n  expected {expected}")
    print(f"[gate] sha256 OK: {label}")


def open_output(path: Path):
    """AP-21: nothing is ever overwritten."""
    if path.exists():
        raise SystemExit(f"[gate] output exists, refusing to overwrite "
                         f"(AP-21): {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# ---- model loading -----------------------------------------------------------

def load_model(arch: str, ckpt_path: Path, device: str,
               expected_sha256: str | None = None,
               expected_bytes: int | None = None):
    """Strict checkpoint load into the matching model class (V3)."""
    if expected_bytes is not None:
        actual = ckpt_path.stat().st_size
        if actual != expected_bytes:
            raise SystemExit(f"[gate] checkpoint size mismatch {ckpt_path}: "
                             f"actual {actual}, expected {expected_bytes}")
    if expected_sha256:
        gate_sha256(ckpt_path, expected_sha256, f"checkpoint {arch}")
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = dict(blob["config_raw"])
    raw.setdefault("model", {})["vocab"] = int(blob["extra"]["vocab"])
    cfg = config_from_dict(raw)
    model = BUILDERS[arch](cfg)
    model.load_state_dict(blob["model"], strict=True)  # raises on any mismatch
    model.to(device).eval()
    return model, cfg, blob


# ---- m0: anchor ----------------------------------------------------------------

def check_anchor(measured: float, expected: float, tol: float) -> dict:
    delta = measured - expected
    return {"measured": measured, "expected": expected,
            "delta_nats": round(delta, 6), "tol_nats": tol,
            "pass": abs(delta) <= tol}


def run_m0(model, ids: torch.Tensor, anchor_cfg: dict, arch: str,
           device: str, precision: str) -> dict:
    t = {"seq_len": int(anchor_cfg["seq_len"]),
         "batch_size": int(anchor_cfg["batch_size"])}
    ev = final_eval(model, ids, t, device, precision)  # training code path
    model.eval()  # final_eval() restores train mode; eval-only session
    verdict = check_anchor(ev["eval_loss"],
                           float(anchor_cfg["expected"][arch]),
                           float(anchor_cfg["tol_nats"]))
    return {"mode": "m0", "arch": arch, "final_eval": ev, "anchor": verdict}


# ---- m1: ppl vs length ---------------------------------------------------------

def eval_cell(model, windows: torch.Tensor, batch_size: int, device: str,
              precision: str, ce_chunk: int = 2048) -> dict:
    """One (model, N) cell: mean NLL + per-target-position sums.

    windows: [W, N] int64. Returns fp64-accumulated sums; positions are
    1-indexed target positions (2..N)."""
    assert not model.training
    n_win, n = windows.shape
    pos_nll = torch.zeros(n + 1, dtype=torch.float64)  # index = target pos
    with torch.no_grad():
        for i in range(0, n_win, batch_size):
            x = windows[i:i + batch_size].to(device)
            with autocast_ctx(device, precision):
                logits = model(x)[0]
            for j in range(0, n - 1, ce_chunk):  # fp32 CE, chunked (memory)
                sl = logits[:, j:min(j + ce_chunk, n - 1)]
                tgt = x[:, j + 1:j + 1 + sl.shape[1]]
                nll = F.cross_entropy(sl.float().flatten(0, 1), tgt.flatten(),
                                      reduction="none")
                nll = nll.view(sl.shape[0], -1).sum(0).double().cpu()
                pos_nll[j + 2:j + 2 + nll.shape[0]] += nll
    total_nll = float(pos_nll[2:].sum())
    total_tok = n_win * (n - 1)
    buckets = {}
    for lo, hi in BUCKETS:
        if lo > n:
            break
        hi_c, lo_c = min(hi, n), max(lo, 2)  # position 1 has no target
        cnt = n_win * (hi_c - lo_c + 1)
        buckets[f"{lo}-{hi}"] = round(
            float(pos_nll[lo_c:hi_c + 1].sum()) / cnt, 6)
    mean = total_nll / total_tok
    return {"n": n, "windows": n_win, "tokens_scored": total_tok,
            "mean_nll": round(mean, 6), "ppl": round(math.exp(mean), 4),
            "bucket_mean_nll": buckets}


def run_m1(model, region: torch.Tensor, grid: list[int], batch_table: dict,
           arch: str, device: str, precision: str) -> dict:
    cells = []
    for n in grid:
        assert len(region) % n == 0, (len(region), n)
        t0 = time.time()
        cell = eval_cell(model, region.view(-1, n), int(batch_table[n]),
                         device, precision)
        cell["wall_s"] = round(time.time() - t0, 1)
        cells.append(cell)
        print(json.dumps({"cell": {"arch": arch, **cell}}))
    base = next(c["ppl"] for c in cells if c["n"] == min(grid))
    for c in cells:
        c["r_vs_min_n"] = round(c["ppl"] / base, 4)
    return {"mode": "m1", "arch": arch, "grid": grid,
            "region_tokens": len(region), "cells": cells}


# ---- m2: needle-lite -----------------------------------------------------------

def load_suite(suite_path: Path, expected_sha256: str | None = None):
    blob = suite_path.read_bytes()
    if expected_sha256:
        actual = hashlib.sha256(blob).hexdigest()
        if actual != expected_sha256:
            raise SystemExit(f"[gate] suite sha256 mismatch: actual {actual},"
                             f" expected {expected_sha256}")
    lines = gzip.decompress(blob).decode().splitlines()
    return [json.loads(ln) for ln in lines]


def greedy_generate(model, prompts: torch.Tensor, eot_id: int, max_new: int,
                    device: str, precision: str) -> list[list[int]]:
    """Greedy argmax via repeated FULL forward (§3: single code path, the
    incremental decode path is deliberately not used). prompts: [B, L]."""
    assert not model.training
    x = prompts.to(device)
    b = x.shape[0]
    done = [False] * b
    gen: list[list[int]] = [[] for _ in range(b)]
    with torch.no_grad():
        for _ in range(max_new):
            with autocast_ctx(device, precision):
                logits = model(x)[0][:, -1]
            nxt = logits.float().argmax(dim=-1)
            for r in range(b):
                if not done[r]:
                    tid = int(nxt[r])
                    gen[r].append(tid)
                    if tid == eot_id:
                        done[r] = True
            if all(done):
                break
            x = torch.cat([x, nxt.unsqueeze(1)], dim=1)
    return gen


def score_generation(tok, gen_ids: list[int], eot_id: int, key: int) -> dict:
    if eot_id in gen_ids:
        gen_ids = gen_ids[:gen_ids.index(eot_id)]
    text = tok.decode(gen_ids)
    m = KEY_RE.search(text)
    return {"generated": text, "extracted": m.group(0) if m else None,
            "correct": bool(m and m.group(0) == str(key))}


def run_m2(model, trials: list[dict], grid: list[int], batch_table: dict,
           tok, eot_id: int, max_new: int, arch: str, device: str,
           precision: str) -> dict:
    cells: dict[tuple, list] = {}
    for tr in trials:
        cells.setdefault((tr["n"], tr["depth"]), []).append(tr)
    out_cells = []
    for (n, depth), group in sorted(cells.items()):
        if n not in grid:
            continue
        bs = int(batch_table[n])
        results = []
        t0 = time.time()
        for i in range(0, len(group), bs):
            chunk = group[i:i + bs]
            prompts = torch.tensor([tr["ids"] for tr in chunk],
                                   dtype=torch.long)
            gens = greedy_generate(model, prompts, eot_id, max_new,
                                   device, precision)
            for tr, g in zip(chunk, gens):
                results.append({"trial": tr["trial"], "key": tr["key"],
                                **score_generation(tok, g, eot_id,
                                                   tr["key"])})
        acc = sum(r["correct"] for r in results) / len(results)
        cell = {"n": n, "depth": depth, "trials": len(results),
                "accuracy": round(acc, 4), "wall_s": round(time.time() - t0, 1),
                "results": results}
        out_cells.append(cell)
        print(json.dumps({"cell": {"arch": arch, "n": n, "depth": depth,
                                   "accuracy": cell["accuracy"],
                                   "wall_s": cell["wall_s"]}}))
    return {"mode": "m2", "arch": arch, "max_new": max_new,
            "cells": out_cells}


# ---- CLI ----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--arch", required=True, choices=("flat", "ssra"))
    ap.add_argument("--mode", required=True, choices=("m0", "m1", "m2"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    precision = cfg["precision"]
    arch = args.arch
    ck = cfg["checkpoints"][arch]
    run_name = cfg["run_names"][arch]
    out_dir = ROOT / cfg["outputs"]["dir"]
    out_path = open_output(out_dir / f"{run_name}-{args.mode}.json")

    model, mcfg, blob = load_model(
        arch, ROOT / ck["local"], args.device,
        expected_sha256=ck.get("sha256") or None,
        expected_bytes=ck.get("bytes") or None)
    meta = {"run": run_name, "arch": arch, "mode": args.mode,
            "config": args.config, "checkpoint": ck["local"],
            "ckpt_step": blob.get("step"), "params": sum(
                p.numel() for p in model.parameters()),
            "device": args.device, "precision": precision,
            "torch": torch.__version__}
    print(json.dumps({"meta": meta}))

    if args.mode == "m0":
        a = cfg["anchor"]
        gate_sha256(ROOT / a["eval_bin"], a["sha256"], "eval_bin")
        ids = torch.from_numpy(np.asarray(
            load_shard(str(ROOT / a["eval_bin"])), dtype=np.int64))
        result = run_m0(model, ids, a, arch, args.device, precision)
    elif args.mode == "m1":
        m1 = cfg["m1"]
        gate_sha256(ROOT / m1["val_bin"], m1["sha256"], "val_bin")
        shard = load_shard(str(ROOT / m1["val_bin"]))
        off, ntok = int(m1["offset"]), int(m1["tokens"])
        region = torch.from_numpy(
            np.asarray(shard[off:off + ntok], dtype=np.int64))
        result = run_m1(model, region, list(m1["grid"]),
                        cfg["batch_table_m1"][arch], arch, args.device,
                        precision)
    else:
        m2 = cfg["m2"]
        trials = load_suite(ROOT / m2["suite"], m2.get("suite_sha256_gz"))
        from tokenizers import Tokenizer
        tok_path = ROOT / m2["tokenizer"]
        gate_sha256(tok_path, m2["tokenizer_sha256"], "tokenizer")
        tok = Tokenizer.from_file(str(tok_path))
        eot_id = tok.token_to_id(EOT_TOKEN)
        result = run_m2(model, trials, list(m2["grid"]),
                        cfg["batch_table_m2"][arch], tok, eot_id,
                        int(m2["max_new_tokens"]), arch, args.device,
                        precision)

    result["meta"] = meta
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"written": str(out_path)}))
    if args.mode == "m0" and not result["anchor"]["pass"]:
        print(json.dumps({"STOP": "M0 anchor FAILED — the harness is not "
                          "measuring the trained function; no further "
                          "measurement (assignment §3 M0)."}))
        raise SystemExit(EXIT_ANCHOR_FAIL)


if __name__ == "__main__":
    main()
