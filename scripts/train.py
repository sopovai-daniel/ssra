"""Config-driven trainer (spec §13). Extends the M1 char-level smoke loop with:
  - tokenized data shards (M2 path): packed uint16 .bin from data_pipeline.py
  - checkpoint / resume to a dir (+ optional GCS), Spot-preemption-safe (AP-11)
  - bf16 autocast option (AP-16); loss/eval accumulation stays fp32
  - throughput logging (M2 calibration): steady-state tok/s + peak VRAM (cuda)
    in the JSONL records and the final summary
The M1 char-level path (data: {url, val_frac}) is unchanged and still valid.

One run = one YAML in experiments/ committed BEFORE launch + a row in
results/runs.md. Functionality only at smoke scale — no quality conclusions
(spec §16, assignment §8).

YAML layout: spec §13 {model:, p3:} sections plus
  arch: ssra | flat | megabyte
  run_name: str
  data:     char mode  -> {url: str, val_frac: float}
            token mode -> {train_bin: str, val_bin: str, vocab: int}
                          (vocab optional if shards_meta.json sits beside .bin)
            optional   -> eval_bin: str   (distinct fixed eval shard; final
                          full-coverage eval on it = the sweep selection
                          metric, M2-phase2-sweep §3)
                          tokenizer: str  (frozen artifact, gate only)
                          sha256: {train_bin|val_bin|eval_bin|tokenizer: hex}
                          (hard gates: every listed file's sha256 must match
                          BEFORE any training step, else abort)
  training: {steps, batch_size, seq_len, lr, warmup_steps, seed, device,
             val_every, val_batches, log_every, lr_min_frac, weight_decay,
             precision: fp32|bf16, ckpt_dir, ckpt_every, resume, gcs_ckpt_dir}

Usage:
  .venv/bin/python scripts/train.py experiments/<run>.yaml [--resume] [--dry-run]
Writes logs/<run_name>.log (JSONL) + final summary line. --dry-run parses the
config, builds the model on CPU and resolves data paths, then exits: zero
training steps, no log file, no checkpoint, no GCS access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from baselines.flat import FlatLM  # noqa: E402
from baselines.megabyte import MegabyteLM  # noqa: E402
from ssra import SSRALM, config_from_dict, lb_loss  # noqa: E402
from ssra.checkpoint import (latest_path, load_checkpoint,  # noqa: E402
                             save_checkpoint)
from ssra.data import load_shard  # noqa: E402
from ssra.pool import P3TopKSelect  # noqa: E402

DATA_DIR = ROOT / "data"

BUILDERS = {"ssra": SSRALM, "flat": FlatLM, "megabyte": MegabyteLM}


# ---- data loading -----------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_data_gates(data_cfg: dict) -> list[str]:
    """Hard sha256 gates (M2-phase2-sweep Task B): every key listed under
    data.sha256 names a path entry in the data section; the on-disk file's
    sha256 must equal the recorded value BEFORE any training step. Mismatch
    or missing file aborts the run."""
    gates = data_cfg.get("sha256") or {}
    verified = []
    for key, expected in sorted(gates.items()):
        rel = data_cfg.get(key)
        if rel is None:
            raise SystemExit(f"[gate] sha256 gate names '{key}' but data.{key} "
                             f"is not set in the config")
        p = ROOT / rel
        if not p.exists():
            raise SystemExit(f"[gate] sha256 gate FAILED: {key} missing: {p}")
        actual = sha256_file(p)
        if actual != expected:
            raise SystemExit(f"[gate] sha256 gate FAILED for {key} ({rel}):\n"
                             f"  actual   {actual}\n  expected {expected}")
        print(f"[gate] sha256 OK: {key} ({rel})")
        verified.append(key)
    return verified

def get_corpus(url: str) -> str:
    DATA_DIR.mkdir(exist_ok=True)
    cache = DATA_DIR / hashlib.sha256(url.encode()).hexdigest()[:16]
    if not cache.exists():
        print(f"downloading {url}")
        with urllib.request.urlopen(url) as r:
            cache.write_bytes(r.read())
    return cache.read_text(encoding="utf-8")


def load_data(data_cfg: dict):
    """Return (train_ids, val_ids, vocab) as 1-D int64 CPU tensors.

    Char mode (M1): data:{url, val_frac}. Token mode (M2): data:{train_bin,
    val_bin, vocab}. Small Phase-0 shards are loaded fully into RAM; the GPU
    path can switch load_shard() to a memmap window sampler unchanged."""
    if "train_bin" in data_cfg:  # token mode
        train = torch.from_numpy(np.asarray(
            load_shard(str(ROOT / data_cfg["train_bin"])), dtype=np.int64))
        val = torch.from_numpy(np.asarray(
            load_shard(str(ROOT / data_cfg["val_bin"])), dtype=np.int64))
        vocab = data_cfg.get("vocab")
        if vocab is None:  # fall back to a sibling shards_meta.json
            meta = (ROOT / data_cfg["train_bin"]).parent / "shards_meta.json"
            vocab = json.loads(meta.read_text())["vocab"]
        return train, val, int(vocab)

    text = get_corpus(data_cfg["url"])  # char mode
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n_val = int(len(ids) * data_cfg.get("val_frac", 0.1))
    return ids[:-n_val], ids[-n_val:], len(chars)


def batches(data: torch.Tensor, batch_size: int, seq_len: int, gen):
    starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,),
                           generator=gen)
    return torch.stack([data[s : s + seq_len] for s in starts])


# ---- schedules --------------------------------------------------------------

def tau_at(step: int, p3cfg) -> float:
    frac = min(1.0, step / max(1, p3cfg.tau_anneal_steps))
    return p3cfg.tau_start + frac * (p3cfg.tau_end - p3cfg.tau_start)


def set_tau(model, tau: float):
    for mod in model.modules():
        if isinstance(mod, P3TopKSelect):
            mod.tau = tau


def lr_at(step: int, t: dict) -> float:
    if step < t["warmup_steps"]:
        return t["lr"] * (step + 1) / t["warmup_steps"]
    frac = (step - t["warmup_steps"]) / max(1, t["steps"] - t["warmup_steps"])
    floor = t.get("lr_min_frac", 0.1)
    return t["lr"] * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * frac)))


def autocast_ctx(device: str, precision: str):
    """bf16 autocast (AP-16) for cuda/cpu; no-op for fp32 or mps."""
    if precision == "bf16" and device in ("cuda", "cpu"):
        return torch.autocast(device_type=device, dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)


@torch.no_grad()
def final_eval(model, ids: torch.Tensor, t: dict, device: str,
               precision: str) -> dict:
    """Selection-metric eval (M2-phase2-sweep §3): one deterministic full pass
    over the fixed eval shard in non-overlapping windows of seq_len+1 at
    stride seq_len — every token after the first is predicted exactly once;
    the trailing partial window is dropped (count recorded). Batching is the
    training batch size, so the schedule is identical for every run of the
    sweep by construction. Loss accumulates in fp32 (AP-16), token-weighted."""
    model.eval()
    seq, bs = t["seq_len"], t["batch_size"]
    n_win = (len(ids) - 1) // seq
    total_nll, total_tok = 0.0, 0
    for i in range(0, n_win, bs):
        rows = [ids[j * seq : j * seq + seq + 1]
                for j in range(i, min(i + bs, n_win))]
        x = torch.stack(rows).to(device)
        with autocast_ctx(device, precision):
            logits = model(x[:, :-1])[0]
        nll = F.cross_entropy(logits.flatten(0, 1).float(),
                              x[:, 1:].flatten(), reduction="sum")
        total_nll += nll.item()
        total_tok += x.shape[0] * seq
    model.train()
    return {"eval_loss": round(total_nll / total_tok, 5),
            "eval_windows": n_win, "eval_tokens": total_tok,
            "eval_tokens_dropped": len(ids) - 1 - n_win * seq}


@torch.no_grad()
def validate(model, val, t: dict, device: str, gen, precision: str) -> float:
    model.eval()
    losses = []
    for _ in range(t.get("val_batches", 8)):
        x = batches(val, t["batch_size"], t["seq_len"], gen).to(device)
        with autocast_ctx(device, precision):
            logits = model(x)[0]
        losses.append(F.cross_entropy(  # accumulate in fp32 (AP-16)
            logits[:, :-1].flatten(0, 1).float(), x[:, 1:].flatten()).item())
    model.train()
    return sum(losses) / len(losses)


# ---- training ---------------------------------------------------------------

def main(path: str, resume_flag: bool, dry_run: bool = False):
    raw = yaml.safe_load(Path(path).read_text())
    arch = raw.pop("arch", "ssra")
    run_name = raw.pop("run_name", Path(path).stem)
    data_cfg = raw.pop("data")
    t = raw.pop("training")
    device = t.get("device", "cpu")
    precision = t.get("precision", "fp32")
    t["precision"] = precision  # ensure recorded in meta via **t (no duplicate)

    if dry_run:
        # Config-load dry run: parse + validate + model construction + path
        # resolution. Zero training steps, no data load (shards may live only
        # in GCS), no log/checkpoint writes, no GCS access.
        vocab = data_cfg.get("vocab")
        if vocab is None:
            raise SystemExit("[dry-run] data.vocab required (shards not loaded)")
        raw.setdefault("model", {})["vocab"] = int(vocab)
        cfg = config_from_dict(raw)
        model = BUILDERS[arch](cfg)
        paths = {k: {"resolved": str(ROOT / data_cfg[k]),
                     "exists": (ROOT / data_cfg[k]).exists()}
                 for k in ("train_bin", "val_bin", "eval_bin", "tokenizer")
                 if k in data_cfg}
        report = {"dry_run": {
            "run": run_name, "arch": arch, "pool": cfg.pool,
            "params": sum(p.numel() for p in model.parameters()),
            "vocab": int(vocab), "paths": paths,
            "sha256_gates": sorted((data_cfg.get("sha256") or {})),
            "steps": t["steps"],
            "tokens_per_step": t["batch_size"] * t["seq_len"],
            "total_tokens": t["steps"] * t["batch_size"] * t["seq_len"],
            "warmup_steps": t["warmup_steps"], "lr": t["lr"],
            "seed": t["seed"], "precision": precision,
            "gcs_ckpt_dir": t.get("gcs_ckpt_dir"),
        }}
        print(json.dumps(report, indent=2))
        return

    gates_verified = verify_data_gates(data_cfg)  # hard gate, aborts on fail
    train_ids, val_ids, vocab = load_data(data_cfg)
    eval_ids = None
    if "eval_bin" in data_cfg:  # distinct fixed eval shard (selection metric)
        eval_ids = torch.from_numpy(np.asarray(
            load_shard(str(ROOT / data_cfg["eval_bin"])), dtype=np.int64))
    raw.setdefault("model", {})["vocab"] = vocab
    cfg = config_from_dict(raw)
    torch.manual_seed(t["seed"])
    model = BUILDERS[arch](cfg).to(device).train()
    n_params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"],
                            weight_decay=t.get("weight_decay", 0.01),
                            betas=(0.9, 0.95))
    gen = torch.Generator().manual_seed(t["seed"])
    val_gen_seed = t["seed"] + 1  # fixed val batches across runs of a pair

    # ---- resume (AP-11) -----------------------------------------------------
    ckpt_dir = ROOT / t.get("ckpt_dir", f"checkpoints/{run_name}")
    ckpt_every = t.get("ckpt_every", 0)
    gcs_ckpt = t.get("gcs_ckpt_dir")
    resume = resume_flag or t.get("resume", False)
    start_step = 0
    if resume and latest_path(ckpt_dir).exists():
        start_step = load_checkpoint(latest_path(ckpt_dir), model=model,
                                     optimizer=opt, data_gen=gen,
                                     map_location=device)
        print(f"[resume] from step {start_step} ({latest_path(ckpt_dir)})")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    log_path = ROOT / "logs" / f"{run_name}.log"
    log = log_path.open("a" if start_step else "w")
    meta = dict(run=run_name, arch=arch, pool=cfg.pool, params=n_params,
                vocab=vocab, data=data_cfg,
                train_tokens=len(train_ids), val_tokens=len(val_ids),
                eval_tokens=len(eval_ids) if eval_ids is not None else None,
                sha256_verified=gates_verified,
                torch=torch.__version__,
                commit=commit, config=str(Path(path)),
                resumed_from=start_step if start_step else None, **t)
    log.write(json.dumps({"meta": meta}) + "\n")
    print(json.dumps({"meta": meta}, indent=2))

    def checkpoint(step: int):
        save_checkpoint(latest_path(ckpt_dir), step=step, model=model,
                        optimizer=opt, data_gen=gen, config_raw=raw,
                        run_name=run_name, extra={"arch": arch, "vocab": vocab},
                        gcs_dir=gcs_ckpt)

    # Throughput accounting (M2 deliverable 2: cost+throughput logging).
    # Steady-state tok/s excludes the first MEAS_SKIP steps (CUDA context init,
    # allocator growth, autotune); per-step wall time is sampled without extra
    # device syncs — the .item() sync every log_every steps bounds the drift.
    MEAS_SKIP = 10
    tokens_per_step = t["batch_size"] * t["seq_len"]
    train_time, meas_steps = 0.0, 0

    def tok_per_s() -> float | None:
        return round(tokens_per_step * meas_steps / train_time, 1) \
            if train_time > 0 else None

    def peak_vram_gib() -> float | None:
        return round(torch.cuda.max_memory_allocated() / 2**30, 3) \
            if device == "cuda" else None

    t0 = time.time()
    rec = {"step": start_step, "val_loss": float("nan"), "val_bpc": float("nan")}
    for step in range(start_step, t["steps"]):
        step_t0 = time.perf_counter()
        for group in opt.param_groups:
            group["lr"] = lr_at(step, t)
        if arch == "ssra":
            set_tau(model, tau_at(step, cfg.p3))
        x = batches(train_ids, t["batch_size"], t["seq_len"], gen).to(device)
        diag = (step % t.get("log_every", 50) == 0)
        with autocast_ctx(device, precision):
            if arch == "ssra":
                logits, aux = model(x, collect_diagnostics=diag)
            else:
                logits, aux = model(x)
            loss = F.cross_entropy(
                logits[:, :-1].flatten(0, 1).float(), x[:, 1:].flatten())
        lb = lb_loss(aux)
        total = loss + cfg.p3.lambda_lb * lb if torch.is_tensor(lb) else loss
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step >= start_step + MEAS_SKIP:
            train_time += time.perf_counter() - step_t0
            meas_steps += 1

        if diag:
            rec = {"step": step, "train_loss": round(loss.item(), 5),
                   "lr": round(lr_at(step, t), 6),
                   "elapsed_s": round(time.time() - t0, 1)}
            if tok_per_s() is not None:
                rec["tok_per_s"] = tok_per_s()
            if peak_vram_gib() is not None:
                rec["peak_vram_gib"] = peak_vram_gib()
            if torch.is_tensor(lb):
                rec["lb_loss"] = round(lb.item(), 6)
                rec["tau"] = round(tau_at(step, cfg.p3), 4)
            if aux.get("p1_entropy"):
                # P-C diagnostics (test 5): entropy of Q_phi attention maps
                # + per-query participation min/max over latent queries
                ent = aux["p1_entropy"]
                part = torch.stack(aux["p1_participation"]).mean(0)
                rec["p1_attn_entropy"] = round(sum(ent) / len(ent), 4)
                rec["p1_participation_min"] = round(part.min().item(), 4)
                rec["p1_participation_max"] = round(part.max().item(), 4)
            if not math.isfinite(loss.item()):
                rec["divergence"] = True
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(rec)
        if step % t["val_every"] == 0 or step == t["steps"] - 1:
            vgen = torch.Generator().manual_seed(val_gen_seed)
            vl = validate(model, val_ids, t, device, vgen, precision)
            rec = {"step": step, "val_loss": round(vl, 5),
                   "val_bpc": round(vl / math.log(2), 5)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(rec)
        if ckpt_every and (step + 1) % ckpt_every == 0:
            checkpoint(step + 1)  # step+1 = next step to run on resume

    if ckpt_every:
        checkpoint(t["steps"])  # final checkpoint at run end (AP-11)
    ev = None
    if eval_ids is not None:  # selection metric: full pass over eval_bin
        ev = final_eval(model, eval_ids, t, device, precision)
        ev["eval_bin"] = data_cfg["eval_bin"]
        log.write(json.dumps({"final_eval": ev}) + "\n")
        log.flush()
        print(json.dumps({"final_eval": ev}))
    summary = {"final_val_loss": rec["val_loss"], "final_val_bpc": rec["val_bpc"],
               "final_eval_loss": ev["eval_loss"] if ev else None,
               "wall_clock_s": round(time.time() - t0, 1), "run": run_name,
               "tok_per_s": tok_per_s(), "meas_steps": meas_steps,
               "tokens_per_step": tokens_per_step,
               "peak_vram_gib": peak_vram_gib()}
    log.write(json.dumps({"summary": summary}) + "\n")
    log.close()
    print(json.dumps({"summary": summary}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--resume", action="store_true",
                    help="resume from <ckpt_dir>/latest.pt if present (AP-11)")
    ap.add_argument("--dry-run", action="store_true",
                    help="parse config + build model + resolve paths, then "
                         "exit: zero steps, no log/checkpoint/GCS writes")
    args = ap.parse_args()
    main(args.config, args.resume, args.dry_run)
