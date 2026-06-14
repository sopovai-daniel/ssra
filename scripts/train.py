"""Config-driven trainer (spec §13). Extends the M1 char-level smoke loop with:
  - tokenized data shards (M2 path): packed uint16 .bin from data_pipeline.py
  - checkpoint / resume to a dir (+ optional GCS), Spot-preemption-safe (AP-11)
  - bf16 autocast option (AP-16); loss/eval accumulation stays fp32
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
  training: {steps, batch_size, seq_len, lr, warmup_steps, seed, device,
             val_every, val_batches, log_every, lr_min_frac, weight_decay,
             precision: fp32|bf16, ckpt_dir, ckpt_every, resume, gcs_ckpt_dir}

Usage:
  .venv/bin/python scripts/train.py experiments/<run>.yaml [--resume]
Writes logs/<run_name>.log (JSONL) + final summary line.
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


# ---- data loading -----------------------------------------------------------

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

def main(path: str, resume_flag: bool):
    raw = yaml.safe_load(Path(path).read_text())
    arch = raw.pop("arch", "ssra")
    run_name = raw.pop("run_name", Path(path).stem)
    data_cfg = raw.pop("data")
    t = raw.pop("training")
    device = t.get("device", "cpu")
    precision = t.get("precision", "fp32")
    t["precision"] = precision  # ensure recorded in meta via **t (no duplicate)

    train_ids, val_ids, vocab = load_data(data_cfg)
    raw.setdefault("model", {})["vocab"] = vocab
    cfg = config_from_dict(raw)
    torch.manual_seed(t["seed"])
    builders = {"ssra": SSRALM, "flat": FlatLM, "megabyte": MegabyteLM}
    model = builders[arch](cfg).to(device).train()
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

    t0 = time.time()
    rec = {"step": start_step, "val_loss": float("nan"), "val_bpc": float("nan")}
    for step in range(start_step, t["steps"]):
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

        if diag:
            rec = {"step": step, "train_loss": round(loss.item(), 5),
                   "lr": round(lr_at(step, t), 6),
                   "elapsed_s": round(time.time() - t0, 1)}
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
    summary = {"final_val_loss": rec["val_loss"], "final_val_bpc": rec["val_bpc"],
               "wall_clock_s": round(time.time() - t0, 1), "run": run_name}
    log.write(json.dumps({"summary": summary}) + "\n")
    log.close()
    print(json.dumps({"summary": summary}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--resume", action="store_true",
                    help="resume from <ckpt_dir>/latest.pt if present (AP-11)")
    args = ap.parse_args()
    main(args.config, args.resume)
