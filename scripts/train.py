"""Char-level smoke-run trainer (assignment §5: functionality only, no
quality conclusions). One run = one YAML in experiments/ committed BEFORE
launch + a row in results/runs.md.

YAML layout: spec §13 {model:, p3:} sections plus
  arch: ssra | flat
  run_name: str
  data:     {url: str, val_frac: float}
  training: {steps, batch_size, seq_len, lr, warmup_steps, seed, device,
             val_every, val_batches, log_every, lr_min_frac}

Implements test-5 hooks: P-C collapse diagnostics (P1 latent-query attention
entropy + per-query participation, spec §14.5) and the P3 stabilization of
spec §5.3 (load-balance loss * lambda_lb, linear tau anneal).

Usage: .venv/bin/python scripts/train.py experiments/<run>.yaml
Writes logs/<run_name>.log (JSONL) + final summary line.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from baselines.flat import FlatLM  # noqa: E402
from baselines.megabyte import MegabyteLM  # noqa: E402
from ssra import SSRALM, config_from_dict, lb_loss  # noqa: E402
from ssra.pool import P3TopKSelect  # noqa: E402

DATA_DIR = ROOT / "data"


def get_corpus(url: str) -> str:
    DATA_DIR.mkdir(exist_ok=True)
    cache = DATA_DIR / hashlib.sha256(url.encode()).hexdigest()[:16]
    if not cache.exists():
        print(f"downloading {url}")
        with urllib.request.urlopen(url) as r:
            cache.write_bytes(r.read())
    return cache.read_text(encoding="utf-8")


def batches(data: torch.Tensor, batch_size: int, seq_len: int, gen):
    starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,),
                           generator=gen)
    return torch.stack([data[s : s + seq_len] for s in starts])


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


@torch.no_grad()
def validate(model, val, t: dict, device: str, gen) -> float:
    model.eval()
    losses = []
    for _ in range(t.get("val_batches", 8)):
        x = batches(val, t["batch_size"], t["seq_len"], gen).to(device)
        logits = model(x)[0]
        losses.append(F.cross_entropy(
            logits[:, :-1].flatten(0, 1), x[:, 1:].flatten()).item())
    model.train()
    return sum(losses) / len(losses)


def main(path: str):
    raw = yaml.safe_load(Path(path).read_text())
    arch = raw.pop("arch", "ssra")
    run_name = raw.pop("run_name", Path(path).stem)
    data_cfg = raw.pop("data")
    t = raw.pop("training")
    device = t.get("device", "mps")

    text = get_corpus(data_cfg["url"])
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n_val = int(len(ids) * data_cfg.get("val_frac", 0.1))
    train_ids, val_ids = ids[:-n_val], ids[-n_val:]

    raw.setdefault("model", {})["vocab"] = len(chars)
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

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    log_path = ROOT / "logs" / f"{run_name}.log"
    log = log_path.open("w")
    meta = dict(run=run_name, arch=arch, pool=cfg.pool, params=n_params,
                vocab=len(chars), corpus_url=data_cfg["url"],
                corpus_bytes=len(text.encode("utf-8")),
                train_tokens=len(train_ids), val_tokens=len(val_ids),
                dtype="float32", torch=torch.__version__,
                commit=commit, config=str(Path(path)), **t)
    log.write(json.dumps({"meta": meta}) + "\n")
    print(json.dumps({"meta": meta}, indent=2))

    t0 = time.time()
    for step in range(t["steps"]):
        for group in opt.param_groups:
            group["lr"] = lr_at(step, t)
        if arch == "ssra":
            set_tau(model, tau_at(step, cfg.p3))
        x = batches(train_ids, t["batch_size"], t["seq_len"], gen).to(device)
        diag = (step % t.get("log_every", 50) == 0)
        if arch == "ssra":
            logits, aux = model(x, collect_diagnostics=diag)
        else:
            logits, aux = model(x)
        loss = F.cross_entropy(logits[:, :-1].flatten(0, 1), x[:, 1:].flatten())
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
            vl = validate(model, val_ids, t, device, vgen)
            rec = {"step": step, "val_loss": round(vl, 5),
                   "val_bpc": round(vl / math.log(2), 5)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(rec)

    summary = {"final_val_loss": rec["val_loss"], "final_val_bpc": rec["val_bpc"],
               "wall_clock_s": round(time.time() - t0, 1), "run": run_name}
    log.write(json.dumps({"summary": summary}) + "\n")
    log.close()
    print(json.dumps({"summary": summary}))


if __name__ == "__main__":
    main(sys.argv[1])
