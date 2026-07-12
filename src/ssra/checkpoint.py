"""Checkpoint / resume for the M2 training harness (assignment AP-11).

A checkpoint captures *everything* needed to resume a continuous loss curve:
model weights, optimizer state, the next step index, and BOTH RNG streams that
drive the trajectory -- the data-sampling generator and torch's global RNG
(dropout). Restoring all of them means the resumed run replays the exact same
batches and stochastic ops, so the loss curve is continuous across a Spot
preemption (AP-11: "resume must produce a continuous loss curve"; max loss on
preemption = one checkpoint interval).

Writes are atomic (tmp file + os.replace) so a kill mid-write cannot corrupt
`latest.pt`. Optional mirror to GCS for Spot-preemption safety.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import torch


def save_checkpoint(path: Path, *, step: int, model, optimizer,
                    data_gen: torch.Generator, config_raw: dict,
                    run_name: str, extra: dict | None = None,
                    gcs_dir: str | None = None) -> Path:
    """Atomically write a checkpoint to `path` (and mirror to GCS if given).

    `step` is the index of the NEXT step to run (i.e. number of completed
    steps), so resume continues without repeating or skipping a step."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "data_gen_state": data_gen.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "config_raw": config_raw,
        "run_name": run_name,
        "extra": extra or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(blob, tmp)
    os.replace(tmp, path)  # atomic on POSIX: latest.pt is never half-written
    if gcs_dir:
        _gcs_upload(path, f"{gcs_dir.rstrip('/')}/{path.name}")
    return path


def load_checkpoint(path: Path, *, model, optimizer,
                    data_gen: torch.Generator, map_location="cpu") -> int:
    """Restore model/optimizer/RNG in place; return the next step to run."""
    blob = torch.load(Path(path), map_location=map_location, weights_only=False)
    model.load_state_dict(blob["model"])
    optimizer.load_state_dict(blob["optimizer"])
    # RNG states must stay CPU ByteTensors: map_location="cuda" (GPU resume)
    # would otherwise move them to the device and set_state() rejects that.
    # Found by the Phase-1 GPU kill+resume verification (AP-11).
    data_gen.set_state(blob["data_gen_state"].cpu())
    torch.set_rng_state(blob["torch_rng_state"].cpu())
    return int(blob["step"])


def latest_path(ckpt_dir: Path) -> Path:
    return Path(ckpt_dir) / "latest.pt"


def _gcs_upload(local: Path, remote: str) -> None:
    r = subprocess.run(["gcloud", "storage", "cp", str(local), remote],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[ckpt] GCS upload failed ({remote}): {r.stderr.strip()}")
