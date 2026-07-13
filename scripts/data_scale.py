"""M2 Task A data scale-up (docs/cc/M2-phase2-sweep.md v1.1 SS2, AP-9/AP-21).

Scales the Phase-0 pipeline to the full Phase 2+3 token budget, changing
nothing methodological: same corpus (FineWeb-Edu sample-10BT), same
deterministic document-disjoint split (sha1(doc.id) % 1000 < val_permille),
same FROZEN Phase-0 tokenizer (sha256 gate below; retraining is an anti-goal).

Differences vs scripts/data_pipeline.py are purely operational:
  - streaming + chunked packing (bounded memory; 900M tokens do not fit the
    Phase-0 all-in-RAM list-of-strings design),
  - no raw-text jsonl cache and no tokenizer stage (the tokenizer is frozen),
  - stop rule: consume stream chunks of `chunk_docs` docs; STOP at the first
    chunk boundary where packed train >= target_train_tokens AND packed
    val >= target_val_tokens (deterministic given the pinned hub revision),
  - fixed eval slice: val-eval-2M = first val_eval_tokens of packed val,
  - sha256 recorded for every produced shard.

Stages: pack (default) -> upload (gcloud storage cp to the AP-21 GCS path).
Manifest -> results/M2-data-900m-manifest.json (committed).

Usage:
  python3.11 scripts/data_scale.py experiments/M2-data-900m.yaml
  python3.11 scripts/data_scale.py <cfg> --stages upload   # re-run upload only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ssra.data import (  # noqa: E402
    EOT_TOKEN, SHARD_DTYPE, split_of, write_json,
)

MANIFEST = ROOT / "results" / "M2-data-900m-manifest.json"
ITEMSIZE = np.dtype(SHARD_DTYPE).itemsize


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _relativize(obj):
    """Committed manifest must carry repo-relative paths only (public history)."""
    root = str(ROOT) + "/"
    if isinstance(obj, str):
        return obj[len(root):] if obj.startswith(root) else obj
    if isinstance(obj, dict):
        return {k: _relativize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_relativize(v) for v in obj]
    return obj


def pack(cfg: dict, manifest: dict) -> None:
    from datasets import load_dataset
    from huggingface_hub import dataset_info
    from tokenizers import Tokenizer

    out_dir = ROOT / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    train_bin = out_dir / "train.bin"
    val_bin = out_dir / "val.bin"
    eval_bin = out_dir / "val-eval-2M.bin"

    # ---- frozen-tokenizer gate (SS2: retraining is an anti-goal) ------------
    tok_path = ROOT / cfg["tokenizer_path"]
    tok_sha = sha256_file(tok_path)
    if tok_sha != cfg["tokenizer_sha256"]:
        sys.exit(f"FATAL: tokenizer sha256 {tok_sha} != frozen "
                 f"{cfg['tokenizer_sha256']} — STOP (frozen artifact only).")
    tok = Tokenizer.from_file(str(tok_path))
    eot = tok.token_to_id(EOT_TOKEN)
    assert eot is not None and eot < np.iinfo(SHARD_DTYPE).max
    print(f"[pack] frozen tokenizer OK: sha256={tok_sha} eot_id={eot}")

    # ---- provenance, live from the hub (AP-9 / Pravidlo W) ------------------
    info = dataset_info(cfg["dataset"])
    license_tag = str((info.card_data or {}).get("license", "UNKNOWN"))
    print(f"[pack] hub revision={info.sha} license={license_tag} "
          f"retrieval_date={cfg['retrieval_date']}")
    # Pin the stream to the recorded revision so provenance == what was read.
    ds = load_dataset(cfg["dataset"], name=cfg["dataset_config"], split="train",
                      streaming=True, revision=info.sha)

    tgt_train, tgt_val = cfg["target_train_tokens"], cfg["target_val_tokens"]
    chunk_docs, max_docs = cfg["chunk_docs"], cfg["max_docs"]
    val_pm = cfg["val_permille"]

    stats = {"train": {"docs": 0, "tokens": 0, "chars": 0},
             "val": {"docs": 0, "tokens": 0, "chars": 0}}
    n_streamed = 0
    t0 = time.time()

    def flush(buf: dict, files: dict) -> None:
        for split, texts in buf.items():
            if not texts:
                continue
            for enc in tok.encode_batch(texts):
                arr = np.asarray(enc.ids + [eot], dtype=SHARD_DTYPE)
                assert arr.max(initial=0) < np.iinfo(SHARD_DTYPE).max
                files[split].write(arr.tobytes())
                stats[split]["tokens"] += arr.size
            stats[split]["docs"] += len(texts)
            stats[split]["chars"] += sum(len(t) for t in texts)
            texts.clear()

    with train_bin.open("wb") as ft, val_bin.open("wb") as fv:
        files = {"train": ft, "val": fv}
        buf: dict = {"train": [], "val": []}
        it = iter(ds)
        done = False
        while not done:
            # ---- one chunk of chunk_docs docs (or stream end) ----------------
            for _ in range(chunk_docs):
                rec = next(it, None)
                if rec is None:
                    done = True
                    break
                doc_id = str(rec.get("id", n_streamed))
                buf[split_of(doc_id, val_pm)].append(rec["text"])
                n_streamed += 1
            flush(buf, files)
            tr, va = stats["train"]["tokens"], stats["val"]["tokens"]
            rate = tr / max(time.time() - t0, 1e-9)
            print(f"[pack] docs={n_streamed} train_tok={tr} ({100*tr/tgt_train:.1f}%) "
                  f"val_tok={va} rate={rate:,.0f} train-tok/s "
                  f"eta={max(tgt_train-tr,0)/max(rate,1e-9)/60:.1f} min", flush=True)
            if tr >= tgt_train and va >= tgt_val:
                done = True
            if n_streamed >= max_docs and not done:
                sys.exit(f"FATAL: max_docs={max_docs} reached with train_tok={tr} "
                         f"< target {tgt_train} — STOP, investigate ratio.")

    tr, va = stats["train"]["tokens"], stats["val"]["tokens"]
    if tr < tgt_train or va < tgt_val:
        sys.exit(f"FATAL: stream exhausted at train_tok={tr}, val_tok={va} "
                 f"< targets ({tgt_train}, {tgt_val}) — STOP.")
    print(f"[pack] DONE in {(time.time()-t0)/60:.1f} min: "
          f"train {tr} tok / {stats['train']['docs']} docs, "
          f"val {va} tok / {stats['val']['docs']} docs")

    # ---- fixed eval slice: first val_eval_tokens of packed val (SS2) --------
    n_eval = cfg["val_eval_tokens"]
    with val_bin.open("rb") as f:
        blob = f.read(n_eval * ITEMSIZE)
    assert len(blob) == n_eval * ITEMSIZE, "val shard shorter than eval slice"
    eval_bin.write_bytes(blob)
    print(f"[pack] val-eval-2M: {n_eval} tokens -> {eval_bin.name}")

    shard_meta = {
        "train": {"path": str(train_bin), "n_tokens": tr,
                  "n_docs": stats["train"]["docs"],
                  "dtype": np.dtype(SHARD_DTYPE).name,
                  "sha256": sha256_file(train_bin)},
        "val": {"path": str(val_bin), "n_tokens": va,
                "n_docs": stats["val"]["docs"],
                "dtype": np.dtype(SHARD_DTYPE).name,
                "sha256": sha256_file(val_bin)},
        "val_eval_2M": {"path": str(eval_bin), "n_tokens": n_eval,
                        "dtype": np.dtype(SHARD_DTYPE).name,
                        "sha256": sha256_file(eval_bin)},
        "vocab": cfg["vocab"],
        "tokenizer_sha256": tok_sha,
    }
    write_json(out_dir / "shards_meta.json", shard_meta)
    manifest["provenance"] = {
        "dataset_repo": cfg["dataset"],
        "dataset_config": cfg["dataset_config"],
        "dataset_sha": info.sha,
        "dataset_license": license_tag,
        "dataset_url": f"https://huggingface.co/datasets/{cfg['dataset']}",
        "retrieval_date": cfg["retrieval_date"],
        "n_docs_streamed": n_streamed,
        "n_train_docs": stats["train"]["docs"],
        "n_val_docs": stats["val"]["docs"],
        "val_permille": val_pm,
        "train_chars": stats["train"]["chars"],
        "val_chars": stats["val"]["chars"],
    }
    manifest["shards"] = shard_meta
    manifest["wall_seconds_pack"] = round(time.time() - t0, 1)
    write_json(MANIFEST, _relativize(manifest))


def upload(cfg: dict, manifest: dict) -> None:
    out_dir = ROOT / cfg["out_dir"]
    dst = f"{cfg['gcs_bucket'].rstrip('/')}/{cfg['gcs_prefix'].strip('/')}"
    print(f"[upload] -> {dst}")
    uploaded = []
    for local in (out_dir / "train.bin", out_dir / "val.bin",
                  out_dir / "val-eval-2M.bin", out_dir / "shards_meta.json",
                  MANIFEST):
        if not local.exists():
            sys.exit(f"FATAL: {local} missing — run the pack stage first.")
        remote = f"{dst}/{local.name}"
        r = subprocess.run(["gcloud", "storage", "cp", str(local), remote],
                           capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"FATAL: upload {local.name} failed: {r.stderr.strip()}")
        uploaded.append(remote)
        print(f"[upload] ok {remote}")
    manifest["gcs"] = {"dst": dst, "uploaded": uploaded}
    write_json(MANIFEST, _relativize(manifest))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--stages", default="pack,upload")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    stages = set(args.stages.split(","))

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    manifest: dict = {}
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text())
    manifest.update({"run_name": cfg["run_name"],
                     "config": str(Path(args.config)), "commit": commit})

    if "pack" in stages:
        pack(cfg, manifest)
    if "upload" in stages:
        upload(cfg, manifest)
    print(f"[done] manifest -> {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
