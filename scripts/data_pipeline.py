"""M2 Phase-0 data pipeline (assignment AP-9), end-to-end on a small shard.

Stages (all run by default; config-driven):
  1. fetch     stream N docs from FineWeb-Edu sample-10BT, deterministic
               document-disjoint train/val split, cache raw docs + provenance
  2. tokenizer train byte-level BPE (vocab 16384) on TRAIN docs only
  3. pack      tokenize train+val into flat uint16 token shards (.bin)
  4. upload    push shards + tokenizer + manifests to the GCS bucket (ew3)

A machine-readable manifest is written to results/M2-phase0-data-manifest.json
(committed); shards live under data/ (gitignored) + GCS. License and dataset
version (sha) are read live from the hub at integration time (Pravidlo W).

Usage:
  .venv/bin/python scripts/data_pipeline.py experiments/M2-phase0-data.yaml
  .venv/bin/python scripts/data_pipeline.py <cfg> --stages fetch,tokenizer,pack
  (omit --upload to skip GCS; default uploads when a gcs_bucket is set)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ssra.data import (  # noqa: E402
    pack_shard, stream_split_docs, train_byte_level_bpe, write_json,
)

MANIFEST = ROOT / "results" / "M2-phase0-data-manifest.json"


def _read_jsonl(path: Path) -> list[str]:
    return [json.loads(line)["text"] for line in path.read_text().splitlines()]


def _write_jsonl(path: Path, texts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for t in texts:
            f.write(json.dumps({"text": t}) + "\n")


def _relativize(obj):
    """Rewrite any absolute repo path to a repo-relative one so the committed
    manifest carries no local home path (history goes public)."""
    root = str(ROOT) + "/"
    if isinstance(obj, str):
        return obj[len(root):] if obj.startswith(root) else obj
    if isinstance(obj, dict):
        return {k: _relativize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_relativize(v) for v in obj]
    return obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--stages", default="fetch,tokenizer,pack,upload")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    stages = set(args.stages.split(","))

    out_dir = ROOT / cfg["out_dir"]
    raw_train = out_dir / "raw_train.jsonl"
    raw_val = out_dir / "raw_val.jsonl"
    tok_path = ROOT / cfg["tokenizer_out"]
    train_bin = out_dir / "train.bin"
    val_bin = out_dir / "val.bin"
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()

    manifest: dict = {"config": str(Path(args.config)), "commit": commit}
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text())
        manifest["config"], manifest["commit"] = str(Path(args.config)), commit

    # ---- 1. fetch + split ---------------------------------------------------
    if "fetch" in stages:
        print("[fetch] streaming FineWeb-Edu sample-10BT ...")
        train_texts, val_texts, prov = stream_split_docs(
            n_docs=cfg["n_docs"], val_permille=cfg["val_permille"],
            retrieval_date=cfg["retrieval_date"])
        _write_jsonl(raw_train, train_texts)
        _write_jsonl(raw_val, val_texts)
        manifest["provenance"] = prov.__dict__
        write_json(MANIFEST, manifest)
        print(f"[fetch] train_docs={prov.n_train_docs} val_docs={prov.n_val_docs} "
              f"license={prov.dataset_license} sha={prov.dataset_sha}")

    # ---- 2. tokenizer (train docs only) -------------------------------------
    if "tokenizer" in stages:
        print("[tokenizer] training byte-level BPE ...")
        train_texts = _read_jsonl(raw_train)
        tok_manifest = train_byte_level_bpe(
            train_texts, vocab_size=cfg["vocab"], out_path=tok_path)
        manifest["tokenizer"] = tok_manifest
        write_json(MANIFEST, manifest)
        print(f"[tokenizer] sha256={tok_manifest['tokenizer_sha256']} "
              f"vocab={tok_manifest['vocab_size']} "
              f"train_docs={tok_manifest['n_train_docs']}")

    # ---- 3. pack shards -----------------------------------------------------
    if "pack" in stages:
        print("[pack] tokenizing + packing shards ...")
        tr = pack_shard(_read_jsonl(raw_train), tok_path, train_bin)
        va = pack_shard(_read_jsonl(raw_val), tok_path, val_bin)
        shard_meta = {"train": tr, "val": va,
                      "vocab": cfg["vocab"],
                      "tokenizer_sha256": manifest.get("tokenizer", {})
                      .get("tokenizer_sha256")}
        write_json(out_dir / "shards_meta.json", shard_meta)
        manifest["shards"] = shard_meta
        write_json(MANIFEST, manifest)
        print(f"[pack] train_tokens={tr['n_tokens']} val_tokens={va['n_tokens']}")

    # ---- 4. upload to GCS ---------------------------------------------------
    if "upload" in stages and cfg.get("gcs_bucket"):
        dst = f"{cfg['gcs_bucket'].rstrip('/')}/{cfg['gcs_prefix'].strip('/')}"
        print(f"[upload] -> {dst}")
        uploaded = []
        for local in (train_bin, val_bin, tok_path,
                       out_dir / "shards_meta.json", MANIFEST):
            if not local.exists():
                continue
            remote = f"{dst}/{local.name}"
            r = subprocess.run(
                ["gcloud", "storage", "cp", str(local), remote],
                capture_output=True, text=True)
            if r.returncode != 0:
                print(f"[upload] FAILED {local.name}: {r.stderr.strip()}")
            else:
                uploaded.append(remote)
                print(f"[upload] ok {remote}")
        manifest["gcs"] = {"dst": dst, "uploaded": uploaded}
        write_json(MANIFEST, manifest)

    manifest = _relativize(manifest)
    write_json(MANIFEST, manifest)
    print(f"[done] manifest -> {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
