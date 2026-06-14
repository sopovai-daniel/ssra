"""M2 data pipeline helpers (assignment AP-9): FineWeb-Edu streaming, a
deterministic document-disjoint train/val split, byte-level BPE training, and
packed uint16 token shards.

Design notes:
- Split is by a stable hash of the document id, so train and val are
  document-disjoint by construction and the assignment to a split does not
  depend on streaming order or document count (AP-9).
- The tokenizer is trained on TRAIN documents only -> it never sees val text
  (AP-9: "document-disjoint from the val split").
- Shards are a flat uint16 token stream (vocab 16384 < 2**16) with documents
  joined by the tokenizer's <|endoftext|> id; this is the standard packed-LM
  layout the harness samples contiguous windows from.

No quality conclusions are drawn here; this module only moves bytes (spec §16,
assignment §8).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

EOT_TOKEN = "<|endoftext|>"          # document separator / BOS-EOS marker
SHARD_DTYPE = np.uint16              # vocab 16384 fits in uint16

DATASET_REPO = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"       # smallest sample subset (AP-9: small shard)
DATASET_LICENSE = "odc-by"           # verified live from the hub card (AP-9)


def split_of(doc_id: str, val_permille: int) -> str:
    """Deterministic document-disjoint split by stable hash of the doc id."""
    h = int(hashlib.sha1(doc_id.encode("utf-8")).hexdigest(), 16)
    return "val" if (h % 1000) < val_permille else "train"


@dataclass
class Provenance:
    dataset_repo: str
    dataset_config: str
    dataset_sha: str           # hub revision = dataset version (AP-9)
    dataset_license: str
    dataset_url: str
    retrieval_date: str        # ISO date, recorded at integration (Pravidlo W)
    n_docs_streamed: int
    n_train_docs: int
    n_val_docs: int
    val_permille: int
    train_chars: int
    val_chars: int


def stream_split_docs(n_docs: int, val_permille: int, retrieval_date: str,
                      log=print) -> tuple[list[str], list[str], Provenance]:
    """Stream n_docs from FineWeb-Edu sample-10BT and split document-disjoint.

    Returns (train_texts, val_texts, provenance). License + version (sha) are
    read live from the hub (AP-9 / Pravidlo W: verify at integration, record
    URL + date)."""
    from datasets import load_dataset
    from huggingface_hub import dataset_info

    info = dataset_info(DATASET_REPO)
    license_tag = (info.card_data or {}).get("license", DATASET_LICENSE)

    ds = load_dataset(DATASET_REPO, name=DATASET_CONFIG, split="train",
                      streaming=True)
    train_texts: list[str] = []
    val_texts: list[str] = []
    for i, rec in enumerate(ds):
        if i >= n_docs:
            break
        text = rec["text"]
        doc_id = str(rec.get("id", i))
        if split_of(doc_id, val_permille) == "val":
            val_texts.append(text)
        else:
            train_texts.append(text)
        if (i + 1) % 2000 == 0:
            log(f"  streamed {i + 1}/{n_docs} docs "
                f"(train={len(train_texts)}, val={len(val_texts)})")

    prov = Provenance(
        dataset_repo=DATASET_REPO,
        dataset_config=DATASET_CONFIG,
        dataset_sha=info.sha,
        dataset_license=str(license_tag),
        dataset_url=f"https://huggingface.co/datasets/{DATASET_REPO}",
        retrieval_date=retrieval_date,
        n_docs_streamed=min(n_docs, len(train_texts) + len(val_texts)),
        n_train_docs=len(train_texts),
        n_val_docs=len(val_texts),
        val_permille=val_permille,
        train_chars=sum(len(t) for t in train_texts),
        val_chars=sum(len(t) for t in val_texts),
    )
    return train_texts, val_texts, prov


def train_byte_level_bpe(texts: list[str], vocab_size: int, out_path: Path,
                         log=print) -> dict:
    """Train a byte-level BPE tokenizer (AP-9: vocab 16384) on `texts`.

    Returns a manifest dict {sha256, vocab_size, n_train_docs, train_chars}.
    The artifact (tokenizer.json) is deterministic for a fixed corpus + params
    (single-threaded trainer)."""
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

    tok = Tokenizer(models.BPE(unk_token=None))
    # Byte-level: lossless over arbitrary bytes, no UNK, GPT-2-style.
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOT_TOKEN],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    log(f"  training byte-level BPE vocab={vocab_size} on "
        f"{len(texts)} docs ({sum(len(t) for t in texts)} chars)...")
    tok.train_from_iterator(texts, trainer=trainer)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path))

    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    return {
        "tokenizer_path": str(out_path),
        "tokenizer_sha256": sha,
        "vocab_size": tok.get_vocab_size(),
        "requested_vocab": vocab_size,
        "n_train_docs": len(texts),
        "train_chars": sum(len(t) for t in texts),
        "eot_token": EOT_TOKEN,
        "eot_id": tok.token_to_id(EOT_TOKEN),
    }


def pack_shard(texts: list[str], tokenizer_path: Path, out_bin: Path,
               log=print) -> dict:
    """Tokenize `texts` and write a flat uint16 token stream, docs joined by
    the <|endoftext|> id. Returns {path, n_tokens, n_docs, dtype}."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tokenizer_path))
    eot = tok.token_to_id(EOT_TOKEN)
    assert eot is not None and eot < np.iinfo(SHARD_DTYPE).max

    out_bin.parent.mkdir(parents=True, exist_ok=True)
    n_tokens = 0
    with out_bin.open("wb") as f:
        for j, enc in enumerate(tok.encode_batch(texts)):
            ids = enc.ids + [eot]
            arr = np.asarray(ids, dtype=SHARD_DTYPE)
            assert arr.max(initial=0) < np.iinfo(SHARD_DTYPE).max
            f.write(arr.tobytes())
            n_tokens += arr.size
            if (j + 1) % 2000 == 0:
                log(f"  packed {j + 1}/{len(texts)} docs ({n_tokens} tokens)")
    return {"path": str(out_bin), "n_tokens": n_tokens,
            "n_docs": len(texts), "dtype": np.dtype(SHARD_DTYPE).name}


def load_shard(bin_path: str) -> np.ndarray:
    """Memmap a packed uint16 token shard as a read-only 1-D array."""
    return np.memmap(bin_path, dtype=SHARD_DTYPE, mode="r")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(obj, "__dataclass_fields__"):
        obj = asdict(obj)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
