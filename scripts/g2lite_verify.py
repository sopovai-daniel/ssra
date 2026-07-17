"""M2 G2-lite pre-launch verifications (docs/cc/M2-g2lite.md §2, V1-V4).

Each subcommand runs one verification block and writes an append-only JSON
record under results/g2lite/ (AP-21: existing outputs are never
overwritten). All checks are read-only with respect to model code and
weights; this script is eval-harness plumbing (assignment §3).

Subcommands:
  local    V1 evidence pointers + V3 config-sha gate + V2(a) n_max guard
           behavior (tiny model) + V2(d) readout_cache build for every grid
           N on CPU (real geometry w=64, m=16).
  v2b      bf16 position-quantization characterization (NO code change):
           instruments rope_rotate at its call sites via wrapper functions
           in THIS script only, runs tiny flat+SSRA models on the eval path
           (model.eval, no_grad, bf16 autocast), records the actual angle
           dtype per site, and tabulates the bf16 integer-position quantum
           per N-range plus the distinct-key-position count in the last
           read-out window.
  ckpt     V2(b,c) + V3 on the downloaded checkpoints: blob sha256 + byte
           size, strict load into the matching class, level_emb shape and
           exact-zero check of rows 11-15, n_max from the stored config.
  eregion  V4: E-region statistics on val.bin (sha256 gate; prefix identity
           with val-eval-2M; token/doc counts; doc-length distribution).

Usage:
  .venv/bin/python scripts/g2lite_verify.py <subcommand> [--out results/g2lite]
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
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from baselines.flat import FlatLM  # noqa: E402
from ssra import ModelConfig, SSRALM  # noqa: E402
from ssra.data import EOT_TOKEN, load_shard  # noqa: E402
from ssra.fenwick import build_readout_index  # noqa: E402

GRID = [1024, 2048, 4096, 8192, 16384, 32768]
CONFIG_COMMIT = "3db45ef"
EXPECTED_CONFIG_SHA16 = {
    "experiments/m2-core-flat-s2-850m-lr6e4.yaml": "ed606161f99e713a",
    "experiments/m2-core-ssra-s2-850m-lr6e4.yaml": "d35a628774f87d65",
}
TOKENIZER = "artifacts/tokenizer/fineweb-edu-bpe-16384.json"
TOKENIZER_SHA = ("019568a206fe6ccc4bc2e90c750d660979d3fd3add159e302"
                 "a0dfa4be0d669a0")
CKPTS = {
    "flat": {"local": "checkpoints/g2lite/flat-latest.pt",
             "gcs": "gs://ssra-poc-ew3/m2/core/m2-core-flat-s2-850m-lr6e4/"
                    "latest.pt", "bytes": 1_011_848_651},
    "ssra": {"local": "checkpoints/g2lite/ssra-latest.pt",
             "gcs": "gs://ssra-poc-ew3/m2/core/m2-core-ssra-s2-850m-lr6e4/"
                    "latest.pt", "bytes": 1_016_124_393},
}
VAL_BIN = "data/m2/val.bin"
VAL_SHA = "03e0dd1a6fb47b57a41e1c800f593b38cd55c24ddb521ac63d1f65cd9e60f35d"
EVAL_BIN = "data/m2/val-eval-2M.bin"
EVAL_SHA = "bde526d2ee244f44fa0ce9be66d8d561dbc4200ff4ba86ce63bf34336bfef55d"
E_OFFSET, E_TOKENS = 2_000_000, 2 ** 21


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_record(out_dir: Path, name: str, record: dict) -> Path:
    out = out_dir / f"verify-{name}.json"
    if out.exists():
        raise SystemExit(f"[gate] output exists, refusing to overwrite "
                         f"(AP-21): {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    print(f"-> {out}")
    return out


# ---- local -------------------------------------------------------------------

def cmd_local(out_dir: Path) -> None:
    rec: dict = {"verification": "local (V1 pointers, V3 config gate, "
                                 "V2a guard, V2d readout_cache)"}

    # V3: config sha256/16 at the training commit AND at HEAD
    cfgs = {}
    for rel, want in EXPECTED_CONFIG_SHA16.items():
        blob = subprocess.run(["git", "show", f"{CONFIG_COMMIT}:{rel}"],
                              capture_output=True, cwd=ROOT, check=True)
        at_commit = hashlib.sha256(blob.stdout).hexdigest()[:16]
        at_head = sha256_file(ROOT / rel)[:16]
        cfgs[rel] = {"expected": want, f"at_{CONFIG_COMMIT}": at_commit,
                     "at_HEAD": at_head,
                     "pass": at_commit == want == at_head}
    rec["config_sha16"] = cfgs

    rec["tokenizer_sha256"] = {
        "expected": TOKENIZER_SHA,
        "actual": sha256_file(ROOT / TOKENIZER),
        "pass": sha256_file(ROOT / TOKENIZER) == TOKENIZER_SHA}

    # V1: evidence pointers (verified by reading; recorded for the report)
    rec["v1_flat_positional"] = {
        "statement": "flat = RoPE at absolute token positions 1..N, same "
                     "rope_base as SSRA, no learned positional parameters, "
                     "no length guard",
        "evidence": ["baselines/flat.py:35-37 (pos=arange(1,n+1), "
                     "rope_rotate on q,k)",
                     "baselines/flat.py:60-81 (FlatLM: token embedding only; "
                     "forward has no n_max check)",
                     "scripts/train.py BUILDERS + final_eval (same "
                     "ModelConfig, autocast bf16, fp32 accumulation)"]}

    # V2(a): the n_max guard line, exercised on a tiny model
    cfg = ModelConfig(d=32, h=4, n_layers=1, vocab=64, n_max=64, m=16, w=8)
    model = SSRALM(cfg).eval()
    with torch.no_grad():
        model(torch.zeros(1, 64, dtype=torch.long))  # N == n_max admitted
        try:
            model(torch.zeros(1, 65, dtype=torch.long))
            guard = "FAIL: N > n_max did not raise"
        except ValueError as e:
            guard = f"raises at N > n_max: {e}"
    rec["v2a_n_max_guard"] = {
        "code": "src/ssra/model.py SSRALM.forward: N > n_max -> ValueError",
        "tiny_model_behavior": guard,
        "grid_admitted": f"trained n_max=32768 admits all grid N {GRID}"}

    # V2(d): readout_cache builds for every grid N on CPU (real geometry)
    geo = ModelConfig(d=640, h=10, n_layers=15, vocab=16384, n_max=32768,
                      m=16, w=64)
    cache_rows = {}
    for n in GRID:
        t0 = time.time()
        idx, mask, vpos = build_readout_index(n, geo.w, geo.s_l)
        cache_rows[n] = {
            "k_max": int(idx.shape[1]),
            "idx_bytes": int(idx.numel() * 8 + mask.numel() + vpos.numel() * 8),
            "build_s": round(time.time() - t0, 2)}
    rec["v2d_readout_cache_cpu"] = cache_rows

    write_record(out_dir, "local", rec)


# ---- v2b ---------------------------------------------------------------------

def cmd_v2b(out_dir: Path) -> None:
    import ssra.model as ssra_model
    import ssra.attention as ssra_attention
    import baselines.flat as flat_mod
    from ssra.rope import rope_rotate as orig_rope

    records: list[dict] = []

    def make_probe(site: str):
        def probe(x, pos, base):
            d_h = x.shape[-1]
            inv_freq = base ** (-torch.arange(0, d_h, 2, dtype=x.dtype,
                                              device=x.device) / d_h)
            angles = pos.to(x.dtype).unsqueeze(-1) * inv_freq
            records.append({"site": site, "x_dtype": str(x.dtype),
                            "pos_dtype": str(pos.dtype),
                            "angle_dtype": str(angles.dtype),
                            "pos_max": int(pos.max())})
            return orig_rope(x, pos, base)
        return probe

    # instrument the three call sites IN THIS PROCESS ONLY (no code change)
    ssra_model.rope_rotate = make_probe("ssra.readout(window/query RoPE)")
    ssra_attention.rope_rotate = make_probe("ssra.node_attn(slot RoPE)")
    flat_mod.rope_rotate = make_probe("flat.attention(absolute RoPE)")
    try:
        cfg = ModelConfig(d=64, h=4, n_layers=1, vocab=64, n_max=128,
                          m=16, w=64)
        x = torch.randint(0, 64, (1, 128))
        for model in (SSRALM(cfg).eval(), FlatLM(cfg).eval()):
            with torch.no_grad(), torch.autocast(device_type="cpu",
                                                 dtype=torch.bfloat16):
                model(x)  # the eval path: eval() + no_grad + bf16 autocast
    finally:
        ssra_model.rope_rotate = orig_rope
        ssra_attention.rope_rotate = orig_rope
        flat_mod.rope_rotate = orig_rope

    sites: dict = {}
    for r in records:
        sites.setdefault(r["site"], r)

    # bf16 integer-position quantization table per grid N
    quant = {}
    for n in GRID:
        pos = torch.arange(1, n + 1, dtype=torch.float64)
        err = pos.to(torch.bfloat16).to(torch.float64) - pos
        win = pos[-65:]  # the last read-out window [N-64, N]
        win_distinct = len(set(win.to(torch.bfloat16).tolist()))
        k = n.bit_length() - 1  # n = 2^k exactly on this grid
        quant[n] = {
            "max_abs_rounding_err": float(err.abs().max()),
            "positions_exact_frac": float((err == 0).double().mean()),
            "quantum_at_top_binade": 2.0 ** (k - 8) if n > 256 else 0.0,
            "distinct_bf16_positions_in_last_window_of_65": win_distinct,
        }
    # empirical quantum per binade [2^k, 2^(k+1))
    binades = {}
    for k in range(7, 15):
        lo = 1 << k
        pos = torch.arange(lo, min(lo * 2, 1 << 15), dtype=torch.float64)
        vals = sorted(set(pos.to(torch.bfloat16).tolist()))
        diffs = {round(b - a, 6) for a, b in zip(vals, vals[1:])}
        binades[f"[{lo}, {min(lo * 2, 1 << 15)})"] = sorted(diffs)

    rec = {
        "verification": "V2b bf16 position quantization (characterization "
                        "only; code untouched, M0 anchor certifies function "
                        "identity)",
        "probe_device": "cpu (autocast bf16; pod pre-flight re-runs this "
                        "probe on cuda before any billable measurement)",
        "angle_dtype_by_site": sites,
        "note_formula": "empirical quantum in binade [2^k, 2^(k+1)) is "
                        "2^(k-7) (bf16 stores 7 mantissa bits; integers "
                        "exact up to 256, first rounding at 257). The "
                        "assignment's parenthetical 'quantum "
                        "2^(floor(log2 t)-8)' is off by one vs this "
                        "empirical table; the table governs (Pravidlo W). "
                        "The qualitative statement (rounding above 256, "
                        "doubling per binade) is unchanged.",
        "quantization_by_grid_n": quant,
        "empirical_quantum_by_binade": binades,
        "shared_effect": "flat: ALL attention positions quantized above "
                         "256; SSRA: read-out window/query RoPE only; SSRA "
                         "intra-node slot positions <= 32 are exact at "
                         "every N",
    }
    write_record(out_dir, "v2b", rec)


def cmd_v2b_window(out_dir: Path) -> None:
    """Verbatim bf16 values for every position of the terminal read-out
    window at N = 32,768 (V2b addendum). The eval-path positions are
    1-indexed (`arange(1, n+1)` in flat attention and the read-out), so the
    last 65-key window of query t = 32,768 is [32704, 32768]; position
    32,703 is included for the 0-indexed reading. Manual round-to-nearest-
    even arithmetic predicts the two representable neighbors
    {32640, 32768} around the window; the verbatim probe values below
    govern (Pravidlo W)."""
    lo, hi = 32703, 32768
    pos = torch.arange(lo, hi + 1, dtype=torch.float64)
    as_bf16 = pos.to(torch.bfloat16).to(torch.float64)
    pairs = {int(p): float(v) for p, v in zip(pos.tolist(), as_bf16.tolist())}
    win_1idx = {p: v for p, v in pairs.items() if p >= 32704}
    rec = {
        "verification": "V2b addendum: verbatim bf16 cast of the terminal "
                        "read-out window positions at N = 32768",
        "indexing": "eval-path positions are 1-indexed arange(1, n+1); "
                    "window of query t=32768 (w=64) = [32704, 32768]; "
                    "32703 recorded for the 0-indexed reading",
        "manual_prediction": "RNE neighbors {32640, 32768}; 32703 -> 32640 "
                             "(distance 63 < 65); 32704 is the exact tie "
                             "32640+64 -> RNE picks the even mantissa "
                             "32768; 32705..32767 -> nearer 32768; 32768 "
                             "exact",
        "verbatim_pos_to_bf16": pairs,
        "distinct_values_window_1indexed_32704_32768":
            sorted(set(win_1idx.values())),
        "distinct_values_incl_32703": sorted(set(pairs.values())),
    }
    write_record(out_dir, "v2b-window", rec)


# ---- ckpt --------------------------------------------------------------------

def cmd_ckpt(out_dir: Path) -> None:
    from baselines.flat import FlatLM
    from ssra import config_from_dict
    rec = {"verification": "V2(b,c) + V3 on downloaded checkpoints "
                           "(first-ever blob read)"}
    for arch, info in CKPTS.items():
        p = ROOT / info["local"]
        if not p.exists():
            raise SystemExit(f"[gate] checkpoint missing: {p} "
                             f"(download from {info['gcs']} first)")
        size = p.stat().st_size
        sha = sha256_file(p)
        blob = torch.load(p, map_location="cpu", weights_only=False)
        raw = dict(blob["config_raw"])
        raw.setdefault("model", {})["vocab"] = int(blob["extra"]["vocab"])
        cfg = config_from_dict(raw)
        model = (SSRALM if arch == "ssra" else FlatLM)(cfg)
        model.load_state_dict(blob["model"], strict=True)  # V3 strict
        entry = {
            "gcs": info["gcs"], "local": info["local"],
            "bytes": size, "bytes_expected": info["bytes"],
            "bytes_match": size == info["bytes"],
            "blob_sha256": sha, "step": int(blob["step"]),
            "run_name": blob.get("run_name"),
            "n_max": cfg.n_max, "params": sum(
                p_.numel() for p_ in model.parameters()),
            "strict_load": "OK (no missing/unexpected keys)",
        }
        if arch == "ssra":
            le = blob["model"]["layers.0.level_emb"]
            shapes = {tuple(blob["model"][k].shape)
                      for k in blob["model"] if k.endswith("level_emb")}
            rows_11_15 = torch.stack(
                [blob["model"][f"layers.{i}.level_emb"][11:16].abs().amax()
                 for i in range(cfg.n_layers)])
            rows_0_10 = torch.stack(
                [blob["model"][f"layers.{i}.level_emb"][:11].abs().amax()
                 for i in range(cfg.n_layers)])
            entry["level_emb"] = {
                "shape_per_layer": sorted(str(s) for s in shapes),
                "n_tables": sum(k.endswith("level_emb")
                                for k in blob["model"]),
                "rows_11_15_max_abs_over_all_layers":
                    float(rows_11_15.max()),
                "rows_11_15_exactly_zero": bool(
                    (rows_11_15 == 0).all()),
                "rows_0_10_max_abs_over_all_layers": float(rows_0_10.max()),
                "dtype": str(le.dtype),
            }
        rec[arch] = entry
    write_record(out_dir, "ckpt", rec)


# ---- eregion -----------------------------------------------------------------

def cmd_eregion(out_dir: Path) -> None:
    from tokenizers import Tokenizer
    val_p, eval_p = ROOT / VAL_BIN, ROOT / EVAL_BIN
    for p, want, label in ((val_p, VAL_SHA, "val.bin"),
                           (eval_p, EVAL_SHA, "val-eval-2M.bin")):
        actual = sha256_file(p)
        if actual != want:
            raise SystemExit(f"[gate] sha256 FAILED for {label}: {actual}")
        print(f"[gate] sha256 OK: {label}")

    val = load_shard(str(val_p))
    ev = load_shard(str(eval_p))
    prefix_identical = bool(np.array_equal(val[:len(ev)], ev))

    tok = Tokenizer.from_file(str(ROOT / TOKENIZER))
    eot_id = tok.token_to_id(EOT_TOKEN)
    region = np.asarray(val[E_OFFSET:E_OFFSET + E_TOKENS])
    eots = np.flatnonzero(region == eot_id)
    # documents fully inside E: spans between consecutive eot markers
    doc_lens = np.diff(eots) - 1 if len(eots) > 1 else np.array([])
    doc_lens = doc_lens[doc_lens > 0]
    rec = {
        "verification": "V4 E-region statistics",
        "region": f"val.bin[{E_OFFSET}, {E_OFFSET + E_TOKENS})",
        "val_bin_tokens": int(len(val)),
        "val_eval_2M_tokens": int(len(ev)),
        "prefix_identity": {
            "statement": "val.bin[:len(val-eval-2M)] == val-eval-2M "
                         "byte-for-byte; E starts at 2,000,000 >= "
                         f"{len(ev)} -> zero overlap",
            "identical": prefix_identical,
            "overlap_tokens": 0 if E_OFFSET >= len(ev) else
            len(ev) - E_OFFSET},
        "e_tokens": int(region.size),
        "eot_id": int(eot_id),
        "eot_count": int(len(eots)),
        "docs_fully_inside": int(len(doc_lens)),
        "doc_len_mean": round(float(doc_lens.mean()), 1) if len(doc_lens)
        else None,
        "doc_len_median": float(np.median(doc_lens)) if len(doc_lens)
        else None,
        "doc_len_p90": float(np.quantile(doc_lens, 0.9)) if len(doc_lens)
        else None,
        "doc_len_max": int(doc_lens.max()) if len(doc_lens) else None,
    }
    write_record(out_dir, "eregion", rec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("subcommand",
                    choices=("local", "v2b", "v2b-window", "ckpt",
                             "eregion"))
    ap.add_argument("--out", default="results/g2lite")
    args = ap.parse_args()
    out_dir = ROOT / args.out
    {"local": cmd_local, "v2b": cmd_v2b, "v2b-window": cmd_v2b_window,
     "ckpt": cmd_ckpt, "eregion": cmd_eregion}[args.subcommand](out_dir)


if __name__ == "__main__":
    main()
