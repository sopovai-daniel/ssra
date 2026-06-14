# M2 Phase-0 report — no-GPU groundwork

**Date:** 2026-06-14 · **Milestone:** M2 (`docs/cc/M2-assignment.md` §4 Phase 0) ·
**Spec:** `docs/spec.md` v1.2 · **Mode:** quota-independent, no GPU, no spend
beyond negligible GCS storage.

**Hard stop respected:** Phase 0 only. Phase 1 (first paid GPU step) **not started**
— GPU quota on `ssra-poc` is 0 (NVIDIA L4 + Preemptible L4 @ europe-west3 not yet
granted). No contingency flags flipped (`summary_pos_override`, `pool_own_proj`,
`p1_diversity_loss>0` all off; `NotImplementedError` stands). No edits to `docs/*`
or `paper/*`. No secrets in the repo.

---

## 1. What landed

| # | deliverable | artifact |
|---|---|---|
| 1 | GCS bucket (europe-west3) | `gs://ssra-poc-ew3` |
| 2 | Pinned deps + GPU image | `requirements.txt` (+data pins), `requirements-gpu.txt`, `docker/Dockerfile` |
| 3 | Data pipeline (FineWeb-Edu → split → tokenizer → packed shards → GCS) | `scripts/data_pipeline.py`, `src/ssra/data.py`, `experiments/M2-phase0-data.yaml`, `results/M2-phase0-data-manifest.json` |
| 4 | Tokenizer (byte-level BPE, vocab 16384) | `artifacts/tokenizer/fineweb-edu-bpe-16384.json` |
| 5 | Harness CPU smoke (extends M1 loop, §13 config-driven, tokenized path, bf16 option AP-16) | `scripts/train.py`, `experiments/M2-phase0-cpu-smoke.yaml`, `logs/M2-phase0-cpu-smoke.log` |
| 6 | Checkpoint/resume (AP-11) + unit test | `src/ssra/checkpoint.py`, `tests/test_checkpoint_resume.py` |

All of the above except the `scripts/train.py` two-line bugfix are committed in
**e4fa16f** (signed). The bugfix is uncommitted, working tree, for Daniel to
commit (see §8).

## 2. GCS bucket

- **Name:** `gs://ssra-poc-ew3`
- **Location:** EUROPE-WEST3 (region) · **Class:** STANDARD · **UBLA:** on ·
  **public-access-prevention:** enforced · soft-delete retention 7 d (default).
- **Contents (Phase 0):** `phase0/data/{train.bin, val.bin,
  fineweb-edu-bpe-16384.json, shards_meta.json, M2-phase0-data-manifest.json}`
  — 5 objects, **14.17 MiB** total (negligible storage).
- Checkpoint mirror path wired in the harness (`training.gcs_ckpt_dir`),
  unused in Phase 0 (local checkpoints only); exercised on GPU in Phase 1.

## 3. Data provenance (AP-9 / Pravidlo W)

License and version **verified live from the hub at integration time**
(`huggingface_hub.dataset_info`), not from memory:

| field | value |
|---|---|
| dataset | `HuggingFaceFW/fineweb-edu`, config `sample-10BT` (smallest sample subset) |
| URL | https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu |
| version (hub revision sha) | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` |
| license | **odc-by** (ODC-By 1.0) |
| retrieval date | **2026-06-14** |
| streamed | 6000 docs → **5722 train / 278 val** (document-disjoint) |
| split rule | `val` iff `sha1(doc.id) % 1000 < 50` — deterministic, order-independent, document-disjoint |
| chars | train 27,848,707 · val 1,428,957 |
| packed tokens | **train 6,526,278 · val 334,393** (uint16, docs joined by `<|endoftext|>`) |

FineWeb-Edu access succeeded; SlimPajama fallback (AP-9) not needed.
Machine-readable manifest: `results/M2-phase0-data-manifest.json` (also in GCS).

## 4. Tokenizer (AP-9)

| field | value |
|---|---|
| type | byte-level BPE (`tokenizers`), no UNK, GPT-2-style ByteLevel pre-tok/decoder |
| vocab | **16384** (requested = achieved); special token `<|endoftext|>` = id 0 |
| trained on | **5722 train docs / 27,848,707 chars** — document-disjoint from val (AP-9) |
| artifact | `artifacts/tokenizer/fineweb-edu-bpe-16384.json` (1.11 MB, committed) |
| **sha256** | **`019568a206fe6ccc4bc2e90c750d660979d3fd3add159e302a0dfa4be0d669a0`** |

Sanity: encode/decode roundtrip is lossless; packing is deterministic (re-pack
reproduced identical token counts).

## 5. Harness CPU smoke (functionality only — no quality conclusions)

Extends the M1 training loop; config-driven per spec §13. Adds: tokenized `.bin`
shard loading, `precision: fp32|bf16` autocast (AP-16; loss/eval accumulation in
fp32), and checkpoint/resume (§6). Char-mode M1 YAMLs remain valid (backward
compatible). Pool = **P1 only** (assignment §3); defaults m=16, w=64, k=2.

Run `M2-phase0-cpu-smoke` (config committed in e4fa16f **before** launch;
`results/runs.md` row):

- SSRA-P1, d=128 / h=4 / L=3, vocab 16384 tied, **2,702,976 params**, seq_len
  256, batch 4, 60 steps, CPU fp32, seed 1337.
- train loss **9.737 → 8.779**; val loss **9.717 → 8.712** (12.568 bits/token);
  **no divergence**, wall-clock 27.4 s.
- P-C diagnostics (test 5, informative): `p1_attn_entropy ≈ ln(32) = 3.466`
  throughout — consistent with the M1 finding (P1 pooling attention ≈ uniform at
  smoke scale; no collapse, no specialization). Logged, not gating.

This is a functionality smoke; **no quality conclusion is drawn** (spec §16,
assignment §8). Loss decreases ⇒ the tokenized end-to-end path trains.

## 6. Checkpoint / resume (AP-11) — continuous loss curve: **YES**

Two independent certificates:

1. **Unit test** `tests/test_checkpoint_resume.py` (CPU fp32, AP-2): an
   interrupted-then-resumed run reproduces the uninterrupted loss curve to
   `atol < 1e-5` at every step, including across the resume seam. Checkpoint
   restores model + optimizer + **both RNG streams** (data sampler + torch
   global) and is written atomically (tmp + `os.replace`). `36 passed, 1
   skipped` for the full suite (the 1 skip = the known M1 Triton import skip).

2. **Smoke-scale preemption simulation** (real harness, single-thread for exact
   numerics): kill the process right after the step-30 checkpoint lands, then
   `--resume`. The resumed run reproduced the uninterrupted reference
   **bit-for-bit**: step 30 train `8.98637`, step 40 train `8.78579` / val
   `8.81195`, step 50 train `8.77946`, final val `8.71173` — all identical.
   No discontinuity at the seam.

   (Aside, documented so it is not mistaken for a bug: the cosine LR schedule
   `lr_at(step, steps)` depends on the *total* step count, so a resume must keep
   the same `training.steps` as the original run — which a real preemption does.
   A naive "train 30 then start a fresh 60-step run" comparison diverges purely
   because the two runs use different LR schedules, not because of resume.)

Max loss on Spot preemption = one checkpoint interval (AP-11). GPU kill+resume
verification is a Phase-1 item.

## 7. Environment snapshot

| | |
|---|---|
| commit | **e4fa16f** (HEAD); `scripts/train.py` bugfix uncommitted (§8) |
| python | 3.11.15 |
| platform | macOS 26.5.1, arm64 (Apple Silicon), CPU/MPS dev box |
| torch | 2.12.0 · numpy 2.4.6 |
| datasets | 5.0.0 · pyarrow 24.0.0 · tokenizers 0.22.2 · huggingface_hub 1.19.0 |
| gcloud | Google Cloud SDK 572.0.0 (auth: daniel@sopovai.com, project ssra-poc) |

GPU image (`docker/Dockerfile`, `requirements-gpu.txt`): CUDA 12.4 base + torch
2.12.0 cu124 wheels — **pin manifest only; built and verified on the L4 box in
Phase 1** (base-image digest + resolved torch CUDA build hash recorded in
`results/M2-calibration.md` then). Not claimed as built/tested here (no GPU).

## 8. The uncommitted `scripts/train.py` change

Two-line **logging-plumbing bugfix**, NOT a design or spec change:
`precision` was being passed into the `meta` log dict both explicitly and again
via `**training_config`, raising `TypeError: dict() got multiple values for
keyword argument 'precision'`. Fix: record `precision` once (via `**t`) by
dropping the duplicate explicit kwarg. It touches only what is written to the
JSONL run-meta header — no change to the model, loss, optimizer, LR schedule,
data, or any spec-governed behavior. The smoke run and both resume certificates
above were produced with the fixed harness. Left uncommitted for Daniel.

## 9. Open questions / spec ambiguities

**None blocking.** Two notes for the record (no proposed D-log change; nothing
unimplementable):

- *FineWeb-Edu streaming teardown warning.* On interpreter exit, `datasets`'
  parquet generator can emit a harmless `AttributeError ('NoneType' has no
  attribute 'ArrowInvalid')` during GC — iteration completes and exit code is 0.
  Cosmetic; no effect on shard contents (verified by deterministic re-pack).
- *`val_bpc` log field is bits-per-token here, not per-character.* The harness
  field name predates the tokenized path (M1 was char-level). Reported as
  bits/token above; the harness label is left unchanged (cosmetic; out of scope
  for the requested minimal harness edits).

## 10. Definition-of-done (Phase 0)

| Phase-0 item (assignment §4) | status |
|---|---|
| GCS bucket (ew3) | ✔ `gs://ssra-poc-ew3` |
| environment image / pinned deps | ✔ requirements(+gpu) + Dockerfile (GPU build deferred to Phase 1) |
| data pipeline end-to-end on a small shard | ✔ FineWeb-Edu → split → shards → GCS |
| tokenizer trained (BPE 16384) + hash + sample size | ✔ sha256 019568a2…, 5722 docs |
| harness CPU smoke (extends M1, §13 config-driven) | ✔ run DONE, no divergence |
| checkpoint + resume unit test (AP-11) | ✔ continuous curve, exact |

**Next (blocked on quota, Daniel's side):** Phase 1 calibration = first paid GPU
step → `pytest tests/` on the box (spec §14.1–.3, .7), measured tok/s + peak
VRAM (S1/S2 candidates, both models), one GPU kill+resume, AP-10 price check,
EUR projection → **STOP gate: Daniel confirms plan + HW before Phase 2.**
