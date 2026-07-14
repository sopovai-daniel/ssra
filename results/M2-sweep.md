# M2 Phase 2 report — symmetric S1 lr/dropout sweep (AP-14)

**Assignment:** `docs/cc/M2-phase2-sweep.md` v1.1 (2026-07-13) · **Status:**
Task A (data scale-up) DONE · Task B local prep DONE 2026-07-14 (§B.0) —
READY FOR POD · Task B pod execution NOT STARTED (awaits Daniel's pod deploy).

---

## Task A — data scale-up to the full Phase 2+3 token budget (DONE 2026-07-13)

### A.1 Pre-flight record

- **Step 0 (blocking):** `GCP_SA_KEY_B64` verified PRESENT in `/proc/1/environ`
  on pod `ssra-m2-data` before any other work — PASSED (presence checked only;
  value never printed, AP-17 prohibitions).
- **AP-17 sanity gate (blocking):** `gsutil ls gs://ssra-poc-ew3` → listed
  `m2/`, `phase0/` — **PASSED** before any billable work
  (`logs/m2-data-900m-bootstrap.log`).
- **AP-19 (GPU Community price):** not applicable to Task A — no GPU booked on
  this pod. The step-0 AP-19 capture is due at the Task B GPU deploy.
- **Pod:** RunPod CPU-only `ssra-m2-data`, 16 vCPU AMD EPYC 4564P / 32 GB RAM
  (cgroup limit 32,000,000,000 B verified) / 60 GB container disk, no network
  volume, template `runpod-ubuntu`. **Rate: $0.568/hr total** (deploy console,
  2026-07-13, per-second billing; recorded per Pravidlo W from the deploy-day
  console — no Community/Secure split was quoted in the deploy message).
- **Thread limits:** cgroup vCPU quota is unlimited on this pod
  (`cpu.cfs_quota_us = -1`), so per the bootstrap fallback
  `OMP_NUM_THREADS = MKL_NUM_THREADS = RAYON_NUM_THREADS = nproc = 16`;
  `TOKENIZERS_PARALLELISM=true`.
- **Tokenizer freeze gate:** `scripts/data_scale.py` refuses to run unless the
  local artifact's sha256 equals the frozen Phase-0 value — verified
  `019568a206fe6ccc4bc2e90c750d660979d3fd3add159e302a0dfa4be0d669a0` ✓
  (no tokenizer retraining anywhere in this task).

### A.2 Environment snapshot (pod `ssra-m2-data`)

Ubuntu 20.04.6 LTS / kernel 6.8.0-51-generic · 16 vCPU AMD EPYC 4564P ·
cgroup memory limit 32 GB · disk 60 GB (59 GB free at start) · Python 3.11.13 ·
Google Cloud SDK 575.0.1 (installed by bootstrap) · pins: datasets 5.0.0,
pyarrow 24.0.0, tokenizers 0.22.2, huggingface_hub 1.19.0, numpy 2.4.6,
PyYAML 6.0.3 · repo commit at execution: `7bb2d1a` (config committed earlier
in `7597ff4`; both pre-launch). Full snapshot: `logs/m2-data-900m-bootstrap.log`.

**pytest:** not run on this pod — the §4 pytest gate binds the Task B GPU pod;
the CPU pod has no torch (see deviation A.6-1), so `tests/` cannot execute here.

### A.3 Data scale-up: what was produced

Config `experiments/M2-data-900m.yaml` (run_name `m2-data-900m`, AP-21), run
log `logs/m2-data-900m.log`, manifest `results/M2-data-900m-manifest.json`.
Nothing methodological changed vs Phase 0: same corpus, same deterministic
document-disjoint split `sha1(doc.id) % 1000 < 50 → val`, same frozen
tokenizer (byte-level BPE, vocab 16,384).

| shard | tokens | docs | bytes (uint16) | sha256 |
|---|---|---|---|---|
| `train.bin` | **913,605,620** (≥ 900M ✓) | 816,994 | 1,827,211,240 | `6d0e47cdbb75a5148c8588671ad92bc56f4afe24683d1665c6d618f909549ea0` |
| `val.bin` | **48,050,671** (≥ 5M ✓) | 43,006 | 96,101,342 | `03e0dd1a6fb47b57a41e1c800f593b38cd55c24ddb521ac63d1f65cd9e60f35d` |
| `val-eval-2M.bin` | 2,000,000 | — (prefix slice) | 4,000,000 | `bde526d2ee244f44fa0ce9be66d8d561dbc4200ff4ba86ce63bf34336bfef55d` |

- **`val-eval-2M`** = first 2,000,000 tokens of packed val, deterministic.
  Verified independently on the pod: sha256 of the first 4,000,000 bytes of
  `val.bin` equals the recorded slice sha256 ✓. This is the sweep selection
  metric input and candidate G1 eval set for Phase 3 (Daniel confirms in the
  Phase 3 assignment).
- **Stop rule:** stream consumed in chunks of 20,000 docs; stopped at the first
  chunk boundary with train ≥ 900M and val ≥ 5M tokens → 860,000 docs streamed
  total (deterministic given the pinned hub revision).
- **Split check:** 43,006 / 860,000 = 5.001 % of docs to val, document-disjoint
  by construction (same `split_of` as Phase 0, loaded from `src/ssra/data.py`).
- **GCS (AP-21 path):** all five objects under
  `gs://ssra-poc-ew3/m2/data/m2-data-900m/` — `train.bin`, `val.bin`,
  `val-eval-2M.bin`, `shards_meta.json`, `M2-data-900m-manifest.json`;
  remote listing sizes match local byte-for-byte (integrity additionally
  checksum-validated by `gcloud storage cp`). Phase-0 objects untouched.

### A.4 Provenance (AP-9, Pravidlo W — re-recorded at integration time)

- Dataset: `HuggingFaceFW/fineweb-edu`, config `sample-10BT`, split `train`,
  streamed with `revision=` pinned to the live hub revision.
- Hub revision at retrieval: `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`
  (identical to Phase 0). License read live from the hub card: `odc-by`.
  Retrieval date: **2026-07-13**.
- 860,000 docs streamed → 816,994 train (3,879,434,104 chars) + 43,006 val
  (203,925,780 chars). Tokenizer sha256 unchanged (frozen, §A.1).

### A.5 Wall-clock and cost

- Pack + eval slice: **195.9 s**; upload ~66 s; end-to-end well under the
  1–3 h ODHAD (sustained ≈ 4.7M train-tok/s: 16-core Rust BPE ≈ 300k tok/s/core
  at ≈ 21 MB/s text ingest — mutually consistent).
- Pod container start 2026-07-13 20:28:37 UTC; terminate signal issued at
  ≈ 21:05 UTC → ≈ 0.61 h wall ⇒ **≈ $0.35 ≈ 0.30 EUR** at $0.568/hr
  (per-second billing; ECB 1.1430 carried per assignment §4).
- **Billed console total (authoritative): $0.5567 ≈ 0.49 EUR** (0.98 h billed,
  region EUR-IS-1; console 2026-07-13 evening, ECB 1.1430; **provisional**
  pending the T+1 re-check per the D-log 2026-07-13 correction row). Pod
  terminated 2026-07-13 on the explicit completion signal (AP-18).

### A.6 Deviations (all explicit, none silent)

1. **Pod template `runpod-ubuntu` instead of the pytorch image** (Daniel's
   deploy decision, recorded in the task brief): torch is not needed for
   Task A; consequence — no pytest on this pod (§A.2) and deviation 2 below.
2. **Two import-time aborts before the run started** (both on the pod, both
   fixed by committed code changes, no config change):
   (a) `ssra/__init__` imports torch → `ModuleNotFoundError` (fix `28268b4`:
   load `src/ssra/data.py` directly); (b) missing `sys.modules` registration
   broke the `Provenance` dataclass (fix `7bb2d1a`). In both aborts the
   process died at import — the config was never loaded, no tokens were read,
   no GCS object was written. Abort tracebacks preserved as
   `logs/m2-data-900m-import-abort.log` / `-abort2.log`. **AP-21 judgment
   (open question A.8-1):** `run_name` was NOT suffixed `-r1` because no run
   work executed and no committed log or GCS object existed to clobber.
3. **No raw-text jsonl cache** (Phase-0 pipeline cached raw docs to feed
   tokenizer training): operational memory-bound change only; the tokenizer is
   frozen, so no stage consumes raw text besides packing. Methodology
   (corpus, split rule, packing layout, EOT joining) unchanged.
4. Thread-limit rule "floor(cgroup vCPU quota)" resolved by fallback: the pod
   cgroup has no CPU quota (`-1`), so nproc=16 was used (§A.1).
5. Overshoot above targets (913.6M/48.05M vs 900M/5M) is inherent to the
   chunk-boundary stop rule — recorded, not trimmed (no silent truncation).

### A.7 runs.md / ledger

Row `m2-data-900m` appended to `results/runs.md`; cost ledger note added there
(billed console total from §A.5). Cumulative M2 spend after Task A and the
Phase-1 billing correction (console 2026-07-13 evening, see the runs.md
ledger-correction note): **≈ 5.39 EUR** of the 300 EUR envelope (≈ 1.8 %) =
4.05 final (ssra-m2-cal, corrected) + 0.85 provisional (ssra-m2-recal) +
0.49 provisional (ssra-m2-data); the 30 EUR pre-approved cap remains scoped
exclusively to the Phase 3 S2 850M run and was not touched.

### A.8 Open questions (proposed D-log entries)

1. **AP-21 vs pre-run aborts [proposed D-log]:** "An execution attempt that
   dies before the config is loaded (environment/import failure), producing no
   tokens, no checkpoint and no GCS object, does not consume the `run_name`;
   the abort log is preserved under `logs/{run_name}-<reason>.log` and the
   incident is reported. Re-executions after ANY run work retain the strict
   AP-21 `-rN` rule." Daniel to confirm or veto (Task A proceeded under this
   reading; if vetoed, note that `m2-data-900m` GCS objects were written
   exactly once — no overwrite occurred under either reading).

---

## Task B — sweep

### B.0 Local prep (DONE 2026-07-14, local machine only — no pod, no spend, no GCS writes)

**Stage-1 configs (6, committed `482bdb5`):**
`m2-sweep-{ssra,flat}-lr{1e3,6e4,3e4}-do00` per AP-21. All cells: S1
d 384 / h 6 / L 10 (SSRA 24,159,744 / flat 24,021,504 params, dry-run
verified — matches calibration ≈24.2M/24.0M), ctx 1024, b16, grad-accum 1,
bf16 (AP-16), AdamW + cosine (floor `lr_min_frac` 0.1), warmup 55 steps
(≈1.5 %), **3,662 steps = 59,998,208 tok** (≈60M budget), seed 1337,
dropout 0.0, SSRA = P1 default with no contingency flags. Generated from a
single template in one loop — cells differ ONLY in `arch`, `lr` and the
AP-21 name/paths (diff-verified). GCS ckpt dir
`gs://ssra-poc-ew3/m2/sweep/{run_name}` (AP-21). Stage-2 configs are NOT
created — written + committed on the pod after stage-1 winners are computed.

**Intervals picked (one value, identical in all 8 sweep runs; delegated by
assignment "pick one interval"):** `val_every` 200 (loss curves; fixed-seed
val batches, `val_batches` 8), `log_every` 25 (SSRA `p1_attn_entropy` +
per-query participation at every log interval), `ckpt_every` 500 (AP-11
preemption safety; ckpt time excluded from tok/s by the harness design).

**Harness readiness (scripts/train.py, commit `482bdb5`):**
- **Hard sha256 gates before any training step** (`data.sha256`): mismatch or
  missing file aborts (SystemExit) BEFORE the log file is opened — verified
  positively (smoke) and negatively (tampered hash → exit 1, no log written,
  zero steps). Gated values in every sweep config, copied from the committed
  Task A manifest: `train.bin 6d0e47cd…`, `val.bin 03e0dd1a…`,
  `val-eval-2M.bin bde526d2…`, tokenizer `019568a2…` (frozen, unchanged).
- **`data.eval_bin` = distinct eval set:** `val-eval-2M.bin` is loaded as its
  own shard (never a runtime prefix slice of val.bin). **Selection metric
  definition (mechanical, symmetric by construction):** one deterministic
  full pass in non-overlapping windows of seq_len+1 at stride seq_len —
  every token after the first predicted exactly once, trailing partial
  window dropped and its size recorded (`eval_tokens_dropped`; 1,151 tokens
  at 2.0M/1024) — batched at the training batch size (16), bf16 forward +
  fp32 loss accumulation (AP-16), token-weighted mean; written to the log as
  `final_eval` and to the summary as `final_eval_loss`. Identical code path,
  data, and batching for both models.
- **`--dry-run` mode:** parse + validation + model construction + path
  resolution, zero steps, no log/ckpt/GCS access. All 6 configs PASS
  (params + 59,998,208 total tokens as above; shard paths resolve to
  `data/m2/`, present only on the pod; tokenizer present locally, sha ✓).
- **Bootstrap:** `scripts/pod_bootstrap.sh` step 6 pulls
  `gs://ssra-poc-ew3/m2/data/m2-data-900m/{train,val,val-eval-2M}.bin` +
  `shards_meta.json` → `data/m2/` (download only; integrity enforced by the
  harness gates). Identical token stream / document order / step count /
  batch schedule for both models follows from identical configs + seed +
  the shared loader (M2-assignment §3).
- `scripts/data_scale.py` NOT run — data is final (Task A).

**Local verification (free):**
- `pytest tests/` → **64 passed, 1 skipped** — the skip is
  `test_loglinear_integration` (tests/test_baselines.py:47, Triton absent on
  the local machine; this is the same known Phase-4-only item that FAILS on
  the GPU pod per assignment §4). No other failure → proceed.
- Local smoke `m2-sweep-localsmoke-r0` (config committed pre-launch per run
  discipline; throwaway name per the task brief, never one of the six AP-21
  names): tiny SSRA on OLD Phase-0 shards, CPU fp32, 12 steps, no GCS.
  Exercised gates (4/4 OK), distinct `eval_bin` load, `final_eval` full pass
  (1306 windows / 334,336 tokens / 56 dropped), P-C diagnostics, JSONL log
  `logs/m2-sweep-localsmoke-r0.log`. **Harness plumbing only — no
  conclusions of any kind from its loss (spec §16).**

**Ledger (T+1, console data from Daniel 2026-07-14):** `ssra-m2-recal`
$0.9700 CONFIRMED (did not grow to the $1.30 estimate) and `ssra-m2-data`
$0.5567 ≈ 0.49 EUR CONFIRMED — both provisional flags closed, appended to
`results/runs.md`; the billed totals in §A.5 and the Task A ledger note were
already filled and match — verified, not modified. Cumulative ≈ **5.39 EUR
≈ 1.8 %** of the 300 EUR cap.

**Open points for Daniel (none blocking, veto applies):** (1) the
`final_eval` windowing definition above (assignment fixed "fp32 accumulation
+ identical eval batching" but not the exact batching — the full
deterministic non-overlapping pass was chosen as the mechanical symmetric
reading); (2) interval picks (200/25/500) — both stand unless vetoed before
pod deploy.

### B.1+ Pod execution (NOT STARTED)

Separate GPU pod launch per assignment §2/§4; sections (i) AP-19 step-0
capture, (ii) GPU env snapshot + pytest, (iv) run table, (v) selection table,
(vi) loss-curve plots, (vii) `p1_attn_entropy` + participation summary,
(viii) full cost ledger, (ix)–(x) to be filled by the Task B report.
