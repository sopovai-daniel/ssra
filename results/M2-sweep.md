# M2 Phase 2 report — symmetric S1 lr/dropout sweep (AP-14)

**Assignment:** `docs/cc/M2-phase2-sweep.md` v1.1 (2026-07-13) · **Status:**
Task A (data scale-up) DONE · Task B local prep DONE 2026-07-14 (§B.0) ·
Task B sweep DONE 2026-07-14 (§B.1–B.9): 8/8 runs completed, no divergence;
**selections: flat (1e-3, 0.0), SSRA (1e-3, 0.0)** (§B.4). Billed total
pending console read-out post-terminate (§B.7).

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

### B.1 Pre-flight record (pod `ssra-m2-sweep`, 2026-07-14)

- **AP-19 step 0 (4th consecutive occurrence):** Community price **NOT
  capturable** in the deploy flow on 2026-07-14 (Daniel, deploy console;
  recorded verbatim, no backfill, Pravidlo W). Prior occurrences: 2026-07-12,
  2026-07-13 (recal), 2026-07-14. Consequence: no Secure-vs-Community cost
  line for the sweep (see open question B.8-1).
- **Booked HW:** 1× A100 SXM 80 GB, Secure, region **US-MD-1**; 16 vCPU
  EPYC 7742 quoted at deploy, container disk 60 GB, no network volume.
  Template "Runpod Pytorch 2.4.0 - SSRA". **Pricing (deploy console
  2026-07-14): GPU $1.49/hr + disk $0.008/hr = $1.50/hr total**, billed per
  millisecond. Pod start 10:40 local. `RUNPOD_POD_ID=bq0ky2rcudcsf4`
  (read from `/proc/1/environ`; absent in SSH session env, the known
  RunPod behavior from Phase 1).
- **AP-17 flow:** SA key decoded from `GCP_SA_KEY_B64` via the
  `/proc/1/environ` fallback (SSH session env empty, as in Phase 1);
  key file verified (`ssra-runpod@ssra-poc.iam.gserviceaccount.com`);
  **blocking sanity gate `gsutil ls gs://ssra-poc-ew3` PASSED before any
  billable work** (`/root/bootstrap.out` on the pod; no secret material
  anywhere in logs/repo/chat).
- **Thread limits:** cgroup v2 `cpu.max = 1360000 100000` → quota 13.6
  vCPU → `OMP_NUM_THREADS = MKL_NUM_THREADS = 13` exported for every
  training/pytest process. (Deviation note: 13.6 vCPU quota ≠ 16 vCPU
  quoted at deploy; recorded, not escalated — informative.)
- **Repo delivery:** GitHub clone from the pod failed (repo private via
  unauthenticated HTTPS) → repo shipped as a git bundle at commit
  `52ec36a`, later fast-forwarded to `21a4d9d` (stage-2 configs) —
  both commits pushed to origin BEFORE the corresponding runs started.
- **Data:** bootstrap step 6 pulled
  `gs://ssra-poc-ew3/m2/data/m2-data-900m/{train,val,val-eval-2M}.bin` +
  `shards_meta.json` → `data/m2/` (byte sizes match Task A manifest);
  integrity then enforced per-run by the harness sha256 hard gates
  (all four gates OK in every run's stdout; `sha256_verified` recorded in
  every log meta line). `data_scale.py` NOT run.

### B.2 Environment snapshot + pytest

Pod `bq0ky2rcudcsf4` (`ssra-m2-sweep` class) · NVIDIA A100-SXM4-80GB ·
driver 580.126.16 · torch **2.12.0+cu126** (image shipped 2.4.1+cu124;
bootstrap installed the pinned wheel per the standing pin-manifest
correction) · CUDA 12.6 · Python 3.11.10 · repo commit `52ec36a` for all
six stage-1 runs, `21a4d9d` for both stage-2 runs.

**pytest (before any run):** `1 failed, 64 passed in 25.54s` — the failure
is exactly `tests/test_baselines.py::test_loglinear_integration` (known
Phase-4-only state per assignment §4). Box-specific detail recorded
verbatim: on this pod the fla import chain fails with
`RuntimeError('operator torchvision::nms does not exist')` instead of the
"deferred to M2 GPU" message the test expects locally (where it SKIPS on
missing Triton). No other failure → proceeded.

### B.3 Run table (all 8 runs; EUR = wall-clock × $1.50/hr, ECB 1.1430;
console-authoritative billed total in §B.7)

| run_name | config commit | status | tok/s | peak GiB | final_eval_loss (val-eval-2M) | wall s | ≈EUR |
|---|---|---|---|---|---|---|---|
| m2-sweep-flat-lr1e3-do00 | 482bdb5 | DONE | 311,776 | 6.345 | **4.28121** | 287 | 0.10 |
| m2-sweep-flat-lr6e4-do00 | 482bdb5 | DONE | 311,334 | 6.345 | 4.42130 | 294 | 0.11 |
| m2-sweep-flat-lr3e4-do00 | 482bdb5 | DONE | 312,395 | 6.345 | 4.80148 | 286 | 0.10 |
| m2-sweep-ssra-lr1e3-do00 | 482bdb5 | DONE | 27,062 | 18.557 | **4.23127** | 2,361 | 0.86 |
| m2-sweep-ssra-lr6e4-do00 | 482bdb5 | DONE | 27,209 | 18.557 | 4.34882 | 2,346 | 0.86 |
| m2-sweep-ssra-lr3e4-do00 | 482bdb5 | DONE | 27,062* | 18.557 | 4.69499 | 2,357 | 0.86 |
| m2-sweep-flat-lr1e3-do01 | 21a4d9d | DONE | 305,511 | 6.464 | 4.36339 | 291 | 0.11 |
| m2-sweep-ssra-lr1e3-do01 | 21a4d9d | DONE | 26,483 | 18.908 | 4.35232 | 2,410 | 0.88 |

*ssra-lr3e4 tok/s from its summary line (see `logs/`); wall times from the
driver progress stamps (start→exit, includes gates + data load + final_eval).
No divergence in any cell: zero `divergence` records; every final val/eval
loss is far below its step-0 value (initial val ≈ 9.68–9.77).

**Throughput sanity (informative, non-gating): PASSED** — first SSRA run
steady-state 27,062 tok/s vs recal anchor 27,079 (−0.06 %, well within
±10 %). Peak VRAM 18.557 GiB = recal value exactly. Flat ≈ 312k tok/s vs
recal 319.9k (−2.5 %).

### B.4 Selection (mechanical, within-model, min final_eval_loss on val-eval-2M)

| model | stage-1 winner lr | do 0.0 | do 0.1 | **selected (lr, dropout)** |
|---|---|---|---|---|
| flat | 1e-3 (4.28121 < 4.42130 < 4.80148) | **4.28121** | 4.36339 | **(1e-3, 0.0)** |
| SSRA | 1e-3 (4.23127 < 4.34882 < 4.69499) | **4.23127** | 4.35232 | **(1e-3, 0.0)** |

Justification (one line, mechanical rule of assignment §3): for both models
min final_eval_loss among {winner@do0.0, winner@do0.1} is the do 0.0 cell
(flat 4.28121 < 4.36339; SSRA 4.23127 < 4.35232).
No quality or architecture conclusions from any loss (spec §16);
SSRA-vs-flat loss comparisons are never evidence of anything in this phase.

### B.5 Loss-curve plots

`results/M2-sweep-curves-flat.png` + `results/M2-sweep-curves-ssra.png`
(train faint, val marked, all cells; regenerated with stage 2 included).

### B.6 p1_attn_entropy + participation (standing, informative, non-gating)

Logged at every log interval (25 steps) on all SSRA runs. Across all three
stage-1 SSRA runs (147 samples each): `p1_attn_entropy` stayed in
[3.4655, 3.4657] ≈ ln(32) = 3.4657 from step 0 to step 3650 — the Q_φ
attention map remains ~uniform at 60M-token scale, the same standing
observation as M1 and Phase 0 (D-log 2026-06-12; informative, carries no
gate). Per-query participation stayed in [0.042, 0.085] around the uniform
value 1/16 = 0.0625, no query collapse. Stage-2 SSRA run (do 0.1): same
picture — entropy [3.4655, 3.4657], participation [0.045, 0.092].

### B.7 Cost ledger (vs 300 EUR envelope)

- Pod `ssra-m2-sweep` (`bq0ky2rcudcsf4`), $1.50/hr total, start 10:40 local
  (08:40 UTC); bootstrap + pytest ≈ 0.32 h; runs 10,632 s ≈ 2.95 h (§B.3,
  sum of driver start→exit stamps); terminate signal issued at ≈ 12:10 UTC
  on completion (AP-18). Wall-clock estimate: ≈ 3.5 h × $1.50 ≈ **$5.25 ≈
  4.59 EUR** (ECB 1.1430 carried, no top-up). Per-run EUR in §B.3 sum to
  ≈ 3.88 EUR; the remainder is bootstrap, pytest, winner computation and
  upload overhead.
- **Console-authoritative billed total: to be read from the console
  post-terminate by Daniel; provisional until T+1 re-check per the D-log
  2026-07-13 correction rule.** Not reconstructed from CC-side timestamps
  (Pravidlo W) — the 4.59 EUR figure above is explicitly a wall-clock
  estimate.
- Cumulative M2 spend: 5.39 EUR (confirmed, §A.7) + this pod ≈ 4.59 EUR
  [wall-clock estimate, console pending] ≈ **9.98 EUR of 300** (≈ 3.3 %).
  The 30 EUR pre-approved cap stays scoped exclusively to Phase 3.

**Addendum (2026-07-14, closure — appended, text above unchanged):**
- Console billed total, pod `ssra-m2-sweep`: **$7.2679 ≈ 6.36 EUR** (ECB
  1.1430) — **FINAL by Daniel's decision** (RunPod documents ~1 h billing
  delay; reading taken ≈ 2.5 h after the 13:31 UTC termination; no further
  re-check per the updated read-out rule, D-log 2026-07-14).
- Decomposition (wall-clock estimate, flagged as such; console total
  authoritative, Pravidlo W): ≈ $5.25 work + ≈ $2.03 post-signal idle
  (signal ≈ 12:10 UTC → terminate 13:31 UTC ≈ 1 h 21 min — 3rd occurrence
  of the post-signal idle pattern in M2; no AP change, known cost pattern).
- Cumulative M2 spend: **11.75 EUR ≈ 3.9 %** of the 300 EUR envelope
  (4.05 + 0.85 + 0.49 + 6.36). Oversight review (Claude, 2026-07-14):
  §B.1–B.9 VERIFIED — selections, ledger arithmetic, tok/s consistency
  independently recomputed.

### B.8 Deviations + open questions

Deviations (all explicit, none silent):
1. Pod cgroup CPU quota 13.6 vCPU vs 16 quoted at deploy → thread limits 13
   (rule applied as written; informative).
2. GitHub HTTPS clone unauthenticated fails on the pod → git-bundle
   delivery (commits pushed to origin before runs; no methodology impact).
3. `test_loglinear_integration` fails with a box-specific error text
   (§B.2) — same known test, recorded verbatim.
4. Loss-curve plots are cross-run artifacts, so they are uploaded to
   `gs://ssra-poc-ew3/m2/sweep/plots/` (AP-21 defines per-run dirs only;
   no existing object overwritten). Stage 2 itself: no deviations — peak
   VRAM slightly above the do 0.0 cells (6.464 / 18.908 GiB vs 6.345 /
   18.557) is the expected dropout-mask cost, recorded as data.

Open questions (proposed D-log entries):
1. **AP-19 step 0 [proposed by Daniel with the pod facts, for D-log]:**
   keep AP-19 step 0 (zero cost) but stop treating the Secure-vs-Community
   comparison as an expected deliverable until the Community tier reappears
   in the deploy flow for this GPU class (4th consecutive "not capturable":
   2026-07-12, 2026-07-13, 2026-07-14). Daniel decides over this report.

### B.9 Definition-of-done check

Stage 1: 6/6 DONE, no divergence ✔ · stage 2: 2/2 DONE, no divergence,
configs committed pre-run (21a4d9d) ✔ · selections §B.4 ✔ ·
plots §B.5 ✔ · runs.md rows ✔ · logs + plots + ckpts in
`gs://ssra-poc-ew3/m2/sweep/{run_name}/` ✔ · AP-17 gate before billable
work ✔ · AP-19 step-0 recorded ✔ · pod terminated on explicit signal, billed
total to be filled post-terminate [provisional until T+1] ✔ · no Phase 3
work ✔.
