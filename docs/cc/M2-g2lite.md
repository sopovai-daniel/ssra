# Zadanie pre CC: M2 G2-lite — pre-registered inference-only length-extrapolation measurement

**Version:** v1 (2026-07-17, Claude draft; veto window runs — handover to CC = approval)
**Authority chain:** `docs/spec.md` v1.2 (frozen) > D-log 2026-07-17 (second entry, formulation B) > this assignment > everything else.
**Authorization:** D-log 2026-07-17 (2nd): G1 = FAIL is recorded and CLOSED. G2-lite is the single authorized measurement: inference-only, both models, the final lr6e4 checkpoints, **scoped cap 10 EUR**, protocol committed BEFORE any pod launch, result published regardless of direction. **No training. No tuning. No further iteration.**
**Report target:** `results/M2-g2lite.md`. No architecture conclusions anywhere in it (spec §16) — every output is an input for Daniel's decision on the paper's shape.

---

## §1 Object of study and framing (binding)

The objects of study are the two trained artifacts exactly as checkpointed:

| model | GCS checkpoint (final; = `step-51880.pt`) | bytes | config (commit `3db45ef`, sha256/16) |
|---|---|---|---|
| flat lr6e4 | `gs://ssra-poc-ew3/m2/core/m2-core-flat-s2-850m-lr6e4/latest.pt` | 1,011,848,651 | `ed606161f99e713a` |
| SSRA-P1 lr6e4 | `gs://ssra-poc-ew3/m2/core/m2-core-ssra-s2-850m-lr6e4/latest.pt` | 1,016,124,393 | `d35a628774f87d65` |

Shared trained geometry: d=640, h=10, L=15, vocab 16,384 (tied), seq_len at training 1,024, rope_base 10,000; SSRA: m=16, w=64, P1, `n_max: 32768` ("e_l table headroom for M3 length extrapolation", config comment — the headroom this measurement uses).

**What this measures (pre-registered honesty statement, goes verbatim into the report):**
1. The eval corpus is packed short FineWeb-Edu documents (mean doc length [ODHAD] ~1.1k tokens; V4 records the exact number). At window lengths N ≫ 1k most of the context is earlier, unrelated documents. **ppl-vs-length here therefore measures degradation robustness at unseen absolute lengths, not long-range modeling benefit.** The expected shape under graceful extrapolation is a roughly flat ppl(N) curve; an exploding curve indicates positional failure.
2. Positional facts established from code (Claude review 2026-07-17; CC re-verifies as V1/V2):
   - **flat** (`baselines/flat.py`): RoPE at absolute token positions, same rope_base, no learned positional parameters, no length guard — mechanically runs at any N. Literature prior: existing position methods including rotary do not extrapolate efficiently beyond training length (Press, Smith & Lewis, "Train Short, Test Long", arXiv:2108.12409, retrieved 2026-07-17) — flat degradation beyond 1,024 is the expected outcome, and observing it is **not** evidence of an SSRA advantage; it is the known cost of the shared positional family both models were given.
   - **SSRA** (`src/ssra/model.py`, spec §6): intra-node slot-RoPE uses slot indices ≤ 2m = 32 (exactly representable at any N); read-out window RoPE uses absolute positions with relative offsets ≤ w = 64 by construction (RoPE relative-offset property, Su et al., RoFormer, arXiv:2104.09864); summary keys are NoPE + e_ℓ. The only length-dependent trained inputs are (a) e_ℓ rows for levels 11–15 (allocated by `n_max=32768`, never exercised at seq 1,024 — expected exactly zero-init, = the documented ablation-OFF state, spec §6; V2 verifies) and (b) a longer Fenwick key list in the one softmax (≤ 65 + 16·16 = 321 keys at 32k vs 225 at 1k).
   - This asymmetry (bounded positional coordinates by design vs unbounded absolute RoPE) is part of the H1 design bet. It is **reported**, not exploited: no positional rescaling of either model (§3 prohibitions).
3. Both expected-shape statements above are pre-registered priors. Either direction of every result is publishable (D-log formulation B); there is no "good" outcome to steer toward.

## §2 Verifications (local, 0 EUR; results written into report §V BEFORE any pod deploy)

- **V1 — flat positional encoding:** confirm from `baselines/flat.py` + the training call path in `scripts/train.py` that the description in §1.2 is exact (Pravidlo W: from code, not from this assignment).
- **V2 — SSRA length mechanics:** download both checkpoints from GCS locally (≈ 2 × 0.95 GiB; egress $0, verified 2026-07-12). Record sha256 of both blobs (first-ever read; goes into report + eval YAML). Verify: (a) `n_max` guard admits N ≤ 32,768; (b) `level_emb` table shape [16, 640] in the SSRA checkpoint; (c) rows 11–15 are **exactly** 0.0 (zero grad + AdamW decay on zero = zero; any nonzero value is a STOP-and-report finding, not something to silently accept); (d) `readout_cache` builds cleanly for every grid N on CPU.
- **V2b — bf16 position quantization (mandatory characterization, no code change):** under the training/eval autocast path, RoPE angles are computed in the activation dtype; `pos.to(bf16)` rounds integers above 256 (quantum 2^(⌊log2 t⌋−8)). Empirically record the actual dtype of the angle tensor on the eval path and tabulate the quantum per N-range. Handling rule (binding): the eval MUST use the **identical** code path as training (the M0 anchor certifies this); do NOT "fix" precision — a fixed path would evaluate a different function than the one trained. The effect is shared by both models' token-level RoPE (flat: all attention; SSRA: read-out window only; SSRA's intra-node slots ≤ 32 are exact at every N) — characterize and report; do not suppress, do not exploit.
- **V3 — checkpoint integrity:** GCS object sizes match §1 table; config sha256/16 of the committed YAMLs match `ed606161f99e713a` / `d35a628774f87d65`; loading each checkpoint into the matching model class succeeds strict (no missing/unexpected keys).
- **V4 — eval-region statistics:** for the E region defined in §3, record token count, document count (eot boundaries; reuse the T5 tooling), mean/median/p90 doc length, and confirm zero overlap with `val-eval-2M` (E starts at offset 2,000,000 of `val.bin`, whose first 2,000,000 tokens ARE `val-eval-2M` — verified by prefix hash in Phase 2).
- **V5 — needle generator:** confirmed ABSENT from the repo (Claude search 2026-07-17: no `*needle*` anywhere; `03` M3 plans "vlastný generátor v repe"). CC implements `scripts/needle_gen.py` + unit tests: canonical passkey format per Mohtashami & Jaggi (Landmark Attention, arXiv:2305.16300, retrieved 2026-07-17 — the task's defining source): repeated filler sentences, one needle sentence "The pass key is <5-digit>. Remember it. <5-digit> is the pass key.", final prompt ending "…What is the pass key? The pass key is". Unit tests must cover: exact target sequence length control; depth placement; tokenizer round-trip of the key digits (frozen tokenizer sha `019568a2…`); determinism under the suite seed. Multi-needle (P-B, k>1) is M3 scope — NOT here.

## §3 Measurement protocol (pre-registered; deviations only via report §Deviations, never silent)

**Grid:** N ∈ {1,024, 2,048, 4,096, 8,192, 16,384, 32,768}, both models, identical inputs per cell. 32,768 = SSRA's `n_max` bound; flat has no bound; grid stops at 32,768 for symmetry. The 16k/32k points are subject to the §5 VRAM gate and the de-scope ladder — never to mid-run improvisation.

**M0 — anchor (gates everything else):** exact replication of the G1 final-eval protocol at N=1,024 on `val-eval-2M` (1,953 non-overlapping windows / 1,999,872 tokens / 127 dropped, deterministic full pass, bf16 autocast + fp32 loss accumulation, AP-16). Must reproduce final_eval_loss **flat 3.19333 / SSRA 3.29065** within ≤ 1e-3 nats (expected exact). Failure = STOP, report, no further measurement — the harness would not be measuring the same function.

**M1 — ppl vs length:** eval region **E = `val.bin` tokens [2,000,000, 2,000,000 + 2,097,152)** (2^21 tokens; disjoint from the G1 set; every grid N divides 2^21, so every cell uses exactly the same tokens, only windowed differently — window count 2^21/N per cell: 2,048 @ 1k … 64 @ 32k). Per cell: mean NLL over all positions of all windows (fp32 accumulation) → ppl. From the same forward passes, record per-position NLL bucket means (buckets: 1–256, 257–512, 513–1,024, 1,025–2,048, …, 16,385–32,768). One measurement per cell; no re-runs for better numbers.

**M2 — needle-lite (passkey):** suite generated once by `scripts/needle_gen.py` at **seed 20260717**, manifest (all prompts + keys + sha256) committed BEFORE the pod. Cells: N × depth ∈ {0.1, 0.5, 0.9} × 20 trials; prompt token budget = **N − 16** (headroom so prompt + generation stays ≤ `n_max`); identical prompts for both models. Generation: **greedy argmax via repeated full forward** for BOTH models (single code path; the incremental decode path is deliberately NOT used — one fewer equivalence surface in a pre-registered measurement), ≤ 12 generated tokens, stop at eot; metric = exact match of the first 5-digit string in the decoded continuation. Report accuracy per (model, N, depth).

**Execution constants:** `model.eval()`, `torch.no_grad`, dropout 0.0 (as trained), precision = bf16 autocast + fp32 accumulation (AP-16) — byte-for-byte the training eval path. Batch sizes per (model, N) chosen by CC from the §5 projection; batching affects wall-clock only, never values.

**Prohibitions (binding):** no RoPE rescaling of any kind (PI/NTK/YaRN/temperature); no sliding-window or chunked eval for flat; no weight, config, or tokenizer modification; no sampling; no per-cell retries; no decode-path usage; no baselines (b)/(c); no multi-needle; no contingency flags (`summary_pos_override` stays off). Eval-harness plumbing (`scripts/g2lite_eval.py`) is new code and is in scope; model code is not touched (AP-20 does not activate; if any model-code change ever seems necessary → STOP and report).

## §4 Pre-registered interpretation criteria (written before data; labels are measurement outcomes, not architecture conclusions — spec §16)

Let r_M(N) = ppl_M(N) / ppl_M(1,024) on region E.

- **O1 anchor:** M0 pass/fail as defined.
- **O2/O3 degradation labels (per model, per N):** r ≤ 1.10 → `stable`; 1.10 < r ≤ 1.50 → `mild`; 1.50 < r ≤ 10 → `marked`; r > 10 → `collapse`. Pre-registered priors: flat expected to leave `stable` beyond 1,024 (Press et al.); SSRA expected `stable`/`mild` if the §1.2 structural story holds. Either violation of a prior is a finding, reported as such.
- **O4 crossover:** the smallest N (if any) with ppl_SSRA(N) < ppl_flat(N), and the ppl_SSRA(N)/ppl_flat(N) ratio per N. Reported as numbers; the +10.22 % gap at 1,024 is the known starting point.
- **O5 needle floor rule:** if pooled accuracy at N=1,024 is < 20 % for BOTH models, all longer-N needle cells are labeled `UNINFORMATIVE-FLOOR` (85M-param base LM on 850M tokens may simply lack the copy behavior; that is a valid, publishable outcome). Otherwise report the full grids; a model retaining ≥ 50 % of its own 1,024 accuracy at N is labeled `retrieval-retaining` at N.
- **O6 per-position signature (largest achieved N):** if every bucket mean for positions > 1,024 is within **+0.10 nats** of that model's 513–1,024 bucket mean → label `positionally graceful`; otherwise report the bucket profile without a label.
- **O7 e_ℓ note:** SSRA cells at N ≥ 2,048 run with exactly-zero e_ℓ rows for levels 11–15 (= spec ablation-OFF state at those levels); stated in the report wherever O2–O6 SSRA numbers appear.
- All labels + numbers go to Daniel as paper-shape inputs. The report draws no conclusion about the mechanism from any of them.

## §5 VRAM, HW, and cost plan (cap 10 EUR, scoped; supervised session)

- **Projection (before "ready for pod"):** CC produces a peak-VRAM projection table per (model, N, candidate B) using the D5 analytical model (`scripts/readout_memory_model.py`) extended for the no-grad inference path, **×1.20 error bar (AP-22)**. Launch gate: projected×1.20 ≤ **95 % of the booked card's VRAM** (the 76/80 GiB rule of AP-22, stated pro-rata for smaller cards — AP-22 interpretation, veto). Logits materialization at 32k ([B, N, 16384]) must be inside the projection (chunked lm_head/loss is admissible plumbing).
- **GPU selection rule (pre-registered):** this is a quality-only measurement — no throughput claims, so GPU class does not affect validity; bf16 support required (Ampere+). Book the **cheapest available** of {RTX A6000 48 GB, L40S 48 GB, A100 PCIe/SXM 80 GB} whose projection passes the gate at usable batch sizes; on-demand Secure; prices from the console on deploy day (Pravidlo W). AP-19 step 0: attempt Community price capture, record verbatim (7th attempt).
- **Wall/cost [ODHAD, planning only; console authoritative]:** SSRA forward-only throughput is unmeasured — training was 12.4k tok/s incl. backward; assume conservatively 2× ⇒ ~25k tok/s. Token volume: M1 = 2 × 6 × 2.097M ≈ 25.2M; M2 ≈ 35M (SSRA generation re-forwards dominate). SSRA ≈ 2.0 h + flat ≈ 0.1 h + setup/download ≈ 0.5 h ⇒ **session ≈ 2.5–3.5 h ≈ 3.3–4.6 EUR @ $1.50/hr (A100) or ≈ 1.1–1.5 EUR @ $0.49/hr (A6000)** — margin to the 10 EUR cap ≥ 2×.
- **Pod-start micro-benchmark (analog of the early cost gate):** before the grid, measure actual forward tok/s per model at N=1,024 and N=8,192; recompute projected session wall. If projected wall > **4.0 h hard cap** → apply the de-scope ladder BEFORE continuing, in order: **D1** drop N=32,768 → **D2** drop N=16,384 → **D3** halve the M1 token budget (2^21 → 2^20 per cell) → **D4** needle trials 20 → 10. Ladder exhausted and still over → STOP, report, Daniel decides.
- **Session discipline:** supervised end-to-end (Daniel present; short run ⇒ R2(c) redesign explicitly not activated, per D-log). AP-17 secret + `gsutil ls` sanity gate before any billable work; AP-23 strict self-terminate at session end (PID-1 env sourcing, Phase 3 lesson); terminate-not-stop (AP-18). ECB fix: carry 1.1430 unless a credit top-up occurs (then fix on top-up day, AP-12 ledger practice).

## §6 Deliverables + acceptance criteria

1. **Pre-launch (all committed BEFORE deploy; this is what "pre-registered" means):**
   - `experiments/m2-g2lite-eval.yaml`: checkpoint URIs + blob sha256 (from V2) + expected sizes; E offsets; grid; per-cell window counts; needle suite params + seed; batch-size table from §5 projection; precision; output paths. (AP-21: names `m2-g2lite-flat`, `m2-g2lite-ssra`; GCS artifact dir `gs://ssra-poc-ew3/m2/g2lite/`; nothing ever overwritten.)
   - `scripts/needle_gen.py` + tests green; committed needle manifest + sha256.
   - `scripts/g2lite_eval.py` + a CPU smoke test (tiny model, N ∈ {64, 128}) proving M0/M1/M2 code paths run and the anchor logic works.
   - Report §V (V1–V5) filled; §5 projection table in the report.
   - runs.md rows appended at launch.
2. **Post-run:** `results/M2-g2lite.md` complete: protocol echo, V-results, M0 anchor, M1 table + per-position buckets, M2 grids, O1–O7 applied mechanically, plots (ppl vs N log-x per model; needle heatmaps; per-position bucket curves), deviations (explicit or "none"), ledger rows + console total per the ≥2 h rule. Raw JSON/CSV committed + mirrored to `gs://ssra-poc-ew3/m2/g2lite/`.
3. **Acceptance:** M0 within tolerance; every grid cell either measured once or covered by an explicit de-scope/STOP record; zero prohibited operations; zero silent deviations; zero model-code diffs; cap ≤ 10 EUR respected; report contains no architecture conclusions.

## §7 AP checklist (this assignment)

| AP | application here |
|---|---|
| AP-12 | scoped cap **10 EUR** (inside it, the ≤ 25 EUR single-run gate is trivially met); cumulative M2 after session [ODHAD] ≈ 75–77 EUR ≪ 50 % threshold |
| AP-16 | bf16 autocast + fp32 accumulation — identical to training eval |
| AP-17 | `GCP_SA_KEY_B64` secret; `gsutil ls gs://ssra-poc-ew3` gate before billable work |
| AP-18 | launch checklist; terminate-not-stop; §5 GPU selection rule is the scoped ladder for this assignment |
| AP-19 | step 0 Community capture attempt, verbatim record (comparison remains suspended) |
| AP-21 | run names + GCS paths above; append-only logs/ledger; no overwrites |
| AP-22 | ×1.20 on every projection; ≤ 95 % card VRAM gate (pro-rata interpretation, veto) |
| AP-23 | strict self-terminate at session end; PID-1 env sourcing |
| AP-24 | not applicable (no training; supervised session) — NaN/inf in any eval forward is still a STOP+report |

## §8 Anti-goals

No training, no tuning, no third iteration (formulation B is exhausted by this measurement). No architecture conclusions (spec §16). No docs/paper edits by CC (report conflicts instead). No decode-path usage. No baselines (b)/(c). No multi-needle. No positional-scaling tricks for either model. No cherry-picking, no per-cell retries, no result-dependent grid changes outside the pre-registered ladder. No spend before the pre-launch deliverables are committed. No prices, throughputs, or VRAM numbers from memory — projection + console + measurement only.
