# CC Assignment: M2 Phase 3b — core pair retune @ lr 6e-4 (single permitted iteration)

**Version:** v1 (2026-07-15) · **Author:** Claude (Fable 5); decisions Daniel (D-log 2026-07-15, second entry)
**Authority chain:** `docs/spec.md` v1.2 > `docs/cc/M2-assignment.md` v1.1 > this doc.
Launch mechanics per `docs/cc/M2-runpod-launch.md` (AP-17/18/19/23) as amended
by later D-log rules (billing read-out 2026-07-14; AP-19 amendment 2026-07-14).
**Predecessor protocol:** `docs/cc/M2-phase3-core-pair.md` v1 — everything
below inherits it verbatim unless explicitly changed here.

## §1 Goal and scope

Re-run the S2 850M core pair with **lr 6e-4** (the pre-declared sweep
runner-up, `results/M2-sweep.md`) to deliver fresh G1 inputs. This is the
**single permitted retune iteration** per `docs/03-poc-plan.md`: on a fail
signal, run the enumerated diagnostics (§6), then STOP — no autonomous
redesign, no further tuning, no third pair. Fallbacks (b)/(c) are Daniel's
decisions.

**Single-variable isolation (binding):** seed **1337 held fixed** —
identical initialization and identical token stream as Phase 3. The only
changed training variable is lr 1e-3 → 6e-4. Every other hyperparameter,
the data, the eval protocol, and the step budget stay exactly as Phase 3.
Rationale for the paper: a spike vanishing ⇒ implicates lr; a spike
recurring ⇒ lr-independent optimization problem. Do not weaken this
isolation for any reason.

## §2 Configs (AP-8, AP-21)

- Generate from the Phase 3 template exactly as predecessor §0.1: the pair
  differs from each other in the same 4 lines (`arch`, `run_name`,
  `training.ckpt_dir`, `training.gcs_ckpt_dir`); differs from the Phase 3
  configs in **lr and names only** (verify with diff, record in report).
- lr **6e-4**; `lr_min_frac` 0.1 (floor 6e-5); `warmup_steps` 778
  unchanged; steps 51,880; batch 16; seq 1024; seed 1337; weight_decay
  0.01; `val_every` 200; `val_batches` 8; `log_every` 25; `ckpt_every`
  1000.
- **Names (AP-21):** `m2-core-flat-s2-850m-lr6e4`,
  `m2-core-ssra-s2-850m-lr6e4`; log filenames and GCS ckpt dirs derived
  from them. Committed logs and GCS objects are never overwritten.
- sha256 hard-gate values injected programmatically from
  `results/M2-data-900m-manifest.json`, re-verified by extraction (4/4) in
  both files.
- Dry-run both configs: params must reproduce 84,301,440 / 84,647,040;
  total tokens 850,001,920. Commit configs BEFORE launch.

## §3 Instrumentation additions (observability only — training semantics unchanged)

Two harness deltas, both mandatory, both outside spec-governed math:

1. **grad_norm logging:** `clip_grad_norm_` already computes and returns
   the global pre-clip norm; capture the return value and add
   `"grad_norm"` to the per-`log_every` JSONL train record. No change to
   clipping behavior, loss, optimizer, LR, or data paths.
2. **Step-tagged checkpoint retention:** after the existing `latest.pt`
   write+mirror, additionally upload the same local file to
   `<gcs_ckpt_dir>/step-<N>.pt` (N = completed steps). Checkpoint blob
   format UNCHANGED; local disk usage unchanged (no local per-step
   copies). ~51 objects ≈ 50 GiB per arm in GCS, temporary — deletion
   after post-run analysis is **Daniel's decision**; never delete
   anything yourself.
   - Pravidlo W item: verify current GCS Standard storage pricing for the
     bucket's region on the day you write the code; record ≈ EUR/month for
     ~100 GiB (both arms) in the report. RunPod egress $0 (verified
     2026-07-12); GCS ingress free — re-verify only if anything looks off.

**Verification gate for §3 (report an AP-20-style statement):**
(i) full §14 suite green (64 pass / 1 known `test_loglinear_integration`
allowed); (ii) AP-11 kill+resume unit test green (blob format untouched);
(iii) light frozen-reference check: a 60-step CPU smoke at fixed seed
produces **bit-identical train losses** before vs after the
instrumentation commit; (iv) the report states the exact diff scope
(files + line counts). If any of (i)–(iii) fails: STOP and surface — do
not work around.

## §4 AP-24 — first active deployment (binding, symmetric for both arms)

- **Trigger:** current val_loss > (running best val_loss of this run +
  **2.0 nats**) on **≥ 6 consecutive val evaluations** (`val_every` 200 ⇒
  1,000 steps sustained, no recovery in between).
- **Action (execute without further confirmation):** write checkpoint
  (latest + step-tagged) → GCS upload → mark run `ABORTED-instability` in
  the log summary and `results/runs.md` → commit+push results collected so
  far → AP-23 strict self-terminate sequence, sourcing `RUNPOD_POD_ID` and
  `RUNPOD_API_KEY` from `/proc/1/environ` **in the terminate command
  itself** (Phase 3 lesson, core-pair report §0.4). Daniel decides
  resume-from-checkpoint vs close afterwards (AP-11 reversibility).
- **Placement [proposal, veto]:** implement in-loop next to the existing
  Phase 3 §3 NaN/inf divergence trigger (minimal surface; val values
  already in scope). The NaN/inf trigger remains in force and takes
  precedence (immediate abort). Covered by the §3 verification gate
  (same commit, same checks).
- Not retroactive; applies to this and subsequent runs per D-log
  2026-07-15.

## §5 Launch, gates, costs

- Bootstrap per launch doc: AP-17 sanity gate before any paid work; AP-19
  step 0 (Community price capture attempt, verbatim record, no backfill);
  AP-23 capability check; thread limits = cgroup quota; pytest on pod
  (64/1 known allowed); repo via git bundle at the pre-launch commit.
- Order: **flat lr6e4 first** (~2.0–2.2 h [ODHAD]), then **SSRA lr6e4**.
- **Early cost gate on the SSRA arm:** `scripts/cost_gate.py`, window
  [1000, 1500], **cap 30 EUR scoped to this single SSRA run** (D-log
  2026-07-15); recompute break-even at the booked rate on deploy day
  (Pravidlo W); anchor sanity vs 12,335 tok/s informative, non-gating.
  Step-tagged uploads add wall/cost outside pure-train throughput
  (~51 × O(10 s) [ODHAD]) — the gate measures pure train, the cap margin
  covers the overhead; if the projection lands within 1 EUR of the cap,
  surface to Daniel before continuing.
- EUR conversion: carry ECB 1.1430 unless credit is topped up (then fix
  the new ECB reference rate per AP-12 practice).
- Credit headroom ≥ $40 confirmed by Daniel at deploy (handoff item).
- Terminate window projection posted pre-deploy; refined ETA after the
  SSRA cost gate. Pod terminate (never stop) immediately after the AP-23
  sequence completes.

## §6 Deliverables & G1 inputs

- Report `results/M2-core-pair-lr6e4.md` mirroring the Phase 3 report
  structure (§0 prep, §i pre-flight, §ii env+pytest, §iii run table, §iv
  G1 inputs, §v plots, §vi P-C incl. `p1_attn_entropy`, §vii ledger,
  §viii M3 handoff, §ix deviations, §x open questions) + the §3
  verification-gate results + AP-20-style statement covering all three
  harness deltas (grad_norm, step-tagged mirror, AP-24 trigger).
- `results/runs.md` rows (AP-21); loss-curve plots; logs committed;
  final_eval on `val-eval-2M`, byte-identical protocol (1,953 windows /
  1,999,872 tok / 127 dropped expected).
- **On a G1-fail signal (spike, AP-24 stop, or out-of-band result):** run
  T2–T4 from `scripts/diagnostics/` on the step-tagged checkpoints
  bracketing the event + the grad_norm timeline around it; deliver
  `results/M2-spike-diagnostics-lr6e4.md`; then STOP. No retry, no
  re-tuning.
- No quality or architecture conclusions anywhere (spec §16); the G1
  verdict is Daniel's.

## §7 Anti-goals

No other hyperparameter changes; no seed change; no model code changes;
no spec/test edits; no silent contingency flags; no retry beyond this
single pair; no deletion of any GCS object; no conclusions from loss
values beyond the G1 metric; ambiguity ⇒ STOP and surface to Daniel.
