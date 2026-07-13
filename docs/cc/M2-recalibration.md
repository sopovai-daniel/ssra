# Zadanie pre CC — M2: re-calibration launch (D-log 2026-07-13, GO)

**Version:** v1 (2026-07-13) · **Milestone:** M2, step 3 of D-log 2026-07-12 option (a) · **Mode:** veto-based.

**Authority chain (binding):** `docs/spec.md` v1.2 > `docs/cc/M2-assignment.md` v1.1 (milestone contract) > `docs/cc/M2-runpod-launch.md` (standing launch checklist, AP-17/AP-18 as amended by D-log 2026-07-12) > this doc. Project state: `docs/00` D-log. Behavior rules: `CLAUDE.md`. Closed decisions (D1–D6, Q1–Q5, MD-1…MD-13, AP-1…AP-20) must not be reopened.

**Code = commit `9417399`** (+ housekeeping `87b0c60`); **no code changes on the box.** Oversight review of the read-out optimization passed 6/6 (D-log 2026-07-13).

## 1. Goal

Measure the post-R1/R4 SSRA memory + throughput on the A100 80 GB and produce the verdict inputs for the D-log measurable target: **S2 b16 trains ∧ ≥ ~11.5k tok/s ⇒ projected ≤ 25 EUR @ 850M tokens** (at $1.39/hr A100 PCIe; recompute at the actually booked price). The verdict itself = Daniel, after the report. Budget envelope ~1–2 EUR; AP-12 STOP thresholds apply.

## 2. Pre-flight (blocking, before any paid work)

1. **AP-19 — BOTH console prices, mandatory this time** (was "not captured" on 2026-07-12): record the Secure on-demand AND the Community price for the booked GPU type, from the deploy console, same day; values + retrieval date into the report (Pravidlo W).
2. **Credit check:** the RunPod account is shared with another project — confirm the balance covers the ≥ 1 h on-demand deploy minimum plus the ~2 EUR envelope before deploy.
3. **ECB EUR/USD fixing of the day** recorded (AP-12 ledger is EUR, RunPod bills USD).
4. **HW ladder:** A100 PCIe 80 GB → A100 SXM → H100 PCIe (re-project EUR per AP-12 if escalating). No network volume (artifacts → GCS). AP-17 secret flow exactly as in calibration — bootstrap `/proc/1/environ` fallback, no start-command needed. Blocking sanity gate `gsutil ls gs://ssra-poc-ew3` before any paid run. Thread limits per bootstrap (OMP/MKL = cgroup quota — thread-thrash lesson).
5. **`pytest tests/` on the box once.** The binding correctness referee stays **CPU fp32** (AP-2): if the new A/B equivalence tests exceed atol on CUDA fp32 due to reduction-order effects, record as informative and continue — CPU-green (already certified) is authoritative. Known skip/red: `test_loglinear_integration` (fla × transformers, Phase-4-only), status unchanged.

## 3. Run set (fixed order)

Same probe protocol as Phase 1 calibration: 120 steps × b × 1024, existing `experiments/M2-cal-*.yaml` configs (unchanged, already committed — run discipline satisfied), one ledger row per run in `results/runs.md` (current code commit recorded), peak VRAM (`torch.cuda.max_memory_allocated`) + tok/s per run, log per run.

| # | run | purpose | D5 projection |
|---|---|---|---|
| 1 | flat S1 b16 | env/pod comparability anchor (~free; calib: 300,978 tok/s / 6.35 GiB) | — |
| 2 | SSRA S1 b16 | regression vs calib 9,457 tok/s / 54.67 GiB | 15.40 GiB |
| 3 | **SSRA S2 b16** | **GATE — measurable-target input** | 36.59 GiB |
| 4 | SSRA S1 b32 | scaling point | 30.44 GiB |
| 5 | optional: SSRA S1 b64 | marginal probe, single attempt | 60.51 GiB |
| 6 | optional: SSRA S2 b32 | marginal probe, single attempt — may OOM; **OOM = valid recorded outcome; no retry, no allocator-config changes** | 71.92 GiB |

S2 b64 is dropped (projected 142.6 GiB — no information value; do not attempt). Optional runs #5/#6 only AFTER the S2 b16 gate measurement is safely captured. No flat re-runs beyond #1 (flat code path untouched by the restructure).

## 4. Report — `results/M2-recalibration.md`

- Table measured vs D5 projected per config — state the model error explicitly (this calibrates the analytic model's error bar for Phase 2 planning).
- tok/s per run; EUR projection for S2 @ 850M tokens at BOTH captured prices (Secure and Community).
- Billed console total at terminate (console number authoritative; no backfilled per-run attribution — Pravidlo W).
- Deviations: each explicit, none silent. No quality conclusions from any loss (spec §16).
- End with the explicit signal: **"re-calibration report ready for verdict"**.

## 5. Anti-goals

- No code edits on the box beyond the pinned commit; no new YAML configs; no §13/config changes.
- No second attempts on OOM probes; no PYTORCH_CUDA_ALLOC_CONF or other environment tweaks (comparability with Phase 1 calibration).
- No Phase 2 start; the verdict is Daniel's D-log decision.
- **Terminate (not stop) the pod immediately** after the last run + signal (AP-18).
