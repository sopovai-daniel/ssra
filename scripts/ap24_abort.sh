#!/usr/bin/env bash
# AP-24 action chain (docs/cc/M2-phase3b-retune.md §4; D-log 2026-07-15).
# Invoked by scripts/train.py AFTER the trigger fired (val_loss > running
# best + 2.0 nats on >= 6 consecutive val evals) and AFTER the checkpoint
# (latest.pt + step-<N>.pt) was written and mirrored to GCS. Executes the
# rest of the enumerated action without further confirmation:
#   runs.md mark -> commit+push of results collected so far -> AP-23 strict
#   self-terminate (uploads verified by listing -> push confirmed ->
#   explicit completion signal -> runpodctl remove pod), sourcing
#   RUNPOD_POD_ID and RUNPOD_API_KEY from /proc/1/environ in the terminate
#   step itself (Phase 3 lesson, results/M2-core-pair.md §0.4).
# Never self-terminates with unverified uploads or unpushed results — on
# any precondition failure it exits and the AP-18 manual terminate window
# becomes the primary path. Daniel decides resume-from-checkpoint vs close
# afterwards (AP-11 reversibility). Never deletes or overwrites any GCS
# object. No secret values are ever printed (AP-17).
set -u

RUN="$1"; STEP="$2"; GCS_DIR="$3"; LOG_PATH="$4"
cd "$(dirname "$0")/.."

echo "[ap24] ABORTED-instability: run=${RUN} completed_steps=${STEP}"

# 1) Mark the run in results/runs.md (append-only ledger note; AP-21 — no
#    existing row or note is rewritten).
printf '\nLedger note (AP-24, %s): run %s ABORTED-instability at step %s — AP-24 trigger (val_loss > running best + 2.0 nats on >= 6 consecutive val evals, docs/cc/M2-phase3b-retune.md §4). Checkpoints: %s/latest.pt + %s/step-%s.pt; log logs/%s.log. Resume-from-checkpoint vs close is Daniel'"'"'s decision (AP-11).\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$RUN" "$STEP" "$GCS_DIR" "$GCS_DIR" "$STEP" "$RUN" >> results/runs.md

# 2) AP-23 precondition: GCS uploads verified by listing.
if ! gcloud storage ls "${GCS_DIR}/latest.pt" "${GCS_DIR}/step-${STEP}.pt"; then
  echo "[ap24] GCS listing verification FAILED — NOT self-terminating (AP-23)."
  echo "[ap24] AP-18 manual terminate window is now the primary path."
  git add results/runs.md "$LOG_PATH" \
    && git commit -m "AP-24: ${RUN} ABORTED-instability at step ${STEP} (GCS verify FAILED)" \
    && git push origin main
  exit 1
fi

# 3) AP-23 precondition: results collected so far committed AND pushed.
git add results/runs.md "$LOG_PATH"
git commit -m "AP-24: ${RUN} ABORTED-instability at step ${STEP}

Enumerated in-flight instability stop (docs/cc/M2-phase3b-retune.md §4).
Checkpoints latest.pt + step-${STEP}.pt mirrored to ${GCS_DIR} and
verified by listing. Resume-vs-close is Daniel's decision (AP-11)."
if ! git push origin main; then
  echo "[ap24] git push FAILED — NOT self-terminating (AP-23: never with unpushed results)."
  echo "[ap24] AP-18 manual terminate window is now the primary path."
  exit 1
fi

# 4) Explicit completion signal (AP-23 sequence step 3).
echo "[ap24] COMPLETE: ${RUN} aborted at step ${STEP}; uploads verified, results pushed."
echo "[ap24] AP-23 self-terminate now (pod id + API key sourced from /proc/1/environ)."

# 5) AP-23 self-terminate. Both values sourced from PID-1 env in the
#    terminate step itself — SSH session env lacks them (known RunPod
#    behavior, Phase 3 first-invocation failure).
POD_ID="$(tr '\0' '\n' </proc/1/environ 2>/dev/null | sed -n 's/^RUNPOD_POD_ID=//p')"
API_KEY="$(tr '\0' '\n' </proc/1/environ 2>/dev/null | sed -n 's/^RUNPOD_API_KEY=//p')"
if [ -z "$POD_ID" ] || [ -z "$API_KEY" ]; then
  echo "[ap24] RUNPOD_POD_ID / RUNPOD_API_KEY not found in /proc/1/environ —"
  echo "[ap24] AP-18 manual terminate window is now the primary path."
  exit 1
fi
RUNPOD_API_KEY="$API_KEY" runpodctl remove pod "$POD_ID"
