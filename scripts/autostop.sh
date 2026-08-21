#!/usr/bin/env bash
# Stop a run the moment convergence is PROVABLE, not whenever someone looks.
#
#   bash scripts/autostop.sh <run_name> [min_step] [interval_s]
#
# Fires `touch runs/<run>/STOP` on scripts/convergence.py's CONVERGED (exit 0)
# after min_step, or immediately on a declared required stage gate
# failure (exit 4). It never cancels scheduler jobs.
#
# It deliberately does NOT stop on CONVERGED_DEGENERATE (exit 3). A run pinned
# at a floor has also "stopped changing", and this project has produced several;
# 6 of 9 such collapses were later escaped, and the best run on record collapsed
# at step 1736 and recovered to become the best. Killing on a degenerate plateau
# would have killed it. So that case is logged loudly and left alone.
#
# `min_step` guards against an early plateau during warmup being mistaken for
# convergence -- LR warmup alone is 2000 steps.
set -uo pipefail
RUN="${1:?usage: autostop.sh <run_name> [min_step] [interval_s]}"
MIN="${2:-20000}"
INT="${3:-300}"
PY=.venv/bin/python
RUN_DIR="${LOOM_RUN_ROOT:-runs}/${RUN}"
while true; do
  if [ -f "${RUN_DIR}/STOP" ]; then echo "$(date -Is) STOP already present; exiting"; break; fi
  STEP=$(awk '{print $2}' "${RUN_DIR}/HEARTBEAT" 2>/dev/null || echo 0)
  STEP=${STEP:-0}
  # Always inspect declared stage gates, even before MIN. Liveness owns its
  # exact configured row window; MIN remains only the legacy convergence guard.
  OUT=$($PY scripts/convergence.py "${RUN_DIR}" 2>&1); RC=$?
  if [ "$RC" -eq 4 ]; then
    echo "$(date -Is) required stage gate FAILED at step ${STEP} -- writing STOP"
    echo "$OUT"
    touch "${RUN_DIR}/STOP"
    break
  fi
  # A partially-written config/metrics line is a retryable read (rc=2), not a
  # method verdict. Required gate schema/data failures use rc=4 above.
  if ! squeue -u "$USER" -h -n "loom_${RUN}" -o "%T" 2>/dev/null | grep -qE 'RUNNING|PENDING'; then
    echo "$(date -Is) run finished on its own; exiting"; break
  fi
  if [ "$STEP" -ge "$MIN" ] 2>/dev/null; then
    case "$RC" in
      0) echo "$(date -Is) CONVERGED at step ${STEP} -- writing STOP"; echo "$OUT"
         touch "${RUN_DIR}/STOP"; break ;;
      3) echo "$(date -Is) step ${STEP}: CONVERGED_DEGENERATE (plateaued ON a floor) -- NOT stopping"
         echo "$OUT" | tail -6 ;;
      *) : ;;   # NOT_CONVERGED / TOO_EARLY -- keep going, quietly
    esac
  fi
  SLEEP_FOR="$INT"
  if [ "$RC" -eq 1 ] && [ "$INT" -gt 5 ] && \
     { [[ "$OUT" == *"LIVENESS: PENDING"* ]] || \
       [[ "$OUT" == *"PHASE_GATE: PENDING"* ]]; }; then
    SLEEP_FOR=5
  fi
  sleep "$SLEEP_FOR"
done
