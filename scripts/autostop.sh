#!/usr/bin/env bash
# Stop a run the moment convergence is PROVABLE, not whenever someone looks.
#
#   bash scripts/autostop.sh <run_name> [min_step] [interval_s]
#
# Fires `touch runs/<run>/STOP` only on scripts/convergence.py's CONVERGED
# (exit 0): every primary metric flat within tolerance AND off every known
# degenerate floor.
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
while true; do
  if [ -f "runs/${RUN}/STOP" ]; then echo "$(date -Is) STOP already present; exiting"; break; fi
  if ! squeue -u "$USER" -h -n "loom_${RUN}" -o "%T" 2>/dev/null | grep -qE 'RUNNING|PENDING'; then
    echo "$(date -Is) run finished on its own; exiting"; break
  fi
  STEP=$(awk '{print $2}' "runs/${RUN}/HEARTBEAT" 2>/dev/null || echo 0)
  STEP=${STEP:-0}
  if [ "$STEP" -ge "$MIN" ] 2>/dev/null; then
    OUT=$($PY scripts/convergence.py "runs/${RUN}" 2>&1); RC=$?
    case "$RC" in
      0) echo "$(date -Is) CONVERGED at step ${STEP} -- writing STOP"; echo "$OUT"
         touch "runs/${RUN}/STOP"; break ;;
      3) echo "$(date -Is) step ${STEP}: CONVERGED_DEGENERATE (plateaued ON a floor) -- NOT stopping"
         echo "$OUT" | tail -6 ;;
      *) : ;;   # NOT_CONVERGED / TOO_EARLY -- keep going, quietly
    esac
  fi
  sleep "$INT"
done
