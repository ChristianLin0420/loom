#!/usr/bin/env bash
# Requeue a link whose heartbeat has gone stale, and shout if delta_op flatlines.
#
# The sbatch's `link start:` echo distinguishes hung-from-dead only before the
# first blocking call. This covers a node that wedges later: the job is RUNNING,
# burning a 4 h chain slot, and producing nothing. `scontrol requeue`, NEVER
# `scancel` -- the last checkpoint is durable, so a requeue costs at most the
# interval since it, while a scancel breaks the chain.
#
#   bash scripts/watchdog.sh <run_name> [max_age_s]
#   */10 * * * * cd <repo> && bash scripts/watchdog.sh r0a >> logs/watchdog.log 2>&1
set -uo pipefail

RUN_NAME="${1:?usage: watchdog.sh <run_name> [max_age_s]}"
MAX_AGE="${2:-1800}"                    # 30 min: well past any checkpoint or eval
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

RUN_DIR="runs/${RUN_NAME}"
HB="${RUN_DIR}/HEARTBEAT"
now() { date -u +%FT%TZ; }

JOB=$(squeue -h -n "loom_${RUN_NAME}" -t RUNNING -o "%i" | head -n1)
if [ -z "$JOB" ]; then
  echo "$(now) no running link for ${RUN_NAME}"
  exit 0
fi

# A job that has not started training yet has no heartbeat; fall back to runtime.
if [ ! -f "$HB" ]; then
  echo "$(now) job=${JOB} no heartbeat yet (runtime $(squeue -h -j "$JOB" -o '%M'))"
  exit 0
fi

AGE=$(( $(date +%s) - $(stat -c %Y "$HB") ))
STEP=$(awk '{print $2}' "$HB")
DELTA=$(awk '{print ($3 == "" ? "nan" : $3)}' "$HB")

if [ "$AGE" -gt "$MAX_AGE" ]; then
  echo "$(now) job=${JOB} heartbeat stale ${AGE}s at step ${STEP} -> requeue"
  scontrol requeue "$JOB"
  exit 0
fi

echo "$(now) job=${JOB} alive, heartbeat ${AGE}s old at step ${STEP} delta_op=${DELTA}"

# delta_op is a build assert, not a metric (PLAN 4.C): if it is flat at ~0 after
# a few thousand steps, the model collapsed to a plain latent policy and A(c) ~ I
# nearly satisfies L_dyn while c carries nothing. Kill and flip
# losses.dyn.negatives rather than burning the remaining links.
awk -v s="$STEP" -v d="$DELTA" 'BEGIN{
  if (d == "nan") exit 0
  if (s+0 > 3000 && d+0 < 0.01)
    printf "  WARNING delta_op=%s at step %s -- L_dyn may have collapsed; check losses.dyn.negatives\n", d, s
}'
