#!/usr/bin/env bash
# Chain N links of 4 h each.
#
# Chaining beats self-requeue: a crashed link does not break the chain, and
# `squeue` shows the remaining budget at a glance. All links share one
# --job-name and --dependency=singleton, so SLURM serialises them.
# --requeue is still set inside the sbatch so SLURM preemption also works.
#
#   bash scripts/submit.sh r0a          # n_links from configs/r0a.yaml
#   bash scripts/submit.sh r0a 3
#   bash scripts/submit.sh r0a 3 --set optim.lr=1e-4
#   bash scripts/submit.sh r0a 3 --steps 60000     # schedule horizon, all links
#
# Stop a run:   touch runs/<name>/STOP        # NEVER scancel
set -euo pipefail

STAGE="${1:?usage: submit.sh <r0a|r0b|r1|r2|r3> [n_links] [extra loop args...]}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

SBATCH="loom/train/slurm/${STAGE}.sbatch"
CONFIG="configs/${STAGE}.yaml"
[ -f "$SBATCH" ] || { echo "no sbatch for stage '${STAGE}': $SBATCH" >&2; exit 2; }
[ -f "$CONFIG" ] || { echo "no config for stage '${STAGE}': $CONFIG" >&2; exit 2; }

PY="${REPO}/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"

# n_links defaults to slurm.n_links in the config, which is sized from the run's
# expected wall time and the hard 4 h cap (~8 h -> 3 links, ~1 d -> 7, ~6 d -> 38).
DEFAULT_LINKS="$("$PY" - "$CONFIG" <<'EOF' 2>/dev/null || echo 3
import sys, yaml
print(int((yaml.safe_load(open(sys.argv[1])) or {}).get("slurm", {}).get("n_links", 3)))
EOF
)"

N_LINKS="${2:-$DEFAULT_LINKS}"
if [ "$#" -ge 2 ]; then shift 2; else shift 1; fi

RUN_NAME="${LOOM_RUN_NAME:-$STAGE}"
RUN_DIR="runs/${RUN_NAME}"
mkdir -p "$RUN_DIR" logs

# A leftover STOP from a previous run stops link 1 at step 0 and looks like a
# silent failure.
if [ -f "$RUN_DIR/STOP" ]; then
  echo "removing stale $RUN_DIR/STOP"
  rm -f "$RUN_DIR/STOP"
fi

for _ in $(seq 1 "$N_LINKS"); do
  sbatch --job-name="loom_${RUN_NAME}" \
         --dependency=singleton \
         --export=ALL,LOOM_RUN_NAME="${RUN_NAME}",LOOM_EXTRA_ARGS="$*" \
         "$SBATCH"
done

cat <<EOF

queued ${N_LINKS} links as loom_${RUN_NAME}   (${SBATCH}, ${CONFIG})
  watch    squeue -n loom_${RUN_NAME}
  logs     tail -F logs/loom_${RUN_NAME}_*.out
  liveness cat ${RUN_DIR}/HEARTBEAT      # "<unix_ts> <step> <delta_op>"
  watchdog bash scripts/watchdog.sh ${RUN_NAME}
  wandb    bash scripts/wandb_sync.sh ${RUN_NAME}    # from a LOGIN node
  stop     touch ${RUN_DIR}/STOP                     # never scancel
EOF
