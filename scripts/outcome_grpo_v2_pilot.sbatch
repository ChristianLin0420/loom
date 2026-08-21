#!/bin/bash
# Outcome-GRPO v2 pilot scaffold. The checked-in config is intentionally
# ineligible and the Python entry point fails before creating RUN_DIR.
# No follow-on, evaluation, promotion, or candidate publication is permitted.
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=polar4,polar3,polar,grizzly,batch_singlenode
#SBATCH --qos=normal
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --signal=USR1@420
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

CONFIG="${CONFIG:-$PWD/configs/r0a_outcome_grpo_v2_pilot.yaml}"
: "${RUN_DIR:?set RUN_DIR to a fresh unique v2 pilot directory}"
STOP_AT="${STOP_AT:-50466}"

test -s "$CONFIG" || { echo "missing config: $CONFIG" >&2; exit 1; }
test -x .venv/bin/python || { echo "missing interpreter: .venv/bin/python" >&2; exit 1; }

# Intentionally do not mkdir RUN_DIR here. The readiness gate must reject the
# unfrozen scaffold before any run-directory mutation.
source .venv/bin/activate
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export CONFIG RUN_DIR STOP_AT

echo "status=INELIGIBLE_UNFROZEN_PILOT candidate_emission=forbidden"
exec .venv/bin/python scripts/train_outcome_grpo_v2.py \
  --config "$CONFIG" --run-dir "$RUN_DIR" --stop-at "$STOP_AT"
