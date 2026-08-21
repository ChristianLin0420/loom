#!/bin/bash
# Proposal-only terminal-outcome GRPO. This script never submits follow-on jobs
# and never evaluates or promotes its output.
#
#   RUN_DIR="$PWD/runs/<fresh-unique-run>" sbatch scripts/outcome_grpo.sbatch
#
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=polar4,polar3,polar,grizzly,batch_singlenode
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --signal=USR1@420
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

CONFIG="${CONFIG:-$PWD/configs/r0a_outcome_grpo.yaml}"
: "${RUN_DIR:?set RUN_DIR to a fresh unique outcome-GRPO directory}"
STOP_AT="${STOP_AT:-54466}"

test -s "$CONFIG" || { echo "missing config: $CONFIG" >&2; exit 1; }
test -x .venv/bin/python || { echo "missing training interpreter: .venv/bin/python" >&2; exit 1; }
mkdir -p "$RUN_DIR" logs

source .venv/bin/activate
source scripts/env.sh
if [ -f .env.local ]; then
  set -a
  source .env.local
  set +a
fi
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export WANDB_MODE="${LOOM_WANDB_MODE:-online}"
export LOOM_TIME_BUDGET_S="${LOOM_TIME_BUDGET_S:-13800}"
export LOOM_RESTART_COUNT="${SLURM_RESTART_COUNT:-0}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

if [ "${SLURM_NNODES:-1}" -gt 1 ]; then
  MASTER_ADDR="$(timeout 30 scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
fi
export MASTER_ADDR="${MASTER_ADDR:-$(hostname)}"
export MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
export CONFIG RUN_DIR STOP_AT

echo "config=$CONFIG"
echo "run_dir=$RUN_DIR stop_at=$STOP_AT world=${SLURM_NTASKS:-8}"
echo "objective=terminal_grpo+sparse_q_action_ce+proposal_switch_balance train_modules=proposal"

srun --kill-on-bad-exit=1 bash -c '
  export RANK="$SLURM_PROCID" WORLD_SIZE="$SLURM_NTASKS" LOCAL_RANK="$SLURM_LOCALID"
  exec .venv/bin/python scripts/train_outcome_grpo.py \
    --config "$CONFIG" --run-dir "$RUN_DIR" --stop-at "$STOP_AT"
'
