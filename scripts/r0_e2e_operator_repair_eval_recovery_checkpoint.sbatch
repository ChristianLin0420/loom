#!/bin/bash
# Verify-only adoption of the existing, already-consolidated step-32k checkpoint.
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=batch_singlenode,polar4,polar3,polar,grizzly
#SBATCH --qos=normal
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source .venv/bin/activate
[ -f .env.local ] && source .env.local
source scripts/env.sh
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

echo "operator-repair recovery stage=${OPERATOR_RECOVERY_STAGE:?} job=${SLURM_JOB_ID} plan=${OPERATOR_RECOVERY_PLAN:?}"
python3 scripts/r0_e2e_operator_repair_eval_recovery.py run-stage
