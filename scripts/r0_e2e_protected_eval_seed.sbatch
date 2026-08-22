#!/bin/bash
# One exact resumable 400-episode LIBERO seed for a protected-action arm.
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=polar4,polar3,polar,grizzly,batch_singlenode
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source .venv/bin/activate
[ -f .env.local ] && source .env.local
export PYTHONUNBUFFERED=1

echo "protected stage=${OPERATOR_REPAIR_STAGE:?} job=${SLURM_JOB_ID} plan=${OPERATOR_REPAIR_PLAN:?}"
python3 scripts/r0_e2e_protected_chain.py run-stage
