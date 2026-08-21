#!/bin/bash
# Merge/report the exact 1200-episode non-formal diagnostic.
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=batch_singlenode,polar4,polar3,polar,grizzly
#SBATCH --qos=normal
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source .venv/bin/activate
[ -f .env.local ] && source .env.local
export PYTHONUNBUFFERED=1

echo "post-abort diagnostic stage=${DIAGNOSTIC_STAGE:?} job=${SLURM_JOB_ID} plan=${DIAGNOSTIC_PLAN:?}"
python3 scripts/r0_e2e_post_abort_eval.py run-stage
