#!/bin/bash
# One and only one canonical LIBERO seed (400 episodes).  Seed 0/1/2 jobs share
# the consolidation dependency and therefore run in parallel without duplicate
# seed-0 work.
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

echo "formal stage=${FORMAL_STAGE:?} job=${SLURM_JOB_ID} plan=${FORMAL_PLAN:?}"
python3 scripts/r0_e2e_formal_chain.py run-stage
