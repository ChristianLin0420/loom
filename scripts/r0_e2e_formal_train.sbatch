#!/bin/bash
# One immutable link of the fresh dual-code R0 formal lineage.  The Python plan
# decides whether this is initial link 1..3 or the conditional 32k->40k link.
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=polar4,polar3,polar,grizzly
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --signal=USR1@420
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source .venv/bin/activate
[ -f .env.local ] && source .env.local
source scripts/env.sh

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

if [ "${SLURM_NNODES:-1}" -gt 1 ]; then
  MASTER_ADDR="$(timeout 30 scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
fi
export MASTER_ADDR="${MASTER_ADDR:-$(hostname)}"
export MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))

echo "formal stage=${FORMAL_STAGE:?} job=${SLURM_JOB_ID} plan=${FORMAL_PLAN:?}"
python3 scripts/r0_e2e_formal_chain.py run-stage
