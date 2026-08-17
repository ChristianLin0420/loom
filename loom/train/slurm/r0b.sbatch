#!/bin/bash
# LOOM R0-B — RoboTwin 2.0 clean from scratch. THE DECISION GATE.
# 16 GPUs = 2 node(s) x 8 A100-SXM4-80GB. One link = 4 h (hard cluster cap).
#
# Chain links, do not self-requeue:
#     bash scripts/submit.sh r0b 7
# Stop:  touch runs/r0b/STOP        # NEVER scancel
#
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
# batch_singlenode carries QOS 128_nodes_per_user_1_node_per_job (max 8 GPUs,
# 1970G per job). Listing it here made sbatch reject every multi-node stage
# outright with QOSMaxMemoryPerJob -- not fall through to a partition that fits.
# Do not add it back to a --nodes>1 job.
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

# No container, no enroot, no module load on this cluster: a plain venv.
# It must hold torch 2.6.0+cu124 -- the node driver is CUDA 12.2, and a +cu13x
# wheel imports fine, reports cuda.is_available()==False, and trains on CPU while
# holding 8 A100s. loom.train.loop raises rather than letting that happen.
source .venv/bin/activate
# WANDB_API_KEY / WANDB_PROJECT / HF_TOKEN. Gitignored; never echoed.
[ -f .env.local ] && source .env.local

CONFIG="configs/r0b.yaml"
RUN_NAME="${LOOM_RUN_NAME:-r0b}"
RUN_DIR="runs/${RUN_NAME}"
mkdir -p "$RUN_DIR" logs

# Stable W&B id: written once, re-exported by every link and every requeue, so
# N offline jobs merge server-side into ONE run.
if [ ! -f "$RUN_DIR/wandb_id" ]; then
  python3 -c "import uuid;print(uuid.uuid4().hex[:16])" > "$RUN_DIR/wandb_id"
fi
export WANDB_RUN_ID="$(cat "$RUN_DIR/wandb_id")"
export WANDB_RESUME=allow
export WANDB_DIR="$RUN_DIR"          # wandb appends wandb/ itself
# Compute nodes DO have an outbound route -- re-measured on batch-block5-04009:
# api.wandb.ai answered in 0.22 s and a real `wandb.init` completed in 1.8 s and
# synced. (An older note in CLAUDE.md said otherwise; it was stale.) So log
# ONLINE and watch the run live. `LOOM_WANDB_MODE` overrides for a node that
# turns out to be walled off.
# wandb_util.init bounds the init and falls back to offline on any failure, so a
# network blip costs live logging, never a 4 h link. If that happens the link
# still lands on the same run id -- recover it with:
#     bash scripts/wandb_sync.sh $RUN_NAME        # from a login node
export WANDB_MODE="${LOOM_WANDB_MODE:-online}"

# Link budget. The separate 420 s safety margin lives in the loop and must equal
# the --signal=USR1@420 above.
export LOOM_TIME_BUDGET_S=$(( 4*3600 - 600 ))
export LOOM_RESTART_COUNT="${SLURM_RESTART_COUNT:-0}"
export OMP_NUM_THREADS=8
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUNBUFFERED=1
# Team A's loader throughput gate needs a real training rate. Measure it/s from
# the first link's log and set this; the 5.0 default is a placeholder.
export LOOM_TRAIN_STEP_HZ="${LOOM_TRAIN_STEP_HZ:-5.0}"
# Data roots, HF cache, Triton cache. Shared with logs/*_smoke.sh so a green
# smoke actually gates the real launch; raises if the feature cache is missing.
# RoboTwin, not LIBERO. env.sh defaults to the LIBERO roots, and without these
# two lines r0b would load the LIBERO cache and report the number as RoboTwin --
# a wrong result that looks entirely healthy. Set before the source; env.sh uses
# :- defaults so an explicit value wins, and it raises if the cache is absent.
export LOOM_DATA_ROOT="${LOOM_DATA_ROOT:-/lustre/fsw/portfolios/edgeai/users/chrislin/datasets/loom/robotwin}"
export LOOM_CACHE_DIR="${LOOM_CACHE_DIR:-/lustre/fsw/portfolios/edgeai/users/chrislin/datasets/loom/robotwin_cache}"
source scripts/env.sh

# Echo FIRST: a link that dies before any output is indistinguishable from a
# hung node, and that costs a whole 4 h chain slot to diagnose.
echo "link start: job=${SLURM_JOB_ID} restart=${LOOM_RESTART_COUNT} node=$(hostname) config=${CONFIG} run=${RUN_NAME}"
# DataLoader workers pass tensors through /dev/shm, and an overflow surfaces as
# "DataLoader worker exited unexpectedly" or a bare bus error -- which reads like
# a code bug. It is 64 MiB on a LOGIN node but 1008 GiB here, so measure it on
# the node rather than inferring it from either value.
df -h /dev/shm | tail -n1 | awk '{printf "shm: size=%s used=%s avail=%s\n", $2, $3, $4}'
ldd --version | head -n1     # glibc differs between login and compute nodes

# scontrol blocks indefinitely if slurmctld is unreachable from the node, and a
# single-node job does not need it at all.
if [ "${SLURM_NNODES:-1}" -gt 1 ]; then
  MASTER_ADDR="$(timeout 30 scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
fi
export MASTER_ADDR="${MASTER_ADDR:-$(hostname)}"
export MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))

# RANK/WORLD_SIZE/LOCAL_RANK come from SLURM, not torchrun. Without this every
# task reads rank 0: eight processes train the same shard, write the same
# checkpoint file, and the loss curve looks perfectly fine.
srun --kill-on-bad-exit=1 bash -c '
  export RANK="$SLURM_PROCID" WORLD_SIZE="$SLURM_NTASKS" LOCAL_RANK="$SLURM_LOCALID"
  exec python3 -m loom.train.loop --config "'"$CONFIG"'" --run_dir "'"$RUN_DIR"'" '"${LOOM_EXTRA_ARGS:-}"'
'
