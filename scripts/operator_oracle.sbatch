#!/bin/bash
# Common-noise whole-closed-loop K=16 LIBERO operator ceiling.
# Required: CKPT, EXPECTED_CONFIG_HASH, EXPECTED_GLOBAL_STEP, OUTDIR.
# Optional: NEPS, NTASKS, SUITES, SEEDS, WORKERS, MAX_STEPS.
# Reissuing the identical command and OUTDIR resumes incomplete arms.
#
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=polar4,polar3,polar,grizzly,batch_singlenode
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

PY=/lustre/fsw/portfolios/edgeai/users/chrislin/envs/loom-libero/bin/python
CKPT="${CKPT:?set CKPT to an absolute consolidated checkpoint path}"
EXPECTED_CONFIG_HASH="${EXPECTED_CONFIG_HASH:?set the embedded checkpoint config hash}"
EXPECTED_GLOBAL_STEP="${EXPECTED_GLOBAL_STEP:?set the checkpoint global step}"
OUTDIR="${OUTDIR:?set OUTDIR to an absolute result directory}"
NEPS="${NEPS:-3}"
NTASKS="${NTASKS:-10}"
SUITES="${SUITES:-libero_spatial,libero_object,libero_goal,libero_10}"
SEEDS="${SEEDS:-0}"
WORKERS="${WORKERS:-8}"
MAX_STEPS="${MAX_STEPS:-512}"

test -s "$CKPT" || { echo "missing checkpoint: $CKPT"; exit 1; }
mkdir -p "$OUTDIR" logs

export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export HF_HOME=/lustre/fsw/portfolios/edgeai/users/chrislin/datasets/loom/hf_cache
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export TRITON_CACHE_DIR="$PWD/.triton_cache"

echo "checkpoint=$CKPT"
echo "identity=config:$EXPECTED_CONFIG_HASH step:$EXPECTED_GLOBAL_STEP"
echo "protocol=tasks:$NTASKS episodes:$NEPS suites:$SUITES seeds:$SEEDS max_steps:$MAX_STEPS"
echo "oracle=arm0_argmax plus15_sampled common_decoder_noise realizability_tau:0.5"

"$PY" scripts/operator_oracle.py \
  --ckpt "$CKPT" \
  --expected-config-hash "$EXPECTED_CONFIG_HASH" \
  --expected-global-step "$EXPECTED_GLOBAL_STEP" \
  --out-dir "$OUTDIR" --workers "$WORKERS" \
  --episodes-per-task "$NEPS" --n-tasks "$NTASKS" \
  --suites "$SUITES" --seeds "$SEEDS" --max-steps "$MAX_STEPS"

"$PY" - "$OUTDIR/summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1]))
assert summary.get("dry_run") is False, "fake oracle output is not a score"
assert summary.get("authoritative_protocol") in (True, False), "missing authority provenance"
assert summary.get("complete") is True, "oracle result is incomplete"
assert summary.get("n_errors") == 0, "oracle contains errors"
print(json.dumps(summary, indent=2))
PY
