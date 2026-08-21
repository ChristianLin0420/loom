#!/bin/bash
# Collect one immutable five-trial LIBERO fold for terminal-outcome proposal GRPO.
# Required: CKPT, OUTDIR, SPLIT=train0..train5|validation. Reissuing an identical command
# resumes only after the manifest and every committed tensor sidecar validate.
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
CKPT="${CKPT:?set CKPT to the absolute consolidated step-49666 checkpoint}"
OUTDIR="${OUTDIR:?set OUTDIR to an absolute recovery collection directory}"
SPLIT="${SPLIT:?set SPLIT to train0..train5 or validation}"
WORKERS="${WORKERS:-8}"

case "$SPLIT" in
  train0|train1|train2|train3|train4|train5) EXPECTED_GROUPS=200 ;;
  validation) EXPECTED_GROUPS=400 ;;
  *) echo "SPLIT must be train0..train5 or validation, got: $SPLIT" >&2; exit 2 ;;
esac
test -x "$PY" || { echo "missing LIBERO interpreter: $PY" >&2; exit 1; }
test -s "$CKPT" || { echo "missing checkpoint: $CKPT" >&2; exit 1; }
mkdir -p "$OUTDIR" logs

source scripts/env.sh
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

echo "checkpoint=$CKPT"
echo "split=$SPLIT groups=$EXPECTED_GROUPS arms=8 terminal_reward_only=1"
echo "output=$OUTDIR workers=$WORKERS"

"$PY" scripts/outcome_recovery.py \
  --checkpoint "$CKPT" --out-dir "$OUTDIR" \
  --split "$SPLIT" --workers "$WORKERS"

"$PY" - "$OUTDIR/manifest.json" "$SPLIT" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1]))
summary = manifest["summary"]
assert manifest["split"] == sys.argv[2]
expected_groups = 400 if sys.argv[2] == "validation" else 200
assert summary["status"] == "COMPLETE" and summary["complete"] is True
assert summary["n_groups"] == summary["n_expected_groups"] == expected_groups
assert summary["n_trajectories"] == summary["n_expected_trajectories"] == 8 * expected_groups
assert len(manifest["groups"]) == expected_groups
print(json.dumps(summary, indent=2, sort_keys=True))
PY
