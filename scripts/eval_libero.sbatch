#!/bin/bash
# Reproducible real-LIBERO evaluation for checkpoint/inference selection.
# Required: CKPT, OUTDIR. Optional: NTASKS, NEPS, SEEDS, GRIPPER_DWELL,
# DECODER_SAMPLES, DURATION_NORMALIZE_SEGMENTS (0/1). The output path must be
# new; stale episodes are never reused.
#
#   CKPT=/abs/ckpt.pt OUTDIR=/abs/results sbatch scripts/eval_libero.sbatch
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
OUTDIR="${OUTDIR:?set OUTDIR to a new absolute result directory}"
NTASKS="${NTASKS:-10}"
NEPS="${NEPS:-10}"
SEEDS="${SEEDS:-0,1,2}"
GRIPPER_DWELL="${GRIPPER_DWELL:-1}"
DECODER_SAMPLES="${DECODER_SAMPLES:-1}"
DURATION_NORMALIZE_SEGMENTS="${DURATION_NORMALIZE_SEGMENTS:-0}"

case "$DURATION_NORMALIZE_SEGMENTS" in
  0) DURATION_NORMALIZE_ARGS=() ;;
  1) DURATION_NORMALIZE_ARGS=(--duration-normalize-segments) ;;
  *)
    echo "DURATION_NORMALIZE_SEGMENTS must be 0 or 1, got: $DURATION_NORMALIZE_SEGMENTS"
    exit 1
    ;;
esac

test -s "$CKPT" || { echo "missing checkpoint: $CKPT"; exit 1; }
test ! -e "$OUTDIR/results.json" || {
  echo "refusing to overwrite existing results: $OUTDIR/results.json"
  exit 1
}
mkdir -p "$OUTDIR"

export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export HF_HOME=/lustre/fsw/portfolios/edgeai/users/chrislin/datasets/loom/hf_cache
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export TRITON_CACHE_DIR="$PWD/.triton_cache"

echo "checkpoint=$CKPT"
echo "protocol=tasks:$NTASKS episodes:$NEPS seeds:$SEEDS max_steps:512"
echo "inference=gripper_dwell:$GRIPPER_DWELL decoder_samples:$DECODER_SAMPLES duration_normalize_segments:$DURATION_NORMALIZE_SEGMENTS"

"$PY" -m loom.eval --bench libero --backend libero \
  --ckpt "$CKPT" --require-real --op-stats --no-resume \
  --n-tasks "$NTASKS" --episodes-per-task "$NEPS" --seeds "$SEEDS" \
  --workers 8 --gripper-dwell "$GRIPPER_DWELL" \
  --decoder-samples "$DECODER_SAMPLES" \
  "${DURATION_NORMALIZE_ARGS[@]}" \
  --out "$OUTDIR/results.json" --md "$OUTDIR/table.md"

"$PY" - "$OUTDIR/results.json" "$DURATION_NORMALIZE_SEGMENTS" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
p = d.get("meta", {}).get("policy") or {}
expected_duration_normalized = bool(int(sys.argv[2]))
assert p.get("is_stub") is False, "stub policy result is invalid"
assert p.get("ckpt_global_step") is not None, "checkpoint provenance is missing"
assert d.get("summary", {}).get("n_errors") == 0, "evaluation contains errors"
assert p.get("duration_normalize_segments") is expected_duration_normalized, (
    "duration-normalization provenance does not match the requested A/B arm"
)
identity_kw = d.get("meta", {}).get("eval_identity", {}).get("policy_kw", {})
assert bool(identity_kw.get("duration_normalize_segments", False)) is expected_duration_normalized, (
    "duration-normalization eval identity does not match the requested A/B arm"
)
print(json.dumps({
    "avg": d["summary"]["avg"],
    "n": d["summary"]["n_episodes"],
    "checkpoint_step": p["ckpt_global_step"],
    "gripper_dwell": p.get("gripper_dwell"),
    "decoder_samples": p.get("decoder_samples"),
    "duration_normalize_segments": p.get("duration_normalize_segments"),
}, indent=2))
PY
