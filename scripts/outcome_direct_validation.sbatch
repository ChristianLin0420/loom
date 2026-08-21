#!/bin/bash
# Paired held-out direct-policy validation for the canonical round-0 descendant.
# Submit only with an afterok dependency on the fail-closed trainer.
#
#   sbatch --dependency=afterok:TRAIN_JOB scripts/outcome_direct_validation.sbatch
#
# This launcher never evaluates official trials 0--9 and never promotes a model.
#
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=polar4,polar3,polar,grizzly,batch_singlenode
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit this launcher from the repository root}"

ROOT="$(pwd -P)"
PY=/lustre/fsw/portfolios/edgeai/users/chrislin/envs/loom-libero/bin/python
CHECKPOINT="$ROOT/runs/r0a_outcome_grpo_round0_s1/ckpt_000050066.pt"
VALIDATION_COLLECTION="$ROOT/runs/outcome_recovery_s49666_validation"
RESULT="$ROOT/runs/outcome_direct_validation_s50066/results.json"
WORKERS="${WORKERS:-8}"

test -x "$PY" || { echo "missing LIBERO interpreter: $PY" >&2; exit 1; }
test -s "$CHECKPOINT" || { echo "missing trust-passing descendant: $CHECKPOINT" >&2; exit 1; }
test -s "$VALIDATION_COLLECTION/manifest.json" || {
  echo "missing validation manifest: $VALIDATION_COLLECTION/manifest.json" >&2
  exit 1
}
case "$WORKERS" in
  ''|*[!0-9]*) echo "WORKERS must be an integer, got: $WORKERS" >&2; exit 2 ;;
esac
test "$WORKERS" -ge 1 && test "$WORKERS" -le 8 || {
  echo "WORKERS must be in [1,8], got: $WORKERS" >&2
  exit 2
}
mkdir -p "$(dirname "$RESULT")" logs

source scripts/env.sh
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

echo "checkpoint=$CHECKPOINT"
echo "validation_collection=$VALIDATION_COLLECTION"
echo "result=$RESULT workers=$WORKERS heldout_trials=40..49 official_trials=untouched"

"$PY" scripts/outcome_direct_validation.py \
  --checkpoint "$CHECKPOINT" \
  --validation-collection "$VALIDATION_COLLECTION" \
  --out "$RESULT" \
  --workers "$WORKERS"

"$PY" - "$RESULT" "$CHECKPOINT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

result_path = Path(sys.argv[1]).resolve()
checkpoint = Path(sys.argv[2]).resolve()
payload = json.loads(result_path.read_text())
summary = payload["summary"]
meta = payload["meta"]
paired = meta["paired_outcomes"]
terminal = meta["direct_validation"]

checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
assert summary["complete"] is True
assert summary["n_episodes"] == 400 and summary["n_errors"] == 0
assert terminal["status"] == "PASS" and terminal["n_conditions"] == 400
assert terminal["n_errors"] == 0
assert terminal["candidate_checkpoint_sha256"] == checkpoint_sha
assert terminal["work_sha256"] == "d28c4df17ae06b4a214c8a994af4dbba6aef6157874d97561f017832b76202c3"
assert paired["aggregate"]["n"] == 400
assert paired["aggregate"]["both"] + paired["aggregate"]["candidate_only"] \
       + paired["aggregate"]["parent_only"] + paired["aggregate"]["neither"] == 400
assert paired["aggregate"]["candidate"] >= paired["aggregate"]["parent_arm0"] + 12
assert paired["aggregate"]["candidate_only"] > paired["aggregate"]["parent_only"]
assert all(
    row["candidate"] >= row["parent_arm0"] - 5
    for row in paired["per_suite"].values()
)
assert meta["eval_identity"]["policy"]["path"] == "estimator->proposal.argmax->decoder"
assert meta["eval_identity"]["policy"]["episodes_per_condition"] == 1
print(json.dumps({
    "status": "PASS",
    "summary": summary,
    "paired_outcomes": paired,
    "checkpoint_sha256": checkpoint_sha,
    "efficacy_gate": {
        "min_absolute_gain_successes": 12,
        "candidate_only_gt_parent_only": True,
        "max_suite_regression_successes": 5,
    },
}, indent=2, sort_keys=True))
PY
