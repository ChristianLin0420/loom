#!/bin/bash
# Offline promotion gate for the action-anchored operator bank.
# Required: CKPT (absolute consolidated checkpoint), OUT (new JSON path).
#
#   CKPT=/abs/ckpt.pt OUT=/abs/bank_gate.json \
#     sbatch scripts/bank_ca_gate.sbatch
#
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=polar4,polar3,polar,grizzly,batch_singlenode
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

CKPT="${CKPT:?set CKPT to an absolute consolidated checkpoint path}"
OUT="${OUT:?set OUT to a new absolute JSON report path}"
CONFIG="${CONFIG:-$PWD/configs/r0a_bank_ca.yaml}"
REFERENCE_CKPT="${REFERENCE_CKPT:-}"

test -s "$CKPT" || { echo "missing checkpoint: $CKPT"; exit 1; }
test -s "$CONFIG" || { echo "missing config: $CONFIG"; exit 1; }
test ! -e "$OUT" || { echo "refusing to overwrite report: $OUT"; exit 1; }
mkdir -p "$(dirname "$OUT")" logs

source .venv/bin/activate
source scripts/env.sh
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

echo "checkpoint=$CKPT"
echo "config=$CONFIG"
echo "reference_checkpoint=${REFERENCE_CKPT:-none}"
echo "report=$OUT"
echo "gate=heldout_demo_49 windows:256 candidates:32 bootstrap:2000"

args=(
  --checkpoint "$CKPT" --config "$CONFIG" --out "$OUT"
  --device cuda:0 --workers 4 --batch-size 8
)
if [ -n "$REFERENCE_CKPT" ]; then
  test -s "$REFERENCE_CKPT" || { echo "missing reference checkpoint: $REFERENCE_CKPT"; exit 1; }
  args+=(--reference-checkpoint "$REFERENCE_CKPT")
fi
python scripts/bank_ca_gate.py "${args[@]}"

python - "$OUT" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
assert report.get("passed") is True, "bank promotion gate failed"
assert report.get("status") == "PASS", "bank report status is not PASS"
print(json.dumps({
    "passed": report["passed"],
    "checkpoint_step": report["candidate"]["global_step"],
    "config_hash": report["candidate"]["config_hash"],
    "reference": report.get("reference"),
    "metrics": report["metrics"],
}, indent=2))
PY
