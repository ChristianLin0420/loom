#!/bin/bash
# Eight-GPU, read-only retrospective diagnostic for the already-failed strict
# outcome-GRPO run.  This launcher never trains, selects, promotes, submits a
# follow-on job, or runs simulator episodes.
#
#   sbatch scripts/outcome_grpo_early_curve_scan.sbatch
#
# The sole output is a fresh INELIGIBLE diagnostic JSON outside the input run.
#
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=polar4,polar3,polar,grizzly,batch_singlenode
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit this launcher from the repository root}"

FAILED_RUN="$PWD/runs/r0a_outcome_grpo_strict_s1"
OUT="${OUT:-$PWD/runs/r0a_outcome_grpo_strict_s1_early_curve_diagnostic/summary.json}"

test -x .venv/bin/python || {
  echo "missing training interpreter: .venv/bin/python" >&2
  exit 1
}
test -s configs/r0a_outcome_grpo.yaml || {
  echo "missing locked outcome-GRPO recipe" >&2
  exit 1
}
test -s "$FAILED_RUN/config.json" || {
  echo "missing failed-run resolved config" >&2
  exit 1
}
test -s "$FAILED_RUN/terminal_report.json" || {
  echo "missing failed terminal report" >&2
  exit 1
}
for step in 000049866 000050066 000050466 000051266; do
  test -s "$FAILED_RUN/outcome_ckpt_${step}.pt" || {
    echo "missing predeclared early checkpoint: $step" >&2
    exit 1
  }
done
test ! -e "$OUT" || {
  echo "refusing existing diagnostic output: $OUT" >&2
  exit 1
}
case "$OUT" in
  "$FAILED_RUN"/*)
    echo "diagnostic output must be outside failed run: $OUT" >&2
    exit 1
    ;;
esac
mkdir -p "$(dirname "$OUT")" logs

source .venv/bin/activate
source scripts/env.sh
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export MASTER_ADDR="${MASTER_ADDR:-$(hostname)}"
export MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
export OUT

echo "diagnostic=outcome_grpo_early_curve input_run=$FAILED_RUN"
echo "steps=49866,50066,50466,51266 world=${SLURM_NTASKS:-8} tasks=surrogate+trust_per_step"
echo "mode=read_only optimizer_steps=0 simulator_episodes=0 promotion=INELIGIBLE"
echo "out=$OUT"

srun --kill-on-bad-exit=1 bash -c '
  export RANK="$SLURM_PROCID" WORLD_SIZE="$SLURM_NTASKS" LOCAL_RANK="$SLURM_LOCALID"
  exec .venv/bin/python scripts/outcome_grpo_early_curve_scan.py --out "$OUT"
'

.venv/bin/python - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
assert report["status"] == "DIAGNOSTIC_COMPLETE"
assert report["diagnostic_only"] is True
assert report["read_only_inputs"] is True
assert report["predeclared_steps"] == [49866, 50066, 50466, 51266]
assert report["optimizer_steps"] == 0 and report["simulator_episodes"] == 0
assert report["candidate_emitted"] is False
assert report["promotion_gate_evaluated"] is False
assert set(report["eligibility"].values()) >= {"INELIGIBLE"}
assert report["summary"]["selection"]["performed"] is False
assert report["summary"]["selection"]["permitted"] is False
assert report["summary"]["selection"]["best_checkpoint"] is None
assert len(report["execution"]["tasks"]) == 8
assert [
    (row["kind"], row["step"]) for row in report["execution"]["tasks"]
] == [
    ("surrogate", 49866), ("trust", 49866),
    ("surrogate", 50066), ("trust", 50066),
    ("surrogate", 50466), ("trust", 50466),
    ("surrogate", 51266), ("trust", 51266),
]
assert set(report["evaluations"]) == {"49866", "50066", "50466", "51266"}
assert all(
    row["eligibility"] == "INELIGIBLE"
    for row in report["summary"]["steps"].values()
)
assert report["no_mutation"]["passed"] is True
print("outcome-GRPO early-curve diagnostic: COMPLETE and INELIGIBLE", flush=True)
PY
