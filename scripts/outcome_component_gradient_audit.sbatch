#!/bin/bash
# Eight-A100 read-only component-gradient audit at the exact outcome-GRPO seed.
# This launcher never submits another job and the Python entry point cannot
# update parameters, emit a checkpoint/candidate, or run an SR evaluation.
#
#   sbatch scripts/outcome_component_gradient_audit.sbatch
#   OUT="$PWD/runs/diagnostics/outcome_component_gradient_audit/outcome_component_gradient_audit_v2_s49666_manual.json" \
#     sbatch scripts/outcome_component_gradient_audit.sbatch
#
#SBATCH --job-name=outcome_component_gradient_audit_v2
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=polar4,polar3,polar,grizzly,batch_singlenode
#SBATCH --qos=normal
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit this launcher from the repository root}"

CONFIG="${CONFIG:-$PWD/configs/r0a_outcome_grpo.yaml}"
DIAGNOSTIC_DIR="$PWD/runs/diagnostics/outcome_component_gradient_audit"
OUT="${OUT:-$DIAGNOSTIC_DIR/outcome_component_gradient_audit_v2_s49666_${SLURM_JOB_ID}.json}"

test -x .venv/bin/python || {
  echo "missing training interpreter: .venv/bin/python" >&2
  exit 1
}
test -s "$CONFIG" || {
  echo "missing locked outcome config: $CONFIG" >&2
  exit 1
}
test -s runs/r0a_deploy_s1_eval/ckpt_000049666.pt || {
  echo "missing authenticated step-49666 parent" >&2
  exit 1
}
test ! -e "$OUT" || {
  echo "refusing existing diagnostic output: $OUT" >&2
  exit 1
}
for FOLD in 0 1 2 3 4 5; do
  test -s "runs/outcome_recovery_s49666_train${FOLD}/manifest.json" || {
    echo "missing authenticated train${FOLD} recovery manifest" >&2
    exit 1
  }
done
test -s runs/outcome_recovery_s49666_validation/manifest.json || {
  echo "missing authenticated validation recovery manifest" >&2
  exit 1
}
mkdir -p logs "$DIAGNOSTIC_DIR"

source .venv/bin/activate
source scripts/env.sh
if [ -f .env.local ]; then
  set -a
  source .env.local
  set +a
fi
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_ADDR="${MASTER_ADDR:-$(hostname)}"
export MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
export CONFIG OUT

echo "audit=outcome_component_gradient format_version=2 seed_step=49666 world=8 folds=6 steps_per_fold=8"
echo "mode=diagnostic_only optimizer_steps=0 candidate_eligible=false evaluation_eligible=false"
echo "gradient_sync=SUM/world heldout_component_attribution=true anchor_scales=0,.1,.25,.5,.75,1"
echo "output=$OUT"

srun --kill-on-bad-exit=1 bash -c '
  export RANK="$SLURM_PROCID" WORLD_SIZE="$SLURM_NTASKS" LOCAL_RANK="$SLURM_LOCALID"
  exec .venv/bin/python scripts/outcome_component_gradient_audit.py \
    --config "$CONFIG" --out "$OUT"
'

.venv/bin/python - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert report["kind"] == "loom_outcome_grpo_component_gradient_audit"
assert report["format_version"] == 2
assert report["status"] == "DIAGNOSTIC_COMPLETE"
assert report["execution_validated"] is True
assert report["eligibility"]["diagnostic_only"] is True
assert report["eligibility"]["candidate_eligible"] is False
assert report["eligibility"]["evaluation_eligible"] is False
assert report["eligibility"]["optimizer_steps"] == 0
assert report["audit_geometry"]["world_size"] == 8
assert report["audit_geometry"]["folds"] == 6
assert report["audit_geometry"]["distinct_authenticated_groups_total"] == 384
assert len(report["directional_checks"]) == 6
assert report["audit_geometry"]["fixed_counterfactual_anchor_scales"] == [
    0.0, 0.1, 0.25, 0.5, 0.75, 1.0,
]
for check in report["directional_checks"]:
    attribution = check["gradient_space_attribution"]
    assert set(attribution["recipe_weighted_components"]) == {
        "grpo", "sparse_q_action_ce", "switch_balance_1e-2",
    }
    assert attribution["scope"] == "loss_gradient_space_only"
    assert [row["anchor_scale"] for row in attribution[
        "counterfactual_anchor_scale_sweep"
    ]["rows"]] == [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
assert report["selected_sidecar_post_use_closure"]["passed"] is True
assert report["selected_sidecar_post_use_closure"]["sidecars"] == 432
assert report["no_mutation"]["proposal_digest_before"] == report["no_mutation"]["proposal_digest_after"]
print("outcome component-gradient diagnostic: COMPLETE", flush=True)
PY
