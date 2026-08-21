#!/bin/bash
# Exact no-update eight-A100 outcome-GRPO v2 round-robin direction gate.
# This launcher does not submit descendants, train, checkpoint, evaluate SR,
# emit candidates, or promote. PASS can authorize only a separate 64-update
# ineligible pilot after its own recipe freeze.
#
#   sbatch scripts/outcome_round_robin_direction_audit.sbatch
#
#SBATCH --job-name=outcome_rr_direction_audit
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=polar4,polar3,polar,grizzly,batch_singlenode
#SBATCH --qos=normal
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit this launcher from the repository root}"

CONFIG="$PWD/configs/r0a_outcome_grpo_v2_pilot.yaml"
DIAGNOSTIC_DIR="$PWD/runs/diagnostics/outcome_round_robin_direction_audit"
OUT="${OUT:-$DIAGNOSTIC_DIR/outcome_round_robin_direction_audit_v2_s49666_${SLURM_JOB_ID}.json}"

test -x .venv/bin/python || {
  echo "missing training interpreter: .venv/bin/python" >&2
  exit 1
}
test -s "$CONFIG" || {
  echo "missing v2 scaffold config: $CONFIG" >&2
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
    echo "missing authenticated train${FOLD} manifest" >&2
    exit 1
  }
done
test -s runs/outcome_recovery_s49666_validation/manifest.json || {
  echo "missing authenticated development validation manifest" >&2
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

echo "audit=outcome_round_robin_direct_direction format_version=2 seed_step=49666 world=8 offsets=0,1,2"
echo "alpha=0 pre_frozen=true panel=48groups/40tasks bootstrap=10000 seed=49666"
echo "primary_coefficients=sparse_alpha0,recovery_beta1,demo_lambda1,balance0.01 predeclared=true"
echo "primary_endpoints=grpo_sgd,full_beta1_lambda1_sgd,full_beta1_lambda1_reset_adamw production_decay=true"
echo "measurement=direct_grpo,direct_full repeated_direct_checks=true attribution_sweeps=false"
echo "mode=diagnostic_only live_optimizer_steps=0 full_training_eligible=false"
echo "pass_authority=64_update_ineligible_pilot_alpha0_beta1_lambda1_only output=$OUT"

srun --kill-on-bad-exit=1 bash -c '
  export RANK="$SLURM_PROCID" WORLD_SIZE="$SLURM_NTASKS" LOCAL_RANK="$SLURM_LOCALID"
  exec .venv/bin/python scripts/outcome_round_robin_direction_audit.py \
    --config "$CONFIG" --out "$OUT"
'

.venv/bin/python - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert report["kind"] == "loom_outcome_grpo_v2_round_robin_direct_direction_audit"
assert report["format_version"] == 2
assert report["status"] in {
    "PASS_TO_64_UPDATE_ALPHA0_BETA1_LAMBDA1_INELIGIBLE_PILOT",
    "ABORT_OUTCOME_OBJECTIVE",
}
assert report["execution_validated"] is True
assert report["config"]["pre_frozen_sparse_ce_alpha"] == 0.0
assert report["config"]["sparse_ce_disabled_graph_excluded"] is True
assert report["direction_construction"]["sparse_ce_graph_included"] is False
assert report["direction_construction"]["disabled_sparse_ce_graph_excluded"] is True
assert report["direction_construction"]["sparse_ce_computed"] is False
assert report["direction_construction"]["measurement_scope"] == "authoritative_direct_gradients_only"
assert report["config"]["prospective_64_pilot_initial_recovery_beta"] == 1.0
assert report["config"]["prospective_64_pilot_demo_reference_lambda"] == 1.0
assert report["geometry"]["world_size"] == 8
assert report["geometry"]["global_steps"] == [49666, 49667, 49668]
assert report["geometry"]["train_draws"] == 24
assert report["geometry"]["panel_groups"] == 48
assert report["geometry"]["panel_tasks"] == 40
assert report["bootstrap_resample_matrix"]["samples"] == 10000
assert report["primary_endpoint_task_bootstrap"]["family_size"] == 3
assert report["direction_construction"]["virtual_clone_optimizer_steps"] == 3
assert len(report["train_points"]) == 3
for point in report["train_points"]:
    assert point["direct_grpo_gradient"]["repeat_consistency"]["passed"] is True
    assert point["direct_grpo_gradient"]["repeat_consistency"]["independent_synchronised_backward_calls"] == 2
    assert point["direct_full_alpha0_beta1_lambda1"]["repeat_consistency"]["passed"] is True
    assert point["direct_full_alpha0_beta1_lambda1"]["repeat_consistency"]["independent_synchronised_backward_calls"] == 2
assert "alpha_sweep_attribution_only" not in report
assert "reset_adamw_without_decay" not in report["direction_construction"]
assert [row["job_id"] for row in report["instrumentation_history"]["previous_invalid_jobs"]] == [32575962, 32576514]
assert report["instrumentation_history"]["both_stopped_before_panel_statistics"] is True
assert report["selected_sidecar_post_use_closure"]["sidecars"] == 72
assert report["no_mutation"]["live_optimizer_steps"] == 0
assert report["no_mutation"]["virtual_clone_optimizer_steps"] == 3
assert report["no_mutation"]["proposal_digest_before"] == report["no_mutation"]["proposal_digest_after"]
assert report["eligibility"]["full_training_eligible"] is False
assert report["eligibility"]["official_evaluation_eligible"] is False
print(f"outcome round-robin direction audit: {report['status']}", flush=True)
PY
