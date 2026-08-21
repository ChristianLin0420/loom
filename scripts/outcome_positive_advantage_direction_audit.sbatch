#!/bin/bash
# Exact no-update eight-A100 positive-advantage direction gate.
# This launcher submits no descendants, trains no parameters, emits no
# checkpoint/candidate, and performs no official evaluation. PASS can support
# only a separately frozen 64-update ineligible PA pilot.
#
#   sbatch scripts/outcome_positive_advantage_direction_audit.sbatch
#
#SBATCH --job-name=outcome_pa_direction_audit
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

CONFIG="$PWD/configs/r0a_outcome_positive_advantage_audit.yaml"
TRIGGER="$PWD/runs/diagnostics/outcome_round_robin_direction_audit/outcome_round_robin_direction_audit_v2_s49666_32577492.json"
DIAGNOSTIC_DIR="$PWD/runs/diagnostics/outcome_positive_advantage_direction_audit"
OUT="${OUT:-$DIAGNOSTIC_DIR/outcome_positive_advantage_direction_audit_v2_s49666_${SLURM_JOB_ID}.json}"

test -x .venv/bin/python || {
  echo "missing training interpreter: .venv/bin/python" >&2
  exit 1
}
test -s "$CONFIG" || {
  echo "missing canonical PA diagnostic scaffold: $CONFIG" >&2
  exit 1
}
test -s "$TRIGGER" || {
  echo "missing authenticated GRPO ABORT trigger: $TRIGGER" >&2
  exit 1
}
test "$(sha256sum "$TRIGGER" | cut -d' ' -f1)" = \
  "1f3c4af95ba97a5976e2cdacee8fc50fda6ff88f725886fca25b9304e4ef9e1f" || {
  echo "GRPO ABORT trigger SHA-256 differs" >&2
  exit 1
}
.venv/bin/python - "$TRIGGER" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert report["kind"] == "loom_outcome_grpo_v2_round_robin_direct_direction_audit"
assert report["format_version"] == 2
assert report["status"] == "ABORT_OUTCOME_OBJECTIVE"
assert report["execution_validated"] is True
assert report["decision"]["passed"] is False
assert report["decision"]["status"] == "ABORT_OUTCOME_OBJECTIVE"
PY
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
  echo "missing authenticated exposed-development manifest" >&2
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
export OUT

echo "audit=outcome_positive_advantage_direction format_version=2 seed_step=49666 world=8 offsets=0,1,2"
echo "trigger_job=32577492 trigger_status=ABORT_OUTCOME_OBJECTIVE trigger_sha256=1f3c4af95ba97a5976e2cdacee8fc50fda6ff88f725886fca25b9304e4ef9e1f"
echo "train_objective=PA1+recovery1+analytic_demo1 grpo=0 balance=0 sparse=0"
echo "train_scoring=read_only_auth_replay+two_objective_graph_exact_B1_passes PA_then_recovery_k3 same_rows_order_old=true"
echo "heldout=signed_GRPO panel=48groups/40tasks bootstrap=10000 seed=49666"
echo "primary_endpoints=pa_sgd,pa_full_sgd,pa_full_reset_adamw production_decay=true"
echo "mode=diagnostic_only live_optimizer_steps=0 full_training_eligible=false output=$OUT"

srun --kill-on-bad-exit=1 bash -c '
  export RANK="$SLURM_PROCID" WORLD_SIZE="$SLURM_NTASKS" LOCAL_RANK="$SLURM_LOCALID"
  exec .venv/bin/python scripts/outcome_positive_advantage_direction_audit.py \
    --config configs/r0a_outcome_positive_advantage_audit.yaml --out "$OUT"
'

.venv/bin/python - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert report["kind"] == "loom_outcome_positive_advantage_round_robin_direction_audit"
assert report["format_version"] == 2
assert report["status"] in {
    "PASS_TO_SEPARATE_64_UPDATE_PA_INELIGIBLE_PILOT_FREEZE",
    "ABORT_POSITIVE_ADVANTAGE_OBJECTIVE",
}
assert report["execution_validated"] is True
assert report["trigger"]["required_sha256"] == "1f3c4af95ba97a5976e2cdacee8fc50fda6ff88f725886fca25b9304e4ef9e1f"
assert report["trigger"]["receipt"]["status"] == "ABORT_OUTCOME_OBJECTIVE"
assert report["config"]["nonlaunchable"] is True
assert report["instrumentation_history"] == [{
    "job_id": 32580600,
    "format_version": 1,
    "terminal_state": "FAILED",
    "exit_code": "2:0",
    "classification": "INVALID_NO_SCIENTIFIC_EVIDENCE",
    "cause": "canonical_PA_disabled_sparse_target_was_passed_to_demo_anchor_constructor",
    "report_emitted": False,
    "panel_statistics_reached": False,
}]
anchor_construction = report["config"]["demo_anchor_construction"]
assert anchor_construction["config_source"] == "authenticated_inherited_v2_resolved_config"
assert anchor_construction["canonical_pa_proposal_target"] == {"enabled": False, "weight": 0.0}
assert anchor_construction["inherited_construction_target"]["enabled"] is True
assert anchor_construction["v2_core_internal_target_producer_weight"] == 1.0
assert anchor_construction["canonical_pa_config_unchanged_after_constructor"] is True
assert anchor_construction["construction_config_unchanged_after_constructor"] is True
assert anchor_construction["sparse_ce_scalar_computed"] is False
assert anchor_construction["sparse_ce_graph_constructed"] is False
assert report["config"]["weights"] == {
    "positive_advantage": 1.0,
    "recovery_reference": 1.0,
    "exact_analytic_vjp_demo_reference": 1.0,
    "grpo_train": 0.0,
    "switch_balance": 0.0,
    "sparse_ce": 0.0,
}
construction = report["direction_construction"]
assert construction["measurement_scope"] == "authoritative_direct_pa_gradients_only"
assert construction["train_scoring_geometry"] == "read-only authentication replay plus two objective-graph exact B1 scorer passes per train point"
assert construction["coefficient_tuning_or_switching"] is False
assert all(not row["computed"] and not row["graph_included"]
           for row in construction["forbidden_train_terms"].values())
assert report["geometry"]["global_steps"] == [49666, 49667, 49668]
assert report["geometry"]["train_draws"] == 24
assert report["geometry"]["panel_groups"] == 48
assert report["geometry"]["panel_tasks"] == 40
assert report["bootstrap_resample_matrix"]["samples"] == 10000
assert report["primary_endpoint_task_bootstrap"]["family_size"] == 3
assert construction["virtual_clone_optimizer_steps"] == 3
assert len(report["train_points"]) == 3
for point in report["train_points"]:
    assert point["direct_positive_advantage_gradient"]["repeat_consistency"]["passed"] is True
    assert point["direct_positive_advantage_gradient"]["repeat_consistency"]["independent_synchronised_backward_calls"] == 2
    assert point["direct_full_pa_beta1_lambda1_gradient"]["repeat_consistency"]["passed"] is True
    assert point["direct_full_pa_beta1_lambda1_gradient"]["repeat_consistency"]["independent_synchronised_backward_calls"] == 2
    assert point["two_pass_train_scoring_identity"]["second_pass_recovery_value_exactly_zero"] is True
    assert isinstance(point["two_pass_train_scoring_identity"]["second_pass_recovery_gradient_bitwise_zero"], bool)
assert report["decision"]["status"] == report["status"]
assert report["decision"]["passed"] is report["status"].startswith("PASS_TO_")
if report["status"].startswith("PASS_TO_"):
    assert report["decision"]["all_six_reference_relative_bounds_passed"] is True
    assert report["decision"]["all_three_recovery_second_pass_gradients_bitwise_zero"] is True
    assert report["decision"]["all_three_demo_analytic_vjp_gradients_bitwise_zero"] is True
else:
    assert report["status"] == "ABORT_POSITIVE_ADVANTAGE_OBJECTIVE"
    assert report["decision"]["passed"] is False
# Complete, finite, nonzero reference VJPs are scientific ABORT evidence.  The
# bitwise-zero conditions are therefore mandatory only for PASS and are not an
# unconditional execution-validity postcondition.
assert report["selected_sidecar_post_use_closure"]["sidecars"] == 72
assert report["no_mutation"]["live_optimizer_steps"] == 0
assert report["no_mutation"]["virtual_clone_optimizer_steps"] == 3
assert report["no_mutation"]["proposal_digest_before"] == report["no_mutation"]["proposal_digest_after"]
assert report["eligibility"]["full_training_eligible"] is False
assert report["eligibility"]["official_evaluation_eligible"] is False
print(f"outcome PA direction audit: {report['status']}", flush=True)
PY
