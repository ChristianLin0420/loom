#!/bin/bash
# One-GPU, one-fold standalone outcome-GRPO round 0. This launcher trains only;
# it never submits, evaluates, or promotes a follow-on job.
#
#   OUT=/absolute/new/ckpt_000050066.pt \
#     sbatch scripts/outcome_grpo_round0.sbatch
#
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=polar4,polar3,polar,grizzly,batch_singlenode
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit this launcher from the repository root}"

ROOT="$(pwd -P)"
PARENT="$ROOT/runs/r0a_deploy_s1_eval/ckpt_000049666.pt"
TRAIN_COLLECTION="$ROOT/runs/outcome_recovery_s49666_train0"
VALIDATION_COLLECTION="$ROOT/runs/outcome_recovery_s49666_validation"
OUT="${OUT:?set OUT to a fresh absolute descendant checkpoint path}"

case "$OUT" in
  /*) ;;
  *) echo "OUT must be absolute, got: $OUT" >&2; exit 2 ;;
esac
test -x .venv/bin/python || { echo "missing training interpreter: .venv/bin/python" >&2; exit 1; }
test -s "$PARENT" || { echo "missing exact parent: $PARENT" >&2; exit 1; }
test -s "$TRAIN_COLLECTION/manifest.json" || {
  echo "missing exact train0 manifest: $TRAIN_COLLECTION/manifest.json" >&2
  exit 1
}
test -s "$VALIDATION_COLLECTION/manifest.json" || {
  echo "missing exact validation manifest: $VALIDATION_COLLECTION/manifest.json" >&2
  exit 1
}
test ! -e "$OUT" || { echo "refusing to overwrite descendant: $OUT" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")" logs

source .venv/bin/activate
source scripts/env.sh
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

echo "parent=$PARENT"
echo "train_collection=$TRAIN_COLLECTION split=train0 groups=200"
echo "validation_collection=$VALIDATION_COLLECTION split=validation groups=400"
echo "output=$OUT updates=400 device=cuda:0"

srun --kill-on-bad-exit=1 --ntasks=1 python scripts/train_outcome_grpo_round0.py \
  --checkpoint "$PARENT" \
  --train-collection "$TRAIN_COLLECTION" \
  --validation-collection "$VALIDATION_COLLECTION" \
  --out "$OUT" \
  --device cuda:0 \
  --chunk-replans 16

# This is checkpoint authentication, not promotion or simulator evaluation.
# It reuses the promotion gate's canonical checkpoint parser, then independently
# proves the round-0 recipe/trust record and byte identity of every frozen tensor.
python - "$OUT" "$PARENT" "$TRAIN_COLLECTION" "$VALIDATION_COLLECTION" <<'PY'
import json
import sys
from pathlib import Path

import torch

from loom.eval import outcome_recovery
from loom.train.outcome_grpo_round0 import (
    _config_hash,
    _source_identity,
    frozen_model_digest,
    proposal_model_digest,
)


out, parent_path, train_path, validation_path = map(
    lambda value: Path(value).resolve(), sys.argv[1:]
)


def load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


child = load(out)
parent = load(parent_path)
training = child["outcome_grpo"]
recipe = child["resolved_config"]["outcome_grpo"]
trust = training["trust_gate"]
promotion = {
    "path": str(out),
    "sha256": outcome_recovery.sha256_file(out),
    "global_step": child["global_step"],
    "config_hash": child["config_hash"],
}

assert promotion["global_step"] == child["global_step"] == 50_066
assert promotion["config_hash"] == child["config_hash"]
assert _config_hash(child["resolved_config"]) == child["config_hash"]
assert training["format_version"] == recipe["format_version"] == 1
assert training["kind"] == "loom_outcome_grpo_round0_proposal_descendant"
assert training["parent_global_step"] == parent["global_step"] == 49_666
assert training["parent"]["sha256"] == outcome_recovery.SEED_CHECKPOINT_SHA256
assert training["parent_config_hash"] == outcome_recovery.SEED_CONFIG_HASH
assert training["descendant_global_step"] == 50_066
assert training["optimizer_steps"] == recipe["optimizer_steps"] == 400
assert recipe["epochs"] == 2 and recipe["groups_per_epoch"] == 200
assert training["training"]["optimizer_steps"] == 400
assert training["training"]["groups_per_epoch"] == 200
assert training["train_collection"]["path"] == str(train_path)
assert training["train_collection"]["split"] == "train0"
assert training["validation_collection"]["path"] == str(validation_path)
assert training["validation_collection"]["split"] == "validation"
assert training["trainer_source"] == _source_identity()

expected_checks = {
    "clip_fraction", "ess_fraction", "coeff_drift_p95_l1", "live_ops",
    "expert_target_identity", "expert_sparse_ce_no_worsening",
    "expert_topk_overlap_decline", "nonfinite", "unexpected_gradients",
}
assert trust["passed"] is True
assert set(trust["checks"]) == expected_checks
assert all(row["pass"] is True for row in trust["checks"].values())
assert trust["counts"]["training_nonfinite"] == 0
assert trust["counts"]["final_nonfinite"] == 0
assert training["training"]["nonfinite"] == 0
assert training["training"]["unexpected_gradients"] == []

parent_frozen = frozen_model_digest(parent["model"])
child_frozen = frozen_model_digest(child["model"])
assert parent_frozen == child_frozen == training["frozen_model"]
assert child["consolidated"]["frozen_model_sha256"] == child_frozen["sha256"]
assert proposal_model_digest(parent["model"]) == training["initial_proposal"]
assert proposal_model_digest(child["model"]) == training["final_proposal"]
assert training["mutated_model_prefixes"] == ["proposal."]

assert child["optimizer"]["kind"] == "proposal_only_adamw"
assert child["optimizer"]["state_reset_at_entry"] is True
optimizer_rows = list(child["optimizer"]["state_dict"]["state"].values())
assert optimizer_rows
assert {int(float(row["step"])) for row in optimizer_rows} == {400}

print(json.dumps({
    "status": "PASS",
    "checkpoint": promotion,
    "kind": training["kind"],
    "optimizer_steps": training["optimizer_steps"],
    "trust_checks": trust["checks"],
    "frozen_model": child_frozen,
}, indent=2, sort_keys=True))
PY
