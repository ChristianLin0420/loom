#!/bin/bash
# One-A100, read-only preflight for the first outcome-GRPO ratio/backward.
# This launcher writes only its Slurm log and never submits a follow-on job.
#
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=polar4,polar3,polar,grizzly,batch_singlenode
#SBATCH --qos=normal
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit this launcher from the repository root}"

test -x .venv/bin/python || {
  echo "missing training interpreter: .venv/bin/python" >&2
  exit 1
}
test -s runs/r0a_deploy_s1_eval/ckpt_000049666.pt || {
  echo "missing authenticated step-49666 parent" >&2
  exit 1
}
test -s runs/outcome_recovery_s49666_train0/manifest.json || {
  echo "missing authenticated train0 collection" >&2
  exit 1
}
mkdir -p logs
OUT="$PWD/runs/outcome_ratio_identity_probe_d5ef53e9.json"
test ! -e "$OUT" || {
  echo "refusing existing probe result: $OUT" >&2
  exit 1
}

source scripts/env.sh
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "probe=outcome_ratio_identity step=49666 rank=0 world_geometry=8 contexts_per_arm=2"
echo "mode=read_only transfer_chunk=32 proposal_batch=1 pl_batch=1 optimizer_steps=0"

srun --kill-on-bad-exit=1 .venv/bin/python \
  scripts/outcome_ratio_identity_probe.py --out "$OUT"

.venv/bin/python - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
assert report["status"] == "PASS"
assert report["source_identity"]["config_hash"] == "25afdedfc9deea5e"
assert report["source_identity"]["trainer_source"]["sha256"] == (
    "d5ef53e9f2e276f17d68f80b4c081c8f09b0d89ea9a966214fc3b63387364a52"
)
assert report["strict_determinism"] == {
    "deterministic_algorithms": True, "warn_only": False,
}
assert report["runtime"]["deterministic_algorithms"] is True
assert report["runtime"]["deterministic_warn_only"] is False
pl = report["pl_rowwise_replay"]
geometry = pl["geometry"]
legacy = pl["legacy_batched_pl"]
fixed = pl["fixed_rowwise"]
rows = pl["witness_rows"]
assert pl["passed"] is True and pl["split"] == "train0"
assert geometry["transfer_chunk_replans"] == 32
assert geometry["transfer_chunk_gt_one"] is True
assert geometry["actual_witness_transfer_rows"] > 1
assert geometry["proposal_batch_size"] == 1
assert geometry["production_pl_batch_size"] == 1
assert geometry["observed_production_pl_batch_sizes"] == [1]
assert legacy["batch_size"] == geometry["actual_witness_transfer_rows"]
assert legacy["mismatch_count"] == len(rows) > 0
assert legacy["max_abs_old_logprob_error"] > 0.0
assert fixed["rows_checked"] == pl["rows_scanned"] > 0
assert fixed["max_abs_coeff_error"] == 0.0
assert fixed["max_abs_old_logprob_error"] == 0.0
assert fixed["all_exact"] is True
assert all(
    row["actual_transfer_rows"] > 1
    and row["legacy_abs_error"] > 0.0
    and row["fixed_abs_error"] == 0.0
    and row["fixed_coeff_max_abs_error"] == 0.0
    for row in rows
)
assert report["ratios"]["ratio_atoms"] == 14
assert report["ratios"]["all_ratio_atoms_exactly_one"] is True
assert report["backward"]["passed"] is True
assert not any(
    "nondetermin" in message.lower() or "non-determin" in message.lower()
    for message in report["backward"]["warnings"]
)
bf16 = report["bf16_attention_backward"]
assert bf16["passed"] is True and bf16["batch_size"] == 8
assert bf16["strict_determinism"] == report["strict_determinism"]
assert bf16["proposal_grad_norm"] > 0.0
assert bf16["proposal_digest_before"] == bf16["proposal_digest_after"]
assert not any(
    "nondetermin" in message.lower() or "non-determin" in message.lower()
    for message in bf16["warnings"]
)
assert report["no_mutation"]["optimizer_steps"] == 0
assert report["no_mutation"]["passed"] is True
print("outcome ratio identity artifact: PASS", flush=True)
PY
