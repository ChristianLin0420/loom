#!/usr/bin/env python3
"""Read-only A100 preflight for the first outcome-GRPO recovery backward.

The probe authenticates the pinned step-49,666 parent and immutable ``train0``
collection.  It first scans stored sampled-policy rows in deterministic
group/arm/replan order, transferring 32 replans at a time while the production
scorer executes both proposal and Plackett--Luce work row-wise at batch one.  A
diagnostic legacy chunk-batched PL reduction must expose at least one concrete
stored-row mismatch, while production coefficients and old log-probabilities
on every scanned row must remain bitwise exact.

It then deep-loads the group selected by the production step/rank sampler at
``(step=49_666, rank=0, world=8)`` and deliberately calls the sampled-group
objective from inside an outer bf16 autocast region.  The scorer must override
that region with its row-wise fp32 recovery geometry and reproduce fourteen
exact unit importance ratios.

Both production attention paths run under strict deterministic algorithms:
the exact fp32 ratio loss exercises Memory-Efficient SDPA and a representative
bf16 proposal loss exercises the Flash-eligible anchor geometry.  Backward is
allowed; an optimizer update is not.  Exact proposal-state digests and an empty
optimizer state are checked before and after backward.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch import Tensor, nn  # noqa: E402

from loom.eval import outcome_recovery as recovery  # noqa: E402
from loom.train import outcome_grpo as grpo  # noqa: E402
from loom.train.atomic import atomic_write_text  # noqa: E402


CHECKPOINT_REL = "runs/r0a_deploy_s1_eval/ckpt_000049666.pt"
COLLECTION_REL = "runs/outcome_recovery_s49666_train0"
EXPECTED_COLLECTION_IDENTITY_DIGEST = (
    "6aae6bbb5f6226de726de64bd8b57d6f4fb673a63c5e23c49ad136a03dd75433"
)
EXPECTED_MANIFEST_SHA256 = (
    "f92f50960e1640b32f2f50c6e9a7c61603204ea21369c6e2493d3770b3683c17"
)
EXPECTED_GROUP_INDEX = 19
EXPECTED_GROUP_ID = "libero_spatial/task=03/trial=14/seed=0"
EXPECTED_SIDECAR = "groups/libero_spatial__task03__trial14__seed0.pt"
EXPECTED_SIDECAR_SHA256 = (
    "6cdb7ac21d2469f2c104ed47dd92903029d9569d2bedc2ba0def5aecb00cb2ef"
)
EXPECTED_SIDECAR_SIZE = 306_690_694
EXPECTED_REPLANS = (93, 97, 97, 97, 97, 97, 97, 97)
EXPECTED_REWARDS = (1, 0, 0, 0, 0, 0, 0, 0)
EXPECTED_REPLAN_INDICES = {
    1: (88, 93),
    2: (69, 11),
    3: (67, 15),
    4: (50, 2),
    5: (93, 0),
    6: (69, 59),
    7: (68, 65),
}
EXPECTED_RATIO_ATOMS = 14
PL_WITNESS_TRANSFER_CHUNK_REPLANS = 32
EXPECTED_CONFIG_HASH = "25afdedfc9deea5e"
EXPECTED_TRAINER_SOURCE_SHA256 = (
    "d5ef53e9f2e276f17d68f80b4c081c8f09b0d89ea9a966214fc3b63387364a52"
)


class ProbeError(RuntimeError):
    """The immutable input, A100, ratio, gradient, or no-mutation gate failed."""


def _require(condition: Any, message: str) -> None:
    if not bool(condition):
        raise ProbeError(message)


def _captured_warning_messages(caught: Sequence[warnings.WarningMessage]) -> list[str]:
    return [str(item.message) for item in caught]


def _require_no_nondeterminism_warning(
    caught: Sequence[warnings.WarningMessage],
    *,
    label: str,
) -> list[str]:
    messages = _captured_warning_messages(caught)
    rejected = [
        message for message in messages
        if "nondetermin" in message.lower() or "non-determin" in message.lower()
    ]
    _require(not rejected,
             f"{label} emitted nondeterministic-algorithm warnings: {rejected}")
    return messages


def _runtime(device: torch.device) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    capability = tuple(torch.cuda.get_device_capability(device))
    _require("A100" in properties.name.upper(),
             f"probe requires a real A100, got {properties.name!r}")
    _require(capability == (8, 0),
             f"probe requires A100 compute capability (8,0), got {capability}")
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": properties.name,
        "gpu_capability": list(capability),
        "visible_gpu_count": torch.cuda.device_count(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_warn_only": bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
    }


def validated_source_identity() -> dict[str, Any]:
    """Bind this preflight to the exact recipe and trainer it authorizes."""
    from loom.train.loop import read_config  # noqa: PLC0415

    cfg = read_config(ROOT / "configs/r0a_outcome_grpo.yaml")
    grpo.validate_recipe_config(cfg)
    config_hash = grpo._config_hash(cfg)
    trainer = grpo._trainer_source_identity()
    _require(config_hash == EXPECTED_CONFIG_HASH,
             f"resolved config hash drifted: {config_hash}")
    _require(trainer.get("sha256") == EXPECTED_TRAINER_SOURCE_SHA256,
             f"trainer source closure drifted: {trainer.get('sha256')}")
    return {
        "config_hash": config_hash,
        "trainer_source": trainer,
        "probe_source": {
            "path": str(Path(__file__).resolve()),
            "sha256": recovery.sha256_file(Path(__file__).resolve()),
        },
    }


def deterministic_rank0_selection(
    collection: grpo.ValidatedRecoveryCollection,
) -> tuple[int, int, dict[int, tuple[int, ...]], Mapping[str, Any]]:
    """Resolve and pin the production fold-0 draw without opening five folds.

    ``DeterministicOutcomeSampler`` only consults fold zero and its identity at
    ``START_STEP``.  Repeating the authenticated fold-zero metadata into the
    unused five slots therefore gives the exact production rank-zero draw while
    keeping this probe restricted to one collection and one sidecar.
    """
    informative = collection.informative_indices()
    sampler = grpo.DeterministicOutcomeSampler(
        [informative for _ in range(grpo.N_FOLDS)],
        seed=grpo.TRAIN_SEED,
        rank=0,
        world_size=grpo.EXPECTED_WORLD_SIZE,
        start_step=grpo.START_STEP,
        updates_per_fold=grpo.UPDATES_PER_FOLD,
        contexts_per_arm=grpo.EXPECTED_CONTEXTS_PER_ARM,
        identity_digests=[collection.identity_digest for _ in range(grpo.N_FOLDS)],
    )
    fold, group_index, visit = sampler.group_at(grpo.START_STEP)
    _require(fold == 0 and visit == 0,
             f"START-step sampler position drifted: {(fold, group_index, visit)}")
    _require(group_index == EXPECTED_GROUP_INDEX,
             f"rank-0 START-step group drifted: {group_index}")
    receipt = collection.receipts[group_index]
    replans = tuple(int(value) for value in receipt["n_replans_by_arm"])
    indices = sampler.replans_at(grpo.START_STEP, replans)
    expected_receipt = {
        "group_id": EXPECTED_GROUP_ID,
        "sidecar": EXPECTED_SIDECAR,
        "sha256": EXPECTED_SIDECAR_SHA256,
        "size": EXPECTED_SIDECAR_SIZE,
        "n_arms": recovery.GROUP_SIZE,
        "n_replans_by_arm": list(EXPECTED_REPLANS),
        "terminal_rewards": list(EXPECTED_REWARDS),
    }
    for key, expected in expected_receipt.items():
        _require(receipt.get(key) == expected,
                 f"fixed group receipt {key} drifted: {receipt.get(key)!r}")
    _require(indices == EXPECTED_REPLAN_INDICES,
             f"rank-0 START-step replan indices drifted: {indices}")
    return group_index, visit, indices, receipt


def _require_ratio_identity(
    metrics: Mapping[str, float], *, device: torch.device,
) -> dict[str, Any]:
    _require(int(metrics["ratio_atoms"]) == EXPECTED_RATIO_ATOMS,
             f"expected fourteen ratio atoms, got {metrics['ratio_atoms']}")
    evidence = grpo._require_initial_behavior_ratio_identity(metrics, device=device)
    _require(
        float(metrics["ratio_min"]) == 1.0
        and float(metrics["ratio_mean"]) == 1.0
        and float(metrics["ratio_max"]) == 1.0
        and float(metrics["ratio_sum"]) == float(EXPECTED_RATIO_ATOMS)
        and float(metrics["ratio_square_sum"]) == float(EXPECTED_RATIO_ATOMS),
        "not all fourteen sampled ratios are exactly one",
    )
    _require(float(metrics["max_abs_logratio"]) == 0.0,
             "sampled max_abs_logratio is not exact zero")
    _require(float(metrics["clip_fraction"]) == 0.0
             and float(metrics["clipped_atoms"]) == 0.0,
             "sampled clipping is not exact zero")
    return {**evidence, "all_ratio_atoms_exactly_one": True}


def _production_stored_order_with_pl_batch_witness(
    proposal: nn.Module,
    z: Tensor,
    lang: Tensor,
    order: Tensor,
) -> tuple[Tensor, Tensor, list[int], Any]:
    """Call the production scorer while observing only its PL batch sizes."""
    production_pl = grpo.pl_log_prob
    observed: list[int] = []

    def witnessed_pl(logits: Tensor, ordered_support: Tensor) -> Tensor:
        observed.append(int(logits.shape[0]))
        return production_pl(logits, ordered_support)

    grpo.pl_log_prob = witnessed_pl
    try:
        score, logits = grpo.stored_order_logprob(proposal, z, lang, order)
    finally:
        grpo.pl_log_prob = production_pl
    return score, logits, observed, production_pl


@torch.no_grad()
def scan_train0_pl_witness(
    proposal: nn.Module,
    collection: grpo.ValidatedRecoveryCollection,
    *,
    device: torch.device,
    transfer_chunk_replans: int = PL_WITNESS_TRANSFER_CHUNK_REPLANS,
) -> dict[str, Any]:
    """Find real rows that distinguish legacy chunked PL from production B=1."""
    _require(collection.split == "train0",
             f"PL witness scan requires train0, got {collection.split!r}")
    _require(transfer_chunk_replans > 1,
             "PL witness transfer chunk must contain more than one row")
    _require(grpo.PROPOSAL_SCORING_BATCH_SIZE == 1,
             "production proposal scorer is not batch one")
    proposal.eval()
    grpo._require_exact_proposal_scoring_environment(proposal, device)

    groups_scanned = 0
    chunks_scanned = 0
    rows_scanned = 0
    observed_production_pl_batches: set[int] = set()
    max_fixed_coeff_error = 0.0
    max_fixed_old_logprob_error = 0.0
    max_legacy_old_logprob_error = 0.0
    witness_rows: list[dict[str, Any]] = []
    witness_batch_size = 0

    for group_index, receipt in enumerate(collection.receipts):
        payload = collection.load(group_index)
        groups_scanned += 1
        _require(payload.get("group_id") == receipt.get("group_id"),
                 f"PL witness group identity drifted at index {group_index}")
        for arm_index in range(1, recovery.GROUP_SIZE):
            arm = payload["arms"][arm_index]
            n_replans = int(arm["z"].shape[0])
            for lo in range(0, n_replans, transfer_chunk_replans):
                hi = min(lo + transfer_chunk_replans, n_replans)
                actual_transfer_rows = hi - lo
                z = arm["z"][lo:hi].to(
                    device=device, dtype=torch.float32, non_blocking=True,
                )
                order = arm["ordered_support"][lo:hi].to(device=device)
                stored_old_logprob = arm["old_logprob"][lo:hi].to(
                    device=device,
                ).float()
                stored_coeff = arm["coeff"][lo:hi].to(device=device).float()
                fixed_old_logprob, logits, pl_batches, legacy_pl = (
                    _production_stored_order_with_pl_batch_witness(
                        proposal, z, arm["lang"], order,
                    )
                )
                observed_production_pl_batches.update(pl_batches)
                _require(
                    pl_batches == [1] * actual_transfer_rows,
                    "production stored-order PL did not execute every row at B=1: "
                    f"{pl_batches}",
                )
                fixed_coeff = grpo.weights_from_logits(
                    logits.float(), order.to(torch.int64), logits.shape[-1],
                ).float()
                coeff_delta = (fixed_coeff - stored_coeff).abs()
                old_logprob_delta = (
                    fixed_old_logprob.float() - stored_old_logprob
                ).abs()
                coeff_error = float(coeff_delta.max())
                old_logprob_error = float(old_logprob_delta.max())
                max_fixed_coeff_error = max(max_fixed_coeff_error, coeff_error)
                max_fixed_old_logprob_error = max(
                    max_fixed_old_logprob_error, old_logprob_error,
                )
                _require(
                    torch.equal(fixed_coeff, stored_coeff),
                    "row-wise production coefficient replay was not exact in "
                    f"{payload['group_id']} arm {arm_index} replans [{lo},{hi})",
                )
                _require(
                    torch.equal(fixed_old_logprob.float(), stored_old_logprob),
                    "row-wise production old-logprob replay was not exact in "
                    f"{payload['group_id']} arm {arm_index} replans [{lo},{hi})",
                )
                chunks_scanned += 1
                rows_scanned += actual_transfer_rows

                # A singleton cannot expose the rejected outer-batch geometry.
                if actual_transfer_rows <= 1:
                    continue
                with torch.autocast(device_type=device.type, enabled=False):
                    legacy_old_logprob = legacy_pl(
                        logits.float(), order.to(torch.int64),
                    ).float()
                legacy_delta = (legacy_old_logprob - stored_old_logprob).abs()
                max_legacy_old_logprob_error = max(
                    max_legacy_old_logprob_error, float(legacy_delta.max()),
                )
                mismatches = torch.nonzero(
                    legacy_delta != 0, as_tuple=False,
                ).flatten().tolist()
                if not mismatches:
                    continue
                witness_batch_size = actual_transfer_rows
                for offset in mismatches:
                    row = lo + int(offset)
                    witness_rows.append({
                        "split": collection.split,
                        "group_index": group_index,
                        "group_id": str(payload["group_id"]),
                        "sidecar": str(receipt["sidecar"]),
                        "sidecar_sha256": str(receipt["sha256"]),
                        "arm": arm_index,
                        "replan": row,
                        "actual_transfer_rows": actual_transfer_rows,
                        "stored_order": order[offset].detach().cpu().tolist(),
                        "stored_old_logprob": float(stored_old_logprob[offset]),
                        "legacy_old_logprob": float(legacy_old_logprob[offset]),
                        "legacy_abs_error": float(legacy_delta[offset]),
                        "fixed_old_logprob": float(fixed_old_logprob[offset]),
                        "fixed_abs_error": float(old_logprob_delta[offset]),
                        "fixed_coeff_max_abs_error": float(
                            coeff_delta[offset].max()
                        ),
                    })
                break
            if witness_rows:
                break
        if witness_rows:
            break
        del payload
        gc.collect()

    _require(witness_rows,
             "train0 scan did not reproduce a legacy chunk-batched PL mismatch")
    _require(witness_batch_size > 1,
             "legacy PL witness did not use a multi-row transfer chunk")
    _require(max_fixed_coeff_error == 0.0,
             "production coefficient replay maximum was not exact zero")
    _require(max_fixed_old_logprob_error == 0.0,
             "production old-logprob replay maximum was not exact zero")
    _require(max_legacy_old_logprob_error > 0.0,
             "legacy chunk-batched PL witness error was not positive")
    _require(observed_production_pl_batches == {1},
             "production PL batch-size witness was not exactly {1}")
    return {
        "passed": True,
        "split": collection.split,
        "scan_order": "manifest_group_then_arm1_to_7_then_replan",
        "stop_condition": "first_transfer_chunk_with_legacy_mismatch",
        "groups_scanned": groups_scanned,
        "chunks_scanned": chunks_scanned,
        "rows_scanned": rows_scanned,
        "geometry": {
            "transfer_chunk_replans": transfer_chunk_replans,
            "transfer_chunk_gt_one": True,
            "actual_witness_transfer_rows": witness_batch_size,
            "proposal_batch_size": grpo.PROPOSAL_SCORING_BATCH_SIZE,
            "production_pl_batch_size": 1,
            "observed_production_pl_batch_sizes": sorted(
                observed_production_pl_batches
            ),
        },
        "legacy_batched_pl": {
            "batch_size": witness_batch_size,
            "mismatch_count": len(witness_rows),
            "max_abs_old_logprob_error": max_legacy_old_logprob_error,
        },
        "fixed_rowwise": {
            "rows_checked": rows_scanned,
            "max_abs_coeff_error": max_fixed_coeff_error,
            "max_abs_old_logprob_error": max_fixed_old_logprob_error,
            "all_exact": True,
        },
        "witness_rows": witness_rows,
    }


def execute_identity_backward(
    proposal: nn.Module,
    payload: Mapping[str, Any],
    replan_indices: Mapping[int, Sequence[int]],
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Run the exact forward/backward and prove that no update occurred."""
    proposal.eval()
    grpo._require_exact_proposal_scoring_environment(proposal, device)
    # The production START-step gate compares the standalone proposal module
    # state, not the parent checkpoint's flat ``proposal.*`` model mapping.
    parent_digest = grpo.proposal_module_digest(proposal.state_dict())
    wrapper = grpo._ProposalOnly(proposal)
    optimizer = grpo.build_optimizer(
        wrapper,
        lr=grpo.BASE_LEARNING_RATE,
        weight_decay=grpo.ADAMW_WEIGHT_DECAY,
        betas=grpo.ADAMW_BETAS,
        lr_scales={"proposal": grpo.PROPOSAL_LR_SCALE},
        module_names=["proposal"],
    )
    before = grpo._require_start_step_checkpoint_identity(
        proposal,
        optimizer,
        global_step=grpo.START_STEP,
        parent_proposal=parent_digest,
    )
    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=True,
    ):
        outer_autocast_entered = bool(torch.is_autocast_enabled(device.type))
        grpo_loss, switch_loss, metrics = grpo.sampled_group_losses(
            proposal,
            payload,
            replan_indices,
            device=device,
        )
        outer_autocast_restored = bool(torch.is_autocast_enabled(device.type))

    _require(outer_autocast_entered and outer_autocast_restored,
             "outer bf16 autocast was not active around sampled_group_losses")
    _require(grpo_loss.dtype == torch.float32 and switch_loss.dtype == torch.float32,
             "inner recovery scorer did not preserve fp32 loss geometry")
    ratio = _require_ratio_identity(metrics, device=device)
    total = grpo_loss + grpo.SWITCH_BALANCE_WEIGHT * switch_loss
    _require(total.requires_grad and total.grad_fn is not None,
             "GRPO plus Switch total is not differentiable")
    _require(bool(torch.isfinite(total.detach())), "total loss is nonfinite")
    differentiability_witness = {
        "grpo_requires_grad": bool(grpo_loss.requires_grad),
        "switch_requires_grad": bool(switch_loss.requires_grad),
        "total_requires_grad": bool(total.requires_grad),
        "total_grad_fn": type(total.grad_fn).__name__,
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        total.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    warning_messages = _require_no_nondeterminism_warning(
        caught, label="exact fp32 outcome backward",
    )
    grad_norm, missing, nonfinite = grpo._proposal_grad_health(proposal)
    _require(not missing, f"proposal parameters missing gradients: {missing[:8]}")
    _require(not nonfinite, f"proposal gradients are nonfinite: {nonfinite[:8]}")
    _require(math.isfinite(grad_norm) and grad_norm > 0.0,
             f"proposal gradient norm is not finite and nonzero: {grad_norm}")

    after = grpo._require_start_step_checkpoint_identity(
        proposal,
        optimizer,
        global_step=grpo.START_STEP,
        parent_proposal=parent_digest,
    )
    final_digest = grpo.proposal_module_digest(proposal.state_dict())
    _require(final_digest == parent_digest,
             "proposal parameters mutated without an optimizer step")
    _require(len(optimizer.state) == 0,
             "optimizer state changed even though no step was allowed")
    return {
        "scoring": {
            "outer_autocast": True,
            "outer_autocast_dtype": "bfloat16",
            "inner_autocast": bool(metrics["proposal_scoring_autocast"]),
            "inner_batch_size": int(metrics["proposal_scoring_batch_size"]),
            "grpo_loss_dtype": str(grpo_loss.dtype),
            "switch_loss_dtype": str(switch_loss.dtype),
        },
        "ratios": ratio,
        "losses": {
            "grpo": float(grpo_loss.detach()),
            "switch": float(switch_loss.detach()),
            "switch_weight": grpo.SWITCH_BALANCE_WEIGHT,
            "total": float(total.detach()),
        },
        "differentiability_witness": differentiability_witness,
        "backward": {
            "passed": True,
            "proposal_grad_norm": float(grad_norm),
            "missing_gradients": [],
            "nonfinite_gradients": [],
            "warnings": warning_messages,
        },
        "no_mutation": {
            "passed": True,
            "optimizer_constructed": True,
            "optimizer_steps": 0,
            "optimizer_state_entries_before": before["optimizer_state_entries"],
            "optimizer_state_entries_after": after["optimizer_state_entries"],
            "proposal_digest_before": parent_digest,
            "proposal_digest_after": final_digest,
        },
    }


def execute_strict_bf16_proposal_backward(
    proposal: nn.Module,
    payload: Mapping[str, Any],
    *,
    device: torch.device,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Exercise the Flash-eligible proposal path without mutating its weights."""
    _require(device.type == "cuda",
             "strict bf16 proposal backward requires CUDA")
    _require(
        grpo._strict_outcome_determinism_state()
        == grpo.STRICT_OUTCOME_DETERMINISM,
        "bf16 proposal smoke is not in strict deterministic mode",
    )
    proposal.eval()
    before = grpo.proposal_module_digest(proposal.state_dict())
    proposal.zero_grad(set_to_none=True)
    arm = payload["arms"][1]
    n = int(arm["z"].shape[0])
    _require(n >= batch_size > 0,
             f"bf16 proposal smoke needs {batch_size} rows, has {n}")
    dtype = next(proposal.parameters()).dtype
    z = arm["z"][:batch_size].to(device=device, dtype=dtype)
    lang = grpo._batched_lang(arm["lang"], batch_size, device, dtype)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=True,
        ):
            logits = proposal.logits(z, lang)
            loss = logits.float().square().mean()
        _require(loss.requires_grad and bool(torch.isfinite(loss.detach())),
                 "strict bf16 proposal loss is not finite/differentiable")
        loss.backward()
        torch.cuda.synchronize(device)
    warning_messages = _require_no_nondeterminism_warning(
        caught, label="bf16 Flash-eligible proposal backward",
    )
    grad_norm, missing, nonfinite = grpo._proposal_grad_health(proposal)
    _require(not missing, f"bf16 proposal parameters missing gradients: {missing[:8]}")
    _require(not nonfinite,
             f"bf16 proposal gradients are nonfinite: {nonfinite[:8]}")
    _require(math.isfinite(grad_norm) and grad_norm > 0.0,
             f"bf16 proposal gradient norm is invalid: {grad_norm}")
    after = grpo.proposal_module_digest(proposal.state_dict())
    _require(after == before,
             "bf16 deterministic backward mutated proposal parameters")
    proposal.zero_grad(set_to_none=True)
    return {
        "passed": True,
        "batch_size": batch_size,
        "input_dtype": str(dtype),
        "autocast_dtype": "torch.bfloat16",
        "logits_dtype": str(logits.dtype),
        "loss": float(loss.detach()),
        "proposal_grad_norm": float(grad_norm),
        "missing_gradients": [],
        "nonfinite_gradients": [],
        "warnings": warning_messages,
        "proposal_digest_before": before,
        "proposal_digest_after": after,
        "strict_determinism": grpo._strict_outcome_determinism_state(),
    }


def run_probe() -> dict[str, Any]:
    _require(torch.cuda.is_available(), "ratio identity probe requires one CUDA GPU")
    _require(torch.cuda.device_count() == 1,
             f"probe requires exactly one visible GPU, saw {torch.cuda.device_count()}")
    device = torch.device("cuda:0")
    strict_determinism = grpo._configure_strict_outcome_determinism()
    geometry = grpo._configure_exact_proposal_scoring(device)
    runtime = _runtime(device)
    started = time.monotonic()
    source_identity = validated_source_identity()

    _require(grpo.START_STEP == recovery.SEED_GLOBAL_STEP == 49_666,
             "trainer/recovery START step drifted")
    _require(grpo.EXPECTED_WORLD_SIZE == 8 and grpo.EXPECTED_CONTEXTS_PER_ARM == 2,
             "rank-0 sampler geometry drifted")
    checkpoint = ROOT / CHECKPOINT_REL
    parent, parent_identity = grpo._load_authenticated_parent(checkpoint)
    grpo._assert_seed_stat(parent_identity)
    proposal = grpo._load_proposal(parent, device=device)
    proposal.eval()
    grpo._require_exact_proposal_scoring_environment(proposal, device)
    embedded_step = int(parent["global_step"])
    embedded_config_hash = str(parent["config_hash"])
    del parent
    gc.collect()

    collection = grpo.ValidatedRecoveryCollection.open(
        ROOT / COLLECTION_REL,
        checkpoint_identity=parent_identity,
        expected_split="train0",
        deep=False,
        verify_sidecars=False,
    )
    _require(collection.identity_digest == EXPECTED_COLLECTION_IDENTITY_DIGEST,
             "fixed train0 collection identity digest drifted")
    _require(collection.manifest_sha256 == EXPECTED_MANIFEST_SHA256,
             "fixed train0 manifest SHA-256 drifted")
    pl_rowwise_replay = scan_train0_pl_witness(
        proposal,
        collection,
        device=device,
    )
    group_index, visit, indices, receipt = deterministic_rank0_selection(collection)
    payload = collection.load(group_index)
    _require(payload.get("group_id") == EXPECTED_GROUP_ID,
             "deep-loaded sidecar group identity drifted")

    execution = execute_identity_backward(
        proposal,
        payload,
        indices,
        device=device,
    )
    bf16_attention_backward = execute_strict_bf16_proposal_backward(
        proposal, payload, device=device,
    )
    torch.cuda.synchronize(device)
    collection.assert_unchanged()
    grpo._assert_seed_stat(parent_identity)
    _require(validated_source_identity() == source_identity,
             "probe/config/trainer source changed during the A100 preflight")

    return {
        "format_version": 1,
        "kind": "loom_outcome_ratio_identity_probe",
        "status": "PASS",
        "source_identity": source_identity,
        "runtime": runtime,
        "parent": {
            **parent_identity,
            "embedded_global_step": embedded_step,
            "embedded_config_hash": embedded_config_hash,
        },
        "collection": {
            "path": str(collection.root),
            "split": collection.split,
            "identity_digest": collection.identity_digest,
            "manifest_sha256": collection.manifest_sha256,
            "metadata_open": "complete authenticated manifest; sidecar hashes deferred",
            "selected_sidecar": {
                "group_index": group_index,
                "group_id": receipt["group_id"],
                "path": receipt["sidecar"],
                "sha256": receipt["sha256"],
                "size": receipt["size"],
                "deep_schema_and_hash_authenticated": True,
            },
        },
        "sampler": {
            "global_step": grpo.START_STEP,
            "rank": 0,
            "world_size": grpo.EXPECTED_WORLD_SIZE,
            "visit": visit,
            "contexts_per_arm": grpo.EXPECTED_CONTEXTS_PER_ARM,
            "replan_indices": {str(key): list(value) for key, value in indices.items()},
        },
        "configured_scoring": geometry,
        "strict_determinism": strict_determinism,
        "pl_rowwise_replay": pl_rowwise_replay,
        "bf16_attention_backward": bf16_attention_backward,
        **execution,
        "wall_seconds": float(time.monotonic() - started),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = args.out.expanduser().resolve()
    if out.exists():
        print(f"OUTCOME_RATIO_IDENTITY_PROBE_FAILED: output exists: {out}", flush=True)
        return 2
    try:
        report = run_probe()
    except Exception as exc:  # noqa: BLE001
        print(
            f"OUTCOME_RATIO_IDENTITY_PROBE_FAILED: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return 2
    report["output"] = str(out)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out, encoded)
    print(encoded, end="", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
