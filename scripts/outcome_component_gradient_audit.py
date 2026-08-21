#!/usr/bin/env python3
"""Read-only eight-A100 component-gradient audit for outcome GRPO.

This is a diagnostic, not a trainer or a promotion gate.  It authenticates the
exact deployed step-49,666 parent and selected recovery sidecars, reconstructs
the existing sparse-q_action expert anchor, and measures four synchronized
proposal gradients on fixed production batches:

* terminal-outcome GRPO (weight 1);
* sparse q_action CE anchor (weight 1);
* Switch balance after its locked 1e-2 weight;
* their directly differentiated sum, before the production norm-1 clip.

The component gradients are all-reduced and divided by eight before norms and
cosines are computed.  Eight consecutive production steps are sampled in each
of the six folds.  That is 48 synchronized points and 384 distinct,
hash-authenticated recovery groups.  At each fold entry, an analytic
directional derivative on eight disjoint validation groups attributes the
negative clipped direction to each recipe-weighted component, their algebraic
sum, and the directly differentiated sum.  A fixed algebraic anchor-scale
sweep is reported without another model forward or parameter perturbation.
These are loss-gradient directions, not virtual AdamW updates.  No optimizer
update, checkpoint, candidate, evaluation, or promotion artifact can be
produced by this file.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
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
from loom.train.atomic import fsync_dir  # noqa: E402
from loom.train.determinism import set_global_seed, set_step_seed  # noqa: E402
from loom.train.loop import read_config  # noqa: E402


FORMAT_VERSION = 2
KIND = "loom_outcome_grpo_component_gradient_audit"
STATUS = "DIAGNOSTIC_COMPLETE"
EXPECTED_CONFIG_HASH = "25afdedfc9deea5e"
EXPECTED_TRAINER_SOURCE_SHA256 = (
    "d5ef53e9f2e276f17d68f80b4c081c8f09b0d89ea9a966214fc3b63387364a52"
)
EXPECTED_SEED_CHECKPOINT = "runs/r0a_deploy_s1_eval/ckpt_000049666.pt"
OUTPUT_DIR_REL = "runs/diagnostics/outcome_component_gradient_audit"
OUTPUT_NAME_PREFIX = "outcome_component_gradient_audit_v2_s49666_"
EXPECTED_WORLD_SIZE = 8
AUDIT_STEPS_PER_FOLD = 8
EXPECTED_SELECTED_GROUPS_PER_FOLD = EXPECTED_WORLD_SIZE * AUDIT_STEPS_PER_FOLD
EXPECTED_SELECTED_GROUPS = grpo.N_FOLDS * EXPECTED_SELECTED_GROUPS_PER_FOLD
DIRECTIONAL_EPSILON = 1e-6
EXPECTED_HELDOUT_DIRECTIONAL_CHECKS = grpo.N_FOLDS
# A direct combined bf16/SDPA backward and three separately executed component
# backwards accumulate dozens of shared proposal uses in different, but each
# strictly deterministic, floating-point orders.  Additivity is mathematical,
# not bitwise: A100 measured 0.427% relative residual at the exact seed.  Keep
# the direct vector authoritative for the formal preclip norm/direction and
# reject only a gross discrepancy beyond this conservative bf16 boundary.
MAX_COMPONENT_ADDITIVITY_RELATIVE_RESIDUAL = 0.02
COMPONENT_ORDER = ("grpo", "sparse_q_action_ce", "switch_balance_1e-2")
ALGEBRAIC_TOTAL = "algebraic_component_sum"
DIRECT_TOTAL = "direct_combined"
ANCHOR_SCALE_SWEEP = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
DIRECTIONAL_SIGN_CONVENTION = (
    "d_loss/d_epsilon < 0 predicts heldout surrogate benefit; "
    "d_loss/d_epsilon > 0 predicts heldout surrogate harm"
)
RAW_DOT_SIGN_CONVENTION = (
    "for a negative-gradient update, heldout_dot_training_gradient > 0 "
    "predicts heldout surrogate benefit before clipping; a value < 0 predicts "
    "harm"
)
GRADIENT_SPACE_CAVEAT = (
    "These are loss-gradient quantities, not virtual AdamW parameter updates. "
    "Directional derivatives apply reference global norm clipping but exclude "
    "AdamW moment normalization and decoupled weight decay; the bounded pilot, "
    "not this diagnostic, tests optimizer dynamics."
)
ELIGIBILITY = {
    "diagnostic_only": True,
    "training_eligible": False,
    "candidate_eligible": False,
    "evaluation_eligible": False,
    "promotion_eligible": False,
    "optimizer_updates_allowed": False,
    "optimizer_steps": 0,
    "parameter_perturbations": 0,
    "checkpoint_emitted": False,
    "candidate_emitted": False,
    "reason": (
        "component-gradient measurement cannot select a recipe, checkpoint, "
        "candidate, or SR result"
    ),
}
_AUDIT_SOURCE_FILES = (
    "configs/r0a_outcome_grpo.yaml",
    "scripts/outcome_component_gradient_audit.py",
    "scripts/outcome_component_gradient_audit.sbatch",
)


class ComponentGradientAuditError(RuntimeError):
    """An immutable input, geometry, numerical, or no-mutation check failed."""


def _require(condition: Any, message: str) -> None:
    if not bool(condition):
        raise ComponentGradientAuditError(message)


def _validate_output_path(path: str | os.PathLike[str]) -> Path:
    """Confine the one report to its dedicated non-training directory."""
    output = Path(path).expanduser().resolve()
    output_dir = (ROOT / OUTPUT_DIR_REL).resolve()
    _require(
        output.parent == output_dir,
        f"diagnostic output must be directly inside {output_dir}: {output}",
    )
    _require(output.suffix == ".json",
             "component-gradient diagnostic output must be one JSON file")
    _require(
        output.name.startswith(OUTPUT_NAME_PREFIX),
        f"format-v2 diagnostic output must start with {OUTPUT_NAME_PREFIX!r}",
    )
    _require(not output.exists(), f"refusing existing diagnostic output: {output}")
    return output


def exclusive_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish a new report without ever replacing a path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ComponentGradientAuditError(
                f"refusing existing diagnostic output: {path}"
            ) from exc
        fsync_dir(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def audit_steps() -> tuple[int, ...]:
    """Predeclared production steps: eight fixed draws in every train fold."""
    return tuple(
        grpo.START_STEP + fold * grpo.UPDATES_PER_FOLD + offset
        for fold in range(grpo.N_FOLDS)
        for offset in range(AUDIT_STEPS_PER_FOLD)
    )


def _warning_messages(caught: Sequence[warnings.WarningMessage]) -> list[str]:
    return [str(item.message) for item in caught]


def _checked_warning_messages(
    caught: Sequence[warnings.WarningMessage],
    *,
    world: int,
    label: str,
) -> list[str]:
    messages = _warning_messages(caught)
    rejected = [
        message for message in messages
        if "nondetermin" in message.lower() or "non-determin" in message.lower()
    ]
    local_error = (
        f"{label} emitted nondeterminism warnings: {rejected}" if rejected else ""
    )
    grpo._raise_if_any_rank_failed(local_error, world, label)
    gathered = _all_gather_object(messages, world)
    return [message for rank_messages in gathered for message in rank_messages]


def _source_identity() -> dict[str, Any]:
    trainer = grpo._trainer_source_identity()
    _require(
        trainer.get("sha256") == EXPECTED_TRAINER_SOURCE_SHA256,
        f"outcome trainer source closure drifted: {trainer.get('sha256')}",
    )
    return {
        "trainer": trainer,
        "audit": grpo._trainer_source_identity(ROOT, _AUDIT_SOURCE_FILES),
    }


def _runtime_local(
    *, rank: int, world: int, local_rank: int, device: torch.device,
) -> dict[str, Any]:
    _require(device.type == "cuda", "component-gradient audit requires CUDA")
    properties = torch.cuda.get_device_properties(device)
    capability = tuple(torch.cuda.get_device_capability(device))
    _require("A100" in properties.name.upper(),
             f"component-gradient audit requires A100, got {properties.name!r}")
    _require(capability == (8, 0),
             f"component-gradient audit requires capability (8,0), got {capability}")
    return {
        "rank": int(rank),
        "world_size": int(world),
        "local_rank": int(local_rank),
        "hostname": platform.node(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": properties.name,
        "gpu_capability": list(capability),
        "visible_gpu_count": torch.cuda.device_count(),
        "device": str(device),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_nnodes": int(os.environ.get("SLURM_NNODES", "1")),
    }


def _all_gather_object(value: Any, world: int) -> list[Any]:
    if world == 1:
        return [value]
    gathered: list[Any] = [None for _ in range(world)]
    torch.distributed.all_gather_object(gathered, value)
    return gathered


def _mean_scalars(
    values: Mapping[str, float], *, world: int, device: torch.device,
) -> dict[str, float]:
    keys = tuple(sorted(values))
    tensor = torch.tensor(
        [float(values[key]) for key in keys], dtype=torch.float64, device=device,
    )
    if world > 1:
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        tensor.div_(float(world))
    return {key: float(tensor[index]) for index, key in enumerate(keys)}


def _max_scalars(
    values: Mapping[str, float], *, world: int, device: torch.device,
) -> dict[str, float]:
    keys = tuple(sorted(values))
    tensor = torch.tensor(
        [float(values[key]) for key in keys], dtype=torch.float64, device=device,
    )
    if world > 1:
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
    return {key: float(tensor[index]) for index, key in enumerate(keys)}


def _sum_scalars(
    values: Mapping[str, float], *, world: int, device: torch.device,
) -> dict[str, float]:
    keys = tuple(sorted(values))
    tensor = torch.tensor(
        [float(values[key]) for key in keys], dtype=torch.float64, device=device,
    )
    if world > 1:
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return {key: float(tensor[index]) for index, key in enumerate(keys)}


def _local_gradient_vector(
    loss: Tensor,
    named_parameters: Sequence[tuple[str, nn.Parameter]],
    *,
    retain_graph: bool,
) -> tuple[Tensor, list[str]]:
    """Differentiate one component without populating ``parameter.grad``."""
    parameters = tuple(parameter for _name, parameter in named_parameters)
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )
    missing: list[str] = []
    rows: list[Tensor] = []
    for (name, parameter), gradient in zip(
        named_parameters, gradients, strict=True,
    ):
        if gradient is None:
            missing.append(name)
            rows.append(torch.zeros_like(parameter).reshape(-1))
            continue
        value = gradient.detach().float().reshape(-1)
        _require(bool(torch.isfinite(value).all()),
                 f"component gradient {name!r} contains nan/inf")
        rows.append(value)
    _require(bool(rows), "proposal has no trainable parameters")
    return torch.cat(rows), missing


def _synchronise_gradient(vector: Tensor, *, world: int) -> Tensor:
    """Production gradient geometry: all-reduce sum, then divide by world."""
    synced = vector.contiguous()
    if world > 1:
        torch.distributed.all_reduce(synced, op=torch.distributed.ReduceOp.SUM)
        synced.div_(float(world))
    _require(bool(torch.isfinite(synced).all()),
             "synchronized proposal gradient contains nan/inf")
    return synced


def _vector_norm(vector: Tensor) -> float:
    square = vector.float().square().sum(dtype=torch.float64)
    value = float(square.sqrt())
    _require(math.isfinite(value), "proposal gradient norm is nonfinite")
    return value


def _cosine(left: Tensor, right: Tensor) -> float | None:
    left_norm = _vector_norm(left)
    right_norm = _vector_norm(right)
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    dot = float((left.float() * right.float()).sum(dtype=torch.float64))
    value = dot / (left_norm * right_norm)
    # A few ulps beyond the mathematical bound are a reduction artefact.
    value = min(1.0, max(-1.0, value))
    _require(math.isfinite(value), "proposal gradient cosine is nonfinite")
    return value


def _dot(left: Tensor, right: Tensor) -> float:
    _require(left.shape == right.shape,
             f"gradient-vector shape mismatch: {left.shape} vs {right.shape}")
    value = float((left.float() * right.float()).sum(dtype=torch.float64))
    _require(math.isfinite(value), "proposal gradient dot product is nonfinite")
    return value


def measure_synchronised_component_gradients(
    proposal: nn.Module,
    losses: Mapping[str, Tensor],
    *,
    world: int,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """Measure weighted component vectors and an independent direct total.

    ``losses`` contains the three already-weighted components.  A fourth
    backward differentiates their sum directly, making the reported preclip
    norm independent of the component-vector addition used for cosines.
    """
    local_error = ""
    try:
        _require(tuple(losses) == COMPONENT_ORDER,
                 f"component loss order changed: {tuple(losses)}")
        named_parameters = tuple(
            (name, parameter) for name, parameter in proposal.named_parameters()
            if parameter.requires_grad
        )
        _require(bool(named_parameters), "proposal has no trainable parameters")
        _require(
            all(parameter.grad is None for _name, parameter in named_parameters),
            "proposal had populated gradients before read-only audit backward",
        )
        for name, loss in losses.items():
            _require(loss.ndim == 0 and loss.requires_grad,
                     f"{name} is not a differentiable scalar")
            _require(bool(torch.isfinite(loss.detach())),
                     f"{name} loss is nonfinite")
    except Exception as exc:  # noqa: BLE001
        local_error = f"{type(exc).__name__}: {exc}"
    grpo._raise_if_any_rank_failed(
        local_error, world, "component-gradient local preflight",
    )

    total_loss = sum(losses.values())
    vectors: dict[str, Tensor] = {}
    missing: dict[str, list[str]] = {}
    for name in COMPONENT_ORDER:
        local_error = ""
        try:
            vector, absent = _local_gradient_vector(
                losses[name], named_parameters, retain_graph=True,
            )
        except Exception as exc:  # noqa: BLE001
            local_error = f"{type(exc).__name__}: {exc}"
        grpo._raise_if_any_rank_failed(
            local_error, world, f"{name} local component gradient",
        )
        vectors[name] = _synchronise_gradient(vector, world=world)
        missing[name] = absent
    local_error = ""
    try:
        direct, direct_missing = _local_gradient_vector(
            total_loss, named_parameters, retain_graph=False,
        )
        _require(not direct_missing,
                 f"direct combined gradient is missing {direct_missing[:8]}")
    except Exception as exc:  # noqa: BLE001
        local_error = f"{type(exc).__name__}: {exc}"
    grpo._raise_if_any_rank_failed(
        local_error, world, "direct combined local gradient",
    )
    vectors[DIRECT_TOTAL] = _synchronise_gradient(direct, world=world)
    missing[DIRECT_TOTAL] = direct_missing
    _require(all(parameter.grad is None for _name, parameter in named_parameters),
             "autograd.grad unexpectedly populated proposal gradient buffers")

    algebraic = sum(vectors[name] for name in COMPONENT_ORDER)
    residual = vectors[DIRECT_TOTAL] - algebraic
    direct_norm = _vector_norm(vectors[DIRECT_TOTAL])
    algebraic_norm = _vector_norm(algebraic)
    residual_norm = _vector_norm(residual)
    component_norm_sum = sum(_vector_norm(vectors[name]) for name in COMPONENT_ORDER)
    relative_residual = residual_norm / max(
        component_norm_sum, torch.finfo(torch.float64).tiny,
    )
    _require(
        relative_residual <= MAX_COMPONENT_ADDITIVITY_RELATIVE_RESIDUAL,
        "direct combined gradient differs from component sum: "
        f"relative_residual={relative_residual:.6g} > "
        f"{MAX_COMPONENT_ADDITIVITY_RELATIVE_RESIDUAL}",
    )

    component_rows = {
        name: {
            "weighted_gradient_norm": _vector_norm(vectors[name]),
            "missing_parameter_gradients": missing[name],
        }
        for name in COMPONENT_ORDER
    }
    pairwise = {
        "grpo__sparse_q_action_ce": _cosine(
            vectors["grpo"], vectors["sparse_q_action_ce"],
        ),
        "grpo__switch_balance_1e-2": _cosine(
            vectors["grpo"], vectors["switch_balance_1e-2"],
        ),
        "sparse_q_action_ce__switch_balance_1e-2": _cosine(
            vectors["sparse_q_action_ce"], vectors["switch_balance_1e-2"],
        ),
    }
    clip_scale = min(1.0, grpo.GRAD_CLIP / max(
        direct_norm, torch.finfo(torch.float64).tiny,
    ))
    retained_vectors = {
        name: vectors[name] for name in COMPONENT_ORDER
    }
    retained_vectors[ALGEBRAIC_TOTAL] = algebraic
    retained_vectors[DIRECT_TOTAL] = vectors[DIRECT_TOTAL]
    result = {
        "components": component_rows,
        "pairwise_cosines": pairwise,
        "combined": {
            "direct_preclip_norm": direct_norm,
            "algebraic_preclip_norm": algebraic_norm,
            "direct_vs_algebraic_cosine": _cosine(
                vectors[DIRECT_TOTAL], algebraic,
            ),
            "algebraic_residual_norm": residual_norm,
            "algebraic_relative_residual": relative_residual,
            "finite_precision_additivity_max_relative_residual": (
                MAX_COMPONENT_ADDITIVITY_RELATIVE_RESIDUAL
            ),
            "authoritative_preclip_vector": "direct_combined",
            "production_clip_max_norm": grpo.GRAD_CLIP,
            "would_clip": direct_norm > grpo.GRAD_CLIP,
            "reference_global_clip_scale": clip_scale,
            "missing_parameter_gradients": direct_missing,
        },
        "parameter_vector_numel": int(vectors[DIRECT_TOTAL].numel()),
        "synchronization": "all_reduce_sum_then_divide_world_size",
    }
    del vectors, residual
    return result, retained_vectors


@torch.no_grad()
def authenticate_selected_contexts(
    proposal: nn.Module,
    payload: Mapping[str, Any],
    indices: Mapping[int, Sequence[int]],
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Bitwise replay the exact selected sampled-policy rows at the seed."""
    _require(set(indices) == set(range(1, recovery.GROUP_SIZE)),
             "selected context authentication requires arms 1..7")
    dtype = next(proposal.parameters()).dtype
    max_logprob_error = 0.0
    max_coeff_error = 0.0
    atoms = 0
    for arm_index in range(1, recovery.GROUP_SIZE):
        arm = payload["arms"][arm_index]
        index = torch.tensor(tuple(indices[arm_index]), dtype=torch.int64)
        z = arm["z"].index_select(0, index).to(
            device=device, dtype=dtype, non_blocking=True,
        )
        order = arm["ordered_support"].index_select(0, index).to(device=device)
        old = arm["old_logprob"].index_select(0, index).to(device=device).float()
        stored_coeff = arm["coeff"].index_select(0, index).to(device=device).float()
        lang = grpo._batched_lang(arm["lang"], int(index.numel()), device, dtype)
        current, logits = grpo.stored_order_logprob(proposal, z, lang, order)
        coeff = grpo.weights_from_logits(
            logits.float(), order.to(torch.int64), logits.shape[-1],
        ).float()
        logprob_error = float((current.float() - old).abs().max())
        coeff_error = float((coeff - stored_coeff).abs().max())
        max_logprob_error = max(max_logprob_error, logprob_error)
        max_coeff_error = max(max_coeff_error, coeff_error)
        _require(torch.equal(current.float(), old),
                 f"selected old-logprob replay differs in arm {arm_index}")
        _require(torch.equal(coeff, stored_coeff),
                 f"selected coefficient replay differs in arm {arm_index}")
        atoms += int(index.numel())
    return {
        "atoms": atoms,
        "max_abs_old_logprob_error": max_logprob_error,
        "max_abs_coeff_error": max_coeff_error,
        "all_exact": max_logprob_error == 0.0 and max_coeff_error == 0.0,
    }


def require_exact_b1_metric_flags(metrics: Mapping[str, float]) -> dict[str, Any]:
    """Fail if the production sampled loss reports any non-B1 scoring path."""
    actual = {
        "proposal_scoring_batch_size": int(
            metrics.get("proposal_scoring_batch_size", -1)
        ),
        "proposal_scoring_autocast": bool(
            metrics.get("proposal_scoring_autocast", 1.0)
        ),
        "proposal_scoring_cuda_matmul_tf32": bool(
            metrics.get("proposal_scoring_cuda_matmul_tf32", 1.0)
        ),
        "proposal_scoring_cudnn_tf32": bool(
            metrics.get("proposal_scoring_cudnn_tf32", 1.0)
        ),
    }
    expected = {
        "proposal_scoring_batch_size": 1,
        "proposal_scoring_autocast": False,
        "proposal_scoring_cuda_matmul_tf32": False,
        "proposal_scoring_cudnn_tf32": False,
    }
    _require(actual == expected,
             f"sampled loss did not report exact B1 scoring flags: {actual}")
    return actual


def heldout_group_selection(
    collection: grpo.ValidatedRecoveryCollection,
    *,
    fold: int,
    rank: int,
    world: int,
    seed: int,
    contexts_per_arm: int,
) -> tuple[int, dict[int, tuple[int, ...]]]:
    """Choose one of 48 disjoint validation groups and fixed arm contexts."""
    _require(collection.split == "validation",
             f"heldout directional split is {collection.split!r}, not validation")
    _require(0 <= fold < grpo.N_FOLDS and 0 <= rank < world,
             f"invalid heldout fold/rank geometry: {fold}/{rank}/{world}")
    informative = collection.informative_indices()
    needed = grpo.N_FOLDS * world
    _require(len(informative) >= needed,
             f"validation has {len(informative)} informative groups, needs {needed}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(grpo._stable_seed(
        "component-gradient-heldout-groups",
        seed,
        collection.identity_digest,
        world,
    ))
    order = torch.randperm(len(informative), generator=generator).tolist()
    group_index = int(informative[order[fold * world + rank]])
    receipt = collection.receipts[group_index]
    replans: dict[int, tuple[int, ...]] = {}
    for arm in range(1, recovery.GROUP_SIZE):
        n = int(receipt["n_replans_by_arm"][arm])
        _require(n >= contexts_per_arm > 0,
                 f"validation group {group_index} arm {arm} has only {n} replans")
        arm_generator = torch.Generator(device="cpu")
        arm_generator.manual_seed(grpo._stable_seed(
            "component-gradient-heldout-replans",
            seed,
            collection.identity_digest,
            fold,
            group_index,
            arm,
        ))
        permutation = torch.randperm(n, generator=arm_generator).tolist()
        replans[arm] = tuple(int(value) for value in permutation[:contexts_per_arm])
    return group_index, replans


def _directional_interpretation(derivative: float) -> str:
    _require(math.isfinite(derivative), "directional derivative is nonfinite")
    return (
        "immediate_first_order_harm" if derivative > 0.0
        else "immediate_first_order_benefit" if derivative < 0.0
        else "first_order_flat"
    )


def _directional_row_from_norm_and_dot(
    *,
    training_gradient_norm: float,
    heldout_dot_training_gradient: float,
    heldout_vs_training_gradient_cosine: float | None,
    epsilon: float,
) -> dict[str, Any]:
    """Describe ``-clip(g, 1)`` using scalar synchronized-vector evidence."""
    _require(
        math.isfinite(training_gradient_norm) and training_gradient_norm >= 0.0,
        f"invalid training gradient norm: {training_gradient_norm}",
    )
    _require(math.isfinite(heldout_dot_training_gradient),
             "heldout/training gradient dot is nonfinite")
    _require(math.isfinite(epsilon) and epsilon > 0.0,
             f"directional epsilon must be positive and finite, got {epsilon}")
    if heldout_vs_training_gradient_cosine is not None:
        _require(
            math.isfinite(heldout_vs_training_gradient_cosine)
            and -1.0 <= heldout_vs_training_gradient_cosine <= 1.0,
            "heldout/training gradient cosine is invalid",
        )
    clip_scale = min(
        1.0,
        grpo.GRAD_CLIP / max(
            training_gradient_norm, torch.finfo(torch.float64).tiny,
        ),
    )
    direction_norm = training_gradient_norm * clip_scale
    _require(direction_norm <= grpo.GRAD_CLIP * (1.0 + 1e-12),
             f"bounded diagnostic direction has norm {direction_norm}")
    derivative = -heldout_dot_training_gradient * clip_scale
    return {
        "training_gradient_preclip_norm": training_gradient_norm,
        "heldout_gradient_dot_training_gradient": (
            heldout_dot_training_gradient
        ),
        "heldout_vs_training_gradient_cosine": (
            heldout_vs_training_gradient_cosine
        ),
        "reference_global_clip_max_norm": grpo.GRAD_CLIP,
        "reference_global_clip_scale": clip_scale,
        "negative_clipped_direction_l2_norm": direction_norm,
        "d_heldout_loss_d_epsilon": derivative,
        "epsilon": float(epsilon),
        "predicted_first_order_heldout_loss_delta": epsilon * derivative,
        "interpretation": _directional_interpretation(derivative),
    }


def _gradient_directional_row(
    heldout_gradient: Tensor,
    training_gradient: Tensor,
    *,
    epsilon: float,
) -> dict[str, Any]:
    _require(heldout_gradient.shape == training_gradient.shape,
             "heldout/training proposal gradient geometry differs")
    return _directional_row_from_norm_and_dot(
        training_gradient_norm=_vector_norm(training_gradient),
        heldout_dot_training_gradient=_dot(
            heldout_gradient, training_gradient,
        ),
        heldout_vs_training_gradient_cosine=_cosine(
            heldout_gradient, training_gradient,
        ),
        epsilon=epsilon,
    )


def _counterfactual_anchor_scale_rows(
    heldout_gradient: Tensor,
    component_gradients: Mapping[str, Tensor],
    *,
    epsilon: float,
) -> list[dict[str, Any]]:
    """Algebraically reweight the recipe anchor from synchronized dot products."""
    _require(tuple(component_gradients) == COMPONENT_ORDER,
             f"counterfactual component order changed: {tuple(component_gradients)}")
    for vector in component_gradients.values():
        _require(vector.shape == heldout_gradient.shape,
                 "counterfactual component gradient geometry differs")

    heldout_dots = {
        name: _dot(heldout_gradient, component_gradients[name])
        for name in COMPONENT_ORDER
    }
    gram = {
        (left, right): _dot(
            component_gradients[left], component_gradients[right],
        )
        for left in COMPONENT_ORDER
        for right in COMPONENT_ORDER
    }
    heldout_norm = _vector_norm(heldout_gradient)
    rows: list[dict[str, Any]] = []
    for anchor_scale in ANCHOR_SCALE_SWEEP:
        coefficients = {
            "grpo": 1.0,
            "sparse_q_action_ce": float(anchor_scale),
            "switch_balance_1e-2": 1.0,
        }
        norm_square = sum(
            coefficients[left] * coefficients[right] * gram[(left, right)]
            for left in COMPONENT_ORDER
            for right in COMPONENT_ORDER
        )
        numerical_scale = sum(
            abs(coefficients[left] * coefficients[right] * gram[(left, right)])
            for left in COMPONENT_ORDER
            for right in COMPONENT_ORDER
        )
        _require(
            norm_square >= -1e-12 * max(1.0, numerical_scale),
            f"counterfactual Gram norm square is negative: {norm_square}",
        )
        norm = math.sqrt(max(0.0, norm_square))
        heldout_dot = sum(
            coefficients[name] * heldout_dots[name]
            for name in COMPONENT_ORDER
        )
        cosine = None
        if norm > 0.0 and heldout_norm > 0.0:
            cosine = min(1.0, max(-1.0, heldout_dot / (heldout_norm * norm)))
        rows.append({
            "anchor_scale": float(anchor_scale),
            "effective_sparse_q_action_ce_weight": float(anchor_scale),
            "gradient_definition": (
                "g_grpo + anchor_scale*g_sparse_q_action_ce + "
                "g_switch_balance_1e-2"
            ),
            **_directional_row_from_norm_and_dot(
                training_gradient_norm=norm,
                heldout_dot_training_gradient=heldout_dot,
                heldout_vs_training_gradient_cosine=cosine,
                epsilon=epsilon,
            ),
        })
    return rows


def measure_heldout_directional_derivative(
    proposal: nn.Module,
    heldout_loss: Tensor,
    training_gradients: Mapping[str, Tensor],
    *,
    world: int,
    epsilon: float = DIRECTIONAL_EPSILON,
) -> dict[str, Any]:
    """Attribute one analytic heldout gradient to synchronized train vectors.

    Each reported direction applies the recipe's reference global norm-1 clip.
    The fixed anchor-scale rows use only component Gram and heldout-dot scalars;
    they trigger no additional forward, backward, update, or parameter change.
    They remain loss-gradient diagnostics rather than virtual AdamW updates.
    """
    local_error = ""
    try:
        _require(math.isfinite(epsilon) and epsilon > 0.0,
                 f"directional epsilon must be positive and finite, got {epsilon}")
        expected = {*COMPONENT_ORDER, ALGEBRAIC_TOTAL, DIRECT_TOTAL}
        _require(set(training_gradients) == expected,
                 f"training gradient vector set changed: {set(training_gradients)}")
        named_parameters = tuple(
            (name, parameter) for name, parameter in proposal.named_parameters()
            if parameter.requires_grad
        )
        _require(heldout_loss.ndim == 0 and heldout_loss.requires_grad,
                 "heldout directional objective is not a differentiable scalar")
        local, missing = _local_gradient_vector(
            heldout_loss, named_parameters, retain_graph=False,
        )
    except Exception as exc:  # noqa: BLE001
        local_error = f"{type(exc).__name__}: {exc}"
    grpo._raise_if_any_rank_failed(
        local_error, world, "heldout local directional gradient",
    )
    heldout_gradient = _synchronise_gradient(local, world=world)
    _require(
        all(vector.shape == heldout_gradient.shape
            for vector in training_gradients.values()),
        "heldout/training proposal gradient geometry differs",
    )

    component_rows = {
        name: _gradient_directional_row(
            heldout_gradient, training_gradients[name], epsilon=epsilon,
        )
        for name in COMPONENT_ORDER
    }
    algebraic_row = _gradient_directional_row(
        heldout_gradient, training_gradients[ALGEBRAIC_TOTAL], epsilon=epsilon,
    )
    direct_row = _gradient_directional_row(
        heldout_gradient, training_gradients[DIRECT_TOTAL], epsilon=epsilon,
    )
    counterfactual_rows = _counterfactual_anchor_scale_rows(
        heldout_gradient,
        {name: training_gradients[name] for name in COMPONENT_ORDER},
        epsilon=epsilon,
    )

    residual = (
        training_gradients[DIRECT_TOTAL]
        - training_gradients[ALGEBRAIC_TOTAL]
    )
    heldout_norm = _vector_norm(heldout_gradient)
    residual_norm = _vector_norm(residual)
    residual_dot = _dot(heldout_gradient, residual)
    cauchy_bound = heldout_norm * residual_norm
    _require(
        abs(residual_dot) <= cauchy_bound * (1.0 + 1e-12) + 1e-12,
        "direct/algebraic heldout-dot residual violates Cauchy bound",
    )

    attribution = {
        "scope": "loss_gradient_space_only",
        "scope_caveat": GRADIENT_SPACE_CAVEAT,
        "training_and_heldout_gradient_synchronization": (
            "all_reduce_sum_then_divide_world_size"
        ),
        "component_vectors_are_recipe_weighted": True,
        "raw_dot_sign_convention": RAW_DOT_SIGN_CONVENTION,
        "clipped_derivative_sign_convention": DIRECTIONAL_SIGN_CONVENTION,
        "no_additional_forward_or_backward_for_attribution": True,
        "recipe_weighted_components": component_rows,
        ALGEBRAIC_TOTAL: algebraic_row,
        DIRECT_TOTAL: direct_row,
        "current_direct_vs_algebraic": {
            "direct_is_authoritative": True,
            "direct_minus_algebraic_gradient_norm": residual_norm,
            "heldout_dot_direct_minus_algebraic_gradient": residual_dot,
            "heldout_dot_cauchy_abs_bound": cauchy_bound,
            "cauchy_bound_passed": True,
            "separate_dot_difference_direct_minus_algebraic": (
                direct_row["heldout_gradient_dot_training_gradient"]
                - algebraic_row["heldout_gradient_dot_training_gradient"]
            ),
            "clipped_derivative_difference_direct_minus_algebraic": (
                direct_row["d_heldout_loss_d_epsilon"]
                - algebraic_row["d_heldout_loss_d_epsilon"]
            ),
        },
        "counterfactual_anchor_scale_sweep": {
            "authority": "finite_precision_algebraic_gradient_space_estimate",
            "not_a_recipe_selection_or_promotion_gate": True,
            "anchor_component_at_recipe_weight_one": True,
            "switch_balance_component_already_weighted_1e-2": True,
            "fixed_anchor_scales": list(ANCHOR_SCALE_SWEEP),
            "norm_source": "synchronized_component_gram_matrix",
            "heldout_dot_source": "linearity_of_synchronized_component_dots",
            "direct_backward_authoritative_only_at_current_recipe": True,
            "rows": counterfactual_rows,
        },
    }

    result = {
        "method": "analytic_gradient_dot_bounded_direction",
        "heldout_objective": "validation_negative_clipped_surrogate_loss",
        "direction": "negative_training_combined_gradient_after_global_norm1_clip",
        "sign_convention": DIRECTIONAL_SIGN_CONVENTION,
        "training_combined_preclip_norm": direct_row[
            "training_gradient_preclip_norm"
        ],
        "direction_l2_norm": direct_row[
            "negative_clipped_direction_l2_norm"
        ],
        "direction_bound": grpo.GRAD_CLIP,
        "heldout_gradient_norm": heldout_norm,
        "heldout_vs_training_combined_cosine": direct_row[
            "heldout_vs_training_gradient_cosine"
        ],
        "d_heldout_loss_d_epsilon": direct_row[
            "d_heldout_loss_d_epsilon"
        ],
        "epsilon": float(epsilon),
        "predicted_first_order_heldout_loss_delta": direct_row[
            "predicted_first_order_heldout_loss_delta"
        ],
        "interpretation": direct_row["interpretation"],
        "gradient_space_attribution": attribution,
        "heldout_missing_parameter_gradients": missing,
        "parameter_perturbations": 0,
        "optimizer_steps": 0,
    }
    del heldout_gradient, residual
    return result


def prepared_anchor_digest(
    prepared: tuple[list[Tensor], Tensor, list[Tensor], str],
) -> dict[str, Any]:
    """Digest the frozen belief/teacher tensors consumed by one anchor loss."""
    beliefs, lang, targets, embodiment = prepared
    tensors: dict[str, Tensor] = {"lang": lang}
    tensors.update({f"belief.{index}": value for index, value in enumerate(beliefs)})
    tensors.update({f"target.{index}": value for index, value in enumerate(targets)})
    digest = grpo.model_state_digest(tensors)
    return {**digest, "embodiment": str(embodiment), "horizons": len(targets)}


def _prepare_anchor_loss(
    anchor: grpo.ExpertAnchor,
    *,
    step: int,
    device: torch.device,
) -> tuple[Tensor, dict[str, float], dict[str, Any]]:
    """Reproduce the formal anchor autocast while retaining batch evidence."""
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        if int(step) in anchor._cache:
            prepared = anchor._cache[int(step)]
        else:
            prepared = anchor._prepare(int(step))
            anchor._cache[int(step)] = prepared
        loss, metrics = anchor.loss(int(step))
    return loss, metrics, prepared_anchor_digest(prepared)


def _summary(values: Sequence[float | None]) -> dict[str, Any]:
    finite = [float(value) for value in values if value is not None]
    _require(all(math.isfinite(value) for value in finite),
             "summary contains nan/inf")
    if not finite:
        return {"n": 0, "min": None, "p05": None, "median": None,
                "mean": None, "p95": None, "max": None}
    tensor = torch.tensor(finite, dtype=torch.float64)
    return {
        "n": int(tensor.numel()),
        "min": float(tensor.min()),
        "p05": float(torch.quantile(tensor, 0.05, interpolation="linear")),
        "median": float(tensor.median()),
        "mean": float(tensor.mean()),
        "p95": float(torch.quantile(tensor, 0.95, interpolation="linear")),
        "max": float(tensor.max()),
        "negative_fraction": float((tensor < 0).double().mean()),
    }


def summarise_directional_attributions(
    checks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize per-fold heldout dots and clipped gradient derivatives."""
    _require(bool(checks), "cannot summarize zero directional checks")

    def row_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        derivatives = [
            float(row["d_heldout_loss_d_epsilon"]) for row in rows
        ]
        return {
            "heldout_gradient_dot_training_gradient": _summary([
                row["heldout_gradient_dot_training_gradient"] for row in rows
            ]),
            "d_heldout_loss_d_epsilon": _summary(derivatives),
            "immediate_first_order_harm_count": sum(
                value > 0.0 for value in derivatives
            ),
            "immediate_first_order_benefit_count": sum(
                value < 0.0 for value in derivatives
            ),
            "first_order_flat_count": sum(
                value == 0.0 for value in derivatives
            ),
        }

    attributions = [
        row["gradient_space_attribution"] for row in checks
    ]
    component_summary = {
        name: row_summary([
            row["recipe_weighted_components"][name]
            for row in attributions
        ])
        for name in COMPONENT_ORDER
    }
    alpha_summary: list[dict[str, Any]] = []
    for index, anchor_scale in enumerate(ANCHOR_SCALE_SWEEP):
        rows = [
            row["counterfactual_anchor_scale_sweep"]["rows"][index]
            for row in attributions
        ]
        _require(
            all(float(row["anchor_scale"]) == float(anchor_scale) for row in rows),
            f"directional anchor-scale row {index} changed",
        )
        alpha_summary.append({
            "anchor_scale": float(anchor_scale),
            **row_summary(rows),
        })
    return {
        "scope": "loss_gradient_space_only",
        "scope_caveat": GRADIENT_SPACE_CAVEAT,
        "recipe_weighted_components": component_summary,
        ALGEBRAIC_TOTAL: row_summary([
            row[ALGEBRAIC_TOTAL] for row in attributions
        ]),
        DIRECT_TOTAL: row_summary([
            row[DIRECT_TOTAL] for row in attributions
        ]),
        "counterfactual_anchor_scale_sweep": alpha_summary,
    }


def summarise_points(points: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(bool(points), "cannot summarize zero audit points")

    def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "n_synchronized_points": len(rows),
            "component_weighted_gradient_norms": {
                name: _summary([
                    row["gradients"]["components"][name]["weighted_gradient_norm"]
                    for row in rows
                ])
                for name in COMPONENT_ORDER
            },
            "pairwise_cosines": {
                name: _summary([
                    row["gradients"]["pairwise_cosines"][name] for row in rows
                ])
                for name in (
                    "grpo__sparse_q_action_ce",
                    "grpo__switch_balance_1e-2",
                    "sparse_q_action_ce__switch_balance_1e-2",
                )
            },
            "combined_direct_preclip_norm": _summary([
                row["gradients"]["combined"]["direct_preclip_norm"]
                for row in rows
            ]),
            "combined_would_clip_fraction": sum(
                bool(row["gradients"]["combined"]["would_clip"]) for row in rows
            ) / len(rows),
        }

    by_fold = {
        str(fold): summarize_rows([
            row for row in points if int(row["fold"]) == fold
        ])
        for fold in range(grpo.N_FOLDS)
    }
    return {"overall": summarize_rows(points), "by_fold": by_fold}


def _validate_config(config_path: Path) -> tuple[dict[str, Any], str]:
    cfg = read_config(config_path)
    grpo.validate_recipe_config(cfg)
    config_hash = grpo._config_hash(cfg)
    _require(config_hash == EXPECTED_CONFIG_HASH,
             f"outcome recipe hash drifted: {config_hash}")
    _require(
        str(cfg["outcome_grpo"]["seed_checkpoint"]) == EXPECTED_SEED_CHECKPOINT,
        "outcome recipe no longer names the exact step-49,666 seed checkpoint",
    )
    _require(int(cfg["outcome_grpo"]["world_size"]) == EXPECTED_WORLD_SIZE,
             "outcome recipe world size is no longer eight")
    _require(float(cfg["losses"]["grpo"]["weight"]) == 1.0,
             "GRPO audit weight drifted")
    _require(float(cfg["losses"]["proposal"]["weight"]) == 1.0,
             "sparse q_action CE audit weight drifted")
    _require(float(cfg["losses"]["balance"]["weight"])
             == grpo.SWITCH_BALANCE_WEIGHT == 1e-2,
             "Switch balance audit weight drifted")
    return cfg, config_hash


def _build_production_sampler(
    collections: Sequence[grpo.ValidatedRecoveryCollection],
    *,
    seed: int,
    rank: int,
    world: int,
    contexts_per_arm: int,
) -> grpo.DeterministicOutcomeSampler:
    _require(len(collections) == grpo.N_FOLDS,
             "component audit requires all six recovery folds")
    informative = [collection.informative_indices() for collection in collections]
    _require(
        all(len(indices) >= EXPECTED_SELECTED_GROUPS_PER_FOLD
            for indices in informative),
        "a fold cannot supply 64 distinct fixed audit groups",
    )
    return grpo.DeterministicOutcomeSampler(
        informative,
        seed=seed,
        rank=rank,
        world_size=world,
        start_step=grpo.START_STEP,
        updates_per_fold=grpo.UPDATES_PER_FOLD,
        contexts_per_arm=contexts_per_arm,
        identity_digests=[collection.identity_digest for collection in collections],
    )


def recheck_rank_selected_sidecars(
    collections: Sequence[grpo.ValidatedRecoveryCollection],
    validation: grpo.ValidatedRecoveryCollection,
    selections: Sequence[Mapping[str, Any]],
    *,
    rank: int,
) -> dict[str, Any]:
    """Post-use SHA/stat check of every sidecar consumed by this rank."""
    local = [row for row in selections if int(row["rank"]) == rank]
    expected = len(audit_steps()) + grpo.N_FOLDS
    _require(len(local) == expected,
             f"rank {rank} selected {len(local)} sidecars, expected {expected}")
    paths: set[str] = set()
    total_bytes = 0
    closure = hashlib.sha256()
    for row in sorted(
        local,
        key=lambda value: (
            str(value["split"]), int(value["group_index"]),
        ),
    ):
        split = str(row["split"])
        if split == "validation":
            collection = validation
        else:
            fold = int(row["fold"])
            _require(split == f"train{fold}",
                     f"selected sidecar split/fold mismatch: {split}/{fold}")
            collection = collections[fold]
        group_index = int(row["group_index"])
        receipt = collection.receipts[group_index]
        _require(str(receipt["group_id"]) == str(row["group_id"])
                 and str(receipt["sidecar"]) == str(row["sidecar"])
                 and str(receipt["sha256"]) == str(row["sidecar_sha256"])
                 and int(receipt["size"]) == int(row["sidecar_size"]),
                 f"selected sidecar receipt changed after use: {split}/{group_index}")
        path = collection._resolved_sidecar(collection.root, receipt)
        path_text = str(path)
        _require(path_text not in paths,
                 f"rank {rank} selected duplicate sidecar path: {path}")
        paths.add(path_text)
        before = path.stat()
        _require(int(before.st_size) == int(receipt["size"]),
                 f"selected sidecar size changed after use: {path}")
        digest = recovery.sha256_file(path)
        after = path.stat()
        _require(
            int(after.st_size) == int(before.st_size)
            and int(after.st_mtime_ns) == int(before.st_mtime_ns),
            f"selected sidecar stat changed during post-use hash: {path}",
        )
        _require(digest == str(receipt["sha256"]),
                 f"selected sidecar SHA-256 changed after use: {path}")
        row_identity = json.dumps({
            "split": split,
            "group_index": group_index,
            "group_id": str(receipt["group_id"]),
            "path": path_text,
            "size": int(receipt["size"]),
            "sha256": digest,
        }, sort_keys=True, separators=(",", ":")).encode("utf-8")
        closure.update(row_identity + b"\0")
        total_bytes += int(receipt["size"])
    return {
        "rank": rank,
        "selected_sidecars": len(local),
        "selected_bytes": total_bytes,
        "post_use_closure_sha256": closure.hexdigest(),
        "post_use_size_sha256_and_stable_stat": True,
        "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _authenticate_parent_once(
    checkpoint: Path, *, rank: int, world: int,
) -> dict[str, Any]:
    box: list[Any] = [None]
    if rank == 0:
        try:
            box[0] = {
                "ok": True,
                "identity": recovery.authenticate_seed_checkpoint(checkpoint),
            }
        except Exception as exc:  # noqa: BLE001
            box[0] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if world > 1:
        torch.distributed.broadcast_object_list(box, src=0)
    _require(bool(box[0].get("ok")),
             f"seed checkpoint authentication failed: {box[0].get('error')}")
    return dict(box[0]["identity"])


def run_audit(
    *,
    config_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
) -> dict[str, Any] | None:
    """Run the fixed read-only audit; rank zero returns the JSON payload."""
    rank, world, local_rank, device = grpo._dist_info()
    _require(world == EXPECTED_WORLD_SIZE,
             f"component-gradient audit requires world=8, got {world}")
    local_error = ""
    try:
        output = _validate_output_path(output_path)
    except Exception as exc:  # noqa: BLE001
        local_error = f"{type(exc).__name__}: {exc}"
    grpo._raise_if_any_rank_failed(local_error, world, "exclusive diagnostic output")

    started = time.monotonic()
    local_error = ""
    try:
        strict = grpo._configure_strict_outcome_determinism()
        scoring = grpo._configure_exact_proposal_scoring(device)
        runtime_local = _runtime_local(
            rank=rank, world=world, local_rank=local_rank, device=device,
        )
    except Exception as exc:  # noqa: BLE001
        local_error = f"{type(exc).__name__}: {exc}"
    grpo._raise_if_any_rank_failed(local_error, world, "A100/scoring geometry")
    runtimes = _all_gather_object(runtime_local, world)
    _require(sorted(int(row["rank"]) for row in runtimes) == list(range(world)),
             "distributed rank identities are incomplete")
    _require(len({(row["hostname"], row["local_rank"]) for row in runtimes}) == world,
             "eight ranks did not map to distinct local A100 devices")
    _require(len({row["hostname"] for row in runtimes}) == 1,
             "component audit requires first-production one-node geometry")

    config_resolved = Path(config_path).expanduser().resolve()
    local_error = ""
    try:
        cfg, config_hash = _validate_config(config_resolved)
        source_identity = _source_identity()
    except Exception as exc:  # noqa: BLE001
        local_error = f"{type(exc).__name__}: {exc}"
    grpo._raise_if_any_rank_failed(local_error, world, "recipe/source validation")
    source_rows = _all_gather_object(source_identity, world)
    _require(all(row == source_rows[0] for row in source_rows),
             "audit/trainer source identity differs across ranks")

    seed = int(cfg["run"].get("seed", grpo.TRAIN_SEED))
    set_global_seed(seed, rank)
    checkpoint = ROOT / EXPECTED_SEED_CHECKPOINT
    parent_identity = _authenticate_parent_once(checkpoint, rank=rank, world=world)
    local_error = ""
    try:
        grpo._assert_seed_stat(parent_identity)
        parent = grpo._load_parent_from_identity(parent_identity)
        grpo._assert_seed_stat(parent_identity)
        proposal = grpo._load_proposal(parent, device=device)
        proposal.eval()
        grpo._require_exact_proposal_scoring_environment(proposal, device)
        proposal_digest_before = grpo.proposal_module_digest(proposal.state_dict())
        parent_proposal_digest = grpo.proposal_model_digest(parent["model"])
        _require(proposal_digest_before == parent_proposal_digest,
                 "runtime proposal differs from authenticated parent proposal")
        anchor = grpo.ExpertAnchor.from_parent(
            parent,
            proposal,
            trainer_cfg=cfg,
            device=device,
            rank=rank,
            world_size=world,
        )
        anchor_preflight = anchor.preflight(grpo.START_STEP)
        wrapper = grpo._ProposalOnly(proposal)
        optimizer = grpo.build_optimizer(
            wrapper,
            lr=float(cfg["optim"]["lr"]),
            weight_decay=float(cfg["optim"]["weight_decay"]),
            betas=tuple(float(value) for value in cfg["optim"]["betas"]),
            lr_scales={"proposal": float(cfg["optim"]["lr_scales"]["proposal"])},
            module_names=["proposal"],
        )
        _require(len(optimizer.state) == 0,
                 "diagnostic optimizer did not start with empty state")
    except Exception as exc:  # noqa: BLE001
        local_error = f"{type(exc).__name__}: {exc}"
    grpo._raise_if_any_rank_failed(local_error, world, "seed/anchor construction")
    anchor_preflight_rows = _all_gather_object(anchor_preflight, world)
    del parent

    collections: list[grpo.ValidatedRecoveryCollection] = []
    local_error = ""
    try:
        for spec in cfg["outcome_grpo"]["folds"]:
            collections.append(grpo.ValidatedRecoveryCollection.open(
                ROOT / str(spec["path"]),
                checkpoint_identity=parent_identity,
                expected_split=str(spec["split"]),
                deep=False,
                verify_sidecars=False,
            ))
        validation_spec = cfg["outcome_grpo"]["validation"]
        validation = grpo.ValidatedRecoveryCollection.open(
            ROOT / str(validation_spec["path"]),
            checkpoint_identity=parent_identity,
            expected_split=str(validation_spec["split"]),
            deep=False,
            verify_sidecars=False,
        )
        sampler = _build_production_sampler(
            collections,
            seed=seed,
            rank=rank,
            world=world,
            contexts_per_arm=int(cfg["outcome_grpo"]["contexts_per_arm"]),
        )
    except Exception as exc:  # noqa: BLE001
        local_error = f"{type(exc).__name__}: {exc}"
    grpo._raise_if_any_rank_failed(local_error, world, "recovery metadata authentication")

    points: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    directional_checks: list[dict[str, Any]] = []
    heldout_selection_rows: list[dict[str, Any]] = []
    non_determinism_warnings: list[str] = []
    for point_index, step in enumerate(audit_steps()):
        set_step_seed(seed, step, rank)
        local_error = ""
        try:
            fold, group_index, visit = sampler.group_at(step)
            expected_fold = (step - grpo.START_STEP) // grpo.UPDATES_PER_FOLD
            _require(fold == expected_fold, "fixed audit step resolved to wrong fold")
            collection = collections[fold]
            receipt = collection.receipts[group_index]
            indices = sampler.replans_at(
                step, receipt["n_replans_by_arm"],
            )
            payload = collection.load(group_index)
            selected_auth = authenticate_selected_contexts(
                proposal, payload, indices, device=device,
            )
            grpo_loss, balance_loss, ratio_metrics = grpo.sampled_group_losses(
                proposal, payload, indices, device=device,
            )
            exact_b1_flags = require_exact_b1_metric_flags(ratio_metrics)
            local_ratio_identity = grpo._require_initial_behavior_ratio_identity(
                ratio_metrics, device=device,
            )
            anchor_loss, anchor_metrics, anchor_digest = _prepare_anchor_loss(
                anchor, step=step, device=device,
            )
            losses = {
                "grpo": grpo_loss,
                "sparse_q_action_ce": anchor_loss,
                "switch_balance_1e-2": grpo.SWITCH_BALANCE_WEIGHT * balance_loss,
            }
        except Exception as exc:  # noqa: BLE001
            local_error = f"{type(exc).__name__}: {exc}"
        grpo._raise_if_any_rank_failed(
            local_error, world, f"audit point {point_index} forward/authentication",
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            local_error = ""
            try:
                gradients, training_gradient_vectors = \
                    measure_synchronised_component_gradients(
                    proposal, losses, world=world,
                )
                torch.cuda.synchronize(device)
                _require(not anchor.unexpected_gradients(),
                         "frozen estimator/q_action received gradients")
            except Exception as exc:  # noqa: BLE001
                local_error = f"{type(exc).__name__}: {exc}"
        grpo._raise_if_any_rank_failed(
            local_error, world, f"audit point {point_index} backward",
        )
        non_determinism_warnings.extend(_checked_warning_messages(
            caught, world=world, label=f"audit point {point_index} backward",
        ))

        # One disjoint validation minibatch per fold measures whether an
        # infinitesimal move along this fold-entry training direction would
        # immediately lower or raise the heldout negative surrogate loss. The 48
        # validation groups are selected once without replacement.
        if (step - grpo.START_STEP) % grpo.UPDATES_PER_FOLD == 0:
            local_error = ""
            try:
                heldout_group, heldout_indices = heldout_group_selection(
                    validation,
                    fold=fold,
                    rank=rank,
                    world=world,
                    seed=seed,
                    contexts_per_arm=int(cfg["outcome_grpo"]["contexts_per_arm"]),
                )
                heldout_receipt = validation.receipts[heldout_group]
                heldout_payload = validation.load(heldout_group)
                heldout_auth = authenticate_selected_contexts(
                    proposal,
                    heldout_payload,
                    heldout_indices,
                    device=device,
                )
                heldout_loss, heldout_balance, heldout_ratio_metrics = (
                    grpo.sampled_group_losses(
                        proposal,
                        heldout_payload,
                        heldout_indices,
                        device=device,
                    )
                )
                heldout_exact_b1_flags = require_exact_b1_metric_flags(
                    heldout_ratio_metrics,
                )
                heldout_local_ratio_identity = (
                    grpo._require_initial_behavior_ratio_identity(
                        heldout_ratio_metrics, device=device,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                local_error = f"{type(exc).__name__}: {exc}"
            grpo._raise_if_any_rank_failed(
                local_error, world,
                f"fold {fold} heldout directional forward/authentication",
            )

            with warnings.catch_warnings(record=True) as heldout_caught:
                warnings.simplefilter("always")
                local_error = ""
                try:
                    directional = measure_heldout_directional_derivative(
                        proposal,
                        heldout_loss,
                        training_gradient_vectors,
                        world=world,
                    )
                    torch.cuda.synchronize(device)
                except Exception as exc:  # noqa: BLE001
                    local_error = f"{type(exc).__name__}: {exc}"
            grpo._raise_if_any_rank_failed(
                local_error, world, f"fold {fold} heldout directional backward",
            )
            non_determinism_warnings.extend(_checked_warning_messages(
                heldout_caught,
                world=world,
                label=f"fold {fold} heldout directional backward",
            ))

            heldout_loss_mean = _mean_scalars({
                "validation_negative_clipped_surrogate_loss": float(
                    heldout_loss.detach()
                ),
            }, world=world, device=device)[
                "validation_negative_clipped_surrogate_loss"
            ]
            heldout_auth_max = _max_scalars({
                "max_abs_old_logprob_error": heldout_auth[
                    "max_abs_old_logprob_error"
                ],
                "max_abs_coeff_error": heldout_auth["max_abs_coeff_error"],
            }, world=world, device=device)
            heldout_auth_sum = _sum_scalars({
                "atoms": float(heldout_auth["atoms"]),
            }, world=world, device=device)
            heldout_reduced_ratio = grpo._reduce_training_metrics(
                heldout_ratio_metrics, world, device,
            )
            _require(heldout_reduced_ratio["max_abs_logratio"] == 0.0
                     and heldout_reduced_ratio["ratio_min"] == 1.0
                     and heldout_reduced_ratio["ratio_mean"] == 1.0
                     and heldout_reduced_ratio["ratio_max"] == 1.0,
                     "heldout directional seed ratios are not bitwise identity")
            _require(heldout_auth_max["max_abs_old_logprob_error"] == 0.0
                     and heldout_auth_max["max_abs_coeff_error"] == 0.0,
                     "heldout directional behavior replay is not bitwise exact")
            local_heldout_selection = {
                "rank": rank,
                "fold_check": fold,
                "split": validation.split,
                "group_index": heldout_group,
                "group_id": str(heldout_receipt["group_id"]),
                "sidecar": str(heldout_receipt["sidecar"]),
                "sidecar_sha256": str(heldout_receipt["sha256"]),
                "sidecar_size": int(heldout_receipt["size"]),
                "replan_indices": {
                    str(arm): list(heldout_indices[arm])
                    for arm in sorted(heldout_indices)
                },
                "local_ratio_identity": heldout_local_ratio_identity,
                "exact_b1_metric_flags": heldout_exact_b1_flags,
                "sidecar_hash_and_deep_schema_authenticated_at_use": True,
            }
            heldout_selections = _all_gather_object(
                local_heldout_selection, world,
            )
            _require(len({int(row["group_index"]) for row in heldout_selections})
                     == world,
                     "heldout directional ranks selected duplicate groups")
            heldout_selection_rows.extend(heldout_selections)
            if rank == 0:
                directional_checks.append({
                    "fold": fold,
                    "training_global_step": step,
                    "disjoint_split": "validation",
                    "global_mean_validation_negative_clipped_surrogate_loss": (
                        heldout_loss_mean
                    ),
                    "global_selected_context_authentication": {
                        "atoms": int(heldout_auth_sum["atoms"]),
                        **heldout_auth_max,
                        "all_exact": True,
                    },
                    "global_ratio_identity": {
                        "ratio_atoms": int(heldout_reduced_ratio["ratio_atoms"]),
                        "ratio_min": heldout_reduced_ratio["ratio_min"],
                        "ratio_mean": heldout_reduced_ratio["ratio_mean"],
                        "ratio_max": heldout_reduced_ratio["ratio_max"],
                        "max_abs_logratio": heldout_reduced_ratio[
                            "max_abs_logratio"
                        ],
                        "all_exact": True,
                    },
                    "rank_local_groups": heldout_selections,
                    **directional,
                })
            del heldout_payload, heldout_loss, heldout_balance, directional

        loss_means = _mean_scalars({
            "grpo": float(grpo_loss.detach()),
            "sparse_q_action_ce": float(anchor_loss.detach()),
            "switch_balance_raw": float(balance_loss.detach()),
            "switch_balance_1e-2": float(
                (grpo.SWITCH_BALANCE_WEIGHT * balance_loss).detach()
            ),
            "direct_combined": float(sum(losses.values()).detach()),
            "anchor_sparse_ce_unweighted": float(anchor_metrics["sparse_ce"]),
        }, world=world, device=device)
        selected_auth_max = _max_scalars({
            "max_abs_old_logprob_error": selected_auth["max_abs_old_logprob_error"],
            "max_abs_coeff_error": selected_auth["max_abs_coeff_error"],
        }, world=world, device=device)
        selected_auth_sum = _sum_scalars({
            "atoms": float(selected_auth["atoms"]),
        }, world=world, device=device)
        reduced_ratio = grpo._reduce_training_metrics(
            ratio_metrics, world, device,
        )
        _require(reduced_ratio["max_abs_logratio"] == 0.0,
                 "global selected seed ratios are not bitwise identity")
        _require(reduced_ratio["ratio_min"] == 1.0
                 and reduced_ratio["ratio_mean"] == 1.0
                 and reduced_ratio["ratio_max"] == 1.0,
                 "global selected seed ratio bounds are not exactly one")
        _require(selected_auth_max["max_abs_old_logprob_error"] == 0.0
                 and selected_auth_max["max_abs_coeff_error"] == 0.0,
                 "global selected behavior replay is not bitwise exact")

        local_selection = {
            "rank": rank,
            "global_step": step,
            "fold": fold,
            "split": collection.split,
            "group_index": group_index,
            "group_id": str(receipt["group_id"]),
            "visit": visit,
            "sidecar": str(receipt["sidecar"]),
            "sidecar_sha256": str(receipt["sha256"]),
            "sidecar_size": int(receipt["size"]),
            "replan_indices": {
                str(arm): list(indices[arm]) for arm in sorted(indices)
            },
            "anchor_batch": anchor_digest,
            "local_ratio_identity": local_ratio_identity,
            "exact_b1_metric_flags": exact_b1_flags,
            "sidecar_hash_and_deep_schema_authenticated_at_use": True,
        }
        selections = _all_gather_object(local_selection, world)
        _require(sorted(int(row["rank"]) for row in selections) == list(range(world)),
                 "audit point lacks one selection per rank")
        _require(len({int(row["group_index"]) for row in selections}) == world,
                 "production ranks selected duplicate groups at one audit point")
        selection_rows.extend(selections)
        if rank == 0:
            points.append({
                "point_index": point_index,
                "global_step": step,
                "fold": fold,
                "split": collection.split,
                "rank_local_groups": selections,
                "global_selected_context_authentication": {
                    "atoms": int(selected_auth_sum["atoms"]),
                    **selected_auth_max,
                    "all_exact": True,
                },
                "global_ratio_identity": {
                    "ratio_atoms": int(reduced_ratio["ratio_atoms"]),
                    "ratio_min": reduced_ratio["ratio_min"],
                    "ratio_mean": reduced_ratio["ratio_mean"],
                    "ratio_max": reduced_ratio["ratio_max"],
                    "max_abs_logratio": reduced_ratio["max_abs_logratio"],
                    "clip_fraction": reduced_ratio["clip_fraction"],
                    "ratio_ess_fraction": reduced_ratio["ratio_ess_fraction"],
                    "all_exact": True,
                },
                "global_mean_losses": loss_means,
                "gradients": gradients,
            })

        del (
            payload,
            grpo_loss,
            balance_loss,
            anchor_loss,
            losses,
            gradients,
            training_gradient_vectors,
        )
        if (point_index + 1) % AUDIT_STEPS_PER_FOLD == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # The first 8*8 draws of a repeated production permutation are distinct.
    for fold in range(grpo.N_FOLDS):
        fold_rows = [row for row in selection_rows if int(row["fold"]) == fold]
        _require(len(fold_rows) == EXPECTED_SELECTED_GROUPS_PER_FOLD,
                 f"fold {fold} audit selection count changed")
        _require(len({int(row["group_index"]) for row in fold_rows})
                 == EXPECTED_SELECTED_GROUPS_PER_FOLD,
                 f"fold {fold} audit groups are not all distinct")
    _require(len(selection_rows) == EXPECTED_SELECTED_GROUPS,
             "global audit selection count changed")
    _require(len(heldout_selection_rows) == grpo.N_FOLDS * world,
             "heldout directional selection count changed")
    _require(len({int(row["group_index"]) for row in heldout_selection_rows})
             == grpo.N_FOLDS * world,
             "heldout directional checks did not use 48 distinct validation groups")

    local_error = ""
    try:
        post_use_sidecars_local = recheck_rank_selected_sidecars(
            collections,
            validation,
            [*selection_rows, *heldout_selection_rows],
            rank=rank,
        )
    except Exception as exc:  # noqa: BLE001
        local_error = f"{type(exc).__name__}: {exc}"
    grpo._raise_if_any_rank_failed(
        local_error, world, "selected sidecar post-use closure",
    )
    post_use_sidecars = _all_gather_object(post_use_sidecars_local, world)
    _require(sum(int(row["selected_sidecars"]) for row in post_use_sidecars)
             == EXPECTED_SELECTED_GROUPS + grpo.N_FOLDS * world,
             "post-use sidecar closure count changed")

    local_error = ""
    try:
        proposal_digest_after = grpo.proposal_module_digest(proposal.state_dict())
    except Exception as exc:  # noqa: BLE001
        local_error = f"{type(exc).__name__}: {exc}"
    grpo._raise_if_any_rank_failed(local_error, world, "final proposal digest")
    digest_rows = _all_gather_object({
        "rank": rank,
        "before": proposal_digest_before,
        "after": proposal_digest_after,
        "parameter_grad_buffers_populated": any(
            parameter.grad is not None for parameter in proposal.parameters()
        ),
        "optimizer_state_entries": len(optimizer.state),
    }, world)
    _require(all(row["before"] == proposal_digest_before
                 and row["after"] == proposal_digest_before for row in digest_rows),
             "proposal digest changed on at least one rank")
    _require(not any(row["parameter_grad_buffers_populated"] for row in digest_rows),
             "read-only autograd left proposal gradient buffers populated")
    _require(not any(int(row["optimizer_state_entries"]) for row in digest_rows),
             "diagnostic optimizer state changed without an update")
    local_error = ""
    try:
        for collection in collections:
            collection.assert_unchanged()
        validation.assert_unchanged()
        grpo._assert_seed_stat(parent_identity)
        final_source_identity = _source_identity()
        _require(final_source_identity == source_identity,
                 "audit/trainer source changed during the diagnostic")
        _require(
            grpo._strict_outcome_determinism_state()
            == grpo.STRICT_OUTCOME_DETERMINISM,
            "strict deterministic flags changed during the diagnostic",
        )
        grpo._require_exact_proposal_scoring_environment(proposal, device)
        torch.cuda.synchronize(device)
    except Exception as exc:  # noqa: BLE001
        local_error = f"{type(exc).__name__}: {exc}"
    grpo._raise_if_any_rank_failed(local_error, world, "final read-only audit checks")

    if rank != 0:
        if world > 1:
            torch.distributed.barrier()
        return None

    _require(len(points) == len(audit_steps()),
             "rank zero did not retain every synchronized point")
    _require(len(directional_checks) == EXPECTED_HELDOUT_DIRECTIONAL_CHECKS,
             "rank zero did not retain one heldout directional check per fold")
    directional_derivatives = [
        row["d_heldout_loss_d_epsilon"] for row in directional_checks
    ]
    report = {
        "format_version": FORMAT_VERSION,
        "kind": KIND,
        "status": STATUS,
        "execution_validated": True,
        "eligibility": dict(ELIGIBILITY),
        "source_identity": source_identity,
        "config": {
            "path": str(config_resolved),
            "hash": config_hash,
            "seed_checkpoint": EXPECTED_SEED_CHECKPOINT,
            "seed_global_step": grpo.START_STEP,
        },
        "parent": parent_identity,
        "runtime_by_rank": runtimes,
        "strict_determinism": strict,
        "scoring_geometry": {
            "outcome_recovery": scoring,
            "outcome_proposal_and_pl_batch_size": 1,
            "anchor_batch_per_gpu": int(cfg["data"]["batch_per_gpu"]),
            "anchor_autocast_dtype": "torch.bfloat16",
            "all_selected_seed_ratios_exactly_one": True,
        },
        "audit_geometry": {
            "world_size": world,
            "nodes": 1,
            "gpus_per_node": world,
            "folds": grpo.N_FOLDS,
            "synchronized_steps_per_fold": AUDIT_STEPS_PER_FOLD,
            "synchronized_points": len(points),
            "distinct_authenticated_groups_per_fold": (
                EXPECTED_SELECTED_GROUPS_PER_FOLD
            ),
            "distinct_authenticated_groups_total": EXPECTED_SELECTED_GROUPS,
            "heldout_directional_checks": EXPECTED_HELDOUT_DIRECTIONAL_CHECKS,
            "distinct_authenticated_validation_groups": grpo.N_FOLDS * world,
            "heldout_component_attribution": True,
            "fixed_counterfactual_anchor_scales": list(ANCHOR_SCALE_SWEEP),
            "directional_scope": "loss_gradient_space_only",
            "contexts_per_arm": int(cfg["outcome_grpo"]["contexts_per_arm"]),
            "ratio_arms": list(range(1, recovery.GROUP_SIZE)),
            "ratio_atoms_per_rank_point": (
                (recovery.GROUP_SIZE - 1)
                * int(cfg["outcome_grpo"]["contexts_per_arm"])
            ),
            "fixed_global_steps": list(audit_steps()),
            "selection": "DeterministicOutcomeSampler production step/rank geometry",
        },
        "components": [
            {"name": "grpo", "weight": 1.0, "precision": "fp32",
             "scoring_batch_size": 1},
            {"name": "sparse_q_action_ce", "weight": 1.0,
             "teacher": "frozen q_action", "precision": "bf16 autocast"},
            {"name": "switch_balance_1e-2", "weight": 1e-2,
             "source": "same exact B1 GRPO logits", "precision": "fp32"},
        ],
        "anchor_preflight_by_rank": anchor_preflight_rows,
        "collections": {
            "training_folds": [
                collection.provenance() for collection in collections
            ],
            "heldout_validation": validation.provenance(),
        },
        "points": points,
        "summary": summarise_points(points),
        "directional_checks": directional_checks,
        "directional_check_summary": {
            "checks": len(directional_checks),
            "distinct_authenticated_validation_groups": len(
                heldout_selection_rows
            ),
            "epsilon": DIRECTIONAL_EPSILON,
            "d_heldout_loss_d_epsilon": _summary(directional_derivatives),
            "immediate_first_order_harm_count": sum(
                value > 0.0 for value in directional_derivatives
            ),
            "immediate_first_order_benefit_count": sum(
                value < 0.0 for value in directional_derivatives
            ),
            "first_order_flat_count": sum(
                value == 0.0 for value in directional_derivatives
            ),
            "gradient_space_attribution": (
                summarise_directional_attributions(directional_checks)
            ),
            "parameter_perturbations": 0,
            "optimizer_steps": 0,
        },
        "warnings": sorted(set(non_determinism_warnings)),
        "selected_sidecar_post_use_closure": {
            "passed": True,
            "sidecars": EXPECTED_SELECTED_GROUPS + grpo.N_FOLDS * world,
            "bytes": sum(int(row["selected_bytes"])
                         for row in post_use_sidecars),
            "rank_evidence": post_use_sidecars,
        },
        "no_mutation": {
            "passed": True,
            "optimizer_constructed_as_sentinel": True,
            "optimizer_steps": 0,
            "optimizer_state_entries_before": 0,
            "optimizer_state_entries_after": 0,
            "parameter_perturbations": 0,
            "proposal_digest_before": proposal_digest_before,
            "proposal_digest_after": proposal_digest_after,
            "rank_evidence": digest_rows,
        },
        "wall_seconds": float(time.monotonic() - started),
        "output": str(output),
    }
    exclusive_json_write(output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), flush=True)
    if world > 1:
        torch.distributed.barrier()
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/r0a_outcome_grpo.yaml",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    exit_code = 0
    try:
        run_audit(config_path=args.config, output_path=args.out)
    except Exception as exc:  # noqa: BLE001
        print(
            "OUTCOME_COMPONENT_GRADIENT_AUDIT_FAILED: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        exit_code = 2
    finally:
        if torch.distributed.is_available() \
                and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
