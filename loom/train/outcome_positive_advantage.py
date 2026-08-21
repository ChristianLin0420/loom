"""Isolated positive-advantage outcome-regression primitives.

This module is deliberately only a mathematical core.  It does not define a
trainer, configuration, launcher, checkpoint, candidate, or evaluation path.
Recovery-policy atoms are scored by the existing collector-matching, stored-
order, row-wise B=1 fp32 Plackett--Luce scorer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from loom.eval import outcome_recovery as recovery
from loom.train import outcome_grpo as v1


__all__ = [
    "FORMAT_VERSION",
    "CORE_KIND",
    "RATIO_CLIP_LOW",
    "RATIO_CLIP_HIGH",
    "PositiveAdvantageError",
    "PositiveAdvantageObjective",
    "standardised_terminal_advantages",
    "positive_advantage_weights",
    "clipped_positive_advantage_objective",
    "group_positive_advantage_loss",
    "sampled_positive_advantage_objective",
    "analytic_categorical_forward_kl",
    "core_source_identity",
    "assert_core_source_identity",
    "core_provenance",
]


FORMAT_VERSION = 1
CORE_KIND = "loom_outcome_positive_advantage_core"
RATIO_CLIP_LOW = 0.8
RATIO_CLIP_HIGH = 1.2
_ROOT = Path(__file__).resolve().parents[2]
_CORE_SOURCE_FILES = (
    "contracts.py",
    "loom/eval/outcome_recovery.py",
    "loom/heads/proposal.py",
    "loom/train/outcome_grpo.py",
    "loom/train/outcome_positive_advantage.py",
)


class PositiveAdvantageError(RuntimeError):
    """The positive-advantage core received invalid numerical state."""


def _require(condition: Any, message: str) -> None:
    if not bool(condition):
        raise PositiveAdvantageError(message)


def standardised_terminal_advantages(
    rewards: Sequence[float] | Tensor,
) -> Tensor:
    """Population-standardise one complete eight-arm terminal-reward group.

    All eight rewards, including arm 0, define the mean and population RMS.
    A constant group maps to eight exact fp32 zeros rather than using an
    epsilon-dependent scale.
    """
    values = torch.as_tensor(rewards, dtype=torch.float32).detach().reshape(-1)
    if values.numel() != recovery.GROUP_SIZE:
        raise ValueError(
            f"one recovery group needs {recovery.GROUP_SIZE} rewards, "
            f"got {values.numel()}"
        )
    if not bool(torch.isfinite(values).all()):
        raise PositiveAdvantageError("terminal rewards contain nan/inf")
    centred = values - values.mean()
    population_variance = centred.square().mean()
    if float(population_variance) == 0.0:
        return torch.zeros_like(values)
    advantages = centred / population_variance.sqrt()
    _require(bool(torch.isfinite(advantages).all()),
             "standardised terminal advantages contain nan/inf")
    return advantages


def positive_advantage_weights(
    rewards: Sequence[float] | Tensor,
) -> tuple[Tensor, Tensor]:
    """Return all-eight standardised advantages and ``max(A, 0)`` weights."""
    advantages = standardised_terminal_advantages(rewards).detach()
    weights = advantages.clamp_min(0.0).detach()
    _require(bool(torch.isfinite(weights).all()),
             "positive-advantage weights contain nan/inf")
    return advantages, weights


def clipped_positive_advantage_objective(
    current_logprob: Tensor,
    old_logprob: Tensor,
    weight: float | Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return per-context loss, importance ratio, and PPO clip indicator.

    The minimized loss is ``-min(rho*w, clamp(rho,.8,1.2)*w)``.  ``w`` must be
    one finite nonnegative scalar.  The lower clip remains part of the exact
    PPO contract even though a nonnegative weight makes only the upper branch
    active in the minimum.
    """
    if current_logprob.shape != old_logprob.shape or current_logprob.numel() == 0:
        raise ValueError(
            "current/old log probabilities must have the same non-empty shape, "
            f"got {tuple(current_logprob.shape)}/{tuple(old_logprob.shape)}"
        )
    with torch.autocast(device_type=current_logprob.device.type, enabled=False):
        current = current_logprob.float()
        old = old_logprob.detach().to(device=current.device, dtype=torch.float32)
        scalar_weight = torch.as_tensor(
            weight, device=current.device, dtype=torch.float32,
        ).detach()
        if scalar_weight.numel() != 1:
            raise ValueError(
                f"positive-advantage weight must be scalar, got {tuple(scalar_weight.shape)}"
            )
        scalar_weight = scalar_weight.reshape(())
        _require(bool(torch.isfinite(current).all()) and bool(torch.isfinite(old).all()),
                 "positive-advantage log probabilities contain nan/inf")
        _require(bool(torch.isfinite(scalar_weight)) and float(scalar_weight) >= 0.0,
                 "positive-advantage weight must be finite and nonnegative")
        ratio = torch.exp(current - old)
        _require(bool(torch.isfinite(ratio).all()),
                 "positive-advantage importance ratio contains nan/inf")
        clipped_ratio = ratio.clamp(RATIO_CLIP_LOW, RATIO_CLIP_HIGH)
        benefit = torch.minimum(
            ratio * scalar_weight,
            clipped_ratio * scalar_weight,
        )
        loss = -benefit
        clipped = (ratio < RATIO_CLIP_LOW) | (ratio > RATIO_CLIP_HIGH)
    _require(bool(torch.isfinite(loss).all()),
             "positive-advantage clipped loss contains nan/inf")
    return loss, ratio, clipped


def group_positive_advantage_loss(
    current_logprobs: Sequence[Tensor],
    old_logprobs: Sequence[Tensor],
    rewards: Sequence[float] | Tensor,
) -> tuple[Tensor, Tensor]:
    """Equal-context-within-arm, then equal-seven-arm positive loss.

    The log-probability arguments contain arms 1--7 only.  Arm 0 participates
    only in the all-eight advantage baseline/scale.  Every sampled arm remains
    an explicit term even when its positive weight is exactly zero.
    """
    expected = recovery.GROUP_SIZE - 1
    if len(current_logprobs) != expected or len(old_logprobs) != expected:
        raise ValueError(
            f"positive-advantage loss expects arms 1..7, got "
            f"{len(current_logprobs)}/{len(old_logprobs)}"
        )
    advantages, weights = positive_advantage_weights(rewards)
    arm_terms: list[Tensor] = []
    ratios: list[Tensor] = []
    for arm, (current, old) in enumerate(
        zip(current_logprobs, old_logprobs, strict=True), start=1,
    ):
        per_context, ratio, _clipped = clipped_positive_advantage_objective(
            current, old, weights[arm],
        )
        arm_terms.append(per_context.mean())
        ratios.append(ratio.reshape(-1))
    # Retain the local name as executable evidence that arm 0 affected only the
    # standardisation above and is never indexed as an objective weight.
    del advantages
    return torch.stack(arm_terms).mean(), torch.cat(ratios)


@dataclass(frozen=True)
class PositiveAdvantageObjective:
    """One sampled recovery group's direct positive-advantage objective."""

    loss: Tensor
    advantages: Tensor
    positive_weights: Tensor
    metrics: dict[str, float]


def _selected_indices(
    replan_indices: Mapping[int, Sequence[int]],
    arm: int,
    n_replans: int,
) -> Tensor:
    raw = tuple(replan_indices[arm])
    if not raw:
        raise ValueError(f"sampled arm {arm} has zero contexts")
    if any(isinstance(value, bool) or int(value) != value for value in raw):
        raise ValueError(f"sampled arm {arm} has a non-integer context index")
    values = tuple(int(value) for value in raw)
    if len(values) != len(set(values)):
        raise ValueError(f"sampled arm {arm} repeats a context index")
    index = torch.tensor(values, dtype=torch.int64)
    if bool((index < 0).any()) or bool((index >= int(n_replans)).any()):
        raise ValueError(f"arm {arm} sampled context outside [0,{n_replans})")
    return index


def sampled_positive_advantage_objective(
    proposal: nn.Module,
    payload: Mapping[str, Any],
    replan_indices: Mapping[int, Sequence[int]],
    *,
    device: torch.device,
) -> PositiveAdvantageObjective:
    """Score one group with the exact stored-order B=1 PL execution path."""
    expected_arms = set(range(1, recovery.GROUP_SIZE))
    if set(replan_indices) != expected_arms:
        raise ValueError(
            "positive-advantage contexts must name exactly arms 1..7; got "
            f"{sorted(replan_indices)}"
        )
    arms = list(payload.get("arms") or ())
    if len(arms) != recovery.GROUP_SIZE:
        raise ValueError(f"recovery group must contain {recovery.GROUP_SIZE} arms")
    rewards = torch.tensor([
        float(torch.as_tensor(arm["terminal_reward"]).detach()) for arm in arms
    ], dtype=torch.float32)
    advantages, weights = positive_advantage_weights(rewards)
    advantages = advantages.to(device)
    weights = weights.to(device)
    try:
        proposal_dtype = next(proposal.parameters()).dtype
    except StopIteration as exc:
        raise ValueError("proposal has no parameters") from exc

    arm_losses: list[Tensor] = []
    ratios: list[Tensor] = []
    clipped: list[Tensor] = []
    logratios: list[Tensor] = []
    contexts_by_arm: list[int] = []
    for arm_index in range(1, recovery.GROUP_SIZE):
        arm = arms[arm_index]
        n_replans = int(arm["z"].shape[0])
        index = _selected_indices(replan_indices, arm_index, n_replans)
        z = arm["z"].index_select(0, index).to(
            device=device, dtype=proposal_dtype, non_blocking=True,
        )
        order = arm["ordered_support"].index_select(0, index).to(device=device)
        old = arm["old_logprob"].detach().index_select(0, index).to(
            device=device, dtype=torch.float32,
        )
        lang = v1._batched_lang(
            arm["lang"], int(index.numel()), device, proposal_dtype,
        )
        current, _logits = v1.stored_order_logprob(proposal, z, lang, order)
        per_context, ratio, was_clipped = clipped_positive_advantage_objective(
            current, old, weights[arm_index],
        )
        arm_losses.append(per_context.mean())
        ratios.append(ratio.detach().reshape(-1))
        clipped.append(was_clipped.detach().reshape(-1))
        logratios.append((current.detach().float() - old).reshape(-1))
        contexts_by_arm.append(int(index.numel()))

    loss = torch.stack(arm_losses).mean()
    ratio_all = torch.cat(ratios)
    clipped_all = torch.cat(clipped)
    logratio_all = torch.cat(logratios)
    ratio64 = ratio_all.double()
    ratio_sum = float(ratio64.sum())
    ratio_square_sum = float(ratio64.square().sum())
    ratio_atoms = float(ratio_all.numel())
    sampled_weights = weights[1:]
    metrics = {
        "positive_advantage_loss": float(loss.detach()),
        "advantage_mean_all_8": float(advantages.mean()),
        "advantage_population_rms_all_8": float(
            advantages.square().mean().sqrt()
        ),
        "sampled_positive_weight_sum": float(sampled_weights.sum()),
        "sampled_positive_weight_arms": float((sampled_weights > 0.0).sum()),
        "sampled_zero_weight_arms": float((sampled_weights == 0.0).sum()),
        "arm_terms_retained": float(len(arm_losses)),
        "arm0_advantage": float(advantages[0]),
        "arm0_positive_weight_ignored": float(weights[0]),
        "arm0_score_atoms": 0.0,
        "sampled_arms_min": 1.0,
        "sampled_arms_max": 7.0,
        "contexts_min_per_arm": float(min(contexts_by_arm)),
        "contexts_max_per_arm": float(max(contexts_by_arm)),
        "ratio_mean": ratio_sum / ratio_atoms,
        "ratio_min": float(ratio_all.float().min()),
        "ratio_max": float(ratio_all.float().max()),
        "max_abs_logratio": float(logratio_all.abs().max()),
        "clip_fraction": float(clipped_all.float().mean()),
        "ratio_atoms": ratio_atoms,
        "ratio_sum": ratio_sum,
        "ratio_square_sum": ratio_square_sum,
        "ratio_ess_fraction": ratio_sum * ratio_sum / max(
            ratio_atoms * ratio_square_sum,
            torch.finfo(torch.float64).tiny,
        ),
        "clipped_atoms": float(clipped_all.sum()),
        "proposal_scoring_batch_size": float(v1.PROPOSAL_SCORING_BATCH_SIZE),
        "proposal_scoring_autocast": float(v1.PROPOSAL_SCORING_AUTOCAST),
    }
    _require(bool(torch.isfinite(loss)),
             "sampled positive-advantage loss contains nan/inf")
    return PositiveAdvantageObjective(
        loss=loss,
        advantages=advantages.detach().clone(),
        positive_weights=weights.detach().clone(),
        metrics=metrics,
    )


def analytic_categorical_forward_kl(
    current_logits: Tensor,
    seed_logits: Tensor,
    *,
    reduction: str = "mean",
) -> Tensor:
    """Exact-value categorical ``KL(seed || current)`` with analytic VJP.

    The returned forward value is the ordinary fp32 categorical KL.  Its graph
    is replaced by the exact analytic derivative ``p_current - p_seed`` using
    ``raw.detach() + surrogate - surrogate.detach()`` per row.  Consequently,
    identical bf16-cast logits produce both an exact-zero value and a bitwise-
    zero current-logit gradient.  Seed logits are always detached.
    """
    if (
        current_logits.shape != seed_logits.shape
        or current_logits.ndim != 2
        or current_logits.shape[0] <= 0
        or current_logits.shape[1] <= 0
    ):
        raise ValueError(
            "current/seed logits must have the same non-empty (B,M) shape, got "
            f"{tuple(current_logits.shape)}/{tuple(seed_logits.shape)}"
        )
    if reduction not in {"none", "sum", "mean"}:
        raise ValueError(f"unknown reduction {reduction!r}")
    with torch.autocast(device_type=current_logits.device.type, enabled=False):
        current = current_logits.float()
        seed = seed_logits.detach().to(
            device=current.device, dtype=torch.float32,
        )
        _require(bool(torch.isfinite(current).all()) and bool(torch.isfinite(seed).all()),
                 "categorical reference logits contain nan/inf")
        log_current = F.log_softmax(current, dim=-1)
        log_seed = F.log_softmax(seed, dim=-1)
        p_current = F.softmax(current, dim=-1)
        p_seed = F.softmax(seed, dim=-1)
        raw = (p_seed * (log_seed - log_current)).sum(dim=-1)
        surrogate = (
            (p_current.detach() - p_seed.detach()) * current
        ).sum(dim=-1)
        # Form the zero-valued correction first.  Besides carrying the analytic
        # VJP, this preserves ``raw`` bit for bit in the forward pass instead of
        # rounding once through ``raw + surrogate - surrogate``.
        per_row = raw.detach() + (surrogate - surrogate.detach())
    _require(bool(torch.isfinite(raw).all()) and bool(torch.isfinite(per_row).all()),
             "categorical reference KL contains nan/inf")
    if reduction == "none":
        return per_row
    if reduction == "sum":
        return per_row.sum()
    return per_row.mean()


def core_source_identity(
    root: str | Path = _ROOT,
) -> dict[str, Any]:
    """Hash the isolated core plus its exact inherited scorer dependencies."""
    return v1._trainer_source_identity(root=root, files=_CORE_SOURCE_FILES)


def assert_core_source_identity(
    expected: Mapping[str, Any],
    *,
    root: str | Path = _ROOT,
) -> None:
    """Fail if any byte in the positive-advantage source closure changed."""
    v1._assert_trainer_source_identity(
        expected, root=root, files=_CORE_SOURCE_FILES,
    )


def core_provenance(
    source_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the nonlaunchable mathematical contract for audit binding."""
    source = (
        core_source_identity()
        if source_identity is None
        else dict(source_identity)
    )
    assert_core_source_identity(source)
    return {
        "format_version": FORMAT_VERSION,
        "kind": CORE_KIND,
        "source_identity": source,
        "execution_surface": "mathematical_core_only",
        "trainer_wired": False,
        "config_present": False,
        "launcher_present": False,
        "candidate_or_evaluation_authority": False,
        "positive_advantage": {
            "rewards": "all_8_terminal_rewards",
            "standardisation": "subtract_mean_divide_population_rms",
            "constant_group": "eight_exact_zeros",
            "weight": "max(standardised_advantage,0)",
            "scored_arms": list(range(1, recovery.GROUP_SIZE)),
            "arm0": "baseline_and_scale_only_never_scored_or_weighted",
            "aggregation": "equal_context_within_arm_then_equal_7_arms",
            "zero_weight_arms": "retained_as_explicit_terms",
            "ratio": "exp(current_stored_order_PL_logprob-old_logprob)",
            "ratio_clip": [RATIO_CLIP_LOW, RATIO_CLIP_HIGH],
            "minimized_atom": "-min(rho*w,clamp(rho,.8,1.2)*w)",
            "proposal_scorer": (
                "outcome_grpo.stored_order_logprob_exact_rowwise_B1_fp32"
            ),
            "excluded_terms": ["switch_balance", "sparse_ce"],
        },
        "categorical_reference": {
            "api": "analytic_categorical_forward_kl",
            "orientation": "KL(softmax(seed)||softmax(current))",
            "forward": "ordinary_fp32_dense_categorical_KL",
            "current_vjp": "p_current_minus_p_seed",
            "seed": "detached_frozen_no_gradient",
            "identity": "exact_zero_value_and_bitwise_zero_current_gradient",
            "state_contract": (
                "exact_authenticated_train_demo_anchor_states_all_horizons"
            ),
            "forward_contract": (
                "live_and_seed_use_identical_inputs_and_autocast_then_logits_cast_fp32"
            ),
            "not_claimed": "full_topk_pl_kl_scalar",
        },
    }
