"""Isolated, fail-closed core for the outcome-GRPO v2 pilot.

This module does not alter :mod:`loom.train.outcome_grpo`.  It reuses only the
authenticated collection and exact proposal-scoring primitives from that
module while giving the corrected method a new sampler, objective, checkpoint
state, config namespace, source closure, and artifact policy.

The checked-in v2 config is intentionally *not launchable*.  Coefficients that
depend on the component-gradient projection, controller limits that depend on
the resume smoke, the fixed train-panel manifest, and the terminal-validation
lineage are unresolved.  ``require_launchable_config`` rejects the scaffold
before creating a run directory.  Even a subsequently frozen v2 pilot remains
ineligible for candidate emission or official evaluation.
"""

from __future__ import annotations

import argparse
import calendar
import copy
import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import contracts as C
from loom.eval import outcome_recovery as recovery
from loom.train import outcome_grpo as v1
from loom.train.atomic import fsync_dir

__all__ = [
    "FORMAT_VERSION",
    "TRAINER_KIND",
    "METHOD_STATUS_SCAFFOLD",
    "METHOD_STATUS_FROZEN_PILOT",
    "METHOD_STATUS_FROZEN_FORMAL",
    "SAMPLER_KIND",
    "CONTROLLER_KIND",
    "TRAIN_PANEL_KIND",
    "START_STEP",
    "PILOT_UPDATES",
    "STOP_STEP",
    "PILOT_SNAPSHOT_STEPS",
    "PROPOSAL_LR_SCALE",
    "BALANCE_WEIGHT",
    "OUTCOME_COLLECTOR_SOURCE",
    "EXPECTED_AUTHENTICATED_TRAIN_LINEAGE",
    "EXPOSED_VALIDATION_MANIFEST_SHA256",
    "EXPOSED_VALIDATION_IDENTITY_DIGEST",
    "OutcomeGRPOV2Error",
    "UnfrozenPilotError",
    "TrainTrustPanelViolation",
    "RoundRobinOutcomeSamplerV3",
    "RecoveryObjectiveV2",
    "sampled_group_objectives_v2",
    "recovery_pl_forward_kl",
    "dense_categorical_forward_kl",
    "DemoReferenceAnchorV2",
    "OneSidedRecoveryKLController",
    "TrainPanelThresholds",
    "TrainOnlyTrustPanel",
    "persist_pilot_failure_no_candidate",
    "validate_scaffold_config",
    "require_launchable_config",
    "trainer_source_identity",
    "assert_trainer_source_identity",
    "pilot_provenance",
    "train_outcome_grpo_v2",
    "build_parser",
    "main",
]


FORMAT_VERSION = 2
TRAINER_KIND = "loom_outcome_grpo_v2_proposal_pilot"
METHOD_STATUS_SCAFFOLD = "INELIGIBLE_UNFROZEN_PILOT"
METHOD_STATUS_FROZEN_PILOT = "INELIGIBLE_FROZEN_PILOT"
METHOD_STATUS_FROZEN_FORMAL = "FROZEN_FORMAL_TRAINING"
SAMPLER_KIND = "pure_step_interleaved_fold_rank_disjoint_v3"
CONTROLLER_KIND = "one_sided_recovery_pl_forward_kl_v1"
TRAIN_PANEL_KIND = "locked_train_only_trust_panel_v1"
START_STEP = v1.START_STEP
PILOT_UPDATES = 800
STOP_STEP = START_STEP + PILOT_UPDATES
PILOT_SNAPSHOT_STEPS = tuple(
    START_STEP + offset for offset in (200, 400, 600, 800)
)
PROPOSAL_LR_SCALE = v1.PROPOSAL_LR_SCALE / 4.0
BALANCE_WEIGHT = v1.SWITCH_BALANCE_WEIGHT
PANEL_EVERY = 200
CONTROLLER_EVERY = 20
EXPECTED_WORLD_SIZE = v1.EXPECTED_WORLD_SIZE
EXPECTED_CONTEXTS_PER_ARM = v1.EXPECTED_CONTEXTS_PER_ARM
EXPOSED_VALIDATION_FORMAT_VERSION = recovery.FORMAT_VERSION
OUTCOME_COLLECTOR_SOURCE = {
    "scheme": "sha256(path-nul-bytes-nul)-v1",
    "sha256": "1622aee21022347a98f82c4e587154f1e13128528441e4d87d96d0d609be1223",
    "files": [
        "contracts.py",
        "loom/data/adapters/libero.py",
        "loom/data/canonical.py",
        "loom/data/tower.py",
        "loom/eval/__init__.py",
        "loom/eval/libero.py",
        "loom/eval/outcome_recovery.py",
        "loom/eval/policy.py",
        "loom/eval/runner.py",
        "loom/heads/decoder.py",
        "loom/heads/proposal.py",
        "loom/model/estimator.py",
        "scripts/outcome_recovery.py",
    ],
}
EXPECTED_AUTHENTICATED_TRAIN_LINEAGE = (
    {
        "split": "train0",
        "path": "runs/outcome_recovery_s49666_train0",
        "manifest_sha256": (
            "f92f50960e1640b32f2f50c6e9a7c61603204ea21369c6e2493d3770b3683c17"
        ),
        "identity_digest": (
            "6aae6bbb5f6226de726de64bd8b57d6f4fb673a63c5e23c49ad136a03dd75433"
        ),
    },
    {
        "split": "train1",
        "path": "runs/outcome_recovery_s49666_train1",
        "manifest_sha256": (
            "8c45d514454598eb1c53c0d3ea3a12b3606f84baae612e5d3c2dc50bfd904421"
        ),
        "identity_digest": (
            "331c812f62a8249d6c4be6a368b2bc9c5bbe6e352a67f6e8ac89a4deeaf984a0"
        ),
    },
    {
        "split": "train2",
        "path": "runs/outcome_recovery_s49666_train2",
        "manifest_sha256": (
            "4a53fca9490840e90319d2fde986f8f6c12a6d236869c5000e5b4f0e1555b29b"
        ),
        "identity_digest": (
            "302d7a4a95c45526338e1bfffd6008a893825ba1eb6157217169cfe661fd77ee"
        ),
    },
    {
        "split": "train3",
        "path": "runs/outcome_recovery_s49666_train3",
        "manifest_sha256": (
            "289c0d6796d1bfccf471bb519c3145b0837147280b5059ae62d821a5b7d3594e"
        ),
        "identity_digest": (
            "f6c0eb2f6b11a477f84be6323ef1a94f720dfe30e8fd854fd1d0c3d25bcef35e"
        ),
    },
    {
        "split": "train4",
        "path": "runs/outcome_recovery_s49666_train4",
        "manifest_sha256": (
            "025a8a81556a733da4401fb489306222e93e72535f3ca8266c4780fc76f9857b"
        ),
        "identity_digest": (
            "206fb752eb2f345284286a35bdc277eee03573c916563abb724c8d3f67dad13c"
        ),
    },
    {
        "split": "train5",
        "path": "runs/outcome_recovery_s49666_train5",
        "manifest_sha256": (
            "61253c67af3ea3c5cda710a78498e098bd2ea54082ba557ad2145a79e14a1700"
        ),
        "identity_digest": (
            "8b19c20610d4dcfed619ca2854e548b6bac5ca2d7a0bc26dd00182d2614f7ea7"
        ),
    },
)
EXPOSED_VALIDATION_MANIFEST_SHA256 = (
    "c7a59139bfe4cc8412a82f735b9668f4865f753e5248957b3430617b3e6b9272"
)
EXPOSED_VALIDATION_IDENTITY_DIGEST = (
    "758b046c52a401c83fccba21523c9ea6add3aef0966ffa623deee581dafeccee"
)

_ROOT = Path(__file__).resolve().parents[2]
_V2_SOURCE_FILES = tuple(sorted(set(v1._TRAINER_SOURCE_FILES + (
    "loom/train/outcome_grpo_v2.py",
    "configs/r0a_outcome_grpo_v2_pilot.yaml",
    "scripts/train_outcome_grpo_v2.py",
    "scripts/outcome_grpo_v2_pilot.sbatch",
))))


class OutcomeGRPOV2Error(RuntimeError):
    """A v2 recipe, state, or safety invariant failed."""


class UnfrozenPilotError(OutcomeGRPOV2Error):
    """The checked-in scaffold lacks a separately audited pilot freeze."""

    def __init__(self, unresolved: Sequence[str]):
        self.unresolved = tuple(str(item) for item in unresolved)
        super().__init__(
            "outcome-GRPO v2 is not launchable; unresolved freeze fields: "
            + ", ".join(self.unresolved)
        )


class TrainTrustPanelViolation(OutcomeGRPOV2Error):
    """A scheduled TRAIN-only panel crossed its fail-closed envelope."""

    def __init__(self, report: Mapping[str, Any]):
        self.report = _json_copy(report)
        failed = [
            name for name, row in self.report.get("checks", {}).items()
            if not bool(row.get("pass"))
        ]
        super().__init__("v2 TRAIN-only trust panel failed: " + ", ".join(failed))


def _require(condition: Any, message: str) -> None:
    if not bool(condition):
        raise OutcomeGRPOV2Error(message)


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _valid_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _parse_utc(value: Any) -> int | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = time.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return None
    return int(calendar.timegm(parsed))


def _resolved_config_identity(cfg: Mapping[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(
        cfg, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return {
        "scheme": "sha256(canonical-json-sort-compact)-v1",
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "bytes": len(encoded),
    }


class RoundRobinOutcomeSamplerV3:
    """Pure interleaved fold sampler with rank-disjoint deterministic draws.

    For zero-based update ``t`` and rank ``r`` the global draw is
    ``d = t * world_size + r``.  ``fold = d % 6`` and ``q = d // 6``;
    ``divmod(q, len(informative[fold]))`` then supplies the visit and position
    in one authenticated, fixed permutation.  Thus every optimizer update
    mixes folds, resume is a pure function of step, and 1,200 updates at world
    size eight produce exactly 1,600 draws per fold.
    """

    def __init__(
        self,
        informative_groups: Sequence[Sequence[int]],
        *,
        seed: int,
        rank: int,
        world_size: int,
        start_step: int = START_STEP,
        total_updates: int = PILOT_UPDATES,
        contexts_per_arm: int = EXPECTED_CONTEXTS_PER_ARM,
        identity_digests: Sequence[str] = (),
    ) -> None:
        self.groups = tuple(
            tuple(int(index) for index in fold) for fold in informative_groups
        )
        if len(self.groups) != v1.N_FOLDS:
            raise ValueError(f"expected {v1.N_FOLDS} folds, got {len(self.groups)}")
        if any(not fold for fold in self.groups):
            raise ValueError("every fold needs at least one informative group")
        if any(len(fold) != len(set(fold)) for fold in self.groups):
            raise ValueError("informative group indices must be unique within each fold")
        if rank < 0 or world_size <= 0 or rank >= world_size:
            raise ValueError(f"invalid rank/world {rank}/{world_size}")
        if int(total_updates) <= 0:
            raise ValueError("total_updates must be positive")
        if int(contexts_per_arm) <= 0:
            raise ValueError("contexts_per_arm must be positive")
        # Ranks assigned to the same fold in one update have consecutive q.
        # Requiring at least ceil(world/6) groups makes those positions unique,
        # including the permutation wrap boundary.
        same_fold_ranks = math.ceil(int(world_size) / len(self.groups))
        if any(len(fold) < same_fold_ranks for fold in self.groups):
            raise ValueError(
                "every fold needs enough groups for within-step rank uniqueness"
            )
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.start_step = int(start_step)
        self.total_updates = int(total_updates)
        self.contexts_per_arm = int(contexts_per_arm)
        self.identity_digests = tuple(str(value) for value in identity_digests)
        if self.identity_digests and len(self.identity_digests) != len(self.groups):
            raise ValueError("collection identity count does not match fold count")
        self._group_permutations: dict[int, tuple[int, ...]] = {}
        self._replan_permutations: dict[
            tuple[int, int, int, int, int], tuple[int, ...]
        ] = {}

    @property
    def stop_step(self) -> int:
        return self.start_step + self.total_updates

    def _offset(self, step: int) -> int:
        offset = int(step) - self.start_step
        if offset < 0 or offset >= self.total_updates:
            raise ValueError(
                f"step {step} is outside [{self.start_step},{self.stop_step})"
            )
        return offset

    def _group_order(self, fold: int) -> tuple[int, ...]:
        if fold not in self._group_permutations:
            values = self.groups[fold]
            generator = torch.Generator(device="cpu")
            generator.manual_seed(v1._stable_seed(
                "outcome-v2-group-v3",
                self.seed,
                fold,
                self.identity_digests[fold] if self.identity_digests else "",
            ))
            order = torch.randperm(len(values), generator=generator).tolist()
            self._group_permutations[fold] = tuple(values[index] for index in order)
        return self._group_permutations[fold]

    def group_at(self, step: int) -> tuple[int, int, int]:
        """Return ``(fold, manifest_group_index, visit)`` for this rank."""
        update = self._offset(step)
        draw = update * self.world_size + self.rank
        fold = draw % len(self.groups)
        quotient = draw // len(self.groups)
        visit, position = divmod(quotient, len(self.groups[fold]))
        return fold, self._group_order(fold)[position], visit

    def _replan_order(
        self,
        fold: int,
        group: int,
        arm: int,
        epoch: int,
        n_replans: int,
    ) -> tuple[int, ...]:
        key = (fold, group, arm, epoch, n_replans)
        if key not in self._replan_permutations:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(v1._stable_seed(
                "outcome-v2-replan-v3",
                self.seed,
                fold,
                group,
                arm,
                epoch,
                self.identity_digests[fold] if self.identity_digests else "",
            ))
            self._replan_permutations[key] = tuple(
                torch.randperm(n_replans, generator=generator).tolist()
            )
        return self._replan_permutations[key]

    def replans_at(
        self,
        step: int,
        n_replans_by_arm: Sequence[int],
    ) -> dict[int, tuple[int, ...]]:
        if len(n_replans_by_arm) != recovery.GROUP_SIZE:
            raise ValueError(
                f"expected {recovery.GROUP_SIZE} replan counts, "
                f"got {len(n_replans_by_arm)}"
            )
        fold, group, visit = self.group_at(step)
        selected: dict[int, tuple[int, ...]] = {}
        for arm in range(1, recovery.GROUP_SIZE):
            n_replans = int(n_replans_by_arm[arm])
            if n_replans <= 0:
                raise ValueError(f"arm {arm} has no replans")
            indices: list[int] = []
            begin = visit * self.contexts_per_arm
            for draw in range(begin, begin + self.contexts_per_arm):
                epoch, position = divmod(draw, n_replans)
                indices.append(self._replan_order(
                    fold, group, arm, epoch, n_replans,
                )[position])
            selected[arm] = tuple(indices)
        return selected

    def state_dict(self, global_step: int) -> dict[str, Any]:
        # Rank stays absent so the rank-0 checkpoint is valid for all ranks.
        return {
            "kind": SAMPLER_KIND,
            "format_version": FORMAT_VERSION,
            "seed": self.seed,
            "world_size": self.world_size,
            "start_step": self.start_step,
            "stop_step": self.stop_step,
            "total_updates": self.total_updates,
            "contexts_per_arm": self.contexts_per_arm,
            "identity_digests": list(self.identity_digests),
            "informative_groups": [list(fold) for fold in self.groups],
            "global_step": int(global_step),
            "draw_formula": "d=(step-start)*world+rank;fold=d%6;q=d//6",
        }

    def validate_state_dict(self, state: Mapping[str, Any], global_step: int) -> None:
        _require(
            dict(state) == self.state_dict(global_step),
            "v2 sampler checkpoint differs from config/world/step",
        )


def recovery_pl_forward_kl(
    current_logprobs: Sequence[Tensor],
    old_logprobs: Sequence[Tensor],
) -> Tensor:
    """Unbiased forward-KL estimator on stored PL samples, arms 1--7 only.

    With ``delta = log p_current - log p_seed``, the control-variate atom
    ``expm1(delta) - delta`` has expectation ``KL(p_seed || p_current)`` under
    seed samples.  It is nonnegative and has exact zero value and first
    derivative at identity.  Each arm is averaged over its replans and the
    seven arm means are then equally averaged.
    """
    expected = recovery.GROUP_SIZE - 1
    if len(current_logprobs) != expected or len(old_logprobs) != expected:
        raise ValueError(
            f"recovery KL expects arms 1..7 only, got "
            f"{len(current_logprobs)}/{len(old_logprobs)}"
        )
    terms: list[Tensor] = []
    for arm, (current, old) in enumerate(
        zip(current_logprobs, old_logprobs, strict=True), start=1,
    ):
        if current.shape != old.shape or current.numel() == 0:
            raise ValueError(f"arm {arm} recovery log-probabilities are invalid")
        with torch.autocast(device_type=current.device.type, enabled=False):
            delta = current.float() - old.to(device=current.device, dtype=torch.float32)
            atom = torch.expm1(delta) - delta
        if not bool(torch.isfinite(atom).all()):
            raise OutcomeGRPOV2Error(f"nonfinite recovery KL atom in arm {arm}")
        terms.append(atom.mean())
    return torch.stack(terms).mean()


@dataclass(frozen=True)
class RecoveryObjectiveV2:
    grpo: Tensor
    balance: Tensor
    recovery_forward_kl: Tensor
    metrics: dict[str, float]


def sampled_group_objectives_v2(
    proposal: nn.Module,
    payload: Mapping[str, Any],
    replan_indices: Mapping[int, Sequence[int]],
    *,
    device: torch.device,
    clip_eps: float = v1.CLIP_EPS,
) -> RecoveryObjectiveV2:
    """Score GRPO, Switch balance, and recovery KL in one proposal pass."""
    expected_arms = set(range(1, recovery.GROUP_SIZE))
    if set(replan_indices) != expected_arms:
        raise ValueError(
            "v2 ratio/recovery losses must use exactly arms 1..7; got "
            f"{sorted(replan_indices)}"
        )
    counts = {len(tuple(replan_indices[arm])) for arm in expected_arms}
    if len(counts) != 1 or next(iter(counts), 0) <= 0:
        raise ValueError("every sampled arm must have the same positive replan count")
    arms = list(payload.get("arms") or ())
    if len(arms) != recovery.GROUP_SIZE:
        raise ValueError(f"recovery group must contain {recovery.GROUP_SIZE} arms")
    rewards = torch.tensor(
        [float(arm["terminal_reward"]) for arm in arms], dtype=torch.float32,
    )
    advantages = v1.normalised_group_advantages(rewards).to(device)
    dtype = next(proposal.parameters()).dtype
    grpo_terms: list[Tensor] = []
    current_logprobs: list[Tensor] = []
    old_logprobs: list[Tensor] = []
    sampled_logits: list[Tensor] = []
    ratios: list[Tensor] = []
    clipped: list[Tensor] = []
    logratios: list[Tensor] = []
    for arm_index in range(1, recovery.GROUP_SIZE):
        arm = arms[arm_index]
        index = torch.tensor(tuple(replan_indices[arm_index]), dtype=torch.int64)
        n_replans = int(arm["z"].shape[0])
        if bool((index < 0).any()) or bool((index >= n_replans).any()):
            raise ValueError(f"arm {arm_index} sampled replan outside [0,{n_replans})")
        z = arm["z"].index_select(0, index).to(
            device=device, dtype=dtype, non_blocking=True,
        )
        order = arm["ordered_support"].index_select(0, index).to(device=device)
        old = arm["old_logprob"].index_select(0, index).to(device=device).float()
        lang = v1._batched_lang(arm["lang"], int(index.numel()), device, dtype)
        current, logits = v1.stored_order_logprob(proposal, z, lang, order)
        objective, ratio, was_clipped = v1.clipped_grpo_objective(
            current, old, advantages[arm_index], clip_eps=clip_eps,
        )
        if not bool(torch.isfinite(objective).all()) or not bool(torch.isfinite(ratio).all()):
            raise OutcomeGRPOV2Error(
                f"nonfinite v2 recovery objective in {payload.get('group_id')} "
                f"arm {arm_index}"
            )
        grpo_terms.append(-objective.mean())
        current_logprobs.append(current)
        old_logprobs.append(old)
        sampled_logits.append(logits)
        ratios.append(ratio.detach().reshape(-1))
        clipped.append(was_clipped.detach().reshape(-1))
        logratios.append((current.detach().float() - old).reshape(-1))
    grpo = torch.stack(grpo_terms).mean()
    balance = v1.proposal_switch_balance(torch.cat(sampled_logits, dim=0))
    recovery_kl = recovery_pl_forward_kl(current_logprobs, old_logprobs)
    ratio_all = torch.cat(ratios)
    clipped_all = torch.cat(clipped)
    logratio_all = torch.cat(logratios)
    ratio64 = ratio_all.double()
    ratio_sum = float(ratio64.sum())
    ratio_square_sum = float(ratio64.square().sum())
    ratio_atoms = float(ratio_all.numel())
    metrics = {
        "grpo_loss": float(grpo.detach()),
        "proposal_balance": float(balance.detach()),
        "recovery_forward_kl": float(recovery_kl.detach()),
        "ratio_mean": ratio_sum / ratio_atoms,
        "ratio_min": float(ratio_all.float().min()),
        "ratio_max": float(ratio_all.float().max()),
        "max_abs_logratio": float(logratio_all.abs().max()),
        "clip_fraction": float(clipped_all.float().mean()),
        "ratio_atoms": ratio_atoms,
        "ratio_sum": ratio_sum,
        "ratio_square_sum": ratio_square_sum,
        "ratio_ess_fraction": ratio_sum * ratio_sum / max(
            ratio_atoms * ratio_square_sum, torch.finfo(torch.float64).tiny,
        ),
        "clipped_atoms": float(clipped_all.sum()),
        "informative_group": float(len(set(rewards.tolist())) > 1),
        "recovery_ratio_arms_min": 1.0,
        "recovery_ratio_arms_max": 7.0,
        "arm0_in_recovery_reference": 0.0,
    }
    return RecoveryObjectiveV2(
        grpo=grpo,
        balance=balance,
        recovery_forward_kl=recovery_kl,
        metrics=metrics,
    )


def dense_categorical_forward_kl(
    current_logits: Tensor,
    seed_logits: Tensor,
    *,
    reduction: str = "mean",
) -> Tensor:
    """``KL(softmax(seed) || softmax(current))`` in fp32.

    Matching this dense base categorical uniquely identifies the associated
    Plackett--Luce distribution, although the returned scalar is deliberately
    labelled categorical KL rather than full top-k PL KL.
    """
    if current_logits.shape != seed_logits.shape or current_logits.ndim != 2:
        raise ValueError(
            "current/seed demo logits must have the same (B,M) shape, got "
            f"{tuple(current_logits.shape)}/{tuple(seed_logits.shape)}"
        )
    with torch.autocast(device_type=current_logits.device.type, enabled=False):
        seed = seed_logits.detach().to(
            device=current_logits.device, dtype=torch.float32,
        )
        current = current_logits.float()
        p_seed = F.softmax(seed, dim=-1)
        per_row = (
            p_seed * (F.log_softmax(seed, dim=-1) - F.log_softmax(current, dim=-1))
        ).sum(-1)
    if not bool(torch.isfinite(per_row).all()):
        raise OutcomeGRPOV2Error("dense demo-reference KL is nonfinite")
    if reduction == "none":
        return per_row
    if reduction == "sum":
        return per_row.sum()
    if reduction == "mean":
        return per_row.mean()
    raise ValueError(f"unknown reduction {reduction!r}")


@dataclass
class DemoReferenceAnchorV2:
    """One-pass sparse q_action anchor plus frozen-seed demo reference.

    The live proposal logits are evaluated once per exact anchor state/horizon
    and reused by sparse CE and dense categorical forward KL.  The frozen seed
    proposal is reconstructed from the authenticated parent, excluded from the
    optimizer/checkpoint payload, and digest-checked before every save boundary.
    """

    anchor: v1.ExpertAnchor
    seed_proposal: nn.Module
    seed_digest: dict[str, Any]

    @classmethod
    def from_parent(
        cls,
        parent: Mapping[str, Any],
        live_proposal: nn.Module,
        *,
        trainer_cfg: Mapping[str, Any],
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
    ) -> "DemoReferenceAnchorV2":
        # v1's anchor loader expects its historical namespace.  Only the
        # immutable manifest and sparse-target recipe are adapted in memory;
        # the resolved v2 config remains distinct and checkpointed as-is.
        adapted = copy.deepcopy(dict(trainer_cfg))
        adapted["outcome_grpo"] = {
            "anchor_manifest": _json_copy(
                trainer_cfg["outcome_grpo_v2"]["anchor_manifest"]
            ),
        }
        # ExpertAnchor is reused here as a raw TRAIN target/state producer.
        # Its v1 constructor rejects a zero sparse-loss weight, but alpha=0 is
        # a valid predeclared v2 choice: only sparse CE is disabled, while the
        # exact anchor states remain required by the frozen-seed demo KL.
        adapted["losses"]["proposal"]["weight"] = 1.0
        anchor = v1.ExpertAnchor.from_parent(
            parent,
            live_proposal,
            trainer_cfg=adapted,
            device=device,
            rank=rank,
            world_size=world_size,
        )
        seed_proposal = v1._load_proposal(parent, device=device)
        seed_proposal.eval().requires_grad_(False)
        seed_digest = v1.proposal_module_digest(seed_proposal.state_dict())
        live_digest = v1.proposal_module_digest(live_proposal.state_dict())
        _require(
            seed_digest == live_digest,
            "live proposal does not equal authenticated seed at v2 entry",
        )
        return cls(anchor=anchor, seed_proposal=seed_proposal, seed_digest=seed_digest)

    def assert_seed_unchanged(self) -> None:
        _require(not self.anchor.proposal.training,
                 "live proposal left inherited eval-mode geometry")
        _require(not self.seed_proposal.training, "frozen seed proposal left eval mode")
        _require(
            not any(parameter.requires_grad for parameter in self.seed_proposal.parameters()),
            "frozen seed proposal unexpectedly requires gradients",
        )
        _require(
            not any(parameter.grad is not None for parameter in self.seed_proposal.parameters()),
            "frozen seed proposal accumulated gradients",
        )
        _require(
            v1.proposal_module_digest(self.seed_proposal.state_dict()) == self.seed_digest,
            "frozen seed proposal digest changed",
        )
        live_parameters = dict(self.anchor.proposal.named_parameters())
        seed_parameters = dict(self.seed_proposal.named_parameters())
        _require(live_parameters.keys() == seed_parameters.keys(),
                 "live/seed proposal parameter names differ")
        _require(
            all(
                live_parameters[name].data_ptr() != seed_parameters[name].data_ptr()
                for name in live_parameters
            ),
            "frozen seed proposal aliases live proposal storage",
        )

    def losses(self, global_step: int) -> tuple[Tensor, Tensor, dict[str, float]]:
        self.assert_seed_unchanged()
        beliefs, lang, targets, _ = self.anchor._prepare(global_step)
        sparse_terms: list[Tensor] = []
        reference_terms: list[Tensor] = []
        forward_autocast = self.anchor.device.type == "cuda"
        for horizon in range(C.DEPTH):
            # Preserve the inherited sparse-anchor proposal geometry exactly:
            # prepared inputs are unchanged and CUDA executes under bf16
            # autocast. Live and frozen seed see the identical input/context.
            # Only CE/KL probability arithmetic is forced back to fp32.
            belief = beliefs[horizon].detach()
            batched_lang = v1._batched_lang(
                lang,
                int(belief.shape[0]),
                belief.device,
                belief.dtype,
            )
            with torch.autocast(
                device_type=belief.device.type,
                dtype=torch.bfloat16,
                enabled=forward_autocast,
            ):
                current_logits = self.anchor.proposal.logits(belief, batched_lang)
                with torch.no_grad():
                    seed_logits = self.seed_proposal.logits(belief, batched_lang)
            target = targets[horizon].detach().float()
            _require(
                current_logits.shape == seed_logits.shape == target.shape,
                f"v2 anchor h{horizon + 1} logits/target shape mismatch",
            )
            with torch.autocast(device_type=belief.device.type, enabled=False):
                log_current = F.log_softmax(
                    current_logits.float() / float(self.anchor.temperature), dim=-1,
                )
                sparse_terms.append(-(target * log_current).sum(-1).mean())
                reference_terms.append(dense_categorical_forward_kl(
                    current_logits, seed_logits,
                ))
        sparse = torch.stack(sparse_terms).mean()
        demo_reference = torch.stack(reference_terms).mean()
        _require(
            bool(torch.isfinite(sparse)) and bool(torch.isfinite(demo_reference)),
            f"nonfinite v2 demo loss at step {global_step}",
        )
        self.assert_seed_unchanged()
        return sparse, demo_reference, {
            "anchor_sparse_ce": float(sparse.detach()),
            "demo_categorical_forward_kl": float(demo_reference.detach()),
            "demo_reference_horizons": float(C.DEPTH),
            "demo_reference_seed_trainable": 0.0,
            "demo_reference_forward_bf16_autocast": float(forward_autocast),
            "demo_reference_probability_math_fp32": 1.0,
        }

    def unexpected_gradients(self) -> list[str]:
        unexpected = list(self.anchor.unexpected_gradients())
        unexpected.extend(
            f"seed_proposal.{name}"
            for name, parameter in self.seed_proposal.named_parameters()
            if parameter.grad is not None
        )
        return unexpected

    def provenance(self) -> dict[str, Any]:
        self.assert_seed_unchanged()
        return {
            "kind": "frozen_seed_dense_categorical_demo_reference_v1",
            "distribution": "base_categorical_identifying_pl_distribution",
            "not_claimed": "full_topk_pl_kl_scalar",
            "states": "exact_authenticated_train_anchor_states_all_horizons",
            "current_logits_reused_with_sparse_ce": True,
            "sparse_anchor_weight_applied_externally": True,
            "zero_sparse_anchor_retains_demo_reference_states": True,
            "proposal_forward_geometry": (
                "v1_anchor_prepared_inputs_cuda_bf16_autocast_cpu_disabled"
            ),
            "live_seed_identical_input_and_autocast": True,
            "ce_kl_probability_math": "float32",
            "seed_proposal": _json_copy(self.seed_digest),
            "seed_in_optimizer": False,
            "seed_in_training_checkpoint": False,
            "seed_requires_grad": False,
        }


@dataclass
class OneSidedRecoveryKLController:
    """Deterministic, checkpoint-complete, increase-only KL multiplier."""

    initial_beta: float
    target_kl: float
    eta: float
    max_beta: float
    interval: int = CONTROLLER_EVERY
    beta: float = field(init=False)
    observed_updates: int = field(default=0, init=False)
    window: list[float] = field(default_factory=list, init=False)
    decisions: list[dict[str, float | int]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        values = {
            "initial_beta": self.initial_beta,
            "target_kl": self.target_kl,
            "eta": self.eta,
            "max_beta": self.max_beta,
        }
        if not all(_finite_number(value) for value in values.values()):
            raise ValueError(f"controller parameters must be finite: {values}")
        if self.initial_beta <= 0.0 or self.target_kl <= 0.0 or self.eta <= 0.0:
            raise ValueError("controller initial_beta/target_kl/eta must be positive")
        if self.max_beta < self.initial_beta:
            raise ValueError("controller max_beta must be >= initial_beta")
        if int(self.interval) <= 0:
            raise ValueError("controller interval must be positive")
        self.initial_beta = float(self.initial_beta)
        self.target_kl = float(self.target_kl)
        self.eta = float(self.eta)
        self.max_beta = float(self.max_beta)
        self.interval = int(self.interval)
        self.beta = self.initial_beta

    def observe(self, accepted_update: int, global_recovery_kl: float) -> dict[str, Any] | None:
        """Observe one post-all-reduce pre-update KL; update beta for next step."""
        expected = self.observed_updates + 1
        if int(accepted_update) != expected:
            raise ValueError(
                f"controller expected accepted_update={expected}, got {accepted_update}"
            )
        if not _finite_number(global_recovery_kl) or float(global_recovery_kl) < -1e-7:
            raise OutcomeGRPOV2Error(
                f"invalid global recovery KL at update {accepted_update}: "
                f"{global_recovery_kl}"
            )
        value = max(0.0, float(global_recovery_kl))
        self.window.append(value)
        self.observed_updates = expected
        if expected % self.interval:
            return None
        _require(
            len(self.window) == self.interval,
            "controller decision window is incomplete",
        )
        window_mean = math.fsum(self.window) / float(self.interval)
        excess = min(max(window_mean / self.target_kl - 1.0, 0.0), 1.0)
        old_beta = self.beta
        self.beta = min(self.max_beta, old_beta * math.exp(self.eta * excess))
        _require(self.beta >= old_beta, "one-sided controller decreased beta")
        decision: dict[str, float | int] = {
            "accepted_update": expected,
            "window_mean_recovery_kl": window_mean,
            "target_kl": self.target_kl,
            "clamped_relative_excess": excess,
            "old_beta": old_beta,
            "new_beta": self.beta,
        }
        self.decisions.append(decision)
        self.window.clear()
        return _json_copy(decision)

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": CONTROLLER_KIND,
            "format_version": FORMAT_VERSION,
            "initial_beta": self.initial_beta,
            "target_kl": self.target_kl,
            "eta": self.eta,
            "max_beta": self.max_beta,
            "interval": self.interval,
            "beta": self.beta,
            "observed_updates": self.observed_updates,
            "window": list(self.window),
            "decisions": _json_copy(self.decisions),
            "update_rule": (
                "beta=min(max_beta,beta*exp(eta*clamp(K/target-1,0,1)))"
            ),
            "decrease_allowed": False,
            "metric_scope": "global_all_rank_mean_pre_update_recovery_kl",
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "OneSidedRecoveryKLController":
        raw = dict(state)
        _require(raw.get("kind") == CONTROLLER_KIND, "controller kind differs")
        _require(int(raw.get("format_version", -1)) == FORMAT_VERSION,
                 "controller format differs")
        controller = cls(
            initial_beta=float(raw["initial_beta"]),
            target_kl=float(raw["target_kl"]),
            eta=float(raw["eta"]),
            max_beta=float(raw["max_beta"]),
            interval=int(raw["interval"]),
        )
        controller.beta = float(raw["beta"])
        controller.observed_updates = int(raw["observed_updates"])
        controller.window = [float(value) for value in raw.get("window", ())]
        controller.decisions = _json_copy(list(raw.get("decisions", ())))
        _require(
            0 <= len(controller.window) < controller.interval
            and len(controller.window) == controller.observed_updates % controller.interval,
            "controller partial window does not match observed updates",
        )
        _require(
            len(controller.decisions)
            == controller.observed_updates // controller.interval,
            "controller decision count does not match observed updates",
        )
        _require(
            controller.initial_beta <= controller.beta <= controller.max_beta,
            "controller beta is outside its fixed envelope",
        )
        previous = controller.initial_beta
        for index, decision in enumerate(controller.decisions, start=1):
            _require(
                int(decision["accepted_update"]) == index * controller.interval,
                "controller decision cadence differs",
            )
            _require(float(decision["old_beta"]) == previous,
                     "controller decision chain is discontinuous")
            new = float(decision["new_beta"])
            _require(new >= previous, "checkpointed controller decreased beta")
            previous = new
        _require(previous == controller.beta, "controller terminal beta differs")
        expected = controller.state_dict()
        _require(raw == expected, "controller checkpoint contains unknown/changed state")
        return controller


@dataclass(frozen=True)
class TrainPanelThresholds:
    recovery_forward_kl_max: float
    clip_fraction_max: float
    ess_fraction_min: float
    arm0_coeff_drift_p95_l2_max: float
    arm0_topk_overlap_change_min: float
    max_abs_logratio_max: float
    demo_categorical_forward_kl_max: float
    demo_topk_overlap_change_min: float

    def __post_init__(self) -> None:
        nonnegative = (
            self.recovery_forward_kl_max,
            self.clip_fraction_max,
            self.ess_fraction_min,
            self.arm0_coeff_drift_p95_l2_max,
            self.max_abs_logratio_max,
            self.demo_categorical_forward_kl_max,
        )
        if not all(
            _finite_number(value) and float(value) >= 0.0
            for value in nonnegative
        ):
            raise ValueError("TRAIN-panel thresholds must be finite and nonnegative")
        overlap_minima = (
            self.arm0_topk_overlap_change_min,
            self.demo_topk_overlap_change_min,
        )
        if not all(
            _finite_number(value) and -1.0 <= float(value) <= 1.0
            for value in overlap_minima
        ):
            raise ValueError("TRAIN-panel overlap-change minima must be in [-1,1]")
        if not 0.0 <= float(self.clip_fraction_max) <= 1.0:
            raise ValueError("clip_fraction_max must be in [0,1]")
        if not 0.0 <= float(self.ess_fraction_min) <= 1.0:
            raise ValueError("ess_fraction_min must be in [0,1]")

    def state_dict(self) -> dict[str, float]:
        return {
            "recovery_forward_kl_max": float(self.recovery_forward_kl_max),
            "clip_fraction_max": float(self.clip_fraction_max),
            "ess_fraction_min": float(self.ess_fraction_min),
            "arm0_coeff_drift_p95_l2_max": float(
                self.arm0_coeff_drift_p95_l2_max
            ),
            "arm0_topk_overlap_change_min": float(
                self.arm0_topk_overlap_change_min
            ),
            "max_abs_logratio_max": float(self.max_abs_logratio_max),
            "demo_categorical_forward_kl_max": float(
                self.demo_categorical_forward_kl_max
            ),
            "demo_topk_overlap_change_min": float(
                self.demo_topk_overlap_change_min
            ),
        }


@dataclass
class TrainOnlyTrustPanel:
    """Checkpointed scheduled gate over a pre-frozen TRAIN-only panel."""

    manifest_digest: str
    item_count: int
    source_splits: tuple[str, ...]
    demo_anchor_manifest_digest: str
    demo_anchor_batches: int
    demo_anchor_start_step: int
    thresholds: TrainPanelThresholds
    every: int = PANEL_EVERY
    reports: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not _valid_sha256(self.manifest_digest):
            raise ValueError("TRAIN-panel manifest digest must be a sha256 hex digest")
        if int(self.item_count) <= 0 or int(self.every) <= 0:
            raise ValueError("TRAIN-panel item_count/every must be positive")
        if self.demo_anchor_manifest_digest != v1.EXPECTED_ANCHOR_MANIFEST["digest"]:
            raise ValueError("demo panel anchor manifest digest differs")
        if int(self.demo_anchor_batches) <= 0:
            raise ValueError("demo panel needs at least one fixed anchor batch")
        if int(self.demo_anchor_start_step) != START_STEP:
            raise ValueError("demo panel start step differs from authenticated seed")
        self.item_count = int(self.item_count)
        self.every = int(self.every)
        self.demo_anchor_batches = int(self.demo_anchor_batches)
        self.demo_anchor_start_step = int(self.demo_anchor_start_step)
        self.source_splits = tuple(str(value) for value in self.source_splits)
        if not self.source_splits or any(
            not split.startswith("train") or "validation" in split.lower()
            for split in self.source_splits
        ):
            raise ValueError("online trust panel must use TRAIN splits only")
        if len(self.source_splits) != len(set(self.source_splits)):
            raise ValueError("online TRAIN-panel source splits must be unique")

    def due(self, accepted_update: int) -> bool:
        return int(accepted_update) > 0 and int(accepted_update) % self.every == 0

    def evaluate(
        self,
        accepted_update: int,
        metrics: Mapping[str, Any],
        *,
        provenance: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not self.due(accepted_update):
            raise ValueError(f"TRAIN panel is not due at update {accepted_update}")
        expected_update = (len(self.reports) + 1) * self.every
        if int(accepted_update) != expected_update:
            raise ValueError(
                f"TRAIN panel expected update {expected_update}, got {accepted_update}"
            )
        expected_provenance = {
            "kind": TRAIN_PANEL_KIND,
            "manifest_sha256": self.manifest_digest,
            "item_count": self.item_count,
            "source_splits": list(self.source_splits),
            "selection_locked_before_training": True,
            "uses_validation": False,
            "uses_holdout": False,
            "demo_panel": {
                "source": "frozen_authenticated_train_anchor_batches",
                "anchor_manifest_digest": self.demo_anchor_manifest_digest,
                "batches": self.demo_anchor_batches,
                "start_step": self.demo_anchor_start_step,
                "selection_locked_before_training": True,
                "uses_validation": False,
                "uses_holdout": False,
            },
        }
        _require(
            dict(provenance) == expected_provenance,
            "TRAIN-panel provenance differs or admits validation/holdout data",
        )
        names = {
            "recovery_forward_kl": "recovery_forward_kl_max",
            "clip_fraction": "clip_fraction_max",
            "ratio_ess_fraction": "ess_fraction_min",
            "arm0_coeff_drift_p95_l2": "arm0_coeff_drift_p95_l2_max",
            "arm0_topk_overlap_change": "arm0_topk_overlap_change_min",
            "max_abs_logratio": "max_abs_logratio_max",
            "demo_categorical_forward_kl": "demo_categorical_forward_kl_max",
            "demo_topk_overlap_change": "demo_topk_overlap_change_min",
        }
        values: dict[str, float] = {}
        for metric in names:
            value = metrics.get(metric)
            _require(_finite_number(value), f"TRAIN-panel {metric} is missing/nonfinite")
            values[metric] = float(value)
        _require(0.0 <= values["clip_fraction"] <= 1.0,
                 "TRAIN-panel clip_fraction is outside [0,1]")
        _require(0.0 <= values["ratio_ess_fraction"] <= 1.0,
                 "TRAIN-panel ratio_ess_fraction is outside [0,1]")
        for metric in ("arm0_topk_overlap_change", "demo_topk_overlap_change"):
            _require(-1.0 <= values[metric] <= 1.0,
                     f"TRAIN-panel {metric} is outside [-1,1]")
        for metric in (
            "recovery_forward_kl", "arm0_coeff_drift_p95_l2",
            "max_abs_logratio", "demo_categorical_forward_kl",
        ):
            _require(values[metric] >= -1e-7,
                     f"TRAIN-panel {metric} is materially negative")
        thresholds = self.thresholds.state_dict()

        def check(metric: str, threshold: str, relation: str) -> dict[str, Any]:
            value = values[metric]
            limit = thresholds[threshold]
            passed = value <= limit if relation == "max" else value >= limit
            return {
                "value": value,
                "threshold": limit,
                "relation": "<=" if relation == "max" else ">=",
                "pass": bool(passed),
            }

        checks = {
            "recovery_forward_kl": check(
                "recovery_forward_kl", "recovery_forward_kl_max", "max",
            ),
            "clip_fraction": check("clip_fraction", "clip_fraction_max", "max"),
            "ratio_ess_fraction": check(
                "ratio_ess_fraction", "ess_fraction_min", "min",
            ),
            "arm0_coeff_drift_p95_l2": check(
                "arm0_coeff_drift_p95_l2",
                "arm0_coeff_drift_p95_l2_max",
                "max",
            ),
            "arm0_topk_overlap_change": check(
                "arm0_topk_overlap_change",
                "arm0_topk_overlap_change_min",
                "min",
            ),
            "max_abs_logratio": check(
                "max_abs_logratio", "max_abs_logratio_max", "max",
            ),
            "demo_categorical_forward_kl": check(
                "demo_categorical_forward_kl",
                "demo_categorical_forward_kl_max",
                "max",
            ),
            "demo_topk_overlap_change": check(
                "demo_topk_overlap_change",
                "demo_topk_overlap_change_min",
                "min",
            ),
        }
        report = {
            "kind": TRAIN_PANEL_KIND,
            "format_version": FORMAT_VERSION,
            "accepted_update": int(accepted_update),
            "passed": all(bool(row["pass"]) for row in checks.values()),
            "checks": checks,
            "provenance": _json_copy(expected_provenance),
            "candidate_emitted": False,
            "online_validation_used": False,
        }
        self.reports.append(_json_copy(report))
        if not report["passed"]:
            raise TrainTrustPanelViolation(report)
        return report

    def provenance(self) -> dict[str, Any]:
        return {
            "kind": TRAIN_PANEL_KIND,
            "manifest_sha256": self.manifest_digest,
            "item_count": self.item_count,
            "source_splits": list(self.source_splits),
            "selection_locked_before_training": True,
            "uses_validation": False,
            "uses_holdout": False,
            "demo_panel": {
                "source": "frozen_authenticated_train_anchor_batches",
                "anchor_manifest_digest": self.demo_anchor_manifest_digest,
                "batches": self.demo_anchor_batches,
                "start_step": self.demo_anchor_start_step,
                "selection_locked_before_training": True,
                "uses_validation": False,
                "uses_holdout": False,
            },
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": TRAIN_PANEL_KIND,
            "format_version": FORMAT_VERSION,
            "manifest_sha256": self.manifest_digest,
            "item_count": self.item_count,
            "source_splits": list(self.source_splits),
            "demo_anchor_manifest_digest": self.demo_anchor_manifest_digest,
            "demo_anchor_batches": self.demo_anchor_batches,
            "demo_anchor_start_step": self.demo_anchor_start_step,
            "thresholds": self.thresholds.state_dict(),
            "every": self.every,
            "reports": _json_copy(self.reports),
            "candidate_emission": "forbidden",
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "TrainOnlyTrustPanel":
        raw = dict(state)
        _require(raw.get("kind") == TRAIN_PANEL_KIND, "TRAIN-panel kind differs")
        _require(int(raw.get("format_version", -1)) == FORMAT_VERSION,
                 "TRAIN-panel format differs")
        panel = cls(
            manifest_digest=str(raw["manifest_sha256"]),
            item_count=int(raw["item_count"]),
            source_splits=tuple(raw["source_splits"]),
            demo_anchor_manifest_digest=str(raw["demo_anchor_manifest_digest"]),
            demo_anchor_batches=int(raw["demo_anchor_batches"]),
            demo_anchor_start_step=int(raw["demo_anchor_start_step"]),
            thresholds=TrainPanelThresholds(**dict(raw["thresholds"])),
            every=int(raw["every"]),
        )
        panel.reports = _json_copy(list(raw.get("reports", ())))
        for index, report in enumerate(panel.reports, start=1):
            _require(
                int(report.get("accepted_update", -1)) == index * panel.every,
                "TRAIN-panel checkpoint cadence differs",
            )
            _require(
                report.get("kind") == TRAIN_PANEL_KIND
                and int(report.get("format_version", -1)) == FORMAT_VERSION,
                "TRAIN-panel checkpoint report identity differs",
            )
            _require(
                report.get("provenance") == panel.provenance(),
                "TRAIN-panel checkpoint report provenance differs",
            )
            _require(report.get("candidate_emitted") is False,
                     "TRAIN-panel checkpoint claims a candidate")
        _require(
            raw == panel.state_dict(),
            "TRAIN-panel checkpoint contains unknown/changed state",
        )
        return panel


def _publish_text_exclusive(path: Path, text: str) -> None:
    """Durably publish text with no-overwrite semantics under one directory."""
    encoded = text.encode("utf-8")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.pending-", dir=str(path.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        # The hard link is the atomic publication point and fails with
        # FileExistsError if another writer won after our earlier checks.
        os.link(temporary, path)
        fsync_dir(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
            fsync_dir(path.parent)


def persist_pilot_failure_no_candidate(
    run_dir: str | os.PathLike[str],
    report: Mapping[str, Any],
) -> Path:
    """Persist an ineligible v2 failure only if no candidate exists."""
    path = Path(run_dir).expanduser().resolve()
    candidate_names = (
        tuple(path.glob("candidate_*.pt"))
        + tuple(path.glob(".candidate_*.pt"))
        if path.exists() else ()
    )
    _require(
        not candidate_names,
        "v2 pilot failure cannot coexist with a candidate artifact",
    )
    _require(
        report.get("passed") is False,
        "v2 pilot failure persistence requires passed=false",
    )
    payload = {
        "format_version": FORMAT_VERSION,
        "kind": TRAINER_KIND,
        "method_status": METHOD_STATUS_FROZEN_PILOT,
        "status": "FAIL",
        "passed": False,
        "candidate_emitted": False,
        "promotion_eligible": False,
        "official_evaluation_eligible": False,
        "created_utc": _utc(),
        "report": _json_copy(report),
    }
    path.mkdir(parents=True, exist_ok=True)
    out = path / "pilot_failure.json"
    _publish_text_exclusive(
        out,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return out


def validate_scaffold_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the v2 namespace, including deliberate unresolved fields."""
    outcome = dict(cfg.get("outcome_grpo_v2", {}) or {})
    _require(outcome, "config lacks outcome_grpo_v2")
    _require("outcome_grpo" not in cfg, "v1 outcome_grpo namespace leaked into v2")
    _require(
        int(outcome.get("format_version", -1)) == FORMAT_VERSION,
        "outcome_grpo_v2 format_version differs",
    )
    _require(outcome.get("trainer_kind") == TRAINER_KIND, "v2 trainer kind differs")
    status = str(outcome.get("method_status") or "")
    _require(
        status in {
            METHOD_STATUS_SCAFFOLD,
            METHOD_STATUS_FROZEN_PILOT,
            METHOD_STATUS_FROZEN_FORMAL,
        },
        f"unknown v2 method status {status!r}",
    )
    sampler = dict(outcome.get("sampler", {}) or {})
    _require(sampler == {
        "kind": SAMPLER_KIND,
        "formula": "d=(step-start)*world+rank;fold=d%6;q=d//6",
        "total_updates": PILOT_UPDATES,
        "contexts_per_arm": EXPECTED_CONTEXTS_PER_ARM,
        "informative_training_groups": "only",
    }, "v2 sampler recipe differs")
    _require(int(outcome.get("start_step", -1)) == START_STEP, "v2 start differs")
    _require(int(outcome.get("stop_step", -1)) == STOP_STEP, "v2 stop differs")
    _require(
        list(outcome.get("snapshot_steps") or ()) == list(PILOT_SNAPSHOT_STEPS),
        "v2 pilot snapshots differ",
    )
    _require(
        int(outcome.get("world_size", -1)) == EXPECTED_WORLD_SIZE,
        "v2 world size differs",
    )
    expected_run_name = (
        "r0a_outcome_grpo_v2_formal"
        if status == METHOD_STATUS_FROZEN_FORMAL
        else "r0a_outcome_grpo_v2_pilot_INELIGIBLE"
    )
    _require(dict(cfg.get("run", {}) or {}) == {
        "name": expected_run_name,
        "project": "loom",
        "seed": v1.TRAIN_SEED,
        "steps": v1.SCHEDULE_STEPS,
        "deterministic": True,
        "log_every": v1.LOG_EVERY,
        "ckpt_every": PANEL_EVERY,
        "keep_last": 0,
        "wandb_mode": "online",
    }, "v2 run recipe differs")
    _require(dict(cfg.get("data", {}) or {}) == {
        "source": "libero",
        "embodiments": ["libero_franka"],
        "batch_per_gpu": v1.EXPECTED_BATCH_PER_GPU,
        "action_free": False,
        "sampling": "uniform_task",
        "trajectory_split": "train",
        "holdout_demo_keys": ["demo_49"],
        "recurrent_burn_in": 4,
        "cache_dir": "cache/",
        "num_workers": 4,
        "pin_memory": True,
        "prefetch_factor": 2,
    }, "v2 TRAIN anchor data recipe differs")
    _require(dict(cfg.get("fsdp", {}) or {}) == {
        "shard": [],
        "replicate": ["proposal"],
        "activation_checkpointing": False,
        "block_names": ["PerceiverBlock", "EstimatorBlock", "Block"],
        "forward_prefetch": True,
        "limit_all_gathers": True,
    }, "v2 proposal replication recipe differs")
    _require(dict(cfg.get("slurm", {}) or {}) == {
        "nodes": 1,
        "gpus_per_node": EXPECTED_WORLD_SIZE,
        "n_links": 1,
    }, "v2 allocation geometry differs")
    for name, expected in {
        "seed_checkpoint": v1.EXPECTED_SEED_CHECKPOINT,
        "seed_global_step": START_STEP,
        "seed_config_hash": recovery.SEED_CONFIG_HASH,
        "seed_checkpoint_sha256": recovery.SEED_CHECKPOINT_SHA256,
        "groups_per_train_fold": 200,
        "minimum_informative_groups_per_fold": v1.MIN_TRAIN_INFORMATIVE_GROUPS,
    }.items():
        _require(outcome.get(name) == expected, f"v2 {name} differs")
    _require(
        list(outcome.get("folds") or ()) == [dict(row) for row in v1.EXPECTED_FOLDS],
        "v2 authenticated TRAIN folds differ",
    )
    authenticated_lineage = dict(
        outcome.get("authenticated_data_lineage", {}) or {}
    )
    expected_development_lineage = {
        "split": "validation",
        "path": v1.EXPECTED_VALIDATION["path"],
        "manifest_sha256": EXPOSED_VALIDATION_MANIFEST_SHA256,
        "identity_digest": EXPOSED_VALIDATION_IDENTITY_DIGEST,
    }
    _require(authenticated_lineage == {
        "kind": "exact_outcome_recovery_manifest_closure_v1",
        "collection_format_version": recovery.FORMAT_VERSION,
        "collector_source": OUTCOME_COLLECTOR_SOURCE,
        "training": [dict(row) for row in EXPECTED_AUTHENTICATED_TRAIN_LINEAGE],
        "exposed_development": expected_development_lineage,
    }, "v2 authenticated TRAIN/development manifest lineage differs")
    _require(
        [
            {"split": row["split"], "path": row["path"]}
            for row in authenticated_lineage["training"]
        ] == [dict(row) for row in v1.EXPECTED_FOLDS],
        "v2 execution folds do not match authenticated data lineage",
    )
    _require(
        dict(outcome.get("anchor_manifest", {}) or {})
        == dict(v1.EXPECTED_ANCHOR_MANIFEST),
        "v2 TRAIN anchor manifest differs",
    )
    _require(dict(outcome.get("authentication", {}) or {}) == {
        "chunk_replans": v1.AUTH_CHUNK_REPLANS,
        "proposal_scoring_batch_size": v1.PROPOSAL_SCORING_BATCH_SIZE,
        "proposal_scoring_dtype": v1.PROPOSAL_SCORING_DTYPE,
        "proposal_scoring_autocast": v1.PROPOSAL_SCORING_AUTOCAST,
        "cuda_matmul_tf32": v1.CUDA_MATMUL_TF32,
        "cudnn_tf32": v1.CUDNN_TF32,
        "float32_matmul_precision": v1.FLOAT32_MATMUL_PRECISION,
        "proposal_scoring_module_mode": v1.PROPOSAL_SCORING_MODULE_MODE,
        "behaviour_logprob_atol": v1.BEHAVIOUR_LOGPROB_ATOL,
        "behaviour_logprob_rtol": v1.BEHAVIOUR_LOGPROB_RTOL,
        "behaviour_coeff_atol": v1.BEHAVIOUR_COEFF_ATOL,
        "behaviour_coeff_rtol": v1.BEHAVIOUR_COEFF_RTOL,
        "identity_max_abs_logratio": v1.BEHAVIOUR_IDENTITY_MAX_ABS_LOGRATIO,
        "identity_max_coeff_error": v1.BEHAVIOUR_IDENTITY_MAX_COEFF_ERROR,
        "initial_ratio_min_ess_fraction": v1.INITIAL_RATIO_MIN_ESS_FRACTION,
    }, "v2 behavior authentication geometry differs")
    optim = dict(cfg.get("optim", {}) or {})
    scales = dict(optim.get("lr_scales", {}) or {})
    _require(
        float(scales.get("proposal", math.nan)) == PROPOSAL_LR_SCALE,
        "v2 proposal LR scale is not exactly 4x below v1",
    )
    _require(optim == {
        "lr": v1.BASE_LEARNING_RATE,
        "warmup": v1.WARMUP_STEPS,
        "min_lr_ratio": v1.MIN_LR_RATIO,
        "grad_clip": v1.GRAD_CLIP,
        "betas": list(v1.ADAMW_BETAS),
        "weight_decay": v1.ADAMW_WEIGHT_DECAY,
        "eps": v1.ADAMW_EPS,
        "lr_scales": {
            "bank": 0.0,
            "decoder": 0.0,
            "ema": 0.0,
            "estimator": 0.0,
            "proposal": PROPOSAL_LR_SCALE,
            "q_action": 0.0,
            "q_delta": 0.0,
        },
        "reset_state_modules": ["proposal"],
        "update_ema": False,
        "bank_lr_mult": 0.1,
        "ema_tau": 0.996,
        "spike_mult": 10.0,
    }, "v2 optimizer recipe differs")
    _require(list(cfg.get("train_modules") or ()) == ["proposal"],
             "v2 train_modules must be exactly [proposal]")
    losses = dict(cfg.get("losses", {}) or {})
    _require(float(dict(losses.get("balance", {}) or {}).get("weight", math.nan))
             == BALANCE_WEIGHT, "v2 balance weight differs")
    recovery_ref = dict(losses.get("recovery_reference", {}) or {})
    _require(recovery_ref.get("kind") == "sampled_pl_forward_kl_control_variate",
             "v2 recovery-reference kind differs")
    _require(recovery_ref.get("arms") == list(range(1, 8)),
             "recovery reference must use arms 1..7 only")
    _require(recovery_ref.get("arm0") == "forbidden",
             "arm0 must be forbidden from recovery reference")
    _require(
        recovery_ref.get("aggregation")
        == "equal_replan_within_arm_then_equal_arm",
        "v2 recovery-reference aggregation differs",
    )
    _require(
        recovery_ref.get("enabled") is True
        and recovery_ref.get("atom")
            == "expm1(current_minus_old_logprob)-current_minus_old_logprob"
        and recovery_ref.get("controller_weight")
            == "outcome_grpo_v2.recovery_kl_controller.beta",
        "v2 recovery-reference estimator/controller binding differs",
    )
    demo_ref = dict(losses.get("demo_reference", {}) or {})
    _require(
        demo_ref.get("kind") == "frozen_seed_dense_categorical_forward_kl",
        "v2 demo-reference kind differs",
    )
    _require(demo_ref.get("states") == "train_anchor_exact_states_all_horizons",
             "v2 demo-reference states differ")
    proposal_loss = dict(losses.get("proposal", {}) or {})
    _require(
        proposal_loss.get("zero_weight_semantics")
        == "sparse_anchor_term_disabled_anchor_states_retained_for_demo_reference",
        "v2 zero sparse-anchor semantics differ",
    )
    _require(
        demo_ref.get("enabled") is True
        and demo_ref.get("distribution")
            == "base_categorical_identifying_pl_distribution"
        and demo_ref.get("seed_requires_grad") is False
        and demo_ref.get("seed_in_optimizer") is False
        and demo_ref.get("seed_in_checkpoint") is False,
        "v2 frozen-seed demo-reference isolation differs",
    )
    controller = dict(outcome.get("recovery_kl_controller", {}) or {})
    _require(controller.get("kind") == CONTROLLER_KIND,
             "v2 controller kind differs")
    _require(int(controller.get("interval", -1)) == CONTROLLER_EVERY,
             "v2 controller cadence differs")
    _require(controller.get("decrease_allowed") is False,
             "v2 controller must be one-sided")
    panel = dict(outcome.get("train_trust_panel", {}) or {})
    _require(panel.get("kind") == TRAIN_PANEL_KIND, "v2 TRAIN-panel kind differs")
    _require(int(panel.get("every", -1)) == PANEL_EVERY,
             "v2 TRAIN-panel cadence differs")
    _require(panel.get("data_scope") == "TRAIN_ONLY_LOCKED_BEFORE_TRAINING",
             "v2 online panel is not TRAIN-only")
    _require(panel.get("uses_validation") is False,
             "v2 online panel must not use validation")
    _require(panel.get("uses_holdout") is False,
             "v2 online panel must not use holdout data")
    _require(
        panel.get("source_splits") == [f"train{index}" for index in range(v1.N_FOLDS)],
        "v2 online panel TRAIN split set differs",
    )
    _require(dict(panel.get("demo_panel", {}) or {}) == {
        "source": "frozen_authenticated_train_anchor_batches",
        "anchor_manifest_digest": v1.EXPECTED_ANCHOR_MANIFEST["digest"],
        "batches": v1.EXPERT_GATE_BATCHES,
        "start_step": START_STEP,
        "uses_validation": False,
        "uses_holdout": False,
    }, "v2 fixed TRAIN demo-panel provenance differs")
    lineage = dict(outcome.get("validation_lineage", {}) or {})
    current = dict(lineage.get("current_development_collection", {}) or {})
    terminal = dict(lineage.get("formal_terminal_collection", {}) or {})
    _require(
        current.get("status") == "DEVELOPMENT_EXPOSED_DO_NOT_FORMAL_GATE",
        "current validation exposure is not declared",
    )
    _require(
        current.get("split") == "validation"
        and current.get("path") == v1.EXPECTED_VALIDATION["path"]
        and current.get("exposures") == [
            "v1_terminal_selection",
            "early_curve_diagnostic",
            "component_gradient_projection",
            "round_robin_direction_audit",
        ],
        "current development validation path/exposure ledger differs",
    )
    _require(
        int(current.get("collection_format_version", -1))
            == EXPOSED_VALIDATION_FORMAT_VERSION
        and current.get("manifest_sha256")
            == EXPOSED_VALIDATION_MANIFEST_SHA256
        and current.get("observed_identity_digest")
            == EXPOSED_VALIDATION_IDENTITY_DIGEST
        and current.get("identity_status")
            == "OBSERVED_DEVELOPMENT_ONLY_NOT_FORMAL_FROZEN",
        "current development validation identity/version is not explicit",
    )
    _require(
        lineage.get("decision_status") in {
            "UNRESOLVED_FREEZE_BEFORE_FRESH_FORMAL_COLLECTION",
            "PILOT_EXPOSED_DEVELOPMENT_ONLY_FORMAL_NOT_COLLECTED",
            "FROZEN_FRESH_FORMAL_COLLECTION",
        },
        "validation lineage decision status is invalid",
    )
    artifact = dict(outcome.get("artifact_policy", {}) or {})
    pilot_artifact_policy = {
        "role": "development_pilot",
        "candidate_emission": "forbidden",
        "promotion": "forbidden",
        "official_evaluation": "forbidden",
        "pilot_checkpoint_only": True,
    }
    formal_artifact_policy = {
        "role": "formal_candidate_training",
        "candidate_emission": "terminal_gates_only",
        "promotion": "external_official_seed0_gate_only",
        "official_evaluation": "candidate_only",
        "pilot_checkpoint_only": False,
    }
    expected_artifact_policy = (
        formal_artifact_policy
        if status == METHOD_STATUS_FROZEN_FORMAL
        else pilot_artifact_policy
    )
    _require(artifact == expected_artifact_policy,
             "v2 status/artifact-role policy differs")

    unresolved: list[str] = []
    required_numeric = {
        "losses.proposal.weight": dict(losses.get("proposal", {}) or {}).get("weight"),
        "losses.demo_reference.weight": demo_ref.get("weight"),
        "outcome_grpo_v2.recovery_kl_controller.initial_beta": controller.get("initial_beta"),
        "outcome_grpo_v2.recovery_kl_controller.target_kl": controller.get("target_kl"),
        "outcome_grpo_v2.recovery_kl_controller.eta": controller.get("eta"),
        "outcome_grpo_v2.recovery_kl_controller.max_beta": controller.get("max_beta"),
    }
    thresholds = dict(panel.get("thresholds", {}) or {})
    for key in (
        "recovery_forward_kl_max", "clip_fraction_max", "ess_fraction_min",
        "arm0_coeff_drift_p95_l2_max", "arm0_topk_overlap_change_min",
        "max_abs_logratio_max", "demo_categorical_forward_kl_max",
        "demo_topk_overlap_change_min",
    ):
        required_numeric[f"outcome_grpo_v2.train_trust_panel.thresholds.{key}"] = (
            thresholds.get(key)
        )
    for name, value in required_numeric.items():
        if not _finite_number(value):
            unresolved.append(name)
    frozen = dict(outcome.get("freeze_evidence", {}) or {})
    for key in (
        "component_projection_report_sha256",
        "round_robin_direction_audit_report_sha256",
        "controller_resume_smoke_report_sha256",
        "frozen_recipe_sha256",
    ):
        if not _valid_sha256(frozen.get(key)):
            unresolved.append(f"outcome_grpo_v2.freeze_evidence.{key}")
    if not _valid_sha256(panel.get("manifest_sha256")):
        unresolved.append("outcome_grpo_v2.train_trust_panel.manifest_sha256")
    if not isinstance(panel.get("item_count"), int) or int(panel.get("item_count") or 0) <= 0:
        unresolved.append("outcome_grpo_v2.train_trust_panel.item_count")
    empty_terminal = {
        "split": None,
        "path": None,
        "collector_seed": None,
        "collection_format_version": None,
        "manifest_sha256": None,
        "identity_digest": None,
        "collector_source": {
            "scheme": None,
            "sha256": None,
            "files": None,
        },
        "identity_status": "NOT_COLLECTED_METHOD_FREEZE_REQUIRED_FIRST",
    }
    empty_formalization = {
        "pilot_terminal_report_sha256": None,
        "method_freeze_receipt_sha256": None,
        "method_freeze_receipt_created_utc": None,
        "receipt_frozen_recipe_sha256": None,
        "formal_collection_started_utc": None,
        "chronology_status": "NOT_AVAILABLE_PILOT_MUST_PASS_FIRST",
    }
    formalization = dict(lineage.get("formalization", {}) or {})
    if status in {METHOD_STATUS_SCAFFOLD, METHOD_STATUS_FROZEN_PILOT}:
        _require(
            terminal == empty_terminal,
            "pilot/scaffold must not pin or inspect a formal terminal collection",
        )
        _require(
            formalization == empty_formalization,
            "pilot/scaffold must precede every formalization receipt/collection",
        )
        expected_decision = (
            "UNRESOLVED_FREEZE_BEFORE_FRESH_FORMAL_COLLECTION"
            if status == METHOD_STATUS_SCAFFOLD
            else "PILOT_EXPOSED_DEVELOPMENT_ONLY_FORMAL_NOT_COLLECTED"
        )
        if lineage.get("decision_status") != expected_decision:
            unresolved.append("outcome_grpo_v2.validation_lineage.decision_status")
        if status == METHOD_STATUS_SCAFFOLD:
            unresolved.append("outcome_grpo_v2.method_status")
    else:
        if lineage.get("decision_status") != "FROZEN_FRESH_FORMAL_COLLECTION":
            unresolved.append("outcome_grpo_v2.validation_lineage.decision_status")
        formal_requirements = {
            "split": bool(terminal.get("split")),
            "path": bool(terminal.get("path")),
            "collector_seed": terminal.get("collector_seed") == 3,
            "collection_format_version": (
                terminal.get("collection_format_version")
                == recovery.FORMAT_VERSION
            ),
            "manifest_sha256": _valid_sha256(
                terminal.get("manifest_sha256")
            ),
            "identity_digest": _valid_sha256(terminal.get("identity_digest")),
            "identity_status": (
                terminal.get("identity_status")
                == "FROZEN_AUTHENTICATED_FORMAL_UNEXPOSED"
            ),
        }
        for key, passed in formal_requirements.items():
            if not passed:
                unresolved.append(
                    "outcome_grpo_v2.validation_lineage."
                    f"formal_terminal_collection.{key}"
                )
        _require(
            terminal.get("path") != current.get("path")
            and terminal.get("manifest_sha256")
                != current.get("manifest_sha256")
            and terminal.get("identity_digest")
                != current.get("observed_identity_digest"),
            "formal terminal collection must be fresh and distinct from development",
        )
        formal_source = dict(terminal.get("collector_source", {}) or {})
        source_files = formal_source.get("files")
        source_closure_valid = (
            set(formal_source) == {"scheme", "sha256", "files"}
            and formal_source.get("scheme")
                == OUTCOME_COLLECTOR_SOURCE["scheme"]
            and _valid_sha256(formal_source.get("sha256"))
            and isinstance(source_files, list)
            and bool(source_files)
            and source_files == sorted(set(source_files))
            and all(
                isinstance(name, str)
                and name
                and not Path(name).is_absolute()
                and ".." not in Path(name).parts
                for name in source_files
            )
        )
        if not source_closure_valid:
            unresolved.append(
                "outcome_grpo_v2.validation_lineage."
                "formal_terminal_collection.collector_source"
            )
        for key in (
            "pilot_terminal_report_sha256",
            "method_freeze_receipt_sha256",
            "receipt_frozen_recipe_sha256",
        ):
            if not _valid_sha256(formalization.get(key)):
                unresolved.append(
                    f"outcome_grpo_v2.validation_lineage.formalization.{key}"
                )
        if (
            _valid_sha256(formalization.get("receipt_frozen_recipe_sha256"))
            and formalization.get("receipt_frozen_recipe_sha256")
                != frozen.get("frozen_recipe_sha256")
        ):
            unresolved.append(
                "outcome_grpo_v2.validation_lineage.formalization."
                "receipt_frozen_recipe_sha256"
            )
        receipt_time = _parse_utc(
            formalization.get("method_freeze_receipt_created_utc")
        )
        collection_time = _parse_utc(
            formalization.get("formal_collection_started_utc")
        )
        if receipt_time is None:
            unresolved.append(
                "outcome_grpo_v2.validation_lineage.formalization."
                "method_freeze_receipt_created_utc"
            )
        if collection_time is None:
            unresolved.append(
                "outcome_grpo_v2.validation_lineage.formalization."
                "formal_collection_started_utc"
            )
        if (
            receipt_time is None
            or collection_time is None
            or receipt_time >= collection_time
            or formalization.get("chronology_status")
                != "VERIFIED_METHOD_FREEZE_PRECEDES_SEED3_COLLECTION"
        ):
            unresolved.append(
                "outcome_grpo_v2.validation_lineage.formalization.chronology_status"
            )
    launchable = status != METHOD_STATUS_SCAFFOLD and not unresolved
    is_formal = status == METHOD_STATUS_FROZEN_FORMAL
    return {
        "format_version": FORMAT_VERSION,
        "trainer_kind": TRAINER_KIND,
        "method_status": status,
        "artifact_role": artifact["role"],
        "launchable": launchable,
        "unresolved": sorted(set(unresolved)),
        "candidate_emission": is_formal,
        "promotion_eligible": False,
        "official_evaluation_eligible": False,
        "formal_terminal_collection_present": is_formal,
    }


def require_launchable_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    report = validate_scaffold_config(cfg)
    if report["unresolved"]:
        raise UnfrozenPilotError(report["unresolved"])
    # Numeric checks belong here so null placeholders remain valid scaffold.
    outcome = dict(cfg["outcome_grpo_v2"])
    controller = dict(outcome["recovery_kl_controller"])
    OneSidedRecoveryKLController(
        initial_beta=float(controller["initial_beta"]),
        target_kl=float(controller["target_kl"]),
        eta=float(controller["eta"]),
        max_beta=float(controller["max_beta"]),
        interval=int(controller["interval"]),
    )
    panel = dict(outcome["train_trust_panel"])
    TrainPanelThresholds(**dict(panel["thresholds"]))
    _require(
        float(dict(cfg["losses"]["proposal"])["weight"]) >= 0.0,
        "frozen expert-anchor weight must be nonnegative",
    )
    _require(
        float(dict(cfg["losses"]["demo_reference"])["weight"]) >= 0.0,
        "frozen demo-reference weight must be nonnegative",
    )
    return report


def trainer_source_identity(
    root: str | os.PathLike[str] = _ROOT,
) -> dict[str, Any]:
    """Hash the complete inherited-plus-v2 executable source closure."""
    return v1._trainer_source_identity(root=root, files=_V2_SOURCE_FILES)


def assert_trainer_source_identity(
    expected: Mapping[str, Any],
    *,
    root: str | os.PathLike[str] = _ROOT,
) -> None:
    v1._assert_trainer_source_identity(
        expected, root=root, files=_V2_SOURCE_FILES,
    )


def pilot_provenance(
    cfg: Mapping[str, Any],
    *,
    sampler: RoundRobinOutcomeSamplerV3,
    controller: OneSidedRecoveryKLController,
    train_panel: TrainOnlyTrustPanel,
    demo_reference: DemoReferenceAnchorV2,
    global_step: int,
    source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Machine-readable v2 checkpoint provenance; never candidate provenance."""
    launch = require_launchable_config(cfg)
    _require(
        launch["method_status"] == METHOD_STATUS_FROZEN_PILOT
        and launch["artifact_role"] == "development_pilot",
        "pilot provenance cannot bind a formal/scaffold config",
    )
    assert_trainer_source_identity(source_identity)
    demo_reference.assert_seed_unchanged()
    return {
        "format_version": FORMAT_VERSION,
        "trainer_kind": TRAINER_KIND,
        "method_status": METHOD_STATUS_FROZEN_PILOT,
        "launch_validation": launch,
        "resolved_config": _resolved_config_identity(cfg),
        "freeze_evidence": _json_copy(
            cfg["outcome_grpo_v2"]["freeze_evidence"]
        ),
        "frozen_coefficients": {
            "expert_anchor_weight": float(cfg["losses"]["proposal"]["weight"]),
            "expert_anchor_term_enabled": bool(
                float(cfg["losses"]["proposal"]["weight"]) > 0.0
            ),
            "zero_expert_anchor_semantics": (
                "sparse_term_disabled_anchor_states_retained_for_demo_reference"
            ),
            "demo_reference_weight": float(
                cfg["losses"]["demo_reference"]["weight"]
            ),
            "balance_weight": float(cfg["losses"]["balance"]["weight"]),
            "proposal_lr_scale": float(cfg["optim"]["lr_scales"]["proposal"]),
        },
        "trainer_source": _json_copy(source_identity),
        "sampler": sampler.state_dict(global_step),
        "recovery_reference": {
            "kind": "sampled_pl_forward_kl_control_variate",
            "atom": "expm1(current_logprob-old_logprob)-delta",
            "arms": list(range(1, 8)),
            "arm0": "forbidden",
            "aggregation": "equal_replan_within_arm_then_equal_arm",
        },
        "demo_reference": demo_reference.provenance(),
        "controller": controller.state_dict(),
        "train_trust_panel": train_panel.state_dict(),
        "authenticated_data_lineage": _json_copy(
            cfg["outcome_grpo_v2"]["authenticated_data_lineage"]
        ),
        "validation_lineage": _json_copy(
            cfg["outcome_grpo_v2"]["validation_lineage"]
        ),
        "artifact_policy": _json_copy(
            cfg["outcome_grpo_v2"]["artifact_policy"]
        ),
        "candidate_emitted": False,
        "promotion_eligible": False,
        "official_evaluation_eligible": False,
    }


def train_outcome_grpo_v2(
    *,
    config: Mapping[str, Any],
    run_dir: str | os.PathLike[str],
    stop_at: int | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    """Fail before mutation until the projection/smoke/lineage freeze exists.

    The numerical core and checkpointable state are implemented above.  The
    checked-in config deliberately exercises this readiness gate; a later,
    separately reviewed freeze will connect these components to the full v2
    distributed pilot loop.  This function must never fall through to v1.
    """
    del run_dir, stop_at, quiet
    require_launchable_config(config)
    raise OutcomeGRPOV2Error(
        "v2 distributed pilot loop is not frozen in this scaffold revision"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--stop-at", type=int, default=STOP_STEP)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from loom.train.loop import read_config  # noqa: PLC0415

        cfg = read_config(args.config)
        report = train_outcome_grpo_v2(
            config=cfg,
            run_dir=args.run_dir,
            stop_at=args.stop_at,
            quiet=args.quiet,
        )
    except UnfrozenPilotError as exc:
        print(json.dumps({
            "status": METHOD_STATUS_SCAFFOLD,
            "launchable": False,
            "unresolved": list(exc.unresolved),
            "candidate_emitted": False,
        }, indent=2, sort_keys=True), flush=True)
        return 5
    except (OutcomeGRPOV2Error, ValueError) as exc:
        print(f"OUTCOME_GRPO_V2_FAILED: {exc}", flush=True)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0
