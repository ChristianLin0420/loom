#!/usr/bin/env python3
"""Authenticated 400-condition held-out direct-policy validation.

This is the outcome-recovery round-0 promotion measurement, not another
eight-arm collection.  It evaluates exactly one deployed direct ``argmax``
trajectory for every canonical LIBERO validation condition (trials 40--49),
using the same environment and decoder seeds as the immutable recovery
collection.  Candidate outcomes are paired condition-by-condition with the
stored seed-policy arm-0 outcomes.

The entry point is intentionally closed: checkpoint, collection, output, and
worker count are the only CLI inputs.  Policy semantics cannot be overridden.
The descendant recipe, its passed trust gate, byte-identical frozen modules,
the complete validation collection, the exact work order, real simulator, and
non-stub direct policy are authenticated before a result can be marked PASS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch import Tensor  # noqa: E402

from loom.eval import DEFAULT_LIBERO_SUITES, EpisodeResult, EvalProtocol  # noqa: E402
from loom.eval import outcome_recovery as recovery  # noqa: E402
from loom.eval import runner  # noqa: E402


FORMAT_VERSION = 1
METHOD_KIND = "loom_outcome_direct_validation"
DESCENDANT_KIND = "loom_outcome_grpo_round0_proposal_descendant"
OPTIMIZER_STEPS = 400
DESCENDANT_GLOBAL_STEP = recovery.SEED_GLOBAL_STEP + OPTIMIZER_STEPS

POLICY_KW: dict[str, Any] = {
    "embodiment": "libero_franka",
    "allow_stub": False,
    "n_candidates": 1,
    "op_stats": False,
    "gripper_dwell": 1,
    "decoder_samples": 1,
    "duration_normalize_segments": False,
    "_include_q_action": False,
}

_DIRECT_SOURCE_FILES = (
    "contracts.py",
    "scripts/outcome_direct_validation.py",
    "loom/eval/__init__.py",
    "loom/eval/runner.py",
    "loom/eval/libero.py",
    "loom/eval/policy.py",
    "loom/eval/outcome_recovery.py",
    "loom/data/adapters/libero.py",
    "loom/data/canonical.py",
    "loom/data/tower.py",
    "loom/model/estimator.py",
    "loom/heads/proposal.py",
    "loom/heads/decoder.py",
    "loom/train/outcome_grpo_round0.py",
)

_TRUST_CHECKS = {
    "clip_fraction": ("<=", 0.20),
    "ess_fraction": (">=", 0.80),
    "coeff_drift_p95_l1": ("<=", 0.05),
    "live_ops": (">=", 16),
    "expert_target_identity": ("==", True),
    "expert_sparse_ce_no_worsening": ("<=", None),
    "expert_topk_overlap_decline": ("<=", 0.05),
    "nonfinite": ("==", 0),
    "unexpected_gradients": ("==", 0),
}


class DirectValidationError(RuntimeError):
    """A provenance, identity, execution, or completeness invariant failed."""


def _require(condition: Any, message: str) -> None:
    if not bool(condition):
        raise DirectValidationError(message)


def _valid_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _config_hash(config: Mapping[str, Any]) -> str:
    experiment = {key: value for key, value in config.items() if key != "link"}
    return hashlib.blake2b(
        json.dumps(experiment, sort_keys=True, default=str).encode(), digest_size=8,
    ).hexdigest()


def _source_identity() -> dict[str, Any]:
    digest = hashlib.sha256()
    files: dict[str, str] = {}
    for relative in _DIRECT_SOURCE_FILES:
        path = ROOT / relative
        _require(path.is_file(), f"direct-validation source is missing: {path}")
        sha = recovery.sha256_file(path)
        files[relative] = sha
        digest.update(relative.encode("utf-8") + b"\0" + bytes.fromhex(sha) + b"\0")
    return {
        "scheme": "sha256(path-nul-sha256-nul)-v1",
        "sha256": digest.hexdigest(),
        "files": files,
    }


def heldout_items() -> tuple[Any, ...]:
    """The exact 400 validation WorkItems; no CLI can remap them."""
    items = tuple(recovery.collection_items("validation"))
    _require(len(items) == 400, f"validation work count is {len(items)}, expected 400")
    _require({item.episode for item in items} == set(range(40, 50)),
             "validation trials are not exactly 40..49")
    _require(all(item.seed == 0 and item.max_steps == 512 for item in items),
             "validation seed/max-step contract drifted")
    _require(tuple(dict.fromkeys(item.suite for item in items))
             == tuple(DEFAULT_LIBERO_SUITES), "validation suite order drifted")
    _require(len({item.key() for item in items}) == len(items),
             "validation work contains duplicate conditions")
    return items


def work_identity(items: Sequence[Any]) -> dict[str, Any]:
    rows = [item.to_dict() for item in items]
    return {
        "scheme": "sha256(canonical-json-work-list)-v1",
        "sha256": hashlib.sha256(_canonical_json(rows)).hexdigest(),
        "n_conditions": len(rows),
        "trial_ids": list(range(40, 50)),
        "protocol_seed": 0,
        "order": "collection_items(validation)",
    }


def direct_protocol() -> EvalProtocol:
    return EvalProtocol(
        bench="libero",
        episodes_per_task=10,
        n_tasks=10,
        suites=DEFAULT_LIBERO_SUITES,
        seeds=(0,),
        max_steps=512,
        notes=(
            "Held-out outcome-recovery validation: exactly one direct argmax "
            "rollout for LIBERO init-state trials 40..49; EpisodeResult.episode "
            "is the absolute trial id, and work_identity authenticates the mapping."
        ),
    )


def _update_tensor_hash(digest: Any, name: str, value: Tensor) -> int:
    tensor = value.detach().cpu().contiguous()
    header = json.dumps(
        {"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    digest.update(header + b"\0")
    digest.update(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()))
    digest.update(b"\0")
    return int(tensor.numel() * tensor.element_size())


def _model_digest(state: Mapping[str, Tensor], *, proposal: bool) -> dict[str, Any]:
    digest = hashlib.sha256()
    n_tensors = n_bytes = 0
    for name in sorted(state):
        if str(name).startswith("proposal.") != bool(proposal):
            continue
        value = state[name]
        _require(isinstance(value, Tensor), f"model state {name!r} is not a tensor")
        n_bytes += _update_tensor_hash(digest, str(name), value)
        n_tensors += 1
    _require(n_tensors > 0, "model digest selected zero tensors")
    return {"sha256": digest.hexdigest(), "n_tensors": n_tensors,
            "n_bytes": n_bytes}


def _finite_tree(value: Any) -> bool:
    if isinstance(value, Tensor):
        return bool(torch.isfinite(value.detach()).all())
    if isinstance(value, Mapping):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _check_threshold(value: Any, op: str, threshold: Any) -> bool:
    if op == "<=":
        return float(value) <= float(threshold)
    if op == ">=":
        return float(value) >= float(threshold)
    if op == "==":
        return value == threshold
    return False


def _validate_trust_gate(value: Any) -> None:
    _require(isinstance(value, Mapping) and value.get("passed") is True,
             "descendant lacks a passed terminal trust gate")
    checks = value.get("checks")
    _require(isinstance(checks, Mapping) and set(checks) == set(_TRUST_CHECKS),
             "terminal trust-gate check set differs")
    for name, (expected_op, locked_threshold) in _TRUST_CHECKS.items():
        row = checks[name]
        _require(isinstance(row, Mapping) and row.get("pass") is True,
                 f"terminal trust check {name} did not pass")
        _require(row.get("op") == expected_op,
                 f"terminal trust check {name} operator differs")
        threshold = row.get("threshold")
        if locked_threshold is not None:
            _require(threshold == locked_threshold,
                     f"terminal trust check {name} threshold differs")
        _require(_finite_tree(row.get("value")) and _finite_tree(threshold),
                 f"terminal trust check {name} is nonfinite")
        _require(_check_threshold(row.get("value"), expected_op, threshold),
                 f"terminal trust check {name} claims PASS inconsistently")
    counts = value.get("counts")
    _require(isinstance(counts, Mapping)
             and int(counts.get("training_nonfinite", -1)) == 0
             and int(counts.get("final_nonfinite", -1)) == 0,
             "terminal trust gate has nonfinite events")
    definitions = value.get("definitions")
    _require(isinstance(definitions, Mapping)
             and definitions.get("scope") == "disjoint recovery validation split"
             and int(definitions.get("arm0_importance_ratios", -1)) == 0,
             "terminal trust-gate scope/arm0 semantics differ")


def _validate_optimizer(value: Any) -> None:
    _require(isinstance(value, Mapping)
             and value.get("kind") == "proposal_only_adamw"
             and value.get("state_reset_at_entry") is True,
             "descendant optimizer is not reset proposal-only AdamW")
    names = value.get("parameter_names")
    state_dict = value.get("state_dict")
    _require(isinstance(names, list) and names and len(set(names)) == len(names),
             "descendant optimizer parameter names are invalid")
    _require(isinstance(state_dict, Mapping), "descendant optimizer state is absent")
    groups = state_dict.get("param_groups")
    states = state_dict.get("state")
    _require(isinstance(groups, list) and len(groups) == 1 and isinstance(states, Mapping),
             "descendant optimizer state topology differs")
    group = groups[0]
    expected = {"lr": 5e-6, "betas": (0.9, 0.95), "weight_decay": 0.05,
                "eps": 1e-8}
    for key, locked in expected.items():
        got = tuple(group.get(key)) if key == "betas" else group.get(key)
        _require(got == locked, f"descendant optimizer {key} differs")
    params = list(group.get("params") or [])
    _require(len(params) == len(names) and set(params) == set(states),
             "descendant optimizer parameter/state coverage differs")
    for parameter in params:
        row = states[parameter]
        _require(isinstance(row, Mapping), "optimizer parameter state is invalid")
        step = row.get("step")
        _require(isinstance(step, Tensor) and step.numel() == 1
                 and int(float(step.detach().cpu())) == OPTIMIZER_STEPS,
                 "optimizer state does not prove exactly 400 reset-stage steps")
    _require(_finite_tree(state_dict), "descendant optimizer state is nonfinite")


def _validate_recipe(recipe: Any) -> None:
    _require(isinstance(recipe, Mapping), "resolved config lacks outcome_grpo")
    expected_keys = {
        "format_version", "algorithm", "reward_normalisation", "aggregation",
        "arm0", "sampled_arms", "clip_eps", "epochs", "groups_per_epoch",
        "optimizer_steps", "optimizer", "objectives", "train_collection_identity",
        "validation_collection_identity", "trust_thresholds",
    }
    _require(set(recipe) == expected_keys, "resolved outcome_grpo fields differ")
    _require(recipe["format_version"] == 1
             and recipe["algorithm"] == "stored_order_pl_clipped_grpo"
             and recipe["sampled_arms"] == list(range(1, 8))
             and recipe["clip_eps"] == 0.20
             and recipe["epochs"] == 2
             and recipe["groups_per_epoch"] == 200
             and recipe["optimizer_steps"] == OPTIMIZER_STEPS,
             "resolved outcome_grpo round-0 contract differs")
    _require(recipe["arm0"] == "normalisation/control only; no ratio or gradient",
             "resolved outcome_grpo arm0 semantics differ")
    _require(recipe["reward_normalisation"] == "complete-group population mean/std"
             and recipe["aggregation"]
             == "mean_replans_then_mean_sampled_trajectories",
             "resolved outcome_grpo objective semantics differ")
    _require(recipe["optimizer"] == {
        "kind": "AdamW", "lr": 5e-6, "betas": [0.9, 0.95],
        "weight_decay": 0.05, "eps": 1e-8, "scheduler": None,
        "grad_clip": 1.0, "proposal_state_reset_at_entry": True,
    }, "resolved outcome_grpo optimizer recipe differs")
    _require(recipe["objectives"] == {
        "grpo": 1.0, "sparse_ce": 1.0, "switch_balance": 1e-2,
    }, "resolved outcome_grpo objective weights differ")
    _require(recipe["trust_thresholds"] == {
        "max_clip_fraction": 0.20, "min_ess_fraction": 0.80,
        "max_coeff_drift_p95_l1": 0.05, "min_live_ops": 16,
        "expert_sparse_ce_no_worsening": True,
        "max_topk_overlap_decline": 0.05, "nonfinite": 0,
    }, "resolved outcome_grpo trust thresholds differ")
    for field in ("train_collection_identity", "validation_collection_identity"):
        _require(_valid_sha256(recipe[field]), f"resolved {field} is invalid")


def validate_descendant_payload(
    payload: Mapping[str, Any],
    *,
    checkpoint_identity: Mapping[str, Any],
    parent_identity: Mapping[str, Any],
    parent_state: Mapping[str, Tensor],
    trainer_source: Mapping[str, Any],
) -> dict[str, Any]:
    """Pure, exact schema/state authentication used by production and tests."""
    _require(isinstance(payload, Mapping), "descendant checkpoint is not a mapping")
    _require("outcome_grpo_round0" not in payload,
             "ambiguous legacy outcome_grpo_round0 payload is present")
    config = payload.get("resolved_config")
    _require(isinstance(config, Mapping), "descendant has no resolved config")
    _require("outcome_grpo_round0" not in config,
             "ambiguous legacy outcome_grpo_round0 config is present")
    _require(_config_hash(config) == payload.get("config_hash"),
             "descendant resolved config hash mismatch")
    recipe = config.get("outcome_grpo")
    _validate_recipe(recipe)

    provenance = payload.get("outcome_grpo")
    _require(isinstance(provenance, Mapping)
             and provenance.get("kind") == DESCENDANT_KIND
             and provenance.get("format_version") == FORMAT_VERSION,
             "descendant outcome_grpo provenance kind/version differs")
    _require(provenance.get("parent") == dict(parent_identity),
             "descendant parent identity differs from pinned seed")
    _require(provenance.get("parent_config_hash") == recovery.SEED_CONFIG_HASH
             and int(provenance.get("parent_global_step", -1))
             == recovery.SEED_GLOBAL_STEP,
             "descendant parent step/config differs")
    _require(int(payload.get("global_step", -1)) == DESCENDANT_GLOBAL_STEP
             and int(provenance.get("descendant_global_step", -1))
             == DESCENDANT_GLOBAL_STEP
             and int(provenance.get("optimizer_steps", -1)) == OPTIMIZER_STEPS,
             "descendant does not prove the exact 400-step round-0 stage")
    _require(provenance.get("descendant_config_hash") == payload.get("config_hash")
             == checkpoint_identity.get("config_hash"),
             "descendant config identity differs")
    _require(provenance.get("mutated_model_prefixes") == ["proposal."],
             "descendant mutated-prefix contract differs")
    _require(provenance.get("trainer_source") == dict(trainer_source),
             "descendant trainer source is not the current authenticated source")

    training = provenance.get("training")
    _require(isinstance(training, Mapping)
             and int(training.get("optimizer_steps", -1)) == OPTIMIZER_STEPS
             and int(training.get("groups_per_epoch", -1)) == 200
             and int(training.get("nonfinite", -1)) == 0
             and list(training.get("unexpected_gradients") or []) == [],
             "descendant training provenance differs")
    behaviour = provenance.get("behaviour_authentication")
    _require(isinstance(behaviour, Mapping)
             and all(isinstance(behaviour.get(split), Mapping)
                     and behaviour[split].get("passed") is True
                     and int(behaviour[split].get("arm0_ratio_eligible_atoms", -1)) == 0
                     for split in ("train", "validation")),
             "descendant behaviour-policy authentication differs")
    _validate_trust_gate(provenance.get("trust_gate"))
    _validate_optimizer(payload.get("optimizer"))

    state = payload.get("model")
    _require(isinstance(state, Mapping) and state
             and all(isinstance(value, Tensor) for value in state.values()),
             "descendant model is not a flat tensor mapping")
    _require(set(state) == set(parent_state), "descendant model key set differs from seed")
    for name, value in state.items():
        parent_value = parent_state[name]
        _require(value.shape == parent_value.shape and value.dtype == parent_value.dtype,
                 f"descendant tensor geometry differs: {name}")
    _require(_finite_tree(state), "descendant model contains nan/inf")
    frozen_parent = _model_digest(parent_state, proposal=False)
    frozen_descendant = _model_digest(state, proposal=False)
    initial_proposal = _model_digest(parent_state, proposal=True)
    final_proposal = _model_digest(state, proposal=True)
    _require(frozen_parent == frozen_descendant == provenance.get("frozen_model"),
             "descendant frozen modules are not byte-identical to the seed")
    _require(initial_proposal == provenance.get("initial_proposal"),
             "descendant initial proposal witness differs from seed")
    _require(final_proposal == provenance.get("final_proposal"),
             "descendant final proposal witness differs from checkpoint")

    consolidated = payload.get("consolidated")
    _require(isinstance(consolidated, Mapping)
             and consolidated.get("tool") == "loom.train.outcome_grpo_round0"
             and consolidated.get("proposal_only_update") is True
             and consolidated.get("section") == "model"
             and int(consolidated.get("step", -1)) == DESCENDANT_GLOBAL_STEP
             and consolidated.get("frozen_model_sha256") == frozen_parent["sha256"],
             "descendant consolidated/proposal-only provenance differs")
    _require(payload.get("stop_reason") == "outcome_grpo_round0_terminal_trust_pass"
             and int(payload.get("world_size", -1)) == 1,
             "descendant terminal/world-size provenance differs")
    return {
        "recipe": dict(recipe), "provenance": dict(provenance),
        "frozen_model": frozen_descendant, "proposal": final_proposal,
    }


def _torch_load(path: Path) -> Mapping[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        value = torch.load(path, map_location="cpu", weights_only=False)
    _require(isinstance(value, Mapping), f"checkpoint {path} is not a mapping")
    return value


@dataclass(frozen=True)
class DescendantCheckpoint:
    identity: dict[str, Any]
    parent_identity: dict[str, Any]
    recipe: dict[str, Any]
    provenance: dict[str, Any]


def authenticate_descendant_checkpoint(path: str | os.PathLike[str]) -> DescendantCheckpoint:
    candidate = Path(path).expanduser().resolve()
    _require(candidate.is_file(), f"descendant checkpoint is missing: {candidate}")
    before = candidate.stat()
    sha = recovery.sha256_file(candidate)
    payload = _torch_load(candidate)
    after = candidate.stat()
    _require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
             "descendant checkpoint changed while authenticating")
    identity = {
        "kind": DESCENDANT_KIND, "path": str(candidate),
        "size": int(before.st_size), "mtime_ns": int(before.st_mtime_ns),
        "sha256": sha, "global_step": int(payload.get("global_step", -1)),
        "config_hash": str(payload.get("config_hash") or ""),
    }
    _require(_valid_sha256(sha), "descendant checkpoint SHA-256 is invalid")
    provenance = payload.get("outcome_grpo")
    _require(isinstance(provenance, Mapping), "descendant lacks outcome_grpo provenance")
    parent_record = provenance.get("parent")
    _require(isinstance(parent_record, Mapping) and parent_record.get("path"),
             "descendant parent identity is absent")
    parent_identity = recovery.authenticate_seed_checkpoint(parent_record["path"])
    _require(dict(parent_record) == parent_identity,
             "descendant parent record differs from authenticated seed")
    parent_payload = _torch_load(Path(parent_identity["path"]))
    _require(int(parent_payload.get("global_step", -1)) == recovery.SEED_GLOBAL_STEP
             and parent_payload.get("config_hash") == recovery.SEED_CONFIG_HASH,
             "seed checkpoint embedded identity differs")
    parent_state = parent_payload.get("model")
    _require(isinstance(parent_state, Mapping), "seed checkpoint model is absent")

    from loom.train import outcome_grpo_round0 as round0  # noqa: PLC0415

    _require(round0.TRAINER_KIND == DESCENDANT_KIND,
             "current canonical trainer kind differs")
    trainer_source = round0._source_identity()
    result = validate_descendant_payload(
        payload, checkpoint_identity=identity, parent_identity=parent_identity,
        parent_state=parent_state, trainer_source=trainer_source,
    )
    del payload, parent_payload
    return DescendantCheckpoint(
        identity=identity, parent_identity=parent_identity,
        recipe=result["recipe"], provenance=result["provenance"],
    )


def _assert_checkpoint_unchanged(identity: Mapping[str, Any]) -> None:
    path = Path(str(identity["path"]))
    stat = path.stat()
    _require(int(stat.st_size) == int(identity["size"])
             and int(stat.st_mtime_ns) == int(identity["mtime_ns"]),
             f"checkpoint stat changed during validation: {path}")
    _require(recovery.sha256_file(path) == identity["sha256"],
             f"checkpoint bytes changed during validation: {path}")


def _open_validation_collection(path: str | os.PathLike[str],
                                parent_identity: Mapping[str, Any]) -> Any:
    from loom.train.outcome_grpo_round0 import ValidatedRecoveryCollection  # noqa: PLC0415

    return ValidatedRecoveryCollection.open(
        path, checkpoint_identity=parent_identity, purpose="validation",
    )


def authenticate_collection_binding(collection: Any,
                                    checkpoint: DescendantCheckpoint) -> dict[str, Any]:
    items = heldout_items()
    _require(tuple(collection.items) == items,
             "authenticated collection WorkItems differ from canonical validation")
    actual = collection.provenance()
    recorded = checkpoint.provenance.get("validation_collection")
    _require(isinstance(recorded, Mapping) and dict(recorded) == actual,
             "descendant validation-collection provenance differs from input")
    identity_digest = str(collection.identity_digest)
    _require(checkpoint.recipe["validation_collection_identity"] == identity_digest,
             "descendant config validation identity differs from input")
    _require(len(collection.receipts) == len(items) == 400,
             "validation collection is not exactly 400 complete groups")
    behaviour = checkpoint.provenance["behaviour_authentication"]["validation"]
    _require(behaviour.get("passed") is True,
             "descendant validation behaviour authentication did not pass")
    return actual


def _parent_controls(collection: Any) -> dict[tuple, dict[str, Any]]:
    controls: dict[tuple, dict[str, Any]] = {}
    for item, receipt in zip(collection.items, collection.receipts):
        group_id = recovery.work_key(item)
        _require(receipt.get("group_id") == group_id,
                 f"collection receipt order differs at {group_id}")
        rewards = list(receipt.get("terminal_rewards") or [])
        _require(len(rewards) == recovery.GROUP_SIZE
                 and all(int(value) in (0, 1) for value in rewards),
                 f"collection terminal rewards are invalid at {group_id}")
        controls[item.key()] = {
            "group_id": group_id,
            "success": bool(int(rewards[0])),
            "source": "validation manifest terminal_rewards[0]",
        }
    _require(len(controls) == 400, "parent arm0 control mapping is incomplete")
    return controls


def _attach_control(record: EpisodeResult, controls: Mapping[tuple, Mapping[str, Any]],
                    validation_identity: str) -> None:
    _require(record.key() in controls, f"runner returned unknown condition {record.key()}")
    record.extra["outcome_recovery_parent_arm0"] = {
        **dict(controls[record.key()]),
        "validation_identity_digest": validation_identity,
    }


def _validate_policy_provenance(value: Any, checkpoint: DescendantCheckpoint) -> None:
    _require(isinstance(value, Mapping), "direct policy provenance is absent")
    _require(value.get("policy") == "LoomPolicy" and value.get("is_stub") is False,
             "held-out validation did not run a real LoomPolicy")
    _require(value.get("ckpt_global_step") == DESCENDANT_GLOBAL_STEP
             and value.get("ckpt_config_hash") == checkpoint.identity["config_hash"],
             "direct policy loaded a different checkpoint identity")
    _require(str(Path(str(value.get("ckpt"))).expanduser().resolve())
             == checkpoint.identity["path"],
             "direct policy loaded a different checkpoint path")
    _require(value.get("gripper_dwell") == 1
             and value.get("decoder_samples") == 1
             and value.get("duration_normalize_segments") is False,
             "direct policy inference semantics differ")
    state = value.get("state_dict")
    _require(isinstance(state, Mapping) and set(state) == {
        "estimator", "proposal", "decoder",
    }, "direct policy did not load exactly estimator/proposal/decoder")
    for name in ("estimator", "proposal", "decoder"):
        row = state[name]
        _require(int(row.get("tensors_loaded", 0)) > 0
                 and int(row.get("unexpected", -1)) == 0,
                 f"direct policy {name} tensor load is unauthenticated")


def _validate_records(store: Any, items: Sequence[Any],
                      controls: Mapping[tuple, Mapping[str, Any]],
                      validation_identity: str, *, complete: bool) -> None:
    expected = {item.key(): item for item in items}
    _require(set(store.records).issubset(expected),
             "result store contains a non-validation condition")
    if complete:
        _require(set(store.records) == set(expected),
                 f"direct validation is incomplete: {len(store.records)}/400")
    for key, record in store.records.items():
        item = expected[key]
        _require(record.env_seed == item.env_seed
                 and record.extra.get("policy_seed") == item.policy_seed,
                 f"condition seed provenance differs at {recovery.work_key(item)}")
        _require(record.error is None,
                 f"direct validation episode errored at {recovery.work_key(item)}")
        _require(1 <= int(record.steps) <= item.max_steps,
                 f"direct validation step count is invalid at {recovery.work_key(item)}")
        paired = record.extra.get("outcome_recovery_parent_arm0")
        expected_pair = {**controls[key],
                         "validation_identity_digest": validation_identity}
        _require(paired == expected_pair,
                 f"parent arm0 pairing differs at {recovery.work_key(item)}")


def paired_summary(records: Iterable[EpisodeResult]) -> dict[str, Any]:
    counts = {name: 0 for name in ("n", "candidate", "parent_arm0", "both",
                                   "candidate_only", "parent_only", "neither")}
    per_suite: dict[str, dict[str, int]] = {}
    for record in records:
        parent = bool(record.extra["outcome_recovery_parent_arm0"]["success"])
        candidate = bool(record.success)
        row = per_suite.setdefault(record.suite, {name: 0 for name in counts})
        for target in (counts, row):
            target["n"] += 1
            target["candidate"] += int(candidate)
            target["parent_arm0"] += int(parent)
            target["both"] += int(candidate and parent)
            target["candidate_only"] += int(candidate and not parent)
            target["parent_only"] += int(parent and not candidate)
            target["neither"] += int(not candidate and not parent)
    _require(counts["n"] == sum(row["n"] for row in per_suite.values()),
             "paired summary count mismatch")
    return {"aggregate": counts,
            "per_suite": {suite: per_suite[suite]
                          for suite in DEFAULT_LIBERO_SUITES}}


def _resume_policy(path: Path, store: Any) -> None:
    if not store.n_resumed:
        return
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    _require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
             and len(raw) == before.st_size,
             "resumed result changed while authenticating metadata")
    blob = json.loads(raw)
    old_meta = blob.get("meta")
    _require(isinstance(old_meta, Mapping)
             and old_meta.get("eval_identity") == store.meta.get("eval_identity"),
             "resumed result metadata identity differs")
    policy = old_meta.get("policy")
    _require(isinstance(policy, Mapping),
             "resumed episodes lack authenticated policy provenance")
    episodes = blob.get("episodes")
    _require(isinstance(episodes, list) and len(episodes) == store.n_resumed,
             "resumed result contains duplicate or malformed episode records")
    keys = [EpisodeResult.from_dict(row).key() for row in episodes]
    _require(len(keys) == len(set(keys)), "resumed result contains duplicate episodes")
    store.meta["policy"] = dict(policy)


def _repair_atomic_pairing_window(
    store: Any, controls: Mapping[tuple, Mapping[str, Any]],
    validation_identity: str,
) -> None:
    """Repair only the deterministic add-before-callback crash window.

    ``runner._run_parallel`` atomically flushes an episode before invoking its
    callback.  A kill in those few instructions can therefore leave a valid
    episode without our derived parent-arm0 annotation.  The annotation is a
    pure function of the already-authenticated manifest and WorkItem, so adding
    a *missing* field is safe; a present-but-different field still fails closed.
    """
    changed = False
    for record in store.records.values():
        if "outcome_recovery_parent_arm0" not in record.extra:
            _attach_control(record, controls, validation_identity)
            changed = True
    if changed:
        store.flush()


def run_direct_validation(
    *, checkpoint_path: str | os.PathLike[str],
    validation_collection: str | os.PathLike[str],
    out: str | os.PathLike[str], workers: int | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    source_before = _source_identity()
    checkpoint = authenticate_descendant_checkpoint(checkpoint_path)
    collection = _open_validation_collection(
        validation_collection, checkpoint.parent_identity,
    )
    collection_provenance = authenticate_collection_binding(collection, checkpoint)
    items = heldout_items()
    work = work_identity(items)
    controls = _parent_controls(collection)
    validation_identity = str(collection.identity_digest)

    mod = runner.bench_module("libero")
    _require(runner.env_available(mod),
             "real LIBERO is unavailable; held-out validation has no fake mode")
    runner.ensure_runtime(mod)
    protocol = direct_protocol()
    eval_identity = {
        "version": FORMAT_VERSION, "kind": METHOD_KIND,
        "checkpoint": checkpoint.identity,
        "parent_seed": checkpoint.parent_identity,
        "validation_collection": collection_provenance,
        "work": work,
        "policy": {
            "path": "estimator->proposal.argmax->decoder",
            "kwargs": dict(POLICY_KW), "q_action": False, "bank": False,
            "fallback": False, "episodes_per_condition": 1,
        },
        "backend": "libero", "source": source_before,
        "pairing": "same WorkItem/env_seed/policy_seed vs manifest arm0",
    }
    out_path = Path(out).expanduser().resolve()
    _require(out_path.suffix == ".json", "--out must name a JSON result file")
    store = runner.ResultStore(
        out_path, protocol, resume=True,
        meta={
            "ckpt": checkpoint.identity["path"], "bench": "libero",
            "backend": "libero", "env_available": True,
            "libero_available": True, "eval_identity": eval_identity,
            "policy_seed_scheme": runner.POLICY_SEED_SCHEME,
            "method": METHOD_KIND, "source": source_before,
            "checkpoint": checkpoint.identity,
            "validation_collection": collection_provenance,
            "work": work, "direct_validation": {"status": "RUNNING"},
            **runner.code_provenance(),
        },
    )
    _resume_policy(out_path, store)
    if store.n_resumed:
        _repair_atomic_pairing_window(store, controls, validation_identity)
    if store.n_resumed:
        _validate_policy_provenance(store.meta.get("policy"), checkpoint)
    _validate_records(store, items, controls, validation_identity, complete=False)
    todo = [item for item in items if not store.has(item.key())]
    n_workers = workers if workers is not None else runner.n_devices()
    n_workers = max(1, min(int(n_workers), max(1, len(todo))))

    completed = len(items) - len(todo)

    def on_episode(record: EpisodeResult) -> None:
        nonlocal completed
        _attach_control(record, controls, validation_identity)
        completed += 1
        if not quiet:
            print(
                f"[outcome-direct-validation] {completed}/400 "
                f"{recovery.work_key(items_by_key[record.key()])} "
                f"candidate={int(bool(record.success))} "
                f"parent_arm0={int(controls[record.key()]['success'])}",
                flush=True,
            )

    items_by_key = {item.key(): item for item in items}
    if todo and n_workers > 1:
        runner._run_parallel(
            todo, store, "libero", checkpoint.identity["path"], "libero",
            dict(POLICY_KW), n_workers, on_episode,
        )
    elif todo:
        policy = runner._default_policy(checkpoint.identity["path"], dict(POLICY_KW))
        store.meta["policy"] = runner._provenance(policy)
        _validate_policy_provenance(store.meta["policy"], checkpoint)
        for item in todo:
            record = runner._run_item(item, policy, mod, None, "libero")
            _attach_control(record, controls, validation_identity)
            completed += 1
            store.add(record)
            if not quiet:
                print(
                    f"[outcome-direct-validation] {completed}/400 "
                    f"{recovery.work_key(item)} candidate={int(bool(record.success))} "
                    f"parent_arm0={int(controls[record.key()]['success'])}",
                    flush=True,
                )

    # Parallel callbacks run after ResultStore.add; this final flush persists
    # their pairing annotation for the last completed episode as well.
    store.flush()
    _validate_policy_provenance(store.meta.get("policy"), checkpoint)
    _validate_records(store, items, controls, validation_identity, complete=True)
    collection.assert_unchanged()
    _assert_checkpoint_unchanged(checkpoint.identity)
    _assert_checkpoint_unchanged(checkpoint.parent_identity)
    _require(_source_identity() == source_before,
             "direct-validation source changed during execution")
    summary = paired_summary(store.episodes())
    store.meta["paired_outcomes"] = summary
    store.meta["direct_validation"] = {
        "status": "PASS", "n_conditions": 400, "n_errors": 0,
        "candidate_checkpoint_sha256": checkpoint.identity["sha256"],
        "validation_identity_digest": validation_identity,
        "work_sha256": work["sha256"],
    }
    store.flush()
    result = store.to_dict()
    _require(result["summary"]["complete"] is True
             and result["summary"]["n_errors"] == 0
             and result["summary"]["n_episodes"] == 400,
             "terminal result summary is not complete/error-free")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True,
                        help="authenticated round-0 proposal descendant")
    parser.add_argument("--validation-collection", required=True,
                        help="terminal COMPLETE recovery validation directory")
    parser.add_argument("--out", required=True,
                        help="new or identity-matching resumable JSON result")
    parser.add_argument("--workers", type=int, default=None,
                        help="one real-policy simulator worker per visible GPU")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_direct_validation(
            checkpoint_path=args.checkpoint,
            validation_collection=args.validation_collection,
            out=args.out, workers=args.workers, quiet=args.quiet,
        )
    except (DirectValidationError, FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"OUTCOME_DIRECT_VALIDATION_FAILED: {exc}", flush=True)
        return 2
    print(json.dumps({
        "status": "PASS", "summary": result["summary"],
        "paired_outcomes": result["meta"]["paired_outcomes"],
        "out": str(Path(args.out).expanduser().resolve()),
    }, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
