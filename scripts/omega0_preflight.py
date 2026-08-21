#!/usr/bin/env python3
"""Deterministic pre-update gate for the reserve QA-omega0 transition.

This gate consumes the verified step-49,666 deploy checkpoint and the declared
``r0a_bank_ca_qa_omega0`` config.  It builds and strictly restores the real
model on CPU, changes only ``bank.omega`` to bit-exact zero in memory, and
checks that every other checkpoint tensor (including the complete direct
estimator/proposal/decoder path) remains exact.  It then evaluates sixteen
fixed action-labelled windows from the authenticated ``demo_49`` holdout.

The transition passes only when the mean sequential bank-rollout error is
strictly below the mean identity error at every one of the four horizons.
Authentication, completeness, and finite-number failures are errors, never
negative scientific results.  The JSON report is published atomically and
exclusively: an existing result is never overwritten.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import Tensor, nn

import contracts as C
from loom.losses.dyn import ln_cosine_distance, sequential_rollout
from scripts import bank_ca_gate as bank_gate


FORMAT_VERSION = 1
BEHAVIOR_SOURCE_DIGEST_SCHEME = "sha256(relpath-nul-file-sha256-nul)-v1"
BEHAVIOR_SOURCE_FILES = (
    "contracts.py",
    "loom/data/adapters/libero.py",
    "loom/data/cache.py",
    "loom/data/canonical.py",
    "loom/data/loader.py",
    "loom/heads/decoder.py",
    "loom/heads/proposal.py",
    "loom/heads/q_action.py",
    "loom/heads/q_delta.py",
    "loom/losses/dyn.py",
    "loom/model/bank.py",
    "loom/model/estimator.py",
    "loom/model/rollout.py",
    "loom/train/determinism.py",
    "loom/train/loop.py",
    "loom/train/schedule.py",
    "scripts/bank_ca_gate.py",
    "scripts/omega0_preflight.py",
    "stubs.py",
)


class PreflightError(RuntimeError):
    """An integrity or completeness failure; the reserve run must not start."""


@dataclass(frozen=True)
class PreflightPins:
    deploy_checkpoint_sha256: str
    deploy_config_hash: str
    deploy_global_step: int
    target_config_hash: str
    transition_reset: Mapping[str, Any]
    manifest_digest: str
    cache_manifest_sha256: str
    loader_n_windows: int
    windows: int
    batch_size: int
    seed: int
    selected_indices: tuple[int, ...]
    selected_records_sha256: str


DEFAULT_PINS = PreflightPins(
    deploy_checkpoint_sha256=(
        "15f286c268caa5327d5aa3abf1f67ebd0555c426a509fef22cb7f537bf6ab4e1"
    ),
    deploy_config_hash="a199324a6205bb6d",
    deploy_global_step=49_666,
    target_config_hash="7a5e8a24327ecc0c",
    transition_reset={
        "source_config_hash": "a199324a6205bb6d",
        "tensors": {"bank.omega": "zero"},
    },
    manifest_digest=(
        "sha256:6d933080a8c902048a50ef2a4805d98318b6ff233f909716eee2086a27886057"
    ),
    cache_manifest_sha256=(
        "0ad6348be15d6baee4563f2b426d16b1b19fa87c74751b697ee8d7cd11144102"
    ),
    loader_n_windows=918,
    windows=16,
    batch_size=4,
    seed=0,
    selected_indices=(
        205, 275, 466, 877, 896, 530, 565, 375,
        607, 474, 704, 660, 797, 410, 688, 334,
    ),
    selected_records_sha256=(
        "aa06b60efcd4206a4c5dfe6f79d1c32d1f766787f216e94c78f521ff36b7725a"
    ),
)

DIRECT_POLICY_MODULES = ("estimator", "proposal", "decoder")
RESET_TENSOR = "bank.omega"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PreflightError(message)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def sha256_file(path: str | os.PathLike, chunk_bytes: int = 8 << 20) -> str:
    source = Path(path).expanduser().resolve()
    _require(source.is_file(), f"required file does not exist: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _behavior_digest_from_entries(entries: Sequence[Mapping[str, str]]) -> str:
    digest = hashlib.sha256()
    digest.update(BEHAVIOR_SOURCE_DIGEST_SCHEME.encode("utf-8") + b"\0")
    for entry in entries:
        name = str(entry["path"])
        file_digest = str(entry["sha256"])
        _require(_is_sha256(file_digest), f"invalid source SHA-256 for {name!r}")
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(bytes.fromhex(file_digest) + b"\0")
    return digest.hexdigest()


def behavior_source_provenance(
    root: str | os.PathLike = ROOT,
    files: Sequence[str] = BEHAVIOR_SOURCE_FILES,
) -> dict[str, Any]:
    """Hash the explicit behavior-bearing source closure in canonical order."""
    source_root = Path(root).expanduser().resolve()
    names = tuple(sorted(str(name) for name in files))
    _require(bool(names), "behavior-source set is empty")
    _require(len(names) == len(set(names)), "behavior-source set has duplicates")
    entries: list[dict[str, str]] = []
    for name in names:
        relative = Path(name)
        _require(
            name == relative.as_posix()
            and not relative.is_absolute()
            and ".." not in relative.parts,
            f"invalid behavior-source path: {name!r}",
        )
        path = (source_root / relative).resolve()
        try:
            path.relative_to(source_root)
        except ValueError as exc:
            raise PreflightError(f"behavior source escapes repository: {name}") from exc
        _require(path.is_file(), f"required behavior source is missing: {name}")
        entries.append({"path": name, "sha256": sha256_file(path)})
    return {
        "behavior_source_digest_scheme": BEHAVIOR_SOURCE_DIGEST_SCHEME,
        "behavior_source_digest": _behavior_digest_from_entries(entries),
        "behavior_source_files": entries,
    }


def assert_behavior_source_digest(
    expected: str,
    *,
    root: str | os.PathLike = ROOT,
    files: Sequence[str] = BEHAVIOR_SOURCE_FILES,
) -> dict[str, Any]:
    _require(_is_sha256(expected), "expected behavior-source digest is invalid")
    current = behavior_source_provenance(root, files)
    _require(
        current["behavior_source_digest"] == expected,
        "behavior-bearing source changed during preflight",
    )
    return current


def authenticate_target_config(
    config_path: str | os.PathLike,
    *,
    pins: PreflightPins = DEFAULT_PINS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve and authenticate the one allowed identity-centred recipe."""
    path = Path(config_path).expanduser().resolve()
    cfg = bank_gate._read_config(path)
    config_hash = bank_gate._experiment_config_hash(cfg)
    _require(
        config_hash == pins.target_config_hash,
        f"target config hash {config_hash} != pinned {pins.target_config_hash}",
    )
    settings = bank_gate._gate_settings(cfg)
    _require(
        settings.get("method_variant") == "joint_q_action_bank_identity_centered",
        "target is not the declared joint QA identity-centred method",
    )
    _require(
        settings.get("transition_parameter_reset") == pins.transition_reset,
        "target transition reset differs from the pinned omega-only reset",
    )
    _require(
        cfg.get("run", {}).get("name") == "r0a_bank_ca_qa_omega0",
        "target run.name is not r0a_bank_ca_qa_omega0",
    )
    return cfg, {
        "path": str(path),
        "file_sha256": sha256_file(path),
        "resolved_config_sha256": _canonical_json_sha256(cfg),
        "config_hash": config_hash,
        "method_variant": settings["method_variant"],
        "transition_parameter_reset": copy.deepcopy(
            settings["transition_parameter_reset"]
        ),
    }


def authenticate_deploy_payload(
    payload: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    target_config: Mapping[str, Any],
    pins: PreflightPins = DEFAULT_PINS,
) -> dict[str, Any]:
    """Authenticate the exact consolidated deploy seed before model building."""
    _require(
        checkpoint_sha256 == pins.deploy_checkpoint_sha256,
        "deploy checkpoint SHA-256 does not match the verified seed",
    )
    _require(isinstance(payload, Mapping), "deploy checkpoint is not a mapping")
    _require(
        payload.get("config_hash") == pins.deploy_config_hash,
        "deploy checkpoint config_hash is not the verified deploy recipe",
    )
    _require(
        payload.get("global_step") == pins.deploy_global_step,
        "deploy checkpoint is not exact step 49,666",
    )
    _require(payload.get("world_size") == 16, "deploy checkpoint world_size is not 16")
    consolidated = payload.get("consolidated")
    _require(isinstance(consolidated, Mapping), "checkpoint is not consolidated")
    _require(
        consolidated.get("tool") == "loom.train.consolidate"
        and consolidated.get("section") == "model"
        and consolidated.get("step") == pins.deploy_global_step
        and consolidated.get("n_shards") == 16,
        "checkpoint consolidation provenance is incomplete or inconsistent",
    )
    state = payload.get("model")
    _require(isinstance(state, Mapping) and bool(state), "checkpoint has no model state")
    _require(
        all(isinstance(name, str) and isinstance(value, Tensor)
            for name, value in state.items()),
        "checkpoint model state is not a flat tensor mapping",
    )
    _require(
        int(consolidated.get("n_keys", -1)) == len(state),
        "checkpoint consolidation key count does not match model state",
    )
    source_cfg = payload.get("resolved_config")
    _require(isinstance(source_cfg, Mapping), "checkpoint has no resolved_config")
    source_hash = bank_gate._experiment_config_hash(source_cfg)
    _require(
        source_hash == pins.deploy_config_hash,
        "embedded deploy config does not reproduce checkpoint config_hash",
    )
    _require(
        target_config.get("model") == source_cfg.get("model"),
        "target model construction differs from the verified deploy checkpoint",
    )
    reset = target_config.get("optim", {}).get("transition_parameter_reset")
    _require(reset == pins.transition_reset, "target reset recipe changed after auth")
    _require(
        reset.get("source_config_hash") == source_hash,
        "target reset does not name the loaded deploy config",
    )
    return {
        "config_hash": source_hash,
        "global_step": int(payload["global_step"]),
        "world_size": int(payload["world_size"]),
        "samples_seen": payload.get("samples_seen"),
        "saved_git_sha": payload.get("git_sha"),
        "resolved_config_sha256": _canonical_json_sha256(source_cfg),
        "consolidated": copy.deepcopy(dict(consolidated)),
        "model_tensors": len(state),
        "model_numel": sum(int(value.numel()) for value in state.values()),
    }


def _update_tensor_digest(digest: Any, name: str, tensor: Tensor) -> None:
    value = tensor.detach()
    _require(value.device.type == "cpu", f"tensor {name} is not on CPU")
    value = value.contiguous()
    digest.update(name.encode("utf-8") + b"\0")
    digest.update(str(value.dtype).encode("ascii") + b"\0")
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode() + b"\0")
    # NumPy cannot expose bfloat16 directly; a byte view preserves exact bits for
    # every dtype and lets hashlib consume the buffer without a second large copy.
    raw = value.reshape(-1).view(torch.uint8).numpy()
    digest.update(memoryview(raw))
    digest.update(b"\0")


def tensor_sha256(tensor: Tensor) -> str:
    digest = hashlib.sha256()
    _update_tensor_digest(digest, "tensor", tensor)
    return digest.hexdigest()


def state_sha256(state: Mapping[str, Tensor], names: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(str(value) for value in names):
        _require(name in state, f"state digest is missing tensor {name}")
        _update_tensor_digest(digest, name, state[name])
    return digest.hexdigest()


def apply_omega_zero_and_audit(
    model: nn.Module,
    source_state: Mapping[str, Tensor],
    *,
    reset_recipe: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the declared reset and prove the complete tensor-level delta."""
    _require(
        reset_recipe == DEFAULT_PINS.transition_reset,
        "reset audit accepts only the pinned bank.omega=zero recipe",
    )
    live_before = model.state_dict()
    _require(
        set(live_before) == set(source_state),
        "strict-loaded model and checkpoint state keys differ",
    )
    _require(RESET_TENSOR in live_before, "model state has no bank.omega")
    parameters = dict(model.named_parameters())
    _require(RESET_TENSOR in parameters, "bank.omega is not a named parameter")

    mismatched_before: list[str] = []
    for name, source in source_state.items():
        live = live_before[name]
        _require(
            live.shape == source.shape and live.dtype == source.dtype,
            f"strict-loaded tensor metadata differs for {name}",
        )
        if source.is_floating_point():
            _require(bool(torch.isfinite(source).all()), f"source tensor {name} is non-finite")
            _require(bool(torch.isfinite(live).all()), f"live tensor {name} is non-finite")
        if not torch.equal(live, source):
            mismatched_before.append(name)
    _require(
        not mismatched_before,
        f"model is not checkpoint-exact before reset: {mismatched_before[:5]}",
    )

    source_omega = source_state[RESET_TENSOR]
    _require(
        bool((source_omega != 0).any()),
        "verified source bank.omega is already all zero; reset witness is vacuous",
    )
    all_names = tuple(source_state)
    non_target_names = tuple(name for name in all_names if name != RESET_TENSOR)
    source_full_digest = state_sha256(source_state, all_names)
    source_non_target_digest = state_sha256(source_state, non_target_names)

    direct_source: dict[str, dict[str, Any]] = {}
    for module in DIRECT_POLICY_MODULES:
        names = tuple(name for name in all_names if name.startswith(module + "."))
        _require(bool(names), f"checkpoint has no direct-policy {module} tensors")
        direct_source[module] = {
            "tensors": len(names),
            "numel": sum(int(source_state[name].numel()) for name in names),
            "source_sha256": state_sha256(source_state, names),
            "_names": names,
        }

    with torch.no_grad():
        parameters[RESET_TENSOR].zero_()

    live_after = model.state_dict()
    _require(set(live_after) == set(source_state), "model state keys changed during reset")
    changed = [
        name for name in all_names
        if not torch.equal(live_after[name], source_state[name])
    ]
    _require(
        changed == [RESET_TENSOR],
        f"reset changed tensors other than bank.omega: {changed}",
    )
    _require(
        bool((live_after[RESET_TENSOR] == 0).all()),
        "bank.omega is not bit-exact zero after reset",
    )
    for name in non_target_names:
        _require(
            torch.equal(live_after[name], source_state[name]),
            f"non-target tensor changed during reset: {name}",
        )
    reset_non_target_digest = state_sha256(live_after, non_target_names)
    _require(
        reset_non_target_digest == source_non_target_digest,
        "aggregate non-target state digest changed during reset",
    )

    direct_report: dict[str, Any] = {}
    for module, item in direct_source.items():
        names = item.pop("_names")
        reset_digest = state_sha256(live_after, names)
        _require(
            reset_digest == item["source_sha256"],
            f"direct-policy module {module} changed during reset",
        )
        direct_report[module] = {
            **item,
            "reset_sha256": reset_digest,
            "tensor_exact": True,
        }

    return {
        "recipe": copy.deepcopy(dict(reset_recipe)),
        "pre_reset_checkpoint_exact": True,
        "source_omega_nonzero": True,
        "source_omega_sha256": tensor_sha256(source_omega),
        "reset_omega_sha256": tensor_sha256(live_after[RESET_TENSOR]),
        "reset_omega_numel": int(live_after[RESET_TENSOR].numel()),
        "reset_omega_bit_exact_zero": True,
        "changed_tensors": changed,
        "all_other_tensors_exact": True,
        "non_target_tensors": len(non_target_names),
        "non_target_numel": sum(int(source_state[name].numel()) for name in non_target_names),
        "source_state_sha256": source_full_digest,
        "reset_state_sha256": state_sha256(live_after, all_names),
        "source_non_target_sha256": source_non_target_digest,
        "reset_non_target_sha256": reset_non_target_digest,
        "direct_policy": direct_report,
    }


def authenticate_selection(
    loader: Any,
    *,
    pins: PreflightPins = DEFAULT_PINS,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    """Authenticate the heldout source and return its fixed dataset rows."""
    source, digest, manifest_source, task_for_trajectory = (
        bank_gate._loader_source_provenance(loader)
    )
    _require(digest == pins.manifest_digest, "heldout trajectory manifest changed")
    _require(manifest_source == "libero", "heldout source is not LIBERO")
    _require(
        source.get("feature_cache_manifest_sha256") == pins.cache_manifest_sha256,
        "feature-cache manifest SHA-256 changed",
    )
    _require(
        source.get("loader_n_windows") == pins.loader_n_windows,
        "heldout loader window count changed",
    )
    _require(source.get("sampling") == "uniform_task", "heldout sampling changed")
    _require(getattr(loader, "batch_size", None) == pins.batch_size, "batch size changed")
    _require(getattr(loader, "num_workers", None) == 0, "workers must be exactly zero")
    datasets = getattr(loader, "datasets", {})
    _require(len(datasets) == 1, "preflight requires one heldout embodiment")
    embodiment, dataset = next(iter(datasets.items()))
    _require(str(embodiment) == "libero_franka", "heldout embodiment changed")
    indices = bank_gate._first_sampler_indices(
        loader.sampler, str(embodiment), pins.windows,
    )
    got_indices = tuple(int(value) for value in indices)
    _require(got_indices == pins.selected_indices, "fixed heldout window indices changed")
    records = bank_gate._selected_window_records(
        loader, indices, task_for_trajectory,
    )
    records_digest = _canonical_json_sha256(records)
    _require(
        records_digest == pins.selected_records_sha256,
        "fixed heldout window identities changed",
    )
    _require(len(records) == pins.windows, "fixed heldout record count is incomplete")
    _require(
        len({row["task_id"] for row in records}) == pins.windows,
        "fixed heldout rows no longer cover sixteen distinct tasks",
    )
    source = {
        **source,
        "fixed_selection": {
            "scheme": "first N seed-0 uniform_task rows",
            "seed": pins.seed,
            "indices": list(got_indices),
            "records_sha256": records_digest,
            "records": copy.deepcopy(records),
            "n_distinct_tasks": len({row["task_id"] for row in records}),
        },
    }
    return dataset, records, source


@torch.inference_mode()
def measure_error_batch(
    model: Any,
    batch: Mapping[str, Any],
) -> tuple[Tensor, Tensor, Tensor]:
    """Return identity error, sequential-rollout error, and the action code."""
    bank_gate._batch_size(batch)
    _require(
        len(batch.get("burn_in_feats", ())) == 4,
        "preflight requires exact B4 recurrent burn-in",
    )
    _require(
        len(batch.get("feats", ())) == C.DEPTH + 1,
        f"preflight requires exactly {C.DEPTH + 1} main states",
    )
    actions = batch.get("actions")
    _require(isinstance(actions, Tensor), "preflight window has no action labels")

    zs = model.beliefs(batch)
    targets = model.target_beliefs(batch)
    _require(
        len(zs) == C.DEPTH + 1 and len(targets) == C.DEPTH + 1,
        "model returned an incomplete belief horizon",
    )
    embodiment = str(batch.get("embodiment", ""))
    _require(embodiment == "libero_franka", "metric embodiment changed")
    q_action = model.q_action[embodiment]
    coefficients: list[Tensor] = []
    for horizon in range(C.DEPTH):
        value = q_action(actions[:, horizon], zs[horizon])
        if isinstance(value, tuple):
            value = value[0]
        coefficients.append(value.detach())
    c_seq = torch.stack(coefficients, dim=1)
    _require(not c_seq.requires_grad, "action-labelled coefficients require grad")
    bank_gate._validate_action_coefficients(c_seq)
    rollout = sequential_rollout(model.bank, zs[0], c_seq)
    identity = torch.stack([
        ln_cosine_distance(zs[0], targets[horizon + 1], "per_slot")
        for horizon in range(C.DEPTH)
    ], dim=1)
    bank_error = torch.stack([
        ln_cosine_distance(rollout[horizon], targets[horizon + 1], "per_slot")
        for horizon in range(C.DEPTH)
    ], dim=1)
    bank_gate._finite_tensor(identity, "identity errors")
    bank_gate._finite_tensor(bank_error, "bank rollout errors")
    return identity, bank_error, c_seq


@torch.inference_mode()
def collect_error_rows(
    model: Any,
    dataset: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    manifest_digest: str,
    batch_size: int = DEFAULT_PINS.batch_size,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Measure identity and action-labelled sequential rollout errors."""
    _require(batch_size > 0, "batch size must be positive")
    _require(len(records) == DEFAULT_PINS.windows, "metric record count changed")
    chunks = [
        [int(row["dataset_index"]) for row in records[lo:lo + batch_size]]
        for lo in range(0, len(records), batch_size)
    ]
    from loom.data.loader import collate_window

    batches: Iterable[Mapping[str, Any]] = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=chunks,
        collate_fn=collate_window,
        num_workers=0,
        pin_memory=False,
    )
    identity_parts: list[np.ndarray] = []
    rollout_parts: list[np.ndarray] = []
    witnesses: list[dict[str, Any]] = []
    offset = 0
    for window in batches:
        tasks, trajectories, clusters = bank_gate._validate_batch_meta(
            window, manifest_digest, "libero",
        )
        _require(trajectories == clusters, "batch cluster identity changed")
        batch = bank_gate._to_device(window, "cpu", torch.float32)
        b = bank_gate._batch_size(batch)
        expected = records[offset:offset + b]
        _require(len(expected) == b, "metric batch has unexpected extra rows")
        for row, task, trajectory in zip(expected, tasks, trajectories):
            _require(
                row["task_id"] == task and row["trajectory_id"] == trajectory,
                "collated metric order differs from authenticated selection",
            )
        actions = batch.get("actions")
        _require(isinstance(actions, Tensor), "moved preflight batch lost actions")
        identity, bank_error, c_seq = measure_error_batch(model, batch)
        identity_np = identity.float().cpu().numpy()
        rollout_np = bank_error.float().cpu().numpy()
        _require(identity_np.shape == (b, C.DEPTH), "identity error shape changed")
        _require(rollout_np.shape == (b, C.DEPTH), "rollout error shape changed")
        identity_parts.append(identity_np)
        rollout_parts.append(rollout_np)
        for local, record in enumerate(expected):
            witnesses.append({
                **copy.deepcopy(dict(record)),
                "action_segment_sha256": tensor_sha256(actions[local]),
                "q_action_coefficients_sha256": tensor_sha256(c_seq[local]),
                "identity_error_per_horizon": [
                    float(value) for value in identity_np[local]
                ],
                "rollout_error_per_horizon": [
                    float(value) for value in rollout_np[local]
                ],
                "identity_minus_rollout_per_horizon": [
                    float(value)
                    for value in identity_np[local] - rollout_np[local]
                ],
            })
        offset += b

    _require(offset == len(records), f"collected {offset}/{len(records)} metric rows")
    identity_all = np.concatenate(identity_parts, axis=0)
    rollout_all = np.concatenate(rollout_parts, axis=0)
    _require(len(witnesses) == len(records), "metric witnesses are incomplete")
    return identity_all, rollout_all, witnesses


def summarize_errors(
    identity_error: np.ndarray,
    rollout_error: np.ndarray,
    *,
    expected_rows: int = DEFAULT_PINS.windows,
) -> dict[str, Any]:
    """Apply the fixed strict per-horizon mean-improvement gate."""
    identity = np.asarray(identity_error, dtype=np.float64)
    rollout = np.asarray(rollout_error, dtype=np.float64)
    wanted = (int(expected_rows), C.DEPTH)
    _require(identity.shape == wanted, f"identity errors have shape {identity.shape}, want {wanted}")
    _require(rollout.shape == wanted, f"rollout errors have shape {rollout.shape}, want {wanted}")
    _require(np.isfinite(identity).all(), "identity errors contain non-finite values")
    _require(np.isfinite(rollout).all(), "rollout errors contain non-finite values")
    _require((identity >= 0).all(), "identity errors contain negative values")
    _require((rollout >= 0).all(), "rollout errors contain negative values")
    identity_mean = identity.mean(axis=0, dtype=np.float64)
    rollout_mean = rollout.mean(axis=0, dtype=np.float64)
    improvement = identity_mean - rollout_mean
    horizons: list[dict[str, Any]] = []
    failures: list[str] = []
    for horizon in range(C.DEPTH):
        passed = bool(rollout_mean[horizon] < identity_mean[horizon])
        row = {
            "horizon": horizon + 1,
            "n_rows": int(expected_rows),
            "identity_mean": float(identity_mean[horizon]),
            "rollout_mean": float(rollout_mean[horizon]),
            "identity_minus_rollout": float(improvement[horizon]),
            "comparison": "rollout_mean strictly_less_than identity_mean",
            "passed": passed,
        }
        horizons.append(row)
        if not passed:
            failures.append(
                f"h{horizon + 1}: rollout mean {rollout_mean[horizon]!r} "
                f">= identity mean {identity_mean[horizon]!r}"
            )
    return {
        "status": "PASS" if not failures else "FAIL",
        "passed": not failures,
        "n_rows": int(expected_rows),
        "n_errors": 0,
        "horizons": horizons,
        "failures": failures,
    }


def _source_runtime_provenance(source: Mapping[str, Any]) -> dict[str, Any]:
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            stderr=subprocess.DEVNULL, timeout=30,
        ).decode().strip()
        git_dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT,
            stderr=subprocess.DEVNULL, timeout=30,
        ).decode().strip())
    except Exception:  # noqa: BLE001 - provenance retains an explicit unknown
        git_sha, git_dirty = "unknown", None
    script = next(
        row for row in source["behavior_source_files"]
        if row["path"] == "scripts/omega0_preflight.py"
    )
    return {
        **copy.deepcopy(dict(source)),
        "script_sha256": script["sha256"],
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "python": sys.version,
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "platform": platform.platform(),
        "hostname": platform.node(),
    }


def execute_preflight(
    args: argparse.Namespace,
    *,
    pins: PreflightPins = DEFAULT_PINS,
    loader_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run the authenticated CPU preflight without modifying any artifact."""
    started = time.monotonic()
    source = behavior_source_provenance()
    behavior_digest = source["behavior_source_digest"]
    target_cfg, target_provenance = authenticate_target_config(args.config, pins=pins)

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    checkpoint_sha = sha256_file(checkpoint)
    _require(
        checkpoint_sha == pins.deploy_checkpoint_sha256,
        "requested checkpoint is not the verified deploy artifact",
    )
    try:
        payload = torch.load(
            str(checkpoint), map_location="cpu", weights_only=False, mmap=True,
        )
    except Exception as exc:  # noqa: BLE001
        raise PreflightError(f"cannot load deploy checkpoint: {exc}") from exc
    checkpoint_provenance = authenticate_deploy_payload(
        payload, checkpoint_sha256=checkpoint_sha,
        target_config=target_cfg, pins=pins,
    )
    checkpoint_provenance.update({
        "path": str(checkpoint),
        "sha256": checkpoint_sha,
        "bytes": checkpoint.stat().st_size,
    })

    from loom.train.determinism import enable_determinism, set_global_seed
    from loom.train.loop import build_model

    enable_determinism()
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_num_threads(8)
    set_global_seed(pins.seed, rank=0)
    build_cfg = copy.deepcopy(target_cfg)
    build_cfg.setdefault("model", {})["use_stubs"] = False
    try:
        model = build_model(build_cfg)
        incompatible = model.load_state_dict(payload["model"], strict=True)
    except Exception as exc:  # noqa: BLE001
        raise PreflightError(f"cannot build/strict-load real deploy model: {exc}") from exc
    _require(not incompatible.missing_keys, f"missing model keys: {incompatible.missing_keys}")
    _require(
        not incompatible.unexpected_keys,
        f"unexpected model keys: {incompatible.unexpected_keys}",
    )
    model.requires_grad_(False).eval().to("cpu")
    model.compute_dtype = None
    reset_audit = apply_omega_zero_and_audit(
        model, payload["model"], reset_recipe=pins.transition_reset,
    )
    assert_behavior_source_digest(behavior_digest)

    data_cfg = copy.deepcopy(target_cfg)
    data_cfg.setdefault("data", {})
    data_cfg["data"].update({
        "batch_per_gpu": pins.batch_size,
        "sampling": "uniform_task",
        "trajectory_split": "gate",
        "num_workers": 0,
        "pin_memory": False,
    })
    if loader_factory is None:
        from loom.data.loader import build_gate_loader as loader_factory
    try:
        loader = loader_factory(
            data_cfg, rank=0, world=1, seed=pins.seed, device="cpu",
            cache_root=args.cache_root,
        )
    except Exception as exc:  # noqa: BLE001
        raise PreflightError(f"cannot build authenticated gate loader: {exc}") from exc

    try:
        dataset, records, data_provenance = authenticate_selection(loader, pins=pins)
        identity, rollout, rows = collect_error_rows(
            model, dataset, records,
            manifest_digest=pins.manifest_digest,
            batch_size=pins.batch_size,
        )
    finally:
        bank_gate._close_loader_caches(loader)
    results = summarize_errors(identity, rollout, expected_rows=pins.windows)
    assert_behavior_source_digest(behavior_digest)
    elapsed = time.monotonic() - started
    return {
        "format_version": FORMAT_VERSION,
        "gate": "r0a_bank_ca_qa_omega0_pre_update",
        "status": results["status"],
        "passed": results["passed"],
        "checkpoint": checkpoint_provenance,
        "target_config": target_provenance,
        "reset_audit": reset_audit,
        "data": data_provenance,
        "rows": rows,
        "results": results,
        "execution": {
            "device": "cpu",
            "compute_dtype": "float32",
            "autocast": False,
            "seed": pins.seed,
            "threads": int(torch.get_num_threads()),
            "workers": 0,
            "batch_size": pins.batch_size,
            "n_batches": pins.windows // pins.batch_size,
            "wall_seconds": float(elapsed),
            "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "simulator_episodes": 0,
            "checkpoint_mutated": False,
        },
        "source_provenance": _source_runtime_provenance(source),
    }


def atomic_publish_json(
    path: str | os.PathLike,
    value: Mapping[str, Any],
) -> str:
    """Publish one complete JSON file without ever replacing an existing one."""
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise PreflightError(
                f"refusing to overwrite existing preflight report: {target}"
            ) from exc
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return hashlib.sha256(raw).hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="runs/r0a_deploy_s1_eval/ckpt_000049666.pt",
        help="verified consolidated deploy checkpoint (content is SHA-pinned)",
    )
    parser.add_argument(
        "--config",
        default="configs/r0a_bank_ca_qa_omega0.yaml",
        help="current omega0 target config (resolved experiment hash is pinned)",
    )
    parser.add_argument("--out", required=True, help="new exclusive JSON result path")
    parser.add_argument(
        "--cache-root",
        help="optional feature-cache path; its manifest content is SHA-pinned",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.monotonic()
    try:
        report = execute_preflight(args)
        code = 0 if report["passed"] else 1
    except Exception as exc:  # noqa: BLE001 - persist every fail-closed outcome
        report = {
            "format_version": FORMAT_VERSION,
            "gate": "r0a_bank_ca_qa_omega0_pre_update",
            "status": "ERROR",
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "requested": vars(args),
            "execution": {
                "device": "cpu",
                "compute_dtype": "float32",
                "wall_seconds": float(time.monotonic() - started),
                "simulator_episodes": 0,
                "checkpoint_mutated": False,
            },
        }
        code = 2
    try:
        report_sha = atomic_publish_json(args.out, report)
    except Exception as exc:  # Never replace or reinterpret an earlier verdict.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3
    print(json.dumps({
        "status": report["status"],
        "passed": report["passed"],
        "out": str(Path(args.out).expanduser().resolve()),
        "report_sha256": report_sha,
        "failures": report.get("results", {}).get("failures", []),
        "error": report.get("error"),
    }, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
