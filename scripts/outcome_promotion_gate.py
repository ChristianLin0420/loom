#!/usr/bin/env python3
"""Fail-closed seed-0 promotion gate for an outcome-GRPO checkpoint.

The gate compares one candidate's official 400-episode LIBERO seed-0 result
with the immutable step-49,666 baseline.  It never runs an environment and it
never changes a checkpoint.  Every WorkItem and both deterministic seeds are
authenticated before paired outcomes are counted.

PASS requires all of the following:

* complete, real, checkpoint-backed evaluations with zero errors;
* the exact authoritative 149/400 baseline outcome vector;
* candidate success >= 164/400 and suite counts >= 35/27/43/24;
* paired candidate-only successes strictly outnumber baseline-only successes;
* the result metadata's checkpoint paths and bytes authenticate the supplied
  consolidated checkpoint; and
* that checkpoint and its no-overwrite terminal report prove the exact frozen
  six-fold trainer recipe, 4,800 accepted updates, and terminal convergence
  plus trust PASS gates.  Round-0 and holdout-only artifacts are ineligible.

The report is written atomically and exclusively.  An existing output is
never replaced, including by a concurrent invocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from loom.eval import DEFAULT_LIBERO_SUITES, EvalProtocol
from loom.eval.policy import submodule_state
from loom.eval.runner import POLICY_SEED_SCHEME, WorkItem, iter_work


FORMAT_VERSION = 1
GATE_NAME = "outcome_grpo_seed0_promotion"
EXPECTED_WORK_ITEMS = 400
EXPECTED_PER_SUITE = 100
EXPECTED_BASELINE_TOTAL = 149
EXPECTED_BASELINE_BY_SUITE = {
    "libero_spatial": 40,
    "libero_object": 32,
    "libero_goal": 48,
    "libero_long": 29,
}
CANDIDATE_MIN_TOTAL = 164
CANDIDATE_MIN_BY_SUITE = {
    "libero_spatial": 35,
    "libero_object": 27,
    "libero_goal": 43,
    "libero_long": 24,
}

# The authoritative artifact is
# runs/eval_r0a_deploy_s1_s49666_seeded1200_v2/seed0/results.json.  Pin both
# its complete bytes and its canonical per-WorkItem outcomes.  The latter
# makes the scientific dependency explicit: swapping which episodes succeeded
# while retaining 149 and the four suite totals is not the same paired test.
AUTHORITATIVE_BASELINE_RESULTS_SHA256 = (
    "95e3ac186c28a6305f5fff3375b45c5184697a0a94f7c08c511ab0d09fd27f3a"
)
AUTHORITATIVE_BASELINE_OUTCOME_SHA256 = (
    "c61eb59aea413fbf594b5329aea8d07a1e8e73223a299c25c26f70bde7de28e9"
)
AUTHORITATIVE_BASELINE_STEP = 49_666

# The only training lineage eligible for this promotion gate.  These values
# are frozen independently of candidate-controlled resolved_config/provenance.
CANONICAL_CANDIDATE_STEP = 54_466
CANONICAL_ACCEPTED_UPDATES = 4_800
CANONICAL_CONFIG_HASH = "25afdedfc9deea5e"
CANONICAL_TRAINER_KIND = "loom_outcome_grpo_proposal_descendant"
CANONICAL_PARENT_CONFIG_HASH = "a199324a6205bb6d"
CANONICAL_PARENT_CHECKPOINT_SHA256 = (
    "15f286c268caa5327d5aa3abf1f67ebd0555c426a509fef22cb7f537bf6ab4e1"
)
CANONICAL_TRAINER_SOURCE_SCHEME = "sha256(path-nul-sha256-nul)-v1"
CANONICAL_TRAINER_SOURCE_SHA256 = (
    "d5ef53e9f2e276f17d68f80b4c081c8f09b0d89ea9a966214fc3b63387364a52"
)
BEHAVIOUR_IDENTITY_MAX_ABS_LOGRATIO = 0.0
BEHAVIOUR_IDENTITY_MAX_COEFF_ERROR = 0.0
INITIAL_RATIO_MIN_ESS_FRACTION = 1.0
CANONICAL_STRICT_DETERMINISM = {
    "deterministic_algorithms": True,
    "warn_only": False,
}
CANONICAL_COLLECTOR_SOURCE_SHA256 = (
    "1622aee21022347a98f82c4e587154f1e13128528441e4d87d96d0d609be1223"
)
CANONICAL_COLLECTOR_SOURCE_FILES = (
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
)
CANONICAL_TRAINER_SOURCE_FILES = (
    "contracts.py",
    "loom/data/adapters/libero.py",
    "loom/data/cache.py",
    "loom/data/canonical.py",
    "loom/data/loader.py",
    "loom/data/tower.py",
    "loom/eval/outcome_recovery.py",
    "loom/eval/policy.py",
    "loom/heads/decoder.py",
    "loom/heads/proposal.py",
    "loom/heads/q_action.py",
    "loom/heads/q_delta.py",
    "loom/losses/dyn.py",
    "loom/losses/proposal_bc.py",
    "loom/model/estimator.py",
    "loom/train/atomic.py",
    "loom/train/ckpt.py",
    "loom/train/determinism.py",
    "loom/train/fsdp.py",
    "loom/train/loop.py",
    "loom/train/outcome_grpo.py",
    "loom/train/preempt.py",
    "loom/train/schedule.py",
    "loom/train/wandb_util.py",
    "scripts/env.sh",
    "scripts/outcome_grpo.sbatch",
    "scripts/train_outcome_grpo.py",
    "stubs.py",
)
CANONICAL_COLLECTION_SPLITS = tuple(f"train{index}" for index in range(6))
CANONICAL_CONVERGENCE_SNAPSHOT_STEPS = (
    AUTHORITATIVE_BASELINE_STEP, 53_666, 53_866, 54_066, 54_266,
    CANONICAL_CANDIDATE_STEP,
)
CANONICAL_CONVERGENCE_CHECKS = {
    "heldout_efficacy",
    "heldout_plateau_all_snapshots",
    "final_approx_kl",
    "accepted_updates",
    "anchor_sparse_ce_block_median_relative_range",
    "anchor_sparse_ce_terminal_block_median",
}
CANONICAL_TRUST_CHECKS = {
    "clip_fraction",
    "ess_fraction",
    "coeff_drift_p95",
    "live_ops",
    "nonfinite",
    "unexpected_gradients",
    "expert_topk_overlap_change",
}
CANONICAL_PROPOSAL_SCORING = {
    "batch_size": 1,
    "dtype": "float32",
    "autocast": False,
    "cuda_matmul_tf32": False,
    "cudnn_tf32": False,
    "float32_matmul_precision": "highest",
    "device_type": "cuda",
    "module_mode": "eval",
    "stored_order": True,
}
CANONICAL_TERMINAL_PARALLELISM = "one_snapshot_or_trust_per_rank"

SOURCE_DIGEST_SCHEME = "sha256(relpath-nul-file-sha256-nul)-v1"
SOURCE_FILES = (
    "loom/eval/__init__.py",
    "loom/eval/libero.py",
    "loom/eval/policy.py",
    "loom/eval/runner.py",
    "scripts/outcome_promotion_gate.py",
)
DIRECT_POLICY_MODULES = ("estimator", "proposal", "decoder")


class PromotionGateError(RuntimeError):
    """Input integrity or completeness failure (distinct from metric FAIL)."""


@dataclass(frozen=True)
class ValidatedResults:
    path: str
    sha256: str
    outcomes: Mapping[tuple[str, str, int, int, int], bool]
    suite_success: Mapping[str, int]
    n_success: int
    work_items_sha256: str
    outcomes_sha256: str
    metadata: Mapping[str, Any]


def _require(condition: Any, message: str) -> None:
    if not bool(condition):
        raise PromotionGateError(message)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _is_config_hash(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 16
        and all(ch in "0123456789abcdef" for ch in value)
    )


def sha256_file(path: str | os.PathLike[str], chunk_bytes: int = 8 << 20) -> str:
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
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _experiment_config_hash(cfg: Mapping[str, Any]) -> str:
    experiment = {key: value for key, value in cfg.items() if key != "link"}
    return hashlib.blake2b(
        json.dumps(experiment, sort_keys=True, default=str).encode("utf-8"),
        digest_size=8,
    ).hexdigest()


def _read_json(path: str | os.PathLike[str], label: str) -> tuple[Path, str, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    _require(source.is_file(), f"{label} does not exist: {source}")
    try:
        raw = source.read_bytes()
    except Exception as exc:  # noqa: BLE001
        raise PromotionGateError(f"cannot read {label} {source}: {exc}") from exc
    digest = hashlib.sha256(raw).hexdigest()
    try:
        value = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        raise PromotionGateError(f"cannot parse {label} {source}: {exc}") from exc
    _require(isinstance(value, dict), f"{label} is not a JSON object")
    return source, digest, value


def official_protocol() -> EvalProtocol:
    return EvalProtocol(
        bench="libero",
        episodes_per_task=10,
        n_tasks=10,
        suites=DEFAULT_LIBERO_SUITES,
        seeds=(0,),
        max_steps=512,
    )


def official_items() -> list[WorkItem]:
    items = iter_work(official_protocol())
    _require(len(items) == EXPECTED_WORK_ITEMS, "official protocol did not yield 400 WorkItems")
    return items


def _item_key(row: Mapping[str, Any]) -> tuple[str, str, int, int, int]:
    fields = ("bench", "suite", "task_id", "episode", "seed")
    _require(all(field in row for field in fields), "episode row is missing its WorkItem key")
    _require(isinstance(row["bench"], str), "episode bench is not a string")
    _require(isinstance(row["suite"], str), "episode suite is not a string")
    for field in ("task_id", "episode", "seed"):
        _require(_is_int(row[field]), f"episode {field} is not an integer")
    return (
        row["bench"], row["suite"], int(row["task_id"]),
        int(row["episode"]), int(row["seed"]),
    )


def _signature_rows(
    rows: Mapping[tuple[str, str, int, int, int], Mapping[str, Any]],
    *,
    outcomes: Mapping[tuple[str, str, int, int, int], bool] | None = None,
) -> list[dict[str, Any]]:
    signature: list[dict[str, Any]] = []
    for key in sorted(rows):
        row = rows[key]
        entry = {
            "bench": key[0],
            "suite": key[1],
            "task_id": key[2],
            "episode": key[3],
            "seed": key[4],
            "env_seed": int(row["env_seed"]),
            "policy_seed": int(row["policy_seed"]),
        }
        if outcomes is not None:
            entry["success"] = bool(outcomes[key])
        signature.append(entry)
    return signature


def _resolved_checkpoint_path(value: Any, label: str) -> Path:
    _require(isinstance(value, str) and value.strip(), f"{label} is missing")
    return Path(value).expanduser().resolve()


def _validate_real_eval_metadata(
    blob: Mapping[str, Any],
    *,
    label: str,
    expected_step: int | None = None,
    expected_config_hash: str | None = None,
) -> dict[str, Any]:
    meta = blob.get("meta")
    _require(isinstance(meta, Mapping), f"{label} metadata is missing")
    _require(meta.get("bench") == "libero", f"{label} metadata bench is not LIBERO")
    _require(meta.get("backend") == "libero", f"{label} did not request real LIBERO")
    _require(meta.get("env_available") is True, f"{label} real environment was unavailable")
    _require(meta.get("libero_available") is True, f"{label} LIBERO was unavailable")
    _require(meta.get("policy_seed_scheme") == POLICY_SEED_SCHEME,
             f"{label} policy seed scheme drifted")

    identity = meta.get("eval_identity")
    _require(isinstance(identity, Mapping), f"{label} eval_identity is missing")
    _require(identity.get("version") == 1, f"{label} eval_identity version is invalid")
    _require(identity.get("policy_source") == "checkpoint_factory",
             f"{label} was not built from a checkpoint")
    _require(identity.get("policy_seed_scheme") == POLICY_SEED_SCHEME,
             f"{label} eval identity seed scheme drifted")
    backend = identity.get("backend")
    _require(isinstance(backend, Mapping), f"{label} backend identity is missing")
    _require(backend.get("requested") == "libero" and backend.get("resolved") == "libero",
             f"{label} did not resolve the real LIBERO backend")
    policy_kw = identity.get("policy_kw")
    _require(isinstance(policy_kw, Mapping), f"{label} policy options are missing")
    _require(policy_kw.get("allow_stub") is False,
             f"{label} did not fail closed on stub loading")
    _require(policy_kw.get("op_stats") is True,
             f"{label} lacks the official op-stats evaluation identity")
    allowed_policy_kw = {"allow_stub", "op_stats", "embodiment"}
    _require(set(policy_kw).issubset(allowed_policy_kw),
             f"{label} has non-official policy options: {sorted(set(policy_kw) - allowed_policy_kw)}")
    if "embodiment" in policy_kw:
        _require(policy_kw["embodiment"] == "libero_franka",
                 f"{label} identity has the wrong embodiment")

    policy = meta.get("policy")
    _require(isinstance(policy, Mapping), f"{label} policy provenance is missing")
    _require(policy.get("policy") == "LoomPolicy", f"{label} did not run LoomPolicy")
    _require(policy.get("is_stub") is False, f"{label} used a stub policy")
    _require(policy.get("embodiment") == "libero_franka",
             f"{label} used the wrong embodiment")
    _require(policy.get("gripper_dwell") == 1,
             f"{label} changed gripper dwell")
    _require(policy.get("decoder_samples") == 1,
             f"{label} changed decoder sample count")
    _require(policy.get("duration_normalize_segments") is False,
             f"{label} enabled duration-normalized segments")
    _require(policy.get("h_op") == 8 and policy.get("fps_canonical") == 30,
             f"{label} operator horizon/canonical FPS drifted")

    step = policy.get("ckpt_global_step")
    _require(_is_int(step) and step >= 0, f"{label} checkpoint step is missing")
    if expected_step is not None:
        _require(step == expected_step,
                 f"{label} checkpoint step {step} != {expected_step}")
    config_hash = policy.get("ckpt_config_hash")
    if expected_config_hash is not None:
        _require(config_hash == expected_config_hash,
                 f"{label} metadata config hash does not match checkpoint")

    state = policy.get("state_dict")
    _require(isinstance(state, Mapping), f"{label} has no module-load provenance")
    for module in DIRECT_POLICY_MODULES:
        row = state.get(module)
        _require(isinstance(row, Mapping), f"{label} did not load {module}")
        _require(_is_int(row.get("tensors_loaded")) and row["tensors_loaded"] > 0,
                 f"{label} loaded no {module} tensors")
        _require(row.get("unexpected") == 0,
                 f"{label} had unexpected {module} checkpoint tensors")

    paths = {
        "meta.ckpt": _resolved_checkpoint_path(meta.get("ckpt"), f"{label} meta.ckpt"),
        "meta.policy.ckpt": _resolved_checkpoint_path(
            policy.get("ckpt"), f"{label} meta.policy.ckpt",
        ),
        "meta.eval_identity.checkpoint": _resolved_checkpoint_path(
            identity.get("checkpoint"), f"{label} eval checkpoint",
        ),
    }
    canonical = {str(path) for path in paths.values()}
    _require(len(canonical) == 1, f"{label} checkpoint paths disagree: {paths}")
    explicit_checkpoint_sha256 = {
        name: value
        for name, value in {
            "meta.checkpoint_sha256": meta.get("checkpoint_sha256"),
            "meta.ckpt_sha256": meta.get("ckpt_sha256"),
            "meta.eval_identity.checkpoint_sha256": identity.get(
                "checkpoint_sha256"
            ),
            "meta.policy.checkpoint_sha256": policy.get("checkpoint_sha256"),
            "meta.policy.ckpt_sha256": policy.get("ckpt_sha256"),
        }.items()
        if value is not None
    }
    for name, value in explicit_checkpoint_sha256.items():
        _require(_is_sha256(value), f"{label} {name} is not a SHA-256")
    return {
        "checkpoint_path": next(iter(canonical)),
        "checkpoint_paths": {name: str(path) for name, path in paths.items()},
        "global_step": int(step),
        "config_hash": config_hash,
        "explicit_checkpoint_sha256": explicit_checkpoint_sha256,
        "eval_identity": json.loads(json.dumps(identity, sort_keys=True)),
        "git_sha": meta.get("git_sha"),
        "slurm_job_id": meta.get("slurm_job_id"),
        "hostname": meta.get("hostname"),
        "started": meta.get("started"),
        "policy": {
            key: policy.get(key)
            for key in (
                "policy", "is_stub", "embodiment", "gripper_dwell",
                "decoder_samples", "duration_normalize_segments", "h_op",
                "fps_canonical", "resampler", "ckpt_global_step",
                "ckpt_config_hash", "state_dict",
            )
        },
    }


def _validate_summary(
    blob: Mapping[str, Any],
    *,
    label: str,
    suite_success: Mapping[str, int],
    n_success: int,
) -> None:
    summary = blob.get("summary")
    _require(isinstance(summary, Mapping), f"{label} summary is missing")
    _require(summary.get("n_episodes") == EXPECTED_WORK_ITEMS,
             f"{label} summary does not contain 400 episodes")
    _require(summary.get("n_expected") == EXPECTED_WORK_ITEMS,
             f"{label} summary expected count is not 400")
    _require(summary.get("n_errors") == 0, f"{label} summary reports errors")
    _require(summary.get("complete") is True, f"{label} summary is incomplete")
    expected_avg = 100.0 * n_success / EXPECTED_WORK_ITEMS
    _require(isinstance(summary.get("avg"), (int, float))
             and math.isfinite(float(summary["avg"]))
             and math.isclose(float(summary["avg"]), expected_avg, abs_tol=1e-12),
             f"{label} summary average disagrees with episode rows")
    per_suite = summary.get("per_suite")
    _require(isinstance(per_suite, Mapping), f"{label} per-suite summary is missing")
    _require(set(per_suite) == set(DEFAULT_LIBERO_SUITES),
             f"{label} summary suite keys are not official")
    for suite in DEFAULT_LIBERO_SUITES:
        row = per_suite[suite]
        _require(isinstance(row, Mapping), f"{label} summary for {suite} is invalid")
        _require(row.get("n_episodes") == EXPECTED_PER_SUITE,
                 f"{label} summary for {suite} does not contain 100 episodes")
        _require(row.get("n_errors") == 0,
                 f"{label} summary for {suite} reports errors")
        expected_rate = float(suite_success[suite])
        _require(isinstance(row.get("success_rate"), (int, float))
                 and math.isfinite(float(row["success_rate"]))
                 and math.isclose(float(row["success_rate"]), expected_rate, abs_tol=1e-12),
                 f"{label} summary success rate for {suite} disagrees with rows")


def validate_results(
    path: str | os.PathLike[str],
    *,
    label: str,
    expected_results_sha256: str | None = None,
    expected_outcomes_sha256: str | None = None,
    expected_step: int | None = None,
    expected_config_hash: str | None = None,
) -> ValidatedResults:
    source, digest, blob = _read_json(path, f"{label} results")
    if expected_results_sha256 is not None:
        _require(_is_sha256(expected_results_sha256),
                 "internal authoritative baseline result SHA-256 is invalid")
        _require(digest == expected_results_sha256,
                 f"{label} results SHA-256 is not authoritative")
    _require(blob.get("version") == 1, f"{label} result version is invalid")
    _require(blob.get("bench") == "libero", f"{label} result bench is not LIBERO")
    try:
        protocol = EvalProtocol.from_dict(blob.get("protocol", {}))
    except Exception as exc:  # noqa: BLE001
        raise PromotionGateError(f"{label} protocol is invalid: {exc}") from exc
    _require(protocol == official_protocol(), f"{label} is not the official seed-0 protocol")

    expected = {item.key(): item for item in official_items()}
    episodes = blob.get("episodes")
    _require(isinstance(episodes, list), f"{label} episodes are missing")
    _require(len(episodes) == EXPECTED_WORK_ITEMS,
             f"{label} must have exactly 400 episode rows")
    rows: dict[tuple[str, str, int, int, int], dict[str, Any]] = {}
    outcomes: dict[tuple[str, str, int, int, int], bool] = {}
    suite_success = {suite: 0 for suite in DEFAULT_LIBERO_SUITES}
    for index, raw in enumerate(episodes):
        _require(isinstance(raw, Mapping), f"{label} episode {index} is not an object")
        row = dict(raw)
        key = _item_key(row)
        _require(key in expected, f"{label} contains unknown WorkItem {key}")
        _require(key not in rows, f"{label} duplicates WorkItem {key}")
        item = expected[key]
        _require(row.get("env_seed") == item.env_seed,
                 f"{label} WorkItem {key} has the wrong env_seed")
        extra = row.get("extra")
        _require(isinstance(extra, Mapping), f"{label} WorkItem {key} has no extra metadata")
        _require(extra.get("policy_seed") == item.policy_seed,
                 f"{label} WorkItem {key} has the wrong policy_seed")
        _require(row.get("error") is None, f"{label} WorkItem {key} has an error")
        _require(type(row.get("success")) is bool,
                 f"{label} WorkItem {key} success is not boolean")
        steps = row.get("steps")
        _require(_is_int(steps) and 1 <= steps <= item.max_steps,
                 f"{label} WorkItem {key} has invalid step count")
        _require(type(row.get("hit_step_cap")) is bool,
                 f"{label} WorkItem {key} hit_step_cap is not boolean")
        expected_hit_cap = not row["success"] and steps >= item.max_steps
        _require(row["hit_step_cap"] is expected_hit_cap,
                 f"{label} WorkItem {key} has inconsistent hit_step_cap")
        replans = row.get("n_replans")
        _require(_is_int(replans) and replans > 0,
                 f"{label} WorkItem {key} has no policy replans")
        wall_s = row.get("wall_s")
        _require(isinstance(wall_s, (int, float)) and not isinstance(wall_s, bool)
                 and math.isfinite(float(wall_s)) and float(wall_s) >= 0.0,
                 f"{label} WorkItem {key} has invalid wall time")
        outcomes[key] = row["success"]
        suite_success[item.suite] += int(row["success"])
        rows[key] = {
            "env_seed": item.env_seed,
            "policy_seed": item.policy_seed,
        }

    _require(set(rows) == set(expected), f"{label} WorkItems are incomplete")
    n_success = sum(outcomes.values())
    _validate_summary(
        blob, label=label, suite_success=suite_success, n_success=n_success,
    )
    metadata = _validate_real_eval_metadata(
        blob, label=label, expected_step=expected_step,
        expected_config_hash=expected_config_hash,
    )
    work_digest = _canonical_json_sha256(_signature_rows(rows))
    outcome_digest = _canonical_json_sha256(
        _signature_rows(rows, outcomes=outcomes)
    )
    if expected_outcomes_sha256 is not None:
        _require(_is_sha256(expected_outcomes_sha256),
                 "internal authoritative baseline outcome SHA-256 is invalid")
        _require(outcome_digest == expected_outcomes_sha256,
                 f"{label} per-WorkItem outcomes are not authoritative")
    return ValidatedResults(
        path=str(source), sha256=digest, outcomes=outcomes,
        suite_success=suite_success, n_success=n_success,
        work_items_sha256=work_digest, outcomes_sha256=outcome_digest,
        metadata=metadata,
    )


def validate_baseline(path: str | os.PathLike[str]) -> ValidatedResults:
    result = validate_results(
        path,
        label="baseline",
        expected_results_sha256=AUTHORITATIVE_BASELINE_RESULTS_SHA256,
        expected_outcomes_sha256=AUTHORITATIVE_BASELINE_OUTCOME_SHA256,
        expected_step=AUTHORITATIVE_BASELINE_STEP,
    )
    _require(result.n_success == EXPECTED_BASELINE_TOTAL,
             f"baseline total {result.n_success} != {EXPECTED_BASELINE_TOTAL}")
    _require(dict(result.suite_success) == EXPECTED_BASELINE_BY_SUITE,
             f"baseline suite counts {dict(result.suite_success)} are not authoritative")
    return result


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        try:
            payload = torch.load(
                str(path), map_location="cpu", weights_only=False, mmap=True,
            )
        except TypeError:
            payload = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        raise PromotionGateError(f"cannot inspect candidate checkpoint {path}: {exc}") from exc
    _require(isinstance(payload, Mapping), "candidate checkpoint is not a mapping")
    return payload


def _trainer_source_digest(files: Mapping[str, Any]) -> str:
    """Recompute the canonical trainer's stored path/content closure."""
    _require(all(isinstance(rel, str) for rel in files),
             "candidate trainer source paths must be strings")
    digest = hashlib.sha256()
    for rel in sorted(files):
        value = files[rel]
        path = Path(rel)
        _require(
            isinstance(rel, str)
            and rel == path.as_posix()
            and not path.is_absolute()
            and ".." not in path.parts,
            f"candidate trainer source path is invalid: {rel!r}",
        )
        _require(_is_sha256(value), f"candidate trainer source hash is invalid: {rel}")
        digest.update(rel.encode("utf-8") + b"\0")
        digest.update(bytes.fromhex(value) + b"\0")
    return digest.hexdigest()


def _validate_trainer_source(value: Any) -> dict[str, Any]:
    _require(isinstance(value, Mapping), "candidate trainer source closure is missing")
    _require(
        value.get("scheme") == CANONICAL_TRAINER_SOURCE_SCHEME,
        "candidate trainer source scheme is not canonical",
    )
    files = value.get("files")
    _require(isinstance(files, Mapping), "candidate trainer source files are missing")
    _require(
        set(files) == set(CANONICAL_TRAINER_SOURCE_FILES),
        "candidate trainer source file closure is not the frozen 28-file closure",
    )
    computed = _trainer_source_digest(files)
    _require(
        value.get("sha256") == computed,
        "candidate trainer source digest does not authenticate its file hashes",
    )
    _require(
        computed == CANONICAL_TRAINER_SOURCE_SHA256,
        "candidate trainer source digest is not the frozen canonical digest",
    )
    return {
        "scheme": CANONICAL_TRAINER_SOURCE_SCHEME,
        "sha256": computed,
        "n_files": len(files),
    }


def _validate_collection_provenance(
    value: Any,
    *,
    split: str,
    n_groups: int,
) -> None:
    _require(isinstance(value, Mapping), f"candidate {split} provenance is missing")
    _require(value.get("split") == split, f"candidate collection split is not {split}")
    _require(value.get("n_groups") == n_groups,
             f"candidate {split} group count is not {n_groups}")
    _require(value.get("n_trajectories") == n_groups * 8,
             f"candidate {split} trajectory count is not {n_groups * 8}")
    _require(_is_sha256(value.get("manifest_sha256")),
             f"candidate {split} manifest hash is invalid")
    _require(_is_sha256(value.get("identity_digest")),
             f"candidate {split} identity digest is invalid")
    for field in ("terminal_successes_by_arm", "replans_by_arm"):
        row = value.get(field)
        _require(isinstance(row, list) and len(row) == 8
                 and all(_is_int(item) and item >= 0 for item in row),
                 f"candidate {split} {field} is invalid")
    collector = value.get("collector_source")
    _require(isinstance(collector, Mapping),
             f"candidate {split} collector source is missing")
    _require(
        collector.get("scheme") == "sha256(path-nul-bytes-nul)-v1"
        and collector.get("sha256") == CANONICAL_COLLECTOR_SOURCE_SHA256
        and collector.get("files") == list(CANONICAL_COLLECTOR_SOURCE_FILES),
        f"candidate {split} was not produced by the frozen collector source",
    )


def _is_exact_number(value: Any, expected: float) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) == float(expected)
    )


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _validate_model_digest(value: Any, *, label: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"candidate {label} digest is missing")
    result = dict(value)
    _require(
        set(result) == {"sha256", "n_tensors", "n_bytes"}
        and _is_sha256(result.get("sha256"))
        and _is_int(result.get("n_tensors"))
        and result["n_tensors"] > 0
        and _is_int(result.get("n_bytes"))
        and result["n_bytes"] > 0,
        f"candidate {label} digest is invalid",
    )
    return result


def _model_state_digest(
    state: Mapping[str, Any], *, proposal: bool,
) -> dict[str, Any]:
    """Reproduce the trainer's exact flat-state tensor digest."""
    digest = hashlib.sha256()
    n_tensors = 0
    n_bytes = 0
    for name in sorted(state):
        selected = str(name).startswith("proposal.")
        if selected is not proposal:
            continue
        value = state[name]
        _require(isinstance(value, torch.Tensor),
                 f"candidate model state {name!r} is not a tensor")
        tensor = value.detach().cpu().contiguous()
        header = json.dumps(
            {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            },
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        raw = tensor.reshape(-1).view(torch.uint8).numpy()
        digest.update(header + b"\0")
        digest.update(memoryview(raw))
        digest.update(b"\0")
        n_tensors += 1
        n_bytes += int(tensor.numel() * tensor.element_size())
    _require(n_tensors > 0, "candidate model digest selected zero tensors")
    return {
        "sha256": digest.hexdigest(),
        "n_tensors": n_tensors,
        "n_bytes": n_bytes,
    }


def _validate_proposal_scoring(value: Any, *, label: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping),
             f"candidate {label} proposal scoring geometry is missing")
    result = dict(value)
    _require(
        result == CANONICAL_PROPOSAL_SCORING,
        f"candidate {label} did not use exact B=1 fp32 stored-order scoring",
    )
    return result


def _validate_behaviour_authentication(
    training: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reports = training.get("behaviour_authentication")
    _require(isinstance(reports, list) and len(reports) == 7,
             "candidate lacks seven-fold behaviour authentication")
    validated: list[dict[str, Any]] = []
    for index, value in enumerate(reports):
        label = f"behaviour authentication split {index}"
        _require(isinstance(value, Mapping), f"candidate {label} is invalid")
        row = dict(value)
        _require(row.get("passed") is True,
                 f"candidate {label} did not pass")
        _require(_is_int(row.get("all_atoms")) and row["all_atoms"] > 0,
                 f"candidate {label} has no authenticated atoms")
        _require(
            _is_int(row.get("ratio_eligible_atoms"))
            and 0 < row["ratio_eligible_atoms"] <= row["all_atoms"]
            and row.get("arm0_ratio_eligible_atoms") == 0,
            f"candidate {label} ratio atom counts are invalid",
        )
        _require(
            _is_finite_number(row.get("max_abs_coeff_error"))
            and 0.0 <= float(row["max_abs_coeff_error"])
                <= BEHAVIOUR_IDENTITY_MAX_COEFF_ERROR,
            f"candidate {label} coefficient replay exceeds identity bound",
        )
        for field in ("max_abs_old_logprob_error", "max_abs_logratio"):
            _require(
                _is_finite_number(row.get(field))
                and 0.0 <= float(row[field])
                    <= BEHAVIOUR_IDENTITY_MAX_ABS_LOGRATIO,
                f"candidate {label} {field} exceeds identity bound",
            )
        _require(row.get("proposal_replay_batch_size") == 1,
                 f"candidate {label} did not replay at B=1")
        _require(
            row.get("transfer_chunk_replans") == 32
            and _is_exact_number(
                row.get("logprob_atol"),
                BEHAVIOUR_IDENTITY_MAX_ABS_LOGRATIO,
            )
            and _is_exact_number(row.get("logprob_rtol"), 0.0)
            and _is_exact_number(row.get("coeff_atol"), 0.0)
            and _is_exact_number(row.get("coeff_rtol"), 0.0),
            f"candidate {label} replay tolerances/chunk geometry differ",
        )
        _validate_proposal_scoring(row.get("proposal_scoring"), label=label)
        validated.append(row)

    exact = training.get("exact_behaviour_identity")
    _require(isinstance(exact, Mapping),
             "candidate exact behaviour identity is missing")
    aggregate = dict(exact)
    _require(
        set(aggregate) == {
            "passed", "splits", "all_atoms", "max_abs_coeff_error",
            "max_abs_old_logprob_error", "max_abs_logratio",
            "max_abs_coeff_error_threshold", "max_abs_logratio_threshold",
        }
        and aggregate.get("passed") is True
        and aggregate.get("splits") == 7
        and aggregate.get("all_atoms") == sum(row["all_atoms"] for row in validated),
        "candidate exact behaviour identity aggregate is invalid",
    )
    _require(
        _is_exact_number(
            aggregate.get("max_abs_coeff_error_threshold"),
            BEHAVIOUR_IDENTITY_MAX_COEFF_ERROR,
        )
        and _is_exact_number(
            aggregate.get("max_abs_logratio_threshold"),
            BEHAVIOUR_IDENTITY_MAX_ABS_LOGRATIO,
        )
        and _is_exact_number(
            aggregate.get("max_abs_coeff_error"),
            max(float(row["max_abs_coeff_error"]) for row in validated),
        )
        and _is_exact_number(
            aggregate.get("max_abs_old_logprob_error"),
            max(float(row["max_abs_old_logprob_error"]) for row in validated),
        )
        and _is_exact_number(
            aggregate.get("max_abs_logratio"),
            max(float(row["max_abs_logratio"]) for row in validated),
        ),
        "candidate behaviour identity aggregate values/thresholds differ",
    )
    return validated, aggregate


def _validate_start_checkpoint_identity(
    training: Mapping[str, Any],
) -> dict[str, Any]:
    value = training.get("start_checkpoint_identity")
    _require(isinstance(value, Mapping),
             "candidate START checkpoint identity is missing")
    report = dict(value)
    _require(
        report.get("checked") is True
        and report.get("passed") is True
        and report.get("global_step") == AUTHORITATIVE_BASELINE_STEP
        and report.get("optimizer_state_entries") == 0
        and report.get("optimizer_reset")
            == {"count": 1, "modules": ["proposal"]},
        "candidate START checkpoint was not checked with one empty optimizer reset",
    )
    _validate_model_digest(report.get("proposal"), label="START parent proposal")
    return report


def _validate_initial_behavior_ratio_identity(
    training: Mapping[str, Any],
    *,
    trainer_source: Mapping[str, Any],
    parent: Mapping[str, Any],
    exact_behaviour: Mapping[str, Any],
    start_checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    value = training.get("initial_behavior_ratio_identity")
    _require(isinstance(value, Mapping),
             "candidate initial behavior ratio identity is missing")
    report = dict(value)
    _require(
        report.get("format_version") == 1
        and report.get("kind") == "outcome_grpo_initial_behavior_identity"
        and report.get("passed") is True
        and report.get("world_size") == 8
        and report.get("config_hash") == CANONICAL_CONFIG_HASH,
        "candidate initial behavior ratio identity header is invalid",
    )
    _require(report.get("trainer_source") == trainer_source,
             "candidate initial ratio identity is not bound to trainer source")
    _require(
        report.get("strict_determinism") == CANONICAL_STRICT_DETERMINISM,
        "candidate initial ratio identity lacks strict determinism evidence",
    )
    _require(report.get("parent") == parent,
             "candidate initial ratio identity is not bound to the parent")
    _require(report.get("exact_behaviour_identity") == exact_behaviour,
             "candidate initial ratio identity is not bound to exact replay")
    _require(report.get("start_checkpoint_identity") == start_checkpoint,
             "candidate initial ratio identity is not bound to START checkpoint")
    geometry = _validate_proposal_scoring(
        report.get("proposal_scoring"), label="initial ratio aggregate",
    )
    ranks = report.get("ranks")
    _require(isinstance(ranks, list) and len(ranks) == 8,
             "candidate initial ratio identity lacks eight rank witnesses")
    for rank, value in enumerate(ranks):
        label = f"initial ratio rank {rank}"
        _require(isinstance(value, Mapping), f"candidate {label} is invalid")
        row = dict(value)
        atoms = row.get("ratio_atoms")
        ratio_low = 1.0 - BEHAVIOUR_IDENTITY_MAX_ABS_LOGRATIO
        ratio_high = 1.0 + BEHAVIOUR_IDENTITY_MAX_ABS_LOGRATIO
        ratio_sum = float(row.get("ratio_sum", math.nan))
        ratio_square_sum = float(row.get("ratio_square_sum", math.nan))
        ratio_mean = float(row.get("ratio_mean", math.nan))
        ratio_ess = float(row.get("ratio_ess_fraction", math.nan))
        derived_mean = ratio_sum / max(int(atoms or 0), 1)
        derived_ess = ratio_sum * ratio_sum / max(
            int(atoms or 0) * ratio_square_sum, sys.float_info.min,
        )
        _require(
            row.get("rank") == rank
            and row.get("passed") is True
            and _is_int(atoms) and atoms == 14
            and _is_finite_number(row.get("max_abs_logratio"))
            and 0.0 <= float(row["max_abs_logratio"])
                <= BEHAVIOUR_IDENTITY_MAX_ABS_LOGRATIO
            and _is_finite_number(row.get("ratio_min"))
            and float(row["ratio_min"]) >= ratio_low
            and _is_finite_number(row.get("ratio_mean"))
            and ratio_low <= float(row["ratio_mean"]) <= ratio_high
            and _is_finite_number(row.get("ratio_max"))
            and float(row["ratio_max"]) <= ratio_high
            and float(row["ratio_min"])
                <= ratio_mean
                <= float(row["ratio_max"])
            and _is_exact_number(row.get("clip_fraction"), 0.0)
            and _is_finite_number(row.get("ratio_sum"))
            and _is_finite_number(row.get("ratio_square_sum"))
            and _is_finite_number(row.get("ratio_ess_fraction"))
            and float(row["ratio_ess_fraction"])
                >= INITIAL_RATIO_MIN_ESS_FRACTION
            and ratio_sum > 0.0
            and ratio_square_sum > 0.0
            and ratio_mean == derived_mean
            and ratio_ess == derived_ess
            and row.get("clipped_atoms") == 0,
            f"candidate {label} exceeds numerical-ratio identity bounds",
        )
        _require(
            _is_exact_number(
                row.get("max_abs_logratio_threshold"),
                BEHAVIOUR_IDENTITY_MAX_ABS_LOGRATIO,
            )
            and _is_exact_number(
                row.get("min_ess_fraction_threshold"),
                INITIAL_RATIO_MIN_ESS_FRACTION,
            ),
            f"candidate {label} ratio thresholds differ",
        )
        _require(_validate_proposal_scoring(
            row.get("proposal_scoring"), label=label,
        ) == geometry, f"candidate {label} scoring geometry differs")
    return report


def _validate_pass_checks(
    value: Any,
    *,
    label: str,
    expected: set[str],
) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"candidate {label} is missing")
    _require(value.get("passed") is True, f"candidate {label} did not pass")
    checks = value.get("checks")
    _require(isinstance(checks, Mapping), f"candidate {label} checks are missing")
    _require(set(checks) == expected,
             f"candidate {label} check set is not canonical")
    for name, row in checks.items():
        _require(isinstance(row, Mapping) and row.get("pass") is True,
                 f"candidate {label} check {name} did not pass")
    return checks


def _validate_terminal_gates(training: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    convergence = training.get("convergence_gate")
    checks = _validate_pass_checks(
        convergence, label="convergence gate",
        expected=CANONICAL_CONVERGENCE_CHECKS,
    )
    assert isinstance(convergence, Mapping)  # narrowed by _validate_pass_checks
    _require(convergence.get("status") == "PASS",
             "candidate convergence gate status is not PASS")
    accepted = checks["accepted_updates"]
    _require(
        accepted.get("value") == CANONICAL_ACCEPTED_UPDATES
        and accepted.get("op") == "=="
        and accepted.get("threshold") == CANONICAL_ACCEPTED_UPDATES,
        "candidate convergence gate does not prove exactly 4800 accepted updates",
    )
    snapshots = convergence.get("snapshots")
    _require(isinstance(snapshots, Mapping),
             "candidate convergence snapshots are missing")
    _require(
        set(snapshots) == {str(step) for step in CANONICAL_CONVERGENCE_SNAPSHOT_STEPS},
        "candidate convergence snapshots are not the six predeclared snapshots",
    )
    for step in CANONICAL_CONVERGENCE_SNAPSHOT_STEPS:
        snapshot = snapshots[str(step)]
        _require(isinstance(snapshot, Mapping),
                 f"candidate convergence snapshot {step} is invalid")
        _validate_proposal_scoring(
            snapshot.get("proposal_scoring"),
            label=f"convergence snapshot {step}",
        )
        _require(snapshot.get("n_groups") == 400 and snapshot.get("n_tasks") == 40,
                 f"candidate convergence snapshot {step} is not full validation")
        groups = snapshot.get("groups")
        _require(isinstance(groups, list) and len(groups) == 400,
                 f"candidate convergence snapshot {step} lacks 400 group rows")
        _require(_is_int(snapshot.get("informative_groups"))
                 and snapshot["informative_groups"] >= 200,
                 f"candidate convergence snapshot {step} lacks informative groups")
        checkpoint = snapshot.get("checkpoint")
        _require(isinstance(checkpoint, Mapping),
                 f"candidate convergence snapshot {step} checkpoint is missing")
        _require(checkpoint.get("global_step") == step,
                 f"candidate convergence snapshot {step} checkpoint step drifted")
        if step != AUTHORITATIVE_BASELINE_STEP:
            _require(checkpoint.get("kind") == "outcome_training_checkpoint"
                     and checkpoint.get("config_hash") == CANONICAL_CONFIG_HASH,
                     f"candidate convergence snapshot {step} is unauthenticated")
    efficacy = convergence.get("efficacy")
    _require(isinstance(efficacy, Mapping) and efficacy.get("pass") is True,
             "candidate heldout efficacy evidence is not PASS")
    plateau = convergence.get("plateau")
    _require(isinstance(plateau, list) and len(plateau) == 4
             and all(isinstance(row, Mapping) and row.get("pass") is True
                     for row in plateau),
             "candidate heldout plateau evidence is not four PASS comparisons")
    metric_gate = convergence.get("training_metrics")
    metric_checks = _validate_pass_checks(
        metric_gate, label="convergence training-metrics gate",
        expected={
            "accepted_updates",
            "anchor_sparse_ce_block_median_relative_range",
            "anchor_sparse_ce_terminal_block_median",
        },
    )
    _require(metric_checks == {name: checks[name] for name in metric_checks},
             "candidate convergence metric checks disagree with the terminal checks")

    trust = training.get("trust_gate")
    trust_checks = _validate_pass_checks(
        trust, label="trust gate", expected=CANONICAL_TRUST_CHECKS,
    )
    assert isinstance(trust, Mapping)  # narrowed by _validate_pass_checks
    _validate_proposal_scoring(
        trust.get("proposal_scoring"), label="terminal trust gate",
    )
    expected_thresholds = {
        "clip_fraction": ("<=", 0.20),
        "ess_fraction": (">=", 0.80),
        "coeff_drift_p95": ("<=", 0.05),
        "live_ops": (">=", 16),
        "nonfinite": ("==", 0),
        "unexpected_gradients": ("==", 0),
    }
    for name, (op, threshold) in expected_thresholds.items():
        row = trust_checks[name]
        _require(row.get("op") == op and row.get("threshold") == threshold,
                 f"candidate trust threshold {name} drifted")
    support = trust_checks["expert_topk_overlap_change"]
    _require(support.get("threshold") == -0.05,
             "candidate expert-support retention threshold drifted")
    counts = trust.get("counts")
    _require(isinstance(counts, Mapping)
             and _is_int(counts.get("ratio_atoms")) and counts["ratio_atoms"] > 0
             and _is_int(counts.get("arm0_drift_atoms")) and counts["arm0_drift_atoms"] > 0
             and _is_int(counts.get("arm0_usage_atoms")) and counts["arm0_usage_atoms"] > 0
             and counts.get("training_nonfinite") == 0
             and counts.get("final_nonfinite") == 0,
             "candidate trust gate atom counts are invalid")
    return convergence, trust


def _validate_terminal_evaluation(
    training: Mapping[str, Any],
    *,
    convergence: Mapping[str, Any],
    final_proposal: Mapping[str, Any],
) -> dict[str, Any]:
    value = training.get("terminal_evaluation")
    _require(isinstance(value, Mapping),
             "candidate terminal execution provenance is missing")
    execution = dict(value)
    _require(
        execution.get("parallelism") == CANONICAL_TERMINAL_PARALLELISM
        and execution.get("world_size") == 8,
        "candidate terminal evaluation parallelism/world size drifted",
    )
    live = _validate_model_digest(
        execution.get("live_proposal"), label="terminal live proposal",
    )
    _require(live == final_proposal,
             "candidate terminal live proposal differs from final proposal")
    tasks = execution.get("tasks")
    _require(isinstance(tasks, list) and len(tasks) == 8,
             "candidate terminal evaluation lacks eight rank tasks")
    assignments = [
        ("snapshot", step) for step in CANONICAL_CONVERGENCE_SNAPSHOT_STEPS
    ] + [("trust", CANONICAL_CANDIDATE_STEP), ("idle", None)]
    for rank, (kind, step) in enumerate(assignments):
        label = f"terminal rank {rank}"
        value = tasks[rank]
        _require(isinstance(value, Mapping), f"candidate {label} task is invalid")
        row = dict(value)
        elapsed = row.get("elapsed_seconds")
        _require(
            row.get("rank") == rank
            and row.get("kind") == kind
            and row.get("step") == step
            and isinstance(elapsed, (int, float))
            and not isinstance(elapsed, bool)
            and math.isfinite(float(elapsed))
            and float(elapsed) >= 0.0,
            f"candidate {label} task assignment is not canonical",
        )
        _validate_proposal_scoring(row.get("proposal_scoring"), label=label)
        task_proposal = _validate_model_digest(
            row.get("live_proposal"), label=f"{label} live proposal",
        )
        _require(task_proposal == live,
                 f"candidate {label} did not hold the common final proposal")

    snapshots = convergence.get("snapshots")
    assert isinstance(snapshots, Mapping)  # validated by _validate_terminal_gates
    final_snapshot = snapshots[str(CANONICAL_CANDIDATE_STEP)]
    assert isinstance(final_snapshot, Mapping)
    checkpoint = final_snapshot.get("checkpoint")
    _require(isinstance(checkpoint, Mapping),
             "candidate final convergence checkpoint is missing")
    snapshot_proposal = _validate_model_digest(
        checkpoint.get("proposal"), label="step-54466 snapshot proposal",
    )
    _require(snapshot_proposal == live,
             "candidate step-54466 snapshot differs from final proposal")
    return execution


def _validate_terminal_report(
    source: Path,
    *,
    checkpoint_sha256: str,
    checkpoint_size: int,
    training: Mapping[str, Any],
) -> dict[str, Any]:
    report_path = source.parent / "terminal_report.json"
    resolved, digest, report = _read_json(report_path, "candidate terminal report")
    _require(report.get("status") == "PASS" and report.get("passed") is True,
             "candidate terminal report is not PASS")
    _require(report.get("candidate_emitted") is True,
             "candidate terminal report does not prove candidate emission")
    _require(_resolved_checkpoint_path(report.get("path"), "terminal report path") == source,
             "candidate terminal report names a different checkpoint")
    _require(report.get("sha256") == checkpoint_sha256,
             "candidate terminal report checkpoint SHA-256 differs")
    _require(report.get("size") == checkpoint_size,
             "candidate terminal report checkpoint size differs")
    _require(report.get("global_step") == CANONICAL_CANDIDATE_STEP
             and report.get("optimizer_steps") == CANONICAL_ACCEPTED_UPDATES
             and report.get("config_hash") == CANONICAL_CONFIG_HASH,
             "candidate terminal report lineage differs from the frozen stage")
    convergence = training["convergence_gate"]
    trust = training["trust_gate"]
    _require(report.get("convergence_gate") == convergence,
             "candidate terminal convergence report differs from checkpoint")
    _require(report.get("trust_gate") == trust,
             "candidate terminal trust report differs from checkpoint")
    combined = {
        **{f"convergence/{name}": row
           for name, row in convergence["checks"].items()},
        **{f"trust/{name}": row for name, row in trust["checks"].items()},
    }
    _require(report.get("checks") == combined,
             "candidate terminal combined checks differ from embedded gates")
    _require(report.get("frozen_model") == training.get("frozen_model")
             and report.get("proposal") == training.get("final_proposal"),
             "candidate terminal model digests differ from checkpoint provenance")
    verification = report.get("verification")
    _require(isinstance(verification, Mapping)
             and verification.get("weights_only") is True
             and verification.get("consolidated_step") == CANONICAL_CANDIDATE_STEP,
             "candidate terminal weights-only verification is missing")
    loaded = verification.get("load_policy")
    _require(isinstance(loaded, Mapping)
             and loaded.get("is_stub") is False
             and loaded.get("global_step") == CANONICAL_CANDIDATE_STEP
             and loaded.get("config_hash") == CANONICAL_CONFIG_HASH,
             "candidate terminal real-policy load verification is missing")
    state = loaded.get("state_dict")
    _require(isinstance(state, Mapping),
             "candidate terminal policy-load tensor provenance is missing")
    for module in DIRECT_POLICY_MODULES:
        row = state.get(module)
        _require(isinstance(row, Mapping)
                 and _is_int(row.get("tensors_loaded"))
                 and row["tensors_loaded"] > 0
                 and row.get("unexpected") == 0,
                 f"candidate terminal verification did not load {module}")
    return {
        "path": str(resolved),
        "sha256": digest,
        "status": "PASS",
        "passed": True,
        "candidate_emitted": True,
    }


def checkpoint_provenance(path: str | os.PathLike[str]) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    digest = sha256_file(source)
    payload = _load_checkpoint(source)
    _require(isinstance(payload.get("consolidated"), Mapping),
             "candidate is not a consolidated checkpoint")
    model = payload.get("model")
    _require(isinstance(model, dict) and model, "candidate checkpoint has no model state")
    for module in DIRECT_POLICY_MODULES:
        state = submodule_state(model, module)
        _require(isinstance(state, dict) and state,
                 f"candidate checkpoint has no {module} state")
        _require(all(isinstance(value, torch.Tensor) for value in state.values()),
                 f"candidate {module} state contains non-tensors")
    actual_final_proposal = _model_state_digest(model, proposal=True)
    actual_frozen_model = _model_state_digest(model, proposal=False)

    _require(source.name == f"candidate_{CANONICAL_CANDIDATE_STEP:09d}.pt",
             "candidate checkpoint name is not the canonical terminal artifact")
    step = payload.get("global_step")
    _require(step == CANONICAL_CANDIDATE_STEP,
             "candidate checkpoint is not the exact step-54,466 artifact")
    cfg = payload.get("resolved_config")
    _require(isinstance(cfg, Mapping), "candidate checkpoint has no resolved_config")
    saved_config_hash = payload.get("config_hash")
    _require(saved_config_hash == CANONICAL_CONFIG_HASH,
             "candidate checkpoint config_hash is not the frozen canonical hash")
    computed_config_hash = _experiment_config_hash(cfg)
    _require(computed_config_hash == saved_config_hash,
             "candidate resolved_config does not match config_hash")

    recipe = cfg.get("outcome_grpo")
    _require(isinstance(recipe, Mapping) and recipe.get("format_version") == 1,
             "candidate resolved_config has no versioned outcome_grpo section")
    _require(
        recipe.get("start_step") == AUTHORITATIVE_BASELINE_STEP
        and recipe.get("stop_step") == CANONICAL_CANDIDATE_STEP
        and recipe.get("updates_per_fold") == 800
        and recipe.get("world_size") == 8
        and cfg.get("train_modules") == ["proposal"],
        "candidate resolved_config is not the frozen six-fold proposal-only stage",
    )
    training = payload.get("outcome_grpo")
    _require(isinstance(training, Mapping),
             "candidate checkpoint has no outcome-GRPO provenance")
    _require(training.get("format_version") == 1
             and training.get("kind") == CANONICAL_TRAINER_KIND,
             "candidate was not emitted by the canonical outcome-GRPO trainer")
    _require(training.get("descendant_config_hash") == CANONICAL_CONFIG_HASH
             and training.get("descendant_global_step") == CANONICAL_CANDIDATE_STEP
             and training.get("optimizer_steps") == CANONICAL_ACCEPTED_UPDATES,
             "candidate embedded descendant lineage is not canonical")
    _require(training.get("parent_config_hash") == CANONICAL_PARENT_CONFIG_HASH
             and training.get("parent_global_step") == AUTHORITATIVE_BASELINE_STEP,
             "candidate embedded parent lineage is not canonical")
    _require(training.get("mutated_model_prefixes") == ["proposal."],
             "candidate mutated modules beyond the proposal")
    parent = training.get("parent")
    _require(isinstance(parent, Mapping)
             and parent.get("sha256") == CANONICAL_PARENT_CHECKPOINT_SHA256
             and parent.get("global_step") == AUTHORITATIVE_BASELINE_STEP
             and parent.get("config_hash") == CANONICAL_PARENT_CONFIG_HASH,
             "candidate parent checkpoint identity is not authoritative")
    trainer_source_value = training.get("trainer_source")
    trainer_source = _validate_trainer_source(trainer_source_value)
    _require(
        training.get("strict_determinism") == CANONICAL_STRICT_DETERMINISM,
        "candidate training provenance lacks strict determinism evidence",
    )
    collections = training.get("collections")
    _require(isinstance(collections, list) and len(collections) == 6,
             "candidate does not contain all six training folds")
    for index, split in enumerate(CANONICAL_COLLECTION_SPLITS):
        _validate_collection_provenance(collections[index], split=split, n_groups=200)
    validation = training.get("validation")
    _validate_collection_provenance(validation, split="validation", n_groups=400)
    _require(training.get("collection") == validation,
             "candidate legacy collection binding is not the validation split")
    _, exact_behaviour = _validate_behaviour_authentication(training)
    start_checkpoint = _validate_start_checkpoint_identity(training)
    assert isinstance(trainer_source_value, Mapping)
    initial_behavior = _validate_initial_behavior_ratio_identity(
        training,
        trainer_source=trainer_source_value,
        parent=parent,
        exact_behaviour=exact_behaviour,
        start_checkpoint=start_checkpoint,
    )
    embedded_recipe = training.get("recipe")
    _require(isinstance(embedded_recipe, Mapping)
             and embedded_recipe.get("algorithm") == "stored_order_pl_clipped_grpo"
             and embedded_recipe.get("reward") == "terminal_LIBERO_success_only"
             and embedded_recipe.get("sampled_arms") == list(range(1, 8))
             and embedded_recipe.get("folds") == 6
             and embedded_recipe.get("updates_per_fold") == 800
             and embedded_recipe.get("forbidden") == ["Phi", "bank", "shaped_reward"],
             "candidate embedded training recipe is not canonical")
    training_stats = training.get("training")
    _require(isinstance(training_stats, Mapping)
             and training_stats.get("optimizer_steps") == CANONICAL_ACCEPTED_UPDATES
             and training_stats.get("unexpected_gradients") == []
             and training_stats.get("nonfinite") == 0,
             "candidate training provenance does not prove 4800 clean updates")
    initial_proposal = _validate_model_digest(
        training.get("initial_proposal"), label="embedded initial proposal",
    )
    training_initial_proposal = _validate_model_digest(
        training_stats.get("initial_proposal"),
        label="training initial proposal",
    )
    _require(
        start_checkpoint.get("proposal")
        == initial_proposal
        == training_initial_proposal,
        "candidate START proposal is not bound to both initial-proposal records",
    )
    _require(training.get("world_size") == 8,
             "candidate training provenance does not prove world size 8")
    reset = training.get("optimizer_reset")
    _require(reset == {
        "count": 1, "modules": ["proposal"],
        "source_global_step": AUTHORITATIVE_BASELINE_STEP,
    }, "candidate proposal optimizer-reset provenance differs")
    convergence, trust = _validate_terminal_gates(training)
    final_proposal = _validate_model_digest(
        training.get("final_proposal"), label="embedded final proposal",
    )
    frozen_model = _validate_model_digest(
        training.get("frozen_model"), label="embedded frozen model",
    )
    _require(final_proposal == actual_final_proposal,
             "candidate embedded final proposal does not match model tensors")
    _require(frozen_model == actual_frozen_model,
             "candidate embedded frozen model does not match model tensors")
    terminal_evaluation = _validate_terminal_evaluation(
        training, convergence=convergence, final_proposal=final_proposal,
    )

    consolidated = payload.get("consolidated")
    _require(isinstance(consolidated, Mapping)
             and consolidated.get("tool") == "loom.train.outcome_grpo"
             and consolidated.get("step") == CANONICAL_CANDIDATE_STEP
             and consolidated.get("derivation") == "proposal-only terminal outcome GRPO"
             and consolidated.get("mutated_model_prefixes") == ["proposal."],
             "candidate consolidated provenance is not canonical")
    consolidated_parent = consolidated.get("parent_checkpoint")
    _require(isinstance(consolidated_parent, Mapping)
             and consolidated_parent.get("sha256") == CANONICAL_PARENT_CHECKPOINT_SHA256
             and consolidated_parent.get("global_step") == AUTHORITATIVE_BASELINE_STEP
             and consolidated_parent.get("config_hash") == CANONICAL_PARENT_CONFIG_HASH,
             "candidate consolidated parent lineage is not authoritative")
    optimizer = payload.get("optimizer")
    _require(isinstance(optimizer, Mapping)
             and optimizer.get("kind") == "proposal_only_adamw"
             and optimizer.get("state_reset_at_entry") is True,
             "candidate optimizer provenance is not proposal-only/reset-once")
    _require(payload.get("world_size") == 8
             and payload.get("stop_reason") == "terminal_outcome_grpo",
             "candidate terminal execution provenance differs")
    terminal_report = _validate_terminal_report(
        source, checkpoint_sha256=digest, checkpoint_size=source.stat().st_size,
        training=training,
    )

    return {
        "path": str(source),
        "sha256": digest,
        "bytes": source.stat().st_size,
        "global_step": int(step),
        "config_hash": str(saved_config_hash),
        "resolved_config_sha256": _canonical_json_sha256(cfg),
        "outcome_grpo_config_sha256": _canonical_json_sha256(recipe),
        "training_provenance_sha256": _canonical_json_sha256(training),
        "trainer_source": trainer_source,
        "convergence_gate_sha256": _canonical_json_sha256(convergence),
        "trust_gate_sha256": _canonical_json_sha256(trust),
        "exact_behaviour_identity_sha256": _canonical_json_sha256(exact_behaviour),
        "initial_behavior_ratio_identity_sha256": _canonical_json_sha256(
            initial_behavior
        ),
        "terminal_evaluation_sha256": _canonical_json_sha256(terminal_evaluation),
        "terminal_report": terminal_report,
        "saved_git_sha": payload.get("git_sha"),
        "consolidated": json.loads(json.dumps(payload["consolidated"], default=str)),
    }


def bind_candidate_checkpoint(
    results: ValidatedResults,
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    meta = results.metadata
    _require(meta.get("global_step") == checkpoint.get("global_step"),
             "candidate result step does not match supplied checkpoint")
    _require(meta.get("config_hash") == checkpoint.get("config_hash"),
             "candidate result config hash does not match supplied checkpoint")
    metadata_path = Path(str(meta["checkpoint_path"])).expanduser().resolve()
    _require(metadata_path.is_file(),
             f"candidate result checkpoint path does not exist: {metadata_path}")
    metadata_sha = sha256_file(metadata_path)
    _require(metadata_sha == checkpoint.get("sha256"),
             "candidate result checkpoint SHA-256 does not match supplied checkpoint")

    # Newer result writers may add an explicit digest.  Every recognized field
    # is authenticated when present; older runner-v1 results are bound by
    # hashing their exact recorded path above.
    explicit = dict(meta.get("explicit_checkpoint_sha256") or {})
    for name, value in explicit.items():
        _require(value == checkpoint.get("sha256"),
                 f"candidate result {name} does not match supplied checkpoint")
    return {
        "metadata_checkpoint_path": str(metadata_path),
        "metadata_checkpoint_sha256": metadata_sha,
        "supplied_checkpoint_path": checkpoint["path"],
        "supplied_checkpoint_sha256": checkpoint["sha256"],
        "same_resolved_path": str(metadata_path) == checkpoint["path"],
        "global_step": checkpoint["global_step"],
        "config_hash": checkpoint["config_hash"],
        "explicit_result_sha256_fields": explicit,
        "passed": True,
    }


def promotion_verdict(
    baseline: ValidatedResults,
    candidate: ValidatedResults,
    *,
    min_total: int = CANDIDATE_MIN_TOTAL,
    min_by_suite: Mapping[str, int] = CANDIDATE_MIN_BY_SUITE,
) -> dict[str, Any]:
    _require(set(baseline.outcomes) == set(candidate.outcomes),
             "baseline and candidate WorkItem keys differ")
    _require(baseline.work_items_sha256 == candidate.work_items_sha256,
             "baseline and candidate WorkItem/env/policy-seed identities differ")
    new_only = sorted(
        key for key in baseline.outcomes
        if not baseline.outcomes[key] and candidate.outcomes[key]
    )
    old_only = sorted(
        key for key in baseline.outcomes
        if baseline.outcomes[key] and not candidate.outcomes[key]
    )
    both = sum(
        baseline.outcomes[key] and candidate.outcomes[key]
        for key in baseline.outcomes
    )
    neither = EXPECTED_WORK_ITEMS - both - len(new_only) - len(old_only)
    checks: dict[str, dict[str, Any]] = {
        "candidate_total": {
            "value": candidate.n_success,
            "requirement": f">= {min_total}",
            "pass": candidate.n_success >= min_total,
        },
        "paired_new_only_gt_old_only": {
            "new_only": len(new_only),
            "old_only": len(old_only),
            "requirement": "new_only > old_only",
            "pass": len(new_only) > len(old_only),
        },
    }
    for suite in DEFAULT_LIBERO_SUITES:
        threshold = int(min_by_suite[suite])
        value = int(candidate.suite_success[suite])
        checks[f"suite_{suite}"] = {
            "value": value,
            "requirement": f">= {threshold}",
            "pass": value >= threshold,
        }
    failures = [name for name, row in checks.items() if row["pass"] is not True]
    return {
        "passed": not failures,
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "checks": checks,
        "baseline": {
            "n_success": baseline.n_success,
            "n": EXPECTED_WORK_ITEMS,
            "success_rate": 100.0 * baseline.n_success / EXPECTED_WORK_ITEMS,
            "suite_success": dict(baseline.suite_success),
        },
        "candidate": {
            "n_success": candidate.n_success,
            "n": EXPECTED_WORK_ITEMS,
            "success_rate": 100.0 * candidate.n_success / EXPECTED_WORK_ITEMS,
            "suite_success": dict(candidate.suite_success),
        },
        "paired": {
            "new_only": len(new_only),
            "old_only": len(old_only),
            "both_success": int(both),
            "neither_success": int(neither),
            "net_success_delta": len(new_only) - len(old_only),
            "new_only_work_items_sha256": _canonical_json_sha256(
                [list(key) for key in new_only]
            ),
            "old_only_work_items_sha256": _canonical_json_sha256(
                [list(key) for key in old_only]
            ),
        },
    }


def _source_digest_from_entries(entries: Sequence[Mapping[str, str]]) -> str:
    digest = hashlib.sha256()
    digest.update(SOURCE_DIGEST_SCHEME.encode("utf-8") + b"\0")
    for entry in entries:
        rel = str(entry["path"])
        file_sha = entry["sha256"]
        _require(_is_sha256(file_sha), f"invalid source SHA-256 for {rel}")
        digest.update(rel.encode("utf-8") + b"\0")
        digest.update(bytes.fromhex(file_sha) + b"\0")
    return digest.hexdigest()


def source_provenance(
    root: str | os.PathLike[str] = ROOT,
    files: Sequence[str] = SOURCE_FILES,
) -> dict[str, Any]:
    base = Path(root).expanduser().resolve()
    entries: list[dict[str, str]] = []
    for rel in files:
        source = base / rel
        _require(source.is_file(), f"promotion-gate source is missing: {source}")
        entries.append({"path": rel, "sha256": sha256_file(source)})
    return {
        "scheme": SOURCE_DIGEST_SCHEME,
        "files": entries,
        "sha256": _source_digest_from_entries(entries),
    }


def assert_source_unchanged(
    expected: Mapping[str, Any],
    *,
    root: str | os.PathLike[str] = ROOT,
) -> None:
    files = expected.get("files")
    _require(isinstance(files, list) and files, "source provenance is incomplete")
    relpaths = [entry.get("path") for entry in files if isinstance(entry, Mapping)]
    _require(all(isinstance(path, str) for path in relpaths),
             "source provenance paths are invalid")
    current = source_provenance(root, relpaths)
    _require(current == expected, "promotion-gate source changed during execution")


def _git_sha() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            stderr=subprocess.DEVNULL, timeout=30,
        ).decode().strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
            stderr=subprocess.DEVNULL, timeout=30,
        ).decode().strip()
        return f"{sha}-dirty" if dirty else sha
    except Exception:  # noqa: BLE001
        return "unknown"


def _result_provenance(result: ValidatedResults) -> dict[str, Any]:
    return {
        "path": result.path,
        "sha256": result.sha256,
        "n_work_items": len(result.outcomes),
        "n_errors": 0,
        "n_success": result.n_success,
        "suite_success": dict(result.suite_success),
        "work_items_sha256": result.work_items_sha256,
        "outcomes_sha256": result.outcomes_sha256,
        "metadata": dict(result.metadata),
    }


def execute_gate(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    source = source_provenance()
    baseline_path = Path(args.baseline).expanduser().resolve()
    candidate_path = Path(args.candidate).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_path = Path(args.out).expanduser().resolve()
    _require(len({baseline_path, candidate_path, checkpoint_path, output_path}) == 4,
             "baseline, candidate, checkpoint, and output paths must be distinct")

    baseline = validate_baseline(baseline_path)
    checkpoint = checkpoint_provenance(checkpoint_path)
    candidate = validate_results(
        candidate_path,
        label="candidate",
        expected_step=checkpoint["global_step"],
        expected_config_hash=checkpoint["config_hash"],
    )
    binding = bind_candidate_checkpoint(candidate, checkpoint)
    _require(baseline.work_items_sha256 == candidate.work_items_sha256,
             "candidate WorkItem/env_seed/policy_seed set differs from baseline")
    verdict = promotion_verdict(baseline, candidate)

    # Rehash all immutable inputs and source after every semantic check.  A
    # concurrent replacement cannot be published under an earlier identity.
    _require(sha256_file(baseline.path) == baseline.sha256,
             "baseline results changed during gate execution")
    _require(sha256_file(candidate.path) == candidate.sha256,
             "candidate results changed during gate execution")
    _require(sha256_file(checkpoint["path"]) == checkpoint["sha256"],
             "candidate checkpoint changed during gate execution")
    _require(
        sha256_file(checkpoint["terminal_report"]["path"])
        == checkpoint["terminal_report"]["sha256"],
        "candidate terminal report changed during gate execution",
    )
    assert_source_unchanged(source)

    return {
        "format_version": FORMAT_VERSION,
        "gate": GATE_NAME,
        "status": verdict["status"],
        "passed": verdict["passed"],
        "requirements": {
            "official_seed": 0,
            "n_work_items": EXPECTED_WORK_ITEMS,
            "n_errors": 0,
            "candidate_min_success": CANDIDATE_MIN_TOTAL,
            "candidate_min_suite_success": dict(CANDIDATE_MIN_BY_SUITE),
            "paired": "new_only > old_only",
            "real_checkpoint_backed": True,
            "canonical_candidate_step": CANONICAL_CANDIDATE_STEP,
            "canonical_accepted_updates": CANONICAL_ACCEPTED_UPDATES,
            "canonical_config_hash": CANONICAL_CONFIG_HASH,
            "canonical_trainer_kind": CANONICAL_TRAINER_KIND,
            "canonical_trainer_source_sha256": CANONICAL_TRAINER_SOURCE_SHA256,
            "terminal_convergence_pass": True,
            "terminal_trust_pass": True,
            "terminal_candidate_emitted": True,
        },
        "inputs": {
            "baseline_results": _result_provenance(baseline),
            "candidate_results": _result_provenance(candidate),
            "candidate_checkpoint": checkpoint,
            "checkpoint_binding": binding,
        },
        "results": verdict,
        "source_provenance": {
            **source,
            "git_sha": _git_sha(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "hostname": os.environ.get("SLURMD_NODENAME") or platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "execution": {
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "wall_seconds": float(time.monotonic() - started),
            "simulator_episodes": 0,
            "checkpoint_mutated": False,
        },
    }


def atomic_publish_json(
    path: str | os.PathLike[str],
    value: Mapping[str, Any],
) -> str:
    """Publish one complete JSON document without replacing an existing path."""
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
            raise PromotionGateError(
                f"refusing to overwrite existing promotion report: {target}"
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", required=True,
        help="authoritative step-49,666 official seed-0 results.json",
    )
    parser.add_argument(
        "--candidate", required=True,
        help="candidate official seed-0 results.json",
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="candidate consolidated outcome-GRPO checkpoint",
    )
    parser.add_argument("--out", required=True, help="new exclusive JSON report path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.monotonic()
    try:
        report = execute_gate(args)
        code = 0 if report["passed"] else 1
    except Exception as exc:  # noqa: BLE001 -- every integrity failure is persisted
        report = {
            "format_version": FORMAT_VERSION,
            "gate": GATE_NAME,
            "status": "ERROR",
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "requested": vars(args),
            "execution": {
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
