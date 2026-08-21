#!/usr/bin/env python3
"""Fail-closed overnight R0 formal-training and LIBERO-evaluation chain.

This file owns orchestration only.  It neither implements the training loss nor
selects a checkpoint from evaluation outcomes.  One immutable plan submits:

  three 4 h links -> 32k convergence -> conditional extension -> terminal gate
  -> one consolidation -> three parallel singleton-seed evaluations -> merge

Every scheduler edge is ``afterok``.  Outputs are new and exclusive, the three
evaluation seed sets are disjoint, and W&B metadata shares one formal lineage.
No job is submitted merely by importing or dry-running this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from loom.train.direct_formal import (
    DIRECT_FORMAL_FORMAT,
    DirectFormalGate,
    receipt_exit_code,
)


ROOT = Path(__file__).resolve().parents[1]
FORMAT_VERSION = 1
KIND = "r0_e2e_formal_overnight_chain"
PROJECT = "loom-r0-e2e-scratch"
CANONICAL_FORMAL_CONFIG = (ROOT / "configs" / "r0a_dual_code_formal.yaml").resolve()
CANONICAL_FORMAL_CONFIG_SHA256 = (
    "68dbf36739b7abf8a80606d1f6d21cf5c450dbd946ebd9e2853738e665758f42"
)
CANONICAL_FORMAL_RESOLVED_HASH = "d030206d56a71718"
DUAL_ACTION_MODE = "dual_q_action_proposal"
STEP_32K = 32_000
STEP_40K = 40_000
INITIAL_LINKS = 3
WORLD_SIZE = 16
SEEDS = (0, 1, 2)
EXPECTED_EPISODES_PER_SEED = 400
EXPECTED_EPISODES_TOTAL = 1_200
BASELINE_SUCCESS_PER_SEED = 149
BASELINE_SUCCESS_TOTAL = 447
BASELINE_CHECKPOINT_STEP = 49_666
CANONICAL_BASELINE_ROOT = (
    ROOT / "runs" / "eval_r0a_deploy_s1_s49666_seeded1200_v2"
).resolve()
BASELINE_RESULT_SHA256 = {
    0: "95e3ac186c28a6305f5fff3375b45c5184697a0a94f7c08c511ab0d09fd27f3a",
    1: "1206ede3fae1c81d5d38f0d0c6cc1c3fe0fde97683a1733c4d44c6f25fea90c4",
    2: "83a7396bc305f43e5ec9ea77bbfa987aa3dd8381eb6525426856a1b26cf4fd56",
}
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 49_666
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_MATRIX_SHA256 = (
    "1e570b6d13426c8fbd58016d0fba6869dc18aa3151dfdbc0bab357373cacf32e"
)
BASELINE_POLICY_KW = {"allow_stub": False, "op_stats": True}
CANDIDATE_POLICY_KW = {
    "allow_stub": False,
    "op_stats": True,
    "embodiment": "libero_franka",
}
EVAL_IDENTITY_NORMALIZATION = {
    "executed_embodiment": "libero_franka",
    "policy_meta_embodiment_required_for_baseline_and_candidate": True,
    "historical_baseline_policy_kw": BASELINE_POLICY_KW,
    "current_candidate_policy_kw": CANDIDATE_POLICY_KW,
    "rationale": (
        "the current runner materializes the same libero_franka bench default "
        "that the historical evaluator left implicit"
    ),
}
LIBERO_EVAL_PYTHON = Path(
    "/lustre/fsw/portfolios/edgeai/users/chrislin/envs/loom-libero/bin/python"
)
WANDB_PUBLISH_ATTEMPTS = 5
WANDB_RETRY_SECONDS = 15
SUITE_FLOORS_PERCENT = {
    "libero_spatial": 35.0,
    "libero_object": 27.0,
    "libero_goal": 43.0,
    "libero_long": 24.0,
}
DIRECT_STATUS_CODES = {"PASS": 0, "MOVING": 1, "INVALID": 2, "ABORT": 3}
ORCHESTRATION_SOURCE_FILES = (
    "contracts.py",
    "loom/__init__.py",
    "loom/data/__init__.py",
    "loom/data/adapters/__init__.py",
    "loom/data/adapters/libero.py",
    "loom/data/cache.py",
    "loom/data/canonical.py",
    "loom/data/loader.py",
    "loom/data/tower.py",
    "loom/heads/decoder.py",
    "loom/heads/__init__.py",
    "loom/heads/proposal.py",
    "loom/heads/q_action.py",
    "loom/heads/q_delta.py",
    "loom/losses/dyn.py",
    "loom/losses/__init__.py",
    "loom/losses/act.py",
    "loom/losses/balance.py",
    "loom/losses/proposal_bc.py",
    "loom/model/bank.py",
    "loom/model/__init__.py",
    "loom/model/estimator.py",
    "loom/model/rollout.py",
    "stubs.py",
    "loom/train/atomic.py",
    "loom/train/__init__.py",
    "loom/train/ckpt.py",
    "loom/train/determinism.py",
    "loom/train/direct_formal.py",
    "loom/train/fsdp.py",
    "loom/train/loop.py",
    "loom/train/preempt.py",
    "loom/train/schedule.py",
    "loom/train/wandb_util.py",
    "loom/train/consolidate.py",
    "loom/eval/__init__.py",
    "loom/eval/__main__.py",
    "loom/eval/libero.py",
    "loom/eval/policy.py",
    "loom/eval/runner.py",
    "loom/eval/table.py",
    "scripts/direct_formal_convergence.py",
    "scripts/env.sh",
    "scripts/r0_e2e_formal_chain.py",
    "scripts/r0_e2e_formal_train_entry.py",
    "scripts/r0_e2e_formal_train.sbatch",
    "scripts/r0_e2e_formal_control.sbatch",
    "scripts/r0_e2e_formal_consolidate.sbatch",
    "scripts/r0_e2e_formal_eval_seed.sbatch",
)


class ChainError(RuntimeError):
    """A fail-closed orchestration, identity, or execution error."""


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _pretty_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def exclusive_json_write(path: str | Path, value: Mapping[str, Any]) -> None:
    """Atomically publish a new JSON file without replacing a raced writer."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = _pretty_json(value)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)


def _absolute_new_path(value: str, *, field: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ChainError(f"{field} must be absolute, got {value!r}")
    return path.resolve()


def _source_closure() -> dict[str, Any]:
    files = {name: sha256_file(ROOT / name) for name in ORCHESTRATION_SOURCE_FILES}
    digest = hashlib.sha256()
    for name in sorted(files):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(files[name].encode())
        digest.update(b"\0")
    return {
        "scheme": "sha256(path-nul-sha256-nul)-v1",
        "sha256": digest.hexdigest(),
        "files": files,
    }


def _require_isolated_paths(*paths: Path) -> None:
    for index, left in enumerate(paths):
        for right in paths[index + 1:]:
            if left == right or left in right.parents or right in left.parents:
                raise ChainError(
                    "run_dir, control_dir, and artifact_root must be pairwise "
                    f"non-nested: {left} vs {right}"
                )


def _load_resolved_config(config_path: Path) -> tuple[dict[str, Any], str]:
    # Import the public config resolver lazily.  The orchestrator deliberately
    # does not copy YAML merge semantics or modify the training loop.
    from loom.train.loop import config_hash, read_config  # noqa: PLC0415

    cfg = read_config(config_path)
    return cfg, config_hash(cfg)


def _expected_direct_formal_config(gate: DirectFormalGate) -> dict[str, Any]:
    """Translate the code-owned gate into the config's provenance schema."""
    return {
        "format": DIRECT_FORMAL_FORMAT,
        "first_check": gate.first_check,
        "check_every": gate.check_every,
        "hard_cap": gate.max_updates,
        "block_size": gate.block_size,
        "block_count": gate.block_count,
        "tolerance": float(gate.tolerance),
        "reference_window": {
            "start_exclusive": gate.reference_start_exclusive,
            "end_inclusive": gate.reference_end_inclusive,
        },
        "primary": list(gate.primary_metrics),
        "health": {
            "delta_op_strict_gt": float(gate.delta_op_strict_gt),
            "delta_sel_h1_to_h4_strict_gt": float(gate.delta_sel_strict_gt),
            "act_align_strict_lt": float(gate.act_align_strict_lt),
            "live_ops_q_a_gte": int(gate.live_ops_q_a_gte),
            "live_ops_q_delta_gte": int(gate.live_ops_q_delta_gte),
            "proposal_ce_strict_lt": float(gate.proposal_off_floor_strict_lt),
            "c_delta_spread_strict_gt": float(gate.c_delta_spread_strict_gt),
            "gnorm_bank_strict_gt": float(gate.gnorm_bank_strict_gt),
            "gnorm_q_delta_strict_gt": float(gate.gnorm_q_delta_strict_gt),
            "other_trainable_gnorm_strict_gt": 0.0,
            "skipped_rate_strict_lt": float(gate.skipped_rate_strict_lt),
            "unexpected_module_gradients": False,
            "nonfinite": False,
        },
        "no_convergence_by_hard_cap": "ABORT_NO_EVALUATION",
    }


def _strict_equal(actual: Any, expected: Any) -> bool:
    """JSON-shaped equality that does not accept ``True == 1`` aliases."""
    if isinstance(expected, Mapping):
        return (
            isinstance(actual, Mapping)
            and set(actual) == set(expected)
            and all(_strict_equal(actual[key], value) for key, value in expected.items())
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_strict_equal(left, right) for left, right in zip(actual, expected))
        )
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, int):
        return isinstance(actual, int) and not isinstance(actual, bool) and actual == expected
    if isinstance(expected, float):
        return isinstance(actual, (int, float)) and not isinstance(actual, bool) and float(actual) == expected
    return type(actual) is type(expected) and actual == expected


def validate_formal_config(
    cfg: Mapping[str, Any],
    *,
    project: str,
    step32: int,
    step40: int,
) -> dict[str, Any]:
    """Require the one-lineage 32k-schedule/40k-hard-cap contract."""
    from loom.train.loop import config_hash  # noqa: PLC0415

    gate = DirectFormalGate()
    if step32 != gate.schedule_horizon or step40 != gate.max_updates:
        raise ChainError("formal plan horizons differ from DirectFormalGate defaults")
    run = cfg.get("run")
    model = cfg.get("model")
    losses = cfg.get("losses")
    if not isinstance(run, Mapping) or not isinstance(model, Mapping):
        raise ChainError("resolved config requires run and model mappings")
    if not isinstance(losses, Mapping) or not isinstance(losses.get("act"), Mapping):
        raise ChainError("resolved config requires losses.act mapping")
    act = losses["act"]
    checks: dict[str, Any] = {
        "resolved_config_hash": config_hash(dict(cfg)),
        "project": run.get("project"),
        "seed": run.get("seed"),
        "steps": run.get("steps"),
        "schedule_horizon": run.get("schedule_horizon"),
        "max_updates": run.get("max_updates"),
        "fresh_start_required": run.get("fresh_start_required"),
        "require_online_wandb": run.get("require_online_wandb"),
        "ckpt_every": run.get("ckpt_every"),
        "keep_last": run.get("keep_last"),
        "use_stubs": model.get("use_stubs"),
        "act_decode_from": act.get("decode_from"),
        "act_enabled": act.get("enabled"),
        "act_weight": act.get("weight"),
        "act_align_to": act.get("align_to"),
        "train_modules": cfg.get("train_modules"),
        "direct_formal": cfg.get("direct_formal"),
        "evaluation_gate": cfg.get("evaluation_gate"),
    }
    expected = {
        "resolved_config_hash": CANONICAL_FORMAL_RESOLVED_HASH,
        "project": project,
        "seed": 0,
        "steps": step40,
        "schedule_horizon": step32,
        "max_updates": step40,
        "fresh_start_required": True,
        "require_online_wandb": True,
        "ckpt_every": gate.check_every,
        "use_stubs": False,
        "act_decode_from": DUAL_ACTION_MODE,
        "act_enabled": True,
        "act_weight": 1.0,
        "act_align_to": "q_a",
        "train_modules": list(gate.expected_gradient_modules),
        "direct_formal": _expected_direct_formal_config(gate),
        "evaluation_gate": {
            "seeds": list(SEEDS),
            "tasks_per_suite": 10,
            "trials_per_task": 10,
            "total_episodes": EXPECTED_EPISODES_TOTAL,
            "internal_baseline_successes": BASELINE_SUCCESS_TOTAL,
            "internal_baseline_episodes": EXPECTED_EPISODES_TOTAL,
            "paired_task_bootstrap_ci_low_strict_gt": 0.0,
            "seed0_stretch_successes_gte": 164,
            "suite_floor_percent": list(SUITE_FLOORS_PERCENT.values()),
            "canonical_baseline": {
                "kind": "r0a_deploy_seeded1200_v2_exact_baseline",
                "checkpoint_step": BASELINE_CHECKPOINT_STEP,
                "successes": BASELINE_SUCCESS_TOTAL,
                "episodes": EXPECTED_EPISODES_TOTAL,
                "seed_result_sha256": {
                    str(seed): digest
                    for seed, digest in BASELINE_RESULT_SHA256.items()
                },
            },
            "paired_task_bootstrap": {
                "kind": "fixed_suite_stratified_task_resample_matrix_v1",
                "samples": BOOTSTRAP_SAMPLES,
                "seed": BOOTSTRAP_SEED,
                "confidence": BOOTSTRAP_CONFIDENCE,
                "lower_quantile": 0.025,
                "upper_quantile": 0.975,
                "lower_interpolation": "lower",
                "upper_interpolation": "higher",
                "matrix_sha256": BOOTSTRAP_MATRIX_SHA256,
            },
        },
    }
    mismatches = {
        key: {"expected": expected[key], "actual": checks[key]}
        for key in expected if not _strict_equal(checks[key], expected[key])
    }
    keep_last = checks["keep_last"]
    if (
        not isinstance(keep_last, int)
        or isinstance(keep_last, bool)
        or keep_last < 20
    ):
        mismatches["keep_last"] = {
            "expected": "integer >= 20",
            "actual": keep_last,
        }
    if mismatches:
        raise ChainError(
            "formal config does not satisfy the frozen method/extension contract: "
            + _canonical_json(mismatches)
        )
    return checks


def _validate_exact_eval_blob(
    blob: Mapping[str, Any], *, seed: int, label: str, identity_profile: str,
) -> dict[tuple[str, str, int, int, int], dict[str, Any]]:
    """Authenticate one exact singleton-seed LIBERO result and its RNG work."""
    from loom.eval import EpisodeResult, EvalProtocol  # noqa: PLC0415
    from loom.eval.runner import iter_work  # noqa: PLC0415

    _validate_eval_method_identity(
        blob, label=label, identity_profile=identity_profile,
    )
    try:
        protocol = EvalProtocol.from_dict(dict(blob.get("protocol", {})))
    except (TypeError, ValueError) as exc:
        raise ChainError(f"{label} has an invalid protocol") from exc
    expected_protocol = {
        "bench": "libero",
        "seeds": (seed,),
        "suites": tuple(SUITE_FLOORS_PERCENT),
        "n_tasks": 10,
        "episodes_per_task": 10,
        "max_steps": 512,
    }
    actual_protocol = {
        "bench": protocol.bench,
        "seeds": protocol.seeds,
        "suites": protocol.suites,
        "n_tasks": protocol.n_tasks,
        "episodes_per_task": protocol.episodes_per_task,
        "max_steps": protocol.max_steps,
    }
    if actual_protocol != expected_protocol or protocol.total_episodes != 400:
        raise ChainError(f"{label} protocol differs from the exact 400-episode seed")
    summary = blob.get("summary")
    if not isinstance(summary, Mapping) or (
        summary.get("complete") is not True
        or summary.get("n_episodes") != 400
        or summary.get("n_expected") != 400
        or summary.get("n_errors") != 0
    ):
        raise ChainError(f"{label} is incomplete or contains evaluation errors")
    rows = blob.get("episodes")
    if not isinstance(rows, list) or len(rows) != 400:
        raise ChainError(f"{label} must contain exactly 400 episode rows")
    expected_work = {item.key(): item for item in iter_work(protocol)}
    actual: dict[tuple[str, str, int, int, int], dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ChainError(f"{label} episode {index} is not an object")
        try:
            record = EpisodeResult.from_dict(dict(row))
        except (TypeError, ValueError) as exc:
            raise ChainError(f"{label} episode {index} is malformed") from exc
        key = record.key()
        if key in actual or key not in expected_work:
            raise ChainError(f"{label} has duplicate or unexpected episode key {key}")
        work = expected_work[key]
        policy_seed = (row.get("extra") or {}).get("policy_seed")
        if record.env_seed != work.env_seed or policy_seed != work.policy_seed:
            raise ChainError(f"{label} RNG identity mismatch for episode key {key}")
        if not isinstance(row.get("success"), bool) or record.error is not None:
            raise ChainError(f"{label} has invalid outcome/error for episode key {key}")
        actual[key] = dict(row)
    if set(actual) != set(expected_work):
        raise ChainError(f"{label} episode keys are not the exact protocol")
    return actual


def _validate_eval_method_identity(
    blob: Mapping[str, Any], *, label: str,
    identity_profile: str,
    checkpoint_step: int | None = None,
    checkpoint_path: str | None = None,
) -> None:
    meta = blob.get("meta")
    if not isinstance(meta, Mapping):
        raise ChainError(f"{label} omitted evaluation metadata")
    identity = meta.get("eval_identity")
    policy = meta.get("policy")
    if not isinstance(identity, Mapping) or not isinstance(policy, Mapping):
        raise ChainError(f"{label} omitted evaluation identity/policy provenance")
    if identity_profile == "historical_baseline":
        expected_policy_kw = BASELINE_POLICY_KW
    elif identity_profile == "current_candidate":
        expected_policy_kw = CANDIDATE_POLICY_KW
    else:
        raise ChainError(f"unknown evaluation identity profile {identity_profile!r}")
    expected_identity = {
        "version": 1,
        "backend": {"requested": "libero", "resolved": "libero"},
        "policy_kw": expected_policy_kw,
        "policy_source": "checkpoint_factory",
        "policy_seed_scheme": "sha256(work-item)-v1",
    }
    checks = {
        "meta_bench": meta.get("bench") == "libero",
        "meta_backend": meta.get("backend") == "libero",
        "meta_policy_seed_scheme": (
            meta.get("policy_seed_scheme") == "sha256(work-item)-v1"
        ),
        **{
            f"identity_{key}": identity.get(key) == value
            for key, value in expected_identity.items()
        },
        "checkpoint_identity": (
            meta.get("ckpt") == identity.get("checkpoint") == policy.get("ckpt")
        ),
        "policy_name": policy.get("policy") == "LoomPolicy",
        "policy_real": policy.get("is_stub") is False,
        "policy_embodiment": policy.get("embodiment") == "libero_franka",
        "gripper_dwell": policy.get("gripper_dwell") == 1,
        "decoder_samples": policy.get("decoder_samples") == 1,
        "duration_normalize_segments": (
            policy.get("duration_normalize_segments") is False
        ),
    }
    if checkpoint_step is not None:
        checks["checkpoint_step"] = policy.get("ckpt_global_step") == checkpoint_step
    if checkpoint_path is not None:
        checks["checkpoint_path"] = meta.get("ckpt") == checkpoint_path
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ChainError(f"{label} method identity mismatch: {failed}")


def _authenticate_baseline(root: Path) -> dict[str, Any]:
    root = root.resolve()
    files: dict[str, Any] = {}
    all_keys: set[tuple[str, str, int, int, int]] = set()
    successes = 0
    for seed in SEEDS:
        path = root / f"seed{seed}" / "results.json"
        if not path.is_file():
            raise ChainError(f"missing canonical baseline seed {seed}: {path}")
        digest = sha256_file(path)
        if digest != BASELINE_RESULT_SHA256[seed]:
            raise ChainError(f"canonical baseline seed {seed} SHA-256 mismatch")
        try:
            blob = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ChainError(f"canonical baseline seed {seed} is invalid JSON") from exc
        if not isinstance(blob, Mapping):
            raise ChainError(f"canonical baseline seed {seed} is not an object")
        rows = _validate_exact_eval_blob(
            blob, seed=seed, label=f"baseline seed {seed}",
            identity_profile="historical_baseline",
        )
        _validate_eval_method_identity(
            blob, label=f"baseline seed {seed}",
            identity_profile="historical_baseline",
            checkpoint_step=BASELINE_CHECKPOINT_STEP,
        )
        seed_successes = sum(bool(row["success"]) for row in rows.values())
        policy = (blob.get("meta") or {}).get("policy") or {}
        if (
            seed_successes != BASELINE_SUCCESS_PER_SEED
            or policy.get("is_stub") is not False
            or policy.get("ckpt_global_step") != BASELINE_CHECKPOINT_STEP
        ):
            raise ChainError(f"canonical baseline seed {seed} outcome/policy mismatch")
        if all_keys.intersection(rows):
            raise ChainError("canonical baseline singleton seeds overlap")
        all_keys.update(rows)
        successes += seed_successes
        files[str(seed)] = {
            "path": str(path),
            "sha256": digest,
            "bytes": path.stat().st_size,
            "episodes": len(rows),
            "successes": seed_successes,
        }
    if len(all_keys) != EXPECTED_EPISODES_TOTAL or successes != BASELINE_SUCCESS_TOTAL:
        raise ChainError("canonical baseline union is not exact 1,200/447")
    return {
        "kind": "r0a_deploy_seeded1200_v2_exact_baseline",
        "root": str(root),
        "files": files,
        "episodes": EXPECTED_EPISODES_TOTAL,
        "successes": BASELINE_SUCCESS_TOTAL,
        "success_rate_percent": 100.0 * BASELINE_SUCCESS_TOTAL / EXPECTED_EPISODES_TOTAL,
        "checkpoint_step": BASELINE_CHECKPOINT_STEP,
    }


def build_plan(
    *,
    config_path: Path,
    run_dir: Path,
    control_dir: Path,
    artifact_root: Path,
    group: str,
    baseline_root: Path = CANONICAL_BASELINE_ROOT,
    project: str = PROJECT,
    require_online: bool = True,
    step32: int = STEP_32K,
    step40: int = STEP_40K,
    initial_links: int = INITIAL_LINKS,
) -> dict[str, Any]:
    if not config_path.is_file():
        raise ChainError(f"missing formal config: {config_path}")
    if config_path.resolve() != CANONICAL_FORMAL_CONFIG:
        raise ChainError(
            "formal config must be the canonical path "
            f"{CANONICAL_FORMAL_CONFIG}, got {config_path.resolve()}"
        )
    if sha256_file(config_path) != CANONICAL_FORMAL_CONFIG_SHA256:
        raise ChainError("canonical formal config raw SHA-256 mismatch")
    if not group or re.fullmatch(r"[A-Za-z0-9_.-]+", group) is None:
        raise ChainError("W&B group must contain only letters, digits, '.', '_' or '-'")
    if step32 <= 0 or step40 <= step32:
        raise ChainError("step32 must be positive and step40 must be greater")
    if initial_links <= 0:
        raise ChainError("initial_links must be positive")
    if not require_online:
        raise ChainError("the formal chain requires online W&B; offline is forbidden")
    if not LIBERO_EVAL_PYTHON.is_file():
        raise ChainError(f"missing pinned LIBERO Python: {LIBERO_EVAL_PYTHON}")
    baseline_root = baseline_root.resolve()
    _require_isolated_paths(run_dir, control_dir, artifact_root)
    for field, path in (
        ("config", config_path), ("run_dir", run_dir),
        ("control_dir", control_dir), ("artifact_root", artifact_root),
        ("baseline_root", baseline_root),
    ):
        if "," in str(path) or "\n" in str(path):
            raise ChainError(f"{field} cannot contain comma/newline in Slurm exports")
    cfg, resolved_hash = _load_resolved_config(config_path)
    if resolved_hash != CANONICAL_FORMAL_RESOLVED_HASH:
        raise ChainError("canonical formal resolved-config hash mismatch")
    selected = validate_formal_config(
        cfg, project=project, step32=step32, step40=step40,
    )
    run_name = str(cfg["run"].get("name") or "").strip()
    if not run_name:
        raise ChainError("resolved config run.name must be non-empty")
    baseline = _authenticate_baseline(baseline_root)

    paths = {
        "gate32": str(control_dir / "gate_32000.json"),
        "endpoint": str(control_dir / "terminal_endpoint.json"),
        "jobs": str(control_dir / "jobs.json"),
        "checkpoint": str(artifact_root / "checkpoint" / "ckpt.pt"),
        "checkpoint_report": str(control_dir / "checkpoint_verification.json"),
        "checkpoint_receipt": str(control_dir / "checkpoint_receipt.json"),
        "merged_results": str(artifact_root / "eval" / "merged" / "results.json"),
        "merged_table": str(artifact_root / "eval" / "merged" / "table.md"),
        "merged_receipt": str(control_dir / "merged_eval_receipt.json"),
    }
    paths["eval"] = {
        str(seed): {
            "out_dir": str(artifact_root / "eval" / f"seed_{seed}"),
            "receipt": str(control_dir / f"eval_seed_{seed}_receipt.json"),
        }
        for seed in SEEDS
    }
    return {
        "format_version": FORMAT_VERSION,
        "kind": KIND,
        "eligibility": "one_direct_formal_training_lineage_no_smoke",
        "method": {
            "fresh_loom_modules": True,
            "frozen_pretrained_siglip": True,
            "dual_action_mode": DUAL_ACTION_MODE,
            "checkpoint_selection_uses_eval": False,
            "direct_formal_gate": DirectFormalGate().as_dict(),
        },
        "orchestration_source_closure": _source_closure(),
        "config": {
            "path": str(config_path),
            "raw_sha256": sha256_file(config_path),
            "resolved_config_hash": resolved_hash,
            "validated_fields": selected,
        },
        "lineage": {
            "run_name": run_name,
            "run_dir": str(run_dir),
            "control_dir": str(control_dir),
            "artifact_root": str(artifact_root),
        },
        "steps": {
            "schedule_horizon": step32,
            "initial_stop": step32,
            "hard_cap": step40,
            "initial_links": initial_links,
            "extension_links": 1,
        },
        "wandb": {
            "project": project,
            "group": group,
            "training_run_id": uuid.uuid4().hex[:16],
            "stage_run_ids": {
                stage: uuid.uuid4().hex[:16]
                for stage in (
                    "consolidate", "eval-seed-0", "eval-seed-1",
                    "eval-seed-2", "eval-summary",
                )
            },
            "training_job_type": "formal-train",
            "require_online": bool(require_online),
            "resume_first_link": "never",
            "resume_later_links": "must",
            "artifact_policy": "upload_small_receipts_and_eval_results_not_checkpoint_bytes",
        },
        "evaluation": {
            "seeds": list(SEEDS),
            "parallel_singleton_seed_jobs": True,
            "suites": [
                "libero_spatial", "libero_object", "libero_goal", "libero_long",
            ],
            "tasks_per_suite": 10,
            "episodes_per_task": 10,
            "max_steps": 512,
            "gripper_dwell": 1,
            "decoder_samples": 1,
            "duration_normalize_segments": False,
            "episodes_per_seed": EXPECTED_EPISODES_PER_SEED,
            "total_episodes": EXPECTED_EPISODES_TOTAL,
            "checkpoint_once_then_fixed": True,
            "identity_normalization": EVAL_IDENTITY_NORMALIZATION,
            "runtime": {
                "python": str(LIBERO_EVAL_PYTHON),
                "result_store_resume": True,
                "workers": 8,
                "backend_requested": "libero",
                "backend_resolved_required": "libero",
            },
        },
        "baseline_comparison": {
            "baseline": baseline,
            "pairing_key": ["bench", "suite", "task_id", "episode", "seed"],
            "pairing_requires_equal_env_seed_and_policy_seed": True,
            "task_reduction": (
                "paired_episode_success_delta_then_equal_30_episode_mean_within_"
                "task_then_equal_mean_across_40_tasks"
            ),
            "bootstrap": {
                "kind": "fixed_suite_stratified_task_resample_matrix_v1",
                "samples": BOOTSTRAP_SAMPLES,
                "seed": BOOTSTRAP_SEED,
                "suite_order": sorted(SUITE_FLOORS_PERCENT),
                "tasks_per_suite_per_replicate": 10,
                "confidence": BOOTSTRAP_CONFIDENCE,
                "lower_quantile": 0.025,
                "upper_quantile": 0.975,
                "lower_interpolation": "lower",
                "upper_interpolation": "higher",
                "matrix_sha256": BOOTSTRAP_MATRIX_SHA256,
            },
            "thresholds": dict(cfg["evaluation_gate"]),
            "scientific_failure_still_publishes": True,
        },
        "failure_policy": {
            "dependencies": "afterok_only",
            "gate32_converged": "select_step_32000",
            "gate32_moving": "extend_until_first_fixed_boundary_pass_or_40000",
            "gate32_or_extension_abort": "terminate_without_evaluation",
            "hard_cap_without_convergence": "terminate_without_evaluation",
            "invalid_gate_or_integrity_failure": "terminate_without_evaluation",
            "eval_scientific_failure": "merge_and_report",
            "nontraining_requeue": "same_job_id_stage_local_authenticated_recovery",
            "partial_consolidation": "verify_and_adopt_atomic_checkpoint_then_receipt",
            "partial_evaluation": "atomic_episode_store_resume_exact_identity",
            "partial_merge": "recompute_and_adopt_only_exact_bytes",
            "wandb_transient_failure": (
                f"same_run_id_bounded_{WANDB_PUBLISH_ATTEMPTS}_attempt_retry"
            ),
            "persistent_execution_or_wandb_failure": (
                "fail_stage_and_afterok_descendants_do_not_run"
            ),
        },
        "paths": paths,
    }


def _plan_stage_specs(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    n_links = int(plan["steps"]["initial_links"])
    specs: list[dict[str, Any]] = []
    previous: list[str] = []
    for index in range(1, n_links + 1):
        name = f"train_{index:02d}"
        specs.append({
            "name": name,
            "sbatch": "scripts/r0_e2e_formal_train.sbatch",
            "depends_on": list(previous),
        })
        previous = [name]
    specs.append({
        "name": "gate32", "sbatch": "scripts/r0_e2e_formal_control.sbatch",
        "depends_on": list(previous),
    })
    specs.append({
        "name": "extension", "sbatch": "scripts/r0_e2e_formal_train.sbatch",
        "depends_on": ["gate32"],
    })
    specs.append({
        "name": "gatefinal", "sbatch": "scripts/r0_e2e_formal_control.sbatch",
        "depends_on": ["extension"],
    })
    specs.append({
        "name": "consolidate", "sbatch": "scripts/r0_e2e_formal_consolidate.sbatch",
        "depends_on": ["gatefinal"],
    })
    for seed in SEEDS:
        specs.append({
            "name": f"eval_seed{seed}",
            "sbatch": "scripts/r0_e2e_formal_eval_seed.sbatch",
            "depends_on": ["consolidate"],
        })
    specs.append({
        "name": "merge", "sbatch": "scripts/r0_e2e_formal_control.sbatch",
        "depends_on": [f"eval_seed{seed}" for seed in SEEDS],
    })
    return specs


def _job_label(group: str, stage: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_-]", "_", f"r0e2e_{group}_{stage}")
    return label[:120]


def _sbatch_command(
    *,
    spec: Mapping[str, Any],
    plan_path: Path,
    plan_sha256: str,
    dependency_ids: Sequence[str],
    group: str,
) -> list[str]:
    command = [
        "sbatch", "--parsable", "--hold", "--kill-on-invalid-dep=yes",
        f"--job-name={_job_label(group, str(spec['name']))}",
    ]
    if dependency_ids:
        command.append("--dependency=afterok:" + ":".join(dependency_ids))
    exported = ",".join((
        "ALL",
        f"FORMAL_PLAN={plan_path}",
        f"FORMAL_PLAN_SHA256={plan_sha256}",
        f"FORMAL_STAGE={spec['name']}",
    ))
    command.extend((f"--export={exported}", str(ROOT / str(spec["sbatch"]))))
    return command


def _parse_job_id(stdout: str) -> str:
    value = stdout.strip().split(";", 1)[0]
    if re.fullmatch(r"[0-9]+(?:_[0-9]+)?", value) is None:
        raise ChainError(f"sbatch returned an invalid job id: {stdout!r}")
    return value


def submit_plan(
    plan: Mapping[str, Any],
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    _assert_plan_inputs_unchanged(plan)
    lineage = plan["lineage"]
    run_dir = Path(lineage["run_dir"])
    control_dir = Path(lineage["control_dir"])
    artifact_root = Path(lineage["artifact_root"])
    for path, label in (
        (run_dir, "run_dir"), (control_dir, "control_dir"),
        (artifact_root, "artifact_root"),
    ):
        if path.exists():
            raise ChainError(f"refusing non-fresh {label}: {path}")
    control_dir.mkdir(parents=True, exist_ok=False)
    plan_path = control_dir / "plan.json"
    exclusive_json_write(plan_path, plan)
    plan_sha = sha256_file(plan_path)

    specs = _plan_stage_specs(plan)
    job_ids: dict[str, str] = {}
    commands: dict[str, list[str]] = {}
    submitted: list[str] = []
    try:
        for spec in specs:
            dependencies = [job_ids[name] for name in spec["depends_on"]]
            command = _sbatch_command(
                spec=spec, plan_path=plan_path, plan_sha256=plan_sha,
                dependency_ids=dependencies, group=plan["wandb"]["group"],
            )
            completed = run(
                command, cwd=ROOT, check=True, text=True, capture_output=True,
            )
            job_id = _parse_job_id(completed.stdout)
            job_ids[str(spec["name"])] = job_id
            commands[str(spec["name"])] = command
            submitted.append(job_id)
        receipt = {
            "format_version": FORMAT_VERSION,
            "kind": "r0_e2e_formal_slurm_submission",
            "plan_path": str(plan_path),
            "plan_sha256": plan_sha,
            "jobs": job_ids,
            "commands": commands,
            "released": False,
        }
        jobs_path = Path(plan["paths"]["jobs"])
        exclusive_json_write(jobs_path, receipt)
        run(["scontrol", "release", ",".join(submitted)], cwd=ROOT, check=True, text=True,
            capture_output=True)
        # Publication remains immutable; a separate marker closes the release.
        exclusive_json_write(control_dir / "released.json", {
            "format_version": FORMAT_VERSION,
            "plan_sha256": plan_sha,
            "job_ids": submitted,
            "released": True,
        })
        return {**receipt, "released": True}
    except Exception:
        if submitted:
            run(["scancel", *submitted], cwd=ROOT, check=False, text=True,
                capture_output=True)
        raise


def load_plan(path: str | Path, expected_sha256: str | None = None) -> dict[str, Any]:
    plan_path = Path(path).resolve()
    if expected_sha256 is not None and sha256_file(plan_path) != expected_sha256:
        raise ChainError("formal plan SHA-256 mismatch")
    plan = json.loads(plan_path.read_text())
    if plan.get("format_version") != FORMAT_VERSION or plan.get("kind") != KIND:
        raise ChainError("unsupported formal plan identity")
    _assert_plan_inputs_unchanged(plan)
    return plan


def _assert_plan_inputs_unchanged(plan: Mapping[str, Any]) -> None:
    if plan.get("orchestration_source_closure") != _source_closure():
        raise ChainError("formal orchestration source closure changed after submission")
    config_path = Path(plan["config"]["path"])
    if config_path.resolve() != CANONICAL_FORMAL_CONFIG:
        raise ChainError("formal plan config is not the canonical path")
    raw_config_sha = sha256_file(config_path)
    if (
        raw_config_sha != CANONICAL_FORMAL_CONFIG_SHA256
        or raw_config_sha != plan["config"]["raw_sha256"]
    ):
        raise ChainError("formal config raw SHA-256 changed after submission")
    cfg, resolved_hash = _load_resolved_config(config_path)
    if (
        resolved_hash != CANONICAL_FORMAL_RESOLVED_HASH
        or resolved_hash != plan["config"]["resolved_config_hash"]
    ):
        raise ChainError("resolved formal config identity changed after submission")
    expected_runtime = {
        "python": str(LIBERO_EVAL_PYTHON),
        "result_store_resume": True,
        "workers": 8,
        "backend_requested": "libero",
        "backend_resolved_required": "libero",
    }
    if plan.get("evaluation", {}).get("runtime") != expected_runtime:
        raise ChainError("formal plan LIBERO runtime differs from the pinned path")
    if plan.get("evaluation", {}).get("identity_normalization") != EVAL_IDENTITY_NORMALIZATION:
        raise ChainError("formal plan evaluation identity normalization changed")
    if not LIBERO_EVAL_PYTHON.is_file():
        raise ChainError(f"missing pinned LIBERO Python: {LIBERO_EVAL_PYTHON}")
    validate_formal_config(
        cfg,
        project=plan["wandb"]["project"],
        step32=int(plan["steps"]["schedule_horizon"]),
        step40=int(plan["steps"]["hard_cap"]),
    )
    baseline = plan.get("baseline_comparison", {}).get("baseline")
    if not isinstance(baseline, Mapping):
        raise ChainError("formal plan omitted the canonical baseline receipt")
    if _authenticate_baseline(Path(str(baseline.get("root")))) != dict(baseline):
        raise ChainError("canonical baseline changed after plan construction")


def _required_plan_from_environment() -> tuple[dict[str, Any], str]:
    path = os.environ.get("FORMAL_PLAN")
    digest = os.environ.get("FORMAL_PLAN_SHA256")
    stage = os.environ.get("FORMAL_STAGE")
    if not path or not digest or not stage:
        raise ChainError("FORMAL_PLAN, FORMAL_PLAN_SHA256, and FORMAL_STAGE are required")
    return load_plan(path, digest), stage


def _latest_step(run_dir: Path) -> int:
    latest_path = run_dir / "LATEST"
    try:
        latest = int(latest_path.read_text().strip())
    except (OSError, ValueError) as exc:
        raise ChainError(f"missing or invalid LATEST under {run_dir}") from exc
    if latest <= 0:
        raise ChainError(f"LATEST={latest} is not a positive completed-update count")
    return latest


def _checkpoint_shards(
    run_dir: Path,
    step: int,
    world_size: int = WORLD_SIZE,
    *,
    require_latest: bool = True,
) -> list[Path]:
    if require_latest:
        latest = _latest_step(run_dir)
        if latest != step:
            raise ChainError(f"LATEST={latest}, expected exact endpoint {step}")
    by_rank: dict[int, Path] = {}
    for path in run_dir.glob(f"ckpt_{step:09d}_rank*.pt"):
        match = re.search(r"_rank([0-9]+)\.pt$", path.name)
        if match is None or path.stat().st_size <= 0:
            raise ChainError(f"invalid checkpoint shard: {path}")
        rank = int(match.group(1))
        if rank in by_rank:
            raise ChainError(f"duplicate checkpoint shard for rank {rank}")
        by_rank[rank] = path
    ranks = sorted(by_rank)
    if ranks != list(range(world_size)):
        raise ChainError(f"checkpoint ranks are {ranks}, expected 0..{world_size - 1}")
    return [by_rank[rank] for rank in ranks]


def _checkpoint_shard_receipt(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    return {
        path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in paths
    }


_DIRECT_RECEIPT_CORE_KEYS = (
    "format", "status", "reason", "current_step", "decision_step",
    "next_check_step", "gate", "input", "evaluations",
)


def _direct_boundary_path(run_dir: Path, step: int) -> Path:
    return run_dir / f"direct_formal_{step:09d}.json"


def _expected_fresh_lineage(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": "loom-fresh-training-lineage-v1",
        "fresh_start_required": True,
        "config_hash": plan["config"]["resolved_config_hash"],
        "seed": 0,
        "world_size": WORLD_SIZE,
        "act_decode_from": DUAL_ACTION_MODE,
        "schedule_horizon": int(plan["steps"]["schedule_horizon"]),
        "max_updates": int(plan["steps"]["hard_cap"]),
    }


def _read_direct_boundary_receipt(
    plan: Mapping[str, Any], step: int,
) -> tuple[dict[str, Any], Path]:
    run_dir = Path(plan["lineage"]["run_dir"])
    path = _direct_boundary_path(run_dir, step)
    try:
        receipt = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ChainError(f"missing or invalid in-loop direct receipt: {path}") from exc
    if not isinstance(receipt, Mapping):
        raise ChainError(f"in-loop direct receipt is not an object: {path}")
    receipt = dict(receipt)
    _direct_core(receipt)
    expected_gate = DirectFormalGate().as_dict()
    checks = {
        "format": receipt.get("format") == DIRECT_FORMAL_FORMAT,
        "current_step": receipt.get("current_step") == step,
        "gate": receipt.get("gate") == expected_gate,
        "config_hash": receipt.get("config_hash") == plan["config"]["resolved_config_hash"],
        "fresh_lineage": receipt.get("fresh_lineage") == _expected_fresh_lineage(plan),
        "status": receipt.get("status") in DIRECT_STATUS_CODES,
        "input": receipt.get("input") == {
            "rows": step,
            "minimum_step": 1,
            "maximum_step": step,
        },
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ChainError(
            f"in-loop direct receipt failed authentication at step {step}: {failed}"
        )
    status = str(receipt["status"])
    if status == "PASS" and receipt.get("decision_step") != step:
        raise ChainError("in-loop PASS did not select its exact current boundary")
    if status == "MOVING" and receipt.get("decision_step") is not None:
        raise ChainError("in-loop MOVING receipt unexpectedly selected a checkpoint")
    evaluations = receipt.get("evaluations")
    if not isinstance(evaluations, list) or not evaluations:
        raise ChainError("in-loop direct receipt omitted candidate evaluations")
    final = evaluations[-1]
    if not isinstance(final, Mapping) or final.get("step") != step:
        raise ChainError("in-loop direct receipt final evaluation is not current step")
    if final.get("status") != status and not (
        status == "ABORT" and receipt.get("reason") == "max_updates_without_convergence"
        and final.get("status") == "MOVING"
    ):
        raise ChainError("in-loop direct status differs from its final evaluation")
    return receipt, path


def _read_receipt(path: str | Path, *, kind: str) -> dict[str, Any]:
    receipt = json.loads(Path(path).read_text())
    if receipt.get("format_version") != FORMAT_VERSION or receipt.get("kind") != kind:
        raise ChainError(f"invalid {kind} receipt at {path}")
    return receipt


def _stage_train(plan: Mapping[str, Any], stage: str) -> int:
    try:
        restart_count = int(os.environ.get("SLURM_RESTART_COUNT", "0"))
    except ValueError as exc:
        raise ChainError("SLURM_RESTART_COUNT must be an integer") from exc
    policy = train_stage_policy(plan, stage, restart_count=restart_count)
    if policy["skip"]:
        print("[formal] 32k converged; extension is a no-mutation no-op", flush=True)
        return 0
    fresh = bool(policy["fresh"])
    require_target = bool(policy["require_target"])
    stop_at = int(policy["stop_at"])
    resume = str(policy["resume"])

    run_dir = Path(plan["lineage"]["run_dir"])
    if fresh:
        if run_dir.exists():
            raise ChainError(f"fresh formal run_dir already exists: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=False)
        # Copy the immutable receipt, not the mutable submission job table.
        source_plan = Path(os.environ["FORMAL_PLAN"])
        target_plan = run_dir / "formal_plan.json"
        _exclusive_text_write(target_plan, source_plan.read_text())
        if sha256_file(target_plan) != sha256_file(source_plan):
            raise ChainError("run-local formal plan copy does not match its source")
    else:
        if not run_dir.is_dir() or not (run_dir / "wandb_id").is_file():
            raise ChainError("continuation lacks the established run_dir/W&B identity")
    if (run_dir / "STOP").exists():
        raise ChainError("STOP exists; formal links never remove or ignore it")

    # A requeued extension must not step past an already selected boundary.
    # The in-loop gate publishes before returning, so this is an idempotent
    # terminal replay rather than a new scientific decision.
    if stage == "extension" and (run_dir / "LATEST").is_file():
        prior_step = _latest_step(run_dir)
        prior_path = _direct_boundary_path(run_dir, prior_step)
        if prior_path.is_file():
            prior, _ = _read_direct_boundary_receipt(plan, prior_step)
            if prior["status"] == "PASS":
                _checkpoint_shards(run_dir, prior_step)
                print(
                    f"[formal] extension already selected step {prior_step}; "
                    "no further optimizer step",
                    flush=True,
                )
                return 0
            if prior["status"] in {"ABORT", "INVALID"}:
                raise ChainError(
                    f"extension already terminated {prior['status']} at {prior_step}"
                )

    from loom.train import wandb_util  # noqa: PLC0415

    expected_wandb_id = str(plan["wandb"]["training_run_id"])
    if fresh:
        os.environ["WANDB_RUN_ID"] = expected_wandb_id
    elif (run_dir / "wandb_id").read_text().strip() != expected_wandb_id:
        raise ChainError("continuation W&B run id differs from the immutable plan")
    wandb_id = wandb_util.stable_run_id(run_dir)
    if wandb_id != expected_wandb_id:
        raise ChainError("materialized W&B run id differs from the immutable plan")
    env = dict(os.environ)
    env.update({
        "FORMAL_CONFIG": plan["config"]["path"],
        "FORMAL_RUN_DIR": str(run_dir),
        "FORMAL_STOP_AT": str(stop_at),
        "WANDB_RUN_ID": wandb_id,
        "WANDB_RESUME": resume,
        "WANDB_DIR": str(run_dir),
        "WANDB_MODE": "online",
        "WANDB_RUN_GROUP": plan["wandb"]["group"],
        "WANDB_JOB_TYPE": plan["wandb"]["training_job_type"],
        "WANDB_TAGS": "formal,r0,e2e-scratch,dual-action",
        "LOOM_WANDB_MODE": "online",
        "LOOM_WANDB_PROJECT": plan["wandb"]["project"],
        "LOOM_WANDB_GROUP": plan["wandb"]["group"],
        "LOOM_WANDB_JOB_TYPE": plan["wandb"]["training_job_type"],
        "LOOM_WANDB_TAGS": "formal,r0,e2e-scratch,dual-action",
        "LOOM_WANDB_REQUIRE_ONLINE": (
            "1" if plan["wandb"]["require_online"] else "0"
        ),
        "LOOM_WANDB_RESUME": resume,
        "LOOM_RESTART_COUNT": str(restart_count),
        "LOOM_TIME_BUDGET_S": str(4 * 3600 - 600),
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "8",
    })
    inner = (
        'export RANK="$SLURM_PROCID" WORLD_SIZE="$SLURM_NTASKS" '
        'LOCAL_RANK="$SLURM_LOCALID"; '
        'exec python3 scripts/r0_e2e_formal_train_entry.py --config "$FORMAL_CONFIG" '
        '--run_dir "$FORMAL_RUN_DIR" --stop_at "$FORMAL_STOP_AT"'
    )
    subprocess.run(
        ["srun", "--kill-on-bad-exit=1", "bash", "-c", inner],
        cwd=ROOT, env=env, check=True,
    )
    if (run_dir / "STOP").exists():
        raise ChainError("formal link created/observed STOP; refusing success")
    latest = int((run_dir / "LATEST").read_text().strip())
    if latest <= 0 or latest > stop_at:
        raise ChainError(f"post-link LATEST={latest} is outside 1..{stop_at}")
    if require_target:
        if stage == "extension":
            initial = int(plan["steps"]["initial_stop"])
            if not initial < latest <= stop_at:
                raise ChainError(
                    f"extension terminal step {latest} is outside ({initial}, {stop_at}]"
                )
            receipt, _ = _read_direct_boundary_receipt(plan, latest)
            if receipt.get("status") != "PASS" or receipt.get("decision_step") != latest:
                raise ChainError(
                    "extension exited without an exact first-PASS direct receipt"
                )
            _checkpoint_shards(run_dir, latest)
        else:
            _checkpoint_shards(run_dir, stop_at)
            receipt, _ = _read_direct_boundary_receipt(plan, stop_at)
            if receipt.get("status") not in {"PASS", "MOVING"}:
                raise ChainError(
                    f"initial horizon ended with terminal {receipt.get('status')!r}"
                )
    return 0


def train_stage_policy(
    plan: Mapping[str, Any], stage: str, *, restart_count: int,
) -> dict[str, Any]:
    if restart_count < 0:
        raise ChainError("SLURM_RESTART_COUNT must be non-negative")
    n_links = int(plan["steps"]["initial_links"])
    initial_match = re.fullmatch(r"train_([0-9]{2})", stage)
    if initial_match:
        index = int(initial_match.group(1))
        if not 1 <= index <= n_links:
            raise ChainError(f"unexpected initial train stage {stage!r}")
        fresh = index == 1 and restart_count == 0
        require_target = index == n_links
        stop_at = int(plan["steps"]["initial_stop"])
        resume = "never" if fresh else "must"
        skip = False
    elif stage == "extension":
        gate = _read_gate32(plan)
        if gate.get("action") == "select_step_32000":
            return {"skip": True}
        if gate.get("action") != "extend_to_step_40000":
            raise ChainError(f"invalid gate32 action: {gate.get('action')!r}")
        fresh = False
        require_target = True
        stop_at = int(plan["steps"]["hard_cap"])
        resume = "must"
        skip = False
    else:
        raise ChainError(f"train wrapper cannot execute stage {stage!r}")
    return {
        "skip": skip,
        "fresh": fresh,
        "require_target": require_target,
        "stop_at": stop_at,
        "resume": resume,
    }


def _direct_core(receipt: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return {key: receipt[key] for key in _DIRECT_RECEIPT_CORE_KEYS}
    except KeyError as exc:
        raise ChainError(f"direct receipt is missing {exc.args[0]!r}") from exc


def _run_convergence(
    plan: Mapping[str, Any], step: int,
) -> tuple[int, dict[str, Any], dict[str, Any]]:
    """Independently recompute and authenticate the in-loop boundary receipt."""
    run_dir = Path(plan["lineage"]["run_dir"])
    _checkpoint_shards(run_dir, step)
    in_loop, in_loop_path = _read_direct_boundary_receipt(plan, step)
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "direct_formal_convergence.py"),
            str(run_dir),
            "--current-step", str(step),
        ],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=False,
    )
    print(completed.stdout, end="", flush=True)
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr, flush=True)
        raise ChainError("direct-formal CLI emitted unexpected stderr")
    try:
        direct = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ChainError("direct-formal CLI did not emit one JSON receipt") from exc
    if not isinstance(direct, Mapping):
        raise ChainError("direct-formal CLI receipt is not an object")
    direct = dict(direct)
    try:
        expected_rc = receipt_exit_code(direct)
    except ValueError as exc:
        raise ChainError(str(exc)) from exc
    if completed.returncode != expected_rc:
        raise ChainError(
            f"direct-formal status/return-code mismatch: {direct.get('status')!r} "
            f"vs {completed.returncode}"
        )
    if direct.get("format") != DIRECT_FORMAL_FORMAT:
        raise ChainError("direct-formal CLI format mismatch")
    if direct.get("current_step") != step:
        raise ChainError("direct-formal CLI evaluated a different current step")
    if direct.get("gate") != DirectFormalGate().as_dict():
        raise ChainError("direct-formal CLI gate differs from frozen defaults")
    if _direct_core(direct) != _direct_core(in_loop):
        raise ChainError("independent direct receipt differs from in-loop receipt")

    metrics_path = run_dir / "metrics.jsonl"
    metrics_source = direct.get("metrics_source")
    if not isinstance(metrics_source, Mapping):
        raise ChainError("direct-formal CLI omitted metrics source identity")
    metrics_identity = {
        "path": str(metrics_path.resolve()),
        "bytes": metrics_path.stat().st_size,
        "sha256": sha256_file(metrics_path),
    }
    if dict(metrics_source) != metrics_identity:
        raise ChainError("direct-formal metrics source changed or mismatched")
    evidence = {
        "cli": str(ROOT / "scripts" / "direct_formal_convergence.py"),
        "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
        "in_loop_receipt": str(in_loop_path),
        "in_loop_receipt_sha256": sha256_file(in_loop_path),
        "metrics_source": metrics_identity,
    }
    return completed.returncode, direct, evidence


def _convergence_receipt(
    plan: Mapping[str, Any], *, step: int, returncode: int,
    direct: Mapping[str, Any], evidence: Mapping[str, Any], action: str,
) -> dict[str, Any]:
    run_dir = Path(plan["lineage"]["run_dir"])
    return {
        "format_version": FORMAT_VERSION,
        "kind": "r0_e2e_convergence_gate",
        "step": step,
        "returncode": returncode,
        "action": action,
        "direct_receipt": dict(direct),
        "recomputation": dict(evidence),
        "metrics_sha256": sha256_file(run_dir / "metrics.jsonl"),
        "run_config_sha256": sha256_file(run_dir / "config.json"),
        "plan_sha256": sha256_file(os.environ["FORMAL_PLAN"]),
    }


_CONVERGENCE_RECEIPT_KEYS = {
    "format_version", "kind", "step", "returncode", "action",
    "direct_receipt", "recomputation", "metrics_sha256",
    "run_config_sha256", "plan_sha256",
}
_RECOMPUTATION_KEYS = {
    "cli", "stdout_sha256", "in_loop_receipt",
    "in_loop_receipt_sha256", "metrics_source",
}


def _authenticate_convergence_evidence(
    plan: Mapping[str, Any],
    convergence: Mapping[str, Any],
    *,
    step: int,
    expected_action: str,
    allow_metrics_prefix: bool,
    label: str,
) -> Mapping[str, Any]:
    """Bind a controller receipt to code-authenticated in-loop evidence."""
    direct = convergence.get("direct_receipt")
    recomputation = convergence.get("recomputation")
    if not isinstance(direct, Mapping) or not isinstance(recomputation, Mapping):
        raise ChainError(f"{label} omitted direct/recomputation evidence")
    _direct_core(direct)
    try:
        direct_returncode = receipt_exit_code(direct)
    except ValueError as exc:
        raise ChainError(str(exc)) from exc

    run_dir = Path(plan["lineage"]["run_dir"])
    metrics_path = run_dir / "metrics.jsonl"
    metrics_source = direct.get("metrics_source")
    if not isinstance(metrics_source, Mapping) or set(metrics_source) != {
        "path", "bytes", "sha256",
    }:
        raise ChainError(f"{label} has invalid metrics source identity")
    prefix_bytes = metrics_source.get("bytes")
    if (
        not isinstance(prefix_bytes, int) or isinstance(prefix_bytes, bool)
        or prefix_bytes <= 0
    ):
        raise ChainError(f"{label} has invalid metrics byte count")
    with metrics_path.open("rb") as handle:
        prefix = handle.read(prefix_bytes)
    in_loop, in_loop_path = _read_direct_boundary_receipt(plan, step)
    plan_sha = sha256_file(Path(os.environ["FORMAL_PLAN"]))
    checks = {
        "convergence_keys": set(convergence) == _CONVERGENCE_RECEIPT_KEYS,
        "step": convergence.get("step") == step,
        "returncode": convergence.get("returncode") == direct_returncode,
        "action": convergence.get("action") == expected_action,
        "direct_format": direct.get("format") == DIRECT_FORMAL_FORMAT,
        "direct_step": direct.get("current_step") == step,
        "direct_gate": direct.get("gate") == DirectFormalGate().as_dict(),
        "direct_in_loop_core": _direct_core(direct) == _direct_core(in_loop),
        "metrics_path": metrics_source.get("path") == str(metrics_path.resolve()),
        "metrics_prefix_length": len(prefix) == prefix_bytes,
        "metrics_prefix_sha256": (
            hashlib.sha256(prefix).hexdigest() == metrics_source.get("sha256")
        ),
        "metrics_full_length": (
            allow_metrics_prefix or metrics_path.stat().st_size == prefix_bytes
        ),
        "metrics_receipt_sha256": (
            convergence.get("metrics_sha256") == metrics_source.get("sha256")
        ),
        "run_config_sha256": convergence.get("run_config_sha256") == sha256_file(
            run_dir / "config.json"
        ),
        "plan_sha256": convergence.get("plan_sha256") == plan_sha,
        "recomputation_keys": set(recomputation) == _RECOMPUTATION_KEYS,
        "recomputation_cli": (
            recomputation.get("cli")
            == str(ROOT / "scripts" / "direct_formal_convergence.py")
        ),
        "recomputation_stdout_sha256": (
            isinstance(recomputation.get("stdout_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", recomputation["stdout_sha256"])
            is not None
        ),
        "recomputation_in_loop_path": (
            recomputation.get("in_loop_receipt") == str(in_loop_path)
        ),
        "recomputation_in_loop_sha256": (
            recomputation.get("in_loop_receipt_sha256")
            == sha256_file(in_loop_path)
        ),
        "recomputation_metrics": recomputation.get("metrics_source") == dict(
            metrics_source
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ChainError(f"{label} failed authentication: {failed}")
    return direct


def _read_gate32(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Reauthenticate the immutable 32k controller receipt before branching."""
    receipt = _read_receipt(
        plan["paths"]["gate32"], kind="r0_e2e_convergence_gate",
    )
    step = int(plan["steps"]["initial_stop"])
    returncode = receipt.get("returncode")
    expected_action = (
        classify_gate32(returncode)
        if isinstance(returncode, int) and not isinstance(returncode, bool)
        else "invalid_stop_descendants"
    )
    _authenticate_convergence_evidence(
        plan, receipt, step=step, expected_action=expected_action,
        allow_metrics_prefix=True, label="32k gate receipt",
    )
    return receipt


def _stage_gate32(plan: Mapping[str, Any]) -> int:
    gate_path = Path(plan["paths"]["gate32"])
    if gate_path.exists():
        existing = _read_gate32(plan)
        action = existing.get("action")
        if action in {"select_step_32000", "extend_to_step_40000"}:
            return 0
        direct = existing.get("direct_receipt")
        status = direct.get("status") if isinstance(direct, Mapping) else None
        raise ChainError(
            f"existing 32k direct gate is terminal {status}, "
            f"action={action}; no evaluation"
        )
    step = int(plan["steps"]["initial_stop"])
    rc, direct, evidence = _run_convergence(plan, step)
    action = classify_gate32(rc)
    exclusive_json_write(
        plan["paths"]["gate32"],
        _convergence_receipt(
            plan, step=step, returncode=rc, direct=direct,
            evidence=evidence, action=action,
        ),
    )
    if action not in {"select_step_32000", "extend_to_step_40000"}:
        raise ChainError(
            f"32k direct gate is terminal {direct.get('status')}, rc={rc}; no evaluation"
        )
    return 0


def classify_gate32(returncode: int) -> str:
    if returncode == 0:
        return "select_step_32000"
    if returncode == 1:
        return "extend_to_step_40000"
    if returncode == 3:
        return "abort_no_evaluation"
    return "invalid_stop_descendants"


def classify_terminal(returncode: int) -> str:
    if returncode == 0:
        return "select_first_passing_checkpoint"
    if returncode == 3:
        return "abort_no_evaluation"
    return "invalid_stop_descendants"


def _stage_gatefinal(plan: Mapping[str, Any]) -> int:
    endpoint_path = Path(plan["paths"]["endpoint"])
    if endpoint_path.exists():
        existing = _read_receipt(
            endpoint_path, kind="r0_e2e_terminal_endpoint",
        )
        if existing.get("eligible_for_eval") is True:
            _authenticate_endpoint(plan)
            return 0
        terminal = _authenticate_terminal_failure_endpoint(plan)
        raise ChainError(
            f"existing terminal direct gate {terminal['status']}; no evaluation"
        )
    gate = _read_gate32(plan)
    if gate.get("action") == "select_step_32000":
        step = int(plan["steps"]["initial_stop"])
        rc, direct, evidence = _run_convergence(plan, step)
    elif gate.get("action") == "extend_to_step_40000":
        step = _latest_step(Path(plan["lineage"]["run_dir"]))
        initial = int(plan["steps"]["initial_stop"])
        hard_cap = int(plan["steps"]["hard_cap"])
        if not initial < step <= hard_cap:
            raise ChainError(
                f"extension endpoint {step} is outside ({initial}, {hard_cap}]"
            )
        rc, direct, evidence = _run_convergence(plan, step)
    else:
        raise ChainError(f"invalid gate32 action: {gate.get('action')!r}")
    action = classify_terminal(rc)
    convergence = _convergence_receipt(
        plan, step=step, returncode=rc, direct=direct,
        evidence=evidence, action=action,
    )
    if action != "select_first_passing_checkpoint":
        terminal_status = (
            "ABORT_NO_EVALUATION" if action == "abort_no_evaluation"
            else "INVALID_TERMINAL_GATE"
        )
        exclusive_json_write(plan["paths"]["endpoint"], {
            "format_version": FORMAT_VERSION,
            "kind": "r0_e2e_terminal_endpoint",
            "execution_validated": action == "abort_no_evaluation",
            "eligible_for_eval": False,
            "step": step,
            "status": terminal_status,
            "plan_sha256": sha256_file(os.environ["FORMAL_PLAN"]),
            "convergence": convergence,
            "checkpoint_selection_used_eval": False,
        })
        raise ChainError(f"terminal direct gate {direct.get('status')}; no evaluation")
    if direct.get("status") != "PASS" or direct.get("decision_step") != step:
        raise ChainError("terminal direct gate did not select its exact LATEST step")
    shards = _checkpoint_shards(Path(plan["lineage"]["run_dir"]), step)
    exclusive_json_write(plan["paths"]["endpoint"], {
        "format_version": FORMAT_VERSION,
        "kind": "r0_e2e_terminal_endpoint",
        "execution_validated": True,
        "eligible_for_eval": True,
        "step": step,
        "status": f"FIRST_PASS_{step}",
        "plan_sha256": sha256_file(os.environ["FORMAL_PLAN"]),
        "convergence": convergence,
        "checkpoint_shards": _checkpoint_shard_receipt(shards),
        "checkpoint_selection_used_eval": False,
    })
    return 0


def _authenticate_endpoint(
    plan: Mapping[str, Any],
) -> tuple[dict[str, Any], str, dict[str, dict[str, Any]]]:
    """Reauthenticate the PASS endpoint and every selected shard byte."""
    endpoint_path = Path(plan["paths"]["endpoint"])
    endpoint = _read_receipt(
        endpoint_path, kind="r0_e2e_terminal_endpoint",
    )
    convergence = endpoint.get("convergence")
    direct = convergence.get("direct_receipt") if isinstance(convergence, Mapping) else None
    step = endpoint.get("step")
    if not isinstance(step, int) or isinstance(step, bool):
        raise ChainError("terminal endpoint has an invalid step")
    _authenticate_convergence_evidence(
        plan, convergence, step=step,
        expected_action="select_first_passing_checkpoint",
        allow_metrics_prefix=False, label="PASS terminal convergence",
    )
    current_plan_sha = sha256_file(Path(os.environ["FORMAL_PLAN"]))
    checks = {
        "execution_validated": endpoint.get("execution_validated") is True,
        "eligible_for_eval": endpoint.get("eligible_for_eval") is True,
        "selection_without_eval": endpoint.get("checkpoint_selection_used_eval") is False,
        "status": endpoint.get("status") == f"FIRST_PASS_{step}",
        "plan_sha256": endpoint.get("plan_sha256") == current_plan_sha,
        "convergence_kind": (
            isinstance(convergence, Mapping)
            and convergence.get("kind") == "r0_e2e_convergence_gate"
        ),
        "convergence_action": (
            isinstance(convergence, Mapping)
            and convergence.get("action") == "select_first_passing_checkpoint"
        ),
        "convergence_returncode": (
            isinstance(convergence, Mapping) and convergence.get("returncode") == 0
        ),
        "convergence_plan": (
            isinstance(convergence, Mapping)
            and convergence.get("plan_sha256") == current_plan_sha
        ),
        "direct_pass": isinstance(direct, Mapping) and direct.get("status") == "PASS",
        "direct_step": (
            isinstance(direct, Mapping)
            and direct.get("current_step") == step
            and direct.get("decision_step") == step
        ),
        "direct_gate": (
            isinstance(direct, Mapping)
            and direct.get("gate") == DirectFormalGate().as_dict()
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ChainError(f"terminal endpoint failed authentication: {failed}")
    shards = _checkpoint_shards(Path(plan["lineage"]["run_dir"]), step)
    current = _checkpoint_shard_receipt(shards)
    if endpoint.get("checkpoint_shards") != current:
        raise ChainError("selected endpoint checkpoint shard bytes changed")
    return endpoint, sha256_file(endpoint_path), current


def _authenticate_terminal_failure_endpoint(
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate an immutable terminal ABORT/INVALID without changing it."""
    endpoint_path = Path(plan["paths"]["endpoint"])
    endpoint = _read_receipt(
        endpoint_path, kind="r0_e2e_terminal_endpoint",
    )
    convergence = endpoint.get("convergence")
    direct = convergence.get("direct_receipt") if isinstance(convergence, Mapping) else None
    if not isinstance(convergence, Mapping) or not isinstance(direct, Mapping):
        raise ChainError("terminal failure endpoint omitted convergence evidence")
    _direct_core(direct)
    step = endpoint.get("step")
    returncode = convergence.get("returncode")
    if (
        not isinstance(step, int) or isinstance(step, bool)
        or not isinstance(returncode, int) or isinstance(returncode, bool)
    ):
        raise ChainError("terminal failure endpoint has invalid step/returncode")
    try:
        direct_returncode = receipt_exit_code(direct)
    except ValueError as exc:
        raise ChainError(str(exc)) from exc
    action = classify_terminal(returncode)
    if action == "abort_no_evaluation":
        expected_status = "ABORT_NO_EVALUATION"
        expected_execution_validated = True
    elif action == "invalid_stop_descendants":
        expected_status = "INVALID_TERMINAL_GATE"
        expected_execution_validated = False
    else:
        raise ChainError("terminal failure endpoint unexpectedly contains PASS")
    _authenticate_convergence_evidence(
        plan, convergence, step=step, expected_action=action,
        allow_metrics_prefix=False, label="failure terminal convergence",
    )

    run_dir = Path(plan["lineage"]["run_dir"])
    plan_sha = sha256_file(Path(os.environ["FORMAL_PLAN"]))
    metrics_path = run_dir / "metrics.jsonl"
    metrics_identity = {
        "path": str(metrics_path.resolve()),
        "bytes": metrics_path.stat().st_size,
        "sha256": sha256_file(metrics_path),
    }
    recomputation = convergence.get("recomputation")
    in_loop_path = _direct_boundary_path(run_dir, step)
    expected_keys = {
        "format_version", "kind", "execution_validated", "eligible_for_eval",
        "step", "status", "plan_sha256", "convergence",
        "checkpoint_selection_used_eval",
    }
    checks = {
        "endpoint_keys": set(endpoint) == expected_keys,
        "execution_validated": (
            endpoint.get("execution_validated") is expected_execution_validated
        ),
        "eligible_for_eval": endpoint.get("eligible_for_eval") is False,
        "selection_without_eval": endpoint.get("checkpoint_selection_used_eval") is False,
        "status": endpoint.get("status") == expected_status,
        "endpoint_plan": endpoint.get("plan_sha256") == plan_sha,
        "convergence_kind": convergence.get("kind") == "r0_e2e_convergence_gate",
        "convergence_step": convergence.get("step") == step,
        "convergence_returncode": returncode == direct_returncode,
        "convergence_action": convergence.get("action") == action,
        "convergence_plan": convergence.get("plan_sha256") == plan_sha,
        "convergence_config": convergence.get("run_config_sha256") == sha256_file(
            run_dir / "config.json"
        ),
        "convergence_metrics": convergence.get("metrics_sha256") == metrics_identity["sha256"],
        "direct_format": direct.get("format") == DIRECT_FORMAL_FORMAT,
        "direct_step": direct.get("current_step") == step,
        "direct_gate": direct.get("gate") == DirectFormalGate().as_dict(),
        "direct_metrics": direct.get("metrics_source") == metrics_identity,
        "recomputation_shape": isinstance(recomputation, Mapping),
        "recomputation_cli": (
            isinstance(recomputation, Mapping)
            and recomputation.get("cli")
            == str(ROOT / "scripts" / "direct_formal_convergence.py")
        ),
        "recomputation_metrics": (
            isinstance(recomputation, Mapping)
            and recomputation.get("metrics_source") == metrics_identity
        ),
        "recomputation_in_loop_path": (
            isinstance(recomputation, Mapping)
            and recomputation.get("in_loop_receipt") == str(in_loop_path)
        ),
        "recomputation_in_loop_sha": (
            isinstance(recomputation, Mapping)
            and in_loop_path.is_file()
            and recomputation.get("in_loop_receipt_sha256")
            == sha256_file(in_loop_path)
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ChainError(
            f"terminal failure endpoint failed authentication: {failed}"
        )
    return endpoint


def _wandb_publish_once(
    plan: Mapping[str, Any], *, stage: str, path: Path, artifact_type: str,
    summary: Mapping[str, Any],
) -> None:
    try:
        import wandb  # noqa: PLC0415
    except ImportError as exc:
        if plan["wandb"]["require_online"]:
            raise ChainError("W&B publication is required but wandb is unavailable") from exc
        print("[formal-wandb] publication skipped: wandb unavailable", flush=True)
        return
    run_dir = Path(plan["lineage"]["control_dir"]) / "wandb" / stage
    run_dir.mkdir(parents=True, exist_ok=True)
    expected_run_id = str(plan["wandb"]["stage_run_ids"][stage])
    run_id_path = run_dir / "wandb_id"
    if run_id_path.exists():
        if run_id_path.read_text().strip() != expected_run_id:
            raise ChainError(f"W&B stage {stage} run id differs from immutable plan")
    else:
        _exclusive_text_write(run_id_path, expected_run_id + "\n")
    kwargs = {
        "project": plan["wandb"]["project"],
        "id": expected_run_id,
        "name": f"{plan['wandb']['group']}-{stage}",
        "group": plan["wandb"]["group"],
        "job_type": stage,
        "tags": ["formal", "r0", "e2e-scratch", "dual-action"],
        "resume": "allow",
        "mode": "online",
        "dir": str(run_dir),
        "config": {
            "formal_plan_sha256": sha256_file(os.environ["FORMAL_PLAN"]),
            "formal_stage": stage,
        },
        "settings": wandb.Settings(init_timeout=90),
    }
    try:
        run = wandb.init(**kwargs)
    except Exception as exc:  # noqa: BLE001
        if plan["wandb"]["require_online"]:
            raise ChainError(
                f"required online W&B publication init failed: {type(exc).__name__}: {exc}"
            ) from exc
        kwargs["mode"] = "offline"
        try:
            run = wandb.init(**kwargs)
        except Exception as offline_exc:  # noqa: BLE001
            print(
                "[formal-wandb] optional offline publication init failed: "
                f"{type(offline_exc).__name__}: {offline_exc}",
                flush=True,
            )
            return
    if plan["wandb"]["require_online"] and bool(getattr(run, "offline", False)):
        run.finish()
        raise ChainError("W&B publication returned offline while online is required")
    try:
        for key, value in summary.items():
            run.summary[key] = value
        artifact = wandb.Artifact(
            name=f"{plan['wandb']['group']}-{stage}",
            type=artifact_type,
            metadata={**dict(summary), "sha256": sha256_file(path)},
        )
        artifact.add_file(str(path), name=path.name)
        run.log_artifact(artifact)
    except Exception as exc:  # noqa: BLE001
        if plan["wandb"]["require_online"]:
            raise ChainError(
                f"required W&B artifact publication failed: {type(exc).__name__}: {exc}"
            ) from exc
        print(f"[formal-wandb] optional artifact publication failed: {exc}", flush=True)
    finally:
        try:
            run.finish()
        except Exception as exc:  # noqa: BLE001
            if plan["wandb"]["require_online"]:
                raise ChainError(
                    f"required W&B finish failed: {type(exc).__name__}: {exc}"
                ) from exc
            print(f"[formal-wandb] optional finish failed: {exc}", flush=True)


def _wandb_publish(
    plan: Mapping[str, Any], *, stage: str, path: Path, artifact_type: str,
    summary: Mapping[str, Any],
) -> None:
    """Publish with bounded retries under the same immutable W&B run id."""
    errors: list[str] = []
    for attempt in range(1, WANDB_PUBLISH_ATTEMPTS + 1):
        try:
            _wandb_publish_once(
                plan, stage=stage, path=path, artifact_type=artifact_type,
                summary=summary,
            )
            return
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")
            if attempt == WANDB_PUBLISH_ATTEMPTS:
                raise ChainError(
                    f"required W&B publication failed after {attempt} attempts: "
                    + " | ".join(errors)
                ) from exc
            print(
                f"[formal-wandb] {stage} attempt {attempt}/"
                f"{WANDB_PUBLISH_ATTEMPTS} failed; retrying in "
                f"{WANDB_RETRY_SECONDS}s: {errors[-1]}",
                flush=True,
            )
            time.sleep(WANDB_RETRY_SECONDS)


def _stage_consolidate(plan: Mapping[str, Any]) -> int:
    endpoint, endpoint_sha, selected_shards = _authenticate_endpoint(plan)
    step = int(endpoint["step"])
    run_dir = Path(plan["lineage"]["run_dir"])
    checkpoint = Path(plan["paths"]["checkpoint"])
    report = Path(plan["paths"]["checkpoint_report"])
    receipt_path = Path(plan["paths"]["checkpoint_receipt"])
    pinned_dir = checkpoint.parent / f"shards_{step:09d}"
    run_config_sha = sha256_file(run_dir / "config.json")
    if receipt_path.exists():
        receipt = _read_receipt(
            receipt_path, kind="r0_e2e_consolidated_checkpoint_receipt",
        )
        if (
            receipt.get("step") != step
            or receipt.get("checkpoint") != str(checkpoint)
            or not checkpoint.is_file()
            or sha256_file(checkpoint) != receipt.get("checkpoint_sha256")
            or not report.is_file()
            or sha256_file(report) != receipt.get("verification_report_sha256")
            or json.loads(report.read_text()).get("pass") is not True
            or receipt.get("terminal_endpoint_sha256") != endpoint_sha
            or receipt.get("selected_checkpoint_shards") != selected_shards
            or receipt.get("pinned_checkpoint_shards") != selected_shards
            or _checkpoint_shard_receipt(
                _checkpoint_shards(pinned_dir, step, require_latest=False)
            ) != selected_shards
            or receipt.get("run_config_sha256") != run_config_sha
        ):
            raise ChainError("existing checkpoint receipt failed immutable retry closure")
        _wandb_publish(
            plan, stage="consolidate", path=receipt_path,
            artifact_type="checkpoint-receipt",
            summary={"checkpoint_step": step, "endpoint_status": endpoint["status"]},
        )
        return 0
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    # The consolidator publishes ckpt.pt with os.replace.  If a requeue lands
    # after that atomic publication but before our receipt, verify/adopt the
    # complete file; otherwise build it.  A malformed existing checkpoint is
    # never replaced under this formal lineage.
    command = [
        sys.executable, "-m", "loom.train.consolidate",
        "--run_dir", str(run_dir), "--step", str(step),
        "--out", str(checkpoint), "--config", str(run_dir / "config.json"),
        "--pin", "--report", str(report),
    ]
    if checkpoint.exists():
        command.append("--verify_only")
    subprocess.run(command, cwd=ROOT, check=True)
    try:
        verification = json.loads(report.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ChainError("consolidation verification report is unreadable") from exc
    if verification.get("pass") is not True:
        raise ChainError("consolidation verification did not pass")
    pinned_shards = _checkpoint_shard_receipt(
        _checkpoint_shards(pinned_dir, step, require_latest=False)
    )
    if pinned_shards != selected_shards:
        raise ChainError("pinned consolidation shards differ from selected endpoint")
    endpoint_after, endpoint_sha_after, selected_shards_after = _authenticate_endpoint(plan)
    if (
        endpoint_after != endpoint
        or endpoint_sha_after != endpoint_sha
        or selected_shards_after != selected_shards
    ):
        raise ChainError("terminal endpoint or selected shards changed during consolidation")
    receipt = {
        "format_version": FORMAT_VERSION,
        "kind": "r0_e2e_consolidated_checkpoint_receipt",
        "step": step,
        "endpoint_status": endpoint["status"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "verification_report": str(report),
        "verification_report_sha256": sha256_file(report),
        "run_config_sha256": run_config_sha,
        "terminal_endpoint_sha256": endpoint_sha,
        "selected_checkpoint_shards": selected_shards,
        "pinned_checkpoint_shards": pinned_shards,
        "checkpoint_bytes_uploaded_to_wandb": False,
    }
    exclusive_json_write(receipt_path, receipt)
    _wandb_publish(
        plan, stage="consolidate", path=receipt_path,
        artifact_type="checkpoint-receipt",
        summary={"checkpoint_step": step, "endpoint_status": endpoint["status"]},
    )
    return 0


def _result_protocol(blob: Mapping[str, Any]):
    from loom.eval import EvalProtocol  # noqa: PLC0415

    return EvalProtocol.from_dict(dict(blob.get("protocol", {})))


def _validate_seed_result(
    plan: Mapping[str, Any], *, seed: int, result_path: Path,
) -> tuple[dict[str, Any], Any]:
    checkpoint_receipt = _read_receipt(
        plan["paths"]["checkpoint_receipt"],
        kind="r0_e2e_consolidated_checkpoint_receipt",
    )
    checkpoint = Path(checkpoint_receipt["checkpoint"])
    if sha256_file(checkpoint) != checkpoint_receipt["checkpoint_sha256"]:
        raise ChainError("consolidated checkpoint changed before/during evaluation")
    blob = json.loads(result_path.read_text())
    protocol = _result_protocol(blob)
    expected_protocol = {
        "bench": "libero",
        "seeds": (seed,),
        "suites": tuple(plan["evaluation"]["suites"]),
        "n_tasks": int(plan["evaluation"]["tasks_per_suite"]),
        "episodes_per_task": int(plan["evaluation"]["episodes_per_task"]),
        "max_steps": int(plan["evaluation"]["max_steps"]),
    }
    actual_protocol = {
        "bench": protocol.bench,
        "seeds": protocol.seeds,
        "suites": protocol.suites,
        "n_tasks": protocol.n_tasks,
        "episodes_per_task": protocol.episodes_per_task,
        "max_steps": protocol.max_steps,
    }
    if actual_protocol != expected_protocol:
        raise ChainError(
            f"seed {seed} protocol mismatch: expected {expected_protocol!r}, "
            f"got {actual_protocol!r}"
        )
    if protocol.total_episodes != EXPECTED_EPISODES_PER_SEED:
        raise ChainError(
            f"seed {seed} protocol has {protocol.total_episodes} expected episodes"
        )
    summary = blob.get("summary", {})
    if (
        summary.get("complete") is not True
        or summary.get("n_episodes") != EXPECTED_EPISODES_PER_SEED
        or summary.get("n_expected") != EXPECTED_EPISODES_PER_SEED
        or summary.get("n_errors") != 0
    ):
        raise ChainError(f"seed {seed} evaluation is incomplete or contains errors")
    policy = blob.get("meta", {}).get("policy") or {}
    if policy.get("is_stub") is not False:
        raise ChainError(f"seed {seed} evaluation did not prove a real policy")
    if policy.get("ckpt_global_step") != checkpoint_receipt["step"]:
        raise ChainError(f"seed {seed} checkpoint step mismatch")
    _validate_exact_eval_blob(
        blob, seed=seed, label=f"candidate seed {seed}",
        identity_profile="current_candidate",
    )
    _validate_eval_method_identity(
        blob, label=f"candidate seed {seed}",
        identity_profile="current_candidate",
        checkpoint_step=int(checkpoint_receipt["step"]),
        checkpoint_path=str(checkpoint.resolve()),
    )
    return blob, protocol


def _formal_eval_command(
    plan: Mapping[str, Any], *, seed: int, checkpoint: Path, out_dir: Path,
) -> tuple[list[str], dict[str, str]]:
    if not LIBERO_EVAL_PYTHON.is_file():
        raise ChainError(f"missing pinned LIBERO Python: {LIBERO_EVAL_PYTHON}")
    result_path = out_dir / "results.json"
    table_path = out_dir / "table.md"
    command = [
        str(LIBERO_EVAL_PYTHON), "-m", "loom.eval",
        "--bench", "libero", "--backend", "libero",
        "--ckpt", str(checkpoint), "--require-real", "--op-stats",
        "--n-tasks", str(plan["evaluation"]["tasks_per_suite"]),
        "--episodes-per-task", str(plan["evaluation"]["episodes_per_task"]),
        "--seeds", str(seed), "--workers", "8",
        "--gripper-dwell", str(plan["evaluation"]["gripper_dwell"]),
        "--decoder-samples", str(plan["evaluation"]["decoder_samples"]),
        "--out", str(result_path), "--md", str(table_path),
    ]
    if plan["evaluation"]["duration_normalize_segments"]:
        command.append("--duration-normalize-segments")
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": str(ROOT),
        "PYTHONUNBUFFERED": "1",
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "MUJOCO_EGL_DEVICE_ID": "0",
        "HF_HOME": (
            "/lustre/fsw/portfolios/edgeai/users/chrislin/datasets/loom/hf_cache"
        ),
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "TRANSFORMERS_VERBOSITY": "error",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "8",
        "TRITON_CACHE_DIR": str(ROOT / ".triton_cache"),
    })
    return command, env


def _stage_eval(plan: Mapping[str, Any], stage: str) -> int:
    match = re.fullmatch(r"eval_seed([0-9]+)", stage)
    if match is None:
        raise ChainError(f"invalid evaluation stage: {stage!r}")
    seed = int(match.group(1))
    if seed not in SEEDS:
        raise ChainError(f"undeclared evaluation seed: {seed}")
    receipt = _read_receipt(
        plan["paths"]["checkpoint_receipt"],
        kind="r0_e2e_consolidated_checkpoint_receipt",
    )
    checkpoint = Path(receipt["checkpoint"])
    if sha256_file(checkpoint) != receipt["checkpoint_sha256"]:
        raise ChainError("checkpoint identity changed before evaluation")
    out_dir = Path(plan["paths"]["eval"][str(seed)]["out_dir"])
    receipt_path = Path(plan["paths"]["eval"][str(seed)]["receipt"])
    result_path = out_dir / "results.json"
    if receipt_path.exists():
        existing = _read_receipt(
            receipt_path, kind="r0_e2e_single_seed_eval_receipt",
        )
        if (
            existing.get("seed") != seed
            or existing.get("result") != str(result_path)
            or not result_path.is_file()
            or sha256_file(result_path) != existing.get("result_sha256")
        ):
            raise ChainError(f"seed {seed} existing receipt failed immutable retry closure")
        blob, _ = _validate_seed_result(plan, seed=seed, result_path=result_path)
        _wandb_publish(
            plan, stage=f"eval-seed-{seed}", path=result_path,
            artifact_type="evaluation-results",
            summary={
                "seed": seed, "episodes": EXPECTED_EPISODES_PER_SEED,
                "success_rate": blob["summary"]["avg"],
                "checkpoint_step": receipt["step"],
            },
        )
        return 0
    if out_dir.exists() and not out_dir.is_dir():
        raise ChainError(f"evaluation output path is not a directory: {out_dir}")
    if result_path.is_symlink():
        raise ChainError(f"evaluation result path must not be a symlink: {result_path}")
    blob: dict[str, Any] | None = None
    if result_path.is_file():
        try:
            blob, _ = _validate_seed_result(
                plan, seed=seed, result_path=result_path,
            )
        except ChainError:
            # An incomplete atomic ResultStore is eligible for exact resume;
            # malformed/mismatched stores are rejected by ResultStore itself.
            blob = None
    if blob is None:
        command, env = _formal_eval_command(
            plan, seed=seed, checkpoint=checkpoint, out_dir=out_dir,
        )
        # ResultStore authenticates protocol/checkpoint/eval identity and
        # atomically resumes only missing episodes.
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        blob, _ = _validate_seed_result(
            plan, seed=seed, result_path=result_path,
        )
    eval_receipt = {
        "format_version": FORMAT_VERSION,
        "kind": "r0_e2e_single_seed_eval_receipt",
        "seed": seed,
        "result": str(result_path),
        "result_sha256": sha256_file(result_path),
        "episodes": EXPECTED_EPISODES_PER_SEED,
        "avg": blob["summary"]["avg"],
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "checkpoint_step": receipt["step"],
    }
    exclusive_json_write(receipt_path, eval_receipt)
    _wandb_publish(
        plan, stage=f"eval-seed-{seed}", path=result_path,
        artifact_type="evaluation-results",
        summary={
            "seed": seed, "episodes": EXPECTED_EPISODES_PER_SEED,
            "success_rate": blob["summary"]["avg"],
            "checkpoint_step": receipt["step"],
        },
    )
    return 0


def _protocol_without_seeds(protocol: Any) -> dict[str, Any]:
    value = protocol.to_dict()
    for key in ("seeds", "total_episodes", "description"):
        value.pop(key, None)
    return value


def _stable_merge_provenance(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Plan-owned merge identity; deliberately excludes worker host/time."""
    return {
        "kind": "r0_e2e_plan_stable_merge_v1",
        "orchestration_source_closure_sha256": plan[
            "orchestration_source_closure"
        ]["sha256"],
        "config_raw_sha256": plan["config"]["raw_sha256"],
        "config_resolved_hash": plan["config"]["resolved_config_hash"],
    }


def _baseline_rows(
    plan: Mapping[str, Any],
) -> dict[tuple[str, str, int, int, int], dict[str, Any]]:
    frozen = plan.get("baseline_comparison", {}).get("baseline")
    if not isinstance(frozen, Mapping):
        raise ChainError("plan has no authenticated canonical baseline")
    current = _authenticate_baseline(Path(str(frozen.get("root"))))
    if current != dict(frozen):
        raise ChainError("canonical baseline receipt changed before comparison")
    rows: dict[tuple[str, str, int, int, int], dict[str, Any]] = {}
    for seed in SEEDS:
        path = Path(current["files"][str(seed)]["path"])
        blob = json.loads(path.read_text())
        seed_rows = _validate_exact_eval_blob(
            blob, seed=seed, label=f"baseline seed {seed}",
            identity_profile="historical_baseline",
        )
        if rows.keys() & seed_rows.keys():
            raise ChainError("canonical baseline contains cross-seed duplicate keys")
        rows.update(seed_rows)
    return rows


def _suite_stratified_bootstrap_matrix(
    task_keys: Sequence[str],
) -> tuple[Any, dict[str, Any]]:
    import torch  # noqa: PLC0415

    keys = tuple(str(key) for key in task_keys)
    if len(keys) != len(set(keys)) or len(keys) != 40:
        raise ChainError("paired bootstrap requires exactly 40 unique task keys")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(BOOTSTRAP_SEED)
    blocks = []
    suite_columns: dict[str, list[int]] = {}
    for suite in sorted(SUITE_FLOORS_PERCENT):
        indices = [index for index, key in enumerate(keys) if key.startswith(f"{suite}/")]
        if len(indices) != 10:
            raise ChainError(f"paired bootstrap suite {suite} has {len(indices)} tasks")
        choices = torch.randint(
            0, 10, (BOOTSTRAP_SAMPLES, 10), generator=generator,
            dtype=torch.int64,
        )
        lookup = torch.tensor(indices, dtype=torch.int64)
        blocks.append(lookup.index_select(0, choices.reshape(-1)).reshape_as(choices))
        suite_columns[suite] = indices
    matrix = torch.cat(blocks, dim=1).contiguous()
    digest = hashlib.sha256(matrix.numpy().tobytes(order="C")).hexdigest()
    if digest != BOOTSTRAP_MATRIX_SHA256:
        raise ChainError(
            f"fixed bootstrap matrix identity changed: {digest}"
        )
    return matrix, {
        "kind": "fixed_suite_stratified_task_resample_matrix_v1",
        "shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
        "samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
        "suite_columns": suite_columns,
        "confidence": BOOTSTRAP_CONFIDENCE,
        "lower_quantile": 0.025,
        "upper_quantile": 0.975,
        "lower_interpolation": "lower",
        "upper_interpolation": "higher",
        "sha256": digest,
    }


def paired_baseline_comparison(
    plan: Mapping[str, Any],
    candidate_rows: Mapping[tuple[str, str, int, int, int], Mapping[str, Any]],
    baseline_rows: Mapping[tuple[str, str, int, int, int], Mapping[str, Any]],
) -> dict[str, Any]:
    """Pair exact episodes, reduce to 40 tasks, and apply the frozen gate."""
    import torch  # noqa: PLC0415

    if set(candidate_rows) != set(baseline_rows) or len(candidate_rows) != 1_200:
        raise ChainError("candidate/baseline episode pairing is not exact 1,200")
    task_differences: dict[str, list[int]] = {}
    new_only = old_only = tie_success = tie_failure = 0
    candidate_successes = baseline_successes = 0
    candidate_by_seed = {seed: 0 for seed in SEEDS}
    suite_counts = {
        suite: {"candidate": 0, "baseline": 0, "episodes": 0}
        for suite in SUITE_FLOORS_PERCENT
    }
    for key in sorted(candidate_rows):
        candidate = candidate_rows[key]
        baseline = baseline_rows[key]
        candidate_policy_seed = (candidate.get("extra") or {}).get("policy_seed")
        baseline_policy_seed = (baseline.get("extra") or {}).get("policy_seed")
        if (
            candidate.get("env_seed") != baseline.get("env_seed")
            or candidate_policy_seed != baseline_policy_seed
        ):
            raise ChainError(f"candidate/baseline RNG mismatch for paired key {key}")
        new = bool(candidate["success"])
        old = bool(baseline["success"])
        candidate_successes += int(new)
        baseline_successes += int(old)
        _, suite, task_id, _, seed = key
        candidate_by_seed[seed] += int(new)
        suite_counts[suite]["candidate"] += int(new)
        suite_counts[suite]["baseline"] += int(old)
        suite_counts[suite]["episodes"] += 1
        task_key = f"{suite}/task={task_id:02d}"
        task_differences.setdefault(task_key, []).append(int(new) - int(old))
        if new and not old:
            new_only += 1
        elif old and not new:
            old_only += 1
        elif new:
            tie_success += 1
        else:
            tie_failure += 1
    if baseline_successes != BASELINE_SUCCESS_TOTAL:
        raise ChainError("paired baseline no longer contains exact 447 successes")
    task_keys = sorted(task_differences)
    if any(len(task_differences[key]) != 30 for key in task_keys):
        raise ChainError("paired task reduction requires exactly 30 episodes per task")
    task_delta_rows = [
        {
            "task_key": key,
            "paired_episodes": len(task_differences[key]),
            "delta_successes": sum(task_differences[key]),
            "delta_percentage_points": (
                100.0 * sum(task_differences[key]) / len(task_differences[key])
            ),
        }
        for key in task_keys
    ]
    values = torch.tensor(
        [row["delta_percentage_points"] for row in task_delta_rows],
        dtype=torch.float64,
    )
    matrix, matrix_receipt = _suite_stratified_bootstrap_matrix(task_keys)
    draws = values.index_select(0, matrix.reshape(-1)).reshape(matrix.shape).mean(dim=1)
    lower = float(torch.quantile(draws, 0.025, interpolation="lower"))
    upper = float(torch.quantile(draws, 0.975, interpolation="higher"))
    point_delta = float(values.mean())
    exact_delta = 100.0 * (candidate_successes - baseline_successes) / 1_200
    if abs(point_delta - exact_delta) > 1.0e-12:
        raise ChainError("equal-task paired point differs from exact balanced protocol delta")

    per_suite: dict[str, Any] = {}
    for suite, counts in suite_counts.items():
        if counts["episodes"] != 300:
            raise ChainError(f"paired suite {suite} does not contain 300 episodes")
        candidate_rate = 100.0 * counts["candidate"] / 300
        baseline_rate = 100.0 * counts["baseline"] / 300
        per_suite[suite] = {
            "episodes": 300,
            "candidate_successes": counts["candidate"],
            "baseline_successes": counts["baseline"],
            "candidate_success_rate_percent": candidate_rate,
            "baseline_success_rate_percent": baseline_rate,
            "delta_percentage_points": candidate_rate - baseline_rate,
        }
    thresholds = plan["baseline_comparison"]["thresholds"]
    candidate_rate = 100.0 * candidate_successes / 1_200
    suite_checks = {
        suite: per_suite[suite]["candidate_success_rate_percent"] >= floor
        for suite, floor in SUITE_FLOORS_PERCENT.items()
    }
    checks = {
        "candidate_success_rate_strict_gt_baseline": (
            candidate_rate > 100.0 * BASELINE_SUCCESS_TOTAL / 1_200
        ),
        "paired_task_bootstrap_ci_low_strict_gt_zero": (
            lower > float(thresholds["paired_task_bootstrap_ci_low_strict_gt"])
        ),
        "seed0_successes_gte_164": (
            candidate_by_seed[0] >= int(thresholds["seed0_stretch_successes_gte"])
        ),
        **{f"suite_floor/{suite}": passed for suite, passed in suite_checks.items()},
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "passed": all(checks.values()),
        "scientific_failure_still_publishes": True,
        "pairing": {
            "key_fields": ["bench", "suite", "task_id", "episode", "seed"],
            "paired_episodes": 1_200,
            "new_only": new_only,
            "old_only": old_only,
            "tie_success": tie_success,
            "tie_failure": tie_failure,
            "rng_identity_equal": True,
        },
        "overall": {
            "candidate_successes": candidate_successes,
            "baseline_successes": baseline_successes,
            "episodes": 1_200,
            "candidate_success_rate_percent": candidate_rate,
            "baseline_success_rate_percent": 100.0 * baseline_successes / 1_200,
            "delta_percentage_points": exact_delta,
        },
        "per_suite": per_suite,
        "per_seed_candidate_successes": {
            str(seed): candidate_by_seed[seed] for seed in SEEDS
        },
        "task_deltas": task_delta_rows,
        "task_deltas_sha256": hashlib.sha256(
            _canonical_json(task_delta_rows).encode()
        ).hexdigest(),
        "paired_task_bootstrap": {
            "method": "suite_stratified_40_task_paired_percentile_two_sided",
            "point_delta_percentage_points": point_delta,
            "ci_low_percentage_points": lower,
            "ci_high_percentage_points": upper,
            "resample_matrix": matrix_receipt,
        },
        "thresholds": dict(thresholds),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
    }


def merge_seed_results(plan: Mapping[str, Any]) -> dict[str, Any]:
    from loom.eval import EpisodeResult  # noqa: PLC0415
    from loom.eval.runner import aggregate, code_provenance, iter_work  # noqa: PLC0415

    blobs: list[dict[str, Any]] = []
    protocols = []
    source_receipts = []
    for seed in SEEDS:
        receipt = _read_receipt(
            plan["paths"]["eval"][str(seed)]["receipt"],
            kind="r0_e2e_single_seed_eval_receipt",
        )
        result_path = Path(receipt["result"])
        if sha256_file(result_path) != receipt["result_sha256"]:
            raise ChainError(f"seed {seed} result changed before merge")
        blob, protocol = _validate_seed_result(
            plan, seed=seed, result_path=result_path,
        )
        blobs.append(blob)
        protocols.append(protocol)
        source_receipts.append(receipt)
    reference = _protocol_without_seeds(protocols[0])
    if any(_protocol_without_seeds(protocol) != reference for protocol in protocols[1:]):
        raise ChainError("singleton-seed protocols differ beyond their seed field")
    target_protocol = protocols[0].replace(seeds=SEEDS)

    records = [
        EpisodeResult.from_dict(row)
        for blob in blobs for row in blob.get("episodes", [])
    ]
    keys = [record.key() for record in records]
    if len(keys) != len(set(keys)):
        raise ChainError("singleton-seed evaluations contain duplicate episode keys")
    expected = {item.key() for item in iter_work(target_protocol)}
    if set(keys) != expected or len(keys) != EXPECTED_EPISODES_TOTAL:
        raise ChainError("singleton-seed episode union is not the exact 1,200 protocol")
    common_ckpt = blobs[0].get("meta", {}).get("ckpt")
    common_identity = blobs[0].get("meta", {}).get("eval_identity")
    for blob in blobs[1:]:
        if blob.get("meta", {}).get("ckpt") != common_ckpt:
            raise ChainError("singleton-seed checkpoint identities differ")
        if blob.get("meta", {}).get("eval_identity") != common_identity:
            raise ChainError("singleton-seed evaluation identities differ")

    candidate_rows: dict[tuple[str, str, int, int, int], dict[str, Any]] = {}
    for seed, blob in zip(SEEDS, blobs):
        rows = _validate_exact_eval_blob(
            blob, seed=seed, label=f"candidate seed {seed}",
            identity_profile="current_candidate",
        )
        if candidate_rows.keys() & rows.keys():
            raise ChainError("candidate singleton seeds overlap")
        candidate_rows.update(rows)
    comparison = paired_baseline_comparison(
        plan, candidate_rows, _baseline_rows(plan),
    )

    summary = aggregate(records, target_protocol)
    if (
        summary.get("complete") is not True
        or summary.get("n_episodes") != EXPECTED_EPISODES_TOTAL
        or summary.get("n_expected") != EXPECTED_EPISODES_TOTAL
        or summary.get("n_errors") != 0
    ):
        raise ChainError("merged evaluation summary failed exact closure")
    return {
        "version": blobs[0].get("version", 1),
        "bench": target_protocol.bench,
        "protocol": target_protocol.to_dict(),
        "meta": {
            "ckpt": common_ckpt,
            "eval_identity": common_identity,
            "policy": blobs[0].get("meta", {}).get("policy"),
            "source_singleton_seed_receipts": source_receipts,
            "merge_provenance": _stable_merge_provenance(plan),
            "merge_rule": "exact_disjoint_seed_union_then_loom.eval.runner.aggregate",
            "baseline_receipt": plan["baseline_comparison"]["baseline"],
        },
        "summary": summary,
        "baseline_comparison": comparison,
        "episodes": [record.to_dict() for record in sorted(records, key=lambda r: r.key())],
    }


def _markdown_table(merged: Mapping[str, Any]) -> str:
    comparison = merged["baseline_comparison"]
    rows = [
        "| suite | candidate | baseline | delta (pp) | episodes |",
        "|---|---:|---:|---:|---:|",
    ]
    for suite, value in merged["summary"]["per_suite"].items():
        paired = comparison["per_suite"][suite]
        rows.append(
            f"| {suite} | {float(value['success_rate']):.2f}% | "
            f"{float(paired['baseline_success_rate_percent']):.2f}% | "
            f"{float(paired['delta_percentage_points']):+.2f} | "
            f"{int(value['n_episodes'])} |"
        )
    overall = comparison["overall"]
    rows.append(
        f"| **average** | **{float(merged['summary']['avg']):.2f}%** | "
        f"**{float(overall['baseline_success_rate_percent']):.2f}%** | "
        f"**{float(overall['delta_percentage_points']):+.2f}** | "
        f"**{int(merged['summary']['n_episodes'])}** |"
    )
    bootstrap = comparison["paired_task_bootstrap"]
    rows.extend((
        "",
        f"Paired task-bootstrap 95% CI: "
        f"[{float(bootstrap['ci_low_percentage_points']):+.3f}, "
        f"{float(bootstrap['ci_high_percentage_points']):+.3f}] pp.",
        f"Scientific evaluation gate: **{comparison['status']}**.",
    ))
    return "\n".join(rows) + "\n"


def _exclusive_text_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _eval_summary_wandb_fields(comparison: Mapping[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "baseline_delta_pp": comparison["overall"]["delta_percentage_points"],
        "paired_ci_low_pp": comparison["paired_task_bootstrap"][
            "ci_low_percentage_points"
        ],
        "paired_ci_high_pp": comparison["paired_task_bootstrap"][
            "ci_high_percentage_points"
        ],
        "seed0_successes": comparison["per_seed_candidate_successes"]["0"],
        "scientific_gate_status": comparison["status"],
        "scientific_gate_passed": comparison["passed"],
    }
    for suite, row in comparison["per_suite"].items():
        fields[f"candidate_success_rate/{suite}"] = row[
            "candidate_success_rate_percent"
        ]
        fields[f"baseline_delta_pp/{suite}"] = row["delta_percentage_points"]
    return fields


def _stage_merge(plan: Mapping[str, Any]) -> int:
    result_path = Path(plan["paths"]["merged_results"])
    table_path = Path(plan["paths"]["merged_table"])
    receipt_path = Path(plan["paths"]["merged_receipt"])
    if receipt_path.exists():
        receipt = _read_receipt(receipt_path, kind="r0_e2e_merged_eval_receipt")
        existing_blob = json.loads(result_path.read_text()) if result_path.is_file() else {}
        existing_comparison = existing_blob.get("baseline_comparison")
        if (
            not result_path.is_file()
            or sha256_file(result_path) != receipt.get("result_sha256")
            or not table_path.is_file()
            or sha256_file(table_path) != receipt.get("table_sha256")
            or receipt.get("episodes") != EXPECTED_EPISODES_TOTAL
            or receipt.get("errors") != 0
            or receipt.get("complete") is not True
            or not isinstance(existing_comparison, Mapping)
            or hashlib.sha256(
                _canonical_json(existing_comparison).encode()
            ).hexdigest() != receipt.get("baseline_comparison_sha256")
            or existing_comparison.get("status") != receipt.get("scientific_gate_status")
            or existing_comparison.get("paired_task_bootstrap", {}).get(
                "resample_matrix", {}
            ).get("sha256") != BOOTSTRAP_MATRIX_SHA256
            or _authenticate_baseline(
                Path(plan["baseline_comparison"]["baseline"]["root"])
            ) != plan["baseline_comparison"]["baseline"]
        ):
            raise ChainError("existing merged receipt failed immutable retry closure")
        _wandb_publish(
            plan, stage="eval-summary", path=result_path,
            artifact_type="evaluation-results",
            summary={
                "episodes": EXPECTED_EPISODES_TOTAL,
                "success_rate": receipt["avg"],
                "n_errors": 0,
                **_eval_summary_wandb_fields(existing_comparison),
            },
        )
        return 0
    merged = merge_seed_results(plan)
    expected_result = _pretty_json(merged)
    expected_table = _markdown_table(merged)
    if result_path.exists():
        if not result_path.is_file() or result_path.read_text() != expected_result:
            raise ChainError("existing partial merged result differs from recomputation")
    else:
        exclusive_json_write(result_path, merged)
    if table_path.exists():
        if not table_path.is_file() or table_path.read_text() != expected_table:
            raise ChainError("existing partial merged table differs from recomputation")
    else:
        _exclusive_text_write(table_path, expected_table)
    comparison = merged["baseline_comparison"]
    receipt = {
        "format_version": FORMAT_VERSION,
        "kind": "r0_e2e_merged_eval_receipt",
        "result": str(result_path),
        "result_sha256": sha256_file(result_path),
        "table": str(table_path),
        "table_sha256": sha256_file(table_path),
        "episodes": merged["summary"]["n_episodes"],
        "errors": merged["summary"]["n_errors"],
        "avg": merged["summary"]["avg"],
        "complete": merged["summary"]["complete"],
        "baseline_comparison_sha256": hashlib.sha256(
            _canonical_json(comparison).encode()
        ).hexdigest(),
        "baseline_file_sha256": {
            seed: row["sha256"]
            for seed, row in plan["baseline_comparison"]["baseline"]["files"].items()
        },
        "bootstrap_matrix_sha256": BOOTSTRAP_MATRIX_SHA256,
        "baseline_delta_pp": comparison["overall"]["delta_percentage_points"],
        "paired_ci_low_pp": comparison["paired_task_bootstrap"][
            "ci_low_percentage_points"
        ],
        "paired_ci_high_pp": comparison["paired_task_bootstrap"][
            "ci_high_percentage_points"
        ],
        "seed0_successes": comparison["per_seed_candidate_successes"]["0"],
        "per_suite": comparison["per_suite"],
        "scientific_gate_status": comparison["status"],
        "scientific_gate_passed": comparison["passed"],
        "failed_scientific_checks": comparison["failed_checks"],
    }
    exclusive_json_write(receipt_path, receipt)
    _wandb_publish(
        plan, stage="eval-summary", path=result_path,
        artifact_type="evaluation-results",
        summary={
            "episodes": EXPECTED_EPISODES_TOTAL,
            "success_rate": merged["summary"]["avg"],
            "n_errors": 0,
            **_eval_summary_wandb_fields(comparison),
        },
    )
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    return 0


def run_environment_stage() -> int:
    plan, stage = _required_plan_from_environment()
    if stage.startswith("train_") or stage == "extension":
        return _stage_train(plan, stage)
    if stage == "gate32":
        return _stage_gate32(plan)
    if stage == "gatefinal":
        return _stage_gatefinal(plan)
    if stage == "consolidate":
        return _stage_consolidate(plan)
    if stage.startswith("eval_seed"):
        return _stage_eval(plan, stage)
    if stage == "merge":
        return _stage_merge(plan)
    raise ChainError(f"unknown formal stage {stage!r}")


def _submit_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--control-dir", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument(
        "--baseline-root", default=str(CANONICAL_BASELINE_ROOT),
        help="read-only root containing the three SHA-pinned canonical results",
    )
    parser.add_argument("--group", required=True)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--initial-links", type=int, default=INITIAL_LINKS)
    parser.add_argument("--step32", type=int, default=STEP_32K)
    parser.add_argument("--step40", type=int, default=STEP_40K)
    parser.add_argument("--dry-run", action="store_true")


def _plan_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return build_plan(
        config_path=_absolute_new_path(args.config, field="config"),
        run_dir=_absolute_new_path(args.run_dir, field="run-dir"),
        control_dir=_absolute_new_path(args.control_dir, field="control-dir"),
        artifact_root=_absolute_new_path(args.artifact_root, field="artifact-root"),
        baseline_root=_absolute_new_path(args.baseline_root, field="baseline-root"),
        group=args.group,
        project=args.project,
        require_online=True,
        step32=args.step32,
        step40=args.step40,
        initial_links=args.initial_links,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    submit_parser = subparsers.add_parser("submit")
    _submit_args(submit_parser)
    subparsers.add_parser("run-stage")
    args = parser.parse_args(argv)
    try:
        if args.command == "run-stage":
            return run_environment_stage()
        plan = _plan_from_args(args)
        if args.dry_run:
            print(json.dumps({
                "plan": plan,
                "stages": _plan_stage_specs(plan),
                "submitted": False,
            }, indent=2, sort_keys=True))
            return 0
        result = submit_plan(plan)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (ChainError, FileExistsError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"FORMAL_CHAIN_INVALID: {exc}", file=sys.stderr, flush=True)
        return 2
    except subprocess.CalledProcessError as exc:
        command = " ".join(shlex.quote(str(item)) for item in exc.cmd)
        print(
            f"FORMAL_CHAIN_FAILED: command exited {exc.returncode}: {command}",
            file=sys.stderr, flush=True,
        )
        return int(exc.returncode or 1)


if __name__ == "__main__":
    raise SystemExit(main())
