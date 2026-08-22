#!/usr/bin/env python3
"""Generic fixed-endpoint train/evaluate chain for protected H/P/I arms.

The heavy integrity, resume, consolidation, LIBERO evaluation, and exact merge
implementation remains the already exercised operator-repair chain.  This
module supplies a closed, versioned profile table and binds one profile before
any base-chain plan is trusted.  Every arm is a fresh seed-0, exact-step-32k
lineage followed by unconditional seeds 0/1/2 x 400 evaluation and an exact
1,200-row merge.  No metric or success-rate threshold controls the DAG.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import r0_e2e_operator_repair_chain as base  # noqa: E402


FORMAT_VERSION = 1
KIND = "r0_e2e_protected_action_fixed_endpoint_chain_v1"
PROFILE_KIND = "loom-r0a-protected-parallel-profile-v1"
PROJECT = "loom-r0-protected-arms"
FIXED_STEP = 32_000
TRAIN_LINKS = 6
PLAN_SELECTOR_MAX_BYTES = 16 * 1024 * 1024

PROTECTED_PROTOCOL = {
    "format": "loom-r0a-protected-operator-arms-v1",
    "initialization": "fresh_seed0",
    "fixed_endpoint_update": FIXED_STEP,
    "checkpoint_selection": "fixed_update_only",
    "evaluation_is_unconditional": True,
    "evaluation_seeds": [0, 1, 2],
    "evaluation_episodes": 1_200,
    "health_thresholds_control_execution": False,
    "outcome_threshold_applied": False,
}
REFERENCE_ACTION_POLICY = {
    "kind": "r0a_dualcode_formal_s0_20260820_v2_step32000",
    "source_config": "configs/r0a_dual_code_formal.yaml",
    "source_config_hash": "d030206d56a71718",
    "checkpoint_step": FIXED_STEP,
    "checkpoint_sha256": (
        "eddcc36d94dc48b9031acbcdaea116b2a1693c8b9e357f96e2573da36c9039b6"
    ),
    "checkpoint_receipt_sha256": (
        "354dd8c1ce0339555c7794e20cb1229e4a3bb4602284012ca395381020ab3e84"
    ),
    "successes": 550,
    "episodes": 1_200,
    "success_rate": 0.4583333333333333,
}


@dataclass(frozen=True)
class ArmProfile:
    arm: str
    config_name: str
    config_sha256: str
    resolved_config_hash: str
    run_name: str
    group: str
    parent: str
    method_delta: str
    isolate_estimator_gradients: bool

    @property
    def tags(self) -> tuple[str, ...]:
        return (
            "protected-action", "fixed-endpoint", "no-gate",
            "fresh", "r0", f"arm-{self.arm.lower()}",
        )

    @property
    def config_path(self) -> Path:
        return (ROOT / self.config_name).resolve()

    @property
    def protected_arm(self) -> dict[str, Any]:
        return {
            "id": self.arm,
            "parent": self.parent,
            "method_delta": self.method_delta,
        }


PROFILES = {
    profile.arm: profile
    for profile in (
        ArmProfile(
            arm="H",
            config_name="configs/r0a_protected_h.yaml",
            config_sha256=(
                "39befbbdc2bfacf62aef4dd9c890bf2747ff3142fcc63ea9769641328defdcce"
            ),
            resolved_config_hash="7c79c09af56ccf16",
            run_name="r0a_protected_h_fresh_s0_20260822",
            group="r0a-protected-fixed32k-s0-20260822-arm-h-v1",
            parent="dual_code_45p83_objective",
            method_delta="history_and_suite_exposure_only",
            isolate_estimator_gradients=False,
        ),
        ArmProfile(
            arm="P",
            config_name="configs/r0a_protected_p.yaml",
            config_sha256=(
                "145e41e0201dd80dc95b0beb4325f26ae3795a39ded0270ff1cb9aa2a1e45948"
            ),
            resolved_config_hash="08ca78d71dc0321b",
            run_name="r0a_protected_p_fresh_s0_20260822",
            group="r0a-protected-fixed32k-s0-20260822-arm-p-v1",
            parent="H",
            method_delta=(
                "q_delta_semantic_formation_plus_effect_contrastive_dynamics"
            ),
            isolate_estimator_gradients=False,
        ),
        ArmProfile(
            arm="I",
            config_name="configs/r0a_protected_i.yaml",
            config_sha256=(
                "71ac5d48b6b7b2f6f15ab098c3ec2063402c544bec4ae12deaad023985a0a73a"
            ),
            resolved_config_hash="3cb3dea37bcb9aaf",
            run_name="r0a_protected_i_fresh_s0_20260822",
            group="r0a-protected-fixed32k-s0-20260822-arm-i-v1",
            parent="P",
            method_delta="isolate_operator_objective_from_online_estimator",
            isolate_estimator_gradients=True,
        ),
    )
}

PROTECTED_SOURCE_FILES = tuple(sorted(set(
    base.COMMON_SOURCE_FILES
    + (
        "scripts/r0_e2e_operator_repair_chain.py",
        "scripts/r0_e2e_operator_repair_train_entry.py",
        "scripts/r0_e2e_protected_chain.py",
        "scripts/r0_e2e_protected_train_entry.py",
        "scripts/r0_e2e_protected_train.sbatch",
        "scripts/r0_e2e_protected_consolidate.sbatch",
        "scripts/r0_e2e_protected_eval_seed.sbatch",
        "scripts/r0_e2e_protected_control.sbatch",
        "configs/base.yaml",
        "configs/r0a.yaml",
        "configs/r0a_dual_code.yaml",
        "configs/r0a_dual_code_formal.yaml",
        "configs/r0a_protected_common.yaml",
        "configs/r0a_protected_h.yaml",
        "configs/r0a_protected_p.yaml",
        "configs/r0a_protected_i.yaml",
    )
)))

_BASE_ASSERT_PLAN = base._assert_plan
_ACTIVE_PROFILE: ArmProfile | None = None
_CHAIN_PROFILED_ATTRS = (
    "KIND", "PROJECT", "TRAIN_LINKS", "RUN_NAME", "CANONICAL_CONFIG",
    "CANONICAL_CONFIG_SHA256", "EXPECTED_RESOLVED_CONFIG_HASH",
    "EXPECTED_TRAIN_TAGS", "EXPECTED_STAGE_TAGS", "REQUIRED_GROUP_TOKENS",
    "EXPECTED_GROUP", "TRAINING_JOB_TYPE", "TRAIN_ENTRY_SCRIPT",
    "STAGE_SBATCH_FILES", "TRAIN_EXTRA_ENV", "EVAL_ROW_LABEL",
    "EVAL_MARKDOWN_TITLE", "EXPECTED_CONFIG_FIELDS",
    "EXPECTED_METHOD_CONTRACT", "SOURCE_FILES", "_assert_plan",
)
_CHAIN_DEFAULTS = {
    name: copy.deepcopy(getattr(base, name))
    for name in _CHAIN_PROFILED_ATTRS
}


class ProtectedChainError(base.OperatorRepairError):
    """Invalid protected-arm orchestration or selector input."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _profile_table_sha256() -> str:
    rows = {
        arm: {**asdict(profile), "tags": list(profile.tags)}
        for arm, profile in sorted(PROFILES.items())
    }
    return hashlib.sha256(_canonical_json(rows).encode()).hexdigest()


def _sweep_receipt(profile: ArmProfile) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "kind": PROFILE_KIND,
        "arm": profile.arm,
        "profile_table_sha256": _profile_table_sha256(),
        "profile": {
            **asdict(profile),
            "tags": list(profile.tags),
            "config_path": str(profile.config_path),
        },
        "comparison_contract": {
            "arms": ["H", "P", "I"],
            "all_results_published": True,
            "cross_arm_checkpoint_selection": False,
            "outcome_threshold_applied": False,
            "common_initialization": "fresh_seed0",
            "common_fixed_endpoint": FIXED_STEP,
            "common_evaluation_episodes_per_arm": 1_200,
            "common_evaluation_seeds": [0, 1, 2],
            "reference_action_policy": copy.deepcopy(REFERENCE_ACTION_POLICY),
        },
    }


def _method_contract(profile: ArmProfile) -> dict[str, Any]:
    return {
        "fresh_loom_modules": True,
        "frozen_siglip_cached_tower": True,
        "dual_action_mode": "dual_q_action_proposal",
        "fixed_endpoint": FIXED_STEP,
        "endpoint_predeclared_before_training": True,
        "health_metrics_role": "observational_only",
        "metrics_ledger": {
            "format": "loom-fresh-metrics-rollback-v1",
            "reconcile_crash_tail_to_latest_checkpoint": True,
            "checkpoint_boundary_fsync": True,
            "direct_formal_decisions": False,
        },
        "protected_action_profile": {
            "format": PROFILE_KIND,
            "arm": profile.arm,
            "parent": profile.parent,
            "method_delta": profile.method_delta,
            "dynamics_coefficient_source": "q_delta",
            "q_action_receives_dynamics_gradient": False,
            "isolate_estimator_gradients": profile.isolate_estimator_gradients,
            "reference_action_successes": 550,
            "reference_action_episodes": 1_200,
        },
    }


def _assert_protected_plan(plan: dict[str, Any]) -> None:
    _BASE_ASSERT_PLAN(plan)
    profile = _ACTIVE_PROFILE
    if profile is None:
        raise ProtectedChainError("protected profile was not activated")
    if plan.get("protected_sweep") != _sweep_receipt(profile):
        raise ProtectedChainError("protected sweep profile receipt changed")


def activate_profile(arm: str) -> ArmProfile:
    """Bind one allowlisted profile before calling the base-chain machinery."""
    global _ACTIVE_PROFILE
    profile = PROFILES.get(str(arm).upper())
    if profile is None:
        raise ProtectedChainError("protected arm must be exactly one of H/P/I")
    if (
        not profile.config_path.is_file()
        or profile.config_path.is_symlink()
        or base.sha256_file(profile.config_path) != profile.config_sha256
    ):
        raise ProtectedChainError(f"arm {profile.arm} canonical config identity changed")
    base.KIND = KIND
    base.PROJECT = PROJECT
    base.TRAIN_LINKS = TRAIN_LINKS
    base.RUN_NAME = profile.run_name
    base.CANONICAL_CONFIG = profile.config_path
    base.CANONICAL_CONFIG_SHA256 = profile.config_sha256
    base.EXPECTED_RESOLVED_CONFIG_HASH = profile.resolved_config_hash
    base.EXPECTED_TRAIN_TAGS = profile.tags
    base.EXPECTED_STAGE_TAGS = list(profile.tags)
    base.REQUIRED_GROUP_TOKENS = (
        "protected", "fixed32k", f"arm-{profile.arm.lower()}",
    )
    base.EXPECTED_GROUP = profile.group
    base.TRAINING_JOB_TYPE = f"protected-arm-{profile.arm.lower()}-train"
    base.TRAIN_ENTRY_SCRIPT = "scripts/r0_e2e_protected_train_entry.py"
    base.STAGE_SBATCH_FILES = {
        "train": "scripts/r0_e2e_protected_train.sbatch",
        "consolidate": "scripts/r0_e2e_protected_consolidate.sbatch",
        "eval": "scripts/r0_e2e_protected_eval_seed.sbatch",
        "control": "scripts/r0_e2e_protected_control.sbatch",
    }
    base.TRAIN_EXTRA_ENV = {"LOOM_PROTECTED_ARM": profile.arm}
    base.EVAL_ROW_LABEL = f"**LOOM · protected arm {profile.arm}**"
    base.EVAL_MARKDOWN_TITLE = (
        f"# R0 protected-action arm {profile.arm} fixed-step evaluation"
    )
    base.EXPECTED_CONFIG_FIELDS = {
        "reference_action_policy": copy.deepcopy(REFERENCE_ACTION_POLICY),
        "protected_protocol": copy.deepcopy(PROTECTED_PROTOCOL),
        "protected_arm": profile.protected_arm,
    }
    base.EXPECTED_METHOD_CONTRACT = _method_contract(profile)
    base.SOURCE_FILES = PROTECTED_SOURCE_FILES
    base._assert_plan = _assert_protected_plan
    _ACTIVE_PROFILE = profile
    return profile


def _restore_base_defaults_for_tests() -> None:
    """Restore import-time defaults so profile tests cannot contaminate legacy tests."""
    global _ACTIVE_PROFILE
    for name, value in _CHAIN_DEFAULTS.items():
        setattr(base, name, copy.deepcopy(value))
    _ACTIVE_PROFILE = None


def build_plan(
    *, arm: str, run_dir: Path, control_dir: Path, artifact_root: Path,
) -> dict[str, Any]:
    profile = activate_profile(arm)
    plan = base.build_plan(
        run_dir=run_dir, control_dir=control_dir, artifact_root=artifact_root,
        group=profile.group, project=PROJECT,
    )
    plan["protected_sweep"] = _sweep_receipt(profile)
    base._assert_plan(plan)
    return plan


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtectedChainError(f"duplicate JSON key in selector input: {key}")
        result[key] = value
    return result


def select_profile_from_environment() -> ArmProfile:
    """Read only a closed arm selector before normal SHA/plan authentication."""
    raw_path = os.environ.get("OPERATOR_REPAIR_PLAN", "")
    expected_sha = os.environ.get("OPERATOR_REPAIR_PLAN_SHA256", "")
    if not raw_path or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None:
        raise ProtectedChainError("protected run-stage requires an absolute plan and SHA-256")
    path = Path(raw_path)
    if not path.is_absolute() or path.name != "plan.json":
        raise ProtectedChainError("selector plan path must be absolute and end in plan.json")
    try:
        mode = os.lstat(path).st_mode
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ProtectedChainError("selector plan is absent") from exc
    if (
        not stat.S_ISREG(mode)
        or resolved != path
        or not resolved.is_relative_to((ROOT / "runs").resolve())
        or not 0 < resolved.stat().st_size <= PLAN_SELECTOR_MAX_BYTES
    ):
        raise ProtectedChainError("selector plan must be a bounded real file below ROOT/runs")
    payload = resolved.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha:
        raise ProtectedChainError("selector plan SHA-256 mismatch")
    try:
        document = json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtectedChainError("selector plan is not canonical JSON") from exc
    if not isinstance(document, dict) or document.get("kind") != KIND:
        raise ProtectedChainError("selector plan kind changed")
    receipt = document.get("protected_sweep")
    config = document.get("config")
    if not isinstance(receipt, dict) or not isinstance(config, dict):
        raise ProtectedChainError("selector plan lacks protected profile/config objects")
    arm = receipt.get("arm")
    profile = PROFILES.get(arm) if isinstance(arm, str) else None
    if (
        profile is None
        or receipt.get("profile_table_sha256") != _profile_table_sha256()
        or config.get("path") != str(profile.config_path)
    ):
        raise ProtectedChainError("selector did not name one closed H/P/I profile")
    return activate_profile(profile.arm)


def run_environment_stage() -> int:
    select_profile_from_environment()
    # The base loader now verifies the caller-provided SHA, complete source and
    # config closure, the exact protected receipt, assets, and environment.
    return base.run_environment_stage()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    submit = subparsers.add_parser("submit")
    submit.add_argument("--arm", required=True, choices=sorted(PROFILES))
    submit.add_argument("--run-dir", required=True)
    submit.add_argument("--control-dir", required=True)
    submit.add_argument("--artifact-root", required=True)
    submit.add_argument("--dry-run", action="store_true")
    subparsers.add_parser("run-stage")
    args = parser.parse_args(argv)
    try:
        if args.command == "run-stage":
            return run_environment_stage()
        plan = build_plan(
            arm=args.arm, run_dir=Path(args.run_dir),
            control_dir=Path(args.control_dir),
            artifact_root=Path(args.artifact_root),
        )
        if args.dry_run:
            print(json.dumps(base._dry_run_payload(plan), indent=2, sort_keys=True))
            return 0
        receipt = base.submit_plan(plan)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    except (
        ProtectedChainError, base.OperatorRepairError,
        base.common.ChainError, subprocess.CalledProcessError,
    ) as exc:
        print(f"PROTECTED_CHAIN_INVALID: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
