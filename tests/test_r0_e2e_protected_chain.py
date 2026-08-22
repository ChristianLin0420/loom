from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from scripts import r0_e2e_operator_repair_chain as base
from scripts import r0_e2e_protected_chain as protected
from scripts import r0_e2e_protected_train_entry as protected_entry


@pytest.fixture(autouse=True)
def restore_legacy_chain_defaults():
    protected._restore_base_defaults_for_tests()
    yield
    protected._restore_base_defaults_for_tests()


@pytest.fixture(scope="session")
def frozen_plan_inputs():
    return {
        "assets": base._expected_asset_receipt(),
        "baseline": base._baseline_contract(),
        "eval_environment": base._expected_eval_environment_receipt(),
    }


@pytest.fixture
def isolated_inputs(monkeypatch, frozen_plan_inputs):
    monkeypatch.setattr(
        base, "_asset_receipt",
        lambda: copy.deepcopy(frozen_plan_inputs["assets"]),
    )
    monkeypatch.setattr(
        base, "_baseline_contract",
        lambda: copy.deepcopy(frozen_plan_inputs["baseline"]),
    )
    monkeypatch.setattr(
        base, "_eval_environment_receipt",
        lambda: copy.deepcopy(frozen_plan_inputs["eval_environment"]),
    )


@pytest.fixture
def root_factory():
    parents: list[Path] = []

    def make(arm: str) -> tuple[Path, Path, Path]:
        parent = base.ROOT / "runs" / f".pytest_protected_{arm}_{uuid.uuid4().hex}"
        parents.append(parent)
        return parent / "run", parent / "control", parent / "artifacts"

    yield make
    for parent in parents:
        shutil.rmtree(parent, ignore_errors=True)


def _plan(arm: str, roots: tuple[Path, Path, Path]) -> dict:
    return protected.build_plan(
        arm=arm, run_dir=roots[0], control_dir=roots[1], artifact_root=roots[2],
    )


@pytest.mark.parametrize("arm", ["H", "P", "I"])
def test_each_profile_freezes_truthful_config_method_and_online_wandb_identity(
    arm, isolated_inputs, root_factory,
):
    profile = protected.PROFILES[arm]
    plan = _plan(arm, root_factory(arm))

    assert plan["kind"] == protected.KIND
    assert plan["lineage"]["run_name"] == profile.run_name
    assert plan["config"]["path"] == str(profile.config_path)
    assert plan["config"]["raw_sha256"] == profile.config_sha256
    assert plan["config"]["resolved_config_hash"] == profile.resolved_config_hash
    assert plan["config"]["resolved_experiment"]["protected_arm"] == (
        profile.protected_arm
    )
    assert plan["config"]["resolved_experiment"]["protected_protocol"] == (
        protected.PROTECTED_PROTOCOL
    )
    assert plan["method"]["protected_action_profile"] == {
        "format": protected.PROFILE_KIND,
        "arm": arm,
        "parent": profile.parent,
        "method_delta": profile.method_delta,
        "dynamics_coefficient_source": "q_delta",
        "q_action_receives_dynamics_gradient": False,
        "isolate_estimator_gradients": profile.isolate_estimator_gradients,
        "reference_action_successes": 550,
        "reference_action_episodes": 1_200,
    }
    assert plan["wandb"]["project"] == "loom-r0-protected-arms"
    assert plan["wandb"]["group"] == profile.group
    assert plan["wandb"]["tags"] == list(profile.tags)
    assert plan["wandb"]["training_job_type"] == (
        f"protected-arm-{arm.lower()}-train"
    )
    assert plan["protected_sweep"] == protected._sweep_receipt(profile)


def test_three_arms_have_exact_same_data_schedule_and_evaluation_without_gate(
    isolated_inputs, root_factory,
):
    plans = {arm: _plan(arm, root_factory(arm)) for arm in protected.PROFILES}
    data = [plan["config"]["resolved_experiment"]["data"] for plan in plans.values()]
    assert data[0] == data[1] == data[2]
    schedules = [plan["schedule"] for plan in plans.values()]
    evaluations = [plan["evaluation"] for plan in plans.values()]
    assert schedules[0] == schedules[1] == schedules[2]
    assert evaluations[0] == evaluations[1] == evaluations[2]
    for plan in plans.values():
        assert plan["schedule"]["fixed_updates"] == 32_000
        assert plan["schedule"]["links"] == 6
        assert plan["evaluation"]["total_episodes"] == 1_200
        assert plan["evaluation"]["seeds"] == [0, 1, 2]
        assert plan["eligibility"] == {
            "fixed_endpoint_full_run": True,
            "formal_convergence_gate": False,
            "checkpoint_selection_by_metrics_or_evaluation": False,
            "evaluation_unconditional_after_integrity": True,
            "promotion_authority": False,
        }
        comparison = plan["protected_sweep"]["comparison_contract"]
        assert comparison["all_results_published"] is True
        assert comparison["cross_arm_checkpoint_selection"] is False
        assert comparison["outcome_threshold_applied"] is False


def test_profile_dag_uses_only_new_launchers_and_has_no_gate(
    isolated_inputs, root_factory,
):
    plan = _plan("P", root_factory("P"))
    payload = base._dry_run_payload(plan)
    names = [row["name"] for row in payload["dag"]]
    assert names == [
        "train_01", "train_02", "train_03", "train_04", "train_05", "train_06",
        "consolidate", "eval_seed0", "eval_seed1", "eval_seed2", "merge",
    ]
    assert payload["job_count"] == 11
    assert not any("gate" in name or "select" in name for name in names)
    assert all(
        "r0_e2e_protected_" in row["sbatch"] for row in payload["dag"]
    )
    assert payload["dag"][-1]["depends_on"] == [
        "eval_seed0", "eval_seed1", "eval_seed2",
    ]
    assert all("--hold" in command for command in payload["commands"].values())


def test_source_closure_covers_all_profiles_shared_loop_and_versioned_launchers(
    isolated_inputs, root_factory,
):
    files = set(_plan("I", root_factory("I"))["source_closure"]["files"])
    for required in (
        "scripts/r0_e2e_operator_repair_chain.py",
        "scripts/r0_e2e_operator_repair_train_entry.py",
        "scripts/r0_e2e_protected_chain.py",
        "scripts/r0_e2e_protected_train_entry.py",
        "scripts/r0_e2e_protected_train.sbatch",
        "scripts/r0_e2e_protected_consolidate.sbatch",
        "scripts/r0_e2e_protected_eval_seed.sbatch",
        "scripts/r0_e2e_protected_control.sbatch",
        "configs/r0a_dual_code_formal.yaml",
        "configs/r0a_protected_common.yaml",
        "configs/r0a_protected_h.yaml",
        "configs/r0a_protected_p.yaml",
        "configs/r0a_protected_i.yaml",
        "loom/train/loop.py",
    ):
        assert required in files


@pytest.mark.parametrize("arm", ["H", "P", "I"])
def test_training_subprocess_contract_is_arm_specific(arm, monkeypatch):
    monkeypatch.setenv("LOOM_PROTECTED_ARM", arm)
    assert protected_entry._arm_from_environment() == arm
    profile = protected.PROFILES[arm]
    assert protected_entry.ARM_TAGS[arm] == profile.tags
    assert protected_entry.PROJECT == protected.PROJECT


def test_train_entry_executes_as_a_file_without_pythonpath() -> None:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("LOOM_PROTECTED_ARM", None)
    completed = subprocess.run(
        [sys.executable, str(base.ROOT / "scripts/r0_e2e_protected_train_entry.py")],
        cwd=base.ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, check=False,
    )
    assert completed.returncode != 0
    assert "ModuleNotFoundError" not in completed.stderr
    assert "LOOM_PROTECTED_ARM must be exactly one of H/P/I" in completed.stderr


def _publish_selector_plan(plan: dict, control: Path, monkeypatch) -> Path:
    control.mkdir(parents=True)
    path = control / "plan.json"
    path.write_text(json.dumps(plan, sort_keys=True) + "\n")
    monkeypatch.setenv("OPERATOR_REPAIR_PLAN", str(path))
    monkeypatch.setenv(
        "OPERATOR_REPAIR_PLAN_SHA256",
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    return path


def test_untrusted_selector_accepts_only_closed_profile_then_full_asserts(
    isolated_inputs, root_factory, monkeypatch,
):
    roots = root_factory("P")
    plan = _plan("P", roots)
    path = _publish_selector_plan(plan, roots[1], monkeypatch)
    assert protected.select_profile_from_environment().arm == "P"
    loaded = base.load_plan(path, hashlib.sha256(path.read_bytes()).hexdigest())
    assert loaded == plan

    tampered = copy.deepcopy(plan)
    tampered["config"]["path"] = str(protected.PROFILES["I"].config_path)
    path.write_text(json.dumps(tampered, sort_keys=True) + "\n")
    monkeypatch.setenv(
        "OPERATOR_REPAIR_PLAN_SHA256",
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    with pytest.raises(protected.ProtectedChainError, match="closed H/P/I"):
        protected.select_profile_from_environment()


def test_selector_rejects_symlink_outside_root_and_duplicate_keys(
    isolated_inputs, root_factory, monkeypatch, tmp_path,
):
    roots = root_factory("H")
    plan = _plan("H", roots)
    real = _publish_selector_plan(plan, roots[1], monkeypatch)

    alias_dir = roots[0].parent / "selector_alias"
    alias_dir.mkdir()
    link = alias_dir / "plan.json"
    link.symlink_to(real)
    monkeypatch.setenv("OPERATOR_REPAIR_PLAN", str(link))
    with pytest.raises(protected.ProtectedChainError, match="bounded real file"):
        protected.select_profile_from_environment()

    outside = tmp_path / "plan.json"
    outside.write_bytes(real.read_bytes())
    monkeypatch.setenv("OPERATOR_REPAIR_PLAN", str(outside))
    monkeypatch.setenv(
        "OPERATOR_REPAIR_PLAN_SHA256",
        hashlib.sha256(outside.read_bytes()).hexdigest(),
    )
    with pytest.raises(protected.ProtectedChainError, match="below ROOT/runs"):
        protected.select_profile_from_environment()

    duplicate = real.read_text().replace(
        '"kind": "r0_e2e_protected_action_fixed_endpoint_chain_v1",',
        '"kind": "r0_e2e_protected_action_fixed_endpoint_chain_v1", '
        '"kind": "r0_e2e_protected_action_fixed_endpoint_chain_v1",',
        1,
    )
    real.write_text(duplicate)
    monkeypatch.setenv("OPERATOR_REPAIR_PLAN", str(real))
    monkeypatch.setenv(
        "OPERATOR_REPAIR_PLAN_SHA256",
        hashlib.sha256(real.read_bytes()).hexdigest(),
    )
    with pytest.raises(protected.ProtectedChainError, match="duplicate JSON key"):
        protected.select_profile_from_environment()


def test_protected_receipt_mutation_fails_after_base_plan_authentication(
    isolated_inputs, root_factory,
):
    plan = _plan("I", root_factory("I"))
    changed = copy.deepcopy(plan)
    changed["protected_sweep"]["comparison_contract"]["all_results_published"] = False
    with pytest.raises(protected.ProtectedChainError, match="profile receipt changed"):
        base._assert_plan(changed)


def test_profile_preserves_exact_base_libero_package_identity(isolated_inputs):
    expected = (
        "094ab6d50e33f1503b5a2cecdce1e52071c967e8978b1a1db898764deb206344",
        118,
        2_304,
    )
    assert (
        base.LIBERO_EVAL_PIP_FREEZE_SHA256,
        base.LIBERO_EVAL_PIP_FREEZE_LINES,
        base.LIBERO_EVAL_PIP_FREEZE_BYTES,
    ) == expected
    protected.activate_profile("H")
    assert (
        base.LIBERO_EVAL_PIP_FREEZE_SHA256,
        base.LIBERO_EVAL_PIP_FREEZE_LINES,
        base.LIBERO_EVAL_PIP_FREEZE_BYTES,
    ) == expected


def test_legacy_defaults_are_restored_after_profile_use(isolated_inputs):
    protected.activate_profile("H")
    assert base.PROJECT == protected.PROJECT
    protected._restore_base_defaults_for_tests()
    assert base.PROJECT == "loom-r0-operator-repair"
    assert base.CANONICAL_CONFIG.name == "r0a_operator_repair.yaml"
    assert base.EXPECTED_CONFIG_FIELDS["method_receipt"]["kind"] == (
        "loom_r0a_operator_repair_v1"
    )
