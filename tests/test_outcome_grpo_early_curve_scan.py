"""CPU contracts for the read-only retrospective early-curve scan."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from loom.train import outcome_grpo as grpo
from scripts import outcome_grpo_early_curve_scan as scan


def _surrogate(step: int, *, gain: float, kl: float = 0.001):
    groups = []
    for index in range(grpo.EXPECTED_VALIDATION_GROUPS):
        task = index // 10
        groups.append({
            "index": index,
            "group_id": f"group-{index:03d}",
            "task": f"suite/task={task:02d}",
            "informative": True,
            "surrogate": gain,
            "approx_kl": kl,
        })
    return {
        "n_groups": grpo.EXPECTED_VALIDATION_GROUPS,
        "n_tasks": grpo.EXPECTED_VALIDATION_TASKS,
        "informative_groups": grpo.EXPECTED_VALIDATION_GROUPS,
        "mean_surrogate": gain,
        "mean_approx_kl": kl,
        "max_abs_logratio": 0.1 + (step - grpo.START_STEP) / 100_000,
        "groups": groups,
    }


def _trust(*, clip_fraction: float = 0.1):
    values = {
        "clip_fraction": clip_fraction,
        "ess_fraction": 0.9,
        "coeff_drift_p95": 0.04,
        "live_ops": 32,
        "nonfinite": 0,
        "unexpected_gradients": 0,
    }
    checks = {}
    for name, (op, threshold) in scan._TRUST_CHECKS.items():
        value = values[name]
        checks[name] = {
            "value": value,
            "op": op,
            "threshold": threshold,
            "pass": scan._check_threshold(value, op, threshold),
        }
    return {
        "passed": all(row["pass"] for row in checks.values()),
        "max_abs_logratio": 0.5,
        "checks": checks,
        "counts": {"ratio_atoms": 100, "arm0_drift_atoms": 20},
    }


def test_task_mapping_is_exactly_one_surrogate_and_trust_per_fixed_step():
    assert scan.EARLY_CHECKPOINT_STEPS == (49_866, 50_066, 50_466, 51_266)
    assert [
        scan.scan_task_assignment(rank, scan.EXPECTED_WORLD_SIZE)
        for rank in range(scan.EXPECTED_WORLD_SIZE)
    ] == [
        ("surrogate", 49_866), ("trust", 49_866),
        ("surrogate", 50_066), ("trust", 50_066),
        ("surrogate", 50_466), ("trust", 50_466),
        ("surrogate", 51_266), ("trust", 51_266),
    ]
    with pytest.raises(scan.EarlyCurveScanError, match="requires world=8"):
        scan.scan_task_assignment(0, 4)
    with pytest.raises(ValueError, match="outside world"):
        scan.scan_task_assignment(8, 8)


def test_seed_paired_aggregation_is_fixed_order_deterministic_and_ineligible():
    seed = _surrogate(grpo.START_STEP, gain=0.0, kl=0.0)
    gains = {
        49_866: 0.001,
        50_066: 0.002,
        50_466: 0.003,
        51_266: 0.004,
    }
    surrogates = {step: _surrogate(step, gain=gain)
                  for step, gain in gains.items()}
    trusts = {step: _trust() for step in scan.EARLY_CHECKPOINT_STEPS}

    first = scan.aggregate_seed_paired_results(seed, surrogates, trusts)
    second = scan.aggregate_seed_paired_results(seed, surrogates, trusts)

    assert first == second
    assert list(first["steps"]) == ["49866", "50066", "50466", "51266"]
    assert first["n_steps_passing_all_diagnostic_gates"] == 4
    assert first["selection"] == {
        "performed": False,
        "permitted": False,
        "best_checkpoint": None,
        "reason": (
            "post-hoc retrospective results may describe the failed curve "
            "but may not select or promote a checkpoint"
        ),
    }
    for step, gain in gains.items():
        row = first["steps"][str(step)]
        paired = row["seed_paired_bootstrap"]
        assert paired["comparison"] == f"{step}-terminal_report_seed_49666"
        assert paired["estimate"] == pytest.approx(gain)
        assert paired["ci_low"] == pytest.approx(gain)
        assert paired["ci_high"] == pytest.approx(gain)
        assert paired["samples"] == 2_000 and paired["seed"] == 0
        assert paired["pass"] is True
        assert row["diagnostic_gate_passed"] is True
        assert row["eligibility"] == "INELIGIBLE"


def test_aggregation_keeps_failed_kl_and_trust_as_diagnostics_not_selection():
    seed = _surrogate(grpo.START_STEP, gain=0.0, kl=0.0)
    surrogates = {
        step: _surrogate(
            step,
            gain=0.01,
            kl=0.02 if step == 50_066 else 0.001,
        )
        for step in scan.EARLY_CHECKPOINT_STEPS
    }
    trusts = {step: _trust() for step in scan.EARLY_CHECKPOINT_STEPS}
    trusts[50_466] = _trust(clip_fraction=0.25)

    report = scan.aggregate_seed_paired_results(seed, surrogates, trusts)

    assert report["steps"]["50066"]["checks"]["heldout_approx_kl"]["pass"] is False
    assert report["steps"]["50466"]["checks"]["trust/clip_fraction"]["pass"] is False
    assert report["steps"]["50066"]["diagnostic_gate_passed"] is False
    assert report["steps"]["50466"]["diagnostic_gate_passed"] is False
    assert report["n_steps_passing_all_diagnostic_gates"] == 2
    assert report["selection"]["performed"] is False
    assert report["selection"]["best_checkpoint"] is None


def test_aggregation_rejects_any_seed_candidate_pairing_drift():
    seed = _surrogate(grpo.START_STEP, gain=0.0, kl=0.0)
    surrogates = {
        step: _surrogate(step, gain=0.01)
        for step in scan.EARLY_CHECKPOINT_STEPS
    }
    trusts = {step: _trust() for step in scan.EARLY_CHECKPOINT_STEPS}
    surrogates[49_866]["groups"][7]["group_id"] = "changed"
    with pytest.raises(scan.EarlyCurveScanError, match="order/identity differs"):
        scan.aggregate_seed_paired_results(seed, surrogates, trusts)


def test_exclusive_output_refuses_overwrite_and_parser_has_no_scan_tuning(tmp_path):
    path = tmp_path / "summary.json"
    scan.exclusive_json_write(path, {"first": True})
    assert json.loads(path.read_text()) == {"first": True}
    with pytest.raises(scan.EarlyCurveScanError, match="refusing to overwrite"):
        scan.exclusive_json_write(path, {"second": True})
    assert json.loads(path.read_text()) == {"first": True}

    actions = {action.dest for action in scan.build_parser()._actions}
    assert actions == {"help", "out"}
    assert not {"steps", "checkpoint", "threshold", "bootstrap_seed"} & actions

    with pytest.raises(scan.EarlyCurveScanError, match="must be directly inside"):
        scan._validate_output_path(tmp_path / "elsewhere.json")


def test_launcher_is_eight_rank_read_only_and_has_no_submission_or_promotion():
    source = Path("scripts/outcome_grpo_early_curve_scan.sbatch").read_text()
    assert "#SBATCH --gpus-per-node=8" in source
    assert "#SBATCH --ntasks-per-node=8" in source
    assert "#SBATCH --time=02:00:00" in source
    assert 'RANK="$SLURM_PROCID"' in source
    assert "optimizer_steps=0" in source
    assert "simulator_episodes=0" in source
    assert "promotion=INELIGIBLE" in source
    assert "sbatch --dependency" not in source
    assert "scancel" not in source
