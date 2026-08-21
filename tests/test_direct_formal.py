"""Tests for the isolated direct-formal schedule and stopping contract."""

from __future__ import annotations

import json
import math
import pathlib
import subprocess
import sys

import pytest

from loom.train.direct_formal import (
    DirectFormalGate,
    DirectFormalSchedule,
    evaluate_direct_formal,
    next_direct_formal_check,
    receipt_exit_code,
    should_evaluate_direct_formal,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts" / "direct_formal_convergence.py"


def _gate() -> DirectFormalGate:
    return DirectFormalGate(
        schedule_horizon=8,
        max_updates=12,
        first_check=8,
        check_every=1,
        block_size=2,
        block_count=4,
        reference_start_exclusive=6,
        reference_end_inclusive=8,
    )


def _rows(count: int = 12) -> list[dict]:
    return [
        {
            "global_step": step,
            "loss/dyn": 1.0,
            "act/decode": 0.3,  # compatibility/logging only; never a gate input
            "act/decode_teacher": 0.3,
            "act/decode_deploy": 0.3,
            "act/align": 0.2,
            "loss/proposal": 1.0,
            "delta_op": 0.1,
            "delta_sel/h1": 0.2,
            "delta_sel/h2": 0.2,
            "delta_sel/h3": 0.2,
            "delta_sel/h4": 0.2,
            "bank/live_ops_q_a": 20,
            "bank/live_ops_q_delta": 20,
            "act/c_delta_spread": 0.2,
            "gnorm/estimator": 0.1,
            "gnorm/bank": 0.1,
            "gnorm/q_delta": 0.1,
            "gnorm/q_action": 0.1,
            "gnorm/decoder": 0.1,
            "gnorm/proposal": 0.1,
            "grad_skipped": 0,
            "embodiment": "test",
        }
        for step in range(1, count + 1)
    ]


class _Optimizer:
    def __init__(self):
        self.param_groups = [
            {"name": "estimator/decay", "lr_scale": 1.0, "lr": 999.0},
            {"name": "bank/decay", "lr_scale": 0.1, "lr": 999.0},
            {"name": "proposal/decay", "lr_scale": 0.3, "lr": 999.0},
        ]


def test_schedule_separates_decay_horizon_from_hard_cap_and_preserves_scales():
    schedule = DirectFormalSchedule(
        base_lr=3.0e-4,
        warmup_steps=2,
        schedule_horizon=8,
        max_updates=12,
        min_lr_ratio=0.05,
    )
    assert schedule.scale_at(0) == 0.5
    assert schedule.scale_at(1) == 1.0
    assert schedule.scale_at(2) == 1.0
    assert schedule.scale_at(8) == 0.05
    assert schedule.scale_at(11) == 0.05

    optimizer = _Optimizer()
    lrs = schedule.apply(optimizer, 8)
    assert lrs == {
        "estimator/decay": pytest.approx(1.5e-5),
        "bank/decay": pytest.approx(1.5e-6),
        "proposal/decay": pytest.approx(4.5e-6),
    }
    with pytest.raises(ValueError, match="outside max_updates"):
        schedule.apply(optimizer, 12)


def test_schedule_config_and_checkpoint_identity_are_fail_closed():
    cfg = {
        "run": {"steps": 99, "schedule_horizon": 8, "max_updates": 12},
        "optim": {"lr": 3.0e-4, "warmup": 2, "min_lr_ratio": 0.05},
    }
    schedule = DirectFormalSchedule.from_config(cfg)
    assert schedule.schedule_horizon == 8
    assert schedule.max_updates == 12
    schedule.load_state_dict(schedule.state_dict())

    changed = dict(schedule.state_dict(), max_updates=13)
    with pytest.raises(ValueError, match="identity mismatch: max_updates"):
        schedule.load_state_dict(changed)
    with pytest.raises(ValueError, match="missing direct-formal config keys"):
        DirectFormalSchedule.from_config({
            "run": {"steps": 12}, "optim": {"lr": 3.0e-4},
        })


def test_default_contract_is_exact_32k_to_40k_recipe():
    gate = DirectFormalGate()
    assert gate.schedule_horizon == gate.first_check == 32_000
    assert gate.max_updates == 40_000
    assert gate.check_every == 500
    assert (gate.block_count, gate.block_size, gate.tolerance) == (4, 2_000, 0.02)
    assert (gate.reference_start_exclusive, gate.reference_end_inclusive) == (
        30_000, 32_000,
    )
    assert gate.primary_metrics == (
        "loss/dyn", "act/decode_teacher", "act/decode_deploy", "act/align",
        "loss/proposal",
    )
    assert gate.proposal_uniform_ce == math.log(128)
    assert gate.proposal_off_floor_strict_lt == math.log(128) - 0.05
    assert should_evaluate_direct_formal(32_000, gate)
    assert should_evaluate_direct_formal(40_000, gate)
    assert not should_evaluate_direct_formal(32_001, gate)
    assert next_direct_formal_check(31_999, gate) == 32_000
    assert next_direct_formal_check(32_000, gate) == 32_500
    assert next_direct_formal_check(40_000, gate) is None


def test_before_first_check_is_moving_without_peeking_at_gate_metrics():
    rows = [{"global_step": step, "loss/dyn": 99.0} for step in range(1, 8)]
    receipt = evaluate_direct_formal(rows, gate=_gate())
    assert receipt["status"] == "MOVING"
    assert receipt["reason"] == "before_first_check"
    assert receipt["next_check_step"] == 8
    assert receipt["evaluations"] == []


def test_first_passing_checkpoint_is_selected_and_stays_selected():
    receipt = evaluate_direct_formal(_rows(), gate=_gate())
    assert receipt["status"] == "PASS"
    assert receipt["decision_step"] == 8
    assert len(receipt["evaluations"]) == 1
    evidence = receipt["evaluations"][0]
    assert evidence["health"]["passed"]
    assert evidence["nonregression"]["passed"]
    assert evidence["convergence"]["passed"]
    assert [block["window"] if "window" in block else (
        block["start_exclusive"], block["end_inclusive"]
    ) for block in evidence["blocks"]] == [(0, 2), (2, 4), (4, 6), (6, 8)]


def test_moving_extension_selects_first_later_pass():
    rows = _rows()
    for row in rows[:2]:
        row["loss/dyn"] = 0.7
    receipt = evaluate_direct_formal(rows, gate=_gate())
    assert receipt["status"] == "PASS"
    assert receipt["decision_step"] == 10
    assert [item["status"] for item in receipt["evaluations"]] == [
        "MOVING", "MOVING", "PASS",
    ]


@pytest.mark.parametrize(
    "metric,value,failure",
    [
        ("delta_op", 0.01, "delta_op"),
        ("delta_sel/h3", 0.0, "delta_sel/h3"),
        ("act/align", 0.50, "act/align"),
        ("bank/live_ops_q_a", 15.0, "bank/live_ops_q_a"),
        ("bank/live_ops_q_delta", 15.0, "bank/live_ops_q_delta"),
        ("act/c_delta_spread", 0.10, "act/c_delta_spread"),
        ("gnorm/estimator", 0.0, "gnorm/estimator"),
        ("gnorm/bank", 1.0e-4, "gnorm/bank"),
        ("gnorm/q_delta", 1.0e-4, "gnorm/q_delta"),
        ("gnorm/q_action", 0.0, "gnorm/q_action"),
        ("gnorm/decoder", 0.0, "gnorm/decoder"),
        ("gnorm/proposal", 0.0, "gnorm/proposal"),
    ],
)
def test_exact_strict_health_boundaries_abort(metric, value, failure):
    rows = _rows(8)
    for row in rows[-2:]:
        row[metric] = value
    receipt = evaluate_direct_formal(rows, gate=_gate())
    assert receipt["status"] == "ABORT"
    assert receipt["reason"] == "health_gate_failed"
    assert failure in receipt["evaluations"][0]["health"]["failures"]


def test_proposal_off_floor_and_unexpected_gradient_abort():
    rows = _rows(8)
    gate = _gate()
    for row in rows[-2:]:
        row["loss/proposal"] = gate.proposal_off_floor_strict_lt
    receipt = evaluate_direct_formal(rows, gate=gate)
    assert receipt["status"] == "ABORT"
    assert "loss/proposal_off_floor" in receipt["evaluations"][0]["health"]["failures"]

    rows = _rows(8)
    rows[-1]["gnorm/frozen_module"] = 0.1
    receipt = evaluate_direct_formal(rows, gate=gate)
    assert receipt["status"] == "ABORT"
    health = receipt["evaluations"][0]["health"]
    assert health["metrics"]["unexpected_module_gradients"] == ["frozen_module"]


def test_skip_rate_is_strict_and_binary():
    rows = _rows(8)
    gate = DirectFormalGate(
        schedule_horizon=8, max_updates=12, first_check=8, check_every=1,
        block_size=2, block_count=4,
        reference_start_exclusive=6, reference_end_inclusive=8,
        skipped_rate_strict_lt=0.5,
    )
    rows[-2]["grad_skipped"] = 1
    receipt = evaluate_direct_formal(rows, gate=gate)
    assert receipt["status"] == "ABORT"  # 1/2 == 0.5; strict less-than.

    rows[-2]["grad_skipped"] = 0.5
    receipt = evaluate_direct_formal(rows, gate=gate)
    assert receipt["status"] == "INVALID"
    assert "exact 0/1" in receipt["error"]


def test_exact_two_percent_plateau_boundary_passes():
    rows = _rows(8)
    for row in rows[:2]:
        row["loss/dyn"] = 98.0
    for row in rows[2:]:
        row["loss/dyn"] = 100.0
    receipt = evaluate_direct_formal(rows, gate=_gate())
    metric = receipt["evaluations"][0]["convergence"]["metrics"]["loss/dyn"]
    assert metric["relative_range"] == 0.02
    assert metric["passed"]
    assert receipt["status"] == "PASS"


def test_aggregate_decode_is_not_a_gate_but_each_paired_branch_is():
    rows = _rows(8)
    for row in rows:
        row["act/decode"] = 1_000.0 if row["global_step"] % 2 else -1_000.0
    assert evaluate_direct_formal(rows, gate=_gate())["status"] == "PASS"

    for row in rows[:2]:
        row["act/decode_deploy"] = 0.1
    receipt = evaluate_direct_formal(rows, gate=_gate())
    assert receipt["status"] == "MOVING"
    assert receipt["evaluations"][0]["convergence"]["failures"] == [
        "act/decode_deploy",
    ]


def test_exact_two_percent_nonregression_boundary_passes_at_later_check():
    rows = _rows(10)
    for row in rows[:2]:
        row["loss/dyn"] = 80.0
    for row in rows[2:8]:
        row["loss/dyn"] = 100.0
    for row in rows[8:10]:
        row["loss/dyn"] = 102.0
    receipt = evaluate_direct_formal(rows, gate=_gate())
    assert receipt["status"] == "PASS"
    assert receipt["decision_step"] == 10
    metric = receipt["evaluations"][-1]["nonregression"]["metrics"]["loss/dyn"]
    assert metric["relative_worsening"] == 0.02
    assert metric["passed"]


def test_late_regression_aborts_before_a_flatness_decision():
    rows = _rows(10)
    for row in rows[:2]:
        row["loss/dyn"] = 0.7  # keep update 8 moving
    for row in rows[8:10]:
        row["loss/dyn"] = 1.03  # >2% worse than fixed (6, 8] reference
    receipt = evaluate_direct_formal(rows, gate=_gate())
    assert receipt["status"] == "ABORT"
    assert receipt["reason"] == "nonregression_gate_failed"
    assert receipt["decision_step"] == 10


def test_hard_cap_aborts_when_healthy_metrics_are_still_moving():
    rows = _rows(12)
    for row in rows[:6]:
        row["loss/dyn"] = 0.7
    receipt = evaluate_direct_formal(rows, gate=_gate())
    assert receipt["status"] == "ABORT"
    assert receipt["reason"] == "max_updates_without_convergence"
    assert receipt["decision_step"] == 12
    assert receipt["evaluations"][-1]["status"] == "MOVING"


@pytest.mark.parametrize("mutation", ["missing_step", "missing_metric", "nan"])
def test_incomplete_or_nonfinite_evidence_is_invalid(mutation):
    rows = _rows(8)
    if mutation == "missing_step":
        del rows[3]
    elif mutation == "missing_metric":
        del rows[-1]["act/decode_teacher"]
    else:
        rows[-1]["delta_op"] = float("nan")
    receipt = evaluate_direct_formal(rows, current_step=8, gate=_gate())
    assert receipt["status"] == "INVALID"
    assert receipt["reason"] == "invalid_input"
    assert "error" in receipt


def test_receipt_exit_codes_are_exhaustive():
    assert receipt_exit_code({"status": "PASS"}) == 0
    assert receipt_exit_code({"status": "MOVING"}) == 1
    assert receipt_exit_code({"status": "INVALID"}) == 2
    assert receipt_exit_code({"status": "ABORT"}) == 3
    with pytest.raises(ValueError, match="unknown"):
        receipt_exit_code({"status": "FAIL"})


def test_cli_emits_source_identity_and_exclusive_receipt(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    metrics = run / "metrics.jsonl"
    metrics.write_text(json.dumps({"global_step": 1, "loss/dyn": 1.0}) + "\n")
    output = tmp_path / "receipt.json"
    result = subprocess.run(
        [sys.executable, str(CLI), str(run), "--output", str(output)],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["status"] == "MOVING"
    assert receipt["metrics_source"]["sha256"]
    assert json.loads(output.read_text()) == receipt

    second = subprocess.run(
        [sys.executable, str(CLI), str(run), "--output", str(output)],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert second.returncode == 2
    assert json.loads(second.stdout)["status"] == "INVALID"


def test_cli_rejects_unterminated_jsonl(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "metrics.jsonl").write_text('{"global_step":1}')
    result = subprocess.run(
        [sys.executable, str(CLI), str(run)], cwd=ROOT,
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 2
    assert "does not end in a newline" in json.loads(result.stdout)["error"]
