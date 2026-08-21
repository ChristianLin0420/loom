"""Stage-aware checks for scripts/convergence.py."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "convergence.py"
AUTOSTOP = ROOT / "scripts" / "autostop.sh"


def _write_metrics(run: Path, *, base: int = 0, count: int = 4,
                   proposal: float | None = 1.6) -> None:
    run.mkdir(exist_ok=True)
    rows = []
    for step in range(1, count + 1):
        late = step > 2
        row = {
            "global_step": base + step,
            "loss": 10.0,
            "loss/dyn": 1.0,
            "loss/act": 1.0,
            "act/decode": 0.1,
            # Deliberately moving, but frozen in r0a_deploy.
            "act/align": 0.2 if late else 0.1,
            # Deliberately on the legacy phase-clock floor, but frozen here.
            "delta_sel": 0.0,
            "act/deploy_c_l2": 0.05,
        }
        if proposal is not None:
            row["loss/proposal"] = proposal
        rows.append(row)
    (run / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )


def _run(run: Path, *args: str, short_window: bool = True) -> subprocess.CompletedProcess[str]:
    window = ["--block", "2", "--blocks", "2"] if short_window else []
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(run), *window, *args],
        text=True, capture_output=True, check=False,
    )


def test_legacy_gate_still_judges_alignment_and_delta_sel(tmp_path):
    run = tmp_path / "legacy"
    _write_metrics(run)

    result = _run(run)

    assert result.returncode == 1
    assert "NOT_CONVERGED" in result.stdout
    assert "act/align" in result.stdout


def test_deploy_yaml_ignores_frozen_alignment_and_delta_sel_floor(tmp_path):
    run = tmp_path / "deploy"
    _write_metrics(run, base=27000)

    result = _run(run, "--config", str(ROOT / "configs" / "r0a_deploy.yaml"))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "VERDICT: CONVERGED" in result.stdout
    assert "act/decode" in result.stdout
    assert "loss/proposal" in result.stdout
    assert "act/deploy_c_l2" in result.stdout
    assert "act/align" not in result.stdout
    assert "delta_sel" not in result.stdout


def test_resolved_run_config_selects_deploy_gate_automatically(tmp_path):
    run = tmp_path / "deploy"
    _write_metrics(run)
    (run / "config.json").write_text(json.dumps({
        "convergence": {
            "primary": ["act/decode", "loss/proposal"],
            "watch": ["act/deploy_c_l2"],
            "floor_checks": ["loss/proposal"],
        },
    }))

    result = _run(run)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "VERDICT: CONVERGED" in result.stdout


def test_deploy_requires_the_complete_configured_post_27k_window(tmp_path):
    run = tmp_path / "deploy"
    config = ("--config", str(ROOT / "configs" / "r0a_deploy.yaml"))
    _write_metrics(run, base=27000, count=7999)

    early = _run(run, *config, short_window=False)
    assert early.returncode == 1
    assert "need 4 complete 2000-step blocks" in early.stdout

    _write_metrics(run, base=27000, count=8000)
    ready = _run(run, *config, short_window=False)
    assert ready.returncode == 0, ready.stdout + ready.stderr


def test_sparse_ce_uniform_floor_is_degenerate(tmp_path):
    run = tmp_path / "deploy"
    _write_metrics(run, base=27000, proposal=math.log(128))

    result = _run(run, "--config", str(ROOT / "configs" / "r0a_deploy.yaml"))

    assert result.returncode == 3
    assert "uniform sparse CE 4.852" in result.stdout


def test_dense_kl_does_not_reuse_a_categorical_floor(tmp_path):
    run = tmp_path / "dense"
    _write_metrics(run, proposal=20.0)
    (run / "config.json").write_text(json.dumps({
        "losses": {"proposal": {"mode": "dense_kl"}},
        "convergence": {
            "primary": ["act/decode", "loss/proposal"],
            "watch": [],
            "floor_checks": ["loss/proposal"],
            "block": 2,
            "blocks": 2,
        },
    }))

    result = _run(run)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "VERDICT: CONVERGED" in result.stdout


def test_missing_configured_primary_cannot_pass(tmp_path):
    run = tmp_path / "deploy"
    _write_metrics(run, base=27000, proposal=None)

    result = _run(run, "--config", str(ROOT / "configs" / "r0a_deploy.yaml"))

    assert result.returncode == 1
    assert "missing configured primary metrics: loss/proposal" in result.stdout


def _stage_config(*, efficacy: bool = False) -> dict:
    cfg = {
        "run": {"name": "qa-test"},
        "losses": {"proposal": {"mode": "sparse_ce"}},
        "train_modules": ["bank", "q_action"],
        "convergence": {
            "start_step": 0,
            "block": 2,
            "blocks": 2,
            "tol": 0.02,
            "primary": ["loss/dyn", "act/decode"],
            "watch": [],
            "floor_checks": [],
        },
        "liveness_gate": {
            "start_exclusive": 0,
            "end_inclusive": 4,
            "rows": 4,
            "requirements": {
                "delta_op_median_strict_gt": 0.01,
                "gnorm_bank_median_strict_gt": 1.0e-4,
                "gnorm_q_action_median_strict_gt": 1.0e-4,
                "skipped_rate_strict_lt": 0.01,
                "unexpected_module_gradients": False,
                "nonfinite": False,
            },
            "required": True,
        },
    }
    if efficacy:
        cfg["efficacy_gate"] = {
            "metric": "act/decode",
            "reference": "first_post_start_block",
            "comparison": "final_convergence_block",
            "max_relative_worsening": 0.0,
            "required": True,
        }
    return cfg


def _write_stage_run(run: Path, *, efficacy: bool = False,
                     final_decode: float = 0.1) -> None:
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(_stage_config(efficacy=efficacy)))
    rows = []
    for step in range(1, 5):
        rows.append({
            "global_step": step,
            "loss/dyn": 1.0,
            "act/decode": 0.1 if step <= 2 else final_decode,
            "delta_op": 0.02,
            "gnorm/bank": 0.1,
            "gnorm/q_action": 0.2,
            "grad_skipped": 0,
        })
    (run / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )


def test_required_liveness_passes_exact_window_and_persists_json(tmp_path):
    run = tmp_path / "qa"
    _write_stage_run(run)

    result = _run(run, short_window=False)

    assert result.returncode == 0, result.stdout + result.stderr
    artifact = json.loads((run / "liveness_000000_000004.json").read_text())
    assert artifact["status"] == "PASS"
    assert artifact["window"] == {
        "lo_exclusive": 0, "hi_inclusive": 4, "rows": 4, "contiguous": True,
    }
    assert artifact["metrics"]["delta_op_median"] == pytest.approx(0.02)


@pytest.mark.parametrize("mutation, message", [
    ("threshold", "delta_op"),
    ("missing_row", "exact/contiguous"),
    ("unexpected_gradient", "unexpected_gradients"),
    ("nonfinite", "nonfinite"),
])
def test_required_liveness_failures_are_fail_closed(tmp_path, mutation, message):
    run = tmp_path / mutation
    _write_stage_run(run)
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    if mutation == "threshold":
        for row in rows:
            row["delta_op"] = 0.01  # strict >, equality must fail
    elif mutation == "missing_row":
        del rows[2]
    elif mutation == "unexpected_gradient":
        rows[0]["gnorm/decoder"] = 0.1
    else:
        rows[0]["loss/dyn"] = float("nan")
    (run / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )

    result = _run(run, short_window=False)

    assert result.returncode == 4
    assert "LIVENESS_FAILED" in result.stdout
    assert message in result.stdout
    artifact = json.loads((run / "liveness_000000_000004.json").read_text())
    assert artifact["status"] == "FAIL"
    assert artifact["passed"] is False


def test_efficacy_is_enforced_only_after_convergence(tmp_path):
    failed = tmp_path / "efficacy-fail"
    _write_stage_run(failed, efficacy=True, final_decode=0.1005)
    fail_result = _run(failed, short_window=False)
    assert fail_result.returncode == 4
    assert "EFFICACY_FAILED" in fail_result.stdout

    passed = tmp_path / "efficacy-pass"
    _write_stage_run(passed, efficacy=True, final_decode=0.099)
    pass_result = _run(passed, short_window=False)
    assert pass_result.returncode == 0, pass_result.stdout + pass_result.stderr
    assert "EFFICACY: PASS" in pass_result.stdout


def test_unterminated_final_metrics_record_is_retryable_not_a_gate_failure(tmp_path):
    run = tmp_path / "partial"
    _write_stage_run(run)
    with open(run / "metrics.jsonl", "a") as stream:
        stream.write('{"global_step": 5, "delta_op":')

    result = _run(run, short_window=False)

    assert result.returncode == 2
    assert "transient metrics read" in result.stdout
    assert "LIVENESS_FAILED" not in result.stdout


def test_malformed_complete_metrics_record_is_fail_closed_and_persisted(tmp_path):
    run = tmp_path / "malformed"
    _write_stage_run(run)
    with open(run / "metrics.jsonl", "a") as stream:
        stream.write('{"global_step": 5, "delta_op":}\n')

    result = _run(run, short_window=False)

    assert result.returncode == 4
    assert "REQUIRED_GATE_INVALID" in result.stdout
    artifact = json.loads((run / "liveness_000000_000004.json").read_text())
    assert artifact["status"] == "FAIL"
    assert "malformed complete metrics" in artifact["failures"][0]


def test_autostop_checks_liveness_before_min_and_writes_stop(tmp_path):
    run_root = tmp_path / "runs"
    run = run_root / "qa"
    _write_stage_run(run)
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    for row in rows:
        row["delta_op"] = 0.0
    (run / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    (run / "HEARTBEAT").write_text("0 4 0\n")
    env = dict(os.environ, LOOM_RUN_ROOT=str(run_root))

    result = subprocess.run(
        ["bash", str(AUTOSTOP), "qa", "999999", "300"], cwd=ROOT,
        env=env, text=True, capture_output=True, check=False, timeout=10,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "required stage gate FAILED" in result.stdout
    assert (run / "STOP").exists()


def test_autostop_does_not_stop_on_transient_config_read(tmp_path):
    run_root = tmp_path / "runs"
    run = run_root / "qa"
    run.mkdir(parents=True)
    (run / "config.json").write_text('{"liveness_gate":')
    (run / "metrics.jsonl").write_text('{"global_step": 4}\n')
    (run / "HEARTBEAT").write_text("0 4 0\n")
    env = dict(os.environ, LOOM_RUN_ROOT=str(run_root))

    result = subprocess.run(
        ["bash", str(AUTOSTOP), "qa", "999999", "300"], cwd=ROOT,
        env=env, text=True, capture_output=True, check=False, timeout=10,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert not (run / "STOP").exists()


def _phase_a_config() -> dict:
    return {
        "run": {"name": "fresh-phase-a-test"},
        "train_modules": ["estimator", "q_delta"],
        "phase_gate": {
            "start_exclusive": 0,
            "end_inclusive": 4,
            "rows": 4,
            "requirements": {
                "act_align_median_strict_lt": 0.4,
                "c_a_spread_median_strict_gt": 0.1,
                "c_delta_spread_median_strict_gt": 0.1,
                "live_ops_q_a_median_gte": 16,
                "live_ops_q_delta_median_gte": 16,
                "proposal_loss_median_strict_lt": math.log(128),
                "skipped_rate_strict_lt": 0.25,
                "bank_gradients": False,
                "expected_module_gradients": True,
                "nonfinite": False,
            },
            "required": True,
        },
    }


def _phase_a_rows() -> list[dict]:
    return [{
        "global_step": step,
        "act/align": 0.2,
        "act/c_a_spread": 0.2,
        "act/c_delta_spread": 0.2,
        # Equality is intentional: both requirements are >=, not strict >.
        "bank/live_ops_q_a": 16,
        "bank/live_ops_q_delta": 16,
        "loss/proposal": 4.0,
        "grad_skipped": 0,
        "gnorm/estimator": 0.1,
        "gnorm/q_delta": 0.2,
    } for step in range(1, 5)]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _write_phase_a_run(run: Path) -> None:
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(_phase_a_config()))
    _write_jsonl(run / "metrics.jsonl", _phase_a_rows())


def test_phase_a_pass_is_atomic_artifact_and_never_terminal_convergence(tmp_path):
    run = tmp_path / "phase-a-pass"
    _write_phase_a_run(run)

    result = _run(run, short_window=False)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "PHASE_GATE: PASS" in result.stdout
    assert "CONVERGED" not in result.stdout
    artifact_path = run / "phase_gate_000000_000004.json"
    artifact = json.loads(artifact_path.read_text())
    assert artifact["status"] == "PASS" and artifact["passed"] is True
    assert artifact["window"] == {
        "lo_exclusive": 0, "hi_inclusive": 4, "rows": 4, "contiguous": True,
    }
    assert artifact["metrics"]["live_ops_q_a_median"] == 16
    assert not list(run.glob(".phase_gate_*.tmp.*"))


@pytest.mark.parametrize(("mutation", "failure"), [
    ("strict_lt_equal", "act/align"),
    ("strict_gt_equal", "act/c_a_spread"),
    ("skip_equal", "grad_skipped"),
    ("bank_gradient", "bank_gradients"),
    ("missing_expected_gradient", "gnorm/q_delta"),
    ("missing_row", "exact/contiguous"),
    ("nonfinite", "nonfinite"),
])
def test_phase_a_threshold_and_completeness_failures_are_artifacted(
    tmp_path, mutation, failure,
):
    run = tmp_path / mutation
    _write_phase_a_run(run)
    rows = _phase_a_rows()
    if mutation == "strict_lt_equal":
        for row in rows:
            row["act/align"] = 0.4
    elif mutation == "strict_gt_equal":
        for row in rows:
            row["act/c_a_spread"] = 0.1
    elif mutation == "skip_equal":
        rows[0]["grad_skipped"] = 1  # 1/4 == threshold; strict < must fail.
    elif mutation == "bank_gradient":
        rows[0]["gnorm/bank"] = 0.1
    elif mutation == "missing_expected_gradient":
        rows[0].pop("gnorm/q_delta")
    elif mutation == "missing_row":
        rows.pop(2)
    else:
        rows[0]["act/align"] = float("nan")
    _write_jsonl(run / "metrics.jsonl", rows)

    result = _run(run, short_window=False)

    assert result.returncode == 4, result.stdout + result.stderr
    assert "PHASE_GATE_FAILED" in result.stdout
    assert failure in result.stdout
    artifact = json.loads((run / "phase_gate_000000_000004.json").read_text())
    assert artifact["status"] == "FAIL" and artifact["passed"] is False
    assert not list(run.glob(".phase_gate_*.tmp.*"))


def test_phase_a_schema_is_exact_and_fails_closed(tmp_path):
    run = tmp_path / "phase-schema"
    _write_phase_a_run(run)
    cfg = _phase_a_config()
    cfg["phase_gate"]["requirements"]["unreviewed_knob"] = 1
    (run / "config.json").write_text(json.dumps(cfg))

    result = _run(run, short_window=False)

    assert result.returncode == 4
    assert "REQUIRED_GATE_INVALID" in result.stdout
    assert "unknown=['unreviewed_knob']" in result.stdout


def test_phase_a_unterminated_writer_race_is_retryable_without_artifact(tmp_path):
    run = tmp_path / "phase-race"
    _write_phase_a_run(run)
    with open(run / "metrics.jsonl", "a") as stream:
        stream.write('{"global_step": 5, "act/align":')

    result = _run(run, short_window=False)

    assert result.returncode == 2
    assert "transient metrics read" in result.stdout
    assert not (run / "phase_gate_000000_000004.json").exists()


def test_phase_a_malformed_complete_row_fails_closed_with_artifact(tmp_path):
    run = tmp_path / "phase-malformed"
    _write_phase_a_run(run)
    with open(run / "metrics.jsonl", "a") as stream:
        stream.write('{"global_step": 5, "act/align":}\n')

    result = _run(run, short_window=False)

    assert result.returncode == 4
    artifact = json.loads((run / "phase_gate_000000_000004.json").read_text())
    assert artifact["status"] == "FAIL"
    assert "malformed complete metrics" in artifact["failures"][0]


def test_autostop_never_writes_stop_for_phase_a_pass(tmp_path):
    run_root = tmp_path / "runs"
    run = run_root / "fresh"
    _write_phase_a_run(run)
    (run / "HEARTBEAT").write_text("0 4 0\n")
    env = dict(os.environ, LOOM_RUN_ROOT=str(run_root))

    result = subprocess.run(
        ["bash", str(AUTOSTOP), "fresh", "0", "300"], cwd=ROOT,
        env=env, text=True, capture_output=True, check=False, timeout=10,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert not (run / "STOP").exists()
    assert json.loads((run / "phase_gate_000000_000004.json").read_text())[
        "status"
    ] == "PASS"


def _fresh_phase_b_config(*, liveness: bool = True,
                          terminal: bool = False) -> dict:
    cfg = {
        "run": {"name": "fresh-phase-b-test"},
        "losses": {"proposal": {"mode": "sparse_ce"}},
        "train_modules": ["bank", "q_delta"],
        "convergence": {
            "start_step": 0,
            "block": 2,
            "blocks": 2,
            "tol": 0.02,
            "primary": ["loss/dyn", "act/decode"],
            "watch": [],
            "floor_checks": [],
        },
    }
    if liveness:
        cfg["liveness_gate"] = {
            "start_exclusive": 0,
            "end_inclusive": 4,
            "rows": 4,
            "requirements": {
                "delta_op_median_strict_gt": 0.01,
                "delta_sel_horizon_medians_strict_gt": 0.0,
                "live_ops_q_delta_median_gte": 16,
                "c_delta_spread_median_strict_gt": 0.1,
                "gnorm_bank_median_strict_gt": 1.0e-4,
                "gnorm_q_delta_median_strict_gt": 1.0e-4,
                "skipped_rate_strict_lt": 0.25,
                "unexpected_module_gradients": False,
                "nonfinite": False,
            },
            "required": True,
        }
    if terminal:
        cfg["terminal_gate"] = {
            "window_start_exclusive": 0,
            "window_end_inclusive": 4,
            "requirements": {
                "delta_op_median_strict_gt": 0.01,
                "delta_sel_horizon_medians_strict_gt": 0.0,
                "act_align_median_strict_lt": 0.5,
                "live_ops_q_a_median_gte": 16,
                "live_ops_q_delta_median_gte": 16,
                "proposal_loss_median_strict_lt": math.log(128),
                "skipped_rate_strict_lt": 0.25,
            },
            "required": True,
        }
    return cfg


def _fresh_phase_b_rows() -> list[dict]:
    return [{
        "global_step": step,
        "loss/dyn": 1.0,
        "act/decode": 0.1,
        "delta_op": 0.02,
        **{f"delta_sel/h{h}": 0.1 for h in range(1, 5)},
        "act/align": 0.2,
        "act/c_delta_spread": 0.2,
        "bank/live_ops_q_a": 16,
        "bank/live_ops_q_delta": 16,
        "loss/proposal": 4.0,
        "gnorm/bank": 0.1,
        "gnorm/q_delta": 0.2,
        "grad_skipped": 0,
    } for step in range(1, 5)]


def _write_fresh_phase_b_run(run: Path, *, liveness=True, terminal=False) -> None:
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(
        _fresh_phase_b_config(liveness=liveness, terminal=terminal)
    ))
    _write_jsonl(run / "metrics.jsonl", _fresh_phase_b_rows())


def test_phase_b_liveness_exact_schema_passes_all_horizons(tmp_path):
    run = tmp_path / "fresh-liveness"
    _write_fresh_phase_b_run(run)

    result = _run(run, short_window=False)

    assert result.returncode == 0, result.stdout + result.stderr
    artifact = json.loads((run / "liveness_000000_000004.json").read_text())
    assert artifact["status"] == "PASS"
    assert artifact["metrics"]["delta_sel_horizon_medians"] == {
        "h1": 0.1, "h2": 0.1, "h3": 0.1, "h4": 0.1,
    }
    assert artifact["metrics"]["live_ops_q_delta_median"] == 16


@pytest.mark.parametrize(("mutation", "failure"), [
    ("horizon_equal", "delta_sel/h3"),
    ("spread_equal", "act/c_delta_spread"),
    ("unexpected_gradient", "unexpected_gradients"),
])
def test_phase_b_liveness_strict_thresholds_fail_closed(tmp_path, mutation, failure):
    run = tmp_path / mutation
    _write_fresh_phase_b_run(run)
    rows = _fresh_phase_b_rows()
    if mutation == "horizon_equal":
        for row in rows:
            row["delta_sel/h3"] = 0.0
    elif mutation == "spread_equal":
        for row in rows:
            row["act/c_delta_spread"] = 0.1
    else:
        rows[0]["gnorm/decoder"] = 0.1
    _write_jsonl(run / "metrics.jsonl", rows)

    result = _run(run, short_window=False)

    assert result.returncode == 4
    assert failure in result.stdout
    assert json.loads((run / "liveness_000000_000004.json").read_text())[
        "status"
    ] == "FAIL"


def test_phase_b_liveness_schema_rejects_partial_horizon_protocol(tmp_path):
    run = tmp_path / "fresh-liveness-schema"
    _write_fresh_phase_b_run(run)
    cfg = _fresh_phase_b_config()
    del cfg["liveness_gate"]["requirements"]["live_ops_q_delta_median_gte"]
    (run / "config.json").write_text(json.dumps(cfg))

    result = _run(run, short_window=False)

    assert result.returncode == 4
    assert "REQUIRED_GATE_INVALID" in result.stdout


def test_terminal_gate_passes_before_converged_and_is_artifacted(tmp_path):
    run = tmp_path / "terminal-pass"
    _write_fresh_phase_b_run(run, liveness=False, terminal=True)

    result = _run(run, short_window=False)

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.index("TERMINAL_GATE: PASS") < result.stdout.index(
        "VERDICT: CONVERGED"
    )
    artifact = json.loads((run / "terminal_gate_000000_000004.json").read_text())
    assert artifact["status"] == "PASS"
    assert artifact["metrics"]["live_ops_q_a_median"] == 16


@pytest.mark.parametrize(("mutation", "failure"), [
    ("delta_equal", "delta_op"),
    ("horizon_equal", "delta_sel/h2"),
    ("align_equal", "act/align"),
    ("proposal_equal", "loss/proposal"),
    ("missing_row", "exact/contiguous"),
    ("nonfinite", "nonfinite"),
])
def test_terminal_gate_thresholds_and_completeness_fail_closed(
    tmp_path, mutation, failure,
):
    run = tmp_path / f"terminal-{mutation}"
    _write_fresh_phase_b_run(run, liveness=False, terminal=True)
    rows = _fresh_phase_b_rows()
    if mutation == "delta_equal":
        for row in rows:
            row["delta_op"] = 0.01
    elif mutation == "horizon_equal":
        for row in rows:
            row["delta_sel/h2"] = 0.0
    elif mutation == "align_equal":
        for row in rows:
            row["act/align"] = 0.5
    elif mutation == "proposal_equal":
        for row in rows:
            row["loss/proposal"] = math.log(128)
    elif mutation == "missing_row":
        rows.pop(1)
    else:
        rows[0]["delta_op"] = float("inf")
    _write_jsonl(run / "metrics.jsonl", rows)

    result = _run(run, short_window=False)

    assert result.returncode == 4
    assert failure in result.stdout
    artifact = json.loads((run / "terminal_gate_000000_000004.json").read_text())
    assert artifact["status"] == "FAIL"
    assert not list(run.glob(".terminal_gate_*.tmp.*"))


def _raw_required_gate(config_name: str, key: str) -> dict:
    cfg = yaml.safe_load((ROOT / "configs" / config_name).read_text())
    return cfg[key]


def test_fresh_gate_thresholds_are_locked_to_the_declared_recipes():
    phase_a = _raw_required_gate("r0a_fresh_phase_a.yaml", "phase_gate")
    assert phase_a == {
        "start_exclusive": 1000,
        "end_inclusive": 2000,
        "rows": 1000,
        "requirements": {
            "act_align_median_strict_lt": 0.40,
            "c_a_spread_median_strict_gt": 0.10,
            "c_delta_spread_median_strict_gt": 0.10,
            "live_ops_q_a_median_gte": 16,
            "live_ops_q_delta_median_gte": 16,
            "proposal_loss_median_strict_lt": 4.852030263919617,
            "skipped_rate_strict_lt": 0.15,
            "bank_gradients": False,
            "expected_module_gradients": True,
            "nonfinite": False,
        },
        "required": True,
    }

    phase_b_liveness = _raw_required_gate(
        "r0a_fresh_phase_b.yaml", "liveness_gate",
    )
    assert phase_b_liveness == {
        "start_exclusive": 2000,
        "end_inclusive": 4000,
        "rows": 2000,
        "requirements": {
            "delta_op_median_strict_gt": 0.01,
            "delta_sel_horizon_medians_strict_gt": 0.0,
            "live_ops_q_delta_median_gte": 16,
            "c_delta_spread_median_strict_gt": 0.10,
            "gnorm_bank_median_strict_gt": 1.0e-4,
            "gnorm_q_delta_median_strict_gt": 1.0e-4,
            "skipped_rate_strict_lt": 0.15,
            "unexpected_module_gradients": False,
            "nonfinite": False,
        },
        "required": True,
    }

    phase_b_terminal = _raw_required_gate(
        "r0a_fresh_phase_b.yaml", "terminal_gate",
    )
    assert phase_b_terminal == {
        "window_start_exclusive": 30000,
        "window_end_inclusive": 32000,
        "requirements": {
            "delta_op_median_strict_gt": 0.01,
            "delta_sel_horizon_medians_strict_gt": 0.0,
            "act_align_median_strict_lt": 0.50,
            "live_ops_q_a_median_gte": 16,
            "live_ops_q_delta_median_gte": 16,
            "proposal_loss_median_strict_lt": 4.852030263919617,
            "skipped_rate_strict_lt": 0.01,
        },
        "required": True,
    }


@pytest.mark.parametrize(("config_name", "expected_hash"), [
    ("r0a_fresh_phase_a.yaml", "3de2324b2e369c20"),
    ("r0a_fresh_phase_b.yaml", "89ba97103054518b"),
    ("r0a_bank_ca_qa.yaml", "0ec8af0a26135ecc"),
])
def test_gate_artifact_identity_matches_training_config_hash(
    tmp_path, config_name, expected_hash,
):
    from loom.train.loop import config_hash, read_config
    from scripts import convergence

    cfg = read_config(ROOT / "configs" / config_name)
    assert config_hash(cfg) == expected_hash
    run = tmp_path / config_name.removesuffix(".yaml")
    run.mkdir()
    (run / "config.json").write_text(json.dumps(cfg))

    gate = convergence.gate_config(run)

    assert gate["identity"] == {
        "run_name": cfg["run"]["name"],
        "config_hash": expected_hash,
    }
    if config_name == "r0a_bank_ca_qa.yaml":
        # Keep the deployed QA parser route and its exact thresholds unchanged.
        assert gate["liveness"] == {
            "start": 50666,
            "end": 52666,
            "rows": 2000,
            "delta_op": 0.01,
            "gnorm/bank": 1.0e-4,
            "gnorm/q_action": 1.0e-4,
            "skipped_rate": 0.01,
        }


def _fresh_convergence_config() -> dict:
    return {
        "start_step": 0,
        "block": 2,
        "blocks": 2,
        "tol": 0.02,
        "primary": ["loss/dyn"],
        "watch": [],
        "floor_checks": [],
    }


def _write_exact_phase_a_run(run: Path, *, mutation: str | None = None) -> None:
    run.mkdir(parents=True, exist_ok=True)
    cfg = {
        "run": {"name": "fresh-phase-a-test"},
        "losses": {"proposal": {"mode": "sparse_ce"}},
        "train_modules": ["estimator", "q_delta"],
        "convergence": _fresh_convergence_config(),
        "phase_gate": _raw_required_gate("r0a_fresh_phase_a.yaml", "phase_gate"),
    }
    cfg["phase_gate"] = json.loads(json.dumps(cfg["phase_gate"]))
    cfg["phase_gate"].update({
        "start_exclusive": 0, "end_inclusive": 4, "rows": 4,
    })
    (run / "config.json").write_text(json.dumps(cfg))
    req = cfg["phase_gate"]["requirements"]
    rows = []
    for step in range(1, 5):
        rows.append({
            "global_step": step,
            "loss/dyn": 1.0,
            "act/align": 0.2,
            "act/c_a_spread": 0.2,
            "act/c_delta_spread": 0.2,
            "bank/live_ops_q_a": 20.0,
            "bank/live_ops_q_delta": 20.0,
            "loss/proposal": 1.0,
            "grad_skipped": 0,
            "gnorm/estimator": 0.1,
            "gnorm/q_delta": 0.2,
        })
    if mutation == "strict_equality":
        for row in rows:
            row["act/align"] = req["act_align_median_strict_lt"]
    elif mutation == "bank_gradient":
        rows[0]["gnorm/bank"] = 0.1
    elif mutation == "missing_expected_gradient":
        del rows[0]["gnorm/q_delta"]
    (run / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )


def test_fresh_phase_a_gate_passes_exact_window_and_persists_artifact(tmp_path):
    run = tmp_path / "phase-a"
    _write_exact_phase_a_run(run)

    result = _run(run, short_window=False)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "PHASE_GATE: PASS — hand off to phase B; do not write STOP" in result.stdout
    artifact = json.loads((run / "phase_gate_000000_000004.json").read_text())
    assert artifact["status"] == "PASS"
    assert artifact["requirements"] == _raw_required_gate(
        "r0a_fresh_phase_a.yaml", "phase_gate",
    )["requirements"]
    assert artifact["metrics"]["unexpected_module_gradients"] == []


@pytest.mark.parametrize("mutation, failure", [
    ("strict_equality", "act/align"),
    ("bank_gradient", "bank_gradients"),
    ("missing_expected_gradient", "gnorm/q_delta"),
])
def test_fresh_phase_a_gate_is_fail_closed(tmp_path, mutation, failure):
    run = tmp_path / mutation
    _write_exact_phase_a_run(run, mutation=mutation)

    result = _run(run, short_window=False)

    assert result.returncode == 4
    assert "PHASE_GATE_FAILED" in result.stdout
    assert failure in result.stdout
    artifact = json.loads((run / "phase_gate_000000_000004.json").read_text())
    assert artifact["status"] == "FAIL"


def _write_phase_b_run(run: Path, *, liveness_mutation: str | None = None,
                       terminal_mutation: str | None = None) -> None:
    run.mkdir(parents=True, exist_ok=True)
    live = _raw_required_gate("r0a_fresh_phase_b.yaml", "liveness_gate")
    terminal = _raw_required_gate("r0a_fresh_phase_b.yaml", "terminal_gate")
    live = json.loads(json.dumps(live))
    terminal = json.loads(json.dumps(terminal))
    live.update({"start_exclusive": 0, "end_inclusive": 4, "rows": 4})
    terminal.update({"window_start_exclusive": 4, "window_end_inclusive": 8})
    cfg = {
        "run": {"name": "fresh-phase-b-test"},
        "losses": {"proposal": {"mode": "sparse_ce"}},
        "train_modules": ["bank", "q_delta"],
        "convergence": _fresh_convergence_config(),
        "liveness_gate": live,
        "terminal_gate": terminal,
    }
    (run / "config.json").write_text(json.dumps(cfg))
    rows = []
    for step in range(1, 9):
        rows.append({
            "global_step": step,
            "loss/dyn": 1.0,
            "delta_op": 0.02,
            "delta_sel/h1": 0.02,
            "delta_sel/h2": 0.02,
            "delta_sel/h3": 0.02,
            "delta_sel/h4": 0.02,
            "bank/live_ops_q_a": 20.0,
            "bank/live_ops_q_delta": 20.0,
            "act/c_delta_spread": 0.2,
            "act/align": 0.2,
            "loss/proposal": 1.0,
            "gnorm/bank": 0.1,
            "gnorm/q_delta": 0.2,
            "grad_skipped": 0,
        })
    if liveness_mutation == "delta_sel_equality":
        threshold = live["requirements"]["delta_sel_horizon_medians_strict_gt"]
        for row in rows[:4]:
            row["delta_sel/h2"] = threshold
    elif liveness_mutation == "unexpected_gradient":
        rows[0]["gnorm/decoder"] = 0.1
    if terminal_mutation == "proposal_equality":
        threshold = terminal["requirements"]["proposal_loss_median_strict_lt"]
        for row in rows[4:]:
            row["loss/proposal"] = threshold
    elif terminal_mutation == "missing_row":
        del rows[6]
    (run / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )


def test_fresh_phase_b_liveness_and_terminal_gates_pass_exact_windows(tmp_path):
    run = tmp_path / "phase-b"
    _write_phase_b_run(run)

    result = _run(run, short_window=False)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "LIVENESS: PASS" in result.stdout
    assert "TERMINAL_GATE: PASS" in result.stdout
    live = json.loads((run / "liveness_000000_000004.json").read_text())
    terminal = json.loads((run / "terminal_gate_000004_000008.json").read_text())
    assert live["requirements"] == _raw_required_gate(
        "r0a_fresh_phase_b.yaml", "liveness_gate",
    )["requirements"]
    assert terminal["requirements"] == _raw_required_gate(
        "r0a_fresh_phase_b.yaml", "terminal_gate",
    )["requirements"]


@pytest.mark.parametrize("mutation, failure", [
    ("delta_sel_equality", "delta_sel/h2"),
    ("unexpected_gradient", "unexpected_gradients"),
])
def test_fresh_phase_b_liveness_is_fail_closed(tmp_path, mutation, failure):
    run = tmp_path / mutation
    _write_phase_b_run(run, liveness_mutation=mutation)

    result = _run(run, short_window=False)

    assert result.returncode == 4
    assert "LIVENESS_FAILED" in result.stdout
    assert failure in result.stdout
    artifact = json.loads((run / "liveness_000000_000004.json").read_text())
    assert artifact["status"] == "FAIL"


@pytest.mark.parametrize("mutation, failure", [
    ("proposal_equality", "loss/proposal"),
    ("missing_row", "exact/contiguous"),
])
def test_fresh_phase_b_terminal_gate_is_fail_closed(tmp_path, mutation, failure):
    run = tmp_path / mutation
    _write_phase_b_run(run, terminal_mutation=mutation)

    result = _run(run, short_window=False)

    assert result.returncode == 4
    assert "TERMINAL_GATE_FAILED" in result.stdout
    assert failure in result.stdout
    artifact = json.loads((run / "terminal_gate_000004_000008.json").read_text())
    assert artifact["status"] == "FAIL"


def test_autostop_checks_phase_gate_before_min_and_writes_stop(tmp_path):
    run_root = tmp_path / "runs"
    run = run_root / "phase-a"
    _write_exact_phase_a_run(run, mutation="strict_equality")
    (run / "HEARTBEAT").write_text("0 4 0\n")
    env = dict(os.environ, LOOM_RUN_ROOT=str(run_root))

    result = subprocess.run(
        ["bash", str(AUTOSTOP), "phase-a", "999999", "300"], cwd=ROOT,
        env=env, text=True, capture_output=True, check=False, timeout=10,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "required stage gate FAILED" in result.stdout
    assert (run / "STOP").exists()


def test_autostop_never_turns_phase_a_pass_into_stop(tmp_path):
    run_root = tmp_path / "runs"
    run = run_root / "phase-a"
    _write_exact_phase_a_run(run)
    (run / "HEARTBEAT").write_text("0 4 0\n")
    env = dict(os.environ, LOOM_RUN_ROOT=str(run_root))

    result = subprocess.run(
        ["bash", str(AUTOSTOP), "phase-a", "4", "300"], cwd=ROOT,
        env=env, text=True, capture_output=True, check=False, timeout=10,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "run finished on its own" in result.stdout
    assert not (run / "STOP").exists()
