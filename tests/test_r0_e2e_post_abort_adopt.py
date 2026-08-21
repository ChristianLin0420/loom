from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts import r0_e2e_formal_chain as formal
from scripts import r0_e2e_post_abort_adopt as adopt
from scripts import r0_e2e_post_abort_eval as v1


ROOT = Path(__file__).resolve().parents[1]


def _actual_blob(seed: int = 0) -> dict:
    return json.loads(adopt._source_result_paths(seed)[0].read_text())


def _plan_without_expensive_io(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict:
    source = v1._load_source_plan()
    prior = adopt._prior_attempt()
    baseline = adopt._expected_baseline(source)
    formal_abort = json.loads((adopt.PRIOR_CONTROL / "plan.json").read_text())[
        "trigger"
    ]
    monkeypatch.setattr(v1, "_load_source_plan", lambda: source)
    monkeypatch.setattr(adopt, "_prior_attempt", lambda: prior)
    monkeypatch.setattr(adopt, "_expected_baseline", lambda _source: baseline)
    monkeypatch.setattr(v1, "_collect_trigger", lambda **_kwargs: formal_abort)
    monkeypatch.setattr(v1, "_assert_trigger", lambda *_args, **_kwargs: None)
    return adopt.build_plan(
        control_dir=(tmp_path / "control").resolve(),
        artifact_root=(tmp_path / "artifacts").resolve(),
        group=adopt.GROUP,
    )


def test_failed_v1_surface_and_outputs_are_exact_and_immutable():
    assert adopt.sha256_file(ROOT / "scripts/r0_e2e_post_abort_eval.py") == (
        "10e2a9be13ca6e235770ba17a6cb92a42522d044778b3f6682a68df2bea46a5f"
    )
    assert v1._source_closure()["sha256"] == (
        "5af2ea2652fd9ab4ce3eed1ff50a2efa1bd7f735afd900ebd51e6ee802552852"
    )
    prior = adopt._prior_attempt()
    assert prior["failure"] == "identity_policy_kw_instrumentation_mismatch"
    assert prior["episode_execution_valid"] is True
    assert prior["receipt_publication_valid"] is False
    assert prior["old_seed_and_merge_receipts_absent"] is True
    assert prior["terminal_job_states"] == adopt.PRIOR_JOB_STATES
    assert set(prior["job_logs"]) == {"consolidate", "seed0", "seed1", "seed2"}


@pytest.mark.parametrize(
    ("seed", "successes", "avg"),
    ((0, 178, 44.5), (1, 180, 45.0), (2, 192, 48.0)),
)
def test_exact_completed_results_authenticate_without_episode_rerun(
    seed: int, successes: int, avg: float,
):
    blob, protocol, rows = adopt._load_validated_result(seed)
    assert len(rows) == 400
    assert protocol.seeds == (seed,)
    assert blob["summary"]["n_errors"] == 0
    assert blob["summary"]["avg"] == avg
    assert sum(row["success"] for row in blob["episodes"]) == successes
    assert blob["meta"]["eval_identity"]["policy_kw"] == {
        "allow_stub": False, "op_stats": True,
    }
    assert blob["meta"]["policy"]["embodiment"] == "libero_franka"


def test_normalization_is_in_memory_only_and_preserves_source_bytes():
    blob = _actual_blob()
    frozen = copy.deepcopy(blob)
    normalized = adopt._normalized_validation_view(blob)
    assert blob == frozen
    assert blob["meta"]["eval_identity"]["policy_kw"] == adopt.RECORDED_POLICY_KW
    assert normalized["meta"]["eval_identity"]["policy_kw"] == (
        formal.CANDIDATE_POLICY_KW
    )
    receipt = adopt._normalization_receipt(blob)
    assert receipt["source_bytes_rewritten"] is False
    assert receipt["normalization_scope"] == "in_memory_validator_view_only"
    assert len(receipt["recorded_eval_identity_sha256"]) == 64
    assert len(receipt["normalized_validation_view_sha256"]) == 64


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda blob: blob["meta"]["eval_identity"]["policy_kw"].update(
            {"embodiment": "libero_franka"}
        ), "historical two-key"),
        (lambda blob: blob["meta"]["policy"].update({"embodiment": "other"}),
         "real-policy provenance"),
        (lambda blob: blob["meta"]["policy"].update({"is_stub": True}),
         "real-policy provenance"),
        (lambda blob: blob["meta"]["policy"].update({"ckpt_global_step": 31_999}),
         "real-policy provenance"),
    ),
)
def test_normalization_rejects_identity_or_policy_mutation(mutation, message):
    blob = _actual_blob()
    mutation(blob)
    with pytest.raises(adopt.AdoptionError, match=message):
        adopt._normalized_validation_view(blob)


def test_plan_is_fresh_nonformal_and_freezes_normalization(monkeypatch, tmp_path):
    plan = _plan_without_expensive_io(monkeypatch, tmp_path)
    assert plan["format_version"] == 2
    assert plan["eligibility"] == adopt.EXPECTED_ELIGIBILITY
    assert plan["method"] == adopt.EXPECTED_METHOD
    assert plan["method"]["environment_episodes_rerun"] == 0
    assert plan["lineage"]["diagnostic_control_dir"] == (
        plan["lineage"]["adoption_control_dir"]
    )
    assert plan["trigger"]["failed_v1_attempt"]["failure"] == (
        "identity_policy_kw_instrumentation_mismatch"
    )
    assert plan["wandb"]["group"] == adopt.GROUP
    assert "not-formal" in plan["wandb"]["tags"]
    assert "adopted-no-rerun" in plan["wandb"]["tags"]
    adopt._assert_plan(plan)


def test_plan_rejects_output_escape_or_formal_group(monkeypatch, tmp_path):
    plan = _plan_without_expensive_io(monkeypatch, tmp_path)
    changed = copy.deepcopy(plan)
    changed["paths"]["merged_results"] = str(
        adopt.PRIOR_ARTIFACT / "forbidden.json"
    )
    with pytest.raises(adopt.AdoptionError, match="output paths changed"):
        adopt._assert_plan(changed)
    changed = copy.deepcopy(plan)
    changed["wandb"]["group"] = json.loads(
        (adopt.PRIOR_CONTROL / "plan.json").read_text()
    )["wandb"]["group"]
    with pytest.raises(adopt.AdoptionError, match="W&B identity changed"):
        adopt._assert_plan(changed)


@pytest.mark.parametrize(
    ("control", "artifact", "message"),
    (
        ("root", "root/artifacts", "mutually disjoint"),
        ("root/control", "root", "mutually disjoint"),
        ("control,bad", "artifacts", "unsafe Slurm"),
        ("control", "artifacts\nbad", "unsafe Slurm"),
    ),
)
def test_output_roots_reject_nesting_and_export_delimiters(
    tmp_path, control, artifact, message,
):
    with pytest.raises(adopt.AdoptionError, match=message):
        adopt._require_isolated(
            (tmp_path / control).resolve(), (tmp_path / artifact).resolve(),
        )


def test_checkpoint_stage_runs_verify_only_and_never_reconsolidates(
    monkeypatch, tmp_path,
):
    report = tmp_path / "control/report.json"
    receipt = tmp_path / "control/receipt.json"
    report.parent.mkdir(parents=True)
    plan = {"paths": {
        "checkpoint_report": str(report), "checkpoint_receipt": str(receipt),
    }}
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(list(command))
        Path(command[command.index("--report") + 1]).write_bytes(
            (adopt.PRIOR_CONTROL / "checkpoint_verification.json").read_bytes()
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(adopt, "_assert_trigger", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(adopt.subprocess, "run", fake_run)
    monkeypatch.setattr(adopt, "_checkpoint_receipt_payload", lambda _plan: {
        "format_version": 2, "kind": adopt.CHECKPOINT_KIND,
    })
    monkeypatch.setattr(adopt, "_validate_checkpoint_receipt", lambda _plan: {})
    monkeypatch.setattr(v1, "_wandb_publish", lambda *_args, **_kwargs: None)
    assert adopt._stage_adopt_checkpoint(plan) == 0
    joined = " ".join(calls[0])
    assert "-m loom.train.consolidate" in joined
    assert "--verify_only" in calls[0]
    assert "--pin" not in calls[0]
    assert "--no_verify" not in calls[0]


def test_checkpoint_stage_ignores_orphan_attempt_but_rejects_corrupt_canonical(
    monkeypatch, tmp_path,
):
    report = tmp_path / "control/report.json"
    receipt = tmp_path / "control/receipt.json"
    report.parent.mkdir(parents=True)
    (report.parent / f".{report.name}.attempt-orphan").write_text("partial")
    plan = {"paths": {
        "checkpoint_report": str(report), "checkpoint_receipt": str(receipt),
    }}

    def fake_run(command, **_kwargs):
        Path(command[command.index("--report") + 1]).write_bytes(
            (adopt.PRIOR_CONTROL / "checkpoint_verification.json").read_bytes()
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(adopt, "_assert_trigger", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(adopt.subprocess, "run", fake_run)
    monkeypatch.setattr(adopt, "_checkpoint_receipt_payload", lambda _plan: {
        "format_version": 2, "kind": adopt.CHECKPOINT_KIND,
    })
    monkeypatch.setattr(adopt, "_validate_checkpoint_receipt", lambda _plan: {})
    monkeypatch.setattr(v1, "_wandb_publish", lambda *_args, **_kwargs: None)
    assert adopt._stage_adopt_checkpoint(plan) == 0
    assert adopt.sha256_file(report) == adopt.PRIOR_CHECKPOINT_REPORT_SHA256

    report.write_text("corrupt canonical")
    receipt.unlink()
    monkeypatch.setattr(
        adopt.subprocess, "run",
        lambda *_args, **_kwargs: pytest.fail("canonical tamper must not be replaced"),
    )
    with pytest.raises(adopt.AdoptionError, match="differs from v1 report"):
        adopt._stage_adopt_checkpoint(plan)


def test_seed_stage_adopts_complete_result_without_subprocess_or_rewrite(
    monkeypatch, tmp_path,
):
    checkpoint_receipt = tmp_path / "checkpoint.json"
    checkpoint_receipt.write_text("checkpoint\n")
    seed_receipt = tmp_path / "seed.json"
    plan_file = tmp_path / "plan.json"
    plan_file.write_text("{}\n")
    plan = {"paths": {
        "checkpoint_receipt": str(checkpoint_receipt),
        "eval_receipts": {"0": str(seed_receipt)},
    }}
    before = [
        adopt.sha256_file(path)
        for path in adopt._source_result_paths(0)
    ]
    monkeypatch.setenv("ADOPTION_PLAN", str(plan_file))
    monkeypatch.setattr(adopt, "_assert_trigger", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(adopt, "_validate_checkpoint_receipt", lambda _plan: {})
    monkeypatch.setattr(v1, "_wandb_publish", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        adopt.subprocess, "run",
        lambda *_args, **_kwargs: pytest.fail("adoption must not run an episode"),
    )
    assert adopt._stage_adopt_seed(plan, "adopt_seed0") == 0
    receipt = json.loads(seed_receipt.read_text())
    assert receipt["completed_result_adopted"] is True
    assert receipt["environment_episodes_rerun"] == 0
    assert receipt["identity_normalization"]["source_bytes_rewritten"] is False
    assert before == [adopt.sha256_file(path) for path in adopt._source_result_paths(0)]


def _merge_plan(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict:
    plan_file = tmp_path / "plan.json"
    plan_file.write_text("{}\n")
    monkeypatch.setenv("ADOPTION_PLAN", str(plan_file))
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text("checkpoint\n")
    plan = {
        "baseline_comparison": adopt._expected_baseline(v1._load_source_plan()),
        "diagnostic_source_closure": {"sha256": "a" * 64},
        "paths": {"checkpoint_receipt": str(checkpoint), "eval_receipts": {}},
    }
    monkeypatch.setattr(adopt, "_validate_checkpoint_receipt", lambda _plan: {})
    for seed in formal.SEEDS:
        path = tmp_path / f"seed_{seed}_receipt.json"
        plan["paths"]["eval_receipts"][str(seed)] = str(path)
        blob, _protocol, _rows = adopt._load_validated_result(seed)
        path.write_text(json.dumps(adopt._seed_receipt_payload(plan, seed, blob)))
    return plan


def test_exact_adopted_merge_oracle_and_counterfactual_failure(monkeypatch, tmp_path):
    merged = adopt.merge_seed_results(_merge_plan(monkeypatch, tmp_path))
    summary = merged["summary"]
    comparison = merged["diagnostic_baseline_comparison"]
    assert summary["n_episodes"] == 1_200
    assert summary["n_errors"] == 0
    assert summary["avg"] == pytest.approx(45.833333333333336)
    assert comparison["overall"] == {
        "candidate_successes": 550,
        "baseline_successes": 447,
        "episodes": 1_200,
        "candidate_success_rate_percent": pytest.approx(45.833333333333336),
        "baseline_success_rate_percent": 37.25,
        "delta_percentage_points": pytest.approx(8.583333333333334),
    }
    bounds = comparison["paired_task_bootstrap"]
    assert bounds["ci_low_percentage_points"] == pytest.approx(1.75)
    assert bounds["ci_high_percentage_points"] == pytest.approx(15.5)
    assert comparison["per_suite"]["libero_long"][
        "candidate_success_rate_percent"
    ] == 13.0
    assert comparison["status"] == "FAIL"
    assert comparison["passed"] is False
    assert comparison["failed_checks"] == ["suite_floor/libero_long"]
    assert comparison["cannot_reverse_formal_abort"] is True


def test_stage_graph_is_verify_three_parallel_adopts_and_merge():
    specs = adopt._stage_specs()
    assert [row["name"] for row in specs] == [
        "adopt_checkpoint", "adopt_seed0", "adopt_seed1", "adopt_seed2",
        "merge_adopted",
    ]
    assert specs[0]["depends_on"] == []
    assert all(row["depends_on"] == ["adopt_checkpoint"] for row in specs[1:4])
    assert specs[4]["depends_on"] == ["adopt_seed0", "adopt_seed1", "adopt_seed2"]


def test_submit_is_held_atomic_release_and_exact_dependencies(monkeypatch, tmp_path):
    plan = _plan_without_expensive_io(monkeypatch, tmp_path)
    monkeypatch.setattr(adopt, "_assert_plan", lambda _plan: None)
    calls: list[list[str]] = []
    ids = iter(("201", "202", "203", "204", "205"))

    def fake_run(command, **_kwargs):
        calls.append(list(command))
        if command[0] == "sbatch":
            return subprocess.CompletedProcess(command, 0, stdout=next(ids), stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    result = adopt.submit_plan(plan, run=fake_run)
    sbatch = [call for call in calls if call[0] == "sbatch"]
    assert len(sbatch) == 5
    assert not any(arg.startswith("--dependency=") for arg in sbatch[0])
    assert all("--dependency=afterok:201" in call for call in sbatch[1:4])
    assert "--dependency=afterok:202:203:204" in sbatch[4]
    assert calls[-1] == ["scontrol", "release", "201,202,203,204,205"]
    assert result["released"] is True


def test_static_surface_has_no_eval_or_training_execution():
    source = (ROOT / "scripts/r0_e2e_post_abort_adopt.py").read_text()
    assert "_formal_eval_command" not in source
    assert '"-m", "loom.eval"' not in source
    assert "def _stage_train" not in source
    assert "optimizer.step" not in source
    assert "environment_episodes_rerun\": 0" in source
    assert '"--verify_only"' in source
    for name in (
        "r0_e2e_post_abort_adopt_checkpoint.sbatch",
        "r0_e2e_post_abort_adopt_seed.sbatch",
        "r0_e2e_post_abort_adopt_control.sbatch",
    ):
        sbatch = (ROOT / "scripts" / name).read_text()
        assert "r0_e2e_post_abort_adopt.py run-stage" in sbatch
        assert "loom.eval" not in sbatch


def test_new_source_closure_is_isolated_and_v1_closure_remains_bound():
    closure = adopt._source_closure()
    assert set(closure["files"]) == set(adopt.SOURCE_FILES)
    assert closure["scheme"] == "sha256(path-nul-sha256-nul)-v1"
    assert len(closure["sha256"]) == 64
    assert v1._source_closure()["sha256"] == (
        "5af2ea2652fd9ab4ce3eed1ff50a2efa1bd7f735afd900ebd51e6ee802552852"
    )
