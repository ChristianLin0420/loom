from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts import r0_e2e_operator_repair_chain as operator
from scripts import r0_e2e_operator_repair_eval_recovery as recovery


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def source_plan() -> dict:
    return recovery._load_source_plan()


@pytest.fixture(scope="module")
def source_trigger() -> dict:
    return recovery._collect_source_trigger(rehash_shards=False)


def _fast_plan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, source_trigger: dict,
) -> dict:
    frozen = copy.deepcopy(source_trigger)
    # Production roots are deliberately restricted to ROOT/runs.  Unit-test
    # roots live below pytest's /tmp tree; containment itself is covered by the
    # dedicated isolation tests and the authenticated real dry-run.
    monkeypatch.setattr(
        operator, "_reject_existing_symlink_components", lambda _path: None,
    )
    monkeypatch.setattr(
        recovery, "_collect_source_trigger", lambda **_kwargs: copy.deepcopy(frozen),
    )
    return recovery.build_plan(
        control_dir=(tmp_path / "control").resolve(),
        artifact_root=(tmp_path / "artifacts").resolve(),
        group=recovery.GROUP,
    )


def _materialize_plan(plan: dict) -> tuple[Path, str]:
    control = Path(plan["lineage"]["control_dir"])
    artifact = Path(plan["lineage"]["artifact_root"])
    control.mkdir(parents=True)
    artifact.mkdir(parents=True)
    path = control / "plan.json"
    operator.common.exclusive_json_write(path, plan)
    digest = recovery.sha256_file(path)
    os.environ["OPERATOR_RECOVERY_PLAN"] = str(path)
    os.environ["OPERATOR_RECOVERY_PLAN_SHA256"] = digest
    os.environ["OPERATOR_REPAIR_PLAN"] = str(path)
    os.environ["OPERATOR_REPAIR_PLAN_SHA256"] = digest
    return path, digest


def test_historical_plan_and_all_53_git_blobs_are_exact(source_plan):
    assert recovery.sha256_file(recovery.SOURCE_PLAN) == recovery.SOURCE_PLAN_SHA256
    assert source_plan["source_closure"]["sha256"] == (
        recovery.HISTORICAL_SOURCE_CLOSURE_SHA256
    )
    receipt = recovery._historical_git_closure(source_plan)
    assert receipt == {
        "kind": "git_authenticated_historical_operator_repair_closure_v1",
        "commit": recovery.HISTORICAL_GIT_COMMIT,
        "tree": recovery.HISTORICAL_GIT_TREE,
        "source_closure_sha256": recovery.HISTORICAL_SOURCE_CLOSURE_SHA256,
        "files": 53,
        "all_plan_file_hashes_match_git": True,
    }


def test_runtime_diff_is_only_unpadded_validator_repair(source_plan):
    receipt = recovery._validator_repair_receipt(source_plan)
    assert receipt["historical_operator_sha256"] == (
        recovery.HISTORICAL_OPERATOR_SHA256
    )
    assert receipt["changed_lines"] == 1
    assert receipt["all_other_historical_closure_files_byte_identical"] is True
    assert receipt["training_or_model_semantics_changed"] is False
    assert receipt["historical_expected_template"].endswith("{rank:05d}.pt")
    assert receipt["producer_and_runtime_template"].endswith("{rank}.pt")


def test_historical_failure_is_after_real_checkpoint_pass(source_trigger):
    assert source_trigger["fixed_endpoint_identity"]["sha256"] == (
        recovery.SOURCE_ENDPOINT_SHA256
    )
    assert set(source_trigger["fixed_endpoint"]["checkpoint_shards"]) == {
        f"ckpt_000032000_rank{rank}.pt" for rank in range(16)
    }
    assert source_trigger["checkpoint_report"]["pass"] is True
    assert source_trigger["checkpoint"]["sha256"] == (
        recovery.SOURCE_CHECKPOINT_SHA256
    )
    assert source_trigger["checkpoint"]["global_step"] == 32_000
    assert source_trigger["checkpoint"]["consolidated"]["n_shards"] == 16
    assert source_trigger["prior_failure"][
        "checkpoint_verification_pass_preceded_validator_failure"
    ] is True
    assert source_trigger["scheduler"]["recorded_terminal_states"]["32651394"] == (
        "FAILED_2:0_AFTER_CHECKPOINT_VERIFY_PASS"
    )
    assert all(not Path(path).exists() for path in recovery.SOURCE_PROTECTED_ABSENT)


def test_dag_is_verify_only_three_parallel_seeds_and_merge():
    specs = recovery._stage_specs()
    assert [row["name"] for row in specs] == [
        "adopt_checkpoint", "eval_seed0", "eval_seed1", "eval_seed2", "merge",
    ]
    assert specs[0]["depends_on"] == []
    assert all(row["depends_on"] == ["adopt_checkpoint"] for row in specs[1:4])
    assert specs[4]["depends_on"] == ["eval_seed0", "eval_seed1", "eval_seed2"]
    assert all("train" not in row["name"] for row in specs)
    assert all("consolidate" not in row["name"] for row in specs)


def test_v2_plan_is_fresh_eval_only_no_gate(
    monkeypatch, tmp_path, source_trigger,
):
    plan = _fast_plan(monkeypatch, tmp_path, source_trigger)
    assert plan["format_version"] == 2
    assert plan["kind"] == recovery.KIND
    assert plan["eligibility"] == recovery.EXPECTED_ELIGIBILITY
    assert plan["method"] == recovery.EXPECTED_METHOD
    assert plan["method"]["recovery_training_updates"] == 0
    assert plan["method"]["checkpoint_reconstructions"] == 0
    assert plan["method"]["scientific_gates"] == 0
    assert plan["schedule"] == {
        "training_jobs": 0, "consolidation_jobs": 0,
        "checkpoint_adoption_jobs": 1, "seed_eval_jobs": 3,
        "merge_jobs": 1, "decision_gate_jobs": [], "fixed_endpoint": 32_000,
    }
    assert plan["paths"]["checkpoint"] == str(recovery.SOURCE_CHECKPOINT)
    assert plan["evaluation"] == source_plan_from_trigger(plan)["evaluation"]
    assert plan["wandb"]["group"] == recovery.GROUP
    host = plan["recovery_host_runtime"]
    assert host["kind"] == "r0_e2e_operator_repair_recovery_host_runtime_v2"
    assert host["probe"]["prefix"] == str(recovery.RECOVERY_HOST_VENV)
    assert host["probe"]["packages"]["torch"]
    assert host["pip_freeze"]["packages"] == len(host["pip_freeze"]["lines"])
    recovery._assert_plan(plan)


def source_plan_from_trigger(_plan: dict) -> dict:
    # The helper keeps the assertion above explicit without exposing a mutable
    # source-plan object through a module global.
    return recovery._load_source_plan()


@pytest.mark.parametrize(
    ("control", "artifact", "message"),
    (
        ("root", "root/artifacts", "mutually disjoint"),
        ("root/control", "root", "mutually disjoint"),
        ("control,bad", "artifacts", "unsafe scheduler"),
        ("control", "artifacts\nbad", "unsafe scheduler"),
    ),
)
def test_output_roots_reject_nesting_and_slurm_delimiters(
    tmp_path, control, artifact, message,
):
    with pytest.raises(recovery.RecoveryError, match=message):
        recovery._require_isolated(
            (tmp_path / control).absolute(), (tmp_path / artifact).absolute(),
        )


def test_output_roots_reject_source_overlap(tmp_path):
    with pytest.raises(recovery.RecoveryError, match="overlaps immutable source"):
        recovery._require_isolated(
            recovery.SOURCE_CONTROL_DIR / "recovery", (tmp_path / "artifacts").resolve(),
        )


def test_compatibility_endpoint_uses_real_unpadded_names_and_new_plan_sha(
    monkeypatch, tmp_path, source_trigger,
):
    plan = _fast_plan(monkeypatch, tmp_path, source_trigger)
    path, digest = _materialize_plan(plan)
    asset_rehashes = []
    monkeypatch.setattr(
        operator, "_asset_receipt",
        lambda: asset_rehashes.append(True) or copy.deepcopy(plan["assets"]),
    )
    before = recovery.sha256_file(recovery.SOURCE_FIXED_ENDPOINT)
    endpoint, endpoint_sha = recovery._publish_compatibility_receipts(plan)
    # Replay validates the immutable receipt without repeating the full cache
    # and raw-data rehash performed before its initial publication.
    recovery._publish_compatibility_receipts(plan)
    assert endpoint["plan_sha256"] == digest
    assert set(endpoint["checkpoint_shards"]) == {
        f"ckpt_000032000_rank{rank}.pt" for rank in range(16)
    }
    assert endpoint["evaluation_recovery_adoption"]["source_bytes_rewritten"] is False
    assert endpoint["evaluation_recovery_adoption"][
        "checkpoint_bytes_rewritten"
    ] is False
    assert endpoint_sha == recovery.sha256_file(plan["paths"]["fixed_endpoint"])
    assert recovery.sha256_file(recovery.SOURCE_FIXED_ENDPOINT) == before
    assert Path(os.environ["OPERATOR_REPAIR_PLAN"]) == path
    assert asset_rehashes == [True]


def test_checkpoint_stage_is_verify_only_and_never_reconstructs_or_pins(
    monkeypatch, tmp_path, source_trigger,
):
    plan = _fast_plan(monkeypatch, tmp_path, source_trigger)
    _materialize_plan(plan)
    monkeypatch.setattr(
        operator, "_asset_receipt", lambda: copy.deepcopy(plan["assets"]),
    )
    endpoint, endpoint_sha = recovery._publish_compatibility_receipts(plan)
    calls: list[list[str]] = []
    fake_operator_receipt = {
        "format_version": 1,
        "kind": "r0_e2e_operator_repair_consolidated_checkpoint_receipt",
    }

    def fake_run(command, **_kwargs):
        calls.append(list(command))
        report = Path(command[command.index("--report") + 1])
        report.write_bytes(recovery.SOURCE_CHECKPOINT_REPORT.read_bytes())
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(
        recovery, "_assert_recovery_execution_closure",
        lambda *_a, **_k: {},
    )
    monkeypatch.setattr(
        recovery, "_publish_compatibility_receipts",
        lambda _plan: (endpoint, endpoint_sha),
    )
    monkeypatch.setattr(recovery.subprocess, "run", fake_run)
    monkeypatch.setattr(
        operator, "_checkpoint_receipt_payload",
        lambda *_args, **_kwargs: copy.deepcopy(fake_operator_receipt),
    )
    monkeypatch.setattr(
        operator, "_validate_checkpoint_receipt",
        lambda *_args, **_kwargs: copy.deepcopy(fake_operator_receipt),
    )
    monkeypatch.setattr(operator, "_wandb_publish", lambda *_args, **_kwargs: None)
    before = recovery._identity(recovery.SOURCE_CHECKPOINT)
    assert recovery._stage_adopt_checkpoint(plan) == 0
    after = recovery._identity(recovery.SOURCE_CHECKPOINT)
    assert before == after
    assert len(calls) == 1
    command = calls[0]
    assert "--verify_only" in command
    assert "--pin" not in command
    assert "--no_verify" not in command
    assert command[command.index("--out") + 1] == str(recovery.SOURCE_CHECKPOINT)
    adoption = json.loads(
        Path(plan["paths"]["checkpoint_adoption_receipt"]).read_text()
    )
    assert adoption["verify_only"] is True
    assert adoption["checkpoint_reconstructed"] is False
    assert adoption["training_updates"] == 0


def test_seed_stage_delegates_to_operator_eval_under_v2_plan(
    monkeypatch, tmp_path, source_trigger,
):
    plan = _fast_plan(monkeypatch, tmp_path, source_trigger)
    _materialize_plan(plan)
    calls: list[tuple[dict, str]] = []
    publications: list[dict] = []
    closure = {
        "recovery_source_closure": plan["recovery_source_closure"],
        "recovery_host_runtime": plan["recovery_host_runtime"],
        "historical_source_closure": plan["historical_source_closure"],
        "source_trigger_sha256": "1" * 64,
    }
    monkeypatch.setattr(
        recovery, "_assert_recovery_execution_closure",
        lambda *_a, **_k: copy.deepcopy(closure),
    )
    monkeypatch.setattr(
        recovery, "_publish_compatibility_receipts", lambda _plan: ({}, "f" * 64),
    )
    monkeypatch.setattr(recovery, "_validate_checkpoint_adoption", lambda _plan: {})
    monkeypatch.setattr(
        recovery, "_eval_source_pending_payload",
        lambda _plan, *, seed: {
            "format_version": 2, "kind": recovery.EVAL_SOURCE_PENDING_KIND,
            "seed": seed,
        },
    )

    def fake_complete(stage_plan, *, seed):
        paths = stage_plan["paths"]["eval"][str(seed)]
        return {
            "format_version": 2, "kind": recovery.EVAL_SOURCE_COMPLETE_KIND,
            "seed": seed,
            "result": recovery._identity(Path(paths["out_dir"]) / "results.json"),
            "operator_seed_receipt": recovery._identity(Path(paths["receipt"])),
        }

    monkeypatch.setattr(recovery, "_eval_source_complete_payload", fake_complete)
    monkeypatch.setattr(
        operator, "_validate_eval_receipt",
        lambda _plan, seed: {
            "kind": "r0_e2e_operator_repair_single_seed_eval_receipt",
            "seed": seed, "avg": 12.5,
        },
    )
    monkeypatch.setattr(
        operator, "_wandb_publish",
        lambda _plan, **kwargs: publications.append(dict(kwargs)),
    )

    def fake_eval(stage_plan, stage):
        calls.append((stage_plan, stage))
        seed = int(stage.removeprefix("eval_seed"))
        paths = stage_plan["paths"]["eval"][str(seed)]
        out = Path(paths["out_dir"])
        out.mkdir(parents=True)
        (out / "results.json").write_text("result\n")
        (out / "table.md").write_text("table\n")
        (out / "active_attempt.json").write_text("{}\n")
        (out / "completed_attempt.json").write_text("{}\n")
        Path(paths["receipt"]).write_text("{}\n")
        operator._wandb_publish(
            stage_plan, stage=f"eval-seed-{seed}",
            path=out / "results.json", artifact_type="test", summary={},
        )
        return 0

    monkeypatch.setattr(
        operator, "_stage_eval", fake_eval,
    )
    assert recovery._stage_eval(plan, "eval_seed2") == 0
    assert calls == [(plan, "eval_seed2")]
    assert Path(plan["paths"]["eval"]["2"]["source_pending"]).is_file()
    assert Path(plan["paths"]["eval"]["2"]["source_complete"]).is_file()
    assert [row["stage"] for row in publications] == ["eval-seed-2"]
    assert recovery._recovery_plan_sha() == recovery.sha256_file(
        os.environ["OPERATOR_RECOVERY_PLAN"]
    )


def test_seed_replay_quarantines_receipt_and_outputs_without_source_completion(
    monkeypatch, tmp_path, source_trigger,
):
    plan = _fast_plan(monkeypatch, tmp_path, source_trigger)
    _materialize_plan(plan)
    seed = 1
    paths = plan["paths"]["eval"][str(seed)]
    pending_payload = {
        "format_version": 2, "kind": recovery.EVAL_SOURCE_PENDING_KIND,
        "seed": seed,
    }
    monkeypatch.setattr(
        recovery, "_eval_source_pending_payload",
        lambda _plan, *, seed: {**pending_payload, "seed": seed},
    )
    pending = Path(paths["source_pending"])
    operator.common.exclusive_json_write(pending, pending_payload)
    out = Path(paths["out_dir"])
    out.mkdir(parents=True)
    for name in (
        "results.json", "table.md", "active_attempt.json",
        "completed_attempt.json",
    ):
        (out / name).write_text(f"uncertain {name}\n")
    Path(paths["receipt"]).write_text("uncertain receipt\n")

    assert recovery._prepare_eval_source_transaction(plan, seed=seed) == "PENDING"
    assert pending.is_file()
    assert json.loads(pending.read_text()) == pending_payload
    assert not Path(paths["source_complete"]).exists()
    assert all(
        not (out / name).exists()
        for name in (
            "results.json", "table.md", "active_attempt.json",
            "completed_attempt.json",
        )
    )
    assert not Path(paths["receipt"]).exists()
    assert len(list((out / "recovery").iterdir())) == 4
    control_recovery = Path(plan["lineage"]["control_dir"]) / "recovery"
    names = {path.name for path in control_recovery.iterdir()}
    assert any(name.startswith(f"eval_seed_{seed}_receipt.json.") for name in names)
    assert any(
        name.startswith(f"eval_seed_{seed}_source_pending_v2.json.")
        for name in names
    )


def test_seed_post_source_failure_quarantines_before_external_wandb(
    monkeypatch, tmp_path, source_trigger,
):
    plan = _fast_plan(monkeypatch, tmp_path, source_trigger)
    _materialize_plan(plan)
    seed = 0
    paths = plan["paths"]["eval"][str(seed)]
    monkeypatch.setattr(
        recovery, "_assert_recovery_execution_closure", lambda *_a, **_k: {},
    )
    monkeypatch.setattr(
        recovery, "_publish_compatibility_receipts", lambda _plan: ({}, "f" * 64),
    )
    monkeypatch.setattr(recovery, "_validate_checkpoint_adoption", lambda _plan: {})
    monkeypatch.setattr(
        recovery, "_eval_source_pending_payload",
        lambda _plan, *, seed: {
            "format_version": 2, "kind": recovery.EVAL_SOURCE_PENDING_KIND,
            "seed": seed,
        },
    )

    def fake_operator_eval(_plan, *, stage, seed):
        assert stage == "eval_seed0" and seed == 0
        out = Path(paths["out_dir"])
        out.mkdir(parents=True)
        for name in (
            "results.json", "table.md", "active_attempt.json",
            "completed_attempt.json",
        ):
            (out / name).write_text(f"uncertain {name}\n")
        Path(paths["receipt"]).write_text("uncertain receipt\n")
        return 0

    monkeypatch.setattr(
        recovery, "_capture_operator_eval_without_wandb", fake_operator_eval,
    )
    monkeypatch.setattr(
        recovery, "_eval_source_complete_payload",
        lambda *_a, **_k: (_ for _ in ()).throw(
            recovery.RecoveryError("runtime source changed after evaluation")
        ),
    )
    monkeypatch.setattr(
        operator, "_wandb_publish",
        lambda *_a, **_k: pytest.fail("uncertain result must not reach W&B"),
    )
    with pytest.raises(recovery.RecoveryError, match="runtime source changed"):
        recovery._stage_eval(plan, "eval_seed0")
    out = Path(paths["out_dir"])
    assert all(
        not (out / name).exists()
        for name in (
            "results.json", "table.md", "active_attempt.json",
            "completed_attempt.json",
        )
    )
    assert not Path(paths["receipt"]).exists()
    assert not Path(paths["source_pending"]).exists()
    assert not Path(paths["source_complete"]).exists()


def test_orphan_seed_source_completion_is_rejected(
    monkeypatch, tmp_path, source_trigger,
):
    plan = _fast_plan(monkeypatch, tmp_path, source_trigger)
    _materialize_plan(plan)
    complete = Path(plan["paths"]["eval"]["0"]["source_complete"])
    complete.write_text("{}\n")
    with pytest.raises(recovery.RecoveryError, match="completion is orphaned"):
        recovery._prepare_eval_source_transaction(plan, seed=0)


def test_checkpoint_adoption_replay_completes_missing_v2_receipt(
    monkeypatch, tmp_path, source_trigger,
):
    plan = _fast_plan(monkeypatch, tmp_path, source_trigger)
    _materialize_plan(plan)
    monkeypatch.setattr(
        operator, "_asset_receipt", lambda: copy.deepcopy(plan["assets"]),
    )
    recovery._publish_compatibility_receipts(plan)
    report = Path(plan["paths"]["checkpoint_report"])
    report.write_bytes(recovery.SOURCE_CHECKPOINT_REPORT.read_bytes())
    operator_receipt = {
        "format_version": 1,
        "kind": "r0_e2e_operator_repair_consolidated_checkpoint_receipt",
    }
    operator.common.exclusive_json_write(
        plan["paths"]["checkpoint_receipt"], operator_receipt,
    )
    monkeypatch.setattr(
        recovery, "_assert_recovery_execution_closure",
        lambda *_a, **_k: {},
    )
    monkeypatch.setattr(
        operator, "_validate_checkpoint_receipt",
        lambda *_args, **_kwargs: copy.deepcopy(operator_receipt),
    )
    monkeypatch.setattr(operator, "_wandb_publish", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        recovery.subprocess, "run",
        lambda *_args, **_kwargs: pytest.fail("replay must not reconsolidate"),
    )
    assert recovery._stage_adopt_checkpoint(plan) == 0
    adoption = Path(plan["paths"]["checkpoint_adoption_receipt"])
    assert adoption.is_file()
    assert json.loads(adoption.read_text())["verify_only"] is True


def test_completion_receipt_reports_raw_sr_without_threshold(tmp_path, monkeypatch):
    control, artifact = tmp_path / "control", tmp_path / "artifacts"
    control.mkdir()
    (artifact / "eval/merged").mkdir(parents=True)
    plan_file = control / "plan.json"
    plan_file.write_text("{}\n")
    monkeypatch.setenv("OPERATOR_RECOVERY_PLAN", str(plan_file))
    eval_paths = {}
    for seed in recovery.SEEDS:
        receipt = control / f"seed{seed}.json"
        receipt.write_text(f"{{\"seed\": {seed}}}\n")
        source_complete = control / f"seed{seed}_source_complete.json"
        source_complete.write_text(f"{{\"seed\": {seed}}}\n")
        eval_paths[str(seed)] = {
            "receipt": str(receipt), "source_complete": str(source_complete),
        }
    checkpoint_adoption = control / "checkpoint.json"
    checkpoint_adoption.write_text("{}\n")
    merged_receipt = control / "merged.json"
    merged_receipt.write_text(json.dumps({
        "episodes": 1_200, "errors": 0, "complete": True,
    }) + "\n")
    result = artifact / "eval/merged/results.json"
    result.write_text(json.dumps({
        "summary": {
            "n_episodes": 1_200, "n_errors": 0, "complete": True, "avg": 42.25,
        },
        "episodes": [{"success": index < 507} for index in range(1_200)],
    }) + "\n")
    table = artifact / "eval/merged/table.md"
    table.write_text("result\n")
    plan = {"paths": {
        "checkpoint_adoption_receipt": str(checkpoint_adoption),
        "eval": eval_paths, "merged_receipt": str(merged_receipt),
        "merged_results": str(result), "merged_table": str(table),
    }}
    host = {"kind": "test-host-runtime"}
    monkeypatch.setattr(
        recovery, "_assert_recovery_execution_closure",
        lambda *_a, **_k: {
            "recovery_source_closure": {"sha256": "a" * 64},
            "recovery_host_runtime": host,
        },
    )
    monkeypatch.setattr(
        recovery, "_validate_eval_source_completion",
        lambda _plan, *, seed: {"seed": seed},
    )
    payload = recovery._completion_payload(plan)
    assert payload["episodes"] == 1_200
    assert payload["successes"] == 507
    assert payload["end_to_end_success_rate_percent"] == 42.25
    assert payload["outcome_threshold_applied"] is False
    assert payload["scientific_gates"] == 0


def test_submission_holds_all_five_jobs_and_releases_once(
    monkeypatch, tmp_path, source_trigger,
):
    plan = _fast_plan(monkeypatch, tmp_path, source_trigger)
    calls: list[list[str]] = []
    next_id = iter(range(800001, 800006))

    def fake_run(command, **_kwargs):
        command = list(command)
        calls.append(command)
        if command[0] == "sbatch":
            return subprocess.CompletedProcess(command, 0, stdout=f"{next(next_id)}\n", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    receipt = recovery.submit_plan(plan, run=fake_run)
    assert receipt["released"] is True
    assert receipt["training_jobs"] == []
    assert receipt["consolidation_jobs"] == []
    assert receipt["decision_gate_jobs"] == []
    sbatches = [command for command in calls if command[0] == "sbatch"]
    assert len(sbatches) == 5
    assert all("--hold" in command for command in sbatches)
    release = [command for command in calls if command[:2] == ["scontrol", "release"]]
    assert release == [["scontrol", "release", "800001,800002,800003,800004,800005"]]
    assert "--dependency=afterok:800001" in sbatches[1]
    assert "--dependency=afterok:800002:800003:800004" in sbatches[4]


def test_submission_failure_cancels_every_job_already_held(
    monkeypatch, tmp_path, source_trigger,
):
    plan = _fast_plan(monkeypatch, tmp_path, source_trigger)
    calls: list[list[str]] = []
    counter = 0

    def fake_run(command, **_kwargs):
        nonlocal counter
        command = list(command)
        calls.append(command)
        if command[0] == "sbatch":
            counter += 1
            if counter == 3:
                raise subprocess.CalledProcessError(1, command)
            return subprocess.CompletedProcess(command, 0, stdout=f"{810000 + counter}\n", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    with pytest.raises(subprocess.CalledProcessError):
        recovery.submit_plan(plan, run=fake_run)
    assert [command for command in calls if command[0] == "scancel"] == [
        ["scancel", "810001", "810002"]
    ]
