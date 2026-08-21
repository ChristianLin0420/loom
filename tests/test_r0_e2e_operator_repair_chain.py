from __future__ import annotations

import copy
import inspect
import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import loom.train.loop as train_loop
from loom.train import wandb_util
from loom.train.loop import _reconcile_fresh_metrics, read_config
from scripts import r0_e2e_operator_repair_chain as chain
from scripts import r0_e2e_operator_repair_train_entry as entry


@pytest.fixture(scope="session")
def frozen_plan_inputs():
    # The cache receipt reads 76 GB by design. Exercise it separately once in
    # preflight; unit plans use the already independently pinned exact receipt.
    return {
        "assets": chain._expected_asset_receipt(),
        "baseline": chain._baseline_contract(),
        "eval_environment": chain._expected_eval_environment_receipt(),
    }


@pytest.fixture
def roots(monkeypatch, frozen_plan_inputs):
    monkeypatch.setattr(
        chain, "_asset_receipt",
        lambda: copy.deepcopy(frozen_plan_inputs["assets"]),
    )
    monkeypatch.setattr(
        chain, "_baseline_contract",
        lambda: copy.deepcopy(frozen_plan_inputs["baseline"]),
    )
    monkeypatch.setattr(
        chain, "_eval_environment_receipt",
        lambda: copy.deepcopy(frozen_plan_inputs["eval_environment"]),
    )
    base = chain.ROOT / "runs" / f".pytest_operator_repair_{uuid.uuid4().hex}"
    values = (base / "run", base / "control", base / "artifacts")
    yield values
    shutil.rmtree(base, ignore_errors=True)


def _plan(roots):
    run, control, artifacts = roots
    return chain.build_plan(
        run_dir=run, control_dir=control, artifact_root=artifacts,
        group="r0a-operator-repair-fixed32k-pytest",
    )


def test_canonical_config_is_exact_operator_repair_without_direct_gate(roots):
    cfg = read_config(chain.CANONICAL_CONFIG)
    plan = _plan(roots)
    assert chain.sha256_file(chain.CANONICAL_CONFIG) == chain.CANONICAL_CONFIG_SHA256
    assert plan["config"]["resolved_config_hash"] == "b47825f0cfba68dd"
    assert plan["config"]["overrides"] == []
    assert cfg["run"]["steps"] == 32_000
    assert cfg["run"]["boundary_policy"] == "fixed_max_updates"
    assert cfg["slurm"]["n_links"] == 6
    assert cfg["optim"]["spike_mult"] == 0
    assert cfg["run"]["log_every"] == 20
    assert cfg["run"]["reconcile_metrics_on_resume"] is True
    assert Path(cfg["data"]["data_root"]).resolve() == chain.DEFAULT_RAW_DATA_ROOT
    assert "schedule_horizon" not in cfg["run"]
    assert "max_updates" not in cfg["run"]
    assert "direct_formal" not in cfg
    assert cfg["method_receipt"] == {
        "kind": "loom_r0a_operator_repair_v1",
        "fixed_endpoint_update": 32_000,
        "evaluation_is_unconditional": True,
        "evaluation_episodes": 1_200,
        "evaluation_seeds": [0, 1, 2],
        "checkpoint_selection": "fixed_update_only",
        "health_thresholds_control_execution": False,
    }


def test_plan_freezes_fixed_endpoint_online_project_assets_and_descriptive_role(roots):
    plan = _plan(roots)
    assert plan["schedule"] == {
        "fixed_updates": 32_000,
        "optimizer_schedule_horizon": 32_000,
        "links": 6,
        "link_walltime": "04:00:00",
        "link_budget_seconds": 13_800,
        "world_size": 16,
        "checkpoint_every": 500,
        "selection_rule": "exact_predeclared_step_32000_only",
    }
    assert plan["wandb"]["project"] == "loom-r0-operator-repair"
    assert plan["wandb"]["tags"] == list(chain.EXPECTED_TRAIN_TAGS)
    assert plan["evaluation"]["runs_even_when_observational_health_is_poor"] is True
    assert plan["evaluation"]["attempt_identity"] == (
        "immutable_pre_environment_and_checkpoint_plus_post_reauthentication"
    )
    assert plan["evaluation"]["attempt_identity_mismatch_policy"] == (
        "content_address_quarantine_before_fail_no_episode_laundering"
    )
    assert plan["baseline_comparison"]["role"].startswith("descriptive_only")
    assert plan["baseline_comparison"]["pairing_snapshot"]["rows_sha256"] == (
        chain.BASELINE_PAIRING_ROWS_SHA256
    )
    assert len(plan["baseline_comparison"]["pairing_snapshot"]["rows"]) == 1_200
    assert plan["baseline_comparison"]["live_baseline_files_required_after_plan_creation"] is False
    assert plan["eligibility"]["formal_convergence_gate"] is False
    assert plan["eligibility"]["promotion_authority"] is False
    assert plan["method"]["metrics_ledger"] == {
        "format": "loom-fresh-metrics-rollback-v1",
        "reconcile_crash_tail_to_latest_checkpoint": True,
        "checkpoint_boundary_fsync": True,
        "direct_formal_decisions": False,
    }
    assert plan["wandb"]["training_log_failure_policy"] == {
        "kind": "consecutive_failures",
        "max_consecutive_failures": 5,
        "log_every_updates": 20,
        "failure_window_updates": 100,
        "success_resets_counter": True,
        "all_rank_outcome_broadcast": True,
    }
    assert plan["assets"]["cache_content"]["manifest"]["sha256"] == (
        "0ad6348be15d6baee4563f2b426d16b1b19fa87c74751b697ee8d7cd11144102"
    )
    assert plan["assets"]["frozen_tower"]["commit"] == chain.SIGLIP_COMMIT
    assert plan["failure_policy"]["post_training_asset_verification_failure"] == (
        "durable_terminal_marker_never_reauthorize"
    )
    assert plan["failure_policy"]["training_asset_post_transaction"] == (
        "pending_before_post_verification_exact_complete_required"
    )
    assert plan["paths"]["training_asset_failure"].endswith(
        "/TRAINING_ASSET_FAILURE.json"
    )


def test_source_closure_covers_new_chain_and_training_method_sources(roots):
    files = set(_plan(roots)["source_closure"]["files"])
    for required in (
        "scripts/r0_e2e_operator_repair_chain.py",
        "scripts/r0_e2e_operator_repair_train_entry.py",
        "scripts/r0_e2e_operator_repair_train.sbatch",
        "scripts/r0_e2e_operator_repair_consolidate.sbatch",
        "scripts/r0_e2e_operator_repair_eval_seed.sbatch",
        "scripts/r0_e2e_operator_repair_control.sbatch",
        "configs/r0a_operator_repair.yaml",
        "loom/data/loader.py", "loom/train/loop.py",
        "loom/train/schedule.py",
        "loom/losses/act.py", "loom/losses/dyn.py",
        "loom/train/consolidate.py", "loom/eval/runner.py", "loom/eval/table.py",
    ):
        assert required in files


def test_dag_has_six_links_no_gate_parallel_seeds_then_merge():
    specs = chain._stage_specs()
    names = [row["name"] for row in specs]
    assert names == [
        "train_01", "train_02", "train_03", "train_04", "train_05", "train_06", "consolidate",
        "eval_seed0", "eval_seed1", "eval_seed2", "merge",
    ]
    by_name = {row["name"]: row for row in specs}
    assert by_name["train_01"]["depends_on"] == []
    assert by_name["train_04"]["depends_on"] == ["train_03"]
    assert by_name["train_06"]["depends_on"] == ["train_05"]
    assert by_name["consolidate"]["depends_on"] == ["train_06"]
    for seed in chain.SEEDS:
        assert by_name[f"eval_seed{seed}"]["depends_on"] == ["consolidate"]
    assert by_name["merge"]["depends_on"] == ["eval_seed0", "eval_seed1", "eval_seed2"]
    assert not any("gate" in name or "convergence" in name for name in names)


def test_train_policy_is_fresh_only_first_attempt_and_sixth_requires_endpoint():
    assert chain._train_stage_policy("train_01", 0, has_latest=False) == {
        "index": 1, "fresh": True, "has_latest": False,
        "resume": "never", "require_endpoint": False,
    }
    assert chain._train_stage_policy("train_01", 1, has_latest=False) == {
        "index": 1, "fresh": False, "has_latest": False,
        "resume": "allow", "require_endpoint": False,
    }
    assert chain._train_stage_policy("train_06", 0, has_latest=True) == {
        "index": 6, "fresh": False, "has_latest": True,
        "resume": "must", "require_endpoint": True,
    }
    with pytest.raises(chain.OperatorRepairError):
        chain._train_stage_policy("train_07", 0, has_latest=False)


def test_train01_step0_requeue_materializes_absent_or_empty_run_plan(tmp_path):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text('{"exact":"plan"}\n')
    policy = chain._train_stage_policy("train_01", 1, has_latest=False)

    absent = tmp_path / "absent_run"
    copy_path = chain._materialize_run_plan_copy(
        run_dir=absent, plan_path=plan_path, policy=policy,
    )
    assert copy_path.read_bytes() == plan_path.read_bytes()

    empty = tmp_path / "empty_run"
    empty.mkdir()
    copy_path = chain._materialize_run_plan_copy(
        run_dir=empty, plan_path=plan_path, policy=policy,
    )
    assert copy_path.read_bytes() == plan_path.read_bytes()

    copy_path.write_text('{"different":true}\n')
    with pytest.raises(chain.OperatorRepairError, match="differs from submission"):
        chain._materialize_run_plan_copy(
            run_dir=empty, plan_path=plan_path, policy=policy,
        )

    later = chain._train_stage_policy("train_02", 0, has_latest=False)
    with pytest.raises(chain.OperatorRepairError, match="lacks its run directory"):
        chain._materialize_run_plan_copy(
            run_dir=tmp_path / "missing_later", plan_path=plan_path, policy=later,
        )


def test_training_argv_uses_only_canonical_recipe_plus_link_local_controls(roots):
    argv = chain._training_argv(roots[0], include_link=True)
    assert argv[:2] == ["--config", str(chain.CANONICAL_CONFIG)]
    assert "--set" not in argv
    assert argv[argv.index("--stop_at") + 1] == "32000"
    assert argv[argv.index("--budget_s") + 1] == "13800"
    assert argv[argv.index("--safety_s") + 1] == "420"
    assert "--schedule_horizon" not in argv
    assert "--max_updates" not in argv


def test_eval_command_explicitly_materializes_embodiment_and_exact_protocol(roots):
    plan = _plan(roots)
    command, env = chain._eval_command(
        plan, seed=2, checkpoint=roots[2] / "checkpoint/ckpt.pt",
        out_dir=roots[2] / "eval/seed_2",
    )
    assert command[command.index("--embodiment") + 1] == "libero_franka"
    assert command[command.index("--suites") + 1] == (
        "libero_spatial,libero_object,libero_goal,libero_long"
    )
    assert command[command.index("--seeds") + 1] == "2"
    assert command[command.index("--max-steps") + 1] == "512"
    assert command[command.index("--row-label") + 1] == "**LOOM · R0 operator repair**"
    assert "--no-resume" not in command
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["LOOM_DATA_ROOT"] == chain.DEFAULT_RAW_DATA_ROOT_TEXT
    assert env["LOOM_LIBERO_BDDL_DIR"].endswith("/libero/libero/bddl_files")
    assert env["LOOM_LIBERO_INIT_STATES_DIR"].endswith("/libero/libero/init_files")
    assert env["LOOM_LIBERO_IMAGE_SIZE"] == "256"
    assert env["LOOM_LIBERO_IMAGE_CONVENTION"] == "opengl"
    assert env["LOOM_LIBERO_PYTHON"] == str(chain.LIBERO_EVAL_PYTHON)
    assert env["TRITON_CACHE_DIR"] == str(
        roots[2] / "eval/seed_2/runtime/triton_cache"
    )
    other = chain._eval_command(
        plan, seed=1, checkpoint=roots[2] / "checkpoint/ckpt.pt",
        out_dir=roots[2] / "eval/seed_1",
    )[1]
    assert other["TRITON_CACHE_DIR"] != env["TRITON_CACHE_DIR"]
    assert env["TRITON_CACHE_DIR"] != str(chain.ROOT / ".triton_cache")


def test_eval_environment_receipt_is_exact_and_parent_overrides_do_not_escape(
    roots, monkeypatch,
):
    plan = _plan(roots)
    expected = chain._expected_eval_environment_receipt()
    assert plan["evaluation"]["environment"] == expected
    monkeypatch.setenv("LOOM_LIBERO_IMAGE_SIZE", "999")
    monkeypatch.setenv("LOOM_DATA_ROOT", "/tmp/not-canonical")
    monkeypatch.setenv("LOOM_TOWER_MODEL", "wrong/model")
    monkeypatch.setenv("LOOM_TOWER_IMAGE_SIZE", "999")
    monkeypatch.setenv("HF_HUB_CACHE", "/tmp/wrong-cache")
    _, env = chain._eval_command(
        plan, seed=0, checkpoint=roots[2] / "checkpoint/ckpt.pt",
        out_dir=roots[2] / "eval/seed_0",
    )
    assert env["LOOM_LIBERO_IMAGE_SIZE"] == "256"
    assert env["LOOM_DATA_ROOT"] == chain.DEFAULT_RAW_DATA_ROOT_TEXT
    assert env["LOOM_TOWER_MODEL"] == chain.SIGLIP_MODEL
    assert env["LOOM_TOWER_IMAGE_SIZE"] == "224"
    assert env["HF_HUB_CACHE"] == str(chain.DEFAULT_HF_HOME / "hub")
    assert expected["libero_repository"]["head"] == chain.LIBERO_REPOSITORY_HEAD
    assert expected["libero_repository"]["clean_status_bytes"] == 0
    assert expected["python"]["pip_freeze"]["sha256"] == (
        chain.LIBERO_EVAL_PIP_FREEZE_SHA256
    )
    assert expected["siglip_snapshot"]["sha256"] == (
        chain.SIGLIP_EVAL_SNAPSHOT_SHA256
    )
    assert expected["siglip_snapshot"]["files"] == 7


def test_siglip_eval_snapshot_rejects_blob_digest_mutation(monkeypatch):
    original = chain.sha256_file

    def changed(path):
        if Path(path).stat().st_size == 3_511_950_624:
            return "0" * 64
        return original(path)

    monkeypatch.setattr(chain, "sha256_file", changed)
    with pytest.raises(chain.OperatorRepairError, match="snapshot content receipt changed"):
        chain._siglip_eval_snapshot_receipt()


def test_submit_holds_eleven_afterok_jobs_then_single_comma_release(roots):
    plan = _plan(roots)
    calls = []
    counter = 100

    def fake_run(command, **kwargs):
        nonlocal counter
        calls.append(list(command))
        if command[0] == "sbatch":
            counter += 1
            return subprocess.CompletedProcess(command, 0, stdout=f"{counter}\n", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    result = chain.submit_plan(plan, run=fake_run)
    sbatch = [row for row in calls if row[0] == "sbatch"]
    assert len(sbatch) == 11
    assert all("--hold" in row and "--kill-on-invalid-dep=yes" in row for row in sbatch)
    assert "--dependency=afterok:106" in result["commands"]["consolidate"]
    assert all(
        "--dependency=afterok:107" in result["commands"][f"eval_seed{seed}"]
        for seed in chain.SEEDS
    )
    assert "--dependency=afterok:108:109:110" in result["commands"]["merge"]
    assert calls[-1] == ["scontrol", "release", ",".join(result["jobs"].values())]
    assert result["decision_gate_jobs"] == []


def test_partial_submission_cancels_only_new_held_jobs(roots):
    plan = _plan(roots)
    calls = []
    count = 0

    def fake_run(command, **kwargs):
        nonlocal count
        calls.append(list(command))
        if command[0] == "sbatch":
            count += 1
            if count == 3:
                raise subprocess.CalledProcessError(1, command)
            return subprocess.CompletedProcess(command, 0, stdout=f"20{count}\n", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    with pytest.raises(subprocess.CalledProcessError):
        chain.submit_plan(plan, run=fake_run)
    assert calls[-1] == ["scancel", "201", "202"]


def test_plan_rejects_nested_outputs_and_nonoperator_group(roots):
    run, control, artifacts = roots
    with pytest.raises(chain.OperatorRepairError, match="pairwise non-nested"):
        chain.build_plan(
            run_dir=run, control_dir=run / "control", artifact_root=artifacts,
            group="r0a-operator-repair-fixed32k-test",
        )
    with pytest.raises(chain.OperatorRepairError, match="must explicitly say"):
        chain.build_plan(
            run_dir=run, control_dir=control, artifact_root=artifacts,
            group="generic-group",
        )


def test_plan_mutations_fail_closed(roots):
    plan = _plan(roots)
    bad = copy.deepcopy(plan)
    bad["schedule"]["fixed_updates"] = 31_999
    with pytest.raises(chain.OperatorRepairError, match="schedule changed"):
        chain._assert_plan(bad)
    bad = copy.deepcopy(plan)
    bad["evaluation"]["seeds"] = [0]
    with pytest.raises(chain.OperatorRepairError, match="protocol changed"):
        chain._assert_plan(bad)
    bad = copy.deepcopy(plan)
    bad["wandb"]["project"] = "loom"
    with pytest.raises(chain.OperatorRepairError, match="W&B contract"):
        chain._assert_plan(bad)


def test_metrics_receipt_is_exact_contiguous_and_observational(tmp_path):
    path = tmp_path / "metrics.jsonl"
    required = {
        "loss/dyn": 1.0, "act/decode_teacher": 1.0,
        "act/decode_deploy": 1.0, "act/align": 0.1,
        "loss/proposal": 2.0, "delta_op": 0.2,
        "delta_sel/h1": 0.1, "delta_sel/h2": 0.1,
        "delta_sel/h3": 0.1, "delta_sel/h4": 0.1,
        "bank/live_ops_q_delta": 20, "bank/live_ops_q_a": 20,
        "grad_norm": 1.0, "grad_skipped": 0,
    }
    # Keep this unit test bounded while exercising the exact endpoint predicate.
    original = chain.FIXED_STEP
    chain.FIXED_STEP = 3
    try:
        path.write_text("".join(
            json.dumps({"global_step": step, **required}) + "\n"
            for step in range(1, 4)
        ))
        receipt = chain._metrics_receipt(tmp_path)
        assert receipt["rows"] == 3
        assert receipt["role"] == "observational_only_never_a_dependency_decision"
        assert receipt["missing_observational_keys"] == []
        path.write_text("".join(
            json.dumps({"global_step": step}) + "\n" for step in range(1, 4)
        ))
        missing = chain._metrics_receipt(tmp_path)
        assert missing["final_snapshot"] == {}
        assert missing["missing_observational_keys"]
        path.write_text(
            json.dumps({"global_step": 1, **required}) + "\n"
            + json.dumps({"global_step": 3, **required}) + "\n"
        )
        with pytest.raises(chain.OperatorRepairError, match="contiguous"):
            chain._metrics_receipt(tmp_path)
    finally:
        chain.FIXED_STEP = original


def test_fixed_endpoint_rejects_any_direct_formal_receipt(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "LATEST").write_text("32000\n")
    (run_dir / "wandb_id").write_text("a" * 16)
    (run_dir / "direct_formal_000032000.json").write_text("{}\n")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}\n")
    monkeypatch.setenv("OPERATOR_REPAIR_PLAN", str(plan_path))
    monkeypatch.setattr(chain, "_reject_training_asset_failure", lambda plan: None)
    monkeypatch.setattr(chain, "_reject_durable_execution_failure", lambda plan: None)
    monkeypatch.setattr(chain, "_run_config_identity", lambda plan: {"sha256": "b" * 64})
    monkeypatch.setattr(chain, "_metrics_receipt", lambda path: {"sha256": "c" * 64})
    monkeypatch.setattr(
        chain.common, "_checkpoint_shards",
        lambda *args, **kwargs: [tmp_path / f"rank{rank}" for rank in range(16)],
    )
    monkeypatch.setattr(chain.common, "_checkpoint_shard_receipt", lambda paths: {})
    plan = {
        "lineage": {"run_dir": str(run_dir)},
        "wandb": {"training_run_id": "a" * 16},
    }
    with pytest.raises(chain.OperatorRepairError, match="direct-formal receipts"):
        chain._fixed_endpoint_payload(plan)


def test_fixed_endpoint_reader_accepts_real_unpadded_rank_shard_names(
    tmp_path, monkeypatch,
):
    run_dir = tmp_path / "run"
    control_dir = tmp_path / "control"
    run_dir.mkdir()
    control_dir.mkdir()
    endpoint = control_dir / "fixed_endpoint_32000.json"
    asset_dir = control_dir / "training_asset_verification"
    asset_dir.mkdir()
    plan_sha = "a" * 64
    wandb_id = "b" * 16
    plan = {
        "lineage": {"run_dir": str(run_dir)},
        "paths": {
            "fixed_endpoint": str(endpoint),
            "training_asset_verification_dir": str(asset_dir),
        },
        "wandb": {"training_run_id": wandb_id},
    }
    receipt = {
        "format_version": chain.FORMAT_VERSION,
        "kind": "r0_e2e_operator_repair_fixed_endpoint",
        "plan_sha256": plan_sha,
        "step": chain.FIXED_STEP,
        "selection": "predeclared_fixed_step_no_metric_or_eval_selection",
        "optimizer_updates": chain.FIXED_STEP,
        "run_config": {
            "path": str((run_dir / "config.json").resolve()),
            "bytes": 1,
            "sha256": "c" * 64,
        },
        "metrics": {
            "path": str((run_dir / "metrics.jsonl").resolve()),
            "sha256": "d" * 64,
            "rows": chain.FIXED_STEP,
            "role": "observational_only_never_a_dependency_decision",
        },
        "checkpoint_shards": {
            f"ckpt_{chain.FIXED_STEP:09d}_rank{rank}.pt": {
                "bytes": rank + 1,
                "sha256": f"{rank + 1:064x}",
            }
            for rank in range(chain.WORLD_SIZE)
        },
        "training_asset_verification": {
            "stage": f"train_{chain.TRAIN_LINKS:02d}",
            "phase": "post",
            "path": str(asset_dir / f"train_{chain.TRAIN_LINKS:02d}_post.json"),
            "sha256": "e" * 64,
        },
        "training_wandb_run_id": wandb_id,
        "direct_formal_receipts": [],
        "health_metrics_used_as_gate": False,
        "evaluation_required_after_integrity": True,
    }
    endpoint.write_text(json.dumps(receipt) + "\n")
    monkeypatch.setattr(chain, "_plan_sha", lambda: plan_sha)

    parsed, endpoint_sha = chain._read_fixed_endpoint(plan)
    assert parsed == receipt
    assert endpoint_sha == chain.sha256_file(endpoint)

    padded = copy.deepcopy(receipt)
    padded["checkpoint_shards"] = {
        f"ckpt_{chain.FIXED_STEP:09d}_rank{rank:05d}.pt": row
        for rank, row in enumerate(receipt["checkpoint_shards"].values())
    }
    endpoint.write_text(json.dumps(padded) + "\n")
    with pytest.raises(chain.OperatorRepairError, match="immutable shape"):
        chain._read_fixed_endpoint(plan)


def test_complete_seed_result_recovers_missing_or_corrupt_table_idempotently(
    roots, monkeypatch,
):
    table = roots[2] / "eval/seed_0/table.md"
    table.parent.mkdir(parents=True)
    monkeypatch.setattr(chain, "_seed_markdown", lambda blob: "exact table\n")
    chain._ensure_seed_table(table, {}, allow_create=True)
    assert table.read_text() == "exact table\n"
    table.write_text("corrupt\n")
    recovered = chain._ensure_seed_table(
        table, {}, allow_create=True, recover_corrupt=True,
    )
    assert recovered["action"] == "RECOVERED_CORRUPT"
    assert table.read_text() == "exact table\n"
    quarantine = Path(recovered["quarantine"])
    assert quarantine.read_text() == "corrupt\n"
    assert chain._ensure_seed_table(
        table, {}, allow_create=True, recover_corrupt=True,
    )["action"] == "NONE"


def test_frozen_baseline_snapshot_never_reopens_live_baseline(
    roots, monkeypatch,
):
    plan = _plan(roots)

    def forbidden(*args, **kwargs):
        raise AssertionError("live historical baseline must not be read after plan creation")

    monkeypatch.setattr(chain.common, "_authenticate_baseline", forbidden)
    monkeypatch.setattr(chain.common, "_baseline_rows", forbidden)
    chain._assert_plan(plan)
    rows = chain._validate_frozen_baseline_contract(plan["baseline_comparison"])
    comparison = chain._descriptive_comparison(plan, rows)
    assert comparison["overall"]["candidate_successes"] == 447
    assert comparison["overall"]["delta_percentage_points"] == 0.0


def test_final_stat_sweep_rejects_late_mutation(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    captured = {first: (1, 2, 3, 4, 5), second: (6, 7, 8, 9, 10)}
    actual = dict(captured)
    actual[first] = (1, 2, 3, 4, 99)
    with pytest.raises(chain.OperatorRepairError, match="final verification sweep"):
        chain._require_unchanged_final_stats(
            captured, label="fixture", stat_fn=lambda path: actual[path],
        )


def test_storage_receipts_use_libero_10_while_eval_uses_libero_long(roots):
    plan = _plan(roots)
    assert set(plan["assets"]["raw_training_input"]["per_suite"]) == {
        "libero_spatial", "libero_object", "libero_goal", "libero_10",
    }
    assert set(plan["assets"]["cache_content"]["per_suite_entries"]) == {
        "libero_spatial", "libero_object", "libero_goal", "libero_10",
    }
    assert set(plan["evaluation"]["suites"]) == {
        "libero_spatial", "libero_object", "libero_goal", "libero_long",
    }


def _candidate_blob_from_baseline(seed: int, checkpoint: Path) -> dict:
    source = chain.common.CANONICAL_BASELINE_ROOT / f"seed{seed}/results.json"
    blob = json.loads(source.read_text())
    path = str(checkpoint.resolve())
    blob["meta"]["ckpt"] = path
    blob["meta"]["eval_identity"]["checkpoint"] = path
    blob["meta"]["eval_identity"]["policy_kw"] = {
        "allow_stub": False, "op_stats": True, "embodiment": "libero_franka",
    }
    blob["meta"]["policy"]["ckpt"] = path
    blob["meta"]["policy"]["ckpt_global_step"] = chain.FIXED_STEP
    blob["meta"]["policy"]["embodiment"] = "libero_franka"
    return blob


def _install_checkpoint_fixture(monkeypatch, checkpoint: Path):
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"checkpoint")
    receipt = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": chain.sha256_file(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
    }
    monkeypatch.setattr(chain, "_validate_checkpoint_receipt", lambda plan: receipt)
    return receipt


def test_seed_summary_is_recomputed_and_tampering_rejected(roots, monkeypatch):
    plan = _plan(roots)
    checkpoint = Path(plan["paths"]["checkpoint"])
    _install_checkpoint_fixture(monkeypatch, checkpoint)
    result = roots[2] / "eval/seed_0/results.json"
    result.parent.mkdir(parents=True)
    blob = _candidate_blob_from_baseline(0, checkpoint)
    result.write_text(json.dumps(blob))
    validated, _ = chain._validate_seed_result(plan, seed=0, result_path=result)
    assert validated["summary"]["n_episodes"] == 400
    blob["summary"]["avg"] += 1.0
    result.write_text(json.dumps(blob))
    with pytest.raises(chain.OperatorRepairError, match="summary differs"):
        chain._validate_seed_result(plan, seed=0, result_path=result)


def test_transient_error_rows_are_quarantined_and_retried_without_threshold(
    roots, monkeypatch,
):
    plan = _plan(roots)
    checkpoint = Path(plan["paths"]["checkpoint"])
    _install_checkpoint_fixture(monkeypatch, checkpoint)
    control = roots[1]
    control.mkdir(parents=True)
    Path(plan["paths"]["checkpoint_receipt"]).write_text("{}\n")
    plan_path = control / "plan.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True))
    monkeypatch.setenv("OPERATOR_REPAIR_PLAN", str(plan_path))
    monkeypatch.setattr(chain, "_wandb_publish", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        chain, "_eval_command",
        lambda *args, **kwargs: (["fake-eval"], dict()),
    )
    clean = _candidate_blob_from_baseline(0, checkpoint)
    failed = copy.deepcopy(clean)
    failed["episodes"][0]["error"] = "transient simulator failure"
    out_dir = Path(plan["paths"]["eval"]["0"]["out_dir"])
    result = out_dir / "results.json"
    calls = []

    def fake_run(command, **kwargs):
        calls.append((list(command), Path(kwargs["cwd"])))
        blob = failed if len(calls) == 1 else clean
        result.write_text(json.dumps(blob))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(chain.subprocess, "run", fake_run)
    assert chain._stage_eval(plan, "eval_seed0") == 0
    assert len(calls) == 2
    assert calls[0][1] == out_dir / "runtime"
    recovery = chain._eval_recovery_receipt(out_dir)
    assert any("results.json.error_rows" in row["name"] for row in recovery["files"])
    receipt = json.loads(Path(plan["paths"]["eval"]["0"]["receipt"]).read_text())
    assert receipt["recovery"] == recovery
    assert "threshold" not in json.dumps(receipt).lower()


@pytest.mark.parametrize("mutation_target", ["environment", "checkpoint"])
def test_eval_post_authentication_failure_quarantines_before_clean_requeue(
    roots, monkeypatch, mutation_target,
):
    plan = _plan(roots)
    checkpoint = Path(plan["paths"]["checkpoint"])
    checkpoint_receipt = _install_checkpoint_fixture(monkeypatch, checkpoint)
    control = roots[1]
    control.mkdir(parents=True)
    Path(plan["paths"]["checkpoint_receipt"]).write_text("{}\n")
    plan_path = control / "plan.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True))
    monkeypatch.setenv("OPERATOR_REPAIR_PLAN", str(plan_path))
    monkeypatch.setattr(chain, "_wandb_publish", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        chain, "_eval_command",
        lambda *args, **kwargs: (["fake-eval"], dict()),
    )
    expected_environment = copy.deepcopy(plan["evaluation"]["environment"])
    environment_calls = 0
    checkpoint_calls = 0

    def environment_receipt():
        nonlocal environment_calls
        environment_calls += 1
        if mutation_target == "environment" and environment_calls == 3:
            raise chain.OperatorRepairError("simulated post environment mutation")
        return copy.deepcopy(expected_environment)

    def checkpoint_validation(plan_arg):
        nonlocal checkpoint_calls
        checkpoint_calls += 1
        if mutation_target == "checkpoint" and checkpoint_calls == 3:
            raise chain.OperatorRepairError("simulated post checkpoint mutation")
        return copy.deepcopy(checkpoint_receipt)

    monkeypatch.setattr(chain, "_eval_environment_receipt", environment_receipt)
    monkeypatch.setattr(chain, "_validate_checkpoint_receipt", checkpoint_validation)
    out_dir = Path(plan["paths"]["eval"]["0"]["out_dir"])
    result = out_dir / "results.json"
    clean = _candidate_blob_from_baseline(0, checkpoint)
    process_calls = []

    def fake_run(command, **kwargs):
        process_calls.append(list(command))
        result.write_text(json.dumps(clean))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(chain.subprocess, "run", fake_run)
    with pytest.raises(chain.OperatorRepairError, match="simulated post"):
        chain._stage_eval(plan, "eval_seed0")
    active, completed = chain._eval_attempt_paths(out_dir)
    assert not result.exists() and not active.exists() and not completed.exists()
    quarantined = chain._eval_recovery_receipt(out_dir)["files"]
    assert any("results.json.post_attempt_authentication_failure" in row["name"] for row in quarantined)
    assert any("active_attempt.json.post_attempt_authentication_failure" in row["name"] for row in quarantined)
    assert not Path(plan["paths"]["eval"]["0"]["receipt"]).exists()

    # Restoring the environment/checkpoint cannot adopt the quarantined rows;
    # the next job performs a fresh physical eval and closes a new sidecar pair.
    monkeypatch.setattr(
        chain, "_eval_environment_receipt",
        lambda: copy.deepcopy(expected_environment),
    )
    monkeypatch.setattr(
        chain, "_validate_checkpoint_receipt",
        lambda plan_arg: copy.deepcopy(checkpoint_receipt),
    )
    assert chain._stage_eval(plan, "eval_seed0") == 0
    assert len(process_calls) == 2
    active_blob = json.loads(active.read_text())
    completed_blob = json.loads(completed.read_text())
    assert active_blob["ordinal"] == 2
    assert active_blob["environment"] == expected_environment
    assert active_blob["checkpoint_sha256"] == checkpoint_receipt["checkpoint_sha256"]
    assert active_blob["result"] == str(result)
    assert active_blob["table"] == str(out_dir / "table.md")
    assert completed_blob["post_environment_and_checkpoint_reauthenticated"] is True


def test_current_chain_contains_no_threshold_or_pass_fail_bundle(roots):
    plan = _plan(roots)
    assert "thresholds" not in plan["baseline_comparison"]
    assert "pass_fail_classification" not in plan["baseline_comparison"]
    source = (chain.ROOT / "scripts/r0_e2e_operator_repair_chain.py").read_text()
    assert "threshold_bundle_status" not in source
    assert "threshold_bundle_passed" not in source
    assert "failed_scientific_checks" not in source
    assert "scientific_gate_status" not in source
    assert "SUITE_FLOORS_PERCENT" not in source


def test_descriptive_self_comparison_is_zero_without_classification(roots):
    plan = _plan(roots)
    rows = chain._validate_frozen_baseline_contract(plan["baseline_comparison"])
    comparison = chain._descriptive_comparison(plan, rows)
    assert comparison["overall"]["delta_percentage_points"] == 0.0
    assert comparison["paired_task_bootstrap"]["ci_low_percentage_points"] == 0.0
    assert comparison["paired_task_bootstrap"]["ci_high_percentage_points"] == 0.0
    assert comparison["paired_task_bootstrap"]["resample_matrix"]["sha256"] == (
        chain.common.BOOTSTRAP_MATRIX_SHA256
    )
    assert not any(
        key in comparison
        for key in ("status", "passed", "checks", "failed_checks", "thresholds")
    )


def test_source_has_no_gate_stage_or_formal_train_entry_dependency():
    source = (chain.ROOT / "scripts/r0_e2e_operator_repair_chain.py").read_text()
    assert "def _stage_gate" not in source
    assert "DirectFormalGate" not in source
    assert "scripts/direct_formal_convergence.py" not in chain.SOURCE_FILES
    assert "r0_e2e_formal_train_entry.py" not in source
    assert "evaluation_unconditional_after_integrity" in source
    assert "--embodiment" in source


def test_train_entry_freezes_online_project_tags_and_no_gate_label():
    source = (chain.ROOT / "scripts/r0_e2e_operator_repair_train_entry.py").read_text()
    assert 'project != "loom-r0-operator-repair"' in source
    assert '"no-gate"' in source
    assert '"fixed_endpoint": 32_000' in source
    assert "offline fallback is forbidden" in source
    assert "run.log(payload, step=int(global_step))" in source
    assert "MAX_CONSECUTIVE_LOG_FAILURES = 5" in source
    assert "LOG_FAILURE_WINDOW_UPDATES" in source
    assert "broadcast_object_list" in source


def _operator_wandb_environment(monkeypatch, tmp_path, *, committed_step=0):
    lineage_sha = "a" * 64
    health_path = tmp_path / "operator_repair_wandb_health.json"
    if not health_path.exists():
        health_path.write_text(json.dumps(
            entry.initial_wandb_health_state(lineage_sha), sort_keys=True,
        ) + "\n")
    values = {
        "LOOM_WANDB_PROJECT": "loom-r0-operator-repair",
        "LOOM_WANDB_GROUP": "r0a-operator-repair-fixed32k-pytest",
        "LOOM_WANDB_JOB_TYPE": "operator-repair-train",
        "LOOM_WANDB_TAGS": ",".join(chain.EXPECTED_TRAIN_TAGS),
        "LOOM_WANDB_RESUME": "must",
        "LOOM_WANDB_REQUIRE_ONLINE": "1",
        "WANDB_MODE": "online",
        "RANK": "0",
        "WORLD_SIZE": "1",
        "LOOM_RESTART_COUNT": "3",
        "LOOM_WANDB_LINEAGE_SHA256": lineage_sha,
        "LOOM_WANDB_HEALTH_STATE": str(health_path),
        "LOOM_WANDB_COMMITTED_STEP": str(committed_step),
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)


class _LoggingRun:
    offline = False

    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.calls = []

    def log(self, payload, *, step):
        self.calls.append((dict(payload), step))
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome


def _install_entry_for_log_test(monkeypatch, tmp_path, *, committed_step=0):
    _operator_wandb_environment(
        monkeypatch, tmp_path, committed_step=committed_step,
    )
    fake = SimpleNamespace(
        init=lambda **kwargs: None,
        Settings=lambda **kwargs: ("settings", kwargs),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake)
    original_init = wandb_util.init
    original_log = wandb_util.log
    entry.install_operator_repair_wandb_contract()
    return original_init, original_log


def test_entry_tolerates_four_consecutive_log_failures_then_fails_all_ranks(
    tmp_path, monkeypatch,
):
    original_init, original_log = _install_entry_for_log_test(monkeypatch, tmp_path)
    run = _LoggingRun([ConnectionError(f"down-{index}") for index in range(5)])
    try:
        for call in range(1, 5):
            wandb_util.log(run, {"loss": float(call)}, call * 20)
        with pytest.raises(entry.OperatorRepairWandbError, match="5 consecutive"):
            wandb_util.log(run, {"loss": 5.0}, 100)
    finally:
        wandb_util.init = original_init
        wandb_util.log = original_log
    assert [step for _, step in run.calls] == [20, 40, 60, 80, 100]


def test_entry_success_resets_log_failure_counter_and_logs_direct_payload(
    tmp_path, monkeypatch,
):
    original_init, original_log = _install_entry_for_log_test(monkeypatch, tmp_path)
    outcomes = [ConnectionError("a")] * 4 + [None] + [ConnectionError("b")] * 4
    run = _LoggingRun(outcomes)
    try:
        for call in range(1, 10):
            wandb_util.log(run, {"loss": float(call)}, call * 20)
    finally:
        wandb_util.init = original_init
        wandb_util.log = original_log
    payload, step = run.calls[4]
    assert step == 100
    assert payload["global_step"] == 100
    assert payload["restart_count"] == 3


def test_entry_broadcasts_each_rank0_log_outcome(tmp_path, monkeypatch):
    original_init, original_log = _install_entry_for_log_test(monkeypatch, tmp_path)
    monkeypatch.setenv("WORLD_SIZE", "2")
    broadcasts = []
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def broadcast(values, *, src):
        broadcasts.append((copy.deepcopy(values), src))

    monkeypatch.setattr(torch.distributed, "broadcast_object_list", broadcast)
    run = _LoggingRun([None])
    try:
        wandb_util.log(run, {"loss": 1.0}, 20)
    finally:
        wandb_util.init = original_init
        wandb_util.log = original_log
    assert broadcasts == [([{
        "ok": True,
        "fatal": False,
        "consecutive_failures": 0,
        "global_step": 20,
    }], 0)]


def test_entry_failure_counter_survives_restart_and_fifth_failure_is_fatal(
    tmp_path, monkeypatch,
):
    original_init, original_log = _install_entry_for_log_test(monkeypatch, tmp_path)
    first = _LoggingRun([ConnectionError("down")] * 4)
    try:
        for step in (20, 40, 60, 80):
            wandb_util.log(first, {"loss": 1.0}, step)
    finally:
        wandb_util.init = original_init
        wandb_util.log = original_log

    original_init, original_log = _install_entry_for_log_test(
        monkeypatch, tmp_path, committed_step=80,
    )
    second = _LoggingRun([ConnectionError("still down")])
    try:
        with pytest.raises(entry.OperatorRepairWandbError, match="5 consecutive"):
            wandb_util.log(second, {"loss": 1.0}, 100)
    finally:
        wandb_util.init = original_init
        wandb_util.log = original_log


def test_entry_success_reset_survives_restart(tmp_path, monkeypatch):
    original_init, original_log = _install_entry_for_log_test(monkeypatch, tmp_path)
    first = _LoggingRun([ConnectionError("down")] * 4 + [None])
    try:
        for step in (20, 40, 60, 80, 100):
            wandb_util.log(first, {"loss": 1.0}, step)
    finally:
        wandb_util.init = original_init
        wandb_util.log = original_log

    original_init, original_log = _install_entry_for_log_test(
        monkeypatch, tmp_path, committed_step=100,
    )
    second = _LoggingRun([ConnectionError("down again")] * 4)
    try:
        for step in (120, 140, 160, 180):
            wandb_util.log(second, {"loss": 1.0}, step)
    finally:
        wandb_util.init = original_init
        wandb_util.log = original_log


def test_wandb_health_history_requires_exact_twenty_step_cadence():
    lineage = "d" * 64
    state = entry.initial_wandb_health_state(lineage)
    state["events"] = [
        {"global_step": 20, "ok": False},
        {"global_step": 60, "ok": False},
    ]
    with pytest.raises(entry.OperatorRepairWandbError, match="history is malformed"):
        entry._validate_wandb_health_state(state, lineage_sha256=lineage)
    state["events"] = [{"global_step": 40, "ok": False}]
    with pytest.raises(entry.OperatorRepairWandbError, match="history is malformed"):
        entry._validate_wandb_health_state(state, lineage_sha256=lineage)


def test_training_bootstrap_partial_materialization_is_idempotent(
    roots, monkeypatch,
):
    plan = _plan(roots)
    run_dir, control, _ = roots
    run_dir.mkdir(parents=True)
    control.mkdir(parents=True)
    plan_path = control / "plan.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True))
    monkeypatch.setenv("OPERATOR_REPAIR_PLAN", str(plan_path))
    marker = run_dir / "fresh_lineage_marker.json"
    health = run_dir / "operator_repair_wandb_health.json"
    wandb_id = run_dir / "wandb_id"

    # Crash after the first member: replay completes exactly the two missing
    # members while authenticating the existing marker.
    marker.write_text(json.dumps(chain._fresh_lineage_marker_payload(plan)) + "\n")
    chain._materialize_training_bootstrap(
        plan, run_dir=run_dir, allow_create_missing=True,
    )
    assert health.is_file() and wandb_id.read_text().strip() == plan["wandb"]["training_run_id"]

    # Crash after marker+health but before ID: a second replay creates only ID.
    wandb_id.unlink()
    chain._materialize_training_bootstrap(
        plan, run_dir=run_dir, allow_create_missing=True,
    )
    assert wandb_id.read_text().strip() == plan["wandb"]["training_run_id"]
    before = (marker.read_bytes(), health.read_bytes(), wandb_id.read_bytes())
    chain._materialize_training_bootstrap(
        plan, run_dir=run_dir, allow_create_missing=True,
    )
    assert before == (marker.read_bytes(), health.read_bytes(), wandb_id.read_bytes())


def test_training_asset_verification_requeue_adopts_exact_receipt(
    roots, monkeypatch,
):
    plan = _plan(roots)
    control = roots[1]
    control.mkdir(parents=True)
    plan_path = control / "plan.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True))
    monkeypatch.setenv("OPERATOR_REPAIR_PLAN", str(plan_path))
    first = chain._publish_training_asset_verification(
        plan, stage="train_01", phase="pre",
    )
    second = chain._publish_training_asset_verification(
        plan, stage="train_01", phase="pre",
    )
    assert second == first
    path = chain._training_asset_verification_path(
        plan, stage="train_01", phase="pre",
    )
    tampered = json.loads(path.read_text())
    tampered["phase"] = "post"
    path.write_text(json.dumps(tampered))
    with pytest.raises(chain.OperatorRepairError, match="replay differs"):
        chain._publish_training_asset_verification(
            plan, stage="train_01", phase="pre",
        )


def test_post_training_asset_mutation_publishes_terminal_marker_and_requeue_rejects(
    roots, monkeypatch,
):
    plan = _plan(roots)
    control = roots[1]
    control.mkdir(parents=True)
    plan_path = control / "plan.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True))
    monkeypatch.setenv("OPERATOR_REPAIR_PLAN", str(plan_path))
    chain._publish_training_asset_verification(
        plan, stage="train_01", phase="pre",
    )
    _, pre_sha = chain._read_training_asset_verification(
        plan, stage="train_01", phase="pre",
    )
    pending = chain._begin_training_asset_post_transaction(
        plan, stage="train_01", restart_count=0,
        starting_checkpoint_step=0, pre_receipt_sha256=pre_sha,
    )
    changed_assets = copy.deepcopy(plan["assets"])
    changed_assets["cache_content"]["sha256"] = "f" * 64
    monkeypatch.setattr(chain, "_asset_receipt", lambda: changed_assets)
    with pytest.raises(chain.OperatorRepairError, match="durably terminal"):
        chain._publish_post_training_asset_verification(
            plan, pending=pending,
        )
    marker_path = Path(plan["paths"]["training_asset_failure"])
    marker = json.loads(marker_path.read_text())
    assert marker["error_category"] == "post_training_asset_verification_failed"
    assert marker["pending"]["sha256"] == chain.sha256_file(
        Path(pending["path"])
    )
    assert marker["pre_verification"]["sha256"] == pre_sha
    assert marker["restored_assets_cannot_reauthorize_this_lineage"] is True

    # Restoration cannot launder checkpoints produced inside the mutation
    # window: every future train/nontraining stage rejects the durable marker.
    monkeypatch.setattr(chain, "_asset_receipt", lambda: copy.deepcopy(plan["assets"]))
    with pytest.raises(chain.OperatorRepairError, match="lineage is terminal"):
        chain._reject_training_asset_failure(plan)
    with pytest.raises(chain.OperatorRepairError, match="lineage is terminal"):
        chain._stage_consolidate(plan)


def test_post_training_asset_marker_publication_failure_cannot_pass(
    roots, monkeypatch,
):
    plan = _plan(roots)
    control = roots[1]
    control.mkdir(parents=True)
    plan_path = control / "plan.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True))
    monkeypatch.setenv("OPERATOR_REPAIR_PLAN", str(plan_path))
    chain._publish_training_asset_verification(
        plan, stage="train_02", phase="pre",
    )
    _, pre_sha = chain._read_training_asset_verification(
        plan, stage="train_02", phase="pre",
    )
    pending = chain._begin_training_asset_post_transaction(
        plan, stage="train_02", restart_count=1,
        starting_checkpoint_step=500, pre_receipt_sha256=pre_sha,
    )
    monkeypatch.setattr(
        chain, "_publish_training_asset_verification",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            chain.OperatorRepairError("post hash changed")
        ),
    )
    monkeypatch.setattr(
        chain, "_publish_training_asset_failure",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("marker unavailable")),
    )
    with pytest.raises(chain.OperatorRepairError, match="could not be published"):
        chain._publish_post_training_asset_verification(
            plan, pending=pending,
        )
    assert not Path(plan["paths"]["training_asset_failure"]).exists()
    with pytest.raises(chain.OperatorRepairError, match="incomplete post-link"):
        chain._reject_training_asset_failure(plan)


def test_training_asset_pending_requires_exact_success_and_rejects_mismatch(
    roots, monkeypatch,
):
    plan = _plan(roots)
    control = roots[1]
    control.mkdir(parents=True)
    plan_path = control / "plan.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True))
    monkeypatch.setenv("OPERATOR_REPAIR_PLAN", str(plan_path))
    chain._publish_training_asset_verification(
        plan, stage="train_03", phase="pre",
    )
    _, pre_sha = chain._read_training_asset_verification(
        plan, stage="train_03", phase="pre",
    )
    pending = chain._begin_training_asset_post_transaction(
        plan, stage="train_03", restart_count=2,
        starting_checkpoint_step=1_000, pre_receipt_sha256=pre_sha,
    )
    with pytest.raises(chain.OperatorRepairError, match="incomplete post-link"):
        chain._reject_training_asset_failure(plan)
    chain._publish_post_training_asset_verification(plan, pending=pending)
    chain._reject_training_asset_failure(plan)

    _, complete_path = chain._training_asset_post_transaction_paths(
        plan, stage="train_03", restart_count=2,
    )
    exact_complete = complete_path.read_bytes()
    complete = json.loads(complete_path.read_text())
    complete["starting_checkpoint_step"] = 999
    complete_path.write_text(json.dumps(complete))
    with pytest.raises(chain.OperatorRepairError, match="completion changed"):
        chain._reject_training_asset_failure(plan)
    complete_path.write_bytes(exact_complete)
    pending_path = Path(pending["path"])
    changed_pending = json.loads(pending_path.read_text())
    changed_pending["pre_verification"]["assets_sha256"] = "0" * 64
    pending_path.write_text(json.dumps(changed_pending))
    with pytest.raises(chain.OperatorRepairError, match="pending transaction changed"):
        chain._reject_training_asset_failure(plan)


def test_pending_publication_failure_never_starts_post_scan_and_order_is_exact(
    roots, monkeypatch,
):
    plan = _plan(roots)
    control = roots[1]
    control.mkdir(parents=True)
    plan_path = control / "plan.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True))
    monkeypatch.setenv("OPERATOR_REPAIR_PLAN", str(plan_path))
    chain._publish_training_asset_verification(
        plan, stage="train_04", phase="pre",
    )
    _, pre_sha = chain._read_training_asset_verification(
        plan, stage="train_04", phase="pre",
    )
    pending_path, _ = chain._training_asset_post_transaction_paths(
        plan, stage="train_04", restart_count=0,
    )
    original_publish = chain.common.exclusive_json_write

    def fail_pending(path, value):
        if Path(path) == pending_path:
            raise OSError("pending unavailable")
        return original_publish(path, value)

    monkeypatch.setattr(chain.common, "exclusive_json_write", fail_pending)
    with pytest.raises(OSError, match="pending unavailable"):
        chain._begin_training_asset_post_transaction(
            plan, stage="train_04", restart_count=0,
            starting_checkpoint_step=2_000, pre_receipt_sha256=pre_sha,
        )
    assert not pending_path.exists()
    stage_source = inspect.getsource(chain._stage_train)
    training_begin = stage_source.rindex("_begin_training_asset_post_transaction")
    assert stage_source.index("subprocess.run") < training_begin
    assert training_begin < stage_source.rindex(
        "_publish_post_training_asset_verification"
    )


def test_in_training_preemption_before_post_transaction_remains_resumable(
    roots, monkeypatch,
):
    plan = _plan(roots)
    control = roots[1]
    control.mkdir(parents=True)
    plan_path = control / "plan.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True))
    monkeypatch.setenv("OPERATOR_REPAIR_PLAN", str(plan_path))
    chain._publish_training_asset_verification(
        plan, stage="train_05", phase="pre",
    )
    _, pre_sha = chain._read_training_asset_verification(
        plan, stage="train_05", phase="pre",
    )

    # A SIGKILL while srun is still active cannot execute parent-side Python.
    # Because PENDING starts only after srun returns, that crash surface has no
    # unresolved transaction and the same Slurm job may resume its checkpoint.
    pending0, complete0 = chain._training_asset_post_transaction_paths(
        plan, stage="train_05", restart_count=0,
    )
    assert not pending0.exists() and not complete0.exists()
    chain._reject_training_asset_failure(plan)

    pending = chain._begin_training_asset_post_transaction(
        plan, stage="train_05", restart_count=1,
        starting_checkpoint_step=2_500, pre_receipt_sha256=pre_sha,
    )
    chain._publish_post_training_asset_verification(plan, pending=pending)
    chain._reject_training_asset_failure(plan)


def test_endpoint_noop_replays_durable_wandb_fatal_and_accepts_success_reset(
    roots, monkeypatch,
):
    plan = _plan(roots)
    run_dir, control, _ = roots
    run_dir.mkdir(parents=True)
    control.mkdir(parents=True)
    plan_text = json.dumps(plan, sort_keys=True) + "\n"
    plan_path = control / "plan.json"
    plan_path.write_text(plan_text)
    (run_dir / "operator_repair_plan.json").write_text(plan_text)
    monkeypatch.setenv("OPERATOR_REPAIR_PLAN", str(plan_path))
    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
    chain._materialize_training_bootstrap(
        plan, run_dir=run_dir, allow_create_missing=True,
    )
    lineage = chain.sha256_file(plan_path)
    health = run_dir / "operator_repair_wandb_health.json"
    state = entry.initial_wandb_health_state(lineage)
    state["events"] = [
        {"global_step": step, "ok": step < 31_920}
        for step in range(20, 32_001, 20)
    ]
    health.write_text(json.dumps(state))
    with pytest.raises(chain.OperatorRepairError, match="durable five-failure"):
        chain._require_nonterminal_wandb_health(
            health, lineage_sha256=lineage,
        )
    (run_dir / "LATEST").write_text("32000\n")
    with pytest.raises(chain.OperatorRepairError, match="durable five-failure"):
        chain._stage_train(plan, "train_06", plan_path)
    state["events"][-1]["ok"] = True
    health.write_text(json.dumps(state))
    assert chain._require_nonterminal_wandb_health(
        health, lineage_sha256=lineage,
    ) == {"consecutive_failures": 0, "last_event_step": 32_000}


def test_logging_failure_marker_survives_checkpoint_and_blocks_requeue(
    tmp_path, monkeypatch,
):
    config = tmp_path / "failure.yaml"
    run_dir = tmp_path / "run"
    config.write_text("""
extends: base.yaml
run:
  name: durable_logging_failure
  steps: 1
  fresh_start_required: true
  reconcile_metrics_on_resume: true
  log_every: 1
  ckpt_every: 1
data:
  source: stub
  batch_per_gpu: 2
model:
  use_stubs: true
optim:
  warmup: 0
  spike_mult: 0
fsdp:
  shard: []
  replicate: []
""")

    def fail_log(*args, **kwargs):
        raise RuntimeError("simulated health-state persistence failure")

    monkeypatch.setattr(wandb_util, "log", fail_log)
    args = ["--config", str(config), "--run_dir", str(run_dir), "--no_wandb"]
    with pytest.raises(RuntimeError, match="simulated health-state"):
        train_loop.main(args)
    marker = json.loads((run_dir / "EXECUTION_FAILURE.json").read_text())
    assert marker["reason"] == "logging_failure"
    assert marker["global_step"] == 1
    assert (run_dir / "LATEST").read_text().strip() == "1"
    with pytest.raises(RuntimeError, match="lineage is terminal"):
        train_loop.main(args)


def test_logging_failure_marker_publication_failure_cannot_advance_latest(
    tmp_path, monkeypatch,
):
    config = tmp_path / "failure.yaml"
    run_dir = tmp_path / "run"
    config.write_text("""
extends: base.yaml
run:
  name: failed_marker_publication
  steps: 1
  fresh_start_required: true
  reconcile_metrics_on_resume: true
  log_every: 1
  ckpt_every: 1
data:
  source: stub
  batch_per_gpu: 2
model:
  use_stubs: true
optim:
  warmup: 0
  spike_mult: 0
fsdp:
  shard: []
  replicate: []
""")
    monkeypatch.setattr(
        wandb_util, "log",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("log failed")),
    )
    monkeypatch.setattr(
        train_loop, "_publish_execution_failure",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("marker failed")),
    )
    args = ["--config", str(config), "--run_dir", str(run_dir), "--no_wandb"]
    with pytest.raises(RuntimeError, match="could not durably publish"):
        train_loop.main(args)
    assert not (run_dir / "LATEST").exists()
    assert not (run_dir / "EXECUTION_FAILURE.json").exists()

def _metrics_bytes(count: int) -> bytes:
    return b"".join(
        (json.dumps({"global_step": step, "loss": float(step)}) + "\n").encode()
        for step in range(1, count + 1)
    )


def test_fresh_metrics_crash_tail_recovery_is_separate_and_idempotent(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    original = _metrics_bytes(5)
    (run_dir / "metrics.jsonl").write_bytes(original)
    identity = {
        "format": "loom-fresh-training-checkpoint-identity-v1",
        "latest_step": 3,
        "payload_global_step": 3,
        "config_hash": "0123456789abcdef",
    }
    first = _reconcile_fresh_metrics(
        run_dir, checkpoint_step=3, checkpoint_identity=identity,
    )
    assert first["format"] == "loom-fresh-metrics-rollback-v1"
    assert first["action"] == "ROLLBACK"
    assert (run_dir / "metrics.jsonl").read_bytes() == _metrics_bytes(3)
    assert Path(first["receipt_path"]).parent.name == "fresh_metrics_rollback"
    assert not (run_dir / "direct_formal_metrics_rollback").exists()

    no_tail = _reconcile_fresh_metrics(
        run_dir, checkpoint_step=3, checkpoint_identity=identity,
    )
    assert no_tail["action"] == "NONE"
    assert len(list((run_dir / "fresh_metrics_rollback").iterdir())) == 3

    # Replaying the same pre-replacement crash window adopts the exact same
    # immutable quarantine/receipt and performs the same one authorized edit.
    (run_dir / "metrics.jsonl").write_bytes(original)
    replay = _reconcile_fresh_metrics(
        run_dir, checkpoint_step=3, checkpoint_identity=identity,
    )
    assert replay == first
    assert (run_dir / "metrics.jsonl").read_bytes() == _metrics_bytes(3)


def test_fresh_metrics_flag_never_enables_direct_formal_decisions(tmp_path, monkeypatch):
    config = tmp_path / "tiny.yaml"
    run_dir = tmp_path / "run"
    config.write_text("""
extends: base.yaml
run:
  name: tiny_fresh_ledger
  steps: 2
  fresh_start_required: true
  reconcile_metrics_on_resume: true
  log_every: 1
  ckpt_every: 1
data:
  source: stub
  batch_per_gpu: 2
model:
  use_stubs: true
optim:
  warmup: 0
  spike_mult: 0
fsdp:
  shard: []
  replicate: []
""")

    def forbidden_decision(*args, **kwargs):
        raise AssertionError("fresh ledger flag must not invoke DirectFormal")

    monkeypatch.setattr(train_loop, "evaluate_direct_formal", forbidden_decision)
    args = ["--config", str(config), "--run_dir", str(run_dir), "--no_wandb"]
    assert train_loop.main(args) == 0
    assert not (run_dir / "fresh_metrics_rollback").exists()
    payload = torch.load(
        run_dir / "ckpt_000000002_rank0.pt", map_location="cpu", weights_only=False,
    )
    assert payload["fresh_lineage"]["metrics_ledger"] == {
        "format": "loom-fresh-metrics-rollback-v1",
        "reconcile_crash_tail_to_latest_checkpoint": True,
        "checkpoint_boundary_fsync": True,
        "direct_formal_decisions": False,
    }

    rows = (run_dir / "metrics.jsonl").read_bytes()
    future = b"".join(
        (json.dumps({"global_step": step, "loss": float(step)}) + "\n").encode()
        for step in (3, 4)
    )
    (run_dir / "metrics.jsonl").write_bytes(rows + future)
    assert train_loop.main(args) == 0
    assert (run_dir / "metrics.jsonl").read_bytes() == rows
    receipt_count = len(list((run_dir / "fresh_metrics_rollback").iterdir()))
    assert train_loop.main(args) == 0
    assert len(list((run_dir / "fresh_metrics_rollback").iterdir())) == receipt_count


def test_fresh_metrics_step0_bootstrap_recovers_precheckpoint_crash(
    tmp_path, monkeypatch,
):
    config = tmp_path / "tiny.yaml"
    run_dir = tmp_path / "run"
    config.write_text("""
extends: base.yaml
run:
  name: tiny_step0_recovery
  steps: 2
  fresh_start_required: true
  reconcile_metrics_on_resume: true
  log_every: 1
  ckpt_every: 500
data:
  source: stub
  batch_per_gpu: 2
model:
  use_stubs: true
optim:
  warmup: 0
  spike_mult: 0
fsdp:
  shard: []
  replicate: []
""")
    args = ["--config", str(config), "--run_dir", str(run_dir), "--no_wandb"]
    parsed = train_loop.parse_args(args)
    materialized = train_loop.load_config(parsed)
    digest = train_loop.config_hash(materialized)
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps(materialized, indent=2))
    (run_dir / "metrics.jsonl").write_bytes(_metrics_bytes(1))
    (run_dir / "HEARTBEAT").write_text("uncommitted\n")
    (run_dir / "fresh_lineage_marker.json").write_text(json.dumps({
        "format": "loom-fresh-training-lineage-marker-v1",
        "config_hash": digest,
        "run_name": "tiny_step0_recovery",
        "metrics_rollback_format": "loom-fresh-metrics-rollback-v1",
    }) + "\n")
    monkeypatch.setenv("LOOM_RESTART_COUNT", "1")
    assert train_loop.main(args) == 0
    assert [
        json.loads(line)["global_step"]
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
    ] == [1, 2]
    receipts = list((run_dir / "fresh_metrics_rollback").glob("rollback.*.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text())
    assert receipt["ledger"]["retained_rows"] == 0
    assert receipt["ledger"]["discarded_step_range"] == [1, 1]


def test_diary_has_append_only_schema_prior_result_and_new_preflight():
    diary = (chain.ROOT / "DIARY.md").read_text()
    assert "loom-experiment-diary-v1" in diary
    assert "append-only" in diary
    assert "550/1,200" in diary
    assert "45.8333%" in diary
    assert "long-horizon 13.0%" in diary
    assert "56,189 windows across 2,000 trajectories" in diary
    assert "12,800 draws" in diary and "6,400" in diary
    assert "7,938–8,056" in diary
    assert "no jobs submitted" in diary


def test_dry_run_builds_eleven_commands_without_creating_outputs(roots):
    plan = _plan(roots)
    payload = chain._dry_run_payload(plan)
    assert payload["job_count"] == 11
    assert payload["submission_performed"] is False
    assert [row["name"] for row in payload["dag"]] == [
        "train_01", "train_02", "train_03", "train_04", "train_05", "train_06", "consolidate",
        "eval_seed0", "eval_seed1", "eval_seed2", "merge",
    ]
    assert all(not path.exists() for path in roots)
