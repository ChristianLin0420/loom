from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest

from loom.eval import DEFAULT_LIBERO_SUITES, EpisodeResult, EvalProtocol
from loom.eval.runner import aggregate, iter_work
from loom.train.loop import read_config
from scripts import r0_e2e_formal_chain as chain


def _formal_cfg():
    return copy.deepcopy(
        read_config(chain.ROOT / "configs" / "r0a_dual_code_formal.yaml")
    )


def _plan(tmp_path, monkeypatch):
    return chain.build_plan(
        config_path=chain.CANONICAL_FORMAL_CONFIG,
        run_dir=(tmp_path / "run").resolve(),
        control_dir=(tmp_path / "control").resolve(),
        artifact_root=(tmp_path / "artifacts").resolve(),
        group="r0-dual-seed0-01234567",
    )


def test_formal_config_requires_dual_real_seed0_and_split_horizons():
    got = chain.validate_formal_config(
        _formal_cfg(), project=chain.PROJECT, step32=32_000, step40=40_000,
    )
    assert got == {
        "resolved_config_hash": chain.CANONICAL_FORMAL_RESOLVED_HASH,
        "project": chain.PROJECT,
        "seed": 0,
        "steps": 40_000,
        "schedule_horizon": 32_000,
        "max_updates": 40_000,
        "fresh_start_required": True,
        "require_online_wandb": True,
        "ckpt_every": 500,
        "keep_last": 20,
        "use_stubs": False,
        "act_decode_from": "dual_q_action_proposal",
        "act_enabled": True,
        "act_weight": 1.0,
        "act_align_to": "q_a",
        "train_modules": list(chain.DirectFormalGate().expected_gradient_modules),
        "direct_formal": chain._expected_direct_formal_config(
            chain.DirectFormalGate()
        ),
        "evaluation_gate": _formal_cfg()["evaluation_gate"],
    }

    for section, key, bad in (
        ("run", "project", "loom"),
        ("run", "seed", 1),
        ("run", "steps", 32_000),
        ("run", "schedule_horizon", 40_000),
        ("run", "max_updates", 32_000),
        ("run", "fresh_start_required", False),
        ("run", "require_online_wandb", False),
        ("run", "ckpt_every", 1_000),
        ("run", "keep_last", 19),
        ("model", "use_stubs", "auto"),
        ("losses.act", "decode_from", "q_action"),
        ("losses.act", "weight", 0.5),
        ("losses.dyn", "weight", 0.5),
        ("optim", "lr", 1.0e-4),
        ("data", "source", "robotwin"),
        ("direct_formal", "primary", ["loss/dyn"]),
        ("direct_formal.health", "gnorm_bank_strict_gt", 0.0),
        ("evaluation_gate", "seed0_stretch_successes_gte", 163),
    ):
        cfg = _formal_cfg()
        node = cfg
        for part in section.split("."):
            node = node[part]
        node[key] = bad
        with pytest.raises(chain.ChainError, match="frozen method/extension"):
            chain.validate_formal_config(
                cfg, project=chain.PROJECT, step32=32_000, step40=40_000,
            )


def test_plan_freezes_single_lineage_parallel_seeds_and_no_eval_selection(
    tmp_path, monkeypatch,
):
    plan = _plan(tmp_path, monkeypatch)
    assert plan["eligibility"] == "one_direct_formal_training_lineage_no_smoke"
    assert plan["method"]["checkpoint_selection_uses_eval"] is False
    assert plan["steps"] == {
        "schedule_horizon": 32_000,
        "initial_stop": 32_000,
        "hard_cap": 40_000,
        "initial_links": 3,
        "extension_links": 1,
    }
    assert plan["evaluation"]["seeds"] == [0, 1, 2]
    assert plan["evaluation"]["parallel_singleton_seed_jobs"] is True
    assert plan["evaluation"]["total_episodes"] == 1_200
    assert plan["evaluation"]["runtime"]["python"] == str(
        chain.LIBERO_EVAL_PYTHON
    )
    assert plan["evaluation"]["identity_normalization"] == (
        chain.EVAL_IDENTITY_NORMALIZATION
    )
    assert plan["wandb"]["project"] == "loom-r0-e2e-scratch"
    assert plan["wandb"]["require_online"] is True
    assert plan["failure_policy"]["hard_cap_without_convergence"] == (
        "terminate_without_evaluation"
    )
    assert "gate40_valid_nonconverged" not in plan["failure_policy"]
    assert plan["baseline_comparison"]["baseline"]["successes"] == 447
    assert plan["baseline_comparison"]["bootstrap"]["matrix_sha256"] == (
        chain.BOOTSTRAP_MATRIX_SHA256
    )
    assert len(plan["wandb"]["training_run_id"]) == 16
    assert all(ch in "0123456789abcdef" for ch in plan["wandb"]["training_run_id"])
    assert plan["orchestration_source_closure"]["scheme"] == (
        "sha256(path-nul-sha256-nul)-v1"
    )
    assert set(plan["orchestration_source_closure"]["files"]) == set(
        chain.ORCHESTRATION_SOURCE_FILES
    )


def test_plan_rejects_nested_run_control_and_artifact_paths(tmp_path, monkeypatch):
    with pytest.raises(chain.ChainError, match="pairwise non-nested"):
        chain.build_plan(
            config_path=chain.CANONICAL_FORMAL_CONFIG,
            run_dir=(tmp_path / "run").resolve(),
            control_dir=(tmp_path / "run" / "control").resolve(),
            artifact_root=(tmp_path / "artifacts").resolve(),
            group="lineage",
        )

    with pytest.raises(chain.ChainError, match="requires online W&B"):
        chain.build_plan(
            config_path=chain.CANONICAL_FORMAL_CONFIG,
            run_dir=(tmp_path / "run2").resolve(),
            control_dir=(tmp_path / "control2").resolve(),
            artifact_root=(tmp_path / "artifacts2").resolve(),
            group="lineage",
            require_online=False,
        )

    noncanonical = tmp_path / "copied_formal.yaml"
    noncanonical.write_bytes(chain.CANONICAL_FORMAL_CONFIG.read_bytes())
    with pytest.raises(chain.ChainError, match="canonical path"):
        chain.build_plan(
            config_path=noncanonical.resolve(),
            run_dir=(tmp_path / "run3").resolve(),
            control_dir=(tmp_path / "control3").resolve(),
            artifact_root=(tmp_path / "artifacts3").resolve(),
            group="lineage",
        )


def test_dag_is_afterok_serial_train_parallel_eval_then_join(tmp_path, monkeypatch):
    specs = chain._plan_stage_specs(_plan(tmp_path, monkeypatch))
    by_name = {spec["name"]: spec for spec in specs}
    assert by_name["train_01"]["depends_on"] == []
    assert by_name["train_02"]["depends_on"] == ["train_01"]
    assert by_name["train_03"]["depends_on"] == ["train_02"]
    assert by_name["gate32"]["depends_on"] == ["train_03"]
    assert by_name["extension"]["depends_on"] == ["gate32"]
    assert by_name["gatefinal"]["depends_on"] == ["extension"]
    assert by_name["consolidate"]["depends_on"] == ["gatefinal"]
    for seed in chain.SEEDS:
        assert by_name[f"eval_seed{seed}"]["depends_on"] == ["consolidate"]
    assert by_name["merge"]["depends_on"] == [
        "eval_seed0", "eval_seed1", "eval_seed2",
    ]


def test_train_link_resume_policy_survives_slurm_requeue(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)
    first = chain.train_stage_policy(plan, "train_01", restart_count=0)
    assert first == {
        "skip": False,
        "fresh": True,
        "require_target": False,
        "stop_at": 32_000,
        "resume": "never",
    }
    requeued = chain.train_stage_policy(plan, "train_01", restart_count=1)
    assert requeued["fresh"] is False
    assert requeued["resume"] == "must"
    assert requeued["require_target"] is False
    final_initial = chain.train_stage_policy(plan, "train_03", restart_count=0)
    assert final_initial["fresh"] is False
    assert final_initial["resume"] == "must"
    assert final_initial["require_target"] is True
    with pytest.raises(chain.ChainError, match="non-negative"):
        chain.train_stage_policy(plan, "train_01", restart_count=-1)


def test_submit_holds_whole_dag_uses_afterok_then_releases(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)
    calls = []
    next_id = 1000

    def fake_run(command, **kwargs):
        nonlocal next_id
        calls.append(list(command))
        if command[0] == "sbatch":
            next_id += 1
            return subprocess.CompletedProcess(command, 0, stdout=f"{next_id}\n", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    result = chain.submit_plan(plan, run=fake_run)
    assert result["released"] is True
    sbatch_calls = [call for call in calls if call[0] == "sbatch"]
    assert len(sbatch_calls) == 11
    assert all("--hold" in call and "--parsable" in call for call in sbatch_calls)
    assert all("--kill-on-invalid-dep=yes" in call for call in sbatch_calls)
    assert not any(part.startswith("--dependency") for part in sbatch_calls[0])
    assert "--dependency=afterok:1001" in sbatch_calls[1]
    eval_calls = {stage: result["commands"][stage] for stage in (
        "eval_seed0", "eval_seed1", "eval_seed2",
    )}
    assert all(
        "--dependency=afterok:" + result["jobs"]["consolidate"] in command
        for command in eval_calls.values()
    )
    merge_dep = "--dependency=afterok:" + ":".join(
        result["jobs"][f"eval_seed{seed}"] for seed in chain.SEEDS
    )
    assert merge_dep in result["commands"]["merge"]
    assert calls[-1] == [
        "scontrol", "release", ",".join(result["jobs"].values()),
    ]
    assert (Path(plan["lineage"]["control_dir"]) / "released.json").is_file()


def test_partial_submission_failure_cancels_every_held_job(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)
    calls = []
    submitted = 0

    def fake_run(command, **kwargs):
        nonlocal submitted
        calls.append(list(command))
        if command[0] == "sbatch":
            submitted += 1
            if submitted == 3:
                raise subprocess.CalledProcessError(1, command)
            return subprocess.CompletedProcess(command, 0, stdout=f"20{submitted}\n", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    with pytest.raises(subprocess.CalledProcessError):
        chain.submit_plan(plan, run=fake_run)
    assert calls[-1] == ["scancel", "201", "202"]


def test_direct_convergence_codes_extend_only_moving_and_evaluate_only_pass():
    assert chain.classify_gate32(0) == "select_step_32000"
    assert chain.classify_gate32(1) == "extend_to_step_40000"
    assert chain.classify_gate32(3) == "abort_no_evaluation"
    assert chain.classify_gate32(2) == "invalid_stop_descendants"
    assert chain.classify_gate32(4) == "invalid_stop_descendants"

    assert chain.classify_terminal(0) == "select_first_passing_checkpoint"
    assert chain.classify_terminal(3) == "abort_no_evaluation"
    assert chain.classify_terminal(1) == "invalid_stop_descendants"
    assert chain.classify_terminal(2) == "invalid_stop_descendants"

    source = (chain.ROOT / "scripts" / "r0_e2e_formal_chain.py").read_text()
    assert '"scripts/direct_formal_convergence.py"' in source
    assert '"scripts/convergence.py"' not in source
    assert "evaluate_fixed_hard_cap_when_nonconverged" not in source


def _controller_fixture(tmp_path, monkeypatch, *, status):
    step = 32_000
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    metrics = run_dir / "metrics.jsonl"
    metrics.write_text("{}\n")
    (run_dir / "config.json").write_text("{}\n")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}\n")
    monkeypatch.setenv("FORMAL_PLAN", str(plan_path))
    metrics_identity = {
        "path": str(metrics.resolve()),
        "bytes": metrics.stat().st_size,
        "sha256": chain.sha256_file(metrics),
    }
    direct = {
        "format": chain.DIRECT_FORMAL_FORMAT,
        "status": status,
        "reason": status.lower(),
        "current_step": step,
        "decision_step": step if status == "PASS" else None,
        "next_check_step": None,
        "gate": chain.DirectFormalGate().as_dict(),
        "input": {"rows": step, "minimum_step": 1, "maximum_step": step},
        "evaluations": [{"step": step, "status": status}],
        "metrics_source": metrics_identity,
    }
    plan = {
        "config": {"resolved_config_hash": "controller-test-config"},
        "steps": {
            "schedule_horizon": step, "initial_stop": step,
            "hard_cap": 40_000,
        },
        "lineage": {"run_dir": str(run_dir)},
        "paths": {
            "gate32": str(tmp_path / "gate32.json"),
            "endpoint": str(tmp_path / "endpoint.json"),
        },
    }
    in_loop = {
        **direct,
        "config_hash": plan["config"]["resolved_config_hash"],
        "fresh_lineage": chain._expected_fresh_lineage(plan),
    }
    chain._direct_boundary_path(run_dir, step).write_text(json.dumps(in_loop))
    return plan, plan_path, direct, metrics_identity


def _controller_recomputation(plan, direct, metrics_identity):
    in_loop = chain._direct_boundary_path(
        Path(plan["lineage"]["run_dir"]), int(direct["current_step"]),
    )
    return {
        "cli": str(chain.ROOT / "scripts" / "direct_formal_convergence.py"),
        "stdout_sha256": "a" * 64,
        "in_loop_receipt": str(in_loop),
        "in_loop_receipt_sha256": chain.sha256_file(in_loop),
        "metrics_source": metrics_identity,
    }


@pytest.mark.parametrize(
    ("status", "returncode", "action"),
    [
        ("PASS", 0, "select_step_32000"),
        ("MOVING", 1, "extend_to_step_40000"),
    ],
)
def test_gate32_crash_window_adopts_authenticated_success_without_recompute(
    tmp_path, monkeypatch, status, returncode, action,
):
    plan, plan_path, direct, metrics_identity = _controller_fixture(
        tmp_path, monkeypatch, status=status,
    )
    receipt = {
        "format_version": chain.FORMAT_VERSION,
        "kind": "r0_e2e_convergence_gate",
        "step": 32_000,
        "returncode": returncode,
        "action": action,
        "direct_receipt": direct,
        "recomputation": _controller_recomputation(
            plan, direct, metrics_identity,
        ),
        "metrics_sha256": metrics_identity["sha256"],
        "run_config_sha256": chain.sha256_file(Path(plan["lineage"]["run_dir"]) / "config.json"),
        "plan_sha256": chain.sha256_file(plan_path),
    }
    Path(plan["paths"]["gate32"]).write_text(json.dumps(receipt))
    monkeypatch.setattr(
        chain, "_run_convergence",
        lambda *args, **kwargs: pytest.fail("authenticated gate must be adopted"),
    )
    assert chain._stage_gate32(plan) == 0

    receipt["action"] = "abort_no_evaluation"
    Path(plan["paths"]["gate32"]).write_text(json.dumps(receipt))
    with pytest.raises(chain.ChainError, match="failed authentication"):
        chain._stage_gate32(plan)


@pytest.mark.parametrize(
    ("status", "returncode", "action"),
    [
        ("ABORT", 3, "abort_no_evaluation"),
        ("INVALID", 2, "invalid_stop_descendants"),
    ],
)
def test_gate32_crash_window_preserves_authenticated_terminal_failure(
    tmp_path, monkeypatch, status, returncode, action,
):
    plan, plan_path, direct, metrics_identity = _controller_fixture(
        tmp_path, monkeypatch, status=status,
    )
    receipt = {
        "format_version": chain.FORMAT_VERSION,
        "kind": "r0_e2e_convergence_gate",
        "step": 32_000,
        "returncode": returncode,
        "action": action,
        "direct_receipt": direct,
        "recomputation": _controller_recomputation(
            plan, direct, metrics_identity,
        ),
        "metrics_sha256": metrics_identity["sha256"],
        "run_config_sha256": chain.sha256_file(Path(plan["lineage"]["run_dir"]) / "config.json"),
        "plan_sha256": chain.sha256_file(plan_path),
    }
    Path(plan["paths"]["gate32"]).write_text(json.dumps(receipt))
    with pytest.raises(chain.ChainError, match="existing 32k direct gate is terminal"):
        chain._stage_gate32(plan)


def test_gate32_rejects_correlated_controller_abort_to_pass_conversion(
    tmp_path, monkeypatch,
):
    plan, plan_path, direct, metrics_identity = _controller_fixture(
        tmp_path, monkeypatch, status="ABORT",
    )
    direct["status"] = "PASS"  # reason/evaluation/decision remain ABORT-shaped
    receipt = {
        "format_version": chain.FORMAT_VERSION,
        "kind": "r0_e2e_convergence_gate",
        "step": 32_000,
        "returncode": 0,
        "action": "select_step_32000",
        "direct_receipt": direct,
        "recomputation": _controller_recomputation(
            plan, direct, metrics_identity,
        ),
        "metrics_sha256": metrics_identity["sha256"],
        "run_config_sha256": chain.sha256_file(
            Path(plan["lineage"]["run_dir"]) / "config.json"
        ),
        "plan_sha256": chain.sha256_file(plan_path),
    }
    Path(plan["paths"]["gate32"]).write_text(json.dumps(receipt))
    with pytest.raises(chain.ChainError, match="direct_in_loop_core"):
        chain._stage_gate32(plan)


@pytest.mark.parametrize(
    ("status", "returncode", "action", "endpoint_status", "validated"),
    [
        ("ABORT", 3, "abort_no_evaluation", "ABORT_NO_EVALUATION", True),
        ("INVALID", 2, "invalid_stop_descendants", "INVALID_TERMINAL_GATE", False),
    ],
)
def test_gatefinal_crash_window_replays_terminal_and_rejects_tamper(
    tmp_path, monkeypatch, status, returncode, action, endpoint_status, validated,
):
    plan, plan_path, direct, metrics_identity = _controller_fixture(
        tmp_path, monkeypatch, status=status,
    )
    run_dir = Path(plan["lineage"]["run_dir"])
    in_loop = chain._direct_boundary_path(run_dir, 32_000)
    convergence = {
        "format_version": chain.FORMAT_VERSION,
        "kind": "r0_e2e_convergence_gate",
        "step": 32_000,
        "returncode": returncode,
        "action": action,
        "direct_receipt": direct,
        "recomputation": _controller_recomputation(
            plan, direct, metrics_identity,
        ),
        "metrics_sha256": metrics_identity["sha256"],
        "run_config_sha256": chain.sha256_file(run_dir / "config.json"),
        "plan_sha256": chain.sha256_file(plan_path),
    }
    endpoint = {
        "format_version": chain.FORMAT_VERSION,
        "kind": "r0_e2e_terminal_endpoint",
        "execution_validated": validated,
        "eligible_for_eval": False,
        "step": 32_000,
        "status": endpoint_status,
        "plan_sha256": chain.sha256_file(plan_path),
        "convergence": convergence,
        "checkpoint_selection_used_eval": False,
    }
    endpoint_path = Path(plan["paths"]["endpoint"])
    endpoint_path.write_text(json.dumps(endpoint))
    assert chain._authenticate_terminal_failure_endpoint(plan) == endpoint
    monkeypatch.setattr(
        chain, "_read_gate32",
        lambda *args, **kwargs: pytest.fail("existing terminal must be replayed"),
    )
    with pytest.raises(chain.ChainError, match=endpoint_status):
        chain._stage_gatefinal(plan)

    endpoint["status"] = "FIRST_PASS_32000"
    endpoint_path.write_text(json.dumps(endpoint))
    with pytest.raises(chain.ChainError, match="failed authentication"):
        chain._stage_gatefinal(plan)

    if status == "ABORT":
        converted = copy.deepcopy(endpoint)
        converted["status"] = "INVALID_TERMINAL_GATE"
        converted["execution_validated"] = False
        converted["convergence"]["returncode"] = 2
        converted["convergence"]["action"] = "invalid_stop_descendants"
        converted["convergence"]["direct_receipt"]["status"] = "INVALID"
        endpoint_path.write_text(json.dumps(converted))
        with pytest.raises(chain.ChainError, match="direct_in_loop_core"):
            chain._stage_gatefinal(plan)
    else:
        converted = copy.deepcopy(endpoint)
        converted["status"] = "FIRST_PASS_32000"
        converted["execution_validated"] = True
        converted["eligible_for_eval"] = True
        converted["convergence"]["returncode"] = 0
        converted["convergence"]["action"] = "select_first_passing_checkpoint"
        converted["convergence"]["direct_receipt"]["status"] = "PASS"
        converted["convergence"]["direct_receipt"]["decision_step"] = 32_000
        run_dir = Path(plan["lineage"]["run_dir"])
        (run_dir / "LATEST").write_text("32000\n")
        for rank in range(chain.WORLD_SIZE):
            (run_dir / f"ckpt_{32_000:09d}_rank{rank}.pt").write_bytes(
                f"rank-{rank}".encode()
            )
        converted["checkpoint_shards"] = chain._checkpoint_shard_receipt(
            chain._checkpoint_shards(run_dir, 32_000)
        )
        endpoint_path.write_text(json.dumps(converted))
        with pytest.raises(chain.ChainError, match="direct_in_loop_core"):
            chain._stage_gatefinal(plan)


def test_gatefinal_crash_window_adopts_authenticated_pass_without_recompute(
    tmp_path, monkeypatch,
):
    endpoint_path = tmp_path / "endpoint.json"
    endpoint_path.write_text(json.dumps({
        "format_version": chain.FORMAT_VERSION,
        "kind": "r0_e2e_terminal_endpoint",
        "eligible_for_eval": True,
    }))
    plan = {"paths": {"endpoint": str(endpoint_path)}}
    adopted = []
    monkeypatch.setattr(
        chain, "_authenticate_endpoint", lambda value: adopted.append(value),
    )
    monkeypatch.setattr(
        chain, "_read_gate32",
        lambda *args, **kwargs: pytest.fail("existing PASS must be adopted"),
    )
    assert chain._stage_gatefinal(plan) == 0
    assert adopted == [plan]


def test_in_loop_direct_receipt_authenticates_exact_selected_latest(tmp_path):
    step = 32_500
    plan = {
        "config": {"resolved_config_hash": "0123456789abcdef"},
        "steps": {"schedule_horizon": 32_000, "hard_cap": 40_000},
        "lineage": {"run_dir": str(tmp_path)},
    }
    receipt = {
        "format": chain.DIRECT_FORMAL_FORMAT,
        "status": "PASS",
        "reason": "first_passing_checkpoint",
        "current_step": step,
        "decision_step": step,
        "next_check_step": None,
        "gate": chain.DirectFormalGate().as_dict(),
        "input": {"rows": step, "minimum_step": 1, "maximum_step": step},
        "evaluations": [{"step": step, "status": "PASS"}],
        "config_hash": "0123456789abcdef",
        "fresh_lineage": chain._expected_fresh_lineage(plan),
    }
    path = chain._direct_boundary_path(tmp_path, step)
    path.write_text(json.dumps(receipt))
    got, got_path = chain._read_direct_boundary_receipt(plan, step)
    assert got["decision_step"] == step
    assert got_path == path

    receipt["decision_step"] = 32_000
    path.write_text(json.dumps(receipt))
    with pytest.raises(chain.ChainError, match="exact current boundary"):
        chain._read_direct_boundary_receipt(plan, step)


def test_checkpoint_shards_use_numeric_rank_order_and_selected_must_be_latest(tmp_path):
    (tmp_path / "LATEST").write_text("32500\n")
    for rank in range(16):
        (tmp_path / f"ckpt_000032500_rank{rank}.pt").write_bytes(b"x")
    shards = chain._checkpoint_shards(tmp_path, 32_500)
    assert [path.name for path in shards] == [
        f"ckpt_000032500_rank{rank}.pt" for rank in range(16)
    ]
    with pytest.raises(chain.ChainError, match="expected exact endpoint"):
        chain._checkpoint_shards(tmp_path, 32_000)


def test_endpoint_reauthenticates_plan_pass_and_every_selected_shard(
    tmp_path, monkeypatch,
):
    plan, plan_path, direct, metrics_identity = _controller_fixture(
        tmp_path, monkeypatch, status="PASS",
    )
    run_dir = Path(plan["lineage"]["run_dir"])
    step = 32_000
    (run_dir / "LATEST").write_text(f"{step}\n")
    for rank in range(chain.WORLD_SIZE):
        (run_dir / f"ckpt_{step:09d}_rank{rank}.pt").write_bytes(
            f"rank-{rank}".encode()
        )
    endpoint_path = Path(plan["paths"]["endpoint"])
    convergence = {
        "format_version": chain.FORMAT_VERSION,
        "kind": "r0_e2e_convergence_gate",
        "step": step,
        "action": "select_first_passing_checkpoint",
        "returncode": 0,
        "direct_receipt": direct,
        "recomputation": _controller_recomputation(
            plan, direct, metrics_identity,
        ),
        "metrics_sha256": metrics_identity["sha256"],
        "run_config_sha256": chain.sha256_file(run_dir / "config.json"),
        "plan_sha256": chain.sha256_file(plan_path),
    }
    shards = chain._checkpoint_shards(run_dir, step)
    endpoint = {
        "format_version": chain.FORMAT_VERSION,
        "kind": "r0_e2e_terminal_endpoint",
        "execution_validated": True,
        "eligible_for_eval": True,
        "checkpoint_selection_used_eval": False,
        "step": step,
        "status": f"FIRST_PASS_{step}",
        "plan_sha256": chain.sha256_file(plan_path),
        "convergence": convergence,
        "checkpoint_shards": chain._checkpoint_shard_receipt(shards),
    }
    endpoint_path.write_text(json.dumps(endpoint))
    got, digest, shard_map = chain._authenticate_endpoint(plan)
    assert got == endpoint
    assert digest == chain.sha256_file(endpoint_path)
    assert shard_map == endpoint["checkpoint_shards"]

    (run_dir / f"ckpt_{step:09d}_rank7.pt").write_bytes(b"changed")
    with pytest.raises(chain.ChainError, match="shard bytes changed"):
        chain._authenticate_endpoint(plan)


def test_controller_recomputes_with_direct_cli_and_matches_in_loop_receipt(
    tmp_path, monkeypatch,
):
    step = 32_000
    plan = {
        "config": {"resolved_config_hash": "0123456789abcdef"},
        "steps": {"schedule_horizon": 32_000, "hard_cap": 40_000},
        "lineage": {"run_dir": str(tmp_path)},
    }
    (tmp_path / "LATEST").write_text(f"{step}\n")
    for rank in range(16):
        (tmp_path / f"ckpt_{step:09d}_rank{rank}.pt").write_bytes(b"x")
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text("{}\n")
    core = {
        "format": chain.DIRECT_FORMAL_FORMAT,
        "status": "PASS",
        "reason": "first_passing_checkpoint",
        "current_step": step,
        "decision_step": step,
        "next_check_step": None,
        "gate": chain.DirectFormalGate().as_dict(),
        "input": {"rows": step, "minimum_step": 1, "maximum_step": step},
        "evaluations": [{"step": step, "status": "PASS"}],
    }
    in_loop = {
        **core,
        "config_hash": "0123456789abcdef",
        "fresh_lineage": chain._expected_fresh_lineage(plan),
    }
    chain._direct_boundary_path(tmp_path, step).write_text(json.dumps(in_loop))
    direct = {**core, "metrics_source": {
        "path": str(metrics.resolve()),
        "bytes": metrics.stat().st_size,
        "sha256": chain.sha256_file(metrics),
    }}
    commands = []

    def fake_run(command, **kwargs):
        commands.append(list(command))
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(direct) + "\n", stderr="",
        )

    monkeypatch.setattr(chain.subprocess, "run", fake_run)
    rc, got, evidence = chain._run_convergence(plan, step)
    assert rc == 0 and got == direct
    assert commands[0][-2:] == ["--current-step", str(step)]
    assert commands[0][1].endswith("scripts/direct_formal_convergence.py")
    assert evidence["in_loop_receipt_sha256"] == chain.sha256_file(
        chain._direct_boundary_path(tmp_path, step)
    )


def test_exclusive_json_publish_never_replaces(tmp_path):
    path = tmp_path / "receipt.json"
    chain.exclusive_json_write(path, {"first": True})
    with pytest.raises(FileExistsError):
        chain.exclusive_json_write(path, {"first": False})
    assert json.loads(path.read_text()) == {"first": True}


def _baseline_plan_fragment():
    return {
        "baseline_comparison": {
            "baseline": chain._authenticate_baseline(chain.CANONICAL_BASELINE_ROOT),
            "thresholds": _formal_cfg()["evaluation_gate"],
        }
    }


def test_canonical_baseline_and_fixed_bootstrap_matrix_are_exact():
    plan = _baseline_plan_fragment()
    baseline = plan["baseline_comparison"]["baseline"]
    assert baseline["episodes"] == 1_200
    assert baseline["successes"] == 447
    assert {
        int(seed): row["sha256"] for seed, row in baseline["files"].items()
    } == chain.BASELINE_RESULT_SHA256

    task_keys = sorted(
        f"{suite}/task={task:02d}"
        for suite in DEFAULT_LIBERO_SUITES for task in range(10)
    )
    matrix, receipt = chain._suite_stratified_bootstrap_matrix(task_keys)
    assert tuple(matrix.shape) == (10_000, 40)
    assert receipt["sha256"] == chain.BOOTSTRAP_MATRIX_SHA256
    assert receipt["lower_quantile"] == 0.025
    assert receipt["upper_quantile"] == 0.975
    assert receipt["lower_interpolation"] == "lower"
    assert receipt["upper_interpolation"] == "higher"


def test_paired_baseline_gate_is_strict_and_scientific_fail_still_reports():
    plan = _baseline_plan_fragment()
    baseline = chain._baseline_rows(plan)
    comparison = chain.paired_baseline_comparison(plan, baseline, baseline)
    assert comparison["status"] == "FAIL"
    assert comparison["passed"] is False
    assert comparison["scientific_failure_still_publishes"] is True
    assert comparison["overall"]["delta_percentage_points"] == 0.0
    assert comparison["paired_task_bootstrap"]["ci_low_percentage_points"] == 0.0
    assert comparison["pairing"] == {
        "key_fields": ["bench", "suite", "task_id", "episode", "seed"],
        "paired_episodes": 1_200,
        "new_only": 0,
        "old_only": 0,
        "tie_success": 447,
        "tie_failure": 753,
        "rng_identity_equal": True,
    }


def test_paired_task_balanced_improvement_passes_all_frozen_thresholds():
    plan = _baseline_plan_fragment()
    baseline = chain._baseline_rows(plan)
    candidate = copy.deepcopy(baseline)
    by_task = {}
    for key in sorted(candidate):
        by_task.setdefault((key[1], key[2]), []).append(key)
    # At least one paired win in every task makes every task-level delta > 0.
    for keys in by_task.values():
        key = next(key for key in keys if not candidate[key]["success"])
        candidate[key]["success"] = True
    # Independently satisfy the prospectively fixed seed-0 stretch count.
    seed0_successes = sum(
        bool(row["success"]) for key, row in candidate.items() if key[-1] == 0
    )
    for key in sorted(candidate):
        if seed0_successes >= 164:
            break
        if key[-1] == 0 and not candidate[key]["success"]:
            candidate[key]["success"] = True
            seed0_successes += 1
    comparison = chain.paired_baseline_comparison(plan, candidate, baseline)
    assert comparison["status"] == "PASS"
    assert comparison["passed"] is True
    assert comparison["paired_task_bootstrap"]["ci_low_percentage_points"] > 0.0
    assert comparison["per_seed_candidate_successes"]["0"] >= 164
    assert all(comparison["checks"].values())


@pytest.mark.parametrize(
    ("owner", "key", "bad"),
    [
        ("policy", "gripper_dwell", 2),
        ("policy", "decoder_samples", 2),
        ("policy", "duration_normalize_segments", True),
        ("identity", "backend", {"requested": "auto", "resolved": "libero"}),
        ("identity", "policy_kw", {"allow_stub": False}),
        ("meta", "policy_seed_scheme", "legacy-global-rng"),
    ],
)
def test_candidate_eval_method_identity_is_explicit_and_fail_closed(
    tmp_path, owner, key, bad,
):
    checkpoint_receipt = {
        "checkpoint": str((tmp_path / "ckpt.pt").resolve()),
        "step": 32_000,
    }
    path = _write_seed_result(tmp_path, checkpoint_receipt, 0)
    blob = json.loads(path.read_text())
    if owner == "policy":
        blob["meta"]["policy"][key] = bad
    elif owner == "identity":
        blob["meta"]["eval_identity"][key] = bad
    else:
        blob["meta"][key] = bad
    with pytest.raises(chain.ChainError, match="method identity mismatch"):
        chain._validate_exact_eval_blob(
            blob, seed=0, label="candidate seed 0",
            identity_profile="current_candidate",
        )


def test_eval_identity_records_historical_implicit_and_current_materialized_body(
    tmp_path,
):
    receipt = {
        "checkpoint": str((tmp_path / "ckpt.pt").resolve()),
        "step": 32_000,
    }
    blob = json.loads(_write_seed_result(tmp_path, receipt, 0).read_text())
    chain._validate_eval_method_identity(
        blob, label="candidate", identity_profile="current_candidate",
    )
    historical = copy.deepcopy(blob)
    historical["meta"]["eval_identity"]["policy_kw"].pop("embodiment")
    chain._validate_eval_method_identity(
        historical, label="historical", identity_profile="historical_baseline",
    )
    with pytest.raises(chain.ChainError, match="identity_policy_kw"):
        chain._validate_eval_method_identity(
            historical, label="candidate", identity_profile="current_candidate",
        )


def test_formal_eval_command_is_exact_and_resumable(tmp_path):
    plan = {"evaluation": {
        "tasks_per_suite": 10,
        "episodes_per_task": 10,
        "gripper_dwell": 1,
        "decoder_samples": 1,
        "duration_normalize_segments": False,
    }}
    command, env = chain._formal_eval_command(
        plan, seed=2, checkpoint=tmp_path / "ckpt.pt", out_dir=tmp_path / "seed2",
    )
    assert command[0] == str(chain.LIBERO_EVAL_PYTHON)
    assert "--no-resume" not in command
    assert command[command.index("--seeds") + 1] == "2"
    assert command[command.index("--backend") + 1] == "libero"
    assert "--require-real" in command and "--op-stats" in command
    assert "--embodiment" not in command  # runner materializes bench default
    assert env["HF_HUB_OFFLINE"] == "1"


def test_eval_requeue_adopts_complete_authenticated_result_without_rerun(
    tmp_path, monkeypatch,
):
    checkpoint = tmp_path / "ckpt.pt"
    checkpoint.write_bytes(b"checkpoint")
    out_dir = tmp_path / "seed0"
    out_dir.mkdir()
    result = out_dir / "results.json"
    result.write_text("{}")
    receipt_path = tmp_path / "seed0-receipt.json"
    checkpoint_receipt = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": chain.sha256_file(checkpoint),
        "step": 32_000,
    }
    plan = {
        "paths": {
            "checkpoint_receipt": str(tmp_path / "checkpoint-receipt.json"),
            "eval": {"0": {
                "out_dir": str(out_dir), "receipt": str(receipt_path),
            }},
        },
        "evaluation": {},
    }
    monkeypatch.setattr(chain, "_read_receipt", lambda *args, **kwargs: checkpoint_receipt)
    monkeypatch.setattr(
        chain, "_validate_seed_result",
        lambda *args, **kwargs: ({"summary": {"avg": 41.0}}, object()),
    )
    monkeypatch.setattr(chain, "_wandb_publish", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        chain.subprocess, "run",
        lambda *args, **kwargs: pytest.fail("complete result must not be rerun"),
    )
    assert chain._stage_eval(plan, "eval_seed0") == 0
    assert json.loads(receipt_path.read_text())["result_sha256"] == (
        chain.sha256_file(result)
    )


def test_wandb_publication_retries_same_stage_id_without_long_wait(
    tmp_path, monkeypatch,
):
    calls = []

    def flaky(*args, **kwargs):
        calls.append((args, kwargs))
        if len(calls) < 3:
            raise ConnectionError("transient")

    monkeypatch.setattr(chain, "_wandb_publish_once", flaky)
    monkeypatch.setattr(chain.time, "sleep", lambda seconds: None)
    chain._wandb_publish(
        {"wandb": {"require_online": True}}, stage="eval-summary",
        path=tmp_path / "result.json", artifact_type="evaluation-results",
        summary={"success_rate": 50.0},
    )
    assert len(calls) == 3
    assert all(call[1]["stage"] == "eval-summary" for call in calls)


def _write_seed_result(root: Path, checkpoint_receipt, seed: int):
    checkpoint = str(Path(checkpoint_receipt["checkpoint"]).resolve())
    protocol = EvalProtocol(
        bench="libero", episodes_per_task=10, n_tasks=10,
        suites=DEFAULT_LIBERO_SUITES, seeds=(seed,), max_steps=512,
    )
    records = [
        EpisodeResult(
            bench=item.bench, suite=item.suite, task_id=item.task_id,
            episode=item.episode, seed=item.seed, env_seed=item.env_seed,
            success=(item.task_id + item.episode + seed) % 3 == 0,
            steps=17, task_name=f"task-{item.task_id}",
            extra={"policy_seed": item.policy_seed},
        )
        for item in iter_work(protocol)
    ]
    blob = {
        "version": 1,
        "bench": "libero",
        "protocol": protocol.to_dict(),
        "meta": {
            "ckpt": checkpoint,
            "bench": "libero",
            "backend": "libero",
            "policy_seed_scheme": "sha256(work-item)-v1",
            "eval_identity": {
                "version": 1,
                "checkpoint": checkpoint,
                "backend": {"requested": "libero", "resolved": "libero"},
                "policy_kw": {
                    "allow_stub": False,
                    "op_stats": True,
                    "embodiment": "libero_franka",
                },
                "policy_source": "checkpoint_factory",
                "policy_seed_scheme": "sha256(work-item)-v1",
            },
            "policy": {
                "policy": "LoomPolicy",
                "is_stub": False,
                "ckpt_global_step": checkpoint_receipt["step"],
                "ckpt": checkpoint,
                "embodiment": "libero_franka",
                "gripper_dwell": 1,
                "decoder_samples": 1,
                "duration_normalize_segments": False,
            },
        },
        "summary": aggregate(records, protocol),
        "episodes": [record.to_dict() for record in records],
    }
    out = root / f"seed{seed}.json"
    out.write_text(json.dumps(blob))
    return out


def test_merge_recomputes_exact_disjoint_1200_episode_summary(tmp_path):
    checkpoint = tmp_path / "ckpt.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_receipt = {
        "format_version": chain.FORMAT_VERSION,
        "kind": "r0_e2e_consolidated_checkpoint_receipt",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": chain.sha256_file(checkpoint),
        "step": 32_000,
    }
    checkpoint_receipt_path = tmp_path / "checkpoint_receipt.json"
    checkpoint_receipt_path.write_text(json.dumps(checkpoint_receipt))
    paths = {"checkpoint_receipt": str(checkpoint_receipt_path), "eval": {}}
    for seed in chain.SEEDS:
        result = _write_seed_result(tmp_path, checkpoint_receipt, seed)
        receipt = {
            "format_version": chain.FORMAT_VERSION,
            "kind": "r0_e2e_single_seed_eval_receipt",
            "seed": seed,
            "result": str(result),
            "result_sha256": chain.sha256_file(result),
        }
        receipt_path = tmp_path / f"seed{seed}_receipt.json"
        receipt_path.write_text(json.dumps(receipt))
        paths["eval"][str(seed)] = {"receipt": str(receipt_path)}
    plan = {"paths": paths, "evaluation": {
        "suites": list(DEFAULT_LIBERO_SUITES),
        "tasks_per_suite": 10,
        "episodes_per_task": 10,
        "max_steps": 512,
    }, "orchestration_source_closure": {"sha256": "source-closure"},
    "config": {
        "raw_sha256": "raw-config", "resolved_config_hash": "resolved-config",
    }, "baseline_comparison": {
        "baseline": chain._authenticate_baseline(chain.CANONICAL_BASELINE_ROOT),
        "thresholds": _formal_cfg()["evaluation_gate"],
    }}

    merged = chain.merge_seed_results(plan)
    assert merged["protocol"]["seeds"] == [0, 1, 2]
    assert merged["protocol"]["total_episodes"] == 1_200
    assert merged["summary"]["n_episodes"] == 1_200
    assert merged["summary"]["n_expected"] == 1_200
    assert merged["summary"]["n_errors"] == 0
    assert merged["summary"]["complete"] is True
    assert merged["baseline_comparison"]["pairing"]["paired_episodes"] == 1_200
    assert merged["baseline_comparison"]["paired_task_bootstrap"][
        "resample_matrix"
    ]["sha256"] == chain.BOOTSTRAP_MATRIX_SHA256
    assert merged["meta"]["merge_provenance"] == {
        "kind": "r0_e2e_plan_stable_merge_v1",
        "orchestration_source_closure_sha256": "source-closure",
        "config_raw_sha256": "raw-config",
        "config_resolved_hash": "resolved-config",
    }
    assert len({
        (row["suite"], row["task_id"], row["episode"], row["seed"])
        for row in merged["episodes"]
    }) == 1_200


def test_merge_rejects_changed_single_seed_result(tmp_path):
    result = tmp_path / "seed0.json"
    result.write_text("{}")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({
        "format_version": chain.FORMAT_VERSION,
        "kind": "r0_e2e_single_seed_eval_receipt",
        "result": str(result),
        "result_sha256": "0" * 64,
    }))
    plan = {"paths": {"eval": {"0": {"receipt": str(receipt)}}}}
    with pytest.raises(chain.ChainError, match="changed before merge"):
        chain.merge_seed_results(plan)


def test_sbatch_wrappers_are_isolated_and_never_submit_descendants():
    wrappers = [
        "r0_e2e_formal_train.sbatch",
        "r0_e2e_formal_control.sbatch",
        "r0_e2e_formal_consolidate.sbatch",
        "r0_e2e_formal_eval_seed.sbatch",
    ]
    for name in wrappers:
        text = (chain.ROOT / "scripts" / name).read_text()
        assert "r0_e2e_formal_chain.py run-stage" in text
        assert "sbatch " not in text
        assert "FORMAL_STAGE" in text and "FORMAL_PLAN" in text
        assert "#SBATCH --requeue" in text
    train = (chain.ROOT / "scripts" / wrappers[0]).read_text()
    assert "--nodes=2" in train and "--gpus-per-node=8" in train
    evaluation = (chain.ROOT / "scripts" / wrappers[-1]).read_text()
    assert "--gpus=8" in evaluation
