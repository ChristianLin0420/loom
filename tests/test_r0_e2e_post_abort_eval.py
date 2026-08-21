from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest

from scripts import r0_e2e_formal_chain as formal
from scripts import r0_e2e_post_abort_eval as diag


ROOT = Path(__file__).resolve().parents[1]


def _actual_source_plan() -> dict:
    return json.loads(diag.SOURCE_FORMAL_PLAN.read_text())


def _fake_trigger() -> dict:
    return {
        "kind": "authenticated_formal_abort_step32000_user_selected_diagnostic_v1",
        "formal_eligible": False,
        "formal_status_remains": "ABORT",
        "formal_reason": "health_gate_failed",
        "authorization_scope": "evaluate_existing_checkpoint_no_retraining",
        "checkpoint_step": 32_000,
        "checkpoint_shards": {
            f"ckpt_000032000_rank{rank}.pt": {
                "bytes": 100 + rank,
                "sha256": f"{rank:064x}",
            }
            for rank in range(16)
        },
        "checkpoint_shard_hashes_included": True,
    }


def _plan_without_source_io(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict:
    source = _actual_source_plan()
    monkeypatch.setattr(diag, "_load_source_plan", lambda: source)
    monkeypatch.setattr(diag, "_collect_trigger", lambda **_: _fake_trigger())
    monkeypatch.setattr(
        formal,
        "_authenticate_baseline",
        lambda _root: source["baseline_comparison"]["baseline"],
    )
    return diag.build_plan(
        control_dir=(tmp_path / "control").resolve(),
        artifact_root=(tmp_path / "artifacts").resolve(),
        group=diag.DIAGNOSTIC_GROUP,
    )


def _minimal_complete_seed_blob(seed: int = 0) -> dict:
    return {
        "bench": "libero",
        "protocol": {
            "bench": "libero", "seeds": [seed],
            "suites": ["libero_spatial", "libero_object", "libero_goal", "libero_long"],
            "n_tasks": 10, "episodes_per_task": 10, "max_steps": 512,
            "notes": "",
        },
        "meta": {"ckpt": "/tmp/ckpt.pt", "env_available": True},
        "summary": {
            "complete": True, "n_episodes": 400, "n_expected": 400,
            "n_errors": 0, "n_hit_step_cap": 0, "mean_episode_len": 1.0,
            "avg": 40.0, "per_suite": {},
        },
        "episodes": [],
    }


def test_exact_source_receipt_files_are_still_the_aborted_v2_lineage():
    for path, expected in diag.SOURCE_SMALL_FILES.values():
        assert path.is_file()
        assert diag.sha256_file(path) == expected
    direct = json.loads(
        (diag.SOURCE_RUN_DIR / "direct_formal_000032000.json").read_text()
    )
    assert direct["status"] == "ABORT"
    assert direct["reason"] == "health_gate_failed"
    assert direct["current_step"] == direct["decision_step"] == 32_000
    assert direct["next_check_step"] is None


def test_exact_source_step_has_sixteen_rank_shards_and_expected_total_bytes():
    stats = diag._source_shards(hash_bytes=False)
    assert list(stats) == [
        f"ckpt_000032000_rank{rank}.pt" for rank in range(16)
    ]
    assert sum(row["bytes"] for row in stats.values()) == 31_969_082_643


def test_stage_graph_is_only_consolidate_three_parallel_seeds_and_merge():
    specs = diag._stage_specs()
    assert [row["name"] for row in specs] == [
        "consolidate", "eval_seed0", "eval_seed1", "eval_seed2", "merge",
    ]
    assert specs[0]["depends_on"] == []
    assert all(specs[index]["depends_on"] == ["consolidate"] for index in (1, 2, 3))
    assert specs[4]["depends_on"] == ["eval_seed0", "eval_seed1", "eval_seed2"]
    assert all("train" not in row["name"] for row in specs)


def test_plan_is_diagnostic_only_and_reuses_exact_protocol(monkeypatch, tmp_path):
    plan = _plan_without_source_io(monkeypatch, tmp_path)
    assert plan["eligibility"] == {
        "formal_eligible": False,
        "promotion_eligible": False,
        "diagnostic_only": True,
        "formal_abort_preserved": True,
        "interpretation": "measurement only; cannot reverse formal ABORT",
    }
    assert plan["method"]["training_updates"] == 0
    assert plan["method"]["optimizer_steps"] == 0
    assert plan["method"]["checkpoint_selection_used_eval"] is False
    assert plan["evaluation"] == _actual_source_plan()["evaluation"]
    assert plan["baseline_comparison"]["baseline"]["successes"] == 447
    assert plan["baseline_comparison"]["baseline"]["episodes"] == 1200
    assert plan["wandb"]["tags"] == [
        "post-abort-diagnostic", "not-formal", "r0", "dual-action",
    ]


def test_output_paths_cannot_overlap_formal_lineage(tmp_path):
    with pytest.raises(diag.DiagnosticError, match="overlaps"):
        diag._require_isolated(
            diag.SOURCE_CONTROL_DIR / "diagnostic", (tmp_path / "artifacts").resolve(),
        )
    with pytest.raises(diag.DiagnosticError, match="overlaps"):
        diag._require_isolated(
            (tmp_path / "control").resolve(), diag.SOURCE_RUN_DIR.parent,
        )


@pytest.mark.parametrize("field", ("paths", "baseline", "wandb"))
def test_plan_rejects_changed_derived_paths_baseline_or_wandb(
    monkeypatch, tmp_path, field,
):
    plan = _plan_without_source_io(monkeypatch, tmp_path)
    changed = copy.deepcopy(plan)
    if field == "paths":
        changed["paths"]["checkpoint"] = str(
            diag.SOURCE_CONTROL_DIR / "forbidden-diagnostic-checkpoint.pt"
        )
        message = "output paths"
    elif field == "baseline":
        changed["baseline_comparison"]["thresholds"][
            "seed0_stretch_successes_gte"
        ] = 163
        message = "baseline"
    else:
        changed["wandb"]["group"] = _actual_source_plan()["wandb"]["group"]
        message = "W&B labels"
    with pytest.raises(diag.DiagnosticError, match=message):
        diag._assert_plan(changed)


def test_plan_rejects_symlink_escape_below_diagnostic_root(monkeypatch, tmp_path):
    plan = _plan_without_source_io(monkeypatch, tmp_path)
    artifact_root = Path(plan["lineage"]["diagnostic_artifact_root"])
    artifact_root.mkdir()
    (artifact_root / "eval").symlink_to(diag.SOURCE_CONTROL_DIR)
    with pytest.raises(diag.DiagnosticError, match="escapes isolated root"):
        diag._assert_plan(plan)


def test_trigger_comparison_rejects_correlated_abort_to_pass(monkeypatch):
    frozen = _fake_trigger()
    changed = dict(frozen)
    changed.update({
        "formal_status_remains": "PASS",
        "formal_reason": "converged",
    })
    monkeypatch.setattr(diag, "_collect_trigger", lambda **_: changed)
    with pytest.raises(diag.DiagnosticError, match="trigger changed"):
        diag._assert_trigger({"trigger": frozen}, rehash_shards=False)


def test_recompute_requires_exact_in_loop_abort_core(monkeypatch):
    in_loop = json.loads(
        (diag.SOURCE_RUN_DIR / "direct_formal_000032000.json").read_text()
    )
    recomputed = dict(in_loop)
    recomputed.update({"status": "PASS", "reason": "converged"})
    recomputed["metrics_source"] = {
        "path": str((diag.SOURCE_RUN_DIR / "metrics.jsonl").resolve()),
        "bytes": (diag.SOURCE_RUN_DIR / "metrics.jsonl").stat().st_size,
        "sha256": diag.SOURCE_METRICS_SHA256,
    }
    monkeypatch.setattr(
        formal, "_read_direct_boundary_receipt",
        lambda _plan, _step: (
            in_loop, diag.SOURCE_RUN_DIR / "direct_formal_000032000.json",
        ),
    )
    monkeypatch.setattr(
        diag.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout=json.dumps(recomputed), stderr="",
        ),
    )
    with pytest.raises(diag.DiagnosticError, match="trigger failed"):
        diag._recompute_abort({})


def test_eval_command_is_exact_resumable_three_seed_protocol(tmp_path):
    source = _actual_source_plan()
    checkpoint = tmp_path / "ckpt.pt"
    for seed in (0, 1, 2):
        command, _ = formal._formal_eval_command(
            source, seed=seed, checkpoint=checkpoint,
            out_dir=tmp_path / f"seed_{seed}",
        )
        joined = " ".join(command)
        assert "--backend libero" in joined
        assert "--require-real --op-stats" in joined
        assert "--n-tasks 10 --episodes-per-task 10" in joined
        assert f"--seeds {seed}" in joined
        assert "--workers 8" in joined
        assert "--gripper-dwell 1 --decoder-samples 1" in joined
        assert "--no-resume" not in command


def _stage_eval_fixture(monkeypatch, tmp_path):
    out_dir = tmp_path / "artifacts/eval/seed_0"
    out_dir.mkdir(parents=True)
    result_path = out_dir / "results.json"
    result_path.write_text("{}\n")
    checkpoint_receipt = tmp_path / "control/checkpoint_receipt.json"
    checkpoint_receipt.parent.mkdir(parents=True)
    checkpoint_receipt.write_text("{}\n")
    checkpoint = tmp_path / "artifacts/checkpoint/ckpt.pt"
    plan = {
        "paths": {
            "checkpoint_receipt": str(checkpoint_receipt),
            "eval": {"0": {
                "out_dir": str(out_dir),
                "receipt": str(tmp_path / "control/eval_seed_0_receipt.json"),
            }},
        },
    }
    blob = _minimal_complete_seed_blob()
    monkeypatch.setattr(diag, "_assert_trigger", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        diag, "_validate_checkpoint_receipt",
        lambda *_args, **_kwargs: {
            "checkpoint": str(checkpoint), "checkpoint_sha256": "a" * 64,
        },
    )
    monkeypatch.setattr(
        diag, "_validate_seed_result", lambda *args, **kwargs: (blob, None),
    )
    monkeypatch.setattr(diag, "_validate_eval_receipt", lambda *args: {})
    monkeypatch.setattr(diag, "_wandb_publish", lambda *args, **kwargs: None)
    monkeypatch.setattr(diag, "_plan_sha", lambda: "b" * 64)
    return plan, blob, out_dir


def test_eval_crash_window_adopts_complete_result_and_mints_missing_table(
    monkeypatch, tmp_path,
):
    plan, blob, out_dir = _stage_eval_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(
        diag.subprocess, "run",
        lambda *args, **kwargs: pytest.fail("complete result must not rerun evaluation"),
    )
    assert diag._stage_eval(plan, "eval_seed0") == 0
    assert (out_dir / "table.md").read_text() == diag._seed_markdown(blob)
    receipt = json.loads(
        Path(plan["paths"]["eval"]["0"]["receipt"]).read_text()
    )
    assert receipt["formal_eligible"] is False
    assert receipt["source_formal_status"] == "ABORT"


def test_eval_crash_window_rejects_corrupt_table_for_complete_result(
    monkeypatch, tmp_path,
):
    plan, _blob, out_dir = _stage_eval_fixture(monkeypatch, tmp_path)
    (out_dir / "table.md").write_text("corrupt\n")
    monkeypatch.setattr(
        diag.subprocess, "run",
        lambda *args, **kwargs: pytest.fail("complete result must not rerun evaluation"),
    )
    with pytest.raises(diag.DiagnosticError, match="table differs"):
        diag._stage_eval(plan, "eval_seed0")
    assert not Path(plan["paths"]["eval"]["0"]["receipt"]).exists()


def test_submit_commands_are_held_then_released_with_exact_dependencies(
    monkeypatch, tmp_path,
):
    plan = _plan_without_source_io(monkeypatch, tmp_path)
    monkeypatch.setattr(diag, "_assert_plan", lambda _plan: None)
    calls: list[list[str]] = []
    ids = iter(("101", "102", "103", "104", "105"))

    def fake_run(command, **kwargs):
        calls.append(list(command))
        if command[0] == "sbatch":
            return subprocess.CompletedProcess(command, 0, stdout=next(ids), stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    result = diag.submit_plan(plan, run=fake_run)
    sbatch = [call for call in calls if call[0] == "sbatch"]
    assert len(sbatch) == 5
    assert not any(arg.startswith("--dependency=") for arg in sbatch[0])
    assert all("--dependency=afterok:101" in call for call in sbatch[1:4])
    assert "--dependency=afterok:102:103:104" in sbatch[4]
    assert calls[-1] == ["scontrol", "release", "101,102,103,104,105"]
    assert result["released"] is True


def test_static_surface_has_no_training_or_formal_receipt_publication():
    source = (ROOT / "scripts/r0_e2e_post_abort_eval.py").read_text()
    assert "def _stage_train" not in source
    assert "r0_e2e_formal_train.sbatch" not in source
    assert "select_first_passing_checkpoint" not in source
    assert '"formal_eligible": False' in source
    assert '"training_updates_performed": 0' in source
    assert '"optimizer_steps_performed": 0' in source
    assert "r0_e2e_post_abort_diagnostic_checkpoint_receipt" in source
    assert "r0_e2e_post_abort_diagnostic_single_seed_eval_receipt" in source
    assert "r0_e2e_post_abort_diagnostic_merged_eval_receipt" in source


def test_markdown_cannot_be_mistaken_for_formal_result(monkeypatch):
    monkeypatch.setattr(
        formal, "_markdown_table",
        lambda _merged: "Scientific evaluation gate: **PASS**.\n",
    )
    text = diag._diagnostic_markdown({
        "diagnostic_baseline_comparison": {"status": "PASS"},
        "summary": {},
    })
    assert text.startswith("# Post-ABORT diagnostic evaluation (not formal)")
    assert "source formal decision remains **ABORT**" in text
    assert "do not establish formal eligibility" in text
    assert "Scientific evaluation gate" not in text
    assert "Counterfactual diagnostic threshold status" in text


def test_merged_replay_recomputes_artifacts_and_rejects_tampered_average(
    monkeypatch, tmp_path,
):
    result_path = tmp_path / "artifacts/eval/merged/results.json"
    table_path = tmp_path / "artifacts/eval/merged/table.md"
    receipt_path = tmp_path / "control/merged_eval_receipt.json"
    checkpoint_receipt = tmp_path / "control/checkpoint_receipt.json"
    eval_receipts = {
        str(seed): tmp_path / f"control/eval_seed_{seed}_receipt.json"
        for seed in formal.SEEDS
    }
    result_path.parent.mkdir(parents=True)
    receipt_path.parent.mkdir(parents=True)
    checkpoint_receipt.write_text("checkpoint\n")
    for seed, path in eval_receipts.items():
        path.write_text(f"seed-{seed}\n")
    comparison = {
        "overall": {"delta_percentage_points": 1.25},
        "paired_task_bootstrap": {
            "ci_low_percentage_points": -0.5,
            "ci_high_percentage_points": 2.0,
        },
        "per_seed_candidate_successes": {"0": 150, "1": 151, "2": 152},
        "per_suite": {}, "status": "FAIL", "passed": False,
        "failed_checks": ["paired_task_bootstrap_ci_low_strict_gt"],
    }
    merged = {
        "summary": {
            "n_episodes": formal.EXPECTED_EPISODES_TOTAL,
            "n_errors": 0, "avg": 37.75, "complete": True,
        },
        "diagnostic_baseline_comparison": comparison,
    }
    plan = {
        "paths": {
            "merged_results": str(result_path), "merged_table": str(table_path),
            "merged_receipt": str(receipt_path),
            "checkpoint_receipt": str(checkpoint_receipt),
            "eval": {
                seed: {"receipt": str(path)} for seed, path in eval_receipts.items()
            },
        },
        "baseline_comparison": {
            "baseline": {"files": {
                str(seed): {"sha256": f"{seed + 1:064x}"}
                for seed in formal.SEEDS
            }},
        },
    }
    result_path.write_text(diag._pretty_json(merged))
    table_path.write_text("diagnostic-table\n")
    monkeypatch.setattr(diag, "merge_seed_results", lambda _plan: merged)
    monkeypatch.setattr(
        diag, "_diagnostic_markdown", lambda _merged: "diagnostic-table\n",
    )
    monkeypatch.setattr(diag, "_plan_sha", lambda: "c" * 64)
    receipt = diag._merged_receipt_payload(
        plan, merged, result_path=result_path, table_path=table_path,
    )
    receipt_path.write_text(json.dumps(receipt))
    assert diag._validate_merged_receipt(plan) == receipt

    receipt["avg"] = 99.0
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(diag.DiagnosticError, match="exact recomputation"):
        diag._validate_merged_receipt(plan)


def test_source_closure_contains_only_new_wrapper_surface_plus_frozen_import():
    closure = diag._source_closure()
    assert set(closure["files"]) == set(diag.DIAGNOSTIC_SOURCE_FILES)
    assert closure["scheme"] == "sha256(path-nul-sha256-nul)-v1"
    assert len(closure["sha256"]) == 64
    assert formal._source_closure()["sha256"] == diag.SOURCE_FORMAL_CLOSURE_SHA256
