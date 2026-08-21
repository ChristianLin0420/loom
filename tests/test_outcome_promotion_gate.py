from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch

from scripts import outcome_promotion_gate as gate
from loom.train.loop import read_config


ROOT = Path(__file__).resolve().parents[1]
BASELINE = (
    ROOT
    / "runs/eval_r0a_deploy_s1_s49666_seeded1200_v2/seed0/results.json"
)


def _outcomes_by_suite(counts):
    outcomes = {}
    by_suite = {suite: [] for suite in gate.DEFAULT_LIBERO_SUITES}
    for item in gate.official_items():
        by_suite[item.suite].append(item.key())
    for suite, keys in by_suite.items():
        for index, key in enumerate(keys):
            outcomes[key] = index < counts[suite]
    return outcomes


def _summary(outcomes):
    suite_counts = {
        suite: sum(value for key, value in outcomes.items() if key[1] == suite)
        for suite in gate.DEFAULT_LIBERO_SUITES
    }
    n_success = sum(outcomes.values())
    return {
        "per_task": {},
        "per_suite": {
            suite: {
                "success_rate": float(suite_counts[suite]),
                "n_tasks": 10,
                "n_episodes": 100,
                "n_errors": 0,
                "n_hit_step_cap": 100 - suite_counts[suite],
                "mean_episode_len": 0.0,
                "per_seed": {"0": float(suite_counts[suite])},
            }
            for suite in gate.DEFAULT_LIBERO_SUITES
        },
        "avg": 100.0 * n_success / gate.EXPECTED_WORK_ITEMS,
        "n_episodes": gate.EXPECTED_WORK_ITEMS,
        "n_expected": gate.EXPECTED_WORK_ITEMS,
        "n_errors": 0,
        "n_hit_step_cap": gate.EXPECTED_WORK_ITEMS - n_success,
        "mean_episode_len": 0.0,
        "complete": True,
    }


def _eval_meta(checkpoint, *, step, config_hash=None, stub=False):
    checkpoint = str(Path(checkpoint).resolve())
    policy = {
        "policy": "LoomPolicy",
        "is_stub": stub,
        "embodiment": "libero_franka",
        "device": "cuda:0",
        "env_fps": 20.0,
        "env_steps_per_segment": 16.0 / 3.0,
        "gripper_dwell": 1,
        "decoder_samples": 1,
        "duration_normalize_segments": False,
        "h_op": 8,
        "fps_canonical": 30,
        "resampler": "loom.data.canonical.to_env_rate",
        "ckpt": checkpoint,
        "ckpt_global_step": step,
        "state_dict": {
            name: {"tensors_loaded": 1, "unexpected": 0}
            for name in gate.DIRECT_POLICY_MODULES
        },
    }
    if config_hash is not None:
        policy["ckpt_config_hash"] = config_hash
    return {
        "ckpt": checkpoint,
        "bench": "libero",
        "backend": "libero",
        "policy_seed_scheme": gate.POLICY_SEED_SCHEME,
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
            "policy_seed_scheme": gate.POLICY_SEED_SCHEME,
        },
        "env_available": True,
        "libero_available": True,
        "started": "2026-08-19T00:00:00",
        "git_sha": "test",
        "slurm_job_id": "1",
        "hostname": "test-host",
        "policy": policy,
    }


def _results(checkpoint, outcomes, *, step, config_hash=None):
    episodes = []
    for item in gate.official_items():
        success = bool(outcomes[item.key()])
        episodes.append({
            "bench": item.bench,
            "suite": item.suite,
            "task_id": item.task_id,
            "episode": item.episode,
            "seed": item.seed,
            "env_seed": item.env_seed,
            "success": success,
            "steps": 10 if success else item.max_steps,
            "hit_step_cap": not success,
            "task_name": f"{item.suite}/task_{item.task_id:02d}",
            "n_replans": 2,
            "wall_s": 1.0,
            "error": None,
            "extra": {"policy_seed": item.policy_seed, "op_m": 128},
        })
    return {
        "version": 1,
        "bench": "libero",
        "protocol": gate.official_protocol().to_dict(),
        "meta": _eval_meta(
            checkpoint, step=step, config_hash=config_hash,
        ),
        "summary": _summary(outcomes),
        "episodes": episodes,
    }


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _collector_source():
    return {
        "scheme": "sha256(path-nul-bytes-nul)-v1",
        "sha256": gate.CANONICAL_COLLECTOR_SOURCE_SHA256,
        "files": list(gate.CANONICAL_COLLECTOR_SOURCE_FILES),
    }


def _collection(split, n_groups):
    return {
        "path": f"/collections/{split}",
        "manifest": f"/collections/{split}/manifest.json",
        "manifest_sha256": hashlib.sha256(f"manifest:{split}".encode()).hexdigest(),
        "identity_digest": hashlib.sha256(f"identity:{split}".encode()).hexdigest(),
        "split": split,
        "n_groups": n_groups,
        "n_trajectories": n_groups * 8,
        "terminal_successes_by_arm": [100] * 8,
        "replans_by_arm": [200] * 8,
        "collector_source": _collector_source(),
    }


def _trainer_source():
    files = {
        rel: gate.sha256_file(ROOT / rel)
        for rel in gate.CANONICAL_TRAINER_SOURCE_FILES
    }
    digest = gate._trainer_source_digest(files)
    assert digest == gate.CANONICAL_TRAINER_SOURCE_SHA256
    return {
        "scheme": gate.CANONICAL_TRAINER_SOURCE_SCHEME,
        "sha256": digest,
        "files": files,
    }


def _proposal_scoring():
    return copy.deepcopy(gate.CANONICAL_PROPOSAL_SCORING)


def _behaviour_authentication():
    reports = []
    replay_error = 0.0
    for _ in range(7):
        reports.append({
            "passed": True,
            "all_atoms": 100,
            "ratio_eligible_atoms": 70,
            "arm0_ratio_eligible_atoms": 0,
            "max_abs_old_logprob_error": replay_error,
            "max_abs_coeff_error": 0.0,
            "max_abs_logratio": replay_error,
            "proposal_replay_batch_size": 1,
            "proposal_scoring": _proposal_scoring(),
            "transfer_chunk_replans": 32,
            "logprob_atol": gate.BEHAVIOUR_IDENTITY_MAX_ABS_LOGRATIO,
            "logprob_rtol": 0.0,
            "coeff_atol": 0.0,
            "coeff_rtol": 0.0,
        })
    exact = {
        "passed": True,
        "splits": 7,
        "all_atoms": 700,
        "max_abs_coeff_error": 0.0,
        "max_abs_old_logprob_error": replay_error,
        "max_abs_logratio": replay_error,
        "max_abs_coeff_error_threshold": gate.BEHAVIOUR_IDENTITY_MAX_COEFF_ERROR,
        "max_abs_logratio_threshold": gate.BEHAVIOUR_IDENTITY_MAX_ABS_LOGRATIO,
    }
    return reports, exact


def _start_checkpoint_identity(parent_proposal):
    return {
        "checked": True,
        "passed": True,
        "global_step": gate.AUTHORITATIVE_BASELINE_STEP,
        "proposal": copy.deepcopy(parent_proposal),
        "optimizer_state_entries": 0,
        "optimizer_reset": {"count": 1, "modules": ["proposal"]},
    }


def _initial_behavior_identity(
    *, trainer_source, parent, exact_behaviour, start_checkpoint,
):
    ranks = []
    for rank in range(8):
        ranks.append({
            "rank": rank,
            "passed": True,
            "max_abs_logratio": 0.0,
            "ratio_min": 1.0,
            "ratio_mean": 1.0,
            "ratio_max": 1.0,
            "clip_fraction": 0.0,
            "ratio_atoms": 14,
            "ratio_sum": 14.0,
            "ratio_square_sum": 14.0,
            "ratio_ess_fraction": 1.0,
            "clipped_atoms": 0,
            "max_abs_logratio_threshold": gate.BEHAVIOUR_IDENTITY_MAX_ABS_LOGRATIO,
            "min_ess_fraction_threshold": gate.INITIAL_RATIO_MIN_ESS_FRACTION,
            "proposal_scoring": _proposal_scoring(),
        })
    return {
        "format_version": 1,
        "kind": "outcome_grpo_initial_behavior_identity",
        "passed": True,
        "world_size": 8,
        "config_hash": gate.CANONICAL_CONFIG_HASH,
        "trainer_source": copy.deepcopy(trainer_source),
        "parent": copy.deepcopy(parent),
        "exact_behaviour_identity": copy.deepcopy(exact_behaviour),
        "start_checkpoint_identity": copy.deepcopy(start_checkpoint),
        "strict_determinism": copy.deepcopy(
            gate.CANONICAL_STRICT_DETERMINISM
        ),
        "proposal_scoring": _proposal_scoring(),
        "ranks": ranks,
    }


def _terminal_evaluation(final_proposal):
    tasks = []
    assignments = [
        ("snapshot", step)
        for step in gate.CANONICAL_CONVERGENCE_SNAPSHOT_STEPS
    ] + [("trust", gate.CANONICAL_CANDIDATE_STEP), ("idle", None)]
    for rank, (kind, step) in enumerate(assignments):
        tasks.append({
            "rank": rank,
            "kind": kind,
            "step": step,
            "elapsed_seconds": 1.0,
            "proposal_scoring": _proposal_scoring(),
            "live_proposal": copy.deepcopy(final_proposal),
        })
    return {
        "parallelism": gate.CANONICAL_TERMINAL_PARALLELISM,
        "world_size": 8,
        "tasks": tasks,
        "live_proposal": copy.deepcopy(final_proposal),
    }


def _terminal_gates(final_proposal):
    metric_checks = {
        "accepted_updates": {
            "value": gate.CANONICAL_ACCEPTED_UPDATES,
            "op": "==", "threshold": gate.CANONICAL_ACCEPTED_UPDATES,
            "pass": True,
        },
        "anchor_sparse_ce_block_median_relative_range": {
            "value": 0.01, "op": "<=", "threshold": 0.02, "pass": True,
        },
        "anchor_sparse_ce_terminal_block_median": {
            "value": 4.0, "op": "<", "threshold": 4.852030263919617,
            "pass": True,
        },
    }
    convergence_checks = {
        "heldout_efficacy": {
            "value": 0.01, "op": ">", "threshold": 0.0, "pass": True,
        },
        "heldout_plateau_all_snapshots": {
            "value": 4, "op": "==", "threshold": 4, "pass": True,
        },
        "final_approx_kl": {
            "value": 0.001, "op": "<=", "threshold": 0.01, "pass": True,
        },
        **copy.deepcopy(metric_checks),
    }
    snapshots = {}
    for step in gate.CANONICAL_CONVERGENCE_SNAPSHOT_STEPS:
        checkpoint = {"global_step": step}
        if step != gate.AUTHORITATIVE_BASELINE_STEP:
            checkpoint.update({
                "kind": "outcome_training_checkpoint",
                "config_hash": gate.CANONICAL_CONFIG_HASH,
            })
        if step == gate.CANONICAL_CANDIDATE_STEP:
            checkpoint["proposal"] = copy.deepcopy(final_proposal)
        snapshots[str(step)] = {
            "n_groups": 400,
            "n_tasks": 40,
            "informative_groups": 240,
            "proposal_scoring": _proposal_scoring(),
            "groups": [{"group_id": f"g{index:03d}"} for index in range(400)],
            "checkpoint": checkpoint,
        }
    convergence = {
        "status": "PASS", "passed": True,
        "checks": convergence_checks,
        "efficacy": {"pass": True},
        "plateau": [{"pass": True} for _ in range(4)],
        "training_metrics": {"passed": True, "checks": metric_checks},
        "snapshots": snapshots,
    }
    trust_checks = {
        "clip_fraction": {
            "value": 0.1, "op": "<=", "threshold": 0.2, "pass": True,
        },
        "ess_fraction": {
            "value": 0.9, "op": ">=", "threshold": 0.8, "pass": True,
        },
        "coeff_drift_p95": {
            "value": 0.02, "op": "<=", "threshold": 0.05, "pass": True,
        },
        "live_ops": {
            "value": 32, "op": ">=", "threshold": 16, "pass": True,
        },
        "nonfinite": {
            "value": 0, "op": "==", "threshold": 0, "pass": True,
        },
        "unexpected_gradients": {
            "value": 0, "op": "==", "threshold": 0, "pass": True,
        },
        "expert_topk_overlap_change": {
            "initial": 0.8, "final": 0.8, "change": 0.0,
            "threshold": -0.05, "pass": True,
        },
    }
    trust = {
        "passed": True,
        "proposal_scoring": _proposal_scoring(),
        "checks": trust_checks,
        "counts": {
            "ratio_atoms": 100, "clipped_atoms": 1,
            "arm0_drift_atoms": 100, "arm0_usage_atoms": 100,
            "training_nonfinite": 0, "final_nonfinite": 0,
        },
    }
    return convergence, trust


def _write_terminal_report(path, payload):
    checkpoint_sha = gate.sha256_file(path)
    training = payload["outcome_grpo"]
    convergence = training["convergence_gate"]
    trust = training["trust_gate"]
    combined = {
        **{f"convergence/{name}": row for name, row in convergence["checks"].items()},
        **{f"trust/{name}": row for name, row in trust["checks"].items()},
    }
    report = {
        "path": str(path.resolve()),
        "sha256": checkpoint_sha,
        "size": path.stat().st_size,
        "global_step": gate.CANONICAL_CANDIDATE_STEP,
        "config_hash": gate.CANONICAL_CONFIG_HASH,
        "optimizer_steps": gate.CANONICAL_ACCEPTED_UPDATES,
        "frozen_model": training["frozen_model"],
        "proposal": training["final_proposal"],
        "verification": {
            "weights_only": True,
            "consolidated_step": gate.CANONICAL_CANDIDATE_STEP,
            "load_policy": {
                "is_stub": False,
                "global_step": gate.CANONICAL_CANDIDATE_STEP,
                "config_hash": gate.CANONICAL_CONFIG_HASH,
                "state_dict": {
                    name: {"tensors_loaded": 1, "unexpected": 0}
                    for name in gate.DIRECT_POLICY_MODULES
                },
            },
        },
        "status": "PASS", "passed": True, "candidate_emitted": True,
        "checks": combined,
        "convergence_gate": convergence,
        "trust_gate": trust,
    }
    _write_json(path.parent / "terminal_report.json", report)
    return report


def _write_checkpoint(path):
    cfg = read_config(str(ROOT / "configs/r0a_outcome_grpo.yaml"))
    config_hash = gate._experiment_config_hash(cfg)
    assert config_hash == gate.CANONICAL_CONFIG_HASH
    parent = {
        "path": str((ROOT / "runs/r0a_deploy_s1_eval/ckpt_000049666.pt").resolve()),
        "sha256": gate.CANONICAL_PARENT_CHECKPOINT_SHA256,
        "global_step": gate.AUTHORITATIVE_BASELINE_STEP,
        "config_hash": gate.CANONICAL_PARENT_CONFIG_HASH,
    }
    collections = [_collection(f"train{index}", 200) for index in range(6)]
    validation = _collection("validation", 400)
    model = {
        "estimator.weight": torch.ones(1),
        "proposal.weight": torch.ones(1),
        "decoder.weight": torch.ones(1),
    }
    frozen_model = gate._model_state_digest(model, proposal=False)
    initial_proposal = {
        "sha256": "2" * 64, "n_tensors": 1, "n_bytes": 4,
    }
    final_proposal = gate._model_state_digest(model, proposal=True)
    convergence, trust = _terminal_gates(final_proposal)
    trainer_source = _trainer_source()
    behaviour, exact_behaviour = _behaviour_authentication()
    start_checkpoint = _start_checkpoint_identity(initial_proposal)
    initial_behavior = _initial_behavior_identity(
        trainer_source=trainer_source,
        parent=parent,
        exact_behaviour=exact_behaviour,
        start_checkpoint=start_checkpoint,
    )
    payload = {
        "model": model,
        "global_step": gate.CANONICAL_CANDIDATE_STEP,
        "config_hash": config_hash,
        "resolved_config": cfg,
        "world_size": 8,
        "stop_reason": "terminal_outcome_grpo",
        "optimizer": {
            "kind": "proposal_only_adamw",
            "parameter_names": ["weight"],
            "state_dict": {},
            "state_reset_at_entry": True,
        },
        "consolidated": {
            "tool": "loom.train.outcome_grpo",
            "step": gate.CANONICAL_CANDIDATE_STEP,
            "parent_checkpoint": parent,
            "derivation": "proposal-only terminal outcome GRPO",
            "mutated_model_prefixes": ["proposal."],
        },
        "outcome_grpo": {
            "format_version": 1,
            "kind": gate.CANONICAL_TRAINER_KIND,
            "trainer_source": trainer_source,
            "strict_determinism": copy.deepcopy(
                gate.CANONICAL_STRICT_DETERMINISM
            ),
            "world_size": 8,
            "collections": collections,
            "validation": validation,
            "collection": validation,
            "behaviour_authentication": behaviour,
            "exact_behaviour_identity": exact_behaviour,
            "start_checkpoint_identity": start_checkpoint,
            "initial_behavior_ratio_identity": initial_behavior,
            "parent": parent,
            "parent_config_hash": gate.CANONICAL_PARENT_CONFIG_HASH,
            "parent_global_step": gate.AUTHORITATIVE_BASELINE_STEP,
            "descendant_config_hash": config_hash,
            "descendant_global_step": gate.CANONICAL_CANDIDATE_STEP,
            "optimizer_steps": gate.CANONICAL_ACCEPTED_UPDATES,
            "mutated_model_prefixes": ["proposal."],
            "frozen_model": frozen_model,
            "initial_proposal": initial_proposal,
            "final_proposal": final_proposal,
            "optimizer_reset": {
                "count": 1, "modules": ["proposal"],
                "source_global_step": gate.AUTHORITATIVE_BASELINE_STEP,
            },
            "recipe": {
                "algorithm": "stored_order_pl_clipped_grpo",
                "reward": "terminal_LIBERO_success_only",
                "sampled_arms": list(range(1, 8)),
                "folds": 6,
                "updates_per_fold": 800,
                "forbidden": ["Phi", "bank", "shaped_reward"],
            },
            "training": {
                "optimizer_steps": gate.CANONICAL_ACCEPTED_UPDATES,
                "initial_proposal": copy.deepcopy(initial_proposal),
                "unexpected_gradients": [], "nonfinite": 0,
            },
            "terminal_evaluation": _terminal_evaluation(final_proposal),
            "convergence_gate": convergence,
            "trust_gate": trust,
        },
    }
    torch.save(payload, path)
    _write_terminal_report(path, payload)
    return payload


def _validated_fixture(tmp_path):
    baseline_counts = dict(gate.EXPECTED_BASELINE_BY_SUITE)
    candidate_counts = {
        "libero_spatial": 42,
        "libero_object": 35,
        "libero_goal": 50,
        "libero_long": 37,
    }
    checkpoint = tmp_path / f"candidate_{gate.CANONICAL_CANDIDATE_STEP:09d}.pt"
    payload = _write_checkpoint(checkpoint)
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_blob = _results(
        tmp_path / "baseline.pt", _outcomes_by_suite(baseline_counts),
        step=gate.AUTHORITATIVE_BASELINE_STEP,
    )
    candidate_blob = _results(
        checkpoint, _outcomes_by_suite(candidate_counts),
        step=payload["global_step"], config_hash=payload["config_hash"],
    )
    baseline_sha = _write_json(baseline_path, baseline_blob)
    _write_json(candidate_path, candidate_blob)
    baseline = gate.validate_results(
        baseline_path, label="baseline",
        expected_results_sha256=baseline_sha,
        expected_step=gate.AUTHORITATIVE_BASELINE_STEP,
    )
    candidate = gate.validate_results(
        candidate_path, label="candidate",
        expected_step=payload["global_step"],
        expected_config_hash=payload["config_hash"],
    )
    return {
        "baseline": baseline,
        "candidate": candidate,
        "baseline_path": baseline_path,
        "candidate_path": candidate_path,
        "checkpoint_path": checkpoint,
        "checkpoint_payload": payload,
        "baseline_blob": baseline_blob,
        "candidate_blob": candidate_blob,
        "baseline_sha": baseline_sha,
    }


def test_authoritative_baseline_artifact_is_exact_and_paired_ready():
    result = gate.validate_baseline(BASELINE)
    assert result.sha256 == gate.AUTHORITATIVE_BASELINE_RESULTS_SHA256
    assert result.outcomes_sha256 == gate.AUTHORITATIVE_BASELINE_OUTCOME_SHA256
    assert result.n_success == 149
    assert result.suite_success == gate.EXPECTED_BASELINE_BY_SUITE
    assert len(result.outcomes) == 400


def test_synthetic_pass_has_exact_counts_pairing_and_checkpoint_binding(tmp_path):
    fixture = _validated_fixture(tmp_path)
    checkpoint = gate.checkpoint_provenance(fixture["checkpoint_path"])
    binding = gate.bind_candidate_checkpoint(fixture["candidate"], checkpoint)
    verdict = gate.promotion_verdict(fixture["baseline"], fixture["candidate"])
    assert binding["passed"] is True
    assert binding["metadata_checkpoint_sha256"] == checkpoint["sha256"]
    assert verdict["status"] == "PASS"
    assert verdict["candidate"]["n_success"] == 164
    assert verdict["paired"]["new_only"] == 15
    assert verdict["paired"]["old_only"] == 0


@pytest.mark.parametrize(
    ("suite", "counts", "failed_check"),
    [
        (
            None,
            {"libero_spatial": 41, "libero_object": 34,
             "libero_goal": 49, "libero_long": 39},
            "candidate_total",
        ),
        (
            "libero_spatial",
            {"libero_spatial": 34, "libero_object": 40,
             "libero_goal": 50, "libero_long": 40},
            "suite_libero_spatial",
        ),
        (
            "libero_object",
            {"libero_spatial": 45, "libero_object": 26,
             "libero_goal": 50, "libero_long": 43},
            "suite_libero_object",
        ),
        (
            "libero_goal",
            {"libero_spatial": 45, "libero_object": 40,
             "libero_goal": 42, "libero_long": 37},
            "suite_libero_goal",
        ),
        (
            "libero_long",
            {"libero_spatial": 47, "libero_object": 45,
             "libero_goal": 49, "libero_long": 23},
            "suite_libero_long",
        ),
    ],
)
def test_thresholds_fail_independently(tmp_path, suite, counts, failed_check):
    fixture = _validated_fixture(tmp_path)
    blob = _results(
        fixture["checkpoint_path"], _outcomes_by_suite(counts),
        step=fixture["checkpoint_payload"]["global_step"],
        config_hash=fixture["checkpoint_payload"]["config_hash"],
    )
    path = tmp_path / "threshold.json"
    _write_json(path, blob)
    candidate = gate.validate_results(
        path, label="candidate",
        expected_step=fixture["checkpoint_payload"]["global_step"],
        expected_config_hash=fixture["checkpoint_payload"]["config_hash"],
    )
    verdict = gate.promotion_verdict(fixture["baseline"], candidate)
    assert verdict["status"] == "FAIL"
    assert failed_check in verdict["failures"]


def test_paired_direction_is_an_explicit_strict_gate(tmp_path):
    fixture = _validated_fixture(tmp_path)
    # A candidate identical to baseline has new_only == old_only == 0.  Lower
    # the other thresholds only to isolate the strict paired condition.
    zero = {suite: 0 for suite in gate.DEFAULT_LIBERO_SUITES}
    verdict = gate.promotion_verdict(
        fixture["baseline"], fixture["baseline"],
        min_total=0, min_by_suite=zero,
    )
    assert verdict["status"] == "FAIL"
    assert verdict["failures"] == ["paired_new_only_gt_old_only"]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "env_seed", "policy_seed", "error"])
def test_work_item_completeness_seeds_and_errors_fail_closed(tmp_path, mutation):
    fixture = _validated_fixture(tmp_path)
    blob = copy.deepcopy(fixture["candidate_blob"])
    if mutation == "missing":
        blob["episodes"].pop()
    elif mutation == "duplicate":
        blob["episodes"][-1] = copy.deepcopy(blob["episodes"][0])
    elif mutation == "env_seed":
        blob["episodes"][0]["env_seed"] += 1
    elif mutation == "policy_seed":
        blob["episodes"][0]["extra"]["policy_seed"] += 1
    else:
        blob["episodes"][0]["error"] = "traceback"
    path = tmp_path / f"bad-{mutation}.json"
    _write_json(path, blob)
    with pytest.raises(gate.PromotionGateError):
        gate.validate_results(
            path, label="candidate",
            expected_step=fixture["checkpoint_payload"]["global_step"],
            expected_config_hash=fixture["checkpoint_payload"]["config_hash"],
        )


def test_summary_is_recomputed_and_authenticated(tmp_path):
    fixture = _validated_fixture(tmp_path)
    blob = copy.deepcopy(fixture["candidate_blob"])
    blob["summary"]["n_errors"] = 1
    blob["summary"]["per_suite"]["libero_goal"]["success_rate"] += 1
    path = tmp_path / "bad-summary.json"
    _write_json(path, blob)
    with pytest.raises(gate.PromotionGateError, match="summary reports errors"):
        gate.validate_results(path, label="candidate")


def test_nonstub_real_checkpoint_metadata_is_mandatory(tmp_path):
    fixture = _validated_fixture(tmp_path)
    blob = copy.deepcopy(fixture["candidate_blob"])
    blob["meta"]["policy"]["is_stub"] = True
    path = tmp_path / "stub.json"
    _write_json(path, blob)
    with pytest.raises(gate.PromotionGateError, match="stub"):
        gate.validate_results(path, label="candidate")


@pytest.mark.parametrize(
    "field", ["step", "config", "checkpoint_sha", "explicit_checkpoint_sha"],
)
def test_checkpoint_step_config_and_content_sha_bind_exactly(tmp_path, field):
    fixture = _validated_fixture(tmp_path)
    checkpoint = gate.checkpoint_provenance(fixture["checkpoint_path"])
    blob = copy.deepcopy(fixture["candidate_blob"])
    if field == "step":
        blob["meta"]["policy"]["ckpt_global_step"] += 1
    elif field == "config":
        blob["meta"]["policy"]["ckpt_config_hash"] = "0" * 16
    elif field == "checkpoint_sha":
        other = tmp_path / "other.pt"
        other.write_bytes(fixture["checkpoint_path"].read_bytes() + b"changed")
        for container, name in (
            (blob["meta"], "ckpt"),
            (blob["meta"]["policy"], "ckpt"),
            (blob["meta"]["eval_identity"], "checkpoint"),
        ):
            container[name] = str(other)
    else:
        blob["meta"]["policy"]["ckpt_sha256"] = "0" * 64
    path = tmp_path / f"mismatch-{field}.json"
    _write_json(path, blob)
    if field in {"step", "config"}:
        with pytest.raises(gate.PromotionGateError):
            gate.validate_results(
                path, label="candidate", expected_step=checkpoint["global_step"],
                expected_config_hash=checkpoint["config_hash"],
            )
    else:
        candidate = gate.validate_results(
            path, label="candidate", expected_step=checkpoint["global_step"],
            expected_config_hash=checkpoint["config_hash"],
        )
        with pytest.raises(gate.PromotionGateError, match="SHA-256|does not match"):
            gate.bind_candidate_checkpoint(candidate, checkpoint)


@pytest.mark.parametrize(
    "mutation", ["config", "recipe", "provenance", "module", "consolidated"],
)
def test_candidate_checkpoint_authentication_fails_closed(tmp_path, mutation):
    path = tmp_path / f"candidate_{gate.CANONICAL_CANDIDATE_STEP:09d}.pt"
    payload = _write_checkpoint(path)
    if mutation == "config":
        payload["resolved_config"]["run"]["name"] = "tampered"
    elif mutation == "recipe":
        payload["resolved_config"].pop("outcome_grpo")
        payload["config_hash"] = gate._experiment_config_hash(payload["resolved_config"])
        payload["outcome_grpo"]["descendant_config_hash"] = payload["config_hash"]
    elif mutation == "provenance":
        payload.pop("outcome_grpo")
    elif mutation == "module":
        payload["model"].pop("decoder.weight")
    else:
        payload.pop("consolidated")
    torch.save(payload, path)
    with pytest.raises(gate.PromotionGateError):
        gate.checkpoint_provenance(path)


@pytest.mark.parametrize(
    "mutation",
    [
        "round0_kind",
        "wrong_step",
        "wrong_updates",
        "wrong_parent",
        "wrong_world_size",
        "wrong_mutated_prefix",
        "wrong_consolidator",
        "source_declared_digest",
        "source_content_digest",
        "source_file_closure",
        "missing_train_fold",
        "holdout_validation",
        "convergence_status",
        "convergence_pass",
        "convergence_check",
        "trust_pass",
        "trust_check",
    ],
)
def test_noncanonical_or_unconverged_training_lineages_are_rejected(
    tmp_path, mutation,
):
    path = tmp_path / f"candidate_{gate.CANONICAL_CANDIDATE_STEP:09d}.pt"
    payload = _write_checkpoint(path)
    training = payload["outcome_grpo"]
    if mutation == "round0_kind":
        training["kind"] = "loom_outcome_grpo_round0_proposal_descendant"
    elif mutation == "wrong_step":
        payload["global_step"] -= 1
        training["descendant_global_step"] -= 1
    elif mutation == "wrong_updates":
        training["optimizer_steps"] = 400
        training["training"]["optimizer_steps"] = 400
    elif mutation == "wrong_parent":
        training["parent"]["sha256"] = "0" * 64
    elif mutation == "wrong_world_size":
        training["world_size"] = 1
    elif mutation == "wrong_mutated_prefix":
        training["mutated_model_prefixes"] = ["proposal.", "decoder."]
    elif mutation == "wrong_consolidator":
        payload["consolidated"]["tool"] = "loom.train.outcome_grpo_round0"
    elif mutation == "source_declared_digest":
        training["trainer_source"]["sha256"] = "0" * 64
    elif mutation == "source_content_digest":
        files = training["trainer_source"]["files"]
        files["contracts.py"] = "0" * 64
        training["trainer_source"]["sha256"] = gate._trainer_source_digest(files)
    elif mutation == "source_file_closure":
        files = training["trainer_source"]["files"]
        files.pop("stubs.py")
        training["trainer_source"]["sha256"] = gate._trainer_source_digest(files)
    elif mutation == "missing_train_fold":
        training["collections"] = training["collections"][:-1]
    elif mutation == "holdout_validation":
        training["validation"]["split"] = "holdout"
        training["collection"] = training["validation"]
    elif mutation == "convergence_status":
        training["convergence_gate"]["status"] = "FAIL"
    elif mutation == "convergence_pass":
        training["convergence_gate"]["passed"] = False
    elif mutation == "convergence_check":
        training["convergence_gate"]["checks"]["heldout_efficacy"]["pass"] = False
    elif mutation == "trust_pass":
        training["trust_gate"]["passed"] = False
    else:
        training["trust_gate"]["checks"]["ess_fraction"]["pass"] = False
    torch.save(payload, path)
    _write_terminal_report(path, payload)
    with pytest.raises(gate.PromotionGateError):
        gate.checkpoint_provenance(path)


@pytest.mark.parametrize(
    "mutation",
    [
        "behaviour_coeff_error",
        "behaviour_logprob_error",
        "behaviour_logratio",
        "behaviour_geometry",
        "exact_behaviour_pass",
        "exact_behaviour_aggregate",
        "start_not_checked",
        "start_optimizer_nonempty",
        "start_proposal_digest",
        "start_initial_binding",
        "training_initial_binding",
        "initial_rank_ratio",
        "initial_rank_clip",
        "initial_rank_sums",
        "initial_rank_geometry",
        "initial_source_binding",
        "initial_exact_binding",
        "initial_strict_determinism",
        "training_strict_determinism",
        "terminal_parallelism",
        "terminal_world",
        "terminal_assignment",
        "terminal_geometry",
        "snapshot_scoring_geometry",
        "trust_scoring_geometry",
        "terminal_rank_proposal",
        "terminal_final_binding",
        "terminal_snapshot_binding",
        "final_model_binding",
    ],
)
def test_exact_behavior_start_ratio_and_terminal_provenance_fail_closed(
    tmp_path, mutation,
):
    path = tmp_path / f"candidate_{gate.CANONICAL_CANDIDATE_STEP:09d}.pt"
    payload = _write_checkpoint(path)
    training = payload["outcome_grpo"]
    if mutation == "behaviour_coeff_error":
        training["behaviour_authentication"][0]["max_abs_coeff_error"] = 1e-8
    elif mutation == "behaviour_logprob_error":
        training["behaviour_authentication"][0]["max_abs_old_logprob_error"] = 1e-3
    elif mutation == "behaviour_logratio":
        training["behaviour_authentication"][0]["max_abs_logratio"] = 1e-3
    elif mutation == "behaviour_geometry":
        training["behaviour_authentication"][0]["proposal_scoring"]["batch_size"] = 2
    elif mutation == "exact_behaviour_pass":
        training["exact_behaviour_identity"]["passed"] = False
    elif mutation == "exact_behaviour_aggregate":
        training["exact_behaviour_identity"]["all_atoms"] -= 1
    elif mutation == "start_not_checked":
        training["start_checkpoint_identity"]["checked"] = False
    elif mutation == "start_optimizer_nonempty":
        training["start_checkpoint_identity"]["optimizer_state_entries"] = 1
    elif mutation == "start_proposal_digest":
        training["start_checkpoint_identity"]["proposal"]["sha256"] = "z" * 64
    elif mutation == "start_initial_binding":
        changed = copy.deepcopy(training["start_checkpoint_identity"])
        changed["proposal"]["sha256"] = "4" * 64
        training["start_checkpoint_identity"] = changed
        training["initial_behavior_ratio_identity"][
            "start_checkpoint_identity"
        ] = copy.deepcopy(changed)
    elif mutation == "training_initial_binding":
        training["training"]["initial_proposal"]["sha256"] = "4" * 64
    elif mutation == "initial_rank_ratio":
        training["initial_behavior_ratio_identity"]["ranks"][0]["ratio_min"] = 0.9
    elif mutation == "initial_rank_clip":
        training["initial_behavior_ratio_identity"]["ranks"][0]["clip_fraction"] = 0.1
    elif mutation == "initial_rank_sums":
        training["initial_behavior_ratio_identity"]["ranks"][0]["ratio_sum"] = 6.0
    elif mutation == "initial_rank_geometry":
        training["initial_behavior_ratio_identity"]["ranks"][0][
            "proposal_scoring"
        ]["autocast"] = True
    elif mutation == "initial_source_binding":
        training["initial_behavior_ratio_identity"]["trainer_source"][
            "sha256"
        ] = "0" * 64
    elif mutation == "initial_exact_binding":
        training["initial_behavior_ratio_identity"]["exact_behaviour_identity"][
            "all_atoms"
        ] -= 1
    elif mutation == "initial_strict_determinism":
        training["initial_behavior_ratio_identity"]["strict_determinism"][
            "warn_only"
        ] = True
    elif mutation == "training_strict_determinism":
        training["strict_determinism"]["warn_only"] = True
    elif mutation == "terminal_parallelism":
        training["terminal_evaluation"]["parallelism"] = "serial_rank0"
    elif mutation == "terminal_world":
        training["terminal_evaluation"]["world_size"] = 7
    elif mutation == "terminal_assignment":
        training["terminal_evaluation"]["tasks"][6]["step"] = None
    elif mutation == "terminal_geometry":
        training["terminal_evaluation"]["tasks"][0]["proposal_scoring"][
            "cuda_matmul_tf32"
        ] = True
    elif mutation == "snapshot_scoring_geometry":
        training["convergence_gate"]["snapshots"][
            str(gate.AUTHORITATIVE_BASELINE_STEP)
        ]["proposal_scoring"]["dtype"] = "bfloat16"
    elif mutation == "trust_scoring_geometry":
        training["trust_gate"]["proposal_scoring"]["module_mode"] = "train"
    elif mutation == "terminal_rank_proposal":
        training["terminal_evaluation"]["tasks"][0]["live_proposal"][
            "sha256"
        ] = "4" * 64
    elif mutation == "terminal_final_binding":
        training["terminal_evaluation"]["live_proposal"]["sha256"] = "4" * 64
    elif mutation == "terminal_snapshot_binding":
        training["convergence_gate"]["snapshots"][
            str(gate.CANONICAL_CANDIDATE_STEP)
        ]["checkpoint"]["proposal"]["sha256"] = "4" * 64
    else:
        training["final_proposal"]["sha256"] = "4" * 64
    torch.save(payload, path)
    _write_terminal_report(path, payload)
    with pytest.raises(gate.PromotionGateError):
        gate.checkpoint_provenance(path)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "status",
        "candidate_emitted",
        "checkpoint_sha",
        "checkpoint_path",
        "convergence_copy",
        "trust_copy",
        "stub_load",
        "missing_module",
    ],
)
def test_terminal_candidate_emission_report_is_mandatory_and_bound(
    tmp_path, mutation,
):
    path = tmp_path / f"candidate_{gate.CANONICAL_CANDIDATE_STEP:09d}.pt"
    _write_checkpoint(path)
    report_path = tmp_path / "terminal_report.json"
    if mutation == "missing":
        report_path.unlink()
    else:
        report = json.loads(report_path.read_text())
        if mutation == "status":
            report["status"] = "FAIL"
            report["passed"] = False
        elif mutation == "candidate_emitted":
            report["candidate_emitted"] = False
        elif mutation == "checkpoint_sha":
            report["sha256"] = "0" * 64
        elif mutation == "checkpoint_path":
            report["path"] = str(tmp_path / "round0_candidate.pt")
        elif mutation == "convergence_copy":
            report["convergence_gate"]["checks"]["heldout_efficacy"]["value"] = 9.0
        elif mutation == "trust_copy":
            report["trust_gate"]["checks"]["ess_fraction"]["value"] = 9.0
        elif mutation == "stub_load":
            report["verification"]["load_policy"]["is_stub"] = True
        else:
            report["verification"]["load_policy"]["state_dict"].pop("decoder")
        _write_json(report_path, report)
    with pytest.raises(gate.PromotionGateError):
        gate.checkpoint_provenance(path)


def test_copied_or_renamed_candidate_is_not_a_canonical_emission(tmp_path):
    canonical = tmp_path / f"candidate_{gate.CANONICAL_CANDIDATE_STEP:09d}.pt"
    _write_checkpoint(canonical)
    copied = tmp_path / "holdout_candidate.pt"
    copied.write_bytes(canonical.read_bytes())
    with pytest.raises(gate.PromotionGateError, match="name is not"):
        gate.checkpoint_provenance(copied)


def test_baseline_outcome_digest_rejects_same_count_swaps(tmp_path):
    fixture = _validated_fixture(tmp_path)
    blob = copy.deepcopy(fixture["baseline_blob"])
    suite_rows = [row for row in blob["episodes"] if row["suite"] == "libero_spatial"]
    succeeded = next(row for row in suite_rows if row["success"])
    failed = next(row for row in suite_rows if not row["success"])
    succeeded["success"], failed["success"] = False, True
    succeeded["steps"], failed["steps"] = 512, 10
    succeeded["hit_step_cap"], failed["hit_step_cap"] = True, False
    path = tmp_path / "swapped-baseline.json"
    _write_json(path, blob)
    with pytest.raises(gate.PromotionGateError, match="outcomes are not authoritative"):
        gate.validate_results(
            path, label="baseline",
            expected_outcomes_sha256=fixture["baseline"].outcomes_sha256,
        )


def test_execute_gate_persists_complete_pass_provenance(tmp_path, monkeypatch):
    fixture = _validated_fixture(tmp_path)
    monkeypatch.setattr(
        gate, "AUTHORITATIVE_BASELINE_RESULTS_SHA256", fixture["baseline_sha"],
    )
    monkeypatch.setattr(
        gate, "AUTHORITATIVE_BASELINE_OUTCOME_SHA256",
        fixture["baseline"].outcomes_sha256,
    )
    out = tmp_path / "promotion.json"
    args = gate.argparse.Namespace(
        baseline=str(fixture["baseline_path"]),
        candidate=str(fixture["candidate_path"]),
        checkpoint=str(fixture["checkpoint_path"]),
        out=str(out),
    )
    report = gate.execute_gate(args)
    assert report["status"] == "PASS"
    assert report["inputs"]["candidate_results"]["n_errors"] == 0
    assert report["inputs"]["checkpoint_binding"]["passed"] is True
    assert report["source_provenance"]["sha256"]
    digest = gate.atomic_publish_json(out, report)
    assert digest == hashlib.sha256(out.read_bytes()).hexdigest()
    stored = json.loads(out.read_text())
    assert stored["results"]["candidate"]["n_success"] == 164


def test_atomic_publish_is_exclusive_under_race(tmp_path):
    target = tmp_path / "gate.json"
    barrier = threading.Barrier(2)

    def publish(worker):
        barrier.wait()
        try:
            digest = gate.atomic_publish_json(target, {"worker": worker})
            return "ok", worker, digest
        except gate.PromotionGateError as exc:
            return "blocked", worker, str(exc)

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(publish, (1, 2)))
    assert [row[0] for row in outcomes].count("ok") == 1
    assert [row[0] for row in outcomes].count("blocked") == 1
    stored = json.loads(target.read_text())
    assert stored["worker"] in (1, 2)
    assert not list(tmp_path.glob(".*.tmp"))
    with pytest.raises(gate.PromotionGateError, match="refusing to overwrite"):
        gate.atomic_publish_json(target, {"worker": 3})
    assert json.loads(target.read_text()) == stored


def test_source_digest_fails_on_changed_or_missing_transitive_source(tmp_path):
    (tmp_path / "a.py").write_text("a = 1\n")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested/b.py").write_text("b = 2\n")
    files = ("a.py", "nested/b.py")
    source = gate.source_provenance(tmp_path, files)
    assert source["sha256"] == gate._source_digest_from_entries(source["files"])
    (tmp_path / "nested/b.py").write_text("b = 3\n")
    with pytest.raises(gate.PromotionGateError, match="changed during"):
        gate.assert_source_unchanged(source, root=tmp_path)
    (tmp_path / "nested/b.py").unlink()
    with pytest.raises(gate.PromotionGateError, match="missing"):
        gate.source_provenance(tmp_path, files)


def test_main_persists_error_and_refuses_to_replace_it(tmp_path):
    out = tmp_path / "error.json"
    argv = [
        "--baseline", str(tmp_path / "missing-baseline.json"),
        "--candidate", str(tmp_path / "missing-candidate.json"),
        "--checkpoint", str(tmp_path / "missing.pt"),
        "--out", str(out),
    ]
    assert gate.main(argv) == 2
    stored = json.loads(out.read_text())
    assert stored["status"] == "ERROR" and stored["passed"] is False
    assert gate.main(argv) == 3
    assert json.loads(out.read_text()) == stored


def test_direct_script_help_works_from_repository_root():
    result = subprocess.run(
        [sys.executable, "scripts/outcome_promotion_gate.py", "--help"],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--baseline" in result.stdout
    assert "--candidate" in result.stdout
    assert "--checkpoint" in result.stdout
    assert "--out" in result.stdout
