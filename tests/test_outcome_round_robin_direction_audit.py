"""CPU contracts for the no-update v2 round-robin direction audit."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from loom.eval import outcome_recovery as recovery
from loom.train import outcome_grpo as v1
from loom.train import outcome_grpo_v2 as v2
from loom.train import schedule as optim_schedule
from scripts import outcome_round_robin_direction_audit as audit


LOCAL_VALIDATION_MANIFEST = Path(
    "runs/outcome_recovery_s49666_validation/manifest.json"
)


class TinyProposal(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[0.5, -1.0], [1.5, 0.25]]))
        self.bias = nn.Parameter(torch.tensor([0.2, -0.3]))


def _cfg() -> dict:
    return {
        "optim": {
            "lr": 3e-4,
            "warmup": 2000,
            "min_lr_ratio": 0.05,
            "grad_clip": 1.0,
            "betas": [0.9, 0.95],
            "weight_decay": 0.05,
            "eps": 1e-8,
            "lr_scales": {"proposal": 0.0125},
        },
    }


def _synthetic_projection() -> tuple[dict[str, str], str]:
    projection = {}
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_long"):
        for task in range(10):
            for trial in (40, 41, 42):
                group_id = f"{suite}/task={task:02d}/trial={trial}/seed=0"
                projection[group_id] = hashlib.sha256(group_id.encode()).hexdigest()
    return projection, audit.EXPECTED_VALIDATION_IDENTITY_DIGEST


def _local_manifest_projection() -> tuple[dict[str, str], str]:
    manifest = json.loads(LOCAL_VALIDATION_MANIFEST.read_text(encoding="utf-8"))
    return (
        {str(row["group_id"]): str(row["sha256"])
         for row in manifest["groups"]},
        str(manifest["identity_digest"]),
    )


def _task_keys() -> list[str]:
    projection, identity = _synthetic_projection()
    receipt = audit.select_outcome_blind_panel(
        projection, identity_digest=identity,
    )
    return sorted({str(row["task_key"]) for row in receipt["ordered_rows"]})


def _bounds(point_means, uppers, lowers=None, benefits=None):
    n = len(point_means)
    return {
        "point_means": list(point_means),
        "upper_confidence_bounds": list(uppers),
        "lower_confidence_bounds": list(lowers or [-1.0] * n),
        "benefit_task_counts": list(benefits or [40] * n),
    }


def test_exact_three_step_round_robin_geometry_is_24_unique_and_four_per_fold():
    groups = [tuple(range(100)) for _ in range(v1.N_FOLDS)]
    draws = []
    for rank in range(audit.EXPECTED_WORLD_SIZE):
        sampler = v2.RoundRobinOutcomeSamplerV3(
            groups,
            seed=v1.TRAIN_SEED,
            rank=rank,
            world_size=audit.EXPECTED_WORLD_SIZE,
            start_step=v2.START_STEP,
            total_updates=v2.PILOT_UPDATES,
            contexts_per_arm=2,
            identity_digests=[f"fold-{fold}" for fold in range(v1.N_FOLDS)],
        )
        for step in audit.AUDIT_STEPS:
            fold, group, _visit = sampler.group_at(step)
            draws.append((fold, group))
    assert audit.AUDIT_STEPS == (49666, 49667, 49668)
    assert len(draws) == len(set(draws)) == audit.EXPECTED_TRAIN_DRAWS == 24
    assert [sum(fold == expected for fold, _group in draws)
            for expected in range(v1.N_FOLDS)] == [4] * v1.N_FOLDS


def test_outcome_blind_panel_receipt_is_exact_pinned_and_covers_all_tasks():
    projection, identity = _synthetic_projection()
    receipt = audit.select_outcome_blind_panel(
        projection, identity_digest=identity,
    )
    assert len(receipt["sha256"]) == 64
    assert receipt["identity_digest_I"] == audit.EXPECTED_VALIDATION_IDENTITY_DIGEST
    assert receipt["terminal_rewards_not_used_or_accessed_by_selection_logic"] is True
    assert receipt["manifest_parser_materializes_unprojected_fields_before_selection"] is True
    assert receipt["sidecar_payload_read_before_receipt"] is False
    assert len(receipt["ordered_rows"]) == 48
    assert len({row["group_id"] for row in receipt["ordered_rows"]}) == 48
    tasks = {row["task_key"] for row in receipt["ordered_rows"]}
    assert len(tasks) == 40
    multiplicities = {
        key: sum(row["task_key"] == key for row in receipt["ordered_rows"])
        for key in tasks
    }
    assert sorted(multiplicities.values()) == [1] * 32 + [2] * 8
    assert {audit.task_suite(key) for key in tasks} == {
        "libero_spatial", "libero_object", "libero_goal", "libero_long",
    }
    for rank in range(audit.EXPECTED_WORLD_SIZE):
        rank_rows = [row for row in receipt["ordered_rows"]
                     if int(row["rank"]) == rank]
        assert len(rank_rows) == 6
        assert len({row["task_key"] for row in rank_rows}) == 5
        assert sum(row["panel_role"] == "second" for row in rank_rows) == 1


def test_panel_hash_domains_and_order_are_exact_utf8_pipe_strings():
    projection, identity = _synthetic_projection()
    receipt = audit.select_outcome_blind_panel(
        projection, identity_digest=identity,
    )
    rows_by_task = {}
    for row in receipt["ordered_rows"]:
        rows_by_task.setdefault(row["task_key"], []).append(row)
    all_by_task = {}
    for group_id in projection:
        all_by_task.setdefault(audit.task_key(group_id), []).append(group_id)
    for key, selected in rows_by_task.items():
        expected = sorted(all_by_task[key], key=lambda group_id: (
            hashlib.sha256(
                f"rr-audit-panel-v1|{identity}|{group_id}".encode("utf-8")
            ).hexdigest(),
            group_id,
        ))
        assert selected[0]["group_id"] == expected[0]
        if len(selected) == 2:
            assert selected[1]["group_id"] == expected[1]
    expected_extras = set(sorted(rows_by_task, key=lambda key: (
        hashlib.sha256(
            f"rr-audit-extra-v1|{identity}|{key}".encode("utf-8")
        ).hexdigest(),
        key,
    ))[:8])
    assert {key for key, rows in rows_by_task.items() if len(rows) == 2} == expected_extras


def test_sidecar_hash_is_pinned_but_does_not_adapt_group_selection():
    projection, identity = _synthetic_projection()
    original = audit.select_outcome_blind_panel(
        projection, identity_digest=identity,
    )
    changed = dict(projection)
    changed[next(iter(changed))] = "f" * 64
    modified = audit.select_outcome_blind_panel(changed, identity_digest=identity)
    assert [row["group_id"] for row in original["ordered_rows"]] == [
        row["group_id"] for row in modified["ordered_rows"]
    ]
    assert original["sha256"] != modified["sha256"]
    with pytest.raises(audit.RoundRobinDirectionAuditError,
                       match="identity digest differs"):
        audit.select_outcome_blind_panel(projection, identity_digest="0" * 64)


def test_hash_replans_choose_two_distinct_contexts_per_sampled_arm():
    first = audit.hashed_replans("suite/task=00/trial=40/seed=0", [9] * 8)
    second = audit.hashed_replans("suite/task=00/trial=40/seed=0", [9] * 8)
    assert first == second
    assert set(first) == set(range(1, recovery.GROUP_SIZE))
    assert all(len(values) == len(set(values)) == 2 for values in first.values())
    with pytest.raises(audit.RoundRobinDirectionAuditError,
                       match="fewer than two"):
        audit.hashed_replans("bad", [1] * 8)


def test_fixed_suite_bootstrap_is_10k_pinned_and_preserves_suite_columns():
    matrix, receipt = audit.make_suite_stratified_resample_matrix(_task_keys())
    assert tuple(matrix.shape) == (10_000, 40)
    assert receipt["samples"] == audit.BOOTSTRAP_SAMPLES == 10_000
    assert receipt["seed"] == audit.BOOTSTRAP_SEED == 49_666
    assert receipt["sha256"] == audit.EXPECTED_BOOTSTRAP_MATRIX_SHA256
    for columns in receipt["suite_columns"].values():
        block = matrix[:, columns[0]:columns[-1] + 1]
        assert int(block.min()) >= columns[0]
        assert int(block.max()) <= columns[-1]


def test_equal_task_weighting_halves_doubled_task_before_task_average():
    doubled = [torch.tensor([2.0]), torch.tensor([6.0])]
    singleton = [torch.tensor([10.0])]
    doubled_mean, doubled_contribution = (
        audit.equal_group_within_task_contribution(doubled, total_tasks=2)
    )
    singleton_mean, singleton_contribution = (
        audit.equal_group_within_task_contribution(singleton, total_tasks=2)
    )
    equal_task = doubled_contribution + singleton_contribution
    naive_group = (doubled[0] + doubled[1] + singleton[0]) / 3.0
    assert doubled_mean.item() == 4.0
    assert singleton_mean.item() == 10.0
    assert equal_task.item() == 7.0
    assert naive_group.item() == 6.0
    assert not torch.equal(equal_task, naive_group)


def test_bonferroni_bounds_use_higher_upper_and_lower_lower(monkeypatch):
    task_values = torch.stack([
        torch.linspace(-2.0, 1.0, 40),
        torch.linspace(-1.5, 0.5, 40),
        torch.linspace(-1.0, 0.0, 40),
    ])
    resamples, _receipt = audit.make_suite_stratified_resample_matrix(
        _task_keys(), samples=200,
    )
    calls = []
    real = torch.quantile

    def wrapped(*args, **kwargs):
        calls.append((args[1], kwargs.get("interpolation")))
        return real(*args, **kwargs)

    monkeypatch.setattr(torch, "quantile", wrapped)
    report = audit.bonferroni_task_bounds(task_values, resamples)
    assert calls == [(1.0 - 0.05 / 3.0, "higher"), (0.05 / 3.0, "lower")]
    assert report["family_size"] == 3
    assert report["upper_quantile_interpolation"] == "higher"
    assert report["lower_quantile_interpolation"] == "lower"
    assert report["upper_quantile"] == pytest.approx(1.0 - 0.05 / 3.0)
    assert report["lower_quantile"] == pytest.approx(0.05 / 3.0)


def test_percentile_bootstrap_numeric_oracle_is_not_basic_interval():
    values = torch.stack([
        torch.tensor([0.0] * 39 + [40.0]),
        torch.tensor([-4.0] * 20 + [2.0] * 20),
        torch.tensor([-1.0] * 40),
    ])
    resamples = torch.stack([
        torch.zeros(40, dtype=torch.int64),
        torch.full((40,), 39, dtype=torch.int64),
        torch.tensor([0, 39] * 20, dtype=torch.int64),
        torch.arange(40, dtype=torch.int64),
    ])
    report = audit.bonferroni_task_bounds(values, resamples)
    # Four bootstrap means for the asymmetric first row are [0,40,20,1].
    # Direct percentile endpoints are therefore [0,40]; a basic/pivot interval
    # around point mean 1 would be different.
    assert report["method"] == (
        "suite_stratified_task_percentile_bonferroni_one_sided"
    )
    assert report["point_means"][0] == 1.0
    assert report["lower_confidence_bounds"][0] == 0.0
    assert report["upper_confidence_bounds"][0] == 40.0


def test_gate_is_nonadaptive_alpha0_and_requires_all_three_endpoints():
    passed = audit.decide_direction_gate(
        endpoint_bounds=_bounds([-3, -2, -1], [-2, -1, -0.1]),
        endpoint_benefit_cosines=[0.2, 0.1, 0.01],
        adamw_increment_bounds=_bounds([-1, -1, -1], [0, 0, 0], [-2, -2, -2]),
        adamw_increment_benefit_cosines=[0.2, 0.0, -0.01],
        reference_gradient_bounds_passed=True,
        reference_vectors_bitwise_zero=False,
    )
    assert passed["status"] == (
        "PASS_TO_64_UPDATE_ALPHA0_BETA1_LAMBDA1_INELIGIBLE_PILOT"
    )
    assert passed["pre_frozen_alpha"] == 0.0
    assert passed["pass_authority"] == (
        "64_update_ineligible_pilot_alpha0_beta1_lambda1_only_subject_to_"
        "exact_recipe_freeze_and_controller_smoke"
    )
    assert passed["full_training_authorized"] is False

    for mutation in ("endpoint", "cosine", "catastrophe", "reference"):
        endpoint = _bounds([-3, -2, -1], [-2, -1, -0.1])
        cosines = [0.2, 0.1, 0.01]
        increment = _bounds([-1, -1, -1], [0, 0, 0], [-2, -2, -2])
        increment_cosines = [0.2, 0.0, -0.01]
        reference_bounds = True
        if mutation == "endpoint":
            endpoint["upper_confidence_bounds"][1] = 0.0
        elif mutation == "cosine":
            cosines[1] = 0.009
        elif mutation == "catastrophe":
            increment["lower_confidence_bounds"][2] = 0.01
        else:
            reference_bounds = False
        result = audit.decide_direction_gate(
            endpoint_bounds=endpoint,
            endpoint_benefit_cosines=cosines,
            adamw_increment_bounds=increment,
            adamw_increment_benefit_cosines=increment_cosines,
            reference_gradient_bounds_passed=reference_bounds,
            reference_vectors_bitwise_zero=False,
        )
        assert result["status"] == "ABORT_OUTCOME_OBJECTIVE"


def test_catastrophe_wrong_way_cosine_is_strictly_below_minus_point01():
    common = dict(
        endpoint_bounds=_bounds([-1, -1, -1], [-0.1, -0.1, -0.1]),
        endpoint_benefit_cosines=[0.1, 0.1, 0.1],
        adamw_increment_bounds=_bounds([-1, -1, -1], [0, 0, 0], [-1, -1, -1]),
    )
    edge = audit.decide_direction_gate(
        **common, adamw_increment_benefit_cosines=[0.0, -0.01, 0.0],
    )
    assert edge["passed"] is True
    harmful = audit.decide_direction_gate(
        **common, adamw_increment_benefit_cosines=[0.0, -0.010001, 0.0],
    )
    assert harmful["passed"] is False


def test_repeated_direct_backward_consistency_is_strict_and_first_authoritative():
    first = torch.tensor([1.0, -2.0, 3.0])
    report = audit.direct_repeat_consistency(
        first, first.clone(), label="direct-test",
    )
    assert report["first_direct_is_authoritative"] is True
    assert report["independent_synchronised_backward_calls"] == 2
    assert report["relative_residual"] == 0.0
    assert report["max_relative_residual"] == 1e-7
    with pytest.raises(audit.RoundRobinDirectionAuditError,
                       match="repeated direct backward is inconsistent"):
        audit.direct_repeat_consistency(
            first, first + torch.tensor([1e-3, 0.0, 0.0]),
            label="direct-test",
        )


def test_real_clone_adamw_replay_matches_independent_multigroup_torch():
    proposal = TinyProposal().eval()
    before = copy.deepcopy(proposal.state_dict())
    gradients = [
        torch.tensor([2.0, -1.0, 0.5, 3.0, 0.2, -0.4]),
        torch.tensor([-1.0, 2.0, 2.5, -0.5, 0.3, 0.7]),
        torch.tensor([0.5, 0.25, -1.5, 2.0, -0.8, 0.1]),
    ]
    report = audit.virtual_adamw_clone_replay(
        proposal, gradients, cfg=_cfg(),
    )

    weight = nn.Parameter(before["weight"].clone())
    bias = nn.Parameter(before["bias"].clone())
    optimizer = torch.optim.AdamW([
        {"name": "proposal/decay", "module": "proposal", "params": [weight],
         "weight_decay": 0.05, "lr_scale": 0.0125},
        {"name": "proposal/nodecay", "module": "proposal", "params": [bias],
         "weight_decay": 0.0, "lr_scale": 0.0125},
    ], lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
    schedule = optim_schedule.CosineWithWarmup(
        3e-4, 2000, v1.SCHEDULE_STEPS, 0.05,
    )
    initial = torch.cat([weight.detach().reshape(-1), bias.detach().reshape(-1)])
    expected = []
    for step, gradient in zip(audit.AUDIT_STEPS, gradients, strict=True):
        optimizer.zero_grad(set_to_none=True)
        weight.grad = gradient[:4].reshape_as(weight).clone()
        bias.grad = gradient[4:].reshape_as(bias).clone()
        torch.nn.utils.clip_grad_norm_([weight, bias], 1.0)
        schedule.apply(optimizer, step)
        optimizer.step()
        expected.append(torch.cat([
            weight.detach().reshape(-1), bias.detach().reshape(-1),
        ]) - initial)
    for actual, wanted in zip(
        report["vectors"]["cumulative"], expected, strict=True,
    ):
        assert torch.equal(actual, wanted)
    assert report["virtual_clone_optimizer_steps"] == 3
    assert report["live_optimizer_steps"] == 0
    assert report["authority"].startswith("real_torch_optim_AdamW")
    assert len(report["step_rows"]) == 3
    assert proposal.weight.grad is proposal.bias.grad is None
    assert all(torch.equal(proposal.state_dict()[key], value)
               for key, value in before.items())


def test_clone_replay_reports_production_decay_and_exact_schedule():
    proposal = TinyProposal().eval()
    gradients = [torch.ones(6), -torch.ones(6), torch.arange(6).float()]
    production = audit.virtual_adamw_clone_replay(
        proposal, gradients, cfg=_cfg(),
    )
    assert production["production_weight_decay_enabled"] is True
    assert production["weight_decay"] == 0.05
    expected_lrs = [audit.proposal_lr_at(_cfg(), step)
                    for step in audit.AUDIT_STEPS]
    assert [row["proposal_lr"] for row in production["step_rows"]] == expected_lrs


def test_production_clipped_sgd_uses_real_clip_and_live_model_is_unchanged():
    proposal = TinyProposal().eval()
    before = copy.deepcopy(proposal.state_dict())
    gradients = [torch.arange(1, 7).float()] * 3
    report = audit.cumulative_clipped_sgd_direction(
        proposal, gradients, cfg=_cfg(),
    )
    assert report["definition"].startswith("-sum_t proposal_lr")
    assert all(norm == pytest.approx(float(gradients[0].norm()))
               for norm in report["preclip_norms_from_torch_clip_grad_norm"])
    assert float(report["delta"].norm()) > 0.0
    assert proposal.weight.grad is proposal.bias.grad is None
    assert all(torch.equal(proposal.state_dict()[key], value)
               for key, value in before.items())


def test_reference_gradient_contract_accepts_exact_zero_and_rejects_material():
    proposal = TinyProposal().eval()
    zero = (proposal.weight * 0.0).sum()
    evidence = audit._reference_gradient_evidence(
        zero, proposal, grpo_norm=2.0, world=1,
        retain_graph=False, label="zero",
    )
    assert evidence["value_exactly_zero"] is True
    assert evidence["synchronised_gradient_bitwise_all_zero"] is True
    assert evidence["synchronised_gradient_norm"] == 0.0

    tiny_roundoff = (
        proposal.weight - proposal.weight.detach()
    ).sum() * 1e-7
    tiny = audit._reference_gradient_evidence(
        tiny_roundoff,
        proposal,
        grpo_norm=1.0,
        world=1,
        retain_graph=False,
        label="tiny-roundoff",
    )
    assert tiny["value_exactly_zero"] is True
    assert tiny["synchronised_gradient_bitwise_all_zero"] is False
    assert tiny["bound_passed"] is True

    material = proposal.weight.sum() * 1e-3
    with pytest.raises(v1.OutcomeGRPOError,
                       match="reference value is not exactly zero"):
        audit._reference_gradient_evidence(
            material, proposal, grpo_norm=1.0, world=1,
            retain_graph=False, label="material",
        )
    zero_value_material_gradient = (
        proposal.weight - proposal.weight.detach()
    ).sum() * 1e-3
    bounded = audit._reference_gradient_evidence(
        zero_value_material_gradient,
        proposal,
        grpo_norm=1.0,
        world=1,
        retain_graph=False,
        label="zero-value-material-gradient",
    )
    assert bounded["value_exactly_zero"] is True
    assert bounded["bound_passed"] is False


def test_benefit_cosine_sign_is_positive_for_loss_reducing_update():
    heldout = torch.tensor([1.0, 0.0])
    assert audit.aggregate_benefit_cosine(
        heldout, torch.tensor([-2.0, 0.0]),
    ) == pytest.approx(1.0)
    assert audit.aggregate_benefit_cosine(
        heldout, torch.tensor([2.0, 0.0]),
    ) == pytest.approx(-1.0)


def test_v2_source_config_and_panel_are_fail_closed():
    assert audit.FORMAT_VERSION == 2
    assert audit.KIND == (
        "loom_outcome_grpo_v2_round_robin_direct_direction_audit"
    )
    assert audit.OUTPUT_NAME_PREFIX.startswith(
        "outcome_round_robin_direction_audit_v2_"
    )
    assert [row["job_id"] for row in audit.INVALID_INSTRUMENTATION_HISTORY] == [
        32575962, 32576514,
    ]
    assert all(row["stopped_before_panel_statistics"] is True
               and row["scientific_report_published"] is False
               for row in audit.INVALID_INSTRUMENTATION_HISTORY)
    cfg, config_hash, scaffold = audit._validate_config(
        Path("configs/r0a_outcome_grpo_v2_pilot.yaml").resolve()
    )
    assert config_hash == "67277938c51075d2"
    assert scaffold["method_status"] == v2.METHOD_STATUS_SCAFFOLD
    identity = audit._source_identity()
    assert identity["v2_trainer"]["sha256"] == audit.EXPECTED_V2_SOURCE_SHA256
    assert set(identity["diagnostic"]["files"]) == set(audit._AUDIT_SOURCE_FILES)
    file_identity = audit._config_file_identity(
        Path("configs/r0a_outcome_grpo_v2_pilot.yaml")
    )
    assert file_identity["sha256"] == audit.EXPECTED_CONFIG_FILE_SHA256
    with pytest.raises(audit.RoundRobinDirectionAuditError,
                       match="canonical checked-in file"):
        audit._validate_config(Path("configs/r0a_outcome_grpo.yaml").resolve())


@pytest.mark.skipif(
    not LOCAL_VALIDATION_MANIFEST.is_file(),
    reason="gitignored authenticated development collection is unavailable",
)
def test_local_authenticated_panel_receipt_matches_frozen_pin():
    projection, identity = _local_manifest_projection()
    receipt = audit.select_outcome_blind_panel(
        projection, identity_digest=identity,
    )
    assert receipt["sha256"] == audit.EXPECTED_PANEL_GROUP_RECEIPT_SHA256
    cfg, _hash, _scaffold = audit._validate_config(
        Path("configs/r0a_outcome_grpo_v2_pilot.yaml").resolve()
    )
    from_config, root = audit.pre_reward_panel_receipt(cfg)
    assert from_config == receipt
    assert root == Path("runs/outcome_recovery_s49666_validation").resolve()


def test_output_is_dedicated_and_exclusive(tmp_path: Path):
    allowed = (
        audit.ROOT / audit.OUTPUT_DIR_REL
        / f"{audit.OUTPUT_NAME_PREFIX}unused_cpu_contract.json"
    ).resolve()
    assert not allowed.exists()
    assert audit._validate_output_path(allowed) == allowed
    with pytest.raises(audit.RoundRobinDirectionAuditError,
                       match="directly inside"):
        audit._validate_output_path(tmp_path / "wrong.json")
    target = tmp_path / "report.json"
    audit.exclusive_json_write(target, {"first": 1})
    original = target.read_bytes()
    with pytest.raises(audit.RoundRobinDirectionAuditError,
                       match="refusing existing"):
        audit.exclusive_json_write(target, {"second": 2})
    assert target.read_bytes() == original


def test_python_entry_point_has_no_training_or_tuning_surface():
    source = Path(audit.__file__).read_text(encoding="utf-8")
    run_source = inspect.getsource(audit.run_audit)
    demo_source = inspect.getsource(audit._demo_reference_only)
    clone_source = inspect.getsource(audit.virtual_adamw_clone_replay)
    decision_source = inspect.getsource(audit.decide_direction_gate)
    assert "RoundRobinOutcomeSamplerV3" in inspect.getsource(
        audit._build_round_robin_sampler
    )
    assert "sampled_group_objectives_v2" in run_source
    assert "direct_grpo" in run_source
    assert "repeated_direct_grpo" in run_source
    assert "direct_full_loss" in run_source
    assert "repeated_direct_full" in run_source
    direct_full_source = run_source.split(
        "direct_full_loss = (", 1
    )[1].split(")", 1)[0]
    assert "objectives.grpo" in direct_full_source
    assert "objectives.balance" in direct_full_source
    assert "objectives.recovery_forward_kl" in direct_full_source
    assert "demo_reference" in direct_full_source
    assert "sparse_ce" not in direct_full_source
    assert "demo_anchor.losses" not in run_source
    assert "sparse_q_action_ce" not in run_source
    assert "proposal_sparse_ce_loss" not in demo_source
    assert "log_current" not in demo_source
    assert '"sparse_ce_computed": False' in demo_source
    assert '"sparse_ce_graph_included": False' in run_source
    assert '"disabled_sparse_ce_graph_excluded": True' in run_source
    assert "measure_synchronised_component_gradients" not in source
    assert "weighted_alpha_gradient" not in source
    assert "alpha_sweep_attribution_only" not in source
    assert "algebraic" not in source
    assert "include_weight_decay" not in source
    assert "adamw_nodecay" not in source
    assert "collection.load(" in run_source
    assert "validation.load(" in run_source
    assert "authenticate_selected_contexts(" in run_source
    assert "torch.autograd.grad(" in inspect.getsource(
        audit.component_audit._local_gradient_vector
    )
    assert "optimizer.step()" in clone_source
    assert run_source.count(".step()") == 0
    assert "train_outcome_grpo_v2(" not in run_source
    assert "write_descendant_checkpoint" not in source
    assert "eval_libero" not in source
    assert "selected_alpha" not in decision_source
    assert "pre_frozen_alpha" in decision_source
    assert "destroy_process_group()" in inspect.getsource(audit.main)
    assert set(action.dest for action in audit.build_parser()._actions) == {
        "help", "config", "out",
    }


def test_launcher_is_30m_eight_a100_diagnostic_only():
    launcher = Path(
        "scripts/outcome_round_robin_direction_audit.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --time=00:30:00" in launcher
    assert "#SBATCH --nodes=1" in launcher
    assert "#SBATCH --gpus-per-node=8" in launcher
    assert "#SBATCH --ntasks-per-node=8" in launcher
    assert "scripts/outcome_round_robin_direction_audit.py" in launcher
    assert "bootstrap=10000 seed=49666" in launcher
    assert "format_version=2" in launcher
    assert "outcome_round_robin_direction_audit_v2_s49666_" in launcher
    assert "alpha=0 pre_frozen=true" in launcher
    assert 'report["config"]["sparse_ce_disabled_graph_excluded"] is True' in launcher
    assert 'report["direction_construction"]["sparse_ce_graph_included"] is False' in launcher
    assert 'report["direction_construction"]["sparse_ce_computed"] is False' in launcher
    assert 'report["direction_construction"]["virtual_clone_optimizer_steps"] == 3' in launcher
    assert "attribution_sweeps=false" in launcher
    assert "32575962, 32576514" in launcher
    assert "alpha_selected_after_results" not in launcher
    assert "live_optimizer_steps=0" in launcher
    assert "64_update_ineligible_pilot_alpha0_beta1_lambda1_only" in launcher
    assert "scripts/train_outcome_grpo_v2.py" not in launcher
    assert "scripts/eval_libero" not in launcher
    assert "sbatch " not in "\n".join(
        line for line in launcher.splitlines() if not line.startswith("#")
    )
