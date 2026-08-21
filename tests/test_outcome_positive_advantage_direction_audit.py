"""Focused contracts for the isolated PA no-update direction gate."""

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
from loom.train import outcome_positive_advantage as pa
from loom.train.loop import read_config
from scripts import outcome_positive_advantage_direction_audit as audit
from scripts import outcome_round_robin_direction_audit as direct_v2


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / audit.EXPECTED_CONFIG_REL
TRIGGER = ROOT / audit.TRIGGER_REPORT_REL


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bounds(
    *,
    points: tuple[float, float, float] = (-1.0, -1.0, -1.0),
    upper: tuple[float, float, float] = (-0.1, -0.1, -0.1),
    lower: tuple[float, float, float] = (-2.0, -2.0, -2.0),
) -> dict:
    return {
        "point_means": list(points),
        "upper_confidence_bounds": list(upper),
        "lower_confidence_bounds": list(lower),
    }


def _cosine(value: float = 0.02) -> dict:
    return {
        "defined": True,
        "value": value,
        "gate_value": value,
        "heldout_gradient_norm": 1.0,
        "direction_norm": 1.0,
        "undefined_reason": None,
    }


def _decision(**overrides):
    values = {
        "primary_names": ("pa_sgd", "full_sgd", "full_adamw"),
        "endpoint_bounds": _bounds(),
        "endpoint_cosines": [_cosine(), _cosine(), _cosine()],
        "increment_names": ("i0", "i1", "i2"),
        "increment_bounds": _bounds(points=(-0.1, -0.1, -0.1)),
        "increment_cosines": [_cosine(), _cosine(), _cosine()],
        "all_reference_relative_bounds_passed": True,
        "all_recovery_second_pass_gradients_bitwise_zero": True,
        "all_demo_analytic_vjp_gradients_bitwise_zero": True,
    }
    values.update(overrides)
    return audit.decide_pa_direction_gate(**values)


def _minimal_recovery_payload() -> tuple[nn.Module, dict, dict[int, tuple[int]]]:
    proposal = nn.Linear(1, 1, bias=False)
    arms: list[dict] = [{"terminal_reward": torch.tensor(0.0)}]
    for _arm in range(1, recovery.GROUP_SIZE):
        arms.append({
            "z": torch.zeros(1, 1, 1),
            "lang": torch.zeros(1, 1),
            "ordered_support": torch.zeros(1, 1, dtype=torch.int64),
            "old_logprob": torch.zeros(1, dtype=torch.float32),
            "terminal_reward": torch.tensor(0.0),
        })
    indices = {arm: (0,) for arm in range(1, recovery.GROUP_SIZE)}
    return proposal, {"arms": arms}, indices


def test_frozen_pa_core_and_immutable_direct_v2_helper_closures_match():
    assert _sha(ROOT / "loom/train/outcome_positive_advantage.py") == (
        audit.EXPECTED_PA_CORE_FILE_SHA256
    )
    assert pa.core_source_identity()["sha256"] == (
        audit.EXPECTED_PA_CORE_SOURCE_SHA256
    )
    assert _sha(ROOT / "scripts/outcome_round_robin_direction_audit.py") == (
        audit.EXPECTED_DIRECT_V2_FILE_SHA256
    )
    direct_identity = direct_v2._source_identity()
    assert direct_identity["diagnostic"]["sha256"] == (
        audit.EXPECTED_DIRECT_V2_DIAGNOSTIC_SHA256
    )
    assert direct_identity["v2_trainer"]["sha256"] == (
        audit.EXPECTED_V2_SOURCE_SHA256
    )


def test_new_config_is_canonical_pinned_and_nonlaunchable():
    cfg, resolved, evidence = audit._validate_config(CONFIG)
    assert audit.FORMAT_VERSION == 2
    assert _sha(CONFIG) == audit.EXPECTED_CONFIG_FILE_SHA256
    assert v1._config_hash(read_config(CONFIG)) == audit.EXPECTED_RESOLVED_CONFIG_HASH
    assert resolved == audit.EXPECTED_RESOLVED_CONFIG_HASH
    assert evidence["passed"] is True
    assert cfg["run"]["steps"] is None
    assert cfg["run"]["ckpt_every"] is None
    assert cfg["train_modules"] == []
    inherited = cfg["outcome_grpo_v2"]
    assert inherited["method_status"] == (
        "RETIRED_TRAIN_RECIPE_AUTHENTICATED_INPUTS_ONLY"
    )
    assert inherited["stop_step"] is None
    assert inherited["snapshot_steps"] == []
    assert inherited["sampler"]["total_updates"] is None
    assert inherited["train_trust_panel"]["every"] is None
    assert inherited["artifact_policy"]["pilot_checkpoint_only"] is False
    exposures = inherited["validation_lineage"]["current_development_collection"][
        "exposures"
    ]
    assert exposures == [
        "v1_terminal_selection",
        "early_curve_diagnostic",
        "component_gradient_projection",
        "round_robin_direction_audit",
        "positive_advantage_direction_audit",
    ]
    assert len(exposures) == len(set(exposures))
    assert cfg["artifact_policy"]["checkpoint_emission"] == "forbidden"
    assert tuple(cfg["outcome_positive_advantage_audit"][
        "instrumentation_history"
    ]) == audit.INVALID_INSTRUMENTATION_HISTORY


def test_config_names_only_pa_plus_two_unit_references_for_train():
    cfg = read_config(CONFIG)
    losses = cfg["losses"]
    assert losses["positive_advantage"]["enabled"] is True
    assert losses["positive_advantage"]["weight"] == 1.0
    assert losses["recovery_reference"]["weight"] == 1.0
    assert losses["demo_reference"]["weight"] == 1.0
    assert losses["demo_reference"]["kind"] == (
        "exact_analytic_vjp_dense_categorical_forward_kl"
    )
    for name in ("grpo", "proposal", "balance"):
        assert losses[name]["enabled"] is False
        assert losses[name]["weight"] == 0.0
    recipe = cfg["outcome_positive_advantage_audit"]["frozen_direction_recipe"]
    assert recipe["train_scoring_geometry"] == (
        "read_only_authentication_replay_plus_two_objective_graph_exact_B1_passes_per_train_point"
    )
    assert recipe["authentication_replay"] == (
        "exact_selected_context_identity_no_objective_graph"
    )
    assert recipe["second_pass_identity_evidence"] == (
        "exact_zero_recovery_value_and_bitwise_zero_gradient"
    )
    assert recipe["second_pass_scorer_authentication"] == (
        "exact_current_float_equals_stored_old_per_selected_row_or_INVALID"
    )
    assert recipe["demo_logit_authentication"] == (
        "exact_live_seed_logits_per_horizon_or_INVALID"
    )
    assert recipe["demo_anchor_construction"] == (
        audit.DEMO_ANCHOR_CONSTRUCTION_RECEIPT
    )
    assert recipe["reference_identity_classification"] == {
        "missing_disconnected_nonfinite_auth_or_ratio_failure": "INVALID_NO_REPORT",
        "complete_finite_nonzero_recovery_or_demo_vjp": "SCIENTIFIC_ABORT",
        "pass_requires_local_and_synchronised_bitwise_zero": True,
    }
    assert recipe["tuning_or_sweep"] == "forbidden"
    assert recipe["coefficient_switching"] == "forbidden"


def test_demo_anchor_uses_authenticated_v2_config_only_for_construction(
    monkeypatch,
):
    canonical = read_config(CONFIG)
    canonical_hash = v1._config_hash(canonical)
    canonical_target = copy.deepcopy(canonical["losses"]["proposal"])
    captured: dict = {}
    sentinel_anchor = object()

    def fake_expert_from_parent(
        parent, live_proposal, *, trainer_cfg, device, rank, world_size,
    ):
        captured["parent"] = parent
        captured["proposal"] = live_proposal
        captured["expert_cfg"] = copy.deepcopy(trainer_cfg)
        captured["device"] = device
        captured["rank"] = rank
        captured["world_size"] = world_size
        return sentinel_anchor

    monkeypatch.setattr(
        audit.v1.ExpertAnchor,
        "from_parent",
        staticmethod(fake_expert_from_parent),
    )
    monkeypatch.setattr(
        audit.v1,
        "_load_proposal",
        lambda _parent, *, device: copy.deepcopy(proposal).to(device).eval(),
    )
    proposal = nn.Linear(1, 1, bias=False)
    parent = {"authenticated": True}
    anchor, receipt = audit._construct_demo_reference_anchor(
        parent, proposal, pa_cfg=canonical, device=torch.device("cpu"),
        rank=0, world=1,
    )

    assert isinstance(anchor, audit.v2.DemoReferenceAnchorV2)
    assert anchor.anchor is sentinel_anchor
    assert captured["parent"] is parent
    assert captured["proposal"] is proposal
    assert captured["rank"] == 0 and captured["world_size"] == 1
    target = captured["expert_cfg"]["losses"]["proposal"]
    assert target["enabled"] is True
    assert target["weight"] == 1.0
    assert target["mode"] == "sparse_ce"
    assert target["temperature"] == 1.0
    assert target["detach_belief"] is True
    assert canonical["losses"]["proposal"] == canonical_target
    assert canonical_target["enabled"] is False
    assert canonical_target["weight"] == 0.0
    assert v1._config_hash(canonical) == canonical_hash
    assert receipt["config_source"] == (
        "authenticated_inherited_v2_resolved_config"
    )
    assert receipt["inherited_construction_target"]["weight"] is None
    assert receipt["v2_core_internal_target_producer_weight"] == 1.0
    assert receipt["canonical_pa_config_unchanged_after_constructor"] is True
    assert receipt["construction_config_unchanged_after_constructor"] is True
    assert receipt["sparse_ce_scalar_computed"] is False
    assert receipt["sparse_ce_graph_constructed"] is False
    wrapper_source = inspect.getsource(audit._construct_demo_reference_anchor)
    assert "proposal_sparse_ce_loss" not in wrapper_source
    assert ".losses(" not in wrapper_source


def test_config_source_provenance_binds_both_immutable_dependencies():
    namespace = read_config(CONFIG)["outcome_positive_advantage_audit"]
    excluded = namespace["inherited_authenticated_recipe"][
        "excluded_semantic_scope"
    ]
    assert excluded == ["train_objective", "coefficient_selection", "launch_readiness"]
    assert len(excluded) == len(set(excluded))
    source = namespace["source_provenance"]
    assert source["positive_advantage_core_file_sha256"] == (
        audit.EXPECTED_PA_CORE_FILE_SHA256
    )
    assert source["positive_advantage_core_closure_sha256"] == (
        audit.EXPECTED_PA_CORE_SOURCE_SHA256
    )
    assert source["immutable_direct_v2_helper_file_sha256"] == (
        audit.EXPECTED_DIRECT_V2_FILE_SHA256
    )
    assert source["immutable_direct_v2_diagnostic_closure_sha256"] == (
        audit.EXPECTED_DIRECT_V2_DIAGNOSTIC_SHA256
    )
    assert source["own_diagnostic_closure"]["files"] == list(
        audit._AUDIT_SOURCE_FILES
    )
    assert source["own_diagnostic_closure"]["scheme"] == (
        "sha256(path-nul-sha256-nul)-v1"
    )


def test_config_path_is_fail_closed(tmp_path: Path):
    copied = tmp_path / CONFIG.name
    copied.write_bytes(CONFIG.read_bytes())
    with pytest.raises(audit.PositiveAdvantageDirectionAuditError,
                       match="canonical file"):
        audit._validate_config(copied)


@pytest.mark.skipif(not TRIGGER.is_file(), reason="local diagnostic report absent")
def test_exact_abort_trigger_report_is_authenticated_and_semantically_valid():
    assert _sha(TRIGGER) == audit.TRIGGER_REPORT_SHA256
    report = json.loads(TRIGGER.read_text(encoding="utf-8"))
    receipt = audit._validate_trigger_payload(report, TRIGGER.resolve())
    assert receipt["status"] == "ABORT_OUTCOME_OBJECTIVE"
    assert receipt["execution_validated"] is True
    assert receipt["decision_passed"] is False
    identity = audit.authenticate_trigger_once(rank=0, world=1)
    assert identity["file"]["sha256"] == audit.TRIGGER_REPORT_SHA256
    assert identity["receipt"] == receipt


@pytest.mark.skipif(not TRIGGER.is_file(), reason="local diagnostic report absent")
@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda row: row.update(status="PASS"), "ABORT status"),
        (lambda row: row.update(execution_validated=False), "execution-valid"),
        (lambda row: row["decision"].update(passed=True), "locked ABORT"),
        (lambda row: row["no_mutation"].update(live_optimizer_steps=1),
         "no-mutation"),
        (lambda row: row["outcome_blind_panel"]["group_receipt"].update(
            sha256="0" * 64), "panel receipt"),
    ],
)
def test_trigger_semantic_mutations_fail_closed(mutation, match):
    report = json.loads(TRIGGER.read_text(encoding="utf-8"))
    mutation(report)
    with pytest.raises(audit.PositiveAdvantageDirectionAuditError, match=match):
        audit._validate_trigger_payload(report, TRIGGER.resolve())


def test_geometry_is_exact_three_offsets_24_draws_and_four_per_fold():
    assert audit.AUDIT_OFFSETS == (0, 1, 2)
    assert audit.AUDIT_STEPS == (49666, 49667, 49668)
    assert audit.EXPECTED_WORLD_SIZE == 8
    draws = []
    for step in audit.AUDIT_STEPS:
        for rank in range(audit.EXPECTED_WORLD_SIZE):
            draw = (step - 49666) * 8 + rank
            fold = draw % 6
            q = draw // 6
            draws.append((draw, fold, q))
    assert len(draws) == len({row[0] for row in draws}) == 24
    assert {fold: sum(row[1] == fold for row in draws) for fold in range(6)} == {
        fold: 4 for fold in range(6)
    }


def test_pa_audit_calls_immutable_panel_bootstrap_clip_and_adamw_helpers():
    source = inspect.getsource(audit.run_audit)
    for call in (
        "direct_v2.pre_reward_panel_receipt",
        "direct_v2.attach_panel_sampling_receipt",
        "direct_v2._build_round_robin_sampler",
        "direct_v2.cumulative_clipped_sgd_direction",
        "direct_v2.virtual_adamw_clone_replay",
        "direct_v2.equal_group_within_task_contribution",
        "direct_v2.rehash_rank_sidecars",
    ):
        assert call in source
    analysis_source = inspect.getsource(audit.analyse_panel_directions)
    assert "direct_v2.make_suite_stratified_resample_matrix" in analysis_source
    assert "direct_v2.bonferroni_task_bounds" in analysis_source
    assert "analyse_panel_directions(" in source
    module_source = inspect.getsource(audit)
    assert "def select_outcome_blind_panel(" not in module_source
    assert "def make_suite_stratified_resample_matrix(" not in module_source
    assert "def virtual_adamw_clone_replay(" not in module_source
    assert "def cumulative_clipped_sgd_direction(" not in module_source


def test_train_geometry_is_two_pass_direct_pa_then_recovery_only():
    source = inspect.getsource(audit.run_audit)
    train = source.split(
        "for offset, step in enumerate(AUDIT_STEPS):", 1
    )[1].split(
        "_require(len(train_selection_rows) == EXPECTED_TRAIN_DRAWS", 1
    )[0]
    assert "pa.sampled_positive_advantage_objective(" in train
    assert "_recovery_reference_only(" in train
    assert train.index("pa.sampled_positive_advantage_objective(") < train.index(
        "_recovery_reference_only("
    )
    assert "v2.sampled_group_objectives_v2(" not in train
    assert "proposal_sparse_ce_loss" not in train
    assert ".balance" not in train
    assert "direct_full_loss = (\n                    pa_objective.loss + recovery_reference + demo_reference" in train
    assert "direct_pa +" not in train
    recovery_source = inspect.getsource(audit._recovery_reference_only)
    assert recovery_source.count("v1.stored_order_logprob(") == 1
    assert "v2.recovery_pl_forward_kl(" in recovery_source
    assert "clipped_grpo_objective" not in recovery_source
    assert "proposal_switch_balance" not in recovery_source


def test_demo_reference_uses_exact_pa_vjp_and_equal_horizon_mean():
    source = inspect.getsource(audit._analytic_demo_reference_only)
    assert "pa.analytic_categorical_forward_kl(" in source
    assert 'reduction="mean"' in source
    assert "torch.stack(terms).mean()" in source
    assert "anchor.anchor._prepare(global_step)" in source
    assert "torch.equal(current_logits, seed_logits)" in source
    assert "dense_categorical_forward_kl" not in source
    assert "sparse" in source  # explicit no-sparse evidence only
    run = inspect.getsource(audit.run_audit)
    identity_guard = 'demo_metrics["live_seed_logits_bitwise_identical"]'
    assert identity_guard in run
    assert run.index(identity_guard) < run.index(
        "_complete_reference_gradient_evidence(\n"
        "                demo_reference"
    )


def test_direct_pa_and_direct_full_are_each_repeated_first_authoritative():
    source = inspect.getsource(audit.run_audit)
    assert source.count("label=f\"offset {offset} first direct PA\"") == 1
    assert source.count("label=f\"offset {offset} repeated direct PA\"") == 1
    assert source.count("label=f\"offset {offset} first direct PA full\"") == 1
    assert source.count("label=f\"offset {offset} repeated direct PA full\"") == 1
    assert source.count("direct_v2.direct_repeat_consistency(") == 2
    assert "algebraic" not in source.lower()


def test_collective_bearing_phases_have_coordinated_local_failure_boundaries():
    wrapper = inspect.getsource(audit._coordinated_synchronised_loss_gradient)
    assert "direct_v2._synchronised_loss_gradient(" in wrapper
    assert "v1._raise_if_any_rank_failed(" in wrapper
    reference = inspect.getsource(audit._complete_reference_gradient_evidence)
    assert "synchronised reference evidence" in reference
    assert reference.count("v1._raise_if_any_rank_failed(") == 2
    run = inspect.getsource(audit.run_audit)
    for label in (
        "gradient preflight",
        "PA repeat evidence",
        "full-scalar construction",
        "full repeat evidence",
        "post-gradient bookkeeping",
        "PA panel gathered-task closure",
        "PA post-panel bootstrap/projection analysis",
        "rank-zero PA report publication",
    ):
        assert label in run
    assert run.index("analysis = analyse_panel_directions(") < run.index(
        '"PA post-panel bootstrap/projection analysis"'
    )


def test_zero_direct_repeat_is_numerically_valid():
    zero = torch.zeros(17)
    evidence = direct_v2.direct_repeat_consistency(zero, zero.clone(), label="PA")
    assert evidence["passed"] is True
    assert evidence["first_norm"] == 0.0
    assert evidence["relative_residual"] == 0.0


def test_complete_reference_helper_preserves_local_zero_and_rejects_missing():
    complete = nn.Linear(2, 1, bias=True)
    zero_loss = sum(parameter.sum() * 0.0 for parameter in complete.parameters())
    evidence = audit._complete_reference_gradient_evidence(
        zero_loss, complete, objective_norm=2.0, world=1,
        retain_graph=False, label="complete zero",
    )
    assert evidence["value_exactly_zero"] is True
    assert evidence["missing_parameter_gradients"] == []
    assert evidence["local_gradient_bitwise_all_zero"] is True
    assert evidence["synchronised_gradient_bitwise_all_zero"] is True
    assert evidence["bound_passed"] is True

    incomplete = nn.Linear(2, 1, bias=True)
    missing_loss = incomplete.weight.sum() * 0.0
    with pytest.raises(Exception, match="missing parameters"):
        audit._complete_reference_gradient_evidence(
            missing_loss, incomplete, objective_norm=1.0, world=1,
            retain_graph=False, label="incomplete",
        )


def test_complete_reference_helper_rejects_nonfinite_graph_and_value():
    graph = nn.Linear(1, 1, bias=False)
    nonfinite_graph = graph.weight.sum() * float("nan")
    with pytest.raises(Exception, match="nan/inf"):
        audit._complete_reference_gradient_evidence(
            nonfinite_graph, graph, objective_norm=1.0, world=1,
            retain_graph=False, label="nonfinite graph",
        )

    value = nn.Linear(1, 1, bias=False)
    nonfinite_value = value.weight.sum() * 0.0 + float("inf")
    with pytest.raises(Exception, match="nonfinite"):
        audit._complete_reference_gradient_evidence(
            nonfinite_value, value, objective_norm=1.0, world=1,
            retain_graph=False, label="nonfinite value",
        )


def test_recovery_second_pass_exactly_authenticates_current_against_old(
    monkeypatch,
):
    proposal, payload, indices = _minimal_recovery_payload()

    def exact_scorer(model, _z, _lang, _order):
        current = model.weight.reshape(-1)[:1] * 0.0
        return current, torch.zeros(1, 1)

    monkeypatch.setattr(audit.v1, "stored_order_logprob", exact_scorer)
    reference, identity = audit._recovery_reference_only(
        proposal, payload, indices, device=torch.device("cpu"),
    )
    assert float(reference.detach()) == 0.0
    assert identity["selected_atoms"] == 7
    assert identity["max_abs_current_old_logprob_error"] == 0.0
    assert identity["current_old_logprobs_bitwise_identical"] is True


def test_recovery_second_pass_current_old_mismatch_is_invalid(monkeypatch):
    proposal, payload, indices = _minimal_recovery_payload()

    def mismatched_scorer(model, _z, _lang, _order):
        current = model.weight.reshape(-1)[:1] * 0.0 + 1e-7
        return current, torch.zeros(1, 1)

    monkeypatch.setattr(audit.v1, "stored_order_logprob", mismatched_scorer)
    with pytest.raises(
        audit.PositiveAdvantageDirectionAuditError,
        match="second-pass current/old identity differs",
    ):
        audit._recovery_reference_only(
            proposal, payload, indices, device=torch.device("cpu"),
        )


def test_small_nonzero_reference_vjp_is_evidenced_not_hidden_by_bound():
    proposal = nn.Linear(1, 1, bias=False)
    parameter = proposal.weight
    zero_value_nonzero_vjp = (
        parameter - parameter.detach()
    ).sum() * 1e-7
    evidence = audit._complete_reference_gradient_evidence(
        zero_value_nonzero_vjp, proposal, objective_norm=1.0, world=1,
        retain_graph=False, label="tiny identity residual",
    )
    assert evidence["value_exactly_zero"] is True
    assert evidence["bound_passed"] is True
    assert evidence["local_gradient_bitwise_all_zero"] is False
    assert evidence["synchronised_gradient_bitwise_all_zero"] is False
    for key in (
        "all_recovery_second_pass_gradients_bitwise_zero",
        "all_demo_analytic_vjp_gradients_bitwise_zero",
    ):
        decision = _decision(**{key: False})
        assert decision["status"] == "ABORT_POSITIVE_ADVANTAGE_OBJECTIVE"
        assert decision["passed"] is False
        assert decision["reference_gate_passed"] is False


def test_exact_vector_sha_hashes_all_values_not_only_norm():
    left = torch.tensor([1.0, 2.0, 3.0])
    right = torch.tensor([1.0, 3.0, 2.0])
    assert torch.linalg.vector_norm(left) == torch.linalg.vector_norm(right)
    assert audit.exact_float32_vector_sha256(left) != (
        audit.exact_float32_vector_sha256(right)
    )
    assert audit.exact_float32_vector_sha256(left) == (
        audit.exact_float32_vector_sha256(left.clone())
    )


def test_zero_pa_direction_is_scientific_abort_not_invalid():
    evidence = audit.benefit_cosine_evidence(torch.ones(5), torch.zeros(5))
    assert evidence["defined"] is False
    assert evidence["value"] is None
    assert evidence["gate_value"] == 0.0
    decision = _decision(endpoint_cosines=[evidence, _cosine(), _cosine()])
    assert decision["passed"] is False
    assert decision["status"] == "ABORT_POSITIVE_ADVANTAGE_OBJECTIVE"
    assert decision["primary_endpoints"][0]["benefit_cosine_defined"] is False


def test_gate_passes_only_all_three_endpoints_no_catastrophe_and_refs():
    passed = _decision()
    assert passed["passed"] is True
    assert passed["status"] == (
        "PASS_TO_SEPARATE_64_UPDATE_PA_INELIGIBLE_PILOT_FREEZE"
    )
    assert all(row["passed"] for row in passed["primary_endpoints"])
    assert not any(
        row["catastrophic"]
        for row in passed["production_adamw_increment_catastrophes"]
    )

    ucb_zero = _decision(endpoint_bounds=_bounds(upper=(-0.1, 0.0, -0.1)))
    assert ucb_zero["passed"] is False
    wrong_way = _decision(increment_cosines=[_cosine(), _cosine(-0.011), _cosine()])
    assert wrong_way["passed"] is False
    assert wrong_way["production_adamw_increment_catastrophes"][1][
        "catastrophic"
    ] is True
    lcb_positive = _decision(
        increment_bounds=_bounds(lower=(-1.0, 1e-12, -1.0))
    )
    assert lcb_positive["passed"] is False
    assert _decision(all_reference_relative_bounds_passed=False)["passed"] is False
    assert _decision(
        all_recovery_second_pass_gradients_bitwise_zero=False
    )["status"] == "ABORT_POSITIVE_ADVANTAGE_OBJECTIVE"
    assert _decision(
        all_demo_analytic_vjp_gradients_bitwise_zero=False
    )["status"] == "ABORT_POSITIVE_ADVANTAGE_OBJECTIVE"


def test_thresholds_are_exactly_inherited_from_direct_v2():
    decision = _decision()
    inherited = decision["threshold_inheritance"]
    assert inherited["minimum_endpoint_benefit_cosine"] == 0.01
    assert inherited["maximum_catastrophic_wrong_way_benefit_cosine"] == -0.01
    assert inherited["reference_gradient_relative_bound"] == 1e-6
    assert inherited["confidence"] == 0.95
    assert inherited["bootstrap_samples"] == 10_000
    assert inherited["bootstrap_seed"] == 49_666


def test_fixed_bootstrap_and_panel_receipts_remain_immutable():
    keys = [
        f"{suite}/task={task:02d}"
        for suite in ("libero_goal", "libero_long", "libero_object", "libero_spatial")
        for task in range(10)
    ]
    _matrix, receipt = direct_v2.make_suite_stratified_resample_matrix(keys)
    assert receipt["sha256"] == audit.EXPECTED_BOOTSTRAP_MATRIX_SHA256
    assert receipt["samples"] == 10_000
    if (ROOT / "runs/outcome_recovery_s49666_validation/manifest.json").is_file():
        cfg = read_config(CONFIG)
        panel, _root = direct_v2.pre_reward_panel_receipt(cfg)
        assert panel["sha256"] == audit.EXPECTED_PANEL_GROUP_RECEIPT_SHA256
        counts = [sum(int(row["rank"]) == rank for row in panel["ordered_rows"])
                  for rank in range(8)]
        assert counts == [6] * 8


def test_output_validation_and_exclusive_publish(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    directory = tmp_path / audit.OUTPUT_DIR_REL
    output = directory / f"{audit.OUTPUT_NAME_PREFIX}123.json"
    assert audit._validate_output_path(output) == output.resolve()
    audit.exclusive_json_write(output.resolve(), {"status": "diagnostic"})
    assert json.loads(output.read_text()) == {"status": "diagnostic"}
    with pytest.raises(audit.PositiveAdvantageDirectionAuditError,
                       match="existing"):
        audit._validate_output_path(output)


def test_launcher_is_fixed_eight_gpu_no_update_and_has_postconditions():
    path = ROOT / "scripts/outcome_positive_advantage_direction_audit.sbatch"
    source = path.read_text(encoding="utf-8")
    assert "#SBATCH --time=00:30:00" in source
    assert "#SBATCH --nodes=1" in source
    assert "#SBATCH --gpus-per-node=8" in source
    assert "#SBATCH --ntasks-per-node=8" in source
    assert "configs/r0a_outcome_positive_advantage_audit.yaml" in source
    assert audit.TRIGGER_REPORT_REL in source
    assert audit.TRIGGER_REPORT_SHA256 in source
    assert audit.OUTPUT_NAME_PREFIX in source
    assert "format_version=2" in source
    assert 'assert report["format_version"] == 2' in source
    assert '"job_id": 32580600' in source
    assert '"INVALID_NO_SCIENTIFIC_EVIDENCE"' in source
    assert 'anchor_construction["sparse_ce_graph_constructed"] is False' in source
    assert "two_objective_graph_exact_B1_passes" in source
    assert 'assert report["no_mutation"]["live_optimizer_steps"] == 0' in source
    assert 'if report["status"].startswith("PASS_TO_"):' in source
    assert 'all_three_recovery_second_pass_gradients_bitwise_zero"] is True' in source
    assert 'all_three_demo_analytic_vjp_gradients_bitwise_zero"] is True' in source
    assert 'assert report["decision"]["passed"] is False' in source
    assert "sbatch " not in "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )


def test_source_closure_contains_only_new_receipt_script_and_launcher():
    identity = audit._source_identity()
    assert identity["positive_advantage_core"]["trainer_wired"] is False
    assert identity["positive_advantage_core"]["candidate_or_evaluation_authority"] is False
    assert list(identity["diagnostic"]["files"]) == list(audit._AUDIT_SOURCE_FILES)
    assert identity["diagnostic"]["sha256"]
    assert identity["diagnostic"]["scheme"] == read_config(CONFIG)[
        "outcome_positive_advantage_audit"
    ]["source_provenance"]["own_diagnostic_closure"]["scheme"]
    assert set(audit._AUDIT_SOURCE_FILES) == {
        "configs/r0a_outcome_positive_advantage_audit.yaml",
        "scripts/outcome_positive_advantage_direction_audit.py",
        "scripts/outcome_positive_advantage_direction_audit.sbatch",
    }


def test_report_source_has_no_tuning_training_or_candidate_surface():
    source = inspect.getsource(audit)
    assert "alpha_sweep" not in source
    assert "weighted_alpha_gradient" not in source
    assert "proposal_sparse_ce_loss" not in source
    assert "optimizer.step()" not in source
    assert "save_checkpoint" not in source
    assert "candidate_emitted\": True" not in source
    assert "full_training_eligible\": True" not in source
    assert "official_evaluation_eligible\": True" not in source
    run_source = inspect.getsource(audit.run_audit)
    assert "PA panel six-slot setup" in run_source
    assert "PA offset {offset} retained-gradient clone" in run_source
    assert "rank-zero PA report publication" in run_source
    assert '"cross_rank_exact_direction_identity"' in run_source
    assert '"exact_float32_vector_sha256": direction_vector_sha256' in run_source
    assert "recovery second objective-graph pass failed exact identity" not in run_source
    assert run_source.index("authenticate_trigger_once(") < run_source.index(
        "direct_v2.pre_reward_panel_receipt(cfg)"
    )
