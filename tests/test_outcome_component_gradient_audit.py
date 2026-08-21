"""CPU contracts for the read-only eight-A100 component-gradient audit."""

from __future__ import annotations

import copy
import inspect
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

import contracts as C
from loom.eval import outcome_recovery as recovery
from loom.train import outcome_grpo as grpo
from scripts import outcome_component_gradient_audit as audit


class VectorProposal(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor([0.25, -0.5, 1.0]))


class FixedLogitProposal(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.operator_logits = nn.Parameter(torch.linspace(-2.0, 2.0, C.M))

    def logits(self, z: Tensor, lang: Tensor) -> Tensor:  # noqa: ARG002
        return self.operator_logits.unsqueeze(0).expand(z.shape[0], -1)


class FakeValidationCollection:
    split = "validation"
    identity_digest = "validation-identity"

    def __init__(self, n: int = 100) -> None:
        self.receipts = tuple({
            "n_replans_by_arm": [10] * recovery.GROUP_SIZE,
            "group_id": f"validation-{index}",
        } for index in range(n))

    def informative_indices(self) -> tuple[int, ...]:
        return tuple(range(len(self.receipts)))


class FakeClosureCollection:
    _resolved_sidecar = staticmethod(
        grpo.ValidatedRecoveryCollection._resolved_sidecar
    )

    def __init__(self, root: Path, split: str, count: int) -> None:
        self.root = root / split
        self.root.joinpath("groups").mkdir(parents=True)
        self.split = split
        rows = []
        for index in range(count):
            relative = f"groups/group-{index}.pt"
            path = self.root / relative
            content = f"{split}:{index}".encode()
            path.write_bytes(content)
            rows.append({
                "group_id": f"{split}-{index}",
                "sidecar": relative,
                "sha256": recovery.sha256_file(path),
                "size": len(content),
            })
        self.receipts = tuple(rows)


def _identity_payload(proposal: FixedLogitProposal, *, n: int = 3) -> dict:
    proposal.eval()
    z = torch.arange(n, dtype=torch.float32).reshape(n, 1, 1)
    lang = torch.zeros(1, 1, dtype=torch.float32)
    with torch.no_grad():
        old, logits = grpo.stored_order_logprob(
            proposal,
            z,
            lang,
            proposal.operator_logits.topk(C.TOPK).indices.reshape(1, -1).expand(
                n, -1,
            ),
        )
        order = logits.topk(C.TOPK, dim=-1).indices
        # Recompute after deriving the exact stored order for clarity.
        old, logits = grpo.stored_order_logprob(proposal, z, lang, order)
        coeff = grpo.weights_from_logits(logits, order, C.M)
    return {
        "group_id": "cpu-fixed",
        "arms": [{
            "z": z.clone(),
            "lang": lang.clone(),
            "ordered_support": order.clone(),
            "old_logprob": old.clone(),
            "coeff": coeff.clone(),
            "terminal_reward": torch.tensor(float(arm == 0)),
        } for arm in range(recovery.GROUP_SIZE)],
    }


def _point(fold: int, value: float) -> dict:
    return {
        "fold": fold,
        "gradients": {
            "components": {
                name: {"weighted_gradient_norm": value + index}
                for index, name in enumerate(audit.COMPONENT_ORDER)
            },
            "pairwise_cosines": {
                "grpo__sparse_q_action_ce": -value,
                "grpo__switch_balance_1e-2": value / 2.0,
                "sparse_q_action_ce__switch_balance_1e-2": 0.0,
            },
            "combined": {
                "direct_preclip_norm": value + 3.0,
                "would_clip": value + 3.0 > 1.0,
            },
        },
    }


def test_fixed_geometry_covers_all_six_folds_with_384_distinct_group_slots():
    steps = audit.audit_steps()
    assert audit.EXPECTED_WORLD_SIZE == grpo.EXPECTED_WORLD_SIZE == 8
    assert audit.AUDIT_STEPS_PER_FOLD == 8
    assert len(steps) == 48
    assert len(set(steps)) == len(steps)
    assert audit.EXPECTED_SELECTED_GROUPS_PER_FOLD == 64
    assert audit.EXPECTED_SELECTED_GROUPS == 384
    for fold in range(grpo.N_FOLDS):
        expected = tuple(
            grpo.START_STEP + fold * grpo.UPDATES_PER_FOLD + offset
            for offset in range(audit.AUDIT_STEPS_PER_FOLD)
        )
        begin = fold * audit.AUDIT_STEPS_PER_FOLD
        assert steps[begin:begin + audit.AUDIT_STEPS_PER_FOLD] == expected


def test_component_gradients_are_weighted_global_vectors_before_norms_and_cosines():
    proposal = VectorProposal()
    value = proposal.value
    grpo_direction = torch.tensor([1.0, 2.0, 3.0])
    anchor_direction = torch.tensor([-2.0, 1.0, 0.5])
    raw_balance_direction = torch.tensor([4.0, -3.0, 2.0])
    losses = {
        "grpo": (value * grpo_direction).sum(),
        "sparse_q_action_ce": (value * anchor_direction).sum(),
        "switch_balance_1e-2": 1e-2 * (value * raw_balance_direction).sum(),
    }
    before = {name: tensor.detach().clone()
              for name, tensor in proposal.state_dict().items()}
    report, vectors = audit.measure_synchronised_component_gradients(
        proposal, losses, world=1,
    )

    weighted_balance = 1e-2 * raw_balance_direction
    expected = grpo_direction + anchor_direction + weighted_balance
    assert set(vectors) == {
        *audit.COMPONENT_ORDER, audit.ALGEBRAIC_TOTAL, audit.DIRECT_TOTAL,
    }
    assert torch.allclose(vectors[audit.DIRECT_TOTAL], expected)
    assert torch.allclose(vectors[audit.ALGEBRAIC_TOTAL], expected)
    assert torch.allclose(vectors["grpo"], grpo_direction)
    assert torch.allclose(vectors["sparse_q_action_ce"], anchor_direction)
    assert torch.allclose(vectors["switch_balance_1e-2"], weighted_balance)
    assert report["synchronization"] == "all_reduce_sum_then_divide_world_size"
    assert report["components"]["grpo"]["weighted_gradient_norm"] == pytest.approx(
        float(grpo_direction.norm())
    )
    assert report["components"]["sparse_q_action_ce"][
        "weighted_gradient_norm"
    ] == pytest.approx(float(anchor_direction.norm()))
    assert report["components"]["switch_balance_1e-2"][
        "weighted_gradient_norm"
    ] == pytest.approx(float(weighted_balance.norm()))
    assert report["combined"]["direct_preclip_norm"] == pytest.approx(
        float(expected.norm())
    )
    assert report["combined"]["algebraic_relative_residual"] < 1e-7
    assert report["combined"][
        "finite_precision_additivity_max_relative_residual"
    ] == audit.MAX_COMPONENT_ADDITIVITY_RELATIVE_RESIDUAL == 0.02
    assert report["combined"]["authoritative_preclip_vector"] == "direct_combined"
    assert report["pairwise_cosines"]["grpo__sparse_q_action_ce"] == pytest.approx(
        torch.nn.functional.cosine_similarity(
            grpo_direction.unsqueeze(0), anchor_direction.unsqueeze(0),
        ).item()
    )
    assert proposal.value.grad is None
    assert all(torch.equal(before[name], tensor)
               for name, tensor in proposal.state_dict().items())


def test_observed_scale_bf16_nonassociativity_passes_but_gross_mismatch_fails(
    monkeypatch,
):
    def run_with_relative_residual(relative_residual: float):
        proposal = VectorProposal()
        losses = {
            name: proposal.value.sum() * (index + 1.0)
            for index, name in enumerate(audit.COMPONENT_ORDER)
        }
        # Three aligned unit component vectors have norm-sum 3.  Offset the
        # independent direct vector to synthesize a controlled relative
        # finite-precision residual without needing a CUDA bf16 kernel.
        component = torch.tensor([1.0, 0.0, 0.0])
        sequence = iter([
            component.clone(), component.clone(), component.clone(),
            torch.tensor([3.0 * (1.0 + relative_residual), 0.0, 0.0]),
        ])

        def fake_local_gradient(loss, named_parameters, *, retain_graph):  # noqa: ARG001
            return next(sequence), []

        monkeypatch.setattr(audit, "_local_gradient_vector", fake_local_gradient)
        return audit.measure_synchronised_component_gradients(
            proposal, losses, world=1,
        )

    report, _direct = run_with_relative_residual(0.00427229)
    assert report["combined"]["algebraic_relative_residual"] == pytest.approx(
        0.00427229, rel=2e-5,
    )
    assert report["combined"]["direct_preclip_norm"] > report["combined"][
        "algebraic_preclip_norm"
    ]

    with pytest.raises(audit.ComponentGradientAuditError,
                       match="direct combined gradient differs"):
        run_with_relative_residual(0.03)


def test_bounded_heldout_directional_derivative_is_analytic_and_no_mutation():
    proposal = VectorProposal()
    before = proposal.value.detach().clone()
    heldout_gradient = torch.tensor([1.0, -2.0, 0.5])
    heldout_loss = (proposal.value * heldout_gradient).sum()
    training_gradients = {
        "grpo": torch.tensor([1.0, 0.0, 0.0]),
        "sparse_q_action_ce": torch.tensor([1.0, 4.0, 0.0]),
        "switch_balance_1e-2": torch.tensor([1.0, 0.0, 0.0]),
    }
    training_gradients[audit.ALGEBRAIC_TOTAL] = sum(
        training_gradients[name] for name in audit.COMPONENT_ORDER
    )
    training_gradients[audit.DIRECT_TOTAL] = torch.tensor([3.0, 4.0, 0.0])

    report = audit.measure_heldout_directional_derivative(
        proposal,
        heldout_loss,
        training_gradients,
        world=1,
    )

    # Norm-5 training gradient is clipped to direction (-0.6, -0.8, 0).
    expected_derivative = float(heldout_gradient @ torch.tensor([-0.6, -0.8, 0.0]))
    assert report["method"] == "analytic_gradient_dot_bounded_direction"
    assert report["heldout_objective"] == "validation_negative_clipped_surrogate_loss"
    assert report["sign_convention"] == (
        "d_loss/d_epsilon < 0 predicts heldout surrogate benefit; "
        "d_loss/d_epsilon > 0 predicts heldout surrogate harm"
    )
    assert report["direction_l2_norm"] == pytest.approx(1.0)
    assert report["direction_bound"] == grpo.GRAD_CLIP == 1.0
    assert report["d_heldout_loss_d_epsilon"] == pytest.approx(expected_derivative)
    assert report["predicted_first_order_heldout_loss_delta"] == pytest.approx(
        audit.DIRECTIONAL_EPSILON * expected_derivative
    )
    assert report["interpretation"] == "immediate_first_order_harm"
    attribution = report["gradient_space_attribution"]
    assert attribution["scope"] == "loss_gradient_space_only"
    assert "not virtual AdamW parameter updates" in attribution["scope_caveat"]
    assert attribution["component_vectors_are_recipe_weighted"] is True
    assert attribution[
        "no_additional_forward_or_backward_for_attribution"
    ] is True
    components = attribution["recipe_weighted_components"]
    assert components["grpo"][
        "heldout_gradient_dot_training_gradient"
    ] == pytest.approx(1.0)
    assert components["grpo"][
        "d_heldout_loss_d_epsilon"
    ] == pytest.approx(-1.0)
    assert components["grpo"]["interpretation"] == (
        "immediate_first_order_benefit"
    )
    assert components["sparse_q_action_ce"][
        "heldout_gradient_dot_training_gradient"
    ] == pytest.approx(-7.0)
    assert components["sparse_q_action_ce"][
        "d_heldout_loss_d_epsilon"
    ] == pytest.approx(7.0 / math.sqrt(17.0))
    assert components["sparse_q_action_ce"]["interpretation"] == (
        "immediate_first_order_harm"
    )
    direct = attribution[audit.DIRECT_TOTAL]
    algebraic = attribution[audit.ALGEBRAIC_TOTAL]
    assert direct["heldout_gradient_dot_training_gradient"] == pytest.approx(-5.0)
    assert direct["d_heldout_loss_d_epsilon"] == pytest.approx(1.0)
    assert direct == algebraic
    discrepancy = attribution["current_direct_vs_algebraic"]
    assert discrepancy["direct_is_authoritative"] is True
    assert discrepancy["direct_minus_algebraic_gradient_norm"] == 0.0
    assert discrepancy["heldout_dot_cauchy_abs_bound"] == 0.0
    assert discrepancy["cauchy_bound_passed"] is True

    sweep = attribution["counterfactual_anchor_scale_sweep"]
    assert sweep["fixed_anchor_scales"] == list(audit.ANCHOR_SCALE_SWEEP)
    assert [row["anchor_scale"] for row in sweep["rows"]] == list(
        audit.ANCHOR_SCALE_SWEEP
    )
    for row in sweep["rows"]:
        alpha = row["anchor_scale"]
        gradient = torch.tensor([2.0 + alpha, 4.0 * alpha, 0.0])
        norm = float(gradient.norm())
        dot = float(heldout_gradient @ gradient)
        scale = min(1.0, grpo.GRAD_CLIP / norm)
        assert row["training_gradient_preclip_norm"] == pytest.approx(norm)
        assert row[
            "heldout_gradient_dot_training_gradient"
        ] == pytest.approx(dot)
        assert row["d_heldout_loss_d_epsilon"] == pytest.approx(-dot * scale)
    assert sweep["rows"][0]["interpretation"] == (
        "immediate_first_order_benefit"
    )
    assert sweep["rows"][-1]["interpretation"] == (
        "immediate_first_order_harm"
    )

    directional_summary = audit.summarise_directional_attributions([report])
    assert directional_summary["scope"] == "loss_gradient_space_only"
    assert directional_summary["recipe_weighted_components"]["grpo"][
        "immediate_first_order_benefit_count"
    ] == 1
    assert directional_summary["counterfactual_anchor_scale_sweep"][0][
        "anchor_scale"
    ] == 0.0
    assert report["parameter_perturbations"] == 0
    assert report["optimizer_steps"] == 0
    assert proposal.value.grad is None
    assert torch.equal(proposal.value.detach(), before)


def test_directional_scalar_rows_cover_unclipped_zero_and_reject_nonfinite():
    unclipped = audit._directional_row_from_norm_and_dot(
        training_gradient_norm=0.5,
        heldout_dot_training_gradient=2.0,
        heldout_vs_training_gradient_cosine=0.25,
        epsilon=1e-6,
    )
    assert unclipped["reference_global_clip_scale"] == 1.0
    assert unclipped["negative_clipped_direction_l2_norm"] == 0.5
    assert unclipped["d_heldout_loss_d_epsilon"] == -2.0
    assert unclipped["interpretation"] == "immediate_first_order_benefit"

    zero = audit._directional_row_from_norm_and_dot(
        training_gradient_norm=0.0,
        heldout_dot_training_gradient=0.0,
        heldout_vs_training_gradient_cosine=None,
        epsilon=1e-6,
    )
    assert zero["reference_global_clip_scale"] == 1.0
    assert zero["negative_clipped_direction_l2_norm"] == 0.0
    assert zero["d_heldout_loss_d_epsilon"] == 0.0
    assert zero["interpretation"] == "first_order_flat"

    with pytest.raises(audit.ComponentGradientAuditError, match="nonfinite"):
        audit._directional_row_from_norm_and_dot(
            training_gradient_norm=1.0,
            heldout_dot_training_gradient=float("nan"),
            heldout_vs_training_gradient_cosine=0.0,
            epsilon=1e-6,
        )


def test_heldout_attribution_retains_direct_algebraic_residual_evidence():
    proposal = VectorProposal()
    heldout_gradient = torch.tensor([2.0, -1.0, 0.5])
    heldout_loss = (proposal.value * heldout_gradient).sum()
    components = {
        "grpo": torch.tensor([1.0, 0.0, 0.0]),
        "sparse_q_action_ce": torch.tensor([0.0, 2.0, 0.0]),
        "switch_balance_1e-2": torch.tensor([0.0, 0.0, 0.25]),
    }
    algebraic = sum(components[name] for name in audit.COMPONENT_ORDER)
    residual = torch.tensor([0.1, -0.2, 0.3])
    report = audit.measure_heldout_directional_derivative(
        proposal,
        heldout_loss,
        {
            **components,
            audit.ALGEBRAIC_TOTAL: algebraic,
            audit.DIRECT_TOTAL: algebraic + residual,
        },
        world=1,
    )

    attribution = report["gradient_space_attribution"]
    evidence = attribution["current_direct_vs_algebraic"]
    expected_dot = float(heldout_gradient @ residual)
    expected_bound = float(heldout_gradient.norm() * residual.norm())
    assert evidence["direct_minus_algebraic_gradient_norm"] == pytest.approx(
        float(residual.norm())
    )
    assert evidence[
        "heldout_dot_direct_minus_algebraic_gradient"
    ] == pytest.approx(expected_dot)
    assert evidence["heldout_dot_cauchy_abs_bound"] == pytest.approx(
        expected_bound
    )
    assert abs(expected_dot) <= evidence["heldout_dot_cauchy_abs_bound"]
    assert evidence["cauchy_bound_passed"] is True
    assert evidence[
        "clipped_derivative_difference_direct_minus_algebraic"
    ] == pytest.approx(
        attribution[audit.DIRECT_TOTAL]["d_heldout_loss_d_epsilon"]
        - attribution[audit.ALGEBRAIC_TOTAL]["d_heldout_loss_d_epsilon"]
    )


def test_heldout_attribution_requires_every_synchronised_training_vector():
    proposal = VectorProposal()
    heldout_loss = proposal.value.sum()
    with pytest.raises(grpo.OutcomeGRPOError,
                       match="training gradient vector set changed"):
        audit.measure_heldout_directional_derivative(
            proposal,
            heldout_loss,
            {audit.DIRECT_TOTAL: torch.ones(3)},
            world=1,
        )


def test_heldout_selection_is_deterministic_disjoint_and_has_fixed_contexts():
    collection = FakeValidationCollection()
    first: list[int] = []
    again: list[int] = []
    for fold in range(grpo.N_FOLDS):
        for rank in range(audit.EXPECTED_WORLD_SIZE):
            group, replans = audit.heldout_group_selection(
                collection,
                fold=fold,
                rank=rank,
                world=audit.EXPECTED_WORLD_SIZE,
                seed=grpo.TRAIN_SEED,
                contexts_per_arm=grpo.EXPECTED_CONTEXTS_PER_ARM,
            )
            first.append(group)
            assert set(replans) == set(range(1, recovery.GROUP_SIZE))
            assert all(len(values) == 2 and len(set(values)) == 2
                       for values in replans.values())
            repeated, repeated_replans = audit.heldout_group_selection(
                collection,
                fold=fold,
                rank=rank,
                world=audit.EXPECTED_WORLD_SIZE,
                seed=grpo.TRAIN_SEED,
                contexts_per_arm=grpo.EXPECTED_CONTEXTS_PER_ARM,
            )
            again.append(repeated)
            assert replans == repeated_replans
    assert first == again
    assert len(first) == len(set(first)) == 48


def test_post_use_closure_rehashes_every_rank_selected_sidecar(tmp_path: Path):
    collections = [
        FakeClosureCollection(tmp_path, f"train{fold}", 8)
        for fold in range(grpo.N_FOLDS)
    ]
    validation = FakeClosureCollection(tmp_path, "validation", grpo.N_FOLDS)
    selections = []
    for fold, collection in enumerate(collections):
        for group_index, receipt in enumerate(collection.receipts):
            selections.append({
                "rank": 0,
                "fold": fold,
                "split": collection.split,
                "group_index": group_index,
                "group_id": receipt["group_id"],
                "sidecar": receipt["sidecar"],
                "sidecar_sha256": receipt["sha256"],
                "sidecar_size": receipt["size"],
            })
    for group_index, receipt in enumerate(validation.receipts):
        selections.append({
            "rank": 0,
            "fold_check": group_index,
            "split": "validation",
            "group_index": group_index,
            "group_id": receipt["group_id"],
            "sidecar": receipt["sidecar"],
            "sidecar_sha256": receipt["sha256"],
            "sidecar_size": receipt["size"],
        })

    report = audit.recheck_rank_selected_sidecars(
        collections, validation, selections, rank=0,
    )
    assert report["selected_sidecars"] == 54
    assert report["post_use_size_sha256_and_stable_stat"] is True
    assert len(report["post_use_closure_sha256"]) == 64

    changed = collections[0].root / collections[0].receipts[0]["sidecar"]
    changed.write_bytes(b"changed")
    with pytest.raises(audit.ComponentGradientAuditError,
                       match="size changed after use"):
        audit.recheck_rank_selected_sidecars(
            collections, validation, selections, rank=0,
        )


def test_selected_rows_require_bitwise_seed_logprob_and_coefficient_identity():
    device = torch.device("cpu")
    grpo._configure_exact_proposal_scoring(device)
    proposal = FixedLogitProposal().eval()
    payload = _identity_payload(proposal)
    indices = {arm: (0, 2) for arm in range(1, recovery.GROUP_SIZE)}

    report = audit.authenticate_selected_contexts(
        proposal, payload, indices, device=device,
    )
    assert report == {
        "atoms": 14,
        "max_abs_old_logprob_error": 0.0,
        "max_abs_coeff_error": 0.0,
        "all_exact": True,
    }

    tampered = copy.deepcopy(payload)
    tampered["arms"][1]["old_logprob"][0].add_(1e-3)
    with pytest.raises(audit.ComponentGradientAuditError,
                       match="old-logprob replay differs"):
        audit.authenticate_selected_contexts(
            proposal, tampered, indices, device=device,
        )


def test_exact_b1_flags_are_explicit_and_fail_closed():
    exact = {
        "proposal_scoring_batch_size": 1.0,
        "proposal_scoring_autocast": 0.0,
        "proposal_scoring_cuda_matmul_tf32": 0.0,
        "proposal_scoring_cudnn_tf32": 0.0,
    }
    assert audit.require_exact_b1_metric_flags(exact) == {
        "proposal_scoring_batch_size": 1,
        "proposal_scoring_autocast": False,
        "proposal_scoring_cuda_matmul_tf32": False,
        "proposal_scoring_cudnn_tf32": False,
    }
    with pytest.raises(audit.ComponentGradientAuditError,
                       match="did not report exact B1"):
        audit.require_exact_b1_metric_flags({
            **exact, "proposal_scoring_batch_size": 32.0,
        })


def test_anchor_batch_digest_binds_all_beliefs_targets_language_and_body():
    prepared = (
        [torch.tensor([[float(index)]]) for index in range(C.DEPTH)],
        torch.tensor([[1.0, 2.0]]),
        [torch.tensor([[float(index), 1.0]]) for index in range(C.DEPTH)],
        "libero_franka",
    )
    first = audit.prepared_anchor_digest(prepared)
    second = audit.prepared_anchor_digest(prepared)
    assert first == second
    assert first["n_tensors"] == 1 + 2 * C.DEPTH
    assert first["embodiment"] == "libero_franka"
    changed = copy.deepcopy(prepared)
    changed[2][0][0, 0].add_(1.0)
    assert audit.prepared_anchor_digest(changed)["sha256"] != first["sha256"]


def test_summary_keeps_fold_structure_and_negative_cosine_fraction():
    points = [_point(fold, 0.1 + fold) for fold in range(grpo.N_FOLDS)]
    summary = audit.summarise_points(points)
    assert summary["overall"]["n_synchronized_points"] == grpo.N_FOLDS
    assert set(summary["by_fold"]) == {str(fold) for fold in range(grpo.N_FOLDS)}
    assert all(row["n_synchronized_points"] == 1
               for row in summary["by_fold"].values())
    cosine = summary["overall"]["pairwise_cosines"][
        "grpo__sparse_q_action_ce"
    ]
    assert cosine["n"] == grpo.N_FOLDS
    assert cosine["negative_fraction"] == 1.0


def test_recipe_source_and_ineligibility_are_fail_closed():
    cfg, config_hash = audit._validate_config(
        Path("configs/r0a_outcome_grpo.yaml").resolve()
    )
    assert config_hash == audit.EXPECTED_CONFIG_HASH
    assert cfg["outcome_grpo"]["seed_global_step"] == grpo.START_STEP == 49_666
    identity = audit._source_identity()
    assert identity["trainer"]["sha256"] == audit.EXPECTED_TRAINER_SOURCE_SHA256
    assert set(identity["audit"]["files"]) == set(audit._AUDIT_SOURCE_FILES)
    assert audit.FORMAT_VERSION == 2
    assert audit.ANCHOR_SCALE_SWEEP == (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
    assert audit.STATUS == "DIAGNOSTIC_COMPLETE"
    assert audit.ELIGIBILITY == {
        "diagnostic_only": True,
        "training_eligible": False,
        "candidate_eligible": False,
        "evaluation_eligible": False,
        "promotion_eligible": False,
        "optimizer_updates_allowed": False,
        "optimizer_steps": 0,
        "parameter_perturbations": 0,
        "checkpoint_emitted": False,
        "candidate_emitted": False,
        "reason": (
            "component-gradient measurement cannot select a recipe, checkpoint, "
            "candidate, or SR result"
        ),
    }


def test_output_is_confined_and_exclusive_publish_never_replaces(tmp_path: Path):
    allowed = (
        audit.ROOT / audit.OUTPUT_DIR_REL
        / f"{audit.OUTPUT_NAME_PREFIX}unused_cpu_contract_report.json"
    ).resolve()
    assert not allowed.exists()
    assert audit._validate_output_path(allowed) == allowed
    with pytest.raises(audit.ComponentGradientAuditError,
                       match="must be directly inside"):
        audit._validate_output_path(tmp_path / "wrong-place.json")
    with pytest.raises(audit.ComponentGradientAuditError, match="one JSON file"):
        audit._validate_output_path(allowed.with_suffix(".txt"))
    with pytest.raises(audit.ComponentGradientAuditError,
                       match="format-v2 diagnostic output"):
        audit._validate_output_path(
            allowed.with_name("outcome_component_gradient_audit_s49666_old.json")
        )

    target = tmp_path / "exclusive.json"
    audit.exclusive_json_write(target, {"first": 1})
    original = target.read_bytes()
    with pytest.raises(audit.ComponentGradientAuditError,
                       match="refusing existing diagnostic output"):
        audit.exclusive_json_write(target, {"second": 2})
    assert target.read_bytes() == original


def test_nondeterminism_warning_fails_even_in_single_rank_contract():
    with pytest.raises(grpo.OutcomeGRPOError, match="nondeterminism warnings"):
        audit._checked_warning_messages(
            [SimpleNamespace(message="operation is nondeterministic")],
            world=1,
            label="cpu-test",
        )


def test_python_entry_point_is_read_only_and_exposes_no_tuning_flags():
    source = Path(audit.__file__).read_text(encoding="utf-8")
    run_source = inspect.getsource(audit.run_audit)
    measure_source = inspect.getsource(
        audit.measure_synchronised_component_gradients
    )
    directional_source = inspect.getsource(
        audit.measure_heldout_directional_derivative
    )

    assert "._load_parent_from_identity(" in run_source
    assert "_authenticate_parent_once(" in run_source
    assert "recovery.authenticate_seed_checkpoint(" in inspect.getsource(
        audit._authenticate_parent_once
    )
    assert "collection.load(" in run_source
    assert "validation.load(" in run_source
    assert "ExpertAnchor.from_parent(" in run_source
    assert "_configure_strict_outcome_determinism(" in run_source
    assert "_configure_exact_proposal_scoring(" in run_source
    assert "torch.autograd.grad(" in inspect.getsource(audit._local_gradient_vector)
    assert measure_source.index("_local_gradient_vector(") < measure_source.index(
        "_synchronise_gradient("
    )
    sync_source = inspect.getsource(audit._synchronise_gradient)
    assert sync_source.index("all_reduce(") < sync_source.index("div_(float(world))")
    assert "gradient_space_attribution" in directional_source
    assert "_counterfactual_anchor_scale_rows(" in directional_source
    assert "no additional forward, backward, update" in directional_source
    assert "parameter" not in directional_source.split("never perturbed", 1)[-1] \
        or "copy_(" not in directional_source
    assert ".step(" not in source
    assert "write_descendant_checkpoint" not in source
    assert "eval_libero" not in source and "python -m loom.eval" not in source
    assert "destroy_process_group()" in inspect.getsource(audit.main)
    assert "torch.distributed.barrier()" in run_source
    actions = {action.dest for action in audit.build_parser()._actions}
    assert actions == {"help", "config", "out"}


def test_launcher_requests_one_eight_a100_node_and_only_runs_diagnostic():
    launcher = Path("scripts/outcome_component_gradient_audit.sbatch").read_text(
        encoding="utf-8",
    )
    assert "#SBATCH --nodes=1" in launcher
    assert "#SBATCH --gpus-per-node=8" in launcher
    assert "#SBATCH --ntasks-per-node=8" in launcher
    assert "scripts/outcome_component_gradient_audit.py" in launcher
    assert "runs/diagnostics/outcome_component_gradient_audit" in launcher
    assert "outcome_component_gradient_audit_v2_s49666_" in launcher
    assert "format_version=2" in launcher
    assert "export RANK=\"$SLURM_PROCID\"" in launcher
    assert "WORLD_SIZE=\"$SLURM_NTASKS\"" in launcher
    assert "LOCAL_RANK=\"$SLURM_LOCALID\"" in launcher
    assert "export CUBLAS_WORKSPACE_CONFIG=:4096:8" in launcher
    assert "optimizer_steps=0" in launcher
    assert "candidate_eligible=false" in launcher
    assert "evaluation_eligible=false" in launcher
    assert "heldout_component_attribution=true" in launcher
    assert "anchor_scales=0,.1,.25,.5,.75,1" in launcher
    assert 'report["format_version"] == 2' in launcher
    assert "selected_sidecar_post_use_closure" in launcher
    assert "scripts/train_outcome_grpo.py" not in launcher
    assert "scripts/eval_libero" not in launcher
    assert "scancel" not in launcher
