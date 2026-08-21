"""Focused contracts for the isolated, ineligible outcome-GRPO v2 core."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import pytest
import torch
from torch import Tensor, nn

import contracts as C
from loom.eval import outcome_recovery as recovery
from loom.heads.proposal import pl_log_prob
from loom.train import outcome_grpo_v2 as v2


class TinyProposal(nn.Module):
    def __init__(self, m: int = C.M):
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(-0.3, 0.3, m))

    def logits(self, z: Tensor, lang: Tensor) -> Tensor:  # noqa: ARG002
        signal = z.float().mean(dim=(-1, -2), keepdim=False).unsqueeze(-1)
        return signal * self.weight.unsqueeze(0) + self.weight.unsqueeze(0)


class FakeAnchor:
    def __init__(self, proposal: nn.Module):
        self.proposal = proposal
        self.temperature = 1.0
        self.device = torch.device("cpu")
        self.estimator = nn.Identity()
        self.q_action = nn.Identity()

    def _prepare(self, step: int):  # noqa: ARG002
        batch = 3
        beliefs = [
            torch.full((batch, 2, 2), 0.1 * (horizon + 1))
            for horizon in range(C.DEPTH)
        ]
        lang = torch.zeros(batch, 1, 2)
        targets = []
        for horizon in range(C.DEPTH):
            target = torch.zeros(batch, C.M)
            target[:, horizon % C.M] = 1.0
            targets.append(target)
        return beliefs, lang, targets, "libero_franka"

    def unexpected_gradients(self):
        return []


def _sample_payload(proposal: nn.Module, n: int = 3) -> dict:
    z = torch.arange(n * 4, dtype=torch.float32).reshape(n, 2, 2) / 10.0
    lang = torch.zeros(1, 2)
    with torch.no_grad():
        logits = proposal.logits(z, lang.unsqueeze(0).expand(n, -1, -1))
        order = logits.topk(C.TOPK, dim=-1).indices
        old = pl_log_prob(logits, order)
    arms = [{
        "z": z.clone(),
        "lang": lang.clone(),
        "ordered_support": order.clone(),
        "old_logprob": old.clone(),
        "terminal_reward": torch.tensor(float(arm % 2)),
    } for arm in range(recovery.GROUP_SIZE)]
    return {"group_id": "tiny", "arms": arms}


def test_v2_identity_and_checked_in_config_are_deliberately_unlaunchable(tmp_path):
    from loom.train.loop import read_config

    cfg = read_config("configs/r0a_outcome_grpo_v2_pilot.yaml")
    report = v2.validate_scaffold_config(cfg)
    assert report["method_status"] == v2.METHOD_STATUS_SCAFFOLD
    assert report["launchable"] is False
    assert report["candidate_emission"] is False
    assert any("losses.proposal.weight" == item for item in report["unresolved"])
    assert not any("formal_terminal_collection" in item for item in report["unresolved"])
    run_dir = tmp_path / "must-not-exist"
    with pytest.raises(v2.UnfrozenPilotError):
        v2.train_outcome_grpo_v2(config=cfg, run_dir=run_dir)
    assert not run_dir.exists()
    assert v2.PROPOSAL_LR_SCALE == pytest.approx(0.0125)
    assert v2.PILOT_SNAPSHOT_STEPS == (49_866, 50_066, 50_266, 50_466)


def test_config_declares_exposed_validation_and_forbids_pilot_artifacts():
    from loom.train.loop import read_config

    cfg = read_config("configs/r0a_outcome_grpo_v2_pilot.yaml")
    outcome = cfg["outcome_grpo_v2"]
    lineage = outcome["validation_lineage"]
    assert lineage["current_development_collection"]["status"] == (
        "DEVELOPMENT_EXPOSED_DO_NOT_FORMAL_GATE"
    )
    assert lineage["current_development_collection"]["collection_format_version"] == 1
    assert lineage["current_development_collection"]["manifest_sha256"] == (
        v2.EXPOSED_VALIDATION_MANIFEST_SHA256
    )
    assert len(lineage["current_development_collection"]["observed_identity_digest"]) == 64
    assert lineage["current_development_collection"]["identity_status"] == (
        "OBSERVED_DEVELOPMENT_ONLY_NOT_FORMAL_FROZEN"
    )
    assert lineage["formal_terminal_collection"]["path"] is None
    assert lineage["formal_terminal_collection"]["collection_format_version"] is None
    assert lineage["formal_terminal_collection"]["identity_digest"] is None
    assert "round_robin_direction_audit" in (
        lineage["current_development_collection"]["exposures"]
    )
    assert outcome["artifact_policy"] == {
        "role": "development_pilot",
        "candidate_emission": "forbidden",
        "promotion": "forbidden",
        "official_evaluation": "forbidden",
        "pilot_checkpoint_only": True,
    }
    authenticated = outcome["authenticated_data_lineage"]
    assert authenticated["training"] == [
        dict(row) for row in v2.EXPECTED_AUTHENTICATED_TRAIN_LINEAGE
    ]
    assert authenticated["exposed_development"]["manifest_sha256"] == (
        v2.EXPOSED_VALIDATION_MANIFEST_SHA256
    )


def test_authenticated_data_lineage_rejects_every_pin_mutation():
    from loom.train.loop import read_config

    cfg = read_config("configs/r0a_outcome_grpo_v2_pilot.yaml")
    mutations = (
        ("training_manifest", lambda value: value["training"][0].__setitem__(
            "manifest_sha256", "0" * 64,
        )),
        ("training_identity", lambda value: value["training"][5].__setitem__(
            "identity_digest", "1" * 64,
        )),
        ("development_manifest", lambda value: value[
            "exposed_development"
        ].__setitem__("manifest_sha256", "2" * 64)),
        ("collector_source", lambda value: value["collector_source"].__setitem__(
            "sha256", "3" * 64,
        )),
    )
    for _name, mutate in mutations:
        changed = copy.deepcopy(cfg)
        mutate(changed["outcome_grpo_v2"]["authenticated_data_lineage"])
        with pytest.raises(
            v2.OutcomeGRPOV2Error,
            match="authenticated TRAIN/development manifest lineage differs",
        ):
            v2.validate_scaffold_config(changed)
    changed_current = copy.deepcopy(cfg)
    changed_current["outcome_grpo_v2"]["validation_lineage"][
        "current_development_collection"
    ]["manifest_sha256"] = "4" * 64
    with pytest.raises(v2.OutcomeGRPOV2Error, match="identity/version"):
        v2.validate_scaffold_config(changed_current)


def test_authenticated_lineage_is_computed_from_manifest_bytes_and_identity():
    rows = [
        *v2.EXPECTED_AUTHENTICATED_TRAIN_LINEAGE,
        {
            "split": "validation",
            "path": "runs/outcome_recovery_s49666_validation",
            "manifest_sha256": v2.EXPOSED_VALIDATION_MANIFEST_SHA256,
            "identity_digest": v2.EXPOSED_VALIDATION_IDENTITY_DIGEST,
        },
    ]
    for row in rows:
        manifest_path = Path(row["path"]) / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        assert v2.recovery.sha256_file(manifest_path) == row["manifest_sha256"]
        assert v2.recovery.identity_digest(manifest["identity"]) == (
            row["identity_digest"]
        )
        assert manifest["identity"]["source"] == v2.OUTCOME_COLLECTOR_SOURCE


def _synthetically_frozen_config(anchor_weight: float) -> dict:
    """Resolve placeholders for readiness-unit tests, never for a real launch."""
    from loom.train.loop import read_config

    cfg = read_config("configs/r0a_outcome_grpo_v2_pilot.yaml")
    cfg["losses"]["proposal"]["weight"] = anchor_weight
    cfg["losses"]["demo_reference"]["weight"] = 0.1
    outcome = cfg["outcome_grpo_v2"]
    outcome["method_status"] = v2.METHOD_STATUS_FROZEN_PILOT
    outcome["recovery_kl_controller"].update({
        "initial_beta": 0.1,
        "target_kl": 0.01,
        "eta": 0.2,
        "max_beta": 2.0,
    })
    outcome["train_trust_panel"].update({
        "manifest_sha256": "a" * 64,
        "item_count": 48,
    })
    outcome["train_trust_panel"]["thresholds"] = {
        "recovery_forward_kl_max": 0.01,
        "clip_fraction_max": 0.15,
        "ess_fraction_min": 0.9,
        "arm0_coeff_drift_p95_l2_max": 0.04,
        "arm0_topk_overlap_change_min": -0.05,
        "max_abs_logratio_max": 1.4,
        "demo_categorical_forward_kl_max": 0.01,
        "demo_topk_overlap_change_min": -0.05,
    }
    outcome["freeze_evidence"] = {
        "component_projection_report_sha256": "b" * 64,
        "round_robin_direction_audit_report_sha256": "f" * 64,
        "controller_resume_smoke_report_sha256": "c" * 64,
        "frozen_recipe_sha256": "d" * 64,
    }
    outcome["validation_lineage"]["decision_status"] = (
        "PILOT_EXPOSED_DEVELOPMENT_ONLY_FORMAL_NOT_COLLECTED"
    )
    return cfg


def _synthetically_formal_config(anchor_weight: float = 0.0) -> dict:
    cfg = _synthetically_frozen_config(anchor_weight)
    cfg["run"]["name"] = "r0a_outcome_grpo_v2_formal"
    outcome = cfg["outcome_grpo_v2"]
    outcome["method_status"] = v2.METHOD_STATUS_FROZEN_FORMAL
    outcome["validation_lineage"]["decision_status"] = (
        "FROZEN_FRESH_FORMAL_COLLECTION"
    )
    outcome["validation_lineage"]["formal_terminal_collection"] = {
        "split": "formal_validation_fresh",
        "path": "runs/fresh_formal_validation",
        "collector_seed": 3,
        "collection_format_version": recovery.FORMAT_VERSION,
        "manifest_sha256": "3" * 64,
        "identity_digest": "e" * 64,
        "collector_source": copy.deepcopy(v2.OUTCOME_COLLECTOR_SOURCE),
        "identity_status": "FROZEN_AUTHENTICATED_FORMAL_UNEXPOSED",
    }
    outcome["validation_lineage"]["formalization"] = {
        "pilot_terminal_report_sha256": "1" * 64,
        "method_freeze_receipt_sha256": "2" * 64,
        "method_freeze_receipt_created_utc": "2026-08-20T10:00:00Z",
        "receipt_frozen_recipe_sha256": "d" * 64,
        "formal_collection_started_utc": "2026-08-20T11:00:00Z",
        "chronology_status": "VERIFIED_METHOD_FREEZE_PRECEDES_SEED3_COLLECTION",
    }
    outcome["artifact_policy"] = {
        "role": "formal_candidate_training",
        "candidate_emission": "terminal_gates_only",
        "promotion": "external_official_seed0_gate_only",
        "official_evaluation": "candidate_only",
        "pilot_checkpoint_only": False,
    }
    return cfg


def test_alpha_zero_disables_only_sparse_term_and_is_readiness_valid():
    cfg = _synthetically_frozen_config(anchor_weight=0.0)
    report = v2.require_launchable_config(cfg)
    assert report["launchable"] is True
    proposal = cfg["losses"]["proposal"]
    assert proposal["enabled"] is True
    assert proposal["zero_weight_semantics"] == (
        "sparse_anchor_term_disabled_anchor_states_retained_for_demo_reference"
    )
    negative = _synthetically_frozen_config(anchor_weight=-0.1)
    with pytest.raises(v2.OutcomeGRPOV2Error, match="must be nonnegative"):
        v2.require_launchable_config(negative)


def test_pilot_readiness_forbids_formal_collection_and_formal_requires_chronology():
    pilot = _synthetically_frozen_config(anchor_weight=0.0)
    pilot_report = v2.require_launchable_config(pilot)
    assert pilot_report["artifact_role"] == "development_pilot"
    assert pilot_report["formal_terminal_collection_present"] is False
    assert pilot_report["candidate_emission"] is False

    leaked = copy.deepcopy(pilot)
    leaked["outcome_grpo_v2"]["validation_lineage"][
        "formal_terminal_collection"
    ]["path"] = "runs/too-early-formal"
    with pytest.raises(v2.OutcomeGRPOV2Error, match="must not pin or inspect"):
        v2.require_launchable_config(leaked)

    formal = _synthetically_formal_config()
    formal_report = v2.require_launchable_config(formal)
    assert formal_report["artifact_role"] == "formal_candidate_training"
    assert formal_report["formal_terminal_collection_present"] is True
    assert formal_report["candidate_emission"] is True

    bad_formal_manifest = copy.deepcopy(formal)
    bad_formal_manifest["outcome_grpo_v2"]["validation_lineage"][
        "formal_terminal_collection"
    ]["manifest_sha256"] = None
    with pytest.raises(v2.UnfrozenPilotError) as manifest_caught:
        v2.require_launchable_config(bad_formal_manifest)
    assert any("manifest_sha256" in item for item in manifest_caught.value.unresolved)

    bad_formal_source = copy.deepcopy(formal)
    bad_formal_source["outcome_grpo_v2"]["validation_lineage"][
        "formal_terminal_collection"
    ]["collector_source"]["sha256"] = None
    with pytest.raises(v2.UnfrozenPilotError) as source_caught:
        v2.require_launchable_config(bad_formal_source)
    assert any("collector_source" in item for item in source_caught.value.unresolved)

    bad_order = copy.deepcopy(formal)
    bad_order["outcome_grpo_v2"]["validation_lineage"]["formalization"][
        "formal_collection_started_utc"
    ] = "2026-08-20T09:00:00Z"
    with pytest.raises(v2.UnfrozenPilotError) as caught:
        v2.require_launchable_config(bad_order)
    assert any("chronology_status" in item for item in caught.value.unresolved)


def test_round_robin_v3_matches_formula_is_rank_disjoint_and_resume_pure():
    groups = [[100 * fold + index for index in range(11)] for fold in range(6)]
    samplers = [v2.RoundRobinOutcomeSamplerV3(
        groups,
        seed=7,
        rank=rank,
        world_size=8,
        total_updates=1_200,
        identity_digests=[f"fold-{fold}" for fold in range(6)],
    ) for rank in range(8)]
    for update in (0, 1, 199, 799, 1_199):
        rows = [sampler.group_at(v2.START_STEP + update) for sampler in samplers]
        for rank, (fold, _group, _visit) in enumerate(rows):
            draw = update * 8 + rank
            assert fold == draw % 6
        for fold in range(6):
            selected = [group for got_fold, group, _ in rows if got_fold == fold]
            assert len(selected) == len(set(selected))
    fold_draws = [0] * 6
    for update in range(1_200):
        for sampler in samplers:
            fold_draws[sampler.group_at(v2.START_STEP + update)[0]] += 1
    assert fold_draws == [1_600] * 6

    state = samplers[0].state_dict(v2.START_STEP + 477)
    assert state["kind"] == v2.SAMPLER_KIND
    resumed_rank5 = v2.RoundRobinOutcomeSamplerV3(
        groups,
        seed=7,
        rank=5,
        world_size=8,
        total_updates=1_200,
        identity_digests=[f"fold-{fold}" for fold in range(6)],
    )
    resumed_rank5.validate_state_dict(state, v2.START_STEP + 477)
    assert resumed_rank5.group_at(v2.START_STEP + 477) == samplers[5].group_at(
        v2.START_STEP + 477,
    )
    # The proposed t=0/1/2 seed-direction audit can replay both the exact
    # per-rank group and exact stored-replan selection without sampler state.
    replan_counts = [5] * recovery.GROUP_SIZE
    replay_rank5 = v2.RoundRobinOutcomeSamplerV3(
        groups,
        seed=7,
        rank=5,
        world_size=8,
        total_updates=1_200,
        identity_digests=[f"fold-{fold}" for fold in range(6)],
    )
    for update in range(3):
        step = v2.START_STEP + update
        assert replay_rank5.group_at(step) == samplers[5].group_at(step)
        assert replay_rank5.replans_at(step, replan_counts) == (
            samplers[5].replans_at(step, replan_counts)
        )


def test_round_robin_v3_800_update_pilot_is_globally_balanced():
    groups = [[index for index in range(101)] for _ in range(6)]
    counts = [0] * 6
    for rank in range(8):
        sampler = v2.RoundRobinOutcomeSamplerV3(
            groups, seed=0, rank=rank, world_size=8,
        )
        for update in range(v2.PILOT_UPDATES):
            counts[sampler.group_at(v2.START_STEP + update)[0]] += 1
    assert max(counts) - min(counts) == 1
    assert sum(counts) == 6_400


def test_recovery_forward_kl_is_zero_value_and_gradient_at_identity():
    current = [torch.zeros(4, requires_grad=True) for _ in range(7)]
    old = [torch.zeros(4) for _ in range(7)]
    loss = v2.recovery_pl_forward_kl(current, old)
    assert float(loss) == 0.0
    loss.backward()
    assert all(torch.equal(value.grad, torch.zeros_like(value)) for value in current)
    shifted = [torch.full((4,), 0.1, requires_grad=True) for _ in range(7)]
    got = v2.recovery_pl_forward_kl(shifted, old)
    assert float(got) == pytest.approx(torch.expm1(torch.tensor(0.1)).item() - 0.1)
    with pytest.raises(ValueError, match="arms 1..7 only"):
        v2.recovery_pl_forward_kl([torch.zeros(1)] * 8, [torch.zeros(1)] * 8)


def test_sampled_v2_objective_reuses_ratios_and_excludes_arm0_reference():
    proposal = TinyProposal()
    proposal.eval()
    result = v2.sampled_group_objectives_v2(
        proposal,
        _sample_payload(proposal),
        {arm: (0, 2) for arm in range(1, 8)},
        device=torch.device("cpu"),
    )
    assert result.grpo.requires_grad
    assert result.balance.requires_grad
    assert result.recovery_forward_kl.requires_grad
    assert float(result.recovery_forward_kl) == 0.0
    assert result.metrics["ratio_atoms"] == 14
    assert result.metrics["arm0_in_recovery_reference"] == 0.0
    assert result.metrics["recovery_ratio_arms_min"] == 1.0
    assert result.metrics["recovery_ratio_arms_max"] == 7.0
    with pytest.raises(ValueError, match="exactly arms 1..7"):
        v2.sampled_group_objectives_v2(
            proposal,
            _sample_payload(proposal),
            {arm: (0,) for arm in range(8)},
            device=torch.device("cpu"),
        )


def test_dense_demo_reference_is_fp32_forward_kl_with_identity_zero_gradient():
    seed = torch.tensor([[1.0, -0.5, 0.2]], dtype=torch.float32)
    current = seed.clone().requires_grad_(True)
    loss = v2.dense_categorical_forward_kl(current, seed)
    assert loss.dtype == torch.float32
    assert float(loss) == 0.0
    loss.backward()
    # The analytic gradient is zero. A single fp32 softmax reduction can leave
    # one unit-roundoff residual; it is far below a bf16 parameter update.
    assert torch.allclose(current.grad, torch.zeros_like(current), atol=5e-8, rtol=0)
    changed = (seed + torch.tensor([[0.0, 1.0, -1.0]])).requires_grad_(True)
    assert float(v2.dense_categorical_forward_kl(changed, seed)) > 0.0


def test_demo_anchor_reuses_live_logits_and_never_gradients_frozen_seed():
    live = TinyProposal().eval()
    seed = copy.deepcopy(live).eval().requires_grad_(False)
    wrapper = v2.DemoReferenceAnchorV2(
        anchor=FakeAnchor(live),
        seed_proposal=seed,
        seed_digest=v2.v1.proposal_module_digest(seed.state_dict()),
    )
    sparse, reference, metrics = wrapper.losses(v2.START_STEP)
    beliefs, lang, targets, _ = wrapper.anchor._prepare(v2.START_STEP)
    inherited_sparse = torch.stack([
        v2.v1.proposal_sparse_ce_loss(
            live, beliefs[horizon], lang, targets[horizon],
            temperature=1.0, detach_belief=True,
        )
        for horizon in range(C.DEPTH)
    ]).mean()
    assert torch.equal(sparse, inherited_sparse)
    assert sparse.requires_grad and reference.requires_grad
    assert float(reference) == 0.0
    (sparse + reference).backward()
    assert live.weight.grad is not None
    assert seed.weight.grad is None
    assert wrapper.unexpected_gradients() == []
    provenance = wrapper.provenance()
    assert provenance["current_logits_reused_with_sparse_ce"] is True
    assert provenance["live_seed_identical_input_and_autocast"] is True
    assert provenance["ce_kl_probability_math"] == "float32"
    assert provenance["seed_in_optimizer"] is False
    assert provenance["seed_in_training_checkpoint"] is False


def test_one_sided_controller_is_deterministic_monotone_and_checkpoint_complete():
    controller = v2.OneSidedRecoveryKLController(
        initial_beta=0.2, target_kl=0.01, eta=0.5, max_beta=2.0, interval=4,
    )
    for update, value in enumerate([0.005] * 4 + [0.02] * 4 + [0.0, 0.01], 1):
        decision = controller.observe(update, value)
        if update == 4:
            assert decision is not None
            assert decision["new_beta"] == decision["old_beta"]
        if update == 8:
            assert decision is not None
            assert decision["new_beta"] > decision["old_beta"]
    state = controller.state_dict()
    resumed = v2.OneSidedRecoveryKLController.from_state_dict(state)
    assert resumed.state_dict() == state
    for update, value in ((11, 0.03), (12, 0.03)):
        controller.observe(update, value)
        resumed.observe(update, value)
    assert resumed.state_dict() == controller.state_dict()
    betas = [float(row["new_beta"]) for row in controller.decisions]
    assert betas == sorted(betas)


def _panel() -> v2.TrainOnlyTrustPanel:
    return v2.TrainOnlyTrustPanel(
        manifest_digest="a" * 64,
        item_count=48,
        source_splits=tuple(f"train{index}" for index in range(6)),
        demo_anchor_manifest_digest=v2.v1.EXPECTED_ANCHOR_MANIFEST["digest"],
        demo_anchor_batches=v2.v1.EXPERT_GATE_BATCHES,
        demo_anchor_start_step=v2.START_STEP,
        thresholds=v2.TrainPanelThresholds(
            recovery_forward_kl_max=0.01,
            clip_fraction_max=0.15,
            ess_fraction_min=0.90,
            arm0_coeff_drift_p95_l2_max=0.04,
            arm0_topk_overlap_change_min=-0.05,
            max_abs_logratio_max=1.4,
            demo_categorical_forward_kl_max=0.01,
            demo_topk_overlap_change_min=-0.05,
        ),
    )


def test_train_only_panel_rejects_validation_and_fails_closed(
    tmp_path: Path, monkeypatch,
):
    with pytest.raises(ValueError, match="TRAIN splits only"):
        v2.TrainOnlyTrustPanel(
            manifest_digest="b" * 64,
            item_count=1,
            source_splits=("validation",),
            demo_anchor_manifest_digest=v2.v1.EXPECTED_ANCHOR_MANIFEST["digest"],
            demo_anchor_batches=v2.v1.EXPERT_GATE_BATCHES,
            demo_anchor_start_step=v2.START_STEP,
            thresholds=_panel().thresholds,
        )
    panel = _panel()
    passing = {
        "recovery_forward_kl": 0.005,
        "clip_fraction": 0.1,
        "ratio_ess_fraction": 0.95,
        "arm0_coeff_drift_p95_l2": 0.02,
        "arm0_topk_overlap_change": -0.01,
        "max_abs_logratio": 1.0,
        "demo_categorical_forward_kl": 0.005,
        "demo_topk_overlap_change": -0.01,
    }
    report = panel.evaluate(200, passing, provenance=panel.provenance())
    assert report["passed"] and report["online_validation_used"] is False
    assert report["provenance"]["demo_panel"] == {
        "source": "frozen_authenticated_train_anchor_batches",
        "anchor_manifest_digest": v2.v1.EXPECTED_ANCHOR_MANIFEST["digest"],
        "batches": v2.v1.EXPERT_GATE_BATCHES,
        "start_step": v2.START_STEP,
        "selection_locked_before_training": True,
        "uses_validation": False,
        "uses_holdout": False,
    }
    resumed = v2.TrainOnlyTrustPanel.from_state_dict(panel.state_dict())
    assert resumed.state_dict() == panel.state_dict()
    failing = dict(passing, clip_fraction=0.2)
    with pytest.raises(v2.TrainTrustPanelViolation) as caught:
        panel.evaluate(400, failing, provenance=panel.provenance())
    assert caught.value.report["candidate_emitted"] is False
    out = v2.persist_pilot_failure_no_candidate(tmp_path, caught.value.report)
    payload = json.loads(out.read_text())
    assert payload["candidate_emitted"] is False
    assert payload["official_evaluation_eligible"] is False
    assert not tuple(tmp_path.glob("candidate_*.pt"))

    demo_bad = dict(passing, demo_categorical_forward_kl=0.02)
    demo_panel = _panel()
    with pytest.raises(v2.TrainTrustPanelViolation) as demo_caught:
        demo_panel.evaluate(200, demo_bad, provenance=demo_panel.provenance())
    assert demo_caught.value.report["checks"][
        "demo_categorical_forward_kl"
    ]["pass"] is False

    arm0_bad = dict(passing, arm0_topk_overlap_change=-0.06)
    arm0_panel = _panel()
    with pytest.raises(v2.TrainTrustPanelViolation) as arm0_caught:
        arm0_panel.evaluate(200, arm0_bad, provenance=arm0_panel.provenance())
    assert arm0_caught.value.report["checks"][
        "arm0_topk_overlap_change"
    ]["pass"] is False

    original = out.read_text()
    with pytest.raises(FileExistsError):
        v2.persist_pilot_failure_no_candidate(tmp_path, caught.value.report)
    assert out.read_text() == original

    race_dir = tmp_path / "race"
    race_dir.mkdir()
    real_link = v2.os.link

    def competing_link(source, destination):
        Path(destination).write_text("competing-writer\n")
        return real_link(source, destination)

    monkeypatch.setattr(v2.os, "link", competing_link)
    with pytest.raises(FileExistsError):
        v2.persist_pilot_failure_no_candidate(race_dir, caught.value.report)
    assert (race_dir / "pilot_failure.json").read_text() == "competing-writer\n"
    assert not tuple(race_dir.glob(".pilot_failure.json.pending-*"))


def test_v2_source_closure_is_distinct_and_detects_mutation(tmp_path: Path):
    source = v2.trainer_source_identity()
    assert source["scheme"] == "sha256(path-nul-sha256-nul)-v1"
    assert "loom/train/outcome_grpo_v2.py" in source["files"]
    assert "loom/train/outcome_grpo.py" in source["files"]
    assert "configs/r0a_outcome_grpo_v2_pilot.yaml" in source["files"]
    v2.assert_trainer_source_identity(source)


def test_v2_cli_has_no_numeric_tuning_flags():
    actions = {action.dest for action in v2.build_parser()._actions}
    assert {"config", "run_dir", "stop_at"} <= actions
    assert not {"lr", "anchor_weight", "beta", "target_kl"} & actions
