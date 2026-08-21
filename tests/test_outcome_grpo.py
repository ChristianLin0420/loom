"""Focused contracts for standalone proposal-only outcome GRPO training."""

from __future__ import annotations

import copy
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

import contracts as C
from loom.eval import outcome_recovery as recovery
from loom.heads.proposal import canonical_order, pl_log_prob, weights_from_logits
from loom.train import outcome_grpo as grpo


class FixedLogits(nn.Module):
    def __init__(self, logits: Tensor):
        super().__init__()
        self.value = nn.Parameter(logits.clone().float())

    def logits(self, z: Tensor, lang: Tensor) -> Tensor:  # noqa: ARG002
        return self.value.unsqueeze(0).expand(z.shape[0], -1)


def test_locked_recipe_constants_and_cli_have_no_tuning_flags():
    assert grpo.START_STEP == 49_666
    assert grpo.STOP_STEP == 54_466
    assert grpo.N_FOLDS == 6
    assert grpo.UPDATES_PER_FOLD == 800
    assert grpo.SCHEDULE_STEPS == 80_000
    assert grpo.CLIP_EPS == 0.2
    assert grpo.MAX_CLIP_FRACTION == 0.2
    assert grpo.MIN_ESS_FRACTION == 0.8
    assert grpo.MAX_COEFF_DRIFT_P95 == 0.05
    assert grpo.MIN_LIVE_OPS == 16
    actions = {action.dest for action in grpo.build_parser()._actions}
    assert {"config", "run_dir", "stop_at"} <= actions
    assert "lr" not in actions and "epochs" not in actions and "clip_eps" not in actions


def test_run_directory_lock_rejects_duplicate_writer_and_releases(tmp_path):
    first = grpo._acquire_run_directory_lock(tmp_path)
    owner = json.loads((tmp_path / ".outcome_grpo.lock").read_text())
    assert owner["pid"] > 0
    with pytest.raises(grpo.OutcomeGRPOError, match="another outcome-GRPO writer"):
        grpo._acquire_run_directory_lock(tmp_path)
    first.close()
    second = grpo._acquire_run_directory_lock(tmp_path)
    second.close()


def test_group_advantages_include_arm0_and_constant_group_is_exact_zero():
    rewards = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    advantage = grpo.normalised_group_advantages(rewards)
    assert float(advantage.mean()) == pytest.approx(0.0, abs=1e-7)
    assert float(advantage.square().mean()) == pytest.approx(1.0, abs=1e-6)
    assert advantage[0] > 0 and bool((advantage[1:] < 0).all())
    assert torch.equal(
        grpo.normalised_group_advantages(torch.ones(8)), torch.zeros(8),
    )
    with pytest.raises(ValueError, match="8 rewards"):
        grpo.normalised_group_advantages(torch.ones(7))


def test_stored_order_pl_never_canonicalises_sampled_atom():
    proposal = FixedLogits(torch.tensor([0.0, 1.0, 2.0, 3.0]))
    proposal.eval()
    z = torch.zeros(1, 1, 1)
    lang = torch.zeros(1, 1, 1)
    sampled_order = torch.tensor([[0, 3]], dtype=torch.int64)
    stored, logits = grpo.stored_order_logprob(proposal, z, lang, sampled_order)
    coeff = weights_from_logits(logits, sampled_order, m=4)
    recovered = canonical_order(coeff, topk=2)
    assert recovered.tolist() == [[3, 0]]
    canonical = pl_log_prob(logits, recovered)
    assert torch.equal(stored, pl_log_prob(logits, sampled_order))
    assert not torch.allclose(stored, canonical)


def test_parent_proposal_loads_in_collector_matching_eval_mode():
    from loom.heads.proposal import Proposal

    kwargs = {
        "dim": 8, "lang_dim": 8, "m": 8, "topk": 4,
        "width": 8, "n_blocks": 1, "n_heads": 1,
    }
    source = Proposal(**kwargs)
    parent = {
        "resolved_config": {"model": {"proposal": kwargs}},
        "model": {
            f"proposal.{name}": value.detach().clone()
            for name, value in source.state_dict().items()
        },
    }
    loaded = grpo._load_proposal(parent, device=torch.device("cpu"))
    assert loaded.training is False
    grpo._require_exact_proposal_scoring_environment(
        loaded, torch.device("cpu"),
    )


def test_clipped_objective_boundaries_are_inclusive_and_sign_correct():
    old = torch.zeros(4)
    ratios = torch.tensor([0.8, 1.2, 0.79, 1.21])
    current = ratios.log()
    positive, got, clipped = grpo.clipped_grpo_objective(current, old, 2.0)
    assert torch.allclose(got, ratios)
    assert clipped.tolist() == [False, False, True, True]
    assert torch.allclose(positive, torch.tensor([1.6, 2.4, 1.58, 2.4]))
    negative, _, _ = grpo.clipped_grpo_objective(current, old, -2.0)
    assert torch.allclose(negative, torch.tensor([-1.6, -2.4, -1.6, -2.42]))


def test_group_loss_means_replans_then_trajectories_not_pooled_atoms():
    rewards = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    lengths = [1, 2, 3, 4, 5, 6, 7]
    current = [torch.full((n,), 0.01 * arm, requires_grad=True)
               for arm, n in enumerate(lengths, 1)]
    old = [torch.zeros_like(value) for value in current]
    loss, ratios = grpo.group_grpo_loss(current, old, rewards)
    advantage = grpo.normalised_group_advantages(rewards)
    manual = []
    for arm, (now, before) in enumerate(zip(current, old, strict=True), 1):
        objective, _, _ = grpo.clipped_grpo_objective(now, before, advantage[arm])
        manual.append(objective.mean())
    assert torch.allclose(loss, -torch.stack(manual).mean())
    assert ratios.numel() == sum(lengths)
    pooled = -torch.cat([
        grpo.clipped_grpo_objective(now, before, advantage[arm])[0]
        for arm, (now, before) in enumerate(zip(current, old, strict=True), 1)
    ]).mean()
    assert not torch.allclose(loss, pooled)
    loss.backward()
    assert all(value.grad is not None for value in current)


def test_deterministic_sampler_is_fold_bounded_rank_disjoint_and_resumable():
    groups = [[10, 11, 12, 13] for _ in range(6)]
    samplers = [grpo.DeterministicOutcomeSampler(
        groups, seed=7, rank=rank, world_size=2, contexts_per_arm=2,
        identity_digests=[str(index) for index in range(6)],
    ) for rank in range(2)]
    first = [sampler.group_at(grpo.START_STEP) for sampler in samplers]
    assert first[0][0] == first[1][0] == 0
    assert first[0][1] != first[1][1]
    assert samplers[0].group_at(grpo.START_STEP + 799)[0] == 0
    assert samplers[0].group_at(grpo.START_STEP + 800)[0] == 1
    counts = [3] * recovery.GROUP_SIZE
    selected = samplers[0].replans_at(grpo.START_STEP, counts)
    assert set(selected) == set(range(1, 8))
    assert {len(value) for value in selected.values()} == {2}
    state = samplers[0].state_dict(grpo.START_STEP + 123)
    resumed = grpo.DeterministicOutcomeSampler(
        groups, seed=7, rank=1, world_size=2, contexts_per_arm=2,
        identity_digests=[str(index) for index in range(6)],
    )
    resumed.validate_state_dict(state, grpo.START_STEP + 123)
    assert resumed.group_at(grpo.START_STEP + 123) == samplers[1].group_at(
        grpo.START_STEP + 123,
    )


def test_switch_balance_uses_dense_logits_and_has_dense_gradient():
    logits = torch.tensor([
        [5.0, 4.0, 3.0, 2.0, -1.0, -2.0],
        [5.0, 4.0, 3.0, 2.0, -1.0, -2.0],
    ], requires_grad=True)
    loss = grpo.proposal_switch_balance(logits, topk=2)
    loss.backward()
    assert float(loss) > 1.0
    assert logits.grad is not None
    assert bool((logits.grad[:, 4:].abs() > 0).all())


def test_sampled_group_loss_uses_equal_arm_counts_and_all_eight_baseline():
    proposal = AnchorProposal()
    proposal.eval()
    n = 3
    z = torch.zeros(n, 1, 1)
    lang = torch.zeros(1, 1)
    with torch.no_grad():
        logits = proposal.logits(z, lang.unsqueeze(0).expand(n, -1, -1))
        order = logits.topk(C.TOPK, dim=-1).indices
        old = pl_log_prob(logits, order)
    arms = [{
        "z": z.clone(), "lang": lang.clone(),
        "ordered_support": order.clone(), "old_logprob": old.clone(),
        "terminal_reward": torch.tensor(float(arm in (0, 2, 4, 6))),
    } for arm in range(8)]
    loss, balance, metrics = grpo.sampled_group_losses(
        proposal, {"group_id": "g", "arms": arms},
        {arm: (0, 2) for arm in range(1, 8)}, device=torch.device("cpu"),
    )
    assert loss.requires_grad and balance.requires_grad
    assert metrics["ratio_atoms"] == 14
    assert metrics["ratio_mean"] == pytest.approx(1.0)
    assert metrics["informative_group"] == 1.0
    with pytest.raises(ValueError, match="exactly arms 1..7"):
        grpo.sampled_group_losses(
            proposal, {"group_id": "g", "arms": arms},
            {arm: (0,) for arm in range(8)}, device=torch.device("cpu"),
        )


def test_resolved_six_fold_config_is_the_locked_recipe():
    from loom.train.loop import read_config

    cfg = read_config("configs/r0a_outcome_grpo.yaml")
    grpo.validate_recipe_config(cfg)
    assert [row["split"] for row in cfg["outcome_grpo"]["folds"]] == [
        f"train{index}" for index in range(6)
    ]
    assert cfg["outcome_grpo"]["anchor_manifest"]["n_trajectories"] == 1960
    assert cfg["outcome_grpo"]["anchor_manifest"]["n_windows"] == 47271
    assert cfg["train_modules"] == ["proposal"]
    auth = cfg["outcome_grpo"]["authentication"]
    assert auth["proposal_scoring_batch_size"] == 1
    assert auth["proposal_scoring_dtype"] == "float32"
    assert auth["proposal_scoring_autocast"] is False
    assert auth["cuda_matmul_tf32"] is False
    assert auth["cudnn_tf32"] is False
    assert auth["float32_matmul_precision"] == "highest"
    assert auth["proposal_scoring_module_mode"] == "eval"
    assert auth["behaviour_logprob_atol"] == grpo.BEHAVIOUR_IDENTITY_MAX_ABS_LOGRATIO
    assert auth["behaviour_logprob_rtol"] == 0.0
    assert auth["behaviour_coeff_atol"] == 0.0
    assert auth["behaviour_coeff_rtol"] == 0.0
    assert auth["identity_max_abs_logratio"] == grpo.BEHAVIOUR_IDENTITY_MAX_ABS_LOGRATIO
    assert auth["identity_max_coeff_error"] == grpo.BEHAVIOUR_IDENTITY_MAX_COEFF_ERROR
    assert auth["initial_ratio_min_ess_fraction"] == grpo.INITIAL_RATIO_MIN_ESS_FRACTION
    assert grpo._config_hash(cfg) == "25afdedfc9deea5e"


class ProvenanceOnly:
    def __init__(self, name: str):
        self.name = name

    def provenance(self):
        return {"identity_digest": self.name, "split": self.name}


def test_trainer_checkpoint_resets_once_then_restores_optimizer_and_step(tmp_path: Path):
    proposal = SmallProposal()
    optimizer = torch.optim.AdamW(proposal.parameters(), lr=3e-4)
    schedule = grpo.CosineWithWarmup(3e-4, 2000, 80_000, 0.05)
    sampler = grpo.DeterministicOutcomeSampler(
        [[0] for _ in range(6)], seed=0, rank=0, world_size=1,
        identity_digests=[str(index) for index in range(6)],
    )
    resolved = {
        "outcome_grpo": {"anchor_manifest": {
            "digest": "d", "n_tasks": 1, "n_trajectories": 1, "n_windows": 1,
        }},
    }
    config_hash = grpo._config_hash(resolved)
    parent = {
        "path": "/seed.pt", "sha256": recovery.SEED_CHECKPOINT_SHA256,
        "global_step": grpo.START_STEP, "config_hash": recovery.SEED_CONFIG_HASH,
    }
    folds = [ProvenanceOnly(f"train{index}") for index in range(6)]
    validation = ProvenanceOnly("validation")
    source = grpo._trainer_source_identity()
    exact_auth = {
        "passed": True, "splits": 7, "all_atoms": 10,
        "max_abs_coeff_error": 0.0,
        "max_abs_old_logprob_error": 0.0,
        "max_abs_logratio": 0.0,
        "max_abs_coeff_error_threshold": grpo.BEHAVIOUR_IDENTITY_MAX_COEFF_ERROR,
        "max_abs_logratio_threshold": grpo.BEHAVIOUR_IDENTITY_MAX_ABS_LOGRATIO,
    }
    start_checkpoint_identity = {
        "checked": True, "passed": True, "global_step": grpo.START_STEP,
        "proposal": grpo.proposal_module_digest(proposal.state_dict()),
        "optimizer_state_entries": 0,
        "optimizer_reset": {"count": 1, "modules": ["proposal"]},
    }
    local_identity = grpo._require_initial_behavior_ratio_identity({
        "max_abs_logratio": 0.0,
        "ratio_min": 1.0,
        "ratio_mean": 1.0,
        "ratio_max": 1.0,
        "clip_fraction": 0.0,
        "ratio_atoms": 14.0,
        "ratio_sum": 14.0,
        "ratio_square_sum": 14.0,
        "ratio_ess_fraction": 1.0,
        "clipped_atoms": 0.0,
    }, device=torch.device("cpu"))
    initial_identity = grpo._assemble_initial_behavior_identity(
        [{"rank": 0, **local_identity}],
        world=1, config_hash=config_hash, trainer_source=source,
        parent_identity=parent, exact_behaviour_identity=exact_auth,
        start_checkpoint_identity=start_checkpoint_identity,
    )
    assert initial_identity["strict_determinism"] == {
        "deterministic_algorithms": True,
        "warn_only": False,
    }
    grpo._save_trainer_checkpoint(
        tmp_path, global_step=grpo.START_STEP, samples_seen=10,
        config_hash=config_hash, resolved_config=resolved,
        parent_identity=parent, collections=folds, validation=validation,
        proposal=proposal, optimizer=optimizer, scheduler=schedule,
        sampler=sampler, wandb_run_id="run",
        exact_behaviour_identity=exact_auth,
        initial_behavior_identity=None,
        trainer_source=source, stop_reason="reset",
    )
    initial = torch.load(
        grpo._trainer_checkpoint_path(tmp_path, grpo.START_STEP),
        map_location="cpu", weights_only=True,
    )
    assert initial["optimizer"]["state"] == {}
    assert initial["optimizer_reset"]["count"] == 1
    proposal.linear.weight.sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    grpo._save_trainer_checkpoint(
        tmp_path, global_step=grpo.START_STEP + 1, samples_seen=20,
        config_hash=config_hash, resolved_config=resolved,
        parent_identity=parent, collections=folds, validation=validation,
        proposal=proposal, optimizer=optimizer, scheduler=schedule,
        sampler=sampler, wandb_run_id="run",
        exact_behaviour_identity=exact_auth,
        initial_behavior_identity=initial_identity,
        trainer_source=source,
    )
    expected = {name: value.detach().clone() for name, value in proposal.state_dict().items()}
    restored = SmallProposal()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=3e-4)
    resumed = grpo._load_trainer_checkpoint(
        tmp_path, config_hash=config_hash, resolved_config=resolved,
        parent_identity=parent, collections=folds, validation=validation,
        proposal=restored, optimizer=restored_optimizer, scheduler=schedule,
        sampler=sampler, exact_behaviour_identity=exact_auth,
        trainer_source=source,
    )
    assert resumed == (grpo.START_STEP + 1, 20, "run", initial_identity)
    assert all(torch.equal(restored.state_dict()[name], value)
               for name, value in expected.items())
    assert len(restored_optimizer.state) == len(optimizer.state) > 0
    checkpoint = grpo._trainer_checkpoint_path(tmp_path, grpo.START_STEP + 1)
    tampered = torch.load(checkpoint, map_location="cpu", weights_only=False)
    tampered["trainer_source"]["sha256"] = "0" * 64
    torch.save(tampered, checkpoint)
    with pytest.raises(grpo.OutcomeGRPOError, match="source closure differs"):
        grpo._load_trainer_checkpoint(
            tmp_path, config_hash=config_hash, resolved_config=resolved,
            parent_identity=parent, collections=folds, validation=validation,
            proposal=restored, optimizer=restored_optimizer, scheduler=schedule,
            sampler=sampler, exact_behaviour_identity=exact_auth,
            trainer_source=source,
        )


def test_model_digests_ignore_only_proposal_and_detect_frozen_byte_change():
    state = {
        "proposal.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
        "estimator.scalar": torch.tensor(3.0),
        "decoder.bytes": torch.tensor([1, 2, 3], dtype=torch.int16),
    }
    frozen = grpo.frozen_model_digest(state)
    changed_proposal = {**state, "proposal.weight": state["proposal.weight"] + 1}
    assert grpo.frozen_model_digest(changed_proposal) == frozen
    changed_frozen = {**state, "estimator.scalar": torch.tensor(4.0)}
    assert grpo.frozen_model_digest(changed_frozen)["sha256"] != frozen["sha256"]
    assert grpo.proposal_model_digest(changed_proposal)["sha256"] \
        != grpo.proposal_model_digest(state)["sha256"]


class AnchorProposal(nn.Module):
    def __init__(self):
        super().__init__()
        self.operator_logits = nn.Parameter(torch.linspace(-1.0, 1.0, C.M))

    def logits(self, z: Tensor, lang: Tensor) -> Tensor:  # noqa: ARG002
        return self.operator_logits.unsqueeze(0).expand(z.shape[0], -1)


class AnchorEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def forward(self, feats, z=None, embodiment=None):  # noqa: ARG002
        value = feats["x"] * self.scale
        return value if z is None else value + 0.01 * z


class AnchorQAction(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def forward(self, actions, z, embodiment=None):  # noqa: ARG002
        target = torch.zeros(actions.shape[0], C.M, device=actions.device)
        target[:, :C.TOPK] = 1.0 / C.TOPK
        return target * self.scale


class AnchorSampler:
    def __init__(self, batch: int = 2):
        self.batch = batch
        self.steps: list[int] = []

    def next(self, step: int):
        self.steps.append(step)
        feats = [{"x": torch.ones(self.batch, 1, 1)} for _ in range(C.DEPTH + 1)]
        return {
            "feats": feats,
            "burn_in_feats": [],
            "lang": torch.zeros(self.batch, 1, 1),
            "actions": torch.zeros(self.batch, C.DEPTH, C.H_OP, 1),
            "embodiment": "libero_franka",
            "data_meta": {
                "source": "libero", "split": "train", "manifest_digest": "test",
            },
        }


def make_anchor(proposal: nn.Module) -> grpo.ExpertAnchor:
    return grpo.ExpertAnchor(
        proposal=proposal,
        estimator=AnchorEstimator(),
        q_action=AnchorQAction(),
        sampler=AnchorSampler(),
        device=torch.device("cpu"),
        temperature=1.0,
        weight=1.0,
        detach_belief=True,
        data_provenance={"trajectory_manifest": {"digest": "test"}},
    )


def test_expert_anchor_is_existing_sparse_ce_and_only_proposal_gets_gradient():
    proposal = AnchorProposal()
    anchor = make_anchor(proposal)
    preflight = anchor.preflight()
    assert preflight["loss"].endswith("proposal_sparse_ce_loss")
    assert anchor.sampler.steps == [49_666]
    loss, metrics = anchor.loss(49_666)
    assert anchor.sampler.steps == [49_666]  # preflight batch was cached
    assert float(loss) == pytest.approx(metrics["sparse_ce"])
    loss.backward()
    assert proposal.operator_logits.grad is not None
    assert anchor.unexpected_gradients() == []
    assert all(parameter.grad is None for parameter in anchor.estimator.parameters())
    assert all(parameter.grad is None for parameter in anchor.q_action.parameters())


def test_expert_anchor_fails_closed_without_actions():
    proposal = AnchorProposal()
    anchor = make_anchor(proposal)
    anchor.sampler.next = lambda step: {  # type: ignore[method-assign]
        "feats": [{"x": torch.ones(1, 1, 1)} for _ in range(C.DEPTH + 1)],
        "lang": torch.zeros(1, 1, 1),
        "actions": None,
        "embodiment": "libero_franka",
        "data_meta": {
            "source": "libero", "split": "train", "manifest_digest": "test",
        },
    }
    with pytest.raises(grpo.ExpertAnchorUnavailable, match="no action targets"):
        anchor.preflight()


class IndexedProposal(nn.Module):
    """Four disjoint live operators for each integer encoded in z[:,0,0]."""

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))

    def logits(self, z: Tensor, lang: Tensor) -> Tensor:  # noqa: ARG002
        batch = z.shape[0]
        out = torch.full((batch, C.M), -10.0, device=z.device)
        base = z[:, 0, 0].long() * C.TOPK
        values = self.scale * torch.tensor([4.0, 3.0, 2.0, 1.0], device=z.device)
        for row in range(batch):
            out[row, base[row]:base[row] + C.TOPK] = values
        return out


class FakeCollection:
    def __init__(self, payload):
        self.payload = payload
        self.receipts = [{}]
        self.split = "train0"

    def load(self, index):
        assert index == 0
        return self.payload

    def assert_unchanged(self):
        return None


class BatchSensitiveProposal(nn.Module):
    """Expose the SDPA failure mode: batch-N logits differ from batch-1."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.seen_batch_sizes: list[int] = []
        self.seen_autocast: list[bool] = []
        self.seen_dtypes: list[torch.dtype] = []

    def logits(self, z: Tensor, lang: Tensor) -> Tensor:  # noqa: ARG002
        self.seen_batch_sizes.append(int(z.shape[0]))
        autocast = torch.is_autocast_enabled(z.device.type)
        self.seen_autocast.append(bool(autocast))
        self.seen_dtypes.append(z.dtype)
        rows = torch.arange(C.M, device=z.device, dtype=z.dtype)
        out = (-rows / 32.0).unsqueeze(0).expand(z.shape[0], -1).clone()
        out = out + self.scale * z[:, :1, :1].reshape(-1, 1) / 100.0
        if z.shape[0] > 1:
            out[:, 0] += 0.1
        if autocast:
            out = out.to(torch.bfloat16)
        return out


def test_stored_order_pl_accumulates_every_collector_atom_at_batch_one(monkeypatch):
    proposal = BatchSensitiveProposal().eval()
    n = 3
    z = torch.arange(n, dtype=torch.float32).reshape(n, 1, 1)
    lang = torch.zeros(1, 1)
    with torch.no_grad():
        row_logits = torch.cat([
            proposal.logits(z[row:row + 1], lang.reshape(1, 1, 1))
            for row in range(n)
        ])
        order = row_logits.topk(C.TOPK, dim=-1).indices
        expected = torch.cat([
            pl_log_prob(row_logits[row:row + 1], order[row:row + 1])
            for row in range(n)
        ])

    real_pl_log_prob = grpo.pl_log_prob
    seen_pl_batches: list[int] = []

    def batch_sensitive_pl(logits: Tensor, ordered_support: Tensor) -> Tensor:
        batch = int(logits.shape[0])
        seen_pl_batches.append(batch)
        score = real_pl_log_prob(logits, ordered_support)
        return score + (0.25 if batch > 1 else 0.0)

    monkeypatch.setattr(grpo, "pl_log_prob", batch_sensitive_pl)
    _clear_scoring_witness(proposal)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual, _ = grpo.stored_order_logprob(proposal, z, lang, order)

    assert torch.equal(actual, expected)
    assert seen_pl_batches == [1] * n
    assert proposal.seen_batch_sizes == [1] * n
    assert proposal.seen_autocast == [False] * n


def test_behaviour_auth_replays_proposal_rowwise_while_transfer_stays_chunked():
    proposal = BatchSensitiveProposal()
    n = 3
    z = torch.arange(n, dtype=torch.float32).reshape(n, 1, 1)
    lang = torch.zeros(1, 1)
    row_logits = torch.cat([
        proposal.logits(z[row:row + 1], lang.reshape(1, 1, 1))
        for row in range(n)
    ])
    order = row_logits.topk(C.TOPK, dim=-1).indices
    coeff = weights_from_logits(row_logits, order, C.M)
    old = pl_log_prob(row_logits, order)
    arms = [{
        "z": z.clone(), "lang": lang.clone(),
        "ordered_support": order.clone(), "coeff": coeff.clone(),
        "old_logprob": old.clone(), "terminal_reward": torch.tensor(float(arm % 2)),
    } for arm in range(recovery.GROUP_SIZE)]
    proposal.seen_batch_sizes.clear()
    proposal.seen_autocast.clear()
    proposal.seen_dtypes.clear()

    with torch.autocast("cpu", dtype=torch.bfloat16):
        report = grpo.authenticate_behaviour_policy(
            proposal, FakeCollection({"group_id": "batch-sensitive", "arms": arms}),
            device=torch.device("cpu"), chunk_replans=n,
        )

    assert report["passed"] is True
    assert report["max_abs_old_logprob_error"] == 0.0
    assert report["max_abs_coeff_error"] == 0.0
    assert report["proposal_replay_batch_size"] == 1
    assert report["transfer_chunk_replans"] == n
    assert report["max_abs_logratio"] == 0.0
    assert report["proposal_scoring"] == {
        "autocast": False,
        "batch_size": 1,
        "cuda_matmul_tf32": False,
        "cudnn_tf32": False,
        "device_type": "cpu",
        "dtype": "float32",
        "float32_matmul_precision": "highest",
        "module_mode": "eval",
        "stored_order": True,
    }
    assert proposal.seen_batch_sizes == [1] * (recovery.GROUP_SIZE * n)
    assert proposal.seen_autocast == [False] * (recovery.GROUP_SIZE * n)
    assert proposal.seen_dtypes == [torch.float32] * (recovery.GROUP_SIZE * n)


def _batch_sensitive_payload(policy: BatchSensitiveProposal, *, n: int = 3) -> dict:
    policy.eval()
    z = torch.arange(n, dtype=torch.float32).reshape(n, 1, 1)
    lang = torch.zeros(1, 1)
    with torch.no_grad():
        logits = torch.cat([
            policy.logits(z[row:row + 1], lang.reshape(1, 1, 1))
            for row in range(n)
        ])
        order = logits.topk(C.TOPK, dim=-1).indices
        coeff = weights_from_logits(logits, order, C.M)
        old = pl_log_prob(logits, order)
    return {
        "group_id": "batch-autocast-sensitive",
        "task": "suite/task=00",
        "arms": [{
            "z": z.clone(), "lang": lang.clone(),
            "ordered_support": order.clone(), "coeff": coeff.clone(),
            "old_logprob": old.clone(),
            "terminal_reward": torch.tensor(float(arm % 2)),
        } for arm in range(recovery.GROUP_SIZE)],
    }


def _clear_scoring_witness(policy: BatchSensitiveProposal) -> None:
    policy.seen_batch_sizes.clear()
    policy.seen_autocast.clear()
    policy.seen_dtypes.clear()


def test_train_validation_and_trust_share_rowwise_fp32_scoring_under_autocast():
    policy = BatchSensitiveProposal()
    payload = _batch_sensitive_payload(policy)

    _clear_scoring_witness(policy)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, balance, metrics = grpo.sampled_group_losses(
            policy, payload, {arm: (0, 2) for arm in range(1, 8)},
            device=torch.device("cpu"),
        )
    assert metrics["max_abs_logratio"] == 0.0
    assert metrics["ratio_min"] == metrics["ratio_mean"] == metrics["ratio_max"] == 1.0
    assert metrics["clip_fraction"] == 0.0
    assert metrics["ratio_sum"] == metrics["ratio_square_sum"] == 14.0
    assert metrics["clipped_atoms"] == 0.0
    assert metrics["ratio_ess_fraction"] == 1.0
    assert policy.seen_batch_sizes == [1] * 14
    assert policy.seen_autocast == [False] * 14
    assert policy.seen_dtypes == [torch.float32] * 14
    (loss + balance).backward()
    assert policy.scale.grad is not None

    collection = FakeCollection(payload)
    _clear_scoring_witness(policy)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        validation = grpo.evaluate_validation_surrogate(
            policy, collection, device=torch.device("cpu"), chunk_replans=3,
        )
    assert validation["max_abs_logratio"] == 0.0
    assert validation["proposal_scoring"]["batch_size"] == 1
    assert policy.seen_batch_sizes == [1] * (7 * 3)
    assert policy.seen_autocast == [False] * (7 * 3)

    _clear_scoring_witness(policy)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        trust = grpo.evaluate_trust_gates(
            policy, collection, device=torch.device("cpu"), chunk_replans=3,
        )
    assert trust["max_abs_logratio"] == 0.0
    assert trust["checks"]["clip_fraction"]["value"] == 0.0
    assert trust["checks"]["coeff_drift_p95"]["value"] == 0.0
    assert trust["proposal_scoring"]["batch_size"] == 1
    # Trust includes arm 0 coefficient drift as well as arms 1..7 ratios.
    assert policy.seen_batch_sizes == [1] * (8 * 3)
    assert policy.seen_autocast == [False] * (8 * 3)


def test_initial_ratio_identity_is_exact_and_checked_before_backward():
    exact = {
        "max_abs_logratio": 0.0,
        "ratio_min": 1.0,
        "ratio_mean": 1.0,
        "ratio_max": 1.0,
        "clip_fraction": 0.0,
        "ratio_atoms": 14.0,
        "ratio_sum": 14.0,
        "ratio_square_sum": 14.0,
        "ratio_ess_fraction": 1.0,
        "clipped_atoms": 0.0,
    }
    evidence = grpo._require_initial_behavior_ratio_identity(
        exact, device=torch.device("cpu"),
    )
    assert evidence["passed"] is True
    assert evidence["max_abs_logratio"] == 0.0
    near = dict(exact)
    delta = 1.5e-5
    ratio_low = math.exp(-delta)
    ratio_high = math.exp(delta)
    ratio_sum = 7.0 * (ratio_low + ratio_high)
    ratio_square_sum = 7.0 * (ratio_low * ratio_low + ratio_high * ratio_high)
    near.update({
        "max_abs_logratio": delta,
        "ratio_min": ratio_low,
        "ratio_mean": ratio_sum / 14.0,
        "ratio_max": ratio_high,
        "ratio_sum": ratio_sum,
        "ratio_square_sum": ratio_square_sum,
        "ratio_ess_fraction": ratio_sum * ratio_sum / (14.0 * ratio_square_sum),
    })
    with pytest.raises(grpo.OutcomeGRPOError, match="before optimizer step"):
        grpo._require_initial_behavior_ratio_identity(
            near, device=torch.device("cpu"),
        )
    for key, value in (
        ("max_abs_logratio", 2.1e-5),
        ("ratio_min", math.exp(-2.1e-5)),
        ("ratio_mean", math.exp(2.1e-5)),
        ("ratio_max", math.exp(2.1e-5)),
        ("clip_fraction", 1e-12),
        ("ratio_ess_fraction", grpo.INITIAL_RATIO_MIN_ESS_FRACTION - 1e-9),
    ):
        bad = dict(exact)
        bad[key] = value
        with pytest.raises(grpo.OutcomeGRPOError, match="before optimizer step"):
            grpo._require_initial_behavior_ratio_identity(
                bad, device=torch.device("cpu"),
            )

    source = Path("loom/train/outcome_grpo.py").read_text(encoding="utf-8")
    loop = source[source.index("while global_step < target_step:"):]
    identity = loop.index("_require_initial_behavior_ratio_identity")
    assert identity < loop.index("scheduler.apply(optimizer, step)")
    assert identity < loop.index("optimizer.zero_grad(set_to_none=True)")
    assert identity < loop.index("total.backward()") < loop.index("optimizer.step()")


def test_ratio_metric_reduction_recomputes_mean_clip_and_ess_from_raw_sums():
    metrics = {
        "ratio_min": 1.0,
        "ratio_max": 2.0,
        "max_abs_logratio": float(torch.tensor(2.0).log()),
        "ratio_sum": 3.0,
        "ratio_square_sum": 5.0,
        "ratio_atoms": 2.0,
        "clipped_atoms": 1.0,
        # Deliberately stale derived values must not survive reduction.
        "ratio_mean": -1.0,
        "clip_fraction": -1.0,
        "ratio_ess_fraction": -1.0,
    }
    reduced = grpo._reduce_training_metrics(
        metrics, world=1, device=torch.device("cpu"),
    )
    assert reduced["ratio_mean"] == 1.5
    assert reduced["clip_fraction"] == 0.5
    assert reduced["ratio_ess_fraction"] == pytest.approx(0.9)
    assert reduced["ratio_sum"] == 3.0
    assert reduced["ratio_square_sum"] == 5.0


def test_exact_deep_auth_and_start_checkpoint_identity_fail_closed():
    report = {
        "all_atoms": 10,
        "max_abs_coeff_error": 0.0,
        "max_abs_old_logprob_error": 0.0,
        "max_abs_logratio": 0.0,
    }
    exact = grpo._require_exact_gathered_behaviour_auth({0: report, 1: report})
    assert exact == {
        "passed": True, "splits": 2, "all_atoms": 20,
        "max_abs_coeff_error": 0.0,
        "max_abs_old_logprob_error": 0.0,
        "max_abs_logratio": 0.0,
        "max_abs_coeff_error_threshold": grpo.BEHAVIOUR_IDENTITY_MAX_COEFF_ERROR,
        "max_abs_logratio_threshold": grpo.BEHAVIOUR_IDENTITY_MAX_ABS_LOGRATIO,
    }
    observed = {
        **report,
        "max_abs_old_logprob_error": 1.52587890625e-5,
        "max_abs_logratio": 1.52587890625e-5,
    }
    with pytest.raises(grpo.OutcomeGRPOError, match="numerical-identity bounds"):
        grpo._require_exact_gathered_behaviour_auth({0: observed})
    for key, value in (
        ("max_abs_coeff_error", grpo.BEHAVIOUR_IDENTITY_MAX_COEFF_ERROR + 1e-9),
        ("max_abs_old_logprob_error", math.nextafter(
            grpo.BEHAVIOUR_IDENTITY_MAX_ABS_LOGRATIO, math.inf,
        )),
        ("max_abs_logratio", math.nextafter(
            grpo.BEHAVIOUR_IDENTITY_MAX_ABS_LOGRATIO, math.inf,
        )),
    ):
        bad = {**report, key: value}
        with pytest.raises(grpo.OutcomeGRPOError, match="numerical-identity bounds"):
            grpo._require_exact_gathered_behaviour_auth({0: bad})

    proposal = AnchorProposal()
    optimizer = torch.optim.AdamW(proposal.parameters(), lr=1e-3, weight_decay=0.0)
    parent_digest = grpo.proposal_module_digest(proposal.state_dict())
    evidence = grpo._require_start_step_checkpoint_identity(
        proposal, optimizer, global_step=grpo.START_STEP,
        parent_proposal=parent_digest,
    )
    assert evidence["passed"] is True
    assert evidence["optimizer_state_entries"] == 0

    changed = AnchorProposal()
    changed.load_state_dict(proposal.state_dict())
    with torch.no_grad():
        changed.operator_logits[0] += 1.0
    with pytest.raises(grpo.OutcomeGRPOError, match="differs from authenticated parent"):
        grpo._require_start_step_checkpoint_identity(
            changed, torch.optim.AdamW(changed.parameters(), lr=1e-3),
            global_step=grpo.START_STEP, parent_proposal=parent_digest,
        )

    # A zero-gradient AdamW step leaves parameters exact when decay is zero,
    # while still creating forbidden START-step optimizer moments.
    proposal.operator_logits.grad = torch.zeros_like(proposal.operator_logits)
    optimizer.step()
    assert grpo.proposal_module_digest(proposal.state_dict()) == parent_digest
    with pytest.raises(grpo.OutcomeGRPOError, match="non-empty proposal optimizer"):
        grpo._require_start_step_checkpoint_identity(
            proposal, optimizer, global_step=grpo.START_STEP,
            parent_proposal=parent_digest,
        )


def test_cuda_tf32_is_explicitly_disabled_before_proposal_load_and_auth():
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        geometry = grpo._configure_exact_proposal_scoring(torch.device("cuda"))
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
        assert geometry["cuda_matmul_tf32"] is False
        assert geometry["cudnn_tf32"] is False
        assert grpo._strict_outcome_determinism_state() == {
            "deterministic_algorithms": True,
            "warn_only": False,
        }
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn
        torch.use_deterministic_algorithms(
            old_deterministic, warn_only=old_warn_only,
        )

    source = Path("loom/train/outcome_grpo.py").read_text(encoding="utf-8")
    body = source[source.index("def train_outcome_grpo("):]
    enabled = body.index("enable_determinism()")
    strict = body.index("_configure_strict_outcome_determinism()")
    configured = body.index("_configure_exact_proposal_scoring(target_device)")
    assert enabled < strict < configured
    assert configured < body.index("_load_proposal(parent, device=target_device)")
    assert configured < body.index("authenticate_behaviour_policy(")


def trust_payload(policy: IndexedProposal) -> dict:
    z = torch.arange(4, dtype=torch.float32).reshape(4, 1, 1)
    lang = torch.zeros(1, 1)
    with torch.no_grad():
        logits = policy.logits(z, lang.unsqueeze(0).expand(4, -1, -1))
        order = logits.topk(C.TOPK, dim=-1).indices
        coeff = weights_from_logits(logits, order, C.M)
        old = pl_log_prob(logits, order)
    arms = []
    for arm in range(recovery.GROUP_SIZE):
        arms.append({
            "z": z.clone(), "lang": lang.clone(),
            "ordered_support": order.clone(), "coeff": coeff.clone(),
            "old_logprob": old.clone(), "terminal_reward": torch.tensor(float(arm % 2)),
        })
    return {"group_id": "test", "arms": arms}


def test_trust_gates_pass_at_unit_ratios_zero_drift_and_sixteen_live_ops():
    policy = IndexedProposal(scale=1.0)
    collection = FakeCollection(trust_payload(policy))
    report = grpo.evaluate_trust_gates(
        policy, collection, device=torch.device("cpu"), chunk_replans=2,
    )
    assert report["passed"] is True
    assert report["checks"]["clip_fraction"]["value"] == 0.0
    assert report["checks"]["ess_fraction"]["value"] == pytest.approx(1.0)
    assert report["checks"]["coeff_drift_p95"]["value"] == 0.0
    assert report["checks"]["live_ops"]["value"] == 16
    assert report["definitions"]["arm0_importance_ratios"] == 0


def test_trust_gates_reject_policy_ratio_and_coeff_drift_and_nonfinite_health():
    old_policy = IndexedProposal(scale=1.0)
    collection = FakeCollection(trust_payload(old_policy))
    new_policy = IndexedProposal(scale=3.0)
    report = grpo.evaluate_trust_gates(
        new_policy, collection, device=torch.device("cpu"), chunk_replans=4,
        training_nonfinite=1,
    )
    assert report["passed"] is False
    assert report["checks"]["clip_fraction"]["pass"] is False
    assert report["checks"]["coeff_drift_p95"]["pass"] is False
    assert report["checks"]["nonfinite"]["pass"] is False


def test_collection_open_streams_one_hash_bound_sidecar_and_rejects_mutation(
    tmp_path: Path, monkeypatch,
):
    root = tmp_path / "fold"
    groups = root / "groups"
    groups.mkdir(parents=True)
    item = SimpleNamespace(name="only")
    monkeypatch.setattr(recovery, "collection_items", lambda split: [item])
    monkeypatch.setattr(recovery, "work_key", lambda value: "only")
    monkeypatch.setattr(recovery, "source_digest", lambda root=None: "a" * 64)
    checkpoint_identity = {"sha256": "b" * 64, "global_step": 49_666}
    identity = {"test": True, "checkpoint": checkpoint_identity}
    monkeypatch.setattr(
        recovery, "collection_identity",
        lambda checkpoint, split, source_sha256: identity,
    )
    monkeypatch.setattr(recovery, "validate_group_payload", lambda *args, **kwargs: None)
    payload = {
        "arms": [
            {"z": torch.zeros(arm + 1, 1),
             "terminal_reward": torch.tensor(float(arm % 2))}
            for arm in range(recovery.GROUP_SIZE)
        ]
    }
    sidecar = groups / "only.pt"
    torch.save(payload, sidecar)
    replans = list(range(1, recovery.GROUP_SIZE + 1))
    rewards = [arm % 2 for arm in range(recovery.GROUP_SIZE)]
    receipt = {
        "group_id": "only", "sidecar": "groups/only.pt",
        "sha256": recovery.sha256_file(sidecar), "size": sidecar.stat().st_size,
        "n_arms": recovery.GROUP_SIZE, "n_replans_by_arm": replans,
        "terminal_rewards": rewards, "worker": {"test": True},
    }
    summary = {
        "status": "COMPLETE", "complete": True,
        "n_groups": 1, "n_expected_groups": 1,
        "n_trajectories": recovery.GROUP_SIZE,
        "n_expected_trajectories": recovery.GROUP_SIZE,
        "terminal_successes_by_arm": rewards,
        "replans_by_arm": replans,
    }
    manifest = {
        "format_version": recovery.FORMAT_VERSION,
        "kind": "loom_outcome_recovery_collection",
        "identity": identity, "identity_digest": recovery.identity_digest(identity),
        "split": "train0", "started_utc": "x", "updated_utc": "y",
        "summary": summary, "groups": [receipt],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    collection = grpo.ValidatedRecoveryCollection.open(
        root, checkpoint_identity=checkpoint_identity, deep=True,
    )
    assert collection.load(0)["arms"][7]["z"].shape[0] == 8
    with sidecar.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(grpo.OutcomeGRPOError, match="sidecar changed"):
        collection.load(0)
    with pytest.raises(grpo.OutcomeGRPOError, match="sidecar changed"):
        collection.assert_all_sidecars_unchanged()


def test_collection_rejects_sidecar_path_escape(tmp_path: Path):
    with pytest.raises(grpo.OutcomeGRPOError, match="escapes"):
        grpo.ValidatedRecoveryCollection._resolved_sidecar(
            tmp_path, {"sidecar": "../outside.pt"},
        )


def test_metadata_reopen_must_match_owner_rank_deep_auth_snapshot():
    collection = SimpleNamespace(
        split="validation", provenance=lambda: {
            "split": "validation", "manifest_sha256": "a" * 64,
        },
    )
    grpo._assert_owner_collection_snapshot(
        collection, {"split": "validation", "manifest_sha256": "a" * 64},
    )
    with pytest.raises(grpo.OutcomeGRPOError, match="owner-rank"):
        grpo._assert_owner_collection_snapshot(
            collection, {"split": "validation", "manifest_sha256": "b" * 64},
        )


class SmallProposal(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)


def test_descendant_changes_only_proposal_and_carries_reset_optimizer(tmp_path: Path):
    proposal = SmallProposal()
    initial = {f"proposal.{name}": value.detach().clone()
               for name, value in proposal.state_dict().items()}
    initial["estimator.frozen"] = torch.arange(8, dtype=torch.float32)
    cfg = {"model": {"proposal": {}}, "data": {"source": "test"}}
    parent = {
        "model": initial,
        "global_step": 49_666,
        "config_hash": grpo._config_hash(cfg),
        "resolved_config": cfg,
        "samples_seen": 123,
    }
    optimizer = torch.optim.AdamW(proposal.parameters(), lr=grpo.BASE_LEARNING_RATE)
    optimizer.zero_grad(set_to_none=True)
    proposal.linear.weight.sum().backward()
    optimizer.step()
    provenance = {
        "collection": {"identity_digest": "c" * 64, "split": "train0"},
        "expert_anchor": {"weight": 1.0, "temperature": 1.0},
    }
    out = tmp_path / "descendant.pt"
    report = grpo.write_descendant_checkpoint(
        out, parent=parent,
        parent_identity={"path": "/parent.pt", "sha256": "d" * 64},
        proposal=proposal, optimizer=optimizer, optimizer_steps=400,
        provenance=provenance,
    )
    saved = torch.load(out, map_location="cpu", weights_only=False)
    assert saved["global_step"] == 50_066 == report["global_step"]
    assert torch.equal(saved["model"]["estimator.frozen"], initial["estimator.frozen"])
    assert not torch.equal(saved["model"]["proposal.linear.weight"],
                           initial["proposal.linear.weight"])
    assert saved["optimizer"]["kind"] == "proposal_only_adamw"
    assert saved["optimizer"]["state_reset_at_entry"] is True
    assert saved["outcome_grpo"]["mutated_model_prefixes"] == ["proposal."]
    assert grpo._config_hash(saved["resolved_config"]) == saved["config_hash"]
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        grpo.write_descendant_checkpoint(
            out, parent=parent,
            parent_identity={"path": "/parent.pt", "sha256": "d" * 64},
            proposal=proposal, optimizer=optimizer, optimizer_steps=400,
            provenance=provenance,
        )


def test_main_returns_distinct_anchor_and_trust_fail_codes(monkeypatch, capsys):
    argv = ["--config", "configs/r0a_outcome_grpo.yaml", "--run-dir", "r"]
    monkeypatch.setattr(
        grpo, "train_outcome_grpo",
        lambda **kwargs: (_ for _ in ()).throw(grpo.ExpertAnchorUnavailable("missing")),
    )
    assert grpo.main(argv) == 3
    assert "EXPERT_ANCHOR_UNAVAILABLE" in capsys.readouterr().out
    report = {"checks": {"ess": {"pass": False}}}
    monkeypatch.setattr(
        grpo, "train_outcome_grpo",
        lambda **kwargs: (_ for _ in ()).throw(grpo.TrustGateError(report)),
    )
    assert grpo.main(argv) == 4
    assert "TRUST_GATE_FAILED" in capsys.readouterr().out


def _set_nested(value, path, replacement):
    cursor = value
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement


@pytest.mark.parametrize(("path", "replacement"), [
    (("run", "seed"), 1),
    (("run", "log_every"), 21),
    (("run", "ckpt_every"), 201),
    (("data", "batch_per_gpu"), 4),
    (("data", "sampling"), "uniform_window"),
    (("data", "recurrent_burn_in"), 3),
    (("optim", "betas"), [0.8, 0.95]),
    (("optim", "weight_decay"), 0.01),
    (("optim", "grad_clip"), 2.0),
    (("optim", "eps"), 1e-7),
    (("losses", "proposal", "temperature"), 0.5),
    (("model", "proposal"), {"hidden": 64}),
    (("outcome_grpo", "contexts_per_arm"), 1),
    (("outcome_grpo", "minimum_informative_groups_per_fold"), 99),
    (("outcome_grpo", "minimum_validation_informative_groups"), 199),
    (("outcome_grpo", "authentication", "proposal_scoring_batch_size"), 2),
    (("outcome_grpo", "authentication", "proposal_scoring_dtype"), "bfloat16"),
    (("outcome_grpo", "authentication", "proposal_scoring_autocast"), True),
    (("outcome_grpo", "authentication", "cuda_matmul_tf32"), True),
    (("outcome_grpo", "authentication", "cudnn_tf32"), True),
    (("outcome_grpo", "authentication", "float32_matmul_precision"), "high"),
    (("outcome_grpo", "authentication", "proposal_scoring_module_mode"), "train"),
    (("outcome_grpo", "authentication", "behaviour_logprob_atol"), 2.0e-5),
    (("outcome_grpo", "authentication", "behaviour_logprob_rtol"), 1.0e-8),
    (("outcome_grpo", "authentication", "behaviour_coeff_atol"), 1.0e-8),
    (("outcome_grpo", "authentication", "behaviour_coeff_rtol"), 1.0e-8),
    (("outcome_grpo", "authentication", "identity_max_abs_logratio"), 2.0e-5),
    (("outcome_grpo", "authentication", "identity_max_coeff_error"), 1.0e-8),
    (("outcome_grpo", "authentication", "initial_ratio_min_ess_fraction"), 0.99),
    (("outcome_grpo", "folds", 0, "path"), "runs/wrong"),
    (("outcome_grpo", "validation", "path"), "runs/wrong"),
    (("validation_gate", "max_clip_fraction"), 0.21),
    (("validation_gate", "expert_batches"), 15),
    (("convergence_gate", "snapshot_steps"), [53_666, 54_466]),
    (("convergence_gate", "terminal_parallelism"), "serial_rank0"),
    (("convergence_gate", "bootstrap", "samples"), 1_999),
    (("convergence_gate", "plateau", "equivalence_high"), 0.02),
    (("convergence_gate", "anchor_sparse_ce",
      "block_median_relative_range_max"), 0.03),
    (("promotion_gate", "candidate_successes_min"), 163),
    (("slurm", "nodes"), 2),
])
def test_every_behavior_changing_recipe_mutation_fails_closed(path, replacement):
    from loom.train.loop import read_config

    cfg = copy.deepcopy(read_config("configs/r0a_outcome_grpo.yaml"))
    _set_nested(cfg, path, replacement)
    with pytest.raises(grpo.OutcomeGRPOError, match="locked recipe|differs"):
        grpo.validate_recipe_config(cfg)


def test_p0_launcher_maps_slurm_only_ranks_and_rendezvous_without_secret_echo():
    text = Path("scripts/outcome_grpo.sbatch").read_text(encoding="utf-8")
    assert '${RUN_DIR:?set RUN_DIR to a fresh unique outcome-GRPO directory}' in text
    assert 'RUN_DIR="${RUN_DIR:-' not in text
    assert 'export MASTER_ADDR="${MASTER_ADDR:-$(hostname)}"' in text
    assert 'SLURM_JOB_ID % 20000' in text
    mapping = (
        'export RANK="$SLURM_PROCID" WORLD_SIZE="$SLURM_NTASKS" '
        'LOCAL_RANK="$SLURM_LOCALID"'
    )
    assert mapping in text
    assert "srun --kill-on-bad-exit=1 bash -c" in text
    assert 'export WANDB_MODE="${LOOM_WANDB_MODE:-online}"' in text
    assert 'export CUBLAS_WORKSPACE_CONFIG=' in text
    assert "torchrun" not in text
    assert "WANDB_API_KEY" not in text


def test_anchor_preserves_cache_placeholder_so_slurm_environment_wins(
    tmp_path: Path, monkeypatch,
):
    from loom.data.loader import resolve_cache_root
    from loom.train.loop import read_config

    cfg = read_config("configs/r0a_outcome_grpo.yaml")
    assert cfg["data"]["cache_dir"] == "cache/"
    monkeypatch.setenv("LOOM_CACHE_DIR", str(tmp_path))
    assert resolve_cache_root(cfg) == tmp_path
    source = inspect.getsource(grpo.ExpertAnchor.from_parent)
    assert 'loader_cfg["data"]["cache_dir"]' not in source


def test_source_closure_and_seed_stat_fail_on_toctou(tmp_path: Path):
    files = ("a.py", "nested/b.py")
    (tmp_path / "nested").mkdir()
    (tmp_path / "a.py").write_text("a=1\n", encoding="utf-8")
    (tmp_path / "nested/b.py").write_text("b=1\n", encoding="utf-8")
    identity = grpo._trainer_source_identity(tmp_path, files)
    grpo._assert_trainer_source_identity(identity, root=tmp_path, files=files)
    (tmp_path / "a.py").write_text("a=222\n", encoding="utf-8")
    with pytest.raises(grpo.OutcomeGRPOError, match="source closure changed"):
        grpo._assert_trainer_source_identity(identity, root=tmp_path, files=files)

    seed = tmp_path / "seed.pt"
    seed.write_bytes(b"seed")
    stat = seed.stat()
    seed_identity = {
        "path": str(seed), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
    }
    grpo._assert_seed_stat(seed_identity)
    seed.write_bytes(b"seed changed")
    with pytest.raises(grpo.OutcomeGRPOError, match="seed checkpoint size changed"):
        grpo._assert_seed_stat(seed_identity)


def test_source_closure_covers_round0_and_runtime_dependencies_and_guard_is_early():
    required = {
        "contracts.py", "stubs.py", "scripts/env.sh",
        "loom/eval/outcome_recovery.py", "loom/eval/policy.py",
        "loom/heads/proposal.py", "loom/heads/decoder.py",
        "loom/losses/proposal_bc.py", "loom/losses/dyn.py",
        "loom/model/estimator.py", "loom/heads/q_action.py",
        "loom/heads/q_delta.py", "loom/train/loop.py",
        "loom/train/determinism.py", "loom/data/loader.py",
        "loom/data/cache.py", "loom/data/canonical.py",
        "loom/data/adapters/libero.py", "loom/data/tower.py",
        "loom/train/schedule.py", "loom/train/preempt.py",
        "loom/train/atomic.py", "loom/train/wandb_util.py",
        "loom/train/ckpt.py", "loom/train/fsdp.py",
    }
    assert required <= set(grpo._TRAINER_SOURCE_FILES)
    source = Path("loom/train/outcome_grpo.py").read_text(encoding="utf-8")
    body = source[source.index("def train_outcome_grpo("):]
    assert body.index("guard = PreemptGuard(run_path)") < body.index(
        "recovery.authenticate_seed_checkpoint(seed_path)"
    )


def test_sampled_switch_balance_is_one_pool_of_all_fourteen_logits(monkeypatch):
    proposal = AnchorProposal()
    proposal.eval()
    n = 3
    z = torch.zeros(n, 1, 1)
    lang = torch.zeros(1, 1)
    with torch.no_grad():
        logits = proposal.logits(z, lang.unsqueeze(0).expand(n, -1, -1))
        order = logits.topk(C.TOPK, dim=-1).indices
        old = pl_log_prob(logits, order)
    arms = [{
        "z": z.clone(), "lang": lang.clone(),
        "ordered_support": order.clone(), "old_logprob": old.clone(),
        "terminal_reward": torch.tensor(float(arm % 2)),
    } for arm in range(8)]
    original = grpo.proposal_switch_balance
    calls: list[tuple[int, ...]] = []

    def capture(value, *, topk=C.TOPK):
        calls.append(tuple(value.shape))
        return original(value, topk=topk)

    monkeypatch.setattr(grpo, "proposal_switch_balance", capture)
    grpo.sampled_group_losses(
        proposal, {"group_id": "g", "arms": arms},
        {arm: (0, 2) for arm in range(1, 8)}, device=torch.device("cpu"),
    )
    assert calls == [(14, C.M)]


class ValidationCollection:
    def __init__(self, payloads):
        self.payloads = payloads
        self.receipts = [{} for _ in payloads]
        self.items = [SimpleNamespace(suite="suite", task_id=index % 2)
                      for index in range(len(payloads))]

    def load(self, index):
        return self.payloads[index]

    def assert_unchanged(self):
        return None


def validation_payload(rewards, *, n=5):
    policy = AnchorProposal()
    z = torch.zeros(n, 1, 1)
    lang = torch.zeros(1, 1)
    with torch.no_grad():
        logits = policy.logits(z, lang.unsqueeze(0).expand(n, -1, -1))
        order = logits.topk(C.TOPK, dim=-1).indices
        old = pl_log_prob(logits, order)
    return {
        "group_id": "heldout",
        "arms": [{
            "z": z.clone(), "lang": lang.clone(),
            "ordered_support": order.clone(), "old_logprob": old.clone(),
            "terminal_reward": torch.tensor(float(rewards[arm])),
        } for arm in range(8)],
    }


def _duplicate_replans(payload):
    out = copy.deepcopy(payload)
    for arm in out["arms"]:
        for key in ("z", "ordered_support", "old_logprob"):
            arm[key] = torch.cat([arm[key], arm[key]], dim=0)
    return out


def test_validation_surrogate_is_reward_sensitive_zero_for_equal_and_chunk_invariant():
    policy = AnchorProposal()
    arm0_success = validation_payload([1, 0, 0, 0, 0, 0, 0, 0])
    sampled_success = validation_payload([0, 1, 0, 0, 0, 0, 0, 0])
    equal = validation_payload([1] * 8)
    first = grpo.evaluate_validation_surrogate(
        policy, ValidationCollection([arm0_success, sampled_success, equal]),
        device=torch.device("cpu"), chunk_replans=1,
    )
    second = grpo.evaluate_validation_surrogate(
        policy, ValidationCollection([
            _duplicate_replans(arm0_success),
            _duplicate_replans(sampled_success),
            _duplicate_replans(equal),
        ]), device=torch.device("cpu"), chunk_replans=3,
    )
    values = [row["surrogate"] for row in first["groups"]]
    assert values[0] < 0 < values[1]
    assert values[2] == 0.0
    assert [row["surrogate"] for row in second["groups"]] == pytest.approx(values)
    assert first["mean_surrogate"] == pytest.approx(second["mean_surrogate"])


def _metric_rows(block_medians=(4.0, 4.0, 4.0, 4.0)):
    rows = []
    tail_start = grpo.EXPECTED_ACCEPTED_UPDATES - 800
    for update in range(1, grpo.EXPECTED_ACCEPTED_UPDATES + 1):
        if update - 1 < tail_start:
            ce = 4.0
        else:
            ce = block_medians[(update - 1 - tail_start) // 200]
        rows.append({
            "global_step": grpo.START_STEP + update,
            "accepted_update": update,
            "anchor_sparse_ce": ce,
            "grad_skipped": 0.0,
        })
    return rows


def _snapshot(value: float, *, approx_kl: float = 0.0):
    groups = [{
        "group_id": f"group-{index:03d}",
        "task": f"task-{index // 10:02d}",
        "surrogate": float(value), "informative": True,
    } for index in range(400)]
    return {
        "groups": groups, "n_groups": 400, "n_tasks": 40,
        "informative_groups": 400, "mean_surrogate": float(value),
        "mean_approx_kl": float(approx_kl),
    }


def _snapshots(seed_value: float, earlier_value: float, final_value: float):
    values = {grpo.START_STEP: _snapshot(seed_value)}
    for step in grpo.CONVERGENCE_SNAPSHOT_STEPS[:-1]:
        values[step] = _snapshot(earlier_value)
    values[grpo.CONVERGENCE_SNAPSHOT_STEPS[-1]] = _snapshot(final_value)
    return values


def test_efficacy_without_plateau_fails_and_plateau_without_efficacy_fails():
    efficacy_only = grpo.evaluate_convergence_gate(
        _snapshots(0.0, 0.0, 0.1), _metric_rows(),
    )
    assert efficacy_only["checks"]["heldout_efficacy"]["pass"] is True
    assert efficacy_only["efficacy"]["samples"] == 2_000
    assert efficacy_only["efficacy"]["seed"] == 0
    assert efficacy_only["efficacy"]["statistical_unit"] == "complete recovery group"
    assert efficacy_only["checks"]["heldout_plateau_all_snapshots"]["pass"] is False
    assert efficacy_only["passed"] is False

    plateau_only = grpo.evaluate_convergence_gate(
        _snapshots(0.1, 0.1, 0.1), _metric_rows(),
    )
    assert plateau_only["checks"]["heldout_efficacy"]["pass"] is False
    assert plateau_only["checks"]["heldout_plateau_all_snapshots"]["pass"] is True
    assert plateau_only["passed"] is False


def test_convergence_pass_requires_final_approx_kl_bound():
    snapshots = _snapshots(0.0, 0.1, 0.1)
    passed = grpo.evaluate_convergence_gate(snapshots, _metric_rows())
    assert passed["passed"] is True
    snapshots[grpo.CONVERGENCE_SNAPSHOT_STEPS[-1]]["mean_approx_kl"] = 0.011
    failed = grpo.evaluate_convergence_gate(snapshots, _metric_rows())
    assert failed["checks"]["final_approx_kl"]["pass"] is False
    assert failed["passed"] is False


def _terminal_rows_for_test():
    rows = []
    live = {
        "sha256": "a" * 64, "n_tensors": 1,
        "n_elements": 1, "n_bytes": 4,
    }
    for rank in range(grpo.EXPECTED_WORLD_SIZE):
        kind, step = grpo._terminal_eval_assignment(rank, grpo.EXPECTED_WORLD_SIZE)
        report = None
        if kind == "snapshot":
            report = _snapshot(float(step))
            report["checkpoint"] = {"proposal": copy.deepcopy(live)}
        elif kind == "trust":
            report = {"passed": True, "checks": {}}
        rows.append({
            "rank": rank, "kind": kind, "step": step, "ok": True,
            "preempted": False, "report": report, "elapsed_seconds": rank + 0.5,
            "parallelism": grpo.TERMINAL_PARALLELISM,
            "proposal_scoring": grpo._proposal_scoring_geometry(
                torch.device("cpu")
            ),
            "live_proposal": copy.deepcopy(live),
        })
    return rows


def test_terminal_tasks_parallelize_six_snapshots_and_trust_without_overlap():
    assignments = [
        grpo._terminal_eval_assignment(rank, grpo.EXPECTED_WORLD_SIZE)
        for rank in range(grpo.EXPECTED_WORLD_SIZE)
    ]
    assert assignments[:6] == [
        ("snapshot", step)
        for step in (grpo.START_STEP, *grpo.CONVERGENCE_SNAPSHOT_STEPS)
    ]
    assert assignments[6] == ("trust", grpo.STOP_STEP)
    assert assignments[7] == ("idle", None)
    assert len(set(assignments[:-1])) == 7

    snapshots, trust, execution = grpo._assemble_terminal_eval_results(
        _terminal_rows_for_test(), world=grpo.EXPECTED_WORLD_SIZE,
    )
    assert set(snapshots) == {
        grpo.START_STEP, *grpo.CONVERGENCE_SNAPSHOT_STEPS,
    }
    assert trust["passed"] is True
    assert execution["parallelism"] == grpo.TERMINAL_PARALLELISM
    assert execution["world_size"] == grpo.EXPECTED_WORLD_SIZE
    assert len(execution["tasks"]) == grpo.EXPECTED_WORLD_SIZE
    assert execution["live_proposal"]["sha256"] == "a" * 64
    production = inspect.getsource(grpo.train_outcome_grpo)
    assert "_run_terminal_eval_task(" in production
    assert "all_gather_object(terminal_rows, local_terminal)" in production
    assert "_assemble_terminal_eval_results(" in production


def test_terminal_task_assembly_fails_closed_on_wrong_rank_error_or_preemption():
    wrong = _terminal_rows_for_test()
    wrong[0]["step"] = grpo.STOP_STEP
    with pytest.raises(grpo.OutcomeGRPOError, match="wrong task"):
        grpo._assemble_terminal_eval_results(
            wrong, world=grpo.EXPECTED_WORLD_SIZE,
        )

    failed = _terminal_rows_for_test()
    failed[3].update({"ok": False, "error": "boom"})
    with pytest.raises(grpo.OutcomeGRPOError, match="rank task failed"):
        grpo._assemble_terminal_eval_results(
            failed, world=grpo.EXPECTED_WORLD_SIZE,
        )

    preempted = _terminal_rows_for_test()
    preempted[5].update({"ok": False, "preempted": True, "error": "budget"})
    with pytest.raises(grpo._PreemptRequested, match="preempted"):
        grpo._assemble_terminal_eval_results(
            preempted, world=grpo.EXPECTED_WORLD_SIZE,
        )

    divergent = _terminal_rows_for_test()
    divergent[6]["live_proposal"]["sha256"] = "b" * 64
    with pytest.raises(grpo.OutcomeGRPOError, match="same final proposal"):
        grpo._assemble_terminal_eval_results(
            divergent, world=grpo.EXPECTED_WORLD_SIZE,
        )


def test_sparse_ce_floor_and_block_drift_fail_independently():
    floor = grpo.evaluate_metric_convergence(
        _metric_rows((grpo.SPARSE_CE_UNIFORM_FLOOR,) * 4),
    )
    assert floor["checks"]["anchor_sparse_ce_block_median_relative_range"]["pass"]
    assert not floor["checks"]["anchor_sparse_ce_terminal_block_median"]["pass"]

    drift = grpo.evaluate_metric_convergence(_metric_rows((4.0, 4.0, 4.0, 4.1)))
    assert not drift["checks"]["anchor_sparse_ce_block_median_relative_range"]["pass"]
    assert drift["checks"]["anchor_sparse_ce_terminal_block_median"]["pass"]


def test_terminal_failure_is_atomic_and_candidate_absent(tmp_path: Path):
    candidate = tmp_path / "candidate_000054466.pt"
    report = {
        "status": "FAIL", "passed": False, "candidate_emitted": False,
        "checks": {"efficacy": {"pass": False}},
    }
    grpo._persist_terminal_failure(tmp_path, report, candidate=candidate)
    saved = json.loads((tmp_path / "terminal_report.json").read_text())
    assert saved == report
    assert not candidate.exists()
    candidate.write_bytes(b"must-not-coexist")
    with pytest.raises(grpo.OutcomeGRPOError, match="cannot coexist"):
        grpo._persist_terminal_failure(tmp_path, report, candidate=candidate)


def test_nondivisible_group_count_is_rank_unique_every_step_and_resumable():
    groups = [[10, 11, 12, 13, 14] for _ in range(6)]
    samplers = [grpo.DeterministicOutcomeSampler(
        groups, seed=5, rank=rank, world_size=4, contexts_per_arm=2,
        identity_digests=[str(index) for index in range(6)],
    ) for rank in range(4)]
    for offset in range(40):
        assignments = [
            sampler.group_at(grpo.START_STEP + offset)[1] for sampler in samplers
        ]
        assert len(set(assignments)) == 4
    state = samplers[0].state_dict(grpo.START_STEP + 17)
    resumed = grpo.DeterministicOutcomeSampler(
        groups, seed=5, rank=3, world_size=4, contexts_per_arm=2,
        identity_digests=[str(index) for index in range(6)],
    )
    resumed.validate_state_dict(state, grpo.START_STEP + 17)
    assert resumed.group_at(grpo.START_STEP + 17) == samplers[3].group_at(
        grpo.START_STEP + 17,
    )


def test_metrics_resume_truncates_ahead_and_partial_rows_to_exact_checkpoint(tmp_path: Path):
    path = tmp_path / "metrics.jsonl"
    config_hash = "abcd"
    rows = [{"global_step": grpo.START_STEP + index, "config_hash": config_hash}
            for index in range(1, 5)]
    text = "".join(json.dumps(row) + "\n" for row in rows) + '{"global_step":'
    path.write_text(text, encoding="utf-8")
    retained = grpo._reconcile_metrics_to_checkpoint(
        path, checkpoint_step=grpo.START_STEP + 2, config_hash=config_hash,
    )
    assert [row["global_step"] for row in retained] == [
        grpo.START_STEP + 1, grpo.START_STEP + 2,
    ]
    assert path.read_text().endswith("\n")
    assert [json.loads(line)["global_step"] for line in path.read_text().splitlines()] \
        == [grpo.START_STEP + 1, grpo.START_STEP + 2]

    path.write_text(json.dumps(rows[1]) + "\n", encoding="utf-8")
    with pytest.raises(grpo.OutcomeGRPOError, match="one contiguous row"):
        grpo._reconcile_metrics_to_checkpoint(
            path, checkpoint_step=grpo.START_STEP + 2, config_hash=config_hash,
        )


def test_metrics_are_fsynced_before_checkpoint_pointer_can_advance(monkeypatch):
    events = []

    class Handle:
        def flush(self):
            events.append("flush")

        def fileno(self):
            events.append("fileno")
            return 17

    monkeypatch.setattr(grpo.os, "fsync", lambda fd: events.append(("fsync", fd)))
    grpo._durable_metrics_barrier(Handle())
    assert events == ["flush", "fileno", ("fsync", 17)]
    source = Path("loom/train/outcome_grpo.py").read_text(encoding="utf-8")
    save_block = source[source.index("if should_save:"):source.index("if stop:", source.index("if should_save:"))]
    assert save_block.index("_durable_metrics_barrier") < save_block.index(
        "_save_trainer_checkpoint"
    )
