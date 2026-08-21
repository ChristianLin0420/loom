from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import contracts as C
from loom.heads.proposal import argmax_coeff, pl_log_prob
from loom.losses.proposal_bc import proposal_sparse_ce_loss
from loom.train import outcome_grpo_round0 as round0


class LookupProposal(nn.Module):
    """Tiny proposal whose stored belief selects one row of trainable logits."""

    def __init__(self, table: torch.Tensor) -> None:
        super().__init__()
        self.table = nn.Parameter(table.clone().float())

    def logits(self, z: torch.Tensor, lang: torch.Tensor) -> torch.Tensor:
        del lang
        return self.table[z[:, 0, 0].long()]


def _orders(table: torch.Tensor) -> torch.Tensor:
    return table.topk(C.TOPK, dim=-1).indices


def test_locked_round0_recipe_and_cli_require_validation() -> None:
    assert round0.LEARNING_RATE == 5e-6
    assert round0.ADAMW_BETAS == (0.9, 0.95)
    assert round0.ADAMW_WEIGHT_DECAY == 0.05
    assert round0.EPOCHS == 2
    assert round0.GRAD_CLIP == 1.0
    assert round0.SWITCH_BALANCE_WEIGHT == 1e-2
    assert round0.expected_optimizer_updates(200) == 400
    with pytest.raises(ValueError):
        round0.expected_optimizer_updates(0)
    with pytest.raises(SystemExit):
        round0.build_parser().parse_args([
            "--checkpoint", "parent.pt", "--train-collection", "train",
            "--out", "child.pt",
        ])


def test_group_advantages_include_arm0_and_constant_group_is_exact_zero() -> None:
    rewards = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.float32)
    got = round0.normalised_group_advantages(rewards)
    expected = (rewards - rewards.mean()) / (
        (rewards - rewards.mean()).square().mean().sqrt()
    )
    assert torch.equal(got, expected)
    assert torch.equal(
        round0.normalised_group_advantages(torch.ones(8)), torch.zeros(8),
    )

    changed_control = rewards.clone()
    changed_control[0] = 1
    changed = round0.normalised_group_advantages(changed_control)
    assert not torch.equal(got[1:], changed[1:])


def test_clipped_objective_uses_ppo_sign_and_inclusive_boundaries() -> None:
    ratio = torch.tensor([1.3, 0.7, 1.2, 0.8], dtype=torch.float64)
    current = ratio.log().requires_grad_()
    old = torch.zeros_like(current)
    positive, got_ratio, clipped = round0.clipped_grpo_objective(
        current, old, 2.0,
    )
    assert torch.allclose(got_ratio, ratio)
    assert torch.allclose(positive, ratio.new_tensor([2.4, 1.4, 2.4, 1.6]))
    assert clipped.tolist() == [True, True, False, False]

    negative, _, _ = round0.clipped_grpo_objective(current, old, -2.0)
    assert torch.allclose(negative, ratio.new_tensor([-2.6, -1.6, -2.4, -1.6]))


def test_group_loss_means_replans_then_sampled_trajectories() -> None:
    rewards = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.float32)
    advantage = round0.normalised_group_advantages(rewards)
    ratios = [
        torch.tensor([1.0]),
        torch.tensor([0.9, 1.1]),
        torch.tensor([1.0, 1.0, 1.0]),
        torch.tensor([1.2]),
        torch.tensor([0.8, 1.0]),
        torch.tensor([1.0]),
        torch.tensor([1.1, 0.9, 1.0, 1.0]),
    ]
    current = [value.log().requires_grad_() for value in ratios]
    old = [torch.zeros_like(value) for value in ratios]
    loss, flat_ratio = round0.group_grpo_loss(current, old, rewards)
    expected_terms = []
    for arm, value in enumerate(ratios, start=1):
        objective, _, _ = round0.clipped_grpo_objective(
            value.log(), torch.zeros_like(value), advantage[arm],
        )
        expected_terms.append(objective.mean())
    assert torch.allclose(loss, -torch.stack(expected_terms).mean())
    assert torch.equal(flat_ratio, torch.cat(ratios))


def test_stored_order_pl_does_not_canonicalise_sampled_atom() -> None:
    table = torch.linspace(-1.0, 1.0, C.M).unsqueeze(0)
    proposal = LookupProposal(table)
    z = torch.zeros(1, 1, 1)
    lang = torch.zeros(1, 1, 1)
    support = table.topk(C.TOPK, dim=-1).indices
    reversed_order = support.flip(-1)
    first, logits = round0.stored_order_logprob(proposal, z, lang, support)
    second, _ = round0.stored_order_logprob(proposal, z, lang, reversed_order)
    assert torch.equal(first, pl_log_prob(logits.float(), support))
    assert torch.equal(second, pl_log_prob(logits.float(), reversed_order))
    assert not torch.equal(first, second)
    first.sum().backward()
    assert proposal.table.grad is not None


def test_switch_matches_existing_definition_and_reaches_unselected_logits() -> None:
    from loom.train.loop import _switch_balance

    logits = torch.randn(9, C.M, generator=torch.Generator().manual_seed(7),
                         requires_grad=True)
    hard = argmax_coeff(logits, C.TOPK, C.M)
    got = round0.proposal_switch_balance(logits)
    expected = _switch_balance(hard, logits)
    assert torch.equal(got, expected)
    selected = torch.zeros_like(logits, dtype=torch.bool).scatter_(
        1, logits.detach().topk(C.TOPK, dim=-1).indices, True,
    )
    got.backward()
    assert logits.grad is not None
    assert bool((logits.grad[~selected].abs() > 0).any())


def test_anchor_objective_is_sparse_ce_plus_one_concatenated_switch() -> None:
    table = torch.randn(2, C.M, generator=torch.Generator().manual_seed(9))
    proposal = LookupProposal(table)
    beliefs = [torch.tensor([[[0.0]], [[1.0]]]) for _ in range(C.DEPTH)]
    lang = torch.zeros(2, 1, 1)
    target = torch.zeros(2, C.M)
    target[:, :C.TOPK] = 1.0 / C.TOPK
    targets = [target.clone() for _ in range(C.DEPTH)]

    total, terms = round0.expert_anchor_objective(
        proposal, beliefs, lang, targets, temperature=1.0, ce_weight=1.0,
    )
    logits = torch.cat([proposal.logits(z, lang) for z in beliefs], dim=0)
    expected_ce = torch.stack([
        proposal_sparse_ce_loss(
            proposal, z, lang, target, temperature=1.0, detach_belief=True,
        )
        for z in beliefs
    ]).mean()
    expected_switch = round0.proposal_switch_balance(logits)
    assert torch.allclose(terms["sparse_ce"], expected_ce)
    assert torch.allclose(terms["switch_balance"], expected_switch)
    assert torch.allclose(total, expected_ce + 1e-2 * expected_switch)
    total.backward()
    assert proposal.table.grad is not None


class _AnchorForUpdate:
    def __init__(self, proposal: LookupProposal) -> None:
        self.proposal = proposal
        self.steps: list[int] = []

    def loss(self, update_index: int):
        self.steps.append(update_index)
        loss = 1e-3 * self.proposal.table.square().mean()
        return loss, {"sparse_ce": 0.2, "switch_balance": 1.1,
                      "total": float(loss.detach())}

    @staticmethod
    def unexpected_gradients() -> list[str]:
        return []


def _training_payload(proposal: LookupProposal) -> dict:
    order = _orders(proposal.table.detach())[0]
    arms: list[dict] = [{"terminal_reward": torch.tensor(0.0)}]
    for arm in range(1, 8):
        n = 1 + arm % 3
        z = torch.zeros(n, 1, 1)
        lang = torch.zeros(1, 1)
        ordered = order.expand(n, -1).clone()
        old = pl_log_prob(proposal.table.detach().expand(n, -1), ordered)
        arms.append({
            "terminal_reward": torch.tensor(float(arm % 2)),
            "z": z, "lang": lang, "ordered_support": ordered,
            "old_logprob": old,
        })
    return {"group_id": "test-group", "arms": arms}


def test_one_update_excludes_arm0_and_applies_global_grad_clip(monkeypatch) -> None:
    proposal = LookupProposal(torch.linspace(-1.0, 1.0, C.M).unsqueeze(0))
    payload = _training_payload(proposal)
    # No arm-0 belief/order/logprob exists: touching it would fail this update.
    assert set(payload["arms"][0]) == {"terminal_reward"}
    anchor = _AnchorForUpdate(proposal)
    optimizer = torch.optim.AdamW(
        proposal.parameters(), lr=round0.LEARNING_RATE,
        betas=round0.ADAMW_BETAS, weight_decay=round0.ADAMW_WEIGHT_DECAY,
        eps=round0.ADAMW_EPS,
    )
    real_clip = torch.nn.utils.clip_grad_norm_
    calls: list[float] = []

    def observed_clip(parameters, max_norm, **kwargs):
        calls.append(float(max_norm))
        return real_clip(parameters, max_norm, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", observed_clip)
    metrics = round0._train_one_group(
        proposal, payload, optimizer, anchor, update_index=17,
        device=torch.device("cpu"), chunk_replans=2,
    )
    assert calls == [1.0]
    assert anchor.steps == [17]
    assert metrics["ratio_atoms"] == sum(1 + arm % 3 for arm in range(1, 8))
    assert metrics["grad_clip"] == 1.0
    assert optimizer.state


def test_frozen_digest_ignores_proposal_and_covers_scalars() -> None:
    state = {
        "proposal.weight": torch.tensor([1.0, 2.0]),
        "estimator.weight": torch.tensor([[3.0]]),
        "q_action.scalar": torch.tensor(4.0),
    }
    baseline = round0.frozen_model_digest(state)
    proposal_changed = copy.deepcopy(state)
    proposal_changed["proposal.weight"].add_(100)
    assert round0.frozen_model_digest(proposal_changed) == baseline
    frozen_changed = copy.deepcopy(state)
    frozen_changed["q_action.scalar"].add_(1)
    assert round0.frozen_model_digest(frozen_changed) != baseline


class FakeValidation:
    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = payloads
        self.receipts = tuple(range(len(payloads)))
        self.checked = False

    def load(self, index: int) -> dict:
        return self.payloads[index]

    def assert_unchanged(self) -> None:
        self.checked = True


def _validation_fixture() -> tuple[LookupProposal, FakeValidation]:
    table = torch.full((4, C.M), -10.0)
    for group in range(4):
        table[group, 4 * group:4 * group + 4] = torch.tensor([4.0, 3.0, 2.0, 1.0])
    proposal = LookupProposal(table)
    payloads: list[dict] = []
    for group in range(4):
        z = torch.tensor([[[float(group)]]])
        lang = torch.zeros(1, 1)
        logits = table[group:group + 1]
        order = _orders(logits)
        coeff = argmax_coeff(logits, C.TOPK, C.M)
        old = pl_log_prob(logits, order)
        arms = [{
            "z": z, "lang": lang, "ordered_support": order.clone(),
            "coeff": coeff.clone(), "old_logprob": old.clone(),
            "terminal_reward": torch.tensor(0.0),
        }]
        for arm in range(1, 8):
            arms.append({
                "z": z.clone(), "lang": lang.clone(),
                "ordered_support": order.clone(), "coeff": coeff.clone(),
                "old_logprob": old.clone(),
                "terminal_reward": torch.tensor(float(arm % 2)),
            })
        payloads.append({"group_id": f"g{group}", "arms": arms})
    return proposal, FakeValidation(payloads)


def _anchor_metrics(*, ce: float = 1.0, overlap: float = 0.8,
                    digest: str = "same") -> dict:
    return {"sparse_ce": ce, "topk_overlap": overlap,
            "target_sha256": digest}


def test_validation_trust_passes_and_uses_l1_coefficient_drift() -> None:
    proposal, validation = _validation_fixture()
    report = round0.evaluate_trust_gates(
        proposal, validation, device=torch.device("cpu"), chunk_replans=3,
        anchor_initial=_anchor_metrics(),
        anchor_final=_anchor_metrics(ce=0.99, overlap=0.751),
    )
    assert report["passed"] is True
    assert validation.checked
    assert report["checks"]["clip_fraction"]["value"] == 0.0
    assert report["checks"]["ess_fraction"]["value"] == pytest.approx(1.0)
    assert report["checks"]["coeff_drift_p95_l1"]["value"] == 0.0
    assert report["checks"]["live_ops"]["value"] == 16
    assert report["definitions"]["arm0_importance_ratios"] == 0

    # L1=0.06 while L2=sqrt(2)*0.03<0.05: this must fail the locked L1 gate.
    for payload in validation.payloads:
        baseline = payload["arms"][0]["coeff"]
        selected = baseline[0].topk(C.TOPK).indices
        baseline[0, selected[0]] += 0.03
        baseline[0, selected[1]] -= 0.03
    failed = round0.evaluate_trust_gates(
        proposal, validation, device=torch.device("cpu"), chunk_replans=3,
        anchor_initial=_anchor_metrics(), anchor_final=_anchor_metrics(),
    )
    row = failed["checks"]["coeff_drift_p95_l1"]
    assert row["pass"] is False
    assert row["value"] == pytest.approx(0.06)


def test_expert_preservation_and_health_are_terminal_trust_gates() -> None:
    proposal, validation = _validation_fixture()
    report = round0.evaluate_trust_gates(
        proposal, validation, device=torch.device("cpu"), chunk_replans=8,
        anchor_initial=_anchor_metrics(),
        anchor_final=_anchor_metrics(ce=1.001, overlap=0.74, digest="changed"),
        training_nonfinite=1, unexpected_gradients=["estimator.x"],
    )
    assert report["passed"] is False
    for name in (
        "expert_target_identity", "expert_sparse_ce_no_worsening",
        "expert_topk_overlap_decline", "nonfinite", "unexpected_gradients",
    ):
        assert report["checks"][name]["pass"] is False


def test_expert_gate_steps_are_after_all_400_training_anchor_steps(monkeypatch) -> None:
    proposal = nn.Linear(1, 1)
    anchor = round0.ExpertAnchor(
        proposal=proposal, estimator=nn.Linear(1, 1), q_action=nn.Linear(1, 1),
        sampler=object(), device=torch.device("cpu"), parent_global_step=49_666,
        temperature=1.0, weight=1.0, data_provenance={},
    )
    observed: list[int] = []

    def prepare(step: int):
        observed.append(step)
        return [], torch.zeros(1), [], "libero_franka"

    monkeypatch.setattr(anchor, "_prepare_step", prepare)
    monkeypatch.setattr(anchor, "evaluate_gate", lambda: {
        "sparse_ce": 1.0, "topk_overlap": 0.5, "target_sha256": "x",
        "n_horizon_examples": 1,
    })
    report = anchor.cache_gate(400)
    assert observed == list(range(50_067, 50_083))
    assert report["first_step"] == 50_067
    assert report["last_step"] == 50_082
    assert 50_066 < report["first_step"]


def _write_fake_collection(monkeypatch, root: Path, *, status: str = "COMPLETE"):
    items = (SimpleNamespace(key="g0"), SimpleNamespace(key="g1"))
    monkeypatch.setattr(round0.recovery, "collection_items", lambda split: items)
    monkeypatch.setattr(round0.recovery, "work_key", lambda item: item.key)
    monkeypatch.setattr(round0.recovery, "source_digest", lambda repo: "a" * 64)
    monkeypatch.setattr(
        round0.recovery, "collection_identity",
        lambda checkpoint, split, source_sha256: {
            "checkpoint": checkpoint, "split": split, "source": source_sha256,
        },
    )
    monkeypatch.setattr(round0.recovery, "validate_group_payload", lambda *a, **k: None)
    identity = round0.recovery.collection_identity(
        checkpoint={"id": 1}, split="train0", source_sha256="a" * 64,
    )
    identity_digest = round0.recovery.identity_digest(identity)
    groups = root / "groups"
    groups.mkdir(parents=True)
    receipts = []
    terminal = [0] * 8
    replans = [0] * 8
    for group_index, item in enumerate(items):
        arms = []
        rewards = []
        counts = []
        for arm in range(8):
            n = 1 + (arm % 2)
            reward = int((group_index + arm) % 2)
            arms.append({"z": torch.zeros(n, 1, 1),
                         "terminal_reward": torch.tensor(float(reward))})
            rewards.append(reward)
            counts.append(n)
            terminal[arm] += reward
            replans[arm] += n
        payload = {"group_id": item.key, "arms": arms}
        sidecar = groups / f"{item.key}.pt"
        torch.save(payload, sidecar)
        receipts.append({
            "group_id": item.key, "sidecar": f"groups/{item.key}.pt",
            "sha256": round0.recovery.sha256_file(sidecar),
            "size": sidecar.stat().st_size, "n_arms": 8,
            "n_replans_by_arm": counts, "terminal_rewards": rewards,
            "worker": {"pid": 1, "device": "test"},
        })
    complete = status == "COMPLETE"
    manifest = {
        "format_version": round0.recovery.FORMAT_VERSION,
        "kind": "loom_outcome_recovery_collection", "identity": identity,
        "identity_digest": identity_digest, "split": "train0",
        "started_utc": "2026-01-01T00:00:00Z",
        "updated_utc": "2026-01-01T00:00:00Z",
        "summary": {
            "status": status, "complete": complete,
            "n_groups": 2, "n_expected_groups": 2,
            "n_trajectories": 16, "n_expected_trajectories": 16,
            "terminal_successes_by_arm": terminal, "replans_by_arm": replans,
        },
        "groups": receipts,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return items


def test_collection_authentication_and_post_open_tamper(monkeypatch, tmp_path) -> None:
    _write_fake_collection(monkeypatch, tmp_path)
    collection = round0.ValidatedRecoveryCollection.open(
        tmp_path, checkpoint_identity={"id": 1}, purpose="train",
    )
    assert len(collection.receipts) == 2
    assert collection.load(0)["group_id"] == "g0"
    sidecar = tmp_path / "groups" / "g0.pt"
    sidecar.write_bytes(sidecar.read_bytes() + b"tamper")
    with pytest.raises(round0.OutcomeGRPOError, match="sidecar changed"):
        collection.load(0)


def test_collection_rejects_running_manifest_before_loading_sidecars(
    monkeypatch, tmp_path,
) -> None:
    _write_fake_collection(monkeypatch, tmp_path, status="RUNNING")
    monkeypatch.setattr(
        round0, "_load_group",
        lambda path: pytest.fail(f"loaded incomplete sidecar {path}"),
    )
    with pytest.raises(round0.OutcomeGRPOError, match="not terminal COMPLETE"):
        round0.ValidatedRecoveryCollection.open(
            tmp_path, checkpoint_identity={"id": 1}, purpose="train",
        )


def _descendant_provenance() -> dict:
    return {
        "train_collection": {"identity_digest": "train"},
        "validation_collection": {"identity_digest": "validation"},
        "training": {"groups_per_epoch": 200, "optimizer_steps": 400},
        "trust_gate": {"passed": True},
    }


def test_descendant_changes_only_proposal_and_carries_optimizer(
    tmp_path, monkeypatch,
) -> None:
    proposal = nn.Linear(2, 2)
    parent_state = {
        "proposal.weight": proposal.weight.detach().clone(),
        "proposal.bias": proposal.bias.detach().clone(),
        "estimator.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
        "decoder.weight": torch.tensor([[5.0]]),
        "q_action.scalar": torch.tensor(7.0),
    }
    resolved = {"model": {}, "link": {"run_dir": "ignored"}}
    parent = {
        "global_step": 49_666, "config_hash": round0._config_hash(resolved),
        "resolved_config": resolved, "model": parent_state,
    }
    optimizer = torch.optim.AdamW(
        proposal.parameters(), lr=round0.LEARNING_RATE,
        betas=round0.ADAMW_BETAS, weight_decay=round0.ADAMW_WEIGHT_DECAY,
        eps=round0.ADAMW_EPS,
    )
    for _ in range(400):
        proposal(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    out = tmp_path / "descendant.pt"
    report = round0.write_descendant_checkpoint(
        out, parent=parent, parent_identity={"sha256": "f" * 64},
        proposal=proposal, optimizer=optimizer, optimizer_steps=400,
        provenance=_descendant_provenance(),
    )
    child = torch.load(out, map_location="cpu", weights_only=False)
    assert report["global_step"] == 50_066
    assert child["global_step"] == 50_066
    assert child["optimizer"]["kind"] == "proposal_only_adamw"
    assert child["optimizer"]["state_reset_at_entry"] is True
    assert child["optimizer"]["state_dict"]["state"]
    recipe = child["resolved_config"]["outcome_grpo"]
    assert recipe["epochs"] == 2
    assert recipe["optimizer_steps"] == 400
    assert recipe["optimizer"]["lr"] == 5e-6
    assert recipe["optimizer"]["scheduler"] is None
    assert "outcome_grpo_round0" not in child["resolved_config"]
    assert "outcome_grpo_round0" not in child
    assert child["outcome_grpo"]["kind"] == round0.TRAINER_KIND
    assert child["consolidated"]["step"] == 50_066
    assert child["consolidated"]["proposal_only_update"] is True
    assert round0._config_hash(child["resolved_config"]) == child["config_hash"]
    assert round0.frozen_model_digest(child["model"]) == child["outcome_grpo"]["frozen_model"]
    assert round0.proposal_model_digest(child["model"]) == child["outcome_grpo"]["final_proposal"]
    for name, value in parent_state.items():
        if not name.startswith("proposal."):
            assert torch.equal(child["model"][name], value)
    assert torch.equal(child["model"]["proposal.weight"], proposal.weight.detach())
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        round0.write_descendant_checkpoint(
            out, parent=parent, parent_identity={"sha256": "f" * 64},
            proposal=proposal, optimizer=optimizer, optimizer_steps=400,
            provenance=_descendant_provenance(),
        )

    invalid = tmp_path / "failed-reload.pt"
    monkeypatch.setattr(
        round0.torch, "load",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("test reload failure")),
    )
    with pytest.raises(round0.OutcomeGRPOError, match="authenticate reloaded"):
        round0.write_descendant_checkpoint(
            invalid, parent=parent, parent_identity={"sha256": "f" * 64},
            proposal=proposal, optimizer=optimizer, optimizer_steps=400,
            provenance=_descendant_provenance(),
        )
    assert not invalid.exists()


def test_source_identity_binds_shared_runtime_and_never_imports_contested_trainer() -> None:
    identity = round0._source_identity()
    for required in (
        "contracts.py", "loom/heads/proposal.py", "loom/losses/proposal_bc.py",
        "loom/model/estimator.py", "loom/heads/q_action.py", "loom/train/loop.py",
        "loom/data/loader.py", "loom/data/canonical.py",
    ):
        assert required in identity["files"]
    core = Path(round0.__file__).read_text(encoding="utf-8")
    wrapper = (Path(round0.__file__).parents[2]
               / "scripts/train_outcome_grpo_round0.py").read_text(encoding="utf-8")
    assert "from loom.train.outcome_grpo import" not in core + wrapper
    assert "import loom.train.outcome_grpo" not in core + wrapper


def test_unique_sbatch_is_single_gpu_and_fail_closed() -> None:
    root = Path(round0.__file__).parents[2]
    launcher = (root / "scripts/outcome_grpo_round0.sbatch").read_text(
        encoding="utf-8",
    )
    for directive in (
        "#SBATCH --time=04:00:00", "#SBATCH --nodes=1",
        "#SBATCH --gpus=1", "#SBATCH --ntasks=1",
    ):
        assert directive in launcher
    for exact_path in (
        "runs/r0a_deploy_s1_eval/ckpt_000049666.pt",
        "runs/outcome_recovery_s49666_train0",
        "runs/outcome_recovery_s49666_validation",
    ):
        assert exact_path in launcher
    for required in (
        "source scripts/env.sh", "test ! -e \"$OUT\"",
        "scripts/train_outcome_grpo_round0.py", "_config_hash",
        "loom_outcome_grpo_round0_proposal_descendant", "optimizer_steps",
        "trust[\"passed\"] is True", "frozen_model_digest",
    ):
        assert required in launcher
    assert "scripts/train_outcome_grpo.py" not in launcher
    assert "outcome_promotion_gate" not in launcher
    assert "sbatch " not in "\n".join(
        line for line in launcher.splitlines() if not line.lstrip().startswith("#")
    )


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (round0.TrustGateError({"checks": {"clip": {"pass": False}}}), 4),
        (round0.ExpertAnchorUnavailable("missing"), 3),
        (round0.OutcomeGRPOError("bad"), 2),
    ],
)
def test_cli_fail_closed_exit_codes(monkeypatch, error, expected) -> None:
    monkeypatch.setattr(round0, "train_outcome_grpo_round0",
                        lambda **kwargs: (_ for _ in ()).throw(error))
    assert round0.main([
        "--checkpoint", "parent.pt", "--train-collection", "train",
        "--validation-collection", "validation", "--out", "child.pt",
    ]) == expected
