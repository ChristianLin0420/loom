"""Focused contracts for the isolated positive-advantage outcome core."""

from __future__ import annotations

import inspect
import math
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F

import contracts as C
from loom.eval import outcome_recovery as recovery
from loom.heads.proposal import pl_log_prob
from loom.train import outcome_positive_advantage as pa


class RowwiseWitnessProposal(nn.Module):
    """Small fp32 proposal that records the inherited scoring geometry."""

    def __init__(self, m: int = C.M):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.125, dtype=torch.float32))
        self.register_buffer("base", torch.linspace(-1.0, 1.0, m))
        self.seen_batch_sizes: list[int] = []
        self.seen_dtypes: list[torch.dtype] = []
        self.seen_autocast: list[bool] = []

    def logits(self, z: Tensor, lang: Tensor) -> Tensor:  # noqa: ARG002
        self.seen_batch_sizes.append(int(z.shape[0]))
        self.seen_dtypes.append(z.dtype)
        self.seen_autocast.append(torch.is_autocast_enabled(z.device.type))
        signal = z.float().mean(dim=(-1, -2)).unsqueeze(-1)
        slope = torch.linspace(-0.5, 0.5, self.base.numel(), device=z.device)
        return self.base.unsqueeze(0) + self.scale * signal * slope.unsqueeze(0)

    def clear_witness(self) -> None:
        self.seen_batch_sizes.clear()
        self.seen_dtypes.clear()
        self.seen_autocast.clear()


def _payload(proposal: RowwiseWitnessProposal, rewards: list[float]) -> dict:
    assert len(rewards) == recovery.GROUP_SIZE
    arms: list[dict] = [{"terminal_reward": torch.tensor(rewards[0])}]
    with torch.no_grad():
        for arm in range(1, recovery.GROUP_SIZE):
            n = 4
            z = (
                torch.arange(n, dtype=torch.float32).reshape(n, 1, 1)
                + float(arm) / 10.0
            )
            lang = torch.zeros(1, 1)
            logits = torch.cat([
                proposal.logits(
                    z[row:row + 1], lang.reshape(1, 1, 1),
                )
                for row in range(n)
            ])
            order = logits.topk(C.TOPK, dim=-1).indices
            old = torch.cat([
                pl_log_prob(logits[row:row + 1], order[row:row + 1])
                for row in range(n)
            ])
            arms.append({
                "z": z,
                "lang": lang,
                "ordered_support": order,
                "old_logprob": old,
                "terminal_reward": torch.tensor(rewards[arm]),
            })
    proposal.clear_witness()
    return {"group_id": "positive-advantage-test", "arms": arms}


def _standard_categorical_forward_kl(
    current: Tensor,
    seed: Tensor,
) -> Tensor:
    current32 = current.float()
    seed32 = seed.detach().to(device=current.device, dtype=torch.float32)
    p_seed = F.softmax(seed32, dim=-1)
    return (
        p_seed
        * (F.log_softmax(seed32, dim=-1) - F.log_softmax(current32, dim=-1))
    ).sum(dim=-1)


def test_advantages_use_all_eight_population_rms_and_constant_is_exact_zero():
    rewards = torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    advantages, weights = pa.positive_advantage_weights(rewards)
    centred = rewards - rewards.mean()
    expected = centred / centred.square().mean().sqrt()

    assert torch.equal(advantages, expected)
    assert float(advantages.mean()) == pytest.approx(0.0, abs=1e-7)
    assert float(advantages.square().mean()) == pytest.approx(1.0, abs=2e-7)
    assert torch.equal(weights, expected.clamp_min(0.0))

    constant, constant_weights = pa.positive_advantage_weights(torch.ones(8))
    assert torch.count_nonzero(constant) == 0
    assert torch.count_nonzero(constant_weights) == 0

    with pytest.raises(ValueError, match="needs 8 rewards"):
        pa.standardised_terminal_advantages(torch.ones(7))
    with pytest.raises(pa.PositiveAdvantageError, match="nan/inf"):
        pa.standardised_terminal_advantages(
            torch.tensor([0.0] * 7 + [float("nan")])
        )


def test_clipped_positive_objective_has_fixed_point8_point12_geometry():
    ratios = torch.tensor([0.7, 0.9, 1.1, 1.3], requires_grad=True)
    current = ratios.log()
    old = torch.zeros_like(current)
    loss, got_ratios, clipped = pa.clipped_positive_advantage_objective(
        current, old, 2.0,
    )

    assert torch.allclose(got_ratios, ratios)
    assert torch.allclose(loss, torch.tensor([-1.4, -1.8, -2.2, -2.4]))
    assert clipped.tolist() == [True, False, False, True]
    assert pa.RATIO_CLIP_LOW == 0.8
    assert pa.RATIO_CLIP_HIGH == 1.2

    zero_loss, _, _ = pa.clipped_positive_advantage_objective(current, old, 0.0)
    zero_loss.sum().backward()
    assert ratios.grad is not None
    assert torch.count_nonzero(ratios.grad) == 0


def test_group_loss_is_nested_equal_arm_mean_and_retains_zero_weight_arms():
    rewards = torch.tensor(
        [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        requires_grad=True,
    )
    advantages, weights = pa.positive_advantage_weights(rewards)
    lengths = [1, 2, 3, 4, 1, 2, 4]
    ratios_by_arm = [
        torch.linspace(0.7 + 0.03 * arm, 1.3 - 0.02 * arm, length)
        for arm, length in enumerate(lengths, start=1)
    ]
    current = [value.log().detach().requires_grad_() for value in ratios_by_arm]
    old = [torch.zeros_like(value, requires_grad=True) for value in current]

    loss, ratios = pa.group_positive_advantage_loss(current, old, rewards)
    manual_arm_terms = []
    for arm, ratio in enumerate(ratios_by_arm, start=1):
        clipped = ratio.clamp(0.8, 1.2)
        manual_arm_terms.append(
            -torch.minimum(ratio * weights[arm], clipped * weights[arm]).mean()
        )
    expected = torch.stack(manual_arm_terms).mean()
    pooled = -torch.cat([
        torch.minimum(
            ratio * weights[arm], ratio.clamp(0.8, 1.2) * weights[arm],
        )
        for arm, ratio in enumerate(ratios_by_arm, start=1)
    ]).mean()

    assert torch.allclose(loss, expected)
    assert not torch.allclose(loss, pooled)
    assert ratios.numel() == sum(lengths)
    loss.backward()
    assert all(value.grad is not None for value in current)
    assert all(value.grad is None for value in old)
    assert rewards.grad is None
    for arm, value in enumerate(current, start=1):
        if advantages[arm] <= 0.0:
            assert torch.count_nonzero(value.grad) == 0


def test_arm0_only_success_has_zero_sampled_loss_and_bitwise_zero_gradients():
    proposal = RowwiseWitnessProposal().eval()
    payload = _payload(proposal, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    indices = {arm: tuple(range(1 + arm % 3)) for arm in range(1, 8)}

    result = pa.sampled_positive_advantage_objective(
        proposal, payload, indices, device=torch.device("cpu"),
    )
    assert result.positive_weights[0] > 0.0
    assert torch.count_nonzero(result.positive_weights[1:]) == 0
    assert float(result.loss) == 0.0
    assert result.metrics["arm0_score_atoms"] == 0.0
    assert result.metrics["arm0_positive_weight_ignored"] > 0.0
    assert result.metrics["arm_terms_retained"] == 7.0
    assert result.metrics["sampled_zero_weight_arms"] == 7.0
    result.loss.backward()
    assert proposal.scale.grad is not None
    assert torch.count_nonzero(proposal.scale.grad) == 0


def test_sampled_objective_uses_exact_b1_stored_order_and_unequal_nested_means():
    proposal = RowwiseWitnessProposal().eval()
    rewards = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    payload = _payload(proposal, rewards)
    for arm_index, arm in enumerate(payload["arms"]):
        arm["terminal_reward"] = arm["terminal_reward"].float().requires_grad_()
        if arm_index > 0:
            arm["old_logprob"] = arm["old_logprob"].detach().requires_grad_()
    indices = {arm: tuple(range(1 + arm % 3)) for arm in range(1, 8)}
    expected_atoms = sum(len(values) for values in indices.values())

    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = pa.sampled_positive_advantage_objective(
            proposal, payload, indices, device=torch.device("cpu"),
        )

    advantages, weights = pa.positive_advantage_weights(rewards)
    assert torch.equal(result.advantages, advantages)
    assert torch.equal(result.positive_weights, weights)
    assert torch.allclose(result.loss, -weights[1:].mean())
    assert result.metrics["ratio_min"] == 1.0
    assert result.metrics["ratio_mean"] == 1.0
    assert result.metrics["ratio_max"] == 1.0
    assert result.metrics["ratio_atoms"] == float(expected_atoms)
    assert result.metrics["ratio_ess_fraction"] == 1.0
    assert result.metrics["clip_fraction"] == 0.0
    assert result.metrics["contexts_min_per_arm"] == 1.0
    assert result.metrics["contexts_max_per_arm"] == 3.0
    assert proposal.seen_batch_sizes == [1] * expected_atoms
    assert proposal.seen_dtypes == [torch.float32] * expected_atoms
    assert proposal.seen_autocast == [False] * expected_atoms
    result.loss.backward()
    assert proposal.scale.grad is not None
    assert bool(torch.isfinite(proposal.scale.grad))
    assert all(arm["terminal_reward"].grad is None for arm in payload["arms"])
    assert all(
        payload["arms"][arm]["old_logprob"].grad is None for arm in range(1, 8)
    )


@pytest.mark.parametrize(
    "indices,match",
    [
        ({arm: (0,) for arm in range(1, 7)}, "exactly arms 1..7"),
        ({arm: (() if arm == 3 else (0,)) for arm in range(1, 8)}, "zero contexts"),
        ({arm: ((0, 0) if arm == 4 else (0,)) for arm in range(1, 8)}, "repeats"),
        ({arm: ((99,) if arm == 5 else (0,)) for arm in range(1, 8)}, "outside"),
    ],
)
def test_sampled_objective_rejects_invalid_arm_context_geometry(indices, match):
    proposal = RowwiseWitnessProposal().eval()
    payload = _payload(proposal, [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match=match):
        pa.sampled_positive_advantage_objective(
            proposal, payload, indices, device=torch.device("cpu"),
        )


def test_positive_objective_rejects_nonfinite_or_invalid_weights():
    good = torch.zeros(2)
    with pytest.raises(pa.PositiveAdvantageError, match="log probabilities"):
        pa.clipped_positive_advantage_objective(
            torch.tensor([0.0, float("nan")]), good, 1.0,
        )
    with pytest.raises(pa.PositiveAdvantageError, match="importance ratio"):
        pa.clipped_positive_advantage_objective(
            torch.tensor([1000.0, 1000.0]), good, 1.0,
        )
    with pytest.raises(pa.PositiveAdvantageError, match="nonnegative"):
        pa.clipped_positive_advantage_objective(good, good, -1.0)
    with pytest.raises(ValueError, match="scalar"):
        pa.clipped_positive_advantage_objective(good, good, torch.ones(2))


def test_low_level_objective_detaches_behavior_and_positive_weight():
    current = torch.tensor([0.0, 0.1], requires_grad=True)
    old = torch.tensor([0.0, -0.1], requires_grad=True)
    weight = torch.tensor(2.0, requires_grad=True)
    loss, _, _ = pa.clipped_positive_advantage_objective(current, old, weight)
    loss.sum().backward()
    assert current.grad is not None
    assert bool((current.grad != 0.0).any())
    assert old.grad is None
    assert weight.grad is None


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_analytic_kl_forward_is_standard_and_seed_is_fully_detached(reduction):
    generator = torch.Generator().manual_seed(49666)
    current = torch.randn(4, 19, generator=generator, requires_grad=True)
    seed = torch.randn(4, 19, generator=generator, requires_grad=True)
    expected_rows = _standard_categorical_forward_kl(current, seed)
    expected = {
        "none": expected_rows,
        "sum": expected_rows.sum(),
        "mean": expected_rows.mean(),
    }[reduction]
    actual = pa.analytic_categorical_forward_kl(
        current, seed, reduction=reduction,
    )
    assert torch.equal(actual, expected)

    upstream = (
        torch.tensor([0.25, -0.5, 1.5, 2.0])
        if reduction == "none" else None
    )
    actual.backward(upstream)
    expected_gradient = current.detach().softmax(-1) - seed.detach().softmax(-1)
    if reduction == "none":
        expected_gradient = expected_gradient * upstream.unsqueeze(-1)
    elif reduction == "mean":
        expected_gradient = expected_gradient / current.shape[0]
    torch.testing.assert_close(
        current.grad, expected_gradient, rtol=2e-6, atol=2e-8,
    )
    assert seed.grad is None


def test_analytic_kl_is_shift_invariant_and_row_reduction_has_no_category_scale():
    generator = torch.Generator().manual_seed(7)
    current = torch.randn(3, 23, generator=generator, requires_grad=True)
    seed = torch.randn(3, 23, generator=generator)
    shifts_current = torch.tensor([[10.0], [-3.0], [0.25]])
    shifts_seed = torch.tensor([[-4.0], [2.0], [8.0]])

    base = pa.analytic_categorical_forward_kl(current, seed, reduction="none")
    shifted_current = (current.detach() + shifts_current).requires_grad_()
    shifted = pa.analytic_categorical_forward_kl(
        shifted_current, seed + shifts_seed, reduction="none",
    )
    torch.testing.assert_close(shifted, base, rtol=2e-5, atol=2e-6)
    base.sum().backward()
    shifted.sum().backward()
    torch.testing.assert_close(
        shifted_current.grad, current.grad, rtol=2e-5, atol=2e-7,
    )
    torch.testing.assert_close(
        current.grad.sum(-1), torch.zeros(3), rtol=0.0, atol=2e-7,
    )

    one_current = current.detach()[0:1].requires_grad_()
    one_seed = seed[0:1]
    one = pa.analytic_categorical_forward_kl(
        one_current, one_seed, reduction="none",
    )
    copies = 5
    duplicate_current = one_current.detach().expand(copies, -1).clone().requires_grad_()
    duplicate_seed = one_seed.expand(copies, -1).clone()
    duplicate_rows = pa.analytic_categorical_forward_kl(
        duplicate_current, duplicate_seed, reduction="none",
    )
    assert torch.equal(duplicate_rows, one.expand(copies))
    assert torch.equal(
        pa.analytic_categorical_forward_kl(
            duplicate_current, duplicate_seed, reduction="sum",
        ),
        one.sum() * copies,
    )
    assert torch.equal(
        pa.analytic_categorical_forward_kl(
            duplicate_current, duplicate_seed, reduction="mean",
        ),
        one.mean(),
    )


def test_external_horizon_average_retains_equal_horizon_scaling():
    generator = torch.Generator().manual_seed(11)
    horizons = [
        (torch.randn(2, 13, generator=generator), torch.randn(2, 13, generator=generator)),
        (torch.randn(5, 13, generator=generator), torch.randn(5, 13, generator=generator)),
        (torch.randn(3, 13, generator=generator), torch.randn(3, 13, generator=generator)),
    ]
    actual = torch.stack([
        pa.analytic_categorical_forward_kl(cur, seed, reduction="mean")
        for cur, seed in horizons
    ]).mean()
    expected = torch.stack([
        _standard_categorical_forward_kl(cur, seed).mean()
        for cur, seed in horizons
    ]).mean()
    pooled = torch.cat([
        _standard_categorical_forward_kl(cur, seed) for cur, seed in horizons
    ]).mean()
    assert torch.equal(actual, expected)
    assert not torch.allclose(actual, pooled)

    differentiable = [
        (cur.detach().requires_grad_(), seed) for cur, seed in horizons
    ]
    horizon_mean = torch.stack([
        pa.analytic_categorical_forward_kl(cur, seed, reduction="mean")
        for cur, seed in differentiable
    ]).mean()
    horizon_mean.backward()
    for cur, seed in differentiable:
        expected_gradient = (
            cur.detach().softmax(-1) - seed.detach().softmax(-1)
        ) / (len(differentiable) * cur.shape[0])
        torch.testing.assert_close(
            cur.grad, expected_gradient, rtol=2e-6, atol=2e-8,
        )


@pytest.mark.parametrize("shape", [(1, C.M), (3, C.M), (16, C.M)])
def test_bf16_cast_identity_has_exact_zero_value_and_bitwise_zero_gradient(shape):
    rows = torch.linspace(-8.0, 8.0, math.prod(shape), dtype=torch.float32)
    current = rows.reshape(shape).to(torch.bfloat16).requires_grad_()
    seed = current.detach().clone().requires_grad_()
    per_row = pa.analytic_categorical_forward_kl(
        current, seed, reduction="none",
    )
    assert torch.count_nonzero(per_row) == 0
    per_row.sum().backward()
    assert current.grad is not None
    assert torch.count_nonzero(current.grad) == 0
    assert seed.grad is None


@pytest.mark.parametrize(
    "current,seed,match",
    [
        (torch.zeros(2, 3), torch.zeros(2, 4), "same non-empty"),
        (torch.zeros(0, 3), torch.zeros(0, 3), "same non-empty"),
        (torch.tensor([[0.0, float("nan")]]), torch.zeros(1, 2), "nan/inf"),
        (torch.zeros(1, 2), torch.tensor([[0.0, float("inf")]]), "nan/inf"),
    ],
)
def test_analytic_kl_rejects_bad_shapes_and_nonfinite(current, seed, match):
    error = ValueError if "same non-empty" in match else pa.PositiveAdvantageError
    with pytest.raises(error, match=match):
        pa.analytic_categorical_forward_kl(current, seed)
    with pytest.raises(ValueError, match="unknown reduction"):
        pa.analytic_categorical_forward_kl(torch.zeros(1, 2), torch.zeros(1, 2), reduction="x")


def test_source_identity_and_provenance_are_isolated_and_nonlaunchable():
    identity = pa.core_source_identity()
    assert identity["scheme"] == "sha256(path-nul-sha256-nul)-v1"
    assert len(identity["sha256"]) == 64
    assert set(identity["files"]) == {
        "contracts.py",
        "loom/eval/outcome_recovery.py",
        "loom/heads/proposal.py",
        "loom/train/outcome_grpo.py",
        "loom/train/outcome_positive_advantage.py",
    }
    pa.assert_core_source_identity(identity)
    provenance = pa.core_provenance(identity)
    assert provenance["kind"] == pa.CORE_KIND
    assert provenance["execution_surface"] == "mathematical_core_only"
    assert provenance["trainer_wired"] is False
    assert provenance["config_present"] is False
    assert provenance["launcher_present"] is False
    assert provenance["candidate_or_evaluation_authority"] is False
    assert provenance["positive_advantage"]["scored_arms"] == list(range(1, 8))
    assert provenance["positive_advantage"]["arm0"].startswith("baseline_and_scale_only")
    assert provenance["positive_advantage"]["excluded_terms"] == [
        "switch_balance", "sparse_ce",
    ]
    assert provenance["categorical_reference"]["current_vjp"] == (
        "p_current_minus_p_seed"
    )

    sampled_source = inspect.getsource(pa.sampled_positive_advantage_objective)
    assert "stored_order_logprob" in sampled_source
    assert "proposal_switch_balance" not in sampled_source
    assert "sparse_ce" not in sampled_source
    assert not Path("configs/outcome_positive_advantage.yaml").exists()
