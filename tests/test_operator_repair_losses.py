"""Focused contracts for the staged operator-repair loss primitives."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from loom.losses.act import sparse_target_ce
from loom.losses.dyn import (
    dyn_loss,
    ln_cosine_distance,
    random_simplex_like,
    sample_within_trajectory_negative_set,
    sample_within_trajectory_negatives,
    sequential_rollout,
)


class AdditiveBank(nn.Module):
    """Tiny operator bank whose code selects a visible local residual."""

    def __init__(self) -> None:
        super().__init__()
        effects = torch.tensor(
            [
                [[2.0, -1.0, -1.0, 0.0, 0.0]],
                [[0.0, 2.0, -1.0, -1.0, 0.0]],
                [[0.0, 0.0, 2.0, -1.0, -1.0]],
                [[-1.0, 0.0, 0.0, 2.0, -1.0]],
            ]
        )
        self.effects = nn.Parameter(effects)  # (M=4,K=1,D=5)

    def step(self, c: Tensor, z: Tensor) -> Tensor:
        effect = torch.einsum("...m,mkd->...kd", c.to(self.effects.dtype), self.effects)
        return z + effect.to(z.dtype)


def _one_hot_sequence(batch: int = 2) -> Tensor:
    base = torch.eye(4)
    return base.unsqueeze(0).expand(batch, -1, -1).clone()


def _trajectory_states(bank: AdditiveBank, z0: Tensor, c_seq: Tensor) -> Tensor:
    return torch.stack(sequential_rollout(bank, z0, c_seq), dim=1)


def test_sparse_target_ce_is_fp32_sparse_ce_and_stops_target_gradient():
    logits = torch.tensor(
        [[1.0, -0.5, 0.25, 2.0], [-1.0, 0.1, 1.5, 0.3]],
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    target = torch.tensor(
        [[0.75, 0.0, 0.25, 0.0], [0.0, 0.4, 0.0, 0.6]],
        requires_grad=True,
    )
    temperature = 0.7

    per_item = sparse_target_ce(
        logits, target, temperature=temperature, reduction="none"
    )
    expected = -(
        target.detach().float()
        * F.log_softmax(logits.float() / temperature, dim=-1)
    ).sum(-1)

    assert per_item.dtype == torch.float32
    assert torch.equal(per_item, expected)
    assert torch.equal(
        sparse_target_ce(logits, target, temperature=temperature, reduction="sum"),
        expected.sum(),
    )
    mean = sparse_target_ce(logits, target, temperature=temperature)
    assert torch.equal(mean, expected.mean())

    mean.backward()
    assert target.grad is None
    assert logits.grad is not None
    # Dense softmax gives learning signal to atoms outside the sparse support.
    assert torch.all(logits.grad != 0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"temperature": 0.0}, "temperature"),
        ({"temperature": float("inf")}, "temperature"),
        ({"reduction": "batchmean"}, "reduction"),
    ],
)
def test_sparse_target_ce_rejects_invalid_options(kwargs, message):
    x = torch.randn(2, 4)
    with pytest.raises(ValueError, match=message):
        sparse_target_ce(x, torch.softmax(x, -1), **kwargs)


def test_sparse_target_ce_requires_matching_shapes():
    with pytest.raises(ValueError, match="same shape"):
        sparse_target_ce(torch.randn(2, 4), torch.randn(2, 3))


def test_negative_set_is_deterministic_detached_and_temporally_valid():
    c_seq = torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(2, 4, 4)
    c_seq.requires_grad_()
    first = sample_within_trajectory_negative_set(
        c_seq, 7, min_gap=2, generator=torch.Generator().manual_seed(1234)
    )
    second = sample_within_trajectory_negative_set(
        c_seq, 7, min_gap=2, generator=torch.Generator().manual_seed(1234)
    )

    assert first.shape == (2, 7, 4, 4)
    assert torch.equal(first, second)
    assert not first.requires_grad
    for batch in range(2):
        for candidate in range(7):
            for horizon in range(4):
                matches = [
                    source
                    for source in range(4)
                    if torch.equal(first[batch, candidate, horizon], c_seq[batch, source])
                ]
                assert matches
                assert all(abs(source - horizon) >= 2 for source in matches)


def test_negative_set_rejects_invalid_count_and_short_windows():
    c_seq = torch.randn(2, 4, 4)
    for count in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            sample_within_trajectory_negative_set(c_seq, count)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cannot supply"):
        sample_within_trajectory_negative_set(torch.randn(2, 2, 4), 2, min_gap=2)


def test_residual_effect_and_multi_negative_metrics_prefer_the_true_code():
    bank = AdditiveBank()
    z0 = torch.randn(2, 1, 5)
    c_seq = _one_hot_sequence(2)
    with torch.no_grad():
        targets = _trajectory_states(bank, z0, c_seq)
    contexts = torch.cat((z0[:, None], targets[:, :-1]), dim=1)
    target_prev = contexts.clone()
    # Rolling by two gives a valid >=2-away code at every horizon for H=4.
    c_neg_set = torch.roll(c_seq, shifts=2, dims=1)[:, None].expand(-1, 3, -1, -1)

    out = dyn_loss(
        bank,
        z0,
        c_seq,
        targets,
        negatives="none",
        z_contexts=contexts,
        z_target_prev=target_prev,
        c_neg_set=c_neg_set,
        effect_weight=1.0,
        contrastive_weight=0.25,
        contrastive_temperature=0.1,
    )

    assert out["state"].item() == pytest.approx(0.0, abs=1e-6)
    assert out["dyn"].item() == pytest.approx(out["state"].item(), abs=0.0)
    assert out["effect"].item() == pytest.approx(0.0, abs=1e-6)
    assert out["effect_gap"].item() > 0.1
    assert out["contrastive_top1"].item() == pytest.approx(1.0)
    assert out["contrastive"].item() > 0.0
    assert torch.allclose(
        out["loss"],
        out["state"] + out["effect"] + 0.25 * out["contrastive"],
        atol=1e-7,
    )
    for metric in (
        "state", "effect", "contrastive", "effect_gap", "contrastive_top1"
    ):
        assert not out[metric].requires_grad


def test_operator_repair_gradients_use_live_codes_but_detach_targets_and_negatives():
    torch.manual_seed(7)
    bank = AdditiveBank()
    b, h, m, k, d = 2, 4, 4, 1, 5
    z0 = torch.randn(b, k, d)
    c_seq = torch.rand(b, h, m, requires_grad=True)
    z_contexts = torch.randn(b, h, k, d, requires_grad=True)
    z_target_prev = torch.randn(b, h, k, d, requires_grad=True)
    z_targets = torch.randn(b, h, k, d, requires_grad=True)
    c_neg_set = torch.rand(b, 3, h, m, requires_grad=True)

    out = dyn_loss(
        bank,
        z0,
        c_seq,
        z_targets,
        negatives="none",
        state_weight=0.0,
        effect_weight=1.0,
        contrastive_weight=0.5,
        z_contexts=z_contexts,
        z_target_prev=z_target_prev,
        c_neg_set=c_neg_set,
    )
    out["loss"].backward()

    assert c_seq.grad is not None and c_seq.grad.abs().sum() > 0
    assert bank.effects.grad is not None and bank.effects.grad.abs().sum() > 0
    assert z_targets.grad is None
    assert z_target_prev.grad is None
    assert c_neg_set.grad is None


def _legacy_dyn_loss(
    bank: AdditiveBank,
    z0: Tensor,
    c_seq: Tensor,
    z_targets: Tensor,
    *,
    generator: torch.Generator,
) -> dict[str, Tensor]:
    """The pre-repair implementation, copied here to pin exact compatibility."""
    weights = (1.0, 0.5, 0.25, 0.125)
    tgt = z_targets.detach()
    z_hat = sequential_rollout(bank, z0, c_seq)
    d_pos = [
        ln_cosine_distance(z_hat[horizon], tgt[:, horizon])
        for horizon in range(c_seq.shape[1])
    ]
    loss_pos = sum(
        weights[horizon] * d_pos[horizon].mean()
        for horizon in range(c_seq.shape[1])
    )
    loss_neg = torch.zeros((), device=z0.device, dtype=loss_pos.dtype)
    c_neg = sample_within_trajectory_negatives(c_seq, 2, generator)
    z_neg = sequential_rollout(bank, z0, c_neg.to(c_seq.dtype))
    for horizon in range(c_seq.shape[1]):
        d_neg = ln_cosine_distance(z_neg[horizon], tgt[:, horizon])
        hinge = F.relu(0.1 - (d_neg - d_pos[horizon])).mean()
        loss_neg = loss_neg + weights[horizon] * hinge

    with torch.no_grad():
        c_rand = random_simplex_like(c_seq[:, 0], generator=generator)
        d_rand = ln_cosine_distance(bank.step(c_rand, z0), tgt[:, 0])
        delta_op = (d_rand - d_pos[0].detach()).mean()
    return {
        "loss": loss_pos + loss_neg,
        "dyn": loss_pos.detach(),
        "neg": loss_neg.detach(),
        "delta_op": delta_op,
        "cos_pos": 1.0 - d_pos[0].detach().mean(),
        "per_h": torch.stack([distance.detach().mean() for distance in d_pos]),
        "z_hat1": z_hat[0],
    }


def test_old_dyn_call_is_numerically_and_stochastically_identical():
    torch.manual_seed(19)
    bank = AdditiveBank()
    z0 = torch.randn(3, 1, 5)
    c_seq = torch.softmax(torch.randn(3, 4, 4), dim=-1)
    targets = torch.randn(3, 4, 1, 5)
    seed = 91827

    expected = _legacy_dyn_loss(
        bank, z0, c_seq, targets, generator=torch.Generator().manual_seed(seed)
    )
    actual = dyn_loss(
        bank,
        z0,
        c_seq,
        targets,
        generator=torch.Generator().manual_seed(seed),
    )

    for key in ("loss", "dyn", "neg", "delta_op", "cos_pos", "per_h", "z_hat1"):
        assert torch.equal(actual[key], expected[key]), key
    assert torch.equal(actual["state"], actual["dyn"])
    for key in ("effect", "contrastive", "effect_gap", "contrastive_top1"):
        assert actual[key].item() == 0.0


def test_operator_repair_requires_paired_contexts():
    bank = AdditiveBank()
    z0 = torch.randn(2, 1, 5)
    c_seq = _one_hot_sequence(2)
    targets = torch.randn(2, 4, 1, 5)
    with pytest.raises(ValueError, match="z_contexts and z_target_prev"):
        dyn_loss(bank, z0, c_seq, targets, effect_weight=1.0)
