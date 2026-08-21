"""Focused contracts for the prospective dual-code R0-A action objective."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

import contracts as C
from loom.heads.proposal import argmax_coeff, argmax_coeff_dense_st
from loom.train.loop import LoomModel, build_model, read_config


ROOT = Path(__file__).resolve().parents[1]


class _Estimator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(0.15))

    def forward(self, feats: dict[str, Tensor], z_prev: Tensor | None) -> Tensor:
        b = feats["views"].shape[0]
        obs = feats["views"].float().mean(tuple(range(1, feats["views"].ndim)))
        z = obs.reshape(b, 1, 1) + self.offset
        return z if z_prev is None else z + 0.1 * z_prev


class _UnusedBank(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))


class _QDelta(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.linspace(-0.2, 0.2, C.M))
        self.coeffs: list[Tensor] = []

    def forward(
        self, z: Tensor, z_next: Tensor, *, return_logits: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        signal = 0.07 * z.float().mean((1, 2), keepdim=False).unsqueeze(-1)
        future = 0.03 * z_next.float().mean((1, 2), keepdim=False).unsqueeze(-1)
        axis = torch.linspace(-1.0, 1.0, C.M, device=z.device)
        logits = self.bias.to(z).unsqueeze(0) + (signal + future) * axis
        c = torch.softmax(logits.float(), dim=-1).to(z.dtype)
        self.coeffs.append(c)
        return (c, logits) if return_logits else c


class _QAction(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.linspace(0.25, -0.25, C.M))
        self.coeffs: list[Tensor] = []

    def forward(
        self, action: Tensor, z: Tensor, *, return_logits: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        action_signal = action.float().flatten(1).mean(-1, keepdim=True)
        belief_signal = z.float().mean((1, 2), keepdim=False).unsqueeze(-1)
        axis = torch.linspace(-0.8, 1.2, C.M, device=z.device)
        logits = self.bias.to(z).unsqueeze(0) + (
            0.11 * action_signal + 0.05 * belief_signal
        ) * axis
        c = torch.softmax(logits.float(), dim=-1).to(z.dtype)
        self.coeffs.append(c)
        return (c, logits) if return_logits else c


class _Proposal(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.linspace(-1.0, 1.0, C.M))
        self.logged_logits: list[Tensor] = []

    def logits(self, z: Tensor, lang: Tensor) -> Tensor:
        belief = z.float().mean((1, 2), keepdim=False).unsqueeze(-1)
        language = lang.float().flatten(1).mean(-1, keepdim=True)
        axis = torch.linspace(-0.6, 0.9, C.M, device=z.device)
        logits = self.bias.to(z).unsqueeze(0) + (
            0.09 * belief + 0.02 * language
        ) * axis
        self.logged_logits.append(logits)
        return logits


class _ConstantProposal(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logit = nn.Parameter(torch.linspace(-0.9, 0.8, C.M))
        self.logged_logits: list[Tensor] = []

    def logits(self, z: Tensor, lang: Tensor) -> Tensor:
        del lang
        logits = self.logit.to(z).unsqueeze(0).expand(z.shape[0], -1)
        self.logged_logits.append(logits)
        return logits


class _MatchingQAction(nn.Module):
    def __init__(self, logit: Tensor) -> None:
        super().__init__()
        self.register_buffer("logit", logit.detach().clone())
        self.coeffs: list[Tensor] = []

    def forward(
        self, action: Tensor, z: Tensor, *, return_logits: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        del action
        logits = self.logit.to(z).unsqueeze(0).expand(z.shape[0], -1)
        c = argmax_coeff(logits, C.TOPK, C.M, straight_through=False)
        self.coeffs.append(c)
        return (c, logits) if return_logits else c


class _RecordingCFMDecoder(nn.Module):
    """A differentiable scalar stand-in that records the paired CFM inputs."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.7))
        self.calls: list[dict[str, Tensor]] = []

    def loss(
        self,
        proprio: Tensor,
        c: Tensor,
        action: Tensor,
        *,
        t: Tensor | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        assert t is not None
        assert noise is not None
        axis = torch.linspace(-1.0, 1.0, c.shape[-1], device=c.device)
        code = (c.float() * axis).sum(-1)
        state = 0.04 * proprio.float().mean(-1)
        source = 0.03 * noise.float().flatten(1).mean(-1)
        target = action.float().flatten(1).mean(-1)
        pred = self.scale.float() * code + state + source + 0.02 * t.float()
        value = (pred - target).square().mean()
        self.calls.append({"c": c, "t": t, "noise": noise, "loss": value})
        return value


class _LegacyDecoder(nn.Module):
    """The pre-dual three-argument call surface, used for compatibility checks."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.55))
        self.calls: list[Tensor] = []

    def loss(self, proprio: Tensor, c: Tensor, action: Tensor) -> Tensor:
        self.calls.append(c)
        axis = torch.linspace(-0.7, 1.1, c.shape[-1], device=c.device)
        pred = self.scale.float() * (c.float() * axis).sum(-1)
        pred = pred + 0.03 * proprio.float().mean(-1)
        target = action.float().flatten(1).mean(-1)
        return (pred - target).square().mean()


def _cfg(mode: str | None = "dual_q_action_proposal") -> dict:
    act = {"enabled": True, "weight": 1.0, "align_to": "q_a"}
    if mode is not None:
        act["decode_from"] = mode
    return {
        "optim": {"ema_tau": C.EMA_TAU},
        "losses": {
            "dyn": {"enabled": False},
            "act": act,
            "proposal": {"enabled": False},
            "balance": {"enabled": False},
            "potential": {"enabled": False},
            "grpo": {"enabled": False},
        },
    }


def _model(
    mode: str | None = "dual_q_action_proposal",
    *,
    decoder: nn.Module | None = None,
) -> LoomModel:
    return LoomModel(
        estimator=_Estimator(),
        bank=_UnusedBank(),
        q_delta=_QDelta(),
        q_action=nn.ModuleDict({"libero_franka": _QAction()}),
        decoder=nn.ModuleDict({
            "libero_franka": decoder or _RecordingCFMDecoder(),
        }),
        proposal=_Proposal(),
        cfg=_cfg(mode),
    )


def _window(batch: int = 3) -> dict:
    feats = []
    for h in range(C.DEPTH + 1):
        feats.append({
            "views": torch.full((batch, 1, 1, 1), 0.1 * (h + 1)),
            "proprio": torch.linspace(-0.3, 0.4, batch * 7).reshape(batch, 7),
            "lang": torch.linspace(-0.2, 0.3, batch * 2).reshape(batch, 2, 1),
        })
    actions = torch.linspace(
        -0.5, 0.6, batch * C.DEPTH * C.H_OP * 7,
    ).reshape(batch, C.DEPTH, C.H_OP, 7)
    return {
        "feats": feats,
        "actions": actions,
        "lang": feats[0]["lang"],
        "embodiment": "libero_franka",
        "src_fps": C.EMBODIMENTS["libero_franka"].env_fps,
    }


def _grad(value: Tensor, parameter: nn.Parameter) -> Tensor | None:
    return torch.autograd.grad(
        value, parameter, retain_graph=True, allow_unused=True,
    )[0]


def test_dual_mode_parser_and_fresh_recipe_are_explicit() -> None:
    assert _model(None).act_decode_from == "q_action"
    assert _model("q_action").act_decode_from == "q_action"
    assert _model("proposal").act_decode_from == "proposal"
    assert _model().act_decode_from == "dual_q_action_proposal"
    with pytest.raises(ValueError, match="decode_from"):
        _model("teacher_and_maybe_student")
    with pytest.raises(TypeError, match="stub proposal mode is unsupported"):
        build_model({
            "model": {"use_stubs": True},
            "data": {"embodiments": ["libero_franka"]},
            "losses": {
                "act": {"enabled": True,
                        "decode_from": "dual_q_action_proposal"},
            },
        })

    cfg = read_config(ROOT / "configs" / "r0a_dual_code.yaml")
    assert cfg["run"]["name"] == "r0a_dual_code"
    assert cfg["run"]["steps"] == 32000
    assert cfg["model"]["use_stubs"] is False
    assert cfg["losses"]["act"] == {
        "enabled": True,
        "weight": 1.0,
        "align_to": "q_a",
        "decode_from": "dual_q_action_proposal",
    }
    assert cfg["train_modules"] == [
        "estimator", "bank", "q_delta", "q_action", "decoder", "proposal",
    ]
    assert cfg["losses"]["potential"]["enabled"] is False
    assert cfg["losses"]["grpo"]["enabled"] is False


def test_dense_ste_is_exact_forward_and_has_only_full_softmax_vjp() -> None:
    logits = torch.tensor([
        [-1.2, 0.3, 1.1, -0.4, 0.8, 0.2],
        [0.7, -0.1, 0.4, 1.3, -0.9, 0.0],
    ], dtype=torch.float32, requires_grad=True)
    upstream = torch.tensor([
        [0.2, -0.7, 0.5, 1.1, -0.3, 0.9],
        [-0.6, 0.4, 0.8, -0.2, 1.0, -0.5],
    ])
    out = argmax_coeff_dense_st(logits, topk=2, m=6)
    hard = argmax_coeff(logits.detach(), topk=2, m=6)
    assert torch.equal(out.detach(), hard)

    got = torch.autograd.grad((out * upstream).sum(), logits)[0]
    probability = torch.softmax(logits.detach(), dim=-1)
    expected = probability * (
        upstream - (probability * upstream).sum(-1, keepdim=True)
    )
    torch.testing.assert_close(got, expected, rtol=1e-6, atol=1e-7)

    bf16_logits = logits.detach().to(torch.bfloat16).requires_grad_(True)
    bf16_out = argmax_coeff_dense_st(bf16_logits, topk=2, m=6)
    bf16_hard = argmax_coeff(bf16_logits.detach(), topk=2, m=6)
    assert torch.equal(bf16_out.detach(), bf16_hard)


def test_legacy_action_modes_remain_exact_and_use_three_argument_decoder() -> None:
    window = _window()
    implicit = _model(None, decoder=_LegacyDecoder())
    explicit = _model("q_action", decoder=_LegacyDecoder())
    explicit.load_state_dict(implicit.state_dict())

    implicit_loss, implicit_metrics = implicit.compute_losses(
        window, step=9, rank=0, seed=3,
    )
    explicit_loss, explicit_metrics = explicit.compute_losses(
        window, step=9, rank=0, seed=3,
    )
    torch.testing.assert_close(implicit_loss, explicit_loss, rtol=0, atol=0)
    assert implicit_metrics == explicit_metrics
    assert len(implicit.decoder["libero_franka"].calls) == C.DEPTH
    assert len(explicit.decoder["libero_franka"].calls) == C.DEPTH
    assert implicit.proposal.logged_logits == []
    assert explicit.proposal.logged_logits == []

    implicit_loss.backward()
    explicit_loss.backward()
    for (name_a, parameter_a), (name_b, parameter_b) in zip(
        implicit.named_parameters(), explicit.named_parameters(), strict=True,
    ):
        assert name_a == name_b
        if parameter_a.grad is None or parameter_b.grad is None:
            assert parameter_a.grad is None and parameter_b.grad is None
        else:
            torch.testing.assert_close(
                parameter_a.grad, parameter_b.grad, rtol=0, atol=0,
            )

    deployed = _model("proposal", decoder=_LegacyDecoder())
    deployed.compute_losses(window, step=9, rank=0, seed=3)
    assert len(deployed.decoder["libero_franka"].calls) == C.DEPTH
    assert len(deployed.proposal.logged_logits) == C.DEPTH


def test_dual_action_formula_shared_randomness_and_exact_gradient_routes() -> None:
    model = _model()
    window = _window()
    total, metrics = model.compute_losses(window, step=17, rank=2, seed=11)

    decoder = model.decoder["libero_franka"]
    q_action = model.q_action["libero_franka"]
    assert len(decoder.calls) == 2 * C.DEPTH
    for h in range(C.DEPTH):
        teacher = decoder.calls[2 * h]
        deployed = decoder.calls[2 * h + 1]
        assert teacher["t"] is deployed["t"]
        assert teacher["noise"] is deployed["noise"]
        torch.testing.assert_close(teacher["t"], deployed["t"], rtol=0, atol=0)
        torch.testing.assert_close(
            teacher["noise"], deployed["noise"], rtol=0, atol=0,
        )
        expected = argmax_coeff(
            model.proposal.logged_logits[h], C.TOPK, C.M,
            straight_through=False,
        )
        torch.testing.assert_close(deployed["c"], expected, rtol=0, atol=0)

    teacher_mean = sum(
        call["loss"] for call in decoder.calls[0::2]
    ) / C.DEPTH
    deployed_mean = sum(
        call["loss"] for call in decoder.calls[1::2]
    ) / C.DEPTH
    alignment = sum(
        ((c_delta - c_action.detach()).square().sum(-1).mean())
        for c_delta, c_action in zip(model.q_delta.coeffs, q_action.coeffs)
    ) / C.DEPTH
    expected_total = 0.5 * teacher_mean + 0.5 * deployed_mean + alignment
    torch.testing.assert_close(total, expected_total, rtol=1e-6, atol=1e-7)
    assert metrics["act/decode"] == pytest.approx(float(
        (0.5 * teacher_mean + 0.5 * deployed_mean).detach()
    ))
    assert metrics["act/decode_teacher"] == pytest.approx(float(teacher_mean.detach()))
    assert metrics["act/decode_deploy"] == pytest.approx(float(deployed_mean.detach()))
    assert metrics["act/decode_gap"] == pytest.approx(float(
        (deployed_mean - teacher_mean).detach()
    ))
    assert metrics["act/align"] == pytest.approx(float(alignment.detach()))

    # Branch-local VJPs prove the semantic/deployment routes independently.
    assert _grad(teacher_mean, q_action.bias) is not None
    assert _grad(teacher_mean, model.proposal.bias) is None
    assert _grad(deployed_mean, model.proposal.bias) is not None
    assert _grad(deployed_mean, q_action.bias) is None
    assert _grad(alignment, q_action.bias) is None
    q_delta_align = _grad(alignment, model.q_delta.bias)
    assert q_delta_align is not None and bool(torch.count_nonzero(q_delta_align))
    for branch in (teacher_mean, deployed_mean):
        grad_decoder = _grad(branch, decoder.scale)
        grad_estimator = _grad(branch, model.estimator.offset)
        assert grad_decoder is not None and bool(torch.count_nonzero(grad_decoder))
        assert grad_estimator is not None and bool(torch.count_nonzero(grad_estimator))

    total.backward()
    for parameter in (
        model.estimator.offset,
        model.q_delta.bias,
        q_action.bias,
        decoder.scale,
        model.proposal.bias,
    ):
        assert parameter.grad is not None
        assert bool(torch.count_nonzero(parameter.grad))
    assert model.bank.weight.grad is None
    assert all(parameter.grad is None for parameter in model.ema.parameters())


def test_matching_teacher_and_deployed_codes_preserve_one_flow_scale() -> None:
    model = _model()
    constant_proposal = _ConstantProposal()
    model.proposal = constant_proposal
    model.q_action["libero_franka"] = _MatchingQAction(constant_proposal.logit)

    _, metrics = model.compute_losses(_window(), step=4, rank=0, seed=13)
    calls = model.decoder["libero_franka"].calls
    for h in range(C.DEPTH):
        teacher, deployed = calls[2 * h], calls[2 * h + 1]
        torch.testing.assert_close(teacher["c"], deployed["c"], rtol=0, atol=0)
        torch.testing.assert_close(
            teacher["loss"], deployed["loss"], rtol=0, atol=0,
        )
    assert metrics["act/decode"] == metrics["act/decode_teacher"]
    assert metrics["act/decode"] == metrics["act/decode_deploy"]
    assert metrics["act/decode_gap"] == 0.0


def test_dual_draws_exactly_one_noise_and_time_per_horizon(monkeypatch) -> None:
    calls = {"randn": 0, "rand": 0}
    original_randn = torch.randn
    original_rand = torch.rand

    def counted_randn(*args, **kwargs):
        calls["randn"] += 1
        return original_randn(*args, **kwargs)

    def counted_rand(*args, **kwargs):
        calls["rand"] += 1
        return original_rand(*args, **kwargs)

    monkeypatch.setattr(torch, "randn", counted_randn)
    monkeypatch.setattr(torch, "rand", counted_rand)
    _model().compute_losses(_window(), step=8, rank=0, seed=21)
    assert calls == {"randn": C.DEPTH, "rand": C.DEPTH}


def test_dual_randomness_is_step_keyed_and_action_free_is_graph_safe() -> None:
    window = _window()
    first = _model()
    second = _model()
    second.load_state_dict(first.state_dict())

    loss_a, _ = first.compute_losses(window, step=29, rank=1, seed=5)
    loss_b, _ = second.compute_losses(window, step=29, rank=1, seed=5)
    calls_a = first.decoder["libero_franka"].calls
    calls_b = second.decoder["libero_franka"].calls
    torch.testing.assert_close(loss_a, loss_b, rtol=0, atol=0)
    for a, b in zip(calls_a, calls_b):
        torch.testing.assert_close(a["t"], b["t"], rtol=0, atol=0)
        torch.testing.assert_close(a["noise"], b["noise"], rtol=0, atol=0)

    third = _model()
    third.load_state_dict(first.state_dict())
    third.compute_losses(window, step=30, rank=1, seed=5)
    calls_c = third.decoder["libero_franka"].calls
    assert any(not torch.equal(a["noise"], c["noise"])
               for a, c in zip(calls_a, calls_c))

    action_free = _window()
    action_free["actions"] = None
    empty = _model()
    loss_empty, metrics = empty.compute_losses(
        action_free, step=29, rank=1, seed=5,
    )
    assert loss_empty.item() == 0.0
    assert metrics["loss"] == 0.0
    assert empty.decoder["libero_franka"].calls == []
    assert empty.proposal.logged_logits == []
