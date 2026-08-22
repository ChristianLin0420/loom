"""Method and gradient-routing contracts for protected H/P/I arms."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

import contracts as C
from loom.train.loop import LoomModel, config_hash, read_config


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
ARM_CONFIGS = {
    arm: CONFIGS / f"r0a_protected_{arm.lower()}.yaml"
    for arm in ("H", "P", "I")
}


def _positive_grad(module: nn.Module) -> bool:
    return any(
        parameter.grad is not None and bool(parameter.grad.detach().abs().sum() > 0)
        for parameter in module.parameters()
    )


def _no_grad(module: nn.Module) -> bool:
    return all(parameter.grad is None for parameter in module.parameters())


class _Estimator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.7))

    def forward(self, feats: dict[str, Tensor], z_prev: Tensor | None) -> Tensor:
        obs = feats["views"].float().flatten(1).mean(-1)
        basis = torch.tensor([1.0, -0.5, 0.25, -1.25], device=obs.device)
        z = self.scale * obs[:, None, None] * basis[None, None]
        return z if z_prev is None else z + 0.15 * z_prev


class _Bank(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.2))
        self.register_buffer("axis", torch.linspace(-1.0, 1.0, C.M))
        self.register_buffer("basis", torch.tensor([1.0, -1.0, 0.5, -0.5]))

    def step(self, c: Tensor, z: Tensor) -> Tensor:
        code = (c.float() * self.axis.to(c)).sum(-1)
        effect = code[..., None, None] * self.basis.to(z)[None, None]
        return z + self.scale * effect


class _QDelta(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.linspace(-0.3, 0.3, C.M))

    def forward(
        self, z: Tensor, z_next: Tensor, *, return_logits: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        signal = (z.float().mean((1, 2)) - z_next.float().mean((1, 2)))[:, None]
        axis = torch.linspace(-0.8, 1.2, C.M, device=z.device)[None]
        logits = self.bias.to(z)[None] + signal * axis
        coeff = torch.softmax(logits.float(), -1).to(z)
        return (coeff, logits) if return_logits else coeff


class _QAction(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.linspace(0.35, -0.35, C.M))
        self.calls = 0

    def forward(
        self, action: Tensor, z: Tensor, *, return_logits: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        self.calls += 1
        action_signal = action.float().flatten(1).mean(-1, keepdim=True)
        belief_signal = z.float().mean((1, 2), keepdim=False)[:, None]
        axis = torch.linspace(-1.1, 0.9, C.M, device=z.device)[None]
        logits = self.bias.to(z)[None] + (action_signal + belief_signal) * axis
        coeff = torch.softmax(logits.float(), -1).to(z)
        return (coeff, logits) if return_logits else coeff


class _Proposal(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.linspace(-0.25, 0.25, C.M))

    def logits(self, z: Tensor, lang: Tensor) -> Tensor:
        signal = z.float().mean((1, 2))[:, None]
        language = lang.float().flatten(1).mean(-1, keepdim=True)
        axis = torch.linspace(-0.7, 1.0, C.M, device=z.device)[None]
        return self.bias.to(z)[None] + (signal + 0.1 * language) * axis


class _Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.4))

    def loss(
        self,
        proprio: Tensor,
        c: Tensor,
        action: Tensor,
        *,
        t: Tensor | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        del t, noise
        axis = torch.linspace(-1.0, 1.0, C.M, device=c.device)
        pred = self.scale * (c.float() * axis).sum(-1)
        pred = pred + 0.05 * proprio.float().mean(-1)
        target = action.float().flatten(1).mean(-1)
        return (pred - target).square().mean()


class _ZeroDecoder(_Decoder):
    def loss(self, *args, **kwargs) -> Tensor:
        del args, kwargs
        return self.scale.square() * 0.0


def _loss_cfg(
    *,
    isolate: bool,
    dyn: bool = True,
    act: bool = False,
    dual: bool = False,
    balance: bool = False,
) -> dict:
    return {
        "optim": {"ema_tau": C.EMA_TAU},
        "losses": {
            "dyn": {
                "enabled": dyn,
                "weight": 1.0,
                "coeff_source": "q_delta",
                "isolate_estimator_gradients": isolate,
                "negatives": "within_trajectory",
                "neg_weight": 0.0,
                "state_weight": 1.0,
                "effect_weight": 1.0,
                "contrastive_weight": 0.0,
            },
            "act": {
                "enabled": act,
                "weight": 1.0,
                "align_to": "q_a",
                "align_mode": "sparse_ce",
                "align_weight": 0.1,
                "decode_from": (
                    "dual_q_action_proposal" if dual else "q_action"
                ),
            },
            "proposal": {
                "enabled": dual,
                "weight": 1.0,
                "mode": "sparse_ce",
                "detach_belief": True,
            },
            "balance": {
                "enabled": balance,
                "weight": 0.02,
                "mode": "per_head",
                "head_weights": {"q_delta": 0.75, "q_action": 0.25},
            },
            "potential": {"enabled": False},
            "grpo": {"enabled": False},
        },
    }


def _model(cfg: dict, *, zero_decoder: bool = False) -> LoomModel:
    return LoomModel(
        estimator=_Estimator(),
        bank=_Bank(),
        q_delta=_QDelta(),
        q_action=nn.ModuleDict({"libero_franka": _QAction()}),
        decoder=nn.ModuleDict({
            "libero_franka": _ZeroDecoder() if zero_decoder else _Decoder(),
        }),
        proposal=_Proposal(),
        cfg=cfg,
    )


def _window(batch: int = 4) -> dict:
    feats = []
    lang = torch.linspace(-0.2, 0.4, batch * 3).reshape(batch, 3, 1)
    for horizon in range(C.DEPTH + 1):
        views = torch.linspace(
            0.1 + 0.03 * horizon, 0.5 + 0.03 * horizon, batch,
        ).reshape(batch, 1, 1, 1)
        proprio = torch.linspace(-0.4, 0.6, batch * 7).reshape(batch, 7)
        feats.append({"views": views, "proprio": proprio, "lang": lang})
    actions = torch.linspace(
        -0.7, 0.8, batch * C.DEPTH * C.H_OP * 7,
    ).reshape(batch, C.DEPTH, C.H_OP, 7)
    return {
        "feats": feats,
        "lang": lang,
        "actions": actions,
        "embodiment": "libero_franka",
    }


def test_three_arms_share_one_fresh_threshold_free_formal_protocol() -> None:
    configs = {arm: read_config(path) for arm, path in ARM_CONFIGS.items()}
    hashes = {config_hash(config) for config in configs.values()}
    assert len(hashes) == 3
    assert all(len(value) == 16 for value in hashes)

    for arm, cfg in configs.items():
        assert cfg["run"]["project"] == "loom-r0-protected-arms"
        assert cfg["run"]["seed"] == 0
        assert cfg["run"]["steps"] == 32_000
        assert cfg["run"]["fresh_start_required"] is True
        assert cfg["run"]["require_online_wandb"] is True
        assert cfg["run"]["boundary_policy"] == "fixed_max_updates"
        assert cfg["optim"]["spike_mult"] == 0
        assert cfg["model"]["use_stubs"] is False
        assert cfg["data"]["sampling"] == "weighted_suite_task"
        assert cfg["data"]["suite_weights"] == {
            "libero_spatial": 0.2,
            "libero_object": 0.2,
            "libero_goal": 0.2,
            "libero_10": 0.4,
        }
        assert cfg["data"]["recurrent_prefix_choices"] == [0, 4, 8, 12]
        assert cfg["protected_arm"]["id"] == arm
        protocol = cfg["protected_protocol"]
        assert protocol["fixed_endpoint_update"] == 32_000
        assert protocol["evaluation_is_unconditional"] is True
        assert protocol["evaluation_episodes"] == 1_200
        assert protocol["health_thresholds_control_execution"] is False
        assert protocol["outcome_threshold_applied"] is False
        assert "direct_formal" not in cfg
        assert "evaluation_gate" not in cfg

        reference = cfg["reference_action_policy"]
        assert reference["source_config_hash"] == "d030206d56a71718"
        assert reference["checkpoint_step"] == 32_000
        assert reference["checkpoint_sha256"] == (
            "eddcc36d94dc48b9031acbcdaea116b2a1693c8b9e357f96e2573da36c9039b6"
        )
        assert (reference["successes"], reference["episodes"]) == (550, 1_200)


def test_h_preserves_the_prior_dual_code_loss_semantics_exactly() -> None:
    prior = read_config(CONFIGS / "r0a_dual_code.yaml")
    history = read_config(ARM_CONFIGS["H"])

    assert history["losses"]["dyn"] == {
        **prior["losses"]["dyn"],
        "coeff_source": "q_delta",
        "isolate_estimator_gradients": False,
    }
    assert history["losses"]["act"] == {
        **prior["losses"]["act"],
        "align_mode": "mse",
        "align_weight": 1.0,
    }
    assert history["losses"]["proposal"] == prior["losses"]["proposal"]
    assert history["losses"]["balance"] == {
        **prior["losses"]["balance"], "mode": "pooled",
    }
    for name in ("potential", "grpo"):
        assert history["losses"][name] == prior["losses"][name]
    assert history["train_modules"] == prior["train_modules"]


def test_p_and_i_are_staged_q_delta_repairs_with_one_isolation_delta() -> None:
    protected = read_config(ARM_CONFIGS["P"])
    isolated = read_config(ARM_CONFIGS["I"])

    for cfg in (protected, isolated):
        dyn = cfg["losses"]["dyn"]
        assert dyn["coeff_source"] == "q_delta"
        assert (dyn["start_update"], dyn["ramp_updates"]) == (2_001, 500)
        assert dyn["effect_weight"] == pytest.approx(1.0)
        assert dyn["contrastive_weight"] == pytest.approx(0.25)
        assert cfg["losses"]["act"]["align_to"] == "q_a"
        assert cfg["losses"]["act"]["align_mode"] == "sparse_ce"
        assert cfg["losses"]["act"]["align_weight"] == pytest.approx(0.1)
        assert cfg["losses"]["balance"]["head_weights"] == {
            "q_delta": 0.75, "q_action": 0.25,
        }

    assert protected["losses"]["dyn"]["isolate_estimator_gradients"] is False
    assert isolated["losses"]["dyn"]["isolate_estimator_gradients"] is True
    normalized_i = copy.deepcopy(isolated)
    normalized_i["run"]["name"] = protected["run"]["name"]
    normalized_i["losses"]["dyn"]["isolate_estimator_gradients"] = False
    normalized_i["protected_arm"] = protected["protected_arm"]
    assert normalized_i == protected

    view = LoomModel.__new__(LoomModel)
    view.loss_cfg = protected["losses"]
    assert LoomModel._loss_scale(view, "dyn", 1_999) == 0.0
    assert LoomModel._loss_scale(view, "dyn", 2_000) == pytest.approx(1 / 500)
    assert LoomModel._loss_scale(view, "dyn", 2_499) == 1.0


def test_p_dynamics_uses_q_delta_and_never_calls_or_updates_q_action() -> None:
    model = _model(_loss_cfg(isolate=False))
    loss, metrics = model.compute_losses(_window(), step=2_600, rank=0, seed=5)
    loss.backward()

    assert metrics["dyn/estimator_isolated"] == 0.0
    assert model.q_action["libero_franka"].calls == 0
    assert _positive_grad(model.estimator)
    assert _positive_grad(model.bank)
    assert _positive_grad(model.q_delta)
    assert _no_grad(model.q_action)
    assert _no_grad(model.decoder)
    assert _no_grad(model.proposal)


def test_i_isolates_all_q_delta_bank_objective_edges_from_estimator() -> None:
    isolated = _model(
        _loss_cfg(isolate=True, act=True, balance=True), zero_decoder=True,
    )
    loss, metrics = isolated.compute_losses(
        _window(), step=2_600, rank=0, seed=7,
    )
    loss.backward()

    assert metrics["dyn/estimator_isolated"] == 1.0
    # Zero-weight graph bookkeeping may materialize an exact-zero .grad tensor;
    # isolation means the gradient value is zero, not that autograd must omit it.
    assert not _positive_grad(isolated.estimator)
    assert _positive_grad(isolated.bank)
    assert _positive_grad(isolated.q_delta)
    # q_action balance remains live on the head itself, but its dedicated
    # balance forward consumes a detached belief and therefore cannot reach E.
    assert _positive_grad(isolated.q_action)

    attached = _model(
        _loss_cfg(isolate=False, act=True, balance=True), zero_decoder=True,
    )
    attached_loss, _ = attached.compute_losses(
        _window(), step=2_600, rank=0, seed=7,
    )
    attached_loss.backward()
    assert _positive_grad(attached.estimator)


def test_i_preserves_direct_action_gradients_into_estimator() -> None:
    model = _model(_loss_cfg(isolate=True, dyn=False, act=True, dual=True))
    loss, _ = model.compute_losses(_window(), step=100, rank=0, seed=11)
    loss.backward()

    assert _positive_grad(model.estimator)
    assert _positive_grad(model.q_action)
    assert _positive_grad(model.decoder)
    assert _positive_grad(model.proposal)
    # Sparse q_delta <- stopgrad(q_action) alignment remains live, but its
    # belief input is detached and therefore cannot account for E's gradient.
    assert _positive_grad(model.q_delta)
    assert _no_grad(model.bank)


def test_isolation_option_is_fail_closed_and_q_delta_only() -> None:
    cfg = _loss_cfg(isolate=False)
    cfg["losses"]["dyn"]["isolate_estimator_gradients"] = "true"
    with pytest.raises(ValueError, match="must be a boolean"):
        _model(cfg)

    cfg = _loss_cfg(isolate=True)
    cfg["losses"]["dyn"]["coeff_source"] = "q_action"
    with pytest.raises(ValueError, match="requires coeff_source='q_delta'"):
        _model(cfg)
