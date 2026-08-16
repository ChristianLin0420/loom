"""
LOOM — shape-correct stubs.

FROZEN AFTER PHASE 0. Do not edit.

Every Protocol in contracts.py has a random-output implementation here with
correct shapes and correct invariants. This is what lets all six teams develop
and test independently on day one, with zero cross-team dependencies.

StubBank genuinely satisfies the contractivity and bias bounds, so downstream
tests written against it are meaningful rather than vacuous.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from contracts import (
    B_MAX, D, DEPTH, EMBODIMENTS, H_OP, K, M, RHO, TOPK,
    Bank, Decoder, Estimator, ObsFeats, Policy, Potential,
    Proposal, QAction, QDelta, env_steps_per_segment,
)

__all__ = [
    "StubEstimator", "StubBank", "StubQDelta", "StubQAction",
    "StubDecoder", "StubProposal", "StubPotential", "StubPolicy",
    "make_obs_feats", "make_window", "sparse_simplex",
]


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def sparse_simplex(*batch: int, device="cpu", dtype=torch.float32) -> Tensor:
    """Random (*batch, M) points with exactly TOPK nonzeros summing to 1."""
    n = 1
    for b in batch:
        n *= b
    idx = torch.stack([torch.randperm(M, device=device)[:TOPK] for _ in range(n)])
    w = torch.rand(n, TOPK, device=device)
    w = w / w.sum(-1, keepdim=True)
    c = torch.zeros(n, M, device=device).scatter_(1, idx, w)
    return c.reshape(*batch, M).to(dtype)


def make_obs_feats(b: int = 2, v: int = 2, p: int = 196, f: int = 1152,
                   dof: int = 7, l: int = 16, device="cpu") -> ObsFeats:
    """Random ObsFeats with plausible dimensions."""
    return ObsFeats(
        views=torch.randn(b, v, p, f, device=device),
        proprio=torch.randn(b, dof, device=device),
        lang=torch.randn(b, l, f, device=device),
    )


def make_window(b: int = 2, embodiment: str = "libero_franka", device="cpu",
                action_free: bool = False) -> dict:
    """Random TransitionWindow with N_STATES boundary states."""
    spec = EMBODIMENTS[embodiment]
    feats = [make_obs_feats(b=b, v=spec.n_views, dof=spec.dof, device=device)
             for _ in range(DEPTH + 1)]
    actions = None if action_free else torch.randn(b, DEPTH, H_OP, spec.dof, device=device)
    return {
        "feats": feats,
        "actions": actions,
        "lang": feats[0]["lang"],
        "embodiment": embodiment,
        "src_fps": spec.env_fps,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  STUBS
# ═══════════════════════════════════════════════════════════════════════════

class StubEstimator(nn.Module):
    """Random belief, correct shape. Satisfies contracts.Estimator."""

    def forward(self, feats: ObsFeats, z_prev: Tensor | None) -> Tensor:
        b = feats["views"].shape[0]
        dev = feats["views"].device
        z = torch.randn(b, K, D, device=dev, dtype=feats["views"].dtype)
        return z if z_prev is None else 0.5 * z + 0.5 * z_prev


class StubBank(nn.Module):
    """Real 2x2 rotation-decay bank. Bounds genuinely hold.

    Deliberately mirrors the intended implementation so that tests written
    against it stay valid when Team B's real bank lands.
    """

    def __init__(self) -> None:
        super().__init__()
        self.log_r = nn.Parameter(torch.randn(M, K, D // 2) * 0.5)
        self.omega = nn.Parameter(torch.rand(M, K, D // 2) * 2 * torch.pi)
        self.b_raw = nn.Parameter(torch.randn(M, K, D) * 0.1)

    def _lam(self) -> tuple[Tensor, Tensor]:
        r = RHO * torch.sigmoid(self.log_r)
        return r * torch.cos(self.omega), r * torch.sin(self.omega)

    def _bias_bank(self) -> Tensor:
        n = self.b_raw.flatten(1).norm(dim=1).clamp(min=B_MAX).view(M, 1, 1)
        return B_MAX * self.b_raw / n

    def mix(self, c: Tensor) -> tuple[Tensor, Tensor]:
        a_bank, b_bank = self._lam()
        c = c.to(a_bank.dtype)
        return (torch.einsum('...m,mkj->...kj', c, a_bank),
                torch.einsum('...m,mkj->...kj', c, b_bank))

    def bias(self, c: Tensor) -> Tensor:
        bank = self._bias_bank()
        return torch.einsum('...m,mkd->...kd', c.to(bank.dtype), bank)

    def step(self, c: Tensor, z: Tensor) -> Tensor:
        a, b = self.mix(c)
        zr = z.reshape(*z.shape[:-1], D // 2, 2)
        x, y = zr[..., 0], zr[..., 1]
        out = torch.stack([a * x - b * y, b * x + a * y], dim=-1)
        return out.reshape(*z.shape) + self.bias(c)

    def rollout(self, c_seq: Tensor, z: Tensor) -> Tensor:
        b, n = c_seq.shape[0], c_seq.shape[1]
        z = z.unsqueeze(1).expand(b, n, K, D)
        for d in range(DEPTH):
            z = self.step(c_seq[:, :, d], z)
        return z


class StubQDelta(nn.Module):
    """Random simplex coefficients. Satisfies contracts.QDelta."""

    def forward(self, z_t: Tensor, z_next: Tensor) -> Tensor:
        return sparse_simplex(z_t.shape[0], device=z_t.device, dtype=z_t.dtype)


class StubQAction(nn.Module):
    """Random simplex coefficients. Satisfies contracts.QAction."""

    def forward(self, a_seg: Tensor, z: Tensor) -> Tensor:
        return sparse_simplex(z.shape[0], device=z.device, dtype=z.dtype)


class StubDecoder(nn.Module):
    """Random action segment of width H_OP. Satisfies contracts.Decoder."""

    def __init__(self, embodiment: str = "libero_franka") -> None:
        super().__init__()
        self.dof = EMBODIMENTS[embodiment].dof

    def forward(self, z: Tensor, c: Tensor) -> Tensor:
        return torch.randn(z.shape[0], H_OP, self.dof, device=z.device, dtype=z.dtype)

    def loss(self, z: Tensor, c: Tensor, a_seg: Tensor) -> Tensor:
        return (self.forward(z, c) - a_seg).pow(2).mean()


class StubProposal(nn.Module):
    """Plackett-Luce over random logits. Sampling and log_prob are consistent."""

    def _logits(self, z: Tensor) -> Tensor:
        g = torch.Generator(device="cpu").manual_seed(int(z.shape[0]))
        return torch.randn(z.shape[0], M, generator=g).to(z.device, z.dtype)

    def sample(self, z: Tensor, lang: Tensor, n: int) -> Tensor:
        return sparse_simplex(z.shape[0], n, device=z.device, dtype=z.dtype)

    def log_prob(self, z: Tensor, lang: Tensor, c: Tensor) -> Tensor:
        """Plackett-Luce log-probability of the support of c, in weight order.

        Weights are deterministic given the support, so the support carries the
        whole probability mass.
        """
        logits = self._logits(z)
        order = c.argsort(dim=-1, descending=True)[:, :TOPK]     # (B, TOPK)
        mask = torch.zeros_like(logits, dtype=torch.bool)
        total = torch.zeros(c.shape[0], device=c.device, dtype=c.dtype)
        for j in range(TOPK):
            idx = order[:, j]
            avail = logits.masked_fill(mask, float("-inf"))
            total = total + avail.gather(1, idx[:, None]).squeeze(1) - avail.logsumexp(-1)
            mask = mask.scatter(1, idx[:, None], True)
        return total


class StubPotential(nn.Module):
    """Random scalar per candidate. Satisfies contracts.Potential."""

    def forward(self, z: Tensor, lang: Tensor) -> Tensor:
        return torch.randn(*z.shape[:-2], device=z.device, dtype=z.dtype)


class StubPolicy:
    """Random actions with the correct segment/resampling loop.

    Team F builds the eval harness against this. The fractional accumulator
    below is the contract for inverting canonical resampling — the real Policy
    must do exactly this, or timing drifts over an episode.
    """

    def __init__(self, embodiment: str = "libero_franka") -> None:
        self.spec = EMBODIMENTS[embodiment]
        self.steps_per_seg = env_steps_per_segment(self.spec.env_fps)
        self.reset()

    def reset(self) -> None:
        self._z = None
        self._buffer: list[Tensor] = []
        self._accum = 0.0
        self.replans = 0

    def act(self, obs: dict, instruction: str) -> Tensor:
        if not self._buffer:
            self._accum += self.steps_per_seg
            n_env = int(self._accum)          # floor; remainder carries forward
            self._accum -= n_env
            self._buffer = [torch.randn(self.spec.dof) for _ in range(max(n_env, 1))]
            self.replans += 1
        return self._buffer.pop(0)


# ═══════════════════════════════════════════════════════════════════════════
#  PROTOCOL CONFORMANCE  (checked at import)
# ═══════════════════════════════════════════════════════════════════════════

_CONFORMANCE = [
    (StubEstimator, Estimator), (StubBank, Bank), (StubQDelta, QDelta),
    (StubQAction, QAction), (StubDecoder, Decoder), (StubProposal, Proposal),
    (StubPotential, Potential), (StubPolicy, Policy),
]
for _impl, _proto in _CONFORMANCE:
    assert isinstance(_impl.__new__(_impl), _proto), \
        f"{_impl.__name__} does not satisfy {_proto.__name__}"
