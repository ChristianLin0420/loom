"""
LOOM — contracts.

FROZEN AFTER PHASE 0. Do not edit.

Every constant, shape validator and Protocol that more than one team touches
lives here. A genuine contract change halts Phase 1, is made once, and all six
teams rebase (PLAN.md 6.4).

TWO OWNER-AUTHORISED CHANGES have been made since the freeze, both for the
R0-A rerun, and they are the only ones:

  1. `Decoder` now takes `(proprio, c)`, not `(z, c)`. The belief is gone from
     the realizer; `c` is the only channel carrying task information into the
     action.
  2. `BALANCE_COEF` 3e-3 -> 1e-2, and the executed form changed from
     KL-of-batch-mean to the Switch auxiliary `M * sum_m f_m P_m`.

Each is documented at its definition. Nothing else in this file moved.

Nothing in this file imports anything from `loom` or `stubs`. It is the root of
the dependency graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from torch import Tensor
from typing_extensions import TypedDict

__all__ = [
    # temporal
    "FPS_CANONICAL", "H_OP", "DEPTH", "H_PLAN", "N_STATES", "CANONICAL_FRAMES",
    # model
    "K", "D", "M", "TOPK", "RHO", "B_MAX", "EMA_TAU",
    # losses
    "DYN_WEIGHTS", "BALANCE_COEF", "REALIZABILITY_TAU",
    # embodiments
    "EmbodimentSpec", "EMBODIMENTS", "register_embodiment", "env_steps_per_segment",
    # typed dicts
    "ObsFeats", "TransitionWindow",
    # protocols
    "Estimator", "Bank", "QDelta", "QAction", "Decoder", "Proposal",
    "Potential", "Policy",
    # validators
    "assert_belief", "assert_simplex", "assert_action_segment",
    "assert_contractive", "assert_bias_bounded",
]


# ═══════════════════════════════════════════════════════════════════════════
#  TEMPORAL
# ═══════════════════════════════════════════════════════════════════════════

FPS_CANONICAL = 30          # every dataset resampled to this before segmenting
H_OP          = 8           # control steps per operator -> 267 ms
DEPTH         = 4           # planning horizon, in operators
H_PLAN        = H_OP * DEPTH        # 32 canonical steps, 1.07 s
N_STATES      = DEPTH + 1           # 5 operator-boundary states per window

#: canonical frame indices of the N_STATES boundary observations
CANONICAL_FRAMES = tuple(H_OP * i for i in range(N_STATES))     # (0, 8, 16, 24, 32)


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL
# ═══════════════════════════════════════════════════════════════════════════

K       = 128           # belief slots
D       = 768           # slot width, MUST be even
M       = 128           # operator bank size
TOPK    = 4             # nonzero coefficients
RHO     = 0.98          # spectral radius bound per operator
B_MAX   = 1.0           # norm bound per bias
EMA_TAU = 0.996         # target-estimator EMA


# ═══════════════════════════════════════════════════════════════════════════
#  LOSSES
# ═══════════════════════════════════════════════════════════════════════════

#: per-horizon weights in L_dyn; one entry per rollout step
DYN_WEIGHTS = (1.0, 0.5, 0.25, 0.125)

#: coefficient on the Switch load-balancing term  M * sum_m f_m P_m
#:
#: CHANGED 3e-3 -> 1e-2 by the project owner (the second of two authorised
#: edits to this frozen file; the other is `Decoder` below).
#:
#: The form changed with it. It used to be KL(mean_batch(c) || uniform(M)),
#: which only sees the batch mean; the executed term is now the Switch
#: auxiliary loss, `M * sum_m f_m P_m`, with `f_m` the fraction of tokens whose
#: hard top-4 support contains m and `P_m` the mean router probability for m.
#: This is the ONLY recruitment force whose gradient reaches an *unselected*
#: operator without passing through the top-4 mask: measured on the R0-A
#: checkpoints, a non-selected operator receives 0.0006 (ctrl) / 0.0001 (zinit)
#: of a selected one's per-entry gradient at q_Delta's logits. `topk_simplex_st`
#: returns `hard + soft - soft.detach()`, so the straight-through backward is
#: already dense -- the deficit is magnitude, not a closed path.
BALANCE_COEF = 1e-2

#: search-time realizability gate: reject root c when ||q_a(D_e(z,c), z) - c|| > tau
REALIZABILITY_TAU = 0.5


# ═══════════════════════════════════════════════════════════════════════════
#  TENSOR ALIASES  (documentation only)
# ═══════════════════════════════════════════════════════════════════════════
#
#   Belief          (..., K, D)             real, never complex
#   Coeff           (..., M)                simplex, <= TOPK nonzero
#   LamPair         (..., K, D//2) x2       real (a, b) meaning a + ib, |a+ib| <= RHO
#   ActionSegment   (..., H_OP, dof_e)      ONE operator's worth. NEVER H_PLAN.
#   CoeffSeq        (B, N, DEPTH, M)        one candidate plan per n


# ═══════════════════════════════════════════════════════════════════════════
#  EMBODIMENT REGISTRY
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class EmbodimentSpec:
    """Everything head dispatch and action de/normalisation needs.

    `env_fps` is the rate the *environment* steps at, not the rate the dataset
    was recorded at (those can differ; the recorded rate travels in
    `TransitionWindow.src_fps`).
    """

    name:        str
    dof:         int
    env_fps:     float
    n_views:     int
    action_low:  tuple[float, ...]
    action_high: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.dof <= 0:
            raise ValueError(f"{self.name}: dof must be positive, got {self.dof}")
        if self.n_views <= 0:
            raise ValueError(f"{self.name}: n_views must be positive, got {self.n_views}")
        if self.env_fps <= 0.0:
            raise ValueError(f"{self.name}: env_fps must be positive, got {self.env_fps}")
        if len(self.action_low) != self.dof:
            raise ValueError(
                f"{self.name}: action_low has {len(self.action_low)} entries, dof is {self.dof}"
            )
        if len(self.action_high) != self.dof:
            raise ValueError(
                f"{self.name}: action_high has {len(self.action_high)} entries, dof is {self.dof}"
            )
        if any(hi <= lo for lo, hi in zip(self.action_low, self.action_high)):
            raise ValueError(f"{self.name}: action_high must exceed action_low elementwise")


#: name -> spec. Adapters register their own embodiment at import time.
EMBODIMENTS: dict[str, EmbodimentSpec] = {}


def register_embodiment(spec: EmbodimentSpec) -> EmbodimentSpec:
    """Idempotent for an identical spec; raises on a conflicting redefinition.

    Adapters are imported more than once (workers, eval, tests). Re-registering
    the same body must be free; silently overwriting a different one must not.
    """
    prev = EMBODIMENTS.get(spec.name)
    if prev is not None and prev != spec:
        raise ValueError(
            f"conflicting registration for {spec.name!r}:\n  have {prev}\n  got  {spec}"
        )
    EMBODIMENTS[spec.name] = spec
    return spec


def env_steps_per_segment(env_fps: float) -> float:
    """How many environment steps one operator spans. Deliberately fractional.

    8 canonical steps at 30 Hz onto a 20 Hz env is 5.333, not 5. A Policy must
    carry the remainder in an accumulator; rounding each segment independently
    drifts by a full second over a 600-step episode.
    """
    return H_OP * float(env_fps) / FPS_CANONICAL


# LIBERO: OSC_POSE delta control, 7-dim (3 pos, 3 axis-angle, 1 gripper), all in
# [-1, 1]; agentview + eye_in_hand; robosuite control_freq 20 Hz.
register_embodiment(EmbodimentSpec(
    name="libero_franka",
    dof=7,
    env_fps=20.0,
    n_views=2,
    action_low=(-1.0,) * 7,
    action_high=(1.0,) * 7,
))


# ═══════════════════════════════════════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════════════════════════════════════

class ObsFeats(TypedDict):
    """Frozen-tower features for ONE canonical frame. Never raw pixels."""

    views:   Tensor   # (B, V, P, F)  V streams; tactile gel-pads are just views
    proprio: Tensor   # (B, dof_e)
    lang:    Tensor   # (B, L, F)


class _TransitionWindowRequired(TypedDict):
    feats:      list[ObsFeats]   # length N_STATES, at CANONICAL_FRAMES
    actions:    Tensor | None    # (B, DEPTH, H_OP, dof_e); None for action-free data
    lang:       Tensor           # (B, L, F)
    embodiment: str              # HOMOGENEOUS within a batch
    src_fps:    float            # original rate, needed to invert resampling at eval


class TransitionWindow(_TransitionWindowRequired, total=False):
    """One training example. Embodiment-homogeneous by construction."""

    burn_in_feats: list[ObsFeats]  # real preceding operator-boundary observations


# ═══════════════════════════════════════════════════════════════════════════
#  PROTOCOLS
# ═══════════════════════════════════════════════════════════════════════════

@runtime_checkable
class Estimator(Protocol):                                # shared
    def forward(self, feats: ObsFeats, z_prev: Tensor | None) -> Tensor: ...


@runtime_checkable
class Bank(Protocol):                                     # shared
    def mix(self, c: Tensor) -> tuple[Tensor, Tensor]: ...      # Coeff -> (a, b)
    def bias(self, c: Tensor) -> Tensor: ...                    # Coeff -> (..., K, D)
    def step(self, c: Tensor, z: Tensor) -> Tensor: ...         # ONE affine step
    def rollout(self, c_seq: Tensor, z: Tensor) -> Tensor: ...
    # (B,N,DEPTH,M), (B,K,D) -> (B,N,K,D). Sequential over DEPTH.
    # There is NO compose(). Composing affine maps gives (A2A1, A2b1+b2);
    # multiplying lambdas alone silently discards the bias.


@runtime_checkable
class QDelta(Protocol):                                   # shared
    def forward(self, z_t: Tensor, z_next: Tensor) -> Tensor: ...


@runtime_checkable
class QAction(Protocol):                                  # ONE PER EMBODIMENT
    def forward(self, a_seg: Tensor, z: Tensor) -> Tensor: ...  # a_seg (B,H_OP,dof_e)


@runtime_checkable
class Decoder(Protocol):                                  # ONE PER EMBODIMENT
    """`D_e(proprio_t, c) -> (B, H_OP, dof_e)`.  The belief is NOT an input.

    CHANGED by the project owner (the first of two authorised edits to this
    frozen file; the other is `BALANCE_COEF` above). The decoder used to take
    the whole `(B, K, D)` belief alongside `c`.

    Why it does not any more: predicting an 8-step action segment from a
    128x768 belief is behaviour cloning, and behaviour cloning needs nothing
    from `c`. Measured on R0-A, `act/decode` fell 0.2489 -> 0.0559 while `c_a`
    held 2-3 distinct top-4 supports over 64 real training windows -- `L_act`
    was exerting no pressure on the coefficient at all. With `z` removed, `c`
    is the ONLY channel carrying task information into the action, which is
    what makes `L_act` a training signal for the operator.

    `proprio` is `ObsFeats["proprio"]`, `(B, dof_e)` -- ONE timestep, the body's
    own state. It is what tells the realizer where the arm currently is; it
    carries no task information, which is the point.
    """

    def forward(self, proprio: Tensor, c: Tensor) -> Tensor: ...    # -> (B,H_OP,dof_e)
    def loss(self, proprio: Tensor, c: Tensor, a_seg: Tensor) -> Tensor: ...


@runtime_checkable
class Proposal(Protocol):                                 # shared
    def sample(self, z: Tensor, lang: Tensor, n: int) -> Tensor: ...   # -> (B,n,M)
    def log_prob(self, z: Tensor, lang: Tensor, c: Tensor) -> Tensor: ...


@runtime_checkable
class Potential(Protocol):                                # shared, R3 only
    def forward(self, z: Tensor, lang: Tensor) -> Tensor: ...          # -> (B,)


@runtime_checkable
class Policy(Protocol):
    """The ONLY interface eval depends on."""

    def reset(self) -> None: ...
    def act(self, obs: dict, instruction: str) -> np.ndarray: ...


# ═══════════════════════════════════════════════════════════════════════════
#  VALIDATORS
#
#  These raise AssertionError, not ValueError. They are build asserts: cheap
#  enough to leave on in tests, stripped by -O in a hot loop.
# ═══════════════════════════════════════════════════════════════════════════

def _tol(t: Tensor) -> float:
    """Slack proportional to the dtype. bf16 has 8 mantissa bits; f32 has 24."""
    return max(8.0 * float(torch.finfo(t.dtype).eps), 1e-6)


def _simplex_draw(n: int, device="cpu", dtype=torch.float32) -> Tensor:
    """(n, M) with exactly TOPK nonzeros summing to 1. Local copy so that
    contracts.py stays at the root of the import graph."""
    idx = torch.argsort(torch.rand(n, M, device=device), dim=1)[:, :TOPK]
    w = torch.rand(n, TOPK, device=device)
    w = w / w.sum(-1, keepdim=True)
    return torch.zeros(n, M, device=device).scatter_(1, idx, w).to(dtype)


def assert_belief(z: Tensor) -> None:
    """z is real, finite, and (..., K, D)."""
    assert isinstance(z, Tensor), f"belief must be a Tensor, got {type(z).__name__}"
    assert not z.is_complex(), (
        "z is real throughout; the 2x2 block algebra is four real elementwise ops "
        "and there is no complex-bf16 dtype"
    )
    assert z.ndim >= 2, f"belief needs at least (K, D), got {tuple(z.shape)}"
    assert z.shape[-2:] == (K, D), f"belief must end in ({K}, {D}), got {tuple(z.shape)}"
    assert torch.isfinite(z.float()).all(), "belief contains nan/inf"


def assert_simplex(c: Tensor, topk: int = TOPK) -> None:
    """c is on the simplex with at most `topk` nonzeros. Every bound depends on this."""
    assert isinstance(c, Tensor), f"coeff must be a Tensor, got {type(c).__name__}"
    assert not c.is_complex(), "coefficients are real"
    assert c.shape[-1] == M, f"coeff must end in M={M}, got {tuple(c.shape)}"
    tol = _tol(c)
    cf = c.float()
    assert (cf >= -tol).all(), f"negative coefficient: min {cf.min().item():.3g}"
    total = cf.sum(-1)
    assert torch.allclose(total, torch.ones_like(total), atol=max(tol, 1e-4)), (
        f"coeff must sum to 1 (a plain softmax over M does not); "
        f"got range [{total.min().item():.4g}, {total.max().item():.4g}]"
    )
    nnz = (cf > tol).sum(-1)
    assert (nnz <= topk).all(), (
        f"coeff must have at most {topk} nonzeros (hard top-k, then renormalise); "
        f"got up to {int(nnz.max())}"
    )


def assert_action_segment(a: Tensor, embodiment: str) -> None:
    """One c = one operator = H_OP control steps. Never H_PLAN."""
    assert embodiment in EMBODIMENTS, f"unregistered embodiment {embodiment!r}"
    spec = EMBODIMENTS[embodiment]
    assert a.ndim >= 2, f"action segment needs (H_OP, dof), got {tuple(a.shape)}"
    assert a.shape[-2] == H_OP, (
        f"an action segment is {H_OP} steps, got {a.shape[-2]}"
        + (f" — that is H_PLAN, not H_OP" if a.shape[-2] == H_PLAN else "")
    )
    assert a.shape[-1] == spec.dof, (
        f"{embodiment} has dof {spec.dof}, got {a.shape[-1]}"
    )


def assert_contractive(bank: Bank, n: int = 2000, device="cpu", tol: float = 1e-4) -> None:
    """||A(c)||_2 <= RHO over n random simplex draws.

    A is block-diagonal in 2x2 rotation-decay blocks, so its spectral norm is
    the largest block magnitude sqrt(a^2 + b^2).
    """
    c = _simplex_draw(n, device=device)
    with torch.no_grad():
        a, b = bank.mix(c)
        r = torch.sqrt(a.float() ** 2 + b.float() ** 2)
    worst = float(r.max())
    assert worst <= RHO + tol, f"spectral radius {worst:.6f} exceeds RHO={RHO}"


def assert_bias_bounded(bank: Bank, n: int = 2000, device="cpu", tol: float = 1e-4) -> None:
    """||b(c)|| <= B_MAX over n random simplex draws."""
    c = _simplex_draw(n, device=device)
    with torch.no_grad():
        nb = bank.bias(c).float().flatten(1).norm(dim=1)
    worst = float(nb.max())
    assert worst <= B_MAX + tol, f"bias norm {worst:.6f} exceeds B_MAX={B_MAX}"
