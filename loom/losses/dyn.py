"""LOOM — L_dyn: what a transformation *does*.

    L_dyn = sum_{h=1..DEPTH} w_h * (1 - cos(LN(z_hat_{t+8h}), sg(LN(z_bar_{t+8h}))))
    w = contracts.DYN_WEIGHTS = (1.0, 0.5, 0.25, 0.125)

`z_hat` comes from **sequential `bank.step`** — never a product of lambdas.
Affine composition is `(A2 A1, A2 b1 + b2)`; multiplying the lambdas alone
silently discards the accumulated bias, and the loss would still go down.

Targets `z_bar` come from the **EMA estimator** (`contracts.EMA_TAU = 0.996`)
under stop-grad. Without the stop-grad the cheapest solution is for the
estimator to collapse `z` to a constant; without the EMA the target moves as
fast as the predictor and the cosine is satisfiable by drifting together.


COSINE AXIS (deliberate)
────────────────────────
`cosine="per_slot"` (default): layer-norm over the D axis, cosine over D within
each slot, then mean over the K slots. The alternative, `cosine="flat"`,
layer-norms and cosines over the flattened K*D vector.

Per-slot is the more informative choice and is the default. A flattened cosine
is dominated by whichever slots carry the largest norm; a rollout that gets the
three high-energy slots right and the other 125 wrong still scores ~0.95. The
belief is a *set* of slots and the operator has to act correctly on all of them,
so every slot gets an equal vote. The cost is that the per-slot version is
harsher — it will not saturate near 1.0 the way the flat one does, so do not
compare the two numbers across runs.

(After LayerNorm each slot is zero-mean unit-variance, so the per-slot cosine
is exactly the Pearson correlation of the two slots. `1 - cos` is in [0, 2].)


NEGATIVES
─────────
`negatives in {"none", "within_trajectory"}`, default `"within_trajectory"`.
A negative is a `c` from **another segment of the SAME trajectory, at least
`min_gap=2` segments away** — same scene, same body, same lighting, genuinely
different effect. The loss then also asks that the true operator land closer to
the target than that negative one does, by a margin.

**Do NOT replace this with uncurated in-batch negatives.** A batch is
embodiment-homogeneous but not scene-homogeneous, and across bodies/scenes two
different `c` frequently denote the *same world effect* ("the gripper closed").
Making those repel is exactly the opposite of what a shared operator bank is
for: the whole point is that one operator index means one world effect no matter
which body produced it. In-batch negatives would teach the bank to be
body-specific, which is the failure this architecture exists to avoid.


Delta_op IS A BUILD ASSERT, NOT A METRIC
────────────────────────────────────────
    Delta_op = d(A(c_rand) z, z+) - d(A(c_true) z, z+)      must be > 0

Mind the sign: `d` is a *distance*, so the true operator must be the CLOSER one
and the difference must be positive.

It is returned in the output dict of every call so Team D can log it every
single step. Why it is load-bearing: latent states 8 canonical steps apart are
~0.95 cosine-similar before training, so `A(c) ~ I` already nearly minimises
L_dyn while `c` carries no information whatsoever. The loss curve looks fine.
`Delta_op` is the only thing that distinguishes "the operator is doing the work"
from "the operator is the identity and the belief is doing the work". If it
flatlines near zero in the first few thousand steps, the model has collapsed to
a plain latent policy — flip `negatives` to `"within_trajectory"` (or raise
`neg_weight`) before burning the full run.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from contracts import DYN_WEIGHTS, EMA_TAU, TOPK, Bank

__all__ = [
    "NEGATIVE_MODES", "ln_cosine", "ln_cosine_distance", "sequential_rollout",
    "sample_within_trajectory_negatives", "random_simplex_like",
    "ema_update", "EmaEstimator", "dyn_loss", "DynLoss",
]

NEGATIVE_MODES = ("none", "within_trajectory")


# ═══════════════════════════════════════════════════════════════════════════
#  DISTANCE
# ═══════════════════════════════════════════════════════════════════════════

def ln_cosine(a: Tensor, b: Tensor, mode: str = "per_slot") -> Tensor:
    """cos(LN(a), LN(b)) for beliefs (..., K, D) -> (...).

    mode="per_slot": LN over D, cosine over D, mean over K   (default, see module doc)
    mode="flat":     LN over (K, D), cosine over the flattened vector
    """
    if mode == "per_slot":
        an = F.layer_norm(a, a.shape[-1:])
        bn = F.layer_norm(b, b.shape[-1:])
        return F.cosine_similarity(an, bn, dim=-1).mean(-1)
    if mode == "flat":
        an = F.layer_norm(a, a.shape[-2:]).flatten(-2)
        bn = F.layer_norm(b, b.shape[-2:]).flatten(-2)
        return F.cosine_similarity(an, bn, dim=-1)
    raise ValueError(f"unknown cosine mode {mode!r}; use 'per_slot' or 'flat'")


def ln_cosine_distance(a: Tensor, b: Tensor, mode: str = "per_slot") -> Tensor:
    """1 - cos(LN(a), LN(b)). A distance: smaller means closer. In [0, 2]."""
    return 1.0 - ln_cosine(a, b, mode)


# ═══════════════════════════════════════════════════════════════════════════
#  ROLLOUT / NEGATIVES
# ═══════════════════════════════════════════════════════════════════════════

def sequential_rollout(bank: Bank, z0: Tensor, c_seq: Tensor) -> list[Tensor]:
    """DEPTH applications of `bank.step`, returning EVERY intermediate state.

    `bank.rollout` only returns the leaf; L_dyn supervises all DEPTH horizons,
    so it needs the whole chain. This is sequential affine composition, which is
    the only correct way to compose these operators.
    """
    out, z = [], z0
    for h in range(c_seq.shape[-2]):
        z = bank.step(c_seq[..., h, :], z)
        out.append(z)
    return out


def random_simplex_like(c: Tensor, topk: int = TOPK,
                        generator: torch.Generator | None = None) -> Tensor:
    """Uniform-support random point on the top-k simplex, shaped like `c`."""
    flat = c.detach().reshape(-1, c.shape[-1]).float()
    n, m = flat.shape
    idx = torch.rand(n, m, device=c.device, generator=generator).argsort(dim=1)[:, :topk]
    w = torch.rand(n, topk, device=c.device, generator=generator)
    w = w / w.sum(-1, keepdim=True)
    out = torch.zeros_like(flat).scatter_(1, idx, w)
    return out.reshape(c.shape).to(c.dtype)


def sample_within_trajectory_negatives(
    c_seq: Tensor,
    min_gap: int = 2,
    generator: torch.Generator | None = None,
) -> Tensor:
    """(B, DEPTH, M) -> (B, DEPTH, M): for each segment, the coefficients of
    another segment of the SAME trajectory at least `min_gap` segments away.

    Same scene, same body, genuinely different effect. See the module docstring
    for why this is not in-batch sampling.

    With DEPTH=4 and min_gap=2 the candidate sets are
    0 -> {2,3}, 1 -> {3}, 2 -> {0}, 3 -> {0,1}. Every segment must have at least
    one candidate, which DEPTH=4, min_gap=2 satisfies exactly.
    """
    if c_seq.ndim != 3:
        raise ValueError(f"expected (B, DEPTH, M), got {tuple(c_seq.shape)}")
    b, depth, m = c_seq.shape
    offs = torch.arange(depth, device=c_seq.device)
    valid = (offs[None, :] - offs[:, None]).abs() >= min_gap       # (DEPTH, DEPTH)
    if not bool(valid.any(-1).all()):
        # e.g. DEPTH=3 with min_gap=2: the middle segment has no partner at all.
        raise ValueError(
            f"a window of {depth} segments cannot supply a negative {min_gap} segments "
            f"away for every segment; use negatives='none' or a longer window"
        )
    pick = torch.multinomial(
        valid.float().expand(b, depth, depth).reshape(b * depth, depth),
        num_samples=1, replacement=True, generator=generator,
    ).view(b, depth)
    return c_seq.detach().gather(1, pick[..., None].expand(b, depth, m))


# ═══════════════════════════════════════════════════════════════════════════
#  EMA TARGET MACHINERY   (L_dyn's target lives here, so it lives in this file)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, tau: float = EMA_TAU) -> None:
    """target <- tau * target + (1 - tau) * online, parameters AND buffers.

    Buffers too: a BatchNorm/running-stat buffer that is copied instead of
    averaged makes the "slow" target follow the fast one exactly, which defeats
    the point of having one.
    """
    if not 0.0 <= tau <= 1.0:
        raise ValueError(f"tau must be in [0, 1], got {tau}")
    for p_t, p_o in zip(target.parameters(), online.parameters()):
        p_t.mul_(tau).add_(p_o.detach().to(p_t.dtype), alpha=1.0 - tau)
    for b_t, b_o in zip(target.buffers(), online.buffers()):
        if b_t.dtype.is_floating_point:
            b_t.mul_(tau).add_(b_o.detach().to(b_t.dtype), alpha=1.0 - tau)
        else:
            b_t.copy_(b_o)


class EmaEstimator(nn.Module):
    """Stop-gradded EMA copy of the estimator; produces L_dyn's targets.

    Everything it returns is detached, and its parameters never require grad, so
    it is structurally impossible to backprop into the target branch.
    """

    def __init__(self, online: nn.Module, tau: float = EMA_TAU) -> None:
        super().__init__()
        self.tau = tau
        self.target = copy.deepcopy(online)
        self.target.requires_grad_(False)
        self.target.eval()

    @torch.no_grad()
    def update(self, online: nn.Module) -> None:
        ema_update(self.target, online, self.tau)

    @torch.no_grad()
    def forward(self, *args, **kwargs) -> Tensor:
        return self.target(*args, **kwargs).detach()

    def train(self, mode: bool = True):     # keep the target in eval forever
        super().train(mode)
        self.target.eval()
        return self


# ═══════════════════════════════════════════════════════════════════════════
#  L_dyn
# ═══════════════════════════════════════════════════════════════════════════

def dyn_loss(
    bank: Bank,
    z0: Tensor,
    c_seq: Tensor,
    z_targets: Tensor,
    *,
    negatives: str = "within_trajectory",
    c_neg: Tensor | None = None,
    min_gap: int = 2,
    neg_weight: float = 1.0,
    neg_margin: float = 0.1,
    weights: tuple[float, ...] = DYN_WEIGHTS,
    cosine: str = "per_slot",
    generator: torch.Generator | None = None,
) -> dict[str, Tensor]:
    """L_dyn plus the `Delta_op` build assert.

    Args:
        bank:      anything satisfying `contracts.Bank` (stubs.StubBank in tests).
        z0:        (B, K, D) belief at the window start, from the ONLINE estimator.
        c_seq:     (B, H, M) operator coefficients, H <= len(weights).
        z_targets: (B, H, K, D) beliefs at canonical frames 8, 16, ... from the
                   EMA estimator. Detached here regardless of what came in.
        negatives: "none" | "within_trajectory". See the module docstring.
        c_neg:     optional explicit negatives (B, H, M); otherwise drawn from
                   `c_seq` with `sample_within_trajectory_negatives`.
        neg_weight/neg_margin: hinge on the gap d_neg - d_pos.
        weights:   per-horizon weights, `contracts.DYN_WEIGHTS`.
        cosine:    "per_slot" (default) or "flat".

    Returns a dict — Team D logs `delta_op` every step:
        loss       scalar, the training objective
        dyn        scalar, the positive term alone (comparable across runs)
        neg        scalar, the hinge term (0.0 when negatives="none")
        delta_op   scalar, MUST be > 0  (build assert, detached)
        cos_pos    scalar, mean cosine of the h=1 prediction to its target
        per_h      (H,) unweighted per-horizon distances, detached
    """
    if negatives not in NEGATIVE_MODES:
        raise ValueError(f"negatives must be one of {NEGATIVE_MODES}, got {negatives!r}")
    if c_seq.ndim != 3:
        raise ValueError(f"c_seq must be (B, H, M), got {tuple(c_seq.shape)}")
    if z_targets.ndim != 4 or z_targets.shape[1] != c_seq.shape[1]:
        raise ValueError(
            f"z_targets must be (B, H, K, D) with H={c_seq.shape[1]}, "
            f"got {tuple(z_targets.shape)}"
        )
    horizons = c_seq.shape[1]
    if horizons > len(weights):
        raise ValueError(f"{horizons} horizons but only {len(weights)} weights")

    tgt = z_targets.detach()                       # sg(.) — never optional
    z_hat = sequential_rollout(bank, z0, c_seq)    # sequential, bias included

    d_pos = [ln_cosine_distance(z_hat[h], tgt[:, h], cosine) for h in range(horizons)]

    loss_pos = sum(float(weights[h]) * d_pos[h].mean() for h in range(horizons))
    loss_neg = torch.zeros((), device=z0.device, dtype=loss_pos.dtype)

    if negatives == "within_trajectory" and neg_weight != 0.0:
        if c_neg is None:
            c_neg = sample_within_trajectory_negatives(c_seq, min_gap, generator)
        z_neg = sequential_rollout(bank, z0, c_neg.to(c_seq.dtype))
        for h in range(horizons):
            d_neg = ln_cosine_distance(z_neg[h], tgt[:, h], cosine)
            hinge = F.relu(neg_margin - (d_neg - d_pos[h])).mean()
            loss_neg = loss_neg + float(weights[h]) * hinge
        loss_neg = neg_weight * loss_neg

    # ── build assert: Delta_op = d(A(c_rand) z, z+) - d(A(c_true) z, z+) > 0 ──
    with torch.no_grad():
        c_rand = random_simplex_like(c_seq[:, 0], generator=generator)
        d_rand = ln_cosine_distance(bank.step(c_rand, z0), tgt[:, 0], cosine)
        delta_op = (d_rand - d_pos[0].detach()).mean()

    return {
        "loss": loss_pos + loss_neg,
        "dyn": loss_pos.detach(),
        "neg": loss_neg.detach(),
        "delta_op": delta_op,
        "cos_pos": (1.0 - d_pos[0].detach().mean()),
        "per_h": torch.stack([d.detach().mean() for d in d_pos]),
    }


class DynLoss(nn.Module):
    """Configured `dyn_loss`. Holds `negatives` so Team D can flip it from yaml.

    >>> crit = DynLoss(negatives="within_trajectory")
    >>> out  = crit(bank, z0, c_seq, z_targets)
    >>> out["loss"].backward();  log(out["delta_op"])    # must stay > 0
    """

    def __init__(
        self,
        negatives: str = "within_trajectory",
        min_gap: int = 2,
        neg_weight: float = 1.0,
        neg_margin: float = 0.1,
        weights: tuple[float, ...] = DYN_WEIGHTS,
        cosine: str = "per_slot",
    ) -> None:
        super().__init__()
        if negatives not in NEGATIVE_MODES:
            raise ValueError(f"negatives must be one of {NEGATIVE_MODES}, got {negatives!r}")
        self.negatives = negatives
        self.min_gap = min_gap
        self.neg_weight = neg_weight
        self.neg_margin = neg_margin
        self.weights = tuple(weights)
        self.cosine = cosine

    def forward(self, bank: Bank, z0: Tensor, c_seq: Tensor, z_targets: Tensor,
                c_neg: Tensor | None = None,
                generator: torch.Generator | None = None) -> dict[str, Tensor]:
        return dyn_loss(
            bank, z0, c_seq, z_targets,
            negatives=self.negatives, c_neg=c_neg, min_gap=self.min_gap,
            neg_weight=self.neg_weight, neg_margin=self.neg_margin,
            weights=self.weights, cosine=self.cosine, generator=generator,
        )
