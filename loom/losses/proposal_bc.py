"""LOOM — L_proposal: behaviour cloning of pi_c.

    L_proposal = -log pi_c( sg(c_a) | z, l )

This is not loss creep. It is **the only thing that makes the model executable**
(PLAN 4.C/4.E): at inference there is no future state, so `q_Delta` is
unavailable, and no ground-truth action, so `q_a` is unavailable. `pi_c` is the
sole source of `c` at test time. A model trained without it cannot be evaluated
at all.

The target is `sg(c_a)` — the coefficients from `q_a^e` (or from `q_Delta` on
action-free data), stop-gradded. The stop-grad matters: without it the cheapest
way to raise `log pi_c(c)` is for the *encoder* to move `c` towards whatever the
proposal already likes, which drags the coefficient space around by its BC head.
The proposal chases the encoders; never the reverse.

Depends only on `contracts.Proposal` (Team E owns the real Plackett-Luce head;
these tests run against `stubs.StubProposal`). `Proposal.log_prob` returns a
(B,) log-probability of the *ordered support*, since the weights are
deterministic given the support.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["proposal_bc_loss"]


def proposal_bc_loss(
    proposal,
    z: Tensor,
    lang: Tensor,
    c: Tensor,
    reduction: str = "mean",
) -> Tensor:
    """-log pi_c(sg(c) | z, lang).

    Args:
        proposal:  anything satisfying `contracts.Proposal`.
        z:         (B, K, D) belief from the online estimator.
        lang:      (B, L, F) language features.
        c:         (B, M) target coefficients. Detached here unconditionally.
        reduction: "mean" -> scalar, "none" -> (B,), "sum" -> scalar.

    Returns non-negative values: `log_prob` of a discrete support is <= 0.
    """
    if c.shape[0] != z.shape[0]:
        raise ValueError(f"batch mismatch: z {z.shape[0]}, c {c.shape[0]}")
    lp = proposal.log_prob(z, lang, c.detach())
    if lp.ndim != 1 or lp.shape[0] != z.shape[0]:
        raise ValueError(
            f"Proposal.log_prob must return (B,), got {tuple(lp.shape)}"
        )
    nll = -lp
    if reduction == "none":
        return nll
    if reduction == "sum":
        return nll.sum()
    if reduction == "mean":
        return nll.mean()
    raise ValueError(f"unknown reduction {reduction!r}")
