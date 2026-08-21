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
import torch.nn.functional as F
from torch import Tensor

__all__ = ["proposal_bc_loss", "proposal_distill_loss", "proposal_sparse_ce_loss"]


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


def proposal_distill_loss(
    proposal,
    z: Tensor,
    lang: Tensor,
    teacher_logits: Tensor,
    *,
    temperature: float = 1.0,
    detach_belief: bool = True,
    reduction: str = "mean",
    return_student_logits: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Smoothly distil the action encoder's operator ranking into ``pi_c``.

    The old Plackett--Luce objective observes only the ordered hard top-k
    support.  A tiny q_a logit crossing therefore changes the entire discrete
    label even though the teacher distribution barely moved.  Matching the
    dense base categorical distributions is smooth across those crossings and
    still identifies the same Plackett--Luce ranking used at inference.

    ``temperature**2`` is the standard distillation rescaling: it keeps the
    gradient scale comparable when temperature changes.  Both teacher logits
    and (by default) the belief are stop-gradded.  The latter makes the proposal
    chase the representation learned by the method losses instead of moving the
    estimator to make its own classification problem easier.
    """
    if not hasattr(proposal, "logits"):
        raise TypeError("dense proposal distillation requires proposal.logits(z, lang)")
    if z.shape[0] != teacher_logits.shape[0]:
        raise ValueError(
            f"batch mismatch: z {z.shape[0]}, teacher_logits {teacher_logits.shape[0]}"
        )
    temperature = float(temperature)
    if not temperature > 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    student = proposal.logits(z.detach() if detach_belief else z, lang)
    if student.shape != teacher_logits.shape:
        raise ValueError(
            f"student/teacher shape mismatch: {tuple(student.shape)} vs "
            f"{tuple(teacher_logits.shape)}"
        )

    # Do the probability math in fp32 even under bf16 autocast.  With M=128,
    # low-probability operators otherwise quantise away and recreate a hard
    # target by accident.
    teacher = F.softmax(teacher_logits.detach().float() / temperature, dim=-1)
    log_student = F.log_softmax(student.float() / temperature, dim=-1)
    per_sample = F.kl_div(log_student, teacher, reduction="none").sum(-1)
    per_sample = per_sample * (temperature * temperature)

    if reduction == "none":
        reduced = per_sample
    elif reduction == "sum":
        reduced = per_sample.sum()
    elif reduction == "mean":
        reduced = per_sample.mean()
    else:
        raise ValueError(f"unknown reduction {reduction!r}")
    return (reduced, student) if return_student_logits else reduced


def proposal_sparse_ce_loss(
    proposal,
    z: Tensor,
    lang: Tensor,
    target_c: Tensor,
    *,
    temperature: float = 1.0,
    detach_belief: bool = True,
    reduction: str = "mean",
    return_student_logits: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Categorical CE from the deployed sparse coefficient to proposal logits.

    ``target_c`` is q_a's detached top-k simplex point, including its relative
    weights on the selected support.  Unlike dense KL this does not ask the
    proposal to imitate q_a's high-entropy tail, and unlike hard PL NLL it stays
    continuous while the weights within a fixed support move.
    """
    if not hasattr(proposal, "logits"):
        raise TypeError("sparse proposal CE requires proposal.logits(z, lang)")
    if z.shape[0] != target_c.shape[0]:
        raise ValueError(f"batch mismatch: z {z.shape[0]}, target_c {target_c.shape[0]}")
    temperature = float(temperature)
    if not temperature > 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    student = proposal.logits(z.detach() if detach_belief else z, lang)
    if student.shape != target_c.shape:
        raise ValueError(
            f"student/target shape mismatch: {tuple(student.shape)} vs "
            f"{tuple(target_c.shape)}"
        )
    target = target_c.detach().float()

    log_student = F.log_softmax(student.float() / temperature, dim=-1)
    per_sample = -(target * log_student).sum(-1)
    if reduction == "none":
        reduced = per_sample
    elif reduction == "sum":
        reduced = per_sample.sum()
    elif reduction == "mean":
        reduced = per_sample.mean()
    else:
        raise ValueError(f"unknown reduction {reduction!r}")
    return (reduced, student) if return_student_logits else reduced
