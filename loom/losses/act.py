"""LOOM — L_act: how *this body* produces the transformation.

A thin dispatcher over `Decoder.loss` (conditional flow matching; the CFM
parameterisation itself lives in `loom/heads/decoder.py`). Nothing is computed
here that a decoder could not compute — the value of this file is the
embodiment routing and the action-free path.

`D_e` takes `(proprio, c)`, not `(z, c)`: with the whole belief available the
decoder is a behaviour-cloning head and `L_act` puts no pressure at all on the
coefficient. See `loom/heads/decoder.py` for the measurement.

ACTION-FREE DATA
────────────────
R1 trains on Ego4D / Ego-Exo4D / HoloAssist, where `TransitionWindow.actions is
None`. `L_act` must then contribute exactly nothing *without* breaking the
backward pass of the combined loss. `zero_loss` returns a zero that is still
attached to the graph when any input requires grad, so
`(L_dyn + L_act + ...).backward()` works whether or not the batch has actions,
and the decoder simply receives no gradient that step. A bare Python `0.0`
would raise "element 0 of tensors does not require grad" on an action-free-only
batch; a `nan`-producing trick would poison every other loss.

q_a REGRESSION
──────────────
`q_action_regression_loss` is the align half of L_act. It is not an extra loss
(PLAN 9) — it is the second half of L_act, and it is the *only* mechanism tying
the two encoders into one coefficient space. It lives here rather than in its
own file because it is dispatched by embodiment exactly like the decoder term.

It takes a `direction`. PLAN 4.C wrote it as `q_a` regressing onto
`sg(q_Delta)`; `configs/r0a.yaml` now ships the reverse (`losses.act.align_to:
q_a`), because the original direction is the channel that copied q_Delta's
phase clock into q_a. See the function docstring for the measurements.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from contracts import H_OP

__all__ = [
    "zero_loss", "act_loss", "sparse_target_ce", "q_action_regression_loss",
]


def zero_loss(*ref: Tensor) -> Tensor:
    """A zero that does not poison the graph.

    Attached (with exactly-zero gradient) if any reference tensor requires grad,
    so it can always be added to the other loss terms and back-propagated.
    """
    for t in ref:
        if isinstance(t, Tensor) and t.requires_grad:
            return t.sum() * 0.0
    for t in ref:
        if isinstance(t, Tensor):
            return torch.zeros((), device=t.device, dtype=t.dtype)
    return torch.zeros(())


def sparse_target_ce(
    student_logits: Tensor,
    target_c: Tensor,
    *,
    temperature: float = 1.0,
    reduction: str = "mean",
) -> Tensor:
    """Cross-entropy from a sparse coefficient target to dense logits.

    ``target_c`` is a (usually top-k sparse) probability vector with exactly the
    same shape as ``student_logits``.  It is always stop-gradded: this loss is
    for moving the student into an already-defined operator space, never for
    pulling the target encoder toward the student.  Probability math is done in
    float32 even during bf16 training so small off-support probabilities and
    their gradients are not rounded away.

    ``reduction`` follows PyTorch's ``"none" | "sum" | "mean"`` convention;
    the unreduced result has ``student_logits.shape[:-1]``.
    """
    if student_logits.ndim < 1:
        raise ValueError("student_logits and target_c need a coefficient axis")
    if student_logits.shape != target_c.shape:
        raise ValueError(
            "student_logits and target_c must have the same shape, got "
            f"{tuple(student_logits.shape)} and {tuple(target_c.shape)}"
        )
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"temperature must be finite and > 0, got {temperature}")
    if reduction not in ("none", "sum", "mean"):
        raise ValueError(
            f"reduction must be 'none', 'sum', or 'mean', got {reduction!r}"
        )

    log_prob = F.log_softmax(student_logits.float() / temperature, dim=-1)
    per_item = -(target_c.detach().float() * log_prob).sum(dim=-1)
    if reduction == "none":
        return per_item
    if reduction == "sum":
        return per_item.sum()
    return per_item.mean()


def _flatten_segments(p: Tensor, c: Tensor, a: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Fold an optional DEPTH axis into the batch.

    Accepts either one segment per example — p (B,dof), c (B,M), a (B,H_OP,dof) —
    or a whole window — p (B,DEPTH,dof), c (B,DEPTH,M), a (B,DEPTH,H_OP,dof).
    The second form is what `TransitionWindow.actions` looks like, with `p` the
    proprio readings at the DEPTH operator-boundary states.
    """
    if a.ndim == 4:
        if p.ndim != 3 or c.ndim != 3:
            raise ValueError(
                "batched-window form needs proprio (B,DEPTH,dof) and c (B,DEPTH,M), "
                f"got proprio {tuple(p.shape)}, c {tuple(c.shape)}"
            )
        if not (p.shape[1] == c.shape[1] == a.shape[1]):
            raise ValueError(
                f"DEPTH mismatch: proprio {p.shape[1]}, c {c.shape[1]}, a {a.shape[1]}"
            )
        return p.flatten(0, 1), c.flatten(0, 1), a.flatten(0, 1)
    if a.ndim != 3 or a.shape[-2] != H_OP:
        raise ValueError(
            f"an action segment is (B, {H_OP}, dof) — one operator, never H_PLAN; "
            f"got {tuple(a.shape)}"
        )
    return p, c, a


def act_loss(
    decoder,
    proprio: Tensor,
    c: Tensor,
    actions: Tensor | None,
    embodiment: str | None = None,
    **decoder_kwargs,
) -> Tensor:
    """Conditional-flow-matching action loss for one embodiment-homogeneous batch.

    Args:
        decoder:    anything satisfying `contracts.Decoder` (per-body module or
                    the `ModuleDict` dispatcher from `loom.heads.decoder`).
        proprio:    (B,dof) or (B,DEPTH,dof) proprio at the operator boundaries.
                    **Not the belief** — `D_e` takes `(proprio, c)` so that `c`
                    is the only channel carrying task information into the
                    action; see `loom/heads/decoder.py`.
        c:          (B,M) or (B,DEPTH,M) operator coefficients, from q_a or q_Delta.
        actions:    (B,H_OP,dof) or (B,DEPTH,H_OP,dof); **None for action-free data**.
        embodiment: routing key. Optional for a single-body decoder.

    Returns a scalar. Zero (graph-safe) when `actions is None`.
    """
    if actions is None:
        return zero_loss(proprio, c)
    p_f, c_f, a_f = _flatten_segments(proprio, c, actions)
    if embodiment is None:
        return decoder.loss(p_f, c_f, a_f, **decoder_kwargs)
    return decoder.loss(p_f, c_f, a_f, embodiment=embodiment, **decoder_kwargs)


def q_action_regression_loss(
    q_action,
    a_seg: Tensor | None,
    z: Tensor,
    c_target: Tensor,
    embodiment: str | None = None,
    mode: str = "mse",
    direction: str = "q_a<-q_delta",
) -> Tensor:
    """One regression between the two coefficient encoders. `direction` says which.

    `direction="q_a<-q_delta"` (this function's original and still-default
        behaviour): `q_a^e(a_seg, z)` regresses onto `sg(q_Delta(z_t, z_{t+8}))`.
    `direction="q_delta<-q_a"` (ALIGN-FLIP, what `configs/r0a.yaml` ships as
        `losses.act.align_to: q_a`): `q_Delta` regresses onto `sg(q_a)` instead,
        and `c_target` is then q_Delta's output.

    Whichever way it points, exactly ONE side is stop-gradded. It is never
    two-directional: if the gradient flowed both ways the two encoders would
    meet in the middle at whatever is cheapest, which is a constant.

    WHICH DIRECTION SHIPS, AND WHY IT CHANGED
    ─────────────────────────────────────────
    The original text here said "q_Delta defines the coefficient space and q_a
    moves into it, never the other way round". That was the design, and it is
    what R0-A measured the consequences of. `q_Delta` became a pure phase clock
    -- `I(c_a; h)/H(c_a) = 99.8%`, `I(c_a; task) ~ 0` -- and this term was the
    channel that copied the clock into `q_a`: at the observed plateau
    `act/align` sits at its disjoint-support floor 0.500, so the align gradient
    on `c_a` has norm `2*sqrt(0.500) = 1.415` and is 100% common-mode, while
    `D_e`'s reconstruction gradient on the same tensor is 0.179 and 94%
    example-dependent. Align outbid decode ~7:1 and `q_a` went blind
    (`frac_var_a` 0.988 at fresh init -> 0.0015-0.0047 trained).

    Reversing it leaves `c_a` with `dec.loss` as its only gradient. Since the
    belief is not an input to `D_e`, `c` is the only channel from the world into
    the action segment, and a phase clock is nearly useless for that job:
    conditioning on `h` buys 3.3% of the action's conditional variance where
    conditioning on proprio buys 61% (`logs/cfm_floor.json`).

    The training loop computes this inline (it needs the per-horizon
    decomposition); this function is the reference implementation and the two
    must agree -- `tests/test_losses.py` pins both directions.

    mode="mse" (default, PLAN wording: "regression"): squared error between the
        two on-simplex vectors, summed over M and meaned over the batch. The
        gradient reaches q_a's logits through the straight-through head, so
        atoms outside q_a's current support still move.
    mode="ce":  cross-entropy of q_a's dense softmax against the target
        coefficients as a soft label. Better conditioned when the target support
        and q_a's support are disjoint; keep as an option, not the default.
    """
    if direction not in ("q_a<-q_delta", "q_delta<-q_a"):
        raise ValueError(
            f"unknown direction {direction!r}; use 'q_a<-q_delta' or 'q_delta<-q_a'")
    if a_seg is None:
        return zero_loss(z, c_target)
    body = q_action.body(embodiment) if hasattr(q_action, "body") else q_action

    if direction == "q_delta<-q_a":
        # q_Delta moves; q_a is the (stop-gradded) target. `c_target` is
        # q_Delta's live output here, not q_a's.
        with torch.no_grad():
            tgt = body(a_seg, z).detach()
        return (c_target - tgt.to(c_target.dtype)).pow(2).sum(-1).mean()

    tgt = c_target.detach()
    if mode == "mse":
        c_a = body(a_seg, z)
        return (c_a - tgt.to(c_a.dtype)).pow(2).sum(-1).mean()
    if mode == "ce":
        logits = body.logits(a_seg, z)
        return -(tgt.to(logits.dtype) * F.log_softmax(logits, dim=-1)).sum(-1).mean()
    raise ValueError(f"unknown mode {mode!r}; use 'mse' or 'ce'")
