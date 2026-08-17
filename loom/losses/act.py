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
`q_action_regression_loss` is the PLAN 4.C training signal for `q_a^e`:
regression onto `sg(q_Delta(z_t, z_{t+8}))`. It is not an extra loss (PLAN 9) —
it is the action-side half of L_act, and it is the *only* mechanism tying the
two encoders into one coefficient space. It lives here rather than in its own
file because it is dispatched by embodiment exactly like the decoder term.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from contracts import H_OP

__all__ = ["zero_loss", "act_loss", "q_action_regression_loss"]


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
) -> Tensor:
    """Regress `q_a^e(a_seg, z)` onto `sg(q_Delta(z_t, z_{t+8}))`.

    `c_target` is stop-gradded here unconditionally: q_Delta defines the
    coefficient space and q_a moves into it, never the other way round. If the
    gradient flowed both ways the two encoders would meet in the middle at
    whatever is cheapest, which is a constant.

    mode="mse" (default, PLAN wording: "regression"): squared error between the
        two on-simplex vectors, summed over M and meaned over the batch. The
        gradient reaches q_a's logits through the straight-through head, so
        atoms outside q_a's current support still move.
    mode="ce":  cross-entropy of q_a's dense softmax against the target
        coefficients as a soft label. Better conditioned when the target support
        and q_a's support are disjoint; keep as an option, not the default.
    """
    if a_seg is None:
        return zero_loss(z, c_target)
    tgt = c_target.detach()
    body = q_action.body(embodiment) if hasattr(q_action, "body") else q_action

    if mode == "mse":
        c_a = body(a_seg, z)
        return (c_a - tgt.to(c_a.dtype)).pow(2).sum(-1).mean()
    if mode == "ce":
        logits = body.logits(a_seg, z)
        return -(tgt.to(logits.dtype) * F.log_softmax(logits, dim=-1)).sum(-1).mean()
    raise ValueError(f"unknown mode {mode!r}; use 'mse' or 'ce'")
