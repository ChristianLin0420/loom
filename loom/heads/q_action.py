"""LOOM — q_a^e: the action-labelled operator encoder.  ONE PER EMBODIMENT, ~30 M each.

`(a_seg (B, H_OP, dof_e), z (B, K, D)) -> Coeff (B, M)`.

Held in a `ModuleDict` keyed by embodiment name. Batches are
embodiment-homogeneous (PLAN 9), so `forward` takes the name and routes once.

WHY THERE IS NO ALIGNMENT LOSS
──────────────────────────────
`q_a^e` is trained by plain regression onto `sg(q_Delta(z_t, z_{t+8}))` — see
`loom/losses/act.py::q_action_regression_loss`. That single stop-gradded
regression is what puts both encoders in *one* coefficient space, by
construction. No KL, no adversarial term, no separate alignment loss (PLAN 9:
do not add losses). The head below is imported from `q_delta.py` rather than
re-implemented for exactly the same reason: two encoders that discretise
differently do not share a space no matter what loss you put between them.

The action encoder is deliberately tiny relative to the trunk. `a_seg` is only
`H_OP * dof_e` = 56 numbers for LIBERO; the hard part is not reading the action,
it is deciding which operator that action *means given this belief*, which is a
joint function of both inputs and lives in the trunk.


WHY THE LOGIT SCALE IS PINNED  (`logit_rms`, and R0-A's actual failure)
──────────────────────────────────────────────────────────────────────
R0-A trained this head for 7004 steps and it never became a function of its
inputs: in fp32, on 1536 real LIBERO windows, it emitted ONE top-4 support --
[34, 56, 68, 126] -- for every window, and shuffling `a_seg` across the batch
moved the logits by 1e-5 rms against a spread of 0.019.

That is not an architecture that cannot see its inputs. A *freshly initialised*
head of exactly this shape, on exactly those beliefs and actions, gives 294
distinct supports over 1536 windows, snr 0.88 and pairwise cosine 0.57 at
`trunk[4]`, and moves 0.749 rms under the same shuffle. The blindness is
LEARNED, and `CenteredReadout`'s `logit_rms` is what stops it being learned;
see that class for the fixed-point argument and the measurements.

The short version: `L_act`'s regression term rewards flat coefficients whenever
q_a's support and q_Delta's are disjoint, the straight-through head pushes the
current top-4 down 32x harder than it pulls the target's atoms up, and the fixed
point of that is every logit equal -- reached most cheaply by collapsing the
hidden state onto a ray. Pinning the operator-axis rms makes the loss exactly
scale-invariant in the logits, which does NOT stop the head switching itself off
while its target is unpredictable, but does stop that being permanent: measured,
after 1500 steps of a deliberately unlearnable target both heads sit at one
support and pairwise cosine 1.0000, and over the next 1500 steps of a learnable
one only the pinned head comes back (cosine 0.63 and align 0.375 against 0.99999
and 0.5248). The scale it would need to recover with has not been destroyed.


WHY `a_seg` IS DIVIDED BY A PER-DOF CONSTANT  (`ACTION_RMS`)
───────────────────────────────────────────────────────────
LIBERO's 7 dof do not share a scale. Measured over the whole 30 Hz corpus
(507 363 frames), the per-dof rms is

    dx 0.226   dy 0.256   dz 0.297   droll 0.027   dpitch 0.042   dyaw 0.053
    gripper 1.000

`step_in` is one `Linear(dof, d_act)` with a shared initialisation, so each dof
enters the embedding in proportion to its variance: the binary gripper is 82% of
it and the three rotation dofs together are 0.44%. Measured at initialisation on
real windows, the batch deviation of `encode_action` has a participation ratio of
1.5 out of 1024 dimensions and its first principal component correlates 0.993
with the gripper bit. The head is handed a 7-dof action and sees one bit.

Dividing by `ACTION_RMS` is a fixed, per-body reparameterisation of the encoder's
input -- rms and not std, so a zero action still maps to `step_in.bias` and the
sign structure is untouched. It is confined to this module: `q_a` emits a
coefficient, never an action, so there is no inverse to maintain anywhere and no
eval seam. `loom/eval/policy.py` and `DecoderBody` are deliberately NOT touched;
the decoder was measured at step 7004 emitting per-dof standard deviations of
[0.228, 0.255, 0.315, 0.027, 0.031, 0.043, 0.999] against data
[0.232, 0.248, 0.311, 0.024, 0.047, 0.045, 1.000] -- within 3% on four dofs and a
third on the worst rotation -- and correlating 0.83/0.90/0.90/0.42/0.55/0.62/0.97
per dof with the ground-truth segment. It does not have this problem, so
normalising it would buy nothing and create an inverse to get wrong. If that ever
changes, the seam is `DecoderBody.loss` (divide `a_seg`) and `DecoderBody.forward`
(multiply back BEFORE the `action_low/high` clamp), both inside this package, so
`loom/eval/policy.py` still never sees normalised units.

A body with no entry in `ACTION_RMS` gets ones, i.e. exactly the previous
behaviour. Add a row when its corpus statistics have actually been measured, not
before.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from contracts import D, EMBODIMENTS, H_OP, K, M, TOPK
from loom.heads.q_delta import AttnPool, mlp_trunk, topk_simplex_st

__all__ = ["ACTION_RMS", "LOGIT_RMS", "QActionBody", "QAction"]


#: Per-dof rms of an action at `FPS_CANONICAL`, per embodiment. Measured, not
#: guessed: `libero_franka` is all 507 363 canonical frames of the 2000
#: trajectories the R0-A loader serves. A body that is absent is not normalised.
#: See the module docstring.
ACTION_RMS: dict[str, tuple[float, ...]] = {
    "libero_franka": (0.22594, 0.25572, 0.29705, 0.02697, 0.04227, 0.05330, 1.0),
}

#: Operator-axis rms every q_a logit row is pinned to. A constant, never learned.
#: 1.0 is where a freshly initialised head already sits (measured spread 0.929 on
#: real windows), so this changes the scale of nothing at step 0 -- it only stops
#: the scale from being driven to zero afterwards.
LOGIT_RMS = 1.0


class QActionBody(nn.Module):
    """The per-embodiment encoder. One of these per registered body, ~30 M."""

    def __init__(
        self,
        dof: int,
        embodiment: str | None = None,
        hidden: int = 2560,
        n_queries: int = 4,
        n_heads: int = 8,
        n_hidden: int = 3,
        d_act: int = 512,
        d_act_out: int = 1024,
        topk: int = TOPK,
        temperature: float = 1.0,
        d: int = D,
        n_ops: int = M,
        n_slots: int = K,
        h_op: int = H_OP,
        d_kv: int | None = None,
        action_rms: Sequence[float] | None = None,
        logit_rms: float | None = LOGIT_RMS,
    ) -> None:
        super().__init__()
        self.dof, self.h_op = dof, h_op
        self.topk, self.temperature = topk, temperature
        self.embodiment = embodiment

        # belief side
        self.pool = AttnPool(d=d, n_queries=n_queries, n_heads=n_heads,
                             n_slots=n_slots, d_kv=d_kv)

        # action side: per-step embedding + learned step position, then one
        # projection over the flattened segment (H_OP is 8; attention is overkill)
        self.step_in = nn.Linear(dof, d_act)
        # zero-init is fine here (unlike the decoder's): the segment is flattened
        # into one projection, so step order is already carried by the weight
        # layout rather than by this embedding.
        self.step_emb = nn.Parameter(torch.zeros(h_op, d_act))
        self.act_norm = nn.LayerNorm(h_op * d_act)
        self.act_out = nn.Linear(h_op * d_act, d_act_out)

        # Per-dof input scale (module docstring). Non-persistent: it is a
        # measured constant of the body, not a learned tensor, so it must not
        # ride in a checkpoint where it could silently diverge from the table.
        if action_rms is None:
            action_rms = ACTION_RMS.get(embodiment or "", (1.0,) * dof)
        if len(action_rms) != dof:
            raise ValueError(
                f"action_rms has {len(action_rms)} entries, dof is {dof}"
                + (f" (embodiment {embodiment!r})" if embodiment else "")
            )
        if any(float(v) <= 0.0 for v in action_rms):
            raise ValueError(f"action_rms must be positive elementwise, got {tuple(action_rms)}")
        self.register_buffer("action_rms", torch.tensor(tuple(float(v) for v in action_rms)),
                             persistent=False)

        self.trunk = mlp_trunk(n_queries * d + d_act_out, hidden, n_ops,
                               n_hidden=n_hidden, logit_rms=logit_rms)

    def encode_action(self, a_seg: Tensor) -> Tensor:
        if a_seg.shape[-2:] != (self.h_op, self.dof):
            raise ValueError(
                f"action segment must be (..., {self.h_op}, {self.dof}), "
                f"got {tuple(a_seg.shape)}"
            )
        x = self.step_in(a_seg / self.action_rms.to(a_seg)) + self.step_emb.to(a_seg.dtype)
        x = x.flatten(-2)
        return self.act_out(self.act_norm(x))

    def logits(self, a_seg: Tensor, z: Tensor) -> Tensor:
        h = torch.cat([self.pool(z), self.encode_action(a_seg.to(z.dtype))], dim=-1)
        return self.trunk(h)

    def forward(self, a_seg: Tensor, z: Tensor, return_logits: bool = False):
        lg = self.logits(a_seg, z)
        c = topk_simplex_st(lg, self.topk, self.temperature)
        return (c, lg) if return_logits else c


class QAction(nn.Module):
    """`ModuleDict` of per-embodiment `QActionBody`. Satisfies `contracts.QAction`.

    Args:
        embodiments: names to build. Default: every body registered in
            `contracts.EMBODIMENTS` at construction time. Modules are built
            eagerly (never lazily inside `forward`) so that the optimizer and
            FSDP see a fixed parameter set; register late bodies explicitly with
            `add_embodiment`.
        default_embodiment: used when `forward` is called without a name. Falls
            back to the single registered body if there is exactly one.
    """

    def __init__(
        self,
        embodiments: list[str] | tuple[str, ...] | None = None,
        default_embodiment: str | None = None,
        **body_kwargs,
    ) -> None:
        super().__init__()
        names = tuple(EMBODIMENTS) if embodiments is None else tuple(embodiments)
        if not names:
            raise ValueError("no embodiments registered; nothing to build")
        self.body_kwargs = dict(body_kwargs)
        self.bodies = nn.ModuleDict()
        for name in names:
            self.add_embodiment(name)
        self.default_embodiment = default_embodiment or (names[0] if len(names) == 1 else None)

    def add_embodiment(self, name: str) -> QActionBody:
        if name not in EMBODIMENTS:
            raise KeyError(f"unregistered embodiment {name!r}")
        if name not in self.bodies:
            # `embodiment=` and not just `dof=`: two registered bodies already
            # share dof 7, so the per-dof action scale cannot be inferred from
            # the width. Passing the name is how the right row of ACTION_RMS
            # gets picked instead of a plausible wrong one.
            self.bodies[name] = QActionBody(EMBODIMENTS[name].dof, embodiment=name,
                                            **self.body_kwargs)
        return self.bodies[name]

    def body(self, embodiment: str | None = None) -> QActionBody:
        name = embodiment or self.default_embodiment
        if name is None:
            raise ValueError(
                "q_action holds several bodies; pass embodiment= "
                f"(have {sorted(self.bodies)})"
            )
        if name not in self.bodies:
            raise KeyError(f"q_action has no body {name!r}; have {sorted(self.bodies)}")
        return self.bodies[name]

    def logits(self, a_seg: Tensor, z: Tensor, embodiment: str | None = None) -> Tensor:
        return self.body(embodiment).logits(a_seg, z)

    def forward(
        self,
        a_seg: Tensor,
        z: Tensor,
        embodiment: str | None = None,
        return_logits: bool = False,
    ):
        return self.body(embodiment)(a_seg, z, return_logits=return_logits)
