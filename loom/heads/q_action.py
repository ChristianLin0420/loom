"""LOOM — q_a^e: the action-labelled operator encoder.  ONE PER EMBODIMENT, ~30 M each.

`(a_seg (B, H_OP, dof_e), z (B, K, D)) -> Coeff (B, M)`.

Held in a `ModuleDict` keyed by embodiment name. Batches are
embodiment-homogeneous (PLAN 9), so `forward` takes the name and routes once.

WHY THERE IS NO ALIGNMENT LOSS  (and which way the one that exists points)
─────────────────────────────────────────────────────────────────────────
There is exactly ONE stop-gradded regression between the two encoders — the
align half of `L_act`, `loom/losses/act.py::q_action_regression_loss`. That is
what puts both encoders in *one* coefficient space, by construction. No KL, no
adversarial term, no fifth loss (PLAN 9: do not add losses). The head below is
imported from `q_delta.py` rather than re-implemented for exactly the same
reason: two encoders that discretise differently do not share a space no matter
what loss you put between them.

**Since ALIGN-FLIP this head is NOT the one that moves.** `configs/r0a.yaml`
ships `losses.act.align_to: q_a`, i.e. `q_Delta` regresses onto `sg(q_a)`, and
`q_a`'s only gradient is `D_e`'s reconstruction of `a_{t:t+7}` from
`(proprio_t, c)`. The paragraph below diagnoses why the old direction had to
go: with the align term pointed *at* this head it rewards flat coefficients
whenever the two supports are disjoint, which is the state R0-A sat in for 7004
steps (`act/align` pinned at the 0.500 disjoint floor). Read it as the record of
the failure, not as a description of the shipping configuration — with
`align_to: q_a` the align gradient no longer reaches this head at all, and the
`logit_rms` pin still matters for the same reason (an unpredictable target is
now `D_e`'s reconstruction target, and this head still switches itself off while
one is unpredictable).

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

`robotwin_aloha` HAS THE SAME PROBLEM, FOR A DIFFERENT REASON
─────────────────────────────────────────────────────────────
Its 14 channels are ABSOLUTE servo targets -- 12 joint angles in radians and 2
normalised gripper widths -- not deltas, so the spread comes from the reachable
joint range rather than from a per-step magnitude. Measured over all 2500
`demo_clean` trajectories on the canonical 30 Hz grid (988 391 frames,
458 560 segments), the per-dof rms spans 0.258 (wrist roll) to 1.509 (shoulder
pitch), a variance ratio of 34.3x, and the same measurement as above gives a
participation ratio of **2.99 / 1024** with the per-dof deviation share running
0.66% (R_j5) to 23.9% (R_j2) -- 36x -- against a uniform 7.14%. The failure is
not the gripper this time: the two shoulder/elbow joints are 40% of the
embedding between them and the two wrist rolls are 1.9%. With the row below the
participation ratio is 5.55 and the share ratio 4.0x, which is where LIBERO sits
after its own row (measured on real windows with the shipped table: 1.58 -> 6.18,
3267x -> 4.3x).

rms and not std for this body too, for table coherence rather than for the
LIBERO reason -- an absolute joint target has no meaningful zero. Per-dof std
was measured as the alternative and is slightly flatter on share (2.8x) and
slightly worse on the headline metric (participation ratio 4.74), because it
pumps the two correlated gripper channels from 7.6% to 16.4% of the deviation
and they then share one direction.

AND THE ROBOTWIN DECODER STILL DOES NOT NEED IT -- MEASURED, NOT ASSUMED
────────────────────────────────────────────────────────────────────────
The obvious worry is that 14 absolute channels spanning 0.26-1.51 rms would let
the wide shoulder joints eat the CFM regression and starve the grippers. They do
not, and the reason is structural: the flow's source is N(0, I), so the per-dof
loss at `v_theta = 0` is `rms_d^2 + 1` and that `+1` compresses the dynamic
range from the data's 34.3x in variance to **3.07x** in loss share (4.31% to
13.24% against a uniform 7.14%; the grippers sit at 6.69% each). LIBERO's same
number is 2.00x, and LIBERO's decoder was fine.

Run directly (`logs/rt_actstats/dec_norm_probe.py`, one A100, 8000 steps,
batch 256, all 458 560 real segments, held-out every 10th episode, identical
seed and coefficient stream per arm): the UNCHANGED decoder emits per-dof
standard deviations 0.9989x the data's on average and within 1.6% on every one
of the 14 dofs, correlates 0.998-0.9997 per dof with the absolute target and
0.81-0.99 (mean 0.911) with the proprio-relative residual, calls the gripper on
the right side of 0.5 in 99.43% / 99.71% of held-out steps, and puts 0.000% of
its samples outside the action box. Dividing `a_seg` by the row above and
multiplying back in `forward` moves mean residual correlation by +0.005, makes
mean residual RMSE 5% WORSE (0.02288 -> 0.02413 rad), and specifically makes the
gripper -- the channel that decides success -- 37% worse (0.0116 -> 0.0159 rad).
Normalising the conditioning proprio too changes nothing further. With an
uninformative coefficient all four arms land inside 3% of each other. So the
seam below stays exactly where it is, and eval keeps seeing radians.

A body with no entry in `ACTION_RMS` gets ones, i.e. exactly the previous
behaviour. Add a row when its corpus statistics have actually been measured, not
before.


THE SEGMENT-ANCHORED DELTA BRANCH  (`DELTA_RMS`, ABSOLUTE bodies only)
─────────────────────────────────────────────────────────────────────
`ACTION_RMS` fixes which dof the head can see. It does NOT fix *when*: this head
must emit a DIFFERENT `c` for each of the DEPTH=4 horizons in a window, and on
`robotwin_aloha` there is almost nothing in `a_seg` to tell them apart.

Measured on real windows cut exactly as the loader cuts them (per-segment means
over `H_OP`, divided by `ACTION_RMS`; 18 360 RoboTwin / 9 082 LIBERO windows,
`logs/dual_gate/gate.py`):

    body              within-window   between-window   ratio
    libero_franka        0.4816           0.8494        0.567
    robotwin_aloha       0.1215           0.8451        0.144

Between-window variation is identical -- both bodies discriminate windows fine.
WITHIN a window RoboTwin's action moves 4x less. LIBERO's actions are deltas and
genuinely differ across horizons; RoboTwin's are 14 absolute joint targets in a
slow trajectory, so `a_seg` is nearly constant across the window. With nothing
in the action distinguishing h=0 from h=3, phase is the only thing left, and `c`
becomes a clock even with `losses.act.align_to: q_a` on -- which is what
`runs/r0b2` did (`bank/live_ops_q_a` 29 -> 15, `loss/proposal` back to 18.96
against the 19.361 uniform Plackett-Luce floor, while `act/decode` fell healthily
0.91 -> 0.23; the decoder was never the problem).

So ABSOLUTE-semantics bodies get a SECOND input branch into the same `d_act`
space: `(a_seg - a_seg[:, :1, :]) / DELTA_RMS`, the displacement of the segment
from its own first step, over the dims registered ABSOLUTE in
`loom/data/canonical.py`. `a_seg[0]` is the anchor and not proprio because q_a's
`(a_seg, z)` signature is frozen (PLAN 4.C) -- and on this body
`action[t] == state[t+1]` bitwise, so `a_seg[0]` IS a proprio reading one control
step ahead. What this anchor drops against `decoder.residual`'s
`(a_seg - proprio_t)` is a per-segment CONSTANT, which the absolute branch still
carries in full.

WHY ANCHORED PER SEGMENT AND NOT CENTRED PER WINDOW. The encoding is a function
of `a_seg` ALONE, so the same physical 8-step segment appearing at h=0 in one
window and h=3 in another encodes identically. Verified exactly, not by probe:
over 2091 physical segments that occur at more than one horizon index, the
max absolute difference in the delta branch's output is 0.000e+00
(`logs/dual_gate/encgate.py`). Window-centring would need the other three
segments, so the same segment would encode differently depending on its
neighbours -- irreducible label noise for both the align target and pi_c's BC
target.

WHY BOTH BRANCHES AND NOT A DELTA REPLACEMENT. A linear probe recovering the
segment's own absolute pose from the encoding (fp32, rank-truncated, held out,
`logs/dual_gate/probes.py`) gives R^2 0.974 today, 0.956 for DUAL, and 0.156 for
delta-only -- with the two gripper channels, which the decoder docstring above
calls the channel that decides success, at 0.975/0.973, 0.953/0.945 and
0.191/0.208. A pure-delta replacement throws the pose away; keeping both does
not.

WHAT IT BUYS, AT THE ONLY LEVEL THAT MATTERS -- the 1024-d `encode_action` output
the trunk actually receives, real module at fresh init, bf16, 2048 real windows:

    LIBERO today     within/between 0.644   within-window VARIANCE SHARE 29.3%
    robotwin today   within/between 0.165   within-window VARIANCE SHARE  2.66%
    robotwin DUAL    within/between 0.427   within-window VARIANCE SHARE 15.41%

a 5.8x increase, landing at ~half of LIBERO's, with `between` untouched
(0.4343 -> 0.4527) -- the healthy axis is not traded away.

AND IT IS NOT A CLOCK. A 4-way linear probe for the horizon index from the
encoding (chance 0.250) reads 0.201 today and 0.200 for DUAL on held-out data,
against 0.202 for the raw `a_seg` itself: `h` is not linearly decodable from one
segment on this body under ANY of these encodings, and the branch does not make
it so.

This is a reparameterisation, not a capacity change. `act_out` is a Linear over
the flattened `(H_OP, d_act)`, so `a[t] - a[0]` is already inside its span; what
changes is that the difference arrives pre-normalised at O(1) instead of as a
2.66%-of-variance direction -- exactly the argument `ACTION_RMS` is justified by
above.

A BODY ABSENT FROM `DELTA_RMS` GETS NO BRANCH: no parameters, no state_dict keys,
and an `encode_action` that is byte-identical to the one before this existed
(`torch.equal`, fp32 and bf16, verified against the module at the previous commit
in `logs/dual_gate/identity.py`). `libero_franka` is deliberately absent and is
doubly excluded -- its semantics are `(delta,)*6 + (hold,)`, i.e. ZERO absolute
dims. Differencing a delta channel gives acceleration; differencing a latched
gripper gives a spike train.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from contracts import D, EMBODIMENTS, H_OP, K, M, TOPK
from loom.data.canonical import ABSOLUTE, action_semantics
from loom.heads.q_delta import AttnPool, mlp_trunk, topk_simplex_st

__all__ = ["ACTION_RMS", "DELTA_RMS", "DELTA_GAIN", "LOGIT_RMS",
           "QActionBody", "QAction", "absolute_dims"]


#: Per-dof rms of an action at `FPS_CANONICAL`, per embodiment. Measured, not
#: guessed: `libero_franka` is all 507 363 canonical frames of the 2000
#: trajectories the R0-A loader serves. A body that is absent is not normalised.
#: See the module docstring.
#: `robotwin_aloha` is all 988 391 canonical frames of the 2500 demo_clean
#: trajectories the R0-B loader serves, in the layout
#: [L_arm j1..j6 | L_grip | R_arm j1..j6 | R_grip].
ACTION_RMS: dict[str, tuple[float, ...]] = {
    "libero_franka": (0.22594, 0.25572, 0.29705, 0.02697, 0.04227, 0.05330, 1.0),
    "robotwin_aloha": (0.43019, 1.47891, 1.12173, 0.75236, 0.25784, 0.56114, 0.80985,
                       0.39910, 1.50929, 1.15780, 0.80002, 0.26210, 0.61682, 0.80965),
}

#: Operator-axis rms every q_a logit row is pinned to. A constant, never learned.
#: 1.0 is where a freshly initialised head already sits (measured spread 0.929 on
#: real windows), so this changes the scale of nothing at step 0 -- it only stops
#: the scale from being driven to zero afterwards.
LOGIT_RMS = 1.0


#: Per-dof rms of the SEGMENT-ANCHORED displacement `a_seg - a_seg[:, :1, :]` on
#: the canonical 30 Hz grid. Measured over all 2500 demo_clean trajectories,
#: 458 560 segments cut exactly as canonical.segment cuts them -- the same corpus
#: and the same segment set as decoder.RESIDUAL_RMS (logs/qa_dual/drms.py).
#: A BODY ABSENT FROM THIS TABLE GETS NO DELTA BRANCH: zero new parameters, zero
#: new state_dict keys, byte-identical `encode_action`. `libero_franka` is
#: deliberately absent and is doubly excluded -- its semantics are
#: (delta,)*6 + (hold,), i.e. ZERO absolute dims. Differencing a delta channel
#: gives acceleration; differencing a latched gripper gives a spike train.
DELTA_RMS: dict[str, tuple[float, ...]] = {
    # [L_j1..L_j6, L_grip | R_j1..R_j6, R_grip]
    "robotwin_aloha": (0.025717, 0.066233, 0.056913, 0.050135, 0.018478,
                       0.038535, 0.048000,
                       0.023919, 0.064609, 0.057042, 0.047655, 0.018305,
                       0.039700, 0.048444),
}

#: Multiplier on the delta branch. 1.0 = both branches enter at unit rms through
#: Linears of identical fan_in and init family, i.e. NO free parameter. The
#: pre-measured sweep, if it ever has to move (within-window variance share of
#: `encode_action`, bf16, real windows; LIBERO today is 29.34%):
#:   gain 0.0 -> 2.49%   1.0 -> 15.41%   1.5 -> 20.31%
#:        2.0 -> 23.73%  3.0 -> 27.99%   4.0 -> 30.43%
DELTA_GAIN = 1.0


def absolute_dims(embodiment: str | None, dof: int) -> tuple[int, ...]:
    """Which action dims are ABSOLUTE servo targets, hence worth differencing.

    `()` unless the body has a MEASURED row in DELTA_RMS -- that membership, and
    not a config key, is the opt-in, exactly as for ACTION_RMS. A body that has
    the row but no registered semantics raises: that combination is a bug, not a
    default.
    """
    if not embodiment or embodiment not in DELTA_RMS:
        return ()
    kinds = action_semantics(embodiment)
    if len(kinds) != dof:
        raise ValueError(f"{embodiment}: {len(kinds)} action kinds for dof {dof}")
    return tuple(i for i, k in enumerate(kinds) if k == ABSOLUTE)


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
        delta_dims: Sequence[int] | None = None,
        delta_rms: Sequence[float] | None = None,
        delta_gain: float = DELTA_GAIN,
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

        # ── the within-window branch (ABSOLUTE-semantics bodies only) ─────
        # Deliberately constructed LAST, so a body that has no delta branch
        # draws an identical module-init RNG stream to the one it drew before
        # this branch existed, and so does every module built after q_action.
        if delta_dims is None:
            delta_dims = absolute_dims(embodiment, dof)
        self.delta_dims = tuple(int(i) for i in delta_dims)
        self.delta_gain = float(delta_gain)
        self.delta_in = None
        if self.delta_dims:
            if any(i < 0 or i >= dof for i in self.delta_dims):
                raise ValueError(
                    f"delta_dims {self.delta_dims} out of range for dof {dof}"
                )
            src = delta_rms if delta_rms is not None else DELTA_RMS[embodiment]
            vals = tuple(float(src[i]) for i in self.delta_dims)
            if any(v <= 0.0 for v in vals):
                raise ValueError(f"delta_rms must be positive elementwise, got {vals}")
            self.register_buffer("delta_idx",
                                 torch.tensor(self.delta_dims, dtype=torch.long),
                                 persistent=False)
            self.register_buffer("delta_rms", torch.tensor(vals), persistent=False)
            self.delta_in = nn.Linear(len(self.delta_dims), d_act)

    def encode_action(self, a_seg: Tensor) -> Tensor:
        if a_seg.shape[-2:] != (self.h_op, self.dof):
            raise ValueError(
                f"action segment must be (..., {self.h_op}, {self.dof}), "
                f"got {tuple(a_seg.shape)}"
            )
        x = self.step_in(a_seg / self.action_rms.to(a_seg)) + self.step_emb.to(a_seg.dtype)
        if self.delta_in is not None:
            # `a_seg[0]` and not proprio: q_a's signature is (a_seg, z), PLAN 4.C,
            # frozen. On this body action[t] == state[t+1] bitwise, so a_seg[0]
            # IS a proprio reading one control step ahead. What this anchor drops
            # against decoder.residual's (a_seg - proprio_t) is a per-segment
            # CONSTANT, which the absolute branch above still carries in full.
            g = a_seg.index_select(-1, self.delta_idx)
            g = (g - g[..., :1, :]) / self.delta_rms.to(g)
            x = x + self.delta_gain * self.delta_in(g)
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
