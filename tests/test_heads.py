"""LOOM — Team C gate, heads half.

`q_Delta`, `q_a^e`, `D_e`. Imports nothing from `loom.model` and nothing from
`loom.heads.proposal`: everything here runs against `contracts` + `stubs`.

Covers PLAN 4.C done-when items 3, 4, 5, 6, 10, 11.
"""

from __future__ import annotations

import importlib
import pathlib
import sys

import pytest
import torch

import contracts as C
import stubs as S
from loom.data.canonical import ABSOLUTE, action_semantics
from loom.heads.decoder import Decoder, DecoderBody
from loom.heads.q_action import (ACTION_RMS, DELTA_RMS, LOGIT_RMS, QAction,
                                 QActionBody, absolute_dims)
from loom.heads.q_delta import AttnPool, CenteredReadout, QDelta, topk_simplex_st

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

torch.manual_seed(0)


# ═══════════════════════════════════════════════════════════════════════════
#  TWO SYNTHETIC BODIES WITH DIFFERENT dof   (done-when 4)
# ═══════════════════════════════════════════════════════════════════════════

DOF_A, DOF_B = 7, 14
BODY_A = C.register_embodiment(C.EmbodimentSpec(
    name="teamc_toy7", dof=DOF_A, env_fps=20.0, n_views=2,
    action_low=(-1.0,) * DOF_A, action_high=(1.0,) * DOF_A,
)).name
BODY_B = C.register_embodiment(C.EmbodimentSpec(
    name="teamc_toy14", dof=DOF_B, env_fps=50.0, n_views=3,
    action_low=(-2.0,) * DOF_B, action_high=(2.0,) * DOF_B,
)).name

#: small configurations — the full-size modules are exercised only in the
#: parameter-budget and full-width bf16 tests, which are marked slow. `d_kv`
#: narrows the attention pool, which is the entire CPU cost of these heads.
SMALL_QD = dict(hidden=64, n_queries=1, n_heads=4, d_kv=32)
SMALL_QA = dict(hidden=64, n_queries=1, n_heads=4, d_kv=32,
                d_act=16, d_act_out=32, n_hidden=2)
SMALL_DEC = dict(d=32, n_blocks=1, n_heads=2, n_queries=1, pool_heads=4,
                 d_kv=32, n_steps=3)


def n_params(m) -> int:
    return sum(p.numel() for p in m.parameters())


def belief(b: int = 3, dtype=torch.float32) -> torch.Tensor:
    return torch.randn(b, C.K, C.D, dtype=dtype)


def proprio(b: int = 3, dof: int = DOF_A, dtype=torch.float32) -> torch.Tensor:
    """`ObsFeats["proprio"]` — ONE timestep, `(B, dof_e)`.

    This, and NOT the belief, is what `D_e` takes. See `loom/heads/decoder.py`:
    given the whole `(B, K, D)` belief the decoder is a behaviour-cloning head
    and `L_act` puts no pressure on `c` at all.
    """
    return torch.randn(b, dof, dtype=dtype)


@torch.no_grad()
def _pin_field(dec: DecoderBody, value: float) -> None:
    """Pin v_theta to a known constant so the CFM algebra can be checked exactly."""
    dec.x_out.weight.zero_()
    dec.x_out.bias.fill_(value)


# ═══════════════════════════════════════════════════════════════════════════
#  THE SHARED TOP-K SIMPLEX HEAD   (done-when 3, 5)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_topk_head_is_on_simplex(dtype):
    c = topk_simplex_st(torch.randn(64, C.M, dtype=dtype) * 3.0)
    assert c.dtype is dtype
    C.assert_simplex(c)


def test_topk_head_is_exactly_the_renormalised_hard_topk():
    """Forward value must be the HARD point, not the soft one. A plain softmax
    sums to 1 as well but has M nonzeros, which breaks both of Team B's bounds."""
    logits = torch.randn(16, C.M) * 2.0
    c = topk_simplex_st(logits)
    soft = torch.softmax(logits, -1)
    vals, idx = soft.topk(C.TOPK, dim=-1)
    expect = torch.zeros_like(soft).scatter_(-1, idx, vals / vals.sum(-1, keepdim=True))
    assert torch.allclose(c, expect, atol=1e-6)
    assert (c > 0).sum(-1).eq(C.TOPK).all()
    assert not torch.allclose(c, soft)                       # not a softmax


def test_topk_head_support_matches_argmax_of_logits():
    logits = torch.randn(8, C.M)
    c = topk_simplex_st(logits)
    assert torch.equal(c.topk(C.TOPK, -1).indices.sort(-1).values,
                       logits.topk(C.TOPK, -1).indices.sort(-1).values)


def test_ste_gradient_reaches_logits_and_the_dead_operators():
    """done-when 5: c.sum().backward() reaches the logits, and the gradient is
    nonzero OUTSIDE the top-k support.

    The out-of-support path is the whole point: L_balance can only resurrect a
    dead operator if raising its logit raises its coefficient in the surrogate.
    A renormalisation with a *differentiable* denominator cancels the global
    softmax normaliser and leaves out-of-support logits with exactly zero
    gradient; this asserts we did not do that.
    """
    logits = (torch.randn(4, C.M) * 2.0).requires_grad_(True)
    c = topk_simplex_st(logits)
    c.sum().backward()

    g = logits.grad
    assert g is not None and torch.isfinite(g).all()
    support = c.detach() > 0
    assert (g[support] != 0).all(), "no gradient inside the support"
    assert (g[~support] != 0).all(), "dead operators receive no gradient"
    # and it is the right sign: push the losers down
    assert (g[~support] < 0).all()


def test_ste_gradient_is_dense_under_a_weighted_objective():
    logits = (torch.randn(4, C.M) * 2.0).requires_grad_(True)
    c = topk_simplex_st(logits)
    (c * torch.randn_like(c)).sum().backward()
    assert (logits.grad.abs() > 0).all()


def test_topk_head_temperature_only_changes_the_gradient_scale():
    logits = torch.randn(8, C.M)
    a, b = topk_simplex_st(logits, temperature=1.0), topk_simplex_st(logits, temperature=0.5)
    assert torch.equal((a > 0), (b > 0))                     # same support
    C.assert_simplex(b)


def test_attn_pool_is_not_permutation_invariant_over_slots():
    """Slot identity must survive pooling — mean-pool would fail this, and so
    would attention without the learned slot embedding."""
    pool = AttnPool(d=C.D, n_queries=2, n_heads=4, d_kv=32)
    z = belief(2)
    perm = torch.randperm(C.K)
    assert not torch.allclose(pool(z), pool(z[:, perm]), atol=1e-5)

    with torch.no_grad():                       # ablate the slot embedding
        pool.slot_emb.zero_()
    assert torch.allclose(pool(z), pool(z[:, perm]), atol=1e-4)


# ═══════════════════════════════════════════════════════════════════════════
#  q_Delta   (done-when 3, 10, 11)
# ═══════════════════════════════════════════════════════════════════════════

def test_q_delta_satisfies_the_protocol_and_returns_a_simplex():
    q = QDelta(**SMALL_QD)
    assert isinstance(q, C.QDelta)
    c = q(belief(5), belief(5))
    assert c.shape == (5, C.M)
    C.assert_simplex(c)


def test_q_delta_uses_the_delta_stream():
    """z_next - z_t is handed to the network explicitly; changing only z_next
    must change the coefficients."""
    q = QDelta(**SMALL_QD).eval()
    z = belief(4)
    with torch.no_grad():
        a = q.logits(z, z + 0.1 * torch.randn_like(z))
        b = q.logits(z, z + 3.0 * torch.randn_like(z))
    assert not torch.allclose(a, b, atol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_q_delta_dtype_is_preserved(dtype):
    """done-when 11: bf16 in, bf16 out, no promotion, never complex."""
    q = QDelta(**SMALL_QD).to(dtype)
    c, lg = q(belief(3, dtype), belief(3, dtype), return_logits=True)
    assert c.dtype is dtype and lg.dtype is dtype
    assert not c.is_complex()
    C.assert_simplex(c)


def test_q_delta_gradient_flows_to_every_parameter():
    q = QDelta(**SMALL_QD)
    (q(belief(3), belief(3)) * torch.randn(3, C.M)).sum().backward()
    dead = [n for n, p in q.named_parameters() if p.grad is None or not p.grad.any()]
    assert not dead, f"no gradient into {dead}"


@pytest.mark.slow
def test_q_delta_parameter_budget():
    """done-when 10: q_Delta ~ 30 M (PLAN 2 budget table), +/- 40%."""
    n = n_params(QDelta())
    assert 18e6 <= n <= 42e6, f"q_delta is {n / 1e6:.1f} M, budget is 30 M"


# ═══════════════════════════════════════════════════════════════════════════
#  q_a^e — PER-EMBODIMENT DISPATCH   (done-when 3, 4, 10, 11)
# ═══════════════════════════════════════════════════════════════════════════

def test_q_action_satisfies_the_protocol():
    assert isinstance(QAction([BODY_A], **SMALL_QA), C.QAction)


@pytest.mark.parametrize("body,dof", [(BODY_A, DOF_A), (BODY_B, DOF_B)])
def test_q_action_routes_by_embodiment(body, dof):
    """done-when 4: two bodies with different dof, routed correctly."""
    q = QAction([BODY_A, BODY_B], **SMALL_QA)
    c = q(torch.randn(3, C.H_OP, dof), belief(3), embodiment=body)
    assert c.shape == (3, C.M)
    C.assert_simplex(c)


def test_q_action_rejects_the_wrong_dof():
    q = QAction([BODY_A, BODY_B], **SMALL_QA)
    with pytest.raises(ValueError):
        q(torch.randn(3, C.H_OP, DOF_B), belief(3), embodiment=BODY_A)
    with pytest.raises(KeyError):
        q(torch.randn(3, C.H_OP, DOF_A), belief(3), embodiment="not_a_body")
    with pytest.raises(ValueError):
        q(torch.randn(3, C.H_OP, DOF_A), belief(3))          # ambiguous, 2 bodies


def test_q_action_bodies_have_separate_parameters():
    q = QAction([BODY_A, BODY_B], **SMALL_QA)
    a = dict(q.body(BODY_A).named_parameters())
    b = dict(q.body(BODY_B).named_parameters())
    assert a.keys() == b.keys()
    assert all(a[k] is not b[k] for k in a)
    assert a["step_in.weight"].shape[1] == DOF_A
    assert b["step_in.weight"].shape[1] == DOF_B


def test_q_action_gradient_is_confined_to_the_addressed_body():
    """Embodiment-homogeneous batches (PLAN 9): the other body must not move."""
    q = QAction([BODY_A, BODY_B], **SMALL_QA)
    c = q(torch.randn(2, C.H_OP, DOF_A), belief(2), embodiment=BODY_A)
    (c * torch.randn_like(c)).sum().backward()
    assert all(p.grad is None or not p.grad.any() for p in q.body(BODY_B).parameters())
    assert any(p.grad is not None and p.grad.any() for p in q.body(BODY_A).parameters())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_q_action_dtype_is_preserved(dtype):
    q = QAction([BODY_A], **SMALL_QA).to(dtype)
    c = q(torch.randn(2, C.H_OP, DOF_A, dtype=dtype), belief(2, dtype), embodiment=BODY_A)
    assert c.dtype is dtype and not c.is_complex()
    C.assert_simplex(c)


def test_q_action_and_q_delta_share_the_identical_head():
    """Same logits must give the same coefficients out of both encoders,
    otherwise they are not writing into one coefficient space."""
    logits = torch.randn(6, C.M)
    q = QDelta(**SMALL_QD)
    qa = QAction([BODY_A], **SMALL_QA)
    from loom.heads import q_action as qa_mod, q_delta as qd_mod
    assert qa_mod.topk_simplex_st is qd_mod.topk_simplex_st
    assert torch.equal(topk_simplex_st(logits, q.topk, q.temperature),
                       topk_simplex_st(logits, qa.body(BODY_A).topk,
                                       qa.body(BODY_A).temperature))


@pytest.mark.slow
def test_q_action_parameter_budget_per_body():
    """done-when 10: q_a ~ 30 M per body, +/- 40%."""
    q = QAction([BODY_A])
    n = n_params(q.body(BODY_A))
    assert 18e6 <= n <= 42e6, f"q_action is {n / 1e6:.1f} M/body, budget is 30 M"


# ═══════════════════════════════════════════════════════════════════════════
#  q_a^e — WHAT KEEPS IT A FUNCTION OF ITS INPUTS
#
#  R0-A trained this head for 7004 steps and it ended up emitting ONE top-4
#  support for all 1536 probed windows, in fp32, with `act/align` pinned on its
#  disjoint-support floor of 0.500 the whole way. The head that shape produces
#  at INITIALISATION gives 294 distinct supports on the same inputs, so the
#  blindness was learned. These are the two mechanisms that stop it being
#  learned, plus the fixed-point argument they rest on.
# ═══════════════════════════════════════════════════════════════════════════

def test_q_action_logit_rms_is_pinned_per_row():
    q = QAction([BODY_A], **SMALL_QA)
    lg = q.logits(torch.randn(16, C.H_OP, DOF_A) * 7.0, belief(16), embodiment=BODY_A)
    rms = lg.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.full_like(rms, LOGIT_RMS), atol=1e-5)
    assert lg.mean(-1).abs().max() < 1e-5          # centred, as CenteredReadout promises


def test_pinning_the_rms_does_not_change_the_ranking():
    """It is a positive per-row rescale of a mean-free vector: same top-k, always."""
    pinned = CenteredReadout(32, C.M, logit_rms=LOGIT_RMS)
    plain = CenteredReadout(32, C.M, logit_rms=None)
    plain.load_state_dict(pinned.state_dict())
    h = torch.randn(64, 32)
    a, b = pinned(h), plain(h)
    assert torch.equal(a.topk(C.TOPK, -1).indices, b.topk(C.TOPK, -1).indices)
    assert torch.equal(a.argsort(-1), b.argsort(-1))
    # ... and the simplex point is still a hard, renormalised top-k
    C.assert_simplex(topk_simplex_st(a, C.TOPK))


def test_q_delta_readout_is_deliberately_not_pinned():
    """q_Delta is trained through the bank by L_dyn, has no flattening pressure,
    and measured a healthy spread of 1.41. Pinning it would change Team B's
    inputs for no reason."""
    assert QDelta(**SMALL_QD).trunk[-1].logit_rms is None
    lg = QDelta(**SMALL_QD).logits(belief(8), belief(8))
    assert lg.pow(2).mean(-1).sqrt().std() > 1e-4        # free to vary per sample


def test_a_pinned_readout_gets_exactly_zero_gradient_from_the_radial_direction():
    """The whole mechanism in one assertion.

    `sum(logits^2)` is a pure function of the logit SCALE. Under the pin it is
    the constant `B * M * logit_rms^2`, so its gradient is analytically zero --
    the direction q_a died along is not merely opposed, it is annihilated. What
    survives is fp32 round-off in the normalisation, five orders of magnitude
    below the gradient the unpinned readout gets from the same objective.
    """
    torch.manual_seed(0)
    h = torch.randn(32, 16)
    g = {}
    for pin in (LOGIT_RMS, None):
        lin = CenteredReadout(16, C.M, logit_rms=pin)
        lin.zero_grad()
        y = lin(h)
        y.pow(2).sum().backward()
        g[pin] = float(lin.weight.grad.abs().max())
        if pin is not None:               # the objective is a constant under the pin
            assert torch.allclose(y.pow(2).sum(), torch.tensor(32.0 * C.M * pin ** 2),
                                  rtol=1e-4)
    assert g[LOGIT_RMS] < 1e-4 * g[None], g


def test_the_align_objective_pays_to_shrink_free_logits_and_pays_nothing_pinned():
    """R0-A's failure, reduced to one scalar and no network.

    `L_act`'s regression term is `||c_a - sg(c_Delta)||^2` between two hard top-4
    renormalised simplex points. With disjoint supports it is
    `sum(c_a^2) + sum(c_Delta^2)`, so the only thing the head controls is
    `sum(c_a^2)` -- monotonically decreasing as the logits shrink towards flat.
    The straight-through head supplies the asymmetry that gets there:
    `d hard_m/d l_m` carries a `1/Z ~ M/TOPK = 32` for an in-support atom and
    nothing for an out-of-support one, so "push down whoever is on top" beats
    "pull up what the target wants" and the fixed point is every logit equal.
    R0-A duly went from a spread of 0.929 to 0.0195 and ended on the flattest
    point of the simplex.

    Hand the head one knob -- a global gain on its logits -- and read off
    `dL/dgain`. Free, it is positive and large: shrinking pays. Pinned, the loss
    does not depend on the gain at all, so it is zero to round-off, and no amount
    of shrinking the readout or rotating it away from `h` buys anything.
    """
    torch.manual_seed(0)
    l0 = torch.randn(256, C.M)                       # rms 1: what the pin produces
    tl = torch.randn(256, C.M) * 1.5
    tl.scatter_(-1, l0.topk(C.TOPK, -1).indices, -30.0)   # target avoids the head's atoms
    target = topk_simplex_st(tl, C.TOPK)
    assert (topk_simplex_st(l0, C.TOPK) * target).sum(-1).max() == 0.0, "not disjoint"

    grads, losses = {}, {}
    for name in ("free", "pinned"):
        gain = torch.nn.Parameter(torch.ones(()))
        lg = l0 * gain
        lg = lg - lg.mean(-1, keepdim=True)
        if name == "pinned":
            lg = lg * (LOGIT_RMS / lg.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6))
        loss = (topk_simplex_st(lg, C.TOPK) - target).pow(2).sum(-1).mean()
        loss.backward()
        grads[name], losses[name] = float(gain.grad), float(loss)

    assert grads["free"] > 1e-3, grads                       # shrinking pays, a lot
    assert abs(grads["pinned"]) < 1e-6 * grads["free"], grads  # and pays nothing here
    # both start above the 0.500 disjoint-support floor and can only reach it by
    # flattening, which is the move the pin refuses to reward
    assert losses["free"] > 0.5 and losses["pinned"] > 0.5, losses


def test_action_rms_table_is_registered_and_well_formed():
    # `libero_franka` is registered by contracts itself; every other row's body
    # is registered by its adapter's import side effect, and whether some other
    # test file already triggered that is a test-order accident. Import it here
    # so this assertion means what it says.
    importlib.import_module("loom.data.adapters.robotwin")
    for name, rms in ACTION_RMS.items():
        assert name in C.EMBODIMENTS, f"{name} is not a registered embodiment"
        assert len(rms) == C.EMBODIMENTS[name].dof
        assert all(v > 0 for v in rms)


def test_action_rms_divides_before_step_in_and_is_not_persistent():
    q = QAction([BODY_A], **SMALL_QA)
    b = q.body(BODY_A)
    assert torch.equal(b.action_rms, torch.ones(DOF_A))      # unlisted body: unchanged
    assert not any("action_rms" in k for k in b.state_dict()), \
        "action_rms is a measured constant, not checkpoint state"

    rms = torch.tensor(ACTION_RMS["libero_franka"])
    scaled = QAction([BODY_A], action_rms=tuple(rms.tolist()), **SMALL_QA).body(BODY_A)
    scaled.load_state_dict(b.state_dict())
    a = torch.randn(4, C.H_OP, DOF_A)
    assert torch.allclose(scaled.encode_action(a), b.encode_action(a / rms), atol=1e-5)


def test_action_rms_brings_the_rotation_dofs_into_view():
    """LIBERO's per-dof rms spans 0.027 to 1.0, and `step_in` is one Linear with a
    shared init, so each dof enters in proportion to its variance: the binary
    gripper is 82% of it and the three rotation dofs are 0.44%. Dividing by the
    measured rms is what makes the head see a 7-dof action instead of one bit."""
    rms = torch.tensor(ACTION_RMS["libero_franka"])
    torch.manual_seed(0)
    a = torch.randn(512, C.H_OP, DOF_A) * rms                 # LIBERO-like scales

    def var_share(body):
        x = body.step_in(a / body.action_rms)                 # (N, H_OP, d_act)
        per = torch.stack([
            (body.step_in.weight[:, d].pow(2).sum() * (a[..., d] / body.action_rms[d]).var())
            for d in range(DOF_A)])
        return per / per.sum(), x

    plain = QAction([BODY_A], **SMALL_QA).body(BODY_A)        # action_rms == ones
    scaled = QAction([BODY_A], action_rms=tuple(rms.tolist()), **SMALL_QA).body(BODY_A)
    scaled.load_state_dict(plain.state_dict())

    share_plain, _ = var_share(plain)
    share_scaled, _ = var_share(scaled)
    assert share_plain[6] > 0.75, share_plain                 # gripper swamps everything
    assert share_plain[3:6].sum() < 0.02, share_plain         # rotation is invisible
    assert share_scaled.max() < 0.25, share_scaled            # no dof dominates
    assert share_scaled[3:6].sum() > 0.30, share_scaled       # rotation is now real


# ═══════════════════════════════════════════════════════════════════════════
#  THE SEGMENT-ANCHORED DELTA BRANCH   (ABSOLUTE-semantics bodies only)
#
#  `q_a` must emit a DIFFERENT c per horizon h. On a body whose actions are
#  ABSOLUTE servo targets in a slow trajectory, `a_seg` is nearly constant
#  across a window, so the only thing left that varies with h is phase and `c`
#  becomes a clock. The delta branch puts the within-window displacement back
#  in. LIBERO must be untouched, byte for byte: `r0a_flip` is live and its
#  remaining links re-import this file from disk.
# ═══════════════════════════════════════════════════════════════════════════

#: q_a's state_dict on the shipping LIBERO body, before the delta branch existed.
#: Both new buffers are `persistent=False`, so a body WITH the branch adds
#: exactly the two `delta_in` parameters and nothing else.
_LIBERO_QA_KEYS = 27


def _robotwin_dof() -> int:
    """Import the adapter for its registration side effects and return its dof.

    Deliberately NOT via `contracts.EMBODIMENTS`: conftest's autouse fixture
    restores that dict after every test while the adapter stays in
    `sys.modules`, so the second test in a session to ask would find the body
    missing. `canonical._ACTION_SEMANTICS` is not restored, which is why the
    delta branch keys off semantics and this helper reads dof off the module.
    """
    mod = importlib.import_module("loom.data.adapters.robotwin")
    return mod.DOF


def test_absolute_dims_excludes_libero_and_every_unmeasured_body():
    """The DELTA_RMS row is the opt-in, exactly as for ACTION_RMS."""
    assert "libero_franka" not in DELTA_RMS          # gate 1: no measured row
    # gate 2: even if the row existed, LIBERO has ZERO absolute dims
    assert ABSOLUTE not in action_semantics("libero_franka")
    assert absolute_dims("libero_franka", 7) == ()
    assert absolute_dims(None, 7) == ()
    assert absolute_dims("teamc_toy7", DOF_A) == ()  # synthetic, no semantics
    assert absolute_dims("teamc_toy14", DOF_B) == ()


def test_libero_body_has_no_delta_branch_and_todays_state_dict():
    b = QActionBody(7, embodiment="libero_franka")
    assert b.delta_dims == ()
    assert b.delta_in is None
    assert len(b.state_dict()) == _LIBERO_QA_KEYS
    assert not any("delta" in k for k in b.state_dict())


def test_a_body_with_no_registered_semantics_gets_no_branch():
    """BODY_A / BODY_B are in `contracts.EMBODIMENTS` but have no action
    semantics. They must fall out on the DELTA_RMS gate, before
    `action_semantics` is ever consulted -- not raise."""
    for body, dof in ((BODY_A, DOF_A), (BODY_B, DOF_B)):
        b = QAction([body], **SMALL_QA).body(body)
        assert b.delta_in is None and b.delta_dims == ()
        assert not any("delta" in k for k in b.state_dict())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_libero_encode_action_is_byte_identical(dtype):
    """The executed statement is unchanged character for character; the branch is
    guarded. Not `allclose` -- `torch.equal`, in the training dtype."""
    b = QActionBody(7, embodiment="libero_franka").to(dtype).eval()
    a = torch.randn(16, C.H_OP, 7, dtype=dtype)
    with torch.no_grad():
        x = b.step_in(a / b.action_rms.to(a)) + b.step_emb.to(dtype)
        expected = b.act_out(b.act_norm(x.flatten(-2)))
        assert torch.equal(b.encode_action(a), expected)


def test_robotwin_body_gets_every_dim_and_exactly_two_new_keys():
    dof = _robotwin_dof()
    b = QActionBody(dof, embodiment="robotwin_aloha")
    assert b.delta_dims == tuple(range(dof))          # all 14 are ABSOLUTE
    assert isinstance(b.delta_in, torch.nn.Linear)
    new_keys = set(b.state_dict()) - set(QActionBody(dof).state_dict())
    assert new_keys == {"delta_in.weight", "delta_in.bias"}
    # the measured constants must NOT ride in a checkpoint
    assert not any("delta_rms" in k or "delta_idx" in k for k in b.state_dict())
    d_act = b.step_in.out_features
    assert (sum(p.numel() for p in b.parameters())
            - sum(p.numel() for p in QActionBody(dof).parameters())
            == dof * d_act + d_act)


def test_delta_branch_is_invariant_to_a_constant_offset():
    """`a_seg + k` moves the ABSOLUTE branch's input and leaves the delta
    branch's alone -- that is what makes it a within-window signal rather than
    another copy of the pose.

    Invariance to `+k` is exact in real arithmetic but NOT bitwise in floating
    point: `(a+k) - (a+k)[0]` rounds differently from `a - a[0]`. Measured
    below at ~1e-6 relative against a signal of order 10, i.e. six orders down
    and irrelevant beside bf16's own 3e-3. The claim that IS bit-exact is the
    one the design rests on -- the SAME physical segment at a different horizon
    index encodes identically, because it is literally the same input bytes,
    with no offset added anywhere (measured on real windows in
    logs/dual_gate/encgate.py, max |diff| exactly 0).
    """
    dof = _robotwin_dof()
    b = QActionBody(dof, embodiment="robotwin_aloha").eval()
    a = torch.randn(8, C.H_OP, dof)
    k = torch.randn(8, 1, dof)                        # per-window constant offset

    def branches(x):
        g = x.index_select(-1, b.delta_idx)
        return (x / b.action_rms.to(x), (g - g[..., :1, :]) / b.delta_rms.to(g))

    abs0, d0 = branches(a)
    abs1, d1 = branches(a + k)
    assert torch.allclose(d0, d1, rtol=1e-4, atol=1e-4)   # delta branch: unmoved
    rel = (d0 - d1).abs().max() / d0.abs().max()
    assert rel < 1e-5, rel
    # and the absolute branch really does see the offset -- otherwise the test
    # would pass on an encoding that ignores its input
    assert not torch.allclose(abs0, abs1)
    assert (abs0 - abs1).abs().max() > 1e-3


def test_delta_branch_actually_changes_the_encoding_and_the_gain_scales_it():
    dof = _robotwin_dof()
    torch.manual_seed(0)
    on = QActionBody(dof, embodiment="robotwin_aloha").eval()
    torch.manual_seed(0)
    off = QActionBody(dof, embodiment="robotwin_aloha", delta_dims=[]).eval()
    assert off.delta_in is None
    off.load_state_dict(off.state_dict())
    a = torch.randn(8, C.H_OP, dof)
    with torch.no_grad():
        assert not torch.allclose(on.encode_action(a), off.encode_action(a))
        # `delta_gain=0` reproduces the kill switch exactly
        torch.manual_seed(0)
        zero = QActionBody(dof, embodiment="robotwin_aloha", delta_gain=0.0).eval()
        assert torch.allclose(zero.encode_action(a), off.encode_action(a), atol=1e-6)


def test_delta_branch_is_drawn_last_so_it_cannot_perturb_the_shared_init():
    """The kill switch must not reshuffle any other weight -- otherwise an
    ablation is not an ablation, and neither is a resume."""
    dof = _robotwin_dof()
    torch.manual_seed(0)
    on = QActionBody(dof, embodiment="robotwin_aloha")
    torch.manual_seed(0)
    off = QActionBody(dof, embodiment="robotwin_aloha", delta_dims=[])
    for k, v in off.state_dict().items():
        assert torch.equal(v, on.state_dict()[k]), k


def test_delta_rms_table_is_registered_and_well_formed():
    _robotwin_dof()                       # import for the registration side effect
    for name, rms in DELTA_RMS.items():
        kinds = action_semantics(name)    # raises if the body never declared any
        assert len(rms) == len(kinds), name
        assert all(v > 0 for v in rms), name
        # a measured row only makes sense where some dim is an absolute target
        assert ABSOLUTE in kinds, name
        # the two per-dof tables must describe the same body
        assert name in ACTION_RMS and len(ACTION_RMS[name]) == len(kinds), name


def test_q_action_module_imports_in_a_fresh_interpreter():
    """The one new import edge is `loom.heads.q_action -> loom.data.canonical`.
    Import it FIRST, with nothing else loaded, so a cycle would show up here."""
    import subprocess
    r = subprocess.run([sys.executable, "-c", "import loom.heads.q_action"],
                       capture_output=True, text=True, cwd=str(_REPO_ROOT))
    assert r.returncode == 0, r.stderr


# ═══════════════════════════════════════════════════════════════════════════
#  D_e — CONDITIONAL FLOW MATCHING   (done-when 4, 6, 10, 11)
# ═══════════════════════════════════════════════════════════════════════════

def test_decoder_satisfies_the_protocol():
    assert isinstance(Decoder([BODY_A], **SMALL_DEC), C.Decoder)


@pytest.mark.parametrize("body,dof", [(BODY_A, DOF_A), (BODY_B, DOF_B)])
def test_decoder_emits_one_operator_never_h_plan(body, dof):
    """done-when 6: shape[-2] == H_OP, explicitly != H_PLAN, validator passes."""
    dec = Decoder([BODY_A, BODY_B], **SMALL_DEC)
    a = dec(proprio(3, dof), S.sparse_simplex(3), embodiment=body)
    assert a.shape == (3, C.H_OP, dof)
    assert a.shape[-2] == C.H_OP
    assert a.shape[-2] != C.H_PLAN
    C.assert_action_segment(a, body)


def test_decoder_loss_is_a_finite_scalar():
    dec = Decoder([BODY_A], **SMALL_DEC)
    loss = dec.loss(proprio(4), S.sparse_simplex(4), torch.randn(4, C.H_OP, DOF_A))
    assert loss.ndim == 0 and torch.isfinite(loss) and loss > 0


def test_decoder_loss_rejects_h_plan_segments():
    dec = Decoder([BODY_A], **SMALL_DEC)
    with pytest.raises(ValueError):
        dec.loss(proprio(2), S.sparse_simplex(2), torch.randn(2, C.H_PLAN, DOF_A))


def test_cfm_target_is_the_conditional_velocity():
    """The regression target must be x_1 - x_0 at x_t = (1-t) x_0 + t x_1.

    Checked by construction: with the field pinned to a constant v, the loss is
    exactly mean((v - (a - x0))^2) for the given (t, x0).
    """
    dec = DecoderBody(BODY_A, **SMALL_DEC)
    _pin_field(dec, 0.25)                      # make v_theta a known constant
    p, c = proprio(4), S.sparse_simplex(4)
    a = torch.randn(4, C.H_OP, DOF_A)
    x0 = torch.randn_like(a)
    t = torch.rand(4)
    got = dec.loss(p, c, a, t=t, noise=x0)
    want = (0.25 - (a - x0)).pow(2).flatten(1).mean(-1).mean()
    assert torch.allclose(got, want, atol=1e-5)


def test_cfm_loss_is_independent_of_t_for_a_constant_field():
    """Sanity on the path: with a constant field the target does not depend on
    t, which is the defining property of the straight (rectified) path."""
    dec = DecoderBody(BODY_A, **SMALL_DEC)
    _pin_field(dec, 0.1)
    p, c = proprio(4), S.sparse_simplex(4)
    a, x0 = torch.randn(4, C.H_OP, DOF_A), torch.randn(4, C.H_OP, DOF_A)
    lo = dec.loss(p, c, a, t=torch.zeros(4), noise=x0)
    hi = dec.loss(p, c, a, t=torch.ones(4), noise=x0)
    assert torch.allclose(lo, hi, atol=1e-6)


def test_euler_integration_uses_the_requested_number_of_steps():
    """x_{i+1} = x_i + (1/n) v(x_i, i/n); with v pinned to a constant the
    sampler must land exactly on x_0 + v."""
    dec = DecoderBody(BODY_A, clamp=False, **{**SMALL_DEC, "n_steps": 7})
    _pin_field(dec, 0.3)
    x0 = torch.randn(2, C.H_OP, DOF_A)
    out = dec(proprio(2), S.sparse_simplex(2), noise=x0)
    assert torch.allclose(out, x0 + 0.3, atol=1e-5)
    assert dec.n_steps == 7


def test_decoder_default_is_ten_euler_steps():
    assert DecoderBody(BODY_A, **{k: v for k, v in SMALL_DEC.items()
                                  if k != "n_steps"}).n_steps == 10


def test_decoder_output_respects_the_action_bounds():
    """Clamped in forward (eval-safe); the CFM target is NOT clamped."""
    dec = Decoder([BODY_B], **SMALL_DEC)
    spec = C.EMBODIMENTS[BODY_B]
    a = dec(proprio(8, DOF_B), S.sparse_simplex(8), embodiment=BODY_B)
    assert (a >= spec.action_low[0] - 1e-5).all() and (a <= spec.action_high[0] + 1e-5).all()

    raw = dec(proprio(8, DOF_B), S.sparse_simplex(8), embodiment=BODY_B, clamp=False,
              noise=torch.full((8, C.H_OP, DOF_B), 9.0))
    assert raw.abs().max() > spec.action_high[0]             # nothing clamps in the field


def test_decoder_gradient_flows_from_loss_to_proprio_and_coefficients():
    dec = Decoder([BODY_A], **SMALL_DEC)
    p = proprio(3).requires_grad_(True)
    c = S.sparse_simplex(3).requires_grad_(True)
    dec.loss(p, c, torch.randn(3, C.H_OP, DOF_A)).backward()
    assert p.grad is not None and p.grad.abs().sum() > 0
    assert c.grad is not None and c.grad.abs().sum() > 0


def test_decoder_refuses_the_belief_where_proprio_belongs():
    """The whole point of the contract change: `D_e` does not take `(B,K,D)`.

    A silently-broadcast belief would give a `(B, K, d)` condition and a decoder
    that trains and scores near zero, which is the failure mode this repo keeps
    paying for. Fail loudly instead.
    """
    dec = Decoder([BODY_A], **SMALL_DEC)
    with pytest.raises(ValueError):
        dec(belief(2), S.sparse_simplex(2))
    with pytest.raises(ValueError):
        dec.loss(belief(2), S.sparse_simplex(2), torch.randn(2, C.H_OP, DOF_A))


def test_decoder_output_depends_on_the_coefficient():
    """With `z` gone, `c` is the only channel carrying task information in.

    Not a strong claim about a trained model -- an untrained field is small --
    but it pins that the coefficient reaches the output at all, which is what
    `L_act` needs in order to be a training signal for the operator.
    """
    torch.manual_seed(3)
    dec = DecoderBody(BODY_A, clamp=False, **SMALL_DEC)
    p = proprio(6)
    x0 = torch.randn(6, C.H_OP, DOF_A)
    c1, c2 = S.sparse_simplex(6), S.sparse_simplex(6)
    a1 = dec(p, c1, noise=x0)
    a2 = dec(p, c2, noise=x0)
    assert (a1 - a2).abs().max() > 0, "c does not reach the action at all"


def test_decoder_bodies_are_separate_and_dispatch_rejects_junk():
    dec = Decoder([BODY_A, BODY_B], **SMALL_DEC)
    assert dec.body(BODY_A).dof == DOF_A and dec.body(BODY_B).dof == DOF_B
    with pytest.raises(KeyError):
        dec.body("not_a_body")
    with pytest.raises(ValueError):
        dec(proprio(2), S.sparse_simplex(2))                 # ambiguous, 2 bodies


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_decoder_dtype_is_preserved(dtype):
    dec = Decoder([BODY_A], **SMALL_DEC).to(dtype)
    p, c = proprio(2, dtype=dtype), S.sparse_simplex(2, dtype=dtype)
    a = dec(p, c)
    assert a.dtype is dtype and not a.is_complex()
    loss = dec.loss(p, c, torch.randn(2, C.H_OP, DOF_A, dtype=dtype))
    assert loss.dtype is dtype and torch.isfinite(loss)


@pytest.mark.slow
def test_decoder_parameter_budget_per_body():
    """done-when 10: D_e ~ 20 M per body, +/- 40%.

    ~18 M since the belief pool left: `AttnPool` + `z_proj` were 2.7 M of the
    20.9 M, replaced by a `Linear(dof, d)` of 4 k. Still inside the budget, and
    deliberately NOT padded back out -- the head's capacity was never the
    problem, its input was.
    """
    n = n_params(Decoder([BODY_A]).body(BODY_A))
    assert 12e6 <= n <= 28e6, f"decoder is {n / 1e6:.1f} M/body, budget is 20 M"


@pytest.mark.slow
def test_full_size_heads_run_end_to_end_in_bf16():
    """done-when 11, at the real widths: no promotion, no complex, all finite.

    B=1 and two Euler steps on purpose — this is a dtype test, and bf16 GEMM on
    CPU is roughly an order of magnitude slower than f32.
    """
    qd = QDelta().to(torch.bfloat16)
    qa = QAction([BODY_A]).to(torch.bfloat16)
    dec = Decoder([BODY_A], n_steps=2).to(torch.bfloat16)
    z, zn = belief(1, torch.bfloat16), belief(1, torch.bfloat16)

    c = qd(z, zn)
    C.assert_simplex(c)
    a = dec(proprio(1, dtype=torch.bfloat16), c, embodiment=BODY_A)
    C.assert_action_segment(a, BODY_A)
    c2 = qa(a, z, embodiment=BODY_A)
    C.assert_simplex(c2)
    for t in (c, a, c2):
        assert t.dtype is torch.bfloat16 and not t.is_complex()
        assert torch.isfinite(t.float()).all()
