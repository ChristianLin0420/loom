"""LOOM — Team C gate, heads half.

`q_Delta`, `q_a^e`, `D_e`. Imports nothing from `loom.model` and nothing from
`loom.heads.proposal`: everything here runs against `contracts` + `stubs`.

Covers PLAN 4.C done-when items 3, 4, 5, 6, 10, 11.
"""

from __future__ import annotations

import pytest
import torch

import contracts as C
import stubs as S
from loom.heads.decoder import Decoder, DecoderBody
from loom.heads.q_action import ACTION_RMS, LOGIT_RMS, QAction
from loom.heads.q_delta import AttnPool, CenteredReadout, QDelta, topk_simplex_st

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
#  D_e — CONDITIONAL FLOW MATCHING   (done-when 4, 6, 10, 11)
# ═══════════════════════════════════════════════════════════════════════════

def test_decoder_satisfies_the_protocol():
    assert isinstance(Decoder([BODY_A], **SMALL_DEC), C.Decoder)


@pytest.mark.parametrize("body,dof", [(BODY_A, DOF_A), (BODY_B, DOF_B)])
def test_decoder_emits_one_operator_never_h_plan(body, dof):
    """done-when 6: shape[-2] == H_OP, explicitly != H_PLAN, validator passes."""
    dec = Decoder([BODY_A, BODY_B], **SMALL_DEC)
    a = dec(belief(3), S.sparse_simplex(3), embodiment=body)
    assert a.shape == (3, C.H_OP, dof)
    assert a.shape[-2] == C.H_OP
    assert a.shape[-2] != C.H_PLAN
    C.assert_action_segment(a, body)


def test_decoder_loss_is_a_finite_scalar():
    dec = Decoder([BODY_A], **SMALL_DEC)
    loss = dec.loss(belief(4), S.sparse_simplex(4), torch.randn(4, C.H_OP, DOF_A))
    assert loss.ndim == 0 and torch.isfinite(loss) and loss > 0


def test_decoder_loss_rejects_h_plan_segments():
    dec = Decoder([BODY_A], **SMALL_DEC)
    with pytest.raises(ValueError):
        dec.loss(belief(2), S.sparse_simplex(2), torch.randn(2, C.H_PLAN, DOF_A))


def test_cfm_target_is_the_conditional_velocity():
    """The regression target must be x_1 - x_0 at x_t = (1-t) x_0 + t x_1.

    Checked by construction: with the field pinned to a constant v, the loss is
    exactly mean((v - (a - x0))^2) for the given (t, x0).
    """
    dec = DecoderBody(BODY_A, **SMALL_DEC)
    _pin_field(dec, 0.25)                      # make v_theta a known constant
    z, c = belief(4), S.sparse_simplex(4)
    a = torch.randn(4, C.H_OP, DOF_A)
    x0 = torch.randn_like(a)
    t = torch.rand(4)
    got = dec.loss(z, c, a, t=t, noise=x0)
    want = (0.25 - (a - x0)).pow(2).flatten(1).mean(-1).mean()
    assert torch.allclose(got, want, atol=1e-5)


def test_cfm_loss_is_independent_of_t_for_a_constant_field():
    """Sanity on the path: with a constant field the target does not depend on
    t, which is the defining property of the straight (rectified) path."""
    dec = DecoderBody(BODY_A, **SMALL_DEC)
    _pin_field(dec, 0.1)
    z, c = belief(4), S.sparse_simplex(4)
    a, x0 = torch.randn(4, C.H_OP, DOF_A), torch.randn(4, C.H_OP, DOF_A)
    lo = dec.loss(z, c, a, t=torch.zeros(4), noise=x0)
    hi = dec.loss(z, c, a, t=torch.ones(4), noise=x0)
    assert torch.allclose(lo, hi, atol=1e-6)


def test_euler_integration_uses_the_requested_number_of_steps():
    """x_{i+1} = x_i + (1/n) v(x_i, i/n); with v pinned to a constant the
    sampler must land exactly on x_0 + v."""
    dec = DecoderBody(BODY_A, clamp=False, **{**SMALL_DEC, "n_steps": 7})
    _pin_field(dec, 0.3)
    x0 = torch.randn(2, C.H_OP, DOF_A)
    out = dec(belief(2), S.sparse_simplex(2), noise=x0)
    assert torch.allclose(out, x0 + 0.3, atol=1e-5)
    assert dec.n_steps == 7


def test_decoder_default_is_ten_euler_steps():
    assert DecoderBody(BODY_A, **{k: v for k, v in SMALL_DEC.items()
                                  if k != "n_steps"}).n_steps == 10


def test_decoder_output_respects_the_action_bounds():
    """Clamped in forward (eval-safe); the CFM target is NOT clamped."""
    dec = Decoder([BODY_B], **SMALL_DEC)
    spec = C.EMBODIMENTS[BODY_B]
    a = dec(belief(8), S.sparse_simplex(8), embodiment=BODY_B)
    assert (a >= spec.action_low[0] - 1e-5).all() and (a <= spec.action_high[0] + 1e-5).all()

    raw = dec(belief(8), S.sparse_simplex(8), embodiment=BODY_B, clamp=False,
              noise=torch.full((8, C.H_OP, DOF_B), 9.0))
    assert raw.abs().max() > spec.action_high[0]             # nothing clamps in the field


def test_decoder_gradient_flows_from_loss_to_belief_and_coefficients():
    dec = Decoder([BODY_A], **SMALL_DEC)
    z = belief(3).requires_grad_(True)
    c = S.sparse_simplex(3).requires_grad_(True)
    dec.loss(z, c, torch.randn(3, C.H_OP, DOF_A)).backward()
    assert z.grad is not None and z.grad.abs().sum() > 0
    assert c.grad is not None and c.grad.abs().sum() > 0


def test_decoder_bodies_are_separate_and_dispatch_rejects_junk():
    dec = Decoder([BODY_A, BODY_B], **SMALL_DEC)
    assert dec.body(BODY_A).dof == DOF_A and dec.body(BODY_B).dof == DOF_B
    with pytest.raises(KeyError):
        dec.body("not_a_body")
    with pytest.raises(ValueError):
        dec(belief(2), S.sparse_simplex(2))                  # ambiguous, 2 bodies


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_decoder_dtype_is_preserved(dtype):
    dec = Decoder([BODY_A], **SMALL_DEC).to(dtype)
    z, c = belief(2, dtype), S.sparse_simplex(2, dtype=dtype)
    a = dec(z, c)
    assert a.dtype is dtype and not a.is_complex()
    loss = dec.loss(z, c, torch.randn(2, C.H_OP, DOF_A, dtype=dtype))
    assert loss.dtype is dtype and torch.isfinite(loss)


@pytest.mark.slow
def test_decoder_parameter_budget_per_body():
    """done-when 10: D_e ~ 20 M per body, +/- 40%."""
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
    a = dec(z, c, embodiment=BODY_A)
    C.assert_action_segment(a, BODY_A)
    c2 = qa(a, z, embodiment=BODY_A)
    C.assert_simplex(c2)
    for t in (c, a, c2):
        assert t.dtype is torch.bfloat16 and not t.is_complex()
        assert torch.isfinite(t.float()).all()
