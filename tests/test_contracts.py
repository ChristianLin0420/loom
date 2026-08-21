"""
LOOM — Phase 0 gate.

`pytest tests/test_contracts.py` must be green before contracts.py freezes and
the six teams fan out. Nothing else starts until this passes.
"""

from __future__ import annotations

import math

import pytest
import torch

import contracts as C
import stubs as S


# ═══════════════════════════════════════════════════════════════════════════
#  STRUCTURAL INVARIANTS
# ═══════════════════════════════════════════════════════════════════════════

def test_temporal_consistency():
    assert C.H_OP * C.DEPTH == C.H_PLAN
    assert C.N_STATES == C.DEPTH + 1
    assert len(C.DYN_WEIGHTS) == C.DEPTH


def test_recurrent_burn_in_is_the_only_optional_window_field():
    assert C.TransitionWindow.__required_keys__ == {
        "feats", "actions", "lang", "embodiment", "src_fps",
    }
    assert C.TransitionWindow.__optional_keys__ == {"burn_in_feats"}


def test_slot_width_even():
    """The operator acts on adjacent pairs as 2x2 blocks."""
    assert C.D % 2 == 0


def test_bounds_are_contractive():
    assert 0.0 < C.RHO < 1.0
    assert C.B_MAX > 0.0


def test_topk_valid():
    assert 1 <= C.TOPK <= C.M


def test_balance_coef_is_the_owner_authorised_value():
    """The second owner-authorised contract change, pinned.

    3e-3 with the old KL-of-batch-mean form left an unselected operator at
    0.0006 (ctrl) / 0.0001 (zinit) of a selected one's per-entry gradient at
    q_Delta's logits. `configs/*.yaml` must agree -- `losses.balance.weight` is
    what the loop actually multiplies by, and
    `tests/test_train.py::test_r0a_config_matches_the_plan` pins the pair.
    """
    assert C.BALANCE_COEF == 1e-2


def test_operator_duration_is_reasonable():
    """One operator should be a plausible linearization interval."""
    ms = 1000.0 * C.H_OP / C.FPS_CANONICAL
    assert 150.0 < ms < 400.0, f"operator spans {ms:.0f} ms"


# ═══════════════════════════════════════════════════════════════════════════
#  EMBODIMENT REGISTRY
# ═══════════════════════════════════════════════════════════════════════════

def test_libero_registered():
    spec = C.EMBODIMENTS["libero_franka"]
    assert spec.dof == 7
    assert spec.n_views == 2
    assert spec.env_fps == 20.0


def test_registration_is_idempotent():
    C.register_embodiment(C.EMBODIMENTS["libero_franka"])


def test_conflicting_registration_rejected():
    bad = C.EmbodimentSpec("libero_franka", 9, 20.0, 2, (-1.0,) * 9, (1.0,) * 9)
    with pytest.raises(ValueError):
        C.register_embodiment(bad)


def test_action_bounds_must_match_dof():
    with pytest.raises(ValueError):
        C.EmbodimentSpec("bad", 7, 30.0, 2, (-1.0,) * 3, (1.0,) * 7)


def test_env_steps_per_segment_is_fractional_for_libero():
    """8 canonical steps at 30 Hz onto a 20 Hz env is 5.333, not an integer.

    A Policy must carry a fractional accumulator. Rounding each segment
    independently drifts over an episode.
    """
    n = C.env_steps_per_segment(20.0)
    assert math.isclose(n, 8 * 20 / 30)
    assert not float(n).is_integer()


def test_env_steps_identity_at_canonical_rate():
    assert C.env_steps_per_segment(C.FPS_CANONICAL) == C.H_OP


# ═══════════════════════════════════════════════════════════════════════════
#  DATA SHAPES
# ═══════════════════════════════════════════════════════════════════════════

def test_window_has_n_states():
    w = S.make_window(b=2)
    assert len(w["feats"]) == C.N_STATES


def test_window_actions_shape():
    w = S.make_window(b=2)
    dof = C.EMBODIMENTS[w["embodiment"]].dof
    assert w["actions"].shape == (2, C.DEPTH, C.H_OP, dof)


def test_action_free_window():
    w = S.make_window(b=2, action_free=True)
    assert w["actions"] is None
    assert len(w["feats"]) == C.N_STATES


def test_window_records_src_fps():
    """Eval needs this to invert canonical resampling."""
    assert S.make_window(b=2)["src_fps"] > 0


# ═══════════════════════════════════════════════════════════════════════════
#  STUB SHAPES
# ═══════════════════════════════════════════════════════════════════════════

def test_estimator_shape():
    z = S.StubEstimator()(S.make_obs_feats(b=3), None)
    C.assert_belief(z)
    assert z.shape == (3, C.K, C.D)


def test_estimator_accepts_recurrence():
    e, feats = S.StubEstimator(), S.make_obs_feats(b=3)
    z1 = e(feats, None)
    z2 = e(feats, z1)
    C.assert_belief(z2)


def test_bank_step_shape():
    bank, z = S.StubBank(), torch.randn(4, C.K, C.D)
    out = bank.step(S.sparse_simplex(4), z)
    assert out.shape == z.shape
    C.assert_belief(out)


def test_bank_step_broadcasts_over_candidates():
    bank = S.StubBank()
    z = torch.randn(2, 5, C.K, C.D)
    out = bank.step(S.sparse_simplex(2, 5), z)
    assert out.shape == z.shape


def test_rollout_shape():
    bank = S.StubBank()
    out = bank.rollout(S.sparse_simplex(2, 7, C.DEPTH), torch.randn(2, C.K, C.D))
    assert out.shape == (2, 7, C.K, C.D)


def test_rollout_matches_sequential_step():
    """rollout must be exactly DEPTH applications of step, bias included.

    This is the test that catches the affine-composition bug: composing
    (A2, b2) after (A1, b1) gives (A2@A1, A2@b1 + b2), so multiplying lambdas
    alone silently drops the accumulated bias.
    """
    bank = S.StubBank()
    z0 = torch.randn(2, C.K, C.D)
    c = S.sparse_simplex(2, 1, C.DEPTH)

    manual = z0.unsqueeze(1)
    for d in range(C.DEPTH):
        manual = bank.step(c[:, :, d], manual)

    assert torch.allclose(bank.rollout(c, z0), manual, atol=1e-5)


def test_decoder_emits_h_op_not_h_plan():
    """A Decoder emits ONE operator's worth of action. Never H_PLAN.

    First argument is `(B, dof_e)` PROPRIO, not the belief: `contracts.Decoder`
    is `forward(proprio, c)`. `stubs.StubDecoder` is frozen and predates that,
    but it only ever reads `shape[0]` / `device` / `dtype` off its first
    argument, so it satisfies the new signature unchanged.
    """
    dec = S.StubDecoder()
    a = dec(torch.randn(3, 7), S.sparse_simplex(3))
    assert a.shape == (3, C.H_OP, 7)
    assert a.shape[1] != C.H_PLAN
    C.assert_action_segment(a, "libero_franka")


def test_decoder_protocol_takes_proprio_and_not_the_belief():
    """The owner-authorised contract change, pinned.

    `D_e(z, c)` made `L_act` behaviour cloning: with the whole belief available
    the decoder needs nothing from `c`, and R0-A measured `act/decode` falling
    0.2489 -> 0.0559 while `c_a` held 2-3 distinct top-4 supports over 64 real
    training windows. Dropping `z` makes `c` the only channel carrying task
    information into the action.
    """
    import inspect

    for name in ("forward", "loss"):
        params = list(inspect.signature(getattr(C.Decoder, name)).parameters)
        assert params[1] == "proprio", (
            f"contracts.Decoder.{name} must take proprio first, got {params}"
        )
        assert "z" not in params, f"the belief is not an input to D_e: {params}"


def test_q_delta_returns_simplex():
    q = S.StubQDelta()
    C.assert_simplex(q(torch.randn(4, C.K, C.D), torch.randn(4, C.K, C.D)))


def test_q_action_returns_simplex():
    q = S.StubQAction()
    C.assert_simplex(q(torch.randn(4, C.H_OP, 7), torch.randn(4, C.K, C.D)))


def test_proposal_sample_shape_and_simplex():
    p = S.StubProposal()
    c = p.sample(torch.randn(2, C.K, C.D), torch.randn(2, 16, 1152), n=11)
    assert c.shape == (2, 11, C.M)
    C.assert_simplex(c)


def test_proposal_log_prob_is_negative_and_finite():
    p = S.StubProposal()
    z, lang = torch.randn(3, C.K, C.D), torch.randn(3, 16, 1152)
    lp = p.log_prob(z, lang, S.sparse_simplex(3))
    assert lp.shape == (3,)
    assert torch.isfinite(lp).all()
    assert (lp <= 0).all()


def test_potential_shape():
    phi = S.StubPotential()
    assert phi(torch.randn(2, 9, C.K, C.D), torch.randn(2, 16, 1152)).shape == (2, 9)


# ═══════════════════════════════════════════════════════════════════════════
#  INVARIANTS THE REAL MODULES MUST ALSO SATISFY
# ═══════════════════════════════════════════════════════════════════════════

def test_stub_bank_is_contractive():
    C.assert_contractive(S.StubBank(), n=2000)


def test_stub_bank_bias_is_bounded():
    C.assert_bias_bounded(S.StubBank(), n=2000)


def test_rollout_does_not_explode():
    """DEPTH applications must stay bounded by rho^T ||z|| + (1-rho^T)/(1-rho) B."""
    bank = S.StubBank()
    z0 = torch.randn(8, C.K, C.D)
    leaves = bank.rollout(S.sparse_simplex(8, 1, C.DEPTH), z0)

    bound = (C.RHO ** C.DEPTH) * z0.flatten(1).norm(dim=1) \
        + (1 - C.RHO ** C.DEPTH) / (1 - C.RHO) * C.B_MAX
    assert (leaves.squeeze(1).flatten(1).norm(dim=1) <= bound + 1e-3).all()


def test_simplex_validator_rejects_dense():
    with pytest.raises(AssertionError):
        C.assert_simplex(torch.full((2, C.M), 1.0 / C.M))


def test_simplex_validator_rejects_unnormalized():
    c = S.sparse_simplex(2) * 2.0
    with pytest.raises(AssertionError):
        C.assert_simplex(c)


def test_belief_validator_rejects_complex():
    """z is real throughout. Complex is only a view of adjacent real pairs."""
    with pytest.raises(AssertionError):
        C.assert_belief(torch.randn(2, C.K, C.D // 2, dtype=torch.complex64))


def test_action_segment_validator_rejects_h_plan():
    with pytest.raises(AssertionError):
        C.assert_action_segment(torch.randn(2, C.H_PLAN, 7), "libero_franka")


# ═══════════════════════════════════════════════════════════════════════════
#  BF16
# ═══════════════════════════════════════════════════════════════════════════

def test_bank_runs_in_bf16_without_promotion():
    """A100 has no FP8 and no complex-bf16. The 2x2 algebra must be real bf16."""
    bank = S.StubBank().to(torch.bfloat16)
    z = torch.randn(2, C.K, C.D, dtype=torch.bfloat16)
    out = bank.step(S.sparse_simplex(2, dtype=torch.bfloat16), z)
    assert out.dtype == torch.bfloat16
    assert not out.is_complex()


def test_view_as_complex_is_not_available_in_bf16():
    """Documents why bank.py must use explicit real 2x2 algebra."""
    z = torch.randn(2, C.K, C.D // 2, 2, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError):
        torch.view_as_complex(z)


# ═══════════════════════════════════════════════════════════════════════════
#  POLICY LOOP
# ═══════════════════════════════════════════════════════════════════════════

def test_policy_emits_single_actions():
    p = S.StubPolicy()
    p.reset()
    a = p.act({}, "pick up the bowl")
    assert a.shape == (7,)


def test_policy_replans_at_segment_boundaries():
    """Over 100 env steps at 20 Hz, expect ~100 / 5.333 replans."""
    p = S.StubPolicy()
    p.reset()
    for _ in range(100):
        p.act({}, "task")
    expected = 100 / C.env_steps_per_segment(20.0)
    assert abs(p.replans - expected) <= 2, f"{p.replans} replans, expected ~{expected:.1f}"


def test_policy_reset_clears_state():
    p = S.StubPolicy()
    for _ in range(20):
        p.act({}, "task")
    p.reset()
    assert p.replans == 0 and p._accum == 0.0
