"""LOOM — Team C gate, losses half.

`L_dyn`, `L_act`, `L_proposal`, `L_balance`, and the EMA target machinery.
Runs against `stubs.StubBank` / `stubs.StubProposal`; imports nothing from
`loom.model` and nothing from `loom.heads.proposal`.

Covers PLAN 4.C done-when items 1, 2, 4, 7, 8, 9.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

import contracts as C
import stubs as S
from loom.heads.decoder import Decoder
from loom.heads.q_action import QAction
from loom.heads.q_delta import QDelta, topk_simplex_st
from loom.losses.act import act_loss, q_action_regression_loss, zero_loss
from loom.losses.balance import balance_kl, balance_loss, operator_usage
from loom.losses.dyn import (
    NEGATIVE_MODES, DynLoss, EmaEstimator, dyn_loss, ema_update,
    ln_cosine_distance, sample_within_trajectory_negatives, sequential_rollout,
)
from loom.losses.proposal_bc import proposal_bc_loss

torch.manual_seed(0)

# Two synthetic bodies with different dof (done-when 4). Registration is
# idempotent for an identical spec, so this agrees with tests/test_heads.py
# without either file importing the other.
DOF_A, DOF_B = 7, 14
BODY_A = C.register_embodiment(C.EmbodimentSpec(
    name="teamc_toy7", dof=DOF_A, env_fps=20.0, n_views=2,
    action_low=(-1.0,) * DOF_A, action_high=(1.0,) * DOF_A,
)).name
BODY_B = C.register_embodiment(C.EmbodimentSpec(
    name="teamc_toy14", dof=DOF_B, env_fps=50.0, n_views=3,
    action_low=(-2.0,) * DOF_B, action_high=(2.0,) * DOF_B,
)).name

SMALL_QD = dict(hidden=64, n_queries=1, n_heads=4, d_kv=32)
SMALL_QA = dict(hidden=64, n_queries=1, n_heads=4, d_kv=32,
                d_act=16, d_act_out=32, n_hidden=2)
SMALL_DEC = dict(d=32, n_blocks=1, n_heads=2, n_queries=1, pool_heads=4,
                 d_kv=32, n_steps=3)


def belief(b: int = 3) -> torch.Tensor:
    return torch.randn(b, C.K, C.D)


def proprio(b: int = 3, dof: int = DOF_A) -> torch.Tensor:
    """`ObsFeats["proprio"]` — ONE timestep, `(B, dof_e)`.

    `L_act` runs the decoder on THIS and the coefficient, never on the belief:
    with the belief available the decoder is a behaviour-cloning head and needs
    nothing from `c`. See `loom/heads/decoder.py`.
    """
    return torch.randn(b, dof)


def frozen_bank() -> S.StubBank:
    """A real 2x2 rotation-decay bank whose bounds genuinely hold (stubs.py)."""
    return S.StubBank().requires_grad_(False)


def rollout_states(bank, z0: torch.Tensor, c_seq: torch.Tensor) -> torch.Tensor:
    """(B,H,K,D) ground-truth states from sequential application of c_seq."""
    with torch.no_grad():
        return torch.stack(sequential_rollout(bank, z0, c_seq), dim=1)


# ═══════════════════════════════════════════════════════════════════════════
#  THE DISTANCE
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("mode", ["per_slot", "flat"])
def test_ln_cosine_distance_is_zero_for_identical_beliefs(mode):
    z = belief(4)
    d = ln_cosine_distance(z, z.clone(), mode)
    assert d.shape == (4,)
    assert torch.allclose(d, torch.zeros_like(d), atol=1e-5)


def test_ln_cosine_distance_is_scale_and_shift_invariant_per_slot():
    """LayerNorm first, so a per-slot affine rescaling must not move the loss."""
    z = belief(4)
    d = ln_cosine_distance(z, 3.0 * z + 5.0, "per_slot")
    assert torch.allclose(d, torch.zeros_like(d), atol=1e-5)


def test_per_slot_is_harsher_than_flat_when_a_few_slots_dominate():
    """Why per_slot is the default: a flat cosine is dominated by the
    highest-norm slots, so getting 3 of 128 slots right can still score ~1."""
    a = torch.randn(1, C.K, C.D)
    b = torch.randn(1, C.K, C.D)
    b[:, :3] = a[:, :3]
    a[:, :3] *= 30.0                      # three high-energy, correct slots
    b[:, :3] *= 30.0
    assert ln_cosine_distance(a, b, "per_slot") > ln_cosine_distance(a, b, "flat")


def test_ln_cosine_rejects_unknown_mode():
    with pytest.raises(ValueError):
        ln_cosine_distance(belief(2), belief(2), "cosine_over_vibes")


# ═══════════════════════════════════════════════════════════════════════════
#  L_dyn — STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════

def test_dyn_matches_the_written_formula_with_contract_weights():
    """L_dyn = sum_h w_h (1 - cos(LN(z_hat_h), sg LN(z_bar_h))), w = DYN_WEIGHTS."""
    bank, z0 = frozen_bank(), belief(3)
    c_seq = S.sparse_simplex(3, C.DEPTH)
    tgt = torch.randn(3, C.DEPTH, C.K, C.D)

    out = dyn_loss(bank, z0, c_seq, tgt, negatives="none")

    z, want = z0, 0.0
    for h in range(C.DEPTH):
        z = bank.step(c_seq[:, h], z)
        want = want + C.DYN_WEIGHTS[h] * ln_cosine_distance(z, tgt[:, h]).mean()
    assert torch.allclose(out["loss"], want, atol=1e-5)
    assert torch.allclose(out["neg"], torch.zeros(()))


def test_dyn_rollout_is_sequential_and_keeps_the_bias():
    """The affine-composition trap: composing (A2,b2) after (A1,b1) is
    (A2A1, A2b1+b2). Multiplying lambdas alone drops the accumulated bias and
    L_dyn would happily go down anyway."""
    bank, z0 = frozen_bank(), belief(2)
    c_seq = S.sparse_simplex(2, C.DEPTH)
    states = sequential_rollout(bank, z0, c_seq)
    assert len(states) == C.DEPTH

    z = z0
    for h in range(C.DEPTH):
        z = bank.step(c_seq[:, h], z)
        assert torch.allclose(states[h], z, atol=1e-6)

    lam_only = z0
    for h in range(C.DEPTH):                              # the WRONG way
        a, b = bank.mix(c_seq[:, h])
        zr = lam_only.reshape(2, C.K, C.D // 2, 2)
        x, y = zr[..., 0], zr[..., 1]
        lam_only = torch.stack([a * x - b * y, b * x + a * y], -1).reshape(2, C.K, C.D)
    assert not torch.allclose(states[-1], lam_only, atol=1e-4)


def test_dyn_targets_are_stop_gradded():
    """Without sg() the cheapest solution is for the estimator to collapse."""
    bank, z0 = frozen_bank(), belief(2)
    tgt = torch.randn(2, C.DEPTH, C.K, C.D, requires_grad=True)
    c_seq = S.sparse_simplex(2, C.DEPTH).requires_grad_(True)
    dyn_loss(bank, z0, c_seq, tgt, negatives="none")["loss"].backward()
    assert tgt.grad is None
    assert c_seq.grad is not None and c_seq.grad.abs().sum() > 0


def test_dyn_horizons_follow_the_targets_not_the_constant():
    """Fewer horizons than DEPTH is legal (short windows); weights are the prefix."""
    bank, z0 = frozen_bank(), belief(2)
    out = dyn_loss(bank, z0, S.sparse_simplex(2, 2), torch.randn(2, 2, C.K, C.D),
                   negatives="none")
    assert out["per_h"].shape == (2,)
    with pytest.raises(ValueError):
        dyn_loss(bank, z0, S.sparse_simplex(2, C.DEPTH), torch.randn(2, 2, C.K, C.D))


def test_dyn_rejects_unknown_negative_mode_and_offers_no_in_batch_option():
    """Deliberate: in-batch negatives would make two bodies that produce the
    same world effect repel, which is the opposite of a shared operator bank."""
    assert NEGATIVE_MODES == ("none", "within_trajectory")
    with pytest.raises(ValueError):
        dyn_loss(frozen_bank(), belief(2), S.sparse_simplex(2, C.DEPTH),
                 torch.randn(2, C.DEPTH, C.K, C.D), negatives="in_batch")
    with pytest.raises(ValueError):
        DynLoss(negatives="in_batch")


# ═══════════════════════════════════════════════════════════════════════════
#  L_dyn — NEGATIVES
# ═══════════════════════════════════════════════════════════════════════════

def test_within_trajectory_negatives_come_from_the_same_trajectory_two_apart():
    c_seq = S.sparse_simplex(16, C.DEPTH)
    neg = sample_within_trajectory_negatives(c_seq, min_gap=2)
    assert neg.shape == c_seq.shape
    for b in range(16):
        for h in range(C.DEPTH):
            matches = [j for j in range(C.DEPTH) if torch.equal(neg[b, h], c_seq[b, j])]
            assert matches, "negative is not from this trajectory"
            assert all(abs(j - h) >= 2 for j in matches), "negative is too close in time"


def test_within_trajectory_negatives_need_a_long_enough_window():
    with pytest.raises(ValueError):
        sample_within_trajectory_negatives(S.sparse_simplex(2, 2), min_gap=2)


def test_negatives_default_is_within_trajectory_and_adds_a_hinge():
    assert DynLoss().negatives == "within_trajectory"
    bank, z0 = frozen_bank(), belief(4)
    c_true = S.sparse_simplex(4, C.DEPTH)
    tgt = rollout_states(bank, z0, c_true)

    on = dyn_loss(bank, z0, c_true, tgt, negatives="within_trajectory")
    off = dyn_loss(bank, z0, c_true, tgt, negatives="none")
    assert off["neg"].item() == 0.0
    assert on["neg"].item() >= 0.0
    assert torch.allclose(on["dyn"], off["dyn"], atol=1e-5)      # positive term unchanged


def test_hinge_matches_the_written_formula():
    """neg = neg_weight * sum_h w_h * mean relu(margin - (d_neg - d_pos))."""
    bank, z0 = frozen_bank(), belief(4)
    c_seq, c_neg = S.sparse_simplex(4, C.DEPTH), S.sparse_simplex(4, C.DEPTH)
    tgt = torch.randn(4, C.DEPTH, C.K, C.D)

    out = dyn_loss(bank, z0, c_seq, tgt, negatives="within_trajectory",
                   c_neg=c_neg, neg_weight=2.0, neg_margin=0.3)

    pos, neg, want = z0, z0, 0.0
    for h in range(C.DEPTH):
        pos, neg = bank.step(c_seq[:, h], pos), bank.step(c_neg[:, h], neg)
        d_pos = ln_cosine_distance(pos, tgt[:, h])
        d_neg = ln_cosine_distance(neg, tgt[:, h])
        want = want + C.DYN_WEIGHTS[h] * torch.relu(0.3 - (d_neg - d_pos)).mean()
    assert torch.allclose(out["neg"], 2.0 * want, atol=1e-5)


def test_hinge_cannot_fire_on_a_perfect_prediction_at_zero_margin():
    """d_pos == 0 and d_neg >= 0, so relu(0 - (d_neg - 0)) is identically 0 —
    the hinge only ever charges for negatives that are *closer* than the margin."""
    bank, z0 = frozen_bank(), belief(4)
    c_true = S.sparse_simplex(4, C.DEPTH)
    tgt = rollout_states(bank, z0, c_true)          # perfect prediction
    out = dyn_loss(bank, z0, c_true, tgt, negatives="within_trajectory", neg_margin=0.0)
    assert out["dyn"].item() == pytest.approx(0.0, abs=1e-4)
    assert out["neg"].item() == pytest.approx(0.0, abs=1e-6)


# ═══════════════════════════════════════════════════════════════════════════
#  Delta_op — THE BUILD ASSERT   (done-when 2)
# ═══════════════════════════════════════════════════════════════════════════

def test_delta_op_is_positive_when_the_operator_is_the_true_one():
    """Sign convention: d is a DISTANCE, so the true operator must be CLOSER
    and Delta_op = d(rand) - d(true) must be positive."""
    bank, z0 = frozen_bank(), belief(8)
    c_true = S.sparse_simplex(8, C.DEPTH)
    out = dyn_loss(bank, z0, c_true, rollout_states(bank, z0, c_true))
    assert out["delta_op"].item() > 0.1
    assert out["cos_pos"].item() == pytest.approx(1.0, abs=1e-4)


def test_delta_op_flatlines_when_the_coefficients_are_uninformative():
    """The failure mode Delta_op exists to catch: `c` that carries nothing.

    A random c scores no better than another random c, so Delta_op sits at ~0
    (measured: -0.02 +/- 0.03) instead of the ~1.0 an informative c produces.
    This is the "collapsed to a plain latent policy" signature.
    """
    bank = frozen_bank()
    deltas = []
    for _ in range(8):
        z0 = belief(8)
        tgt = rollout_states(bank, z0, S.sparse_simplex(8, C.DEPTH))
        deltas.append(dyn_loss(bank, z0, S.sparse_simplex(8, C.DEPTH), tgt,
                               negatives="none")["delta_op"].item())
    mean = sum(deltas) / len(deltas)
    assert mean < 0.1, f"uninformative c should show no signal, got {mean:.3f}"


def test_delta_op_is_detached_and_always_reported():
    bank, z0 = frozen_bank(), belief(2)
    out = dyn_loss(bank, z0, S.sparse_simplex(2, C.DEPTH),
                   torch.randn(2, C.DEPTH, C.K, C.D))
    assert "delta_op" in out and not out["delta_op"].requires_grad


# ═══════════════════════════════════════════════════════════════════════════
#  EMA TARGET MACHINERY
# ═══════════════════════════════════════════════════════════════════════════

def test_ema_update_is_the_documented_convex_combination():
    online = nn.Linear(4, 4)
    target = nn.Linear(4, 4)
    w_t, w_o = target.weight.detach().clone(), online.weight.detach().clone()
    ema_update(target, online, tau=C.EMA_TAU)
    assert torch.allclose(target.weight, C.EMA_TAU * w_t + (1 - C.EMA_TAU) * w_o, atol=1e-6)


def test_ema_update_tau_extremes():
    online, target = nn.Linear(3, 3), nn.Linear(3, 3)
    frozen = target.weight.detach().clone()
    ema_update(target, online, tau=1.0)
    assert torch.allclose(target.weight, frozen)
    ema_update(target, online, tau=0.0)
    assert torch.allclose(target.weight, online.weight)
    with pytest.raises(ValueError):
        ema_update(target, online, tau=1.5)


def test_ema_estimator_is_a_detached_copy_with_the_contract_tau():
    online = S.StubEstimator()
    ema = EmaEstimator(online, tau=C.EMA_TAU)
    assert ema.tau == C.EMA_TAU == 0.996
    assert ema.target is not online
    assert all(not p.requires_grad for p in ema.target.parameters())

    z = ema(S.make_obs_feats(b=3), None)
    C.assert_belief(z)
    assert not z.requires_grad
    ema.update(online)                       # no-op for a parameterless stub, must not raise


def test_ema_estimator_target_stays_in_eval():
    ema = EmaEstimator(nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.5)))
    ema.train()
    assert not ema.target.training


# ═══════════════════════════════════════════════════════════════════════════
#  DONE-WHEN 1 + 2 — THE SYNTHETIC TASK
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
def test_dyn_decreases_and_delta_op_turns_positive_on_a_synthetic_task():
    """PLAN 4.C done-when 1 and 2, together, on the one task that can separate
    "the operator is doing the work" from "the belief is doing the work".

    Setup — `c` is the ONLY informative input:
      * a real bank (stubs.StubBank: the 2x2 rotation-decay algebra, frozen),
      * i.i.d. Gaussian beliefs, so z_t itself predicts nothing,
      * a known ground-truth operator per segment, drawn from a small table of
        TOPK-sparse simplex points,
      * targets produced by APPLYING that operator sequentially.

    q_Delta sees only (z_{h-1}, z_h) and must recover which operator was
    applied; L_dyn is the only training signal (no supervision on c at all).
    If the head, the straight-through gradient or the rollout were wrong, the
    loss would sit flat here — a random z_t gives it nothing else to fit.

    Measured before/after on a HELD-OUT set of batches, not on the training
    curve: the first few training steps already move the loss, so "step 0 vs
    step N" understates the drop.

    Sized for a CPU box: two horizons and `negatives="none"`, because the hinge
    branch triples the rollout cost and is checked exactly, against its written
    formula, in `test_hinge_matches_the_written_formula`.

    TWO PROPERTIES OF THE TOY THAT ARE DELIBERATE, NOT CONVENIENCE
    ---------------------------------------------------------------
    (a) Ground-truth operators are TOPK atoms at EQUAL weight, not vertices.
        A renormalised top-k head represents an equal-weight TOPK mixture
        exactly, so the loss genuinely floors at 0. A one-hot target is not
        reachable at all — it needs an unbounded within-support logit gap — and
        the toy would then be measuring the head's saturation rate. (Measured
        here: a one-hot table reaches only 0.91x in the same budget.)
    (b) The operators are variations on a theme: they share TOPK-1 atoms and
        differ in one, which is the atom that has to be read off (z_t, z_next).
        With fully disjoint supports the loss surface is flat until the support
        is recovered almost exactly — A(c) is a sum of TOPK random-phase
        rotations, so getting one atom of four right barely moves the cosine —
        and a few hundred CPU steps then measure exploration luck rather than
        whether L_dyn and its gradient path work. (Measured: disjoint supports
        stall at ~0.8x and 1 of 4 atoms even at batch 32 and 300 steps.)
        Support exploration at M=128 is a 100k-step training-dynamics question,
        and the thing that addresses it is L_balance, not L_dyn.
    """
    torch.manual_seed(7)
    b, horizons, steps, n_ops = 16, 2, 150, 4

    bank = frozen_bank()
    gen = torch.Generator().manual_seed(11)
    pool = torch.randperm(C.M, generator=gen)[:C.TOPK - 1 + n_ops]
    table = torch.zeros(n_ops, C.M)
    for i in range(n_ops):                          # shared atoms + one distinct
        table[i, torch.cat([pool[:C.TOPK - 1], pool[C.TOPK - 1 + i, None]])] = 1.0 / C.TOPK
    C.assert_simplex(table)
    qd = QDelta(hidden=384, n_queries=1, n_heads=4, d_kv=32)
    opt = torch.optim.Adam(qd.parameters(), lr=2e-3)
    crit = DynLoss(negatives="none")

    def batch():
        idx = torch.randint(0, n_ops, (b, horizons), generator=gen)
        z0 = torch.randn(b, C.K, C.D, generator=gen)
        c_true = table[idx]
        return z0, c_true, rollout_states(bank, z0, c_true)

    def predict(z0, tgt):
        states = [z0] + [tgt[:, h] for h in range(horizons)]
        return torch.stack([qd(states[h], states[h + 1]) for h in range(horizons)], 1)

    held_out = [batch() for _ in range(2)]

    @torch.no_grad()
    def evaluate():
        """(L_dyn, Delta_op, recovered fraction of the true operator's mass)."""
        outs = [crit(bank, z0, predict(z0, tgt), tgt) for z0, _, tgt in held_out]
        # <c_hat, c_true> / <c_true, c_true>: 1.0 means the operator was recovered
        overlap = [(predict(z0, tgt) * c).sum(-1).mean() / (c * c).sum(-1).mean()
                   for z0, c, tgt in held_out]
        n = len(outs)
        return (sum(o["dyn"].item() for o in outs) / n,
                sum(o["delta_op"].item() for o in outs) / n,
                sum(o.item() for o in overlap) / n)

    before, dop_before, ov_before = evaluate()
    for _ in range(steps):
        z0, _, tgt = batch()
        out = crit(bank, z0, predict(z0, tgt), tgt)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        opt.step()
    after, dop_after, ov_after = evaluate()

    print(f"\n[synthetic] L_dyn {before:.4f} -> {after:.4f} ({after / before:.2f}x), "
          f"Delta_op {dop_before:+.4f} -> {dop_after:+.4f}, "
          f"operator recovered {ov_before:.2f} -> {ov_after:.2f}")

    # 1. L_dyn drops materially
    assert after < 0.5 * before, f"L_dyn went {before:.4f} -> {after:.4f}, not a material drop"
    # 2. Delta_op is positive: the recovered operator really is the closer one
    assert dop_after > 0.1, f"Delta_op flatlined at {dop_after:.4f}; c carries nothing"
    # 3. and it went down for the right reason — the true operator was recovered,
    #    not some other c that happens to score well
    assert ov_after > 0.5, f"only {ov_after:.2f} of the true operator's mass recovered"
    # the head is still emitting legal coefficients after training
    C.assert_simplex(predict(*[held_out[0][i] for i in (0, 2)]).detach().flatten(0, 1))


# ═══════════════════════════════════════════════════════════════════════════
#  L_act   (done-when 4, 7)
# ═══════════════════════════════════════════════════════════════════════════

def test_act_loss_is_the_decoder_loss():
    torch.manual_seed(3)
    dec = Decoder([BODY_A], **SMALL_DEC)
    pr, c = proprio(4), S.sparse_simplex(4)
    a, x0, t = torch.randn(4, C.H_OP, DOF_A), torch.randn(4, C.H_OP, DOF_A), torch.rand(4)
    got = act_loss(dec, pr, c, a, embodiment=BODY_A, t=t, noise=x0)
    assert torch.allclose(got, dec.loss(pr, c, a, embodiment=BODY_A, t=t, noise=x0))


def test_act_loss_handles_action_free_data():
    """done-when 7: R1 data has actions=None. Zero, and the graph survives."""
    dec = Decoder([BODY_A], **SMALL_DEC)
    pr = proprio(3).requires_grad_(True)
    c = S.sparse_simplex(3)
    loss = act_loss(dec, pr, c, None, embodiment=BODY_A)
    assert loss.ndim == 0 and loss.item() == 0.0
    loss.backward()                                  # must not raise
    assert torch.count_nonzero(pr.grad) == 0
    assert all(p.grad is None for p in dec.parameters())


def test_act_loss_action_free_sums_with_other_losses():
    dec = Decoder([BODY_A], **SMALL_DEC)
    z, c = belief(2), S.sparse_simplex(2)
    total = act_loss(dec, proprio(2), c, None, embodiment=BODY_A) + balance_loss(c) \
        + dyn_loss(frozen_bank(), z, S.sparse_simplex(2, C.DEPTH),
                   torch.randn(2, C.DEPTH, C.K, C.D), negatives="none")["loss"]
    assert torch.isfinite(total)


def test_zero_loss_is_detached_when_nothing_requires_grad():
    z = torch.randn(2, 3)
    assert zero_loss(z).item() == 0.0 and not zero_loss(z).requires_grad
    assert zero_loss(z.clone().requires_grad_(True)).requires_grad


@pytest.mark.parametrize("body,dof", [(BODY_A, DOF_A), (BODY_B, DOF_B)])
def test_act_loss_dispatches_by_embodiment(body, dof):
    """done-when 4: two synthetic bodies, dof 7 and 14, routed by name."""
    dec = Decoder([BODY_A, BODY_B], **SMALL_DEC)
    loss = act_loss(dec, proprio(3, dof), S.sparse_simplex(3),
                    torch.randn(3, C.H_OP, dof), embodiment=body)
    assert torch.isfinite(loss) and loss.ndim == 0
    loss.backward()
    other = BODY_B if body == BODY_A else BODY_A
    assert all(p.grad is None for p in dec.body(other).parameters())


def test_act_loss_rejects_the_wrong_dof_for_the_body():
    dec = Decoder([BODY_A, BODY_B], **SMALL_DEC)
    with pytest.raises(ValueError):
        act_loss(dec, proprio(2), S.sparse_simplex(2),
                 torch.randn(2, C.H_OP, DOF_B), embodiment=BODY_A)


def test_act_loss_accepts_a_whole_window_of_segments():
    """TransitionWindow.actions is (B, DEPTH, H_OP, dof); folding DEPTH into the
    batch must give exactly the same number as looping."""
    torch.manual_seed(5)
    dec = Decoder([BODY_A], **SMALL_DEC)
    b = 3
    pr = torch.randn(b, C.DEPTH, DOF_A)
    c = S.sparse_simplex(b, C.DEPTH)
    a = torch.randn(b, C.DEPTH, C.H_OP, DOF_A)
    x0, t = torch.randn(b * C.DEPTH, C.H_OP, DOF_A), torch.rand(b * C.DEPTH)

    got = act_loss(dec, pr, c, a, embodiment=BODY_A, t=t, noise=x0)
    want = dec.loss(pr.flatten(0, 1), c.flatten(0, 1), a.flatten(0, 1),
                    embodiment=BODY_A, t=t, noise=x0)
    assert torch.allclose(got, want, atol=1e-6)


def test_act_loss_rejects_h_plan_segments():
    dec = Decoder([BODY_A], **SMALL_DEC)
    with pytest.raises(ValueError):
        act_loss(dec, proprio(2), S.sparse_simplex(2),
                 torch.randn(2, C.H_PLAN, DOF_A), embodiment=BODY_A)


# ═══════════════════════════════════════════════════════════════════════════
#  q_a REGRESSION ONTO sg(q_Delta)
# ═══════════════════════════════════════════════════════════════════════════

def test_q_action_regression_target_is_stop_gradded():
    """q_Delta defines the coefficient space; q_a moves into it, never both."""
    qd, qa = QDelta(**SMALL_QD), QAction([BODY_A], **SMALL_QA)
    z = belief(3)
    c_delta = qd(z, belief(3))
    loss = q_action_regression_loss(qa, torch.randn(3, C.H_OP, DOF_A), z,
                                    c_delta, embodiment=BODY_A)
    loss.backward()
    assert all(p.grad is None or not p.grad.any() for p in qd.parameters())
    assert any(p.grad is not None and p.grad.any() for p in qa.body(BODY_A).parameters())


def test_q_action_regression_is_zero_at_a_perfect_match():
    qa = QAction([BODY_A], **SMALL_QA)
    z, a = belief(3), torch.randn(3, C.H_OP, DOF_A)
    c_a = qa(a, z, embodiment=BODY_A).detach()
    loss = q_action_regression_loss(qa, a, z, c_a, embodiment=BODY_A)
    assert loss.item() == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("mode", ["mse", "ce"])
def test_q_action_regression_modes_are_finite_and_dispatch(mode):
    qa = QAction([BODY_A, BODY_B], **SMALL_QA)
    loss = q_action_regression_loss(qa, torch.randn(2, C.H_OP, DOF_B), belief(2),
                                    S.sparse_simplex(2), embodiment=BODY_B, mode=mode)
    assert torch.isfinite(loss) and loss.item() >= 0.0


def test_align_flip_moves_the_gradient_to_the_other_encoder():
    """`align_to: q_a` -- q_Delta regresses onto sg(q_a), and ONLY q_Delta moves.

    Both directions of the align term, as the training loop computes them
    inline, plus the reference impl in `losses/act.py`. The inline form is what
    actually trains (`loom/train/loop.py::compute_losses`); this pins that the
    two agree and that exactly one side is stop-gradded either way.
    """
    torch.manual_seed(5)
    qd, qa = QDelta(**SMALL_QD), QAction([BODY_A], **SMALL_QA)
    z, zn = belief(3), belief(3)
    a = torch.randn(3, C.H_OP, DOF_A)

    def leaves():
        c_delta = qd(z, zn)
        c_a = qa(a, z, embodiment=BODY_A)
        c_delta.retain_grad(); c_a.retain_grad()
        return c_delta, c_a

    # ── ALIGN-FLIP: (c_delta - sg(c_a))^2 ────────────────────────────────
    qd.zero_grad(); qa.zero_grad()
    c_delta, c_a = leaves()
    ((c_delta - c_a.detach()) ** 2).sum(-1).mean().backward()
    assert c_delta.grad is not None and c_delta.grad.abs().sum() > 0
    assert c_a.grad is None or float(c_a.grad.abs().sum()) == 0.0
    assert any(p.grad is not None and p.grad.any() for p in qd.parameters())
    assert all(p.grad is None or not p.grad.any()
               for p in qa.body(BODY_A).parameters())

    # ── default: (c_a - sg(c_delta))^2 ───────────────────────────────────
    qd.zero_grad(); qa.zero_grad()
    c_delta, c_a = leaves()
    ((c_a - c_delta.detach()) ** 2).sum(-1).mean().backward()
    assert c_a.grad is not None and c_a.grad.abs().sum() > 0
    assert c_delta.grad is None or float(c_delta.grad.abs().sum()) == 0.0
    assert any(p.grad is not None and p.grad.any()
               for p in qa.body(BODY_A).parameters())
    assert all(p.grad is None or not p.grad.any() for p in qd.parameters())

    # ── the reference impl agrees, both ways ─────────────────────────────
    qd.zero_grad(); qa.zero_grad()
    ref = q_action_regression_loss(qa, a, z, qd(z, zn), embodiment=BODY_A,
                                   direction="q_delta<-q_a")
    ref.backward()
    assert any(p.grad is not None and p.grad.any() for p in qd.parameters())
    assert all(p.grad is None or not p.grad.any()
               for p in qa.body(BODY_A).parameters())
    with torch.no_grad():
        expect = ((qd(z, zn) - qa(a, z, embodiment=BODY_A)) ** 2).sum(-1).mean()
    assert float(ref) == pytest.approx(float(expect), rel=1e-6)

    with pytest.raises(ValueError, match="direction"):
        q_action_regression_loss(qa, a, z, qd(z, zn), embodiment=BODY_A,
                                 direction="both")


def test_q_action_regression_handles_action_free_data():
    qa = QAction([BODY_A], **SMALL_QA)
    z = belief(2).requires_grad_(True)
    loss = q_action_regression_loss(qa, None, z, S.sparse_simplex(2), embodiment=BODY_A)
    assert loss.item() == 0.0
    loss.backward()


# ═══════════════════════════════════════════════════════════════════════════
#  L_proposal   (done-when 9)
# ═══════════════════════════════════════════════════════════════════════════

def test_proposal_bc_is_finite_non_negative_and_scalar():
    p = S.StubProposal()
    z, lang = belief(5), torch.randn(5, 16, 1152)
    loss = proposal_bc_loss(p, z, lang, S.sparse_simplex(5))
    assert loss.ndim == 0 and torch.isfinite(loss) and loss.item() >= 0.0


def test_proposal_bc_per_sample_shape():
    p = S.StubProposal()
    z, lang, c = belief(5), torch.randn(5, 16, 1152), S.sparse_simplex(5)
    nll = proposal_bc_loss(p, z, lang, c, reduction="none")
    assert nll.shape == (5,)
    assert torch.isfinite(nll).all() and (nll >= 0).all()
    assert torch.allclose(nll.mean(), proposal_bc_loss(p, z, lang, c))
    assert torch.allclose(nll.sum(), proposal_bc_loss(p, z, lang, c, reduction="sum"))
    with pytest.raises(ValueError):
        proposal_bc_loss(p, z, lang, c, reduction="median")


class _DifferentiableProposal(nn.Module):
    """Minimal `contracts.Proposal` whose log_prob depends on its own parameter
    AND on `c`, so the stop-grad on `c` is actually observable. StubProposal's
    log_prob has no graph at all, which would make the test vacuous."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.zeros(C.M))

    def sample(self, z, lang, n):
        return S.sparse_simplex(z.shape[0], n, device=z.device, dtype=z.dtype)

    def log_prob(self, z, lang, c):
        return (c * torch.log_softmax(self.w, -1)).sum(-1)      # <= 0


def test_proposal_bc_target_is_stop_gradded():
    """Otherwise the encoders get dragged towards whatever pi_c already likes."""
    p = _DifferentiableProposal()
    assert isinstance(p, C.Proposal)
    c = S.sparse_simplex(4).requires_grad_(True)
    loss = proposal_bc_loss(p, belief(4), torch.randn(4, 16, 1152), c)
    assert loss.item() >= 0.0
    loss.backward()
    assert p.w.grad is not None and p.w.grad.abs().sum() > 0      # pi_c learns
    assert c.grad is None                                        # the target does not


def test_proposal_bc_prefers_the_likelier_support():
    """A real BC signal: the loss must be lower for coefficients the proposal
    actually assigns more mass to."""
    p = S.StubProposal()
    z, lang = belief(2), torch.randn(2, 16, 1152)
    logits = p._logits(z)
    best = torch.zeros_like(logits).scatter_(
        1, logits.topk(C.TOPK, -1).indices, 1.0 / C.TOPK)
    worst = torch.zeros_like(logits).scatter_(
        1, (-logits).topk(C.TOPK, -1).indices, 1.0 / C.TOPK)
    assert proposal_bc_loss(p, z, lang, best) < proposal_bc_loss(p, z, lang, worst)


def test_proposal_bc_rejects_a_batch_mismatch():
    with pytest.raises(ValueError):
        proposal_bc_loss(S.StubProposal(), belief(4), torch.randn(4, 16, 1152),
                         S.sparse_simplex(3))


# ═══════════════════════════════════════════════════════════════════════════
#  L_balance   (done-when 8)
# ═══════════════════════════════════════════════════════════════════════════

def uniform_usage_batch() -> torch.Tensor:
    """M / TOPK samples whose supports tile all M operators exactly once."""
    n = C.M // C.TOPK
    c = torch.zeros(n, C.M)
    order = torch.randperm(C.M)
    for i in range(n):
        c[i, order[i * C.TOPK:(i + 1) * C.TOPK]] = 1.0 / C.TOPK
    return c


def test_balance_is_zero_for_a_uniform_batch_mean_and_positive_otherwise():
    """done-when 8. Each sample is still hard-sparse; only the MEAN is flat."""
    c = uniform_usage_batch()
    C.assert_simplex(c)
    assert torch.allclose(operator_usage(c), torch.full((C.M,), 1.0 / C.M), atol=1e-7)
    assert balance_kl(c).item() == pytest.approx(0.0, abs=1e-6)

    skewed = torch.zeros(8, C.M)
    skewed[:, :C.TOPK] = 1.0 / C.TOPK               # everyone uses the same 4
    assert balance_kl(skewed).item() > 0.0
    assert balance_kl(skewed).item() == pytest.approx(math.log(C.M / C.TOPK), abs=1e-5)


def test_balance_is_minimised_at_uniform():
    """Perturbing away from the flat usage can only increase the loss."""
    base = uniform_usage_batch()
    assert balance_kl(base).item() == pytest.approx(0.0, abs=1e-6)
    for _ in range(8):
        pert = torch.cat([base, S.sparse_simplex(3)], 0)
        assert balance_kl(pert).item() > 1e-4        # any skew strictly costs


def test_balance_uses_the_contract_coefficient():
    c = S.sparse_simplex(8)
    assert torch.allclose(balance_loss(c), C.BALANCE_COEF * balance_kl(c))
    # 3e-3 -> 1e-2, owner-authorised. `balance_kl` itself is unchanged and is
    # NOT what R0-A executes any more: `loop._switch_balance` is (see that
    # function, and `tests/test_train.py::test_switch_balance_*`). This file
    # still pins the KL because it is the reference the Switch form replaced.
    assert C.BALANCE_COEF == 1e-2


def test_balance_direction_is_forward_kl_and_stays_finite_on_dead_operators():
    """KL(usage || uniform), not KL(uniform || usage). The reverse direction is
    +inf as soon as one operator goes unused — and with hard top-k and any batch
    smaller than M/TOPK, most operators are unused every single step."""
    c = S.sparse_simplex(2)                          # at most 8 of 128 used
    kl = balance_kl(c)
    assert torch.isfinite(kl) and kl.item() > 0.0

    p = operator_usage(c)
    forward = (p * (p.clamp_min(1e-8).log() + math.log(C.M))).sum()
    assert torch.allclose(kl, forward, atol=1e-5)
    assert torch.isinf(-(math.log(C.M) + (1.0 / C.M) * p.log().sum()))   # the reverse


def test_balance_gradient_pushes_a_dead_operator_back_up():
    """The whole point. Requires the straight-through head to pass gradient to
    logits OUTSIDE the current top-k support."""
    logits = torch.randn(16, C.M)
    logits[:, 5] = -10.0                             # operator 5 is dead
    logits = logits.requires_grad_(True)
    c = topk_simplex_st(logits)
    assert c[:, 5].sum() == 0.0
    balance_loss(c).backward()
    assert logits.grad[:, 5].abs().sum() > 0, "a dead operator gets no gradient"
    assert (logits.grad[:, 5] < 0).all(), "gradient must push the dead operator UP"


def test_balance_rejects_a_wrong_width():
    with pytest.raises(ValueError):
        balance_kl(torch.rand(4, C.M + 1))


# ═══════════════════════════════════════════════════════════════════════════
#  THE FOUR TERMS TOGETHER  (dyn + act + proposal + balance, PLAN 9 — no more)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("action_free", [False, True])
def test_the_full_r0a_loss_sum_backpropagates(action_free):
    torch.manual_seed(13)
    bank = frozen_bank()
    qd, qa = QDelta(**SMALL_QD), QAction([BODY_A], **SMALL_QA)
    dec, prop = Decoder([BODY_A], **SMALL_DEC), S.StubProposal()

    b = 2
    z = [belief(b) for _ in range(C.N_STATES)]
    lang = torch.randn(b, 16, 1152)
    actions = None if action_free else torch.randn(b, C.DEPTH, C.H_OP, DOF_A)

    c_delta = torch.stack([qd(z[h], z[h + 1]) for h in range(C.DEPTH)], 1)
    tgt = torch.stack(z[1:], 1)

    total = dyn_loss(bank, z[0], c_delta, tgt)["loss"] + balance_loss(c_delta)
    total = total + proposal_bc_loss(prop, z[0], lang, c_delta[:, 0])
    if actions is not None:
        c_act = qa(actions[:, 0], z[0], embodiment=BODY_A)
        total = total + q_action_regression_loss(qa, actions[:, 0], z[0],
                                                 c_delta[:, 0], embodiment=BODY_A)
        total = total + act_loss(dec, proprio(b), c_act, actions[:, 0],
                                 embodiment=BODY_A)
    else:
        total = total + act_loss(dec, proprio(b), c_delta[:, 0], None,
                                 embodiment=BODY_A)

    assert torch.isfinite(total)
    total.backward()
    assert any(p.grad is not None and p.grad.any() for p in qd.parameters())
