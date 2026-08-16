"""
LOOM — Team B gate (PLAN.md 4.B "Done when").

Covers `loom/model/{bank,rollout,estimator}.py`:

  1  ||A(c)||_2 <= RHO and ||b(c)|| <= B_MAX over 10k random simplex draws
  2  rollout matches a naive explicit loop to 1e-5
  3  rollout is NOT lambda composition — the affine-composition trap
  4  N=1000, DEPTH=4 under 5 ms on one A100                        [gpu]
  5  estimator >= 30 Hz with 7 streams                             [gpu]
  6  no dtype promotion out of bf16, nothing complex
  7  shape conformance against contracts + Protocol conformance
  8  broadcasting over arbitrary leading dims
  9  the S4D init actually produces a spread of timescales
 10  z_prev genuinely conditions the estimator
 11  parameter budgets: bank ~25 M, estimator ~150 M
 12  `view_as_complex` appears nowhere in loom/model/

GPU items are budgets, not correctness; they skip on a CPU box.
"""

from __future__ import annotations

import io
import math
import time
import tokenize
from pathlib import Path

import pytest
import torch

import contracts as C
import stubs as S
from loom.model.bank import OperatorBank
from loom.model.estimator import Estimator
from loom.model.rollout import rollout

MODEL_DIR = Path(__file__).resolve().parents[1] / "loom" / "model"


# ═══════════════════════════════════════════════════════════════════════════
#  FIXTURES / HELPERS
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def bank() -> OperatorBank:
    torch.manual_seed(0)
    return OperatorBank()


@pytest.fixture(scope="module")
def bank_bf16() -> OperatorBank:
    torch.manual_seed(0)
    return OperatorBank().to(torch.bfloat16)


def _small_estimator(**kw) -> Estimator:
    """A depth-2, narrow-input estimator. Same code path, CPU-affordable."""
    torch.manual_seed(0)
    return Estimator(feat_dim=64, depth=2, heads=8, **kw)


def _small_feats(b: int = 2, v: int = 2, p: int = 8, dof: int = 7, l: int = 4):
    return S.make_obs_feats(b=b, v=v, p=p, f=64, dof=dof, l=l)


def _naive_step(bank: OperatorBank, c: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Reference `A(c) z + b(c)` built as an explicit (2, 2) matmul.

    Deliberately a different code path from `OperatorBank.step`'s four
    elementwise ops, so agreement is evidence rather than tautology.
    """
    a, b = bank.mix(c)
    mat = torch.stack([torch.stack([a, -b], dim=-1),
                       torch.stack([b, a], dim=-1)], dim=-2)      # (..., K, J, 2, 2)
    zr = z.reshape(*z.shape[:-1], a.shape[-1], 2)
    out = torch.einsum('...ij,...j->...i', mat, zr)
    return out.reshape(z.shape) + bank.bias(c)


def _block_apply(a: torch.Tensor, b: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Apply a block pair `(a, b)` to `z` without any bias."""
    zr = z.reshape(*z.shape[:-1], a.shape[-1], 2)
    x, y = zr[..., 0], zr[..., 1]
    return torch.stack([a * x - b * y, b * x + a * y], dim=-1).reshape(z.shape)


# ═══════════════════════════════════════════════════════════════════════════
#  1 · BOUNDS
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
def test_bounds_over_10k_simplex_draws(bank):
    """PLAN 4.B: both bounds over 10k draws.

    Chunked 20 x 500 for peak memory, not for speed. `mix` on a `(n, M)` batch
    materialises two `(n, K, D//2)` tensors plus the `sqrt(a^2 + b^2)`
    intermediate; at n=10000 that is ~2 GiB each in fp32 and gets the process
    killed on a shared login node. At n=500 the peak is ~100 MiB. The draws are
    i.i.d., so 20 chunks of 500 is exactly 10k draws of coverage.
    """
    torch.manual_seed(1)
    draws = 0
    for _ in range(20):
        C.assert_contractive(bank, n=500)
        C.assert_bias_bounded(bank, n=500)
        draws += 500
    assert draws == 10_000


def test_spectral_radius_directly(bank):
    """sqrt(a^2 + b^2) <= RHO. A is block-diagonal, so that IS its spectral norm."""
    torch.manual_seed(2)
    c = S.sparse_simplex(256)
    with torch.no_grad():
        a, b = bank.mix(c)
        r = torch.sqrt(a ** 2 + b ** 2)
    assert float(r.max()) <= C.RHO + 1e-5, f"max block magnitude {float(r.max()):.6f}"
    # non-vacuous: the bank is not trivially near zero
    assert float(r.max()) > 0.5 * C.RHO


def test_bias_norm_directly(bank):
    """||b(c)|| <= B_MAX."""
    torch.manual_seed(3)
    c = S.sparse_simplex(256)
    with torch.no_grad():
        nb = bank.bias(c).flatten(1).norm(dim=1)
    assert float(nb.max()) <= C.B_MAX + 1e-5, f"max bias norm {float(nb.max()):.6f}"


def test_bias_cap_is_not_vacuous():
    """With a large `b_raw` the cap must bind at exactly B_MAX, not merely pass."""
    torch.manual_seed(4)
    b = OperatorBank(bias_init_norm=50.0)
    with torch.no_grad():
        per_op = b.bias_bank().flatten(1).norm(dim=1)
    assert torch.allclose(per_op, torch.full_like(per_op, C.B_MAX), atol=1e-4)
    C.assert_bias_bounded(b, n=512)


def test_bounds_hold_off_the_init(bank):
    """The bounds are structural, not a property of the initialisation."""
    torch.manual_seed(5)
    rogue = OperatorBank()
    with torch.no_grad():
        rogue.log_r.normal_(0.0, 20.0)      # push sigmoid hard against both rails
        rogue.omega.normal_(0.0, 20.0)
        rogue.b_raw.normal_(0.0, 20.0)
    C.assert_contractive(rogue, n=1024)
    C.assert_bias_bounded(rogue, n=1024)


# ═══════════════════════════════════════════════════════════════════════════
#  2 · ROLLOUT == NAIVE LOOP
# ═══════════════════════════════════════════════════════════════════════════

def test_rollout_matches_naive_loop(bank):
    torch.manual_seed(6)
    z0 = torch.randn(2, C.K, C.D)
    c = S.sparse_simplex(2, 3, C.DEPTH)

    ref = z0.unsqueeze(1).expand(2, 3, C.K, C.D)
    for d in range(C.DEPTH):
        ref = _naive_step(bank, c[:, :, d], ref)

    out = bank.rollout(c, z0)
    assert out.shape == (2, 3, C.K, C.D)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5), \
        f"max |rollout - naive| = {float((out - ref).abs().max()):.3e}"


def test_free_function_and_method_agree(bank):
    torch.manual_seed(7)
    z0 = torch.randn(2, C.K, C.D)
    c = S.sparse_simplex(2, 3, C.DEPTH)
    assert torch.equal(rollout(bank, c, z0), bank.rollout(c, z0))


def test_rollout_works_against_any_bank():
    """The free function takes a contracts.Bank, not an OperatorBank."""
    torch.manual_seed(8)
    stub = S.StubBank()
    z0 = torch.randn(2, C.K, C.D)
    c = S.sparse_simplex(2, 3, C.DEPTH)
    assert torch.allclose(rollout(stub, c, z0), stub.rollout(c, z0), atol=1e-6)


def test_rollout_rejects_wrong_depth(bank):
    """One c is one operator is H_OP steps. Never H_PLAN."""
    z0 = torch.randn(2, C.K, C.D)
    with pytest.raises(ValueError):
        bank.rollout(S.sparse_simplex(2, 3, C.H_PLAN), z0)


def test_rollout_does_not_explode(bank):
    """rho^T ||z|| + (1 - rho^T)/(1 - rho) B_MAX, same bound as test_contracts."""
    torch.manual_seed(9)
    z0 = torch.randn(8, C.K, C.D)
    leaves = bank.rollout(S.sparse_simplex(8, 1, C.DEPTH), z0)
    bound = (C.RHO ** C.DEPTH) * z0.flatten(1).norm(dim=1) \
        + (1 - C.RHO ** C.DEPTH) / (1 - C.RHO) * C.B_MAX
    assert (leaves.squeeze(1).flatten(1).norm(dim=1) <= bound + 1e-3).all()


# ═══════════════════════════════════════════════════════════════════════════
#  3 · THE AFFINE-COMPOSITION TRAP
# ═══════════════════════════════════════════════════════════════════════════

def test_rollout_is_not_lambda_composition():
    """DEPTH sequential affine steps != one composed lambda plus one bias.

    Composing `(A_d, b_d)` gives

        z_D = A_D...A_1 z_0 + A_D...A_2 b_1 + ... + A_D b_{D-1} + b_D

    Multiplying the lambdas alone and adding a single bias keeps the first term
    and silently drops every propagated bias. The shapes match, the spectral
    bound still holds, and the answer is wrong.

    **If this test ever fails, someone has "optimised" rollout into a compose.
    Revert it.** There is no compose() and there must not be one.
    """
    torch.manual_seed(10)
    trap = OperatorBank(bias_init_norm=1.0)         # bias at the B_MAX cap
    z0 = torch.randn(2, C.K, C.D) * 0.01            # so bias is not swamped
    c = S.sparse_simplex(2, 4, C.DEPTH)

    with torch.no_grad():
        correct = trap.rollout(c, z0)

        a, b = trap.mix(c[:, :, 0])
        for d in range(1, C.DEPTH):
            a2, b2 = trap.mix(c[:, :, d])
            a, b = a2 * a - b2 * b, b2 * a + a2 * b     # lambda product only
        zx = z0.unsqueeze(1).expand(2, 4, C.K, C.D)
        wrong_last = _block_apply(a, b, zx) + trap.bias(c[:, :, -1])
        wrong_sum = _block_apply(a, b, zx) + sum(trap.bias(c[:, :, d])
                                                 for d in range(C.DEPTH))

    for name, wrong in (("last-bias", wrong_last), ("summed-bias", wrong_sum)):
        rel = float((correct - wrong).flatten(1).norm(dim=1).max()
                    / correct.flatten(1).norm(dim=1).max())
        assert not torch.allclose(correct, wrong, atol=1e-3), \
            f"{name} composition matched sequential rollout — the bias term vanished"
        assert rel > 0.05, f"{name} relative difference only {rel:.4f}"

    # ...and the lambda product itself is a legitimate contraction, which is
    # exactly why the bug is silent.
    assert float(torch.sqrt(a ** 2 + b ** 2).max()) <= C.RHO + 1e-5


def test_bank_exposes_no_compose(bank):
    assert not hasattr(bank, "compose"), "there is NO compose(); see PLAN.md 4.B"


def test_rollout_depth_order_matters(bank):
    """Sequential composition is non-commutative; a reversed plan differs."""
    torch.manual_seed(11)
    z0 = torch.randn(2, C.K, C.D)
    c = S.sparse_simplex(2, 4, C.DEPTH)
    fwd = bank.rollout(c, z0)
    rev = bank.rollout(c.flip(2).contiguous(), z0)
    assert not torch.allclose(fwd, rev, atol=1e-3)


# ═══════════════════════════════════════════════════════════════════════════
#  4 · ROLLOUT PERF BUDGET                                              [gpu]
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.gpu
@pytest.mark.bench
def test_rollout_under_5ms_on_a100():
    """N=1000, DEPTH=4 under 5 ms. This is the planning inner loop.

    Also reports peak transient allocation. The rollout is inference-only
    (PLAN 5 runs it under the search), so nothing here is retained for backward;
    the analytic figure is ~0.9 GB of transient bf16 activations at N=1000.
    """
    pytest.importorskip("torch.cuda")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")

    torch.manual_seed(12)
    dev = torch.device("cuda")
    b = OperatorBank().to(dev).to(torch.bfloat16).eval()
    z0 = torch.randn(1, C.K, C.D, device=dev, dtype=torch.bfloat16)
    c = S.sparse_simplex(1, 1000, C.DEPTH, device=dev, dtype=torch.bfloat16)

    with torch.no_grad():
        for _ in range(5):                     # warm up kernels + autotune
            b.rollout(c, z0)
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        reps = 20
        t0 = time.perf_counter()
        for _ in range(reps):
            b.rollout(c, z0)
        torch.cuda.synchronize()
        ms = 1000.0 * (time.perf_counter() - t0) / reps
        peak_gb = (torch.cuda.max_memory_allocated() - base) / 1e9

    assert ms < 5.0, (
        f"rollout N=1000 DEPTH=4 took {ms:.3f} ms (budget 5 ms); "
        f"peak transient {peak_gb:.2f} GB"
    )
    assert peak_gb < 4.0, f"rollout peak transient {peak_gb:.2f} GB at N=1000"


# ═══════════════════════════════════════════════════════════════════════════
#  5 · ESTIMATOR THROUGHPUT BUDGET                                      [gpu]
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.gpu
@pytest.mark.bench
def test_estimator_at_least_30hz_with_7_streams():
    """batch 1, V=7, bf16. E runs once per executed segment (3.75 Hz), so 30 Hz
    is a 8x margin — but it is the number PLAN 4.B commits to."""
    pytest.importorskip("torch.cuda")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")

    torch.manual_seed(13)
    dev = torch.device("cuda")
    est = Estimator().to(dev).to(torch.bfloat16).eval()
    feats = S.make_obs_feats(b=1, v=7, p=196, f=1152, dof=7, l=16, device=dev)
    feats = {k: v.to(torch.bfloat16) for k, v in feats.items()}
    z_prev = torch.randn(1, C.K, C.D, device=dev, dtype=torch.bfloat16)

    with torch.no_grad():
        for _ in range(5):
            est(feats, z_prev)
        torch.cuda.synchronize()
        reps = 20
        t0 = time.perf_counter()
        for _ in range(reps):
            est(feats, z_prev)
        torch.cuda.synchronize()
        ms = 1000.0 * (time.perf_counter() - t0) / reps

    hz = 1000.0 / ms
    assert hz >= 30.0, f"estimator ran at {hz:.1f} Hz ({ms:.2f} ms), budget 30 Hz"


# ═══════════════════════════════════════════════════════════════════════════
#  6 · BF16, NO PROMOTION, NOTHING COMPLEX
# ═══════════════════════════════════════════════════════════════════════════

def test_bank_bf16_no_promotion(bank_bf16):
    """A100 has no FP8 and PyTorch has no complex-bf16. Four real ops in bf16."""
    torch.manual_seed(14)
    z = torch.randn(2, C.K, C.D, dtype=torch.bfloat16)
    c = S.sparse_simplex(2, dtype=torch.bfloat16)

    a, b = bank_bf16.mix(c)
    bi = bank_bf16.bias(c)
    out = bank_bf16.step(c, z)
    leaf = bank_bf16.rollout(S.sparse_simplex(2, 3, C.DEPTH, dtype=torch.bfloat16), z)

    for name, t in (("a", a), ("b", b), ("bias", bi), ("step", out), ("rollout", leaf)):
        assert t.dtype == torch.bfloat16, f"{name} promoted to {t.dtype}"
        assert not t.is_complex(), f"{name} is complex"
    C.assert_belief(out)
    C.assert_belief(leaf)


def test_bank_bf16_accepts_fp32_coefficients(bank_bf16):
    """Heads may emit fp32 `c` under autocast; the bank must not promote on it."""
    torch.manual_seed(15)
    z = torch.randn(2, C.K, C.D, dtype=torch.bfloat16)
    out = bank_bf16.step(S.sparse_simplex(2), z)
    assert out.dtype == torch.bfloat16


def test_estimator_bf16_no_promotion():
    est = _small_estimator().to(torch.bfloat16).eval()
    feats = {k: v.to(torch.bfloat16) for k, v in _small_feats(b=1).items()}
    z_prev = torch.randn(1, C.K, C.D, dtype=torch.bfloat16)
    with torch.no_grad():
        z = est(feats, z_prev)
    assert z.dtype == torch.bfloat16, f"estimator promoted to {z.dtype}"
    assert not z.is_complex()
    C.assert_belief(z)


def test_bank_bf16_bounds_still_hold(bank_bf16):
    """bf16 has 8 mantissa bits; the bound must survive the rounding."""
    torch.manual_seed(16)
    c = S.sparse_simplex(256, dtype=torch.bfloat16)
    with torch.no_grad():
        a, b = bank_bf16.mix(c)
        r = torch.sqrt(a.float() ** 2 + b.float() ** 2)
        nb = bank_bf16.bias(c).float().flatten(1).norm(dim=1)
    assert float(r.max()) <= C.RHO + 1e-2, f"bf16 spectral radius {float(r.max()):.5f}"
    assert float(nb.max()) <= C.B_MAX + 1e-2, f"bf16 bias norm {float(nb.max()):.5f}"


# ═══════════════════════════════════════════════════════════════════════════
#  7 · SHAPE / PROTOCOL CONFORMANCE
# ═══════════════════════════════════════════════════════════════════════════

def test_bank_satisfies_bank_protocol(bank):
    assert isinstance(bank, C.Bank)


def test_estimator_satisfies_estimator_protocol():
    assert isinstance(_small_estimator(), C.Estimator)


def test_bank_shapes(bank):
    torch.manual_seed(17)
    c = S.sparse_simplex(4)
    C.assert_simplex(c)
    a, b = bank.mix(c)
    assert a.shape == b.shape == (4, C.K, C.D // 2)
    assert bank.bias(c).shape == (4, C.K, C.D)
    out = bank.step(c, torch.randn(4, C.K, C.D))
    assert out.shape == (4, C.K, C.D)
    C.assert_belief(out)


def test_rollout_shape(bank):
    torch.manual_seed(18)
    out = bank.rollout(S.sparse_simplex(2, 7, C.DEPTH), torch.randn(2, C.K, C.D))
    assert out.shape == (2, 7, C.K, C.D)
    C.assert_belief(out)


def test_estimator_shapes():
    est = _small_estimator()
    with torch.no_grad():
        z = est(_small_feats(b=3), None)
    assert z.shape == (3, C.K, C.D)
    C.assert_belief(z)


def test_estimator_variable_stream_count():
    """V varies per embodiment: 2 for LIBERO, 7 for a bimanual rig."""
    est = _small_estimator()
    with torch.no_grad():
        for v in (1, 2, 7):
            z = est(_small_feats(b=1, v=v), None)
            C.assert_belief(z)
    with pytest.raises(ValueError):
        est(_small_feats(b=1, v=9), None)


@pytest.fixture
def second_body():
    """A synthetic 14-dof body, unregistered again on teardown.

    LIBERO is single-embodiment, so the dispatch has to be exercised with a
    second synthetic body or it is never tested at all (PLAN 4.A).
    """
    name = "loom_test_body14"
    C.register_embodiment(C.EmbodimentSpec(name, 14, 30.0, 3, (-1.0,) * 14, (1.0,) * 14))
    try:
        yield name
    finally:
        C.EMBODIMENTS.pop(name, None)


def test_estimator_per_embodiment_proprio_dispatch(second_body):
    """ModuleDict keyed by embodiment; batches are embodiment-homogeneous."""
    est = _small_estimator(embodiments=("libero_franka", second_body))
    assert set(est.proprio_proj) == {"libero_franka", second_body}
    with torch.no_grad():
        z7 = est(_small_feats(b=2, dof=7), None, embodiment="libero_franka")
        z14 = est(_small_feats(b=2, dof=14), None, embodiment=second_body)
        z7_inferred = est(_small_feats(b=2, dof=7), None)     # inferred from dof
    for z in (z7, z14, z7_inferred):
        C.assert_belief(z)
    with pytest.raises(KeyError):
        est(_small_feats(b=1, dof=7), None, embodiment="no_such_body")
    with pytest.raises(KeyError):
        est(_small_feats(b=1, dof=99), None)


def test_estimator_rejects_wrong_feat_dim():
    est = _small_estimator()
    with pytest.raises(ValueError):
        est(S.make_obs_feats(b=1, v=2, p=4, f=128, dof=7, l=4), None)


# ═══════════════════════════════════════════════════════════════════════════
#  8 · BROADCASTING
# ═══════════════════════════════════════════════════════════════════════════

def test_step_broadcasts_batch_only(bank):
    torch.manual_seed(19)
    z = torch.randn(4, C.K, C.D)
    out = bank.step(S.sparse_simplex(4), z)
    assert out.shape == z.shape


def test_step_broadcasts_over_candidates(bank):
    torch.manual_seed(20)
    z = torch.randn(3, 5, C.K, C.D)
    out = bank.step(S.sparse_simplex(3, 5), z)
    assert out.shape == z.shape


def test_step_candidate_axis_is_independent(bank):
    """`step` on (B, N, ...) must equal N separate (B, ...) calls."""
    torch.manual_seed(21)
    z = torch.randn(3, 5, C.K, C.D)
    c = S.sparse_simplex(3, 5)
    batched = bank.step(c, z)
    for n in range(5):
        one = bank.step(c[:, n], z[:, n])
        assert torch.allclose(batched[:, n], one, atol=1e-6)


def test_step_rejects_rank_mismatch(bank):
    """c (B, M) against z (B, N, K, D) would right-align and misalign the batch."""
    with pytest.raises(ValueError):
        bank.step(S.sparse_simplex(3), torch.randn(3, 5, C.K, C.D))


# ═══════════════════════════════════════════════════════════════════════════
#  9 · INITIALISATION SPREAD
# ═══════════════════════════════════════════════════════════════════════════

def test_init_has_a_real_spread_of_timescales(bank):
    """S4D-style init: log-uniform decay time constants across the D/2 axis.

    A depth-4 rollout is four applications, so a constant `r` gives the bank one
    timescale where it should have D/2. The 90th/10th percentile ratio of the
    per-channel time constant `tau = -1 / log r` is the sharpest test of that.
    """
    with torch.no_grad():
        r = C.RHO * torch.sigmoid(bank.log_r.double())      # (M, K, D//2)
        tau = -1.0 / torch.log(r)

    # spread specifically along the D/2 axis, averaged over operators and slots
    per_channel = tau.mean(dim=(0, 1))                      # (D//2,)
    p10, p90 = torch.quantile(per_channel, torch.tensor([0.1, 0.9], dtype=torch.float64))
    ratio = float(p90 / p10)
    assert ratio > 4.0, (
        f"per-channel time constants span only {ratio:.2f}x "
        f"(p10={float(p10):.2f}, p90={float(p90):.2f} operator steps) — "
        f"that is a constant-r init, not S4D"
    )

    # and the same spread survives elementwise, not just in the channel mean
    q = torch.quantile(tau.flatten()[::97], torch.tensor([0.1, 0.9], dtype=torch.float64))
    assert float(q[1] / q[0]) > 4.0


def test_init_r_is_near_rho_but_not_saturated(bank):
    """`r` must reach toward RHO without pinning sigmoid(log_r) at 1.

    If sigmoid(log_r) ~ 1 everywhere there is no spread and no gradient on the
    slow channels; if max r is far below RHO, a depth-4 rollout forgets z_0.
    """
    with torch.no_grad():
        sig = torch.sigmoid(bank.log_r.double())
        r = C.RHO * sig

    assert float(r.max()) <= C.RHO, "the bound is structural and must never be exceeded"
    assert float(r.max()) > 0.95, f"slowest channel only reaches r={float(r.max()):.4f}"
    assert float(sig.max()) < 0.999, \
        f"sigmoid(log_r) saturated at {float(sig.max()):.6f} — no gradient, no spread"
    assert float(sig.mean()) < 0.98, \
        f"mean sigmoid(log_r) = {float(sig.mean()):.4f}; r is effectively constant at RHO"
    assert float(r.min()) < 0.6, \
        f"fastest channel is r={float(r.min()):.4f}; no short-timescale modes"


def test_init_omega_is_spread_not_constant(bank):
    """S4D-Lin frequency ramp along the same axis, paired with the decay ramp."""
    with torch.no_grad():
        w = bank.omega.double().abs().mean(dim=(0, 1))      # (D//2,)
        # slow channels are low-frequency: a mode that survives 40 steps must not
        # oscillate at Nyquist, and a mode that dies in 1 step cannot resolve DC
        tau = (-1.0 / torch.log(C.RHO * torch.sigmoid(bank.log_r.double()))).mean(dim=(0, 1))
        corr = torch.corrcoef(torch.stack([tau, w]))[0, 1]
    assert float(w.max() - w.min()) > 1.0, "omega is effectively constant"
    assert float(corr) < -0.5, f"tau/omega pairing correlation {float(corr):.3f}"


def test_init_breaks_symmetry_between_operators(bank):
    """Without jitter every A_m starts identical and L_balance fights a tie."""
    with torch.no_grad():
        a, _ = bank.lam_bank()
        spread = a.std(dim=0).mean()
    assert float(spread) > 1e-3, f"operators are near-identical at init ({float(spread):.2e})"


def test_init_bias_is_below_the_cap(bank):
    """Below B_MAX so the bias magnitude is learnable, not only its direction."""
    with torch.no_grad():
        n = bank.b_raw.flatten(1).norm(dim=1)
    assert float(n.max()) < C.B_MAX, \
        f"bias initialised at the cap ({float(n.max()):.3f}); radial gradient is zero there"
    assert float(n.min()) > 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  10 · THE RECURRENCE
# ═══════════════════════════════════════════════════════════════════════════

def test_estimator_z_prev_changes_output():
    """If this passes trivially the recurrence has been silently dropped."""
    torch.manual_seed(22)
    est = _small_estimator().eval()
    feats = _small_feats(b=2)
    with torch.no_grad():
        z_none = est(feats, None)
        z_a = est(feats, torch.randn(2, C.K, C.D))
        z_b = est(feats, torch.randn(2, C.K, C.D))

    for name, other in (("z_prev=None vs z_prev=a", z_none), ("a vs b", z_b)):
        rel = float((z_a - other).norm() / z_a.norm())
        assert rel > 1e-2, f"{name}: output moved by only {rel:.2e} — z_prev is ignored"


def test_estimator_z_prev_receives_gradient():
    """Conditioning must be differentiable or L_dyn cannot train through it."""
    torch.manual_seed(23)
    est = _small_estimator()
    z_prev = torch.randn(1, C.K, C.D, requires_grad=True)
    est(_small_feats(b=1), z_prev).sum().backward()
    assert z_prev.grad is not None
    assert float(z_prev.grad.abs().max()) > 0.0


def test_estimator_observation_still_matters():
    """The dual guard: z_prev must not swamp the observation."""
    torch.manual_seed(24)
    est = _small_estimator().eval()
    z_prev = torch.randn(2, C.K, C.D)
    with torch.no_grad():
        z1 = est(_small_feats(b=2), z_prev)
        z2 = est(_small_feats(b=2), z_prev)
    assert float((z1 - z2).norm() / z1.norm()) > 1e-2


def test_estimator_grad_checkpointing_matches():
    """Team D needs activation checkpointing on the estimator blocks."""
    torch.manual_seed(25)
    plain = _small_estimator()
    torch.manual_seed(25)
    ckpt = _small_estimator(grad_checkpoint=True)
    ckpt.train()
    plain.train()

    feats = _small_feats(b=2)
    z_prev = torch.randn(2, C.K, C.D)

    plain(feats, z_prev).pow(2).sum().backward()
    ckpt(feats, z_prev).pow(2).sum().backward()

    num = den = 0.0
    for (n1, p1), (n2, p2) in zip(plain.named_parameters(), ckpt.named_parameters()):
        assert n1 == n2
        assert p1.grad is not None and p2.grad is not None, n1
        num += float((p1.grad - p2.grad).pow(2).sum())
        den += float(p1.grad.pow(2).sum())
    rel = math.sqrt(num / den)
    assert rel < 1e-5, f"checkpointed gradients differ by {rel:.2e} relative"


# ═══════════════════════════════════════════════════════════════════════════
#  11 · PARAMETER BUDGETS
# ═══════════════════════════════════════════════════════════════════════════

def test_bank_param_count(bank):
    """log_r (M,K,D/2) + omega (M,K,D/2) + b_raw (M,K,D) = 2*M*K*D = 25.2 M."""
    n = sum(p.numel() for p in bank.parameters())
    assert n == 2 * C.M * C.K * C.D
    assert 24e6 <= n <= 26e6, f"bank has {n / 1e6:.1f} M params, budget is 25 M"
    assert bank.log_r.shape == (C.M, C.K, C.D // 2)
    assert bank.omega.shape == (C.M, C.K, C.D // 2)
    assert bank.b_raw.shape == (C.M, C.K, C.D)


def test_estimator_param_count():
    """10 blocks at d=768 with mlp_ratio 4. If this reads 600 M, the FFNs grew."""
    est = Estimator()
    n = sum(p.numel() for p in est.parameters())
    assert 100e6 <= n <= 200e6, f"estimator has {n / 1e6:.1f} M params, budget is 150 M"


def test_estimator_feat_dim_is_configurable():
    """Team A has not finalised the tower; F must not be hard-coded."""
    est = Estimator(feat_dim=768, depth=1)
    assert est.view_proj.in_features == 768
    with torch.no_grad():
        C.assert_belief(est(S.make_obs_feats(b=1, v=2, p=4, f=768, dof=7, l=4), None))


# ═══════════════════════════════════════════════════════════════════════════
#  12 · SOURCE HYGIENE
# ═══════════════════════════════════════════════════════════════════════════

def _code_identifiers(path: Path) -> set[str]:
    """Every NAME token in a file — comments and docstrings excluded."""
    with open(path, "rb") as fh:
        src = fh.read().decode()
    names = set()
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.NAME:
            names.add(tok.string)
    return names


def test_no_view_as_complex_in_loom_model():
    """PLAN 9: there is no complex-bf16 dtype. Four real elementwise ops only.

    Tokenised rather than grepped so the ban can still be *documented* in the
    module docstrings without tripping its own test.
    """
    files = sorted(MODEL_DIR.glob("*.py"))
    assert files, f"no sources found under {MODEL_DIR}"
    banned = {"view_as_complex", "view_as_real", "complex64", "complex128", "cfloat"}
    for path in files:
        hits = _code_identifiers(path) & banned
        assert not hits, f"{path.name} uses {sorted(hits)}; z is real throughout"


def test_no_compose_in_loom_model():
    """PLAN 9: multi-step rollout is sequential affine. Never multiply lambdas."""
    for path in sorted(MODEL_DIR.glob("*.py")):
        assert "compose" not in _code_identifiers(path), \
            f"{path.name} defines or calls something named `compose`"


def test_rollout_source_is_a_sequential_loop():
    """A structural guard on the one loop the whole planner depends on."""
    src = (MODEL_DIR / "rollout.py").read_text()
    assert "for d in range(DEPTH)" in src
    assert "bank.step(" in src
