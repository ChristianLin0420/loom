"""
LOOM — Team E gate: proposal (`pi_c`), potential (`Phi`), shooting.

`pi_c` is the inference path (PLAN 4.E): without it there is no `c` at test
time and the model cannot be evaluated at all.  So the headline tests here are
the Plackett-Luce ones — brute-force enumeration at `M=6, k=2`, finite-difference
gradients, and the Gumbel-top-k / sequential-PL equivalence that lets us sample
`N=1000` candidates with one kernel instead of a `k`-step loop.

Everything runs on CPU and imports nothing but `contracts`, `stubs` and torch.
"""

from __future__ import annotations

import functools
import itertools
import math

import pytest
import torch
from torch import Tensor, nn

import contracts as C
import stubs as S
from loom.heads.potential import Potential
from loom.heads.proposal import (
    Proposal, canonical_order, gumbel_topk, pl_log_prob, weights_from_logits,
)
from loom.search.shooting import realizability_residual, shooting

TINY_M, TINY_K = 6, 2          # brute-force enumeration size


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def tiny_proposal(m: int = TINY_M, topk: int = TINY_K, seed: int = 0) -> Proposal:
    """A `Proposal` small enough to enumerate exhaustively, in float64.

    `m` / `topk` are constructor arguments precisely so this test can exist.
    """
    torch.manual_seed(seed)
    return Proposal(dim=8, lang_dim=8, m=m, topk=topk,
                    width=16, n_blocks=1, n_heads=2).double()


def tiny_inputs(b: int = 1, seed: int = 1) -> tuple[Tensor, Tensor]:
    torch.manual_seed(seed)
    return torch.randn(b, 5, 8, dtype=torch.float64), torch.randn(b, 3, 8, dtype=torch.float64)


def tiny_logits(p: Proposal, z: Tensor, lang: Tensor) -> Tensor:
    return p.logits(z, lang)[0].detach()


def small_full_proposal(seed: int = 0) -> Proposal:
    """Real `M`/`TOPK`/`D` so `contracts.assert_simplex` applies, but narrow."""
    torch.manual_seed(seed)
    return Proposal(dim=C.D, lang_dim=32, width=64, n_blocks=1, n_heads=4)


@functools.lru_cache(maxsize=1)
def full_proposal() -> Proposal:
    """The real ~50M head.  Built once: initialising 48M parameters is not free."""
    torch.manual_seed(0)
    return Proposal()


@functools.lru_cache(maxsize=1)
def stub_bank() -> S.StubBank:
    """`StubBank` holds 3 x (M, K, D)-ish parameter tensors; build it once."""
    torch.manual_seed(0)
    return S.StubBank()


def analytic_pl(logits: Tensor, order: tuple[int, ...]) -> float:
    """Reference PL probability of one ordered subset, written out longhand.

    Deliberately *not* vectorised and deliberately not sharing code with the
    implementation: this is the oracle.
    """
    p, remaining = 1.0, list(range(len(logits)))
    for i in order:
        denom = sum(math.exp(float(logits[j])) for j in remaining)
        p *= math.exp(float(logits[i])) / denom
        remaining.remove(i)
    return p


def scatter_c(logits: Tensor, support: tuple[int, ...]) -> Tensor:
    """`c = softmax(logits | support)` on the simplex, built by hand."""
    sel = torch.tensor([float(logits[i]) for i in support], dtype=logits.dtype)
    w = torch.softmax(sel, dim=0)
    c = torch.zeros(len(logits), dtype=logits.dtype)
    for j, i in enumerate(support):
        c[i] = w[j]
    return c


# ═══════════════════════════════════════════════════════════════════════════
#  1 · PLACKETT-LUCE vs BRUTE-FORCE ENUMERATION   (the headline test)
# ═══════════════════════════════════════════════════════════════════════════

def test_pl_ordered_subsets_sum_to_one():
    """All M!/(M-k)! ordered subsets are the atoms; they must sum to 1."""
    p = tiny_proposal()
    z, lang = tiny_inputs()
    logits = tiny_logits(p, z, lang)

    orders = list(itertools.permutations(range(TINY_M), TINY_K))
    assert len(orders) == 30

    total_analytic = sum(analytic_pl(logits, o) for o in orders)
    assert abs(total_analytic - 1.0) < 1e-10, total_analytic

    idx = torch.tensor(orders)                                   # (30, k)
    lp = pl_log_prob(logits.expand(len(orders), TINY_M), idx)    # (30,)
    assert abs(float(lp.exp().sum()) - 1.0) < 1e-10
    for o, val in zip(orders, lp.tolist()):
        assert abs(val - math.log(analytic_pl(logits, o))) < 1e-9, o


def test_log_prob_matches_brute_force_enumeration():
    """`log_prob(c)` == log PL(canonical order of c's support), for every support.

    The canonical order is descending weight.  Weights are `softmax(l | S)`,
    strictly increasing in the logit, so descending weight *is* descending
    logit — that is the whole correctness argument for order recovery.
    """
    p = tiny_proposal()
    z, lang = tiny_inputs()
    logits = tiny_logits(p, z, lang)

    supports = list(itertools.combinations(range(TINY_M), TINY_K))
    assert len(supports) == 15

    for sup in supports:
        c = scatter_c(logits, sup)
        assert abs(float(c.sum()) - 1.0) < 1e-12
        assert int((c > 0).sum()) == TINY_K

        # order recovery: descending weight
        recovered = tuple(canonical_order(c, TINY_K).tolist())
        expected = tuple(sorted(sup, key=lambda i: -float(logits[i])))
        assert recovered == expected, (sup, recovered, expected)

        got = float(p.log_prob(z, lang, c[None])[0].detach())
        want = math.log(analytic_pl(logits, expected))
        assert abs(got - want) < 1e-9, (sup, got, want)


def test_canonical_orderings_do_not_exhaust_the_mass():
    """Documents what `log_prob` scores: one ordered atom, not the whole support.

    Summing the canonical atom over every support gives < 1; adding the reverse
    ordering recovers exactly 1 at k=2.  `log_prob` is therefore an exact atom
    probability, not a set probability — which is what PPO/GRPO ratios need.
    """
    p = tiny_proposal()
    z, lang = tiny_inputs()
    logits = tiny_logits(p, z, lang)

    canonical, both = 0.0, 0.0
    for sup in itertools.combinations(range(TINY_M), TINY_K):
        desc = tuple(sorted(sup, key=lambda i: -float(logits[i])))
        canonical += analytic_pl(logits, desc)
        both += analytic_pl(logits, desc) + analytic_pl(logits, desc[::-1])
    assert canonical < 1.0
    assert abs(both - 1.0) < 1e-10


def test_log_prob_is_negative_and_finite_at_full_size():
    p = small_full_proposal()
    z, lang = torch.randn(3, C.K, C.D), torch.randn(3, 7, 32)
    lp = p.log_prob(z, lang, S.sparse_simplex(3))
    assert lp.shape == (3,)
    assert torch.isfinite(lp).all() and (lp <= 0).all()


def test_log_prob_broadcasts_over_candidates():
    """(B, n, M) -> (B, n): GRPO scores a whole candidate set at once."""
    p = small_full_proposal()
    z, lang = torch.randn(2, C.K, C.D), torch.randn(2, 7, 32)
    c = p.sample(z, lang, 5)
    lp = p.log_prob(z, lang, c)
    assert lp.shape == (2, 5)
    for b in range(2):
        for j in range(5):
            one = p.log_prob(z[b : b + 1], lang[b : b + 1], c[b, j][None])
            assert torch.allclose(lp[b, j], one[0], atol=1e-5)


def test_sample_and_log_prob_agree_on_the_support():
    """The sampler's weights must be exactly `softmax(l | S)`, or log_prob lies."""
    p = tiny_proposal()
    z, lang = tiny_inputs()
    logits = p.logits(z, lang)
    c = p.sample(z, lang, 32)                                     # (1, 32, m)
    idx = canonical_order(c, TINY_K)
    rebuilt = weights_from_logits(logits[:, None].expand_as(c), idx, TINY_M)
    assert torch.allclose(c, rebuilt, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════════
#  2 · GRADIENTS vs FINITE DIFFERENCES
# ═══════════════════════════════════════════════════════════════════════════

def test_log_prob_gradcheck_wrt_logits():
    """`torch.autograd.gradcheck` *is* a central-difference check, in float64."""
    torch.manual_seed(0)
    ell = torch.randn(3, TINY_M, dtype=torch.float64, requires_grad=True)
    order = torch.tensor([[0, 4], [2, 1], [5, 3]])
    assert torch.autograd.gradcheck(lambda x: pl_log_prob(x, order), (ell,),
                                    eps=1e-6, atol=1e-8, rtol=1e-5,
                                    check_batched_grad=False)


def test_log_prob_finite_differences_on_network_logits():
    """Explicit central differences on the real head's logits, float64."""
    p = tiny_proposal()
    z, lang = tiny_inputs()

    ell = p.logits(z, lang).detach()[0].requires_grad_(True)
    c = p.sample(z, lang, 1)[0, 0].detach()
    order = canonical_order(c, TINY_K)

    pl_log_prob(ell, order).backward()
    analytic = ell.grad.clone()

    eps = 1e-6
    for i in range(TINY_M):
        d = torch.zeros(TINY_M, dtype=torch.float64)
        d[i] = eps
        hi = float(pl_log_prob((ell.detach() + d), order))
        lo = float(pl_log_prob((ell.detach() - d), order))
        fd = (hi - lo) / (2 * eps)
        assert abs(fd - float(analytic[i])) < 1e-6, (i, fd, float(analytic[i]))


def test_gradients_reach_every_parameter():
    """BC (`-log pi_c(sg(c_a)|z,l)`) must train the whole head, not part of it."""
    p = small_full_proposal()
    z, lang = torch.randn(2, C.K, C.D), torch.randn(2, 7, 32)
    (-p.log_prob(z, lang, S.sparse_simplex(2)).mean()).backward()
    for name, prm in p.named_parameters():
        assert prm.grad is not None, name
        assert torch.isfinite(prm.grad).all(), name
    assert any(float(prm.grad.abs().sum()) > 0 for prm in p.parameters())


# ═══════════════════════════════════════════════════════════════════════════
#  3 · GUMBEL TOP-k  ==  SEQUENTIAL PL WITHOUT REPLACEMENT
# ═══════════════════════════════════════════════════════════════════════════

REF_LOGITS = torch.tensor([0.9, -0.4, 0.2, 1.3, -1.1, 0.5])


def _ordered_pair_histogram(idx: Tensor, m: int) -> Tensor:
    code = idx[:, 0] * m + idx[:, 1]
    return torch.bincount(code, minlength=m * m).double()


def _sequential_pl(logits: Tensor, k: int, n: int, gen: torch.Generator) -> Tensor:
    """Reference sampler: k masked categorical draws, one at a time."""
    live = logits.expand(n, -1).clone()
    out = []
    for _ in range(k):
        i = torch.multinomial(torch.softmax(live, dim=-1), 1, generator=gen)
        out.append(i)
        live = live.scatter(-1, i, -1e30)
    return torch.cat(out, dim=-1)


def test_gumbel_topk_matches_analytic_pl():
    """Gumbel top-k frequencies match the analytic PL law within MC error."""
    m, k, n = TINY_M, TINY_K, 200_000
    gen = torch.Generator().manual_seed(1234)
    idx = gumbel_topk(REF_LOGITS.expand(n, m), k, generator=gen)
    counts = _ordered_pair_histogram(idx, m)

    expected = torch.zeros(m * m, dtype=torch.float64)
    for i, j in itertools.permutations(range(m), k):
        expected[i * m + j] = analytic_pl(REF_LOGITS, (i, j)) * n
    assert abs(float(expected.sum()) - n) < 1e-6

    live = expected > 0
    assert float(counts[~live].sum()) == 0.0, "sampled a repeated index"

    tv = 0.5 * float((counts[live] - expected[live]).abs().sum()) / n
    chi2 = float(((counts[live] - expected[live]) ** 2 / expected[live]).sum())
    df = int(live.sum()) - 1
    assert df == 29
    print(f"\n[gumbel-vs-PL] N={n}  TV={tv:.5f}  chi2={chi2:.2f} (df={df})")
    assert tv < 0.01, tv
    assert chi2 < 3.0 * df, chi2


def test_gumbel_topk_matches_sequential_sampler():
    """Two samplers, one law: TV between the empirical distributions is MC-small."""
    m, k, n = TINY_M, TINY_K, 200_000
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(8)
    a = _ordered_pair_histogram(gumbel_topk(REF_LOGITS.expand(n, m), k, generator=g1), m)
    b = _ordered_pair_histogram(_sequential_pl(REF_LOGITS, k, n, g2), m)
    tv = 0.5 * float((a - b).abs().sum()) / n
    print(f"[gumbel-vs-sequential] N={n}  TV={tv:.5f}")
    assert tv < 0.01, tv


def test_sample_support_frequencies_match_pl():
    """End-to-end through the network: `sample` realises the same law.

    `c` loses the sampled order (the weights are order-invariant), so this
    compares *support* frequencies against the order-marginalised PL law.
    """
    m, k, n = TINY_M, TINY_K, 100_000
    p = tiny_proposal()
    z, lang = tiny_inputs()
    logits = tiny_logits(p, z, lang)

    torch.manual_seed(99)
    c = p.sample(z, lang, n)[0]
    sup = canonical_order(c, k).sort(dim=-1).values
    code = sup[:, 0] * m + sup[:, 1]
    counts = torch.bincount(code, minlength=m * m).double()

    expected = torch.zeros(m * m, dtype=torch.float64)
    for i, j in itertools.combinations(range(m), k):
        expected[i * m + j] = (analytic_pl(logits, (i, j))
                               + analytic_pl(logits, (j, i))) * n
    tv = 0.5 * float((counts - expected).abs().sum()) / n
    print(f"[sample-support-vs-PL] N={n}  TV={tv:.5f}")
    assert tv < 0.02, tv


# ═══════════════════════════════════════════════════════════════════════════
#  4 · SAMPLE / SAMPLE_SEQ SHAPES AND SIMPLEX
# ═══════════════════════════════════════════════════════════════════════════

def test_sample_shape_and_simplex():
    p = small_full_proposal()
    z, lang = torch.randn(2, C.K, C.D), torch.randn(2, 7, 32)
    c = p.sample(z, lang, 64)
    assert c.shape == (2, 64, C.M)
    C.assert_simplex(c)
    assert (c != 0).sum(-1).eq(C.TOPK).all(), "must have exactly TOPK nonzeros"
    assert not c.is_complex()


def test_sample_is_efficient_at_n_1000():
    """`N = 1000` is the planning budget; the network must run once, not 1000x."""
    p = small_full_proposal()
    z, lang = torch.randn(2, C.K, C.D), torch.randn(2, 7, 32)
    c = p.sample(z, lang, 1000)
    assert c.shape == (2, 1000, C.M)
    C.assert_simplex(c)


def test_sample_seq_shape_and_per_depth_simplex():
    p = small_full_proposal()
    z, lang = torch.randn(2, C.K, C.D), torch.randn(2, 7, 32)
    cs = p.sample_seq(z, lang, 16, C.DEPTH)
    assert cs.shape == (2, 16, C.DEPTH, C.M)
    C.assert_simplex(cs)
    for d in range(C.DEPTH):
        C.assert_simplex(cs[:, :, d])
        assert (cs[:, :, d] != 0).sum(-1).eq(C.TOPK).all()


def test_sample_seq_feeds_bank_rollout():
    """(B,N,DEPTH,M) is exactly what `Bank.rollout` consumes."""
    p, bank = small_full_proposal(), stub_bank()
    z, lang = torch.randn(2, C.K, C.D), torch.randn(2, 7, 32)
    leaf = bank.rollout(p.sample_seq(z, lang, 5, C.DEPTH), z)
    assert leaf.shape == (2, 5, C.K, C.D)
    C.assert_belief(leaf)


def test_sample_is_stochastic_and_seedable():
    p = small_full_proposal()
    z, lang = torch.randn(2, C.K, C.D), torch.randn(2, 7, 32)
    g1 = torch.Generator().manual_seed(3)
    g2 = torch.Generator().manual_seed(3)
    a = p.sample(z, lang, 32, generator=g1)
    b = p.sample(z, lang, 32, generator=g2)
    assert torch.equal(a, b), "same seed must reproduce the candidate set"
    g3 = torch.Generator().manual_seed(4)
    assert not torch.equal(a, p.sample(z, lang, 32, generator=g3))


def test_sample_rejects_wrong_belief_rank():
    p = small_full_proposal()
    with pytest.raises(ValueError):
        p.sample(torch.randn(2, 3, C.K, C.D), torch.randn(2, 7, 32), 4)


# ═══════════════════════════════════════════════════════════════════════════
#  5 · ARGMAX — the R0 inference path (no search)
# ═══════════════════════════════════════════════════════════════════════════

def test_argmax_is_deterministic_and_on_simplex():
    p = small_full_proposal()
    z, lang = torch.randn(3, C.K, C.D), torch.randn(3, 7, 32)
    a, b = p.argmax(z, lang), p.argmax(z, lang)
    assert torch.equal(a, b)
    assert a.shape == (3, C.M)
    C.assert_simplex(a)
    assert (a != 0).sum(-1).eq(C.TOPK).all()


def test_argmax_selects_the_largest_logits():
    p = small_full_proposal()
    z, lang = torch.randn(3, C.K, C.D), torch.randn(3, 7, 32)
    logits = p.logits(z, lang)
    a = p.argmax(z, lang)
    want = logits.topk(C.TOPK, dim=-1).indices.sort(-1).values
    got = a.topk(C.TOPK, dim=-1).indices.sort(-1).values
    assert torch.equal(want, got)


def test_argmax_is_the_mode_of_the_support_distribution():
    """Greedy sequential argmax == unperturbed top-k; check by enumeration."""
    p = tiny_proposal()
    z, lang = tiny_inputs()
    logits = tiny_logits(p, z, lang)
    best = max(itertools.permutations(range(TINY_M), TINY_K),
               key=lambda o: analytic_pl(logits, o))
    got = tuple(canonical_order(p.argmax(z, lang)[0], TINY_K).tolist())
    assert got == best


# ═══════════════════════════════════════════════════════════════════════════
#  6 · SHOOTING
# ═══════════════════════════════════════════════════════════════════════════

class FixedProposal(nn.Module):
    """A known candidate set, so the planner's selection is checkable."""

    def __init__(self, c_seq: Tensor) -> None:
        super().__init__()
        self.register_buffer("c_seq", c_seq)

    def sample_seq(self, z, lang, n, depth=C.DEPTH, generator=None):
        assert n == self.c_seq.shape[1] and depth == self.c_seq.shape[2]
        return self.c_seq.to(z.dtype)

    def sample(self, z, lang, n, generator=None):
        return self.sample_seq(z, lang, n)[:, :, 0]

    def log_prob(self, z, lang, c):
        return torch.zeros(c.shape[:-1], device=c.device, dtype=c.dtype)


class PassBank(nn.Module):
    """`Bank.rollout` that broadcasts `z` unchanged.

    Used by the tests that exercise *ranking* — scoring order, the gate
    fall-through — where the dynamics are irrelevant and `StubBank`'s
    (M, K, D) einsums are pure cost.  `StubBank` is used wherever the affine
    rollout is actually part of the assertion.
    """

    def rollout(self, c_seq: Tensor, z: Tensor) -> Tensor:
        b, n = c_seq.shape[0], c_seq.shape[1]
        return z.unsqueeze(1).expand(b, n, C.K, C.D)


class TargetPotential(nn.Module):
    """Rigged `Phi`: maximal (zero) at a designated leaf, negative elsewhere."""

    def __init__(self, target_leaf: Tensor) -> None:
        super().__init__()
        self.register_buffer("target", target_leaf)              # (B, K, D)

    def forward(self, z: Tensor, lang: Tensor) -> Tensor:
        return -(z.float() - self.target.float().unsqueeze(1)).flatten(2).norm(dim=-1)


class EchoDecoder(nn.Module):
    """Records the `c` it was asked to realize so the rigged q_a can see it."""

    def __init__(self, dof: int = 7) -> None:
        super().__init__()
        self.dof, self.last_c = dof, None

    def forward(self, z: Tensor, c: Tensor) -> Tensor:
        self.last_c = c
        return torch.zeros(z.shape[0], C.H_OP, self.dof, device=z.device, dtype=z.dtype)

    def loss(self, z, c, a_seg):
        return (self.forward(z, c) - a_seg).pow(2).mean()


class RiggedQAction(nn.Module):
    """Returns `c` unchanged (residual 0) except on a blacklist (residual > tau).

    Blacklisting by exact `c` match is safe here because `FixedProposal` hands
    the planner a fixed candidate tensor.
    """

    def __init__(self, dec: EchoDecoder, reject: list[Tensor]) -> None:
        super().__init__()
        self.dec = dec
        self.reject = reject

    def forward(self, a_seg: Tensor, z: Tensor) -> Tensor:
        c = self.dec.last_c
        far = c.roll(C.M // 2, dims=-1)          # disjoint support => ||far - c|| > tau
        hit = torch.zeros(c.shape[0], 1, dtype=torch.bool, device=c.device)
        for bad in self.reject:
            hit = hit | (c == bad.to(c)).all(-1, keepdim=True)
        return torch.where(hit, far, c)


def _fixed_candidates(b: int, n: int, seed: int = 5) -> Tensor:
    torch.manual_seed(seed)
    return S.sparse_simplex(b, n, C.DEPTH)


def test_shooting_returns_root_segment_only():
    b, n = 2, 12
    bank, phi = stub_bank(), Potential(lang_dim=32)
    z, lang = torch.randn(b, C.K, C.D), torch.randn(b, 7, 32)
    c_seq = _fixed_candidates(b, n)
    prop = FixedProposal(c_seq)

    c_root, info = shooting(prop, bank, phi, z, lang, n=n)

    assert c_root.shape == (b, C.M), "root segment only: one operator, not DEPTH"
    C.assert_simplex(c_root)
    chosen = info["index"]
    assert torch.equal(c_root, c_seq[torch.arange(b), chosen, 0])
    assert info["plan"].shape == (b, C.DEPTH, C.M)
    # the tail exists but is discarded; the returned c is genuinely depth 0
    assert not torch.equal(c_root, info["plan"][:, 1])


def test_shooting_selects_the_leaf_maximising_the_potential():
    """Rig `Phi` so a known candidate's leaf is the unique maximiser."""
    b, n, want = 2, 8, 5
    bank = stub_bank()
    z, lang = torch.randn(b, C.K, C.D), torch.randn(b, 7, 32)
    c_seq = _fixed_candidates(b, n)
    target_leaf = bank.rollout(c_seq[:, want : want + 1], z).squeeze(1)
    phi = TargetPotential(target_leaf)

    c_root, info = shooting(FixedProposal(c_seq), bank, phi, z, lang, n=n)

    assert torch.equal(info["index"], torch.full((b,), want))
    assert torch.allclose(info["chosen_score"], torch.zeros(b), atol=1e-4)
    assert torch.equal(c_root, c_seq[:, want, 0])
    assert info["scores"].shape == (b, n)
    assert not info["gate_applied"]


def test_shooting_score_is_the_leaf_potential_alone():
    """No root term, no cost term: `info['scores'] == Phi(bank.rollout(C, z), l)`."""
    b, n = 2, 6
    bank, phi = stub_bank(), Potential(lang_dim=32)
    z, lang = torch.randn(b, C.K, C.D), torch.randn(b, 7, 32)
    c_seq = _fixed_candidates(b, n)
    _, info = shooting(FixedProposal(c_seq), bank, phi, z, lang, n=n)
    with torch.no_grad():
        want = phi(bank.rollout(c_seq, z), lang)
    assert torch.allclose(info["scores"], want, atol=1e-5)


def test_shooting_runs_against_the_real_proposal():
    """Integration shape check on the intended (proposal, bank, potential) triple."""
    b, n = 2, 32
    prop, bank, phi = small_full_proposal(), PassBank(), Potential(lang_dim=32)
    z, lang = torch.randn(b, C.K, C.D), torch.randn(b, 7, 32)
    c_root, info = shooting(prop, bank, phi, z, lang, n=n)
    assert c_root.shape == (b, C.M)
    C.assert_simplex(c_root)
    assert info["scores"].shape == (b, n)
    assert info["n_candidates"] == n and info["depth"] == C.DEPTH


# ── realizability gate ───────────────────────────────────────────────────

def test_realizability_residual_is_zero_for_a_perfect_pair():
    dec = EchoDecoder()
    qa = RiggedQAction(dec, reject=[])
    z, c = torch.randn(2, C.K, C.D), S.sparse_simplex(2)
    assert float(realizability_residual(z, c, qa, dec).max()) < 1e-6


def test_gate_falls_through_to_the_runner_up():
    b, n = 1, 10
    bank = PassBank()
    z, lang = torch.randn(b, C.K, C.D), torch.randn(b, 7, 32)
    c_seq = _fixed_candidates(b, n, seed=11)
    scores = torch.tensor([[float(n - i) for i in range(n)]])     # rank == index
    phi = _ScorePotential(scores)

    top, runner_up = c_seq[:, 0, 0], c_seq[:, 1, 0]
    dec = EchoDecoder()
    qa = RiggedQAction(dec, reject=[top[0]])

    c_root, info = shooting(FixedProposal(c_seq), bank, phi, z, lang,
                            n=n, q_action=qa, decoder=dec)

    assert torch.equal(c_root, runner_up), "must fall through to the runner-up"
    assert int(info["index"]) == 1 and int(info["rank"]) == 1
    assert int(info["n_rejected"]) == 1
    assert not bool(info["gate_exhausted"])
    assert info["gate_applied"]


def test_gate_walks_the_ranking_not_a_single_retry():
    """Three consecutive rejections must land on the fourth-best, not give up."""
    b, n = 1, 10
    bank = PassBank()
    z, lang = torch.randn(b, C.K, C.D), torch.randn(b, 7, 32)
    c_seq = _fixed_candidates(b, n, seed=13)
    phi = _ScorePotential(torch.tensor([[float(n - i) for i in range(n)]]))

    dec = EchoDecoder()
    qa = RiggedQAction(dec, reject=[c_seq[0, i, 0] for i in range(3)])

    c_root, info = shooting(FixedProposal(c_seq), bank, phi, z, lang,
                            n=n, q_action=qa, decoder=dec)

    assert int(info["rank"]) == 3 and int(info["n_rejected"]) == 3
    assert torch.equal(c_root, c_seq[:, 3, 0])
    assert not bool(info["gate_exhausted"])


def test_gate_never_returns_nothing():
    """If every candidate fails, return the best-scoring one and flag it."""
    b, n = 1, 6
    bank = PassBank()
    z, lang = torch.randn(b, C.K, C.D), torch.randn(b, 7, 32)
    c_seq = _fixed_candidates(b, n, seed=17)
    phi = _ScorePotential(torch.tensor([[float(n - i) for i in range(n)]]))

    dec = EchoDecoder()
    qa = RiggedQAction(dec, reject=[c_seq[0, i, 0] for i in range(n)])

    c_root, info = shooting(FixedProposal(c_seq), bank, phi, z, lang,
                            n=n, q_action=qa, decoder=dec)

    assert bool(info["gate_exhausted"])
    assert int(info["rank"]) == 0 and int(info["n_rejected"]) == n
    assert torch.equal(c_root, c_seq[:, 0, 0])
    C.assert_simplex(c_root)


def test_gate_is_per_batch_element():
    b, n = 2, 8
    bank = PassBank()
    z, lang = torch.randn(b, C.K, C.D), torch.randn(b, 7, 32)
    c_seq = _fixed_candidates(b, n, seed=19)
    phi = _ScorePotential(torch.tensor([[float(n - i) for i in range(n)]] * b))

    dec = EchoDecoder()
    qa = RiggedQAction(dec, reject=[c_seq[0, 0, 0]])          # only element 0's top
    _, info = shooting(FixedProposal(c_seq), bank, phi, z, lang,
                       n=n, q_action=qa, decoder=dec)
    assert info["rank"].tolist() == [1, 0]
    assert info["n_rejected"].tolist() == [1, 0]


def test_gate_requires_both_heads():
    b, n = 1, 4
    bank, phi = PassBank(), Potential(lang_dim=32)
    z, lang = torch.randn(b, C.K, C.D), torch.randn(b, 7, 32)
    with pytest.raises(ValueError):
        shooting(FixedProposal(_fixed_candidates(b, n)), bank, phi, z, lang,
                 n=n, decoder=EchoDecoder())


def test_gate_runs_against_the_stub_heads():
    """Smoke test with `stubs.StubQAction` / `StubDecoder`: random c => rejected."""
    b, n = 2, 8
    bank, phi = PassBank(), Potential(lang_dim=32)
    z, lang = torch.randn(b, C.K, C.D), torch.randn(b, 7, 32)
    c_root, info = shooting(FixedProposal(_fixed_candidates(b, n)), bank, phi, z, lang,
                            n=n, q_action=S.StubQAction(), decoder=S.StubDecoder(),
                            max_gate_evals=3)
    assert c_root.shape == (b, C.M)
    C.assert_simplex(c_root)
    assert info["n_rejected"].sum() > 0     # unrelated random c: gate rejects


class _ScorePotential(nn.Module):
    """`Phi` that ignores the leaf and returns a prescribed (B, N) score."""

    def __init__(self, scores: Tensor) -> None:
        super().__init__()
        self.register_buffer("s", scores)

    def forward(self, z: Tensor, lang: Tensor) -> Tensor:
        return self.s.to(z.dtype).expand(z.shape[0], z.shape[1])


# ═══════════════════════════════════════════════════════════════════════════
#  7 · PARAMETER BUDGET   (PLAN 2: pi_c 50M, Phi 0.2M)
# ═══════════════════════════════════════════════════════════════════════════

def test_proposal_param_count_is_about_50m():
    n = full_proposal().n_params()
    print(f"\n[params] proposal = {n/1e6:.2f}M")
    assert 30e6 <= n <= 70e6, f"{n/1e6:.1f}M is outside 50M +/- 40%"


def test_potential_is_genuinely_tiny():
    """It runs N=1000 times per cycle; 0.2M is the budget, 0.5M the hard cap."""
    n = Potential().n_params()
    print(f"[params] potential = {n/1e6:.3f}M")
    assert n <= 0.5e6, f"{n/1e6:.3f}M exceeds the 0.5M cap"
    assert n >= 50e3, "suspiciously small; check the constructor"


def test_potential_is_far_smaller_than_the_proposal():
    assert Potential().n_params() * 20 < full_proposal().n_params()


# ═══════════════════════════════════════════════════════════════════════════
#  8 · POTENTIAL SHAPES
# ═══════════════════════════════════════════════════════════════════════════

def test_potential_shapes_match_the_contract():
    phi = Potential(lang_dim=32)
    lang = torch.randn(2, 7, 32)
    assert phi(torch.randn(2, C.K, C.D), lang).shape == (2,)
    assert phi(torch.randn(2, 9, C.K, C.D), lang).shape == (2, 9)


def test_potential_accepts_pooled_language():
    phi = Potential(lang_dim=32)
    assert phi(torch.randn(2, 9, C.K, C.D), torch.randn(2, 32)).shape == (2, 9)


def test_potential_is_permutation_dependent_on_language():
    """Different instructions must give different potentials, or search is blind."""
    phi = Potential(lang_dim=32)
    z = torch.randn(4, 6, C.K, C.D)
    a = phi(z, torch.randn(4, 7, 32))
    b = phi(z, torch.randn(4, 7, 32))
    assert not torch.allclose(a, b)


def test_potential_discriminates_between_candidate_leaves():
    """`Phi` must vary with the leaf, or `score.argmax(1)` is a coin flip.

    Guards the pooling: `LayerNorm` *then* mean over `K=128` slots shrinks the
    belief signal by ~sqrt(K) and drowns it under the language term.
    """
    phi = Potential(lang_dim=32)
    with torch.no_grad():
        std = float(phi(torch.randn(2, 64, C.K, C.D), torch.randn(2, 7, 32)).std())
    assert std > 1e-3, f"Phi is nearly constant across leaves: std={std:.2e}"


def test_potential_broadcasts_language_across_candidates():
    """Candidate n of batch b must see batch b's instruction, not batch 0's."""
    phi = Potential(lang_dim=32)
    z, lang = torch.randn(3, 5, C.K, C.D), torch.randn(3, 7, 32)
    full = phi(z, lang)
    for b in range(3):
        one = phi(z[b : b + 1], lang[b : b + 1])
        assert torch.allclose(full[b], one[0], atol=1e-5)


def test_potential_rejects_wrong_slot_width():
    with pytest.raises(ValueError):
        Potential(lang_dim=32)(torch.randn(2, C.K, C.D + 1), torch.randn(2, 7, 32))


# ═══════════════════════════════════════════════════════════════════════════
#  9 · BF16 — no promotion, no complex   (PLAN 9)
# ═══════════════════════════════════════════════════════════════════════════

def test_proposal_runs_in_bf16_without_promotion():
    p = small_full_proposal().to(torch.bfloat16)
    z = torch.randn(2, C.K, C.D, dtype=torch.bfloat16)
    lang = torch.randn(2, 7, 32, dtype=torch.bfloat16)
    for out in (p.logits(z, lang), p.sample(z, lang, 8), p.argmax(z, lang),
                p.sample_seq(z, lang, 4, C.DEPTH)):
        assert out.dtype == torch.bfloat16 and not out.is_complex()
    C.assert_simplex(p.sample(z, lang, 8))
    lp = p.log_prob(z, lang, p.argmax(z, lang))
    assert lp.dtype == torch.bfloat16 and torch.isfinite(lp.float()).all()


def test_potential_runs_in_bf16_without_promotion():
    phi = Potential(lang_dim=32).to(torch.bfloat16)
    out = phi(torch.randn(2, 5, C.K, C.D, dtype=torch.bfloat16),
              torch.randn(2, 7, 32, dtype=torch.bfloat16))
    assert out.dtype == torch.bfloat16 and not out.is_complex()
    assert out.shape == (2, 5)


def test_shooting_runs_in_bf16_without_promotion():
    b, n = 2, 8
    prop = small_full_proposal().to(torch.bfloat16)
    bank = PassBank()
    phi = Potential(lang_dim=32).to(torch.bfloat16)
    z = torch.randn(b, C.K, C.D, dtype=torch.bfloat16)
    lang = torch.randn(b, 7, 32, dtype=torch.bfloat16)
    c_root, info = shooting(prop, bank, phi, z, lang, n=n)
    assert c_root.dtype == torch.bfloat16 and not c_root.is_complex()
    C.assert_simplex(c_root)
    assert info["scores"].dtype == torch.bfloat16


# ═══════════════════════════════════════════════════════════════════════════
#  10 · PROTOCOL CONFORMANCE
# ═══════════════════════════════════════════════════════════════════════════

def test_proposal_satisfies_the_contract():
    assert isinstance(small_full_proposal(), C.Proposal)


def test_potential_satisfies_the_contract():
    assert isinstance(Potential(), C.Potential)


def test_proposal_is_drop_in_for_the_stub():
    """Anything written against `stubs.StubProposal` must accept the real head."""
    real, stub = small_full_proposal(), S.StubProposal()
    z, lang = torch.randn(2, C.K, C.D), torch.randn(2, 7, 32)
    for p in (real, stub):
        assert p.sample(z, lang, 4).shape == (2, 4, C.M)
        assert p.log_prob(z, lang, S.sparse_simplex(2)).shape == (2,)
