"""
LOOM — `pi_c`, the proposal head.  Team E.

**This is the inference path.**  At test time there is no ground-truth action
and no future observation, so neither `q_a` nor `q_Delta` can produce a `c`.
Without a BC-trained `pi_c` the model cannot be evaluated at all (PLAN 4.E).
R0 uses `c <- argmax pi_c(.|z, l)` with no search; R3 adds shooting on top of
the same head.

Distribution — Plackett-Luce over sampling *without replacement*
================================================================

    network  ->  logits  l in R^M                     conditioned on (z, lang)
    sample   k = TOPK indices S = (i_1 ... i_k) sequentially WITHOUT replacement
    weights  deterministic given S:   c_{i_j} = softmax(l restricted to S)_{i_j}

    log pi(S) = sum_{j=1..k} [ l_{i_j} - logsumexp_{m not in {i_1..i_{j-1}}} l_m ]

**The stochastic variable is the ordered subset.**  The weights carry no
independent randomness; they are a deterministic function of the support.  That
is exactly why the support carries the whole probability mass, and it is what
gives PPO/GRPO in R3 a real, exact log-probability instead of a surrogate.

Sampling: Gumbel top-k
======================

Adding i.i.d. Gumbel(0, 1) noise to every logit and taking the top-`k` in
descending perturbed order yields an ordered subset distributed *exactly* as
sequential PL sampling without replacement (Yellott 1977; Vieira 2014).  Sketch:
`argmax_m (l_m + G_m)` is Categorical(softmax(l)) by the Gumbel-max lemma, and
the Gumbel-max argmax is independent of the max, so conditioning on the winner
and repeating over the remaining items reproduces the sequential draw.  One
`topk` kernel replaces a `k`-step loop, which matters at `N = 1000` candidates
x `DEPTH = 4` per planning cycle.  `tests/test_search.py` proves the
equivalence empirically.

Recovering the order in `log_prob`
==================================

`log_prob` receives `c`, not `S`, and PL is a distribution over *ordered*
tuples — so the order has to be recovered.  It is recoverable because the
weights are `softmax(l | S)`, which is *strictly increasing* in the logit:

    c_i > c_j   <=>   l_i > l_j        for i, j in S.

So sorting the support by descending weight *is* sorting it by descending
logit, and we take that descending-logit ordering as the canonical PL order of
the support.  This is the subtle correctness point of the whole file: `c`
determines the support and the descending-logit permutation of it, and nothing
else, so `log_prob(c)` is the well-defined log-probability of that canonical
ordered tuple.  (The sequential sampler may of course have drawn the same
support in a different order; those are distinct atoms of the same
distribution.  We score the canonical one, consistently, in both BC and RL, so
sampler and scorer agree on the atom and importance ratios stay exact.)

Everything is real and bf16-safe: no complex dtype anywhere (PLAN 9).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from contracts import D, DEPTH, M, TOPK

__all__ = [
    "Proposal",
    "pl_log_prob",
    "canonical_order",
    "weights_from_logits",
    "gumbel_topk",
]


# ═══════════════════════════════════════════════════════════════════════════
#  PLACKETT-LUCE PRIMITIVES  (module-level: tests and losses/proposal_bc.py
#  need the math without instantiating a 50M-param network)
# ═══════════════════════════════════════════════════════════════════════════

def pl_log_prob(logits: Tensor, order: Tensor) -> Tensor:
    """Exact PL log-probability of the ordered subset `order`.

    logits (..., M), order (..., k) with distinct indices -> (...)

        log pi(S) = sum_j [ l_{i_j} - logsumexp_{m not in {i_1..i_{j-1}}} l_m ]

    Differentiable w.r.t. `logits`.  The `k`-step loop is over `TOPK = 4`, so it
    is four fused kernels, not a bottleneck.
    """
    k = order.shape[-1]
    mask = torch.zeros_like(logits, dtype=torch.bool)
    total = logits.new_zeros(logits.shape[:-1])
    neg_inf = torch.finfo(logits.dtype).min
    for j in range(k):
        idx = order[..., j : j + 1]                          # (..., 1)
        avail = logits.masked_fill(mask, neg_inf)
        total = total + avail.gather(-1, idx).squeeze(-1) - avail.logsumexp(-1)
        mask = mask.scatter(-1, idx, True)
    return total


def canonical_order(c: Tensor, topk: int = TOPK) -> Tensor:
    """Recover the canonical PL order of `c`'s support: descending weight.

    c (..., M) -> (..., topk) long.  See the module docstring: weights are
    `softmax(l | S)`, monotone in the logit, so descending weight *is*
    descending logit.
    """
    return c.topk(topk, dim=-1).indices


def weights_from_logits(logits: Tensor, idx: Tensor, m: int = M) -> Tensor:
    """Scatter `softmax(logits restricted to idx)` onto the M-simplex.

    logits (..., M), idx (..., k) -> (..., M) with exactly k nonzeros summing
    to 1.  This is the *deterministic* half of the distribution.

    The softmax runs at fp32 or better and is cast back, so a bf16 head still
    lands on the simplex to fp32 accuracy; `assert_simplex` is unforgiving.
    """
    acc = torch.promote_types(logits.dtype, torch.float32)
    sel = logits.gather(-1, idx)
    w = torch.softmax(sel.to(acc), dim=-1).to(logits.dtype)
    c = torch.zeros(*idx.shape[:-1], m, device=logits.device, dtype=logits.dtype)
    return c.scatter(-1, idx, w)


def gumbel_topk(
    logits: Tensor,
    k: int,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Ordered subset of size `k` sampled without replacement, PL-distributed.

    logits (..., M) -> (..., k) long, in descending perturbed-logit order.

    The noise and the comparison run at fp32 or better regardless of the head's
    dtype, for two reasons: a uniform drawn directly in bf16 hits exact 0 often
    enough to produce `inf`, and bf16's 8 mantissa bits would quantise the
    Gumbels into ties that `topk` then breaks by index — a systematic bias
    towards low-numbered operators.  The output is integer indices, so nothing
    is promoted downstream and the weights stay in the head's dtype.
    """
    acc = torch.promote_types(logits.dtype, torch.float32)
    g = -torch.empty(logits.shape, device=logits.device, dtype=acc) \
        .exponential_(generator=generator).clamp_min(torch.finfo(acc).tiny).log()
    return (logits.to(acc) + g).topk(k, dim=-1).indices


# ═══════════════════════════════════════════════════════════════════════════
#  NETWORK
# ═══════════════════════════════════════════════════════════════════════════

class _Attention(nn.Module):
    """Multi-head attention with separate q / kv projections (cross or self)."""

    def __init__(self, width: int, n_heads: int, kv_dim: int | None = None) -> None:
        super().__init__()
        assert width % n_heads == 0, f"width {width} not divisible by n_heads {n_heads}"
        self.n_heads = n_heads
        self.head_dim = width // n_heads
        self.q = nn.Linear(width, width)
        self.kv = nn.Linear(kv_dim if kv_dim is not None else width, 2 * width)
        self.proj = nn.Linear(width, width)

    def forward(self, x: Tensor, ctx: Tensor) -> Tensor:
        b, nq, _ = x.shape
        nc = ctx.shape[1]
        q = self.q(x).view(b, nq, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(ctx).view(b, nc, 2, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, kv[0], kv[1])
        return self.proj(o.transpose(1, 2).reshape(b, nq, -1))


class _Block(nn.Module):
    """cross-attend to (z, lang) -> self-attend over operator queries -> MLP."""

    def __init__(self, width: int, n_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        hidden = int(width * mlp_ratio)
        self.ln_x = nn.LayerNorm(width)
        self.cross = _Attention(width, n_heads)
        self.ln_s = nn.LayerNorm(width)
        self.slf = _Attention(width, n_heads)
        self.ln_m = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, width)
        )

    def forward(self, x: Tensor, ctx: Tensor) -> Tensor:
        x = x + self.cross(self.ln_x(x), ctx)
        h = self.ln_s(x)
        x = x + self.slf(h, h)
        return x + self.mlp(self.ln_m(x))


class Proposal(nn.Module):
    """`pi_c(c | z, lang)` — Plackett-Luce over top-`TOPK` supports.  ~50M params.

    Satisfies `contracts.Proposal` and adds the three methods the rest of the
    system needs: `argmax` (R0 inference), `sample_seq` (shooting), and
    `logits` (RL / diagnostics).

    Conditioning: `m` learned queries — one per operator in the bank — cross
    attend to `concat(proj(z), proj(lang))` and self-attend among themselves,
    then a shared readout turns query token `m` into logit `m`.  One query per
    operator is the natural inductive bias for a distribution over a bank.

    `lang_dim` is a constructor argument (default 1152, SigLIP-so400m) because
    Team A has not finalised the text tower.  `m` / `topk` are constructor
    arguments so the PL math can be brute-force enumerated at `M=6, k=2`.
    """

    def __init__(
        self,
        dim: int = D,
        lang_dim: int = 1152,
        m: int = M,
        topk: int = TOPK,
        width: int = 768,
        n_blocks: int = 5,
        n_heads: int = 12,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        if not 1 <= topk <= m:
            raise ValueError(f"topk must be in [1, m]; got topk={topk}, m={m}")
        self.dim, self.lang_dim, self.m, self.topk, self.width = dim, lang_dim, m, topk, width

        self.z_in = nn.Linear(dim, width)
        self.lang_in = nn.Linear(lang_dim, width)
        self.ln_ctx = nn.LayerNorm(width)
        self.query = nn.Parameter(torch.randn(m, width) * 0.02)
        self.blocks = nn.ModuleList(
            [_Block(width, n_heads, mlp_ratio) for _ in range(n_blocks)]
        )
        self.ln_out = nn.LayerNorm(width)
        self.readout = nn.Linear(width, 1)
        self.apply(self._init)

    @staticmethod
    def _init(mod: nn.Module) -> None:
        if isinstance(mod, nn.Linear):
            nn.init.trunc_normal_(mod.weight, std=0.02)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)

    # ── conditioning ──────────────────────────────────────────────────────

    def logits(self, z: Tensor, lang: Tensor) -> Tensor:
        """(B, K, dim), (B, L, lang_dim) -> (B, m).  `lang` may also be (B, lang_dim)."""
        if z.ndim != 3:
            raise ValueError(f"proposal conditions on one belief per batch element, got {tuple(z.shape)}")
        if lang.ndim == 2:
            lang = lang.unsqueeze(1)
        b = z.shape[0]
        ctx = self.ln_ctx(torch.cat([self.z_in(z), self.lang_in(lang)], dim=1))
        x = self.query.unsqueeze(0).expand(b, -1, -1).to(ctx.dtype)
        for blk in self.blocks:
            x = blk(x, ctx)
        return self.readout(self.ln_out(x)).squeeze(-1)          # (B, m)

    # ── contracts.Proposal ────────────────────────────────────────────────

    def sample(
        self, z: Tensor, lang: Tensor, n: int, generator: torch.Generator | None = None
    ) -> Tensor:
        """(B, n, m) on the simplex with exactly `topk` nonzeros.

        Efficient at `n = 1000`: the network runs **once**, then one Gumbel
        top-k kernel over an `(B, n, m)` noise tensor.  Sequential PL sampling
        would be `topk` masked categorical draws instead; see the module
        docstring for why they are the same distribution.
        """
        logits = self.logits(z, lang)                            # (B, m)
        return self._sample_from(logits, (n,), generator)

    def sample_seq(
        self,
        z: Tensor,
        lang: Tensor,
        n: int,
        depth: int = DEPTH,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """(B, n, depth, m) — one candidate *plan* per `n`, for the shooting planner.

        Open-loop: every depth slice is an independent draw from the same
        `pi_c(.|z_t, lang)`.  Closed-loop conditioning would need `pi_c` to see
        the rolled-out `z_hat_d`, which requires the bank — and `contracts.Proposal`
        has no bank, deliberately: all expressivity lives in the filter, none of
        it inside the planning loop (PLAN 1).  The planner re-filters and
        re-plans after every executed root segment, so the closed loop is
        recovered at the outer rate.
        """
        logits = self.logits(z, lang)
        return self._sample_from(logits, (n, depth), generator)

    def _sample_from(
        self, logits: Tensor, extra: tuple[int, ...], generator: torch.Generator | None
    ) -> Tensor:
        b = logits.shape[0]
        wide = logits.view(b, *(1,) * len(extra), self.m).expand(b, *extra, self.m)
        idx = gumbel_topk(wide, self.topk, generator)            # (B, *extra, topk)
        return weights_from_logits(wide, idx, self.m)

    def log_prob(self, z: Tensor, lang: Tensor, c: Tensor) -> Tensor:
        """Exact PL log-probability of `c`'s canonical ordered support.

        c (B, m) -> (B,);  c (B, n, m) -> (B, n).  Differentiable w.r.t. both
        the network and (for RL) the sampled support.  This is what
        `losses/proposal_bc.py` maximises and what GRPO ratios divide.
        """
        logits = self.logits(z, lang)                            # (B, m)
        if c.shape[-1] != self.m:
            raise ValueError(f"coeff must end in m={self.m}, got {tuple(c.shape)}")
        if c.ndim > 2:
            logits = logits.view(logits.shape[0], *(1,) * (c.ndim - 2), self.m)
            logits = logits.expand(*c.shape)
        return pl_log_prob(logits, canonical_order(c, self.topk))

    # ── deterministic inference (R0: no search) ───────────────────────────

    def argmax(self, z: Tensor, lang: Tensor) -> Tensor:
        """(B, m) — deterministic top-`topk` of the logits, no Gumbel.

        This is the R0-A inference path: `z <- E(o)`, `c <- argmax pi_c(.|z, l)`,
        `a <- D_e(z, c)`.  It is the mode of the PL support distribution: the
        greedy sequential argmax picks the largest remaining logit at every
        step, which is exactly the unperturbed top-`k`.
        """
        logits = self.logits(z, lang)
        idx = logits.topk(self.topk, dim=-1).indices
        return weights_from_logits(logits, idx, self.m)

    # ── misc ──────────────────────────────────────────────────────────────

    def forward(self, z: Tensor, lang: Tensor) -> Tensor:       # convenience
        return self.logits(z, lang)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return f"m={self.m}, topk={self.topk}, width={self.width}, params={self.n_params()/1e6:.1f}M"
