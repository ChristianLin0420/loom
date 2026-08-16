"""LOOM — q_Delta: the action-free operator encoder.  SHARED, ~30 M params.

`(z_t, z_next) -> Coeff (B, M)`, on the simplex with hard top-`TOPK`.

This file also owns the two pieces that `q_action.py` and `decoder.py` reuse:

* `topk_simplex_st` — the *only* coefficient head in the repo. `q_Delta` and
  `q_a^e` must use the identical function or they write into two different
  coefficient spaces and the regression that ties them together is meaningless.
* `AttnPool` — belief pooling over the K axis.


POOLING CHOICE (deliberate, see PLAN 4.C)
─────────────────────────────────────────
Three candidates over the K=128 slot axis:

1. mean-pool          128*768 -> 768.  Free, but *permutation-invariant*: a
   belief where slot 3 holds the cup and slot 90 holds the gripper pools to
   exactly the same vector as the belief with those two contents swapped.
   Team B's estimator gives slots a learned identity, so throwing it away here
   would make "the cup moved" and "the arm moved" indistinguishable to q_Delta
   — precisely the distinction the operator is supposed to name.
2. flatten-then-project   K*D = 98 304 -> h.  Keeps everything, but a single
   h=2048 layer is 201 M params: 6.7x the entire q_Delta budget.
3. **learned attention pool (chosen).** `n_queries` learned queries cross-attend
   to the K slots, plus a learned per-slot embedding added to the keys so the
   map is *not* permutation-invariant. 1.9 M params, keeps slot identity, and
   the queries are free to specialise (e.g. "what moved" vs "where").

We take (3) with `n_queries=4`, and use three separate pools — for `z_t`,
`z_next` and the difference `z_next - z_t` — because those three streams have
very different statistics (state, state, near-zero delta) and sharing one
LayerNorm across them would be dominated by the two state streams.

The explicit `z_next - z_t` stream is not redundant with an MLP on the two
states: latent states 8 canonical steps apart are ~0.95 cosine-similar, so the
informative part of the input is a small difference of two large vectors, and
handing that difference to the network directly is worth more than the 1.9 M
params it costs.


STRAIGHT-THROUGH ESTIMATOR (exact form, see `topk_simplex_st`)
──────────────────────────────────────────────────────────────
    soft = softmax(logits / T)                      # dense, sums to 1
    S    = topk(soft, TOPK).indices                 # hard support
    Z    = soft[S].sum().detach()                   # detached normaliser
    hard = soft * mask_S / Z                        # value == renormalised top-k
    c    = hard + soft - soft.detach()

* forward value is exactly the hard, sparse, **renormalised** simplex point:
  `soft - soft.detach()` is bit-exactly zero, and `hard` sums to 1 because Z is
  its own (detached) sum. `contracts.assert_simplex` passes in f32 and bf16.
  A plain softmax over M would sum to 1 but have M nonzeros, which breaks
  BOTH of Team B's bounds (they are convexity arguments over <= TOPK atoms).
* backward is dense. Both terms are differentiable in *every* logit:
  `hard` reaches out-of-support logits through the global softmax denominator
  (d hard_j / d l_i = -soft_j soft_i / Z for i not in S), and the extra
  `soft - soft.detach()` term contributes the plain dense softmax Jacobian.
  This is what lets `L_balance` resurrect a dead operator: an operator that has
  fallen out of every support still receives d c_m/d l_m = soft_m(1-soft_m)/... > 0.
  The naive alternative — renormalising with a *differentiable* Z — silently
  cancels the global denominator, and then out-of-support logits get exactly
  zero gradient and a dead operator is dead forever.
* consequence worth knowing: `c.sum()` is 1 by construction, but its gradient
  is *not* zero here (it is -soft_i for i outside the support, i.e. "push the
  losers down"), because only the numerator of the renormalisation is
  differentiable. That is the gradient path the balance loss rides.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from contracts import D, K, M, TOPK

__all__ = ["topk_simplex_st", "AttnPool", "mlp_trunk", "QDelta"]


# ═══════════════════════════════════════════════════════════════════════════
#  THE SHARED COEFFICIENT HEAD
# ═══════════════════════════════════════════════════════════════════════════

def topk_simplex_st(
    logits: Tensor,
    topk: int = TOPK,
    temperature: float = 1.0,
    eps: float = 1e-6,
) -> Tensor:
    """Dense logits -> hard top-k renormalised simplex point, straight-through.

    Args:
        logits: (..., M) unnormalised scores.
        topk:   number of retained atoms. `contracts.TOPK` everywhere in LOOM.
        temperature: divides the logits before the softmax. Only affects the
            backward pass and the *choice* of support, never the on-simplex
            property of the forward value.
        eps: floor on the detached renormaliser.

    Returns:
        (..., M) with exactly `topk` nonzeros summing to 1.

    The exact estimator and why it is that one: see the module docstring.
    """
    if logits.shape[-1] < topk:
        raise ValueError(f"cannot take top-{topk} of {logits.shape[-1]} logits")
    if temperature != 1.0:
        logits = logits / temperature

    soft = torch.softmax(logits, dim=-1)                    # dense, sums to 1
    idx = soft.topk(topk, dim=-1).indices
    mask = torch.zeros_like(soft).scatter_(-1, idx, 1.0)

    sel = soft * mask
    z = sel.sum(-1, keepdim=True).detach().clamp_min(eps)   # DETACHED on purpose
    hard = sel / z

    # value: hard (the second pair is bit-exactly zero).  gradient: dense.
    return hard + soft - soft.detach()


# ═══════════════════════════════════════════════════════════════════════════
#  BELIEF POOLING
# ═══════════════════════════════════════════════════════════════════════════

class AttnPool(nn.Module):
    """Learned multi-head attention pool over the K slot axis.

    `(B, K, d) -> (B, n_queries * d_out)`.

    A learned per-slot embedding is added to the keys, so this is NOT
    permutation-invariant over slots: slot identity survives the pool.
    """

    def __init__(
        self,
        d: int = D,
        n_queries: int = 4,
        n_heads: int = 8,
        d_out: int | None = None,
        n_slots: int = K,
        d_kv: int | None = None,
    ) -> None:
        """`d_kv` is the attention bottleneck width (defaults to `d`). It is the
        one knob that moves this module's cost, which is 2*B*K*d*d_kv; the tests
        use a narrow one so a CPU run stays seconds rather than minutes."""
        super().__init__()
        inner = d_kv or d
        if inner % n_heads:
            raise ValueError(f"d_kv={inner} not divisible by n_heads={n_heads}")
        self.d, self.n_queries, self.n_heads = d, n_queries, n_heads
        self.d_kv = inner
        self.d_head = inner // n_heads
        self.d_out = d_out or d

        self.norm = nn.LayerNorm(d)
        # small random, not zeros: with a zero slot embedding the pool is
        # permutation-equivariant over slots at init, and slot identity would
        # only start to exist after the first optimizer step.
        self.slot_emb = nn.Parameter(torch.randn(n_slots, d) * 0.02)
        self.query = nn.Parameter(torch.randn(n_queries, inner) * inner ** -0.5)
        self.k_proj = nn.Linear(d, inner, bias=False)
        self.v_proj = nn.Linear(d, inner, bias=False)
        self.o_proj = nn.Linear(inner, self.d_out)

    def forward(self, z: Tensor) -> Tensor:
        b, n_slots, _ = z.shape
        x = self.norm(z) + self.slot_emb.to(z.dtype)

        k = self.k_proj(x).view(b, n_slots, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(b, n_slots, self.n_heads, self.d_head).transpose(1, 2)
        q = (self.query.to(z.dtype)
             .view(self.n_queries, self.n_heads, self.d_head)
             .permute(1, 0, 2).unsqueeze(0).expand(b, -1, -1, -1))

        att = torch.softmax(q @ k.transpose(-1, -2) * self.d_head ** -0.5, dim=-1)
        out = (att @ v).transpose(1, 2).reshape(b, self.n_queries, self.d_kv)
        return self.o_proj(out).reshape(b, self.n_queries * self.d_out)


def mlp_trunk(in_dim: int, hidden: int, out_dim: int, n_hidden: int = 2) -> nn.Sequential:
    """Pre-LN MLP: in -> hidden (x n_hidden) -> out.  Final layer small-init.

    Small-init (not zero-init) on the logit layer matters: with exactly-zero
    logits every sample ties and `topk` breaks the tie by index, so operators
    0..TOPK-1 would win every draw at step 0 and the rest would start dead.
    """
    layers: list[nn.Module] = [nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(hidden, hidden), nn.GELU()]
    layers += [nn.LayerNorm(hidden), nn.Linear(hidden, out_dim)]
    trunk = nn.Sequential(*layers)
    nn.init.normal_(trunk[-1].weight, std=0.02)
    nn.init.zeros_(trunk[-1].bias)
    return trunk


# ═══════════════════════════════════════════════════════════════════════════
#  q_Delta
# ═══════════════════════════════════════════════════════════════════════════

class QDelta(nn.Module):
    """Action-free operator encoder.  Satisfies `contracts.QDelta`.

    `(z_t (B,K,D), z_next (B,K,D)) -> Coeff (B,M)`.

    Shared across every embodiment — this is the module that defines what the
    coefficient space *means*; `q_a^e` is regressed onto its (stop-gradded)
    output, per body.

    Defaults are sized to the PLAN parameter budget (~30 M). Tests build small
    ones by passing `hidden` / `n_queries`.
    """

    def __init__(
        self,
        hidden: int = 2048,
        n_queries: int = 4,
        n_heads: int = 8,
        n_hidden: int = 2,
        topk: int = TOPK,
        temperature: float = 1.0,
        d: int = D,
        n_ops: int = M,
        n_slots: int = K,
        d_kv: int | None = None,
    ) -> None:
        super().__init__()
        self.topk, self.temperature = topk, temperature

        pool = lambda: AttnPool(d=d, n_queries=n_queries, n_heads=n_heads,
                                n_slots=n_slots, d_kv=d_kv)
        self.pool_t = pool()        # z_t
        self.pool_next = pool()     # z_next
        self.pool_delta = pool()    # z_next - z_t, the informative stream

        self.trunk = mlp_trunk(3 * n_queries * d, hidden, n_ops, n_hidden=n_hidden)

    def logits(self, z_t: Tensor, z_next: Tensor) -> Tensor:
        """(B,K,D) x2 -> (B,M) dense logits, before the top-k head."""
        h = torch.cat(
            [self.pool_t(z_t), self.pool_next(z_next), self.pool_delta(z_next - z_t)],
            dim=-1,
        )
        return self.trunk(h)

    def forward(self, z_t: Tensor, z_next: Tensor, return_logits: bool = False):
        lg = self.logits(z_t, z_next)
        c = topk_simplex_st(lg, self.topk, self.temperature)
        return (c, lg) if return_logits else c
