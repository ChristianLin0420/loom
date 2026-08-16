"""LOOM — q_Delta: the action-free operator encoder.  SHARED, ~30 M params.

`(z_t, z_next) -> Coeff (B, M)`, on the simplex with hard top-`TOPK`.

This file also owns the two pieces that `q_action.py` and `decoder.py` reuse:

* `topk_simplex_st` — the *only* coefficient head in the repo. `q_Delta` and
  `q_a^e` must use the identical function or they write into two different
  coefficient spaces and the regression that ties them together is meaningless.
* `CenteredReadout` — the logit layer. The operator-axis mean is an exact null
  direction of everything downstream and has to be projected out in fp32,
  before autocast rounds; see the class docstring.
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
import torch.nn.functional as F
from torch import Tensor, nn

from contracts import D, K, M, TOPK

__all__ = ["topk_simplex_st", "CenteredReadout", "AttnPool", "mlp_trunk", "QDelta"]


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


class CenteredReadout(nn.Linear):
    """The logit layer, with the operator-axis mean projected out **in fp32**.

    Everything downstream of these logits -- `topk_simplex_st`'s softmax, the
    top-k support, the renormalised weights, `L_balance` -- is EXACTLY invariant
    to a constant added to every operator's logit. So the row-mean of `weight`
    (and the mean of `bias`) is an unregularised null direction of the loss: it
    receives zero gradient, and `bias` sits in `schedule._NO_DECAY`, so nothing
    pulls it back either. It is the same direction that `pi_c`'s `readout.bias`
    random-walked along until bf16 swallowed the entire spread (commit 18ef0d5).

    q_a and q_Delta did not run away as far as `pi_c` did, but R0-A measured the
    direction costing real resolution anyway. bf16's ulp is magnitude
    proportional, `2**(floor(log2|x|) - 7)`. At step 7004, on 1536 real LIBERO
    windows under the loop's own `autocast(bf16)`:

        head      common mode   spread over M   distinct/row   top-4 support
                                                               vs the fp32 rank
        q_Delta      +7.12          1.41            35.8         3.08 of 4
        q_Delta       0.00          1.41            92.8         3.71 of 4  (centred)
        q_a          +3.88          0.0210           8.3         0.04 of 4
        q_a           0.00          0.0195         118.2         2.54 of 4  (centred)

    So one operator in four was being chosen by a rounding tie rather than by
    the ranking the head had actually learned.

    Subtracting the mean row from `weight` and the mean from `bias` is exactly
    `logits - logits.mean(-1)`, but done while the numbers are still fp32.
    Doing it *after* the readout does not work -- the rounding has already
    happened (measured on `pi_c`: 2.95 distinct values per row).

    Pure reparameterisation: identical softmax, identical top-k in exact
    arithmetic, and `weight`/`bias` keep their names so existing checkpoints
    stay loadable. The projection also lands on the backward pass, so the null
    direction now gets exactly zero gradient instead of a random walk.

    This does NOT fix an input-blind head. q_a's own collapse at step 7004 was
    upstream of here -- its trunk input carried a batch-std/batch-mean ratio of
    0.25 against q_Delta's 1.80, and by `trunk[4]` every sample's hidden state
    was the same vector up to scale (pairwise cosine 1.0000), which the
    pre-readout LayerNorm then divides out. Centring makes the readout report
    that faithfully instead of hiding it under quantisation noise. That failure
    is what `logit_rms` below addresses.


    `logit_rms` -- WHY THE LOGIT SCALE IS PINNED, AND ONLY FOR q_a
    ─────────────────────────────────────────────────────────────
    `None` (q_Delta) is the historical behaviour, bit for bit. A float (q_a)
    rescales each row to that rms over the operator axis, in fp32:

        logits <- logit_rms * l / rms_M(l)          (l already mean-free)

    Monotone and per-sample positive, so the ranking, the top-k support and the
    on-simplex property are all untouched; `topk_simplex_st` never sees a
    different kind of object. What changes is the *gradient*: the radial
    direction (scale every logit by the same factor) is projected out and gets
    exactly zero gradient.

    That direction is the one q_a died along. `L_act`'s regression term is
    `||c_a - sg(c_Delta)||^2` between two hard top-4 renormalised simplex points.
    When the two supports are disjoint -- which is the situation at
    initialisation -- the loss is `sum(c_a^2) + sum(c_Delta^2)` and the ONLY term
    q_a controls is `sum(c_a^2)`, minimised by flat coefficients, i.e. by tied
    logits. Worse, the straight-through head has a built-in asymmetry that drives
    exactly that: an in-support atom's coefficient moves with
    `d hard_m/d l_m = soft_m(1-soft_m)/Z`, an out-of-support atom's only with
    `soft_m(1-soft_m)`, and `Z ~ TOPK/M`, so "push down whoever is on top now" is
    `M/TOPK = 32x` stronger than "pull up the atom the target wants". Its fixed
    point is all logits equal. Measured on R0-A: the operator-axis spread fell
    0.929 (fresh init, on real beliefs) to 0.0195 at step 7004, `act/align` sat
    on its disjoint-support floor of 0.500 for 7004 steps, and the head reached
    the flattest point of the simplex, top-4 weights
    [0.25058, 0.24991, 0.24979, 0.24971].

    The route it took is worth recording, because it is what makes the collapse
    irreversible. Flattening the logits *for every input at once* is cheapest if
    the hidden state has no directions left to disagree about: with `h` confined
    to a ray, one orthogonality constraint on the readout flattens everything,
    where a full-rank `h` would need the whole readout shrunk. So the loss pays
    for collapsing `h`. It duly collapsed -- `act_out`'s gain along the batch-mean
    direction of its input went 0.284 (fresh) to 9.333 (step 7004) while its gain
    on the deviations stayed 0.294 to 0.315, and the readout's row deviations
    turned orthogonal to the mean hidden state (spread at `h_bar` 0.685 to 0.023
    against 0.558 for a random readout of the same norm). By `trunk[4]` the
    pairwise cosine over 1536 real windows was exactly 1.00000.

    Pinning the rms makes the whole loss EXACTLY invariant to the scale of the
    logits, so that entire route is closed: `dL/d(gain)` is analytically zero
    (`tests/test_heads.py::test_the_align_objective_pays_to_shrink_free_logits...`
    measures 9.7e-1 free against 1.9e-8 pinned), and shrinking the readout -- or
    rotating it orthogonal to `h` -- buys the loss nothing whatsoever. What is
    left of the gradient is purely rotational: it moves the support towards the
    target's, which is the learning we want.

    Be precise about what this does NOT claim, because it was measured and it is
    not what you would hope. A flat coefficient vector is still *representable*
    under the pin -- four tied logits high and 124 tied low has rms 1 -- so a head
    whose target is genuinely unpredictable still switches itself off. What the
    pin buys is that the switch is REVERSIBLE.

    The experiment (`logs/probe_qa_train.py --noise_until 1500`, one A100, 60 s a
    variant, R0-A's exact optimiser settings on real LIBERO beliefs): train q_a
    for 1500 steps against a fresh random 4-of-128 target every step -- R0-A's
    first ~2000 steps, when `delta_op` is ~0 and q_Delta carries nothing -- then
    switch to the real, learnable, frozen q_Delta target for 1500 more.

        after 1500 noise steps       pinned            unpinned
          distinct supports / 1536      1                 1
          pairwise cosine trunk[4]      1.0000            1.0000
          operator-axis spread          1.004 (pinned)    0.254
          shuffle a_seg, logit rms      5e-6              7.8e-5

        after 1500 more REAL steps   pinned            unpinned
          distinct supports             2                 2
          pairwise cosine trunk[4]      0.6286            0.99999
          operator-axis spread          1.004 (pinned)    0.0031
          shuffle a_seg, logit rms      3.8e-4            2.1e-5
          top-4 overlap with target     1.13 of 4         0.00 of 4
          align MSE                     0.375             0.5248

    The unpinned head's 0.5248 with a single constant support and a spread of
    0.003 IS R0-A at step 7004 (0.4497, one support, spread 0.0195), reproduced
    from scratch. It cannot come back because the gradient that would re-establish
    input dependence is proportional to the logit scale it has already destroyed.
    The pinned head has that scale held at 1 by construction, so the same gradient
    is O(1) and it climbs out.

    q_Delta keeps `None` -- it is trained through the bank by `L_dyn`, has no
    flattening pressure, and measured a healthy spread of 1.41; pinning it would
    change Team B's inputs for no reason.

    A pinned scale is also the end of the bf16 resolution problem for q_a: the
    spread is `logit_rms` by construction rather than 0.0195, which is 1.2 ulps
    at a common mode of 3.88.

    `logit_rms` is a CONSTANT, never a parameter. A learnable one is the same
    null direction with an extra step, and would be driven to zero by exactly the
    gradient this projects out.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 *, logit_rms: float | None = None, **kw) -> None:
        super().__init__(in_features, out_features, bias=bias, **kw)
        if logit_rms is not None and not float(logit_rms) > 0.0:
            raise ValueError(f"logit_rms must be positive or None, got {logit_rms}")
        self.logit_rms = None if logit_rms is None else float(logit_rms)

    def extra_repr(self) -> str:
        s = super().extra_repr()
        return s if self.logit_rms is None else f"{s}, logit_rms={self.logit_rms}"

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight
        w32 = (w.float() - w.float().mean(0, keepdim=True))
        wc = w32.to(w.dtype)
        b32 = bc = None
        if self.bias is not None:
            b = self.bias
            b32 = (b.float() - b.float().mean())
            bc = b32.to(b.dtype)
        y = F.linear(x, wc, bc)
        if self.logit_rms is None:
            return y
        # The pinned path redoes the projection in fp32 and normalises there.
        # `y` is only consulted for its dtype, so autocast's contract is kept.
        #
        # Not a micro-optimisation to skip: the pin leaves the PRE-normalisation
        # scale of the readout completely unconstrained (the loss is invariant to
        # it, which is the whole point), so it random-walks -- measured 0.93 at
        # init and 317.4 after 4000 steps of R0-A. bf16's ulp is magnitude
        # proportional, so rounding `y` at |317| quantised the row to 34.8
        # distinct values of 128 and left the bf16 top-4 agreeing with the fp32
        # ranking on only 1.37 of 4 atoms. Dividing afterwards cannot undo that;
        # this is the same "centre before the rounding, not after" argument as
        # above, applied to the scale. The extra matmul is (B, hidden) x
        # (hidden, M) -- 10 MFLOP at the training batch, against a 150 M
        # estimator.
        with torch.autocast(device_type=x.device.type, enabled=False):
            f = F.linear(x.float(), w32, b32)
        scale = self.logit_rms / f.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        return (f * scale).to(y.dtype)


def mlp_trunk(in_dim: int, hidden: int, out_dim: int, n_hidden: int = 2,
              logit_rms: float | None = None) -> nn.Sequential:
    """Pre-LN MLP: in -> hidden (x n_hidden) -> out.  Final layer small-init.

    Small-init (not zero-init) on the logit layer matters: with exactly-zero
    logits every sample ties and `topk` breaks the tie by index, so operators
    0..TOPK-1 would win every draw at step 0 and the rest would start dead.

    The readout is a `CenteredReadout`, not a plain `nn.Linear`: see that class
    for why the operator-axis mean has to leave in fp32, and for what `logit_rms`
    is for. `None` -- q_Delta's setting and this function's default -- is the
    plain centred readout.
    """
    layers: list[nn.Module] = [nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(hidden, hidden), nn.GELU()]
    layers += [nn.LayerNorm(hidden), CenteredReadout(hidden, out_dim, logit_rms=logit_rms)]
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
