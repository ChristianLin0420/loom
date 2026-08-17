"""LOOM — L_balance: keep every operator alive.

    L_balance = BALANCE_COEF * KL( mean_batch(c) || uniform(M) )
              = BALANCE_COEF * ( sum_m p_m log p_m + log M ),   p = mean_batch(c)

**This is not the form R0-A executes any more.** The owner replaced it with the
Switch auxiliary `M * sum_m f_m P_m`, which lives in
`loom.train.loop._switch_balance` because it needs the heads' DENSE logits and
this module only ever sees `c`. The KL below is unchanged, still tested, and is
the reference the Switch form replaced; `contracts.BALANCE_COEF` is now 1e-2 for
both. See `_switch_balance` for the measurement that motivated the swap.

`contracts.BALANCE_COEF` was 3e-3 and is now 1e-2. Still small: hard top-k
already provides all the sparsity this model needs, so this term exists ONLY to
stop operators from dying. Turn it up far enough and it fights the task loss for
control of the support.


KL DIRECTION — this is the part that is easy to get backwards
─────────────────────────────────────────────────────────────
Forward KL, **batch-mean usage first, uniform second**: `KL(p || u)`.

* `KL(p || u) = -H(p) + log M`. It is exactly the (negated) entropy of the usage
  histogram, is finite and smooth at `p_m = 0` (0 log 0 = 0), is zero iff usage
  is uniform, and is positive otherwise. Its gradient w.r.t. an unused
  operator's mass is `log p_m + 1 + log M`, which is large and negative — a
  steady pull upward on dead operators, which is precisely the job.
* The reverse, `KL(u || p) = -log M - (1/M) sum_m log p_m`, is `+inf` the moment
  a single operator goes unused, and its gradient blows up as `1/p_m`. With
  hard top-k and batch sizes below M/TOPK, *most* operators are unused in any
  given batch by construction, so the reverse direction is not merely worse-
  conditioned, it is undefined on almost every step.

Note this is a batch-level constraint, not a per-sample one: individual `c`s
must stay sparse (that is what the bounds rest on); it is only their *average*
that should be flat. Applying an entropy penalty per sample would fight the
top-k head directly.

`c` is expected to be on the simplex already (it comes out of
`topk_simplex_st`); the batch mean of simplex points is a distribution.
Gradient reaches the logits through the straight-through head, including atoms
outside the current top-k support — that path is what makes resurrection
possible, see `loom/heads/q_delta.py`.
"""

from __future__ import annotations

import math

from torch import Tensor

from contracts import BALANCE_COEF, M

__all__ = ["operator_usage", "balance_kl", "balance_loss"]


def operator_usage(c: Tensor) -> Tensor:
    """(..., M) -> (M,) mean coefficient mass per operator over all leading axes."""
    if c.shape[-1] != M:
        raise ValueError(f"coeff must end in M={M}, got {tuple(c.shape)}")
    return c.reshape(-1, c.shape[-1]).mean(0)


def balance_kl(c: Tensor, eps: float = 1e-8) -> Tensor:
    """KL(mean_batch(c) || uniform(M)), unweighted. >= 0, == 0 iff usage is flat."""
    p = operator_usage(c)
    p = p / p.sum().clamp_min(eps)                    # guard: c may be bf16
    return (p * (p.clamp_min(eps).log() + math.log(M))).sum()


def balance_loss(c: Tensor, coef: float = BALANCE_COEF, eps: float = 1e-8) -> Tensor:
    """`coef * KL(mean_batch(c) || uniform(M))`. Default coef is BALANCE_COEF."""
    return coef * balance_kl(c, eps)
