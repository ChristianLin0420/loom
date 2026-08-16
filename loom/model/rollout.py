"""
LOOM — multi-step rollout.

Sequential over `DEPTH`, batched over `N`. This is the whole planning inner
loop: `N x DEPTH` affine applications per cycle, no nonlinearity, no attention.

**There is no `compose()` and there must never be one.** Composing two affine
maps gives `(A2 A1, A2 b1 + b2)`. Multiplying the lambdas alone gives `A2 A1`
and silently drops `A2 b1`, which is the accumulated bias of every step but the
last. The result still has the right shape, still satisfies the spectral bound,
and is wrong. `tests/test_model.py::test_rollout_is_not_lambda_composition`
exists specifically to fail loudly if someone "optimises" this loop away.

The free function takes any `contracts.Bank`, so it works against `StubBank` as
well as `OperatorBank`; `OperatorBank.rollout` is wired to it.
"""

from __future__ import annotations

from torch import Tensor

from contracts import Bank, D, DEPTH, K

__all__ = ["rollout"]


def rollout(bank: Bank, c_seq: Tensor, z: Tensor) -> Tensor:
    """`(B, N, DEPTH, M)`, `(B, K, D)` -> `(B, N, K, D)`.

    One candidate plan per `n`; the `B` root beliefs are shared across candidates
    and broadcast with `expand`, so no memory is spent replicating `z`.

    Parameters
    ----------
    bank
        Anything satisfying `contracts.Bank`. Only `step` is used.
    c_seq
        `(B, N, DEPTH, M)`, each `c_seq[b, n, d]` on the simplex.
    z
        `(B, K, D)` root belief.
    """
    if c_seq.ndim != 4:
        raise ValueError(f"c_seq must be (B, N, DEPTH, M), got {tuple(c_seq.shape)}")
    if z.ndim != 3 or z.shape[-2:] != (K, D):
        raise ValueError(f"z must be (B, {K}, {D}), got {tuple(z.shape)}")
    if c_seq.shape[2] != DEPTH:
        raise ValueError(
            f"c_seq axis 2 is the planning horizon and must be DEPTH={DEPTH}, "
            f"got {c_seq.shape[2]}. One c is one operator is H_OP steps; never H_PLAN."
        )
    if c_seq.shape[0] != z.shape[0]:
        raise ValueError(
            f"batch mismatch: c_seq {tuple(c_seq.shape)} vs z {tuple(z.shape)}"
        )

    b, n = c_seq.shape[0], c_seq.shape[1]
    z = z.unsqueeze(1).expand(b, n, K, D)
    for d in range(DEPTH):
        z = bank.step(c_seq[:, :, d], z)
    return z
