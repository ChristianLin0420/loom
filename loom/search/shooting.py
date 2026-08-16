"""
LOOM — batched shooting.  Team E.  R3 only (PLAN 5, Phase 1B).

    C     = proposal.sample_seq(z, lang, n=N, depth=DEPTH)   # (B, N, DEPTH, M)
    leaf  = bank.rollout(C, z)                               # (B, N, K, D)
    score = potential(leaf, lang)                            # (B, N)
    best  = score.argmax(1)
    return C[arange(B), best, 0]                             # ROOT SEGMENT ONLY

That is the whole algorithm.  Three deliberate omissions, each of which is a
decision and not an oversight:

**No tree, no MCTS.**  A single `c` is a top-4 mixture, so the action set is
`C(128, 4) = 10.7M` supports times a continuous weight simplex on each.  "A
128-arm bandit at each node" is not a description of this action space, and any
discretisation that makes it one throws away the mixture.  Beyond that, MCTS
pays for itself through prefix reuse, and at `DEPTH = 4` with `N = 1000` i.i.d.
open-loop plans there is essentially no shared prefix to reuse — the rollout is
`N x DEPTH` affine steps either way, and the flat version is one batched kernel
instead of a pointer-chasing loop.  The outer loop re-filters and re-plans after
every executed root segment, which is where the real feedback lives.

**No root term, no uncertainty term, no cost term.**  The score is
`Phi(z_hat_DEPTH, lang)` alone.  `Phi(z_t, lang)` is identical across candidates
and cancels in the argmax, so adding it is free but pointless; uncertainty and
cost terms have no contract in `contracts.py`, and inventing them here is
exactly the drift PLAN 9 forbids.

**Root segment only.**  `C[b, best]` is a 4-operator plan; we execute
`C[b, best, 0]` and throw the tail away.  One `c` = one operator = `H_OP = 8`
control steps.  Never `H_PLAN`.

Realizability gate
==================

A candidate that scores well in belief space is useless if this body cannot
produce it.  Reject root `c` when

    || q_a(D_e(z, c), z) - c ||_2  >  REALIZABILITY_TAU

and **fall through to the runner-up** — a proper ranked walk down the sorted
candidates, not a single retry, because the failure mode is correlated: if the
best candidate is unrealizable, its neighbours in score usually are too.  The
walk is batched across `B` and exits as soon as every batch element is
resolved, so the common case costs one gate evaluation.  If the whole ranking
fails, return the best-scoring candidate anyway and raise
`info["gate_exhausted"]` — the planner never returns nothing, because the
episode has to keep stepping.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from contracts import DEPTH, REALIZABILITY_TAU, Bank, Decoder, Potential, Proposal, QAction

__all__ = ["shooting", "realizability_residual"]


def realizability_residual(
    z: Tensor, c: Tensor, q_action: QAction, decoder: Decoder
) -> Tensor:
    """`|| q_a(D_e(z, c), z) - c ||_2` per batch element.  (B, K, D), (B, M) -> (B,)."""
    a_seg = decoder(z, c)                        # (B, H_OP, dof_e)
    c_hat = q_action(a_seg, z)                   # (B, M)
    return (c_hat.float() - c.float()).norm(dim=-1)


@torch.no_grad()
def shooting(
    proposal: Proposal,
    bank: Bank,
    potential: Potential,
    z: Tensor,
    lang: Tensor,
    *,
    n: int = 1000,
    depth: int = DEPTH,
    q_action: QAction | None = None,
    decoder: Decoder | None = None,
    tau: float = REALIZABILITY_TAU,
    max_gate_evals: int | None = None,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, dict[str, Any]]:
    """Plan by shooting; return `(c_root, info)`.

    Args:
        proposal:  `pi_c`; must expose `sample_seq(z, lang, n, depth)`.
        bank:      operator bank; `rollout((B,N,DEPTH,M), (B,K,D)) -> (B,N,K,D)`.
        potential: `Phi`; `(B,N,K,D), lang -> (B,N)`.
        z:         `(B, K, D)` belief at the planning instant.
        lang:      `(B, L, F)` instruction features.
        n:         candidate plans.  `N = 1000` in PLAN 1.
        depth:     planning horizon in operators.  `DEPTH = 4`.
        q_action, decoder: the *current embodiment's* heads.  Pass both to arm
            the realizability gate; pass neither to disable it (R3 ablations,
            and any caller that has not got per-body heads wired yet).
        tau:       gate threshold, `contracts.REALIZABILITY_TAU`.
        max_gate_evals: cap on how far down the ranking to walk.  `None` walks
            the whole ranking; the loop exits early once every batch element is
            resolved, so `None` is cheap in practice and only bites in the
            pathological case the `gate_exhausted` flag reports.
        generator: for reproducible candidate sets.

    Returns:
        `c_root`: `(B, M)` on the simplex with `<= TOPK` nonzeros — ONE
            operator, to be realized by `D_e(z, c_root)` as `H_OP` steps.
        `info`: everything Team F logs — see the keys below.
    """
    if z.ndim != 3:
        raise ValueError(f"shooting plans from one belief per batch element, got {tuple(z.shape)}")
    if (q_action is None) != (decoder is None):
        raise ValueError(
            "the realizability gate needs both q_action and decoder "
            "(it evaluates ||q_a(D_e(z, c), z) - c||); pass both or neither"
        )

    b = z.shape[0]
    dev = z.device
    rows = torch.arange(b, device=dev)

    # ── 1. propose ─ 2. roll out ─ 3. score ───────────────────────────────
    c_seq = proposal.sample_seq(z, lang, n, depth, generator=generator)   # (B,N,DEPTH,M)
    leaf = bank.rollout(c_seq, z)                                         # (B,N,K,D)
    score = potential(leaf, lang)                                         # (B,N)
    if score.shape != (b, n):
        raise ValueError(f"potential must return (B, N) = {(b, n)}, got {tuple(score.shape)}")

    order = score.float().argsort(dim=1, descending=True)                 # (B,N)

    # ── 4. rank, with realizability fall-through ──────────────────────────
    rank = torch.zeros(b, dtype=torch.long, device=dev)
    n_rejected = torch.zeros(b, dtype=torch.long, device=dev)
    resolved = torch.zeros(b, dtype=torch.bool, device=dev)
    gate_on = q_action is not None

    if gate_on:
        budget = n if max_gate_evals is None else min(n, max_gate_evals)
        for r in range(budget):
            cand = c_seq[rows, order[:, r], 0]                            # (B,M)
            ok = realizability_residual(z, cand, q_action, decoder) <= tau
            active = ~resolved
            n_rejected = n_rejected + (active & ~ok).long()
            accept = active & ok
            rank = torch.where(accept, torch.full_like(rank, r), rank)
            resolved = resolved | accept
            if bool(resolved.all()):
                break
    else:
        resolved = torch.ones(b, dtype=torch.bool, device=dev)

    # exhausted elements keep rank 0 -> the best-scoring candidate. Never nothing.
    gate_exhausted = ~resolved

    chosen = order.gather(1, rank[:, None]).squeeze(1)                    # (B,)
    plan = c_seq[rows, chosen]                                            # (B,DEPTH,M)
    c_root = plan[:, 0]                                                   # (B,M) ROOT ONLY

    info: dict[str, Any] = {
        "scores": score,                                  # (B, N)
        "index": chosen,                                  # (B,)  candidate index
        "rank": rank,                                     # (B,)  0 == top-scoring
        "chosen_score": score.gather(1, chosen[:, None]).squeeze(1),      # (B,)
        "best_score": score.gather(1, order[:, :1]).squeeze(1),           # (B,)
        "n_rejected": n_rejected,                         # (B,) gate rejections
        "gate_exhausted": gate_exhausted,                 # (B,) every candidate failed
        "gate_applied": gate_on,
        "plan": plan,                                     # (B, DEPTH, M) tail is discarded
        "n_candidates": n,
        "depth": depth,
        "tau": tau,
    }
    return c_root, info
