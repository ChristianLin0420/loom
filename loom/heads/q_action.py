"""LOOM — q_a^e: the action-labelled operator encoder.  ONE PER EMBODIMENT, ~30 M each.

`(a_seg (B, H_OP, dof_e), z (B, K, D)) -> Coeff (B, M)`.

Held in a `ModuleDict` keyed by embodiment name. Batches are
embodiment-homogeneous (PLAN 9), so `forward` takes the name and routes once.

WHY THERE IS NO ALIGNMENT LOSS
──────────────────────────────
`q_a^e` is trained by plain regression onto `sg(q_Delta(z_t, z_{t+8}))` — see
`loom/losses/act.py::q_action_regression_loss`. That single stop-gradded
regression is what puts both encoders in *one* coefficient space, by
construction. No KL, no adversarial term, no separate alignment loss (PLAN 9:
do not add losses). The head below is imported from `q_delta.py` rather than
re-implemented for exactly the same reason: two encoders that discretise
differently do not share a space no matter what loss you put between them.

The action encoder is deliberately tiny relative to the trunk. `a_seg` is only
`H_OP * dof_e` = 56 numbers for LIBERO; the hard part is not reading the action,
it is deciding which operator that action *means given this belief*, which is a
joint function of both inputs and lives in the trunk.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from contracts import D, EMBODIMENTS, H_OP, K, M, TOPK
from loom.heads.q_delta import AttnPool, mlp_trunk, topk_simplex_st

__all__ = ["QActionBody", "QAction"]


class QActionBody(nn.Module):
    """The per-embodiment encoder. One of these per registered body, ~30 M."""

    def __init__(
        self,
        dof: int,
        hidden: int = 2560,
        n_queries: int = 4,
        n_heads: int = 8,
        n_hidden: int = 3,
        d_act: int = 512,
        d_act_out: int = 1024,
        topk: int = TOPK,
        temperature: float = 1.0,
        d: int = D,
        n_ops: int = M,
        n_slots: int = K,
        h_op: int = H_OP,
        d_kv: int | None = None,
    ) -> None:
        super().__init__()
        self.dof, self.h_op = dof, h_op
        self.topk, self.temperature = topk, temperature

        # belief side
        self.pool = AttnPool(d=d, n_queries=n_queries, n_heads=n_heads,
                             n_slots=n_slots, d_kv=d_kv)

        # action side: per-step embedding + learned step position, then one
        # projection over the flattened segment (H_OP is 8; attention is overkill)
        self.step_in = nn.Linear(dof, d_act)
        # zero-init is fine here (unlike the decoder's): the segment is flattened
        # into one projection, so step order is already carried by the weight
        # layout rather than by this embedding.
        self.step_emb = nn.Parameter(torch.zeros(h_op, d_act))
        self.act_norm = nn.LayerNorm(h_op * d_act)
        self.act_out = nn.Linear(h_op * d_act, d_act_out)

        self.trunk = mlp_trunk(n_queries * d + d_act_out, hidden, n_ops, n_hidden=n_hidden)

    def encode_action(self, a_seg: Tensor) -> Tensor:
        if a_seg.shape[-2:] != (self.h_op, self.dof):
            raise ValueError(
                f"action segment must be (..., {self.h_op}, {self.dof}), "
                f"got {tuple(a_seg.shape)}"
            )
        x = self.step_in(a_seg) + self.step_emb.to(a_seg.dtype)
        x = x.flatten(-2)
        return self.act_out(self.act_norm(x))

    def logits(self, a_seg: Tensor, z: Tensor) -> Tensor:
        h = torch.cat([self.pool(z), self.encode_action(a_seg.to(z.dtype))], dim=-1)
        return self.trunk(h)

    def forward(self, a_seg: Tensor, z: Tensor, return_logits: bool = False):
        lg = self.logits(a_seg, z)
        c = topk_simplex_st(lg, self.topk, self.temperature)
        return (c, lg) if return_logits else c


class QAction(nn.Module):
    """`ModuleDict` of per-embodiment `QActionBody`. Satisfies `contracts.QAction`.

    Args:
        embodiments: names to build. Default: every body registered in
            `contracts.EMBODIMENTS` at construction time. Modules are built
            eagerly (never lazily inside `forward`) so that the optimizer and
            FSDP see a fixed parameter set; register late bodies explicitly with
            `add_embodiment`.
        default_embodiment: used when `forward` is called without a name. Falls
            back to the single registered body if there is exactly one.
    """

    def __init__(
        self,
        embodiments: list[str] | tuple[str, ...] | None = None,
        default_embodiment: str | None = None,
        **body_kwargs,
    ) -> None:
        super().__init__()
        names = tuple(EMBODIMENTS) if embodiments is None else tuple(embodiments)
        if not names:
            raise ValueError("no embodiments registered; nothing to build")
        self.body_kwargs = dict(body_kwargs)
        self.bodies = nn.ModuleDict()
        for name in names:
            self.add_embodiment(name)
        self.default_embodiment = default_embodiment or (names[0] if len(names) == 1 else None)

    def add_embodiment(self, name: str) -> QActionBody:
        if name not in EMBODIMENTS:
            raise KeyError(f"unregistered embodiment {name!r}")
        if name not in self.bodies:
            self.bodies[name] = QActionBody(EMBODIMENTS[name].dof, **self.body_kwargs)
        return self.bodies[name]

    def body(self, embodiment: str | None = None) -> QActionBody:
        name = embodiment or self.default_embodiment
        if name is None:
            raise ValueError(
                "q_action holds several bodies; pass embodiment= "
                f"(have {sorted(self.bodies)})"
            )
        if name not in self.bodies:
            raise KeyError(f"q_action has no body {name!r}; have {sorted(self.bodies)}")
        return self.bodies[name]

    def logits(self, a_seg: Tensor, z: Tensor, embodiment: str | None = None) -> Tensor:
        return self.body(embodiment).logits(a_seg, z)

    def forward(
        self,
        a_seg: Tensor,
        z: Tensor,
        embodiment: str | None = None,
        return_logits: bool = False,
    ):
        return self.body(embodiment)(a_seg, z, return_logits=return_logits)
