"""
LOOM — the belief estimator `E`.

`z_t = E(o_t, l, z_{t-1})`, `z in R^(K x D)` with `K = 128`, `D = 768`.

A Perceiver: `K` learned latent queries cross-attend to the concatenation of
`views`, `proprio`, `lang` and `z_prev`. 10 blocks, `d = D = 768`, 16 heads,
pre-LN, mlp_ratio 4.

Filtering and prediction are separate (PLAN.md 1). *All* the nonlinearity of the
model lives here; it runs once per executed segment at 3.75 Hz. The rollout is
pure affine algebra and runs `N x DEPTH` times per cycle. So this file is
allowed to be the expensive one, and it is the only one that is.

Design decisions that are load-bearing
--------------------------------------

**Learned slot embeddings on the K axis, no RoPE and no sinusoids.** The 128
queries are slots, not a sequence. There is no reason slot 12 should be nearer
slot 13 than slot 80, and a positional encoding that says otherwise injects an
ordering the belief does not have. Spatial structure stays inside the frozen
vision tokens, where it belongs. The slot embedding is added to the query at
every attention (and to the `z_prev` context tokens, so the previous belief
arrives as identified slots rather than a bag), which is what keeps slot
identity stable across the 10 blocks and across the recurrence.

**`feat_dim` is a constructor argument.** Team A has not finalised the frozen
tower. 1152 is SigLIP-so400m; anything else is one keyword away.

**Per-stream embedding, not per-stream projection.** `views` arrives as
`(B, V, P, F)` and V varies by embodiment (2 for LIBERO, 7 for a bimanual rig
with tactile gel-pads). V*P is flattened into one token axis and a small learned
embedding tags the stream, so the same weights serve any V up to `max_streams`.

**Proprio projection is a `ModuleDict` keyed by embodiment.** `dof` differs per
body and there is no shared meaning across bodies for joint 5 anyway. Batches
are embodiment-homogeneous by construction (PLAN.md 9), so dispatch is a dict
lookup once per forward — free. The alternative, zero-padding every body to a
global max dof, teaches the shared trunk that "joint 12 is always zero for
LIBERO", which is exactly the leakage the homogeneous-batch rule exists to
prevent.

**`z_prev` enters twice — and the additive path is a switch, not a feature.**
`z_prev` reaches the trunk as context tokens (so slot `i` can read slot `j`'s
previous value — belief mixing) *and*, when `z_prev_residual=True`, as an
additive residual on the queries (so slot `i` starts from its own previous
value — belief persistence). With `z_prev=None` the queries are the learned
latents alone, which is the episode-start case unless `learned_z_init=True`.

The additive residual was the default, and it made `z` an over-damped tracker.
Measured on the R0-A2 checkpoint at step 5040, closed-loop LIBERO: the belief's
step-to-step motion `1 - cos(LN z_t, LN z_{t-1})` averaged **0.0043** for `t>=5`
while consecutive *observations* moved **0.0155** — `z` was low-passing the
observation with a time constant longer than an episode. It is not a fixed
point: swapping the observation still moved `z` (0.2069 at `t=1`, rising to
0.2692 at `t=19`), so the recurrence was live, just too heavy. Ablating **only**
the additive residual at inference on those same weights took step-motion to
**0.02887** and collapsed `cosdist(z_t, z_0)` from 1.286 — anti-correlated with
its own start — to 0.2298, next to a fresh-init model's 0.198. Scaling `||zp||`
down does *not* reproduce this: at `alpha=0.02` the proposal entropy is
unchanged from `alpha=1` (4.8444), and only `alpha=0` moves it. The regime
switch is discontinuous in the *presence* of the residual, not graded in its
magnitude, which is why this is a boolean and not a coefficient.

The context path alone still carries the recurrence, so
`tests/test_model.py::test_estimator_z_prev_changes_output` holds with
`z_prev_residual=False` — `tests/test_model.py::test_estimator_recurrence_paths`
asserts that for every combination of the two flags.

`learned_z_init=True` replaces the `z_prev=None` special case with a learned
`(K, D)` initial belief, so one code path always runs and there is no
episode-start seam between "queries are the latents" and "queries are the
latents plus a projection of something".

**Trained from scratch, the damping story is confirmed and the fix is not.**
Three 7000-step 16-GPU R0-A runs, same seed, same data order, differing only in
these two flags (`teamb_ctrl` / `teamb_nores` / `teamb_zinit`), scored on 300
closed-loop LIBERO episodes:

    arm                          step-motion  x-episode cos   LIBERO @4500  @7000
    z_prev_residual=True (ship)      0.0086         0.94          16.0      10.0
    residual removed                 0.0438         0.56           5.3       1.0
    residual removed + z_init        1.04           0.95          12.0      18.0

Removing the residual does exactly what the inference-time ablation predicted --
`z` moves 5x more and the beliefs of different episodes stop agreeing (0.94 ->
0.56, the least collapsed belief measured anywhere in this repo) -- **and the
score falls to 1.0.** Unfreezing the belief is not what the policy needed. Add
the learned `z_init` on top and the recurrence turns into a period-3 orbit
(`switch_frac` exactly 1.000: the operator changes at *every* replan, 3 per
episode) which scores 18.0, the best number this repo has produced, and the only
arm whose score rises with training. So `learned_z_init` is not a seam fix here;
it is what makes a residual-free recurrence oscillate instead of drift, and the
oscillation is worth 18 points. Neither arm makes the policy condition on the
observation: all three pick ONE operator at replan 0 across all 300 episodes.

**Neither flag is in `state_dict`, and both change what the weights mean.**
`z_prev_residual` is not a parameter at all, and a checkpoint trained with
`learned_z_init=True` carries `z_init` as an *unexpected* key for a default
build. `loom/eval/policy.py` constructs `Estimator(embodiments=[e])` with the
defaults and loads `strict=False`, so pointing it at one of these checkpoints
scores a **different model** than the one that trained, with nothing in the log
to say so: arm (i) silently regains the residual, arm (ii) silently regains it
*and* drops `z_init`. The run's `config.json` holds `model.estimator`, and
`loom.train.consolidate` reads it; any other reader has to be told. Adding the
flags to `state_dict` would fix that at the cost of a missing key on every
checkpoint written before them, which breaks `policy.py`'s "refusing to score a
partly-loaded module" guard for the entire existing R0-A family -- so the flags
stay out and this paragraph is the interlock.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as Fn
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from contracts import D, EMBODIMENTS, K, ObsFeats

__all__ = ["Estimator"]


# ═══════════════════════════════════════════════════════════════════════════
#  BLOCKS
# ═══════════════════════════════════════════════════════════════════════════

class Attention(nn.Module):
    """Multi-head attention with separate q / kv sources. `4 d^2` params."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim {dim} not divisible by heads {heads}")
        self.heads = heads
        self.head_dim = dim // heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

    def _split(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, q_src: Tensor, kv_src: Tensor) -> Tensor:
        q = self._split(self.q(q_src))
        k = self._split(self.k(kv_src))
        v = self._split(self.v(kv_src))
        x = Fn.scaled_dot_product_attention(q, k, v)
        b, _, t, _ = x.shape
        return self.o(x.transpose(1, 2).reshape(b, t, self.heads * self.head_dim))


def _mlp(dim: int, ratio: int) -> nn.Sequential:
    """`2 * ratio * d^2` params."""
    hidden = dim * ratio
    return nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))


class PerceiverBlock(nn.Module):
    """cross-attn -> FFN -> self-attn -> FFN, all pre-LN. `24 d^2` params.

    The context is LayerNormed once outside the loop (it is identical for every
    block), so there is no per-block kv LN.
    """

    def __init__(self, dim: int, heads: int, mlp_ratio: int) -> None:
        super().__init__()
        self.ln_xq = nn.LayerNorm(dim)
        self.xattn = Attention(dim, heads)
        self.ln_m1 = nn.LayerNorm(dim)
        self.mlp1 = _mlp(dim, mlp_ratio)
        self.ln_sa = nn.LayerNorm(dim)
        self.sattn = Attention(dim, heads)
        self.ln_m2 = nn.LayerNorm(dim)
        self.mlp2 = _mlp(dim, mlp_ratio)

    def forward(self, x: Tensor, ctx: Tensor, slot: Tensor) -> Tensor:
        x = x + self.xattn(self.ln_xq(x) + slot, ctx)
        x = x + self.mlp1(self.ln_m1(x))
        h = self.ln_sa(x) + slot                    # slot embed on q *and* k
        x = x + self.sattn(h, h)
        x = x + self.mlp2(self.ln_m2(x))
        return x


# ═══════════════════════════════════════════════════════════════════════════
#  ESTIMATOR
# ═══════════════════════════════════════════════════════════════════════════

class Estimator(nn.Module):
    """`E(o_t, l, z_{t-1}) -> z_t`, `(B, K, D)`. Satisfies `contracts.Estimator`.

    ~144 M parameters at the defaults — the 150 M row of the budget table.
    Per block: `4d^2` cross-attn + `8d^2` FFN + `4d^2` self-attn + `8d^2` FFN
    = `24 d^2` = 14.2 M, times 10 blocks = 141.6 M, plus ~2.5 M of input
    projections and embeddings. mlp_ratio stays at 4; raising it is the usual way
    this module silently becomes 600 M.
    """

    def __init__(
        self,
        feat_dim: int = 1152,           # SigLIP-so400m; Team A may change it
        lang_dim: int | None = None,    # defaults to feat_dim
        dim: int = D,
        depth: int = 10,
        heads: int = 16,
        mlp_ratio: int = 4,
        n_slots: int = K,
        max_streams: int = 8,
        embodiments: Sequence[str] | None = None,
        grad_checkpoint: bool = False,
        z_prev_residual: bool = True,
        learned_z_init: bool = False,
    ) -> None:
        super().__init__()
        if dim != D:
            raise ValueError(
                f"the belief width is frozen at D={D} in contracts.py; got dim={dim}"
            )
        self.dim = dim
        self.n_slots = n_slots
        self.max_streams = max_streams
        self.feat_dim = feat_dim
        self.lang_dim = feat_dim if lang_dim is None else lang_dim
        self.grad_checkpoint = grad_checkpoint
        #: add `z_prev` onto the queries as well as into the context. See the
        #: module docstring: True is the shipped over-damped tracker.
        self.z_prev_residual = bool(z_prev_residual)

        # ── latents ───────────────────────────────────────────────────────
        self.latents = nn.Parameter(torch.empty(n_slots, dim))
        self.slot_embed = nn.Parameter(torch.empty(n_slots, dim))
        #: the belief before the first observation. With it, `z_prev=None` is
        #: not a separate code path -- the same forward runs at every t.
        self.z_init = nn.Parameter(torch.empty(n_slots, dim)) if learned_z_init else None

        # ── input projections ─────────────────────────────────────────────
        self.view_proj = nn.Linear(feat_dim, dim)
        self.lang_proj = nn.Linear(self.lang_dim, dim)
        self.z_prev_ln = nn.LayerNorm(dim)
        self.z_prev_proj = nn.Linear(dim, dim)

        names = tuple(EMBODIMENTS) if embodiments is None else tuple(embodiments)
        if not names:
            raise ValueError("no embodiments registered; cannot build proprio dispatch")
        for name in names:
            if name not in EMBODIMENTS:
                raise KeyError(f"unregistered embodiment {name!r}")
        self.proprio_proj = nn.ModuleDict(
            {name: nn.Linear(EMBODIMENTS[name].dof, dim) for name in names}
        )
        #: dof -> embodiment name, for the single-candidate inference path
        self._dof_index: dict[int, list[str]] = {}
        for name in names:
            self._dof_index.setdefault(EMBODIMENTS[name].dof, []).append(name)

        # ── token tags ────────────────────────────────────────────────────
        self.stream_embed = nn.Parameter(torch.empty(max_streams, dim))
        self.type_embed = nn.Parameter(torch.empty(4, dim))   # view/proprio/lang/z_prev

        # ── trunk ─────────────────────────────────────────────────────────
        self.ctx_ln = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList(
            PerceiverBlock(dim, heads, mlp_ratio) for _ in range(depth)
        )
        self.out_ln = nn.LayerNorm(dim)

        self._init_weights(depth)

        # `loop._try_build` catches a TypeError from `cls(**kwargs)` and retries
        # `cls()`, so a misspelled `model.estimator.*` key silently builds the
        # SHIPPED estimator and the whole arm is a null. Nothing else in the run
        # would say so. One line in the log, only when the recurrence is not the
        # default, is the difference between an ablation and a wasted 16 GPUs.
        if not self.z_prev_residual or self.z_init is not None:
            print(f"[estimator] recurrence: z_prev_residual={self.z_prev_residual} "
                  f"learned_z_init={self.z_init is not None}", flush=True)

    # ── init ──────────────────────────────────────────────────────────────

    def _init_weights(self, depth: int) -> None:
        embeds = [self.latents, self.slot_embed, self.stream_embed, self.type_embed]
        if self.z_init is not None:
            embeds.append(self.z_init)
        for p in embeds:
            nn.init.normal_(p, std=0.02)
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.normal_(mod.weight, std=0.02)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
        # GPT-2 style: damp the residual branches so 4*depth additions do not
        # blow the activation scale before the LNs have adapted.
        scale = 1.0 / math.sqrt(2.0 * depth)
        with torch.no_grad():
            for blk in self.blocks:
                blk.xattn.o.weight.mul_(scale)
                blk.sattn.o.weight.mul_(scale)
                blk.mlp1[-1].weight.mul_(scale)
                blk.mlp2[-1].weight.mul_(scale)
            # z_prev is added straight onto the queries, so its projection must
            # come out at the same scale as `latents` (std 0.02). At the default
            # std=0.02 the projection of a LayerNormed input has std
            # 0.02*sqrt(d) ~ 0.55, which would drown the learned latents by 25x
            # at step 0 and make the estimator a pass-through of z_prev.
            #
            # Kept unconditionally, including when `z_prev_residual=False`, so
            # the two arms differ in exactly one line. It costs nothing on the
            # context path: `ctx_ln` LayerNorms every token independently, so
            # only zp's size *relative to* `type_embed[3] + slot_embed` survives,
            # and at std 0.02 it is on par with both.
            self.z_prev_proj.weight.mul_(1.0 / math.sqrt(self.dim))

    # ── dispatch ──────────────────────────────────────────────────────────

    def _resolve(self, proprio: Tensor, embodiment: str | None) -> str:
        """Pick the proprio projection. Batches are embodiment-homogeneous."""
        if embodiment is not None:
            if embodiment not in self.proprio_proj:
                raise KeyError(
                    f"no proprio projection for {embodiment!r}; the estimator was built "
                    f"with {sorted(self.proprio_proj)}"
                )
            return embodiment
        dof = int(proprio.shape[-1])
        cands = self._dof_index.get(dof, [])
        if len(cands) == 1:
            return cands[0]
        if not cands:
            raise KeyError(
                f"proprio has dof {dof}, which matches none of {sorted(self.proprio_proj)}; "
                f"pass embodiment=... explicitly"
            )
        raise KeyError(
            f"dof {dof} is ambiguous between {cands}; pass embodiment=... explicitly"
        )

    # ── forward ───────────────────────────────────────────────────────────

    def _context(self, feats: ObsFeats, zp: Tensor | None, embodiment: str | None) -> Tensor:
        views, proprio, lang = feats["views"], feats["proprio"], feats["lang"]
        if views.ndim != 4:
            raise ValueError(f"views must be (B, V, P, F), got {tuple(views.shape)}")
        b, v, p, f = views.shape
        if f != self.feat_dim:
            raise ValueError(
                f"views has feature width {f}, estimator was built for feat_dim="
                f"{self.feat_dim}"
            )
        if v > self.max_streams:
            raise ValueError(
                f"{v} view streams exceeds max_streams={self.max_streams}"
            )

        tv = self.view_proj(views)                                       # (B,V,P,d)
        tv = tv + self.stream_embed[:v].view(1, v, 1, self.dim) + self.type_embed[0]
        tv = tv.reshape(b, v * p, self.dim)

        name = self._resolve(proprio, embodiment)
        tp = self.proprio_proj[name](proprio).unsqueeze(1) + self.type_embed[1]

        tl = self.lang_proj(lang) + self.type_embed[2]                   # (B,L,d)

        parts = [tv, tp, tl]
        if zp is not None:
            parts.append(zp + self.type_embed[3] + self.slot_embed)
        return self.ctx_ln(torch.cat(parts, dim=1))

    def forward(
        self,
        feats: ObsFeats,
        z_prev: Tensor | None = None,
        embodiment: str | None = None,
    ) -> Tensor:
        """`ObsFeats`, optional `(B, K, D)` -> `(B, K, D)`.

        `embodiment` selects the proprio projection. When omitted it is inferred
        from `proprio.shape[-1]`, which is unambiguous whenever no two registered
        bodies share a dof; pass it explicitly otherwise.
        """
        b = feats["views"].shape[0]

        if z_prev is None and self.z_init is not None:
            # no episode-start seam: t=0 runs the same forward as every other t,
            # with a learned belief standing in for "what was there before".
            z_prev = self.z_init.unsqueeze(0).expand(b, -1, -1)

        zp = None
        if z_prev is not None:
            if z_prev.shape[-2:] != (self.n_slots, self.dim):
                raise ValueError(
                    f"z_prev must be (B, {self.n_slots}, {self.dim}), "
                    f"got {tuple(z_prev.shape)}"
                )
            zp = self.z_prev_proj(self.z_prev_ln(z_prev))

        ctx = self._context(feats, zp, embodiment)

        x = self.latents.unsqueeze(0).expand(b, -1, -1)
        if zp is not None and self.z_prev_residual:
            x = x + zp                       # belief persistence, slot-aligned
        slot = self.slot_embed.unsqueeze(0)

        use_ckpt = self.grad_checkpoint and self.training and torch.is_grad_enabled()
        for blk in self.blocks:
            if use_ckpt:
                x = checkpoint(blk, x, ctx, slot, use_reentrant=False)
            else:
                x = blk(x, ctx, slot)
        return self.out_ln(x)

    def extra_repr(self) -> str:
        return (
            f"feat_dim={self.feat_dim}, K={self.n_slots}, D={self.dim}, "
            f"blocks={len(self.blocks)}, grad_checkpoint={self.grad_checkpoint}, "
            f"z_prev_residual={self.z_prev_residual}, "
            f"learned_z_init={self.z_init is not None}"
        )
