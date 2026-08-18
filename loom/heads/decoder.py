"""LOOM — D_e: the body-specific realizer.  ONE PER EMBODIMENT, ~18 M each.

`D_e(proprio (B,dof_e), c (B,M)) -> (B, H_OP, dof_e)`.

ONE OPERATOR = ONE 8-STEP SEGMENT. NEVER H_PLAN=32. The plan emits DEPTH
operators; only the root one is ever decoded and executed, and it is decoded
into `H_OP` control steps. `contracts.assert_action_segment` is the guard.

No pixel decoding, no VAE, no video DiT (PLAN 9). The only generative object in
this repo is this 56-number action segment.


THE BELIEF IS NOT AN INPUT  (owner-authorised contract change)
─────────────────────────────────────────────────────────────
This head used to take `(z (B,K,D), c (B,M))`. It does not any more, and the
reason is the whole point of the architecture rather than a capacity tweak.

Given the full 128x768 belief, predicting an 8x7 action segment is behaviour
cloning — and behaviour cloning needs nothing whatsoever from `c`. Measured on
R0-A: `act/decode` fell 0.2489 -> 0.0559 (4.5x) over 7000 steps while `c_a`
held only 2-3 distinct top-4 supports across 64 real training windows. `L_act`
was descending entirely through the belief path and exerting no pressure on the
coefficient at all, which is exactly the failure `L_act` exists to prevent.

With `z` gone, `c` is the ONLY channel carrying task information into the
action. `proprio` is `ObsFeats["proprio"]`, `(B, dof_e)` — ONE timestep of the
body's own state (for `libero_franka`: ee position, ee orientation as an
axis-angle, one gripper coordinate). It tells the realizer where the arm *is*;
it cannot tell it where the target is, so it cannot substitute for `c`.

**Expect `L_act` to rise.** LIBERO is OSC end-effector delta control and the
target position lives only in the image, so a proprio-only decoder genuinely
loses information a belief-conditioned one had. `act/decode` plateauing well
above 0.0559 *while* `c_a` diversifies is the bottleneck being tight, not the
idea failing. Do not restore the belief and do not open a visual channel here
on your own initiative.


CONDITIONAL FLOW MATCHING — exact parameterisation
──────────────────────────────────────────────────
Rectified-flow / optimal-transport CFM with a Gaussian source:

    x_0 ~ N(0, I)                       source, same shape as a_seg
    x_1 = a_seg                         data
    t   ~ U(0, 1)                       one t per sample (not per step)
    x_t = (1 - t) * x_0 + t * x_1       straight conditional path
    u_t = x_1 - x_0                     conditional target velocity (constant in t)

    L_act = E_{t,x_0,(p,c)} || v_theta(x_t, t, p, c) - u_t ||^2      (mean over H_OP*dof)

`forward` integrates the probability-flow ODE `dx/dt = v_theta(x, t, p, c)` from
`x(0) = x_0 ~ N(0,I)` to `x(1)` with `n_steps` *fixed* forward-Euler steps
(default 10, constructor arg `n_steps`):

    x <- x + (1/n) * v_theta(x, i/n, p, c),   i = 0 .. n-1

Euler and not Heun/midpoint: with a straight conditional path the learned field
is close to constant along a trajectory, the error is O(1/n) on a 56-dim state,
and the eval loop runs this at 3.75 Hz with a fixed compute budget — a fixed
small step count is worth more than adaptive accuracy.

Conditioning is adaLN-Zero (DiT-style) over a length-`H_OP` token sequence: one
token per control step, so the network can shape the *within-segment* profile
rather than emitting 8 independent samples. Every block gate is zero at init, so
`v_theta ~ 0` and `forward` returns approximately pure noise; that is the
intended cold start. (The final projection is small-init rather than zero-init —
see the note at its initialisation.)


ACTION RANGE
────────────
`EmbodimentSpec.action_low/high` are registered as buffers.

* Inputs (`a_seg` in `loss`) are ASSUMED PRE-NORMALISED by the data pipeline
  into that range. We do NOT clamp or rescale the target: clamping the data
  side of a flow-matching pair biases the velocity target and quietly teaches
  the field to push out of the box near the boundary. `q_action.py` records the
  seam where a per-dof rescale would have to go (`loss` divides, `forward`
  multiplies back BEFORE the clamp) and why it is deliberately not taken here —
  dropping `z` does not change that, and nothing in this file normalises
  `a_seg`, so `loom/eval/policy.py` still never sees normalised units.
* Outputs of `forward` ARE clamped to `[low, high]` by default (`clamp=True`).
  The ODE integrates a Gaussian source, so a few percent of samples land
  outside the box; the environment would clip them anyway, and clipping in a
  known place beats clipping in robosuite. Pass `clamp=False` to inspect the
  raw sample (the tests do both).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

# `D` / `K` survive only as the defaults of the two accepted-and-ignored belief
# kwargs (`d_belief`, `n_slots`). There is no belief pooling in this head any
# more, so `q_delta.AttnPool` is no longer imported.
from contracts import D, EMBODIMENTS, H_OP, K, M

__all__ = ["DecoderBody", "Decoder"]


# ═══════════════════════════════════════════════════════════════════════════
#  RESIDUAL TARGET  (`residual=True`, OFF by default)
# ═══════════════════════════════════════════════════════════════════════════
#
# On a body whose action semantics are ABSOLUTE, `a_seg ~ proprio` repeated is
# already most of the answer: measured on all 458 560 RoboTwin segments, copying
# `proprio_t` across the 8 steps explains **99.03%** of `a_seg`'s variance
# (per-dof R^2 97.91-99.23%; 1 - mean(a_seg - proprio_t)^2 / var(a_seg), from
# logs/rt_actstats/raw.json). The decoder therefore reaches 99% of its target with
# ZERO information from `c`, which is the same disease that dropping `z` cured --
# `L_act` stops putting pressure on the coefficient. Measured on RoboTwin
# (runs/r0b_sanity, 16 GPUs): `act/decode` 0.85 -> 0.026 by step 3000 while
# `gnorm/q_action` fell to 0.0031 (LIBERO's r0a_flip: 0.044, 14x higher), the
# routing collapsed onto the load-balance objective (`loss/balance` 1.05 against
# its perfectly-uniform floor of 1.0) and `loss/proposal` pinned at the uniform
# Plackett-Luce value 19.3608 -- i.e. `pi_c`, the only head that runs at
# inference, learned nothing at all. `libero_franka` does not have this problem:
# its semantics are `('delta',)*6 + ('hold',)`, so proprio predicts nothing.
#
# With `residual=True` the flow's data side is `(a_seg - proprio_t) / rms`, so
# 100% of what the field has to explain must arrive through `c`. Both halves are
# here and only here: `loss` subtracts and divides, `forward` multiplies and adds
# back BEFORE the action_low/high clamp, so `loom/eval/policy.py` still receives
# absolute radians and nothing outside this file changes units. Half of this
# change is far worse than none -- an eval that rebuilds the body without the
# flag would read residuals as joint targets.
#
# `RESIDUAL_RMS` is the per-dof rms of `a_seg - proprio_t` over the whole cache
# (2500 demo_clean trajectories resampled to 30 Hz, 458 560 segments;
# logs/rt_actstats/raw.json, sqrt(res_sumsq / (nseg * H_OP))). An unregistered
# body gets 1.0 -- a plain residual with no rescale.
RESIDUAL_RMS: dict[str, tuple[float, ...]] = {
    # [L_j1..L_j6, L_grip | R_j1..R_j6, R_grip]
    "robotwin_aloha": (0.035389, 0.091560, 0.078513, 0.069049, 0.025436,
                       0.053096, 0.065479,
                       0.032432, 0.089284, 0.078586, 0.065594, 0.025165,
                       0.054703, 0.066018),
}


# ═══════════════════════════════════════════════════════════════════════════
#  BUILDING BLOCKS
# ═══════════════════════════════════════════════════════════════════════════

def timestep_embedding(t: Tensor, dim: int, max_period: float = 1e4) -> Tensor:
    """(B,) in [0,1] -> (B, dim) sinusoidal features. Scaled by 1000 as is
    conventional for continuous-time diffusion/flow models."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    ang = (1000.0 * t.float())[:, None] * freqs[None]
    emb = torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb.to(t.dtype)


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """Pre-LN self-attention + MLP over the H_OP token axis, adaLN-Zero conditioned."""

    def __init__(self, d: int, n_heads: int, mlp_ratio: float = 3.0) -> None:
        super().__init__()
        if d % n_heads:
            raise ValueError(f"d={d} not divisible by n_heads={n_heads}")
        self.n_heads, self.d_head = n_heads, d // n_heads
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False)
        hidden = int(d * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))
        self.ada = nn.Linear(d, 6 * d)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)          # adaLN-Zero: identity at init

    def _attn(self, x: Tensor) -> Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.n_heads, self.d_head).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        att = torch.softmax(q @ k.transpose(-1, -2) * self.d_head ** -0.5, dim=-1)
        return self.proj((att @ v).transpose(1, 2).reshape(b, n, d))

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        s1, c1, g1, s2, c2, g2 = self.ada(cond).chunk(6, dim=-1)
        x = x + g1.unsqueeze(1) * self._attn(_modulate(self.norm1(x), s1, c1))
        x = x + g2.unsqueeze(1) * self.mlp(_modulate(self.norm2(x), s2, c2))
        return x


# ═══════════════════════════════════════════════════════════════════════════
#  PER-EMBODIMENT DECODER
# ═══════════════════════════════════════════════════════════════════════════

class DecoderBody(nn.Module):
    """CFM velocity field + Euler sampler for one body. ~18 M.

    `n_queries` / `pool_heads` / `d_belief` / `n_slots` / `d_kv` are accepted
    and ignored. They described the belief pooling that this head no longer
    has; they stay in the signature so a `model.decoder:` block in an existing
    config still constructs, rather than dying with an opaque TypeError inside
    `loop._try_build`'s fallback path.
    """

    def __init__(
        self,
        embodiment: str,
        d: int = 512,
        n_blocks: int = 4,
        n_heads: int = 8,
        mlp_ratio: float = 3.0,
        n_queries: int = 4,
        pool_heads: int = 8,
        n_steps: int = 10,
        clamp: bool = True,
        residual: bool = False,
        d_belief: int = D,
        n_ops: int = M,
        n_slots: int = K,
        h_op: int = H_OP,
        d_kv: int | None = None,
    ) -> None:
        super().__init__()
        if embodiment not in EMBODIMENTS:
            raise KeyError(f"unregistered embodiment {embodiment!r}")
        spec = EMBODIMENTS[embodiment]
        self.embodiment, self.dof, self.h_op = embodiment, spec.dof, h_op
        self.n_steps, self.clamp, self.d = n_steps, clamp, d

        self.register_buffer("action_low", torch.tensor(spec.action_low), persistent=False)
        self.register_buffer("action_high", torch.tensor(spec.action_high), persistent=False)

        # Residual target. See RESIDUAL_RMS above. Non-persistent, like the
        # action box: a fixed reparameterisation, never carried in a checkpoint,
        # so `residual` must come from the run config on BOTH sides.
        self.residual = bool(residual)
        rms = RESIDUAL_RMS.get(embodiment, (1.0,) * spec.dof)
        if len(rms) != spec.dof:
            raise ValueError(
                f"RESIDUAL_RMS[{embodiment!r}] has {len(rms)} entries, dof is {spec.dof}")
        self.register_buffer("residual_rms", torch.tensor(rms), persistent=False)

        # ── conditioning: proprio, coefficient, time ─────────────────────
        # No belief pooling. See the module docstring: with `z` in here the
        # decoder is a behaviour-cloning head and `c` is decorative.
        # A plain Linear, no LayerNorm: the 7 dofs of `proprio` are
        # heterogeneous (metres, radians, a gripper coordinate) and normalising
        # ACROSS them would subtract a mean with no physical meaning and throw
        # away the absolute ee position, which is the one thing this input has.
        self.p_proj = nn.Linear(self.dof, d)
        self.c_proj = nn.Linear(n_ops, d)
        self.t_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.cond_mlp = nn.Sequential(nn.LayerNorm(d), nn.SiLU(), nn.Linear(d, d))

        # ── velocity field over H_OP tokens ──────────────────────────────
        self.x_in = nn.Linear(self.dof, d)
        # small random, not zeros: the blocks are self-attention over the H_OP
        # tokens, which is permutation-equivariant, so a zero step embedding
        # would leave "control step 3" and "control step 5" indistinguishable
        # except by their noise content.
        # NOT weight-decayed: `schedule.build_optimizer` excludes it by name.
        self.step_emb = nn.Parameter(torch.randn(h_op, d) * 0.02)
        self.blocks = nn.ModuleList(DiTBlock(d, n_heads, mlp_ratio) for _ in range(n_blocks))
        self.norm_out = nn.LayerNorm(d, elementwise_affine=False)
        self.ada_out = nn.Linear(d, 2 * d)
        self.x_out = nn.Linear(d, self.dof)
        # Small-init, NOT zero-init, on the two output layers. The blocks are
        # already adaLN-Zero (identity at init), so if the final projection were
        # also zero the whole conditioning path — belief and coefficients alike —
        # would receive exactly zero gradient on step 0 and only unblock on step
        # 1 via the weight gradients. Small-init keeps v_theta ~ 0 at init while
        # keeping d loss / d z and d loss / d c live from the first batch.
        for m in (self.ada_out, self.x_out):
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    # ── conditioning / field ─────────────────────────────────────────────

    def condition(self, proprio: Tensor, c: Tensor) -> Tensor:
        """(B,dof), (B,M) -> (B,d). Time-independent part; hoisted out of the
        Euler loop so it is built once per segment, not once per Euler step.

        `proprio` must be ONE timestep — `ObsFeats["proprio"]`, `(B, dof_e)`.
        A `(B, H_OP, dof)` argument is an action segment that has been handed
        in by mistake, and it would broadcast into a `(B, H_OP, d)` condition
        that `_modulate`'s `unsqueeze(1)` then silently mis-shapes.
        """
        if proprio.ndim != 2 or proprio.shape[-1] != self.dof:
            raise ValueError(
                f"{self.embodiment}: proprio must be (B, {self.dof}) — one "
                f"timestep of ObsFeats['proprio'] — got {tuple(proprio.shape)}"
            )
        return self.p_proj(proprio) + self.c_proj(c.to(proprio.dtype))

    def velocity(self, x: Tensor, t: Tensor, cond: Tensor) -> Tensor:
        """v_theta(x_t, t, proprio, c). x (B,H_OP,dof), t (B,), cond (B,d)."""
        cond = self.cond_mlp(cond + self.t_mlp(timestep_embedding(t, self.d)))
        h = self.x_in(x) + self.step_emb.to(x.dtype)
        for blk in self.blocks:
            h = blk(h, cond)
        shift, scale = self.ada_out(cond).chunk(2, dim=-1)
        return self.x_out(_modulate(self.norm_out(h), shift, scale))

    # ── contracts.Decoder ────────────────────────────────────────────────

    def forward(
        self,
        proprio: Tensor,
        c: Tensor,
        n_steps: int | None = None,
        clamp: bool | None = None,
        noise: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Integrate the flow from noise. -> (B, H_OP, dof_e). Never H_PLAN."""
        n = int(n_steps or self.n_steps)
        cond = self.condition(proprio, c)
        x = self._noise(proprio.shape[0], proprio, noise, generator)
        dt = 1.0 / n
        for i in range(n):
            t = torch.full((x.shape[0],), i * dt, device=x.device, dtype=x.dtype)
            x = x + dt * self.velocity(x, t, cond)
        if self.residual:
            # BEFORE the clamp: the box is in absolute action units.
            x = x * self.residual_rms.to(x) + proprio.unsqueeze(1).to(x)
        if self.clamp if clamp is None else clamp:
            x = torch.max(torch.min(x, self.action_high.to(x)), self.action_low.to(x))
        return x

    def loss(
        self,
        proprio: Tensor,
        c: Tensor,
        a_seg: Tensor,
        t: Tensor | None = None,
        noise: Tensor | None = None,
        generator: torch.Generator | None = None,
        reduction: str = "mean",
    ) -> Tensor:
        """Conditional flow-matching regression. See the module docstring."""
        a_seg = a_seg.to(proprio.dtype)
        if a_seg.shape[-2:] != (self.h_op, self.dof):
            raise ValueError(
                f"{self.embodiment}: action segment must be (..., {self.h_op}, "
                f"{self.dof}), got {tuple(a_seg.shape)}"
            )
        if self.residual:
            a_seg = (a_seg - proprio.unsqueeze(1)) / self.residual_rms.to(a_seg)
        b = a_seg.shape[0]
        x0 = self._noise(b, a_seg, noise, generator)
        if t is None:
            t = torch.rand(b, device=a_seg.device, dtype=torch.float32, generator=generator)
        t = t.to(a_seg.dtype).reshape(b)

        tt = t.view(b, 1, 1)
        x_t = (1 - tt) * x0 + tt * a_seg
        target = a_seg - x0                        # constant-in-t conditional velocity
        v = self.velocity(x_t, t, self.condition(proprio, c))
        per_sample = (v - target).pow(2).flatten(1).mean(-1)
        if reduction == "none":
            return per_sample
        if reduction == "mean":
            return per_sample.mean()
        raise ValueError(f"unknown reduction {reduction!r}; use 'mean' or 'none'")

    # ── helpers ──────────────────────────────────────────────────────────

    def _noise(self, b: int, like: Tensor, noise: Tensor | None,
               generator: torch.Generator | None) -> Tensor:
        if noise is not None:
            return noise.to(device=like.device, dtype=like.dtype)
        return torch.randn(b, self.h_op, self.dof, device=like.device,
                           dtype=like.dtype, generator=generator)


# ═══════════════════════════════════════════════════════════════════════════
#  DISPATCH
# ═══════════════════════════════════════════════════════════════════════════

class Decoder(nn.Module):
    """`ModuleDict` of per-embodiment `DecoderBody`. Satisfies `contracts.Decoder`.

    Built eagerly from `contracts.EMBODIMENTS` (or an explicit list) for the
    same reason as `QAction`: a lazily-grown `ModuleDict` would add parameters
    after the optimizer and FSDP wrapper already exist.
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

    def add_embodiment(self, name: str) -> DecoderBody:
        if name not in EMBODIMENTS:
            raise KeyError(f"unregistered embodiment {name!r}")
        if name not in self.bodies:
            self.bodies[name] = DecoderBody(name, **self.body_kwargs)
        return self.bodies[name]

    def body(self, embodiment: str | None = None) -> DecoderBody:
        name = embodiment or self.default_embodiment
        if name is None:
            raise ValueError(
                f"decoder holds several bodies; pass embodiment= (have {sorted(self.bodies)})"
            )
        if name not in self.bodies:
            raise KeyError(f"decoder has no body {name!r}; have {sorted(self.bodies)}")
        return self.bodies[name]

    def forward(self, proprio: Tensor, c: Tensor, embodiment: str | None = None,
                **kw) -> Tensor:
        return self.body(embodiment)(proprio, c, **kw)

    def loss(self, proprio: Tensor, c: Tensor, a_seg: Tensor,
             embodiment: str | None = None, **kw) -> Tensor:
        return self.body(embodiment).loss(proprio, c, a_seg, **kw)
