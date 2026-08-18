"""Optimiser, LR schedule, EMA target, freeze schedule.

Everything here is **step-based**. Nothing may read the wall clock or an epoch
count: a schedule derived from wall time silently changes behaviour across a 4 h
requeue boundary, which is the hardest class of bug to see in a loss curve.

PLAN 4.D: AdamW, cosine, 2k warmup, grad clip 1.0, and **bank parameters at 10x
lower LR than the estimator** because the spectral parameters (``log_r``,
``omega``) are sensitive -- a step that is fine for a Perceiver block moves a
pole across half the unit disc.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch
from torch import nn

from contracts import EMA_TAU

__all__ = [
    "CosineWithWarmup", "BANK_LR_MULT", "param_groups", "build_optimizer",
    "clip_grad", "module_grad_norms", "SpikeGuard", "EMATarget", "FreezeSchedule",
]

#: Bank LR is exactly this multiple of the estimator LR. Locked by a test.
BANK_LR_MULT = 0.1

#: Parameter-name fragments that must not be weight-decayed.
# "embed" does NOT match `step_emb` or `slot_emb`, so seven embedding tensors
# across q_action, q_delta and the decoder were being weight-decayed. Against the
# real CosineWithWarmup(3e-4, 2000, 60000) that is a 4.4% multiplicative shrink by
# step 4000, 8.6% by 7004, and 37.6% over the full schedule -- noise for a probe,
# material for a real run. "_emb" matches exactly those seven and nothing else;
# no parameter name contains "embodiment".
_NO_DECAY = ("bias", "norm", "ln_", "layernorm", "embed", "_emb", "log_r", "omega")


# ═══════════════════════════════════════════════════════════════════════════
#  LR SCHEDULE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CosineWithWarmup:
    """Linear warmup then cosine decay, as a pure function of the step.

    ``total_steps`` is the *schedule horizon* and is identical on every link of a
    chained run. ``--stop_at`` truncates a link; it must never appear here.
    """

    base_lr: float
    warmup_steps: int
    total_steps: int
    min_lr_ratio: float = 0.05

    def scale_at(self, step: int) -> float:
        if step < self.warmup_steps:
            return (step + 1) / max(1, self.warmup_steps)
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cos

    def lr_at(self, step: int) -> float:
        return self.base_lr * self.scale_at(step)

    def apply(self, optimizer, step: int) -> dict[str, float]:
        """Set every group's LR to ``base_lr * group.lr_scale * scale_at(step)``.

        Returns ``{group_name: lr}`` so the caller can log per-module LRs and a
        test can assert the bank/estimator ratio.
        """
        s = self.scale_at(step)
        out: dict[str, float] = {}
        for g in optimizer.param_groups:
            g["lr"] = self.base_lr * g.get("lr_scale", 1.0) * s
            out[g.get("name", "group")] = g["lr"]
        return out

    def state_dict(self) -> dict:
        return {"base_lr": self.base_lr, "warmup_steps": self.warmup_steps,
                "total_steps": self.total_steps, "min_lr_ratio": self.min_lr_ratio}

    def load_state_dict(self, sd: dict) -> None:
        for k, v in sd.items():
            setattr(self, k, v)


# ═══════════════════════════════════════════════════════════════════════════
#  PARAMETER GROUPS
# ═══════════════════════════════════════════════════════════════════════════

def _module_of(name: str, module_names: Sequence[str]) -> str:
    """Map a fully qualified parameter name onto its owning top-level module."""
    head = name.split(".", 1)[0]
    return head if head in module_names else "other"


def param_groups(model: nn.Module, *, weight_decay: float = 0.05,
                 lr_scales: dict[str, float] | None = None,
                 module_names: Sequence[str] | None = None) -> list[dict]:
    """Per-module LR groups. The bank gets ``BANK_LR_MULT`` by default.

    Groups are named ``"<module>/<decay|nodecay>"`` so the LR of each module is
    loggable and assertable. Frozen modules still get a group -- freezing is a
    ``requires_grad`` schedule, not a group edit, so that unfreezing at 30% does
    not require rebuilding the optimizer and losing its state.
    """
    lr_scales = dict(lr_scales or {})
    lr_scales.setdefault("bank", BANK_LR_MULT)
    if module_names is None:
        module_names = [n for n, _ in model.named_children()]

    buckets: dict[tuple[str, bool], list[nn.Parameter]] = {}
    for name, p in model.named_parameters():
        mod = _module_of(name, module_names)
        decay = weight_decay > 0.0 and p.ndim > 1 and not any(t in name.lower() for t in _NO_DECAY)
        buckets.setdefault((mod, decay), []).append(p)

    groups = []
    for (mod, decay), params in sorted(buckets.items(), key=lambda kv: (kv[0][0], not kv[0][1])):
        groups.append({
            "name": f"{mod}/{'decay' if decay else 'nodecay'}",
            "module": mod,
            "params": params,
            "weight_decay": weight_decay if decay else 0.0,
            "lr_scale": lr_scales.get(mod, 1.0),
        })
    return groups


def build_optimizer(model: nn.Module, *, lr: float, weight_decay: float = 0.05,
                    betas: tuple[float, float] = (0.9, 0.95),
                    lr_scales: dict[str, float] | None = None,
                    module_names: Sequence[str] | None = None):
    groups = param_groups(model, weight_decay=weight_decay, lr_scales=lr_scales,
                          module_names=module_names)
    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=1e-8)


def clip_grad(model: nn.Module, max_norm: float = 1.0, sync=None) -> float:
    """Global grad-norm clip across a mixed sharded/replicated model.

    The naive ``clip_grad_norm_`` is wrong here in two different ways at once:
    the estimator's gradients are FSDP-sharded, so each rank sees only a slice of
    that sum, while the replicated gradients are already identical on every rank
    (``ReplicaSync.all_reduce_grads`` ran first) and must NOT be counted once per
    rank. So: all-reduce only the sharded contribution, add the replicated one
    locally, and scale everything by the one global factor.

    ``sync=None`` (no distributed) degrades to the ordinary global norm, which is
    the same number.
    """
    if max_norm is None or max_norm <= 0:
        return 0.0
    params = [p for p in model.parameters() if p.grad is not None]
    if not params:
        return 0.0

    if sync is None or not getattr(sync, "enabled", False):
        return float(torch.nn.utils.clip_grad_norm_(params, max_norm))

    import torch.distributed as dist

    sharded = {id(p) for p in sync.sharded_params()}
    dev = params[0].grad.device

    def _sq(ps):
        if not ps:
            return torch.zeros((), device=dev, dtype=torch.float32)
        return torch.stack([p.grad.detach().float().pow(2).sum() for p in ps]).sum()

    sq_sharded = _sq([p for p in params if id(p) in sharded])
    dist.all_reduce(sq_sharded, op=dist.ReduceOp.SUM)
    total = torch.sqrt(sq_sharded + _sq([p for p in params if id(p) not in sharded]))

    coef = max_norm / (float(total) + 1e-6)
    if coef < 1.0:
        for p in params:
            p.grad.detach().mul_(coef)
    return float(total)


# ═══════════════════════════════════════════════════════════════════════════
#  PER-MODULE GRAD NORMS
# ═══════════════════════════════════════════════════════════════════════════

def module_grad_norms(model: nn.Module, sync=None,
                      module_names: Sequence[str] | None = None
                      ) -> dict[str, float]:
    """Pre-clip grad norm of every top-level child, in ONE small collective.

    ``clip_grad`` reports a single global number, which is exactly the number
    that cannot tell you anything: a run whose global norm jumps from 2 to 10252
    has one module doing that, and the global norm is the one statistic that
    hides which. This reduces a length-``n_modules`` vector instead of a scalar,
    so it costs one extra all-reduce of ~7 floats per step -- the same latency
    class as the reduce ``clip_grad`` already does, and nothing measurable
    against a 16-GPU step.

    Call it BEFORE ``clip_grad``; afterwards every number is scaled by ``coef``
    and the decomposition is still correct but no longer in the units the spike
    happened in.

    Sharded (FSDP) parameters hold a slice of their gradient per rank, so their
    squares must be summed across ranks; replicated parameters are bit-identical
    on every rank after ``ReplicaSync.all_reduce_grads`` and must be counted
    once. Same split as ``clip_grad``, so ``sqrt(sum of squares)`` over the
    returned dict reproduces ``clip_grad``'s total.
    """
    if module_names is None:
        module_names = [n for n, _ in model.named_children()]
    names = list(module_names)
    if "other" not in names:
        names = names + ["other"]
    index = {n: i for i, n in enumerate(names)}

    named = [(n, p) for n, p in model.named_parameters() if p.grad is not None]
    if not named:
        return {}
    dev = named[0][1].grad.device

    enabled = sync is not None and getattr(sync, "enabled", False)
    sharded_ids = {id(p) for p in sync.sharded_params()} if enabled else set()

    sq_sh = torch.zeros(len(names), device=dev, dtype=torch.float32)
    sq_rep = torch.zeros(len(names), device=dev, dtype=torch.float32)
    for name, p in named:
        i = index[_module_of(name, names)]
        s = p.grad.detach().float().pow(2).sum()
        if id(p) in sharded_ids:
            sq_sh[i] += s
        else:
            sq_rep[i] += s

    if enabled:
        import torch.distributed as dist

        dist.all_reduce(sq_sh, op=dist.ReduceOp.SUM)

    total_sq = (sq_sh + sq_rep).cpu()
    return {n: float(total_sq[i].clamp_min(0).sqrt()) for n, i in index.items()
            if float(total_sq[i]) > 0.0}


# ═══════════════════════════════════════════════════════════════════════════
#  SPIKE REJECTION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SpikeGuard:
    """Skip the optimizer step when the pre-clip grad norm is an outlier.

    WHY REJECTION AND NOT CLIPPING
    ──────────────────────────────
    The optimizer is AdamW, and AdamW's update is invariant to a global rescale
    of the gradient once its moments have equilibrated: scaling every ``g`` by
    ``s`` scales ``m`` by ``s`` and ``sqrt(v)`` by ``s``, and ``m / sqrt(v)`` is
    unchanged. ``clip_grad(1.0)`` multiplies a norm-10252 gradient by 1e-4 and
    hands AdamW a *unit-norm gradient pointing the same way*, which AdamW then
    turns into a full learning-rate step. Clipping bounds the magnitude of a
    quantity the optimizer does not use. It cannot bound the step.

    What the spike actually costs is therefore (a) one full-size step along a
    direction estimated from one pathological batch, (b) that direction retained
    in ``m`` for ~``1/(1-beta1)`` = 10 further steps, and (c) every *other*
    module's gradient shrunk by the same 1e-4 for that step, so the healthy
    modules stop learning exactly while the pathological one drives. Not taking
    the step removes all three; clipping removes none.

    THE RULE
    ────────
    A step is rejected when ``gnorm > mult * exp(ema_log)``, i.e. when it exceeds
    a multiple of the running *geometric* mean of the norm. Geometric, because
    the norm is heavy-tailed and log-scaled -- on R0-A2 the per-window median
    moves between 1.5 and 29 over 5k steps, so no absolute threshold is right
    for the whole run, while ``log gnorm`` is comfortably stationary within a
    window.

    The reference is updated on EVERY step, but a rejected step contributes
    ``log(threshold)`` rather than its own value: bounded influence, not
    exclusion. Both halves of that are load bearing, and the exclusion variant
    is not merely worse, it deadlocks -- measured, run ``tdgradB``:

      * Updating only on ACCEPTED steps looks obviously right and is fatal. Once
        the gradient regime shifts above the threshold, nothing is accepted, so
        nothing updates the reference, so the threshold can never follow, so
        nothing is ever accepted again. tdgradB skipped 457 consecutive steps
        with ``gnorm`` pinned at 93 and the threshold frozen at 23.35: the
        weights stopped changing at step ~2144 and the job burned its remaining
        walltime doing forward and backward passes it then threw away. Nothing
        in the loss curve says "halted" -- ``loss`` just goes flat.
      * Updating with the RAW value on a skip is the opposite failure: a burst
        of 10^4 spikes walks the threshold up behind itself and re-admits the
        next one.

      ``min(gnorm, threshold)`` gives a guard that un-sticks geometrically -- a
      skipped step raises the threshold by exactly ``mult**(1-beta)`` (4.7% at
      the defaults), so a genuine regime shift of 4x is re-admitted after ~30
      steps, while a lone spike ratchets the bar by 4.7% and no more.

    STATE, AND WHY IT IS IN THE CHECKPOINT
    ──────────────────────────────────────
    ``(ema_log, n)`` is optimizer state in every sense that matters, so it rides
    in the checkpoint payload. A guard that reset at every 4 h link boundary
    would be a schedule derived from wall clock by the back door: the first
    ``warmup`` steps of each link would be unguarded, and the run would behave
    differently depending on where SLURM happened to cut it.

    The decision is a pure function of the *global* norm, which ``clip_grad``
    has already all-reduced, so every rank decides identically without an extra
    collective. It must stay that way: a guard that could skip on one rank and
    not another desynchronises the optimizer across the world.
    """

    mult: float = 8.0
    beta: float = 0.98
    warmup: int = 100
    ema_log: float = 0.0
    n: int = 0
    #: Cap on how far the reference may climb within ONE unbroken run of skipped
    #: steps, relative to where it stood when the burst began. See `check`.
    #: 10x is generous against the legitimate case the class docstring names --
    #: a genuine 4x regime shift is re-admitted after ~30 skips -- while ruling
    #: out the 171x runaway R0-B measured.
    burst_cap: float = 10.0
    #: Reference at the start of the current skip run; None when not in one.
    _burst_ref: float | None = None

    @property
    def enabled(self) -> bool:
        return self.mult > 0.0

    @property
    def threshold(self) -> float:
        """Current rejection threshold, or ``inf`` while still warming up."""
        if not self.enabled or self.n < self.warmup:
            return float("inf")
        return self.mult * math.exp(self.ema_log)

    def check(self, gnorm: float) -> bool:
        """Update the reference and return True if this step must be SKIPPED.

        Non-finite norms are always rejected, and do not move the reference: a
        NaN gradient must never reach the moments, where one step of ``m``
        poisons every subsequent update for the rest of the run regardless of
        what the data does, and ``log(nan)`` would destroy the reference too.
        """
        if not self.enabled:
            return False
        if not math.isfinite(gnorm):
            return True
        thr = self.threshold
        skip = gnorm > thr
        # Bounded influence: a rejected step still moves the reference, but only
        # as far as the threshold it failed. See the class docstring -- excluding
        # it entirely deadlocks the guard, and admitting it raw defeats it.
        ref = min(gnorm, thr) if math.isfinite(thr) else gnorm
        self.ema_log = (self.beta * self.ema_log
                        + (1.0 - self.beta) * math.log(max(ref, 1e-12)))

        # BURST CAP. `min(gnorm, thr)` bounds what ONE skipped step contributes,
        # but not what a sustained burst does: each skip raises the threshold by
        # mult**(1-beta) = 4.7% at the defaults, and 1.047**170 is ~2600x. R0-B
        # measured a 171x ratchet (threshold 136 -> 23334) across a single burst
        # near step 1000, after which 25 steps with gnorm > 1000 were ACCEPTED,
        # the largest at 15958, in a run whose healthy gnorm is 10-40. The guard
        # did not fail to fire; it adapted until the explosion was inside its own
        # definition of normal, and the coefficient space never recovered
        # (live_ops_q_a 29 -> 13, loss/proposal pinned at the 19.361 uniform
        # floor).
        #
        # So the reference may still climb during a burst -- that is what stops
        # the deadlock the class docstring documents (tdgradB, 457 consecutive
        # skips) -- but not without bound. `burst_cap` is generous against the
        # legitimate case the docstring names: a genuine 4x regime shift is
        # re-admitted after ~30 skips, i.e. 4x, well inside the cap.
        if skip:
            if self._burst_ref is None:
                self._burst_ref = math.exp(self.ema_log)
            ceiling = math.log(self._burst_ref * self.burst_cap)
            if self.ema_log > ceiling:
                self.ema_log = ceiling
        else:
            self._burst_ref = None

        self.n += 1
        return skip

    def state_dict(self) -> dict:
        return {"mult": self.mult, "beta": self.beta, "warmup": self.warmup,
                "ema_log": self.ema_log, "n": self.n,
                "burst_ref": self._burst_ref}

    def load_state_dict(self, sd: dict) -> None:
        # mult/beta/warmup come from the config, not the checkpoint: changing the
        # guard between links is a legitimate intervention, resuming into the old
        # value silently is not.
        self.ema_log = float(sd.get("ema_log", self.ema_log))
        self.n = int(sd.get("n", self.n))
        br = sd.get("burst_ref")
        self._burst_ref = None if br is None else float(br)


# ═══════════════════════════════════════════════════════════════════════════
#  EMA TARGET
# ═══════════════════════════════════════════════════════════════════════════

#: Wrapper prefixes that FSDP / activation checkpointing / torch.compile insert
#: into parameter names. The online and target estimators are wrapped separately,
#: so their raw names differ even though the tensors correspond one to one.
_WRAPPER_FRAGMENTS = ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module.",
                      "_orig_mod.", "module.")


def _norm_param_name(name: str) -> str:
    for frag in _WRAPPER_FRAGMENTS:
        name = name.replace(frag, "")
    return name


class EMATarget(nn.Module):
    """The frozen target estimator that ``L_dyn`` regresses onto.

    ``tau = contracts.EMA_TAU``. Held as a full module rather than a shadow dict
    because ``L_dyn`` needs to *run* it (``z_bar = E_target(o)``), not just read
    its weights.

    Under FSDP the target is wrapped with the same policy as the online module,
    so ``named_parameters()`` yields matching local shards and the lerp is
    shard-wise correct without any all-gather. The names differ by wrapper
    prefixes, hence :func:`_norm_param_name`; a mismatch that silently matched
    nothing would leave the target frozen at initialisation forever and ``L_dyn``
    would look like it was training.
    """

    def __init__(self, src: nn.Module, tau: float = EMA_TAU):
        super().__init__()
        import copy

        self.register_buffer("tau_buf", torch.tensor(float(tau)))
        self.module = copy.deepcopy(src)
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.module.eval()

    @property
    def tau(self) -> float:
        return float(self.tau_buf)

    def forward(self, *a, **kw):
        return self.module(*a, **kw)

    @torch.no_grad()
    def update(self, src: nn.Module) -> None:
        tau = self.tau
        tgt = {_norm_param_name(n): p for n, p in self.module.named_parameters()}
        matched = 0
        for name, p in src.named_parameters():
            t = tgt.get(_norm_param_name(name))
            if t is None or t.shape != p.shape:
                continue
            t.lerp_(p.detach().to(t.dtype), 1.0 - tau)
            matched += 1
        if matched == 0 and any(True for _ in src.parameters()):
            raise RuntimeError(
                "EMATarget.update matched no parameters: the target would stay at "
                "its initialisation forever while L_dyn looked healthy. Names are "
                f"{list(tgt)[:3]} vs {[n for n, _ in src.named_parameters()][:3]}."
            )
        tbuf = {_norm_param_name(n): b for n, b in self.module.named_buffers()}
        for name, b in src.named_buffers():
            t = tbuf.get(_norm_param_name(name))
            if t is not None and t.shape == b.shape:
                t.copy_(b.detach())


# ═══════════════════════════════════════════════════════════════════════════
#  FREEZE SCHEDULE  (R2 warm-up)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FreezeSchedule:
    """R2 warm-up: freeze ``E`` **and** the bank together for the first 30%.

    Freezing the bank alone does not preserve the coefficient space -- the
    operators would be fixed vectors in a drifting basis. With both frozen,
    ``q_a`` and ``D_e`` are forced into the coordinate system R1 established,
    which is what lets us drop an explicit alignment loss (PLAN 7).

    Empty ``modules`` (every run except R2) makes this a no-op.
    """

    modules: tuple[str, ...] = ()
    until_frac: float = 0.0
    total_steps: int = 1

    def __post_init__(self) -> None:
        self.modules = tuple(self.modules)
        if self.modules and not (0.0 <= self.until_frac <= 1.0):
            raise ValueError(f"until_frac must be in [0, 1], got {self.until_frac}")
        if self.modules and "bank" in self.modules and "estimator" not in self.modules:
            raise ValueError(
                "freezing the bank without the estimator leaves the operators as "
                "fixed vectors in a drifting basis (PLAN 7); freeze both or neither"
            )

    @property
    def until_step(self) -> int:
        return int(round(self.until_frac * self.total_steps))

    def frozen_at(self, step: int) -> bool:
        return bool(self.modules) and step < self.until_step

    def apply(self, model: nn.Module, step: int, trainable: Iterable[str]) -> bool:
        """Set ``requires_grad`` on the scheduled modules. Returns "is frozen now".

        ``trainable`` is the config's ``train_modules``: a module absent from it
        is never unfrozen, so R1 does not accidentally start training ``q_a``.
        """
        frozen = self.frozen_at(step)
        trainable = set(trainable)
        for name in self.modules:
            sub = getattr(model, name, None)
            if sub is None:
                continue
            want = (not frozen) and (name in trainable)
            for p in sub.parameters():
                p.requires_grad_(want)
        return frozen
