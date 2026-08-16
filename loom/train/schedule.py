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
    "clip_grad", "EMATarget", "FreezeSchedule",
]

#: Bank LR is exactly this multiple of the estimator LR. Locked by a test.
BANK_LR_MULT = 0.1

#: Parameter-name fragments that must not be weight-decayed.
_NO_DECAY = ("bias", "norm", "ln_", "layernorm", "embed", "log_r", "omega")


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
