"""FSDP wrapping and the memory budget.

PLAN 4.D / 9:

  * **full-shard on ``E``** (the estimator, ~150 M and the only large module),
  * **bank and heads replicated** -- they are small and every rank needs the
    whole operator bank on every step, so sharding 25 M params buys nothing and
    costs an all-gather inside the hot path,
  * **bf16 throughout. A100 has no FP8; there is no FP8 path here and there
    must never be one.**
  * activation checkpointing on estimator blocks with
    ``CheckpointImpl.NO_REENTRANT`` -- reentrant checkpointing breaks FSDP,
  * **the frozen vision/text tower never enters the training graph.** Features
    are cached by Team A. This module asserts that rather than assuming it.

Replication is expressed as ``ShardingStrategy.NO_SHARD``, which is DDP
semantics inside an FSDP unit: gradients are all-reduced, parameters are not
sharded. That keeps one wrapping mechanism instead of mixing FSDP and DDP,
which do not compose.

Everything degrades to a no-op when ``torch.distributed`` is not initialised, so
the whole training loop runs unchanged on a CPU login node.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn

from contracts import D, K

__all__ = [
    "FSDPConfig", "wrap_for_training", "assert_frozen_tower_absent",
    "assert_features_are_cached", "memory_estimate", "fits_on_a100_80gb",
    "sharded_state_dict", "BYTES_PER_PARAM_ADAMW", "TOWER_PATTERNS",
]

#: fp32 master (4) + fp32 grad (4) + AdamW exp_avg/exp_avg_sq (8).
BYTES_PER_PARAM_ADAMW = 16

#: Class/attribute name fragments that mean "a frozen encoder is in the graph".
TOWER_PATTERNS = (
    "siglip", "clip", "dinov", "eva", "vit_", "visiontower", "vision_tower",
    "texttower", "text_tower", "t5", "llama", "qwen", "imageencoder",
    "image_encoder", "textencoder", "text_encoder", "visualencoder", "tokenizer",
)


@dataclass
class FSDPConfig:
    shard: tuple[str, ...] = ("estimator", "ema")
    replicate: tuple[str, ...] = ("bank", "q_delta", "q_action", "decoder",
                                  "proposal", "potential")
    activation_checkpointing: bool = True
    #: Class names of the estimator's repeated blocks. Team B names them; the
    #: list is config-driven so a rename does not silently disable AC.
    block_names: tuple[str, ...] = ("PerceiverBlock", "EstimatorBlock", "Block")
    limit_all_gathers: bool = True
    forward_prefetch: bool = True

    @classmethod
    def from_dict(cls, d: dict | None) -> "FSDPConfig":
        d = dict(d or {})
        return cls(
            shard=tuple(d.get("shard", cls.shard)),
            replicate=tuple(d.get("replicate", cls.replicate)),
            activation_checkpointing=bool(d.get("activation_checkpointing", True)),
            block_names=tuple(d.get("block_names", cls.block_names)),
            limit_all_gathers=bool(d.get("limit_all_gathers", True)),
            forward_prefetch=bool(d.get("forward_prefetch", True)),
        )


# ═══════════════════════════════════════════════════════════════════════════
#  BUILD ASSERTS
# ═══════════════════════════════════════════════════════════════════════════

def assert_frozen_tower_absent(model: nn.Module) -> None:
    """The frozen vision/text tower must never enter the training graph.

    Features are cached to disk by ``loom/data/cache.py``. A tower that sneaks in
    under ``no_grad`` still costs its activations and its all-gather, and a tower
    that sneaks in *without* ``no_grad`` quietly triples the parameter count and
    destroys the cached features it was supposed to reproduce.
    """
    offenders = []
    for name, sub in model.named_modules():
        hay = f"{name}.{type(sub).__name__}".lower()
        for pat in TOWER_PATTERNS:
            if pat in hay:
                offenders.append(f"{name or '<root>'} ({type(sub).__name__})")
                break
    if offenders:
        raise RuntimeError(
            "a frozen vision/text tower is inside the training model: "
            + ", ".join(sorted(set(offenders))[:8])
            + ". PLAN 9: the tower never enters the training graph; features are "
              "cached. Move it into loom/data/cache.py."
        )


def assert_bf16(t: torch.Tensor, what: str) -> None:
    """PLAN 9: bf16 only, and never complex.

    The bank casts ``c`` to its own parameter dtype (both the real bank and
    ``StubBank`` do), so an fp32 master-weight bank fed a bf16 coefficient hands
    back **fp32** and the whole rollout silently leaves the intended precision.
    Keep the bank inside autocast, or cast at the call site -- and assert here
    rather than trusting it, because the only symptom is a memory number.
    """
    if t.dtype is not torch.bfloat16:
        raise RuntimeError(
            f"{what} is {t.dtype}, not bfloat16. The bank returns its parameter "
            f"dtype, not c's; wrap the step in autocast(bfloat16) or cast c at the "
            f"call site. PLAN 9: bf16 only, and A100 has no FP8."
        )
    if t.is_complex():
        raise RuntimeError(f"{what} is complex; z is real throughout (PLAN 9)")


def assert_features_are_cached(window: dict) -> None:
    """Cached features carry no autograd history and no grad requirement."""
    for seq_name in ("burn_in_feats", "feats"):
        for i, feats in enumerate(window.get(seq_name, ())):
            for key in ("views", "proprio", "lang"):
                t = feats[key]
                if t.requires_grad or t.grad_fn is not None:
                    raise RuntimeError(
                        f"{seq_name}[{i}][{key!r}] carries autograd history: the "
                        "frozen tower is in the training graph. PLAN 9 -- features "
                        "are cached."
                    )


# ═══════════════════════════════════════════════════════════════════════════
#  WRAPPING
# ═══════════════════════════════════════════════════════════════════════════

def _block_check(block_names: Sequence[str]):
    lowered = {b.lower() for b in block_names}

    def check(mod: nn.Module) -> bool:
        return type(mod).__name__.lower() in lowered

    return check


def _apply_activation_checkpointing(module: nn.Module, block_names: Sequence[str]) -> int:
    """NO_REENTRANT only. Reentrant checkpointing breaks FSDP."""
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper,
    )
    from functools import partial

    check = _block_check(block_names)
    n = sum(1 for _, m in module.named_modules() if check(m))
    if n == 0:
        return 0
    apply_activation_checkpointing(
        module,
        checkpoint_wrapper_fn=partial(checkpoint_wrapper,
                                      checkpoint_impl=CheckpointImpl.NO_REENTRANT),
        check_fn=check,
    )
    return n


class ReplicaSync:
    """DDP semantics for modules whose protocol is not ``forward``.

    ``contracts.Bank`` exposes ``step``/``rollout``, ``Proposal`` exposes
    ``log_prob``, ``Decoder`` exposes ``loss``. **None of those go through
    ``nn.Module.forward``**, and both DDP and FSDP install their gradient-sync
    hooks in the forward pre-hook. A ``FSDP(bank)`` called as ``bank.step(...)``
    silently bypasses every hook: each rank keeps its own gradients, the ranks
    drift apart, and the loss curve looks perfect. (Measured on 2 A100s: an
    FSDP-wrapped head was also unsubscriptable, because FSDP forwards
    ``__getitem__`` but not ``__contains__``, so ``emb in heads`` fell back to
    integer iteration.)

    So the replicated modules stay plain ``nn.Module``s and this does DDP by
    hand: broadcast once at startup (per-rank seeding means their inits differ),
    all-reduce the gradients once per step. It is the same traffic DDP would
    move, without the bucketed overlap -- ~155 M params, which is the small half
    of the model.
    """

    def __init__(self, replicated: dict[str, nn.Module],
                 sharded: dict[str, nn.Module]):
        self.replicated = replicated
        self.sharded = sharded
        self.enabled = False
        try:
            import torch.distributed as dist

            self.enabled = dist.is_available() and dist.is_initialized()
            self.world = dist.get_world_size() if self.enabled else 1
        except ImportError:
            self.world = 1

    def replicated_params(self) -> list[nn.Parameter]:
        return [p for m in self.replicated.values() for p in m.parameters()]

    def sharded_params(self) -> list[nn.Parameter]:
        return [p for m in self.sharded.values() for p in m.parameters()]

    def broadcast(self) -> None:
        """Make every rank start from rank 0's weights, as DDP does at __init__."""
        if not self.enabled:
            return
        import torch.distributed as dist

        with torch.no_grad():
            for m in self.replicated.values():
                for t in list(m.parameters()) + list(m.buffers()):
                    dist.broadcast(t.data, src=0)

    def all_reduce_grads(self) -> None:
        """Average the replicated gradients. Call after backward, before clipping."""
        if not self.enabled:
            return
        import torch.distributed as dist

        grads = [p.grad for p in self.replicated_params() if p.grad is not None]
        if not grads:
            return
        flat = torch._utils._flatten_dense_tensors(grads)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(self.world)
        for g, synced in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
            g.copy_(synced)


def wrap_for_training(model: nn.Module, cfg: FSDPConfig | dict | None = None,
                      device: str = "cuda", verbose: bool = True
                      ) -> tuple[nn.Module, ReplicaSync]:
    """Wrap ``model``'s children in place. Returns ``(model, replica_sync)``.

    Only the estimator (and its EMA target) is FSDP-wrapped, because only the
    estimator is invoked through ``forward``. Everything else is replicated by
    :class:`ReplicaSync` -- see its docstring for why wrapping it would be worse
    than not wrapping it.

    No-op unless ``torch.distributed`` is initialised, so tests and single-GPU
    debug runs take the identical code path.
    """
    cfg = cfg if isinstance(cfg, FSDPConfig) else FSDPConfig.from_dict(cfg)
    assert_frozen_tower_absent(model)

    def present(names):
        return {n: getattr(model, n) for n in names
                if getattr(model, n, None) is not None
                and any(True for _ in getattr(model, n).parameters())}

    import torch.distributed as dist

    distributed = dist.is_available() and dist.is_initialized()
    if not distributed or device != "cuda" or not torch.cuda.is_available():
        if verbose:
            print(f"[fsdp] {'no CUDA device' if distributed else 'not distributed'}; "
                  f"running unwrapped", flush=True)
        return model, ReplicaSync(present(cfg.replicate), present(cfg.shard))

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
    from functools import partial

    # bf16 params and comms, fp32 reductions. No FP8 -- A100 has none (PLAN 9).
    mp = MixedPrecision(param_dtype=torch.bfloat16,
                        reduce_dtype=torch.float32,
                        buffer_dtype=torch.bfloat16)
    common = dict(mixed_precision=mp,
                  device_id=torch.cuda.current_device(),
                  limit_all_gathers=cfg.limit_all_gathers,
                  forward_prefetch=cfg.forward_prefetch,
                  use_orig_params=True)
    block_policy = partial(lambda_auto_wrap_policy, lambda_fn=_block_check(cfg.block_names))

    for name in cfg.shard:
        sub = getattr(model, name, None)
        if sub is None or not any(True for _ in sub.parameters()):
            continue
        # The EMA target runs under no_grad; checkpointing it only costs a recompute.
        if cfg.activation_checkpointing and any(p.requires_grad for p in sub.parameters()):
            n = _apply_activation_checkpointing(sub, cfg.block_names)
            if verbose and n == 0:
                print(f"[fsdp] WARNING no blocks matched {cfg.block_names} inside "
                      f"{name!r}: activation checkpointing is OFF for it", flush=True)
        setattr(model, name, FSDP(sub, sharding_strategy=ShardingStrategy.FULL_SHARD,
                                  auto_wrap_policy=block_policy, **common))
        if verbose:
            print(f"[fsdp] FULL_SHARD {name}", flush=True)

    sync = ReplicaSync(present(cfg.replicate), present(cfg.shard))
    sync.broadcast()
    if verbose:
        print(f"[fsdp] replicated (broadcast + grad all-reduce): "
              f"{sorted(sync.replicated)}", flush=True)
    return model, sync


@contextlib.contextmanager
def sharded_state_dict(model: nn.Module):
    """Save/load per-rank shards instead of an all-gathered full state dict.

    ``ckpt.save`` writes one file per rank, so the state dict must be the local
    shard. The default ``FULL_STATE_DICT`` all-gathers 150 M parameters onto rank
    0 on every checkpoint and returns empty dicts elsewhere -- which looks like a
    successful save and restores a randomly initialised estimator.

    A no-op when nothing is FSDP-wrapped, so the CPU path is identical.
    """
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import StateDictType
    except ImportError:
        yield
        return
    if not any(isinstance(m, FSDP) for m in model.modules()):
        yield
        return
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        yield


# ═══════════════════════════════════════════════════════════════════════════
#  MEMORY BUDGET
# ═══════════════════════════════════════════════════════════════════════════

def memory_estimate(*, world_size: int, batch_per_gpu: int,
                    estimator_params: float = 150e6,
                    replicated_params: float = 155e6,
                    n_blocks: int = 10, n_states: int = 5,
                    n_views: int = 2, patches: int = 196, feat_dim: int = 1152,
                    activation_checkpointing: bool = True,
                    k: int = K, d: int = D) -> dict[str, float]:
    """Analytic per-GPU peak, in GiB. Asserted by a CPU test; measured by a GPU one.

    Defaults follow the PLAN 2 budget: 150 M estimator, ~105 M other shared
    (bank 25 + q_delta 30 + pi_c 50) + 50 M per embodiment = ~155 M replicated.

    Components
      sharded     estimator params/grads/AdamW, divided by world_size
      replicated  everything else, full copy on every rank
      gather      the transient bf16 all-gather buffer for the largest FSDP unit
      acts        with AC, only block-boundary beliefs are kept, plus one block's
                  worth of recompute; without it, all of them
      feats       the cached tower features that come in with the batch
    """
    gib = 1024 ** 3
    b = batch_per_gpu

    sharded = estimator_params * BYTES_PER_PARAM_ADAMW / max(1, world_size)
    replicated = replicated_params * BYTES_PER_PARAM_ADAMW
    # Two units in flight (prefetch), bf16, one block's parameters each.
    gather = 2 * (estimator_params / max(1, n_blocks)) * 2

    belief = b * n_states * k * d * 2                     # bf16 belief tensor
    if activation_checkpointing:
        acts = belief * n_blocks + belief * 8             # boundaries + one recompute
    else:
        acts = belief * n_blocks * 8                      # ~8 saved tensors per block

    feats = b * n_states * n_views * patches * feat_dim * 2
    # cross-attention scores: K queries against the tower tokens, bf16
    attn = b * n_states * k * (n_views * patches) * 2

    out = {
        "sharded_gib": sharded / gib,
        "replicated_gib": replicated / gib,
        "gather_gib": gather / gib,
        "activations_gib": acts / gib,
        "features_gib": feats / gib,
        "attention_gib": attn / gib,
    }
    out["total_gib"] = sum(out.values())
    out["headroom_gib"] = 80.0 - out["total_gib"]
    return out


def fits_on_a100_80gb(est: dict[str, float], reserve_frac: float = 0.15) -> bool:
    """A100-SXM4-80GB with a fragmentation/allocator reserve."""
    return est["total_gib"] <= 80.0 * (1.0 - reserve_frac)
