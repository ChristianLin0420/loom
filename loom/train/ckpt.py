"""Checkpoint save / load.

Completeness is the whole point. Omitting any field below makes resume silently
*wrong* rather than loudly broken:

    model (E, bank, q_delta, q_a, D_e, pi_c, Phi), optimizer, LR-schedule
    horizon, EMA target estimator, sampler cursor, per-rank RNG, global_step,
    samples_seen, config hash, git SHA, world size, W&B run id.

``build_state`` enumerates those slots **explicitly**, on purpose.
``tests/test_train.py::test_state_coverage_by_reflection`` walks the live
``TrainState`` by reflection and fails if any object exposing ``state_dict()``
is not among them -- so the next stateful object someone adds fails a test
instead of failing on link 2 of a 37-link chain.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from loom.train import atomic
from loom.train.determinism import capture_rng_state, restore_rng_state

__all__ = [
    "shard_name", "build_state", "save", "load_latest", "latest_step", "restore",
    "git_sha", "STATE_SLOTS",
]

#: Every object in TrainState that carries mutable training state.
STATE_SLOTS = ("model", "optimizer", "scheduler", "ema", "sampler")


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
            cwd=str(Path(__file__).resolve().parents[2]),
        ).decode().strip()
    except Exception:
        return "unknown"


def _rank() -> int:
    return int(os.environ.get("RANK", 0))


def shard_name(step: int, rank: int) -> str:
    """The ONE place a checkpoint filename is spelled.

    Two ranks resolving to the same name silently overwrite each other's shard,
    and the loss curve of the resulting run looks entirely normal.
    """
    return f"ckpt_{step:09d}_rank{rank}.pt"


def build_state(state, *, config_hash: str = "", world_size: int = 1,
                wandb_run_id: str = "", extra: dict | None = None) -> dict[str, Any]:
    """Explicit enumeration. See the module docstring for why it is not reflective."""
    payload: dict[str, Any] = {
        "format": 1,
        # -- the five state_dict-bearing slots --
        "model": state.model.state_dict(),
        "optimizer": state.optimizer.state_dict(),
        "scheduler": state.scheduler.state_dict(),
        "ema": state.ema.state_dict() if state.ema is not None else None,
        "sampler": state.sampler.state_dict() if state.sampler is not None else None,
        # -- scalars --
        "global_step": int(state.global_step),
        "samples_seen": int(state.samples_seen),
        "rng": capture_rng_state(),
        "config_hash": config_hash,
        "git_sha": git_sha(),
        "world_size": int(world_size),
        "wandb_run_id": wandb_run_id,
    }
    if extra:
        payload.update(extra)
    return payload


def save(payload: dict[str, Any], run_dir: str | Path, step: int,
         keep_last: int = 3, permanent_every: int = 10000) -> Path:
    """Atomic, sharded, pointer-last.

    Order is load bearing:
      1. every rank writes its own shard through ``.tmp`` + ``os.replace``
      2. ``barrier()`` -- every shard is now durable
      3. rank 0 advances LATEST, then prunes
      4. ``barrier()`` again -- nobody races ahead of the prune
    """
    import torch
    import torch.distributed as dist

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    rank = _rank()
    target = run_dir / shard_name(step, rank)

    atomic.atomic_via_writer(target, lambda tmp: torch.save(payload, tmp))

    distributed = dist.is_available() and dist.is_initialized()
    if distributed:
        dist.barrier()
    if rank == 0:
        atomic.write_pointer(run_dir, step)          # only after every payload is durable
        _prune(run_dir, keep_last, permanent_every)
    if distributed:
        dist.barrier()
    return target


def latest_step(run_dir: str | Path) -> int | None:
    return atomic.read_pointer(run_dir)


def load_latest(run_dir: str | Path, map_location="cpu",
                allow_reshard: bool = False) -> dict[str, Any] | None:
    import torch

    step = latest_step(run_dir)
    if step is None:
        return None
    run_dir = Path(run_dir)
    rank = _rank()
    p = run_dir / shard_name(step, rank)
    if not p.exists():
        peers = sorted(run_dir.glob(f"ckpt_{step:09d}_rank*.pt"))
        if not peers:
            return None
        if not allow_reshard:
            raise RuntimeError(
                f"LATEST points at step {step} but this rank's shard {p.name} is "
                f"missing; {len(peers)} shards exist. The world size changed since "
                f"that checkpoint. FSDP shards are per-rank, so silently loading a "
                f"peer's shard would corrupt the estimator. Reshard, or rerun with "
                f"the original world size, or pass --allow_reshard if every module "
                f"in this run is replicated."
            )
        p = peers[0]
    return torch.load(p, map_location=map_location, weights_only=False)


def restore(payload: dict[str, Any], state, *, world_size: int = 1,
            strict: bool = True) -> dict[str, Any]:
    """Load a payload back into a live TrainState. Mutates ``state``."""
    state.model.load_state_dict(payload["model"], strict=strict)
    state.optimizer.load_state_dict(payload["optimizer"])
    if payload.get("scheduler") is not None:
        state.scheduler.load_state_dict(payload["scheduler"])
    if state.ema is not None and payload.get("ema") is not None:
        state.ema.load_state_dict(payload["ema"])
    if state.sampler is not None and payload.get("sampler") is not None:
        state.sampler.load_state_dict(payload["sampler"])
    restore_rng_state(payload.get("rng"))
    state.global_step = int(payload["global_step"])
    state.samples_seen = int(payload["samples_seen"])

    saved_world = int(payload.get("world_size", world_size))
    if saved_world != world_size:
        print(
            f"[ckpt] WARNING resuming a world_size={saved_world} checkpoint on "
            f"world_size={world_size}. Sample ordering and the per-rank RNG "
            f"streams differ from the original run.",
            flush=True,
        )
    return {
        "global_step": state.global_step,
        "samples_seen": state.samples_seen,
        "config_hash": payload.get("config_hash", ""),
        "git_sha": payload.get("git_sha", "unknown"),
        "world_size": saved_world,
    }


def _step_of(p: Path) -> int:
    return int(p.name.split("_")[1])


def _prune(run_dir: Path, keep_last: int, permanent_every: int) -> None:
    shards = list(run_dir.glob("ckpt_*_rank*.pt"))
    steps = sorted({_step_of(p) for p in shards})
    keep = set(steps[-keep_last:])
    if permanent_every > 0:
        keep |= {s for s in steps if s % permanent_every == 0}
    for p in shards:
        if _step_of(p) not in keep:
            p.unlink(missing_ok=True)
