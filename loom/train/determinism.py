"""All randomness in the training path derives from ``(seed, global_step, rank)``.

There is no bare ``torch.randn`` in the LOOM training path. Every draw either
comes from a generator built by :func:`torch_generator`, or happens inside a
step whose global RNG was reseeded by :func:`set_step_seed`. That is what makes
"resume with continuous loss" assertable by *equality* on one process rather
than by eyeball.

The functions that matter for resume correctness are torch-free on purpose so
they can be unit tested without a GPU.
"""

from __future__ import annotations

import hashlib
import os
import random
from typing import Any

import numpy as np

__all__ = [
    "mix", "np_rng", "torch_generator", "rng_fingerprint", "rank_identity",
    "set_global_seed", "set_step_seed", "capture_rng_state", "restore_rng_state",
    "enable_determinism",
]

_U63 = 2 ** 63 - 1
_U32 = 2 ** 32


def mix(seed: int, step: int, rank: int, tag: str = "") -> int:
    """Stable 64-bit stream id. Must not depend on process state or import order."""
    h = hashlib.blake2b(f"{seed}|{step}|{rank}|{tag}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "little")


def np_rng(seed: int, step: int, rank: int, tag: str = "") -> np.random.Generator:
    """Per-(step, rank) numpy generator. Two calls with the same args are identical."""
    return np.random.default_rng(mix(seed, step, rank, tag))


def torch_generator(seed: int, step: int, rank: int, tag: str = "", device: str = "cpu"):
    import torch

    g = torch.Generator(device=device)
    g.manual_seed(mix(seed, step, rank, tag) % _U63)
    return g


def rng_fingerprint(seed: int, step: int, rank: int, tag: str = "data", n: int = 8) -> str:
    """Short digest of the draws this rank actually makes.

    Two ranks reporting the same fingerprint means they will train on the same
    windows. That failure is silent -- the loss curve looks entirely plausible --
    so it is asserted at launch rather than discovered at step 40k.
    """
    draws = np_rng(seed, step, rank, tag).random(n)
    return hashlib.blake2b(draws.tobytes(), digest_size=6).hexdigest()


def rank_identity(seed: int, rank: int, local_rank: int, world: int,
                  step: int = 0) -> dict[str, Any]:
    """Everything needed to prove ranks are distinct, in one loggable record."""
    import socket

    ident: dict[str, Any] = {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world,
        "host": socket.gethostname(),
        "device": "cpu",
        "rng_fingerprint": rng_fingerprint(seed, step, rank),
    }
    try:
        import torch

        if torch.cuda.is_available():
            idx = local_rank % torch.cuda.device_count()
            ident["device"] = f"cuda:{idx}({torch.cuda.get_device_name(idx)})"
    except ImportError:
        pass
    return ident


def set_global_seed(seed: int, rank: int) -> None:
    """Once, at process start. Covers module init and anything outside a step."""
    random.seed(mix(seed, 0, rank, "py") % _U32)
    np.random.seed(mix(seed, 0, rank, "np") % _U32)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(mix(seed, 0, rank, "torch") % _U63)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(mix(seed, 0, rank, "cuda") % _U63)
    except ImportError:
        pass


def set_step_seed(seed: int, step: int, rank: int) -> None:
    """Reseed the global streams at the top of every step.

    This is the cheap way to make a step a pure function of
    ``(seed, step, rank, params)``. Without it, resume continuity depends on the
    RNG state round-tripping through the checkpoint *and* on no library having
    consumed a different number of draws before the crash -- which is exactly
    the kind of thing that holds on CPU and breaks on 64 GPUs.
    """
    random.seed(mix(seed, step, rank, "py") % _U32)
    np.random.seed(mix(seed, step, rank, "np") % _U32)
    try:
        import torch

        torch.manual_seed(mix(seed, step, rank, "torch") % _U63)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(mix(seed, step, rank, "cuda") % _U63)
    except ImportError:
        pass


def capture_rng_state() -> dict[str, Any]:
    """Everything needed to make step N+1 identical after a restart."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    try:
        import torch

        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
    except ImportError:
        pass
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    if state is None:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    try:
        import torch

        # RNG states are ByteTensors and must be on CPU. Checkpoints are loaded
        # with map_location="cuda", which drags them onto the GPU and makes
        # set_rng_state raise -- a failure invisible until the first GPU resume,
        # while every CPU test stays green.
        def _cpu_bytes(t):
            return t.cpu().to(torch.uint8)

        if "torch" in state:
            torch.set_rng_state(_cpu_bytes(state["torch"]))
        if "cuda" in state and torch.cuda.is_available():
            saved = [_cpu_bytes(s) for s in state["cuda"]]
            # Resuming onto a different GPU count is legal (a 64-GPU run can be
            # debugged on 8). Restore what maps over, drop the rest.
            for i in range(min(len(saved), torch.cuda.device_count())):
                torch.cuda.set_rng_state(saved[i], i)
    except ImportError:
        pass


def enable_determinism() -> None:
    """Slower, but lets the single-process resume test assert equality."""
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
