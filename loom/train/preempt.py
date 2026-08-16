"""Three independent exit paths, all required.

Signal delivery through ``srun`` -> worker is not guaranteed on every cluster
configuration, so signals alone are not enough:

  (a) ``SIGUSR1`` / ``SIGTERM`` handler        -- SLURM's ``--signal=USR1@420``
  (b) a wall-clock budget                      -- the path that always works
  (c) a ``runs/<name>/STOP`` sentinel file     -- the manual stop; never scancel

``decide_local()`` is torch-free on purpose so all three paths are unit testable
without a GPU.

The ``safety_s`` margin and the sbatch ``--signal=USR1@N`` must agree: LOOM uses
420 s in both. ``LOOM_TIME_BUDGET_S`` (4 h - 600 s) is the *link* budget and is
separate from that margin -- the link stops at ``budget - safety``.
"""

from __future__ import annotations

import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "StopDecision", "decide_local", "write_heartbeat", "read_heartbeat",
    "heartbeat_age_s", "PreemptGuard", "DEFAULT_BUDGET_S", "DEFAULT_SAFETY_S",
]

#: 4 h cap minus 10 min of slurm/venv/startup slack.
DEFAULT_BUDGET_S = 4 * 3600 - 600
#: Must equal the sbatch `--signal=USR1@420`.
DEFAULT_SAFETY_S = 420.0


@dataclass(frozen=True)
class StopDecision:
    stop: bool
    reason: str


def decide_local(now: float, deadline: float, safety_s: float,
                 signalled: bool, sentinel_exists: bool) -> StopDecision:
    """Pure. The precedence matters only for the logged reason, not the outcome."""
    if signalled:
        return StopDecision(True, "signal")
    if sentinel_exists:
        return StopDecision(True, "sentinel")
    if now > deadline - safety_s:
        return StopDecision(True, "budget")
    return StopDecision(False, "")


def write_heartbeat(run_dir: str | Path, step: int, rank: int | None = None,
                    delta_op: float | None = None) -> None:
    """Liveness marker: ``<unix_ts> <step> <delta_op>``. Rank 0 only.

    Rank 0 is sufficient *and* necessary. Sufficient: every rank agrees on
    stopping through the all_reduce in ``PreemptGuard.should_stop``, so if any
    rank wedges, rank 0 blocks in that collective and stops emitting. Necessary:
    ranks reach ``step % log_every == 0`` together, so unguarded writes race on
    the shared ``.tmp`` path and the loser's ``os.replace`` raises
    ``FileNotFoundError`` -- which, uncaught, kills the link under
    ``--kill-on-bad-exit=1``.

    ``delta_op`` rides along because it is a build assert, not a metric: a
    flatlined Delta_op means the model collapsed to a plain latent policy and the
    run should be killed rather than left to burn six days.
    """
    if rank is None:
        rank = int(os.environ.get("RANK", 0))
    if rank != 0:
        return
    p = Path(run_dir) / "HEARTBEAT"
    tmp = p.with_name(p.name + ".tmp")
    d = "nan" if delta_op is None else f"{delta_op:.6g}"
    # int(): a rounded-up stamp reads as a negative age in the watchdog.
    tmp.write_text(f"{int(time.time())} {step} {d}\n")
    os.replace(tmp, p)


def read_heartbeat(run_dir: str | Path) -> tuple[float, int, float] | None:
    p = Path(run_dir) / "HEARTBEAT"
    if not p.exists():
        return None
    try:
        parts = p.read_text().split()
        return float(parts[0]), int(parts[1]), float(parts[2]) if len(parts) > 2 else float("nan")
    except (ValueError, IndexError, OSError):
        return None


def heartbeat_age_s(run_dir: str | Path) -> float | None:
    hb = read_heartbeat(run_dir)
    return None if hb is None else time.time() - hb[0]


class PreemptGuard:
    def __init__(self, run_dir: str | Path, budget_s: float | None = None,
                 safety_s: float = DEFAULT_SAFETY_S, install_handlers: bool = True):
        if budget_s is None:
            budget_s = float(os.environ.get("LOOM_TIME_BUDGET_S", DEFAULT_BUDGET_S))
        self.start = time.time()
        self.deadline = self.start + float(budget_s)
        self.safety_s = float(safety_s)
        self.sentinel = Path(run_dir) / "STOP"
        self._signalled = False
        self.reason = ""
        if install_handlers:
            for sig in (signal.SIGUSR1, signal.SIGTERM):
                try:
                    signal.signal(sig, self._on_signal)
                except (ValueError, OSError):
                    pass  # not on the main thread, or unsupported platform

    def _on_signal(self, signum, _frame) -> None:
        self._signalled = True
        self.reason = f"signal:{signum}"

    @property
    def seconds_left(self) -> float:
        return max(0.0, self.deadline - self.safety_s - time.time())

    def should_stop(self) -> bool:
        """MUST be called on EVERY rank EVERY step.

        The all_reduce is not optional. If one rank decides to save while the
        others keep training, the next collective hangs until SLURM kills the job
        and the whole interval since the last checkpoint is lost.
        """
        d = decide_local(
            now=time.time(),
            deadline=self.deadline,
            safety_s=self.safety_s,
            signalled=self._signalled,
            sentinel_exists=self.sentinel.exists(),
        )
        local = d.stop
        if d.stop and not self.reason:
            self.reason = d.reason

        try:
            import torch
            import torch.distributed as dist
        except ImportError:
            return local

        if dist.is_available() and dist.is_initialized():
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            flag = torch.tensor([int(local)], dtype=torch.int32, device=dev)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            agreed = bool(flag.item())
            if agreed and not self.reason:
                self.reason = "peer"
            return agreed
        return local
