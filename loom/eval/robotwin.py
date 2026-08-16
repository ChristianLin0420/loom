"""LOOM — RoboTwin 2.0 evaluation. **Phase 1B: seam only, not built out.**

PLAN 5 puts this after the first LIBERO score, and PLAN 7 makes R0-B the
decision gate (< 55 clean = kill). This module therefore exists now with the
*shape* of the finished thing — the same `make_env` seam, the same
`EvalProtocol`, the same `run_episode_safe` contract — so Phase 1B is a
fill-in of `_make_real_env` rather than a rewrite of the harness.

The episode loop is deliberately shared with `loom.eval.libero`: it is
bench-agnostic apart from the 15 LIBERO settle actions, which are passed as
`settle_steps=0` here until the RoboTwin reset convention is confirmed.
"""

from __future__ import annotations

from typing import Any

from loom.eval import EvalProtocol
from loom.eval.libero import run_episode, run_episode_safe  # bench-agnostic loop

__all__ = [
    "SUITES", "TASKS", "N_TASKS", "DEFAULT_PROTOCOL",
    "make_env", "task_name", "task_instruction", "n_tasks",
    "run_episode", "run_episode_safe", "robotwin_available",
]


# ═══════════════════════════════════════════════════════════════════════════
#  INSTALLATION CONSTANTS  (Phase 1B — nothing is provisioned yet)
# ═══════════════════════════════════════════════════════════════════════════

ROBOTWIN_ROOT: str | None = None        # set when the install lands
ROBOTWIN_CONDA_ENV: str | None = None


#: PLAN 8 splits RoboTwin by domain-randomisation setting, not by task family.
SUITES: tuple[str, ...] = ("clean", "randomized")

#: The four per-task columns PLAN 8 asks for, in column order.
TASKS: tuple[str, ...] = (
    "hanging mug",
    "turn switch",
    "place can basket",
    "handover block",
)

N_TASKS = {s: len(TASKS) for s in SUITES}

DEFAULT_PROTOCOL = EvalProtocol(
    bench="robotwin",
    episodes_per_task=10,
    n_tasks=len(TASKS),
    suites=SUITES,
    seeds=(0, 1, 2),
    max_steps=512,
    notes="Phase 1B. Protocol mirrors the LIBERO default until the RoboTwin 2.0 "
          "source table (Fast-WAM Table 1) protocol is confirmed.",
)


def robotwin_available() -> bool:
    return False


def n_tasks(suite: str) -> int:
    return N_TASKS.get(suite, len(TASKS))


def task_name(suite: str, task_id: int) -> str:
    return f"{suite}/{TASKS[int(task_id) % len(TASKS)]}"


def task_instruction(suite: str, task_id: int) -> str:
    return TASKS[int(task_id) % len(TASKS)]


def make_env(suite: str, task_id: int, seed: int, **kw: Any):
    """The seam. Phase 1B fills this in; everything around it already works."""
    raise NotImplementedError(
        "RoboTwin 2.0 env construction is Phase 1B (PLAN 5). The harness, "
        "protocol, results JSON and table emitter are already in place — this "
        "is the only missing piece. Set ROBOTWIN_ROOT and construct the env here."
    )
