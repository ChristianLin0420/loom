"""LOOM — LIBERO-Plus zero-shot transfer. **Phase 1B: seam only, not built out.**

LIBERO-Plus perturbs a solved LIBERO task along one axis at a time. PLAN 8
reports seven axes plus a geometric average; `geo avg` is the mean of
camera / robot-init / layout and is computed by `loom.eval.table.geo_avg`, not
copied. Light and background sit at ~96 and are solved; camera and robot-init
at 60-80 are where the headroom is.

Same structure as `loom.eval.libero` — one `make_env` seam, one `EvalProtocol`,
the shared episode loop — so Phase 1B is a fill-in rather than a rewrite.
"""

from __future__ import annotations

from typing import Any

from loom.eval import EvalProtocol
from loom.eval.libero import SETTLE_STEPS, run_episode, run_episode_safe

__all__ = [
    "SUITES", "GEO_AXES", "N_TASKS", "DEFAULT_PROTOCOL",
    "make_env", "task_name", "task_instruction", "n_tasks",
    "run_episode", "run_episode_safe", "libero_plus_available",
]


# ═══════════════════════════════════════════════════════════════════════════
#  INSTALLATION CONSTANTS  (Phase 1B — nothing is provisioned yet)
# ═══════════════════════════════════════════════════════════════════════════

LIBERO_PLUS_ROOT: str | None = None     # set when the install lands


#: One perturbation axis per suite, in PLAN 8 column order.
SUITES: tuple[str, ...] = (
    "camera", "robot_init", "layout", "light", "background", "language", "noise",
)

#: The three axes that make up `geo avg`.
GEO_AXES: tuple[str, ...] = ("camera", "robot_init", "layout")

N_TASKS = {s: 10 for s in SUITES}

DEFAULT_PROTOCOL = EvalProtocol(
    bench="libero_plus",
    episodes_per_task=10,
    n_tasks=10,
    suites=SUITES,
    seeds=(0, 1, 2),
    max_steps=512,
    notes="Phase 1B. Zero-shot transfer; no LIBERO-Plus training. Protocol "
          "mirrors the LIBERO default until OA-WAM Table 2's is confirmed.",
)


def libero_plus_available() -> bool:
    return False


def n_tasks(suite: str) -> int:
    return N_TASKS.get(suite, 10)


def task_name(suite: str, task_id: int) -> str:
    return f"libero_plus/{suite}/task_{int(task_id):02d}"


def task_instruction(suite: str, task_id: int) -> str:
    return f"libero-plus {suite} perturbation, task {int(task_id)}"


def make_env(suite: str, task_id: int, seed: int, **kw: Any):
    """The seam. Phase 1B fills this in; everything around it already works."""
    raise NotImplementedError(
        "LIBERO-Plus env construction is Phase 1B (PLAN 5). It reuses the "
        f"LIBERO episode loop (settle_steps={SETTLE_STEPS}) with a perturbed "
        "scene; set LIBERO_PLUS_ROOT and construct the env here."
    )
