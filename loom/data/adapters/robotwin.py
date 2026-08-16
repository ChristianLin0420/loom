"""
LOOM — RoboTwin 2.0 adapter. **PHASE 1B SKELETON. NOT IMPLEMENTED.**

PLAN §4.A: "LIBERO only for now — do not build seven adapters before the first
score exists." This file exists so the shape of the work is written down and so
nobody builds it by accident; every entry point raises.

R0-B is the decision gate (PLAN §7): under 55 clean kills the operator
formulation. It runs *after* R0-A produces a LIBERO number.

What this adapter will have to settle, in this order:

1. **Embodiment.** RoboTwin 2.0 is bimanual (two 7-dof arms + 2 grippers). That
   is a *different body* from ``libero_franka``: a new
   ``contracts.register_embodiment`` with its own dof, and its own ``q_a`` and
   ``D_e`` entries in Team C's ``ModuleDict``s. It must never share a batch with
   LIBERO — the loader already guarantees that.
2. **Action semantics per dimension** — ``canonical.register_action_semantics``.
   Get this wrong and the resampled motion is silently rescaled. Check whether
   the released controller is joint-position (``ABSOLUTE``) or end-effector
   delta (``DELTA``); it is not the same choice as LIBERO's OSC_POSE.
3. **Control rate** -> ``src_fps`` for ``canonical.to_canonical``. RoboTwin
   records at a different rate from LIBERO; that is precisely why the canonical
   30 Hz clock exists.
4. **Views.** More than LIBERO's two streams (head + two wrist cameras at
   least), so ``CacheSpec.n_views`` rises and the per-window bytes rise with it
   — this is the case where the int8 codec in ``cache.py`` may stop being
   optional (see the profiling table there).

Nothing else in ``loom/data`` needs to change: ``canonical.py``, ``cache.py``
and ``loader.py`` are dataset-agnostic by construction.
"""

from __future__ import annotations

from typing import Any

__all__ = ["EMBODIMENT", "discover", "robotwin_trajectories", "encode_to_cache"]

#: Reserved name. Registering it is part of the Phase 1B work, not of this stub.
EMBODIMENT = "robotwin_bimanual"

_MSG = (
    "RoboTwin is Phase 1B (PLAN §5) and deliberately unimplemented. The first "
    "LIBERO score comes first; building seven adapters before it is what PLAN "
    "§4.A rules out. See this module's docstring for the four decisions it needs."
)


def discover(*args: Any, **kwargs: Any):
    raise NotImplementedError(_MSG)


def robotwin_trajectories(*args: Any, **kwargs: Any):
    raise NotImplementedError(_MSG)


def encode_to_cache(*args: Any, **kwargs: Any):
    raise NotImplementedError(_MSG)
