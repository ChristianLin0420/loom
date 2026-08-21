"""
LOOM — the canonical time base.

Every dataset is resampled to ``contracts.FPS_CANONICAL`` (30 Hz) before it is
segmented, because **different datasets ship at different control rates, and 8
steps must mean the same physical duration everywhere or the shared operator
bank is meaningless.** One operator spans ``H_OP`` = 8 canonical steps = 267 ms,
on LIBERO at 20 Hz and on a 50 Hz teleop log alike.

A window is ``N_STATES`` = 5 observations at ``CANONICAL_FRAMES`` = (0, 8, 16,
24, 32) plus ``DEPTH`` = 4 action segments of ``H_OP`` steps each. Every window
records ``src_fps``: eval must invert this resampling to drive the environment
at its own rate, and a decoded 8-step canonical segment is *not* 8 environment
steps (PLAN §4.F).

Observations and actions resample differently
---------------------------------------------
An observation is a *sample* of the world at an instant. A canonical frame is
served by the **nearest source frame** — images and proprio together, from the
same source index. Interpolating proprio while nearest-sampling the image would
pair a state with a pose it never had.

An action is a *sequence*, and how it resamples depends on what its numbers
mean. This is the subtlest correctness issue in this file:

``delta``     the number is a displacement applied over one *source* step
              (LIBERO's OSC_POSE is a delta controller). Interpolating the
              stream directly keeps the per-step magnitude and changes the step
              count, so 20 -> 30 Hz silently commands 1.5x the motion. Instead
              integrate to cumulative displacement, resample *that* piecewise
              linearly, and difference it back. Integrated motion is preserved
              exactly; per-step magnitude scales by ``src_fps / dst_fps``, which
              is the physically correct rescaling.
``absolute``  the number is a target the controller servos to (joint position,
              absolute EE pose). Linear interpolation of the values.
``hold``      the number is discrete or latched (a binary gripper command).
              Zero-order hold from the previous source sample; interpolating a
              ±1 gripper into 0.37 commands a grip force that never occurred.

Semantics are **per dimension** and must be registered explicitly. LIBERO's
7-vector is six ``delta`` channels plus one ``hold`` gripper channel. There is
no safe default: guessing ``absolute`` for a delta stream rescales the commanded
motion, and guessing ``delta`` for an absolute stream destroys it outright, so
``action_semantics`` raises rather than assume.

Short trajectories
------------------
A trajectory shorter than ``H_PLAN + 1`` = 33 canonical frames yields **zero
windows and is dropped, never padded**. Padding actions with zeros teaches a
fictitious "hold still" operator into the bank and pollutes ``L_dyn``; the cost
of dropping is a handful of short demos.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from contracts import (
    CANONICAL_FRAMES,
    DEPTH,
    EMBODIMENTS,
    FPS_CANONICAL,
    H_OP,
    H_PLAN,
    N_STATES,
)

__all__ = [
    "DELTA", "ABSOLUTE", "HOLD", "ACTION_KINDS",
    "register_action_semantics", "action_semantics",
    "canonical_obs_count", "canonical_action_count",
    "obs_source_indices", "resample_actions", "to_env_rate",
    "CanonicalTrajectory", "WindowIndex",
    "to_canonical", "segment", "window_actions", "required_source_frames",
    "min_source_frames",
]


# ═══════════════════════════════════════════════════════════════════════════
#  ACTION SEMANTICS
# ═══════════════════════════════════════════════════════════════════════════

DELTA = "delta"          # displacement per source step; integrate before resampling
ABSOLUTE = "absolute"    # servo target; linear interpolation
HOLD = "hold"            # discrete / latched; zero-order hold

ACTION_KINDS = (DELTA, ABSOLUTE, HOLD)

#: embodiment -> per-dimension kinds. Deliberately separate from
#: ``contracts.EmbodimentSpec`` (frozen) and deliberately without a default.
_ACTION_SEMANTICS: dict[str, tuple[str, ...]] = {}


def register_action_semantics(embodiment: str, kinds: Sequence[str]) -> tuple[str, ...]:
    """Declare what each action dimension *means*. Idempotent, conflict-loud.

    Mirrors ``contracts.register_embodiment``: re-registering the same tuple is
    free (adapters are imported in every worker), redefining it is an error.
    """
    kinds = tuple(kinds)
    spec = EMBODIMENTS.get(embodiment)
    if spec is None:
        raise KeyError(
            f"unregistered embodiment {embodiment!r}; call "
            f"contracts.register_embodiment first"
        )
    if len(kinds) != spec.dof:
        raise ValueError(
            f"{embodiment}: {len(kinds)} action kinds for dof {spec.dof}"
        )
    bad = sorted(set(kinds) - set(ACTION_KINDS))
    if bad:
        raise ValueError(f"{embodiment}: unknown action kind(s) {bad}, expected {ACTION_KINDS}")
    prev = _ACTION_SEMANTICS.get(embodiment)
    if prev is not None and prev != kinds:
        raise ValueError(
            f"conflicting action semantics for {embodiment!r}:\n  have {prev}\n  got  {kinds}"
        )
    _ACTION_SEMANTICS[embodiment] = kinds
    return kinds


def action_semantics(embodiment: str) -> tuple[str, ...]:
    """Per-dimension kinds for `embodiment`. Raises if never declared."""
    try:
        return _ACTION_SEMANTICS[embodiment]
    except KeyError:
        raise KeyError(
            f"no action semantics registered for {embodiment!r}. There is no safe "
            f"default: treating a delta stream as absolute rescales the commanded "
            f"motion by src_fps/dst_fps and the model trains fine while scoring "
            f"near zero. Call canonical.register_action_semantics({embodiment!r}, ...)."
        ) from None


# LIBERO OSC_POSE: (dx, dy, dz, drx, dry, drz) are per-step deltas in [-1, 1];
# the 7th channel is a latched gripper command, not a displacement.
register_action_semantics("libero_franka", (DELTA,) * 6 + (HOLD,))


# ═══════════════════════════════════════════════════════════════════════════
#  RESAMPLING
# ═══════════════════════════════════════════════════════════════════════════

def canonical_obs_count(n_src: int, src_fps: float, dst_fps: float = FPS_CANONICAL) -> int:
    """Observation frames after resampling. Never extrapolates past the last sample.

    Source frame k sits at t = k / src_fps, so the record spans
    [0, (n_src - 1) / src_fps].
    """
    if n_src <= 0:
        return 0
    _check_fps(src_fps, dst_fps)
    return int(math.floor((n_src - 1) * dst_fps / src_fps)) + 1


def canonical_action_count(n_src: int, src_fps: float, dst_fps: float = FPS_CANONICAL) -> int:
    """Action steps after resampling.

    Action k *occupies* the interval [k / src_fps, (k+1) / src_fps), so n_src
    actions span a duration of n_src / src_fps, one step longer than the
    observation span.
    """
    if n_src <= 0:
        return 0
    _check_fps(src_fps, dst_fps)
    return int(math.floor(n_src * dst_fps / src_fps))


def _check_fps(src_fps: float, dst_fps: float) -> None:
    if not (src_fps > 0.0) or not math.isfinite(src_fps):
        raise ValueError(f"src_fps must be positive and finite, got {src_fps}")
    if not (dst_fps > 0.0) or not math.isfinite(dst_fps):
        raise ValueError(f"dst_fps must be positive and finite, got {dst_fps}")


def obs_source_indices(
    n_src: int, src_fps: float, dst_fps: float = FPS_CANONICAL
) -> np.ndarray:
    """(n_dst,) int64 — nearest source frame for each canonical frame.

    Nearest-index, not interpolation: an observation is one instant of the world
    and every modality in it must come from the same instant.
    """
    n_dst = canonical_obs_count(n_src, src_fps, dst_fps)
    if n_dst == 0:
        return np.zeros(0, dtype=np.int64)
    t = np.arange(n_dst, dtype=np.float64) * (src_fps / dst_fps)
    idx = np.floor(t + 0.5).astype(np.int64)        # half-up, not numpy's half-even
    return np.clip(idx, 0, n_src - 1)


def resample_actions(
    actions: np.ndarray,
    src_fps: float,
    dst_fps: float,
    semantics: Sequence[str],
    n_dst: int | None = None,
) -> np.ndarray:
    """(T, dof) at `src_fps` -> (n_dst, dof) at `dst_fps`, per-dimension semantics.

    Exact identity when ``src_fps == dst_fps``. For ``delta`` dimensions the
    *integrated* displacement over the shared time span is preserved exactly
    (not the per-step magnitude, which scales by ``src_fps / dst_fps`` — that is
    the point). If `n_dst` runs past the end of the source record the trailing
    delta steps are zero and absolute/hold dimensions latch the final value;
    nothing is extrapolated.
    """
    _check_fps(src_fps, dst_fps)
    a = np.asarray(actions)
    if a.ndim != 2:
        raise ValueError(f"actions must be (T, dof), got {a.shape}")
    t_src, dof = a.shape
    kinds = tuple(semantics)
    if len(kinds) != dof:
        raise ValueError(f"{len(kinds)} action kinds for dof {dof}")
    if n_dst is None:
        n_dst = canonical_action_count(t_src, src_fps, dst_fps)
    if n_dst <= 0 or t_src == 0:
        return np.zeros((max(n_dst, 0), dof), dtype=np.float32)

    a = a.astype(np.float64, copy=False)
    out = np.empty((n_dst, dof), dtype=np.float64)

    # sample times of the *values* (absolute / hold) and of the interval
    # *edges* (delta). Actions are intervals, observations are instants.
    t_val_src = np.arange(t_src, dtype=np.float64) / src_fps
    t_val_dst = np.arange(n_dst, dtype=np.float64) / dst_fps
    t_edge_src = np.arange(t_src + 1, dtype=np.float64) / src_fps
    t_edge_dst = np.arange(n_dst + 1, dtype=np.float64) / dst_fps

    for d, kind in enumerate(kinds):
        col = a[:, d]
        if kind == DELTA:
            # cumulative displacement at interval edges, piecewise linear in
            # time, then differenced back onto the destination grid.
            cum = np.concatenate(([0.0], np.cumsum(col)))
            cum_dst = np.interp(t_edge_dst, t_edge_src, cum)
            out[:, d] = np.diff(cum_dst)
        elif kind == ABSOLUTE:
            out[:, d] = np.interp(t_val_dst, t_val_src, col)
        elif kind == HOLD:
            src_idx = np.floor(t_val_dst * src_fps + 1e-9).astype(np.int64)
            out[:, d] = col[np.clip(src_idx, 0, t_src - 1)]
        else:  # pragma: no cover - guarded by register_action_semantics
            raise ValueError(f"unknown action kind {kind!r}")

    return out.astype(np.float32)


def to_env_rate(
    a_seg: np.ndarray,
    embodiment: str,
    n_env_steps: int,
    src_fps: float | None = None,
    *,
    duration_normalized: bool = False,
) -> np.ndarray:
    """Canonical decoder output -> environment control rate. The eval-side inverse.

    `a_seg` is ``(H_OP, dof)`` at ``FPS_CANONICAL``; the caller supplies how many
    environment steps this segment covers (fractional accumulator, see
    ``contracts.env_steps_per_segment`` — rounding each segment independently
    drifts by a full second over a 600-step episode).

    By default the destination bins remain fixed at the environment's physical
    control rate.  A short fractional-clock chunk can therefore cover slightly
    less time than the source segment (LIBERO's 5-step chunk covers 7.5 of the
    8 canonical action intervals).  ``duration_normalized=True`` instead maps
    the complete source-segment duration onto exactly ``n_env_steps`` bins.  It
    still delegates to :func:`resample_actions`, so DELTA dimensions preserve
    their full integral and HOLD dimensions remain zero-order held.
    """
    spec = EMBODIMENTS[embodiment]
    dst = float(spec.env_fps) if src_fps is None else float(src_fps)
    a = np.asarray(a_seg)
    if duration_normalized and n_env_steps > 0 and a.ndim == 2 and a.shape[0] > 0:
        # A synthetic destination rate is just a compact way to make the final
        # destination edge coincide with the final source edge:
        #   n_env_steps / dst == len(a_seg) / FPS_CANONICAL.
        # The actual env rate still owns SegmentClock and therefore the number
        # of bins; this changes only how the source segment fills those bins.
        dst = float(n_env_steps) * FPS_CANONICAL / float(a.shape[0])
    return resample_actions(
        a, FPS_CANONICAL, dst, action_semantics(embodiment), n_dst=n_env_steps
    )


# ═══════════════════════════════════════════════════════════════════════════
#  TRAJECTORIES AND WINDOWS
# ═══════════════════════════════════════════════════════════════════════════

def min_source_frames(src_fps: float) -> int:
    """Fewest source observation frames that can yield one window."""
    return int(math.ceil(H_PLAN * src_fps / FPS_CANONICAL)) + 1


@dataclass(frozen=True)
class CanonicalTrajectory:
    """One demo on the canonical clock.

    Observations are kept as an *index map* into the source record rather than
    resampled arrays: the frozen-tower features are cached per source frame
    (``cache.py``) and a window only ever needs 5 of them.
    """

    traj_id: str
    embodiment: str
    src_fps: float
    obs_src_index: np.ndarray          # (n_frames,) int64 -> source frame
    actions: np.ndarray | None         # (n_actions, dof) float32 at FPS_CANONICAL
    lang: str = ""

    @property
    def n_frames(self) -> int:
        return int(self.obs_src_index.shape[0])

    @property
    def n_actions(self) -> int:
        return 0 if self.actions is None else int(self.actions.shape[0])

    @property
    def action_free(self) -> bool:
        return self.actions is None

    def __post_init__(self) -> None:
        if self.src_fps <= 0.0:
            raise ValueError(f"{self.traj_id}: src_fps must be positive, got {self.src_fps}")
        if self.embodiment not in EMBODIMENTS:
            raise KeyError(f"unregistered embodiment {self.embodiment!r}")
        if self.actions is not None:
            dof = EMBODIMENTS[self.embodiment].dof
            if self.actions.ndim != 2 or self.actions.shape[1] != dof:
                raise ValueError(
                    f"{self.traj_id}: actions must be (T, {dof}), got {self.actions.shape}"
                )


@dataclass(frozen=True)
class WindowIndex:
    """Where one training window's data lives. No tensors, cheap to hold by the million."""

    traj_id: str
    embodiment: str
    src_fps: float
    start: int                          # canonical frame of state 0
    obs_src_index: tuple[int, ...]      # length N_STATES, source frames
    act_lo: int | None                  # canonical action slice, None when action-free
    act_hi: int | None

    @property
    def action_free(self) -> bool:
        return self.act_lo is None

    @property
    def canonical_frames(self) -> tuple[int, ...]:
        return tuple(self.start + f for f in CANONICAL_FRAMES)


def to_canonical(
    n_src_frames: int,
    src_fps: float,
    embodiment: str,
    traj_id: str,
    actions: np.ndarray | None = None,
    lang: str = "",
) -> CanonicalTrajectory:
    """Put one raw trajectory on the canonical clock.

    `n_src_frames` is the number of recorded observations; `actions` is the raw
    ``(T_a, dof)`` action stream at `src_fps`, or None for action-free data
    (Ego4D-style, R1).
    """
    if embodiment not in EMBODIMENTS:
        raise KeyError(f"unregistered embodiment {embodiment!r}")
    obs_idx = obs_source_indices(n_src_frames, src_fps)
    can_actions = None
    if actions is not None:
        can_actions = resample_actions(
            np.asarray(actions), src_fps, FPS_CANONICAL, action_semantics(embodiment)
        )
    return CanonicalTrajectory(
        traj_id=traj_id,
        embodiment=embodiment,
        src_fps=float(src_fps),
        obs_src_index=obs_idx,
        actions=can_actions,
        lang=lang,
    )


def segment(traj: CanonicalTrajectory, stride: int = H_OP) -> list[WindowIndex]:
    """Cut a canonical trajectory into ``TransitionWindow``-shaped indices.

    `stride` is in canonical frames and defaults to ``H_OP``. Keeping it a
    multiple of ``H_OP`` means every boundary observation falls on a multiple of
    ``H_OP``, so the cache only ever has to encode 1 frame in 8 (see
    ``required_source_frames``). Trajectories shorter than ``H_PLAN + 1``
    canonical frames produce no windows — dropped, not padded.
    """
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    last_start = traj.n_frames - 1 - H_PLAN
    if not traj.action_free:
        # actions t .. t+H_PLAN-1 must exist for a window starting at t
        last_start = min(last_start, traj.n_actions - H_PLAN)
    if last_start < 0:
        return []

    out: list[WindowIndex] = []
    for start in range(0, last_start + 1, stride):
        frames = tuple(int(traj.obs_src_index[start + f]) for f in CANONICAL_FRAMES)
        out.append(
            WindowIndex(
                traj_id=traj.traj_id,
                embodiment=traj.embodiment,
                src_fps=traj.src_fps,
                start=start,
                obs_src_index=frames,
                act_lo=None if traj.action_free else start,
                act_hi=None if traj.action_free else start + H_PLAN,
            )
        )
    return out


def window_actions(traj: CanonicalTrajectory, w: WindowIndex) -> np.ndarray | None:
    """(DEPTH, H_OP, dof) — one operator per row. Never a flat (H_PLAN, dof)."""
    if w.action_free or traj.actions is None:
        return None
    seg = traj.actions[w.act_lo:w.act_hi]
    if seg.shape[0] != H_PLAN:
        raise ValueError(
            f"{w.traj_id}@{w.start}: expected {H_PLAN} canonical actions, got {seg.shape[0]}"
        )
    dof = seg.shape[1]
    return seg.reshape(DEPTH, H_OP, dof)


def required_source_frames(windows: Iterable[WindowIndex]) -> dict[str, np.ndarray]:
    """traj_id -> sorted unique source frames that must be encoded.

    With the default ``stride == H_OP`` the 5 states of consecutive windows
    overlap 4-deep, so encoding per window would cost ~5x. This is what the
    cache is keyed on instead.
    """
    acc: dict[str, set[int]] = {}
    for w in windows:
        acc.setdefault(w.traj_id, set()).update(w.obs_src_index)
    return {k: np.array(sorted(v), dtype=np.int64) for k, v in acc.items()}


# sanity: the module's own layout constants agree with the frozen contract
assert len(CANONICAL_FRAMES) == N_STATES
assert CANONICAL_FRAMES[-1] == H_PLAN
