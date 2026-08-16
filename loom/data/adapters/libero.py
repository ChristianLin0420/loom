"""
LOOM — LIBERO -> canonical adapter.

Reads the robomimic-style demo HDF5s with ``h5py`` directly. The ``libero``
python package is **not** required and is not installed on this cluster; only
the eval harness needs the simulator.

The embodiment ``libero_franka`` (dof 7, env_fps 20, n_views 2, actions in
[-1,1]^7) is already registered in the frozen ``contracts.py``; this module adds
only what that spec cannot carry: the per-dimension action semantics (six
OSC_POSE deltas plus one latched gripper) and the image orientation convention.

LIBERO records at 20 Hz and the canonical clock is 30 Hz, so every trajectory is
**upsampled** 20 -> 30. OSC_POSE is a *delta* controller, which is exactly the
case ``canonical.resample_actions`` handles by integrating before resampling:
interpolating the delta stream directly would command 1.5x the motion.

Image orientation
-----------------
Every demo file carries ``macros_image_convention = 'opengl'``: the stored RGB
is bottom-up, vertically mirrored relative to a normal image. The live
simulator's orientation depends on ``robosuite.macros.IMAGE_CONVENTION`` at eval
time. **If train and eval disagree the model sees mirrored inputs only at eval
and scores near zero with no other symptom.** So this module fixes one canonical
orientation, applies it on the dataset side, exposes the same helper for the
eval side, and records the applied transform in the cache manifest so a
mismatch is auditable rather than invisible. ``best_matching_transform`` exists
so Team F/G can *verify* the pairing against a live reset once the simulator
lands, instead of assuming it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from contracts import EMBODIMENTS, EmbodimentSpec, register_embodiment

from ..cache import CacheSpec, FeatureCache, FeatureCacheWriter
from ..canonical import (
    DELTA,
    HOLD,
    CanonicalTrajectory,
    register_action_semantics,
    required_source_frames,
    segment,
    to_canonical,
)
from ..loader import CachedWindowDataset

__all__ = [
    "DATA_ROOT", "SUITES", "SRC_FPS", "EMBODIMENT", "WINDOW_STRIDE",
    "IMAGE_CONVENTION", "CANONICAL_ORIENTATION", "VIEW_KEYS", "PROPRIO_KEYS",
    "LiberoDemo", "discover", "read_actions", "read_proprio", "read_images",
    "orient_dataset_image", "orient_env_image", "best_matching_transform",
    "libero_trajectories", "encode_to_cache", "libero_dataset",
]


# ═══════════════════════════════════════════════════════════════════════════
#  ── CONSTANTS BLOCK ─────────────────────────────────────────────────────
#
#  Everything installation-specific lives here and nowhere else. Override the
#  root with $LOOM_DATA_ROOT (either the `libero/` directory itself or a parent
#  containing it — both are accepted) or by passing `root=`.
#
#  Verified 2026-08-16 on this cluster:
#    * all four suites present under DATA_ROOT, 10 task files each
#    * data.attrs: num_demos=50, total, bddl_file_name, env_name, env_args,
#      macros_image_convention='opengl', tag='libero-v1', problem_info
#    * problem_info is JSON carrying `language_instruction`
#    * demo_<i>: actions (T,7) float64, dones, rewards, states, robot_states, obs
#    * obs: agentview_rgb / eye_in_hand_rgb (T,128,128,3) uint8, ee_pos (T,3),
#      ee_ori (T,3), ee_states (T,6), gripper_states (T,2), joint_states (T,7)
#    * demo lengths ~80-110 frames -> ~120-165 canonical frames -> ~11-17 windows
#
#  `<root>/libero_object` is a symlink into another project's read-only copy.
#  NEVER write inside DATA_ROOT; the feature cache goes somewhere else entirely.
# ═══════════════════════════════════════════════════════════════════════════

#: Light-WAM Table 1 column order (PLAN §8). Keep it.
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")

_DEFAULT_DATA_ROOT = Path(
    "/lustre/fsw/portfolios/edgeai/users/chrislin/datasets/loom/libero"
)


def _resolve_data_root(env_root: str | None) -> Path:
    """The directory that directly contains the suite directories.

    ``$LOOM_DATA_ROOT`` is set to the ``libero/`` directory itself by the R0-A
    sbatch, but reads naturally as "the datasets root". Accept both rather than
    pick one: guessing wrong appends a second ``libero/`` and the run dies at
    load time with a path nobody typed.
    """
    if not env_root:
        return _DEFAULT_DATA_ROOT
    p = Path(env_root).expanduser()
    if any((p / s).is_dir() for s in SUITES):
        return p                        # already the libero directory
    if any((p / "libero" / s).is_dir() for s in SUITES):
        return p / "libero"             # a datasets root containing libero/
    return p                            # neither: discover() names the miss


DATA_ROOT = _resolve_data_root(os.environ.get("LOOM_DATA_ROOT"))

EMBODIMENT = "libero_franka"
SRC_FPS = 20.0                      # robosuite control_freq; == EMBODIMENTS[...].env_fps

#: Canonical frames between window starts. H_OP, so every boundary state falls
#: on a multiple of H_OP and the cache holds ~1 source frame in 8. A ~90-frame
#: demo gives ~13 overlapping windows; a larger stride would leave R0-A short of
#: data, a smaller one would multiply the cache without adding new observations.
WINDOW_STRIDE = 8

VIEW_KEYS = ("agentview_rgb", "eye_in_hand_rgb")     # order is the V axis, fixed
PROPRIO_KEYS = ("ee_states", "gripper_states")       # 6 + 1 -> dof 7

#: What the files say they contain, and what this repo normalises everything to.
IMAGE_CONVENTION = "opengl"          # stored bottom-up, per data.attrs
CANONICAL_ORIENTATION = "top_down"   # row 0 is the top of the scene, everywhere

# ═══════════════════════════════════════════════════════════════════════════

# contracts.py already froze the spec; re-register idempotently so importing
# this module alone is enough to have the body available.
register_embodiment(EmbodimentSpec(
    name=EMBODIMENT, dof=7, env_fps=SRC_FPS, n_views=2,
    action_low=(-1.0,) * 7, action_high=(1.0,) * 7,
))
# OSC_POSE: (dx, dy, dz, drx, dry, drz) are per-step deltas; ch 6 is a latched
# gripper command in {-1, +1} and must never be interpolated.
register_action_semantics(EMBODIMENT, (DELTA,) * 6 + (HOLD,))


# ═══════════════════════════════════════════════════════════════════════════
#  IMAGE ORIENTATION
# ═══════════════════════════════════════════════════════════════════════════

_TRANSFORMS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "identity": lambda x: x,
    "vflip": lambda x: x[..., ::-1, :, :],
    "hflip": lambda x: x[..., :, ::-1, :],
    "rot180": lambda x: x[..., ::-1, ::-1, :],
}


def orient_dataset_image(img: np.ndarray) -> np.ndarray:
    """HDF5 image (opengl, bottom-up) -> CANONICAL_ORIENTATION. Vertical flip.

    Accepts (..., H, W, 3).
    """
    return np.ascontiguousarray(_TRANSFORMS["vflip"](img))


def orient_env_image(img: np.ndarray, env_convention: str = IMAGE_CONVENTION) -> np.ndarray:
    """Live-simulator image -> CANONICAL_ORIENTATION.

    `env_convention` is ``robosuite.macros.IMAGE_CONVENTION`` as configured by
    the eval harness. ``opencv`` means robosuite already flipped for you.
    """
    if env_convention == "opencv":
        return np.ascontiguousarray(img)
    if env_convention == "opengl":
        return orient_dataset_image(img)
    raise ValueError(f"unknown image convention {env_convention!r}")


def best_matching_transform(dataset_img: np.ndarray, env_img: np.ndarray) -> tuple[str, float]:
    """Which flip maps a live env frame onto the dataset frame of the same state.

    Run this once against a real reset when the simulator lands. If it does not
    return the transform this module applies, train and eval disagree and the
    score will be near zero with no other symptom.
    """
    a = np.asarray(dataset_img, dtype=np.float32)
    scores = {
        name: float(np.abs(a - fn(np.asarray(env_img, dtype=np.float32))).mean())
        for name, fn in _TRANSFORMS.items()
    }
    best = min(scores, key=scores.get)
    return best, scores[best]


# ═══════════════════════════════════════════════════════════════════════════
#  DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class LiberoDemo:
    """One demonstration inside one task HDF5."""

    path: Path
    suite: str
    task: str
    demo_key: str
    instruction: str
    n_frames: int

    @property
    def traj_id(self) -> str:
        return f"{self.suite}/{self.task}/{self.demo_key}"


def _h5py():
    import h5py                       # imported lazily: canonical/loader do not need it
    return h5py


def _instruction(attrs) -> str:
    raw = attrs.get("problem_info", "{}")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        return str(json.loads(raw).get("language_instruction", "")).strip()
    except json.JSONDecodeError:
        return ""


def discover(
    root: str | os.PathLike | None = None,
    suites: Sequence[str] = SUITES,
    tasks: Sequence[str] | None = None,
    max_demos: int | None = None,
) -> list[LiberoDemo]:
    """Enumerate demos without reading any pixels."""
    h5py = _h5py()
    base = Path(root) if root is not None else DATA_ROOT
    out: list[LiberoDemo] = []
    for suite in suites:
        sdir = base / suite
        if not sdir.is_dir():
            raise FileNotFoundError(f"{sdir} does not exist; set $LOOM_DATA_ROOT or pass root=")
        for path in sorted(sdir.glob("*.hdf5")):
            task = path.stem.removesuffix("_demo")
            if tasks is not None and task not in tasks:
                continue
            with h5py.File(path, "r") as f:
                data = f["data"]
                instruction = _instruction(data.attrs)
                keys = sorted(data.keys(), key=lambda s: int(s.split("_")[1]))
                if max_demos is not None:
                    keys = keys[:max_demos]
                for k in keys:
                    out.append(LiberoDemo(
                        path=path, suite=suite, task=task, demo_key=k,
                        instruction=instruction,
                        n_frames=int(data[k]["actions"].shape[0]),
                    ))
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  READING
# ═══════════════════════════════════════════════════════════════════════════

def _h5_index(frames: Sequence[int]) -> np.ndarray:
    """h5py fancy indexing requires a strictly increasing selection."""
    idx = np.asarray(frames, dtype=np.int64)
    if idx.ndim != 1 or (idx.size > 1 and (np.diff(idx) <= 0).any()):
        raise ValueError(
            f"frames must be strictly increasing for h5py fancy indexing, got {idx.tolist()}"
        )
    return idx


def read_actions(demo: LiberoDemo) -> np.ndarray:
    """(T, 7) float32 raw OSC_POSE at SRC_FPS. Not resampled."""
    with _h5py().File(demo.path, "r") as f:
        return np.asarray(f["data"][demo.demo_key]["actions"], dtype=np.float32)


def read_proprio(demo: LiberoDemo, frames: Sequence[int] | None = None) -> np.ndarray:
    """(n, 7) float32: ee_states (pos 3 + ori 3) plus one gripper finger.

    Deliberately in the *same* coordinates as the action space rather than joint
    angles — ``dof_e`` is 7 either way, but a proprio vector the decoder can
    difference against its own output is more useful than one it cannot.
    """
    with _h5py().File(demo.path, "r") as f:
        obs = f["data"][demo.demo_key]["obs"]
        sel = slice(None) if frames is None else _h5_index(frames)
        parts = [np.asarray(obs["ee_states"][sel], dtype=np.float32),
                 np.asarray(obs["gripper_states"][sel], dtype=np.float32)[..., :1]]
    out = np.concatenate(parts, axis=-1)
    dof = EMBODIMENTS[EMBODIMENT].dof
    if out.shape[-1] != dof:
        raise ValueError(f"proprio width {out.shape[-1]}, embodiment dof {dof}")
    return out


def read_images(demo: LiberoDemo, frames: Sequence[int]) -> np.ndarray:
    """(n, V, H, W, 3) uint8 in CANONICAL_ORIENTATION, V ordered as VIEW_KEYS."""
    idx = _h5_index(frames)
    with _h5py().File(demo.path, "r") as f:
        obs = f["data"][demo.demo_key]["obs"]
        views = [orient_dataset_image(np.asarray(obs[k][idx])) for k in VIEW_KEYS]
    return np.stack(views, axis=1)


def libero_trajectories(
    root: str | os.PathLike | None = None,
    suites: Sequence[str] = SUITES,
    tasks: Sequence[str] | None = None,
    max_demos: int | None = None,
    demos: Iterable[LiberoDemo] | None = None,
) -> list[CanonicalTrajectory]:
    """Every demo on the canonical 30 Hz clock. Reads actions only, no pixels.

    Demos too short for one window are dropped by ``canonical.segment`` and
    filtered out here so they never reach the cache builder.
    """
    demos = list(demos) if demos is not None else discover(root, suites, tasks, max_demos)
    h5py = _h5py()
    out: list[CanonicalTrajectory] = []
    by_file: dict[Path, list[LiberoDemo]] = {}
    for d in demos:
        by_file.setdefault(d.path, []).append(d)
    # one open per task file, not per demo: 50 demos share a 780 MB container
    for path, group in by_file.items():
        with h5py.File(path, "r") as f:
            for d in group:
                actions = np.asarray(f["data"][d.demo_key]["actions"], dtype=np.float32)
                traj = to_canonical(
                    n_src_frames=d.n_frames, src_fps=SRC_FPS, embodiment=EMBODIMENT,
                    traj_id=d.traj_id, actions=actions, lang=d.instruction,
                )
                if segment(traj, stride=WINDOW_STRIDE):
                    out.append(traj)
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  CACHE BUILD
# ═══════════════════════════════════════════════════════════════════════════

#: images (n, V, H, W, 3) uint8 + instruction -> views (n, V, P, F), lang (L, F).
#: Supplied by the caller so the frozen tower stays out of loom/data (PLAN §9).
Encoder = Callable[[np.ndarray, str], "tuple[np.ndarray, np.ndarray]"]


def encode_to_cache(
    encoder: Encoder,
    cache_root: str | os.PathLike,
    *,
    root: str | os.PathLike | None = None,
    suites: Sequence[str] = SUITES,
    tasks: Sequence[str] | None = None,
    max_demos: int | None = None,
    codec: str | None = None,
    chunk: int = 16,
) -> list[CanonicalTrajectory]:
    """Encode every demo once with the frozen tower and write the feature cache.

    Only the source frames some window actually lands on are encoded — with
    ``WINDOW_STRIDE == H_OP`` that is ~1 frame in 8. Returns the canonical
    trajectories so the caller can build a dataset without re-reading the HDF5s.
    """
    from ..cache import DEFAULT_CODEC

    demos = discover(root, suites, tasks, max_demos)
    by_id = {d.traj_id: d for d in demos}
    trajs = libero_trajectories(demos=demos)
    need = required_source_frames(
        [w for t in trajs for w in segment(t, stride=WINDOW_STRIDE)]
    )

    spec: CacheSpec | None = None
    writer: FeatureCacheWriter | None = None
    try:
        for t in trajs:
            demo = by_id[t.traj_id]
            frames = need[t.traj_id]
            chunks_v, chunks_l = [], None
            for lo in range(0, len(frames), chunk):
                imgs = read_images(demo, frames[lo:lo + chunk])
                v, lang = encoder(imgs, demo.instruction)
                chunks_v.append(np.asarray(v, dtype=np.float32))
                chunks_l = np.asarray(lang, dtype=np.float32)
            views = np.concatenate(chunks_v, axis=0)
            if spec is None:
                spec = CacheSpec(
                    codec=codec or DEFAULT_CODEC,
                    n_views=views.shape[1], n_patches=views.shape[2],
                    feat_dim=views.shape[3], dof=EMBODIMENTS[EMBODIMENT].dof,
                    lang_len=chunks_l.shape[0],
                )
                writer = FeatureCacheWriter(cache_root, spec)
            writer.write(
                t.traj_id,
                frames=frames,
                views=views,
                proprio=read_proprio(demo, frames),
                lang=chunks_l,
                embodiment=EMBODIMENT,
                src_fps=SRC_FPS,
                meta={
                    "suite": demo.suite, "task": demo.task,
                    "instruction": demo.instruction,
                    "image_orientation": CANONICAL_ORIENTATION,
                    "source_convention": IMAGE_CONVENTION,
                },
            )
    finally:
        if writer is not None:
            writer.flush()
    if writer is None:
        raise RuntimeError("no LIBERO demos found to encode")
    return trajs


def libero_dataset(
    cache_root: str | os.PathLike,
    root: str | os.PathLike | None = None,
    suites: Sequence[str] = SUITES,
    tasks: Sequence[str] | None = None,
    max_demos: int | None = None,
) -> CachedWindowDataset:
    """A ``CachedWindowDataset`` over an already-encoded cache."""
    cache = FeatureCache(cache_root)
    trajs = [t for t in libero_trajectories(root, suites, tasks, max_demos) if t.traj_id in cache]
    if not trajs:
        raise RuntimeError(f"no cached LIBERO trajectories in {cache_root}")
    return CachedWindowDataset(trajs, cache, stride=WINDOW_STRIDE)
