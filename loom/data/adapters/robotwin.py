"""
LOOM — RoboTwin 2.0 -> canonical adapter.  **R0-B, the decision gate (PLAN §7).**

Reads the released ``demo_clean`` HDF5s with ``h5py`` directly. Neither the
RoboTwin package nor SAPIEN is required here; only the eval harness needs the
simulator. Everything installation-specific is in the CONSTANTS BLOCK below and
every number in it is *measured*, not read off a paper. The long-form evidence,
with the source line numbers it was cross-checked against, is
``docs/ENV_ROBOTWIN.md``.

The three things that are wrong-but-plausible
---------------------------------------------

**1. `src_fps` is 250/15 = 16.6667 Hz, NOT 15.**  Every released episode carries
``/additional_info/frequency = 15``, which reads like hertz. It is
``save_freq`` — a decimation factor on a 250 Hz physics step
(``RoboTwin/scripts/process_data_xpolicylab.py:409`` writes ``save_freq`` into a
field named ``frequency``; ``envs/_base_task.py:223`` sets
``scene.set_timestep(1/250)``; ``take_action`` calls ``_take_picture()`` when
``i % save_freq == 0`` after each ``scene.step()``). Measured independently from
the released data: RoboTwin plans every gripper motion as
``np.linspace(now, target, 200)`` (``envs/robot/planner.py:426``) advanced one
element per physics step, so a saved-frame-to-saved-frame gripper delta is
``k/199`` where ``k`` is the number of physics steps between saved frames. Over
15 episodes in 5 tasks, 92.9 % of nonzero gripper deltas were exactly ``15/199``
(the remaining 7.1 % were ``4/199`` — the unconditional ``_take_picture()`` at
each action-segment boundary lands off phase), and over 60 episodes the largest
gripper delta seen anywhere was ``0.075377 == 15/199`` exactly.

``env_steps_per_segment(16.6667) = 4.4444``; the wrong 15.0 gives a tidy
``4.0000`` and makes every executed segment 11 % too slow. The tidy number is
the wrong one.

**2. Actions are ABSOLUTE joint targets — all 14 channels, gripper included.**
``pkl2hdf5.py:149`` writes ``action = joint_target[1:]`` and ``state =
joint_target[:-1]``, so ``action[t] == state[t+1]`` *exactly* — verified bitwise
over all 2500 episodes. The gripper is not a latched ±1 like LIBERO's: it is a
normalised width in [0, 1] that the planner ramps linearly over 200 physics
steps, so it is a continuous servo target and ``ABSOLUTE`` (linear
interpolation) is the physically correct resampling. ``HOLD`` would staircase a
ramp the data itself interpolated; ``DELTA`` would destroy the stream outright.
No 2π joint wraparound occurs between adjacent frames (max adjacent arm delta
0.158 rad over 165 336 samples), so interpolating across a pair never sweeps
the long way round.

**3. The stored JPEGs are BGR-as-RGB.**  RoboTwin encodes with
``cv2.imencode(".jpg", rgb_image)`` and says so in a comment
(``envs/utils/pkl2hdf5.py:29``): *"OpenCV intentionally interprets this RGB
array as BGR while encoding."* ``cv2.imdecode`` inverts that and returns the
original array; **PIL does not** — ``PIL.open(...).convert("RGB")`` returns the
channels reversed. The live simulator hands eval true RGB
(``envs/camera/camera.py:328``, straight off SAPIEN's ``"Color"`` picture), so a
cache built from an un-swapped PIL decode would train on channel-swapped images
and be scored on correct ones. ``decode_frame`` does the swap; see
``CHANNEL_ORDER`` for the three independent confirmations.

Unlike LIBERO there is **no vertical flip**: RoboTwin frames are already
row-0-is-top (confirmed by eye — bottles stand upright, the "AGILE X" decal on
the arm reads left-to-right, and in ``stack_blocks_three``'s final frame the
blue block is above the green one above the red one, exactly as the instruction
says).

Sizing
------
2 500 episodes (50 tasks × 50), 549 787 source frames. At ``WINDOW_STRIDE ==
H_OP`` that is **114 375 windows over 124 375 cached source frames** (49.8 per
episode — the cache is keyed by source frame, so the 4-deep overlap between
consecutive windows is stored once). At V=4, P=256, F=1152 fp16 that is
**273.6 GiB** and 497 500 tower images. LIBERO for comparison: 56 189 windows,
64 189 frames, 70.8 GiB, V=2.
"""

from __future__ import annotations

import io
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from contracts import EmbodimentSpec, register_embodiment

from ..cache import CacheSpec, FeatureCache, FeatureCacheWriter
from ..canonical import (
    ABSOLUTE,
    CanonicalTrajectory,
    register_action_semantics,
    required_source_frames,
    segment,
    to_canonical,
)
from ..loader import CachedWindowDataset

__all__ = [
    "DATA_ROOT", "SUITES", "SRC_FPS", "EMBODIMENT", "WINDOW_STRIDE",
    "VIEW_KEYS", "EVAL_VIEW_KEYS", "ACTION_KEY", "PROPRIO_KEY",
    "CHANNEL_ORDER", "SOURCE_ENCODING", "ACTION_SEMANTICS", "DOF",
    "RobotwinEpisode", "discover", "tasks",
    "decode_frame", "orient_env_image",
    "read_actions", "read_proprio", "read_images", "read_instruction",
    "robotwin_trajectories", "encode_to_cache", "robotwin_dataset",
    "merge_shards",
]


# ═══════════════════════════════════════════════════════════════════════════
#  ── CONSTANTS BLOCK ─────────────────────────────────────────────────────
#
#  Everything installation-specific lives here and nowhere else. Override the
#  root with $LOOM_DATA_ROOT (the `robotwin/` directory, its `data/` child, or a
#  parent containing `robotwin/` — all three are accepted) or by passing `root=`.
#
#  Verified 2026-08-16 against the mounted copy:
#    * 50 task dirs x 50 episodes = 2500 episode HDF5s, all `aloha_agilex`
#    * T (episode length): min 74, max 894, mean 219.9, median 166, total 549787
#    * /additional_info/frequency == 15 in all 2500 (== save_freq, NOT Hz)
#    * /action/joint_states (T,14) float32 == [L_arm(6) | L_grip | R_arm(6) | R_grip]
#      -- checked against the concatenation of the four per-limb datasets
#    * action[t] == state[t+1] bitwise in all 2500
#    * /vision/{cam_head,cam_left_wrist,cam_right_wrist,cam_third_view}/colors
#      (T,) JPEG bytes -> (240,320,3) uint8, every stream non-black
#    * /instruction is the phrasing actually used for the episode; /instructions
#      is the template pool it was drawn from
#
#  NEVER write inside DATA_ROOT; the feature cache goes somewhere else entirely.
# ═══════════════════════════════════════════════════════════════════════════

#: Domain-randomisation setting, PLAN §8's `clean` / `rand` columns. R0-B is
#: scored on clean and only `demo_clean` is downloaded.
SUITES = ("demo_clean",)

#: The embodiment directory inside each task. RoboTwin also ships arx-x5,
#: franka-panda, piper and ur5-wsg; PLAN §8's baseline table is aloha-agilex.
ROBOT_DIR = "aloha_agilex"

_DEFAULT_DATA_ROOT = Path(
    "/lustre/fsw/portfolios/edgeai/users/chrislin/datasets/loom/robotwin/data"
)


def _resolve_data_root(env_root: str | None) -> Path:
    """The directory that directly contains the suite directories.

    ``$LOOM_CACHE_DIR``'s sibling ``$LOOM_DATA_ROOT`` is set to ``.../robotwin``
    by ``loom/train/slurm/r0b.sbatch``, but reads naturally as "the datasets
    root". Accept every spelling rather than pick one: guessing wrong appends a
    second ``data/`` and the run dies at load time on a path nobody typed.
    """
    if not env_root:
        return _DEFAULT_DATA_ROOT
    p = Path(env_root).expanduser()
    for cand in (p, p / "data", p / "robotwin" / "data"):
        if any((cand / s).is_dir() for s in SUITES):
            return cand
    return p                            # none matched: discover() names the miss


DATA_ROOT = _resolve_data_root(os.environ.get("LOOM_DATA_ROOT"))

EMBODIMENT = "robotwin_aloha"       # == configs/r0b.yaml data.embodiments

#: 250 Hz physics / save_freq 15. See the module docstring; do NOT use 15.0.
SRC_FPS = 250.0 / 15.0              # 16.666666666666668

DOF = 14                            # (6 arm + 1 gripper) x 2

#: Canonical frames between window starts. H_OP, so every boundary state falls
#: on a multiple of H_OP and the cache holds ~1 source frame in 4.4 (the ratio
#: is H_OP * SRC_FPS / FPS_CANONICAL, not H_OP, because RoboTwin is *upsampled*
#: 16.67 -> 30). Same value as LIBERO; changing it invalidates the cache.
WINDOW_STRIDE = 8

#: The V axis, fixed. Order MUST match `EVAL_VIEW_KEYS`; swapping it is a silent
#: near-zero score. head + both wrists + the third-person/front view, which is
#: exactly the four streams the live env exposes.
VIEW_KEYS = ("cam_head", "cam_left_wrist", "cam_right_wrist", "cam_third_view")

#: The same four streams as the *live* RoboTwin observation dict names them,
#: in the same order. `loom.data.tower.obs_featurizer(spec, view_keys=...)`
#: takes this; its `EVAL_VIEW_KEYS` default is LIBERO's two.
EVAL_VIEW_KEYS = ("head_camera", "left_camera", "right_camera", "front_camera")

ACTION_KEY = "action/joint_states"       # (T, 14) absolute joint targets
PROPRIO_KEY = "state/joint_states"       # (T, 14) same coordinates as the action

#: What `decode_frame` returns, and what the live env hands eval. The stored
#: JPEG is BGR-as-RGB (see the module docstring); this is the normalised form.
CHANNEL_ORDER = "rgb"
SOURCE_ENCODING = "cv2_bgr_jpeg"

#: Per-dimension action semantics. All 14 are absolute servo targets:
#: 12 joint angles in radians and 2 normalised gripper widths in [0, 1].
ACTION_SEMANTICS = (ABSOLUTE,) * DOF

#: URDF revolute limits, `arx5_description_isaac.urdf` `<limit lower="-10"
#: upper="10">` on fl_joint1..6 / fr_joint1..6. Deliberately permissive: these
#: are the admissible range, and `heads.decoder` clamps its samples to them.
#: The *empirical* per-dimension spread is roughly [-6.4, 5.7] rad; see
#: docs/ENV_ROBOTWIN.md §1. Do not tighten these to the empirical bounds — an
#: unseen scene at eval would then be clipped mid-reach.
_ARM_LO, _ARM_HI = (-10.0,) * 6, (10.0,) * 6
_GRIP_LO, _GRIP_HI = 0.0, 1.0

# ═══════════════════════════════════════════════════════════════════════════

register_embodiment(EmbodimentSpec(
    name=EMBODIMENT,
    dof=DOF,
    env_fps=SRC_FPS,
    n_views=len(VIEW_KEYS),
    action_low=_ARM_LO + (_GRIP_LO,) + _ARM_LO + (_GRIP_LO,),
    action_high=_ARM_HI + (_GRIP_HI,) + _ARM_HI + (_GRIP_HI,),
))
register_action_semantics(EMBODIMENT, ACTION_SEMANTICS)


# ═══════════════════════════════════════════════════════════════════════════
#  PIXELS
# ═══════════════════════════════════════════════════════════════════════════

def decode_frame(buf: bytes | np.ndarray) -> np.ndarray:
    """One stored JPEG -> ``(H, W, 3)`` uint8 in **true RGB**.

    The channel reversal is the whole point of this function and is not
    cosmetic. RoboTwin encodes with ``cv2.imencode(".jpg", rgb_array)``, which
    interprets its input as BGR; ``cv2.imdecode`` inverts that and returns the
    original array, but Pillow returns the JPEG's *semantic* RGB, which is the
    original array reversed. Three independent confirmations that the swap is
    the right way round:

    1. RoboTwin's own comment at ``envs/utils/pkl2hdf5.py:29``.
    2. ``stack_blocks_three``'s last frame — instruction "stack blue block on
       green block, followed by green block on red block" — reads blue-on-top
       only after the swap; ``blocks_ranking_rgb``'s final left-to-right order
       reads red, green, blue only after the swap.
    3. Direct inspection: a Coca-Cola bottle in ``put_bottles_dustbin`` is red
       after the swap and blue before it.
    """
    from PIL import Image                     # noqa: PLC0415 (lazy: numpy-only import)

    raw = buf.tobytes() if isinstance(buf, np.generic) else bytes(buf)
    img = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
    return np.ascontiguousarray(img[..., ::-1])         # BGR -> RGB


def orient_env_image(img: np.ndarray) -> np.ndarray:
    """Live-simulator frame -> the cache's convention. Identity, on purpose.

    The eval-side twin of `decode_frame`. SAPIEN's ``camera.get_picture("Color")``
    is already true RGB, row 0 at the top (``envs/camera/camera.py:328``), which
    is exactly what `decode_frame` normalises the stored JPEGs to. LIBERO needs
    a vertical flip here; RoboTwin needs nothing. This function exists so both
    sides go through a *named* transform and a future disagreement is a diff
    rather than an invisible near-zero score.
    """
    a = np.asarray(img)
    if a.ndim < 3 or a.shape[-1] != 3:
        raise ValueError(f"expected (..., H, W, 3) RGB, got {a.shape}")
    return np.ascontiguousarray(a)


# ═══════════════════════════════════════════════════════════════════════════
#  DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class RobotwinEpisode:
    """One episode HDF5. One file, unlike LIBERO's 50-demos-per-container."""

    path: Path
    suite: str
    task: str
    episode: str
    n_frames: int

    @property
    def traj_id(self) -> str:
        return f"{self.suite}/{self.task}/{self.episode}"


def _h5py():
    import h5py                       # noqa: PLC0415 (canonical/loader do not need it)
    return h5py


def tasks(root: str | os.PathLike | None = None, suite: str = SUITES[0]) -> list[str]:
    """Task directory names present under `suite`, sorted."""
    base = (Path(root) if root is not None else DATA_ROOT) / suite
    if not base.is_dir():
        raise FileNotFoundError(f"{base} does not exist; set $LOOM_DATA_ROOT or pass root=")
    return sorted(p.name for p in base.iterdir() if (p / ROBOT_DIR / "data").is_dir())


def discover(
    root: str | os.PathLike | None = None,
    suites: Sequence[str] = SUITES,
    tasks_: Sequence[str] | None = None,
    max_episodes: int | None = None,
) -> list[RobotwinEpisode]:
    """Enumerate episodes. Opens each file for its length; reads no pixels."""
    h5py = _h5py()
    base = Path(root) if root is not None else DATA_ROOT
    out: list[RobotwinEpisode] = []
    for suite in suites:
        sdir = base / suite
        if not sdir.is_dir():
            raise FileNotFoundError(
                f"{sdir} does not exist; set $LOOM_DATA_ROOT (currently "
                f"{os.environ.get('LOOM_DATA_ROOT')!r}) or pass root="
            )
        for task in tasks(base, suite):
            if tasks_ is not None and task not in tasks_:
                continue
            paths = sorted((sdir / task / ROBOT_DIR / "data").glob("episode_*.hdf5"))
            if max_episodes is not None:
                paths = paths[:max_episodes]
            for path in paths:
                with h5py.File(path, "r") as f:
                    n = int(f[ACTION_KEY].shape[0])
                out.append(RobotwinEpisode(
                    path=path, suite=suite, task=task, episode=path.stem, n_frames=n,
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


def read_instruction(ep: RobotwinEpisode) -> str:
    """The phrasing this episode was actually recorded with.

    ``/instructions`` is the 50–100 template pool; ``/instruction`` is the one
    drawn for this episode (``process_data_xpolicylab.py:_choose_instruction``
    takes ``instructions[episode_idx % len]``). Using the pool's first entry for
    every episode would throw away the language variation the data ships with.
    """
    with _h5py().File(ep.path, "r") as f:
        raw = f["instruction"][()]
    return (raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)).strip()


def read_actions(ep: RobotwinEpisode) -> np.ndarray:
    """``(T, 14)`` float32 absolute joint targets at SRC_FPS. Not resampled."""
    with _h5py().File(ep.path, "r") as f:
        a = np.asarray(f[ACTION_KEY], dtype=np.float32)
    if a.shape[1] != DOF:
        raise ValueError(f"{ep.traj_id}: {ACTION_KEY} width {a.shape[1]}, expected {DOF}")
    return a


def read_proprio(ep: RobotwinEpisode, frames: Sequence[int] | None = None) -> np.ndarray:
    """``(n, 14)`` float32 measured joint state at the given SOURCE frames.

    Deliberately the *same* coordinates as the action space: ``state[t]`` is the
    configuration the frame-``t`` image was taken in and ``action[t]`` is the
    target commanded from it, so the decoder can difference its own output
    against the proprio it was conditioned on.
    """
    with _h5py().File(ep.path, "r") as f:
        ds = f[PROPRIO_KEY]
        out = (np.asarray(ds, dtype=np.float32) if frames is None
               else np.asarray(ds[_h5_index(frames)], dtype=np.float32))
    if out.shape[-1] != DOF:
        raise ValueError(f"{ep.traj_id}: proprio width {out.shape[-1]}, dof {DOF}")
    return out


def read_images(ep: RobotwinEpisode, frames: Sequence[int]) -> np.ndarray:
    """``(n, V, 240, 320, 3)`` uint8 true RGB, V ordered as `VIEW_KEYS`.

    Each camera's ``colors`` dataset is read **whole** and indexed in memory.
    It is contiguous and uncompressed at ~6–17 MiB per camera per episode, and
    Lustre serves one sequential read of that at ~650 MiB/s where the same bytes
    fetched as ~50 scattered rows measured 13 MiB/s (see `cache.py`).
    """
    idx = _h5_index(frames)
    views = []
    with _h5py().File(ep.path, "r") as f:
        for key in VIEW_KEYS:
            blob = f[f"vision/{key}/colors"][()]         # (T,) fixed-width bytes
            views.append(np.stack([decode_frame(blob[i]) for i in idx], axis=0))
    return np.stack(views, axis=1)


def robotwin_trajectories(
    root: str | os.PathLike | None = None,
    suites: Sequence[str] = SUITES,
    tasks_: Sequence[str] | None = None,
    max_episodes: int | None = None,
    episodes: Iterable[RobotwinEpisode] | None = None,
) -> list[CanonicalTrajectory]:
    """Every episode on the canonical 30 Hz clock. Reads actions only, no pixels.

    16.6667 -> 30 Hz is an *upsample*, and because every channel is ``ABSOLUTE``
    the resampling is a straight linear interpolation of the joint path. Demos
    too short for one window are dropped by ``canonical.segment`` and filtered
    out here so they never reach the cache builder (none are, in practice: the
    shortest episode is 74 frames against a floor of 19).
    """
    eps = list(episodes) if episodes is not None else discover(
        root, suites, tasks_, max_episodes)
    h5py = _h5py()
    out: list[CanonicalTrajectory] = []
    for ep in eps:
        with h5py.File(ep.path, "r") as f:
            actions = np.asarray(f[ACTION_KEY], dtype=np.float32)
            raw = f["instruction"][()]
        lang = (raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)).strip()
        traj = to_canonical(
            n_src_frames=ep.n_frames, src_fps=SRC_FPS, embodiment=EMBODIMENT,
            traj_id=ep.traj_id, actions=actions, lang=lang,
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
    tasks_: Sequence[str] | None = None,
    max_episodes: int | None = None,
    codec: str | None = None,
    chunk: int = 8,
    shard: tuple[int, int] = (0, 1),
    flush_every: int = 8,
    on_episode: Callable[[int, int, str, float], None] | None = None,
) -> list[CanonicalTrajectory]:
    """Encode this shard's episodes with the frozen tower into `cache_root`.

    Only the source frames some window actually lands on are encoded — 49.8 of
    an average 220, i.e. 1 in 4.4. Returns the canonical trajectories of the
    episodes **this shard owns**, so a caller can build a dataset without
    re-reading the HDF5s.

    `shard` is ``(index, count)`` over the discovered episode list. Shards must
    use **separate roots**: ``FeatureCacheWriter`` rewrites the manifest whole,
    so two processes sharing a root clobber each other. `merge_shards` folds
    them into one afterwards.

    Resumable by construction: the writer reloads an existing manifest, and any
    trajectory already in it is skipped. A link killed at the 4 h wall loses at
    most `flush_every` episodes of work.
    """
    import json                                           # noqa: PLC0415

    from ..cache import DEFAULT_CODEC, MANIFEST_NAME       # noqa: PLC0415

    i_shard, n_shard = int(shard[0]), int(shard[1])
    if not (0 <= i_shard < n_shard):
        raise ValueError(f"bad shard {shard}")

    found = discover(root, suites, tasks_, max_episodes)
    if not found:
        # An empty *dataset* is a configuration error (bad root or tasks filter)
        # and must be loud. An empty *shard* of a small dataset is not.
        raise RuntimeError(
            f"no RoboTwin episodes under {root or DATA_ROOT} for suites={tuple(suites)} "
            f"tasks={tasks_}; set $LOOM_DATA_ROOT or pass root="
        )
    eps = found[i_shard::n_shard]
    by_id = {e.traj_id: e for e in eps}
    trajs = robotwin_trajectories(episodes=eps)
    need = required_source_frames(
        [w for t in trajs for w in segment(t, stride=WINDOW_STRIDE)]
    )

    # Resume: whatever a previous link already wrote. Read before the loop so a
    # completed episode costs zero tower time, not one wasted encode.
    prev = Path(cache_root) / MANIFEST_NAME
    done_ids: set[str] = set()
    if prev.exists():
        done_ids = set(json.loads(prev.read_text()).get("entries", {}))

    spec: CacheSpec | None = None
    writer: FeatureCacheWriter | None = None
    done = 0
    try:
        for k, t in enumerate(trajs):
            ep = by_id[t.traj_id]
            frames = need[t.traj_id]
            t0 = time.perf_counter()

            if t.traj_id in done_ids:
                if on_episode is not None:
                    on_episode(k, len(trajs), t.traj_id, 0.0)
                continue

            imgs = read_images(ep, frames)                # (n, V, H, W, 3)
            chunks_v, lang = [], None
            for lo in range(0, len(frames), chunk):
                v, lg = encoder(imgs[lo:lo + chunk], t.lang)
                chunks_v.append(np.asarray(v, dtype=np.float32))
                lang = np.asarray(lg, dtype=np.float32)
            views = np.concatenate(chunks_v, axis=0)

            if spec is None:
                spec = CacheSpec(
                    codec=codec or DEFAULT_CODEC,
                    n_views=views.shape[1], n_patches=views.shape[2],
                    feat_dim=views.shape[3], dof=DOF, lang_len=lang.shape[0],
                )
                writer = FeatureCacheWriter(cache_root, spec)

            writer.write(
                t.traj_id,
                frames=frames,
                views=views,
                proprio=read_proprio(ep, frames),
                lang=lang,
                embodiment=EMBODIMENT,
                src_fps=SRC_FPS,
                meta={
                    "suite": ep.suite, "task": ep.task, "episode": ep.episode,
                    "instruction": t.lang,
                    "channel_order": CHANNEL_ORDER,
                    "source_encoding": SOURCE_ENCODING,
                    "views": list(VIEW_KEYS),
                },
            )
            done += 1
            if done % flush_every == 0:
                writer.flush()
            if on_episode is not None:
                on_episode(k, len(trajs), t.traj_id, time.perf_counter() - t0)
    finally:
        if writer is not None:
            writer.flush()
    return trajs


def merge_shards(
    shard_roots: Sequence[str | os.PathLike],
    dest_root: str | os.PathLike,
) -> dict:
    """Fold per-task shard caches into one readable cache root.

    ``cache.py`` deliberately does not automate this ("merging shards is a `cp`
    of the ``feats/`` dirs plus a ``json`` union"). It is a **rename**, not a
    copy: the shards live under ``<dest>/_shards/`` on the same filesystem, so
    273 GiB moves in O(number of files). Idempotent — re-running after a partial
    merge finishes the job.
    """
    import json                                            # noqa: PLC0415
    import shutil                                          # noqa: PLC0415
    from ..cache import CACHE_FORMAT_VERSION, CacheFormatError, FEAT_DIR, MANIFEST_NAME

    dest = Path(dest_root)
    (dest / FEAT_DIR).mkdir(parents=True, exist_ok=True)
    spec: dict | None = None
    entries: dict[str, dict] = {}
    man = dest / MANIFEST_NAME
    if man.exists():
        doc = json.loads(man.read_text())
        spec, entries = doc["spec"], doc["entries"]

    for sr in shard_roots:
        sp = Path(sr) / MANIFEST_NAME
        if not sp.exists():
            continue
        doc = json.loads(sp.read_text())
        if doc.get("format_version") != CACHE_FORMAT_VERSION:
            raise CacheFormatError(f"{sr}: format version {doc.get('format_version')!r}")
        if spec is None:
            spec = doc["spec"]
        elif doc["spec"] != spec:
            raise CacheFormatError(f"{sr}: spec {doc['spec']} != {spec}")
        for tid, e in doc["entries"].items():
            src = Path(sr) / e["file"]
            dst = dest / e["file"]
            if src.exists():
                os.replace(src, dst)            # same filesystem: O(1)
            elif not dst.exists():
                raise FileNotFoundError(f"{tid}: neither {src} nor {dst} exists")
            entries[tid] = e

    if spec is None:
        raise RuntimeError(f"no shard manifests under {list(shard_roots)}")
    tmp = dest / (MANIFEST_NAME + ".tmp")
    tmp.write_text(json.dumps(
        {"format_version": CACHE_FORMAT_VERSION, "spec": spec, "entries": entries},
        separators=(",", ":"),
    ))
    os.replace(tmp, man)
    for sr in shard_roots:                       # empty shells only
        shutil.rmtree(sr, ignore_errors=True)
    return {"entries": len(entries), "spec": spec,
            "bytes": sum(int(e["nbytes"]) for e in entries.values())}


def robotwin_dataset(
    cache_root: str | os.PathLike,
    root: str | os.PathLike | None = None,
    suites: Sequence[str] = SUITES,
    tasks_: Sequence[str] | None = None,
    max_episodes: int | None = None,
) -> CachedWindowDataset:
    """A ``CachedWindowDataset`` over an already-encoded cache."""
    cache = FeatureCache(cache_root)
    trajs = [t for t in robotwin_trajectories(root, suites, tasks_, max_episodes)
             if t.traj_id in cache]
    if not trajs:
        raise RuntimeError(f"no cached RoboTwin trajectories in {cache_root}")
    return CachedWindowDataset(trajs, cache, stride=WINDOW_STRIDE)
