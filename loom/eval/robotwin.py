"""LOOM — closed-loop RoboTwin 2.0 evaluation, behind the same seam as LIBERO.

    python -m loom.eval --bench robotwin --ckpt <path> --out results.json

RoboTwin 2.0 is the **R0-B decision gate** (PLAN §7): `< 55` clean kills the
operator formulation. Everything in this file exists so that number is
trustworthy, and it is deliberately structured like `loom.eval.libero` — the
same `make_env` seam, the same `EvalProtocol`, the same `run_episode_safe`
crash contract, the same "real env when importable, fake otherwise" backend
switch — because that path is verified end to end and a second architecture
would be a second set of bugs.

Five things here are wrong-but-plausible if written from intuition. All five
are measured; the long form with source line numbers is `docs/ENV_ROBOTWIN.md`.

**1. `env_fps` is 250/15 = 16.6667 Hz, not the `frequency: 15` in the HDF5.**
That field is literally `save_freq`, a physics-step decimation factor written
into a slot labelled "frequency". `contracts.env_steps_per_segment(16.6667) =
4.4444`; the wrong 15.0 gives a tidy 4.0000 and makes every executed segment
11 % too slow. `SegmentClock` carries the 0.4444 remainder in its accumulator,
exactly as it does for LIBERO's 5.333. This module never resamples anything
itself — `loom.eval.policy` calls Team A's `canonical.to_env_rate` and there is
no second opinion about it here (`tests/test_eval.py::test_eval_has_no_second_resampler`).

**2. All 14 action channels are ABSOLUTE joint targets**, grippers included —
`action[t] == state[t+1]` bitwise over all 2 500 released episodes. This module
therefore hands `take_action(..., action_type="qpos")` the policy's 14-vector
unchanged. LIBERO's `(delta,)*6 + (hold,)` normaliser must never be shared with
this body.

**3. `orient_env_image` is the identity here.** SAPIEN hands eval true RGB, row
0 at the top. The *stored* JPEGs are BGR-as-RGB and `adapters.robotwin.decode_frame`
swaps them; the live path needs no transform. As in `loom.eval.libero`, the
transform is imported from Team A rather than written twice.

**4. A headless render failure is silent.** A GPU node with a broken Vulkan ICD
returns correctly-shaped, perfectly black frames and every number downstream
stays "real" while the score is zero. `RobotwinEnv.reset` therefore *measures*
per-camera variance and raises `BlackFrameError` below `MIN_PIXEL_VARIANCE`.
The statistics go into `EpisodeResult.extra` whether or not they pass.

**5. Success is RoboTwin's own `check_success()`**, which `take_action` already
calls and latches into `eval_success`. It is proven to fire rather than assumed:
`logs/teamf_robotwin_seam.py --check success_replay` replays a recorded expert
demonstration through *this adapter* and asserts the flag comes up.

Episode termination has two exits, and they are not the same thing: success
(latched, any point in the episode) and the per-task budget from
`env_cfg/task_config/_eval_step_limit.yml`, which is counted in `take_action`
calls. `take_action` becomes a **silent no-op** at the budget, so truncation is
tracked here rather than inferred from the environment doing nothing.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from contracts import EMBODIMENTS
from loom.eval import EpisodeResult, EvalProtocol

# Team A owns the train/eval orientation convention; for RoboTwin it is the
# identity, and importing it is still the point — a future disagreement is then
# a diff rather than an invisible near-zero score. Importing the adapter also
# registers `robotwin_aloha` in `contracts.EMBODIMENTS`.
from loom.data.adapters.robotwin import EVAL_VIEW_KEYS, SRC_FPS, orient_env_image

__all__ = [
    "SUITES", "TASKS", "TASK_IDS", "N_TASKS", "DEFAULT_PROTOCOL",
    "TASK_CONFIG_BY_SUITE", "EVAL_STEP_LIMIT", "MEASURED_CONTROL_FREQ",
    "MIN_PIXEL_VARIANCE", "EMBODIMENT", "VIEW_KEYS", "SETTLE_STEPS",
    "ROBOTWIN_ROOT", "ROBOTWIN_DATA", "ROBOTWIN_PYTHON",
    "EXPERT_CHECK", "SEED_STRIDE", "SEED_ATTEMPTS", "PROTOCOL_NOTE",
    "apply_headless_env", "ensure_robotwin_runtime", "robotwin_available",
    "make_env", "RobotwinEnv", "FakeRobotwinEnv", "BlackFrameError",
    "ExpertCheckExhausted", "UnstableScene",
    "orient_env_image", "extract_obs", "frame_stats",
    "task_name", "task_instruction", "n_tasks", "episode_seed",
    "run_episode", "run_episode_safe", "max_steps_for",
]


# ═══════════════════════════════════════════════════════════════════════════
#  INSTALLATION CONSTANTS
#
#  Every path and version the real RoboTwin run needs, in ONE block, all
#  overridable from the environment. Every value is verified by Team H's
#  `scripts/smoke_robotwin.py`, which is 81/81 green on an A100 for all four
#  PLAN §8 tasks. Nothing in this module reads a path outside this block.
# ═══════════════════════════════════════════════════════════════════════════

#: The RoboTwin 2.0 checkout. `cwd` must be this directory: `_embodiment_config.yml`
#: stores relative asset paths (`./assets/embodiments/aloha-agilex/`).
ROBOTWIN_ROOT = os.environ.get(
    "ROBOTWIN_ROOT",
    "/lustre/fsw/portfolios/edgeai/users/chrislin/projects/loom-deps/RoboTwin",
)

#: Assets + released demonstrations. Only the demos are read here, and only by
#: the success-replay check (`seed.txt` + `episode_*.hdf5`).
ROBOTWIN_DATA = os.environ.get(
    "ROBOTWIN_DATA",
    "/lustre/fsw/portfolios/edgeai/users/chrislin/datasets/loom/robotwin",
)

#: Separate interpreter for the real env. RoboTwin needs python 3.10 + sapien
#: 3.0.0b1 + numpy<2; the training venv is 3.13. `transformers` for the frozen
#: tower is not in that conda env either — it lives in an overlay directory that
#: the sbatch prepends to `PYTHONPATH`, so the shared env is never mutated.
ROBOTWIN_PYTHON = os.environ.get(
    "LOOM_ROBOTWIN_PYTHON",
    "/lustre/fsw/portfolios/edgeai/users/chrislin/envs/loom-robotwin/bin/python",
)
ROBOTWIN_PYTHONPATH_OVERLAY = os.environ.get(
    "LOOM_ROBOTWIN_OVERLAY",
    "/lustre/fsw/portfolios/edgeai/users/chrislin/envs/loom-robotwin-tf",
)

#: Headless Vulkan, applied by `apply_headless_env()` before `import sapien`.
#: `/usr/share/vulkan/icd.d/` on these compute nodes also holds Mesa's `lvp_icd`
#: — the llvmpipe SOFTWARE rasteriser. If the loader picks it, SAPIEN renders on
#: the CPU: no crash, plausible non-black images, ~1 fps, and not the GPU path
#: the benchmark assumes. That is why the ICD is pinned rather than left to the
#: loader's search order. Both spellings are set: SAPIEN bundles its own
#: `libvulkan` and the loader ≥1.3.207 renamed the variable.
VULKAN_ICD_CANDIDATES = (
    "/etc/vulkan/icd.d/nvidia_icd.json",
    "/usr/share/vulkan/icd.d/nvidia_icd.json",
)
HEADLESS_ENV = {
    "__GLX_VENDOR_LIBRARY_NAME": "nvidia",
    "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics",
}

#: Which Vulkan/CUDA device SAPIEN renders on. SAPIEN 3.0.0b1 exposes no
#: `render.set_device`, and `_base_task.setup_scene` constructs
#: `sapien.SapienRenderer()` with **no device**, so every worker would render on
#: physical GPU 0 regardless of `CUDA_VISIBLE_DEVICES` — the SAPIEN twin of the
#: `MUJOCO_EGL_DEVICE_ID` problem. `_pin_render_device()` wraps the renderer
#: constructor so the device is explicit. `"cuda"` means "the current CUDA
#: device", which under a per-worker `CUDA_VISIBLE_DEVICES` is the right one.
SAPIEN_DEVICE = os.environ.get("LOOM_SAPIEN_DEVICE", "cuda")

#: PLAN §8 splits RoboTwin by domain-randomisation setting, not by task family.
#: R0-B is scored on **clean** (PLAN §7); `rand` is a second run with
#: `--suites randomized`. Selection is by task config file and the two files
#: differ in exactly two blocks: `domain_randomization.*` and `eval_instruction`
#: (`seen` vs `unseen`) — so evaluating "clean" with unseen instructions would
#: silently measure language generalisation against the clean baseline column.
SUITES: tuple[str, ...] = ("clean", "randomized")
TASK_CONFIG_BY_SUITE = {"clean": "demo_clean", "randomized": "demo_randomized"}

#: The four per-task columns PLAN §8 asks for, in column order. `TASKS` are the
#: table's display names; `TASK_IDS` are RoboTwin's module *and* class names
#: under `envs/`, which is what `importlib.import_module(f"envs.{id}")` needs.
TASKS: tuple[str, ...] = (
    "hanging mug",
    "turn switch",
    "place can basket",
    "handover block",
)
TASK_IDS: tuple[str, ...] = (
    "hanging_mug",
    "turn_switch",
    "place_can_basket",
    "handover_block",
)

N_TASKS = {s: len(TASKS) for s in SUITES}

#: `env_cfg/task_config/_eval_step_limit.yml`, counted in `take_action` calls,
#: loaded into `self.step_lim` only when `args["eval_mode"] = True`. Not one
#: number for every task the way LIBERO's 512 is. Read from the checkout when it
#: is present; these are the values in the pinned commit and are what the
#: protocol's `max_steps` is derived from.
EVAL_STEP_LIMIT: dict[str, int] = {
    "hanging_mug": 900,
    "turn_switch": 400,
    "place_can_basket": 700,
    "handover_block": 800,
}

#: 250 Hz physics decimated by `save_freq: 15`. Measured on an A100 by
#: `scripts/smoke_robotwin.py` (median 15.0 physx steps per recorded frame), and
#: identical to `adapters.robotwin.SRC_FPS` / `EMBODIMENTS[...].env_fps`, which
#: is what `SegmentClock` actually reads. Do NOT use the HDF5's `frequency: 15`.
MEASURED_CONTROL_FREQ = 250.0 / 15.0

#: A camera stream whose pixel variance is below this did not render. A real
#: RoboTwin frame scores 300–2800; the lowest ever observed across the four
#: tasks is 298.55 (turn_switch front_camera). A black frame scores 0.
MIN_PIXEL_VARIANCE = float(os.environ.get("LOOM_ROBOTWIN_MIN_PIXEL_VAR", "5.0"))

EMBODIMENT = "robotwin_aloha"

#: One rate, three places it could have disagreed. `SegmentClock` reads
#: `EmbodimentSpec.env_fps`, the cache was built at `adapters.robotwin.SRC_FPS`,
#: and the constant above is what Team H measured on the live simulator. If they
#: ever diverge, the executed segment length is wrong and nothing else says so.
if not (MEASURED_CONTROL_FREQ == SRC_FPS == EMBODIMENTS[EMBODIMENT].env_fps):
    raise RuntimeError(
        f"env_fps disagreement: measured {MEASURED_CONTROL_FREQ}, adapter "
        f"{SRC_FPS}, contracts {EMBODIMENTS[EMBODIMENT].env_fps}"
    )

#: The live simulator's camera names, in the V order the feature cache was built
#: with (`adapters.robotwin.EVAL_VIEW_KEYS`, whose training-side twin is
#: `cam_head, cam_left_wrist, cam_right_wrist, cam_third_view`). Swapping this
#: order is a silent near-zero score, which is why there is one definition of it
#: and this module imports rather than restates it.
VIEW_KEYS: tuple[str, ...] = tuple(EVAL_VIEW_KEYS)

#: LIBERO needs 15 dummy actions after `set_init_state` so the scene settles.
#: RoboTwin does not: `setup_demo` already runs the scene to rest and RoboTwin's
#: own eval script steps the policy immediately.
SETTLE_STEPS = 0

#: Run the scripted expert on a candidate seed before scoring the policy on it,
#: and skip the seed if the expert cannot solve the scene. This is RoboTwin's
#: own protocol (`scripts/eval_policy_xpolicylab.py`, `expert_check=True` by
#: default) and it changes the number materially — without it, unsolvable scenes
#: are counted as policy failures. It costs one full expert episode per trial.
EXPERT_CHECK = os.environ.get("LOOM_ROBOTWIN_EXPERT_CHECK", "1") not in ("0", "false", "")

#: RoboTwin draws seeds from a single advancing counter `st_seed = 100000*(1+seed)`
#: and consumes them in order, which is inherently serial. This harness is
#: sharded across GPUs, so each (task, episode) gets its own disjoint search
#: window of `SEED_ATTEMPTS` seeds starting at `st_seed + episode*SEED_STRIDE`.
#: Same origin, same skip rule, deterministic and parallel-safe. Stated with the
#: protocol because it is a deviation.
SEED_STRIDE = int(os.environ.get("LOOM_ROBOTWIN_SEED_STRIDE", "64"))
SEED_ATTEMPTS = int(os.environ.get("LOOM_ROBOTWIN_SEED_ATTEMPTS", "64"))

PROTOCOL_NOTE = (
    "RoboTwin 2.0's own protocol, replicated: 100 episodes/task with "
    "EXPERT-CHECKED seeds (scripts/eval_policy_xpolicylab.py defaults to "
    "test_num=100 and expert_check=True; a seed the scripted expert cannot "
    "solve is SKIPPED, not counted as a policy failure), seeds drawn from "
    "st_seed = 100000*(1+seed), demo_clean for the `clean` column. R0-B is "
    "scored on clean (PLAN §7). max_steps here is the largest per-task budget; "
    "each task is actually capped at its own _eval_step_limit.yml value "
    "(hanging_mug 900, turn_switch 400, place_can_basket 700, handover_block "
    "800), which is what RoboTwin counts. Deviation from upstream: seeds are "
    "consumed from a per-episode disjoint window (episode*64) instead of one "
    "serial counter, so the run shards across GPUs deterministically."
)

DEFAULT_PROTOCOL = EvalProtocol(
    bench="robotwin",
    episodes_per_task=100,
    n_tasks=len(TASKS),
    suites=("clean",),
    seeds=(0,),
    max_steps=max(EVAL_STEP_LIMIT.values()),
    notes=PROTOCOL_NOTE,
)


# ═══════════════════════════════════════════════════════════════════════════
#  RUNTIME SETUP
# ═══════════════════════════════════════════════════════════════════════════

class BlackFrameError(RuntimeError):
    """A camera returned a correctly-shaped frame with no content in it."""


class UnstableScene(RuntimeError):
    """`setup_demo` raised `UnStableError`: the sampled objects settled badly."""


class ExpertCheckExhausted(RuntimeError):
    """No seed in this episode's window produced a scene the expert can solve."""


def vulkan_icd() -> str | None:
    for path in VULKAN_ICD_CANDIDATES:
        if os.path.isfile(path):
            return path
    return None


def apply_headless_env() -> None:
    """Pin the Vulkan ICD and the driver capabilities. Before `import sapien`.

    Idempotent and non-destructive (`setdefault`), so a sbatch that already
    sourced `robotwin_env.sh` wins. `DISPLAY` is removed because
    `render_freq > 0` would otherwise open a GUI viewer and fail headless.
    """
    icd = os.environ.get("VK_ICD_FILENAMES") or os.environ.get("VK_DRIVER_FILES") or vulkan_icd()
    if icd:
        os.environ.setdefault("VK_ICD_FILENAMES", icd)
        os.environ.setdefault("VK_DRIVER_FILES", icd)
    for k, v in HEADLESS_ENV.items():
        os.environ.setdefault(k, v)
    os.environ.pop("DISPLAY", None)


def _pin_render_device() -> str:
    """Give `sapien.SapienRenderer()` an explicit device. Returns what it got.

    `envs/_base_task.setup_scene` calls `sapien.SapienRenderer()` with no
    argument, and SAPIEN then picks a physical Vulkan device on its own —
    typically index 0 for every process, so eight eval workers would serialise
    the simulator behind one GPU while their policies ran on eight. There is no
    `sapien.render.set_device` in 3.0.0b1, so the constructor is wrapped here.
    `"cuda"` resolves through the CUDA runtime, which *does* honour
    `CUDA_VISIBLE_DEVICES`, and `runner._init_worker` sets that per worker.
    """
    import sapien                                          # noqa: PLC0415

    if getattr(sapien.SapienRenderer, "_loom_pinned", False):
        return SAPIEN_DEVICE
    original = sapien.SapienRenderer

    def _renderer(device=None):                            # noqa: ANN001
        if device is None:
            try:
                device = sapien.Device(SAPIEN_DEVICE)
            except Exception:                              # noqa: BLE001 — let SAPIEN choose
                device = None
        if device is None:
            return original()
        # `SapienRenderer.__init__(self, device: Device = None)` is pybind11
        # keyword-only in 3.0.0b1 — passing it positionally is a TypeError.
        try:
            return original(device=device)
        except TypeError:
            return original()

    _renderer._loom_pinned = True                          # type: ignore[attr-defined]
    sapien.SapienRenderer = _renderer                      # type: ignore[assignment]
    sapien.render.SapienRenderer = _renderer               # type: ignore[assignment]
    return SAPIEN_DEVICE


#: Status of the one-time runtime setup, reported into the results JSON.
ROBOTWIN_RUNTIME_STATUS: str | None = None


def ensure_robotwin_runtime() -> str:
    """Idempotent, once per process, before anything touches the real env.

    Deliberately NOT done at import: it chdirs, mutates `sys.path` and patches a
    SAPIEN constructor, and `loom.eval.robotwin` is imported by the CPU test
    suite too.
    """
    global ROBOTWIN_RUNTIME_STATUS
    if ROBOTWIN_RUNTIME_STATUS is not None:
        return ROBOTWIN_RUNTIME_STATUS
    apply_headless_env()
    root = str(Path(ROBOTWIN_ROOT).resolve())
    # `_embodiment_config.yml` stores relative asset paths; RoboTwin resolves
    # everything off its own root, so cwd is load-bearing.
    os.chdir(root)
    for p in (root, os.path.join(root, "scripts"),
              os.path.join(root, "description", "utils")):
        if p not in sys.path:
            sys.path.insert(0, p)
    device = _pin_render_device()
    ROBOTWIN_RUNTIME_STATUS = (
        f"cwd={root} icd={os.environ.get('VK_ICD_FILENAMES')} "
        f"sapien_device={device}"
    )
    return ROBOTWIN_RUNTIME_STATUS


_ROBOTWIN_OK: bool | None = None


def robotwin_available() -> bool:
    """True when the real simulator can be driven. Cached; never raises.

    `import envs.<task>` needs a GPU — curobo allocates a CUDA tensor as a class
    body default argument at import time, and on a CPU node that surfaces as the
    misleading `ImportError: cannot import name 'CuroboPlanner'`. So the check
    is: the checkout is there, `sapien` imports, and torch sees a GPU. Anything
    less and `make_env` falls back to `FakeRobotwinEnv`, exactly as the LIBERO
    path falls back to `FakeLiberoEnv`.
    """
    global _ROBOTWIN_OK
    if _ROBOTWIN_OK is None:
        try:
            import importlib                               # noqa: PLC0415

            import torch                                   # noqa: PLC0415

            ok = Path(ROBOTWIN_ROOT, "envs", "_base_task.py").is_file()
            ok = ok and torch.cuda.is_available()
            if ok:
                apply_headless_env()
                importlib.import_module("sapien")
            _ROBOTWIN_OK = bool(ok)
        except Exception:                                  # noqa: BLE001
            _ROBOTWIN_OK = False
    return _ROBOTWIN_OK


# ═══════════════════════════════════════════════════════════════════════════
#  SUITES / TASKS
# ═══════════════════════════════════════════════════════════════════════════

def task_config(suite: str) -> str:
    try:
        return TASK_CONFIG_BY_SUITE[suite]
    except KeyError:
        raise KeyError(
            f"unknown RoboTwin suite {suite!r}; have {sorted(TASK_CONFIG_BY_SUITE)}"
        ) from None


def n_tasks(suite: str) -> int:
    return N_TASKS.get(suite, len(TASKS))


def task_id_of(task_id: int) -> str:
    return TASK_IDS[int(task_id) % len(TASK_IDS)]


def task_name(suite: str, task_id: int) -> str:
    """`clean/turn_switch` — RoboTwin's own task id, so it greps against theirs."""
    return f"{suite}/{task_id_of(task_id)}"


def task_display_name(task_id: int) -> str:
    """The PLAN §8 column header for this task index."""
    return TASKS[int(task_id) % len(TASKS)]


def max_steps_for(task_id: int, protocol_max: int | None = None) -> int:
    """This task's own `_eval_step_limit.yml` budget, clipped to the protocol."""
    lim = EVAL_STEP_LIMIT[task_id_of(task_id)]
    return int(min(lim, protocol_max)) if protocol_max else int(lim)


def task_instruction(suite: str, task_id: int) -> str:
    """The language goal used when the env cannot supply its own.

    The real per-episode instruction is drawn by RoboTwin from
    `description/task_instruction/<task>.json` (50 `seen` templates, 10 held-out
    `unseen`) and is set on the env; `run_episode` prefers `env.language` for
    exactly that reason. This is the deterministic fallback, and it is also what
    the fake backend uses.
    """
    return task_display_name(task_id)


def episode_seed(seed: int, bench: str, suite: str, task_id: int, episode: int) -> int:
    """RoboTwin's seed origin, not a SHA-256 of the tuple.

    `runner.iter_work` uses the bench module's `episode_seed` when it defines
    one. RoboTwin's own eval starts at `st_seed = 100000 * (1 + seed)` and walks
    forward, skipping seeds whose expert fails; reproducing the *origin* matters
    because a released `seed.txt` is the list of seeds that survived that walk.
    Each episode owns a disjoint window of `SEED_STRIDE` seeds so the search can
    run in parallel.
    """
    return 100000 * (1 + int(seed)) + int(episode) * SEED_STRIDE


# ═══════════════════════════════════════════════════════════════════════════
#  OBSERVATION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def frame_stats(obs: dict) -> dict[str, dict[str, float]]:
    """Per-camera mean/std/min/max/var of an extracted observation.

    Recorded into `EpisodeResult.extra` on the first frame of every episode.
    An all-black render is the classic silent headless-GPU failure: every shape
    is right, every number downstream is "real", and the score is zero. The only
    defence is to look at the pixels and write down what was there.
    """
    out: dict[str, dict[str, float]] = {}
    for key in VIEW_KEYS:
        img = obs.get(key)
        if img is None:
            continue
        a = np.asarray(img, dtype=np.float32)
        out[key] = {
            "mean": round(float(a.mean()), 4),
            "std": round(float(a.std()), 4),
            "var": round(float(a.var()), 4),
            "min": float(a.min()),
            "max": float(a.max()),
        }
    return out


def check_not_black(obs: dict, where: str = "") -> dict[str, dict[str, float]]:
    """Raise `BlackFrameError` if any camera is below `MIN_PIXEL_VARIANCE`."""
    stats = frame_stats(obs)
    missing = [k for k in VIEW_KEYS if k not in stats]
    if missing:
        raise BlackFrameError(
            f"{where}: observation has no {missing}; present: {sorted(obs)}"
        )
    dead = {k: v for k, v in stats.items() if v["var"] < MIN_PIXEL_VARIANCE}
    if dead:
        raise BlackFrameError(
            f"{where}: {sorted(dead)} rendered with pixel variance "
            f"{ {k: v['var'] for k, v in dead.items()} } < {MIN_PIXEL_VARIANCE}. "
            f"A headless Vulkan/EGL failure returns correctly-shaped BLACK frames "
            f"and everything downstream still 'works'. Check VK_ICD_FILENAMES "
            f"({os.environ.get('VK_ICD_FILENAMES')}) and that SAPIEN did not fall "
            f"through to the llvmpipe software rasteriser."
        )
    return stats


def extract_obs(raw: dict) -> dict:
    """Raw RoboTwin obs -> the dict the policy's featuriser consumes.

    `get_obs()` returns `{"observation": {cam: {"rgb": (240,320,3) uint8, ...}},
    "joint_action": {"vector": (14,) ...}, ...}`. The 14-vector is
    `[L_arm j1..j6 | L_grip | R_arm j1..j6 | R_grip]` — the *same* coordinates
    as the action space, which is why the decoder can difference its own output
    against the proprio it was conditioned on.
    """
    out: dict[str, Any] = {}
    cams = raw.get("observation") or {}
    for key in VIEW_KEYS:
        cam = cams.get(key)
        rgb = cam.get("rgb") if isinstance(cam, dict) else cam
        if rgb is not None:
            out[key] = orient_env_image(np.asarray(rgb))
    vec = (raw.get("joint_action") or {}).get("vector")
    if vec is not None:
        out["state"] = np.asarray(vec, dtype=np.float32).reshape(-1)
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  ENV SEAM
# ═══════════════════════════════════════════════════════════════════════════

class EnvLike(Protocol):
    """The whole surface the harness needs. Real and fake both satisfy it."""

    def reset(self) -> dict: ...
    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]: ...
    def close(self) -> None: ...


def make_env(
    suite: str,
    task_id: int,
    seed: int,
    *,
    trial_id: int = 0,
    backend: str | None = None,
    max_steps: int | None = None,
    **kw: Any,
) -> EnvLike:
    """The seam. `backend` in {None (auto), "robotwin", "fake"}.

    Auto picks the real simulator when it can be driven and the fake otherwise,
    so the CPU test suite runs today and the GPU run is one import away.
    """
    backend = backend or ("robotwin" if robotwin_available() else "fake")
    if backend == "fake":
        return FakeRobotwinEnv(suite, task_id, seed, trial_id=trial_id,
                               max_steps=max_steps, **kw)
    if backend in ("robotwin", "libero"):     # "libero" only from a stale --backend
        return RobotwinEnv(suite, task_id, seed, trial_id=trial_id,
                           max_steps=max_steps, **kw)
    raise ValueError(f"unknown env backend {backend!r}")


def _load_task_args(root: Path, task: str, config: str) -> dict:
    """`env_cfg/task_config/<config>.yml` -> the kwargs `setup_demo` wants.

    A transcription of `scripts/eval_policy_xpolicylab.load_task_args` restricted
    to a single (dual-arm) embodiment. `eval_mode=True` is what loads
    `_eval_step_limit.yml` into `step_lim`; without it `step_lim` stays `None`
    and `take_action` never terminates on budget. `save_data=False` keeps
    `_take_picture` off disk and `render_freq=0` keeps the GUI viewer shut.
    """
    import yaml                                            # noqa: PLC0415

    cfg_root = root / "env_cfg" / "task_config"
    with open(cfg_root / f"{config}.yml", "r", encoding="utf-8") as f:
        args = yaml.safe_load(f)
    with open(cfg_root / "_embodiment_config.yml", "r", encoding="utf-8") as f:
        embodiments = yaml.safe_load(f)
    with open(cfg_root / "_camera_config.yml", "r", encoding="utf-8") as f:
        cameras = yaml.safe_load(f)

    args["task_name"] = task
    args["task_config"] = config
    head = args["camera"]["head_camera_type"]
    args["head_camera_h"] = cameras[head]["h"]
    args["head_camera_w"] = cameras[head]["w"]

    emb = args["embodiment"]
    if len(emb) != 1:
        raise ValueError(f"expected a single embodiment, got {emb}")
    robot_file = embodiments[emb[0]]["file_path"]
    args["left_robot_file"] = robot_file
    args["right_robot_file"] = robot_file
    args["dual_arm_embodied"] = True
    args["embodiment_name"] = str(emb[0])
    for side in ("left", "right"):
        with open(os.path.join(robot_file, "config.yml"), "r", encoding="utf-8") as f:
            args[f"{side}_embodiment_config"] = yaml.safe_load(f)

    args["eval_mode"] = True
    args["save_data"] = False
    args["render_freq"] = 0
    args["eval_video_log"] = False
    dt = args.setdefault("data_type", {})
    dt["rgb"] = True
    dt["qpos"] = True
    dt["endpose"] = True
    dt["pointcloud"] = False
    return args


def _eval_step_limit(root: Path, task: str) -> int:
    """The task's budget, read from the checkout when present."""
    try:
        import yaml                                        # noqa: PLC0415

        path = root / "env_cfg" / "task_config" / "_eval_step_limit.yml"
        with open(path, "r", encoding="utf-8") as f:
            return int(yaml.safe_load(f)[task])
    except Exception:                                      # noqa: BLE001
        return int(EVAL_STEP_LIMIT[task])


class RobotwinEnv:
    """Thin adapter over a RoboTwin 2.0 `Base_Task`, with RoboTwin's own protocol.

    Constructed exactly as `scripts/eval_policy_xpolicylab.py` constructs it, so
    the number this harness produces is comparable with the published table:

    * `setup_demo(now_ep_num=..., seed=..., is_test=True, **args)` with
      `eval_mode=True`, `save_data=False`, `render_freq=0`;
    * an **expert check** first — the scripted expert runs the sampled scene and
      the seed is skipped if it cannot solve it, rather than the policy being
      blamed for an unsolvable scene;
    * the episode's instruction drawn from RoboTwin's own template pool for the
      suite's `seen`/`unseen` setting.

    The three things it adds are all guards: the render is measured rather than
    assumed non-black, truncation at `step_lim` is tracked (because `take_action`
    silently no-ops there), and `UnStableError` is re-raised as `UnstableScene`
    so a bad scene is distinguishable from a bad policy in the results JSON.
    """

    def __init__(
        self,
        suite: str,
        task_id: int,
        seed: int,
        *,
        trial_id: int = 0,
        max_steps: int | None = None,
        expert_check: bool | None = None,
        seed_attempts: int = SEED_ATTEMPTS,
        **_ignored: Any,
    ) -> None:
        ensure_robotwin_runtime()
        import importlib                                   # noqa: PLC0415

        from envs.utils.create_actor import UnStableError  # noqa: PLC0415

        self.suite = suite
        self.task_id = int(task_id)
        self.task = task_id_of(task_id)
        self.trial_id = int(trial_id)
        self.config = task_config(suite)
        self.expert_check = EXPERT_CHECK if expert_check is None else bool(expert_check)
        self._unstable = UnStableError

        root = Path(ROBOTWIN_ROOT).resolve()
        self.args = _load_task_args(root, self.task, self.config)
        self.step_lim = _eval_step_limit(root, self.task)
        self.max_steps = min(int(max_steps), self.step_lim) if max_steps else self.step_lim

        module = importlib.import_module(f"envs.{self.task}")
        self.env = getattr(module, self.task)()

        self.requested_seed = int(seed)
        self.seed, self.n_seeds_skipped, self._episode_info = self._pick_seed(
            int(seed), int(seed_attempts))

        self.env.setup_demo(now_ep_num=self.trial_id, seed=self.seed,
                            is_test=True, **self.args)
        self.language = self._instruction()
        self.env.set_instruction(instruction=self.language)

        self.n_steps = 0
        self.truncated = False
        self.frame_stats: dict[str, dict[str, float]] = {}
        self.closed = False

    # ── RoboTwin's seed protocol ──────────────────────────────────────────

    def _pick_seed(self, start: int, attempts: int) -> tuple[int, int, dict]:
        """Walk forward from `start` until the scripted expert solves the scene.

        This is `eval_policy_xpolicylab.eval_remote_policy`'s loop: `setup_demo`,
        `play_once`, `close_env`, and advance the seed on `UnStableError`, on any
        expert exception, or when `plan_success and check_success()` is False.
        The episode that the policy is then scored on is rebuilt from the *same*
        seed, which reconstructs the identical scene.
        """
        if not self.expert_check:
            return start, 0, {"info": {}}
        for k in range(attempts):
            seed = start + k
            try:
                self.env.setup_demo(now_ep_num=self.trial_id, seed=seed,
                                    is_test=True, **self.args)
                info = self.env.play_once()
                ok = bool(self.env.plan_success and self.env.check_success())
            except self._unstable:
                self._close_quietly()
                continue
            except Exception:                              # noqa: BLE001 — RoboTwin skips these too
                self._close_quietly()
                continue
            self._close_quietly()
            if ok:
                return seed, k, (info if isinstance(info, dict) else {"info": {}})
        raise ExpertCheckExhausted(
            f"{self.task}: the scripted expert solved none of the {attempts} seeds "
            f"[{start}, {start + attempts}) — the seed window is exhausted, which "
            f"RoboTwin would report as `expert_failed` and skip. This is an "
            f"environment/seed outcome, not a policy score."
        )

    def _instruction(self) -> str:
        """RoboTwin's own per-episode instruction, `seen` or `unseen` per suite.

        `demo_clean` evaluates on language the policy trained on and
        `demo_randomized` swaps in the 10 held-out phrasings; since LOOM
        conditions on language (`ℓ` feeds both `E` and `π_c`), getting this
        backwards would silently measure language generalisation against the
        clean baseline column. Falls back to the task name exactly as
        `build_instruction` does.

        **This path needs the expert check.** `generate_episode_descriptions`
        keys off the `episode_info` that `play_once()` returns — which object was
        sampled, which of its attributes the phrasing may refer to — so with
        `expert_check=False` it prints `"Episode 0: No valid instructions found"`
        and this falls back to the task name. Measured on `turn_switch`,
        seed 100000: with the check on, `'Trigger the switch with darker
        rectangle with precision'`; with it off, `'turn switch'`. That is another
        reason the protocol default is on, and it is why a fast seam run and a
        scoring run do not see the same language.
        """
        kind = str(self.args.get("eval_instruction", "seen")).strip().lower()
        try:
            from generate_episode_instructions import (   # noqa: PLC0415
                generate_episode_descriptions,
            )

            results = generate_episode_descriptions(
                self.task, [self._episode_info.get("info", {})], 100)
            cands = results[0].get(kind) or []
            if cands:
                # RoboTwin uses np.random.choice off a global RNG it re-seeds in
                # `_init_task_env_`; a local generator keyed on the episode seed
                # is the same draw distribution and is reproducible.
                rng = np.random.default_rng(self.seed)
                return str(cands[int(rng.integers(len(cands)))])
        except Exception:                                  # noqa: BLE001
            pass
        return task_display_name(self.task_id)

    # ── surface ───────────────────────────────────────────────────────────

    def reset(self) -> dict:
        """The scene is already built by `setup_demo`; this reads the first obs.

        RoboTwin has no `reset()` that leaves the scene intact — `setup_demo` is
        the reset — so re-entering it would rebuild the world and throw away the
        expert-checked seed. The all-black check runs here, on the first frame.
        """
        raw = self.env.get_obs()
        obs = extract_obs(raw)
        self.frame_stats = check_not_black(obs, f"{self.task}/seed{self.seed} reset")
        self._raw = raw
        return raw

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        """One `take_action` call: 14 ABSOLUTE joint targets, `action_type="qpos"`.

        `take_action` returns silently once `take_action_cnt == step_lim` or
        `eval_success` latches, so both are tracked here; a caller that only
        watched the environment would loop forever against a no-op.
        """
        if self.truncated or self.env.eval_success:
            return self._raw, 0.0, True, {"success": bool(self.env.eval_success),
                                          "truncated": self.truncated}
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape != (EMBODIMENTS[EMBODIMENT].dof,):
            raise ValueError(
                f"action must be ({EMBODIMENTS[EMBODIMENT].dof},), got {a.shape}"
            )
        self.env.take_action(a, action_type="qpos")
        self.n_steps = int(self.env.take_action_cnt)
        self.truncated = self.n_steps >= self.max_steps
        success = bool(self.env.eval_success)
        self._raw = self.env.get_obs()
        done = bool(success or self.truncated)
        return self._raw, float(success), done, {"success": success,
                                                 "truncated": self.truncated}

    def check_success(self) -> bool:
        """RoboTwin's own goal predicate. Latched by `take_action`.

        `eval_success` is the latch and `check_success()` is the live predicate;
        the former is what RoboTwin scores on, so it is checked first. Proven to
        fire by replaying a recorded expert demonstration through this adapter —
        see `logs/teamf_robotwin_seam.py --check success_replay`.
        """
        try:
            return bool(self.env.eval_success or self.env.check_success())
        except Exception:                                  # noqa: BLE001
            return bool(getattr(self.env, "eval_success", False))

    def _close_quietly(self) -> None:
        try:
            self.env.close_env()
        except Exception:                                  # noqa: BLE001
            pass

    def close(self) -> None:
        self._close_quietly()
        self.closed = True

    # ── diagnostics carried into the results JSON ─────────────────────────

    def episode_extra(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "task_config": self.config,
            "seed_requested": self.requested_seed,
            "seed_used": self.seed,
            "seeds_skipped_by_expert_check": self.n_seeds_skipped,
            "expert_check": self.expert_check,
            "step_lim": self.step_lim,
            "instruction": self.language,
            "frame_stats": self.frame_stats,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  FAKE ENV
# ═══════════════════════════════════════════════════════════════════════════

class FakeRobotwinEnv:
    """Same surface, random outcomes, no SAPIEN. The twin of `FakeLiberoEnv`.

    Deterministic in `seed`, and it *validates* what the policy sends it — shape,
    dtype, finiteness, bounds — so the 14-channel absolute-target rate-conversion
    path is exercised for real on a CPU node rather than merely executed.
    """

    def __init__(
        self,
        suite: str = "clean",
        task_id: int = 0,
        seed: int = 0,
        *,
        trial_id: int = 0,
        image_size: tuple[int, int] = (240, 320),
        p_success: float = 0.5,
        max_steps: int | None = None,
        crash_at: int | None = None,
        strict_bounds: bool = False,
        **_ignored: Any,
    ) -> None:
        self.suite, self.task_id, self.seed = suite, int(task_id), int(seed)
        self.task = task_id_of(task_id)
        self.trial_id = int(trial_id)
        self.spec = EMBODIMENTS[EMBODIMENT]
        self.image_size = image_size
        self.crash_at = crash_at
        self.strict_bounds = strict_bounds
        self.step_lim = EVAL_STEP_LIMIT[self.task]
        self.max_steps = min(int(max_steps), self.step_lim) if max_steps else self.step_lim
        self.language = task_instruction(suite, task_id)

        rng = np.random.default_rng(self.seed)
        self._will_succeed = bool(rng.random() < p_success)
        self._solve_step = int(rng.integers(5, max(6, self.max_steps)))
        self._rng = rng
        self.n_steps = 0
        self.truncated = False
        self.closed = False
        self.frame_stats: dict[str, dict[str, float]] = {}

    def reset(self) -> dict:
        self.n_steps = 0
        self.truncated = False
        raw = self._obs()
        self.frame_stats = check_not_black(extract_obs(raw), "fake reset")
        return raw

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        a = np.asarray(action)
        if a.shape != (self.spec.dof,):
            raise ValueError(f"action must be ({self.spec.dof},), got {a.shape}")
        if not np.isfinite(a).all():
            raise ValueError("action contains nan/inf")
        if self.strict_bounds:
            lo = np.asarray(self.spec.action_low)
            hi = np.asarray(self.spec.action_high)
            if (a < lo - 1e-4).any() or (a > hi + 1e-4).any():
                raise ValueError(f"action outside {self.spec.name} bounds: {a}")
        self.n_steps += 1
        if self.crash_at is not None and self.n_steps >= self.crash_at:
            raise RuntimeError(f"FakeRobotwinEnv: injected crash at step {self.n_steps}")
        self.truncated = self.n_steps >= self.max_steps
        success = self._will_succeed and self.n_steps >= self._solve_step
        return self._obs(), float(success), bool(success or self.truncated), \
            {"success": success, "truncated": self.truncated}

    def check_success(self) -> bool:
        return self._will_succeed and self.n_steps >= self._solve_step

    def close(self) -> None:
        self.closed = True

    def episode_extra(self) -> dict[str, Any]:
        return {"task": self.task, "task_config": task_config(self.suite),
                "seed_used": self.seed, "step_lim": self.step_lim,
                "instruction": self.language, "frame_stats": self.frame_stats,
                "backend": "fake"}

    def _obs(self) -> dict:
        h, w = self.image_size
        cams = {k: {"rgb": self._rng.integers(0, 256, (h, w, 3), dtype=np.uint8)}
                for k in VIEW_KEYS}
        vec = self._rng.normal(size=self.spec.dof).astype(np.float32)
        vec[6] = vec[13] = float(self._rng.random())
        return {"observation": cams, "joint_action": {"vector": vec}}


# ═══════════════════════════════════════════════════════════════════════════
#  EPISODE LOOP
#
#  Structurally `loom.eval.libero.run_episode`, with the two RoboTwin
#  differences made explicit rather than hidden in a flag: no settle phase, and
#  a per-task step budget that the environment enforces by *doing nothing*.
# ═══════════════════════════════════════════════════════════════════════════

def _episode_success(done: bool, info: Any, env: Any) -> bool:
    """RoboTwin's success signal, however this env chooses to expose it."""
    if isinstance(info, dict) and "success" in info:
        return bool(np.asarray(info["success"]).all())
    fn = getattr(env, "check_success", None)
    if callable(fn):
        try:
            return bool(fn())
        except Exception:                                  # noqa: BLE001
            pass
    return bool(done) and not bool(getattr(env, "truncated", False))


def run_episode(
    policy: Any,
    env: EnvLike,
    instruction: str,
    max_steps: int,
    *,
    settle_steps: int = SETTLE_STEPS,
    obs_fn: Callable[[dict], dict] = extract_obs,
) -> dict:
    """One closed-loop episode. Returns the measurable fields of `EpisodeResult`.

    Success latches: the task is solved if `eval_success` comes up at any point,
    which is RoboTwin's own convention. `max_steps` is the protocol cap; the env
    additionally stops at its own `_eval_step_limit.yml` budget and says so via
    `truncated`, so an episode that ran out of budget is distinguishable from one
    that ran out of protocol.
    """
    raw = env.reset()
    for _ in range(settle_steps):                          # 0 for RoboTwin
        raw, _, _, _ = env.step(np.asarray(obs_fn(raw)["state"], dtype=np.float32))

    # The env's own per-episode instruction when it has one: RoboTwin draws it
    # from the task's template pool and `ℓ` feeds both `E` and `π_c`, so scoring
    # on a hand-made string would not be the published protocol.
    lang = str(getattr(env, "language", "") or instruction)

    policy.reset()
    success, steps = False, 0
    for steps in range(1, max_steps + 1):
        action = policy.act(obs_fn(raw), lang)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        raw, _, done, info = env.step(action)
        if _episode_success(done, info, env):
            success = True
            break
        if bool(getattr(env, "truncated", False)) or (isinstance(info, dict)
                                                      and info.get("truncated")):
            break

    out: dict[str, Any] = {
        "success": success,
        "steps": steps,
        "hit_step_cap": bool(not success and (steps >= max_steps
                                              or bool(getattr(env, "truncated", False)))),
        "n_replans": int(getattr(policy, "replans", 0)) or None,
    }
    extra: dict[str, Any] = {}
    describe = getattr(env, "episode_extra", None)
    if callable(describe):
        extra.update(describe())
    extra["instruction_used"] = lang
    # The segment clock's own account of itself: replans x steps_per_segment
    # against steps actually dispatched. RoboTwin's 4.4444 is fractional on
    # purpose and a truncating clock would show up here as a growing drift.
    clock = getattr(policy, "clock", None)
    if clock is not None:
        extra["clock"] = {
            "env_fps": round(float(clock.env_fps), 6),
            "steps_per_segment": round(float(clock.steps_per_segment), 6),
            "n_replans": int(clock.n_replans),
            "n_steps_dispatched": int(clock.n_steps_dispatched),
            "drift": round(float(clock.drift), 9),
        }
    summary = getattr(policy, "op_stats_summary", None)
    if callable(summary):
        extra.update(summary() or {})
    if extra:
        out["extra"] = extra
    return out


def run_episode_safe(
    policy: Any,
    env_factory: Callable[[], EnvLike],
    instruction: str,
    max_steps: int,
    *,
    record: EpisodeResult,
    **kw: Any,
) -> EpisodeResult:
    """`run_episode` with the crash contract: one bad episode is a failure, not a run.

    `ExpertCheckExhausted` and `UnstableScene` are recorded with a `skipped`
    marker: RoboTwin's own protocol skips those seeds rather than counting them
    against the policy, and burying them in a traceback would make an
    environment problem read as a low score.
    """
    env = None
    try:
        env = env_factory()
        out = run_episode(policy, env, instruction, max_steps, **kw)
        for k, v in out.items():
            if k == "extra" and isinstance(v, dict):
                record.extra.update(v)
            else:
                setattr(record, k, v)
    except (ExpertCheckExhausted, UnstableScene) as e:
        record.success = False
        record.extra["skipped"] = type(e).__name__
        record.error = traceback.format_exc()
    except Exception:                                      # noqa: BLE001 — one episode, not the run
        record.success = False
        record.error = traceback.format_exc()
        if env is not None:
            stats = getattr(env, "frame_stats", None)
            if stats:
                record.extra.setdefault("frame_stats", stats)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:                              # noqa: BLE001
                pass
    return record
