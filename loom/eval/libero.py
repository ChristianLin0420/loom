"""LOOM — closed-loop LIBERO evaluation, behind a seam.

There is no working LIBERO installation on this cluster yet (Team G is building
one in a separate conda env: python 3.10 + robosuite 1.4 + numpy<2, because
upstream LIBERO cannot run in the py3.13 training venv). So every entry point
here goes through `make_env`, which returns the real `OffScreenRenderEnv` when
`libero` imports and a `FakeLiberoEnv` with the same surface when it does not.
The fake path is not a nicety — today it is the only path that runs, and it is
what exercises the runner, the results JSON and the markdown emitter.

Two details would otherwise be silent near-zero-score bugs:

* **15 dummy `[0,0,0,0,0,0,-1]` actions after `set_init_state`**, to let the
  scene settle before the policy acts (from the working in-tree reference,
  `cosmos_rl/simulators/libero/env_wrapper.py`). These are part of the
  *episode loop*, not of the env wrapper, and they do not count against
  `max_steps`.
* **Image orientation is delegated to Team A**, not hardcoded here. The
  training HDF5s are stored `macros.IMAGE_CONVENTION = 'opengl'` (bottom-up);
  the in-tree reference flips live frames `[::-1, ::-1]`, which is a 180
  rotation, not a vertical flip, and those are different transforms. Which one
  is right is being settled empirically by Team G against a live env using
  `orient_env_image()` / `best_matching_transform()`. This module calls
  `orient_env_image` so that whatever they measure propagates here without an
  edit, and `LIBERO_ENV_IMAGE_CONVENTION` below is the single knob.

Success follows LIBERO's own convention: a task counts as solved if the
environment raises its success flag at **any** point in the episode, not only
on the final step. `_episode_success` latches.
"""

from __future__ import annotations

import math
import os
import traceback
from typing import Any, Callable, Protocol

import numpy as np

from contracts import EMBODIMENTS
from loom.eval import DEFAULT_LIBERO_SUITES, EpisodeResult, EvalProtocol

# Team A owns the train/eval orientation convention. Importing their helper is
# the point: a second flip implementation here is how train and eval come to
# disagree about which way is up.
from loom.data.adapters.libero import orient_env_image

__all__ = [
    "SUITES", "SUITE_ALIASES", "N_TASKS", "MAX_STEPS_BY_SUITE",
    "SETTLE_STEPS", "DUMMY_ACTION", "LIBERO_ENV_IMAGE_CONVENTION",
    "LIBERO_PYTHON", "MEASURED_CONTROL_FREQ",
    "patch_torch_load_for_init_states", "ensure_libero_runtime",
    "libero_available", "make_env", "FakeLiberoEnv",
    "orient_env_image", "quat2axisangle", "extract_obs",
    "task_name", "task_instruction", "n_tasks",
    "run_episode", "DEFAULT_PROTOCOL",
]


# ═══════════════════════════════════════════════════════════════════════════
#  INSTALLATION CONSTANTS
#
#  Every path and version the real LIBERO run needs, in ONE block, all
#  overridable from the environment. Nothing here is guessed — every value is
#  measured by Team G's `scripts/smoke_libero.py`, which is green on an A100 for
#  all four suites. Nothing in this module reads a path outside this block.
# ═══════════════════════════════════════════════════════════════════════════

#: Dataset root: <root>/{libero_spatial,libero_object,libero_goal,libero_10}/
#: 10 tasks x 50 demos per suite. bddl files and .pruned_init states (10 + 10
#: per suite, 50 init states per task across all 40 tasks) ship in-repo, so
#: `trial_id % n_trials` below indexes against 50.
LOOM_DATA_ROOT = os.environ.get(
    "LOOM_DATA_ROOT",
    "/lustre/fsw/portfolios/edgeai/users/chrislin/datasets/loom/libero",
)

#: Separate interpreter for the real env. Eval does NOT run in the training venv.
#: python 3.10.20, mujoco 2.3.2 (HARD PIN), robosuite 1.4.1, numpy 1.26.4,
#: torch 2.6.0+cu124, gym 0.25.2.
LIBERO_PYTHON = os.environ.get(
    "LOOM_LIBERO_PYTHON",
    "/lustre/fsw/portfolios/edgeai/users/chrislin/envs/loom-libero/bin/python",
)
LIBERO_CONDA_ENV = os.path.dirname(os.path.dirname(LIBERO_PYTHON))

#: Headless rendering on a compute node. Applied by `apply_headless_env()`.
#: EGL only: `osmesa` is not available as a fallback, and the whole LIBERO stack
#: is unimportable on the login node — even a physics-only check must go through
#: `srun` on a GPU node.
HEADLESS_ENV = {
    "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl"),
    "PYOPENGL_PLATFORM": os.environ.get("PYOPENGL_PLATFORM", "egl"),
    "MUJOCO_EGL_DEVICE_ID": os.environ.get("MUJOCO_EGL_DEVICE_ID", "0"),
}

#: Measured: control_timestep 0.05 s / model_timestep 0.002 s -> exactly 20.0 Hz,
#: confirming the frozen `contracts.EMBODIMENTS["libero_franka"].env_fps`. The
#: whole fractional-accumulator path depends on this number.
MEASURED_CONTROL_FREQ = 20.0

#: Measured throughput, for shard sizing: 36-55 env steps/s per process, i.e.
#: ~10-15 s per 512-step episode. The default protocol is 1200 episodes ~= 4-5
#: GPU-hours single-process, which does NOT fit the 4 h walltime cap alone;
#: sharded across 8 GPUs it is ~30-40 min. See `runner.run_eval(workers=...)`.
EPISODE_SECONDS = (10.0, 15.0)

#: bddl files and init states resolve through `libero.libero.get_libero_path`,
#: which reads ~/.libero/config.yaml. Set these only to override that.
LIBERO_BDDL_DIR = os.environ.get("LOOM_LIBERO_BDDL_DIR") or None
LIBERO_INIT_STATES_DIR = os.environ.get("LOOM_LIBERO_INIT_STATES_DIR") or None

#: Camera resolution used by the reference harness.
IMAGE_SIZE = int(os.environ.get("LOOM_LIBERO_IMAGE_SIZE", "256"))

#: `robosuite.macros.IMAGE_CONVENTION` as this harness configures it, and the
#: only orientation knob in eval. "opengl" (bottom-up, what robosuite returns
#: by default) means `orient_env_image` applies Team A's flip; "opencv" means
#: robosuite already flipped and eval must not flip again. Team G is measuring
#: which is correct against a live env with `best_matching_transform`.
LIBERO_ENV_IMAGE_CONVENTION = os.environ.get("LOOM_LIBERO_IMAGE_CONVENTION", "opengl")


def apply_headless_env() -> None:
    """Set the offscreen-render variables. Call before importing `libero`."""
    for k, v in HEADLESS_ENV.items():
        os.environ.setdefault(k, v)


def patch_torch_load_for_init_states() -> str:
    """LIBERO reads `.pruned_init` files with a bare `torch.load(path)`.

    torch >= 2.6 flipped that call's default to `weights_only=True`, which
    refuses to unpickle the plain-python payload in those files. The symptom is
    an `UnpicklingError: Weights only load failed` the first time you ask for an
    init state — i.e. **every evaluation episode fails at reset and you never
    get a single score.**

    Copied verbatim from `scripts/smoke_libero.py` (Team G), which is the only
    place this has been verified against the real files. Returns a short status
    string, which `runner` records in the results JSON so that a run which died
    at reset says why.
    """
    import torch                                        # noqa: PLC0415

    major_minor = tuple(int(x) for x in torch.__version__.split(".")[:2])
    if major_minor < (2, 6):
        return f"torch {torch.__version__}: no shim needed"
    _orig_load = torch.load

    def _load(*a, **kw):
        kw.setdefault("weights_only", False)
        return _orig_load(*a, **kw)

    torch.load = _load
    return f"torch {torch.__version__}: torch.load patched to weights_only=False"


#: Status of the one-time runtime setup, reported into the results JSON.
LIBERO_RUNTIME_STATUS: str | None = None


def ensure_libero_runtime() -> str:
    """Idempotent, once per process, before anything touches the real env.

    Deliberately NOT done at import: `torch.load` is process-global state and
    `loom.eval` is imported by the training venv too. It runs only when a real
    LIBERO env or task suite is actually constructed.
    """
    global LIBERO_RUNTIME_STATUS
    if LIBERO_RUNTIME_STATUS is None:
        apply_headless_env()
        LIBERO_RUNTIME_STATUS = patch_torch_load_for_init_states()
    return LIBERO_RUNTIME_STATUS


# ═══════════════════════════════════════════════════════════════════════════
#  SUITES
# ═══════════════════════════════════════════════════════════════════════════

SUITES: tuple[str, ...] = DEFAULT_LIBERO_SUITES

#: PLAN 8 calls the long-horizon suite `long`; LIBERO's benchmark registry
#: calls it `libero_10`. Same 10 tasks.
SUITE_ALIASES = {
    "libero_long": "libero_10",
    "libero_10": "libero_10",
    "long": "libero_10",
    "libero_spatial": "libero_spatial",
    "libero_object": "libero_object",
    "libero_goal": "libero_goal",
}

#: Every standard suite has 10 tasks.
N_TASKS = {s: 10 for s in SUITES}

#: LIBERO_MAX_STEPS_MAP — 512 for every suite. Not an invented number.
MAX_STEPS_BY_SUITE = {s: 512 for s in SUITES}

#: Scene-settling actions, executed after `set_init_state` and before the
#: policy acts. Reproducing published success rates depends on this.
SETTLE_STEPS = 15
DUMMY_ACTION = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)

DEFAULT_PROTOCOL = EvalProtocol(
    bench="libero",
    episodes_per_task=10,
    n_tasks=10,
    suites=SUITES,
    seeds=(0, 1, 2),
    max_steps=512,
)


def benchmark_name(suite: str) -> str:
    """PLAN 8 suite name -> LIBERO benchmark registry name."""
    try:
        return SUITE_ALIASES[suite]
    except KeyError:
        raise KeyError(f"unknown LIBERO suite {suite!r}; have {sorted(set(SUITES))}") from None


def n_tasks(suite: str) -> int:
    return N_TASKS.get(suite, 10)


def task_name(suite: str, task_id: int) -> str:
    return f"{suite}/task_{int(task_id):02d}"


def task_instruction(suite: str, task_id: int) -> str:
    """The language goal. Real suite when available, deterministic text otherwise."""
    if libero_available():
        try:
            suite_obj = _task_suite(suite)
            return str(suite_obj.get_task(int(task_id)).language)
        except Exception:                                # noqa: BLE001
            pass
    return f"fake instruction for {task_name(suite, task_id)}"


# ═══════════════════════════════════════════════════════════════════════════
#  OBSERVATION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def _orient(img: np.ndarray) -> np.ndarray:
    """Live env frame -> the orientation the policy was trained on.

    Delegates to `loom.data.adapters.libero.orient_env_image`. Eval owns *which
    convention the simulator is configured with*; Team A owns *what transform
    that implies*. Feeding the live orientation to a policy trained on the
    recorded one is a silent near-zero-score bug with no other symptom, and the
    surest way to get it wrong is for two teams to each write a flip.
    """
    a = np.asarray(img)
    if a.ndim < 3:
        raise ValueError(f"expected (..., H, W, C), got shape {a.shape}")
    return orient_env_image(a, LIBERO_ENV_IMAGE_CONVENTION)


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """(x, y, z, w) -> axis-angle. Local copy of robosuite's transform_utils."""
    q = np.asarray(quat, dtype=np.float64).copy()
    q[3] = min(max(float(q[3]), -1.0), 1.0)
    den = math.sqrt(max(1.0 - q[3] * q[3], 0.0))
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


def extract_obs(raw: dict) -> dict:
    """Raw robosuite obs -> the dict the policy's featuriser consumes.

    State vector is `robot0_eef_pos` + `quat2axisangle(robot0_eef_quat)` +
    `robot0_gripper_qpos`, matching the reference harness exactly.
    """
    out: dict[str, Any] = {}
    if "agentview_image" in raw:
        out["full_image"] = _orient(raw["agentview_image"])
    if "robot0_eye_in_hand_image" in raw:
        out["wrist_image"] = _orient(raw["robot0_eye_in_hand_image"])
    if "robot0_eef_pos" in raw:
        out["state"] = np.concatenate([
            np.asarray(raw["robot0_eef_pos"], dtype=np.float32).reshape(-1),
            quat2axisangle(raw["robot0_eef_quat"]).astype(np.float32),
            np.asarray(raw["robot0_gripper_qpos"], dtype=np.float32).reshape(-1),
        ])
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  ENV SEAM
# ═══════════════════════════════════════════════════════════════════════════

class EnvLike(Protocol):
    """The whole surface the harness needs. Real and fake both satisfy it."""

    def reset(self) -> dict: ...
    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]: ...
    def close(self) -> None: ...


def libero_available() -> bool:
    """True when the real package imports. Cached; never raises."""
    global _LIBERO_OK
    if _LIBERO_OK is None:
        try:
            import importlib

            importlib.import_module("libero.libero")
            _LIBERO_OK = True
        except Exception:                                # noqa: BLE001
            _LIBERO_OK = False
    return _LIBERO_OK


_LIBERO_OK: bool | None = None
_SUITE_CACHE: dict[str, Any] = {}


def _task_suite(suite: str) -> Any:
    name = benchmark_name(suite)
    if name not in _SUITE_CACHE:
        # `get_task_init_states` calls a bare torch.load; without the shim every
        # episode dies at reset.
        ensure_libero_runtime()
        import libero.libero.benchmark as benchmark      # noqa: PLC0415  (lazy by design)

        _SUITE_CACHE[name] = benchmark.get_benchmark(name)()
    return _SUITE_CACHE[name]


def make_env(
    suite: str,
    task_id: int,
    seed: int,
    *,
    trial_id: int = 0,
    image_size: int = IMAGE_SIZE,
    backend: str | None = None,
    **kw: Any,
) -> EnvLike:
    """The seam. `backend` in {None (auto), "libero", "fake"}.

    Auto picks the real env when `libero` imports and the fake otherwise, so
    tests run today and the real run is one import away.
    """
    backend = backend or ("libero" if libero_available() else "fake")
    if backend == "fake":
        return FakeLiberoEnv(suite, task_id, seed, trial_id=trial_id,
                             image_size=image_size, **kw)
    if backend == "libero":
        return _make_real_env(suite, task_id, seed, trial_id=trial_id,
                              image_size=image_size, **kw)
    raise ValueError(f"unknown env backend {backend!r}")


class LiberoEnv:
    """Thin adapter over `libero.libero.envs.OffScreenRenderEnv`.

    Constructed exactly as the reference harness does. Not exercised by the
    test suite — there is no LIBERO on this cluster yet — so it is deliberately
    a transcription with no cleverness in it.

    Note for whoever wires the real run: the reference constructs this inside a
    **spawn-context subprocess**, because robosuite/mujoco does not survive
    fork and because eval runs under a different interpreter than training.
    `runner.py` already shards work across processes; point its `env_factory`
    at this class from inside the worker.
    """

    def __init__(self, suite: str, task_id: int, seed: int, *,
                 trial_id: int = 0, image_size: int = IMAGE_SIZE,
                 **_ignored: Any) -> None:
        # headless render vars + the torch.load shim, before anything imports
        # libero or asks for an init state
        ensure_libero_runtime()
        from libero.libero import get_libero_path        # noqa: PLC0415  (lazy by design)
        from libero.libero.envs import OffScreenRenderEnv  # noqa: PLC0415

        self.suite, self.task_id, self.trial_id = suite, int(task_id), int(trial_id)
        suite_obj = _task_suite(suite)
        task = suite_obj.get_task(self.task_id)
        bddl_dir = LIBERO_BDDL_DIR or get_libero_path("bddl_files")
        self.bddl_file = os.path.join(bddl_dir, task.problem_folder, task.bddl_file)
        self.language = str(task.language)

        self.env = OffScreenRenderEnv(
            bddl_file_name=self.bddl_file,
            camera_heights=image_size,
            camera_widths=image_size,
        )
        self.env.seed(int(seed))
        init_states = suite_obj.get_task_init_states(self.task_id)
        self._init_state = init_states[self.trial_id % len(init_states)]

    def reset(self) -> dict:
        self.env.reset()
        return self.env.set_init_state(self._init_state)

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        return self.env.step(np.asarray(action, dtype=np.float64))

    def check_success(self) -> bool:
        """The task's own goal predicate, straight from the BDDL problem.

        `bddl_base_domain.step` already overwrites `done` with
        `self._check_success()`, so `done` and this agree — but only as long as
        nothing ever wraps the env in something that reports horizon exhaustion
        as `done`. Exposing the predicate explicitly means `_episode_success`
        reads the goal state rather than a termination flag, which is the
        difference between "the policy failed" and "the detector never fires".
        Verified live by replaying a ground-truth demo: see
        `logs/eval_libero_probe.py --check success_replay`.
        """
        return bool(self.env.check_success())

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:                                # noqa: BLE001
            pass


def _make_real_env(suite: str, task_id: int, seed: int, **kw: Any) -> EnvLike:
    return LiberoEnv(suite, task_id, seed, **kw)


# ═══════════════════════════════════════════════════════════════════════════
#  FAKE ENV
# ═══════════════════════════════════════════════════════════════════════════

class FakeLiberoEnv:
    """Same `reset/step/done/success` surface, random outcomes, no MuJoCo.

    Deterministic in `seed`: the same env seed always produces the same
    outcome and the same episode length, which is what makes the runner's
    determinism guarantee testable without LIBERO installed.

    It also *validates* what the policy sends it — shape, dtype, finiteness and
    action bounds — so the rate-conversion path is exercised for real rather
    than merely executed.
    """

    def __init__(
        self,
        suite: str = "libero_spatial",
        task_id: int = 0,
        seed: int = 0,
        *,
        trial_id: int = 0,
        image_size: int = 64,
        p_success: float = 0.5,
        embodiment: str = "libero_franka",
        crash_at: int | None = None,
        max_steps: int = 512,
        strict_bounds: bool = False,
    ) -> None:
        self.strict_bounds = strict_bounds
        self.suite, self.task_id, self.seed = suite, int(task_id), int(seed)
        self.trial_id = int(trial_id)          # real env: index into the init states
        self.spec = EMBODIMENTS[embodiment]
        self.image_size = image_size
        self.crash_at = crash_at
        self.max_steps = max_steps
        self.language = task_instruction(suite, task_id)

        rng = np.random.default_rng(self.seed)
        self._will_succeed = bool(rng.random() < p_success)
        # `n_steps` counts the settle phase too, so a solve can only land after it
        lo = SETTLE_STEPS + 5
        self._solve_step = int(rng.integers(lo, max(lo + 1, max_steps)))
        self._rng = rng
        self.n_steps = 0
        self.closed = False

    # ── surface ───────────────────────────────────────────────────────────

    def reset(self) -> dict:
        self.n_steps = 0
        return self._obs()

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
            raise RuntimeError(
                f"FakeLiberoEnv: injected crash at step {self.n_steps}"
            )
        success = self._will_succeed and self.n_steps >= self._solve_step
        done = bool(success)
        return self._obs(), float(success), done, {"success": success}

    def close(self) -> None:
        self.closed = True

    def check_success(self) -> bool:
        return self._will_succeed and self.n_steps >= self._solve_step

    # ── obs ───────────────────────────────────────────────────────────────

    def _obs(self) -> dict:
        s = self.image_size
        img = self._rng.integers(0, 256, size=(s, s, 3), dtype=np.uint8)
        return {
            "agentview_image": img,
            "robot0_eye_in_hand_image": img.copy(),
            "robot0_eef_pos": self._rng.normal(size=3).astype(np.float32),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "robot0_gripper_qpos": self._rng.normal(size=2).astype(np.float32),
        }


# ═══════════════════════════════════════════════════════════════════════════
#  EPISODE LOOP
# ═══════════════════════════════════════════════════════════════════════════

def _episode_success(done: bool, info: Any, env: Any) -> bool:
    """LIBERO's success signal, however this env chooses to expose it.

    The reference harness treats `done` from `env.step` as the success flag —
    OffScreenRenderEnv raises it only on task completion and leaves the step
    cap to the caller. `info["success"]` and `env.check_success()` are checked
    first because wrappers vary.
    """
    if isinstance(info, dict):
        for k in ("success", "is_success", "task_success"):
            if k in info:
                return bool(np.asarray(info[k]).all())
    for name in ("check_success", "is_success", "_check_success"):
        fn = getattr(env, name, None)
        if callable(fn):
            try:
                return bool(fn())
            except Exception:                            # noqa: BLE001
                pass
    return bool(done)


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

    Success latches: the task is solved if the env raises its flag at any point
    in the episode, not only on the final step. Settle steps do not count
    against `max_steps`.
    """
    raw = env.reset()

    # 15 dummy actions so the scene settles before the policy sees anything.
    for _ in range(settle_steps):
        raw, _, _, _ = env.step(DUMMY_ACTION.copy())

    policy.reset()
    success, steps = False, 0
    for steps in range(1, max_steps + 1):
        action = policy.act(obs_fn(raw), instruction)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        raw, _, done, info = env.step(action)
        if _episode_success(done, info, env):
            success = True
            break

    out = {
        "success": success,
        "steps": steps,
        "hit_step_cap": bool(not success and steps >= max_steps),
        "n_replans": int(getattr(policy, "replans", 0)) or None,
    }
    # Opt-in policy diagnostics (`LoomPolicy(op_stats=True)`): which operator
    # `pi_c` actually selected at each replan. Empty dict on every other policy
    # and on a default run, so nothing changes in the normal results JSON.
    summary = getattr(policy, "op_stats_summary", None)
    if callable(summary):
        extra = summary()
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

    The traceback goes into the record and therefore into the results JSON.
    """
    env = None
    try:
        env = env_factory()
        out = run_episode(policy, env, instruction, max_steps, **kw)
        for k, v in out.items():
            if k == "extra" and isinstance(v, dict):
                record.extra.update(v)      # merge; `_run_item` also writes here
            else:
                setattr(record, k, v)
    except Exception:                                    # noqa: BLE001 — one episode, not the run
        record.success = False
        record.error = traceback.format_exc()
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:                            # noqa: BLE001
                pass
    return record
