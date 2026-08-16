#!/usr/bin/env python3
"""
LOOM — RoboTwin 2.0 smoke test (Team H).

MUST RUN ON A GPU COMPUTE NODE. RoboTwin 2.0 renders through SAPIEN 3 /
Vulkan; the login node has no GPU and no Vulkan ICD, so every check below
would either fail or -- worse -- silently return black frames.

    source $ENV_PREFIX/robotwin_env.sh
    python scripts/smoke_robotwin.py

What it proves, per PLAN.md §7 (R0-B is the kill/proceed gate):

  [1] SAPIEN initialises Vulkan against the NVIDIA ICD and reports a GPU.
  [2] Each of PLAN §8's four named tasks builds, resets and steps.
  [3] Every camera stream returns the documented resolution AND is not
      all-black. An all-black render is the classic silent headless-GPU
      failure: everything "works", every number is real, and the score is
      zero. This is the single most important assertion in the file.
  [4] The action vector's dimension and bounds match what the adapter must
      register in contracts.py.
  [5] The success / termination flag actually flips over a full episode.
  [6] The environment's control frequency, MEASURED rather than assumed --
      PLAN §9 requires decoded actions be resampled from canonical 30 Hz
      back to the env rate, and an 11% error there scores near zero.

Exit code 0 iff every check passed.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

# PLAN §8, RoboTwin 2.0 table: per-task columns.
NAMED_TASKS = ["hanging_mug", "turn_switch", "place_can_basket", "handover_block"]

# A camera stream whose per-channel variance is below this is treated as a
# failed render. A real RoboTwin frame scores ~2000; a black frame scores 0.
MIN_PIXEL_VARIANCE = 5.0

DEFAULT_ROBOTWIN = "/lustre/fsw/portfolios/edgeai/users/chrislin/projects/loom-deps/RoboTwin"
DEFAULT_DATA = "/lustre/fsw/portfolios/edgeai/users/chrislin/datasets/loom/robotwin"

FAILURES: list[str] = []
CHECKS = 0


def check(cond: bool, label: str, detail: str = "") -> bool:
    global CHECKS
    CHECKS += 1
    mark = "PASS" if cond else "FAIL"
    colour = "\033[32m" if cond else "\033[31m"
    print(f"  [{colour}{mark}\033[0m] {label}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILURES.append(f"{label} {detail}".strip())
    return cond


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ---------------------------------------------------------------------------
# [1] host / renderer environment
# ---------------------------------------------------------------------------
def report_environment() -> None:
    section("[1] host and renderer")
    print(f"  hostname          {platform.node()}")
    print(f"  python            {sys.version.split()[0]}")
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=60,
        )
        gpu = smi.stdout.strip().splitlines()
        check(bool(gpu), "nvidia-smi reports a GPU", gpu[0] if gpu else "(none)")
    except Exception as exc:  # pragma: no cover - login node
        check(False, "nvidia-smi reports a GPU", f"{type(exc).__name__}: {exc}")

    icd = os.environ.get("VK_ICD_FILENAMES") or os.environ.get("VK_DRIVER_FILES")
    check(bool(icd), "Vulkan ICD pinned via VK_ICD_FILENAMES / VK_DRIVER_FILES", str(icd))
    if icd and os.path.isfile(icd):
        print(f"  ICD contents      {json.load(open(icd))['ICD']}")

    import torch
    print(f"  torch             {torch.__version__}")
    check(torch.cuda.is_available(),
          "torch.cuda.is_available()",
          f"devices={torch.cuda.device_count()}")

    import sapien
    print(f"  sapien            {sapien.__version__}")

    # SAPIEN enumerates physical devices through its own Vulkan loader. If the
    # loader falls through to Mesa's lvp (llvmpipe) SOFTWARE ICD everything
    # still "works" -- on the CPU, at ~1 fps, producing plausible non-black
    # frames that are not the GPU path this benchmark assumes. So assert that
    # a hardware device is what SAPIEN actually picked.
    import sapien.render as sr
    summary = ""
    try:
        summary = str(sr.get_device_summary())
    except Exception as exc:
        summary = f"<unavailable: {type(exc).__name__}: {exc}>"
    print("  sapien device summary:")
    for line in summary.splitlines():
        print(f"    {line}")
    low = summary.lower()
    check(not summary.startswith("<unavailable"),
          "SAPIEN can enumerate Vulkan devices")
    # get_device_summary() prints a table whose header mentions CUDA, so a
    # substring match on 'cuda' proves nothing. llvmpipe/lavapipe appearing as
    # the selected device does prove something, and so does the render
    # throughput measured per task below.
    check("llvmpipe" not in low and "lavapipe" not in low,
          "SAPIEN did NOT fall through to the llvmpipe software rasteriser")


# ---------------------------------------------------------------------------
# task construction -- mirrors RoboTwin's own scripts/eval_policy_xpolicylab.py
# ---------------------------------------------------------------------------
def load_task_args(robotwin_root: Path, task_name: str, task_config: str) -> dict:
    import yaml
    cfg_root = robotwin_root / "env_cfg" / "task_config"
    with open(cfg_root / f"{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.safe_load(f)

    args["task_name"] = task_name
    args["task_config"] = task_config

    with open(cfg_root / "_embodiment_config.yml", "r", encoding="utf-8") as f:
        embodiments = yaml.safe_load(f)
    with open(cfg_root / "_camera_config.yml", "r", encoding="utf-8") as f:
        cameras = yaml.safe_load(f)

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

    def emb_cfg(path):
        with open(os.path.join(path, "config.yml"), "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    args["left_embodiment_config"] = emb_cfg(robot_file)
    args["right_embodiment_config"] = emb_cfg(robot_file)

    # eval_mode makes _init_task_env_ load step_lim from _eval_step_limit.yml.
    args["eval_mode"] = True
    args["save_data"] = False
    args["render_freq"] = 0
    args["data_type"]["rgb"] = True
    args["data_type"]["qpos"] = True
    args["data_type"]["endpose"] = True
    return args


def build_task(task_name: str, args: dict, seed: int, ep: int = 0):
    module = importlib.import_module(f"envs.{task_name}")
    task = getattr(module, task_name)()
    task.setup_demo(now_ep_num=ep, seed=seed, is_test=True, **args)
    return task


class _SceneProxy:
    """Transparent stand-in for sapien.Scene that counts step() calls.

    sapien's Scene is a pybind11 type without dynamic attributes, so
    `scene.step = wrapper` raises AttributeError. RoboTwin always reaches the
    scene through `self.scene`, so swapping that one reference is enough --
    the cameras and articulations keep their handle on the real object.
    """

    def __init__(self, scene, counter):
        object.__setattr__(self, "_scene", scene)
        object.__setattr__(self, "_counter", counter)

    def step(self, *a, **k):
        self._counter.steps += 1
        return self._scene.step(*a, **k)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_scene"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_scene"), name, value)


class StepCounter:
    """Counts physx steps and recorded frames without touching RoboTwin code."""

    def __init__(self, task):
        self.task = task
        self.steps = 0
        self.frames = 0
        #: physx step index at which each demonstration frame was captured.
        self.frame_at_step: list[int] = []
        self._real_scene = task.scene
        self._take_picture = task._take_picture
        self._patched_scene_attr = False

        # Capture the ORIGINAL bound method before overwriting the attribute.
        # `self._real_scene.step` would re-resolve to the wrapper we are about
        # to install and recurse until RecursionError.
        orig_step = task.scene.step

        def counted_step(*a, **k):
            self.steps += 1
            return orig_step(*a, **k)

        try:
            task.scene.step = counted_step          # works if dynamic attrs allowed
            self._patched_scene_attr = True
        except (AttributeError, TypeError):
            task.scene = _SceneProxy(self._real_scene, self)

        def counted_picture(*a, **k):
            self.frames += 1
            self.frame_at_step.append(self.steps)
            return self._take_picture(*a, **k)

        task._take_picture = counted_picture

    def frame_gap(self) -> float:
        """Median physx-step gap between consecutive recorded frames.

        This -- not steps/frames -- is `save_freq`. RoboTwin also snaps one
        frame immediately before and one immediately after each planned motion
        segment, so a plain ratio is biased low by ~2 frames per `move()` call.
        Taking the median over positive gaps discards those zero/short gaps.
        """
        import numpy as np
        gaps = np.diff(np.asarray(self.frame_at_step, dtype=np.int64))
        gaps = gaps[gaps > 0]
        return float(np.median(gaps)) if gaps.size else float("nan")

    def restore(self):
        if self._patched_scene_attr:
            with contextlib.suppress(Exception):
                del self.task.scene.step
        else:
            self.task.scene = self._real_scene
        self.task._take_picture = self._take_picture


# ---------------------------------------------------------------------------
# [2][3][4][6] per-task checks
# ---------------------------------------------------------------------------
def check_task(robotwin_root: Path, task_name: str, task_config: str,
               seed: int, n_steps: int) -> dict | None:
    import numpy as np

    section(f"[2-4,6] task: {task_name}   (config={task_config}, seed={seed})")
    t0 = time.time()
    try:
        args = load_task_args(robotwin_root, task_name, task_config)
        task = build_task(task_name, args, seed=seed)
    except Exception as exc:
        check(False, f"{task_name}: build + reset",
              f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return None
    check(True, f"{task_name}: build + reset", f"{time.time() - t0:.1f}s")

    result: dict = {"task": task_name}

    # -- observation & rendering -------------------------------------------
    obs = task.get_obs()
    cams = sorted(obs["observation"].keys())
    result["cameras"] = cams
    check(len(cams) > 0, f"{task_name}: observation has camera streams", str(cams))

    expected_hw = (args["head_camera_h"], args["head_camera_w"])
    all_bright = True
    for cam in cams:
        rgb = obs["observation"][cam].get("rgb")
        if rgb is None:
            check(False, f"{task_name}: {cam} has an rgb array")
            all_bright = False
            continue
        rgb = np.asarray(rgb)
        var = float(rgb.var())
        shape_ok = rgb.shape[:2] == expected_hw and rgb.shape[2] == 3
        check(shape_ok, f"{task_name}: {cam} resolution",
              f"{rgb.shape} expected {(*expected_hw, 3)}")
        # THE check: a headless GPU without a working ICD renders pure black.
        bright = var >= MIN_PIXEL_VARIANCE
        check(bright, f"{task_name}: {cam} render is NOT all-black",
              f"var={var:9.2f} mean={float(rgb.mean()):6.2f} "
              f"min={int(rgb.min())} max={int(rgb.max())}")
        all_bright &= bright
    result["render_ok"] = all_bright
    result["n_views"] = len(cams)

    # -- action vector ------------------------------------------------------
    qpos = np.asarray(obs["joint_action"]["vector"], dtype=np.float64)
    left_js = task.robot.get_left_arm_jointState()
    right_js = task.robot.get_right_arm_jointState()
    left_arm_dim = len(left_js) - 1
    right_arm_dim = len(right_js) - 1
    dof = left_arm_dim + 1 + right_arm_dim + 1
    result["dof"] = dof
    result["left_arm_dim"] = left_arm_dim
    result["right_arm_dim"] = right_arm_dim
    check(qpos.shape == (dof,),
          f"{task_name}: action/proprio width",
          f"dof={dof} = ({left_arm_dim}+1)+({right_arm_dim}+1), vector{qpos.shape}")

    # Bounds. RoboTwin defines no gym action_space: the arm joints are bounded
    # by the URDF revolute limits and the grippers are hard-clipped to [0,1]
    # inside Robot.set_gripper().
    lo, hi = joint_limits(task, left_arm_dim, right_arm_dim)
    result["action_low"] = lo
    result["action_high"] = hi
    print(f"    action_low  = {np.round(lo, 4).tolist()}")
    print(f"    action_high = {np.round(hi, 4).tolist()}")
    check(lo[left_arm_dim] == 0.0 and hi[left_arm_dim] == 1.0
          and lo[-1] == 0.0 and hi[-1] == 1.0,
          f"{task_name}: gripper dims are normalised to [0, 1]",
          f"idx {left_arm_dim} and {dof - 1}")

    # -- step -------------------------------------------------------------
    counter = StepCounter(task)
    rng = np.random.default_rng(seed)
    wall = []
    phys = []
    base = qpos.copy()
    for i in range(n_steps):
        # Small perturbations around the home pose: large random joint targets
        # make TOPP fail and measure nothing useful.
        act = base.copy()
        act[:left_arm_dim] += rng.normal(0, 0.02, left_arm_dim)
        act[left_arm_dim + 1:left_arm_dim + 1 + right_arm_dim] += \
            rng.normal(0, 0.02, right_arm_dim)
        act[left_arm_dim] = float(np.clip(rng.uniform(0, 1), 0, 1))
        act[-1] = float(np.clip(rng.uniform(0, 1), 0, 1))
        s0, t0 = counter.steps, time.time()
        task.take_action(act, action_type="qpos")
        wall.append(time.time() - t0)
        phys.append(counter.steps - s0)
        base = np.asarray(task.get_obs()["joint_action"]["vector"], dtype=np.float64)

    check(len(wall) == n_steps, f"{task_name}: stepped {n_steps} actions",
          f"take_action_cnt={task.take_action_cnt}")
    check(task.take_action_cnt == n_steps,
          f"{task_name}: take_action_cnt tracks steps",
          f"{task.take_action_cnt} == {n_steps}")
    check(isinstance(task.eval_success, bool),
          f"{task_name}: eval_success flag present", f"={task.eval_success}")
    check(task.step_lim is not None,
          f"{task_name}: eval step limit loaded", f"step_lim={task.step_lim}")

    dt = physics_timestep(task)
    result["physics_dt"] = dt
    result["physx_steps_per_action_mean"] = float(np.mean(phys))
    result["wall_per_action_mean"] = float(np.mean(wall))
    print(f"    physics timestep        {dt:.6f} s  ({1 / dt:.1f} Hz)")
    print(f"    physx steps / action    mean {np.mean(phys):6.1f}  "
          f"min {min(phys)}  max {max(phys)}")
    print(f"    wall time / action      {np.mean(wall) * 1e3:7.1f} ms  "
          f"({1 / max(np.mean(wall), 1e-9):.2f} action/s incl. rendering)")

    # Rendering still lively after stepping?
    obs2 = task.get_obs()
    v2 = float(np.asarray(obs2["observation"][cams[0]]["rgb"]).var())
    check(v2 >= MIN_PIXEL_VARIANCE,
          f"{task_name}: render still non-black after {n_steps} steps",
          f"var={v2:.2f}")

    counter.restore()
    with contextlib.suppress(Exception):
        task.close_env()
    return result


def physics_timestep(task) -> float:
    for getter in (lambda: task.scene.timestep,
                   lambda: task.scene.get_timestep(),
                   lambda: task.scene.physx_system.timestep):
        try:
            v = getter()
            if v:
                return float(v)
        except Exception:
            continue
    return 1.0 / 250.0  # RoboTwin's _base_task.setup_scene default


def joint_limits(task, left_arm_dim: int, right_arm_dim: int):
    """Per-dimension bounds of the 14-vector RoboTwin's take_action consumes."""
    import numpy as np

    def arm_limits(joints):
        out = []
        for j in joints:
            try:
                lim = np.asarray(j.get_limits()).reshape(-1, 2)[0]
            except Exception:
                lim = np.array([-np.inf, np.inf])
            out.append(lim)
        return np.asarray(out)

    left = arm_limits(task.robot.left_arm_joints)[:left_arm_dim]
    right = arm_limits(task.robot.right_arm_joints)[:right_arm_dim]
    lo = np.concatenate([left[:, 0], [0.0], right[:, 0], [0.0]])
    hi = np.concatenate([left[:, 1], [1.0], right[:, 1], [1.0]])
    return lo.tolist(), hi.tolist()


# ---------------------------------------------------------------------------
# [6] control frequency, measured from the expert data-collection path
# ---------------------------------------------------------------------------
def measure_control_rate(robotwin_root: Path, data_root: Path, task_name: str,
                         task_config: str, seed: int) -> dict | None:
    """
    RoboTwin records a demonstration frame every `save_freq` physics steps, so
    the rate at which the released trajectories are authored is

        env_fps = 1 / (physics_dt * save_freq)

    Measure it instead of trusting the config: run the scripted expert once and
    count physx steps against recorded frames.

    NB the `additional_info/frequency` field written into every released HDF5 is
    literally `save_freq` (see scripts/process_data_xpolicylab.py:409), i.e. a
    physics-step decimation factor, NOT a value in Hz.
    """
    import numpy as np

    section(f"[6] measured control frequency (expert path, {task_name})")
    args = load_task_args(robotwin_root, task_name, task_config)
    save_freq = args.get("save_freq")
    print(f"  task config save_freq   {save_freq}")

    # Not every seed yields a solvable scene -- RoboTwin's own collector
    # searches for seeds whose expert succeeds and records them in seed.txt.
    # Prefer those; they are the seeds the released demos were collected with.
    seeds = [seed]
    seed_file = (data_root / "data" / task_config / task_name /
                 "aloha_agilex" / "seed.txt")
    if seed_file.is_file():
        known = [int(s) for s in seed_file.read_text().split()][:4]
        seeds = known or seeds
        print(f"  seeds from seed.txt     {seeds}")

    task = counter = None
    for s in seeds:
        try:
            task = build_task(task_name, args, seed=s)
        except Exception as exc:
            print(f"  seed {s}: build failed ({type(exc).__name__}: {exc})")
            continue
        counter = StepCounter(task)
        try:
            task.play_once()
        except Exception as exc:
            print(f"  seed {s}: expert raised {type(exc).__name__}: {exc}")
        if task.plan_success and task.check_success():
            print(f"  seed {s}: expert succeeded")
            break
        print(f"  seed {s}: expert did not reach success, trying next seed")
        counter.restore()
        with contextlib.suppress(Exception):
            task.close_env()
        task = counter = None

    if task is None or counter is None:
        check(False, "control-rate probe: expert episode",
              f"no seed in {seeds} produced a runnable expert episode")
        return None

    dt = physics_timestep(task)
    steps, frames = counter.steps, counter.frames
    gap = counter.frame_gap()                 # median physx steps between frames
    ratio = steps / max(frames - 1, 1)        # naive, biased low; shown for contrast
    env_fps_measured = 1.0 / (dt * gap)
    env_fps_config = 1.0 / (dt * save_freq)

    print(f"  physics timestep        {dt:.6f} s  ({1 / dt:.1f} Hz)")
    print(f"  physx steps (expert)    {steps}")
    print(f"  recorded frames         {frames}")
    print(f"  median steps / frame    {gap:.1f}      <- this is save_freq")
    print(f"  naive steps/frame       {ratio:.2f}     (biased low: 2 extra frames per move())")
    print(f"  env_fps  MEASURED       {env_fps_measured:.4f} Hz  (= 1/({dt:.6f} * {gap:.0f}))")
    print(f"  env_fps  from save_freq {env_fps_config:.4f} Hz  (= 1/({dt:.6f} * {save_freq}))")
    print(f"  HDF5 'frequency' field  {save_freq}  <- decimation factor, NOT Hz")

    check(abs(gap - save_freq) < 0.5,
          "measured median steps/frame equals task-config save_freq",
          f"{gap:.1f} vs {save_freq}")
    check(abs(env_fps_measured - 250.0 / 15.0) < 0.01,
          "measured env_fps == 250/15 == 16.6667 Hz",
          f"{env_fps_measured:.4f} Hz")

    # -- [5] success flag plumbing ------------------------------------------
    section(f"[5] success / termination plumbing ({task_name})")
    succeeded = bool(task.check_success())
    check(task.plan_success, "expert motion plan succeeded", f"={task.plan_success}")
    check(succeeded, "check_success() returns True after a full expert episode",
          f"={succeeded}")
    print(f"  step_lim                {task.step_lim}")
    print(f"  eval_success (pre)      {task.eval_success}")

    # take_action refuses to advance once eval_success latches -- that is the
    # termination path the eval runner relies on.
    task.eval_success = True
    before = task.take_action_cnt
    obs = task.get_obs()
    task.take_action(np.asarray(obs["joint_action"]["vector"]), action_type="qpos")
    check(task.take_action_cnt == before,
          "take_action() is a no-op once eval_success latches",
          f"cnt {before} -> {task.take_action_cnt}")

    counter.restore()
    with contextlib.suppress(Exception):
        task.close_env()

    return {
        "physics_dt": dt,
        "save_freq": save_freq,
        "median_steps_per_frame": gap,
        "steps_per_frame": ratio,
        "env_fps_measured": env_fps_measured,
        "env_fps_config": env_fps_config,
        "expert_success": succeeded,
    }


# ---------------------------------------------------------------------------
# [7] end-to-end replay of a recorded demonstration through the eval path
# ---------------------------------------------------------------------------
def replay_demo(robotwin_root: Path, data_root: Path, task_name: str,
                task_config: str, episode: int = 0) -> dict | None:
    """
    Drive the *policy* path with the recorded actions of one demonstration.

    This is the strongest available check that the embodiment contract is
    right: it reconstructs the exact scene from seed.txt, feeds
    `/action/joint_states` -- 14 absolute joint targets per frame -- straight
    into `take_action`, and asks whether `eval_success` latches. If dof, the
    channel order, or the absolute-vs-delta reading were wrong, nothing would
    succeed.

    INFORMATIONAL, not a hard gate: replay is open-loop and RoboTwin re-plans
    each step with TOPP rather than replaying the original dense trajectory,
    so a miss is not by itself evidence of a broken environment.
    """
    import h5py
    import numpy as np
    from PIL import Image  # noqa: F401  (h5py stores JPEG bytes)

    section(f"[7] recorded-demo replay through take_action ({task_name}, ep {episode})")
    ep_dir = data_root / "data" / task_config / task_name / "aloha_agilex"
    h5 = ep_dir / "data" / f"episode_{episode:07d}.hdf5"
    seed_file = ep_dir / "seed.txt"
    if not h5.is_file() or not seed_file.is_file():
        print(f"  skipped: no demonstration at {h5}")
        return None

    seeds = [int(s) for s in seed_file.read_text().split()]
    seed = seeds[episode]
    with h5py.File(h5, "r") as f:
        actions = np.asarray(f["/action/joint_states"][:], dtype=np.float64)
        states = np.asarray(f["/state/joint_states"][:], dtype=np.float64)
        instruction = f["/instruction"][()]
        declared_freq = int(f["/additional_info/frequency"][()])
    print(f"  seed                    {seed}   (seed.txt[{episode}])")
    print(f"  instruction             {instruction!r}")
    print(f"  actions                 {actions.shape}")
    print(f"  HDF5 additional_info/frequency = {declared_freq}   "
          f"(this is save_freq, NOT Hz -- see docs/ENV_ROBOTWIN.md §4)")

    # The absolute-target claim, re-verified on this episode.
    d_same = float(np.abs(actions[:-1] - states[1:]).max())
    print(f"  max|action[t] - state[t+1]| = {d_same:.3e}   "
          f"(0 => actions are absolute joint targets, not deltas)")
    check(d_same < 1e-5,
          "actions are ABSOLUTE joint targets (state[t+1] == action[t])",
          f"max abs diff {d_same:.3e}")

    args = load_task_args(robotwin_root, task_name, task_config)
    try:
        task = build_task(task_name, args, seed=seed, ep=episode)
    except Exception as exc:
        check(False, "replay: env build", f"{type(exc).__name__}: {exc}")
        return None

    counter = StepCounter(task)
    t0 = time.time()
    n = 0
    for a in actions:
        task.take_action(a, action_type="qpos")
        n += 1
        if task.eval_success or task.take_action_cnt >= (task.step_lim or 10 ** 9):
            break
    wall = time.time() - t0

    print(f"  replayed                {n}/{len(actions)} actions in {wall:.1f}s "
          f"({n / max(wall, 1e-9):.2f} action/s)")
    print(f"  physx steps             {counter.steps}  "
          f"({counter.steps / max(n, 1):.1f} per action)")
    print(f"  eval_success            {task.eval_success}")
    print(f"  check_success()         {task.check_success()}")
    ok = bool(task.eval_success or task.check_success())
    print(f"  \033[32mREPLAY REPRODUCED THE DEMONSTRATION\033[0m" if ok else
          "  \033[33mreplay did not reach success (informational only -- open-loop "
          "replay re-plans with TOPP and can drift)\033[0m")

    counter.restore()
    with contextlib.suppress(Exception):
        task.close_env()
    return {
        "episode": episode, "seed": seed, "n_actions": int(n),
        "eval_success": bool(task.eval_success), "replay_ok": ok,
        "physx_steps_per_action": counter.steps / max(n, 1),
    }


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robotwin-root",
                    default=os.environ.get("ROBOTWIN_ROOT", DEFAULT_ROBOTWIN))
    ap.add_argument("--task-config", default="demo_clean",
                    help="demo_clean (R0-B is scored on clean) or demo_randomized")
    ap.add_argument("--tasks", nargs="*", default=NAMED_TASKS)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rate-task", default="turn_switch",
                    help="task used for the expert-path control-rate measurement")
    ap.add_argument("--skip-rate", action="store_true")
    ap.add_argument("--skip-replay", action="store_true")
    ap.add_argument("--data-root", default=os.environ.get("ROBOTWIN_DATA", DEFAULT_DATA))
    ap.add_argument("--out", default=None, help="write a JSON summary here")
    a = ap.parse_args()

    root = Path(a.robotwin_root).resolve()
    # RoboTwin resolves every asset through relative paths off its own root.
    os.chdir(root)
    sys.path.insert(0, str(root))

    print(f"RoboTwin root      {root}")
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=30)
        print(f"RoboTwin commit    {head.stdout.strip()}")
    except Exception:
        pass
    print(f"task config        {a.task_config}")

    report_environment()

    summary: dict = {"task_config": a.task_config, "tasks": {}}
    for task_name in a.tasks:
        res = check_task(root, task_name, a.task_config, a.seed, a.steps)
        if res:
            summary["tasks"][task_name] = res

    if not a.skip_rate:
        rate = measure_control_rate(root, Path(a.data_root), a.rate_task,
                                    a.task_config, a.seed)
        if rate:
            summary["control_rate"] = rate

    if not a.skip_replay:
        rep = replay_demo(root, Path(a.data_root), a.rate_task, a.task_config)
        if rep:
            summary["replay"] = rep

    section("summary")
    dofs = {r["dof"] for r in summary["tasks"].values()}
    views = {r["n_views"] for r in summary["tasks"].values()}
    check(len(dofs) == 1, "dof is identical across all four tasks", str(dofs))
    check(len(views) == 1, "n_views is identical across all four tasks", str(views))
    for name, r in summary["tasks"].items():
        print(f"  {name:20s} dof={r['dof']:3d} n_views={r['n_views']}  "
              f"cams={r['cameras']}  render_ok={r['render_ok']}")
    if "control_rate" in summary:
        cr = summary["control_rate"]
        print(f"  measured env_fps     {cr['env_fps_measured']:.4f} Hz  "
              f"(physics {1 / cr['physics_dt']:.0f} Hz / {cr['median_steps_per_frame']:.0f})")
    if "replay" in summary:
        rp = summary["replay"]
        print(f"  demo replay          seed={rp['seed']} actions={rp['n_actions']} "
              f"-> eval_success={rp['eval_success']}")

    print(f"\n  {CHECKS - len(FAILURES)}/{CHECKS} checks passed")
    if FAILURES:
        print("\n  FAILURES:")
        for f in FAILURES:
            print(f"    - {f}")

    if a.out:
        Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
        print(f"\n  summary -> {a.out}")

    print("\n\033[32mSMOKE TEST PASSED\033[0m" if not FAILURES
          else "\n\033[31mSMOKE TEST FAILED\033[0m")
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    sys.exit(main())
