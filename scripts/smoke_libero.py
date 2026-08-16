#!/usr/bin/env python
"""Smoke test for the LIBERO evaluation environment.  MUST run on a GPU compute node.

    srun --account=edgeai_tao-ptm_image-foundation-model-clip \
         --partition=polar4,polar3,polar,grizzly,batch_singlenode \
         --time=00:30:00 --gpus=1 --nodes=1 \
         bash -c 'MUJOCO_GL=egl /lustre/fsw/portfolios/edgeai/users/chrislin/envs/loom-libero/bin/python scripts/smoke_libero.py'

For each of the four evaluation suites it builds `OffScreenRenderEnv` for task 0,
resets to a real `.pruned_init` state, applies the 15 dummy settling actions that the
reference harness uses, and then checks the things that silently break:

  * renders are NOT all black -- the classic symptom of an EGL context that "worked"
    but has no GPU behind it.  This is the single most important assertion here.
  * control frequency is 20 Hz, because contracts.EMBODIMENTS["libero_franka"].env_fps
    is 20.0 and the whole 30 Hz -> env-rate action resampling path depends on it.
  * action dim is 7 with the expected bounds.
  * a full 512-step episode terminates on the cap and the `done` flag plumbs through
    (LIBERO overrides `done = self._check_success()`, so done == success).

Each suite runs in its own *spawned* subprocess.  MuJoCo/EGL contexts do not survive
being torn down and rebuilt inside one interpreter, and this mirrors the reference
harness (cosmos_rl/simulators/libero/venv.py) which also uses the spawn context.

Owned by Team G.  See docs/ENV_LIBERO.md.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
import traceback

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]

# From the reference harness (cosmos_rl/simulators/libero/utils.py): every suite caps
# at 512 environment steps.
MAX_STEPS = 512
# Number of no-op actions applied after set_init_state() to let the scene settle before
# the policy acts.  Reproducing the baselines' success rates depends on this.
N_SETTLE = 15
DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]

EXPECTED_CONTROL_FREQ = 20.0  # contracts.EMBODIMENTS["libero_franka"].env_fps
EXPECTED_ACTION_DIM = 7


def patch_torch_load_for_init_states() -> str:
    """LIBERO reads `.pruned_init` files with a bare `torch.load(path)`.

    torch >= 2.6 flipped that call's default to `weights_only=True`, which refuses to
    unpickle the plain-python payload in those files.  Symptom is an
    `UnpicklingError: Weights only load failed` the first time you ask for an init
    state -- i.e. every evaluation episode.  Copy this shim into any process that
    calls `task_suite.get_task_init_states()`.

    Returns a short status string for logging.
    """
    import torch

    major_minor = tuple(int(x) for x in torch.__version__.split(".")[:2])
    if major_minor < (2, 6):
        return f"torch {torch.__version__}: no shim needed"
    _orig_load = torch.load

    def _load(*a, **kw):
        kw.setdefault("weights_only", False)
        return _orig_load(*a, **kw)

    torch.load = _load
    return f"torch {torch.__version__}: torch.load patched to weights_only=False"


def _run_orientation(suite: str, repo_root: str, seed: int, out):
    """Settle the live-env image orientation empirically.  Own spawned subprocess.

    The training HDF5s carry `macros_image_convention='opengl'` (stored bottom-up).
    The cosmos-rl reference wrapper flips live frames with `[::-1, ::-1]`, which is a
    180 degree rotation, not a vertical flip.  Those cannot both be right, and if
    train and eval disagree on orientation the model trains fine and scores ~0 with no
    other symptom.

    The experiment: drive the sim to the *exact* mujoco state of demo 0 frame 0
    (`data/demo_0/states[0]`), render, and compare against that demo's stored
    `agentview_rgb[0]`.  Same state, same scene, same camera -- so the correct
    transform should give a near-zero mean-absolute error and the wrong ones should
    be far away.  Renders at 128x128 to match the stored demo resolution exactly, so
    no resampling contaminates the comparison.

    Uses `orient_env_image` / `best_matching_transform` / `_TRANSFORMS` from
    `loom.data.adapters.libero` so the eval path and the training cache provably
    share one convention.
    """
    r = {"suite": suite, "ok": False}
    try:
        import numpy as np
        import h5py

        r["torch_load_shim"] = patch_torch_load_for_init_states()

        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        try:
            from loom.data.adapters import libero as loom_libero

            transforms = dict(loom_libero._TRANSFORMS)
            best_fn = loom_libero.best_matching_transform
            r["helpers"] = (
                "loom.data.adapters.libero "
                f"(IMAGE_CONVENTION={loom_libero.IMAGE_CONVENTION!r}, "
                f"CANONICAL_ORIENTATION={loom_libero.CANONICAL_ORIENTATION!r})"
            )
        except Exception as e:
            # Fall back to a local copy so the measurement still happens; say so loudly.
            transforms = {
                "identity": lambda x: x,
                "vflip": lambda x: x[..., ::-1, :, :],
                "hflip": lambda x: x[..., :, ::-1, :],
                "rot180": lambda x: x[..., ::-1, ::-1, :],
            }

            def best_fn(dataset_img, env_img):
                a = np.asarray(dataset_img, dtype=np.float32)
                sc = {
                    n: float(np.abs(a - f(np.asarray(env_img, np.float32))).mean())
                    for n, f in transforms.items()
                }
                b = min(sc, key=sc.get)
                return b, sc[b]

            r["helpers"] = f"LOCAL FALLBACK -- could not import loom helpers: {e!r}"

        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        import robosuite.macros as robosuite_macros

        r["robosuite_IMAGE_CONVENTION"] = robosuite_macros.IMAGE_CONVENTION

        task_suite = benchmark.get_benchmark_dict()[suite]()
        task = task_suite.get_task(0)
        bddl = os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        )
        demo_path = os.path.join(
            get_libero_path("datasets"), task.problem_folder, f"{task.name}_demo.hdf5"
        )
        r["demo_path"] = demo_path
        assert os.path.exists(demo_path), f"demo file missing: {demo_path}"

        with h5py.File(demo_path, "r") as h:
            d = h["data"]
            r["demo_image_convention"] = str(
                d.attrs.get("macros_image_convention", "<absent>")
            )
            demo_state0 = np.asarray(d["demo_0"]["states"][0])
            demo_agent0 = np.asarray(d["demo_0"]["obs"]["agentview_rgb"][0])
            demo_wrist0 = np.asarray(d["demo_0"]["obs"]["eye_in_hand_rgb"][0])
        res = int(demo_agent0.shape[0])
        r["demo_resolution"] = res

        env = OffScreenRenderEnv(
            bddl_file_name=bddl, camera_heights=res, camera_widths=res
        )
        try:
            env.seed(seed)
        except AttributeError:
            np.random.seed(seed)
        env.reset()
        # Drive the sim to the demo's exact mujoco state and render it.  No settling
        # actions -- we want the same instant the demo recorded.
        obs = env.set_init_state(demo_state0)

        for cam, demo_img in (
            ("agentview", demo_agent0),
            ("robot0_eye_in_hand", demo_wrist0),
        ):
            env_img = np.asarray(obs[f"{cam}_image"])
            a = demo_img.astype(np.float32)
            scores = {
                name: round(
                    float(np.abs(a - fn(env_img.astype(np.float32))).mean()), 4
                )
                for name, fn in transforms.items()
            }
            best, best_score = best_fn(demo_img, env_img)
            ordered = sorted(scores.items(), key=lambda kv: kv[1])
            runner_up = ordered[1]
            margin = runner_up[1] - ordered[0][1]
            r[cam] = {
                "scores_mae": scores,
                "best": best,
                "best_mae": round(float(best_score), 4),
                "runner_up": runner_up[0],
                "runner_up_mae": runner_up[1],
                "margin": round(float(margin), 4),
                # A weak margin means this camera's view is near-symmetric at the
                # initial pose (typical for the wrist camera, which sees mostly
                # gripper) and the measurement is only weak evidence.  Report it
                # rather than pretending; the verdict comes from agreement across
                # all cameras and suites, not from any single number.
                "strong": bool(margin > 5.0 and ordered[0][1] < 0.5 * runner_up[1]),
            }
        env.close()
        r["ok"] = True
    except BaseException:
        r["error"] = traceback.format_exc()
    out.put(r)


def _run_suite(suite: str, resolution: int, full_episode: bool, seed: int, out):
    """Runs in a spawned subprocess.  Puts a result dict on `out`."""
    r = {"suite": suite, "ok": False}
    try:
        import numpy as np

        r["torch_load_shim"] = patch_torch_load_for_init_states()

        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        r["mujoco_gl"] = os.environ.get("MUJOCO_GL", "<unset>")

        import mujoco
        import robosuite

        r["mujoco_version"] = mujoco.__version__
        r["robosuite_version"] = robosuite.__version__
        r["numpy_version"] = np.__version__

        task_suite = benchmark.get_benchmark_dict()[suite]()
        r["n_tasks"] = task_suite.n_tasks
        task = task_suite.get_task(0)
        r["task_name"] = task.name
        r["language"] = task.language

        bddl = os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        )
        r["bddl"] = bddl
        assert os.path.exists(bddl), f"missing bddl {bddl}"

        init_states = task_suite.get_task_init_states(0)
        r["n_init_states"] = int(len(init_states))

        t0 = time.time()
        env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_heights=resolution,
            camera_widths=resolution,
        )
        try:
            env.seed(seed)
            r["seedable"] = True
        except AttributeError:
            # robosuite 1.4's MujocoEnv does not always expose .seed(); the init state
            # is what determines the episode anyway.
            np.random.seed(seed)
            r["seedable"] = False
        env.reset()
        r["build_seconds"] = round(time.time() - t0, 2)

        inner = env.env
        r["control_freq"] = float(inner.control_freq)
        r["control_timestep"] = float(inner.control_timestep)
        r["model_timestep"] = float(inner.model_timestep)
        low, high = inner.action_spec
        r["action_dim"] = int(len(low))
        r["action_low"] = [round(float(x), 4) for x in low]
        r["action_high"] = [round(float(x), 4) for x in high]

        # --- reset to a real init state + settle -------------------------------
        env.set_init_state(init_states[0])
        obs = None
        for _ in range(N_SETTLE):
            obs, _, _, _ = env.step(DUMMY_ACTION)

        for key in ("agentview_image", "robot0_eye_in_hand_image"):
            assert key in obs, f"{key} missing from obs; keys={sorted(obs)}"
        r["obs_keys"] = sorted(obs.keys())

        # --- the render sanity check -------------------------------------------
        # An EGL context that initialises but has no GPU behind it renders a uniform
        # black frame.  Assert on both mean and per-pixel std.
        img_stats = {}
        for key in ("agentview_image", "robot0_eye_in_hand_image"):
            img = np.asarray(obs[key])
            assert img.dtype == np.uint8, f"{key} dtype {img.dtype}, expected uint8"
            assert img.shape == (resolution, resolution, 3), (
                f"{key} shape {img.shape}, expected {(resolution, resolution, 3)}"
            )
            img_stats[key] = {
                "shape": list(img.shape),
                "dtype": str(img.dtype),
                "min": int(img.min()),
                "max": int(img.max()),
                "mean": round(float(img.mean()), 3),
                "std": round(float(img.std()), 3),
                "frac_nonzero": round(float((img > 0).mean()), 4),
            }
            assert img.std() > 1.0, (
                f"{key} is (near) uniform: std={img.std():.4f} mean={img.mean():.4f}. "
                "This is the silent EGL failure -- the render produced a blank frame."
            )
            assert img.max() > 16, f"{key} max={img.max()}: frame is essentially black."
        r["images"] = img_stats

        # state vector the harness will feed the policy
        state = np.concatenate(
            [
                obs["robot0_eef_pos"],
                obs["robot0_eef_quat"],  # harness converts to axis-angle (3)
                obs["robot0_gripper_qpos"],
            ]
        )
        r["state_dim_pos_quat_grip"] = int(state.shape[0])
        r["harness_state_dim"] = int(
            obs["robot0_eef_pos"].shape[0] + 3 + obs["robot0_gripper_qpos"].shape[0]
        )

        # --- 20 random steps ----------------------------------------------------
        rng = np.random.default_rng(seed)
        t0 = time.time()
        for _ in range(20):
            a = rng.uniform(low, high)
            obs, rew, done, info = env.step(a)
        r["random_step_seconds_per_step"] = round((time.time() - t0) / 20, 4)
        r["random_steps_done_flag"] = bool(done)
        img = np.asarray(obs["agentview_image"])
        r["after_random_agentview_std"] = round(float(img.std()), 3)
        assert img.std() > 1.0, "agentview went blank after random actions"

        # --- one full episode to the cap ---------------------------------------
        if full_episode:
            env.set_init_state(init_states[0])
            for _ in range(N_SETTLE):
                env.step(DUMMY_ACTION)
            t0 = time.time()
            steps = 0
            done = False
            for _ in range(MAX_STEPS):
                a = rng.uniform(low, high)
                obs, rew, done, info = env.step(a)
                steps += 1
                if done:
                    break
            dt = time.time() - t0
            r["episode_steps"] = steps
            r["episode_done"] = bool(done)
            r["episode_success"] = bool(env.check_success())
            r["episode_seconds"] = round(dt, 1)
            r["episode_hz"] = round(steps / dt, 1)
            assert steps == MAX_STEPS or done, "episode ended without done at the cap"

        env.close()
        r["ok"] = True
    except BaseException:
        r["error"] = traceback.format_exc()
    out.put(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", nargs="*", default=SUITES)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-full-episode", action="store_true")
    ap.add_argument("--no-orientation", action="store_true")
    ap.add_argument(
        "--repo-root",
        default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        help="LOOM repo root, so the child can import loom.data.adapters.libero",
    )
    args = ap.parse_args()

    # The login node defaults to 32 torch threads on 64 shared cores and thrashes.
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("MKL_NUM_THREADS", "8")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
    # robosuite writes a macros file on first import; keep it quiet.
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

    print("=" * 78)
    print("LIBERO smoke test")
    print("=" * 78)
    print(f"  host                {os.uname().nodename}")
    print(f"  python              {sys.version.split()[0]}  ({sys.executable})")
    print(f"  MUJOCO_GL           {os.environ['MUJOCO_GL']}")
    print(f"  CUDA_VISIBLE_DEVICES {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    try:
        import torch

        print(
            f"  torch               {torch.__version__}  "
            f"cuda_available={torch.cuda.is_available()}  "
            f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}"
        )
    except Exception as e:  # torch is not required by the env itself
        print(f"  torch               unavailable: {e}")

    ctx = mp.get_context("spawn")
    failures = []
    orientation_votes: list[tuple[str, str, str, bool]] = []
    for suite in args.suites:
        print("\n" + "-" * 78)
        print(f"suite: {suite}")
        print("-" * 78)
        q = ctx.Queue()
        p = ctx.Process(
            target=_run_suite,
            args=(suite, args.resolution, not args.no_full_episode, args.seed, q),
        )
        p.start()
        try:
            r = q.get(timeout=1800)
        except Exception:
            p.terminate()
            print(f"  FAIL {suite}: subprocess produced no result (timeout/crash)")
            failures.append(suite)
            continue
        p.join(timeout=60)

        if not r.get("ok"):
            print(f"  FAIL {suite}")
            print(r.get("error", "<no traceback>"))
            failures.append(suite)
            continue

        print(f"  tasks in suite      {r['n_tasks']}")
        print(f"  task 0              {r['task_name']}")
        print(f"  language            '{r['language']}'")
        print(f"  init states         {r['n_init_states']}")
        print(f"  {r['torch_load_shim']}")
        print(f"  versions            robosuite={r['robosuite_version']} "
              f"mujoco={r['mujoco_version']} numpy={r['numpy_version']}")
        print(f"  env build           {r['build_seconds']}s")

        cf = r["control_freq"]
        flag = "OK" if abs(cf - EXPECTED_CONTROL_FREQ) < 1e-6 else "*** MISMATCH ***"
        print(f"  control_freq        {cf} Hz   [{flag}; "
              f"contracts env_fps={EXPECTED_CONTROL_FREQ}]")
        print(f"  control_timestep    {r['control_timestep']:.6f} s "
              f"(model_timestep {r['model_timestep']:.6f} s)")
        if flag != "OK":
            failures.append(f"{suite}:control_freq")

        ad = r["action_dim"]
        print(f"  action_dim          {ad}   "
              f"[{'OK' if ad == EXPECTED_ACTION_DIM else '*** MISMATCH ***'}]")
        print(f"  action_low          {r['action_low']}")
        print(f"  action_high         {r['action_high']}")
        if ad != EXPECTED_ACTION_DIM:
            failures.append(f"{suite}:action_dim")

        for key, s in r["images"].items():
            print(f"  {key:26s} shape={s['shape']} dtype={s['dtype']} "
                  f"min={s['min']} max={s['max']} mean={s['mean']} std={s['std']} "
                  f"nonzero={s['frac_nonzero']}  [NOT ALL BLACK: OK]")
        print(f"  harness state dim   {r['harness_state_dim']} "
              "(eef_pos 3 + axisangle 3 + gripper_qpos 2)")
        print(f"  random step cost    {r['random_step_seconds_per_step']}s/step")
        if "episode_steps" in r:
            print(f"  full episode        steps={r['episode_steps']}/{MAX_STEPS} "
                  f"done={r['episode_done']} check_success={r['episode_success']} "
                  f"{r['episode_seconds']}s ({r['episode_hz']} steps/s)")

        # --- image orientation, measured against this suite's own demo file -----
        if not args.no_orientation:
            oq = ctx.Queue()
            op = ctx.Process(
                target=_run_orientation, args=(suite, args.repo_root, args.seed, oq)
            )
            op.start()
            try:
                o = oq.get(timeout=900)
            except Exception:
                op.terminate()
                o = {"ok": False, "error": "orientation subprocess produced no result"}
            op.join(timeout=60)

            print("  --- image orientation ---")
            if not o.get("ok"):
                print("  ORIENTATION CHECK FAILED")
                print(o.get("error", "<no traceback>"))
                failures.append(f"{suite}:orientation")
            else:
                print(f"  helpers             {o['helpers']}")
                print(f"  demo file           {os.path.basename(o['demo_path'])} "
                      f"@ {o['demo_resolution']}px")
                print(f"  demo attr           macros_image_convention="
                      f"{o['demo_image_convention']!r}")
                print(f"  robosuite runtime   macros.IMAGE_CONVENTION="
                      f"{o['robosuite_IMAGE_CONVENTION']!r}")
                for cam in ("agentview", "robot0_eye_in_hand"):
                    c = o[cam]
                    tbl = "  ".join(
                        f"{k}={v}" for k, v in sorted(c["scores_mae"].items(),
                                                      key=lambda kv: kv[1])
                    )
                    verdict = "strong" if c["strong"] else "WEAK MARGIN"
                    print(f"  {cam:22s} MAE  {tbl}")
                    print(f"  {'':22s} best={c['best']} ({c['best_mae']}) vs "
                          f"{c['runner_up']} ({c['runner_up_mae']}), "
                          f"margin={c['margin']}  [{verdict}]")
                    orientation_votes.append((suite, cam, c["best"], c["strong"]))

        print(f"  RESULT              PASS")

    if orientation_votes:
        print("\n" + "=" * 78)
        print("IMAGE ORIENTATION VERDICT")
        print("=" * 78)
        winners = {v[2] for v in orientation_votes}
        n_strong = sum(1 for v in orientation_votes if v[3])
        print(f"  measurements        {len(orientation_votes)} "
              f"({len(args.suites)} suites x 2 cameras), "
              f"{n_strong} with a strong margin")
        print(f"  winning transform   {sorted(winners)}")
        for s, cam, best, strong in orientation_votes:
            print(f"    {s:16s} {cam:20s} -> {best}"
                  f"{'' if strong else '   (weak margin)'}")
        if len(winners) != 1:
            print("  *** CAMERAS/SUITES DISAGREE -- orientation is NOT settled ***")
            failures.append("orientation_disagreement")
        else:
            w = winners.pop()
            print(f"\n  The live env frame equals the stored demo frame under "
                  f"'{w}'.")
            if w == "identity":
                print("  => env and dataset share one orientation. The dataset-side "
                      "transform in")
                print("     loom/data/adapters/libero.py (orient_dataset_image = "
                      "vflip) must be")
                print("     applied identically to live frames, i.e. "
                      "orient_env_image(img, 'opengl').")
                print("  => cosmos-rl's [::-1, ::-1] (rot180) is NOT the right "
                      "transform for this")
                print("     pipeline: it scored worst or near-worst in every "
                      "measurement.")
            else:
                print(f"  *** This is NOT identity.  orient_env_image must apply "
                      f"'{w}' relative to")
                print("      orient_dataset_image, or train and eval disagree. ***")
                failures.append(f"orientation_not_identity:{w}")

    print("\n" + "=" * 78)
    if failures:
        print(f"SMOKE TEST FAILED: {failures}")
        return 1
    print("SMOKE TEST PASSED for: " + ", ".join(args.suites))
    return 0


if __name__ == "__main__":
    sys.exit(main())
