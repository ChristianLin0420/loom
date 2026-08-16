# LIBERO environment + data

Everything needed to reproduce the LIBERO evaluation environment on this cluster.
Owned by Team G. `scripts/setup_libero.sh` automates all of it and is safe to re-run.

> **TL;DR**
> ```bash
> bash scripts/setup_libero.sh
> srun --account=edgeai_tao-ptm_image-foundation-model-clip \
>      --partition=polar4,polar3,polar,grizzly,batch_singlenode \
>      --time=00:30:00 --gpus=1 --nodes=1 \
>      bash -c 'MUJOCO_GL=egl /lustre/fsw/portfolios/edgeai/users/chrislin/envs/loom-libero/bin/python scripts/smoke_libero.py'
> ```

---

## 1. Why a separate environment

LIBERO is pinned to a stack that cannot coexist with LOOM's training `.venv`:

| | LOOM training `.venv` | LIBERO eval env |
|---|---|---|
| python | 3.13 | **3.10** |
| numpy | 2.x | **<2** |
| RL API | — | `gym` (not `gymnasium`) |
| robosuite | — | **1.4.x** |

The other robosuite installs already on this filesystem (`pse/.venv`, `bond-dex`) carry
**robosuite 1.5.2**, which upstream LIBERO does not support. Do not try to reuse them.

The eval harness therefore runs out-of-process from training. Nothing in `loom/**`
imports `libero`; the harness shells into this interpreter.

## 2. Paths

| what | where |
|---|---|
| conda env | `/lustre/fsw/portfolios/edgeai/users/chrislin/envs/loom-libero` |
| LIBERO checkout | `/lustre/fsw/portfolios/edgeai/users/chrislin/projects/loom-deps/LIBERO` |
| datasets | `/lustre/fsw/portfolios/edgeai/users/chrislin/datasets/loom/libero` |
| libero config | `~/.libero/config.yaml` (override with `$LIBERO_CONFIG_PATH`) |

`/lustre/fsw/portfolios/edgeai/users/chrislin` is a symlink to
`/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin`.
Both spellings work; scripts use the `fsw` one.

## 3. LIBERO source: upstream, not the fork

Used **`https://github.com/Lifelong-Robot-Learning/LIBERO.git`** (upstream), pinned at
commit `8f1084e3132a39270c3a13ebe37270a43ece2a01` ("Add support for dataset download
from huggingface", 2025-03-15 — current master).

`cosmos-rl` pins the `fwd4/LIBERO` fork, but the baseline table we fill in
(PLAN.md §8, source Light-WAM Table 1) is measured on upstream. Matching the
baselines' benchmark is the point, so upstream it is. **No fallback to the fork was
needed** — upstream runs green on all four suites (§8). The problems that had to be
solved were packaging ones (mujoco pin, editable-install mode, a missing `future`
dependency), not anything wrong with upstream LIBERO itself.

Note that upstream's `setup.py` has `install_requires=[]`; `requirements.txt` is
**not** applied by `pip install -e .`. That is convenient — its pins
(`numpy==1.22.4`, `transformers==4.21.1`, `robomimic==0.2.0`, `hydra-core==1.2.0`)
are from 2022 and do not resolve on python 3.10 today. We install a curated set
instead. See §5 for what actually landed.

## 4. Assets ship with the repo

Both are in-tree; there is no separate download.

```
libero/libero/bddl_files/<suite>/*.bddl
libero/libero/init_files/<suite>/*.pruned_init
```

| suite | `.bddl` | `.pruned_init` |
|---|---|---|
| `libero_spatial` | 10 | 10 |
| `libero_object` | 10 | 10 |
| `libero_goal` | 10 | 10 |
| `libero_10` | 10 | 10 |
| `libero_90` | 90 | 90 |

Each `.pruned_init` holds that task's evaluation init states. Read the count at
runtime with `len(task_suite.get_task_init_states(task_id))`; never hard-code it.
`libero_goal` is the only suite that does not also ship the unpruned `.init` files;
nothing uses them.

## 5. Versions that landed

| package | version | why this one |
|---|---|---|
| python | **3.10.20** | LIBERO supports 3.8–3.10 |
| numpy | **1.26.4** | `<2`; robosuite 1.4 and gym 0.25 both break on numpy 2 |
| **mujoco** | **2.3.2** | **hard pin, see below** |
| robosuite | **1.4.1** | last 1.4.x; 1.5.x is not supported by LIBERO |
| bddl | **1.0.1** | 3.x changed the parser API LIBERO calls |
| future | 1.0.0 | bddl 1.0.1 imports it but does not declare it |
| gym | **0.25.2** | `<0.26`; LIBERO's `venv.py` uses the 4-tuple `step` API |
| torch | **2.6.0+cu124** | driver is CUDA 12.2; a default wheel silently runs on CPU |
| torchvision | 0.21.0+cu124 | must match torch or it stays ABI-linked to the wheel it replaced |
| h5py | 3.16.0 | |
| opencv-python | 4.11.0 | |
| scipy / numba | 1.15.3 / 0.67.0 | robosuite deps |
| hydra-core | 1.3.5 | LIBERO pins 1.2.0; 1.3.5 works |
| robomimic | 0.3.0 (`--no-deps`) | only `libero.lifelong` needs it; we do not |
| libero | 0.1.0, editable, `editable_mode=compat` | |

**`mujoco==2.3.2` is the single most important pin.** robosuite 1.4.1 only declares
`mujoco>=2.3.0`, so pip resolves 3.11.0, which removed `MjData.qM`. Every env build
then dies with

```
File ".../robosuite/controllers/base_controller.py", line 156, in update
    mujoco.mj_fullM(self.sim.model._model, mass_matrix, self.sim.data.qM)
AttributeError: 'MjData' object has no attribute 'qM'
```

and because it happens *after* EGL has already initialised (you get a wall of
`EGLError(EGL_NOT_INITIALIZED)` from `__del__` teardown first), it reads like a
rendering problem. It is not.

## 6. Headless rendering

`MUJOCO_GL=egl`. It works on the GPU compute nodes and **only** there.

* **Login node**: no GPU, no `/usr/share/glvnd/egl_vendor.d/`, no `libEGL` in
  `ldconfig -p`. Any render will fail or silently produce black frames. Never
  smoke-test on the login node.
* **GPU compute node** (`batch-block*`, driver **535.129.03**, A100-SXM4-80GB): the
  driver libraries are already present, no conda GL packages are needed.
  ```
  /usr/share/glvnd/egl_vendor.d/10_nvidia.json
  /cm/local/apps/cuda/libs/current/lib64/libEGL.so.1
  /cm/local/apps/cuda/libs/current/lib64/libEGL_nvidia.so.0
  /cm/local/apps/cuda/libs/current/lib64/libGLdispatch.so.0
  ```
* `osmesa` is **not** available anywhere (no `libOSMesa` in `ldconfig -p` on either
  node type). If EGL ever breaks, osmesa would have to be installed from conda-forge
  (`mesalib`) first — it is not a zero-work fallback here.

Set both, because robosuite/mujoco and any PyOpenGL path read different variables:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

`scripts/smoke_libero.py` sets these itself if they are unset.

## 7. Datasets

Source: HuggingFace dataset `yifengzhu-hf/LIBERO-datasets`, revision
**`f13aa24a3da8c43c7225569f28c562979fa0e35a`** (pin it — the repo is mutable).
Token comes from `<repo>/.env.local` (`$HF_TOKEN`); never echo it.

```
/lustre/fsw/portfolios/edgeai/users/chrislin/datasets/loom/libero/
├── libero_spatial/   10 × *.hdf5   5.9 GB
├── libero_object/    10 × *.hdf5   7.0 GB   (hardlinks into PSE's copy)
├── libero_goal/      10 × *.hdf5   6.0 GB
└── libero_10/        10 × *.hdf5    13 GB
```

**All four suites: 10 tasks × 50 demos = 500 demos per suite, 2000 total.**

`libero_object` is **hardlinked** from
`/lustre/fsw/portfolios/edgeai/users/chrislin/projects/pse/data/libero/libero_object`
(same lustre filesystem, so it costs zero extra bytes and survives PSE moving their
directory). **Never write to or delete the PSE copy — another project trains on it.**

### HDF5 layout

```
data                                     attrs: num_demos=50, bddl_file_name,
                                                env_name, problem_info
                                                (problem_info is JSON, holds
                                                 'language_instruction')
data/demo_<i>/
    actions        (T, 7)   float64
    dones          (T,)
    rewards        (T,)
    states         (T, S)   full mujoco state; S is per-suite
                            (spatial 92, object 110, goal 79, libero_10 47)
    robot_states   (T, R)
    obs/
        agentview_rgb    (T, 128, 128, 3) uint8
        eye_in_hand_rgb  (T, 128, 128, 3) uint8
        ee_pos, ee_ori, ee_states, gripper_states, joint_states
```

`env_name` is the robosuite problem: `Libero_Tabletop_Manipulation` (spatial, goal),
`Libero_Floor_Manipulation` (object), `Libero_Kitchen_Tabletop_Manipulation`
(libero_10). The demo images are 128×128; the eval harness renders at 256×256 —
these are independent knobs.

## 8. Smoke test

`scripts/smoke_libero.py`. GPU node only. For each of the four suites it builds
`OffScreenRenderEnv` on task 0, resets to a real `.pruned_init` state, applies the 15
settling actions, and asserts on:

* rendered frames are **not all black** (`std > 1`, `max > 16`) — the silent EGL
  failure mode. A shape-only check passes while the policy sees nothing.
* `control_freq == 20 Hz`, against `contracts.EMBODIMENTS["libero_franka"].env_fps`.
* `action_dim == 7`, plus the reported bounds.
* a full 512-step episode terminates at the cap with the `done` flag plumbed through.
* **image orientation, measured** — see §8.1.

Measured on `batch-block5-02014` (A100-SXM4-80GB, driver 535.129.03), all four suites
PASS:

| | value |
|---|---|
| `MUJOCO_GL` | `egl` |
| `torch.cuda.is_available()` | `True` |
| **`control_freq`** | **20.0 Hz** — matches the frozen `env_fps=20.0` |
| `control_timestep` / `model_timestep` | 0.05 s / 0.002 s (25 physics substeps) |
| **`action_dim`** | **7**, low `[-1]*7`, high `[+1]*7` — matches the frozen spec |
| render | 256×256×3 uint8, per-frame `std` 22–65, `max` 221–255, `frac_nonzero` ≥ 0.978 — **not all black** |
| harness state dim | 8 (eef_pos 3 + axis-angle 3 + gripper_qpos 2) |
| episode | 512/512 steps, `done=False`, `check_success()=False` (random actions) |
| throughput | 0.016–0.028 s/step, i.e. **36–55 env steps/s** on one A100, one env |
| env build | 4–13 s per env |

That throughput sets the eval budget: one 512-step episode is ~10–15 s, so the
10 ep/task × 10 tasks × 4 suites protocol is ~400 episodes ≈ 1.5 h single-process,
and needs to be parallelised across GPUs/processes to fit comfortably in a 4 h link.

### 8.1 Image orientation

Train and eval must agree on orientation or the model trains fine and scores near
zero with no other symptom. Two claims were in conflict:

* the demo HDF5s carry `macros_image_convention = 'opengl'` (stored bottom-up), and
  `loom/data/adapters/libero.py` applies `orient_dataset_image` = **vertical flip**;
* the cosmos-rl reference wrapper flips live env frames with `[::-1, ::-1]`, a **180°
  rotation**.

The smoke test settles it empirically rather than by reading source. It drives the sim
to the *exact* mujoco state of `data/demo_0/states[0]`, renders at the demo's own
128×128, and scores the live frame against the stored `agentview_rgb[0]` under all four
candidates (`identity`, `vflip`, `hflip`, `rot180`) using
`best_matching_transform()` and `_TRANSFORMS` imported from
`loom/data/adapters/libero.py` — the same objects the training cache uses, so the two
paths cannot drift. It reports the full MAE table, the winner, the runner-up and the
margin, and marks the result **NOT CONCLUSIVE** if the margin is small.

**Measured result — the answer is `identity`, on all 8 measurements (4 suites × 2
cameras).** Mean absolute error, live env frame vs stored demo frame at the same
mujoco state:

| suite | camera | identity | hflip | vflip | rot180 |
|---|---|---:|---:|---:|---:|
| libero_spatial | agentview | **3.29** | 38.97 | 55.36 | 70.27 |
| libero_spatial | eye_in_hand | **5.59** | 44.34 | 67.52 | 74.54 |
| libero_object | agentview | **3.99** | 21.84 | 54.54 | 57.15 |
| libero_object | eye_in_hand | **12.21** | 47.37 | 62.45 | 60.02 |
| libero_goal | agentview | **4.65** | 47.89 | 58.03 | 73.35 |
| libero_goal | eye_in_hand | **8.82** | 50.88 | 77.38 | 75.71 |
| libero_10 | agentview | **3.35** | 38.58 | 44.57 | 47.17 |
| libero_10 | eye_in_hand | **7.56** | 13.47 | 20.16 | 20.74 |

Consequences:

1. **The live env frame and the stored demo frame are in the same orientation.** Both
   are `opengl` / bottom-up — the demo attribute says `macros_image_convention =
   'opengl'` and `robosuite.macros.IMAGE_CONVENTION` is `'opengl'` at runtime, and the
   measurement confirms robosuite is not flipping anything for us.
2. **`orient_env_image(img, env_convention='opengl')` is correct** — the eval harness
   must apply exactly the same vertical flip to live frames that
   `orient_dataset_image` applies to demo frames. That is already the default in
   `loom/data/adapters/libero.py`; no change needed.
3. **cosmos-rl's `[::-1, ::-1]` is wrong for this pipeline.** `rot180` is the worst or
   second-worst candidate in every one of the 8 measurements. Do not copy that line
   from the reference wrapper.

Caveat, stated because the margin matters: the residual for `identity` is not exactly
zero (3–12 MAE) because the live render is at a fresh `reset()` while the demo frame
was captured mid-collection, so lighting/AA and the gripper's exact sub-step pose
differ slightly. Seven of the eight measurements separate `identity` from the runner-up
by a factor of 3.9–11.5. The eighth, `libero_10` eye_in_hand, separates by only 1.8×
(7.56 vs 13.47) — the wrist camera at that task's initial pose is filled by the
gripper and is nearly symmetric, so it is weak evidence on its own. It still votes
`identity`, and the agentview for the same suite is decisive (3.35 vs 38.58), so the
conclusion is not in doubt. The script prints `WEAK MARGIN` on that row rather than
hiding it.

## 9. Closed-loop harness facts (for `loom/eval/libero.py`)

Extracted from the working reference at
`/lustre/fsw/portfolios/edgeai/users/chrislin/projects/cr/cosmos-reason2/cosmos-rl/cosmos_rl/simulators/libero/`
and confirmed against this checkout.

* bddl path:
  `os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)`
* init state:
  `task_suite.get_task_init_states(task_id)[trial_id % n_trials]`, then
  **15 dummy `[0,0,0,0,0,0,-1]` actions** to let the scene settle before the policy
  acts. Reproducing published success rates depends on this.
* observations are flipped `[::-1, ::-1]` from `agentview_image` and
  `robot0_eye_in_hand_image`.
* proprio state = `robot0_eef_pos` (3) + `quat2axisangle(robot0_eef_quat)` (3) +
  `robot0_gripper_qpos` (2) = **8**.
* episode cap **512 steps** for every suite.
* success is the `done` flag returned by `env.step`. LIBERO overrides
  `bddl_base_domain.step` with `done = self._check_success()`, so `done` is success,
  not horizon exhaustion. Terminate on `done` or at 512 steps.
* env class `libero.libero.envs.OffScreenRenderEnv`, built inside a **spawn**
  subprocess. MuJoCo/EGL contexts do not survive teardown-and-rebuild inside one
  interpreter — one env per process.
* control frequency **20 Hz** (`ControlEnv.__init__` default `control_freq=20`),
  matching `contracts.EMBODIMENTS["libero_franka"].env_fps == 20.0`. The decoder emits
  `H_OP` steps at `FPS_CANONICAL = 30`; resample 30 → 20 before executing.
* action space is 7-dim (OSC_POSE 6 + gripper), bounds `[-1, 1]`.
* `Benchmark.__init__` takes `task_order_index=0` and permutes the 10 tasks through
  `benchmark.task_orders[task_order_index]`. **Always instantiate with the default
  `task_order_index=0`** (identity order) or `get_task(i)` silently evaluates a
  different task than the demo file name suggests. The suite orderings are the ones in
  `libero/libero/benchmark/libero_suite_task_map.py`.
* `Task` carries `name`, `language`, `problem_folder`, `bddl_file`,
  `init_states_file`. `task.language` is derived from the filename and matches the
  `language_instruction` in the demo HDF5.
* **`get_task_init_states` calls a bare `torch.load(path)`.** torch >= 2.6 defaults
  that to `weights_only=True` and refuses the payload. Patch `torch.load` to
  `weights_only=False` before the first call — see
  `patch_torch_load_for_init_states()` in `scripts/smoke_libero.py`; copy it.

## 10. Gotchas

* **`import libero.libero` calls `input()`** when `~/.libero/config.yaml` is missing,
  which hangs any batch job forever with no output. `touch ~/.libero/config.yaml`
  *before* the first import, then call `set_libero_default_path()` to write real
  contents. `scripts/setup_libero.sh` does this.
* An empty `config.yaml` makes `get_libero_path` raise
  `TypeError: 'NoneType' object is not iterable` — the touch is only a prompt
  suppressor, the yaml still has to be written.
* **torch must be installed last, from `https://download.pytorch.org/whl/cu124`.**
  `robomimic` depends on torch and pulls a default wheel that imports fine and reports
  `torch.cuda.is_available() == False` on this CUDA-12.2 driver. See CLAUDE.md.
* Compute nodes have **no outbound network**. Everything (env, repo, datasets) must be
  fully materialised from a login node first.
* **Always set `CONDA_PKGS_DIRS` to a node-local path.** The default cache
  `~/miniconda3/pkgs` is on lustre and is shared. Two agents running
  `conda create python=3.10` at the same time raced on it here and produced a
  half-extracted package — 40 lines of
  `CondaVerificationError: The package for python ... appears to be corrupted` and no
  env at all, after 25 minutes. Rebuilding with `CONDA_PKGS_DIRS=/tmp/...` took ~2 min.
  Do **not** `conda clean` the shared cache to fix this; another job may be using it.
* `conda create` writing into lustre is I/O-bound and slow (25 min for a bare python
  3.10 when the cache is also on lustre). It is not hung — check
  `/proc/<pid>/io` for a rising `wchar` before assuming otherwise.
* pip is much faster with `PIP_CACHE_DIR` and `TMPDIR` on node-local `/tmp`
  (`/` is a 969 GB overlay here). `scripts/setup_libero.sh` sets both.
* **`pip install -e LIBERO` needs `--config-settings editable_mode=compat`.** LIBERO's
  outer `libero/` directory has no `__init__.py`, so it is a PEP-420 namespace package.
  setuptools >= 64's strict editable finder calls `find_packages()`, gets an empty
  mapping, and installs a distribution that `pip list` reports as present but that
  `import libero` cannot find. compat mode writes the old-style `.pth` pointing at the
  project root, which is what LIBERO's install instructions assume. The setup script
  falls back to writing that `.pth` by hand if compat mode ever stops working.
* **robomimic cannot be installed with its dependencies here.** Its `egl_probe`
  dependency compiles against EGL headers that are not on the image and cannot be
  installed without root. `pip install --no-deps robomimic` works and imports fine;
  nothing in the eval path needs it (only `libero.lifelong` does).
* Import order matters at *runtime* too: `import robosuite` on a node without EGL dies
  with `AttributeError: 'NoneType' object has no attribute 'eglQueryString'`. So no
  part of the LIBERO stack can be imported on the login node — every check, even a
  physics-only one, has to go through `srun`.
* robosuite prints `No private macro file found!` on every import. Harmless; do **not**
  run `setup_macros.py` — the default `IMAGE_CONVENTION = "opengl"` is the one the
  orientation measurement in §8.1 was taken under.
* The dataset `hdf5` files are big (up to 1.4 GB each). `snapshot_download` with
  `max_workers=8` pulls all 30 files in ~2.5 min from the login node.
