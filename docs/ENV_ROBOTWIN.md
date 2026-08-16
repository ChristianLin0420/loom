# RoboTwin 2.0 — environment, assets, and the embodiment contract

Owner: **Team H**. RoboTwin 2.0 is the **R0-B decision gate** (PLAN §7): `<55` kills the
operator formulation, `55–75` proceeds to the 13-day pretraining chain, `≥75` is a strong
pass. Everything in this file exists so that number is trustworthy.

---

## 1. TL;DR — the numbers the adapter must register

RoboTwin 2.0's default embodiment is **aloha-agilex** (a dual-arm AgileX/ARX-5 platform on a
mobile base). Every value below is verified against the running environment and the released
trajectories, not assumed.

```python
# loom/data/adapters/robotwin.py — register at import time
from contracts import EmbodimentSpec, register_embodiment

_ARM_LO = (-10.0,) * 6      # URDF revolute limits, arx5_description_isaac.urdf
_ARM_HI = ( 10.0,) * 6

register_embodiment(EmbodimentSpec(
    name="robotwin_aloha_agilex",
    dof=14,                      # (6 arm + 1 gripper) x 2
    env_fps=250.0 / 15.0,        # = 16.6667 Hz   <-- read §4, do NOT use 15.0
    n_views=4,                   # head, front(third-view), left wrist, right wrist
    action_low =_ARM_LO + (0.0,) + _ARM_LO + (0.0,),
    action_high=_ARM_HI + (1.0,) + _ARM_HI + (1.0,),
))
```

| field | value | where it comes from |
|---|---|---|
| `dof` | **14** | `env_cfg/robot/_robot_info.json` → `aloha_agilex: arm_dim [6,6], ee_dim [1,1]`; `assets/embodiments/aloha-agilex/config.yml` → `arm_joints_name` is 6 per arm; every released HDF5 has `/action/joint_states` of shape `(T, 14)` |
| `env_fps` | **16.6667 Hz** (= 250 / 15) | physics timestep `1/250` s (`envs/_base_task.py:223`) decimated by `save_freq: 15` (`env_cfg/task_config/demo_clean.yml`) |
| `n_views` | **4** | `config.yml: static_camera_list = [head_camera, front_camera]` plus the two wrist cameras; the released HDF5 carries `cam_head`, `cam_third_view`, `cam_left_wrist`, `cam_right_wrist` |
| resolution | **240 × 320 × 3**, fovy 37° | `env_cfg/task_config/_camera_config.yml` → `D435: {w: 320, h: 240, fovy: 37}` |
| `action_low` / `action_high` | arms `±10.0` rad, grippers `[0, 1]` | arm: URDF `<limit lower="-10" upper="10">` on `fl_joint1..6` / `fr_joint1..6`; gripper: `Robot.set_gripper()` does `np.clip(gripper_val, 0, 1)` |
| absolute or delta | **ABSOLUTE joint targets** | proved empirically, see §5 |

Action vector layout (index → meaning):

```
[ 0.. 5]  left  arm joints fl_joint1..fl_joint6     absolute radians
[ 6   ]   left  gripper                             normalised, 0 = closed, 1 = open
[ 7..12]  right arm joints fr_joint1..fr_joint6     absolute radians
[13   ]   right gripper                             normalised, 0 = closed, 1 = open
```

### The URDF bounds are permissive; use the empirical ones for normalisation

`±10 rad` is what the URDF declares, and it is what `EmbodimentSpec` should carry because it
is the true admissible range. It is useless for normalisation. Measured over the released
`demo_clean` aloha-agilex trajectories for the four PLAN §8 tasks (200 episodes, 48 454
frames):

| idx | joint | min | max |
|---|---|---|---|
| 0 | L_j1 | −6.3779 | 5.6815 |
| 1 | L_j2 | −0.0000 | 3.8880 |
| 2 | L_j3 | −0.0296 | 4.5009 |
| 3 | L_j4 | −1.8917 | 1.7898 |
| 4 | L_j5 | −1.4217 | 1.5633 |
| 5 | L_j6 | −6.2323 | 1.2372 |
| 6 | L_grip | 0.0000 | 1.0000 |
| 7 | R_j1 | −6.3166 | 1.1083 |
| 8 | R_j2 | −0.5743 | 2.9860 |
| 9 | R_j3 | −0.0000 | 2.9049 |
| 10 | R_j4 | −1.9476 | 1.9531 |
| 11 | R_j5 | −2.1286 | 1.1588 |
| 12 | R_j6 | −6.2695 | 3.1658 |
| 13 | R_grip | 0.0000 | 1.0000 |

Values near `±2π` are joint-angle wraparound, not travel: the wrist joints are continuous and
the planner emits `θ` and `θ − 2π` interchangeably. **Unwrap before computing statistics**, or
a per-dimension standardisation will be dominated by the wrap and the decoder will chase a
bimodal target.

---

## 2. Paths

| what | where |
|---|---|
| conda env | `/lustre/fsw/portfolios/edgeai/users/chrislin/envs/loom-robotwin` |
| env activation shim | `…/envs/loom-robotwin/robotwin_env.sh` (sets `PYTHONPATH` + Vulkan) |
| RoboTwin checkout | `/lustre/fsw/portfolios/edgeai/users/chrislin/projects/loom-deps/RoboTwin` |
| assets (objects/textures/embodiments) | `/lustre/fsw/portfolios/edgeai/users/chrislin/datasets/loom/robotwin/assets` |
| asset zips (kept for re-extraction) | `…/datasets/loom/robotwin/assets_zip` |
| demonstrations | `…/datasets/loom/robotwin/data/demo_clean/<task>/aloha_agilex/data/episode_*.hdf5` |
| demonstration zips | `…/datasets/loom/robotwin/data_archives` |
| install logs | `/lustre/fsw/portfolios/edgeai/users/chrislin/envs/loom-robotwin-logs` |
| smoke-test evidence | `/lustre/fsw/portfolios/edgeai/users/chrislin/projects/loom-deps/_probe/{smoke_output.txt,smoke.json}` |

Disk footprint: env 11 GB, assets 16 GB (+14 GB of zips), demonstrations 33 GB (+22 GB of
zips) — **84 GB** under the dataset path. The `*_zip` / `*_archives` directories are kept so
extraction is repeatable without re-downloading; delete them to reclaim 36 GB.

`RoboTwin/assets/{embodiments,objects,background_texture}` and `RoboTwin/data` are **symlinks**
into the dataset volume. RoboTwin resolves everything through
`envs/_GLOBAL_CONFIGS.ASSETS_PATH`, which is hard-wired to `<checkout>/assets/`, so the
symlinks are load-bearing — do not replace them with copies or the curobo YAMLs
(which store *absolute* expanded paths) will disagree with the loader.

---

## 3. Reproducing the environment

```bash
# From a LOGIN node (compute nodes have no outbound network).
cd <loom repo>
bash scripts/setup_robotwin.sh                 # all stages, ~2 h wall
bash scripts/setup_robotwin.sh --list          # stage names
bash scripts/setup_robotwin.sh curobo assets   # re-run selected stages
```

Rough stage costs (Lustre metadata throughput, not CPU, is the bottleneck):
`env` ~15 min · `deps` ~25 min (mostly `azure`) · `torch` ~5 min · `curobo` ~50 min
(dependency resolve on the login node + a ~25 min CUDA compile that the script `srun`s onto a
compute node) · `assets` ~20 min for 14 GB of zips. The stage that needs SLURM will queue.

Every stage writes a stamp to `$ENV_PREFIX/.loom-stamps/<stage>`; re-running is a no-op.
Delete a stamp to force that stage.

### Versions actually landed

| package | version | note |
|---|---|---|
| python | 3.10.20 | RoboTwin requires 3.10; SAPIEN 3.0.0b1 ships cp310 wheels only |
| sapien | **3.0.0b1** | pinned by `scripts/requirements.txt`. 3.0.3 still exposes the `sapien.Engine()` / `SapienRenderer()` API that `_base_task.setup_scene()` uses (checked), so it may work — but it is untested against RoboTwin 2.0 and nothing here needs it. Stay on the pin |
| torch | **2.6.0+cu124** | cluster driver is CUDA 12.2; a default wheel is `+cu13x`, imports fine, and reports `cuda.is_available() == False` while holding 8 A100s |
| numpy | **1.26.4** | RoboTwin's native extensions need the NumPy 1.x ABI. `torchvision` will happily drag 2.x back in — the setup script re-pins after every torch install |
| mplib | 0.2.1 | patched, see below |
| gymnasium | 0.29.1 | |
| curobo | v0.7.8 | built from source against the in-env nvcc (`cuda-nvcc` 12.4 from conda-forge), `sm_80` |
| warp-lang | 1.12.0 | curobo resolves 1.16; RoboTwin pins it back to 1.12.0 |
| scipy | 1.15.3 | RoboTwin pins `1.10.1`, curobo overrides it. Upstream `_install.sh` accepts the same drift |
| setuptools | 69.5.1 | `sapien/__init__.py` does `import pkg_resources`; conda-forge python 3.10 ships no setuptools, so `import sapien` dies without this |
| ninja | 1.13.0 | needed by curobo's torch-JIT fallback path |
| RoboTwin | `266f3aadf505a4f7fe9af0faa41a20f5f47cd123` (main, 2026-08-11) | RoboTwin 2.0. The 1.0 lineage lives on the `RoboTwin-1.0` / `early_version` branches — PLAN §8's table is drawn from 2.0 |

### Deviations from RoboTwin's own `scripts/_install.sh`

1. **torch** — upstream pins `torch==2.4.1` from PyPI. We strip torch/torchvision from
   `requirements.txt` and install `2.6.0` from `https://download.pytorch.org/whl/cu124`.
2. **pytorch3d is skipped.** It is only imported by `envs/camera/camera.py` for
   farthest-point-sampling of point clouds, and `demo_clean.yml` sets `pointcloud: false`.
   RoboTwin's own docs say a failed pytorch3d install does not affect the project if you are
   not using 3D data. LOOM is RGB-only (PLAN §9: no pixel decoding, no point clouds).
3. **XPolicyLab is not installed.** It is the policy-serving/eval stack for third-party
   baselines. LOOM's eval harness (`loom/eval/robotwin.py`, Team F) drives
   `Base_Task.take_action()` directly through `contracts.Policy`, so the submodule is dead
   weight. Everything RoboTwin's env needs is in `envs/`.
4. **`azure==4.0.0` is in `requirements.txt`** (for RoboTwin's LLM-based description
   generation) and drags in ~200 `azure-mgmt-*` packages. It is installed for fidelity but
   costs ~20 min on Lustre; set `ROBOTWIN_SKIP_AZURE=1` if you do not need description gen.

### Two source patches, both prescribed upstream and applied idempotently

Originals are saved as `*.loom-orig`.

* `sapien/wrapper/urdf_loader.py` — open URDF/SRDF with `encoding="utf-8"`. Without it the
  RoboTwin URDFs (which contain non-ASCII) fail to parse under a non-UTF-8 locale.
* `mplib/planner.py` — drop `or collide` from the screw-plan guard, so a path that merely
  touches the collision margin is not rejected outright. Without it a large fraction of
  `move_to_pose` calls return `"screw plan failed"` and the expert never completes.

### curobo is **not** optional

`assets/embodiments/aloha-agilex/config.yml` sets `planner: "curobo"`, and
`envs/robot/robot.py::set_planner()` constructs a `CuroboPlanner` unconditionally. Worse, the
import is at module scope: `envs/robot/robot.py:15` does `from .planner import CuroboPlanner`,
so if curobo is broken you cannot even `import envs.turn_switch` — it fails with
`ImportError: cannot import name 'CuroboPlanner'`, which hides the real cause. Always read the
traceback printed just above that line.

There is no nvcc on this cluster and no root, so the setup script installs `cuda-nvcc=12.4`
plus the CUDA dev libs from conda-forge **into the env**. Two things then bite:

1. **conda-forge puts CUDA headers in `$PREFIX/targets/x86_64-linux/include`, not
   `$PREFIX/include`.** `nvcc` resolves them relative to its own path, but the host `gcc`
   pass that compiles torch's C++ glue does not, and fails with
   `c10/cuda/CUDAStream.h:3:10: fatal error: cuda_runtime_api.h: No such file or directory`.
   Fix: `CPATH` / `LIBRARY_PATH`, which plain gcc honours.

2. **The login and compute nodes run different distros.** See below — this one is general and
   bites any team that compiles a C++/CUDA extension.

### ⚠ The login node and the compute nodes have different glibc

| | OS | glibc | gcc |
|---|---|---|---|
| login node | Ubuntu 22.04.3 | **2.35** | 11.4.0 |
| compute nodes | Ubuntu 20.04.5 | **2.31** | 9.4.0 |

Anything compiled on the login node with the system toolchain links `GLIBC_2.32+` symbols and
**cannot be loaded on any compute node**:

```
ImportError: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.32' not found
  (required by .../curobo/src/curobo/curobolib/kinematics_fused_cu.cpython-310-x86_64-linux-gnu.so)
```

This is invisible until you actually run on a GPU node, because the login node imports the
same `.so` happily. PyPI manylinux wheels (torch, sapien, mplib) are unaffected — they target
an old glibc on purpose. Only locally compiled extensions break.

So the curobo stage is **two-phase**, and `scripts/setup_robotwin.sh` does both:

* **Phase 1, login node** (needs outbound network): `pip install -e envs/curobo
  --no-build-isolation` resolves and installs the pure-python dependency tree.
* **Phase 2, compute node** (no network needed, right libc): the script `srun`s
  `python setup.py build_ext --inplace` to recompile the extensions in place.

`ninja` is also installed, because curobo falls back to torch's JIT `load()` when a prebuilt
`.so` will not import, and that path fails with `RuntimeError: Ninja is required to load C++
extensions` — which is a *symptom*, not the cause. The cause is always the failed import
reported one traceback higher.

---

## 4. `env_fps` — the load-bearing number, and the trap

**Use `env_fps = 250/15 = 16.6667 Hz`. Do not use 15.0.**

RoboTwin's simulation runs at a **250 Hz physics timestep** (`envs/_base_task.py:223`,
`self.scene.set_timestep(kwargs.get("timestep", 1 / 250))`) and records a demonstration frame
every `save_freq` physics steps. `demo_clean.yml` and `demo_randomized.yml` both set
`save_freq: 15`. So consecutive actions in a released trajectory are

```
15 physics steps x (1/250 s) = 0.06 s  ->  16.6667 Hz
```

**The trap:** every released HDF5 contains `/additional_info/frequency = 15`, which reads like
15 Hz. It is not. `scripts/process_data_xpolicylab.py:409` says

```python
frequency = args.frequency or int(task_cfg.get("save_freq") or 15)
```

i.e. the field is literally `save_freq` — a **physics-step decimation factor** — written into
a slot labelled "frequency". Taking it as Hz makes every resampled action **11 % too slow**.
PLAN §9 requires all decoded actions be resampled from canonical 30 Hz back to the env rate;
`contracts.env_steps_per_segment` gives

```
env_steps_per_segment(16.6667) = 8 * 16.6667 / 30 = 4.444   <- correct
env_steps_per_segment(15.0)    = 8 * 15.0    / 30 = 4.000   <- 11% wrong, and suspiciously tidy
```

The tidy number is the wrong one. `contracts.env_steps_per_segment` is deliberately
fractional and its docstring already warns to carry the remainder in an accumulator rather
than round per segment.

Separately, note that at **evaluation** time RoboTwin's control loop is not clocked. Each
`Base_Task.take_action(a)` TOPP-interpolates from the current joint state to the commanded
target and executes however many 250 Hz physics steps that takes (see the measured
steps-per-action in §8). So there is no wall-clock rate the environment enforces; what
16.6667 Hz fixes is the **spacing of consecutive absolute joint targets**, which is exactly
what the resampler must reproduce. The episode budget in
`env_cfg/task_config/_eval_step_limit.yml` is counted in `take_action` calls, not seconds.

---

## 5. Actions are absolute joint targets, not deltas

Three independent confirmations:

1. **Code.** `Base_Task.take_action` builds `left_path = np.vstack((left_current_qpos,
   left_arm_actions))` and hands it to `mplib`'s TOPP with `1/250` — the action is the
   *endpoint*. `Robot.set_arm_joints` then calls `joint.set_drive_target(target_position[j])`.
2. **Recording.** `Robot.get_left_arm_jointState()` returns `joint.get_drive_target()`, i.e.
   the commanded target, not the measured `qpos`.
3. **Data.** In every released episode, `state[t+1] == action[t]` exactly:

   ```
   |action[t] - state[t]  | mean = 0.008239      (one step of motion)
   |action[t] - state[t+1]| mean = 0.000000      (exact)
   ```

The gripper channels are already normalised to `[0, 1]` and clipped by `Robot.set_gripper`, so
they need no per-dataset rescaling — but they are **not** the same convention as LIBERO, whose
gripper is a `[-1, 1]` delta. Do not share a normaliser between the two bodies.

An `action_type="ee"` mode also exists (`take_action(..., action_type='ee')`), where the
vector is `7 + 1 + 7 + 1 = 16` — position + quaternion + gripper per arm. LOOM uses `qpos`;
the joint-space path is what the PLAN §8 baseline table was produced with.

---

## 6. The four PLAN §8 tasks

Task ids are the module *and* class names under `RoboTwin/envs/`; the registry is
`importlib.import_module(f"envs.{task_name}")` then `getattr(module, task_name)`.

| PLAN §8 column | task id | file | eval step limit |
|---|---|---|---|
| hanging mug | `hanging_mug` | `envs/hanging_mug.py` | 900 |
| turn switch | `turn_switch` | `envs/turn_switch.py` | 400 |
| place can basket | `place_can_basket` | `envs/place_can_basket.py` | 700 |
| handover block | `handover_block` | `envs/handover_block.py` | 800 |

All 50 task ids are listed in `env_cfg/eval/all_tasks.yml`. Step limits are in
`env_cfg/task_config/_eval_step_limit.yml` and are loaded into `self.step_lim` **only when
`args["eval_mode"] = True`**; otherwise `step_lim` stays `None` and `take_action` never
terminates on budget.

### RoboTwin's own eval protocol

`scripts/eval_policy_xpolicylab.py` defaults to `test_num = 100` episodes per task, with
seeds drawn from `st_seed = 100000 * (1 + seed)`. Before each trial it runs an **expert
check**: if the scripted expert cannot solve the sampled scene, the seed is *skipped* rather
than counted as a policy failure (`reason: "expert_failed"`), and `setup_demo` raising
`UnStableError` skips the seed too. Success rate is `suc_num / test_num`.

PLAN §4F's rule — replicate the protocol of the source paper, do not invent an episode count
— applies here as well: `100 episodes/task` with expert-checked seeds is RoboTwin's own
default and is what the leaderboard uses. State the seed and episode count with any number
reported into PLAN §8.

---

## 7. Clean vs randomized — get this right or the columns are incomparable

PLAN §8's RoboTwin table has a `clean` and a `rand` column. **R0-B is scored on clean.** The
baseline numbers in the `rand` column came from the randomized configuration.

Selection is by **task config file**, `env_cfg/task_config/{demo_clean,demo_randomized}.yml`,
passed as `--task-config` / `args["task_config"]`. The two files differ in exactly two blocks:

| key | `demo_clean` | `demo_randomized` |
|---|---|---|
| `domain_randomization.random_background` | `false` | `true` |
| `domain_randomization.cluttered_table` | `false` | `true` |
| `domain_randomization.clean_background_rate` | `1` | `0.02` |
| `domain_randomization.random_light` | `false` | `true` |
| `domain_randomization.crazy_random_light_rate` | `0` | `0.02` |
| `domain_randomization.random_table_height` | `0` | `0.03` |
| `domain_randomization.random_head_camera_dis` | `0` | `0` |
| `eval_instruction` | `seen` | `unseen` |

Everything else — embodiment, cameras, `save_freq`, `episode_num` — is identical.

Two consequences that are easy to miss:

* **`eval_instruction` is part of the randomization.** `description/task_instruction/<task>.json`
  holds 50 `seen` templates and 10 held-out `unseen` templates. The clean setting evaluates on
  language the policy trained on; the randomized setting swaps in unseen phrasings. Since LOOM
  conditions on language (`ℓ` feeds both `E` and `π_c`), evaluating clean with `unseen`
  instructions silently measures language generalisation and will read low against the `clean`
  baseline column.
* The randomized config also changes the **training** distribution when used for data
  collection, not just eval. The two are separate archives on HuggingFace (§9).

---

## 8. Vulkan / rendering — what actually works here

This was expected to be the hard part. It is solved, and no root was needed.

**Findings on the GPU compute nodes** (A100-SXM4-80GB, driver 535.129.03):

* `/etc/vulkan/icd.d/nvidia_icd.json` **exists** and points at `libGLX_nvidia.so.0`.
* `libGLX_nvidia.so.0` resolves via `ldconfig` to
  `/cm/local/apps/cuda/libs/current/lib64/libGLX_nvidia.so.0`. The driver also ships
  `libnvidia-rtcore.so` and `libnvidia-glvkspirv.so`.
* **The login node has neither** — no `/etc/vulkan`, no `/usr/share/vulkan`, no
  `libvulkan.so.1`. Every rendering check must go through `srun`.
* `/usr/share/vulkan/icd.d/` on the compute nodes holds `intel_icd`, `radeon_icd` and
  **`lvp_icd`** — Mesa's llvmpipe *software* rasteriser. If the loader picks lvp, SAPIEN
  renders on the CPU: no crash, plausible non-black images, ~1 fps, and not the GPU path the
  benchmark assumes. **This is why the ICD is pinned explicitly rather than left to the
  loader's search order.**

The incantation (written by the setup script into `$ENV_PREFIX/robotwin_env.sh`):

```bash
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export VK_DRIVER_FILES=$VK_ICD_FILENAMES     # loader >= 1.3.207 renamed the variable
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
unset DISPLAY
```

`VK_ICD_FILENAMES` and `VK_DRIVER_FILES` are both set because SAPIEN bundles its own
`libvulkan` (it warns `Failed to find system libvulkan. Fallback to SAPIEN builtin
libvulkan.`) and the bundled loader version varies between SAPIEN builds. If a future node
image drops `/etc/vulkan/icd.d/nvidia_icd.json`, `robotwin_env.sh` synthesises an equivalent
ICD JSON inside the env — nothing here requires root.

### Ray tracing works on A100

`envs/_base_task.py:214-217` unconditionally selects SAPIEN's **ray-tracing** shader:

```python
sapien.render.set_camera_shader_dir("rt")
sapien.render.set_ray_tracing_samples_per_pixel(32)
sapien.render.set_ray_tracing_path_depth(8)
sapien.render.set_ray_tracing_denoiser("oidn")
```

A100 (GA100) has **no RT cores**, so this was the main open risk. It is not a problem:
NVIDIA's Vulkan driver exposes ray query on A100 anyway, and SAPIEN reports it. From the
verified smoke run (§10):

```
  sapien device summary:
    GPU: NVIDIA A100-SXM4-80GB
      Supported: 1
      Present:   0
      cudaId:    0
      rayTrace:  1          <-- ray tracing available
      cudaMode   0
```

An isolated A/B of the two shaders on a GPU node (sapien 3.0.3, a scripted box scene) also
renders non-black either way:

```
host: batch-block5-00113   VK_ICD_FILENAMES = /etc/vulkan/icd.d/nvidia_icd.json
--- shader_dir='default' ---  image (240,320,3) var=664.53 mean=206.05  -> OK (non-black)
--- shader_dir='rt'      ---  image (240,320,3) var=594.15 mean=205.98  -> OK (non-black)
```

`Present: 0` is expected and harmless — it means no swapchain/display surface, i.e. headless.
Offscreen camera rendering does not need one.

Throughput with `rt` at 32 spp is fine for eval: 59–90 ms per `take_action` including four
320×240 renders per observation, i.e. 11–17 policy actions/s on one A100 (§10).

No patch to `_base_task.py` is needed, and none should be applied: changing the shader would
change the observation distribution relative to the published baselines.

---

## 9. Assets and data

### Assets — 16 GB extracted, from `TianxingChen/RoboTwin2.0` (HF dataset repo)

| pack | extracted size | contents |
|---|---|---|
| `background_texture` | 9.9 GB | wall/table texture library, split `seen/` and `unseen/` |
| `objects` | 4.4 GB | RoboTwin-OD, 129 object categories with per-instance meshes + grasp annotations |
| `embodiments` | 901 MB | `aloha-agilex`, `ARX-X5`, `franka-panda`, `piper`, `ur5-wsg` (URDF, SRDF, meshes, curobo configs) |

After extraction, `scripts/update_embodiment_config_path.py` expands `${ASSETS_PATH}` in every
`*_tmp.yml` → `*.yml` (the curobo kinematics configs store absolute paths). **Re-run it if you
ever move the assets directory.**

### Demonstrations — 33 GB extracted

The full R0-B clean training set is downloaded: **50 tasks × 50 episodes = 2 500 episodes**,
aloha-agilex, `demo_clean`.

```
datasets/loom/robotwin/data/demo_clean/<task>/aloha_agilex/
├── data/episode_0000000.hdf5 … episode_0000049.hdf5
├── instruction/episode_*.json     # per-episode seen/unseen instruction sets
├── video/                          # head-camera mp4 preview
├── scene_info.json                 # per-episode object ids and texture choices
└── seed.txt                        # whitespace-separated seeds; episode i used seed[i]
```

`seed.txt` makes episodes exactly reproducible: `setup_demo(now_ep_num=i, seed=seed[i], ...)`
rebuilds the identical scene, so a recorded action sequence can be replayed through
`take_action` for end-to-end validation.

Per-episode HDF5 layout. T is the episode length; over the four PLAN §8 tasks
(200 episodes) it averages 242 frames with a range of 90–373, so at 16.6667 Hz a typical
episode is ~15 s of simulated time:

```
/action/joint_states            (T, 14) float32   <- the 14-vector, absolute targets
/action/{left,right}_arm_joint_states   (T, 6)
/action/{left,right}_ee_joint_states    (T, 1)    gripper, [0,1]
/action/{left,right}_ee_poses           (T, 7)    xyz + quat
/state/…                        same tree, state[t+1] == action[t]
/vision/cam_head/colors         (T,)     JPEG-encoded bytes, decode to (240,320,3) uint8
/vision/cam_head/{intrinsic_matrix,extrinsics_matrix,shape}
/vision/cam_left_wrist/…   /vision/cam_right_wrist/…   /vision/cam_third_view/…
/instruction                    ()       the instruction used for this episode
/instructions                   (100,)   the 100-template pool (language_num: 100)
/additional_info/frequency      ()       int32 = 15  <-- decimation factor, NOT Hz (§4)
/data_format_version            ()       b'v1.0'
/pointclouds                    (T, 0)   empty; demo_clean sets pointcloud: false
```

Note `/vision/*/colors` is **JPEG bytes**, not raw arrays — decode with PIL/cv2. This is why
30 GB holds 605 k frames × 4 views.

### Other archives available on the same HF repo

`dataset/<task>/` contains, per task:

| archive | what | total across 50 tasks |
|---|---|---|
| `demo_clean.zip` | aloha-agilex, clean, 50 episodes — **downloaded** | 23 GB |
| `aloha-agilex_clean_50.zip` | same content, explicit naming | 23.8 GB |
| `aloha-agilex_randomized_500.zip` | randomized, **500** episodes/task | 391 GB |
| `{arx-x5,franka,piper,ur5}_{clean_50,randomized_500}.zip` | cross-embodiment | — |

Download more with RoboTwin's own script (it normalises the layout and is resumable):

```bash
source <env>/robotwin_env.sh
export ROBOTWIN_DATA_ROOT=$ROBOTWIN_DATA/data
export HF_ARCHIVE_CACHE=$ROBOTWIN_DATA/data_archives
export HF_ARCHIVE_NAME=aloha-agilex_randomized_500.zip     # default: demo_clean.zip
cd $ROBOTWIN_ROOT && bash scripts/download_xpolicylab_data.sh hanging_mug turn_switch
```

The `rand` column of PLAN §8 needs `aloha-agilex_randomized_500.zip`; at 391 GB for all 50
tasks, pull only the tasks you will report.

---

## 10. Running the smoke test

Rendering **cannot** be validated on the login node. Always go through `srun`:

```bash
srun -A edgeai_tao-ptm_image-foundation-model-clip \
     -p interactive_singlenode,polar4,polar3,batch_singlenode \
     -t 00:40:00 --gpus-per-node=1 -N1 \
     bash -c 'source /lustre/fsw/portfolios/edgeai/users/chrislin/envs/loom-robotwin/robotwin_env.sh && \
              /lustre/fsw/portfolios/edgeai/users/chrislin/envs/loom-robotwin/bin/python \
              <loom>/scripts/smoke_robotwin.py --out /tmp/robotwin_smoke.json'
```

It checks, and exits non-zero on any failure:

1. `nvidia-smi` sees a GPU, `torch.cuda.is_available()`, the pinned Vulkan ICD, and that
   SAPIEN enumerates a **hardware** (non-llvmpipe) device.
2. All four PLAN §8 tasks build, reset and step.
3. Every camera stream returns `(240, 320, 3)` and has pixel variance above `5.0` —
   **the all-black check**. A headless GPU with a broken ICD returns a perfectly valid,
   perfectly black frame; every downstream number stays "real" and the score is zero.
   Re-checked after 20 steps, since some failures only appear once the render loop runs.
4. `dof`, the arm/gripper split, and the per-dimension bounds.
5. Success/termination plumbing: the scripted expert runs a full episode, `check_success()`
   returns `True`, and `take_action` becomes a no-op once `eval_success` latches.
6. The control frequency, **measured**: physx steps and recorded frames are counted during one
   expert episode and `env_fps = 1 / (dt · steps_per_frame)` is reported next to the
   config-derived value.
7. A recorded demonstration replayed action-for-action through `take_action` — informational,
   but the strongest single confirmation that the contract in §1 is right.

### Verified run — 81/81 checks, `batch-block5-04009` (A100-SXM4-80GB, driver 535.129.03)

```
RoboTwin root      .../loom-deps/RoboTwin
RoboTwin commit    266f3aadf505a4f7fe9af0faa41a20f5f47cd123
task config        demo_clean

==============================================================================
[1] host and renderer
==============================================================================
  hostname          batch-block5-04009
  python            3.10.20
  [PASS] nvidia-smi reports a GPU   NVIDIA A100-SXM4-80GB, 535.129.03, 81920 MiB
  [PASS] Vulkan ICD pinned via VK_ICD_FILENAMES / VK_DRIVER_FILES   /etc/vulkan/icd.d/nvidia_icd.json
  ICD contents      {'library_path': 'libGLX_nvidia.so.0', 'api_version': '1.3.242'}
  torch             2.6.0+cu124
  [PASS] torch.cuda.is_available()   devices=1
  sapien            3.0.0b1
  sapien device summary:
    GPU: NVIDIA A100-SXM4-80GB
      Supported: 1
      Present:   0
      cudaId:    0
      rayTrace:  1
      cudaMode   0
  [PASS] SAPIEN can enumerate Vulkan devices
  [PASS] SAPIEN did NOT fall through to the llvmpipe software rasteriser

==============================================================================
[2-4,6] task: hanging_mug   (config=demo_clean, seed=0)
==============================================================================
  [PASS] hanging_mug: build + reset   27.3s
  [PASS] hanging_mug: observation has camera streams   ['front_camera', 'head_camera', 'left_camera', 'right_camera']
  [PASS] hanging_mug: front_camera resolution   (240, 320, 3) expected (240, 320, 3)
  [PASS] hanging_mug: front_camera render is NOT all-black   var=  2821.75 mean=214.72 min=26 max=255
  [PASS] hanging_mug: head_camera resolution   (240, 320, 3) expected (240, 320, 3)
  [PASS] hanging_mug: head_camera render is NOT all-black   var=  2045.98 mean=232.73 min=10 max=255
  [PASS] hanging_mug: left_camera resolution   (240, 320, 3) expected (240, 320, 3)
  [PASS] hanging_mug: left_camera render is NOT all-black   var=   742.74 mean=234.57 min=40 max=255
  [PASS] hanging_mug: right_camera resolution   (240, 320, 3) expected (240, 320, 3)
  [PASS] hanging_mug: right_camera render is NOT all-black   var=  1368.64 mean=222.46 min=41 max=255
  [PASS] hanging_mug: action/proprio width   dof=14 = (6+1)+(6+1), vector(14,)
    action_low  = [-10.0 x6, 0.0, -10.0 x6, 0.0]
    action_high = [ 10.0 x6, 1.0,  10.0 x6, 1.0]
  [PASS] hanging_mug: gripper dims are normalised to [0, 1]   idx 6 and 13
  [PASS] hanging_mug: stepped 20 actions   take_action_cnt=20
  [PASS] hanging_mug: take_action_cnt tracks steps   20 == 20
  [PASS] hanging_mug: eval_success flag present   =False
  [PASS] hanging_mug: eval step limit loaded   step_lim=900
    physics timestep        0.004000 s  (250.0 Hz)
    physx steps / action    mean   97.8  min 80  max 124
    wall time / action         72.5 ms  (13.79 action/s incl. rendering)
  [PASS] hanging_mug: render still non-black after 20 steps   var=2816.12

  ... turn_switch / place_can_basket / handover_block identical in structure:
      4 streams, all (240,320,3), all non-black, dof=14, gripper [0,1].
      step_lim 400 / 700 / 800. Lowest observed pixel variance anywhere: 298.55
      (turn_switch front_camera; the all-black threshold is 5.0).
      wall time / action: 59.1 / 89.8 / 71.3 ms.

==============================================================================
[6] measured control frequency (expert path, turn_switch)
==============================================================================
  task config save_freq   15
  seeds from seed.txt     [0, 1, 2, 3]
  seed 0: expert succeeded
  physics timestep        0.004000 s  (250.0 Hz)
  physx steps (expert)    1277
  recorded frames         94
  median steps / frame    15.0      <- this is save_freq
  naive steps/frame       13.73     (biased low: 2 extra frames per move())
  env_fps  MEASURED       16.6667 Hz  (= 1/(0.004000 * 15))
  env_fps  from save_freq 16.6667 Hz  (= 1/(0.004000 * 15))
  HDF5 'frequency' field  15  <- decimation factor, NOT Hz
  [PASS] measured median steps/frame equals task-config save_freq   15.0 vs 15
  [PASS] measured env_fps == 250/15 == 16.6667 Hz   16.6667 Hz

==============================================================================
[5] success / termination plumbing (turn_switch)
==============================================================================
  [PASS] expert motion plan succeeded   =True
  [PASS] check_success() returns True after a full expert episode   =True
  step_lim                400
  eval_success (pre)      False
  [PASS] take_action() is a no-op once eval_success latches   cnt 0 -> 0

==============================================================================
[7] recorded-demo replay through take_action (turn_switch, ep 0)
==============================================================================
  seed                    0   (seed.txt[0])
  instruction             b'Directly interact with the plastic switch with angled edges'
  actions                 (92, 14)
  HDF5 additional_info/frequency = 15   (this is save_freq, NOT Hz -- see §4)
  max|action[t] - state[t+1]| = 0.000e+00   (0 => actions are absolute joint targets, not deltas)
  [PASS] actions are ABSOLUTE joint targets (state[t+1] == action[t])   max abs diff 0.000e+00
  replayed                63/92 actions in 3.6s (17.32 action/s)
  physx steps             5601  (88.9 per action)
  eval_success            True
  check_success()         True
  REPLAY REPRODUCED THE DEMONSTRATION

==============================================================================
summary
==============================================================================
  [PASS] dof is identical across all four tasks   {14}
  [PASS] n_views is identical across all four tasks   {4}
  hanging_mug          dof= 14 n_views=4  render_ok=True
  turn_switch          dof= 14 n_views=4  render_ok=True
  place_can_basket     dof= 14 n_views=4  render_ok=True
  handover_block       dof= 14 n_views=4  render_ok=True
  measured env_fps     16.6667 Hz  (physics 250 Hz / 15)
  demo replay          seed=0 actions=63 -> eval_success=True

  81/81 checks passed

SMOKE TEST PASSED
```

Three things worth reading off that output:

* **`rayTrace: 1`** — the A100 exposes Vulkan ray query despite having no RT cores, so
  RoboTwin's unconditional `set_camera_shader_dir("rt")` at 32 spp with the OIDN denoiser runs
  on the GPU as intended. No patch, no deviation from the published observation distribution.
* **The expert episode produced 94 recorded frames at a median gap of exactly 15 physx
  steps**, against 92 frames in the released `episode_0000000.hdf5` for the same seed. The
  measured rate and the shipped data agree.
* **The replay reached success in 63 of 92 recorded actions.** It terminates early because
  `check_success()` latches mid-trajectory — the recorded episode keeps going to a rest pose.
  ~89 physx steps per `take_action` (vs 15 during collection) is expected: at eval RoboTwin
  re-plans each step with TOPP instead of replaying the dense trajectory, which is exactly why
  the wall-clock control rate is emergent and `env_fps` describes target *spacing*, not
  wall-clock (§4).

---

## 11. Gotchas

* **`import envs.<task>` requires a GPU.** curobo's `motion_gen.py` evaluates
  `Pose.from_list([0, 0, -0.15, 1, 0, 0, 0])` as a *class-body default argument*, which
  allocates a CUDA tensor at import time. On the login node this raises `RuntimeError: Found
  no NVIDIA driver on your system`, surfacing as the same misleading
  `ImportError: cannot import name 'CuroboPlanner'`. Any RoboTwin adapter test must therefore
  be marked `@pytest.mark.gpu`; it can never pass in the CPU-only suite.
* **`cwd` must be the RoboTwin checkout.** `_embodiment_config.yml` stores relative paths
  (`./assets/embodiments/aloha-agilex/`). `smoke_robotwin.py` does the `os.chdir` for you.
* **`eval_mode=True` or `step_lim` is `None`.** Without it, episodes never hit the budget.
* **`render_freq > 0` opens a GUI viewer** and will fail headless. `demo_clean.yml` sets it to
  `0`; keep it there.
* **`setup_demo()` can raise `UnStableError`** for a seed where the sampled objects settle
  badly. RoboTwin's own eval skips such seeds. The eval harness must catch it and resample,
  not count it as a failure.
* **`_take_picture` writes to disk when `save_data=True`.** For eval, keep `save_data=False`.
* `envs/robot/robot.py` starts **two extra processes** per env when left and right arms have
  different curobo configs. For aloha-agilex they are identical, so planners are in-process —
  but budget for it if you ever run a mixed embodiment.
* RoboTwin re-seeds global RNG in `_init_task_env_` (`np.random.seed`, `torch.manual_seed`).
  It will stomp on LOOM's `(seed, global_step, rank)` discipline; re-seed after env
  construction in the eval runner.
