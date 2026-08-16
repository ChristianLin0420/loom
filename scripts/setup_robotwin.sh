#!/usr/bin/env bash
# ============================================================================
# LOOM — RoboTwin 2.0 environment setup  (Team H)
#
# Builds a self-contained conda env for the RoboTwin 2.0 benchmark, which is
# the R0-B decision gate (PLAN.md §7).
#
#   env      : $ENV_PREFIX                (conda, python 3.10)
#   code     : $ROBOTWIN_DIR              (RoboTwin 2.0, `main` branch)
#   assets   : $DATA_DIR/assets           (symlinked into $ROBOTWIN_DIR/assets)
#   demos    : $DATA_DIR/data
#
# Idempotent: every stage writes a stamp under $STAMP_DIR and is skipped on
# re-run. Delete a stamp to force that stage to re-run.
#
#   bash scripts/setup_robotwin.sh              # all stages
#   bash scripts/setup_robotwin.sh env deps     # selected stages
#   bash scripts/setup_robotwin.sh --list       # show stage names
#
# NOTE: run this from a LOGIN node (needs outbound network). The rendering
# check must run on a GPU node -- see scripts/smoke_robotwin.py.
# ============================================================================
set -euo pipefail

USER_ROOT=/lustre/fsw/portfolios/edgeai/users/chrislin
ENV_PREFIX="${ENV_PREFIX:-$USER_ROOT/envs/loom-robotwin}"
ROBOTWIN_DIR="${ROBOTWIN_DIR:-$USER_ROOT/projects/loom-deps/RoboTwin}"
DATA_DIR="${DATA_DIR:-$USER_ROOT/datasets/loom/robotwin}"
CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$USER_ROOT/.conda-pkgs}"
export CONDA_PKGS_DIRS

# RoboTwin 2.0 has no release tags; pin the commit that this env was validated
# against. Override with ROBOTWIN_COMMIT=<sha> (or `main` to track HEAD).
ROBOTWIN_COMMIT="${ROBOTWIN_COMMIT:-266f3aadf505a4f7fe9af0faa41a20f5f47cd123}"
ROBOTWIN_REPO=https://github.com/RoboTwin-Platform/RoboTwin.git

# torch: the cluster driver is CUDA 12.2, a default wheel pulls +cu13x, imports
# fine, and reports cuda.is_available()==False while holding 8 A100s.
TORCH_VERSION="${TORCH_VERSION:-2.6.0}"
TORCH_INDEX=https://download.pytorch.org/whl/cu124
CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"   # A100

# A100 is compute capability 8.0 and does NOT have RT cores. RoboTwin's
# _base_task.setup_scene() unconditionally selects SAPIEN's ray-tracing shader.
# See docs/ENV_ROBOTWIN.md for the measured outcome.

STAMP_DIR="$ENV_PREFIX/.loom-stamps"
LOG_DIR="${LOG_DIR:-$USER_ROOT/envs/loom-robotwin-logs}"

PY="$ENV_PREFIX/bin/python"
PIP="$ENV_PREFIX/bin/pip"

ALL_STAGES="env robotwin deps torch patches curobo assets vulkan verify"

log()  { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn ]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[fail ]\033[0m %s\n' "$*" >&2; exit 1; }

stamped() { [ -f "$STAMP_DIR/$1" ]; }
stamp()   { mkdir -p "$STAMP_DIR"; date -Is > "$STAMP_DIR/$1"; }

# Recompile curobo's CUDA/C++ extensions in place. Must run where the built
# objects will be *used* -- see the glibc note in the curobo stage.
# Driven entirely by env vars so it can be shipped to a compute node with
# `declare -f`.
build_curobo_ext() {
  set -e
  echo "curobo-ext: host=$(hostname) glibc=$(ldd --version | head -1)"
  local T="$ENV_PREFIX/targets/x86_64-linux"
  export CUDA_HOME="$ENV_PREFIX" CUDA_PATH="$ENV_PREFIX"
  export CPATH="$T/include:${CPATH:-}"
  export LIBRARY_PATH="$T/lib:$T/lib/stubs:${LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH="$T/lib:${LD_LIBRARY_PATH:-}"
  export TORCH_CUDA_ARCH_LIST="$CUDA_ARCH_LIST"
  export PATH="$ENV_PREFIX/bin:$PATH"
  cd "$CUROBO_DIR"
  rm -f src/curobo/curobolib/*.so
  "$ENV_PREFIX/bin/python" setup.py build_ext --inplace 2>&1 | tail -4
  # `import torch` FIRST: the extensions link -lc10 -ltorch without an rpath
  # into torch/lib, so they resolve only once torch has loaded those objects
  # into the process. Importing them standalone gives a misleading
  # "ImportError: libc10.so: cannot open shared object file".
  "$ENV_PREFIX/bin/python" -c \
    'import torch; from curobo.wrap.reacher.motion_gen import MotionGen; print("curobo-ext: MotionGen import OK")'
}

if [ "${1:-}" = "--list" ]; then echo "$ALL_STAGES"; exit 0; fi
STAGES="${*:-$ALL_STAGES}"
want() { case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }

mkdir -p "$LOG_DIR" "$CONDA_PKGS_DIRS" "$DATA_DIR"

# ---------------------------------------------------------------- 1. conda env
if want env; then
  if stamped env && [ -x "$PY" ]; then
    log "env: already present at $ENV_PREFIX ($($PY -V 2>&1))"
  else
    log "env: creating conda env at $ENV_PREFIX (python 3.10, ~15 min on Lustre)"
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    # RoboTwin pins python 3.10; SAPIEN 3.0.0b1 ships cp310 wheels only.
    conda create -p "$ENV_PREFIX" -c conda-forge python=3.10 -y \
      2>&1 | tee "$LOG_DIR/01-env.log" | tail -5
    [ -x "$PY" ] || die "env: conda create did not produce $PY"
    stamp env
  fi
fi
[ -x "$PY" ] || die "no interpreter at $PY -- run the 'env' stage first"

# ------------------------------------------------------------- 2. RoboTwin src
if want robotwin; then
  if [ ! -d "$ROBOTWIN_DIR/.git" ]; then
    log "robotwin: cloning $ROBOTWIN_REPO"
    mkdir -p "$(dirname "$ROBOTWIN_DIR")"
    git clone "$ROBOTWIN_REPO" "$ROBOTWIN_DIR" 2>&1 | tail -3
  fi
  cd "$ROBOTWIN_DIR"
  if [ "$ROBOTWIN_COMMIT" != "main" ]; then
    if [ "$(git rev-parse HEAD)" != "$ROBOTWIN_COMMIT" ]; then
      log "robotwin: checking out pinned commit $ROBOTWIN_COMMIT"
      git fetch --depth 200 origin main 2>&1 | tail -2 || true
      git checkout -q "$ROBOTWIN_COMMIT"
    fi
  fi
  log "robotwin: HEAD = $(git rev-parse --short HEAD) ($(git log -1 --format=%ad --date=short))"
  stamp robotwin
fi

# ------------------------------------------------------------- 3. python deps
if want deps; then
  if stamped deps; then
    log "deps: already installed"
  else
    log "deps: installing scripts/requirements.txt (torch pin dropped -- see stage 'torch')"
    # RoboTwin pins torch==2.4.1; we install 2.6.0+cu124 separately for this
    # cluster, so strip torch/torchvision here to avoid a CPU wheel landing
    # first and being half-overwritten.
    _drop='^(torch|torchvision)([=<>]|$)'
    if [ "${ROBOTWIN_SKIP_AZURE:-0}" = "1" ]; then
      # azure==4.0.0 exists only for RoboTwin's LLM description generation and
      # drags in ~200 azure-mgmt-* distributions (~20 min on Lustre).
      warn "deps: ROBOTWIN_SKIP_AZURE=1 -- omitting azure* (description generation will not work)"
      _drop='^(torch|torchvision|azure)([=<>-]|$)'
    fi
    grep -vE "$_drop" "$ROBOTWIN_DIR/scripts/requirements.txt" \
      > "$LOG_DIR/requirements.no-torch.txt"
    # sapien/__init__.py does `import pkg_resources`, and python 3.10 from
    # conda-forge ships no setuptools (modern pip stopped vendoring it), so
    # `import sapien` dies with ModuleNotFoundError before anything else runs.
    # RoboTwin's own _install.sh pins this, but only at the very end.
    "$PIP" install --no-input "setuptools==69.5.1" 2>&1 | tail -2

    "$PIP" install --no-input -r "$LOG_DIR/requirements.no-torch.txt" \
      2>&1 | tee "$LOG_DIR/03-deps.log" | tail -5
    # RoboTwin's native extensions need the NumPy 1.x ABI; requirements.txt
    # already pins 1.26.4 but later installs can silently bump it.
    "$PIP" install --no-input "numpy==1.26.4" 2>&1 | tail -2
    stamp deps
  fi
fi

# ------------------------------------------------------------------- 4. torch
if want torch; then
  if stamped torch; then
    log "torch: already installed ($("$PY" -c 'import torch;print(torch.__version__)' 2>/dev/null))"
  else
    log "torch: installing torch==$TORCH_VERSION from $TORCH_INDEX"
    "$PIP" install --no-input --index-url "$TORCH_INDEX" \
      "torch==$TORCH_VERSION" torchvision 2>&1 | tee "$LOG_DIR/04-torch.log" | tail -3
    # torchvision may drag a newer numpy back in.
    "$PIP" install --no-input "numpy==1.26.4" 2>&1 | tail -1
    V="$("$PY" -c 'import torch;print(torch.__version__)')"
    case "$V" in
      *cu124*) log "torch: $V  (cu124 OK)" ;;
      *) die "torch: got '$V', expected a +cu124 build. A cu13x wheel reports cuda.is_available()==False on this driver." ;;
    esac
    stamp torch
  fi
fi

# ------------------------------------------------------------------ 5. patches
# Both are prescribed by RoboTwin's scripts/_install.sh. Applied idempotently.
if want patches; then
  SAPIEN_DIR="$("$PIP" show sapien 2>/dev/null | awk '/^Location:/{print $2}')/sapien"
  MPLIB_DIR="$("$PIP" show mplib  2>/dev/null | awk '/^Location:/{print $2}')/mplib"

  URDF_LOADER="$SAPIEN_DIR/wrapper/urdf_loader.py"
  if [ -f "$URDF_LOADER" ]; then
    if grep -q 'open(urdf_file, "r")' "$URDF_LOADER"; then
      log "patches: sapien/wrapper/urdf_loader.py -> utf-8 encoding"
      cp -n "$URDF_LOADER" "$URDF_LOADER.loom-orig"
      sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' "$URDF_LOADER"
    else
      log "patches: sapien urdf_loader already patched"
    fi
  else
    warn "patches: $URDF_LOADER not found (sapien not installed?)"
  fi

  PLANNER="$MPLIB_DIR/planner.py"
  if [ -f "$PLANNER" ]; then
    if grep -q 'or collide or not within_joint_limit' "$PLANNER"; then
      log "patches: mplib/planner.py -> drop 'or collide' from the screw-plan guard"
      cp -n "$PLANNER" "$PLANNER.loom-orig"
      sed -i -E 's/(if np\.linalg\.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' "$PLANNER"
    else
      log "patches: mplib planner already patched"
    fi
  else
    warn "patches: $PLANNER not found (mplib not installed?)"
  fi
  stamp patches
fi

# ------------------------------------------------------------------- 6. curobo
# RoboTwin's aloha-agilex config sets planner: "curobo"; envs/robot/robot.py
# constructs a CuroboPlanner unconditionally in set_planner(). It is NOT
# optional -- without it env setup raises NameError.
if want curobo; then
  if stamped curobo; then
    log "curobo: already installed"
  else
    if ! "$PY" -c 'import curobo' >/dev/null 2>&1; then
      # curobo compiles CUDA extensions; there is no nvcc on this cluster and
      # no root, so pull a matching toolchain into the env from conda-forge.
      if [ ! -x "$ENV_PREFIX/bin/nvcc" ]; then
        log "curobo: installing cuda-nvcc 12.4 toolchain into the env (no system nvcc, no root)"
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda install -p "$ENV_PREFIX" -c conda-forge -y \
          cuda-nvcc=12.4 cuda-cudart-dev=12.4 cuda-cccl=12.4 \
          libcurand-dev libcublas-dev libcusolver-dev libcusparse-dev \
          2>&1 | tee "$LOG_DIR/06-cuda.log" | tail -5
      fi
      [ -x "$ENV_PREFIX/bin/nvcc" ] || die "curobo: no nvcc at $ENV_PREFIX/bin/nvcc"

      if [ ! -d "$ROBOTWIN_DIR/envs/curobo" ]; then
        log "curobo: cloning NVlabs/curobo v0.7.8"
        git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git \
          "$ROBOTWIN_DIR/envs/curobo" 2>&1 | tail -2
      fi
      log "curobo: building (TORCH_CUDA_ARCH_LIST=$CUDA_ARCH_LIST; ~20 min, no GPU needed)"
      # conda-forge's CUDA packages put headers and libs under
      # $PREFIX/targets/x86_64-linux/{include,lib}, NOT $PREFIX/{include,lib}.
      # nvcc finds them relative to its own path, but the *host* gcc pass that
      # compiles torch's C++ glue does not, and dies with
      #   c10/cuda/CUDAStream.h:3:10: fatal error: cuda_runtime_api.h
      # CPATH / LIBRARY_PATH are what plain gcc honours.
      _tgt="$ENV_PREFIX/targets/x86_64-linux"
      CUDA_HOME="$ENV_PREFIX" \
      CUDA_PATH="$ENV_PREFIX" \
      CPATH="$_tgt/include:${CPATH:-}" \
      LIBRARY_PATH="$_tgt/lib:$_tgt/lib/stubs:${LIBRARY_PATH:-}" \
      LD_LIBRARY_PATH="$_tgt/lib:${LD_LIBRARY_PATH:-}" \
      TORCH_CUDA_ARCH_LIST="$CUDA_ARCH_LIST" \
      "$PIP" install --no-input -e "$ROBOTWIN_DIR/envs/curobo" --no-build-isolation \
        2>&1 | tee "$LOG_DIR/06-curobo.log" | tail -5
    fi
    # curobo resolves warp-lang to 1.16 and scipy past RoboTwin's ==1.10.1 pin;
    # upstream _install.sh accepts the same drift and pins warp back afterwards.
    "$PIP" install --no-input "warp-lang==1.12.0" "setuptools==69.5.1" 2>&1 | tail -2
    # ...and re-pin numpy: a fresh scipy/scikit-image can drag in the 2.x ABI,
    # which breaks sapien and RoboTwin's native extensions at import.
    "$PIP" install --no-input "numpy==1.26.4" 2>&1 | tail -1
    # curobo falls back to torch's JIT `load()` when a prebuilt .so will not
    # import; that path needs ninja.
    "$PIP" install --no-input ninja 2>&1 | tail -1

    # ---- glibc: the reason this is a TWO-PHASE build -------------------
    # Login node  : Ubuntu 22.04, glibc 2.35
    # Compute node: Ubuntu 20.04, glibc 2.31
    # Extensions compiled against the login node's libc link GLIBC_2.32+ and
    # die on every compute node with
    #   ImportError: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.32'
    #   not found (required by .../kinematics_fused_cu...so)
    # Phase 1 (above, login node) resolves and installs the pure-python deps,
    # which needs outbound network. Phase 2 recompiles the extensions on a
    # compute node, which has none but has the right libc.
    if [ -z "${SLURM_JOB_ID:-}" ]; then
      log "curobo: recompiling extensions on a compute node (login glibc is too new)"
      srun -A "${SLURM_ACCOUNT:-edgeai_tao-ptm_image-foundation-model-clip}" \
           -p "${SLURM_PARTITIONS:-interactive_singlenode,polar4,polar3,batch_singlenode}" \
           -t 01:00:00 --gpus-per-node=1 -N1 --job-name=curobo-build \
           bash -c "$(declare -f build_curobo_ext); \
                    ENV_PREFIX='$ENV_PREFIX' CUROBO_DIR='$ROBOTWIN_DIR/envs/curobo' \
                    CUDA_ARCH_LIST='$CUDA_ARCH_LIST' build_curobo_ext" \
        2>&1 | tee "$LOG_DIR/06-curobo-compute.log" | tail -8
    else
      ENV_PREFIX="$ENV_PREFIX" CUROBO_DIR="$ROBOTWIN_DIR/envs/curobo" \
      CUDA_ARCH_LIST="$CUDA_ARCH_LIST" build_curobo_ext | tail -8
    fi
    stamp curobo
  fi
fi

# ------------------------------------------------------------------- 7. assets
# RoboTwin-OD objects, the texture library and the embodiment models live on
# HuggingFace. They are large (~14 GB of zips), so they are staged under
# $DATA_DIR and symlinked into the checkout rather than copied.
if want assets; then
  if stamped assets; then
    log "assets: already staged at $DATA_DIR/assets"
  else
    mkdir -p "$DATA_DIR/assets_zip" "$DATA_DIR/assets"
    if [ -z "${HF_TOKEN:-}" ]; then
      REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
      # shellcheck disable=SC1091
      [ -f "$REPO_ROOT/.env.local" ] && { set +u; . "$REPO_ROOT/.env.local"; set -u; }
    fi
    [ -n "${HF_TOKEN:-}" ] || warn "assets: HF_TOKEN unset; public download may be rate-limited"
    export HF_TOKEN

    log "assets: downloading embodiments/objects/background_texture from TianxingChen/RoboTwin2.0"
    HF_ASSET_DEST="$DATA_DIR/assets_zip" "$PY" - <<'PYEOF'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="TianxingChen/RoboTwin2.0",
    allow_patterns=["background_texture.zip", "embodiments.zip", "objects.zip"],
    local_dir=os.environ["HF_ASSET_DEST"],
    repo_type="dataset",
    max_workers=8,
)
PYEOF

    for z in embodiments objects background_texture; do
      if [ -d "$DATA_DIR/assets/$z" ]; then
        log "assets: $z already extracted"
      else
        log "assets: extracting $z.zip"
        unzip -q -o "$DATA_DIR/assets_zip/$z.zip" -d "$DATA_DIR/assets/"
      fi
    done
    rm -rf "$DATA_DIR/assets/__MACOSX"

    # RoboTwin resolves assets via envs/_GLOBAL_CONFIGS.ASSETS_PATH, which is
    # hard-wired to <checkout>/assets/. Symlink the three payload dirs so the
    # checkout stays small and the data stays on the dataset volume.
    for z in embodiments objects background_texture; do
      tgt="$ROBOTWIN_DIR/assets/$z"
      [ -L "$tgt" ] && rm -f "$tgt"
      [ -d "$tgt" ] && [ ! -L "$tgt" ] && die "assets: $tgt is a real dir; refusing to replace"
      ln -s "$DATA_DIR/assets/$z" "$tgt"
    done

    # Expand ${ASSETS_PATH} in the *_tmp.yml curobo/embodiment configs. The
    # upstream helper prompts on stdin when it guesses wrong, so drive it from
    # the checkout root where its cwd heuristic succeeds.
    log "assets: expanding \${ASSETS_PATH} in embodiment *_tmp.yml configs"
    ( cd "$ROBOTWIN_DIR" && "$PY" scripts/update_embodiment_config_path.py </dev/null ) | tail -4
    stamp assets
  fi
fi

# ------------------------------------------------------------------- 8. vulkan
# SAPIEN 3 renders through Vulkan. The login node has no GPU and no ICD; the
# GPU nodes carry the NVIDIA ICD at /etc/vulkan/icd.d/nvidia_icd.json pointing
# at libGLX_nvidia.so.0 (resolved by ldconfig from
# /cm/local/apps/cuda/libs/current/lib64). /usr/share/vulkan/icd.d additionally
# holds intel/radeon/lvp ICDs -- lvp is Mesa's llvmpipe SOFTWARE rasteriser, and
# if the loader picks it SAPIEN renders on CPU (very slow, and not the GPU path
# we are benchmarking). So pin the loader to the NVIDIA ICD explicitly.
if want vulkan; then
  ENVSH="$ENV_PREFIX/robotwin_env.sh"
  log "vulkan: writing $ENVSH"
  cat > "$ENVSH" <<EOF
# Source this before running anything that renders. GPU node only.
export ROBOTWIN_ROOT="$ROBOTWIN_DIR"
export ROBOTWIN_DATA="$DATA_DIR"
export PYTHONPATH="\$ROBOTWIN_ROOT:\${PYTHONPATH:-}"

# --- Vulkan ---------------------------------------------------------------
_loom_icd=""
for c in /etc/vulkan/icd.d/nvidia_icd.json \\
         /usr/share/vulkan/icd.d/nvidia_icd.json \\
         "$ENV_PREFIX/share/vulkan/icd.d/nvidia_icd.json"; do
  [ -f "\$c" ] && { _loom_icd="\$c"; break; }
done
if [ -z "\$_loom_icd" ]; then
  # No system ICD: synthesise one. libGLX_nvidia.so.0 is the NVIDIA Vulkan
  # driver entry point and is on the ldconfig path on every GPU node here.
  mkdir -p "$ENV_PREFIX/share/vulkan/icd.d"
  cat > "$ENV_PREFIX/share/vulkan/icd.d/nvidia_icd.json" <<'JSON'
{ "file_format_version": "1.0.0",
  "ICD": { "library_path": "libGLX_nvidia.so.0", "api_version": "1.3.242" } }
JSON
  _loom_icd="$ENV_PREFIX/share/vulkan/icd.d/nvidia_icd.json"
fi
# VK_ICD_FILENAMES is the pre-1.3.207 name, VK_DRIVER_FILES the current one.
# Set both: SAPIEN bundles its own loader and its version varies.
export VK_ICD_FILENAMES="\$_loom_icd"
export VK_DRIVER_FILES="\$_loom_icd"
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
unset _loom_icd

# Headless: no DISPLAY, and RoboTwin only opens a viewer when render_freq>0.
unset DISPLAY 2>/dev/null || true
EOF
  stamp vulkan
fi

# ------------------------------------------------------------------- 9. verify
if want verify; then
  log "verify: import check (CPU-only; rendering must be checked on a GPU node)"
  # shellcheck disable=SC1090
  source "$ENV_PREFIX/robotwin_env.sh"
  cd "$ROBOTWIN_DIR"
  "$PY" - <<'PYEOF'
import importlib, sys
ok = True
for m in ["numpy", "torch", "sapien", "mplib", "gymnasium", "trimesh",
          "transforms3d", "toppra", "curobo", "h5py", "yaml"]:
    try:
        mod = importlib.import_module(m)
        print(f"  {m:14s} {getattr(mod, '__version__', 'n/a')}")
    except Exception as e:
        ok = False
        print(f"  {m:14s} FAILED: {type(e).__name__}: {e}")
import torch
have_gpu = torch.cuda.is_available()
try:
    from envs.hanging_mug import hanging_mug   # noqa: F401
    print("  envs.hanging_mug   import OK")
except Exception as e:
    # curobo's motion_gen evaluates `Pose.from_list([...])` as a class-body
    # default argument, which allocates a CUDA tensor AT IMPORT TIME. So
    # `import envs.<task>` cannot succeed without a GPU -- not a broken env.
    msg = f"{type(e).__name__}: {e}"
    if not have_gpu:
        print(f"  envs.hanging_mug   import needs a GPU (expected on the login node): {msg[:90]}")
    else:
        ok = False
        print(f"  envs.hanging_mug   FAILED: {msg}")
sys.exit(0 if ok else 1)
PYEOF
  log "verify: OK"
  cat <<EOF

Next: prove rendering on a GPU node (login node has no GPU and no Vulkan ICD):

  srun -A edgeai_tao-ptm_image-foundation-model-clip \\
       -p interactive_singlenode,polar4,polar3,batch_singlenode \\
       -t 00:40:00 --gpus-per-node=1 -N1 \\
       bash -lc 'source $ENV_PREFIX/robotwin_env.sh && \\
                 $PY <loom>/scripts/smoke_robotwin.py'
EOF
fi

log "done: stages [$STAGES]"
