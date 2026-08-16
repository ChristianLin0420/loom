#!/usr/bin/env bash
# Reproducible, idempotent setup of the LIBERO evaluation environment for LOOM.
#
#   bash scripts/setup_libero.sh
#
# Creates a standalone conda env (python 3.10) that is completely separate from the
# LOOM training .venv (python 3.13).  LIBERO pins robosuite 1.4 / numpy<2 / gym, none
# of which coexist with the training stack, so it gets its own interpreter.
#
# Everything here is safe to re-run: each step is guarded by an existence check.
#
# Owned by Team G.  See docs/ENV_LIBERO.md.
set -euo pipefail

# ----------------------------------------------------------------------------- paths
USER_ROOT="/lustre/fsw/portfolios/edgeai/users/chrislin"
ENV_PREFIX="${LOOM_LIBERO_ENV:-${USER_ROOT}/envs/loom-libero}"
DEPS_DIR="${LOOM_DEPS_DIR:-${USER_ROOT}/projects/loom-deps}"
LIBERO_DIR="${DEPS_DIR}/LIBERO"
DATA_ROOT="${LOOM_LIBERO_DATA:-${USER_ROOT}/datasets/loom/libero}"
PSE_OBJECT_DIR="${USER_ROOT}/projects/pse/data/libero/libero_object"

# Upstream Lifelong-Robot-Learning/LIBERO -- this is the benchmark the papers in
# PLAN.md Sec.8 (Light-WAM Table 1) evaluate on.  Do NOT swap this for the fwd4 fork
# without saying so in the results table; it changes comparability.
LIBERO_REPO="https://github.com/Lifelong-Robot-Learning/LIBERO.git"
LIBERO_COMMIT="${LIBERO_COMMIT:-}"   # optional pin; empty = whatever master is

CONDA_BIN="${CONDA_BIN:-$(command -v conda || echo /home/chrislin/miniconda3/bin/conda)}"
PY="${ENV_PREFIX}/bin/python"
PIP="${ENV_PREFIX}/bin/pip"

HF_REPO="yifengzhu-hf/LIBERO-datasets"
HF_REVISION="f13aa24a3da8c43c7225569f28c562979fa0e35a"
SUITES=(libero_spatial libero_object libero_goal libero_10)

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

# ------------------------------------------------------------------- 1. conda env
say "1/7 conda env at ${ENV_PREFIX}"
# Use a private, node-local package cache.  The shared ~/miniconda3/pkgs lives on
# lustre; two concurrent `conda create python=3.10` runs raced on it and left a
# half-extracted python package, which surfaces as a wall of
# "CondaVerificationError: ... appears to be corrupted" and no env.  Local /tmp is
# also 5-10x faster to extract into.
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/tmp/loom_libero_pkgs}"
mkdir -p "${CONDA_PKGS_DIRS}"
if [[ -x "${PY}" ]]; then
  echo "    exists: $(${PY} -V 2>&1)"
else
  "${CONDA_BIN}" create -y -p "${ENV_PREFIX}" python=3.10
fi

# ------------------------------------------------------------------- 2. clone LIBERO
say "2/7 LIBERO checkout at ${LIBERO_DIR}"
mkdir -p "${DEPS_DIR}"
if [[ -d "${LIBERO_DIR}/.git" ]]; then
  echo "    exists: $(cd "${LIBERO_DIR}" && git rev-parse --short HEAD)"
else
  git clone "${LIBERO_REPO}" "${LIBERO_DIR}"
fi
if [[ -n "${LIBERO_COMMIT}" ]]; then
  (cd "${LIBERO_DIR}" && git checkout -q "${LIBERO_COMMIT}")
fi

# ------------------------------------------------------------------- 3. python deps
# Order matters:
#   a) LIBERO's own requirements.txt (it pins robosuite==1.4.* itself),
#   b) hard constraints we re-assert on top,
#   c) torch LAST from the cu124 index.  robomimic depends on torch and will happily
#      drag in a +cu13x wheel that imports fine and reports cuda.is_available()==False
#      on this cluster (driver is CUDA 12.2).  Installing torch last overwrites it.
say "3/7 python dependencies"
# lustre is very slow for many small files; keep pip's cache and build tree on the
# node-local overlay.  Cuts the install from ~40 min to ~10.
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/loom_pipcache}"
export TMPDIR="${TMPDIR:-/tmp/loom_piptmp}"
export PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p "${PIP_CACHE_DIR}" "${TMPDIR}"

"${PIP}" install -U pip setuptools wheel

# NOTE: upstream LIBERO's setup.py has `install_requires=[]`, so `pip install -e .`
# installs nothing.  requirements.txt is from 2022 (numpy==1.22.4, transformers==4.21.1,
# hydra-core==1.2.0) and does not resolve; we install a curated set instead.
# `bddl` MUST be 1.0.1 -- bddl 3.x changed the parser API LIBERO calls.  bddl 1.0.1
#   imports `future` but does not declare it, so install it explicitly.
# `gym` MUST be <0.26 -- LIBERO's venv.py uses the old 4-tuple step API.
# `mujoco` MUST be 2.3.2.  robosuite 1.4.1 only declares `mujoco>=2.3.0`, so pip
#   happily installs 3.11, which removed `MjData.qM`.  Every env build then dies in
#   robosuite/controllers/base_controller.py with
#   "AttributeError: 'MjData' object has no attribute 'qM'" -- after EGL has already
#   initialised, so it looks like a rendering problem and is not.
"${PIP}" install \
  "numpy<2" \
  "robosuite==1.4.1" \
  "mujoco==2.3.2" \
  "bddl==1.0.1" "future" \
  "gym==0.25.2" \
  "h5py" \
  "opencv-python" \
  "imageio" "imageio-ffmpeg" \
  "easydict" \
  "hydra-core" \
  "matplotlib" \
  "cloudpickle" \
  "termcolor" "tqdm" "pyyaml" "einops"

# `editable_mode=compat` is REQUIRED.  LIBERO's outer `libero/` directory has no
# __init__.py, so it is a PEP-420 namespace package; setuptools >= 64's strict
# editable finder calls find_packages(), gets an empty mapping, and installs a
# `libero` distribution that cannot be imported ("ModuleNotFoundError: No module
# named 'libero'" from a `pip list` entry that says it is installed).  compat mode
# writes the old-style .pth pointing at the project root, which is what LIBERO's
# own install instructions assume.
"${PIP}" install --no-build-isolation --config-settings editable_mode=compat \
  -e "${LIBERO_DIR}"
"${PY}" -c "import libero.libero" || {
  echo "${LIBERO_DIR}" > "${ENV_PREFIX}/lib/python3.10/site-packages/zz_libero_src.pth"
  "${PY}" -c "import libero.libero"
}

# torch BEFORE robomimic, explicitly from cu124.  See CLAUDE.md "Environment".
# robomimic depends on torch AND torchvision; left to itself it downloads ~4 GB of
# default-index torch 2.13 + nvidia-*-cu13 wheels that import cleanly and report
# torch.cuda.is_available() == False on this CUDA-12.2 driver.  Installing torch
# first and then robomimic with --no-deps avoids the download entirely.
"${PIP}" install --index-url https://download.pytorch.org/whl/cu124 \
  "torch==2.6.0" "torchvision==0.21.0"

# robomimic is only needed by libero.lifelong (LIBERO's own training code), which we
# do not use -- the eval harness imports libero.libero.envs only.  Its egl_probe
# dependency builds from source and needs cmake, so this step may fail harmlessly.
# NB: robomimic's egl_probe dependency compiles against EGL headers that are not
# installed on this cluster and cannot be, so it is installed with --no-deps.
"${PIP}" install psutil tensorboard tensorboardX || true
"${PIP}" install --no-deps robomimic \
  || echo "    WARNING: robomimic not installed (not required for the eval harness)"

# ------------------------------------------------------------------- 4. libero config
# libero/libero/__init__.py calls input() at import time when ~/.libero/config.yaml is
# missing -- which hangs any non-interactive job.  Creating the file first suppresses
# the prompt; set_libero_default_path() then writes the real contents.
say "4/7 ~/.libero/config.yaml"
mkdir -p ~/.libero
touch ~/.libero/config.yaml
"${PY}" - <<PYEOF
from libero.libero import set_libero_default_path
set_libero_default_path()
PYEOF

# Point "datasets" at our tree; leave bddl_files/init_states/assets on the package.
"${PY}" - <<PYEOF
import os, yaml
cfg_path = os.path.expanduser("~/.libero/config.yaml")
with open(cfg_path) as f:
    cfg = yaml.safe_load(f) or {}
cfg["datasets"] = "${DATA_ROOT}"
with open(cfg_path, "w") as f:
    yaml.dump(cfg, f)
print(yaml.dump(cfg))
PYEOF

# ------------------------------------------------------------------- 5. assets check
say "5/7 bddl_files / init_files"
"${PY}" - <<'PYEOF'
import os, glob
from libero.libero import get_libero_path
for key in ("bddl_files", "init_states", "assets", "datasets"):
    p = get_libero_path(key)
    print(f"  {key:12s} {p}  exists={os.path.isdir(p)}")
bddl = get_libero_path("bddl_files")
init = get_libero_path("init_states")
for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"):
    nb = len(glob.glob(os.path.join(bddl, suite, "*.bddl")))
    ni = len(glob.glob(os.path.join(init, suite, "*.pruned_init")))
    print(f"  {suite:16s} bddl={nb:3d}  pruned_init={ni:3d}")
PYEOF

# ------------------------------------------------------------------- 6. datasets
say "6/7 datasets under ${DATA_ROOT}"
mkdir -p "${DATA_ROOT}"

# libero_object: reuse PSE's 7 GB copy via hardlinks (same lustre filesystem, so this
# costs zero extra bytes).  Hardlinks, not symlinks: our tree survives if PSE moves
# theirs.  NEVER write to or delete the PSE copy -- another project trains on it.
if [[ -d "${PSE_OBJECT_DIR}" ]]; then
  mkdir -p "${DATA_ROOT}/libero_object"
  for f in "${PSE_OBJECT_DIR}"/*.hdf5; do
    b="$(basename "$f")"
    if [[ ! -e "${DATA_ROOT}/libero_object/${b}" ]]; then
      ln "$f" "${DATA_ROOT}/libero_object/${b}" 2>/dev/null \
        || ln -s "$f" "${DATA_ROOT}/libero_object/${b}"
    fi
  done
  echo "    libero_object: $(ls "${DATA_ROOT}"/libero_object/*.hdf5 | wc -l) files (hardlinked from PSE)"
fi

# The other three suites come from HuggingFace at a pinned revision.  ~25 GB.
need_dl=0
for s in libero_spatial libero_goal libero_10; do
  n=$(ls "${DATA_ROOT}/${s}"/*.hdf5 2>/dev/null | wc -l)
  [[ "$n" -eq 10 ]] || need_dl=1
done
if [[ "${need_dl}" -eq 1 ]]; then
  if [[ -z "${HF_TOKEN:-}" && -f "$(dirname "$0")/../.env.local" ]]; then
    set -a; source "$(dirname "$0")/../.env.local"; set +a
  fi
  : "${HF_TOKEN:?HF_TOKEN not set -- source .env.local first}"
  "${PIP}" install -q "huggingface_hub>=0.30"
  HF_REPO="${HF_REPO}" HF_REVISION="${HF_REVISION}" DATA_ROOT="${DATA_ROOT}" "${PY}" - <<'PYEOF'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=os.environ["HF_REPO"],
    repo_type="dataset",
    revision=os.environ["HF_REVISION"],
    allow_patterns=["libero_spatial/*", "libero_goal/*", "libero_10/*"],
    local_dir=os.environ["DATA_ROOT"],
    token=os.environ["HF_TOKEN"],
    max_workers=8,
)
print("download complete")
PYEOF
else
  echo "    libero_spatial / libero_goal / libero_10 already present"
fi

for s in "${SUITES[@]}"; do
  printf '    %-16s %2d files  %s\n' "$s" \
    "$(ls "${DATA_ROOT}/${s}"/*.hdf5 2>/dev/null | wc -l)" \
    "$(du -sh "${DATA_ROOT}/${s}" 2>/dev/null | cut -f1)"
done

# ------------------------------------------------------------------- 7. summary
say "7/7 versions"
"${PY}" - <<'PYEOF'
import importlib
for m in ("numpy", "torch", "robosuite", "mujoco", "h5py", "gym", "bddl", "robomimic", "libero"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m:12s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  {m:12s} IMPORT FAILED: {type(e).__name__}: {e}")
PYEOF

cat <<EOF

Setup complete.

  env      ${ENV_PREFIX}
  LIBERO   ${LIBERO_DIR}
  data     ${DATA_ROOT}

Rendering is headless EGL.  GPU compute nodes carry libEGL.so.1 in
/cm/local/apps/cuda/libs/current/lib64 plus /usr/share/glvnd/egl_vendor.d/10_nvidia.json,
so no extra conda GL packages are needed.  The LOGIN NODE HAS NO EGL -- anything that
renders must run under srun.

  export MUJOCO_GL=egl
  export PYOPENGL_PLATFORM=egl

Smoke test (GPU node required):

  srun --account=edgeai_tao-ptm_image-foundation-model-clip \\
       --partition=polar4,polar3,polar,grizzly,batch_singlenode \\
       --time=00:30:00 --gpus=1 --nodes=1 \\
       bash -c 'MUJOCO_GL=egl ${PY} scripts/smoke_libero.py'
EOF
