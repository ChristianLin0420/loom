#!/usr/bin/env bash
# Non-secret paths shared by every launcher: the sbatch stages AND the smoke
# scripts under logs/.
#
# This file exists because the two diverged. logs/r0a_smoke.sh exported
# LOOM_CACHE_DIR and the sbatch did not, so a green 1-GPU smoke was followed by
# a 16-GPU link that died at init with "$LOOM_CACHE_DIR is None". The smoke is
# only a gate if it runs in the same environment as the thing it gates.
#
# Secrets stay in .env.local (gitignored). Nothing here is a secret.
#
#   source scripts/env.sh

# Where the raw demos and the frozen-tower feature cache live. Both are on fsw,
# not fs11: the cache is ~71 GiB and is read every step.
: "${LOOM_DATASETS:=/lustre/fsw/portfolios/edgeai/users/chrislin/datasets/loom}"
export LOOM_DATASETS
export LOOM_DATA_ROOT="${LOOM_DATA_ROOT:-${LOOM_DATASETS}/libero}"
export LOOM_CACHE_DIR="${LOOM_CACHE_DIR:-${LOOM_DATASETS}/libero_cache}"

# The frozen SigLIP tower. Compute nodes have no route to huggingface.co, so the
# weights must already be here and the hub must be told not to try.
export HF_HOME="${HF_HOME:-${LOOM_DATASETS}/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# Login nodes run glibc 2.35, compute nodes 2.31. The shared ~/.triton/cache holds
# a cuda_utils.so linked against glibc >= 2.34, so Triton reuses it on a compute
# node and every torch.compile dies with "GLIBC_2.34 not found" -- which reads as
# "inductor is unsupported on this cluster". A project-local cache is the whole
# fix, and it is worth 12.2 ms -> 4.0 ms on the rollout.
# Do NOT clear ~/.triton/cache: the sibling pse project is mid-run and shares it.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PWD/.triton_cache}"

# Fail loudly here rather than 16 ranks deep. A missing cache is the difference
# between a real number and eight hours of torch.randn.
if [ ! -d "$LOOM_CACHE_DIR" ]; then
  echo "FATAL: LOOM_CACHE_DIR does not exist: $LOOM_CACHE_DIR" >&2
  echo "       build it with the frozen tower first (logs/build_cache.sh)." >&2
  return 1 2>/dev/null || exit 1
fi
