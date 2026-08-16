#!/usr/bin/env bash
# Decides WANDB_MODE, and re-verifies the CUDA wheel matches the node driver.
#
# Run this BEFORE the first long chain. Discovering the answer after 20 links
# means 20 unsynced W&B directories, and discovering the wheel mismatch after 20
# links means 20 links of CPU training that held 8 A100s each.
#
# Measured 2026-08: compute nodes have NO outbound route to api.wandb.ai, and the
# driver is CUDA 12.2 (so torch must be 2.6.0+cu124, not a +cu13x wheel).
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

srun --account=edgeai_tao-ptm_image-foundation-model-clip \
     --partition=polar4,polar3,polar,grizzly,batch_singlenode \
     --time=00:05:00 --nodes=1 --gpus-per-node=1 --ntasks=1 \
     bash -c '
       cd "'"$REPO"'"
       echo "node: $(hostname)"
       curl -sS -m 10 -o /dev/null -w "api.wandb.ai -> %{http_code}\n" https://api.wandb.ai/ \
         || echo "NO OUTBOUND NETWORK: keep WANDB_MODE=offline in every sbatch"
       source .venv/bin/activate 2>/dev/null || true
       python3 - <<PY
import torch
print("torch", torch.__version__, "built for cuda", torch.version.cuda)
print("cuda.is_available()", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("FAIL: this wheel cannot see the GPUs. The driver is CUDA 12.2 -- install")
    print("      pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0")
else:
    print("device", torch.cuda.get_device_name(0),
          f"{torch.cuda.get_device_properties(0).total_memory/2**30:.0f} GiB")
    print("bf16 supported", torch.cuda.is_bf16_supported(), "(A100 has NO fp8)")
PY
     '
