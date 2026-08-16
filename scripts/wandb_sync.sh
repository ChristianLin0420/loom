#!/usr/bin/env bash
# Run from a LOGIN node. Compute nodes have no route to api.wandb.ai, so every
# link writes an offline dir under runs/<name>/wandb/ and this pushes them.
#
# Every link reuses runs/<name>/wandb_id, so all of them merge server-side into
# ONE run. Safe to run repeatedly and while the chain is still going.
#
#   bash scripts/wandb_sync.sh r0a
set -euo pipefail

RUN_NAME="${1:?usage: wandb_sync.sh <run_name>}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

RUN_DIR="runs/${RUN_NAME}"
[ -d "$RUN_DIR" ] || { echo "no such run dir: $RUN_DIR" >&2; exit 2; }

[ -f .env.local ] && source .env.local        # WANDB_API_KEY; never echoed
[ -x .venv/bin/activate ] || true
# shellcheck disable=SC1091
[ -f .venv/bin/activate ] && source .venv/bin/activate

unset WANDB_MODE                              # sync must not be offline

N=$(find "$RUN_DIR" -maxdepth 3 -type d -name 'offline-run-*' | wc -l)
echo "syncing ${N} offline dir(s) from ${RUN_DIR} (id $(cat "$RUN_DIR/wandb_id" 2>/dev/null || echo '?'))"
find "$RUN_DIR" -maxdepth 3 -type d -name 'offline-run-*' -print0 \
  | xargs -0 -r -n1 wandb sync
