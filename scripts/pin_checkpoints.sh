#!/usr/bin/env bash
# Hardlink every checkpoint out of ckpt._prune's reach, while the run continues.
#
# _prune keeps keep_last=3 and permanent_every=10000. At ckpt_every=500 and
# ~1.3 it/s that is a ~19 minute window before a step is deleted forever. The
# best model this project has trained -- r0a_conv steps 2000-2800, act/align
# 0.047 and loss/proposal 5.00 against a 19.361 uniform baseline -- was pruned
# before anyone consolidated it, and a gradient spike destroyed the run 200
# steps later. Hardlinks cost no extra bytes and delete nothing.
#
#   bash scripts/pin_checkpoints.sh <run_name> [interval_s]
set -uo pipefail
RUN="${1:?usage: pin_checkpoints.sh <run_name> [interval_s]}"
INT="${2:-120}"
SRC="runs/${RUN}"; DST="runs/${RUN}_pinned"
mkdir -p "$DST"
while true; do
  for f in "$SRC"/ckpt_*.pt; do
    [ -e "$f" ] || continue
    b=$(basename "$f")
    [ -e "$DST/$b" ] || ln "$f" "$DST/$b" 2>/dev/null && :
  done
  # stop when the run is gone AND we have caught its final checkpoints
  squeue -u "$USER" -h -n "loom_${RUN}" -o "%T" 2>/dev/null | grep -qE 'RUNNING|PENDING' || {
    sleep 20
    for f in "$SRC"/ckpt_*.pt; do
      [ -e "$f" ] || continue; b=$(basename "$f"); [ -e "$DST/$b" ] || ln "$f" "$DST/$b" 2>/dev/null && :
    done
    echo "pinner: run finished; $(ls "$DST" | grep -c ckpt_ || echo 0) shard files pinned"
    break
  }
  sleep "$INT"
done
