#!/usr/bin/env bash
# Wait for a run to finish, pick its BEST checkpoint by training metrics,
# consolidate it, and submit the full 1200-episode LIBERO eval. No human step.
#
#   bash scripts/eval_when_done.sh <run_name> [poll_s]
#
# "Best" is NOT the last checkpoint. Runs here oscillate: r0a_final held
# align 0.073/prop 7.2 for 1400 steps and then sat on both floors, and the
# 20.08 came from step 4000 of 60000. So the final checkpoint is routinely not
# the good one, which is why scripts/pin_checkpoints.sh keeps them all.
#
# Selection metric: lowest act/align over a 500-step block, tie-broken by
# loss/proposal, among blocks that have a pinned checkpoint. Reported, not
# silent -- the chosen step and its numbers are printed before anything is spent.
set -uo pipefail
RUN="${1:?usage: eval_when_done.sh <run_name> [poll_s]}"
POLL="${2:-300}"
PY=.venv/bin/python

while squeue -u "$USER" -h -n "loom_${RUN}" -o "%T" 2>/dev/null | grep -qE 'RUNNING|PENDING'; do
  sleep "$POLL"
done
echo "$(date -Is) ${RUN} finished; selecting best checkpoint"
sleep 60          # let the pinner catch the final checkpoints

BEST=$($PY - "$RUN" <<'PYEOF'
import json, sys, os, re, statistics as st
run = sys.argv[1]
pin = f"runs/{run}_pinned"
have = set()
if os.path.isdir(pin):
    for f in os.listdir(pin):
        m = re.match(r"ckpt_0*(\d+)_rank\d+\.pt$", f)
        if m: have.add(int(m.group(1)))
rows = [json.loads(l) for l in open(f"runs/{run}/metrics.jsonl") if l.strip()]
best = None
for s in sorted(have):
    w = [r for r in rows if s - 500 < r.get("global_step", -1) <= s]
    if len(w) < 50: continue
    def med(k):
        v = [r[k] for r in w if k in r and r[k] is not None]
        return st.median(v) if v else float("inf")
    a, p = med("act/align"), med("loss/proposal")
    if best is None or (a, p) < (best[1], best[2]): best = (s, a, p)
print(json.dumps({"step": best[0], "align": best[1], "prop": best[2]} if best else {}))
PYEOF
)
echo "selected: $BEST"
STEP=$($PY -c "import json,sys; d=json.loads(sys.argv[1]); print(d.get('step',''))" "$BEST")
[ -n "$STEP" ] || { echo "no pinned checkpoint with metrics; nothing to evaluate"; exit 1; }

cp -n "runs/${RUN}/config.json" "runs/${RUN}_pinned/config.json" 2>/dev/null
$PY -m loom.train.consolidate --run_dir "runs/${RUN}_pinned" --step "$STEP" --pin
CKPT="$PWD/runs/${RUN}_pinned_eval/ckpt_$(printf '%09d' "$STEP").pt"
[ -e "$CKPT" ] || { echo "consolidation produced no file at $CKPT"; exit 1; }

echo "$(date -Is) submitting 1200-episode eval on $CKPT"
sbatch --job-name="eval_${RUN}" --export=ALL,CKPT="$CKPT" logs/eval_final.sh
