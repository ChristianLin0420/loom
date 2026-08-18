#!/usr/bin/env python
"""Is a run converged? Answers with evidence, not assertion.

Convergence here is NOT "the loss stopped moving" -- a run pinned at a degenerate
floor also stops moving, and this project has produced several. So the check is
two-sided:

  PLATEAU   the primary metrics' 2000-step block medians change by less than
            `tol` (relative) across the last `n_blocks` blocks, AND
  NOT-FLOOR the run is not sitting on a known degenerate floor:
              loss/proposal  uniform Plackett-Luce = sum(log(128-i), i<4) = 19.361
              act/align      disjoint-support MSE  = 8 * 0.25^2 = 0.500
              delta_sel      the clock signature   = |.| < 1e-4 (flat noise band)

A run that plateaus ON a floor is CONVERGED-DEGENERATE, which is a different
answer from CONVERGED and must not be reported as one.

    python logs/convergence.py runs/r0a_conv [--block 2000] [--blocks 4] [--tol 0.02]
"""
import argparse, json, math, statistics as st, sys, pathlib

UNIFORM_PL = sum(math.log(128 - i) for i in range(4))   # 19.360813
ALIGN_FLOOR = 8 * 0.25 ** 2                             # 0.500
PRIMARY = ("loss", "loss/dyn", "loss/act", "loss/proposal", "act/align")
WATCH = ("delta_op", "delta_sel", "bank/live_ops", "grad_norm")


def blocks(rows, size, n):
    if not rows:
        return []
    last = rows[-1]["global_step"]
    out = []
    for b in range(n):
        hi, lo = last - b * size, last - (b + 1) * size
        w = [r for r in rows if lo < r.get("global_step", -1) <= hi]
        if w:
            out.append((lo, hi, w))
    return list(reversed(out))


def med(w, k):
    v = [r[k] for r in w if k in r and r[k] is not None]
    return st.median(v) if v else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--block", type=int, default=2000)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--tol", type=float, default=0.02)
    a = ap.parse_args()

    p = pathlib.Path(a.run_dir) / "metrics.jsonl"
    if not p.exists():
        print(f"no metrics at {p}")
        return 2
    rows = [json.loads(l) for l in open(p) if l.strip()]
    if not rows:
        print("metrics.jsonl is empty")
        return 2
    step = rows[-1]["global_step"]
    bl = blocks(rows, a.block, a.blocks)
    print(f"run={a.run_dir}  last_step={step}  blocks={len(bl)} x {a.block}\n")

    if len(bl) < 2:
        print(f"VERDICT: TOO_EARLY — need >= {2 * a.block} steps to judge a plateau")
        return 1

    hdr = "metric".ljust(16) + "".join(f"{lo}-{hi}".rjust(16) for lo, hi, _ in bl) + "   rel.drift"
    print(hdr); print("-" * len(hdr))
    plateaued = {}
    for k in PRIMARY + WATCH:
        vals = [med(w, k) for _, _, w in bl]
        if all(math.isnan(v) for v in vals):
            continue
        fin = [v for v in vals if not math.isnan(v)]
        scale = max(abs(max(fin)), 1e-9)
        drift = (max(fin) - min(fin)) / scale
        if k in PRIMARY:
            plateaued[k] = drift <= a.tol
        mark = "" if k not in PRIMARY else ("  ok" if drift <= a.tol else "  MOVING")
        print(k.ljust(16) + "".join(f"{v:16.4f}" for v in vals) + f"{drift:12.3%}{mark}")

    last = bl[-1][2]
    prop, align, dsel = med(last, "loss/proposal"), med(last, "act/align"), med(last, "delta_sel")
    print()
    floors = []
    if not math.isnan(prop) and prop >= UNIFORM_PL - 0.05:
        floors.append(f"loss/proposal {prop:.3f} >= uniform PL {UNIFORM_PL:.3f}")
    if not math.isnan(align) and align >= ALIGN_FLOOR - 0.005:
        floors.append(f"act/align {align:.4f} >= disjoint floor {ALIGN_FLOOR}")
    if not math.isnan(dsel) and abs(dsel) < 1e-4:
        floors.append(f"|delta_sel| {abs(dsel):.2e} < 1e-4 (phase-clock signature)")

    all_flat = plateaued and all(plateaued.values())
    if not all_flat:
        moving = [k for k, v in plateaued.items() if not v]
        print(f"VERDICT: NOT_CONVERGED — still moving beyond {a.tol:.1%}: {', '.join(moving)}")
        return 1
    if floors:
        print("VERDICT: CONVERGED_DEGENERATE — plateaued, but ON a floor:")
        for f in floors:
            print(f"  - {f}")
        print("  Report the number, but do NOT report it as the method working.")
        return 3
    print(f"VERDICT: CONVERGED — all primaries flat within {a.tol:.1%} over "
          f"{len(bl) * a.block} steps, and off every known degenerate floor.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
