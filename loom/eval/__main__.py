"""LOOM — eval CLI.

    python -m loom.eval --bench libero --ckpt <path> --out results.json

Every protocol knob is a flag, and whatever it ends up as is written into the
results JSON and printed above the table. Nothing about the protocol is
hardcoded in a loop (PLAN 4.F).

With no `--ckpt` (or with modules the other teams have not landed yet) this
runs on stub modules against `FakeLiberoEnv` and still emits a correctly shaped
table — that is the Phase 1A deliverable: plumbing, not numbers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loom.eval import EvalProtocol
from loom.eval.runner import bench_module, git_sha, n_devices, run_eval
from loom.eval.table import render_report

#: Which PLAN 8 row the numbers fill. `--row-label` overrides; when it is left
#: at the default, the bench picks (R0-A is the LIBERO row, R0-B the RoboTwin
#: one, and pasting a RoboTwin number into the LIBERO row would be worse than
#: no number at all).
ROW_LABEL_DEFAULT = "**LOOM · R0-A**"
DEFAULT_ROW_LABEL = {"libero": "**LOOM · R0-A**", "robotwin": "**LOOM · R0-B**",
                     "libero_plus": "**LOOM · R2**"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m loom.eval",
        description="LOOM evaluation harness. Emits results JSON + PLAN 8 markdown.",
    )
    p.add_argument("--bench", default="libero",
                   choices=["libero", "robotwin", "libero_plus"])
    p.add_argument("--ckpt", default=None, help="checkpoint; omit for stub modules")
    p.add_argument("--out", default="results.json", help="results JSON (written incrementally)")
    p.add_argument("--md", default=None, help="also write the markdown block here")

    g = p.add_argument_group("protocol (defaults are stated in the emitted table)")
    g.add_argument("--episodes-per-task", type=int, default=None)
    g.add_argument("--n-tasks", type=int, default=None)
    g.add_argument("--suites", default=None, help="comma-separated")
    g.add_argument("--seeds", default=None, help="comma-separated")
    g.add_argument("--max-steps", type=int, default=None)

    r = p.add_argument_group("execution")
    r.add_argument("--workers", type=int, default=None,
                   help="default: one per visible GPU, 1 on CPU")
    r.add_argument("--backend", default=None,
                   choices=["libero", "robotwin", "fake"],
                   help="default: real env when importable, fake otherwise")
    r.add_argument("--embodiment", default=None,
                   help="body to evaluate; default follows the bench "
                        "(libero -> libero_franka, robotwin -> robotwin_aloha)")
    r.add_argument("--no-resume", action="store_true",
                   help="ignore an existing --out instead of continuing it")
    r.add_argument("--require-real", action="store_true",
                   help="fail instead of falling back to stub modules / zero "
                        "features. Already implied by --ckpt; set this to also "
                        "refuse the stub path when no checkpoint is given.")
    r.add_argument("--op-stats", action="store_true",
                   help="record which operator `argmax pi_c` selected at every "
                        "replan into EpisodeResult.extra. Diagnostic only — it "
                        "does not change the action the policy takes.")
    r.add_argument("--gripper-dwell", type=int, default=1,
                   help="execute a HOLD-channel polarity reversal only after N "
                        "consecutive replans; 1 is the original path")
    r.add_argument("--decoder-samples", type=int, default=1,
                   help="average N deterministic CFM decoder samples per replan")
    r.add_argument("--duration-normalize-segments", action="store_true",
                   help="map each complete decoded 8-step segment onto its "
                        "SegmentClock-selected env bins (opt-in A/B arm)")
    r.add_argument("--row-label", default=ROW_LABEL_DEFAULT,
                   help="which PLAN 8 LOOM row these numbers fill; defaults per "
                        "bench (libero -> R0-A, robotwin -> R0-B)")
    r.add_argument("--quiet", action="store_true")
    return p


def protocol_from_args(args) -> EvalProtocol:
    base: EvalProtocol = bench_module(args.bench).DEFAULT_PROTOCOL
    kw = {}
    if args.episodes_per_task is not None:
        kw["episodes_per_task"] = args.episodes_per_task
    if args.n_tasks is not None:
        kw["n_tasks"] = args.n_tasks
    if args.suites:
        kw["suites"] = tuple(s.strip() for s in args.suites.split(",") if s.strip())
    if args.seeds:
        kw["seeds"] = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    if args.max_steps is not None:
        kw["max_steps"] = args.max_steps
    return base.replace(**kw) if kw else base


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protocol = protocol_from_args(args)

    # Absolute before anything runs: the RoboTwin backend `chdir`s into the
    # simulator's checkout (its configs store relative asset paths), so a
    # relative --out/--md/--ckpt would resolve somewhere nobody typed.
    args.out = str(Path(args.out).resolve())
    if args.md:
        args.md = str(Path(args.md).resolve())
    if args.ckpt:
        args.ckpt = str(Path(args.ckpt).resolve())
    if args.row_label == ROW_LABEL_DEFAULT:
        args.row_label = DEFAULT_ROW_LABEL.get(args.bench, args.row_label)

    if not args.quiet:
        print(f"[loom.eval] {protocol.bench}: {protocol.describe()}", file=sys.stderr)
        print(f"[loom.eval] workers={args.workers or n_devices()} "
              f"out={args.out}", file=sys.stderr)
        print(f"[loom.eval] git_sha={git_sha()}", file=sys.stderr)

    policy_kw: dict = {}
    if args.embodiment:
        policy_kw["embodiment"] = args.embodiment
    if args.require_real:
        policy_kw["allow_stub"] = False
    if args.op_stats:
        policy_kw["op_stats"] = True
    if args.gripper_dwell != 1:
        policy_kw["gripper_dwell"] = int(args.gripper_dwell)
    if args.decoder_samples != 1:
        policy_kw["decoder_samples"] = int(args.decoder_samples)
    if args.duration_normalize_segments:
        policy_kw["duration_normalize_segments"] = True

    done = {"n": 0}

    def tick(rec):
        done["n"] += 1
        if not args.quiet and done["n"] % 25 == 0:
            print(f"[loom.eval] {done['n']}/{protocol.total_episodes} episodes",
                  file=sys.stderr)

    results = run_eval(
        protocol,
        bench=args.bench,
        ckpt=args.ckpt,
        out=args.out,
        workers=args.workers,
        resume=not args.no_resume,
        backend=args.backend,
        policy_kw=policy_kw or None,
        on_episode=tick,
    )

    if not args.quiet:
        prov = results.get("meta", {}).get("policy") or {}
        print(f"[loom.eval] policy: is_stub={prov.get('is_stub')} "
              f"featurizer={prov.get('featurizer')} "
              f"ckpt_step={prov.get('ckpt_global_step')}", file=sys.stderr)

    md = render_report(results, row_label=args.row_label)
    if args.md:
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
