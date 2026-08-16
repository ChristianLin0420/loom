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
from loom.eval.runner import bench_module, n_devices, run_eval
from loom.eval.table import render_report


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
    r.add_argument("--backend", default=None, choices=["libero", "fake"],
                   help="default: real env when importable, fake otherwise")
    r.add_argument("--no-resume", action="store_true",
                   help="ignore an existing --out instead of continuing it")
    r.add_argument("--row-label", default="**LOOM · R0-A**",
                   help="which PLAN 8 LOOM row these numbers fill")
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

    if not args.quiet:
        print(f"[loom.eval] {protocol.bench}: {protocol.describe()}", file=sys.stderr)
        print(f"[loom.eval] workers={args.workers or n_devices()} "
              f"out={args.out}", file=sys.stderr)

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
        on_episode=tick,
    )

    md = render_report(results, row_label=args.row_label)
    if args.md:
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
