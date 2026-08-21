#!/usr/bin/env python
"""Emit the frozen direct-formal convergence receipt for a metrics prefix.

Exit codes are stable: PASS=0, MOVING=1, INVALID=2, ABORT=3.  An optional
receipt path is created exclusively so a prior decision can never be replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from collections.abc import Mapping

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loom.train.direct_formal import (  # noqa: E402
    DirectFormalGate,
    evaluate_direct_formal,
    receipt_exit_code,
)


def _invalid_receipt(error: str, *, current_step: int | None = None) -> dict:
    receipt = evaluate_direct_formal([], current_step=0, gate=DirectFormalGate())
    receipt.update({
        "status": "INVALID",
        "reason": "invalid_input",
        "current_step": current_step,
        "decision_step": None,
        "next_check_step": None,
        "evaluations": [],
        "error": error,
    })
    return receipt


def _read_rows(path: pathlib.Path) -> tuple[list[Mapping], dict]:
    raw = path.read_bytes()
    identity = {
        "path": str(path.resolve()),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    if raw and not raw.endswith(b"\n"):
        raise ValueError("metrics.jsonl does not end in a newline")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"metrics.jsonl is not valid UTF-8: {error}") from error

    rows: list[Mapping] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise ValueError(f"metrics.jsonl line {line_number} is blank")
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(
                f"metrics.jsonl line {line_number} is not valid JSON: {error}"
            ) from error
        if not isinstance(row, Mapping):
            raise ValueError(f"metrics.jsonl line {line_number} must be an object")
        rows.append(row)
    return rows, identity


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=pathlib.Path)
    parser.add_argument(
        "--current-step", type=int, default=None,
        help="authenticate an expected completed-update count (default: last row)",
    )
    parser.add_argument(
        "--output", type=pathlib.Path, default=None,
        help="create the canonical JSON receipt exclusively at this path",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    metrics_path = args.run_dir / "metrics.jsonl"
    try:
        rows, identity = _read_rows(metrics_path)
        receipt = evaluate_direct_formal(rows, current_step=args.current_step)
        receipt["metrics_source"] = identity
    except (OSError, ValueError) as error:
        receipt = _invalid_receipt(str(error), current_step=args.current_step)
        receipt["metrics_source"] = {"path": str(metrics_path.resolve())}

    encoded = json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
    if args.output is not None:
        try:
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(encoded)
        except OSError as error:
            receipt = _invalid_receipt(
                f"could not exclusively publish receipt {args.output}: {error}",
                current_step=receipt.get("current_step"),
            )
            receipt["metrics_source"] = {"path": str(metrics_path.resolve())}
            encoded = json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"

    sys.stdout.write(encoded)
    return receipt_exit_code(receipt)


if __name__ == "__main__":
    raise SystemExit(main())
