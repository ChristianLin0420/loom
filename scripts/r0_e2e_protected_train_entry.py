#!/usr/bin/env python3
"""Strict-online W&B entry point for one authenticated protected-action arm."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import r0_e2e_operator_repair_train_entry as entry


PROJECT = "loom-r0-protected-arms"
ARM_TAGS = {
    arm: (
        "protected-action", "fixed-endpoint", "no-gate",
        "fresh", "r0", f"arm-{arm.lower()}",
    )
    for arm in ("H", "P", "I")
}


def _arm_from_environment() -> str:
    arm = os.environ.get("LOOM_PROTECTED_ARM", "").strip()
    if arm not in ARM_TAGS:
        raise entry.OperatorRepairWandbError(
            "LOOM_PROTECTED_ARM must be exactly one of H/P/I"
        )
    return arm


def main(argv: Sequence[str] | None = None) -> int:
    arm = _arm_from_environment()
    entry.EXPECTED_PROJECT = PROJECT
    entry.EXPECTED_TAGS = ARM_TAGS[arm]
    entry.LOG_PREFIX = f"protected-arm-{arm.lower()}-wandb"
    return entry.main(list(argv) if argv is not None else None)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
