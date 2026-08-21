#!/usr/bin/env python3
"""Run the standalone, fail-closed round-0 outcome-GRPO trainer."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loom.train.outcome_grpo_round0 import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
