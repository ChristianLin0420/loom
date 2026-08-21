#!/usr/bin/env python3
"""Entry point for fail-closed proposal-only outcome GRPO training."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loom.train.outcome_grpo import main


if __name__ == "__main__":
    raise SystemExit(main())
