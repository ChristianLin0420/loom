#!/usr/bin/env python3
"""Entry point for the deliberately nonlaunchable outcome-GRPO v2 scaffold."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loom.train.outcome_grpo_v2 import main


if __name__ == "__main__":
    raise SystemExit(main())
