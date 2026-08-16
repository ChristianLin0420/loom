"""Put the repo root on sys.path so `import contracts` / `import stubs` work
from tests regardless of how pytest was invoked."""

import sys
from pathlib import Path

ROOT = str(Path(__file__).parent.resolve())
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
