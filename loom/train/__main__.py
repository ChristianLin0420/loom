"""``python -m loom.train --config configs/rX.yaml``.

The real entry point is :func:`loom.train.loop.main`; this only makes the short
form work as PLAN 4.D specifies it. ``python -m loom.train.loop`` is equivalent
and is what the sbatch files use, because it names the file that actually runs.
"""

from loom.train.loop import main

if __name__ == "__main__":
    raise SystemExit(main())
