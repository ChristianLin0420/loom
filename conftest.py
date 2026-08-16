"""Shared pytest setup.

Puts the repo root on sys.path so `import contracts` / `import stubs` work from
tests regardless of how pytest was invoked, and keeps the embodiment registry
hermetic across test modules.
"""

import os
import sys
from pathlib import Path

# Before anything imports torch. This box reports 64 cores, so torch defaults to
# 32 intra-op threads; on a shared login node running several suites at once that
# thrashes rather than parallelises — the same suite measured 249 s at 32 threads
# and 8 s at 8. Only set when the caller has not chosen a value.
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import pytest  # noqa: E402

ROOT = str(Path(__file__).parent.resolve())
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(autouse=True)
def _isolate_embodiment_registry():
    """Restore `contracts.EMBODIMENTS` after every test.

    `register_embodiment` mutates a module-level dict, so a synthetic body
    registered by one team's tests stays registered for every test that runs
    after it — in file order, not in the order any one team ran locally.

    That is not hypothetical: a two-body dispatch test registered a 7-dof body,
    which made the estimator's dof-based embodiment inference ambiguous against
    `libero_franka` and failed nine unrelated estimator tests with
    `KeyError: dof 7 is ambiguous`. Each team's suite was green alone.

    This only contains registrations made *inside* a test. A module-scope
    `register_embodiment(...)` runs at collection, before any fixture, so it is
    already in the snapshot and stays visible for the whole session. Callers
    must therefore name their embodiment rather than rely on dof inference —
    which is the production path anyway, since `TransitionWindow` carries it.
    """
    import contracts

    saved = dict(contracts.EMBODIMENTS)
    try:
        yield
    finally:
        contracts.EMBODIMENTS.clear()
        contracts.EMBODIMENTS.update(saved)
