"""Shared pytest setup.

Puts the repo root on sys.path so `import contracts` / `import stubs` work from
tests regardless of how pytest was invoked, and keeps the embodiment registry
hermetic across test modules.
"""

import os
import sys
from pathlib import Path

def _cpu_quota(default: int = 8) -> int:
    """Threads we may actually use, from the cgroup quota rather than core count.

    `nproc` and `sched_getaffinity` both report 64 here, but the cgroup caps us
    at `cpu.max = 400000/100000` = 4 CPUs. Torch reads the core count, defaults
    to 32 intra-op threads, and oversubscribes the quota 8x — which shows up as
    inexplicable slowness, not as an error. Measured: one suite 249 s at 32
    threads vs 8 s when reined in.
    """
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            return max(1, int(int(quota) // int(period)))
    except (OSError, ValueError):
        pass
    return default


# Must run before anything imports torch.
_N = str(_cpu_quota())
os.environ.setdefault("OMP_NUM_THREADS", _N)
os.environ.setdefault("MKL_NUM_THREADS", _N)

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
