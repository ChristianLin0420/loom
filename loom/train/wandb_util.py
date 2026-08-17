"""W&B that survives dozens of requeues.

Continuity is carried by *the run id plus a monotone global_step*, not by
``resume="allow"``.

  - The id is created once in ``runs/<name>/wandb_id`` and re-exported by every
    link, so 40 offline jobs merge server-side into one run.
  - Offline mode prints a scary ``"resume will be ignored"`` warning but still
    honours the requested id.
  - The step must never regress: always log with the ``global_step`` restored
    from the checkpoint and accept the small gap between the last checkpoint and
    the crash. W&B silently drops out-of-order steps.
  - ``wandb.init`` on rank 0 ONLY. Any other rank calling it produces 8 runs per
    link.
  - ``WANDB_DIR`` is the run dir, not ``<run_dir>/wandb`` -- wandb appends
    ``wandb/`` itself.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

__all__ = ["stable_run_id", "init", "log", "finish"]


def stable_run_id(run_dir: str | Path) -> str:
    """Created once, reused by every link and every requeue."""
    p = Path(run_dir) / "wandb_id"
    if p.exists():
        rid = p.read_text().strip()
        if rid:
            return rid
    rid = os.environ.get("WANDB_RUN_ID") or uuid.uuid4().hex[:16]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(rid)
    return rid


def init(run_dir: str | Path, project: str, config: dict, rank: int = 0,
         name: str | None = None):
    """Rank 0 only. Returns None everywhere else, and ``log`` is a no-op on None."""
    if rank != 0:
        return None
    try:
        import wandb
    except ImportError:
        print("[wandb] not installed; running without logging", flush=True)
        return None

    kw = dict(
        project=project or os.environ.get("WANDB_PROJECT", "loom"),
        id=stable_run_id(run_dir),
        name=name,
        resume="allow",
        dir=str(run_dir),          # wandb appends wandb/ itself
        config=config,
    )

    # Online mode reaches the network at init. Compute nodes DO have a route
    # (measured: api.wandb.ai in 0.22 s, a real run inits in 1.8 s), but a blip
    # at the wrong moment must never take down a 4 h link -- logging is a
    # convenience and training is the deliverable. So: bound the init, and on
    # any failure fall back to offline for this link. The stable run id means an
    # offline link still merges into the same run once `wandb_sync.sh` runs.
    mode = os.environ.get("WANDB_MODE", "online")
    try:
        run = wandb.init(settings=wandb.Settings(init_timeout=90), **kw)
    except Exception as e:                                  # noqa: BLE001
        print(f"[wandb] {mode} init failed ({type(e).__name__}: {e}); "
              f"falling back to offline for this link. Sync later with "
              f"`bash scripts/wandb_sync.sh <run>`.", flush=True)
        os.environ["WANDB_MODE"] = "offline"
        try:
            run = wandb.init(settings=wandb.Settings(init_timeout=90), **kw)
        except Exception as e2:                             # noqa: BLE001
            print(f"[wandb] offline init also failed ({type(e2).__name__}: {e2}); "
                  f"training continues without logging.", flush=True)
            return None

    print(f"[wandb] mode={os.environ.get('WANDB_MODE', mode)} id={kw['id']} "
          f"url={getattr(run, 'url', None)}", flush=True)
    run.define_metric("*", step_metric="global_step")
    return run


_LOG_FAILURES = 0


def log(run, metrics: dict, global_step: int) -> None:
    if run is None:
        return
    payload = dict(metrics)
    payload["global_step"] = global_step
    payload.setdefault("restart_count", int(os.environ.get("LOOM_RESTART_COUNT", 0)))
    # Online mode talks to the network on every call. wandb buffers and retries
    # internally, but an exception escaping here would kill training for a
    # logging failure, which is the wrong trade at 16 GPUs. Warn a few times so
    # it is visible in the log, then stay quiet rather than flooding it --
    # metrics.jsonl on Lustre is the durable record either way.
    global _LOG_FAILURES
    try:
        run.log(payload, step=global_step)
    except Exception as e:                                  # noqa: BLE001
        _LOG_FAILURES += 1
        if _LOG_FAILURES <= 3:
            print(f"[wandb] log failed at step {global_step} "
                  f"({type(e).__name__}: {e}); training continues, "
                  f"metrics.jsonl is unaffected.", flush=True)


def finish(run) -> None:
    if run is not None:
        run.finish()
