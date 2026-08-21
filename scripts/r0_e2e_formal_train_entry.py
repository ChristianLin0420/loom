#!/usr/bin/env python3
"""Formal-only W&B contract adapter, then the unchanged training-loop entry.

Historical trainers and completed diagnostic source closures keep importing the
byte-identical ``loom.train.wandb_util``.  Only this new formal launcher patches
the SDK boundary in its own process to add group/job type/tags, link-specific
resume policy, and an optional strict-online requirement.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from typing import Any


class FormalWandbError(RuntimeError):
    pass


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise FormalWandbError(f"{name} is required")
    return value


def _truth(value: str, *, name: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise FormalWandbError(f"{name} must be a boolean, got {value!r}")


def install_formal_wandb_contract() -> dict[str, Any]:
    import wandb
    from loom.train import wandb_util

    project = _required("LOOM_WANDB_PROJECT")
    group = _required("LOOM_WANDB_GROUP")
    job_type = _required("LOOM_WANDB_JOB_TYPE")
    resume = _required("LOOM_WANDB_RESUME").lower()
    if resume not in {"allow", "must", "never", "auto"}:
        raise FormalWandbError(f"invalid LOOM_WANDB_RESUME={resume!r}")
    tags = [tag.strip() for tag in _required("LOOM_WANDB_TAGS").split(",")]
    if any(not tag for tag in tags) or len(tags) != len(set(tags)):
        raise FormalWandbError("LOOM_WANDB_TAGS must be unique non-empty tags")
    require_online = _truth(
        _required("LOOM_WANDB_REQUIRE_ONLINE"),
        name="LOOM_WANDB_REQUIRE_ONLINE",
    )
    if os.environ.get("WANDB_MODE", "online") != "online":
        raise FormalWandbError("formal entry must begin with WANDB_MODE=online")

    sdk_init = wandb.init

    def formal_sdk_init(*args, **kwargs):
        if require_online and os.environ.get("WANDB_MODE", "online") != "online":
            raise FormalWandbError("offline fallback is forbidden for this formal run")
        actual_project = kwargs.get("project")
        if actual_project != project:
            raise FormalWandbError(
                f"config-owned W&B project {actual_project!r} != {project!r}"
            )
        kwargs.update({
            "group": group,
            "job_type": job_type,
            "tags": tags,
            "resume": resume,
            "mode": os.environ.get("WANDB_MODE", "online"),
        })
        run = sdk_init(*args, **kwargs)
        if require_online and bool(getattr(run, "offline", False)):
            try:
                run.finish()
            finally:
                raise FormalWandbError("W&B returned offline while online is required")
        return run

    wandb.init = formal_sdk_init
    base_init = wandb_util.init

    def formal_train_init(*args, **kwargs):
        run = base_init(*args, **kwargs)
        rank = int(kwargs.get("rank", args[3] if len(args) > 3 else 0))
        if rank == 0 and run is None:
            raise FormalWandbError("formal W&B initialization produced no run")
        if rank == 0 and require_online and bool(getattr(run, "offline", False)):
            raise FormalWandbError("formal W&B run is offline")
        return run

    wandb_util.init = formal_train_init
    receipt = {
        "project": project,
        "group": group,
        "job_type": job_type,
        "tags": tags,
        "resume": resume,
        "require_online": require_online,
    }
    print(f"[formal-wandb] {receipt}", flush=True)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    install_formal_wandb_contract()
    from loom.train.loop import main as train_main

    return int(train_main(list(argv) if argv is not None else None))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
