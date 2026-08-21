#!/usr/bin/env python3
"""Strict-online W&B adapter for the fixed-step operator-repair trainer."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from loom.train import atomic as atomic_mod


LOG_EVERY_UPDATES = 20
MAX_CONSECUTIVE_LOG_FAILURES = 5
LOG_FAILURE_WINDOW_UPDATES = (
    LOG_EVERY_UPDATES * MAX_CONSECUTIVE_LOG_FAILURES
)
WANDB_HEALTH_FORMAT = "loom-operator-repair-wandb-health-v1"


class OperatorRepairWandbError(RuntimeError):
    pass


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise OperatorRepairWandbError(f"{name} is required")
    return value


def initial_wandb_health_state(lineage_sha256: str) -> dict[str, Any]:
    if len(lineage_sha256) != 64:
        raise OperatorRepairWandbError("W&B health lineage SHA-256 is invalid")
    return {
        "format": WANDB_HEALTH_FORMAT,
        "lineage_sha256": lineage_sha256,
        "log_every_updates": LOG_EVERY_UPDATES,
        "max_consecutive_failures": MAX_CONSECUTIVE_LOG_FAILURES,
        "events": [],
        "reconciliations": [],
    }


def _validate_wandb_health_state(
    value: Any, *, lineage_sha256: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise OperatorRepairWandbError("W&B health state is not an object")
    if set(value) != {
        "format", "lineage_sha256", "log_every_updates",
        "max_consecutive_failures", "events", "reconciliations",
    }:
        raise OperatorRepairWandbError("W&B health state keys changed")
    if (
        value.get("format") != WANDB_HEALTH_FORMAT
        or value.get("lineage_sha256") != lineage_sha256
        or value.get("log_every_updates") != LOG_EVERY_UPDATES
        or value.get("max_consecutive_failures")
        != MAX_CONSECUTIVE_LOG_FAILURES
    ):
        raise OperatorRepairWandbError("W&B health state contract changed")
    events = value.get("events")
    reconciliations = value.get("reconciliations")
    if not isinstance(events, list) or not isinstance(reconciliations, list):
        raise OperatorRepairWandbError("W&B health history is malformed")
    previous = 0
    for event in events:
        if (
            not isinstance(event, dict)
            or set(event) != {"global_step", "ok"}
            or not isinstance(event.get("global_step"), int)
            or isinstance(event.get("global_step"), bool)
            or event["global_step"] != previous + LOG_EVERY_UPDATES
            or not isinstance(event.get("ok"), bool)
        ):
            raise OperatorRepairWandbError("W&B health event history is malformed")
        previous = event["global_step"]
    for row in reconciliations:
        if not (
            isinstance(row, dict)
            and set(row) == {
                "committed_step", "discarded_events", "discarded_sha256",
            }
            and isinstance(row.get("committed_step"), int)
            and not isinstance(row.get("committed_step"), bool)
            and row["committed_step"] >= 0
            and isinstance(row.get("discarded_events"), int)
            and row["discarded_events"] > 0
            and isinstance(row.get("discarded_sha256"), str)
            and len(row["discarded_sha256"]) == 64
        ):
            raise OperatorRepairWandbError(
                "W&B health reconciliation history is malformed"
            )
    return value


def _consecutive_failures(events: Sequence[dict[str, Any]]) -> int:
    count = 0
    for event in reversed(events):
        if event["ok"]:
            break
        count += 1
    return count


def _load_persistent_wandb_health(
    *, path: Path, lineage_sha256: str, committed_step: int,
) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise OperatorRepairWandbError(
            "rank0 W&B health state must be an existing absolute regular file"
        )
    if committed_step < 0:
        raise OperatorRepairWandbError("committed checkpoint step is invalid")
    try:
        state = _validate_wandb_health_state(
            json.loads(path.read_text()), lineage_sha256=lineage_sha256,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise OperatorRepairWandbError(
            f"W&B health state is unreadable: {error}"
        ) from error
    retained = [
        event for event in state["events"]
        if event["global_step"] <= committed_step
    ]
    discarded = state["events"][len(retained):]
    if discarded:
        encoded = json.dumps(
            discarded, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()
        state["events"] = retained
        state["reconciliations"].append({
            "committed_step": committed_step,
            "discarded_events": len(discarded),
            "discarded_sha256": hashlib.sha256(encoded).hexdigest(),
        })
        atomic_mod.atomic_write_text(
            path,
            json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n",
        )
    return state


def _persist_wandb_health(path: Path, state: dict[str, Any]) -> None:
    atomic_mod.atomic_write_text(
        path,
        json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
    )


def install_operator_repair_wandb_contract() -> dict[str, Any]:
    import wandb
    from loom.train import wandb_util

    project = _required("LOOM_WANDB_PROJECT")
    group = _required("LOOM_WANDB_GROUP")
    job_type = _required("LOOM_WANDB_JOB_TYPE")
    resume = _required("LOOM_WANDB_RESUME").lower()
    if project != "loom-r0-operator-repair":
        raise OperatorRepairWandbError("operator-repair project identity changed")
    if resume not in {"allow", "must", "never"}:
        raise OperatorRepairWandbError(
            "fixed lineage resume must be never, bootstrap allow, or must"
        )
    tags = [value.strip() for value in _required("LOOM_WANDB_TAGS").split(",")]
    expected_tags = [
        "operator-repair", "fixed-endpoint", "no-gate",
        "fresh", "r0", "dual-action",
    ]
    if tags != expected_tags:
        raise OperatorRepairWandbError("operator-repair W&B tags changed")
    if _required("LOOM_WANDB_REQUIRE_ONLINE").lower() not in {"1", "true"}:
        raise OperatorRepairWandbError("operator-repair W&B must be required online")
    if os.environ.get("WANDB_MODE", "online") != "online":
        raise OperatorRepairWandbError("offline W&B is forbidden")

    lineage_sha256 = _required("LOOM_WANDB_LINEAGE_SHA256")
    health_path = Path(_required("LOOM_WANDB_HEALTH_STATE"))
    try:
        committed_step = int(_required("LOOM_WANDB_COMMITTED_STEP"))
    except ValueError as error:
        raise OperatorRepairWandbError(
            "LOOM_WANDB_COMMITTED_STEP must be an integer"
        ) from error
    rank = int(os.environ.get("RANK", "0"))
    health_state = (
        _load_persistent_wandb_health(
            path=health_path, lineage_sha256=lineage_sha256,
            committed_step=committed_step,
        )
        if rank == 0 else None
    )

    sdk_init = wandb.init

    def strict_sdk_init(*args, **kwargs):
        if os.environ.get("WANDB_MODE", "online") != "online":
            raise OperatorRepairWandbError("offline fallback is forbidden")
        if kwargs.get("project") != project:
            raise OperatorRepairWandbError(
                f"config project {kwargs.get('project')!r} != {project!r}"
            )
        kwargs.update({
            "group": group, "job_type": job_type, "tags": tags,
            "resume": resume, "mode": "online",
        })
        run = sdk_init(*args, **kwargs)
        if bool(getattr(run, "offline", False)):
            try:
                run.finish()
            finally:
                raise OperatorRepairWandbError("W&B returned an offline run")
        return run

    wandb.init = strict_sdk_init
    base_init = wandb_util.init

    def strict_train_init(*args, **kwargs):
        run = base_init(*args, **kwargs)
        rank = int(kwargs.get("rank", args[3] if len(args) > 3 else 0))
        if rank == 0 and (run is None or bool(getattr(run, "offline", False))):
            raise OperatorRepairWandbError("rank0 did not obtain an online W&B run")
        return run

    wandb_util.init = strict_train_init

    # The shared utility intentionally treats W&B as best-effort. This lineage
    # prospectively chooses a stricter, still transient-tolerant contract: rank
    # 0 calls the live SDK directly, resets the consecutive counter after any
    # success, and broadcasts each outcome so all ranks either continue or fail
    # together. Five failed calls at the frozen 20-update cadence is 100 updates.
    def strict_train_log(run, metrics, global_step):
        import torch.distributed as dist

        rank = int(os.environ.get("RANK", "0"))
        world = int(os.environ.get("WORLD_SIZE", "1"))
        packet: dict[str, Any] | None = None
        if rank == 0:
            assert health_state is not None
            payload = dict(metrics)
            payload["global_step"] = int(global_step)
            payload.setdefault(
                "restart_count", int(os.environ.get("LOOM_RESTART_COUNT", "0")),
            )
            try:
                if run is None:
                    raise OperatorRepairWandbError(
                        "rank0 online W&B run disappeared before logging"
                    )
                run.log(payload, step=int(global_step))
            except Exception as error:  # noqa: BLE001
                try:
                    if (
                        health_state["events"]
                        and int(global_step)
                        <= health_state["events"][-1]["global_step"]
                    ):
                        raise OperatorRepairWandbError(
                            "W&B health steps are not strictly increasing"
                        )
                    health_state["events"].append({
                        "global_step": int(global_step), "ok": False,
                    })
                    _persist_wandb_health(health_path, health_state)
                    consecutive = _consecutive_failures(health_state["events"])
                except Exception as state_error:  # noqa: BLE001
                    packet = {
                        "ok": False, "fatal": True,
                        "consecutive_failures": MAX_CONSECUTIVE_LOG_FAILURES,
                        "global_step": int(global_step),
                        "error": (
                            "health-state persistence failed: "
                            f"{type(state_error).__name__}: {state_error}"
                        ),
                    }
                else:
                    packet = {
                        "ok": False,
                        "fatal": consecutive >= MAX_CONSECUTIVE_LOG_FAILURES,
                        "consecutive_failures": consecutive,
                        "global_step": int(global_step),
                        "error": f"{type(error).__name__}: {error}",
                    }
            else:
                try:
                    if (
                        health_state["events"]
                        and int(global_step)
                        <= health_state["events"][-1]["global_step"]
                    ):
                        raise OperatorRepairWandbError(
                            "W&B health steps are not strictly increasing"
                        )
                    health_state["events"].append({
                        "global_step": int(global_step), "ok": True,
                    })
                    _persist_wandb_health(health_path, health_state)
                except Exception as state_error:  # noqa: BLE001
                    packet = {
                        "ok": False, "fatal": True,
                        "consecutive_failures": MAX_CONSECUTIVE_LOG_FAILURES,
                        "global_step": int(global_step),
                        "error": (
                            "health-state persistence failed: "
                            f"{type(state_error).__name__}: {state_error}"
                        ),
                    }
                else:
                    packet = {
                        "ok": True,
                        "fatal": False,
                        "consecutive_failures": 0,
                        "global_step": int(global_step),
                    }

        distributed = dist.is_available() and dist.is_initialized()
        if world > 1 and not distributed:
            raise OperatorRepairWandbError(
                "multi-rank strict W&B logging requires initialized distributed"
            )
        if distributed:
            packet_box = [packet]
            dist.broadcast_object_list(packet_box, src=0)
            packet = packet_box[0]
        if packet is None:
            raise OperatorRepairWandbError(
                "strict W&B logging did not receive the rank0 outcome"
            )
        if not packet["ok"]:
            if rank == 0:
                disposition = "fatal" if packet["fatal"] else "tolerated"
                print(
                    "[operator-repair-wandb] log failure "
                    f"{packet['consecutive_failures']}/"
                    f"{MAX_CONSECUTIVE_LOG_FAILURES} at update "
                    f"{packet['global_step']} ({disposition}; "
                    f"{packet['error']})",
                    flush=True,
                )
            if packet["fatal"]:
                raise OperatorRepairWandbError(
                    "online W&B logging failed for "
                    f"{packet['consecutive_failures']} consecutive calls "
                    f"through update {packet['global_step']}: {packet['error']}"
                )

    wandb_util.log = strict_train_log
    receipt = {
        "project": project, "group": group, "job_type": job_type,
        "tags": tags, "resume": resume, "require_online": True,
        "fixed_endpoint": 32_000, "decision_gate": False,
        "persistent_health_state": {
            "format": WANDB_HEALTH_FORMAT,
            "path": str(health_path),
            "lineage_sha256": lineage_sha256,
            "committed_step_at_entry": committed_step,
        },
        "log_failure_policy": {
            "kind": "consecutive_failures",
            "max_consecutive_failures": MAX_CONSECUTIVE_LOG_FAILURES,
            "log_every_updates": LOG_EVERY_UPDATES,
            "failure_window_updates": LOG_FAILURE_WINDOW_UPDATES,
            "success_resets_counter": True,
            "all_rank_outcome_broadcast": True,
        },
    }
    print(f"[operator-repair-wandb] {receipt}", flush=True)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    install_operator_repair_wandb_contract()
    from loom.train.loop import main as train_main

    return int(train_main(list(argv) if argv is not None else None))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
