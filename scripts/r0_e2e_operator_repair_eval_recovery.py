#!/usr/bin/env python3
"""Versioned evaluation recovery for the completed R0 operator-repair endpoint.

The original fixed-step run reached update 32,000 and produced a consolidated,
fully verified checkpoint.  Its consolidation job then failed *after* the
verification PASS because the orchestration validator expected zero-padded rank
names while the checkpoint producer and receipt used real unpadded names.

This v2 workflow is intentionally evaluation-only.  It authenticates both the
historical training closure and the current one-line validator repair, adopts
the already-consolidated checkpoint through a fresh verify-only receipt, runs
the three predeclared singleton seed evaluations, and merges exactly 1,200
episodes.  It performs no training update, checkpoint reconstruction,
checkpoint selection, performance gate, or promotion.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import r0_e2e_operator_repair_chain as operator  # noqa: E402


FORMAT_VERSION = 2
KIND = "r0_e2e_operator_repair_eval_recovery_v2"
CHECKPOINT_ADOPTION_KIND = (
    "r0_e2e_operator_repair_eval_recovery_checkpoint_adoption_v2"
)
COMPLETION_KIND = "r0_e2e_operator_repair_eval_recovery_completion_v2"
EVAL_SOURCE_PENDING_KIND = (
    "r0_e2e_operator_repair_eval_runtime_source_pending_v2"
)
EVAL_SOURCE_COMPLETE_KIND = (
    "r0_e2e_operator_repair_eval_runtime_source_complete_v2"
)
STEP = operator.FIXED_STEP
WORLD_SIZE = operator.WORLD_SIZE
SEEDS = operator.SEEDS
PROJECT = operator.PROJECT
GROUP = "r0a-operator-repair-fixed32k-s0-20260821-eval-recovery-v2"
RECOVERY_HOST_VENV = (ROOT / ".venv").resolve()
RECOVERY_HOST_PYTHON = ROOT / ".venv/bin/python3"
RECOVERY_HOST_PYVENV_CFG = ROOT / ".venv/pyvenv.cfg"

SOURCE_RUN_DIR = (
    ROOT / "runs/r0a_operator_repair_s0_20260821_v1"
).resolve()
SOURCE_CONTROL_DIR = (
    ROOT / "runs/r0a_operator_repair_s0_20260821_v1_control"
).resolve()
SOURCE_ARTIFACT_ROOT = (
    ROOT / "runs/r0a_operator_repair_s0_20260821_v1_artifacts"
).resolve()
SOURCE_PLAN = SOURCE_CONTROL_DIR / "plan.json"
SOURCE_RUN_PLAN_COPY = SOURCE_RUN_DIR / "operator_repair_plan.json"
SOURCE_FIXED_ENDPOINT = SOURCE_CONTROL_DIR / "fixed_endpoint_32000.json"
SOURCE_FINAL_ASSET = (
    SOURCE_CONTROL_DIR / "training_asset_verification/train_06_post.json"
)
SOURCE_CHECKPOINT_REPORT = SOURCE_CONTROL_DIR / "checkpoint_verification.json"
SOURCE_PINNED_DIR = SOURCE_ARTIFACT_ROOT / "checkpoint/shards_000032000"
SOURCE_CHECKPOINT = SOURCE_ARTIFACT_ROOT / "checkpoint/ckpt.pt"
SOURCE_CONSOLIDATE_LOG = (
    ROOT / "logs/r0repair_r0a-operator-repair-fixed32k-s0-20260821-v1_"
    "consolidate_32651394.out"
)

HISTORICAL_GIT_COMMIT = "5138a527c68dfa55d00a780ca312b2202b5f9d77"
HISTORICAL_GIT_TREE = "049665062542085b57862db0328c4b9ca347920c"
HISTORICAL_SOURCE_CLOSURE_SHA256 = (
    "dbbb5d012fb9120163492a7cf91256521e75d2c3d1262e22d30b2b2758730813"
)
HISTORICAL_OPERATOR_SHA256 = (
    "5db5bf7639565b6371562f41d4d696161a11b87e48a5ff39a3670503efdca12b"
)
SOURCE_PLAN_SHA256 = "a409ce347fe1c56fdf4bf01558d03111f3fb22a2e0c98b9219b923dfd452be8a"
SOURCE_JOBS_SHA256 = "d0ec335f9763446015533ac04b4ef725fd30d93a43cbe697b0595f97dade4ddc"
SOURCE_RELEASE_SHA256 = "2eaefc1e8e84fd022b9e83808f70c81df1c5b95d87525e732d94411421e9f3ad"
SOURCE_ENDPOINT_SHA256 = "435838dec946835adbec33585dc75eef3c14607234a5f1c725af9df18e9ef597"
SOURCE_FINAL_ASSET_SHA256 = (
    "bf6ff0e8ff279085a61b99b81d17d26289b16c84e58511b207b2b7226f951071"
)
SOURCE_REPORT_SHA256 = "0e8606568228157bf0aeb948e13ec42a253a01c9ea369f78eb6a65b2b2523e74"
SOURCE_CHECKPOINT_SHA256 = (
    "ee8d3d583624be8c87cf6222c2d1716905d0ea21a4e1af5db094ef5d8273b36c"
)
SOURCE_CHECKPOINT_BYTES = 1_760_597_436
SOURCE_LOG_SHA256 = "4ea0ea5b010d255efaa1d49b277eb90760399106c362fbb57aea85627f6ceb8e"
SOURCE_CONFIG_SHA256 = "d8848f13a2773b1fd82ceb16334484889a7c24dfcef4ecf9113654be555fa77c"
SOURCE_METRICS_SHA256 = "3ceed51a2fe68e204e83363727f156e50922c86a719807ef9b20791dc1e46b79"
SOURCE_LATEST_SHA256 = "492f431bae35265f2e5f4ed49bd8c58dda912431be561504846988d00d05d117"
SOURCE_WANDB_ID_SHA256 = "f66feaa1b5bc0ba87fc7530a02219e421048f5cf8781644dfdbf047ce24e3d41"

SOURCE_IDENTITIES: dict[str, tuple[Path, str, int]] = {
    "plan": (SOURCE_PLAN, SOURCE_PLAN_SHA256, 323_332),
    "run_plan_copy": (SOURCE_RUN_PLAN_COPY, SOURCE_PLAN_SHA256, 323_332),
    "jobs": (SOURCE_CONTROL_DIR / "jobs.json", SOURCE_JOBS_SHA256, 8_852),
    "release": (
        SOURCE_CONTROL_DIR / "released.json", SOURCE_RELEASE_SHA256, 455,
    ),
    "fixed_endpoint": (SOURCE_FIXED_ENDPOINT, SOURCE_ENDPOINT_SHA256, 4_666),
    "final_asset": (SOURCE_FINAL_ASSET, SOURCE_FINAL_ASSET_SHA256, 3_753),
    "checkpoint_report": (
        SOURCE_CHECKPOINT_REPORT, SOURCE_REPORT_SHA256, 2_379,
    ),
    "checkpoint": (
        SOURCE_CHECKPOINT, SOURCE_CHECKPOINT_SHA256, SOURCE_CHECKPOINT_BYTES,
    ),
    "consolidate_log": (SOURCE_CONSOLIDATE_LOG, SOURCE_LOG_SHA256, 9_612),
    "run_config": (
        SOURCE_RUN_DIR / "config.json", SOURCE_CONFIG_SHA256, 4_285,
    ),
    "metrics": (
        SOURCE_RUN_DIR / "metrics.jsonl", SOURCE_METRICS_SHA256, 63_293_771,
    ),
    "latest": (SOURCE_RUN_DIR / "LATEST", SOURCE_LATEST_SHA256, 5),
    "wandb_id": (SOURCE_RUN_DIR / "wandb_id", SOURCE_WANDB_ID_SHA256, 17),
}

SOURCE_JOB_IDS = {
    "train_01": "32651388", "train_02": "32651389",
    "train_03": "32651390", "train_04": "32651391",
    "train_05": "32651392", "train_06": "32651393",
    "consolidate": "32651394", "eval_seed0": "32651395",
    "eval_seed1": "32651396", "eval_seed2": "32651397",
    "merge": "32651398",
}
SOURCE_TERMINAL_STATES = {
    **{str(job): "COMPLETED_0:0" for job in range(32651388, 32651394)},
    "32651394": "FAILED_2:0_AFTER_CHECKPOINT_VERIFY_PASS",
    "32651395": "CANCELLED_AFTEROK",
    "32651396": "CANCELLED_AFTEROK",
    "32651397": "CANCELLED_AFTEROK",
    "32651398": "CANCELLED_AFTEROK",
}

SOURCE_PROTECTED_ABSENT = (
    SOURCE_CONTROL_DIR / "checkpoint_receipt.json",
    SOURCE_CONTROL_DIR / "eval_seed_0_receipt.json",
    SOURCE_CONTROL_DIR / "eval_seed_1_receipt.json",
    SOURCE_CONTROL_DIR / "eval_seed_2_receipt.json",
    SOURCE_CONTROL_DIR / "merged_eval_receipt.json",
    SOURCE_ARTIFACT_ROOT / "eval",
)

RECOVERY_SOURCE_FILES = tuple(sorted(set(
    operator.SOURCE_FILES + (
        "scripts/r0_e2e_operator_repair_eval_recovery.py",
        "scripts/r0_e2e_operator_repair_eval_recovery_checkpoint.sbatch",
        "scripts/r0_e2e_operator_repair_eval_recovery_seed.sbatch",
        "scripts/r0_e2e_operator_repair_eval_recovery_control.sbatch",
    )
)))

EXPECTED_ELIGIBILITY = {
    "fixed_endpoint_full_run": True,
    "formal_convergence_gate": False,
    "checkpoint_selection_by_metrics_or_evaluation": False,
    "evaluation_unconditional_after_integrity": True,
    "promotion_authority": False,
    "versioned_evaluation_recovery": True,
}
EXPECTED_METHOD = {
    "source_optimizer_updates": STEP,
    "recovery_training_updates": 0,
    "recovery_optimizer_steps": 0,
    "checkpoint_reconstructions": 0,
    "checkpoint_selection_used_metrics_or_evaluation": False,
    "checkpoint": "exact_existing_step_32000_consolidated_bytes",
    "checkpoint_adoption": "fresh_verify_only_no_source_write",
    "scientific_gates": 0,
    "evaluation_episodes": 1_200,
    "evaluation_seeds": list(SEEDS),
}


class RecoveryError(RuntimeError):
    """Fail-closed recovery provenance or execution error."""


def sha256_file(path: str | Path) -> str:
    return operator.sha256_file(path)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _pretty_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _identity(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RecoveryError(f"required source is absent/nonregular: {path}")
    return {
        "path": str(path.resolve()), "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _checked_identity(path: Path, digest: str, size: int, *, label: str) -> dict[str, Any]:
    identity = _identity(path)
    if identity["sha256"] != digest or identity["bytes"] != size:
        raise RecoveryError(f"historical source {label} identity changed")
    return identity


def _closure(files: Sequence[str]) -> dict[str, Any]:
    rows = {name: sha256_file(ROOT / name) for name in files}
    digest = hashlib.sha256()
    for name in sorted(rows):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(rows[name].encode())
        digest.update(b"\0")
    return {
        "scheme": "sha256(path-nul-sha256-nul)-v1",
        "sha256": digest.hexdigest(), "files": rows,
    }


def _runtime_source_closure() -> dict[str, Any]:
    return _closure(RECOVERY_SOURCE_FILES)


def _recovery_host_runtime_receipt() -> dict[str, Any]:
    """Authenticate the .venv used by every recovery sbatch.

    The evaluation child has its own frozen LIBERO interpreter receipt in the
    inherited operator plan.  This separate receipt covers the Python process
    that verifies the checkpoint, orchestrates each seed, and performs merge.
    """
    if (
        not RECOVERY_HOST_PYTHON.is_symlink()
        or not RECOVERY_HOST_PYVENV_CFG.is_file()
        or RECOVERY_HOST_PYVENV_CFG.is_symlink()
    ):
        raise RecoveryError("recovery host .venv structure changed")
    resolved = RECOVERY_HOST_PYTHON.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise RecoveryError("recovery host Python target is not a regular file")
    try:
        frozen = subprocess.run(
            [str(RECOVERY_HOST_PYTHON), "-m", "pip", "freeze", "--all"],
            cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
        probe_code = (
            "import importlib.metadata as m,json,sys;"
            "names=('numpy','PyYAML','torch','transformers','wandb');"
            "print(json.dumps({'base_prefix':sys.base_prefix,"
            "'executable':sys.executable,'prefix':sys.prefix,"
            "'python_version':sys.version,"
            "'packages':{n:m.version(n) for n in names}},sort_keys=True))"
        )
        probe_text = subprocess.run(
            [str(RECOVERY_HOST_PYTHON), "-c", probe_code], cwd=ROOT,
            check=True, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
        probe = json.loads(probe_text)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise RecoveryError("could not authenticate recovery host .venv") from exc
    lines = sorted(line.strip() for line in frozen.splitlines() if line.strip())
    normalized = "\n".join(lines) + "\n"
    if (
        probe.get("prefix") != str(RECOVERY_HOST_VENV)
        or Path(str(probe.get("executable", ""))).resolve() != resolved
    ):
        raise RecoveryError("recovery host Python did not activate the frozen .venv")
    return {
        "kind": "r0_e2e_operator_repair_recovery_host_runtime_v2",
        "configured_python": str(RECOVERY_HOST_PYTHON),
        "configured_python_symlink": os.readlink(RECOVERY_HOST_PYTHON),
        "resolved_python": _identity(resolved),
        "pyvenv_cfg": _identity(RECOVERY_HOST_PYVENV_CFG),
        "pip_freeze": {
            "command": [str(RECOVERY_HOST_PYTHON), "-m", "pip", "freeze", "--all"],
            "lines": lines, "packages": len(lines),
            "sha256": hashlib.sha256(normalized.encode()).hexdigest(),
        },
        "probe": probe,
    }


def _require_active_recovery_host_runtime() -> None:
    if (
        Path(sys.prefix).resolve() != RECOVERY_HOST_VENV
        or Path(sys.executable).resolve() != RECOVERY_HOST_PYTHON.resolve()
    ):
        raise RecoveryError("run-stage must execute inside the frozen recovery .venv")


def _git_output(command: Sequence[str], *, label: str) -> bytes:
    try:
        completed = subprocess.run(
            list(command), cwd=ROOT, check=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RecoveryError(f"failed to authenticate historical git {label}") from exc
    return completed.stdout


def _historical_git_closure(source_plan: Mapping[str, Any]) -> dict[str, Any]:
    tree = _git_output(
        ["git", "show", "-s", "--format=%T", HISTORICAL_GIT_COMMIT],
        label="tree",
    ).decode().strip()
    if tree != HISTORICAL_GIT_TREE:
        raise RecoveryError("historical git tree changed")
    frozen = source_plan.get("source_closure")
    if not isinstance(frozen, Mapping) or frozen.get("sha256") != (
        HISTORICAL_SOURCE_CLOSURE_SHA256
    ):
        raise RecoveryError("historical plan source closure changed")
    files = frozen.get("files")
    if not isinstance(files, Mapping) or len(files) != 53:
        raise RecoveryError("historical plan does not own exactly 53 source files")
    verified: dict[str, str] = {}
    for name, expected in sorted(files.items()):
        if not isinstance(name, str) or not re.fullmatch(r"[0-9a-f]{64}", str(expected)):
            raise RecoveryError("historical closure row is malformed")
        blob = _git_output(
            ["git", "show", f"{HISTORICAL_GIT_COMMIT}:{name}"],
            label=name,
        )
        actual = hashlib.sha256(blob).hexdigest()
        if actual != expected:
            raise RecoveryError(f"historical git blob differs for {name}")
        verified[name] = actual
    digest = hashlib.sha256()
    for name in sorted(verified):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(verified[name].encode())
        digest.update(b"\0")
    if digest.hexdigest() != HISTORICAL_SOURCE_CLOSURE_SHA256:
        raise RecoveryError("historical git closure aggregate changed")
    return {
        "kind": "git_authenticated_historical_operator_repair_closure_v1",
        "commit": HISTORICAL_GIT_COMMIT, "tree": HISTORICAL_GIT_TREE,
        "source_closure_sha256": HISTORICAL_SOURCE_CLOSURE_SHA256,
        "files": len(verified), "all_plan_file_hashes_match_git": True,
    }


def _validator_repair_receipt(source_plan: Mapping[str, Any]) -> dict[str, Any]:
    relative = "scripts/r0_e2e_operator_repair_chain.py"
    historical = _git_output(
        ["git", "show", f"{HISTORICAL_GIT_COMMIT}:{relative}"],
        label="historical operator chain",
    )
    current = (ROOT / relative).read_bytes()
    old = b'f"ckpt_{FIXED_STEP:09d}_rank{rank:05d}.pt"'
    new = b'f"ckpt_{FIXED_STEP:09d}_rank{rank}.pt"'
    if historical.count(old) != 1 or historical.count(new) != 0:
        raise RecoveryError("historical validator source does not contain the exact bug")
    if current.count(new) != 1 or current.count(old) != 0:
        raise RecoveryError("current validator does not contain the exact repair")
    if historical.replace(old, new) != current:
        raise RecoveryError("runtime operator chain differs by more than the one-line repair")
    historical_files = source_plan["source_closure"]["files"]
    for name, expected in historical_files.items():
        if name == relative:
            continue
        if sha256_file(ROOT / name) != expected:
            raise RecoveryError(f"runtime dependency changed outside validator repair: {name}")
    historical_sha = hashlib.sha256(historical).hexdigest()
    if historical_sha != HISTORICAL_OPERATOR_SHA256:
        raise RecoveryError("historical operator chain SHA changed")
    return {
        "kind": "unpadded_checkpoint_shard_validator_repair_v1",
        "historical_operator_sha256": historical_sha,
        "runtime_operator_sha256": hashlib.sha256(current).hexdigest(),
        "historical_expected_template": "ckpt_{step:09d}_rank{rank:05d}.pt",
        "producer_and_runtime_template": "ckpt_{step:09d}_rank{rank}.pt",
        "changed_lines": 1,
        "all_other_historical_closure_files_byte_identical": True,
        "training_or_model_semantics_changed": False,
    }


def _load_source_plan() -> dict[str, Any]:
    _checked_identity(
        SOURCE_PLAN, SOURCE_PLAN_SHA256, SOURCE_IDENTITIES["plan"][2], label="plan",
    )
    _checked_identity(
        SOURCE_RUN_PLAN_COPY, SOURCE_PLAN_SHA256,
        SOURCE_IDENTITIES["run_plan_copy"][2], label="run plan copy",
    )
    try:
        value = json.loads(SOURCE_PLAN.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RecoveryError("historical operator plan is unreadable") from exc
    if not isinstance(value, Mapping):
        raise RecoveryError("historical operator plan is not an object")
    plan = dict(value)
    if (
        plan.get("format_version") != operator.FORMAT_VERSION
        or plan.get("kind") != operator.KIND
        or plan.get("source_closure", {}).get("sha256")
        != HISTORICAL_SOURCE_CLOSURE_SHA256
        or plan.get("lineage", {}).get("run_dir") != str(SOURCE_RUN_DIR)
        or plan.get("lineage", {}).get("control_dir") != str(SOURCE_CONTROL_DIR)
        or plan.get("lineage", {}).get("artifact_root") != str(SOURCE_ARTIFACT_ROOT)
        or plan.get("schedule", {}).get("fixed_updates") != STEP
        or plan.get("eligibility", {}).get("formal_convergence_gate") is not False
        or plan.get("eligibility", {}).get("evaluation_unconditional_after_integrity")
        is not True
    ):
        raise RecoveryError("historical operator plan semantic identity changed")
    _historical_git_closure(plan)
    _validator_repair_receipt(plan)
    return plan


def _source_endpoint(*, rehash_shards: bool) -> dict[str, Any]:
    endpoint_identity = _checked_identity(
        SOURCE_FIXED_ENDPOINT, SOURCE_ENDPOINT_SHA256,
        SOURCE_IDENTITIES["fixed_endpoint"][2], label="fixed endpoint",
    )
    try:
        endpoint = json.loads(SOURCE_FIXED_ENDPOINT.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RecoveryError("historical fixed endpoint is unreadable") from exc
    expected_names = {
        f"ckpt_{STEP:09d}_rank{rank}.pt" for rank in range(WORLD_SIZE)
    }
    shards = endpoint.get("checkpoint_shards") if isinstance(endpoint, Mapping) else None
    if not (
        endpoint.get("format_version") == operator.FORMAT_VERSION
        and endpoint.get("kind") == "r0_e2e_operator_repair_fixed_endpoint"
        and endpoint.get("plan_sha256") == SOURCE_PLAN_SHA256
        and endpoint.get("step") == STEP
        and endpoint.get("optimizer_updates") == STEP
        and endpoint.get("selection")
        == "predeclared_fixed_step_no_metric_or_eval_selection"
        and endpoint.get("direct_formal_receipts") == []
        and endpoint.get("health_metrics_used_as_gate") is False
        and endpoint.get("evaluation_required_after_integrity") is True
        and endpoint.get("run_config", {}).get("sha256") == SOURCE_CONFIG_SHA256
        and endpoint.get("metrics", {}).get("sha256") == SOURCE_METRICS_SHA256
        and endpoint.get("metrics", {}).get("rows") == STEP
        and endpoint.get("training_asset_verification", {}).get("sha256")
        == SOURCE_FINAL_ASSET_SHA256
        and isinstance(shards, Mapping) and set(shards) == expected_names
        and all(
            isinstance(row, Mapping)
            and isinstance(row.get("bytes"), int) and row["bytes"] > 0
            and re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256")))
            for row in shards.values()
        )
    ):
        raise RecoveryError("historical fixed endpoint contract changed")
    if rehash_shards:
        try:
            source_paths = operator.common._checkpoint_shards(
                SOURCE_RUN_DIR, STEP, WORLD_SIZE, require_latest=True,
            )
            pinned_paths = operator.common._checkpoint_shards(
                SOURCE_PINNED_DIR, STEP, WORLD_SIZE, require_latest=False,
            )
            source_receipt = operator.common._checkpoint_shard_receipt(source_paths)
            pinned_receipt = operator.common._checkpoint_shard_receipt(pinned_paths)
        except operator.common.ChainError as exc:
            raise RecoveryError(str(exc)) from exc
        if source_receipt != shards or pinned_receipt != shards:
            raise RecoveryError("source/pinned shards differ from fixed endpoint")
        source_by_name = {path.name: path for path in source_paths}
        pinned_by_name = {path.name: path for path in pinned_paths}
        if any(
            not os.path.samefile(source_by_name[name], pinned_by_name[name])
            for name in expected_names
        ):
            raise RecoveryError("pinned checkpoint shards are not exact source hardlinks")
    return {"identity": endpoint_identity, "receipt": dict(endpoint)}


def _checkpoint_payload_identity() -> dict[str, Any]:
    identity = _checked_identity(
        SOURCE_CHECKPOINT, SOURCE_CHECKPOINT_SHA256, SOURCE_CHECKPOINT_BYTES,
        label="consolidated checkpoint",
    )
    try:
        import torch  # noqa: PLC0415

        payload = torch.load(
            str(SOURCE_CHECKPOINT), map_location="cpu", weights_only=True, mmap=True,
        )
    except Exception as exc:  # noqa: BLE001
        raise RecoveryError("consolidated checkpoint cannot be safely loaded") from exc
    consolidated = payload.get("consolidated")
    expected_shards = [
        f"ckpt_{STEP:09d}_rank{rank}.pt" for rank in range(WORLD_SIZE)
    ]
    if not (
        payload.get("global_step") == STEP
        and payload.get("config_hash")
        == _load_source_plan()["config"]["resolved_config_hash"]
        and isinstance(payload.get("model"), Mapping)
        and len(payload["model"]) == 923
        and isinstance(consolidated, Mapping)
        and consolidated.get("tool") == "loom.train.consolidate"
        and consolidated.get("section") == "model"
        and consolidated.get("step") == STEP
        and consolidated.get("n_shards") == WORLD_SIZE
        and consolidated.get("n_keys") == 923
        and consolidated.get("run_dir") == str(SOURCE_PINNED_DIR)
        and consolidated.get("shard_files") == expected_shards
    ):
        raise RecoveryError("consolidated checkpoint metadata changed")
    return {
        **identity,
        "global_step": STEP,
        "config_hash": payload["config_hash"],
        "n_model_keys": len(payload["model"]),
        "consolidated": copy.deepcopy(dict(consolidated)),
    }


def _checkpoint_report_identity() -> dict[str, Any]:
    identity = _checked_identity(
        SOURCE_CHECKPOINT_REPORT, SOURCE_REPORT_SHA256,
        SOURCE_IDENTITIES["checkpoint_report"][2], label="checkpoint report",
    )
    try:
        report = json.loads(SOURCE_CHECKPOINT_REPORT.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RecoveryError("historical checkpoint report is unreadable") from exc
    checks = report.get("checks") if isinstance(report, Mapping) else None
    if not (
        report.get("pass") is True
        and report.get("out") == str(SOURCE_CHECKPOINT)
        and report.get("embodiment") == "libero_franka"
        and isinstance(checks, Mapping)
        and set(checks) == {"structure", "numeric", "policy"}
        and all(checks[name].get("pass") is True for name in checks)
    ):
        raise RecoveryError("historical checkpoint verification did not PASS")
    return {**identity, "pass": True, "checks": sorted(checks)}


def _prior_failure_identity() -> dict[str, Any]:
    identity = _checked_identity(
        SOURCE_CONSOLIDATE_LOG, SOURCE_LOG_SHA256,
        SOURCE_IDENTITIES["consolidate_log"][2], label="consolidation log",
    )
    text = SOURCE_CONSOLIDATE_LOG.read_text()
    passed = "[verify] OVERALL PASS"
    rejected = (
        "OPERATOR_REPAIR_INVALID: fixed endpoint receipt failed immutable "
        "shape/plan closure"
    )
    if text.count(passed) != 1 or text.count(rejected) != 1:
        raise RecoveryError("historical consolidation log failure signature changed")
    if text.index(passed) >= text.index(rejected):
        raise RecoveryError("historical validator failed before checkpoint verification")
    return {
        **identity,
        "checkpoint_verification_pass_preceded_validator_failure": True,
        "failure": "old_validator_expected_zero_padded_rank_names",
        "job_id": SOURCE_JOB_IDS["consolidate"], "exit_code": "2:0",
    }


def _check_source_protected_absence() -> list[str]:
    present = [str(path) for path in SOURCE_PROTECTED_ABSENT if path.exists()]
    if present:
        raise RecoveryError(
            "historical failed path unexpectedly acquired downstream outputs: "
            + repr(present)
        )
    return [str(path) for path in SOURCE_PROTECTED_ABSENT]


def _source_job_identity() -> dict[str, Any]:
    jobs_identity = _checked_identity(
        SOURCE_CONTROL_DIR / "jobs.json", SOURCE_JOBS_SHA256,
        SOURCE_IDENTITIES["jobs"][2], label="jobs",
    )
    release_identity = _checked_identity(
        SOURCE_CONTROL_DIR / "released.json", SOURCE_RELEASE_SHA256,
        SOURCE_IDENTITIES["release"][2], label="release",
    )
    jobs = json.loads((SOURCE_CONTROL_DIR / "jobs.json").read_text())
    release = json.loads((SOURCE_CONTROL_DIR / "released.json").read_text())
    if (
        jobs.get("jobs") != SOURCE_JOB_IDS
        or jobs.get("plan_sha256") != SOURCE_PLAN_SHA256
        or jobs.get("decision_gate_jobs") != []
        or release.get("released") is not True
        or release.get("job_ids") != list(SOURCE_JOB_IDS.values())
        or release.get("jobs_sha256") != SOURCE_JOBS_SHA256
    ):
        raise RecoveryError("historical scheduler receipt changed")
    return {
        "jobs": jobs_identity, "release": release_identity,
        "job_ids": dict(SOURCE_JOB_IDS),
        "recorded_terminal_states": dict(SOURCE_TERMINAL_STATES),
    }


def _collect_source_trigger(*, rehash_shards: bool) -> dict[str, Any]:
    source_plan = _load_source_plan()
    identities = {}
    for label in (
        "run_config", "metrics", "latest", "wandb_id", "final_asset",
    ):
        path, digest, size = SOURCE_IDENTITIES[label]
        identities[label] = _checked_identity(path, digest, size, label=label)
    endpoint = _source_endpoint(rehash_shards=rehash_shards)
    return {
        "kind": "authenticated_operator_repair_step32000_eval_recovery_trigger_v2",
        "authorization_scope": "evaluate_existing_checkpoint_no_retraining",
        "source_plan": _identity(SOURCE_PLAN),
        "historical_git_closure": _historical_git_closure(source_plan),
        "validator_repair": _validator_repair_receipt(source_plan),
        "scheduler": _source_job_identity(),
        "fixed_endpoint": endpoint["receipt"],
        "fixed_endpoint_identity": endpoint["identity"],
        "source_files": identities,
        "checkpoint_report": _checkpoint_report_identity(),
        "checkpoint": _checkpoint_payload_identity(),
        "prior_failure": _prior_failure_identity(),
        "protected_downstream_paths_absent": _check_source_protected_absence(),
        "shard_content_rehashed_when_plan_was_built": True,
        "source_bytes_are_read_only": True,
    }


def _assert_source_trigger(plan: Mapping[str, Any], *, rehash_shards: bool) -> None:
    frozen = plan.get("trigger")
    if not isinstance(frozen, Mapping):
        raise RecoveryError("recovery plan omitted source trigger")
    current = _collect_source_trigger(rehash_shards=rehash_shards)
    if current != dict(frozen):
        raise RecoveryError("historical source or checkpoint changed")


def _clean_absolute(path: Path, *, field: str) -> Path:
    if not path.is_absolute():
        raise RecoveryError(f"{field} must be absolute")
    if any(character in str(path) for character in (",", "\n", "\r")):
        raise RecoveryError(f"{field} contains an unsafe scheduler delimiter")
    return path


def _require_isolated(control_dir: Path, artifact_root: Path) -> None:
    control_dir = _clean_absolute(control_dir, field="control-dir")
    artifact_root = _clean_absolute(artifact_root, field="artifact-root")
    if (
        control_dir == artifact_root
        or control_dir.is_relative_to(artifact_root)
        or artifact_root.is_relative_to(control_dir)
    ):
        raise RecoveryError("recovery control/artifact roots must be mutually disjoint")
    for candidate in (control_dir, artifact_root):
        operator._reject_existing_symlink_components(candidate)
        for source in (SOURCE_RUN_DIR, SOURCE_CONTROL_DIR, SOURCE_ARTIFACT_ROOT):
            if (
                candidate == source or candidate.is_relative_to(source)
                or source.is_relative_to(candidate)
            ):
                raise RecoveryError(f"recovery output overlaps immutable source {source}")


def _expected_paths(control_dir: Path, artifact_root: Path) -> dict[str, Any]:
    return {
        "training_asset_verification_dir": str(
            control_dir / "compatibility_training_asset_verification"
        ),
        "training_asset_failure": str(control_dir / "TRAINING_ASSET_FAILURE.json"),
        "fixed_endpoint": str(control_dir / "adopted_fixed_endpoint_32000.json"),
        "jobs": str(control_dir / "jobs.json"),
        # The checkpoint is an immutable source input.  Every other artifact path
        # below is fresh and contained by the v2 roots.
        "checkpoint": str(SOURCE_CHECKPOINT),
        "checkpoint_report": str(control_dir / "adopted_checkpoint_verification.json"),
        "checkpoint_receipt": str(control_dir / "adopted_checkpoint_receipt.json"),
        "checkpoint_adoption_receipt": str(
            control_dir / "checkpoint_adoption_v2.json"
        ),
        "merged_results": str(artifact_root / "eval/merged/results.json"),
        "merged_table": str(artifact_root / "eval/merged/table.md"),
        "merged_receipt": str(control_dir / "merged_eval_receipt.json"),
        "completion_receipt": str(control_dir / "completion_v2.json"),
        "eval": {
            str(seed): {
                "out_dir": str(artifact_root / "eval" / f"seed_{seed}"),
                "receipt": str(control_dir / f"eval_seed_{seed}_receipt.json"),
                "source_pending": str(
                    control_dir / f"eval_seed_{seed}_source_pending_v2.json"
                ),
                "source_complete": str(
                    control_dir / f"eval_seed_{seed}_source_complete_v2.json"
                ),
            }
            for seed in SEEDS
        },
    }


def _require_paths_contained(plan: Mapping[str, Any]) -> None:
    lineage = plan["lineage"]
    control = Path(lineage["control_dir"])
    artifact = Path(lineage["artifact_root"])
    paths = plan["paths"]
    control_outputs = [
        paths["training_asset_verification_dir"], paths["training_asset_failure"],
        paths["fixed_endpoint"], paths["jobs"], paths["checkpoint_report"],
        paths["checkpoint_receipt"], paths["checkpoint_adoption_receipt"],
        paths["merged_receipt"], paths["completion_receipt"],
        *(paths["eval"][str(seed)]["receipt"] for seed in SEEDS),
        *(paths["eval"][str(seed)]["source_pending"] for seed in SEEDS),
        *(paths["eval"][str(seed)]["source_complete"] for seed in SEEDS),
    ]
    artifact_outputs = [
        paths["merged_results"], paths["merged_table"],
        *(paths["eval"][str(seed)]["out_dir"] for seed in SEEDS),
    ]
    for root, values in ((control, control_outputs), (artifact, artifact_outputs)):
        for value in values:
            path = Path(value)
            operator._reject_existing_symlink_components(path)
            if not path.resolve().is_relative_to(root.resolve()):
                raise RecoveryError(f"recovery output escapes isolated root {root}: {path}")
    if Path(paths["checkpoint"]).resolve() != SOURCE_CHECKPOINT:
        raise RecoveryError("recovery checkpoint input changed")


def _new_wandb(group: str, source_plan: Mapping[str, Any]) -> dict[str, Any]:
    stages = (
        "recovery-checkpoint", "eval-seed-0", "eval-seed-1",
        "eval-seed-2", "eval-summary",
    )
    return {
        "project": PROJECT, "group": group, "require_online": True,
        "tags": list(operator.EXPECTED_STAGE_TAGS),
        "training_run_id": source_plan["wandb"]["training_run_id"],
        "stage_run_ids": {stage: uuid.uuid4().hex[:16] for stage in stages},
        "artifact_policy": "receipts_and_eval_results_only_no_checkpoint_upload",
        "checkpoint_bytes_uploaded": False,
    }


def build_plan(
    *, control_dir: Path, artifact_root: Path, group: str = GROUP,
    project: str = PROJECT,
) -> dict[str, Any]:
    control_dir, artifact_root = control_dir.resolve(), artifact_root.resolve()
    _require_isolated(control_dir, artifact_root)
    if control_dir.exists() or artifact_root.exists():
        raise RecoveryError("recovery output roots must be fresh")
    if group != GROUP or project != PROJECT:
        raise RecoveryError("recovery W&B project/group differs from frozen identity")
    source_plan = _load_source_plan()
    trigger = _collect_source_trigger(rehash_shards=True)
    runtime_closure = _runtime_source_closure()
    return {
        "format_version": FORMAT_VERSION, "kind": KIND,
        "eligibility": copy.deepcopy(EXPECTED_ELIGIBILITY),
        "method": copy.deepcopy(EXPECTED_METHOD),
        # Compatibility key consumed by the reused operator merge provenance.
        "source_closure": runtime_closure,
        "recovery_source_closure": runtime_closure,
        "recovery_host_runtime": _recovery_host_runtime_receipt(),
        "historical_source_closure": copy.deepcopy(
            source_plan["source_closure"]
        ),
        "trigger": trigger,
        "assets": copy.deepcopy(source_plan["assets"]),
        "config": copy.deepcopy(source_plan["config"]),
        "lineage": {
            "run_name": source_plan["lineage"]["run_name"],
            # Compatibility aliases consumed by current eval/checkpoint helpers.
            "run_dir": str(SOURCE_RUN_DIR),
            "control_dir": str(control_dir),
            "artifact_root": str(artifact_root),
            "source_run_dir": str(SOURCE_RUN_DIR),
            "source_control_dir": str(SOURCE_CONTROL_DIR),
            "source_artifact_root": str(SOURCE_ARTIFACT_ROOT),
            "recovery_control_dir": str(control_dir),
            "recovery_artifact_root": str(artifact_root),
        },
        "schedule": {
            "training_jobs": 0, "consolidation_jobs": 0,
            "checkpoint_adoption_jobs": 1, "seed_eval_jobs": 3,
            "merge_jobs": 1, "decision_gate_jobs": [],
            "fixed_endpoint": STEP,
        },
        "evaluation": copy.deepcopy(source_plan["evaluation"]),
        "baseline_comparison": copy.deepcopy(source_plan["baseline_comparison"]),
        "wandb": _new_wandb(group, source_plan),
        "failure_policy": {
            "only_integrity_or_execution_failure_blocks_afterok": True,
            "performance_outcome_never_blocks_publication": True,
            "partial_evaluation": "operator_content_addressed_resume",
            "partial_merge": "operator_exact_recompute_then_adopt",
            "source_mutation": "reject",
        },
        "paths": _expected_paths(control_dir, artifact_root),
    }


def _assert_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("format_version") != FORMAT_VERSION or plan.get("kind") != KIND:
        raise RecoveryError("unsupported recovery plan")
    runtime = _runtime_source_closure()
    if (
        plan.get("source_closure") != runtime
        or plan.get("recovery_source_closure") != runtime
    ):
        raise RecoveryError("recovery executable source closure changed")
    if plan.get("recovery_host_runtime") != _recovery_host_runtime_receipt():
        raise RecoveryError("recovery host runtime changed")
    source_plan = _load_source_plan()
    if plan.get("historical_source_closure") != source_plan["source_closure"]:
        raise RecoveryError("historical source closure changed in recovery plan")
    if plan.get("eligibility") != EXPECTED_ELIGIBILITY or plan.get("method") != EXPECTED_METHOD:
        raise RecoveryError("recovery no-training/no-gate contract changed")
    if plan.get("assets") != source_plan["assets"] or plan.get("config") != source_plan["config"]:
        raise RecoveryError("recovery source config/assets changed")
    if (
        plan.get("evaluation") != source_plan["evaluation"]
        or plan.get("baseline_comparison") != source_plan["baseline_comparison"]
    ):
        raise RecoveryError("recovery evaluation/baseline protocol changed")
    lineage = plan.get("lineage", {})
    control = Path(lineage.get("control_dir", "")).resolve()
    artifact = Path(lineage.get("artifact_root", "")).resolve()
    if not (
        lineage.get("run_name") == source_plan["lineage"]["run_name"]
        and Path(lineage.get("run_dir", "")).resolve() == SOURCE_RUN_DIR
        and Path(lineage.get("source_run_dir", "")).resolve() == SOURCE_RUN_DIR
        and Path(lineage.get("source_control_dir", "")).resolve() == SOURCE_CONTROL_DIR
        and Path(lineage.get("source_artifact_root", "")).resolve()
        == SOURCE_ARTIFACT_ROOT
        and Path(lineage.get("recovery_control_dir", "")).resolve() == control
        and Path(lineage.get("recovery_artifact_root", "")).resolve() == artifact
    ):
        raise RecoveryError("recovery lineage changed")
    _require_isolated(control, artifact)
    if plan.get("paths") != _expected_paths(control, artifact):
        raise RecoveryError("recovery derived paths changed")
    _require_paths_contained(plan)
    if plan.get("schedule") != {
        "training_jobs": 0, "consolidation_jobs": 0,
        "checkpoint_adoption_jobs": 1, "seed_eval_jobs": 3,
        "merge_jobs": 1, "decision_gate_jobs": [], "fixed_endpoint": STEP,
    }:
        raise RecoveryError("recovery evaluation-only DAG contract changed")
    if plan.get("failure_policy") != {
        "only_integrity_or_execution_failure_blocks_afterok": True,
        "performance_outcome_never_blocks_publication": True,
        "partial_evaluation": "operator_content_addressed_resume",
        "partial_merge": "operator_exact_recompute_then_adopt",
        "source_mutation": "reject",
    }:
        raise RecoveryError("recovery failure policy changed")
    wandb = plan.get("wandb", {})
    ids = wandb.get("stage_run_ids", {})
    source_ids = {
        source_plan["wandb"]["training_run_id"],
        *source_plan["wandb"]["stage_run_ids"].values(),
    }
    if not (
        wandb.get("project") == PROJECT and wandb.get("group") == GROUP
        and wandb.get("require_online") is True
        and wandb.get("tags") == list(operator.EXPECTED_STAGE_TAGS)
        and wandb.get("training_run_id") == source_plan["wandb"]["training_run_id"]
        and wandb.get("artifact_policy")
        == "receipts_and_eval_results_only_no_checkpoint_upload"
        and wandb.get("checkpoint_bytes_uploaded") is False
        and set(ids) == {
            "recovery-checkpoint", "eval-seed-0", "eval-seed-1",
            "eval-seed-2", "eval-summary",
        }
        and len(set(ids.values())) == 5
        and all(re.fullmatch(r"[0-9a-f]{16}", str(value)) for value in ids.values())
        and set(ids.values()).isdisjoint(source_ids)
    ):
        raise RecoveryError("recovery W&B identity changed")
    _assert_source_trigger(plan, rehash_shards=False)


def load_plan(path: str | Path, expected_sha256: str | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    if expected_sha256 and sha256_file(path) != expected_sha256:
        raise RecoveryError("recovery plan SHA-256 mismatch")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RecoveryError("recovery plan is unreadable") from exc
    if not isinstance(value, Mapping):
        raise RecoveryError("recovery plan is not an object")
    plan = dict(value)
    _assert_plan(plan)
    return plan


def _recovery_plan_sha() -> str:
    value = os.environ.get("OPERATOR_RECOVERY_PLAN")
    if not value:
        raise RecoveryError("OPERATOR_RECOVERY_PLAN is required")
    return sha256_file(value)


def _bind_operator_plan(plan_path: Path, digest: str) -> None:
    for name, expected in (
        ("OPERATOR_REPAIR_PLAN", str(plan_path)),
        ("OPERATOR_REPAIR_PLAN_SHA256", digest),
    ):
        existing = os.environ.get(name)
        if existing not in (None, expected):
            raise RecoveryError(f"{name} conflicts with the recovery plan")
        os.environ[name] = expected


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RecoveryError(f"{label} is absent/nonregular")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RecoveryError(f"{label} is unreadable") from exc
    if not isinstance(value, Mapping):
        raise RecoveryError(f"{label} is not an object")
    return dict(value)


def _publish_exact(path: Path, payload: Mapping[str, Any], *, label: str) -> None:
    expected = _pretty_json(payload)
    if path.exists():
        if not path.is_file() or path.is_symlink() or path.read_text() != expected:
            raise RecoveryError(f"existing {label} differs from exact recomputation")
        return
    operator.common.exclusive_json_write(path, payload)


def _compatibility_asset_payload(
    plan: Mapping[str, Any], *, assets: Mapping[str, Any],
) -> dict[str, Any]:
    if dict(assets) != plan["assets"]:
        raise RecoveryError("live training assets differ from the frozen plan")
    return {
        "format_version": operator.FORMAT_VERSION,
        "kind": "r0_e2e_operator_repair_training_asset_verification",
        "plan_sha256": _recovery_plan_sha(),
        "stage": f"train_{operator.TRAIN_LINKS:02d}", "phase": "post",
        "assets": copy.deepcopy(dict(assets)),
        "raw_and_cache_full_semantic_content_rehashed": True,
        "controls_training_integrity_not_scientific_selection": True,
    }


def _compatibility_endpoint_payload(plan: Mapping[str, Any]) -> dict[str, Any]:
    endpoint = copy.deepcopy(plan["trigger"]["fixed_endpoint"])
    endpoint["plan_sha256"] = _recovery_plan_sha()
    asset_path = operator._training_asset_verification_path(
        plan, stage=f"train_{operator.TRAIN_LINKS:02d}", phase="post",
    )
    endpoint["training_asset_verification"] = {
        "stage": f"train_{operator.TRAIN_LINKS:02d}", "phase": "post",
        "path": str(asset_path), "sha256": sha256_file(asset_path),
    }
    endpoint["evaluation_recovery_adoption"] = {
        "kind": "fixed_endpoint_receipt_adoption_after_validator_bug_v2",
        "source_receipt": copy.deepcopy(
            plan["trigger"]["fixed_endpoint_identity"]
        ),
        "source_plan_sha256": SOURCE_PLAN_SHA256,
        "source_checkpoint_shard_names_are_real_unpadded_names": True,
        "source_bytes_rewritten": False,
        "checkpoint_bytes_rewritten": False,
        "selection_changed": False,
    }
    return endpoint


def _publish_compatibility_receipts(plan: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    asset_dir = Path(plan["paths"]["training_asset_verification_dir"])
    operator._reject_existing_symlink_components(asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)
    asset_path = operator._training_asset_verification_path(
        plan, stage=f"train_{operator.TRAIN_LINKS:02d}", phase="post",
    )
    if asset_path.exists() or asset_path.is_symlink():
        # The adoption job already performed the expensive full raw/cache
        # semantic rehash before publishing this immutable compatibility
        # receipt.  Later seed/merge jobs validate its exact bytes only.
        assets = plan["assets"]
    else:
        try:
            assets = operator._asset_receipt()
        except operator.OperatorRepairError as exc:
            raise RecoveryError(str(exc)) from exc
        if assets != plan["assets"]:
            raise RecoveryError("training assets changed before checkpoint adoption")
    _publish_exact(
        asset_path, _compatibility_asset_payload(plan, assets=assets),
        label="compatibility training-asset receipt",
    )
    endpoint = _compatibility_endpoint_payload(plan)
    endpoint_path = Path(plan["paths"]["fixed_endpoint"])
    _publish_exact(endpoint_path, endpoint, label="adopted fixed endpoint")
    try:
        parsed, digest = operator._read_fixed_endpoint(plan)
    except operator.OperatorRepairError as exc:
        raise RecoveryError(str(exc)) from exc
    if parsed != endpoint:
        raise RecoveryError("operator validator changed the adopted endpoint")
    return endpoint, digest


def _checkpoint_adoption_payload(
    plan: Mapping[str, Any], *, operator_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION, "kind": CHECKPOINT_ADOPTION_KIND,
        "recovery_plan_sha256": _recovery_plan_sha(),
        "source_plan_sha256": SOURCE_PLAN_SHA256,
        "source_fixed_endpoint_sha256": SOURCE_ENDPOINT_SHA256,
        "source_checkpoint": copy.deepcopy(plan["trigger"]["checkpoint"]),
        "historical_checkpoint_report": copy.deepcopy(
            plan["trigger"]["checkpoint_report"]
        ),
        "operator_checkpoint_receipt": {
            "path": plan["paths"]["checkpoint_receipt"],
            "sha256": sha256_file(plan["paths"]["checkpoint_receipt"]),
            "kind": operator_receipt["kind"],
        },
        "verify_only": True, "checkpoint_reconstructed": False,
        "checkpoint_or_source_shards_written": False,
        "training_updates": 0, "optimizer_steps": 0,
        "scientific_gate": False,
    }


def _validate_checkpoint_adoption(plan: Mapping[str, Any]) -> dict[str, Any]:
    report = Path(plan["paths"]["checkpoint_report"])
    if (
        not report.is_file() or report.is_symlink()
        or report.read_bytes() != SOURCE_CHECKPOINT_REPORT.read_bytes()
    ):
        raise RecoveryError(
            "recovery checkpoint report differs from the historical verified PASS"
        )
    _checked_identity(
        SOURCE_CHECKPOINT, SOURCE_CHECKPOINT_SHA256, SOURCE_CHECKPOINT_BYTES,
        label="consolidated checkpoint",
    )
    try:
        receipt = operator._validate_checkpoint_receipt(plan, rehash_pinned=False)
    except operator.OperatorRepairError as exc:
        raise RecoveryError(str(exc)) from exc
    path = Path(plan["paths"]["checkpoint_adoption_receipt"])
    actual = _read_json(path, label="checkpoint adoption receipt")
    expected = _checkpoint_adoption_payload(plan, operator_receipt=receipt)
    if actual != expected:
        raise RecoveryError("checkpoint adoption receipt differs from recomputation")
    return actual


def _assert_recovery_execution_closure(
    plan: Mapping[str, Any], *, rehash_shards: bool,
) -> dict[str, Any]:
    source = _runtime_source_closure()
    host = _recovery_host_runtime_receipt()
    if (
        plan.get("source_closure") != source
        or plan.get("recovery_source_closure") != source
        or plan.get("recovery_host_runtime") != host
    ):
        raise RecoveryError("recovery runtime source/host closure changed")
    _assert_source_trigger(plan, rehash_shards=rehash_shards)
    return {
        "recovery_source_closure": source,
        "recovery_host_runtime": host,
        "historical_source_closure": copy.deepcopy(
            plan["historical_source_closure"]
        ),
        "source_trigger_sha256": hashlib.sha256(
            _canonical_json(plan["trigger"]).encode()
        ).hexdigest(),
    }


def _stage_adopt_checkpoint(plan: Mapping[str, Any]) -> int:
    _assert_recovery_execution_closure(plan, rehash_shards=True)
    endpoint, endpoint_sha = _publish_compatibility_receipts(plan)
    report = Path(plan["paths"]["checkpoint_report"])
    receipt_path = Path(plan["paths"]["checkpoint_receipt"])
    adoption_path = Path(plan["paths"]["checkpoint_adoption_receipt"])
    if adoption_path.exists() and not receipt_path.exists():
        raise RecoveryError("checkpoint adoption exists without its operator receipt")
    if receipt_path.exists():
        try:
            operator_receipt = operator._validate_checkpoint_receipt(
                plan, rehash_pinned=False,
            )
        except operator.OperatorRepairError as exc:
            raise RecoveryError(str(exc)) from exc
        if not adoption_path.exists():
            operator.common.exclusive_json_write(
                adoption_path,
                _checkpoint_adoption_payload(
                    plan, operator_receipt=operator_receipt,
                ),
            )
        _validate_checkpoint_adoption(plan)
        _assert_recovery_execution_closure(plan, rehash_shards=False)
        operator._wandb_publish(
            plan, stage="recovery-checkpoint", path=adoption_path,
            artifact_type="operator-repair-eval-recovery-checkpoint-receipt",
            summary={
                "checkpoint_step": STEP, "verify_only": True,
                "training_updates": 0, "scientific_gate": False,
            },
        )
        return 0
    before = _identity(SOURCE_CHECKPOINT)
    attempt_report = report.with_name(
        f".{report.name}.attempt-{os.getpid()}-{uuid.uuid4().hex}"
    )
    command = [
        sys.executable, "-m", "loom.train.consolidate",
        "--run_dir", str(SOURCE_PINNED_DIR), "--step", str(STEP),
        "--out", str(SOURCE_CHECKPOINT),
        "--config", str(SOURCE_RUN_DIR / "config.json"),
        "--verify_only", "--report", str(attempt_report),
    ]
    try:
        subprocess.run(command, cwd=ROOT, check=True)
        after = _identity(SOURCE_CHECKPOINT)
        if before != after or after["sha256"] != SOURCE_CHECKPOINT_SHA256:
            raise RecoveryError("source checkpoint changed during verify-only adoption")
        if attempt_report.read_bytes() != SOURCE_CHECKPOINT_REPORT.read_bytes():
            raise RecoveryError("fresh verify-only report differs from historical PASS")
        if report.exists():
            if not report.is_file() or report.is_symlink() or (
                report.read_bytes() != attempt_report.read_bytes()
            ):
                raise RecoveryError("canonical recovery report differs")
        else:
            operator.common._exclusive_text_write(report, attempt_report.read_text())
    finally:
        attempt_report.unlink(missing_ok=True)
    pinned = endpoint["checkpoint_shards"]
    operator_receipt = operator._checkpoint_receipt_payload(
        plan, endpoint=endpoint, endpoint_sha=endpoint_sha,
        report=report, checkpoint=SOURCE_CHECKPOINT, pinned=pinned,
    )
    operator.common.exclusive_json_write(receipt_path, operator_receipt)
    try:
        validated = operator._validate_checkpoint_receipt(plan, rehash_pinned=False)
    except operator.OperatorRepairError as exc:
        raise RecoveryError(str(exc)) from exc
    if validated != operator_receipt:
        raise RecoveryError("adopted operator checkpoint receipt changed")
    adoption = _checkpoint_adoption_payload(plan, operator_receipt=validated)
    operator.common.exclusive_json_write(adoption_path, adoption)
    _validate_checkpoint_adoption(plan)
    _assert_recovery_execution_closure(plan, rehash_shards=False)
    operator._wandb_publish(
        plan, stage="recovery-checkpoint", path=adoption_path,
        artifact_type="operator-repair-eval-recovery-checkpoint-receipt",
        summary={
            "checkpoint_step": STEP, "verify_only": True,
            "training_updates": 0, "scientific_gate": False,
        },
    )
    return 0


def _eval_source_pending_payload(
    plan: Mapping[str, Any], *, seed: int,
) -> dict[str, Any]:
    closure = _assert_recovery_execution_closure(plan, rehash_shards=False)
    _validate_checkpoint_adoption(plan)
    paths = plan["paths"]["eval"][str(seed)]
    return {
        "format_version": FORMAT_VERSION, "kind": EVAL_SOURCE_PENDING_KIND,
        "recovery_plan_sha256": _recovery_plan_sha(), "seed": seed,
        **closure,
        "checkpoint_adoption_receipt": _identity(
            Path(plan["paths"]["checkpoint_adoption_receipt"])
        ),
        "operator_checkpoint_receipt": _identity(
            Path(plan["paths"]["checkpoint_receipt"])
        ),
        "fixed_endpoint": _identity(Path(plan["paths"]["fixed_endpoint"])),
        "source_checkpoint": _identity(SOURCE_CHECKPOINT),
        "operator_seed_receipt": paths["receipt"],
        "result": str(Path(paths["out_dir"]) / "results.json"),
        "table": str(Path(paths["out_dir"]) / "table.md"),
        "control_recovery_before_attempt": _seed_control_recovery_receipt(
            plan, seed=seed,
        ),
        "post_runtime_source_authentication_required": True,
    }


def _seed_control_recovery_receipt(
    plan: Mapping[str, Any], *, seed: int,
) -> dict[str, Any]:
    recovery = Path(plan["lineage"]["control_dir"]) / "recovery"
    if not recovery.exists():
        return {"kind": "content_addressed_seed_control_recovery_v2", "files": []}
    if not recovery.is_dir() or recovery.is_symlink():
        raise RecoveryError("seed control recovery path is not a real directory")
    prefix = f"eval_seed_{seed}_"
    files = []
    for path in sorted(recovery.iterdir(), key=lambda item: item.name):
        if not path.name.startswith(prefix):
            continue
        if not path.is_file() or path.is_symlink():
            raise RecoveryError("seed control recovery contains nonregular entry")
        files.append({
            "name": path.name, "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return {"kind": "content_addressed_seed_control_recovery_v2", "files": files}


def _seed_operator_state_paths(
    plan: Mapping[str, Any], *, seed: int,
) -> tuple[Path, ...]:
    paths = plan["paths"]["eval"][str(seed)]
    out_dir = Path(paths["out_dir"])
    active, completed = operator._eval_attempt_paths(out_dir)
    return (
        out_dir / "results.json", out_dir / "table.md", active, completed,
        Path(paths["receipt"]),
    )


def _path_present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _quarantine_incomplete_seed_source_transaction(
    plan: Mapping[str, Any], *, seed: int,
) -> None:
    paths = plan["paths"]["eval"][str(seed)]
    out_dir = Path(paths["out_dir"])
    pending = Path(paths["source_pending"])
    complete = Path(paths["source_complete"])
    if _path_present(complete):
        raise RecoveryError("refusing to quarantine a published seed source completion")
    candidates = _seed_operator_state_paths(plan, seed=seed)
    for path in candidates:
        if _path_present(path) and (not path.is_file() or path.is_symlink()):
            raise RecoveryError(f"cannot quarantine nonregular seed state: {path}")
    # Output files first and the durable pending marker last.  If any move
    # fails, pending remains and forces the same quarantine transaction on
    # every replay rather than allowing an uncertain result to be adopted.
    try:
        operator._quarantine_eval_attempt_outputs(
            out_dir, reason="missing_runtime_source_completion",
        )
        receipt = Path(paths["receipt"])
        if _path_present(receipt):
            operator._quarantine_eval_file(
                receipt, reason="missing_runtime_source_completion",
            )
        if not _path_present(pending):
            raise RecoveryError("seed source pending marker disappeared during quarantine")
        if not pending.is_file() or pending.is_symlink():
            raise RecoveryError("seed source pending marker is nonregular")
        operator._quarantine_eval_file(
            pending, reason="missing_runtime_source_completion",
        )
    except operator.OperatorRepairError as exc:
        raise RecoveryError(str(exc)) from exc


def _validate_eval_source_pending(
    plan: Mapping[str, Any], *, seed: int,
) -> dict[str, Any]:
    path = Path(plan["paths"]["eval"][str(seed)]["source_pending"])
    actual = _read_json(path, label=f"seed {seed} source pending receipt")
    expected = _eval_source_pending_payload(plan, seed=seed)
    if actual != expected:
        raise RecoveryError(f"seed {seed} source pending receipt changed")
    return actual


def _prepare_eval_source_transaction(
    plan: Mapping[str, Any], *, seed: int,
) -> str:
    paths = plan["paths"]["eval"][str(seed)]
    pending = Path(paths["source_pending"])
    complete = Path(paths["source_complete"])
    pending_present = _path_present(pending)
    complete_present = _path_present(complete)
    if complete_present:
        if not pending_present:
            raise RecoveryError(f"seed {seed} source completion is orphaned")
        _validate_eval_source_completion(plan, seed=seed)
        return "COMPLETE"
    orphan_operator_state = any(
        _path_present(path) for path in _seed_operator_state_paths(plan, seed=seed)
    )
    if not pending_present:
        # Publish a durable transaction marker before touching any orphaned
        # state so an interrupted quarantine is forced to resume fail-closed.
        _publish_exact(
            pending, _eval_source_pending_payload(plan, seed=seed),
            label=f"seed {seed} source pending receipt",
        )
    if pending_present or orphan_operator_state:
        _quarantine_incomplete_seed_source_transaction(plan, seed=seed)
        _publish_exact(
            pending, _eval_source_pending_payload(plan, seed=seed),
            label=f"seed {seed} replacement source pending receipt",
        )
    _validate_eval_source_pending(plan, seed=seed)
    return "PENDING"


def _eval_source_complete_payload(
    plan: Mapping[str, Any], *, seed: int,
) -> dict[str, Any]:
    closure = _assert_recovery_execution_closure(plan, rehash_shards=False)
    pending = _validate_eval_source_pending(plan, seed=seed)
    try:
        operator_receipt = operator._validate_eval_receipt(plan, seed)
    except operator.OperatorRepairError as exc:
        raise RecoveryError(str(exc)) from exc
    paths = plan["paths"]["eval"][str(seed)]
    result = Path(paths["out_dir"]) / "results.json"
    table = Path(paths["out_dir"]) / "table.md"
    return {
        "format_version": FORMAT_VERSION, "kind": EVAL_SOURCE_COMPLETE_KIND,
        "recovery_plan_sha256": _recovery_plan_sha(), "seed": seed,
        **closure,
        "source_pending": _identity(Path(paths["source_pending"])),
        "source_pending_kind": pending["kind"],
        "checkpoint_adoption_receipt": _identity(
            Path(plan["paths"]["checkpoint_adoption_receipt"])
        ),
        "operator_seed_receipt": _identity(Path(paths["receipt"])),
        "operator_seed_receipt_kind": operator_receipt["kind"],
        "result": _identity(result), "table": _identity(table),
        "control_recovery": _seed_control_recovery_receipt(plan, seed=seed),
        "post_runtime_source_reauthenticated": True,
    }


def _validate_eval_source_completion(
    plan: Mapping[str, Any], *, seed: int,
) -> dict[str, Any]:
    path = Path(plan["paths"]["eval"][str(seed)]["source_complete"])
    actual = _read_json(path, label=f"seed {seed} source completion receipt")
    expected = _eval_source_complete_payload(plan, seed=seed)
    if actual != expected:
        raise RecoveryError(f"seed {seed} source completion differs from recomputation")
    return actual


def _capture_operator_eval_without_wandb(
    plan: Mapping[str, Any], *, stage: str, seed: int,
) -> int:
    publisher = operator._wandb_publish
    deferred: list[tuple[Mapping[str, Any], dict[str, Any]]] = []

    def capture(stage_plan: Mapping[str, Any], **kwargs: Any) -> None:
        deferred.append((stage_plan, dict(kwargs)))

    operator._wandb_publish = capture
    try:
        result = operator._stage_eval(plan, stage)
    finally:
        operator._wandb_publish = publisher
    paths = plan["paths"]["eval"][str(seed)]
    if (
        result != 0 or len(deferred) != 1 or deferred[0][0] != plan
        or deferred[0][1].get("stage") != f"eval-seed-{seed}"
        or Path(deferred[0][1].get("path", ""))
        != Path(paths["out_dir"]) / "results.json"
    ):
        raise RecoveryError("operator seed stage changed its publication contract")
    return result


def _publish_seed_wandb(plan: Mapping[str, Any], *, seed: int) -> None:
    try:
        receipt = operator._validate_eval_receipt(plan, seed)
    except operator.OperatorRepairError as exc:
        raise RecoveryError(str(exc)) from exc
    result = Path(plan["paths"]["eval"][str(seed)]["out_dir"]) / "results.json"
    operator._wandb_publish(
        plan, stage=f"eval-seed-{seed}", path=result,
        artifact_type="operator-repair-evaluation-results",
        summary={
            "seed": seed, "episodes": operator.common.EXPECTED_EPISODES_PER_SEED,
            "success_rate": receipt["avg"], "checkpoint_step": STEP,
            "evaluation_unconditional": True,
        },
    )


def _stage_eval(plan: Mapping[str, Any], stage: str) -> int:
    if re.fullmatch(r"eval_seed[0-2]", stage) is None:
        raise RecoveryError(f"invalid recovery seed stage {stage!r}")
    seed = int(stage.removeprefix("eval_seed"))
    _assert_recovery_execution_closure(plan, rehash_shards=False)
    _publish_compatibility_receipts(plan)
    _validate_checkpoint_adoption(plan)
    state = _prepare_eval_source_transaction(plan, seed=seed)
    if state == "COMPLETE":
        _publish_seed_wandb(plan, seed=seed)
        return 0
    try:
        result = _capture_operator_eval_without_wandb(
            plan, stage=stage, seed=seed,
        )
    except operator.OperatorRepairError as exc:
        raise RecoveryError(str(exc)) from exc
    complete = Path(plan["paths"]["eval"][str(seed)]["source_complete"])
    try:
        payload = _eval_source_complete_payload(plan, seed=seed)
    except BaseException as authentication_error:
        try:
            _quarantine_incomplete_seed_source_transaction(plan, seed=seed)
        except BaseException as quarantine_error:
            raise RecoveryError(
                "seed runtime-source authentication failed and uncertain outputs "
                "could not be quarantined"
            ) from authentication_error
        raise
    _publish_exact(
        complete, payload, label=f"seed {seed} source completion receipt",
    )
    _validate_eval_source_completion(plan, seed=seed)
    _publish_seed_wandb(plan, seed=seed)
    return result


def _completion_payload(plan: Mapping[str, Any]) -> dict[str, Any]:
    closure = _assert_recovery_execution_closure(plan, rehash_shards=False)
    for seed in SEEDS:
        _validate_eval_source_completion(plan, seed=seed)
    merged_receipt = _read_json(
        Path(plan["paths"]["merged_receipt"]), label="merged evaluation receipt",
    )
    result = Path(plan["paths"]["merged_results"])
    table = Path(plan["paths"]["merged_table"])
    blob = _read_json(result, label="merged evaluation result")
    summary = blob.get("summary", {})
    if not (
        merged_receipt.get("episodes") == 1_200
        and merged_receipt.get("errors") == 0
        and merged_receipt.get("complete") is True
        and summary.get("n_episodes") == 1_200
        and summary.get("n_errors") == 0
        and summary.get("complete") is True
    ):
        raise RecoveryError("merged evaluation is not exact complete 1,200")
    successes = sum(
        int(row.get("success") is True) for row in blob.get("episodes", [])
    )
    if len(blob.get("episodes", [])) != 1_200:
        raise RecoveryError("merged episode rows are not exact 1,200")
    return {
        "format_version": FORMAT_VERSION, "kind": COMPLETION_KIND,
        "recovery_plan_sha256": _recovery_plan_sha(),
        "source_plan_sha256": SOURCE_PLAN_SHA256,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "recovery_source_closure_sha256": closure[
            "recovery_source_closure"
        ]["sha256"],
        "recovery_host_runtime_sha256": hashlib.sha256(
            _canonical_json(closure["recovery_host_runtime"]).encode()
        ).hexdigest(),
        "checkpoint_adoption_receipt_sha256": sha256_file(
            plan["paths"]["checkpoint_adoption_receipt"]
        ),
        "seed_source_completion_sha256": {
            str(seed): sha256_file(
                plan["paths"]["eval"][str(seed)]["source_complete"]
            )
            for seed in SEEDS
        },
        "seed_eval_receipt_sha256": {
            str(seed): sha256_file(plan["paths"]["eval"][str(seed)]["receipt"])
            for seed in SEEDS
        },
        "merged_receipt": {
            "path": plan["paths"]["merged_receipt"],
            "sha256": sha256_file(plan["paths"]["merged_receipt"]),
        },
        "result": {"path": str(result), "sha256": sha256_file(result)},
        "table": {"path": str(table), "sha256": sha256_file(table)},
        "episodes": 1_200, "errors": 0, "successes": successes,
        "end_to_end_success_rate_percent": summary["avg"],
        "training_updates": 0, "optimizer_steps": 0,
        "checkpoint_reconstructions": 0, "scientific_gates": 0,
        "outcome_threshold_applied": False,
        "promotion_authority": False,
    }


def _validate_completion(plan: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(plan["paths"]["completion_receipt"])
    actual = _read_json(path, label="recovery completion receipt")
    expected = _completion_payload(plan)
    if actual != expected:
        raise RecoveryError("recovery completion receipt differs from recomputation")
    return actual


def _capture_operator_merge_without_wandb(plan: Mapping[str, Any]) -> int:
    publisher = operator._wandb_publish
    deferred: list[tuple[Mapping[str, Any], dict[str, Any]]] = []

    def capture(stage_plan: Mapping[str, Any], **kwargs: Any) -> None:
        deferred.append((stage_plan, dict(kwargs)))

    operator._wandb_publish = capture
    try:
        result = operator._stage_merge(plan)
    finally:
        operator._wandb_publish = publisher
    if (
        result != 0 or len(deferred) != 1 or deferred[0][0] != plan
        or deferred[0][1].get("stage") != "eval-summary"
        or Path(deferred[0][1].get("path", ""))
        != Path(plan["paths"]["merged_results"])
    ):
        raise RecoveryError("operator merge changed its publication contract")
    return result


def _publish_merge_wandb(plan: Mapping[str, Any]) -> None:
    try:
        receipt = operator._validate_merged_receipt(plan)
    except operator.OperatorRepairError as exc:
        raise RecoveryError(str(exc)) from exc
    merged = _read_json(
        Path(plan["paths"]["merged_results"]), label="merged evaluation result",
    )
    operator._wandb_publish(
        plan, stage="eval-summary", path=Path(plan["paths"]["merged_results"]),
        artifact_type="operator-repair-evaluation-results",
        summary={
            "episodes": 1_200, "success_rate": receipt["avg"], "n_errors": 0,
            **operator._summary_fields(merged),
        },
    )


def _stage_merge(plan: Mapping[str, Any]) -> int:
    _assert_recovery_execution_closure(plan, rehash_shards=False)
    _publish_compatibility_receipts(plan)
    _validate_checkpoint_adoption(plan)
    for seed in SEEDS:
        _validate_eval_source_completion(plan, seed=seed)
    try:
        _capture_operator_merge_without_wandb(plan)
    except operator.OperatorRepairError as exc:
        raise RecoveryError(str(exc)) from exc
    completion_path = Path(plan["paths"]["completion_receipt"])
    if completion_path.exists():
        completion = _validate_completion(plan)
    else:
        completion = _completion_payload(plan)
        operator.common.exclusive_json_write(completion_path, completion)
        _validate_completion(plan)
    _assert_recovery_execution_closure(plan, rehash_shards=False)
    _publish_merge_wandb(plan)
    print(json.dumps(completion, indent=2, sort_keys=True), flush=True)
    return 0


def _stage_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "adopt_checkpoint",
            "sbatch": "scripts/r0_e2e_operator_repair_eval_recovery_checkpoint.sbatch",
            "depends_on": [],
        },
        *[
            {
                "name": f"eval_seed{seed}",
                "sbatch": "scripts/r0_e2e_operator_repair_eval_recovery_seed.sbatch",
                "depends_on": ["adopt_checkpoint"],
            }
            for seed in SEEDS
        ],
        {
            "name": "merge",
            "sbatch": "scripts/r0_e2e_operator_repair_eval_recovery_control.sbatch",
            "depends_on": [f"eval_seed{seed}" for seed in SEEDS],
        },
    ]


def _sbatch_command(
    *, spec: Mapping[str, Any], plan_path: Path, plan_sha: str,
    dependencies: Sequence[str],
) -> list[str]:
    label = re.sub(
        r"[^A-Za-z0-9_-]", "_", f"r0repair_recovery_v2_{spec['name']}"
    )[:120]
    command = [
        "sbatch", "--parsable", "--hold", "--kill-on-invalid-dep=yes",
        f"--job-name={label}",
    ]
    if dependencies:
        command.append("--dependency=afterok:" + ":".join(dependencies))
    export = ",".join((
        "ALL", f"OPERATOR_RECOVERY_PLAN={plan_path}",
        f"OPERATOR_RECOVERY_PLAN_SHA256={plan_sha}",
        f"OPERATOR_RECOVERY_STAGE={spec['name']}",
    ))
    command.extend((f"--export={export}", str(ROOT / str(spec["sbatch"]))))
    return command


def _parse_job_id(stdout: str) -> str:
    value = stdout.strip().split(";", 1)[0]
    if re.fullmatch(r"[0-9]+(?:_[0-9]+)?", value) is None:
        raise RecoveryError(f"invalid recovery sbatch job id: {stdout!r}")
    return value


def submit_plan(
    plan: Mapping[str, Any], *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    _assert_plan(plan)
    control = Path(plan["lineage"]["control_dir"])
    artifact = Path(plan["lineage"]["artifact_root"])
    if control.exists() or artifact.exists():
        raise RecoveryError("recovery output roots are not fresh")
    control.mkdir(parents=True, exist_ok=False)
    artifact.mkdir(parents=True, exist_ok=False)
    plan_path = control / "plan.json"
    operator.common.exclusive_json_write(plan_path, plan)
    plan_sha = sha256_file(plan_path)
    jobs: dict[str, str] = {}
    commands: dict[str, list[str]] = {}
    submitted: list[str] = []
    try:
        for spec in _stage_specs():
            command = _sbatch_command(
                spec=spec, plan_path=plan_path, plan_sha=plan_sha,
                dependencies=[jobs[name] for name in spec["depends_on"]],
            )
            completed = run(
                command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, check=True,
            )
            job_id = _parse_job_id(completed.stdout)
            jobs[str(spec["name"])] = job_id
            commands[str(spec["name"])] = command
            submitted.append(job_id)
        receipt = {
            "format_version": FORMAT_VERSION,
            "kind": "r0_e2e_operator_repair_eval_recovery_jobs_v2",
            "plan": str(plan_path), "plan_sha256": plan_sha,
            "jobs": jobs, "commands": commands, "released": False,
            "training_jobs": [], "consolidation_jobs": [],
            "decision_gate_jobs": [], "fixed_endpoint": STEP,
        }
        operator.common.exclusive_json_write(control / "jobs.json", receipt)
        run(
            ["scontrol", "release", ",".join(submitted)], cwd=ROOT,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
        )
        operator.common.exclusive_json_write(control / "released.json", {
            "format_version": FORMAT_VERSION,
            "kind": "r0_e2e_operator_repair_eval_recovery_release_v2",
            "plan_sha256": plan_sha,
            "jobs_sha256": sha256_file(control / "jobs.json"),
            "job_ids": submitted, "released": True,
        })
        return {**receipt, "released": True}
    except Exception:
        if submitted:
            run(
                ["scancel", *submitted], cwd=ROOT, check=False, text=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        raise


def _required_plan() -> tuple[dict[str, Any], str, Path, str]:
    value = os.environ.get("OPERATOR_RECOVERY_PLAN")
    digest = os.environ.get("OPERATOR_RECOVERY_PLAN_SHA256")
    stage = os.environ.get("OPERATOR_RECOVERY_STAGE")
    if not value or not digest or not stage:
        raise RecoveryError(
            "OPERATOR_RECOVERY_PLAN, OPERATOR_RECOVERY_PLAN_SHA256, and "
            "OPERATOR_RECOVERY_STAGE are required"
        )
    _require_active_recovery_host_runtime()
    path = Path(value).resolve()
    plan = load_plan(path, digest)
    _bind_operator_plan(path, digest)
    return plan, stage, path, digest


def run_stage() -> int:
    plan, stage, _path, _digest = _required_plan()
    if stage == "adopt_checkpoint":
        return _stage_adopt_checkpoint(plan)
    if stage.startswith("eval_seed"):
        return _stage_eval(plan, stage)
    if stage == "merge":
        return _stage_merge(plan)
    raise RecoveryError(f"unknown recovery stage {stage!r}")


def _dry_run_payload(plan: Mapping[str, Any]) -> dict[str, Any]:
    plan_path = Path(plan["lineage"]["control_dir"]) / "plan.json"
    ids: dict[str, str] = {}
    commands: dict[str, list[str]] = {}
    for index, spec in enumerate(_stage_specs(), 1):
        ids[str(spec["name"])] = str(910_000 + index)
        commands[str(spec["name"])] = _sbatch_command(
            spec=spec, plan_path=plan_path, plan_sha="<plan-sha256>",
            dependencies=[ids[name] for name in spec["depends_on"]],
        )
    return {
        "plan": plan, "stages": _stage_specs(), "commands": commands,
        "submitted": False, "training_jobs": 0, "consolidation_jobs": 0,
        "scientific_gate_jobs": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    submit = subparsers.add_parser("submit")
    submit.add_argument("--control-dir", required=True)
    submit.add_argument("--artifact-root", required=True)
    submit.add_argument("--group", default=GROUP)
    submit.add_argument("--project", default=PROJECT)
    submit.add_argument("--dry-run", action="store_true")
    subparsers.add_parser("run-stage")
    args = parser.parse_args(argv)
    try:
        if args.command == "run-stage":
            return run_stage()
        control, artifact = Path(args.control_dir), Path(args.artifact_root)
        if not control.is_absolute() or not artifact.is_absolute():
            raise RecoveryError("control-dir and artifact-root must be absolute")
        plan = build_plan(
            control_dir=control, artifact_root=artifact,
            group=args.group, project=args.project,
        )
        if args.dry_run:
            _assert_plan(plan)
            print(json.dumps(_dry_run_payload(plan), indent=2, sort_keys=True))
            return 0
        result = submit_plan(plan)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (
        RecoveryError, operator.OperatorRepairError, FileExistsError,
        FileNotFoundError, json.JSONDecodeError,
    ) as exc:
        print(f"OPERATOR_REPAIR_RECOVERY_INVALID: {exc}", file=sys.stderr, flush=True)
        return 2
    except subprocess.CalledProcessError as exc:
        command = " ".join(shlex.quote(str(item)) for item in exc.cmd)
        print(
            f"OPERATOR_REPAIR_RECOVERY_FAILED: command exited "
            f"{exc.returncode}: {command}",
            file=sys.stderr, flush=True,
        )
        return int(exc.returncode or 1)


if __name__ == "__main__":
    raise SystemExit(main())
