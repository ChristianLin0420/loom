#!/usr/bin/env python3
"""Diagnostic-only evaluation of the authenticated aborted R0 step-32k model.

This isolated chain never trains, never creates or changes a formal gate or
endpoint, and never writes below the source formal run/control/artifact paths.
The formal decision remains ABORT regardless of these diagnostic measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loom.train.direct_formal import receipt_exit_code  # noqa: E402
from scripts import r0_e2e_formal_chain as formal  # noqa: E402

FORMAT_VERSION = 1
KIND = "r0_e2e_post_abort_diagnostic_eval"
STEP = 32_000
WORLD_SIZE = 16
PROJECT = formal.PROJECT
DIAGNOSTIC_GROUP = (
    "r0a-dualcode-formal-s0-20260820-v2-postabort-diagnostic-s32000"
)
SOURCE_RUN_DIR = (ROOT / "runs/r0a_dualcode_formal_s0_20260820_v2").resolve()
SOURCE_CONTROL_DIR = (
    ROOT / "runs/r0a_dualcode_formal_s0_20260820_v2_control"
).resolve()
SOURCE_FORMAL_ARTIFACT_ROOT = (
    ROOT / "runs/r0a_dualcode_formal_s0_20260820_v2_artifacts"
).resolve()
SOURCE_FORMAL_PLAN = SOURCE_CONTROL_DIR / "plan.json"
SOURCE_FORMAL_PLAN_SHA256 = "798a536cb466ecc275cc7a21da9bf09e30a16f92c1d6e1ea79afe1c3ae75cdaf"
SOURCE_FORMAL_CLOSURE_SHA256 = "8bc6d4d7f0cd07a50b93273c795cf8623825666ddb3f264fd0d53277aad7f370"
SOURCE_JOBS_SHA256 = "bfebfe5b8d4fd02d009e2f26272ab05ea121067bae65d40f04bd7d91b3f5e062"
SOURCE_RELEASED_SHA256 = "1734c946e383e589b6cf95bb15a23361e7011df30656afcf38dc7c49ae48826f"
SOURCE_DIRECT_SHA256 = "e7ca6cb243adfdf58e12c18fef35323a87815e266ff6bed1b449e009682ecb87"
SOURCE_RUN_CONFIG_SHA256 = "d9345b7814659fcff07c63d1c11057f589dff429455f8bb6741381e965ee7f72"
SOURCE_METRICS_SHA256 = "bb9563775a60be5f7660f50a7b61fbe0a647850c9664243fe94933af634774c7"
SOURCE_LATEST_SHA256 = "492f431bae35265f2e5f4ed49bd8c58dda912431be561504846988d00d05d117"

SOURCE_SMALL_FILES = {
    "formal_plan": (SOURCE_FORMAL_PLAN, SOURCE_FORMAL_PLAN_SHA256),
    "formal_plan_run_copy": (SOURCE_RUN_DIR / "formal_plan.json", SOURCE_FORMAL_PLAN_SHA256),
    "formal_jobs": (SOURCE_CONTROL_DIR / "jobs.json", SOURCE_JOBS_SHA256),
    "formal_release": (SOURCE_CONTROL_DIR / "released.json", SOURCE_RELEASED_SHA256),
    "direct_abort": (
        SOURCE_RUN_DIR / f"direct_formal_{STEP:09d}.json", SOURCE_DIRECT_SHA256,
    ),
    "run_config": (SOURCE_RUN_DIR / "config.json", SOURCE_RUN_CONFIG_SHA256),
    "metrics": (SOURCE_RUN_DIR / "metrics.jsonl", SOURCE_METRICS_SHA256),
    "latest": (SOURCE_RUN_DIR / "LATEST", SOURCE_LATEST_SHA256),
}
PROTECTED_ABSENT_PATHS = (
    SOURCE_CONTROL_DIR / "gate_32000.json",
    SOURCE_CONTROL_DIR / "terminal_endpoint.json",
    SOURCE_CONTROL_DIR / "checkpoint_receipt.json",
    SOURCE_CONTROL_DIR / "checkpoint_verification.json",
    SOURCE_CONTROL_DIR / "eval_seed_0_receipt.json",
    SOURCE_CONTROL_DIR / "eval_seed_1_receipt.json",
    SOURCE_CONTROL_DIR / "eval_seed_2_receipt.json",
    SOURCE_CONTROL_DIR / "merged_eval_receipt.json",
    SOURCE_FORMAL_ARTIFACT_ROOT,
)
DIAGNOSTIC_SOURCE_FILES = (
    "scripts/r0_e2e_post_abort_eval.py",
    "scripts/r0_e2e_post_abort_consolidate.sbatch",
    "scripts/r0_e2e_post_abort_eval_seed.sbatch",
    "scripts/r0_e2e_post_abort_control.sbatch",
)
WANDB_ATTEMPTS = 5
WANDB_RETRY_SECONDS = 15
EXPECTED_ELIGIBILITY = {
    "formal_eligible": False,
    "promotion_eligible": False,
    "diagnostic_only": True,
    "formal_abort_preserved": True,
    "interpretation": "measurement only; cannot reverse formal ABORT",
}
EXPECTED_METHOD = {
    "training_updates": 0,
    "optimizer_steps": 0,
    "checkpoint_selection_used_eval": False,
    "checkpoint_selection": "explicit_user_selected_exact_step_32000",
}


class DiagnosticError(RuntimeError):
    """Fail-closed diagnostic provenance or execution error."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _pretty_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def sha256_file(path: str | Path) -> str:
    return formal.sha256_file(path)


def _identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _source_closure() -> dict[str, Any]:
    files = {name: sha256_file(ROOT / name) for name in DIAGNOSTIC_SOURCE_FILES}
    digest = hashlib.sha256()
    for name in sorted(files):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(files[name].encode())
        digest.update(b"\0")
    return {
        "scheme": "sha256(path-nul-sha256-nul)-v1",
        "sha256": digest.hexdigest(),
        "files": files,
    }


def _check_protected_absence() -> None:
    present = [str(path) for path in PROTECTED_ABSENT_PATHS if path.exists()]
    if present:
        raise DiagnosticError(
            "protected formal outputs appeared; refusing diagnostic continuation: "
            + repr(present)
        )


def _require_isolated(control_dir: Path, artifact_root: Path) -> None:
    if control_dir == artifact_root:
        raise DiagnosticError("diagnostic control and artifact paths must differ")
    for candidate in (control_dir, artifact_root):
        if not candidate.is_absolute():
            raise DiagnosticError("diagnostic paths must be absolute")
        for source in (SOURCE_RUN_DIR, SOURCE_CONTROL_DIR, SOURCE_FORMAL_ARTIFACT_ROOT):
            if candidate == source or candidate.is_relative_to(source) or source.is_relative_to(candidate):
                raise DiagnosticError(f"diagnostic path overlaps protected formal path {source}")


def _expected_paths(control_dir: Path, artifact_root: Path) -> dict[str, Any]:
    return {
        "checkpoint": str(artifact_root / "checkpoint/ckpt.pt"),
        "checkpoint_report": str(control_dir / "checkpoint_verification.json"),
        "checkpoint_receipt": str(control_dir / "checkpoint_receipt.json"),
        "merged_results": str(artifact_root / "eval/merged/results.json"),
        "merged_table": str(artifact_root / "eval/merged/table.md"),
        "merged_receipt": str(control_dir / "merged_eval_receipt.json"),
        "eval": {
            str(seed): {
                "out_dir": str(artifact_root / f"eval/seed_{seed}"),
                "receipt": str(control_dir / f"eval_seed_{seed}_receipt.json"),
            }
            for seed in formal.SEEDS
        },
    }


def _require_output_paths_contained(
    paths: Mapping[str, Any], *, control_dir: Path, artifact_root: Path,
) -> None:
    control_outputs = (
        paths["checkpoint_report"], paths["checkpoint_receipt"],
        paths["merged_receipt"],
        *(paths["eval"][str(seed)]["receipt"] for seed in formal.SEEDS),
    )
    artifact_outputs = (
        paths["checkpoint"], paths["merged_results"], paths["merged_table"],
        *(paths["eval"][str(seed)]["out_dir"] for seed in formal.SEEDS),
    )
    for root, outputs in (
        (control_dir, control_outputs), (artifact_root, artifact_outputs),
    ):
        for raw in outputs:
            path = Path(raw)
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise DiagnosticError(
                    f"diagnostic output escapes isolated root {root}: {path}"
                )


def _expected_baseline_comparison(
    source_plan: Mapping[str, Any], baseline: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        **source_plan["baseline_comparison"],
        "baseline": dict(baseline),
        "role": "counterfactual_post_abort_diagnostic_comparison",
        "cannot_change_formal_status": True,
    }


def _load_source_plan() -> dict[str, Any]:
    for label, (path, expected) in SOURCE_SMALL_FILES.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise DiagnosticError(f"source {label} identity changed")
    try:
        plan = formal.load_plan(SOURCE_FORMAL_PLAN, SOURCE_FORMAL_PLAN_SHA256)
    except formal.ChainError as exc:
        raise DiagnosticError(f"source formal plan failed authentication: {exc}") from exc
    checks = {
        "closure": plan["orchestration_source_closure"]["sha256"]
        == SOURCE_FORMAL_CLOSURE_SHA256,
        "current_closure": formal._source_closure()
        == plan["orchestration_source_closure"],
        "run_dir": Path(plan["lineage"]["run_dir"]).resolve() == SOURCE_RUN_DIR,
        "control_dir": Path(plan["lineage"]["control_dir"]).resolve()
        == SOURCE_CONTROL_DIR,
        "artifact_root": Path(plan["lineage"]["artifact_root"]).resolve()
        == SOURCE_FORMAL_ARTIFACT_ROOT,
        "step": plan["steps"]["initial_stop"] == STEP,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise DiagnosticError(f"source formal plan mismatch: {failed}")
    return plan


def _recompute_abort(source_plan: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        in_loop, in_loop_path = formal._read_direct_boundary_receipt(source_plan, STEP)
    except formal.ChainError as exc:
        raise DiagnosticError(str(exc)) from exc
    completed = subprocess.run(
        [
            sys.executable, str(ROOT / "scripts/direct_formal_convergence.py"),
            str(SOURCE_RUN_DIR), "--current-step", str(STEP),
        ],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=False,
    )
    if completed.stderr:
        raise DiagnosticError("direct convergence recomputation emitted stderr")
    try:
        recomputed = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise DiagnosticError("direct convergence recomputation was not JSON") from exc
    try:
        rc = receipt_exit_code(recomputed)
        core_equal = formal._direct_core(recomputed) == formal._direct_core(in_loop)
    except (ValueError, formal.ChainError) as exc:
        raise DiagnosticError(str(exc)) from exc
    metrics = recomputed.get("metrics_source")
    checks = {
        "rc": completed.returncode == rc == 3,
        "core": core_equal,
        "status": recomputed.get("status") == in_loop.get("status") == "ABORT",
        "reason": recomputed.get("reason") == "health_gate_failed",
        "step": recomputed.get("current_step") == recomputed.get("decision_step") == STEP,
        "next": recomputed.get("next_check_step") is None,
        "metrics": isinstance(metrics, Mapping)
        and metrics.get("path") == str((SOURCE_RUN_DIR / "metrics.jsonl").resolve())
        and metrics.get("sha256") == SOURCE_METRICS_SHA256
        and metrics.get("bytes") == (SOURCE_RUN_DIR / "metrics.jsonl").stat().st_size,
        "path": in_loop_path.resolve()
        == (SOURCE_RUN_DIR / f"direct_formal_{STEP:09d}.json").resolve(),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise DiagnosticError(f"source ABORT trigger failed: {failed}")
    return dict(in_loop), dict(recomputed)


def _source_shards(*, hash_bytes: bool) -> dict[str, dict[str, Any]]:
    try:
        paths = formal._checkpoint_shards(
            SOURCE_RUN_DIR, STEP, WORLD_SIZE, require_latest=True,
        )
    except formal.ChainError as exc:
        raise DiagnosticError(str(exc)) from exc
    if hash_bytes:
        return formal._checkpoint_shard_receipt(paths)
    return {path.name: {"bytes": path.stat().st_size} for path in paths}


def _collect_trigger(*, hash_shards: bool) -> dict[str, Any]:
    _check_protected_absence()
    source_plan = _load_source_plan()
    in_loop, recomputed = _recompute_abort(source_plan)
    core = formal._direct_core(recomputed)
    return {
        "kind": "authenticated_formal_abort_step32000_user_selected_diagnostic_v1",
        "formal_eligible": False,
        "formal_status_remains": "ABORT",
        "formal_reason": "health_gate_failed",
        "authorization_scope": "evaluate_existing_checkpoint_no_retraining",
        "source_formal_plan": _identity(SOURCE_FORMAL_PLAN),
        "source_formal_jobs": _identity(SOURCE_CONTROL_DIR / "jobs.json"),
        "source_formal_release": _identity(SOURCE_CONTROL_DIR / "released.json"),
        "source_orchestration_closure_sha256": SOURCE_FORMAL_CLOSURE_SHA256,
        "source_run_dir": str(SOURCE_RUN_DIR),
        "source_run_config": _identity(SOURCE_RUN_DIR / "config.json"),
        "source_metrics": _identity(SOURCE_RUN_DIR / "metrics.jsonl"),
        "source_latest": _identity(SOURCE_RUN_DIR / "LATEST"),
        "source_latest_step": formal._latest_step(SOURCE_RUN_DIR),
        "source_direct_receipt": _identity(
            SOURCE_RUN_DIR / f"direct_formal_{STEP:09d}.json"
        ),
        "direct_core_sha256": hashlib.sha256(_canonical_json(core).encode()).hexdigest(),
        "in_loop_recomputed_core_equal": formal._direct_core(in_loop) == core,
        "independent_recomputation": {
            "status": recomputed["status"], "reason": recomputed["reason"],
            "returncode": 3, "current_step": STEP, "decision_step": STEP,
            "next_check_step": None, "metrics_source": recomputed["metrics_source"],
        },
        "checkpoint_step": STEP,
        "checkpoint_shards": _source_shards(hash_bytes=hash_shards),
        "checkpoint_shard_hashes_included": hash_shards,
        "protected_formal_paths_absent": [str(path) for path in PROTECTED_ABSENT_PATHS],
    }


def _assert_trigger(plan: Mapping[str, Any], *, rehash_shards: bool) -> None:
    frozen = plan.get("trigger")
    if not isinstance(frozen, Mapping):
        raise DiagnosticError("plan omitted trigger")
    current = _collect_trigger(hash_shards=rehash_shards)
    if rehash_shards:
        if current != dict(frozen):
            raise DiagnosticError("source ABORT trigger or shard bytes changed")
        return
    frozen_small = dict(frozen)
    frozen_shards = frozen_small.pop("checkpoint_shards", None)
    frozen_small.pop("checkpoint_shard_hashes_included", None)
    current_small = dict(current)
    current_shards = current_small.pop("checkpoint_shards", None)
    current_small.pop("checkpoint_shard_hashes_included", None)
    if frozen_small != current_small:
        raise DiagnosticError("source ABORT trigger changed")
    expected_stats = {
        name: {"bytes": row.get("bytes")}
        for name, row in (frozen_shards or {}).items()
    }
    if expected_stats != current_shards:
        raise DiagnosticError("source shard names/sizes changed")


def build_plan(
    *, control_dir: Path, artifact_root: Path, group: str, project: str = PROJECT,
) -> dict[str, Any]:
    control_dir, artifact_root = control_dir.resolve(), artifact_root.resolve()
    _require_isolated(control_dir, artifact_root)
    if control_dir.exists() or artifact_root.exists():
        raise DiagnosticError("diagnostic output paths must be fresh")
    if group != DIAGNOSTIC_GROUP:
        raise DiagnosticError(f"W&B group must be exact diagnostic group {DIAGNOSTIC_GROUP}")
    if project != PROJECT:
        raise DiagnosticError(f"project must remain {PROJECT}")
    source_plan = _load_source_plan()
    trigger = _collect_trigger(hash_shards=True)
    baseline = formal._authenticate_baseline(formal.CANONICAL_BASELINE_ROOT)
    paths = _expected_paths(control_dir, artifact_root)
    return {
        "format_version": FORMAT_VERSION,
        "kind": KIND,
        "eligibility": dict(EXPECTED_ELIGIBILITY),
        "method": dict(EXPECTED_METHOD),
        "trigger": trigger,
        "diagnostic_source_closure": _source_closure(),
        "source_formal_config": source_plan["config"],
        "lineage": {
            "source_run_dir": str(SOURCE_RUN_DIR),
            "source_control_dir": str(SOURCE_CONTROL_DIR),
            "diagnostic_control_dir": str(control_dir),
            "diagnostic_artifact_root": str(artifact_root),
        },
        "evaluation": source_plan["evaluation"],
        "baseline_comparison": _expected_baseline_comparison(
            source_plan, baseline,
        ),
        "wandb": {
            "project": project, "group": group, "require_online": True,
            "tags": ["post-abort-diagnostic", "not-formal", "r0", "dual-action"],
            "stage_run_ids": {
                stage: uuid.uuid4().hex[:16]
                for stage in (
                    "diag-consolidate", "diag-eval-seed-0", "diag-eval-seed-1",
                    "diag-eval-seed-2", "diag-summary",
                )
            },
            "artifact_policy": "receipts_and_results_only_no_checkpoint_bytes",
        },
        "paths": paths,
    }


def _assert_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("format_version") != FORMAT_VERSION or plan.get("kind") != KIND:
        raise DiagnosticError("unsupported diagnostic plan")
    if plan.get("diagnostic_source_closure") != _source_closure():
        raise DiagnosticError("diagnostic source closure changed")
    if (
        plan.get("eligibility") != EXPECTED_ELIGIBILITY
        or plan.get("method") != EXPECTED_METHOD
    ):
        raise DiagnosticError("diagnostic eligibility contract changed")
    lineage = plan["lineage"]
    if (
        Path(lineage["source_run_dir"]).resolve() != SOURCE_RUN_DIR
        or Path(lineage["source_control_dir"]).resolve() != SOURCE_CONTROL_DIR
    ):
        raise DiagnosticError("source lineage changed")
    _require_isolated(
        Path(lineage["diagnostic_control_dir"]).resolve(),
        Path(lineage["diagnostic_artifact_root"]).resolve(),
    )
    control_dir = Path(lineage["diagnostic_control_dir"]).resolve()
    artifact_root = Path(lineage["diagnostic_artifact_root"]).resolve()
    if plan.get("paths") != _expected_paths(control_dir, artifact_root):
        raise DiagnosticError("diagnostic output paths differ from isolated lineage")
    _require_output_paths_contained(
        plan["paths"], control_dir=control_dir, artifact_root=artifact_root,
    )
    source_plan = _load_source_plan()
    if plan.get("source_formal_config") != source_plan["config"]:
        raise DiagnosticError("source config receipt changed")
    if plan.get("evaluation") != source_plan["evaluation"]:
        raise DiagnosticError("evaluation protocol changed")
    baseline = formal._authenticate_baseline(formal.CANONICAL_BASELINE_ROOT)
    if plan.get("baseline_comparison") != _expected_baseline_comparison(
        source_plan, baseline,
    ):
        raise DiagnosticError("baseline changed")
    wandb = plan.get("wandb", {})
    stage_ids = wandb.get("stage_run_ids", {})
    source_wandb = source_plan["wandb"]
    source_ids = {
        source_wandb["training_run_id"], *source_wandb["stage_run_ids"].values(),
    }
    expected_stages = {
        "diag-consolidate", "diag-eval-seed-0", "diag-eval-seed-1",
        "diag-eval-seed-2", "diag-summary",
    }
    if not (
        wandb.get("project") == PROJECT
        and wandb.get("group") == DIAGNOSTIC_GROUP
        and wandb.get("require_online") is True
        and wandb.get("artifact_policy")
        == "receipts_and_results_only_no_checkpoint_bytes"
        and wandb.get("tags")
        == ["post-abort-diagnostic", "not-formal", "r0", "dual-action"]
        and isinstance(stage_ids, Mapping)
        and set(stage_ids) == expected_stages
        and len(set(stage_ids.values())) == len(expected_stages)
        and all(
            isinstance(value, str) and re.fullmatch(r"[0-9a-f]{16}", value)
            for value in stage_ids.values()
        )
        and set(stage_ids.values()).isdisjoint(source_ids)
    ):
        raise DiagnosticError("diagnostic W&B labels changed")
    _assert_trigger(plan, rehash_shards=False)


def load_plan(path: str | Path, expected_sha256: str | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    if expected_sha256 and sha256_file(path) != expected_sha256:
        raise DiagnosticError("diagnostic plan SHA mismatch")
    try:
        plan = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise DiagnosticError("diagnostic plan unreadable") from exc
    if not isinstance(plan, Mapping):
        raise DiagnosticError("diagnostic plan is not an object")
    plan = dict(plan)
    _assert_plan(plan)
    return plan


def _required_plan() -> tuple[dict[str, Any], str]:
    path = os.environ.get("DIAGNOSTIC_PLAN")
    digest = os.environ.get("DIAGNOSTIC_PLAN_SHA256")
    stage = os.environ.get("DIAGNOSTIC_STAGE")
    if not path or not digest or not stage:
        raise DiagnosticError("DIAGNOSTIC_PLAN/SHA256/STAGE are required")
    return load_plan(path, digest), stage


def _plan_sha() -> str:
    path = os.environ.get("DIAGNOSTIC_PLAN")
    if not path:
        raise DiagnosticError("DIAGNOSTIC_PLAN is required")
    return sha256_file(path)


def _read_receipt(path: str | Path, *, kind: str) -> dict[str, Any]:
    try:
        receipt = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise DiagnosticError(f"invalid {kind} receipt") from exc
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("format_version") != FORMAT_VERSION
        or receipt.get("kind") != kind
    ):
        raise DiagnosticError(f"invalid {kind} receipt")
    return dict(receipt)


def _checkpoint_step(path: Path) -> int:
    import torch  # noqa: PLC0415

    payload = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
    try:
        return int(payload.get("global_step", -1))
    finally:
        del payload


def _pinned_shards(plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    pinned_dir = Path(plan["paths"]["checkpoint"]).parent / f"shards_{STEP:09d}"
    try:
        paths = formal._checkpoint_shards(
            pinned_dir, STEP, WORLD_SIZE, require_latest=False,
        )
    except formal.ChainError as exc:
        raise DiagnosticError(str(exc)) from exc
    return formal._checkpoint_shard_receipt(paths)


def _validate_checkpoint_receipt(
    plan: Mapping[str, Any], *, rehash_pinned: bool = False,
) -> dict[str, Any]:
    receipt = _read_receipt(
        plan["paths"]["checkpoint_receipt"],
        kind="r0_e2e_post_abort_diagnostic_checkpoint_receipt",
    )
    checkpoint = Path(receipt.get("checkpoint", ""))
    report = Path(receipt.get("verification_report", ""))
    try:
        verification = json.loads(report.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise DiagnosticError("checkpoint verification report unreadable") from exc
    checks = {
        "plan": receipt.get("diagnostic_plan_sha256") == _plan_sha(),
        "formal": receipt.get("formal_eligible") is False
        and receipt.get("source_formal_status") == "ABORT",
        "step": receipt.get("step") == STEP,
        "checkpoint_path": checkpoint == Path(plan["paths"]["checkpoint"]),
        "checkpoint_sha": checkpoint.is_file()
        and sha256_file(checkpoint) == receipt.get("checkpoint_sha256"),
        "checkpoint_bytes": checkpoint.is_file()
        and checkpoint.stat().st_size == receipt.get("checkpoint_bytes"),
        "checkpoint_step": checkpoint.is_file() and _checkpoint_step(checkpoint) == STEP,
        "report_path": report == Path(plan["paths"]["checkpoint_report"]),
        "report_sha": report.is_file()
        and sha256_file(report) == receipt.get("verification_report_sha256"),
        "report_pass": verification.get("pass") is True,
        "source_plan": receipt.get("source_formal_plan_sha256")
        == SOURCE_FORMAL_PLAN_SHA256,
        "source_direct": receipt.get("source_direct_receipt_sha256")
        == SOURCE_DIRECT_SHA256,
        "source_config": receipt.get("source_run_config_sha256")
        == SOURCE_RUN_CONFIG_SHA256,
        "source_shards": receipt.get("source_checkpoint_shards")
        == plan["trigger"]["checkpoint_shards"],
        "pinned_shards": receipt.get("pinned_checkpoint_shards")
        == plan["trigger"]["checkpoint_shards"],
    }
    if rehash_pinned:
        checks["pinned_live"] = (
            _pinned_shards(plan) == plan["trigger"]["checkpoint_shards"]
        )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise DiagnosticError(f"checkpoint receipt failed: {failed}")
    return receipt


def _wandb_publish_once(
    plan: Mapping[str, Any], *, stage: str, path: Path,
    artifact_type: str, summary: Mapping[str, Any],
) -> None:
    try:
        import wandb  # noqa: PLC0415
    except ImportError as exc:
        raise DiagnosticError("required online W&B is unavailable") from exc
    run_dir = Path(plan["lineage"]["diagnostic_control_dir"]) / "wandb" / stage
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(plan["wandb"]["stage_run_ids"][stage])
    id_path = run_dir / "wandb_id"
    if id_path.exists():
        if id_path.read_text().strip() != run_id:
            raise DiagnosticError(f"W&B run id changed for {stage}")
    else:
        formal._exclusive_text_write(id_path, run_id + "\n")
    run = wandb.init(
        project=plan["wandb"]["project"], id=run_id,
        name=f"{plan['wandb']['group']}-{stage}",
        group=plan["wandb"]["group"], job_type=stage,
        tags=list(plan["wandb"]["tags"]), resume="allow", mode="online",
        dir=str(run_dir),
        config={
            "diagnostic_plan_sha256": _plan_sha(),
            "diagnostic_stage": stage,
            "formal_eligible": False,
            "source_formal_status": "ABORT",
            "source_formal_plan_sha256": SOURCE_FORMAL_PLAN_SHA256,
            "source_direct_receipt_sha256": SOURCE_DIRECT_SHA256,
            "checkpoint_step": STEP,
        },
        settings=wandb.Settings(init_timeout=90),
    )
    if bool(getattr(run, "offline", False)):
        run.finish()
        raise DiagnosticError("W&B returned offline for required publication")
    try:
        fields = {
            **dict(summary), "formal_eligible": False,
            "source_formal_status": "ABORT", "diagnostic_only": True,
        }
        for key, value in fields.items():
            run.summary[key] = value
        artifact = wandb.Artifact(
            name=f"{plan['wandb']['group']}-{stage}", type=artifact_type,
            metadata={**fields, "sha256": sha256_file(path)},
        )
        artifact.add_file(str(path), name=path.name)
        run.log_artifact(artifact)
    finally:
        run.finish()


def _wandb_publish(
    plan: Mapping[str, Any], *, stage: str, path: Path,
    artifact_type: str, summary: Mapping[str, Any],
) -> None:
    errors: list[str] = []
    for attempt in range(1, WANDB_ATTEMPTS + 1):
        try:
            _wandb_publish_once(
                plan, stage=stage, path=path,
                artifact_type=artifact_type, summary=summary,
            )
            return
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")
            if attempt == WANDB_ATTEMPTS:
                raise DiagnosticError(
                    "required diagnostic W&B publication failed: " + " | ".join(errors)
                ) from exc
            print(
                f"[post-abort-wandb] {stage} attempt {attempt} failed; "
                f"retrying in {WANDB_RETRY_SECONDS}s: {errors[-1]}", flush=True,
            )
            time.sleep(WANDB_RETRY_SECONDS)


def _stage_consolidate(plan: Mapping[str, Any]) -> int:
    _assert_trigger(plan, rehash_shards=True)
    receipt_path = Path(plan["paths"]["checkpoint_receipt"])
    if receipt_path.exists():
        _validate_checkpoint_receipt(plan, rehash_pinned=True)
        _wandb_publish(
            plan, stage="diag-consolidate", path=receipt_path,
            artifact_type="diagnostic-checkpoint-receipt",
            summary={"checkpoint_step": STEP, "verification_passed": True},
        )
        return 0
    checkpoint = Path(plan["paths"]["checkpoint"])
    report = Path(plan["paths"]["checkpoint_report"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "loom.train.consolidate",
        "--run_dir", str(SOURCE_RUN_DIR), "--step", str(STEP),
        "--out", str(checkpoint), "--config", str(SOURCE_RUN_DIR / "config.json"),
        "--pin", "--report", str(report),
    ]
    if checkpoint.exists():
        command.append("--verify_only")
    subprocess.run(command, cwd=ROOT, check=True)
    try:
        verification = json.loads(report.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise DiagnosticError("consolidation report unreadable") from exc
    if verification.get("pass") is not True:
        raise DiagnosticError("consolidation verification failed")
    pinned = _pinned_shards(plan)
    if pinned != plan["trigger"]["checkpoint_shards"]:
        raise DiagnosticError("pinned shards differ from authenticated source")
    _assert_trigger(plan, rehash_shards=True)
    if _checkpoint_step(checkpoint) != STEP:
        raise DiagnosticError("consolidated checkpoint step differs from 32000")
    receipt = {
        "format_version": FORMAT_VERSION,
        "kind": "r0_e2e_post_abort_diagnostic_checkpoint_receipt",
        "diagnostic_plan_sha256": _plan_sha(),
        "formal_eligible": False,
        "source_formal_status": "ABORT",
        "source_formal_reason": "health_gate_failed",
        "step": STEP,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "verification_report": str(report),
        "verification_report_sha256": sha256_file(report),
        "source_formal_plan_sha256": SOURCE_FORMAL_PLAN_SHA256,
        "source_direct_receipt_sha256": SOURCE_DIRECT_SHA256,
        "source_run_config_sha256": SOURCE_RUN_CONFIG_SHA256,
        "source_checkpoint_shards": plan["trigger"]["checkpoint_shards"],
        "pinned_checkpoint_shards": pinned,
        "checkpoint_bytes_uploaded_to_wandb": False,
        "training_updates_performed": 0,
        "optimizer_steps_performed": 0,
    }
    formal.exclusive_json_write(receipt_path, receipt)
    _validate_checkpoint_receipt(plan, rehash_pinned=True)
    _wandb_publish(
        plan, stage="diag-consolidate", path=receipt_path,
        artifact_type="diagnostic-checkpoint-receipt",
        summary={"checkpoint_step": STEP, "verification_passed": True},
    )
    return 0


def _validate_seed_result(
    plan: Mapping[str, Any], *, seed: int, result_path: Path,
) -> tuple[dict[str, Any], Any]:
    checkpoint_receipt = _validate_checkpoint_receipt(plan)
    checkpoint = Path(checkpoint_receipt["checkpoint"])
    try:
        blob = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise DiagnosticError(f"seed {seed} result unreadable") from exc
    if not isinstance(blob, Mapping):
        raise DiagnosticError(f"seed {seed} result is not an object")
    blob = dict(blob)
    try:
        protocol = formal._result_protocol(blob)
        rows = formal._validate_exact_eval_blob(
            blob, seed=seed, label=f"diagnostic candidate seed {seed}",
            identity_profile="current_candidate",
        )
        formal._validate_eval_method_identity(
            blob, label=f"diagnostic candidate seed {seed}",
            identity_profile="current_candidate", checkpoint_step=STEP,
            checkpoint_path=str(checkpoint.resolve()),
        )
    except formal.ChainError as exc:
        raise DiagnosticError(str(exc)) from exc
    evaluation = plan["evaluation"]
    actual = {
        "bench": protocol.bench, "seeds": protocol.seeds,
        "suites": protocol.suites, "n_tasks": protocol.n_tasks,
        "episodes_per_task": protocol.episodes_per_task,
        "max_steps": protocol.max_steps,
    }
    expected = {
        "bench": "libero", "seeds": (seed,),
        "suites": tuple(evaluation["suites"]),
        "n_tasks": int(evaluation["tasks_per_suite"]),
        "episodes_per_task": int(evaluation["episodes_per_task"]),
        "max_steps": int(evaluation["max_steps"]),
    }
    if actual != expected or len(rows) != formal.EXPECTED_EPISODES_PER_SEED:
        raise DiagnosticError(f"seed {seed} protocol/result count changed")
    return blob, protocol


def _validate_eval_receipt(plan: Mapping[str, Any], seed: int) -> dict[str, Any]:
    receipt = _read_receipt(
        plan["paths"]["eval"][str(seed)]["receipt"],
        kind="r0_e2e_post_abort_diagnostic_single_seed_eval_receipt",
    )
    expected_dir = Path(plan["paths"]["eval"][str(seed)]["out_dir"])
    result = expected_dir / "results.json"
    table = expected_dir / "table.md"
    blob, _ = _validate_seed_result(plan, seed=seed, result_path=result)
    _ensure_seed_table(table, blob, allow_create=False)
    checkpoint_receipt = _validate_checkpoint_receipt(plan)
    expected = _eval_receipt_payload(
        plan, seed=seed, blob=blob, result_path=result, table_path=table,
        checkpoint_receipt=checkpoint_receipt,
    )
    if receipt != expected:
        raise DiagnosticError(f"seed {seed} receipt differs from exact recomputation")
    return receipt


def _seed_markdown(blob: Mapping[str, Any]) -> str:
    from loom.eval.table import render_report  # noqa: PLC0415

    return render_report(blob, row_label="**LOOM · R0-A**")


def _ensure_seed_table(
    table_path: Path, blob: Mapping[str, Any], *, allow_create: bool,
) -> None:
    expected = _seed_markdown(blob)
    if table_path.exists():
        if not table_path.is_file() or table_path.is_symlink():
            raise DiagnosticError("seed evaluation table is not a regular file")
        if table_path.read_text() != expected:
            raise DiagnosticError("seed evaluation table differs from result")
        return
    if not allow_create:
        raise DiagnosticError("seed evaluation receipt references a missing table")
    formal._exclusive_text_write(table_path, expected)


def _eval_receipt_payload(
    plan: Mapping[str, Any], *, seed: int, blob: Mapping[str, Any],
    result_path: Path, table_path: Path,
    checkpoint_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "kind": "r0_e2e_post_abort_diagnostic_single_seed_eval_receipt",
        "diagnostic_plan_sha256": _plan_sha(),
        "formal_eligible": False,
        "source_formal_status": "ABORT",
        "seed": seed,
        "result": str(result_path), "result_sha256": sha256_file(result_path),
        "table": str(table_path), "table_sha256": sha256_file(table_path),
        "episodes": formal.EXPECTED_EPISODES_PER_SEED,
        "errors": 0, "avg": blob["summary"]["avg"],
        "checkpoint_step": STEP,
        "checkpoint_sha256": checkpoint_receipt["checkpoint_sha256"],
        "checkpoint_receipt_sha256": sha256_file(
            plan["paths"]["checkpoint_receipt"]
        ),
        "training_updates_performed": 0, "optimizer_steps_performed": 0,
    }


def _stage_eval(plan: Mapping[str, Any], stage: str) -> int:
    match = re.fullmatch(r"eval_seed([0-9]+)", stage)
    if match is None or int(match.group(1)) not in formal.SEEDS:
        raise DiagnosticError(f"invalid seed stage {stage}")
    seed = int(match.group(1))
    _assert_trigger(plan, rehash_shards=False)
    checkpoint_receipt = _validate_checkpoint_receipt(plan)
    checkpoint = Path(checkpoint_receipt["checkpoint"])
    out_dir = Path(plan["paths"]["eval"][str(seed)]["out_dir"])
    receipt_path = Path(plan["paths"]["eval"][str(seed)]["receipt"])
    result_path, table_path = out_dir / "results.json", out_dir / "table.md"
    if receipt_path.exists():
        _validate_eval_receipt(plan, seed)
        blob, _ = _validate_seed_result(plan, seed=seed, result_path=result_path)
        _wandb_publish(
            plan, stage=f"diag-eval-seed-{seed}", path=result_path,
            artifact_type="diagnostic-evaluation-results",
            summary={
                "seed": seed, "episodes": formal.EXPECTED_EPISODES_PER_SEED,
                "success_rate": blob["summary"]["avg"], "checkpoint_step": STEP,
            },
        )
        return 0
    if out_dir.exists() and not out_dir.is_dir():
        raise DiagnosticError(f"seed {seed} output is not a directory")
    if result_path.is_symlink() or table_path.is_symlink():
        raise DiagnosticError(f"seed {seed} output must not be a symlink")
    blob: dict[str, Any] | None = None
    if result_path.is_file():
        try:
            blob, _ = _validate_seed_result(plan, seed=seed, result_path=result_path)
        except DiagnosticError:
            blob = None
    if blob is None:
        command, env = formal._formal_eval_command(
            plan, seed=seed, checkpoint=checkpoint, out_dir=out_dir,
        )
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        blob, _ = _validate_seed_result(plan, seed=seed, result_path=result_path)
    _ensure_seed_table(table_path, blob, allow_create=True)
    receipt = _eval_receipt_payload(
        plan, seed=seed, blob=blob, result_path=result_path,
        table_path=table_path, checkpoint_receipt=checkpoint_receipt,
    )
    formal.exclusive_json_write(receipt_path, receipt)
    _validate_eval_receipt(plan, seed)
    _wandb_publish(
        plan, stage=f"diag-eval-seed-{seed}", path=result_path,
        artifact_type="diagnostic-evaluation-results",
        summary={
            "seed": seed, "episodes": formal.EXPECTED_EPISODES_PER_SEED,
            "success_rate": blob["summary"]["avg"], "checkpoint_step": STEP,
        },
    )
    return 0


def _stable_merge_provenance(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": "r0_e2e_post_abort_diagnostic_plan_stable_merge_v1",
        "diagnostic_plan_sha256": _plan_sha(),
        "diagnostic_source_closure_sha256": plan["diagnostic_source_closure"][
            "sha256"
        ],
        "source_formal_plan_sha256": SOURCE_FORMAL_PLAN_SHA256,
        "source_direct_receipt_sha256": SOURCE_DIRECT_SHA256,
        "formal_eligible": False,
    }


def merge_seed_results(plan: Mapping[str, Any]) -> dict[str, Any]:
    from loom.eval import EpisodeResult  # noqa: PLC0415
    from loom.eval.runner import aggregate, iter_work  # noqa: PLC0415

    blobs: list[dict[str, Any]] = []
    protocols = []
    source_receipts = []
    for seed in formal.SEEDS:
        source_receipts.append(_validate_eval_receipt(plan, seed))
        result_path = (
            Path(plan["paths"]["eval"][str(seed)]["out_dir"]) / "results.json"
        )
        blob, protocol = _validate_seed_result(
            plan, seed=seed, result_path=result_path,
        )
        blobs.append(blob)
        protocols.append(protocol)
    reference = formal._protocol_without_seeds(protocols[0])
    if any(
        formal._protocol_without_seeds(protocol) != reference
        for protocol in protocols[1:]
    ):
        raise DiagnosticError("singleton-seed protocols differ")
    target_protocol = protocols[0].replace(seeds=formal.SEEDS)
    records = [
        EpisodeResult.from_dict(row)
        for blob in blobs for row in blob.get("episodes", [])
    ]
    keys = [record.key() for record in records]
    expected = {item.key() for item in iter_work(target_protocol)}
    if (
        len(keys) != len(set(keys))
        or set(keys) != expected
        or len(keys) != formal.EXPECTED_EPISODES_TOTAL
    ):
        raise DiagnosticError("seed union is not the exact 1200 protocol")
    common_ckpt = blobs[0].get("meta", {}).get("ckpt")
    common_identity = blobs[0].get("meta", {}).get("eval_identity")
    for blob in blobs[1:]:
        if (
            blob.get("meta", {}).get("ckpt") != common_ckpt
            or blob.get("meta", {}).get("eval_identity") != common_identity
        ):
            raise DiagnosticError("singleton checkpoint/eval identities differ")
    candidate_rows: dict[tuple[str, str, int, int, int], dict[str, Any]] = {}
    for seed, blob in zip(formal.SEEDS, blobs):
        try:
            rows = formal._validate_exact_eval_blob(
                blob, seed=seed, label=f"diagnostic candidate seed {seed}",
                identity_profile="current_candidate",
            )
        except formal.ChainError as exc:
            raise DiagnosticError(str(exc)) from exc
        if candidate_rows.keys() & rows.keys():
            raise DiagnosticError("singleton seeds overlap")
        candidate_rows.update(rows)
    try:
        comparison = formal.paired_baseline_comparison(
            plan, candidate_rows, formal._baseline_rows(plan),
        )
    except formal.ChainError as exc:
        raise DiagnosticError(str(exc)) from exc
    comparison = {
        **comparison,
        "role": "counterfactual_post_abort_diagnostic_gate",
        "formal_eligible": False,
        "cannot_reverse_formal_abort": True,
    }
    summary = aggregate(records, target_protocol)
    if (
        summary.get("complete") is not True
        or summary.get("n_episodes") != formal.EXPECTED_EPISODES_TOTAL
        or summary.get("n_expected") != formal.EXPECTED_EPISODES_TOTAL
        or summary.get("n_errors") != 0
    ):
        raise DiagnosticError("merged result failed exact closure")
    return {
        "version": blobs[0].get("version", 1),
        "kind": "r0_e2e_post_abort_diagnostic_merged_results",
        "formal_eligible": False,
        "source_formal_status": "ABORT",
        "source_formal_reason": "health_gate_failed",
        "checkpoint_step": STEP,
        "bench": target_protocol.bench,
        "protocol": target_protocol.to_dict(),
        "meta": {
            "ckpt": common_ckpt,
            "eval_identity": common_identity,
            "policy": blobs[0].get("meta", {}).get("policy"),
            "source_singleton_seed_receipts": source_receipts,
            "source_checkpoint_receipt_sha256": sha256_file(
                plan["paths"]["checkpoint_receipt"]
            ),
            "merge_provenance": _stable_merge_provenance(plan),
            "merge_rule": "exact_disjoint_seed_union_then_loom_eval_aggregate",
            "baseline_receipt": plan["baseline_comparison"]["baseline"],
            "interpretation": (
                "diagnostic measurement only; cannot reverse formal ABORT"
            ),
        },
        "summary": summary,
        "diagnostic_baseline_comparison": comparison,
        "episodes": [
            record.to_dict() for record in sorted(records, key=lambda item: item.key())
        ],
    }


def _diagnostic_markdown(merged: Mapping[str, Any]) -> str:
    comparison = merged["diagnostic_baseline_comparison"]
    wrapper = {**merged, "baseline_comparison": comparison}
    table = formal._markdown_table(wrapper).replace(
        "Scientific evaluation gate:",
        "Counterfactual diagnostic threshold status:",
    )
    return (
        "# Post-ABORT diagnostic evaluation (not formal)\n\n"
        "The source formal decision remains **ABORT**. These measurements do not "
        "establish formal eligibility or authorize promotion.\n\n"
        + table
    )


def _summary_fields(comparison: Mapping[str, Any]) -> dict[str, Any]:
    fields = formal._eval_summary_wandb_fields(comparison)
    fields["counterfactual_diagnostic_gate_status"] = fields.pop(
        "scientific_gate_status"
    )
    fields["counterfactual_diagnostic_gate_passed"] = fields.pop(
        "scientific_gate_passed"
    )
    return fields


def _merged_receipt_payload(
    plan: Mapping[str, Any], merged: Mapping[str, Any], *,
    result_path: Path, table_path: Path,
) -> dict[str, Any]:
    comparison = merged["diagnostic_baseline_comparison"]
    return {
        "format_version": FORMAT_VERSION,
        "kind": "r0_e2e_post_abort_diagnostic_merged_eval_receipt",
        "diagnostic_plan_sha256": _plan_sha(),
        "formal_eligible": False, "promotion_eligible": False,
        "source_formal_status": "ABORT",
        "source_formal_reason": "health_gate_failed",
        "result": str(result_path), "result_sha256": sha256_file(result_path),
        "table": str(table_path), "table_sha256": sha256_file(table_path),
        "episodes": merged["summary"]["n_episodes"],
        "errors": merged["summary"]["n_errors"],
        "avg": merged["summary"]["avg"],
        "complete": merged["summary"]["complete"],
        "checkpoint_step": STEP,
        "checkpoint_receipt_sha256": sha256_file(
            plan["paths"]["checkpoint_receipt"]
        ),
        "source_eval_receipt_sha256": {
            str(seed): sha256_file(plan["paths"]["eval"][str(seed)]["receipt"])
            for seed in formal.SEEDS
        },
        "diagnostic_comparison_sha256": hashlib.sha256(
            _canonical_json(comparison).encode()
        ).hexdigest(),
        "baseline_file_sha256": {
            seed: row["sha256"]
            for seed, row in plan["baseline_comparison"]["baseline"]["files"].items()
        },
        "bootstrap_matrix_sha256": formal.BOOTSTRAP_MATRIX_SHA256,
        "baseline_delta_pp": comparison["overall"]["delta_percentage_points"],
        "paired_ci_low_pp": comparison["paired_task_bootstrap"][
            "ci_low_percentage_points"
        ],
        "paired_ci_high_pp": comparison["paired_task_bootstrap"][
            "ci_high_percentage_points"
        ],
        "seed0_successes": comparison["per_seed_candidate_successes"]["0"],
        "per_suite": comparison["per_suite"],
        "counterfactual_diagnostic_gate_status": comparison["status"],
        "counterfactual_diagnostic_gate_passed": comparison["passed"],
        "failed_diagnostic_checks": comparison["failed_checks"],
        "formal_status_unchanged_by_result": True,
        "training_updates_performed": 0, "optimizer_steps_performed": 0,
    }


def _validate_merged_receipt(plan: Mapping[str, Any]) -> dict[str, Any]:
    receipt = _read_receipt(
        plan["paths"]["merged_receipt"],
        kind="r0_e2e_post_abort_diagnostic_merged_eval_receipt",
    )
    result_path = Path(plan["paths"]["merged_results"])
    table_path = Path(plan["paths"]["merged_table"])
    merged = merge_seed_results(plan)
    if (
        not result_path.is_file()
        or result_path.read_text() != _pretty_json(merged)
        or not table_path.is_file()
        or table_path.read_text() != _diagnostic_markdown(merged)
    ):
        raise DiagnosticError("merged artifacts differ from exact recomputation")
    expected = _merged_receipt_payload(
        plan, merged, result_path=result_path, table_path=table_path,
    )
    if receipt != expected:
        raise DiagnosticError("merged receipt differs from exact recomputation")
    return receipt


def _stage_merge(plan: Mapping[str, Any]) -> int:
    _assert_trigger(plan, rehash_shards=False)
    result_path = Path(plan["paths"]["merged_results"])
    table_path = Path(plan["paths"]["merged_table"])
    receipt_path = Path(plan["paths"]["merged_receipt"])
    if receipt_path.exists():
        receipt = _validate_merged_receipt(plan)
        merged = json.loads(result_path.read_text())
        comparison = merged["diagnostic_baseline_comparison"]
        _wandb_publish(
            plan, stage="diag-summary", path=result_path,
            artifact_type="diagnostic-evaluation-results",
            summary={
                "episodes": formal.EXPECTED_EPISODES_TOTAL,
                "success_rate": receipt["avg"], "n_errors": 0,
                **_summary_fields(comparison),
            },
        )
        return 0
    merged = merge_seed_results(plan)
    expected_result = _pretty_json(merged)
    expected_table = _diagnostic_markdown(merged)
    if result_path.exists():
        if not result_path.is_file() or result_path.read_text() != expected_result:
            raise DiagnosticError("partial merged result differs")
    else:
        formal.exclusive_json_write(result_path, merged)
    if table_path.exists():
        if not table_path.is_file() or table_path.read_text() != expected_table:
            raise DiagnosticError("partial merged table differs")
    else:
        formal._exclusive_text_write(table_path, expected_table)
    comparison = merged["diagnostic_baseline_comparison"]
    receipt = _merged_receipt_payload(
        plan, merged, result_path=result_path, table_path=table_path,
    )
    formal.exclusive_json_write(receipt_path, receipt)
    _validate_merged_receipt(plan)
    _wandb_publish(
        plan, stage="diag-summary", path=result_path,
        artifact_type="diagnostic-evaluation-results",
        summary={
            "episodes": formal.EXPECTED_EPISODES_TOTAL,
            "success_rate": merged["summary"]["avg"], "n_errors": 0,
            **_summary_fields(comparison),
        },
    )
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    return 0


def _stage_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "consolidate",
            "sbatch": "scripts/r0_e2e_post_abort_consolidate.sbatch",
            "depends_on": [],
        },
        *[
            {
                "name": f"eval_seed{seed}",
                "sbatch": "scripts/r0_e2e_post_abort_eval_seed.sbatch",
                "depends_on": ["consolidate"],
            }
            for seed in formal.SEEDS
        ],
        {
            "name": "merge",
            "sbatch": "scripts/r0_e2e_post_abort_control.sbatch",
            "depends_on": [f"eval_seed{seed}" for seed in formal.SEEDS],
        },
    ]


def _sbatch_command(
    *, spec: Mapping[str, Any], plan_path: Path, plan_sha: str,
    dependencies: Sequence[str], group: str,
) -> list[str]:
    label = re.sub(r"[^A-Za-z0-9_-]", "_", f"r0diag_{group}_{spec['name']}")[:120]
    command = [
        "sbatch", "--parsable", "--hold", "--kill-on-invalid-dep=yes",
        f"--job-name={label}",
    ]
    if dependencies:
        command.append("--dependency=afterok:" + ":".join(dependencies))
    command.extend((
        "--export=" + ",".join((
            "ALL", f"DIAGNOSTIC_PLAN={plan_path}",
            f"DIAGNOSTIC_PLAN_SHA256={plan_sha}",
            f"DIAGNOSTIC_STAGE={spec['name']}",
        )),
        str(ROOT / spec["sbatch"]),
    ))
    return command


def _parse_job_id(stdout: str) -> str:
    value = stdout.strip().split(";", 1)[0]
    if re.fullmatch(r"[0-9]+(?:_[0-9]+)?", value) is None:
        raise DiagnosticError(f"invalid sbatch job id: {stdout!r}")
    return value


def submit_plan(
    plan: Mapping[str, Any], *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    _assert_plan(plan)
    control_dir = Path(plan["lineage"]["diagnostic_control_dir"])
    artifact_root = Path(plan["lineage"]["diagnostic_artifact_root"])
    if control_dir.exists() or artifact_root.exists():
        raise DiagnosticError("diagnostic output paths are not fresh")
    control_dir.mkdir(parents=True, exist_ok=False)
    plan_path = control_dir / "plan.json"
    formal.exclusive_json_write(plan_path, plan)
    plan_sha = sha256_file(plan_path)
    job_ids: dict[str, str] = {}
    commands: dict[str, list[str]] = {}
    submitted: list[str] = []
    try:
        for spec in _stage_specs():
            command = _sbatch_command(
                spec=spec, plan_path=plan_path, plan_sha=plan_sha,
                dependencies=[job_ids[name] for name in spec["depends_on"]],
                group=plan["wandb"]["group"],
            )
            completed = run(
                command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, check=True,
            )
            job_id = _parse_job_id(completed.stdout)
            job_ids[spec["name"]] = job_id
            commands[spec["name"]] = command
            submitted.append(job_id)
        receipt = {
            "format_version": FORMAT_VERSION,
            "kind": "r0_e2e_post_abort_diagnostic_jobs",
            "diagnostic_plan": str(plan_path),
            "diagnostic_plan_sha256": plan_sha,
            "formal_eligible": False, "source_formal_status": "ABORT",
            "jobs": job_ids, "commands": commands, "released": False,
        }
        formal.exclusive_json_write(control_dir / "jobs.json", receipt)
        run(
            ["scontrol", "release", ",".join(submitted)], cwd=ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
        )
        formal.exclusive_json_write(control_dir / "released.json", {
            "format_version": FORMAT_VERSION,
            "kind": "r0_e2e_post_abort_diagnostic_release",
            "diagnostic_plan_sha256": plan_sha,
            "jobs_sha256": sha256_file(control_dir / "jobs.json"),
            "job_ids": submitted, "released": True,
        })
        return {**receipt, "released": True}
    except Exception:
        if submitted:
            subprocess.run(
                ["scancel", *submitted], cwd=ROOT, check=False,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        raise


def run_environment_stage() -> int:
    plan, stage = _required_plan()
    if stage == "consolidate":
        return _stage_consolidate(plan)
    if stage.startswith("eval_seed"):
        return _stage_eval(plan, stage)
    if stage == "merge":
        return _stage_merge(plan)
    raise DiagnosticError(f"unknown diagnostic stage {stage}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    submit = subparsers.add_parser("submit")
    submit.add_argument("--control-dir", required=True)
    submit.add_argument("--artifact-root", required=True)
    submit.add_argument("--group", required=True)
    submit.add_argument("--project", default=PROJECT)
    submit.add_argument("--dry-run", action="store_true")
    subparsers.add_parser("run-stage")
    args = parser.parse_args(argv)
    try:
        if args.command == "run-stage":
            return run_environment_stage()
        control_dir = Path(args.control_dir)
        artifact_root = Path(args.artifact_root)
        if not control_dir.is_absolute() or not artifact_root.is_absolute():
            raise DiagnosticError("control-dir and artifact-root must be absolute")
        plan = build_plan(
            control_dir=control_dir, artifact_root=artifact_root,
            group=args.group, project=args.project,
        )
        if args.dry_run:
            print(json.dumps({
                "plan": plan, "stages": _stage_specs(), "submitted": False,
            }, indent=2, sort_keys=True))
            return 0
        result = submit_plan(plan)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (
        DiagnosticError, formal.ChainError, FileExistsError, FileNotFoundError,
        json.JSONDecodeError,
    ) as exc:
        print(f"POST_ABORT_DIAGNOSTIC_INVALID: {exc}", file=sys.stderr, flush=True)
        return 2
    except subprocess.CalledProcessError as exc:
        command = " ".join(shlex.quote(str(item)) for item in exc.cmd)
        print(
            f"POST_ABORT_DIAGNOSTIC_FAILED: command exited {exc.returncode}: {command}",
            file=sys.stderr, flush=True,
        )
        return int(exc.returncode or 1)


if __name__ == "__main__":
    raise SystemExit(main())
