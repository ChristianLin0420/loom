#!/usr/bin/env python3
"""Adopt completed post-ABORT results after the format-v1 identity bug.

This format-v2 path performs no environment episode, training update, optimizer
step, checkpoint selection, or formal promotion.  It authenticates the failed
format-v1 attempt and its exact completed results, mints fresh adoption
receipts, and reports the already-measured 1,200-episode diagnostic.
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

from scripts import r0_e2e_formal_chain as formal  # noqa: E402
from scripts import r0_e2e_post_abort_eval as v1  # noqa: E402

FORMAT_VERSION = 2
KIND = "r0_e2e_post_abort_diagnostic_completed_result_adoption_v2"
STEP = v1.STEP
PROJECT = v1.PROJECT
GROUP = "r0a-dualcode-formal-s0-20260820-v2-postabort-diagnostic-s32000-adoption-v2"

PRIOR_CONTROL = (
    ROOT / "runs/r0a_dualcode_formal_s0_20260820_v2_postabort_diag_s32000_control"
).resolve()
PRIOR_ARTIFACT = (
    ROOT / "runs/r0a_dualcode_formal_s0_20260820_v2_postabort_diag_s32000_artifacts"
).resolve()
PRIOR_PLAN_SHA256 = "3a54a24f5b388359d6ff6b807426d785536bbbeb7e17e1df101820359686aedc"
PRIOR_JOBS_SHA256 = "ff07c140f9e0a75d284814066649950f196ce4d2125b009df8087e4d53c27561"
PRIOR_RELEASED_SHA256 = "e2e34f5888d9e50cdc0f01fd1553b96f8899efc6b4bc466905c517c4a37468b0"
PRIOR_CHECKPOINT_RECEIPT_SHA256 = (
    "af296d54eefab29dfd67f22586b18dc3190529d808e1c0892aa2e53f9de0850b"
)
PRIOR_CHECKPOINT_REPORT_SHA256 = (
    "e9fce017291643ee441bb1b3cc2f335fb1acd550fa199b41e35433949e04f126"
)
PRIOR_CHECKPOINT_SHA256 = "eddcc36d94dc48b9031acbcdaea116b2a1693c8b9e357f96e2573da36c9039b6"
PRIOR_CHECKPOINT_BYTES = 1_760_598_524
PRIOR_RESULT_SHA256 = {
    0: "b43a9c8997bba6950a5c93234035f480f2f735732de3ca918132584d091702d7",
    1: "f07d33918faa493d4ef598ee5a9b377b089502e0a7035f7bccd63f14839c332c",
    2: "c4ddc82258bdc36992f0dc2623dab312267adef8602c0e42c82e4244dc5c0128",
}
PRIOR_TABLE_SHA256 = {
    0: "2781688b47e0886a18701271c94945540698ee3bad2547ee4c93a338c9feee82",
    1: "d7aba0c39b9241f8b4d0dc92de9d770bc078d8a5f15e17a2acb584ad1f0f94d2",
    2: "50bd85d6aed7d68f553db5c9863f1b5587bd88a6a867766b03a9e765c643f5ac",
}
PRIOR_SUCCESS_COUNTS = {0: 178, 1: 180, 2: 192}
PRIOR_JOB_STATES = {
    "32639504": "COMPLETED_0:0",
    "32639505": "FAILED_2:0_IDENTITY_POLICY_KW_AFTER_400",
    "32639506": "FAILED_2:0_IDENTITY_POLICY_KW_AFTER_400",
    "32639507": "FAILED_2:0_IDENTITY_POLICY_KW_AFTER_400",
    "32639508": "CANCELLED_AFTEROK",
}
PRIOR_LOGS = {
    "consolidate": (
        ROOT / "logs/r0diag_r0a-dualcode-formal-s0-20260820-v2-postabort-"
        "diagnostic-s32000_consolidate_32639504.out",
        "ffa8c0551e7ae798f95c8cf7223652cf144524841932a4e84ef1bd652d84114d",
    ),
    "seed0": (
        ROOT / "logs/r0diag_r0a-dualcode-formal-s0-20260820-v2-postabort-"
        "diagnostic-s32000_eval_seed0_32639505.out",
        "ca2fbfaafeb2b38471882623f6d6645568564d7968ff4e7be91a4ca0d5a6d584",
    ),
    "seed1": (
        ROOT / "logs/r0diag_r0a-dualcode-formal-s0-20260820-v2-postabort-"
        "diagnostic-s32000_eval_seed1_32639506.out",
        "0aa27c79f4d924e599e7e3848c49dad5199760519090845a5cfd9e910ef81b0f",
    ),
    "seed2": (
        ROOT / "logs/r0diag_r0a-dualcode-formal-s0-20260820-v2-postabort-"
        "diagnostic-s32000_eval_seed2_32639507.out",
        "da8c0ff3d3c9e10e5c05fd60d09748ff4884c6c422785b64f9245b2bce457766",
    ),
}
RECORDED_POLICY_KW = {"allow_stub": False, "op_stats": True}
NORMALIZATION = {
    "kind": "historical_cli_implicit_libero_franka_normalization_v1",
    "recorded_eval_identity_policy_kw": RECORDED_POLICY_KW,
    "required_policy_provenance": {
        "policy": "LoomPolicy", "is_stub": False,
        "embodiment": "libero_franka", "ckpt_global_step": STEP,
    },
    "interpretation": (
        "the historical CLI serialized only explicit policy kwargs; the real "
        "policy provenance independently authenticates the LIBERO embodiment"
    ),
}
EXPECTED_ELIGIBILITY = {
    "formal_eligible": False,
    "promotion_eligible": False,
    "diagnostic_only": True,
    "formal_abort_preserved": True,
    "instrumentation_repair_adoption_only": True,
}
EXPECTED_METHOD = {
    "training_updates": 0,
    "optimizer_steps": 0,
    "environment_episodes_rerun": 0,
    "checkpoint_selection_used_eval": False,
    "completed_result_adoption": True,
    "identity_normalization": NORMALIZATION,
}
SOURCE_FILES = (
    "scripts/r0_e2e_post_abort_adopt.py",
    "scripts/r0_e2e_post_abort_adopt_checkpoint.sbatch",
    "scripts/r0_e2e_post_abort_adopt_seed.sbatch",
    "scripts/r0_e2e_post_abort_adopt_control.sbatch",
)

CHECKPOINT_KIND = "r0_e2e_post_abort_diagnostic_adopted_checkpoint_receipt_v2"
SEED_KIND = "r0_e2e_post_abort_diagnostic_adopted_seed_receipt_v2"
MERGED_KIND = "r0_e2e_post_abort_diagnostic_adopted_merged_receipt_v2"


class AdoptionError(RuntimeError):
    """Fail-closed source or adoption error."""


def sha256_file(path: str | Path) -> str:
    return formal.sha256_file(path)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _pretty_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _identity(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise AdoptionError(f"source is not a regular file: {path}")
    return {
        "path": str(path.resolve()), "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _source_closure() -> dict[str, Any]:
    files = {name: sha256_file(ROOT / name) for name in SOURCE_FILES}
    digest = hashlib.sha256()
    for name in sorted(files):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(files[name].encode())
        digest.update(b"\0")
    return {
        "scheme": "sha256(path-nul-sha256-nul)-v1",
        "sha256": digest.hexdigest(), "files": files,
    }


def _source_result_paths(seed: int) -> tuple[Path, Path]:
    root = PRIOR_ARTIFACT / f"eval/seed_{seed}"
    return root / "results.json", root / "table.md"


def _normalized_validation_view(blob: Mapping[str, Any]) -> dict[str, Any]:
    meta = blob.get("meta")
    if not isinstance(meta, Mapping):
        raise AdoptionError("completed result omitted metadata")
    identity = meta.get("eval_identity")
    policy = meta.get("policy")
    if not isinstance(identity, Mapping) or not isinstance(policy, Mapping):
        raise AdoptionError("completed result omitted identity/policy provenance")
    expected_identity_keys = {
        "version", "checkpoint", "backend", "policy_kw", "policy_source",
        "policy_seed_scheme",
    }
    if set(identity) != expected_identity_keys:
        raise AdoptionError("recorded evaluation identity fields changed")
    if identity.get("policy_kw") != RECORDED_POLICY_KW:
        raise AdoptionError("recorded policy_kw is not the exact historical two-key form")
    expected_checkpoint = str((PRIOR_ARTIFACT / "checkpoint/ckpt.pt").resolve())
    checks = {
        "policy": policy.get("policy") == "LoomPolicy",
        "real": policy.get("is_stub") is False,
        "embodiment": policy.get("embodiment") == "libero_franka",
        "step": policy.get("ckpt_global_step") == STEP,
        "config": policy.get("ckpt_config_hash") == "d030206d56a71718",
        "checkpoint": (
            meta.get("ckpt") == identity.get("checkpoint")
            == policy.get("ckpt") == expected_checkpoint
        ),
        "dwell": policy.get("gripper_dwell") == 1,
        "samples": policy.get("decoder_samples") == 1,
        "duration": policy.get("duration_normalize_segments") is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AdoptionError(f"recorded real-policy provenance failed: {failed}")
    normalized = copy.deepcopy(dict(blob))
    normalized["meta"]["eval_identity"]["policy_kw"] = dict(
        formal.CANDIDATE_POLICY_KW
    )
    if blob["meta"]["eval_identity"]["policy_kw"] != RECORDED_POLICY_KW:
        raise AdoptionError("in-memory validation view mutated the source object")
    return normalized


def _normalization_receipt(blob: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _normalized_validation_view(blob)
    original_identity = blob["meta"]["eval_identity"]
    normalized_identity = normalized["meta"]["eval_identity"]
    return {
        **NORMALIZATION,
        "source_blob_sha256": hashlib.sha256(
            _canonical_json(blob).encode()
        ).hexdigest(),
        "recorded_eval_identity_sha256": hashlib.sha256(
            _canonical_json(original_identity).encode()
        ).hexdigest(),
        "normalized_validation_view_sha256": hashlib.sha256(
            _canonical_json(normalized).encode()
        ).hexdigest(),
        "normalized_eval_identity_sha256": hashlib.sha256(
            _canonical_json(normalized_identity).encode()
        ).hexdigest(),
        "source_bytes_rewritten": False,
        "normalization_scope": "in_memory_validator_view_only",
    }


def _load_validated_result(
    seed: int,
) -> tuple[dict[str, Any], Any, dict[tuple[str, str, int, int, int], dict[str, Any]]]:
    result_path, table_path = _source_result_paths(seed)
    if sha256_file(result_path) != PRIOR_RESULT_SHA256[seed]:
        raise AdoptionError(f"seed {seed} completed result SHA changed")
    if sha256_file(table_path) != PRIOR_TABLE_SHA256[seed]:
        raise AdoptionError(f"seed {seed} table SHA changed")
    try:
        blob = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AdoptionError(f"seed {seed} result unreadable") from exc
    if not isinstance(blob, Mapping):
        raise AdoptionError(f"seed {seed} result is not an object")
    blob = dict(blob)
    normalized = _normalized_validation_view(blob)
    try:
        rows = formal._validate_exact_eval_blob(
            normalized, seed=seed, label=f"adopted diagnostic seed {seed}",
            identity_profile="current_candidate",
        )
        formal._validate_eval_method_identity(
            normalized, label=f"adopted diagnostic seed {seed}",
            identity_profile="current_candidate", checkpoint_step=STEP,
            checkpoint_path=str((PRIOR_ARTIFACT / "checkpoint/ckpt.pt").resolve()),
        )
        protocol = formal._result_protocol(normalized)
    except formal.ChainError as exc:
        raise AdoptionError(str(exc)) from exc
    successes = sum(row.get("success") is True for row in blob["episodes"])
    if successes != PRIOR_SUCCESS_COUNTS[seed]:
        raise AdoptionError(f"seed {seed} success count changed")
    if v1._seed_markdown(blob) != table_path.read_text():
        raise AdoptionError(f"seed {seed} table differs from exact result rendering")
    return blob, protocol, rows


def _prior_attempt() -> dict[str, Any]:
    paths = {
        "plan": (PRIOR_CONTROL / "plan.json", PRIOR_PLAN_SHA256),
        "jobs": (PRIOR_CONTROL / "jobs.json", PRIOR_JOBS_SHA256),
        "release": (PRIOR_CONTROL / "released.json", PRIOR_RELEASED_SHA256),
        "checkpoint_receipt": (
            PRIOR_CONTROL / "checkpoint_receipt.json",
            PRIOR_CHECKPOINT_RECEIPT_SHA256,
        ),
        "checkpoint_report": (
            PRIOR_CONTROL / "checkpoint_verification.json",
            PRIOR_CHECKPOINT_REPORT_SHA256,
        ),
    }
    identities = {}
    for name, (path, expected) in paths.items():
        identity = _identity(path)
        if identity["sha256"] != expected:
            raise AdoptionError(f"prior {name} identity changed")
        identities[name] = identity
    log_identities = {}
    for name, (path, expected) in PRIOR_LOGS.items():
        identity = _identity(path)
        if identity["sha256"] != expected:
            raise AdoptionError(f"prior {name} job log identity changed")
        log_identities[name] = identity
    for seed in formal.SEEDS:
        log_text = PRIOR_LOGS[f"seed{seed}"][0].read_text()
        if (
            "POST_ABORT_DIAGNOSTIC_INVALID" not in log_text
            or "identity_policy_kw" not in log_text
            or "[loom.eval] 400/400 episodes" not in log_text
            or "Ran 400/400 episodes, 0 crashed" not in log_text
        ):
            raise AdoptionError(f"prior seed {seed} log does not prove the exact failure")
    prior_plan = json.loads((PRIOR_CONTROL / "plan.json").read_text())
    if (
        prior_plan.get("format_version") != 1
        or prior_plan.get("kind") != v1.KIND
        or prior_plan.get("diagnostic_source_closure") != v1._source_closure()
        or prior_plan.get("eligibility", {}).get("formal_eligible") is not False
        or prior_plan.get("trigger", {}).get("formal_status_remains") != "ABORT"
    ):
        raise AdoptionError("prior diagnostic plan is not the authenticated failed v1")
    jobs = json.loads((PRIOR_CONTROL / "jobs.json").read_text()).get("jobs")
    if jobs != {
        "consolidate": "32639504", "eval_seed0": "32639505",
        "eval_seed1": "32639506", "eval_seed2": "32639507",
        "merge": "32639508",
    }:
        raise AdoptionError("prior diagnostic job IDs changed")
    checkpoint = PRIOR_ARTIFACT / "checkpoint/ckpt.pt"
    checkpoint_identity = _identity(checkpoint)
    if (
        checkpoint_identity["sha256"] != PRIOR_CHECKPOINT_SHA256
        or checkpoint_identity["bytes"] != PRIOR_CHECKPOINT_BYTES
        or v1._checkpoint_step(checkpoint) != STEP
    ):
        raise AdoptionError("prior consolidated checkpoint changed")
    result_receipts = {}
    for seed in formal.SEEDS:
        blob, _protocol, _rows = _load_validated_result(seed)
        result_path, table_path = _source_result_paths(seed)
        result_receipts[str(seed)] = {
            "result": _identity(result_path), "table": _identity(table_path),
            "episodes": blob["summary"]["n_episodes"],
            "errors": blob["summary"]["n_errors"],
            "successes": PRIOR_SUCCESS_COUNTS[seed], "avg": blob["summary"]["avg"],
            "recorded_policy_kw": blob["meta"]["eval_identity"]["policy_kw"],
            "policy_embodiment": blob["meta"]["policy"]["embodiment"],
        }
    unexpectedly_published = [
        str(PRIOR_CONTROL / f"eval_seed_{seed}_receipt.json")
        for seed in formal.SEEDS
        if (PRIOR_CONTROL / f"eval_seed_{seed}_receipt.json").exists()
    ]
    if (PRIOR_CONTROL / "merged_eval_receipt.json").exists():
        unexpectedly_published.append(str(PRIOR_CONTROL / "merged_eval_receipt.json"))
    if unexpectedly_published:
        raise AdoptionError(f"prior failed attempt unexpectedly published: {unexpectedly_published}")
    return {
        "kind": "failed_v1_identity_validator_after_complete_results_v1",
        "failure": "identity_policy_kw_instrumentation_mismatch",
        "episode_execution_valid": True,
        "receipt_publication_valid": False,
        "files": identities, "job_logs": log_identities,
        "checkpoint": checkpoint_identity,
        "results": result_receipts, "terminal_job_states": PRIOR_JOB_STATES,
        "old_seed_and_merge_receipts_absent": True,
        "immutable_v1_source_closure": v1._source_closure(),
    }


def _expected_paths(control_dir: Path, artifact_root: Path) -> dict[str, Any]:
    return {
        "checkpoint_report": str(control_dir / "adopted_checkpoint_verification.json"),
        "checkpoint_receipt": str(control_dir / "adopted_checkpoint_receipt.json"),
        "merged_results": str(artifact_root / "eval/merged/results.json"),
        "merged_table": str(artifact_root / "eval/merged/table.md"),
        "merged_receipt": str(control_dir / "adopted_merged_eval_receipt.json"),
        "eval_receipts": {
            str(seed): str(control_dir / f"adopted_seed_{seed}_receipt.json")
            for seed in formal.SEEDS
        },
    }


def _require_isolated(control_dir: Path, artifact_root: Path) -> None:
    if (
        control_dir == artifact_root
        or control_dir.is_relative_to(artifact_root)
        or artifact_root.is_relative_to(control_dir)
    ):
        raise AdoptionError(
            "adoption control and artifact roots must be mutually disjoint"
        )
    protected = (
        v1.SOURCE_RUN_DIR, v1.SOURCE_CONTROL_DIR, v1.SOURCE_FORMAL_ARTIFACT_ROOT,
        PRIOR_CONTROL, PRIOR_ARTIFACT,
    )
    for candidate in (control_dir, artifact_root):
        if not candidate.is_absolute():
            raise AdoptionError("adoption roots must be absolute")
        if any(character in str(candidate) for character in (",", "\n", "\r")):
            raise AdoptionError("adoption roots contain unsafe Slurm export delimiters")
        for source in protected:
            if (
                candidate == source or candidate.is_relative_to(source)
                or source.is_relative_to(candidate)
            ):
                raise AdoptionError(f"adoption output overlaps immutable source {source}")


def _expected_baseline(source_plan: Mapping[str, Any]) -> dict[str, Any]:
    baseline = formal._authenticate_baseline(formal.CANONICAL_BASELINE_ROOT)
    return {
        **source_plan["baseline_comparison"], "baseline": dict(baseline),
        "role": "counterfactual_post_abort_diagnostic_comparison",
        "cannot_change_formal_status": True,
    }


def build_plan(
    *, control_dir: Path, artifact_root: Path, group: str, project: str = PROJECT,
) -> dict[str, Any]:
    control_dir, artifact_root = control_dir.resolve(), artifact_root.resolve()
    _require_isolated(control_dir, artifact_root)
    if control_dir.exists() or artifact_root.exists():
        raise AdoptionError("format-v2 adoption output roots must be fresh")
    if group != GROUP or project != PROJECT:
        raise AdoptionError("adoption W&B project/group differs from frozen identity")
    source_plan = v1._load_source_plan()
    prior = _prior_attempt()
    return {
        "format_version": FORMAT_VERSION, "kind": KIND,
        "eligibility": dict(EXPECTED_ELIGIBILITY),
        "method": dict(EXPECTED_METHOD),
        "diagnostic_source_closure": _source_closure(),
        "immutable_v1_source_closure": v1._source_closure(),
        "trigger": {
            "formal_abort": v1._collect_trigger(hash_shards=True),
            "failed_v1_attempt": prior,
        },
        "source_formal_config": source_plan["config"],
        "lineage": {
            "source_formal_run_dir": str(v1.SOURCE_RUN_DIR),
            "source_formal_control_dir": str(v1.SOURCE_CONTROL_DIR),
            "source_failed_v1_control_dir": str(PRIOR_CONTROL),
            "source_failed_v1_artifact_root": str(PRIOR_ARTIFACT),
            "adoption_control_dir": str(control_dir),
            "adoption_artifact_root": str(artifact_root),
            # Compatibility alias used only by the immutable diagnostic W&B
            # publisher.  It is required to be exactly the fresh v2 control dir.
            "diagnostic_control_dir": str(control_dir),
        },
        "evaluation": source_plan["evaluation"],
        "baseline_comparison": _expected_baseline(source_plan),
        "paths": _expected_paths(control_dir, artifact_root),
        "wandb": {
            "project": project, "group": group, "require_online": True,
            "tags": [
                "post-abort-diagnostic", "not-formal", "instrumentation-repair",
                "adopted-no-rerun", "r0", "dual-action",
            ],
            "stage_run_ids": {
                stage: uuid.uuid4().hex[:16]
                for stage in (
                    "adopt-checkpoint", "adopt-seed-0", "adopt-seed-1",
                    "adopt-seed-2", "adopt-summary",
                )
            },
            "artifact_policy": "receipts_and_results_only_no_checkpoint_bytes",
        },
    }


def _assert_trigger(plan: Mapping[str, Any], *, rehash_shards: bool) -> None:
    trigger = plan.get("trigger")
    if not isinstance(trigger, Mapping):
        raise AdoptionError("adoption plan omitted trigger")
    try:
        v1._assert_trigger(
            {"trigger": trigger.get("formal_abort")},
            rehash_shards=rehash_shards,
        )
    except v1.DiagnosticError as exc:
        raise AdoptionError(str(exc)) from exc
    if trigger.get("failed_v1_attempt") != _prior_attempt():
        raise AdoptionError("failed v1 attempt or completed result bytes changed")


def _assert_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("format_version") != FORMAT_VERSION or plan.get("kind") != KIND:
        raise AdoptionError("unsupported adoption plan")
    if (
        plan.get("diagnostic_source_closure") != _source_closure()
        or plan.get("immutable_v1_source_closure") != v1._source_closure()
    ):
        raise AdoptionError("adoption or immutable v1 source closure changed")
    if (
        plan.get("eligibility") != EXPECTED_ELIGIBILITY
        or plan.get("method") != EXPECTED_METHOD
    ):
        raise AdoptionError("adoption method/eligibility changed")
    lineage = plan.get("lineage", {})
    fixed_lineage = {
        "source_formal_run_dir": str(v1.SOURCE_RUN_DIR),
        "source_formal_control_dir": str(v1.SOURCE_CONTROL_DIR),
        "source_failed_v1_control_dir": str(PRIOR_CONTROL),
        "source_failed_v1_artifact_root": str(PRIOR_ARTIFACT),
    }
    if not isinstance(lineage, Mapping) or set(lineage) != {
        *fixed_lineage,
        "adoption_control_dir", "adoption_artifact_root",
        "diagnostic_control_dir",
    }:
        raise AdoptionError("adoption lineage fields changed")
    if any(lineage.get(key) != value for key, value in fixed_lineage.items()):
        raise AdoptionError("source lineage changed")
    control_dir = Path(lineage["adoption_control_dir"]).resolve()
    artifact_root = Path(lineage["adoption_artifact_root"]).resolve()
    if Path(lineage["diagnostic_control_dir"]).resolve() != control_dir:
        raise AdoptionError("diagnostic W&B path is not the adoption control dir")
    _require_isolated(control_dir, artifact_root)
    if plan.get("paths") != _expected_paths(control_dir, artifact_root):
        raise AdoptionError("adoption output paths changed")
    for raw in (
        *[value for key, value in plan["paths"].items() if key != "eval_receipts"],
        *plan["paths"]["eval_receipts"].values(),
    ):
        path = Path(raw)
        if path.is_relative_to(artifact_root):
            root = artifact_root
        elif path.is_relative_to(control_dir):
            root = control_dir
        else:
            raise AdoptionError("adoption output path is outside isolated roots")
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise AdoptionError("adoption output path escapes isolated root")
    source_plan = v1._load_source_plan()
    if (
        plan.get("source_formal_config") != source_plan["config"]
        or plan.get("evaluation") != source_plan["evaluation"]
        or plan.get("baseline_comparison") != _expected_baseline(source_plan)
    ):
        raise AdoptionError("formal config/evaluation/baseline contract changed")
    wandb = plan.get("wandb", {})
    stage_ids = wandb.get("stage_run_ids", {})
    prior_plan = json.loads((PRIOR_CONTROL / "plan.json").read_text())
    forbidden_ids = {
        source_plan["wandb"]["training_run_id"],
        *source_plan["wandb"]["stage_run_ids"].values(),
        *prior_plan["wandb"]["stage_run_ids"].values(),
    }
    expected_stages = {
        "adopt-checkpoint", "adopt-seed-0", "adopt-seed-1",
        "adopt-seed-2", "adopt-summary",
    }
    if not (
        wandb.get("project") == PROJECT and wandb.get("group") == GROUP
        and wandb.get("require_online") is True
        and wandb.get("tags") == [
            "post-abort-diagnostic", "not-formal", "instrumentation-repair",
            "adopted-no-rerun", "r0", "dual-action",
        ]
        and wandb.get("artifact_policy")
        == "receipts_and_results_only_no_checkpoint_bytes"
        and isinstance(stage_ids, Mapping) and set(stage_ids) == expected_stages
        and len(set(stage_ids.values())) == len(expected_stages)
        and all(
            isinstance(value, str) and re.fullmatch(r"[0-9a-f]{16}", value)
            for value in stage_ids.values()
        )
        and set(stage_ids.values()).isdisjoint(forbidden_ids)
    ):
        raise AdoptionError("adoption W&B identity changed")
    _assert_trigger(plan, rehash_shards=False)


def load_plan(path: str | Path, expected_sha256: str | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    if expected_sha256 and sha256_file(path) != expected_sha256:
        raise AdoptionError("adoption plan SHA mismatch")
    try:
        plan = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AdoptionError("adoption plan unreadable") from exc
    if not isinstance(plan, Mapping):
        raise AdoptionError("adoption plan is not an object")
    plan = dict(plan)
    _assert_plan(plan)
    return plan


def _plan_sha() -> str:
    path = os.environ.get("ADOPTION_PLAN")
    if not path:
        raise AdoptionError("ADOPTION_PLAN is required")
    return sha256_file(path)


def _read_receipt(path: Path, kind: str) -> dict[str, Any]:
    try:
        receipt = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AdoptionError(f"invalid {kind} receipt") from exc
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("format_version") != FORMAT_VERSION
        or receipt.get("kind") != kind
    ):
        raise AdoptionError(f"invalid {kind} receipt")
    return dict(receipt)


def _checkpoint_receipt_payload(plan: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = PRIOR_ARTIFACT / "checkpoint/ckpt.pt"
    report = Path(plan["paths"]["checkpoint_report"])
    return {
        "format_version": FORMAT_VERSION, "kind": CHECKPOINT_KIND,
        "adoption_plan_sha256": _plan_sha(),
        "formal_eligible": False, "source_formal_status": "ABORT",
        "source_formal_reason": "health_gate_failed", "step": STEP,
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "verification_report": str(report),
        "verification_report_sha256": sha256_file(report),
        "fresh_verify_only_pass": True,
        "fresh_verification_matches_failed_v1_report_bitwise": True,
        "adopted_from_v1_plan_sha256": PRIOR_PLAN_SHA256,
        "adopted_from_v1_checkpoint_receipt_sha256": PRIOR_CHECKPOINT_RECEIPT_SHA256,
        "checkpoint_reconstruction_performed": False,
        "checkpoint_bytes_uploaded_to_wandb": False,
        "training_updates_performed": 0, "optimizer_steps_performed": 0,
    }


def _validate_checkpoint_receipt(plan: Mapping[str, Any]) -> dict[str, Any]:
    receipt = _read_receipt(Path(plan["paths"]["checkpoint_receipt"]), CHECKPOINT_KIND)
    checkpoint = PRIOR_ARTIFACT / "checkpoint/ckpt.pt"
    report = Path(plan["paths"]["checkpoint_report"])
    if (
        sha256_file(checkpoint) != PRIOR_CHECKPOINT_SHA256
        or checkpoint.stat().st_size != PRIOR_CHECKPOINT_BYTES
        or v1._checkpoint_step(checkpoint) != STEP
        or sha256_file(report) != PRIOR_CHECKPOINT_REPORT_SHA256
        or receipt != _checkpoint_receipt_payload(plan)
    ):
        raise AdoptionError("adopted checkpoint receipt failed exact recomputation")
    return receipt


def _stage_adopt_checkpoint(plan: Mapping[str, Any]) -> int:
    _assert_trigger(plan, rehash_shards=True)
    report_path = Path(plan["paths"]["checkpoint_report"])
    receipt_path = Path(plan["paths"]["checkpoint_receipt"])
    if receipt_path.exists():
        receipt = _validate_checkpoint_receipt(plan)
    else:
        if not report_path.exists():
            attempt_report = report_path.with_name(
                f".{report_path.name}.attempt-{uuid.uuid4().hex}"
            )
            try:
                completed = subprocess.run(
                    [
                        sys.executable, "-m", "loom.train.consolidate",
                        "--run_dir", str(v1.SOURCE_RUN_DIR),
                        "--step", str(STEP),
                        "--out", str(PRIOR_ARTIFACT / "checkpoint/ckpt.pt"),
                        "--config", str(v1.SOURCE_RUN_DIR / "config.json"),
                        "--verify_only", "--report", str(attempt_report),
                    ],
                    cwd=ROOT, text=True, check=True,
                )
                if completed.returncode != 0:
                    raise AdoptionError("fresh checkpoint verification failed")
                if sha256_file(attempt_report) != PRIOR_CHECKPOINT_REPORT_SHA256:
                    raise AdoptionError(
                        "fresh checkpoint verification differs from v1 report"
                    )
                formal._exclusive_text_write(report_path, attempt_report.read_text())
            finally:
                attempt_report.unlink(missing_ok=True)
        if sha256_file(report_path) != PRIOR_CHECKPOINT_REPORT_SHA256:
            raise AdoptionError("fresh checkpoint verification differs from v1 report")
        _assert_trigger(plan, rehash_shards=True)
        receipt = _checkpoint_receipt_payload(plan)
        formal.exclusive_json_write(receipt_path, receipt)
        _validate_checkpoint_receipt(plan)
    v1._wandb_publish(
        plan, stage="adopt-checkpoint", path=receipt_path,
        artifact_type="diagnostic-adopted-checkpoint-receipt",
        summary={"checkpoint_step": STEP, "adopted_no_reconstruction": True},
    )
    _assert_trigger(plan, rehash_shards=True)
    return 0


def _seed_receipt_payload(
    plan: Mapping[str, Any], seed: int, blob: Mapping[str, Any],
) -> dict[str, Any]:
    result, table = _source_result_paths(seed)
    return {
        "format_version": FORMAT_VERSION, "kind": SEED_KIND,
        "adoption_plan_sha256": _plan_sha(),
        "formal_eligible": False, "source_formal_status": "ABORT",
        "source_formal_reason": "health_gate_failed", "seed": seed,
        "source_result": str(result), "source_result_sha256": sha256_file(result),
        "source_table": str(table), "source_table_sha256": sha256_file(table),
        "episodes": 400, "errors": 0,
        "successes": PRIOR_SUCCESS_COUNTS[seed], "avg": blob["summary"]["avg"],
        "checkpoint_step": STEP, "checkpoint_sha256": PRIOR_CHECKPOINT_SHA256,
        "checkpoint_receipt_sha256": sha256_file(
            plan["paths"]["checkpoint_receipt"]
        ),
        "identity_normalization": _normalization_receipt(blob),
        "completed_result_adopted": True, "environment_episodes_rerun": 0,
        "training_updates_performed": 0, "optimizer_steps_performed": 0,
    }


def _validate_seed_receipt(
    plan: Mapping[str, Any], seed: int,
) -> tuple[dict[str, Any], dict[str, Any], Any, dict[Any, Any]]:
    receipt = _read_receipt(
        Path(plan["paths"]["eval_receipts"][str(seed)]), SEED_KIND,
    )
    _validate_checkpoint_receipt(plan)
    blob, protocol, rows = _load_validated_result(seed)
    if receipt != _seed_receipt_payload(plan, seed, blob):
        raise AdoptionError(f"seed {seed} adoption receipt differs from recomputation")
    return receipt, blob, protocol, rows


def _stage_adopt_seed(plan: Mapping[str, Any], stage: str) -> int:
    match = re.fullmatch(r"adopt_seed([0-9]+)", stage)
    if match is None or int(match.group(1)) not in formal.SEEDS:
        raise AdoptionError(f"invalid adoption seed stage {stage}")
    seed = int(match.group(1))
    _assert_trigger(plan, rehash_shards=False)
    _validate_checkpoint_receipt(plan)
    blob, _protocol, _rows = _load_validated_result(seed)
    receipt_path = Path(plan["paths"]["eval_receipts"][str(seed)])
    if receipt_path.exists():
        _validate_seed_receipt(plan, seed)
    else:
        formal.exclusive_json_write(
            receipt_path, _seed_receipt_payload(plan, seed, blob),
        )
        _validate_seed_receipt(plan, seed)
    result_path, _ = _source_result_paths(seed)
    v1._wandb_publish(
        plan, stage=f"adopt-seed-{seed}", path=result_path,
        artifact_type="diagnostic-adopted-evaluation-results",
        summary={
            "seed": seed, "episodes": 400, "successes": PRIOR_SUCCESS_COUNTS[seed],
            "success_rate": blob["summary"]["avg"], "checkpoint_step": STEP,
            "environment_episodes_rerun": 0,
        },
    )
    _assert_trigger(plan, rehash_shards=False)
    return 0


def merge_seed_results(plan: Mapping[str, Any]) -> dict[str, Any]:
    from loom.eval import EpisodeResult  # noqa: PLC0415
    from loom.eval.runner import aggregate, iter_work  # noqa: PLC0415

    blobs, protocols, all_rows, receipts = [], [], {}, []
    for seed in formal.SEEDS:
        receipt, blob, protocol, rows = _validate_seed_receipt(plan, seed)
        if all_rows.keys() & rows.keys():
            raise AdoptionError("adopted singleton seed rows overlap")
        all_rows.update(rows)
        blobs.append(blob)
        protocols.append(protocol)
        receipts.append(receipt)
    reference = formal._protocol_without_seeds(protocols[0])
    if any(formal._protocol_without_seeds(item) != reference for item in protocols[1:]):
        raise AdoptionError("adopted singleton protocols differ")
    target = protocols[0].replace(seeds=formal.SEEDS)
    records = [
        EpisodeResult.from_dict(row)
        for blob in blobs for row in blob.get("episodes", [])
    ]
    keys = [row.key() for row in records]
    expected = {item.key() for item in iter_work(target)}
    if len(keys) != len(set(keys)) or set(keys) != expected or len(keys) != 1_200:
        raise AdoptionError("adopted union is not the exact 1200-episode protocol")
    common_ckpt = blobs[0]["meta"]["ckpt"]
    common_identity = blobs[0]["meta"]["eval_identity"]
    if any(
        blob["meta"]["ckpt"] != common_ckpt
        or blob["meta"]["eval_identity"] != common_identity
        for blob in blobs[1:]
    ):
        raise AdoptionError("adopted checkpoint/evaluation identities differ")
    try:
        comparison = formal.paired_baseline_comparison(
            plan, all_rows, formal._baseline_rows(plan),
        )
    except formal.ChainError as exc:
        raise AdoptionError(str(exc)) from exc
    comparison = {
        **comparison,
        "role": "counterfactual_post_abort_diagnostic_gate",
        "formal_eligible": False, "cannot_reverse_formal_abort": True,
        "source_results_adopted_without_episode_rerun": True,
    }
    summary = aggregate(records, target)
    if (
        summary.get("complete") is not True
        or summary.get("n_episodes") != 1_200
        or summary.get("n_expected") != 1_200
        or summary.get("n_errors") != 0
    ):
        raise AdoptionError("adopted merged result failed exact closure")
    return {
        "version": blobs[0].get("version", 1),
        "kind": "r0_e2e_post_abort_diagnostic_adopted_merged_results_v2",
        "formal_eligible": False, "source_formal_status": "ABORT",
        "source_formal_reason": "health_gate_failed", "checkpoint_step": STEP,
        "bench": target.bench, "protocol": target.to_dict(),
        "meta": {
            "ckpt": common_ckpt, "eval_identity": common_identity,
            "policy": blobs[0]["meta"]["policy"],
            "source_seed_adoption_receipts": receipts,
            "identity_normalization": NORMALIZATION,
            "merge_provenance": {
                "kind": "post_abort_completed_result_adoption_merge_v2",
                "adoption_plan_sha256": _plan_sha(),
                "adoption_source_closure_sha256": plan[
                    "diagnostic_source_closure"
                ]["sha256"],
                "failed_v1_plan_sha256": PRIOR_PLAN_SHA256,
                "formal_eligible": False,
            },
            "baseline_receipt": plan["baseline_comparison"]["baseline"],
            "interpretation": "diagnostic only; cannot reverse formal ABORT",
        },
        "summary": summary, "diagnostic_baseline_comparison": comparison,
        "episodes": [
            row.to_dict() for row in sorted(records, key=lambda item: item.key())
        ],
    }


def _markdown(merged: Mapping[str, Any]) -> str:
    return v1._diagnostic_markdown(merged).replace(
        "# Post-ABORT diagnostic evaluation (not formal)",
        "# Adopted post-ABORT diagnostic evaluation (not formal; zero rerun)",
    )


def _merged_receipt_payload(
    plan: Mapping[str, Any], merged: Mapping[str, Any],
    result_path: Path, table_path: Path,
) -> dict[str, Any]:
    comparison = merged["diagnostic_baseline_comparison"]
    return {
        "format_version": FORMAT_VERSION, "kind": MERGED_KIND,
        "adoption_plan_sha256": _plan_sha(),
        "formal_eligible": False, "promotion_eligible": False,
        "source_formal_status": "ABORT", "source_formal_reason": "health_gate_failed",
        "result": str(result_path), "result_sha256": sha256_file(result_path),
        "table": str(table_path), "table_sha256": sha256_file(table_path),
        "episodes": merged["summary"]["n_episodes"],
        "errors": merged["summary"]["n_errors"], "avg": merged["summary"]["avg"],
        "successes": sum(PRIOR_SUCCESS_COUNTS.values()),
        "complete": merged["summary"]["complete"], "checkpoint_step": STEP,
        "source_seed_adoption_receipt_sha256": {
            str(seed): sha256_file(plan["paths"]["eval_receipts"][str(seed)])
            for seed in formal.SEEDS
        },
        "diagnostic_comparison_sha256": hashlib.sha256(
            _canonical_json(comparison).encode()
        ).hexdigest(),
        "bootstrap_matrix_sha256": formal.BOOTSTRAP_MATRIX_SHA256,
        "baseline_delta_pp": comparison["overall"]["delta_percentage_points"],
        "paired_ci_low_pp": comparison["paired_task_bootstrap"][
            "ci_low_percentage_points"
        ],
        "paired_ci_high_pp": comparison["paired_task_bootstrap"][
            "ci_high_percentage_points"
        ],
        "per_suite": comparison["per_suite"],
        "counterfactual_diagnostic_gate_status": comparison["status"],
        "counterfactual_diagnostic_gate_passed": comparison["passed"],
        "failed_diagnostic_checks": comparison["failed_checks"],
        "formal_status_unchanged_by_result": True,
        "environment_episodes_rerun": 0,
        "training_updates_performed": 0, "optimizer_steps_performed": 0,
    }


def _validate_merged_receipt(plan: Mapping[str, Any]) -> dict[str, Any]:
    receipt = _read_receipt(Path(plan["paths"]["merged_receipt"]), MERGED_KIND)
    merged = merge_seed_results(plan)
    result = Path(plan["paths"]["merged_results"])
    table = Path(plan["paths"]["merged_table"])
    if (
        not result.is_file() or result.read_text() != _pretty_json(merged)
        or not table.is_file() or table.read_text() != _markdown(merged)
        or receipt != _merged_receipt_payload(plan, merged, result, table)
    ):
        raise AdoptionError("adopted merged publication differs from recomputation")
    return receipt


def _stage_merge(plan: Mapping[str, Any]) -> int:
    _assert_trigger(plan, rehash_shards=False)
    result = Path(plan["paths"]["merged_results"])
    table = Path(plan["paths"]["merged_table"])
    receipt_path = Path(plan["paths"]["merged_receipt"])
    if receipt_path.exists():
        receipt = _validate_merged_receipt(plan)
        merged = json.loads(result.read_text())
    else:
        merged = merge_seed_results(plan)
        expected_result, expected_table = _pretty_json(merged), _markdown(merged)
        if result.exists():
            if not result.is_file() or result.read_text() != expected_result:
                raise AdoptionError("partial adopted merged result differs")
        else:
            formal.exclusive_json_write(result, merged)
        if table.exists():
            if not table.is_file() or table.read_text() != expected_table:
                raise AdoptionError("partial adopted merged table differs")
        else:
            formal._exclusive_text_write(table, expected_table)
        receipt = _merged_receipt_payload(plan, merged, result, table)
        formal.exclusive_json_write(receipt_path, receipt)
        _validate_merged_receipt(plan)
    comparison = merged["diagnostic_baseline_comparison"]
    v1._wandb_publish(
        plan, stage="adopt-summary", path=result,
        artifact_type="diagnostic-adopted-evaluation-summary",
        summary={
            "episodes": 1_200, "successes": sum(PRIOR_SUCCESS_COUNTS.values()),
            "success_rate": merged["summary"]["avg"], "n_errors": 0,
            "environment_episodes_rerun": 0,
            **v1._summary_fields(comparison),
        },
    )
    _assert_trigger(plan, rehash_shards=False)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    return 0


def _stage_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "adopt_checkpoint",
            "sbatch": "scripts/r0_e2e_post_abort_adopt_checkpoint.sbatch",
            "depends_on": [],
        },
        *[
            {
                "name": f"adopt_seed{seed}",
                "sbatch": "scripts/r0_e2e_post_abort_adopt_seed.sbatch",
                "depends_on": ["adopt_checkpoint"],
            }
            for seed in formal.SEEDS
        ],
        {
            "name": "merge_adopted",
            "sbatch": "scripts/r0_e2e_post_abort_adopt_control.sbatch",
            "depends_on": [f"adopt_seed{seed}" for seed in formal.SEEDS],
        },
    ]


def _sbatch_command(
    spec: Mapping[str, Any], plan_path: Path, plan_sha: str,
    dependencies: Sequence[str],
) -> list[str]:
    command = [
        "sbatch", "--parsable", "--hold", "--kill-on-invalid-dep=yes",
        f"--job-name={('r0diag_adopt_v2_' + spec['name'])[:120]}",
    ]
    if dependencies:
        command.append("--dependency=afterok:" + ":".join(dependencies))
    command.extend((
        "--export=" + ",".join((
            "ALL", f"ADOPTION_PLAN={plan_path}",
            f"ADOPTION_PLAN_SHA256={plan_sha}",
            f"ADOPTION_STAGE={spec['name']}",
        )),
        str(ROOT / spec["sbatch"]),
    ))
    return command


def submit_plan(
    plan: Mapping[str, Any], *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    _assert_plan(plan)
    control = Path(plan["lineage"]["adoption_control_dir"])
    artifact = Path(plan["lineage"]["adoption_artifact_root"])
    if control.exists() or artifact.exists():
        raise AdoptionError("adoption output roots are not fresh")
    control.mkdir(parents=True, exist_ok=False)
    artifact.mkdir(parents=True, exist_ok=False)
    plan_path = control / "plan.json"
    formal.exclusive_json_write(plan_path, plan)
    plan_sha = sha256_file(plan_path)
    job_ids, commands, submitted = {}, {}, []
    try:
        for spec in _stage_specs():
            command = _sbatch_command(
                spec, plan_path, plan_sha,
                [job_ids[name] for name in spec["depends_on"]],
            )
            completed = run(
                command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, check=True,
            )
            job_id = completed.stdout.strip().split(";", 1)[0]
            if re.fullmatch(r"[0-9]+", job_id) is None:
                raise AdoptionError(f"invalid adoption job id {job_id!r}")
            job_ids[spec["name"]] = job_id
            commands[spec["name"]] = command
            submitted.append(job_id)
        jobs = {
            "format_version": FORMAT_VERSION,
            "kind": "r0_e2e_post_abort_diagnostic_adoption_jobs_v2",
            "adoption_plan": str(plan_path), "adoption_plan_sha256": plan_sha,
            "formal_eligible": False, "environment_episodes_rerun": 0,
            "jobs": job_ids, "commands": commands, "released": False,
        }
        formal.exclusive_json_write(control / "jobs.json", jobs)
        run(
            ["scontrol", "release", ",".join(submitted)], cwd=ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
        )
        formal.exclusive_json_write(control / "released.json", {
            "format_version": FORMAT_VERSION,
            "kind": "r0_e2e_post_abort_diagnostic_adoption_release_v2",
            "adoption_plan_sha256": plan_sha,
            "jobs_sha256": sha256_file(control / "jobs.json"),
            "job_ids": submitted, "released": True,
        })
        return {**jobs, "released": True}
    except Exception:
        if submitted:
            subprocess.run(
                ["scancel", *submitted], cwd=ROOT, check=False,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        raise


def run_stage() -> int:
    path = os.environ.get("ADOPTION_PLAN")
    digest = os.environ.get("ADOPTION_PLAN_SHA256")
    stage = os.environ.get("ADOPTION_STAGE")
    if not path or not digest or not stage:
        raise AdoptionError("ADOPTION_PLAN/SHA256/STAGE are required")
    diagnostic_plan = os.environ.get("DIAGNOSTIC_PLAN")
    if diagnostic_plan not in (None, path):
        raise AdoptionError("DIAGNOSTIC_PLAN conflicts with the adoption plan")
    os.environ["DIAGNOSTIC_PLAN"] = path
    os.environ["DIAGNOSTIC_PLAN_SHA256"] = digest
    plan = load_plan(path, digest)
    if stage == "adopt_checkpoint":
        return _stage_adopt_checkpoint(plan)
    if stage.startswith("adopt_seed"):
        return _stage_adopt_seed(plan, stage)
    if stage == "merge_adopted":
        return _stage_merge(plan)
    raise AdoptionError(f"unknown adoption stage {stage}")


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
            return run_stage()
        control, artifact = Path(args.control_dir), Path(args.artifact_root)
        if not control.is_absolute() or not artifact.is_absolute():
            raise AdoptionError("adoption output roots must be absolute")
        plan = build_plan(
            control_dir=control, artifact_root=artifact,
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
    except (AdoptionError, formal.ChainError, v1.DiagnosticError, FileExistsError) as exc:
        print(f"POST_ABORT_ADOPTION_INVALID: {exc}", file=sys.stderr, flush=True)
        return 2
    except subprocess.CalledProcessError as exc:
        command = " ".join(shlex.quote(str(item)) for item in exc.cmd)
        print(
            f"POST_ABORT_ADOPTION_FAILED: command exited {exc.returncode}: {command}",
            file=sys.stderr, flush=True,
        )
        return int(exc.returncode or 1)


if __name__ == "__main__":
    raise SystemExit(main())
