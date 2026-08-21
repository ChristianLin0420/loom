#!/usr/bin/env python3
"""Read-only retrospective scan of four fixed early outcome-GRPO snapshots.

This is a diagnostic of the already-failed ``r0a_outcome_grpo_strict_s1``
run.  It cannot emit a checkpoint, select a checkpoint, authorize promotion,
or launch simulator evaluation.  The four steps are fixed in source before
execution.  Eight ranks evaluate one immutable task each: one exact heldout
surrogate/KL scan and one exact heldout trust scan per checkpoint.

The numerical implementations are not copied here.  Proposal construction,
checkpoint authentication, deterministic fp32 row-wise scoring, heldout
surrogate/KL evaluation, trust gates, and task-stratified paired bootstrap all
come directly from :mod:`loom.train.outcome_grpo`.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch import Tensor  # noqa: E402

from loom.eval import outcome_recovery as recovery  # noqa: E402
from loom.train import outcome_grpo as grpo  # noqa: E402
from loom.train.atomic import fsync_dir  # noqa: E402
from loom.train.determinism import set_global_seed  # noqa: E402
from loom.train.loop import read_config  # noqa: E402


FORMAT_VERSION = 1
KIND = "loom_outcome_grpo_retrospective_early_curve_diagnostic"
EXPECTED_WORLD_SIZE = 8
EARLY_CHECKPOINT_STEPS = (49_866, 50_066, 50_466, 51_266)
SCAN_TASKS = tuple(
    task
    for step in EARLY_CHECKPOINT_STEPS
    for task in (("surrogate", step), ("trust", step))
)

FAILED_RUN_REL = "runs/r0a_outcome_grpo_strict_s1"
OUTPUT_DIR_REL = "runs/r0a_outcome_grpo_strict_s1_early_curve_diagnostic"
RECIPE_CONFIG_REL = "configs/r0a_outcome_grpo.yaml"
RUN_CONFIG_REL = f"{FAILED_RUN_REL}/config.json"
TERMINAL_REPORT_REL = f"{FAILED_RUN_REL}/terminal_report.json"

# These bytes existed before this diagnostic was written.  Pinning them makes
# the retrospective question reproducible and prevents a same-path replacement
# from silently changing the curve under inspection.
EXPECTED_CONFIG_HASH = "25afdedfc9deea5e"
EXPECTED_TRAINER_SOURCE_SHA256 = (
    "d5ef53e9f2e276f17d68f80b4c081c8f09b0d89ea9a966214fc3b63387364a52"
)
EXPECTED_RUN_CONFIG_SHA256 = (
    "455346e374692370cb3255105d584c367378e98eb8c10f01ef0d947cd41686bc"
)
EXPECTED_RECIPE_CONFIG_SHA256 = (
    "74cba9855d9b18d69de67c55ee5c15b505edde2fa2c6de44da535dd311eeb1ec"
)
EXPECTED_TERMINAL_REPORT_SHA256 = (
    "e42de09dcc0cf691c570f8a73b06144ad09767915565f876136d96fd36c94337"
)
EXPECTED_CHECKPOINT_SHA256 = {
    49_866: "93cb5c0eec2186512feccc64c9e8f77b8ca7acb1ca278892e2e1f8dd6ec8dd04",
    50_066: "bb0e10f549bcb97a5f5bed741f7dbb7c1ec5b70928a2ebde96015e652f965e4b",
    50_466: "2f67499096f25eb446d34f6f7652180d64d9a9414bfc5491b811ef1fde389a58",
    51_266: "b6597ff1fa4deea40fcdf443ba11b5cbfc2273fceef673c9ad701be722527c36",
}

_SCAN_SOURCE_FILES = (
    "scripts/outcome_grpo_early_curve_scan.py",
    "scripts/outcome_grpo_early_curve_scan.sbatch",
)

_TRUST_CHECKS = {
    "clip_fraction": ("<=", grpo.MAX_CLIP_FRACTION),
    "ess_fraction": (">=", grpo.MIN_ESS_FRACTION),
    "coeff_drift_p95": ("<=", grpo.MAX_COEFF_DRIFT_P95),
    "live_ops": (">=", grpo.MIN_LIVE_OPS),
    "nonfinite": ("==", 0),
    "unexpected_gradients": ("==", 0),
}


class EarlyCurveScanError(RuntimeError):
    """An input, execution, aggregation, or diagnostic-only invariant failed."""


def _require(condition: Any, message: str) -> None:
    if not bool(condition):
        raise EarlyCurveScanError(message)


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _valid_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _repo_path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise EarlyCurveScanError(f"repository input escapes root: {relative}") from exc
    return path


def authenticated_file(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str,
    label: str,
) -> dict[str, Any]:
    """Hash one fixed file while rejecting replacement during the read."""
    value = Path(path).expanduser().resolve()
    _require(value.is_file(), f"missing {label}: {value}")
    before = value.stat()
    digest = recovery.sha256_file(value)
    after = value.stat()
    _require(
        int(before.st_size) == int(after.st_size)
        and int(before.st_mtime_ns) == int(after.st_mtime_ns),
        f"{label} changed while hashing: {value}",
    )
    _require(digest == expected_sha256,
             f"{label} SHA-256 differs: {digest} != {expected_sha256}")
    return {
        "path": str(value),
        "sha256": digest,
        "size": int(after.st_size),
        "mtime_ns": int(after.st_mtime_ns),
    }


def scan_source_identity() -> dict[str, Any]:
    """Bind execution to this diagnostic and its eight-rank launcher bytes."""
    digest = hashlib.sha256()
    files: dict[str, str] = {}
    for relative in sorted(_SCAN_SOURCE_FILES):
        path = _repo_path(relative)
        _require(path.is_file(), f"scan source is missing: {path}")
        sha = recovery.sha256_file(path)
        files[relative] = sha
        digest.update(relative.encode("utf-8") + b"\0" + bytes.fromhex(sha) + b"\0")
    return {
        "scheme": "sha256(path-nul-sha256-nul)-v1",
        "sha256": digest.hexdigest(),
        "files": files,
    }


def scan_task_assignment(rank: int, world: int) -> tuple[str, int]:
    """Map each of exactly eight ranks to one predeclared immutable task."""
    _require(world == EXPECTED_WORLD_SIZE,
             f"early-curve scan requires world={EXPECTED_WORLD_SIZE}, got {world}")
    _require(len(SCAN_TASKS) == world, "early-curve task/world geometry drifted")
    if not 0 <= int(rank) < int(world):
        raise ValueError(f"rank {rank} is outside world {world}")
    return SCAN_TASKS[int(rank)]


def _all_gather_object(value: Any, world: int) -> list[Any]:
    if world == 1:
        return [value]
    gathered: list[Any] = [None for _ in range(world)]
    torch.distributed.all_gather_object(gathered, value)
    return gathered


def _synchronise_errors(
    *, rank: int, world: int, label: str, error: str,
) -> None:
    rows = _all_gather_object({"rank": int(rank), "error": str(error)}, world)
    failed = [row for row in rows if str(row.get("error") or "")]
    _require(not failed, f"{label} failed: {failed[:2]}")


def _broadcast_root(value: Any, *, rank: int, world: int) -> Any:
    box = [value if rank == 0 else None]
    if world > 1:
        torch.distributed.broadcast_object_list(box, src=0)
    return box[0]


def _runtime(device: torch.device, *, rank: int, world: int) -> dict[str, Any]:
    _require(device.type == "cuda", "formal early-curve scan requires CUDA")
    properties = torch.cuda.get_device_properties(device)
    return {
        "rank": int(rank),
        "world_size": int(world),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": properties.name,
        "gpu_capability": list(torch.cuda.get_device_capability(device)),
        "visible_gpu_count": torch.cuda.device_count(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_local_id": os.environ.get("SLURM_LOCALID"),
    }


def _load_static_inputs() -> dict[str, Any]:
    run_config_path = _repo_path(RUN_CONFIG_REL)
    recipe_path = _repo_path(RECIPE_CONFIG_REL)
    terminal_path = _repo_path(TERMINAL_REPORT_REL)
    identities = {
        "run_config": authenticated_file(
            run_config_path,
            expected_sha256=EXPECTED_RUN_CONFIG_SHA256,
            label="failed-run resolved config",
        ),
        "recipe_config": authenticated_file(
            recipe_path,
            expected_sha256=EXPECTED_RECIPE_CONFIG_SHA256,
            label="locked recipe config",
        ),
        "terminal_report": authenticated_file(
            terminal_path,
            expected_sha256=EXPECTED_TERMINAL_REPORT_SHA256,
            label="failed terminal report",
        ),
    }
    try:
        resolved = json.loads(run_config_path.read_text(encoding="utf-8"))
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EarlyCurveScanError(f"cannot read fixed run metadata: {exc}") from exc
    canonical = read_config(recipe_path)
    grpo.validate_recipe_config(canonical)
    _require(resolved == _json_copy(canonical),
             "failed-run resolved config differs from the locked recipe")
    config_hash = grpo._config_hash(resolved)
    _require(config_hash == EXPECTED_CONFIG_HASH,
             f"failed-run config hash differs: {config_hash}")

    trainer_source = grpo._trainer_source_identity()
    _require(trainer_source.get("sha256") == EXPECTED_TRAINER_SOURCE_SHA256,
             "current trainer source closure differs from the failed run")
    _require(terminal.get("trainer_source") == trainer_source,
             "terminal report trainer source differs from current authenticated source")
    _require(
        terminal.get("status") == "FAIL"
        and terminal.get("passed") is False
        and terminal.get("candidate_emitted") is False,
        "retrospective input is not the fail-closed terminal report",
    )
    _require(int(terminal.get("global_step", -1)) == grpo.STOP_STEP,
             "failed terminal report is not the completed strict run")
    _require(str(terminal.get("config_hash") or "") == config_hash,
             "failed terminal report config hash differs")
    _require(terminal.get("strict_determinism") == grpo.STRICT_OUTCOME_DETERMINISM,
             "failed terminal report lacks strict determinism evidence")
    exact_auth = terminal.get("exact_behaviour_identity")
    _require(isinstance(exact_auth, Mapping) and exact_auth.get("passed") is True,
             "failed terminal report lacks exact behavior authentication")
    initial_identity = terminal.get("initial_behavior_ratio_identity")
    _require(isinstance(initial_identity, Mapping)
             and initial_identity.get("passed") is True,
             "failed terminal report lacks initial ratio identity")
    _require(not list(_repo_path(FAILED_RUN_REL).glob("candidate_*.pt")),
             "failed strict run unexpectedly contains a candidate")
    return {
        "config": resolved,
        "config_hash": config_hash,
        "terminal": terminal,
        "trainer_source": trainer_source,
        "exact_behaviour_identity": dict(exact_auth),
        "initial_behavior_identity": dict(initial_identity),
        "file_identities": identities,
        "scan_source": scan_source_identity(),
    }


def _authenticate_seed_reference(
    terminal: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove that the terminal-report seed rows name the pinned parent bytes."""
    seed_path = _repo_path(grpo.EXPECTED_SEED_CHECKPOINT)
    parent_identity = recovery.authenticate_seed_checkpoint(seed_path)
    _require(parent_identity.get("sha256") == recovery.SEED_CHECKPOINT_SHA256,
             "authenticated seed checkpoint SHA-256 differs")
    _require(terminal.get("parent") == parent_identity,
             "terminal report parent identity differs from authenticated seed")
    parent = grpo._load_parent_from_identity(parent_identity)
    proposal = grpo.proposal_model_digest(parent["model"])
    del parent
    gc.collect()

    convergence = terminal.get("convergence_gate")
    _require(isinstance(convergence, Mapping),
             "terminal report lacks convergence gate")
    snapshots = convergence.get("snapshots")
    _require(isinstance(snapshots, Mapping),
             "terminal report lacks convergence snapshots")
    seed_report = snapshots.get(str(grpo.START_STEP))
    _require(isinstance(seed_report, Mapping),
             "terminal report lacks the seed heldout report")
    expected_checkpoint = {**_json_copy(parent_identity), "proposal": proposal}
    _require(seed_report.get("checkpoint") == expected_checkpoint,
             "terminal seed report checkpoint identity differs from pinned parent")
    return {
        "parent_identity": _json_copy(parent_identity),
        "seed_checkpoint": expected_checkpoint,
    }


def _collection_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    outcome = dict(config["outcome_grpo"])
    return [dict(row) for row in outcome["folds"]] + [dict(outcome["validation"])]


def _authenticate_collections(
    config: Mapping[str, Any],
    terminal: Mapping[str, Any],
    parent_identity: Mapping[str, Any],
    *,
    rank: int,
    world: int,
) -> tuple[
    list[grpo.ValidatedRecoveryCollection],
    dict[int, grpo.ValidatedRecoveryCollection],
    list[dict[str, Any]],
]:
    """Deep-authenticate each collection once, distributed across ranks."""
    specs = _collection_specs(config)
    _require(len(specs) == grpo.N_FOLDS + 1,
             "fixed recipe does not name six folds plus validation")
    local_objects: dict[int, grpo.ValidatedRecoveryCollection] = {}
    local_rows: list[dict[str, Any]] = []
    try:
        for index, spec in enumerate(specs):
            if index % world != rank:
                continue
            collection = grpo.ValidatedRecoveryCollection.open(
                _repo_path(str(spec["path"])),
                checkpoint_identity=parent_identity,
                expected_split=str(spec["split"]),
                deep=True,
            )
            local_objects[index] = collection
            local_rows.append({
                "rank": int(rank), "index": int(index), "ok": True,
                "provenance": collection.provenance(),
            })
    except Exception as exc:  # noqa: BLE001
        local_rows.append({
            "rank": int(rank), "index": -1, "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        })
    gathered = _all_gather_object(local_rows, world)
    rows = [dict(row) for rank_rows in gathered for row in rank_rows]
    failures = [row for row in rows if not bool(row.get("ok"))]
    _require(not failures, f"deep collection authentication failed: {failures[:2]}")
    by_index = {int(row["index"]): dict(row["provenance"]) for row in rows}
    _require(set(by_index) == set(range(len(specs))),
             "deep collection authentication is missing or duplicated")

    expected_terminal = list(terminal.get("collections") or ()) + [
        terminal.get("validation")
    ]
    _require(len(expected_terminal) == len(specs),
             "terminal report collection provenance is incomplete")
    _require([by_index[index] for index in range(len(specs))]
             == expected_terminal,
             "deep-authenticated collection provenance differs from terminal report")

    all_collections: list[grpo.ValidatedRecoveryCollection] = []
    for index, spec in enumerate(specs):
        collection = local_objects.get(index)
        if collection is None:
            collection = grpo.ValidatedRecoveryCollection.open(
                _repo_path(str(spec["path"])),
                checkpoint_identity=parent_identity,
                expected_split=str(spec["split"]),
                deep=False,
                verify_sidecars=False,
            )
        grpo._assert_owner_collection_snapshot(collection, by_index[index])
        all_collections.append(collection)
    return all_collections, local_objects, [
        by_index[index] for index in range(len(specs))
    ]


def _validate_seed_groups(
    seed_report: Mapping[str, Any],
    validation: grpo.ValidatedRecoveryCollection,
    *,
    device_type: str = "cuda",
) -> None:
    groups = list(seed_report.get("groups") or ())
    _require(len(groups) == grpo.EXPECTED_VALIDATION_GROUPS,
             "terminal seed report has wrong heldout group count")
    _require(int(seed_report.get("n_tasks", -1)) == grpo.EXPECTED_VALIDATION_TASKS,
             "terminal seed report has wrong heldout task count")
    _require(int(seed_report.get("informative_groups", -1))
             >= grpo.MIN_VALIDATION_INFORMATIVE_GROUPS,
             "terminal seed report lacks the informative-group floor")
    _require(seed_report.get("proposal_scoring")
             == grpo._proposal_scoring_geometry(torch.device(device_type)),
             "terminal seed report used different proposal scoring geometry")
    _require(float(seed_report.get("mean_approx_kl", math.inf)) == 0.0
             and float(seed_report.get("max_abs_logratio", math.inf)) == 0.0,
             "terminal seed report is not exact unit-ratio scoring")
    for index, (row, receipt, item) in enumerate(zip(
        groups, validation.receipts, validation.items, strict=True,
    )):
        expected_task = f"{item.suite}/task={int(item.task_id):02d}"
        _require(
            int(row.get("index", -1)) == index
            and row.get("group_id") == receipt["group_id"]
            and row.get("task") == expected_task
            and list(row.get("n_replans_by_arm") or ())
                == list(receipt["n_replans_by_arm"])
            and list(row.get("terminal_rewards") or ())
                == list(receipt["terminal_rewards"]),
            f"terminal seed group {index} differs from validation receipt",
        )


def _assert_checkpoint_unchanged(identity: Mapping[str, Any]) -> None:
    path = Path(str(identity.get("path") or ""))
    _require(path.is_file(), f"scanned checkpoint disappeared: {path}")
    stat = path.stat()
    _require(int(stat.st_size) == int(identity.get("size", -1))
             and int(stat.st_mtime_ns) == int(identity.get("mtime_ns", -1)),
             f"scanned checkpoint stat changed: {path}")
    _require(recovery.sha256_file(path) == identity.get("sha256"),
             f"scanned checkpoint bytes changed: {path}")


def run_rank_task(
    *,
    rank: int,
    world: int,
    device: torch.device,
    run_dir: Path,
    config: Mapping[str, Any],
    config_hash: str,
    parent_identity: Mapping[str, Any],
    collections: Sequence[grpo.ValidatedRecoveryCollection],
    validation: grpo.ValidatedRecoveryCollection,
    trainer_source: Mapping[str, Any],
    exact_behaviour_identity: Mapping[str, Any],
    initial_behavior_identity: Mapping[str, Any],
    start_scan_source: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate and run exactly one rank-owned diagnostic task."""
    kind, step = scan_task_assignment(rank, world)
    started = time.time()
    row: dict[str, Any] = {
        "rank": int(rank), "kind": kind, "step": int(step),
        "task_mapping": "one_surrogate_and_one_trust_rank_per_fixed_step",
    }
    try:
        path = grpo._trainer_checkpoint_path(run_dir, step)
        raw, checkpoint = grpo._authenticated_snapshot_payload(
            path,
            step=step,
            config_hash=config_hash,
            resolved_config=config,
            parent_identity=parent_identity,
            collections=collections,
            validation=validation,
            trainer_source=trainer_source,
            exact_behaviour_identity=exact_behaviour_identity,
            initial_behavior_identity=initial_behavior_identity,
        )
        _require(checkpoint.get("sha256") == EXPECTED_CHECKPOINT_SHA256[step],
                 f"step-{step} checkpoint SHA-256 differs")
        proposal = grpo._proposal_architecture(config, device=device).eval()
        proposal.load_state_dict(raw["proposal"], strict=True)
        _require(grpo.proposal_module_digest(proposal.state_dict())
                 == checkpoint["proposal"],
                 f"step-{step} loaded proposal digest differs")
        del raw
        gc.collect()
        grpo._require_exact_proposal_scoring_environment(proposal, device)
        chunk_replans = int(
            config["outcome_grpo"]["authentication"]["chunk_replans"]
        )
        if kind == "surrogate":
            report = grpo.evaluate_validation_surrogate(
                proposal, validation, device=device,
                chunk_replans=chunk_replans,
            )
        else:
            _require(kind == "trust", f"unknown scan task: {kind}")
            report = grpo.evaluate_trust_gates(
                proposal, validation, device=device,
                chunk_replans=chunk_replans,
                training_nonfinite=0,
                unexpected_gradients=(),
            )
        _assert_checkpoint_unchanged(checkpoint)
        grpo._assert_trainer_source_identity(trainer_source)
        _require(scan_source_identity() == dict(start_scan_source),
                 "diagnostic source changed during rank task")
        row.update({
            "ok": True,
            "checkpoint": checkpoint,
            "proposal_scoring": grpo._proposal_scoring_geometry(device),
            "strict_determinism": grpo._strict_outcome_determinism_state(),
            "report": report,
        })
    except Exception as exc:  # noqa: BLE001
        row.update({
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        })
    row["elapsed_seconds"] = time.time() - started
    return row


def _check_threshold(value: Any, op: str, threshold: Any) -> bool:
    if op == "<=":
        return float(value) <= float(threshold)
    if op == ">=":
        return float(value) >= float(threshold)
    if op == "==":
        return value == threshold
    return False


def _validate_trust_report(report: Mapping[str, Any]) -> None:
    checks = report.get("checks")
    _require(isinstance(checks, Mapping) and set(checks) == set(_TRUST_CHECKS),
             "retrospective trust check set differs from production")
    for name, (op, threshold) in _TRUST_CHECKS.items():
        row = checks[name]
        _require(isinstance(row, Mapping), f"trust check {name} is not a mapping")
        _require(row.get("op") == op and row.get("threshold") == threshold,
                 f"trust check {name} threshold/operator differs")
        value = row.get("value")
        _require(isinstance(value, (int, float)) and math.isfinite(float(value)),
                 f"trust check {name} is nonfinite")
        _require(bool(row.get("pass")) == _check_threshold(value, op, threshold),
                 f"trust check {name} pass flag is inconsistent")
    _require(bool(report.get("passed"))
             == all(bool(row["pass"]) for row in checks.values()),
             "retrospective trust aggregate pass flag is inconsistent")
    counts = report.get("counts")
    _require(isinstance(counts, Mapping)
             and int(counts.get("ratio_atoms", 0)) > 0
             and int(counts.get("arm0_drift_atoms", 0)) > 0,
             "retrospective trust report has no heldout atoms")


def aggregate_seed_paired_results(
    seed_report: Mapping[str, Any],
    surrogates: Mapping[int, Mapping[str, Any]],
    trusts: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build fixed-step summaries; deliberately perform no ranking/selection."""
    expected = set(EARLY_CHECKPOINT_STEPS)
    _require(set(int(step) for step in surrogates) == expected,
             "surrogate task set differs from predeclared early steps")
    _require(set(int(step) for step in trusts) == expected,
             "trust task set differs from predeclared early steps")
    reference_groups = list(seed_report.get("groups") or ())
    _require(len(reference_groups) == grpo.EXPECTED_VALIDATION_GROUPS,
             "terminal seed reference has wrong group count")
    identities = [
        (row.get("index"), row.get("group_id"), row.get("task"))
        for row in reference_groups
    ]
    _require(len(set(identities)) == len(identities),
             "terminal seed reference has duplicate group identities")
    tasks = [str(row["task"]) for row in reference_groups]
    reference = [float(row["surrogate"]) for row in reference_groups]
    _require(all(math.isfinite(value) for value in reference),
             "terminal seed surrogate contains nan/inf")

    rows: dict[str, Any] = {}
    all_gate_count = 0
    for step in EARLY_CHECKPOINT_STEPS:
        surrogate = dict(surrogates[step])
        trust = dict(trusts[step])
        groups = list(surrogate.get("groups") or ())
        _require(len(groups) == grpo.EXPECTED_VALIDATION_GROUPS,
                 f"step-{step} surrogate has wrong group count")
        got_identities = [
            (row.get("index"), row.get("group_id"), row.get("task"))
            for row in groups
        ]
        _require(got_identities == identities,
                 f"step-{step} heldout group order/identity differs from seed")
        _require(int(surrogate.get("n_tasks", -1))
                 == grpo.EXPECTED_VALIDATION_TASKS,
                 f"step-{step} surrogate has wrong task count")
        _require(int(surrogate.get("informative_groups", -1))
                 >= grpo.MIN_VALIDATION_INFORMATIVE_GROUPS,
                 f"step-{step} lacks the informative-group floor")
        values = [float(row["surrogate"]) for row in groups]
        _require(all(math.isfinite(value) for value in values),
                 f"step-{step} surrogate contains nan/inf")
        bootstrap = grpo.task_stratified_paired_bootstrap(
            values,
            reference,
            tasks,
            samples=grpo.CONVERGENCE_BOOTSTRAP_SAMPLES,
            confidence=grpo.CONVERGENCE_CONFIDENCE,
            seed=grpo.CONVERGENCE_BOOTSTRAP_SEED,
        )
        bootstrap.update({
            "comparison": f"{step}-terminal_report_seed_{grpo.START_STEP}",
            "pass": float(bootstrap["ci_low"]) > 0.0,
            "criterion": "paired surrogate CI lower bound > 0",
        })
        mean_kl = float(surrogate.get("mean_approx_kl", math.nan))
        kl_check = {
            "value": mean_kl,
            "op": "<=",
            "threshold": grpo.MAX_APPROX_KL,
            "pass": math.isfinite(mean_kl) and mean_kl <= grpo.MAX_APPROX_KL,
        }
        _validate_trust_report(trust)
        checks = {
            "seed_paired_heldout_efficacy": {
                "value": float(bootstrap["ci_low"]),
                "op": ">",
                "threshold": 0.0,
                "pass": bool(bootstrap["pass"]),
            },
            "heldout_approx_kl": kl_check,
            **{
                f"trust/{name}": _json_copy(row)
                for name, row in trust["checks"].items()
            },
        }
        diagnostic_gate_passed = all(bool(row["pass"]) for row in checks.values())
        all_gate_count += int(diagnostic_gate_passed)
        rows[str(step)] = {
            "global_step": int(step),
            "updates_from_seed": int(step - grpo.START_STEP),
            "mean_surrogate": float(surrogate["mean_surrogate"]),
            "mean_approx_kl": mean_kl,
            "max_abs_logratio": float(surrogate["max_abs_logratio"]),
            "seed_paired_bootstrap": bootstrap,
            "checks": checks,
            "diagnostic_gate_passed": diagnostic_gate_passed,
            "eligibility": "INELIGIBLE",
        }
    return {
        "seed_global_step": grpo.START_STEP,
        "seed_mean_surrogate": float(seed_report["mean_surrogate"]),
        "predeclared_steps": list(EARLY_CHECKPOINT_STEPS),
        "steps": rows,
        "n_steps_passing_all_diagnostic_gates": all_gate_count,
        "selection": {
            "performed": False,
            "permitted": False,
            "best_checkpoint": None,
            "reason": (
                "post-hoc retrospective results may describe the failed curve "
                "but may not select or promote a checkpoint"
            ),
        },
    }


def assemble_rank_results(
    rank_rows: Sequence[Mapping[str, Any]],
    seed_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed on task coverage and pair each step's two rank reports."""
    _require(len(rank_rows) == EXPECTED_WORLD_SIZE,
             f"scan gathered {len(rank_rows)} ranks, expected {EXPECTED_WORLD_SIZE}")
    by_rank = {int(row.get("rank", -1)): dict(row) for row in rank_rows}
    _require(set(by_rank) == set(range(EXPECTED_WORLD_SIZE)),
             "scan rank identities are missing or duplicated")
    failed = [row for row in by_rank.values() if not bool(row.get("ok"))]
    _require(not failed, f"early-curve rank task failed: {failed[:2]}")

    surrogates: dict[int, dict[str, Any]] = {}
    trusts: dict[int, dict[str, Any]] = {}
    checkpoints: dict[int, dict[str, Any]] = {}
    execution: list[dict[str, Any]] = []
    for rank in range(EXPECTED_WORLD_SIZE):
        row = by_rank[rank]
        expected_kind, expected_step = scan_task_assignment(rank, EXPECTED_WORLD_SIZE)
        _require(row.get("kind") == expected_kind
                 and int(row.get("step", -1)) == expected_step,
                 f"rank {rank} executed the wrong early-curve task")
        _require(row.get("strict_determinism") == grpo.STRICT_OUTCOME_DETERMINISM,
                 f"rank {rank} lacks strict determinism evidence")
        geometry = dict(row.get("proposal_scoring") or {})
        device_type = str(geometry.get("device_type") or "")
        _require(device_type in {"cpu", "cuda"}
                 and geometry == grpo._proposal_scoring_geometry(
                     torch.device(device_type)
                 ), f"rank {rank} changed exact proposal scoring")
        checkpoint = dict(row.get("checkpoint") or {})
        _require(checkpoint.get("sha256")
                 == EXPECTED_CHECKPOINT_SHA256[expected_step],
                 f"rank {rank} checkpoint identity differs")
        if expected_step in checkpoints:
            _require(checkpoints[expected_step] == checkpoint,
                     f"step-{expected_step} surrogate/trust checkpoint differs")
        else:
            checkpoints[expected_step] = checkpoint
        report = row.get("report")
        _require(isinstance(report, Mapping), f"rank {rank} emitted no report")
        target = surrogates if expected_kind == "surrogate" else trusts
        _require(expected_step not in target,
                 f"duplicate {expected_kind} task for step {expected_step}")
        target[expected_step] = dict(report)
        execution.append({
            "rank": rank,
            "kind": expected_kind,
            "step": expected_step,
            "elapsed_seconds": float(row.get("elapsed_seconds", 0.0)),
            "checkpoint": checkpoint,
            "proposal_scoring": geometry,
            "strict_determinism": _json_copy(row["strict_determinism"]),
        })
    aggregate = aggregate_seed_paired_results(seed_report, surrogates, trusts)
    evaluations = {
        str(step): {
            "checkpoint": checkpoints[step],
            "surrogate_and_kl": surrogates[step],
            "trust_gate": trusts[step],
        }
        for step in EARLY_CHECKPOINT_STEPS
    }
    return {
        "summary": aggregate,
        "evaluations": evaluations,
        "execution": {
            "world_size": EXPECTED_WORLD_SIZE,
            "mapping": "one_surrogate_and_one_trust_rank_per_fixed_step",
            "tasks": execution,
        },
    }


def _validate_output_path(path: str | os.PathLike[str]) -> Path:
    output = Path(path).expanduser().resolve()
    failed_run = _repo_path(FAILED_RUN_REL)
    output_dir = _repo_path(OUTPUT_DIR_REL)
    try:
        output.relative_to(failed_run)
    except ValueError:
        pass
    else:
        raise EarlyCurveScanError(
            f"diagnostic output may not modify the failed run directory: {output}"
        )
    _require(
        output.parent == output_dir,
        f"diagnostic output must be directly inside {output_dir}: {output}",
    )
    _require(output.suffix == ".json", "diagnostic output must be one JSON file")
    _require(not output.exists(), f"refusing to overwrite diagnostic output: {output}")
    return output


def exclusive_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish one new JSON path; never replace an existing path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise EarlyCurveScanError(
                f"refusing to overwrite diagnostic output: {path}"
            ) from exc
        fsync_dir(path.parent)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def run_scan(output: str | os.PathLike[str]) -> dict[str, Any]:
    rank, world, _local_rank, device = grpo._dist_info()
    _require(world == EXPECTED_WORLD_SIZE,
             f"early-curve scan requires eight ranks, got {world}")
    output_path: Path | None = None
    static: dict[str, Any] | None = None
    local_error = ""
    try:
        output_path = _validate_output_path(output)
        static = _load_static_inputs()
        set_global_seed(grpo.CONVERGENCE_BOOTSTRAP_SEED, rank)
        strict = grpo._configure_strict_outcome_determinism()
        scoring = grpo._configure_exact_proposal_scoring(device)
        _require(strict == grpo.STRICT_OUTCOME_DETERMINISM,
                 "strict deterministic state differs")
        _require(scoring == grpo._proposal_scoring_geometry(device),
                 "exact proposal scoring geometry differs")
    except Exception as exc:  # noqa: BLE001
        local_error = f"{type(exc).__name__}: {exc}"
    _synchronise_errors(
        rank=rank, world=world, label="static input/scoring authentication",
        error=local_error,
    )
    assert static is not None and output_path is not None

    seed_box: dict[str, Any] | None = None
    if rank == 0:
        try:
            seed_box = {
                "ok": True,
                "value": _authenticate_seed_reference(static["terminal"]),
            }
        except Exception as exc:  # noqa: BLE001
            seed_box = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    seed_box = _broadcast_root(seed_box, rank=rank, world=world)
    _require(isinstance(seed_box, Mapping) and seed_box.get("ok") is True,
             f"seed authentication failed: {dict(seed_box or {}).get('error')}")
    seed_identity = dict(seed_box["value"])

    all_collections, owned_collections, collection_provenance = (
        _authenticate_collections(
            static["config"], static["terminal"],
            seed_identity["parent_identity"], rank=rank, world=world,
        )
    )
    collections = all_collections[:grpo.N_FOLDS]
    validation = all_collections[grpo.N_FOLDS]
    seed_report = static["terminal"]["convergence_gate"]["snapshots"][
        str(grpo.START_STEP)
    ]
    _validate_seed_groups(seed_report, validation)
    grpo._validate_initial_behavior_identity(
        static["initial_behavior_identity"],
        global_step=grpo.STOP_STEP,
        world=world,
        config_hash=static["config_hash"],
        trainer_source=static["trainer_source"],
        parent_identity=seed_identity["parent_identity"],
        exact_behaviour_identity=static["exact_behaviour_identity"],
    )

    runtime = _runtime(device, rank=rank, world=world)
    local_task = run_rank_task(
        rank=rank,
        world=world,
        device=device,
        run_dir=_repo_path(FAILED_RUN_REL),
        config=static["config"],
        config_hash=static["config_hash"],
        parent_identity=seed_identity["parent_identity"],
        collections=collections,
        validation=validation,
        trainer_source=static["trainer_source"],
        exact_behaviour_identity=static["exact_behaviour_identity"],
        initial_behavior_identity=static["initial_behavior_identity"],
        start_scan_source=static["scan_source"],
    )

    post_error = ""
    try:
        for collection in owned_collections.values():
            collection.assert_all_sidecars_unchanged()
        grpo._assert_trainer_source_identity(static["trainer_source"])
        _require(scan_source_identity() == static["scan_source"],
                 "diagnostic source changed during execution")
        for label, identity in static["file_identities"].items():
            authenticated_file(
                identity["path"], expected_sha256=identity["sha256"], label=label,
            )
        _require(not list(_repo_path(FAILED_RUN_REL).glob("candidate_*.pt")),
                 "diagnostic run observed a candidate in the failed run")
    except Exception as exc:  # noqa: BLE001
        post_error = f"{type(exc).__name__}: {exc}"
    _synchronise_errors(
        rank=rank, world=world, label="post-scan input immutability",
        error=post_error,
    )

    rank_rows = _all_gather_object(local_task, world)
    runtime_rows = _all_gather_object(runtime, world)
    completion: dict[str, Any] | None = None
    if rank == 0:
        try:
            assembled = assemble_rank_results(rank_rows, seed_report)
            report = {
                "format_version": FORMAT_VERSION,
                "kind": KIND,
                "status": "DIAGNOSTIC_COMPLETE",
                "created_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
                "diagnostic_only": True,
                "read_only_inputs": True,
                "optimizer_steps": 0,
                "simulator_episodes": 0,
                "candidate_emitted": False,
                "promotion_gate_evaluated": False,
                "eligibility": {
                    "promotion": "INELIGIBLE",
                    "checkpoint_selection": "INELIGIBLE",
                    "official_evaluation": "INELIGIBLE",
                    "reason": (
                        "retrospective post-hoc analysis of an already-failed "
                        "strict run; no result may choose or promote a checkpoint"
                    ),
                },
                "predeclared_steps": list(EARLY_CHECKPOINT_STEPS),
                "authentication": {
                    "config_hash": static["config_hash"],
                    "files": static["file_identities"],
                    "trainer_source": static["trainer_source"],
                    "diagnostic_source": static["scan_source"],
                    "parent": seed_identity["parent_identity"],
                    "seed_checkpoint": seed_identity["seed_checkpoint"],
                    "exact_behaviour_identity": static[
                        "exact_behaviour_identity"
                    ],
                    "initial_behavior_identity": static[
                        "initial_behavior_identity"
                    ],
                    "collections": collection_provenance,
                    "strict_determinism": grpo.STRICT_OUTCOME_DETERMINISM,
                    "proposal_scoring": grpo._proposal_scoring_geometry(device),
                },
                "terminal_report_seed_reference": _json_copy(seed_report),
                **assembled,
                "runtime": runtime_rows,
                "no_mutation": {
                    "passed": True,
                    "input_run": str(_repo_path(FAILED_RUN_REL)),
                    "output_outside_input_run": True,
                    "optimizer_steps": 0,
                    "model_artifacts_written": 0,
                    "candidate_emitted": False,
                },
            }
            exclusive_json_write(output_path, report)
            completion = {
                "ok": True,
                "output": str(output_path),
                "summary": report["summary"],
            }
        except Exception as exc:  # noqa: BLE001
            completion = {
                "ok": False, "error": f"{type(exc).__name__}: {exc}",
            }
    completion = _broadcast_root(completion, rank=rank, world=world)
    _require(isinstance(completion, Mapping) and completion.get("ok") is True,
             f"diagnostic aggregation/write failed: "
             f"{dict(completion or {}).get('error')}")
    grpo._barrier()
    return dict(completion)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", required=True,
        help="Fresh JSON output outside the failed run; existing paths are refused.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_scan(args.out)
    except Exception as exc:  # noqa: BLE001
        print(f"[outcome-early-curve] ERROR {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        return 2
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False),
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
