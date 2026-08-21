#!/usr/bin/env python3
"""Exact direct-only no-update gate for the outcome-GRPO v2 round robin.

This is an isolated, fail-closed diagnostic at the authenticated step-49,666
seed.  It measures the first three production v2 sampler updates (24 rank-local
TRAIN draws), differentiates only the authoritative direct GRPO and pre-frozen
full objectives, then projects their production-clipped SGD and reset-AdamW
clone directions onto a newly selected outcome-blind 48-group development
panel.  The panel covers all 40 LIBERO tasks, gives the eight duplicated tasks
no extra task weight, and is judged with one fixed 10,000-row suite-stratified
task bootstrap.  Independent repeated direct backwards check both objectives.

No live optimizer step, proposal perturbation, checkpoint, candidate, formal
evaluation, or promotion artifact is possible here.  A statistical PASS can
authorize only a separately frozen 64-update ineligible pilot.  Authentication,
numerical, source, or mutation failures are INVALID executions and publish no
scientific report.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import re
import sys
import tempfile
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch import Tensor, nn  # noqa: E402

from loom.eval import outcome_recovery as recovery  # noqa: E402
from loom.train import outcome_grpo as v1  # noqa: E402
from loom.train import outcome_grpo_v2 as v2  # noqa: E402
from loom.train import schedule as optim_schedule  # noqa: E402
from loom.train.atomic import fsync_dir  # noqa: E402
from loom.train.determinism import set_global_seed, set_step_seed  # noqa: E402
from loom.train.loop import read_config  # noqa: E402
from scripts import outcome_component_gradient_audit as component_audit  # noqa: E402


FORMAT_VERSION = 2
KIND = "loom_outcome_grpo_v2_round_robin_direct_direction_audit"
EXPECTED_WORLD_SIZE = 8
AUDIT_OFFSETS = (0, 1, 2)
AUDIT_STEPS = tuple(v2.START_STEP + offset for offset in AUDIT_OFFSETS)
EXPECTED_TRAIN_DRAWS = len(AUDIT_STEPS) * EXPECTED_WORLD_SIZE
EXPECTED_FOLD_DRAWS = 4
PANEL_GROUPS = 48
PANEL_TASKS = 40
PANEL_EXTRA_GROUPS = PANEL_GROUPS - PANEL_TASKS
PANEL_TASKS_PER_RANK = PANEL_TASKS // EXPECTED_WORLD_SIZE
PANEL_REPLANS_PER_ARM = 2
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 49_666
CONFIDENCE = 0.95
BALANCE_WEIGHT = 1e-2
PILOT_SPARSE_ALPHA = 0.0
PILOT_INITIAL_RECOVERY_BETA = 1.0
PILOT_DEMO_REFERENCE_LAMBDA = 1.0
MAX_DIRECT_REPEAT_RELATIVE_RESIDUAL = 1e-7
MIN_ENDPOINT_BENEFIT_COSINE = 0.01
MAX_CATASTROPHIC_WRONG_WAY_BENEFIT_COSINE = -0.01
REFERENCE_GRADIENT_RELATIVE_BOUND = 1e-6
EXPECTED_SEED_CHECKPOINT = "runs/r0a_deploy_s1_eval/ckpt_000049666.pt"
EXPECTED_CONFIG_REL = "configs/r0a_outcome_grpo_v2_pilot.yaml"
EXPECTED_RESOLVED_CONFIG_HASH = "67277938c51075d2"
EXPECTED_CONFIG_FILE_SHA256 = (
    "249a221d84032f8c3801a7430e48597a364fa16b02b378e9a17d30c8a56cdf44"
)
EXPECTED_VALIDATION_IDENTITY_DIGEST = (
    "758b046c52a401c83fccba21523c9ea6add3aef0966ffa623deee581dafeccee"
)
EXPECTED_V2_SOURCE_SHA256 = (
    "fae8a7117e76934f3f58efe226359dff6a666812888c7348243ee09f638f4c50"
)
EXPECTED_PANEL_GROUP_RECEIPT_SHA256 = (
    "924e28cb96d49ff581ad5907fe8069ccba76f3aff404a3a75e434dcd90c0e329"
)
EXPECTED_BOOTSTRAP_MATRIX_SHA256 = (
    "1e570b6d13426c8fbd58016d0fba6869dc18aa3151dfdbc0bab357373cacf32e"
)
OUTPUT_DIR_REL = "runs/diagnostics/outcome_round_robin_direction_audit"
OUTPUT_NAME_PREFIX = "outcome_round_robin_direction_audit_v2_s49666_"
_AUDIT_SOURCE_FILES = (
    "scripts/outcome_component_gradient_audit.py",
    "scripts/outcome_round_robin_direction_audit.py",
    "scripts/outcome_round_robin_direction_audit.sbatch",
)
_TASK_RE = re.compile(r"^(libero_(?:spatial|object|goal|long))/task=(\d{2})/")

INVALID_INSTRUMENTATION_HISTORY = (
    {
        "job_id": 32575962,
        "status": "INVALID_INSTRUMENTATION",
        "stopped_before_panel_statistics": True,
        "scientific_report_published": False,
        "reason": "component-additivity instrumentation rejected t0",
    },
    {
        "job_id": 32576514,
        "status": "INVALID_INSTRUMENTATION",
        "stopped_before_panel_statistics": True,
        "scientific_report_published": False,
        "reason": "component-additivity instrumentation rejected t0",
        "observed_relative_residual": 0.026973897,
        "retired_bound": 0.02,
    },
)

ELIGIBILITY = {
    "diagnostic_only": True,
    "full_training_eligible": False,
    "candidate_eligible": False,
    "official_evaluation_eligible": False,
    "promotion_eligible": False,
    "maximum_authority_on_pass": "64_update_ineligible_pilot_only",
    "live_optimizer_steps": 0,
    "live_parameter_perturbations": 0,
    "checkpoint_emitted": False,
    "candidate_emitted": False,
}


class RoundRobinDirectionAuditError(RuntimeError):
    """The diagnostic is INVALID and must not produce scientific evidence."""


def _require(condition: Any, message: str) -> None:
    if not bool(condition):
        raise RoundRobinDirectionAuditError(message)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_hash(domain: str, *parts: object) -> str:
    encoded = "\x1f".join((domain, *(str(part) for part in parts))).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_output_path(path: str | os.PathLike[str]) -> Path:
    output = Path(path).expanduser().resolve()
    directory = (ROOT / OUTPUT_DIR_REL).resolve()
    _require(output.parent == directory,
             f"diagnostic output must be directly inside {directory}: {output}")
    _require(output.suffix == ".json", "direction audit output must be JSON")
    _require(output.name.startswith(OUTPUT_NAME_PREFIX),
             f"direction audit output must start with {OUTPUT_NAME_PREFIX!r}")
    _require(not output.exists(), f"refusing existing diagnostic output: {output}")
    return output


def exclusive_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish a new diagnostic with an exclusive hard link, never replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False,
    ) + "\n").encode("utf-8")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise RoundRobinDirectionAuditError(
                f"refusing existing diagnostic output: {path}"
            ) from exc
        fsync_dir(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def task_key(group_id: str) -> str:
    """Return the outcome-free suite/task identity encoded in a group ID."""
    match = _TASK_RE.match(str(group_id))
    _require(match is not None, f"unrecognized validation group ID: {group_id!r}")
    return f"{match.group(1)}/task={match.group(2)}"


def task_suite(key: str) -> str:
    suite = str(key).split("/", 1)[0]
    _require(suite in {"libero_spatial", "libero_object", "libero_goal", "libero_long"},
             f"unknown task suite: {key!r}")
    return suite


def select_outcome_blind_panel(
    group_sidecar_sha256: Mapping[str, str],
    *,
    identity_digest: str,
) -> dict[str, Any]:
    """Select 40 primary plus eight duplicate groups using group IDs only.

    This function deliberately cannot receive rewards, payloads, or metrics.
    Sidecar hashes are pinned in the receipt but never enter a selection hash.
    Selection domains are exact UTF-8 strings with pipe delimiters and no
    newline: ``rr-audit-panel-v1|I|group_id`` and
    ``rr-audit-extra-v1|I|task_key``, where ``I`` is the authenticated
    validation identity digest value.
    """
    _require(str(identity_digest) == EXPECTED_VALIDATION_IDENTITY_DIGEST,
             "validation identity digest differs from the pinned development panel")
    ids = tuple(str(value) for value in group_sidecar_sha256)
    _require(len(ids) == len(set(ids)), "validation group IDs are not unique")
    _require(all(
        len(str(group_sidecar_sha256[group_id])) == 64
        and all(char in "0123456789abcdef"
                for char in str(group_sidecar_sha256[group_id]))
        for group_id in ids
    ), "validation sidecar receipt contains an invalid SHA-256")
    grouped: dict[str, list[str]] = defaultdict(list)
    for group_id in ids:
        grouped[task_key(group_id)].append(group_id)
    tasks = tuple(sorted(grouped))
    _require(len(tasks) == PANEL_TASKS,
             f"validation group IDs cover {len(tasks)} tasks, expected {PANEL_TASKS}")
    suites = {suite: [key for key in tasks if task_suite(key) == suite]
              for suite in sorted({task_suite(key) for key in tasks})}
    _require(len(suites) == 4 and all(len(rows) == 10 for rows in suites.values()),
             "validation group IDs do not cover four suites of ten tasks")
    _require(all(len(rows) >= 2 for rows in grouped.values()),
             "every panel task needs at least two available groups")

    ordered_tasks = list(tasks)
    ordered_extra_tasks = sorted(
        tasks,
        key=lambda key: (
            hashlib.sha256(
                f"rr-audit-extra-v1|{identity_digest}|{key}".encode("utf-8")
            ).hexdigest(),
            key,
        ),
    )[:PANEL_EXTRA_GROUPS]
    extra_tasks = set(ordered_extra_tasks)
    _require(len(extra_tasks) == PANEL_EXTRA_GROUPS,
             "panel did not choose exactly eight extra tasks")
    # Assign one doubled task plus four singleton tasks per rank.  This changes
    # no selection outcome, but gives every rank an identical six-forward
    # collective schedule and prevents distributed error-check cross-wiring.
    assigned = {
        key: rank for rank, key in enumerate(ordered_extra_tasks)
    }
    singleton_tasks = [key for key in ordered_tasks if key not in extra_tasks]
    for index, key in enumerate(singleton_tasks):
        assigned[key] = index % EXPECTED_WORLD_SIZE
    _require(all(sum(assigned[key] == rank for key in tasks) == PANEL_TASKS_PER_RANK
                 for rank in range(EXPECTED_WORLD_SIZE)),
             "balanced panel task sharding is not five tasks per rank")

    rows: list[dict[str, Any]] = []
    for key in ordered_tasks:
        candidates = sorted(grouped[key], key=lambda group_id: (
            hashlib.sha256(
                f"rr-audit-panel-v1|{identity_digest}|{group_id}".encode("utf-8")
            ).hexdigest(),
            group_id,
        ))
        primary = candidates[0]
        rows.append({
            "rank": assigned[key], "task_key": key, "suite": task_suite(key),
            "panel_role": "primary", "group_id": primary,
            "sidecar_sha256": str(group_sidecar_sha256[primary]),
        })
        if key in extra_tasks:
            second = candidates[1]
            rows.append({
                "rank": assigned[key], "task_key": key, "suite": task_suite(key),
                "panel_role": "second", "group_id": second,
                "sidecar_sha256": str(group_sidecar_sha256[second]),
            })
    rows.sort(key=lambda row: (
        int(row["rank"]), str(row["task_key"]), str(row["panel_role"]),
    ))
    _require(len(rows) == PANEL_GROUPS, "panel group count changed")
    _require(len({row["group_id"] for row in rows}) == PANEL_GROUPS,
             "panel contains duplicate groups")
    _require(all(sum(int(row["rank"]) == rank for row in rows) == 6
                 for rank in range(EXPECTED_WORLD_SIZE)),
             "balanced panel sharding is not six groups per rank")
    _require(all(sum(row["task_key"] == key for row in rows)
                 == (2 if key in extra_tasks else 1) for key in tasks),
             "panel task multiplicities differ from 32 singles plus eight doubles")
    receipt = {
        "kind": "outcome_blind_group_id_panel_v1",
        "selection_algorithm": (
            "per task sort sha256(utf8('rr-audit-panel-v1|I|group_id')) then "
            "take first; double the eight tasks with smallest "
            "sha256(utf8('rr-audit-extra-v1|I|task_key')) and take second"
        ),
        "identity_digest_I": str(identity_digest),
        "hash_encoding": "UTF-8 bytes, exact pipe delimiters, no newline",
        "selection_inputs": "group_id only; sidecar SHA-256 pinned but not ranked",
        "distributed_assignment": (
            "one selected doubled task plus four selected singleton tasks per rank; "
            "assignment has no effect on group selection"
        ),
        "terminal_rewards_not_used_or_accessed_by_selection_logic": True,
        "manifest_parser_materializes_unprojected_fields_before_selection": True,
        "sidecar_payload_read_before_receipt": False,
        "tasks": PANEL_TASKS,
        "groups": PANEL_GROUPS,
        "extra_groups": PANEL_EXTRA_GROUPS,
        "ordered_rows": rows,
    }
    return {**receipt, "sha256": _canonical_sha256(receipt)}


def hashed_replans(group_id: str, n_replans_by_arm: Sequence[int]) -> dict[int, tuple[int, ...]]:
    """Choose exactly two contexts per sampled arm from structural counts."""
    _require(len(n_replans_by_arm) == recovery.GROUP_SIZE,
             f"{group_id} has wrong replan-count vector")
    selected: dict[int, tuple[int, ...]] = {}
    for arm in range(1, recovery.GROUP_SIZE):
        count = int(n_replans_by_arm[arm])
        _require(count >= PANEL_REPLANS_PER_ARM,
                 f"{group_id} arm {arm} has fewer than two replans")
        order = sorted(
            range(count),
            key=lambda index: (
                _stable_hash("rr-panel-replan-v1", group_id, arm, index), index,
            ),
        )
        selected[arm] = tuple(order[:PANEL_REPLANS_PER_ARM])
    return selected


def attach_panel_sampling_receipt(
    group_receipt: Mapping[str, Any],
    receipts_by_group_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Attach structural replan indices after the group-only receipt is frozen."""
    _require(group_receipt.get("sha256") == _canonical_sha256({
        key: value for key, value in group_receipt.items() if key != "sha256"
    }), "outcome-blind panel receipt hash is invalid")
    rows: list[dict[str, Any]] = []
    for raw in group_receipt["ordered_rows"]:
        row = dict(raw)
        receipt = receipts_by_group_id.get(str(row["group_id"]))
        _require(receipt is not None, f"selected panel group is absent: {row['group_id']}")
        replans = hashed_replans(
            str(row["group_id"]), receipt["n_replans_by_arm"],
        )
        rows.append({
            **row,
            "replan_indices": {str(arm): list(values)
                               for arm, values in sorted(replans.items())},
        })
    sampling = {
        "kind": "outcome_blind_hash_replan_sampling_v1",
        "parent_group_receipt_sha256": group_receipt["sha256"],
        "selection_inputs": "group_id plus n_replans_by_arm only",
        "terminal_rewards_used": False,
        "contexts_per_sampled_arm": PANEL_REPLANS_PER_ARM,
        "rows": rows,
    }
    return {**sampling, "sha256": _canonical_sha256(sampling)}


def make_suite_stratified_resample_matrix(
    task_keys: Sequence[str],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[Tensor, dict[str, Any]]:
    """Build one fixed task-index matrix, sampling within each suite."""
    keys = tuple(str(value) for value in task_keys)
    _require(len(keys) == len(set(keys)) == PANEL_TASKS,
             "bootstrap requires exactly 40 unique task keys")
    _require(int(samples) > 0, "bootstrap sample count must be positive")
    suites = sorted({task_suite(key) for key in keys})
    _require(len(suites) == 4, "bootstrap requires four LIBERO suites")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    blocks: list[Tensor] = []
    suite_columns: dict[str, list[int]] = {}
    for suite in suites:
        indices = [index for index, key in enumerate(keys) if task_suite(key) == suite]
        _require(len(indices) == 10, f"bootstrap suite {suite} has {len(indices)} tasks")
        choices = torch.randint(
            0, len(indices), (int(samples), len(indices)), generator=generator,
            dtype=torch.int64,
        )
        lookup = torch.tensor(indices, dtype=torch.int64)
        blocks.append(lookup.index_select(0, choices.reshape(-1)).reshape_as(choices))
        suite_columns[suite] = indices
    matrix = torch.cat(blocks, dim=1).contiguous()
    _require(tuple(matrix.shape) == (int(samples), PANEL_TASKS),
             f"bootstrap matrix shape changed: {tuple(matrix.shape)}")
    digest = hashlib.sha256(matrix.numpy().tobytes(order="C")).hexdigest()
    return matrix, {
        "kind": "fixed_suite_stratified_task_resample_matrix_v1",
        "samples": int(samples), "tasks_per_row": PANEL_TASKS,
        "seed": int(seed), "suite_columns": suite_columns,
        "dtype": str(matrix.dtype), "sha256": digest,
    }


def bonferroni_task_bounds(
    task_values: Tensor,
    resamples: Tensor,
    *,
    confidence: float = CONFIDENCE,
) -> dict[str, Any]:
    """Percentile task-bootstrap bounds with fixed-family correction."""
    values = task_values.detach().to(dtype=torch.float64, device="cpu")
    _require(values.ndim == 2 and values.shape[1] == PANEL_TASKS,
             f"bound values must be (directions,40), got {tuple(values.shape)}")
    _require(resamples.ndim == 2 and resamples.shape[1] == PANEL_TASKS,
             "bound resample matrix must have 40 task columns")
    _require(bool(torch.isfinite(values).all()), "task values contain nan/inf")
    means = values.mean(dim=1)
    boot = torch.stack([
        row.index_select(0, resamples.reshape(-1)).reshape(resamples.shape).mean(dim=1)
        for row in values
    ], dim=1)
    family_size = int(values.shape[0])
    _require(float(confidence) == 0.95,
             "audit bounds require the predeclared 95% family confidence")
    tail = 0.05 / family_size
    upper = torch.quantile(
        boot, 1.0 - tail, dim=0, interpolation="higher",
    )
    lower = torch.quantile(
        boot, tail, dim=0, interpolation="lower",
    )
    return {
        "method": "suite_stratified_task_percentile_bonferroni_one_sided",
        "confidence": float(confidence),
        "family_size": family_size,
        "per_tail_probability": tail,
        "upper_quantile": 1.0 - tail,
        "lower_quantile": tail,
        "upper_quantile_interpolation": "higher",
        "lower_quantile_interpolation": "lower",
        "point_means": [float(value) for value in means],
        "upper_confidence_bounds": [float(value) for value in upper],
        "lower_confidence_bounds": [float(value) for value in lower],
        "benefit_task_counts": [int((row < 0.0).sum()) for row in values],
        "harm_task_counts": [int((row > 0.0).sum()) for row in values],
        "flat_task_counts": [int((row == 0.0).sum()) for row in values],
    }


def decide_direction_gate(
    *,
    endpoint_bounds: Mapping[str, Any],
    endpoint_benefit_cosines: Sequence[float],
    adamw_increment_bounds: Mapping[str, Any],
    adamw_increment_benefit_cosines: Sequence[float],
    reference_gradient_bounds_passed: bool = True,
    reference_vectors_bitwise_zero: bool = False,
) -> dict[str, Any]:
    """Apply the non-adaptive, pre-frozen alpha-zero endpoint rule."""
    endpoint_names = (
        "grpo_only_cumulative_clipped_sgd",
        "full_alpha0_beta1_lambda1_cumulative_clipped_sgd",
        "full_alpha0_beta1_lambda1_reset_adamw_with_production_decay",
    )
    _require(len(endpoint_bounds["point_means"]) == len(endpoint_names),
             "primary endpoint family changed")
    _require(len(endpoint_benefit_cosines) == len(endpoint_names),
             "endpoint cosine family changed")
    _require(len(adamw_increment_bounds["point_means"]) == len(AUDIT_STEPS),
             "AdamW increment family changed")
    _require(len(adamw_increment_benefit_cosines) == len(AUDIT_STEPS),
             "AdamW increment cosine family changed")
    endpoint_rows = []
    for index, name in enumerate(endpoint_names):
        point = float(endpoint_bounds["point_means"][index])
        upper = float(endpoint_bounds["upper_confidence_bounds"][index])
        cosine = float(endpoint_benefit_cosines[index])
        row = {
            "name": name,
            "point_mean_loss_delta": point,
            "bonferroni_one_sided_95_ucb": upper,
            "aggregate_benefit_cosine": cosine,
            "point_mean_strictly_beneficial": point < 0.0,
            "ucb_strictly_beneficial": upper < 0.0,
            "benefit_cosine_at_least_0p01": cosine >= MIN_ENDPOINT_BENEFIT_COSINE,
        }
        row["passed"] = all((
            row["point_mean_strictly_beneficial"],
            row["ucb_strictly_beneficial"],
            row["benefit_cosine_at_least_0p01"],
        ))
        endpoint_rows.append(row)
    increment_rows = []
    for offset, step in enumerate(AUDIT_STEPS):
        lower = float(adamw_increment_bounds["lower_confidence_bounds"][offset])
        cosine = float(adamw_increment_benefit_cosines[offset])
        row = {
            "offset": offset,
            "global_step": step,
            "point_mean_loss_delta": float(
                adamw_increment_bounds["point_means"][offset]
            ),
            "bonferroni_one_sided_95_lcb": lower,
            "aggregate_benefit_cosine": cosine,
            "benefit_cosine_below_minus_0p01": (
                cosine < MAX_CATASTROPHIC_WRONG_WAY_BENEFIT_COSINE
            ),
            "lcb_strictly_harmful": lower > 0.0,
        }
        row["catastrophic"] = (
            row["benefit_cosine_below_minus_0p01"]
            or row["lcb_strictly_harmful"]
        )
        increment_rows.append(row)
    passed = (
        all(row["passed"] for row in endpoint_rows)
        and not any(row["catastrophic"] for row in increment_rows)
        and bool(reference_gradient_bounds_passed)
    )
    return {
        "status": (
            "PASS_TO_64_UPDATE_ALPHA0_BETA1_LAMBDA1_INELIGIBLE_PILOT"
            if passed else "ABORT_OUTCOME_OBJECTIVE"
        ),
        "passed": passed,
        "pre_frozen_alpha": 0.0,
        "reference_gradient_bounds_passed": bool(reference_gradient_bounds_passed),
        "reference_vectors_bitwise_zero": bool(reference_vectors_bitwise_zero),
        "prospective_64_pilot_coefficients": {
            "sparse_ce_alpha": PILOT_SPARSE_ALPHA,
            "initial_recovery_beta": PILOT_INITIAL_RECOVERY_BETA,
            "demo_reference_lambda": PILOT_DEMO_REFERENCE_LAMBDA,
            "switch_balance": BALANCE_WEIGHT,
        },
        "pilot_prerequisites": [
            "freeze the exact alpha0/beta1/lambda1 recipe before launch",
            "pass the separately required recovery-controller resume smoke",
        ],
        "primary_endpoints": endpoint_rows,
        "production_adamw_increment_catastrophes": increment_rows,
        "selection_rule": (
            "all three predeclared alpha0 endpoints require point<0, Bonferroni "
            "one-sided 95% UCB<0, benefit cosine>=0.01; no production-AdamW "
            "increment may have benefit cosine<-0.01 or Bonferroni LCB>0; all "
            "six recovery/demo reference-gradient relative bounds must pass"
        ),
        "pass_authority": (
            "64_update_ineligible_pilot_alpha0_beta1_lambda1_only_subject_to_"
            "exact_recipe_freeze_and_controller_smoke"
        ),
        "full_training_authorized": False,
    }


def _named_live_parameters(proposal: nn.Module) -> tuple[tuple[str, nn.Parameter], ...]:
    rows = tuple(
        (name, parameter) for name, parameter in proposal.named_parameters()
        if parameter.requires_grad
    )
    _require(bool(rows), "proposal has no trainable parameters")
    _require(all(parameter.grad is None for _name, parameter in rows),
             "live proposal has populated gradient buffers")
    return rows


def _flat_parameters(named: Sequence[tuple[str, nn.Parameter]]) -> Tensor:
    return torch.cat([parameter.detach().float().reshape(-1)
                      for _name, parameter in named])


def _clone_parameter_groups(
    named: Sequence[tuple[str, nn.Parameter]],
    *,
    weight_decay: float,
    lr_scale: float,
) -> tuple[list[nn.Parameter], list[dict[str, Any]]]:
    clones = [nn.Parameter(parameter.detach().clone(), requires_grad=True)
              for _name, parameter in named]
    buckets: dict[bool, list[nn.Parameter]] = defaultdict(list)
    for (name, parameter), clone in zip(named, clones, strict=True):
        full_name = f"proposal.{name}"
        decay = (
            float(weight_decay) > 0.0
            and parameter.ndim > 1
            and not any(fragment in full_name.lower()
                        for fragment in optim_schedule._NO_DECAY)
        )
        buckets[decay].append(clone)
    groups = [{
        "name": f"proposal/{'decay' if decay else 'nodecay'}",
        "module": "proposal",
        "params": buckets[decay],
        "weight_decay": float(weight_decay) if decay else 0.0,
        "lr_scale": float(lr_scale),
    } for decay in sorted(buckets, key=lambda value: not value)]
    _require(sum(len(group["params"]) for group in groups) == len(clones),
             "clone parameter grouping lost parameters")
    return clones, groups


def _assign_flat_gradient(
    clones: Sequence[nn.Parameter],
    gradient: Tensor,
) -> None:
    _require(int(gradient.numel()) == sum(parameter.numel() for parameter in clones),
             "flat gradient size differs from proposal clone")
    offset = 0
    for parameter in clones:
        count = parameter.numel()
        value = gradient[offset:offset + count].reshape_as(parameter).to(
            device=parameter.device, dtype=parameter.dtype,
        )
        parameter.grad = value.detach().clone()
        offset += count


def _flat_clone_parameters(clones: Sequence[nn.Parameter]) -> Tensor:
    return torch.cat([parameter.detach().float().reshape(-1) for parameter in clones])


def _flat_clone_gradients(clones: Sequence[nn.Parameter]) -> Tensor:
    _require(all(parameter.grad is not None for parameter in clones),
             "clone gradient is absent after clipping")
    return torch.cat([
        parameter.grad.detach().float().reshape(-1) for parameter in clones
    ])


def proposal_lr_schedule(cfg: Mapping[str, Any]) -> optim_schedule.CosineWithWarmup:
    optim = cfg["optim"]
    return optim_schedule.CosineWithWarmup(
        base_lr=float(optim["lr"]),
        warmup_steps=int(optim["warmup"]),
        total_steps=v1.SCHEDULE_STEPS,
        min_lr_ratio=float(optim["min_lr_ratio"]),
    )


def proposal_lr_at(cfg: Mapping[str, Any], step: int) -> float:
    schedule = proposal_lr_schedule(cfg)
    # Match CosineWithWarmup.apply's multiplication order bit for bit.
    return (
        schedule.base_lr
        * float(cfg["optim"]["lr_scales"]["proposal"])
        * schedule.scale_at(int(step))
    )


def production_clip_flat_like_proposal(
    proposal: nn.Module,
    gradient: Tensor,
    *,
    max_norm: float,
) -> tuple[Tensor, float]:
    """Use torch's production clipping kernel on proposal-shaped clones."""
    named = _named_live_parameters(proposal)
    clones, _groups = _clone_parameter_groups(
        named, weight_decay=0.0, lr_scale=1.0,
    )
    _assign_flat_gradient(clones, gradient)
    preclip = float(torch.nn.utils.clip_grad_norm_(clones, float(max_norm)))
    clipped = _flat_clone_gradients(clones)
    _require(bool(torch.isfinite(clipped).all()) and math.isfinite(preclip),
             "production clone clipping produced nan/inf")
    del clones
    return clipped, preclip


def virtual_adamw_clone_replay(
    proposal: nn.Module,
    gradients: Sequence[Tensor],
    *,
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay three gradients with production AdamW on disposable clones.

    The synchronized gradients were all measured at the unchanged seed; this is
    not a recomputed virtual trajectory.  Real ``torch.optim.AdamW``, real
    proposal parameter groups, real ``clip_grad_norm_``, and the production
    absolute cosine schedule are authoritative for the returned clone deltas.
    """
    _require(len(gradients) == len(AUDIT_STEPS),
             "clone replay requires exactly the first three v2 gradients")
    named = _named_live_parameters(proposal)
    live_digest_before = v1.proposal_module_digest(proposal.state_dict())
    seed = _flat_parameters(named)
    optim = cfg["optim"]
    weight_decay = float(optim["weight_decay"])
    clones, groups = _clone_parameter_groups(
        named,
        weight_decay=weight_decay,
        lr_scale=float(optim["lr_scales"]["proposal"]),
    )
    optimizer = torch.optim.AdamW(
        groups,
        lr=float(optim["lr"]),
        betas=tuple(float(value) for value in optim["betas"]),
        eps=float(optim["eps"]),
    )
    _require(len(optimizer.state) == 0, "virtual clone AdamW did not reset state")
    schedule = proposal_lr_schedule(cfg)
    cumulative: list[Tensor] = []
    increments: list[Tensor] = []
    clipped_gradients: list[Tensor] = []
    step_rows: list[dict[str, Any]] = []
    previous = seed.clone()
    for offset, (step, gradient) in enumerate(zip(
        AUDIT_STEPS, gradients, strict=True,
    )):
        optimizer.zero_grad(set_to_none=True)
        _assign_flat_gradient(clones, gradient)
        preclip = float(torch.nn.utils.clip_grad_norm_(
            clones, float(optim["grad_clip"]),
        ))
        clipped = _flat_clone_gradients(clones).detach().clone()
        lr_by_group = schedule.apply(optimizer, int(step))
        expected_lr = proposal_lr_at(cfg, step)
        _require(all(float(value) == expected_lr for value in lr_by_group.values()),
                 f"virtual clone proposal LR differs at step {step}")
        optimizer.step()
        current = _flat_clone_parameters(clones)
        increment = current - previous
        total = current - seed
        _require(bool(torch.isfinite(increment).all())
                 and bool(torch.isfinite(total).all()),
                 f"virtual AdamW clone delta is nonfinite at offset {offset}")
        increments.append(increment.detach().clone())
        cumulative.append(total.detach().clone())
        clipped_gradients.append(clipped)
        step_rows.append({
            "offset": offset,
            "global_step": int(step),
            "proposal_lr": expected_lr,
            "preclip_norm_from_torch_clip_grad_norm": preclip,
            "clipped_gradient_norm": component_audit._vector_norm(clipped),
            "increment_norm": component_audit._vector_norm(increment),
            "cumulative_delta_norm": component_audit._vector_norm(total),
            "optimizer_state_entries": len(optimizer.state),
        })
        previous = current
    live_digest_after = v1.proposal_module_digest(proposal.state_dict())
    _require(live_digest_after == live_digest_before,
             "live proposal changed during virtual clone replay")
    _require(all(parameter.grad is None for _name, parameter in named),
             "virtual clone replay populated live proposal gradients")
    result = {
        "authority": "real_torch_optim_AdamW_on_disposable_proposal_shaped_clones",
        "gradients_recomputed_at_virtual_parameters": False,
        "frozen_seed_gradients_replayed": True,
        "production_weight_decay_enabled": True,
        "weight_decay": weight_decay,
        "betas": [float(value) for value in optim["betas"]],
        "eps": float(optim["eps"]),
        "grad_clip": float(optim["grad_clip"]),
        "virtual_clone_optimizer_steps": len(AUDIT_STEPS),
        "live_optimizer_steps": 0,
        "step_rows": step_rows,
        "live_proposal_digest_before": live_digest_before,
        "live_proposal_digest_after": live_digest_after,
        "vectors": {
            "increments": increments,
            "cumulative": cumulative,
            "clipped_gradients": clipped_gradients,
        },
    }
    del optimizer, clones
    return result


def cumulative_clipped_sgd_direction(
    proposal: nn.Module,
    gradients: Sequence[Tensor],
    *,
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """Return ``-sum_t lr_t*clip(g_t,1)`` using production clone clipping."""
    _require(len(gradients) == len(AUDIT_STEPS),
             "clipped SGD control requires exactly three gradients")
    clipped: list[Tensor] = []
    preclip: list[float] = []
    for gradient in gradients:
        value, norm = production_clip_flat_like_proposal(
            proposal, gradient, max_norm=float(cfg["optim"]["grad_clip"]),
        )
        clipped.append(value)
        preclip.append(norm)
    lrs = [proposal_lr_at(cfg, step) for step in AUDIT_STEPS]
    delta = -sum(lr * gradient for lr, gradient in zip(lrs, clipped, strict=True))
    return {
        "delta": delta,
        "clipped_gradients": clipped,
        "preclip_norms_from_torch_clip_grad_norm": preclip,
        "proposal_lrs": lrs,
        "delta_norm": component_audit._vector_norm(delta),
        "definition": "-sum_t proposal_lr(t)*torch_clip_grad_norm_1(g_t)",
    }


def aggregate_benefit_cosine(heldout_gradient: Tensor, update_delta: Tensor) -> float:
    """Positive means the update opposes, and should lower, heldout loss."""
    cosine = component_audit._cosine(heldout_gradient, update_delta)
    _require(cosine is not None, "benefit cosine is undefined for a zero vector")
    return -float(cosine)


def _sum_synchronised_gradient(vector: Tensor, *, world: int) -> Tensor:
    result = vector.contiguous()
    if world > 1:
        torch.distributed.all_reduce(result, op=torch.distributed.ReduceOp.SUM)
    _require(bool(torch.isfinite(result).all()),
             "SUM-synchronized gradient contains nan/inf")
    return result


def _synchronised_loss_gradient(
    loss: Tensor,
    named: Sequence[tuple[str, nn.Parameter]],
    *,
    world: int,
    retain_graph: bool,
    label: str,
    require_complete: bool = False,
) -> tuple[Tensor, list[str]]:
    """Coordinate local autograd failure before entering the SUM/world reduce."""
    local_error = ""
    try:
        local, missing = component_audit._local_gradient_vector(
            loss, named, retain_graph=retain_graph,
        )
        if require_complete and missing:
            raise RoundRobinDirectionAuditError(
                f"{label} gradient missing parameters: {missing[:8]}"
            )
    except Exception as exc:  # noqa: BLE001
        local_error = f"{type(exc).__name__}: {exc}"
    v1._raise_if_any_rank_failed(local_error, world, f"{label} local gradient")
    return component_audit._synchronise_gradient(local, world=world), missing


def direct_repeat_consistency(
    first: Tensor,
    repeated: Tensor,
    *,
    label: str,
) -> dict[str, Any]:
    """Fail unless two independently synchronized direct backwards agree."""
    _require(first.shape == repeated.shape,
             f"{label} repeated direct gradient shape changed")
    first_norm = component_audit._vector_norm(first)
    repeated_norm = component_audit._vector_norm(repeated)
    residual_norm = component_audit._vector_norm(repeated - first)
    scale = max(first_norm, repeated_norm, 1.0)
    relative_residual = residual_norm / scale
    _require(
        relative_residual <= MAX_DIRECT_REPEAT_RELATIVE_RESIDUAL,
        f"{label} repeated direct backward is inconsistent: "
        f"relative_residual={relative_residual:.9g} > "
        f"{MAX_DIRECT_REPEAT_RELATIVE_RESIDUAL:.9g}; "
        f"residual_norm={residual_norm:.9g}; first_norm={first_norm:.9g}; "
        f"repeated_norm={repeated_norm:.9g}",
    )
    return {
        "first_direct_is_authoritative": True,
        "independent_synchronised_backward_calls": 2,
        "first_norm": first_norm,
        "repeated_norm": repeated_norm,
        "residual_norm": residual_norm,
        "relative_residual": relative_residual,
        "max_relative_residual": MAX_DIRECT_REPEAT_RELATIVE_RESIDUAL,
        "passed": True,
    }


def _reference_gradient_evidence(
    loss: Tensor,
    proposal: nn.Module,
    *,
    grpo_norm: float,
    world: int,
    retain_graph: bool,
    label: str,
) -> dict[str, Any]:
    named = _named_live_parameters(proposal)
    gradient, missing = _synchronised_loss_gradient(
        loss,
        named,
        world=world,
        retain_graph=retain_graph,
        label=label,
    )
    norm = component_audit._vector_norm(gradient)
    bitwise_all_zero = int(torch.count_nonzero(gradient)) == 0
    bound = REFERENCE_GRADIENT_RELATIVE_BOUND * max(float(grpo_norm), 1.0)
    local_error = ""
    try:
        _require(float(loss.detach()) == 0.0,
                 f"{label} reference value is not exactly zero")
    except Exception as exc:  # noqa: BLE001
        local_error = f"{type(exc).__name__}: {exc}"
    v1._raise_if_any_rank_failed(local_error, world, f"{label} reference bound")
    return {
        "value": float(loss.detach()),
        "value_exactly_zero": True,
        "synchronised_gradient_norm": norm,
        "synchronised_gradient_bitwise_all_zero": bitwise_all_zero,
        "relative_bound": REFERENCE_GRADIENT_RELATIVE_BOUND,
        "absolute_bound": bound,
        "bound_passed": norm <= bound,
        "missing_parameter_gradients": missing,
    }


def _demo_reference_only(
    anchor: v2.DemoReferenceAnchorV2,
    global_step: int,
    *,
    cache_prepared_for_reuse: bool = False,
) -> tuple[Tensor, dict[str, Any]]:
    """Evaluate only the frozen-seed demo KL; never construct sparse CE."""
    anchor.assert_seed_unchanged()
    prepared = anchor.anchor._prepare(global_step)
    beliefs, lang, targets, _embodiment = prepared
    horizons = len(targets)
    _require(horizons > 0 and len(beliefs) >= horizons,
             "demo-reference anchor has no complete belief horizons")
    reference_terms: list[Tensor] = []
    forward_autocast = anchor.anchor.device.type == "cuda"
    for horizon in range(horizons):
        belief = beliefs[horizon].detach()
        batched_lang = v1._batched_lang(
            lang,
            int(belief.shape[0]),
            belief.device,
            belief.dtype,
        )
        with torch.autocast(
            device_type=belief.device.type,
            dtype=torch.bfloat16,
            enabled=forward_autocast,
        ):
            current_logits = anchor.anchor.proposal.logits(belief, batched_lang)
            with torch.no_grad():
                seed_logits = anchor.seed_proposal.logits(belief, batched_lang)
        _require(current_logits.shape == seed_logits.shape,
                 f"demo-reference h{horizon + 1} live/seed shape mismatch")
        reference_terms.append(v2.dense_categorical_forward_kl(
            current_logits, seed_logits,
        ))
    demo_reference = torch.stack(reference_terms).mean()
    _require(bool(torch.isfinite(demo_reference)),
             f"nonfinite demo reference at step {global_step}")
    anchor.assert_seed_unchanged()
    if cache_prepared_for_reuse:
        _require(int(global_step) not in anchor.anchor._cache,
                 "demo-reference prepared-step cache unexpectedly occupied")
        anchor.anchor._cache[int(global_step)] = prepared
    return demo_reference, {
        "demo_categorical_forward_kl": float(demo_reference.detach()),
        "demo_reference_horizons": horizons,
        "demo_reference_seed_trainable": False,
        "demo_reference_forward_bf16_autocast": forward_autocast,
        "demo_reference_probability_math_fp32": True,
        "demo_reference_only": True,
        "sparse_ce_computed": False,
        "sparse_ce_graph_constructed": False,
    }


def _source_identity() -> dict[str, Any]:
    trainer = v2.trainer_source_identity()
    _require(trainer.get("sha256") == EXPECTED_V2_SOURCE_SHA256,
             f"v2 trainer source closure drifted: {trainer.get('sha256')}")
    return {
        "v2_trainer": trainer,
        "diagnostic": v1._trainer_source_identity(
            root=ROOT, files=_AUDIT_SOURCE_FILES,
        ),
    }


def _config_file_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    expected = (ROOT / EXPECTED_CONFIG_REL).resolve()
    _require(resolved == expected,
             f"audit config must be the canonical checked-in file {expected}")
    stat = resolved.stat()
    identity = {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": recovery.sha256_file(resolved),
    }
    _require(identity["sha256"] == EXPECTED_CONFIG_FILE_SHA256,
             f"canonical v2 config bytes drifted: {identity['sha256']}")
    return identity


def _validate_config(path: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    _config_file_identity(path)
    cfg = read_config(path)
    scaffold = v2.validate_scaffold_config(cfg)
    _require(str(cfg["outcome_grpo_v2"]["seed_checkpoint"])
             == EXPECTED_SEED_CHECKPOINT,
             "v2 scaffold no longer names the authenticated step-49,666 seed")
    _require(int(cfg["outcome_grpo_v2"]["world_size"]) == EXPECTED_WORLD_SIZE,
             "v2 scaffold world size is not eight")
    _require(float(cfg["losses"]["grpo"]["weight"]) == 1.0,
             "v2 GRPO weight differs from one")
    _require(float(cfg["losses"]["balance"]["weight"])
             == BALANCE_WEIGHT == v2.BALANCE_WEIGHT,
             "v2 Switch balance weight differs from 1e-2")
    _require(float(cfg["optim"]["lr"]) == v1.BASE_LEARNING_RATE,
             "v2 base learning rate changed")
    _require(tuple(float(value) for value in cfg["optim"]["betas"])
             == v1.ADAMW_BETAS,
             "v2 AdamW betas changed")
    _require(float(cfg["optim"]["eps"]) == v1.ADAMW_EPS,
             "v2 AdamW epsilon changed")
    _require(float(cfg["optim"]["weight_decay"]) == v1.ADAMW_WEIGHT_DECAY,
             "v2 AdamW decay changed")
    _require(float(cfg["optim"]["grad_clip"]) == v1.GRAD_CLIP,
             "v2 proposal clip changed")
    _require(float(cfg["optim"]["lr_scales"]["proposal"])
             == v2.PROPOSAL_LR_SCALE,
             "v2 proposal LR scale changed")
    resolved_hash = v1._config_hash(cfg)
    _require(resolved_hash == EXPECTED_RESOLVED_CONFIG_HASH,
             f"resolved v2 config hash drifted: {resolved_hash}")
    return cfg, resolved_hash, scaffold


def _runtime_local(
    *, rank: int, world: int, local_rank: int, device: torch.device,
) -> dict[str, Any]:
    base = component_audit._runtime_local(
        rank=rank, world=world, local_rank=local_rank, device=device,
    )
    return {**base, "audit_kind": KIND}


def _authenticate_parent_once(
    checkpoint: Path, *, rank: int, world: int,
) -> dict[str, Any]:
    box: list[Any] = [None]
    if rank == 0:
        try:
            box[0] = {
                "ok": True,
                "identity": recovery.authenticate_seed_checkpoint(checkpoint),
            }
        except Exception as exc:  # noqa: BLE001
            box[0] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if world > 1:
        torch.distributed.broadcast_object_list(box, src=0)
    _require(bool(box[0].get("ok")),
             f"seed checkpoint authentication failed: {box[0].get('error')}")
    return dict(box[0]["identity"])


def pre_reward_panel_receipt(cfg: Mapping[str, Any]) -> tuple[dict[str, Any], Path]:
    """Freeze a receipt from an explicit ID/SHA projection, never outcomes."""
    current = cfg["outcome_grpo_v2"]["validation_lineage"][
        "current_development_collection"
    ]
    root = (ROOT / str(current["path"])).resolve()
    manifest_path = root / "manifest.json"
    _require(manifest_path.is_file(), f"development manifest is missing: {manifest_path}")
    manifest_sha256 = recovery.sha256_file(manifest_path)
    _require(manifest_sha256 == str(current.get("manifest_sha256") or ""),
             "development validation manifest SHA differs from v2 lineage pin")
    # json.loads necessarily materializes complete receipt objects.  Selection
    # control flow immediately projects only group_id/sha256, never indexes or
    # passes terminal_rewards, and the selector's type surface accepts only
    # that projected mapping.
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = str(manifest.get("identity_digest") or "")
    _require(identity == EXPECTED_VALIDATION_IDENTITY_DIGEST,
             "development validation identity differs from pinned audit identity")
    configured_identity = str(current.get("observed_identity_digest") or "")
    _require(configured_identity == identity,
             "v2 development validation identity pin differs from manifest")
    projected = {
        str(row["group_id"]): str(row["sha256"])
        for row in manifest["groups"]
    }
    receipt = select_outcome_blind_panel(
        projected, identity_digest=identity,
    )
    _require(receipt["sha256"] == EXPECTED_PANEL_GROUP_RECEIPT_SHA256,
             f"outcome-blind panel receipt drifted: {receipt['sha256']}")
    return receipt, root


def _exact_scoring_evidence(proposal: nn.Module, device: torch.device) -> dict[str, Any]:
    v1._require_exact_proposal_scoring_environment(proposal, device)
    evidence = {
        "proposal_scoring_batch_size": int(v1.PROPOSAL_SCORING_BATCH_SIZE),
        "proposal_scoring_dtype": str(v1.PROPOSAL_SCORING_DTYPE),
        "proposal_scoring_autocast": bool(v1.PROPOSAL_SCORING_AUTOCAST),
        "cuda_matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "proposal_training": bool(proposal.training),
    }
    _require(evidence == {
        "proposal_scoring_batch_size": 1,
        "proposal_scoring_dtype": "float32",
        "proposal_scoring_autocast": False,
        "cuda_matmul_tf32": False,
        "cudnn_tf32": False,
        "float32_matmul_precision": "highest",
        "proposal_training": False,
    }, f"exact B1 scoring environment changed: {evidence}")
    return evidence


def _build_round_robin_sampler(
    collections: Sequence[v1.ValidatedRecoveryCollection],
    *,
    seed: int,
    rank: int,
    contexts_per_arm: int,
) -> v2.RoundRobinOutcomeSamplerV3:
    _require(len(collections) == v1.N_FOLDS,
             "round-robin audit requires all six TRAIN folds")
    informative = [collection.informative_indices() for collection in collections]
    return v2.RoundRobinOutcomeSamplerV3(
        informative,
        seed=seed,
        rank=rank,
        world_size=EXPECTED_WORLD_SIZE,
        start_step=v2.START_STEP,
        total_updates=v2.PILOT_UPDATES,
        contexts_per_arm=contexts_per_arm,
        identity_digests=[collection.identity_digest for collection in collections],
    )


def _receipt_row(
    collection: v1.ValidatedRecoveryCollection,
    index: int,
    *,
    rank: int,
    replan_indices: Mapping[int, Sequence[int]],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    receipt = collection.receipts[int(index)]
    return {
        "rank": int(rank),
        "split": collection.split,
        "group_index": int(index),
        "group_id": str(receipt["group_id"]),
        "sidecar": str(receipt["sidecar"]),
        "sidecar_sha256": str(receipt["sha256"]),
        "sidecar_size": int(receipt["size"]),
        "replan_indices": {
            str(arm): list(replan_indices[arm]) for arm in sorted(replan_indices)
        },
        **dict(extra or {}),
    }


def rehash_rank_sidecars(
    collections: Mapping[str, v1.ValidatedRecoveryCollection],
    rows: Sequence[Mapping[str, Any]],
    *,
    rank: int,
) -> dict[str, Any]:
    """Post-use SHA/stat recheck of each sidecar consumed by this rank."""
    local = [row for row in rows if int(row["rank"]) == int(rank)]
    closure = hashlib.sha256()
    total_bytes = 0
    seen: set[tuple[str, int]] = set()
    for row in sorted(local, key=lambda value: (
        str(value["split"]), int(value["group_index"]),
    )):
        split = str(row["split"])
        collection = collections[split]
        index = int(row["group_index"])
        key = (split, index)
        _require(key not in seen, f"rank {rank} consumed duplicate sidecar {key}")
        seen.add(key)
        receipt = collection.receipts[index]
        _require(
            str(receipt["group_id"]) == str(row["group_id"])
            and str(receipt["sidecar"]) == str(row["sidecar"])
            and str(receipt["sha256"]) == str(row["sidecar_sha256"])
            and int(receipt["size"]) == int(row["sidecar_size"]),
            f"selected sidecar receipt changed after use: {key}",
        )
        path = collection._resolved_sidecar(collection.root, receipt)
        before = path.stat()
        _require(int(before.st_size) == int(receipt["size"]),
                 f"selected sidecar size changed after use: {path}")
        digest = recovery.sha256_file(path)
        after = path.stat()
        _require(int(after.st_size) == int(before.st_size)
                 and int(after.st_mtime_ns) == int(before.st_mtime_ns),
                 f"selected sidecar stat changed while rehashing: {path}")
        _require(digest == str(receipt["sha256"]),
                 f"selected sidecar SHA changed after use: {path}")
        closure.update(json.dumps({
            "split": split, "group_index": index,
            "group_id": str(receipt["group_id"]),
            "sidecar_sha256": digest, "size": int(receipt["size"]),
        }, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\0")
        total_bytes += int(receipt["size"])
    return {
        "rank": int(rank),
        "selected_sidecars": len(local),
        "selected_bytes": total_bytes,
        "post_use_closure_sha256": closure.hexdigest(),
        "post_use_size_sha256_and_stable_stat": True,
    }


def _warnings_checked(
    caught: Sequence[warnings.WarningMessage], *, world: int, label: str,
) -> list[str]:
    return component_audit._checked_warning_messages(
        caught, world=world, label=label,
    )


def _mean_scalar(value: float, *, world: int, device: torch.device) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    if world > 1:
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        tensor.div_(float(world))
    return float(tensor)


def equal_group_within_task_contribution(
    group_gradients: Sequence[Tensor],
    *,
    total_tasks: int = PANEL_TASKS,
) -> tuple[Tensor, Tensor]:
    """Return one task mean and its equal-task aggregate contribution."""
    _require(bool(group_gradients), "a panel task has no group gradients")
    _require(int(total_tasks) > 0, "equal-task denominator must be positive")
    shape = group_gradients[0].shape
    _require(all(gradient.shape == shape for gradient in group_gradients),
             "group gradients within one task have different shapes")
    task_gradient = sum(group_gradients) / float(len(group_gradients))
    return task_gradient, task_gradient / float(total_tasks)


def _without_vectors(replay: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in replay.items() if key != "vectors"}


def run_audit(
    *,
    config_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
) -> dict[str, Any] | None:
    """Execute the fixed eight-rank read-only diagnostic."""
    rank, world, local_rank, device = v1._dist_info()
    _require(world == EXPECTED_WORLD_SIZE,
             f"round-robin direction audit requires world=8, got {world}")
    local_error = ""
    try:
        output = _validate_output_path(output_path)
    except Exception as exc:  # noqa: BLE001
        local_error = f"{type(exc).__name__}: {exc}"
    v1._raise_if_any_rank_failed(local_error, world, "exclusive diagnostic output")
    started = time.monotonic()

    local_error = ""
    try:
        strict = v1._configure_strict_outcome_determinism()
        scoring_config = v1._configure_exact_proposal_scoring(device)
        runtime_local = _runtime_local(
            rank=rank, world=world, local_rank=local_rank, device=device,
        )
    except Exception as exc:  # noqa: BLE001
        local_error = f"{type(exc).__name__}: {exc}"
    v1._raise_if_any_rank_failed(local_error, world, "A100 deterministic geometry")
    runtimes = component_audit._all_gather_object(runtime_local, world)
    _require(sorted(int(row["rank"]) for row in runtimes) == list(range(world)),
             "distributed runtime rank evidence is incomplete")
    _require(len({(row["hostname"], row["local_rank"]) for row in runtimes}) == world,
             "ranks do not map to eight distinct local A100s")
    _require(len({row["hostname"] for row in runtimes}) == 1,
             "direction audit requires one-node first-production geometry")

    config = Path(config_path).expanduser().resolve()
    local_error = ""
    try:
        cfg, config_hash, scaffold_validation = _validate_config(config)
        config_file_identity = _config_file_identity(config)
        source_identity = _source_identity()
        # This receipt is canonicalized and checked before collection.open()
        # accesses terminal_rewards to validate manifest summaries.
        group_receipt, validation_root = pre_reward_panel_receipt(cfg)
    except Exception as exc:  # noqa: BLE001
        local_error = f"{type(exc).__name__}: {exc}"
    v1._raise_if_any_rank_failed(
        local_error, world, "v2 source/config and pre-reward panel receipt",
    )
    source_rows = component_audit._all_gather_object(source_identity, world)
    _require(all(row == source_rows[0] for row in source_rows),
             "source identity differs across ranks")
    panel_receipt_rows = component_audit._all_gather_object(group_receipt, world)
    _require(all(row == panel_receipt_rows[0] for row in panel_receipt_rows),
             "pre-reward panel receipt differs across ranks")

    seed = int(cfg["run"].get("seed", v1.TRAIN_SEED))
    set_global_seed(seed, rank)
    checkpoint = ROOT / EXPECTED_SEED_CHECKPOINT
    parent_identity = _authenticate_parent_once(checkpoint, rank=rank, world=world)
    local_error = ""
    try:
        v1._assert_seed_stat(parent_identity)
        parent = v1._load_parent_from_identity(parent_identity)
        v1._assert_seed_stat(parent_identity)
        proposal = v1._load_proposal(parent, device=device)
        proposal.eval()
        scoring_evidence = _exact_scoring_evidence(proposal, device)
        proposal_digest_before = v1.proposal_module_digest(proposal.state_dict())
        _require(proposal_digest_before == v1.proposal_model_digest(parent["model"]),
                 "runtime proposal differs from authenticated seed proposal")
        demo_anchor = v2.DemoReferenceAnchorV2.from_parent(
            parent,
            proposal,
            trainer_cfg=cfg,
            device=device,
            rank=rank,
            world_size=world,
        )
        with torch.no_grad():
            preflight_reference, preflight_metrics = _demo_reference_only(
                demo_anchor,
                v2.START_STEP,
                cache_prepared_for_reuse=True,
            )
        _require(float(preflight_reference) == 0.0,
                 "demo-reference-only preflight is not exactly zero")
        anchor_preflight = {
            "passed": True,
            "objective": "dense_categorical_forward_kl_only",
            "global_step": v2.START_STEP,
            "prepared_batch_cached_for_first_audit_point": True,
            "sparse_ce_computed": False,
            "sparse_ce_graph_constructed": False,
            "data": dict(demo_anchor.anchor.data_provenance),
            **preflight_metrics,
        }
        live_wrapper = v1._ProposalOnly(proposal)
        live_optimizer_sentinel = v1.build_optimizer(
            live_wrapper,
            lr=float(cfg["optim"]["lr"]),
            weight_decay=float(cfg["optim"]["weight_decay"]),
            betas=tuple(float(value) for value in cfg["optim"]["betas"]),
            lr_scales={"proposal": float(cfg["optim"]["lr_scales"]["proposal"])},
            module_names=["proposal"],
        )
        _require(len(live_optimizer_sentinel.state) == 0,
                 "live optimizer sentinel did not begin with empty state")
    except Exception as exc:  # noqa: BLE001
        local_error = f"{type(exc).__name__}: {exc}"
    v1._raise_if_any_rank_failed(local_error, world, "seed/demo-anchor construction")
    anchor_preflight_rows = component_audit._all_gather_object(anchor_preflight, world)
    del parent

    train_collections: list[v1.ValidatedRecoveryCollection] = []
    local_error = ""
    try:
        for spec in cfg["outcome_grpo_v2"]["folds"]:
            train_collections.append(v1.ValidatedRecoveryCollection.open(
                ROOT / str(spec["path"]),
                checkpoint_identity=parent_identity,
                expected_split=str(spec["split"]),
                deep=False,
                verify_sidecars=False,
            ))
        lineage_rows = cfg["outcome_grpo_v2"]["authenticated_data_lineage"][
            "training"
        ]
        _require(len(lineage_rows) == len(train_collections),
                 "v2 TRAIN lineage row count changed")
        for collection, pinned in zip(
            train_collections, lineage_rows, strict=True,
        ):
            _require(
                collection.split == str(pinned["split"])
                and str(collection.root) == str((ROOT / str(pinned["path"])).resolve())
                and collection.manifest_sha256 == str(pinned["manifest_sha256"])
                and collection.identity_digest == str(pinned["identity_digest"]),
                f"opened {collection.split} differs from v2 authenticated lineage",
            )
        validation = v1.ValidatedRecoveryCollection.open(
            validation_root,
            checkpoint_identity=parent_identity,
            expected_split="validation",
            deep=False,
            verify_sidecars=False,
        )
        _require(validation.identity_digest == EXPECTED_VALIDATION_IDENTITY_DIGEST,
                 "opened development collection identity differs from panel receipt")
        current_development = cfg["outcome_grpo_v2"]["validation_lineage"][
            "current_development_collection"
        ]
        _require(validation.manifest_sha256
                 == str(current_development["manifest_sha256"]),
                 "opened development manifest differs from v2 lineage pin")
        receipt_by_group = {
            str(receipt["group_id"]): receipt for receipt in validation.receipts
        }
        sampling_receipt = attach_panel_sampling_receipt(
            group_receipt, receipt_by_group,
        )
        for row in group_receipt["ordered_rows"]:
            current = receipt_by_group[str(row["group_id"])]
            _require(str(current["sha256"]) == str(row["sidecar_sha256"]),
                     f"panel sidecar pin changed: {row['group_id']}")
        sampler = _build_round_robin_sampler(
            train_collections,
            seed=seed,
            rank=rank,
            contexts_per_arm=int(
                cfg["outcome_grpo_v2"]["sampler"]["contexts_per_arm"]
            ),
        )
    except Exception as exc:  # noqa: BLE001
        local_error = f"{type(exc).__name__}: {exc}"
    v1._raise_if_any_rank_failed(local_error, world, "recovery collection authentication")

    train_points: list[dict[str, Any]] = []
    train_selection_rows: list[dict[str, Any]] = []
    step_direct_gradients: list[dict[str, Tensor]] = []
    nondeterminism_warnings: list[str] = []
    for offset, step in enumerate(AUDIT_STEPS):
        set_step_seed(seed, step, rank)
        local_error = ""
        try:
            fold, group_index, visit = sampler.group_at(step)
            collection = train_collections[fold]
            receipt = collection.receipts[group_index]
            replan_indices = sampler.replans_at(
                step, receipt["n_replans_by_arm"],
            )
            payload = collection.load(group_index)
            authentication = component_audit.authenticate_selected_contexts(
                proposal, payload, replan_indices, device=device,
            )
            objectives = v2.sampled_group_objectives_v2(
                proposal, payload, replan_indices, device=device,
            )
            ratio_identity = v1._require_initial_behavior_ratio_identity(
                objectives.metrics, device=device,
            )
            demo_reference, demo_reference_metrics = _demo_reference_only(
                demo_anchor, step,
            )
        except Exception as exc:  # noqa: BLE001
            local_error = f"{type(exc).__name__}: {exc}"
        v1._raise_if_any_rank_failed(
            local_error, world, f"round-robin offset {offset} forward/authentication",
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            local_error = ""
            try:
                named = _named_live_parameters(proposal)
                direct_grpo, direct_grpo_missing = _synchronised_loss_gradient(
                    objectives.grpo,
                    named,
                    world=world,
                    retain_graph=True,
                    label=f"offset {offset} first direct GRPO",
                    require_complete=True,
                )
                repeated_direct_grpo, repeated_grpo_missing = (
                    _synchronised_loss_gradient(
                        objectives.grpo,
                        named,
                        world=world,
                        retain_graph=True,
                        label=f"offset {offset} repeated direct GRPO",
                        require_complete=True,
                    )
                )
                grpo_repeat = direct_repeat_consistency(
                    direct_grpo,
                    repeated_direct_grpo,
                    label=f"offset {offset} GRPO",
                )
                grpo_norm = component_audit._vector_norm(direct_grpo)
                recovery_reference_evidence = _reference_gradient_evidence(
                    objectives.recovery_forward_kl,
                    proposal,
                    grpo_norm=grpo_norm,
                    world=world,
                    retain_graph=True,
                    label="recovery PL",
                )
                demo_reference_evidence = _reference_gradient_evidence(
                    demo_reference,
                    proposal,
                    grpo_norm=grpo_norm,
                    world=world,
                    retain_graph=True,
                    label="demo categorical",
                )
                direct_full_loss = (
                    objectives.grpo
                    + BALANCE_WEIGHT * objectives.balance
                    + objectives.recovery_forward_kl
                    + demo_reference
                )
                direct_full, direct_full_missing = (
                    _synchronised_loss_gradient(
                        direct_full_loss,
                        named,
                        world=world,
                        retain_graph=True,
                        label=f"offset {offset} first direct full",
                        require_complete=True,
                    )
                )
                repeated_direct_full, repeated_full_missing = (
                    _synchronised_loss_gradient(
                        direct_full_loss,
                        named,
                        world=world,
                        retain_graph=False,
                        label=f"offset {offset} repeated direct full",
                        require_complete=True,
                    )
                )
                full_repeat = direct_repeat_consistency(
                    direct_full,
                    repeated_direct_full,
                    label=f"offset {offset} full alpha0 beta1 lambda1",
                )
                reference_vectors_bitwise_zero = (
                    recovery_reference_evidence[
                        "synchronised_gradient_bitwise_all_zero"
                    ]
                    and demo_reference_evidence[
                        "synchronised_gradient_bitwise_all_zero"
                    ]
                )
                _require(not demo_anchor.unexpected_gradients(),
                         "frozen estimator/q_action/seed proposal received gradients")
                torch.cuda.synchronize(device)
            except Exception as exc:  # noqa: BLE001
                local_error = f"{type(exc).__name__}: {exc}"
        v1._raise_if_any_rank_failed(
            local_error, world, f"round-robin offset {offset} gradient audit",
        )
        nondeterminism_warnings.extend(_warnings_checked(
            caught, world=world, label=f"round-robin offset {offset} backward",
        ))
        local_selection = _receipt_row(
            collection,
            group_index,
            rank=rank,
            replan_indices=replan_indices,
            extra={
                "global_step": int(step), "offset": offset,
                "fold": int(fold), "visit": int(visit),
                "selected_context_authentication": authentication,
                "ratio_identity": ratio_identity,
            },
        )
        selections = component_audit._all_gather_object(local_selection, world)
        _require(len({(row["split"], row["group_index"]) for row in selections})
                 == world,
                 f"round-robin offset {offset} selected duplicate rank groups")
        train_selection_rows.extend(selections)
        reduced_metrics = v1._reduce_training_metrics(
            objectives.metrics, world, device,
        )
        _require(reduced_metrics["ratio_min"] == 1.0
                 and reduced_metrics["ratio_mean"] == 1.0
                 and reduced_metrics["ratio_max"] == 1.0
                 and reduced_metrics["max_abs_logratio"] == 0.0,
                 f"round-robin offset {offset} seed ratio is not exact identity")
        mean_losses = {
            key: _mean_scalar(value, world=world, device=device)
            for key, value in {
                "grpo": float(objectives.grpo.detach()),
                "switch_balance_1e-2": float(
                    (BALANCE_WEIGHT * objectives.balance).detach()
                ),
                "recovery_reference": float(
                    objectives.recovery_forward_kl.detach()
                ),
                "demo_reference": float(demo_reference.detach()),
                "direct_full_alpha0_beta1_lambda1": float(
                    direct_full_loss.detach()
                ),
            }.items()
        }
        if rank == 0:
            train_points.append({
                "offset": offset, "global_step": int(step),
                "rank_local_groups": selections,
                "fold_counts": {
                    str(fold_index): sum(int(row["fold"]) == fold_index
                                         for row in selections)
                    for fold_index in range(v1.N_FOLDS)
                },
                "global_mean_losses": mean_losses,
                "global_ratio_identity": {
                    "ratio_atoms": int(reduced_metrics["ratio_atoms"]),
                    "ratio_min": reduced_metrics["ratio_min"],
                    "ratio_mean": reduced_metrics["ratio_mean"],
                    "ratio_max": reduced_metrics["ratio_max"],
                    "max_abs_logratio": reduced_metrics["max_abs_logratio"],
                    "all_exact": True,
                },
                "reference_gradients": {
                    "recovery_pl_forward_kl": recovery_reference_evidence,
                    "demo_categorical_forward_kl": demo_reference_evidence,
                },
                "direct_grpo_gradient": {
                    "authoritative_vector": "first_direct_backward",
                    "scalar": "GRPO",
                    "preclip_norm": grpo_norm,
                    "missing_parameter_gradients": direct_grpo_missing,
                    "repeated_missing_parameter_gradients": (
                        repeated_grpo_missing
                    ),
                    "repeat_consistency": grpo_repeat,
                },
                "direct_full_alpha0_beta1_lambda1": {
                    "authoritative_vector": "first_direct_backward",
                    "authoritative_for_full_gated_endpoints": True,
                    "scalar": (
                        "GRPO + .01*balance + 1*recovery_ref + 1*demo_ref"
                    ),
                    "sparse_ce_weight": PILOT_SPARSE_ALPHA,
                    "sparse_ce_graph_included": False,
                    "disabled_sparse_ce_graph_excluded": True,
                    "prospective_64_pilot_coefficients_predeclared": {
                        "sparse_ce_alpha": PILOT_SPARSE_ALPHA,
                        "initial_recovery_beta": PILOT_INITIAL_RECOVERY_BETA,
                        "demo_reference_lambda": PILOT_DEMO_REFERENCE_LAMBDA,
                    },
                    "preclip_norm": component_audit._vector_norm(
                        direct_full
                    ),
                    "missing_parameter_gradients": direct_full_missing,
                    "repeated_missing_parameter_gradients": (
                        repeated_full_missing
                    ),
                    "repeat_consistency": full_repeat,
                    "reference_vectors_bitwise_zero": (
                        reference_vectors_bitwise_zero
                    ),
                    "reference_gradient_bounds_passed": (
                        recovery_reference_evidence["bound_passed"]
                        and demo_reference_evidence["bound_passed"]
                    ),
                },
                "demo_reference_metrics_rank0": demo_reference_metrics,
            })
        step_direct_gradients.append({
            "direct_grpo": direct_grpo.detach().clone(),
            "direct_prefrozen_full": direct_full.detach().clone(),
            "reference_vectors_bitwise_zero": torch.tensor(
                float(reference_vectors_bitwise_zero), device=device,
            ),
            "reference_gradient_bounds_passed": torch.tensor(
                float(
                    recovery_reference_evidence["bound_passed"]
                    and demo_reference_evidence["bound_passed"]
                ),
                device=device,
            ),
        })
        del payload, objectives, demo_reference, direct_full_loss
        del repeated_direct_grpo, repeated_direct_full
        gc.collect()
        torch.cuda.empty_cache()

    _require(len(train_selection_rows) == EXPECTED_TRAIN_DRAWS,
             "round-robin audit did not retain exactly 24 TRAIN draws")
    _require(len({(row["split"], int(row["group_index"]))
                  for row in train_selection_rows}) == EXPECTED_TRAIN_DRAWS,
             "round-robin TRAIN draws are not all distinct")
    fold_totals = {
        fold: sum(int(row["fold"]) == fold for row in train_selection_rows)
        for fold in range(v1.N_FOLDS)
    }
    _require(all(value == EXPECTED_FOLD_DRAWS for value in fold_totals.values()),
             f"three-step fold mixture is not four draws per fold: {fold_totals}")

    grpo_gradients = [
        row["direct_grpo"] for row in step_direct_gradients
    ]
    prefrozen_full_gradients = [
        row["direct_prefrozen_full"] for row in step_direct_gradients
    ]
    reference_vectors_bitwise_zero = all(
        bool(int(row["reference_vectors_bitwise_zero"].item()))
        for row in step_direct_gradients
    )
    reference_gradient_bounds_passed = all(
        bool(int(row["reference_gradient_bounds_passed"].item()))
        for row in step_direct_gradients
    )
    grpo_sgd = cumulative_clipped_sgd_direction(
        proposal, grpo_gradients, cfg=cfg,
    )
    full_sgd = cumulative_clipped_sgd_direction(
        proposal, prefrozen_full_gradients, cfg=cfg,
    )
    adamw_production = virtual_adamw_clone_replay(
        proposal, prefrozen_full_gradients, cfg=cfg,
    )
    for expected, actual in zip(
        full_sgd["clipped_gradients"],
        adamw_production["vectors"]["clipped_gradients"],
        strict=True,
    ):
        _require(torch.equal(expected, actual),
                 "SGD control and AdamW replay used different clipped gradients")

    primary_vectors = {
        "grpo_only_cumulative_clipped_sgd": grpo_sgd["delta"],
        "full_alpha0_beta1_lambda1_cumulative_clipped_sgd": full_sgd["delta"],
        "full_alpha0_beta1_lambda1_reset_adamw_with_production_decay": (
            adamw_production["vectors"]["cumulative"][-1]
        ),
    }
    increment_vectors = {
        f"full_alpha0_beta1_lambda1_adamw_decay_increment_t{offset}": value
        for offset, value in enumerate(
            adamw_production["vectors"]["increments"]
        )
    }

    all_direction_vectors = {
        **primary_vectors, **increment_vectors,
    }
    direction_norms = {
        key: component_audit._vector_norm(value)
        for key, value in all_direction_vectors.items()
    }
    direction_digest = _canonical_sha256({
        key: {"norm": direction_norms[key], "numel": int(value.numel())}
        for key, value in all_direction_vectors.items()
    })
    direction_rows = component_audit._all_gather_object(
        {"rank": rank, "digest": direction_digest, "norms": direction_norms}, world,
    )
    _require(all(row["digest"] == direction_digest for row in direction_rows),
             "constructed direction summaries differ across ranks")

    # Each task stays on exactly one rank, so its one/two group gradients can be
    # averaged before the 40 tasks are equally aggregated across ranks.
    sampling_by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in sampling_receipt["rows"]:
        if int(row["rank"]) == rank:
            sampling_by_task[str(row["task_key"])].append(row)
    _require(len(sampling_by_task) == PANEL_TASKS_PER_RANK,
             f"rank {rank} did not receive five panel tasks")
    panel_local_task_rows: list[dict[str, Any]] = []
    panel_local_selection_rows: list[dict[str, Any]] = []
    heldout_local = torch.zeros_like(next(iter(primary_vectors.values())))
    group_index_by_id = {
        str(receipt["group_id"]): index
        for index, receipt in enumerate(validation.receipts)
    }
    ordered_task_keys = sorted(
        str(row["task_key"]) for row in group_receipt["ordered_rows"]
    )
    ordered_task_keys = sorted(set(ordered_task_keys))
    for task_index, key in enumerate(sorted(sampling_by_task)):
        group_gradients: list[Tensor] = []
        group_losses: list[float] = []
        task_authentication: list[dict[str, Any]] = []
        for selected in sampling_by_task[key]:
            group_id = str(selected["group_id"])
            group_index = group_index_by_id[group_id]
            receipt = validation.receipts[group_index]
            _require(str(receipt["sha256"]) == str(selected["sidecar_sha256"]),
                     f"panel sidecar SHA pin differs at use: {group_id}")
            replans = {
                int(arm): tuple(int(value) for value in values)
                for arm, values in selected["replan_indices"].items()
            }
            set_step_seed(seed, v2.START_STEP + 100 + task_index, rank)
            local_error = ""
            try:
                payload = validation.load(group_index)
                authentication = component_audit.authenticate_selected_contexts(
                    proposal, payload, replans, device=device,
                )
                objective = v2.sampled_group_objectives_v2(
                    proposal, payload, replans, device=device,
                )
                ratio_identity = v1._require_initial_behavior_ratio_identity(
                    objective.metrics, device=device,
                )
                local_gradient, missing = component_audit._local_gradient_vector(
                    objective.grpo,
                    _named_live_parameters(proposal),
                    retain_graph=False,
                )
                _require(not missing, f"panel GRPO gradient missing parameters: {missing[:8]}")
                _require(all(parameter.grad is None
                             for parameter in proposal.parameters()),
                         "panel autograd populated live proposal gradients")
            except Exception as exc:  # noqa: BLE001
                local_error = f"{type(exc).__name__}: {exc}"
            v1._raise_if_any_rank_failed(
                local_error, world,
                f"panel task {key} group {group_id} authentication/gradient",
            )
            group_gradients.append(local_gradient)
            group_losses.append(float(objective.grpo.detach()))
            task_authentication.append({
                "group_id": group_id,
                "selected_context_authentication": authentication,
                "ratio_identity": ratio_identity,
            })
            panel_local_selection_rows.append(_receipt_row(
                validation,
                group_index,
                rank=rank,
                replan_indices=replans,
                extra={"task_key": key, "panel_role": selected["panel_role"]},
            ))
            del payload, objective, local_gradient
        task_gradient, task_contribution = equal_group_within_task_contribution(
            group_gradients,
        )
        heldout_local.add_(task_contribution)
        projections = {
            name: component_audit._dot(task_gradient, direction)
            for name, direction in all_direction_vectors.items()
        }
        panel_local_task_rows.append({
            "task_key": key,
            "suite": task_suite(key),
            "rank": rank,
            "groups": len(group_gradients),
            "equal_group_within_task_loss": sum(group_losses) / len(group_losses),
            "task_gradient_norm": component_audit._vector_norm(task_gradient),
            "authentication": task_authentication,
            "projections": projections,
        })
        del group_gradients, task_gradient
        gc.collect()
        torch.cuda.empty_cache()

    heldout_gradient = _sum_synchronised_gradient(heldout_local, world=world)
    gathered_task_lists = component_audit._all_gather_object(
        panel_local_task_rows, world,
    )
    panel_task_rows = [row for rank_rows in gathered_task_lists for row in rank_rows]
    panel_task_rows.sort(key=lambda row: str(row["task_key"]))
    _require(len(panel_task_rows) == PANEL_TASKS
             and len({row["task_key"] for row in panel_task_rows}) == PANEL_TASKS,
             "panel task gradients do not cover all 40 tasks exactly once")
    _require({row["groups"] for row in panel_task_rows}.issubset({1, 2})
             and sum(int(row["groups"]) for row in panel_task_rows) == PANEL_GROUPS,
             "panel group-within-task multiplicities changed")
    panel_selection_lists = component_audit._all_gather_object(
        panel_local_selection_rows, world,
    )
    panel_selection_rows = [row for rank_rows in panel_selection_lists
                            for row in rank_rows]
    _require(len(panel_selection_rows) == PANEL_GROUPS,
             "panel did not authenticate exactly 48 sidecars")

    resample_matrix, bootstrap_receipt = make_suite_stratified_resample_matrix(
        [str(row["task_key"]) for row in panel_task_rows],
    )
    _require(bootstrap_receipt["sha256"] == EXPECTED_BOOTSTRAP_MATRIX_SHA256,
             f"fixed bootstrap matrix drifted: {bootstrap_receipt['sha256']}")
    primary_names = tuple(primary_vectors)
    endpoint_values = torch.tensor([
        [float(row["projections"][name]) for row in panel_task_rows]
        for name in primary_names
    ], dtype=torch.float64)
    endpoint_bounds = bonferroni_task_bounds(endpoint_values, resample_matrix)
    increment_names = tuple(increment_vectors)
    increment_values = torch.tensor([
        [float(row["projections"][name]) for row in panel_task_rows]
        for name in increment_names
    ], dtype=torch.float64)
    increment_bounds = bonferroni_task_bounds(increment_values, resample_matrix)
    endpoint_cosines = [
        aggregate_benefit_cosine(heldout_gradient, primary_vectors[name])
        for name in primary_names
    ]
    increment_cosines = [
        aggregate_benefit_cosine(heldout_gradient, increment_vectors[name])
        for name in increment_names
    ]
    decision = decide_direction_gate(
        endpoint_bounds=endpoint_bounds,
        endpoint_benefit_cosines=endpoint_cosines,
        adamw_increment_bounds=increment_bounds,
        adamw_increment_benefit_cosines=increment_cosines,
        reference_gradient_bounds_passed=reference_gradient_bounds_passed,
        reference_vectors_bitwise_zero=reference_vectors_bitwise_zero,
    )

    # Check equal-task scalar projections against the explicitly reduced
    # aggregate heldout gradient.  The small tolerance covers only fp32 SUM
    # order; the bootstrap always uses the 40 task scalars directly.
    projection_closure: dict[str, Any] = {}
    for name, direction in all_direction_vectors.items():
        task_mean = sum(float(row["projections"][name])
                        for row in panel_task_rows) / PANEL_TASKS
        aggregate_dot = component_audit._dot(heldout_gradient, direction)
        residual = aggregate_dot - task_mean
        mean_absolute_task_projection = sum(
            abs(float(row["projections"][name])) for row in panel_task_rows
        ) / PANEL_TASKS
        scale = max(
            abs(aggregate_dot), abs(task_mean),
            mean_absolute_task_projection, 1e-30,
        )
        relative = abs(residual) / scale
        _require(relative <= 2e-4,
                 f"equal-task aggregate projection residual too large for {name}: {relative}")
        projection_closure[name] = {
            "task_mean": task_mean,
            "aggregate_gradient_dot": aggregate_dot,
            "absolute_residual": abs(residual),
            "mean_absolute_task_projection": mean_absolute_task_projection,
            "relative_residual": relative,
            "max_relative_residual": 2e-4,
            "passed": True,
        }

    collection_map = {collection.split: collection for collection in train_collections}
    collection_map[validation.split] = validation
    all_selection_rows = [*train_selection_rows, *panel_selection_rows]
    local_error = ""
    try:
        rehash_local = rehash_rank_sidecars(
            collection_map, all_selection_rows, rank=rank,
        )
    except Exception as exc:  # noqa: BLE001
        local_error = f"{type(exc).__name__}: {exc}"
    v1._raise_if_any_rank_failed(local_error, world, "72-sidecar post-use rehash")
    rehash_rows = component_audit._all_gather_object(rehash_local, world)
    _require(sum(int(row["selected_sidecars"]) for row in rehash_rows)
             == EXPECTED_TRAIN_DRAWS + PANEL_GROUPS,
             "post-use closure did not rehash all 72 selected sidecars")

    local_error = ""
    try:
        proposal_digest_after = v1.proposal_module_digest(proposal.state_dict())
        demo_anchor.assert_seed_unchanged()
        _require(proposal_digest_after == proposal_digest_before,
                 "live proposal digest changed during diagnostic")
        _require(len(live_optimizer_sentinel.state) == 0,
                 "live optimizer sentinel accumulated state")
        _require(all(parameter.grad is None for parameter in proposal.parameters()),
                 "live proposal retained gradient buffers")
        _require(not demo_anchor.unexpected_gradients(),
                 "frozen anchor/reference modules retained gradients")
        for collection in train_collections:
            collection.assert_unchanged()
        validation.assert_unchanged()
        v1._assert_seed_stat(parent_identity)
        _require(_source_identity() == source_identity,
                 "source closure changed during direction audit")
        _require(_config_file_identity(config) == config_file_identity,
                 "canonical config stat/SHA changed during direction audit")
        _require(v1._strict_outcome_determinism_state()
                 == v1.STRICT_OUTCOME_DETERMINISM,
                 "strict deterministic flags changed during direction audit")
        _exact_scoring_evidence(proposal, device)
        torch.cuda.synchronize(device)
    except Exception as exc:  # noqa: BLE001
        local_error = f"{type(exc).__name__}: {exc}"
    v1._raise_if_any_rank_failed(local_error, world, "final no-mutation closure")
    mutation_rows = component_audit._all_gather_object({
        "rank": rank,
        "proposal_digest_before": proposal_digest_before,
        "proposal_digest_after": proposal_digest_after,
        "live_optimizer_state_entries": len(live_optimizer_sentinel.state),
        "live_parameter_grad_buffers": sum(
            parameter.grad is not None for parameter in proposal.parameters()
        ),
    }, world)

    if rank != 0:
        torch.distributed.barrier()
        return None

    _require(len(train_points) == len(AUDIT_STEPS),
             "rank zero did not retain three synchronized TRAIN points")
    report = {
        "format_version": FORMAT_VERSION,
        "kind": KIND,
        "status": decision["status"],
        "execution_validated": True,
        "decision": decision,
        "eligibility": dict(ELIGIBILITY),
        "instrumentation_history": {
            "decision_input": False,
            "previous_invalid_jobs": [
                dict(row) for row in INVALID_INSTRUMENTATION_HISTORY
            ],
            "both_stopped_before_panel_statistics": True,
            "both_published_no_scientific_report": True,
        },
        "source_identity": source_identity,
        "config": {
            "path": str(config), "resolved_hash": config_hash,
            "raw_file_identity": config_file_identity,
            "scaffold_validation": scaffold_validation,
            "pre_frozen_sparse_ce_alpha": 0.0,
            "sparse_ce_disabled_graph_excluded": True,
            "prospective_64_pilot_initial_recovery_beta": (
                PILOT_INITIAL_RECOVERY_BETA
            ),
            "prospective_64_pilot_demo_reference_lambda": (
                PILOT_DEMO_REFERENCE_LAMBDA
            ),
            "seed_checkpoint": EXPECTED_SEED_CHECKPOINT,
            "seed_global_step": v2.START_STEP,
        },
        "parent": parent_identity,
        "runtime_by_rank": runtimes,
        "strict_determinism": strict,
        "exact_scoring": {
            "configured": scoring_config,
            "validated": scoring_evidence,
            "all_train_and_panel_selected_contexts_bitwise_authenticated": True,
            "all_seed_ratios_exactly_one": True,
        },
        "geometry": {
            "world_size": world, "nodes": 1, "gpus_per_node": world,
            "global_steps": list(AUDIT_STEPS),
            "round_robin_offsets": list(AUDIT_OFFSETS),
            "train_draws": EXPECTED_TRAIN_DRAWS,
            "train_draws_per_fold": EXPECTED_FOLD_DRAWS,
            "fold_draw_counts": {str(key): value for key, value in fold_totals.items()},
            "panel_tasks": PANEL_TASKS,
            "panel_groups": PANEL_GROUPS,
            "panel_extra_second_groups": PANEL_EXTRA_GROUPS,
            "panel_contexts_per_sampled_arm": PANEL_REPLANS_PER_ARM,
            "equal_group_within_task_then_equal_task": True,
        },
        "outcome_blind_panel": {
            "terminal_rewards_not_used_or_accessed_by_selection_logic": True,
            "manifest_parser_materializes_unprojected_fields": True,
            "group_receipt": group_receipt,
            "sampling_receipt": sampling_receipt,
            "collection": validation.provenance(),
            "task_rows": panel_task_rows,
        },
        "train_collections": [collection.provenance()
                              for collection in train_collections],
        "train_points": train_points,
        "anchor_preflight_by_rank": anchor_preflight_rows,
        "direction_construction": {
            "measurement_scope": "authoritative_direct_gradients_only",
            "primary_alpha": PILOT_SPARSE_ALPHA,
            "primary_initial_recovery_beta": PILOT_INITIAL_RECOVERY_BETA,
            "primary_demo_reference_lambda": PILOT_DEMO_REFERENCE_LAMBDA,
            "direct_full_scalar": (
                "GRPO + 0.01*Switch_balance + "
                "1*recovery_reference + 1*demo_reference"
            ),
            "direct_grpo_scalar": "GRPO",
            "first_direct_vectors_authoritative": True,
            "independent_repeat_checks": {
                "direct_grpo": True,
                "direct_full_alpha0_beta1_lambda1": True,
                "max_relative_residual": (
                    MAX_DIRECT_REPEAT_RELATIVE_RESIDUAL
                ),
            },
            "sparse_ce_weight": PILOT_SPARSE_ALPHA,
            "sparse_ce_graph_included": False,
            "disabled_sparse_ce_graph_excluded": True,
            "sparse_ce_computed": False,
            "recovery_and_demo_reference_first_derivatives": "required_bounded",
            "reference_gradient_bounds_passed": reference_gradient_bounds_passed,
            "reference_vectors_bitwise_zero": reference_vectors_bitwise_zero,
            "prospective_coefficients_frozen_before_gpu_result": True,
            "pilot_still_requires_exact_recipe_freeze_and_controller_smoke": True,
            "primary_endpoint_order": list(primary_names),
            "primary_endpoint_norms": {
                name: direction_norms[name] for name in primary_names
            },
            "grpo_only_clipped_sgd": {
                key: value for key, value in grpo_sgd.items()
                if key not in {"delta", "clipped_gradients"}
            },
            "full_alpha0_beta1_lambda1_clipped_sgd": {
                key: value for key, value in full_sgd.items()
                if key not in {"delta", "clipped_gradients"}
            },
            "reset_adamw_with_production_decay": _without_vectors(
                adamw_production
            ),
            "live_optimizer_steps": 0,
            "virtual_clone_optimizer_steps": 3,
            "frozen_gradient_replay_caveat": (
                "all three gradients are measured at the unchanged seed; clone AdamW "
                "does not recompute gradients at its virtual parameters"
            ),
        },
        "primary_endpoint_task_bootstrap": endpoint_bounds,
        "production_adamw_increment_task_bootstrap": increment_bounds,
        "aggregate_heldout_gradient": {
            "definition": "equal group within task, then equal mean over 40 tasks",
            "norm": component_audit._vector_norm(heldout_gradient),
            "primary_endpoint_benefit_cosines": {
                name: endpoint_cosines[index]
                for index, name in enumerate(primary_names)
            },
            "production_adamw_increment_benefit_cosines": {
                name: increment_cosines[index]
                for index, name in enumerate(increment_names)
            },
            "projection_closure": projection_closure,
        },
        "bootstrap_resample_matrix": bootstrap_receipt,
        "selected_sidecar_post_use_closure": {
            "passed": True,
            "sidecars": EXPECTED_TRAIN_DRAWS + PANEL_GROUPS,
            "rank_evidence": rehash_rows,
            "bytes": sum(int(row["selected_bytes"]) for row in rehash_rows),
        },
        "no_mutation": {
            "passed": True,
            "live_optimizer_constructed_as_empty_state_sentinel": True,
            "live_optimizer_steps": 0,
            "virtual_clone_optimizer_steps": 3,
            "live_parameter_perturbations": 0,
            "proposal_digest_before": proposal_digest_before,
            "proposal_digest_after": proposal_digest_after,
            "rank_evidence": mutation_rows,
            "checkpoint_emitted": False,
            "candidate_emitted": False,
        },
        "warnings": sorted(set(nondeterminism_warnings)),
        "wall_seconds": float(time.monotonic() - started),
        "output": str(output),
    }
    exclusive_json_write(output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), flush=True)
    torch.distributed.barrier()
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/r0a_outcome_grpo_v2_pilot.yaml",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    exit_code = 0
    try:
        run_audit(config_path=args.config, output_path=args.out)
    except Exception as exc:  # noqa: BLE001
        print(
            "OUTCOME_ROUND_ROBIN_DIRECTION_AUDIT_INVALID: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        exit_code = 2
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
