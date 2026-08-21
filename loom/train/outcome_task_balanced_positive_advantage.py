"""Task-balanced full-coverage positive-advantage primitives.

This module is an isolated mathematical and scheduling core.  It has no
trainer, launcher, checkpoint writer, candidate path, or evaluation authority.
It defines one terminal prospective method: every authenticated informative
TRAIN group is used exactly once in one of three deterministic macro shards,
with equal group weight inside each task and equal weight across all 40 tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from loom.eval import outcome_recovery as recovery
from loom.train import outcome_grpo as v1
from loom.train import outcome_grpo_v2 as v2
from loom.train import outcome_positive_advantage as pa


FORMAT_VERSION = 1
CORE_KIND = "loom_task_balanced_positive_advantage_core"
PARTITION_KIND = "loom_task_balanced_positive_advantage_selection"
PARTITION_DOMAIN = "pa-full-coverage-equal-task-v1"
PARTITION_SCHEME = (
    "sha256-domain-task-and-group-order-then-task-position-rotated-mod3-v1"
)
N_SHARDS = 3
EXPECTED_WORLD_SIZE = 8
CONTEXTS_PER_ARM = 2
SAMPLER_SEED = 0
START_STEP = 49_666
MACRO_UPDATES = 3
STOP_STEP = START_STEP + MACRO_UPDATES
EXPECTED_INFORMATIVE_GROUPS = 903
SUITES = ("libero_goal", "libero_long", "libero_object", "libero_spatial")
TASK_KEYS = tuple(
    f"{suite}/task={task:02d}" for suite in SUITES for task in range(10)
)
TASK_COUNT = len(TASK_KEYS)
TASKS_PER_RANK = TASK_COUNT // EXPECTED_WORLD_SIZE
_TASK_PATTERN = re.compile(
    r"^(libero_goal|libero_long|libero_object|libero_spatial)/"
    r"task=(\d{2})/trial=\d+/seed=\d+$"
)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ROOT = Path(__file__).resolve().parents[2]
_CORE_SOURCE_FILES = (
    "contracts.py",
    "loom/eval/outcome_recovery.py",
    "loom/heads/proposal.py",
    "loom/train/outcome_grpo.py",
    "loom/train/outcome_grpo_v2.py",
    "loom/train/outcome_positive_advantage.py",
    "loom/train/outcome_task_balanced_positive_advantage.py",
)


class TaskBalancedPositiveAdvantageError(RuntimeError):
    """The fixed task-balanced method received unauthenticated/invalid state."""


def _require(condition: Any, message: str) -> None:
    if not bool(condition):
        raise TaskBalancedPositiveAdvantageError(message)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(
            dict(row), sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _stable_file_bytes(path: Path) -> tuple[bytes, dict[str, Any]]:
    resolved = path.expanduser().resolve()
    before = resolved.stat()
    encoded = resolved.read_bytes()
    after = resolved.stat()
    before_key = (
        int(before.st_dev), int(before.st_ino), int(before.st_size),
        int(before.st_mtime_ns),
    )
    after_key = (
        int(after.st_dev), int(after.st_ino), int(after.st_size),
        int(after.st_mtime_ns),
    )
    _require(before_key == after_key and len(encoded) == int(after.st_size),
             f"file changed while reading: {resolved}")
    return encoded, {
        "path": str(resolved),
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "size": int(after.st_size),
        "mtime_ns": int(after.st_mtime_ns),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def task_key_from_group_id(group_id: str) -> str:
    match = _TASK_PATTERN.fullmatch(str(group_id))
    if match is None:
        raise ValueError(f"group_id does not name one canonical LIBERO task: {group_id!r}")
    task_key = f"{match.group(1)}/task={int(match.group(2)):02d}"
    _require(task_key in TASK_KEYS, f"unknown task key {task_key!r}")
    return task_key


def informative_terminal_rewards(rewards: Sequence[float] | Tensor) -> bool:
    values = torch.as_tensor(rewards, dtype=torch.float32).detach().reshape(-1)
    if values.numel() != recovery.GROUP_SIZE:
        raise ValueError(
            f"terminal reward vector must contain {recovery.GROUP_SIZE} arms, "
            f"got {values.numel()}"
        )
    _require(bool(torch.isfinite(values).all()),
             "terminal reward vector contains nan/inf")
    _require(bool(((values == 0.0) | (values == 1.0)).all()),
             "terminal success rewards must be canonical exact 0/1 values")
    # This deliberately preserves ValidatedRecoveryCollection's inherited
    # eligibility definition.  The exact-{0,1} check above prevents an invalid
    # fractional receipt from being silently changed by the integer cast.
    return len({int(value) for value in values.tolist()}) > 1


@dataclass(frozen=True)
class AuthenticatedInformativeGroup:
    split: str
    manifest_sha256: str
    identity_digest: str
    group_index: int
    group_id: str
    task_key: str
    sidecar: str
    sidecar_sha256: str
    sidecar_size: int
    n_replans_by_arm: tuple[int, ...]
    terminal_rewards: tuple[float, ...]

    @property
    def identity_key(self) -> tuple[str, int]:
        return self.split, self.group_index

    def union_row(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "manifest_sha256": self.manifest_sha256,
            "identity_digest": self.identity_digest,
            "group_index": self.group_index,
            "group_id": self.group_id,
            "task_key": self.task_key,
            "sidecar": self.sidecar,
            "sidecar_sha256": self.sidecar_sha256,
            "sidecar_size": self.sidecar_size,
            "n_replans_by_arm": list(self.n_replans_by_arm),
            "terminal_rewards": list(self.terminal_rewards),
        }


@dataclass(frozen=True)
class TaskBalancedAssignment:
    group: AuthenticatedInformativeGroup
    task_position: int
    within_task_position: int
    shard: int
    rank: int
    order_sha256: str

    def row(self) -> dict[str, Any]:
        return {
            **self.group.union_row(),
            "task_position": self.task_position,
            "within_task_position": self.within_task_position,
            "shard": self.shard,
            "rank": self.rank,
            "order_sha256": self.order_sha256,
        }


def _validate_source_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "split", "path", "manifest_sha256", "identity_digest",
        "informative_groups",
    }
    _require(set(spec) == required,
             f"TRAIN source pin fields differ: {sorted(spec)}")
    row = dict(spec)
    _require(re.fullmatch(r"train[0-5]", str(row["split"])) is not None,
             f"invalid TRAIN split {row['split']!r}")
    _require(_HEX64.fullmatch(str(row["manifest_sha256"])) is not None,
             f"invalid manifest SHA for {row['split']}")
    _require(_HEX64.fullmatch(str(row["identity_digest"])) is not None,
             f"invalid identity digest for {row['split']}")
    _require(int(row["informative_groups"]) > 0,
             f"nonpositive informative count for {row['split']}")
    return row


def load_authenticated_informative_groups(
    root: str | Path,
    source_specs: Sequence[Mapping[str, Any]],
) -> tuple[tuple[AuthenticatedInformativeGroup, ...], tuple[dict[str, Any], ...]]:
    """Read exactly six pinned manifests and return all informative receipts."""
    base = Path(root).expanduser().resolve()
    specs = tuple(_validate_source_spec(spec) for spec in source_specs)
    _require(len(specs) == 6, "full-coverage partition requires six TRAIN sources")
    _require([row["split"] for row in specs] == [f"train{i}" for i in range(6)],
             "TRAIN sources must be ordered train0..train5")
    groups: list[AuthenticatedInformativeGroup] = []
    source_receipts: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for spec in specs:
        manifest_path = (base / str(spec["path"]) / "manifest.json").resolve()
        _require(manifest_path.is_relative_to(base),
                 f"TRAIN manifest escapes root: {manifest_path}")
        encoded, file_identity = _stable_file_bytes(manifest_path)
        _require(file_identity["sha256"] == str(spec["manifest_sha256"]),
                 f"{spec['split']} manifest SHA differs")
        manifest = json.loads(encoded.decode("utf-8"))
        _require(isinstance(manifest, dict),
                 f"{spec['split']} manifest root is not an object")
        _require(manifest.get("kind") == "loom_outcome_recovery_collection"
                 and int(manifest.get("format_version", -1)) == 1,
                 f"{spec['split']} manifest kind/version differs")
        _require(manifest.get("split") == spec["split"],
                 f"{spec['split']} manifest split differs")
        _require(manifest.get("identity_digest") == spec["identity_digest"],
                 f"{spec['split']} manifest identity differs")
        summary = dict(manifest.get("summary") or {})
        _require(summary.get("complete") is True
                 and summary.get("status") == "COMPLETE",
                 f"{spec['split']} collection is not complete")
        rows = list(manifest.get("groups") or ())
        _require(len(rows) == int(summary.get("n_groups", -1)) == 200,
                 f"{spec['split']} manifest does not contain 200 groups")
        informative_count = 0
        for group_index, raw in enumerate(rows):
            row = dict(raw)
            group_id = str(row.get("group_id") or "")
            _require(group_id and group_id not in seen_ids,
                     f"duplicate/empty global group_id {group_id!r}")
            seen_ids.add(group_id)
            rewards = tuple(float(value) for value in row.get("terminal_rewards") or ())
            informative = informative_terminal_rewards(rewards)
            if not informative:
                continue
            informative_count += 1
            sidecar_sha256 = str(row.get("sha256") or "")
            _require(_HEX64.fullmatch(sidecar_sha256) is not None,
                     f"invalid sidecar SHA for {group_id}")
            sidecar = str(row.get("sidecar") or "")
            _require(sidecar and not Path(sidecar).is_absolute()
                     and ".." not in Path(sidecar).parts,
                     f"unsafe sidecar path for {group_id}")
            _require(int(row.get("size", 0)) > 0,
                     f"nonpositive sidecar size for {group_id}")
            n_replans_by_arm = tuple(
                int(value) for value in row.get("n_replans_by_arm") or ()
            )
            _require(len(n_replans_by_arm) == recovery.GROUP_SIZE
                     and all(value > 0 for value in n_replans_by_arm),
                     f"invalid per-arm replan counts for {group_id}")
            groups.append(AuthenticatedInformativeGroup(
                split=str(spec["split"]),
                manifest_sha256=str(spec["manifest_sha256"]),
                identity_digest=str(spec["identity_digest"]),
                group_index=group_index,
                group_id=group_id,
                task_key=task_key_from_group_id(group_id),
                sidecar=sidecar,
                sidecar_sha256=sidecar_sha256,
                sidecar_size=int(row["size"]),
                n_replans_by_arm=n_replans_by_arm,
                terminal_rewards=rewards,
            ))
        _require(informative_count == int(spec["informative_groups"]),
                 f"{spec['split']} informative group count differs")
        source_receipts.append({
            **spec,
            "manifest_file_identity": file_identity,
            "manifest_groups": len(rows),
            "observed_informative_groups": informative_count,
        })
    _require(len(groups) == EXPECTED_INFORMATIVE_GROUPS,
             f"expected {EXPECTED_INFORMATIVE_GROUPS} informative groups, got {len(groups)}")
    _require({group.task_key for group in groups} == set(TASK_KEYS),
             "informative pool does not cover exactly 40 tasks")
    return tuple(groups), tuple(source_receipts)


def _task_order_sha256(task_key: str) -> str:
    return hashlib.sha256(
        f"{PARTITION_DOMAIN}|task|{task_key}".encode("utf-8")
    ).hexdigest()


def _group_order_sha256(group: AuthenticatedInformativeGroup) -> str:
    return hashlib.sha256(
        (
            f"{PARTITION_DOMAIN}|group|{group.split}|{group.group_id}|"
            f"{group.sidecar_sha256}"
        ).encode("utf-8")
    ).hexdigest()


def frozen_v3_visit0_replan_indices(
    group: AuthenticatedInformativeGroup,
) -> dict[int, tuple[int, ...]]:
    """Exact inherited RoundRobinOutcomeSamplerV3 seed-0 visit-0 contexts."""
    _require(re.fullmatch(r"train[0-5]", group.split) is not None,
             f"invalid TRAIN split for replan selection: {group.split!r}")
    fold = int(group.split.removeprefix("train"))
    selected: dict[int, tuple[int, ...]] = {}
    for arm in range(1, recovery.GROUP_SIZE):
        n_replans = int(group.n_replans_by_arm[arm])
        indices: list[int] = []
        # Visit zero begins at draw zero.  Retain the inherited epoch rollover
        # exactly even though every authenticated selected arm currently has
        # at least two replans.
        for draw in range(CONTEXTS_PER_ARM):
            epoch, position = divmod(draw, n_replans)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(v1._stable_seed(
                "outcome-v2-replan-v3",
                SAMPLER_SEED,
                fold,
                group.group_index,
                arm,
                epoch,
                group.identity_digest,
            ))
            order = torch.randperm(n_replans, generator=generator).tolist()
            indices.append(int(order[position]))
        selected[arm] = tuple(indices)
    return selected


def _context_row(row: TaskBalancedAssignment) -> dict[str, Any]:
    return {
        "split": row.group.split,
        "manifest_group_index": row.group.group_index,
        "group_id": row.group.group_id,
        "sidecar_sha256": row.group.sidecar_sha256,
        "identity_digest": row.group.identity_digest,
        "shard": row.shard,
        "rank": row.rank,
        "visit": 0,
        "replan_indices": {
            str(arm): list(indices)
            for arm, indices in frozen_v3_visit0_replan_indices(row.group).items()
        },
    }


def build_task_stratified_partition(
    groups: Sequence[AuthenticatedInformativeGroup],
) -> tuple[TaskBalancedAssignment, ...]:
    """Assign the complete pool once across three deterministic task shards."""
    values = tuple(groups)
    _require(len(values) == EXPECTED_INFORMATIVE_GROUPS,
             "partition input is not the complete 903-group pool")
    _require(len({group.identity_key for group in values}) == len(values),
             "partition input repeats a split/group index")
    _require(len({group.group_id for group in values}) == len(values),
             "partition input repeats a group_id")
    by_task: dict[str, list[AuthenticatedInformativeGroup]] = {
        task: [] for task in TASK_KEYS
    }
    for group in values:
        _require(group.task_key in by_task,
                 f"partition group has unknown task {group.task_key!r}")
        by_task[group.task_key].append(group)
    _require(all(by_task.values()), "one or more tasks have no informative groups")
    ordered_tasks = sorted(
        TASK_KEYS, key=lambda task: (_task_order_sha256(task), task),
    )
    assignments: list[TaskBalancedAssignment] = []
    for task_position, task_key in enumerate(ordered_tasks):
        ordered_groups = sorted(
            by_task[task_key],
            key=lambda group: (
                _group_order_sha256(group), group.split,
                group.group_index, group.group_id,
            ),
        )
        rotation = task_position % N_SHARDS
        rank = task_position % EXPECTED_WORLD_SIZE
        for within_position, group in enumerate(ordered_groups):
            assignments.append(TaskBalancedAssignment(
                group=group,
                task_position=task_position,
                within_task_position=within_position,
                shard=(within_position + rotation) % N_SHARDS,
                rank=rank,
                order_sha256=_group_order_sha256(group),
            ))
    result = tuple(sorted(
        assignments,
        key=lambda row: (
            row.shard, row.task_position, row.within_task_position,
            row.group.split, row.group.group_index,
        ),
    ))
    validate_task_stratified_partition(result)
    return result


def validate_task_stratified_partition(
    assignments: Sequence[TaskBalancedAssignment],
) -> dict[str, Any]:
    rows = tuple(assignments)
    _require(len(rows) == EXPECTED_INFORMATIVE_GROUPS,
             "partition does not contain exactly 903 assignments")
    _require(len({row.group.identity_key for row in rows}) == len(rows),
             "partition repeats a split/group index")
    _require(len({row.group.group_id for row in rows}) == len(rows),
             "partition repeats a group_id")
    _require({row.shard for row in rows} == set(range(N_SHARDS)),
             "partition does not contain exactly shards 0..2")
    task_counts: dict[str, list[int]] = {
        task: [0] * N_SHARDS for task in TASK_KEYS
    }
    task_ranks: dict[str, set[int]] = {task: set() for task in TASK_KEYS}
    for row in rows:
        _require(0 <= row.rank < EXPECTED_WORLD_SIZE,
                 f"invalid rank {row.rank}")
        task_counts[row.group.task_key][row.shard] += 1
        task_ranks[row.group.task_key].add(row.rank)
    _require(all(min(counts) > 0 and max(counts) - min(counts) <= 1
                 for counts in task_counts.values()),
             "per-task shard counts are empty or differ by more than one")
    _require(all(len(ranks) == 1 for ranks in task_ranks.values()),
             "one task is assigned to multiple ranks")
    rank_tasks = {
        rank: sorted(task for task, ranks in task_ranks.items() if rank in ranks)
        for rank in range(EXPECTED_WORLD_SIZE)
    }
    _require(all(len(tasks) == TASKS_PER_RANK for tasks in rank_tasks.values()),
             "task-to-rank assignment is not exactly five tasks per rank")
    shard_totals = [sum(row.shard == shard for row in rows)
                    for shard in range(N_SHARDS)]
    _require(shard_totals == [301, 301, 301],
             f"global shard totals differ from 301/301/301: {shard_totals}")
    return {
        "passed": True,
        "informative_groups": len(rows),
        "tasks": len(task_counts),
        "shard_totals": shard_totals,
        "minimum_groups_per_task_shard": min(
            min(counts) for counts in task_counts.values()
        ),
        "maximum_per_task_shard_count_difference": max(
            max(counts) - min(counts) for counts in task_counts.values()
        ),
        "tasks_per_rank": TASKS_PER_RANK,
    }


def partition_receipt(
    assignments: Sequence[TaskBalancedAssignment],
    source_receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return a compact receipt that uniquely binds all 903 assignments."""
    rows = tuple(assignments)
    validation = validate_task_stratified_partition(rows)
    assignment_rows = [row.row() for row in rows]
    context_rows = [_context_row(row) for row in rows]
    union_rows = sorted(
        (row.group.union_row() for row in rows),
        key=lambda row: (row["split"], row["group_index"], row["group_id"]),
    )
    task_rows: list[dict[str, Any]] = []
    ordered_tasks = sorted(
        TASK_KEYS, key=lambda task: (_task_order_sha256(task), task),
    )
    for task_position, task_key in enumerate(ordered_tasks):
        selected = [row for row in rows if row.group.task_key == task_key]
        task_rows.append({
            "task_key": task_key,
            "task_position": task_position,
            "rank": task_position % EXPECTED_WORLD_SIZE,
            "rotation": task_position % N_SHARDS,
            "total_groups": len(selected),
            "groups_by_shard": [
                sum(row.shard == shard for row in selected)
                for shard in range(N_SHARDS)
            ],
            "assignment_sha256": _rows_sha256([row.row() for row in selected]),
        })
    shard_rows: list[dict[str, Any]] = []
    for shard in range(N_SHARDS):
        selected = [row for row in rows if row.shard == shard]
        bytes_by_rank = [
            sum(row.group.sidecar_size for row in selected if row.rank == rank)
            for rank in range(EXPECTED_WORLD_SIZE)
        ]
        mean_bytes = sum(bytes_by_rank) / float(EXPECTED_WORLD_SIZE)
        shard_rows.append({
            "shard": shard,
            "macro_offset": shard,
            "global_step": START_STEP + shard,
            "groups": len(selected),
            "tasks": len({row.group.task_key for row in selected}),
            "groups_by_rank": [
                sum(row.rank == rank for row in selected)
                for rank in range(EXPECTED_WORLD_SIZE)
            ],
            "sidecar_bytes": sum(bytes_by_rank),
            "sidecar_bytes_by_rank": bytes_by_rank,
            "maximum_rank_sidecar_bytes": max(bytes_by_rank),
            "maximum_to_mean_sidecar_bytes": max(bytes_by_rank) / mean_bytes,
            "assignment_sha256": _rows_sha256([row.row() for row in selected]),
            "context_assignment_sha256": _rows_sha256([
                _context_row(row) for row in selected
            ]),
        })
    rank_rows = []
    for rank in range(EXPECTED_WORLD_SIZE):
        selected = [row for row in rows if row.rank == rank]
        rank_rows.append({
            "rank": rank,
            "tasks": sorted({row.group.task_key for row in selected}),
            "groups_by_shard": [
                sum(row.shard == shard for row in selected)
                for shard in range(N_SHARDS)
            ],
            "sidecar_bytes_by_shard": [
                sum(row.group.sidecar_size for row in selected if row.shard == shard)
                for shard in range(N_SHARDS)
            ],
            "local_group_weight_sum_per_shard": [1.0 / EXPECTED_WORLD_SIZE] * N_SHARDS,
        })
    source_rows = []
    for receipt in source_receipts:
        row = dict(receipt)
        row.pop("manifest_file_identity", None)
        source_rows.append(row)
    receipt = {
        "format_version": FORMAT_VERSION,
        "kind": PARTITION_KIND,
        "method_status": "FROZEN_SELECTION_NONLAUNCHABLE",
        "domain": PARTITION_DOMAIN,
        "scheme": PARTITION_SCHEME,
        "sources": source_rows,
        "pool": {
            "raw_groups": 1200,
            "informative_definition": (
                "inherited_int_distinct_over_8_exact_binary_terminal_rewards"
            ),
            "informative_groups": EXPECTED_INFORMATIVE_GROUPS,
            "tasks": TASK_COUNT,
            "all_groups_used_exactly_once": True,
            "union_sha256": _rows_sha256(union_rows),
            "sidecar_bytes": sum(row.group.sidecar_size for row in rows),
        },
        "partition": {
            "shards": N_SHARDS,
            "assignment_sha256": _rows_sha256(assignment_rows),
            "context_selection": {
                "sampler_kind": v2.SAMPLER_KIND,
                "stable_seed_domain": "outcome-v2-replan-v3",
                "sampler_seed": SAMPLER_SEED,
                "visit": 0,
                "contexts_per_arm": CONTEXTS_PER_ARM,
                "arms": list(range(1, recovery.GROUP_SIZE)),
                "atoms": len(rows) * (recovery.GROUP_SIZE - 1) * CONTEXTS_PER_ARM,
                "assignment_sha256": _rows_sha256(context_rows),
                "prior_round_robin_24_draw_equivalence_required": True,
            },
            "task_order_sha256": _rows_sha256([
                {"task_position": index, "task_key": task}
                for index, task in enumerate(ordered_tasks)
            ]),
            "task_rows": task_rows,
            "shard_rows": shard_rows,
            "rank_rows": rank_rows,
            "validation": validation,
        },
        "aggregation": {
            "per_group": "equal_context_within_arm_then_equal_7_arms",
            "contexts_per_arm": CONTEXTS_PER_ARM,
            "macro": "equal_group_within_task_then_equal_40_tasks",
            "distributed": (
                "production_SUM_over_world_mean_of_world_scaled_rank_local_"
                "contributions_equals_explicit_SUM"
            ),
            "tasks_per_rank": TASKS_PER_RANK,
            "group_local_weight": "1/(40*groups_for_task_in_shard)",
            "rank_local_task_weight_sum": "5/40=1/8",
            "production_mean_reducer_prescale": EXPECTED_WORLD_SIZE,
            "weight_application": "direct_group_scalar_before_autograd",
            "execution": (
                "fixed_order_one_group_graph_then_detached_gradient_"
                "accumulation_demo_once"
            ),
            "scope": "equal_group_within_task_then_equal_40_tasks_per_macro",
            "three_macro_average_not_equal_to_one_shot_full_pool_group_mean": True,
            "partition_interpretation": (
                "prospective_design_unbiased_deterministic_balanced_"
                "three_minibatch_one_pass"
            ),
        },
        "resume": {
            "start_step": START_STEP,
            "stop_step": STOP_STEP,
            "actual_macro_updates": MACRO_UPDATES,
            "one_full_903_group_pass": True,
        },
    }
    return receipt


def validate_partition_receipt(
    assignments: Sequence[TaskBalancedAssignment],
    source_receipts: Sequence[Mapping[str, Any]],
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    observed = partition_receipt(assignments, source_receipts)
    _require(observed == dict(expected), "frozen task-balanced selection receipt differs")
    return {"passed": True, "receipt_sha256": canonical_sha256(observed)}


def assignments_for(
    assignments: Sequence[TaskBalancedAssignment],
    *,
    shard: int,
    rank: int | None = None,
) -> tuple[TaskBalancedAssignment, ...]:
    if shard not in range(N_SHARDS):
        raise ValueError(f"shard must be in [0,{N_SHARDS}), got {shard}")
    if rank is not None and rank not in range(EXPECTED_WORLD_SIZE):
        raise ValueError(f"rank must be in [0,{EXPECTED_WORLD_SIZE}), got {rank}")
    return tuple(row for row in assignments
                 if row.shard == shard and (rank is None or row.rank == rank))


def local_group_weights(
    assignments: Sequence[TaskBalancedAssignment],
    *,
    shard: int,
    rank: int,
) -> dict[tuple[str, int], float]:
    """Weights for SUM-reduced gradients; each rank contributes exactly 1/8."""
    selected = assignments_for(assignments, shard=shard, rank=rank)
    task_keys = {row.group.task_key for row in selected}
    _require(len(task_keys) == TASKS_PER_RANK,
             f"rank {rank} shard {shard} does not contain five tasks")
    counts = {
        task: sum(row.group.task_key == task for row in selected)
        for task in task_keys
    }
    result = {
        row.group.identity_key: 1.0 / (TASK_COUNT * counts[row.group.task_key])
        for row in selected
    }
    _require(math.isclose(sum(result.values()), 1.0 / EXPECTED_WORLD_SIZE,
                          rel_tol=0.0, abs_tol=1e-15),
             "rank-local group weights do not sum to 1/8")
    return result


def rank_task_keys(
    assignments: Sequence[TaskBalancedAssignment], rank: int,
) -> tuple[str, ...]:
    if rank not in range(EXPECTED_WORLD_SIZE):
        raise ValueError(f"rank must be in [0,{EXPECTED_WORLD_SIZE}), got {rank}")
    tasks = sorted({row.group.task_key for row in assignments if row.rank == rank})
    _require(len(tasks) == TASKS_PER_RANK,
             f"rank {rank} does not own exactly five tasks")
    return tuple(tasks)


@dataclass(frozen=True)
class StreamingTaskGroup:
    assignment: TaskBalancedAssignment
    mathematical_sum_weight: float
    production_mean_reducer_weight: float


def streaming_group_rows(
    assignments: Sequence[TaskBalancedAssignment],
    *,
    shard: int,
    rank: int,
) -> tuple[StreamingTaskGroup, ...]:
    """Exact fixed microbatch order/weights; callers release each graph."""
    validate_task_stratified_partition(assignments)
    selected = assignments_for(assignments, shard=shard, rank=rank)
    weights = local_group_weights(assignments, shard=shard, rank=rank)
    return tuple(StreamingTaskGroup(
        assignment=row,
        mathematical_sum_weight=weights[row.group.identity_key],
        production_mean_reducer_weight=(
            weights[row.group.identity_key] * EXPECTED_WORLD_SIZE
        ),
    ) for row in selected)


def production_weighted_group_scalar(
    item: StreamingTaskGroup,
    direct_group_loss: Tensor,
) -> Tensor:
    """Apply world/(40*n_task,shard) before autograd for frozen geometry."""
    if not isinstance(direct_group_loss, Tensor) or direct_group_loss.numel() != 1:
        raise ValueError("direct group loss must be one scalar tensor")
    _require(bool(torch.isfinite(direct_group_loss)),
             "direct group loss contains nan/inf")
    return direct_group_loss.reshape(()) * item.production_mean_reducer_weight


class StreamingMacroGradientAccumulator:
    """Detached fixed-order accumulator; it never retains an autograd graph."""

    def __init__(
        self,
        assignments: Sequence[TaskBalancedAssignment],
        *,
        shard: int,
        rank: int,
    ) -> None:
        self.shard = int(shard)
        self.rank = int(rank)
        self.rows = streaming_group_rows(assignments, shard=shard, rank=rank)
        self._next = 0
        self._buffer: Tensor | None = None
        self._demo_added = False

    def add_weighted_group_gradient(
        self, identity_key: tuple[str, int], gradient: Tensor,
    ) -> None:
        _require(self._next < len(self.rows),
                 "streaming accumulator received too many group gradients")
        expected = self.rows[self._next]
        _require(identity_key == expected.assignment.group.identity_key,
                 "streaming group gradient is missing, duplicated, or out of order")
        _require(isinstance(gradient, Tensor) and gradient.ndim == 1
                 and gradient.numel() > 0 and bool(torch.isfinite(gradient).all()),
                 "streaming group gradient must be one finite flat vector")
        # Weighting is intentionally applied to the direct scalar before
        # autograd via production_weighted_group_scalar.  That preserves the
        # frozen bf16 backward geometry while this accumulator owns only the
        # detached, already-weighted vector.
        value = gradient.detach().clone()
        if self._buffer is None:
            self._buffer = value
        else:
            _require(value.shape == self._buffer.shape
                     and value.device == self._buffer.device
                     and value.dtype == self._buffer.dtype,
                     "streaming group gradient vector geometry changed")
            self._buffer.add_(value)
        self._next += 1

    def add_demo_gradient_once(self, gradient: Tensor) -> None:
        _require(self._next == len(self.rows),
                 "demo gradient must follow every rank-local group")
        _require(not self._demo_added, "demo gradient was added more than once")
        _require(isinstance(gradient, Tensor) and gradient.ndim == 1
                 and gradient.numel() > 0 and bool(torch.isfinite(gradient).all()),
                 "demo gradient must be one finite flat vector")
        value = gradient.detach().clone()
        if self._buffer is None:
            self._buffer = torch.zeros_like(value)
        _require(value.shape == self._buffer.shape
                 and value.device == self._buffer.device
                 and value.dtype == self._buffer.dtype,
                 "demo gradient vector geometry differs from group gradients")
        self._buffer.add_(value)
        self._demo_added = True

    def finalize(self, *, require_demo: bool) -> tuple[Tensor, dict[str, Any]]:
        _require(self._next == len(self.rows),
                 "streaming macro ended before every group was consumed")
        _require(self._demo_added == bool(require_demo),
                 "streaming macro demo-reference cardinality differs")
        _require(self._buffer is not None,
                 "streaming macro has no accumulated gradient")
        receipt = {
            "shard": self.shard,
            "rank": self.rank,
            "groups": len(self.rows),
            "groups_consumed": self._next,
            "fixed_order_complete": True,
            "demo_gradients_added": int(self._demo_added),
            "mathematical_group_weight_sum": sum(
                row.mathematical_sum_weight for row in self.rows
            ),
            "production_group_weight_sum": sum(
                row.production_mean_reducer_weight for row in self.rows
            ),
            "weight_application": "direct_scalar_before_autograd",
        }
        return self._buffer.detach().clone(), receipt


@dataclass(frozen=True)
class TaskGroupComponents:
    positive_advantage: Tensor
    recovery_k3: Tensor
    metrics: dict[str, float]


def sampled_task_group_components(
    proposal: nn.Module,
    payload: Mapping[str, Any],
    replan_indices: Mapping[int, Sequence[int]],
    *,
    device: torch.device,
    require_recovery_identity: bool,
) -> TaskGroupComponents:
    """Two-pass exact B=1 group scorer: PA then recovery k3, arms 1..7."""
    positive = pa.sampled_positive_advantage_objective(
        proposal, payload, replan_indices, device=device,
    )
    expected = set(range(1, recovery.GROUP_SIZE))
    _require(set(replan_indices) == expected,
             "task-balanced recovery requires exactly arms 1..7")
    _require(all(len(tuple(replan_indices[arm])) == CONTEXTS_PER_ARM
                 for arm in sorted(expected)),
             "task-balanced method requires exactly two contexts per scored arm")
    arms = list(payload.get("arms") or ())
    _require(len(arms) == recovery.GROUP_SIZE,
             "task-balanced group does not contain exactly eight arms")
    try:
        dtype = next(proposal.parameters()).dtype
    except StopIteration as exc:
        raise ValueError("proposal has no parameters") from exc
    arm_terms: list[Tensor] = []
    ratio_rows: list[Tensor] = []
    delta_rows: list[Tensor] = []
    for arm_index in range(1, recovery.GROUP_SIZE):
        arm = arms[arm_index]
        index = pa._selected_indices(
            replan_indices, arm_index, int(arm["z"].shape[0]),
        )
        z = arm["z"].index_select(0, index).to(
            device=device, dtype=dtype, non_blocking=True,
        )
        order = arm["ordered_support"].index_select(0, index).to(device=device)
        old = arm["old_logprob"].detach().index_select(0, index).to(
            device=device, dtype=torch.float32,
        )
        lang = v1._batched_lang(arm["lang"], int(index.numel()), device, dtype)
        current, _logits = v1.stored_order_logprob(proposal, z, lang, order)
        with torch.autocast(device_type=device.type, enabled=False):
            delta = current.float() - old
            atom = torch.expm1(delta) - delta
        _require(bool(torch.isfinite(current).all()) and bool(torch.isfinite(old).all()),
                 f"task-balanced recovery scorer is nonfinite in arm {arm_index}")
        if require_recovery_identity:
            _require(torch.equal(current.float(), old),
                     "task-balanced recovery identity current.float()!=stored old")
        _require(bool(torch.isfinite(atom).all()),
                 f"task-balanced recovery k3 is nonfinite in arm {arm_index}")
        arm_terms.append(atom.mean())
        ratio_rows.append(torch.exp(delta.detach()).reshape(-1))
        delta_rows.append(delta.detach().reshape(-1))
    recovery_k3 = torch.stack(arm_terms).mean()
    ratios = torch.cat(ratio_rows)
    deltas = torch.cat(delta_rows)
    _require(bool(torch.isfinite(recovery_k3)) and bool(torch.isfinite(ratios).all()),
             "task-balanced recovery result is nonfinite")
    return TaskGroupComponents(
        positive_advantage=positive.loss,
        recovery_k3=recovery_k3,
        metrics={
            **positive.metrics,
            "recovery_k3": float(recovery_k3.detach()),
            "recovery_ratio_min": float(ratios.min()),
            "recovery_ratio_mean": float(ratios.double().mean()),
            "recovery_ratio_max": float(ratios.max()),
            "recovery_max_abs_current_minus_old": float(deltas.abs().max()),
            "recovery_identity_exact": float(bool(torch.equal(
                deltas, torch.zeros_like(deltas),
            ))),
            "recovery_identity_required": float(require_recovery_identity),
            "recovery_atoms": float(deltas.numel()),
            "recovery_arms": float(len(arm_terms)),
            "arm0_in_recovery": 0.0,
            "scorer_passes": 2.0,
        },
    )


@dataclass(frozen=True)
class ThreeMacroUpdateSchedule:
    partition_receipt_sha256: str
    recipe_file_sha256: str
    core_closure_sha256: str
    seed_checkpoint_sha256: str
    start_step: int = START_STEP
    macro_updates: int = MACRO_UPDATES

    def __post_init__(self) -> None:
        for label, value in (
            ("partition receipt", self.partition_receipt_sha256),
            ("recipe file", self.recipe_file_sha256),
            ("core closure", self.core_closure_sha256),
            ("seed checkpoint", self.seed_checkpoint_sha256),
        ):
            _require(_HEX64.fullmatch(str(value)) is not None,
                     f"schedule {label} SHA must be 64 lowercase hex characters")
        _require(int(self.start_step) == START_STEP,
                 f"macro schedule must begin at {START_STEP}")
        _require(int(self.macro_updates) == MACRO_UPDATES,
                 "terminal pilot must contain exactly three macro updates")

    @property
    def stop_step(self) -> int:
        return self.start_step + self.macro_updates

    def shard_at(self, global_step: int) -> int:
        offset = int(global_step) - self.start_step
        if offset not in range(self.macro_updates):
            raise ValueError(
                f"global_step {global_step} is outside "
                f"[{self.start_step},{self.stop_step})"
            )
        return offset

    def state_dict(self, next_global_step: int) -> dict[str, Any]:
        step = int(next_global_step)
        if step < self.start_step or step > self.stop_step:
            raise ValueError(
                f"next_global_step {step} is outside "
                f"[{self.start_step},{self.stop_step}]"
            )
        return {
            "format_version": FORMAT_VERSION,
            "kind": "loom_task_balanced_pa_three_macro_schedule",
            "partition_receipt_sha256": self.partition_receipt_sha256,
            "recipe_file_sha256": self.recipe_file_sha256,
            "core_closure_sha256": self.core_closure_sha256,
            "seed_checkpoint_sha256": self.seed_checkpoint_sha256,
            "start_step": self.start_step,
            "stop_step": self.stop_step,
            "macro_updates": self.macro_updates,
            "next_global_step": step,
            "completed_macro_updates": step - self.start_step,
            "optimizer_steps_completed": step - self.start_step,
            "post_update_trust_dev_gates_completed": step - self.start_step,
            "completed_shards": list(range(step - self.start_step)),
            "remaining_shards": list(range(step - self.start_step, N_SHARDS)),
            "informative_groups_consumed": 301 * (step - self.start_step),
            "next_shard": (None if step == self.stop_step else self.shard_at(step)),
            "one_full_903_group_pass_complete": step == self.stop_step,
        }

    def validate_state_dict(
        self, state: Mapping[str, Any], next_global_step: int,
    ) -> None:
        _require(dict(state) == self.state_dict(next_global_step),
                 "task-balanced macro resume state differs")


def core_source_identity(root: str | Path = _ROOT) -> dict[str, Any]:
    return v1._trainer_source_identity(root=root, files=_CORE_SOURCE_FILES)


def assert_core_source_identity(
    expected: Mapping[str, Any], *, root: str | Path = _ROOT,
) -> None:
    v1._assert_trainer_source_identity(
        expected, root=root, files=_CORE_SOURCE_FILES,
    )


def core_provenance(
    source_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source = core_source_identity() if source_identity is None else dict(source_identity)
    assert_core_source_identity(source)
    return {
        "format_version": FORMAT_VERSION,
        "kind": CORE_KIND,
        "source_identity": source,
        "execution_surface": "mathematical_partition_weighting_resume_core_only",
        "trainer_wired": False,
        "launcher_present": False,
        "candidate_or_evaluation_authority": False,
        "partition": {
            "domain": PARTITION_DOMAIN,
            "scheme": PARTITION_SCHEME,
            "informative_groups": EXPECTED_INFORMATIVE_GROUPS,
            "tasks": TASK_COUNT,
            "shards": N_SHARDS,
            "contexts_per_arm": CONTEXTS_PER_ARM,
            "all_groups_once": True,
            "replan_selection": (
                "inherited_round_robin_v3_seed0_visit0_exact_two_per_arm"
            ),
        },
        "objective": {
            "positive_advantage": 1.0,
            "task_balanced_recovery_k3": 1.0,
            "analytic_demo_reference": 1.0,
            "grpo": 0.0,
            "switch_balance": 0.0,
            "sparse_ce": 0.0,
            "aggregation": (
                "equal_context_within_arm_then_equal_7_arms_then_"
                "equal_group_within_task_then_equal_40_tasks"
            ),
            "distributed_gradient_reduce": (
                "production_SUM_over_world_mean_of_world_scaled_rank_local_"
                "1_over_40_task_contributions_equals_explicit_SUM"
            ),
            "execution": (
                "fixed_order_one_group_at_a_time_weight_direct_scalar_before_"
                "autograd_then_detached_gradient_accumulation_demo_once_"
                "no_collectives_in_unequal_loops"
            ),
            "scope": "equal_group_within_task_and_equal_40_tasks_per_macro",
            "three_macro_caveat": (
                "arithmetic_mean_not_one_shot_full_pool_equal_group_objective_"
                "because_per_task_shard_counts_can_differ_by_one"
            ),
            "partition_interpretation": (
                "prospective_design_unbiased_deterministic_balanced_"
                "three_minibatch_one_pass"
            ),
        },
        "maximum_pass_authority": (
            "separate_3_actual_macro_update_ineligible_pilot_"
            "one_full_903_group_pass_only"
        ),
    }


__all__ = [
    "FORMAT_VERSION", "CORE_KIND", "PARTITION_KIND", "PARTITION_DOMAIN",
    "PARTITION_SCHEME", "N_SHARDS", "EXPECTED_WORLD_SIZE", "CONTEXTS_PER_ARM",
    "SAMPLER_SEED", "START_STEP",
    "MACRO_UPDATES", "STOP_STEP", "EXPECTED_INFORMATIVE_GROUPS", "SUITES",
    "TASK_KEYS", "TASK_COUNT", "TASKS_PER_RANK",
    "TaskBalancedPositiveAdvantageError", "AuthenticatedInformativeGroup",
    "TaskBalancedAssignment", "TaskGroupComponents", "StreamingTaskGroup",
    "StreamingMacroGradientAccumulator", "ThreeMacroUpdateSchedule",
    "canonical_sha256", "task_key_from_group_id",
    "informative_terminal_rewards", "load_authenticated_informative_groups",
    "frozen_v3_visit0_replan_indices",
    "build_task_stratified_partition", "validate_task_stratified_partition",
    "partition_receipt", "validate_partition_receipt", "assignments_for",
    "local_group_weights", "rank_task_keys", "streaming_group_rows",
    "production_weighted_group_scalar",
    "sampled_task_group_components",
    "core_source_identity", "assert_core_source_identity", "core_provenance",
]
