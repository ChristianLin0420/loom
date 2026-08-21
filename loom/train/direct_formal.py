"""Frozen schedule and convergence primitives for the direct-formal run.

This module deliberately has no training-loop, logging, or orchestration side
effects.  The loop supplies an update index to :class:`DirectFormalSchedule`
and a metrics prefix to :func:`evaluate_direct_formal`; the returned receipt is
the complete decision.  Re-evaluating a longer prefix therefore cannot move a
previously selected first-passing checkpoint.

The optimization recipe has two different horizons:

* ``schedule_horizon`` is the end of cosine decay (32,000 updates), and
* ``max_updates`` is the hard execution cap (40,000 updates).

Updates after ``schedule_horizon`` execute at the exact configured LR floor.
The statistical gate first evaluates update 32,000, then every 500 updates.  At
each candidate it uses four non-overlapping 2,000-update blocks, exact 2%
plateau checks, a fixed health gate, and 2% non-regression against the immutable
``(30,000, 32,000]`` reference block.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "DIRECT_FORMAL_FORMAT",
    "DirectFormalGate",
    "DirectFormalSchedule",
    "evaluate_direct_formal",
    "next_direct_formal_check",
    "receipt_exit_code",
    "should_evaluate_direct_formal",
]


DIRECT_FORMAL_FORMAT = "loom-direct-formal-convergence-v1"
_STATUS_EXIT_CODES = {"PASS": 0, "MOVING": 1, "INVALID": 2, "ABORT": 3}


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_number(value: object, label: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(float(value))):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


@dataclass(frozen=True)
class DirectFormalSchedule:
    """Linear warmup/cosine schedule with a separate hard update cap.

    ``step`` is the zero-based update index, matching ``CosineWithWarmup`` in
    :mod:`loom.train.schedule`.  Thus the first extension update has
    ``step == schedule_horizon`` and receives the exact minimum LR.

    The schedule is stateless.  Its checkpoint payload is an identity receipt;
    loading a different recipe fails instead of silently changing either
    horizon on resume.
    """

    base_lr: float
    warmup_steps: int
    schedule_horizon: int = 32_000
    max_updates: int = 40_000
    min_lr_ratio: float = 0.05

    def __post_init__(self) -> None:
        base_lr = _finite_number(self.base_lr, "base_lr")
        min_lr_ratio = _finite_number(self.min_lr_ratio, "min_lr_ratio")
        if base_lr <= 0.0:
            raise ValueError("base_lr must be > 0")
        if not _is_int(self.warmup_steps) or self.warmup_steps < 0:
            raise ValueError("warmup_steps must be a non-negative integer")
        if not _is_int(self.schedule_horizon) or self.schedule_horizon <= 0:
            raise ValueError("schedule_horizon must be a positive integer")
        if not _is_int(self.max_updates) or self.max_updates < self.schedule_horizon:
            raise ValueError("max_updates must be an integer >= schedule_horizon")
        if self.warmup_steps > self.schedule_horizon:
            raise ValueError("warmup_steps must not exceed schedule_horizon")
        if not 0.0 <= min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1]")

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "DirectFormalSchedule":
        """Construct from the explicit new config keys, with no legacy fallback.

        Required keys are ``run.schedule_horizon``, ``run.max_updates``, and
        ``optim.lr``.  Refusing to infer either horizon from ``run.steps`` is
        what keeps an extension from stretching the cosine decay.
        """

        if not isinstance(cfg, Mapping):
            raise ValueError("config must be a mapping")
        run = cfg.get("run")
        optim = cfg.get("optim")
        if not isinstance(run, Mapping):
            raise ValueError("config.run must be a mapping")
        if not isinstance(optim, Mapping):
            raise ValueError("config.optim must be a mapping")
        missing = [
            key for key, owner in (
                ("run.schedule_horizon", run),
                ("run.max_updates", run),
                ("optim.lr", optim),
            )
            if key.rsplit(".", 1)[1] not in owner
        ]
        if missing:
            raise ValueError("missing direct-formal config keys: " + ", ".join(missing))
        return cls(
            base_lr=optim["lr"],
            warmup_steps=optim.get("warmup", 2_000),
            schedule_horizon=run["schedule_horizon"],
            max_updates=run["max_updates"],
            min_lr_ratio=optim.get("min_lr_ratio", 0.05),
        )

    @staticmethod
    def _step(step: object, *, for_update: bool, max_updates: int) -> int:
        if not _is_int(step) or step < 0:
            raise ValueError("step must be a non-negative integer")
        if for_update and step >= max_updates:
            raise ValueError(
                f"update index {step} is outside max_updates={max_updates}"
            )
        return step

    def scale_at(self, step: int) -> float:
        step = self._step(step, for_update=False, max_updates=self.max_updates)
        if step < self.warmup_steps:
            return (step + 1) / max(1, self.warmup_steps)
        if step >= self.schedule_horizon:
            return float(self.min_lr_ratio)
        progress = (step - self.warmup_steps) / max(
            1, self.schedule_horizon - self.warmup_steps
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def lr_at(self, step: int) -> float:
        return float(self.base_lr) * self.scale_at(step)

    def apply(self, optimizer: Any, step: int) -> dict[str, float]:
        """Apply this update's LR, preserving each optimizer group multiplier."""

        step = self._step(step, for_update=True, max_updates=self.max_updates)
        scale = self.scale_at(step)
        out: dict[str, float] = {}
        for group in optimizer.param_groups:
            lr = float(self.base_lr) * float(group.get("lr_scale", 1.0)) * scale
            group["lr"] = lr
            out[str(group.get("name", "group"))] = lr
        return out

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": "loom-direct-formal-schedule-v1",
            "base_lr": float(self.base_lr),
            "warmup_steps": self.warmup_steps,
            "schedule_horizon": self.schedule_horizon,
            "max_updates": self.max_updates,
            "min_lr_ratio": float(self.min_lr_ratio),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Authenticate a checkpoint recipe; this stateless object never mutates."""

        if not isinstance(state, Mapping):
            raise ValueError("direct-formal schedule state must be a mapping")
        expected = self.state_dict()
        missing = sorted(set(expected) - set(state))
        unknown = sorted(set(state) - set(expected))
        if missing or unknown:
            raise ValueError(
                "direct-formal schedule state keys mismatch; "
                f"missing={missing}, unknown={unknown}"
            )
        got = dict(state)
        if got != expected:
            differences = sorted(key for key in expected if got[key] != expected[key])
            raise ValueError(
                "direct-formal schedule identity mismatch: " + ", ".join(differences)
            )


@dataclass(frozen=True)
class DirectFormalGate:
    """Prospectively fixed direct-formal stopping contract."""

    schedule_horizon: int = 32_000
    max_updates: int = 40_000
    first_check: int = 32_000
    check_every: int = 500
    block_size: int = 2_000
    block_count: int = 4
    tolerance: float = 0.02
    reference_start_exclusive: int = 30_000
    reference_end_inclusive: int = 32_000
    primary_metrics: tuple[str, ...] = (
        "loss/dyn", "act/decode_teacher", "act/decode_deploy", "act/align",
        "loss/proposal",
    )
    delta_op_strict_gt: float = 0.01
    delta_sel_strict_gt: float = 0.0
    act_align_strict_lt: float = 0.50
    live_ops_q_a_gte: float = 16.0
    live_ops_q_delta_gte: float = 16.0
    proposal_uniform_ce: float = math.log(128.0)
    proposal_floor_margin: float = 0.05
    c_delta_spread_strict_gt: float = 0.10
    gnorm_bank_strict_gt: float = 1.0e-4
    gnorm_q_delta_strict_gt: float = 1.0e-4
    skipped_rate_strict_lt: float = 0.01
    expected_gradient_modules: tuple[str, ...] = (
        "estimator", "bank", "q_delta", "q_action", "decoder", "proposal",
    )

    def __post_init__(self) -> None:
        positive_ints = {
            "schedule_horizon": self.schedule_horizon,
            "max_updates": self.max_updates,
            "first_check": self.first_check,
            "check_every": self.check_every,
            "block_size": self.block_size,
            "block_count": self.block_count,
            "reference_end_inclusive": self.reference_end_inclusive,
        }
        for name, value in positive_ints.items():
            if not _is_int(value) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (not _is_int(self.reference_start_exclusive) or
                self.reference_start_exclusive < 0):
            raise ValueError("reference_start_exclusive must be a non-negative integer")
        if self.first_check != self.schedule_horizon:
            raise ValueError("first_check must equal schedule_horizon")
        if self.max_updates < self.first_check:
            raise ValueError("max_updates must be >= first_check")
        if (self.max_updates - self.first_check) % self.check_every:
            raise ValueError("max_updates must lie on the fixed check cadence")
        if self.first_check < self.block_size * self.block_count:
            raise ValueError("first_check cannot supply the requested complete blocks")
        if self.reference_end_inclusive != self.first_check:
            raise ValueError("the immutable reference must end at first_check")
        if (self.reference_end_inclusive - self.reference_start_exclusive !=
                self.block_size):
            raise ValueError("the immutable reference must contain one exact block")
        if (not self.primary_metrics or len(set(self.primary_metrics)) !=
                len(self.primary_metrics) or
                not all(isinstance(name, str) and name for name in self.primary_metrics)):
            raise ValueError("primary_metrics must be unique non-empty strings")
        if (not self.expected_gradient_modules or
                len(set(self.expected_gradient_modules)) !=
                len(self.expected_gradient_modules) or
                not all(isinstance(name, str) and name
                        for name in self.expected_gradient_modules)):
            raise ValueError(
                "expected_gradient_modules must be unique non-empty strings"
            )
        finite_fields = (
            "tolerance", "delta_op_strict_gt", "delta_sel_strict_gt",
            "act_align_strict_lt", "live_ops_q_a_gte", "live_ops_q_delta_gte",
            "proposal_uniform_ce", "proposal_floor_margin",
            "c_delta_spread_strict_gt", "gnorm_bank_strict_gt",
            "gnorm_q_delta_strict_gt", "skipped_rate_strict_lt",
        )
        for name in finite_fields:
            _finite_number(getattr(self, name), name)
        if not 0.0 <= self.tolerance < 1.0:
            raise ValueError("tolerance must be in [0, 1)")
        if self.proposal_floor_margin < 0.0:
            raise ValueError("proposal_floor_margin must be >= 0")
        if not 0.0 <= self.skipped_rate_strict_lt <= 1.0:
            raise ValueError("skipped_rate_strict_lt must be in [0, 1]")

    @property
    def proposal_off_floor_strict_lt(self) -> float:
        return self.proposal_uniform_ce - self.proposal_floor_margin

    def as_dict(self) -> dict[str, Any]:
        return {
            "schedule_horizon": self.schedule_horizon,
            "max_updates": self.max_updates,
            "first_check": self.first_check,
            "check_every": self.check_every,
            "blocks": {"count": self.block_count, "size": self.block_size},
            "tolerance": float(self.tolerance),
            "reference_window": {
                "start_exclusive": self.reference_start_exclusive,
                "end_inclusive": self.reference_end_inclusive,
            },
            "primary_metrics": list(self.primary_metrics),
            "health_thresholds": {
                "delta_op_median_strict_gt": float(self.delta_op_strict_gt),
                "delta_sel_h1_to_h4_medians_strict_gt": float(
                    self.delta_sel_strict_gt
                ),
                "act_align_median_strict_lt": float(self.act_align_strict_lt),
                "live_ops_q_a_median_gte": float(self.live_ops_q_a_gte),
                "live_ops_q_delta_median_gte": float(self.live_ops_q_delta_gte),
                "proposal_ce_median_strict_lt": float(self.proposal_uniform_ce),
                "proposal_off_floor_median_strict_lt": float(
                    self.proposal_off_floor_strict_lt
                ),
                "c_delta_spread_median_strict_gt": float(
                    self.c_delta_spread_strict_gt
                ),
                "gnorm_bank_median_strict_gt": float(self.gnorm_bank_strict_gt),
                "gnorm_q_delta_median_strict_gt": float(
                    self.gnorm_q_delta_strict_gt
                ),
                "all_other_expected_gnorm_medians_strict_gt": 0.0,
                "skipped_rate_strict_lt": float(self.skipped_rate_strict_lt),
                "unexpected_module_gradients": False,
                "nonfinite": False,
            },
            "expected_gradient_modules": list(self.expected_gradient_modules),
        }


def should_evaluate_direct_formal(
    completed_updates: int, gate: DirectFormalGate | None = None,
) -> bool:
    """Return whether ``completed_updates`` is a predeclared decision boundary."""

    gate = gate or DirectFormalGate()
    if not _is_int(completed_updates) or completed_updates < 0:
        raise ValueError("completed_updates must be a non-negative integer")
    return (
        gate.first_check <= completed_updates <= gate.max_updates and
        (completed_updates - gate.first_check) % gate.check_every == 0
    )


def next_direct_formal_check(
    completed_updates: int, gate: DirectFormalGate | None = None,
) -> int | None:
    """Return the next fixed check strictly after ``completed_updates``."""

    gate = gate or DirectFormalGate()
    if not _is_int(completed_updates) or completed_updates < 0:
        raise ValueError("completed_updates must be a non-negative integer")
    if completed_updates < gate.first_check:
        return gate.first_check
    offset = completed_updates - gate.first_check
    candidate = gate.first_check + (offset // gate.check_every + 1) * gate.check_every
    return candidate if candidate <= gate.max_updates else None


class _InvalidMetrics(ValueError):
    pass


def _exact_window(
    by_step: Mapping[int, Mapping[str, Any]], start: int, end: int, label: str,
) -> list[Mapping[str, Any]]:
    missing = [step for step in range(start + 1, end + 1) if step not in by_step]
    if missing:
        preview = missing[:8]
        suffix = "..." if len(missing) > len(preview) else ""
        raise _InvalidMetrics(
            f"{label} is not exact/contiguous for ({start}, {end}]; "
            f"missing steps={preview}{suffix}"
        )
    return [by_step[step] for step in range(start + 1, end + 1)]


def _metric_values(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for index, row in enumerate(rows):
        if key not in row:
            raise _InvalidMetrics(f"metric {key!r} is missing from window row {index}")
        try:
            values.append(_finite_number(row[key], f"metric {key!r}"))
        except ValueError as error:
            raise _InvalidMetrics(str(error)) from error
    return values


def _require_finite_numeric(
    rows: Sequence[Mapping[str, Any]], label: str,
) -> None:
    for row in rows:
        step = row["global_step"]
        for key, value in row.items():
            if (not isinstance(value, bool) and isinstance(value, (int, float))
                    and not math.isfinite(float(value))):
                raise _InvalidMetrics(
                    f"nonfinite numeric metric {key!r} at global_step {step} "
                    f"in {label}"
                )


def _median(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(statistics.median(_metric_values(rows, key)))


def _block_evidence(
    by_step: Mapping[int, Mapping[str, Any]], step: int, gate: DirectFormalGate,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    blocks: list[dict[str, Any]] = []
    first = step - gate.block_count * gate.block_size
    for block_index in range(gate.block_count):
        start = first + block_index * gate.block_size
        end = start + gate.block_size
        rows = _exact_window(by_step, start, end, f"candidate {step} block {block_index}")
        blocks.append({
            "index": block_index,
            "start_exclusive": start,
            "end_inclusive": end,
            "rows": len(rows),
            "medians": {key: _median(rows, key) for key in gate.primary_metrics},
        })

    plateau: dict[str, dict[str, Any]] = {}
    for key in gate.primary_metrics:
        values = [block["medians"][key] for block in blocks]
        high, low = max(values), min(values)
        scale = max(abs(high), 1.0e-9)
        relative_range = (high - low) / scale
        plateau[key] = {
            "block_medians": values,
            "relative_range": relative_range,
            "threshold_lte": float(gate.tolerance),
            "passed": relative_range <= gate.tolerance,
        }
    return blocks, plateau


def _health_evidence(
    rows: Sequence[Mapping[str, Any]], start: int, end: int,
    gate: DirectFormalGate,
) -> dict[str, Any]:
    medians = {
        key: _median(rows, key)
        for key in (
            "delta_op", "act/align", "bank/live_ops_q_a",
            "bank/live_ops_q_delta", "loss/proposal", "act/c_delta_spread",
        )
    }
    gradient_medians = {
        module: _median(rows, f"gnorm/{module}")
        for module in gate.expected_gradient_modules
    }
    delta_sel = {
        f"h{h}": _median(rows, f"delta_sel/h{h}") for h in range(1, 5)
    }
    skipped = _metric_values(rows, "grad_skipped")
    if any(value not in (0.0, 1.0) for value in skipped):
        raise _InvalidMetrics("grad_skipped must contain only exact 0/1 values")
    skipped_rate = sum(skipped) / len(skipped)

    expected = set(gate.expected_gradient_modules)
    nonzero_modules: set[str] = set()
    for row in rows:
        for key, value in row.items():
            if not isinstance(key, str) or not key.startswith("gnorm/"):
                continue
            try:
                numeric = _finite_number(value, f"metric {key!r}")
            except ValueError as error:
                raise _InvalidMetrics(str(error)) from error
            if numeric != 0.0:
                nonzero_modules.add(key.removeprefix("gnorm/"))
    unexpected = sorted(nonzero_modules - expected)

    checks = {
        "delta_op": medians["delta_op"] > gate.delta_op_strict_gt,
        **{
            f"delta_sel/{name}": value > gate.delta_sel_strict_gt
            for name, value in delta_sel.items()
        },
        "act/align": medians["act/align"] < gate.act_align_strict_lt,
        "bank/live_ops_q_a": medians["bank/live_ops_q_a"] >= gate.live_ops_q_a_gte,
        "bank/live_ops_q_delta": (
            medians["bank/live_ops_q_delta"] >= gate.live_ops_q_delta_gte
        ),
        "loss/proposal_below_uniform": (
            medians["loss/proposal"] < gate.proposal_uniform_ce
        ),
        "loss/proposal_off_floor": (
            medians["loss/proposal"] < gate.proposal_off_floor_strict_lt
        ),
        "act/c_delta_spread": (
            medians["act/c_delta_spread"] > gate.c_delta_spread_strict_gt
        ),
        **{
            f"gnorm/{module}": value > (
                gate.gnorm_bank_strict_gt if module == "bank" else
                gate.gnorm_q_delta_strict_gt if module == "q_delta" else 0.0
            )
            for module, value in gradient_medians.items()
        },
        "grad_skipped": skipped_rate < gate.skipped_rate_strict_lt,
        "unexpected_gradients": not unexpected,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "window": {
            "start_exclusive": start,
            "end_inclusive": end,
            "rows": len(rows),
        },
        "metrics": {
            "medians": medians,
            "gradient_medians": gradient_medians,
            "delta_sel_horizon_medians": delta_sel,
            "skipped_rate": skipped_rate,
            "unexpected_module_gradients": unexpected,
        },
        "checks": checks,
        "failures": failed,
        "passed": not failed,
    }


def _reference_medians(
    by_step: Mapping[int, Mapping[str, Any]], gate: DirectFormalGate,
) -> dict[str, float]:
    rows = _exact_window(
        by_step, gate.reference_start_exclusive, gate.reference_end_inclusive,
        "immutable non-regression reference",
    )
    return {key: _median(rows, key) for key in gate.primary_metrics}


def _nonregression_evidence(
    candidate: Mapping[str, float], reference: Mapping[str, float],
    gate: DirectFormalGate,
) -> dict[str, Any]:
    metrics: dict[str, dict[str, Any]] = {}
    for key in gate.primary_metrics:
        reference_value = reference[key]
        candidate_value = candidate[key]
        relative_worsening = (
            (candidate_value - reference_value) /
            max(abs(reference_value), 1.0e-12)
        )
        metrics[key] = {
            "reference_median": reference_value,
            "candidate_median": candidate_value,
            "relative_worsening": relative_worsening,
            "threshold_lte": float(gate.tolerance),
            "passed": relative_worsening <= gate.tolerance,
        }
    failed = [key for key, evidence in metrics.items() if not evidence["passed"]]
    return {"metrics": metrics, "failures": failed, "passed": not failed}


def _checkpoint_evidence(
    by_step: Mapping[int, Mapping[str, Any]], step: int, gate: DirectFormalGate,
) -> dict[str, Any]:
    blocks, plateau = _block_evidence(by_step, step, gate)
    terminal_start = step - gate.block_size
    terminal_rows = _exact_window(
        by_step, terminal_start, step, f"candidate {step} terminal health window",
    )
    health = _health_evidence(terminal_rows, terminal_start, step, gate)
    reference = _reference_medians(by_step, gate)
    candidate = blocks[-1]["medians"]
    nonregression = _nonregression_evidence(candidate, reference, gate)
    plateau_failures = [key for key, value in plateau.items() if not value["passed"]]

    if not health["passed"]:
        status, reason = "ABORT", "health_gate_failed"
    elif not nonregression["passed"]:
        status, reason = "ABORT", "nonregression_gate_failed"
    elif not plateau_failures:
        status, reason = "PASS", "first_passing_checkpoint"
    else:
        status, reason = "MOVING", "convergence_criteria_moving"
    return {
        "step": step,
        "status": status,
        "reason": reason,
        "blocks": blocks,
        "convergence": {
            "metrics": plateau,
            "failures": plateau_failures,
            "passed": not plateau_failures,
        },
        "health": health,
        "nonregression": nonregression,
    }


def _base_receipt(
    gate: DirectFormalGate, current_step: object, rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    numeric_steps = [
        row.get("global_step") for row in rows
        if isinstance(row, Mapping) and _is_int(row.get("global_step"))
    ]
    return {
        "format": DIRECT_FORMAL_FORMAT,
        "status": "INVALID",
        "reason": "invalid_input",
        "current_step": current_step,
        "decision_step": None,
        "next_check_step": None,
        "gate": gate.as_dict(),
        "input": {
            "rows": len(rows),
            "minimum_step": min(numeric_steps) if numeric_steps else None,
            "maximum_step": max(numeric_steps) if numeric_steps else None,
        },
        "evaluations": [],
    }


def evaluate_direct_formal(
    rows: Iterable[Mapping[str, Any]], *, current_step: int | None = None,
    gate: DirectFormalGate | None = None,
) -> dict[str, Any]:
    """Evaluate a metrics prefix and return a JSON-safe decision receipt.

    Input/authentication failures are represented as ``INVALID`` receipts.
    Complete finite evidence that violates a health or non-regression gate is a
    scientific ``ABORT``.  A healthy but not-yet-flat prefix is ``MOVING`` until
    the hard cap, where it becomes ``ABORT``.  The first chronological passing
    checkpoint is terminal and remains selected when a longer prefix is read.
    """

    gate = gate or DirectFormalGate()
    try:
        materialized = list(rows)
    except Exception as error:  # pragma: no cover - exotic iterator failures
        materialized = []
        receipt = _base_receipt(gate, current_step, materialized)
        receipt["error"] = f"could not materialize metrics rows: {error}"
        return receipt

    inferred_step: object = current_step
    receipt = _base_receipt(gate, inferred_step, materialized)
    try:
        by_step: dict[int, Mapping[str, Any]] = {}
        previous = -1
        for index, row in enumerate(materialized):
            if not isinstance(row, Mapping):
                raise _InvalidMetrics(f"metrics row {index} must be a mapping")
            step = row.get("global_step")
            if not _is_int(step) or step <= 0:
                raise _InvalidMetrics(
                    f"metrics row {index} global_step must be a positive integer"
                )
            if step <= previous:
                raise _InvalidMetrics("metrics global_step values must be strictly increasing")
            previous = step
            by_step[step] = row

        if current_step is None:
            current_step = previous if materialized else 0
        if not _is_int(current_step) or current_step < 0:
            raise _InvalidMetrics("current_step must be a non-negative integer")
        receipt["current_step"] = current_step
        if current_step > gate.max_updates:
            raise _InvalidMetrics(
                f"current_step {current_step} exceeds max_updates={gate.max_updates}"
            )
        if materialized and previous != current_step:
            raise _InvalidMetrics(
                f"latest metrics step {previous} does not equal current_step {current_step}"
            )
        if not materialized and current_step != 0:
            raise _InvalidMetrics("nonzero current_step requires a matching metrics row")

        if current_step < gate.first_check:
            receipt.update({
                "status": "MOVING",
                "reason": "before_first_check",
                "next_check_step": gate.first_check,
            })
            return receipt

        required_start = gate.first_check - gate.block_count * gate.block_size
        required_prefix = _exact_window(
            by_step, required_start, current_step,
            "direct-formal decision metrics prefix",
        )
        _require_finite_numeric(required_prefix, "direct-formal decision metrics prefix")

        evaluations: list[dict[str, Any]] = []
        last_boundary = min(current_step, gate.max_updates)
        for step in range(gate.first_check, last_boundary + 1, gate.check_every):
            evidence = _checkpoint_evidence(by_step, step, gate)
            evaluations.append(evidence)
            if evidence["status"] in ("PASS", "ABORT"):
                receipt.update({
                    "status": evidence["status"],
                    "reason": evidence["reason"],
                    "decision_step": step,
                    "evaluations": evaluations,
                })
                return receipt

        receipt["evaluations"] = evaluations
        if current_step == gate.max_updates:
            receipt.update({
                "status": "ABORT",
                "reason": "max_updates_without_convergence",
                "decision_step": gate.max_updates,
            })
            return receipt

        receipt.update({
            "status": "MOVING",
            "reason": "convergence_criteria_moving",
            "next_check_step": next_direct_formal_check(current_step, gate),
        })
        return receipt
    except (_InvalidMetrics, ValueError) as error:
        receipt["error"] = str(error)
        return receipt


def receipt_exit_code(receipt: Mapping[str, Any]) -> int:
    """Map the four receipt states to stable CLI exit codes 0/1/2/3."""

    status = receipt.get("status") if isinstance(receipt, Mapping) else None
    if status not in _STATUS_EXIT_CODES:
        raise ValueError(f"unknown direct-formal receipt status: {status!r}")
    return _STATUS_EXIT_CODES[status]
