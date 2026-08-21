#!/usr/bin/env python
"""Is a run converged? Answers with evidence, not assertion.

Convergence here is NOT "the loss stopped moving" -- a run pinned at a degenerate
floor also stops moving, and this project has produced several. So the check is
two-sided:

  PLATEAU   the configured primary metrics' 2000-step block medians change by
            less than `tol` (relative) across the last `n_blocks` blocks, AND
  NOT-FLOOR the run is not sitting on a configured degenerate floor.

The legacy defaults retain the original full-stage checks. The proposal floor
is selected from the resolved proposal objective:

              loss/proposal  uniform Plackett-Luce = sum(log(128-i), i<4) = 19.361
              loss/proposal  uniform sparse CE = log(128) = 4.852
              act/align      disjoint-support MSE  = 8 * 0.25^2 = 0.500
              delta_sel      the clock signature   = |.| < 1e-4 (flat noise band)

A run's resolved ``config.json`` may override ``convergence.primary``,
``convergence.watch``, and ``convergence.floor_checks``. This lets a refinement
stage judge only the modules it can still train while old runs, whose configs do
not contain that section, continue to use the legacy gate.

Required ``phase_gate`` / ``liveness_gate`` / ``terminal_gate`` declarations
are checked on their exact immutable row windows. Required ``efficacy_gate``
declarations are checked only after convergence. Any required-gate failure
returns exit 4 so autostop can write the run's STOP sentinel without cancelling
scheduler jobs.

A run that plateaus ON a floor is CONVERGED-DEGENERATE, which is a different
answer from CONVERGED and must not be reported as one.

    python logs/convergence.py runs/r0a_conv [--block 2000] [--blocks 4] [--tol 0.02]
"""
import argparse, hashlib, json, math, os, statistics as st, sys, pathlib

UNIFORM_PL = sum(math.log(128 - i) for i in range(4))   # 19.360813
UNIFORM_SPARSE_CE = math.log(128)                        # 4.852030
ALIGN_FLOOR = 8 * 0.25 ** 2                             # 0.500
PRIMARY = ("loss", "loss/dyn", "loss/act", "loss/proposal", "act/align")
WATCH = ("delta_op", "delta_sel", "bank/live_ops", "grad_norm")
FLOOR_CHECKS = ("loss/proposal", "act/align", "delta_sel")
DEFAULT_BLOCK = 2000
DEFAULT_BLOCKS = 4
DEFAULT_TOL = 0.02
REQUIRED_GATE_FAILED = 4


class RequiredGateError(ValueError):
    """A declared required stage gate cannot be evaluated safely."""


class TransientMetricsRead(RuntimeError):
    """The writer has not terminated its final JSONL record yet."""


def _mapping(value, key):
    if not isinstance(value, dict):
        raise RequiredGateError(f"{key} must be a mapping")
    return value


def _finite_number(value, key, *, minimum=None):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(float(value))):
        raise RequiredGateError(f"{key} must be a finite number")
    out = float(value)
    if minimum is not None and out < minimum:
        raise RequiredGateError(f"{key} must be >= {minimum}")
    return out


def _exact_keys(value, expected, key):
    missing = sorted(set(expected) - set(value))
    unknown = sorted(set(value) - set(expected))
    if missing or unknown:
        raise RequiredGateError(
            f"{key} keys mismatch; missing={missing}, unknown={unknown}"
        )


def _required_gate(value, key):
    if value is None:
        return None
    gate = _mapping(value, key)
    required = gate.get("required")
    if not isinstance(required, bool):
        raise RequiredGateError(f"{key}.required must be a boolean")
    return gate if required else None


def _parse_liveness(value):
    gate = _required_gate(value, "liveness_gate")
    if gate is None:
        return None
    _exact_keys(gate, {
        "start_exclusive", "end_inclusive", "rows", "requirements", "required",
    }, "liveness_gate")
    start = gate["start_exclusive"]
    end = gate["end_inclusive"]
    count = gate["rows"]
    for raw, key in ((start, "start_exclusive"), (end, "end_inclusive"),
                     (count, "rows")):
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise RequiredGateError(f"liveness_gate.{key} must be a non-negative integer")
    if end <= start or count != end - start:
        raise RequiredGateError(
            "liveness_gate must describe every update in (start_exclusive, "
            "end_inclusive] exactly once"
        )
    req = _mapping(gate["requirements"], "liveness_gate.requirements")
    qa_expected = {
        "delta_op_median_strict_gt", "gnorm_bank_median_strict_gt",
        "gnorm_q_action_median_strict_gt", "skipped_rate_strict_lt",
        "unexpected_module_gradients", "nonfinite",
    }
    fresh_expected = {
        "delta_op_median_strict_gt",
        "delta_sel_horizon_medians_strict_gt",
        "live_ops_q_delta_median_gte",
        "c_delta_spread_median_strict_gt",
        "gnorm_bank_median_strict_gt",
        "gnorm_q_delta_median_strict_gt",
        "skipped_rate_strict_lt",
        "unexpected_module_gradients",
        "nonfinite",
    }
    fresh_only = fresh_expected - qa_expected
    if set(req) & fresh_only:
        _exact_keys(req, fresh_expected, "liveness_gate.requirements")
        for key in ("unexpected_module_gradients", "nonfinite"):
            if req[key] is not False:
                raise RequiredGateError(
                    f"liveness_gate.requirements.{key} must be boolean false"
                )
        return {
            "kind": "fresh_phase_b",
            "start": start,
            "end": end,
            "rows": count,
            "delta_op": _finite_number(
                req["delta_op_median_strict_gt"],
                "liveness_gate.requirements.delta_op_median_strict_gt",
            ),
            "delta_sel/horizon": _finite_number(
                req["delta_sel_horizon_medians_strict_gt"],
                "liveness_gate.requirements."
                "delta_sel_horizon_medians_strict_gt",
            ),
            "bank/live_ops_q_delta": _finite_number(
                req["live_ops_q_delta_median_gte"],
                "liveness_gate.requirements.live_ops_q_delta_median_gte",
                minimum=0.0,
            ),
            "act/c_delta_spread": _finite_number(
                req["c_delta_spread_median_strict_gt"],
                "liveness_gate.requirements.c_delta_spread_median_strict_gt",
                minimum=0.0,
            ),
            "gnorm/bank": _finite_number(
                req["gnorm_bank_median_strict_gt"],
                "liveness_gate.requirements.gnorm_bank_median_strict_gt",
                minimum=0.0,
            ),
            "gnorm/q_delta": _finite_number(
                req["gnorm_q_delta_median_strict_gt"],
                "liveness_gate.requirements.gnorm_q_delta_median_strict_gt",
                minimum=0.0,
            ),
            "skipped_rate": _finite_number(
                req["skipped_rate_strict_lt"],
                "liveness_gate.requirements.skipped_rate_strict_lt",
                minimum=0.0,
            ),
        }

    # Preserve the already-deployed QA schema and its exact strictness.
    _exact_keys(req, qa_expected, "liveness_gate.requirements")
    parsed = {
        "delta_op": _finite_number(
            req["delta_op_median_strict_gt"],
            "liveness_gate.requirements.delta_op_median_strict_gt",
        ),
        "gnorm/bank": _finite_number(
            req["gnorm_bank_median_strict_gt"],
            "liveness_gate.requirements.gnorm_bank_median_strict_gt",
            minimum=0.0,
        ),
        "gnorm/q_action": _finite_number(
            req["gnorm_q_action_median_strict_gt"],
            "liveness_gate.requirements.gnorm_q_action_median_strict_gt",
            minimum=0.0,
        ),
        "skipped_rate": _finite_number(
            req["skipped_rate_strict_lt"],
            "liveness_gate.requirements.skipped_rate_strict_lt",
            minimum=0.0,
        ),
    }
    for key in ("unexpected_module_gradients", "nonfinite"):
        if req[key] is not False:
            raise RequiredGateError(
                f"liveness_gate.requirements.{key} must be boolean false"
            )
    return {"start": start, "end": end, "rows": count, **parsed}


def _parse_phase_gate(value):
    gate = _required_gate(value, "phase_gate")
    if gate is None:
        return None
    _exact_keys(gate, {
        "start_exclusive", "end_inclusive", "rows", "requirements", "required",
    }, "phase_gate")
    start = gate["start_exclusive"]
    end = gate["end_inclusive"]
    count = gate["rows"]
    for raw, key in ((start, "start_exclusive"), (end, "end_inclusive"),
                     (count, "rows")):
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise RequiredGateError(
                f"phase_gate.{key} must be a non-negative integer"
            )
    if end <= start or count != end - start:
        raise RequiredGateError(
            "phase_gate must describe every update in (start_exclusive, "
            "end_inclusive] exactly once"
        )
    req = _mapping(gate["requirements"], "phase_gate.requirements")
    expected = {
        "act_align_median_strict_lt",
        "c_a_spread_median_strict_gt",
        "c_delta_spread_median_strict_gt",
        "live_ops_q_a_median_gte",
        "live_ops_q_delta_median_gte",
        "proposal_loss_median_strict_lt",
        "skipped_rate_strict_lt",
        "bank_gradients",
        "expected_module_gradients",
        "nonfinite",
    }
    _exact_keys(req, expected, "phase_gate.requirements")
    if req["bank_gradients"] is not False:
        raise RequiredGateError(
            "phase_gate.requirements.bank_gradients must be boolean false"
        )
    if req["expected_module_gradients"] is not True:
        raise RequiredGateError(
            "phase_gate.requirements.expected_module_gradients must be boolean true"
        )
    if req["nonfinite"] is not False:
        raise RequiredGateError(
            "phase_gate.requirements.nonfinite must be boolean false"
        )
    return {
        "start": start,
        "end": end,
        "rows": count,
        "act/align": _finite_number(
            req["act_align_median_strict_lt"],
            "phase_gate.requirements.act_align_median_strict_lt",
        ),
        "act/c_a_spread": _finite_number(
            req["c_a_spread_median_strict_gt"],
            "phase_gate.requirements.c_a_spread_median_strict_gt",
            minimum=0.0,
        ),
        "act/c_delta_spread": _finite_number(
            req["c_delta_spread_median_strict_gt"],
            "phase_gate.requirements.c_delta_spread_median_strict_gt",
            minimum=0.0,
        ),
        "bank/live_ops_q_a": _finite_number(
            req["live_ops_q_a_median_gte"],
            "phase_gate.requirements.live_ops_q_a_median_gte",
            minimum=0.0,
        ),
        "bank/live_ops_q_delta": _finite_number(
            req["live_ops_q_delta_median_gte"],
            "phase_gate.requirements.live_ops_q_delta_median_gte",
            minimum=0.0,
        ),
        "loss/proposal": _finite_number(
            req["proposal_loss_median_strict_lt"],
            "phase_gate.requirements.proposal_loss_median_strict_lt",
        ),
        "skipped_rate": _finite_number(
            req["skipped_rate_strict_lt"],
            "phase_gate.requirements.skipped_rate_strict_lt",
            minimum=0.0,
        ),
    }


def _parse_terminal_gate(value):
    gate = _required_gate(value, "terminal_gate")
    if gate is None:
        return None
    _exact_keys(gate, {
        "window_start_exclusive", "window_end_inclusive", "requirements",
        "required",
    }, "terminal_gate")
    start = gate["window_start_exclusive"]
    end = gate["window_end_inclusive"]
    for raw, key in ((start, "window_start_exclusive"),
                     (end, "window_end_inclusive")):
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise RequiredGateError(
                f"terminal_gate.{key} must be a non-negative integer"
            )
    if end <= start:
        raise RequiredGateError(
            "terminal_gate.window_end_inclusive must exceed "
            "window_start_exclusive"
        )
    req = _mapping(gate["requirements"], "terminal_gate.requirements")
    expected = {
        "delta_op_median_strict_gt",
        "delta_sel_horizon_medians_strict_gt",
        "act_align_median_strict_lt",
        "live_ops_q_a_median_gte",
        "live_ops_q_delta_median_gte",
        "proposal_loss_median_strict_lt",
        "skipped_rate_strict_lt",
    }
    _exact_keys(req, expected, "terminal_gate.requirements")
    return {
        "start": start,
        "end": end,
        "rows": end - start,
        "delta_op": _finite_number(
            req["delta_op_median_strict_gt"],
            "terminal_gate.requirements.delta_op_median_strict_gt",
        ),
        "delta_sel/horizon": _finite_number(
            req["delta_sel_horizon_medians_strict_gt"],
            "terminal_gate.requirements.delta_sel_horizon_medians_strict_gt",
        ),
        "act/align": _finite_number(
            req["act_align_median_strict_lt"],
            "terminal_gate.requirements.act_align_median_strict_lt",
        ),
        "bank/live_ops_q_a": _finite_number(
            req["live_ops_q_a_median_gte"],
            "terminal_gate.requirements.live_ops_q_a_median_gte",
            minimum=0.0,
        ),
        "bank/live_ops_q_delta": _finite_number(
            req["live_ops_q_delta_median_gte"],
            "terminal_gate.requirements.live_ops_q_delta_median_gte",
            minimum=0.0,
        ),
        "loss/proposal": _finite_number(
            req["proposal_loss_median_strict_lt"],
            "terminal_gate.requirements.proposal_loss_median_strict_lt",
        ),
        "skipped_rate": _finite_number(
            req["skipped_rate_strict_lt"],
            "terminal_gate.requirements.skipped_rate_strict_lt",
            minimum=0.0,
        ),
    }


def _parse_efficacy(value):
    gate = _required_gate(value, "efficacy_gate")
    if gate is None:
        return None
    _exact_keys(gate, {
        "metric", "reference", "comparison", "max_relative_worsening", "required",
    }, "efficacy_gate")
    metric = gate["metric"]
    if not isinstance(metric, str) or not metric:
        raise RequiredGateError("efficacy_gate.metric must be a non-empty string")
    if gate["reference"] != "first_post_start_block":
        raise RequiredGateError(
            "efficacy_gate.reference must be first_post_start_block"
        )
    if gate["comparison"] != "final_convergence_block":
        raise RequiredGateError(
            "efficacy_gate.comparison must be final_convergence_block"
        )
    return {
        "metric": metric,
        "max_relative_worsening": _finite_number(
            gate["max_relative_worsening"],
            "efficacy_gate.max_relative_worsening", minimum=0.0,
        ),
    }


def _metric_values(rows, key):
    values = []
    for row in rows:
        value = row.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(float(value))):
            raise RequiredGateError(f"metric {key!r} is missing or nonfinite")
        values.append(float(value))
    return values


def _read_metric_rows(path):
    rows = []
    with open(path) as stream:
        lines = stream.readlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        if index == len(lines) - 1 and not line.endswith("\n"):
            # Rank 0 writes line-buffered JSONL. Seeing the last write between
            # its payload and newline is a read race, never a method failure.
            raise TransientMetricsRead("unterminated final metrics JSONL record")
        rows.append(json.loads(line))
    return rows


def _exact_window(rows, lo, hi, count, label):
    selected = [row for row in rows if lo < row.get("global_step", -1) <= hi]
    steps = [row.get("global_step") for row in selected]
    expected = list(range(lo + 1, hi + 1))
    if len(selected) != count or steps != expected:
        raise RequiredGateError(
            f"{label} is not exact/contiguous: expected {count} rows "
            f"for ({lo}, {hi}], got {len(selected)}"
        )
    return selected


def _require_all_numeric_finite(rows, label):
    for row in rows:
        for key, value in row.items():
            if (not isinstance(value, bool) and isinstance(value, (int, float))
                    and not math.isfinite(float(value))):
                raise RequiredGateError(f"nonfinite metric {key!r} in {label}")


def _skipped_rate(rows):
    skipped = _metric_values(rows, "grad_skipped")
    if any(value not in (0.0, 1.0) for value in skipped):
        raise RequiredGateError("grad_skipped must contain only 0/1 values")
    return sum(skipped) / len(skipped)


def _expected_gradient_modules(train_modules):
    if (not isinstance(train_modules, list) or
            not all(isinstance(name, str) and name for name in train_modules) or
            not train_modules or len(set(train_modules)) != len(train_modules)):
        raise RequiredGateError(
            "train_modules must uniquely identify expected gradient modules"
        )
    return set(train_modules)


def _nonzero_gradient_modules(rows):
    out = set()
    for row in rows:
        for key, value in row.items():
            if not key.startswith("gnorm/"):
                continue
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(float(value))):
                raise RequiredGateError(f"metric {key!r} is missing or nonfinite")
            if float(value) != 0.0:
                out.add(key.removeprefix("gnorm/"))
    return out


def _write_fixed_gate_artifact(run_dir, name, gate, identity, *, passed,
                               requirements, metrics, failures):
    path = pathlib.Path(run_dir) / (
        f"{name}_{gate['start']:06d}_{gate['end']:06d}.json"
    )
    payload = {
        "run": identity["run_name"],
        "config_hash": identity["config_hash"],
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "window": {
            "lo_exclusive": gate["start"],
            "hi_inclusive": gate["end"],
            "rows": gate["rows"],
            "contiguous": not any("exact/contiguous" in item for item in failures),
        },
        "requirements": requirements,
        "metrics": metrics,
        "failures": failures,
    }
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with open(tmp, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    except (OSError, TypeError, ValueError) as error:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise RequiredGateError(
            f"could not persist authoritative {name} verdict at {path}: {error}"
        ) from error
    return path


def _write_liveness_artifact(run_dir, gate, identity, *, passed, metrics, failures):
    path = pathlib.Path(run_dir) / (
        f"liveness_{gate['start']:06d}_{gate['end']:06d}.json"
    )
    if gate.get("kind") == "fresh_phase_b":
        requirements = {
            "delta_op_median_strict_gt": gate["delta_op"],
            "delta_sel_horizon_medians_strict_gt": gate["delta_sel/horizon"],
            "live_ops_q_delta_median_gte": gate["bank/live_ops_q_delta"],
            "c_delta_spread_median_strict_gt": gate["act/c_delta_spread"],
            "gnorm_bank_median_strict_gt": gate["gnorm/bank"],
            "gnorm_q_delta_median_strict_gt": gate["gnorm/q_delta"],
            "skipped_rate_strict_lt": gate["skipped_rate"],
            "unexpected_module_gradients": False,
            "nonfinite": False,
        }
    else:
        # Keep the deployed QA artifact schema exact.
        requirements = {
            "delta_op_median_strict_gt": gate["delta_op"],
            "gnorm_bank_median_strict_gt": gate["gnorm/bank"],
            "gnorm_q_action_median_strict_gt": gate["gnorm/q_action"],
            "skipped_rate_strict_lt": gate["skipped_rate"],
            "unexpected_module_gradients": False,
            "nonfinite": False,
        }
    payload = {
        "run": identity["run_name"],
        "config_hash": identity["config_hash"],
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "window": {
            "lo_exclusive": gate["start"],
            "hi_inclusive": gate["end"],
            "rows": gate["rows"],
            "contiguous": not any("exact/contiguous" in item for item in failures),
        },
        "requirements": requirements,
        "metrics": metrics,
        "failures": failures,
    }
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with open(tmp, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    except (OSError, TypeError, ValueError) as error:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise RequiredGateError(
            f"could not persist authoritative liveness verdict at {path}: {error}"
        ) from error
    return path


def phase_gate_verdict(rows, gate, train_modules, run_dir, identity):
    """Evaluate the fresh-R0 Phase-A handoff gate."""
    if gate is None:
        return None
    latest = max((row.get("global_step", -1) for row in rows), default=-1)
    if latest < gate["end"]:
        print(f"PHASE_GATE: PENDING — exact window ends at {gate['end']}; latest={latest}")
        return 1
    requirements = {
        "act_align_median_strict_lt": gate["act/align"],
        "c_a_spread_median_strict_gt": gate["act/c_a_spread"],
        "c_delta_spread_median_strict_gt": gate["act/c_delta_spread"],
        "live_ops_q_a_median_gte": gate["bank/live_ops_q_a"],
        "live_ops_q_delta_median_gte": gate["bank/live_ops_q_delta"],
        "proposal_loss_median_strict_lt": gate["loss/proposal"],
        "skipped_rate_strict_lt": gate["skipped_rate"],
        "bank_gradients": False,
        "expected_module_gradients": True,
        "nonfinite": False,
    }
    try:
        window = _exact_window(
            rows, gate["start"], gate["end"], gate["rows"], "phase gate window",
        )
        _require_all_numeric_finite(window, "phase gate window")
        metric_names = (
            "act/align", "act/c_a_spread", "act/c_delta_spread",
            "bank/live_ops_q_a", "bank/live_ops_q_delta", "loss/proposal",
        )
        medians = {key: st.median(_metric_values(window, key)) for key in metric_names}
        skipped_rate = _skipped_rate(window)
        expected = _expected_gradient_modules(train_modules)
        grad_medians = {
            name: st.median(_metric_values(window, f"gnorm/{name}"))
            for name in sorted(expected)
        }
        nonzero = _nonzero_gradient_modules(window)
        unexpected = sorted(nonzero - expected)
    except RequiredGateError as error:
        failure = str(error)
        try:
            artifact = _write_fixed_gate_artifact(
                run_dir, "phase_gate", gate, identity, passed=False,
                requirements=requirements, metrics={}, failures=[failure],
            )
            print(f"PHASE_GATE: wrote {artifact}")
        except RequiredGateError as persist_error:
            failure = f"{failure}; {persist_error}"
        print(f"VERDICT: PHASE_GATE_FAILED — {failure}")
        return REQUIRED_GATE_FAILED

    checks = {
        "act/align": medians["act/align"] < gate["act/align"],
        "act/c_a_spread": medians["act/c_a_spread"] > gate["act/c_a_spread"],
        "act/c_delta_spread": (
            medians["act/c_delta_spread"] > gate["act/c_delta_spread"]
        ),
        "bank/live_ops_q_a": (
            medians["bank/live_ops_q_a"] >= gate["bank/live_ops_q_a"]
        ),
        "bank/live_ops_q_delta": (
            medians["bank/live_ops_q_delta"] >= gate["bank/live_ops_q_delta"]
        ),
        "loss/proposal": medians["loss/proposal"] < gate["loss/proposal"],
        "grad_skipped": skipped_rate < gate["skipped_rate"],
        "bank_gradients": "bank" not in nonzero,
        "expected_gradients": (
            not unexpected and all(value > 0.0 for value in grad_medians.values())
        ),
    }
    metrics = {
        "act_align_median": medians["act/align"],
        "c_a_spread_median": medians["act/c_a_spread"],
        "c_delta_spread_median": medians["act/c_delta_spread"],
        "live_ops_q_a_median": medians["bank/live_ops_q_a"],
        "live_ops_q_delta_median": medians["bank/live_ops_q_delta"],
        "proposal_loss_median": medians["loss/proposal"],
        "skipped_rate": skipped_rate,
        "gradient_medians": grad_medians,
        "unexpected_module_gradients": unexpected,
    }
    failed = [key for key, passed in checks.items() if not passed]
    print(
        f"PHASE_GATE: window=({gate['start']}, {gate['end']}] rows={len(window)} "
        f"align={medians['act/align']:.8g} "
        f"c_a={medians['act/c_a_spread']:.8g} "
        f"c_delta={medians['act/c_delta_spread']:.8g} "
        f"live_q_a={medians['bank/live_ops_q_a']:.8g} "
        f"live_q_delta={medians['bank/live_ops_q_delta']:.8g} "
        f"proposal={medians['loss/proposal']:.8g} skips={skipped_rate:.3%}"
    )
    try:
        artifact = _write_fixed_gate_artifact(
            run_dir, "phase_gate", gate, identity, passed=not failed,
            requirements=requirements, metrics=metrics, failures=failed,
        )
    except RequiredGateError as error:
        print(f"VERDICT: PHASE_GATE_FAILED — {error}")
        return REQUIRED_GATE_FAILED
    print(f"PHASE_GATE: wrote {artifact}")
    if failed:
        print("VERDICT: PHASE_GATE_FAILED — " + "; ".join(failed))
        return REQUIRED_GATE_FAILED
    # A successful phase boundary authorizes handoff; it is not terminal
    # convergence. Keep rc=1 so autostop can never turn PASS into STOP even if
    # its generic minimum-step threshold is configured at this exact boundary.
    print("PHASE_GATE: PASS — hand off to phase B; do not write STOP")
    return 1


def _fresh_liveness_verdict(rows, gate, train_modules, run_dir, identity):
    latest = max((row.get("global_step", -1) for row in rows), default=-1)
    if latest < gate["end"]:
        print(f"LIVENESS: PENDING — exact window ends at {gate['end']}; latest={latest}")
        return 1
    try:
        window = _exact_window(
            rows, gate["start"], gate["end"], gate["rows"], "liveness window",
        )
        _require_all_numeric_finite(window, "liveness window")
        medians = {
            key: st.median(_metric_values(window, key))
            for key in (
                "delta_op", "bank/live_ops_q_delta", "act/c_delta_spread",
                "gnorm/bank", "gnorm/q_delta",
            )
        }
        delta_sel = {
            f"h{h}": st.median(_metric_values(window, f"delta_sel/h{h}"))
            for h in range(1, 5)
        }
        skipped_rate = _skipped_rate(window)
        expected = _expected_gradient_modules(train_modules)
        unexpected = sorted(_nonzero_gradient_modules(window) - expected)
    except RequiredGateError as error:
        failure = str(error)
        try:
            artifact = _write_liveness_artifact(
                run_dir, gate, identity, passed=False, metrics={}, failures=[failure],
            )
            print(f"LIVENESS: wrote {artifact}")
        except RequiredGateError as persist_error:
            failure = f"{failure}; {persist_error}"
        print(f"VERDICT: LIVENESS_FAILED — {failure}")
        return REQUIRED_GATE_FAILED

    checks = {
        "delta_op": medians["delta_op"] > gate["delta_op"],
        **{
            f"delta_sel/{name}": value > gate["delta_sel/horizon"]
            for name, value in delta_sel.items()
        },
        "bank/live_ops_q_delta": (
            medians["bank/live_ops_q_delta"] >= gate["bank/live_ops_q_delta"]
        ),
        "act/c_delta_spread": (
            medians["act/c_delta_spread"] > gate["act/c_delta_spread"]
        ),
        "gnorm/bank": medians["gnorm/bank"] > gate["gnorm/bank"],
        "gnorm/q_delta": medians["gnorm/q_delta"] > gate["gnorm/q_delta"],
        "grad_skipped": skipped_rate < gate["skipped_rate"],
        "unexpected_gradients": not unexpected,
    }
    metrics = {
        "delta_op_median": medians["delta_op"],
        "delta_sel_horizon_medians": delta_sel,
        "live_ops_q_delta_median": medians["bank/live_ops_q_delta"],
        "c_delta_spread_median": medians["act/c_delta_spread"],
        "gnorm_bank_median": medians["gnorm/bank"],
        "gnorm_q_delta_median": medians["gnorm/q_delta"],
        "skipped_rate": skipped_rate,
        "unexpected_module_gradients": unexpected,
    }
    failed = [key for key, passed in checks.items() if not passed]
    print(
        f"LIVENESS: window=({gate['start']}, {gate['end']}] rows={len(window)} "
        f"delta_op={medians['delta_op']:.8g} "
        f"delta_sel={[delta_sel[f'h{h}'] for h in range(1, 5)]} "
        f"live_q_delta={medians['bank/live_ops_q_delta']:.8g} "
        f"c_delta={medians['act/c_delta_spread']:.8g} "
        f"bank={medians['gnorm/bank']:.8g} "
        f"q_delta={medians['gnorm/q_delta']:.8g} skips={skipped_rate:.3%}"
    )
    try:
        artifact = _write_liveness_artifact(
            run_dir, gate, identity, passed=not failed, metrics=metrics,
            failures=failed,
        )
    except RequiredGateError as error:
        print(f"VERDICT: LIVENESS_FAILED — {error}")
        return REQUIRED_GATE_FAILED
    print(f"LIVENESS: wrote {artifact}")
    if failed:
        print("VERDICT: LIVENESS_FAILED — " + "; ".join(failed))
        return REQUIRED_GATE_FAILED
    print("LIVENESS: PASS")
    return None


def liveness_verdict(rows, gate, train_modules, run_dir, identity):
    """Return None on PASS/not-applicable, 1 while pending, or 4 on failure."""
    if gate is None:
        return None
    if gate.get("kind") == "fresh_phase_b":
        return _fresh_liveness_verdict(
            rows, gate, train_modules, run_dir, identity,
        )
    latest = max((row.get("global_step", -1) for row in rows), default=-1)
    if latest < gate["end"]:
        print(f"LIVENESS: PENDING — exact window ends at {gate['end']}; latest={latest}")
        return 1
    try:
        window = _exact_window(
            rows, gate["start"], gate["end"], gate["rows"], "liveness window",
        )
        # The declared nonfinite check covers every numeric metric, not just
        # the three thresholded series.
        for row in window:
            for key, value in row.items():
                if (not isinstance(value, bool) and isinstance(value, (int, float))
                        and not math.isfinite(float(value))):
                    raise RequiredGateError(f"nonfinite metric {key!r} in liveness window")
        medians = {
            key: st.median(_metric_values(window, key))
            for key in ("delta_op", "gnorm/bank", "gnorm/q_action")
        }
        skipped = _metric_values(window, "grad_skipped")
        if any(value not in (0.0, 1.0) for value in skipped):
            raise RequiredGateError("grad_skipped must contain only 0/1 values")
        skipped_rate = sum(skipped) / len(skipped)
        if (not isinstance(train_modules, list) or
                not all(isinstance(name, str) and name for name in train_modules)):
            raise RequiredGateError("train_modules must identify expected gradient modules")
        expected_modules = set(train_modules)
        unexpected = sorted({
            key.removeprefix("gnorm/")
            for row in window for key, value in row.items()
            if key.startswith("gnorm/") and key.removeprefix("gnorm/") not in expected_modules
            and isinstance(value, (int, float)) and not isinstance(value, bool)
            and float(value) != 0.0
        })
    except RequiredGateError as error:
        failure = str(error)
        try:
            artifact = _write_liveness_artifact(
                run_dir, gate, identity, passed=False, metrics={}, failures=[failure],
            )
            print(f"LIVENESS: wrote {artifact}")
        except RequiredGateError as persist_error:
            failure = f"{failure}; {persist_error}"
        print(f"VERDICT: LIVENESS_FAILED — {failure}")
        return REQUIRED_GATE_FAILED

    checks = {
        "delta_op": medians["delta_op"] > gate["delta_op"],
        "gnorm/bank": medians["gnorm/bank"] > gate["gnorm/bank"],
        "gnorm/q_action": medians["gnorm/q_action"] > gate["gnorm/q_action"],
        "grad_skipped": skipped_rate < gate["skipped_rate"],
        "unexpected_gradients": not unexpected,
    }
    print(
        f"LIVENESS: window=({gate['start']}, {gate['end']}] rows={len(window)} "
        f"delta_op={medians['delta_op']:.8g} bank={medians['gnorm/bank']:.8g} "
        f"q_action={medians['gnorm/q_action']:.8g} skips={skipped_rate:.3%}"
    )
    metrics = {
        "delta_op_median": medians["delta_op"],
        "gnorm_bank_median": medians["gnorm/bank"],
        "gnorm_q_action_median": medians["gnorm/q_action"],
        "skipped_rate": skipped_rate,
        "unexpected_module_gradients": unexpected,
    }
    failed = [key for key, passed in checks.items() if not passed]
    if not all(checks.values()):
        if unexpected:
            failed.append("unexpected=" + ",".join(unexpected))
        try:
            artifact = _write_liveness_artifact(
                run_dir, gate, identity, passed=False, metrics=metrics, failures=failed,
            )
            print(f"LIVENESS: wrote {artifact}")
        except RequiredGateError as error:
            failed.append(str(error))
        print("VERDICT: LIVENESS_FAILED — " + "; ".join(failed))
        return REQUIRED_GATE_FAILED
    try:
        artifact = _write_liveness_artifact(
            run_dir, gate, identity, passed=True, metrics=metrics, failures=[],
        )
    except RequiredGateError as error:
        print(f"VERDICT: LIVENESS_FAILED — {error}")
        return REQUIRED_GATE_FAILED
    print(f"LIVENESS: wrote {artifact}")
    print("LIVENESS: PASS")
    return None


def terminal_gate_verdict(rows, gate, run_dir, identity):
    """Evaluate the immutable final Phase-B outcome window."""
    if gate is None:
        return None
    latest = max((row.get("global_step", -1) for row in rows), default=-1)
    if latest < gate["end"]:
        print(
            f"TERMINAL_GATE: PENDING — exact window ends at {gate['end']}; "
            f"latest={latest}"
        )
        return 1
    requirements = {
        "delta_op_median_strict_gt": gate["delta_op"],
        "delta_sel_horizon_medians_strict_gt": gate["delta_sel/horizon"],
        "act_align_median_strict_lt": gate["act/align"],
        "live_ops_q_a_median_gte": gate["bank/live_ops_q_a"],
        "live_ops_q_delta_median_gte": gate["bank/live_ops_q_delta"],
        "proposal_loss_median_strict_lt": gate["loss/proposal"],
        "skipped_rate_strict_lt": gate["skipped_rate"],
    }
    try:
        window = _exact_window(
            rows, gate["start"], gate["end"], gate["rows"],
            "terminal gate window",
        )
        _require_all_numeric_finite(window, "terminal gate window")
        medians = {
            key: st.median(_metric_values(window, key))
            for key in (
                "delta_op", "act/align", "bank/live_ops_q_a",
                "bank/live_ops_q_delta", "loss/proposal",
            )
        }
        delta_sel = {
            f"h{h}": st.median(_metric_values(window, f"delta_sel/h{h}"))
            for h in range(1, 5)
        }
        skipped_rate = _skipped_rate(window)
    except RequiredGateError as error:
        failure = str(error)
        try:
            artifact = _write_fixed_gate_artifact(
                run_dir, "terminal_gate", gate, identity, passed=False,
                requirements=requirements, metrics={}, failures=[failure],
            )
            print(f"TERMINAL_GATE: wrote {artifact}")
        except RequiredGateError as persist_error:
            failure = f"{failure}; {persist_error}"
        print(f"VERDICT: TERMINAL_GATE_FAILED — {failure}")
        return REQUIRED_GATE_FAILED

    checks = {
        "delta_op": medians["delta_op"] > gate["delta_op"],
        **{
            f"delta_sel/{name}": value > gate["delta_sel/horizon"]
            for name, value in delta_sel.items()
        },
        "act/align": medians["act/align"] < gate["act/align"],
        "bank/live_ops_q_a": (
            medians["bank/live_ops_q_a"] >= gate["bank/live_ops_q_a"]
        ),
        "bank/live_ops_q_delta": (
            medians["bank/live_ops_q_delta"] >= gate["bank/live_ops_q_delta"]
        ),
        "loss/proposal": medians["loss/proposal"] < gate["loss/proposal"],
        "grad_skipped": skipped_rate < gate["skipped_rate"],
    }
    metrics = {
        "delta_op_median": medians["delta_op"],
        "delta_sel_horizon_medians": delta_sel,
        "act_align_median": medians["act/align"],
        "live_ops_q_a_median": medians["bank/live_ops_q_a"],
        "live_ops_q_delta_median": medians["bank/live_ops_q_delta"],
        "proposal_loss_median": medians["loss/proposal"],
        "skipped_rate": skipped_rate,
    }
    failed = [key for key, passed in checks.items() if not passed]
    print(
        f"TERMINAL_GATE: window=({gate['start']}, {gate['end']}] "
        f"rows={len(window)} delta_op={medians['delta_op']:.8g} "
        f"delta_sel={[delta_sel[f'h{h}'] for h in range(1, 5)]} "
        f"align={medians['act/align']:.8g} "
        f"live_q_a={medians['bank/live_ops_q_a']:.8g} "
        f"live_q_delta={medians['bank/live_ops_q_delta']:.8g} "
        f"proposal={medians['loss/proposal']:.8g} skips={skipped_rate:.3%}"
    )
    try:
        artifact = _write_fixed_gate_artifact(
            run_dir, "terminal_gate", gate, identity, passed=not failed,
            requirements=requirements, metrics=metrics, failures=failed,
        )
    except RequiredGateError as error:
        print(f"VERDICT: TERMINAL_GATE_FAILED — {error}")
        return REQUIRED_GATE_FAILED
    print(f"TERMINAL_GATE: wrote {artifact}")
    if failed:
        print("VERDICT: TERMINAL_GATE_FAILED — " + "; ".join(failed))
        return REQUIRED_GATE_FAILED
    print("TERMINAL_GATE: PASS")
    return None


def efficacy_verdict(rows, gate, convergence, final_step):
    if gate is None:
        return None
    start = convergence["start_step"]
    block = convergence["block"]
    if start is None:
        print("VERDICT: EFFICACY_FAILED — convergence.start_step is required")
        return REQUIRED_GATE_FAILED
    try:
        reference = _exact_window(
            rows, start, start + block, block, "efficacy reference block",
        )
        comparison = _exact_window(
            rows, final_step - block, final_step, block,
            "efficacy final convergence block",
        )
        first = st.median(_metric_values(reference, gate["metric"]))
        final = st.median(_metric_values(comparison, gate["metric"]))
    except RequiredGateError as error:
        print(f"VERDICT: EFFICACY_FAILED — {error}")
        return REQUIRED_GATE_FAILED
    relative = (final - first) / max(abs(first), 1e-12)
    print(
        f"EFFICACY: metric={gate['metric']} first={first:.8g} final={final:.8g} "
        f"relative_worsening={relative:.3%} "
        f"max={gate['max_relative_worsening']:.3%}"
    )
    if relative > gate["max_relative_worsening"]:
        print("VERDICT: EFFICACY_FAILED — final block worsened beyond the configured maximum")
        return REQUIRED_GATE_FAILED
    print("EFFICACY: PASS")
    return None


def _metric_names(value, key):
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        raise ValueError(f"convergence.{key} must be a list of metric names")
    return tuple(value)


def _positive_int(value, key):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"convergence.{key} must be a positive integer")
    return value


def _positive_float(value, key):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"convergence.{key} must be positive")
    return float(value)


def _proposal_mode(cfg):
    losses = cfg.get("losses", {})
    if not isinstance(losses, dict):
        raise ValueError("losses must be a mapping")
    proposal = losses.get("proposal", {})
    if not isinstance(proposal, dict):
        raise ValueError("losses.proposal must be a mapping")
    mode = str(proposal.get("mode", "pl"))
    if mode not in ("pl", "sparse_ce", "dense_kl"):
        raise ValueError(f"unknown losses.proposal.mode: {mode!r}")
    return mode


def _legacy_gate(proposal_mode="pl"):
    return {
        "primary": PRIMARY,
        "watch": WATCH,
        "floor_checks": FLOOR_CHECKS,
        "block": DEFAULT_BLOCK,
        "blocks": DEFAULT_BLOCKS,
        "tol": DEFAULT_TOL,
        "start_step": None,
        "require_full_window": False,
        "proposal_mode": proposal_mode,
        "phase": None,
        "liveness": None,
        "terminal": None,
        "efficacy": None,
        "train_modules": [],
        "identity": {"run_name": "", "config_hash": ""},
    }


def gate_config(run_dir, config_path=None):
    """Load a stage gate while preserving config-less legacy semantics."""
    path = (pathlib.Path(config_path) if config_path
            else pathlib.Path(run_dir) / "config.json")
    if not path.exists():
        if config_path:
            raise ValueError(f"convergence config does not exist: {path}")
        return _legacy_gate()

    if path.suffix == ".json":
        with open(path) as f:
            cfg = json.load(f)
    else:
        import yaml
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"convergence config must be a mapping: {path}")
    proposal_mode = _proposal_mode(cfg)
    hashed = {k: v for k, v in cfg.items() if k != "link"}
    common = {
        "phase": _parse_phase_gate(cfg.get("phase_gate")),
        "liveness": _parse_liveness(cfg.get("liveness_gate")),
        "terminal": _parse_terminal_gate(cfg.get("terminal_gate")),
        "efficacy": _parse_efficacy(cfg.get("efficacy_gate")),
        "train_modules": cfg.get("train_modules", []),
        "identity": {
            "run_name": str(cfg.get("run", {}).get("name", "")),
            "config_hash": hashlib.blake2b(
                json.dumps(hashed, sort_keys=True, default=str).encode(), digest_size=8,
            ).hexdigest(),
        },
    }
    convergence = cfg.get("convergence")
    if convergence is None:
        out = _legacy_gate(proposal_mode)
        out.update(common)
        return out
    if not isinstance(convergence, dict):
        raise ValueError("convergence must be a mapping")

    out = {
        "primary": _metric_names(
            convergence.get("primary", list(PRIMARY)), "primary",
        ),
        "watch": _metric_names(
            convergence.get("watch", list(WATCH)), "watch",
        ),
        "floor_checks": _metric_names(
            convergence.get("floor_checks", list(FLOOR_CHECKS)), "floor_checks",
        ),
        "block": _positive_int(
            convergence.get("block", DEFAULT_BLOCK), "block",
        ),
        "blocks": _positive_int(
            convergence.get("blocks", DEFAULT_BLOCKS), "blocks",
        ),
        "tol": _positive_float(convergence.get("tol", DEFAULT_TOL), "tol"),
        "start_step": convergence.get("start_step"),
        # A declared stage gate is a promise about a complete window. The
        # config-less legacy path intentionally keeps the historical >=2 rule.
        "require_full_window": True,
        "proposal_mode": proposal_mode,
        **common,
    }
    if (out["start_step"] is not None and
            (isinstance(out["start_step"], bool) or
             not isinstance(out["start_step"], int) or out["start_step"] < 0)):
        raise ValueError("convergence.start_step must be a non-negative integer")
    unknown = sorted(set(out["floor_checks"]) - set(FLOOR_CHECKS))
    if unknown:
        raise ValueError(f"unknown convergence floor_checks: {', '.join(unknown)}")
    return out


def blocks(rows, size, n):
    if not rows:
        return []
    last = rows[-1]["global_step"]
    out = []
    for b in range(n):
        hi, lo = last - b * size, last - (b + 1) * size
        w = [r for r in rows if lo < r.get("global_step", -1) <= hi]
        if w:
            out.append((lo, hi, w))
    return list(reversed(out))


def med(w, k):
    v = [r[k] for r in w if k in r and r[k] is not None]
    return st.median(v) if v else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--block", type=int, default=None)
    ap.add_argument("--blocks", type=int, default=None)
    ap.add_argument("--tol", type=float, default=None)
    ap.add_argument("--start-step", type=int, default=None)
    ap.add_argument("--config", default=None,
                    help="default: <run_dir>/config.json; absent uses the legacy gate")
    a = ap.parse_args()

    try:
        gate = gate_config(a.run_dir, a.config)
    except RequiredGateError as e:
        print(f"VERDICT: REQUIRED_GATE_INVALID — {e}")
        return REQUIRED_GATE_FAILED
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f"invalid convergence config: {e}")
        return 2

    block = a.block if a.block is not None else gate["block"]
    n_blocks = a.blocks if a.blocks is not None else gate["blocks"]
    tol = a.tol if a.tol is not None else gate["tol"]
    start_step = a.start_step if a.start_step is not None else gate["start_step"]
    if block <= 0 or n_blocks <= 0 or tol <= 0 or (start_step is not None and start_step < 0):
        print("invalid convergence config: block, blocks, and tol must be positive; "
              "start_step must be non-negative")
        return 2

    p = pathlib.Path(a.run_dir) / "metrics.jsonl"
    if not p.exists():
        print(f"no metrics at {p}")
        return 2
    try:
        rows = _read_metric_rows(p)
    except TransientMetricsRead as e:
        print(f"transient metrics read: {e}")
        return 2
    except json.JSONDecodeError as e:
        failure = f"malformed complete metrics JSONL record: {e}"
        if gate["phase"] is not None:
            phase = gate["phase"]
            try:
                artifact = _write_fixed_gate_artifact(
                    a.run_dir, "phase_gate", phase, gate["identity"], passed=False,
                    requirements={
                        "act_align_median_strict_lt": phase["act/align"],
                        "c_a_spread_median_strict_gt": phase["act/c_a_spread"],
                        "c_delta_spread_median_strict_gt": (
                            phase["act/c_delta_spread"]
                        ),
                        "live_ops_q_a_median_gte": phase["bank/live_ops_q_a"],
                        "live_ops_q_delta_median_gte": (
                            phase["bank/live_ops_q_delta"]
                        ),
                        "proposal_loss_median_strict_lt": phase["loss/proposal"],
                        "skipped_rate_strict_lt": phase["skipped_rate"],
                        "bank_gradients": False,
                        "expected_module_gradients": True,
                        "nonfinite": False,
                    },
                    metrics={}, failures=[failure],
                )
                print(f"PHASE_GATE: wrote {artifact}")
            except RequiredGateError as persist_error:
                failure = f"{failure}; {persist_error}"
            print(f"VERDICT: REQUIRED_GATE_INVALID — {failure}")
            return REQUIRED_GATE_FAILED
        if gate["liveness"] is not None:
            try:
                artifact = _write_liveness_artifact(
                    a.run_dir, gate["liveness"], gate["identity"], passed=False,
                    metrics={}, failures=[failure],
                )
                print(f"LIVENESS: wrote {artifact}")
            except RequiredGateError as persist_error:
                failure = f"{failure}; {persist_error}"
            print(f"VERDICT: REQUIRED_GATE_INVALID — {failure}")
            return REQUIRED_GATE_FAILED
        if gate["terminal"] is not None:
            terminal = gate["terminal"]
            try:
                artifact = _write_fixed_gate_artifact(
                    a.run_dir, "terminal_gate", terminal, gate["identity"],
                    passed=False,
                    requirements={
                        "delta_op_median_strict_gt": terminal["delta_op"],
                        "delta_sel_horizon_medians_strict_gt": (
                            terminal["delta_sel/horizon"]
                        ),
                        "act_align_median_strict_lt": terminal["act/align"],
                        "live_ops_q_a_median_gte": terminal["bank/live_ops_q_a"],
                        "live_ops_q_delta_median_gte": (
                            terminal["bank/live_ops_q_delta"]
                        ),
                        "proposal_loss_median_strict_lt": (
                            terminal["loss/proposal"]
                        ),
                        "skipped_rate_strict_lt": terminal["skipped_rate"],
                    },
                    metrics={}, failures=[failure],
                )
                print(f"TERMINAL_GATE: wrote {artifact}")
            except RequiredGateError as persist_error:
                failure = f"{failure}; {persist_error}"
            print(f"VERDICT: REQUIRED_GATE_INVALID — {failure}")
            return REQUIRED_GATE_FAILED
        if gate["efficacy"] is not None:
            print(f"VERDICT: REQUIRED_GATE_INVALID — {failure}")
            return REQUIRED_GATE_FAILED
        print(f"invalid metrics JSON: {e}")
        return 2
    if not rows:
        print("metrics.jsonl is empty")
        return 2
    phase_rc = phase_gate_verdict(
        rows, gate["phase"], gate["train_modules"], a.run_dir, gate["identity"],
    )
    if phase_rc is not None:
        return phase_rc
    live_rc = liveness_verdict(
        rows, gate["liveness"], gate["train_modules"], a.run_dir, gate["identity"],
    )
    if live_rc is not None:
        return live_rc
    terminal_rc = terminal_gate_verdict(
        rows, gate["terminal"], a.run_dir, gate["identity"],
    )
    if terminal_rc is not None:
        return terminal_rc
    if start_step is not None:
        rows = [r for r in rows if r.get("global_step", -1) > start_step]
        if not rows:
            print(f"VERDICT: TOO_EARLY — no metrics after configured start step {start_step}")
            return 1
    step = rows[-1]["global_step"]
    bl = blocks(rows, block, n_blocks)
    print(f"run={a.run_dir}  last_step={step}  blocks={len(bl)} x {block}\n")

    if gate["require_full_window"]:
        span = step - rows[0]["global_step"] + 1
        need = block * n_blocks
        if len(bl) < n_blocks or span < need:
            print(f"VERDICT: TOO_EARLY — need {n_blocks} complete {block}-step blocks "
                  f"({need} steps after stage start); have {span}")
            return 1
    elif len(bl) < 2:
        print(f"VERDICT: TOO_EARLY — need >= {2 * block} steps to judge a plateau")
        return 1

    hdr = "metric".ljust(16) + "".join(f"{lo}-{hi}".rjust(16) for lo, hi, _ in bl) + "   rel.drift"
    print(hdr); print("-" * len(hdr))
    plateaued = {}
    missing = []
    for k in gate["primary"] + gate["watch"]:
        vals = [med(w, k) for _, _, w in bl]
        if all(math.isnan(v) for v in vals):
            if k in gate["primary"]:
                missing.append(k)
            continue
        if k in gate["primary"] and any(math.isnan(v) for v in vals):
            missing.append(k)
        fin = [v for v in vals if not math.isnan(v)]
        scale = max(abs(max(fin)), 1e-9)
        drift = (max(fin) - min(fin)) / scale
        if k in gate["primary"]:
            plateaued[k] = drift <= tol and k not in missing
        mark = ("" if k not in gate["primary"] else
                ("  ok" if drift <= tol and k not in missing else "  MOVING"))
        print(k.ljust(16) + "".join(f"{v:16.4f}" for v in vals) + f"{drift:12.3%}{mark}")

    if missing:
        print()
        print("VERDICT: NOT_CONVERGED — missing configured primary metrics: "
              + ", ".join(dict.fromkeys(missing)))
        return 1

    last = bl[-1][2]
    print()
    floors = []
    if "loss/proposal" in gate["floor_checks"]:
        prop = med(last, "loss/proposal")
        # Dense KL has no teacher-independent uniform floor: its value depends
        # on teacher entropy, so applying either categorical constant is wrong.
        proposal_floor = {
            "pl": (UNIFORM_PL, "uniform PL"),
            "sparse_ce": (UNIFORM_SPARSE_CE, "uniform sparse CE"),
            "dense_kl": None,
        }[gate["proposal_mode"]]
        if proposal_floor is not None:
            prop_floor, floor_name = proposal_floor
            if not math.isnan(prop) and prop >= prop_floor - 0.05:
                floors.append(
                    f"loss/proposal {prop:.3f} >= {floor_name} {prop_floor:.3f}"
                )
    if "act/align" in gate["floor_checks"]:
        align = med(last, "act/align")
        if not math.isnan(align) and align >= ALIGN_FLOOR - 0.005:
            floors.append(f"act/align {align:.4f} >= disjoint floor {ALIGN_FLOOR}")
    if "delta_sel" in gate["floor_checks"]:
        dsel = med(last, "delta_sel")
        if not math.isnan(dsel) and abs(dsel) < 1e-4:
            floors.append(f"|delta_sel| {abs(dsel):.2e} < 1e-4 (phase-clock signature)")

    all_flat = plateaued and all(plateaued.values())
    if not all_flat:
        moving = [k for k, v in plateaued.items() if not v]
        print(f"VERDICT: NOT_CONVERGED — still moving beyond {tol:.1%}: {', '.join(moving)}")
        return 1
    if floors:
        print("VERDICT: CONVERGED_DEGENERATE — plateaued, but ON a floor:")
        for f in floors:
            print(f"  - {f}")
        print("  Report the number, but do NOT report it as the method working.")
        return 3
    efficacy_rc = efficacy_verdict(rows, gate["efficacy"], gate, step)
    if efficacy_rc is not None:
        return efficacy_rc
    print(f"VERDICT: CONVERGED — all primaries flat within {tol:.1%} over "
          f"{len(bl) * block} steps, and off every known degenerate floor.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
