"""The only training loop. Every run (R0-A ... R3) is a config, not a new script.

    python -m loom.train       --config configs/r0a.yaml
    python -m loom.train.loop  --config configs/r0a.yaml

Design notes that are load bearing:

* ``build_model(cfg)`` is the **single integration point**. Until Teams A/B/C/E
  land, every module falls back to ``stubs.*``. Swapping a stub for a real module
  is a one-line change inside that function and nothing else in this file moves.

* ``--steps`` is the *schedule horizon* and is identical on every link of a
  chained run. ``--stop_at`` / ``--budget_s`` / ``--safety_s`` / ``--run_dir``
  end *this link* and are excluded from ``config_hash``. Mixing the two makes
  the LR curve depend on how long a job happened to run.

* Every step begins with ``set_step_seed(seed, global_step, rank)``, so a step is
  a pure function of ``(seed, step, rank, params)``. That is what makes resume
  continuity assertable rather than eyeballable.

* Batches are embodiment-homogeneous (PLAN 9). The loop reads
  ``window["embodiment"]`` and dispatches ``q_a`` / ``D_e`` through a ModuleDict.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

# `contracts` and `stubs` live at the repo root, not inside the package.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import contracts as C
import stubs as S
from loom.heads.proposal import argmax_coeff, argmax_coeff_dense_st
from loom.losses.act import sparse_target_ce
from loom.losses.dyn import dyn_loss, ln_cosine_distance
from loom.losses.proposal_bc import (
    proposal_bc_loss, proposal_distill_loss, proposal_sparse_ce_loss,
)
from loom.train import atomic as atomic_mod
from loom.train import ckpt as ckpt_mod
from loom.train import fsdp as fsdp_mod
from loom.train import wandb_util
from loom.train.direct_formal import (
    DirectFormalSchedule,
    evaluate_direct_formal,
    should_evaluate_direct_formal,
)
from loom.train.determinism import (
    enable_determinism, rank_identity, set_global_seed, set_step_seed, torch_generator,
)
from loom.train.preempt import PreemptGuard, write_heartbeat
from loom.train.schedule import (
    BANK_LR_MULT, CosineWithWarmup, EMATarget, FreezeSchedule, SpikeGuard,
    build_optimizer, clip_grad, module_grad_norms,
)

__all__ = [
    "LoomModel", "WindowSampler", "TrainState", "build_model", "load_config",
    "config_hash", "parse_args", "main", "LINK_LOCAL_KEYS", "MODULE_NAMES",
]

#: Top-level trainable modules, in PLAN order. Used for LR groups and freezing.
MODULE_NAMES = ("estimator", "bank", "q_delta", "q_action", "decoder",
                "proposal", "potential")

#: ``data.source`` -> the module whose import registers that source's
#: embodiment(s) in ``contracts.EMBODIMENTS``. Adapters register at import time
#: and nothing in the training path imports one before ``build_model``.
ADAPTER_MODULES = {
    "libero": "loom.data.adapters.libero",
    "robotwin": "loom.data.adapters.robotwin",
}

#: Knobs describing *this link*, not the experiment. Never in the config hash.
LINK_LOCAL_KEYS = ("run_dir", "stop_at", "budget_s", "safety_s", "no_wandb",
                   "allow_reshard", "config_path")

#: How often to measure the per-entry gradient ratio at q_Delta's logits
#: (unselected : selected). It needs the backward pass to have run, so it is not
#: free the way a forward-only statistic is; `optim.grad_probe_every` overrides,
#: 0 turns it off.
GRAD_PROBE_EVERY = 100

# ``Optimizer.state_dict`` mixes Adam moments (which must resume) with the
# experiment's current group recipe (which must not be inherited accidentally).
_OPT_GROUP_CONFIG_KEYS = (
    "name", "module", "lr", "lr_scale", "weight_decay", "betas", "eps",
    "amsgrad", "maximize", "foreach", "capturable", "differentiable", "fused",
)

_CONFIG_DIR = _ROOT / "configs"

_DIRECT_FORMAL_METRICS_ROLLBACK_FORMAT = (
    "loom-direct-formal-metrics-rollback-v1"
)
_FRESH_METRICS_ROLLBACK_FORMAT = "loom-fresh-metrics-rollback-v1"
_EXECUTION_FAILURE_FORMAT = "loom-training-execution-failure-v1"


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════

def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _read_yaml(path: Path) -> dict:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f) or {}


def _resolve(path: str, relative_to: Path) -> Path:
    p = Path(path)
    for cand in (p, relative_to / p, _CONFIG_DIR / p, _ROOT / p):
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(f"config {path!r} not found (looked near {relative_to})")


def read_config(path: str | Path, _seen: tuple[str, ...] = ()) -> dict:
    """YAML with a single-parent ``extends:`` key, deep merged child-over-parent."""
    p = _resolve(str(path), Path.cwd())
    if str(p) in _seen:
        raise ValueError(f"circular extends: {' -> '.join(_seen + (str(p),))}")
    cfg = _read_yaml(p)
    parent = cfg.pop("extends", None)
    if parent:
        cfg = _deep_merge(read_config(_resolve(parent, p.parent), _seen + (str(p),)), cfg)
    return cfg


def _coerce(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def load_config(args) -> dict:
    """Merged config + link-local section. This dict is the whole experiment."""
    cfg = read_config(args.config)
    cfg.setdefault("run", {})
    cfg.setdefault("data", {})
    cfg.setdefault("model", {})
    cfg.setdefault("optim", {})
    cfg.setdefault("losses", {})
    cfg.setdefault("fsdp", {})
    cfg.setdefault("freeze", {})
    cfg.setdefault("train_modules", list(MODULE_NAMES))

    # generic --set a.b=value overrides, applied before the named ones
    for item in args.set or []:
        key, _, val = item.partition("=")
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _coerce(val)

    if args.steps is not None and (
        "schedule_horizon" in cfg["run"] or "max_updates" in cfg["run"]
    ):
        raise ValueError(
            "--steps cannot override a direct-formal config; freeze "
            "run.schedule_horizon and run.max_updates in YAML"
        )
    if args.steps is not None:
        cfg["run"]["steps"] = args.steps
    if args.seed is not None:
        cfg["run"]["seed"] = args.seed
    if args.lr is not None:
        cfg["optim"]["lr"] = args.lr
    if args.batch is not None:
        cfg["data"]["batch_per_gpu"] = args.batch
    if args.ckpt_every is not None:
        cfg["run"]["ckpt_every"] = args.ckpt_every
    if args.log_every is not None:
        cfg["run"]["log_every"] = args.log_every
    if args.deterministic:
        cfg["run"]["deterministic"] = True

    name = cfg["run"].get("name", "loom")
    cfg["link"] = {
        "run_dir": args.run_dir or str(_ROOT / "runs" / name),
        "stop_at": args.stop_at,
        "budget_s": args.budget_s,
        "safety_s": args.safety_s,
        "no_wandb": bool(args.no_wandb),
        "allow_reshard": bool(args.allow_reshard),
        "config_path": str(args.config),
    }
    return cfg


def config_hash(cfg: dict) -> str:
    """Identity of the *experiment*. ``cfg["link"]`` is excluded by construction."""
    d = {k: v for k, v in cfg.items() if k != "link"}
    return hashlib.blake2b(
        json.dumps(d, sort_keys=True, default=str).encode(), digest_size=8
    ).hexdigest()


def _exclusive_publish_bytes(path: Path, payload: bytes) -> None:
    """Publish immutable bytes without an overwrite race.

    A temporary inode is fully written and fsynced before ``link`` gives it the
    final name.  Replaying an interrupted recovery may encounter an existing
    destination; exact bytes are idempotent, while any difference fails closed.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise RuntimeError(f"immutable recovery artifact differs: {path}")
        else:
            atomic_mod.fsync_dir(path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


def _execution_failure_payload(
    *, config_hash_value: str, global_step: int, reason: str,
) -> dict[str, Any]:
    if not isinstance(config_hash_value, str) or not config_hash_value:
        raise ValueError("execution-failure config hash must be non-empty")
    if (
        not isinstance(global_step, int) or isinstance(global_step, bool)
        or global_step < 0
    ):
        raise ValueError("execution-failure global_step must be non-negative")
    if reason != "logging_failure":
        raise ValueError("unsupported durable execution-failure reason")
    return {
        "format": _EXECUTION_FAILURE_FORMAT,
        "config_hash": config_hash_value,
        "global_step": global_step,
        "reason": reason,
    }


def _publish_execution_failure(
    run_dir: Path, *, config_hash_value: str, global_step: int, reason: str,
) -> dict[str, Any]:
    """Idempotently publish a terminal execution failure before checkpointing."""
    path = Path(run_dir) / "EXECUTION_FAILURE.json"
    expected = _execution_failure_payload(
        config_hash_value=config_hash_value,
        global_step=global_step,
        reason=reason,
    )
    encoded = (
        json.dumps(expected, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()
    if not path.exists():
        try:
            _exclusive_publish_bytes(path, encoded)
        except FileExistsError:
            pass
    if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
        raise RuntimeError("durable execution-failure marker differs from expected")
    return {**expected, "path": str(path)}


def _read_execution_failure(
    run_dir: Path, *, config_hash_value: str,
) -> dict[str, Any] | None:
    path = Path(run_dir) / "EXECUTION_FAILURE.json"
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("durable execution-failure marker is nonregular")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("durable execution-failure marker is unreadable") from error
    if not isinstance(value, Mapping) or set(value) != {
        "format", "config_hash", "global_step", "reason",
    }:
        raise RuntimeError("durable execution-failure marker is malformed")
    expected = _execution_failure_payload(
        config_hash_value=config_hash_value,
        global_step=value.get("global_step"),
        reason=value.get("reason"),
    )
    if value != expected:
        raise RuntimeError("durable execution-failure marker identity changed")
    return {**expected, "path": str(path)}


def _reconcile_metrics_to_checkpoint(
    run_dir: Path,
    *,
    checkpoint_step: int,
    checkpoint_identity: Mapping[str, Any],
    rollback_dir_name: str,
    receipt_format: str,
    ledger_label: str,
) -> dict[str, Any]:
    """Authenticate a metrics ledger and remove only a crash tail.

    The checkpoint is the durable optimizer/model authority.  A metrics ledger
    may be ahead of it when a process dies after logging later updates but
    before the next checkpoint.  The only repairable state is one exact row for
    every step ``1..N`` with ``N > checkpoint_step``.  The complete original
    and discarded tail are quarantined under content-addressed names, an
    immutable deterministic rollback receipt is published, and only then is
    ``metrics.jsonl`` atomically replaced by its committed prefix.
    """

    if (
        not isinstance(checkpoint_step, int)
        or isinstance(checkpoint_step, bool)
        or checkpoint_step < 0
    ):
        raise ValueError("checkpoint_step must be a non-negative integer")
    if not isinstance(checkpoint_identity, Mapping):
        raise ValueError("checkpoint_identity must be a mapping")
    if (
        not rollback_dir_name
        or Path(rollback_dir_name).name != rollback_dir_name
        or rollback_dir_name in {".", ".."}
    ):
        raise ValueError("rollback_dir_name must be one plain path component")
    if not receipt_format or not ledger_label:
        raise ValueError("receipt_format and ledger_label must be non-empty")

    metrics_path = Path(run_dir) / "metrics.jsonl"
    if not metrics_path.exists():
        if checkpoint_step == 0:
            return {
                "action": "NONE",
                "checkpoint_step": 0,
                "rows": 0,
                "metrics_sha256": hashlib.sha256(b"").hexdigest(),
            }
        raise RuntimeError(
            f"{ledger_label} metrics ledger is missing behind checkpoint "
            f"{checkpoint_step}"
        )

    original = metrics_path.read_bytes()
    if original and not original.endswith(b"\n"):
        raise RuntimeError(
            f"{ledger_label} metrics ledger has an unterminated final row"
        )

    encoded_lines = original.splitlines(keepends=True)
    rows: list[Mapping[str, Any]] = []

    def _reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant {value!r}")

    for line_number, encoded in enumerate(encoded_lines, start=1):
        if not encoded.endswith(b"\n"):
            raise RuntimeError(
                f"{ledger_label} metrics row {line_number} is not "
                "newline-terminated"
            )
        try:
            row = json.loads(
                encoded.decode("utf-8"), parse_constant=_reject_constant,
            )
        except (UnicodeDecodeError, ValueError) as error:
            raise RuntimeError(
                f"{ledger_label} metrics row {line_number} is malformed: {error}"
            ) from error
        if not isinstance(row, Mapping):
            raise RuntimeError(
                f"{ledger_label} metrics row {line_number} is not a JSON object"
            )
        step = row.get("global_step")
        if (
            not isinstance(step, int)
            or isinstance(step, bool)
            or step != line_number
        ):
            raise RuntimeError(
                f"{ledger_label} metrics ledger is not exactly contiguous: "
                f"row {line_number} has global_step={step!r}"
            )
        rows.append(row)

    if len(rows) < checkpoint_step:
        raise RuntimeError(
            f"{ledger_label} metrics ledger is behind checkpoint: "
            f"{len(rows)} rows for checkpoint {checkpoint_step}"
        )
    if len(rows) == checkpoint_step:
        return {
            "action": "NONE",
            "checkpoint_step": checkpoint_step,
            "rows": len(rows),
            "metrics_sha256": hashlib.sha256(original).hexdigest(),
        }

    prefix = b"".join(encoded_lines[:checkpoint_step])
    tail = b"".join(encoded_lines[checkpoint_step:])
    original_sha = hashlib.sha256(original).hexdigest()
    prefix_sha = hashlib.sha256(prefix).hexdigest()
    tail_sha = hashlib.sha256(tail).hexdigest()
    rollback_dir = Path(run_dir) / rollback_dir_name
    created_dir = not rollback_dir.exists()
    rollback_dir.mkdir(parents=False, exist_ok=True)
    if created_dir:
        atomic_mod.fsync_dir(rollback_dir.parent)
    if not rollback_dir.is_dir():
        raise RuntimeError(f"metrics rollback quarantine is not a directory: {rollback_dir}")

    original_path = rollback_dir / f"metrics.full.sha256-{original_sha}.jsonl"
    tail_path = rollback_dir / f"metrics.tail.sha256-{tail_sha}.jsonl"
    receipt_path = rollback_dir / (
        f"rollback.step-{checkpoint_step:09d}.sha256-{original_sha}.json"
    )
    receipt = {
        "format": receipt_format,
        "reason": "crash_tail_beyond_latest_checkpoint",
        "checkpoint": dict(checkpoint_identity),
        "ledger": {
            "path": "metrics.jsonl",
            "original_rows": len(rows),
            "original_bytes": len(original),
            "original_sha256": original_sha,
            "retained_rows": checkpoint_step,
            "retained_bytes": len(prefix),
            "retained_sha256": prefix_sha,
            "discarded_rows": len(rows) - checkpoint_step,
            "discarded_bytes": len(tail),
            "discarded_sha256": tail_sha,
            "discarded_step_range": [checkpoint_step + 1, len(rows)],
        },
        "quarantine": {
            "full_original": str(original_path.relative_to(run_dir)),
            "discarded_tail": str(tail_path.relative_to(run_dir)),
        },
    }
    receipt_bytes = (
        json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")

    # Receipt-before-replace is intentional: after a crash at any point, the
    # original bytes remain recoverable and replay either completes the exact
    # authorized replacement or verifies that it already happened.
    _exclusive_publish_bytes(original_path, original)
    _exclusive_publish_bytes(tail_path, tail)
    _exclusive_publish_bytes(receipt_path, receipt_bytes)
    atomic_mod.atomic_write_bytes(metrics_path, prefix)
    if metrics_path.read_bytes() != prefix:
        raise RuntimeError(
            f"{ledger_label} metrics atomic rollback verification failed"
        )
    return {"action": "ROLLBACK", "receipt_path": str(receipt_path), **receipt}


def _reconcile_direct_formal_metrics(
    run_dir: Path,
    *,
    checkpoint_step: int,
    checkpoint_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Preserve the authenticated DirectFormal rollback contract exactly."""

    return _reconcile_metrics_to_checkpoint(
        run_dir,
        checkpoint_step=checkpoint_step,
        checkpoint_identity=checkpoint_identity,
        rollback_dir_name="direct_formal_metrics_rollback",
        receipt_format=_DIRECT_FORMAL_METRICS_ROLLBACK_FORMAT,
        ledger_label="formal",
    )


def _reconcile_fresh_metrics(
    run_dir: Path,
    *,
    checkpoint_step: int,
    checkpoint_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconcile a non-DirectFormal fresh lineage without making decisions."""

    return _reconcile_metrics_to_checkpoint(
        run_dir,
        checkpoint_step=checkpoint_step,
        checkpoint_identity=checkpoint_identity,
        rollback_dir_name="fresh_metrics_rollback",
        receipt_format=_FRESH_METRICS_ROLLBACK_FORMAT,
        ledger_label="fresh",
    )


def _optimizer_group_config(optimizer) -> list[dict[str, Any]]:
    """Copy config-owned optimizer fields before checkpoint moments are loaded."""
    return [
        {k: group[k] for k in _OPT_GROUP_CONFIG_KEYS if k in group}
        for group in optimizer.param_groups
    ]


def _reapply_optimizer_group_config(optimizer, wanted: list[dict[str, Any]]) -> None:
    """Keep loaded Adam moments but restore LR/WD/betas from this run's config."""
    if len(optimizer.param_groups) != len(wanted):
        raise RuntimeError(
            "checkpoint optimizer group count differs from the current model: "
            f"{len(optimizer.param_groups)} vs {len(wanted)}"
        )
    for group, configured in zip(optimizer.param_groups, wanted):
        group.update(configured)


def _optimizer_state_reset_modules(optimizer, configured: Any) -> tuple[str, ...]:
    """Validate the config-owned list of optimizer modules to reset.

    Module labels come from the live optimizer groups built *after* FSDP
    wrapping. Using those live groups, rather than serialized parameter IDs or
    model parameter names, also works for FSDP original/sharded parameters.
    """
    if configured is None:
        return ()
    if not isinstance(configured, list):
        raise ValueError(
            "optim.reset_state_modules must be a list of optimizer module names, "
            f"got {configured!r}"
        )
    if any(not isinstance(name, str) or not name for name in configured):
        raise ValueError(
            "optim.reset_state_modules entries must be non-empty strings, "
            f"got {configured!r}"
        )
    if len(configured) != len(set(configured)):
        raise ValueError(
            "optim.reset_state_modules contains duplicate module names: "
            f"{configured!r}"
        )

    available = {group.get("module") for group in optimizer.param_groups}
    unknown = sorted(set(configured) - available)
    if unknown:
        raise ValueError(
            "optim.reset_state_modules names modules with no optimizer group: "
            f"{unknown}; available modules are {sorted(str(x) for x in available)}"
        )
    return tuple(configured)


def _reset_optimizer_state_modules(
        optimizer, modules: Sequence[str]) -> dict[str, int]:
    """Drop Adam state for selected live parameter groups, and nothing else."""
    selected = set(modules)
    cleared = {name: 0 for name in modules}
    for group in optimizer.param_groups:
        module = group.get("module")
        if module not in selected:
            continue
        for parameter in group["params"]:
            if parameter in optimizer.state:
                del optimizer.state[parameter]
                cleared[module] += 1
    return cleared


def _reset_optimizer_state_for_config_transition(
        optimizer, modules: Sequence[str], *, checkpoint_config_hash: str,
        current_config_hash: str) -> dict[str, int] | None:
    """Apply a selective reset once when entering a new config identity.

    The next checkpoint carries ``current_config_hash``. Matching-hash resumes
    are ordinary requeues and must retain moments accumulated after the phase
    transition, so they deliberately return ``None`` without touching state.
    """
    if not modules or checkpoint_config_hash == current_config_hash:
        return None
    return _reset_optimizer_state_modules(optimizer, modules)


def _reset_parameters_for_config_transition(
        model: nn.Module, configured: Any, *, checkpoint_config_hash: str,
        current_config_hash: str) -> dict[str, int] | None:
    """Apply the one declared value reset exactly once at a config transition.

    This deliberately is not a general checkpoint surgery language.  The only
    supported intervention is the audited identity-centred bank contingency:
    zero ``bank.omega`` while retaining ``bank.log_r`` and ``bank.b_raw``.
    Binding the reset to the source experiment hash prevents an accidentally
    seeded run from silently mutating a checkpoint from another method.

    A checkpoint written after the transition carries ``current_config_hash``;
    same-hash requeues therefore return ``None`` and preserve the learned omega
    tensor byte-for-byte.
    """
    if configured is None:
        return None
    if not isinstance(configured, Mapping):
        raise ValueError(
            "optim.transition_parameter_reset must be a mapping, got "
            f"{configured!r}"
        )
    expected_keys = {"source_config_hash", "tensors"}
    if set(configured) != expected_keys:
        raise ValueError(
            "optim.transition_parameter_reset must have exactly "
            f"{sorted(expected_keys)}, got {sorted(str(k) for k in configured)}"
        )
    source_hash = configured["source_config_hash"]
    if not isinstance(source_hash, str) or len(source_hash) != 16 or any(
            ch not in "0123456789abcdef" for ch in source_hash):
        raise ValueError(
            "optim.transition_parameter_reset.source_config_hash must be a "
            f"16-character lowercase hexadecimal config hash, got {source_hash!r}"
        )
    tensors = configured["tensors"]
    declared = {"bank.omega": "zero"}
    if not isinstance(tensors, Mapping) or dict(tensors) != declared:
        raise ValueError(
            "optim.transition_parameter_reset.tensors must be exactly "
            f"{declared!r}, got {tensors!r}"
        )

    # Validate the live model even on a same-hash resume. A typo must not hide
    # indefinitely just because the first transition happened on another link.
    named = dict(model.named_parameters())
    parameter = named.get("bank.omega")
    if parameter is None:
        raise ValueError(
            "optim.transition_parameter_reset names missing parameter bank.omega"
        )

    if checkpoint_config_hash == current_config_hash:
        return None
    if checkpoint_config_hash != source_hash:
        raise ValueError(
            "optim.transition_parameter_reset source mismatch: checkpoint "
            f"config_hash={checkpoint_config_hash!r}, declared source={source_hash!r}"
        )
    with torch.no_grad():
        parameter.zero_()
    return {"bank.omega": int(parameter.numel())}


# ═══════════════════════════════════════════════════════════════════════════
#  TRAINABLE STUB SHIMS
#
#  stubs.py is frozen and its modules are deliberately parameter-free, so a loop
#  built on them would carry no gradient at all: the optimizer would be a no-op,
#  the resume test would be vacuous and the R2 freeze schedule would have nothing
#  to freeze. These shims add one real parameter each and keep the stub's random
#  output, which is exactly enough to exercise every path.
# ═══════════════════════════════════════════════════════════════════════════

class _StubEstimator(S.StubEstimator):
    def __init__(self) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.ones(1))
        self.shift = nn.Parameter(torch.zeros(C.K, C.D))

    def forward(self, feats, z_prev):
        z = super().forward(feats, z_prev)
        return z * self.gain.to(z.dtype) + self.shift.to(z.dtype)


class _StubDecoder(S.StubDecoder):
    """`stubs.StubDecoder` is frozen and predates the `(proprio, c)` contract.

    It never reads its first argument beyond `shape[0]` / `device` / `dtype`, so
    handing it `(B, dof_e)` proprio instead of `(B, K, D)` belief works
    unchanged; only the names here move, so the shape the loop passes matches
    what the REAL decoder now takes.
    """

    def __init__(self, embodiment: str) -> None:
        super().__init__(embodiment)
        self.scale = nn.Parameter(torch.ones(self.dof))

    def loss(
        self,
        proprio: Tensor,
        c: Tensor,
        a_seg: Tensor,
        *,
        t: Tensor | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        # ``t`` and ``noise`` are accepted so the dual-code path can exercise the
        # same paired-CFM call contract in CPU stub tests. The frozen stub decoder
        # is an ordinary MSE shim and therefore has no stochastic flow path to
        # condition; real R0-A runs fail closed on ``model.use_stubs: false``.
        del t
        pred = self.forward(proprio, c) if noise is None else noise
        return ((pred * self.scale.to(proprio.dtype)
                 - a_seg) ** 2).mean()


class _StubQAction(S.StubQAction):
    def __init__(self) -> None:
        super().__init__()
        self.temp = nn.Parameter(torch.zeros(1))

    def forward(self, a_seg: Tensor, z: Tensor) -> Tensor:
        c = super().forward(a_seg, z)
        # straight-through: keeps c exactly on the simplex, carries a gradient
        return c + (self.temp.to(c.dtype) - self.temp.detach().to(c.dtype))


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL
# ═══════════════════════════════════════════════════════════════════════════

class _Bound:
    """Binds one embodiment of a shared per-embodiment head.

    Team C's real ``QAction`` / ``Decoder`` are themselves keyed by embodiment
    (``forward(..., embodiment=)``), while the stubs are one module per body.
    Both are presented to the loop as ``heads[embodiment]``, so the dispatch in
    ``compute_losses`` does not care which it got.
    """

    __slots__ = ("_inner", "_emb")

    def __init__(self, inner: nn.Module, emb: str):
        self._inner, self._emb = inner, emb

    def __call__(self, *a, **kw):
        return self._inner(*a, embodiment=self._emb, **kw)

    def loss(self, *a, **kw):
        return self._inner.loss(*a, embodiment=self._emb, **kw)


class EmbodimentHeads(nn.Module):
    """``heads[embodiment]`` over either a ModuleDict or one shared container."""

    def __init__(self, inner: nn.Module, names: Sequence[str], shared: bool):
        super().__init__()
        self.inner = inner
        self._names = tuple(names)
        self.shared = bool(shared)

    def keys(self):
        return self._names

    def __contains__(self, name: str) -> bool:
        return name in self._names

    def __getitem__(self, name: str):
        if name not in self._names:
            raise KeyError(f"no head for embodiment {name!r}; have {self._names}")
        return _Bound(self.inner, name) if self.shared else self.inner[name]


def _import(module_path: str):
    try:
        return importlib.import_module(module_path), None
    except Exception as e:              # not written yet, or broken mid-edit
        return None, e


def _try_build(module_path: str, class_names: Sequence[str], kwargs: dict,
               fallback, mode: str, what: str):
    """Import a real module if it exists, otherwise fall back to a stub.

    ``mode`` is ``model.use_stubs``:
      ``True``  always stubs -- no real import is attempted at all
      ``"auto"`` real if importable, stub otherwise (the default)
      ``False`` real required; a missing module is a hard error
    """
    if mode is True:
        return fallback(), "stub"
    mod, err = _import(module_path)
    if mod is not None:
        for cn in class_names:
            cls = getattr(mod, cn, None)
            if cls is None:
                continue
            try:
                return cls(**kwargs), "real"
            except TypeError as e:
                err = e
                try:
                    return cls(), "real"
                except Exception as e2:
                    err = e2
            except Exception as e:
                err = e
    if mode is False:
        raise RuntimeError(
            f"model.use_stubs is false but {what} could not be built from "
            f"{module_path}: {err!r}"
        )
    return fallback(), "stub"


def _build_heads(module_path: str, class_names: Sequence[str], kwargs: dict,
                 embodiments: Sequence[str], stub_factory, mode: str,
                 what: str) -> tuple[EmbodimentHeads, str]:
    """Per-embodiment heads (``q_a``, ``D_e``), whichever shape the real one has."""
    if mode is not True:
        mod, err = _import(module_path)
        if mod is not None:
            for cn in class_names:
                cls = getattr(mod, cn, None)
                if cls is None:
                    continue
                try:                     # Team C's container form
                    return EmbodimentHeads(
                        cls(embodiments=list(embodiments), **kwargs),
                        embodiments, shared=True), "real"
                except Exception as e:
                    err = e
        if mode is False:
            raise RuntimeError(
                f"model.use_stubs is false but {what} could not be built from "
                f"{module_path}: {err!r}"
            )
    d = nn.ModuleDict({e: stub_factory(e) for e in embodiments})
    return EmbodimentHeads(d, embodiments, shared=False), "stub"


def build_model(cfg: dict) -> "LoomModel":
    """THE integration point. Real modules if importable, ``stubs.*`` otherwise.

    Integration order is B -> C -> A -> E -> D -> F (PLAN 6.2); each team's module
    starts being used the moment it imports cleanly, with no edit here.

    ``model.use_stubs``
      ``true``   never import a real module. What ``tests/test_train.py`` uses:
                 a 150 M Perceiver takes minutes per CPU step and would make the
                 login-node suite unrunnable.
      ``"auto"`` real if importable, stub otherwise. The default.
      ``false``  a missing real module is a hard error. Set this in the R0-A
                 config once Phase 2 starts, so a typo cannot silently train a
                 random-output stub for eight hours.

    The four loss terms are computed in :meth:`LoomModel.compute_losses` straight
    from PLAN 4.C. ``L_dyn`` now calls Team C's ``loom/losses/dyn.py::dyn_loss``
    -- the inline version it replaced had no negatives at all, so the configured
    ``dyn.negatives: within_trajectory`` was inert and a constant ``z`` drove the
    term to zero for free. ``L_act`` / ``L_proposal`` stay inline: they are one
    call to ``Decoder.loss`` and one to ``Proposal.log_prob`` respectively, and
    the loop needs the per-horizon decomposition that the wrappers hide.
    """
    mcfg = dict(cfg.get("model", {}))
    mode = mcfg.get("use_stubs", "auto")
    embodiments = list(cfg.get("data", {}).get("embodiments", ["libero_franka"]))
    # contracts.py registers `libero_franka` itself; every other body is
    # registered by its adapter's import side effect. build_model runs BEFORE
    # build_sampler, which is the only other thing that imports an adapter, so
    # configs/r0b.yaml died at startup on all 16 ranks with "unregistered
    # embodiment 'robotwin_aloha'" (job 32394843). Guarded on the miss so the
    # LIBERO path imports nothing new and its behaviour is bit-identical.
    if any(e not in C.EMBODIMENTS for e in embodiments):
        mod = ADAPTER_MODULES.get(str(cfg.get("data", {}).get("source", "")))
        if mod:
            importlib.import_module(mod)
    for e in embodiments:
        if e not in C.EMBODIMENTS:
            raise ValueError(f"unregistered embodiment {e!r}; adapters register at import")

    estimator, k_est = _try_build(
        "loom.model.estimator", ("Estimator", "PerceiverEstimator", "LoomEstimator"),
        dict(mcfg.get("estimator", {})), _StubEstimator, mode, "estimator")
    bank, k_bank = _try_build(
        "loom.model.bank", ("OperatorBank", "Bank", "LoomBank"),
        dict(mcfg.get("bank", {})), S.StubBank, mode, "bank")
    q_delta, k_qd = _try_build(
        "loom.heads.q_delta", ("QDelta", "QDeltaHead"),
        dict(mcfg.get("q_delta", {})), S.StubQDelta, mode, "q_delta")
    proposal, k_prop = _try_build(
        "loom.heads.proposal", ("Proposal", "ProposalHead", "PolicyProposal"),
        dict(mcfg.get("proposal", {})), S.StubProposal, mode, "proposal")

    q_action, k_qa = _build_heads(
        "loom.heads.q_action", ("QAction", "QActionHead"),
        dict(mcfg.get("q_action", {})), embodiments, lambda e: _StubQAction(),
        mode, "q_action")
    decoder, k_dec = _build_heads(
        "loom.heads.decoder", ("Decoder", "DecoderHead", "FlowDecoder"),
        dict(mcfg.get("decoder", {})), embodiments, _StubDecoder, mode, "decoder")

    potential = None
    k_pot = "off"
    if cfg.get("losses", {}).get("potential", {}).get("enabled", False):
        potential, k_pot = _try_build(
            "loom.heads.potential", ("Potential", "PotentialHead"),
            dict(mcfg.get("potential", {})), S.StubPotential, mode, "potential")

    print("[build] " + " ".join(
        f"{n}={k}" for n, k in [("E", k_est), ("bank", k_bank), ("q_delta", k_qd),
                                ("q_action", k_qa), ("decoder", k_dec),
                                ("proposal", k_prop), ("potential", k_pot)]), flush=True)

    return LoomModel(estimator=estimator, bank=bank, q_delta=q_delta,
                     q_action=q_action, decoder=decoder, proposal=proposal,
                     potential=potential, cfg=cfg)


def _accepts(module, name: str) -> bool:
    """Does ``module.forward`` take a keyword called ``name``?"""
    import inspect

    try:
        return name in inspect.signature(module.forward).parameters
    except (TypeError, ValueError):
        return False


def _cos_dist(a: Tensor, b: Tensor) -> Tensor:
    """1 - cos(LN(a), LN(b)), averaged over slots and batch. PLAN 4.C.

    Exactly `losses.dyn.ln_cosine_distance(a, b, "per_slot").mean()`, kept as a
    local so the diagnostics below read the same as they did before `L_dyn`
    itself moved into `loom/losses/dyn.py`, and so `delta_op` stays comparable
    with every run logged before that.
    """
    return ln_cosine_distance(a, b, "per_slot").mean()


def _coeff_and_logits(head, *args, **kw) -> tuple[Tensor, Tensor | None]:
    """`c` and, when the head exposes them, the DENSE logits behind it.

    Team C's `QDelta` / `QAction` take `return_logits=True`; the frozen stubs do
    not. The Switch load-balancing term needs the dense router distribution
    (`P_m`), not the sparse `c` -- with `P_m` read off `c` the gradient on an
    operator that is in no support is exactly zero, which is the closed path the
    term exists to open. `None` here means "stub", and `_switch_balance` falls
    back to `c`.
    """
    try:
        out = head(*args, return_logits=True, **kw)
    except TypeError:
        return head(*args, **kw), None
    if isinstance(out, tuple):
        return out[0], out[1]
    return out, None


def _switch_balance(c: Tensor, logits: Tensor | None, topk: int = C.TOPK) -> Tensor:
    """The Switch auxiliary load-balancing loss,  `M * sum_m f_m P_m`.

    * `f_m` -- the fraction of ROUTING SLOTS that went to operator `m`. One
      token contributes `TOPK` slots (its hard top-4 support), so `sum_m f_m`
      is 1 by construction and this reduces to Switch's own definition at
      `TOPK = 1`. Non-differentiable and detached; it is a count.
    * `P_m` -- the mean DENSE router probability for `m`, `softmax(logits)`
      averaged over tokens. `sum_m P_m == 1`.

    Range: `1.0` when both are uniform (the degenerate floor -- read it as a
    plateau, not as progress) up to `M / TOPK = 32` when every token routes to
    the same four operators and the router is certain about it.

    What it replaces and why (owner's call; the measurement behind it):
    `KL(mean_batch(c) || uniform(M))` is a function of `c` alone, and `c` is
    exactly zero for every operator outside the top-4 support -- so the whole
    term sees only the hard routing decision, never how nearly an operator was
    chosen. Its gradient still reaches an unselected logit (`topk_simplex_st`
    returns `hard + soft - soft.detach()`, so the backward is dense) but
    measured on the R0-A checkpoints it arrived at 0.0006 (ctrl) / 0.0001
    (zinit) of a selected operator's per-entry size. The Switch form is a
    function of the DENSE router as well as of the routing, so the loss changes
    when an out-of-support logit moves at all -- and its coefficient went
    3e-3 -> 1e-2 with it, which is the part that is unambiguously a magnitude
    change. `grad_ratio/q_delta_logits` is logged so this is read and not
    assumed.

    Gradient, for the record:
        dL/dl_{t,m} = (M/T) * (f_m - sum_j f_j P_{t,j}) * P_{t,m}
    negative -- i.e. pushing the logit UP -- for every operator whose load is
    below the load-weighted average, including the ones at exactly zero.
    """
    m = c.shape[-1]
    k = min(topk, m)
    cf = c.detach().float().reshape(-1, m)          # `f` is a count: detached
    t = cf.shape[0]
    idx = cf.topk(k, dim=-1).indices
    f = torch.zeros_like(cf).scatter_(1, idx, 1.0).sum(0) / float(t * k)
    # `P` carries the gradient. The stub path has no dense router to read, so
    # it falls back to `c` -- which is the OLD behaviour and is why the real
    # heads are asked for `return_logits`.
    dense = (torch.softmax(logits.float().reshape(-1, m), dim=-1)
             if logits is not None else c.float().reshape(-1, m))
    return float(m) * (f * dense.mean(0)).sum()


def _usage(cs: Sequence[Tensor]) -> tuple[float, float]:
    """(operators used, usage entropy in nats) for one head's coefficients.

    Aggregate statistics hide structure (CLAUDE.md), which is why this is
    reported per HEAD rather than pooled: `q_a` and `q_Delta` had 7 and 19
    operators alive respectively on the R0-A checkpoints, and one pooled number
    cannot say that. `1e-4` is the same liveness threshold the pooled
    `bank/live_ops` used, so the two stay comparable.
    """
    u = torch.stack(list(cs), 0).detach().float().flatten(0, -2).mean(0)
    u = u / u.sum().clamp_min(1e-12)
    return (float((u > 1e-4).sum()),
            float(-(u * u.clamp_min(1e-12).log()).sum()))


class LoomModel(nn.Module):
    """Everything that carries a gradient, plus the EMA target ``L_dyn`` needs.

    Child names are exactly :data:`MODULE_NAMES` plus ``ema``, because
    ``schedule.param_groups`` and ``FreezeSchedule`` address modules by name and
    ``fsdp.wrap_for_training`` wraps them by name.
    """

    def __init__(self, *, estimator, bank, q_delta, q_action, decoder, proposal,
                 potential=None, cfg: dict):
        super().__init__()
        self.estimator = estimator
        self.bank = bank
        self.q_delta = q_delta
        self.q_action = q_action
        self.decoder = decoder
        self.proposal = proposal
        if potential is not None:
            self.potential = potential
        self.ema = EMATarget(estimator, tau=float(cfg.get("optim", {}).get(
            "ema_tau", C.EMA_TAU)))
        update_ema = cfg.get("optim", {}).get("update_ema", True)
        if not isinstance(update_ema, bool):
            raise ValueError(
                f"optim.update_ema must be a boolean, got {update_ema!r}"
            )
        # A frozen-coordinate refinement must be able to hold both estimator
        # copies exactly fixed. An EMA update is a parameter mutation even when E
        # is optimizer-frozen, so LR scales cannot express this switch.
        self.update_ema = update_ema
        self.cfg = cfg
        lc = cfg.get("losses", {})
        self.loss_cfg = {k: dict(v) for k, v in lc.items()}
        dcfg = self.loss_cfg.get("dyn", {})
        self.negatives = dcfg.get("negatives", "within_trajectory")
        # Action-labelled data knows which body motion produced each transition.
        # R0 historically ignored that label in L_dyn and always routed the bank
        # with q_delta. Keep that action-free default, but make the labelled
        # operator source an explicit, config-hashed method choice.
        self.dyn_coeff_source = str(dcfg.get("coeff_source", "q_delta"))
        if self.dyn_coeff_source not in ("q_delta", "q_action"):
            raise ValueError(
                "losses.dyn.coeff_source must be 'q_delta' or 'q_action', got "
                f"{self.dyn_coeff_source!r}"
            )
        detach_coeff = dcfg.get("detach_coeff", True)
        if not isinstance(detach_coeff, bool):
            raise ValueError(
                "losses.dyn.detach_coeff must be a boolean, got "
                f"{detach_coeff!r}"
            )
        # The historical and bank-only action anchor is a frozen semantic label.
        # A joint action-code/bank stage may make that one existing L_dyn edge
        # live explicitly; the default remains detached so old recipes and active
        # runs retain byte-for-byte gradient routing.
        self.dyn_detach_coeff = detach_coeff
        #: which side of ``L_act``'s align term carries the gradient.
        #: ``"q_delta"`` (the original) -- ``q_a`` regresses onto ``sg(q_Delta)``.
        #: ``"q_a"``     (ALIGN-FLIP)   -- ``q_Delta`` regresses onto ``sg(q_a)``,
        #: so ``c_a`` is not pulled toward the transition head. A recipe may
        #: additionally route L_dyn with q_a on action-labelled data.
        self.align_to = str(self.loss_cfg.get("act", {}).get("align_to", "q_delta"))
        if self.align_to not in ("q_delta", "q_a"):
            raise ValueError(
                f"losses.act.align_to must be 'q_delta' or 'q_a', got {self.align_to!r}")
        act_cfg = self.loss_cfg.get("act", {})
        self.align_mode = str(act_cfg.get("align_mode", "mse"))
        if self.align_mode not in ("mse", "sparse_ce"):
            raise ValueError(
                "losses.act.align_mode must be 'mse' or 'sparse_ce', got "
                f"{self.align_mode!r}"
            )
        if self.align_mode == "sparse_ce" and self.align_to != "q_a":
            raise ValueError(
                "sparse_ce alignment is defined only for q_delta <- q_action; "
                "set losses.act.align_to='q_a'"
            )
        self.align_temperature = float(act_cfg.get("align_temperature", 1.0))
        self.align_weight = float(act_cfg.get("align_weight", 1.0))
        if (
            not math.isfinite(self.align_temperature)
            or self.align_temperature <= 0.0
            or not math.isfinite(self.align_weight)
            or self.align_weight < 0.0
        ):
            raise ValueError(
                "act align_temperature must be >0 and align_weight must be >=0"
            )
        balance_cfg = self.loss_cfg.get("balance", {})
        self.balance_mode = str(balance_cfg.get("mode", "pooled"))
        if self.balance_mode not in ("pooled", "per_head"):
            raise ValueError(
                "losses.balance.mode must be 'pooled' or 'per_head', got "
                f"{self.balance_mode!r}"
            )
        raw_head_weights = balance_cfg.get(
            "head_weights", {"q_delta": 0.5, "q_action": 0.5},
        )
        if not isinstance(raw_head_weights, Mapping):
            raise ValueError("losses.balance.head_weights must be a mapping")
        self.balance_head_weights = {
            str(name): float(value) for name, value in raw_head_weights.items()
        }
        if set(self.balance_head_weights) - {"q_delta", "q_action"}:
            raise ValueError("balance head_weights names must be q_delta/q_action")
        if any(
            not math.isfinite(value) or value < 0.0
            for value in self.balance_head_weights.values()
        ):
            raise ValueError("balance head weights must be finite and non-negative")
        # Which coefficient D_e sees while learning to realize the demonstrated
        # action.  The historical path uses q_a's action-conditioned teacher.
        # ``proposal`` uses the exact sparse coefficient deployed by R0.
        # ``dual_q_action_proposal`` averages paired teacher/deployed CFM terms,
        # preserving q_a's action-semantic anchor while closing that exposure gap.
        # Neither adds an input: D_e still receives only (proprio, c).
        self.act_decode_from = str(
            self.loss_cfg.get("act", {}).get("decode_from", "q_action")
        )
        if self.act_decode_from not in (
            "q_action", "proposal", "dual_q_action_proposal",
        ):
            raise ValueError(
                "losses.act.decode_from must be 'q_action', 'proposal', or "
                "'dual_q_action_proposal', got "
                f"{self.act_decode_from!r}"
            )
        if (
            self.act_decode_from == "dual_q_action_proposal"
            and not callable(getattr(self.proposal, "logits", None))
        ):
            raise TypeError(
                "losses.act.decode_from='dual_q_action_proposal' requires a real "
                "proposal.logits(z, lang) surface; stub proposal mode is unsupported. "
                "Set model.use_stubs=false for this method."
            )
        pcfg = self.loss_cfg.get("proposal", {})
        self.proposal_mode = str(pcfg.get("mode", "pl"))
        if self.proposal_mode not in ("pl", "dense_kl", "sparse_ce"):
            raise ValueError(
                "losses.proposal.mode must be 'pl', 'dense_kl', or 'sparse_ce', got "
                f"{self.proposal_mode!r}"
            )
        self.proposal_temperature = float(pcfg.get("temperature", 1.0))
        if self.proposal_temperature <= 0.0:
            raise ValueError("losses.proposal.temperature must be > 0")
        self.proposal_detach_belief = bool(pcfg.get("detach_belief", False))
        self.proposal_hard_weight = float(pcfg.get("hard_weight", 0.0))
        if not math.isfinite(self.proposal_hard_weight) or self.proposal_hard_weight < 0.0:
            raise ValueError("losses.proposal.hard_weight must be finite and >= 0")
        if self.proposal_mode != "dense_kl" and self.proposal_hard_weight != 0.0:
            raise ValueError(
                "losses.proposal.hard_weight is only valid with mode='dense_kl'"
            )
        # Team B's estimator holds per-embodiment proprio projections and infers
        # the body from proprio.shape[-1] when not told. Two registered bodies
        # already share dof=7, so inference is ambiguous and picking the wrong
        # projection is a trains-fine-scores-zero bug. The window names the body;
        # pass it through whenever the estimator can accept it.
        self._est_takes_embodiment = _accepts(estimator, "embodiment")
        #: set by main() on the CUDA path; PLAN 9 build assert, see fsdp.assert_bf16
        self.check_bf16 = False
        #: bf16 on CUDA, None on CPU. See `_cast`.
        self.compute_dtype: torch.dtype | None = None
        #: set by main() on the steps that measure the q_Delta logit-grad ratio
        self._probe_grad = False
        #: this step's dense q_Delta logits, one per horizon (None on the stub
        #: path). Rebuilt every `compute_losses`, read once after `backward`.
        self._qd_logits: list[Tensor | None] = []

    # ── beliefs ────────────────────────────────────────────────────────────
    def _cast(self, z: Tensor) -> Tensor:
        """Pin the belief to the compute dtype at the estimator boundary.

        Autocast is not enough on its own, and the reason is worth writing down.
        `E` is a pre-LN Perceiver, so its last op is a LayerNorm -- and
        `layer_norm` sits in autocast's **fp32** cast policy, so `z` comes back
        fp32 even with every matmul in the block running bf16. `bank.step` then
        does `a * x` with `a` bf16 (einsum, autocast-lowered) and `x` fp32, which
        promotes the whole affine rollout to fp32: 2x the activation memory for
        the one part of the model that is pure elementwise algebra.

        The cast belongs here rather than inside the bank because the bank casts
        `c` to its *parameter* dtype and hands back that, so the call site is the
        only place that controls the belief's dtype. It is also correct under
        FSDP, whose fp32 master weights are a storage detail: the forward already
        runs on bf16 shards, and the LayerNorm upcast happens either way.
        """
        return z if self.compute_dtype is None else z.to(self.compute_dtype)

    def _est_kw(self, window: dict) -> dict:
        return {"embodiment": window["embodiment"]} if self._est_takes_embodiment else {}

    def update_target(self) -> None:
        """Apply the configured EMA mutation, or hold target coordinates fixed."""
        if self.update_ema:
            self.ema.update(self.estimator)

    def beliefs(self, window: dict) -> list[Tensor]:
        kw = self._est_kw(window)
        z, out = None, []
        prefix = window.get("burn_in_feats", ())
        if prefix:
            # Replay real preceding observations but truncate the graph at the
            # selected window boundary. The unchanged five main states below are
            # the only states through which a future estimator-enabled recipe may
            # backpropagate.
            with torch.no_grad():
                for feats in prefix:
                    z = self._cast(self.estimator(feats, z, **kw))
            z = z.detach()
        for feats in window["feats"]:
            z = self._cast(self.estimator(feats, z, **kw))
            out.append(z)
        return out

    @torch.no_grad()
    def target_beliefs(self, window: dict) -> list[Tensor]:
        kw = self._est_kw(window)
        z, out = None, []
        # The target replays its own prefix from its own episode-start state. Do
        # not seed it from the online prefix: online and EMA coordinates can differ.
        for feats in window.get("burn_in_feats", ()):
            z = self._cast(self.ema(feats, z, **kw))
        for feats in window["feats"]:
            z = self._cast(self.ema(feats, z, **kw))
            out.append(z.detach())
        return out

    # ── the four losses ────────────────────────────────────────────────────
    def compute_losses(self, window: dict, step: int, rank: int,
                       seed: int) -> tuple[Tensor, dict[str, float]]:
        emb = window["embodiment"]
        # `.keys()` and not `emb in self.q_action`: an FSDP wrapper forwards
        # __getitem__ but not __contains__, so `in` falls back to the old
        # integer-iteration protocol and raises KeyError("... 0"). Measured on 2
        # A100s. The heads are unwrapped now, but the idiom stays safe either way.
        if emb not in tuple(self.q_action.keys()):
            raise KeyError(
                f"batch embodiment {emb!r} has no q_a/D_e; data.embodiments is "
                f"{list(self.q_action.keys())}. Batches are embodiment-homogeneous "
                f"(PLAN 9), so this is a loader/config mismatch, not a padding issue."
            )
        fsdp_mod.assert_features_are_cached(window)

        zs = self.beliefs(window)                      # online,  N_STATES
        zts = self.target_beliefs(window)              # EMA target, stop-grad
        dev = zs[0].device
        metrics: dict[str, float] = {}
        terms: dict[str, Tensor] = {}
        zero = torch.zeros((), device=dev)

        # coefficients from q_Delta -- action-free, available on every dataset.
        # The dense logits come back too: `L_balance` needs the router
        # distribution, and the grad probe needs a tensor to hang `retain_grad`
        # on. `_qd_logits` is cleared every step so a stale graph cannot be read.
        self._qd_logits = []
        c_delta = []
        need_delta_coeff = (
            self.dyn_coeff_source == "q_delta" or self._on("act", step) or
            self._on("proposal", step) or self._on("balance", step)
        )
        if need_delta_coeff:
            for h in range(C.DEPTH):
                c_h, lg_h = _coeff_and_logits(self.q_delta, zs[h], zts[h + 1])
                c_delta.append(c_h)
                self._qd_logits.append(lg_h)
        # A head-only refinement can freeze q_delta while retaining the probe
        # cadence inherited from the base recipe. Frozen logits have no graph;
        # skip their diagnostic hook instead of aborting a valid refinement.
        if self._probe_grad and all(
                lg is not None and lg.requires_grad for lg in self._qd_logits):
            for lg in self._qd_logits:
                lg.retain_grad()

        # Compute q_a exactly once when either enabled objective needs it. Keep
        # this separate from c_act: proposal/balance historically consume q_a
        # only when L_act is enabled, and selecting a dynamics source must not
        # silently change those objectives too.
        c_action_all: list[Tensor] | None = None
        action_logits_all: list[Tensor | None] = []
        actions = window.get("actions")
        action_dyn = self._on("dyn", step) and self.dyn_coeff_source == "q_action"
        if action_dyn and actions is None:
            raise ValueError(
                "losses.dyn.coeff_source='q_action' requires labelled action "
                "segments; TransitionWindow.actions is None"
            )
        need_action_coeff = actions is not None and (self._on("act", step) or action_dyn)
        if need_action_coeff:
            qa = self.q_action[emb]

            def _action_coefficients():
                coeffs, logits = [], []
                for h in range(C.DEPTH):
                    c_a, lg_a = _coeff_and_logits(qa, actions[:, h], zs[h])
                    coeffs.append(c_a)
                    logits.append(lg_a)
                return coeffs, logits

            action_dyn_needs_grad = action_dyn and not self.dyn_detach_coeff
            if self._on("act", step) or action_dyn_needs_grad:
                # One shared q_a forward serves L_act and action-anchored L_dyn.
                # Each objective decides below whether its view remains attached.
                c_action_all, action_logits_all = _action_coefficients()
            else:
                # Bank-only action anchoring never needs a q_a/z graph at all.
                with torch.no_grad():
                    c_action_all, action_logits_all = _action_coefficients()

        # ── L_dyn ──────────────────────────────────────────────────────────
        #
        # `loom/losses/dyn.py`, not an inline cosine. The inline version had NO
        # negatives at all: a plain `1 - cos(A(c)z, z+)` that a constant `z` and
        # `A(c) ~ I` drive to zero for free while `c` carries nothing. Every
        # config has asked for `negatives: within_trajectory` since R0-A and
        # nothing was reading it.
        if self._on("dyn", step):
            dcfg = self.loss_cfg.get("dyn", {})
            use_q_action = self.dyn_coeff_source == "q_action"
            if use_q_action:
                assert c_action_all is not None  # checked loudly above
                c_dyn = (
                    [c.detach() for c in c_action_all]
                    if self.dyn_detach_coeff else c_action_all
                )
            else:
                c_dyn = c_delta
            c_seq = torch.stack(c_dyn, dim=1)                         # (B,DEPTH,M)
            z_tgt = torch.stack([zts[h + 1] for h in range(C.DEPTH)], dim=1)
            cosine = str(dcfg.get("cosine", "per_slot"))
            out = dyn_loss(
                self.bank, zs[0], c_seq, z_tgt,
                negatives=self.negatives,
                min_gap=int(dcfg.get("min_gap", 2)),
                neg_weight=float(dcfg.get("neg_weight", 1.0)),
                neg_margin=float(dcfg.get("neg_margin", 0.1)),
                weights=tuple(float(value) for value in dcfg.get(
                    "weights", C.DYN_WEIGHTS,
                )),
                cosine=cosine,
                # CPU generator: the negatives' multinomial and the delta_op
                # draw both happen where it lives and the indices move to the
                # coefficients' device. See `losses.dyn._draw`.
                generator=torch_generator(seed, step, rank, tag="dyn"),
                z_contexts=torch.stack(zs[:C.DEPTH], dim=1),
                z_target_prev=torch.stack(zts[:C.DEPTH], dim=1),
                state_weight=float(dcfg.get("state_weight", 1.0)),
                effect_weight=float(dcfg.get("effect_weight", 0.0)),
                contrastive_weight=float(dcfg.get("contrastive_weight", 0.0)),
                contrastive_temperature=float(
                    dcfg.get("contrastive_temperature", 0.1)
                ),
                contrastive_negatives=int(dcfg.get("contrastive_negatives", 4)),
            )
            if self.check_bf16:
                fsdp_mod.assert_bf16(out["z_hat1"], "bank.step output (L_dyn rollout)")
            terms["dyn"] = out["loss"]
            metrics["dyn/pos"] = float(out["dyn"])
            metrics["dyn/neg"] = float(out["neg"])
            metrics["dyn/cos_pos"] = float(out["cos_pos"])
            # Third-party/legacy test doubles implementing the historical
            # ``dyn_loss`` surface do not have the repair-only diagnostics.
            # Preserve that callable contract while the real implementation
            # always supplies all five keys. These fallbacks affect logging
            # only; ``terms['dyn']`` above remains the authoritative scalar.
            metrics["dyn/state"] = float(out.get("state", out["dyn"]))
            metrics["dyn/effect"] = float(out.get("effect", zero.detach()))
            metrics["dyn/contrastive"] = float(
                out.get("contrastive", zero.detach())
            )
            metrics["dyn/effect_gap"] = float(
                out.get("effect_gap", zero.detach())
            )
            metrics["dyn/contrastive_top1"] = float(
                out.get("contrastive_top1", zero.detach())
            )
            # The build assert, unchanged, so the number stays comparable with
            # every prior run. Delta_op says the BANK is alive; it is not the
            # discrimination test -- `Delta_sel` below is.
            metrics["delta_op"] = self._delta_op(zs, zts, c_dyn, step, rank, seed)
            metrics.update(self._delta_sel(zs, zts, c_dyn))

        # ── L_act ──────────────────────────────────────────────────────────
        c_act: list[Tensor] | None = None
        c_act_lg: list[Tensor | None] = []
        if self._on("act", step) and actions is not None:
            if c_action_all is None:
                raise RuntimeError("L_act requires action coefficients")
            dec = self.decoder[emb]
            c_act = c_action_all
            c_act_lg = action_logits_all
            l_act, l_align = zero, zero
            l_decode_teacher, l_decode_deployed = zero, zero
            align_overlap = zero
            deploy_l2, deploy_overlap = zero, zero
            dual_decode = self.act_decode_from == "dual_q_action_proposal"
            # One rank/step stream, consumed in fixed horizon order. Each horizon
            # draws one CFM source and one time and passes those exact tensors to
            # both code paths, so their difference is coefficient conditioning,
            # not Monte-Carlo noise. An explicit device generator is required:
            # CPU generators cannot drive CUDA ``randn``/``rand`` kernels.
            act_cfm_generator = (
                torch_generator(
                    seed, step, rank, tag="act_dual_cfm", device=str(dev),
                )
                if dual_decode else None
            )
            for h in range(C.DEPTH):
                a_seg = actions[:, h]                           # (B, H_OP, dof_e)
                c_a = c_act[h]
                # D_e(proprio_t, c) -- NOT D_e(z, c). Given the whole belief the
                # decoder is a behaviour-cloning head and needs nothing from `c`,
                # so `L_act` exerts no pressure on the coefficient (measured:
                # act/decode 0.2489 -> 0.0559 while c_a held 2-3 distinct top-4
                # supports over 64 real windows). `feats[h]["proprio"]` is
                # (B, dof_e) -- ONE timestep, at the START of segment h.
                proprio = window["feats"][h]["proprio"]
                c_deployed = None
                if self.act_decode_from in ("proposal", "dual_q_action_proposal"):
                    # Same forward value as Proposal.argmax: top-4 support and
                    # restricted-softmax weights.  The straight-through tail is
                    # only a gradient estimator, allowing the existing L_act to
                    # teach pi_c which deployed coefficient realizes the action.
                    p_lg = self.proposal.logits(zs[h], window["lang"])
                    c_deployed = (
                        argmax_coeff_dense_st(p_lg, C.TOPK, p_lg.shape[-1])
                        if dual_decode else
                        argmax_coeff(
                            p_lg, C.TOPK, p_lg.shape[-1], straight_through=True,
                        )
                    )
                    with torch.no_grad():
                        deploy_l2 = deploy_l2 + (
                            (c_deployed.float() - c_a.detach().float()).pow(2)
                            .sum(-1).mean()
                        )
                        ai = c_a.detach().float().topk(C.TOPK, dim=-1).indices
                        pi = c_deployed.detach().float().topk(C.TOPK, dim=-1).indices
                        deploy_overlap = deploy_overlap + (
                            (pi.unsqueeze(-1) == ai.unsqueeze(-2))
                            .any(-1).float().mean()
                        )
                if dual_decode:
                    assert c_deployed is not None
                    assert act_cfm_generator is not None
                    # Decoder.loss first casts the data side to proprio.dtype, so
                    # draw the explicit shared source in that same dtype. ``t`` is
                    # fp32 exactly as Decoder.loss's internal default draw.
                    shared_noise = torch.randn(
                        tuple(a_seg.shape), device=a_seg.device,
                        dtype=proprio.dtype, generator=act_cfm_generator,
                    )
                    shared_t = torch.rand(
                        a_seg.shape[0], device=a_seg.device, dtype=torch.float32,
                        generator=act_cfm_generator,
                    )
                    teacher_h = dec.loss(
                        proprio, c_a, a_seg, t=shared_t, noise=shared_noise,
                    )
                    deployed_h = dec.loss(
                        proprio, c_deployed, a_seg,
                        t=shared_t, noise=shared_noise,
                    )
                    l_decode_teacher = l_decode_teacher + teacher_h
                    l_decode_deployed = l_decode_deployed + deployed_h
                    # Averaging retains the one-decoder-term scale of R0-A. The
                    # alignment below remains a single full-weight semantic edge.
                    l_act = l_act + 0.5 * (teacher_h + deployed_h)
                else:
                    c_decode = c_deployed if c_deployed is not None else c_a
                    l_act = l_act + dec.loss(proprio, c_decode, a_seg)
                # ALIGN. One term, one direction, chosen by `losses.act.align_to`.
                #
                # "q_delta" (original): q_a regresses onto sg(q_Delta) -- one
                #   coefficient space by construction, q_Delta defines it.
                # "q_a" (ALIGN-FLIP): q_Delta regresses onto sg(q_a) instead.
                #   In the direct-policy recipe q_a learns action reconstruction;
                #   an action-labelled dynamics recipe may also route L_dyn with
                #   this same coefficient. The original direction
                #   transmits q_Delta's phase clock INTO q_a: at the observed
                #   plateau the align gradient on c_a has norm 2*sqrt(0.500) =
                #   1.415 and is 100% common-mode, against decode's 0.179 of
                #   which 94% is example-dependent, and q_a duly went blind
                #   (frac_var_a 0.988 fresh -> 0.0015 trained).
                if self.align_to == "q_a":
                    if self.align_mode == "sparse_ce":
                        qd_logits = self._qd_logits[h]
                        if qd_logits is None:
                            raise RuntimeError(
                                "sparse_ce alignment requires dense q_delta logits"
                            )
                        l_align = l_align + sparse_target_ce(
                            qd_logits, c_a,
                            temperature=self.align_temperature,
                        )
                        with torch.no_grad():
                            qi = qd_logits.float().topk(C.TOPK, dim=-1).indices
                            ai = c_a.detach().float().topk(C.TOPK, dim=-1).indices
                            align_overlap = align_overlap + (
                                (qi.unsqueeze(-1) == ai.unsqueeze(-2))
                                .any(-1).float().mean()
                            )
                    else:
                        l_align = l_align + (
                            (c_delta[h] - c_a.detach()) ** 2
                        ).sum(-1).mean()
                else:
                    l_align = l_align + ((c_a - c_delta[h].detach()) ** 2).sum(-1).mean()
            terms["act"] = (l_act + self.align_weight * l_align) / C.DEPTH
            metrics["act/decode"] = float(l_act.detach()) / C.DEPTH
            metrics["act/align"] = float(l_align.detach()) / C.DEPTH
            if self.align_mode == "sparse_ce":
                metrics["act/align_ce"] = float(l_align.detach()) / C.DEPTH
                metrics["act/align_topk_overlap"] = (
                    float(align_overlap.detach()) / C.DEPTH
                )
            if dual_decode:
                metrics["act/decode_teacher"] = (
                    float(l_decode_teacher.detach()) / C.DEPTH
                )
                metrics["act/decode_deploy"] = (
                    float(l_decode_deployed.detach()) / C.DEPTH
                )
                metrics["act/decode_gap"] = (
                    float((l_decode_deployed - l_decode_teacher).detach()) / C.DEPTH
                )
            if self.act_decode_from in ("proposal", "dual_q_action_proposal"):
                metrics["act/deploy_c_l2"] = float(deploy_l2) / C.DEPTH
                metrics["act/deploy_topk_overlap"] = float(deploy_overlap) / C.DEPTH
            # Did the flip engage? `c_a` batch-constant reads ~0 here. One
            # scalar under no_grad; nothing below enters the training graph.
            with torch.no_grad():
                ca = torch.stack(c_act, 1).float()                  # (B,DEPTH,M)
                metrics["act/c_a_spread"] = float(
                    (ca - ca.mean(0, keepdim=True)).norm(dim=-1).mean())
                cd = torch.stack(c_delta, 1).float()
                metrics["act/c_delta_spread"] = float(
                    (cd - cd.mean(0, keepdim=True)).norm(dim=-1).mean())

        # ── L_proposal ─────────────────────────────────────────────────────
        if self._on("proposal", step):
            src = c_act if c_act is not None else c_delta
            if self.proposal_mode == "dense_kl":
                src_logits = c_act_lg if c_act is not None else self._qd_logits
                if not all(x is not None for x in src_logits):
                    raise RuntimeError(
                        "dense_kl proposal training requires dense teacher logits; "
                        "the configured q_a/q_delta returned coefficients only"
                    )
                loss_prop, dense_kl, hard_nll = zero, zero, zero
                overlap, teacher_ent, student_ent = zero, zero, zero
                for h in range(C.DEPTH):
                    teacher_lg = src_logits[h]
                    loss_h, student_lg = proposal_distill_loss(
                        self.proposal, zs[h], window["lang"], teacher_lg,
                        temperature=self.proposal_temperature,
                        detach_belief=self.proposal_detach_belief,
                        return_student_logits=True,
                    )
                    dense_kl = dense_kl + loss_h
                    loss_prop = loss_prop + loss_h
                    if self.proposal_hard_weight > 0.0:
                        z_prop = (zs[h].detach() if self.proposal_detach_belief
                                  else zs[h])
                        hard_h = proposal_bc_loss(
                            self.proposal, z_prop, window["lang"], src[h])
                        hard_nll = hard_nll + hard_h
                        loss_prop = loss_prop + self.proposal_hard_weight * hard_h
                    with torch.no_grad():
                        ti = teacher_lg.float().topk(C.TOPK, dim=-1).indices
                        si = student_lg.float().topk(C.TOPK, dim=-1).indices
                        overlap = overlap + (si.unsqueeze(-1) == ti.unsqueeze(-2)).any(-1).float().mean()
                        tp = torch.softmax(teacher_lg.float(), dim=-1)
                        sp = torch.softmax(student_lg.float(), dim=-1)
                        teacher_ent = teacher_ent - (tp * tp.clamp_min(1e-12).log()).sum(-1).mean()
                        student_ent = student_ent - (sp * sp.clamp_min(1e-12).log()).sum(-1).mean()
                terms["proposal"] = loss_prop / C.DEPTH
                metrics["proposal/dense_kl"] = float(dense_kl.detach()) / C.DEPTH
                if self.proposal_hard_weight > 0.0:
                    metrics["proposal/hard_nll"] = float(hard_nll.detach()) / C.DEPTH
                metrics["proposal/topk_overlap"] = float(overlap) / C.DEPTH
                metrics["proposal/teacher_entropy"] = float(teacher_ent) / C.DEPTH
                metrics["proposal/student_entropy"] = float(student_ent) / C.DEPTH
            elif self.proposal_mode == "sparse_ce":
                loss_prop = zero
                overlap, teacher_ent, student_ent = zero, zero, zero
                for h in range(C.DEPTH):
                    loss_h, student_lg = proposal_sparse_ce_loss(
                        self.proposal, zs[h], window["lang"], src[h],
                        temperature=self.proposal_temperature,
                        detach_belief=self.proposal_detach_belief,
                        return_student_logits=True,
                    )
                    loss_prop = loss_prop + loss_h
                    with torch.no_grad():
                        target = src[h].detach().float()
                        ti = target.topk(C.TOPK, dim=-1).indices
                        si = student_lg.float().topk(C.TOPK, dim=-1).indices
                        overlap = overlap + (si.unsqueeze(-1) == ti.unsqueeze(-2)).any(-1).float().mean()
                        student_p = torch.softmax(student_lg.float(), dim=-1)
                        teacher_ent = teacher_ent - (
                            target * target.clamp_min(1e-12).log()).sum(-1).mean()
                        student_ent = student_ent - (
                            student_p * student_p.clamp_min(1e-12).log()).sum(-1).mean()
                terms["proposal"] = loss_prop / C.DEPTH
                metrics["proposal/sparse_ce"] = float(loss_prop.detach()) / C.DEPTH
                metrics["proposal/topk_overlap"] = float(overlap) / C.DEPTH
                metrics["proposal/teacher_entropy"] = float(teacher_ent) / C.DEPTH
                metrics["proposal/student_entropy"] = float(student_ent) / C.DEPTH
            else:
                lp = zero
                for h in range(C.DEPTH):
                    z_prop = zs[h].detach() if self.proposal_detach_belief else zs[h]
                    lp = lp + self.proposal.log_prob(
                        z_prop, window["lang"], src[h].detach()).mean()
                terms["proposal"] = -lp / C.DEPTH

        # ── L_balance ──────────────────────────────────────────────────────
        if self._on("balance", step):
            allc = torch.stack(c_delta + (c_act or []), dim=0).flatten(0, -2)
            all_lg = self._qd_logits + (c_act_lg if c_act is not None else [])
            lg = (torch.stack(all_lg, 0).flatten(0, -2)
                  if all(x is not None for x in all_lg) else None)
            if self.balance_mode == "per_head":
                head_terms: dict[str, Tensor] = {}
                qd_lg = (
                    torch.stack(self._qd_logits, 0).flatten(0, -2)
                    if all(item is not None for item in self._qd_logits) else None
                )
                head_terms["q_delta"] = _switch_balance(
                    torch.stack(c_delta, 0).flatten(0, -2), qd_lg,
                )
                if c_act is not None:
                    qa_lg = (
                        torch.stack(c_act_lg, 0).flatten(0, -2)
                        if all(item is not None for item in c_act_lg) else None
                    )
                    head_terms["q_action"] = _switch_balance(
                        torch.stack(c_act, 0).flatten(0, -2), qa_lg,
                    )
                weights = {
                    name: self.balance_head_weights.get(name, 0.0)
                    for name in head_terms
                }
                denominator = sum(weights.values())
                if denominator <= 0.0:
                    raise ValueError(
                        "per_head balance requires positive weight for a present head"
                    )
                terms["balance"] = sum(
                    weights[name] / denominator * value
                    for name, value in head_terms.items()
                )
                for name, value in head_terms.items():
                    metrics[f"balance/{name}"] = float(value.detach())
            else:
                terms["balance"] = _switch_balance(allc, lg)
            # pooled, unchanged, so it stays comparable with the prior runs
            cbar = allc.detach().float().mean(0).clamp_min(1e-9)
            cbar = cbar / cbar.sum()
            metrics["bank/live_ops"] = float((cbar > 1e-4).sum())
            # ...and split by head, which is the number that carries information
            ops, ent = _usage(c_delta)
            metrics["bank/live_ops_q_delta"], metrics["bank/entropy_q_delta"] = ops, ent
            if c_act is not None:
                ops, ent = _usage(c_act)
                metrics["bank/live_ops_q_a"], metrics["bank/entropy_q_a"] = ops, ent

        # ── R3: potential + GRPO ───────────────────────────────────────────
        if self._on("potential", step) and getattr(self, "potential", None) is not None:
            reward = window.get("reward")
            if reward is None:
                reward = torch.zeros(zs[0].shape[0], device=dev, dtype=zs[0].dtype)
            phi = self.potential(zs[-1], window["lang"])
            terms["potential"] = F.mse_loss(phi.float(), reward.float())
        if self._on("grpo", step) and getattr(self, "potential", None) is not None:
            terms["grpo"] = self._grpo(zs[0], window, dev)

        total = zero
        for name, t in terms.items():
            w = float(self.loss_cfg.get(name, {}).get("weight", 1.0))
            scale = self._loss_scale(name, step)
            total = total + w * scale * t
            metrics[f"loss/{name}"] = float(t.detach())
            metrics[f"schedule/{name}_scale"] = scale
        metrics["loss"] = float(total.detach())
        return total, metrics

    def _loss_scale(self, name: str, step: int) -> float:
        """Config-hashed one-based start/ramp schedule for one objective."""
        cfg = self.loss_cfg.get(name, {})
        update = int(step) + 1
        start = int(cfg.get("start_update", 1))
        ramp = int(cfg.get("ramp_updates", 0))
        if start < 1 or ramp < 0:
            raise ValueError(
                f"losses.{name}.start_update must be >=1 and ramp_updates >=0"
            )
        if update < start:
            return 0.0
        if ramp == 0:
            return 1.0
        return min(1.0, float(update - start + 1) / float(ramp))

    def _on(self, name: str, step: int | None = None) -> bool:
        enabled = bool(self.loss_cfg.get(name, {}).get("enabled", False))
        return enabled and (step is None or self._loss_scale(name, step) > 0.0)

    @torch.no_grad()
    def _delta_op(self, zs, zts, c_true, step: int, rank: int, seed: int) -> float:
        """Delta_op = d(A(c_rand) z, z+) - d(A(c_true) z, z+), which must be > 0.

        A build assert, not a metric (PLAN 4.C). Latent states 8 steps apart are
        ~0.95 cosine-similar before training, so ``A(c) ~ I`` nearly satisfies
        ``L_dyn`` while ``c`` carries nothing.

        **This is NOT the discrimination test.** It compares the true operator
        against a *uniform random* simplex point, so it says the bank is alive
        and nothing more; a collapsed ``c`` makes the comparison vacuous in the
        other direction too (CLAUDE.md). ``_delta_sel`` is the real guard.

        ``within_trajectory`` negatives are ``c`` from another segment of the SAME
        trajectory at least 2 segments away -- same scene, same body, genuinely
        different effect. Uncurated in-batch negatives would make two bodies
        producing the same world effect negatives for each other, which is the
        opposite of what a shared bank should learn.
        """
        g = torch_generator(seed, step, rank, tag="delta_op")
        negs = []
        for h in range(C.DEPTH):
            if self.negatives == "within_trajectory":
                far = [j for j in range(C.DEPTH) if abs(j - h) >= 2]
                negs.append(c_true[far[int(torch.randint(len(far), (1,), generator=g))]])
            else:
                negs.append(S.sparse_simplex(zs[h].shape[0], device=zs[h].device,
                                             dtype=zs[h].dtype))
        # One batched bank.step for all 2 * DEPTH probes. Delta_op is computed on
        # EVERY step, and the bank rebuilds its (M, K, D/2) lambda tables inside
        # every call, so 8 separate calls put ~30% of the step into a diagnostic.
        z_in = torch.cat([zs[h] for h in range(C.DEPTH)], 0)
        z_tgt = torch.cat([zts[h + 1] for h in range(C.DEPTH)], 0)
        out = self.bank.step(torch.cat(c_true + negs, 0), torch.cat([z_in, z_in], 0))
        n = z_in.shape[0]
        return float(_cos_dist(out[n:], z_tgt) - _cos_dist(out[:n], z_tgt))

    @torch.no_grad()
    def _delta_sel(self, zs, zts, c_true) -> dict[str, float]:
        """THE discrimination guard.  `Delta_sel > 0` or the coefficient is decoration.

            c_other   = c.roll(1, dims=0)          # a REAL c, from another window
            Delta_sel = d(A(c_other) z, z+) - d(A(c_true) z, z+)

        Same distance `L_dyn` uses, same batched single `bank.step` as
        `_delta_op`, reported per horizon and as the mean.

        `Delta_op` compares the true operator against a *uniform random* point
        on the simplex, so it answers "is the bank alive". This asks the only
        question that matters for the method: does the coefficient this window
        produced predict this window's transition BETTER THAN a coefficient a
        different window produced? On the R0-A checkpoints the answer was
        +0.0002 (ctrl) / +0.0000 (zinit) -- any other window's operator
        predicted the transition exactly as well, which means `c` was not
        carrying the transition at all.

        `roll(1, dims=0)` needs B >= 2 to be a different window; at B=1 it is the
        identity and this is identically zero by construction.
        """
        z_in = torch.cat([zs[h] for h in range(C.DEPTH)], 0)
        z_tgt = torch.cat([zts[h + 1] for h in range(C.DEPTH)], 0)
        c_pos = torch.cat(list(c_true), 0)
        c_oth = torch.cat([c.roll(1, dims=0) for c in c_true], 0)
        out = self.bank.step(torch.cat([c_pos, c_oth], 0), torch.cat([z_in, z_in], 0))
        n = z_in.shape[0]
        gap = (ln_cosine_distance(out[n:], z_tgt)
               - ln_cosine_distance(out[:n], z_tgt)).view(C.DEPTH, -1).mean(1)
        m = {f"delta_sel/h{h + 1}": float(gap[h]) for h in range(C.DEPTH)}
        m["delta_sel"] = float(gap.mean())
        return m

    def grad_probe_metrics(self) -> dict[str, float]:
        """Per-entry |grad| at q_Delta's logits, unselected : selected.

        Called by `main` AFTER `loss.backward()` and BEFORE `zero_grad`, on the
        steps `_probe_grad` marked. Returns `{}` on every other step and on the
        stub path.

        This is the number `L_balance` exists to move. `topk_simplex_st` returns
        `hard + soft - soft.detach()`, so the backward into an out-of-support
        logit is not blocked -- it is simply small, and a ratio near zero means
        an operator that has fallen out of every support is receiving no useful
        signal to come back with.
        """
        if not self._probe_grad:
            return {}
        num = den = 0.0
        n_num = n_den = 0
        for lg in self._qd_logits:
            g = None if lg is None else lg.grad
            if g is None:
                continue
            g = g.detach().float().abs()
            # selected == in the hard top-4 support, read off the logits
            # themselves so this does not depend on `c` still being alive.
            sel = torch.zeros_like(g).scatter_(
                -1, lg.detach().float().topk(C.TOPK, dim=-1).indices, 1.0).bool()
            num += float(g[~sel].sum()); n_num += int((~sel).sum())
            den += float(g[sel].sum()); n_den += int(sel.sum())
        if n_num == 0 or n_den == 0 or den == 0.0:
            return {}
        per_unsel, per_sel = num / n_num, den / n_den
        return {"grad_ratio/q_delta_logits": per_unsel / per_sel if per_sel else 0.0,
                "grad_per_entry/q_delta_unselected": per_unsel,
                "grad_per_entry/q_delta_selected": per_sel}

    def _grpo(self, z: Tensor, window: dict, dev) -> Tensor:
        """Group-relative advantage on pi_c, scored by Phi. R3 only.

        PLAN 7 names "potential + GRPO" for R3 but does not specify the estimator,
        and nothing in this repo produces sim rollouts yet. This is the plumbing:
        sample a group, score the rollout leaf with Phi, centre the reward inside
        the group, and reinforce. Team E owns the search; revisit before R3.
        """
        n = int(self.loss_cfg.get("grpo", {}).get("group", 8))
        c_seq = self.proposal.sample(z, window["lang"], n=n)         # (B, n, M)
        with torch.no_grad():
            plan = c_seq.unsqueeze(2).expand(-1, -1, C.DEPTH, -1)
            leaf = self.bank.rollout(plan, z)                        # (B, n, K, D)
            r = self.potential(leaf, window["lang"]).float()         # (B, n)
            adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-6)
        b, k, d = z.shape
        zf = z.unsqueeze(1).expand(b, n, k, d).reshape(b * n, k, d)
        lf = window["lang"].unsqueeze(1).expand(-1, n, -1, -1).flatten(0, 1)
        lp = self.proposal.log_prob(zf, lf, c_seq.reshape(b * n, C.M)).view(b, n)
        return -(adv * lp).mean()


# ═══════════════════════════════════════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════════════════════════════════════

class WindowSampler:
    """Deterministic window source with a checkpointable cursor.

    Team A owns ``loom/data/loader.py``; ``source: stub`` here keeps Team D
    unblocked and is swapped by ``build_sampler``. Whatever replaces it must keep
    two properties: batches are embodiment-homogeneous, and the cursor is
    ``state_dict``-able so a resumed link does not re-see the same windows.
    """

    def __init__(self, cfg: dict, rank: int, world: int, seed: int, device: str = "cpu"):
        dcfg = cfg.get("data", {})
        self.embodiments = list(dcfg.get("embodiments", ["libero_franka"]))
        self.batch = int(dcfg.get("batch_per_gpu", 2))
        self.action_free = bool(dcfg.get("action_free", False))
        self.rank, self.world, self.seed = rank, world, seed
        self.device = device
        self.cursor = 0
        self.epoch = 0

    def embodiment_for(self, step: int) -> str:
        """One embodiment per batch, cycled between batches (PLAN 9)."""
        return self.embodiments[(step + self.rank) % len(self.embodiments)]

    def next(self, step: int) -> dict:
        emb = self.embodiment_for(step)
        w = S.make_window(b=self.batch, embodiment=emb, device=self.device,
                          action_free=self.action_free)
        self.cursor += self.batch
        return w

    def state_dict(self) -> dict:
        return {"cursor": self.cursor, "epoch": self.epoch,
                "embodiments": self.embodiments, "batch": self.batch}

    def load_state_dict(self, sd: dict) -> None:
        self.cursor = int(sd.get("cursor", 0))
        self.epoch = int(sd.get("epoch", 0))


def log_shm_headroom(cfg: dict) -> None:
    """/dev/shm is 64 MiB on this cluster. DataLoader workers pass tensors through
    it, so a LIBERO prefetch queue does not fit and the failure surfaces as
    ``DataLoader worker (pid N) exited unexpectedly`` or a bare bus error -- which
    reads like a code bug, not a resource limit. Both torch sharing strategies use
    /dev/shm, so switching strategy does not help.

    Team A's ``LoomLoader`` calls ``fit_workers()`` to shrink workers/prefetch (down
    to in-process) so the queue fits; this only makes the measured number visible in
    the log of every link, because a compute node's value may differ from the login
    node's.
    """
    try:
        from loom.data.loader import shm_free_bytes
    except Exception:
        return
    try:
        free = shm_free_bytes()
    except Exception as e:
        print(f"[data] /dev/shm unreadable ({e!r})", flush=True)
        return
    dcfg = cfg.get("data", {})
    print(f"[data] /dev/shm free {free / 2 ** 20:.0f} MiB, "
          f"num_workers={dcfg.get('num_workers', 0)} "
          f"prefetch_factor={dcfg.get('prefetch_factor', 2)} "
          f"(LoomLoader shrinks these with fit_workers to fit)", flush=True)


#: (factory name, kwargs builder). `build_loader` is the agreed Team A factory;
#: the class constructors are the fallback and spell `world_size`, not `world`.
_LOADER_FACTORIES = (
    ("build_loader", lambda r, w, s, d: dict(rank=r, world=w, seed=s, device=d)),
    ("WindowLoader", lambda r, w, s, d: dict(rank=r, world_size=w, seed=s, device=d)),
    ("LoomLoader", lambda r, w, s, d: dict(rank=r, world_size=w, seed=s)),
)


def build_sampler(cfg: dict, rank: int, world: int, seed: int, device: str):
    """Team A's loader for a real source; the stub sampler ONLY for `source: stub`.

    Falling back to stub windows when the config asked for LIBERO is a wasted run,
    not a degraded one: R0-A would have trained 16 GPUs for eight hours on
    `torch.randn` and produced a first score that reads like a modelling result.
    A missing loader is therefore fatal here, and the traceback names every
    factory that was tried and why each failed.
    """
    source = cfg.get("data", {}).get("source", "stub")
    if source == "stub":
        return WindowSampler(cfg, rank, world, seed, device)

    tried: list[str] = []
    try:
        mod = importlib.import_module("loom.data.loader")
    except Exception as e:
        raise RuntimeError(
            f"data.source is {source!r} but loom.data.loader could not be imported "
            f"({e!r}). Set data.source: stub to train on random windows on purpose."
        ) from e

    for name, kwargs_for in _LOADER_FACTORIES:
        fn = getattr(mod, name, None)
        if fn is None:
            tried.append(f"{name}: absent")
            continue
        try:
            # Never hardcode num_workers here: fit_workers/_fit_shared_memory
            # inside the loader shrink it to what /dev/shm actually holds, and
            # that differs between login (64 MiB) and compute (1008 GiB) nodes.
            return fn(cfg, **kwargs_for(rank, world, seed, device))
        except Exception as e:
            tried.append(f"{name}: {e!r}")

    raise RuntimeError(
        f"data.source is {source!r} but no usable loader factory was found in "
        f"loom.data.loader:\n  " + "\n  ".join(tried) +
        "\nTeam A owns build_loader(cfg, *, rank, world, seed, device). "
        "Set data.source: stub to train on random windows on purpose."
    )


def _to_device(window: dict, device: str, dtype=None) -> dict:
    """Move and, on the CUDA path, pin every float tensor to the compute dtype.

    Cached tower features arrive in whatever Team A's cache stored (fp16 or
    fp32); leaving them fp32 drags the heads out of bf16 the same way the belief
    did, and doubles the resident size of the largest tensor in the batch.
    Integer tensors are left alone.
    """
    if device == "cpu" and dtype is None:
        return window

    def _m(v):
        v = v.to(device, non_blocking=True)
        return v.to(dtype) if dtype is not None and v.is_floating_point() else v

    out = dict(window)
    out["feats"] = [{k: _m(v) for k, v in f.items()} for f in window["feats"]]
    if window.get("burn_in_feats") is not None:
        out["burn_in_feats"] = [
            {k: _m(v) for k, v in f.items()} for f in window["burn_in_feats"]
        ]
    out["lang"] = _m(window["lang"])
    if window.get("actions") is not None:
        out["actions"] = _m(window["actions"])
    if window.get("reward") is not None:
        out["reward"] = _m(window["reward"])
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  TRAIN STATE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TrainState:
    """Everything that must survive a requeue.

    ``tests/test_train.py::test_state_coverage_by_reflection`` walks this object
    and fails if any field exposing ``state_dict()`` is not saved by
    ``ckpt.build_state``. Add a field here without adding it there and a test
    goes red instead of a 37-link chain going silently wrong.
    """

    model: nn.Module
    optimizer: Any
    scheduler: Any
    ema: Any
    sampler: Any
    #: spike-rejection reference. Optimizer state in every sense that matters:
    #: left out of the checkpoint it would reset at every 4 h link boundary, and
    #: the first ``warmup`` steps of every link would run unguarded.
    guard: Any = None
    global_step: int = 0
    samples_seen: int = 0

    #: fields that are configuration or a live handle, not mutable training state
    NOT_STATE = ("global_step", "samples_seen")


# ═══════════════════════════════════════════════════════════════════════════
#  LAUNCH ASSERTS
# ═══════════════════════════════════════════════════════════════════════════

def assert_ranks_distinct(ident: dict) -> None:
    """Log this rank's identity and prove no two ranks collide.

    Under plain ``srun`` nothing sets ``RANK``; SLURM exports ``SLURM_PROCID``.
    If the sbatch forgets to map it, every task reads rank 0, draws the same
    windows and writes the same checkpoint shard -- and the loss curve looks
    completely normal.
    """
    ident = dict(ident, ckpt_shard=ckpt_mod.shard_name(0, ident["rank"]))
    print(f"[rank{ident['rank']}] " + " ".join(f"{k}={v}" for k, v in ident.items()),
          flush=True)

    # A job allocated GPUs that cannot see them must not quietly train on CPU:
    # a +cu13x wheel on this CUDA-12.2 driver imports fine, reports
    # is_available()==False, and trains on CPU while holding 8 A100s.
    if os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_GPUS_ON_NODE"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "SLURM allocated GPUs but torch.cuda.is_available() is False. "
                "The node driver is CUDA 12.2 -- install torch==2.6.0+cu124 from "
                "https://download.pytorch.org/whl/cu124, not the default wheel."
            )

    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return
    gathered: list = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, ident)
    for f in ("rank", "rng_fingerprint", "ckpt_shard"):
        seen = [g[f] for g in gathered]
        if len(set(seen)) != len(seen):
            raise RuntimeError(
                f"ranks collide on {f!r}: {seen}. Every task is running as the same "
                f"rank -- check that the sbatch maps SLURM_PROCID into RANK."
            )


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args(argv=None):
    p = argparse.ArgumentParser("loom.train")
    p.add_argument("--config", required=True, help="configs/rX.yaml")
    # ── experiment-defining (all in config_hash) ──
    p.add_argument("--steps", type=int, default=None,
                   help="schedule horizon; IDENTICAL on every link of a chain")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--ckpt_every", type=int, default=None)
    p.add_argument("--log_every", type=int, default=None)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--set", action="append", default=[], metavar="a.b=value",
                   help="override any config key, JSON-parsed")
    # ── link-local (excluded from config_hash) ──
    p.add_argument("--run_dir", default=None)
    p.add_argument("--stop_at", type=int, default=None,
                   help="end THIS link at this global_step; --steps still sets the "
                        "schedule, so links are interchangeable")
    p.add_argument("--budget_s", type=float, default=None)
    p.add_argument("--safety_s", type=float, default=None)
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--allow_reshard", action="store_true")
    return p.parse_args(argv)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args)
    link = cfg["link"]
    run_dir = Path(link["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    rcfg, ocfg, dcfg = cfg["run"], cfg["optim"], cfg["data"]
    seed = int(rcfg.get("seed", 0))
    direct_keys = ("schedule_horizon", "max_updates")
    direct_formal = any(key in rcfg for key in direct_keys)
    if direct_formal and not all(key in rcfg for key in direct_keys):
        raise ValueError(
            "direct-formal scheduling requires both run.schedule_horizon and "
            "run.max_updates"
        )
    direct_schedule = DirectFormalSchedule.from_config(cfg) if direct_formal else None
    if direct_schedule is not None:
        steps = direct_schedule.max_updates
        # ``run.steps`` remains in inherited configs and in generic reporting.
        # It must name the execution cap, never a third conflicting horizon.
        legacy_steps = int(rcfg.get("steps", steps))
        if legacy_steps != steps:
            raise ValueError(
                "direct-formal run.steps must equal run.max_updates; the LR "
                "horizon belongs only in run.schedule_horizon"
            )
    else:
        steps = int(rcfg.get("steps", 1000))
    log_every = int(rcfg.get("log_every", 20))
    ckpt_every = int(rcfg.get("ckpt_every", 500))
    reconcile_metrics_on_resume_value = rcfg.get(
        "reconcile_metrics_on_resume", False,
    )
    if not isinstance(reconcile_metrics_on_resume_value, bool):
        raise ValueError("run.reconcile_metrics_on_resume must be a boolean")
    reconcile_metrics_on_resume = reconcile_metrics_on_resume_value
    chash = config_hash(cfg)

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if world > 1:
        import torch.distributed as dist

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank % torch.cuda.device_count())
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

    # A formal fresh lineage may resume its own exact checkpoints, but it may
    # never inherit an unrelated run merely because that directory has LATEST.
    # Authenticate the empty first-link surface before rank 0 writes config.json.
    fresh_lineage_required = bool(rcfg.get("fresh_start_required", False))
    if reconcile_metrics_on_resume and not fresh_lineage_required:
        raise ValueError(
            "run.reconcile_metrics_on_resume requires "
            "run.fresh_start_required=true"
        )
    if reconcile_metrics_on_resume and direct_formal:
        raise ValueError(
            "run.reconcile_metrics_on_resume is the non-DirectFormal ledger "
            "contract; DirectFormal already owns its authenticated rollback"
        )
    resume_integrity_required = direct_formal or reconcile_metrics_on_resume
    execution_failure_error = ""
    if rank == 0 and reconcile_metrics_on_resume:
        try:
            durable_failure = _read_execution_failure(
                run_dir, config_hash_value=chash,
            )
        except Exception as error:  # noqa: BLE001
            execution_failure_error = (
                "durable execution-failure marker is invalid: "
                f"{type(error).__name__}: {error}"
            )
        else:
            if durable_failure is not None:
                execution_failure_error = (
                    "lineage is terminal after durable execution failure at "
                    f"step {durable_failure['global_step']}: "
                    f"{durable_failure['reason']}"
                )
    if world > 1:
        execution_failure_box = [execution_failure_error]
        dist.broadcast_object_list(execution_failure_box, src=0)
        execution_failure_error = str(execution_failure_box[0])
    if execution_failure_error:
        raise RuntimeError(execution_failure_error)
    try:
        restart_count = int(os.environ.get("LOOM_RESTART_COUNT", "0"))
    except ValueError as error:
        raise ValueError("LOOM_RESTART_COUNT must be an integer") from error
    if restart_count < 0:
        raise ValueError("LOOM_RESTART_COUNT must be non-negative")
    latest_checkpoint_step = ckpt_mod.latest_step(run_dir)
    fresh_start = latest_checkpoint_step is None
    fresh_error: str | None = None
    bootstrap_metrics_recovery = False
    if rank == 0 and fresh_lineage_required:
        if fresh_start:
            stale: list[str] = []
            for name in ("metrics.jsonl", "HEARTBEAT", "STOP", "config.json"):
                if (run_dir / name).exists():
                    stale.append(name)
            stale.extend(
                path.name for path in sorted(run_dir.glob("ckpt_*_rank*.pt"))
            )
            if stale:
                marker_path = run_dir / "fresh_lineage_marker.json"
                if reconcile_metrics_on_resume and restart_count > 0:
                    try:
                        marker = json.loads(marker_path.read_text())
                        previous_cfg = json.loads((run_dir / "config.json").read_text())
                    except Exception as error:  # noqa: BLE001
                        fresh_error = (
                            "fresh step-0 recovery marker/config is unreadable: "
                            f"{error}"
                        )
                    else:
                        expected_marker = {
                            "format": "loom-fresh-training-lineage-marker-v1",
                            "config_hash": chash,
                            "run_name": rcfg.get("name"),
                            "metrics_rollback_format": (
                                _FRESH_METRICS_ROLLBACK_FORMAT
                            ),
                        }
                        if (
                            marker_path.is_symlink()
                            or marker != expected_marker
                            or config_hash(previous_cfg) != chash
                        ):
                            fresh_error = (
                                "fresh step-0 recovery lineage/config changed"
                            )
                        else:
                            bootstrap_metrics_recovery = True
                else:
                    fresh_error = (
                        "fresh_start_required refuses prior training state in "
                        f"{run_dir}: {stale[:8]}"
                    )
        else:
            config_path = run_dir / "config.json"
            try:
                previous_cfg = json.loads(config_path.read_text())
                previous_hash = config_hash(previous_cfg)
            except Exception as error:  # noqa: BLE001
                fresh_error = f"fresh formal resume config is unreadable: {error}"
            else:
                if previous_hash != chash:
                    fresh_error = (
                        "fresh formal resume config mismatch before load: "
                        f"run directory {previous_hash}, current {chash}"
                    )
    if world > 1:
        payload_error = [fresh_error, bootstrap_metrics_recovery]
        dist.broadcast_object_list(payload_error, src=0)
        fresh_error = payload_error[0]
        bootstrap_metrics_recovery = bool(payload_error[1])
    if fresh_error is not None:
        raise RuntimeError(fresh_error)

    if rcfg.get("deterministic"):
        enable_determinism()
    set_global_seed(seed, rank)
    assert_ranks_distinct(rank_identity(seed, rank, local_rank, world))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lineage_receipt = {
        "kind": "loom-fresh-training-lineage-v1",
        "fresh_start_required": fresh_lineage_required,
        "config_hash": chash,
        "seed": seed,
        "world_size": world,
        "act_decode_from": cfg.get("losses", {}).get("act", {}).get(
            "decode_from", "q_action"
        ),
        "schedule_horizon": (
            direct_schedule.schedule_horizon
            if direct_schedule is not None else steps
        ),
        "max_updates": steps,
    }
    if reconcile_metrics_on_resume:
        lineage_receipt["metrics_ledger"] = {
            "format": _FRESH_METRICS_ROLLBACK_FORMAT,
            "reconcile_crash_tail_to_latest_checkpoint": True,
            "checkpoint_boundary_fsync": True,
            "direct_formal_decisions": False,
        }
    if rank == 0:
        config_path = run_dir / "config.json"
        if not bootstrap_metrics_recovery:
            atomic_mod.atomic_write_text(
                config_path, json.dumps(cfg, indent=2, default=str),
            )
        schedule_text = (
            f" schedule_horizon={direct_schedule.schedule_horizon}"
            if direct_schedule is not None else ""
        )
        print(f"[rank0] run={rcfg.get('name')} config_hash={chash} "
              f"max_updates={steps}{schedule_text} fresh_start={int(fresh_start)} "
              f"world={world} device={device}", flush=True)

    # ── model ──────────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    trainable = set(cfg.get("train_modules", list(MODULE_NAMES)))
    for name in MODULE_NAMES:
        sub = getattr(model, name, None)
        if sub is not None:
            sub.requires_grad_(name in trainable)

    freeze = FreezeSchedule(
        modules=tuple(cfg["freeze"].get("modules", ())),
        until_frac=float(cfg["freeze"].get("until_frac", 0.0)),
        total_steps=steps,
    )
    model, sync = fsdp_mod.wrap_for_training(model, cfg.get("fsdp"), device=device,
                                             verbose=rank == 0)

    # bank_lr_mult is the PLAN 4.D rule; lr_scales lets R3 additionally tune E
    # and the bank "lightly" without touching the head LRs.
    lr_scales = {"bank": float(ocfg.get("bank_lr_mult", BANK_LR_MULT))}
    lr_scales.update({k: float(v) for k, v in (ocfg.get("lr_scales") or {}).items()})
    opt = build_optimizer(
        model, lr=float(ocfg.get("lr", 3e-4)),
        weight_decay=float(ocfg.get("weight_decay", 0.05)),
        betas=tuple(ocfg.get("betas", (0.9, 0.95))),
        lr_scales=lr_scales,
        module_names=MODULE_NAMES + ("ema",),
    )
    configured_opt_groups = _optimizer_group_config(opt)
    reset_state_modules = _optimizer_state_reset_modules(
        opt, ocfg.get("reset_state_modules", [])
    )
    sched = (
        direct_schedule
        if direct_schedule is not None else
        CosineWithWarmup(
            float(ocfg.get("lr", 3e-4)), int(ocfg.get("warmup", 2000)), steps,
            float(ocfg.get("min_lr_ratio", 0.05)),
        )
    )
    configured_scheduler = sched.state_dict().copy()
    if rank == 0:
        log_shm_headroom(cfg)
    sampler = build_sampler(cfg, rank, world, seed, "cpu")
    # `spike_mult: 0` is OFF and is the default: a guard is an intervention, and a
    # chain already in flight must not silently acquire one at a link boundary.
    spike = SpikeGuard(mult=float(ocfg.get("spike_mult", 0.0)),
                       beta=float(ocfg.get("spike_beta", 0.98)),
                       warmup=int(ocfg.get("spike_warmup", 100)))
    state = TrainState(model=model, optimizer=opt, scheduler=sched, ema=model.ema,
                       sampler=sampler, guard=spike)
    if rank == 0 and not model.update_ema:
        print("[rank0] EMA updates OFF: online and target estimator coordinates "
              "remain checkpoint-exact", flush=True)
    if rank == 0 and spike.enabled:
        print(f"[rank0] spike guard ON: skip when gnorm > {spike.mult}x the "
              f"running geometric mean (beta={spike.beta}, warmup={spike.warmup})",
              flush=True)

    # bf16 throughout (PLAN 9). FSDP's MixedPrecision covers the wrapped modules;
    # autocast covers the single-GPU debug path where nothing is wrapped.
    amp = device == "cuda" and str(cfg["fsdp"].get("precision", "bf16")) == "bf16"
    # LoomModel._cast pins the belief to bf16 at the estimator boundary, so the
    # rollout is bf16 on the stub path too and the assert is checked everywhere
    # amp is on. It was previously skipped for stubs, which is exactly why a GPU
    # smoke could not have caught the fp32 rollout.
    model.compute_dtype = torch.bfloat16 if amp else None
    model.check_bf16 = amp

    # ── resume ─────────────────────────────────────────────────────────────
    payload: dict[str, Any] | None = None
    with fsdp_mod.sharded_state_dict(model):
        payload = ckpt_mod.load_latest(run_dir, map_location=device,
                                       allow_reshard=link["allow_reshard"])
        if payload is not None:
            if fresh_lineage_required:
                if payload.get("config_hash") != chash:
                    raise RuntimeError(
                        "fresh formal resume config mismatch: checkpoint "
                        f"{payload.get('config_hash')!r}, current {chash!r}"
                    )
                if payload.get("fresh_lineage") != lineage_receipt:
                    raise RuntimeError(
                        "fresh formal resume lineage receipt is absent or changed"
                    )
            got = ckpt_mod.restore(payload, state, world_size=world)
            # torch.optim serialises group hyperparameters alongside moments,
            # and this scheduler serialises its whole (stateless) recipe.
            # Reapply current config after restoring state; otherwise a tuning
            # continuation that requests proposal=0.3x / 32k silently executes
            # the checkpoint's old proposal=1.0x / 60k recipe.
            _reapply_optimizer_group_config(opt, configured_opt_groups)
            sched.load_state_dict(configured_scheduler)
            parameter_reset = _reset_parameters_for_config_transition(
                model, ocfg.get("transition_parameter_reset"),
                checkpoint_config_hash=got["config_hash"],
                current_config_hash=chash,
            )
            if parameter_reset is not None:
                detail = ", ".join(
                    f"{name}={count}" for name, count in parameter_reset.items()
                )
                print(
                    f"[rank{rank}] reset parameter values for config transition "
                    f"({detail} tensor elements zeroed)", flush=True,
                )
            reset_counts = _reset_optimizer_state_for_config_transition(
                opt, reset_state_modules,
                checkpoint_config_hash=got["config_hash"],
                current_config_hash=chash,
            )
            if reset_counts is not None:
                detail = ", ".join(
                    f"{name}={count}" for name, count in reset_counts.items()
                )
                print(
                    f"[rank{rank}] reset optimizer state for config transition "
                    f"({detail} parameter states cleared)", flush=True,
                )
            if got["config_hash"] and got["config_hash"] != chash:
                print(f"[rank{rank}] WARNING config_hash changed "
                      f"{got['config_hash']} -> {chash}; this is a different "
                      f"experiment resuming into the same run dir", flush=True)
            print(f"[rank{rank}] resumed at step {state.global_step} "
                  f"(git {got['git_sha'][:8]})", flush=True)

    resumed = payload is not None
    if resume_integrity_required:
        local_resume_error = ""
        if latest_checkpoint_step is not None and payload is None:
            local_resume_error = (
                f"LATEST points at step {latest_checkpoint_step} but no local "
                "checkpoint payload was loaded"
            )
        elif payload is not None:
            payload_step = payload.get("global_step")
            if (
                not isinstance(payload_step, int)
                or isinstance(payload_step, bool)
                or payload_step != latest_checkpoint_step
                or state.global_step != latest_checkpoint_step
            ):
                local_resume_error = (
                    "authenticated checkpoint/LATEST step mismatch: "
                    f"LATEST={latest_checkpoint_step!r}, "
                    f"payload={payload_step!r}, state={state.global_step!r}"
                )
        if world > 1:
            resume_errors: list[str | None] = [None] * world
            dist.all_gather_object(resume_errors, local_resume_error)
            local_resume_error = "; ".join(
                f"rank{index}: {error}"
                for index, error in enumerate(resume_errors) if error
            )
        if local_resume_error:
            contract = "direct-formal" if direct_formal else "fresh-ledger"
            raise RuntimeError(
                f"{contract} checkpoint resume authentication failed: "
                + local_resume_error
            )

    # A hard crash can leave line-buffered metrics ahead of the pointer-last
    # checkpoint. Reconcile before W&B initialization and before opening the
    # ledger for append, so a repeated update can never create duplicate steps.
    if resume_integrity_required and (resumed or bootstrap_metrics_recovery):
        reconciliation_packet: dict[str, Any] | None = None
        if rank == 0:
            try:
                checkpoint_identity = {
                    "format": (
                        "loom-direct-formal-checkpoint-identity-v1"
                        if direct_formal else
                        "loom-fresh-training-checkpoint-identity-v1"
                    ),
                    "latest_step": latest_checkpoint_step,
                    "payload_global_step": (
                        payload.get("global_step") if payload is not None else 0
                    ),
                    "config_hash": (
                        payload.get("config_hash") if payload is not None else chash
                    ),
                    "git_sha": (
                        payload.get("git_sha") if payload is not None else None
                    ),
                    "world_size": (
                        payload.get("world_size") if payload is not None else world
                    ),
                    "fresh_lineage": (
                        payload.get("fresh_lineage")
                        if payload is not None else lineage_receipt
                    ),
                }
                reconcile = (
                    _reconcile_direct_formal_metrics
                    if direct_formal else _reconcile_fresh_metrics
                )
                result = reconcile(
                    run_dir, checkpoint_step=state.global_step,
                    checkpoint_identity=checkpoint_identity,
                )
                reconciliation_packet = {"result": result}
            except Exception as error:  # noqa: BLE001
                reconciliation_packet = {
                    "error": f"{type(error).__name__}: {error}",
                }
        if world > 1:
            reconciliation_box = [reconciliation_packet]
            dist.broadcast_object_list(reconciliation_box, src=0)
            reconciliation_packet = reconciliation_box[0]
        assert reconciliation_packet is not None
        if "error" in reconciliation_packet:
            contract = "direct-formal" if direct_formal else "fresh-ledger"
            raise RuntimeError(
                f"{contract} metrics resume reconciliation failed: "
                f"{reconciliation_packet['error']}"
            )
        if rank == 0 and reconciliation_packet["result"]["action"] == "ROLLBACK":
            contract = "direct-formal" if direct_formal else "fresh-lineage"
            print(
                f"[rank0] {contract} metrics crash tail quarantined and "
                f"rolled back to step {state.global_step}",
                flush=True,
            )

    # ── logging ────────────────────────────────────────────────────────────
    run = None if link["no_wandb"] else wandb_util.init(
        run_dir, rcfg.get("project", "loom"), cfg, rank=rank, name=rcfg.get("name"))
    metrics_fp = None
    if rank == 0:
        metrics_fp = open(run_dir / "metrics.jsonl", "a", buffering=1)

    guard = PreemptGuard(run_dir, budget_s=link["budget_s"],
                         safety_s=link["safety_s"] if link["safety_s"] is not None else 420.0)
    stop_at = min(link["stop_at"], steps) if link["stop_at"] else steps
    batch = int(dcfg.get("batch_per_gpu", 2))
    grad_clip = float(ocfg.get("grad_clip", 1.0))
    grad_report = bool(ocfg.get("grad_report", True))
    probe_every = int(ocfg.get("grad_probe_every", GRAD_PROBE_EVERY))
    t0, last_delta, last_sel = time.time(), float("nan"), float("nan")
    direct_terminal_status: str | None = None
    suite_metric_keys = (
        "loss", "act/decode_deploy", "proposal/sparse_ce",
        "dyn/effect", "dyn/contrastive",
    )
    suite_accumulator: dict[str, dict[str, float]] = {}
    prefix_accumulator: dict[int, int] = {}

    def _save(step: int, stop_reason: str = "") -> None:
        # stop_reason rides in the payload so a chain of 38 links can be triaged
        # from its checkpoints alone: "signal" is SLURM preempting on schedule,
        # "budget" is the link running out of its own clock, "sentinel" is a
        # human, "" is a periodic save. A run that keeps stopping for "budget"
        # near step 0 is a startup that got slower, not a preemption problem.
        # In authenticated ledger modes the metrics prefix is durable evidence.
        # Make it durable before LATEST can advance; a rank-0 I/O error is
        # broadcast before any rank enters the checkpoint barriers.
        if resume_integrity_required:
            durability_packet: dict[str, str] = {}
            if rank == 0:
                try:
                    assert metrics_fp is not None
                    metrics_fp.flush()
                    os.fsync(metrics_fp.fileno())
                except Exception as error:  # noqa: BLE001
                    durability_packet = {
                        "error": f"{type(error).__name__}: {error}",
                    }
            if world > 1:
                durability_box = [durability_packet]
                dist.broadcast_object_list(durability_box, src=0)
                durability_packet = durability_box[0]
            if "error" in durability_packet:
                contract = "direct-formal" if direct_formal else "fresh-ledger"
                raise RuntimeError(
                    f"{contract} metrics durability failed before checkpoint: "
                    f"{durability_packet['error']}"
                )

        checkpoint_extra = {"stop_reason": stop_reason}
        if fresh_lineage_required:
            checkpoint_extra["fresh_lineage"] = lineage_receipt
        with fsdp_mod.sharded_state_dict(model):
            ckpt_mod.save(
                ckpt_mod.build_state(state, config_hash=chash, world_size=world,
                                     wandb_run_id=wandb_util.stable_run_id(run_dir),
                                     extra=checkpoint_extra),
                run_dir, step, keep_last=int(rcfg.get("keep_last", 3)))

    def _direct_formal_decision(step: int) -> dict[str, Any]:
        """Evaluate/publish one fixed boundary with collective-safe errors."""
        packet: dict[str, Any] | None = None
        if rank == 0:
            try:
                assert metrics_fp is not None
                metrics_fp.flush()
                os.fsync(metrics_fp.fileno())
                rows = [
                    json.loads(line)
                    for line in (run_dir / "metrics.jsonl").read_text().splitlines()
                    if line.strip()
                ]
                receipt = evaluate_direct_formal(rows, current_step=step)
                receipt.update({
                    "config_hash": chash,
                    "fresh_lineage": lineage_receipt,
                })
                encoded = json.dumps(
                    receipt, sort_keys=True, separators=(",", ":"),
                ) + "\n"
                target = run_dir / f"direct_formal_{step:09d}.json"
                _exclusive_publish_bytes(target, encoded.encode("utf-8"))
                # Return the immutable published object on replay rather than a
                # second in-memory copy. Byte equality above authenticates it.
                packet = {"receipt": json.loads(target.read_text())}
            except Exception as error:  # noqa: BLE001
                packet = {"error": f"{type(error).__name__}: {error}"}
        if world > 1:
            gathered = [packet]
            dist.broadcast_object_list(gathered, src=0)
            packet = gathered[0]
        assert packet is not None
        if "error" in packet:
            raise RuntimeError(f"direct-formal boundary failed: {packet['error']}")
        return packet["receipt"]

    if direct_formal and resumed and should_evaluate_direct_formal(state.global_step):
        decision = _direct_formal_decision(state.global_step)
        status = str(decision["status"])
        if rank == 0:
            print(
                f"[rank0] direct-formal replay step={state.global_step} "
                f"status={status} reason={decision.get('reason')}",
                flush=True,
            )
        if status in ("PASS", "ABORT", "INVALID"):
            direct_terminal_status = status

    while direct_terminal_status is None and state.global_step < stop_at:
        step = state.global_step
        set_step_seed(seed, step, rank)
        is_frozen = freeze.apply(model, step, trainable)
        lrs = sched.apply(opt, step)

        window = _to_device(sampler.next(step), device, model.compute_dtype)
        # `retain_grad` on q_Delta's logits, on this step only. Cheap (a (B, M)
        # tensor per horizon) but it is still a diagnostic, and it needs the
        # backward to have run, so it is not on every step.
        model._probe_grad = probe_every > 0 and step % probe_every == 0
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            loss, metrics = model.compute_losses(window, step, rank, seed)

        opt.zero_grad(set_to_none=True)
        gnorm, skipped, gparts = 0.0, False, {}
        if loss.requires_grad:
            loss.backward()
            # BEFORE any all-reduce or clip: these are activation gradients on
            # THIS rank's own logits, and the question is their relative size.
            metrics.update(model.grad_probe_metrics())
            # Replicated modules are invoked through step()/log_prob()/loss(), not
            # forward(), so no DDP/FSDP hook ever fires for them. Sync by hand.
            sync.all_reduce_grads()
            # BEFORE clip_grad: afterwards every number carries the same `coef`
            # and the decomposition is no longer in the units the spike happened
            # in. One extra all-reduce of ~7 floats, unconditional on every rank.
            if grad_report:
                gparts = module_grad_norms(model, sync=sync,
                                           module_names=MODULE_NAMES)
            gnorm = clip_grad(model, grad_clip, sync=sync)
            if not state.guard.enabled and not math.isfinite(gnorm):
                raise FloatingPointError(
                    "non-finite global gradient norm with magnitude guard "
                    "disabled; refusing to count or apply this update"
                )
            # `gnorm` is the globally reduced pre-clip norm, so every rank feeds
            # the guard the identical number and reaches the identical verdict
            # without another collective. It must stay that way: a guard that
            # skipped on one rank only would desynchronise the optimizer.
            skipped = state.guard.check(gnorm)
            if not skipped:
                opt.step()
        model.update_target()

        state.global_step += 1
        state.samples_seen += batch * world
        last_delta = metrics.get("delta_op", last_delta)
        last_sel = metrics.get("delta_sel", last_sel)
        sampling_suite = window.get("sampling_suite")
        burn_in_steps = int(window.get("burn_in_steps", 0))

        if rank == 0 and sampling_suite is not None:
            bucket = suite_accumulator.setdefault(
                str(sampling_suite), {"count": 0.0},
            )
            bucket["count"] += 1.0
            for key in suite_metric_keys:
                if key in metrics:
                    bucket[key] = bucket.get(key, 0.0) + float(metrics[key])
            prefix_accumulator[burn_in_steps] = (
                prefix_accumulator.get(burn_in_steps, 0) + 1
            )

        if rank == 0 and metrics_fp is not None:
            metrics_fp.write(json.dumps({
                "global_step": state.global_step, "lr": lrs.get("estimator/decay",
                                                                sched.lr_at(step)),
                "grad_norm": gnorm, "frozen": is_frozen,
                "grad_skipped": int(skipped),
                # null, not Infinity: json.dumps emits a bare `Infinity` for the
                # disabled/warming-up guard, which Python reads back but jq and
                # every other JSON reader rejects.
                "grad_thresh": (state.guard.threshold
                                if math.isfinite(state.guard.threshold) else None),
                **{f"gnorm/{k}": v for k, v in gparts.items()},
                "embodiment": window["embodiment"],
                "sampling_suite": sampling_suite,
                "burn_in_steps": burn_in_steps,
                **metrics}) + "\n")

        if state.global_step % log_every == 0:
            write_heartbeat(run_dir, state.global_step, rank, last_delta)
            if rank == 0:
                parts = " ".join(f"{k[:3]}={v:.1f}" for k, v in gparts.items())
                print(f"[rank0] step {state.global_step} loss={metrics['loss']:.4f} "
                      f"delta_op={last_delta:+.4f} delta_sel={last_sel:+.4f} "
                      f"lr={sched.lr_at(step):.3e} "
                      f"gnorm={gnorm:.3f}{' SKIP' if skipped else ''} "
                      f"[{parts}] frozen={int(is_frozen)} "
                      f"emb={window['embodiment']} "
                      f"{state.global_step / max(1e-6, time.time() - t0):.2f} it/s",
                      flush=True)
            stratified_metrics: dict[str, float] = {}
            if rank == 0:
                for suite, values in suite_accumulator.items():
                    count = values["count"]
                    for key, value in values.items():
                        if key != "count":
                            stratified_metrics[
                                f"suite/{suite}/{key.replace('/', '_')}"
                            ] = value / count
                total_prefix = max(1, sum(prefix_accumulator.values()))
                for prefix, count in prefix_accumulator.items():
                    stratified_metrics[f"data/prefix_{prefix}_fraction"] = (
                        count / total_prefix
                    )
            try:
                wandb_util.log(run, {
                    **metrics, "grad_norm": gnorm,
                    "samples_seen": state.samples_seen,
                    "frozen": float(is_frozen),
                    "grad_skipped": float(skipped),
                    **{f"gnorm/{k}": v for k, v in gparts.items()},
                    "seconds_to_budget": guard.seconds_left,
                    **{f"lr/{k}": v for k, v in lrs.items()},
                    **stratified_metrics,
                }, state.global_step)
            except Exception:
                # Strict entry-local logging broadcasts one fatal outcome to
                # every rank. Publish a terminal marker before advancing LATEST;
                # this remains authoritative even if persistence of the W&B
                # health event itself was the failure.
                marker_packet: dict[str, Any] = {}
                if reconcile_metrics_on_resume:
                    if rank == 0:
                        try:
                            marker_packet = {
                                "marker": _publish_execution_failure(
                                    run_dir, config_hash_value=chash,
                                    global_step=state.global_step,
                                    reason="logging_failure",
                                ),
                            }
                        except Exception as marker_error:  # noqa: BLE001
                            marker_packet = {
                                "error": (
                                    f"{type(marker_error).__name__}: {marker_error}"
                                ),
                            }
                    if world > 1:
                        marker_box = [marker_packet]
                        dist.broadcast_object_list(marker_box, src=0)
                        marker_packet = marker_box[0]
                    if "error" in marker_packet:
                        raise RuntimeError(
                            "could not durably publish terminal logging failure: "
                            f"{marker_packet['error']}"
                        )
                # Save only after marker publication succeeds. A marker write
                # failure therefore cannot create an apparently valid endpoint.
                _save(state.global_step, "logging_failure")
                raise
            if rank == 0:
                suite_accumulator.clear()
                prefix_accumulator.clear()

        # EVERY rank, EVERY step. One rank saving while the others train hangs
        # the next collective until SLURM kills the job.
        stop = guard.should_stop()
        if stop or state.global_step % ckpt_every == 0 or state.global_step >= stop_at:
            _save(state.global_step, guard.reason if stop else "")
        if direct_formal and should_evaluate_direct_formal(state.global_step):
            decision = _direct_formal_decision(state.global_step)
            status = str(decision["status"])
            if rank == 0:
                print(
                    f"[rank0] direct-formal step={state.global_step} "
                    f"status={status} reason={decision.get('reason')}",
                    flush=True,
                )
            if status in ("PASS", "ABORT", "INVALID"):
                direct_terminal_status = status
                break
        if stop:
            print(f"[rank{rank}] stopping at {state.global_step} ({guard.reason})",
                  flush=True)
            break

    if metrics_fp is not None:
        metrics_fp.close()
    wandb_util.finish(run)
    print(f"[rank{rank}] exit at {state.global_step}/{steps}", flush=True)
    if direct_terminal_status == "INVALID":
        return 2
    if direct_terminal_status == "ABORT":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
