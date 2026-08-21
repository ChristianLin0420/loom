#!/usr/bin/env python3
"""Held-out promotion gate for the action-anchored R0 operator bank.

This is deliberately an offline gate.  It never launches LIBERO and it never
mutates a checkpoint.  The candidate passes only when the lower endpoint of a
paired, task/trajectory-cluster bootstrap is strictly positive for all of:

* ``delta_sel`` at each of the four operator horizons;
* identity error minus sequential bank-rollout error at each horizon; and
* pairwise LN-cosine spread between leaves of proposal-sampled plans.

For the declared joint q_action+bank stage it additionally authenticates the
step-49,666 deploy reference, verifies every frozen module is tensor-exact,
requires held-out action decoding not to degrade and proposal/q_action support
overlap to stay within its fixed non-inferiority margin relative to that
reference, and rejects a changed coefficient convention
when at least 1% of held-out roots have no proposal candidate satisfying
``||q_action(D(proprio,c),z)-c||_2 <= 0.5``.  All candidates at one root share
the same deterministic decoder noise.

The data entry point is intentionally narrow: ``loom.data.loader.build_gate_loader``
owns the train/gate trajectory split and supplies ``data_meta`` with task and
whole-trajectory cluster identities.  A normal training loader is never an
accepted fallback.  Missing metadata, provenance, model state, or a non-finite
number makes the gate fail closed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import traceback
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

# Direct ``python scripts/bank_ca_gate.py`` execution puts ``scripts/`` rather
# than the repository root on ``sys.path``.  Match the bootstrap used by the
# other executable scripts before importing repository modules.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

import contracts as C
from loom.losses.dyn import ln_cosine_distance, sequential_rollout


DEFAULT_BOOTSTRAP_SAMPLES = 2_000
DEFAULT_CONFIDENCE = 0.95
DEFAULT_BATCH_SIZE = 8
FORMAT_VERSION = 1
PINNED_WINDOWS = 256
PINNED_CANDIDATES = 32
REQUIREMENT_KEYS = (
    "delta_sel_ci_low_per_horizon",
    "identity_minus_rollout_ci_low_per_horizon",
    "proposal_candidate_leaf_spread_ci_low",
)
PRESERVATION_KEYS = (
    "reference_checkpoint_sha256",
    "reference_config_hash",
    "reference_global_step",
    "action_decode_improvement_ci_low",
    "proposal_support_overlap_change_ci_low",
    "q_action_residual_max",
    "max_root_exhaustion_rate",
)
IDENTITY_CENTERED_RESET = {
    "source_config_hash": "a199324a6205bb6d",
    "tensors": {"bank.omega": "zero"},
}


class GateError(RuntimeError):
    """An integrity or completeness failure; promotion must stop."""


@dataclass
class Candidate:
    model: Any
    config: dict[str, Any]
    provenance: dict[str, Any]
    gate_settings: dict[str, Any]


@dataclass
class ReferenceHeads:
    q_action: Any
    provenance: dict[str, Any]


@dataclass
class MetricRows:
    delta_sel: np.ndarray
    identity_minus_rollout: np.ndarray
    leaf_spread: np.ndarray
    task_ids: list[str]
    trajectory_ids: list[str]
    cluster_ids: list[str]
    root_q_action_residual: np.ndarray | None = None
    action_decode_improvement: np.ndarray | None = None
    proposal_support_overlap_change: np.ndarray | None = None

    @property
    def n(self) -> int:
        return int(self.delta_sel.shape[0])


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256_file(path: str | os.PathLike, chunk_bytes: int = 8 << 20) -> str:
    """Streaming SHA-256, including multi-GiB consolidated checkpoints."""
    p = Path(path)
    _require(p.is_file(), f"required file does not exist: {p}")
    h = hashlib.sha256()
    with p.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_bytes)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _experiment_config_hash(cfg: Mapping[str, Any]) -> str:
    """Same authenticated experiment hash as ``loom.train.loop.config_hash``."""
    experiment = {k: v for k, v in cfg.items() if k != "link"}
    return hashlib.blake2b(
        json.dumps(experiment, sort_keys=True, default=str).encode(), digest_size=8,
    ).hexdigest()


def _read_config(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"config does not exist: {path}")
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text())
    else:
        # This preserves the repository's extends semantics for a YAML fallback.
        from loom.train.loop import read_config

        value = read_config(path)
    _require(isinstance(value, dict), f"config is not a mapping: {path}")
    return value


def _authenticated_config(
    payload: Mapping[str, Any], config_path: str | os.PathLike | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected = str(payload.get("config_hash", ""))
    _require(bool(expected), "candidate checkpoint has no config_hash")

    embedded = payload.get("resolved_config")
    if embedded is not None:
        _require(isinstance(embedded, dict), "resolved_config is not a mapping")
        cfg = copy.deepcopy(embedded)
        source: dict[str, Any] = {
            "kind": "checkpoint.resolved_config",
            "canonical_sha256": _canonical_json_sha256(cfg),
        }
        if config_path is not None:
            external_path = Path(config_path).resolve()
            external = _read_config(external_path)
            _require(
                _experiment_config_hash(external) == expected,
                "external config hash does not match candidate config_hash",
            )
            _require(
                _experiment_config_hash(external) == _experiment_config_hash(cfg),
                "external and embedded configs describe different experiments",
            )
            source["checked_external_path"] = str(external_path)
            source["checked_external_sha256"] = sha256_file(external_path)
    else:
        _require(
            config_path is not None,
            "candidate has no embedded resolved_config; --config is required",
        )
        external_path = Path(config_path).resolve()
        cfg = _read_config(external_path)
        source = {
            "kind": "external",
            "path": str(external_path),
            "file_sha256": sha256_file(external_path),
            "canonical_sha256": _canonical_json_sha256(cfg),
        }

    got = _experiment_config_hash(cfg)
    _require(
        got == expected,
        f"resolved config hash {got} != checkpoint config_hash {expected}",
    )
    source["experiment_hash"] = got
    return cfg, source


def _assert_module_finite(module: Any, name: str) -> None:
    params = getattr(module, "state_dict", None)
    _require(callable(params), f"candidate is missing usable module {name!r}")
    state = params()
    _require(bool(state), f"candidate module {name!r} has an empty state_dict")
    for key, value in state.items():
        if isinstance(value, Tensor) and value.is_floating_point():
            if not bool(torch.isfinite(value).all()):
                n_bad = int((~torch.isfinite(value)).sum())
                raise GateError(f"candidate {name}.{key} has {n_bad} non-finite values")


def load_candidate(
    checkpoint: str | os.PathLike,
    *,
    config_path: str | os.PathLike | None = None,
    device: str = "cpu",
    audit: dict[str, Any] | None = None,
) -> Candidate:
    """Load a consolidated, config-authenticated real model strictly."""
    ckpt = Path(checkpoint).resolve()
    checkpoint_sha = sha256_file(ckpt)
    if audit is not None:
        audit["candidate_artifact"] = {
            "path": str(ckpt),
            "sha256": checkpoint_sha,
            "bytes": ckpt.stat().st_size,
        }
    try:
        payload = torch.load(
            str(ckpt), map_location="cpu", weights_only=False, mmap=True,
        )
    except Exception as exc:  # noqa: BLE001 - convert to fail-closed gate error
        raise GateError(f"cannot load candidate checkpoint {ckpt}: {exc}") from exc

    _require(isinstance(payload, dict), "candidate checkpoint is not a mapping")
    _require(
        isinstance(payload.get("consolidated"), dict),
        "candidate is not a consolidated checkpoint",
    )
    _require(isinstance(payload.get("model"), dict), "candidate has no model state")
    _require(
        isinstance(payload.get("global_step"), int) and payload["global_step"] >= 0,
        "candidate has no valid global_step",
    )
    cfg, config_source = _authenticated_config(payload, config_path)
    settings = _gate_settings(cfg)

    # Initial construction consumes RNG before strict state restoration.  Pin
    # it as well as proposal sampling so a gate run has one authenticated seed.
    from loom.train.determinism import enable_determinism, set_global_seed

    enable_determinism()
    # The training helper uses warn-only for long-running links.  A promotion
    # gate must instead fail closed if an executed kernel has no deterministic
    # implementation.
    torch.use_deterministic_algorithms(True, warn_only=False)
    set_global_seed(settings["seed"], rank=0)

    # Stubs are never a valid promotion candidate.  Change only the in-memory
    # construction switch; the authenticated config and hash recorded below stay
    # byte-for-byte those from the checkpoint.
    build_cfg = copy.deepcopy(cfg)
    build_cfg.setdefault("model", {})["use_stubs"] = False
    from loom.train.loop import build_model

    try:
        model = build_model(build_cfg)
        incompatible = model.load_state_dict(payload["model"], strict=True)
    except Exception as exc:  # noqa: BLE001
        raise GateError(f"candidate model cannot be built/loaded strictly: {exc}") from exc
    _require(not incompatible.missing_keys, f"missing model keys: {incompatible.missing_keys}")
    _require(
        not incompatible.unexpected_keys,
        f"unexpected model keys: {incompatible.unexpected_keys}",
    )

    try:
        model.requires_grad_(False).eval().to(device)
    except Exception as exc:  # noqa: BLE001
        raise GateError(f"cannot move candidate to {device}: {exc}") from exc
    if str(device).startswith("cuda"):
        model.compute_dtype = torch.bfloat16

    # These are exactly the modules the gate executes.  Scanning them catches a
    # corrupt/non-finite candidate before a finite-looking aggregate can mask it.
    for name in ("estimator", "ema", "bank", "q_action", "proposal"):
        _assert_module_finite(getattr(model, name, None), name)

    provenance = {
        "path": str(ckpt),
        "sha256": checkpoint_sha,
        "bytes": ckpt.stat().st_size,
        "global_step": int(payload["global_step"]),
        "samples_seen": payload.get("samples_seen"),
        "config_hash": str(payload["config_hash"]),
        "saved_git_sha": payload.get("git_sha"),
        "world_size": payload.get("world_size"),
        "consolidated": payload["consolidated"],
        "config_source": config_source,
        "module_types": {
            name: type(getattr(model, name)).__module__ + "." + type(getattr(model, name)).__name__
            for name in ("estimator", "ema", "bank", "q_action", "proposal")
        },
    }
    provenance["determinism"] = _determinism_provenance(settings["seed"])
    return Candidate(
        model=model,
        config=cfg,
        provenance=provenance,
        gate_settings=settings,
    )


def _module_checkpoint_state(
    state: Mapping[str, Any], module: str,
) -> dict[str, Tensor]:
    prefix = module + "."
    return {
        str(key)[len(prefix):]: value
        for key, value in state.items()
        if str(key).startswith(prefix) and isinstance(value, Tensor)
    }


def load_reference_heads(
    reference_checkpoint: str | os.PathLike,
    candidate: Candidate,
    preservation: Mapping[str, Any],
    *,
    device: str,
) -> ReferenceHeads:
    """Authenticate the deploy seed and exact frozen coordinates.

    The joint stage is allowed to move only ``bank`` and ``q_action``. Comparing
    checkpoint tensors directly avoids relying on optimizer gradients as proof
    that AdamW/EMA/resume logic did not mutate another coordinate.
    """
    path = Path(reference_checkpoint).resolve()
    got_sha = sha256_file(path)
    expected_sha = str(preservation["reference_checkpoint_sha256"])
    _require(got_sha == expected_sha,
             f"reference checkpoint sha256 {got_sha} != {expected_sha}")
    try:
        reference = torch.load(
            str(path), map_location="cpu", weights_only=False, mmap=True,
        )
        candidate_payload = torch.load(
            candidate.provenance["path"], map_location="cpu",
            weights_only=False, mmap=True,
        )
    except Exception as exc:  # noqa: BLE001
        raise GateError(f"cannot load preservation checkpoints: {exc}") from exc
    for label, payload in (("reference", reference), ("candidate", candidate_payload)):
        _require(isinstance(payload, Mapping), f"{label} checkpoint is not a mapping")
        _require(isinstance(payload.get("model"), Mapping),
                 f"{label} checkpoint has no model state")
        _require(isinstance(payload.get("consolidated"), Mapping),
                 f"{label} checkpoint is not consolidated")
    _require(
        reference.get("config_hash") == preservation["reference_config_hash"],
        "reference checkpoint config hash does not match QA preservation recipe",
    )
    _require(
        reference.get("global_step") == preservation["reference_global_step"],
        "reference checkpoint global step does not match QA preservation recipe",
    )
    ref_cfg = reference.get("resolved_config")
    cand_cfg = candidate_payload.get("resolved_config")
    _require(isinstance(ref_cfg, Mapping) and isinstance(cand_cfg, Mapping),
             "preservation checkpoints need embedded resolved configs")
    _require(ref_cfg.get("model") == cand_cfg.get("model"),
             "candidate model construction differs from deploy reference")

    ref_state = reference["model"]
    cand_state = candidate_payload["model"]
    frozen = ("estimator", "ema", "q_delta", "decoder", "proposal", "potential")
    frozen_report: dict[str, Any] = {}
    for name in frozen:
        ref = _module_checkpoint_state(ref_state, name)
        cand = _module_checkpoint_state(cand_state, name)
        if not ref and not cand:
            # Potential is not constructed in these configs. Authenticate the
            # shared absence instead of inventing a required empty module.
            _require(name == "potential",
                     f"preservation comparison has no {name} tensors")
            frozen_report[name] = {
                "present": False,
                "tensors": 0,
                "numel": 0,
                "tensor_exact": True,
            }
            continue
        _require(ref and cand, f"frozen {name} presence changed")
        _require(set(ref) == set(cand), f"frozen {name} tensor keys changed")
        mismatched = [key for key in ref if not torch.equal(ref[key], cand[key])]
        _require(not mismatched,
                 f"frozen {name} changed ({len(mismatched)} tensors; first {mismatched[:3]})")
        frozen_report[name] = {
            "present": True,
            "tensors": len(ref),
            "numel": sum(int(value.numel()) for value in ref.values()),
            "tensor_exact": True,
        }

    # Clone only the small body-specific encoder, not the 440M-parameter model.
    # Its architecture comes from the already strict-loaded candidate; its
    # coordinates come exclusively from the authenticated deploy checkpoint.
    q_action = copy.deepcopy(candidate.model.q_action).cpu()
    ref_q_action = _module_checkpoint_state(ref_state, "q_action")
    incompatible = q_action.load_state_dict(ref_q_action, strict=True)
    _require(not incompatible.missing_keys and not incompatible.unexpected_keys,
             "reference q_action did not load strictly")
    q_action.requires_grad_(False).eval().to(device)
    _assert_module_finite(q_action, "reference_q_action")
    return ReferenceHeads(
        q_action=q_action,
        provenance={
            "path": str(path),
            "sha256": got_sha,
            "bytes": path.stat().st_size,
            "global_step": int(reference["global_step"]),
            "config_hash": str(reference["config_hash"]),
            "model_config_exact": True,
            "frozen_modules": frozen_report,
        },
    )


def _finite_tensor(value: Tensor, name: str) -> None:
    _require(value.is_floating_point(), f"{name} must be floating point")
    if not bool(torch.isfinite(value).all()):
        n_bad = int((~torch.isfinite(value)).sum())
        raise GateError(f"{name} has {n_bad} non-finite values")


def proposal_leaf_pairwise_spread(leaves: Tensor) -> Tensor:
    """Mean pairwise ``1-cos(LN(.), LN(.))`` without an ``N x N x K x D`` copy.

    ``leaves`` is ``(B, N, K, D)``.  For unit vectors ``u_i``, the mean of all
    unordered pair dot products is ``(||sum_i u_i||^2 - N)/(N(N-1))``.  Applying
    that identity per slot is exactly the pairwise LN-cos distance used by
    ``L_dyn``, at O(B*N*K*D) memory and work.
    """
    _require(leaves.ndim == 4, f"leaves must be (B,N,K,D), got {tuple(leaves.shape)}")
    n = int(leaves.shape[1])
    _require(n >= 2, "proposal leaf spread needs at least two candidates")
    _finite_tensor(leaves, "proposal leaves")
    x = F.layer_norm(leaves.float(), leaves.shape[-1:])
    unit = F.normalize(x, dim=-1)
    sum_sq = unit.sum(dim=1).square().sum(dim=-1)       # (B,K)
    # Use the computed self-dots, not the idealised value N, so this is exactly
    # the off-diagonal mean after fp32 normalisation and its rounding error.
    self_sum = unit.square().sum(dim=-1).sum(dim=1)     # (B,K)
    mean_cos = (sum_sq - self_sum) / float(n * (n - 1))
    out = (1.0 - mean_cos).mean(dim=-1)
    _finite_tensor(out, "proposal leaf pairwise spread")
    return out


def dynamics_metric_rows(
    bank: Any,
    zs: Sequence[Tensor],
    z_targets: Sequence[Tensor],
    c_seq: Tensor,
    *,
    cosine: str = "per_slot",
) -> tuple[Tensor, Tensor]:
    """Per-primary-window ``delta_sel`` and sequential rollout improvement.

    Inputs contain one context row followed by one or more primary rows.  For
    primary row ``i``, comparison coefficient ``i`` is paired with selected
    coefficient ``i+1``.  The collector makes the context the previous row in
    one global cyclic selected order, independent of runtime chunk size.
    Positive ``delta_sel`` means the action-labelled coefficient predicts its
    own transition better.  The second return is
    ``d(identity, target_h) - d(sequential_rollout_h, target_h)``; positive means
    the bank improves on doing nothing.
    """
    _require(c_seq.ndim == 3, f"c_seq must be (B,H,M), got {tuple(c_seq.shape)}")
    b_context, horizons = int(c_seq.shape[0]), int(c_seq.shape[1])
    _require(b_context >= 2, "metric chunk needs a context and a primary row")
    _require(len(zs) >= horizons + 1, "online belief list is shorter than c_seq")
    _require(len(z_targets) >= horizons + 1, "target belief list is shorter than c_seq")
    for label, beliefs in (("online", zs), ("target", z_targets)):
        for h, value in enumerate(beliefs[:horizons + 1]):
            _require(
                int(value.shape[0]) == b_context,
                f"{label} belief h{h} has {value.shape[0]} rows, expected {b_context}",
            )
    _finite_tensor(c_seq, "action-labelled coefficients")

    n_primary = b_context - 1
    z_in = torch.cat([zs[h][1:] for h in range(horizons)], dim=0)
    z_tgt = torch.cat([z_targets[h + 1][1:] for h in range(horizons)], dim=0)
    c_pos = torch.cat([c_seq[1:, h] for h in range(horizons)], dim=0)
    c_other = torch.cat([c_seq[:-1, h] for h in range(horizons)], dim=0)
    out = bank.step(
        torch.cat([c_pos, c_other], dim=0),
        torch.cat([z_in, z_in], dim=0),
    )
    n = horizons * n_primary
    d_pos = ln_cosine_distance(out[:n], z_tgt, cosine)
    d_other = ln_cosine_distance(out[n:], z_tgt, cosine)
    delta = (d_other - d_pos).reshape(horizons, n_primary).transpose(0, 1)

    rollout = sequential_rollout(bank, zs[0][1:], c_seq[1:])
    gains = []
    for h in range(horizons):
        target = z_targets[h + 1][1:]
        identity_error = ln_cosine_distance(zs[0][1:], target, cosine)
        rollout_error = ln_cosine_distance(rollout[h], target, cosine)
        gains.append(identity_error - rollout_error)
    gain = torch.stack(gains, dim=1)
    _finite_tensor(delta, "delta_sel rows")
    _finite_tensor(gain, "identity-minus-rollout rows")
    return delta, gain


def _batch_size(window: Mapping[str, Any]) -> int:
    actions = window.get("actions")
    _require(isinstance(actions, Tensor), "held-out batch has no action tensor")
    _require(
        actions.ndim == 4 and actions.shape[1] == C.DEPTH
        and actions.shape[2] == C.H_OP,
        "held-out actions must be (B,DEPTH,H_OP,dof), got " + str(tuple(actions.shape)),
    )
    return int(actions.shape[0])


def _validate_batch_meta(
    window: Mapping[str, Any], expected_digest: str, expected_source: str,
) -> tuple[list[str], list[str], list[str]]:
    b = _batch_size(window)
    meta = window.get("data_meta")
    _require(isinstance(meta, Mapping), "gate batch is missing data_meta")
    _require(
        str(meta.get("source", "")) == expected_source,
        "batch data source does not match loader trajectory manifest",
    )
    _require(meta.get("split") == "gate", f"batch split is not gate: {meta.get('split')!r}")
    _require(
        str(meta.get("manifest_digest", "")) == expected_digest,
        "batch manifest_digest does not match loader trajectory manifest",
    )

    def ids(name: str) -> list[str]:
        value = meta.get(name)
        _require(isinstance(value, (list, tuple)), f"data_meta.{name} is missing")
        _require(len(value) == b, f"data_meta.{name} has {len(value)} rows, expected {b}")
        out = [str(v) for v in value]
        _require(all(out), f"data_meta.{name} contains an empty identity")
        return out

    task = ids("task_ids")
    trajectory = ids("trajectory_ids")
    cluster = ids("trajectory_cluster_ids")
    _require(
        trajectory == cluster,
        "trajectory_cluster_ids must identify the exact whole trajectories",
    )
    return task, trajectory, cluster


def _to_device(window: Mapping[str, Any], device: str, dtype: torch.dtype | None) -> dict:
    """Gate-local equivalent of the training mover, retaining ``data_meta``."""
    def move(value: Tensor) -> Tensor:
        value = value.to(device, non_blocking=True)
        if dtype is not None and value.is_floating_point():
            value = value.to(dtype)
        return value

    out = dict(window)
    out["feats"] = [{k: move(v) for k, v in f.items()} for f in window["feats"]]
    if window.get("burn_in_feats") is not None:
        out["burn_in_feats"] = [
            {k: move(v) for k, v in f.items()} for f in window["burn_in_feats"]
        ]
    out["lang"] = move(window["lang"])
    out["actions"] = move(window["actions"])
    return out


def _seed64(seed: int, tag: str) -> int:
    raw = hashlib.blake2b(f"{int(seed)}|{tag}".encode(), digest_size=8).digest()
    return int.from_bytes(raw, "little") & ((1 << 63) - 1)


def cyclic_context_chunks(
    selected: Sequence[Any], batch_size: int,
) -> Iterable[tuple[int, list[Any], list[Any]]]:
    """Yield ``(ordinal, primaries, [cyclic_predecessor, *primaries])``."""
    _require(batch_size > 0, "batch_size must be positive")
    values = list(selected)
    _require(bool(values), "selected window order is empty")
    n = len(values)
    for lo in range(0, n, batch_size):
        primaries = values[lo:lo + batch_size]
        yield lo, primaries, [values[(lo - 1) % n], *primaries]


def _first_sampler_indices(
    sampler: Any, body: str, windows: int, *, rank: int = 0,
) -> np.ndarray:
    """Flatten the first fixed uniform-task sample ordinals from a sampler."""
    _require(windows > 0, "windows must be positive")
    _require(getattr(sampler, "sampling", None) == "uniform_task",
             "gate sampler must use uniform_task")
    out: list[int] = []
    step = 0
    while len(out) < windows:
        selected_body, indices = sampler.batch_at(step, rank)
        _require(str(selected_body) == str(body),
                 f"sampler selected {selected_body!r}, expected {body!r}")
        out.extend(int(i) for i in np.asarray(indices).reshape(-1))
        step += 1
    result = np.asarray(out[:windows], dtype=np.int64)
    _require(len(np.unique(result)) == windows,
             "the first fixed uniform-task rows contain duplicate windows")
    return result


def proposal_plans_from_logits(
    logits: Tensor,
    *,
    n_candidates: int,
    depth: int,
    seed: int,
    ordinals: Sequence[int],
    topk: int = C.TOPK,
) -> Tensor:
    """Sample plans with one RNG stream per fixed selected-window ordinal."""
    _require(logits.ndim == 2, f"proposal logits must be (B,M), got {logits.shape}")
    _require(n_candidates >= 2, "proposal candidate count must be >=2")
    _require(depth > 0, "proposal depth must be positive")
    fixed_ordinals = [int(value) for value in ordinals]
    _require(len(fixed_ordinals) == logits.shape[0], "proposal ordinal count mismatch")
    _require(len(set(fixed_ordinals)) == len(fixed_ordinals),
             "proposal ordinals must be unique")
    from loom.heads.proposal import gumbel_topk, weights_from_logits

    rows: list[Tensor] = []
    m = int(logits.shape[-1])
    _require(1 <= int(topk) <= m, "proposal topk is outside operator width")
    for row, ordinal in zip(logits, fixed_ordinals):
        generator = torch.Generator(device=row.device)
        generator.manual_seed(_seed64(seed, f"bank_ca_gate/proposal/{ordinal}"))
        wide = row.view(1, 1, m).expand(n_candidates, depth, m)
        support = gumbel_topk(wide, int(topk), generator=generator)
        rows.append(weights_from_logits(wide, support, m))
    plans = torch.stack(rows, dim=0)
    _finite_tensor(plans, "proposal plans")
    return plans


@torch.inference_mode()
def proposal_root_q_action_residuals(
    decoder: Any,
    q_action: Any,
    proprio: Tensor,
    z: Tensor,
    root_coeff: Tensor,
    *,
    seed: int,
    ordinals: Sequence[int],
) -> Tensor:
    """Measure realizability of proposal roots with common decoder noise.

    Each fixed held-out ordinal gets one SHA-derived decoder noise segment.  That
    exact segment is expanded across all proposal candidates for the root, so a
    residual difference can only come from the coefficient and not from a noise
    lottery.  A separate generator per ordinal makes the result independent of
    runtime chunking.
    """
    _require(root_coeff.ndim == 3 and root_coeff.shape[-1] == C.M,
             f"proposal roots must be (B,N,{C.M}), got {root_coeff.shape}")
    b, n_candidates, _ = root_coeff.shape
    _require(proprio.ndim == 2 and proprio.shape[0] == b,
             "proposal-root proprio shape does not match coefficients")
    _require(z.ndim == 3 and z.shape[0] == b,
             "proposal-root belief shape does not match coefficients")
    fixed_ordinals = [int(value) for value in ordinals]
    _require(len(fixed_ordinals) == b, "proposal-root ordinal count mismatch")
    _require(len(set(fixed_ordinals)) == b, "proposal-root ordinals must be unique")

    rows: list[Tensor] = []
    for row, ordinal in enumerate(fixed_ordinals):
        c = root_coeff[row]
        p = proprio[row:row + 1].expand(n_candidates, -1)
        belief = z[row:row + 1].expand(n_candidates, -1, -1)
        generator = torch.Generator(device=proprio.device)
        generator.manual_seed(_seed64(seed, f"bank_ca_gate/decoder/{ordinal}"))
        noise = torch.randn(
            1, C.H_OP, proprio.shape[-1], device=proprio.device,
            dtype=proprio.dtype, generator=generator,
        ).expand(n_candidates, -1, -1)
        segment = decoder(p, c, noise=noise)
        _require(isinstance(segment, Tensor), "decoder did not return a tensor")
        _require(segment.shape == noise.shape,
                 f"decoder root segment has wrong shape {segment.shape}")
        c_hat = q_action(segment, belief)
        if isinstance(c_hat, tuple):
            c_hat = c_hat[0]
        _require(isinstance(c_hat, Tensor) and c_hat.shape == c.shape,
                 "q_action residual coefficient has the wrong shape")
        residual = (c_hat.float() - c.float()).norm(dim=-1)
        _finite_tensor(residual, "proposal-root q_action residual")
        rows.append(residual)
    out = torch.stack(rows, dim=0)
    _require(out.shape == (b, n_candidates), "proposal-root residual matrix mismatch")
    return out


@torch.inference_mode()
def action_anchor_preservation_rows(
    decoder: Any,
    proposal: Any,
    reference_q_action: Any,
    actions: Tensor,
    zs: Sequence[Tensor],
    proprio: Sequence[Tensor],
    lang: Tensor,
    candidate_coeff: Tensor,
    *,
    seed: int,
    ordinals: Sequence[int],
) -> tuple[Tensor, Tensor]:
    """Paired candidate-minus-reference action semantics for primary rows.

    Positive decode improvement means the candidate q_action coefficient lowers
    frozen-D_e CFM error. Positive overlap change means it agrees with the
    frozen proposal support at least as well as the deploy q_action did. Every
    comparison shares the exact noise/time draw and input belief.
    """
    _require(candidate_coeff.ndim == 3 and candidate_coeff.shape[1:] == (C.DEPTH, C.M),
             "candidate action coefficients have the wrong shape")
    b = int(candidate_coeff.shape[0])
    fixed_ordinals = [int(value) for value in ordinals]
    _require(len(fixed_ordinals) == b and len(set(fixed_ordinals)) == b,
             "action-anchor ordinals must be unique and match the primary rows")
    _require(actions.shape[:2] == (b, C.DEPTH),
             "action-anchor actions do not match primary rows")
    _require(len(zs) == C.DEPTH and len(proprio) == C.DEPTH,
             "action-anchor inputs do not cover every horizon")

    decode_rows: list[Tensor] = []
    overlap_rows: list[Tensor] = []
    for row, ordinal in enumerate(fixed_ordinals):
        decode_h: list[Tensor] = []
        overlap_h: list[Tensor] = []
        for h in range(C.DEPTH):
            action = actions[row:row + 1, h]
            belief = zs[h][row:row + 1]
            value = reference_q_action(action, belief)
            c_ref = value[0] if isinstance(value, tuple) else value
            c_new = candidate_coeff[row:row + 1, h]
            _require(c_ref.shape == c_new.shape, "reference q_action shape changed")

            generator = torch.Generator(device=action.device)
            generator.manual_seed(
                _seed64(seed, f"bank_ca_gate/action_anchor/{ordinal}/{h}")
            )
            noise = torch.randn(
                1, C.H_OP, action.shape[-1], device=action.device,
                dtype=action.dtype, generator=generator,
            )
            t = torch.rand(1, device=action.device, dtype=torch.float32,
                           generator=generator)
            p = proprio[h][row:row + 1]
            ref_loss = decoder.loss(
                p, c_ref, action, t=t, noise=noise, reduction="none",
            )
            new_loss = decoder.loss(
                p, c_new, action, t=t, noise=noise, reduction="none",
            )
            decode_h.append((ref_loss - new_loss).float().squeeze(0))

            proposal_logits = proposal.logits(
                belief, lang[row:row + 1],
            )
            proposal_idx = proposal_logits.float().topk(C.TOPK, dim=-1).indices
            new_idx = c_new.float().topk(C.TOPK, dim=-1).indices
            ref_idx = c_ref.float().topk(C.TOPK, dim=-1).indices

            def overlap(index: Tensor) -> Tensor:
                return (
                    (proposal_idx.unsqueeze(-1) == index.unsqueeze(-2))
                    .any(-1).float().mean(-1)
                )

            overlap_h.append((overlap(new_idx) - overlap(ref_idx)).squeeze(0))
        decode_rows.append(torch.stack(decode_h).mean())
        overlap_rows.append(torch.stack(overlap_h).mean())

    decode = torch.stack(decode_rows)
    overlap = torch.stack(overlap_rows)
    _finite_tensor(decode, "action decode improvement")
    _finite_tensor(overlap, "proposal support overlap change")
    return decode, overlap


def _validate_action_coefficients(c_seq: Tensor) -> None:
    _require(
        c_seq.ndim == 3 and c_seq.shape[1:] == (C.DEPTH, C.M),
        f"q_action coefficients must be (B,{C.DEPTH},{C.M}), got {c_seq.shape}",
    )
    _finite_tensor(c_seq, "detached q_action coefficients")
    fp32 = c_seq.float()
    _require(bool((fp32 >= 0).all()), "q_action coefficients contain negative weights")
    support = (fp32 > 0).sum(dim=-1)
    _require(bool((support == C.TOPK).all()),
             f"q_action coefficients must have exactly {C.TOPK} nonzeros")
    _require(
        bool(torch.allclose(
            fp32.sum(dim=-1), torch.ones_like(fp32[..., 0]), atol=5e-3, rtol=0,
        )),
        "q_action coefficients do not sum to one",
    )


@torch.inference_mode()
def measure_batch(
    model: Any,
    window: Mapping[str, Any],
    *,
    n_candidates: int,
    seed: int,
    ordinals: Sequence[int],
    device: str,
    cosine: str,
    measure_residual: bool = False,
    reference_q_action: Any | None = None,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray | None,
    np.ndarray | None, np.ndarray | None,
]:
    # Cached features are fp16.  CPU layers are fp32, while production CUDA
    # execution uses bf16 autocast, so both paths need an explicit input cast.
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    batch = _to_device(window, device, dtype)
    b_context = _batch_size(batch)
    _require(len(ordinals) == b_context - 1,
             "fixed selected ordinals do not match primary metric rows")
    emb = str(batch.get("embodiment", ""))
    _require(bool(emb), "held-out batch has no embodiment")
    _require(batch.get("actions") is not None, "held-out gate requires actions")
    _require(len(batch.get("burn_in_feats", ())) == 4,
             "held-out gate requires exactly B4 recurrent burn-in")
    _require(len(batch.get("feats", ())) == C.DEPTH + 1,
             f"held-out gate requires exactly {C.DEPTH + 1} main states")

    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if str(device).startswith("cuda") else nullcontext()
    )
    with autocast:
        zs = model.beliefs(batch)
        zts = model.target_beliefs(batch)
        _require(len(zs) == C.DEPTH + 1, f"expected {C.DEPTH + 1} online beliefs")
        _require(len(zts) == C.DEPTH + 1, f"expected {C.DEPTH + 1} target beliefs")
        qa = model.q_action[emb]
        coeffs = []
        for h in range(C.DEPTH):
            value = qa(batch["actions"][:, h], zs[h])
            if isinstance(value, tuple):
                value = value[0]
            coeffs.append(value.detach())
        c_seq = torch.stack(coeffs, dim=1)
        _require(not c_seq.requires_grad, "q_action coefficients must be detached")
        _validate_action_coefficients(c_seq)
        delta, gain = dynamics_metric_rows(
            model.bank, zs, zts, c_seq, cosine=cosine,
        )

        proposal_z = zs[0][1:]
        proposal_lang = batch["lang"][1:]
        logits = model.proposal.logits(proposal_z, proposal_lang)
        plans = proposal_plans_from_logits(
            logits,
            n_candidates=n_candidates,
            depth=C.DEPTH,
            seed=seed,
            ordinals=ordinals,
            topk=int(getattr(model.proposal, "topk", C.TOPK)),
        )
        leaves = model.bank.rollout(plans, proposal_z)
        spread = proposal_leaf_pairwise_spread(leaves)
        root_residual = None
        decode_improvement = None
        support_overlap_change = None
        if measure_residual:
            _require(reference_q_action is not None,
                     "joint preservation metrics require the deploy q_action")
            root_residual = proposal_root_q_action_residuals(
                model.decoder[emb], qa,
                batch["feats"][0]["proprio"][1:], proposal_z,
                plans[:, :, 0, :], seed=seed, ordinals=ordinals,
            )
            decode_improvement, support_overlap_change = action_anchor_preservation_rows(
                model.decoder[emb], model.proposal, reference_q_action[emb],
                batch["actions"][1:], [z[1:] for z in zs[:C.DEPTH]],
                [batch["feats"][h]["proprio"][1:] for h in range(C.DEPTH)],
                batch["lang"][1:], c_seq[1:], seed=seed, ordinals=ordinals,
            )
        else:
            _require(reference_q_action is None,
                     "bank-only metric collection received an undeclared reference")

    return (
        delta.float().cpu().numpy(),
        gain.float().cpu().numpy(),
        spread.float().cpu().numpy(),
        None if root_residual is None else root_residual.float().cpu().numpy(),
        (None if decode_improvement is None
         else decode_improvement.float().cpu().numpy()),
        (None if support_overlap_change is None
         else support_overlap_change.float().cpu().numpy()),
    )


def validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the gate split manifest and independently authenticate its body."""
    required = {
        "version", "source", "split", "holdout_demo_keys", "n_tasks",
        "n_trajectories", "tasks", "trajectory_ids", "digest",
    }
    missing = sorted(required - set(manifest))
    _require(not missing, f"trajectory manifest is missing fields: {missing}")
    _require(manifest["version"] == 1, f"unsupported manifest version {manifest['version']!r}")
    _require(manifest["source"] == "libero",
             f"manifest source is not libero: {manifest['source']!r}")
    _require(manifest["split"] == "gate", f"manifest split is not gate: {manifest['split']!r}")
    _require(isinstance(manifest["tasks"], Mapping), "manifest.tasks is not a mapping")
    _require(isinstance(manifest["trajectory_ids"], list), "manifest.trajectory_ids is not a list")
    _require(int(manifest["n_tasks"]) == len(manifest["tasks"]), "manifest n_tasks mismatch")
    _require(
        int(manifest["n_trajectories"]) == len(manifest["trajectory_ids"]),
        "manifest n_trajectories mismatch",
    )
    _require(int(manifest["n_tasks"]) > 1, "gate manifest needs at least two tasks")
    _require(int(manifest["n_trajectories"]) > 1, "gate manifest needs at least two trajectories")
    _require(
        manifest["holdout_demo_keys"] == ["demo_49"],
        "manifest holdout_demo_keys must be exactly ['demo_49']",
    )

    flat = sorted(str(x) for values in manifest["tasks"].values() for x in values)
    ids = sorted(str(x) for x in manifest["trajectory_ids"])
    _require(flat == ids, "manifest tasks do not partition trajectory_ids exactly")
    digest = str(manifest["digest"])
    _require(digest.startswith("sha256:"), "manifest digest is not sha256-prefixed")
    body = {k: manifest[k] for k in manifest if k != "digest"}
    independent = "sha256:" + _canonical_json_sha256(body)
    _require(
        digest == independent,
        f"manifest digest {digest} != independently computed {independent}",
    )
    return copy.deepcopy(dict(manifest))


def _loader_source_provenance(
    loader: Any,
) -> tuple[dict[str, Any], str, str, dict[str, str]]:
    datasets = getattr(loader, "datasets", None)
    _require(isinstance(datasets, Mapping), "gate loader exposes no datasets mapping")
    _require(len(datasets) == 1, f"bank gate requires one embodiment, got {sorted(datasets)}")
    embodiment, dataset = next(iter(datasets.items()))
    manifest_fn = getattr(loader, "trajectory_manifest", None)
    _require(callable(manifest_fn), "gate loader has no trajectory_manifest()")
    manifest = validate_manifest(manifest_fn())

    cache = getattr(dataset, "cache", None)
    _require(cache is not None, "gate dataset exposes no feature cache")
    raw_cache_root = getattr(cache, "root", None)
    _require(raw_cache_root is not None and str(raw_cache_root),
             "gate feature cache has no root identity")
    cache_root = Path(raw_cache_root).resolve()
    cache_manifest = cache_root / "manifest.json"
    _require(cache_manifest.is_file(), f"feature cache manifest missing: {cache_manifest}")
    provenance = {
        "embodiment": str(embodiment),
        "trajectory_manifest": manifest,
        "manifest_digest": manifest["digest"],
        "feature_cache_root": str(cache_root),
        "feature_cache_manifest": str(cache_manifest),
        "feature_cache_manifest_sha256": sha256_file(cache_manifest),
        "loader_n_windows": int(getattr(loader, "n_windows", -1)),
        "sampling": getattr(loader, "sampling", None),
    }
    task_for_trajectory = {
        str(trajectory): str(task)
        for task, trajectories in manifest["tasks"].items()
        for trajectory in trajectories
    }
    return (
        provenance,
        str(manifest["digest"]),
        str(manifest["source"]),
        task_for_trajectory,
    )


def _selected_window_records(
    loader: Any,
    indices: Sequence[int],
    task_for_trajectory: Mapping[str, str],
) -> list[dict[str, Any]]:
    datasets = getattr(loader, "datasets", {})
    _require(len(datasets) == 1, "selected-window identity needs one gate dataset")
    dataset = next(iter(datasets.values()))
    _require(int(getattr(dataset, "recurrent_burn_in", -1)) == 4,
             "gate dataset does not use exact B4 recurrent burn-in")
    indexed_windows = getattr(dataset, "windows", None)
    _require(isinstance(indexed_windows, list), "gate dataset exposes no window index")
    records: list[dict[str, Any]] = []
    identities: set[tuple[str, int]] = set()
    for ordinal, raw_index in enumerate(indices):
        index = int(raw_index)
        _require(0 <= index < len(indexed_windows), f"selected dataset index {index} is invalid")
        window = indexed_windows[index]
        trajectory = str(window.traj_id)
        start = int(window.start)
        _require(start >= 4 * C.H_OP,
                 f"selected B4 window {trajectory}@{start} has no full prefix")
        _require(len(window.obs_src_index) == C.DEPTH + 1,
                 f"selected window {trajectory}@{start} has a malformed state index")
        _require(not bool(window.action_free),
                 f"selected held-out window {trajectory}@{start} has no action labels")
        identity = (trajectory, start)
        _require(identity not in identities,
                 f"selected held-out window identity is duplicated: {identity}")
        identities.add(identity)
        task = task_for_trajectory.get(trajectory)
        _require(bool(task), f"selected trajectory {trajectory!r} is absent from manifest tasks")
        records.append({
            "ordinal": ordinal,
            "dataset_index": index,
            "task_id": str(task),
            "trajectory_id": trajectory,
            "canonical_start": start,
            "source_observation_indices": [int(v) for v in window.obs_src_index],
        })
    return records


def collect_metric_rows(
    model: Any,
    loader: Any,
    *,
    windows: int,
    batch_size: int,
    n_candidates: int,
    seed: int,
    device: str,
    cosine: str = "per_slot",
    measure_residual: bool = False,
    reference_q_action: Any | None = None,
    audit: dict[str, Any] | None = None,
) -> tuple[MetricRows, dict[str, Any]]:
    _require(windows > 0, "windows must be positive")
    _require(batch_size > 0, "batch_size must be positive")
    _require(n_candidates >= 2, "candidates must be >=2")
    _require(
        int(getattr(loader, "n_windows", -1)) >= windows,
        f"gate split has fewer than requested {windows} windows",
    )
    source, digest, manifest_source, task_for_trajectory = _loader_source_provenance(loader)
    if audit is not None:
        audit["data"] = copy.deepcopy(source)
    manifest_tasks = set(task_for_trajectory.values())
    _require(
        getattr(loader, "batch_size", None) == batch_size,
        f"gate loader batch_size {getattr(loader, 'batch_size', None)} != {batch_size}",
    )
    _require(getattr(loader, "sampling", None) == "uniform_task",
             "gate loader must select the fixed order with uniform_task sampling")
    datasets = getattr(loader, "datasets", {})
    embodiment, dataset = next(iter(datasets.items()))
    selected = _first_sampler_indices(loader.sampler, str(embodiment), windows)
    records = _selected_window_records(loader, selected, task_for_trajectory)
    _require({record["task_id"] for record in records} == manifest_tasks,
             "first fixed uniform-task rows do not cover every gate task")

    chunks = list(cyclic_context_chunks(selected.tolist(), batch_size))
    context_batches = [context for _, _, context in chunks]
    fit = getattr(loader, "_fit_shared_memory", None)
    if callable(fit):
        fit()
    effective_workers = int(getattr(loader, "effective_workers", 0))
    effective_prefetch = int(getattr(loader, "effective_prefetch", 2))
    dl_kwargs: dict[str, Any] = {}
    if effective_workers > 0:
        dl_kwargs.update(
            prefetch_factor=effective_prefetch,
            persistent_workers=False,
        )
    from loom.data.loader import collate_window

    batches: Iterable[Mapping[str, Any]] = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=context_batches,
        collate_fn=collate_window,
        num_workers=effective_workers,
        pin_memory=bool(getattr(loader, "pin_memory", False)),
        **dl_kwargs,
    )

    deltas: list[np.ndarray] = []
    gains: list[np.ndarray] = []
    spreads: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    decode_improvements: list[np.ndarray] = []
    support_overlap_changes: list[np.ndarray] = []
    tasks: list[str] = []
    trajectories: list[str] = []
    clusters: list[str] = []
    try:
        n_seen_batches = 0
        for (lo, primaries, _), window in zip(chunks, batches):
            n_seen_batches += 1
            task, trajectory, cluster = _validate_batch_meta(
                window, digest, manifest_source,
            )
            # Row zero is pairing context only.  Every remaining row is a
            # primary, in the authenticated selected order.
            task = task[1:]
            trajectory = trajectory[1:]
            cluster = cluster[1:]
            expected_records = records[lo:lo + len(primaries)]
            _require(len(task) == len(expected_records), "primary metadata row count mismatch")
            for record, task_id, trajectory_id in zip(expected_records, task, trajectory):
                _require(
                    record["task_id"] == task_id
                    and record["trajectory_id"] == trajectory_id,
                    "collated primary order does not match selected-window identity",
                )
            d, g, s, r, action_delta, support_delta = measure_batch(
                model, window, n_candidates=n_candidates,
                seed=seed,
                ordinals=range(lo, lo + len(primaries)),
                device=device,
                cosine=cosine,
                measure_residual=measure_residual,
                reference_q_action=reference_q_action,
            )
            expected_shape = (len(primaries), C.DEPTH)
            _require(d.shape == expected_shape, f"bad delta_sel shape {d.shape}")
            _require(g.shape == expected_shape, f"bad rollout gain shape {g.shape}")
            _require(s.shape == (len(primaries),), f"bad leaf spread shape {s.shape}")
            if measure_residual:
                _require(r is not None, "joint gate omitted proposal-root residuals")
                _require(r.shape == (len(primaries), n_candidates),
                         f"bad proposal-root residual shape {r.shape}")
                _require(action_delta is not None and
                         action_delta.shape == (len(primaries),),
                         "bad action-decode preservation shape")
                _require(support_delta is not None and
                         support_delta.shape == (len(primaries),),
                         "bad proposal-support preservation shape")
                residuals.append(r)
                decode_improvements.append(action_delta)
                support_overlap_changes.append(support_delta)
            else:
                _require(r is None, "bank-only gate returned undeclared residuals")
                _require(action_delta is None and support_delta is None,
                         "bank-only gate returned undeclared preservation metrics")
            deltas.append(d); gains.append(g); spreads.append(s)
            tasks.extend(task); trajectories.extend(trajectory); clusters.extend(cluster)
    except GateError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise GateError(f"held-out metric collection failed: {exc}") from exc

    _require(n_seen_batches == len(chunks),
             f"loader yielded {n_seen_batches}/{len(chunks)} context chunks")
    rows = MetricRows(
        delta_sel=np.concatenate(deltas, axis=0),
        identity_minus_rollout=np.concatenate(gains, axis=0),
        leaf_spread=np.concatenate(spreads, axis=0),
        task_ids=tasks,
        trajectory_ids=trajectories,
        cluster_ids=clusters,
        root_q_action_residual=(
            np.concatenate(residuals, axis=0) if measure_residual else None
        ),
        action_decode_improvement=(
            np.concatenate(decode_improvements, axis=0) if measure_residual else None
        ),
        proposal_support_overlap_change=(
            np.concatenate(support_overlap_changes, axis=0)
            if measure_residual else None
        ),
    )
    _require(rows.n == windows, f"collected {rows.n}/{windows} metric rows")
    _require(set(rows.task_ids) == manifest_tasks, "selected windows do not cover every gate task")
    for name, value in (
        ("delta_sel", rows.delta_sel),
        ("identity_minus_rollout", rows.identity_minus_rollout),
        ("leaf_spread", rows.leaf_spread),
    ):
        _require(np.isfinite(value).all(), f"collected {name} contains non-finite values")
    if measure_residual:
        _require(rows.root_q_action_residual is not None, "missing residual matrix")
        _require(rows.root_q_action_residual.shape == (windows, n_candidates),
                 "collected residual matrix has the wrong shape")
        _require(np.isfinite(rows.root_q_action_residual).all(),
                 "collected proposal-root residual contains non-finite values")
        _require(rows.action_decode_improvement is not None and
                 rows.action_decode_improvement.shape == (windows,),
                 "collected action-decode preservation has the wrong shape")
        _require(rows.proposal_support_overlap_change is not None and
                 rows.proposal_support_overlap_change.shape == (windows,),
                 "collected support-overlap preservation has the wrong shape")
        _require(np.isfinite(rows.action_decode_improvement).all(),
                 "collected action-decode preservation contains non-finite values")
        _require(np.isfinite(rows.proposal_support_overlap_change).all(),
                 "collected support-overlap preservation contains non-finite values")

    source["selected_windows"] = windows
    source["selection_rule"] = "first N rows of seeded uniform_task ordinal stream"
    source["selection_unique"] = True
    source["selected_window_order"] = records
    source["selected_window_order_sha256"] = (
        "sha256:" + _canonical_json_sha256(records)
    )
    source["delta_sel_pairing"] = (
        "global cyclic previous selected row: primary ordinal i uses "
        "comparison ordinal (i-1) mod N"
    )
    source["selected_task_count"] = len(set(tasks))
    source["selected_trajectory_count"] = len(set(trajectories))
    source["selected_cluster_count"] = len(set(clusters))
    source["runtime_batch_size"] = batch_size
    source["requested_workers"] = int(getattr(loader, "num_workers", 0))
    source["effective_workers"] = effective_workers
    if audit is not None:
        audit["data"] = copy.deepcopy(source)
    return rows, source


def _metric_matrix(rows: MetricRows) -> tuple[np.ndarray, list[str]]:
    _require(rows.delta_sel.ndim == 2, "delta_sel rows must be a matrix")
    _require(rows.identity_minus_rollout.shape == rows.delta_sel.shape,
             "rollout and delta_sel horizons differ")
    _require(rows.delta_sel.shape[1] == C.DEPTH,
             f"expected {C.DEPTH} metric horizons")
    _require(rows.leaf_spread.shape == (rows.n,), "leaf spread row count mismatch")
    names = (
        [f"delta_sel/h{h + 1}" for h in range(C.DEPTH)]
        + [f"identity_minus_rollout/h{h + 1}" for h in range(C.DEPTH)]
        + ["proposal_leaf_pairwise_ln_cos_spread"]
    )
    matrix = np.concatenate(
        [rows.delta_sel, rows.identity_minus_rollout, rows.leaf_spread[:, None]], axis=1,
    ).astype(np.float64, copy=False)
    _require(np.isfinite(matrix).all(), "metric matrix contains non-finite values")
    _require(matrix.shape == (rows.n, len(names)), "metric matrix shape mismatch")
    return matrix, names


def paired_cluster_bootstrap(
    values: np.ndarray,
    task_ids: Sequence[str],
    cluster_ids: Sequence[str],
    *,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> dict[str, Any]:
    """Task-balanced, whole-trajectory clustered percentile bootstrap.

    Windows are first averaged within their exact trajectory cluster.  The point
    estimate is the mean of trajectory means within each task, then the mean of
    task means.  Every bootstrap replicate samples tasks with replacement and,
    within each selected task, trajectory clusters with replacement.  One draw
    is shared by every metric column, preserving all paired differences.
    """
    x = np.asarray(values, dtype=np.float64)
    _require(x.ndim == 2 and x.shape[0] > 0, "bootstrap values must be nonempty (N,P)")
    _require(np.isfinite(x).all(), "bootstrap values contain non-finite numbers")
    _require(samples >= 1, "bootstrap samples must be positive")
    _require(0.0 < confidence < 1.0, "confidence must be between 0 and 1")
    _require(len(task_ids) == x.shape[0], "task identity count mismatch")
    _require(len(cluster_ids) == x.shape[0], "cluster identity count mismatch")

    task_for_cluster: dict[str, str] = {}
    rows_by_cluster: dict[str, list[int]] = {}
    for i, (task, cluster) in enumerate(zip(task_ids, cluster_ids)):
        task, cluster = str(task), str(cluster)
        _require(task and cluster, "empty task/trajectory cluster identity")
        old = task_for_cluster.setdefault(cluster, task)
        _require(old == task, f"trajectory cluster {cluster!r} spans tasks {old!r}/{task!r}")
        rows_by_cluster.setdefault(cluster, []).append(i)

    cluster_mean = {
        cluster: x[np.asarray(idx, dtype=np.int64)].mean(axis=0)
        for cluster, idx in rows_by_cluster.items()
    }
    clusters_by_task: dict[str, list[str]] = {}
    for cluster, task in task_for_cluster.items():
        clusters_by_task.setdefault(task, []).append(cluster)
    tasks = sorted(clusters_by_task)
    _require(len(tasks) >= 2, "cluster bootstrap needs at least two tasks")
    _require(len(cluster_mean) >= 2, "cluster bootstrap needs at least two trajectories")
    per_task = [
        np.stack([cluster_mean[c] for c in sorted(clusters_by_task[t])], axis=0)
        for t in tasks
    ]
    point = np.stack([v.mean(axis=0) for v in per_task], axis=0).mean(axis=0)

    rng = np.random.default_rng(int(seed))
    draws = np.empty((samples, x.shape[1]), dtype=np.float64)
    n_tasks = len(tasks)
    for r in range(samples):
        task_pick = rng.integers(0, n_tasks, size=n_tasks)
        total = np.zeros(x.shape[1], dtype=np.float64)
        for task_index in task_pick:
            table = per_task[int(task_index)]
            cluster_pick = rng.integers(0, table.shape[0], size=table.shape[0])
            total += table[cluster_pick].mean(axis=0)
        draws[r] = total / float(n_tasks)

    alpha = (1.0 - confidence) / 2.0
    lower = np.quantile(draws, alpha, axis=0)
    upper = np.quantile(draws, 1.0 - alpha, axis=0)
    for name, value in (("point", point), ("lower", lower), ("upper", upper)):
        _require(np.isfinite(value).all(), f"bootstrap {name} contains non-finite values")
    return {
        "point": point,
        "lower": lower,
        "upper": upper,
        "samples": int(samples),
        "confidence": float(confidence),
        "seed": int(seed),
        "n_tasks": n_tasks,
        "n_trajectory_clusters": len(cluster_mean),
        "n_windows": int(x.shape[0]),
        "estimator": "task-mean(trajectory-cluster-mean(window paired differences))",
        "resampling": "tasks with replacement; trajectories within selected task with replacement",
    }


def summarize_gate(
    rows: MetricRows,
    *,
    requirements: Mapping[str, float],
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
    preservation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _require(set(requirements) == set(REQUIREMENT_KEYS),
             "gate summary requirements do not have the exact authenticated keys")
    thresholds = {key: float(requirements[key]) for key in REQUIREMENT_KEYS}
    _require(all(math.isfinite(value) for value in thresholds.values()),
             "gate summary thresholds must be finite")
    matrix, names = _metric_matrix(rows)
    boot = paired_cluster_bootstrap(
        matrix, rows.task_ids, rows.cluster_ids,
        samples=bootstrap_samples, confidence=confidence, seed=seed,
    )
    metric_thresholds = (
        [thresholds[REQUIREMENT_KEYS[0]]] * C.DEPTH
        + [thresholds[REQUIREMENT_KEYS[1]]] * C.DEPTH
        + [thresholds[REQUIREMENT_KEYS[2]]]
    )
    metrics: dict[str, Any] = {}
    failures: list[str] = []
    for i, name in enumerate(names):
        point = float(boot["point"][i])
        lower = float(boot["lower"][i])
        upper = float(boot["upper"][i])
        threshold = metric_thresholds[i]
        passed = math.isfinite(lower) and lower > threshold
        metrics[name] = {
            "point": point,
            "ci_lower": lower,
            "ci_upper": upper,
            "threshold": threshold,
            "comparison": "ci_lower strictly_greater_than threshold",
            "passed": passed,
        }
        if not passed:
            failures.append(
                f"{name}: {confidence:.1%} CI lower {lower!r} <= threshold {threshold!r}"
            )

    delta_lower = [float(value) for value in boot["lower"][:C.DEPTH]]
    rollout_lower = [
        float(value) for value in boot["lower"][C.DEPTH:2 * C.DEPTH]
    ]
    spread_lower = float(boot["lower"][-1])
    gates = {
        REQUIREMENT_KEYS[0]: {
            "threshold": thresholds[REQUIREMENT_KEYS[0]],
            "ci_lower_per_horizon": delta_lower,
            "comparison": "every ci_lower strictly_greater_than threshold",
            "passed": all(
                value > thresholds[REQUIREMENT_KEYS[0]] for value in delta_lower
            ),
        },
        REQUIREMENT_KEYS[1]: {
            "threshold": thresholds[REQUIREMENT_KEYS[1]],
            "ci_lower_per_horizon": rollout_lower,
            "comparison": "every ci_lower strictly_greater_than threshold",
            "passed": all(
                value > thresholds[REQUIREMENT_KEYS[1]] for value in rollout_lower
            ),
        },
        REQUIREMENT_KEYS[2]: {
            "threshold": thresholds[REQUIREMENT_KEYS[2]],
            "ci_lower": spread_lower,
            "comparison": "ci_lower strictly_greater_than threshold",
            "passed": spread_lower > thresholds[REQUIREMENT_KEYS[2]],
        },
    }
    if preservation is None:
        _require(rows.root_q_action_residual is None,
                 "residual rows were collected without an authenticated preservation gate")
        _require(rows.action_decode_improvement is None and
                 rows.proposal_support_overlap_change is None,
                 "baseline-relative rows were collected without a preservation gate")
    else:
        _require(set(preservation) == set(PRESERVATION_KEYS),
                 "preservation settings do not have the exact authenticated keys")
        residual_max = float(preservation["q_action_residual_max"])
        exhaustion_max = float(preservation["max_root_exhaustion_rate"])
        action_threshold = float(preservation["action_decode_improvement_ci_low"])
        support_threshold = float(
            preservation["proposal_support_overlap_change_ci_low"]
        )
        _require(math.isfinite(residual_max) and residual_max >= 0.0,
                 "q_action residual maximum must be finite and nonnegative")
        _require(math.isfinite(exhaustion_max) and 0.0 < exhaustion_max < 1.0,
                 "root exhaustion maximum must be strictly between zero and one")
        residual = rows.root_q_action_residual
        _require(residual is not None, "joint gate has no proposal-root residual rows")
        _require(residual.ndim == 2 and residual.shape[0] == rows.n,
                 "proposal-root residual matrix has the wrong shape")
        _require(residual.shape[1] >= 2,
                 "proposal-root residual gate requires at least two candidates")
        _require(np.isfinite(residual).all(),
                 "proposal-root residual matrix contains non-finite values")
        action_delta = rows.action_decode_improvement
        support_delta = rows.proposal_support_overlap_change
        _require(action_delta is not None and action_delta.shape == (rows.n,),
                 "joint gate has no valid action-decode improvement rows")
        _require(support_delta is not None and support_delta.shape == (rows.n,),
                 "joint gate has no valid proposal-support change rows")
        _require(np.isfinite(action_delta).all() and np.isfinite(support_delta).all(),
                 "joint baseline-relative preservation rows contain non-finite values")
        preservation_boot = paired_cluster_bootstrap(
            np.stack([action_delta, support_delta], axis=1),
            rows.task_ids, rows.cluster_ids, samples=bootstrap_samples,
            confidence=confidence, seed=seed,
        )
        preservation_names = (
            "action_decode_improvement_vs_deploy",
            "proposal_support_overlap_change_vs_deploy",
        )
        preservation_thresholds = (action_threshold, support_threshold)
        baseline_passed = True
        for index, (name, threshold) in enumerate(zip(
                preservation_names, preservation_thresholds)):
            point = float(preservation_boot["point"][index])
            lower = float(preservation_boot["lower"][index])
            upper = float(preservation_boot["upper"][index])
            passed = lower >= threshold
            baseline_passed = baseline_passed and passed
            metrics[name] = {
                "point": point,
                "ci_lower": lower,
                "ci_upper": upper,
                "threshold": threshold,
                "comparison": "ci_lower greater_than_or_equal_to threshold",
                "passed": passed,
            }
            if not passed:
                failures.append(
                    f"{name}: {confidence:.1%} CI lower {lower!r} < threshold {threshold!r}"
                )
        gates["deploy_action_semantics_preservation"] = {
            "action_decode_improvement_ci_low": float(
                preservation_boot["lower"][0]
            ),
            "proposal_support_overlap_change_ci_low": float(
                preservation_boot["lower"][1]
            ),
            "thresholds": {
                "action_decode_improvement_ci_low": action_threshold,
                "proposal_support_overlap_change_ci_low": support_threshold,
            },
            "comparison": "both ci_lower greater_than_or_equal_to threshold",
            "passed": baseline_passed,
        }
        eligible = residual <= residual_max
        exhausted = ~eligible.any(axis=1)
        exhaustion_rate = float(exhausted.mean())
        preservation_passed = exhaustion_rate < exhaustion_max
        root_min = residual.min(axis=1)
        metrics["proposal_root_q_action_residual"] = {
            "threshold": residual_max,
            "eligibility_comparison": "residual_l2 less_than_or_equal_to threshold",
            "n_roots": int(residual.shape[0]),
            "candidates_per_root": int(residual.shape[1]),
            "eligible_candidates": int(eligible.sum()),
            "eligible_candidate_rate": float(eligible.mean()),
            "root_min_median": float(np.median(root_min)),
            "root_min_max": float(root_min.max()),
            "exhausted_roots": int(exhausted.sum()),
            "root_exhaustion_rate": exhaustion_rate,
            "max_root_exhaustion_rate": exhaustion_max,
            "comparison": "root_exhaustion_rate strictly_less_than maximum",
            "passed": preservation_passed,
        }
        gates["proposal_root_q_action_residual_preservation"] = {
            "q_action_residual_max": residual_max,
            "max_root_exhaustion_rate": exhaustion_max,
            "root_exhaustion_rate": exhaustion_rate,
            "comparison": "root_exhaustion_rate strictly_less_than maximum",
            "passed": preservation_passed,
        }
        if not preservation_passed:
            failures.append(
                "proposal-root q_action residual: exhaustion rate "
                f"{exhaustion_rate!r} >= maximum {exhaustion_max!r}"
            )
    _require((not failures) == all(item["passed"] for item in gates.values()),
             "per-metric and named gate verdicts disagree")
    return {
        "passed": not failures,
        "status": "PASS" if not failures else "FAIL",
        "metrics": metrics,
        "gates": gates,
        "failures": failures,
        "bootstrap": {
            **{k: v for k, v in boot.items() if k not in ("point", "lower", "upper")},
            "interval": "paired task/whole-trajectory-cluster percentile",
            "quantile_method": "numpy default linear",
            "lower_quantile": (1.0 - confidence) / 2.0,
            "upper_quantile": 1.0 - (1.0 - confidence) / 2.0,
        },
    }


def _determinism_provenance(seed: int) -> dict[str, Any]:
    warn_only = getattr(torch, "is_deterministic_algorithms_warn_only_enabled", None)
    return {
        "base_seed": int(seed),
        "global_seed_scheme": "loom.train.determinism.set_global_seed(seed, rank=0)",
        "proposal_seed_scheme": (
            "blake2b(base_seed|bank_ca_gate/proposal/<selected_ordinal>, digest_size=8)"
        ),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic_warn_only": bool(warn_only()) if callable(warn_only) else None,
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cpu_compute_dtype": "float32",
        "cuda_compute_dtype": "bfloat16",
    }


def _git_provenance() -> dict[str, Any]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            stderr=subprocess.DEVNULL, timeout=30,
        ).decode().strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
            stderr=subprocess.DEVNULL, timeout=30,
        ).decode().strip())
    except Exception:  # noqa: BLE001
        sha, dirty = "unknown", None
    source = Path(__file__).resolve()
    return {
        "git_sha": sha,
        "git_dirty": dirty,
        "gate_source": str(source),
        "gate_source_sha256": sha256_file(source),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": platform.node(),
    }


def _resolve_device(value: str) -> str:
    if value != "auto":
        if value.startswith("cuda"):
            _require(torch.cuda.is_available(), f"requested {value} but CUDA is unavailable")
        return value
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _gate_settings(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Authenticate one of the two declared action-labelled bank methods.

    This is intentionally stricter than merely finding fields needed to build a
    model.  Running a statistically valid gate on the wrong training method
    would still produce a polished but irrelevant promotion report.
    """
    _require(isinstance(cfg, Mapping), "candidate config is not a mapping")

    def get(path: str) -> Any:
        node: Any = cfg
        for part in path.split("."):
            _require(isinstance(node, Mapping) and part in node,
                     f"candidate config is missing {path}")
            node = node[part]
        return node

    def exact(path: str, wanted: Any) -> None:
        got = get(path)
        _require(got == wanted, f"{path} must be {wanted!r}, got {got!r}")

    def optional(path: str, default: Any) -> Any:
        node: Any = cfg
        for part in path.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    # Shared method identity: one action-labelled dynamics objective, with either
    # a detached q_action label or the one declared joint q_action+bank route.
    exact("run.steps", 80_000)
    exact("data.source", "libero")
    exact("data.embodiments", ["libero_franka"])
    exact("data.sampling", "uniform_window")
    exact("data.trajectory_split", "train")
    exact("data.holdout_demo_keys", ["demo_49"])
    exact("data.recurrent_burn_in", 4)
    exact("optim.update_ema", False)
    exact("losses.dyn.enabled", True)
    exact("losses.dyn.weight", 1.0)
    exact("losses.dyn.coeff_source", "q_action")
    exact("losses.dyn.negatives", "within_trajectory")
    exact("losses.dyn.min_gap", 2)
    exact("losses.dyn.neg_margin", 0.1)
    exact("losses.dyn.cosine", "per_slot")
    for name in ("proposal", "balance", "potential", "grpo"):
        exact(f"losses.{name}.enabled", False)
        exact(f"losses.{name}.weight", 0.0)
    scales = get("optim.lr_scales")
    _require(isinstance(scales, Mapping), "optim.lr_scales is not a mapping")
    for name in ("estimator", "q_delta", "ema", "proposal", "decoder", "potential"):
        _require(float(scales.get(name, float("nan"))) == 0.0,
                 f"optim.lr_scales.{name} must be 0.0")
    _require(float(scales.get("bank", float("nan"))) == 0.1,
             "optim.lr_scales.bank must be 0.1")

    train_modules = get("train_modules")
    run_name = get("run.name")
    preservation: dict[str, Any] | None = None
    if train_modules == ["bank"]:
        variant = "bank_only"
        _require(run_name in ("r0a_bank_ca", "r0a_bank_ca_n4"),
                 f"bank-only run.name is not declared: {run_name!r}")
        exact("optim.reset_state_modules", ["bank"])
        _require(float(scales.get("q_action", float("nan"))) == 0.0,
                 "bank-only optim.lr_scales.q_action must be 0.0")
        _require(optional("losses.dyn.detach_coeff", True) is True,
                 "bank-only L_dyn coefficients must remain detached")
        exact("losses.act.enabled", False)
        exact("losses.act.weight", 0.0)
        expected_neg_weight = 1.0 if run_name == "r0a_bank_ca" else 4.0
        _require(float(get("losses.dyn.neg_weight")) == expected_neg_weight,
                 f"{run_name} losses.dyn.neg_weight must be {expected_neg_weight}")
        _require("preservation" not in get("offline_gate"),
                 "bank-only recipe must not declare the joint preservation gate")
        _require(optional("optim.transition_parameter_reset", None) is None,
                 "bank-only recipe must not declare a parameter reset")
    elif train_modules == ["bank", "q_action"]:
        run_name = get("run.name")
        parameter_reset = optional("optim.transition_parameter_reset", None)
        if run_name == "r0a_bank_ca_qa":
            variant = "joint_q_action_bank"
            _require(parameter_reset is None,
                     "base joint QA recipe must not declare a parameter reset")
        elif run_name == "r0a_bank_ca_qa_omega0":
            variant = "joint_q_action_bank_identity_centered"
            _require(parameter_reset == IDENTITY_CENTERED_RESET,
                     "identity-centered QA transition reset must be exact")
        else:
            raise GateError(f"joint run.name is not declared: {run_name!r}")
        exact("optim.reset_state_modules", ["bank", "q_action"])
        _require(float(scales.get("q_action", float("nan"))) == 1.0,
                 "joint optim.lr_scales.q_action must be 1.0")
        _require(get("losses.dyn.detach_coeff") is False,
                 "joint losses.dyn.detach_coeff must be boolean false")
        exact("losses.dyn.neg_weight", 4.0)
        # q_action's semantic gradient in the base method is frozen D_e's
        # reconstruction loss. An L_dyn-only joint fit may invent a private
        # q_action/bank convention, so it is not an authenticated QA candidate.
        exact("losses.act.enabled", True)
        exact("losses.act.weight", 1.0)
        exact("losses.act.align_to", "q_a")
        exact("losses.act.decode_from", "q_action")
        exact("convergence.start_step", 49_666)
        exact("convergence.block", 2_000)
        exact("convergence.blocks", 4)
        exact("convergence.tol", 0.02)
        exact("convergence.primary", ["loss/dyn", "act/decode"])
        exact("convergence.watch", [
            "dyn/pos", "dyn/neg", "delta_op", "delta_sel/h1", "delta_sel/h2",
            "delta_sel/h3", "delta_sel/h4", "act/align", "act/c_a_spread",
            "grad_norm",
        ])
        exact("convergence.floor_checks", ["delta_sel"])
        exact("efficacy_gate.metric", "act/decode")
        exact("efficacy_gate.reference", "first_post_start_block")
        exact("efficacy_gate.comparison", "final_convergence_block")
        exact("efficacy_gate.max_relative_worsening", 0.0)
        exact("efficacy_gate.required", True)
        exact("liveness_gate.start_exclusive", 50_666)
        exact("liveness_gate.end_inclusive", 52_666)
        exact("liveness_gate.rows", 2_000)
        exact("liveness_gate.requirements.delta_op_median_strict_gt", 0.01)
        exact("liveness_gate.requirements.gnorm_bank_median_strict_gt", 1.0e-4)
        exact("liveness_gate.requirements.gnorm_q_action_median_strict_gt", 1.0e-4)
        exact("liveness_gate.requirements.skipped_rate_strict_lt", 0.01)
        exact("liveness_gate.requirements.unexpected_module_gradients", False)
        exact("liveness_gate.requirements.nonfinite", False)
        exact("liveness_gate.required", True)
        exact("liveness_gate", {
            "start_exclusive": 50_666,
            "end_inclusive": 52_666,
            "rows": 2_000,
            "requirements": {
                "delta_op_median_strict_gt": 0.01,
                "gnorm_bank_median_strict_gt": 1.0e-4,
                "gnorm_q_action_median_strict_gt": 1.0e-4,
                "skipped_rate_strict_lt": 0.01,
                "unexpected_module_gradients": False,
                "nonfinite": False,
            },
            "required": True,
        })
        raw_preservation = get("offline_gate.preservation")
        _require(isinstance(raw_preservation, Mapping),
                 "joint offline_gate.preservation is not a mapping")
        missing = sorted(set(PRESERVATION_KEYS) - set(raw_preservation))
        unknown = sorted(set(raw_preservation) - set(PRESERVATION_KEYS))
        _require(not missing and not unknown,
                 f"offline_gate.preservation missing={missing} unknown={unknown}")
        exact(
            "offline_gate.preservation.reference_checkpoint_sha256",
            "15f286c268caa5327d5aa3abf1f67ebd0555c426a509fef22cb7f537bf6ab4e1",
        )
        exact("offline_gate.preservation.reference_config_hash", "a199324a6205bb6d")
        exact("offline_gate.preservation.reference_global_step", 49_666)
        exact("offline_gate.preservation.action_decode_improvement_ci_low", 0.0)
        exact(
            "offline_gate.preservation.proposal_support_overlap_change_ci_low", -0.05,
        )
        exact("offline_gate.preservation.q_action_residual_max", 0.5)
        exact("offline_gate.preservation.max_root_exhaustion_rate", 0.01)
        preservation = copy.deepcopy(dict(raw_preservation))
    else:
        raise GateError(
            "train_modules must be exactly ['bank'] or ['bank', 'q_action'], got "
            f"{train_modules!r}"
        )

    exact("offline_gate.script", "scripts/bank_ca_gate.py")
    exact("offline_gate.required", True)
    exact("offline_gate.direct_e2e", False)
    exact("offline_gate.confidence", DEFAULT_CONFIDENCE)
    exact("offline_gate.bootstrap_samples", DEFAULT_BOOTSTRAP_SAMPLES)
    exact("offline_gate.seed", 0)
    exact("offline_gate.windows", PINNED_WINDOWS)
    exact("offline_gate.candidates", PINNED_CANDIDATES)
    requirements = get("offline_gate.requirements")
    _require(isinstance(requirements, Mapping), "offline_gate.requirements is not a mapping")
    missing = sorted(set(REQUIREMENT_KEYS) - set(requirements))
    unknown = sorted(set(requirements) - set(REQUIREMENT_KEYS))
    _require(not missing and not unknown,
             f"offline_gate.requirements missing={missing} unknown={unknown}")
    for key in REQUIREMENT_KEYS:
        value = requirements[key]
        _require(isinstance(value, (int, float)) and math.isfinite(float(value)),
                 f"offline_gate.requirements.{key} must be finite")
        _require(float(value) == 0.0,
                 f"offline_gate.requirements.{key} must be exactly 0.0")

    return {
        "windows": PINNED_WINDOWS,
        "candidates": PINNED_CANDIDATES,
        "bootstrap_samples": DEFAULT_BOOTSTRAP_SAMPLES,
        "confidence": DEFAULT_CONFIDENCE,
        "seed": 0,
        "requirements": {key: 0.0 for key in REQUIREMENT_KEYS},
        "method_variant": variant,
        "transition_parameter_reset": copy.deepcopy(
            optional("optim.transition_parameter_reset", None)
        ),
        "preservation": preservation,
        "cosine": "per_slot",
    }


def _runtime_recipe(cfg: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    pinned = _gate_settings(cfg)
    windows = int(args.windows if args.windows is not None else pinned["windows"])
    candidates = int(args.candidates if args.candidates is not None else pinned["candidates"])
    bootstrap_samples = int(
        args.bootstrap_samples
        if args.bootstrap_samples is not None else pinned["bootstrap_samples"]
    )
    confidence = float(args.confidence if args.confidence is not None else pinned["confidence"])
    seed = int(args.seed if args.seed is not None else pinned["seed"])
    _require(windows == pinned["windows"],
             f"runtime windows {windows} != authenticated gate {pinned['windows']}")
    _require(candidates == pinned["candidates"],
             f"runtime candidates {candidates} != authenticated gate {pinned['candidates']}")
    _require(bootstrap_samples == pinned["bootstrap_samples"],
             "runtime bootstrap_samples differs from authenticated gate")
    _require(confidence == pinned["confidence"],
             "runtime confidence differs from authenticated gate")
    _require(seed == pinned["seed"], "runtime seed differs from authenticated gate")
    _require(int(args.batch_size) > 0, "runtime batch_size must be positive")
    _require(int(args.workers) >= 0, "runtime workers must be nonnegative")
    return {
        "windows": windows,
        "candidates": candidates,
        "batch_size": int(args.batch_size),
        "bootstrap_samples": bootstrap_samples,
        "confidence": confidence,
        "seed": seed,
        "requirements": pinned["requirements"],
        "method_variant": pinned["method_variant"],
        "transition_parameter_reset": pinned["transition_parameter_reset"],
        "preservation": pinned["preservation"],
        "cosine": "per_slot",
        "data_sampling": "uniform_task",
        "trajectory_split": "gate",
        "selected_order": "first N seeded uniform_task rows",
        "delta_sel_pairing": "global cyclic previous selected row",
        "proposal_rng": "blake2b(base_seed, fixed selected ordinal), one stream per root",
        "decoder_rng": (
            "blake2b(base_seed, fixed selected ordinal), one common noise segment "
            "expanded across proposal roots"
            if pinned["preservation"] is not None else None
        ),
        "semantic_cli_overrides": {
            "windows": args.windows is not None,
            "candidates": args.candidates is not None,
            "bootstrap_samples": args.bootstrap_samples is not None,
            "confidence": args.confidence is not None,
            "seed": args.seed is not None,
        },
        "runtime": {
            "batch_size": int(args.batch_size),
            "workers": int(args.workers),
            "device_request": str(args.device),
            "cache_root_override": (
                str(Path(args.cache_root).resolve()) if args.cache_root else None
            ),
        },
    }


def _close_loader_caches(loader: Any) -> None:
    seen: set[int] = set()
    for dataset in getattr(loader, "datasets", {}).values():
        cache = getattr(dataset, "cache", None)
        if cache is None or id(cache) in seen:
            continue
        seen.add(id(cache))
        close = getattr(cache, "close", None)
        if callable(close):
            close()


def execute_gate(
    args: argparse.Namespace,
    *,
    loader_factory: Callable[..., Any] | None = None,
    audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if audit is None:
        audit = {}
    device = _resolve_device(args.device)
    audit["device"] = device
    candidate = load_candidate(
        args.checkpoint, config_path=args.config, device=device, audit=audit,
    )
    audit["candidate"] = copy.deepcopy(candidate.provenance)
    recipe = _runtime_recipe(candidate.config, args)
    recipe["runtime"]["resolved_device"] = device
    recipe["determinism"] = copy.deepcopy(candidate.provenance["determinism"])
    reference: ReferenceHeads | None = None
    if recipe["preservation"] is not None:
        _require(bool(args.reference_checkpoint),
                 "joint QA gate requires --reference-checkpoint")
        reference = load_reference_heads(
            args.reference_checkpoint, candidate, recipe["preservation"],
            device=device,
        )
        recipe["runtime"]["reference_checkpoint"] = str(
            Path(args.reference_checkpoint).resolve()
        )
        audit["reference"] = copy.deepcopy(reference.provenance)
    else:
        _require(args.reference_checkpoint is None,
                 "bank-only gate does not accept --reference-checkpoint")
    audit["recipe"] = copy.deepcopy(recipe)

    # The gate deliberately overrides only gate-local data traversal.  Model
    # construction used the authenticated checkpoint config above.
    data_cfg = copy.deepcopy(candidate.config)
    data_cfg.setdefault("data", {})
    data_cfg["data"]["batch_per_gpu"] = recipe["batch_size"]
    data_cfg["data"]["sampling"] = "uniform_task"
    data_cfg["data"]["trajectory_split"] = "gate"
    data_cfg["data"]["num_workers"] = int(args.workers)

    if loader_factory is None:
        try:
            from loom.data.loader import build_gate_loader as loader_factory
        except Exception as exc:  # noqa: BLE001
            raise GateError(
                "loom.data.loader.build_gate_loader is unavailable; refusing to "
                "fall back to the training/all-data loader"
            ) from exc
    try:
        loader = loader_factory(
            data_cfg, rank=0, world=1, seed=recipe["seed"], device="cpu",
            cache_root=args.cache_root,
        )
    except Exception as exc:  # noqa: BLE001
        raise GateError(f"cannot build strict gate-only held-out loader: {exc}") from exc

    try:
        rows, source = collect_metric_rows(
            candidate.model, loader,
            windows=recipe["windows"], batch_size=recipe["batch_size"],
            n_candidates=recipe["candidates"], seed=recipe["seed"],
            device=device, cosine=recipe["cosine"],
            measure_residual=recipe["preservation"] is not None,
            reference_q_action=(None if reference is None else reference.q_action),
            audit=audit,
        )
        decision = summarize_gate(
            rows,
            requirements=recipe["requirements"],
            bootstrap_samples=recipe["bootstrap_samples"],
            confidence=recipe["confidence"], seed=recipe["seed"],
            preservation=recipe["preservation"],
        )
        return {
            "format_version": FORMAT_VERSION,
            **decision,
            "overall_verdict": decision["status"],
            "candidate": candidate.provenance,
            "reference": None if reference is None else reference.provenance,
            "data": source,
            "recipe": recipe,
            "source_provenance": _git_provenance(),
            "direct_e2e_run": False,
        }
    finally:
        _close_loader_caches(loader)


def atomic_write_json(path: str | os.PathLike, value: Mapping[str, Any]) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(
            json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False)
            + "\n"
        )
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="consolidated candidate checkpoint")
    ap.add_argument(
        "--reference-checkpoint",
        help="authenticated deploy seed; required only for joint q_action+bank QA",
    )
    ap.add_argument("--config", help="optional resolved config to cross-check/fallback")
    ap.add_argument("--out", required=True, help="atomic JSON report path")
    ap.add_argument("--cache-root", help="feature-cache override passed to gate loader")
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--windows", type=int, default=None)
    ap.add_argument("--candidates", type=int, default=None)
    ap.add_argument("--bootstrap-samples", type=int, default=None)
    ap.add_argument("--confidence", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    audit: dict[str, Any] = {}
    try:
        report = execute_gate(args, audit=audit)
        code = 0 if report["passed"] else 1
    except Exception as exc:  # noqa: BLE001 - the gate must persist every failure
        report = {
            "format_version": FORMAT_VERSION,
            "passed": False,
            "status": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "requested": vars(args),
            **audit,
            "source_provenance": _git_provenance(),
            "direct_e2e_run": False,
        }
        code = 2
    atomic_write_json(args.out, report)
    print(json.dumps({
        "status": report["status"],
        "passed": report["passed"],
        "out": str(Path(args.out).resolve()),
        "failures": report.get("failures", []),
        "error": report.get("error"),
    }, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
