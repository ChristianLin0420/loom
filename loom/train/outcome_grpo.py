"""Fail-closed proposal-only training on outcome-recovery sidecars.

The collector in :mod:`loom.eval.outcome_recovery` records complete groups of
eight trajectories for one LIBERO reset.  Arm 0 is the deployed deterministic
policy and arms 1--7 are exact ordered Plackett--Luce samples.  This module is
the deliberately separate training half:

* terminal rewards are normalised inside each complete eight-arm group;
* arm 0 is a control/baseline only and never enters an importance ratio;
* sampled atoms are scored in their *stored* order, never by recovering a
  canonical order from their coefficients;
* the clipped objective is averaged over replans inside each trajectory and
  then over the seven sampled trajectories;
* the only trainable module is ``proposal``.  The existing sparse-CE target
  path supplies an expert anchor from frozen estimator/q_action modules and
  labelled demonstration windows.

Every collection byte is authenticated and deeply schema-validated before the
first update.  The final checkpoint is emitted only after the locked heldout
convergence and trust gates pass.  It is a full eval-compatible descendant,
but every non-proposal model tensor is required to survive save/reload
byte-for-byte unchanged.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import inspect
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn

import contracts as C
from loom.eval import outcome_recovery as recovery
from loom.eval.policy import submodule_state
from loom.heads.proposal import (
    Proposal,
    argmax_coeff,
    pl_log_prob,
    weights_from_logits,
)
from loom.losses.proposal_bc import proposal_sparse_ce_loss
from loom.train import wandb_util
from loom.train.atomic import atomic_write_text, fsync_dir, read_pointer
from loom.train.determinism import enable_determinism, set_global_seed, set_step_seed
from loom.train.preempt import PreemptGuard, write_heartbeat
from loom.train.schedule import CosineWithWarmup, build_optimizer

__all__ = [
    "FORMAT_VERSION",
    "START_STEP",
    "STOP_STEP",
    "N_FOLDS",
    "UPDATES_PER_FOLD",
    "SCHEDULE_STEPS",
    "CLIP_EPS",
    "MAX_CLIP_FRACTION",
    "MIN_ESS_FRACTION",
    "MAX_COEFF_DRIFT_P95",
    "MIN_LIVE_OPS",
    "CONVERGENCE_SNAPSHOT_STEPS",
    "CONVERGENCE_BOOTSTRAP_SAMPLES",
    "CONVERGENCE_PLATEAU_MARGIN",
    "MAX_APPROX_KL",
    "OutcomeGRPOError",
    "ExpertAnchorUnavailable",
    "TrustGateError",
    "ValidatedRecoveryCollection",
    "DeterministicOutcomeSampler",
    "ExpertAnchor",
    "normalised_group_advantages",
    "clipped_grpo_objective",
    "group_grpo_loss",
    "proposal_switch_balance",
    "stored_order_logprob",
    "evaluate_validation_surrogate",
    "task_stratified_paired_bootstrap",
    "evaluate_convergence_gate",
    "evaluate_metric_convergence",
    "model_state_digest",
    "frozen_model_digest",
    "evaluate_trust_gates",
    "write_descendant_checkpoint",
    "train_outcome_grpo",
    "build_parser",
    "main",
]


FORMAT_VERSION = 1
TRAINER_KIND = "loom_outcome_grpo_proposal_descendant"
START_STEP = recovery.SEED_GLOBAL_STEP
N_FOLDS = len(recovery.TRAIN_FOLDS)
UPDATES_PER_FOLD = 800
STOP_STEP = START_STEP + N_FOLDS * UPDATES_PER_FOLD
SCHEDULE_STEPS = 80_000
CLIP_EPS = 0.20
MAX_CLIP_FRACTION = 0.20
MIN_ESS_FRACTION = 0.80
MAX_COEFF_DRIFT_P95 = 0.05
MIN_LIVE_OPS = 16

# The terminal objective is judged only from authenticated validation groups
# and predeclared, durable proposal snapshots.  These are absolute optimizer
# steps, not wall-clock or epoch-derived choices.
CONVERGENCE_SNAPSHOT_STEPS = (53_666, 53_866, 54_066, 54_266, 54_466)
CONVERGENCE_BOOTSTRAP_SAMPLES = 2_000
CONVERGENCE_BOOTSTRAP_SEED = 0
CONVERGENCE_CONFIDENCE = 0.95
CONVERGENCE_PLATEAU_MARGIN = 0.01
MAX_APPROX_KL = 0.01
CONVERGENCE_CE_BLOCK_SIZE = 200
CONVERGENCE_CE_BLOCKS = 4
MAX_CE_BLOCK_MEDIAN_RELATIVE_RANGE = 0.02
SPARSE_CE_UNIFORM_FLOOR = math.log(C.M)
TERMINAL_PARALLELISM = "one_snapshot_or_trust_per_rank"
EXPECTED_ACCEPTED_UPDATES = STOP_STEP - START_STEP
EXPECTED_VALIDATION_GROUPS = 400
EXPECTED_VALIDATION_TASKS = 40
EXPECTED_WORLD_SIZE = 8
EXPECTED_BATCH_PER_GPU = 8
EXPECTED_CONTEXTS_PER_ARM = 2
EXPECTED_INITIAL_RATIO_ATOMS_PER_RANK = (
    (recovery.GROUP_SIZE - 1) * EXPECTED_CONTEXTS_PER_ARM
)
MIN_TRAIN_INFORMATIVE_GROUPS = 100
MIN_VALIDATION_INFORMATIVE_GROUPS = 200
LOG_EVERY = 20
CKPT_EVERY = 200
AUTH_CHUNK_REPLANS = 32
PROPOSAL_SCORING_BATCH_SIZE = 1
PROPOSAL_SCORING_DTYPE = "float32"
PROPOSAL_SCORING_AUTOCAST = False
CUDA_MATMUL_TF32 = False
CUDNN_TF32 = False
FLOAT32_MATMUL_PRECISION = "highest"
PROPOSAL_SCORING_MODULE_MODE = "eval"
STRICT_OUTCOME_DETERMINISM = {
    "deterministic_algorithms": True,
    "warn_only": False,
}
# Collector and trainer execute the same B=1 fp32 proposal and stored-order PL
# reduction. Any nonzero replay error identifies a scoring-path defect and
# fails before the first update.
BEHAVIOUR_IDENTITY_MAX_ABS_LOGRATIO = 0.0
BEHAVIOUR_IDENTITY_MAX_COEFF_ERROR = 0.0
INITIAL_RATIO_MIN_ESS_FRACTION = 1.0

# Keep the parent AdamW geometry while resetting every proposal moment.  The
# proposal retains the deployed-stage 0.05 multiplier on the existing absolute
# 80k cosine; a requeue never restarts or shortens that schedule.
BASE_LEARNING_RATE = 3e-4
PROPOSAL_LR_SCALE = 0.05
WARMUP_STEPS = 2_000
MIN_LR_RATIO = 0.05
ADAMW_BETAS = (0.9, 0.95)
ADAMW_WEIGHT_DECAY = 0.05
ADAMW_EPS = 1e-8
GRAD_CLIP = 1.0
SWITCH_BALANCE_WEIGHT = 1e-2
EXPERT_GATE_BATCHES = 16
MAX_TOPK_OVERLAP_DECLINE = 0.05
TRAIN_SEED = 0

# Six heldout proposal snapshots (the seed plus five predeclared checkpoints)
# and the final trust evaluation are independent, immutable jobs.  Assigning
# one to each rank preserves exact row-wise B=1 proposal execution while
# avoiding a >1.5M-forward serial rank-0 tail.  Rank 7 is deliberately idle so
# the fixed eight-rank training world does not alter a statistical unit.
_TERMINAL_EVAL_TASKS = (
    *(("snapshot", int(step))
      for step in (START_STEP, *CONVERGENCE_SNAPSHOT_STEPS)),
    ("trust", STOP_STEP),
)

EXPECTED_SEED_CHECKPOINT = "runs/r0a_deploy_s1_eval/ckpt_000049666.pt"
EXPECTED_FOLDS = tuple(
    {"split": f"train{index}",
     "path": f"runs/outcome_recovery_s49666_train{index}"}
    for index in range(N_FOLDS)
)
EXPECTED_VALIDATION = {
    "split": "validation", "path": "runs/outcome_recovery_s49666_validation",
}
EXPECTED_ANCHOR_MANIFEST = {
    "digest": "sha256:f61c453864dc8a84e274a65e834e037a83ef8407ed4e9635f84c78d814fe2e7e",
    "n_tasks": 40,
    "n_trajectories": 1960,
    "n_windows": 47271,
}

# Every repository file whose implementation is executed by this standalone
# path.  The identity is checkpoint-bound, so a requeue cannot silently cross
# a behavior change even when the resolved experiment config is unchanged.
_TRAINER_SOURCE_FILES = (
    "contracts.py",
    "stubs.py",
    "loom/train/outcome_grpo.py",
    "scripts/train_outcome_grpo.py",
    "scripts/outcome_grpo.sbatch",
    "scripts/env.sh",
    "loom/eval/outcome_recovery.py",
    "loom/eval/policy.py",
    "loom/heads/proposal.py",
    "loom/heads/decoder.py",
    "loom/losses/proposal_bc.py",
    "loom/losses/dyn.py",
    "loom/model/estimator.py",
    "loom/heads/q_action.py",
    "loom/heads/q_delta.py",
    "loom/train/loop.py",
    "loom/train/ckpt.py",
    "loom/train/fsdp.py",
    "loom/train/determinism.py",
    "loom/train/schedule.py",
    "loom/train/preempt.py",
    "loom/train/atomic.py",
    "loom/train/wandb_util.py",
    "loom/data/loader.py",
    "loom/data/cache.py",
    "loom/data/canonical.py",
    "loom/data/adapters/libero.py",
    "loom/data/tower.py",
)

# Match the aggregate numerical-identity gate above.  Relative tolerances can
# silently widen with log-probability magnitude, so only the measured absolute
# fp32 reduction boundary is allowed.
BEHAVIOUR_LOGPROB_ATOL = BEHAVIOUR_IDENTITY_MAX_ABS_LOGRATIO
BEHAVIOUR_LOGPROB_RTOL = 0.0
BEHAVIOUR_COEFF_ATOL = BEHAVIOUR_IDENTITY_MAX_COEFF_ERROR
BEHAVIOUR_COEFF_RTOL = 0.0

_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST_FIELDS = {
    "format_version", "kind", "identity", "identity_digest", "split",
    "started_utc", "updated_utc", "summary", "groups",
}
_RECEIPT_FIELDS = {
    "group_id", "sidecar", "sha256", "size", "n_arms",
    "n_replans_by_arm", "terminal_rewards", "worker",
}


class OutcomeGRPOError(RuntimeError):
    """An authenticated input or training invariant failed."""


class ExpertAnchorUnavailable(OutcomeGRPOError):
    """The existing q_action sparse-CE target path cannot be used exactly."""


class TrustGateError(OutcomeGRPOError):
    """The locked post-training trust envelope rejected the proposal."""

    def __init__(self, report: Mapping[str, Any]):
        self.report = dict(report)
        failed = [name for name, row in self.report.get("checks", {}).items()
                  if not bool(row.get("pass"))]
        super().__init__("outcome-GRPO trust gate failed: " + ", ".join(failed))


class _PreemptRequested(OutcomeGRPOError):
    """Internal, non-failure exit before an optimizer update is attempted."""


@dataclass
class _RunDirectoryLock:
    """Process-lifetime advisory lock preventing two writers in one run."""

    path: Path
    fd: int

    def close(self) -> None:
        if self.fd < 0:
            return
        try:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
        finally:
            os.close(self.fd)
            self.fd = -1

    def __del__(self) -> None:
        try:
            self.close()
        except OSError:
            pass


def _acquire_run_directory_lock(run_dir: Path) -> _RunDirectoryLock:
    """Acquire the sole-writer lock, or fail before authentication/training."""

    path = run_dir / ".outcome_grpo.lock"
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(fd)
        raise OutcomeGRPOError(
            f"another outcome-GRPO writer holds the run directory lock: {path}"
        ) from exc
    owner = {
        "hostname": os.environ.get("SLURMD_NODENAME") or os.uname().nodename,
        "pid": os.getpid(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "started_utc": _utc(),
    }
    encoded = (json.dumps(owner, sort_keys=True, allow_nan=False) + "\n").encode()
    os.ftruncate(fd, 0)
    os.write(fd, encoded)
    os.fsync(fd)
    return _RunDirectoryLock(path=path, fd=fd)


def _require(condition: Any, message: str) -> None:
    if not bool(condition):
        raise OutcomeGRPOError(message)


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _valid_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(c in "0123456789abcdef" for c in text)


def _config_hash(cfg: Mapping[str, Any]) -> str:
    experiment = {k: v for k, v in cfg.items() if k != "link"}
    return hashlib.blake2b(
        json.dumps(experiment, sort_keys=True, default=str).encode(),
        digest_size=8,
    ).hexdigest()


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _tensor_bytes(value: Tensor) -> tuple[Tensor, memoryview]:
    tensor = value.detach().cpu().contiguous()
    # A zero-dimensional tensor cannot change element size through ``view``
    # until it has first been made one-dimensional.
    raw = tensor.reshape(-1).view(torch.uint8).numpy()
    return tensor, memoryview(raw)


def model_state_digest(
    state: Mapping[str, Tensor],
    *,
    include: Any | None = None,
) -> dict[str, Any]:
    """Hash tensor names, metadata, and exact storage bytes in a model state."""
    h = hashlib.sha256()
    n_tensors = 0
    n_bytes = 0
    for name in sorted(state):
        if include is not None and not include(name):
            continue
        value = state[name]
        _require(isinstance(value, Tensor), f"model state {name!r} is not a tensor")
        tensor, raw = _tensor_bytes(value)
        header = json.dumps(
            {"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        h.update(header + b"\0")
        h.update(raw)
        h.update(b"\0")
        n_tensors += 1
        n_bytes += int(tensor.numel() * tensor.element_size())
    _require(n_tensors > 0, "model-state digest selected zero tensors")
    return {"sha256": h.hexdigest(), "n_tensors": n_tensors, "n_bytes": n_bytes}


def frozen_model_digest(state: Mapping[str, Tensor]) -> dict[str, Any]:
    """Exact digest of every checkpoint model tensor except ``proposal.*``."""
    return model_state_digest(
        state, include=lambda name: not str(name).startswith("proposal."),
    )


def proposal_model_digest(state: Mapping[str, Tensor]) -> dict[str, Any]:
    return model_state_digest(
        state, include=lambda name: str(name).startswith("proposal."),
    )


def proposal_module_digest(state: Mapping[str, Tensor]) -> dict[str, Any]:
    """Hash a standalone Proposal under its canonical checkpoint key names."""
    _require(
        state and all(not str(name).startswith("proposal.") for name in state),
        "standalone proposal state unexpectedly contains checkpoint prefixes",
    )
    return model_state_digest({f"proposal.{name}": value for name, value in state.items()})


def _all_finite_tensors(values: Iterable[Tensor]) -> bool:
    return all(bool(torch.isfinite(value.detach()).all()) for value in values)


def normalised_group_advantages(rewards: Sequence[float] | Tensor) -> Tensor:
    """Population-normalise exactly one complete eight-trajectory reward group.

    A constant-reward group carries no policy information and therefore maps to
    eight exact zeros instead of being divided by an arbitrary epsilon.
    """
    value = torch.as_tensor(rewards, dtype=torch.float32).reshape(-1)
    if value.numel() != recovery.GROUP_SIZE:
        raise ValueError(
            f"one recovery group has {recovery.GROUP_SIZE} rewards, got {value.numel()}"
        )
    if not bool(torch.isfinite(value).all()):
        raise ValueError("group rewards contain nan/inf")
    centred = value - value.mean()
    variance = centred.square().mean()
    if float(variance) == 0.0:
        return torch.zeros_like(value)
    return centred / variance.sqrt()


def clipped_grpo_objective(
    current_logprob: Tensor,
    old_logprob: Tensor,
    advantage: float | Tensor,
    *,
    clip_eps: float = CLIP_EPS,
) -> tuple[Tensor, Tensor, Tensor]:
    """Per-replan PPO/GRPO objective, exact ratio, and clipping indicator."""
    if current_logprob.shape != old_logprob.shape:
        raise ValueError(
            f"current/old logprob shape mismatch: {tuple(current_logprob.shape)} "
            f"vs {tuple(old_logprob.shape)}"
        )
    if not 0.0 < float(clip_eps) < 1.0:
        raise ValueError(f"clip_eps must be in (0,1), got {clip_eps}")
    ratio = _recovery_importance_ratio_fp32(current_logprob, old_logprob)
    # Clipping/objective arithmetic remains in the same fp32, autocast-disabled
    # recovery-policy boundary as the ratio itself.
    with torch.autocast(device_type=current_logprob.device.type, enabled=False):
        adv = torch.as_tensor(advantage, device=ratio.device, dtype=torch.float32)
        clipped_ratio = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
        objective = torch.minimum(ratio * adv, clipped_ratio * adv)
        clipped = (ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps)
    return objective, ratio, clipped


def _recovery_importance_ratio_fp32(
    current_logprob: Tensor,
    old_logprob: Tensor,
) -> Tensor:
    """Compute ``exp(current-old)`` in fp32 outside every caller autocast."""
    if current_logprob.shape != old_logprob.shape:
        raise ValueError(
            f"current/old logprob shape mismatch: {tuple(current_logprob.shape)} "
            f"vs {tuple(old_logprob.shape)}"
        )
    with torch.autocast(device_type=current_logprob.device.type, enabled=False):
        current = current_logprob.float()
        old = old_logprob.to(device=current.device, dtype=torch.float32)
        return torch.exp(current - old)


def group_grpo_loss(
    current_logprobs: Sequence[Tensor],
    old_logprobs: Sequence[Tensor],
    rewards: Sequence[float] | Tensor,
    *,
    clip_eps: float = CLIP_EPS,
) -> tuple[Tensor, Tensor]:
    """Exact ``mean(trajectory_mean(replan_objective))`` over arms 1--7.

    Arm 0 is deliberately absent from both log-probability arguments.  Its
    reward is still element 0 of ``rewards`` and participates in the group
    normalisation.
    """
    expected = recovery.GROUP_SIZE - 1
    if len(current_logprobs) != expected or len(old_logprobs) != expected:
        raise ValueError(
            f"GRPO expects {expected} sampled trajectories, got "
            f"{len(current_logprobs)}/{len(old_logprobs)}"
        )
    advantages = normalised_group_advantages(rewards)
    trajectory_terms: list[Tensor] = []
    ratios: list[Tensor] = []
    for arm, (current, old) in enumerate(
        zip(current_logprobs, old_logprobs, strict=True), start=1,
    ):
        if current.numel() == 0:
            raise ValueError(f"sampled arm {arm} has zero replans")
        objective, ratio, _ = clipped_grpo_objective(
            current, old, advantages[arm], clip_eps=clip_eps,
        )
        trajectory_terms.append(objective.mean())
        ratios.append(ratio.reshape(-1))
    return -torch.stack(trajectory_terms).mean(), torch.cat(ratios)


def proposal_switch_balance(logits: Tensor, *, topk: int = C.TOPK) -> Tensor:
    """Switch load balance on the proposal's dense router probabilities.

    The hard top-k routing frequency is detached and the dense softmax carries
    the gradient.  This is the same definition as the existing LOOM balance
    term, specialised here so the isolated trainer does not call into the
    monolithic training loop.
    """
    if logits.ndim != 2 or logits.shape[0] == 0:
        raise ValueError(f"proposal logits must be a non-empty (N,M), got {tuple(logits.shape)}")
    m = int(logits.shape[-1])
    k = min(int(topk), m)
    if k <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    hard = logits.detach().float().topk(k, dim=-1).indices
    f = torch.zeros_like(logits, dtype=torch.float32).scatter_(1, hard, 1.0)
    f = f.sum(0) / float(logits.shape[0] * k)
    dense = torch.softmax(logits.float(), dim=-1).mean(0)
    return float(m) * (f * dense).sum()


def _batched_lang(lang: Tensor, n: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    value = lang.to(device=device, dtype=dtype, non_blocking=True)
    if value.ndim == 2:
        value = value.unsqueeze(0).expand(n, -1, -1)
    elif value.ndim == 3 and value.shape[0] == 1:
        value = value.expand(n, -1, -1)
    elif value.ndim != 3 or value.shape[0] != n:
        raise OutcomeGRPOError(
            f"language tensor cannot batch to {n}: shape={tuple(value.shape)}"
        )
    return value


def _proposal_scoring_geometry(device: torch.device) -> dict[str, Any]:
    """Machine-readable execution contract for behaviour-policy scoring."""
    return {
        "batch_size": PROPOSAL_SCORING_BATCH_SIZE,
        "dtype": PROPOSAL_SCORING_DTYPE,
        "autocast": PROPOSAL_SCORING_AUTOCAST,
        "cuda_matmul_tf32": CUDA_MATMUL_TF32,
        "cudnn_tf32": CUDNN_TF32,
        "float32_matmul_precision": FLOAT32_MATMUL_PRECISION,
        "device_type": device.type,
        "module_mode": PROPOSAL_SCORING_MODULE_MODE,
        "stored_order": True,
    }


def _strict_outcome_determinism_state() -> dict[str, bool]:
    """Return the two PyTorch flags that define strict algorithm selection."""
    warn_only = getattr(
        torch, "is_deterministic_algorithms_warn_only_enabled", None,
    )
    _require(callable(warn_only),
             "PyTorch cannot report deterministic warn-only state")
    return {
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "warn_only": bool(warn_only()),
    }


def _configure_strict_outcome_determinism() -> dict[str, bool]:
    """Make the locked deterministic recipe strict before any proposal graph.

    The shared training helper intentionally uses ``warn_only=True`` for older
    repository paths.  Outcome GRPO is a separately authenticated formal path:
    its Flash and Memory-Efficient SDPA backwards must use PyTorch's supported
    deterministic implementations rather than merely emitting a warning.
    """
    torch.use_deterministic_algorithms(True, warn_only=False)
    state = _strict_outcome_determinism_state()
    _require(
        state == STRICT_OUTCOME_DETERMINISM,
        f"outcome trainer did not enter strict deterministic mode: {state}",
    )
    return state


def _configure_exact_proposal_scoring(device: torch.device) -> dict[str, Any]:
    """Disable reduced-precision CUDA paths used nowhere in collection.

    The recovery collector called the fp32 proposal with one replan at a time.
    A100 SDPA is both batch-shape- and TF32-sensitive, so determinism flags are
    insufficient: exact importance ratios require this explicit geometry.
    """
    _configure_strict_outcome_determinism()
    torch.set_float32_matmul_precision(FLOAT32_MATMUL_PRECISION)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = CUDA_MATMUL_TF32
        torch.backends.cudnn.allow_tf32 = CUDNN_TF32
        _require(
            bool(torch.backends.cuda.matmul.allow_tf32) is CUDA_MATMUL_TF32,
            "could not disable CUDA matmul TF32 for proposal scoring",
        )
        _require(
            bool(torch.backends.cudnn.allow_tf32) is CUDNN_TF32,
            "could not disable cuDNN TF32 for proposal scoring",
        )
        _require(
            torch.get_float32_matmul_precision() == FLOAT32_MATMUL_PRECISION,
            "could not select highest float32 matmul precision for proposal scoring",
        )
    return _proposal_scoring_geometry(device)


def _require_exact_proposal_scoring_environment(
    proposal: nn.Module,
    device: torch.device,
) -> None:
    floating = [
        (name, value.dtype)
        for name, value in list(proposal.named_parameters())
        + list(proposal.named_buffers())
        if value.is_floating_point() and value.dtype != torch.float32
    ]
    _require(
        not floating,
        f"proposal scoring requires fp32 parameters/buffers, got {floating[:8]}",
    )
    _require(
        not proposal.training,
        "recovery-policy proposal scoring requires collector-matching eval mode",
    )
    if device.type == "cuda":
        _require(
            not bool(torch.backends.cuda.matmul.allow_tf32),
            "CUDA matmul TF32 was enabled during exact proposal scoring",
        )
        _require(
            not bool(torch.backends.cudnn.allow_tf32),
            "cuDNN TF32 was enabled during exact proposal scoring",
        )
        _require(
            torch.get_float32_matmul_precision() == FLOAT32_MATMUL_PRECISION,
            "float32 matmul precision changed during exact proposal scoring",
        )


def _rowwise_proposal_logits_fp32(
    proposal: nn.Module,
    z: Tensor,
    lang: Tensor,
) -> Tensor:
    """Run proposal logits exactly as collection: stored row order, B=1, fp32."""
    if z.ndim != 3 or z.shape[0] <= 0:
        raise ValueError(f"proposal belief must be non-empty (N,S,D), got {tuple(z.shape)}")
    device = z.device
    _require_exact_proposal_scoring_environment(proposal, device)
    z32 = z.to(device=device, dtype=torch.float32)
    lang32 = _batched_lang(lang, int(z.shape[0]), device, torch.float32)
    rows: list[Tensor] = []
    # This nested context overrides both the trainer's anchor autocast and any
    # caller/test autocast.  TF32 is separately disabled and checked above.
    with torch.autocast(device_type=device.type, enabled=False):
        for row in range(int(z.shape[0])):
            logits = proposal.logits(
                z32[row:row + PROPOSAL_SCORING_BATCH_SIZE],
                lang32[row:row + PROPOSAL_SCORING_BATCH_SIZE],
            )
            _require(
                logits.ndim == 2
                and logits.shape[0] == PROPOSAL_SCORING_BATCH_SIZE
                and logits.shape[1] > 0,
                f"row-wise proposal logits have shape {tuple(logits.shape)}",
            )
            _require(
                logits.dtype == torch.float32,
                f"row-wise proposal logits are {logits.dtype}, expected float32",
            )
            _require(bool(torch.isfinite(logits).all()),
                     "row-wise proposal logits contain nan/inf")
            rows.append(logits)
    return torch.cat(rows, dim=0)


def stored_order_logprob(
    proposal: nn.Module,
    z: Tensor,
    lang: Tensor,
    ordered_support: Tensor,
) -> tuple[Tensor, Tensor]:
    """Score stored PL atoms with exact collection execution geometry."""
    if z.ndim != 3 or ordered_support.ndim != 2 or z.shape[0] != ordered_support.shape[0]:
        raise ValueError(
            f"invalid stored-order batch z={tuple(z.shape)} "
            f"order={tuple(ordered_support.shape)}"
        )
    logits = _rowwise_proposal_logits_fp32(proposal, z, lang)
    if logits.shape[:-1] != ordered_support.shape[:-1]:
        raise ValueError(
            f"proposal/order batch mismatch: {tuple(logits.shape)} vs "
            f"{tuple(ordered_support.shape)}"
        )
    # Collection accumulated each PL atom while its logits still had shape
    # (1,M).  Keep that B=1 geometry here too: CUDA logsumexp can differ by a
    # few fp32 ulps when the same rows are reduced as one outer batch.
    with torch.autocast(device_type=logits.device.type, enabled=False):
        order64 = ordered_support.to(torch.int64)
        score = torch.cat([
            pl_log_prob(
                logits[row:row + PROPOSAL_SCORING_BATCH_SIZE].float(),
                order64[row:row + PROPOSAL_SCORING_BATCH_SIZE],
            )
            for row in range(int(logits.shape[0]))
        ], dim=0)
    return score, logits


def _load_group(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        raise OutcomeGRPOError(f"cannot load recovery sidecar {path}: {exc}") from exc
    _require(isinstance(payload, Mapping), f"recovery sidecar {path} is not a mapping")
    return payload


@dataclass(frozen=True)
class ValidatedRecoveryCollection:
    """A complete immutable fold plus canonical, hash-bound sidecar receipts."""

    root: Path
    manifest_path: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    manifest_size: int
    manifest_mtime_ns: int
    split: str
    identity_digest: str
    items: tuple[Any, ...]
    receipts: tuple[dict[str, Any], ...]

    @classmethod
    def open(
        cls,
        root: str | os.PathLike[str],
        *,
        checkpoint_identity: Mapping[str, Any],
        expected_split: str | None = None,
        deep: bool = True,
        verify_sidecars: bool = True,
        stop_check: Any | None = None,
    ) -> "ValidatedRecoveryCollection":
        directory = Path(root).expanduser().resolve()
        manifest_path = directory / "manifest.json"
        _require(manifest_path.is_file(), f"recovery manifest is missing: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise OutcomeGRPOError(f"cannot read recovery manifest {manifest_path}: {exc}") from exc
        _require(isinstance(manifest, dict), "recovery manifest is not a mapping")
        _require(set(manifest) == _MANIFEST_FIELDS,
                 f"recovery manifest fields differ: "
                 f"{sorted(set(manifest) ^ _MANIFEST_FIELDS)}")
        _require(int(manifest["format_version"]) == recovery.FORMAT_VERSION,
                 "recovery manifest format version mismatch")
        _require(manifest["kind"] == "loom_outcome_recovery_collection",
                 "input is not an outcome-recovery collection")
        split = str(manifest["split"])
        allowed = set(recovery.TRAIN_FOLDS) | {"validation"}
        _require(split in allowed,
                 f"trainer accepts train0..train5 or validation, got {split!r}")
        if expected_split is not None:
            _require(split == str(expected_split),
                     f"recovery split {split!r} != configured {expected_split!r}")
        identity = manifest["identity"]
        _require(isinstance(identity, Mapping), "recovery identity is not a mapping")
        digest = recovery.identity_digest(identity)
        _require(digest == manifest["identity_digest"],
                 "recovery manifest identity digest mismatch")

        current_source = recovery.source_digest(_ROOT)
        expected_identity = recovery.collection_identity(
            checkpoint=dict(checkpoint_identity), split=split,
            source_sha256=current_source,
        )
        _require(identity == expected_identity,
                 "recovery collection identity does not exactly match the "
                 "authenticated checkpoint, split, and current collector source")

        items = tuple(recovery.collection_items(split))
        expected = {recovery.work_key(item): item for item in items}
        rows = manifest["groups"]
        _require(isinstance(rows, list), "recovery manifest groups is not a list")
        _require(len(rows) == len(items),
                 f"recovery fold is incomplete: {len(rows)}/{len(items)} groups")
        receipts: list[dict[str, Any]] = []
        for index, (key, item) in enumerate(expected.items()):
            if stop_check is not None and bool(stop_check()):
                raise _PreemptRequested(
                    f"preemption requested while authenticating {split} at group {index}"
                )
            raw = rows[index]
            _require(isinstance(raw, Mapping), f"receipt {index} is not a mapping")
            receipt = dict(raw)
            _require(set(receipt) == _RECEIPT_FIELDS,
                     f"receipt {index} fields differ: "
                     f"{sorted(set(receipt) ^ _RECEIPT_FIELDS)}")
            _require(receipt["group_id"] == key,
                     f"receipt order/identity mismatch at {index}: {receipt['group_id']!r}")
            _require(_valid_sha256(receipt["sha256"]), f"receipt {key} has invalid SHA-256")
            _require(int(receipt["n_arms"]) == recovery.GROUP_SIZE,
                     f"receipt {key} has wrong arm count")
            replans = [int(n) for n in receipt["n_replans_by_arm"]]
            rewards = [int(r) for r in receipt["terminal_rewards"]]
            _require(len(replans) == len(rewards) == recovery.GROUP_SIZE,
                     f"receipt {key} has wrong arm-vector length")
            _require(all(n > 0 for n in replans), f"receipt {key} has zero replans")
            _require(all(r in (0, 1) for r in rewards),
                     f"receipt {key} has a non-terminal reward")
            sidecar = cls._resolved_sidecar(directory, receipt)
            _require(sidecar.is_file(), f"recovery sidecar is missing: {sidecar}")
            _require(sidecar.stat().st_size == int(receipt["size"]),
                     f"recovery sidecar size changed: {sidecar}")
            if verify_sidecars:
                _require(recovery.sha256_file(sidecar) == receipt["sha256"],
                         f"recovery sidecar hash changed: {sidecar}")
            elif deep:
                raise ValueError("deep validation requires verify_sidecars=True")
            if deep:
                payload = _load_group(sidecar)
                recovery.validate_group_payload(
                    payload, item=item, expected_identity_digest=digest,
                    expected_split=split,
                )
                cls._match_receipt(payload, receipt)
            receipts.append(receipt)

        terminal = [sum(int(row["terminal_rewards"][arm]) for row in receipts)
                    for arm in range(recovery.GROUP_SIZE)]
        replans = [sum(int(row["n_replans_by_arm"][arm]) for row in receipts)
                   for arm in range(recovery.GROUP_SIZE)]
        expected_summary = {
            "status": "COMPLETE",
            "complete": True,
            "n_groups": len(items),
            "n_expected_groups": len(items),
            "n_trajectories": len(items) * recovery.GROUP_SIZE,
            "n_expected_trajectories": len(items) * recovery.GROUP_SIZE,
            "terminal_successes_by_arm": terminal,
            "replans_by_arm": replans,
        }
        _require(manifest["summary"] == expected_summary,
                 "recovery manifest COMPLETE summary does not match its receipts")
        stat = manifest_path.stat()
        return cls(
            root=directory,
            manifest_path=manifest_path,
            manifest=_json_copy(manifest),
            manifest_sha256=recovery.sha256_file(manifest_path),
            manifest_size=int(stat.st_size),
            manifest_mtime_ns=int(stat.st_mtime_ns),
            split=split,
            identity_digest=digest,
            items=items,
            receipts=tuple(receipts),
        )

    @staticmethod
    def _resolved_sidecar(root: Path, receipt: Mapping[str, Any]) -> Path:
        rel = Path(str(receipt.get("sidecar") or ""))
        _require(not rel.is_absolute() and ".." not in rel.parts,
                 "manifest sidecar path escapes collection directory")
        path = (root / rel).resolve()
        _require(path.parent == (root / "groups").resolve(),
                 "manifest sidecar is not directly inside groups/")
        return path

    @staticmethod
    def _match_receipt(payload: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
        arms = list(payload["arms"])
        replans = [int(arm["z"].shape[0]) for arm in arms]
        rewards = [int(float(arm["terminal_reward"])) for arm in arms]
        _require(replans == [int(n) for n in receipt["n_replans_by_arm"]],
                 f"sidecar {receipt['group_id']} replan counts differ from receipt")
        _require(rewards == [int(r) for r in receipt["terminal_rewards"]],
                 f"sidecar {receipt['group_id']} rewards differ from receipt")

    def assert_unchanged(self) -> None:
        stat = self.manifest_path.stat()
        _require(int(stat.st_size) == self.manifest_size,
                 "recovery manifest size changed during training")
        _require(int(stat.st_mtime_ns) == self.manifest_mtime_ns,
                 "recovery manifest mtime changed during training")
        _require(recovery.sha256_file(self.manifest_path) == self.manifest_sha256,
                 "recovery manifest bytes changed during training")

    def load(self, index: int) -> Mapping[str, Any]:
        self.assert_unchanged()
        receipt = self.receipts[index]
        path = self._resolved_sidecar(self.root, receipt)
        _require(path.is_file() and path.stat().st_size == int(receipt["size"]),
                 f"recovery sidecar changed: {path}")
        _require(recovery.sha256_file(path) == receipt["sha256"],
                 f"recovery sidecar hash changed: {path}")
        payload = _load_group(path)
        recovery.validate_group_payload(
            payload, item=self.items[index],
            expected_identity_digest=self.identity_digest,
            expected_split=self.split,
        )
        self._match_receipt(payload, receipt)
        return payload

    def assert_all_sidecars_unchanged(self, stop_check: Any | None = None) -> None:
        """Rehash the complete manifest closure, including unused equal groups."""
        self.assert_unchanged()
        for index, receipt in enumerate(self.receipts):
            if stop_check is not None and bool(stop_check()):
                raise _PreemptRequested(
                    f"preemption requested while rehashing {self.split} at {index}"
                )
            path = self._resolved_sidecar(self.root, receipt)
            _require(path.is_file() and path.stat().st_size == int(receipt["size"]),
                     f"recovery sidecar changed: {path}")
            _require(recovery.sha256_file(path) == receipt["sha256"],
                     f"recovery sidecar hash changed: {path}")
        self.assert_unchanged()

    def provenance(self) -> dict[str, Any]:
        return {
            "path": str(self.root),
            "manifest": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "identity_digest": self.identity_digest,
            "split": self.split,
            "n_groups": len(self.receipts),
            "n_trajectories": len(self.receipts) * recovery.GROUP_SIZE,
            "terminal_successes_by_arm": list(
                self.manifest["summary"]["terminal_successes_by_arm"]
            ),
            "replans_by_arm": list(self.manifest["summary"]["replans_by_arm"]),
            "collector_source": _json_copy(self.manifest["identity"]["source"]),
        }

    def informative_indices(self) -> tuple[int, ...]:
        """Groups whose eight terminal rewards are not all identical."""
        return tuple(
            index for index, receipt in enumerate(self.receipts)
            if len({int(value) for value in receipt["terminal_rewards"]}) > 1
        )


def _assert_owner_collection_snapshot(
    collection: ValidatedRecoveryCollection,
    expected: Mapping[str, Any],
) -> None:
    _require(
        collection.provenance() == _json_copy(expected),
        f"metadata reopen for {collection.split} differs from the owner-rank "
        "deep-authenticated manifest snapshot",
    )


def _stable_seed(*parts: object) -> int:
    raw = "\x1f".join(repr(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") & 0x7FFF_FFFF_FFFF_FFFF


class DeterministicOutcomeSampler:
    """Pure step/rank sampler with equal arms and uniform replan traversal."""

    def __init__(
        self,
        informative_groups: Sequence[Sequence[int]],
        *,
        seed: int,
        rank: int,
        world_size: int,
        start_step: int = START_STEP,
        updates_per_fold: int = UPDATES_PER_FOLD,
        contexts_per_arm: int = 2,
        identity_digests: Sequence[str] = (),
    ) -> None:
        self.groups = tuple(tuple(int(index) for index in fold)
                            for fold in informative_groups)
        if len(self.groups) != N_FOLDS:
            raise ValueError(f"expected {N_FOLDS} folds, got {len(self.groups)}")
        if any(not fold for fold in self.groups):
            raise ValueError("every fold needs at least one informative reward group")
        if rank < 0 or world_size <= 0 or rank >= world_size:
            raise ValueError(f"invalid rank/world {rank}/{world_size}")
        if contexts_per_arm <= 0:
            raise ValueError("contexts_per_arm must be positive")
        if any(len(fold) < world_size for fold in self.groups):
            raise ValueError(
                "every fold needs at least world_size informative groups for "
                "within-step rank uniqueness"
            )
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.start_step = int(start_step)
        self.updates_per_fold = int(updates_per_fold)
        self.contexts_per_arm = int(contexts_per_arm)
        self.identity_digests = tuple(str(value) for value in identity_digests)
        if self.identity_digests and len(self.identity_digests) != N_FOLDS:
            raise ValueError("collection identity count does not match fold count")
        self._group_permutations: dict[tuple[int, int], tuple[int, ...]] = {}
        self._replan_permutations: dict[tuple[int, int, int, int, int], tuple[int, ...]] = {}

    @property
    def stop_step(self) -> int:
        return self.start_step + len(self.groups) * self.updates_per_fold

    def _position(self, step: int) -> tuple[int, int]:
        offset = int(step) - self.start_step
        if offset < 0 or int(step) >= self.stop_step:
            raise ValueError(
                f"step {step} is outside [{self.start_step},{self.stop_step})"
            )
        return offset // self.updates_per_fold, offset % self.updates_per_fold

    def _group_order(self, fold: int, cycle: int) -> tuple[int, ...]:
        # Repeat one authenticated, seeded permutation.  Re-shuffling at every
        # cycle makes a non-divisible cycle boundary capable of assigning the
        # same group to two ranks in one optimizer step.  A repeated
        # permutation keeps every contiguous world-size slice duplicate-free
        # while retaining exact balanced traversal and pure step/rank resume.
        key = (fold, 0)
        if key not in self._group_permutations:
            values = self.groups[fold]
            generator = torch.Generator(device="cpu")
            generator.manual_seed(_stable_seed(
                "outcome-group", self.seed, fold,
                self.identity_digests[fold] if self.identity_digests else "",
            ))
            permutation = torch.randperm(len(values), generator=generator).tolist()
            self._group_permutations[key] = tuple(values[index] for index in permutation)
        return self._group_permutations[key]

    def group_at(self, step: int) -> tuple[int, int, int]:
        """Return ``(fold_index, manifest_group_index, visit_cycle)``."""
        fold, fold_step = self._position(step)
        global_draw = fold_step * self.world_size + self.rank
        cycle, offset = divmod(global_draw, len(self.groups[fold]))
        return fold, self._group_order(fold, cycle)[offset], cycle

    def _replan_order(
        self, fold: int, group: int, arm: int, epoch: int, n_replans: int,
    ) -> tuple[int, ...]:
        key = (fold, group, arm, epoch, n_replans)
        if key not in self._replan_permutations:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(_stable_seed(
                "outcome-replan", self.seed, fold, group, arm, epoch,
                self.identity_digests[fold] if self.identity_digests else "",
            ))
            self._replan_permutations[key] = tuple(
                torch.randperm(n_replans, generator=generator).tolist()
            )
        return self._replan_permutations[key]

    def replans_at(
        self, step: int, n_replans_by_arm: Sequence[int],
    ) -> dict[int, tuple[int, ...]]:
        if len(n_replans_by_arm) != recovery.GROUP_SIZE:
            raise ValueError(
                f"expected {recovery.GROUP_SIZE} replan counts, "
                f"got {len(n_replans_by_arm)}"
            )
        fold, group, visit = self.group_at(step)
        result: dict[int, tuple[int, ...]] = {}
        for arm in range(1, recovery.GROUP_SIZE):
            n = int(n_replans_by_arm[arm])
            if n <= 0:
                raise ValueError(f"arm {arm} has no replans")
            chosen: list[int] = []
            begin = visit * self.contexts_per_arm
            for draw in range(begin, begin + self.contexts_per_arm):
                epoch, offset = divmod(draw, n)
                chosen.append(self._replan_order(
                    fold, group, arm, epoch, n,
                )[offset])
            result[arm] = tuple(chosen)
        return result

    def state_dict(self, global_step: int) -> dict[str, Any]:
        # Rank is deliberately absent: one rank-0 checkpoint is valid on every
        # rank, while rank still participates in every pure draw.
        return {
            "kind": "pure_step_outcome_sampler_rank_unique_v2",
            "seed": self.seed,
            "world_size": self.world_size,
            "start_step": self.start_step,
            "stop_step": self.stop_step,
            "updates_per_fold": self.updates_per_fold,
            "contexts_per_arm": self.contexts_per_arm,
            "identity_digests": list(self.identity_digests),
            "informative_groups": [list(fold) for fold in self.groups],
            "global_step": int(global_step),
        }

    def validate_state_dict(self, state: Mapping[str, Any], global_step: int) -> None:
        _require(dict(state) == self.state_dict(global_step),
                 "outcome sampler checkpoint differs from this config/world/step")


def _load_authenticated_parent(
    checkpoint: str | os.PathLike[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = recovery.authenticate_seed_checkpoint(checkpoint)
    return _load_parent_from_identity(identity), identity


def _load_parent_from_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Load after one rank has broadcast the full-file authenticated identity."""
    try:
        payload = torch.load(identity["path"], map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        raise OutcomeGRPOError(f"cannot load authenticated parent checkpoint: {exc}") from exc
    _require(isinstance(payload, dict), "parent checkpoint is not a mapping")
    _require(int(payload.get("global_step", -1)) == recovery.SEED_GLOBAL_STEP,
             "parent checkpoint embedded global_step mismatch")
    _require(str(payload.get("config_hash") or "") == recovery.SEED_CONFIG_HASH,
             "parent checkpoint embedded config_hash mismatch")
    cfg = payload.get("resolved_config")
    _require(isinstance(cfg, dict), "parent checkpoint has no resolved_config")
    _require(_config_hash(cfg) == payload["config_hash"],
             "parent checkpoint resolved_config hash mismatch")
    state = payload.get("model")
    _require(isinstance(state, dict) and state, "parent checkpoint has no model state")
    _require(all(isinstance(v, Tensor) for v in state.values()),
             "parent checkpoint model state is not a flat tensor mapping")
    _require(_all_finite_tensors(state.values()), "parent checkpoint contains nan/inf")
    return payload


def _load_proposal(parent: Mapping[str, Any], *, device: torch.device) -> Proposal:
    cfg = parent["resolved_config"]
    kwargs = dict(cfg.get("model", {}).get("proposal", {}) or {})
    proposal = Proposal(**kwargs)
    state = submodule_state(parent["model"], "proposal")
    _require(state is not None, "parent checkpoint has no proposal tensors")
    try:
        proposal.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise OutcomeGRPOError(f"parent proposal state is not exact: {exc}") from exc
    proposal.to(device)
    # Collection and deployment score the proposal in eval mode.  Returning a
    # train-mode module here would fail the exact-scoring preflight before the
    # authentication function gets a chance to call ``eval()`` itself.
    return proposal.eval()


@torch.no_grad()
def authenticate_behaviour_policy(
    proposal: nn.Module,
    collection: ValidatedRecoveryCollection,
    *,
    device: torch.device,
    chunk_replans: int,
    stop_check: Any | None = None,
) -> dict[str, Any]:
    """Recompute every recorded old PL score/weight under the parent proposal."""
    _require(chunk_replans > 0, "chunk_replans must be positive")
    proposal.eval()
    dtype = next(proposal.parameters()).dtype
    max_lp_error = 0.0
    max_coeff_error = 0.0
    atoms = 0
    sampled_atoms = 0
    for group_index in range(len(collection.receipts)):
        if stop_check is not None and bool(stop_check()):
            raise _PreemptRequested(
                f"preemption requested during {collection.split} behaviour replay "
                f"at group {group_index}"
            )
        payload = collection.load(group_index)
        for arm_index, arm in enumerate(payload["arms"]):
            n = int(arm["z"].shape[0])
            lang = _batched_lang(arm["lang"], n, device, dtype)
            for lo in range(0, n, chunk_replans):
                if stop_check is not None and bool(stop_check()):
                    raise _PreemptRequested(
                        f"preemption requested during {collection.split} behaviour replay"
                    )
                hi = min(lo + chunk_replans, n)
                z = arm["z"][lo:hi].to(device=device, dtype=dtype, non_blocking=True)
                order = arm["ordered_support"][lo:hi].to(device=device)
                # Sidecar transfer remains chunked, while the shared scorer
                # reproduces the collector's B=1 fp32 execution row by row.
                current, logits = stored_order_logprob(
                    proposal, z, lang[lo:hi], order,
                )
                expected_lp = arm["old_logprob"][lo:hi].to(device=device).float()
                expected_coeff = arm["coeff"][lo:hi].to(device=device).float()
                current_coeff = weights_from_logits(
                    logits.float(), order.to(torch.int64), logits.shape[-1],
                ).float()
                lp_err = (float((current.float() - expected_lp).abs().max())
                          if arm_index > 0 else 0.0)
                coeff_err = float((current_coeff - expected_coeff).abs().max())
                max_lp_error = max(max_lp_error, lp_err)
                max_coeff_error = max(max_coeff_error, coeff_err)
                if arm_index > 0:
                    _require(torch.allclose(
                        current.float(), expected_lp,
                        atol=BEHAVIOUR_LOGPROB_ATOL, rtol=BEHAVIOUR_LOGPROB_RTOL,
                    ), f"parent proposal old-logprob mismatch in group "
                       f"{payload['group_id']} arm {arm_index}; max_abs={lp_err:.6g}")
                _require(torch.allclose(
                    current_coeff, expected_coeff,
                    atol=BEHAVIOUR_COEFF_ATOL, rtol=BEHAVIOUR_COEFF_RTOL,
                ), f"parent proposal coefficient mismatch in group "
                   f"{payload['group_id']} arm {arm_index}; max_abs={coeff_err:.6g}")
                atoms += hi - lo
                if arm_index > 0:
                    sampled_atoms += hi - lo
    collection.assert_unchanged()
    geometry = _proposal_scoring_geometry(device)
    return {
        "passed": True,
        "all_atoms": atoms,
        "ratio_eligible_atoms": sampled_atoms,
        "arm0_ratio_eligible_atoms": 0,
        "max_abs_old_logprob_error": max_lp_error,
        "max_abs_coeff_error": max_coeff_error,
        "max_abs_logratio": max_lp_error,
        "proposal_replay_batch_size": PROPOSAL_SCORING_BATCH_SIZE,
        "proposal_scoring": geometry,
        "transfer_chunk_replans": int(chunk_replans),
        "logprob_atol": BEHAVIOUR_LOGPROB_ATOL,
        "logprob_rtol": BEHAVIOUR_LOGPROB_RTOL,
        "coeff_atol": BEHAVIOUR_COEFF_ATOL,
        "coeff_rtol": BEHAVIOUR_COEFF_RTOL,
    }


def _require_exact_gathered_behaviour_auth(
    reports: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Require numerical identity for every deeply replayed behavior atom."""
    rows = []
    for index in sorted(reports):
        report = reports[index]
        rows.append({
            "index": int(index),
            "all_atoms": int(report.get("all_atoms", 0)),
            "max_abs_coeff_error": float(
                report.get("max_abs_coeff_error", math.inf)
            ),
            "max_abs_old_logprob_error": float(
                report.get("max_abs_old_logprob_error", math.inf)
            ),
            "max_abs_logratio": float(report.get("max_abs_logratio", math.inf)),
        })
    bounded = bool(rows) and all(
        row["all_atoms"] > 0
        and math.isfinite(row["max_abs_coeff_error"])
        and math.isfinite(row["max_abs_old_logprob_error"])
        and math.isfinite(row["max_abs_logratio"])
        and row["max_abs_coeff_error"] >= 0.0
        and row["max_abs_old_logprob_error"] >= 0.0
        and row["max_abs_logratio"] >= 0.0
        and row["max_abs_coeff_error"] <= BEHAVIOUR_IDENTITY_MAX_COEFF_ERROR
        and row["max_abs_old_logprob_error"]
            <= BEHAVIOUR_IDENTITY_MAX_ABS_LOGRATIO
        and row["max_abs_logratio"] <= BEHAVIOUR_IDENTITY_MAX_ABS_LOGRATIO
        for row in rows
    )
    _require(
        bounded,
        "deep behaviour replay exceeded numerical-identity bounds: "
        + json.dumps(rows, sort_keys=True, allow_nan=True),
    )
    return {
        "passed": True,
        "splits": len(rows),
        "all_atoms": sum(row["all_atoms"] for row in rows),
        "max_abs_coeff_error": max(row["max_abs_coeff_error"] for row in rows),
        "max_abs_old_logprob_error": max(
            row["max_abs_old_logprob_error"] for row in rows
        ),
        "max_abs_logratio": max(row["max_abs_logratio"] for row in rows),
        "max_abs_coeff_error_threshold": BEHAVIOUR_IDENTITY_MAX_COEFF_ERROR,
        "max_abs_logratio_threshold": BEHAVIOUR_IDENTITY_MAX_ABS_LOGRATIO,
    }


def _call_estimator(estimator: nn.Module, feats: Mapping[str, Tensor], z: Tensor | None,
                    embodiment: str) -> Tensor:
    try:
        accepts = "embodiment" in inspect.signature(estimator.forward).parameters
    except (TypeError, ValueError):
        accepts = False
    return estimator(feats, z, embodiment=embodiment) if accepts else estimator(feats, z)


def _call_q_action(q_action: nn.Module, actions: Tensor, z: Tensor,
                   embodiment: str) -> Tensor:
    try:
        accepts = "embodiment" in inspect.signature(q_action.forward).parameters
    except (TypeError, ValueError):
        accepts = False
    return (q_action(actions, z, embodiment=embodiment)
            if accepts else q_action(actions, z))


@dataclass
class ExpertAnchor:
    """Frozen demonstration target producer for the existing sparse-CE loss."""

    proposal: nn.Module
    estimator: nn.Module
    q_action: nn.Module
    sampler: Any
    device: torch.device
    temperature: float
    weight: float
    detach_belief: bool
    data_provenance: dict[str, Any]
    _cache: dict[int, tuple[list[Tensor], Tensor, list[Tensor], str]] = field(
        default_factory=dict,
    )

    @classmethod
    def from_parent(
        cls,
        parent: Mapping[str, Any],
        proposal: nn.Module,
        *,
        trainer_cfg: Mapping[str, Any],
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
    ) -> "ExpertAnchor":
        parent_cfg = parent["resolved_config"]
        cfg = copy.deepcopy(dict(trainer_cfg))
        dcfg = dict(cfg.get("data", {}) or {})
        pcfg = dict(cfg.get("losses", {}).get("proposal", {}) or {})
        if dcfg.get("source") != "libero" or bool(dcfg.get("action_free", False)):
            raise ExpertAnchorUnavailable(
                "expert anchor requires labelled LIBERO demonstrations"
            )
        if str(dcfg.get("trajectory_split")) != "train":
            raise ExpertAnchorUnavailable(
                "expert anchor must use data.trajectory_split='train'"
            )
        if list(dcfg.get("holdout_demo_keys") or ()) != ["demo_49"]:
            raise ExpertAnchorUnavailable(
                "expert anchor must exclude exactly holdout_demo_keys=[demo_49]"
            )
        if not bool(pcfg.get("enabled", False)) or pcfg.get("mode") != "sparse_ce":
            raise ExpertAnchorUnavailable(
                "parent recipe does not define the existing sparse-CE proposal target"
            )
        temperature = float(pcfg.get("temperature", 1.0))
        weight = float(pcfg.get("weight", 0.0))
        detach = bool(pcfg.get("detach_belief", False))
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ExpertAnchorUnavailable("parent sparse-CE temperature is invalid")
        if not math.isfinite(weight) or weight <= 0.0:
            raise ExpertAnchorUnavailable("parent sparse-CE weight is not positive")
        if not detach:
            raise ExpertAnchorUnavailable(
                "parent sparse-CE does not stop-gradient its belief target"
            )

        embodiments = list(dcfg.get("embodiments", ()))
        if not embodiments:
            raise ExpertAnchorUnavailable("parent data config has no embodiments")
        try:
            from loom.model.estimator import Estimator  # noqa: PLC0415
            from loom.heads.q_action import QAction  # noqa: PLC0415
            from loom.data.loader import build_loader  # noqa: PLC0415

            ekw = dict(parent_cfg.get("model", {}).get("estimator", {}) or {})
            ekw.setdefault("embodiments", embodiments)
            qkw = dict(parent_cfg.get("model", {}).get("q_action", {}) or {})
            qkw.pop("embodiments", None)
            qkw.pop("default_embodiment", None)
            estimator = Estimator(**ekw)
            q_action = QAction(
                embodiments=embodiments,
                default_embodiment=(embodiments[0] if len(embodiments) == 1 else None),
                **qkw,
            )
            estate = submodule_state(parent["model"], "estimator")
            qstate = submodule_state(parent["model"], "q_action")
            if estate is None or qstate is None:
                raise ExpertAnchorUnavailable(
                    "parent checkpoint lacks estimator/q_action target tensors"
                )
            estimator.load_state_dict(estate, strict=True)
            q_action.load_state_dict(qstate, strict=True)
            estimator.to(device).eval().requires_grad_(False)
            q_action.to(device).eval().requires_grad_(False)

            # Preserve the inherited ``cache/`` placeholder.  The canonical
            # loader deliberately lets LOOM_CACHE_DIR outrank that placeholder;
            # making it absolute here would turn it into an explicit override
            # and bypass the real cache exported by scripts/env.sh.
            loader_cfg = copy.deepcopy(cfg)
            sampler = build_loader(
                loader_cfg, rank=rank, world=world_size,
                seed=int(cfg.get("run", {}).get("seed", TRAIN_SEED)),
                device=str(device),
            )
        except ExpertAnchorUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ExpertAnchorUnavailable(
                "cannot construct the existing estimator/q_action demo target path: "
                f"{exc}"
            ) from exc

        try:
            manifest = sampler.trajectory_manifest()
        except Exception as exc:  # noqa: BLE001
            raise ExpertAnchorUnavailable(
                f"expert anchor has no authenticated trajectory manifest: {exc}"
            ) from exc
        expected_manifest = dict(
            cfg.get("outcome_grpo", {}).get("anchor_manifest", {}) or {}
        )
        required_expected = {"digest", "n_tasks", "n_trajectories", "n_windows"}
        if set(expected_manifest) != required_expected:
            raise ExpertAnchorUnavailable(
                "outcome_grpo.anchor_manifest must pin digest, n_tasks, "
                "n_trajectories, and n_windows"
            )
        for key in ("digest", "n_tasks", "n_trajectories"):
            if manifest.get(key) != expected_manifest[key]:
                raise ExpertAnchorUnavailable(
                    f"expert anchor manifest {key}={manifest.get(key)!r} != "
                    f"configured {expected_manifest[key]!r}"
                )
        if int(sampler.n_windows) != int(expected_manifest["n_windows"]):
            raise ExpertAnchorUnavailable(
                f"expert anchor windows={sampler.n_windows} != configured "
                f"{expected_manifest['n_windows']}"
            )
        data_provenance = {
            "source": dcfg.get("source"),
            "embodiments": embodiments,
            "action_free": bool(dcfg.get("action_free", False)),
            "batch_per_gpu": int(dcfg.get("batch_per_gpu", 0)),
            "sampling": dcfg.get("sampling"),
            "loader": f"{type(sampler).__module__}.{type(sampler).__qualname__}",
            "n_windows": (int(sampler.n_windows)
                          if getattr(sampler, "n_windows", None) is not None else None),
            "trajectory_manifest": manifest,
            "sampler_seed": int(cfg.get("run", {}).get("seed", TRAIN_SEED)),
            "sampler_rank": int(rank),
            "sampler_world": int(world_size),
        }
        return cls(
            proposal=proposal,
            estimator=estimator,
            q_action=q_action,
            sampler=sampler,
            device=device,
            temperature=temperature,
            weight=weight,
            detach_belief=detach,
            data_provenance=data_provenance,
        )

    def _prepare(self, global_step: int) -> tuple[list[Tensor], Tensor, list[Tensor], str]:
        step = int(global_step)
        if step in self._cache:
            return self._cache.pop(step)
        try:
            from loom.train.loop import _to_device  # noqa: PLC0415

            window = self.sampler.next(step)
            dtype = next(self.proposal.parameters()).dtype
            window = _to_device(window, str(self.device), dtype=dtype)
        except Exception as exc:  # noqa: BLE001
            raise ExpertAnchorUnavailable(
                f"cannot read labelled expert-anchor batch at deterministic step {step}: {exc}"
            ) from exc
        meta = dict(window.get("data_meta") or {})
        manifest = self.data_provenance["trajectory_manifest"]
        expected_meta = {
            "source": "libero",
            "split": "train",
            "manifest_digest": manifest["digest"],
        }
        for key, expected in expected_meta.items():
            if meta.get(key) != expected:
                raise ExpertAnchorUnavailable(
                    f"expert-anchor batch provenance {key}={meta.get(key)!r} != {expected!r}"
                )
        actions = window.get("actions")
        if not isinstance(actions, Tensor):
            raise ExpertAnchorUnavailable(
                f"expert-anchor batch at step {step} has no action targets"
            )
        feats = list(window.get("feats") or ())
        if len(feats) < C.DEPTH:
            raise ExpertAnchorUnavailable(
                f"expert-anchor batch has {len(feats)} states, needs at least {C.DEPTH}"
            )
        if actions.ndim != 4 or actions.shape[1] < C.DEPTH:
            raise ExpertAnchorUnavailable(
                f"expert-anchor actions have invalid shape {tuple(actions.shape)}"
            )
        lang = window.get("lang")
        embodiment = str(window.get("embodiment") or "")
        if not isinstance(lang, Tensor) or not embodiment:
            raise ExpertAnchorUnavailable("expert-anchor batch lacks language/embodiment")

        with torch.no_grad():
            z: Tensor | None = None
            for prefix in window.get("burn_in_feats", ()) or ():
                z = _call_estimator(self.estimator, prefix, z, embodiment)
            if z is not None:
                z = z.detach()
            beliefs: list[Tensor] = []
            for state in feats:
                z = _call_estimator(self.estimator, state, z, embodiment)
                beliefs.append(z.detach())
            targets = [
                _call_q_action(
                    self.q_action, actions[:, horizon], beliefs[horizon], embodiment,
                ).detach()
                for horizon in range(C.DEPTH)
            ]
        for horizon, target in enumerate(targets):
            if target.shape != (beliefs[horizon].shape[0], C.M):
                raise ExpertAnchorUnavailable(
                    f"q_action target h{horizon + 1} has shape {tuple(target.shape)}"
                )
            if not bool(torch.isfinite(target).all()) or bool((target < 0).any()):
                raise ExpertAnchorUnavailable(
                    f"q_action target h{horizon + 1} is nonfinite/negative"
                )
            target_tol = max(
                8.0 * float(torch.finfo(target.dtype).eps), 1e-4,
            ) if target.is_floating_point() else 1e-4
            if not torch.allclose(
                target.float().sum(-1), torch.ones(target.shape[0], device=target.device),
                atol=target_tol, rtol=0,
            ):
                raise ExpertAnchorUnavailable(
                    f"q_action target h{horizon + 1} is not on the simplex"
                )
        return beliefs, lang, targets, embodiment

    def preflight(self, global_step: int = START_STEP) -> dict[str, Any]:
        prepared = self._prepare(global_step)
        self._cache[int(global_step)] = prepared
        beliefs, lang, targets, _ = prepared
        with torch.no_grad():
            values = [proposal_sparse_ce_loss(
                self.proposal, beliefs[h], lang, targets[h],
                temperature=self.temperature,
                detach_belief=self.detach_belief,
            ) for h in range(C.DEPTH)]
        if not _all_finite_tensors(values):
            raise ExpertAnchorUnavailable("expert-anchor sparse CE is nonfinite")
        return {
            "passed": True,
            "loss": "loom.losses.proposal_bc.proposal_sparse_ce_loss",
            "teacher": "frozen q_action(action_segment, frozen_estimator_belief)",
            "temperature": self.temperature,
            "weight": self.weight,
            "horizons": C.DEPTH,
            "first_batch_sparse_ce": float(torch.stack(values).mean()),
            "data": _json_copy(self.data_provenance),
        }

    def loss(self, global_step: int) -> tuple[Tensor, dict[str, float]]:
        beliefs, lang, targets, _ = self._prepare(global_step)
        terms = [proposal_sparse_ce_loss(
            self.proposal, beliefs[h], lang, targets[h],
            temperature=self.temperature,
            detach_belief=self.detach_belief,
        ) for h in range(C.DEPTH)]
        loss = torch.stack(terms).mean()
        if not bool(torch.isfinite(loss)):
            raise OutcomeGRPOError(
                f"expert-anchor sparse CE became nonfinite at step {global_step}"
            )
        return self.weight * loss, {
            "sparse_ce": float(loss.detach()),
            "weighted_sparse_ce": float((self.weight * loss).detach()),
        }

    def unexpected_gradients(self) -> list[str]:
        out: list[str] = []
        for prefix, module in (("estimator", self.estimator), ("q_action", self.q_action)):
            for name, parameter in module.named_parameters():
                if parameter.grad is not None:
                    out.append(f"{prefix}.{name}")
        return out


@torch.no_grad()
def expert_support_overlap(
    anchor: ExpertAnchor,
    *,
    start_step: int = START_STEP,
    batches: int = EXPERT_GATE_BATCHES,
) -> float:
    """Mean top-k support overlap with frozen q_action on fixed train batches."""
    if batches <= 0:
        raise ValueError("expert support gate needs at least one batch")
    total = 0.0
    count = 0
    for step in range(int(start_step), int(start_step) + int(batches)):
        beliefs, lang, targets, _ = anchor._prepare(step)
        for horizon in range(C.DEPTH):
            student = _rowwise_proposal_logits_fp32(
                anchor.proposal, beliefs[horizon], lang,
            ).float()
            si = student.topk(C.TOPK, dim=-1).indices
            ti = targets[horizon].float().topk(C.TOPK, dim=-1).indices
            overlap = (si.unsqueeze(-1) == ti.unsqueeze(-2)).any(-1).float().mean()
            total += float(overlap)
            count += 1
    return total / count


def _proposal_grad_health(proposal: nn.Module) -> tuple[float, list[str], list[str]]:
    total = 0.0
    missing: list[str] = []
    nonfinite: list[str] = []
    for name, parameter in proposal.named_parameters():
        grad = parameter.grad
        if grad is None:
            missing.append(name)
            continue
        if not bool(torch.isfinite(grad).all()):
            nonfinite.append(name)
            continue
        total += float(grad.detach().float().square().sum())
    return math.sqrt(total), missing, nonfinite


def _optimizer_finite(optimizer: torch.optim.Optimizer) -> bool:
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, Tensor) and not bool(torch.isfinite(value).all()):
                return False
    return True


def sampled_group_losses(
    proposal: nn.Module,
    payload: Mapping[str, Any],
    replan_indices: Mapping[int, Sequence[int]],
    *,
    device: torch.device,
    clip_eps: float = CLIP_EPS,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    """One equally weighted group/arm/replan terminal objective.

    ``replan_indices`` must contain arms 1--7 only and the same positive count
    for each arm.  Arm 0 enters the eight-reward normalisation and nowhere else.
    """
    expected_arms = set(range(1, recovery.GROUP_SIZE))
    if set(replan_indices) != expected_arms:
        raise ValueError(
            f"ratio/loss indices must be exactly arms 1..7, got {sorted(replan_indices)}"
        )
    counts = {len(tuple(replan_indices[arm])) for arm in expected_arms}
    if len(counts) != 1 or next(iter(counts), 0) <= 0:
        raise ValueError("every sampled arm must contribute the same positive replan count")
    rewards = torch.tensor(
        [float(arm["terminal_reward"]) for arm in payload["arms"]],
        dtype=torch.float32,
    )
    advantages = normalised_group_advantages(rewards).to(device)
    dtype = next(proposal.parameters()).dtype
    arm_losses: list[Tensor] = []
    sampled_logits: list[Tensor] = []
    ratios: list[Tensor] = []
    clipped: list[Tensor] = []
    logratios: list[Tensor] = []
    for arm_index in range(1, recovery.GROUP_SIZE):
        arm = payload["arms"][arm_index]
        index = torch.tensor(tuple(replan_indices[arm_index]), dtype=torch.int64)
        n = int(arm["z"].shape[0])
        if bool((index < 0).any()) or bool((index >= n).any()):
            raise ValueError(f"arm {arm_index} sampled replan outside [0,{n})")
        z = arm["z"].index_select(0, index).to(
            device=device, dtype=dtype, non_blocking=True,
        )
        order = arm["ordered_support"].index_select(0, index).to(device=device)
        old = arm["old_logprob"].index_select(0, index).to(device=device).float()
        lang = _batched_lang(arm["lang"], int(index.numel()), device, dtype)
        current, logits = stored_order_logprob(proposal, z, lang, order)
        objective, ratio, was_clipped = clipped_grpo_objective(
            current, old, advantages[arm_index], clip_eps=clip_eps,
        )
        if not bool(torch.isfinite(objective).all()) or not bool(torch.isfinite(ratio).all()):
            raise OutcomeGRPOError(
                f"nonfinite sampled GRPO term in {payload['group_id']} arm {arm_index}"
            )
        arm_losses.append(-objective.mean())
        sampled_logits.append(logits)
        ratios.append(ratio.detach().reshape(-1))
        clipped.append(was_clipped.detach().reshape(-1))
        logratios.append((current.detach().float() - old).reshape(-1))
    loss = torch.stack(arm_losses).mean()
    # One Switch statistic over the complete equal-weight 7-arm x context
    # batch.  Averaging seven independent two-row balance terms changes the
    # hard-routing frequency and is not the declared 14-context router loss.
    balance = proposal_switch_balance(torch.cat(sampled_logits, dim=0))
    ratio_all = torch.cat(ratios)
    clipped_all = torch.cat(clipped)
    logratio_all = torch.cat(logratios)
    # Accumulate sufficient statistics in fp64.  A float32 reduction over only
    # fourteen near-unit ratios can lose more ESS precision than the complete
    # admissible log-ratio envelope itself.
    ratio64 = ratio_all.double()
    ratio_sum = float(ratio64.sum())
    ratio_square_sum = float(ratio64.square().sum())
    ratio_atoms = float(ratio_all.numel())
    return loss, balance, {
        "grpo_loss": float(loss.detach()),
        "proposal_balance": float(balance.detach()),
        "ratio_mean": ratio_sum / ratio_atoms,
        "ratio_min": float(ratio_all.float().min()),
        "ratio_max": float(ratio_all.float().max()),
        "max_abs_logratio": float(logratio_all.abs().max()),
        "clip_fraction": float(clipped_all.float().mean()),
        "ratio_atoms": ratio_atoms,
        "ratio_sum": ratio_sum,
        "ratio_square_sum": ratio_square_sum,
        "ratio_ess_fraction": (
            ratio_sum * ratio_sum
            / max(ratio_atoms * ratio_square_sum, torch.finfo(torch.float64).tiny)
        ),
        "clipped_atoms": float(clipped_all.sum()),
        "proposal_scoring_batch_size": float(PROPOSAL_SCORING_BATCH_SIZE),
        "proposal_scoring_autocast": float(PROPOSAL_SCORING_AUTOCAST),
        "proposal_scoring_cuda_matmul_tf32": float(CUDA_MATMUL_TF32),
        "proposal_scoring_cudnn_tf32": float(CUDNN_TF32),
        "informative_group": float(len(set(rewards.tolist())) > 1),
    }


def _require_initial_behavior_ratio_identity(
    metrics: Mapping[str, float],
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Fail before backward unless ratios are numerical identity with behavior."""
    evidence = {
        "passed": False,
        "max_abs_logratio": float(metrics.get("max_abs_logratio", math.inf)),
        "ratio_min": float(metrics.get("ratio_min", math.nan)),
        "ratio_mean": float(metrics.get("ratio_mean", math.nan)),
        "ratio_max": float(metrics.get("ratio_max", math.nan)),
        "clip_fraction": float(metrics.get("clip_fraction", math.nan)),
        "ratio_atoms": int(metrics.get("ratio_atoms", 0)),
        "ratio_sum": float(metrics.get("ratio_sum", math.nan)),
        "ratio_square_sum": float(metrics.get("ratio_square_sum", math.nan)),
        "ratio_ess_fraction": float(
            metrics.get("ratio_ess_fraction", math.nan)
        ),
        "clipped_atoms": int(metrics.get("clipped_atoms", -1)),
        "max_abs_logratio_threshold": BEHAVIOUR_IDENTITY_MAX_ABS_LOGRATIO,
        "min_ess_fraction_threshold": INITIAL_RATIO_MIN_ESS_FRACTION,
        "proposal_scoring": _proposal_scoring_geometry(device),
    }
    values = [
        evidence["max_abs_logratio"], evidence["ratio_min"],
        evidence["ratio_mean"], evidence["ratio_max"],
        evidence["clip_fraction"], evidence["ratio_sum"],
        evidence["ratio_square_sum"], evidence["ratio_ess_fraction"],
    ]
    # The production ratio is fp32 exp(log-ratio).  At +/-2^-16 its outward
    # representable endpoints are exactly 1+/-2^-16; double exp would be a
    # slightly narrower and therefore incorrect acceptance envelope.
    ratio_low = 1.0 - BEHAVIOUR_IDENTITY_MAX_ABS_LOGRATIO
    ratio_high = 1.0 + BEHAVIOUR_IDENTITY_MAX_ABS_LOGRATIO
    derived_mean = evidence["ratio_sum"] / max(evidence["ratio_atoms"], 1)
    derived_ess = (
        evidence["ratio_sum"] * evidence["ratio_sum"]
        / max(
            evidence["ratio_atoms"] * evidence["ratio_square_sum"],
            torch.finfo(torch.float64).tiny,
        )
    )
    bounded = (
        all(math.isfinite(float(value)) for value in values)
        and evidence["ratio_atoms"] == EXPECTED_INITIAL_RATIO_ATOMS_PER_RANK
        and evidence["max_abs_logratio"] >= 0.0
        and evidence["max_abs_logratio"] <= BEHAVIOUR_IDENTITY_MAX_ABS_LOGRATIO
        and evidence["ratio_min"] >= ratio_low
        and ratio_low <= evidence["ratio_mean"] <= ratio_high
        and evidence["ratio_max"] <= ratio_high
        and evidence["ratio_min"] <= evidence["ratio_mean"] <= evidence["ratio_max"]
        and evidence["clip_fraction"] == 0.0
        and evidence["ratio_ess_fraction"] >= INITIAL_RATIO_MIN_ESS_FRACTION
        and evidence["ratio_mean"] == derived_mean
        and evidence["ratio_ess_fraction"] == derived_ess
        and evidence["clipped_atoms"] == 0
    )
    evidence["passed"] = bounded
    _require(
        bounded,
        "initial behavior ratio exceeds numerical-identity bounds before "
        "optimizer step: "
        + json.dumps(evidence, sort_keys=True, allow_nan=True),
    )
    return evidence


def _assemble_initial_behavior_identity(
    rows: Sequence[Mapping[str, Any]],
    *,
    world: int,
    config_hash: str,
    trainer_source: Mapping[str, Any],
    parent_identity: Mapping[str, Any],
    exact_behaviour_identity: Mapping[str, Any],
    start_checkpoint_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind every rank's first production forward before any backward/update."""
    _require(len(rows) == world,
             f"initial behavior gate gathered {len(rows)} ranks, expected {world}")
    by_rank = {int(row.get("rank", -1)): dict(row) for row in rows}
    _require(set(by_rank) == set(range(world)),
             "initial behavior gate rank identities are missing or duplicated")
    evidence = [
        _validate_initial_behavior_rank(by_rank[rank], expected_rank=rank)
        for rank in range(world)
    ]
    geometries = [dict(row.get("proposal_scoring") or {}) for row in evidence]
    _require(all(row == geometries[0] for row in geometries),
             "initial behavior identity used different rank scoring geometry")
    _validate_start_checkpoint_identity(start_checkpoint_identity)
    return {
        "format_version": FORMAT_VERSION,
        "kind": "outcome_grpo_initial_behavior_identity",
        "passed": True,
        "world_size": int(world),
        "config_hash": str(config_hash),
        "trainer_source": _json_copy(trainer_source),
        "parent": _json_copy(parent_identity),
        "exact_behaviour_identity": _json_copy(exact_behaviour_identity),
        "start_checkpoint_identity": _json_copy(start_checkpoint_identity),
        "strict_determinism": _configure_strict_outcome_determinism(),
        "proposal_scoring": _json_copy(geometries[0]),
        "ranks": _json_copy(evidence),
    }


def _validate_initial_behavior_rank(
    row: Mapping[str, Any],
    *,
    expected_rank: int,
) -> dict[str, Any]:
    """Require the complete numerical-identity witness for one rank."""
    value = dict(row)
    atoms = int(value.get("ratio_atoms", 0))
    ratio_sum = float(value.get("ratio_sum", math.nan))
    ratio_square_sum = float(value.get("ratio_square_sum", math.nan))
    ratio_mean = float(value.get("ratio_mean", math.nan))
    ratio_ess = float(value.get("ratio_ess_fraction", math.nan))
    derived_mean = ratio_sum / max(atoms, 1)
    derived_ess = ratio_sum * ratio_sum / max(
        atoms * ratio_square_sum, torch.finfo(torch.float64).tiny,
    )
    _require(
        int(value.get("rank", -1)) == int(expected_rank)
        and bool(value.get("passed"))
        and atoms == EXPECTED_INITIAL_RATIO_ATOMS_PER_RANK
        and float(value.get("max_abs_logratio", math.inf)) >= 0.0
        and float(value.get("max_abs_logratio", math.inf))
            <= BEHAVIOUR_IDENTITY_MAX_ABS_LOGRATIO
        and float(value.get("ratio_min", math.nan))
            >= 1.0 - BEHAVIOUR_IDENTITY_MAX_ABS_LOGRATIO
        and 1.0 - BEHAVIOUR_IDENTITY_MAX_ABS_LOGRATIO
            <= float(value.get("ratio_mean", math.nan))
            <= 1.0 + BEHAVIOUR_IDENTITY_MAX_ABS_LOGRATIO
        and float(value.get("ratio_max", math.nan))
            <= 1.0 + BEHAVIOUR_IDENTITY_MAX_ABS_LOGRATIO
        and float(value.get("ratio_min", math.nan))
            <= ratio_mean
            <= float(value.get("ratio_max", math.nan))
        and float(value.get("clip_fraction", math.nan)) == 0.0
        and float(value.get("ratio_ess_fraction", -math.inf))
            >= INITIAL_RATIO_MIN_ESS_FRACTION
        and ratio_sum > 0.0
        and ratio_square_sum > 0.0
        and ratio_mean == derived_mean
        and ratio_ess == derived_ess
        and int(value.get("clipped_atoms", -1)) == 0,
        f"initial behavior identity rank {expected_rank} exceeds bounds",
    )
    _require(
        float(value.get("max_abs_logratio_threshold", math.nan))
            == BEHAVIOUR_IDENTITY_MAX_ABS_LOGRATIO
        and float(value.get("min_ess_fraction_threshold", math.nan))
            == INITIAL_RATIO_MIN_ESS_FRACTION,
        f"initial behavior identity rank {expected_rank} changed thresholds",
    )
    geometry = dict(value.get("proposal_scoring") or {})
    device_type = str(geometry.get("device_type") or "")
    _require(
        device_type in {"cpu", "cuda"}
        and geometry == _proposal_scoring_geometry(torch.device(device_type)),
        f"initial behavior identity rank {expected_rank} changed scoring geometry",
    )
    return _json_copy(value)


def _validate_start_checkpoint_identity(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the optimizer-reset checkpoint bound into first-forward proof."""
    report = dict(value)
    proposal = dict(report.get("proposal") or {})
    _require(
        bool(report.get("checked"))
        and bool(report.get("passed"))
        and int(report.get("global_step", -1)) == START_STEP
        and int(report.get("optimizer_state_entries", -1)) == 0
        and report.get("optimizer_reset")
            == {"count": 1, "modules": ["proposal"]}
        and _valid_sha256(proposal.get("sha256"))
        and int(proposal.get("n_tensors", 0)) > 0
        and int(proposal.get("n_bytes", 0)) > 0,
        "initial behavior identity has invalid START checkpoint evidence",
    )
    return _json_copy(report)


def _validate_initial_behavior_identity(
    report: Mapping[str, Any] | None,
    *,
    global_step: int,
    world: int,
    config_hash: str,
    trainer_source: Mapping[str, Any],
    parent_identity: Mapping[str, Any],
    exact_behaviour_identity: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Authenticate the checkpoint-bound first-forward gate on resume."""
    if int(global_step) == START_STEP:
        _require(report is None,
                 "START-step checkpoint unexpectedly contains post-forward evidence")
        return None
    _require(isinstance(report, Mapping),
             "post-start checkpoint lacks initial behavior identity evidence")
    value = dict(report)
    _require(
        value.get("kind") == "outcome_grpo_initial_behavior_identity"
        and bool(value.get("passed"))
        and int(value.get("world_size", -1)) == int(world)
        and str(value.get("config_hash") or "") == str(config_hash),
        "initial behavior identity header differs",
    )
    _require(value.get("trainer_source") == _json_copy(trainer_source),
             "initial behavior identity source closure differs")
    _require(value.get("parent") == _json_copy(parent_identity),
             "initial behavior identity parent differs")
    _require(value.get("exact_behaviour_identity")
             == _json_copy(exact_behaviour_identity),
             "initial behavior identity all-atom authentication differs")
    _require(value.get("strict_determinism") == STRICT_OUTCOME_DETERMINISM,
             "initial behavior identity lacks strict determinism evidence")
    rows = list(value.get("ranks") or ())
    _require(len(rows) == world,
             "initial behavior identity rank evidence differs")
    validated_rows = [
        _validate_initial_behavior_rank(row, expected_rank=rank)
        for rank, row in enumerate(rows)
    ]
    geometry = dict(value.get("proposal_scoring") or {})
    _require(
        all(dict(row["proposal_scoring"]) == geometry for row in validated_rows),
        "initial behavior identity scoring geometry differs",
    )
    _validate_start_checkpoint_identity(
        dict(value.get("start_checkpoint_identity") or {})
    )
    return _json_copy(value)


def _require_start_step_checkpoint_identity(
    proposal: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    global_step: int,
    parent_proposal: Mapping[str, Any],
) -> dict[str, Any]:
    """A START-step resume may not replace the deeply-authenticated behaviour."""
    if int(global_step) != START_STEP:
        return {"checked": False, "global_step": int(global_step)}
    current = proposal_module_digest(proposal.state_dict())
    expected = _json_copy(parent_proposal)
    _require(
        current == expected,
        "START-step checkpoint proposal differs from authenticated parent proposal",
    )
    _require(
        len(optimizer.state) == 0,
        "START-step checkpoint restored non-empty proposal optimizer state",
    )
    return {
        "checked": True,
        "passed": True,
        "global_step": START_STEP,
        "proposal": current,
        "optimizer_state_entries": 0,
        "optimizer_reset": {"count": 1, "modules": ["proposal"]},
    }


@torch.no_grad()
def evaluate_trust_gates(
    proposal: nn.Module,
    collection: ValidatedRecoveryCollection,
    *,
    device: torch.device,
    chunk_replans: int,
    training_nonfinite: int = 0,
    unexpected_gradients: Sequence[str] = (),
    stop_check: Any | None = None,
) -> dict[str, Any]:
    """Evaluate the locked final-policy trust envelope on all authenticated atoms."""
    proposal.eval()
    dtype = next(proposal.parameters()).dtype
    ratio_sum = 0.0
    ratio_square_sum = 0.0
    ratio_atoms = 0
    clipped_atoms = 0
    drift: list[Tensor] = []
    usage = torch.zeros(C.M, dtype=torch.float64)
    usage_atoms = 0
    final_nonfinite = 0
    max_abs_logratio = 0.0

    for group_index in range(len(collection.receipts)):
        if stop_check is not None and bool(stop_check()):
            raise _PreemptRequested(
                f"preemption requested during trust evaluation at group {group_index}"
            )
        payload = collection.load(group_index)
        for arm_index, arm in enumerate(payload["arms"]):
            n = int(arm["z"].shape[0])
            lang = _batched_lang(arm["lang"], n, device, dtype)
            for lo in range(0, n, chunk_replans):
                hi = min(lo + chunk_replans, n)
                z = arm["z"][lo:hi].to(device=device, dtype=dtype, non_blocking=True)
                current_order = arm["ordered_support"][lo:hi].to(device=device)
                if arm_index == 0:
                    logits = _rowwise_proposal_logits_fp32(
                        proposal, z, lang[lo:hi],
                    )
                    coeff = argmax_coeff(logits.float(), C.TOPK, C.M).float()
                    baseline = arm["coeff"][lo:hi].to(device=device).float()
                    if not bool(torch.isfinite(coeff).all()):
                        final_nonfinite += int((~torch.isfinite(coeff)).sum())
                    else:
                        drift.append((coeff - baseline).norm(p=2, dim=-1).cpu())
                        usage += coeff.double().sum(0).cpu()
                        usage_atoms += int(coeff.shape[0])
                    continue

                current, _ = stored_order_logprob(
                    proposal, z, lang[lo:hi], current_order,
                )
                old = arm["old_logprob"][lo:hi].to(device=device).float()
                delta = current.float() - old
                max_abs_logratio = max(
                    max_abs_logratio, float(delta.abs().max()),
                )
                ratio = _recovery_importance_ratio_fp32(current, old)
                finite = torch.isfinite(ratio)
                if not bool(finite.all()):
                    final_nonfinite += int((~finite).sum())
                    continue
                ratio_sum += float(ratio.sum())
                ratio_square_sum += float(ratio.square().sum())
                ratio_atoms += int(ratio.numel())
                clipped_atoms += int(((ratio < 1.0 - CLIP_EPS)
                                      | (ratio > 1.0 + CLIP_EPS)).sum())

    collection.assert_unchanged()
    if ratio_atoms == 0 or usage_atoms == 0 or not drift:
        raise OutcomeGRPOError("trust evaluation observed no ratio/drift atoms")
    clip_fraction = clipped_atoms / ratio_atoms
    ess = (ratio_sum * ratio_sum) / max(ratio_square_sum, torch.finfo(torch.float64).tiny)
    ess_fraction = ess / ratio_atoms
    drift_all = torch.cat(drift).float()
    drift_p95 = float(torch.quantile(drift_all, 0.95, interpolation="linear"))
    mean_usage = usage / usage_atoms
    live_ops = int((mean_usage > 1e-4).sum())
    total_nonfinite = int(training_nonfinite) + int(final_nonfinite)

    checks = {
        "clip_fraction": {
            "value": clip_fraction, "op": "<=", "threshold": MAX_CLIP_FRACTION,
            "pass": clip_fraction <= MAX_CLIP_FRACTION,
        },
        "ess_fraction": {
            "value": ess_fraction, "op": ">=", "threshold": MIN_ESS_FRACTION,
            "pass": ess_fraction >= MIN_ESS_FRACTION,
        },
        "coeff_drift_p95": {
            "value": drift_p95, "op": "<=", "threshold": MAX_COEFF_DRIFT_P95,
            "pass": drift_p95 <= MAX_COEFF_DRIFT_P95,
        },
        "live_ops": {
            "value": live_ops, "op": ">=", "threshold": MIN_LIVE_OPS,
            "pass": live_ops >= MIN_LIVE_OPS,
        },
        "nonfinite": {
            "value": total_nonfinite, "op": "==", "threshold": 0,
            "pass": total_nonfinite == 0,
        },
        "unexpected_gradients": {
            "value": len(unexpected_gradients), "op": "==", "threshold": 0,
            "pass": len(unexpected_gradients) == 0,
        },
    }
    return {
        "passed": all(bool(row["pass"]) for row in checks.values()),
        "proposal_scoring": _proposal_scoring_geometry(device),
        "max_abs_logratio": max_abs_logratio,
        "checks": checks,
        "definitions": {
            "clip_fraction": "fraction of arm1..7 final/old ratios outside [0.8,1.2]",
            "ess_fraction": "(sum ratio)^2/(N*sum ratio^2), arm1..7 replans",
            "coeff_drift_p95": (
                "torch linear p95 of L2(final deployed argmax coeff - stored arm0 coeff)"
            ),
            "live_ops": "arm0-state mean final deployed coefficient > 1e-4",
            "arm0_importance_ratios": 0,
        },
        "counts": {
            "ratio_atoms": ratio_atoms,
            "clipped_atoms": clipped_atoms,
            "arm0_drift_atoms": int(drift_all.numel()),
            "arm0_usage_atoms": usage_atoms,
            "training_nonfinite": int(training_nonfinite),
            "final_nonfinite": int(final_nonfinite),
        },
    }


def _validation_task_label(
    collection: Any,
    index: int,
    payload: Mapping[str, Any],
) -> str:
    items = getattr(collection, "items", ())
    if index < len(items):
        item = items[index]
        if hasattr(item, "suite") and hasattr(item, "task_id"):
            return f"{item.suite}/task={int(item.task_id):02d}"
    work = dict(payload.get("work_item") or {})
    if "suite" in work and "task_id" in work:
        return f"{work['suite']}/task={int(work['task_id']):02d}"
    if "task" in payload:
        return str(payload["task"])
    raise OutcomeGRPOError(
        f"validation group {payload.get('group_id', index)!r} has no task identity"
    )


@torch.no_grad()
def evaluate_validation_surrogate(
    proposal: nn.Module,
    collection: ValidatedRecoveryCollection,
    *,
    device: torch.device,
    chunk_replans: int,
    stop_check: Any | None = None,
) -> dict[str, Any]:
    """Exact heldout group→arm→replan clipped surrogate on every group."""
    _require(chunk_replans > 0, "validation chunk_replans must be positive")
    proposal.eval()
    dtype = next(proposal.parameters()).dtype
    group_rows: list[dict[str, Any]] = []
    ratio_atoms = 0
    max_abs_logratio = 0.0
    for group_index in range(len(collection.receipts)):
        if stop_check is not None and bool(stop_check()):
            raise _PreemptRequested(
                f"preemption requested during heldout snapshot at group {group_index}"
            )
        payload = collection.load(group_index)
        arms = list(payload["arms"])
        rewards = torch.tensor(
            [float(arm["terminal_reward"]) for arm in arms], dtype=torch.float32,
        )
        advantages = normalised_group_advantages(rewards).to(device)
        informative = len(set(float(value) for value in rewards.tolist())) > 1
        arm_surrogates: list[Tensor] = []
        arm_kls: list[Tensor] = []
        for arm_index in range(1, recovery.GROUP_SIZE):
            arm = arms[arm_index]
            n = int(arm["z"].shape[0])
            _require(n > 0, f"validation arm {arm_index} has zero replans")
            lang = _batched_lang(arm["lang"], n, device, dtype)
            objectives: list[Tensor] = []
            kls: list[Tensor] = []
            for lo in range(0, n, chunk_replans):
                hi = min(lo + chunk_replans, n)
                z = arm["z"][lo:hi].to(
                    device=device, dtype=dtype, non_blocking=True,
                )
                order = arm["ordered_support"][lo:hi].to(device=device)
                current, _ = stored_order_logprob(
                    proposal, z, lang[lo:hi], order,
                )
                old = arm["old_logprob"][lo:hi].to(device=device).float()
                objective, ratio, _ = clipped_grpo_objective(
                    current, old, advantages[arm_index], clip_eps=CLIP_EPS,
                )
                log_ratio = current.float() - old
                max_abs_logratio = max(
                    max_abs_logratio, float(log_ratio.abs().max()),
                )
                approx_kl = (ratio.float() - 1.0) - log_ratio
                _require(bool(torch.isfinite(objective).all())
                         and bool(torch.isfinite(approx_kl).all()),
                         f"nonfinite validation statistic in group {group_index} "
                         f"arm {arm_index}")
                objectives.append(objective.detach().double().cpu())
                kls.append(approx_kl.detach().double().cpu())
                ratio_atoms += int(ratio.numel())
            arm_surrogates.append(torch.cat(objectives).mean())
            arm_kls.append(torch.cat(kls).mean())
        # Multiplication by an exact zero advantage is mathematically zero, but
        # spelling this branch out makes the all-equal validation contract
        # bit-exact even if a future ratio kernel produces signed zeros.
        group_surrogate = (
            0.0 if not informative
            else float(torch.stack(arm_surrogates).mean())
        )
        group_kl = float(torch.stack(arm_kls).mean())
        group_rows.append({
            "index": group_index,
            "group_id": str(payload.get("group_id") or group_index),
            "task": _validation_task_label(collection, group_index, payload),
            "terminal_rewards": [int(value) for value in rewards.tolist()],
            "informative": informative,
            "n_replans_by_arm": [int(arm["z"].shape[0]) for arm in arms],
            "surrogate": group_surrogate,
            "approx_kl": group_kl,
        })
    collection.assert_unchanged()
    _require(group_rows, "heldout convergence evaluation saw no groups")
    tasks = sorted({str(row["task"]) for row in group_rows})
    informative_groups = sum(bool(row["informative"]) for row in group_rows)
    return {
        "definition": (
            "mean_group(mean_arm1..7(mean_replan(min(r*A,clip(r,.8,1.2)*A))))"
        ),
        "aggregation": "equal group -> equal arm1..7 -> equal replans",
        "all_equal_groups": "included with exact zero surrogate",
        "n_groups": len(group_rows),
        "n_tasks": len(tasks),
        "informative_groups": informative_groups,
        "uninformative_groups": len(group_rows) - informative_groups,
        "ratio_atoms": ratio_atoms,
        "max_abs_logratio": max_abs_logratio,
        "proposal_scoring": _proposal_scoring_geometry(device),
        "mean_surrogate": sum(float(row["surrogate"]) for row in group_rows)
        / len(group_rows),
        "mean_approx_kl": sum(float(row["approx_kl"]) for row in group_rows)
        / len(group_rows),
        "groups": group_rows,
    }


def task_stratified_paired_bootstrap(
    candidate: Sequence[float] | Tensor,
    reference: Sequence[float] | Tensor,
    task_ids: Sequence[str],
    *,
    samples: int = CONVERGENCE_BOOTSTRAP_SAMPLES,
    confidence: float = CONVERGENCE_CONFIDENCE,
    seed: int = CONVERGENCE_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Paired percentile CI, resampling complete groups within each task."""
    x = torch.as_tensor(candidate, dtype=torch.float64).reshape(-1)
    y = torch.as_tensor(reference, dtype=torch.float64).reshape(-1)
    if x.shape != y.shape or x.numel() != len(task_ids) or x.numel() == 0:
        raise ValueError("paired bootstrap values/tasks have incompatible lengths")
    if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all()):
        raise ValueError("paired bootstrap values contain nan/inf")
    if int(samples) <= 0 or not 0.0 < float(confidence) < 1.0:
        raise ValueError("paired bootstrap samples/confidence are invalid")
    task_names = sorted({str(value) for value in task_ids})
    if not task_names:
        raise ValueError("paired bootstrap has no task strata")
    indices = [
        torch.tensor(
            [index for index, value in enumerate(task_ids) if str(value) == task],
            dtype=torch.int64,
        )
        for task in task_names
    ]
    if any(index.numel() == 0 for index in indices):
        raise ValueError("paired bootstrap contains an empty task stratum")
    difference = x - y
    estimate = torch.stack([difference[index].mean() for index in indices]).mean()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    draws = torch.zeros(int(samples), dtype=torch.float64)
    for index in indices:
        choices = torch.randint(
            int(index.numel()),
            (int(samples), int(index.numel())),
            generator=generator,
        )
        draws += difference[index[choices]].mean(dim=1) / len(indices)
    alpha = (1.0 - float(confidence)) / 2.0
    low = torch.quantile(draws, alpha, interpolation="linear")
    high = torch.quantile(draws, 1.0 - alpha, interpolation="linear")
    return {
        "method": "task_stratified_whole_group_paired_percentile",
        "estimate": float(estimate), "ci_low": float(low), "ci_high": float(high),
        "confidence": float(confidence), "samples": int(samples), "seed": int(seed),
        "n_groups": int(x.numel()), "n_tasks": len(task_names),
        "statistical_unit": "complete recovery group",
    }


def _median(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("median of empty values")
    mid = len(ordered) // 2
    return (ordered[mid] if len(ordered) % 2
            else 0.5 * (ordered[mid - 1] + ordered[mid]))


def evaluate_metric_convergence(
    rows: Sequence[Mapping[str, Any]],
    *,
    start_step: int = START_STEP,
    stop_step: int = STOP_STEP,
    block_size: int = CONVERGENCE_CE_BLOCK_SIZE,
    blocks: int = CONVERGENCE_CE_BLOCKS,
    max_relative_range: float = MAX_CE_BLOCK_MEDIAN_RELATIVE_RANGE,
    terminal_ce_floor: float = SPARSE_CE_UNIFORM_FLOOR,
) -> dict[str, Any]:
    """Verify the exact accepted-update ledger and sparse-CE convergence floor."""
    expected_updates = int(stop_step) - int(start_step)
    _require(len(rows) == expected_updates,
             f"accepted-update ledger has {len(rows)} rows, expected {expected_updates}")
    expected_steps = list(range(int(start_step) + 1, int(stop_step) + 1))
    got_steps = [int(row.get("global_step", -1)) for row in rows]
    _require(got_steps == expected_steps,
             "accepted-update ledger is not exactly contiguous")
    _require(int(block_size) > 0 and int(blocks) > 0
             and int(block_size) * int(blocks) <= len(rows),
             "sparse-CE block geometry is invalid")
    tail = list(rows[-int(block_size) * int(blocks):])
    block_rows = [tail[index:index + int(block_size)]
                  for index in range(0, len(tail), int(block_size))]
    medians: list[float] = []
    reports: list[dict[str, Any]] = []
    for values in block_rows:
        ce = [float(row.get("anchor_sparse_ce", float("nan"))) for row in values]
        _require(all(math.isfinite(value) and value >= 0.0 for value in ce),
                 "anchor_sparse_ce ledger contains invalid values")
        median = _median(ce)
        medians.append(median)
        reports.append({
            "first_step": int(values[0]["global_step"]),
            "last_step": int(values[-1]["global_step"]),
            "n_updates": len(values), "median": median,
        })
    minimum = min(medians)
    relative_range = (0.0 if max(medians) == minimum == 0.0
                      else float(torch.finfo(torch.float64).max)
                      if minimum <= 0.0
                      else (max(medians) - minimum) / minimum)
    checks = {
        "accepted_updates": {
            "value": len(rows), "op": "==", "threshold": expected_updates,
            "pass": len(rows) == expected_updates,
        },
        "anchor_sparse_ce_block_median_relative_range": {
            "value": relative_range, "op": "<=", "threshold": max_relative_range,
            "pass": relative_range <= float(max_relative_range),
        },
        "anchor_sparse_ce_terminal_block_median": {
            "value": medians[-1], "op": "<", "threshold": terminal_ce_floor,
            "pass": medians[-1] < float(terminal_ce_floor),
        },
    }
    return {
        "passed": all(bool(row["pass"]) for row in checks.values()),
        "checks": checks, "blocks": reports,
        "block_medians": medians, "relative_range": relative_range,
        "relative_range_definition": (
            "(max(block_medians)-min(block_medians))/min(block_medians)"
        ),
    }


def evaluate_convergence_gate(
    snapshots: Mapping[int, Mapping[str, Any]],
    metric_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Combine heldout efficacy/plateau/KL with exact train-ledger CE gates."""
    expected_steps = (START_STEP, *CONVERGENCE_SNAPSHOT_STEPS)
    _require(set(int(step) for step in snapshots) == set(expected_steps),
             "convergence snapshots do not match the predeclared absolute steps")
    ordered = {step: dict(snapshots[step]) for step in expected_steps}
    reference_groups = list(ordered[START_STEP].get("groups") or ())
    _require(len(reference_groups) == EXPECTED_VALIDATION_GROUPS,
             "seed convergence snapshot has wrong validation group count")
    identities = [(row.get("group_id"), row.get("task")) for row in reference_groups]
    for step, report in ordered.items():
        groups = list(report.get("groups") or ())
        _require(len(groups) == EXPECTED_VALIDATION_GROUPS,
                 f"snapshot {step} has wrong validation group count")
        _require([(row.get("group_id"), row.get("task")) for row in groups] == identities,
                 f"snapshot {step} validation group order/identity changed")
        _require(int(report.get("n_tasks", -1)) == EXPECTED_VALIDATION_TASKS,
                 f"snapshot {step} validation task count changed")
        _require(int(report.get("informative_groups", -1))
                 >= MIN_VALIDATION_INFORMATIVE_GROUPS,
                 f"snapshot {step} lacks the locked informative-group floor")
    tasks = [str(row["task"]) for row in reference_groups]

    def values(step: int) -> list[float]:
        return [float(row["surrogate"]) for row in ordered[step]["groups"]]

    final_step = CONVERGENCE_SNAPSHOT_STEPS[-1]
    efficacy = task_stratified_paired_bootstrap(
        values(final_step), values(START_STEP), tasks,
    )
    efficacy["comparison"] = f"{final_step}-seed_{START_STEP}"
    efficacy["pass"] = efficacy["ci_low"] > 0.0
    plateau: list[dict[str, Any]] = []
    for earlier in CONVERGENCE_SNAPSHOT_STEPS[:-1]:
        row = task_stratified_paired_bootstrap(
            values(final_step), values(earlier), tasks,
        )
        row.update({
            "comparison": f"{final_step}-{earlier}",
            "equivalence_low": -CONVERGENCE_PLATEAU_MARGIN,
            "equivalence_high": CONVERGENCE_PLATEAU_MARGIN,
            "pass": (
                row["ci_low"] >= -CONVERGENCE_PLATEAU_MARGIN
                and row["ci_high"] <= CONVERGENCE_PLATEAU_MARGIN
            ),
        })
        plateau.append(row)
    final_kl = float(ordered[final_step]["mean_approx_kl"])
    metrics = evaluate_metric_convergence(metric_rows)
    checks = {
        "heldout_efficacy": {
            "value": efficacy["ci_low"], "op": ">", "threshold": 0.0,
            "pass": bool(efficacy["pass"]),
        },
        "heldout_plateau_all_snapshots": {
            "value": sum(bool(row["pass"]) for row in plateau), "op": "==",
            "threshold": len(plateau), "pass": all(bool(row["pass"]) for row in plateau),
        },
        "final_approx_kl": {
            "value": final_kl, "op": "<=", "threshold": MAX_APPROX_KL,
            "pass": math.isfinite(final_kl) and final_kl <= MAX_APPROX_KL,
        },
        **metrics["checks"],
    }
    return {
        "status": "PASS" if all(bool(row["pass"]) for row in checks.values()) else "FAIL",
        "passed": all(bool(row["pass"]) for row in checks.values()),
        "checks": checks,
        "efficacy": efficacy,
        "plateau": plateau,
        "training_metrics": metrics,
        "snapshots": {str(step): ordered[step] for step in expected_steps},
        "definitions": {
            "efficacy": "paired final-minus-seed 95% CI lower bound > 0",
            "plateau": "each paired final-minus-earlier 95% CI inside [-0.01,+0.01]",
            "bootstrap": "task-stratified paired complete-group percentile bootstrap",
            "approx_kl": "mean_group(mean_arm(mean_replan((r-1)-log(r))))",
        },
    }


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite descendant checkpoint: {path}")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        fsync_dir(path.parent)
        parent_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _persist_terminal_failure(
    run_dir: Path,
    report: Mapping[str, Any],
    *,
    candidate: Path,
) -> None:
    _require(not candidate.exists(),
             "terminal FAIL cannot coexist with an emitted candidate")
    payload = _json_copy(report)
    _require(payload.get("status") == "FAIL" and payload.get("passed") is False,
             "terminal failure report must be explicit FAIL/passed=false")
    _require(payload.get("candidate_emitted") is False,
             "terminal failure report must declare candidate_emitted=false")
    atomic_write_text(
        run_dir / "terminal_report.json",
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _trainer_source_identity(
    root: str | os.PathLike[str] = _ROOT,
    files: Sequence[str] = _TRAINER_SOURCE_FILES,
) -> dict[str, Any]:
    source_root = Path(root).expanduser().resolve()
    names = tuple(sorted(str(value) for value in files))
    _require(bool(names), "trainer source closure is empty")
    _require(len(names) == len(set(names)), "trainer source closure has duplicates")
    h = hashlib.sha256()
    rows: dict[str, str] = {}
    for rel in names:
        relative = Path(rel)
        _require(
            rel == relative.as_posix()
            and not relative.is_absolute()
            and ".." not in relative.parts,
            f"invalid trainer source path: {rel!r}",
        )
        path = (source_root / relative).resolve()
        try:
            path.relative_to(source_root)
        except ValueError as exc:
            raise OutcomeGRPOError(
                f"trainer source escapes repository: {rel}"
            ) from exc
        _require(path.is_file(), f"trainer source is missing: {path}")
        digest = recovery.sha256_file(path)
        rows[rel] = digest
        h.update(rel.encode("utf-8") + b"\0" + bytes.fromhex(digest) + b"\0")
    return {"scheme": "sha256(path-nul-sha256-nul)-v1", "sha256": h.hexdigest(),
            "files": rows}


def _assert_trainer_source_identity(
    expected: Mapping[str, Any],
    *,
    root: str | os.PathLike[str] = _ROOT,
    files: Sequence[str] = _TRAINER_SOURCE_FILES,
) -> None:
    _require(
        _trainer_source_identity(root, files) == dict(expected),
        "outcome trainer source closure changed during or between links",
    )


def _assert_seed_stat(identity: Mapping[str, Any]) -> None:
    """Detect replacement of the authenticated parent before/after loading."""
    path = Path(str(identity.get("path") or ""))
    _require(path.is_file(), f"authenticated seed checkpoint disappeared: {path}")
    stat = path.stat()
    _require(
        int(stat.st_size) == int(identity.get("size", -1)),
        "authenticated seed checkpoint size changed",
    )
    _require(
        int(stat.st_mtime_ns) == int(identity.get("mtime_ns", -1)),
        "authenticated seed checkpoint mtime changed",
    )


def write_descendant_checkpoint(
    out: str | os.PathLike[str],
    *,
    parent: Mapping[str, Any],
    parent_identity: Mapping[str, Any],
    proposal: nn.Module,
    optimizer: torch.optim.Optimizer,
    optimizer_steps: int,
    provenance: Mapping[str, Any],
    global_step: int | None = None,
    resolved_config: Mapping[str, Any] | None = None,
    samples_seen: int | None = None,
    wandb_run_id: str | None = None,
    verify_policy: bool = False,
) -> dict[str, Any]:
    """Atomically write and reload-verify an eval-compatible descendant."""
    out_path = Path(out).expanduser().resolve()
    parent_state = parent["model"]
    frozen_before = frozen_model_digest(parent_state)
    initial_proposal = proposal_model_digest(parent_state)
    trained = {name: value.detach().cpu().clone()
               for name, value in proposal.state_dict().items()}
    state = dict(parent_state)
    expected_keys = {name[len("proposal."):]
                     for name in state if name.startswith("proposal.")}
    _require(set(trained) == expected_keys,
             "trained proposal state keys differ from parent checkpoint")
    for name, value in trained.items():
        key = f"proposal.{name}"
        _require(value.shape == state[key].shape,
                 f"trained proposal tensor shape changed for {key}")
        state[key] = value
    frozen_candidate = frozen_model_digest(state)
    _require(frozen_candidate == frozen_before,
             "non-proposal checkpoint tensors changed before save")
    final_proposal = proposal_model_digest(state)

    if resolved_config is None:
        # Backward-compatible unit seam; production always supplies the exact
        # config-hashed six-fold recipe.
        resolved = copy.deepcopy(parent["resolved_config"])
        resolved["outcome_grpo"] = {
            "format_version": FORMAT_VERSION,
            "algorithm": "stored_order_pl_clipped_grpo",
            "clip_eps": CLIP_EPS,
            "optimizer_steps": int(optimizer_steps),
            "collection_identity_digest": provenance["collection"]["identity_digest"],
            "collection_split": provenance["collection"]["split"],
        }
    else:
        resolved = copy.deepcopy(dict(resolved_config))
        _require(int(optimizer_steps) == EXPECTED_ACCEPTED_UPDATES,
                 "production descendant requires exactly 4800 accepted updates")
        _require(int(global_step if global_step is not None else -1) == STOP_STEP,
                 "production descendant requires the exact terminal global step")
        _require(bool(dict(provenance.get("convergence_gate") or {}).get("passed")),
                 "production descendant requires convergence_gate PASS evidence")
        _require(bool(dict(provenance.get("trust_gate") or {}).get("passed")),
                 "production descendant requires trust_gate PASS evidence")
    descendant_hash = _config_hash(resolved)
    descendant_step = (int(parent["global_step"]) + int(optimizer_steps)
                       if global_step is None else int(global_step))
    _require(descendant_step == int(parent["global_step"]) + int(optimizer_steps),
             "descendant global step does not equal seed step plus optimizer updates")
    top_provenance = _json_copy(provenance)
    top_provenance.update({
        "format_version": FORMAT_VERSION,
        "kind": TRAINER_KIND,
        "created_utc": _utc(),
        "parent": dict(parent_identity),
        "parent_config_hash": parent["config_hash"],
        "descendant_config_hash": descendant_hash,
        "parent_global_step": int(parent["global_step"]),
        "descendant_global_step": descendant_step,
        "optimizer_steps": int(optimizer_steps),
        "mutated_model_prefixes": ["proposal."],
        "frozen_model": frozen_before,
        "initial_proposal": initial_proposal,
        "final_proposal": final_proposal,
    })

    payload = dict(parent)
    payload["model"] = state
    payload["global_step"] = descendant_step
    if samples_seen is not None:
        payload["samples_seen"] = int(samples_seen)
    payload["config_hash"] = descendant_hash
    payload["resolved_config"] = resolved
    if wandb_run_id:
        payload["wandb_run_id"] = str(wandb_run_id)
    payload["world_size"] = int(provenance.get("world_size", 1))
    payload["stop_reason"] = "terminal_outcome_grpo"
    payload["optimizer"] = {
        "kind": "proposal_only_adamw",
        "parameter_names": [name for name, _ in proposal.named_parameters()],
        "state_dict": _cpu_tree(optimizer.state_dict()),
        "state_reset_at_entry": True,
    }
    payload["outcome_grpo"] = top_provenance
    payload["consolidated"] = {
        "tool": "loom.train.outcome_grpo",
        "step": descendant_step,
        "created_unix": time.time(),
        "parent_checkpoint": {
            "path": str(parent_identity.get("path") or ""),
            "sha256": str(parent_identity.get("sha256") or ""),
            "global_step": int(parent_identity.get("global_step", parent["global_step"])),
            "config_hash": str(parent_identity.get("config_hash", parent["config_hash"])),
        },
        "derivation": "proposal-only terminal outcome GRPO",
        "mutated_model_prefixes": ["proposal."],
        "not_resumable": (
            "eval artifact; resume from the outcome trainer checkpoint named by LATEST"
        ),
    }
    _atomic_torch_save(out_path, payload)

    try:
        reloaded = torch.load(out_path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        raise OutcomeGRPOError(f"cannot reload descendant checkpoint {out_path}: {exc}") from exc
    _require(isinstance(reloaded, dict), "reloaded descendant is not a mapping")
    _require(_config_hash(reloaded["resolved_config"]) == reloaded["config_hash"],
             "reloaded descendant resolved_config is unauthenticated")
    frozen_after = frozen_model_digest(reloaded["model"])
    proposal_after = proposal_model_digest(reloaded["model"])
    _require(frozen_after == frozen_before,
             "frozen checkpoint tensors are not byte-identical after save/reload")
    _require(proposal_after == final_proposal,
             "trained proposal tensors changed during save/reload")
    _require(reloaded["optimizer"]["state_reset_at_entry"] is True,
             "descendant optimizer reset provenance is absent")
    _require(int(reloaded["consolidated"]["step"]) == descendant_step,
             "descendant consolidated.step is stale")
    _require(int(reloaded["global_step"]) == descendant_step,
             "descendant top-level global_step is stale")
    try:
        safe = torch.load(out_path, map_location="cpu", weights_only=True, mmap=True)
    except Exception as exc:  # noqa: BLE001
        raise OutcomeGRPOError(
            f"descendant is not weights_only-loadable like a consolidated artifact: {exc}"
        ) from exc
    _require(set(safe["model"]) == set(parent_state),
             "descendant model key set differs from consolidated parent")
    verification: dict[str, Any] = {
        "weights_only": True,
        "model_keys": len(safe["model"]),
        "consolidated_step": int(safe["consolidated"]["step"]),
    }
    del safe
    if verify_policy:
        from loom.eval.policy import load_policy, policy_provenance  # noqa: PLC0415

        embodiments = list(resolved.get("data", {}).get("embodiments") or ())
        _require(len(embodiments) == 1,
                 "real load_policy verification requires one configured embodiment")
        policy = load_policy(
            str(out_path), embodiment=str(embodiments[0]), device="cpu",
            allow_stub=False, _include_q_action=False,
        )
        loaded = policy_provenance(policy)
        _require(loaded.get("is_stub") is False,
                 "real load_policy verification fell back to stubs")
        _require(int(loaded.get("ckpt_global_step", -1)) == descendant_step,
                 "real load_policy saw a stale descendant step")
        _require(str(loaded.get("ckpt_config_hash") or "") == descendant_hash,
                 "real load_policy saw a stale descendant config hash")
        verification["load_policy"] = {
            "is_stub": False,
            "global_step": int(loaded["ckpt_global_step"]),
            "config_hash": str(loaded["ckpt_config_hash"]),
            "state_dict": _json_copy(loaded.get("state_dict") or {}),
        }
    return {
        "path": str(out_path),
        "sha256": recovery.sha256_file(out_path),
        "size": int(out_path.stat().st_size),
        "global_step": int(reloaded["global_step"]),
        "config_hash": str(reloaded["config_hash"]),
        "optimizer_steps": int(optimizer_steps),
        "frozen_model": frozen_after,
        "proposal": proposal_after,
        "verification": verification,
    }


def _resolve_repo_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (_ROOT / path).resolve()


def validate_recipe_config(cfg: Mapping[str, Any]) -> None:
    """Fail closed unless the resolved config is the audited fixed recipe."""
    run = dict(cfg.get("run", {}) or {})
    optim = dict(cfg.get("optim", {}) or {})
    losses = dict(cfg.get("losses", {}) or {})
    data = dict(cfg.get("data", {}) or {})
    outcome = dict(cfg.get("outcome_grpo", {}) or {})

    def exact(label: str, actual: Any, expected: Any) -> None:
        _require(actual == expected,
                 f"{label} differs from locked recipe: {actual!r} != {expected!r}")

    exact("top-level config fields", set(cfg) - {"link"}, {
        "convergence", "convergence_gate", "data", "freeze", "fsdp", "losses",
        "model", "optim", "outcome_grpo", "promotion_gate", "run", "slurm",
        "train_modules", "validation_gate",
    })
    exact("run", run, {
        "name": "r0a_outcome_grpo", "project": "loom", "seed": TRAIN_SEED,
        "steps": SCHEDULE_STEPS, "deterministic": True,
        "log_every": LOG_EVERY, "ckpt_every": CKPT_EVERY, "keep_last": 0,
        "wandb_mode": "online",
    })
    _require(list(cfg.get("train_modules") or ()) == ["proposal"],
             "train_modules must be exactly [proposal]")
    exact("data", data, {
        "source": "libero", "embodiments": ["libero_franka"],
        "batch_per_gpu": EXPECTED_BATCH_PER_GPU, "action_free": False,
        "sampling": "uniform_task", "trajectory_split": "train",
        "holdout_demo_keys": ["demo_49"], "recurrent_burn_in": 4,
        "cache_dir": "cache/", "num_workers": 4, "pin_memory": True,
        "prefetch_factor": 2,
    })
    exact("fsdp", dict(cfg.get("fsdp", {}) or {}), {
        "shard": [], "replicate": ["proposal"],
        "activation_checkpointing": False,
        "block_names": ["PerceiverBlock", "EstimatorBlock", "Block"],
        "forward_prefetch": True, "limit_all_gathers": True,
    })
    exact("slurm", dict(cfg.get("slurm", {}) or {}), {
        "nodes": 1, "gpus_per_node": EXPECTED_WORLD_SIZE, "n_links": 1,
    })
    exact("model", dict(cfg.get("model", {}) or {}), {
        "bank": {}, "decoder": {},
        "estimator": {"learned_z_init": True, "z_prev_residual": False},
        "potential": {}, "proposal": {}, "q_action": {}, "q_delta": {},
        "use_stubs": "auto",
    })
    exact("freeze", dict(cfg.get("freeze", {}) or {}), {
        "modules": [], "until_frac": 0.0,
    })
    exact("legacy convergence", dict(cfg.get("convergence", {}) or {}), {
        "block": 2000, "blocks": 4,
        "floor_checks": ["loss/proposal"],
        "primary": ["act/decode", "loss/proposal"],
        "start_step": 27000, "tol": 0.02,
        "watch": ["act/deploy_c_l2", "act/deploy_topk_overlap", "grad_norm"],
    })
    exact("outcome_grpo", outcome, {
        "format_version": FORMAT_VERSION,
        "seed_checkpoint": EXPECTED_SEED_CHECKPOINT,
        "seed_global_step": START_STEP,
        "seed_config_hash": recovery.SEED_CONFIG_HASH,
        "seed_checkpoint_sha256": recovery.SEED_CHECKPOINT_SHA256,
        "start_step": START_STEP, "stop_step": STOP_STEP,
        "updates_per_fold": UPDATES_PER_FOLD,
        "schedule_steps": SCHEDULE_STEPS,
        "world_size": EXPECTED_WORLD_SIZE,
        "contexts_per_arm": EXPECTED_CONTEXTS_PER_ARM,
        "groups_per_train_fold": 200,
        "validation_groups": EXPECTED_VALIDATION_GROUPS,
        "validation_tasks": EXPECTED_VALIDATION_TASKS,
        "informative_training_groups": "only",
        "minimum_informative_groups_per_fold": MIN_TRAIN_INFORMATIVE_GROUPS,
        "minimum_validation_informative_groups": MIN_VALIDATION_INFORMATIVE_GROUPS,
        "authentication": {
            "chunk_replans": AUTH_CHUNK_REPLANS,
            "proposal_scoring_batch_size": PROPOSAL_SCORING_BATCH_SIZE,
            "proposal_scoring_dtype": PROPOSAL_SCORING_DTYPE,
            "proposal_scoring_autocast": PROPOSAL_SCORING_AUTOCAST,
            "cuda_matmul_tf32": CUDA_MATMUL_TF32,
            "cudnn_tf32": CUDNN_TF32,
            "float32_matmul_precision": FLOAT32_MATMUL_PRECISION,
            "proposal_scoring_module_mode": PROPOSAL_SCORING_MODULE_MODE,
            "behaviour_logprob_atol": BEHAVIOUR_LOGPROB_ATOL,
            "behaviour_logprob_rtol": BEHAVIOUR_LOGPROB_RTOL,
            "behaviour_coeff_atol": BEHAVIOUR_COEFF_ATOL,
            "behaviour_coeff_rtol": BEHAVIOUR_COEFF_RTOL,
            "identity_max_abs_logratio": BEHAVIOUR_IDENTITY_MAX_ABS_LOGRATIO,
            "identity_max_coeff_error": BEHAVIOUR_IDENTITY_MAX_COEFF_ERROR,
            "initial_ratio_min_ess_fraction": INITIAL_RATIO_MIN_ESS_FRACTION,
        },
        "folds": [dict(row) for row in EXPECTED_FOLDS],
        "validation": dict(EXPECTED_VALIDATION),
        "anchor_manifest": dict(EXPECTED_ANCHOR_MANIFEST),
    })
    exact("optim", optim, {
        "lr": BASE_LEARNING_RATE, "warmup": WARMUP_STEPS,
        "min_lr_ratio": MIN_LR_RATIO, "grad_clip": GRAD_CLIP,
        "betas": list(ADAMW_BETAS), "weight_decay": ADAMW_WEIGHT_DECAY,
        "eps": ADAMW_EPS,
        "lr_scales": {
            "bank": 0.0, "decoder": 0.3, "ema": 0.0, "estimator": 0.0,
            "proposal": PROPOSAL_LR_SCALE, "q_action": 0.0, "q_delta": 0.0,
        },
        "reset_state_modules": ["proposal"], "update_ema": False,
        "bank_lr_mult": 0.1, "ema_tau": 0.996, "spike_mult": 10.0,
    })
    exact("losses", losses, {
        "act": {
            "align_to": "q_a", "decode_from": "proposal",
            "enabled": False, "weight": 0.0,
        },
        "balance": {
            "enabled": True, "source": "proposal",
            "weight": SWITCH_BALANCE_WEIGHT,
        },
        "dyn": {
            "cosine": "per_slot", "enabled": False, "min_gap": 2,
            "neg_margin": 0.1, "neg_weight": 1.0,
            "negatives": "within_trajectory", "weight": 0.0,
        },
        "grpo": {
            "enabled": True, "weight": 1.0, "group": recovery.GROUP_SIZE,
            "clip_eps": CLIP_EPS, "reward": "terminal_libero_success_only",
            "baseline_arms": list(range(8)), "ratio_arms": list(range(1, 8)),
        },
        "potential": {"enabled": False, "weight": 0.0},
        "proposal": {
            "enabled": True, "weight": 1.0, "mode": "sparse_ce",
            "temperature": 1.0, "detach_belief": True,
        },
    })
    grpo_cfg = dict(losses.get("grpo") or {})
    exact("losses.grpo", grpo_cfg, {
        "enabled": True, "weight": 1.0, "group": recovery.GROUP_SIZE,
        "clip_eps": CLIP_EPS, "reward": "terminal_libero_success_only",
        "baseline_arms": list(range(8)), "ratio_arms": list(range(1, 8)),
    })
    proposal_cfg = dict(losses.get("proposal") or {})
    exact("losses.proposal", proposal_cfg, {
        "enabled": True, "weight": 1.0, "mode": "sparse_ce",
        "temperature": 1.0, "detach_belief": True,
    })
    balance = dict(losses.get("balance") or {})
    exact("losses.balance", balance, {
        "enabled": True, "weight": SWITCH_BALANCE_WEIGHT, "source": "proposal",
    })
    exact("validation_gate", dict(cfg.get("validation_gate", {}) or {}), {
        "required": True, "max_clip_fraction": MAX_CLIP_FRACTION,
        "min_ess_fraction": MIN_ESS_FRACTION,
        "max_coeff_drift_p95_l2": MAX_COEFF_DRIFT_P95,
        "min_live_ops": MIN_LIVE_OPS,
        "max_topk_overlap_decline": MAX_TOPK_OVERLAP_DECLINE,
        "expert_batches": EXPERT_GATE_BATCHES,
        "nonfinite": 0, "unexpected_module_gradients": False,
    })
    exact("convergence_gate", dict(cfg.get("convergence_gate", {}) or {}), {
        "required": True,
        "terminal_parallelism": TERMINAL_PARALLELISM,
        "metric": "heldout_clipped_surrogate",
        "aggregation": "group_arm_replan",
        "constant_reward_groups": "exact_zero_included",
        "seed_snapshot_step": START_STEP,
        "snapshot_steps": list(CONVERGENCE_SNAPSHOT_STEPS),
        "validation_groups": EXPECTED_VALIDATION_GROUPS,
        "validation_tasks": EXPECTED_VALIDATION_TASKS,
        "bootstrap": {
            "method": "task_stratified_whole_group_paired_percentile",
            "samples": CONVERGENCE_BOOTSTRAP_SAMPLES,
            "seed": CONVERGENCE_BOOTSTRAP_SEED,
            "confidence": CONVERGENCE_CONFIDENCE,
        },
        "efficacy": {
            "comparison": "final_minus_seed",
            "ci_lower_strictly_greater_than": 0.0,
        },
        "plateau": {
            "comparison": "final_minus_each_earlier_snapshot",
            "equivalence_low": -CONVERGENCE_PLATEAU_MARGIN,
            "equivalence_high": CONVERGENCE_PLATEAU_MARGIN,
        },
        "final_approx_kl_max": MAX_APPROX_KL,
        "accepted_updates": EXPECTED_ACCEPTED_UPDATES,
        "metrics_first_step": START_STEP + 1,
        "metrics_last_step": STOP_STEP,
        "anchor_sparse_ce": {
            "block_size": CONVERGENCE_CE_BLOCK_SIZE,
            "blocks": CONVERGENCE_CE_BLOCKS,
            "block_median_relative_range_max": (
                MAX_CE_BLOCK_MEDIAN_RELATIVE_RANGE
            ),
            "relative_range_denominator": "minimum_block_median",
            "terminal_block_median_strictly_less_than": SPARSE_CE_UNIFORM_FLOOR,
        },
    })
    exact("promotion_gate", dict(cfg.get("promotion_gate", {}) or {}), {
        "required": True, "official_seed": 0, "candidate_successes_min": 164,
        "episodes": 400,
        "paired_improvement_over_reference_required": True,
        "reference_successes": 149,
    })


class _ProposalOnly(nn.Module):
    def __init__(self, proposal: nn.Module) -> None:
        super().__init__()
        self.proposal = proposal


def _dist_info() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if world > 1:
        import datetime
        import torch.distributed as dist

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank % torch.cuda.device_count())
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                timeout=datetime.timedelta(hours=4),
            )
    device = torch.device("cuda", local_rank % max(1, torch.cuda.device_count())) \
        if torch.cuda.is_available() else torch.device("cpu")
    return rank, world, local_rank, device


def _barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _guard_local_should_stop(guard: PreemptGuard) -> bool:
    """Non-collective startup/terminal poll used inside asymmetric I/O loops."""
    if getattr(guard, "_signalled", False):
        if not guard.reason:
            guard.reason = "signal"
        return True
    if guard.sentinel.exists():
        if not guard.reason:
            guard.reason = "sentinel"
        return True
    if guard.seconds_left <= 0.0:
        if not guard.reason:
            guard.reason = "budget"
        return True
    return False


def _sync_proposal_grads(proposal: nn.Module, world: int) -> float:
    grad_norm, missing, nonfinite = _proposal_grad_health(proposal)
    local_bad = int(bool(missing or nonfinite or not math.isfinite(grad_norm)))
    if world > 1:
        import torch.distributed as dist

        flag = torch.tensor(local_bad, dtype=torch.int32,
                            device=next(proposal.parameters()).device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        local_bad = int(flag.item())
    if local_bad:
        raise OutcomeGRPOError(
            f"proposal gradient failure missing={missing[:8]} nonfinite={nonfinite[:8]}"
        )
    if world > 1:
        import torch.distributed as dist

        for parameter in proposal.parameters():
            _require(parameter.grad is not None, "proposal gradient disappeared before sync")
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(float(world))
    return float(torch.nn.utils.clip_grad_norm_(proposal.parameters(), GRAD_CLIP))


def _mean_across_ranks(metrics: Mapping[str, float], world: int) -> dict[str, float]:
    if world == 1:
        return {key: float(value) for key, value in metrics.items()}
    import torch.distributed as dist

    keys = sorted(metrics)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    values = torch.tensor([float(metrics[key]) for key in keys],
                          dtype=torch.float64, device=device)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values.div_(float(world))
    return {key: float(values[index]) for index, key in enumerate(keys)}


def _max_across_ranks(value: float, world: int, device: torch.device) -> float:
    result = torch.tensor(float(value), dtype=torch.float64, device=device)
    if world > 1:
        import torch.distributed as dist

        dist.all_reduce(result, op=dist.ReduceOp.MAX)
    return float(result)


def _reduce_training_metrics(
    metrics: Mapping[str, float],
    world: int,
    device: torch.device,
) -> dict[str, float]:
    """Mean scalar losses while reducing ratio sufficient statistics exactly."""
    reduced = _mean_across_ranks(metrics, world)
    if world > 1:
        import torch.distributed as dist

        minimum = torch.tensor(float(metrics["ratio_min"]), dtype=torch.float64,
                               device=device)
        maxima = torch.tensor([
            float(metrics["ratio_max"]), float(metrics["max_abs_logratio"]),
        ], dtype=torch.float64, device=device)
        totals = torch.tensor([
            float(metrics["ratio_sum"]),
            float(metrics["ratio_square_sum"]),
            float(metrics["ratio_atoms"]),
            float(metrics["clipped_atoms"]),
        ], dtype=torch.float64, device=device)
        dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
        dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        reduced["ratio_min"] = float(minimum)
        reduced["ratio_max"] = float(maxima[0])
        reduced["max_abs_logratio"] = float(maxima[1])
        reduced["ratio_sum"] = float(totals[0])
        reduced["ratio_square_sum"] = float(totals[1])
        reduced["ratio_atoms"] = float(totals[2])
        reduced["clipped_atoms"] = float(totals[3])
    _require(reduced["ratio_atoms"] > 0.0,
             "distributed training metrics contain no ratio atoms")
    reduced["ratio_mean"] = reduced["ratio_sum"] / reduced["ratio_atoms"]
    reduced["clip_fraction"] = reduced["clipped_atoms"] / reduced["ratio_atoms"]
    reduced["ratio_ess_fraction"] = (
        reduced["ratio_sum"] * reduced["ratio_sum"]
        / max(
            reduced["ratio_atoms"] * reduced["ratio_square_sum"],
            torch.finfo(torch.float64).tiny,
        )
    )
    return reduced


def _raise_if_any_rank_failed(local_error: str, world: int, where: str) -> None:
    errors: list[Any]
    if world > 1:
        errors = [None for _ in range(world)]
        torch.distributed.all_gather_object(errors, str(local_error))
    else:
        errors = [str(local_error)]
    failed = [(rank, error) for rank, error in enumerate(errors) if error]
    if failed:
        raise OutcomeGRPOError(f"{where} failed on rank(s): {failed[:4]}")


def _atomic_replace_torch(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        fsync_dir(path.parent)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _trainer_checkpoint_path(run_dir: Path, step: int) -> Path:
    return run_dir / f"outcome_ckpt_{int(step):09d}.pt"


def _read_complete_metric_rows(path: Path) -> tuple[list[dict[str, Any]], bool]:
    if not path.exists():
        return [], False
    raw = path.read_bytes()
    had_partial_tail = bool(raw and not raw.endswith(b"\n"))
    complete = raw.split(b"\n")[:-1] if had_partial_tail else raw.split(b"\n")
    rows: list[dict[str, Any]] = []
    for line_number, encoded in enumerate(complete, start=1):
        if not encoded.strip():
            continue
        try:
            row = json.loads(encoded)
        except (UnicodeDecodeError, ValueError) as exc:
            raise OutcomeGRPOError(
                f"metrics.jsonl has malformed complete line {line_number}: {exc}"
            ) from exc
        _require(isinstance(row, dict),
                 f"metrics.jsonl line {line_number} is not an object")
        rows.append(row)
    return rows, had_partial_tail


def _reconcile_metrics_to_checkpoint(
    path: Path,
    *,
    checkpoint_step: int,
    config_hash: str,
) -> list[dict[str, Any]]:
    """Atomically discard crash-tail rows and prove a contiguous durable prefix."""
    rows, partial = _read_complete_metric_rows(path)
    retained: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        try:
            step = int(row["global_step"])
        except (KeyError, TypeError, ValueError) as exc:
            raise OutcomeGRPOError(
                f"metrics.jsonl row {index} has no integer global_step"
            ) from exc
        _require(str(row.get("config_hash") or "") == str(config_hash),
                 f"metrics.jsonl row {step} has a different config hash")
        if step <= int(checkpoint_step):
            retained.append(row)
    expected_steps = list(range(START_STEP + 1, int(checkpoint_step) + 1))
    got_steps = [int(row["global_step"]) for row in retained]
    _require(
        got_steps == expected_steps,
        "metrics.jsonl durable prefix is not exactly one contiguous row per "
        f"accepted update through checkpoint {checkpoint_step}",
    )
    rewrite = partial or len(retained) != len(rows)
    if rewrite:
        encoded = "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
            for row in retained
        )
        atomic_write_text(path, encoded)
    return retained


def _exact_terminal_metrics(
    path: Path,
    *,
    config_hash: str,
) -> list[dict[str, Any]]:
    rows, partial = _read_complete_metric_rows(path)
    _require(not partial, "terminal metrics.jsonl has an unterminated tail")
    _require(len(rows) == EXPECTED_ACCEPTED_UPDATES,
             f"terminal metrics contain {len(rows)} accepted updates, expected "
             f"{EXPECTED_ACCEPTED_UPDATES}")
    expected = list(range(START_STEP + 1, STOP_STEP + 1))
    got = [int(row.get("global_step", -1)) for row in rows]
    _require(got == expected,
             "terminal metrics are not the exact contiguous accepted-update ledger")
    _require(all(str(row.get("config_hash") or "") == config_hash for row in rows),
             "terminal metrics config hash differs")
    _require(
        [int(row.get("accepted_update", -1)) for row in rows]
        == list(range(1, EXPECTED_ACCEPTED_UPDATES + 1)),
        "terminal metrics accepted_update ledger is not exact",
    )
    _require(all(float(row.get("grad_skipped", float("nan"))) == 0.0 for row in rows),
             "terminal metrics contain a skipped/non-accepted update")
    return rows


def _durable_metrics_barrier(handle: Any) -> None:
    """Make every accepted row durable before LATEST can advance past it."""
    handle.flush()
    os.fsync(handle.fileno())


def _save_trainer_checkpoint(
    run_dir: Path,
    *,
    global_step: int,
    samples_seen: int,
    config_hash: str,
    resolved_config: Mapping[str, Any],
    parent_identity: Mapping[str, Any],
    collections: Sequence[ValidatedRecoveryCollection],
    validation: ValidatedRecoveryCollection,
    proposal: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineWithWarmup,
    sampler: DeterministicOutcomeSampler,
    wandb_run_id: str,
    exact_behaviour_identity: Mapping[str, Any],
    initial_behavior_identity: Mapping[str, Any] | None,
    trainer_source: Mapping[str, Any] | None = None,
    stop_reason: str = "",
) -> Path:
    source = (_trainer_source_identity() if trainer_source is None
              else _json_copy(trainer_source))
    _assert_trainer_source_identity(source)
    initial_identity = _validate_initial_behavior_identity(
        initial_behavior_identity,
        global_step=global_step,
        world=sampler.world_size,
        config_hash=config_hash,
        trainer_source=source,
        parent_identity=parent_identity,
        exact_behaviour_identity=exact_behaviour_identity,
    )
    reset = {
        "count": 1,
        "modules": ["proposal"],
        "source_global_step": START_STEP,
        "source_config_hash": recovery.SEED_CONFIG_HASH,
        "source_checkpoint_sha256": recovery.SEED_CHECKPOINT_SHA256,
    }
    payload = {
        "format": FORMAT_VERSION,
        "kind": "loom_outcome_grpo_training_checkpoint",
        "global_step": int(global_step),
        "samples_seen": int(samples_seen),
        "config_hash": str(config_hash),
        "resolved_config": _json_copy(resolved_config),
        "world_size": int(sampler.world_size),
        "wandb_run_id": str(wandb_run_id),
        "stop_reason": str(stop_reason),
        "parent": _json_copy(parent_identity),
        "trainer_source": source,
        "exact_behaviour_identity": _json_copy(exact_behaviour_identity),
        "initial_behavior_identity": initial_identity,
        "collections": [_json_copy(item.provenance()) for item in collections],
        "validation": _json_copy(validation.provenance()),
        "proposal": _cpu_tree(proposal.state_dict()),
        "optimizer": _cpu_tree(optimizer.state_dict()),
        "scheduler": scheduler.state_dict(),
        "sampler": sampler.state_dict(global_step),
        "anchor_sampler": {
            "kind": "loom_loader_pure_step",
            "global_step": int(global_step),
            "seed": int(sampler.seed),
            "world_size": int(sampler.world_size),
            "trajectory_manifest": _json_copy(
                dict(resolved_config["outcome_grpo"]["anchor_manifest"])
            ),
        },
        "optimizer_reset": reset,
        "rng": {
            "scheme": "set_step_seed(seed,global_step,rank)",
            "seed": int(sampler.seed),
        },
    }
    path = _trainer_checkpoint_path(run_dir, global_step)
    _atomic_replace_torch(path, payload)
    check = torch.load(path, map_location="cpu", weights_only=True)
    _require(int(check["global_step"]) == int(global_step)
             and check["optimizer_reset"] == reset
             and check["exact_behaviour_identity"]
                 == _json_copy(exact_behaviour_identity)
             and check["initial_behavior_identity"] == initial_identity,
             "trainer checkpoint reload/reset verification failed")
    atomic_write_text(run_dir / "LATEST", f"{int(global_step)}\n")
    return path


def _load_trainer_checkpoint(
    run_dir: Path,
    *,
    config_hash: str,
    resolved_config: Mapping[str, Any],
    parent_identity: Mapping[str, Any],
    collections: Sequence[ValidatedRecoveryCollection],
    validation: ValidatedRecoveryCollection,
    proposal: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineWithWarmup,
    sampler: DeterministicOutcomeSampler,
    exact_behaviour_identity: Mapping[str, Any],
    trainer_source: Mapping[str, Any] | None = None,
) -> tuple[int, int, str, dict[str, Any] | None] | None:
    step = read_pointer(run_dir)
    if step is None:
        return None
    path = _trainer_checkpoint_path(run_dir, step)
    _require(path.is_file(), f"LATEST points at missing trainer checkpoint {path}")
    raw = torch.load(path, map_location="cpu", weights_only=False)
    _require(raw.get("kind") == "loom_outcome_grpo_training_checkpoint",
             "LATEST is not an outcome-GRPO trainer checkpoint")
    _require(str(raw.get("config_hash") or "") == config_hash
             and raw.get("resolved_config") == _json_copy(resolved_config),
             "same-directory resume has a different config hash/config")
    _require(raw.get("parent") == _json_copy(parent_identity),
             "trainer resume parent identity differs")
    source = (_trainer_source_identity() if trainer_source is None
              else _json_copy(trainer_source))
    _assert_trainer_source_identity(source)
    _require(raw.get("trainer_source") == source,
             "trainer resume source closure differs")
    _require(
        raw.get("exact_behaviour_identity")
        == _json_copy(exact_behaviour_identity),
        "trainer resume all-atom behavior identity differs",
    )
    expected_collections = [_json_copy(item.provenance()) for item in collections]
    _require(raw.get("collections") == expected_collections
             and raw.get("validation") == _json_copy(validation.provenance()),
             "trainer resume collection identity differs")
    global_step = int(raw.get("global_step", -1))
    _require(START_STEP <= global_step <= STOP_STEP,
             f"trainer resume step {global_step} is outside the fixed stage")
    expected_reset = {
        "count": 1, "modules": ["proposal"],
        "source_global_step": START_STEP,
        "source_config_hash": recovery.SEED_CONFIG_HASH,
        "source_checkpoint_sha256": recovery.SEED_CHECKPOINT_SHA256,
    }
    _require(raw.get("optimizer_reset") == expected_reset,
             "proposal optimizer reset evidence differs or repeated")
    initial_identity = _validate_initial_behavior_identity(
        raw.get("initial_behavior_identity"),
        global_step=global_step,
        world=sampler.world_size,
        config_hash=config_hash,
        trainer_source=source,
        parent_identity=parent_identity,
        exact_behaviour_identity=exact_behaviour_identity,
    )
    proposal.load_state_dict(raw["proposal"], strict=True)
    optimizer.load_state_dict(raw["optimizer"])
    expected_schedule = scheduler.state_dict()
    _require(raw.get("scheduler") == expected_schedule,
             "resume attempted to change the absolute LR schedule")
    sampler.validate_state_dict(raw["sampler"], global_step)
    _require(raw.get("anchor_sampler") == {
        "kind": "loom_loader_pure_step",
        "global_step": global_step,
        "seed": sampler.seed,
        "world_size": sampler.world_size,
        "trajectory_manifest": _json_copy(
            dict(resolved_config["outcome_grpo"]["anchor_manifest"])
        ),
    }, "anchor sampler resume state differs")
    return (
        global_step,
        int(raw.get("samples_seen", 0)),
        str(raw.get("wandb_run_id") or ""),
        initial_identity,
    )


def _authenticated_snapshot_payload(
    path: Path,
    *,
    step: int,
    config_hash: str,
    resolved_config: Mapping[str, Any],
    parent_identity: Mapping[str, Any],
    collections: Sequence[ValidatedRecoveryCollection],
    validation: ValidatedRecoveryCollection,
    trainer_source: Mapping[str, Any],
    exact_behaviour_identity: Mapping[str, Any],
    initial_behavior_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(path.is_file(), f"required convergence snapshot is missing: {path}")
    before = path.stat()
    digest = recovery.sha256_file(path)
    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001
        raise OutcomeGRPOError(f"cannot load convergence snapshot {path}: {exc}") from exc
    after = path.stat()
    _require(
        int(before.st_size) == int(after.st_size)
        and int(before.st_mtime_ns) == int(after.st_mtime_ns),
        f"convergence snapshot changed while loading: {path}",
    )
    _require(isinstance(raw, dict)
             and raw.get("kind") == "loom_outcome_grpo_training_checkpoint",
             f"convergence snapshot has wrong kind: {path}")
    _require(int(raw.get("global_step", -1)) == int(step),
             f"convergence snapshot embedded step differs: {path}")
    _require(str(raw.get("config_hash") or "") == config_hash
             and raw.get("resolved_config") == _json_copy(resolved_config),
             f"convergence snapshot config differs: {path}")
    _require(raw.get("parent") == _json_copy(parent_identity),
             f"convergence snapshot parent differs: {path}")
    _require(raw.get("trainer_source") == _json_copy(trainer_source),
             f"convergence snapshot source closure differs: {path}")
    _require(
        raw.get("exact_behaviour_identity")
        == _json_copy(exact_behaviour_identity),
        f"convergence snapshot all-atom behavior identity differs: {path}",
    )
    _require(
        _validate_initial_behavior_identity(
            raw.get("initial_behavior_identity"),
            global_step=step,
            world=int(resolved_config["outcome_grpo"]["world_size"]),
            config_hash=config_hash,
            trainer_source=trainer_source,
            parent_identity=parent_identity,
            exact_behaviour_identity=exact_behaviour_identity,
        ) == _json_copy(initial_behavior_identity),
        f"convergence snapshot first-forward identity differs: {path}",
    )
    _require(raw.get("collections") == [item.provenance() for item in collections]
             and raw.get("validation") == validation.provenance(),
             f"convergence snapshot collection provenance differs: {path}")
    state = raw.get("proposal")
    _require(isinstance(state, dict) and state
             and all(isinstance(value, Tensor) for value in state.values()),
             f"convergence snapshot has no proposal tensor state: {path}")
    _require(_all_finite_tensors(state.values()),
             f"convergence snapshot proposal is nonfinite: {path}")
    identity = {
        "kind": "outcome_training_checkpoint",
        "path": str(path), "sha256": digest, "size": int(after.st_size),
        "mtime_ns": int(after.st_mtime_ns), "global_step": int(step),
        "config_hash": config_hash,
        "proposal": proposal_module_digest(state),
    }
    return raw, identity


def _terminal_eval_assignment(rank: int, world: int) -> tuple[str, int | None]:
    """Return the one immutable terminal task owned by ``rank``."""
    _require(world == EXPECTED_WORLD_SIZE,
             f"terminal evaluation requires world={EXPECTED_WORLD_SIZE}, got {world}")
    _require(len(_TERMINAL_EVAL_TASKS) <= world,
             "terminal evaluation has more tasks than ranks")
    if not 0 <= int(rank) < int(world):
        raise ValueError(f"rank {rank} is outside world {world}")
    if rank >= len(_TERMINAL_EVAL_TASKS):
        return "idle", None
    return _TERMINAL_EVAL_TASKS[rank]


def _proposal_architecture(
    resolved_config: Mapping[str, Any],
    *,
    device: torch.device,
) -> Proposal:
    kwargs = dict(resolved_config.get("model", {}).get("proposal", {}) or {})
    return Proposal(**kwargs).to(device)


def _evaluate_one_convergence_snapshot(
    step: int,
    *,
    parent: Mapping[str, Any] | None,
    parent_identity: Mapping[str, Any],
    live_proposal: nn.Module,
    run_dir: Path,
    config_hash: str,
    resolved_config: Mapping[str, Any],
    collections: Sequence[ValidatedRecoveryCollection],
    validation: ValidatedRecoveryCollection,
    trainer_source: Mapping[str, Any],
    exact_behaviour_identity: Mapping[str, Any],
    initial_behavior_identity: Mapping[str, Any],
    device: torch.device,
    chunk_replans: int,
    stop_check: Any | None = None,
) -> dict[str, Any]:
    """Evaluate exactly one predeclared proposal state on all heldout groups."""
    allowed = {START_STEP, *CONVERGENCE_SNAPSHOT_STEPS}
    _require(int(step) in allowed, f"unexpected convergence snapshot step {step}")
    if stop_check is not None and bool(stop_check()):
        raise _PreemptRequested(
            f"preemption requested before convergence snapshot {step}"
        )
    _assert_trainer_source_identity(trainer_source)
    if int(step) == START_STEP:
        _require(parent is not None, "seed snapshot rank has no authenticated parent")
        evaluator = _load_proposal(parent, device=device)
        identity = {
            **_json_copy(parent_identity),
            "proposal": proposal_model_digest(parent["model"]),
        }
    else:
        raw, identity = _authenticated_snapshot_payload(
            _trainer_checkpoint_path(run_dir, step), step=step,
            config_hash=config_hash, resolved_config=resolved_config,
            parent_identity=parent_identity, collections=collections,
            validation=validation, trainer_source=trainer_source,
            exact_behaviour_identity=exact_behaviour_identity,
            initial_behavior_identity=initial_behavior_identity,
        )
        evaluator = _proposal_architecture(resolved_config, device=device)
        evaluator.load_state_dict(raw["proposal"], strict=True)
        if int(step) == STOP_STEP:
            _require(
                proposal_module_digest(evaluator.state_dict())
                == proposal_module_digest(live_proposal.state_dict()),
                "terminal live proposal differs from the step-54466 snapshot",
            )
    report = evaluate_validation_surrogate(
        evaluator, validation, device=device, chunk_replans=chunk_replans,
        stop_check=stop_check,
    )
    report["checkpoint"] = identity
    _assert_trainer_source_identity(trainer_source)
    return report


def _run_terminal_eval_task(
    *,
    rank: int,
    world: int,
    parent: Mapping[str, Any] | None,
    parent_identity: Mapping[str, Any],
    live_proposal: nn.Module,
    run_dir: Path,
    config_hash: str,
    resolved_config: Mapping[str, Any],
    collections: Sequence[ValidatedRecoveryCollection],
    validation: ValidatedRecoveryCollection,
    trainer_source: Mapping[str, Any],
    exact_behaviour_identity: Mapping[str, Any],
    initial_behavior_identity: Mapping[str, Any],
    device: torch.device,
    chunk_replans: int,
    unexpected_gradients: Sequence[str],
    stop_check: Any | None = None,
) -> dict[str, Any]:
    """Run one rank-local terminal task without stranding collective peers."""
    started = time.time()
    row: dict[str, Any] = {"rank": int(rank)}
    try:
        kind, step = _terminal_eval_assignment(rank, world)
        row.update({
            "kind": kind, "step": step,
            "parallelism": TERMINAL_PARALLELISM,
            "proposal_scoring": _proposal_scoring_geometry(device),
            "live_proposal": proposal_module_digest(live_proposal.state_dict()),
        })
        if kind == "snapshot":
            row["report"] = _evaluate_one_convergence_snapshot(
                int(step), parent=parent, parent_identity=parent_identity,
                live_proposal=live_proposal, run_dir=run_dir,
                config_hash=config_hash, resolved_config=resolved_config,
                collections=collections, validation=validation,
                trainer_source=trainer_source,
                exact_behaviour_identity=exact_behaviour_identity,
                initial_behavior_identity=initial_behavior_identity,
                device=device,
                chunk_replans=chunk_replans, stop_check=stop_check,
            )
        elif kind == "trust":
            row["report"] = evaluate_trust_gates(
                live_proposal, validation, device=device,
                chunk_replans=chunk_replans, training_nonfinite=0,
                unexpected_gradients=unexpected_gradients,
                stop_check=stop_check,
            )
        else:
            _require(kind == "idle" and step is None,
                     f"invalid terminal task {(kind, step)}")
            row["report"] = None
        row["ok"] = True
    except _PreemptRequested as exc:
        row.update({
            "ok": False, "preempted": True,
            "error": f"{type(exc).__name__}: {exc}",
        })
    except Exception as exc:  # noqa: BLE001
        row.update({
            "ok": False, "preempted": False,
            "error": f"{type(exc).__name__}: {exc}",
        })
    row["elapsed_seconds"] = time.time() - started
    return row


def _assemble_terminal_eval_results(
    rows: Sequence[Mapping[str, Any]],
    *,
    world: int,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Fail closed and reconstruct the predeclared reports after all-gather."""
    _require(len(rows) == world,
             f"terminal evaluation gathered {len(rows)} ranks, expected {world}")
    by_rank = {int(row.get("rank", -1)): dict(row) for row in rows}
    _require(set(by_rank) == set(range(world)),
             "terminal evaluation rank identities are missing or duplicated")
    preempted = [row for row in by_rank.values() if row.get("preempted")]
    if preempted:
        raise _PreemptRequested(
            "terminal evaluation preempted: "
            + "; ".join(str(row.get("error")) for row in preempted[:2])
        )
    failed = [row for row in by_rank.values() if not bool(row.get("ok"))]
    _require(not failed, f"terminal rank task failed: {failed[:2]}")

    snapshots: dict[int, dict[str, Any]] = {}
    trust_rows: list[dict[str, Any]] = []
    execution_rows: list[dict[str, Any]] = []
    scoring_geometries: list[dict[str, Any]] = []
    live_proposals: list[dict[str, Any]] = []
    for rank in range(world):
        row = by_rank[rank]
        expected_kind, expected_step = _terminal_eval_assignment(rank, world)
        _require(
            row.get("kind") == expected_kind and row.get("step") == expected_step,
            f"terminal rank {rank} executed the wrong task",
        )
        _require(row.get("parallelism") == TERMINAL_PARALLELISM,
                 f"terminal rank {rank} changed parallelism")
        geometry = dict(row.get("proposal_scoring") or {})
        device_type = str(geometry.get("device_type") or "")
        _require(device_type in {"cpu", "cuda"},
                 f"terminal rank {rank} has invalid scoring device")
        _require(
            geometry == _proposal_scoring_geometry(torch.device(device_type)),
            f"terminal rank {rank} changed exact proposal scoring geometry",
        )
        scoring_geometries.append(geometry)
        live = dict(row.get("live_proposal") or {})
        _require(_valid_sha256(live.get("sha256")),
                 f"terminal rank {rank} has no live proposal digest")
        live_proposals.append(live)
        report = row.get("report")
        if expected_kind == "snapshot":
            _require(isinstance(report, Mapping),
                     f"terminal snapshot rank {rank} has no report")
            snapshots[int(expected_step)] = dict(report)
        elif expected_kind == "trust":
            _require(isinstance(report, Mapping),
                     f"terminal trust rank {rank} has no report")
            trust_rows.append(dict(report))
        else:
            _require(report is None, "idle terminal rank emitted a report")
        execution_rows.append({
            "rank": rank, "kind": expected_kind, "step": expected_step,
            "elapsed_seconds": float(row.get("elapsed_seconds", 0.0)),
            "proposal_scoring": _json_copy(geometry),
            "live_proposal": _json_copy(live),
        })
    expected_steps = {START_STEP, *CONVERGENCE_SNAPSHOT_STEPS}
    _require(set(snapshots) == expected_steps,
             "terminal convergence snapshots are incomplete")
    _require(len(trust_rows) == 1, "terminal trust report is missing or duplicated")
    _require(all(row == scoring_geometries[0] for row in scoring_geometries),
             "terminal ranks used different proposal scoring geometry")
    _require(all(row == live_proposals[0] for row in live_proposals),
             "terminal ranks do not hold the same final proposal")
    final_checkpoint = snapshots[STOP_STEP].get("checkpoint") or {}
    _require(
        dict(final_checkpoint.get("proposal") or {}) == live_proposals[0],
        "terminal final snapshot differs from the proposal on every rank",
    )
    execution = {
        "parallelism": TERMINAL_PARALLELISM,
        "world_size": int(world),
        "tasks": execution_rows,
        "live_proposal": _json_copy(live_proposals[0]),
    }
    return snapshots, trust_rows[0], execution


def train_outcome_grpo(
    *,
    config: Mapping[str, Any],
    run_dir: str | os.PathLike[str],
    stop_at: int | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    """Run/resume the fixed six-fold stage on the absolute 80k schedule."""
    cfg = _json_copy(dict(config))
    validate_recipe_config(cfg)
    rank, world, _local_rank, target_device = _dist_info()
    _require(world == int(cfg["outcome_grpo"]["world_size"]),
             f"runtime world_size={world} != locked recipe world_size="
             f"{cfg['outcome_grpo']['world_size']}")
    run_path = Path(run_dir).expanduser().resolve()
    if rank == 0:
        run_path.mkdir(parents=True, exist_ok=True)
    _barrier()
    # Rank 0 holds this descriptor for the lifetime of the function.  All
    # ranks fail together if a duplicate Slurm allocation targets this run.
    run_lock: _RunDirectoryLock | None = None
    lock_box: list[Any] = [None]
    if rank == 0:
        try:
            run_lock = _acquire_run_directory_lock(run_path)
            lock_box[0] = {"ok": True}
        except Exception as exc:  # noqa: BLE001
            lock_box[0] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if world > 1:
        torch.distributed.broadcast_object_list(lock_box, src=0)
    _require(bool(lock_box[0].get("ok")), str(lock_box[0].get("error")))
    # Start the allocation clock and install signal handlers before hashing the
    # 1.7 GiB parent or any of the seven deep-authenticated collections.
    guard = PreemptGuard(run_path)
    if guard.should_stop():
        return {
            "status": "PREEMPTED_BEFORE_AUTH", "global_step": START_STEP,
            "reason": guard.reason,
        }
    start_source = _trainer_source_identity()
    if bool(cfg["run"].get("deterministic")):
        enable_determinism()
    strict_determinism = _configure_strict_outcome_determinism()
    proposal_scoring_geometry = _configure_exact_proposal_scoring(target_device)
    if rank == 0:
        print(
            "[outcome-grpo] strict_determinism="
            + json.dumps(strict_determinism, sort_keys=True),
            flush=True,
        )
        print(
            "[outcome-grpo] proposal_scoring="
            + json.dumps(proposal_scoring_geometry, sort_keys=True),
            flush=True,
        )
    seed = int(cfg["run"].get("seed", TRAIN_SEED))
    set_global_seed(seed, rank)
    config_hash = _config_hash(cfg)
    if rank == 0:
        config_path = run_path / "config.json"
        encoded = json.dumps(cfg, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if config_path.exists():
            _require(json.loads(config_path.read_text()) == cfg,
                     "run directory already contains a different resolved config")
        else:
            atomic_write_text(config_path, encoded)
    _barrier()
    _assert_trainer_source_identity(start_source)
    if guard.should_stop():
        return {
            "status": "PREEMPTED_BEFORE_SEED_AUTH", "global_step": START_STEP,
            "reason": guard.reason,
        }

    # Authenticate the 1.7 GiB seed once, then broadcast its immutable identity.
    identity_box: list[Any] = [None]
    if rank == 0:
        try:
            seed_path = _resolve_repo_path(cfg["outcome_grpo"]["seed_checkpoint"])
            identity_box[0] = {"ok": True,
                               "identity": recovery.authenticate_seed_checkpoint(seed_path)}
        except Exception as exc:  # noqa: BLE001
            identity_box[0] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if world > 1:
        torch.distributed.broadcast_object_list(identity_box, src=0)
    _require(identity_box[0]["ok"], f"seed authentication failed: {identity_box[0].get('error')}")
    parent_identity = dict(identity_box[0]["identity"])
    _assert_seed_stat(parent_identity)
    parent = _load_parent_from_identity(parent_identity)
    _assert_seed_stat(parent_identity)
    _assert_trainer_source_identity(start_source)
    if guard.should_stop():
        return {
            "status": "PREEMPTED_AFTER_SEED_AUTH", "global_step": START_STEP,
            "reason": guard.reason,
        }
    parent_samples_seen = int(parent.get("samples_seen", 0))
    proposal = _load_proposal(parent, device=target_device)
    _require_exact_proposal_scoring_environment(proposal, target_device)
    proposal_initial = proposal_model_digest(parent["model"])
    parent_runtime_proposal = proposal_module_digest(proposal.state_dict())

    # Seven collections are divided across ranks for the expensive deep schema
    # and behaviour-policy replay. Every result is gathered before any update.
    fold_specs = [dict(row) for row in cfg["outcome_grpo"]["folds"]]
    validation_spec = dict(cfg["outcome_grpo"]["validation"])
    specs = fold_specs + [validation_spec]
    local_auth: list[dict[str, Any]] = []
    local_objects: dict[int, ValidatedRecoveryCollection] = {}
    try:
        for index, spec in enumerate(specs):
            if index % world != rank:
                continue
            collection = ValidatedRecoveryCollection.open(
                _resolve_repo_path(spec["path"]),
                checkpoint_identity=parent_identity,
                expected_split=str(spec["split"]), deep=True,
                stop_check=lambda: _guard_local_should_stop(guard),
            )
            behaviour = authenticate_behaviour_policy(
                proposal, collection, device=target_device,
                chunk_replans=int(cfg["outcome_grpo"]["authentication"]["chunk_replans"]),
                stop_check=lambda: _guard_local_should_stop(guard),
            )
            local_objects[index] = collection
            local_auth.append({
                "index": index, "ok": True, "behaviour": behaviour,
                "provenance": collection.provenance(),
            })
    except Exception as exc:  # noqa: BLE001
        local_auth.append({"index": -1, "ok": False,
                           "preempted": isinstance(exc, _PreemptRequested),
                           "error": f"{type(exc).__name__}: {exc}"})
    gathered: list[Any] = [None for _ in range(world)]
    if world > 1:
        torch.distributed.all_gather_object(gathered, local_auth)
    else:
        gathered = [local_auth]
    flat_auth = [row for rank_rows in gathered for row in rank_rows]
    preempted_auth = [row for row in flat_auth if row.get("preempted")]
    if preempted_auth:
        return {
            "status": "PREEMPTED_DURING_COLLECTION_AUTH",
            "global_step": START_STEP, "reason": guard.reason,
        }
    failures = [row for row in flat_auth if not row.get("ok")]
    _require(not failures, f"collection/behaviour authentication failed: {failures[:2]}")
    by_index = {int(row["index"]): row["behaviour"] for row in flat_auth}
    owner_provenance = {
        int(row["index"]): _json_copy(row["provenance"]) for row in flat_auth
    }
    _require(set(by_index) == set(range(len(specs))),
             "not every configured recovery collection was deeply authenticated")
    exact_auth = _require_exact_gathered_behaviour_auth(by_index)
    if rank == 0:
        print(
            "[outcome-grpo] behaviour_auth=" + json.dumps([
                {"split": str(specs[index]["split"]), **by_index[index]}
                for index in range(len(specs))
            ], sort_keys=True),
            flush=True,
        )
        print(
            "[outcome-grpo] exact_behaviour_identity="
            + json.dumps(exact_auth, sort_keys=True),
            flush=True,
        )
    all_collections: list[ValidatedRecoveryCollection] = []
    for index, spec in enumerate(specs):
        collection = local_objects.get(index)
        if collection is None:
            collection = ValidatedRecoveryCollection.open(
                _resolve_repo_path(spec["path"]),
                checkpoint_identity=parent_identity,
                expected_split=str(spec["split"]), deep=False,
                verify_sidecars=False,
            )
        _assert_owner_collection_snapshot(collection, owner_provenance[index])
        all_collections.append(collection)
    collections = all_collections[:N_FOLDS]
    validation = all_collections[N_FOLDS]
    _assert_trainer_source_identity(start_source)
    if guard.should_stop():
        return {
            "status": "PREEMPTED_AFTER_COLLECTION_AUTH",
            "global_step": START_STEP, "reason": guard.reason,
        }

    local_error = ""
    try:
        anchor = ExpertAnchor.from_parent(
            parent, proposal, trainer_cfg=cfg, device=target_device,
            rank=rank, world_size=world,
        )
        anchor_preflight = anchor.preflight(START_STEP)
        initial_support_overlap = expert_support_overlap(anchor, start_step=START_STEP)
    except Exception as exc:  # noqa: BLE001
        local_error = f"{type(exc).__name__}: {exc}"
    _raise_if_any_rank_failed(local_error, world, "expert-anchor preflight")
    if guard.should_stop():
        return {
            "status": "PREEMPTED_DURING_ANCHOR_PREFLIGHT",
            "global_step": START_STEP, "reason": guard.reason,
        }
    initial_support_overlap = _mean_across_ranks(
        {"overlap": initial_support_overlap}, world,
    )["overlap"]
    if rank != 0:
        # Rank 0 alone needs the full consolidated parent to emit the terminal
        # descendant; other ranks retain only their frozen E/q_action copies.
        del parent

    expected_fold_groups = int(cfg["outcome_grpo"]["groups_per_train_fold"])
    _require(all(len(collection.receipts) == expected_fold_groups
                 for collection in collections),
             "authenticated train-fold group counts differ from the locked recipe")
    _require(len(validation.receipts) == int(cfg["outcome_grpo"]["validation_groups"]),
             "authenticated validation group count differs from the locked recipe")
    validation_tasks = {
        (str(item.suite), int(item.task_id)) for item in validation.items
    }
    _require(len(validation_tasks) == int(cfg["outcome_grpo"]["validation_tasks"]),
             "authenticated validation task count differs from the locked recipe")
    informative = [collection.informative_indices() for collection in collections]
    minimum_train_informative = int(
        cfg["outcome_grpo"]["minimum_informative_groups_per_fold"]
    )
    _require(all(len(indices) >= minimum_train_informative for indices in informative),
             "a train fold does not meet the locked informative-group power floor")
    validation_informative = validation.informative_indices()
    _require(
        len(validation_informative)
        >= int(cfg["outcome_grpo"]["minimum_validation_informative_groups"]),
        "validation does not meet the locked informative-group power floor",
    )
    informative_report = [{
        "split": collection.split,
        "informative_groups": len(indices),
        "all_equal_groups": len(collection.receipts) - len(indices),
        "total_groups": len(collection.receipts),
        "fraction": len(indices) / len(collection.receipts),
        "minimum_required": minimum_train_informative,
        "validated": True,
    } for collection, indices in zip(collections, informative, strict=True)]
    validation_informative_report = {
        "split": validation.split,
        "informative_groups": len(validation_informative),
        "all_equal_groups": len(validation.receipts) - len(validation_informative),
        "total_groups": len(validation.receipts),
        "fraction": len(validation_informative) / len(validation.receipts),
        "minimum_required": int(
            cfg["outcome_grpo"]["minimum_validation_informative_groups"]
        ),
        "n_tasks": len(validation_tasks),
        "validated": True,
    }
    sampler = DeterministicOutcomeSampler(
        informative, seed=seed, rank=rank, world_size=world,
        start_step=START_STEP, updates_per_fold=UPDATES_PER_FOLD,
        contexts_per_arm=int(cfg["outcome_grpo"]["contexts_per_arm"]),
        identity_digests=[collection.identity_digest for collection in collections],
    )
    wrapper = _ProposalOnly(proposal)
    optimizer = build_optimizer(
        wrapper, lr=float(cfg["optim"]["lr"]),
        weight_decay=float(cfg["optim"]["weight_decay"]),
        betas=tuple(float(value) for value in cfg["optim"]["betas"]),
        lr_scales={"proposal": float(cfg["optim"]["lr_scales"]["proposal"])},
        module_names=["proposal"],
    )
    scheduler = CosineWithWarmup(
        base_lr=float(cfg["optim"]["lr"]),
        warmup_steps=int(cfg["optim"]["warmup"]),
        total_steps=SCHEDULE_STEPS,
        min_lr_ratio=float(cfg["optim"]["min_lr_ratio"]),
    )
    _require(len(optimizer.state) == 0,
             "proposal optimizer state was not empty at stage entry")
    wandb_id = wandb_util.stable_run_id(run_path) if rank == 0 else ""
    id_box = [wandb_id]
    if world > 1:
        torch.distributed.broadcast_object_list(id_box, src=0)
    wandb_id = str(id_box[0])
    resumed = _load_trainer_checkpoint(
        run_path, config_hash=config_hash, resolved_config=cfg,
        parent_identity=parent_identity, collections=collections,
        validation=validation, proposal=proposal, optimizer=optimizer,
        scheduler=scheduler, sampler=sampler,
        exact_behaviour_identity=exact_auth, trainer_source=start_source,
    )
    if resumed is None:
        global_step = START_STEP
        samples_seen = parent_samples_seen
        initial_behavior_identity: dict[str, Any] | None = None
        if rank == 0:
            _save_trainer_checkpoint(
                run_path, global_step=global_step, samples_seen=samples_seen,
                config_hash=config_hash, resolved_config=cfg,
                parent_identity=parent_identity, collections=collections,
                validation=validation, proposal=proposal, optimizer=optimizer,
                scheduler=scheduler, sampler=sampler, wandb_run_id=wandb_id,
                exact_behaviour_identity=exact_auth,
                initial_behavior_identity=initial_behavior_identity,
                trainer_source=start_source,
                stop_reason="optimizer_reset_at_entry",
            )
    else:
        (global_step, samples_seen, saved_wandb_id,
         initial_behavior_identity) = resumed
        _require(saved_wandb_id == wandb_id, "W&B stable run ID changed on resume")
    start_checkpoint_identity = _require_start_step_checkpoint_identity(
        proposal, optimizer, global_step=global_step,
        parent_proposal=parent_runtime_proposal,
    )
    if rank == 0 and start_checkpoint_identity["checked"]:
        print(
            "[outcome-grpo] start_checkpoint_identity="
            + json.dumps(start_checkpoint_identity, sort_keys=True),
            flush=True,
        )
    _barrier()

    metrics_path = run_path / "metrics.jsonl"
    metrics_error = ""
    if rank == 0:
        try:
            _reconcile_metrics_to_checkpoint(
                metrics_path, checkpoint_step=global_step, config_hash=config_hash,
            )
        except Exception as exc:  # noqa: BLE001
            metrics_error = f"{type(exc).__name__}: {exc}"
    _raise_if_any_rank_failed(metrics_error, world, "metrics resume reconciliation")
    _assert_trainer_source_identity(start_source)
    if guard.should_stop():
        return {
            "status": "CHECKPOINTED", "global_step": global_step,
            "config_hash": config_hash, "wandb_run_id": wandb_id,
            "reason": guard.reason,
        }

    target_step = STOP_STEP if stop_at is None else int(stop_at)
    _require(global_step <= target_step <= STOP_STEP,
             f"stop_at must be in [{global_step},{STOP_STEP}]")
    os.environ["WANDB_MODE"] = str(
        os.environ.get(
            "LOOM_WANDB_MODE",
            os.environ.get("WANDB_MODE", cfg["run"]["wandb_mode"]),
        )
    )
    run = wandb_util.init(
        run_path, str(cfg["run"].get("project", "loom")), cfg,
        rank=rank, name=str(cfg["run"].get("name", "r0a_outcome_grpo")),
    )
    metrics_handle = metrics_path.open("a", encoding="utf-8") if rank == 0 else None
    log_every = int(cfg["run"]["log_every"])
    ckpt_every = int(cfg["run"]["ckpt_every"])
    anchor_batch = int(cfg["data"]["batch_per_gpu"])
    outcome_batch = int(cfg["outcome_grpo"]["contexts_per_arm"]) * 7
    t0 = time.time()
    # Collection scored the proposal in eval mode.  The module currently has no
    # dropout/BN and eval mode retains all parameter gradients; keeping it here
    # makes that contract future-proof if stateful layers are ever introduced.
    proposal.eval()
    stopped_for_guard = False
    while global_step < target_step:
        step = global_step
        set_step_seed(seed, step, rank)
        local_error = ""
        local_initial_identity: dict[str, Any] | None = None
        try:
            fold, group_index, _visit = sampler.group_at(step)
            collection = collections[fold]
            payload = collection.load(group_index)
            indices = sampler.replans_at(
                step, collection.receipts[group_index]["n_replans_by_arm"],
            )
            amp = target_device.type == "cuda"
            # Importance ratios and Switch logits exactly replay collection:
            # row-wise B=1 fp32, outside the anchor's bf16 autocast region.
            grpo_loss, balance_loss, metrics = sampled_group_losses(
                proposal, payload, indices, device=target_device,
            )
            if step == START_STEP:
                local_initial_identity = _require_initial_behavior_ratio_identity(
                    metrics, device=target_device,
                )
            with torch.autocast(
                device_type=target_device.type,
                dtype=torch.bfloat16,
                enabled=amp,
            ):
                anchor_loss, anchor_metrics = anchor.loss(step)
            total = grpo_loss + anchor_loss + SWITCH_BALANCE_WEIGHT * balance_loss
            _require(bool(torch.isfinite(total)), f"nonfinite total loss at step {step}")
        except Exception as exc:  # noqa: BLE001
            local_error = f"{type(exc).__name__}: {exc}"
        _raise_if_any_rank_failed(local_error, world, f"step {step} forward/data")
        metrics["max_abs_logratio"] = _max_across_ranks(
            metrics["max_abs_logratio"], world, target_device,
        )
        if step == START_STEP:
            _require(local_initial_identity is not None,
                     "initial behavior identity evidence was not produced")
            local_initial_row = {
                "rank": rank,
                **local_initial_identity,
            }
            initial_rows: list[Any]
            if world > 1:
                initial_rows = [None for _ in range(world)]
                torch.distributed.all_gather_object(
                    initial_rows, local_initial_row,
                )
            else:
                initial_rows = [local_initial_row]
            initial_behavior_identity = _assemble_initial_behavior_identity(
                initial_rows,
                world=world,
                config_hash=config_hash,
                trainer_source=start_source,
                parent_identity=parent_identity,
                exact_behaviour_identity=exact_auth,
                start_checkpoint_identity=start_checkpoint_identity,
            )
            if rank == 0:
                print(
                    "[outcome-grpo] initial_behavior_ratio_identity="
                    + json.dumps(
                        initial_behavior_identity,
                        sort_keys=True,
                        allow_nan=False,
                    ),
                    flush=True,
                )
        # The first-update identity assertion and its cross-rank failure gather
        # complete before either of these optimizer mutations or backward.
        lrs = scheduler.apply(optimizer, step)
        optimizer.zero_grad(set_to_none=True)
        local_error = ""
        try:
            total.backward()
            unexpected = anchor.unexpected_gradients()
            _require(not unexpected,
                     f"frozen estimator/q_action received gradients: {unexpected[:8]}")
        except Exception as exc:  # noqa: BLE001
            local_error = f"{type(exc).__name__}: {exc}"
        _raise_if_any_rank_failed(local_error, world, f"step {step} backward")
        grad_norm = _sync_proposal_grads(proposal, world)
        optimizer.step()
        _require(_all_finite_tensors(proposal.parameters()),
                 f"proposal parameters became nonfinite at step {step}")
        _require(_optimizer_finite(optimizer),
                 f"proposal optimizer became nonfinite at step {step}")
        global_step += 1
        samples_seen += (outcome_batch + anchor_batch) * world
        metrics.update({
            "loss": float(total.detach()),
            "anchor_sparse_ce": anchor_metrics["sparse_ce"],
            "anchor_loss": anchor_metrics["weighted_sparse_ce"],
            "balance_weighted": SWITCH_BALANCE_WEIGHT * float(balance_loss.detach()),
            "grad_norm": float(grad_norm),
            "grad_skipped": 0.0,
            "fold": float(fold),
            "informative_fraction": informative_report[fold]["fraction"],
        })
        reduced = _reduce_training_metrics(metrics, world, target_device)
        if rank == 0 and metrics_handle is not None:
            metrics_handle.write(json.dumps({
                "global_step": global_step,
                "accepted_update": global_step - START_STEP,
                "lr": min(lrs.values()),
                "config_hash": config_hash,
                **reduced,
            }, sort_keys=True, allow_nan=False) + "\n")
            metrics_handle.flush()
        if (global_step - START_STEP) % log_every == 0:
            write_heartbeat(run_path, global_step, rank)
            if rank == 0 and not quiet:
                print(
                    f"[outcome-grpo] step={global_step}/{STOP_STEP} "
                    f"fold={fold} loss={reduced['loss']:.5f} "
                    f"grpo={reduced['grpo_loss']:.5f} "
                    f"anchor={reduced['anchor_sparse_ce']:.5f} "
                    f"balance={reduced['proposal_balance']:.5f} "
                    f"clip={reduced['clip_fraction']:.3f} "
                    f"gnorm={reduced['grad_norm']:.3f} "
                    f"lr={min(lrs.values()):.3e} "
                    f"{(global_step - START_STEP) / max(1e-6, time.time() - t0):.2f} it/s",
                    flush=True,
                )
            wandb_util.log(run, {
                **reduced, "samples_seen": samples_seen,
                "seconds_to_budget": guard.seconds_left,
                **{f"lr/{key}": value for key, value in lrs.items()},
            }, global_step)
        stop = guard.should_stop()
        should_save = (stop or (global_step - START_STEP) % ckpt_every == 0
                       or global_step >= target_step)
        if should_save:
            if rank == 0 and metrics_handle is not None:
                _durable_metrics_barrier(metrics_handle)
            source_error = ""
            try:
                _assert_trainer_source_identity(start_source)
            except Exception as exc:  # noqa: BLE001
                source_error = f"{type(exc).__name__}: {exc}"
            _raise_if_any_rank_failed(source_error, world, "trainer source TOCTOU check")
            _barrier()
            if rank == 0:
                _save_trainer_checkpoint(
                    run_path, global_step=global_step, samples_seen=samples_seen,
                    config_hash=config_hash, resolved_config=cfg,
                    parent_identity=parent_identity, collections=collections,
                    validation=validation, proposal=proposal, optimizer=optimizer,
                    scheduler=scheduler, sampler=sampler, wandb_run_id=wandb_id,
                    exact_behaviour_identity=exact_auth,
                    initial_behavior_identity=initial_behavior_identity,
                    trainer_source=start_source,
                    stop_reason=guard.reason if stop else "",
                )
            _barrier()
        if stop:
            stopped_for_guard = True
            break

    if metrics_handle is not None:
        metrics_handle.close()
    if global_step < STOP_STEP or stopped_for_guard:
        wandb_util.finish(run)
        return {
            "status": "CHECKPOINTED",
            "global_step": global_step,
            "config_hash": config_hash,
            "wandb_run_id": wandb_id,
        }

    # Exact B=1 proposal replay is intentionally not vectorised: collection ran
    # one replan at a time.  Use the otherwise-idle replicated ranks to evaluate
    # the seed + five immutable snapshots + final trust gate concurrently.
    # Rank 0 alone assembles those authenticated reports and may emit a sole
    # descendant; no simulator evaluation is launched here.
    if guard.should_stop():
        wandb_util.finish(run)
        return {
            "status": "CHECKPOINTED", "global_step": global_step,
            "config_hash": config_hash, "wandb_run_id": wandb_id,
            "reason": guard.reason,
        }
    proposal.eval()
    local_error = ""
    try:
        final_support_overlap = expert_support_overlap(
            anchor, start_step=START_STEP,
        )
    except Exception as exc:  # noqa: BLE001
        local_error = f"{type(exc).__name__}: {exc}"
        final_support_overlap = math.nan
    _raise_if_any_rank_failed(local_error, world, "final expert-support gate")
    final_support_overlap = _mean_across_ranks(
        {"overlap": final_support_overlap}, world,
    )["overlap"]
    if guard.should_stop():
        wandb_util.finish(run)
        return {
            "status": "CHECKPOINTED", "global_step": global_step,
            "config_hash": config_hash, "wandb_run_id": wandb_id,
            "reason": guard.reason,
        }
    local_terminal = _run_terminal_eval_task(
        rank=rank, world=world,
        parent=parent if rank == 0 else None,
        parent_identity=parent_identity, live_proposal=proposal,
        run_dir=run_path, config_hash=config_hash, resolved_config=cfg,
        collections=collections, validation=validation,
        trainer_source=start_source, device=target_device,
        exact_behaviour_identity=exact_auth,
        initial_behavior_identity=initial_behavior_identity,
        chunk_replans=int(
            cfg["outcome_grpo"]["authentication"]["chunk_replans"]
        ),
        unexpected_gradients=anchor.unexpected_gradients(),
        stop_check=lambda: _guard_local_should_stop(guard),
    )
    terminal_rows: list[Any]
    if world > 1:
        terminal_rows = [None for _ in range(world)]
        torch.distributed.all_gather_object(terminal_rows, local_terminal)
    else:
        terminal_rows = [local_terminal]
    terminal_box: list[Any] = [None]
    if rank == 0:
        convergence: dict[str, Any] | None = None
        trust: dict[str, Any] | None = None
        terminal_execution: dict[str, Any] | None = None
        pending_candidate: Path | None = None
        candidate = run_path / f"candidate_{global_step:09d}.pt"
        try:
            metric_rows = _exact_terminal_metrics(
                metrics_path, config_hash=config_hash,
            )
            snapshot_reports, trust, terminal_execution = (
                _assemble_terminal_eval_results(terminal_rows, world=world)
            )
            convergence = evaluate_convergence_gate(snapshot_reports, metric_rows)
            support_check = {
                "initial": initial_support_overlap,
                "final": final_support_overlap,
                "change": final_support_overlap - initial_support_overlap,
                "threshold": -MAX_TOPK_OVERLAP_DECLINE,
                "pass": final_support_overlap - initial_support_overlap
                        >= -MAX_TOPK_OVERLAP_DECLINE,
            }
            trust["checks"]["expert_topk_overlap_change"] = support_check
            trust["passed"] = bool(trust["passed"] and support_check["pass"])
            combined_checks = {
                **{f"convergence/{name}": row
                   for name, row in convergence["checks"].items()},
                **{f"trust/{name}": row for name, row in trust["checks"].items()},
            }
            combined_passed = bool(convergence["passed"] and trust["passed"])
            if not combined_passed:
                raise TrustGateError({
                    "status": "FAIL", "passed": False,
                    "checks": combined_checks,
                    "convergence_gate": convergence, "trust_gate": trust,
                })
            for collection in collections + [validation]:
                collection.assert_all_sidecars_unchanged(
                    lambda: _guard_local_should_stop(guard)
                )
            _assert_seed_stat(parent_identity)
            _assert_trainer_source_identity(start_source)
            provenance = {
                "trainer_source": start_source,
                "strict_determinism": strict_determinism,
                "world_size": world,
                "exact_behaviour_identity": _json_copy(exact_auth),
                "start_checkpoint_identity": _json_copy(
                    initial_behavior_identity["start_checkpoint_identity"]
                ),
                "initial_behavior_ratio_identity": _json_copy(
                    initial_behavior_identity
                ),
                "collections": [item.provenance() for item in collections],
                "validation": validation.provenance(),
                # Legacy singular key remains useful to generic artifact readers.
                "collection": validation.provenance(),
                "behaviour_authentication": [by_index[index] for index in range(len(specs))],
                "expert_anchor": anchor_preflight,
                "informative_groups": informative_report,
                "validation_informative_groups": validation_informative_report,
                "terminal_evaluation": terminal_execution,
                "sampler": sampler.state_dict(global_step),
                "optimizer_reset": {
                    "count": 1, "modules": ["proposal"],
                    "source_global_step": START_STEP,
                },
                "recipe": {
                    "algorithm": "stored_order_pl_clipped_grpo",
                    "reward": "terminal_LIBERO_success_only",
                    "arm0": "all-eight baseline only; no ratio/loss",
                    "sampled_arms": list(range(1, 8)),
                    "aggregation": "equal group/arm/replan",
                    "clip_eps": CLIP_EPS,
                    "folds": N_FOLDS,
                    "updates_per_fold": UPDATES_PER_FOLD,
                    "absolute_schedule_steps": SCHEDULE_STEPS,
                    "forbidden": ["Phi", "bank", "shaped_reward"],
                },
                "training": {
                    "optimizer_steps": global_step - START_STEP,
                    "samples_seen": samples_seen,
                    "initial_proposal": proposal_initial,
                    "unexpected_gradients": [],
                    "nonfinite": 0,
                },
                "convergence_gate": convergence,
                "trust_gate": trust,
            }
            _require(not candidate.exists(),
                     f"refusing to overwrite existing candidate: {candidate}")
            pending_candidate = run_path / (
                f".candidate_{global_step:09d}.pending-{os.getpid()}.pt"
            )
            _require(not pending_candidate.exists(),
                     f"stale pending candidate exists: {pending_candidate}")
            report = write_descendant_checkpoint(
                pending_candidate, parent=parent, parent_identity=parent_identity,
                proposal=proposal, optimizer=optimizer,
                optimizer_steps=global_step - START_STEP,
                provenance=provenance, global_step=global_step,
                resolved_config=cfg, samples_seen=samples_seen,
                wandb_run_id=wandb_id, verify_policy=True,
            )
            _assert_trainer_source_identity(start_source)
            # Hard-link publication is no-overwrite and leaves no moment where
            # a failed verification is visible under the candidate name.
            os.link(pending_candidate, candidate)
            fsync_dir(run_path)
            pending_candidate.unlink()
            fsync_dir(run_path)
            pending_candidate = None
            report.update({
                "path": str(candidate), "status": "PASS", "passed": True,
                "candidate_emitted": True,
                "checks": combined_checks,
                "convergence_gate": convergence, "trust_gate": trust,
                "wandb_run_id": wandb_id,
            })
            atomic_write_text(
                run_path / "terminal_report.json",
                json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            )
            terminal_box[0] = {"ok": True, "report": report}
        except _PreemptRequested as exc:
            terminal_box[0] = {
                "ok": False, "preempted": True,
                "error": f"{type(exc).__name__}: {exc}",
            }
        except Exception as exc:  # noqa: BLE001
            if pending_candidate is not None and pending_candidate.exists():
                pending_candidate.unlink()
                fsync_dir(run_path)
            embedded = getattr(exc, "report", None)
            failure = {
                "status": "FAIL", "passed": False,
                "global_step": global_step, "optimizer_steps": global_step - START_STEP,
                "config_hash": config_hash, "wandb_run_id": wandb_id,
                "created_utc": _utc(),
                "trainer_source": start_source,
                "strict_determinism": strict_determinism,
                "parent": _json_copy(parent_identity),
                "exact_behaviour_identity": _json_copy(exact_auth),
                "start_checkpoint_identity": _json_copy(
                    initial_behavior_identity["start_checkpoint_identity"]
                ),
                "initial_behavior_ratio_identity": _json_copy(
                    initial_behavior_identity
                ),
                "collections": [item.provenance() for item in collections],
                "validation": validation.provenance(),
                "informative_groups": informative_report,
                "validation_informative_groups": validation_informative_report,
                "terminal_evaluation": terminal_execution,
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "checks": (dict(embedded.get("checks", {}))
                           if isinstance(embedded, Mapping) else {}),
                "convergence_gate": (
                    embedded.get("convergence_gate")
                    if isinstance(embedded, Mapping)
                    and embedded.get("convergence_gate") is not None
                    else convergence
                ),
                "trust_gate": (
                    embedded.get("trust_gate")
                    if isinstance(embedded, Mapping)
                    and embedded.get("trust_gate") is not None
                    else trust
                ),
                "candidate_emitted": False,
            }
            persistence_error = ""
            try:
                _persist_terminal_failure(run_path, failure, candidate=candidate)
            except Exception as persist_exc:  # noqa: BLE001
                persistence_error = (
                    f"; terminal report persistence failed: "
                    f"{type(persist_exc).__name__}: {persist_exc}"
                )
            terminal_box[0] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}{persistence_error}",
                "report": failure if not persistence_error else None,
            }
    if world > 1:
        torch.distributed.broadcast_object_list(terminal_box, src=0)
    wandb_util.finish(run)
    if not terminal_box[0]["ok"]:
        if terminal_box[0].get("preempted"):
            return {
                "status": "CHECKPOINTED", "global_step": global_step,
                "config_hash": config_hash, "wandb_run_id": wandb_id,
                "reason": guard.reason or "terminal_preemption",
            }
        if terminal_box[0].get("report") is not None:
            raise TrustGateError(terminal_box[0]["report"])
        raise OutcomeGRPOError(terminal_box[0]["error"])
    return dict(terminal_box[0]["report"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True,
                        help="resolved by the standard YAML extends loader")
    parser.add_argument("--run-dir", required=True,
                        help="resumable trainer directory with LATEST/checkpoints")
    parser.add_argument("--stop-at", type=int, default=STOP_STEP,
                        help="link-local absolute stop; never changes the 80k LR curve")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from loom.train.loop import read_config  # noqa: PLC0415

        cfg = read_config(args.config)
        report = train_outcome_grpo(
            config=cfg,
            run_dir=args.run_dir,
            stop_at=args.stop_at,
            quiet=args.quiet,
        )
    except TrustGateError as exc:
        print(json.dumps({"status": "TRUST_GATE_FAILED", "report": exc.report},
                         indent=2, sort_keys=True, allow_nan=False), flush=True)
        return 4
    except ExpertAnchorUnavailable as exc:
        print(f"EXPERT_ANCHOR_UNAVAILABLE: {exc}", flush=True)
        return 3
    except (OutcomeGRPOError, FileExistsError, ValueError) as exc:
        print(f"OUTCOME_GRPO_FAILED: {exc}", flush=True)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True,
                     allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
