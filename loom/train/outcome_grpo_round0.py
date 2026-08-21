"""Minimal fail-closed round-0 trainer for terminal-outcome proposal GRPO.

This module consumes one complete authenticated recovery training fold and the
disjoint recovery validation split.  It updates only ``proposal`` for exactly
two canonical passes (200 groups x 2 = 400 joint optimizer updates):

``L = L_clipped_GRPO + L_sparse_CE + 1e-2 * L_Switch``.

All eight terminal rewards define the per-group normalisation.  Arm 0 is only
the deployed control: it never enters an importance ratio or a gradient path.
Arms 1--7 are scored with the exact stored Plackett--Luce order, with replans
averaged inside each trajectory before trajectories are averaged.  Sparse CE
uses the existing frozen estimator/q_action demonstration target path; Switch
balance reuses those four student-logit batches once per update.

No checkpoint is emitted unless validation ratios, ESS, L1 coefficient drift,
operator liveness, held-out expert-anchor preservation, finite health, and
byte-identical frozen tensors all pass their locked gates.
"""

from __future__ import annotations

import argparse
import copy
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
from loom.heads.proposal import Proposal, argmax_coeff, pl_log_prob, weights_from_logits
from loom.losses.proposal_bc import proposal_sparse_ce_loss

__all__ = [
    "FORMAT_VERSION", "LEARNING_RATE", "EPOCHS", "CLIP_EPS",
    "SWITCH_BALANCE_WEIGHT", "GRAD_CLIP", "MAX_CLIP_FRACTION",
    "MIN_ESS_FRACTION", "MAX_COEFF_DRIFT_P95_L1", "MIN_LIVE_OPS",
    "MAX_TOPK_OVERLAP_DECLINE", "EXPERT_GATE_BATCHES",
    "OutcomeGRPOError", "ExpertAnchorUnavailable", "TrustGateError",
    "ValidatedRecoveryCollection", "ExpertAnchor",
    "normalised_group_advantages", "clipped_grpo_objective",
    "group_grpo_loss", "proposal_switch_balance", "expert_anchor_objective",
    "stored_order_logprob", "expected_optimizer_updates",
    "model_state_digest", "frozen_model_digest", "evaluate_trust_gates",
    "write_descendant_checkpoint", "train_outcome_grpo_round0",
    "build_parser", "main",
]


FORMAT_VERSION = 1
TRAINER_KIND = "loom_outcome_grpo_round0_proposal_descendant"
LEARNING_RATE = 5e-6
EPOCHS = 2
CLIP_EPS = 0.20
SWITCH_BALANCE_WEIGHT = 1e-2
GRAD_CLIP = 1.0
MAX_CLIP_FRACTION = 0.20
MIN_ESS_FRACTION = 0.80
MAX_COEFF_DRIFT_P95_L1 = 0.05
MIN_LIVE_OPS = 16
MAX_TOPK_OVERLAP_DECLINE = 0.05
EXPERT_GATE_BATCHES = 16
TRAIN_SEED = 0
EXPECTED_TRAIN_GROUPS = 200

# Every mutable repository file whose implementation is executed by this
# standalone path.  Binding only this module would let a concurrent edit to a
# shared head, loss, loader, or expert-target implementation silently change
# the trained descendant while preserving the advertised trainer identity.
_TRAINER_SOURCE_FILES = (
    "contracts.py",
    "loom/train/outcome_grpo_round0.py",
    "scripts/train_outcome_grpo_round0.py",
    "loom/eval/outcome_recovery.py",
    "loom/eval/policy.py",
    "loom/heads/proposal.py",
    "loom/losses/proposal_bc.py",
    "loom/model/estimator.py",
    "loom/heads/q_action.py",
    "loom/heads/q_delta.py",
    "loom/train/loop.py",
    "loom/train/determinism.py",
    "loom/data/loader.py",
    "loom/data/cache.py",
    "loom/data/canonical.py",
    "loom/data/adapters/libero.py",
    "loom/data/tower.py",
)

ADAMW_BETAS = (0.9, 0.95)
ADAMW_WEIGHT_DECAY = 0.05
ADAMW_EPS = 1e-8

ANCHOR_MANIFEST = {
    "digest": "sha256:f61c453864dc8a84e274a65e834e037a83ef8407ed4e9635f84c78d814fe2e7e",
    "n_tasks": 40,
    "n_trajectories": 1960,
    "n_windows": 47271,
}

BEHAVIOUR_LOGPROB_ATOL = 2e-4
BEHAVIOUR_LOGPROB_RTOL = 2e-5
BEHAVIOUR_COEFF_ATOL = 2e-5
BEHAVIOUR_COEFF_RTOL = 2e-5

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
    """An authenticated input, training, or output invariant failed."""


class ExpertAnchorUnavailable(OutcomeGRPOError):
    """The existing sparse-CE/q_action demonstration path is unavailable."""


class TrustGateError(OutcomeGRPOError):
    """The locked terminal trust envelope rejected the candidate."""

    def __init__(self, report: Mapping[str, Any]):
        self.report = dict(report)
        failed = [name for name, row in self.report.get("checks", {}).items()
                  if not bool(row.get("pass"))]
        super().__init__("round-0 trust gate failed: " + ", ".join(failed))


def _require(condition: Any, message: str) -> None:
    if not bool(condition):
        raise OutcomeGRPOError(message)


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _valid_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(c in "0123456789abcdef" for c in text)


def _config_hash(cfg: Mapping[str, Any]) -> str:
    experiment = {key: value for key, value in cfg.items() if key != "link"}
    return hashlib.blake2b(
        json.dumps(experiment, sort_keys=True, default=str).encode(), digest_size=8,
    ).hexdigest()


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _tensor_raw(value: Tensor) -> tuple[Tensor, memoryview]:
    tensor = value.detach().cpu().contiguous()
    return tensor, memoryview(tensor.reshape(-1).view(torch.uint8).numpy())


def _update_tensor_hash(digest: Any, label: str, value: Tensor) -> int:
    tensor, raw = _tensor_raw(value)
    header = json.dumps(
        {"name": label, "dtype": str(tensor.dtype), "shape": list(tensor.shape)},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    digest.update(header + b"\0")
    digest.update(raw)
    digest.update(b"\0")
    return int(tensor.numel() * tensor.element_size())


def model_state_digest(
    state: Mapping[str, Tensor], *, include: Any | None = None,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    n_tensors = n_bytes = 0
    for name in sorted(state):
        if include is not None and not include(name):
            continue
        value = state[name]
        _require(isinstance(value, Tensor), f"model state {name!r} is not a tensor")
        n_bytes += _update_tensor_hash(digest, name, value)
        n_tensors += 1
    _require(n_tensors > 0, "model-state digest selected zero tensors")
    return {"sha256": digest.hexdigest(), "n_tensors": n_tensors, "n_bytes": n_bytes}


def frozen_model_digest(state: Mapping[str, Tensor]) -> dict[str, Any]:
    return model_state_digest(
        state, include=lambda name: not str(name).startswith("proposal."),
    )


def proposal_model_digest(state: Mapping[str, Tensor]) -> dict[str, Any]:
    return model_state_digest(
        state, include=lambda name: str(name).startswith("proposal."),
    )


def _all_finite(values: Iterable[Tensor]) -> bool:
    return all(bool(torch.isfinite(value.detach()).all()) for value in values)


def expected_optimizer_updates(n_groups: int) -> int:
    value = int(n_groups)
    if value <= 0:
        raise ValueError("n_groups must be positive")
    return EPOCHS * value


def normalised_group_advantages(rewards: Sequence[float] | Tensor) -> Tensor:
    value = torch.as_tensor(rewards, dtype=torch.float32).reshape(-1)
    if value.numel() != recovery.GROUP_SIZE:
        raise ValueError(
            f"one recovery group has {recovery.GROUP_SIZE} rewards, got {value.numel()}"
        )
    if not bool(torch.isfinite(value).all()):
        raise ValueError("group rewards contain nan/inf")
    centred = value - value.mean()
    variance = centred.square().mean()
    return torch.zeros_like(value) if float(variance) == 0.0 else centred / variance.sqrt()


def clipped_grpo_objective(
    current_logprob: Tensor,
    old_logprob: Tensor,
    advantage: float | Tensor,
    *,
    clip_eps: float = CLIP_EPS,
) -> tuple[Tensor, Tensor, Tensor]:
    if current_logprob.shape != old_logprob.shape:
        raise ValueError(
            f"current/old shape mismatch {tuple(current_logprob.shape)} != "
            f"{tuple(old_logprob.shape)}"
        )
    if not 0.0 < float(clip_eps) < 1.0:
        raise ValueError("clip_eps must be in (0,1)")
    ratio = torch.exp(current_logprob - old_logprob.to(current_logprob))
    advantage = torch.as_tensor(advantage, device=ratio.device, dtype=ratio.dtype)
    clipped_ratio = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
    objective = torch.minimum(ratio * advantage, clipped_ratio * advantage)
    clipped = (ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps)
    return objective, ratio, clipped


def group_grpo_loss(
    current_logprobs: Sequence[Tensor],
    old_logprobs: Sequence[Tensor],
    rewards: Sequence[float] | Tensor,
    *,
    clip_eps: float = CLIP_EPS,
) -> tuple[Tensor, Tensor]:
    expected = recovery.GROUP_SIZE - 1
    if len(current_logprobs) != expected or len(old_logprobs) != expected:
        raise ValueError(f"GRPO requires {expected} sampled trajectories")
    advantage = normalised_group_advantages(rewards)
    trajectories: list[Tensor] = []
    ratios: list[Tensor] = []
    for arm, (current, old) in enumerate(
        zip(current_logprobs, old_logprobs, strict=True), start=1,
    ):
        if current.numel() == 0:
            raise ValueError(f"arm {arm} has zero replans")
        objective, ratio, _ = clipped_grpo_objective(
            current, old, advantage[arm], clip_eps=clip_eps,
        )
        trajectories.append(objective.mean())
        ratios.append(ratio.reshape(-1))
    return -torch.stack(trajectories).mean(), torch.cat(ratios)


def proposal_switch_balance(logits: Tensor, *, topk: int = C.TOPK) -> Tensor:
    """Existing Switch definition ``M * sum_m f_m P_m`` on proposal logits."""
    if logits.ndim != 2 or logits.shape[0] == 0:
        raise ValueError(f"logits must be non-empty (N,M), got {tuple(logits.shape)}")
    width = int(logits.shape[-1])
    k = min(int(topk), width)
    if k <= 0:
        raise ValueError("topk must be positive")
    hard = logits.detach().float().topk(k, dim=-1).indices
    frequency = torch.zeros_like(logits, dtype=torch.float32).scatter_(1, hard, 1.0)
    frequency = frequency.sum(0) / float(logits.shape[0] * k)
    probability = torch.softmax(logits.float(), dim=-1).mean(0)
    return float(width) * (frequency * probability).sum()


def expert_anchor_objective(
    proposal: nn.Module,
    beliefs: Sequence[Tensor],
    lang: Tensor,
    targets: Sequence[Tensor],
    *,
    temperature: float,
    ce_weight: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    if len(beliefs) != C.DEPTH or len(targets) != C.DEPTH:
        raise ValueError(f"expert anchor requires exactly {C.DEPTH} horizons")
    ce_terms: list[Tensor] = []
    student_logits: list[Tensor] = []
    for horizon in range(C.DEPTH):
        ce, logits = proposal_sparse_ce_loss(
            proposal, beliefs[horizon], lang, targets[horizon],
            temperature=temperature, detach_belief=True,
            return_student_logits=True,
        )
        ce_terms.append(ce)
        student_logits.append(logits)
    sparse_ce = torch.stack(ce_terms).mean()
    switch = proposal_switch_balance(torch.cat(student_logits, dim=0))
    total = float(ce_weight) * sparse_ce + SWITCH_BALANCE_WEIGHT * switch
    return total, {"sparse_ce": sparse_ce, "switch_balance": switch}


def _batched_lang(lang: Tensor, n: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    value = lang.to(device=device, dtype=dtype, non_blocking=True)
    if value.ndim == 2:
        return value.unsqueeze(0).expand(n, -1, -1)
    if value.ndim == 3 and value.shape[0] == 1:
        return value.expand(n, -1, -1)
    if value.ndim == 3 and value.shape[0] == n:
        return value
    raise OutcomeGRPOError(f"language shape {tuple(value.shape)} cannot batch to {n}")


def stored_order_logprob(
    proposal: nn.Module, z: Tensor, lang: Tensor, ordered_support: Tensor,
) -> tuple[Tensor, Tensor]:
    if z.ndim != 3 or ordered_support.ndim != 2 or z.shape[0] != ordered_support.shape[0]:
        raise ValueError(
            f"invalid stored atom z={tuple(z.shape)} order={tuple(ordered_support.shape)}"
        )
    logits = proposal.logits(z, lang)
    if logits.shape[:-1] != ordered_support.shape[:-1]:
        raise ValueError("proposal/order batch mismatch")
    return pl_log_prob(logits.float(), ordered_support.to(torch.int64)), logits


def _load_group(path: Path) -> Mapping[str, Any]:
    try:
        raw = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        raise OutcomeGRPOError(f"cannot load recovery sidecar {path}: {exc}") from exc
    _require(isinstance(raw, Mapping), f"sidecar {path} is not a mapping")
    return raw


@dataclass(frozen=True)
class ValidatedRecoveryCollection:
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
        purpose: str,
    ) -> "ValidatedRecoveryCollection":
        directory = Path(root).expanduser().resolve()
        manifest_path = directory / "manifest.json"
        _require(manifest_path.is_file(), f"recovery manifest missing: {manifest_path}")
        try:
            stat = manifest_path.stat()
            manifest_bytes = manifest_path.read_bytes()
            manifest = json.loads(manifest_bytes)
        except (OSError, ValueError) as exc:
            raise OutcomeGRPOError(f"cannot read recovery manifest: {exc}") from exc
        _require(len(manifest_bytes) == int(stat.st_size),
                 "recovery manifest changed while it was read")
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        _require(isinstance(manifest, dict), "recovery manifest is not a mapping")
        _require(set(manifest) == _MANIFEST_FIELDS,
                 f"manifest fields differ: {sorted(set(manifest) ^ _MANIFEST_FIELDS)}")
        _require(int(manifest["format_version"]) == recovery.FORMAT_VERSION,
                 "recovery manifest version mismatch")
        _require(manifest["kind"] == "loom_outcome_recovery_collection",
                 "input is not a recovery collection")
        split = str(manifest["split"])
        if purpose == "train":
            _require(split in recovery.TRAIN_FOLDS,
                     f"training collection must be train0..train5, got {split!r}")
        elif purpose == "validation":
            _require(split == "validation",
                     f"trust collection must be validation, got {split!r}")
        else:
            raise ValueError(f"unknown collection purpose {purpose!r}")
        summary_value = manifest["summary"]
        _require(isinstance(summary_value, Mapping)
                 and summary_value.get("status") == "COMPLETE"
                 and summary_value.get("complete") is True,
                 "recovery collection is not terminal COMPLETE")
        identity = manifest["identity"]
        _require(isinstance(identity, Mapping), "recovery identity is not a mapping")
        identity_sha = recovery.identity_digest(identity)
        _require(identity_sha == manifest["identity_digest"], "identity digest mismatch")
        source_sha = recovery.source_digest(_ROOT)
        expected_identity = recovery.collection_identity(
            checkpoint=dict(checkpoint_identity), split=split, source_sha256=source_sha,
        )
        _require(identity == expected_identity,
                 "collection identity does not match checkpoint/split/collector source")

        items = tuple(recovery.collection_items(split))
        expected = {recovery.work_key(item): item for item in items}
        rows = manifest["groups"]
        _require(isinstance(rows, list) and len(rows) == len(items),
                 f"collection incomplete: {len(rows) if isinstance(rows, list) else '?'}"
                 f"/{len(items)} groups")
        receipts: list[dict[str, Any]] = []
        for index, (group_id, item) in enumerate(expected.items()):
            raw = rows[index]
            _require(isinstance(raw, Mapping), f"receipt {index} is not a mapping")
            receipt = dict(raw)
            _require(set(receipt) == _RECEIPT_FIELDS,
                     f"receipt fields differ: {sorted(set(receipt) ^ _RECEIPT_FIELDS)}")
            _require(receipt["group_id"] == group_id,
                     f"receipt order mismatch at {index}")
            _require(_valid_sha256(receipt["sha256"]), f"invalid SHA for {group_id}")
            replans = [int(value) for value in receipt["n_replans_by_arm"]]
            rewards = [int(value) for value in receipt["terminal_rewards"]]
            _require(int(receipt["n_arms"]) == recovery.GROUP_SIZE,
                     f"wrong arm count for {group_id}")
            _require(len(replans) == len(rewards) == recovery.GROUP_SIZE,
                     f"wrong receipt vectors for {group_id}")
            _require(all(value > 0 for value in replans), f"zero replans in {group_id}")
            _require(all(value in (0, 1) for value in rewards),
                     f"invalid terminal reward in {group_id}")
            path = cls._sidecar_path(directory, receipt)
            _require(path.is_file() and path.stat().st_size == int(receipt["size"]),
                     f"sidecar missing/size changed: {path}")
            _require(recovery.sha256_file(path) == receipt["sha256"],
                     f"sidecar hash changed: {path}")
            payload = _load_group(path)
            recovery.validate_group_payload(
                payload, item=item, expected_identity_digest=identity_sha,
                expected_split=split,
            )
            cls._match_receipt(payload, receipt)
            receipts.append(receipt)

        terminal = [sum(int(row["terminal_rewards"][arm]) for row in receipts)
                    for arm in range(recovery.GROUP_SIZE)]
        replans = [sum(int(row["n_replans_by_arm"][arm]) for row in receipts)
                   for arm in range(recovery.GROUP_SIZE)]
        summary = {
            "status": "COMPLETE", "complete": True,
            "n_groups": len(items), "n_expected_groups": len(items),
            "n_trajectories": len(items) * recovery.GROUP_SIZE,
            "n_expected_trajectories": len(items) * recovery.GROUP_SIZE,
            "terminal_successes_by_arm": terminal, "replans_by_arm": replans,
        }
        _require(manifest["summary"] == summary,
                 "COMPLETE summary does not match sidecar receipts")
        final_stat = manifest_path.stat()
        _require(int(final_stat.st_size) == int(stat.st_size)
                 and int(final_stat.st_mtime_ns) == int(stat.st_mtime_ns)
                 and recovery.sha256_file(manifest_path) == manifest_sha256,
                 "recovery manifest changed during authentication")
        return cls(
            root=directory, manifest_path=manifest_path, manifest=_json_copy(manifest),
            manifest_sha256=manifest_sha256,
            manifest_size=int(stat.st_size), manifest_mtime_ns=int(stat.st_mtime_ns),
            split=split, identity_digest=identity_sha, items=items,
            receipts=tuple(receipts),
        )

    @staticmethod
    def _sidecar_path(root: Path, receipt: Mapping[str, Any]) -> Path:
        rel = Path(str(receipt.get("sidecar") or ""))
        _require(not rel.is_absolute() and ".." not in rel.parts,
                 "sidecar path escapes collection")
        path = (root / rel).resolve()
        _require(path.parent == (root / "groups").resolve(),
                 "sidecar is not directly inside groups/")
        return path

    @staticmethod
    def _match_receipt(payload: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
        arms = list(payload["arms"])
        replans = [int(arm["z"].shape[0]) for arm in arms]
        rewards = [int(float(arm["terminal_reward"])) for arm in arms]
        _require(replans == [int(value) for value in receipt["n_replans_by_arm"]],
                 f"sidecar replan counts differ for {receipt['group_id']}")
        _require(rewards == [int(value) for value in receipt["terminal_rewards"]],
                 f"sidecar rewards differ for {receipt['group_id']}")

    def assert_unchanged(self) -> None:
        stat = self.manifest_path.stat()
        _require(int(stat.st_size) == self.manifest_size,
                 "manifest size changed during training")
        _require(int(stat.st_mtime_ns) == self.manifest_mtime_ns,
                 "manifest mtime changed during training")
        _require(recovery.sha256_file(self.manifest_path) == self.manifest_sha256,
                 "manifest bytes changed during training")

    def load(self, index: int) -> Mapping[str, Any]:
        self.assert_unchanged()
        receipt = self.receipts[index]
        path = self._sidecar_path(self.root, receipt)
        _require(path.is_file() and path.stat().st_size == int(receipt["size"]),
                 f"sidecar changed: {path}")
        _require(recovery.sha256_file(path) == receipt["sha256"],
                 f"sidecar hash changed: {path}")
        payload = _load_group(path)
        recovery.validate_group_payload(
            payload, item=self.items[index], expected_identity_digest=self.identity_digest,
            expected_split=self.split,
        )
        self._match_receipt(payload, receipt)
        return payload

    def provenance(self) -> dict[str, Any]:
        return {
            "path": str(self.root), "manifest": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "identity_digest": self.identity_digest, "split": self.split,
            "n_groups": len(self.receipts),
            "n_trajectories": len(self.receipts) * recovery.GROUP_SIZE,
            "terminal_successes_by_arm": list(
                self.manifest["summary"]["terminal_successes_by_arm"]
            ),
            "replans_by_arm": list(self.manifest["summary"]["replans_by_arm"]),
            "collector_source": _json_copy(self.manifest["identity"]["source"]),
        }


def _load_parent(path: str | os.PathLike[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = recovery.authenticate_seed_checkpoint(path)
    try:
        payload = torch.load(identity["path"], map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        raise OutcomeGRPOError(f"cannot load authenticated parent: {exc}") from exc
    _require(isinstance(payload, dict), "parent checkpoint is not a mapping")
    _require(int(payload.get("global_step", -1)) == recovery.SEED_GLOBAL_STEP,
             "parent embedded step mismatch")
    _require(str(payload.get("config_hash") or "") == recovery.SEED_CONFIG_HASH,
             "parent embedded config hash mismatch")
    cfg = payload.get("resolved_config")
    _require(isinstance(cfg, dict) and _config_hash(cfg) == payload["config_hash"],
             "parent resolved config is unauthenticated")
    state = payload.get("model")
    _require(isinstance(state, dict) and state and
             all(isinstance(value, Tensor) for value in state.values()),
             "parent model is not a flat tensor mapping")
    _require(_all_finite(state.values()), "parent model contains nan/inf")
    _assert_parent_unchanged(identity)
    return payload, identity


def _assert_parent_unchanged(identity: Mapping[str, Any]) -> None:
    path = Path(str(identity["path"]))
    try:
        stat = path.stat()
    except OSError as exc:
        raise OutcomeGRPOError(f"parent checkpoint became unavailable: {exc}") from exc
    _require(int(stat.st_size) == int(identity["size"])
             and int(stat.st_mtime_ns) == int(identity["mtime_ns"]),
             "parent checkpoint stat changed after authentication")
    _require(recovery.sha256_file(path) == identity["sha256"],
             "parent checkpoint bytes changed after authentication")


def _load_proposal(parent: Mapping[str, Any], device: torch.device) -> Proposal:
    kwargs = dict(parent["resolved_config"].get("model", {}).get("proposal", {}) or {})
    proposal = Proposal(**kwargs)
    state = submodule_state(parent["model"], "proposal")
    _require(state is not None, "parent has no proposal tensors")
    try:
        proposal.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise OutcomeGRPOError(f"parent proposal state mismatch: {exc}") from exc
    return proposal.to(device)


@torch.no_grad()
def authenticate_behaviour_policy(
    proposal: nn.Module,
    collection: ValidatedRecoveryCollection,
    *,
    device: torch.device,
    chunk_replans: int,
) -> dict[str, Any]:
    proposal.eval()
    dtype = next(proposal.parameters()).dtype
    max_lp_error = max_coeff_error = 0.0
    sampled_atoms = all_atoms = 0
    for group_index in range(len(collection.receipts)):
        payload = collection.load(group_index)
        for arm_index, arm in enumerate(payload["arms"]):
            n = int(arm["z"].shape[0])
            lang = _batched_lang(arm["lang"], n, device, dtype)
            for lo in range(0, n, chunk_replans):
                hi = min(lo + chunk_replans, n)
                z = arm["z"][lo:hi].to(device=device, dtype=dtype, non_blocking=True)
                order = arm["ordered_support"][lo:hi].to(device=device)
                current, logits = stored_order_logprob(proposal, z, lang[lo:hi], order)
                expected_coeff = arm["coeff"][lo:hi].to(device=device).float()
                current_coeff = weights_from_logits(
                    logits.float(), order.to(torch.int64), logits.shape[-1],
                ).float()
                coeff_error = float((current_coeff - expected_coeff).abs().max())
                max_coeff_error = max(max_coeff_error, coeff_error)
                _require(torch.allclose(
                    current_coeff, expected_coeff,
                    atol=BEHAVIOUR_COEFF_ATOL, rtol=BEHAVIOUR_COEFF_RTOL,
                ), f"parent coefficient mismatch in {payload['group_id']} arm {arm_index}")
                if arm_index > 0:
                    expected_lp = arm["old_logprob"][lo:hi].to(device=device).float()
                    lp_error = float((current.float() - expected_lp).abs().max())
                    max_lp_error = max(max_lp_error, lp_error)
                    _require(torch.allclose(
                        current.float(), expected_lp,
                        atol=BEHAVIOUR_LOGPROB_ATOL, rtol=BEHAVIOUR_LOGPROB_RTOL,
                    ), f"parent old-logprob mismatch in {payload['group_id']} arm {arm_index}")
                    sampled_atoms += hi - lo
                all_atoms += hi - lo
    return {
        "passed": True, "all_atoms": all_atoms,
        "ratio_eligible_atoms": sampled_atoms, "arm0_ratio_eligible_atoms": 0,
        "max_abs_old_logprob_error": max_lp_error,
        "max_abs_coeff_error": max_coeff_error,
        "logprob_atol": BEHAVIOUR_LOGPROB_ATOL,
        "logprob_rtol": BEHAVIOUR_LOGPROB_RTOL,
        "coeff_atol": BEHAVIOUR_COEFF_ATOL, "coeff_rtol": BEHAVIOUR_COEFF_RTOL,
    }


def _call_estimator(module: nn.Module, feats: Mapping[str, Tensor], z: Tensor | None,
                    embodiment: str) -> Tensor:
    try:
        accepts = "embodiment" in inspect.signature(module.forward).parameters
    except (TypeError, ValueError):
        accepts = False
    return module(feats, z, embodiment=embodiment) if accepts else module(feats, z)


def _call_q_action(module: nn.Module, actions: Tensor, z: Tensor,
                   embodiment: str) -> Tensor:
    try:
        accepts = "embodiment" in inspect.signature(module.forward).parameters
    except (TypeError, ValueError):
        accepts = False
    return module(actions, z, embodiment=embodiment) if accepts else module(actions, z)


@dataclass
class ExpertAnchor:
    proposal: nn.Module
    estimator: nn.Module
    q_action: nn.Module
    sampler: Any
    device: torch.device
    parent_global_step: int
    temperature: float
    weight: float
    data_provenance: dict[str, Any]
    gate_cache: list[tuple[list[Tensor], Tensor, list[Tensor], str]] = field(
        default_factory=list,
    )

    @classmethod
    def from_parent(
        cls, parent: Mapping[str, Any], proposal: nn.Module, *, device: torch.device,
    ) -> "ExpertAnchor":
        parent_cfg = parent["resolved_config"]
        cfg = copy.deepcopy(parent_cfg)
        data = cfg.setdefault("data", {})
        data.update({
            "source": "libero", "embodiments": ["libero_franka"],
            "action_free": False, "sampling": "uniform_task",
            "trajectory_split": "train", "holdout_demo_keys": ["demo_49"],
            "recurrent_burn_in": 4,
        })
        proposal_cfg = dict(cfg.get("losses", {}).get("proposal", {}) or {})
        if (not bool(proposal_cfg.get("enabled", False))
                or proposal_cfg.get("mode") != "sparse_ce"
                or not bool(proposal_cfg.get("detach_belief", False))):
            raise ExpertAnchorUnavailable(
                "parent does not define the existing detached sparse-CE proposal path"
            )
        temperature = float(proposal_cfg.get("temperature", 1.0))
        weight = float(proposal_cfg.get("weight", 0.0))
        if not math.isfinite(temperature) or temperature <= 0 or weight != 1.0:
            raise ExpertAnchorUnavailable("expert-anchor temperature/weight contract mismatch")
        try:
            from loom.model.estimator import Estimator  # noqa: PLC0415
            from loom.heads.q_action import QAction  # noqa: PLC0415
            from loom.train.loop import build_sampler  # noqa: PLC0415

            estimator_kwargs = dict(parent_cfg.get("model", {}).get("estimator", {}) or {})
            estimator_kwargs.setdefault("embodiments", ["libero_franka"])
            q_kwargs = dict(parent_cfg.get("model", {}).get("q_action", {}) or {})
            q_kwargs.pop("embodiments", None)
            q_kwargs.pop("default_embodiment", None)
            estimator = Estimator(**estimator_kwargs)
            q_action = QAction(
                embodiments=["libero_franka"], default_embodiment="libero_franka",
                **q_kwargs,
            )
            estimator_state = submodule_state(parent["model"], "estimator")
            q_state = submodule_state(parent["model"], "q_action")
            if estimator_state is None or q_state is None:
                raise ExpertAnchorUnavailable("parent lacks estimator/q_action tensors")
            estimator.load_state_dict(estimator_state, strict=True)
            q_action.load_state_dict(q_state, strict=True)
            estimator.to(device).eval().requires_grad_(False)
            q_action.to(device).eval().requires_grad_(False)
            sampler = build_sampler(
                cfg, rank=0, world=1,
                seed=int(cfg.get("run", {}).get("seed", TRAIN_SEED)),
                device=str(device),
            )
        except ExpertAnchorUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ExpertAnchorUnavailable(f"cannot build exact expert anchor: {exc}") from exc
        try:
            manifest = sampler.trajectory_manifest()
        except Exception as exc:  # noqa: BLE001
            raise ExpertAnchorUnavailable(f"anchor trajectory manifest unavailable: {exc}") from exc
        for key in ("digest", "n_tasks", "n_trajectories"):
            if manifest.get(key) != ANCHOR_MANIFEST[key]:
                raise ExpertAnchorUnavailable(
                    f"anchor manifest {key}={manifest.get(key)!r} != {ANCHOR_MANIFEST[key]!r}"
                )
        if int(getattr(sampler, "n_windows", -1)) != ANCHOR_MANIFEST["n_windows"]:
            raise ExpertAnchorUnavailable("anchor window count mismatch")
        provenance = {
            "source": "libero", "trajectory_split": "train",
            "holdout_demo_keys": ["demo_49"], "sampling": "uniform_task",
            "recurrent_burn_in": 4, "manifest": manifest,
            "n_windows": int(sampler.n_windows), "batch_size": int(data["batch_per_gpu"]),
            "sampler_seed": int(cfg.get("run", {}).get("seed", TRAIN_SEED)),
        }
        return cls(
            proposal=proposal, estimator=estimator, q_action=q_action, sampler=sampler,
            device=device, parent_global_step=int(parent["global_step"]),
            temperature=temperature, weight=weight, data_provenance=provenance,
        )

    def _prepare_step(self, step: int) -> tuple[list[Tensor], Tensor, list[Tensor], str]:
        try:
            from loom.train.loop import _to_device  # noqa: PLC0415

            window = self.sampler.next(int(step))
            dtype = next(self.proposal.parameters()).dtype
            window = _to_device(window, str(self.device), dtype=dtype)
        except Exception as exc:  # noqa: BLE001
            raise ExpertAnchorUnavailable(f"cannot read anchor step {step}: {exc}") from exc
        meta = dict(window.get("data_meta") or {})
        manifest = self.data_provenance["manifest"]
        for key, expected in {
            "source": "libero", "split": "train", "manifest_digest": manifest["digest"],
        }.items():
            if meta.get(key) != expected:
                raise ExpertAnchorUnavailable(
                    f"anchor batch {key}={meta.get(key)!r} != {expected!r}"
                )
        actions = window.get("actions")
        feats = list(window.get("feats") or ())
        lang = window.get("lang")
        embodiment = str(window.get("embodiment") or "")
        if not isinstance(actions, Tensor) or actions.ndim != 4:
            raise ExpertAnchorUnavailable("anchor batch lacks action segments")
        if len(feats) < C.DEPTH or not isinstance(lang, Tensor) or not embodiment:
            raise ExpertAnchorUnavailable("anchor batch lacks states/language/embodiment")
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
                raise ExpertAnchorUnavailable(f"q_action target h{horizon + 1} shape mismatch")
            if (not bool(torch.isfinite(target).all()) or bool((target < 0).any())
                    or not torch.allclose(
                        target.float().sum(-1),
                        torch.ones(target.shape[0], device=target.device),
                        atol=1e-4, rtol=0,
                    )):
                raise ExpertAnchorUnavailable(f"q_action target h{horizon + 1} is invalid")
        return beliefs[:C.DEPTH], lang, targets, embodiment

    def loss(self, update_index: int) -> tuple[Tensor, dict[str, float]]:
        step = self.parent_global_step + 1 + int(update_index)
        beliefs, lang, targets, _ = self._prepare_step(step)
        total, terms = expert_anchor_objective(
            self.proposal, beliefs, lang, targets,
            temperature=self.temperature, ce_weight=self.weight,
        )
        if not bool(torch.isfinite(total)):
            raise OutcomeGRPOError(f"expert anchor nonfinite at update {update_index}")
        return total, {
            "sparse_ce": float(terms["sparse_ce"].detach()),
            "switch_balance": float(terms["switch_balance"].detach()),
            "total": float(total.detach()),
        }

    def cache_gate(self, training_updates: int) -> dict[str, Any]:
        if self.gate_cache:
            raise OutcomeGRPOError("expert gate was cached twice")
        first = self.parent_global_step + int(training_updates) + 1
        for offset in range(EXPERT_GATE_BATCHES):
            self.gate_cache.append(self._prepare_step(first + offset))
        metrics = self.evaluate_gate()
        return {
            "first_step": first, "last_step": first + EXPERT_GATE_BATCHES - 1,
            "n_batches": EXPERT_GATE_BATCHES, **metrics,
        }

    @torch.no_grad()
    def evaluate_gate(self) -> dict[str, Any]:
        if len(self.gate_cache) != EXPERT_GATE_BATCHES:
            raise OutcomeGRPOError("expert gate cache is incomplete")
        was_training = self.proposal.training
        self.proposal.eval()
        ce_sum = overlap_sum = 0.0
        examples = 0
        digest = hashlib.sha256()
        for batch_index, (beliefs, lang, targets, embodiment) in enumerate(self.gate_cache):
            _update_tensor_hash(digest, f"{batch_index}/lang", lang)
            digest.update(embodiment.encode("utf-8") + b"\0")
            for horizon in range(C.DEPTH):
                _update_tensor_hash(digest, f"{batch_index}/z{horizon}", beliefs[horizon])
                _update_tensor_hash(digest, f"{batch_index}/target{horizon}", targets[horizon])
                ce, logits = proposal_sparse_ce_loss(
                    self.proposal, beliefs[horizon], lang, targets[horizon],
                    temperature=self.temperature, detach_belief=True,
                    reduction="sum", return_student_logits=True,
                )
                batch = int(logits.shape[0])
                student = logits.float().topk(C.TOPK, dim=-1).indices
                teacher = targets[horizon].float().topk(C.TOPK, dim=-1).indices
                overlap = (student.unsqueeze(-1) == teacher.unsqueeze(-2)).any(-1)
                ce_sum += float(ce)
                overlap_sum += float(overlap.float().sum()) / C.TOPK
                examples += batch
        self.proposal.train(was_training)
        return {
            "sparse_ce": ce_sum / examples,
            "topk_overlap": overlap_sum / examples,
            "target_sha256": digest.hexdigest(),
            "n_horizon_examples": examples,
        }

    def unexpected_gradients(self) -> list[str]:
        out: list[str] = []
        for prefix, module in (("estimator", self.estimator), ("q_action", self.q_action)):
            out.extend(f"{prefix}.{name}" for name, parameter in module.named_parameters()
                       if parameter.grad is not None)
        return out


def _proposal_grad_health(proposal: nn.Module) -> tuple[float, list[str], list[str]]:
    square = 0.0
    missing: list[str] = []
    nonfinite: list[str] = []
    for name, parameter in proposal.named_parameters():
        if parameter.grad is None:
            missing.append(name)
        elif not bool(torch.isfinite(parameter.grad).all()):
            nonfinite.append(name)
        else:
            square += float(parameter.grad.detach().float().square().sum())
    return math.sqrt(square), missing, nonfinite


def _optimizer_finite(optimizer: torch.optim.Optimizer) -> bool:
    return all(
        bool(torch.isfinite(value).all())
        for state in optimizer.state.values() for value in state.values()
        if isinstance(value, Tensor)
    )


def _train_one_group(
    proposal: nn.Module,
    payload: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    anchor: ExpertAnchor,
    *,
    update_index: int,
    device: torch.device,
    chunk_replans: int,
) -> dict[str, float]:
    optimizer.zero_grad(set_to_none=True)
    rewards = [float(arm["terminal_reward"]) for arm in payload["arms"]]
    advantage = normalised_group_advantages(rewards).to(device)
    dtype = next(proposal.parameters()).dtype
    grpo_value = ratio_sum = 0.0
    ratio_count = clipped_count = 0
    sampled_arms = recovery.GROUP_SIZE - 1
    for arm_index in range(1, recovery.GROUP_SIZE):
        arm = payload["arms"][arm_index]
        n = int(arm["z"].shape[0])
        lang = _batched_lang(arm["lang"], n, device, dtype)
        for lo in range(0, n, chunk_replans):
            hi = min(lo + chunk_replans, n)
            z = arm["z"][lo:hi].to(device=device, dtype=dtype, non_blocking=True)
            order = arm["ordered_support"][lo:hi].to(device=device)
            old = arm["old_logprob"][lo:hi].to(device=device).float()
            current, _ = stored_order_logprob(proposal, z, lang[lo:hi], order)
            objective, ratio, clipped = clipped_grpo_objective(
                current, old, advantage[arm_index], clip_eps=CLIP_EPS,
            )
            contribution = -objective.sum() / float(n * sampled_arms)
            if not bool(torch.isfinite(contribution)) or not bool(torch.isfinite(ratio).all()):
                raise OutcomeGRPOError(
                    f"nonfinite GRPO in {payload['group_id']} arm {arm_index}"
                )
            contribution.backward()
            grpo_value += float(contribution.detach())
            ratio_sum += float(ratio.detach().sum())
            ratio_count += int(ratio.numel())
            clipped_count += int(clipped.sum())

    anchor_loss, anchor_metrics = anchor.loss(update_index)
    anchor_loss.backward()
    grad_norm, missing, nonfinite = _proposal_grad_health(proposal)
    if missing or nonfinite or not math.isfinite(grad_norm):
        raise OutcomeGRPOError(
            f"proposal gradient health failed at {update_index}: "
            f"missing={missing[:8]} nonfinite={nonfinite[:8]}"
        )
    unexpected = anchor.unexpected_gradients()
    if unexpected:
        raise OutcomeGRPOError(f"frozen modules received gradients: {unexpected[:8]}")
    clipped_norm = torch.nn.utils.clip_grad_norm_(
        proposal.parameters(), GRAD_CLIP, error_if_nonfinite=True,
    )
    optimizer.step()
    if not _all_finite(proposal.parameters()) or not _optimizer_finite(optimizer):
        raise OutcomeGRPOError(f"proposal/optimizer nonfinite after update {update_index}")
    return {
        "grpo_loss": grpo_value,
        "anchor_sparse_ce": anchor_metrics["sparse_ce"],
        "switch_balance": anchor_metrics["switch_balance"],
        "anchor_total": anchor_metrics["total"],
        "grad_norm_preclip": float(clipped_norm),
        "grad_clip": GRAD_CLIP,
        "ratio_mean": ratio_sum / max(1, ratio_count),
        "clip_fraction": clipped_count / max(1, ratio_count),
        "ratio_atoms": float(ratio_count),
    }


@torch.no_grad()
def evaluate_trust_gates(
    proposal: nn.Module,
    validation: Any,
    *,
    device: torch.device,
    chunk_replans: int,
    anchor_initial: Mapping[str, Any],
    anchor_final: Mapping[str, Any],
    training_nonfinite: int = 0,
    unexpected_gradients: Sequence[str] = (),
) -> dict[str, Any]:
    proposal.eval()
    dtype = next(proposal.parameters()).dtype
    ratio_sum = ratio_square_sum = 0.0
    ratio_atoms = clipped_atoms = 0
    drift: list[Tensor] = []
    usage = torch.zeros(C.M, dtype=torch.float64)
    usage_atoms = final_nonfinite = 0
    for group_index in range(len(validation.receipts)):
        payload = validation.load(group_index)
        for arm_index, arm in enumerate(payload["arms"]):
            n = int(arm["z"].shape[0])
            lang = _batched_lang(arm["lang"], n, device, dtype)
            for lo in range(0, n, chunk_replans):
                hi = min(lo + chunk_replans, n)
                z = arm["z"][lo:hi].to(device=device, dtype=dtype, non_blocking=True)
                if arm_index == 0:
                    coeff = argmax_coeff(
                        proposal.logits(z, lang[lo:hi]).float(), C.TOPK, C.M,
                    ).float()
                    baseline = arm["coeff"][lo:hi].to(device=device).float()
                    if not bool(torch.isfinite(coeff).all()):
                        final_nonfinite += int((~torch.isfinite(coeff)).sum())
                    else:
                        # Locked contract: coefficient drift is L1, not L2.
                        drift.append((coeff - baseline).abs().sum(-1).cpu())
                        usage += coeff.double().sum(0).cpu()
                        usage_atoms += int(coeff.shape[0])
                    continue
                order = arm["ordered_support"][lo:hi].to(device=device)
                current, _ = stored_order_logprob(proposal, z, lang[lo:hi], order)
                old = arm["old_logprob"][lo:hi].to(device=device).float()
                ratio = torch.exp(current.double() - old.double())
                finite = torch.isfinite(ratio)
                if not bool(finite.all()):
                    final_nonfinite += int((~finite).sum())
                    continue
                ratio_sum += float(ratio.sum())
                ratio_square_sum += float(ratio.square().sum())
                ratio_atoms += int(ratio.numel())
                clipped_atoms += int(((ratio < 1 - CLIP_EPS) | (ratio > 1 + CLIP_EPS)).sum())
    validation.assert_unchanged()
    _require(ratio_atoms > 0 and usage_atoms > 0 and drift,
             "validation trust gate observed no atoms")
    clip_fraction = clipped_atoms / ratio_atoms
    ess_fraction = ((ratio_sum * ratio_sum) /
                    max(ratio_square_sum * ratio_atoms, torch.finfo(torch.float64).tiny))
    drift_values = torch.cat(drift).float()
    drift_p95 = float(torch.quantile(drift_values, 0.95, interpolation="linear"))
    live_ops = int(((usage / usage_atoms) > 1e-4).sum())
    total_nonfinite = int(training_nonfinite) + final_nonfinite
    target_same = anchor_initial["target_sha256"] == anchor_final["target_sha256"]
    ce_initial = float(anchor_initial["sparse_ce"])
    ce_final = float(anchor_final["sparse_ce"])
    overlap_initial = float(anchor_initial["topk_overlap"])
    overlap_final = float(anchor_final["topk_overlap"])
    overlap_decline = overlap_initial - overlap_final
    checks = {
        "clip_fraction": {"value": clip_fraction, "op": "<=",
                          "threshold": MAX_CLIP_FRACTION,
                          "pass": clip_fraction <= MAX_CLIP_FRACTION},
        "ess_fraction": {"value": ess_fraction, "op": ">=",
                         "threshold": MIN_ESS_FRACTION,
                         "pass": ess_fraction >= MIN_ESS_FRACTION},
        "coeff_drift_p95_l1": {"value": drift_p95, "op": "<=",
                               "threshold": MAX_COEFF_DRIFT_P95_L1,
                               "pass": drift_p95 <= MAX_COEFF_DRIFT_P95_L1},
        "live_ops": {"value": live_ops, "op": ">=", "threshold": MIN_LIVE_OPS,
                     "pass": live_ops >= MIN_LIVE_OPS},
        "expert_target_identity": {"value": target_same, "op": "==",
                                   "threshold": True, "pass": target_same},
        "expert_sparse_ce_no_worsening": {"value": ce_final, "op": "<=",
                                          "threshold": ce_initial,
                                          "pass": ce_final <= ce_initial},
        "expert_topk_overlap_decline": {"value": overlap_decline, "op": "<=",
                                        "threshold": MAX_TOPK_OVERLAP_DECLINE,
                                        "pass": overlap_decline <= MAX_TOPK_OVERLAP_DECLINE},
        "nonfinite": {"value": total_nonfinite, "op": "==", "threshold": 0,
                      "pass": total_nonfinite == 0},
        "unexpected_gradients": {"value": len(unexpected_gradients), "op": "==",
                                 "threshold": 0,
                                 "pass": len(unexpected_gradients) == 0},
    }
    return {
        "passed": all(bool(row["pass"]) for row in checks.values()),
        "checks": checks,
        "definitions": {
            "scope": "disjoint recovery validation split",
            "clip_fraction": "arm1..7 final/old ratios outside [0.8,1.2]",
            "ess_fraction": "(sum r)^2/(N*sum r^2), arm1..7 replans",
            "coeff_drift_p95_l1": (
                "linear p95 of L1(final deployed argmax coeff - stored arm0 coeff)"
            ),
            "live_ops": "mean final deployed coeff on validation arm0 states >1e-4",
            "arm0_importance_ratios": 0,
            "expert_gate": "same cached post-training-range demo beliefs/targets",
        },
        "expert_anchor": {"initial": dict(anchor_initial), "final": dict(anchor_final)},
        "counts": {
            "ratio_atoms": ratio_atoms, "clipped_atoms": clipped_atoms,
            "arm0_drift_atoms": int(drift_values.numel()),
            "arm0_usage_atoms": usage_atoms,
            "training_nonfinite": int(training_nonfinite),
            "final_nonfinite": int(final_nonfinite),
        },
    }


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite descendant: {path}")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=str(path.parent))
    published = False
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        published = True
        parent_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        try:
            os.unlink(path if published else tmp_name)
        except FileNotFoundError:
            pass
        raise


def _source_identity() -> dict[str, Any]:
    digest = hashlib.sha256()
    rows: dict[str, str] = {}
    for rel in _TRAINER_SOURCE_FILES:
        path = _ROOT / rel
        _require(path.is_file(), f"trainer source missing: {path}")
        sha = recovery.sha256_file(path)
        rows[rel] = sha
        digest.update(rel.encode("utf-8") + b"\0" + bytes.fromhex(sha) + b"\0")
    return {"scheme": "sha256(path-nul-sha256-nul)-v1",
            "sha256": digest.hexdigest(), "files": rows}


def _authenticate_round0_optimizer(
    optimizer: torch.optim.Optimizer,
    proposal: nn.Module,
    *,
    optimizer_steps: int,
) -> None:
    _require(int(optimizer_steps) == EPOCHS * EXPECTED_TRAIN_GROUPS == 400,
             f"descendant requires exactly 400 optimizer steps, got {optimizer_steps}")
    _require(type(optimizer) is torch.optim.AdamW,
             "descendant optimizer is not exact torch.optim.AdamW")
    _require(len(optimizer.param_groups) == 1,
             "proposal optimizer must have exactly one parameter group")
    group = optimizer.param_groups[0]
    expected_hparams = {
        "lr": LEARNING_RATE, "betas": ADAMW_BETAS,
        "weight_decay": ADAMW_WEIGHT_DECAY, "eps": ADAMW_EPS,
    }
    for key, expected in expected_hparams.items():
        _require(group.get(key) == expected,
                 f"proposal optimizer {key}={group.get(key)!r} != {expected!r}")
    parameters = list(proposal.parameters())
    _require(tuple(map(id, group["params"])) == tuple(map(id, parameters)),
             "optimizer parameter group is not exactly the proposal")
    _require({id(parameter) for parameter in optimizer.state}
             == {id(parameter) for parameter in parameters},
             "proposal optimizer state is incomplete or contains foreign parameters")
    for parameter in parameters:
        state = optimizer.state[parameter]
        step = state.get("step")
        _require(isinstance(step, Tensor) and step.numel() == 1
                 and int(float(step.detach().cpu())) == int(optimizer_steps),
                 "proposal optimizer state does not prove a reset 400-step stage")
    _require(_optimizer_finite(optimizer), "proposal optimizer state contains nan/inf")


def write_descendant_checkpoint(
    out: str | os.PathLike[str],
    *,
    parent: Mapping[str, Any],
    parent_identity: Mapping[str, Any],
    proposal: nn.Module,
    optimizer: torch.optim.Optimizer,
    optimizer_steps: int,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    out_path = Path(out).expanduser().resolve()
    _authenticate_round0_optimizer(
        optimizer, proposal, optimizer_steps=int(optimizer_steps),
    )
    training_provenance = provenance.get("training")
    trust_provenance = provenance.get("trust_gate")
    _require(isinstance(training_provenance, Mapping)
             and int(training_provenance.get("groups_per_epoch", -1))
             == EXPECTED_TRAIN_GROUPS
             and int(training_provenance.get("optimizer_steps", -1))
             == int(optimizer_steps),
             "training provenance does not authenticate the exact 200x2 stage")
    _require(isinstance(trust_provenance, Mapping)
             and trust_provenance.get("passed") is True,
             "refusing descendant without a passed terminal trust gate")
    parent_state = parent["model"]
    frozen_before = frozen_model_digest(parent_state)
    initial_proposal = proposal_model_digest(parent_state)
    trained = {name: value.detach().cpu().clone()
               for name, value in proposal.state_dict().items()}
    state = dict(parent_state)
    expected = {name[len("proposal."):] for name in state if name.startswith("proposal.")}
    _require(set(trained) == expected, "trained proposal keys differ from parent")
    for name, value in trained.items():
        key = f"proposal.{name}"
        _require(value.shape == state[key].shape, f"proposal shape changed: {key}")
        state[key] = value
    _require(frozen_model_digest(state) == frozen_before,
             "frozen tensors changed before save")
    final_proposal = proposal_model_digest(state)

    resolved = copy.deepcopy(parent["resolved_config"])
    _require("outcome_grpo" not in resolved and "outcome_grpo_round0" not in resolved,
             "parent already contains an ambiguous outcome-GRPO recipe")
    resolved["outcome_grpo"] = {
        "format_version": FORMAT_VERSION,
        "algorithm": "stored_order_pl_clipped_grpo",
        "reward_normalisation": "complete-group population mean/std",
        "aggregation": "mean_replans_then_mean_sampled_trajectories",
        "arm0": "normalisation/control only; no ratio or gradient",
        "sampled_arms": list(range(1, recovery.GROUP_SIZE)),
        "clip_eps": CLIP_EPS, "epochs": EPOCHS,
        "groups_per_epoch": provenance["training"]["groups_per_epoch"],
        "optimizer_steps": int(optimizer_steps),
        "optimizer": {
            "kind": "AdamW", "lr": LEARNING_RATE, "betas": list(ADAMW_BETAS),
            "weight_decay": ADAMW_WEIGHT_DECAY, "eps": ADAMW_EPS,
            "scheduler": None, "grad_clip": GRAD_CLIP,
            "proposal_state_reset_at_entry": True,
        },
        "objectives": {"grpo": 1.0, "sparse_ce": 1.0,
                       "switch_balance": SWITCH_BALANCE_WEIGHT},
        "train_collection_identity": provenance["train_collection"]["identity_digest"],
        "validation_collection_identity": (
            provenance["validation_collection"]["identity_digest"]
        ),
        "trust_thresholds": {
            "max_clip_fraction": MAX_CLIP_FRACTION,
            "min_ess_fraction": MIN_ESS_FRACTION,
            "max_coeff_drift_p95_l1": MAX_COEFF_DRIFT_P95_L1,
            "min_live_ops": MIN_LIVE_OPS,
            "expert_sparse_ce_no_worsening": True,
            "max_topk_overlap_decline": MAX_TOPK_OVERLAP_DECLINE,
            "nonfinite": 0,
        },
    }
    config_sha = _config_hash(resolved)
    top = _json_copy(provenance)
    top.update({
        "format_version": FORMAT_VERSION, "kind": TRAINER_KIND,
        "created_utc": _utc(), "parent": dict(parent_identity),
        "parent_config_hash": parent["config_hash"],
        "descendant_config_hash": config_sha,
        "parent_global_step": int(parent["global_step"]),
        "descendant_global_step": int(parent["global_step"]) + int(optimizer_steps),
        "optimizer_steps": int(optimizer_steps),
        "mutated_model_prefixes": ["proposal."],
        "frozen_model": frozen_before, "initial_proposal": initial_proposal,
        "final_proposal": final_proposal,
        "parent_consolidated": _json_copy(parent.get("consolidated", {})),
        "parent_samples_seen": int(parent.get("samples_seen", 0)),
        "sample_counter_semantics": (
            "top-level samples_seen is inherited; this offline stage counts "
            "optimizer updates and complete recovery groups in its own provenance"
        ),
    })
    descendant_step = int(parent["global_step"]) + int(optimizer_steps)
    consolidated = {
        "tool": "loom.train.outcome_grpo_round0",
        "created_utc": top["created_utc"], "step": descendant_step,
        "section": "model", "n_keys": len(state),
        "n_params": sum(int(value.numel()) for value in state.values()),
        "source_parent": dict(parent_identity), "proposal_only_update": True,
        "frozen_model_sha256": frozen_before["sha256"],
        "optimizer": "proposal-only AdamW state included; no scheduler/sampler/RNG",
    }
    payload = dict(parent)
    payload.update({
        "model": state,
        "global_step": descendant_step,
        "config_hash": config_sha, "resolved_config": resolved,
        "optimizer": {
            "kind": "proposal_only_adamw",
            "parameter_names": [name for name, _ in proposal.named_parameters()],
            "state_dict": _cpu_tree(optimizer.state_dict()),
            "state_reset_at_entry": True,
        },
        "outcome_grpo": top,
        "consolidated": consolidated,
        "world_size": 1,
        "stop_reason": "outcome_grpo_round0_terminal_trust_pass",
        "wandb_run_id": None,
    })
    _atomic_torch_save(out_path, payload)
    try:
        reloaded = torch.load(out_path, map_location="cpu", weights_only=False)
        _require(_config_hash(reloaded["resolved_config"]) == reloaded["config_hash"],
                 "reloaded config authentication failed")
        frozen_after = frozen_model_digest(reloaded["model"])
        proposal_after = proposal_model_digest(reloaded["model"])
        _require(frozen_after == frozen_before,
                 "frozen checkpoint tensors changed across save/reload")
        _require(proposal_after == final_proposal,
                 "proposal tensors changed across save/reload")
    except BaseException as exc:
        # This path did not exist on entry and was published by us.  A failed
        # reload/authentication must not leave an artifact that looks usable.
        try:
            out_path.unlink()
        except FileNotFoundError:
            pass
        if isinstance(exc, OutcomeGRPOError):
            raise
        raise OutcomeGRPOError(f"cannot authenticate reloaded descendant: {exc}") from exc
    return {
        "path": str(out_path), "sha256": recovery.sha256_file(out_path),
        "size": int(out_path.stat().st_size),
        "global_step": int(reloaded["global_step"]),
        "config_hash": str(reloaded["config_hash"]),
        "optimizer_steps": int(optimizer_steps),
        "frozen_model": frozen_after, "proposal": proposal_after,
    }


def _mean_metrics(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {key: sum(float(row[key]) for row in rows) / len(rows)
            for key in rows[0]}


def train_outcome_grpo_round0(
    *,
    checkpoint: str | os.PathLike[str],
    train_collection_dir: str | os.PathLike[str],
    validation_collection_dir: str | os.PathLike[str],
    out: str | os.PathLike[str],
    device: str = "cuda",
    chunk_replans: int = 16,
    quiet: bool = False,
) -> dict[str, Any]:
    if chunk_replans <= 0:
        raise ValueError("chunk_replans must be positive")
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise OutcomeGRPOError("CUDA requested but unavailable")
    source_before = _source_identity()
    torch.manual_seed(TRAIN_SEED)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(TRAIN_SEED)
    parent, parent_identity = _load_parent(checkpoint)
    train = ValidatedRecoveryCollection.open(
        train_collection_dir, checkpoint_identity=parent_identity, purpose="train",
    )
    validation = ValidatedRecoveryCollection.open(
        validation_collection_dir, checkpoint_identity=parent_identity,
        purpose="validation",
    )
    _require(len(train.receipts) == EXPECTED_TRAIN_GROUPS,
             f"round-0 requires {EXPECTED_TRAIN_GROUPS} training groups")
    updates = expected_optimizer_updates(len(train.receipts))
    _require(updates == 400, f"round-0 update count is {updates}, expected 400")
    proposal = _load_proposal(parent, target_device)
    train_behaviour = authenticate_behaviour_policy(
        proposal, train, device=target_device, chunk_replans=chunk_replans,
    )
    validation_behaviour = authenticate_behaviour_policy(
        proposal, validation, device=target_device, chunk_replans=chunk_replans,
    )
    anchor = ExpertAnchor.from_parent(parent, proposal, device=target_device)
    anchor_initial = anchor.cache_gate(updates)

    optimizer = torch.optim.AdamW(
        proposal.parameters(), lr=LEARNING_RATE, betas=ADAMW_BETAS,
        weight_decay=ADAMW_WEIGHT_DECAY, eps=ADAMW_EPS,
    )
    _require(len(optimizer.state) == 0, "proposal optimizer was not reset")
    proposal.train()
    epoch_means: list[dict[str, float]] = []
    all_rows: list[dict[str, float]] = []
    update_index = 0
    for epoch in range(EPOCHS):
        rows: list[dict[str, float]] = []
        for group_index in range(len(train.receipts)):
            payload = train.load(group_index)
            metrics = _train_one_group(
                proposal, payload, optimizer, anchor,
                update_index=update_index, device=target_device,
                chunk_replans=chunk_replans,
            )
            rows.append(metrics)
            all_rows.append(metrics)
            update_index += 1
            if not quiet and ((group_index + 1) % 20 == 0
                              or group_index + 1 == len(train.receipts)):
                print(
                    f"[outcome-grpo-round0] epoch={epoch + 1}/{EPOCHS} "
                    f"group={group_index + 1}/{len(train.receipts)} "
                    f"grpo={metrics['grpo_loss']:.6g} "
                    f"ce={metrics['anchor_sparse_ce']:.6g} "
                    f"switch={metrics['switch_balance']:.6g}", flush=True,
                )
        epoch_means.append(_mean_metrics(rows))
    _require(update_index == updates == 400,
             f"executed {update_index} updates, expected exactly 400")
    optimizer.zero_grad(set_to_none=True)
    anchor_final = {
        "first_step": anchor_initial["first_step"],
        "last_step": anchor_initial["last_step"],
        "n_batches": anchor_initial["n_batches"],
        **anchor.evaluate_gate(),
    }
    unexpected = anchor.unexpected_gradients()
    trust = evaluate_trust_gates(
        proposal, validation, device=target_device,
        chunk_replans=chunk_replans, anchor_initial=anchor_initial,
        anchor_final=anchor_final, training_nonfinite=0,
        unexpected_gradients=unexpected,
    )
    if not trust["passed"]:
        raise TrustGateError(trust)
    train.assert_unchanged()
    validation.assert_unchanged()
    _assert_parent_unchanged(parent_identity)
    _require(recovery.source_digest(_ROOT)
             == train.manifest["identity"]["source"]["sha256"]
             == validation.manifest["identity"]["source"]["sha256"],
             "collector source changed during training")
    _require(_source_identity() == source_before,
             "round-0 trainer source changed during training")
    provenance = {
        "trainer_source": source_before,
        "train_collection": train.provenance(),
        "validation_collection": validation.provenance(),
        "behaviour_authentication": {
            "train": train_behaviour, "validation": validation_behaviour,
        },
        "expert_anchor": {
            "loss": "loom.losses.proposal_bc.proposal_sparse_ce_loss",
            "teacher": "frozen q_action(action, frozen estimator belief)",
            "weight": anchor.weight, "temperature": anchor.temperature,
            "switch_balance_weight": SWITCH_BALANCE_WEIGHT,
            "data": _json_copy(anchor.data_provenance),
            "gate_initial": anchor_initial, "gate_final": anchor_final,
        },
        "recipe": {
            "algorithm": "stored_order_pl_clipped_grpo",
            "arm0": "normalisation/control only; zero ratios and gradients",
            "sampled_arms": list(range(1, recovery.GROUP_SIZE)),
            "reward_normalisation": "per complete group population mean/std",
            "aggregation": "mean replans then mean sampled trajectories",
            "group_order": "manifest canonical, no shuffle",
            "epochs": EPOCHS, "groups_per_epoch": len(train.receipts),
            "optimizer_steps": update_index, "clip_eps": CLIP_EPS,
            "optimizer": {
                "kind": "AdamW", "lr": LEARNING_RATE,
                "betas": list(ADAMW_BETAS), "weight_decay": ADAMW_WEIGHT_DECAY,
                "eps": ADAMW_EPS, "scheduler": None, "grad_clip": GRAD_CLIP,
                "proposal_state_reset_at_entry": True,
            },
            "seed": TRAIN_SEED, "chunk_replans": int(chunk_replans),
        },
        "training": {
            "optimizer_steps": update_index,
            "groups_per_epoch": len(train.receipts),
            "epoch_means": epoch_means,
            "all_update_means": _mean_metrics(all_rows),
            "nonfinite": 0, "unexpected_gradients": unexpected,
        },
        "trust_gate": trust,
    }
    report = write_descendant_checkpoint(
        out, parent=parent, parent_identity=parent_identity,
        proposal=proposal, optimizer=optimizer,
        optimizer_steps=update_index, provenance=provenance,
    )
    report.update({"trust_gate": trust, "train_collection": train.provenance(),
                   "validation_collection": validation.provenance()})
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-collection", required=True)
    parser.add_argument("--validation-collection", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-replans", type=int, default=16)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = train_outcome_grpo_round0(
            checkpoint=args.checkpoint,
            train_collection_dir=args.train_collection,
            validation_collection_dir=args.validation_collection,
            out=args.out, device=args.device, chunk_replans=args.chunk_replans,
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
        print(f"OUTCOME_GRPO_ROUND0_FAILED: {exc}", flush=True)
        return 2
    print(json.dumps({"status": "PASS", **report}, indent=2, sort_keys=True,
                     allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
