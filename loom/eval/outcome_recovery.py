"""Collect full-trajectory, terminal-outcome groups for proposal GRPO.

This is deliberately a collector, not a training implementation.  Every
LIBERO reset is evaluated as one fixed group of eight complete trajectories:

* arm 0 is the deployed direct ``argmax pi_c`` policy;
* arms 1..7 independently sample the exact ordered Plackett--Luce atom at
  every replan.

All arms share the environment condition and decoder RNG stream.  Sampled
arms have independent, SHA-derived proposal streams.  There is no q_action,
bank, realizability filter, or fallback: the sampled coefficient is decoded
and executed.  The sidecar stores detached beliefs/language, the ordered
support, and the old-policy PL log-probability at each replan.  The only reward
is the scalar terminal LIBERO success bit.  SHA witnesses bind every record to
both the raw decoded 8-step segment and the exact post-resample/post-clip action
prefix actually dispatched before termination.

The production path is pinned to the consolidated R0-A deploy checkpoint at
step 49,666.  A group sidecar is written atomically only after all eight arms
finish and pass validation; the parent then atomically commits its SHA-256
receipt to the manifest.  Resume validates every committed sidecar and adopts
a valid orphan left by a crash between those two atomic renames.  Any mixed
identity, missing file, hash mismatch, incomplete group, or malformed tensor
fails closed.

Import discipline follows the rest of :mod:`loom.eval`: proposal primitives
are imported lazily inside the policy method, never from ``loom.heads`` at
module scope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from contracts import D, H_OP, K, M, TOPK
from loom.eval import DEFAULT_LIBERO_SUITES, episode_seed, policy_seed
from loom.eval.policy import (
    LoomPolicy,
    PolicyModules,
    _call,
    _module_dtype,
    feats_to,
    load_policy,
    policy_provenance,
)
from loom.eval.runner import WorkItem, _run_item, claim_device

__all__ = [
    "FORMAT_VERSION",
    "GROUP_SIZE",
    "SEED_GLOBAL_STEP",
    "SEED_CONFIG_HASH",
    "SEED_CHECKPOINT_SHA256",
    "TRIAL_SPLITS",
    "TRAIN_FOLDS",
    "PROPOSAL_SEED_SCHEME",
    "ACTION_SEGMENT_WITNESS_SCHEME",
    "OutcomeRecoveryPolicy",
    "RecoveryStore",
    "RecoveryBundle",
    "proposal_seed",
    "action_segment_sha256",
    "split_trials",
    "collection_items",
    "work_key",
    "authenticate_seed_checkpoint",
    "load_recovery_bundle",
    "collect_group",
    "validate_group_payload",
    "build_parser",
    "main",
]


FORMAT_VERSION = 1
GROUP_SIZE = 8
N_SAMPLED_ARMS = GROUP_SIZE - 1
SEED_GLOBAL_STEP = 49_666
SEED_CONFIG_HASH = "a199324a6205bb6d"
SEED_CHECKPOINT_SHA256 = (
    "15f286c268caa5327d5aa3abf1f67ebd0555c426a509fef22cb7f537bf6ab4e1"
)
PROTOCOL_SEED = 0
MAX_STEPS = 512
N_TASKS = 10
PROPOSAL_SEED_SCHEME = (
    "sha256(outcome-recovery-proposal|policy-seed|arm)-v1"
)
DECODER_WITNESS_SCHEME = "sha256(torch.Generator.get_state)-v1"
ACTION_SEGMENT_WITNESS_SCHEME = "sha256(dtype-shape-bytes)-v1"
IDENTITY_DIGEST_SCHEME = "sha256(canonical-json)-v1"
SOURCE_DIGEST_SCHEME = "sha256(path-nul-bytes-nul)-v1"

# These are LIBERO's 50 distinct init-state indices.  They are a method
# contract, not CLI defaults: callers may select a split but cannot remap it.
# ``train0`` .. ``train5`` partition the aggregate train set into the immutable
# five-trial/200-group folds consumed one at a time by recovery macro-rounds.
_AGGREGATE_TRIAL_SPLITS: dict[str, tuple[int, ...]] = {
    "official": tuple(range(0, 10)),
    "train": tuple(range(10, 40)),
    "validation": tuple(range(40, 50)),
}
TRAIN_FOLDS: dict[str, tuple[int, ...]] = {
    f"train{fold}": tuple(range(10 + 5 * fold, 15 + 5 * fold))
    for fold in range(6)
}
TRIAL_SPLITS: dict[str, tuple[int, ...]] = {
    **_AGGREGATE_TRIAL_SPLITS,
    **TRAIN_FOLDS,
}

_SOURCE_FILES = (
    "contracts.py",
    "loom/data/adapters/libero.py",
    "loom/data/canonical.py",
    "loom/data/tower.py",
    "loom/eval/__init__.py",
    "loom/eval/libero.py",
    "loom/eval/outcome_recovery.py",
    "loom/eval/policy.py",
    "loom/eval/runner.py",
    "loom/heads/decoder.py",
    "loom/heads/proposal.py",
    "loom/model/estimator.py",
    "scripts/outcome_recovery.py",
)

_GROUP_KIND = "loom_outcome_recovery_group"
_MANIFEST_KIND = "loom_outcome_recovery_collection"


def _require(condition: Any, message: str) -> None:
    if not bool(condition):
        raise RuntimeError(message)


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def identity_digest(identity: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(identity)).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source_digest(root: str | os.PathLike[str] | None = None) -> str:
    repo = (Path(root).resolve() if root is not None
            else Path(__file__).resolve().parents[2])
    h = hashlib.sha256()
    for rel in _SOURCE_FILES:
        path = repo / rel
        if not path.is_file():
            raise FileNotFoundError(f"recovery provenance source is missing: {path}")
        h.update(rel.encode("utf-8") + b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                h.update(block)
        h.update(b"\0")
    return h.hexdigest()


def _state_witness(generator: torch.Generator | None) -> str:
    _require(generator is not None, "decoder RNG generator is absent")
    state = generator.get_state().detach().cpu().contiguous()
    return hashlib.sha256(state.numpy().tobytes()).hexdigest()


def action_segment_sha256(segment: np.ndarray | Tensor) -> str:
    """Hash exact decoded/dispatched action content, including shape and dtype."""
    if isinstance(segment, Tensor):
        value = segment.detach().cpu().contiguous().numpy()
    else:
        value = np.ascontiguousarray(np.asarray(segment))
    header = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header + b"\0" + value.tobytes(order="C")).hexdigest()


def _valid_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(c in "0123456789abcdef" for c in text)


def split_trials(split: str) -> tuple[int, ...]:
    try:
        return TRIAL_SPLITS[str(split)]
    except KeyError as exc:
        raise ValueError(
            f"unknown recovery split {split!r}; choose one of {tuple(TRIAL_SPLITS)}"
        ) from exc


def proposal_seed(base_policy_seed: int, arm: int) -> int:
    """Independent, process-stable PL stream for sampled arm 1..7."""
    if not 1 <= int(arm) < GROUP_SIZE:
        raise ValueError(
            f"proposal seeds are defined only for sampled arms 1..{GROUP_SIZE - 1}"
        )
    raw = (
        f"outcome-recovery-proposal|{int(base_policy_seed)}|arm={int(arm)}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


def collection_items(split: str) -> list[WorkItem]:
    """The immutable work set for one recovery split, in canonical order."""
    trials = split_trials(split)
    items: list[WorkItem] = []
    for suite in DEFAULT_LIBERO_SUITES:
        for task_id in range(N_TASKS):
            for trial in trials:
                items.append(WorkItem(
                    bench="libero",
                    suite=suite,
                    task_id=task_id,
                    episode=trial,
                    seed=PROTOCOL_SEED,
                    env_seed=episode_seed(
                        PROTOCOL_SEED, "libero", suite, task_id, trial,
                    ),
                    policy_seed=policy_seed(
                        PROTOCOL_SEED, "libero", suite, task_id, trial,
                    ),
                    max_steps=MAX_STEPS,
                ))
    expected = len(DEFAULT_LIBERO_SUITES) * N_TASKS * len(trials)
    _require(len(items) == expected, "internal recovery work-count mismatch")
    return items


def work_key(item: WorkItem) -> str:
    return (
        f"{item.suite}/task={int(item.task_id):02d}/"
        f"trial={int(item.episode):02d}/seed={int(item.seed)}"
    )


def group_filename(item: WorkItem) -> str:
    return (
        f"{item.suite}__task{int(item.task_id):02d}__"
        f"trial{int(item.episode):02d}__seed{int(item.seed)}.pt"
    )


class OutcomeRecoveryPolicy(LoomPolicy):
    """One arm of the eight-trajectory outcome-recovery group.

    The estimator, proposal, and decoder are the deployed direct-policy
    modules.  q_action is rejected rather than ignored so an oracle/gated
    bundle cannot silently enter this dataset.
    """

    def __init__(self, modules: PolicyModules, *, arm: int) -> None:
        if not 0 <= int(arm) < GROUP_SIZE:
            raise ValueError(f"recovery arm must be in [0,{GROUP_SIZE - 1}]")
        if getattr(modules, "q_action", None) is not None:
            raise ValueError(
                "outcome recovery requires q_action=None; there is no "
                "realizability rejection or fallback"
            )
        if not callable(getattr(modules.proposal, "logits", None)):
            raise TypeError(
                "outcome recovery requires proposal.logits for exact ordered PL"
            )
        self.recovery_arm = int(arm)
        super().__init__(
            modules,
            n_candidates=1,
            clip_actions=True,
            op_stats=False,
            gripper_dwell=1,
            decoder_samples=1,
            duration_normalize_segments=False,
        )
        if not self._decoder_accepts_generator:
            raise TypeError(
                "outcome recovery requires decoder.forward(..., generator=...) "
                "for common-noise witnesses"
            )

    def reset(self) -> None:
        super().reset()
        self._proposal_generator: torch.Generator | None = None
        if self._policy_seed is not None and self.recovery_arm > 0:
            self._proposal_generator = torch.Generator(device=self.device)
            self._proposal_generator.manual_seed(
                proposal_seed(self._policy_seed, self.recovery_arm)
            )
        self._traj_z: list[Tensor] = []
        self._traj_order: list[Tensor] = []
        self._traj_coeff: list[Tensor] = []
        self._traj_old_logprob: list[Tensor] = []
        self._decoder_before: list[str] = []
        self._decoder_after: list[str] = []
        self._proposal_before: list[str] = []
        self._proposal_after: list[str] = []
        self._decoded_action_sha256: list[str] = []
        self._executed_action_segments: list[list[np.ndarray]] = []
        self._traj_lang: Tensor | None = None
        self._instruction_sha256: str | None = None

    def set_policy_seed(self, seed: int) -> None:
        # ``run_episode`` calls reset once more after this; both paths construct
        # the same independent generators, so worker history is irrelevant.
        super().set_policy_seed(seed)
        if self.recovery_arm > 0:
            self._proposal_generator = torch.Generator(device=self.device)
            self._proposal_generator.manual_seed(proposal_seed(seed, self.recovery_arm))

    def act(self, obs: dict, instruction: str) -> np.ndarray:
        """Record the exact post-resample/post-clip actions actually dispatched."""
        replans_before = self.replans
        action = super().act(obs, instruction)
        if self.replans > replans_before:
            _require(self.replans == replans_before + 1,
                     "one act call unexpectedly created multiple replans")
            self._executed_action_segments.append([])
        _require(bool(self._executed_action_segments),
                 "dispatched an action without an active recovery segment")
        self._executed_action_segments[-1].append(
            np.asarray(action, dtype=np.float32).copy(),
        )
        return action

    @torch.no_grad()
    def _plan(self, obs: dict, instruction: str) -> np.ndarray:
        # Lazy import is mandatory for eval's module-scope dependency boundary.
        from loom.heads.proposal import (                 # noqa: PLC0415
            gumbel_topk,
            pl_log_prob,
            weights_from_logits,
        )

        modules = self.modules
        feats = modules.featurize(obs, instruction)
        feats = feats_to(feats, self.device, _module_dtype(modules.estimator))
        z = _call(modules.estimator, feats, self._z)
        self._z = z

        _require(z.ndim == 3 and tuple(z.shape[1:]) == (K, D),
                 f"estimator must emit (B,{K},{D}), got {tuple(z.shape)}")
        _require(z.shape[0] == 1, "collection executes one environment per arm")
        lang = feats["lang"]
        if lang.ndim == 2:
            lang = lang.unsqueeze(1)
        _require(lang.ndim == 3 and lang.shape[0] == 1,
                 f"language must be (1,L,F), got {tuple(lang.shape)}")

        logits = modules.proposal.logits(z, lang)
        _require(tuple(logits.shape) == (1, M),
                 f"proposal logits must be (1,{M}), got {tuple(logits.shape)}")
        _require(bool(torch.isfinite(logits).all()), "proposal logits contain nan/inf")

        if self.recovery_arm == 0:
            order = logits.float().topk(TOPK, dim=-1).indices
        else:
            _require(self._proposal_generator is not None,
                     "sampled arm has no proposal generator")
            self._proposal_before.append(_state_witness(self._proposal_generator))
            order = gumbel_topk(logits, TOPK, self._proposal_generator)
            self._proposal_after.append(_state_witness(self._proposal_generator))
        coeff = weights_from_logits(logits, order, M)
        # Score the *sampled order*, never Proposal.log_prob(c), whose public
        # compatibility surface canonicalises by descending coefficient.
        old_logprob = pl_log_prob(logits.float(), order)
        self.last_coeff = coeff

        decoder_before = _state_witness(self._decoder_generator)
        action = _call(
            modules.decoder,
            feats["proprio"],
            coeff,
            generator=self._decoder_generator,
        )
        decoder_after = _state_witness(self._decoder_generator)

        z_cpu = z[0].detach().cpu().contiguous()
        lang_cpu = lang[0].detach().cpu().contiguous()
        if self._traj_lang is None:
            self._traj_lang = lang_cpu
            self._instruction_sha256 = hashlib.sha256(
                instruction.encode("utf-8")
            ).hexdigest()
        else:
            _require(
                tuple(lang_cpu.shape) == tuple(self._traj_lang.shape)
                and torch.equal(lang_cpu, self._traj_lang),
                "language sidecar changed within one trajectory",
            )
            _require(
                hashlib.sha256(instruction.encode("utf-8")).hexdigest()
                == self._instruction_sha256,
                "instruction changed within one trajectory",
            )

        self._traj_z.append(z_cpu)
        self._traj_order.append(order[0].detach().cpu().to(torch.int64).contiguous())
        self._traj_coeff.append(coeff[0].detach().cpu().contiguous())
        self._traj_old_logprob.append(
            old_logprob[0].detach().cpu().to(torch.float32).contiguous()
        )
        self._decoder_before.append(decoder_before)
        self._decoder_after.append(decoder_after)

        out = action.detach().to(torch.float32).cpu().numpy()
        if out.ndim == 3:
            out = out[0]
        if out.shape != (H_OP, self.spec.dof):
            raise ValueError(
                f"decoder must emit ({H_OP}, {self.spec.dof}), got {out.shape}"
            )
        self._decoded_action_sha256.append(action_segment_sha256(out))
        return out

    def trajectory_payload(self, *, terminal_success: bool) -> dict[str, Any]:
        """Detached CPU tensors plus terminal reward; no shaped rewards."""
        n = len(self._traj_z)
        _require(n > 0, "cannot export a trajectory with zero replans")
        _require(self._traj_lang is not None, "trajectory language is absent")
        _require(
            len(self._traj_order) == len(self._traj_coeff)
            == len(self._traj_old_logprob) == len(self._decoder_before)
            == len(self._decoder_after) == len(self._decoded_action_sha256) == n,
            "incomplete per-replan recovery record",
        )
        _require(len(self._executed_action_segments) == n,
                 "executed action-segment witnesses are incomplete")
        _require(all(bool(segment) for segment in self._executed_action_segments),
                 "a recovery replan dispatched no environment action")
        executed_steps = [len(segment) for segment in self._executed_action_segments]
        executed_sha = [
            action_segment_sha256(np.stack(segment, axis=0))
            for segment in self._executed_action_segments
        ]
        if self.recovery_arm == 0:
            _require(not self._proposal_before and not self._proposal_after,
                     "direct arm unexpectedly consumed proposal RNG")
        else:
            _require(
                len(self._proposal_before) == len(self._proposal_after) == n,
                "sampled arm proposal witnesses are incomplete",
            )
        return {
            "arm": self.recovery_arm,
            "behavior": ("direct_argmax" if self.recovery_arm == 0
                         else "ordered_pl_sample"),
            # Arm 0 is retained as the within-reset deployed baseline.  Its PL
            # score is diagnostic only and must never enter an importance ratio.
            "behavior_logprob_valid": self.recovery_arm > 0,
            "policy_seed": int(self._policy_seed) if self._policy_seed is not None else None,
            "proposal_seed": (
                proposal_seed(int(self._policy_seed), self.recovery_arm)
                if self.recovery_arm > 0 and self._policy_seed is not None else None
            ),
            "instruction_sha256": self._instruction_sha256,
            "z": torch.stack(self._traj_z, dim=0),
            "lang": self._traj_lang,
            "ordered_support": torch.stack(self._traj_order, dim=0),
            "coeff": torch.stack(self._traj_coeff, dim=0),
            "old_logprob": torch.stack(self._traj_old_logprob, dim=0),
            "decoder_rng_before": list(self._decoder_before),
            "decoder_rng_after": list(self._decoder_after),
            "proposal_rng_before": list(self._proposal_before),
            "proposal_rng_after": list(self._proposal_after),
            "decoded_action_segment_sha256": list(self._decoded_action_sha256),
            "executed_action_segment_sha256": executed_sha,
            "executed_action_steps": executed_steps,
            "terminal_reward": torch.tensor(
                float(bool(terminal_success)), dtype=torch.float32,
            ),
        }


@dataclass(frozen=True)
class RecoveryBundle:
    modules: PolicyModules
    provenance: dict[str, Any]


def authenticate_seed_checkpoint(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Full-file authentication of the one permitted initial recovery seed."""
    ckpt = Path(path).expanduser().resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"recovery seed checkpoint does not exist: {ckpt}")
    stat = ckpt.stat()
    digest = sha256_file(ckpt)
    _require(
        digest == SEED_CHECKPOINT_SHA256,
        f"checkpoint SHA-256 {digest} is not the pinned step-{SEED_GLOBAL_STEP} seed",
    )
    return {
        "kind": "consolidated",
        "path": str(ckpt),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest,
        "global_step": SEED_GLOBAL_STEP,
        "config_hash": SEED_CONFIG_HASH,
    }


def _assert_checkpoint_stat(identity: Mapping[str, Any]) -> None:
    path = Path(str(identity["path"]))
    stat = path.stat()
    _require(int(stat.st_size) == int(identity["size"]),
             "recovery checkpoint size changed after authentication")
    _require(int(stat.st_mtime_ns) == int(identity["mtime_ns"]),
             "recovery checkpoint mtime changed after authentication")


def load_recovery_bundle(
    checkpoint: str | os.PathLike[str],
    *,
    device: str,
) -> RecoveryBundle:
    """Load only E, pi_c, and D_e and verify embedded step/config identity."""
    base = load_policy(
        str(checkpoint),
        embodiment="libero_franka",
        device=device,
        allow_stub=False,
        n_candidates=1,
        op_stats=False,
        gripper_dwell=1,
        decoder_samples=1,
        duration_normalize_segments=False,
        _include_q_action=False,
    )
    modules = base.modules
    _require(not modules.is_stub, "recovery seed loaded stub modules")
    _require(modules.q_action is None,
             "recovery loader unexpectedly constructed q_action")
    meta = dict(modules.meta or {})
    _require(
        int(meta.get("ckpt_global_step", -1)) == SEED_GLOBAL_STEP,
        f"loaded checkpoint step {meta.get('ckpt_global_step')!r} != {SEED_GLOBAL_STEP}",
    )
    _require(
        str(meta.get("ckpt_config_hash") or "") == SEED_CONFIG_HASH,
        f"loaded config hash {meta.get('ckpt_config_hash')!r} != {SEED_CONFIG_HASH!r}",
    )
    state = dict(meta.get("state_dict") or {})
    for name in ("estimator", "proposal", "decoder"):
        info = dict(state.get(name) or {})
        _require(int(info.get("tensors_loaded", 0)) > 0,
                 f"recovery seed has no authenticated {name} tensors")
        _require(int(info.get("unexpected", -1)) == 0,
                 f"recovery seed has unexpected {name} tensors")
    provenance = policy_provenance(base)
    provenance["outcome_recovery"] = {
        "format_version": FORMAT_VERSION,
        "group_size": GROUP_SIZE,
        "direct_arms": [0],
        "sampled_arms": list(range(1, GROUP_SIZE)),
        "proposal_seed_scheme": PROPOSAL_SEED_SCHEME,
        "common_decoder_noise": True,
        "q_action": False,
        "terminal_reward_only": True,
    }
    return RecoveryBundle(modules=modules, provenance=provenance)


def _tensor_detached_finite(value: Any, name: str) -> Tensor:
    _require(isinstance(value, Tensor), f"{name} must be a tensor")
    _require(not value.requires_grad, f"{name} must be detached")
    _require(value.device.type == "cpu", f"{name} must be stored on CPU")
    _require(bool(torch.isfinite(value).all()), f"{name} contains nan/inf")
    return value


def _validate_arm_payload(raw: Mapping[str, Any], item: WorkItem, arm: int) -> None:
    allowed = {
        "arm", "behavior", "behavior_logprob_valid", "policy_seed",
        "proposal_seed", "instruction_sha256", "z", "lang",
        "ordered_support", "coeff", "old_logprob", "decoder_rng_before",
        "decoder_rng_after", "proposal_rng_before", "proposal_rng_after",
        "decoded_action_segment_sha256", "executed_action_segment_sha256",
        "executed_action_steps",
        "terminal_reward",
    }
    _require(set(raw) == allowed,
             f"arm {arm} sidecar fields differ: {sorted(set(raw) ^ allowed)}")
    reward_fields = [name for name in raw if "reward" in name.lower()]
    _require(reward_fields == ["terminal_reward"],
             f"arm {arm} contains non-terminal reward fields {reward_fields}")
    _require(not any("q_action" in name.lower() for name in raw),
             f"arm {arm} contains q_action state")
    _require(int(raw["arm"]) == arm, f"sidecar arm index mismatch at {arm}")
    expected_behavior = "direct_argmax" if arm == 0 else "ordered_pl_sample"
    _require(raw["behavior"] == expected_behavior,
             f"arm {arm} behavior is not {expected_behavior}")
    _require(raw["behavior_logprob_valid"] is (arm > 0),
             f"arm {arm} behavior-logprob validity is wrong")
    _require(int(raw["policy_seed"]) == int(item.policy_seed),
             f"arm {arm} policy seed mismatch")
    if arm == 0:
        _require(raw["proposal_seed"] is None,
                 "direct arm must not have a proposal seed")
    else:
        _require(int(raw["proposal_seed"]) == proposal_seed(item.policy_seed, arm),
                 f"arm {arm} proposal seed mismatch")
    _require(_valid_sha256(raw["instruction_sha256"]),
             f"arm {arm} instruction digest is invalid")

    z = _tensor_detached_finite(raw["z"], f"arm {arm} z")
    lang = _tensor_detached_finite(raw["lang"], f"arm {arm} lang")
    order = _tensor_detached_finite(raw["ordered_support"], f"arm {arm} support")
    coeff = _tensor_detached_finite(raw["coeff"], f"arm {arm} coeff")
    old_lp = _tensor_detached_finite(raw["old_logprob"], f"arm {arm} old_logprob")
    terminal = _tensor_detached_finite(
        raw["terminal_reward"], f"arm {arm} terminal_reward",
    )
    _require(z.is_floating_point(), f"arm {arm} z must be floating point")
    _require(lang.is_floating_point(), f"arm {arm} lang must be floating point")
    _require(coeff.is_floating_point(), f"arm {arm} coeff must be floating point")
    _require(old_lp.is_floating_point(),
             f"arm {arm} old_logprob must be floating point")
    _require(terminal.is_floating_point(),
             f"arm {arm} terminal reward must be floating point")
    _require(z.ndim == 3 and tuple(z.shape[1:]) == (K, D),
             f"arm {arm} z shape is {tuple(z.shape)}")
    n = int(z.shape[0])
    _require(n > 0, f"arm {arm} has no replans")
    _require(lang.ndim == 2 and lang.numel() > 0,
             f"arm {arm} language shape is {tuple(lang.shape)}")
    _require(order.dtype == torch.int64 and tuple(order.shape) == (n, TOPK),
             f"arm {arm} support shape/dtype is {tuple(order.shape)}/{order.dtype}")
    _require(tuple(coeff.shape) == (n, M),
             f"arm {arm} coeff shape is {tuple(coeff.shape)}")
    _require(tuple(old_lp.shape) == (n,),
             f"arm {arm} old-logprob shape is {tuple(old_lp.shape)}")
    _require(terminal.numel() == 1 and float(terminal) in (0.0, 1.0),
             f"arm {arm} terminal reward must be scalar 0/1")
    _require(bool(((order >= 0) & (order < M)).all()),
             f"arm {arm} support index is outside [0,{M})")
    sorted_order = order.sort(dim=-1).values
    _require(bool((sorted_order[:, 1:] != sorted_order[:, :-1]).all()),
             f"arm {arm} support contains duplicate indices")
    _require(bool((coeff >= 0).all()), f"arm {arm} coeff is negative")
    # ``weights_from_logits`` normalises in fp32 and casts back to the deployed
    # head dtype.  bf16's four selected weights can sum a few ulps away from
    # one; use the same dtype-proportional principle as contracts.assert_simplex.
    coeff_tol = max(8.0 * float(torch.finfo(coeff.dtype).eps), 1e-4)
    _require(torch.allclose(
        coeff.float().sum(-1), torch.ones(n), atol=coeff_tol, rtol=0,
    ),
             f"arm {arm} coeff is not on the simplex")
    nonzero = coeff != 0
    _require(bool((nonzero.sum(-1) == TOPK).all()),
             f"arm {arm} coeff is not exactly {TOPK}-sparse")
    gathered = nonzero.gather(-1, order)
    _require(bool(gathered.all()),
             f"arm {arm} ordered support does not match coefficient support")

    before = list(raw["decoder_rng_before"])
    after = list(raw["decoder_rng_after"])
    _require(len(before) == len(after) == n,
             f"arm {arm} decoder witnesses are incomplete")
    _require(all(_valid_sha256(x) for x in before + after),
             f"arm {arm} has invalid decoder RNG witness")
    p_before = list(raw["proposal_rng_before"])
    p_after = list(raw["proposal_rng_after"])
    expected_n = 0 if arm == 0 else n
    _require(len(p_before) == len(p_after) == expected_n,
             f"arm {arm} proposal witnesses are incomplete")
    _require(all(_valid_sha256(x) for x in p_before + p_after),
             f"arm {arm} has invalid proposal RNG witness")
    decoded_sha = list(raw["decoded_action_segment_sha256"])
    executed_sha = list(raw["executed_action_segment_sha256"])
    executed_steps = list(raw["executed_action_steps"])
    _require(len(decoded_sha) == len(executed_sha) == len(executed_steps) == n,
             f"arm {arm} action-segment witnesses are incomplete")
    _require(all(_valid_sha256(x) for x in decoded_sha + executed_sha),
             f"arm {arm} has invalid action-segment SHA witness")
    _require(all(1 <= int(steps) <= 6 for steps in executed_steps),
             f"arm {arm} dispatched an invalid number of actions per replan")
    _require(all(int(steps) in (5, 6) for steps in executed_steps[:-1]),
             f"arm {arm} has a truncated non-terminal action segment")


def validate_group_payload(
    raw: Mapping[str, Any],
    *,
    item: WorkItem,
    expected_identity_digest: str,
    expected_split: str,
) -> None:
    allowed = {
        "format_version", "kind", "identity_digest", "split", "group_id",
        "work_item", "arms", "common_decoder_rng",
    }
    _require(isinstance(raw, Mapping), "group sidecar is not a mapping")
    _require(set(raw) == allowed,
             f"group sidecar fields differ: {sorted(set(raw) ^ allowed)}")
    _require(int(raw["format_version"]) == FORMAT_VERSION,
             "group sidecar format version mismatch")
    _require(raw["kind"] == _GROUP_KIND, "group sidecar kind mismatch")
    _require(raw["identity_digest"] == expected_identity_digest,
             "group sidecar collection identity mismatch")
    _require(raw["split"] == expected_split, "group sidecar split mismatch")
    _require(raw["group_id"] == work_key(item), "group sidecar work key mismatch")
    _require(raw["work_item"] == item.to_dict(), "group sidecar work item mismatch")
    arms = raw["arms"]
    _require(isinstance(arms, list) and len(arms) == GROUP_SIZE,
             f"group sidecar must contain exactly {GROUP_SIZE} arms")
    for arm, payload in enumerate(arms):
        _require(isinstance(payload, Mapping), f"arm {arm} payload is not a mapping")
        _validate_arm_payload(payload, item, arm)

    common = raw["common_decoder_rng"]
    _require(isinstance(common, Mapping), "common decoder witness is absent")
    _require(set(common) == {
        "scheme", "seed", "passed", "shared_prefix_replans_by_arm",
    }, "common decoder witness fields differ")
    _require(common["scheme"] == DECODER_WITNESS_SCHEME,
             "common decoder witness scheme mismatch")
    _require(int(common["seed"]) == item.policy_seed,
             "common decoder seed mismatch")
    _require(common["passed"] is True, "common decoder RNG check did not pass")
    prefixes = list(common["shared_prefix_replans_by_arm"])
    _require(len(prefixes) == GROUP_SIZE,
             "common decoder prefix evidence has wrong arm count")
    reference = arms[0]
    for arm, payload in enumerate(arms):
        n = min(len(reference["decoder_rng_before"]),
                len(payload["decoder_rng_before"]))
        _require(int(prefixes[arm]) == n,
                 f"arm {arm} common-noise prefix count mismatch")
        _require(reference["decoder_rng_before"][:n]
                 == payload["decoder_rng_before"][:n],
                 f"arm {arm} decoder pre-state diverges inside common prefix")
        _require(reference["decoder_rng_after"][:n]
                 == payload["decoder_rng_after"][:n],
                 f"arm {arm} decoder post-state diverges inside common prefix")


def _make_group_payload(
    item: WorkItem,
    *,
    split: str,
    collection_identity_digest: str,
    arms: list[dict[str, Any]],
) -> dict[str, Any]:
    reference = arms[0]
    prefixes: list[int] = []
    for arm, payload in enumerate(arms):
        n = min(len(reference["decoder_rng_before"]),
                len(payload["decoder_rng_before"]))
        _require(reference["decoder_rng_before"][:n]
                 == payload["decoder_rng_before"][:n],
                 f"arm {arm} did not receive common decoder pre-states")
        _require(reference["decoder_rng_after"][:n]
                 == payload["decoder_rng_after"][:n],
                 f"arm {arm} did not receive common decoder post-states")
        prefixes.append(n)
    return {
        "format_version": FORMAT_VERSION,
        "kind": _GROUP_KIND,
        "identity_digest": collection_identity_digest,
        "split": split,
        "group_id": work_key(item),
        "work_item": item.to_dict(),
        "arms": arms,
        "common_decoder_rng": {
            "scheme": DECODER_WITNESS_SCHEME,
            "seed": item.policy_seed,
            "passed": True,
            "shared_prefix_replans_by_arm": prefixes,
        },
    }


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing recovery sidecar: {path}"
        )
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
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


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
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


def _load_group(path: Path) -> Mapping[str, Any]:
    try:
        raw = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"cannot load recovery sidecar {path}: {exc}") from exc
    _require(isinstance(raw, Mapping), f"recovery sidecar {path} is not a mapping")
    return raw


def _receipt(
    sidecar: Path,
    payload: Mapping[str, Any],
    *,
    out_dir: Path,
) -> dict[str, Any]:
    arms = list(payload["arms"])
    rewards = [int(float(a["terminal_reward"])) for a in arms]
    replans = [int(a["z"].shape[0]) for a in arms]
    return {
        "group_id": payload["group_id"],
        "sidecar": str(sidecar.relative_to(out_dir)),
        "sha256": sha256_file(sidecar),
        "size": int(sidecar.stat().st_size),
        "n_arms": GROUP_SIZE,
        "n_replans_by_arm": replans,
        "terminal_rewards": rewards,
    }


def collect_group(
    item: WorkItem,
    bundle: RecoveryBundle,
    *,
    split: str,
    collection_identity_digest: str,
    out_dir: str | os.PathLike[str],
    env_factory: Callable[..., Any] | None = None,
    backend: str = "libero",
    bench_module: Any | None = None,
) -> dict[str, Any]:
    """Execute and atomically persist one complete eight-arm group."""
    _require(item.episode in split_trials(split),
             f"trial {item.episode} is outside split {split}")
    if bench_module is None:
        from loom.eval import libero as bench_module       # noqa: PLC0415

    arm_payloads: list[dict[str, Any]] = []
    for arm in range(GROUP_SIZE):
        policy = OutcomeRecoveryPolicy(bundle.modules, arm=arm)
        result = _run_item(item, policy, bench_module, env_factory, backend)
        if result.error is not None:
            raise RuntimeError(
                f"recovery group {work_key(item)} arm {arm} failed; "
                f"no group sidecar was committed:\n{result.error}"
            )
        _require(result.n_replans == policy.replans,
                 f"arm {arm} result/policy replan count mismatch")
        arm_payloads.append(
            policy.trajectory_payload(terminal_success=bool(result.success))
        )

    payload = _make_group_payload(
        item,
        split=split,
        collection_identity_digest=collection_identity_digest,
        arms=arm_payloads,
    )
    validate_group_payload(
        payload,
        item=item,
        expected_identity_digest=collection_identity_digest,
        expected_split=split,
    )
    root = Path(out_dir).expanduser().resolve()
    sidecar = root / "groups" / group_filename(item)
    _atomic_torch_save(sidecar, payload)
    receipt = _receipt(sidecar, payload, out_dir=root)
    receipt["worker"] = {
        "pid": os.getpid(),
        "device": str(bundle.modules.device),
    }
    return receipt


class RecoveryStore:
    """Atomic manifest with strict identity and sidecar-hash resumption."""

    def __init__(
        self,
        out_dir: str | os.PathLike[str],
        *,
        identity: Mapping[str, Any],
        split: str,
        items: Sequence[WorkItem],
    ) -> None:
        self.out_dir = Path(out_dir).expanduser().resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.out_dir / "manifest.json"
        self.identity = json.loads(json.dumps(dict(identity), allow_nan=False))
        self.identity_digest = identity_digest(self.identity)
        self.split = str(split)
        self.items = list(items)
        self.expected = {work_key(item): item for item in self.items}
        _require(len(self.expected) == len(self.items),
                 "recovery work set contains duplicate group identities")
        self.groups: dict[str, dict[str, Any]] = {}
        self.started_utc = _utc()
        if self.manifest_path.exists():
            old = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            self._load_manifest(old)
        else:
            self.flush(status="RUNNING")
        self._verify_committed_sidecars()
        self._adopt_orphans()

    def _load_manifest(self, old: Mapping[str, Any]) -> None:
        _require(old.get("kind") == _MANIFEST_KIND,
                 "existing output is not an outcome-recovery manifest")
        _require(int(old.get("format_version", -1)) == FORMAT_VERSION,
                 "existing recovery manifest version mismatch")
        _require(old.get("identity") == self.identity,
                 "existing recovery manifest has a different immutable identity")
        _require(old.get("identity_digest") == self.identity_digest,
                 "existing recovery manifest identity digest mismatch")
        _require(old.get("split") == self.split,
                 "existing recovery manifest split mismatch")
        self.started_utc = str(old.get("started_utc") or self.started_utc)
        rows = old.get("groups")
        _require(isinstance(rows, list), "existing recovery manifest groups is not a list")
        for raw in rows:
            _require(isinstance(raw, Mapping), "manifest group receipt is not a mapping")
            receipt = dict(raw)
            key = str(receipt.get("group_id") or "")
            _require(key in self.expected,
                     f"manifest contains out-of-split group {key!r}")
            _require(key not in self.groups,
                     f"manifest contains duplicate group {key!r}")
            self.groups[key] = receipt

    def _sidecar_path(self, receipt: Mapping[str, Any]) -> Path:
        rel = Path(str(receipt.get("sidecar") or ""))
        _require(not rel.is_absolute() and ".." not in rel.parts,
                 "manifest sidecar path escapes output directory")
        path = (self.out_dir / rel).resolve()
        _require(path.parent == (self.out_dir / "groups").resolve(),
                 "manifest sidecar is not directly inside groups/")
        return path

    def _validate_receipt(self, receipt: Mapping[str, Any], *, deep: bool) -> None:
        allowed = {
            "group_id", "sidecar", "sha256", "size", "n_arms",
            "n_replans_by_arm", "terminal_rewards", "worker",
        }
        _require(set(receipt) == allowed,
                 f"manifest receipt fields differ: {sorted(set(receipt) ^ allowed)}")
        key = str(receipt["group_id"])
        _require(key in self.expected, f"receipt has unknown group {key!r}")
        _require(_valid_sha256(receipt["sha256"]),
                 f"receipt {key} has invalid SHA-256")
        _require(int(receipt["n_arms"]) == GROUP_SIZE,
                 f"receipt {key} does not contain {GROUP_SIZE} arms")
        replans = list(receipt["n_replans_by_arm"])
        rewards = list(receipt["terminal_rewards"])
        _require(len(replans) == len(rewards) == GROUP_SIZE,
                 f"receipt {key} has wrong arm-vector length")
        _require(all(int(n) > 0 for n in replans),
                 f"receipt {key} has zero-replan trajectory")
        _require(all(int(r) in (0, 1) for r in rewards),
                 f"receipt {key} has non-terminal reward")
        path = self._sidecar_path(receipt)
        _require(path.is_file(), f"committed recovery sidecar is missing: {path}")
        _require(int(path.stat().st_size) == int(receipt["size"]),
                 f"committed recovery sidecar size changed: {path}")
        _require(sha256_file(path) == receipt["sha256"],
                 f"committed recovery sidecar hash changed: {path}")
        if deep:
            payload = _load_group(path)
            validate_group_payload(
                payload,
                item=self.expected[key],
                expected_identity_digest=self.identity_digest,
                expected_split=self.split,
            )

    def _verify_committed_sidecars(self, *, deep: bool = True) -> None:
        # Validate both the receipt hash and payload schema before skipping any
        # group.  Loading is one sidecar at a time, so memory stays bounded even
        # though fail-closed resume intentionally pays a full collection scan.
        for receipt in self.groups.values():
            self._validate_receipt(receipt, deep=deep)

    def _adopt_orphans(self) -> None:
        """Finish the one safe crash window: sidecar rename before manifest."""
        for key, item in self.expected.items():
            if key in self.groups:
                continue
            path = self.out_dir / "groups" / group_filename(item)
            if not path.exists():
                continue
            payload = _load_group(path)
            validate_group_payload(
                payload,
                item=item,
                expected_identity_digest=self.identity_digest,
                expected_split=self.split,
            )
            receipt = _receipt(path, payload, out_dir=self.out_dir)
            receipt["worker"] = {"pid": None, "device": "orphan-adopted"}
            self.add(receipt)

    def has(self, item: WorkItem) -> bool:
        return work_key(item) in self.groups

    def add(self, receipt: Mapping[str, Any]) -> None:
        value = dict(receipt)
        key = str(value.get("group_id") or "")
        _require(key in self.expected, f"worker returned unknown group {key!r}")
        _require(key not in self.groups, f"worker returned duplicate group {key!r}")
        self._validate_receipt(value, deep=True)
        self.groups[key] = value
        self.flush(status="RUNNING")

    def ordered_groups(self) -> list[dict[str, Any]]:
        return [self.groups[key] for key in self.expected if key in self.groups]

    def _summary(self, status: str) -> dict[str, Any]:
        rows = self.ordered_groups()
        terminal = [0] * GROUP_SIZE
        replans = [0] * GROUP_SIZE
        for row in rows:
            for arm in range(GROUP_SIZE):
                terminal[arm] += int(row["terminal_rewards"][arm])
                replans[arm] += int(row["n_replans_by_arm"][arm])
        return {
            "status": status,
            "complete": status == "COMPLETE",
            "n_groups": len(rows),
            "n_expected_groups": len(self.items),
            "n_trajectories": len(rows) * GROUP_SIZE,
            "n_expected_trajectories": len(self.items) * GROUP_SIZE,
            "terminal_successes_by_arm": terminal,
            "replans_by_arm": replans,
        }

    def flush(self, *, status: str) -> dict[str, Any]:
        document = {
            "format_version": FORMAT_VERSION,
            "kind": _MANIFEST_KIND,
            "identity": self.identity,
            "identity_digest": self.identity_digest,
            "split": self.split,
            "started_utc": self.started_utc,
            "updated_utc": _utc(),
            "summary": self._summary(status),
            "groups": self.ordered_groups(),
        }
        _atomic_json(self.manifest_path, document)
        return document

    def finalize(self) -> dict[str, Any]:
        missing = [key for key in self.expected if key not in self.groups]
        _require(not missing,
                 f"cannot finalize recovery collection; {len(missing)} groups missing")
        # Every group was deeply validated at add/adoption time.  Re-hash all
        # bytes at the terminal commit to catch an in-run mutation without a
        # second full tensor deserialisation pass.
        self._verify_committed_sidecars(deep=False)
        return self.flush(status="COMPLETE")


def collection_identity(
    *,
    checkpoint: Mapping[str, Any],
    split: str,
    source_sha256: str,
) -> dict[str, Any]:
    trials = split_trials(split)
    _require(_valid_sha256(source_sha256), "source digest is not SHA-256")
    return {
        "format_version": FORMAT_VERSION,
        "method": "full_trajectory_terminal_grpo_collection",
        "seed_checkpoint": dict(checkpoint),
        "source": {
            "scheme": SOURCE_DIGEST_SCHEME,
            "sha256": source_sha256,
            "files": list(_SOURCE_FILES),
        },
        "split": {
            "name": split,
            "trial_ids": list(trials),
            "official_trial_ids": list(_AGGREGATE_TRIAL_SPLITS["official"]),
            "train_trial_ids": list(_AGGREGATE_TRIAL_SPLITS["train"]),
            "validation_trial_ids": list(_AGGREGATE_TRIAL_SPLITS["validation"]),
            "train_folds": {
                name: list(fold_trials) for name, fold_trials in TRAIN_FOLDS.items()
            },
        },
        "work": {
            "bench": "libero",
            "suites": list(DEFAULT_LIBERO_SUITES),
            "n_tasks": N_TASKS,
            "protocol_seed": PROTOCOL_SEED,
            "max_steps": MAX_STEPS,
            "n_groups": len(collection_items(split)),
        },
        "group": {
            "size": GROUP_SIZE,
            "direct_arms": [0],
            "sampled_arms": list(range(1, GROUP_SIZE)),
            "proposal_distribution": "ordered_plackett_luce_without_replacement",
            "ordered_support_size": TOPK,
            "proposal_seed_scheme": PROPOSAL_SEED_SCHEME,
            "decoder_witness_scheme": DECODER_WITNESS_SCHEME,
            "action_segment_witness_scheme": ACTION_SEGMENT_WITNESS_SCHEME,
            "common_decoder_noise": True,
        },
        "policy": {
            "path": "estimator->proposal->{argmax|ordered_PL}->decoder",
            "q_action": False,
            "bank": False,
            "fallback": False,
            "duration_normalize_segments": False,
            "gripper_dwell": 1,
            "decoder_samples": 1,
        },
        "sidecar": {
            "belief_shape_per_replan": [K, D],
            "coefficient_width": M,
            "old_logprob": "exact_PL_of_stored_order_under_collection_policy",
            "detached": True,
            "reward": "terminal_LIBERO_success_only",
            "atomic_unit": "complete_eight_arm_group",
        },
        "identity_digest_scheme": IDENTITY_DIGEST_SCHEME,
    }


_WORKER_STATE: dict[str, Any] = {}


def _init_worker(
    device_queue: Any,
    checkpoint: Mapping[str, Any],
    split: str,
    collection_identity_digest: str,
    out_dir: str,
    expected_source_digest: str,
) -> None:
    root = Path(__file__).resolve().parents[2]
    _require(source_digest(root) == expected_source_digest,
             "recovery source changed before worker initialization")
    _assert_checkpoint_stat(checkpoint)
    device = claim_device(device_queue)
    from loom.eval import libero                              # noqa: PLC0415

    libero.ensure_libero_runtime()
    bundle = load_recovery_bundle(str(checkpoint["path"]), device=device)
    _WORKER_STATE.clear()
    _WORKER_STATE.update({
        "bundle": bundle,
        "split": split,
        "identity_digest": collection_identity_digest,
        "out_dir": out_dir,
    })


def _worker_collect(item_dict: Mapping[str, Any]) -> dict[str, Any]:
    item = WorkItem(**dict(item_dict))
    return collect_group(
        item,
        _WORKER_STATE["bundle"],
        split=_WORKER_STATE["split"],
        collection_identity_digest=_WORKER_STATE["identity_digest"],
        out_dir=_WORKER_STATE["out_dir"],
        backend="libero",
    )


def run_collection(
    *,
    checkpoint: Mapping[str, Any],
    split: str,
    out_dir: str | os.PathLike[str],
    workers: int,
    quiet: bool = False,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    src = source_digest(root)
    identity = collection_identity(
        checkpoint=checkpoint, split=split, source_sha256=src,
    )
    items = collection_items(split)
    store = RecoveryStore(out_dir, identity=identity, split=split, items=items)
    todo = [item for item in items if not store.has(item)]
    if not todo:
        return store.finalize()

    workers = max(1, min(int(workers), len(todo)))
    if workers == 1:
        _assert_checkpoint_stat(checkpoint)
        _require(source_digest(root) == src,
                 "recovery source changed after manifest creation")
        device = claim_device(None)
        from loom.eval import libero                       # noqa: PLC0415

        libero.ensure_libero_runtime()
        bundle = load_recovery_bundle(str(checkpoint["path"]), device=device)
        for index, item in enumerate(todo, 1):
            _assert_checkpoint_stat(checkpoint)
            _require(source_digest(root) == src,
                     "recovery source changed during collection")
            receipt = collect_group(
                item,
                bundle,
                split=split,
                collection_identity_digest=store.identity_digest,
                out_dir=store.out_dir,
                backend="libero",
            )
            store.add(receipt)
            if not quiet:
                print(
                    f"[outcome-recovery] {index}/{len(todo)} {work_key(item)}",
                    flush=True,
                )
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: PLC0415
        from loom.eval.runner import _mp_context                         # noqa: PLC0415

        context = _mp_context()
        queue = context.Queue()
        n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
        for index in range(workers):
            queue.put(index % n_gpu if n_gpu else None)
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_init_worker,
            initargs=(
                queue,
                dict(checkpoint),
                split,
                store.identity_digest,
                str(store.out_dir),
                src,
            ),
        ) as pool:
            futures = {pool.submit(_worker_collect, item.to_dict()): item
                       for item in todo}
            for index, future in enumerate(as_completed(futures), 1):
                _assert_checkpoint_stat(checkpoint)
                _require(source_digest(root) == src,
                         "recovery source changed during collection")
                receipt = future.result()
                store.add(receipt)
                if not quiet:
                    print(
                        f"[outcome-recovery] {index}/{len(todo)} "
                        f"{receipt['group_id']}",
                        flush=True,
                    )
    return store.finalize()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", required=True,
        help=f"consolidated step-{SEED_GLOBAL_STEP} seed (exact SHA is enforced)",
    )
    parser.add_argument(
        "--out-dir", required=True,
        help="new or identity-matching resumable collection directory",
    )
    parser.add_argument(
        "--split", required=True, choices=tuple(TRIAL_SPLITS),
        help=("fixed LIBERO init-state partition; train0..train5 are the "
              "immutable five-trial/200-group macro-round folds"),
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="one simulator/policy worker per visible GPU by default",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint = authenticate_seed_checkpoint(args.checkpoint)
    from loom.eval.libero import libero_available             # noqa: PLC0415

    if not libero_available():
        raise SystemExit(
            "real LIBERO is unavailable; run with the pinned LOOM_LIBERO_PYTHON "
            "environment. This collector has no fake/stub production mode."
        )
    workers = args.workers
    if workers is None:
        workers = max(1, torch.cuda.device_count() if torch.cuda.is_available() else 1)
    if int(workers) < 1:
        raise SystemExit("--workers must be >= 1")
    result = run_collection(
        checkpoint=checkpoint,
        split=args.split,
        out_dir=args.out_dir,
        workers=int(workers),
        quiet=bool(args.quiet),
    )
    if not args.quiet:
        print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
