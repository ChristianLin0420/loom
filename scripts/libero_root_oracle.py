#!/usr/bin/env python3
"""Common-noise, 16-arm LIBERO root-intervention operator oracle.

For each of the official seed-0 400 LIBERO work items this executable creates
16 fresh policy/env branches from the same reset.  Arm 0 forces the exact
``proposal.argmax`` root; arms 1..15 force independently SHA-seeded proposal
samples.  Only that first coefficient is intervened on.  Every later replan is
the frozen direct R0 policy (``proposal.argmax -> decoder``).

The decoder is called exactly once for the root.  Its private generator begins
from the same state in every arm, and ``q_action`` measures realizability from
the very tensor returned by that call and handed to the policy's execution
path--never from a second, differently-noised decode.  An arm is eligible iff
that L2 residual is at most 0.5.  The masked oracle reward is terminal LIBERO
success only.

This script is deliberately standalone.  It reuses the official eval WorkItem,
environment, policy, rate-conversion, and per-item RNG seams without changing
training, data loading, or normal evaluation behavior.  It writes one nested
group per WorkItem atomically and is resumable.  Final PASS is fail-closed and
requires 400 groups / 6,400 arm rows / zero errors, all parity witnesses, exact
arm-0 reproduction of 149/400 (40/32/48/29), and masked oracle >= 360/400.
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
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import Tensor

import contracts as C
from loom.eval import EvalProtocol
from loom.eval import libero
from loom.eval.policy import (
    LoomPolicy,
    PolicyModules,
    _accepts_keyword,
    _run_model_kwargs,
    make_policy,
    policy_provenance,
    submodule_state,
    to_env_rate,
)
from loom.eval.runner import WorkItem, _mp_context, claim_device, iter_work


FORMAT_VERSION = 1
N_ARMS = 16
N_SAMPLED_ROOTS = N_ARMS - 1
REALIZABILITY_TAU = 0.5
MASKED_ORACLE_MIN = 360
EXPECTED_WORK_ITEMS = 400
EXPECTED_ROWS = EXPECTED_WORK_ITEMS * N_ARMS
EXPECTED_ARM0_BY_SUITE = {
    "libero_spatial": 40,
    "libero_object": 32,
    "libero_goal": 48,
    "libero_long": 29,
}
EXPECTED_ARM0_TOTAL = sum(EXPECTED_ARM0_BY_SUITE.values())
ROOT_SEED_SCHEME = "sha256(work-item,arm)-v1"
CANDIDATE_RECIPE_BANK_ONLY = "action_anchored_bank_only"
CANDIDATE_RECIPE_JOINT_Q_ACTION_BANK = "action_anchored_q_action_bank_joint"
CANDIDATE_RECIPE_JOINT_Q_ACTION_BANK_IDENTITY_CENTERED = (
    "action_anchored_q_action_bank_joint_identity_centered"
)
IDENTITY_CENTERED_RESET = {
    "source_config_hash": "a199324a6205bb6d",
    "tensors": {"bank.omega": "zero"},
}
BEHAVIOR_SOURCE_DIGEST_SCHEME = "sha256(relpath-nul-file-sha256-nul)-v1"
# Complete behavior-bearing source closure for the real root-oracle path.  Keep
# this explicit: resume identity must move when policy construction, reset/RNG,
# image/rate conversion, or any loaded inference module changes in a dirty tree.
BEHAVIOR_SOURCE_FILES = (
    "contracts.py",
    "loom/data/adapters/libero.py",
    "loom/data/canonical.py",
    "loom/data/tower.py",
    "loom/eval/__init__.py",
    "loom/eval/libero.py",
    "loom/eval/policy.py",
    "loom/eval/runner.py",
    "loom/heads/decoder.py",
    "loom/heads/proposal.py",
    "loom/heads/q_action.py",
    "loom/heads/q_delta.py",
    "loom/model/estimator.py",
    "scripts/libero_root_oracle.py",
)
BANK_GATE_REQUIREMENTS = (
    "delta_sel_ci_low_per_horizon",
    "identity_minus_rollout_ci_low_per_horizon",
    "proposal_candidate_leaf_spread_ci_low",
)
JOINT_BANK_GATE_REQUIREMENTS = (
    "deploy_action_semantics_preservation",
    "proposal_root_q_action_residual_preservation",
)
QA_REFERENCE_SHA256 = (
    "15f286c268caa5327d5aa3abf1f67ebd0555c426a509fef22cb7f537bf6ab4e1"
)
QA_REFERENCE_CONFIG_HASH = "a199324a6205bb6d"
QA_REFERENCE_GLOBAL_STEP = 49_666
PARITY_CHECK_KEYS = (
    "sixteen_fresh_branches",
    "identical_env_seed",
    "identical_policy_seed",
    "identical_reset_policy_input",
    "identical_decoder_rng_before",
    "identical_decoder_rng_after",
    "one_forced_root_each",
    "direct_argmax_continuation_only",
    "residual_used_exact_planned_segment",
    "root_gripper_path_unchanged",
    "executed_root_prefix_exact",
    "complete_root_segment_executed",
    "arm0_exact_argmax",
)
DEFAULT_BASELINE_RESULTS = (
    ROOT / "runs/eval_r0a_deploy_s1_s49666_seeded1200_v2/seed0/results.json"
)


class OracleError(RuntimeError):
    """A protocol, provenance, parity, or completeness failure."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OracleError(message)


def official_protocol() -> EvalProtocol:
    """The immutable official seed-0 selection protocol: exactly 400 items."""
    _require(float(C.REALIZABILITY_TAU) == REALIZABILITY_TAU,
             "shared realizability threshold drifted from 0.5")
    protocol = libero.DEFAULT_PROTOCOL.replace(seeds=(0,))
    _require(protocol.bench == "libero", "official protocol bench is not libero")
    _require(protocol.episodes_per_task == 10, "official protocol needs 10 episodes/task")
    _require(protocol.n_tasks == 10, "official protocol needs 10 tasks/suite")
    _require(tuple(protocol.suites) == tuple(libero.SUITES), "official suite order drifted")
    _require(protocol.max_steps == 512, "official protocol max_steps drifted")
    _require(protocol.total_episodes == EXPECTED_WORK_ITEMS, "official protocol is not 400")
    return protocol


def sha256_file(path: str | os.PathLike, chunk_bytes: int = 8 << 20) -> str:
    p = Path(path).expanduser().resolve()
    _require(p.is_file(), f"required file does not exist: {p}")
    h = hashlib.sha256()
    with p.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_bytes)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _is_sha256_hex(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _behavior_digest_from_entries(entries: Sequence[Mapping[str, str]]) -> str:
    """Canonical aggregate over sorted relative paths and their file hashes."""
    h = hashlib.sha256()
    h.update(BEHAVIOR_SOURCE_DIGEST_SCHEME.encode("utf-8") + b"\0")
    for entry in entries:
        rel = str(entry["path"])
        digest = str(entry["sha256"])
        _require(_is_sha256_hex(digest), f"invalid source SHA-256 for {rel!r}")
        h.update(rel.encode("utf-8") + b"\0")
        h.update(bytes.fromhex(digest) + b"\0")
    return h.hexdigest()


def behavior_source_provenance(
    root: str | os.PathLike = ROOT,
    files: Sequence[str] = BEHAVIOR_SOURCE_FILES,
) -> dict[str, Any]:
    """Hash the canonical behavior-source set, failing closed on any omission."""
    source_root = Path(root).expanduser().resolve()
    names = tuple(sorted(str(name) for name in files))
    _require(bool(names), "behavior-source set is empty")
    _require(len(names) == len(set(names)), "behavior-source set contains duplicates")
    entries: list[dict[str, str]] = []
    for name in names:
        rel = Path(name)
        _require(
            name == rel.as_posix() and not rel.is_absolute() and ".." not in rel.parts,
            f"behavior-source path is not a canonical repository-relative path: {name!r}",
        )
        path = (source_root / rel).resolve()
        try:
            path.relative_to(source_root)
        except ValueError as exc:
            raise OracleError(f"behavior source escapes repository root: {name}") from exc
        _require(path.is_file(), f"required behavior source is missing: {name}")
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise OracleError(f"cannot read behavior source {name}: {exc}") from exc
        entries.append({"path": name, "sha256": digest})
    return {
        "behavior_source_digest_scheme": BEHAVIOR_SOURCE_DIGEST_SCHEME,
        "behavior_source_digest": _behavior_digest_from_entries(entries),
        "behavior_source_files": entries,
    }


def assert_behavior_source_digest(
    expected: str,
    *,
    root: str | os.PathLike = ROOT,
    files: Sequence[str] = BEHAVIOR_SOURCE_FILES,
) -> dict[str, Any]:
    """Refuse resume/execution after any behavior-bearing source changes."""
    _require(_is_sha256_hex(expected), "expected behavior-source digest is invalid")
    current = behavior_source_provenance(root, files)
    _require(
        current["behavior_source_digest"] == expected,
        "behavior-bearing source changed after oracle identity was established",
    )
    return current


def _canonical_json_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _hash_value(value: Any) -> str:
    """Stable content digest for reset observations and tensor witnesses."""
    h = hashlib.sha256()

    def update(v: Any) -> None:
        if isinstance(v, Tensor):
            a = v.detach().to(device="cpu")
            if a.dtype == torch.bfloat16:
                a = a.float()
            a = a.contiguous().numpy()
            h.update(b"tensor\0")
            h.update(str(a.dtype).encode())
            h.update(json.dumps(list(a.shape)).encode())
            h.update(a.tobytes(order="C"))
        elif isinstance(v, np.ndarray):
            a = np.ascontiguousarray(v)
            h.update(b"ndarray\0")
            h.update(str(a.dtype).encode())
            h.update(json.dumps(list(a.shape)).encode())
            h.update(a.tobytes(order="C"))
        elif isinstance(v, Mapping):
            h.update(b"mapping\0")
            for key in sorted(v, key=str):
                update(str(key)); update(v[key])
        elif isinstance(v, (list, tuple)):
            h.update(b"sequence\0")
            for item in v:
                update(item)
        elif v is None or isinstance(v, (bool, int, float, str)):
            h.update(json.dumps(v, sort_keys=True, allow_nan=False).encode())
            h.update(b"\0")
        else:
            raise OracleError(f"cannot hash value of type {type(v).__name__}")

    update(value)
    return "sha256:" + h.hexdigest()


def work_key(item: WorkItem | Mapping[str, Any]) -> str:
    d = item.to_dict() if isinstance(item, WorkItem) else item
    return (
        f"{d['bench']}|{d['suite']}|task={int(d['task_id']):02d}|"
        f"episode={int(d['episode']):02d}|seed={int(d['seed'])}"
    )


def root_seed(item: WorkItem | Mapping[str, Any], arm_id: int) -> int:
    _require(1 <= int(arm_id) < N_ARMS, "sampled-root arm must be in [1,15]")
    raw = f"{ROOT_SEED_SCHEME}|{work_key(item)}|arm={int(arm_id)}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & ((1 << 63) - 1)


def _generator_state_sha(generator: torch.Generator | None) -> str:
    _require(generator is not None, "decoder did not receive its private generator")
    return _hash_value(generator.get_state())


def _simplex_evidence(c: Tensor) -> dict[str, Any]:
    _require(c.ndim == 2 and c.shape == (1, C.M),
             f"root coefficient must be (1,{C.M}), got {tuple(c.shape)}")
    x = c.detach().float()
    _require(bool(torch.isfinite(x).all()), "root coefficient is non-finite")
    _require(bool((x >= 0).all()), "root coefficient has negative entries")
    _require(bool(torch.allclose(x.sum(-1), torch.ones(1, device=x.device),
                                 atol=1e-5, rtol=0.0)),
             "root coefficient does not sum to one")
    support = torch.nonzero(x[0] > 0, as_tuple=False).flatten()
    _require(0 < int(support.numel()) <= C.TOPK,
             f"root support has {support.numel()} entries, expected <= {C.TOPK}")
    values = x[0, support]
    return {
        "sha256": _hash_value(x),
        "support": [int(i) for i in support.cpu().tolist()],
        "weights": [float(v) for v in values.cpu().tolist()],
        "sum": float(x.sum()),
    }


@dataclass
class RootCandidateCache:
    """One immutable root set and reset-input witness for a WorkItem."""

    item: WorkItem
    roots: list[Tensor] = field(default_factory=list)
    root_evidence: list[dict[str, Any]] = field(default_factory=list)
    reference_input_sha256: str | None = None
    input_matches: dict[int, bool] = field(default_factory=dict)

    @torch.no_grad()
    def choose(
        self,
        arm_id: int,
        proposal: Any,
        z: Tensor,
        lang: Tensor,
        observation_sha256: str,
    ) -> Tensor:
        signature = _hash_value({
            "observation": observation_sha256,
            "belief": z,
            "language_features": lang,
        })
        if not self.roots:
            _require(arm_id == 0, "arm 0 must establish the shared root set")
            argmax = getattr(proposal, "argmax", None)
            sample = getattr(proposal, "sample", None)
            _require(callable(argmax), "candidate 0 requires exact proposal.argmax")
            _require(callable(sample), "sampled roots require proposal.sample")
            roots = [argmax(z, lang).detach()]
            evidence = [{
                "arm_id": 0,
                "kind": "proposal.argmax",
                "sample_seed": None,
                **_simplex_evidence(roots[0]),
            }]
            for sampled_arm in range(1, N_ARMS):
                seed = root_seed(self.item, sampled_arm)
                generator = torch.Generator(device=z.device)
                generator.manual_seed(seed)
                sampled = sample(z, lang, 1, generator=generator)
                _require(
                    isinstance(sampled, Tensor) and sampled.shape == (1, 1, C.M),
                    f"proposal.sample must return (1,1,{C.M}), got "
                    f"{getattr(sampled, 'shape', None)}",
                )
                root = sampled[:, 0].detach()
                roots.append(root)
                evidence.append({
                    "arm_id": sampled_arm,
                    "kind": "proposal.sample",
                    "sample_seed": seed,
                    **_simplex_evidence(root),
                })
            self.roots = roots
            self.root_evidence = evidence
            self.reference_input_sha256 = signature
        self.input_matches[arm_id] = signature == self.reference_input_sha256
        _require(self.input_matches[arm_id],
                 f"arm {arm_id} did not reach the identical reset/policy input")
        return self.roots[arm_id].to(device=z.device).clone()


class RootInterventionProposal:
    """Force one cached root, then delegate every continuation to argmax."""

    def __init__(self, base: Any, cache: RootCandidateCache, arm_id: int) -> None:
        self.base = base
        self.cache = cache
        self.arm_id = int(arm_id)
        self.calls = 0
        self.n_forced = 0
        self.n_direct = 0
        self.observation_sha256: str | None = None
        self.root_z: Tensor | None = None
        self.root_c: Tensor | None = None

    def argmax(self, z: Tensor, lang: Tensor) -> Tensor:
        self.calls += 1
        if self.calls == 1:
            _require(self.observation_sha256 is not None,
                     "root observation witness was not set before proposal")
            self.root_z = z.detach()
            self.root_c = self.cache.choose(
                self.arm_id, self.base, z, lang, self.observation_sha256,
            )
            self.n_forced += 1
            return self.root_c
        fn = getattr(self.base, "argmax", None)
        _require(callable(fn), "frozen direct continuation requires proposal.argmax")
        self.n_direct += 1
        return fn(z, lang)


class RootAuditDecoder:
    """Decode once and compute residual from that exact returned root segment."""

    def __init__(self, base: Any, q_action: Any, proposal: RootInterventionProposal) -> None:
        self.base = base
        self.q_action = q_action
        self.proposal = proposal
        self.base_accepts_generator = _accepts_keyword(base, "generator")
        self.root: dict[str, Any] | None = None

    def forward(
        self, proprio: Tensor, c: Tensor, *, generator: torch.Generator | None = None,
    ) -> Tensor:
        is_root = self.root is None
        before = _generator_state_sha(generator) if is_root else None
        kwargs = {"generator": generator} if self.base_accepts_generator else {}
        out = self.base(proprio, c, **kwargs)
        _require(isinstance(out, Tensor), "decoder did not return a tensor")
        _require(bool(torch.isfinite(out).all()), "decoder returned a non-finite segment")
        if is_root:
            _require(self.base_accepts_generator,
                     "real oracle decoder must expose the common-noise generator seam")
            _require(self.proposal.root_z is not None and self.proposal.root_c is not None,
                     "decoder ran before the root intervention was established")
            _require(torch.equal(c, self.proposal.root_c),
                     "decoder root coefficient differs from the forced coefficient")
            _require(out.shape == (1, C.H_OP, C.EMBODIMENTS["libero_franka"].dof),
                     f"decoder root segment has wrong shape {tuple(out.shape)}")
            c_hat = self.q_action(out, self.proposal.root_z)
            if isinstance(c_hat, tuple):
                c_hat = c_hat[0]
            _require(isinstance(c_hat, Tensor) and c_hat.shape == c.shape,
                     "q_action returned the wrong coefficient shape")
            residual = (c_hat.float() - c.float()).norm(dim=-1)
            _require(bool(torch.isfinite(residual).all()), "q_action residual is non-finite")
            # Hash in the same canonical fp32 NumPy representation that
            # ``LoomPolicy._plan`` hands to its execution path.  Tensor and
            # ndarray hashes deliberately carry different type tags elsewhere.
            segment_hash = _hash_value(
                out[0].detach().to(device="cpu", dtype=torch.float32).numpy()
            )
            self.root = {
                "decoder_rng_before_sha256": before,
                "decoder_rng_after_sha256": _generator_state_sha(generator),
                "residual_segment_sha256": segment_hash,
                "decoded_segment_sha256": segment_hash,
                "q_action_coeff_sha256": _hash_value(c_hat.detach().float()),
                "residual_l2": float(residual[0]),
            }
        return out


class OracleArmPolicy(LoomPolicy):
    """Normal LoomPolicy with observable root execution; no changed continuation."""

    def __init__(
        self,
        modules: PolicyModules,
        proposal: RootInterventionProposal,
        decoder: RootAuditDecoder,
    ) -> None:
        self._oracle_proposal = proposal
        self._oracle_decoder = decoder
        arm_modules = PolicyModules(
            estimator=modules.estimator,
            proposal=proposal,
            decoder=decoder,
            featurize=modules.featurize,
            embodiment=modules.embodiment,
            device=modules.device,
            is_stub=modules.is_stub,
            meta=modules.meta,
        )
        self._root_plan: np.ndarray | None = None
        self._root_gated: np.ndarray | None = None
        self._root_env_actions: list[np.ndarray] = []
        self._root_env_expected: int | None = None
        super().__init__(
            arm_modules,
            n_candidates=1,
            clip_actions=True,
            op_stats=False,
            gripper_dwell=1,
            decoder_samples=1,
            duration_normalize_segments=False,
        )

    @torch.no_grad()
    def _plan(self, obs: dict, instruction: str) -> np.ndarray:
        if self._oracle_proposal.calls == 0:
            self._oracle_proposal.observation_sha256 = _hash_value(obs)
        seg = super()._plan(obs, instruction)
        if self._root_plan is None:
            self._root_plan = np.asarray(seg, dtype=np.float32).copy()
        return seg

    def _gate_gripper(self, seg: np.ndarray) -> np.ndarray:
        out = super()._gate_gripper(seg)
        if self._root_gated is None:
            self._root_gated = np.asarray(out, dtype=np.float32).copy()
        return out

    def act(self, obs: dict, instruction: str) -> np.ndarray:
        action = super().act(obs, instruction)
        if self._root_env_expected is None:
            _require(self.replans == 1, "first action did not create exactly one root plan")
            self._root_env_expected = int(self.clock.n_steps_dispatched)
        if len(self._root_env_actions) < self._root_env_expected:
            self._root_env_actions.append(np.asarray(action, dtype=np.float32).copy())
        return action

    def oracle_evidence(self) -> dict[str, Any]:
        _require(self._root_plan is not None, "policy never decoded a root segment")
        _require(self._root_gated is not None, "policy never gated its root segment")
        _require(self._root_env_expected is not None, "policy never dispatched its root segment")
        _require(self._oracle_decoder.root is not None, "root decoder audit is missing")
        planned_hash = _hash_value(self._root_plan)
        gated_hash = _hash_value(self._root_gated)
        residual_hash = self._oracle_decoder.root["residual_segment_sha256"]
        expected = to_env_rate(
            self._root_gated,
            self.embodiment,
            self._root_env_expected,
            src_fps=self.env_fps,
        )
        expected = np.clip(expected, self._low, self._high).astype(np.float32)
        executed = np.stack(self._root_env_actions, axis=0)
        expected_prefix = expected[: len(executed)]
        execution_match = (
            executed.shape == expected_prefix.shape
            and np.array_equal(executed, expected_prefix)
        )
        return {
            **self._oracle_decoder.root,
            "planned_root_segment_sha256": planned_hash,
            "post_gripper_root_segment_sha256": gated_hash,
            "residual_uses_planned_segment": residual_hash == planned_hash,
            "root_gripper_path_unchanged": gated_hash == planned_hash,
            "executed_root_prefix_sha256": _hash_value(executed),
            "expected_root_prefix_sha256": _hash_value(expected_prefix),
            "executed_root_prefix_matches": execution_match,
            "root_env_steps_expected": self._root_env_expected,
            "root_env_steps_executed": len(executed),
            "root_env_segment_complete": len(executed) == self._root_env_expected,
            "proposal_calls": self._oracle_proposal.calls,
            "n_forced_roots": self._oracle_proposal.n_forced,
            "n_direct_continuations": self._oracle_proposal.n_direct,
        }


@dataclass
class OracleBundle:
    modules: PolicyModules
    q_action: Any
    provenance: dict[str, Any]


def load_oracle_bundle(checkpoint: str | os.PathLike, device: str) -> OracleBundle:
    """Load direct R0 modules through eval, plus the frozen q_action witness."""
    checkpoint = str(Path(checkpoint).expanduser().resolve())
    direct = make_policy(
        checkpoint,
        embodiment="libero_franka",
        device=device,
        allow_stub=False,
        n_candidates=1,
        op_stats=False,
        gripper_dwell=1,
        decoder_samples=1,
        duration_normalize_segments=False,
    )
    provenance = policy_provenance(direct)
    _require(provenance.get("is_stub") is False, "oracle loaded a stub direct policy")
    _require(_accepts_keyword(direct.modules.decoder, "generator"),
             "candidate decoder has no common-noise generator seam")

    try:
        payload = torch.load(
            checkpoint, map_location="cpu", weights_only=False, mmap=True,
        )
    except Exception as exc:  # noqa: BLE001
        raise OracleError(f"cannot load q_action from {checkpoint}: {exc}") from exc
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    q_state = submodule_state(state, "q_action")
    _require(q_state is not None, "checkpoint has no q_action state")
    from loom.heads.q_action import QAction  # lazy: eval interpreter boundary

    q_kw = _run_model_kwargs(checkpoint, "q_action", payload)
    q_container = QAction(
        embodiments=["libero_franka"],
        default_embodiment="libero_franka",
        **q_kw,
    )
    try:
        incompatible = q_container.load_state_dict(q_state, strict=True)
    except Exception as exc:  # noqa: BLE001
        raise OracleError(f"q_action does not load strictly: {exc}") from exc
    _require(not incompatible.missing_keys and not incompatible.unexpected_keys,
             "q_action state is incomplete")
    q_container.requires_grad_(False).eval().to(device)
    q_action = q_container.body("libero_franka")
    for name, module in (
        ("estimator", direct.modules.estimator),
        ("proposal", direct.modules.proposal),
        ("decoder", direct.modules.decoder),
        ("q_action", q_action),
    ):
        req = getattr(module, "requires_grad_", None)
        if callable(req):
            req(False)
        ev = getattr(module, "eval", None)
        if callable(ev):
            ev()
        for key, value in getattr(module, "state_dict", lambda: {})().items():
            if isinstance(value, Tensor) and value.is_floating_point():
                _require(bool(torch.isfinite(value).all()),
                         f"{name}.{key} contains non-finite values")
    provenance = {
        **provenance,
        "q_action": {
            "type": type(q_action).__module__ + "." + type(q_action).__name__,
            "tensors_loaded": len(q_state),
            "strict": True,
            "frozen": True,
        },
        "oracle_modules_frozen": True,
    }
    del payload, state, q_state, q_container
    return OracleBundle(direct.modules, q_action, provenance)


def _arm_error(item: WorkItem, arm_id: int, message: str) -> dict[str, Any]:
    return {
        "arm_id": int(arm_id),
        "root_kind": "proposal.argmax" if arm_id == 0 else "proposal.sample",
        "root_sample_seed": None if arm_id == 0 else root_seed(item, arm_id),
        "terminal_success": False,
        "eligible": False,
        "residual_l2": None,
        "steps": 0,
        "hit_step_cap": False,
        "n_replans": None,
        "error": message,
    }


def execute_group(
    item: WorkItem,
    bundle: OracleBundle,
    *,
    backend: str = "libero",
    env_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Execute all 16 branches for one WorkItem, sequentially on shared weights."""
    started = time.time()
    cache = RootCandidateCache(item)
    arms: list[dict[str, Any]] = []
    factory = env_factory or libero.make_env
    for arm_id in range(N_ARMS):
        try:
            proposal = RootInterventionProposal(bundle.modules.proposal, cache, arm_id)
            decoder = RootAuditDecoder(bundle.modules.decoder, bundle.q_action, proposal)
            policy = OracleArmPolicy(bundle.modules, proposal, decoder)
            # `_run_item` sets this same seed before `run_episode_safe`; keeping
            # that official seam is what makes root decoder noise common.
            from loom.eval.runner import _run_item  # local: explicit reused seam

            rec = _run_item(item, policy, libero, factory, backend)
            arm: dict[str, Any] = {
                "arm_id": arm_id,
                "root_kind": "proposal.argmax" if arm_id == 0 else "proposal.sample",
                "root_sample_seed": None if arm_id == 0 else root_seed(item, arm_id),
                "terminal_success": bool(rec.success),
                "steps": int(rec.steps),
                "hit_step_cap": bool(rec.hit_step_cap),
                "n_replans": rec.n_replans,
                "error": rec.error,
                "wall_s": float(rec.wall_s),
                "env_seed": item.env_seed,
                "policy_seed": item.policy_seed,
            }
            if rec.error is None:
                evidence = policy.oracle_evidence()
                residual = float(evidence["residual_l2"])
                root_ev = cache.root_evidence[arm_id]
                arm.update({
                    "root": copy.deepcopy(root_ev),
                    "residual_l2": residual,
                    "eligible": math.isfinite(residual) and residual <= REALIZABILITY_TAU,
                    "execution": evidence,
                    "reset_input_sha256": cache.reference_input_sha256,
                    "reset_input_matches": cache.input_matches.get(arm_id, False),
                })
            else:
                arm.update({"residual_l2": None, "eligible": False})
            arms.append(arm)
        except Exception:  # noqa: BLE001 - one arm is a recorded failed row
            arms.append(_arm_error(item, arm_id, traceback.format_exc()))

    successful_eligible = [
        a["arm_id"] for a in arms
        if a.get("error") is None and a.get("eligible") and a.get("terminal_success")
    ]
    valid = [a for a in arms if a.get("error") is None]
    before = {a.get("execution", {}).get("decoder_rng_before_sha256") for a in valid}
    after = {a.get("execution", {}).get("decoder_rng_after_sha256") for a in valid}
    inputs = {a.get("reset_input_sha256") for a in valid}
    parity_checks = {
        "sixteen_fresh_branches": len(arms) == N_ARMS,
        "identical_env_seed": {a.get("env_seed") for a in valid} == {item.env_seed},
        "identical_policy_seed": {a.get("policy_seed") for a in valid} == {item.policy_seed},
        "identical_reset_policy_input": len(inputs) == 1 and None not in inputs,
        "identical_decoder_rng_before": len(before) == 1 and None not in before,
        "identical_decoder_rng_after": len(after) == 1 and None not in after,
        "one_forced_root_each": all(
            a.get("execution", {}).get("n_forced_roots") == 1 for a in valid
        ),
        "direct_argmax_continuation_only": all(
            a.get("execution", {}).get("n_direct_continuations")
            == max(int(a.get("n_replans") or 0) - 1, 0)
            for a in valid
        ),
        "residual_used_exact_planned_segment": all(
            a.get("execution", {}).get("residual_uses_planned_segment") is True
            for a in valid
        ),
        "root_gripper_path_unchanged": all(
            a.get("execution", {}).get("root_gripper_path_unchanged") is True
            for a in valid
        ),
        "executed_root_prefix_exact": all(
            a.get("execution", {}).get("executed_root_prefix_matches") is True
            for a in valid
        ),
        "complete_root_segment_executed": all(
            a.get("execution", {}).get("root_env_segment_complete") is True
            for a in valid
        ),
        "arm0_exact_argmax": bool(
            valid and arms[0].get("root_kind") == "proposal.argmax"
            and arms[0].get("root", {}).get("kind") == "proposal.argmax"
        ),
    }
    parity = {
        "passed": len(valid) == N_ARMS and all(parity_checks.values()),
        "checks": parity_checks,
        "reset_input_sha256": next(iter(inputs)) if len(inputs) == 1 else None,
        "decoder_rng_before_sha256": next(iter(before)) if len(before) == 1 else None,
        "decoder_rng_after_sha256": next(iter(after)) if len(after) == 1 else None,
    }
    return {
        "group_id": work_key(item),
        "work_item": item.to_dict(),
        "arms": arms,
        "parity": parity,
        "arm0_terminal_success": bool(arms[0].get("terminal_success")),
        "unmasked_oracle_success": any(a.get("terminal_success") for a in arms),
        "masked_oracle_success": bool(successful_eligible),
        "successful_eligible_arm_ids": successful_eligible,
        "wall_s": time.time() - started,
    }


def _expected_item_map(protocol: EvalProtocol | None = None) -> dict[str, dict[str, Any]]:
    protocol = protocol or official_protocol()
    return {work_key(item): item.to_dict() for item in iter_work(protocol)}


def validate_baseline_results(path: str | os.PathLike) -> dict[str, Any]:
    """Authenticate the authoritative seed-0 149/400 direct result."""
    p = Path(path).expanduser().resolve()
    digest = sha256_file(p)
    try:
        blob = json.loads(p.read_text())
    except Exception as exc:  # noqa: BLE001
        raise OracleError(f"cannot read baseline results {p}: {exc}") from exc
    protocol = EvalProtocol.from_dict(blob.get("protocol", {}))
    _require(protocol == official_protocol(), "baseline is not the official seed-0 protocol")
    expected = _expected_item_map(protocol)
    episodes = blob.get("episodes")
    _require(isinstance(episodes, list) and len(episodes) == EXPECTED_WORK_ITEMS,
             "baseline must have exactly 400 episode rows")
    outcomes: dict[str, bool] = {}
    suite_success = {suite: 0 for suite in EXPECTED_ARM0_BY_SUITE}
    for row in episodes:
        key = work_key(row)
        _require(key in expected, f"baseline has unknown WorkItem {key}")
        _require(key not in outcomes, f"baseline duplicates WorkItem {key}")
        for field in ("env_seed", "seed", "suite", "task_id", "episode"):
            _require(row.get(field) == expected[key].get(field),
                     f"baseline {key} has wrong {field}")
        _require(row.get("error") is None, f"baseline {key} has an error")
        policy_seed = (row.get("extra") or {}).get("policy_seed")
        _require(policy_seed == expected[key]["policy_seed"],
                 f"baseline {key} has wrong policy_seed")
        success = bool(row.get("success"))
        outcomes[key] = success
        suite_success[str(row["suite"])] += int(success)
    _require(set(outcomes) == set(expected), "baseline WorkItems are incomplete")
    _require(suite_success == EXPECTED_ARM0_BY_SUITE,
             f"baseline suite counts {suite_success} != {EXPECTED_ARM0_BY_SUITE}")
    _require(sum(outcomes.values()) == EXPECTED_ARM0_TOTAL,
             f"baseline total is not {EXPECTED_ARM0_TOTAL}")
    meta = blob.get("meta") or {}
    policy = meta.get("policy") or {}
    _require(meta.get("backend") == "libero", "baseline did not use real LIBERO")
    _require(meta.get("env_available") is True, "baseline has no real-env provenance")
    _require(policy.get("is_stub") is False, "baseline used a stub policy")
    _require(policy.get("ckpt_global_step") == 49_666,
             "baseline is not the selected step-49,666 policy")
    _require(policy.get("embodiment") == "libero_franka",
             "baseline used the wrong embodiment")
    _require(policy.get("decoder_samples") == 1,
             "baseline did not use one decoder sample")
    _require(policy.get("gripper_dwell") == 1,
             "baseline enabled a different gripper path")
    _require(policy.get("duration_normalize_segments") is False,
             "baseline enabled duration-normalized segments")
    _require(meta.get("policy_seed_scheme") == "sha256(work-item)-v1",
             "baseline policy seed scheme drifted")
    policy_kw = ((meta.get("eval_identity") or {}).get("policy_kw") or {})
    _require(policy_kw.get("allow_stub") is False,
             "baseline identity did not fail closed on stub loading")
    return {
        "path": str(p),
        "sha256": digest,
        "n_work_items": len(outcomes),
        "n_errors": 0,
        "n_success": EXPECTED_ARM0_TOTAL,
        "suite_success": suite_success,
        "checkpoint": meta.get("ckpt"),
        "checkpoint_global_step": policy.get("ckpt_global_step"),
        "outcomes": outcomes,
    }


def authenticate_candidate_recipe(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Authenticate either frozen-q_action bank fit or its exact joint successor.

    The joint recipe is one coupled method delta: q_action joins the bank in the
    optimizer, its action coefficient stays attached to L_dyn, and both modules'
    old Adam state is reset.  Accepting any partial mixture would make the
    promoted artifact scientifically ambiguous, so the four switches are
    validated as an atomic recipe identity.
    """
    _require(isinstance(cfg, Mapping), "candidate config is not a mapping")
    no_default = object()
    missing = object()

    def get(path: str, default: Any = no_default) -> Any:
        node: Any = cfg
        for part in path.split("."):
            if not isinstance(node, Mapping) or part not in node:
                if default is not no_default:
                    return default
                raise OracleError(f"candidate config is missing {path}")
            node = node[part]
        return node

    def exact(path: str, wanted: Any) -> None:
        got = get(path)
        matches = got is wanted if type(wanted) is bool else got == wanted
        _require(matches, f"candidate {path} must be {wanted!r}, got {got!r}")

    def exact_number(path: str, wanted: float) -> float:
        got = get(path)
        _require(
            type(got) in (int, float) and math.isfinite(float(got))
            and float(got) == float(wanted),
            f"candidate {path} must be {wanted}, got {got!r}",
        )
        return float(got)

    # Shared action-anchored stage identity.  These are inherited unchanged by
    # the joint child; checking them prevents an unrelated trainable-q_action
    # recipe from being mislabeled as the approved successor.
    exact("run.steps", 80_000)
    exact("data.source", "libero")
    exact("data.embodiments", ["libero_franka"])
    exact("data.sampling", "uniform_window")
    exact("data.trajectory_split", "train")
    exact("data.holdout_demo_keys", ["demo_49"])
    exact("data.recurrent_burn_in", 4)
    exact("optim.update_ema", False)
    exact("losses.dyn.enabled", True)
    exact_number("losses.dyn.weight", 1.0)
    exact("losses.dyn.coeff_source", "q_action")
    exact("losses.dyn.negatives", "within_trajectory")
    exact("losses.dyn.min_gap", 2)
    exact_number("losses.dyn.neg_margin", 0.1)
    exact("losses.dyn.cosine", "per_slot")
    for name in ("proposal", "balance", "potential", "grpo"):
        exact(f"losses.{name}.enabled", False)
        exact_number(f"losses.{name}.weight", 0.0)

    scales = get("optim.lr_scales")
    _require(isinstance(scales, Mapping), "candidate optim.lr_scales is not a mapping")
    exact_number("optim.lr_scales.bank", 0.1)
    for name in ("estimator", "q_delta", "ema", "proposal", "decoder", "potential"):
        exact_number(f"optim.lr_scales.{name}", 0.0)

    train_modules = get("train_modules")
    reset_modules = get("optim.reset_state_modules")
    detach_raw = get("losses.dyn.detach_coeff", missing)
    q_action_lr = get("optim.lr_scales.q_action")
    _require(
        type(q_action_lr) in (int, float) and math.isfinite(float(q_action_lr)),
        f"candidate optim.lr_scales.q_action must be finite, got {q_action_lr!r}",
    )
    neg_weight = get("losses.dyn.neg_weight")
    _require(
        type(neg_weight) in (int, float) and math.isfinite(float(neg_weight)),
        f"candidate losses.dyn.neg_weight must be finite, got {neg_weight!r}",
    )

    if train_modules == ["bank"]:
        exact("losses.act.enabled", False)
        exact_number("losses.act.weight", 0.0)
        _require(
            reset_modules == ["bank"],
            "bank-only candidate must reset exactly optim state [bank]",
        )
        _require(
            float(q_action_lr) == 0.0,
            "bank-only candidate must keep optim.lr_scales.q_action at 0.0",
        )
        _require(
            detach_raw is missing or detach_raw is True,
            "bank-only candidate must retain detached q_action coefficients",
        )
        _require(
            float(neg_weight) in (1.0, 4.0),
            "bank-only candidate must be the base or N4 action-anchored recipe",
        )
        kind = CANDIDATE_RECIPE_BANK_ONLY
        detach_coeff = True
    elif train_modules == ["bank", "q_action"]:
        exact("losses.act.enabled", True)
        exact_number("losses.act.weight", 1.0)
        exact("losses.act.align_to", "q_a")
        exact("losses.act.decode_from", "q_action")
        _require(
            reset_modules == ["bank", "q_action"],
            "joint candidate must reset exactly optim state [bank, q_action]",
        )
        _require(
            float(q_action_lr) == 1.0,
            "joint candidate must set optim.lr_scales.q_action to 1.0",
        )
        _require(
            detach_raw is False,
            "joint candidate must set losses.dyn.detach_coeff to false",
        )
        _require(
            float(neg_weight) == 4.0,
            "joint candidate must inherit the N4 dynamics negative weight",
        )
        run_name = get("run.name")
        parameter_reset = get("optim.transition_parameter_reset", missing)
        if run_name == "r0a_bank_ca_qa":
            _require(
                parameter_reset is missing,
                "base joint QA candidate must not declare a parameter reset",
            )
            kind = CANDIDATE_RECIPE_JOINT_Q_ACTION_BANK
        elif run_name == "r0a_bank_ca_qa_omega0":
            _require(
                parameter_reset == IDENTITY_CENTERED_RESET,
                "identity-centered candidate transition reset must be exact",
            )
            kind = CANDIDATE_RECIPE_JOINT_Q_ACTION_BANK_IDENTITY_CENTERED
        else:
            raise OracleError(f"joint candidate run.name is not declared: {run_name!r}")
        detach_coeff = False
    else:
        raise OracleError(
            "candidate train_modules must be exactly [bank] or [bank, q_action], "
            f"got {train_modules!r}"
        )

    recipe = {
        "kind": kind,
        "train_modules": list(train_modules),
        "detach_coeff": detach_coeff,
        "detach_coeff_explicit": detach_raw is not missing,
        "reset_state_modules": list(reset_modules),
        "bank_lr_scale": 0.1,
        "q_action_lr_scale": float(q_action_lr),
        "dyn_neg_weight": float(neg_weight),
    }
    if kind in (
        CANDIDATE_RECIPE_JOINT_Q_ACTION_BANK,
        CANDIDATE_RECIPE_JOINT_Q_ACTION_BANK_IDENTITY_CENTERED,
    ):
        recipe["action_anchor"] = {
            "enabled": True,
            "weight": 1.0,
            "align_to": "q_a",
            "decode_from": "q_action",
        }
    if kind == CANDIDATE_RECIPE_JOINT_Q_ACTION_BANK_IDENTITY_CENTERED:
        recipe["transition_parameter_reset"] = copy.deepcopy(
            IDENTITY_CENTERED_RESET
        )
    return recipe


def checkpoint_provenance(path: str | os.PathLike) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    digest = sha256_file(p)
    try:
        payload = torch.load(str(p), map_location="cpu", weights_only=False, mmap=True)
    except Exception as exc:  # noqa: BLE001
        raise OracleError(f"cannot inspect checkpoint {p}: {exc}") from exc
    _require(isinstance(payload, dict), "checkpoint is not a mapping")
    _require(isinstance(payload.get("consolidated"), dict),
             "oracle requires a consolidated checkpoint")
    _require(isinstance(payload.get("model"), dict), "checkpoint has no model state")
    for name in ("estimator", "proposal", "decoder", "q_action", "bank"):
        _require(submodule_state(payload["model"], name) is not None,
                 f"checkpoint has no {name} state")
    cfg = payload.get("resolved_config")
    _require(isinstance(cfg, dict), "checkpoint has no embedded resolved_config")
    config_hash = str(payload.get("config_hash", ""))
    experiment = {k: v for k, v in cfg.items() if k != "link"}
    got = hashlib.blake2b(
        json.dumps(experiment, sort_keys=True, default=str).encode(), digest_size=8,
    ).hexdigest()
    _require(config_hash and got == config_hash,
             "checkpoint resolved_config does not match config_hash")

    candidate_recipe = authenticate_candidate_recipe(cfg)
    step = payload.get("global_step")
    _require(isinstance(step, int) and step >= 49_666, "candidate has invalid global_step")
    out = {
        "path": str(p),
        "sha256": digest,
        "bytes": p.stat().st_size,
        "global_step": step,
        "samples_seen": payload.get("samples_seen"),
        "config_hash": config_hash,
        "resolved_config_sha256": _canonical_json_sha256(cfg),
        "candidate_recipe": candidate_recipe,
        "saved_git_sha": payload.get("git_sha"),
        "world_size": payload.get("world_size"),
        "consolidated": payload.get("consolidated"),
    }
    del payload
    return out


def validate_bank_gate(
    path: str | os.PathLike,
    checkpoint_sha256: str,
    checkpoint_config_hash: str | None = None,
    checkpoint_recipe: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Require the offline heldout promotion report for this exact candidate."""
    p = Path(path).expanduser().resolve()
    digest = sha256_file(p)
    try:
        blob = json.loads(p.read_text())
    except Exception as exc:  # noqa: BLE001
        raise OracleError(f"cannot read bank gate report {p}: {exc}") from exc
    _require(blob.get("passed") is True and blob.get("status") == "PASS",
             "bank gate report is not PASS")
    _require(blob.get("format_version") == 1,
             "bank gate report format is not recognized")
    _require(blob.get("overall_verdict") == "PASS",
             "bank gate overall verdict is not PASS")
    _require(blob.get("direct_e2e_run") is False,
             "bank gate report is not the required offline gate")
    candidate = blob.get("candidate") or {}
    _require(candidate.get("sha256") == checkpoint_sha256,
             "bank gate candidate hash does not match --checkpoint")
    if checkpoint_config_hash is not None:
        _require(candidate.get("config_hash") == checkpoint_config_hash,
                 "bank gate candidate config hash does not match --checkpoint")
    data = blob.get("data") or {}
    _require(str(data.get("manifest_digest", "")).startswith("sha256:"),
             "bank gate has no heldout manifest digest")
    manifest = data.get("trajectory_manifest") or {}
    _require(manifest.get("digest") == data.get("manifest_digest"),
             "bank gate manifest digest fields disagree")
    _require(manifest.get("source") == "libero" and manifest.get("split") == "gate",
             "bank gate did not use the heldout LIBERO trajectory split")
    gates = blob.get("gates") or {}
    method_variant = (blob.get("recipe") or {}).get("method_variant")
    if checkpoint_recipe is not None:
        expected_kind = checkpoint_recipe.get("kind")
        if expected_kind == CANDIDATE_RECIPE_JOINT_Q_ACTION_BANK:
            _require(method_variant == "joint_q_action_bank",
                     "joint checkpoint lacks its full QA preservation gate")
        elif expected_kind == CANDIDATE_RECIPE_JOINT_Q_ACTION_BANK_IDENTITY_CENTERED:
            _require(method_variant == "joint_q_action_bank_identity_centered",
                     "identity-centered checkpoint lacks its exact QA gate")
        elif expected_kind == CANDIDATE_RECIPE_BANK_ONLY:
            _require(method_variant in (None, "bank_only"),
                     "bank-only checkpoint has a mismatched gate method")
        else:
            raise OracleError("checkpoint recipe kind is not authenticated")
    joint = method_variant in (
        "joint_q_action_bank", "joint_q_action_bank_identity_centered",
    )
    expected_gates = set(BANK_GATE_REQUIREMENTS)
    if joint:
        expected_gates.update(JOINT_BANK_GATE_REQUIREMENTS)
    _require(set(gates) == expected_gates,
             "bank gate named requirements are incomplete")
    for name in BANK_GATE_REQUIREMENTS:
        verdict = gates[name]
        _require(isinstance(verdict, Mapping) and verdict.get("passed") is True,
                 f"bank gate requirement {name} did not pass")
        threshold = verdict.get("threshold")
        _require(isinstance(threshold, (int, float)) and math.isfinite(float(threshold)),
                 f"bank gate requirement {name} has no finite threshold")
        lower = verdict.get("ci_lower_per_horizon", verdict.get("ci_lower"))
        lowers = lower if isinstance(lower, list) else [lower]
        _require(bool(lowers), f"bank gate requirement {name} has no CI lower bound")
        _require(all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            and float(value) > float(threshold)
            for value in lowers
        ), f"bank gate requirement {name} is not strictly above threshold")
    if joint:
        semantics = gates["deploy_action_semantics_preservation"]
        _require(isinstance(semantics, Mapping) and semantics.get("passed") is True,
                 "joint bank gate action-semantics preservation did not pass")
        thresholds = semantics.get("thresholds") or {}
        _require(thresholds == {
            "action_decode_improvement_ci_low": 0.0,
            "proposal_support_overlap_change_ci_low": -0.05,
        }, "joint bank gate preservation thresholds drifted")
        action_lower = semantics.get("action_decode_improvement_ci_low")
        support_lower = semantics.get("proposal_support_overlap_change_ci_low")
        _require(type(action_lower) in (int, float) and math.isfinite(float(action_lower))
                 and float(action_lower) >= 0.0,
                 "joint action decoding regressed against deploy reference")
        _require(type(support_lower) in (int, float) and math.isfinite(float(support_lower))
                 and float(support_lower) >= -0.05,
                 "joint proposal/q_action support exceeded non-inferiority margin")

        residual = gates["proposal_root_q_action_residual_preservation"]
        _require(isinstance(residual, Mapping) and residual.get("passed") is True,
                 "joint root realizability preservation did not pass")
        _require(residual.get("q_action_residual_max") == 0.5,
                 "joint realizability residual threshold drifted")
        _require(residual.get("max_root_exhaustion_rate") == 0.01,
                 "joint realizability exhaustion threshold drifted")
        rate = residual.get("root_exhaustion_rate")
        _require(type(rate) in (int, float) and math.isfinite(float(rate))
                 and float(rate) < 0.01,
                 "joint root realizability exhaustion did not pass")

        reference = blob.get("reference") or {}
        _require(reference.get("sha256") == QA_REFERENCE_SHA256,
                 "joint gate deploy reference SHA is not authenticated")
        _require(reference.get("config_hash") == QA_REFERENCE_CONFIG_HASH,
                 "joint gate deploy reference config is not authenticated")
        _require(reference.get("global_step") == QA_REFERENCE_GLOBAL_STEP,
                 "joint gate deploy reference step is not authenticated")
        _require(reference.get("model_config_exact") is True,
                 "joint gate did not preserve model construction config")
        frozen = reference.get("frozen_modules") or {}
        expected_frozen = {
            "estimator", "ema", "q_delta", "decoder", "proposal", "potential",
        }
        _require(set(frozen) == expected_frozen,
                 "joint gate frozen-module provenance is incomplete")
        _require(all(
            isinstance(item, Mapping) and item.get("tensor_exact") is True
            and (
                (item.get("present") is True
                 and isinstance(item.get("tensors"), int) and item["tensors"] > 0)
                or (name == "potential" and item.get("present") is False
                    and item.get("tensors") == 0)
            )
            for name, item in frozen.items()
        ), "joint gate did not prove frozen-module tensor exactness/absence")
    return {
        "path": str(p),
        "sha256": digest,
        "status": "PASS",
        "passed": True,
        "candidate_sha256": candidate.get("sha256"),
        "candidate_config_hash": candidate.get("config_hash"),
        "method_variant": method_variant or "bank_only",
        "manifest_digest": data.get("manifest_digest"),
        "gate_source_sha256": (blob.get("source_provenance") or {}).get(
            "gate_source_sha256"
        ),
    }


def source_provenance() -> dict[str, Any]:
    source = Path(__file__).resolve()
    behavior = behavior_source_provenance()
    try:
        git = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            stderr=subprocess.DEVNULL, timeout=30,
        ).decode().strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT,
            stderr=subprocess.DEVNULL, timeout=30,
        ).decode().strip())
    except Exception:  # noqa: BLE001
        git, dirty = "unknown", None
    return {
        "script": str(source),
        "script_sha256": sha256_file(source),
        **behavior,
        "git_sha": git,
        "git_dirty": dirty,
        "python": sys.version,
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


def recipe() -> dict[str, Any]:
    return {
        "name": "common-noise-root-intervention-libero-oracle",
        "version": 1,
        "n_arms": N_ARMS,
        "arm0": "exact proposal.argmax root",
        "sampled_arms": N_SAMPLED_ROOTS,
        "sample_seed_scheme": ROOT_SEED_SCHEME,
        "intervention": "force root coefficient only",
        "continuation": "frozen direct proposal.argmax -> decoder",
        "decoder_noise": "same WorkItem policy_seed and generator state in all arms",
        "realizability": "L2(q_action(exact decoded/executed root segment,z), root_c)",
        "root_execution": "complete decoded root segment must execute exactly",
        "realizability_tau": REALIZABILITY_TAU,
        "eligibility": "residual_l2 <= 0.5",
        "reward": "terminal LIBERO success only",
        "expected_work_items": EXPECTED_WORK_ITEMS,
        "expected_rows": EXPECTED_ROWS,
        "required_arm0_success": EXPECTED_ARM0_TOTAL,
        "required_arm0_suite_success": EXPECTED_ARM0_BY_SUITE,
        "required_masked_oracle_success": MASKED_ORACLE_MIN,
        "decoder_samples": 1,
        "gripper_dwell": 1,
        "duration_normalize_segments": False,
    }


def _is_sha256_witness(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    digest = value.removeprefix("sha256:")
    return len(digest) == 64 and all(ch in "0123456789abcdef" for ch in digest)


def _arm_schema_failures(
    key: str,
    item: Mapping[str, Any],
    arm: Mapping[str, Any],
    arm_id: int,
) -> list[str]:
    """Independently authenticate one successful arm row before aggregation."""
    prefix = f"{key}/arm{arm_id}"
    failures: list[str] = []

    def need(condition: bool, message: str) -> None:
        if not condition:
            failures.append(f"{prefix}: {message}")

    expected_kind = "proposal.argmax" if arm_id == 0 else "proposal.sample"
    expected_seed = None if arm_id == 0 else root_seed(item, arm_id)
    need(arm.get("arm_id") == arm_id, "arm id mismatch")
    need(arm.get("root_kind") == expected_kind, "root kind mismatch")
    need(arm.get("root_sample_seed") == expected_seed, "root sample seed mismatch")
    need(arm.get("env_seed") == item["env_seed"], "env seed mismatch")
    need(arm.get("policy_seed") == item["policy_seed"], "policy seed mismatch")
    need(type(arm.get("terminal_success")) is bool, "terminal success is not boolean")
    need(type(arm.get("eligible")) is bool, "eligibility is not boolean")
    need(type(arm.get("hit_step_cap")) is bool, "step-cap flag is not boolean")
    steps = arm.get("steps")
    n_replans = arm.get("n_replans")
    need(isinstance(steps, int) and 1 <= steps <= int(item["max_steps"]),
         "episode steps are invalid")
    need(isinstance(n_replans, int) and n_replans >= 1, "replan count is invalid")
    if isinstance(steps, int):
        need(
            arm.get("hit_step_cap")
            is (not bool(arm.get("terminal_success")) and steps >= int(item["max_steps"])),
            "step-cap flag disagrees with the terminal outcome",
        )
    wall_s = arm.get("wall_s")
    need(isinstance(wall_s, (int, float)) and math.isfinite(float(wall_s))
         and float(wall_s) >= 0.0, "wall time is invalid")

    residual = arm.get("residual_l2")
    need(isinstance(residual, (int, float)) and math.isfinite(float(residual)),
         "residual is invalid")
    if isinstance(residual, (int, float)) and math.isfinite(float(residual)):
        need(arm.get("eligible") is (float(residual) <= REALIZABILITY_TAU),
             "eligibility does not equal residual <= tau")
    need(arm.get("reset_input_matches") is True, "reset/policy input did not match")
    need(_is_sha256_witness(arm.get("reset_input_sha256")),
         "reset/policy input witness is invalid")

    root = arm.get("root")
    if not isinstance(root, Mapping):
        failures.append(f"{prefix}: root evidence is missing")
    else:
        need(root.get("arm_id") == arm_id, "root evidence arm id mismatch")
        need(root.get("kind") == expected_kind, "root evidence kind mismatch")
        need(root.get("sample_seed") == expected_seed,
             "root evidence sample seed mismatch")
        support = root.get("support")
        weights = root.get("weights")
        support_ok = (
            isinstance(support, list)
            and 1 <= len(support) <= C.TOPK
            and all(isinstance(index, int) and 0 <= index < C.M for index in support)
            and support == sorted(set(support))
        )
        weights_ok = (
            isinstance(weights, list) and isinstance(support, list)
            and len(weights) == len(support)
            and all(isinstance(value, (int, float)) and math.isfinite(float(value))
                    and float(value) > 0.0 for value in weights)
        )
        need(support_ok, "root simplex support is invalid")
        need(weights_ok, "root simplex weights are invalid")
        if support_ok and weights_ok:
            coeff = torch.zeros(1, C.M, dtype=torch.float32)
            coeff[0, support] = torch.tensor(weights, dtype=torch.float32)
            need(abs(float(coeff.sum()) - 1.0) <= 1e-5,
                 "root simplex weights do not sum to one")
            root_sum = root.get("sum")
            need(isinstance(root_sum, (int, float)) and math.isfinite(float(root_sum))
                 and abs(float(root_sum) - float(coeff.sum())) <= 1e-5,
                 "root simplex recorded sum is inconsistent")
            need(root.get("sha256") == _hash_value(coeff),
                 "root coefficient witness does not match support/weights")
        need(_is_sha256_witness(root.get("sha256")),
             "root coefficient witness is invalid")

    execution = arm.get("execution")
    if not isinstance(execution, Mapping):
        failures.append(f"{prefix}: execution evidence is missing")
    else:
        execution_residual = execution.get("residual_l2")
        need(isinstance(execution_residual, (int, float))
             and math.isfinite(float(execution_residual)),
             "execution residual is invalid")
        if isinstance(residual, (int, float)) and isinstance(
            execution_residual, (int, float)
        ):
            need(float(execution_residual) == float(residual),
                 "row residual differs from q_action execution evidence")
        for field in (
            "decoder_rng_before_sha256", "decoder_rng_after_sha256",
            "residual_segment_sha256", "decoded_segment_sha256",
            "q_action_coeff_sha256", "planned_root_segment_sha256",
            "post_gripper_root_segment_sha256", "executed_root_prefix_sha256",
            "expected_root_prefix_sha256",
        ):
            need(_is_sha256_witness(execution.get(field)),
                 f"execution witness {field} is invalid")
        planned = execution.get("planned_root_segment_sha256")
        need(execution.get("residual_segment_sha256") == planned,
             "q_action did not consume the exact planned root segment")
        need(execution.get("decoded_segment_sha256") == planned,
             "decoded and planned root segments differ")
        need(execution.get("post_gripper_root_segment_sha256") == planned,
             "gripper path changed the root segment")
        need(
            execution.get("executed_root_prefix_sha256")
            == execution.get("expected_root_prefix_sha256"),
            "executed root prefix differs from the planned rate conversion",
        )
        for field in (
            "residual_uses_planned_segment", "root_gripper_path_unchanged",
            "executed_root_prefix_matches",
        ):
            need(execution.get(field) is True, f"execution parity {field} failed")
        if isinstance(n_replans, int):
            need(execution.get("proposal_calls") == n_replans,
                 "proposal calls do not equal replans")
            need(execution.get("n_forced_roots") == 1,
                 "branch did not force exactly one root")
            need(execution.get("n_direct_continuations") == max(n_replans - 1, 0),
                 "continuation was not direct argmax only")
        n_expected = execution.get("root_env_steps_expected")
        n_executed = execution.get("root_env_steps_executed")
        need(isinstance(n_expected, int) and n_expected >= 1,
             "expected root execution length is invalid")
        need(isinstance(n_executed, int) and isinstance(n_expected, int)
             and 1 <= n_executed <= n_expected,
             "executed root prefix length is invalid")
        if isinstance(n_expected, int) and isinstance(n_executed, int):
            need(execution.get("root_env_segment_complete") is (n_executed == n_expected),
                 "root segment completeness flag is inconsistent")
            need(n_executed == n_expected,
                 "terminal branch did not execute the complete root segment")
    return failures


def summarize_groups(
    groups: Sequence[Mapping[str, Any]],
    *,
    baseline_outcomes: Mapping[str, bool],
    provenance: Mapping[str, Any],
    require_real: bool = True,
) -> dict[str, Any]:
    """Validate all rows and compute the fail-closed promotion decision."""
    failures: list[str] = []
    expected = _expected_item_map()
    if set(baseline_outcomes) != set(expected):
        failures.append("authoritative baseline WorkItem set is incomplete or mismatched")
    if any(type(value) is not bool for value in baseline_outcomes.values()):
        failures.append("authoritative baseline outcomes are not boolean")
    by_key: dict[str, Mapping[str, Any]] = {}
    n_rows = 0
    n_errors = 0
    n_parity_failures = 0
    for group in groups:
        if not isinstance(group, Mapping):
            failures.append("non-mapping WorkItem group")
            continue
        key = str(group.get("group_id", ""))
        if key not in expected:
            failures.append(f"unknown WorkItem group {key!r}")
            continue
        if key in by_key:
            failures.append(f"duplicate WorkItem group {key}")
            continue
        by_key[key] = group
        if group.get("work_item") != expected[key]:
            failures.append(f"{key}: WorkItem provenance mismatch")
        arms = group.get("arms")
        if not isinstance(arms, list) or len(arms) != N_ARMS:
            failures.append(f"{key}: expected {N_ARMS} arms")
            continue
        n_rows += len(arms)
        ids = [a.get("arm_id") for a in arms if isinstance(a, Mapping)]
        if ids != list(range(N_ARMS)):
            failures.append(f"{key}: arm ids are not exactly 0..15")
        parity_failed = False
        for arm_id, arm in enumerate(arms):
            if not isinstance(arm, Mapping):
                n_errors += 1
                parity_failed = True
                failures.append(f"{key}: non-mapping arm row")
                continue
            if arm.get("error") is not None:
                n_errors += 1
                parity_failed = True
                continue
            arm_failures = _arm_schema_failures(key, expected[key], arm, arm_id)
            failures.extend(arm_failures)
            parity_failed = parity_failed or bool(arm_failures)
            residual = arm.get("residual_l2")
            if not isinstance(residual, (int, float)) or not math.isfinite(float(residual)):
                failures.append(f"{key}/arm{arm.get('arm_id')}: invalid residual")
        raw_parity = group.get("parity")
        parity = raw_parity if isinstance(raw_parity, Mapping) else {}
        raw_checks = parity.get("checks")
        checks = raw_checks if isinstance(raw_checks, Mapping) else {}
        if parity.get("passed") is not True \
                or set(checks) != set(PARITY_CHECK_KEYS) \
                or any(checks.get(name) is not True for name in PARITY_CHECK_KEYS):
            parity_failed = True
        if all(isinstance(arm, Mapping) and arm.get("error") is None for arm in arms):
            reset_inputs = {arm.get("reset_input_sha256") for arm in arms}
            executions = [
                arm.get("execution") if isinstance(arm.get("execution"), Mapping) else {}
                for arm in arms
            ]
            rng_before = {
                execution.get("decoder_rng_before_sha256") for execution in executions
            }
            rng_after = {
                execution.get("decoder_rng_after_sha256") for execution in executions
            }
            if len(reset_inputs) != 1 or None in reset_inputs:
                failures.append(f"{key}: reset/policy input differs across arms")
                parity_failed = True
            if len(rng_before) != 1 or None in rng_before:
                failures.append(f"{key}: decoder RNG start differs across arms")
                parity_failed = True
            if len(rng_after) != 1 or None in rng_after:
                failures.append(f"{key}: decoder RNG consumption differs across arms")
                parity_failed = True
            if parity.get("reset_input_sha256") not in reset_inputs:
                failures.append(f"{key}: group reset witness is inconsistent")
                parity_failed = True
            if parity.get("decoder_rng_before_sha256") not in rng_before:
                failures.append(f"{key}: group decoder-start witness is inconsistent")
                parity_failed = True
            if parity.get("decoder_rng_after_sha256") not in rng_after:
                failures.append(f"{key}: group decoder-end witness is inconsistent")
                parity_failed = True

            arm0 = bool(arms[0].get("terminal_success"))
            masked_ids = [
                arm.get("arm_id") for arm in arms
                if arm.get("eligible") and arm.get("terminal_success")
            ]
            unmasked = any(bool(arm.get("terminal_success")) for arm in arms)
            need_derived = (
                ("arm0_terminal_success", arm0),
                ("masked_oracle_success", bool(masked_ids)),
                ("unmasked_oracle_success", unmasked),
                ("successful_eligible_arm_ids", masked_ids),
            )
            for field, wanted in need_derived:
                if group.get(field) != wanted:
                    failures.append(f"{key}: derived field {field} is inconsistent")
                    parity_failed = True
        n_parity_failures += int(parity_failed)

    missing = sorted(set(expected) - set(by_key))
    if missing:
        failures.append(f"missing {len(missing)} official WorkItem groups")
    if len(by_key) != EXPECTED_WORK_ITEMS:
        failures.append(f"n_work_items {len(by_key)} != {EXPECTED_WORK_ITEMS}")
    if n_rows != EXPECTED_ROWS:
        failures.append(f"n_rows {n_rows} != {EXPECTED_ROWS}")
    if n_errors != 0:
        failures.append(f"n_errors {n_errors} != 0")
    if n_parity_failures != 0:
        failures.append(f"parity failures {n_parity_failures} != 0")

    arm0_suite = {suite: 0 for suite in EXPECTED_ARM0_BY_SUITE}
    oracle_suite = {suite: 0 for suite in EXPECTED_ARM0_BY_SUITE}
    arm0_total = masked_total = unmasked_total = 0
    arm0_reference_mismatches: list[str] = []
    n_eligible = 0
    for key, group in by_key.items():
        arms = group.get("arms") or []
        if len(arms) != N_ARMS or not all(isinstance(arm, Mapping) for arm in arms):
            continue
        suite = expected[key]["suite"]
        arm0 = bool(arms[0].get("terminal_success"))
        masked = any(
            a.get("error") is None and a.get("eligible")
            and a.get("terminal_success") for a in arms
        )
        unmasked = any(a.get("error") is None and a.get("terminal_success") for a in arms)
        arm0_total += int(arm0)
        masked_total += int(masked)
        unmasked_total += int(unmasked)
        arm0_suite[suite] += int(arm0)
        oracle_suite[suite] += int(masked)
        n_eligible += sum(bool(a.get("eligible")) for a in arms)
        if key in baseline_outcomes and arm0 != bool(baseline_outcomes[key]):
            arm0_reference_mismatches.append(key)

    if arm0_total != EXPECTED_ARM0_TOTAL:
        failures.append(f"arm0 success {arm0_total} != {EXPECTED_ARM0_TOTAL}")
    if arm0_suite != EXPECTED_ARM0_BY_SUITE:
        failures.append(f"arm0 suite counts {arm0_suite} != {EXPECTED_ARM0_BY_SUITE}")
    if arm0_reference_mismatches:
        failures.append(
            f"arm0 differs from authoritative result on "
            f"{len(arm0_reference_mismatches)} WorkItems"
        )
    if masked_total < MASKED_ORACLE_MIN:
        failures.append(f"masked oracle {masked_total}/400 < {MASKED_ORACLE_MIN}/400")

    required_provenance = (
        ("checkpoint", "sha256"),
        ("checkpoint", "config_hash"),
        ("checkpoint", "candidate_recipe"),
        ("bank_gate", "sha256"),
        ("bank_gate", "manifest_digest"),
        ("baseline", "sha256"),
        ("source", "script_sha256"),
        ("source", "behavior_source_digest"),
    )
    for section, field in required_provenance:
        if not (provenance.get(section) or {}).get(field):
            failures.append(f"missing provenance {section}.{field}")
    checkpoint_prov = provenance.get("checkpoint") or {}
    gate_prov = provenance.get("bank_gate") or {}
    baseline_prov = provenance.get("baseline") or {}
    source_prov = provenance.get("source") or {}
    source_entries = source_prov.get("behavior_source_files")
    if source_prov.get("behavior_source_digest_scheme") != BEHAVIOR_SOURCE_DIGEST_SCHEME:
        failures.append("behavior-source digest scheme is invalid")
    if not isinstance(source_entries, list) or not all(
        isinstance(entry, Mapping) for entry in source_entries
    ):
        failures.append("behavior-source file provenance is missing or invalid")
    else:
        expected_paths = list(BEHAVIOR_SOURCE_FILES)
        got_paths = [entry.get("path") for entry in source_entries]
        if got_paths != expected_paths:
            failures.append("behavior-source file set is incomplete or reordered")
        elif not all(_is_sha256_hex(entry.get("sha256")) for entry in source_entries):
            failures.append("behavior-source file hash is invalid")
        else:
            recorded_digest = source_prov.get("behavior_source_digest")
            if not _is_sha256_hex(recorded_digest) \
                    or _behavior_digest_from_entries(source_entries) != recorded_digest:
                failures.append("behavior-source aggregate digest is inconsistent")
            script_rows = [
                entry for entry in source_entries
                if entry.get("path") == "scripts/libero_root_oracle.py"
            ]
            if len(script_rows) != 1 \
                    or script_rows[0].get("sha256") != source_prov.get("script_sha256"):
                failures.append("root-oracle script/source digests are inconsistent")
    if not isinstance(checkpoint_prov.get("global_step"), int) \
            or checkpoint_prov.get("global_step", -1) < 49_666:
        failures.append("candidate checkpoint global step is invalid")
    recorded_recipe = checkpoint_prov.get("candidate_recipe")
    if not isinstance(recorded_recipe, Mapping):
        failures.append("candidate checkpoint recipe provenance is invalid")
    else:
        kind = recorded_recipe.get("kind")
        shared_recipe_ok = recorded_recipe.get("bank_lr_scale") == 0.1
        if kind == CANDIDATE_RECIPE_BANK_ONLY:
            recipe_ok = (
                recorded_recipe.get("train_modules") == ["bank"]
                and recorded_recipe.get("detach_coeff") is True
                and recorded_recipe.get("reset_state_modules") == ["bank"]
                and recorded_recipe.get("q_action_lr_scale") == 0.0
                and recorded_recipe.get("dyn_neg_weight") in (1.0, 4.0)
            )
        elif kind == CANDIDATE_RECIPE_JOINT_Q_ACTION_BANK:
            recipe_ok = (
                recorded_recipe.get("train_modules") == ["bank", "q_action"]
                and recorded_recipe.get("detach_coeff") is False
                and recorded_recipe.get("detach_coeff_explicit") is True
                and recorded_recipe.get("reset_state_modules") == ["bank", "q_action"]
                and recorded_recipe.get("q_action_lr_scale") == 1.0
                and recorded_recipe.get("dyn_neg_weight") == 4.0
                and recorded_recipe.get("action_anchor") == {
                    "enabled": True,
                    "weight": 1.0,
                    "align_to": "q_a",
                    "decode_from": "q_action",
                }
            )
        elif kind == CANDIDATE_RECIPE_JOINT_Q_ACTION_BANK_IDENTITY_CENTERED:
            recipe_ok = (
                recorded_recipe.get("train_modules") == ["bank", "q_action"]
                and recorded_recipe.get("detach_coeff") is False
                and recorded_recipe.get("detach_coeff_explicit") is True
                and recorded_recipe.get("reset_state_modules") == ["bank", "q_action"]
                and recorded_recipe.get("q_action_lr_scale") == 1.0
                and recorded_recipe.get("dyn_neg_weight") == 4.0
                and recorded_recipe.get("action_anchor") == {
                    "enabled": True,
                    "weight": 1.0,
                    "align_to": "q_a",
                    "decode_from": "q_action",
                }
                and recorded_recipe.get("transition_parameter_reset")
                == IDENTITY_CENTERED_RESET
            )
        else:
            recipe_ok = False
        if not shared_recipe_ok or not recipe_ok:
            failures.append("candidate checkpoint recipe provenance is inconsistent")
    if gate_prov.get("passed") is not True or gate_prov.get("status") != "PASS":
        failures.append("bank promotion provenance is not PASS")
    if gate_prov.get("candidate_sha256") != checkpoint_prov.get("sha256"):
        failures.append("bank gate/checkpoint SHA provenance is inconsistent")
    if gate_prov.get("candidate_config_hash") != checkpoint_prov.get("config_hash"):
        failures.append("bank gate/checkpoint config provenance is inconsistent")
    candidate_kind = (checkpoint_prov.get("candidate_recipe") or {}).get("kind")
    expected_gate_variant = {
        CANDIDATE_RECIPE_BANK_ONLY: "bank_only",
        CANDIDATE_RECIPE_JOINT_Q_ACTION_BANK: "joint_q_action_bank",
        CANDIDATE_RECIPE_JOINT_Q_ACTION_BANK_IDENTITY_CENTERED:
            "joint_q_action_bank_identity_centered",
    }.get(candidate_kind)
    if gate_prov.get("method_variant") != expected_gate_variant:
        failures.append("bank gate/checkpoint method provenance is inconsistent")
    if baseline_prov.get("checkpoint_global_step") != 49_666 \
            or baseline_prov.get("n_success") != EXPECTED_ARM0_TOTAL \
            or baseline_prov.get("n_errors") != 0:
        failures.append("authoritative step-49,666 baseline provenance is invalid")
    workers = provenance.get("policy_workers") or []
    if not workers:
        failures.append("missing policy worker provenance")
    else:
        for worker in workers:
            policy = (worker.get("policy") or {}) if isinstance(worker, Mapping) else {}
            if policy.get("is_stub") is not False:
                failures.append("a policy worker used a stub")
            if policy.get("oracle_modules_frozen") is not True \
                    or (policy.get("q_action") or {}).get("strict") is not True \
                    or (policy.get("q_action") or {}).get("frozen") is not True:
                failures.append("a policy worker lacks frozen strict q_action provenance")
            if policy.get("decoder_samples") != 1 \
                    or policy.get("gripper_dwell") != 1 \
                    or policy.get("duration_normalize_segments") is not False \
                    or policy.get("embodiment") != "libero_franka":
                failures.append("a policy worker used the wrong direct inference recipe")
            if policy.get("ckpt_config_hash") != checkpoint_prov.get("config_hash"):
                failures.append("a policy worker loaded a different checkpoint config")
            if policy.get("ckpt_global_step") != checkpoint_prov.get("global_step"):
                failures.append("a policy worker loaded a different checkpoint step")
    if require_real:
        runtime = provenance.get("runtime") or {}
        if runtime.get("backend") != "libero" or runtime.get("env_available") is not True:
            failures.append("oracle did not run against the real LIBERO backend")

    return {
        "passed": not failures,
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "n_work_items": len(by_key),
        "n_expected_work_items": EXPECTED_WORK_ITEMS,
        "n_rows": n_rows,
        "n_expected_rows": EXPECTED_ROWS,
        "n_errors": n_errors,
        "n_parity_failures": n_parity_failures,
        "arm0": {
            "n_success": arm0_total,
            "n": EXPECTED_WORK_ITEMS,
            "success_rate": 100.0 * arm0_total / EXPECTED_WORK_ITEMS,
            "suite_success": arm0_suite,
            "reference_mismatches": arm0_reference_mismatches,
            "reference_exact": not arm0_reference_mismatches,
        },
        "masked_oracle": {
            "n_success": masked_total,
            "n": EXPECTED_WORK_ITEMS,
            "success_rate": 100.0 * masked_total / EXPECTED_WORK_ITEMS,
            "suite_success": oracle_suite,
            "required_n_success": MASKED_ORACLE_MIN,
        },
        "unmasked_oracle": {
            "n_success": unmasked_total,
            "n": EXPECTED_WORK_ITEMS,
            "success_rate": 100.0 * unmasked_total / EXPECTED_WORK_ITEMS,
        },
        "eligibility": {
            "n_eligible_rows": n_eligible,
            "n_rows": n_rows,
            "rate": (100.0 * n_eligible / n_rows) if n_rows else 0.0,
            "tau": REALIZABILITY_TAU,
        },
    }


def atomic_write_json(path: str | os.PathLike, value: Mapping[str, Any]) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n"
    )
    os.replace(tmp, target)


class OracleStore:
    """Parent-owned, group-atomic, identity-checked resumable result store."""

    def __init__(
        self,
        path: str | os.PathLike,
        *,
        identity: Mapping[str, Any],
        provenance: Mapping[str, Any],
        resume: bool,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.identity = copy.deepcopy(dict(identity))
        self.provenance = copy.deepcopy(dict(provenance))
        self.groups: dict[str, dict[str, Any]] = {}
        self.started = time.strftime("%Y-%m-%dT%H:%M:%S")
        if resume and self.path.is_file():
            old = json.loads(self.path.read_text())
            _require(old.get("identity") == self.identity,
                     "existing oracle output has a different immutable identity")
            self.started = old.get("started", self.started)
            for group in old.get("groups", []):
                key = str(group.get("group_id", ""))
                _require(key and key not in self.groups, "existing output has duplicate groups")
                self.groups[key] = group
            for worker in (old.get("provenance") or {}).get("policy_workers", []):
                self.add_worker_provenance(worker)

    def add_worker_provenance(self, worker: Mapping[str, Any]) -> None:
        workers = self.provenance.setdefault("policy_workers", [])
        value = copy.deepcopy(dict(worker))
        key = (value.get("pid"), value.get("device"))
        if not any((v.get("pid"), v.get("device")) == key for v in workers):
            workers.append(value)

    def add(self, group: Mapping[str, Any]) -> None:
        key = str(group.get("group_id", ""))
        _require(key in _expected_item_map(), f"worker returned unknown group {key!r}")
        self.groups[key] = copy.deepcopy(dict(group))
        self.flush_running()

    def ordered_groups(self) -> list[dict[str, Any]]:
        order = _expected_item_map()
        return [self.groups[k] for k in order if k in self.groups]

    def _document(self, summary: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "kind": "libero_common_noise_root_operator_oracle",
            "identity": self.identity,
            "protocol": official_protocol().to_dict(),
            "recipe": recipe(),
            "provenance": self.provenance,
            "started": self.started,
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "summary": dict(summary),
            "groups": self.ordered_groups(),
        }

    def flush_running(self) -> None:
        n_groups = len(self.groups)
        n_rows = sum(len(g.get("arms", [])) for g in self.groups.values())
        atomic_write_json(self.path, self._document({
            "passed": False,
            "status": "RUNNING",
            "n_work_items": n_groups,
            "n_expected_work_items": EXPECTED_WORK_ITEMS,
            "n_rows": n_rows,
            "n_expected_rows": EXPECTED_ROWS,
        }))

    def finalize(self, baseline_outcomes: Mapping[str, bool], *, require_real: bool) -> dict:
        summary = summarize_groups(
            self.ordered_groups(),
            baseline_outcomes=baseline_outcomes,
            provenance=self.provenance,
            require_real=require_real,
        )
        document = self._document(summary)
        atomic_write_json(self.path, document)
        return document


_WORKER: dict[str, Any] = {}


def _init_worker(
    device_queue: Any,
    checkpoint: str,
    backend: str,
    behavior_source_digest: str,
) -> None:
    assert_behavior_source_digest(behavior_source_digest)
    device = claim_device(device_queue)
    if backend == "libero":
        libero.ensure_libero_runtime()
    bundle = load_oracle_bundle(checkpoint, device)
    _WORKER.update({
        "bundle": bundle,
        "backend": backend,
        "reported": False,
        "provenance": {
            "pid": os.getpid(),
            "device": device,
            "policy": bundle.provenance,
        },
    })


def _worker_group(item_dict: Mapping[str, Any]) -> dict[str, Any]:
    item = WorkItem(**dict(item_dict))
    try:
        group = execute_group(item, _WORKER["bundle"], backend=_WORKER["backend"])
    except Exception:  # noqa: BLE001
        message = traceback.format_exc()
        group = {
            "group_id": work_key(item),
            "work_item": item.to_dict(),
            "arms": [_arm_error(item, arm, message) for arm in range(N_ARMS)],
            "parity": {"passed": False, "checks": {}, "fatal": message},
            "arm0_terminal_success": False,
            "unmasked_oracle_success": False,
            "masked_oracle_success": False,
            "successful_eligible_arm_ids": [],
            "wall_s": 0.0,
        }
    if not _WORKER["reported"]:
        group["_worker_provenance"] = _WORKER["provenance"]
        _WORKER["reported"] = True
    return group


def _run_parallel(
    todo: Sequence[WorkItem],
    store: OracleStore,
    checkpoint: str,
    backend: str,
    workers: int,
    behavior_source_digest: str,
) -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    assert_behavior_source_digest(behavior_source_digest)
    ctx = _mp_context()
    queue = ctx.Queue()
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    for i in range(workers):
        queue.put(i % n_gpu if n_gpu else None)
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(queue, checkpoint, backend, behavior_source_digest),
    ) as pool:
        futures = [pool.submit(_worker_group, item.to_dict()) for item in todo]
        for future in as_completed(futures):
            group = future.result()
            assert_behavior_source_digest(behavior_source_digest)
            worker = group.pop("_worker_provenance", None)
            if worker is not None:
                store.add_worker_provenance(worker)
            store.add(group)
    assert_behavior_source_digest(behavior_source_digest)


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol = official_protocol()
    items = iter_work(protocol)
    baseline = validate_baseline_results(args.baseline_results)
    checkpoint = checkpoint_provenance(args.checkpoint)
    bank_gate = validate_bank_gate(
        args.bank_gate, checkpoint["sha256"], checkpoint["config_hash"],
        checkpoint["candidate_recipe"],
    )
    backend = str(args.backend)
    _require(backend == "libero", "official oracle requires the real LIBERO backend")
    env_available = bool(libero.libero_available())
    _require(env_available, "real LIBERO is unavailable; refusing fake fallback")

    source = source_provenance()
    behavior_source_digest = source["behavior_source_digest"]
    assert_behavior_source_digest(behavior_source_digest)
    identity = {
        "version": 1,
        "checkpoint_sha256": checkpoint["sha256"],
        "bank_gate_sha256": bank_gate["sha256"],
        "baseline_results_sha256": baseline["sha256"],
        "script_sha256": source["script_sha256"],
        "behavior_source_digest": behavior_source_digest,
        "protocol_sha256": _canonical_json_sha256(protocol.to_dict()),
        "recipe_sha256": _canonical_json_sha256(recipe()),
        "backend": backend,
    }
    provenance = {
        "checkpoint": checkpoint,
        "bank_gate": bank_gate,
        "baseline": {k: v for k, v in baseline.items() if k != "outcomes"},
        "source": source,
        "runtime": {
            "backend": backend,
            "env_available": env_available,
            "libero_runtime_status": (
                libero.ensure_libero_runtime() if backend == "libero" else None
            ),
        },
        "policy_workers": [],
    }
    store = OracleStore(
        args.out,
        identity=identity,
        provenance=provenance,
        resume=not args.no_resume,
    )
    done = set(store.groups)
    todo = [item for item in items if work_key(item) not in done]
    workers = max(1, min(int(args.workers or 1), max(1, len(todo))))
    if todo and workers > 1:
        _run_parallel(
            todo,
            store,
            str(Path(args.checkpoint).resolve()),
            backend,
            workers,
            behavior_source_digest,
        )
    elif todo:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if backend == "libero":
            libero.ensure_libero_runtime()
        bundle = load_oracle_bundle(args.checkpoint, device)
        store.add_worker_provenance({
            "pid": os.getpid(), "device": device, "policy": bundle.provenance,
        })
        for item in todo:
            assert_behavior_source_digest(behavior_source_digest)
            group = execute_group(item, bundle, backend=backend)
            assert_behavior_source_digest(behavior_source_digest)
            store.add(group)
    assert_behavior_source_digest(behavior_source_digest)
    return store.finalize(baseline["outcomes"], require_real=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True,
                        help="promoted consolidated bank-fit checkpoint")
    parser.add_argument("--bank-gate", required=True,
                        help="PASS report from scripts/bank_ca_gate.py")
    parser.add_argument("--baseline-results", default=str(DEFAULT_BASELINE_RESULTS),
                        help="authoritative step-49666 seed-0 149/400 result")
    parser.add_argument("--out", required=True, help="atomic grouped JSON output")
    parser.add_argument("--workers", type=int, default=None,
                        help="default: one process per visible GPU")
    parser.add_argument("--backend", choices=("libero", "fake"), default="libero")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workers is None:
        from loom.eval.runner import n_devices

        args.workers = n_devices()
    try:
        result = run(args)
        summary = result["summary"]
        print(json.dumps({
            "status": summary["status"],
            "passed": summary["passed"],
            "n_rows": summary["n_rows"],
            "n_errors": summary["n_errors"],
            "arm0": summary["arm0"]["n_success"],
            "masked_oracle": summary["masked_oracle"]["n_success"],
            "out": str(Path(args.out).resolve()),
        }, indent=2))
        return 0 if summary["passed"] else 1
    except Exception as exc:  # noqa: BLE001 - persist preflight failure too
        requested_out = Path(args.out).expanduser().resolve()
        error_out = (
            requested_out
            if not requested_out.exists()
            else requested_out.with_name(requested_out.name + ".error.json")
        )
        try:
            failure_source = source_provenance()
        except Exception as source_exc:  # noqa: BLE001 - preserve original failure
            failure_source = {
                "error": f"{type(source_exc).__name__}: {source_exc}",
            }
        report = {
            "format_version": FORMAT_VERSION,
            "kind": "libero_common_noise_root_operator_oracle",
            "summary": {
                "passed": False,
                "status": "ERROR",
                "failures": [f"{type(exc).__name__}: {exc}"],
            },
            "requested": vars(args),
            "requested_out": str(requested_out),
            "traceback": traceback.format_exc(),
            "source_provenance": failure_source,
        }
        atomic_write_json(error_out, report)
        print(json.dumps({
            "status": "ERROR",
            "passed": False,
            "error": report["summary"]["failures"][0],
            "out": str(error_out),
            "requested_out_preserved": error_out != requested_out,
        }, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
