"""Common-noise, realizability-filtered LIBERO operator oracle.

This is the corrected form of the historical ``search_ceiling_env.py``
measurement.  Each environment condition is run under exactly sixteen
closed-loop policy arms:

* arm 0 uses the deployed ``argmax pi_c`` at every replan;
* arms 1..15 independently draw one ``pi_c`` sample from their arm-specific
  proposal stream at every replan.

All arms receive the same work-item decoder seed.  Each sampled arm has an
independent SHA-derived proposal stream, replacing the old process-salted
``hash()`` seeds while retaining fifteen independent proposal policies.  A
sampled coefficient that fails
``||q_a(D_e(p,c),z)-c|| <= tau`` falls back to the direct coefficient.  The
decoder generator is restored before that fallback, so the executed direct
segment uses exactly the same flow noise as arm 0 and every policy consumes one
decoder draw per replan.  This removes the confound in the old 79.17% number,
where every arm was seeded differently.

The oracle is terminal success over the sixteen complete rollouts.  It does
not call the operator bank or Phi, assign a shaped reward, or alter the default
R0 path.  ``loom.eval.policy.make_policy`` reaches this module only when its
explicit ``operator_oracle`` option is present.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from contracts import EMBODIMENTS, H_OP, M, REALIZABILITY_TAU, TOPK, assert_simplex
from loom.eval import DEFAULT_LIBERO_SUITES, EvalProtocol
from loom.eval.policy import (
    LoomPolicy,
    PolicyModules,
    _argmax_coeff,
    _call,
    _module_dtype,
    feats_to,
    load_policy,
    zeros_featurizer,
)

__all__ = [
    "ORACLE_CANDIDATES",
    "ORACLE_SAMPLED_CANDIDATES",
    "ORACLE_VERSION",
    "PROPOSAL_SEED_SCHEME",
    "OracleArmSpec",
    "OperatorOraclePolicy",
    "load_operator_oracle_policy",
    "aggregate_oracle_results",
    "build_parser",
    "main",
]


ORACLE_VERSION = 1
ORACLE_CANDIDATES = 16
ORACLE_SAMPLED_CANDIDATES = ORACLE_CANDIDATES - 1
PROPOSAL_SEED_SCHEME = "sha256(operator-oracle-proposal|policy-seed|arm)-v1"

# A dirty git suffix is insufficient provenance: two different working trees
# both say ``<sha>-dirty``.  These are the behavior-bearing sources hashed into
# the oracle manifest and every arm's resume identity.
_SOURCE_FILES = (
    "contracts.py",
    "loom/data/canonical.py",
    "loom/data/tower.py",
    "loom/eval/__init__.py",
    "loom/eval/libero.py",
    "loom/eval/operator_oracle.py",
    "loom/eval/policy.py",
    "loom/eval/runner.py",
    "loom/heads/decoder.py",
    "loom/heads/proposal.py",
    "loom/heads/q_action.py",
    "scripts/operator_oracle.py",
)


@dataclass(frozen=True)
class OracleArmSpec:
    """Immutable behavior identity for one of the sixteen oracle arms."""

    arm: int
    expected_config_hash: str
    expected_global_step: int
    checkpoint_sha256: str
    source_digest: str
    version: int = ORACLE_VERSION
    n_candidates: int = ORACLE_CANDIDATES
    tau: float = REALIZABILITY_TAU
    proposal_seed_scheme: str = PROPOSAL_SEED_SCHEME

    def __post_init__(self) -> None:
        if self.version != ORACLE_VERSION:
            raise ValueError(
                f"operator-oracle version must be {ORACLE_VERSION}, got {self.version}"
            )
        if self.n_candidates != ORACLE_CANDIDATES:
            raise ValueError(
                f"operator oracle is fixed at candidate 0 + 15 samples "
                f"({ORACLE_CANDIDATES} total), got {self.n_candidates}"
            )
        if not 0 <= int(self.arm) < ORACLE_CANDIDATES:
            raise ValueError(
                f"oracle arm must be in [0,{ORACLE_CANDIDATES - 1}], got {self.arm}"
            )
        if float(self.tau) != float(REALIZABILITY_TAU):
            raise ValueError(
                f"realizability tau is the frozen contract value "
                f"{REALIZABILITY_TAU}, got {self.tau}"
            )
        if not self.expected_config_hash:
            raise ValueError("expected_config_hash is required; refusing an unauthenticated run")
        if int(self.expected_global_step) < 0:
            raise ValueError("expected_global_step must be non-negative")
        for name, digest in (("checkpoint_sha256", self.checkpoint_sha256),
                             ("source_digest", self.source_digest)):
            if len(str(digest)) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if self.proposal_seed_scheme != PROPOSAL_SEED_SCHEME:
            raise ValueError(
                f"proposal seed scheme must be {PROPOSAL_SEED_SCHEME!r}, got "
                f"{self.proposal_seed_scheme!r}"
            )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "OracleArmSpec":
        if not isinstance(raw, Mapping):
            raise TypeError("operator_oracle must be a JSON-like mapping")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"unknown operator-oracle options: {unknown}")
        return cls(**dict(raw))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _accepts_keyword(fn: Any, name: str) -> bool:
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        p.name == name or p.kind == inspect.Parameter.VAR_KEYWORD for p in params
    )


def proposal_seed(policy_seed: int, arm: int) -> int:
    if not 1 <= int(arm) < ORACLE_CANDIDATES:
        raise ValueError("proposal seeds are defined only for sampled arms 1..15")
    raw = f"operator-oracle-proposal|{int(policy_seed)}|arm={int(arm)}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


class OperatorOraclePolicy(LoomPolicy):
    """One fixed arm of the sixteen-rollout operator ceiling measurement.

    Arm zero deliberately repeats the same estimator -> argmax -> decoder
    operations as ``LoomPolicy._plan``.  The additional q_action forward is
    read-only and occurs after decoding, so its emitted action is bit-identical
    to direct R0 under the same private decoder generator.
    """

    def __init__(
        self,
        modules: PolicyModules,
        *,
        arm: int,
        tau: float = REALIZABILITY_TAU,
        n_candidates: int = ORACLE_CANDIDATES,
        op_stats: bool = False,
        gripper_dwell: int = 1,
        decoder_samples: int = 1,
        duration_normalize_segments: bool = False,
    ) -> None:
        self.oracle_arm = int(arm)
        self.oracle_n_candidates = int(n_candidates)
        self.oracle_tau = float(tau)
        if self.oracle_n_candidates != ORACLE_CANDIDATES:
            raise ValueError(
                f"oracle requires exactly {ORACLE_CANDIDATES} candidates, got "
                f"{self.oracle_n_candidates}"
            )
        if not 0 <= self.oracle_arm < self.oracle_n_candidates:
            raise ValueError(f"invalid oracle arm {self.oracle_arm}")
        if self.oracle_tau != float(REALIZABILITY_TAU):
            raise ValueError(
                f"oracle tau is frozen at {REALIZABILITY_TAU}, got {self.oracle_tau}"
            )
        if modules.q_action is None:
            raise ValueError("operator oracle requires checkpoint-loaded q_action")
        if decoder_samples != 1:
            raise ValueError("operator oracle fixes decoder_samples=1 for common noise")
        if gripper_dwell != 1 or duration_normalize_segments:
            raise ValueError(
                "operator oracle measures the direct execution recipe: "
                "gripper_dwell=1 and duration_normalize_segments=False"
            )
        if not _accepts_keyword(getattr(modules.decoder, "forward", modules.decoder),
                                "generator"):
            raise ValueError(
                "operator oracle requires decoder.forward(..., generator=): "
                "common per-replan flow noise cannot be certified otherwise"
            )
        sample = getattr(modules.proposal, "sample", None)
        if not callable(sample) or not _accepts_keyword(sample, "generator"):
            raise ValueError(
                "operator oracle requires proposal.sample(..., generator=): "
                "the fifteen arms need independent reproducible proposal streams"
            )
        super().__init__(
            modules,
            n_candidates=n_candidates,
            op_stats=op_stats,
            gripper_dwell=gripper_dwell,
            decoder_samples=decoder_samples,
            duration_normalize_segments=duration_normalize_segments,
        )

    def reset(self) -> None:
        super().reset()
        self._oracle_log: list[dict[str, Any]] = []
        self._proposal_generator: torch.Generator | None = None
        if self._policy_seed is not None:
            self._reset_proposal_generator()

    def set_policy_seed(self, seed: int) -> None:
        super().set_policy_seed(seed)
        self._reset_proposal_generator()

    def _reset_proposal_generator(self) -> None:
        if self._policy_seed is None:
            raise RuntimeError("policy seed must be set before oracle sampling")
        self._proposal_generator = None
        if self.oracle_arm == 0:
            return
        self._proposal_generator = torch.Generator(device=self.device)
        self._proposal_generator.manual_seed(
            proposal_seed(self._policy_seed, self.oracle_arm),
        )

    def _decode(self, proprio: Tensor, c: Tensor) -> Tensor:
        if self._decoder_generator is None:
            raise RuntimeError(
                "operator oracle has no private decoder generator; runner must call "
                "set_policy_seed before each episode"
            )
        return _call(
            self.modules.decoder, proprio, c, generator=self._decoder_generator,
        )

    def _residual(self, a: Tensor, z: Tensor, c: Tensor) -> Tensor:
        c_hat = _call(self.modules.q_action, a, z)
        if c_hat.shape != c.shape:
            raise ValueError(
                f"q_action must return coefficient shape {tuple(c.shape)}, got "
                f"{tuple(c_hat.shape)}"
            )
        return (c_hat.float() - c.float()).norm(dim=-1)

    @torch.no_grad()
    def _plan(self, obs: dict, instruction: str) -> np.ndarray:
        m = self.modules
        feats = m.featurize(obs, instruction)
        feats = feats_to(feats, self.device, _module_dtype(m.estimator))

        z = _call(m.estimator, feats, self._z)
        self._z = z
        proprio = feats["proprio"]

        if self.oracle_arm == 0:
            requested = _argmax_coeff(m.proposal, z, feats["lang"],
                                      self.oracle_n_candidates)
        else:
            if self._proposal_generator is None:
                raise RuntimeError("proposal generator was not initialized")
            # One persistent, independently SHA-seeded PL stream per arm.  Its
            # next draw is consumed at every replan.
            candidates = m.proposal.sample(
                z, feats["lang"], 1,
                generator=self._proposal_generator,
            )
            want = (z.shape[0], 1, M)
            if tuple(candidates.shape) != want:
                raise ValueError(
                    f"proposal.sample must emit {want}, got {tuple(candidates.shape)}"
                )
            assert_simplex(candidates)
            requested = candidates[:, 0]
        assert_simplex(requested)

        # Save the state immediately before decoding.  A rejected sample is
        # replaced by direct argmax after restoring this state, so the fallback
        # uses the same flow noise and leaves the stream advanced exactly once.
        decoder_state = self._decoder_generator.get_state()
        a = self._decode(proprio, requested)
        requested_residual = self._residual(a, z, requested)
        rejected = self.oracle_arm > 0 and bool(
            (requested_residual > self.oracle_tau).any()
        )
        executed = requested
        executed_residual = requested_residual
        if rejected:
            self._decoder_generator.set_state(decoder_state)
            direct = _argmax_coeff(
                m.proposal, z, feats["lang"], self.oracle_n_candidates,
            )
            assert_simplex(direct)
            executed = direct
            a = self._decode(proprio, executed)
            executed_residual = self._residual(a, z, executed)

        exhausted = bool((executed_residual > self.oracle_tau).any())
        self.last_coeff = executed
        if self.op_stats:
            self._log_operator(m.proposal, z, feats["lang"], executed)
        self._oracle_log.append({
            "requested_residual": float(requested_residual.max()),
            "executed_residual": float(executed_residual.max()),
            "rejected": bool(rejected),
            "gate_exhausted": bool(exhausted),
            "fallback_to_argmax": bool(rejected),
        })

        a_np = a.detach().to(torch.float32).cpu().numpy()
        if a_np.ndim == 3:
            a_np = a_np[0]
        if a_np.shape != (H_OP, self.spec.dof):
            raise ValueError(
                f"decoder must emit ({H_OP}, {self.spec.dof}), got {a_np.shape}"
            )
        return a_np

    def op_stats_summary(self) -> dict[str, Any]:
        out = super().op_stats_summary()
        log = self._oracle_log
        req = [r["requested_residual"] for r in log]
        exe = [r["executed_residual"] for r in log]
        out.update({
            "oracle_version": ORACLE_VERSION,
            "oracle_arm": self.oracle_arm,
            "oracle_mode": "direct_argmax" if self.oracle_arm == 0 else "proposal_sample",
            "oracle_n_candidates": self.oracle_n_candidates,
            "oracle_tau": self.oracle_tau,
            "oracle_proposal_seed_scheme": PROPOSAL_SEED_SCHEME,
            "oracle_common_decoder_noise": True,
            "oracle_n_replans": len(log),
            "oracle_n_rejected": sum(r["rejected"] for r in log),
            "oracle_n_gate_exhausted": sum(r["gate_exhausted"] for r in log),
            "oracle_requested_residual_mean": (sum(req) / len(req) if req else None),
            "oracle_requested_residual_max": (max(req) if req else None),
            "oracle_executed_residual_mean": (sum(exe) / len(exe) if exe else None),
            "oracle_executed_residual_max": (max(exe) if exe else None),
        })
        return out


def load_operator_oracle_policy(
    ckpt: str | None,
    *,
    oracle: Mapping[str, Any],
    embodiment: str = "libero_franka",
    device: str = "cpu",
    allow_stub: bool = False,
    n_candidates: int = ORACLE_CANDIDATES,
    op_stats: bool = False,
    gripper_dwell: int = 1,
    decoder_samples: int = 1,
    duration_normalize_segments: bool = False,
) -> OperatorOraclePolicy:
    """Load a real, provenance-matched policy plus q_action for one arm."""
    spec = OracleArmSpec.from_mapping(oracle)
    if ckpt is None:
        raise ValueError("operator oracle requires a consolidated checkpoint")
    if allow_stub:
        raise ValueError("operator oracle refuses allow_stub=True")
    root = Path(__file__).resolve().parents[2]
    got_source = _source_digest(root)
    if got_source != spec.source_digest:
        raise RuntimeError(
            f"operator-oracle source digest {got_source} != manifest "
            f"{spec.source_digest}; refusing mixed behavior"
        )
    base = load_policy(
        ckpt,
        embodiment=embodiment,
        device=device,
        allow_stub=False,
        n_candidates=n_candidates,
        op_stats=op_stats,
        gripper_dwell=gripper_dwell,
        decoder_samples=decoder_samples,
        duration_normalize_segments=duration_normalize_segments,
        _include_q_action=True,
    )
    got_hash = str(base.modules.meta.get("ckpt_config_hash") or "")
    got_step = base.modules.meta.get("ckpt_global_step")
    if got_hash != spec.expected_config_hash:
        raise RuntimeError(
            f"operator-oracle checkpoint config hash {got_hash!r} != expected "
            f"{spec.expected_config_hash!r}"
        )
    if got_step is None or int(got_step) != spec.expected_global_step:
        raise RuntimeError(
            f"operator-oracle checkpoint step {got_step!r} != expected "
            f"{spec.expected_global_step}"
        )
    state_meta = base.modules.meta.get("state_dict") or {}
    q_meta = dict(state_meta.get("q_action") or {})
    if int(q_meta.get("tensors_loaded", 0)) <= 0 or int(q_meta.get("unexpected", -1)) != 0:
        raise RuntimeError(
            "q_action was not completely authenticated and loaded from the checkpoint"
        )
    bank_meta = dict(state_meta.get("bank") or {})
    if int(bank_meta.get("tensors_present", 0)) <= 0:
        raise RuntimeError(
            "operator-bank state is absent; this is not the promoted bank-stage checkpoint"
        )
    base.modules.meta["operator_oracle"] = spec.to_dict()
    return OperatorOraclePolicy(
        base.modules,
        arm=spec.arm,
        tau=spec.tau,
        n_candidates=spec.n_candidates,
        op_stats=op_stats,
        gripper_dwell=gripper_dwell,
        decoder_samples=decoder_samples,
        duration_normalize_segments=duration_normalize_segments,
    )


def _episode_key(row: Mapping[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(row["suite"]), int(row["task_id"]), int(row["episode"]), int(row["seed"]),
    )


def _pct(n: int | float, d: int | float) -> float:
    return round(100.0 * float(n) / float(d), 4) if d else 0.0


def _suite_summary(
    keys: list[tuple[str, int, int, int]],
    by_arm: dict[int, dict[tuple[str, int, int, int], Mapping[str, Any]]],
) -> dict[str, Any]:
    direct = [bool(by_arm[0][k]["success"]) for k in keys]
    oracle = [any(bool(by_arm[a][k]["success"]) for a in range(ORACLE_CANDIDATES))
              for k in keys]
    sampled = [bool(by_arm[a][k]["success"])
               for k in keys for a in range(1, ORACLE_CANDIDATES)]
    per_seed: dict[str, float] = {}
    for seed in sorted({k[3] for k in keys}):
        idx = [i for i, k in enumerate(keys) if k[3] == seed]
        per_seed[str(seed)] = _pct(sum(oracle[i] for i in idx), len(idx))
    return {
        "n_conditions": len(keys),
        "direct_argmax_sr": _pct(sum(direct), len(direct)),
        "sample_arm_marginal_sr": _pct(sum(sampled), len(sampled)),
        "oracle_at_16_sr": _pct(sum(oracle), len(oracle)),
        "oracle_new_only": sum(o and not d for o, d in zip(oracle, direct)),
        "direct_successes": sum(direct),
        "oracle_successes": sum(oracle),
        "per_seed_oracle_at_16_sr": per_seed,
    }


def aggregate_oracle_results(
    arm_results: Mapping[int, Mapping[str, Any]],
    protocol: EvalProtocol,
) -> dict[str, Any]:
    """Strict terminal-success aggregation. Missing/mixed arms raise."""
    want_arms = set(range(ORACLE_CANDIDATES))
    if set(arm_results) != want_arms:
        raise ValueError(
            f"operator oracle needs arms {sorted(want_arms)}, got {sorted(arm_results)}"
        )

    from loom.eval.runner import iter_work              # noqa: PLC0415

    expected_items = iter_work(protocol)
    expected = {
        (x.suite, x.task_id, x.episode, x.seed): x for x in expected_items
    }
    by_arm: dict[int, dict[tuple[str, int, int, int], Mapping[str, Any]]] = {}
    n_errors = 0
    gate_replans = gate_rejected = gate_exhausted = 0
    shared_oracle_identity: dict[str, Any] | None = None
    for arm in range(ORACLE_CANDIDATES):
        result = arm_results[arm]
        got_protocol = EvalProtocol.from_dict(dict(result.get("protocol") or {}))
        if got_protocol != protocol:
            raise ValueError(f"arm {arm} protocol mismatch")
        meta = dict(result.get("meta") or {})
        policy_meta = dict(meta.get("policy") or {})
        oracle_meta = dict(policy_meta.get("operator_oracle") or {})
        try:
            arm_spec = OracleArmSpec.from_mapping(oracle_meta)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"arm {arm} has invalid policy provenance: {exc}") from exc
        if arm_spec.arm != arm:
            raise ValueError(f"arm {arm} lacks matching policy provenance")
        if policy_meta.get("is_stub") is not False:
            raise ValueError(f"arm {arm} did not run a real checkpoint policy")
        shared = arm_spec.to_dict()
        shared.pop("arm")
        if shared_oracle_identity is None:
            shared_oracle_identity = shared
        elif shared != shared_oracle_identity:
            raise ValueError(f"arm {arm} has mixed operator-oracle provenance")
        if str(policy_meta.get("ckpt_config_hash") or "") != arm_spec.expected_config_hash:
            raise ValueError(f"arm {arm} loaded config hash does not match its manifest")
        if int(policy_meta.get("ckpt_global_step", -1)) != arm_spec.expected_global_step:
            raise ValueError(f"arm {arm} loaded step does not match its manifest")
        state_meta = dict(policy_meta.get("state_dict") or {})
        q_meta = dict(state_meta.get("q_action") or {})
        bank_meta = dict(state_meta.get("bank") or {})
        if int(q_meta.get("tensors_loaded", 0)) <= 0 or int(q_meta.get("unexpected", -1)) != 0:
            raise ValueError(f"arm {arm} lacks a completely loaded q_action")
        if int(bank_meta.get("tensors_present", 0)) <= 0:
            raise ValueError(f"arm {arm} lacks promoted operator-bank provenance")
        rows: dict[tuple[str, int, int, int], Mapping[str, Any]] = {}
        for row in result.get("episodes", []):
            key = _episode_key(row)
            if key in rows:
                raise ValueError(f"arm {arm} has duplicate condition {key}")
            rows[key] = row
            item = expected.get(key)
            if item is None:
                raise ValueError(f"arm {arm} has out-of-protocol condition {key}")
            if int(row.get("env_seed", -1)) != item.env_seed:
                raise ValueError(f"arm {arm} env seed mismatch for {key}")
            extra = dict(row.get("extra") or {})
            if int(extra.get("policy_seed", -1)) != item.policy_seed:
                raise ValueError(f"arm {arm} policy seed mismatch for {key}")
            required = {
                "oracle_version", "oracle_arm", "oracle_mode",
                "oracle_n_candidates", "oracle_tau",
                "oracle_proposal_seed_scheme", "oracle_common_decoder_noise",
                "oracle_n_replans", "oracle_n_rejected",
                "oracle_n_gate_exhausted", "oracle_requested_residual_mean",
                "oracle_requested_residual_max", "oracle_executed_residual_mean",
                "oracle_executed_residual_max",
            }
            absent = sorted(required - set(extra))
            if absent:
                raise ValueError(
                    f"arm {arm} condition {key} lacks oracle evidence: {absent}"
                )
            expected_mode = "direct_argmax" if arm == 0 else "proposal_sample"
            if (
                int(extra["oracle_version"]) != ORACLE_VERSION
                or int(extra["oracle_arm"]) != arm
                or str(extra["oracle_mode"]) != expected_mode
                or int(extra["oracle_n_candidates"]) != ORACLE_CANDIDATES
                or float(extra["oracle_tau"]) != float(REALIZABILITY_TAU)
                or str(extra["oracle_proposal_seed_scheme"]) != PROPOSAL_SEED_SCHEME
                or extra["oracle_common_decoder_noise"] is not True
            ):
                raise ValueError(f"arm {arm} condition {key} has invalid oracle identity")
            replans = int(extra["oracle_n_replans"])
            rejected = int(extra["oracle_n_rejected"])
            exhausted = int(extra["oracle_n_gate_exhausted"])
            if replans < 0 or not (0 <= rejected <= replans) or not (0 <= exhausted <= replans):
                raise ValueError(f"arm {arm} condition {key} has invalid gate counts")
            if arm == 0 and rejected != 0:
                raise ValueError(f"direct arm condition {key} claims sample rejection")
            if row.get("n_replans") is None or int(row["n_replans"]) != replans:
                raise ValueError(f"arm {arm} condition {key} replan count mismatch")
            residual_names = (
                "oracle_requested_residual_mean", "oracle_requested_residual_max",
                "oracle_executed_residual_mean", "oracle_executed_residual_max",
            )
            if replans:
                for name in residual_names:
                    value = extra[name]
                    if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
                        raise ValueError(
                            f"arm {arm} condition {key} has invalid {name}"
                        )
            elif any(extra[name] is not None for name in residual_names):
                raise ValueError(
                    f"arm {arm} condition {key} has residuals without replans"
                )
            n_errors += int(row.get("error") is not None)
            if arm > 0:
                gate_replans += replans
                gate_rejected += rejected
                gate_exhausted += exhausted
        missing = set(expected) - set(rows)
        extra_keys = set(rows) - set(expected)
        if missing or extra_keys:
            raise ValueError(
                f"arm {arm} incomplete: missing={len(missing)} extra={len(extra_keys)}"
            )
        by_arm[arm] = rows

    suite_rows: dict[str, dict[str, Any]] = {}
    for suite in protocol.suites:
        keys = sorted(k for k in expected if k[0] == suite)
        suite_rows[suite] = _suite_summary(keys, by_arm)
    all_keys = sorted(expected)
    overall = _suite_summary(all_keys, by_arm)
    avg = sum(suite_rows[s]["oracle_at_16_sr"] for s in protocol.suites) / len(protocol.suites)
    full_protocol = (
        protocol.bench == "libero"
        and protocol.episodes_per_task == 10
        and protocol.n_tasks == 10
        and protocol.suites == tuple(DEFAULT_LIBERO_SUITES)
        and protocol.seeds == (0, 1, 2)
        and protocol.max_steps == 512
    )
    every_suite_90 = all(
        suite_rows[s]["oracle_at_16_sr"] >= 90.0 for s in protocol.suites
    )
    return {
        "version": ORACLE_VERSION,
        "candidate_contract": "arm0=direct_argmax; arms1..15=proposal_sample",
        "common_decoder_noise": True,
        "proposal_seed_scheme": PROPOSAL_SEED_SCHEME,
        "realizability": {
            "tau": REALIZABILITY_TAU,
            "sampled_replans": gate_replans,
            "rejected_to_same_noise_argmax": gate_rejected,
            "gate_exhausted": gate_exhausted,
            "rejection_rate": _pct(gate_rejected, gate_replans),
            "exhaustion_rate": _pct(gate_exhausted, gate_replans),
        },
        "protocol": protocol.to_dict(),
        "n_arm_episodes": ORACLE_CANDIDATES * len(expected),
        "n_conditions": len(expected),
        "n_errors": n_errors,
        "complete": True,
        "authoritative_protocol": full_protocol,
        "per_suite": suite_rows,
        "overall": {**overall, "suite_mean_oracle_at_16_sr": round(avg, 4)},
        "passes_90_every_suite": bool(full_protocol and n_errors == 0 and every_suite_90),
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_digest(root: Path) -> str:
    h = hashlib.sha256()
    for rel in _SOURCE_FILES:
        p = root / rel
        if not p.is_file():
            raise FileNotFoundError(f"oracle provenance source is missing: {p}")
        h.update(rel.encode() + b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _protocol_from_args(args: argparse.Namespace) -> EvalProtocol:
    if args.dry_run:
        return EvalProtocol(
            bench="libero", episodes_per_task=1, n_tasks=1,
            suites=("libero_spatial",), seeds=(0,), max_steps=2,
            notes="operator-oracle fake-env dry run; not a score",
        )
    suites = tuple(x.strip() for x in args.suites.split(",") if x.strip())
    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    return EvalProtocol(
        bench="libero",
        episodes_per_task=args.episodes_per_task,
        n_tasks=args.n_tasks,
        suites=suites,
        seeds=seeds,
        max_steps=args.max_steps,
    )


def _dry_policy(arm: int, source_digest: str) -> OperatorOraclePolicy:
    """Small injected modules for the CLI dry run; never a scored policy."""
    import stubs                                           # noqa: PLC0415

    class DryProposal(nn.Module):
        def _one(self, idx: Tensor, dtype: torch.dtype) -> Tensor:
            c = torch.zeros(*idx.shape[:-1], M, device=idx.device, dtype=dtype)
            return c.scatter(-1, idx, 1.0 / TOPK)

        def argmax(self, z: Tensor, lang: Tensor) -> Tensor:
            idx = torch.arange(TOPK, device=z.device).expand(z.shape[0], TOPK)
            return self._one(idx, z.dtype)

        def sample(self, z: Tensor, lang: Tensor, n: int, *, generator=None) -> Tensor:
            noise = torch.rand(z.shape[0], n, M, device=z.device, generator=generator)
            return self._one(noise.topk(TOPK, dim=-1).indices, z.dtype)

    class DryDecoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.last_c: Tensor | None = None

        def forward(self, proprio: Tensor, c: Tensor, *, generator=None) -> Tensor:
            self.last_c = c
            return 0.01 * torch.randn(
                c.shape[0], H_OP, 7, device=c.device, dtype=c.dtype,
                generator=generator,
            )

    class DryQAction(nn.Module):
        def __init__(self, decoder: DryDecoder) -> None:
            super().__init__()
            self.decoder = decoder

        def forward(self, a: Tensor, z: Tensor) -> Tensor:
            if self.decoder.last_c is None:
                raise RuntimeError("decoder was not called")
            return self.decoder.last_c

    decoder = DryDecoder()
    spec = OracleArmSpec(
        arm=arm,
        expected_config_hash="dry-run",
        expected_global_step=0,
        checkpoint_sha256="0" * 64,
        source_digest=source_digest,
    )
    modules = PolicyModules(
        estimator=stubs.StubEstimator(),
        proposal=DryProposal(),
        decoder=decoder,
        q_action=DryQAction(decoder),
        featurize=zeros_featurizer(EMBODIMENTS["libero_franka"]),
        embodiment="libero_franka",
        device="cpu",
        is_stub=True,
        meta={
            "dry_run": True,
            "ckpt_config_hash": "dry-run",
            "ckpt_global_step": 0,
            "state_dict": {
                "q_action": {"tensors_loaded": 1, "unexpected": 0},
                "bank": {"tensors_present": 1, "loaded": False},
            },
            "operator_oracle": spec.to_dict(),
        },
    )
    return OperatorOraclePolicy(modules, arm=arm)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python scripts/operator_oracle.py",
        description=(
            "Measure the fixed common-noise 16-arm LIBERO operator oracle. "
            "Runs no Phi and no bank-based search."
        ),
    )
    p.add_argument("--ckpt", default=None, help="promoted consolidated bank checkpoint")
    p.add_argument("--out-dir", required=True, help="manifest, arm JSONs, summary")
    p.add_argument("--expected-config-hash", default=None)
    p.add_argument("--expected-global-step", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--episodes-per-task", type=int, default=10)
    p.add_argument("--n-tasks", type=int, default=10)
    p.add_argument("--suites", default=",".join(DEFAULT_LIBERO_SUITES))
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--max-steps", type=int, default=512)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="16-arm fake-env plumbing check; never a score")
    p.add_argument("--quiet", action="store_true")
    return p


def _validate_arm_result(
    result: Mapping[str, Any], arm: int, *, dry: bool,
    identity: Mapping[str, Any] | None = None,
) -> None:
    meta = dict(result.get("meta") or {})
    policy = dict(meta.get("policy") or {})
    oracle = dict(policy.get("operator_oracle") or {})
    if int(oracle.get("arm", -1)) != arm:
        raise RuntimeError(f"arm {arm} result has mismatched policy provenance: {oracle}")
    if not dry and policy.get("is_stub") is not False:
        raise RuntimeError(f"arm {arm} used stubs; refusing an oracle claim")
    if identity is not None and not dry:
        expected_ckpt = str(identity["checkpoint"]["path"])
        if str(meta.get("ckpt")) != expected_ckpt:
            raise RuntimeError(f"arm {arm} checkpoint provenance mismatch")
        want_oracle = OracleArmSpec(
            arm=arm,
            expected_config_hash=str(identity["expected_config_hash"]),
            expected_global_step=int(identity["expected_global_step"]),
            checkpoint_sha256=str(identity["checkpoint"]["sha256"]),
            source_digest=str(identity["source_digest"]),
        ).to_dict()
        eval_identity = dict(meta.get("eval_identity") or {})
        want_eval_identity = {
            "version": 1,
            "checkpoint": expected_ckpt,
            "backend": {"requested": "libero", "resolved": "libero"},
            "policy_kw": {
                "allow_stub": False,
                "embodiment": "libero_franka",
                "operator_oracle": want_oracle,
            },
            "policy_source": "checkpoint_factory",
            "policy_seed_scheme": "sha256(work-item)-v1",
        }
        if eval_identity != want_eval_identity:
            raise RuntimeError(f"arm {arm} behavior identity mismatch")
        if (
            meta.get("bench") != "libero"
            or meta.get("backend") != "libero"
            or meta.get("env_available") is not True
            or meta.get("libero_available") is not True
            or meta.get("policy_seed_scheme") != "sha256(work-item)-v1"
        ):
            raise RuntimeError(f"arm {arm} lacks real-LIBERO backend provenance")
        if policy.get("operator_oracle") != want_oracle:
            raise RuntimeError(f"arm {arm} loaded oracle provenance mismatch")
        if str(policy.get("ckpt_config_hash") or "") != want_oracle["expected_config_hash"]:
            raise RuntimeError(f"arm {arm} loaded config hash mismatch")
        if int(policy.get("ckpt_global_step", -1)) != want_oracle["expected_global_step"]:
            raise RuntimeError(f"arm {arm} loaded checkpoint step mismatch")
        state_meta = dict(policy.get("state_dict") or {})
        q_meta = dict(state_meta.get("q_action") or {})
        bank_meta = dict(state_meta.get("bank") or {})
        if int(q_meta.get("tensors_loaded", 0)) <= 0 or int(q_meta.get("unexpected", -1)) != 0:
            raise RuntimeError(f"arm {arm} lacks completely loaded q_action provenance")
        if int(bank_meta.get("tensors_present", 0)) <= 0:
            raise RuntimeError(f"arm {arm} lacks bank-stage checkpoint provenance")
    summary = dict(result.get("summary") or {})
    if not summary.get("complete") or int(summary.get("n_errors", -1)) != 0:
        raise RuntimeError(
            f"arm {arm} is not a clean complete run: "
            f"complete={summary.get('complete')} errors={summary.get('n_errors')}"
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protocol = _protocol_from_args(args)
    root = Path(__file__).resolve().parents[2]
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    source_digest = _source_digest(root)

    if args.dry_run:
        checkpoint = {"kind": "dry_run", "path": None}
        expected_hash, expected_step = "dry-run", 0
    else:
        if not args.ckpt or not args.expected_config_hash or args.expected_global_step is None:
            raise SystemExit(
                "production oracle requires --ckpt, --expected-config-hash, and "
                "--expected-global-step"
            )
        ckpt = Path(args.ckpt).expanduser().resolve()
        if not ckpt.is_file():
            raise SystemExit(f"checkpoint does not exist: {ckpt}")
        st = ckpt.stat()
        checkpoint = {
            "kind": "consolidated",
            "path": str(ckpt),
            "size": st.st_size,
            "mtime_ns": st.st_mtime_ns,
            "sha256": _sha256_file(ckpt),
        }
        expected_hash = str(args.expected_config_hash)
        expected_step = int(args.expected_global_step)
        from loom.eval.libero import libero_available      # noqa: PLC0415

        if not libero_available():
            raise SystemExit(
                "real LIBERO is unavailable in this interpreter; use the pinned "
                "LOOM_LIBERO_PYTHON environment (or --dry-run for plumbing only)"
            )

    identity = {
        "version": ORACLE_VERSION,
        "checkpoint": checkpoint,
        "expected_config_hash": expected_hash,
        "expected_global_step": expected_step,
        "source_digest": source_digest,
        "protocol": protocol.to_dict(),
        "n_candidates": ORACLE_CANDIDATES,
        "tau": REALIZABILITY_TAU,
        "policy_seed_scheme": "sha256(work-item)-v1",
        "proposal_seed_scheme": PROPOSAL_SEED_SCHEME,
        "backend": "fake" if args.dry_run else "libero",
    }
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists() and not args.no_resume:
        old = json.loads(manifest_path.read_text())
        if old.get("identity") != identity:
            raise RuntimeError(
                "operator-oracle manifest identity changed; use a new --out-dir "
                "or explicitly pass --no-resume"
            )
    else:
        _atomic_json(manifest_path, {
            "identity": identity,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

    from loom.eval.runner import run_eval                 # noqa: PLC0415

    arm_results: dict[int, Mapping[str, Any]] = {}
    for arm in range(ORACLE_CANDIDATES):
        # Fail if source or checkpoint identity changes between arms of a long
        # run.  Full checkpoint SHA is paid once; stat catches in-run mutation.
        if _source_digest(root) != source_digest:
            raise RuntimeError("oracle source changed after the manifest was written")
        if not args.dry_run:
            now = Path(checkpoint["path"]).stat()
            if now.st_size != checkpoint["size"] or now.st_mtime_ns != checkpoint["mtime_ns"]:
                raise RuntimeError("oracle checkpoint changed after hashing")
        out = out_dir / f"arm_{arm:02d}.json"
        if not args.quiet:
            print(f"[operator-oracle] arm {arm:02d}/{ORACLE_CANDIDATES - 1:02d}",
                  file=sys.stderr, flush=True)
        if out.exists() and not args.no_resume:
            result = json.loads(out.read_text())
            try:
                _validate_arm_result(
                    result, arm, dry=args.dry_run, identity=identity,
                )
                arm_results[arm] = result
                continue
            except RuntimeError:
                # A clean incomplete file is resumable by runner; a complete
                # identity mismatch is not.  Completeness is the discriminator.
                old_summary = dict(result.get("summary") or {})
                if old_summary.get("complete"):
                    raise
        if args.dry_run:
            result = run_eval(
                protocol,
                bench="libero",
                ckpt=None,
                out=out,
                workers=1,
                resume=not args.no_resume,
                backend="fake",
                policy=_dry_policy(arm, source_digest),
            )
        else:
            spec = OracleArmSpec(
                arm=arm,
                expected_config_hash=expected_hash,
                expected_global_step=expected_step,
                checkpoint_sha256=str(checkpoint["sha256"]),
                source_digest=source_digest,
            )
            result = run_eval(
                protocol,
                bench="libero",
                ckpt=checkpoint["path"],
                out=out,
                workers=args.workers,
                resume=not args.no_resume,
                backend="libero",
                policy_kw={
                    "allow_stub": False,
                    "operator_oracle": spec.to_dict(),
                },
            )
        _validate_arm_result(result, arm, dry=args.dry_run, identity=identity)
        arm_results[arm] = result

    # Dry policies are explicitly stub-backed, so strict production aggregation
    # would (correctly) reject them.  Relabel only the in-memory provenance for
    # the plumbing check; the arm files remain visibly ``is_stub=true``.
    aggregate_input: dict[int, Mapping[str, Any]] = {}
    for arm, result in arm_results.items():
        if not args.dry_run:
            aggregate_input[arm] = result
            continue
        clone = json.loads(json.dumps(result))
        clone["meta"]["policy"]["is_stub"] = False
        aggregate_input[arm] = clone
    summary = aggregate_oracle_results(aggregate_input, protocol)
    summary["dry_run"] = bool(args.dry_run)
    summary["manifest"] = str(manifest_path)
    _atomic_json(out_dir / "summary.json", summary)
    if not args.quiet:
        print(json.dumps(summary, indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
