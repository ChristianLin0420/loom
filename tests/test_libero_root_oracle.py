"""Isolated synthetic tests for the common-noise root operator oracle."""

from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest
import torch
from torch import nn

import contracts as C
from loom.eval.policy import PolicyModules
from loom.eval.runner import iter_work
from scripts import libero_root_oracle as oracle


class TinyEstimator(nn.Module):
    def forward(self, feats, z_prev=None):  # noqa: ARG002
        return torch.zeros(1, C.K, C.D)


class TinyProposal:
    @staticmethod
    def _one(index: int) -> torch.Tensor:
        out = torch.zeros(1, C.M)
        out[0, int(index) % C.M] = 1.0
        return out

    def argmax(self, z, lang):  # noqa: ARG002
        return self._one(0)

    def sample(self, z, lang, n, generator=None):  # noqa: ARG002
        idx = torch.randint(1, C.M, (n,), generator=generator)
        return torch.stack([self._one(int(i))[0] for i in idx], dim=0).unsqueeze(0)


class TinyDecoder(nn.Module):
    def forward(self, proprio, c, *, generator=None):
        assert generator is not None
        out = 0.01 * torch.randn(
            proprio.shape[0], C.H_OP, 7,
            generator=generator, device=proprio.device, dtype=proprio.dtype,
        )
        out[:, :, 0] += c.argmax(-1).to(out.dtype)[:, None] / 256.0
        out[:, :, 6] = -1.0
        return out


class TinyQAction(nn.Module):
    def forward(self, action, z):  # noqa: ARG002
        out = torch.zeros(action.shape[0], C.M, device=action.device, dtype=action.dtype)
        out[:, 0] = 1.0
        return out


def tiny_bundle() -> oracle.OracleBundle:
    def featurize(obs, instruction):  # noqa: ARG001
        state = torch.from_numpy(np.asarray(obs["state"], dtype=np.float32)).unsqueeze(0)
        return {
            "views": torch.zeros(1, 2, 1, 1),
            "proprio": state,
            "lang": torch.zeros(1, 1, 1),
        }

    modules = PolicyModules(
        estimator=TinyEstimator(),
        proposal=TinyProposal(),
        decoder=TinyDecoder(),
        featurize=featurize,
        embodiment="libero_franka",
        device="cpu",
        is_stub=False,
        meta={"synthetic": True},
    )
    return oracle.OracleBundle(
        modules=modules,
        q_action=TinyQAction(),
        provenance={"is_stub": False, "synthetic": True},
    )


class ShortEnv:
    """Identical reset, settles for 15 steps, succeeds on policy step seven."""

    def __init__(self):
        self.steps = 0

    @staticmethod
    def _obs():
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        return {
            "agentview_image": image,
            "robot0_eye_in_hand_image": image.copy(),
            "robot0_eef_pos": np.zeros(3, dtype=np.float32),
            "robot0_eef_quat": np.array([0, 0, 0, 1], dtype=np.float32),
            "robot0_gripper_qpos": np.zeros(1, dtype=np.float32),
        }

    def reset(self):
        self.steps = 0
        return self._obs()

    def step(self, action):
        assert np.asarray(action).shape == (7,)
        self.steps += 1
        success = self.steps >= oracle.libero.SETTLE_STEPS + 7
        return self._obs(), float(success), success, {"success": success}

    def check_success(self):
        return self.steps >= oracle.libero.SETTLE_STEPS + 7

    def close(self):
        pass


def _baseline_outcomes() -> dict[str, bool]:
    remaining = dict(oracle.EXPECTED_ARM0_BY_SUITE)
    outcomes = {}
    for item in iter_work(oracle.official_protocol()):
        key = oracle.work_key(item)
        success = remaining[item.suite] > 0
        outcomes[key] = success
        remaining[item.suite] -= int(success)
    assert all(value == 0 for value in remaining.values())
    return outcomes


def _synthetic_groups(*, masked_successes: int = 400):
    baseline = _baseline_outcomes()
    groups = []
    for ordinal, item in enumerate(iter_work(oracle.official_protocol())):
        key = oracle.work_key(item)
        masked = ordinal < masked_successes
        reset_sha = oracle._hash_value(f"reset:{key}")
        rng_before = oracle._hash_value(f"rng-before:{key}")
        rng_after = oracle._hash_value(f"rng-after:{key}")
        arms = []
        for arm_id in range(oracle.N_ARMS):
            terminal = baseline[key] if arm_id == 0 else (masked and arm_id == 1)
            kind = "proposal.argmax" if arm_id == 0 else "proposal.sample"
            sample_seed = None if arm_id == 0 else oracle.root_seed(item, arm_id)
            coeff = torch.zeros(1, C.M, dtype=torch.float32)
            coeff[0, arm_id] = 1.0
            segment_sha = oracle._hash_value(f"segment:{key}:{arm_id}")
            prefix_sha = oracle._hash_value(f"prefix:{key}:{arm_id}")
            arms.append({
                "arm_id": arm_id,
                "root_kind": kind,
                "root_sample_seed": sample_seed,
                "error": None,
                "residual_l2": 0.1,
                "eligible": True,
                "terminal_success": terminal,
                "steps": 10,
                "hit_step_cap": False,
                "n_replans": 2,
                "wall_s": 0.01,
                "env_seed": item.env_seed,
                "policy_seed": item.policy_seed,
                "root": {
                    "arm_id": arm_id,
                    "kind": kind,
                    "sample_seed": sample_seed,
                    "sha256": oracle._hash_value(coeff),
                    "support": [arm_id],
                    "weights": [1.0],
                    "sum": 1.0,
                },
                "reset_input_sha256": reset_sha,
                "reset_input_matches": True,
                "execution": {
                    "residual_l2": 0.1,
                    "decoder_rng_before_sha256": rng_before,
                    "decoder_rng_after_sha256": rng_after,
                    "residual_segment_sha256": segment_sha,
                    "decoded_segment_sha256": segment_sha,
                    "q_action_coeff_sha256": oracle._hash_value(
                        f"q-action:{key}:{arm_id}"
                    ),
                    "planned_root_segment_sha256": segment_sha,
                    "post_gripper_root_segment_sha256": segment_sha,
                    "residual_uses_planned_segment": True,
                    "root_gripper_path_unchanged": True,
                    "executed_root_prefix_sha256": prefix_sha,
                    "expected_root_prefix_sha256": prefix_sha,
                    "executed_root_prefix_matches": True,
                    "root_env_steps_expected": 5,
                    "root_env_steps_executed": 5,
                    "root_env_segment_complete": True,
                    "proposal_calls": 2,
                    "n_forced_roots": 1,
                    "n_direct_continuations": 1,
                },
            })
        successful_eligible = [
            arm["arm_id"] for arm in arms
            if arm["eligible"] and arm["terminal_success"]
        ]
        groups.append({
            "group_id": key,
            "work_item": item.to_dict(),
            "arms": arms,
            "parity": {
                "passed": True,
                "checks": {name: True for name in oracle.PARITY_CHECK_KEYS},
                "reset_input_sha256": reset_sha,
                "decoder_rng_before_sha256": rng_before,
                "decoder_rng_after_sha256": rng_after,
            },
            "arm0_terminal_success": arms[0]["terminal_success"],
            "masked_oracle_success": bool(successful_eligible),
            "unmasked_oracle_success": any(
                arm["terminal_success"] for arm in arms
            ),
            "successful_eligible_arm_ids": successful_eligible,
        })
    return groups, baseline


def _provenance(*, identity_centered: bool = False):
    source_entries = [
        {"path": path, "sha256": "e" * 64}
        for path in oracle.BEHAVIOR_SOURCE_FILES
    ]
    candidate_recipe = {
        "kind": oracle.CANDIDATE_RECIPE_BANK_ONLY,
        "train_modules": ["bank"],
        "detach_coeff": True,
        "detach_coeff_explicit": False,
        "reset_state_modules": ["bank"],
        "bank_lr_scale": 0.1,
        "q_action_lr_scale": 0.0,
        "dyn_neg_weight": 4.0,
    }
    method_variant = "bank_only"
    if identity_centered:
        candidate_recipe = oracle.authenticate_candidate_recipe(
            _candidate_recipe_config(joint=True, identity_centered=True)
        )
        method_variant = "joint_q_action_bank_identity_centered"
    return {
        "checkpoint": {
            "sha256": "c" * 64,
            "config_hash": "cfg",
            "global_step": 57_666,
            "candidate_recipe": candidate_recipe,
        },
        "bank_gate": {
            "sha256": "d" * 64,
            "manifest_digest": "sha256:m",
            "passed": True,
            "status": "PASS",
            "candidate_sha256": "c" * 64,
            "candidate_config_hash": "cfg",
            "method_variant": method_variant,
        },
        "baseline": {
            "sha256": "b" * 64,
            "checkpoint_global_step": 49_666,
            "n_success": 149,
            "n_errors": 0,
        },
        "source": {
            "script_sha256": "e" * 64,
            "behavior_source_digest_scheme": oracle.BEHAVIOR_SOURCE_DIGEST_SCHEME,
            "behavior_source_digest": oracle._behavior_digest_from_entries(
                source_entries
            ),
            "behavior_source_files": source_entries,
        },
        "runtime": {"backend": "libero", "env_available": True},
        "policy_workers": [{"policy": {
            "is_stub": False,
            "oracle_modules_frozen": True,
            "q_action": {"strict": True, "frozen": True},
            "decoder_samples": 1,
            "gripper_dwell": 1,
            "duration_normalize_segments": False,
            "embodiment": "libero_franka",
            "ckpt_config_hash": "cfg",
            "ckpt_global_step": 57_666,
        }}],
    }


def _candidate_recipe_config(
    *, joint: bool = False, identity_centered: bool = False,
) -> dict:
    assert not identity_centered or joint
    cfg = {
        "run": {"name": "r0a_bank_ca_n4", "steps": 80_000},
        "data": {
            "source": "libero",
            "embodiments": ["libero_franka"],
            "sampling": "uniform_window",
            "trajectory_split": "train",
            "holdout_demo_keys": ["demo_49"],
            "recurrent_burn_in": 4,
        },
        "optim": {
            "update_ema": False,
            "reset_state_modules": ["bank"],
            "lr_scales": {
                "estimator": 0.0,
                "bank": 0.1,
                "q_delta": 0.0,
                "q_action": 0.0,
                "ema": 0.0,
                "proposal": 0.0,
                "decoder": 0.0,
                "potential": 0.0,
            },
        },
        "losses": {
            "dyn": {
                "enabled": True,
                "weight": 1.0,
                "coeff_source": "q_action",
                "negatives": "within_trajectory",
                "min_gap": 2,
                "neg_weight": 4.0,
                "neg_margin": 0.1,
                "cosine": "per_slot",
            },
            "act": {"enabled": False, "weight": 0.0},
            "proposal": {"enabled": False, "weight": 0.0},
            "balance": {"enabled": False, "weight": 0.0},
            "potential": {"enabled": False, "weight": 0.0},
            "grpo": {"enabled": False, "weight": 0.0},
        },
        "train_modules": ["bank"],
    }
    if joint:
        cfg["run"]["name"] = "r0a_bank_ca_qa"
        cfg["train_modules"] = ["bank", "q_action"]
        cfg["optim"]["reset_state_modules"] = ["bank", "q_action"]
        cfg["optim"]["lr_scales"]["q_action"] = 1.0
        cfg["losses"]["dyn"]["detach_coeff"] = False
        cfg["losses"]["act"] = {
            "enabled": True,
            "weight": 1.0,
            "align_to": "q_a",
            "decode_from": "q_action",
        }
    if identity_centered:
        cfg["run"]["name"] = "r0a_bank_ca_qa_omega0"
        cfg["optim"]["transition_parameter_reset"] = copy.deepcopy(
            oracle.IDENTITY_CENTERED_RESET
        )
    return cfg


def test_official_seed0_work_items_and_stable_arm_seeds():
    protocol = oracle.official_protocol()
    items = iter_work(protocol)
    assert len(items) == 400
    assert {item.seed for item in items} == {0}
    assert len({item.key() for item in items}) == 400
    one = [oracle.root_seed(items[0], arm) for arm in range(1, 16)]
    two = [oracle.root_seed(items[0], arm) for arm in range(1, 16)]
    assert one == two
    assert len(set(one)) == 15
    assert oracle.root_seed(items[1], 1) != one[0]


def test_behavior_source_closure_is_complete_and_self_consistent():
    required = {
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
    }
    assert set(oracle.BEHAVIOR_SOURCE_FILES) == required
    source = oracle.source_provenance()
    assert source["behavior_source_digest_scheme"] == (
        oracle.BEHAVIOR_SOURCE_DIGEST_SCHEME
    )
    assert [row["path"] for row in source["behavior_source_files"]] == sorted(required)
    assert source["behavior_source_digest"] == oracle._behavior_digest_from_entries(
        source["behavior_source_files"]
    )
    script = next(
        row for row in source["behavior_source_files"]
        if row["path"] == "scripts/libero_root_oracle.py"
    )
    assert script["sha256"] == source["script_sha256"]
    oracle.assert_behavior_source_digest(source["behavior_source_digest"])


@pytest.mark.parametrize(
    ("mutation", "failure"),
    [("content", "source changed"), ("missing", "source is missing")],
)
def test_behavior_source_digest_fails_closed_on_mutation_or_missing(
    tmp_path, mutation, failure,
):
    files = ("a.py", "nested/b.py")
    (tmp_path / "nested").mkdir()
    (tmp_path / "a.py").write_text("a = 1\n")
    (tmp_path / "nested/b.py").write_text("b = 1\n")
    source = oracle.behavior_source_provenance(tmp_path, reversed(files))
    assert [row["path"] for row in source["behavior_source_files"]] == list(files)
    oracle.assert_behavior_source_digest(
        source["behavior_source_digest"], root=tmp_path, files=files,
    )
    if mutation == "content":
        (tmp_path / "nested/b.py").write_text("b = 2\n")
    else:
        (tmp_path / "nested/b.py").unlink()
    with pytest.raises(oracle.OracleError, match=failure):
        oracle.assert_behavior_source_digest(
            source["behavior_source_digest"], root=tmp_path, files=files,
        )


def test_candidate_recipe_accepts_exact_bank_only_and_joint_stages():
    bank = _candidate_recipe_config()
    got_bank = oracle.authenticate_candidate_recipe(bank)
    assert got_bank == {
        "kind": oracle.CANDIDATE_RECIPE_BANK_ONLY,
        "train_modules": ["bank"],
        "detach_coeff": True,
        "detach_coeff_explicit": False,
        "reset_state_modules": ["bank"],
        "bank_lr_scale": 0.1,
        "q_action_lr_scale": 0.0,
        "dyn_neg_weight": 4.0,
    }
    # The historical base bank stage used neg_weight=1 and omitted the flag;
    # an explicit true is the same detached method after the new knob lands.
    base_bank = copy.deepcopy(bank)
    base_bank["losses"]["dyn"]["neg_weight"] = 1.0
    base_bank["losses"]["dyn"]["detach_coeff"] = True
    assert oracle.authenticate_candidate_recipe(base_bank)["kind"] == (
        oracle.CANDIDATE_RECIPE_BANK_ONLY
    )

    joint = _candidate_recipe_config(joint=True)
    got_joint = oracle.authenticate_candidate_recipe(joint)
    assert got_joint == {
        "kind": oracle.CANDIDATE_RECIPE_JOINT_Q_ACTION_BANK,
        "train_modules": ["bank", "q_action"],
        "detach_coeff": False,
        "detach_coeff_explicit": True,
        "reset_state_modules": ["bank", "q_action"],
        "bank_lr_scale": 0.1,
        "q_action_lr_scale": 1.0,
        "dyn_neg_weight": 4.0,
        "action_anchor": {
            "enabled": True,
            "weight": 1.0,
            "align_to": "q_a",
            "decode_from": "q_action",
        },
    }

    identity = oracle.authenticate_candidate_recipe(
        _candidate_recipe_config(joint=True, identity_centered=True)
    )
    assert identity["kind"] == (
        oracle.CANDIDATE_RECIPE_JOINT_Q_ACTION_BANK_IDENTITY_CENTERED
    )
    assert identity["transition_parameter_reset"] == oracle.IDENTITY_CENTERED_RESET
    assert identity["action_anchor"] == got_joint["action_anchor"]


@pytest.mark.parametrize(
    ("joint", "path", "value", "failure"),
    [
        (True, "train_modules", ["q_action", "bank"], "train_modules"),
        (True, "optim.reset_state_modules", ["bank"], "reset exactly"),
        (True, "optim.lr_scales.q_action", 0.0, "q_action to 1.0"),
        (True, "losses.dyn.detach_coeff", True, "detach_coeff to false"),
        (True, "losses.dyn.detach_coeff", None, "detach_coeff to false"),
        (True, "losses.dyn.neg_weight", 1.0, "N4 dynamics negative weight"),
        (True, "losses.act.enabled", False, "losses.act.enabled"),
        (True, "losses.act.weight", 0.0, "losses.act.weight"),
        (True, "losses.act.align_to", "q_delta", "losses.act.align_to"),
        (True, "losses.act.decode_from", "proposal", "losses.act.decode_from"),
        (False, "optim.reset_state_modules", ["bank", "q_action"], "reset exactly"),
        (False, "optim.lr_scales.q_action", 1.0, "q_action at 0.0"),
        (False, "losses.dyn.detach_coeff", False, "detached q_action"),
        (False, "optim.lr_scales.bank", 1.0, "lr_scales.bank"),
        (False, "optim.lr_scales.proposal", 0.1, "lr_scales.proposal"),
    ],
)
def test_candidate_recipe_rejects_partial_or_noninvariant_combinations(
    joint, path, value, failure,
):
    cfg = _candidate_recipe_config(joint=joint)
    node = cfg
    parts = path.split(".")
    for part in parts[:-1]:
        node = node[part]
    if value is None:
        node.pop(parts[-1])
    else:
        node[parts[-1]] = value
    with pytest.raises(oracle.OracleError, match=failure):
        oracle.authenticate_candidate_recipe(cfg)


@pytest.mark.parametrize(("mutation", "failure"), [
    ("source", "transition reset"),
    ("tensor", "transition reset"),
    ("name", "must not declare a parameter reset"),
])
def test_identity_centered_candidate_recipe_rejects_reset_drift(mutation, failure):
    cfg = _candidate_recipe_config(joint=True, identity_centered=True)
    if mutation == "source":
        cfg["optim"]["transition_parameter_reset"]["source_config_hash"] = (
            "0ec8af0a26135ecc"
        )
    elif mutation == "tensor":
        cfg["optim"]["transition_parameter_reset"]["tensors"] = {
            "bank.log_r": "zero"
        }
    else:
        cfg["run"]["name"] = "r0a_bank_ca_qa"
    with pytest.raises(oracle.OracleError, match=failure):
        oracle.authenticate_candidate_recipe(cfg)


def test_checkpoint_provenance_persists_joint_recipe_identity(tmp_path):
    cfg = _candidate_recipe_config(joint=True)
    experiment = {key: value for key, value in cfg.items() if key != "link"}
    config_hash = hashlib.blake2b(
        json.dumps(experiment, sort_keys=True, default=str).encode(), digest_size=8,
    ).hexdigest()
    checkpoint = tmp_path / "candidate.pt"
    torch.save({
        "consolidated": {},
        "model": {
            name: {"weight": torch.ones(1)}
            for name in ("estimator", "proposal", "decoder", "q_action", "bank")
        },
        "resolved_config": cfg,
        "config_hash": config_hash,
        "global_step": 57_666,
        "samples_seen": 1,
        "git_sha": "test",
        "world_size": 16,
    }, checkpoint)
    provenance = oracle.checkpoint_provenance(checkpoint)
    recipe = provenance["candidate_recipe"]
    assert recipe["kind"] == oracle.CANDIDATE_RECIPE_JOINT_Q_ACTION_BANK
    assert recipe["detach_coeff"] is False
    assert recipe["reset_state_modules"] == ["bank", "q_action"]
    assert recipe["q_action_lr_scale"] == 1.0
    assert recipe["action_anchor"]["decode_from"] == "q_action"

    identity_cfg = _candidate_recipe_config(
        joint=True, identity_centered=True,
    )
    identity_hash = hashlib.blake2b(
        json.dumps(identity_cfg, sort_keys=True, default=str).encode(), digest_size=8,
    ).hexdigest()
    identity_checkpoint = tmp_path / "identity-candidate.pt"
    torch.save({
        "consolidated": {},
        "model": {
            name: {"weight": torch.ones(1)}
            for name in ("estimator", "proposal", "decoder", "q_action", "bank")
        },
        "resolved_config": identity_cfg,
        "config_hash": identity_hash,
        "global_step": 57_666,
    }, identity_checkpoint)
    identity_recipe = oracle.checkpoint_provenance(
        identity_checkpoint
    )["candidate_recipe"]
    assert identity_recipe["kind"] == (
        oracle.CANDIDATE_RECIPE_JOINT_Q_ACTION_BANK_IDENTITY_CENTERED
    )
    assert identity_recipe["transition_parameter_reset"] == (
        oracle.IDENTITY_CENTERED_RESET
    )


def test_root_cache_keeps_argmax_exact_and_sha_samples_stable():
    item = iter_work(oracle.official_protocol())[0]
    proposal = TinyProposal()
    z = torch.zeros(1, C.K, C.D)
    lang = torch.zeros(1, 1, 1)
    one = oracle.RootCandidateCache(item)
    got0 = one.choose(0, proposal, z, lang, "sha256:obs")
    assert torch.equal(got0, proposal.argmax(z, lang))
    got7 = one.choose(7, proposal, z, lang, "sha256:obs")

    two = oracle.RootCandidateCache(item)
    two.choose(0, proposal, z, lang, "sha256:obs")
    again7 = two.choose(7, proposal, z, lang, "sha256:obs")
    assert torch.equal(got7, again7)
    assert one.root_evidence[0]["kind"] == "proposal.argmax"
    assert one.root_evidence[7]["sample_seed"] == oracle.root_seed(item, 7)

    with pytest.raises(oracle.OracleError, match="identical reset"):
        one.choose(1, proposal, z, lang, "sha256:different")


def test_execute_group_uses_common_noise_exact_segment_and_direct_continuation():
    item = iter_work(oracle.official_protocol())[0]
    envs = []

    def fresh_env_factory(*args, **kwargs):  # noqa: ARG001
        env = ShortEnv()
        envs.append(env)
        return env

    group = oracle.execute_group(
        item, tiny_bundle(), backend="fake", env_factory=fresh_env_factory,
    )
    assert len(envs) == 16 and len({id(env) for env in envs}) == 16
    assert len(group["arms"]) == 16
    assert group["parity"]["passed"]
    assert group["parity"]["checks"]["complete_root_segment_executed"]
    assert all(arm["error"] is None for arm in group["arms"])
    assert all(arm["terminal_success"] for arm in group["arms"])
    assert group["arms"][0]["root_kind"] == "proposal.argmax"
    assert all(arm["execution"]["n_forced_roots"] == 1 for arm in group["arms"])
    assert all(arm["execution"]["n_direct_continuations"] == 1
               for arm in group["arms"])
    assert all(arm["execution"]["residual_uses_planned_segment"]
               for arm in group["arms"])
    assert all(arm["execution"]["executed_root_prefix_matches"]
               for arm in group["arms"])
    assert all(arm["execution"]["root_env_segment_complete"]
               for arm in group["arms"])
    assert len({
        arm["execution"]["decoder_rng_before_sha256"] for arm in group["arms"]
    }) == 1
    assert group["arms"][0]["eligible"]
    assert any(not arm["eligible"] for arm in group["arms"][1:])


def test_final_gate_requires_all_rows_parity_exact_arm0_and_360_masked():
    groups, baseline = _synthetic_groups(masked_successes=400)
    summary = oracle.summarize_groups(
        groups, baseline_outcomes=baseline, provenance=_provenance(),
    )
    assert summary["passed"] and summary["status"] == "PASS"
    assert summary["n_work_items"] == 400
    assert summary["n_rows"] == 6400
    assert summary["n_errors"] == 0
    assert summary["arm0"]["n_success"] == 149
    assert summary["arm0"]["suite_success"] == {"libero_spatial": 40,
                                                   "libero_object": 32,
                                                   "libero_goal": 48,
                                                   "libero_long": 29}
    assert summary["arm0"]["reference_exact"]
    assert summary["masked_oracle"]["n_success"] == 400

    low_groups, _ = _synthetic_groups(masked_successes=359)
    # Remove arm-0 successes after the requested cutoff so they cannot lift the
    # masked total back above the explicit synthetic threshold.
    for ordinal, group in enumerate(low_groups):
        if ordinal >= 359:
            group["arms"][0]["eligible"] = False
            group["arms"][0]["residual_l2"] = 0.6
    low = oracle.summarize_groups(
        low_groups, baseline_outcomes=baseline, provenance=_provenance(),
    )
    assert not low["passed"]
    assert any("masked oracle" in failure for failure in low["failures"])


def test_final_gate_accepts_exact_identity_centered_recipe_provenance():
    groups, baseline = _synthetic_groups(masked_successes=400)
    summary = oracle.summarize_groups(
        groups,
        baseline_outcomes=baseline,
        provenance=_provenance(identity_centered=True),
    )
    assert summary["passed"] and summary["status"] == "PASS"


@pytest.mark.parametrize(
    "mutation",
    [
        "missing", "error", "parity", "arm0", "source_missing",
        "source_mutated", "candidate_recipe_mutated",
    ],
)
def test_final_gate_fails_closed_on_incomplete_or_inconsistent_rows(mutation):
    groups, baseline = _synthetic_groups()
    provenance = _provenance()
    if mutation == "missing":
        groups.pop()
    elif mutation == "error":
        groups[0]["arms"][4]["error"] = "traceback"
    elif mutation == "parity":
        groups[0]["parity"]["passed"] = False
    elif mutation == "arm0":
        groups[0]["arms"][0]["terminal_success"] = not baseline[groups[0]["group_id"]]
    elif mutation == "source_missing":
        provenance["source"].pop("behavior_source_digest")
    elif mutation == "source_mutated":
        provenance["source"]["behavior_source_files"][0]["sha256"] = "f" * 64
    else:
        provenance["checkpoint"]["candidate_recipe"]["q_action_lr_scale"] = 1.0
    summary = oracle.summarize_groups(
        groups, baseline_outcomes=baseline, provenance=provenance,
    )
    assert not summary["passed"]
    assert summary["failures"]


@pytest.mark.parametrize(
    ("mutation", "failure_text"),
    [
        ("missing_execution", "execution evidence is missing"),
        ("mutated_root", "root coefficient witness does not match"),
        ("mutated_rng", "decoder RNG start differs across arms"),
        ("incomplete_root", "complete root segment"),
        ("missing_parity_check", "parity failures"),
        ("derived_masked", "derived field masked_oracle_success"),
    ],
)
def test_final_gate_recomputes_stored_arm_witnesses(mutation, failure_text):
    groups, baseline = _synthetic_groups()
    group = groups[0]
    if mutation == "missing_execution":
        group["arms"][3].pop("execution")
    elif mutation == "mutated_root":
        group["arms"][3]["root"]["sha256"] = oracle._hash_value("wrong root")
    elif mutation == "mutated_rng":
        group["arms"][3]["execution"]["decoder_rng_before_sha256"] = (
            oracle._hash_value("wrong rng")
        )
    elif mutation == "incomplete_root":
        execution = group["arms"][3]["execution"]
        execution["root_env_steps_executed"] = 4
        execution["root_env_segment_complete"] = False
    elif mutation == "missing_parity_check":
        group["parity"]["checks"].pop("identical_decoder_rng_before")
    else:
        group["masked_oracle_success"] = False
    summary = oracle.summarize_groups(
        groups, baseline_outcomes=baseline, provenance=_provenance(),
    )
    assert not summary["passed"]
    assert any(failure_text in failure for failure in summary["failures"])


def test_baseline_and_bank_gate_provenance_are_authenticated(tmp_path):
    outcomes = _baseline_outcomes()
    episodes = []
    for item in iter_work(oracle.official_protocol()):
        episodes.append({
            **item.to_dict(),
            "success": outcomes[oracle.work_key(item)],
            "error": None,
            "extra": {"policy_seed": item.policy_seed},
        })
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps({
        "protocol": oracle.official_protocol().to_dict(),
        "meta": {
            "backend": "libero",
            "env_available": True,
            "ckpt": "/candidate/step49666.pt",
            "policy_seed_scheme": "sha256(work-item)-v1",
            "eval_identity": {"policy_kw": {"allow_stub": False}},
            "policy": {
                "is_stub": False,
                "ckpt_global_step": 49_666,
                "embodiment": "libero_franka",
                "decoder_samples": 1,
                "gripper_dwell": 1,
                "duration_normalize_segments": False,
            },
        },
        "episodes": episodes,
    }))
    baseline = oracle.validate_baseline_results(baseline_path)
    assert baseline["n_success"] == 149
    assert baseline["n_errors"] == 0

    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps({
        "passed": True,
        "status": "PASS",
        "overall_verdict": "PASS",
        "format_version": 1,
        "direct_e2e_run": False,
        "candidate": {"sha256": "a" * 64, "config_hash": "cfg"},
        "data": {
            "manifest_digest": "sha256:heldout",
            "trajectory_manifest": {
                "digest": "sha256:heldout", "source": "libero", "split": "gate",
            },
        },
        "gates": {
            name: {
                "passed": True,
                "threshold": 0.0,
                **({"ci_lower": 0.1} if "spread" in name else {
                    "ci_lower_per_horizon": [0.1, 0.1, 0.1, 0.1]
                }),
            }
            for name in oracle.BANK_GATE_REQUIREMENTS
        },
        "source_provenance": {"gate_source_sha256": "b" * 64},
    }))
    gate = oracle.validate_bank_gate(gate_path, "a" * 64, "cfg")
    assert gate["passed"] and gate["candidate_sha256"] == "a" * 64
    with pytest.raises(oracle.OracleError, match="does not match"):
        oracle.validate_bank_gate(gate_path, "c" * 64)


def test_joint_bank_gate_requires_full_action_semantics_provenance(tmp_path):
    base_gates = {
        name: {
            "passed": True,
            "threshold": 0.0,
            **({"ci_lower": 0.1} if "spread" in name else {
                "ci_lower_per_horizon": [0.1] * C.DEPTH,
            }),
        }
        for name in oracle.BANK_GATE_REQUIREMENTS
    }
    frozen = {
        name: {"present": True, "tensors": 1, "numel": 1, "tensor_exact": True}
        for name in ("estimator", "ema", "q_delta", "decoder", "proposal", "potential")
    }
    report = {
        "passed": True,
        "status": "PASS",
        "overall_verdict": "PASS",
        "format_version": 1,
        "direct_e2e_run": False,
        "candidate": {"sha256": "a" * 64, "config_hash": "qa"},
        "reference": {
            "sha256": oracle.QA_REFERENCE_SHA256,
            "config_hash": oracle.QA_REFERENCE_CONFIG_HASH,
            "global_step": oracle.QA_REFERENCE_GLOBAL_STEP,
            "model_config_exact": True,
            "frozen_modules": frozen,
        },
        "recipe": {"method_variant": "joint_q_action_bank"},
        "data": {
            "manifest_digest": "sha256:heldout",
            "trajectory_manifest": {
                "digest": "sha256:heldout", "source": "libero", "split": "gate",
            },
        },
        "gates": {
            **base_gates,
            "deploy_action_semantics_preservation": {
                "passed": True,
                "action_decode_improvement_ci_low": 0.0,
                "proposal_support_overlap_change_ci_low": -0.05,
                "thresholds": {
                    "action_decode_improvement_ci_low": 0.0,
                    "proposal_support_overlap_change_ci_low": -0.05,
                },
            },
            "proposal_root_q_action_residual_preservation": {
                "passed": True,
                "q_action_residual_max": 0.5,
                "max_root_exhaustion_rate": 0.01,
                "root_exhaustion_rate": 0.0,
            },
        },
        "source_provenance": {"gate_source_sha256": "b" * 64},
    }
    path = tmp_path / "joint_gate.json"
    path.write_text(json.dumps(report))
    joint_recipe = oracle.authenticate_candidate_recipe(
        _candidate_recipe_config(joint=True)
    )
    assert oracle.validate_bank_gate(
        path, "a" * 64, "qa", joint_recipe,
    )["passed"]

    identity_report = copy.deepcopy(report)
    identity_report["recipe"]["method_variant"] = (
        "joint_q_action_bank_identity_centered"
    )
    path.write_text(json.dumps(identity_report))
    identity_recipe = oracle.authenticate_candidate_recipe(
        _candidate_recipe_config(joint=True, identity_centered=True)
    )
    assert oracle.validate_bank_gate(
        path, "a" * 64, "qa", identity_recipe,
    )["method_variant"] == "joint_q_action_bank_identity_centered"

    broken = copy.deepcopy(report)
    broken["reference"]["frozen_modules"]["decoder"]["tensor_exact"] = False
    path.write_text(json.dumps(broken))
    with pytest.raises(oracle.OracleError, match="tensor exactness"):
        oracle.validate_bank_gate(path, "a" * 64, "qa", joint_recipe)


def test_grouped_store_is_atomic_resumable_and_identity_checked(tmp_path):
    groups, _ = _synthetic_groups()
    path = tmp_path / "oracle.json"
    identity = {
        "checkpoint_sha256": "x",
        "behavior_source_digest": "a" * 64,
        "recipe": 1,
    }
    store = oracle.OracleStore(
        path, identity=identity, provenance=_provenance(), resume=True,
    )
    store.add(groups[0])
    blob = json.loads(path.read_text())
    assert blob["summary"]["status"] == "RUNNING"
    assert len(blob["groups"]) == 1
    assert not list(tmp_path.glob("oracle.json.tmp"))

    resumed = oracle.OracleStore(
        path, identity=identity, provenance=_provenance(), resume=True,
    )
    assert list(resumed.groups) == [groups[0]["group_id"]]
    with pytest.raises(oracle.OracleError, match="different immutable identity"):
        oracle.OracleStore(
            path,
            identity={**identity, "behavior_source_digest": "b" * 64},
            provenance=_provenance(), resume=True,
        )


def test_cli_is_pinned_to_real_libero_and_requires_gate():
    args = oracle.parse_args([
        "--checkpoint", "candidate.pt",
        "--bank-gate", "gate.json",
        "--out", "oracle.json",
    ])
    assert args.backend == "libero"
    assert args.baseline_results.endswith(
        "eval_r0a_deploy_s1_s49666_seeded1200_v2/seed0/results.json"
    )
    assert args.workers is None
