"""Focused contracts for full-trajectory terminal-outcome collection."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import Tensor, nn

import contracts as C
from loom.eval import libero
from loom.eval import outcome_recovery as recovery
from loom.eval.policy import LoomPolicy, PolicyModules, zeros_featurizer
from loom.eval.runner import WorkItem


class FixedEstimator(nn.Module):
    def forward(self, feats, z_prev=None):
        batch = feats["proprio"].shape[0]
        offset = 0.0 if z_prev is None else 0.01
        return torch.full(
            (batch, C.K, C.D), offset,
            device=feats["proprio"].device,
            dtype=feats["proprio"].dtype,
        )


class FixedProposal(nn.Module):
    def logits(self, z, lang):
        # Non-uniform, non-tied logits make both order and old log-prob exact.
        return torch.linspace(-2.0, 2.0, C.M, device=z.device, dtype=z.dtype) \
            .expand(z.shape[0], C.M)

    def argmax(self, z, lang):
        from loom.heads.proposal import weights_from_logits

        logits = self.logits(z, lang)
        order = logits.float().topk(C.TOPK, dim=-1).indices
        return weights_from_logits(logits, order, C.M)


class NoiseDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls: list[tuple[Tensor, Tensor]] = []
        self.outputs: list[Tensor] = []

    def forward(self, proprio, coeff, *, generator=None):
        noise = torch.randn(
            coeff.shape[0], C.H_OP, 7,
            generator=generator, device=coeff.device, dtype=coeff.dtype,
        )
        self.calls.append((coeff.detach().clone(), noise.detach().clone()))
        out = 0.01 * noise + 0.001 * coeff[..., :7].unsqueeze(1)
        self.outputs.append(out.detach().clone())
        return out


def modules(*, q_action=None) -> PolicyModules:
    return PolicyModules(
        estimator=FixedEstimator(),
        proposal=FixedProposal(),
        decoder=NoiseDecoder(),
        q_action=q_action,
        featurize=zeros_featurizer(C.EMBODIMENTS["libero_franka"]),
        embodiment="libero_franka",
        device="cpu",
        is_stub=False,
        meta={
            "ckpt_global_step": recovery.SEED_GLOBAL_STEP,
            "ckpt_config_hash": recovery.SEED_CONFIG_HASH,
        },
    )


OBS = {"state": np.zeros(7, dtype=np.float32)}


def item(*, trial: int = 10, max_steps: int = 1) -> WorkItem:
    suite = "libero_spatial"
    return WorkItem(
        bench="libero", suite=suite, task_id=0, episode=trial, seed=0,
        env_seed=1234, policy_seed=5678, max_steps=max_steps,
    )


def fake_env_factory(suite, task_id, seed, *, trial_id, backend, max_steps):
    assert backend == "fake"
    env = libero.FakeLiberoEnv(
        suite, task_id, seed, trial_id=trial_id, max_steps=max_steps,
        p_success=1.0, image_size=8,
    )
    # Solve on the first policy action, after LIBERO's 15 settle actions.
    env._will_succeed = True
    env._solve_step = 16
    return env


def fake_bundle() -> recovery.RecoveryBundle:
    return recovery.RecoveryBundle(
        modules=modules(), provenance={"test": True},
    )


def identity(split: str = "train0") -> dict:
    return {
        "format_version": recovery.FORMAT_VERSION,
        "test": True,
        "split": split,
        "checkpoint": {
            "global_step": recovery.SEED_GLOBAL_STEP,
            "config_hash": recovery.SEED_CONFIG_HASH,
            "sha256": recovery.SEED_CHECKPOINT_SHA256,
        },
    }


def test_trial_partitions_and_work_sets_are_exact_and_disjoint():
    assert recovery.split_trials("official") == tuple(range(0, 10))
    assert recovery.split_trials("train") == tuple(range(10, 40))
    assert recovery.split_trials("validation") == tuple(range(40, 50))
    parts = [set(recovery.split_trials(name))
             for name in ("official", "train", "validation")]
    assert set.union(*parts) == set(range(50))
    assert all(not (a & b) for i, a in enumerate(parts) for b in parts[i + 1:])
    assert len(recovery.collection_items("official")) == 400
    assert len(recovery.collection_items("train")) == 1200
    assert len(recovery.collection_items("validation")) == 400
    for fold in range(6):
        name = f"train{fold}"
        assert recovery.split_trials(name) == tuple(
            range(10 + 5 * fold, 15 + 5 * fold)
        )
        assert len(recovery.collection_items(name)) == 200
    fold_sets = [set(v) for v in recovery.TRAIN_FOLDS.values()]
    assert set.union(*fold_sets) == set(range(10, 40))
    assert all(not (a & b) for i, a in enumerate(fold_sets)
               for b in fold_sets[i + 1:])
    assert {x.episode for x in recovery.collection_items("train")} == set(range(10, 40))
    with pytest.raises(ValueError, match="unknown recovery split"):
        recovery.split_trials("gate")


def test_group_is_fixed_at_one_direct_plus_seven_sampled_arms():
    assert recovery.GROUP_SIZE == 8
    assert recovery.N_SAMPLED_ARMS == 7
    seeds = [recovery.proposal_seed(99, arm) for arm in range(1, 8)]
    assert len(set(seeds)) == 7
    assert seeds == [recovery.proposal_seed(99, arm) for arm in range(1, 8)]
    with pytest.raises(ValueError, match="sampled arms"):
        recovery.proposal_seed(99, 0)
    with pytest.raises(ValueError, match="sampled arms"):
        recovery.proposal_seed(99, 8)


def test_sample_records_exact_draw_order_and_exact_old_pl_logprob():
    from loom.heads.proposal import gumbel_topk, pl_log_prob

    policy = recovery.OutcomeRecoveryPolicy(modules(), arm=3)
    policy.set_policy_seed(91)
    policy.reset()

    logits = policy.modules.proposal.logits(
        torch.zeros(1, C.K, C.D), torch.zeros(1, 64, 1152),
    )
    generator = torch.Generator().manual_seed(recovery.proposal_seed(91, 3))
    expected_order = gumbel_topk(logits, C.TOPK, generator)
    expected_lp = pl_log_prob(logits.float(), expected_order)

    policy.act(OBS, "pick the object")
    payload = policy.trajectory_payload(terminal_success=True)
    assert torch.equal(payload["ordered_support"], expected_order)
    assert torch.equal(payload["old_logprob"], expected_lp.float())
    assert payload["behavior"] == "ordered_pl_sample"
    assert payload["behavior_logprob_valid"] is True
    assert payload["proposal_seed"] == recovery.proposal_seed(91, 3)
    support = payload["coeff"].ne(0)
    assert bool(support.gather(-1, payload["ordered_support"]).all())


def test_direct_arm_is_action_exact_but_marked_out_of_grpo_logprob():
    direct_modules = modules()
    recovery_modules = copy.deepcopy(direct_modules)
    direct = LoomPolicy(direct_modules)
    arm0 = recovery.OutcomeRecoveryPolicy(recovery_modules, arm=0)
    for policy in (direct, arm0):
        policy.set_policy_seed(321)
        policy.reset()
    expected = direct.act(OBS, "task")
    actual = arm0.act(OBS, "task")
    assert np.array_equal(actual, expected)
    assert torch.equal(direct.last_coeff, arm0.last_coeff)
    assert torch.equal(
        direct_modules.decoder.calls[-1][1],
        recovery_modules.decoder.calls[-1][1],
    )
    payload = arm0.trajectory_payload(terminal_success=False)
    assert payload["behavior"] == "direct_argmax"
    assert payload["behavior_logprob_valid"] is False
    assert payload["proposal_seed"] is None
    assert payload["proposal_rng_before"] == []


def test_sample_is_executed_directly_without_q_action_or_fallback():
    policy = recovery.OutcomeRecoveryPolicy(modules(), arm=1)
    policy.set_policy_seed(12)
    policy.reset()
    policy.act(OBS, "task")
    payload = policy.trajectory_payload(terminal_success=False)
    decoded_coeff = policy.modules.decoder.calls[-1][0][0]
    assert torch.equal(decoded_coeff, payload["coeff"][0])
    assert not torch.equal(payload["coeff"][0], modules().proposal.argmax(
        torch.zeros(1, C.K, C.D), torch.zeros(1, 64, 1152),
    )[0])
    with pytest.raises(ValueError, match="q_action=None"):
        recovery.OutcomeRecoveryPolicy(modules(q_action=object()), arm=1)


def test_detached_z_and_language_are_sidecars_and_reward_is_terminal_only():
    policy = recovery.OutcomeRecoveryPolicy(modules(), arm=2)
    policy.set_policy_seed(17)
    policy.reset()
    executed = policy.act(OBS, "task")
    payload = policy.trajectory_payload(terminal_success=True)
    assert payload["z"].shape == (1, C.K, C.D)
    assert payload["lang"].shape == (64, 1152)
    assert payload["z"].device.type == payload["lang"].device.type == "cpu"
    assert not payload["z"].requires_grad and not payload["lang"].requires_grad
    assert payload["terminal_reward"].shape == torch.Size([])
    assert float(payload["terminal_reward"]) == 1.0
    assert [k for k in payload if "reward" in k] == ["terminal_reward"]
    decoded = policy.modules.decoder.outputs[-1][0].float().cpu().numpy()
    assert payload["decoded_action_segment_sha256"] == [
        recovery.action_segment_sha256(decoded)
    ]
    assert payload["executed_action_segment_sha256"] == [
        recovery.action_segment_sha256(executed[None])
    ]
    assert payload["executed_action_steps"] == [1]


def test_common_decoder_rng_is_aligned_by_replan_not_proposal_arm():
    policies = [recovery.OutcomeRecoveryPolicy(modules(), arm=a) for a in (0, 1, 2)]
    for policy in policies:
        policy.set_policy_seed(345)
        policy.reset()
        # The first clock segment is five env actions; call six times to enter
        # replan two while leaving its exact one-action executed prefix.
        for _ in range(6):
            policy.act(OBS, "task")
    payloads = [p.trajectory_payload(terminal_success=False) for p in policies]
    for payload in payloads[1:]:
        assert payload["decoder_rng_before"] == payloads[0]["decoder_rng_before"]
        assert payload["decoder_rng_after"] == payloads[0]["decoder_rng_after"]
    assert not torch.equal(payloads[1]["ordered_support"], payloads[2]["ordered_support"])
    assert payloads[1]["proposal_rng_before"] != payloads[2]["proposal_rng_before"]
    assert all(payload["executed_action_steps"] == [5, 1] for payload in payloads)


def test_collect_group_writes_one_atomic_valid_eight_arm_sidecar(tmp_path):
    ident = identity()
    digest = recovery.identity_digest(ident)
    receipt = recovery.collect_group(
        item(), fake_bundle(), split="train0",
        collection_identity_digest=digest, out_dir=tmp_path,
        env_factory=fake_env_factory, backend="fake", bench_module=libero,
    )
    assert receipt["n_arms"] == 8
    assert receipt["terminal_rewards"] == [1] * 8
    assert receipt["n_replans_by_arm"] == [1] * 8
    sidecar = tmp_path / receipt["sidecar"]
    assert sidecar.is_file() and recovery.sha256_file(sidecar) == receipt["sha256"]
    assert not list(sidecar.parent.glob(".*.tmp-*"))
    payload = torch.load(sidecar, map_location="cpu", weights_only=False)
    recovery.validate_group_payload(
        payload, item=item(), expected_identity_digest=digest,
        expected_split="train0",
    )
    assert payload["common_decoder_rng"]["passed"] is True
    assert payload["common_decoder_rng"]["shared_prefix_replans_by_arm"] == [1] * 8


def test_failed_arm_commits_no_group_sidecar(tmp_path):
    def crash_factory(suite, task_id, seed, *, trial_id, backend, max_steps):
        return libero.FakeLiberoEnv(
            suite, task_id, seed, trial_id=trial_id, max_steps=max_steps,
            p_success=0.0, crash_at=1, image_size=8,
        )

    with pytest.raises(RuntimeError, match="no group sidecar was committed"):
        recovery.collect_group(
            item(), fake_bundle(), split="train0",
            collection_identity_digest="a" * 64, out_dir=tmp_path,
            env_factory=crash_factory, backend="fake", bench_module=libero,
        )
    assert not list((tmp_path / "groups").glob("*.pt")) if (tmp_path / "groups").exists() else True


def test_manifest_resume_validates_and_orphan_adoption_is_atomic(tmp_path):
    work = [item()]
    ident = identity()
    store = recovery.RecoveryStore(
        tmp_path, identity=ident, split="train0", items=work,
    )
    receipt = recovery.collect_group(
        work[0], fake_bundle(), split="train0",
        collection_identity_digest=store.identity_digest, out_dir=tmp_path,
        env_factory=fake_env_factory, backend="fake", bench_module=libero,
    )
    # Simulate a crash after sidecar rename but before manifest commit.
    assert not store.has(work[0])
    resumed = recovery.RecoveryStore(
        tmp_path, identity=ident, split="train0", items=work,
    )
    assert resumed.has(work[0])
    final = resumed.finalize()
    assert final["summary"]["status"] == "COMPLETE"
    assert final["summary"]["n_trajectories"] == 8
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["groups"][0]["sha256"] == receipt["sha256"]
    assert not list(tmp_path.glob(".manifest.json.tmp-*"))


def test_resume_fails_closed_on_identity_or_sidecar_tamper(tmp_path):
    work = [item()]
    ident = identity()
    store = recovery.RecoveryStore(
        tmp_path, identity=ident, split="train0", items=work,
    )
    receipt = recovery.collect_group(
        work[0], fake_bundle(), split="train0",
        collection_identity_digest=store.identity_digest, out_dir=tmp_path,
        env_factory=fake_env_factory, backend="fake", bench_module=libero,
    )
    store.add(receipt)
    with pytest.raises(RuntimeError, match="different immutable identity"):
        recovery.RecoveryStore(
            tmp_path, identity={**ident, "test": False}, split="train0", items=work,
        )
    with (tmp_path / receipt["sidecar"]).open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(RuntimeError, match="size changed"):
        recovery.RecoveryStore(
            tmp_path, identity=ident, split="train0", items=work,
        )


def test_manifest_rejects_malformed_terminal_or_common_noise_payload(tmp_path):
    ident = identity()
    digest = recovery.identity_digest(ident)
    receipt = recovery.collect_group(
        item(), fake_bundle(), split="train0",
        collection_identity_digest=digest, out_dir=tmp_path,
        env_factory=fake_env_factory, backend="fake", bench_module=libero,
    )
    payload = torch.load(
        tmp_path / receipt["sidecar"], map_location="cpu", weights_only=False,
    )
    shaped = copy.deepcopy(payload)
    shaped["arms"][1]["dense_reward"] = torch.ones(1)
    with pytest.raises(RuntimeError, match="sidecar fields differ"):
        recovery.validate_group_payload(
            shaped, item=item(), expected_identity_digest=digest,
            expected_split="train0",
        )
    divergent = copy.deepcopy(payload)
    divergent["arms"][2]["decoder_rng_after"][0] = "f" * 64
    with pytest.raises(RuntimeError, match="decoder post-state diverges"):
        recovery.validate_group_payload(
            divergent, item=item(), expected_identity_digest=digest,
            expected_split="train0",
        )


def test_checkpoint_authentication_is_exact_sha_step_and_config(tmp_path, monkeypatch):
    path = tmp_path / "seed.pt"
    path.write_bytes(b"known consolidated checkpoint bytes")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(recovery, "SEED_CHECKPOINT_SHA256", digest)
    got = recovery.authenticate_seed_checkpoint(path)
    assert got["sha256"] == digest
    assert got["global_step"] == 49_666
    assert got["config_hash"] == "a199324a6205bb6d"
    path.write_bytes(b"wrong bytes")
    with pytest.raises(RuntimeError, match="not the pinned step-49666 seed"):
        recovery.authenticate_seed_checkpoint(path)


def test_load_bundle_fails_closed_on_step_config_or_q_action(monkeypatch):
    class Base:
        def __init__(self, mods):
            self.modules = mods
            self.env_fps = 20.0
            self.clock = type("Clock", (), {"steps_per_segment": 16 / 3})()
            self.gripper_dwell = 1
            self.decoder_samples = 1
            self.duration_normalize_segments = False

    good = modules()
    good.meta["state_dict"] = {
        name: {"tensors_loaded": 1, "unexpected": 0}
        for name in ("estimator", "proposal", "decoder")
    }
    monkeypatch.setattr(recovery, "load_policy", lambda *a, **k: Base(good))
    assert recovery.load_recovery_bundle("ignored", device="cpu").modules is good

    bad_step = copy.deepcopy(good)
    bad_step.meta["ckpt_global_step"] = 49_665
    monkeypatch.setattr(recovery, "load_policy", lambda *a, **k: Base(bad_step))
    with pytest.raises(RuntimeError, match="loaded checkpoint step"):
        recovery.load_recovery_bundle("ignored", device="cpu")

    bad_q = copy.deepcopy(good)
    bad_q.q_action = object()
    monkeypatch.setattr(recovery, "load_policy", lambda *a, **k: Base(bad_q))
    with pytest.raises(RuntimeError, match="unexpectedly constructed q_action"):
        recovery.load_recovery_bundle("ignored", device="cpu")


def test_collection_identity_authenticates_method_and_all_three_splits():
    checkpoint = {
        "path": "/seed.pt", "size": 1, "mtime_ns": 2,
        "sha256": recovery.SEED_CHECKPOINT_SHA256,
        "global_step": recovery.SEED_GLOBAL_STEP,
        "config_hash": recovery.SEED_CONFIG_HASH,
        "kind": "consolidated",
    }
    got = recovery.collection_identity(
        checkpoint=checkpoint, split="validation", source_sha256="a" * 64,
    )
    assert got["group"]["size"] == 8
    assert got["group"]["sampled_arms"] == list(range(1, 8))
    assert got["split"]["train_trial_ids"] == list(range(10, 40))
    assert got["split"]["validation_trial_ids"] == list(range(40, 50))
    assert got["split"]["official_trial_ids"] == list(range(0, 10))
    assert got["split"]["train_folds"] == {
        f"train{fold}": list(range(10 + 5 * fold, 15 + 5 * fold))
        for fold in range(6)
    }
    assert got["policy"] == {
        "path": "estimator->proposal->{argmax|ordered_PL}->decoder",
        "q_action": False, "bank": False, "fallback": False,
        "duration_normalize_segments": False,
        "gripper_dwell": 1, "decoder_samples": 1,
    }
    assert got["sidecar"]["reward"] == "terminal_LIBERO_success_only"
    assert "loom/data/adapters/libero.py" in got["source"]["files"]
    assert got["group"]["action_segment_witness_scheme"] == (
        recovery.ACTION_SEGMENT_WITNESS_SCHEME
    )


def test_cli_has_no_fake_or_mutable_trial_partition_options():
    parser = recovery.build_parser()
    args = parser.parse_args([
        "--checkpoint", "/seed.pt", "--out-dir", "/out", "--split", "train0",
    ])
    assert args.split == "train0" and args.workers is None
    options = {flag for action in parser._actions for flag in action.option_strings}
    assert "--backend" not in options
    assert "--dry-run" not in options
    assert "--trials" not in options
    assert "--seed" not in options


def test_eval_module_keeps_training_heads_out_of_module_scope():
    path = Path(recovery.__file__)
    tree = ast.parse(path.read_text())
    imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert not [name for name in imports if name.startswith("loom.heads")]
    assert not [name for name in imports if name.startswith("loom.model")]
    assert not [name for name in imports if name.startswith("loom.train")]
