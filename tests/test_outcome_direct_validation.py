"""Focused contracts for authenticated held-out direct validation."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from loom.eval import EpisodeResult, episode_seed, policy_seed
from loom.eval import outcome_recovery as recovery
from scripts import outcome_direct_validation as direct


def _trust_gate() -> dict:
    rows = {
        "clip_fraction": {"value": 0.1, "op": "<=", "threshold": 0.2,
                          "pass": True},
        "ess_fraction": {"value": 0.9, "op": ">=", "threshold": 0.8,
                         "pass": True},
        "coeff_drift_p95_l1": {"value": 0.02, "op": "<=", "threshold": 0.05,
                                "pass": True},
        "live_ops": {"value": 20, "op": ">=", "threshold": 16, "pass": True},
        "expert_target_identity": {"value": True, "op": "==", "threshold": True,
                                   "pass": True},
        "expert_sparse_ce_no_worsening": {"value": 0.9, "op": "<=",
                                           "threshold": 1.0, "pass": True},
        "expert_topk_overlap_decline": {"value": 0.01, "op": "<=",
                                         "threshold": 0.05, "pass": True},
        "nonfinite": {"value": 0, "op": "==", "threshold": 0, "pass": True},
        "unexpected_gradients": {"value": 0, "op": "==", "threshold": 0,
                                 "pass": True},
    }
    return {
        "passed": True, "checks": rows,
        "definitions": {
            "scope": "disjoint recovery validation split",
            "arm0_importance_ratios": 0,
        },
        "counts": {"training_nonfinite": 0, "final_nonfinite": 0},
    }


def _recipe(validation_identity: str = "2" * 64) -> dict:
    return {
        "format_version": 1,
        "algorithm": "stored_order_pl_clipped_grpo",
        "reward_normalisation": "complete-group population mean/std",
        "aggregation": "mean_replans_then_mean_sampled_trajectories",
        "arm0": "normalisation/control only; no ratio or gradient",
        "sampled_arms": list(range(1, 8)),
        "clip_eps": 0.2, "epochs": 2, "groups_per_epoch": 200,
        "optimizer_steps": 400,
        "optimizer": {
            "kind": "AdamW", "lr": 5e-6, "betas": [0.9, 0.95],
            "weight_decay": 0.05, "eps": 1e-8, "scheduler": None,
            "grad_clip": 1.0, "proposal_state_reset_at_entry": True,
        },
        "objectives": {"grpo": 1.0, "sparse_ce": 1.0,
                       "switch_balance": 1e-2},
        "train_collection_identity": "1" * 64,
        "validation_collection_identity": validation_identity,
        "trust_thresholds": {
            "max_clip_fraction": 0.2, "min_ess_fraction": 0.8,
            "max_coeff_drift_p95_l1": 0.05, "min_live_ops": 16,
            "expert_sparse_ce_no_worsening": True,
            "max_topk_overlap_decline": 0.05, "nonfinite": 0,
        },
    }


def _parent_identity() -> dict:
    return {
        "kind": "consolidated", "path": "/pinned/seed.pt", "size": 123,
        "mtime_ns": 456, "sha256": recovery.SEED_CHECKPOINT_SHA256,
        "global_step": recovery.SEED_GLOBAL_STEP,
        "config_hash": recovery.SEED_CONFIG_HASH,
    }


def _trainer_source() -> dict:
    return {"scheme": "test", "sha256": "3" * 64,
            "files": {"trainer.py": "4" * 64}}


def _optimizer() -> dict:
    return {
        "kind": "proposal_only_adamw", "parameter_names": ["weight"],
        "state_reset_at_entry": True,
        "state_dict": {
            "state": {
                0: {"step": torch.tensor(400.0), "exp_avg": torch.tensor([0.1]),
                    "exp_avg_sq": torch.tensor([0.2])},
            },
            "param_groups": [{
                "params": [0], "lr": 5e-6, "betas": (0.9, 0.95),
                "weight_decay": 0.05, "eps": 1e-8,
            }],
        },
    }


def _payload(validation_provenance: dict | None = None):
    parent_state = {
        "estimator.weight": torch.tensor([1.0]),
        "proposal.weight": torch.tensor([2.0]),
        "decoder.weight": torch.tensor([3.0]),
    }
    state = {name: value.clone() for name, value in parent_state.items()}
    state["proposal.weight"] = torch.tensor([2.25])
    config = {"model": {"test": True}, "outcome_grpo": _recipe()}
    config_hash = direct._config_hash(config)
    trainer_source = _trainer_source()
    provenance = {
        "format_version": 1, "kind": direct.DESCENDANT_KIND,
        "trainer_source": trainer_source,
        "parent": _parent_identity(),
        "parent_config_hash": recovery.SEED_CONFIG_HASH,
        "parent_global_step": recovery.SEED_GLOBAL_STEP,
        "descendant_config_hash": config_hash,
        "descendant_global_step": direct.DESCENDANT_GLOBAL_STEP,
        "optimizer_steps": 400, "mutated_model_prefixes": ["proposal."],
        "training": {"optimizer_steps": 400, "groups_per_epoch": 200,
                     "nonfinite": 0, "unexpected_gradients": []},
        "behaviour_authentication": {
            "train": {"passed": True, "arm0_ratio_eligible_atoms": 0},
            "validation": {"passed": True, "arm0_ratio_eligible_atoms": 0},
        },
        "trust_gate": _trust_gate(),
        "validation_collection": validation_provenance or {},
    }
    provenance["frozen_model"] = direct._model_digest(parent_state, proposal=False)
    provenance["initial_proposal"] = direct._model_digest(parent_state, proposal=True)
    provenance["final_proposal"] = direct._model_digest(state, proposal=True)
    payload = {
        "model": state, "global_step": direct.DESCENDANT_GLOBAL_STEP,
        "config_hash": config_hash, "resolved_config": config,
        "optimizer": _optimizer(), "outcome_grpo": provenance,
        "consolidated": {
            "tool": "loom.train.outcome_grpo_round0", "proposal_only_update": True,
            "section": "model", "step": direct.DESCENDANT_GLOBAL_STEP,
            "frozen_model_sha256": provenance["frozen_model"]["sha256"],
        },
        "stop_reason": "outcome_grpo_round0_terminal_trust_pass",
        "world_size": 1,
    }
    identity = {
        "kind": direct.DESCENDANT_KIND, "path": "/candidate.pt",
        "size": 999, "mtime_ns": 111, "sha256": "5" * 64,
        "global_step": direct.DESCENDANT_GLOBAL_STEP, "config_hash": config_hash,
    }
    return payload, parent_state, identity, trainer_source


def _validated(payload, parent_state, identity, source):
    return direct.validate_descendant_payload(
        payload, checkpoint_identity=identity, parent_identity=_parent_identity(),
        parent_state=parent_state, trainer_source=source,
    )


def test_work_is_exact_trials_40_49_with_common_collection_seeds():
    items = direct.heldout_items()
    assert len(items) == 400
    assert {item.episode for item in items} == set(range(40, 50))
    assert {item.seed for item in items} == {0}
    assert {item.suite for item in items} == set(direct.DEFAULT_LIBERO_SUITES)
    for item in items:
        assert item.env_seed == episode_seed(
            0, "libero", item.suite, item.task_id, item.episode,
        )
        assert item.policy_seed == policy_seed(
            0, "libero", item.suite, item.task_id, item.episode,
        )
    protocol = direct.direct_protocol()
    assert protocol.total_episodes == 400
    assert protocol.seeds == (0,)
    assert "trials 40..49" in protocol.notes
    work = direct.work_identity(items)
    assert work == direct.work_identity(items)
    assert len(work["sha256"]) == 64


def test_policy_recipe_is_closed_direct_argmax_and_no_q_action_or_fallback():
    assert direct.POLICY_KW == {
        "embodiment": "libero_franka", "allow_stub": False,
        "n_candidates": 1, "op_stats": False, "gripper_dwell": 1,
        "decoder_samples": 1, "duration_normalize_segments": False,
        "_include_q_action": False,
    }
    options = {action.dest for action in direct.build_parser()._actions}
    assert options == {
        "help", "checkpoint", "validation_collection", "out", "workers", "quiet",
    }
    source = Path(direct.__file__).read_text()
    assert "collect_group(" not in source
    assert "run_collection(" not in source
    assert "OutcomeRecoveryPolicy" not in source


def test_exact_round0_payload_and_byte_preservation_pass():
    payload, parent, identity, source = _payload()
    result = _validated(payload, parent, identity, source)
    assert result["recipe"]["optimizer_steps"] == 400
    assert result["provenance"]["kind"] == direct.DESCENDANT_KIND
    assert result["frozen_model"] == direct._model_digest(parent, proposal=False)
    assert result["proposal"] == direct._model_digest(payload["model"], proposal=True)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda value: value["outcome_grpo"].update(kind="wrong"), "kind/version"),
        (lambda value: value["outcome_grpo"].update(optimizer_steps=399),
         "exact 400-step"),
        (lambda value: value["outcome_grpo"]["trust_gate"].update(passed=False),
         "passed terminal trust"),
        (lambda value: value["resolved_config"].update(outcome_grpo_round0={}),
         "legacy outcome_grpo_round0"),
        (lambda value: value["optimizer"]["state_dict"]["state"][0].update(
            step=torch.tensor(399.0)), "exactly 400"),
    ],
)
def test_payload_fails_closed_on_recipe_trust_or_step_drift(mutate, match):
    payload, parent, identity, source = _payload()
    mutate(payload)
    with pytest.raises(direct.DirectValidationError, match=match):
        _validated(payload, parent, identity, source)


def test_payload_rejects_any_frozen_tensor_change_even_with_claimed_provenance():
    payload, parent, identity, source = _payload()
    payload["model"]["decoder.weight"].add_(1.0)
    payload["outcome_grpo"]["frozen_model"] = direct._model_digest(
        payload["model"], proposal=False,
    )
    payload["consolidated"]["frozen_model_sha256"] = payload["outcome_grpo"][
        "frozen_model"
    ]["sha256"]
    with pytest.raises(direct.DirectValidationError, match="byte-identical"):
        _validated(payload, parent, identity, source)


def test_trust_gate_recomputes_claimed_pass_not_boolean_only():
    payload, parent, identity, source = _payload()
    payload["outcome_grpo"]["trust_gate"]["checks"]["ess_fraction"].update(
        value=0.1, pass_=True,
    )
    # Preserve the real key; ``pass_`` is an extra adversarial distraction.
    payload["outcome_grpo"]["trust_gate"]["checks"]["ess_fraction"]["pass"] = True
    with pytest.raises(direct.DirectValidationError, match="claims PASS inconsistently"):
        _validated(payload, parent, identity, source)


class FakeCollection:
    def __init__(self):
        self.items = direct.heldout_items()
        self.identity_digest = "2" * 64
        self.receipts = tuple({
            "group_id": recovery.work_key(item),
            "terminal_rewards": [int(index % 3 == 0), 0, 0, 0, 0, 0, 0, 0],
        } for index, item in enumerate(self.items))
        self.unchanged_calls = 0
        self._provenance = {
            "path": "/validation", "manifest": "/validation/manifest.json",
            "manifest_sha256": "6" * 64, "identity_digest": self.identity_digest,
            "split": "validation", "n_groups": 400, "n_trajectories": 3200,
            "terminal_successes_by_arm": [134, 0, 0, 0, 0, 0, 0, 0],
            "replans_by_arm": [1] * 8,
            "collector_source": {"sha256": "7" * 64},
        }

    def provenance(self):
        return copy.deepcopy(self._provenance)

    def assert_unchanged(self):
        self.unchanged_calls += 1


def _checkpoint(collection: FakeCollection) -> direct.DescendantCheckpoint:
    payload, parent, identity, source = _payload(collection.provenance())
    result = _validated(payload, parent, identity, source)
    return direct.DescendantCheckpoint(
        identity=identity, parent_identity=_parent_identity(),
        recipe=result["recipe"], provenance=result["provenance"],
    )


def test_collection_binding_is_exact_and_parent_arm0_is_condition_keyed():
    collection = FakeCollection()
    checkpoint = _checkpoint(collection)
    assert direct.authenticate_collection_binding(collection, checkpoint) \
        == collection.provenance()
    controls = direct._parent_controls(collection)
    assert len(controls) == 400
    first = collection.items[0]
    assert controls[first.key()] == {
        "group_id": recovery.work_key(first), "success": True,
        "source": "validation manifest terminal_rewards[0]",
    }
    collection.items = tuple(reversed(collection.items))
    with pytest.raises(direct.DirectValidationError, match="WorkItems differ"):
        direct.authenticate_collection_binding(collection, checkpoint)


def _policy_provenance(checkpoint: direct.DescendantCheckpoint) -> dict:
    return {
        "policy": "LoomPolicy", "is_stub": False,
        "ckpt": checkpoint.identity["path"],
        "ckpt_global_step": direct.DESCENDANT_GLOBAL_STEP,
        "ckpt_config_hash": checkpoint.identity["config_hash"],
        "gripper_dwell": 1, "decoder_samples": 1,
        "duration_normalize_segments": False,
        "state_dict": {
            name: {"tensors_loaded": 1, "unexpected": 0}
            for name in ("estimator", "proposal", "decoder")
        },
    }


def test_policy_provenance_rejects_stub_or_extra_q_action():
    checkpoint = _checkpoint(FakeCollection())
    value = _policy_provenance(checkpoint)
    direct._validate_policy_provenance(value, checkpoint)
    bad = copy.deepcopy(value)
    bad["is_stub"] = True
    with pytest.raises(direct.DirectValidationError, match="real LoomPolicy"):
        direct._validate_policy_provenance(bad, checkpoint)
    bad = copy.deepcopy(value)
    bad["state_dict"]["q_action"] = {"tensors_loaded": 1, "unexpected": 0}
    with pytest.raises(direct.DirectValidationError, match="exactly"):
        direct._validate_policy_provenance(bad, checkpoint)


def test_resume_repairs_only_missing_deterministic_parent_pair(tmp_path):
    collection = FakeCollection()
    controls = direct._parent_controls(collection)
    item = collection.items[0]
    record = EpisodeResult(
        bench=item.bench, suite=item.suite, task_id=item.task_id,
        episode=item.episode, seed=item.seed, env_seed=item.env_seed,
        success=False, steps=1, n_replans=1, task_name="fake",
        extra={"policy_seed": item.policy_seed},
    )

    class Store:
        records = {item.key(): record}
        flushed = 0

        def flush(self):
            self.flushed += 1

    store = Store()
    direct._repair_atomic_pairing_window(store, controls, collection.identity_digest)
    assert store.flushed == 1
    assert record.extra["outcome_recovery_parent_arm0"]["group_id"] \
        == recovery.work_key(item)
    # A present value is never rewritten; downstream exact validation rejects it.
    record.extra["outcome_recovery_parent_arm0"] = {"success": "tampered"}
    direct._repair_atomic_pairing_window(store, controls, collection.identity_digest)
    assert store.flushed == 1


def test_one_episode_per_condition_pairs_and_resumes_without_reexecution(
    tmp_path, monkeypatch,
):
    collection = FakeCollection()
    checkpoint = _checkpoint(collection)
    source = {"scheme": "test", "sha256": "8" * 64, "files": {}}
    monkeypatch.setattr(direct, "_source_identity", lambda: copy.deepcopy(source))
    monkeypatch.setattr(direct, "authenticate_descendant_checkpoint",
                        lambda path: checkpoint)
    monkeypatch.setattr(direct, "_open_validation_collection",
                        lambda path, parent: collection)
    monkeypatch.setattr(direct, "_assert_checkpoint_unchanged",
                        lambda identity: None)
    monkeypatch.setattr(direct.runner, "bench_module", lambda bench: object())
    monkeypatch.setattr(direct.runner, "env_available", lambda mod: True)
    monkeypatch.setattr(direct.runner, "ensure_runtime", lambda mod: None)
    monkeypatch.setattr(direct.runner, "_default_policy",
                        lambda ckpt, kw: object())
    monkeypatch.setattr(direct.runner, "_provenance",
                        lambda policy: _policy_provenance(checkpoint))
    calls: list[tuple] = []

    def fake_run(item, policy, mod, env_factory, backend):
        calls.append(item.key())
        return EpisodeResult(
            bench=item.bench, suite=item.suite, task_id=item.task_id,
            episode=item.episode, seed=item.seed, env_seed=item.env_seed,
            success=(item.task_id + item.episode) % 2 == 0,
            steps=7, n_replans=2, task_name="fake",
            extra={"policy_seed": item.policy_seed},
        )

    monkeypatch.setattr(direct.runner, "_run_item", fake_run)
    out = tmp_path / "direct.json"
    result = direct.run_direct_validation(
        checkpoint_path="candidate.pt", validation_collection="validation",
        out=out, workers=1, quiet=True,
    )
    assert len(calls) == 400
    assert result["summary"] == {
        **result["summary"], "n_episodes": 400, "n_expected": 400,
        "n_errors": 0, "complete": True,
    }
    assert result["meta"]["direct_validation"]["status"] == "PASS"
    assert result["meta"]["paired_outcomes"]["aggregate"]["n"] == 400
    assert all("outcome_recovery_parent_arm0" in row["extra"]
               for row in result["episodes"])

    # Same authenticated identity skips all 400 persisted conditions, retains
    # real-policy provenance, recomputes pairing, and remains terminal PASS.
    result2 = direct.run_direct_validation(
        checkpoint_path="candidate.pt", validation_collection="validation",
        out=out, workers=1, quiet=True,
    )
    assert len(calls) == 400
    assert result2["summary"]["complete"] is True
    assert result2["meta"]["policy"]["is_stub"] is False
    assert collection.unchanged_calls == 2


def test_resume_rejects_source_identity_change(tmp_path, monkeypatch):
    collection = FakeCollection()
    checkpoint = _checkpoint(collection)
    source = {"scheme": "test", "sha256": "8" * 64, "files": {}}
    monkeypatch.setattr(direct, "_source_identity", lambda: copy.deepcopy(source))
    monkeypatch.setattr(direct, "authenticate_descendant_checkpoint",
                        lambda path: checkpoint)
    monkeypatch.setattr(direct, "_open_validation_collection",
                        lambda path, parent: collection)
    monkeypatch.setattr(direct, "_assert_checkpoint_unchanged", lambda identity: None)
    monkeypatch.setattr(direct.runner, "bench_module", lambda bench: object())
    monkeypatch.setattr(direct.runner, "env_available", lambda mod: True)
    monkeypatch.setattr(direct.runner, "ensure_runtime", lambda mod: None)
    monkeypatch.setattr(direct.runner, "_default_policy", lambda ckpt, kw: object())
    monkeypatch.setattr(direct.runner, "_provenance",
                        lambda policy: _policy_provenance(checkpoint))
    monkeypatch.setattr(direct.runner, "_run_item", lambda item, *args: EpisodeResult(
        bench=item.bench, suite=item.suite, task_id=item.task_id,
        episode=item.episode, seed=item.seed, env_seed=item.env_seed,
        success=False, steps=1, n_replans=1, task_name="fake",
        extra={"policy_seed": item.policy_seed},
    ))
    out = tmp_path / "direct.json"
    direct.run_direct_validation(
        checkpoint_path="candidate.pt", validation_collection="validation",
        out=out, workers=1, quiet=True,
    )
    source["sha256"] = "9" * 64
    with pytest.raises(ValueError, match="different evaluation identity"):
        direct.run_direct_validation(
            checkpoint_path="candidate.pt", validation_collection="validation",
            out=out, workers=1, quiet=True,
        )


def test_paired_summary_is_exact():
    rows = []
    outcomes = [(0, 0), (1, 0), (0, 1), (1, 1)]
    for index, (candidate, parent) in enumerate(outcomes):
        rows.append(EpisodeResult(
            bench="libero", suite="libero_spatial", task_id=index,
            episode=40, seed=0, env_seed=index, success=bool(candidate), steps=1,
            extra={"outcome_recovery_parent_arm0": {"success": bool(parent)}},
        ))
    # Supply empty rows for the other suites so the production protocol-order
    # output remains defined without weakening exact aggregate arithmetic.
    for suite in direct.DEFAULT_LIBERO_SUITES[1:]:
        rows.append(EpisodeResult(
            bench="libero", suite=suite, task_id=0, episode=40, seed=0,
            env_seed=0, success=False, steps=1,
            extra={"outcome_recovery_parent_arm0": {"success": False}},
        ))
    summary = direct.paired_summary(rows)
    assert summary["aggregate"] == {
        "n": 7, "candidate": 2, "parent_arm0": 2, "both": 1,
        "candidate_only": 1, "parent_only": 1, "neither": 4,
    }
    assert summary["per_suite"]["libero_spatial"]["n"] == 4
