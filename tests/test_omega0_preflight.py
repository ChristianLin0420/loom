"""Focused CPU tests for the reserve omega-zero pre-update gate."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import contracts as C
from scripts import bank_ca_gate
from scripts import omega0_preflight as gate


ROOT = Path(__file__).parents[1]


class TinyBank(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.omega = nn.Parameter(torch.tensor([[1.0, -2.0], [3.0, -4.0]]))
        self.log_r = nn.Parameter(torch.tensor([0.25, -0.5]))
        self.b_raw = nn.Parameter(torch.tensor([0.75, -1.0]))


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.estimator = nn.Linear(2, 2)
        self.bank = TinyBank()
        self.proposal = nn.Linear(2, 3)
        self.decoder = nn.ModuleDict({"libero_franka": nn.Linear(3, 2)})
        self.register_buffer("integer_witness", torch.tensor(7))


def _source_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def test_current_target_config_is_exactly_authenticated():
    cfg, provenance = gate.authenticate_target_config(
        ROOT / "configs" / "r0a_bank_ca_qa_omega0.yaml"
    )
    assert provenance["config_hash"] == "7a5e8a24327ecc0c"
    assert provenance["method_variant"] == "joint_q_action_bank_identity_centered"
    assert provenance["transition_parameter_reset"] == gate.DEFAULT_PINS.transition_reset
    assert cfg["train_modules"] == ["bank", "q_action"]


def test_target_config_authentication_fails_closed_on_drift(tmp_path, monkeypatch):
    cfg = bank_ca_gate._read_config(ROOT / "configs" / "r0a_bank_ca_qa_omega0.yaml")
    cfg["optim"]["transition_parameter_reset"]["tensors"] = {"bank.omega": "one"}
    path = tmp_path / "candidate.yaml"
    path.write_text("placeholder: true\n")
    monkeypatch.setattr(bank_ca_gate, "_read_config", lambda unused: cfg)
    with pytest.raises(gate.PreflightError, match="target config hash"):
        gate.authenticate_target_config(path)


def test_omega_reset_changes_only_one_tensor_and_preserves_direct_policy():
    model = TinyModel()
    source = _source_state(model)
    result = gate.apply_omega_zero_and_audit(
        model, source, reset_recipe=gate.DEFAULT_PINS.transition_reset,
    )
    assert result["changed_tensors"] == ["bank.omega"]
    assert result["reset_omega_bit_exact_zero"] is True
    assert result["all_other_tensors_exact"] is True
    assert result["source_non_target_sha256"] == result["reset_non_target_sha256"]
    assert torch.count_nonzero(model.bank.omega) == 0
    assert torch.equal(model.bank.log_r, source["bank.log_r"])
    assert torch.equal(model.bank.b_raw, source["bank.b_raw"])
    for name in gate.DIRECT_POLICY_MODULES:
        witness = result["direct_policy"][name]
        assert witness["tensor_exact"] is True
        assert witness["source_sha256"] == witness["reset_sha256"]


@pytest.mark.parametrize("failure", ["preload_drift", "nonfinite", "wrong_recipe"])
def test_omega_reset_audit_fails_closed(failure):
    model = TinyModel()
    source = _source_state(model)
    recipe = copy.deepcopy(gate.DEFAULT_PINS.transition_reset)
    if failure == "preload_drift":
        with torch.no_grad():
            model.estimator.weight.add_(1.0)
        match = "not checkpoint-exact"
    elif failure == "nonfinite":
        source["estimator.weight"][0, 0] = float("nan")
        match = "non-finite"
    else:
        recipe["tensors"] = {"bank.omega": "one"}
        match = "only the pinned"
    with pytest.raises(gate.PreflightError, match=match):
        gate.apply_omega_zero_and_audit(model, source, reset_recipe=recipe)


def test_tensor_digest_supports_scalar_and_bfloat16_and_is_bit_sensitive():
    scalar = torch.tensor(1.0)
    assert gate.tensor_sha256(scalar) == gate.tensor_sha256(scalar.clone())
    assert gate.tensor_sha256(scalar) != gate.tensor_sha256(torch.tensor(2.0))
    value = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    assert len(gate.tensor_sha256(value)) == 64
    state = {"scalar": torch.tensor(7), "bf16": value}
    assert len(gate.state_sha256(state, state)) == 64


def _checkpoint_fixture():
    source_cfg = {"run": {"name": "deploy"}, "model": {"width": 2}}
    source_hash = bank_ca_gate._experiment_config_hash(source_cfg)
    reset = {
        "source_config_hash": source_hash,
        "tensors": {"bank.omega": "zero"},
    }
    pins = replace(
        gate.DEFAULT_PINS,
        deploy_checkpoint_sha256="a" * 64,
        deploy_config_hash=source_hash,
        deploy_global_step=11,
        transition_reset=reset,
    )
    payload = {
        "config_hash": source_hash,
        "global_step": 11,
        "world_size": 16,
        "samples_seen": 100,
        "git_sha": "git",
        "consolidated": {
            "tool": "loom.train.consolidate",
            "section": "model",
            "step": 11,
            "n_shards": 16,
            "n_keys": 1,
        },
        "model": {"bank.omega": torch.ones(2)},
        "resolved_config": source_cfg,
    }
    target = {
        "model": {"width": 2},
        "optim": {"transition_parameter_reset": copy.deepcopy(reset)},
    }
    return payload, target, pins


def test_deploy_payload_authenticates_consolidation_config_and_model_identity():
    payload, target, pins = _checkpoint_fixture()
    result = gate.authenticate_deploy_payload(
        payload, checkpoint_sha256="a" * 64, target_config=target, pins=pins,
    )
    assert result["global_step"] == 11
    assert result["config_hash"] == pins.deploy_config_hash
    assert result["model_tensors"] == 1


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("sha", "SHA-256"),
        ("step", "exact step"),
        ("shards", "consolidation provenance"),
        ("model", "model construction differs"),
        ("reset", "reset recipe changed"),
    ],
)
def test_deploy_payload_authentication_rejects_mutation(mutation, match):
    payload, target, pins = _checkpoint_fixture()
    sha = "a" * 64
    if mutation == "sha":
        sha = "b" * 64
    elif mutation == "step":
        payload["global_step"] = 12
    elif mutation == "shards":
        payload["consolidated"]["n_shards"] = 15
    elif mutation == "model":
        target["model"]["width"] = 3
    else:
        target["optim"]["transition_parameter_reset"]["tensors"] = {
            "bank.omega": "one"
        }
    with pytest.raises(gate.PreflightError, match=match):
        gate.authenticate_deploy_payload(
            payload, checkpoint_sha256=sha, target_config=target, pins=pins,
        )


class RollBank:
    def step(self, coefficient, belief):  # noqa: ARG002
        return torch.roll(belief, shifts=1, dims=-1)


class FixedQAction:
    def __call__(self, actions, belief):  # noqa: ARG002
        out = torch.zeros(actions.shape[0], C.M)
        out[:, :C.TOPK] = 1.0 / C.TOPK
        return out


class MetricModel:
    def __init__(self, base):
        self.bank = RollBank()
        self.q_action = {"libero_franka": FixedQAction()}
        self._base = base

    def beliefs(self, batch):
        return [self._base.clone() for _ in range(C.DEPTH + 1)]

    def target_beliefs(self, batch):
        out = [self._base.clone()]
        value = self._base
        for _ in range(C.DEPTH):
            value = torch.roll(value, shifts=1, dims=-1)
            out.append(value.clone())
        return out


def test_synthetic_action_labelled_sequential_rollout_metric():
    batch_size = 2
    base = torch.arange(C.D, dtype=torch.float32).view(1, 1, C.D)
    base = base.expand(batch_size, C.K, C.D).clone()
    batch = {
        "actions": torch.zeros(batch_size, C.DEPTH, C.H_OP, 7),
        "burn_in_feats": [None] * 4,
        "feats": [None] * (C.DEPTH + 1),
        "embodiment": "libero_franka",
    }
    identity, rollout, coefficients = gate.measure_error_batch(
        MetricModel(base), batch,
    )
    assert identity.shape == rollout.shape == (batch_size, C.DEPTH)
    assert coefficients.shape == (batch_size, C.DEPTH, C.M)
    assert torch.all(rollout.abs() < 1e-6)
    assert torch.all(identity > rollout)


def test_strict_per_horizon_summary_and_fail_closed_numbers():
    identity = np.full((16, C.DEPTH), 0.2)
    rollout = np.full((16, C.DEPTH), 0.1)
    passed = gate.summarize_errors(identity, rollout)
    assert passed["status"] == "PASS"
    assert passed["n_rows"] == 16
    assert all(row["passed"] for row in passed["horizons"])

    rollout[:, 2] = identity[:, 2]
    failed = gate.summarize_errors(identity, rollout)
    assert failed["status"] == "FAIL"
    assert failed["horizons"][2]["passed"] is False
    assert "h3" in failed["failures"][0]

    identity[0, 0] = np.nan
    with pytest.raises(gate.PreflightError, match="non-finite"):
        gate.summarize_errors(identity, rollout)
    with pytest.raises(gate.PreflightError, match="shape"):
        gate.summarize_errors(np.ones((15, 4)), np.ones((15, 4)))


def _fake_loader(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "manifest.json").write_text('{"cache":"fixed"}\n')
    body = {
        "version": 1,
        "source": "libero",
        "split": "gate",
        "holdout_demo_keys": ["demo_49"],
        "n_tasks": 2,
        "n_trajectories": 2,
        "tasks": {"task/a": ["task/a/demo_49"], "task/b": ["task/b/demo_49"]},
        "trajectory_ids": ["task/a/demo_49", "task/b/demo_49"],
    }
    manifest = {
        **body,
        "digest": "sha256:" + bank_ca_gate._canonical_json_sha256(body),
    }
    windows = [
        SimpleNamespace(
            traj_id="task/a/demo_49", start=40,
            obs_src_index=(1, 2, 3, 4, 5), action_free=False,
        ),
        SimpleNamespace(
            traj_id="task/b/demo_49", start=48,
            obs_src_index=(6, 7, 8, 9, 10), action_free=False,
        ),
    ]
    dataset = SimpleNamespace(
        cache=SimpleNamespace(root=cache), recurrent_burn_in=4, windows=windows,
    )

    class Sampler:
        sampling = "uniform_task"

        @staticmethod
        def batch_at(step, rank):  # noqa: ARG004
            return "libero_franka", np.asarray([1, 0])

    loader = SimpleNamespace(
        datasets={"libero_franka": dataset},
        trajectory_manifest=lambda: copy.deepcopy(manifest),
        n_windows=2,
        sampling="uniform_task",
        sampler=Sampler(),
        batch_size=2,
        num_workers=0,
    )
    records = bank_ca_gate._selected_window_records(
        loader, [1, 0], {"task/a/demo_49": "task/a", "task/b/demo_49": "task/b"},
    )
    pins = replace(
        gate.DEFAULT_PINS,
        manifest_digest=manifest["digest"],
        cache_manifest_sha256=gate.sha256_file(cache / "manifest.json"),
        loader_n_windows=2,
        windows=2,
        batch_size=2,
        selected_indices=(1, 0),
        selected_records_sha256=gate._canonical_json_sha256(records),
    )
    return loader, pins


def test_fixed_selection_authenticates_manifest_cache_indices_and_rows(tmp_path):
    loader, pins = _fake_loader(tmp_path)
    dataset, records, source = gate.authenticate_selection(loader, pins=pins)
    assert dataset is loader.datasets["libero_franka"]
    assert [row["dataset_index"] for row in records] == [1, 0]
    assert source["fixed_selection"]["n_distinct_tasks"] == 2
    with pytest.raises(gate.PreflightError, match="indices changed"):
        gate.authenticate_selection(
            loader, pins=replace(pins, selected_indices=(0, 1)),
        )


def test_behavior_source_digest_detects_content_change_and_missing_file(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "a.py").write_text("a = 1\n")
    (tmp_path / "nested" / "b.py").write_text("b = 1\n")
    files = ("a.py", "nested/b.py")
    source = gate.behavior_source_provenance(tmp_path, reversed(files))
    gate.assert_behavior_source_digest(
        source["behavior_source_digest"], root=tmp_path, files=files,
    )
    (tmp_path / "nested" / "b.py").write_text("b = 2\n")
    with pytest.raises(gate.PreflightError, match="changed during"):
        gate.assert_behavior_source_digest(
            source["behavior_source_digest"], root=tmp_path, files=files,
        )
    (tmp_path / "nested" / "b.py").unlink()
    with pytest.raises(gate.PreflightError, match="missing"):
        gate.behavior_source_provenance(tmp_path, files)


def test_behavior_source_closure_includes_ema_target_and_metric_path():
    required = {
        "contracts.py",
        "loom/data/loader.py",
        "loom/heads/q_action.py",
        "loom/losses/dyn.py",
        "loom/model/bank.py",
        "loom/model/estimator.py",
        "loom/train/loop.py",
        "loom/train/schedule.py",
        "scripts/bank_ca_gate.py",
        "scripts/omega0_preflight.py",
    }
    assert required <= set(gate.BEHAVIOR_SOURCE_FILES)
    source = gate.behavior_source_provenance()
    assert source["behavior_source_digest"] == gate._behavior_digest_from_entries(
        source["behavior_source_files"]
    )


def test_atomic_publish_is_exclusive_under_a_race(tmp_path):
    target = tmp_path / "gate.json"
    barrier = threading.Barrier(2)

    def publish(worker):
        barrier.wait()
        try:
            digest = gate.atomic_publish_json(target, {"worker": worker})
            return "ok", worker, digest
        except gate.PreflightError as exc:
            return "blocked", worker, str(exc)

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(publish, (1, 2)))
    assert [row[0] for row in outcomes].count("ok") == 1
    assert [row[0] for row in outcomes].count("blocked") == 1
    stored = json.loads(target.read_text())
    assert stored["worker"] in (1, 2)
    assert not list(tmp_path.glob(".*.tmp"))
    with pytest.raises(gate.PreflightError, match="refusing to overwrite"):
        gate.atomic_publish_json(target, {"worker": 3})
    assert json.loads(target.read_text()) == stored


def test_direct_script_help_works_from_repository_root():
    result = subprocess.run(
        [sys.executable, "scripts/omega0_preflight.py", "--help"],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--checkpoint" in result.stdout
    assert "--out" in result.stdout
