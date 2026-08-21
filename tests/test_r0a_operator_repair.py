"""Method-level contracts for the fresh fixed-endpoint operator repair."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import loom.train.loop as train_loop
from loom.train.loop import LoomModel, config_hash, read_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "r0a_operator_repair.yaml"


def test_operator_repair_recipe_preserves_loom_and_unconditionally_evaluates() -> None:
    cfg = read_config(CONFIG)

    assert cfg["run"]["project"] == "loom-r0-operator-repair"
    assert cfg["run"]["seed"] == 0
    assert cfg["run"]["steps"] == 32_000
    assert cfg["run"]["boundary_policy"] == "fixed_max_updates"
    assert "schedule_horizon" not in cfg["run"]
    assert "max_updates" not in cfg["run"]
    assert "direct_formal" not in cfg
    assert cfg["run"]["fresh_start_required"] is True
    assert cfg["run"]["require_online_wandb"] is True
    assert cfg["model"]["use_stubs"] is False
    assert cfg["optim"]["spike_mult"] == 0

    assert cfg["data"]["sampling"] == "weighted_suite_task"
    assert cfg["data"]["suite_weights"] == {
        "libero_spatial": 0.2,
        "libero_object": 0.2,
        "libero_goal": 0.2,
        "libero_10": 0.4,
    }
    assert cfg["data"]["recurrent_prefix_choices"] == [0, 4, 8, 12]
    assert cfg["losses"]["act"]["decode_from"] == "dual_q_action_proposal"
    assert cfg["losses"]["act"]["align_mode"] == "sparse_ce"
    assert cfg["losses"]["dyn"]["coeff_source"] == "q_action"
    assert cfg["losses"]["dyn"]["detach_coeff"] is False
    assert cfg["losses"]["dyn"]["effect_weight"] == 1.0
    assert cfg["losses"]["dyn"]["contrastive_weight"] == 0.25
    assert cfg["losses"]["balance"]["mode"] == "per_head"
    assert cfg["losses"]["balance"]["head_weights"] == {
        "q_delta": 0.75, "q_action": 0.25,
    }

    receipt = cfg["method_receipt"]
    assert receipt["fixed_endpoint_update"] == 32_000
    assert receipt["evaluation_is_unconditional"] is True
    assert receipt["evaluation_episodes"] == 1_200
    assert receipt["health_thresholds_control_execution"] is False
    assert cfg["slurm"]["n_links"] == 6
    assert len(config_hash(cfg)) == 16


def test_dynamics_formation_schedule_is_one_based_and_exact() -> None:
    model_view = SimpleNamespace(
        loss_cfg={"dyn": {"start_update": 2001, "ramp_updates": 500}}
    )
    scale = lambda step: LoomModel._loss_scale(model_view, "dyn", step)

    assert scale(0) == 0.0
    assert scale(1_999) == 0.0       # update 2,000 is still formation-only
    assert scale(2_000) == pytest.approx(1.0 / 500.0)
    assert scale(2_001) == pytest.approx(2.0 / 500.0)
    assert scale(2_498) == pytest.approx(499.0 / 500.0)
    assert scale(2_499) == 1.0
    assert scale(31_999) == 1.0


def test_every_configured_objective_has_a_method_role() -> None:
    cfg = read_config(CONFIG)
    enabled = {
        name for name, spec in cfg["losses"].items()
        if isinstance(spec, dict) and spec.get("enabled")
    }
    assert enabled == {"dyn", "act", "proposal", "balance"}
    assert cfg["train_modules"] == [
        "estimator", "bank", "q_delta", "q_action", "decoder", "proposal",
    ]


def test_threshold_free_training_fails_before_counting_nonfinite_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "tiny.yaml"
    config.write_text("""
extends: base.yaml
run: {name: nonfinite_operator_repair_test, steps: 1, log_every: 1, ckpt_every: 1}
data: {source: stub, batch_per_gpu: 2}
model: {use_stubs: true}
optim: {warmup: 0, spike_mult: 0}
fsdp: {shard: [], replicate: []}
""")
    monkeypatch.setattr(train_loop, "clip_grad", lambda *args, **kwargs: float("nan"))

    with pytest.raises(FloatingPointError, match="refusing to count or apply"):
        train_loop.main([
            "--config", str(config),
            "--run_dir", str(tmp_path / "run"),
            "--no_wandb",
        ])
    assert not (tmp_path / "run" / "LATEST").exists()
