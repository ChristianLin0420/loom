"""Integration contracts for the fresh dual-code formal lineage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import loom.train.loop as train_loop
from loom.train.direct_formal import DirectFormalGate, DirectFormalSchedule
from loom.train.loop import (
    _reconcile_direct_formal_metrics,
    config_hash,
    load_config,
    main,
    parse_args,
    read_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "r0a_dual_code_formal.yaml"


def test_formal_recipe_pins_method_lineage_schedule_and_online_logging() -> None:
    cfg = read_config(CONFIG)
    schedule = DirectFormalSchedule.from_config(cfg)
    gate = DirectFormalGate()

    assert cfg["run"] == {
        **cfg["run"],
        "name": "r0a_dualcode_fresh_s0_20260820",
        "project": "loom-r0-e2e-scratch",
        "seed": 0,
        "steps": 40_000,
        "schedule_horizon": 32_000,
        "max_updates": 40_000,
        "fresh_start_required": True,
        "require_online_wandb": True,
        "keep_last": 20,
    }
    assert schedule.schedule_horizon == gate.schedule_horizon == 32_000
    assert schedule.max_updates == gate.max_updates == 40_000
    assert cfg["model"]["use_stubs"] is False
    assert cfg["losses"]["act"]["decode_from"] == "dual_q_action_proposal"
    assert cfg["train_modules"] == [
        "estimator", "bank", "q_delta", "q_action", "decoder", "proposal",
    ]
    assert cfg["direct_formal"]["primary"] == list(gate.primary_metrics)
    assert cfg["direct_formal"]["no_convergence_by_hard_cap"] == (
        "ABORT_NO_EVALUATION"
    )
    assert len(config_hash(cfg)) == 16


def test_direct_formal_rejects_legacy_steps_override() -> None:
    args = parse_args(["--config", str(CONFIG), "--steps", "32000"])
    with pytest.raises(ValueError, match="cannot override a direct-formal config"):
        load_config(args)


def test_lr_floor_and_module_scales_are_exact_after_32k() -> None:
    cfg = read_config(CONFIG)
    schedule = DirectFormalSchedule.from_config(cfg)
    assert schedule.lr_at(32_000) == pytest.approx(1.5e-5)
    assert schedule.lr_at(39_999) == pytest.approx(1.5e-5)
    assert schedule.lr_at(40_000) == pytest.approx(1.5e-5)
    assert schedule.lr_at(32_000) * cfg["optim"]["bank_lr_mult"] == pytest.approx(
        1.5e-6
    )
    assert schedule.lr_at(32_000) * cfg["optim"]["lr_scales"]["proposal"] == (
        pytest.approx(4.5e-6)
    )


def _write_tiny_direct_config(path: Path, *, max_updates: int = 3) -> None:
    path.write_text(f"""
extends: base.yaml
run:
  name: tiny_direct
  steps: {max_updates}
  schedule_horizon: 2
  max_updates: {max_updates}
  fresh_start_required: true
  log_every: 1
  ckpt_every: 1
data:
  source: stub
  batch_per_gpu: 2
model:
  use_stubs: true
optim:
  warmup: 0
  spike_mult: 0
fsdp:
  shard: []
  replicate: []
""")


def test_tiny_direct_run_starts_fresh_resumes_exactly_and_pins_schedule(
    tmp_path: Path,
) -> None:
    config = tmp_path / "tiny.yaml"
    run_dir = tmp_path / "run"
    _write_tiny_direct_config(config)

    args = [
        "--config", str(config), "--run_dir", str(run_dir), "--no_wandb",
    ]
    assert main(args) == 0
    assert (run_dir / "LATEST").read_text().strip() == "3"
    payload = torch.load(
        run_dir / "ckpt_000000003_rank0.pt", map_location="cpu",
        weights_only=False,
    )
    assert payload["scheduler"] == {
        "format": "loom-direct-formal-schedule-v1",
        "base_lr": 3.0e-4,
        "warmup_steps": 0,
        "schedule_horizon": 2,
        "max_updates": 3,
        "min_lr_ratio": 0.05,
    }
    assert payload["fresh_lineage"]["config_hash"] == config_hash(
        read_config(config)
    )
    assert payload["fresh_lineage"]["fresh_start_required"] is True
    assert main(args) == 0  # exact same-lineage resume is a no-op success

    _write_tiny_direct_config(config, max_updates=4)
    with pytest.raises(RuntimeError, match="resume config mismatch before load"):
        main(args)


def test_fresh_lineage_rejects_stale_uncheckpointed_metrics(tmp_path: Path) -> None:
    config = tmp_path / "tiny.yaml"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.jsonl").write_text("{}\n")
    _write_tiny_direct_config(config)
    with pytest.raises(RuntimeError, match="refuses prior training state"):
        main([
            "--config", str(config), "--run_dir", str(run_dir), "--no_wandb",
        ])


def _metrics_bytes(count: int) -> bytes:
    return b"".join(
        (json.dumps({"global_step": step, "loss": float(step)}) + "\n").encode()
        for step in range(1, count + 1)
    )


def test_formal_metrics_crash_tail_is_content_addressed_and_rolled_back(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    original = _metrics_bytes(5)
    (run_dir / "metrics.jsonl").write_bytes(original)
    identity = {
        "format": "loom-direct-formal-checkpoint-identity-v1",
        "latest_step": 3,
        "payload_global_step": 3,
        "config_hash": "0123456789abcdef",
        "world_size": 1,
    }

    result = _reconcile_direct_formal_metrics(
        run_dir, checkpoint_step=3, checkpoint_identity=identity,
    )
    assert result["action"] == "ROLLBACK"
    assert (run_dir / "metrics.jsonl").read_bytes() == _metrics_bytes(3)
    full_path = run_dir / result["quarantine"]["full_original"]
    tail_path = run_dir / result["quarantine"]["discarded_tail"]
    receipt_path = Path(result["receipt_path"])
    assert full_path.read_bytes() == original
    assert tail_path.read_bytes() == _metrics_bytes(5)[len(_metrics_bytes(3)):]
    assert result["ledger"]["discarded_step_range"] == [4, 5]
    assert result["ledger"]["original_sha256"] in full_path.name
    assert result["ledger"]["discarded_sha256"] in tail_path.name
    assert json.loads(receipt_path.read_text())["checkpoint"] == identity

    # An interruption after publishing immutable artifacts but before replacing
    # the ledger replays byte-identically and completes the same rollback.
    (run_dir / "metrics.jsonl").write_bytes(original)
    replay = _reconcile_direct_formal_metrics(
        run_dir, checkpoint_step=3, checkpoint_identity=identity,
    )
    assert replay == result
    assert (run_dir / "metrics.jsonl").read_bytes() == _metrics_bytes(3)
    assert len(list((run_dir / "direct_formal_metrics_rollback").iterdir())) == 3


@pytest.mark.parametrize(
    "payload,checkpoint_step,error",
    [
        (_metrics_bytes(2), 3, "behind checkpoint"),
        (b'{"global_step":1}', 1, "unterminated final row"),
        (
            b'{"global_step":1}\n{"global_step":1}\n',
            1,
            "not exactly contiguous",
        ),
        (
            b'{"global_step":1}\n{"global_step":3}\n',
            1,
            "not exactly contiguous",
        ),
        (b'{"global_step":1}\nnot-json\n', 1, "is malformed"),
        (b'{"global_step":1}\n{"global_step":NaN}\n', 1, "is malformed"),
    ],
)
def test_formal_metrics_reconciliation_fails_closed_on_ambiguous_ledgers(
    tmp_path: Path, payload: bytes, checkpoint_step: int, error: str,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.jsonl").write_bytes(payload)
    with pytest.raises(RuntimeError, match=error):
        _reconcile_direct_formal_metrics(
            run_dir,
            checkpoint_step=checkpoint_step,
            checkpoint_identity={"latest_step": checkpoint_step},
        )


def test_formal_metrics_reconciliation_rejects_missing_committed_prefix(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="ledger is missing behind checkpoint"):
        _reconcile_direct_formal_metrics(
            tmp_path,
            checkpoint_step=1,
            checkpoint_identity={"latest_step": 1},
        )


@pytest.mark.parametrize("status,expected_exit", [("PASS", 0), ("ABORT", 3), ("INVALID", 2)])
def test_restored_terminal_boundary_replays_before_any_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str, expected_exit: int,
) -> None:
    config = tmp_path / "tiny.yaml"
    run_dir = tmp_path / "run"
    _write_tiny_direct_config(config)
    args = ["--config", str(config), "--run_dir", str(run_dir), "--no_wandb"]
    assert main([*args, "--stop_at", "2"]) == 0

    events: list[tuple[str, int]] = []

    def fake_evaluate(rows, *, current_step, gate=None):
        del rows, gate
        events.append(("decision", current_step))
        return {
            "status": status,
            "reason": "test_boundary",
            "current_step": current_step,
            "decision_step": current_step,
        }

    original_apply = DirectFormalSchedule.apply

    def recording_apply(self, optimizer, step):
        events.append(("update", step))
        return original_apply(self, optimizer, step)

    monkeypatch.setattr(train_loop, "evaluate_direct_formal", fake_evaluate)
    monkeypatch.setattr(
        train_loop, "should_evaluate_direct_formal", lambda step: step == 2,
    )
    monkeypatch.setattr(DirectFormalSchedule, "apply", recording_apply)

    assert main(args) == expected_exit
    assert events == [("decision", 2)]
    assert (run_dir / "LATEST").read_text().strip() == "2"
    assert [json.loads(line)["global_step"] for line in (
        run_dir / "metrics.jsonl"
    ).read_text().splitlines()] == [1, 2]


def test_resume_rolls_back_future_rows_then_replays_moving_boundary_before_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "tiny.yaml"
    run_dir = tmp_path / "run"
    _write_tiny_direct_config(config)
    args = ["--config", str(config), "--run_dir", str(run_dir), "--no_wandb"]
    assert main([*args, "--stop_at", "2"]) == 0

    rows = [json.loads(line) for line in (
        run_dir / "metrics.jsonl"
    ).read_text().splitlines()]
    for future_step in (3, 4):
        future = dict(rows[-1], global_step=future_step)
        rows.append(future)
    (run_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )

    events: list[tuple[str, int]] = []

    def fake_evaluate(metric_rows, *, current_step, gate=None):
        del gate
        events.append(("decision", current_step))
        assert [row["global_step"] for row in metric_rows] == [1, 2]
        return {
            "status": "MOVING",
            "reason": "test_boundary",
            "current_step": current_step,
            "decision_step": None,
        }

    original_apply = DirectFormalSchedule.apply

    def recording_apply(self, optimizer, step):
        events.append(("update", step))
        return original_apply(self, optimizer, step)

    monkeypatch.setattr(train_loop, "evaluate_direct_formal", fake_evaluate)
    monkeypatch.setattr(
        train_loop, "should_evaluate_direct_formal", lambda step: step == 2,
    )
    monkeypatch.setattr(DirectFormalSchedule, "apply", recording_apply)

    assert main(args) == 0
    assert events[:2] == [("decision", 2), ("update", 2)]
    final_steps = [json.loads(line)["global_step"] for line in (
        run_dir / "metrics.jsonl"
    ).read_text().splitlines()]
    assert final_steps == [1, 2, 3]
    assert len(list((run_dir / "direct_formal_metrics_rollback").glob(
        "rollback.*.json"
    ))) == 1
