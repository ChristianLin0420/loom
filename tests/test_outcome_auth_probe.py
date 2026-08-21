from __future__ import annotations

import torch
import pytest

from scripts import outcome_auth_probe as probe


def _row(*, arm: int, replan: int, logits, order, coeff, stored_coeff,
         old: float, stored_old: float):
    return {
        "arm": arm, "replan": replan,
        "logits": torch.tensor(logits, dtype=torch.float32),
        "order": torch.tensor(order, dtype=torch.int64),
        "coeff": torch.tensor(coeff, dtype=torch.float32),
        "stored_coeff": torch.tensor(stored_coeff, dtype=torch.float32),
        "old_logprob": old, "stored_old_logprob": stored_old,
    }


def test_mode_matrix_is_narrow_and_contains_required_replay_comparisons():
    modes = probe.replay_modes()
    assert modes[0] == probe.ReplayMode(
        "collector_b1_fp32", 1, "single", False, False, None,
    )
    assert any(mode.batch_size == 32 and mode.autocast_dtype is None for mode in modes)
    assert {(mode.allow_tf32, mode.autocast_dtype) for mode in modes} >= {
        (False, None), (True, None), (False, "bfloat16"),
    }
    assert all(mode.batch_size in (1, 32) for mode in modes)


def test_summary_reports_per_replan_coeff_and_sampled_only_logprob_errors():
    rows = [
        _row(
            arm=0, replan=0,
            logits=[4.0, 3.0, 2.0, 1.0], order=[0, 1, 2, 3],
            coeff=[0.4, 0.3, 0.2, 0.1],
            stored_coeff=[0.4, 0.3, 0.2, 0.1],
            old=99.0, stored_old=-99.0,
        ),
        _row(
            arm=1, replan=0,
            logits=[4.0, 3.0, 2.0, 1.0], order=[0, 1, 2, 3],
            coeff=[0.4, 0.3, 0.2, 0.1],
            stored_coeff=[0.39, 0.31, 0.2, 0.1],
            old=-5.0, stored_old=-5.25,
        ),
    ]
    report = probe.summarize_replay(rows, reference_logits=None)
    coeff = report["coeff_abs_error_per_replan_linf"]
    old = report["sampled_old_logprob_abs_error"]
    assert coeff["n"] == 2
    assert coeff["max"] == pytest.approx(0.01, abs=3e-8)
    assert old == {"n": 1, "max": 0.25, "p95_linear": 0.25, "mean": 0.25}
    assert report["worst_coeff"]["arm"] == 1
    assert report["worst_sampled_old_logprob"]["arm"] == 1
    # Arm-zero diagnostic old-logprob is deliberately excluded.
    assert old["max"] != 198.0


def test_summary_support_identity_detects_order_and_set_changes():
    row = _row(
        arm=0, replan=0,
        logits=[4.0, 3.0, 2.0, 1.0, 0.0], order=[0, 1, 2, 3],
        coeff=[0.4, 0.3, 0.2, 0.1, 0.0],
        stored_coeff=[0.4, 0.3, 0.2, 0.1, 0.0],
        old=-1.0, stored_old=-1.0,
    )
    sampled = {**row, "arm": 1}
    same = probe.summarize_replay(
        [row, sampled], reference_logits=[row["logits"], sampled["logits"]],
    )
    support = same["support_identity"]
    assert support["arm0_argmax_order_equals_stored"] == 1
    assert support["arm0_argmax_support_equals_stored"] == 1
    assert support["vs_collector_b1_argmax_order"] == 2

    changed_reference = torch.tensor([3.0, 4.0, 2.0, 1.0, 0.0])
    changed = probe.summarize_replay(
        [row, sampled], reference_logits=[changed_reference, sampled["logits"]],
    )
    assert changed["support_identity"]["vs_collector_b1_argmax_order"] == 1
    assert changed["support_identity"]["vs_collector_b1_argmax_support"] == 2


def test_fixed_probe_inputs_are_the_terminal_collection_receipt_identity():
    assert probe.CHECKPOINT_SHA256 == (
        "15f286c268caa5327d5aa3abf1f67ebd0555c426a509fef22cb7f537bf6ab4e1"
    )
    assert probe.SIDECAR_SHA256 == (
        "441004267ecca795e3a0e1ecdd7ec4efaa4a1503ee339e260694b046a956d961"
    )
    assert probe.GROUP_ID == "libero_spatial/task=00/trial=10/seed=0"
    assert sum(probe.EXPECTED_REPLANS) == 469
