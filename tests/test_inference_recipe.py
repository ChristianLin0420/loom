"""Inference-only recipe knobs: decoder averaging and gripper debounce."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import contracts as C
from loom.eval import policy as pol

GRIP = 6


class ScriptedDecoder:
    def __init__(self, signs) -> None:
        self.signs = [float(s) for s in signs]
        self.i = 0

    def forward(self, proprio, c):  # noqa: ARG002
        sign = self.signs[min(self.i, len(self.signs) - 1)]
        self.i += 1
        out = torch.zeros(proprio.shape[0], C.H_OP, 7)
        out[..., GRIP] = 0.8 * sign
        return out


def _executed_polarity(signs, dwell):
    modules = pol._stub_modules("libero_franka", "cpu")
    modules.decoder = ScriptedDecoder(signs)
    policy = pol.LoomPolicy(modules, n_candidates=2, gripper_dwell=dwell)
    out = []
    for _ in signs:
        policy._queue = []
        out.append(float(np.sign(policy.act({}, "task")[GRIP])))
    return out, policy.gripper_summary()


def test_gripper_dwell_one_is_the_original_pass_through():
    script = [1, 1, -1, 1, -1, -1]
    out, summary = _executed_polarity(script, 1)
    assert out == [float(x) for x in script]
    assert summary["grip_suppressed"] == 0
    assert summary["grip_exec_flips"] == summary["grip_prop_flips"]


def test_gripper_dwell_two_suppresses_one_replan_reversals():
    out, summary = _executed_polarity([1, 1, 1, -1, 1, 1], 2)
    assert out == [1.0] * 6
    assert summary["grip_prop_flips"] == 2
    assert summary["grip_exec_flips"] == 0
    assert summary["grip_suppressed"] == 1


def test_gripper_dwell_two_executes_a_persistent_reversal_one_replan_late():
    out, _ = _executed_polarity([1, 1, -1, -1, -1], 2)
    assert out == [1.0, 1.0, 1.0, -1.0, -1.0]


def test_gripper_gate_reflects_only_hold_channel_and_preserves_magnitude():
    modules = pol._stub_modules("libero_franka", "cpu")
    policy = pol.LoomPolicy(modules, n_candidates=2, gripper_dwell=2)
    policy.clock.next_segment_len()
    policy._gate_gripper(np.full((C.H_OP, 7), 0.37, dtype=np.float32))
    policy.clock.next_segment_len()
    seg = np.zeros((C.H_OP, 7), dtype=np.float32)
    seg[:, GRIP] = -0.37
    gated = policy._gate_gripper(seg)
    assert np.allclose(gated[:, GRIP], 0.37)
    assert np.array_equal(gated[:, :GRIP], seg[:, :GRIP])


class BatchedDecoder:
    def __init__(self) -> None:
        self.batch = None

    def forward(self, proprio, c, *, generator=None):  # noqa: ARG002
        self.batch = proprio.shape[0]
        value = torch.arange(self.batch, dtype=proprio.dtype).view(-1, 1, 1)
        return value.expand(self.batch, C.H_OP, 7)


def test_decoder_samples_are_batched_and_averaged():
    modules = pol._stub_modules("libero_franka", "cpu")
    decoder = BatchedDecoder()
    modules.decoder = decoder
    policy = pol.LoomPolicy(modules, n_candidates=2, decoder_samples=4)
    out = policy._plan({}, "task")
    assert decoder.batch == 4
    assert np.allclose(out, 1.5)


def test_inference_recipe_validation_and_provenance():
    modules = pol._stub_modules("libero_franka", "cpu")
    with pytest.raises(ValueError, match="gripper_dwell"):
        pol.LoomPolicy(modules, gripper_dwell=0)
    with pytest.raises(ValueError, match="decoder_samples"):
        pol.LoomPolicy(modules, decoder_samples=0)
    policy = pol.LoomPolicy(modules, gripper_dwell=2, decoder_samples=4)
    provenance = pol.policy_provenance(policy)
    assert provenance["gripper_dwell"] == 2
    assert provenance["decoder_samples"] == 4
