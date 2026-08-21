"""Contract tests for the common-noise sixteen-arm operator oracle."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch
from torch import Tensor, nn

import contracts as C
from loom.eval import EvalProtocol
from loom.eval import operator_oracle as oracle
from loom.eval import policy as pol
from loom.eval import runner


class FixedEstimator(nn.Module):
    def forward(self, feats, z_prev=None):
        b = feats["proprio"].shape[0]
        return torch.zeros(b, C.K, C.D, device=feats["proprio"].device,
                           dtype=feats["proprio"].dtype)


class CommonProposal(nn.Module):
    def _coeff(self, idx: Tensor, dtype: torch.dtype) -> Tensor:
        c = torch.zeros(*idx.shape[:-1], C.M, device=idx.device, dtype=dtype)
        return c.scatter(-1, idx, 1.0 / C.TOPK)

    def argmax(self, z, lang):
        idx = torch.arange(C.TOPK, device=z.device).expand(z.shape[0], C.TOPK)
        return self._coeff(idx, z.dtype)

    def sample(self, z, lang, n, *, generator=None):
        noise = torch.rand(z.shape[0], n, C.M, generator=generator, device=z.device)
        return self._coeff(noise.topk(C.TOPK, dim=-1).indices, z.dtype)

    def logits(self, z, lang):
        return torch.arange(C.M, device=z.device, dtype=z.dtype).expand(z.shape[0], C.M)


class NoiseDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_c = None
        self.noises: list[Tensor] = []

    def forward(self, proprio, c, *, generator=None):
        self.last_c = c.detach().clone()
        noise = torch.randn(
            c.shape[0], C.H_OP, 7, generator=generator,
            device=c.device, dtype=c.dtype,
        )
        self.noises.append(noise.detach().clone())
        # Coefficient dependence makes a different operator a different action,
        # while the recorded tensor lets the common random numbers be checked.
        return 0.01 * noise + c[..., :7].unsqueeze(1)


class EchoQAction(nn.Module):
    def __init__(self, decoder: NoiseDecoder, *, reject_samples: bool = False):
        super().__init__()
        self.decoder = decoder
        self.reject_samples = reject_samples

    def forward(self, action, z):
        c = self.decoder.last_c
        if c is None:
            raise RuntimeError("decoder was not called")
        if self.reject_samples and bool((c[..., C.TOPK:] != 0).any()):
            return c.roll(C.M // 2, dims=-1)
        return c


def modules(*, reject_samples: bool = False):
    decoder = NoiseDecoder()
    return pol.PolicyModules(
        estimator=FixedEstimator(),
        proposal=CommonProposal(),
        decoder=decoder,
        q_action=EchoQAction(decoder, reject_samples=reject_samples),
        featurize=pol.zeros_featurizer(C.EMBODIMENTS["libero_franka"]),
        embodiment="libero_franka",
        device="cpu",
        meta={},
    )


OBS = {"state": np.zeros(7, dtype=np.float32)}


def test_oracle_contract_is_fixed_at_argmax_plus_fifteen_samples():
    s = oracle.OracleArmSpec(
        arm=15, expected_config_hash="hash", expected_global_step=7,
        checkpoint_sha256="c" * 64, source_digest="a" * 64,
    )
    assert s.n_candidates == 16 and s.tau == C.REALIZABILITY_TAU
    with pytest.raises(ValueError, match="fixed"):
        oracle.OracleArmSpec(
            arm=0, n_candidates=8, expected_config_hash="hash",
            expected_global_step=7, checkpoint_sha256="c" * 64,
            source_digest="a" * 64,
        )
    with pytest.raises(ValueError, match="unknown"):
        oracle.OracleArmSpec.from_mapping({**s.to_dict(), "temperature": 2.0})


def test_arm_zero_is_bit_exact_to_the_direct_policy():
    direct_modules = modules()
    oracle_modules = copy.deepcopy(direct_modules)
    direct = pol.LoomPolicy(direct_modules)
    arm0 = oracle.OperatorOraclePolicy(oracle_modules, arm=0)
    direct.set_policy_seed(123)
    arm0.set_policy_seed(123)
    direct.reset()
    arm0.reset()

    got_direct = direct._plan(OBS, "task")
    got_oracle = arm0._plan(OBS, "task")
    assert np.array_equal(got_direct, got_oracle)
    assert torch.equal(direct.last_coeff, arm0.last_coeff)
    assert torch.equal(direct_modules.decoder.noises[-1],
                       oracle_modules.decoder.noises[-1])
    assert torch.equal(direct._decoder_generator.get_state(),
                       arm0._decoder_generator.get_state())


def test_sample_arms_use_independent_sha_streams_and_common_decoder_noise():
    a_modules, b_modules = modules(), modules()
    arm1 = oracle.OperatorOraclePolicy(a_modules, arm=1)
    arm2 = oracle.OperatorOraclePolicy(b_modules, arm=2)
    for p in (arm1, arm2):
        p.set_policy_seed(987)
        p.reset()
    arm1._plan(OBS, "task")
    arm2._plan(OBS, "task")

    assert not torch.equal(arm1.last_coeff, arm2.last_coeff)
    assert torch.equal(a_modules.decoder.noises[-1], b_modules.decoder.noises[-1])
    assert oracle.proposal_seed(987, 1) != oracle.proposal_seed(987, 2)
    assert not torch.equal(arm1._proposal_generator.get_state(),
                           arm2._proposal_generator.get_state())
    assert torch.equal(arm1._decoder_generator.get_state(),
                       arm2._decoder_generator.get_state())

    arm1._plan(OBS, "task")
    arm2._plan(OBS, "task")
    assert torch.equal(a_modules.decoder.noises[-1], b_modules.decoder.noises[-1]), (
        "common decoder noise must remain aligned at every replan"
    )

    # Rebuilding the same arm reproduces its proposal draw exactly.
    clone_modules = modules()
    clone = oracle.OperatorOraclePolicy(clone_modules, arm=1)
    clone.set_policy_seed(987)
    clone.reset()
    clone._plan(OBS, "task")
    clone._plan(OBS, "task")
    assert torch.equal(arm1.last_coeff, clone.last_coeff)


def test_rejected_sample_falls_back_to_argmax_with_the_same_noise():
    d_modules = modules()
    r_modules = modules(reject_samples=True)
    direct = oracle.OperatorOraclePolicy(d_modules, arm=0)
    rejected = oracle.OperatorOraclePolicy(r_modules, arm=1)
    for p in (direct, rejected):
        p.set_policy_seed(44)
        p.reset()
    direct_action = direct._plan(OBS, "task")
    fallback_action = rejected._plan(OBS, "task")

    assert np.array_equal(direct_action, fallback_action)
    assert len(r_modules.decoder.noises) == 2, "sample probe + restored fallback"
    assert torch.equal(d_modules.decoder.noises[-1], r_modules.decoder.noises[-1])
    assert torch.equal(direct._decoder_generator.get_state(),
                       rejected._decoder_generator.get_state())
    stats = rejected.op_stats_summary()
    assert stats["oracle_n_rejected"] == 1
    assert stats["oracle_n_gate_exhausted"] == 0


def _arm_result(protocol: EvalProtocol, arm: int, successes: dict[tuple, bool]):
    spec = oracle.OracleArmSpec(
        arm=arm, expected_config_hash="test-hash", expected_global_step=11,
        checkpoint_sha256="c" * 64, source_digest="a" * 64,
    )
    rows = []
    for item in runner.iter_work(protocol):
        key = (item.suite, item.task_id, item.episode, item.seed)
        rows.append({
            "bench": "libero", "suite": item.suite, "task_id": item.task_id,
            "episode": item.episode, "seed": item.seed, "env_seed": item.env_seed,
            "success": bool(successes.get(key, False)), "steps": 2,
            "hit_step_cap": False, "task_name": "task", "n_replans": 1,
            "wall_s": 0.1, "error": None,
            "extra": {
                "policy_seed": item.policy_seed,
                "oracle_version": oracle.ORACLE_VERSION,
                "oracle_arm": arm,
                "oracle_mode": "direct_argmax" if arm == 0 else "proposal_sample",
                "oracle_n_candidates": oracle.ORACLE_CANDIDATES,
                "oracle_tau": C.REALIZABILITY_TAU,
                "oracle_proposal_seed_scheme": oracle.PROPOSAL_SEED_SCHEME,
                "oracle_common_decoder_noise": True,
                "oracle_n_replans": 1, "oracle_n_rejected": int(arm == 2),
                "oracle_n_gate_exhausted": 0,
                "oracle_requested_residual_mean": 0.1,
                "oracle_requested_residual_max": 0.1,
                "oracle_executed_residual_mean": 0.1,
                "oracle_executed_residual_max": 0.1,
            },
        })
    return {
        "protocol": protocol.to_dict(),
        "meta": {"policy": {
            "is_stub": False,
            "ckpt_config_hash": "test-hash",
            "ckpt_global_step": 11,
            "state_dict": {
                "q_action": {"tensors_loaded": 1, "unexpected": 0},
                "bank": {"tensors_present": 1, "loaded": False},
            },
            "operator_oracle": spec.to_dict(),
        }},
        "episodes": rows,
    }


def test_terminal_oracle_aggregation_is_paired_per_suite_and_fail_closed():
    protocol = EvalProtocol(
        episodes_per_task=2, n_tasks=1, suites=("libero_spatial",),
        seeds=(0,), max_steps=2, notes="test",
    )
    items = runner.iter_work(protocol)
    k0 = (items[0].suite, items[0].task_id, items[0].episode, items[0].seed)
    k1 = (items[1].suite, items[1].task_id, items[1].episode, items[1].seed)
    results = {
        arm: _arm_result(
            protocol, arm,
            ({k0: True} if arm == 0 else {k1: True} if arm == 3 else {}),
        )
        for arm in range(oracle.ORACLE_CANDIDATES)
    }
    out = oracle.aggregate_oracle_results(results, protocol)
    suite = out["per_suite"]["libero_spatial"]
    assert suite["direct_argmax_sr"] == 50.0
    assert suite["oracle_at_16_sr"] == 100.0
    assert suite["oracle_new_only"] == 1
    assert out["realizability"]["rejected_to_same_noise_argmax"] == 2
    assert not out["authoritative_protocol"] and not out["passes_90_every_suite"]

    mixed = copy.deepcopy(results)
    mixed[15]["meta"]["policy"]["operator_oracle"]["source_digest"] = "b" * 64
    with pytest.raises(ValueError, match="mixed operator-oracle provenance"):
        oracle.aggregate_oracle_results(mixed, protocol)

    missing_evidence = copy.deepcopy(results)
    del missing_evidence[1]["episodes"][0]["extra"]["oracle_common_decoder_noise"]
    with pytest.raises(ValueError, match="lacks oracle evidence"):
        oracle.aggregate_oracle_results(missing_evidence, protocol)

    del results[15]
    with pytest.raises(ValueError, match="needs arms"):
        oracle.aggregate_oracle_results(results, protocol)


def test_cli_dry_run_exercises_all_sixteen_arms_and_resumes(tmp_path):
    out = tmp_path / "oracle"
    argv = ["--dry-run", "--out-dir", str(out), "--quiet"]
    assert oracle.main(argv) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["dry_run"] is True
    assert summary["n_arm_episodes"] == 16
    assert summary["n_errors"] == 0 and summary["complete"]
    assert len(list(out.glob("arm_*.json"))) == 16
    # A completed run is read, identity-checked, and reused without rebuilding
    # policies; this guards the production resume path as well.
    assert oracle.main(argv) == 0
