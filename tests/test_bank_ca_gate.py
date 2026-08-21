"""Pure CPU tests for the deterministic action-anchored bank gate."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import contracts as C
from loom.data.loader import HomogeneousSampler
from loom.losses.dyn import ln_cosine_distance
from scripts import bank_ca_gate as gate


class TemplateBank:
    """A coefficient selects a fixed belief, making metric signs exact."""

    def __init__(self, templates: torch.Tensor) -> None:
        self.templates = templates

    def step(self, c: torch.Tensor, z: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        return torch.einsum("...m,mkd->...kd", c, self.templates.to(c))


def _template_problem(order: list[int]):
    """Build beliefs/coefficients for an explicit context+primary row order."""
    h, m, k, d = C.DEPTH, 4, 2, 4
    base = torch.tensor([
        [1.0, -1.0, 0.0, 0.0],
        [0.0, 1.0, -1.0, 0.0],
        [0.0, 0.0, 1.0, -1.0],
        [-1.0, 0.0, 0.0, 1.0],
    ])
    templates = base[:, None, :].expand(m, k, d).clone()
    bank = TemplateBank(templates)
    index = torch.tensor(order)
    c_seq = torch.eye(m)[index, None, :].expand(len(order), h, m).clone()
    z0 = torch.ones(len(order), k, d)
    target = templates[index]
    zs = [z0.clone() for _ in range(h + 1)]
    zts = [z0.clone()] + [target.clone() for _ in range(h)]
    return bank, zs, zts, c_seq


def _rows(value: float = 0.5, residual: np.ndarray | None = None) -> gate.MetricRows:
    task_ids: list[str] = []
    trajectory_ids: list[str] = []
    for task in range(4):
        for trajectory in range(2):
            cluster = f"task{task}/demo{trajectory}"
            for _ in range(3):
                task_ids.append(f"task{task}")
                trajectory_ids.append(cluster)
    n = len(task_ids)
    return gate.MetricRows(
        delta_sel=np.full((n, C.DEPTH), value, dtype=np.float64),
        identity_minus_rollout=np.full((n, C.DEPTH), 2 * value, dtype=np.float64),
        leaf_spread=np.full(n, value / 2, dtype=np.float64),
        task_ids=task_ids,
        trajectory_ids=trajectory_ids,
        cluster_ids=list(trajectory_ids),
        root_q_action_residual=residual,
        action_decode_improvement=(
            np.zeros(n, dtype=np.float64) if residual is not None else None
        ),
        proposal_support_overlap_change=(
            np.zeros(n, dtype=np.float64) if residual is not None else None
        ),
    )


def _requirements(value: float = 0.0) -> dict[str, float]:
    return {key: value for key in gate.REQUIREMENT_KEYS}


def _manifest() -> dict:
    body = {
        "version": 1,
        "source": "libero",
        "split": "gate",
        "holdout_demo_keys": ["demo_49"],
        "n_tasks": 2,
        "n_trajectories": 2,
        "tasks": {
            "suite/a": ["suite/a/demo_49"],
            "suite/b": ["suite/b/demo_49"],
        },
        "trajectory_ids": ["suite/a/demo_49", "suite/b/demo_49"],
    }
    return {**body, "digest": "sha256:" + gate._canonical_json_sha256(body)}


def _valid_gate_config() -> dict:
    disabled = {"enabled": False, "weight": 0.0}
    return {
        "run": {"name": "r0a_bank_ca", "steps": 80_000},
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
                "neg_weight": 1.0,
                "neg_margin": 0.1,
                "cosine": "per_slot",
            },
            "act": dict(disabled),
            "proposal": dict(disabled),
            "balance": dict(disabled),
            "potential": dict(disabled),
            "grpo": dict(disabled),
        },
        "train_modules": ["bank"],
        "offline_gate": {
            "script": "scripts/bank_ca_gate.py",
            "required": True,
            "direct_e2e": False,
            "confidence": 0.95,
            "bootstrap_samples": 2_000,
            "seed": 0,
            "windows": 256,
            "candidates": 32,
            "requirements": _requirements(),
        },
    }


def _valid_joint_gate_config(*, identity_centered: bool = False) -> dict:
    cfg = _valid_gate_config()
    cfg["run"]["name"] = "r0a_bank_ca_qa"
    cfg["optim"]["reset_state_modules"] = ["bank", "q_action"]
    cfg["optim"]["lr_scales"]["q_action"] = 1.0
    cfg["losses"]["dyn"]["detach_coeff"] = False
    cfg["losses"]["dyn"]["neg_weight"] = 4.0
    cfg["losses"]["act"] = {
        "enabled": True,
        "weight": 1.0,
        "align_to": "q_a",
        "decode_from": "q_action",
    }
    cfg["train_modules"] = ["bank", "q_action"]
    cfg["convergence"] = {
        "start_step": 49_666,
        "block": 2_000,
        "blocks": 4,
        "tol": 0.02,
        "primary": ["loss/dyn", "act/decode"],
        "watch": [
            "dyn/pos", "dyn/neg", "delta_op", "delta_sel/h1", "delta_sel/h2",
            "delta_sel/h3", "delta_sel/h4", "act/align", "act/c_a_spread",
            "grad_norm",
        ],
        "floor_checks": ["delta_sel"],
    }
    cfg["efficacy_gate"] = {
        "metric": "act/decode",
        "reference": "first_post_start_block",
        "comparison": "final_convergence_block",
        "max_relative_worsening": 0.0,
        "required": True,
    }
    cfg["liveness_gate"] = {
        "start_exclusive": 50_666,
        "end_inclusive": 52_666,
        "rows": 2_000,
        "requirements": {
            "delta_op_median_strict_gt": 0.01,
            "gnorm_bank_median_strict_gt": 1.0e-4,
            "gnorm_q_action_median_strict_gt": 1.0e-4,
            "skipped_rate_strict_lt": 0.01,
            "unexpected_module_gradients": False,
            "nonfinite": False,
        },
        "required": True,
    }
    cfg["offline_gate"]["preservation"] = {
        "reference_checkpoint_sha256":
            "15f286c268caa5327d5aa3abf1f67ebd0555c426a509fef22cb7f537bf6ab4e1",
        "reference_config_hash": "a199324a6205bb6d",
        "reference_global_step": 49_666,
        "action_decode_improvement_ci_low": 0.0,
        "proposal_support_overlap_change_ci_low": -0.05,
        "q_action_residual_max": 0.5,
        "max_root_exhaustion_rate": 0.01,
    }
    if identity_centered:
        cfg["run"]["name"] = "r0a_bank_ca_qa_omega0"
        cfg["optim"]["transition_parameter_reset"] = deepcopy(
            gate.IDENTITY_CENTERED_RESET
        )
    return cfg


def _set_path(cfg: dict, path: str, value) -> None:
    node = cfg
    parts = path.split(".")
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


def test_cyclic_context_chunks_preserve_order_for_all_runtime_batch_sizes():
    selected = list(range(256))
    for batch_size in (1, 7, 8, 256):
        rebuilt: list[int] = []
        for lo, primaries, context in gate.cyclic_context_chunks(selected, batch_size):
            rebuilt.extend(primaries)
            assert context[0] == selected[(lo - 1) % len(selected)]
            assert context[1:] == primaries
        assert rebuilt == selected


def test_first_uniform_task_rows_are_unique_and_batch_size_independent():
    size = 800
    pools = {
        f"task-{task:02d}": np.arange(task, size, 40, dtype=np.int64)
        for task in range(40)
    }
    orders = []
    for batch_size in (1, 7, 8, 256):
        sampler = HomogeneousSampler(
            {"libero_franka": size},
            batch_size=batch_size,
            seed=0,
            sampling="uniform_task",
            task_indices={"libero_franka": pools},
        )
        order = gate._first_sampler_indices(sampler, "libero_franka", 256)
        assert len(np.unique(order)) == 256
        orders.append(order)
    for order in orders[1:]:
        np.testing.assert_array_equal(order, orders[0])


def test_collector_uses_fixed_primary_order_and_context_only_rows(tmp_path, monkeypatch):
    size, batch_size = 320, 7
    manifest = _manifest()
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    (cache_root / "manifest.json").write_text('{"synthetic":true}\n')
    trajectories = ("suite/a/demo_49", "suite/b/demo_49")
    tasks = ("suite/a", "suite/b")

    class Dataset:
        recurrent_burn_in = 4
        cache = SimpleNamespace(root=cache_root)
        windows = [
            SimpleNamespace(
                traj_id=trajectories[i % 2],
                start=32 + 2 * (i // 2),
                action_free=False,
                obs_src_index=(0, 1, 2, 3, 4),
            )
            for i in range(size)
        ]

        def __len__(self):
            return size

        def __getitem__(self, i):
            feature = {
                "views": torch.zeros(1),
                "proprio": torch.zeros(1),
                "lang": torch.zeros(1, 1),
            }
            return {
                "feats": [deepcopy(feature) for _ in range(C.DEPTH + 1)],
                "burn_in_feats": [deepcopy(feature) for _ in range(4)],
                "actions": torch.zeros(C.DEPTH, C.H_OP, 7),
                "lang": torch.zeros(1, 1),
                "embodiment": "libero_franka",
                "src_fps": 20.0,
                "data_meta": {
                    "source": "libero",
                    "split": "gate",
                    "manifest_digest": manifest["digest"],
                    "task_id": tasks[i % 2],
                    "trajectory_id": trajectories[i % 2],
                    "trajectory_cluster_id": trajectories[i % 2],
                },
            }

    dataset = Dataset()
    pools = {
        tasks[0]: np.arange(0, size, 2),
        tasks[1]: np.arange(1, size, 2),
    }
    sampler = HomogeneousSampler(
        {"libero_franka": size}, batch_size=batch_size, seed=0,
        sampling="uniform_task", task_indices={"libero_franka": pools},
    )
    loader = SimpleNamespace(
        datasets={"libero_franka": dataset},
        sampler=sampler,
        sampling="uniform_task",
        batch_size=batch_size,
        n_windows=size,
        effective_workers=0,
        num_workers=0,
        pin_memory=False,
        trajectory_manifest=lambda: manifest,
    )
    seen_ordinals: list[int] = []

    def fake_measure(_model, window, *, ordinals, **_kwargs):
        ordinal_list = list(ordinals)
        n = len(ordinal_list)
        seen_ordinals.extend(ordinal_list)
        assert window["actions"].shape[0] == n + 1
        return (
            np.full((n, C.DEPTH), 0.1),
            np.full((n, C.DEPTH), 0.2),
            np.full(n, 0.3),
            None,
            None,
            None,
        )

    monkeypatch.setattr(gate, "measure_batch", fake_measure)
    rows, source = gate.collect_metric_rows(
        object(), loader, windows=256, batch_size=batch_size,
        n_candidates=32, seed=0, device="cpu",
    )
    assert rows.n == 256
    assert seen_ordinals == list(range(256))
    assert source["selection_unique"] is True
    assert len(source["selected_window_order"]) == 256
    assert source["selected_window_order_sha256"] == (
        "sha256:" + gate._canonical_json_sha256(source["selected_window_order"])
    )


def test_context_metric_math_is_batch_independent_and_sequential():
    selected = [0, 1, 2, 3]
    bank, zs, zts, c_seq = _template_problem([3, *selected])
    expected_delta, expected_gain = gate.dynamics_metric_rows(bank, zs, zts, c_seq)
    assert expected_delta.shape == expected_gain.shape == (4, C.DEPTH)
    assert torch.all(expected_delta > 0)
    torch.testing.assert_close(
        expected_gain, torch.ones_like(expected_gain), atol=1e-6, rtol=0,
    )

    for batch_size in (1, 2, 3, 4):
        deltas, gains = [], []
        for _, _, context in gate.cyclic_context_chunks(selected, batch_size):
            chunk_bank, chunk_zs, chunk_zts, chunk_c = _template_problem(context)
            delta, gain = gate.dynamics_metric_rows(
                chunk_bank, chunk_zs, chunk_zts, chunk_c,
            )
            deltas.append(delta)
            gains.append(gain)
        torch.testing.assert_close(torch.cat(deltas), expected_delta)
        torch.testing.assert_close(torch.cat(gains), expected_gain)


def test_per_root_proposal_stream_is_chunk_independent():
    logits = torch.linspace(-1.0, 1.0, 9 * 16).reshape(9, 16)
    whole = gate.proposal_plans_from_logits(
        logits, n_candidates=32, depth=C.DEPTH, seed=0,
        ordinals=range(9), topk=C.TOPK,
    )
    chunked = torch.cat([
        gate.proposal_plans_from_logits(
            logits[lo:hi], n_candidates=32, depth=C.DEPTH, seed=0,
            ordinals=range(lo, hi), topk=C.TOPK,
        )
        for lo, hi in ((0, 1), (1, 4), (4, 9))
    ])
    torch.testing.assert_close(chunked, whole, rtol=0, atol=0)
    assert torch.all((whole > 0).sum(-1) == C.TOPK)
    torch.testing.assert_close(whole.sum(-1), torch.ones_like(whole[..., 0]))


def test_proposal_root_residual_uses_common_chunk_independent_decoder_noise():
    state = SimpleNamespace(coeff=None, common_noise=[])

    class Decoder:
        def __call__(self, proprio, coeff, *, noise):  # noqa: ARG002
            state.coeff = coeff
            state.common_noise.append(bool(torch.equal(
                noise, noise[:1].expand_as(noise),
            )))
            return noise

    class QAction:
        def __call__(self, segment, belief):  # noqa: ARG002
            out = state.coeff.clone()
            out[:, 0] = out[:, 0] + segment[:, 0, 0]
            return out

    b, n = 5, 6
    proprio = torch.zeros(b, 7)
    belief = torch.zeros(b, C.K, C.D)
    coeff = torch.zeros(b, n, C.M)
    coeff[..., :C.TOPK] = 1.0 / C.TOPK
    whole = gate.proposal_root_q_action_residuals(
        Decoder(), QAction(), proprio, belief, coeff,
        seed=3, ordinals=range(b),
    )
    chunked = torch.cat([
        gate.proposal_root_q_action_residuals(
            Decoder(), QAction(), proprio[lo:hi], belief[lo:hi], coeff[lo:hi],
            seed=3, ordinals=range(lo, hi),
        )
        for lo, hi in ((0, 2), (2, 5))
    ])
    torch.testing.assert_close(chunked, whole, rtol=0, atol=0)
    assert state.common_noise and all(state.common_noise)
    assert torch.all(whole.std(dim=1) == 0), "candidates must share root noise"


def test_action_anchor_rows_are_paired_to_deploy_decode_and_support():
    class ReferenceQAction:
        def __call__(self, action, belief):  # noqa: ARG002
            out = torch.zeros(action.shape[0], C.M, device=action.device)
            out[:, :C.TOPK] = 1.0 / C.TOPK
            return out

    class Decoder:
        def loss(self, proprio, coeff, action, **kwargs):  # noqa: ARG002
            return (coeff[:, 0] - 0.25).pow(2)

    class Proposal:
        def logits(self, belief, lang):  # noqa: ARG002
            out = torch.full((belief.shape[0], C.M), -10.0)
            out[:, :C.TOPK] = torch.arange(C.TOPK, dtype=out.dtype)
            return out

    b = 3
    actions = torch.zeros(b, C.DEPTH, C.H_OP, 7)
    zs = [torch.zeros(b, C.K, C.D) for _ in range(C.DEPTH)]
    proprio = [torch.zeros(b, 7) for _ in range(C.DEPTH)]
    lang = torch.zeros(b, 1, 8)
    candidate = torch.zeros(b, C.DEPTH, C.M)
    candidate[..., :C.TOPK] = 1.0 / C.TOPK
    decode, support = gate.action_anchor_preservation_rows(
        Decoder(), Proposal(), ReferenceQAction(), actions, zs, proprio, lang,
        candidate, seed=9, ordinals=range(b),
    )
    torch.testing.assert_close(decode, torch.zeros_like(decode))
    torch.testing.assert_close(support, torch.zeros_like(support))

    drifted = torch.zeros_like(candidate)
    drifted[..., C.TOPK:2 * C.TOPK] = 1.0 / C.TOPK
    decode, support = gate.action_anchor_preservation_rows(
        Decoder(), Proposal(), ReferenceQAction(), actions, zs, proprio, lang,
        drifted, seed=9, ordinals=range(b),
    )
    assert torch.all(decode < 0)
    assert torch.all(support < 0)


def test_pairwise_leaf_spread_matches_explicit_unordered_pairs():
    leaves = torch.tensor([[[[1.0, -1.0, 0.0, 0.0]],
                            [[0.0, 1.0, -1.0, 0.0]],
                            [[0.0, 0.0, 1.0, -1.0]]]])
    got = gate.proposal_leaf_pairwise_spread(leaves)
    explicit = torch.stack([
        ln_cosine_distance(leaves[:, i], leaves[:, j], "per_slot")
        for i in range(3) for j in range(i + 1, 3)
    ]).mean(0)
    torch.testing.assert_close(got, explicit)


def test_cpu_window_mover_promotes_cached_fp16_and_preserves_metadata():
    feature = {"views": torch.ones(2, dtype=torch.float16),
               "proprio": torch.ones(2, dtype=torch.float16),
               "lang": torch.ones(2, dtype=torch.float16)}
    window = {
        "feats": [deepcopy(feature) for _ in range(C.DEPTH + 1)],
        "burn_in_feats": [deepcopy(feature) for _ in range(4)],
        "lang": torch.ones(1, 2, dtype=torch.float16),
        "actions": torch.ones(1, C.DEPTH, C.H_OP, 7, dtype=torch.float16),
        "data_meta": {"identity": "unchanged"},
    }
    moved = gate._to_device(window, "cpu", torch.float32)
    assert moved["actions"].dtype == torch.float32
    assert moved["feats"][0]["views"].dtype == torch.float32
    assert moved["burn_in_feats"][0]["proprio"].dtype == torch.float32
    assert moved["data_meta"] is window["data_meta"]


def test_reference_heads_authenticate_seed_and_frozen_tensor_exactness(tmp_path):
    qa = torch.nn.Linear(3, 2)
    reference_qa = torch.nn.Linear(3, 2)
    frozen_names = ("estimator", "ema", "q_delta", "decoder", "proposal")

    def payload(q_action, *, frozen_shift=0.0):
        state = {
            f"{name}.weight": torch.tensor([1.0 + frozen_shift])
            for name in frozen_names
        }
        state.update({f"q_action.{key}": value.detach().clone()
                      for key, value in q_action.state_dict().items()})
        return {
            "global_step": 49_666,
            "config_hash": "a199324a6205bb6d",
            "consolidated": {"tool": "test"},
            "resolved_config": {"model": {"same": True}},
            "model": state,
        }

    reference_path = tmp_path / "reference.pt"
    candidate_path = tmp_path / "candidate.pt"
    torch.save(payload(reference_qa), reference_path)
    torch.save(payload(qa), candidate_path)
    preservation = {
        **_valid_joint_gate_config()["offline_gate"]["preservation"],
        "reference_checkpoint_sha256": gate.sha256_file(reference_path),
    }
    candidate = gate.Candidate(
        model=SimpleNamespace(q_action=qa), config={},
        provenance={"path": str(candidate_path)}, gate_settings={},
    )
    loaded = gate.load_reference_heads(
        reference_path, candidate, preservation, device="cpu",
    )
    for got, expected in zip(loaded.q_action.parameters(), reference_qa.parameters()):
        torch.testing.assert_close(got, expected)
    assert all(item["tensor_exact"]
               for item in loaded.provenance["frozen_modules"].values())
    assert loaded.provenance["frozen_modules"]["potential"] == {
        "present": False, "tensors": 0, "numel": 0, "tensor_exact": True,
    }

    torch.save(payload(qa, frozen_shift=1.0), candidate_path)
    with pytest.raises(gate.GateError, match="frozen estimator changed"):
        gate.load_reference_heads(
            reference_path, candidate, preservation, device="cpu",
        )


def test_paired_cluster_bootstrap_is_deterministic_and_shares_draws():
    rows = _rows()
    values, _ = gate._metric_matrix(rows)
    x = values[:, :1]
    paired = np.concatenate([x, 2.0 * x + 1.0], axis=1)
    one = gate.paired_cluster_bootstrap(
        paired, rows.task_ids, rows.cluster_ids, samples=500, seed=17,
    )
    two = gate.paired_cluster_bootstrap(
        paired, rows.task_ids, rows.cluster_ids, samples=500, seed=17,
    )
    for key in ("point", "lower", "upper"):
        np.testing.assert_array_equal(one[key], two[key])
        assert one[key][1] == pytest.approx(2.0 * one[key][0] + 1.0)
    assert one["n_tasks"] == 4
    assert one["n_trajectory_clusters"] == 8


def test_bootstrap_fails_closed_on_nonfinite_or_cross_task_cluster():
    rows = _rows()
    values, _ = gate._metric_matrix(rows)
    values[0, 0] = np.nan
    with pytest.raises(gate.GateError, match="non-finite"):
        gate.paired_cluster_bootstrap(values, rows.task_ids, rows.cluster_ids)
    values[0, 0] = 1.0
    bad_tasks = list(rows.task_ids)
    bad_tasks[0] = "different-task"
    with pytest.raises(gate.GateError, match="spans tasks"):
        gate.paired_cluster_bootstrap(values, bad_tasks, rows.cluster_ids)


def test_gate_uses_exact_named_thresholds_and_strict_lower_bound():
    passing = gate.summarize_gate(
        _rows(), requirements=_requirements(), bootstrap_samples=250, seed=3,
    )
    assert passing["passed"] and passing["status"] == "PASS"
    assert set(passing["gates"]) == set(gate.REQUIREMENT_KEYS)
    assert all(item["passed"] for item in passing["gates"].values())

    equality = _rows()
    equality.delta_sel[:, 2] = 0.0
    failing = gate.summarize_gate(
        equality, requirements=_requirements(), bootstrap_samples=250, seed=3,
    )
    assert not failing["passed"] and failing["status"] == "FAIL"
    assert not failing["metrics"]["delta_sel/h3"]["passed"]
    assert not failing["gates"][gate.REQUIREMENT_KEYS[0]]["passed"]
    assert any("delta_sel/h3" in item and "<= threshold" in item
               for item in failing["failures"])


def test_joint_gate_adds_strict_proposal_root_residual_preservation():
    n, candidates = 24, 32
    preservation = _valid_joint_gate_config()["offline_gate"]["preservation"]
    passing = gate.summarize_gate(
        _rows(residual=np.zeros((n, candidates))),
        requirements=_requirements(), bootstrap_samples=250, seed=3,
        preservation=preservation,
    )
    assert passing["passed"]
    residual = np.ones((n, candidates))
    failing = gate.summarize_gate(
        _rows(residual=residual), requirements=_requirements(),
        bootstrap_samples=250, seed=3, preservation=preservation,
    )
    assert not failing["passed"]
    metric = failing["metrics"]["proposal_root_q_action_residual"]
    assert metric["root_exhaustion_rate"] == 1.0
    assert metric["comparison"] == "root_exhaustion_rate strictly_less_than maximum"
    assert not failing["gates"]["proposal_root_q_action_residual_preservation"]["passed"]


def test_joint_gate_rejects_baseline_action_or_support_regression():
    preservation = _valid_joint_gate_config()["offline_gate"]["preservation"]
    rows = _rows(residual=np.zeros((24, 32)))
    rows.action_decode_improvement[:] = -0.1
    result = gate.summarize_gate(
        rows, requirements=_requirements(), bootstrap_samples=250, seed=3,
        preservation=preservation,
    )
    assert not result["passed"]
    assert not result["metrics"]["action_decode_improvement_vs_deploy"]["passed"]
    assert not result["gates"]["deploy_action_semantics_preservation"]["passed"]

    rows.action_decode_improvement[:] = 0.0
    rows.proposal_support_overlap_change[:] = -0.1
    result = gate.summarize_gate(
        rows, requirements=_requirements(), bootstrap_samples=250, seed=3,
        preservation=preservation,
    )
    assert not result["metrics"][
        "proposal_support_overlap_change_vs_deploy"
    ]["passed"]


def test_manifest_digest_and_selected_identity_helpers_are_deterministic():
    manifest = _manifest()
    assert gate.validate_manifest(manifest) == manifest
    corrupt = deepcopy(manifest)
    corrupt["tasks"]["suite/renamed"] = corrupt["tasks"].pop("suite/a")
    with pytest.raises(gate.GateError, match="independently computed"):
        gate.validate_manifest(corrupt)

    windows = [
        SimpleNamespace(
            traj_id="suite/a/demo_49",
            start=start,
            action_free=False,
            obs_src_index=(start, start + 8, start + 16, start + 24, start + 32),
        )
        for start in (32, 34)
    ]
    dataset = SimpleNamespace(recurrent_burn_in=4, windows=windows)
    loader = SimpleNamespace(datasets={"libero_franka": dataset})
    records = gate._selected_window_records(
        loader, [1, 0], {"suite/a/demo_49": "suite/a"},
    )
    assert [(item["ordinal"], item["canonical_start"]) for item in records] == [
        (0, 34), (1, 32),
    ]
    assert gate._canonical_json_sha256(records) == gate._canonical_json_sha256(
        json.loads(json.dumps(records)),
    )


def test_gate_settings_authenticate_exact_bank_only_recipe():
    assert gate._gate_settings(_valid_gate_config()) == {
        "windows": 256,
        "candidates": 32,
        "bootstrap_samples": 2_000,
        "confidence": 0.95,
        "seed": 0,
        "requirements": _requirements(),
        "method_variant": "bank_only",
        "transition_parameter_reset": None,
        "preservation": None,
        "cosine": "per_slot",
    }
    n4 = _valid_gate_config()
    n4["run"]["name"] = "r0a_bank_ca_n4"
    n4["losses"]["dyn"]["neg_weight"] = 4.0
    assert gate._gate_settings(n4)["method_variant"] == "bank_only"


def test_gate_settings_authenticate_exact_joint_recipe_and_preservation():
    settings = gate._gate_settings(_valid_joint_gate_config())
    assert settings["method_variant"] == "joint_q_action_bank"
    assert settings["transition_parameter_reset"] is None
    assert settings["preservation"] == {
        "reference_checkpoint_sha256":
            "15f286c268caa5327d5aa3abf1f67ebd0555c426a509fef22cb7f537bf6ab4e1",
        "reference_config_hash": "a199324a6205bb6d",
        "reference_global_step": 49_666,
        "action_decode_improvement_ci_low": 0.0,
        "proposal_support_overlap_change_ci_low": -0.05,
        "q_action_residual_max": 0.5,
        "max_root_exhaustion_rate": 0.01,
    }

    identity = gate._gate_settings(
        _valid_joint_gate_config(identity_centered=True)
    )
    assert identity["method_variant"] == "joint_q_action_bank_identity_centered"
    assert identity["transition_parameter_reset"] == gate.IDENTITY_CENTERED_RESET


@pytest.mark.parametrize(("path", "bad_value"), [
    ("run.name", "r0a_bank_ca_qa"),
    ("optim.transition_parameter_reset.source_config_hash", "0ec8af0a26135ecc"),
    ("optim.transition_parameter_reset.tensors", {"bank.omega": "one"}),
])
def test_identity_centered_gate_settings_fail_closed_on_recipe_drift(path, bad_value):
    cfg = _valid_joint_gate_config(identity_centered=True)
    _set_path(cfg, path, bad_value)
    with pytest.raises(gate.GateError):
        gate._gate_settings(cfg)


@pytest.mark.parametrize(("path", "bad_value"), [
    ("run.steps", 79_999),
    ("losses.dyn.enabled", False),
    ("losses.dyn.coeff_source", "q_delta"),
    ("losses.act.enabled", True),
    ("losses.proposal.weight", 0.1),
    ("train_modules", ["bank", "proposal"]),
    ("optim.update_ema", True),
    ("optim.lr_scales.bank", 1.0),
    ("data.recurrent_burn_in", 0),
    ("data.trajectory_split", "all"),
    ("offline_gate.script", "other.py"),
    ("offline_gate.required", False),
    ("offline_gate.direct_e2e", True),
    ("offline_gate.bootstrap_samples", 10_000),
    ("offline_gate.windows", 512),
    ("offline_gate.candidates", 16),
    ("offline_gate.requirements.delta_sel_ci_low_per_horizon", -0.01),
])
def test_gate_settings_fail_closed_on_recipe_drift(path, bad_value):
    cfg = _valid_gate_config()
    _set_path(cfg, path, bad_value)
    with pytest.raises(gate.GateError):
        gate._gate_settings(cfg)


@pytest.mark.parametrize(("path", "bad_value"), [
    ("run.name", "r0a_bank_ca_n4"),
    ("train_modules", ["q_action", "bank"]),
    ("optim.reset_state_modules", ["q_action", "bank"]),
    ("optim.lr_scales.q_action", 0.1),
    ("losses.dyn.detach_coeff", True),
    ("losses.dyn.detach_coeff", 0),
    ("losses.dyn.neg_weight", 1.0),
    ("losses.act.enabled", False),
    ("losses.act.weight", 0.0),
    ("losses.act.align_to", "q_delta"),
    ("losses.act.decode_from", "proposal"),
    ("convergence.primary", ["loss/dyn"]),
    ("efficacy_gate.required", False),
    ("liveness_gate.end_inclusive", 52_667),
    ("liveness_gate.requirements.gnorm_q_action_median_strict_gt", 0.0),
    ("offline_gate.preservation.reference_config_hash", "wrong"),
    ("offline_gate.preservation.action_decode_improvement_ci_low", -0.01),
    ("offline_gate.preservation.proposal_support_overlap_change_ci_low", -0.06),
    ("offline_gate.preservation.q_action_residual_max", 0.4),
    ("offline_gate.preservation.max_root_exhaustion_rate", 0.02),
])
def test_joint_gate_settings_fail_closed_on_recipe_drift(path, bad_value):
    cfg = _valid_joint_gate_config()
    _set_path(cfg, path, bad_value)
    with pytest.raises(gate.GateError):
        gate._gate_settings(cfg)


def test_runtime_semantics_default_to_config_and_exact_overrides_only():
    args = gate.parse_args(["--checkpoint", "candidate.pt", "--out", "gate.json",
                            "--batch-size", "7", "--workers", "0"])
    assert args.bootstrap_samples is None
    recipe = gate._runtime_recipe(_valid_gate_config(), args)
    assert recipe["bootstrap_samples"] == 2_000
    assert recipe["runtime"]["batch_size"] == 7
    assert not any(recipe["semantic_cli_overrides"].values())

    exact = gate.parse_args([
        "--checkpoint", "candidate.pt", "--out", "gate.json",
        "--windows", "256", "--candidates", "32",
        "--bootstrap-samples", "2000", "--confidence", "0.95", "--seed", "0",
    ])
    assert all(gate._runtime_recipe(
        _valid_gate_config(), exact,
    )["semantic_cli_overrides"].values())
    exact.windows = 255
    with pytest.raises(gate.GateError, match="runtime windows"):
        gate._runtime_recipe(_valid_gate_config(), exact)


def test_hash_and_atomic_manifest_helpers(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"bank-ca\x00evidence")
    assert gate.sha256_file(artifact, chunk_bytes=3) == hashlib.sha256(
        b"bank-ca\x00evidence"
    ).hexdigest()

    report = tmp_path / "report.json"
    gate.atomic_write_json(report, {"status": "FAIL", "passed": False})
    first = report.read_bytes()
    gate.atomic_write_json(report, {"passed": False, "status": "FAIL"})
    assert report.read_bytes() == first
    assert json.loads(first) == {"passed": False, "status": "FAIL"}
    assert not list(tmp_path.glob(".*.tmp"))

    with pytest.raises(ValueError):
        gate.atomic_write_json(report, {"bad": float("nan")})
    assert not list(tmp_path.glob(".*.tmp"))
