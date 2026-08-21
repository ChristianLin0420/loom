"""Contracts for the terminal full-coverage task-balanced PA scaffold."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn
import yaml

import contracts as C
from loom.eval import outcome_recovery as recovery
from loom.heads.proposal import pl_log_prob
from loom.train import outcome_task_balanced_positive_advantage as tb
from loom.train.loop import read_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "r0a_outcome_task_balanced_pa_gate.yaml"
RECEIPT = ROOT / "configs" / "r0a_outcome_task_balanced_pa_selection_v1.json"
SOURCE_SPECS = (
    {
        "split": "train0",
        "path": "runs/outcome_recovery_s49666_train0",
        "manifest_sha256": "f92f50960e1640b32f2f50c6e9a7c61603204ea21369c6e2493d3770b3683c17",
        "identity_digest": "6aae6bbb5f6226de726de64bd8b57d6f4fb673a63c5e23c49ad136a03dd75433",
        "informative_groups": 146,
    },
    {
        "split": "train1",
        "path": "runs/outcome_recovery_s49666_train1",
        "manifest_sha256": "8c45d514454598eb1c53c0d3ea3a12b3606f84baae612e5d3c2dc50bfd904421",
        "identity_digest": "331c812f62a8249d6c4be6a368b2bc9c5bbe6e352a67f6e8ac89a4deeaf984a0",
        "informative_groups": 159,
    },
    {
        "split": "train2",
        "path": "runs/outcome_recovery_s49666_train2",
        "manifest_sha256": "4a53fca9490840e90319d2fde986f8f6c12a6d236869c5000e5b4f0e1555b29b",
        "identity_digest": "302d7a4a95c45526338e1bfffd6008a893825ba1eb6157217169cfe661fd77ee",
        "informative_groups": 143,
    },
    {
        "split": "train3",
        "path": "runs/outcome_recovery_s49666_train3",
        "manifest_sha256": "289c0d6796d1bfccf471bb519c3145b0837147280b5059ae62d821a5b7d3594e",
        "identity_digest": "f6c0eb2f6b11a477f84be6323ef1a94f720dfe30e8fd854fd1d0c3d25bcef35e",
        "informative_groups": 148,
    },
    {
        "split": "train4",
        "path": "runs/outcome_recovery_s49666_train4",
        "manifest_sha256": "025a8a81556a733da4401fb489306222e93e72535f3ca8266c4780fc76f9857b",
        "identity_digest": "206fb752eb2f345284286a35bdc277eee03573c916563abb724c8d3f67dad13c",
        "informative_groups": 148,
    },
    {
        "split": "train5",
        "path": "runs/outcome_recovery_s49666_train5",
        "manifest_sha256": "61253c67af3ea3c5cda710a78498e098bd2ea54082ba557ad2145a79e14a1700",
        "identity_digest": "8b19c20610d4dcfed619ca2854e548b6bac5ca2d7a0bc26dd00182d2614f7ea7",
        "informative_groups": 159,
    },
)
PRIOR_FROZEN_SHA256 = {
    "loom/train/outcome_grpo_v2.py": "d37ac75b3a2f075cff76b208b2eed6a71bfeb5ff7e14a478aff981a059acbd6a",
    "scripts/train_outcome_grpo_v2.py": "f03d79831b6f80242a104ed07a83bc3a899646a12c318652a151c7f04e807975",
    "scripts/outcome_grpo_v2_pilot.sbatch": "d05b085a65ab204a08bfa6264500e2303533de2ef5ecf57a68592af8fd13c3bd",
    "tests/test_outcome_grpo_v2.py": "8af14ae49c7d2efa6c6f4d856852b4dd1e3a7be57f269f7369a60b2119b4d1c9",
    "loom/train/outcome_positive_advantage.py": "c70dfa6239ff8ef1eda9cf16b167c7bf5d3b7de00c918b8fa3c0b69faa69a358",
    "tests/test_outcome_positive_advantage.py": "3328361920e44d7ab18a253a61b04146e1d2a100fbc882a2e0933378cefe2c8d",
    "configs/r0a_outcome_positive_advantage_audit.yaml": "2e375a3db095006d8b4dbe972f9b938da534bbc8b1bd7b7c1dfb4e186d2e9dd4",
    "scripts/outcome_positive_advantage_direction_audit.py": "33c9affa3370cbad6f1ff1b9c925dc5f640c4efb1d1b48e0903d4271c486d858",
    "scripts/outcome_positive_advantage_direction_audit.sbatch": "1d3ed5ead9db4d33cca9e00544fec63ce3d7f1565d7d3346fda73352e4eb2e4c",
    "tests/test_outcome_positive_advantage_direction_audit.py": "63520da30820665267907916acf78c77c920ff0d3a2e1e4f2b430c4247cab54c",
    "scripts/outcome_round_robin_direction_audit.py": "0e9a163f288ec57183493294e829a2d02960fa3af0db0d77aed629e93bc19977",
    "scripts/outcome_round_robin_direction_audit.sbatch": "ec0a9625db7b47a4c89706495f4ebfa4d9314dc7c4814434f9c70ff96e1319f8",
    "tests/test_outcome_round_robin_direction_audit.py": "6aad3046037eeffcb33ab138863c36ede91628930d009a0577058bbd052c2f15",
    "configs/r0a_outcome_grpo_v2_pilot.yaml": "249a221d84032f8c3801a7430e48597a364fa16b02b378e9a17d30c8a56cdf44",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def pool_partition():
    groups, sources = tb.load_authenticated_informative_groups(ROOT, SOURCE_SPECS)
    assignments = tb.build_task_stratified_partition(groups)
    return groups, sources, assignments


class RowwiseWitnessProposal(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.125, dtype=torch.float32))
        self.register_buffer("base", torch.linspace(-1.0, 1.0, C.M))
        self.batch_sizes: list[int] = []
        self.autocast: list[bool] = []

    def logits(self, z: Tensor, lang: Tensor) -> Tensor:  # noqa: ARG002
        self.batch_sizes.append(int(z.shape[0]))
        self.autocast.append(torch.is_autocast_enabled(z.device.type))
        signal = z.float().mean(dim=(-1, -2)).unsqueeze(-1)
        slope = torch.linspace(-0.5, 0.5, C.M, device=z.device)
        return self.base.unsqueeze(0) + self.scale * signal * slope.unsqueeze(0)


def _payload(proposal: RowwiseWitnessProposal) -> dict:
    rewards = [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    arms: list[dict] = [{"terminal_reward": torch.tensor(rewards[0])}]
    with torch.no_grad():
        for arm in range(1, recovery.GROUP_SIZE):
            z = torch.arange(4, dtype=torch.float32).reshape(4, 1, 1) + arm / 10.0
            lang = torch.zeros(1, 1)
            logits = torch.cat([
                proposal.logits(z[row:row + 1], lang.reshape(1, 1, 1))
                for row in range(4)
            ])
            order = logits.topk(C.TOPK, dim=-1).indices
            old = torch.cat([
                pl_log_prob(logits[row:row + 1], order[row:row + 1])
                for row in range(4)
            ])
            arms.append({
                "z": z,
                "lang": lang,
                "ordered_support": order,
                "old_logprob": old,
                "terminal_reward": torch.tensor(rewards[arm]),
            })
    proposal.batch_sizes.clear()
    proposal.autocast.clear()
    return {"group_id": "task-balanced-test", "arms": arms}


def test_prior_frozen_files_remain_byte_identical():
    assert {path: _sha256(ROOT / path) for path in PRIOR_FROZEN_SHA256} == (
        PRIOR_FROZEN_SHA256
    )


def test_binary_informative_definition_exactly_preserves_inherited_int_rule():
    assert not tb.informative_terminal_rewards([0.0] * 8)
    assert not tb.informative_terminal_rewards([1.0] * 8)
    assert tb.informative_terminal_rewards([0.0, 1.0] + [0.0] * 6)
    with pytest.raises(tb.TaskBalancedPositiveAdvantageError, match="exact 0/1"):
        tb.informative_terminal_rewards([0.0, 0.5] + [0.0] * 6)
    with pytest.raises(tb.TaskBalancedPositiveAdvantageError, match="nan/inf"):
        tb.informative_terminal_rewards([0.0, float("nan")] + [0.0] * 6)
    with pytest.raises(ValueError, match="contain 8"):
        tb.informative_terminal_rewards([0.0] * 7)


def test_all_pinned_manifests_produce_exact_903_group_pool(pool_partition):
    groups, sources, _assignments = pool_partition
    assert len(groups) == tb.EXPECTED_INFORMATIVE_GROUPS == 903
    assert len(sources) == 6
    assert [row["observed_informative_groups"] for row in sources] == [
        146, 159, 143, 148, 148, 159,
    ]
    assert {group.task_key for group in groups} == set(tb.TASK_KEYS)
    assert len({group.identity_key for group in groups}) == 903
    assert len({group.group_id for group in groups}) == 903


def test_manifest_and_collection_identity_pin_mutations_fail_closed():
    bad_manifest = [dict(row) for row in SOURCE_SPECS]
    bad_manifest[2]["manifest_sha256"] = "0" * 64
    with pytest.raises(tb.TaskBalancedPositiveAdvantageError, match="manifest SHA"):
        tb.load_authenticated_informative_groups(ROOT, bad_manifest)

    bad_identity = [dict(row) for row in SOURCE_SPECS]
    bad_identity[4]["identity_digest"] = "f" * 64
    with pytest.raises(tb.TaskBalancedPositiveAdvantageError, match="identity differs"):
        tb.load_authenticated_informative_groups(ROOT, bad_identity)


def test_partition_is_exact_reviewed_hash_rotation_assignment(pool_partition):
    groups, _sources, assignments = pool_partition
    assert len(assignments) == len(groups) == 903
    validation = tb.validate_task_stratified_partition(assignments)
    assert validation == {
        "passed": True,
        "informative_groups": 903,
        "tasks": 40,
        "shard_totals": [301, 301, 301],
        "minimum_groups_per_task_shard": 2,
        "maximum_per_task_shard_count_difference": 1,
        "tasks_per_rank": 5,
    }
    assert [
        [len(tb.assignments_for(assignments, shard=shard, rank=rank))
         for rank in range(8)]
        for shard in range(3)
    ] == [
        [39, 40, 36, 33, 34, 41, 37, 41],
        [39, 42, 35, 30, 35, 43, 36, 41],
        [38, 41, 36, 31, 35, 42, 37, 41],
    ]
    expected_bytes = [
        [9769985322, 7955141808, 7558186264, 6681857414,
         8393843404, 10338023222, 8619368158, 8375430326],
        [9950958570, 8404222524, 7441884178, 6094502580,
         8807362834, 10716468674, 7885717080, 9711670006],
        [8975309412, 8047430838, 7653172504, 6351494010,
         9000884434, 10237565436, 8175109470, 8889779702],
    ]
    assert [
        [sum(row.group.sidecar_size for row in assignments
             if row.shard == shard and row.rank == rank)
         for rank in range(8)]
        for shard in range(3)
    ] == expected_bytes
    for row in assignments:
        assert row.rank == row.task_position % 8
        assert row.shard == (
            row.within_task_position + row.task_position % 3
        ) % 3


def test_partition_is_input_order_independent_and_tamper_rejected(pool_partition):
    groups, _sources, assignments = pool_partition
    reversed_rows = tb.build_task_stratified_partition(tuple(reversed(groups)))
    assert [row.row() for row in reversed_rows] == [row.row() for row in assignments]
    tampered = list(assignments)
    tampered[0] = replace(tampered[0], shard=1)
    with pytest.raises(tb.TaskBalancedPositiveAdvantageError):
        tb.validate_task_stratified_partition(tampered)


def test_frozen_selection_receipt_binds_exact_sources_union_and_assignment(
    pool_partition,
):
    _groups, sources, assignments = pool_partition
    expected = json.loads(RECEIPT.read_text())
    observed = tb.partition_receipt(assignments, sources)
    assert observed == expected
    assert _sha256(RECEIPT) == (
        "fbb360f7a7beeb468ca4e3532c7e0d1966b5eecbd90fb661306e67223ef3aeb2"
    )
    assert tb.canonical_sha256(expected) == (
        "d609725da4eb6010de20f20bb0df0ec6f8917a6049e211203190ca852e05cf79"
    )
    assert expected["pool"]["union_sha256"] == (
        "1b831f4845413f4104c5dd25497ced3a00e17e47f147867264714d6875c3e7fd"
    )
    assert expected["partition"]["assignment_sha256"] == (
        "79f2a7a3980fbf744d3652256205083b94681d04597b3f79780c956d0c2d1920"
    )
    assert expected["partition"]["context_selection"]["assignment_sha256"] == (
        "64cef16c7662615655c84cce12077a95c015e9f83d8d9db691e9ce9036d13634"
    )


def test_exact_v3_seed0_visit0_contexts_cover_903x7x2(pool_partition):
    groups, _sources, assignments = pool_partition
    assert tb.SAMPLER_SEED == 0
    assert tb.CONTEXTS_PER_ARM == 2
    atoms = 0
    for group in groups:
        selected = tb.frozen_v3_visit0_replan_indices(group)
        assert set(selected) == set(range(1, 8))
        for arm, indices in selected.items():
            assert len(indices) == 2
            assert len(set(indices)) == 2
            assert all(0 <= value < group.n_replans_by_arm[arm] for value in indices)
            atoms += len(indices)
    assert atoms == 903 * 7 * 2
    receipt = tb.partition_receipt(assignments, pool_partition[1])
    contexts = receipt["partition"]["context_selection"]
    assert contexts["sampler_seed"] == 0
    assert contexts["visit"] == 0
    assert contexts["atoms"] == atoms


def test_context_rule_exactly_replays_all_prior_24_audit_rows(pool_partition):
    groups, _sources, _assignments = pool_partition
    by_key = {(group.split, group.group_index): group for group in groups}
    report = json.loads((
        ROOT / "runs" / "diagnostics" / "outcome_positive_advantage_direction_audit"
        / "outcome_positive_advantage_direction_audit_v2_s49666_32581242.json"
    ).read_text())
    compared = 0
    for point in report["train_points"]:
        for row in point["rank_local_groups"]:
            assert row["visit"] == 0
            expected = {
                int(arm): tuple(values)
                for arm, values in row["replan_indices"].items()
            }
            assert tb.frozen_v3_visit0_replan_indices(
                by_key[(row["split"], row["group_index"])]
            ) == expected
            compared += 1
    assert compared == 24


def test_group_weights_are_equal_inside_task_then_equal_over_40(pool_partition):
    _groups, _sources, assignments = pool_partition
    for shard in range(3):
        task_weight = {task: 0.0 for task in tb.TASK_KEYS}
        global_weight = 0.0
        for rank in range(8):
            weights = tb.local_group_weights(
                assignments, shard=shard, rank=rank,
            )
            assert sum(weights.values()) == pytest.approx(1.0 / 8.0, abs=1e-15)
            for row in tb.assignments_for(assignments, shard=shard, rank=rank):
                task_weight[row.group.task_key] += weights[row.group.identity_key]
                global_weight += weights[row.group.identity_key]
        assert global_weight == pytest.approx(1.0, abs=4e-15)
        assert all(value == pytest.approx(1.0 / 40.0, abs=2e-16)
                   for value in task_weight.values())


def test_exact_assignment_aggregation_and_distributed_sum_geometry(pool_partition):
    _groups, _sources, assignments = pool_partition
    shard = 1
    task_value = {task: float(index + 1) for index, task in enumerate(tb.TASK_KEYS)}
    production_pa = []
    production_full = []
    for rank in range(8):
        pa_accumulator = tb.StreamingMacroGradientAccumulator(
            assignments, shard=shard, rank=rank,
        )
        full_accumulator = tb.StreamingMacroGradientAccumulator(
            assignments, shard=shard, rank=rank,
        )
        for item in tb.streaming_group_rows(assignments, shard=shard, rank=rank):
            key = item.assignment.group.identity_key
            value = task_value[item.assignment.group.task_key]
            pa_accumulator.add_weighted_group_gradient(
                key, torch.tensor([value * item.production_mean_reducer_weight]),
            )
            # This is a direct per-group full vector witness, not a component
            # reconstruction inside the accumulator.
            full_accumulator.add_weighted_group_gradient(
                key,
                torch.tensor([
                    3.0 * value * item.production_mean_reducer_weight
                ]),
            )
        pa_vector, pa_receipt = pa_accumulator.finalize(require_demo=False)
        full_accumulator.add_demo_gradient_once(torch.tensor([3.0]))
        full_vector, full_receipt = full_accumulator.finalize(require_demo=True)
        production_pa.append(pa_vector)
        production_full.append(full_vector)
        assert pa_receipt["production_group_weight_sum"] == pytest.approx(1.0)
        assert full_receipt["demo_gradients_added"] == 1
    expected_pa = sum(task_value.values()) / 40.0
    expected_full = 3.0 * expected_pa + 3.0
    assert float(torch.stack(production_pa).mean()) == pytest.approx(expected_pa)
    assert float(torch.stack(production_full).mean()) == pytest.approx(expected_full)


def test_streaming_accumulator_rejects_missing_duplicate_or_out_of_order(
    pool_partition,
):
    _groups, _sources, assignments = pool_partition
    rows = tb.streaming_group_rows(assignments, shard=0, rank=0)
    accumulator = tb.StreamingMacroGradientAccumulator(
        assignments, shard=0, rank=0,
    )
    with pytest.raises(tb.TaskBalancedPositiveAdvantageError, match="ended before"):
        accumulator.finalize(require_demo=False)
    with pytest.raises(tb.TaskBalancedPositiveAdvantageError, match="out of order"):
        accumulator.add_weighted_group_gradient(
            rows[1].assignment.group.identity_key, torch.ones(2),
        )
    accumulator.add_weighted_group_gradient(
        rows[0].assignment.group.identity_key, torch.ones(2),
    )
    with pytest.raises(tb.TaskBalancedPositiveAdvantageError, match="out of order"):
        accumulator.add_weighted_group_gradient(
            rows[0].assignment.group.identity_key, torch.ones(2),
        )
    with pytest.raises(tb.TaskBalancedPositiveAdvantageError, match="must follow"):
        accumulator.add_demo_gradient_once(torch.ones(2))


def test_streaming_vs_monolithic_toy_has_exact_world_scaled_group_weights(
    pool_partition,
):
    _groups, _sources, assignments = pool_partition
    shard = 2
    dimension = len(assignments)
    key_position = {
        row.group.identity_key: index for index, row in enumerate(assignments)
    }
    production_vectors = []
    for rank in range(8):
        accumulator = tb.StreamingMacroGradientAccumulator(
            assignments, shard=shard, rank=rank,
        )
        for item in tb.streaming_group_rows(assignments, shard=shard, rank=rank):
            vector = torch.zeros(dimension, dtype=torch.float64, requires_grad=True)
            vector.data[key_position[item.assignment.group.identity_key]] = 1.0
            accumulator.add_weighted_group_gradient(
                item.assignment.group.identity_key,
                vector * item.production_mean_reducer_weight,
            )
            assert vector.grad is None
        got, _receipt = accumulator.finalize(require_demo=False)
        assert got.requires_grad is False
        production_vectors.append(got)
    distributed_mean = torch.stack(production_vectors).mean(0)
    expected = {}
    for rank in range(8):
        expected.update(tb.local_group_weights(assignments, shard=shard, rank=rank))
    for key, weight in expected.items():
        assert float(distributed_mean[key_position[key]]) == pytest.approx(
            weight, abs=1e-18,
        )


def test_streaming_direct_scalar_weighting_matches_monolithic_toy_autograd(
    pool_partition,
):
    _groups, _sources, assignments = pool_partition
    items = tb.streaming_group_rows(assignments, shard=0, rank=3)
    parameter = torch.tensor(0.25, dtype=torch.float32, requires_grad=True)
    accumulator = tb.StreamingMacroGradientAccumulator(
        assignments, shard=0, rank=3,
    )
    weighted_scalars = []
    for index, item in enumerate(items):
        direct = parameter.square() * float(index + 1)
        weighted = tb.production_weighted_group_scalar(item, direct)
        gradient = torch.autograd.grad(weighted, parameter, retain_graph=True)[0]
        accumulator.add_weighted_group_gradient(
            item.assignment.group.identity_key, gradient.reshape(1),
        )
        weighted_scalars.append(weighted)
    streamed, receipt = accumulator.finalize(require_demo=False)
    monolithic = torch.autograd.grad(
        torch.stack(weighted_scalars).sum(), parameter,
    )[0]
    torch.testing.assert_close(streamed, monolithic.reshape(1), rtol=2e-6, atol=1e-7)
    assert receipt["weight_application"] == "direct_scalar_before_autograd"
    with pytest.raises(ValueError, match="scalar"):
        tb.production_weighted_group_scalar(items[0], torch.ones(2))


def test_three_macro_average_is_not_misclaimed_as_one_shot_full_pool_mean(
    pool_partition,
):
    _groups, _sources, assignments = pool_partition
    # A varying within-task signal witnesses the q/q+1 shard-count distinction.
    per_shard = []
    for shard in range(3):
        task_means = []
        for task in tb.TASK_KEYS:
            rows = [row for row in assignments
                    if row.shard == shard and row.group.task_key == task]
            task_means.append(sum(row.within_task_position ** 2 for row in rows) / len(rows))
        per_shard.append(sum(task_means) / 40.0)
    full_task_means = []
    for task in tb.TASK_KEYS:
        rows = [row for row in assignments if row.group.task_key == task]
        full_task_means.append(
            sum(row.within_task_position ** 2 for row in rows) / len(rows)
        )
    assert sum(per_shard) / 3.0 != pytest.approx(sum(full_task_means) / 40.0)
    receipt = json.loads(RECEIPT.read_text())
    aggregation = receipt["aggregation"]
    assert aggregation[
        "three_macro_average_not_equal_to_one_shot_full_pool_group_mean"
    ] is True


def test_two_pass_group_scorer_requires_two_contexts_and_exact_identity():
    proposal = RowwiseWitnessProposal().eval()
    payload = _payload(proposal)
    indices = {arm: (0, 2) for arm in range(1, 8)}
    result = tb.sampled_task_group_components(
        proposal, payload, indices, device=torch.device("cpu"),
        require_recovery_identity=True,
    )
    assert result.metrics["scorer_passes"] == 2.0
    assert result.metrics["recovery_arms"] == 7.0
    assert result.metrics["recovery_atoms"] == 14.0
    assert result.metrics["recovery_identity_exact"] == 1.0
    assert result.metrics["recovery_max_abs_current_minus_old"] == 0.0
    assert float(result.recovery_k3) == 0.0
    assert proposal.batch_sizes == [1] * 28
    assert proposal.autocast == [False] * 28
    result.recovery_k3.backward()
    assert proposal.scale.grad is not None
    assert torch.count_nonzero(proposal.scale.grad) == 0

    with pytest.raises(tb.TaskBalancedPositiveAdvantageError, match="two contexts"):
        tb.sampled_task_group_components(
            proposal, payload, {arm: (0,) for arm in range(1, 8)},
            device=torch.device("cpu"), require_recovery_identity=True,
        )


def test_recovery_identity_mismatch_is_invalid_but_post_update_path_is_explicit():
    proposal = RowwiseWitnessProposal().eval()
    payload = _payload(proposal)
    payload["arms"][4]["old_logprob"] = (
        payload["arms"][4]["old_logprob"].detach() + 1.0e-3
    )
    indices = {arm: (0, 2) for arm in range(1, 8)}
    with pytest.raises(tb.TaskBalancedPositiveAdvantageError, match="current.float"):
        tb.sampled_task_group_components(
            proposal, payload, indices, device=torch.device("cpu"),
            require_recovery_identity=True,
        )
    result = tb.sampled_task_group_components(
        proposal, payload, indices, device=torch.device("cpu"),
        require_recovery_identity=False,
    )
    assert result.metrics["recovery_identity_required"] == 0.0
    assert result.metrics["recovery_identity_exact"] == 0.0
    assert result.metrics["recovery_max_abs_current_minus_old"] > 0.0
    assert float(result.recovery_k3) > 0.0


def test_three_macro_resume_contract_is_exact_and_tamper_evident():
    schedule = tb.ThreeMacroUpdateSchedule(
        partition_receipt_sha256=tb.canonical_sha256(json.loads(RECEIPT.read_text())),
        recipe_file_sha256=_sha256(CONFIG),
        core_closure_sha256=tb.core_source_identity()["sha256"],
        seed_checkpoint_sha256=(
            "15f286c268caa5327d5aa3abf1f67ebd0555c426a509fef22cb7f537bf6ab4e1"
        ),
    )
    assert schedule.stop_step == 49669
    assert [schedule.shard_at(step) for step in range(49666, 49669)] == [0, 1, 2]
    for step in range(49666, 49670):
        state = schedule.state_dict(step)
        schedule.validate_state_dict(state, step)
        assert state["completed_macro_updates"] == step - 49666
        assert state["optimizer_steps_completed"] == step - 49666
        assert state["post_update_trust_dev_gates_completed"] == step - 49666
        assert state["informative_groups_consumed"] == 301 * (step - 49666)
        assert state["completed_shards"] == list(range(step - 49666))
        assert state["one_full_903_group_pass_complete"] == (step == 49669)
    bad = schedule.state_dict(49668)
    bad["next_shard"] = 0
    with pytest.raises(tb.TaskBalancedPositiveAdvantageError, match="resume"):
        schedule.validate_state_dict(bad, 49668)
    with pytest.raises(ValueError, match="outside"):
        schedule.shard_at(49669)


def test_config_is_nonlaunchable_and_freezes_terminal_three_macro_authority():
    raw = yaml.safe_load(CONFIG.read_text())
    cfg = read_config(CONFIG)
    method = cfg["outcome_task_balanced_positive_advantage"]
    assert raw["extends"] == "r0a_outcome_grpo_v2_pilot.yaml"
    assert cfg["run"]["steps"] is None
    assert cfg["run"]["ckpt_every"] is None
    assert cfg["train_modules"] == []
    assert method["method_status"] == "FROZEN_SELECTION_NONLAUNCHABLE_GATE_NOT_RUN"
    assert method["training_loop_present"] is False
    assert method["launcher_present"] is False
    assert method["job_submitted"] is False
    assert method["prospective_pilot_maximum_authority"]["actual_macro_updates"] == 3
    assert method["prospective_pilot_maximum_authority"]["one_full_903_group_pass"] is True
    assert method["prospective_pilot_maximum_authority"]["longer_continuation"] == "forbidden"
    assert method["direction_gate"]["on_fail"] == (
        "terminal_stop_for_outcome_conditioned_proposal_optimization"
    )
    assert "64" not in json.dumps(method["prospective_pilot_maximum_authority"])
    assert "800" not in json.dumps(method["prospective_pilot_maximum_authority"])
    assert all(cfg["losses"][name]["enabled"] is False
               for name in ("grpo", "proposal", "balance"))
    assert all(float(cfg["losses"][name]["weight"]) == 0.0
               for name in ("grpo", "proposal", "balance"))
    assert all(float(cfg["losses"][name]["weight"]) == 1.0
               for name in ("positive_advantage", "recovery_reference", "demo_reference"))
    assert cfg["optim"]["lr_scales"]["proposal"] == 0.0125
    assert cfg["optim"]["grad_clip"] == 1.0
    assert cfg["optim"]["reset_state_modules"] == ["proposal"]


def test_config_pins_receipt_sources_chronology_and_exposed_dev_only():
    cfg = read_config(CONFIG)
    method = cfg["outcome_task_balanced_positive_advantage"]
    selection = method["frozen_selection"]
    assert selection["receipt_file_sha256"] == _sha256(RECEIPT)
    assert selection["receipt_canonical_sha256"] == tb.canonical_sha256(
        json.loads(RECEIPT.read_text())
    )
    assert method["authenticated_training_pool"]["sources"] == list(SOURCE_SPECS)
    chronology = method["exposed_development_chronology"]
    names = [row["name"] for row in chronology["ordered_exposures"]]
    assert names == [
        "v1_terminal_selection",
        "early_curve_diagnostic",
        "component_gradient_projection",
        "round_robin_direction_audit",
        "positive_advantage_direction_audit",
        "terminal_task_balanced_positive_advantage_direction_gate",
    ]
    for row in chronology["ordered_exposures"][:-1]:
        assert _sha256(ROOT / row["artifact"]) == row["artifact_sha256"]
    assert chronology["ordered_exposures"][-1]["status"] == (
        "PROSPECTIVELY_FROZEN_NOT_RUN"
    )
    assert chronology["formal_terminal_collection_accessed"] is False
    assert cfg["artifact_policy"]["candidate_emission"] == "forbidden"
    assert cfg["artifact_policy"]["success_rate_evaluation"] == "forbidden"


def test_direction_gate_exactly_reuses_locked_panel_bootstrap_and_thresholds():
    cfg = read_config(CONFIG)
    gate = cfg["outcome_task_balanced_positive_advantage"]["direction_gate"]
    trigger_path = ROOT / gate["trigger"]["report_path"]
    report = json.loads(trigger_path.read_text())
    assert _sha256(trigger_path) == gate["trigger"]["report_sha256"] == (
        "91896553aa34aebf6ee6b5cfa46670c04b5bcbc54b9132d8a78818d7fc5cd0b7"
    )
    assert report["kind"] == gate["trigger"]["required_kind"]
    assert report["format_version"] == gate["trigger"]["required_format_version"]
    assert report["execution_validated"] is gate["trigger"]["required_execution_validated"]
    assert report["status"] == gate["trigger"]["required_status"]
    assert report["decision"]["passed"] is gate["trigger"]["required_decision_passed"]
    panel = gate["heldout_panel"]
    scalar = panel["signed_grpo_scalar"]
    assert scalar["api"] == "outcome_grpo_v2.sampled_group_objectives_v2.grpo"
    assert scalar["atom"] == "-min(rho*A,clamp(rho,0.8,1.2)*A)"
    assert scalar["arms"] == list(range(1, 8))
    assert scalar["benefit_cosine"] == (
        "negative_cosine_of_heldout_gradient_and_update_delta"
    )
    assert panel["group_receipt_sha256"] == (
        report["outcome_blind_panel"]["group_receipt"]["sha256"]
    )
    assert panel["group_receipt_sha256"] == (
        "924e28cb96d49ff581ad5907fe8069ccba76f3aff404a3a75e434dcd90c0e329"
    )
    assert panel["sampling_receipt_sha256"] == (
        report["outcome_blind_panel"]["sampling_receipt"]["sha256"]
    )
    assert panel["sampling_receipt_sha256"] == (
        "63060e11f2bc8382ef05fecb702e19e155070b4e5f1a1bd78dbf7fc039fbd10a"
    )
    assert gate["bootstrap"]["matrix_sha256"] == (
        report["bootstrap_resample_matrix"]["sha256"]
    )
    assert gate["bootstrap"]["matrix_sha256"] == (
        "1e570b6d13426c8fbd58016d0fba6869dc18aa3151dfdbc0bab357373cacf32e"
    )
    assert gate["direction_families"]["primary_endpoint_order"] == [
        row["name"] for row in report["decision"]["primary_endpoints"]
    ]
    assert gate["direction_families"]["production_adamw_increment_order"] == [
        row["name"]
        for row in report["decision"]["production_adamw_increment_catastrophes"]
    ]
    assert gate["direction_families"]["direct_first_repeat"][
        "maximum_relative_residual"
    ] == 1.0e-7
    thresholds = gate["locked_thresholds"]
    inherited = report["decision"]["threshold_inheritance"]
    assert thresholds["endpoint_benefit_cosine_min"] == (
        inherited["minimum_endpoint_benefit_cosine"]
    )
    assert thresholds["catastrophic_increment_benefit_cosine_min_exclusive"] == (
        inherited["maximum_catastrophic_wrong_way_benefit_cosine"]
    )
    assert thresholds["reference_gradient_relative_bound_max"] == (
        inherited["reference_gradient_relative_bound"]
    )


def test_direction_gate_fail_closed_classification_is_frozen():
    gate = read_config(CONFIG)["outcome_task_balanced_positive_advantage"][
        "direction_gate"
    ]
    classification = gate["failure_classification"]
    assert classification["complete_finite_zero_PA_or_full_direction"] == (
        "SCIENTIFIC_ABORT_UNDEFINED_REQUIRED_COSINE"
    )
    assert classification[
        "complete_finite_nonzero_reference_vjp_after_exact_identity"
    ] == "SCIENTIFIC_ABORT"
    assert all(value == "INVALID_NO_REPORT" for key, value in classification.items()
               if key not in {
                   "complete_finite_zero_PA_or_full_direction",
                   "complete_finite_nonzero_reference_vjp_after_exact_identity",
               })


def test_demo_anchor_construction_adapter_is_exact_and_sparse_graph_stays_absent():
    cfg = read_config(CONFIG)
    method = cfg["outcome_task_balanced_positive_advantage"]
    anchor = method["frozen_objective"]["demo_anchor"]
    source = anchor["construction_config_source"]
    assert source == {
        "path": "configs/r0a_outcome_grpo_v2_pilot.yaml",
        "raw_sha256": "249a221d84032f8c3801a7430e48597a364fa16b02b378e9a17d30c8a56cdf44",
        "resolved_hash": "67277938c51075d2",
    }
    assert anchor["canonical_task_balanced_target"] == {
        "enabled": False, "weight": 0.0,
    }
    assert anchor["construction_only_target"] == {
        "enabled": True,
        "weight": None,
        "mode": "sparse_ce",
        "temperature": 1.0,
        "detach_belief": True,
    }
    assert anchor["v2_core_internal_target_producer_weight"] == 1.0
    assert anchor["canonical_config_mutated"] is False
    assert anchor["sparse_ce_scalar_computed"] is False
    assert anchor["sparse_ce_graph_constructed"] is False
    assert cfg["losses"]["proposal"]["enabled"] is False
    assert cfg["losses"]["proposal"]["weight"] == 0.0


def test_core_provenance_and_static_surface_exclude_retired_objectives_and_runner():
    provenance = tb.core_provenance()
    assert provenance["trainer_wired"] is False
    assert provenance["launcher_present"] is False
    assert provenance["candidate_or_evaluation_authority"] is False
    objective = provenance["objective"]
    assert objective["positive_advantage"] == 1.0
    assert objective["task_balanced_recovery_k3"] == 1.0
    assert objective["analytic_demo_reference"] == 1.0
    assert objective["grpo"] == objective["switch_balance"] == objective["sparse_ce"] == 0.0
    source = inspect.getsource(tb)
    for forbidden in (
        "optimizer.step(", "dist.init_process_group", "torch.distributed",
        "save_checkpoint", "candidate.pt", "official_evaluation",
    ):
        assert forbidden not in source
    assert not (ROOT / "scripts" / "train_outcome_task_balanced_positive_advantage.py").exists()
    assert not (ROOT / "scripts" / "outcome_task_balanced_positive_advantage.sbatch").exists()


def test_core_source_identity_is_pinned_by_recipe():
    cfg = read_config(CONFIG)
    source = tb.core_source_identity()
    pins = cfg["outcome_task_balanced_positive_advantage"]["source_provenance"]
    assert pins["core_file_sha256"] == _sha256(
        ROOT / "loom" / "train" / "outcome_task_balanced_positive_advantage.py"
    )
    assert pins["core_closure_sha256"] == source["sha256"]
    tb.assert_core_source_identity(source)
