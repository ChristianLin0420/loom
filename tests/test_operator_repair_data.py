"""Focused contracts for the operator-repair sampler and dynamic history."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
import torch

import contracts as C
from loom.data import cache as CA
from loom.data import canonical as CN
from loom.data.loader import CachedWindowDataset, HomogeneousSampler, collate_window


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
PREFIXES = (0, 4, 8, 12)


def _suite_sampler(seed: int = 17) -> tuple[HomogeneousSampler, dict]:
    """Eight synthetic tasks whose indices remain identifiable after filtering."""
    baseline: dict[str, np.ndarray] = {}
    owner: dict[int, tuple[str, str]] = {}
    cursor = 0
    for suite in SUITES:
        for task_index in range(2):
            task = f"{suite}/task_{task_index}"
            indices = np.arange(cursor, cursor + 40, dtype=np.int64)
            baseline[task] = indices
            owner.update({int(index): (suite, task) for index in indices})
            cursor += len(indices)

    by_prefix = {
        burn_in: {
            task: indices[burn_in // 4 :]
            for task, indices in baseline.items()
        }
        for burn_in in PREFIXES
    }
    sampler = HomogeneousSampler(
        {"libero_franka": cursor},
        batch_size=2,
        world_size=2,
        seed=seed,
        sampling="weighted_suite_task",
        task_indices_by_burn_in={"libero_franka": by_prefix},
        suite_weights={
            "libero_spatial": 0.2,
            "libero_object": 0.2,
            "libero_goal": 0.2,
            "libero_10": 0.4,
        },
        suite_block=20,
        recurrent_prefix_choices=PREFIXES,
    )
    return sampler, {"owner": owner, "by_prefix": by_prefix}


def test_weighted_suite_schedule_is_exact_rank_shared_and_prefix_eligible() -> None:
    sampler, meta = _suite_sampler()
    suite_counts: Counter[str] = Counter()

    for step in range(20):
        specs = [sampler.batch_spec_at(step, rank) for rank in range(2)]
        assert {spec.body for spec in specs} == {"libero_franka"}
        assert len({spec.suite for spec in specs}) == 1
        assert len({spec.burn_in for spec in specs}) == 1
        suite = specs[0].suite
        burn_in = specs[0].burn_in
        assert suite is not None and burn_in in PREFIXES
        suite_counts[suite] += 1

        # A global step is one homogeneous suite/history batch and rank slices
        # cannot alias one another.
        assert set(specs[0].indices.tolist()).isdisjoint(specs[1].indices.tolist())
        eligible = {
            int(index)
            for task, indices in meta["by_prefix"][burn_in].items()
            if task.startswith(f"{suite}/")
            for index in indices
        }
        for spec in specs:
            assert set(spec.indices.tolist()) <= eligible
            assert {
                meta["owner"][int(index)][0] for index in spec.indices
            } == {suite}

    assert suite_counts == {
        "libero_spatial": 4,
        "libero_object": 4,
        "libero_goal": 4,
        "libero_10": 8,
    }


def test_weighted_sampler_random_access_resume_is_pure() -> None:
    sequential, _ = _suite_sampler(seed=91)
    resumed, _ = _suite_sampler(seed=91)
    for step in (0, 1, 19, 20, 137, 19_999, 31_999):
        for rank in range(2):
            expected = sequential.batch_spec_at(step, rank)
            actual = resumed.batch_spec_at(step, rank)
            assert actual.body == expected.body
            assert actual.suite == expected.suite
            assert actual.burn_in == expected.burn_in
            np.testing.assert_array_equal(actual.indices, expected.indices)


def test_weighted_sampler_never_duplicates_across_ranks_at_pool_rollover() -> None:
    sampler, _ = _suite_sampler(seed=23)
    # Traverse far beyond the smallest 37-window prefix-aware task pool, so
    # every suite/prefix/task stream crosses several wrap boundaries.
    for step in range(2_000):
        global_indices = np.concatenate([
            sampler.batch_spec_at(step, rank).indices for rank in range(2)
        ])
        assert len(global_indices) == len(set(global_indices.tolist())), step


def test_weighted_sampler_rejects_pool_too_small_for_worst_cycle_span() -> None:
    def build(per_task: int) -> HomogeneousSampler:
        pools = {
            f"suite/task_{task}": np.arange(
                task * per_task, (task + 1) * per_task, dtype=np.int64,
            )
            for task in range(10)
        }
        return HomogeneousSampler(
            {"libero_franka": 10 * per_task},
            batch_size=8,
            world_size=16,
            seed=0,
            sampling="weighted_suite_task",
            task_indices_by_burn_in={"libero_franka": {0: pools}},
            suite_weights={"suite": 1.0},
            suite_block=1,
            recurrent_prefix_choices=(0,),
        )

    # P=128,n=10 and starts at multiples of P can touch 14 task cycles.
    with pytest.raises(ValueError, match="provide 14 distinct windows"):
        build(13)
    sampler = build(14)
    for step in range(50):
        indices = np.concatenate([
            sampler.batch_spec_at(step, rank).indices for rank in range(16)
        ])
        assert len(indices) == len(set(indices.tolist()))


def _dynamic_dataset(root) -> CachedWindowDataset:
    rng = np.random.default_rng(9)
    trajectory = CN.to_canonical(
        n_src_frames=220,
        src_fps=20.0,
        embodiment="libero_franka",
        traj_id="libero_long/task/demo_0",
        actions=rng.normal(size=(220, 7)),
        lang="do a long task",
    )
    frames = CN.required_source_frames(CN.segment(trajectory))[trajectory.traj_id]
    spec = CA.CacheSpec("fp16", 2, 4, 8, 7, 3)
    with CA.FeatureCacheWriter(root, spec) as writer:
        writer.write(
            trajectory.traj_id,
            frames=frames,
            views=rng.normal(size=(len(frames), 2, 4, 8)),
            proprio=rng.normal(size=(len(frames), 7)),
            lang=rng.normal(size=(3, 8)),
            embodiment="libero_franka",
            src_fps=20.0,
            meta={"suite": "libero_long", "task": "task"},
        )
    return CachedWindowDataset([trajectory], CA.FeatureCache(root))


def test_dynamic_prefix_loads_exact_history_without_deleting_early_windows(tmp_path) -> None:
    dataset = _dynamic_dataset(tmp_path / "cache")
    original_size = len(dataset)
    index = next(
        i for i, window in enumerate(dataset.windows)
        if window.start == 12 * C.H_OP
    )

    plain = dataset[index]
    dynamic = dataset[(index, 12, "libero_long")]
    assert len(dataset) == original_size
    assert "burn_in_feats" not in plain
    assert len(dynamic["burn_in_feats"]) == 12
    assert dynamic["burn_in_steps"] == 12
    assert dynamic["sampling_suite"] == "libero_long"
    assert torch.equal(dynamic["actions"], plain["actions"])
    for got, expected in zip(dynamic["feats"], plain["feats"]):
        assert torch.equal(got["views"], expected["views"])
        assert torch.equal(got["proprio"], expected["proprio"])

    batch = collate_window([dynamic, dataset[(index + 1, 12, "libero_long")]])
    assert batch["burn_in_steps"] == 12
    assert batch["sampling_suite"] == "libero_long"
    assert len(batch["burn_in_feats"]) == 12

    with pytest.raises(ValueError, match="cannot supply burn_in"):
        dataset[(0, 12, "libero_long")]
