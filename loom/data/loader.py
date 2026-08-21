"""
LOOM — embodiment-homogeneous batching.

**Every minibatch contains exactly one embodiment.** That fixes ``dof``, proprio
width, action normalisation and head dispatch for the whole step, and it
eliminates action padding and masking outright — there is no ragged action
tensor to mask because there is only one body in the batch. Embodiments mix
*between* batches, never *within*. LIBERO is single-embodiment; the dispatch is
built and tested with two synthetic bodies anyway (PLAN §4.A).

Which embodiment a step draws is a function of ``(seed, global_step)`` **only,
never of rank** — every rank must agree, or ranks would run different
per-embodiment heads at the same step and collectives would deadlock. *Which
windows* it draws is a function of ``(seed, global_step, rank)``.

There is no stateful sampler
----------------------------
Sampling is a pure function. ``sampler.batch_at(step, rank)`` is the whole
interface; nothing accumulates. Resume therefore needs only ``global_step``,
which is exactly why Team D's checkpoint deliberately stores no sampler cursor.
Within one embodiment the default ``uniform_window`` schedule is
epoch-permutation based: for a given epoch every rank takes a disjoint slice,
so an epoch is covered with no duplicates within a rank and none across ranks
either. ``uniform_task`` instead cycles uniformly over task identities and
draws without replacement from each task's own window pool. Both modes are pure
functions of ``(seed, global_step, rank)`` and therefore resume without a
cursor.

The schedule block
------------------
Interleaving is periodic with period ``block``: each block holds a fixed,
weight-apportioned count per body, permuted by ``(seed, block_index)``. This is
what makes ``local_step`` — how many times body *e* has been drawn before
global step *t* — O(block) to evaluate instead of O(t). Resume at step 400k
costs microseconds, not a replay.

Throughput
----------
``measure_throughput`` and ``Throughput.sustains`` are the PLAN §4.A gate: the
pipeline must sustain ≥1.3x measured training consumption or the GPUs starve.
The consumption rate is configurable (``LOOM_TRAIN_STEP_HZ``) because it is a
property of the model and the batch size, not of this module.

Shared memory
-------------
DataLoader workers pass tensors to the main process through ``/dev/shm``. A
2-stream LIBERO window is ~4.3 MiB, so a batch of 8 with 4 workers and prefetch
2 needs ~1 GiB in flight. This login node has a 64 MiB ``/dev/shm`` and the
failure mode is "DataLoader worker exited unexpectedly / bus error", which reads
like a code bug and is not one. Both torch sharing strategies live in
``/dev/shm``, so switching strategy does not help — ``fit_workers`` shrinks the
worker/prefetch queue to what fits, down to an in-process loader, and
``shm_headroom`` reports the requirement so an sbatch can size the node instead.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from contracts import DEPTH, EMBODIMENTS, H_OP, N_STATES, ObsFeats, TransitionWindow

from .cache import CacheFormatError, FeatureCache
from .canonical import CanonicalTrajectory, WindowIndex, segment, window_actions

__all__ = [
    "STARVATION_MARGIN", "DEFAULT_CONSUMPTION_HZ",
    "SAMPLING_MODES", "TRAJECTORY_SPLITS",
    "BatchSpec", "HomogeneousSampler", "CachedWindowDataset", "collate_window",
    "LoomLoader", "Throughput", "measure_throughput",
    "shm_free_bytes", "shm_headroom", "fit_workers",
    "build_loader", "build_gate_loader", "resolve_cache_root", "DataConfigError",
]

#: PLAN §4.A: the pipeline must produce at least this multiple of what training eats.
STARVATION_MARGIN = 1.3

#: Training steps/s the loader must feed. A property of the model and the batch
#: size, not of this module, so it stays configurable.
#:
#: MEASURED: the R0-A smoke (job 32245392, 1x A100-80GB, batch_per_gpu=8, all
#: real modules) ran at **1.22 it/s**, i.e. ~9.8 windows/s ~= 62 MiB/s of
#: features. The 5.0 default is deliberately left ~4x above that: the gate then
#: asserts a stricter requirement than R0-A actually imposes, and it does not
#: have to be revisited when the model gets faster on 16 GPUs.
DEFAULT_CONSUMPTION_HZ = float(os.environ.get("LOOM_TRAIN_STEP_HZ", "5.0"))

#: ``uniform_window`` preserves the original flat-window data distribution.
#: ``uniform_task`` matches LIBERO's macro task metric: task -> window, both
#: uniformly. The config spelling is intentionally closed so a typo cannot
#: silently launch the old recipe.
SAMPLING_MODES = ("uniform_window", "uniform_task", "weighted_suite_task")

#: Whole-trajectory selection is separate from window sampling. ``all`` is the
#: backward-compatible default; ``train`` and ``gate`` are complements defined
#: by exact demo leaf names such as LIBERO's ``demo_49``.
TRAJECTORY_SPLITS = ("all", "train", "gate")


# ═══════════════════════════════════════════════════════════════════════════
#  DETERMINISTIC SAMPLING
# ═══════════════════════════════════════════════════════════════════════════

def _seed_of(*parts: object) -> int:
    """Stable 64-bit seed from arbitrary parts.

    ``hash()`` is salted per process (PYTHONHASHSEED) and would make the sampler
    non-reproducible across ranks and across restarts. blake2b is not.
    """
    payload = "\x1f".join(repr(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def _apportion(weights: np.ndarray, block: int) -> np.ndarray:
    """Largest-remainder apportionment of `block` slots, min 1 slot per body."""
    n = len(weights)
    if block < n:
        raise ValueError(f"schedule block {block} is smaller than the {n} bodies")
    w = np.asarray(weights, dtype=np.float64)
    if (w <= 0).any():
        raise ValueError(f"weights must be positive, got {w.tolist()}")
    raw = w / w.sum() * block
    base = np.floor(raw).astype(np.int64)
    rem = block - int(base.sum())
    if rem > 0:
        order = np.argsort(-(raw - base), kind="stable")
        base[order[:rem]] += 1
    # a body with zero slots would never be sampled at all
    while (base == 0).any():
        take = int(np.argmax(base))
        give = int(np.argmin(base))
        if base[take] <= 1:
            raise ValueError(f"cannot give every body a slot in a block of {block}")
        base[take] -= 1
        base[give] += 1
    return base


@dataclass(frozen=True)
class BatchSpec:
    """One rank's deterministic batch plus rank-shared method metadata."""

    body: str
    indices: np.ndarray
    suite: str | None = None
    burn_in: int = 0


class HomogeneousSampler:
    """`(seed, global_step, rank) -> (embodiment, local window indices)`. Pure.

    ``sizes`` maps embodiment -> number of windows. ``batch_size`` is per rank.
    """

    def __init__(
        self,
        sizes: Mapping[str, int],
        batch_size: int,
        world_size: int = 1,
        seed: int = 0,
        weights: Mapping[str, float] | None = None,
        block: int = 64,
        sampling: str = "uniform_window",
        task_indices: Mapping[str, Mapping[str, Sequence[int]]] | None = None,
        task_indices_by_burn_in: Mapping[
            str, Mapping[int, Mapping[str, Sequence[int]]]
        ] | None = None,
        suite_weights: Mapping[str, float] | None = None,
        suite_block: int = 20,
        recurrent_prefix_choices: Sequence[int] = (0,),
    ) -> None:
        if not sizes:
            raise ValueError("no bodies")
        if batch_size <= 0 or world_size <= 0:
            raise ValueError("batch_size and world_size must be positive")
        self.bodies = tuple(sizes.keys())
        self.sizes = {k: int(v) for k, v in sizes.items()}
        self.batch_size = int(batch_size)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.block = int(block)
        self.per_step = self.batch_size * self.world_size
        self.sampling = str(sampling)
        if self.sampling not in SAMPLING_MODES:
            raise ValueError(
                f"sampling must be one of {SAMPLING_MODES}, got {self.sampling!r}"
            )

        for name, n in self.sizes.items():
            if n < self.per_step:
                raise ValueError(
                    f"{name}: {n} windows cannot fill one global batch of "
                    f"{self.per_step} (batch_size {batch_size} x world_size {world_size})"
                )

        w = np.array(
            [float(weights[b]) if weights else float(self.sizes[b]) for b in self.bodies],
            dtype=np.float64,
        )
        self.counts = _apportion(w, self.block)           # slots per body per block
        self._perm_memo: dict[tuple[str, int], np.ndarray] = {}
        self._task_order_memo: dict[tuple[str, int], np.ndarray] = {}
        self._task_perm_memo: dict[tuple[str, int, int], np.ndarray] = {}
        self._task_names: dict[str, tuple[str, ...]] = {}
        self._task_pools: dict[str, tuple[np.ndarray, ...]] = {}
        if self.sampling == "uniform_task":
            self._init_task_pools(task_indices)
        self.recurrent_prefix_choices = tuple(
            int(value) for value in recurrent_prefix_choices
        )
        if (
            not self.recurrent_prefix_choices
            or any(value < 0 for value in self.recurrent_prefix_choices)
            or len(set(self.recurrent_prefix_choices)) != len(self.recurrent_prefix_choices)
        ):
            raise ValueError(
                "recurrent_prefix_choices must be distinct non-negative integers"
            )
        self.suite_block = int(suite_block)
        self._suite_names: dict[str, tuple[str, ...]] = {}
        self._suite_counts: dict[str, np.ndarray] = {}
        self._suite_task_names: dict[tuple[str, str], tuple[str, ...]] = {}
        self._suite_task_pools: dict[
            tuple[str, int, str], tuple[np.ndarray, ...]
        ] = {}
        self._suite_task_perm_memo: dict[
            tuple[str, str, str, int], np.ndarray
        ] = {}
        if self.sampling == "weighted_suite_task":
            self._init_suite_task_pools(
                task_indices_by_burn_in, suite_weights=suite_weights,
            )

    def _init_task_pools(
        self, task_indices: Mapping[str, Mapping[str, Sequence[int]]] | None
    ) -> None:
        """Validate one exact partition of every body's windows into tasks."""
        if task_indices is None:
            raise ValueError("sampling='uniform_task' requires task_indices")
        for body, size in self.sizes.items():
            groups = task_indices.get(body)
            if not groups:
                raise ValueError(f"{body}: uniform_task sampling found no task groups")
            names = tuple(sorted(str(k) for k in groups))
            pools = tuple(np.asarray(groups[name], dtype=np.int64).reshape(-1) for name in names)
            if any(pool.size == 0 for pool in pools):
                empty = [name for name, pool in zip(names, pools) if pool.size == 0]
                raise ValueError(f"{body}: empty task window pools {empty}")
            flat = np.concatenate(pools)
            if flat.size != size or not np.array_equal(np.sort(flat), np.arange(size)):
                raise ValueError(
                    f"{body}: task window pools must partition indices [0, {size}); "
                    f"got {flat.size} entries"
                )
            self._task_names[body] = names
            self._task_pools[body] = pools

    def _init_suite_task_pools(
        self,
        task_indices_by_burn_in: Mapping[
            str, Mapping[int, Mapping[str, Sequence[int]]]
        ] | None,
        *,
        suite_weights: Mapping[str, float] | None,
    ) -> None:
        """Validate suite -> task -> eligible-window pools for every prefix."""
        if task_indices_by_burn_in is None:
            raise ValueError(
                "sampling='weighted_suite_task' requires task_indices_by_burn_in"
            )
        if not suite_weights:
            raise ValueError(
                "sampling='weighted_suite_task' requires data.suite_weights"
            )
        configured_suites = tuple(sorted(str(name) for name in suite_weights))
        configured_weights = np.asarray(
            [float(suite_weights[name]) for name in configured_suites],
            dtype=np.float64,
        )
        counts = _apportion(configured_weights, self.suite_block)
        for body, size in self.sizes.items():
            by_prefix = task_indices_by_burn_in.get(body)
            if not by_prefix:
                raise ValueError(f"{body}: no prefix-aware task pools")
            baseline = by_prefix.get(0)
            if not baseline:
                raise ValueError(f"{body}: prefix-aware pools omit burn_in=0")
            suites_present = tuple(sorted({str(task).split("/", 1)[0]
                                           for task in baseline}))
            if suites_present != configured_suites:
                raise ValueError(
                    f"{body}: configured suites {configured_suites} do not match "
                    f"task suites {suites_present}"
                )
            flat0 = np.concatenate([
                np.asarray(indices, dtype=np.int64).reshape(-1)
                for indices in baseline.values()
            ])
            if flat0.size != size or not np.array_equal(
                np.sort(flat0), np.arange(size)
            ):
                raise ValueError(
                    f"{body}: burn_in=0 task pools must partition [0, {size})"
                )
            self._suite_names[body] = configured_suites
            self._suite_counts[body] = counts.copy()
            for suite in configured_suites:
                task_names = tuple(sorted(
                    task for task in baseline if task.split("/", 1)[0] == suite
                ))
                if not task_names:
                    raise ValueError(f"{body}: suite {suite!r} has no tasks")
                self._suite_task_names[(body, suite)] = task_names
                for burn_in in self.recurrent_prefix_choices:
                    groups = by_prefix.get(burn_in)
                    if groups is None:
                        raise ValueError(
                            f"{body}: task pools omit burn_in={burn_in}"
                        )
                    pools = tuple(
                        np.asarray(groups.get(task, ()), dtype=np.int64).reshape(-1)
                        for task in task_names
                    )
                    if any(pool.size == 0 for pool in pools):
                        missing = [task for task, pool in zip(task_names, pools)
                                   if pool.size == 0]
                        raise ValueError(
                            f"{body}: burn_in={burn_in} empties tasks {missing}"
                        )
                    # A single distributed batch must never reuse a window on
                    # two ranks, including when its ordinal interval straddles
                    # several independently shuffled task-order cycles.
                    # A P-long ordinal interval begins at a multiple of P. Its
                    # start residue modulo n_tasks ranges over multiples of
                    # gcd(P, n_tasks), so it can touch this many task-order
                    # cycles in the worst case. Independent order permutations
                    # may select the same task once in every touched cycle.
                    n_tasks = len(task_names)
                    minimum = math.ceil(
                        (
                            self.per_step + n_tasks
                            - math.gcd(self.per_step, n_tasks)
                        ) / n_tasks
                    )
                    too_small = [
                        task for task, pool in zip(task_names, pools)
                        if len(pool) < minimum
                    ]
                    if too_small:
                        raise ValueError(
                            f"{body}: burn_in={burn_in} task pools {too_small} "
                            f"cannot provide {minimum} distinct windows per "
                            "distributed batch"
                        )
                    self._suite_task_pools[(body, burn_in, suite)] = pools

    # ── schedule ─────────────────────────────────────────────────────────
    def steps_per_epoch(self, body: str) -> int:
        if self.sampling == "uniform_task":
            pools = self._task_pools[body]
            # One balanced epoch visits the largest task once and oversamples
            # every smaller task to the same number of draws.
            return len(pools) * max(len(pool) for pool in pools) // self.per_step
        return self.sizes[body] // self.per_step

    def _pattern(self, block_index: int) -> np.ndarray:
        ids = np.repeat(np.arange(len(self.bodies), dtype=np.int64), self.counts)
        rng = np.random.default_rng(_seed_of(self.seed, "schedule", block_index))
        return rng.permutation(ids)

    def embodiment_at(self, step: int) -> str:
        """Rank-independent by construction: every rank runs the same body."""
        pat = self._pattern(step // self.block)
        return self.bodies[int(pat[step % self.block])]

    def local_step(self, step: int) -> int:
        """How many times this step's body has been drawn before `step`."""
        b = step // self.block
        pos = step % self.block
        pat = self._pattern(b)
        bid = int(pat[pos])
        return b * int(self.counts[bid]) + int((pat[:pos] == bid).sum())

    def _suite_pattern(self, body: str, block_index: int) -> np.ndarray:
        counts = self._suite_counts[body]
        ids = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
        rng = np.random.default_rng(
            _seed_of(self.seed, "suite_schedule", body, block_index)
        )
        return rng.permutation(ids)

    def suite_at(self, step: int, body: str | None = None) -> str | None:
        if self.sampling != "weighted_suite_task":
            return None
        body = self.embodiment_at(step) if body is None else body
        local = self.local_step(step)
        pattern = self._suite_pattern(body, local // self.suite_block)
        return self._suite_names[body][int(pattern[local % self.suite_block])]

    def _suite_local_step(self, step: int, body: str, suite: str) -> int:
        local = self.local_step(step)
        block_index, pos = divmod(local, self.suite_block)
        pattern = self._suite_pattern(body, block_index)
        suite_id = self._suite_names[body].index(suite)
        return (
            block_index * int(self._suite_counts[body][suite_id])
            + int((pattern[:pos] == suite_id).sum())
        )

    def burn_in_at(self, step: int) -> int:
        if self.sampling != "weighted_suite_task":
            return 0
        pick = _seed_of(self.seed, "recurrent_prefix", int(step))
        return self.recurrent_prefix_choices[pick % len(self.recurrent_prefix_choices)]

    # ── indices ──────────────────────────────────────────────────────────
    def _permutation(self, body: str, epoch: int) -> np.ndarray:
        key = (body, epoch)
        perm = self._perm_memo.get(key)
        if perm is None:
            rng = np.random.default_rng(_seed_of(self.seed, "perm", body, epoch))
            perm = rng.permutation(self.sizes[body])
            if len(self._perm_memo) > 8:
                self._perm_memo.clear()
            self._perm_memo[key] = perm
        return perm

    def _task_order(self, body: str, cycle: int) -> np.ndarray:
        """One seeded permutation of all tasks; each appears once per cycle."""
        key = (body, cycle)
        order = self._task_order_memo.get(key)
        if order is None:
            rng = np.random.default_rng(_seed_of(self.seed, "task_order", body, cycle))
            order = rng.permutation(len(self._task_pools[body]))
            if len(self._task_order_memo) > 64:
                self._task_order_memo.clear()
            self._task_order_memo[key] = order
        return order

    def _task_permutation(self, body: str, task: int, epoch: int) -> np.ndarray:
        """Permutation within one task, independent of every other task's size."""
        key = (body, task, epoch)
        perm = self._task_perm_memo.get(key)
        if perm is None:
            name = self._task_names[body][task]
            rng = np.random.default_rng(_seed_of(self.seed, "task_perm", body, name, epoch))
            perm = rng.permutation(len(self._task_pools[body][task]))
            if len(self._task_perm_memo) > 64:
                self._task_perm_memo.clear()
            self._task_perm_memo[key] = perm
        return perm

    def _uniform_task_batch(self, body: str, local_step: int, rank: int) -> np.ndarray:
        """Uniform task -> uniform window for one rank's contiguous global slice.

        Global sample ordinal ``q`` is divided into cycles of ``n_tasks``. Each
        cycle contains every task exactly once in a seeded order. A task's cycle
        number is also its occurrence number, which indexes a separate seeded
        permutation of that task's windows. This gives exact task balance,
        disjoint rank slices until a per-task epoch rolls over, and O(batch)
        resume at an arbitrary step.
        """
        n_tasks = len(self._task_pools[body])
        start = local_step * self.per_step + rank * self.batch_size
        out = np.empty(self.batch_size, dtype=np.int64)
        for j, q in enumerate(range(start, start + self.batch_size)):
            cycle, pos = divmod(q, n_tasks)
            task = int(self._task_order(body, cycle)[pos])
            pool = self._task_pools[body][task]
            epoch, within = divmod(cycle, len(pool))
            pick = int(self._task_permutation(body, task, epoch)[within])
            out[j] = pool[pick]
        return out

    def _suite_task_batch(
        self,
        body: str,
        suite: str,
        burn_in: int,
        local_step: int,
        rank: int,
    ) -> np.ndarray:
        task_names = self._suite_task_names[(body, suite)]
        pools = self._suite_task_pools[(body, burn_in, suite)]
        n_tasks = len(task_names)
        start = local_step * self.per_step + rank * self.batch_size
        out = np.empty(self.batch_size, dtype=np.int64)
        for j, ordinal in enumerate(range(start, start + self.batch_size)):
            cycle, pos = divmod(ordinal, n_tasks)
            order_rng = np.random.default_rng(
                _seed_of(self.seed, "suite_task_order", body, suite, cycle)
            )
            task = int(order_rng.permutation(n_tasks)[pos])
            pool = pools[task]
            # ``cycle`` is this task's exact occurrence count because every
            # task appears once in each task-order cycle. Use one seeded cyclic
            # permutation rather than independent epoch permutations: the
            # latter can put the same window at the end of epoch e and the
            # beginning of e+1, duplicating it across ranks in one global
            # batch. A fixed cyclic stream is without replacement across that
            # boundary and still gives every window exactly equal exposure.
            key = (body, suite, task_names[task], burn_in)
            permutation = self._suite_task_perm_memo.get(key)
            if permutation is None:
                perm_rng = np.random.default_rng(
                    _seed_of(
                        self.seed, "suite_task_perm", body, suite,
                        task_names[task], burn_in,
                    )
                )
                permutation = perm_rng.permutation(len(pool))
                self._suite_task_perm_memo[key] = permutation
            out[j] = pool[int(permutation[cycle % len(pool)])]
        return out

    def batch_spec_at(self, step: int, rank: int = 0) -> BatchSpec:
        if not (0 <= rank < self.world_size):
            raise ValueError(f"rank {rank} outside world_size {self.world_size}")
        body = self.embodiment_at(step)
        local_step = self.local_step(step)
        if self.sampling == "weighted_suite_task":
            suite = self.suite_at(step, body)
            assert suite is not None
            burn_in = self.burn_in_at(step)
            suite_step = self._suite_local_step(step, body, suite)
            indices = self._suite_task_batch(
                body, suite, burn_in, suite_step, rank,
            )
            return BatchSpec(body, indices, suite=suite, burn_in=burn_in)
        if self.sampling == "uniform_task":
            indices = self._uniform_task_batch(body, local_step, rank)
        else:
            spe = self.steps_per_epoch(body)
            perm = self._permutation(body, local_step // spe)
            k = local_step % spe
            lo = (k * self.world_size + rank) * self.batch_size
            indices = perm[lo:lo + self.batch_size]
        return BatchSpec(body, indices)

    def batch_at(self, step: int, rank: int = 0) -> tuple[str, np.ndarray]:
        spec = self.batch_spec_at(step, rank)
        return spec.body, spec.indices


# ═══════════════════════════════════════════════════════════════════════════
#  DATASET
# ═══════════════════════════════════════════════════════════════════════════

def _libero_trajectory_identity(traj_id: str) -> tuple[str, str]:
    """Return (suite/task, demo key) from the adapter's canonical id."""
    parts = str(traj_id).split("/")
    if len(parts) != 3 or any(not part for part in parts):
        raise DataConfigError(
            "LIBERO whole-trajectory splitting requires canonical ids "
            f"'suite/task/demo_key'; got {traj_id!r}"
        )
    return f"{parts[0]}/{parts[1]}", parts[2]


def _trajectory_split_config(dcfg: Mapping) -> tuple[str, tuple[str, ...], bool]:
    """Validate and canonicalise the optional whole-trajectory split block."""
    configured = "trajectory_split" in dcfg or "holdout_demo_keys" in dcfg
    split = str(dcfg.get("trajectory_split", "all"))
    if split not in TRAJECTORY_SPLITS:
        raise DataConfigError(
            f"data.trajectory_split must be one of {TRAJECTORY_SPLITS}, got {split!r}"
        )

    raw = dcfg.get("holdout_demo_keys", ())
    if raw is None:
        raw = ()
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise DataConfigError(
            "data.holdout_demo_keys must be a list of exact demo leaf names, "
            f"for example [demo_49]; got {raw!r}"
        )
    keys = tuple(sorted(str(key) for key in raw))
    if any(not key or "/" in key for key in keys):
        raise DataConfigError(
            "data.holdout_demo_keys entries must be non-empty demo leaf names "
            f"without '/'; got {list(keys)!r}"
        )
    if len(set(keys)) != len(keys):
        raise DataConfigError(
            f"data.holdout_demo_keys contains duplicates: {list(keys)!r}"
        )
    if split != "all" and not keys:
        raise DataConfigError(
            f"data.trajectory_split={split!r} requires non-empty "
            "data.holdout_demo_keys"
        )
    return split, keys, configured


def _libero_manifest(
    trajectories: Sequence[CanonicalTrajectory],
    *,
    split: str,
    holdout_demo_keys: Sequence[str],
) -> dict:
    """Stable, path-independent provenance for a selected trajectory set."""
    tasks: dict[str, list[str]] = {}
    for traj in sorted(trajectories, key=lambda item: item.traj_id):
        task_id, _ = _libero_trajectory_identity(traj.traj_id)
        tasks.setdefault(task_id, []).append(traj.traj_id)
    tasks = {task: sorted(ids) for task, ids in sorted(tasks.items())}
    trajectory_ids = sorted(t.traj_id for t in trajectories)
    payload = {
        "version": 1,
        "source": "libero",
        "split": split,
        "holdout_demo_keys": sorted(str(key) for key in holdout_demo_keys),
        "n_tasks": len(tasks),
        "n_trajectories": len(trajectory_ids),
        "tasks": tasks,
        "trajectory_ids": trajectory_ids,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["digest"] = f"sha256:{hashlib.sha256(encoded).hexdigest()}"
    return payload


def _select_libero_trajectories(
    trajectories: Sequence[CanonicalTrajectory],
    *,
    split: str,
    holdout_demo_keys: Sequence[str],
    expected_task_ids: Sequence[str] | None = None,
) -> tuple[list[CanonicalTrajectory], dict]:
    """Select an exact whole-trajectory complement and return its manifest.

    Selection never depends on adapter order, rank, seed, or sampler state. For
    ``train``/``gate``, every expected task must contain every requested demo
    key and must retain at least one non-heldout trajectory. This deliberately
    makes ``max_demos: 49`` with ``demo_49`` fail instead of silently leaking or
    constructing an incomplete gate.
    """
    if split not in TRAJECTORY_SPLITS:
        raise DataConfigError(
            f"trajectory split must be one of {TRAJECTORY_SPLITS}, got {split!r}"
        )
    keys = tuple(sorted(str(key) for key in holdout_demo_keys))
    ordered = sorted(trajectories, key=lambda item: item.traj_id)
    ids = [traj.traj_id for traj in ordered]
    if len(ids) != len(set(ids)):
        seen: set[str] = set()
        dupes: set[str] = set()
        for traj_id in ids:
            if traj_id in seen:
                dupes.add(traj_id)
            else:
                seen.add(traj_id)
        raise DataConfigError(f"duplicate LIBERO trajectory ids: {sorted(dupes)}")

    by_task: dict[str, list[tuple[CanonicalTrajectory, str]]] = {}
    for traj in ordered:
        task_id, demo_key = _libero_trajectory_identity(traj.traj_id)
        by_task.setdefault(task_id, []).append((traj, demo_key))
    expected = set(str(task) for task in (expected_task_ids or by_task))
    missing_tasks = sorted(expected - set(by_task))
    if missing_tasks:
        raise DataConfigError(
            "LIBERO cache is missing every trajectory for expected tasks: "
            + ", ".join(missing_tasks)
        )

    if split in ("train", "gate"):
        wanted = set(keys)
        missing_keys: dict[str, list[str]] = {}
        no_train: list[str] = []
        for task_id in sorted(expected):
            present = {demo_key for _, demo_key in by_task[task_id]}
            missing = sorted(wanted - present)
            if missing:
                missing_keys[task_id] = missing
            if not (present - wanted):
                no_train.append(task_id)
        if missing_keys:
            detail = "; ".join(
                f"{task}: {missing}" for task, missing in missing_keys.items()
            )
            raise DataConfigError(
                "configured LIBERO holdout demos are absent from task(s) "
                f"({detail}). Check data.max_demos and cache completeness."
            )
        if no_train:
            raise DataConfigError(
                "LIBERO holdout leaves no training trajectory for task(s): "
                + ", ".join(no_train)
            )
        selected = [
            traj for traj in ordered
            if ((_libero_trajectory_identity(traj.traj_id)[1] in wanted)
                == (split == "gate"))
        ]
    else:
        selected = ordered

    selected_tasks = {
        _libero_trajectory_identity(traj.traj_id)[0] for traj in selected
    }
    if selected_tasks != expected:
        missing = sorted(expected - selected_tasks)
        raise DataConfigError(
            f"LIBERO trajectory split {split!r} omits task(s): {missing}"
        )
    return selected, _libero_manifest(
        selected, split=split, holdout_demo_keys=keys,
    )


class CachedWindowDataset:
    """Windows of ONE embodiment, features served from a ``FeatureCache``.

    Holds ``WindowIndex`` objects and canonical action arrays only; the heavy
    tensors stay on disk until ``__getitem__``.
    """

    def __init__(
        self,
        trajectories: Sequence[CanonicalTrajectory],
        cache: FeatureCache,
        stride: int = H_OP,
        recurrent_burn_in: int = 0,
        data_manifest: Mapping | None = None,
    ) -> None:
        if not trajectories:
            raise ValueError("no trajectories")
        bodies = {t.embodiment for t in trajectories}
        if len(bodies) != 1:
            raise ValueError(
                f"a CachedWindowDataset is one embodiment; got {sorted(bodies)}. "
                f"Batches are embodiment-homogeneous — build one dataset per body."
            )
        rates = {t.src_fps for t in trajectories}
        if len(rates) != 1:
            raise ValueError(
                f"mixed source rates {sorted(rates)}; TransitionWindow.src_fps is a "
                f"single float per batch, so one dataset holds one rate"
            )
        self.embodiment = trajectories[0].embodiment
        self.src_fps = float(trajectories[0].src_fps)
        self.dof = EMBODIMENTS[self.embodiment].dof
        self.cache = cache
        self.stride = int(stride)
        if (isinstance(recurrent_burn_in, bool)
                or not isinstance(recurrent_burn_in, int)
                or recurrent_burn_in < 0):
            raise ValueError(
                "recurrent_burn_in must be a non-negative integer number of "
                f"operator-boundary observations, got {recurrent_burn_in!r}"
            )
        self.recurrent_burn_in = recurrent_burn_in
        trajectory_ids = tuple(t.traj_id for t in trajectories)
        if len(set(trajectory_ids)) != len(trajectory_ids):
            raise ValueError("duplicate trajectory ids in CachedWindowDataset")
        self.trajectory_ids = trajectory_ids
        self.actions: dict[str, np.ndarray | None] = {t.traj_id: t.actions for t in trajectories}
        self._traj = {t.traj_id: t for t in trajectories}
        self._data_manifest: dict | None = None
        self._task_id_by_trajectory: dict[str, str] = {}
        if data_manifest is not None:
            # JSON round-trip both validates serialisability and prevents a
            # caller from mutating the provenance after dataset construction.
            manifest = json.loads(json.dumps(dict(data_manifest)))
            manifest_ids = tuple(manifest.get("trajectory_ids", ()))
            if manifest_ids != tuple(sorted(trajectory_ids)):
                raise ValueError(
                    "data manifest trajectory_ids do not exactly match the dataset"
                )
            for task_id, ids in manifest.get("tasks", {}).items():
                for traj_id in ids:
                    if traj_id in self._task_id_by_trajectory:
                        raise ValueError(
                            f"data manifest assigns trajectory {traj_id!r} twice"
                        )
                    self._task_id_by_trajectory[str(traj_id)] = str(task_id)
            if set(self._task_id_by_trajectory) != set(trajectory_ids):
                raise ValueError(
                    "data manifest tasks do not partition the dataset trajectories"
                )
            self._data_manifest = manifest
        self.windows: list[WindowIndex] = []
        for t in trajectories:
            windows = segment(t, stride=stride)
            if self.recurrent_burn_in:
                # Burn-in counts policy replans, whose canonical spacing is H_OP,
                # not dataset-window strides or raw source frames. Earlier windows
                # remain in the cache and supply these prefix observations.
                first = self.recurrent_burn_in * H_OP
                windows = [w for w in windows if w.start >= first]
            self.windows.extend(windows)

    def __len__(self) -> int:
        return len(self.windows)

    def trajectory_manifest(self) -> dict:
        """Return an isolated JSON-ready copy of split/source provenance."""
        if self._data_manifest is None:
            raise ValueError("this dataset was built without trajectory split provenance")
        return json.loads(json.dumps(self._data_manifest))

    def task_indices(self, burn_in: int | None = None) -> dict[str, np.ndarray]:
        """Task identity -> local window indices, using cache metadata first.

        LIBERO's cache records both suite and task, so identically named tasks in
        different suites cannot collide. ``CanonicalTrajectory.lang`` is the
        portable fallback for synthetic and future adapters; a trajectory id is
        used only when neither source supplied a task identity.
        """
        effective_burn_in = self.recurrent_burn_in if burn_in is None else int(burn_in)
        if effective_burn_in < 0:
            raise ValueError("burn_in must be non-negative")
        groups: dict[str, list[int]] = {}
        entries = getattr(self.cache, "entries", {})
        for i, window in enumerate(self.windows):
            if window.start < effective_burn_in * H_OP:
                continue
            traj = self._traj[window.traj_id]
            meta = entries.get(window.traj_id, {}).get("meta", {})
            task = str(meta.get("task") or traj.lang or window.traj_id)
            suite = str(meta.get("suite") or "")
            key = f"{suite}/{task}" if suite else task
            groups.setdefault(key, []).append(i)
        return {key: np.asarray(idx, dtype=np.int64) for key, idx in groups.items()}

    def __getitem__(self, i: int | tuple[int, int, str | None]) -> dict:
        sampling_suite: str | None = None
        burn_in = self.recurrent_burn_in
        if isinstance(i, tuple):
            if len(i) != 3:
                raise ValueError("dynamic dataset index must be (index, burn_in, suite)")
            i, burn_in, sampling_suite = i
            i, burn_in = int(i), int(burn_in)
            if burn_in < 0:
                raise ValueError("dynamic burn_in must be non-negative")
        w = self.windows[i]
        traj = self._traj[w.traj_id]
        if w.start < burn_in * H_OP:
            raise ValueError(
                f"window start {w.start} cannot supply burn_in={burn_in}"
            )
        prefix_src = tuple(
            int(traj.obs_src_index[t])
            for t in range(
                w.start - burn_in * H_OP,
                w.start,
                H_OP,
            )
        )
        blob = self.cache.read(w.traj_id, prefix_src + w.obs_src_index)
        views = torch.from_numpy(np.ascontiguousarray(blob["views"]))
        proprio = torch.from_numpy(np.ascontiguousarray(blob["proprio"]))
        lang = torch.from_numpy(np.ascontiguousarray(blob["lang"]))         # (L,F)
        all_feats = [
            ObsFeats(views=views[s], proprio=proprio[s], lang=lang)
            for s in range(burn_in + N_STATES)
        ]
        a = window_actions(traj, w)
        out = {
            "feats": all_feats[burn_in:],
            "actions": None if a is None else torch.from_numpy(np.ascontiguousarray(a)),
            "lang": lang,
            "embodiment": w.embodiment,
            "src_fps": w.src_fps,
        }
        if self._data_manifest is not None:
            out["data_meta"] = {
                "source": self._data_manifest["source"],
                "split": self._data_manifest["split"],
                "manifest_digest": self._data_manifest["digest"],
                "task_id": self._task_id_by_trajectory[w.traj_id],
                "trajectory_id": w.traj_id,
                # Bootstrap/resampling clusters are complete demonstrations,
                # never individual windows from the same demonstration.
                "trajectory_cluster_id": w.traj_id,
            }
        # Omit the optional key at B=0 so every existing source and consumer sees
        # precisely the historical TransitionWindow dictionary.
        if burn_in:
            out["burn_in_feats"] = all_feats[:burn_in]
        if sampling_suite is not None:
            out["sampling_suite"] = str(sampling_suite)
            out["burn_in_steps"] = burn_in
        return out


def collate_window(samples: Sequence[dict]) -> TransitionWindow:
    """Stack per-sample dicts into one ``contracts.TransitionWindow``.

    Asserts homogeneity rather than padding. If this ever fires, the sampler is
    broken — not the collate.
    """
    bodies = {s["embodiment"] for s in samples}
    if len(bodies) != 1:
        raise ValueError(f"batch mixes embodiments {sorted(bodies)}; batches are homogeneous")
    rates = {s["src_fps"] for s in samples}
    if len(rates) != 1:
        raise ValueError(f"batch mixes source rates {sorted(rates)}")
    body = samples[0]["embodiment"]

    def _stack(key: str, n: int) -> list[ObsFeats]:
        return [ObsFeats(
            views=torch.stack([x[key][s]["views"] for x in samples]),
            proprio=torch.stack([x[key][s]["proprio"] for x in samples]),
            lang=torch.stack([x[key][s]["lang"] for x in samples]),
        ) for s in range(n)]

    feats = _stack("feats", N_STATES)
    prefix_lengths = {len(x.get("burn_in_feats", ())) for x in samples}
    if len(prefix_lengths) != 1:
        raise ValueError(
            f"batch mixes recurrent burn-in lengths {sorted(prefix_lengths)}"
        )
    n_prefix = next(iter(prefix_lengths))
    action_free = [x["actions"] is None for x in samples]
    if any(action_free) and not all(action_free):
        raise ValueError("batch mixes action-labelled and action-free windows")
    actions = None if action_free[0] else torch.stack([x["actions"] for x in samples])
    out = TransitionWindow(
        feats=feats,
        actions=actions,
        lang=torch.stack([x["lang"] for x in samples]),
        embodiment=body,
        src_fps=float(samples[0]["src_fps"]),
    )
    if n_prefix:
        out["burn_in_feats"] = _stack("burn_in_feats", n_prefix)
    sampled_suites = {sample.get("sampling_suite") for sample in samples}
    burn_in_values = {int(sample.get("burn_in_steps", n_prefix)) for sample in samples}
    if len(sampled_suites) > 1 or len(burn_in_values) != 1:
        raise ValueError(
            "batch mixes suite/prefix sampling metadata: "
            f"suites={sorted(str(value) for value in sampled_suites)} "
            f"burn_in={sorted(burn_in_values)}"
        )
    sampled_suite = next(iter(sampled_suites))
    if sampled_suite is not None:
        out["sampling_suite"] = sampled_suite
        out["burn_in_steps"] = next(iter(burn_in_values))
    has_meta = ["data_meta" in sample for sample in samples]
    if any(has_meta) and not all(has_meta):
        raise ValueError("batch mixes windows with and without data provenance")
    if all(has_meta):
        metadata = [sample["data_meta"] for sample in samples]
        common_keys = ("source", "split", "manifest_digest")
        for key in common_keys:
            values = {item[key] for item in metadata}
            if len(values) != 1:
                raise ValueError(f"batch mixes data provenance {key}: {sorted(values)}")
        out["data_meta"] = {
            **{key: metadata[0][key] for key in common_keys},
            "task_ids": tuple(item["task_id"] for item in metadata),
            "trajectory_ids": tuple(item["trajectory_id"] for item in metadata),
            "trajectory_cluster_ids": tuple(
                item["trajectory_cluster_id"] for item in metadata
            ),
        }
    return out


def shm_free_bytes() -> int:
    """Free bytes on /dev/shm, 0 if it cannot be read."""
    try:
        st = os.statvfs("/dev/shm")
        return int(st.f_bavail) * int(st.f_frsize)
    except OSError:                                  # pragma: no cover
        return 0


def shm_headroom(bytes_per_batch: int, num_workers: int, prefetch_factor: int) -> tuple[int, int]:
    """(bytes needed in flight, bytes free on /dev/shm).

    DataLoader workers hand tensors to the main process through shared memory.
    A 2-stream LIBERO window is ~4.3 MiB, so 2 workers x prefetch 2 is ~35 MiB
    per in-flight batch of 4 — more than a container-default 64 MiB /dev/shm.
    Exhausting it surfaces as "DataLoader worker exited unexpectedly / bus
    error", which looks like a code bug and is not one.

    Both torch sharing strategies live in ``/dev/shm``; ``file_system`` only
    changes who unlinks the files (and leaks them on a hard kill). Shrinking the
    in-flight queue is the fix, not switching strategy.
    """
    need = int(bytes_per_batch) * max(num_workers, 0) * (max(prefetch_factor, 1) + 1)
    return need, shm_free_bytes()


def fit_workers(
    bytes_per_batch: int, num_workers: int, prefetch_factor: int, headroom: float = 0.7
) -> tuple[int, int]:
    """Largest (workers, prefetch) <= requested whose queue fits in /dev/shm.

    Returns ``(0, prefetch)`` when not even a single worker's queue fits: an
    in-process loader is slower than a starving one but it does not die at
    step 900 with a bus error.
    """
    if num_workers <= 0 or bytes_per_batch <= 0:
        return 0, prefetch_factor
    budget = headroom * shm_free_bytes()
    for w in range(num_workers, 0, -1):
        for p in range(prefetch_factor, 0, -1):
            if bytes_per_batch * w * (p + 1) <= budget:
                return w, p
    return 0, prefetch_factor


class _ConcatBodies(torch.utils.data.Dataset):
    """Global index -> (body dataset, local index). Offsets only, no copying."""

    def __init__(self, datasets: Mapping[str, CachedWindowDataset]) -> None:
        self.names = tuple(datasets.keys())
        self.datasets = [datasets[n] for n in self.names]
        self.offsets = np.cumsum([0] + [len(d) for d in self.datasets]).astype(np.int64)

    def offset_of(self, body: str) -> int:
        return int(self.offsets[self.names.index(body)])

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def __getitem__(self, i: int | tuple[int, int, str | None]) -> dict:
        burn_in, suite = 0, None
        if isinstance(i, tuple):
            if len(i) != 3:
                raise ValueError("batch index must be (global_index, burn_in, suite)")
            i, burn_in, suite = i
        i = int(i)
        j = int(np.searchsorted(self.offsets, i, side="right")) - 1
        local = i - int(self.offsets[j])
        if suite is None and burn_in == 0:
            return self.datasets[j][local]
        return self.datasets[j][(local, int(burn_in), suite)]


class _StepBatchSampler:
    """Yields global index lists for steps ``[start, start + n)``. Stateless."""

    def __init__(
        self,
        sampler: HomogeneousSampler,
        concat: _ConcatBodies,
        rank: int,
        start_step: int,
        n_steps: int,
    ) -> None:
        self.sampler, self.concat, self.rank = sampler, concat, rank
        self.start_step, self.n_steps = int(start_step), int(n_steps)

    def __len__(self) -> int:
        return self.n_steps

    def __iter__(self) -> Iterator[list[object]]:
        for t in range(self.start_step, self.start_step + self.n_steps):
            spec = self.sampler.batch_spec_at(t, self.rank)
            off = self.concat.offset_of(spec.body)
            if spec.suite is None and spec.burn_in == 0:
                yield [off + int(i) for i in spec.indices]
            else:
                yield [
                    (off + int(i), spec.burn_in, spec.suite)
                    for i in spec.indices
                ]


# ═══════════════════════════════════════════════════════════════════════════
#  LOADER
# ═══════════════════════════════════════════════════════════════════════════

class LoomLoader:
    """Batches of one embodiment each, driven by ``global_step``.

    ``state_dict`` carries ``global_step`` and nothing else: everything the
    sampler does is recomputed from ``(seed, step, rank)``.
    """

    def __init__(
        self,
        datasets: Mapping[str, CachedWindowDataset],
        batch_size: int,
        world_size: int = 1,
        rank: int = 0,
        seed: int = 0,
        weights: Mapping[str, float] | None = None,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        pin_memory: bool = False,
        block: int = 64,
        sampling: str = "uniform_window",
        suite_weights: Mapping[str, float] | None = None,
        suite_block: int = 20,
        recurrent_prefix_choices: Sequence[int] = (0,),
    ) -> None:
        for name, ds in datasets.items():
            if ds.embodiment != name:
                raise ValueError(f"dataset keyed {name!r} holds body {ds.embodiment!r}")
        self.datasets = dict(datasets)
        self.concat = _ConcatBodies(self.datasets)
        task_indices = (
            {name: ds.task_indices() for name, ds in self.datasets.items()}
            if sampling == "uniform_task" else None
        )
        prefix_choices = tuple(int(value) for value in recurrent_prefix_choices)
        task_indices_by_burn_in = (
            {
                name: {
                    burn_in: ds.task_indices(burn_in)
                    for burn_in in prefix_choices
                }
                for name, ds in self.datasets.items()
            }
            if sampling == "weighted_suite_task" else None
        )
        self.sampler = HomogeneousSampler(
            {k: len(v) for k, v in self.datasets.items()},
            batch_size=batch_size, world_size=world_size, seed=seed,
            weights=weights, block=block, sampling=sampling, task_indices=task_indices,
            task_indices_by_burn_in=task_indices_by_burn_in,
            suite_weights=suite_weights, suite_block=suite_block,
            recurrent_prefix_choices=prefix_choices,
        )
        self.sampling = self.sampler.sampling
        self.batch_size = int(batch_size)
        self.rank = int(rank)
        self.num_workers = int(num_workers)
        self.prefetch_factor = int(prefetch_factor)
        self.pin_memory = bool(pin_memory)
        self.global_step = 0
        self.effective_workers = self.num_workers
        self.effective_prefetch = self.prefetch_factor
        self._bytes_per_batch: int | None = None
        self._iter: Iterator[TransitionWindow] | None = None
        self._next_step = 0
        self.recurrent_prefix_choices = prefix_choices

    @property
    def n_windows(self) -> int:
        return len(self.concat)

    def trajectory_manifest(self, body: str | None = None) -> dict:
        """Manifest for one body; LIBERO's single body needs no argument."""
        if body is None:
            if len(self.datasets) != 1:
                raise ValueError(
                    "body is required when requesting a multi-embodiment manifest"
                )
            body = next(iter(self.datasets))
        if body not in self.datasets:
            raise KeyError(f"loader has no dataset for embodiment {body!r}")
        return self.datasets[body].trajectory_manifest()

    # ── shared memory ────────────────────────────────────────────────────
    def bytes_per_batch(self) -> int:
        """Exact in-RAM size of one collated batch. Reads one window from disk."""
        if self._bytes_per_batch is None:
            if self.sampling == "weighted_suite_task":
                max_prefix = max(self.recurrent_prefix_choices)
                body = self.sampler.bodies[0]
                suite = self.sampler._suite_names[body][0]
                pool = self.sampler._suite_task_pools[(body, max_prefix, suite)][0]
                global_index = self.concat.offset_of(body) + int(pool[0])
                one = collate_window([
                    self.concat[(global_index, max_prefix, suite)]
                ])
            else:
                one = collate_window([self.concat[0]])
            self._bytes_per_batch = _batch_bytes(one) * self.batch_size
        return self._bytes_per_batch

    def _fit_shared_memory(self) -> None:
        """Shrink the worker queue to what /dev/shm can hold.

        Cheaper than crashing 40 minutes into a link with a bus error. Compute
        nodes normally have /dev/shm at half of RAM and keep the request as-is;
        this only bites on a small-shm node such as this login node.
        """
        if self.num_workers <= 0:
            return
        self.effective_workers, self.effective_prefetch = fit_workers(
            self.bytes_per_batch(), self.num_workers, self.prefetch_factor
        )

    # ── iteration ────────────────────────────────────────────────────────
    def batches(self, start_step: int = 0, n_steps: int = 1000) -> Iterator[TransitionWindow]:
        """Yield `n_steps` batches beginning at `start_step`. Advances `global_step`."""
        self._fit_shared_memory()
        bs = _StepBatchSampler(self.sampler, self.concat, self.rank, start_step, n_steps)
        kw: dict = {}
        if self.effective_workers > 0:
            kw.update(prefetch_factor=self.effective_prefetch, persistent_workers=False)
        dl = torch.utils.data.DataLoader(
            self.concat,
            batch_sampler=bs,
            collate_fn=collate_window,
            num_workers=self.effective_workers,
            pin_memory=self.pin_memory,
            **kw,
        )
        step = start_step
        for batch in dl:
            self.global_step = step
            yield batch
            step += 1
        self.global_step = step

    # ── step-indexed access (what loom.train.loop calls) ─────────────────
    #: how many steps one underlying DataLoader iterator covers before it is
    #: rebuilt. Large, because rebuilding respawns workers.
    ITER_CHUNK = 1 << 20

    def next(self, step: int) -> TransitionWindow:
        """The batch for `step`. Sequential calls stream; a jump re-seeks.

        ``loom.train.loop`` drives the loader as ``sampler.next(global_step)``.
        Because ``batch_at`` is a pure function of ``(seed, step, rank)``, a
        resumed link that calls ``next(4137)`` first gets exactly the batch it
        would have got had it run from step 0 — the re-seek is a fast path, not
        a correctness mechanism.
        """
        step = int(step)
        if self._iter is None or step != self._next_step:
            self._seek(step)
        try:
            batch = next(self._iter)
        except StopIteration:                    # chunk exhausted; open the next
            self._seek(step)
            batch = next(self._iter)
        self.global_step = step
        self._next_step = step + 1
        return batch

    def _seek(self, step: int) -> None:
        self._iter = iter(self.batches(step, self.ITER_CHUNK))

    def embodiment_for(self, step: int) -> str:
        """Which body this step draws. Rank-independent (see the module docstring)."""
        return self.sampler.embodiment_at(step)

    def state_dict(self) -> dict:
        return {"global_step": int(self.global_step)}

    def load_state_dict(self, state: dict) -> None:
        """Tolerant of a checkpoint written by the stub sampler.

        Sampling is a pure function of ``(seed, step, rank)`` and ``next(step)``
        takes the step explicitly, so this state is a convenience, not the
        resume mechanism. A stub checkpoint carries ``cursor``, not
        ``global_step``; refusing to load it would block the very migration this
        factory exists for.
        """
        self.global_step = int(state.get("global_step", self.global_step))
        self._iter = None                        # force a re-seek on the next call


# ═══════════════════════════════════════════════════════════════════════════
#  THROUGHPUT
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
#  FACTORY  —  the entry point loom.train.loop probes for
# ═══════════════════════════════════════════════════════════════════════════

class DataConfigError(RuntimeError):
    """The configured dataset cannot be built. Never fall back to stubs on this.

    A stub fallback is correct for ``data.source: stub``. For a real source it
    converts a crash into a wasted multi-GPU run that trains on random tensors
    and produces a meaningless score, so every path here raises.
    """


#: the placeholder in configs/base.yaml — present in every merged config, so it
#: is a default rather than an explicit choice. See ``resolve_cache_root``.
BASE_CACHE_DIR_PLACEHOLDER = "cache/"


def resolve_cache_root(cfg: Mapping, override: str | os.PathLike | None = None) -> Path:
    """Where the feature cache lives. Precedence, highest first:

    1. an explicit `override` argument
    2. ``data.cache_dir`` **when a run config actually set it** — i.e. when it
       differs from the ``cache/`` placeholder every config inherits from
       ``configs/base.yaml``
    3. ``$LOOM_CACHE_DIR`` (what the sbatch exports)
    4. ``data.cache_dir`` as written, resolved against the cwd

    Rule 2 before rule 3 keeps "explicit config beats env"; rule 3 before rule 4
    is what makes the inherited placeholder lose to a real deployment path
    instead of silently pointing the run at a nonexistent ``./cache``.
    """
    if override is not None:
        return Path(override).expanduser()
    cfg_dir = str(cfg.get("data", {}).get("cache_dir", "") or "")
    if cfg_dir and cfg_dir.rstrip("/") != BASE_CACHE_DIR_PLACEHOLDER.rstrip("/"):
        return Path(cfg_dir).expanduser()
    env_dir = os.environ.get("LOOM_CACHE_DIR")
    if env_dir:
        return Path(env_dir).expanduser()
    return Path(cfg_dir or BASE_CACHE_DIR_PLACEHOLDER).expanduser()


def _open_cache(root: Path) -> FeatureCache:
    """Open the cache or die with the resolved path in the message."""
    if not root.is_dir():
        raise DataConfigError(
            f"feature cache {root} does not exist. Resolution order is "
            f"data.cache_dir (when not the base placeholder) > $LOOM_CACHE_DIR > "
            f"data.cache_dir as written; $LOOM_CACHE_DIR is "
            f"{os.environ.get('LOOM_CACHE_DIR')!r}. Build it with the frozen "
            f"tower before launching — training on stub windows is not a fallback."
        )
    try:
        return FeatureCache(root)
    except CacheFormatError as e:
        raise DataConfigError(f"feature cache {root} is unusable: {e}") from e


def build_loader(
    cfg: Mapping,
    *,
    rank: int = 0,
    world: int = 1,
    seed: int | None = None,
    device: str = "cpu",
    cache_root: str | os.PathLike | None = None,
) -> LoomLoader:
    """Config -> a ready ``LoomLoader``. The signature ``loom.train.loop`` calls.

    A thin adapter over ``LoomLoader.__init__``, which still takes already-built
    datasets. Raises ``DataConfigError`` on anything it cannot satisfy; it never
    degrades to synthetic data, because a silent degradation here costs a full
    16-GPU run and yields a score that means nothing.

    Geometry (``n_patches``, ``feat_dim``, ``lang_len``) is read from the cache
    manifest, never assumed — the frozen tower decides it, not this module.
    """
    dcfg = dict(cfg.get("data", {}))
    source = str(dcfg.get("source", "stub"))
    if source == "stub":
        raise DataConfigError(
            "data.source is 'stub'; build_loader serves real datasets only. "
            "The training loop's own stub sampler covers this case."
        )

    seed = int(cfg.get("run", {}).get("seed", 0)) if seed is None else int(seed)
    batch = int(dcfg.get("batch_per_gpu", 8))
    bodies = list(dcfg.get("embodiments", []) or [])
    if not bodies:
        raise DataConfigError("data.embodiments is empty; nothing to load")
    trajectory_split, holdout_keys, _ = _trajectory_split_config(dcfg)
    if source != "libero" and (trajectory_split != "all" or holdout_keys):
        raise DataConfigError(
            "whole-trajectory holdout_demo_keys are currently implemented only "
            f"for data.source='libero', not {source!r}"
        )

    root = resolve_cache_root(cfg, cache_root)
    cache = _open_cache(root)

    if (
        str(dcfg.get("sampling", "uniform_window")) == "weighted_suite_task"
        and int(dcfg.get("recurrent_burn_in", 0)) != 0
    ):
        raise DataConfigError(
            "weighted_suite_task uses recurrent_prefix_choices and requires "
            "data.recurrent_burn_in=0 so early windows remain eligible"
        )

    datasets: dict[str, CachedWindowDataset] = {}
    for body in bodies:
        datasets[body] = _dataset_for(body, source, cache, dcfg)

    loader = LoomLoader(
        datasets,
        batch_size=batch,
        world_size=int(world),
        rank=int(rank),
        seed=seed,
        sampling=str(dcfg.get("sampling", "uniform_window")),
        num_workers=int(dcfg.get("num_workers", 0)),
        prefetch_factor=int(dcfg.get("prefetch_factor", 2)),
        # the loop hands us device="cpu" and moves tensors itself, so pinning is
        # keyed on whether a GPU exists at all, not on that argument
        pin_memory=bool(dcfg.get("pin_memory", False)) and torch.cuda.is_available(),
        suite_weights=dcfg.get("suite_weights"),
        suite_block=int(dcfg.get("suite_block", 20)),
        recurrent_prefix_choices=dcfg.get("recurrent_prefix_choices", (0,)),
    )
    # eager, so the numbers below are the ones training will actually use and a
    # broken cache fails here rather than 40 minutes in
    loader._fit_shared_memory()

    spec = cache.spec
    print(
        f"[data] real loader: source={source} cache={root} "
        f"trajectories={sum(len(ds.trajectory_ids) for ds in datasets.values())}/"
        f"{len(cache)} windows={loader.n_windows} "
        f"bodies={sorted(datasets)} batch_per_gpu={batch} world={world} rank={rank} "
        f"sampling={loader.sampling} recurrent_burn_in="
        f"{int(dcfg.get('recurrent_burn_in', 0))} "
        f"recurrent_prefix_choices={list(loader.recurrent_prefix_choices)} "
        f"trajectory_split={trajectory_split} "
        f"codec={spec.codec} V={spec.n_views} P={spec.n_patches} F={spec.feat_dim} "
        f"L={spec.lang_len} "
        f"workers={loader.effective_workers}/{loader.num_workers} "
        f"prefetch={loader.effective_prefetch} "
        f"batch={loader.bytes_per_batch() / 2 ** 20:.1f} MiB "
        f"shm_free={shm_free_bytes() / 2 ** 20:.0f} MiB",
        flush=True,
    )
    return loader


def build_gate_loader(
    cfg: Mapping,
    *,
    rank: int = 0,
    world: int = 1,
    seed: int | None = None,
    device: str = "cpu",
    cache_root: str | os.PathLike | None = None,
) -> LoomLoader:
    """Build only the configured heldout whole trajectories.

    The standalone offline gate can consume the training config directly; this
    wrapper changes only ``data.trajectory_split`` to ``gate`` and leaves its
    exact ``holdout_demo_keys`` and all loader/sampler settings untouched.
    """
    gate_cfg = dict(cfg)
    gate_cfg["data"] = {**dict(cfg.get("data", {})), "trajectory_split": "gate"}
    return build_loader(
        gate_cfg, rank=rank, world=world, seed=seed, device=device,
        cache_root=cache_root,
    )


def _dataset_for(
    body: str, source: str, cache: FeatureCache, dcfg: Mapping
) -> CachedWindowDataset:
    """One embodiment's windows, from the cache plus that adapter's metadata."""
    trajectory_split, holdout_keys, split_configured = _trajectory_split_config(dcfg)
    if source == "robotwin" and body == "robotwin_aloha":
        if trajectory_split != "all" or holdout_keys:
            raise DataConfigError(
                "whole-trajectory holdout_demo_keys are currently implemented "
                "only for LIBERO"
            )
        from dataclasses import replace  # noqa: PLC0415

        from .adapters import robotwin as RT  # noqa: PLC0415

        trajs = RT.robotwin_trajectories(
            root=dcfg.get("data_root"),
            tasks_=tuple(dcfg["tasks"]) if dcfg.get("tasks") else None,
            max_episodes=dcfg.get("max_episodes"),
        )
        trajs = [t for t in trajs if t.traj_id in cache]
        if not trajs:
            raise DataConfigError(
                f"no cached RoboTwin trajectories in {cache.root}. The cache holds "
                f"{len(cache)} entries; check data.tasks and $LOOM_DATA_ROOT "
                f"(currently {os.environ.get('LOOM_DATA_ROOT')!r})."
            )
        if dcfg.get("action_free", False):
            trajs = [replace(t, actions=None) for t in trajs]
        return CachedWindowDataset(
            trajs, cache, stride=RT.WINDOW_STRIDE,
            recurrent_burn_in=dcfg.get("recurrent_burn_in", 0),
        )

    if source != "libero" or body != "libero_franka":
        raise DataConfigError(
            f"no adapter wired for source={source!r} body={body!r}. LIBERO is the "
            f"only Phase 1A dataset (PLAN §4.A); the rest are Phase 1B."
        )

    from .adapters import libero as LB

    try:
        trajs = LB.libero_trajectories(
            root=dcfg.get("data_root"),
            suites=tuple(dcfg.get("suites", LB.SUITES)),
            max_demos=dcfg.get("max_demos"),
        )
    except FileNotFoundError as e:
        raise DataConfigError(
            f"cannot read the LIBERO demo files: {e}. The cache holds features but "
            f"not actions, so the raw HDF5s are still needed at load time; set "
            f"$LOOM_DATA_ROOT (currently {os.environ.get('LOOM_DATA_ROOT')!r})."
        ) from e

    produced = [t.traj_id for t in trajs]
    expected_task_ids = (
        sorted({_libero_trajectory_identity(t.traj_id)[0] for t in trajs})
        if split_configured else None
    )
    trajs = [t for t in trajs if t.traj_id in cache]
    if not trajs:
        raise DataConfigError(
            f"none of the {len(produced)} discovered LIBERO trajectories are in the "
            f"cache. The cache holds {len(cache)} entries keyed like "
            f"{next(iter(cache.keys()), '<empty>')!r}; the adapter produced ids like "
            f"{(produced[0] if produced else '<none>')!r}. The two must agree — "
            f"re-encode, or fix data.suites / $LOOM_DATA_ROOT."
        )
    data_manifest = None
    if split_configured:
        trajs, data_manifest = _select_libero_trajectories(
            trajs,
            split=trajectory_split,
            holdout_demo_keys=holdout_keys,
            expected_task_ids=expected_task_ids,
        )
    if dcfg.get("action_free", False):
        trajs = [replace(t, actions=None) for t in trajs]
    return CachedWindowDataset(
        trajs, cache, stride=LB.WINDOW_STRIDE,
        recurrent_burn_in=dcfg.get("recurrent_burn_in", 0),
        data_manifest=data_manifest,
    )


@dataclass
class Throughput:
    n_batches: int
    seconds: float
    batch_size: int
    bytes_moved: int

    @property
    def batches_per_s(self) -> float:
        return self.n_batches / max(self.seconds, 1e-9)

    @property
    def samples_per_s(self) -> float:
        return self.batches_per_s * self.batch_size

    @property
    def mib_per_s(self) -> float:
        return (self.bytes_moved / (1024.0 * 1024.0)) / max(self.seconds, 1e-9)

    @property
    def mib_per_batch(self) -> float:
        return self.bytes_moved / (1024.0 * 1024.0) / max(self.n_batches, 1)

    def ratio(self, consumption_hz: float = DEFAULT_CONSUMPTION_HZ) -> float:
        """Produced batches/s over consumed batches/s. Must clear STARVATION_MARGIN."""
        return self.batches_per_s / max(consumption_hz, 1e-9)

    def sustains(
        self,
        consumption_hz: float = DEFAULT_CONSUMPTION_HZ,
        margin: float = STARVATION_MARGIN,
    ) -> bool:
        return self.ratio(consumption_hz) >= margin

    def __str__(self) -> str:
        return (f"{self.batches_per_s:.2f} batch/s  {self.samples_per_s:.1f} window/s  "
                f"{self.mib_per_s:.0f} MiB/s  ({self.mib_per_batch:.2f} MiB/batch)")


def _batch_bytes(w: TransitionWindow) -> int:
    n = sum(f[k].nbytes for f in w["feats"] for k in ("views", "proprio", "lang"))
    n += sum(
        f[k].nbytes
        for f in w.get("burn_in_feats", ())
        for k in ("views", "proprio", "lang")
    )
    if w["actions"] is not None:
        n += w["actions"].nbytes
    return int(n)


def measure_throughput(
    loader: LoomLoader,
    n_batches: int = 32,
    warmup: int = 4,
    start_step: int = 0,
) -> Throughput:
    """Wall-clock production rate of the pipeline, warmup excluded.

    `warmup` covers DataLoader worker spawn and the first page faults; including
    it would understate steady state, which is what training actually sees.
    """
    total = warmup + n_batches
    t0 = 0.0
    moved = 0
    seen = 0
    for i, batch in enumerate(loader.batches(start_step, total)):
        if i == warmup:
            t0 = time.perf_counter()
        if i >= warmup:
            moved += _batch_bytes(batch)
            seen += 1
    elapsed = time.perf_counter() - t0
    return Throughput(
        n_batches=seen, seconds=elapsed, batch_size=loader.batch_size, bytes_moved=moved
    )
