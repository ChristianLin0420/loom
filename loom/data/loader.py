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
Within one embodiment the schedule is epoch-permutation based: for a given
epoch every rank takes a disjoint slice, so an epoch is covered with no
duplicates within a rank and none across ranks either.

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
    "HomogeneousSampler", "CachedWindowDataset", "collate_window",
    "LoomLoader", "Throughput", "measure_throughput",
    "shm_free_bytes", "shm_headroom", "fit_workers",
    "build_loader", "resolve_cache_root", "DataConfigError",
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

    # ── schedule ─────────────────────────────────────────────────────────
    def steps_per_epoch(self, body: str) -> int:
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

    def batch_at(self, step: int, rank: int = 0) -> tuple[str, np.ndarray]:
        if not (0 <= rank < self.world_size):
            raise ValueError(f"rank {rank} outside world_size {self.world_size}")
        body = self.embodiment_at(step)
        ls = self.local_step(step)
        spe = self.steps_per_epoch(body)
        perm = self._permutation(body, ls // spe)
        k = ls % spe
        lo = (k * self.world_size + rank) * self.batch_size
        return body, perm[lo:lo + self.batch_size]


# ═══════════════════════════════════════════════════════════════════════════
#  DATASET
# ═══════════════════════════════════════════════════════════════════════════

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
        self.actions: dict[str, np.ndarray | None] = {t.traj_id: t.actions for t in trajectories}
        self.windows: list[WindowIndex] = []
        for t in trajectories:
            self.windows.extend(segment(t, stride=stride))
        self._traj = {t.traj_id: t for t in trajectories}

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int) -> dict:
        w = self.windows[i]
        blob = self.cache.read(w.traj_id, w.obs_src_index)
        views = torch.from_numpy(np.ascontiguousarray(blob["views"]))       # (5,V,P,F)
        proprio = torch.from_numpy(np.ascontiguousarray(blob["proprio"]))   # (5,dof)
        lang = torch.from_numpy(np.ascontiguousarray(blob["lang"]))         # (L,F)
        feats = [
            ObsFeats(views=views[s], proprio=proprio[s], lang=lang)
            for s in range(N_STATES)
        ]
        a = window_actions(self._traj[w.traj_id], w)
        return {
            "feats": feats,
            "actions": None if a is None else torch.from_numpy(np.ascontiguousarray(a)),
            "lang": lang,
            "embodiment": w.embodiment,
            "src_fps": w.src_fps,
        }


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

    feats: list[ObsFeats] = []
    for s in range(N_STATES):
        feats.append(ObsFeats(
            views=torch.stack([x["feats"][s]["views"] for x in samples]),
            proprio=torch.stack([x["feats"][s]["proprio"] for x in samples]),
            lang=torch.stack([x["feats"][s]["lang"] for x in samples]),
        ))
    action_free = [x["actions"] is None for x in samples]
    if any(action_free) and not all(action_free):
        raise ValueError("batch mixes action-labelled and action-free windows")
    actions = None if action_free[0] else torch.stack([x["actions"] for x in samples])
    return TransitionWindow(
        feats=feats,
        actions=actions,
        lang=torch.stack([x["lang"] for x in samples]),
        embodiment=body,
        src_fps=float(samples[0]["src_fps"]),
    )


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

    def __getitem__(self, i: int) -> dict:
        j = int(np.searchsorted(self.offsets, i, side="right")) - 1
        return self.datasets[j][i - int(self.offsets[j])]


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

    def __iter__(self) -> Iterator[list[int]]:
        for t in range(self.start_step, self.start_step + self.n_steps):
            body, idx = self.sampler.batch_at(t, self.rank)
            off = self.concat.offset_of(body)
            yield [off + int(i) for i in idx]


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
    ) -> None:
        for name, ds in datasets.items():
            if ds.embodiment != name:
                raise ValueError(f"dataset keyed {name!r} holds body {ds.embodiment!r}")
        self.datasets = dict(datasets)
        self.concat = _ConcatBodies(self.datasets)
        self.sampler = HomogeneousSampler(
            {k: len(v) for k, v in self.datasets.items()},
            batch_size=batch_size, world_size=world_size, seed=seed,
            weights=weights, block=block,
        )
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

    @property
    def n_windows(self) -> int:
        return len(self.concat)

    # ── shared memory ────────────────────────────────────────────────────
    def bytes_per_batch(self) -> int:
        """Exact in-RAM size of one collated batch. Reads one window from disk."""
        if self._bytes_per_batch is None:
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

    root = resolve_cache_root(cfg, cache_root)
    cache = _open_cache(root)

    datasets: dict[str, CachedWindowDataset] = {}
    for body in bodies:
        datasets[body] = _dataset_for(body, source, cache, dcfg)

    loader = LoomLoader(
        datasets,
        batch_size=batch,
        world_size=int(world),
        rank=int(rank),
        seed=seed,
        num_workers=int(dcfg.get("num_workers", 0)),
        prefetch_factor=int(dcfg.get("prefetch_factor", 2)),
        # the loop hands us device="cpu" and moves tensors itself, so pinning is
        # keyed on whether a GPU exists at all, not on that argument
        pin_memory=bool(dcfg.get("pin_memory", False)) and torch.cuda.is_available(),
    )
    # eager, so the numbers below are the ones training will actually use and a
    # broken cache fails here rather than 40 minutes in
    loader._fit_shared_memory()

    spec = cache.spec
    print(
        f"[data] real loader: source={source} cache={root} "
        f"trajectories={len(cache)} windows={loader.n_windows} "
        f"bodies={sorted(datasets)} batch_per_gpu={batch} world={world} rank={rank} "
        f"codec={spec.codec} V={spec.n_views} P={spec.n_patches} F={spec.feat_dim} "
        f"L={spec.lang_len} "
        f"workers={loader.effective_workers}/{loader.num_workers} "
        f"prefetch={loader.effective_prefetch} "
        f"batch={loader.bytes_per_batch() / 2 ** 20:.1f} MiB "
        f"shm_free={shm_free_bytes() / 2 ** 20:.0f} MiB",
        flush=True,
    )
    return loader


def _dataset_for(
    body: str, source: str, cache: FeatureCache, dcfg: Mapping
) -> CachedWindowDataset:
    """One embodiment's windows, from the cache plus that adapter's metadata."""
    if source == "robotwin" and body == "robotwin_aloha":
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
        return CachedWindowDataset(trajs, cache, stride=RT.WINDOW_STRIDE)

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
    trajs = [t for t in trajs if t.traj_id in cache]
    if not trajs:
        raise DataConfigError(
            f"none of the {len(produced)} discovered LIBERO trajectories are in the "
            f"cache. The cache holds {len(cache)} entries keyed like "
            f"{next(iter(cache.keys()), '<empty>')!r}; the adapter produced ids like "
            f"{(produced[0] if produced else '<none>')!r}. The two must agree — "
            f"re-encode, or fix data.suites / $LOOM_DATA_ROOT."
        )
    if dcfg.get("action_free", False):
        trajs = [replace(t, actions=None) for t in trajs]
    return CachedWindowDataset(trajs, cache, stride=LB.WINDOW_STRIDE)


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
