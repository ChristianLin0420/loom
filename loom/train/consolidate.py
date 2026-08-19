"""Offline consolidation of FSDP per-rank shards into ONE eval-ready checkpoint.

    .venv/bin/python -m loom.train.consolidate --run_dir runs/r0a --pin

which picks the newest complete step, writes ``runs/r0a_eval/ckpt_<step>.pt``,
and prints the three-part verification. **No GPU, no srun, no process group,
one process** -- it runs on a login node while the chain keeps training. Name a
step with ``--step``; skip the 878 M frozen tower with ``--no_policy_check``.

``--pin`` is not optional on a live run. ``ckpt._prune`` keeps three steps, and
R0-A checkpoints every 500 steps at ~6 min a step, so a named step is deleted
about eighteen minutes after it appears -- less than one consolidate + verify
cycle. ``--pin`` hardlinks the 16 shards out of the run dir first (instant, no
extra bytes, deletes nothing) and reads those. Step 500 was already gone by the
time this module was first pointed at it.

**Read-side only.** Nothing in the training path imports this module
(``loom/train/__init__.py`` is empty), and it does not change how ``ckpt.save``
or ``fsdp.sharded_state_dict`` *write*. A live chain can be consolidated from a
login node while it keeps running.

Why this exists
---------------
``fsdp.sharded_state_dict`` saves with ``StateDictType.SHARDED_STATE_DICT``, one
file per rank, which is the right call: ``FULL_STATE_DICT`` all-gathers 150 M
parameters onto rank 0 and returns *empty dicts* everywhere else -- a save that
looks successful and restores a random estimator. But nothing consolidated the
shards afterwards, and ``loom.eval.policy`` loads a **single** file. Loading one
of those shards outside a process group dies with::

    RuntimeError: Need to initialize default process group using
    "init_process_group" before loading ShardedTensor

and loading it *inside* a world=1 group dies with "Local rank at save time was
5, but at load time was 0". So neither 1 rank nor 16 ranks can read shard *r*
without also being rank *r*.

How it works, and why no process group is needed
------------------------------------------------
``ShardedTensor.__setstate__`` refuses to rebuild without a default process
group, and then validates the saved rank against it -- but the *data* it is
refusing to rebuild is already fully self-describing:

  * ``state[0]`` is this rank's list of ``Shard`` (tensor + offsets),
  * ``state[1]`` is the **global** ``ShardedTensorMetadata``: the full tensor
    size and a ``ShardMetadata`` for *every* rank's piece.

The process group is only needed to *communicate*. Offline we do not
communicate; we read all 16 files. :func:`sharded_tensor_without_pg` therefore
swaps in a ``__setstate__`` that keeps the shards and the metadata and skips the
rank validation, and the reassembly is a pure memcpy driven by the metadata.
The patch is a context manager scoped to the load and is never installed in any
process that trains.

``fsdp._shard_utils._create_chunk_sharded_tensor`` chunks the **unflattened**
parameter along dim 0 (``tensor.chunk(world_size, dim=0)``), so a shard is a
contiguous dim-0 slice of the real parameter, not a slice of a flat buffer.
Note ``chunk`` returns *fewer* than ``world_size`` pieces when
``size(0) < world_size`` -- ``estimator.type_embed`` is ``(4, 768)`` and lives on
ranks 0-3 only, ranks 4-15 hold nothing for it. Anything that assumes "16 files,
16 pieces, concatenate" silently corrupts those four tensors, which is why
placement is driven by ``shards_metadata`` and coverage is asserted rather than
assumed.

What the output is, and is not
------------------------------
The output carries ``payload["model"]`` -- every parameter of the unwrapped
``LoomModel``, real tensors, no ShardedTensor -- plus the scalars eval reports
(``global_step``, ``config_hash``, ``git_sha``, ``wandb_run_id``, ...).

It deliberately does **not** carry ``optimizer`` / ``scheduler`` / ``sampler`` /
``rng``. Under ``use_orig_params=True`` the optimizer state dict holds each
rank's *local* slice keyed by parameter index with no metadata to place it, so
it is not consolidatable by this route, and eval has no use for it. **Resume
still reads the per-rank shards and is untouched by this file.** The output
records that in ``payload["consolidated"]["not_resumable"]``.

Verification is part of the tool, not an afterthought, because the dangerous
failure here is a consolidator that emits a shape-correct partly-random model:
``loom.eval.policy`` calls ``load_state_dict(strict=False)`` and will not
complain. See :func:`verify`.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch

__all__ = [
    "sharded_tensor_without_pg", "load_shard", "shard_paths", "complete_steps",
    "is_sharded", "consolidate_state_dict", "consolidate", "verify", "pin_shards",
    "EVAL_SECTIONS", "DROPPED_SECTIONS", "main",
]

#: Sections of a training payload that the consolidated file carries verbatim.
EVAL_SECTIONS = ("format", "global_step", "samples_seen", "config_hash",
                 "git_sha", "world_size", "wandb_run_id", "stop_reason")

#: Sections deliberately dropped, and why. Recorded in the output payload so a
#: later reader does not have to guess whether they were forgotten.
DROPPED_SECTIONS = {
    "optimizer": "per-rank local slices keyed by param index; not placeable "
                 "offline, and eval does not need it. Resume reads the shards.",
    "scheduler": "trivially rebuilt from the config; link-local.",
    "sampler":   "per-rank cursor; meaningless outside its own rank.",
    "rng":       "per-rank ByteTensors; torch.load onto cuda then breaks "
                 "set_rng_state, and eval does not restore RNG.",
    "ema":       "top-level duplicate of the model's ema.* keys, which ARE "
                 "consolidated into payload['model'].",
}


# ═══════════════════════════════════════════════════════════════════════════
#  READING A SHARD WITHOUT A PROCESS GROUP
# ═══════════════════════════════════════════════════════════════════════════

@contextlib.contextmanager
def sharded_tensor_without_pg() -> Iterator[None]:
    """Let ``torch.load`` rebuild a ``ShardedTensor`` outside a process group.

    Scoped, and restored on the way out, so nothing else in the interpreter --
    least of all a training process that imported torch for real reasons -- ever
    sees the patched class. The replacement keeps exactly the two fields
    reassembly needs (``_local_shards``, ``_metadata``) and leaves
    ``_process_group`` None, so any attempt to *use* the object collectively
    fails loudly instead of silently communicating on the wrong group.
    """
    from torch.distributed._shard.sharded_tensor.api import ShardedTensor

    original = ShardedTensor.__setstate__

    def _offline_setstate(self, state):                # noqa: ANN001
        self._sharded_tensor_id = None
        (self._local_shards, self._metadata, _pg_state,
         self._sharding_spec, self._init_rrefs) = state
        self._process_group = None

    ShardedTensor.__setstate__ = _offline_setstate     # type: ignore[method-assign]
    try:
        yield
    finally:
        ShardedTensor.__setstate__ = original          # type: ignore[method-assign]


def is_sharded(v: Any) -> bool:
    """True for a ShardedTensor (or a DTensor, which we refuse rather than guess)."""
    return type(v).__name__ in ("ShardedTensor", "DTensor")


def load_shard(path: str | Path, *, mmap: bool = True) -> dict[str, Any]:
    """One rank's checkpoint file, CPU, ShardedTensors intact but inert.

    ``mmap=True`` keeps the 2 GB of optimizer state we are about to ignore out of
    RAM: the pages we never touch are never read off Lustre. ``map_location``
    stays "cpu" -- CLAUDE.md's gotcha about RNG ByteTensors landing on a GPU.
    """
    with sharded_tensor_without_pg():
        return torch.load(str(path), map_location="cpu", weights_only=False, mmap=mmap)


# ═══════════════════════════════════════════════════════════════════════════
#  FINDING THE SHARDS
# ═══════════════════════════════════════════════════════════════════════════

def _rank_of(p: Path) -> int:
    return int(p.stem.split("rank")[-1])


def complete_steps(run_dir: str | Path) -> list[tuple[int, int]]:
    """``[(step, n_shards), ...]`` ascending, for every step with shards on disk."""
    run_dir = Path(run_dir)
    seen: dict[int, int] = {}
    for p in run_dir.glob("ckpt_*_rank*.pt"):
        try:
            step = int(p.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        seen[step] = seen.get(step, 0) + 1
    return sorted(seen.items())


def shard_paths(run_dir: str | Path, step: int | None = None
                ) -> tuple[int, list[Path]]:
    """``(step, [path per rank, rank-ordered])``.

    ``step=None`` picks the newest step whose shard count matches the ``world_size``
    recorded *inside* rank 0's file. A run that is mid-checkpoint has a partial
    step on disk; consolidating it would produce a model that is a few ranks
    stale in the estimator and current everywhere else, with nothing to see.
    """
    run_dir = Path(run_dir)
    steps = complete_steps(run_dir)
    if not steps:
        raise FileNotFoundError(f"no ckpt_*_rank*.pt under {run_dir}")

    if step is None:
        for cand, n in reversed(steps):
            paths = sorted(run_dir.glob(f"ckpt_{cand:09d}_rank*.pt"), key=_rank_of)
            head = load_shard(paths[0])
            world = int(head.get("world_size", n))
            del head
            if n == world:
                step = cand
                break
        else:
            raise RuntimeError(
                f"no step under {run_dir} has a complete shard set: {steps}. The "
                f"run is mid-checkpoint, or shards were pruned unevenly."
            )

    paths = sorted(run_dir.glob(f"ckpt_{step:09d}_rank*.pt"), key=_rank_of)
    if not paths:
        raise FileNotFoundError(f"no shards for step {step} under {run_dir}")
    ranks = [_rank_of(p) for p in paths]
    if ranks != list(range(len(ranks))):
        raise RuntimeError(
            f"step {step} shard ranks are {ranks}, not 0..{len(ranks) - 1}. A "
            f"missing rank is a missing slice of the estimator, not a smaller model."
        )
    return int(step), paths


def pin_shards(run_dir: str | Path, step: int, dest: str | Path) -> list[Path]:
    """Hardlink a step's 16 shards out of the live run dir, then read those.

    ``ckpt._prune`` keeps three steps. R0-A checkpoints every 500 steps at
    roughly six minutes a step, so a named step is deleted about eighteen
    minutes after it appears -- comfortably inside the time it takes to
    consolidate *and* verify 32 GB. Reading a file that gets unlinked mid-run is
    fine for an open mmap and fatal for the next ``open()``, which is exactly the
    shape of bug that shows up once and is then unreproducible.

    A hardlink is instantaneous, costs no extra bytes, writes nothing into
    ``run_dir`` and deletes nothing from it: ``_prune``'s ``unlink`` drops the
    run's directory entry while this one keeps the inode alive. Falls back to a
    copy if the destination is on another filesystem.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    _, paths = shard_paths(run_dir, step)
    out = []
    for p in paths:
        q = dest / p.name
        if not q.exists():
            try:
                os.link(p, q)
            except OSError:                              # cross-device
                import shutil

                shutil.copy2(p, q)
        out.append(q)
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  CONSOLIDATION
# ═══════════════════════════════════════════════════════════════════════════

def _placement_rank(placement: Any) -> int:
    fn = getattr(placement, "rank", None)
    if callable(fn):
        return int(fn())
    return int(str(placement).split("rank:")[1].split("/")[0])


def _tile_check(key: str, md: Any) -> None:
    """The global shard metadata must tile the global tensor exactly, along dim 0.

    ``_create_chunk_sharded_tensor`` only ever chunks dim 0, so a metadata set
    that disagrees means a different sharding scheme than this reassembler
    implements. Refuse rather than fill in the gaps with zeros.
    """
    size = list(md.size)
    cursor = 0
    for sm in md.shards_metadata:
        off, sz = list(sm.shard_offsets), list(sm.shard_sizes)
        if off[1:] != [0] * (len(size) - 1) or sz[1:] != size[1:]:
            raise RuntimeError(
                f"{key}: shard {off}/{sz} is not a whole dim-0 slice of {size}. "
                f"This consolidator only understands dim-0 chunking."
            )
        if off[0] != cursor:
            raise RuntimeError(
                f"{key}: shard offsets are not contiguous -- expected dim0 offset "
                f"{cursor}, got {off[0]}."
            )
        cursor += sz[0]
    if cursor != size[0]:
        raise RuntimeError(
            f"{key}: shards cover {cursor} of {size[0]} rows. A partly covered "
            f"parameter is a partly random parameter."
        )


def consolidate_state_dict(paths: Sequence[Path], section: str = "model",
                           *, verbose: bool = True) -> tuple[dict[str, torch.Tensor],
                                                             dict[str, Any]]:
    """Reassemble one section of the payload from every rank's shard.

    Replicated entries (plain tensors) are taken from rank 0. Sharded entries are
    allocated **full of NaN** and then filled from the per-rank pieces, so an
    unwritten byte is a NaN and not a plausible-looking zero; the NaN sweep at
    the end is a second, independent witness to the offset-coverage check.
    """
    full: dict[str, torch.Tensor] = {}
    meta: dict[str, Any] = {}                       # key -> ShardedTensorMetadata
    covered: dict[str, set[tuple[int, ...]]] = {}
    replicated: list[str] = []
    sharded: list[str] = []

    for r, path in enumerate(paths):
        t0 = time.time()
        payload = load_shard(path)
        sd = payload.get(section)
        if not isinstance(sd, dict):
            raise KeyError(f"{path} has no dict section {section!r}")

        if r == 0:
            for k, v in sd.items():
                if is_sharded(v):
                    if type(v).__name__ == "DTensor":
                        raise RuntimeError(
                            f"{k} is a DTensor. FSDP was configured with a device "
                            f"mesh; this reassembler reads ShardedTensor only."
                        )
                    md = v._metadata
                    _tile_check(k, md)
                    dtype = md.tensor_properties.dtype
                    if dtype.is_floating_point:
                        full[k] = torch.full(tuple(md.size), float("nan"), dtype=dtype)
                    else:                            # ints: coverage set is the check
                        full[k] = torch.zeros(tuple(md.size), dtype=dtype)
                    meta[k] = md
                    covered[k] = set()
                    sharded.append(k)
                else:
                    full[k] = v.detach().clone()     # mmap-backed: clone off the file
                    replicated.append(k)
        elif set(sd) != set(full):
            missing = sorted(set(full) - set(sd))[:4]
            extra = sorted(set(sd) - set(full))[:4]
            raise RuntimeError(
                f"rank {r} disagrees with rank 0 on the {section!r} key set "
                f"(missing e.g. {missing}, extra e.g. {extra})"
            )

        for k in sharded:
            v = sd[k]
            if not is_sharded(v):
                raise RuntimeError(f"rank {r}: {k} is {type(v).__name__}, "
                                   f"ShardedTensor on rank 0")
            if list(v._metadata.size) != list(meta[k].size):
                raise RuntimeError(
                    f"rank {r}: {k} global size {list(v._metadata.size)} != rank 0's "
                    f"{list(meta[k].size)}")
            for shard in v._local_shards:
                off = tuple(shard.metadata.shard_offsets)
                sz = list(shard.metadata.shard_sizes)
                if _placement_rank(shard.metadata.placement) != r:
                    raise RuntimeError(
                        f"{path} holds a shard placed on rank "
                        f"{_placement_rank(shard.metadata.placement)}; the shard "
                        f"files are misnamed or were written by colliding ranks.")
                if off in covered[k]:
                    raise RuntimeError(f"{k}: offset {off} written twice")
                if list(shard.tensor.shape) != sz:
                    raise RuntimeError(f"{k}: shard tensor {list(shard.tensor.shape)} "
                                       f"!= metadata {sz}")
                full[k].narrow(0, off[0], sz[0]).copy_(shard.tensor)
                covered[k].add(off)

        del payload, sd
        if verbose:
            print(f"[consolidate] rank {r:>2} {path.name} "
                  f"({time.time() - t0:.1f}s)", flush=True)

    # ── coverage, two independent witnesses ────────────────────────────────
    for k in sharded:
        want = {tuple(sm.shard_offsets) for sm in meta[k].shards_metadata}
        if covered[k] != want:
            raise RuntimeError(
                f"{k}: filled offsets {sorted(covered[k])} != expected "
                f"{sorted(want)}. Some rows of this parameter were never written.")
        if full[k].dtype.is_floating_point and not torch.isfinite(full[k]).all():
            n = int((~torch.isfinite(full[k])).sum())
            raise RuntimeError(
                f"{k}: {n} non-finite entries after reassembly. Either a shard was "
                f"missed (NaN sentinel survived) or training itself diverged.")

    report = {
        "section": section,
        "n_keys": len(full),
        "n_replicated_keys": len(replicated),
        "n_sharded_keys": len(sharded),
        "n_params": int(sum(v.numel() for v in full.values())),
        "n_sharded_params": int(sum(full[k].numel() for k in sharded)),
        "n_shard_pieces": int(sum(len(covered[k]) for k in sharded)),
        "sharded_keys": sharded,
        "replicated_keys": replicated,
    }
    return full, report


def consolidate(run_dir: str | Path, out: str | Path, *, step: int | None = None,
                verbose: bool = True) -> dict[str, Any]:
    """Write one eval-ready checkpoint. Returns a report dict.

    ``out`` must not be inside ``run_dir``: the run is live and prunes anything
    matching ``ckpt_*_rank*.pt``.
    """
    run_dir, out = Path(run_dir), Path(out)
    if out.resolve().parent == run_dir.resolve():
        raise ValueError(
            f"refusing to write into the live run dir {run_dir}. ckpt._prune "
            f"deletes by glob there. Use e.g. {run_dir}_eval/.")

    step, paths = shard_paths(run_dir, step)
    if verbose:
        print(f"[consolidate] step {step}, {len(paths)} shards from {run_dir}",
              flush=True)

    head = load_shard(paths[0])
    world = int(head.get("world_size", len(paths)))
    if world != len(paths):
        raise RuntimeError(
            f"checkpoint says world_size={world} but {len(paths)} shards are on "
            f"disk. Consolidating a partial world silently drops estimator rows.")
    scalars = {k: head[k] for k in EVAL_SECTIONS if k in head}
    del head

    model, report = consolidate_state_dict(paths, "model", verbose=verbose)

    payload: dict[str, Any] = dict(scalars)
    payload["model"] = model
    payload["consolidated"] = {
        "tool": "loom.train.consolidate",
        "run_dir": str(run_dir.resolve()),
        "step": step,
        "n_shards": len(paths),
        "shard_files": [p.name for p in paths],
        "created_unix": time.time(),
        # str(), not the TorchVersion object: anything but a plain builtin here
        # makes the whole file un-loadable with weights_only=True, and eval runs
        # under a different interpreter (py3.10 + LIBERO) than training.
        "torch": str(torch.__version__),
        "not_resumable": (
            "eval only: optimizer/scheduler/sampler/rng are NOT here. Resume "
            "still loads the per-rank shards through fsdp.sharded_state_dict."),
        "dropped": DROPPED_SECTIONS,
        **{k: v for k, v in report.items() if not k.endswith("_keys") or
           k.startswith("n_")},
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, out)
    report["out"] = str(out)
    report["step"] = step
    report["bytes"] = out.stat().st_size
    if verbose:
        print(f"[consolidate] wrote {out} "
              f"({report['bytes'] / 2 ** 30:.2f} GiB, "
              f"{report['n_params'] / 1e6:.1f} M params)", flush=True)
    return report


# ═══════════════════════════════════════════════════════════════════════════
#  VERIFICATION
#
#  Three checks, because the failure that matters is a consolidator that looks
#  like it worked. `load_state_dict(strict=False)` in loom.eval.policy accepts a
#  tensor of zeros of the right shape without a word.
# ═══════════════════════════════════════════════════════════════════════════

def _model_reference(config_path: str | Path) -> dict[str, torch.Size]:
    """``{key: shape}`` of a freshly built, UNWRAPPED ``LoomModel``.

    Built from the run's own ``config.json``, so the reference is the same model
    the run is training, not a guess. ~3 s on a login node -- construction only,
    never a forward.
    """
    from loom.train.loop import build_model                # noqa: PLC0415

    with open(config_path) as f:
        cfg = json.load(f)
    model = build_model(cfg)
    return {k: v.shape for k, v in model.state_dict().items()}


def _embodiment_of(cfg: dict) -> str:
    """The body this run actually trained, from its own config.

    Defaulting to "libero_franka" made the policy check build LIBERO modules
    against a RoboTwin checkpoint: nothing matched, `loaded` came back empty and
    `torch.stack([])` raised `stack expects a non-empty TensorList` -- an error
    that reads like corruption and is really a wrong default. Batches are
    embodiment-homogeneous by contract, so data.embodiments[0] is the answer.
    """
    e = (cfg.get("data", {}) or {}).get("embodiments") or []
    return str(e[0]) if e else "libero_franka"


def verify(out: str | Path, run_dir: str | Path, *, step: int | None = None,
           config_path: str | Path | None = None, embodiment: str | None = None,
           check_policy: bool = True, verbose: bool = True) -> dict[str, Any]:
    """Prove the consolidated file structurally, numerically, and through eval.

    1. **Structure.** Every key and shape of an unwrapped ``LoomModel``, nothing
       left as a ShardedTensor/DTensor, no ``_fsdp_wrapped_module.`` or
       ``_flat_param`` residue.
    2. **Numerics.** Re-reads all 16 shards from disk. Replicated modules must be
       bit-identical to *every* rank's copy (which also proves ``ReplicaSync``
       kept the ranks together); each sharded tensor's dim-0 slice must be
       bit-identical to the rank that owns it. A tensor of zeros fails this and
       passes check 1.
    3. **Eval.** ``loom.eval.policy.make_policy`` on the file, with the returned
       ``IncompatibleKeys`` inspected rather than trusted.
    """
    out, run_dir = Path(out), Path(run_dir)
    if embodiment is None:
        cfg_p = Path(config_path) if config_path else run_dir / "config.json"
        try:
            with open(cfg_p) as f:
                embodiment = _embodiment_of(json.load(f))
        except OSError:
            embodiment = "libero_franka"
    res: dict[str, Any] = {"out": str(out), "checks": {}, "embodiment": embodiment}
    payload = torch.load(str(out), map_location="cpu", weights_only=False)
    model = payload["model"]
    step = int(payload.get("global_step", step or -1))

    # ── 1. structure ───────────────────────────────────────────────────────
    bad_type = sorted(k for k, v in model.items()
                      if is_sharded(v) or not isinstance(v, torch.Tensor))
    residue = sorted(k for k in model
                     if "_fsdp_wrapped_module" in k or "_flat_param" in k
                     or "_orig_mod." in k)
    # Eval runs under a different interpreter (py3.10 + LIBERO). A payload that
    # needs weights_only=False to unpickle is a payload that needs the *training*
    # stack importable on the eval side; the per-rank shards do, because of
    # ShardedTensor. This one must not.
    try:
        torch.load(str(out), map_location="cpu", weights_only=True, mmap=True)
        wo_err = ""
    except Exception as e:                                  # noqa: BLE001
        wo_err = f"{type(e).__name__}: {e}"[:300]
    c1: dict[str, Any] = {"n_keys": len(model), "non_tensor_or_sharded": bad_type,
                          "fsdp_residue": residue,
                          "loads_with_weights_only": not wo_err,
                          "weights_only_error": wo_err}
    cfg_path = Path(config_path) if config_path else run_dir / "config.json"
    if cfg_path.is_file():
        ref = _model_reference(cfg_path)
        missing = sorted(set(ref) - set(model))
        unexpected = sorted(set(model) - set(ref))
        wrong = sorted(k for k in set(ref) & set(model)
                       if tuple(ref[k]) != tuple(model[k].shape))
        c1.update({"reference": str(cfg_path), "reference_keys": len(ref),
                   "missing_vs_unwrapped": missing,
                   "unexpected_vs_unwrapped": unexpected,
                   "shape_mismatch": wrong})
        c1["pass"] = not (bad_type or residue or missing or unexpected or wrong)
    else:
        c1.update({"reference": None, "pass": not (bad_type or residue)})
    res["checks"]["structure"] = c1

    # ── 2. numerics, against the shards themselves ─────────────────────────
    _, paths = shard_paths(run_dir, step)
    n_rep_cmp = n_sh_cmp = 0
    rep_bad: list[str] = []
    sh_bad: list[str] = []
    ranks_disagree: dict[str, list[int]] = {}
    for r, path in enumerate(paths):
        sd = load_shard(path)["model"]
        for k, v in sd.items():
            if is_sharded(v):
                for shard in v._local_shards:
                    off = list(shard.metadata.shard_offsets)
                    sz = list(shard.metadata.shard_sizes)
                    piece = model[k].narrow(0, off[0], sz[0])
                    n_sh_cmp += 1
                    if not torch.equal(piece, shard.tensor):
                        sh_bad.append(f"rank{r}:{k}@{off}")
            else:
                n_rep_cmp += 1
                if not torch.equal(model[k], v):
                    rep_bad.append(f"rank{r}:{k}")
                    ranks_disagree.setdefault(k, []).append(r)
        del sd
        if verbose:
            print(f"[verify] rank {r:>2} compared", flush=True)
    res["checks"]["numeric"] = {
        "replicated_tensors_compared": n_rep_cmp,
        "replicated_mismatches": rep_bad[:20],
        "n_replicated_mismatches": len(rep_bad),
        "sharded_pieces_compared": n_sh_cmp,
        "sharded_mismatches": sh_bad[:20],
        "n_sharded_mismatches": len(sh_bad),
        "ranks_disagreeing_on_replicated": {k: v for k, v in
                                            list(ranks_disagree.items())[:8]},
        "pass": not rep_bad and not sh_bad,
    }

    # ── 3. eval accepts it, with IncompatibleKeys inspected ────────────────
    del payload, model                       # _verify_policy loads its own copy
    if check_policy:
        res["checks"]["policy"] = _verify_policy(out, embodiment, verbose=verbose)

    res["pass"] = all(c.get("pass") for c in res["checks"].values())
    return res


def _verify_policy(out: Path, embodiment: str, *, verbose: bool = True) -> dict[str, Any]:
    """Load exactly as ``loom.eval.policy._try_real_modules`` does -- then check.

    ``policy.py`` uses ``strict=False`` and only *raises* on missing keys; it
    counts unexpected ones and moves on. Here both are surfaced, because an
    unexpected key means eval built a different module than training saved.
    """
    from loom.eval.policy import (                        # noqa: PLC0415
        _run_model_kwargs, make_policy, policy_provenance, submodule_state,
    )
    from loom.heads.decoder import Decoder                 # noqa: PLC0415
    from loom.heads.proposal import Proposal               # noqa: PLC0415
    from loom.model.estimator import Estimator             # noqa: PLC0415

    payload = torch.load(str(out), map_location="cpu", weights_only=False)
    state = payload["model"]
    # The estimator's architecture flags are NOT in state_dict, so they must come
    # from the run config -- exactly as `make_policy` does a few lines below.
    # Building the default here instead reported `unexpected_keys: ["z_init"]` and
    # failed a checkpoint whose structure and numeric checks both passed and which
    # `make_policy` loaded in the same breath with 339 tensors and 0 unexpected.
    # A verify that disagrees with the loader it is verifying is worse than none.
    est_kw = _run_model_kwargs(out, "estimator")
    # Same for the decoder: `residual` is not a parameter either, so a mismatch
    # is invisible in missing/unexpected keys and only shows up as an eval that
    # feeds ~0.03 rad residuals to the robot as absolute joint targets.
    dec_kw = _run_model_kwargs(out, "decoder")
    mods = {
        "estimator": Estimator(embodiments=[embodiment], **est_kw),
        "proposal": Proposal(),
        "decoder": Decoder(embodiments=[embodiment], default_embodiment=embodiment,
                           **dec_kw),
    }
    detail: dict[str, Any] = {}
    ok = True
    for name, mod in mods.items():
        sd = submodule_state(state, name)
        if sd is None:
            detail[name] = {"error": "submodule_state returned None"}
            ok = False
            continue
        inc = mod.load_state_dict(sd, strict=False)
        missing = list(inc.missing_keys)
        unexpected = list(inc.unexpected_keys)
        loaded = [v for k, v in sd.items() if k not in set(unexpected)]
        detail[name] = {
            "tensors_in_ckpt": len(sd),
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "params_loaded": int(sum(v.numel() for v in loaded)),
            "mean_abs": float(torch.stack([v.float().abs().mean()
                                           for v in loaded]).mean()),
            "n_all_zero": int(sum(1 for v in loaded if float(v.float().abs().sum()) == 0)),
        }
        ok = ok and not missing and not unexpected
    del payload, state

    try:
        policy = make_policy(str(out), embodiment=embodiment, device="cpu")
        prov = policy_provenance(policy)
        detail["make_policy"] = {"is_stub": prov.get("is_stub"),
                                 "ckpt_global_step": prov.get("ckpt_global_step"),
                                 "tower": prov.get("tower"),
                                 "state_dict": prov.get("state_dict")}
        ok = ok and prov.get("is_stub") is False
    except Exception as e:                                  # noqa: BLE001
        detail["make_policy"] = {"error": f"{type(e).__name__}: {e}"}
        ok = False
    detail["pass"] = ok
    if verbose:
        print(f"[verify] policy check pass={ok}", flush=True)
    return detail


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser("loom.train.consolidate")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--step", type=int, default=None,
                   help="default: newest step with a complete shard set")
    p.add_argument("--out", default=None,
                   help="default: <run_dir>_eval/ckpt_<step>.pt")
    p.add_argument("--embodiment", default=None,
                   help="default: data.embodiments[0] from the run config")
    p.add_argument("--config", default=None, help="default: <run_dir>/config.json")
    p.add_argument("--no_verify", action="store_true")
    p.add_argument("--no_policy_check", action="store_true",
                   help="skip make_policy (which loads the 878 M frozen tower)")
    p.add_argument("--verify_only", action="store_true")
    p.add_argument("--pin", action="store_true",
                   help="hardlink the shards out of the live run dir first, so "
                        "ckpt._prune cannot delete them mid-consolidation")
    p.add_argument("--report", default=None, help="write the verification JSON here")
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir)
    step, _ = shard_paths(run_dir, args.step)
    out_dir = (Path(args.out).parent if args.out
               else run_dir.parent / f"{run_dir.name}_eval")
    out = Path(args.out) if args.out else out_dir / f"ckpt_{step:09d}.pt"
    config_path = Path(args.config) if args.config else run_dir / "config.json"

    src = run_dir
    if args.pin:
        src = out_dir / f"shards_{step:09d}"
        pin_shards(run_dir, step, src)
        print(f"[consolidate] pinned {len(list(src.glob('ckpt_*')))} shards "
              f"into {src} (hardlinks; run_dir untouched)", flush=True)

    if not args.verify_only:
        consolidate(src, out, step=step)

    res: dict[str, Any] = {}
    if not args.no_verify:
        res = verify(out, src, step=step, config_path=config_path,
                     embodiment=args.embodiment,
                     check_policy=not args.no_policy_check)
        print(json.dumps(res, indent=2, default=str), flush=True)
        if args.report:
            Path(args.report).parent.mkdir(parents=True, exist_ok=True)
            Path(args.report).write_text(json.dumps(res, indent=2, default=str))
        print(f"[verify] OVERALL {'PASS' if res.get('pass') else 'FAIL'}", flush=True)
        return 0 if res.get("pass") else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
