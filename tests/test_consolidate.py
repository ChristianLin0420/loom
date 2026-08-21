"""`loom.train.consolidate` — the reassembly logic, on the CPU login node.

The real thing is verified against `runs/r0a`'s 16 live shards and, on 8 A100s,
against FSDP's own `FULL_STATE_DICT` all-gather (`logs/resume_check.py`, check
C: 929/929 keys, max_abs_diff 0.0). Neither of those runs here — there is no GPU
and no process group on a login node — so this pins the part that is pure
arithmetic and would silently corrupt a parameter if it were wrong.

The trap being guarded is `torch.chunk(world_size, dim=0)` returning **fewer**
than `world_size` pieces when `size(0) < world_size`. R0-A's
`estimator.type_embed` is `(4, 768)` on 16 ranks: it lives on ranks 0-3 and
ranks 4-15 hold nothing for it. "16 files, 16 pieces, concatenate" produces a
shape-correct, wrong tensor, and `load_state_dict(strict=False)` in
`loom.eval.policy` accepts it without a word.

The fakes below duck-type `torch.distributed._shard`'s objects rather than
building real ones, because constructing a real `ShardedTensor` requires the
process group this module exists to do without.
"""

from __future__ import annotations

import json
import math

import pytest
import torch

from loom.train.consolidate import (
    _config_hash, consolidate, consolidate_state_dict, is_sharded, shard_paths,
)


# ── duck types for torch.distributed._shard ────────────────────────────────
# Module level, not nested, so pickle can find them by reference.

class _Placement:
    def __init__(self, rank: int) -> None:
        self._rank = rank

    def rank(self) -> int:
        return self._rank


class _ShardMetadata:
    def __init__(self, offsets, sizes, rank):
        self.shard_offsets = list(offsets)
        self.shard_sizes = list(sizes)
        self.placement = _Placement(rank)


class _Shard:
    def __init__(self, tensor, metadata):
        self.tensor = tensor
        self.metadata = metadata


class _Props:
    def __init__(self, dtype):
        self.dtype = dtype


class _Meta:
    def __init__(self, size, shards_metadata, dtype):
        self.size = torch.Size(size)
        self.shards_metadata = shards_metadata
        self.tensor_properties = _Props(dtype)


class ShardedTensor:                      # name matters: is_sharded() reads it
    def __init__(self, local_shards, metadata):
        self._local_shards = local_shards
        self._metadata = metadata


def _chunk(tensor: torch.Tensor, world: int):
    """Exactly `fsdp._shard_utils._create_chunk_sharded_tensor`'s tiling."""
    chunks = tensor.chunk(world, dim=0)
    metas, off = [], 0
    for r, c in enumerate(chunks):
        metas.append(_ShardMetadata([off] + [0] * (tensor.dim() - 1),
                                    list(c.shape), r))
        off += c.shape[0]
    return chunks, metas


def _write_fake_run(tmp_path, world: int, step: int = 7, drop=None,
                    config_hash: str = ""):
    """A `world`-rank checkpoint whose ground truth we know. Returns it."""
    torch.manual_seed(0)
    truth = {
        "estimator.w": torch.randn(10, 3),        # 4 ranks -> sizes 3,3,3,1
        "estimator.type_embed": torch.randn(2, 5),  # fewer rows than ranks
        "bank.lam": torch.randn(4, 2),            # replicated, plain tensor
    }
    sharded = ("estimator.w", "estimator.type_embed")
    tiles = {k: _chunk(truth[k], world) for k in sharded}

    for r in range(world):
        model = {}
        for k, v in truth.items():
            if k not in sharded:
                model[k] = v.clone()
                continue
            chunks, metas = tiles[k]
            local = []
            if r < len(chunks) and (drop is None or (k, r) != drop):
                local = [_Shard(chunks[r].clone(), metas[r])]
            model[k] = ShardedTensor(local, _Meta(truth[k].shape, metas,
                                                  truth[k].dtype))
        torch.save({"model": model, "world_size": world, "global_step": step,
                    "config_hash": config_hash},
                   tmp_path / f"ckpt_{step:09d}_rank{r}.pt")
    return truth


def test_reassembles_exactly(tmp_path):
    truth = _write_fake_run(tmp_path, world=4)
    _, paths = shard_paths(tmp_path, 7)
    assert [p.name for p in paths] == [f"ckpt_000000007_rank{r}.pt" for r in range(4)]

    full, report = consolidate_state_dict(paths, "model", verbose=False)

    assert set(full) == set(truth)
    assert report["n_sharded_keys"] == 2
    assert report["n_replicated_keys"] == 1
    # (4 chunks for the 10-row tensor) + (2 for the 2-row one), NOT 4 + 4
    assert report["n_shard_pieces"] == 4 + 2
    for k, v in truth.items():
        assert torch.equal(full[k], v), k
    assert not any(is_sharded(v) for v in full.values())


def test_short_tensor_is_not_spread_over_every_rank(tmp_path):
    """`(2, 5)` on 4 ranks lives on ranks 0-1 only. The other two hold nothing."""
    _write_fake_run(tmp_path, world=4)
    _, paths = shard_paths(tmp_path, 7)
    counts = []
    for p in paths:
        sd = torch.load(p, map_location="cpu", weights_only=False)["model"]
        counts.append(len(sd["estimator.type_embed"]._local_shards))
    assert counts == [1, 1, 0, 0]


def test_a_missing_shard_is_an_error_not_a_zero(tmp_path):
    """The whole point. A hole must not survive as a plausible-looking number."""
    _write_fake_run(tmp_path, world=4, drop=("estimator.w", 2))
    _, paths = shard_paths(tmp_path, 7)
    with pytest.raises(RuntimeError, match="never written|non-finite"):
        consolidate_state_dict(paths, "model", verbose=False)


def test_missing_rank_file_is_refused(tmp_path):
    _write_fake_run(tmp_path, world=4)
    (tmp_path / "ckpt_000000007_rank2.pt").unlink()
    with pytest.raises(RuntimeError, match="not 0.."):
        shard_paths(tmp_path, 7)


def test_chunk_tiling_matches_torch(tmp_path):
    """`ceil`-based offsets and `torch.chunk` must agree, or every offset is off."""
    for rows, world in ((10, 4), (4, 16), (16, 16), (17, 8), (1, 8)):
        t = torch.arange(rows * 2, dtype=torch.float32).view(rows, 2)
        chunks, metas = _chunk(t, world)
        assert sum(c.shape[0] for c in chunks) == rows
        for r, m in enumerate(metas):
            assert m.shard_offsets[0] == math.ceil(rows / world) * r


def test_consolidation_embeds_only_hash_authenticated_resolved_config(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    cfg = {
        "model": {"estimator": {"z_prev_residual": False}, "decoder": {}},
        "data": {"embodiments": ["libero_franka"]},
        "link": {"run_dir": str(run), "stop_at": 7},
    }
    (run / "config.json").write_text(json.dumps(cfg))
    _write_fake_run(run, world=2, config_hash=_config_hash(cfg))
    out = tmp_path / "eval" / "ckpt.pt"

    consolidate(run, out, step=7, verbose=False)
    payload = torch.load(out, map_location="cpu", weights_only=False)
    assert payload["resolved_config"] == {k: v for k, v in cfg.items() if k != "link"}
    assert payload["config_hash"] == _config_hash(payload["resolved_config"])


def test_consolidation_refuses_a_config_from_a_different_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    cfg = {"model": {"estimator": {"z_prev_residual": False}}}
    (run / "config.json").write_text(json.dumps(cfg))
    _write_fake_run(run, world=2, config_hash="not-the-config-on-disk")

    with pytest.raises(RuntimeError, match="different run"):
        consolidate(run, tmp_path / "eval" / "ckpt.pt", step=7, verbose=False)
