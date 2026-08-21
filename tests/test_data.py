"""
LOOM — Team A gate (PLAN §4.A "Done when").

Resampling verified against known-rate fixtures; windows have 5 states and 4
segments at the right offsets; the ``actions=None`` path works; batches are
embodiment-homogeneous; the cache round-trips and a version bump is loud; the
pipeline sustains >=1.3x measured training consumption.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

import contracts as C
from loom.data import cache as CA
from loom.data import canonical as CN
from loom.data.loader import (
    STARVATION_MARGIN,
    CachedWindowDataset,
    HomogeneousSampler,
    LoomLoader,
    collate_window,
    measure_throughput,
)


# ═══════════════════════════════════════════════════════════════════════════
#  SYNTHETIC BODIES  —  LIBERO is single-embodiment, so the dispatch is tested
#  with two bodies of different dof (PLAN §4.A).
# ═══════════════════════════════════════════════════════════════════════════

SYNTH_A = C.register_embodiment(C.EmbodimentSpec(
    name="synth_a", dof=4, env_fps=25.0, n_views=2,
    action_low=(-1.0,) * 4, action_high=(1.0,) * 4,
))
SYNTH_B = C.register_embodiment(C.EmbodimentSpec(
    name="synth_b", dof=9, env_fps=50.0, n_views=2,
    action_low=(-1.0,) * 9, action_high=(1.0,) * 9,
))
CN.register_action_semantics("synth_a", (CN.DELTA,) * 3 + (CN.HOLD,))
CN.register_action_semantics("synth_b", (CN.ABSOLUTE,) * 9)


V, P, F, L = 2, 8, 16, 4          # tiny feature geometry for correctness tests


def _make_body(
    root,
    body: str,
    *,
    n_traj: int = 3,
    n_src: int = 150,
    src_fps: float = 20.0,
    action_free: bool = False,
    codec: str = "fp16",
    n_patches: int = P,
    feat_dim: int = F,
    seed: int = 0,
    recurrent_burn_in: int = 0,
) -> CachedWindowDataset:
    """Canonicalise `n_traj` synthetic demos and write their features to a cache."""
    spec = C.EMBODIMENTS[body]
    rng = np.random.default_rng(seed)
    trajs = []
    for i in range(n_traj):
        actions = None if action_free else rng.normal(0, 0.2, size=(n_src, spec.dof))
        trajs.append(CN.to_canonical(
            n_src_frames=n_src, src_fps=src_fps, embodiment=body,
            traj_id=f"{body}/demo_{i}", actions=actions, lang=f"do task {i}",
        ))

    need = CN.required_source_frames([w for t in trajs for w in CN.segment(t)])
    cspec = CA.CacheSpec(codec, spec.n_views, n_patches, feat_dim, spec.dof, L)
    with CA.FeatureCacheWriter(root, cspec) as w:
        for t in trajs:
            frames = need[t.traj_id]
            w.write(
                t.traj_id,
                frames=frames,
                views=rng.normal(size=(len(frames), spec.n_views, n_patches, feat_dim)),
                proprio=rng.normal(size=(len(frames), spec.dof)),
                lang=rng.normal(size=(L, feat_dim)),
                embodiment=body,
                src_fps=src_fps,
            )
    return CachedWindowDataset(
        trajs, CA.FeatureCache(root), recurrent_burn_in=recurrent_burn_in,
    )


def _make_libero_holdout_source(root, n_tasks: int = 4):
    """Tiny cache with two whole demos per LIBERO-shaped task id."""
    rng = np.random.default_rng(73)
    trajs = []
    for task_index in range(n_tasks):
        suite = f"libero_suite_{task_index // 10}"
        task = f"task_{task_index:02d}"
        for demo_key in ("demo_0", "demo_49"):
            trajs.append(CN.to_canonical(
                n_src_frames=80,
                src_fps=20.0,
                embodiment="libero_franka",
                traj_id=f"{suite}/{task}/{demo_key}",
                actions=rng.normal(0, 0.1, size=(80, 7)),
                lang=f"do {task}",
            ))

    need = CN.required_source_frames([w for t in trajs for w in CN.segment(t)])
    spec = CA.CacheSpec("fp16", 2, P, F, 7, L)
    with CA.FeatureCacheWriter(root, spec) as writer:
        for traj in trajs:
            frames = need[traj.traj_id]
            suite, task, _ = traj.traj_id.split("/")
            writer.write(
                traj.traj_id,
                frames=frames,
                views=rng.normal(size=(len(frames), 2, P, F)),
                proprio=rng.normal(size=(len(frames), 7)),
                lang=rng.normal(size=(L, F)),
                embodiment="libero_franka",
                src_fps=20.0,
                meta={"suite": suite, "task": task},
            )
    return trajs, CA.FeatureCache(root)


# ═══════════════════════════════════════════════════════════════════════════
#  1 · RESAMPLING AGAINST KNOWN-RATE FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("src_fps", [10.0, 20.0, 30.0, 50.0])
def test_absolute_resample_matches_analytic_signal(src_fps):
    """An absolute channel resampled to 30 Hz must track the analytic signal.

    Bound is the exact linear-interpolation error for a sinusoid over a step h:
    max |sin - lerp| = omega^2 h^2 / 8. Only the span the source actually covers
    is compared; past the last sample nothing is extrapolated (tested below).
    """
    omega = 2.0 * math.pi * 0.5
    n_src = int(4.0 * src_fps)
    t_src = np.arange(n_src) / src_fps
    a = np.sin(omega * t_src)[:, None]

    out = CN.resample_actions(a, src_fps, C.FPS_CANONICAL, (CN.ABSOLUTE,))
    t_out = np.arange(len(out)) / C.FPS_CANONICAL
    keep = t_out <= t_src[-1] + 1e-9

    err = np.abs(out[keep, 0] - np.sin(omega * t_out[keep])).max()
    bound = omega ** 2 / src_fps ** 2 / 8.0 * 1.05 + 1e-6
    assert err <= bound, f"{src_fps} Hz -> 30 Hz: err {err:.3e} > {bound:.3e}"


@pytest.mark.parametrize("src_fps", [10.0, 20.0, 30.0, 50.0])
def test_canonical_rate_is_thirty(src_fps):
    """The whole point: N source seconds become N canonical seconds at 30 Hz."""
    n_src = int(6.0 * src_fps)
    n_can = CN.canonical_action_count(n_src, src_fps)
    assert n_can == int(6.0 * C.FPS_CANONICAL)
    # H_OP canonical steps are 267 ms of wall clock regardless of src_fps
    assert C.H_OP / C.FPS_CANONICAL == pytest.approx(8 / 30)


@pytest.mark.parametrize("src_fps", [10.0, 20.0, 30.0, 50.0])
def test_resample_is_identity_at_matching_rates(src_fps):
    rng = np.random.default_rng(0)
    a = rng.normal(size=(40, 3))
    kinds = (CN.DELTA, CN.ABSOLUTE, CN.HOLD)
    out = CN.resample_actions(a, src_fps, src_fps, kinds)
    np.testing.assert_allclose(out, a.astype(np.float32), atol=1e-6)


def test_delta_resample_preserves_integrated_motion_20_to_30():
    """THE delta test. 20 -> 30 Hz must preserve the integral, not the step size."""
    rng = np.random.default_rng(1)
    a = rng.normal(0, 0.1, size=(100, 3))
    out = CN.resample_actions(a, 20.0, 30.0, (CN.DELTA,) * 3)

    assert out.shape == (150, 3)
    np.testing.assert_allclose(out.sum(0), a.sum(0), atol=1e-5)

    # the same array read as absolute is the naive bug: 1.5x the commanded motion
    naive = CN.resample_actions(a, 20.0, 30.0, (CN.ABSOLUTE,) * 3)
    assert np.abs(naive.sum(0)).sum() > 1.3 * np.abs(a.sum(0)).sum()


def test_delta_constant_stream_rescales_per_step_magnitude_exactly():
    """A constant delta of 0.3 at 20 Hz is 0.2 per step at 30 Hz. Same motion."""
    a = np.full((100, 2), 0.3)
    out = CN.resample_actions(a, 20.0, 30.0, (CN.DELTA,) * 2)
    np.testing.assert_allclose(out, 0.3 * 20.0 / 30.0, atol=1e-6)
    np.testing.assert_allclose(out.sum(0), a.sum(0), atol=1e-4)


def test_delta_downsample_also_preserves_integral():
    rng = np.random.default_rng(2)
    a = rng.normal(0, 0.1, size=(120, 4))           # 50 Hz, 2.4 s
    out = CN.resample_actions(a, 50.0, 30.0, (CN.DELTA,) * 4)
    assert out.shape == (72, 4)                      # 2.4 s at 30 Hz
    np.testing.assert_allclose(out.sum(0), a.sum(0), atol=1e-5)


def test_delta_round_trip_through_canonical_preserves_motion():
    """20 -> 30 -> 20 is what eval does in reverse (PLAN §4.F). Integral survives."""
    rng = np.random.default_rng(3)
    a = rng.normal(0, 0.1, size=(60, 3))
    up = CN.resample_actions(a, 20.0, 30.0, (CN.DELTA,) * 3)
    back = CN.resample_actions(up, 30.0, 20.0, (CN.DELTA,) * 3)
    assert back.shape == a.shape
    np.testing.assert_allclose(back.sum(0), a.sum(0), atol=1e-5)


def test_hold_channel_keeps_discrete_values():
    """A latched gripper must never be interpolated into a value it never held."""
    a = np.where(np.arange(60)[:, None] < 30, -1.0, 1.0)
    out = CN.resample_actions(a, 20.0, 30.0, (CN.HOLD,))
    assert set(np.unique(out).tolist()) <= {-1.0, 1.0}
    lerp = CN.resample_actions(a, 20.0, 30.0, (CN.ABSOLUTE,))
    assert not set(np.unique(lerp).tolist()) <= {-1.0, 1.0}   # the bug it prevents


def test_resample_does_not_extrapolate_past_the_record():
    """Overrunning n_dst holds absolute targets and emits zero delta, never invents."""
    a = np.stack([np.arange(10.0), np.ones(10)], axis=1)
    out = CN.resample_actions(a, 10.0, 10.0, (CN.ABSOLUTE, CN.DELTA), n_dst=14)
    np.testing.assert_allclose(out[10:, 0], 9.0)      # absolute latches
    np.testing.assert_allclose(out[10:, 1], 0.0)      # delta stops commanding


def test_libero_semantics_are_six_delta_and_one_hold():
    kinds = CN.action_semantics("libero_franka")
    assert kinds == (CN.DELTA,) * 6 + (CN.HOLD,)
    assert len(kinds) == C.EMBODIMENTS["libero_franka"].dof


def test_action_semantics_must_be_declared():
    C.register_embodiment(C.EmbodimentSpec(
        "undeclared_body", 3, 30.0, 1, (-1.0,) * 3, (1.0,) * 3))
    with pytest.raises(KeyError, match="no safe default"):
        CN.action_semantics("undeclared_body")


def test_action_semantics_conflict_is_loud():
    with pytest.raises(ValueError, match="conflicting"):
        CN.register_action_semantics("libero_franka", (CN.ABSOLUTE,) * 7)


def test_observations_are_nearest_sampled_not_interpolated():
    idx = CN.obs_source_indices(100, 20.0)
    assert idx.shape == (CN.canonical_obs_count(100, 20.0),)
    assert idx.dtype == np.int64
    assert idx[0] == 0 and idx[-1] <= 99
    assert (np.diff(idx) >= 0).all()
    t_can = np.arange(len(idx)) / C.FPS_CANONICAL
    assert (np.abs(idx / 20.0 - t_can) <= 0.5 / 20.0 + 1e-9).all()


# ═══════════════════════════════════════════════════════════════════════════
#  2 · WINDOW STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════

def test_window_offsets_are_the_contract_frames():
    traj = CN.to_canonical(200, 20.0, "libero_franka", "t0",
                           actions=np.zeros((200, 7)))
    ws = CN.segment(traj)
    assert ws, "a 200-frame 20 Hz demo must yield windows"
    for w in ws:
        assert len(w.obs_src_index) == C.N_STATES
        assert w.canonical_frames == tuple(w.start + f for f in C.CANONICAL_FRAMES)
        assert w.act_hi - w.act_lo == C.H_PLAN
        assert w.src_fps == 20.0 > 0


def test_window_actions_are_depth_by_h_op():
    traj = CN.to_canonical(200, 20.0, "libero_franka", "t0",
                           actions=np.arange(200 * 7, dtype=np.float64).reshape(200, 7))
    w = CN.segment(traj)[1]
    a = CN.window_actions(traj, w)
    assert a.shape == (C.DEPTH, C.H_OP, 7)
    # segment d is exactly canonical steps [start + 8d, start + 8d + 8)
    flat = traj.actions[w.act_lo:w.act_hi]
    for d in range(C.DEPTH):
        np.testing.assert_allclose(a[d], flat[d * C.H_OP:(d + 1) * C.H_OP])


def test_stride_default_is_h_op_and_shrinks_the_cache():
    """stride == H_OP keeps every boundary on a multiple of 8: cache 1 frame in 8."""
    traj = CN.to_canonical(200, 20.0, "libero_franka", "t0", actions=np.zeros((200, 7)))
    ws = CN.segment(traj)
    assert ws[1].start - ws[0].start == C.H_OP
    need = CN.required_source_frames(ws)["t0"]
    assert len(need) < traj.n_frames / 4


def test_short_trajectory_is_dropped_not_padded():
    """Fewer than H_PLAN + 1 canonical frames -> zero windows, never zero-padding."""
    n_min = CN.min_source_frames(20.0)
    short = CN.to_canonical(n_min - 1, 20.0, "libero_franka", "short",
                            actions=np.zeros((n_min - 1, 7)))
    assert CN.canonical_obs_count(n_min - 1, 20.0) < C.H_PLAN + 1
    assert CN.segment(short) == []

    ok = CN.to_canonical(n_min, 20.0, "libero_franka", "ok", actions=np.zeros((n_min, 7)))
    assert len(CN.segment(ok)) == 1


def test_action_free_trajectory_segments_without_actions():
    traj = CN.to_canonical(200, 20.0, "libero_franka", "t0", actions=None)
    ws = CN.segment(traj)
    assert ws and all(w.action_free for w in ws)
    assert CN.window_actions(traj, ws[0]) is None


def test_to_env_rate_inverts_the_canonical_clock():
    """One decoded 8-step canonical segment is 5.333 env steps at 20 Hz."""
    n_env = 5
    seg = np.full((C.H_OP, 7), 0.1)
    seg[:, 6] = 1.0
    out = CN.to_env_rate(seg, "libero_franka", n_env)
    assert out.shape == (n_env, 7)
    # delta channels: motion over the covered span scales with the covered time
    covered = min(n_env / 20.0, C.H_OP / C.FPS_CANONICAL)
    np.testing.assert_allclose(out[:, 0].sum(), 0.1 * C.FPS_CANONICAL * covered, atol=1e-5)
    assert set(np.unique(out[:, 6]).tolist()) == {1.0}


# ═══════════════════════════════════════════════════════════════════════════
#  5 · CACHE
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("codec", list(CA.CODECS))
def test_cache_round_trips(tmp_path, codec):
    rng = np.random.default_rng(0)
    spec = CA.CacheSpec(codec, 2, 8, 16, 7, 4)
    views = rng.normal(size=(6, 2, 8, 16)).astype(np.float32)
    proprio = rng.normal(size=(6, 7)).astype(np.float32)
    lang = rng.normal(size=(4, 16)).astype(np.float32)
    frames = [0, 8, 16, 24, 32, 40]

    with CA.FeatureCacheWriter(tmp_path / "c", spec) as w:
        w.write("demo", frames=frames, views=views, proprio=proprio, lang=lang,
                embodiment="libero_franka", src_fps=20.0)

    cache = CA.FeatureCache(tmp_path / "c")
    got = cache.read("demo", [0, 16, 40])
    assert got["views"].shape == (3, 2, 8, 16)
    assert got["src_fps"] == 20.0

    ref = views[[0, 2, 5]]
    tol = 3e-3 if codec == "fp16" else 2e-2          # int8 is lossy by design
    rel = np.linalg.norm(got["views"].astype(np.float32) - ref) / np.linalg.norm(ref)
    assert rel < tol, f"{codec}: relative L2 {rel:.4f}"
    np.testing.assert_allclose(got["proprio"], proprio[[0, 2, 5]], atol=1e-6)
    np.testing.assert_allclose(got["lang"].astype(np.float32), lang, atol=3e-3)


def test_cache_version_bump_is_a_loud_failure(tmp_path):
    spec = CA.CacheSpec("fp16", 1, 4, 8, 7, 2)
    with CA.FeatureCacheWriter(tmp_path / "c", spec) as w:
        w.write("d", frames=[0], views=np.zeros((1, 1, 4, 8)), proprio=np.zeros((1, 7)),
                lang=np.zeros((2, 8)), embodiment="libero_franka", src_fps=20.0)

    man = tmp_path / "c" / CA.MANIFEST_NAME
    doc = json.loads(man.read_text())
    doc["format_version"] = CA.CACHE_FORMAT_VERSION + 1
    man.write_text(json.dumps(doc))

    with pytest.raises(CA.CacheFormatError, match="format version"):
        CA.FeatureCache(tmp_path / "c")


def test_cache_rejects_wrong_shapes(tmp_path):
    spec = CA.CacheSpec("fp16", 2, 8, 16, 7, 4)
    with CA.FeatureCacheWriter(tmp_path / "c", spec) as w:
        with pytest.raises(CA.CacheFormatError):
            w.write("d", frames=[0], views=np.zeros((1, 2, 8, 99)),
                    proprio=np.zeros((1, 7)), lang=np.zeros((4, 16)),
                    embodiment="libero_franka", src_fps=20.0)


def test_cache_missing_frame_names_the_stride(tmp_path):
    spec = CA.CacheSpec("fp16", 1, 4, 8, 7, 2)
    with CA.FeatureCacheWriter(tmp_path / "c", spec) as w:
        w.write("d", frames=[0, 8], views=np.zeros((2, 1, 4, 8)), proprio=np.zeros((2, 7)),
                lang=np.zeros((2, 8)), embodiment="libero_franka", src_fps=20.0)
    with pytest.raises(KeyError, match="stride"):
        CA.FeatureCache(tmp_path / "c").read("d", [0, 3])


def test_int8_reconstruction_error_is_bounded(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 2, 32, 64)).astype(np.float32)
    x[..., ::13] *= 8.0                                # ViT-style channel outliers
    q, s = CA.quantize_int8(x)
    rel = np.linalg.norm(CA.dequantize_int8(q, s).astype(np.float32) - x) / np.linalg.norm(x)
    assert rel < 0.05, f"int8 relative L2 {rel:.4f}"


def test_bytes_per_window_matches_the_plan_estimate():
    """PLAN §4.A: ~1 MiB per 2-stream LIBERO state, 5-15 MiB per window."""
    fp16 = CA.CacheSpec("fp16", 2, 196, 1152, 7, 16)
    int8 = CA.CacheSpec("int8", 2, 196, 1152, 7, 16)
    assert 0.8 < fp16.bytes_per_frame / 2 ** 20 < 1.2
    assert 4.0 < fp16.bytes_per_window / 2 ** 20 < 6.0
    assert int8.bytes_per_window < 0.55 * fp16.bytes_per_window


# ═══════════════════════════════════════════════════════════════════════════
#  4 · EMBODIMENT-HOMOGENEOUS BATCHES
# ═══════════════════════════════════════════════════════════════════════════

def test_batch_has_five_states_four_segments_and_positive_src_fps(tmp_path):
    ds = _make_body(tmp_path / "a", "libero_franka")
    loader = LoomLoader({"libero_franka": ds}, batch_size=4, seed=0)
    batch = next(iter(loader.batches(0, 1)))

    assert len(batch["feats"]) == C.N_STATES
    assert batch["actions"].shape == (4, C.DEPTH, C.H_OP, 7)
    assert batch["src_fps"] > 0
    for d in range(C.DEPTH):
        C.assert_action_segment(batch["actions"][:, d], batch["embodiment"])
    for f in batch["feats"]:
        assert f["views"].shape == (4, 2, P, F)
        assert f["proprio"].shape == (4, 7)
        assert f["lang"].shape == (4, L, F)


def test_recurrent_burn_in_uses_real_prior_operator_boundaries(tmp_path):
    base = _make_body(tmp_path / "a", "libero_franka", n_traj=2, seed=7)
    trajs = list(base._traj.values())
    burned = CachedWindowDataset(
        trajs, base.cache, recurrent_burn_in=4,
    )

    # Every sufficiently long trajectory loses exactly its first four windows.
    assert len(burned) == len(base) - 4 * len(trajs)
    w = burned.windows[0]
    assert w.start == 4 * C.H_OP
    traj = burned._traj[w.traj_id]
    expected_src = tuple(
        int(traj.obs_src_index[t]) for t in (0, C.H_OP, 2 * C.H_OP, 3 * C.H_OP)
    )
    blob = base.cache.read(w.traj_id, expected_src)
    sample = burned[0]
    assert len(sample["burn_in_feats"]) == 4
    for i, feat in enumerate(sample["burn_in_feats"]):
        torch.testing.assert_close(feat["views"], torch.from_numpy(blob["views"][i]))
        torch.testing.assert_close(feat["proprio"], torch.from_numpy(blob["proprio"][i]))

    # Burn-in changes history only. The five main observations and labelled
    # actions are identical to the historical window with the same start.
    j = next(i for i, old in enumerate(base.windows)
             if old.traj_id == w.traj_id and old.start == w.start)
    old = base[j]
    for got, want in zip(sample["feats"], old["feats"]):
        for key in ("views", "proprio", "lang"):
            assert torch.equal(got[key], want[key])
    assert torch.equal(sample["actions"], old["actions"])


def test_zero_burn_in_is_exactly_backward_compatible(tmp_path):
    implicit = _make_body(tmp_path / "a", "libero_franka", seed=4)
    explicit = CachedWindowDataset(
        list(implicit._traj.values()), implicit.cache, recurrent_burn_in=0,
    )
    assert len(explicit) == len(implicit)
    for i in (0, len(implicit) // 2, len(implicit) - 1):
        old, new = implicit[i], explicit[i]
        assert old.keys() == new.keys()
        assert "burn_in_feats" not in old and "burn_in_feats" not in new
        assert old["embodiment"] == new["embodiment"]
        assert old["src_fps"] == new["src_fps"]
        assert torch.equal(old["actions"], new["actions"])
        assert torch.equal(old["lang"], new["lang"])
        for got, want in zip(new["feats"], old["feats"]):
            for key in ("views", "proprio", "lang"):
                assert torch.equal(got[key], want[key])


@pytest.mark.parametrize("bad", [-1, 1.5, True, "4"])
def test_recurrent_burn_in_rejects_invalid_lengths(tmp_path, bad):
    base = _make_body(tmp_path / "a", "libero_franka", n_traj=1)
    with pytest.raises(ValueError, match="recurrent_burn_in"):
        CachedWindowDataset(
            list(base._traj.values()), base.cache, recurrent_burn_in=bad,
        )


def test_burn_in_collates_separately_and_rejects_mixed_lengths(tmp_path):
    base = _make_body(tmp_path / "a", "libero_franka", n_traj=1)
    burned = CachedWindowDataset(
        list(base._traj.values()), base.cache, recurrent_burn_in=4,
    )
    batch = collate_window([burned[0], burned[1]])
    assert len(batch["burn_in_feats"]) == 4
    assert len(batch["feats"]) == C.N_STATES
    for feat in batch["burn_in_feats"]:
        assert feat["views"].shape == (2, 2, P, F)
        assert feat["proprio"].shape == (2, 7)
        assert feat["lang"].shape == (2, L, F)

    with pytest.raises(ValueError, match="burn-in lengths"):
        collate_window([base[0], burned[0]])


def test_action_free_path_works_end_to_end(tmp_path):
    ds = _make_body(tmp_path / "af", "libero_franka", action_free=True)
    loader = LoomLoader({"libero_franka": ds}, batch_size=4, seed=0)
    seen = 0
    for batch in loader.batches(0, 5):
        assert batch["actions"] is None
        assert len(batch["feats"]) == C.N_STATES
        assert batch["feats"][0]["views"].shape[0] == 4
        seen += 1
    assert seen == 5


def test_batches_are_embodiment_homogeneous_and_both_bodies_appear(tmp_path):
    a = _make_body(tmp_path / "a", "synth_a", seed=1)
    b = _make_body(tmp_path / "b", "synth_b", seed=2)
    loader = LoomLoader({"synth_a": a, "synth_b": b}, batch_size=4, seed=7)

    seen: dict[str, int] = {}
    for batch in loader.batches(0, 48):
        body = batch["embodiment"]
        seen[body] = seen.get(body, 0) + 1
        dof = C.EMBODIMENTS[body].dof
        assert batch["actions"].shape == (4, C.DEPTH, C.H_OP, dof)
        for f in batch["feats"]:
            assert f["proprio"].shape == (4, dof)
        for d in range(C.DEPTH):
            C.assert_action_segment(batch["actions"][:, d], body)
    assert set(seen) == {"synth_a", "synth_b"}, seen
    assert min(seen.values()) > 0


def test_collate_refuses_a_mixed_batch(tmp_path):
    a = _make_body(tmp_path / "a", "synth_a", seed=1)
    b = _make_body(tmp_path / "b", "synth_b", seed=2)
    with pytest.raises(ValueError, match="homogeneous"):
        collate_window([a[0], b[0]])


def test_dataset_refuses_two_bodies():
    t1 = CN.to_canonical(100, 20.0, "synth_a", "x", actions=np.zeros((100, 4)))
    t2 = CN.to_canonical(100, 20.0, "synth_b", "y", actions=np.zeros((100, 9)))
    with pytest.raises(ValueError, match="one embodiment"):
        CachedWindowDataset([t1, t2], cache=None)


# ═══════════════════════════════════════════════════════════════════════════
#  7 · SAMPLING DETERMINISM
# ═══════════════════════════════════════════════════════════════════════════

def _sampler(n=64, bs=4, world=2, seed=0, sizes=None):
    return HomogeneousSampler(sizes or {"synth_a": n}, batch_size=bs,
                              world_size=world, seed=seed)


def _task_sampler(seed=0, bs=2, world=2):
    pools = {
        "short": np.arange(0, 4),
        "medium": np.arange(4, 12),
        "long": np.arange(12, 24),
    }
    return HomogeneousSampler(
        {"synth_a": 24}, batch_size=bs, world_size=world, seed=seed,
        sampling="uniform_task", task_indices={"synth_a": pools},
    ), pools


def test_same_seed_step_rank_gives_same_indices():
    s1, s2 = _sampler(), _sampler()
    for step in range(20):
        for rank in (0, 1):
            b1, i1 = s1.batch_at(step, rank)
            b2, i2 = s2.batch_at(step, rank)
            assert b1 == b2
            np.testing.assert_array_equal(i1, i2)


def test_different_rank_gives_different_indices():
    s = _sampler()
    for step in range(20):
        _, i0 = s.batch_at(step, 0)
        _, i1 = s.batch_at(step, 1)
        assert set(i0.tolist()).isdisjoint(i1.tolist())


def test_different_seed_gives_different_indices():
    a, b = _sampler(seed=0), _sampler(seed=1)
    diff = sum(
        not np.array_equal(a.batch_at(t, 0)[1], b.batch_at(t, 0)[1]) for t in range(16)
    )
    assert diff >= 15


def test_uniform_task_sampling_is_deterministic_at_an_arbitrary_resume_step():
    a, _ = _task_sampler(seed=11)
    b, _ = _task_sampler(seed=11)
    for step in range(30):
        for rank in range(2):
            np.testing.assert_array_equal(a.batch_at(step, rank)[1],
                                          b.batch_at(step, rank)[1])

    # A newly constructed sampler can seek directly to a resumed global step;
    # no replay or hidden task cursor is allowed.
    resumed, _ = _task_sampler(seed=11)
    for rank in range(2):
        np.testing.assert_array_equal(resumed.batch_at(137, rank)[1],
                                      a.batch_at(137, rank)[1])


def test_uniform_task_sampling_balances_tasks_then_windows():
    sampler, pools = _task_sampler(seed=5)
    owner = {int(i): task for task, idx in pools.items() for i in idx}
    by_task = {task: [] for task in pools}

    # 9 global steps * 4 samples = 36 = 12 draws from each of three tasks.
    # The underlying pools are deliberately 4/8/12 windows.
    for step in range(9):
        for rank in range(2):
            _, idx = sampler.batch_at(step, rank)
            for i in idx:
                by_task[owner[int(i)]].append(int(i))

    assert {task: len(idx) for task, idx in by_task.items()} == {
        "short": 12, "medium": 12, "long": 12,
    }
    for task, seen in by_task.items():
        pool = pools[task]
        # Each task has its own without-replacement permutation. Across repeated
        # task epochs every window's exposure can differ by at most one.
        counts = [seen.count(int(i)) for i in pool]
        assert set(seen[:len(pool)]) == set(pool.tolist())
        assert max(counts) - min(counts) <= 1


def test_epoch_is_covered_without_duplicates_within_a_rank():
    n, bs, world = 64, 4, 2
    s = _sampler(n=n, bs=bs, world=world)
    spe = s.steps_per_epoch("synth_a")
    assert spe == n // (bs * world)

    per_rank = {r: [] for r in range(world)}
    for step in range(spe):
        for r in range(world):
            per_rank[r].extend(s.batch_at(step, r)[1].tolist())
    for r, idx in per_rank.items():
        assert len(idx) == len(set(idx)) == spe * bs, f"rank {r} saw duplicates"
    allidx = sorted(per_rank[0] + per_rank[1])
    assert allidx == list(range(n)), "one epoch must cover every window exactly once"


def test_next_epoch_reshuffles():
    s = _sampler()
    spe = s.steps_per_epoch("synth_a")
    e0 = [s.batch_at(t, 0)[1].tolist() for t in range(spe)]
    e1 = [s.batch_at(spe + t, 0)[1].tolist() for t in range(spe)]
    assert e0 != e1


def test_embodiment_choice_is_rank_independent():
    """Every rank must run the same body at the same step or collectives deadlock."""
    s = _sampler(world=4, sizes={"synth_a": 64, "synth_b": 128})
    for step in range(200):
        bodies = {s.batch_at(step, r)[0] for r in range(4)}
        assert len(bodies) == 1


def test_local_step_is_contiguous_per_body():
    """The block schedule must count draws exactly — this is what resume relies on."""
    s = _sampler(world=1, sizes={"synth_a": 64, "synth_b": 128})
    seen: dict[str, list[int]] = {"synth_a": [], "synth_b": []}
    for step in range(300):
        seen[s.embodiment_at(step)].append(s.local_step(step))
    for body, ls in seen.items():
        assert ls == list(range(len(ls))), body


def test_body_share_tracks_weights():
    s = _sampler(world=1, sizes={"synth_a": 64, "synth_b": 192})
    n = 640
    share = sum(s.embodiment_at(t) == "synth_b" for t in range(n)) / n
    assert abs(share - 0.75) < 0.05


@pytest.mark.parametrize("sampling", ["uniform_window", "uniform_task"])
@pytest.mark.parametrize("recurrent_burn_in", [0, 4])
def test_resume_needs_only_global_step(tmp_path, sampling, recurrent_burn_in):
    """Team D's checkpoint stores no sampler cursor. It must not have to."""
    ds = _make_body(
        tmp_path / "a", "libero_franka",
        recurrent_burn_in=recurrent_burn_in,
    )
    l1 = LoomLoader({"libero_franka": ds}, batch_size=4, seed=3,
                    sampling=sampling)
    ref = [(
        b["feats"][0]["views"].clone(),
        [f["views"].clone() for f in b.get("burn_in_feats", ())],
    ) for b in l1.batches(0, 6)]
    assert l1.state_dict() == {"global_step": 6}

    l2 = LoomLoader({"libero_franka": ds}, batch_size=4, seed=3,
                    sampling=sampling)
    l2.load_state_dict({"global_step": 4})
    resumed = list(l2.batches(l2.global_step, 2))
    for i, got in enumerate(resumed):
        torch.testing.assert_close(got["feats"][0]["views"], ref[4 + i][0])
        assert len(got.get("burn_in_feats", ())) == len(ref[4 + i][1])
        for got_prefix, want_prefix in zip(
                got.get("burn_in_feats", ()), ref[4 + i][1]):
            torch.testing.assert_close(got_prefix["views"], want_prefix)


def test_sampler_rejects_a_body_too_small_for_one_batch():
    with pytest.raises(ValueError, match="cannot fill one global batch"):
        HomogeneousSampler({"synth_a": 6}, batch_size=4, world_size=2)


def test_sampler_rejects_unknown_sampling_mode():
    with pytest.raises(ValueError, match="sampling must be one of"):
        HomogeneousSampler({"synth_a": 16}, batch_size=4, sampling="taskish")


# ═══════════════════════════════════════════════════════════════════════════
#  6 · THROUGHPUT  —  >= 1.3x measured training consumption
# ═══════════════════════════════════════════════════════════════════════════

def test_pipeline_sustains_training_consumption(tmp_path):
    """Realistic geometry: V=2, P=196, F=1152 fp16 -> 4.34 MiB per 5-state window.

    Measured in both modes because the deployment picks whichever is faster on
    the node it lands on: in-process (no IPC, no shared memory) and worker-fed.
    On a node with a 64 MiB /dev/shm the worker queue is capped to one batch and
    the IPC copy dominates, so in-process wins; on a compute node with a large
    /dev/shm the workers overlap I/O with the step and win instead.
    """
    from loom.data.loader import shm_free_bytes

    ds = _make_body(tmp_path / "hot", "libero_franka",
                    n_traj=3, n_src=150, n_patches=196, feat_dim=1152)
    consumption_hz = float(__import__("os").environ.get("LOOM_TRAIN_STEP_HZ", "5.0"))

    results = {}
    for nw in (0, 2):
        loader = LoomLoader({"libero_franka": ds}, batch_size=4, seed=0, num_workers=nw)
        tp = measure_throughput(loader, n_batches=16, warmup=4)
        results[nw] = tp
        print(f"\nloader workers={nw} (effective {loader.effective_workers}/"
              f"prefetch {loader.effective_prefetch}): {tp}  "
              f"ratio {tp.ratio(consumption_hz):.1f}x at {consumption_hz} step/s")
    print(f"/dev/shm free: {shm_free_bytes() / 2 ** 20:.0f} MiB")

    best = max(results.values(), key=lambda t: t.batches_per_s)
    assert best.mib_per_batch > 4 * 4.0, "fixture is not moving realistic bytes"
    assert best.sustains(consumption_hz), (
        f"{best.batches_per_s:.2f} batch/s vs {consumption_hz} step/s consumed "
        f"= {best.ratio(consumption_hz):.2f}x, below {STARVATION_MARGIN}x"
    )


def test_worker_queue_is_sized_to_shared_memory(tmp_path):
    """A 64 MiB /dev/shm cannot hold a LIBERO prefetch queue. Shrink, don't crash."""
    from loom.data.loader import fit_workers, shm_headroom

    need, free = shm_headroom(bytes_per_batch=8 * 4 * 2 ** 20, num_workers=4,
                              prefetch_factor=2)
    assert need == 8 * 4 * 2 ** 20 * 4 * 3
    assert free >= 0

    # a batch larger than all of /dev/shm can only be served in-process
    assert fit_workers(free + 2 ** 30, 8, 4) == (0, 4)
    # a tiny batch keeps everything that was asked for
    assert fit_workers(1024, 8, 4) == (8, 4)

    ds = _make_body(tmp_path / "s", "libero_franka", n_traj=1, n_src=120)
    loader = LoomLoader({"libero_franka": ds}, batch_size=4, num_workers=1)
    assert loader.bytes_per_batch() > 0
    next(iter(loader.batches(0, 1)))
    assert 0 <= loader.effective_workers <= loader.num_workers


def test_burn_in_bytes_are_included_in_worker_queue_accounting(tmp_path):
    base = _make_body(tmp_path / "a", "libero_franka", n_traj=1)
    burned = CachedWindowDataset(
        list(base._traj.values()), base.cache, recurrent_burn_in=4,
    )
    batch_size = 4
    old = LoomLoader({"libero_franka": base}, batch_size=batch_size)
    new = LoomLoader({"libero_franka": burned}, batch_size=batch_size)
    one = collate_window([burned[0]])
    prefix_bytes = sum(
        feat[key].nbytes
        for feat in one["burn_in_feats"]
        for key in ("views", "proprio", "lang")
    )
    assert new.bytes_per_batch() == old.bytes_per_batch() + batch_size * prefix_bytes


# ═══════════════════════════════════════════════════════════════════════════
#  FACTORY  —  the interface loom.train.loop probes for
# ═══════════════════════════════════════════════════════════════════════════

def test_build_loader_accepts_the_loops_call_signature():
    """REGRESSION: the loop calls fn(cfg, rank=, world=, seed=, device=).

    It previously found `LoomLoader` (which wants datasets, not a config), the
    TypeError was swallowed, and the run silently trained on stub windows.
    """
    import inspect

    from loom.data.loader import build_loader

    sig = inspect.signature(build_loader)
    sig.bind({}, rank=0, world=1, seed=0, device="cpu")     # must not raise

    # the loop probes these names in order and takes the first that exists
    import loom.data.loader as LD
    probed = [n for n in ("build_loader", "WindowLoader", "Loader", "LoomLoader")
              if getattr(LD, n, None) is not None]
    assert probed[0] == "build_loader", (
        f"the loop would pick {probed[0]!r} first; it must find build_loader"
    )


def test_loader_exposes_the_samplers_runtime_interface(tmp_path):
    """The loop drives `.next(step)` and checkpoints `.state_dict()`."""
    ds = _make_body(tmp_path / "a", "libero_franka")
    loader = LoomLoader({"libero_franka": ds}, batch_size=4, seed=5)
    for name in ("next", "state_dict", "load_state_dict", "embodiment_for"):
        assert callable(getattr(loader, name)), name
    assert loader.n_windows == len(ds)


def test_next_matches_the_batches_generator(tmp_path):
    ds = _make_body(tmp_path / "a", "libero_franka")
    a = LoomLoader({"libero_franka": ds}, batch_size=4, seed=5)
    b = LoomLoader({"libero_franka": ds}, batch_size=4, seed=5)
    ref = [w["feats"][0]["views"].clone() for w in b.batches(0, 4)]
    for s in range(4):
        torch.testing.assert_close(a.next(s)["feats"][0]["views"], ref[s])


def test_next_reseeks_on_a_jump(tmp_path):
    """A resumed link calls next(4137) first and must get step 4137's batch."""
    ds = _make_body(tmp_path / "a", "libero_franka")
    a = LoomLoader({"libero_franka": ds}, batch_size=4, seed=5)
    sequential = [a.next(s)["feats"][0]["views"].clone() for s in range(6)]

    b = LoomLoader({"libero_franka": ds}, batch_size=4, seed=5)
    torch.testing.assert_close(b.next(5)["feats"][0]["views"], sequential[5])
    torch.testing.assert_close(b.next(2)["feats"][0]["views"], sequential[2])
    torch.testing.assert_close(b.next(3)["feats"][0]["views"], sequential[3])


def test_load_state_dict_tolerates_a_stub_checkpoint(tmp_path):
    """Migrating off the stub sampler must not be blocked by its cursor format."""
    ds = _make_body(tmp_path / "a", "libero_franka")
    loader = LoomLoader({"libero_franka": ds}, batch_size=4, seed=5)
    loader.next(7)
    assert loader.state_dict() == {"global_step": 7}
    loader.load_state_dict({"cursor": 123, "epoch": 1})      # stub format
    assert loader.state_dict() == {"global_step": 7}
    loader.load_state_dict({"global_step": 11})
    assert loader.state_dict() == {"global_step": 11}


def test_resolve_cache_root_precedence(tmp_path, monkeypatch):
    from loom.data.loader import BASE_CACHE_DIR_PLACEHOLDER, resolve_cache_root

    monkeypatch.setenv("LOOM_CACHE_DIR", "/env/cache")

    # the base.yaml placeholder is a default, not a choice: env wins
    base = {"data": {"cache_dir": BASE_CACHE_DIR_PLACEHOLDER}}
    assert resolve_cache_root(base) == Path("/env/cache")

    # a run config that really set cache_dir beats the env
    explicit = {"data": {"cache_dir": "/run/cache"}}
    assert resolve_cache_root(explicit) == Path("/run/cache")

    # an explicit argument beats everything
    assert resolve_cache_root(explicit, "/arg/cache") == Path("/arg/cache")

    # with no env at all the placeholder is used as written
    monkeypatch.delenv("LOOM_CACHE_DIR")
    assert resolve_cache_root(base) == Path(BASE_CACHE_DIR_PLACEHOLDER)


def test_build_loader_refuses_stub_source():
    from loom.data.loader import DataConfigError, build_loader

    with pytest.raises(DataConfigError, match="stub"):
        build_loader({"data": {"source": "stub"}}, rank=0, world=1, seed=0, device="cpu")


def test_build_loader_dies_on_a_missing_cache(tmp_path):
    """A real source with no cache must raise, never substitute stubs."""
    from loom.data.loader import DataConfigError, build_loader

    cfg = {"run": {"seed": 0},
           "data": {"source": "libero", "embodiments": ["libero_franka"],
                    "cache_dir": str(tmp_path / "nope")}}
    with pytest.raises(DataConfigError, match="does not exist"):
        build_loader(cfg, rank=0, world=1, seed=0, device="cpu")


def test_build_loader_dies_on_a_cache_version_bump(tmp_path):
    from loom.data.loader import DataConfigError, build_loader

    _make_body(tmp_path / "c", "libero_franka", n_traj=1, n_src=120)
    man = tmp_path / "c" / CA.MANIFEST_NAME
    doc = json.loads(man.read_text())
    doc["format_version"] = CA.CACHE_FORMAT_VERSION + 1
    man.write_text(json.dumps(doc))

    cfg = {"run": {"seed": 0},
           "data": {"source": "libero", "embodiments": ["libero_franka"],
                    "cache_dir": str(tmp_path / "c")}}
    with pytest.raises(DataConfigError, match="format version"):
        build_loader(cfg, rank=0, world=1, seed=0, device="cpu")


def test_build_loader_rejects_an_unwired_body(tmp_path):
    from loom.data.loader import DataConfigError, build_loader

    _make_body(tmp_path / "c", "libero_franka", n_traj=1, n_src=120)
    cfg = {"run": {"seed": 0},
           "data": {"source": "robotwin", "embodiments": ["synth_a"],
                    "cache_dir": str(tmp_path / "c")}}
    with pytest.raises(DataConfigError, match="no adapter wired"):
        build_loader(cfg, rank=0, world=1, seed=0, device="cpu")


def test_real_dataset_factory_passes_recurrent_burn_in(tmp_path, monkeypatch):
    from loom.data.adapters import libero as libero_adapter
    from loom.data.loader import _dataset_for

    base = _make_body(tmp_path / "c", "libero_franka", n_traj=2)
    trajs = list(base._traj.values())
    monkeypatch.setattr(
        libero_adapter, "libero_trajectories", lambda **kwargs: trajs,
    )
    burned = _dataset_for(
        "libero_franka", "libero", base.cache,
        {"recurrent_burn_in": 4, "action_free": False},
    )
    assert burned.recurrent_burn_in == 4
    assert len(burned) == len(base) - 4 * len(trajs)


def test_demo49_whole_trajectory_split_is_exact_complete_and_order_stable():
    from loom.data.loader import _select_libero_trajectories

    # Match the released LIBERO cardinality without reading 2,000 HDF5 action
    # arrays: four suites x ten tasks x fifty whole demonstrations.
    trajs = [
        CN.CanonicalTrajectory(
            traj_id=f"libero_{suite}/task_{task:02d}/demo_{demo}",
            embodiment="libero_franka",
            src_fps=20.0,
            obs_src_index=np.arange(C.H_PLAN + 1, dtype=np.int64),
            actions=None,
        )
        for suite in range(4)
        for task in range(10)
        for demo in range(50)
    ]

    train, train_manifest = _select_libero_trajectories(
        list(reversed(trajs)), split="train", holdout_demo_keys=("demo_49",),
    )
    gate, gate_manifest = _select_libero_trajectories(
        trajs, split="gate", holdout_demo_keys=("demo_49",),
    )
    _, gate_manifest_reordered = _select_libero_trajectories(
        list(reversed(trajs)), split="gate", holdout_demo_keys=("demo_49",),
    )

    train_ids = set(train_manifest["trajectory_ids"])
    gate_ids = set(gate_manifest["trajectory_ids"])
    assert len(train) == train_manifest["n_trajectories"] == 1960
    assert len(gate) == gate_manifest["n_trajectories"] == 40
    assert train_manifest["n_tasks"] == gate_manifest["n_tasks"] == 40
    assert train_ids.isdisjoint(gate_ids)
    assert train_ids | gate_ids == {traj.traj_id for traj in trajs}
    assert all(len(ids) == 49 for ids in train_manifest["tasks"].values())
    assert all(len(ids) == 1 for ids in gate_manifest["tasks"].values())
    assert all(traj_id.endswith("/demo_49") for traj_id in gate_ids)
    assert gate_manifest_reordered == gate_manifest
    assert gate_manifest["digest"].startswith("sha256:")


def test_whole_trajectory_split_fails_if_any_task_lacks_the_holdout():
    from loom.data.loader import DataConfigError, _select_libero_trajectories

    # Keep this fixture metadata-only: the failure must happen before windows or
    # cached tensors can influence selection.
    trajs = [
        CN.CanonicalTrajectory(
            traj_id=f"libero_goal/task_{task}/demo_{demo}",
            embodiment="libero_franka",
            src_fps=20.0,
            obs_src_index=np.arange(C.H_PLAN + 1, dtype=np.int64),
            actions=None,
        )
        for task in range(2)
        for demo in (0, 49)
        if not (task == 1 and demo == 49)
    ]
    with pytest.raises(DataConfigError, match="absent from task"):
        _select_libero_trajectories(
            trajs, split="gate", holdout_demo_keys=("demo_49",),
        )


def test_gate_loader_manifest_and_clusters_are_rank_resume_stable(
    tmp_path, monkeypatch,
):
    from loom.data.adapters import libero as libero_adapter
    from loom.data.loader import build_gate_loader, build_loader

    trajs, cache = _make_libero_holdout_source(tmp_path / "c", n_tasks=4)
    monkeypatch.setattr(
        libero_adapter, "libero_trajectories",
        lambda **kwargs: list(reversed(trajs)),
    )
    cfg = {
        "run": {"seed": 17},
        "data": {
            "source": "libero",
            "embodiments": ["libero_franka"],
            "batch_per_gpu": 2,
            "sampling": "uniform_task",
            "num_workers": 0,
            "pin_memory": False,
            "trajectory_split": "train",
            "holdout_demo_keys": ["demo_49"],
        },
    }

    rank0 = build_loader(cfg, rank=0, world=2, cache_root=cache.root)
    rank1 = build_loader(cfg, rank=1, world=2, cache_root=cache.root)
    gate = build_gate_loader(cfg, rank=0, world=1, cache_root=cache.root)
    train_manifest = rank0.trajectory_manifest()
    gate_manifest = gate.trajectory_manifest()

    assert rank1.trajectory_manifest() == train_manifest
    assert train_manifest["n_tasks"] == gate_manifest["n_tasks"] == 4
    assert train_manifest["n_trajectories"] == gate_manifest["n_trajectories"] == 4
    assert set(train_manifest["trajectory_ids"]).isdisjoint(
        gate_manifest["trajectory_ids"]
    )
    assert all(len(ids) == 1 for ids in train_manifest["tasks"].values())
    assert all(len(ids) == 1 for ids in gate_manifest["tasks"].values())

    at_resume = rank0.next(137)["data_meta"]
    other_rank = rank1.next(137)["data_meta"]
    resumed = build_loader(cfg, rank=0, world=2, cache_root=cache.root)
    resumed.load_state_dict({"global_step": 137})
    after_resume = resumed.next(resumed.global_step)["data_meta"]
    assert after_resume == at_resume
    for metadata in (at_resume, other_rank):
        assert metadata["source"] == "libero"
        assert metadata["split"] == "train"
        assert metadata["manifest_digest"] == train_manifest["digest"]
        assert metadata["trajectory_cluster_ids"] == metadata["trajectory_ids"]
        assert set(metadata["trajectory_ids"]) <= set(train_manifest["trajectory_ids"])

    gate_metadata = gate.next(0)["data_meta"]
    assert gate_metadata["split"] == "gate"
    assert gate_metadata["manifest_digest"] == gate_manifest["digest"]
    assert set(gate_metadata["task_ids"]) <= set(gate_manifest["tasks"])
    assert set(gate_metadata["trajectory_ids"]) <= set(gate_manifest["trajectory_ids"])


def test_bank_ca_config_trains_on_demo49_complement():
    from loom.train.loop import read_config

    cfg = read_config(Path(__file__).parents[1] / "configs" / "r0a_bank_ca.yaml")
    assert cfg["data"]["trajectory_split"] == "train"
    assert cfg["data"]["holdout_demo_keys"] == ["demo_49"]


# ═══════════════════════════════════════════════════════════════════════════
#  LIBERO ADAPTER  —  skipped where the demo files are not mounted
# ═══════════════════════════════════════════════════════════════════════════

from loom.data.adapters import libero as LB           # noqa: E402  (after fixtures)

_HAS_LIBERO = LB.DATA_ROOT.is_dir() and any(LB.DATA_ROOT.glob("*/*.hdf5"))
needs_libero = pytest.mark.skipif(_HAS_LIBERO is False, reason=f"no demos at {LB.DATA_ROOT}")


def test_libero_registers_delta_semantics_not_absolute():
    """OSC_POSE is a delta controller. This is the 20 -> 30 Hz rescaling trap."""
    assert CN.action_semantics(LB.EMBODIMENT) == (CN.DELTA,) * 6 + (CN.HOLD,)
    assert LB.SRC_FPS == C.EMBODIMENTS[LB.EMBODIMENT].env_fps == 20.0
    assert LB.WINDOW_STRIDE == C.H_OP


def test_libero_orientation_helpers_agree():
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 255, size=(4, 6, 5, 3), dtype=np.uint8)
    oriented = LB.orient_dataset_image(raw)
    assert oriented.shape == raw.shape
    np.testing.assert_array_equal(LB.orient_dataset_image(oriented), raw)   # involution
    # an env still in opengl convention must get the same flip as the dataset
    np.testing.assert_array_equal(LB.orient_env_image(raw, "opengl"), oriented)
    np.testing.assert_array_equal(LB.orient_env_image(raw, "opencv"), raw)
    assert LB.best_matching_transform(oriented, raw)[0] == "vflip"


@needs_libero
def test_libero_discovery_and_instructions():
    demos = LB.discover(suites=("libero_goal",), max_demos=2)
    assert len(demos) == 20, "10 tasks x 2 demos"
    d = demos[0]
    assert d.instruction and d.instruction == d.instruction.strip()
    assert d.n_frames > 0
    assert d.traj_id.startswith("libero_goal/")


@needs_libero
def test_libero_actions_and_proprio_match_the_frozen_spec():
    d = LB.discover(suites=("libero_goal",), max_demos=1)[0]
    a = LB.read_actions(d)
    spec = C.EMBODIMENTS[LB.EMBODIMENT]
    assert a.shape == (d.n_frames, spec.dof)
    assert a.min() >= -1.0 - 1e-6 and a.max() <= 1.0 + 1e-6
    assert set(np.unique(a[:, 6]).tolist()) <= {-1.0, 1.0}, "gripper is latched, hence HOLD"
    p = LB.read_proprio(d, [0, 1, 2])
    assert p.shape == (3, spec.dof)


@needs_libero
def test_libero_trajectory_lands_on_the_canonical_clock():
    d = LB.discover(suites=("libero_goal",), max_demos=1)[0]
    traj = LB.libero_trajectories(demos=[d])[0]
    assert traj.src_fps == 20.0
    assert traj.n_actions == CN.canonical_action_count(d.n_frames, 20.0)
    assert traj.n_actions == int(d.n_frames * 1.5)
    raw = LB.read_actions(d)
    # the whole point of the delta path: same commanded motion, more steps.
    # A 30 Hz grid can fall up to half a 20 Hz step short of the record's end,
    # so the only motion that may be missing is that last partial step's.
    missing = np.abs(raw[:, :6].sum(0) - traj.actions[:, :6].sum(0))
    assert missing.max() <= np.abs(raw[-1, :6]).max() + 1e-3, missing
    if d.n_frames % 2 == 0:                       # grids coincide exactly
        np.testing.assert_allclose(traj.actions[:, :6].sum(0), raw[:, :6].sum(0), atol=1e-3)
    ws = CN.segment(traj, stride=LB.WINDOW_STRIDE)
    assert ws, f"{d.n_frames}-frame demo should yield windows"
    assert all(w.act_hi - w.act_lo == C.H_PLAN for w in ws)


@needs_libero
def test_libero_images_are_reoriented_once():
    import h5py
    d = LB.discover(suites=("libero_goal",), max_demos=1)[0]
    imgs = LB.read_images(d, [0, 5])
    assert imgs.shape[:2] == (2, len(LB.VIEW_KEYS))
    assert imgs.dtype == np.uint8
    with h5py.File(d.path, "r") as f:
        raw = np.asarray(f["data"][d.demo_key]["obs"][LB.VIEW_KEYS[0]][[0, 5]])
    np.testing.assert_array_equal(imgs[:, 0], raw[:, ::-1, :, :])


# ═══════════════════════════════════════════════════════════════════════════
#  ROBOTWIN 2.0 ADAPTER  —  R0-B (Team H).
#
#  These replace the Phase-1B "deliberately unimplemented" guard. Everything
#  here runs on CPU with no demo files; the three that need the 33 GB of HDF5s
#  are marked `needs_robotwin`.
# ═══════════════════════════════════════════════════════════════════════════

from loom.data.adapters import robotwin as RT           # noqa: E402

_HAS_ROBOTWIN = RT.DATA_ROOT.is_dir() and any(
    (RT.DATA_ROOT / RT.SUITES[0]).glob(f"*/{RT.ROBOT_DIR}/data/episode_*.hdf5")
)
needs_robotwin = pytest.mark.skipif(
    _HAS_ROBOTWIN is False, reason=f"no RoboTwin demos at {RT.DATA_ROOT}"
)


def test_robotwin_src_fps_is_250_over_15_not_15():
    """`/additional_info/frequency = 15` is save_freq, a 250 Hz decimation
    factor, not hertz. Taking it as hertz makes every executed segment 11 %
    too slow and the wrong number is the tidy-looking one."""
    spec = C.EMBODIMENTS[RT.EMBODIMENT]
    assert RT.SRC_FPS == spec.env_fps == pytest.approx(250.0 / 15.0)
    assert C.env_steps_per_segment(spec.env_fps) == pytest.approx(4.44444, abs=1e-4)
    assert C.env_steps_per_segment(15.0) == 4.0          # the trap, for contrast


def test_robotwin_registers_absolute_semantics_for_all_14_channels():
    """Joint-position control: `action[t] == state[t+1]` bitwise in the released
    data. The grippers are a normalised width ramped linearly over 200 physics
    steps, so they are ABSOLUTE too — not LIBERO's latched HOLD."""
    spec = C.EMBODIMENTS[RT.EMBODIMENT]
    assert spec.dof == RT.DOF == 14
    assert spec.n_views == len(RT.VIEW_KEYS) == len(RT.EVAL_VIEW_KEYS) == 4
    kinds = CN.action_semantics(RT.EMBODIMENT)
    assert kinds == (CN.ABSOLUTE,) * 14
    assert CN.HOLD not in kinds and CN.DELTA not in kinds
    # gripper channels are the two normalised ones; the rest are radians
    assert spec.action_low[6] == spec.action_low[13] == 0.0
    assert spec.action_high[6] == spec.action_high[13] == 1.0
    assert RT.WINDOW_STRIDE == C.H_OP


def test_robotwin_absolute_resampling_is_a_plain_interpolation():
    """A straight-line joint path upsampled 16.67 -> 30 Hz stays a straight line.
    Under DELTA it would be integrated and differenced; under HOLD it would
    staircase. Both are silent in the training loss."""
    t = np.arange(40, dtype=np.float32)
    raw = np.stack([t * 0.01] * 14, axis=1)              # (40, 14) linear ramp
    got = CN.resample_actions(raw, RT.SRC_FPS, C.FPS_CANONICAL,
                              CN.action_semantics(RT.EMBODIMENT))
    n = CN.canonical_action_count(40, RT.SRC_FPS)
    assert got.shape == (n, 14)
    # ...and it never extrapolates: destination samples past the last source
    # instant latch the final value rather than continuing the ramp.
    want = np.minimum(np.arange(n) * (RT.SRC_FPS / C.FPS_CANONICAL), 39.0) * 0.01
    np.testing.assert_allclose(got[:, 0], want, atol=1e-5)


@needs_robotwin
def test_robotwin_discovery_and_per_episode_instructions():
    eps = RT.discover(tasks_=("handover_block",), max_episodes=3)
    assert len(eps) == 3
    assert [e.traj_id for e in eps] == [
        f"demo_clean/handover_block/episode_{i:07d}" for i in range(3)
    ]
    assert all(e.n_frames > C.H_PLAN for e in eps)
    texts = [RT.read_instruction(e) for e in eps]
    assert all(t and t == t.strip() for t in texts)
    assert len(set(texts)) > 1, "RoboTwin draws a different phrasing per episode"


@needs_robotwin
def test_robotwin_actions_are_the_next_state_exactly():
    """The single strongest evidence that the stream is absolute, not delta."""
    ep = RT.discover(tasks_=("turn_switch",), max_episodes=1)[0]
    a = RT.read_actions(ep)
    s = RT.read_proprio(ep)
    assert a.shape == s.shape == (ep.n_frames, 14)
    np.testing.assert_array_equal(a[:-1], s[1:])          # bitwise, not approx
    assert not np.array_equal(a[:-1], s[:-1])
    g = a[:, [6, 13]]
    assert g.min() >= 0.0 and g.max() <= 1.0
    # planner ramps the gripper as linspace(now, target, 200) advanced one
    # element per physics step, so a saved-frame delta of 15/199 IS save_freq.
    assert np.abs(np.diff(g, axis=0)).max() <= 15.0 / 199.0 + 1e-6


@needs_robotwin
def test_robotwin_frames_decode_to_true_rgb_and_are_not_black():
    """The stored JPEGs are BGR-as-RGB (RoboTwin encodes with cv2.imencode on an
    RGB array and says so). `decode_frame` swaps; PIL alone does not."""
    import io

    import h5py
    from PIL import Image

    ep = RT.discover(tasks_=("handover_block",), max_episodes=1)[0]
    imgs = RT.read_images(ep, [0, 4, 9])
    assert imgs.shape == (3, 4, 240, 320, 3)
    assert imgs.dtype == np.uint8
    for v in range(4):
        assert imgs[:, v].var() > 5.0, f"{RT.VIEW_KEYS[v]} is (near) all black"

    with h5py.File(ep.path, "r") as f:
        raw = f[f"vision/{RT.VIEW_KEYS[0]}/colors"][0].tobytes()
    pil = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
    np.testing.assert_array_equal(imgs[0, 0], pil[..., ::-1])
    assert not np.array_equal(imgs[0, 0], pil), "the channel swap is the point"
    # the eval-side twin is the identity: SAPIEN already hands out true RGB
    np.testing.assert_array_equal(RT.orient_env_image(imgs[0, 0]), imgs[0, 0])


@needs_robotwin
def test_robotwin_cache_round_trips_through_the_loader(tmp_path):
    """Shard -> merge -> FeatureCache -> CachedWindowDataset -> collate, with a
    fake encoder. This is the whole R0-B data path minus the frozen tower."""
    p, f_, l = 8, 6, 4

    def enc(images, instruction):
        n, v = images.shape[0], images.shape[1]
        x = images.reshape(n, v, -1)[..., : p * f_].astype(np.float32) / 255.0
        return x.reshape(n, v, p, f_), np.full((l, f_), float(len(instruction)), np.float32)

    roots = []
    for i in range(2):
        r = tmp_path / "_shards" / f"rank{i}"
        RT.encode_to_cache(enc, r, tasks_=("turn_switch",), max_episodes=2,
                           shard=(i, 2), flush_every=1)
        roots.append(r)
    # a re-run is a no-op: resume must not re-encode or duplicate
    RT.encode_to_cache(enc, roots[0], tasks_=("turn_switch",), max_episodes=2,
                       shard=(0, 2), flush_every=1)
    info = RT.merge_shards(roots, tmp_path / "cache")
    assert info["entries"] == 2
    assert info["spec"]["n_views"] == 4 and info["spec"]["dof"] == 14
    assert info["spec"]["codec"] == CA.DEFAULT_CODEC

    ds = RT.robotwin_dataset(tmp_path / "cache", tasks_=("turn_switch",), max_episodes=2)
    assert len(ds) > 0
    w = collate_window([ds[i] for i in range(min(3, len(ds)))])
    assert w["embodiment"] == RT.EMBODIMENT
    assert w["src_fps"] == pytest.approx(250.0 / 15.0)
    assert len(w["feats"]) == C.N_STATES
    assert w["actions"].shape[1:] == (C.DEPTH, C.H_OP, 14)
    C.assert_action_segment(w["actions"][:, 0], RT.EMBODIMENT)


def test_throughput_ratio_arithmetic():
    from loom.data.loader import Throughput
    tp = Throughput(n_batches=10, seconds=1.0, batch_size=8, bytes_moved=10 * 2 ** 20)
    assert tp.batches_per_s == pytest.approx(10.0)
    assert tp.samples_per_s == pytest.approx(80.0)
    assert tp.ratio(5.0) == pytest.approx(2.0)
    assert tp.sustains(5.0) and not tp.sustains(10.0)
