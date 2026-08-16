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
    return CachedWindowDataset(trajs, CA.FeatureCache(root))


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


def test_resume_needs_only_global_step(tmp_path):
    """Team D's checkpoint stores no sampler cursor. It must not have to."""
    ds = _make_body(tmp_path / "a", "libero_franka")
    l1 = LoomLoader({"libero_franka": ds}, batch_size=4, seed=3)
    ref = [b["feats"][0]["views"].clone() for b in l1.batches(0, 6)]
    assert l1.state_dict() == {"global_step": 6}

    l2 = LoomLoader({"libero_franka": ds}, batch_size=4, seed=3)
    l2.load_state_dict({"global_step": 4})
    resumed = [b["feats"][0]["views"] for b in l2.batches(l2.global_step, 2)]
    for i, got in enumerate(resumed):
        torch.testing.assert_close(got, ref[4 + i])


def test_sampler_rejects_a_body_too_small_for_one_batch():
    with pytest.raises(ValueError, match="cannot fill one global batch"):
        HomogeneousSampler({"synth_a": 6}, batch_size=4, world_size=2)


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
#  PHASE 1B GUARD
# ═══════════════════════════════════════════════════════════════════════════

def test_robotwin_is_deliberately_unimplemented():
    from loom.data.adapters import robotwin
    with pytest.raises(NotImplementedError, match="Phase 1B"):
        robotwin.discover()


def test_throughput_ratio_arithmetic():
    from loom.data.loader import Throughput
    tp = Throughput(n_batches=10, seconds=1.0, batch_size=8, bytes_moved=10 * 2 ** 20)
    assert tp.batches_per_s == pytest.approx(10.0)
    assert tp.samples_per_s == pytest.approx(80.0)
    assert tp.ratio(5.0) == pytest.approx(2.0)
    assert tp.sustains(5.0) and not tp.sustains(10.0)
