"""Team I — the frozen vision/text tower.

The load-bearing test in this file is
`test_train_and_eval_preprocessing_are_identical`. Everything else guards a
shape or a dtype; that one guards the score. A resize, an interpolation mode or
a mean/std that differ between `encode_to_cache` and `eval.policy` produce a
model that trains normally and evaluates at chance, with no other symptom
(PLAN §7 names it as one of the two things to check first).

Most tests run on a **tiny randomly-initialised SigLIP** so the whole file is a
few seconds on the login node's 4 CPUs. Frozenness, shapes, parity, dtype and
determinism are all properties of the wiring, not of the weights. Tests that
genuinely need the 878 M checkpoint are marked `gpu` or `slow` and skip cleanly
when it is not on disk.
"""

from __future__ import annotations

import inspect
import time

import numpy as np
import pytest
import torch

import contracts as C
from loom.data import tower as T

RNG = np.random.default_rng(0)


# ═══════════════════════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

def _transformers():
    try:
        import transformers                              # noqa: PLC0415
    except Exception:                                    # noqa: BLE001
        pytest.skip("transformers not installed in this interpreter")
    return transformers


def _tokenizer():
    tf = _transformers()
    hub = T._hub_dir()
    if not T.weights_available():
        pytest.skip(f"{T.TOWER_MODEL_ID} not in {hub}; "
                    f"run `python -m loom.data.tower --download` on a login node")
    return tf.AutoTokenizer.from_pretrained(
        T.TOWER_MODEL_ID, local_files_only=True, cache_dir=str(hub))


@pytest.fixture(scope="module")
def tiny() -> T.FrozenTower:
    """A shape-faithful, weight-free tower: same geometry, 2 layers, F=64.

    Same `image_size` / `patch_size` / `max_position_embeddings` / tokenizer as
    the real checkpoint, so every code path under test is the real one.
    """
    tf = _transformers()
    tok = _tokenizer()
    cfg = tf.SiglipConfig(
        text_config=dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=2, vocab_size=tok.vocab_size,
                         max_position_embeddings=T.LANG_LEN),
        vision_config=dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                           num_attention_heads=2, image_size=T.NATIVE_IMAGE_SIZE,
                           patch_size=T.PATCH_SIZE),
    )
    torch.manual_seed(0)
    return T.FrozenTower(model=tf.SiglipModel(cfg), tokenizer=tok, model_id="tiny")


@pytest.fixture(scope="module")
def real_cpu() -> T.FrozenTower:
    """The genuine 878 M checkpoint on CPU. Skips when the weights are absent."""
    _transformers()
    try:
        return T.get_tower(device="cpu")
    except T.TowerUnavailable as e:
        pytest.skip(str(e))


def _frames(n: int = 3, v: int = 2, size: int = 128) -> np.ndarray:
    """LIBERO-shaped uint8 frames: (n, V, H, W, 3), demo resolution."""
    return RNG.integers(0, 256, (n, v, size, size, 3), dtype=np.uint8)


# ═══════════════════════════════════════════════════════════════════════════
#  1 · FROZEN
# ═══════════════════════════════════════════════════════════════════════════

def test_no_parameter_requires_grad(tiny):
    bad = [n for n, p in tiny._model.named_parameters() if p.requires_grad]
    assert not bad, f"frozen tower has trainable parameters: {bad[:5]}"


def test_outputs_carry_no_grad_fn(tiny):
    """Even called under `enable_grad`, nothing reaches the training graph."""
    with torch.enable_grad():
        views = tiny.encode_images(_frames(1))
        lang = tiny.encode_text("pick up the black bowl")
    for name, t in (("views", views), ("lang", lang)):
        assert t.grad_fn is None, f"{name} carries a grad_fn into the training graph"
        assert not t.requires_grad, f"{name}.requires_grad is True"


def test_tower_params_are_invisible_to_a_trainable_module(tiny):
    """PLAN §9. `FrozenTower` is not an `nn.Module`, so this holds by
    construction rather than by remembering to filter the optimizer."""
    class Trainable(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.real = torch.nn.Linear(8, 8)
            self.tower = tiny                          # the mistake we must survive

    m = Trainable()
    mine = {id(p) for p in m.parameters()}
    theirs = {id(p) for p in tiny.parameters()}
    assert theirs, "the tower has no parameters at all — fixture is wrong"
    assert not (mine & theirs), "frozen tower parameters leaked into model.parameters()"
    assert not any(k.startswith("tower.") for k in m.state_dict()), \
        "frozen tower leaked into state_dict(); it would be sharded by FSDP"
    assert sum(p.numel() for p in m.parameters()) == 8 * 8 + 8


def test_tower_stays_in_eval_mode(tiny):
    assert not tiny._model.training


# ═══════════════════════════════════════════════════════════════════════════
#  2 · SHAPES AND F
# ═══════════════════════════════════════════════════════════════════════════

def test_encode_images_preserves_leading_dims(tiny):
    """(n, V, H, W, 3) -> (n, V, P, F) for the cache; (V, ...) -> (V, P, F) for eval."""
    assert tuple(tiny.encode_images(_frames(3, 2)).shape) == (3, 2, tiny.n_patches, 64)
    assert tuple(tiny.encode_images(_frames(1, 2)[0]).shape) == (2, tiny.n_patches, 64)


def test_obs_feats_conforms_to_contracts_obsfeats(tiny):
    spec = C.EMBODIMENTS["libero_franka"]
    feats = tiny.obs_feats(
        _frames(1, spec.n_views)[0], torch.zeros(1, spec.dof), "open the drawer"
    )
    assert set(feats) == {"views", "proprio", "lang"}
    b, v, p, f = feats["views"].shape
    assert (b, v, p, f) == (1, spec.n_views, tiny.n_patches, tiny.feat_dim)
    assert tuple(feats["proprio"].shape) == (1, spec.dof)
    bl, ll, fl = feats["lang"].shape
    assert (bl, ll, fl) == (1, T.LANG_LEN, tiny.feat_dim)
    assert f == fl, "ObsFeats.views and ObsFeats.lang must share F"


def test_patch_count_matches_the_declared_grid(tiny):
    assert T.N_PATCHES == (T.IMAGE_SIZE // T.PATCH_SIZE) ** 2 == tiny.n_patches
    assert tiny.encode_images(_frames(1)).shape[-2] == T.N_PATCHES


def test_feat_dim_matches_estimator_and_proposal_defaults():
    """F is *measured* from the checkpoint config, then checked against the two
    constructor defaults that would have to change if it were not 1152."""
    from loom.heads.potential import Potential
    from loom.heads.proposal import Proposal
    from loom.model.estimator import Estimator

    assert inspect.signature(Estimator).parameters["feat_dim"].default == T.FEAT_DIM
    assert inspect.signature(Proposal).parameters["lang_dim"].default == T.FEAT_DIM
    assert inspect.signature(Potential).parameters["lang_dim"].default == T.FEAT_DIM


def test_checkpoint_config_confirms_f_and_the_patch_grid():
    """The one assertion that reads the real checkpoint. Config only — no weights,
    no network, milliseconds."""
    tf = _transformers()
    hub = T._hub_dir()
    if not T.weights_available():
        pytest.skip(f"{T.TOWER_MODEL_ID} not on local disk")
    cfg = tf.AutoConfig.from_pretrained(
        T.TOWER_MODEL_ID, local_files_only=True, cache_dir=str(hub))
    assert int(cfg.vision_config.hidden_size) == T.FEAT_DIM
    assert int(cfg.text_config.hidden_size) == T.FEAT_DIM, \
        "views and lang must share F; ObsFeats has one feature width"
    assert int(cfg.vision_config.patch_size) == T.PATCH_SIZE
    assert int(cfg.vision_config.image_size) == T.NATIVE_IMAGE_SIZE
    assert _tokenizer().model_max_length == T.LANG_LEN


def test_cache_spec_geometry_matches_the_tower():
    """`CacheSpec` is what the loader validates every read against."""
    from loom.data.cache import CacheSpec

    spec = C.EMBODIMENTS["libero_franka"]
    cs = CacheSpec("fp16", spec.n_views, T.N_PATCHES, T.FEAT_DIM, spec.dof, T.LANG_LEN)
    assert cs.bytes_per_frame == spec.n_views * T.N_PATCHES * T.FEAT_DIM * 2 + spec.dof * 4
    assert cs.bytes_per_window == C.N_STATES * cs.bytes_per_frame + T.LANG_LEN * T.FEAT_DIM * 2


# ═══════════════════════════════════════════════════════════════════════════
#  3 · TRAIN / EVAL PREPROCESSING PARITY   ← the load-bearing one
# ═══════════════════════════════════════════════════════════════════════════

def test_train_and_eval_preprocessing_are_identical(tiny):
    """One array, both call paths, bitwise-identical features.

    Train:  `adapters.libero.read_images` -> `encode_to_cache` -> `tower.encode`
    Eval:   `eval.libero.extract_obs`     -> `policy.default_featurizer`

    If these ever diverge — a different resize, a different interpolation, a
    different mean/std, an extra flip — the model trains on one distribution and
    is evaluated on another, and the only symptom is a near-zero score.
    """
    import loom.eval.policy as pol

    spec = C.EMBODIMENTS["libero_franka"]
    instruction = "put the bowl on the plate"
    imgs = _frames(1, spec.n_views)[0]                       # (V, H, W, 3) uint8

    # ── training path: exactly what `encode_to_cache` calls ──────────────
    train_views, train_lang = tiny.encode(imgs[None], instruction)   # (1,V,P,F), (L,F)

    # ── eval path: exactly what `LoomPolicy._plan` calls ─────────────────
    featurize = pol.default_featurizer(spec, tower=tiny)
    obs = {k: imgs[i] for i, k in enumerate(T.EVAL_VIEW_KEYS)}
    obs["state"] = np.arange(spec.dof, dtype=np.float32)
    feats = featurize(obs, instruction)

    eval_views = feats["views"].float().numpy()
    eval_lang = feats["lang"][0].float().numpy()

    assert eval_views.shape == train_views.shape
    np.testing.assert_array_equal(
        train_views, eval_views,
        err_msg="train and eval view features differ — the two preprocessing "
                "paths have drifted apart",
    )
    np.testing.assert_array_equal(train_lang, eval_lang)


def test_the_two_paths_share_one_preprocess_call(tiny):
    """Parity at the tensor level too, so a failure localises to preprocessing
    rather than to the tower."""
    img = _frames(1, 2)[0]
    a = T.preprocess_images(img, image_size=tiny.image_size, dtype=tiny.dtype)
    b = T.preprocess_images(img[None], image_size=tiny.image_size, dtype=tiny.dtype)
    assert torch.equal(a, b)
    assert a.shape == (2, 3, tiny.image_size, tiny.image_size)


def test_eval_view_key_order_matches_the_dataset_v_axis():
    """`views[:, 0]` must mean agentview on both paths. Swapping V is silent."""
    from loom.data.adapters.libero import VIEW_KEYS

    assert len(T.EVAL_VIEW_KEYS) == len(VIEW_KEYS) == 2
    assert VIEW_KEYS == ("agentview_rgb", "eye_in_hand_rgb")
    assert T.EVAL_VIEW_KEYS == ("full_image", "wrist_image")


def test_preprocessing_adds_no_flip_of_its_own():
    """Orientation is Team A's, applied upstream on both paths. A second flip
    here would cancel one of them and only at eval time."""
    img = RNG.integers(0, 256, (1, 32, 32, 3), dtype=np.uint8)
    px = T.preprocess_images(img, image_size=32, dtype=torch.float32)
    want = torch.from_numpy(img).permute(0, 3, 1, 2).float().div(255.0).sub(0.5).div(0.5)
    assert torch.allclose(px, want, atol=0), "preprocess must not reorient or resample at 1:1"


def test_source_resolutions_both_reach_the_same_shape(tiny):
    """128 px demo frames and 256 px live frames must land on one geometry."""
    demo = tiny.encode_images(_frames(1, 2, size=128))
    live = tiny.encode_images(_frames(1, 2, size=256))
    assert demo.shape == live.shape == (1, 2, tiny.n_patches, tiny.feat_dim)


# ═══════════════════════════════════════════════════════════════════════════
#  4 · BF16
# ═══════════════════════════════════════════════════════════════════════════

def test_bf16_in_bf16_out_no_promotion(tiny):
    px = T.preprocess_images(_frames(1), image_size=tiny.image_size, dtype=torch.bfloat16)
    assert px.dtype == torch.bfloat16
    views = tiny.encode_images(_frames(2))
    lang = tiny.encode_text(["a", "b"])
    assert views.dtype is torch.bfloat16, f"views promoted to {views.dtype}"
    assert lang.dtype is torch.bfloat16, f"lang promoted to {lang.dtype}"
    assert all(p.dtype is torch.bfloat16 for p in tiny.parameters())


def test_encoder_hands_the_cache_float32(tiny):
    """`FeatureCacheWriter.write` casts to the codec; it needs plain float in."""
    views, lang = tiny.encode(_frames(2), "open the drawer")
    assert views.dtype == np.float32 and lang.dtype == np.float32
    assert views.shape == (2, 2, tiny.n_patches, tiny.feat_dim)
    assert lang.shape == (T.LANG_LEN, tiny.feat_dim)
    assert np.isfinite(views).all() and np.isfinite(lang).all()


# ═══════════════════════════════════════════════════════════════════════════
#  5 · DETERMINISM
# ═══════════════════════════════════════════════════════════════════════════

def test_same_input_twice_gives_identical_features(tiny):
    imgs = _frames(2)
    a = tiny.encode_images(imgs)
    b = tiny.encode_images(imgs.copy())
    assert torch.equal(a, b), "the frozen tower is not a pure function of its input"
    la = tiny.encode_text("stack the bowls")
    lb = tiny.encode_text("stack the bowls")
    assert torch.equal(la, lb)


def test_batching_does_not_change_features(tiny):
    """A cache built with `chunk=16` and one built with `chunk=1` must agree."""
    imgs = _frames(5)
    big = T.FrozenTower(model=tiny._model, tokenizer=tiny.tokenizer,
                        model_id="tiny", batch_size=64).encode_images(imgs)
    small = T.FrozenTower(model=tiny._model, tokenizer=tiny.tokenizer,
                          model_id="tiny", batch_size=2).encode_images(imgs)
    assert torch.equal(big, small)


def test_get_tower_is_a_singleton(tiny):
    T.reset_tower()
    try:
        T.get_tower(model_id="tiny", model=tiny._model, tokenizer=tiny.tokenizer)
        a = T.get_tower(model_id="tiny")
        b = T.get_tower(model_id="tiny")
        assert a is b, "the 3.3 GiB checkpoint would be loaded twice"
    finally:
        T.reset_tower()


# ═══════════════════════════════════════════════════════════════════════════
#  6 · REAL WEIGHTS
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
def test_real_tower_on_cpu(real_cpu):
    """Skips cleanly when the checkpoint is absent, so the login node can still
    run the whole suite."""
    assert real_cpu.feat_dim == T.FEAT_DIM
    assert real_cpu.n_patches == T.N_PATCHES
    assert not any(p.requires_grad for p in real_cpu.parameters())
    views = real_cpu.encode_images(_frames(1, 1))
    assert views.shape == (1, 1, T.N_PATCHES, T.FEAT_DIM)
    assert views.dtype is torch.bfloat16 and views.grad_fn is None
    assert torch.isfinite(views.float()).all()


@pytest.mark.gpu
@pytest.mark.bench
def test_encode_throughput_on_gpu():
    """windows/s with the real checkpoint, so Team A's >=1.3x loader margin can
    be re-checked against a real encoder rather than a modelled one."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    try:
        tower = T.get_tower(device="cuda")
    except T.TowerUnavailable as e:
        pytest.skip(str(e))

    spec = C.EMBODIMENTS["libero_franka"]
    per_window = C.N_STATES * spec.n_views                   # 10 images
    imgs = _frames(C.N_STATES, spec.n_views)                 # one window of pixels

    for _ in range(3):
        tower.encode_images(imgs)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    iters = 20
    for _ in range(iters):
        out = tower.encode_images(imgs)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters

    assert out.shape == (C.N_STATES, spec.n_views, T.N_PATCHES, T.FEAT_DIM)
    print(
        f"\n[tower] {T.TOWER_MODEL_ID} @ {T.IMAGE_SIZE}px  P={T.N_PATCHES}  "
        f"F={T.FEAT_DIM}  {tower.n_params / 1e6:.0f}M params\n"
        f"[tower] {dt * 1e3:.1f} ms/window ({per_window} images), "
        f"{1.0 / dt:.1f} windows/s, {per_window / dt:.0f} images/s\n"
        f"[tower] full LIBERO cache build: 64189 frames / "
        f"{per_window / C.N_STATES / dt * C.N_STATES:.0f} img/s -> "
        f"{64189 * spec.n_views / (per_window / dt) / 60:.1f} min"
    )
    assert 1.0 / dt > 1.0, "under one window/s on an A100 is a broken forward"


@pytest.mark.gpu
def test_real_tower_shapes_and_frozenness_on_gpu():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    try:
        tower = T.get_tower(device="cuda")
    except T.TowerUnavailable as e:
        pytest.skip(str(e))

    spec = C.EMBODIMENTS["libero_franka"]
    with torch.enable_grad():
        feats = tower.obs_feats(
            _frames(1, spec.n_views)[0],
            torch.zeros(1, spec.dof, device="cuda"),
            "pick up the black bowl and place it on the plate",
        )
    assert feats["views"].shape == (1, spec.n_views, T.N_PATCHES, T.FEAT_DIM)
    assert feats["lang"].shape == (1, T.LANG_LEN, T.FEAT_DIM)
    assert feats["views"].dtype is torch.bfloat16
    assert feats["views"].grad_fn is None and feats["lang"].grad_fn is None
    assert not any(p.requires_grad for p in tower.parameters())


@pytest.mark.gpu
@pytest.mark.parametrize("est_dtype", [torch.bfloat16, torch.float32])
def test_real_tower_feeds_the_estimator(est_dtype):
    """The whole point: `ObsFeats` from the tower goes straight into `E`.

    Both dtypes, because eval has no autocast: the tower emits bf16 and proprio
    arrives float32 (exactly as the fp16 cache hands them to training), and a
    float32 checkpoint would otherwise die on `mat1 and mat2 must have the same
    dtype`. `policy.feats_to` is the one place that reconciles them.
    """
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    try:
        tower = T.get_tower(device="cuda")
    except T.TowerUnavailable as e:
        pytest.skip(str(e))
    from loom.eval.policy import feats_to
    from loom.model.estimator import Estimator

    spec = C.EMBODIMENTS["libero_franka"]
    est = Estimator(feat_dim=T.FEAT_DIM).to("cuda", est_dtype).eval()
    feats = tower.obs_feats(
        _frames(1, spec.n_views)[0],
        torch.zeros(1, spec.dof, device="cuda"),        # float32, like the cache
        "pick up the black bowl",
    )
    assert feats["views"].dtype is torch.bfloat16
    assert feats["proprio"].dtype is torch.float32
    with torch.no_grad():
        z = est(feats_to(feats, "cuda", est_dtype), None, embodiment=spec.name)
    C.assert_belief(z)
    assert z.shape == (1, C.K, C.D) and z.dtype is est_dtype
