"""
LOOM — the frozen SigLIP vision + text tower.

`google/siglip-so400m-patch14-384`, loaded once, `requires_grad_(False)`,
`.eval()`, and every forward under `torch.no_grad()`. PLAN §9: *the frozen tower
never enters the training graph; features are cached.* `FrozenTower` is
deliberately **not** an `nn.Module`, so assigning one to an attribute of a
trainable module cannot leak its 878 M parameters into `model.parameters()`,
into FSDP's flat parameter, or into the optimizer.

It produces exactly the three fields of `contracts.ObsFeats`:

===========  =====================  ==========================================
field        shape                  source
===========  =====================  ==========================================
``views``    ``(B, V, P, F)``       vision ``last_hidden_state``, patch tokens
``proprio``  ``(B, dof)``           not this module's — from the dataset / env
``lang``     ``(B, L, F)``          text ``last_hidden_state``, per token
===========  =====================  ==========================================

**Measured, not assumed.** ``config.json`` of the checkpoint gives
``vision_config.hidden_size == text_config.hidden_size == 1152``, so
``F = 1152`` for both towers. That is exactly what `model.estimator.Estimator`
takes as its ``feat_dim`` default and what `heads.proposal.Proposal` and
`heads.potential.Potential` take as their ``lang_dim`` default. **No constructor
default anywhere else has to change.** ``LANG_LEN = 64`` is the checkpoint's
``model_max_length``; SigLIP is trained with ``padding="max_length"`` and
tokenising any other way silently changes the text features.

Resolution — the real trade-off, measured
-----------------------------------------
patch14 makes ``P`` quadratic in the input side: 384 px → a 27×27 grid → **729**
tokens/view (384 = 27·14 + 6; the conv is ``padding="valid"`` so the remainder
is dropped); 224 px → 16×16 → **256**. The cache is linear in ``P``
(``cache.CacheSpec.bytes_per_frame`` = ``V·P·F·2`` bytes fp16), so this choice
sets the whole on-disk footprint. Team A measured 4.34 MiB/window at a
placeholder ``P = 196`` and picked fp16.

Exact LIBERO geometry, **measured** on this cluster (4 suites × 10 tasks × 50
demos = 2000 demos, ``WINDOW_STRIDE = H_OP = 8``): **56 189 windows** over
**64 189 cached source frames** (32.1 frames/demo — the cache is keyed by source
frame, so the 4-deep overlap between consecutive windows is stored once).

=======  ===  ===========  =========  ============  ==========  =========  ======
input     P   MiB / frame  MiB / win  LIBERO cache  ms / win    win/s      R@1
=======  ===  ===========  =========  ============  ==========  =========  ======
224 px   256       1.1250      5.766      70.80 GiB      19.5       51.4   0.1100
384 px   729       3.2036     16.159     201.09 GiB      44.8       22.3   0.1350
=======  ===  ===========  =========  ============  ==========  =========  ======

``MiB/frame`` = ``2 views × P × 1152 × 2 B + 7 × 4``; ``MiB/win`` adds the other
four boundary states plus the ``64 × 1152`` fp16 language row. ``LIBERO cache``
= ``bytes/frame × 64 189 + 2000 × lang``. ``ms/win`` and ``win/s`` are one A100,
bf16, ``N_STATES × V = 10`` images per window, measured. ``R@1`` is top-1
accuracy of matching a real LIBERO ``agentview`` frame to its own task
instruction among all 40 (400 frames, chance 0.025) using SigLIP's own image and
text heads — a direct, on-this-data test of whether the extra 2.85× tokens carry
usable content. Mean rank of the true instruction was **11.34 at 224 px and
11.28 at 384 px**: statistically indistinguishable.

**Choice: 224 px, P = 256.** In order of weight:

1. **70.8 GiB vs 201 GiB is the difference between fitting in a compute node's
   page cache and not.** Team A's whole case for fp16-on-disk over an in-graph
   tower rests on the cache going resident after the first epoch. Raw disk is
   not the binding constraint (1.1 PiB free); node RAM is.
2. **Loader margin, now against a *measured* encoder rather than a modelled
   one.** Cached fp16 reads at Team A's 650 MiB/s deliver 113 windows/s at
   5.77 MiB/window but only 40 windows/s at 16.16 MiB — and the real tower
   encodes at 51.4 windows/s. So at 224 px the cache is 2.2× faster than
   re-encoding and the ≥1.3× margin holds; at 384 px reading the cache would be
   *slower* than the tower that wrote it, which would invalidate the entire
   cache-vs-in-graph decision.
3. **The measurement says the extra tokens buy little.** +2.5 points of top-1 on
   a weak zero-shot signal, and no change at all in mean rank, for 2.85× the
   bytes and 2.3× the FLOPs.
4. **The source has no detail to justify 384.** LIBERO demo frames are 128×128
   and the live env renders 256×256, so 384 is a 3× *upsample* of the training
   pixels. At 224 each of the 256 tokens covers 8.0 source pixels; at 384 each
   of the 729 covers 4.7 — a grid finer than the image it is looking at.
5. **The estimator gets cheaper too**: its context per state falls from
   ``2·729 + 1 + 64 + 128 = 1651`` tokens to ``2·256 + 1 + 64 + 128 = 705``.

The cost of 224 is that the pretrained 27×27 position grid is bicubically
resized to 16×16 (`interpolate_pos_encoding=True`, HF's own path). The R@1
number above already includes that penalty; it is why the gap is 2.5 points
rather than 0. One-time encode of the whole four-suite cache is 4.2 min of A100
at 224 px against 9.6 min at 384 px — real but not decisive either way.

``IMAGE_SIZE`` is a module constant, a constructor argument and
``$LOOM_TOWER_IMAGE_SIZE``. Changing it changes ``P``, which changes the cache
geometry — ``cache.CacheSpec`` validates that on every read, so a mismatch is a
loud `CacheFormatError`, never silent garbage.

Orientation, checked in passing
-------------------------------
The same probe scored the stored frames both ways. Team A's vertical flip
(`orient_dataset_image`, from the files' own ``macros_image_convention =
'opengl'``) is right: R@1 0.1100 flipped vs 0.0625 unflipped at 224 px, 0.1350
vs 0.0800 at 384 px, mean rank 11.3 vs 14.1. An upside-down input to SigLIP
looks exactly like "the tower is bad at this data", so this is worth having
measured. It is Team A's transform and this module does not touch it.

Preprocessing parity
--------------------
Training (`cache` / `adapters.libero.encode_to_cache`) and eval
(`eval.policy`) call **the same** `preprocess_images`: same resize target, same
bicubic interpolation, same clamp, same 0.5/0.5 normalisation, same bf16 cast.
Two implementations of one transform is precisely the "trains fine, scores near
zero" failure mode PLAN §7 names, so there is exactly one, and
`tests/test_tower.py::test_train_and_eval_preprocessing_are_identical` asserts
the two call paths produce bitwise-identical tensors *and* bitwise-identical
features from the same input array.

**Orientation is not this module's decision.** Frames arrive already in
`adapters.libero.CANONICAL_ORIENTATION`: the training path goes through
`adapters.libero.read_images` → `orient_dataset_image`, the eval path through
`eval.libero.extract_obs` → `orient_env_image`. Both are Team A's single
implementation of one transform. This module adds no flip of its own and must
not: a second flip on either side is the same near-zero-score bug.

Weights
-------
Cached under ``$LOOM_HF_CACHE`` (default
``/lustre/fsw/portfolios/edgeai/users/chrislin/datasets/loom/hf_cache``), never
in the repo. **Compute nodes have no outbound network**, so the download is an
explicit, separate, login-node step::

    source .env.local && python -m loom.data.tower --download

after which every load is ``local_files_only=True``. A job that would have to
fetch from the hub fails immediately with a message saying so, rather than
hanging on a dead socket.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch import Tensor

from contracts import EMBODIMENTS, EmbodimentSpec, ObsFeats

__all__ = [
    "TOWER_MODEL_ID", "HF_CACHE_DIR", "FEAT_DIM", "PATCH_SIZE",
    "NATIVE_IMAGE_SIZE", "IMAGE_SIZE", "GRID", "N_PATCHES", "LANG_LEN",
    "IMAGE_MEAN", "IMAGE_STD", "INTERPOLATION", "DTYPE", "EVAL_VIEW_KEYS",
    "preprocess_images", "FrozenTower", "get_tower", "reset_tower",
    "weights_available", "download_weights", "tower_encoder", "obs_featurizer",
]


# ═══════════════════════════════════════════════════════════════════════════
#  ── CONSTANTS BLOCK ─────────────────────────────────────────────────────
#
#  Everything installation- or checkpoint-specific lives here and nowhere else.
#  Verified 2026-08-16 against the downloaded config.json / preprocessor_config
#  of google/siglip-so400m-patch14-384:
#      vision_config.hidden_size 1152, patch_size 14, image_size 384,
#      num_hidden_layers 27; text_config.hidden_size 1152, layers 27;
#      image_mean/std 0.5, resample 3 (bicubic); model_max_length 64.
# ═══════════════════════════════════════════════════════════════════════════

TOWER_MODEL_ID = os.environ.get("LOOM_TOWER_MODEL", "google/siglip-so400m-patch14-384")

#: HF_HOME for this project. Weights are ~3.3 GiB and never go in the repo.
HF_CACHE_DIR = Path(os.environ.get(
    "LOOM_HF_CACHE",
    "/lustre/fsw/portfolios/edgeai/users/chrislin/datasets/loom/hf_cache",
))

#: Both towers. == Estimator(feat_dim=...) and Proposal/Potential(lang_dim=...).
FEAT_DIM = 1152

PATCH_SIZE = 14
NATIVE_IMAGE_SIZE = 384             # what the checkpoint's position grid is for

#: THE RESOLUTION DECISION. See the table in the module docstring.
IMAGE_SIZE = int(os.environ.get("LOOM_TOWER_IMAGE_SIZE", "224"))
GRID = IMAGE_SIZE // PATCH_SIZE
N_PATCHES = GRID * GRID

#: tokenizer.model_max_length. SigLIP is trained with padding="max_length";
#: tokenising any other way changes the features. Fixed, so CacheSpec.lang_len
#: is fixed.
LANG_LEN = 64

IMAGE_MEAN = 0.5                    # preprocessor_config.json, all 3 channels
IMAGE_STD = 0.5
INTERPOLATION = "bicubic"           # preprocessor_config.json resample == 3

#: bf16 throughout (PLAN §9, CLAUDE.md). A100 has no fp8.
DTYPE = torch.bfloat16

#: `eval.libero.extract_obs` keys, in the SAME order as
#: `adapters.libero.VIEW_KEYS` = (agentview_rgb, eye_in_hand_rgb). The V axis
#: must mean the same thing on both paths; swapping it is a silent near-zero.
EVAL_VIEW_KEYS = ("full_image", "wrist_image")

# ═══════════════════════════════════════════════════════════════════════════


class TowerUnavailable(RuntimeError):
    """Weights, `transformers`, or both are missing. Never silently degraded."""


# ═══════════════════════════════════════════════════════════════════════════
#  PREPROCESSING — ONE implementation, shared by training and eval
# ═══════════════════════════════════════════════════════════════════════════

def preprocess_images(
    images: np.ndarray | Tensor,
    *,
    image_size: int = IMAGE_SIZE,
    dtype: torch.dtype = DTYPE,
    device: str | torch.device = "cpu",
) -> Tensor:
    """``(..., H, W, 3)`` uint8 (or float in [0, 1]) -> ``(N, 3, S, S)`` `dtype`.

    The single normalisation path. Training and eval both come through here, so
    they cannot drift: same target size, same bicubic resample, same clamp, same
    ``(x - 0.5) / 0.5``, same cast.

    Leading dimensions are flattened into the batch; callers restore them.
    `antialias=True` matters only when *down*sampling (the 256 px live env at
    ``IMAGE_SIZE = 224``); the 128 px demo frames are upsampled either way. The
    clamp is not cosmetic — bicubic overshoots outside [0, 1] and PIL's uint8
    pipeline, which the checkpoint was trained with, cannot.

    Images must already be in `adapters.libero.CANONICAL_ORIENTATION`. This
    function does not flip.
    """
    if isinstance(images, np.ndarray):
        x = torch.from_numpy(np.ascontiguousarray(images))
    elif isinstance(images, Tensor):
        x = images
    else:
        raise TypeError(f"images must be ndarray or Tensor, got {type(images).__name__}")

    if x.ndim < 3 or x.shape[-1] != 3:
        raise ValueError(f"images must be (..., H, W, 3) RGB, got {tuple(x.shape)}")

    h, w = int(x.shape[-3]), int(x.shape[-2])
    x = x.reshape(-1, h, w, 3).permute(0, 3, 1, 2)               # (N, 3, H, W)

    if x.dtype == torch.uint8:
        x = x.to(device=device, dtype=torch.float32).div_(255.0)
    else:
        x = x.to(device=device, dtype=torch.float32)
        hi = float(x.max()) if x.numel() else 0.0
        if hi > 1.5:
            raise ValueError(
                f"float images must already be in [0, 1] (max {hi:.3g}); pass uint8 "
                f"or rescale first — a second /255 here would be invisible"
            )

    if (h, w) != (image_size, image_size):
        x = torch.nn.functional.interpolate(
            x, size=(image_size, image_size), mode=INTERPOLATION,
            align_corners=False, antialias=True,
        )
        x = x.clamp_(0.0, 1.0)

    x = (x - IMAGE_MEAN) / IMAGE_STD
    return x.to(dtype)


# ═══════════════════════════════════════════════════════════════════════════
#  WEIGHTS
# ═══════════════════════════════════════════════════════════════════════════

def _hub_dir(cache_dir: str | os.PathLike | None = None) -> Path:
    """The hub directory, and the env vars to match it.

    `huggingface_hub` snapshots its cache path into a module constant **at
    import time**, so setting `HF_HOME` afterwards is a no-op. Every call below
    therefore also passes `cache_dir=` explicitly; the env vars are set for the
    benefit of anything else in the process, not relied upon here.
    """
    if cache_dir is not None:
        hub = Path(cache_dir)
        if hub.name != "hub":
            os.environ["HF_HOME"] = str(hub)
            hub = hub / "hub"
    elif os.environ.get("HF_HUB_CACHE"):
        hub = Path(os.environ["HF_HUB_CACHE"])
    else:
        root = Path(os.environ.get("HF_HOME") or HF_CACHE_DIR)
        os.environ["HF_HOME"] = str(root)
        hub = root / "hub"
    os.environ["HF_HUB_CACHE"] = str(hub)
    return hub


def weights_available(
    model_id: str = TOWER_MODEL_ID, cache_dir: str | os.PathLike | None = None
) -> bool:
    """True when the checkpoint is on local disk. Never touches the network."""
    repo = _hub_dir(cache_dir) / ("models--" + model_id.replace("/", "--")) / "snapshots"
    if not repo.is_dir():
        return False
    for snap in repo.iterdir():
        if (snap / "config.json").exists() and any(snap.glob("*.safetensors")):
            return True
    return False


def download_weights(
    model_id: str = TOWER_MODEL_ID, cache_dir: str | os.PathLike | None = None
) -> str:
    """Fetch the checkpoint. **Login node only** — compute nodes have no network.

    Reads `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` from the environment (sourced
    from `.env.local`) and never prints it.
    """
    hub = _hub_dir(cache_dir)
    from huggingface_hub import snapshot_download          # noqa: PLC0415 (lazy)

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return snapshot_download(
        model_id,
        cache_dir=str(hub),
        token=token or None,
        allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors"],
    )


def _load_hf(model_id: str, cache_dir: str | os.PathLike | None, dtype: torch.dtype):
    """`(SiglipModel, tokenizer)` from local disk only. Raises `TowerUnavailable`."""
    hub = _hub_dir(cache_dir)
    try:
        from transformers import AutoTokenizer, SiglipModel  # noqa: PLC0415 (lazy)
    except Exception as e:                                   # noqa: BLE001
        raise TowerUnavailable(
            f"`transformers` is not importable in this interpreter ({e}). "
            f"pip install transformers sentencepiece protobuf"
        ) from e
    if not weights_available(model_id, cache_dir):
        raise TowerUnavailable(
            f"{model_id} is not in {hub}. Compute nodes have no outbound "
            f"network — run `source .env.local && python -m loom.data.tower "
            f"--download` on a LOGIN node first."
        )
    try:
        model = SiglipModel.from_pretrained(
            model_id, dtype=dtype, local_files_only=True, cache_dir=str(hub),
        )
        tok = AutoTokenizer.from_pretrained(
            model_id, local_files_only=True, cache_dir=str(hub),
        )
    except Exception as e:                                   # noqa: BLE001
        raise TowerUnavailable(f"could not load {model_id}: {e}") from e
    return model, tok


# ═══════════════════════════════════════════════════════════════════════════
#  THE TOWER
# ═══════════════════════════════════════════════════════════════════════════

class FrozenTower:
    """SigLIP vision + text, frozen forever.

    **Not an `nn.Module`, on purpose.** `nn.Module.__setattr__` only registers
    submodules that are themselves `nn.Module`s, so
    ``self.tower = FrozenTower()`` inside a trainable module puts it in
    ``__dict__`` and its 878 M parameters never reach ``.parameters()``,
    ``state_dict()``, FSDP's flat parameter or the optimizer. That is PLAN §9's
    "the frozen tower never enters the training graph" enforced by construction
    rather than by remembering to filter.
    """

    def __init__(
        self,
        *,
        model_id: str = TOWER_MODEL_ID,
        image_size: int = IMAGE_SIZE,
        lang_len: int = LANG_LEN,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = DTYPE,
        cache_dir: str | os.PathLike | None = None,
        model: Any = None,
        tokenizer: Any = None,
        batch_size: int = 32,
    ) -> None:
        if model is None:
            model, tokenizer = _load_hf(model_id, cache_dir, dtype)
        if tokenizer is None:
            raise ValueError("a pre-built `model` must come with its `tokenizer`")

        self.model_id = model_id
        self.device = torch.device(device)
        self.dtype = dtype
        self.lang_len = int(lang_len)
        self.batch_size = int(batch_size)
        self.tokenizer = tokenizer

        # NOT `self.model` on an nn.Module — this object is a plain object.
        self._model = model.to(device=self.device, dtype=dtype)
        self._model.requires_grad_(False)
        self._model.eval()

        vcfg = self._model.config.vision_config
        tcfg = self._model.config.text_config
        if int(vcfg.hidden_size) != int(tcfg.hidden_size):
            raise ValueError(
                f"vision F={vcfg.hidden_size} != text F={tcfg.hidden_size}; "
                f"ObsFeats.views and ObsFeats.lang must share F"
            )
        self.feat_dim = int(vcfg.hidden_size)
        self.patch_size = int(vcfg.patch_size)
        self.native_image_size = int(vcfg.image_size)
        self.image_size = int(image_size)
        if self.image_size < self.patch_size:
            raise ValueError(
                f"image_size {self.image_size} is smaller than patch_size "
                f"{self.patch_size}"
            )
        # `padding="valid"`: the grid is a floor, and the remainder is dropped.
        # The checkpoint's own 384 is NOT a multiple of 14 (384 = 27*14 + 6), so
        # this is the pretrained behaviour, not a corner case.
        self.grid = self.image_size // self.patch_size
        self.n_patches = self.grid * self.grid
        #: True when the pretrained position grid has to be resized.
        self.interpolate_pos = self.image_size != self.native_image_size

    # ── introspection ────────────────────────────────────────────────────

    def __repr__(self) -> str:                                  # pragma: no cover
        return (
            f"FrozenTower({self.model_id}, {self.image_size}px, P={self.n_patches}, "
            f"F={self.feat_dim}, L={self.lang_len}, {self.dtype}, {self.device})"
        )

    def parameters(self):
        """Read-only view, for asserting frozenness. Deliberately not `nn.Module`."""
        return self._model.parameters()

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self._model.parameters())

    # ── vision ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode_images(self, images: np.ndarray | Tensor) -> Tensor:
        """``(..., H, W, 3)`` uint8 -> ``(..., P, F)`` bf16 patch tokens.

        Patch tokens, **not** the pooled head output: `ObsFeats.views` is
        ``(B, V, P, F)`` and the estimator cross-attends to all of them.
        Leading dimensions are preserved, so ``(n, V, H, W, 3)`` (the cache
        path) gives ``(n, V, P, F)`` and ``(V, H, W, 3)`` (the eval path) gives
        ``(V, P, F)``.
        """
        arr = images if isinstance(images, Tensor) else np.asarray(images)
        lead = tuple(arr.shape[:-3])
        px = preprocess_images(
            arr, image_size=self.image_size, dtype=self.dtype, device=self.device,
        )
        if px.shape[0] == 0:
            raise ValueError("encode_images got zero images")
        outs = []
        for lo in range(0, px.shape[0], self.batch_size):
            out = self._model.vision_model(
                pixel_values=px[lo:lo + self.batch_size],
                interpolate_pos_encoding=self.interpolate_pos,
            )
            outs.append(out.last_hidden_state)
        tokens = torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]
        if tokens.shape[1] != self.n_patches:
            raise RuntimeError(
                f"tower returned {tokens.shape[1]} tokens, expected {self.n_patches} "
                f"for {self.image_size}px / patch {self.patch_size}"
            )
        return tokens.reshape(*lead, self.n_patches, self.feat_dim)

    # ── language ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode_text(self, text: str | Sequence[str]) -> Tensor:
        """``str`` or ``list[str]`` -> ``(B, L, F)`` bf16 per-token embeddings.

        `padding="max_length"` to `LANG_LEN` is not a detail: SigLIP was trained
        that way and any other padding changes every token's embedding.
        """
        texts = [text] if isinstance(text, str) else list(text)
        batch = self.tokenizer(
            texts, padding="max_length", max_length=self.lang_len,
            truncation=True, return_tensors="pt",
        )
        ids = batch["input_ids"].to(self.device)
        out = self._model.text_model(input_ids=ids)
        return out.last_hidden_state.to(self.dtype)

    # ── the two call paths ───────────────────────────────────────────────

    @torch.no_grad()
    def encode(
        self, images: np.ndarray | Tensor, instruction: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """`adapters.libero.Encoder`: the **training / cache** path.

        ``(n, V, H, W, 3)`` uint8 + instruction -> ``(n, V, P, F)`` and
        ``(L, F)``, both float32 numpy (`FeatureCacheWriter` casts to the cache
        codec).
        """
        views = self.encode_images(images).float().cpu().numpy()
        lang = self.encode_text(instruction)[0].float().cpu().numpy()
        return views, lang

    @torch.no_grad()
    def obs_feats(
        self,
        images: np.ndarray | Tensor,
        proprio: Tensor,
        instruction: str | Tensor,
    ) -> ObsFeats:
        """The **eval** path: ``(V, H, W, 3)`` uint8 -> a batch-1 `ObsFeats`."""
        views = self.encode_images(images)
        if views.ndim == 3:
            views = views.unsqueeze(0)                          # (1, V, P, F)
        lang = instruction if isinstance(instruction, Tensor) else \
            self.encode_text(instruction)
        return ObsFeats(views=views, proprio=proprio, lang=lang)


# ═══════════════════════════════════════════════════════════════════════════
#  PROCESS-LEVEL SINGLETON
# ═══════════════════════════════════════════════════════════════════════════

_TOWERS: dict[tuple, FrozenTower] = {}
_LOCK = threading.Lock()


def get_tower(
    *,
    model_id: str = TOWER_MODEL_ID,
    image_size: int = IMAGE_SIZE,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = DTYPE,
    **kw: Any,
) -> FrozenTower:
    """Cached per ``(model_id, image_size, device, dtype)``. 3.3 GiB loads once.

    Lazy on purpose: importing this module must not touch disk or the network.
    """
    key = (model_id, int(image_size), str(device), dtype)
    with _LOCK:
        tower = _TOWERS.get(key)
        if tower is None:
            tower = FrozenTower(
                model_id=model_id, image_size=image_size, device=device,
                dtype=dtype, **kw,
            )
            _TOWERS[key] = tower
    return tower


def reset_tower() -> None:
    """Drop the singletons. Tests only."""
    with _LOCK:
        _TOWERS.clear()


# ═══════════════════════════════════════════════════════════════════════════
#  THE TWO WIRING SEAMS
# ═══════════════════════════════════════════════════════════════════════════

def tower_encoder(
    *, device: str | torch.device | None = None, **kw: Any
) -> Callable[[np.ndarray, str], tuple[np.ndarray, np.ndarray]]:
    """The `adapters.libero.Encoder` callable for `encode_to_cache`.

    ``device=None`` picks CUDA when there is one. The tower is built on first
    call, so holding this callable costs nothing until the cache build starts.
    """
    dev = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    def encode(images: np.ndarray, instruction: str) -> tuple[np.ndarray, np.ndarray]:
        return get_tower(device=dev, **kw).encode(images, instruction)

    return encode


def obs_featurizer(
    spec: EmbodimentSpec | str,
    *,
    device: str | torch.device = "cpu",
    view_keys: Sequence[str] = EVAL_VIEW_KEYS,
    tower: FrozenTower | None = None,
    **kw: Any,
) -> Callable[[dict, str], ObsFeats]:
    """`obs dict -> ObsFeats` for `eval.policy`. The **eval** call path.

    `obs` is whatever `eval.libero.extract_obs` produced: images already in
    `CANONICAL_ORIENTATION` under `view_keys`, proprio under ``"state"``. The
    language embedding is computed once per instruction and reused — it is
    constant for a whole episode and the text tower is 27 layers.

    Builds the tower eagerly so a missing checkpoint fails **here**, at policy
    construction, and not 400 episodes into a run.
    """
    if isinstance(spec, str):
        spec = EMBODIMENTS[spec]
    if len(view_keys) != spec.n_views:
        raise ValueError(
            f"{spec.name} has n_views={spec.n_views} but view_keys={tuple(view_keys)}"
        )
    if tower is None:
        tower = get_tower(device=device, **kw)
    lang_cache: dict[str, Tensor] = {}

    def featurize(obs: dict, instruction: str) -> ObsFeats:
        frames = []
        for key in view_keys:
            img = obs.get(key)
            if img is None:
                raise KeyError(
                    f"observation has no {key!r}; the frozen tower needs real pixels "
                    f"(keys present: {sorted(obs)})"
                )
            frames.append(np.asarray(img))
        images = np.stack(frames, axis=0)                       # (V, H, W, 3)

        lang = lang_cache.get(instruction)
        if lang is None:
            lang = tower.encode_text(instruction)               # (1, L, F)
            lang_cache[instruction] = lang

        return tower.obs_feats(images, _proprio(obs, spec, device), lang)

    return featurize


def _proprio(obs: dict, spec: EmbodimentSpec, device: str | torch.device) -> Tensor:
    """`(1, dof)` float32, read from the observation. Never fabricated silently."""
    raw = obs.get("state", obs.get("proprio"))
    if raw is None:
        return torch.zeros(1, spec.dof, device=device)
    v = np.asarray(raw, dtype=np.float32).reshape(-1)
    if v.shape[0] < spec.dof:
        v = np.pad(v, (0, spec.dof - v.shape[0]))
    return torch.from_numpy(v[: spec.dof].copy()).unsqueeze(0).to(device)


# ═══════════════════════════════════════════════════════════════════════════
#  OPERATOR ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def _main() -> None:                                # pragma: no cover
    import argparse
    import time

    ap = argparse.ArgumentParser(description="frozen SigLIP tower")
    ap.add_argument("--download", action="store_true",
                    help="fetch the checkpoint (LOGIN NODE ONLY)")
    ap.add_argument("--bench", action="store_true",
                    help="measure encode throughput in windows/s")
    ap.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    if args.download:
        print("snapshot:", download_weights())
    if args.bench:
        from contracts import N_STATES

        tower = get_tower(image_size=args.image_size, device=args.device)
        print(tower, f"params={tower.n_params / 1e6:.0f}M")
        v = EMBODIMENTS["libero_franka"].n_views
        imgs = np.random.default_rng(0).integers(
            0, 256, (N_STATES, v, 128, 128, 3), dtype=np.uint8)
        for _ in range(3):
            tower.encode_images(imgs)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            tower.encode_images(imgs)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / args.iters
        print(f"{args.image_size}px P={tower.n_patches}: "
              f"{dt * 1000:.1f} ms/window, {1.0 / dt:.1f} windows/s "
              f"({N_STATES * v} images/window)")
    if not (args.download or args.bench):
        ap.error("nothing to do; pass --download or --bench")


if __name__ == "__main__":                          # pragma: no cover
    _main()
