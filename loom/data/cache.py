"""
LOOM — frozen-tower feature cache.

Encode once, never re-encode. The frozen vision tower never enters the training
graph (PLAN §9), so its output is a pure function of the pixels and belongs on
disk. This module owns that on-disk format, its version, and its manifest.

Keyed by **source frame**, not by window. With ``canonical.segment``'s default
stride of ``H_OP`` the 5 boundary states of consecutive windows overlap 4 deep;
caching per window would cost ~5x. See ``canonical.required_source_frames``.

Format
------
::

    <root>/manifest.json            format version, codec, shapes, every entry
    <root>/feats/<slug>.bin         one trajectory: views | scales | proprio | lang

``manifest.json`` carries ``format_version``. Opening a cache written by a
different version raises ``CacheFormatError`` — a format change is a loud
failure, never silent garbage. Shapes and codec are validated on every read.

Storage-format profiling (the day-one call, PLAN §4.A)
-----------------------------------------------------
Measured 2026-08-16 on this cluster's Lustre (fs11), login node, single stream,
**cold reads** — each ``.bin`` is evicted with ``posix_fadvise(DONTNEED)``
between write and read. Geometry is one LIBERO state: V=2 views, P=196 patches,
F=1152 (2-stream); a window is ``N_STATES`` = 5 states. Reproduce with
``python -m loom.data.cache --profile --root .``.

==========  ============  =============  ============  ============  ==========
codec       MiB / window  write (MiB/s)  read (MiB/s)  read (win/s)  rel L2 err
==========  ============  =============  ============  ============  ==========
fp16            4.34           127            650          150        exact
int8            2.19            94            189           86        0.0284
in-graph        0 (I/O)          —              —           75 (*)    exact
==========  ============  =============  ============  ============  ==========

(*) modelled, not measured — the login node has no GPU. A SigLIP-SO400M-class
tower is ~1.7e11 FLOPs per 196-token image (2·N·T, N=428M); a window is 5 states
x 2 views = 10 images = 1.7 TFLOP; an A100 at 312 TFLOP/s bf16 peak and a
realistic 40% MFU delivers 125 TFLOP/s → **13.4 ms of tower forward per
window**, i.e. 107 ms for a batch of 8, every step, forever.

Reading the table
-----------------
* **Bytes are not the deciding axis; windows/s is.** int8 halves the bytes and
  is still 1.7x *slower* per window, because a window then costs two preads
  (codes + per-token scales) plus a dequantise. Halving I/O that was never the
  bottleneck buys nothing and costs 2.8% relative L2 on the features.
* fp16 costs 6.7 ms/window of *cold* I/O against option (c)'s 13.4 ms/window of
  FLOPs — 2x cheaper before any of the structural arguments even apply, and the
  structural arguments are the real ones: I/O runs in DataLoader workers and
  overlaps the training step, and the cold cost is paid **once** —
  the whole four-suite LIBERO cache is ~2000 demos x ~28 cached frames x 0.86
  MiB ≈ **48 GiB fp16 / 24 GiB int8**, which sits in a compute node's page cache
  after the first epoch. Option (c)'s FLOPs contend with the training step on
  the same GPU and recur on every sample of every epoch.
* Numbers vary 2-3x run to run on a shared login node (a concurrent run of the
  same profile measured fp16 at 273 MiB/s). The ordering is stable.

**Update (Team I, tower landed).** The geometry above was profiled at a placeholder
``P = 196``. The real tower (``loom.data.tower``, SigLIP-so400m-patch14 at 224 px)
gives ``P = 256``, ``F = 1152``, ``L = 64``: **1.125 MiB/frame, 5.77 MiB/window**,
and the measured LIBERO build — 2000 demos, 56 189 windows over 64 189 cached
source frames — is **70.8 GiB fp16**, not 48. The ordering of the table is
unchanged and so is the choice. ``tower_cache_spec()`` below returns that
geometry without encoding anything. Building the cache at the tower's *native*
384 px instead would be ``P = 729`` and **201 GiB**, which is why it is not the
default; see `loom.data.tower`'s docstring for that measurement.

**Recommendation and default: fp16.** Lossless at the precision the model trains
in, fastest per window, and its 71 GiB fits in node RAM after one epoch.
``int8`` is implemented and tested and remains the lever for the 7-stream bodies
in R2 (3x the views → ~13 MiB/window); if it is ever turned on, interleave the
scales into the frame row first so a window is still one read. Option (c) is
rejected: it pays the tower on every epoch and violates PLAN §9's "the frozen
tower never enters the training graph / features are cached".
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from contracts import N_STATES

__all__ = [
    "CACHE_FORMAT_VERSION", "CODECS", "DEFAULT_CODEC",
    "CacheFormatError", "CacheSpec", "FeatureCacheWriter", "FeatureCache",
    "quantize_int8", "dequantize_int8",
    "FormatProfile", "profile_formats", "estimate_in_graph_cost",
    "default_encoder", "tower_cache_spec",
]


# ═══════════════════════════════════════════════════════════════════════════
#  FORMAT
# ═══════════════════════════════════════════════════════════════════════════

#: Bump on ANY layout change. Readers refuse a mismatch; they do not adapt.
CACHE_FORMAT_VERSION = 1

MANIFEST_NAME = "manifest.json"
FEAT_DIR = "feats"

CODECS = ("fp16", "int8")
DEFAULT_CODEC = "fp16"          # see the profiling table in the module docstring

_ALIGN = 64                     # section alignment inside a .bin


class CacheFormatError(RuntimeError):
    """Raised on a version, codec or shape mismatch. Never recovered from."""


@dataclass(frozen=True)
class CacheSpec:
    """Geometry every entry in one cache root must share."""

    codec: str
    n_views: int
    n_patches: int
    feat_dim: int
    dof: int
    lang_len: int

    def __post_init__(self) -> None:
        if self.codec not in CODECS:
            raise CacheFormatError(f"unknown codec {self.codec!r}, expected one of {CODECS}")
        for field in ("n_views", "n_patches", "feat_dim", "dof", "lang_len"):
            if getattr(self, field) <= 0:
                raise CacheFormatError(f"{field} must be positive, got {getattr(self, field)}")

    @property
    def bytes_per_frame(self) -> int:
        n = self.n_views * self.n_patches
        if self.codec == "fp16":
            views = n * self.feat_dim * 2
            scales = 0
        else:
            views = n * self.feat_dim * 1
            scales = n * 2                       # one fp16 scale per (view, patch)
        return views + scales + self.dof * 4

    @property
    def bytes_per_window(self) -> int:
        """N_STATES frames plus the shared language embedding."""
        return N_STATES * self.bytes_per_frame + self.lang_len * self.feat_dim * 2


# ═══════════════════════════════════════════════════════════════════════════
#  THE ENCODER  —  what fills the cache
#
#  `adapters.libero.encode_to_cache` takes an `Encoder` callable so the frozen
#  tower stays out of `loom/data` proper. This is where that callable comes
#  from. **Lazy on purpose:** `import loom.data.cache` must stay a pure-numpy
#  import that downloads nothing and loads no weights — the loader imports it in
#  every DataLoader worker.
# ═══════════════════════════════════════════════════════════════════════════

def default_encoder(**kw) -> "Callable[[np.ndarray, str], tuple[np.ndarray, np.ndarray]]":
    """The frozen SigLIP tower, as `adapters.libero.Encoder`.

    ``encode_to_cache(default_encoder(), cache_root)``. The tower itself is not
    constructed until the first call, so holding the callable is free.
    """
    from .tower import tower_encoder                # noqa: PLC0415 (lazy by design)

    return tower_encoder(**kw)


def tower_cache_spec(embodiment: str = "libero_franka", codec: str | None = None) -> CacheSpec:
    """The `CacheSpec` `default_encoder` will produce, without encoding anything.

    Multiply ``bytes_per_frame`` by the number of frames `encode_to_cache` will
    visit to size the cache before committing a GPU-hour to filling it.
    """
    from contracts import EMBODIMENTS                # noqa: PLC0415
    from .tower import FEAT_DIM, LANG_LEN, N_PATCHES  # noqa: PLC0415

    spec = EMBODIMENTS[embodiment]
    return CacheSpec(
        codec=codec or DEFAULT_CODEC,
        n_views=spec.n_views, n_patches=N_PATCHES, feat_dim=FEAT_DIM,
        dof=spec.dof, lang_len=LANG_LEN,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  INT8 CODEC
# ═══════════════════════════════════════════════════════════════════════════

def quantize_int8(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric per-token int8. Returns (int8 codes, fp16 scales).

    The scale is per (frame, view, patch) — one per token, not per tensor. ViT
    activations have channel outliers; a single tensor-wide scale loses ~4x more.
    """
    x = np.asarray(x, dtype=np.float32)
    amax = np.abs(x).max(axis=-1, keepdims=True)
    scale = np.maximum(amax, 1e-8) / 127.0
    q = np.rint(x / scale).clip(-127, 127).astype(np.int8)
    return q, scale.squeeze(-1).astype(np.float16)


def dequantize_int8(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """(codes, scales) -> float16 features."""
    return (q.astype(np.float32) * scale.astype(np.float32)[..., None]).astype(np.float16)


# ═══════════════════════════════════════════════════════════════════════════
#  LAYOUT
# ═══════════════════════════════════════════════════════════════════════════

def _slug(traj_id: str) -> str:
    """Filesystem-safe, collision-free file stem for an arbitrary trajectory id."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", traj_id)[:96]
    digest = hashlib.blake2b(traj_id.encode("utf-8"), digest_size=6).hexdigest()
    return f"{safe}-{digest}"


def _pad(n: int) -> int:
    return (n + _ALIGN - 1) // _ALIGN * _ALIGN


def _sections(spec: CacheSpec, n_frames: int) -> dict[str, tuple[int, tuple[int, ...], str]]:
    """name -> (byte offset, shape, dtype). Sections are 64B-aligned.

    Frame-major: row `i` of ``views`` is all of frame `i`. The cache stores only
    the frames windows land on, in ascending order, so one window's 5 boundary
    frames are 5 consecutive rows and read as one contiguous span.
    """
    v, p, f = spec.n_views, spec.n_patches, spec.feat_dim
    out: dict[str, tuple[int, tuple[int, ...], str]] = {}
    off = 0
    if spec.codec == "fp16":
        out["views"] = (off, (n_frames, v, p, f), "float16")
        off = _pad(off + n_frames * v * p * f * 2)
    else:
        out["views"] = (off, (n_frames, v, p, f), "int8")
        off = _pad(off + n_frames * v * p * f)
        out["scales"] = (off, (n_frames, v, p), "float16")
        off = _pad(off + n_frames * v * p * 2)
    out["proprio"] = (off, (n_frames, spec.dof), "float32")
    off = _pad(off + n_frames * spec.dof * 4)
    out["lang"] = (off, (spec.lang_len, f), "float16")
    off = _pad(off + spec.lang_len * f * 2)
    out["__total__"] = (off, (), "")
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  WRITER
# ═══════════════════════════════════════════════════════════════════════════

class FeatureCacheWriter:
    """Single-process writer. Use as a context manager; the manifest lands on exit.

    Concurrent writers must use separate roots (one per SLURM task) — the
    manifest is rewritten wholesale, so two processes sharing a root would
    clobber each other. Merging shards is a `cp` of the ``feats/`` dirs plus a
    ``json`` union; deliberately not automated here.
    """

    def __init__(self, root: str | os.PathLike, spec: CacheSpec, overwrite: bool = False) -> None:
        self.root = Path(root)
        self.spec = spec
        if self.root.exists() and overwrite:
            shutil.rmtree(self.root)
        (self.root / FEAT_DIR).mkdir(parents=True, exist_ok=True)
        self.entries: dict[str, dict] = {}
        manifest = self.root / MANIFEST_NAME
        if manifest.exists():
            prev = json.loads(manifest.read_text())
            _check_version(prev, self.root)
            if prev["spec"] != asdict(spec):
                raise CacheFormatError(
                    f"{self.root}: existing cache has spec {prev['spec']}, writer has {asdict(spec)}"
                )
            self.entries = prev["entries"]

    # ── writing ──────────────────────────────────────────────────────────
    def write(
        self,
        traj_id: str,
        *,
        frames: Sequence[int],
        views: np.ndarray,          # (n, V, P, F) float
        proprio: np.ndarray,        # (n, dof)
        lang: np.ndarray,           # (L, F)
        embodiment: str,
        src_fps: float,
        meta: dict | None = None,
    ) -> None:
        spec = self.spec
        frames = [int(x) for x in frames]
        n = len(frames)
        views = np.asarray(views, dtype=np.float32)
        proprio = np.asarray(proprio, dtype=np.float32)
        lang = np.asarray(lang, dtype=np.float32)
        want = (n, spec.n_views, spec.n_patches, spec.feat_dim)
        if views.shape != want:
            raise CacheFormatError(f"{traj_id}: views {views.shape}, cache spec wants {want}")
        if proprio.shape != (n, spec.dof):
            raise CacheFormatError(
                f"{traj_id}: proprio {proprio.shape}, cache spec wants {(n, spec.dof)}"
            )
        if lang.shape != (spec.lang_len, spec.feat_dim):
            raise CacheFormatError(
                f"{traj_id}: lang {lang.shape}, cache spec wants "
                f"{(spec.lang_len, spec.feat_dim)}"
            )
        if len(set(frames)) != n:
            raise ValueError(f"{traj_id}: duplicate source frames in {frames}")

        sec = _sections(spec, n)
        path = self.root / FEAT_DIR / f"{_slug(traj_id)}.bin"
        buf = bytearray(sec["__total__"][0])
        mv = memoryview(buf)

        def put(name: str, arr: np.ndarray) -> None:
            off, shape, dt = sec[name]
            a = np.ascontiguousarray(arr, dtype=np.dtype(dt))
            assert a.shape == shape, (name, a.shape, shape)
            mv[off:off + a.nbytes] = a.tobytes()

        if spec.codec == "fp16":
            put("views", views)
        else:
            q, scale = quantize_int8(views)
            put("views", q)
            put("scales", scale)
        put("proprio", proprio)
        put("lang", lang)

        tmp = path.with_suffix(".bin.tmp")
        with open(tmp, "wb") as fh:
            fh.write(buf)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)

        self.entries[traj_id] = {
            "file": f"{FEAT_DIR}/{path.name}",
            "frames": frames,
            "n_frames": n,
            "embodiment": embodiment,
            "src_fps": float(src_fps),
            "nbytes": len(buf),
            "meta": meta or {},
        }

    def flush(self) -> None:
        """Write the manifest. Atomic: tmp file + rename."""
        doc = {
            "format_version": CACHE_FORMAT_VERSION,
            "spec": asdict(self.spec),
            "entries": self.entries,
        }
        tmp = self.root / (MANIFEST_NAME + ".tmp")
        tmp.write_text(json.dumps(doc, separators=(",", ":")))
        os.replace(tmp, self.root / MANIFEST_NAME)

    def __enter__(self) -> "FeatureCacheWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.flush()


# ═══════════════════════════════════════════════════════════════════════════
#  READER
# ═══════════════════════════════════════════════════════════════════════════

def _check_version(doc: dict, root: Path) -> None:
    got = doc.get("format_version")
    if got != CACHE_FORMAT_VERSION:
        raise CacheFormatError(
            f"{root}: cache format version {got!r}, this build reads "
            f"{CACHE_FORMAT_VERSION}. Re-encode; there is no migration path and "
            f"reading it as-is would be silent garbage."
        )


class FeatureCache:
    """Read-only, fork-safe, random-access reader over one cache root.

    Reads with coalesced ``pread``, not ``mmap``. On Lustre an mmap read is
    served by 4 KiB page faults with no useful readahead; a window's five
    boundary frames are *consecutive* rows in the file (the cache stores exactly
    the frames the windows land on, in order), so they collapse into a single
    contiguous ~4 MiB read instead of 1100 page faults.
    """

    MAX_OPEN = 64

    def __init__(self, root: str | os.PathLike) -> None:
        self.root = Path(root)
        manifest = self.root / MANIFEST_NAME
        if not manifest.exists():
            raise CacheFormatError(f"{self.root}: no {MANIFEST_NAME}")
        doc = json.loads(manifest.read_text())
        _check_version(doc, self.root)
        try:
            self.spec = CacheSpec(**doc["spec"])
        except TypeError as e:
            raise CacheFormatError(f"{self.root}: unreadable spec {doc.get('spec')!r} ({e})") from e
        self.entries: dict[str, dict] = doc["entries"]
        self._pos: dict[str, dict[int, int]] = {}
        self._sec: dict[str, dict] = {}
        self._fds: dict[str, int] = {}
        self._pid = os.getpid()

    # ── introspection ────────────────────────────────────────────────────
    def __contains__(self, traj_id: str) -> bool:
        return traj_id in self.entries

    def __len__(self) -> int:
        return len(self.entries)

    def keys(self) -> Iterable[str]:
        return self.entries.keys()

    def frames(self, traj_id: str) -> np.ndarray:
        return np.asarray(self.entries[traj_id]["frames"], dtype=np.int64)

    @property
    def total_bytes(self) -> int:
        return sum(int(e["nbytes"]) for e in self.entries.values())

    # ── reading ──────────────────────────────────────────────────────────
    def _rows(self, traj_id: str, frames: Sequence[int]) -> np.ndarray:
        pos = self._pos.get(traj_id)
        if pos is None:
            pos = {int(f): i for i, f in enumerate(self.entries[traj_id]["frames"])}
            self._pos[traj_id] = pos
        try:
            return np.array([pos[int(f)] for f in frames], dtype=np.int64)
        except KeyError as e:
            raise KeyError(
                f"{traj_id}: source frame {e.args[0]} is not in the cache "
                f"(cached frames are {self.entries[traj_id]['frames'][:8]}...). "
                f"The window stride used to build the cache must match the one "
                f"used to sample it."
            ) from None

    def _layout(self, traj_id: str) -> dict:
        sec = self._sec.get(traj_id)
        if sec is None:
            sec = _sections(self.spec, int(self.entries[traj_id]["n_frames"]))
            self._sec[traj_id] = sec
        return sec

    def _fd(self, traj_id: str) -> int:
        if self._pid != os.getpid():        # forked worker: the parent's fds are copies
            self._fds.clear()
            self._pid = os.getpid()
        fd = self._fds.get(traj_id)
        if fd is None:
            if len(self._fds) >= self.MAX_OPEN:
                victim, vfd = next(iter(self._fds.items()))
                self._fds.pop(victim)
                try:
                    os.close(vfd)
                except OSError:             # pragma: no cover
                    pass
            fd = os.open(self.root / self.entries[traj_id]["file"], os.O_RDONLY)
            self._fds[traj_id] = fd
        return fd

    def _read_rows(self, traj_id: str, name: str, rows: np.ndarray) -> np.ndarray:
        """Rows of one section, coalescing consecutive rows into one pread."""
        off, shape, dt = self._layout(traj_id)[name]
        dtype = np.dtype(dt)
        row_shape = tuple(shape[1:])
        row_bytes = int(np.prod(row_shape)) * dtype.itemsize
        fd = self._fd(traj_id)
        out = np.empty((len(rows),) + row_shape, dtype=dtype)
        raw = memoryview(out).cast("B")     # preadv lands straight in `out`
        i, n = 0, len(rows)
        while i < n:
            j = i + 1
            while j < n and int(rows[j]) == int(rows[j - 1]) + 1:
                j += 1
            want = (j - i) * row_bytes
            got = os.preadv(fd, [raw[i * row_bytes:j * row_bytes]],
                            off + int(rows[i]) * row_bytes)
            if got != want:
                raise CacheFormatError(
                    f"{traj_id}/{name}: short read, {got} of {want} bytes — "
                    f"the .bin does not match the manifest"
                )
            i = j
        return out

    def _read_whole(self, traj_id: str, name: str) -> np.ndarray:
        off, shape, dt = self._layout(traj_id)[name]
        dtype = np.dtype(dt)
        nbytes = int(np.prod(shape)) * dtype.itemsize
        buf = os.pread(self._fd(traj_id), nbytes, off)
        if len(buf) != nbytes:
            raise CacheFormatError(f"{traj_id}/{name}: short read")
        return np.frombuffer(buf, dtype=dtype).reshape(shape).copy()

    def read(self, traj_id: str, frames: Sequence[int]) -> dict:
        """Features for the given SOURCE frames of one trajectory.

        Returns ``views`` (n, V, P, F) float16, ``proprio`` (n, dof) float32,
        ``lang`` (L, F) float16. int8 caches are dequantised here.
        """
        if traj_id not in self.entries:
            raise KeyError(f"{traj_id!r} not in cache {self.root}")
        rows = self._rows(traj_id, frames)
        if self.spec.codec == "fp16":
            views = self._read_rows(traj_id, "views", rows)
        else:
            views = dequantize_int8(self._read_rows(traj_id, "views", rows),
                                    self._read_rows(traj_id, "scales", rows))
        return {
            "views": views,
            "proprio": self._read_rows(traj_id, "proprio", rows),
            "lang": self._read_whole(traj_id, "lang"),
            "embodiment": self.entries[traj_id]["embodiment"],
            "src_fps": float(self.entries[traj_id]["src_fps"]),
        }

    def close(self) -> None:
        for fd in self._fds.values():
            try:
                os.close(fd)
            except OSError:                 # pragma: no cover
                pass
        self._fds.clear()

    # file descriptors do not pickle; workers reopen them lazily
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_fds"] = {}
        return state


# ═══════════════════════════════════════════════════════════════════════════
#  PROFILING  —  the day-one storage-format call
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FormatProfile:
    codec: str
    mib_per_window: float
    write_mib_s: float
    read_mib_s: float
    rel_l2_err: float
    ms_per_window: float = 0.0

    def row(self) -> str:
        w = f"{self.write_mib_s:10.0f}" if self.write_mib_s else " " * 10
        r = f"{self.read_mib_s:10.0f}" if self.read_mib_s else " " * 10
        return (f"{self.codec:<12}{self.mib_per_window:>10.2f}{w}{r}"
                f"{self.rel_l2_err:>14.4f}")


def _drop_cache(path: Path) -> None:
    """Evict a file from the page cache so the next read is honest."""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except (OSError, AttributeError):       # pragma: no cover - platform dependent
        pass


def profile_formats(
    root: str | os.PathLike,
    *,
    n_views: int = 2,
    n_patches: int = 196,
    feat_dim: int = 1152,
    dof: int = 7,
    lang_len: int = 16,
    n_traj: int = 8,
    frames_per_traj: int = 32,
    n_reads: int = 64,
    seed: int = 0,
) -> list[FormatProfile]:
    """Measure fp16 vs int8 on the real filesystem: bytes, write, read, error.

    Reads are cold — each file is dropped from the page cache after writing.
    """
    root = Path(root)
    rng = np.random.default_rng(seed)
    # ViT activations: mostly unit-scale with a few large channels
    base = rng.standard_normal((frames_per_traj, n_views, n_patches, feat_dim), dtype=np.float32)
    base[..., ::97] *= 8.0
    proprio = rng.standard_normal((frames_per_traj, dof), dtype=np.float32)
    lang = rng.standard_normal((lang_len, feat_dim), dtype=np.float32)

    out: list[FormatProfile] = []
    for codec in CODECS:
        spec = CacheSpec(codec, n_views, n_patches, feat_dim, dof, lang_len)
        sub = root / f"profile_{codec}"
        if sub.exists():
            shutil.rmtree(sub)
        written = 0
        t0 = time.perf_counter()
        with FeatureCacheWriter(sub, spec) as w:
            for i in range(n_traj):
                w.write(
                    f"traj_{i}",
                    frames=list(range(frames_per_traj)),
                    views=base,
                    proprio=proprio,
                    lang=lang,
                    embodiment="libero_franka",
                    src_fps=20.0,
                )
                written += spec.bytes_per_frame * frames_per_traj
        write_s = time.perf_counter() - t0

        for p in (sub / FEAT_DIR).glob("*.bin"):
            _drop_cache(p)

        cache = FeatureCache(sub)
        starts = rng.integers(0, frames_per_traj - N_STATES, size=n_reads)
        plan = [(f"traj_{int(rng.integers(0, n_traj))}",
                 list(range(int(s), int(s) + N_STATES))) for s in starts]

        read_bytes = 0
        last = None
        t0 = time.perf_counter()                       # reconstruction error is
        for tid, fr in plan:                           # measured outside the timer
            last = cache.read(tid, fr)
            read_bytes += spec.bytes_per_frame * N_STATES
        read_s = time.perf_counter() - t0

        rel = 0.0
        if codec == "int8" and last is not None:
            ref = base[plan[-1][1]].astype(np.float32)
            d = last["views"].astype(np.float32) - ref
            rel = float(np.linalg.norm(d) / np.linalg.norm(ref))
        cache.close()
        shutil.rmtree(sub, ignore_errors=True)

        mib = 1024.0 * 1024.0
        out.append(FormatProfile(
            codec=codec,
            mib_per_window=spec.bytes_per_window / mib,
            write_mib_s=(written / mib) / max(write_s, 1e-9),
            read_mib_s=(read_bytes / mib) / max(read_s, 1e-9),
            rel_l2_err=rel,
        ))

    out.append(estimate_in_graph_cost(n_views=n_views, n_patches=n_patches))
    return out


def estimate_in_graph_cost(
    *,
    n_views: int = 2,
    n_patches: int = 196,
    tower_params: float = 428e6,
    device_tflops: float = 312.0,
    mfu: float = 0.40,
) -> FormatProfile:
    """Option (c): tower resident under no_grad, FLOPs instead of I/O.

    Modelled, not measured — the login node has no GPU. FLOPs per image is the
    standard 2·N·T forward estimate for a SigLIP-SO400M-class tower.
    """
    flops = 2.0 * tower_params * n_patches * n_views * N_STATES
    ms = 1000.0 * flops / (device_tflops * 1e12 * mfu)
    return FormatProfile("in-graph", 0.0, 0.0, 0.0, 0.0, ms_per_window=ms)


def _main() -> None:  # pragma: no cover - operator entry point
    import argparse
    import tempfile

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--build", metavar="CACHE_ROOT", default=None,
                    help="encode LIBERO with the frozen tower into CACHE_ROOT (needs a GPU)")
    ap.add_argument("--root", default=None, help="where to write the scratch cache")
    ap.add_argument("--suites", default=None, help="comma-separated; default all four")
    ap.add_argument("--max-demos", type=int, default=None)
    ap.add_argument("--n-traj", type=int, default=8)
    ap.add_argument("--frames-per-traj", type=int, default=32)
    args = ap.parse_args()
    if not (args.profile or args.build):
        ap.error("nothing to do; pass --profile or --build <cache_root>")

    if args.build:
        from .adapters.libero import SUITES, encode_to_cache

        suites = tuple(args.suites.split(",")) if args.suites else SUITES
        t0 = time.perf_counter()
        trajs = encode_to_cache(
            default_encoder(), args.build, suites=suites, max_demos=args.max_demos,
        )
        cache = FeatureCache(args.build)
        print(f"encoded {len(trajs)} trajectories, {cache.total_bytes / 2**30:.1f} GiB, "
              f"{time.perf_counter() - t0:.0f}s -> {args.build}")
    if not args.profile:
        return

    with tempfile.TemporaryDirectory(dir=args.root) as tmp:
        rows = profile_formats(tmp, n_traj=args.n_traj, frames_per_traj=args.frames_per_traj)
    print(f"{'codec':<12}{'MiB/win':>10}{'write MiB/s':>10}{'read MiB/s':>10}{'int8 relL2':>14}")
    for r in rows:
        print(r.row(), end="")
        print(f"   {r.ms_per_window:.1f} ms/window (modelled)" if r.ms_per_window else "")


if __name__ == "__main__":  # pragma: no cover
    _main()
