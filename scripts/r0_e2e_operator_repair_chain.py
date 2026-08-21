#!/usr/bin/env python3
"""Fresh fixed-step R0 operator-repair train/evaluate orchestration.

This is deliberately not a convergence-gated or checkpoint-selecting workflow.
One immutable plan executes six resumable links toward the single predeclared
step-32,000 endpoint, consolidates that exact endpoint, evaluates seeds 0/1/2
in parallel (400 LIBERO episodes each), and always merges the exact 1,200
episodes.  Training health and the paired historical comparison are descriptive
outputs only.  Only execution/provenance/checkpoint/evaluation integrity can
stop a downstream ``afterok`` dependency.

Importing or dry-running this file never submits a Slurm job.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shlex
import struct
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import r0_e2e_formal_chain as common  # noqa: E402
from scripts import r0_e2e_operator_repair_train_entry as train_entry  # noqa: E402
from loom.data.adapters.libero import SUITES as LIBERO_STORAGE_SUITES  # noqa: E402
from loom.eval import DEFAULT_LIBERO_SUITES  # noqa: E402


FORMAT_VERSION = 1
KIND = "r0_e2e_operator_repair_fixed_endpoint_chain"
PROJECT = "loom-r0-operator-repair"
FIXED_STEP = 32_000
TRAIN_LINKS = 6
WORLD_SIZE = 16
SEEDS = (0, 1, 2)
TRAINING_LIBERO_SUITES = tuple(LIBERO_STORAGE_SUITES)
RUN_NAME = "r0a_operator_repair_fresh_s0_20260821"
CANONICAL_CONFIG = (ROOT / "configs/r0a_operator_repair.yaml").resolve()
CANONICAL_CONFIG_SHA256 = (
    "7d4586f4d1caa1c76ec3f97e7ed6ce9eb5d02b4776108ecd6b957bc12ace8143"
)
TRAIN_OVERRIDES: tuple[str, ...] = ()
EXPECTED_RESOLVED_CONFIG_HASH = "b47825f0cfba68dd"
EXPECTED_TRAIN_TAGS = (
    "operator-repair", "fixed-endpoint", "no-gate", "fresh", "r0", "dual-action",
)
EXPECTED_STAGE_TAGS = list(EXPECTED_TRAIN_TAGS)
LIBERO_EVAL_PYTHON = common.LIBERO_EVAL_PYTHON
DEFAULT_CACHE_ROOT = Path(
    "/lustre/fsw/portfolios/edgeai/users/chrislin/datasets/loom/libero_cache"
).resolve()
DEFAULT_RAW_DATA_ROOT_TEXT = (
    "/lustre/fsw/portfolios/edgeai/users/chrislin/datasets/loom/libero"
)
DEFAULT_RAW_DATA_ROOT = Path(
    DEFAULT_RAW_DATA_ROOT_TEXT
).resolve()
DEFAULT_HF_HOME = Path(
    "/lustre/fsw/portfolios/edgeai/users/chrislin/datasets/loom/hf_cache"
).resolve()
SIGLIP_COMMIT = "9fdffc58afc957d1a03a25b10dba0329ab15c2a3"
SIGLIP_MODEL = "google/siglip-so400m-patch14-384"
SIGLIP_EVAL_IMAGE_SIZE = 224
SIGLIP_EVAL_SNAPSHOT_SCHEME = "loom-siglip-eval-snapshot-v1"
SIGLIP_EVAL_SNAPSHOT_SHA256 = (
    "9f09ac5abb47908c28f0724d77dbd684a45f5be5b134d384fcdc1a611ffe48dc"
)
SIGLIP_EVAL_SNAPSHOT_BYTES = 3_515_150_375
SIGLIP_EVAL_SNAPSHOT_FILES = {
    "config.json": (576, "adc04928d8fd19a61822584fe0cf2e813e5ebac17f3e49fb1ea096860ae6457b"),
    "model.safetensors": (
        3_511_950_624,
        "ea2abad2b7f8a9c1aa5e49a244d5d57ffa71c56f720c94bc5d240ef4d6e1d94a",
    ),
    "preprocessor_config.json": (
        368, "f59da2f87c3cd079bd4f8f3037e81b277c60c498e279a8020331f67a5a3157e8",
    ),
    "special_tokens_map.json": (
        409, "2b6a1ff67a27e0df9ac0c7d93250fc0d87431c7b366b3d5669217104f9088a26",
    ),
    "spiece.model": (
        798_330,
        "1e5036bed065526c3c212dfbe288752391797c4bb1a284aa18c9a0b23fcaf8ec",
    ),
    "tokenizer.json": (
        2_399_357,
        "c6e405cb7c670d56636a9402c81023a55bc6c3c53d89cf02b92f5c5005bfe920",
    ),
    "tokenizer_config.json": (
        711, "d6423dae508cc3a129d22ea443841c111832a1a73125b8f25ea8736951698bcb",
    ),
}
RAW_RECEIPT_SCHEME = "loom-libero-raw-training-input-v1"
RAW_RECEIPT_SHA256 = (
    "7c02425ac1c8bbfc149ffb1b862d88f27f31e177968c6548e46d70ccbfd66fb2"
)
RAW_RECEIPT_FILES = 40
RAW_RECEIPT_DEMOS = 2_000
RAW_RECEIPT_ROWS = 338_575
RAW_RECEIPT_CANONICAL_BYTES = 9_480_100
CACHE_RECEIPT_SCHEME = "loom-libero-cache-content-v1"
CACHE_RECEIPT_SHA256 = (
    "ba754444526d56eb1314fd35ae2b5d8106282677298f300aee3efe98bd74d188"
)
CACHE_RECEIPT_ENTRIES = 2_000
CACHE_RECEIPT_FILES = 2_000
CACHE_RECEIPT_BYTES = 76_017_192_576
CACHE_HASH_WORKERS = 8
CACHE_MANIFEST_SHA256 = (
    "0ad6348be15d6baee4563f2b426d16b1b19fa87c74751b697ee8d7cd11144102"
)
CACHE_MANIFEST_BYTES = 1_234_624
BASELINE_PAIRING_ROWS_SHA256 = (
    "c2a208e0a9edd548321a9c454260881502e50ce4d0affaf4707d347cc09e9029"
)
BASELINE_PAIRING_ROWS_BYTES = 122_253
LIBERO_REPOSITORY = Path(
    "/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/"
    "users/chrislin/projects/loom-deps/LIBERO"
).resolve()
LIBERO_REPOSITORY_HEAD = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
LIBERO_REPOSITORY_CLEAN_SHA256 = (
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
)
LIBERO_TREE_SHA256 = "76662bff6faf421f2c11e509593fd242037a3d93"
LIBERO_BDDL_TREE_SHA256 = "9f8938926ccb741da4d69323d570552822a3101f"
LIBERO_INIT_TREE_SHA256 = "9515d661db9d7274054e84a5b41e0cda7d2f9ed9"
LIBERO_BDDL_FILES = 135
LIBERO_INIT_FILES = 250
LIBERO_EVAL_PYTHON_RESOLVED = Path(
    "/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/"
    "users/chrislin/envs/loom-libero/bin/python3.10"
)
LIBERO_EVAL_PYTHON_SHA256 = (
    "02d2cde925e03d34f4f7a18200b518809087819367aa67dc4e98bada0ba5197a"
)
LIBERO_EVAL_PYTHON_BYTES = 17_467_520
LIBERO_EVAL_PIP_FREEZE_SHA256 = (
    "094ab6d50e33f1503b5a2cecdce1e52071c967e8978b1a1db898764deb206344"
)
LIBERO_EVAL_PIP_FREEZE_LINES = 118
LIBERO_EVAL_PIP_FREEZE_BYTES = 2_304
LIBERO_EVAL_KEY_PACKAGES = {
    "gym": "0.25.2",
    "h5py": "3.16.0",
    "mujoco": "2.3.2",
    "numpy": "1.26.4",
    "robosuite": "1.4.1",
    "torch": "2.6.0+cu124",
    "transformers": "5.15.0",
    "triton": "3.2.0",
}
EVAL_MAX_PROCESS_ATTEMPTS = 3
WANDB_ATTEMPTS = 5
WANDB_RETRY_SECONDS = 15

NEW_SOURCE_FILES = (
    "scripts/r0_e2e_operator_repair_chain.py",
    "scripts/r0_e2e_operator_repair_train_entry.py",
    "scripts/r0_e2e_operator_repair_train.sbatch",
    "scripts/r0_e2e_operator_repair_consolidate.sbatch",
    "scripts/r0_e2e_operator_repair_eval_seed.sbatch",
    "scripts/r0_e2e_operator_repair_control.sbatch",
)
CONFIG_SOURCE_FILES = (
    "configs/base.yaml", "configs/r0a.yaml", "configs/r0a_dual_code.yaml",
    "configs/r0a_operator_repair.yaml",
)
# The common module is used only for deterministic JSON publication, checkpoint
# shard enumeration, exact LIBERO validation, and paired descriptive statistics.
# Its complete non-launcher dependency surface is nevertheless authenticated.
COMMON_SOURCE_FILES = tuple(
    name for name in common.ORCHESTRATION_SOURCE_FILES
    if not name.startswith("scripts/r0_e2e_formal_")
    and name != "scripts/direct_formal_convergence.py"
) + ("scripts/r0_e2e_formal_chain.py",)
SOURCE_FILES = tuple(sorted(set(
    NEW_SOURCE_FILES + CONFIG_SOURCE_FILES + COMMON_SOURCE_FILES
)))


class OperatorRepairError(RuntimeError):
    """Fail-closed orchestration/integrity error."""


class EvalAttemptMismatch(OperatorRepairError):
    """A prior eval attempt used a different authenticated environment."""


def sha256_file(path: str | Path) -> str:
    return common.sha256_file(path)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _pretty_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _identity(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise OperatorRepairError(f"required immutable file is absent/non-regular: {path}")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _source_closure() -> dict[str, Any]:
    files = {name: sha256_file(ROOT / name) for name in SOURCE_FILES}
    digest = hashlib.sha256()
    for name in sorted(files):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(files[name].encode())
        digest.update(b"\0")
    return {
        "scheme": "sha256(path-nul-sha256-nul)-v1",
        "sha256": digest.hexdigest(),
        "files": files,
    }


def _stable_stat(path: Path) -> tuple[int, int, int, int, int]:
    value = path.stat()
    return (
        value.st_dev, value.st_ino, value.st_size,
        value.st_mtime_ns, value.st_ctime_ns,
    )


def _require_unchanged_final_stats(
    captured: Mapping[Path, tuple[int, int, int, int, int]],
    *,
    label: str,
    stat_fn: Callable[[Path], tuple[int, int, int, int, int]] | None = None,
) -> None:
    read_stat = _stable_stat if stat_fn is None else stat_fn
    changed = [str(path) for path, before in captured.items() if read_stat(path) != before]
    if changed:
        raise OperatorRepairError(
            f"{label} changed before the final verification sweep: {changed[:3]}"
        )


def _raw_training_input_receipt(root: Path) -> dict[str, Any]:
    import h5py  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    from loom.data.adapters.libero import _instruction  # noqa: PLC0415

    root = root.resolve()
    if not root.is_dir() or root.is_symlink():
        raise OperatorRepairError("canonical raw LIBERO root is absent/non-directory")
    suites = tuple(sorted(TRAINING_LIBERO_SUITES))
    suite_dirs = {path.name: path for path in root.glob("libero_*") if path.is_dir()}
    if set(suite_dirs) != set(suites) or any(path.is_symlink() for path in suite_dirs.values()):
        raise OperatorRepairError("raw LIBERO suite directory set changed")
    paths = sorted(root.rglob("*.hdf5"), key=lambda path: path.relative_to(root).as_posix())
    if len(paths) != RAW_RECEIPT_FILES:
        raise OperatorRepairError(f"raw LIBERO HDF5 count changed: {len(paths)}")
    if any(
        not path.is_file() or path.is_symlink()
        or path.parent.name not in suites
        for path in paths
    ):
        raise OperatorRepairError("raw LIBERO HDF5 target is nonregular/outside suites")
    per_suite = {
        suite: {"files": 0, "demos": 0, "rows": 0, "canonical_action_bytes": 0}
        for suite in suites
    }
    digest = hashlib.sha256()
    digest.update((RAW_RECEIPT_SCHEME + "\0").encode())
    digest.update(struct.pack("<Q", len(paths)))
    demos = rows = canonical_bytes = 0
    original_stats = {path: _stable_stat(path) for path in paths}
    for path in paths:
        relative = path.relative_to(root).as_posix()
        suite = relative.split("/", 1)[0]
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with h5py.File(path, "r") as handle:
            data = handle["data"]
            instruction = _instruction(data.attrs)
            digest.update(instruction.encode("utf-8"))
            digest.update(b"\0")
            demo_keys = sorted(data.keys(), key=lambda value: int(value.split("_")[1]))
            digest.update(struct.pack("<Q", len(demo_keys)))
            per_suite[suite]["files"] += 1
            for demo_key in demo_keys:
                actions = np.asarray(
                    data[demo_key]["actions"], dtype=np.dtype("<f4"), order="C",
                )
                if not actions.flags.c_contiguous:
                    actions = np.ascontiguousarray(actions, dtype=np.dtype("<f4"))
                action_bytes = actions.tobytes(order="C")
                digest.update(str(demo_key).encode("utf-8"))
                digest.update(b"\0")
                digest.update(struct.pack("<Q", actions.ndim))
                for dimension in actions.shape:
                    digest.update(struct.pack("<Q", int(dimension)))
                digest.update(struct.pack("<Q", len(action_bytes)))
                digest.update(action_bytes)
                demo_rows = int(actions.shape[0])
                demos += 1
                rows += demo_rows
                canonical_bytes += len(action_bytes)
                per_suite[suite]["demos"] += 1
                per_suite[suite]["rows"] += demo_rows
                per_suite[suite]["canonical_action_bytes"] += len(action_bytes)
        if _stable_stat(path) != original_stats[path]:
            raise OperatorRepairError(f"raw LIBERO HDF5 changed while hashing: {relative}")
    paths_after = sorted(
        path.relative_to(root).as_posix() for path in root.rglob("*.hdf5")
    )
    if paths_after != [path.relative_to(root).as_posix() for path in paths]:
        raise OperatorRepairError("raw LIBERO HDF5 set changed while hashing")
    if any(not path.is_file() or path.is_symlink() for path in paths):
        raise OperatorRepairError("raw LIBERO HDF5 type changed while hashing")
    _require_unchanged_final_stats(original_stats, label="raw LIBERO HDF5")
    receipt = {
        "scheme": RAW_RECEIPT_SCHEME,
        "configured_root": DEFAULT_RAW_DATA_ROOT_TEXT,
        "resolved_root": str(root),
        "sha256": digest.hexdigest(),
        "files": len(paths), "demos": demos, "rows": rows,
        "canonical_action_bytes": canonical_bytes,
        "per_suite": per_suite,
        "semantic_inputs": (
            "relative_path_instruction_numeric_demo_key_shape_and_canonical_f4_actions"
        ),
    }
    if not (
        receipt["sha256"] == RAW_RECEIPT_SHA256
        and receipt["files"] == RAW_RECEIPT_FILES
        and receipt["demos"] == RAW_RECEIPT_DEMOS
        and receipt["rows"] == RAW_RECEIPT_ROWS
        and receipt["canonical_action_bytes"] == RAW_RECEIPT_CANONICAL_BYTES
        and all(
            row["files"] == 10 and row["demos"] == 500
            for row in per_suite.values()
        )
    ):
        raise OperatorRepairError(f"raw LIBERO semantic receipt changed: {receipt}")
    return receipt


def _cache_content_receipt(root: Path) -> dict[str, Any]:
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    root = root.resolve()
    manifest_path = root / "manifest.json"
    feats = root / "feats"
    if (
        not root.is_dir() or root.is_symlink()
        or not feats.is_dir() or feats.is_symlink()
        or not manifest_path.is_file() or manifest_path.is_symlink()
    ):
        raise OperatorRepairError("canonical feature-cache structure changed")
    manifest_before = _stable_stat(manifest_path)
    try:
        document = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise OperatorRepairError("feature-cache manifest is unreadable") from exc
    entries = document.get("entries") if isinstance(document, Mapping) else None
    if not isinstance(entries, Mapping) or len(entries) != CACHE_RECEIPT_ENTRIES:
        raise OperatorRepairError("feature-cache manifest entry count changed")
    rows: list[tuple[str, str, Path, int, str]] = []
    referenced: set[str] = set()
    per_suite = {suite: 0 for suite in sorted(TRAINING_LIBERO_SUITES)}
    for entry_id in sorted(entries):
        row = entries[entry_id]
        if not isinstance(row, Mapping) or not isinstance(row.get("file"), str):
            raise OperatorRepairError(f"cache manifest entry is malformed: {entry_id}")
        relative = row["file"]
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or relative_path.as_posix() != relative
            or relative_path.parts[:1] != ("feats",)
            or ".." in relative_path.parts
            or relative in referenced
        ):
            raise OperatorRepairError(f"cache manifest file path is invalid: {relative!r}")
        referenced.add(relative)
        path = root / relative_path
        if not path.is_file() or path.is_symlink():
            raise OperatorRepairError(f"cache binary is absent/nonregular: {relative}")
        size = path.stat().st_size
        if int(row.get("nbytes", -1)) != size:
            raise OperatorRepairError(f"cache binary size differs from manifest: {relative}")
        suite = str(entry_id).split("/", 1)[0]
        if suite not in per_suite:
            raise OperatorRepairError(f"cache entry suite changed: {entry_id}")
        per_suite[suite] += 1
        rows.append((str(entry_id), relative, path, size, suite))
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*.bin")
        if path.is_file()
    }
    if actual != referenced or len(referenced) != CACHE_RECEIPT_FILES:
        raise OperatorRepairError(
            "cache binary set has missing/duplicate/unreferenced files"
        )
    initial_stats = {path: _stable_stat(path) for _, _, path, _, _ in rows}

    def _hash_row(row: tuple[str, str, Path, int, str]) -> tuple[str, str]:
        entry_id, _, path, _, _ = row
        digest = sha256_file(path).lower()
        if _stable_stat(path) != initial_stats[path]:
            raise OperatorRepairError(f"cache binary changed while hashing: {entry_id}")
        return entry_id, digest

    with ThreadPoolExecutor(max_workers=CACHE_HASH_WORKERS) as pool:
        hashes = dict(pool.map(_hash_row, rows))
    if _stable_stat(manifest_path) != manifest_before:
        raise OperatorRepairError("cache manifest changed while hashing binaries")
    actual_after = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*.bin")
        if path.is_file()
    }
    if actual_after != referenced:
        raise OperatorRepairError("cache binary set changed while hashing")
    if any(not path.is_file() or path.is_symlink() for _, _, path, _, _ in rows):
        raise OperatorRepairError("cache binary type changed while hashing")
    _require_unchanged_final_stats(initial_stats, label="cache binary")
    if _stable_stat(manifest_path) != manifest_before:
        raise OperatorRepairError("cache manifest changed before final verification")
    digest = hashlib.sha256()
    digest.update((CACHE_RECEIPT_SCHEME + "\0").encode())
    digest.update(struct.pack("<Q", len(rows)))
    total_bytes = 0
    for entry_id, relative, _, size, _ in rows:
        digest.update(entry_id.encode("utf-8")); digest.update(b"\0")
        digest.update(relative.encode("utf-8")); digest.update(b"\0")
        digest.update(str(size).encode("ascii")); digest.update(b"\0")
        digest.update(hashes[entry_id].encode("ascii")); digest.update(b"\0")
        total_bytes += size
    manifest = _identity(manifest_path)
    receipt = {
        "scheme": CACHE_RECEIPT_SCHEME,
        "root": str(root), "sha256": digest.hexdigest(),
        "entries": len(rows), "files": len(referenced), "bytes": total_bytes,
        "duplicates": 0, "missing": 0, "unreferenced": 0,
        "per_suite_entries": per_suite,
        "hash_workers": CACHE_HASH_WORKERS,
        "manifest": manifest,
    }
    if not (
        receipt["sha256"] == CACHE_RECEIPT_SHA256
        and receipt["entries"] == CACHE_RECEIPT_ENTRIES
        and receipt["files"] == CACHE_RECEIPT_FILES
        and receipt["bytes"] == CACHE_RECEIPT_BYTES
        and manifest["sha256"] == CACHE_MANIFEST_SHA256
        and manifest["bytes"] == CACHE_MANIFEST_BYTES
        and all(value == 500 for value in per_suite.values())
    ):
        raise OperatorRepairError(f"feature-cache content receipt changed: {receipt}")
    return receipt


def _expected_asset_receipt() -> dict[str, Any]:
    cache_manifest = DEFAULT_CACHE_ROOT / "manifest.json"
    model_root = (
        DEFAULT_HF_HOME / "hub" / "models--google--siglip-so400m-patch14-384"
    )
    ref = model_root / "refs" / "main"
    tree = model_root / "trees" / f"{SIGLIP_COMMIT}.json"
    if ref.read_text().strip() != SIGLIP_COMMIT:
        raise OperatorRepairError("frozen SigLIP ref does not name the expected commit")
    return {
        "raw_training_input": {
            "scheme": RAW_RECEIPT_SCHEME,
            "configured_root": DEFAULT_RAW_DATA_ROOT_TEXT,
            "resolved_root": str(DEFAULT_RAW_DATA_ROOT),
            "sha256": RAW_RECEIPT_SHA256,
            "files": RAW_RECEIPT_FILES,
            "demos": RAW_RECEIPT_DEMOS,
            "rows": RAW_RECEIPT_ROWS,
            "canonical_action_bytes": RAW_RECEIPT_CANONICAL_BYTES,
            "per_suite": {
                "libero_10": {
                    "files": 10, "demos": 500, "rows": 138_090,
                    "canonical_action_bytes": 3_866_520,
                },
                "libero_goal": {
                    "files": 10, "demos": 500, "rows": 63_728,
                    "canonical_action_bytes": 1_784_384,
                },
                "libero_object": {
                    "files": 10, "demos": 500, "rows": 74_507,
                    "canonical_action_bytes": 2_086_196,
                },
                "libero_spatial": {
                    "files": 10, "demos": 500, "rows": 62_250,
                    "canonical_action_bytes": 1_743_000,
                },
            },
            "semantic_inputs": (
                "relative_path_instruction_numeric_demo_key_shape_and_canonical_f4_actions"
            ),
        },
        "cache_root": str(DEFAULT_CACHE_ROOT),
        "cache_content": {
            "scheme": CACHE_RECEIPT_SCHEME,
            "root": str(DEFAULT_CACHE_ROOT),
            "sha256": CACHE_RECEIPT_SHA256,
            "entries": CACHE_RECEIPT_ENTRIES,
            "files": CACHE_RECEIPT_FILES,
            "bytes": CACHE_RECEIPT_BYTES,
            "duplicates": 0, "missing": 0, "unreferenced": 0,
            "per_suite_entries": {
                suite: 500 for suite in sorted(TRAINING_LIBERO_SUITES)
            },
            "hash_workers": CACHE_HASH_WORKERS,
            "manifest": _identity(cache_manifest),
        },
        "frozen_tower": {
            "model": SIGLIP_MODEL,
            "hf_home": str(DEFAULT_HF_HOME),
            "commit": SIGLIP_COMMIT,
            "ref": _identity(ref),
            "tree": _identity(tree),
            "training_gradient": False,
            "cached_feature_manifest_is_training_input_authority": True,
        },
    }


def _asset_receipt() -> dict[str, Any]:
    cache_root = Path(os.environ.get("LOOM_CACHE_DIR", str(DEFAULT_CACHE_ROOT))).resolve()
    raw_root = Path(
        os.environ.get("LOOM_DATA_ROOT", DEFAULT_RAW_DATA_ROOT_TEXT)
    ).resolve()
    hf_home = Path(os.environ.get("HF_HOME", str(DEFAULT_HF_HOME))).resolve()
    if (
        cache_root != DEFAULT_CACHE_ROOT
        or raw_root != DEFAULT_RAW_DATA_ROOT
        or hf_home != DEFAULT_HF_HOME
    ):
        raise OperatorRepairError(
            "operator-repair assets must use canonical raw/cache/HF paths"
        )
    model_root = hf_home / "hub" / "models--google--siglip-so400m-patch14-384"
    ref = model_root / "refs" / "main"
    tree = model_root / "trees" / f"{SIGLIP_COMMIT}.json"
    if ref.read_text().strip() != SIGLIP_COMMIT:
        raise OperatorRepairError("frozen SigLIP ref does not name the expected commit")
    receipt = {
        "raw_training_input": _raw_training_input_receipt(raw_root),
        "cache_root": str(cache_root),
        "cache_content": _cache_content_receipt(cache_root),
        "frozen_tower": {
            "model": SIGLIP_MODEL,
            "hf_home": str(hf_home),
            "commit": SIGLIP_COMMIT,
            "ref": _identity(ref),
            "tree": _identity(tree),
            "training_gradient": False,
            "cached_feature_manifest_is_training_input_authority": True,
        },
    }
    expected = _expected_asset_receipt()
    if receipt != expected:
        raise OperatorRepairError("live training asset receipt differs from frozen contract")
    return receipt


def _training_argv(run_dir: Path, *, include_link: bool) -> list[str]:
    argv = ["--config", str(CANONICAL_CONFIG)]
    if include_link:
        argv.extend((
            "--run_dir", str(run_dir),
            "--stop_at", str(FIXED_STEP),
            "--budget_s", str(4 * 3600 - 600),
            "--safety_s", "420",
        ))
    for value in TRAIN_OVERRIDES:
        argv.extend(("--set", value))
    return argv


def _resolved_training_config(run_dir: Path) -> tuple[dict[str, Any], str]:
    from loom.train.loop import config_hash, load_config, parse_args  # noqa: PLC0415

    cfg = load_config(parse_args(_training_argv(run_dir, include_link=True)))
    experiment = {key: copy.deepcopy(value) for key, value in cfg.items() if key != "link"}
    return experiment, config_hash(cfg)


def _validate_training_config(cfg: Mapping[str, Any], digest: str) -> None:
    run = cfg.get("run", {})
    losses = cfg.get("losses", {})
    act = losses.get("act", {})
    forbidden = {"schedule_horizon", "max_updates", "direct_formal"}
    if forbidden.intersection(run) or "direct_formal" in cfg:
        raise OperatorRepairError("operator-repair config must not enable direct-formal gates")
    expected = {
        "digest": digest == EXPECTED_RESOLVED_CONFIG_HASH,
        "name": run.get("name") == RUN_NAME,
        "project": run.get("project") == PROJECT,
        "steps": run.get("steps") == FIXED_STEP,
        "seed": run.get("seed") == 0,
        "ckpt_every": run.get("ckpt_every") == 500,
        "log_every": run.get("log_every") == train_entry.LOG_EVERY_UPDATES,
        "keep_last": run.get("keep_last") == 4,
        "fresh": run.get("fresh_start_required") is True,
        "metrics_reconciliation": run.get("reconcile_metrics_on_resume") is True,
        "online": run.get("require_online_wandb") is True,
        "real": cfg.get("model", {}).get("use_stubs") is False,
        "dual": act.get("decode_from") == "dual_q_action_proposal",
        "act": act.get("enabled") is True and float(act.get("weight")) == 1.0,
        "seed_schedule": cfg.get("slurm", {}).get("n_links") == TRAIN_LINKS,
        "raw_data_root": Path(str(cfg.get("data", {}).get("data_root", ""))).resolve()
        == DEFAULT_RAW_DATA_ROOT,
        "no_skipped_updates": float(cfg.get("optim", {}).get("spike_mult")) == 0.0,
        "fixed_boundary": run.get("boundary_policy") == "fixed_max_updates",
        "method_receipt": cfg.get("method_receipt") == {
            "kind": "loom_r0a_operator_repair_v1",
            "fixed_endpoint_update": FIXED_STEP,
            "evaluation_is_unconditional": True,
            "evaluation_episodes": 1_200,
            "evaluation_seeds": [0, 1, 2],
            "checkpoint_selection": "fixed_update_only",
            "health_thresholds_control_execution": False,
        },
        "modules": cfg.get("train_modules") == [
            "estimator", "bank", "q_delta", "q_action", "decoder", "proposal",
        ],
    }
    failed = [key for key, passed in expected.items() if not passed]
    if failed:
        raise OperatorRepairError(f"fixed training config mismatch: {failed}")


def _require_clean_absolute(path: Path, *, field: str) -> Path:
    if not path.is_absolute():
        raise OperatorRepairError(f"{field} must be absolute")
    if any(token in str(path) for token in (",", "\n", "\r")):
        raise OperatorRepairError(f"{field} cannot contain comma/newline")
    return path.resolve()


def _require_isolated(run_dir: Path, control_dir: Path, artifact_root: Path) -> None:
    values = (run_dir, control_dir, artifact_root)
    for index, left in enumerate(values):
        if ROOT / "runs" not in (left, *left.parents):
            raise OperatorRepairError(f"operator-repair output must remain below ROOT/runs: {left}")
        for right in values[index + 1:]:
            if left == right or left in right.parents or right in left.parents:
                raise OperatorRepairError(
                    f"run/control/artifact roots must be pairwise non-nested: {left}, {right}"
                )


def _reject_existing_symlink_components(path: Path) -> None:
    current = path
    stop = (ROOT / "runs").resolve()
    while True:
        if current.is_symlink():
            raise OperatorRepairError(f"output path component is a symlink: {current}")
        if current == stop:
            return
        if current == current.parent:
            raise OperatorRepairError(f"output path escaped ROOT/runs: {path}")
        current = current.parent


def _expected_paths(control_dir: Path, artifact_root: Path) -> dict[str, Any]:
    return {
        "training_asset_verification_dir": str(
            control_dir / "training_asset_verification"
        ),
        "training_asset_failure": str(control_dir / "TRAINING_ASSET_FAILURE.json"),
        "fixed_endpoint": str(control_dir / "fixed_endpoint_32000.json"),
        "jobs": str(control_dir / "jobs.json"),
        "checkpoint": str(artifact_root / "checkpoint" / "ckpt.pt"),
        "checkpoint_report": str(control_dir / "checkpoint_verification.json"),
        "checkpoint_receipt": str(control_dir / "checkpoint_receipt.json"),
        "merged_results": str(artifact_root / "eval" / "merged" / "results.json"),
        "merged_table": str(artifact_root / "eval" / "merged" / "table.md"),
        "merged_receipt": str(control_dir / "merged_eval_receipt.json"),
        "eval": {
            str(seed): {
                "out_dir": str(artifact_root / "eval" / f"seed_{seed}"),
                "receipt": str(control_dir / f"eval_seed_{seed}_receipt.json"),
            }
            for seed in SEEDS
        },
    }


def _expected_baseline_receipt() -> dict[str, Any]:
    root = common.CANONICAL_BASELINE_ROOT.resolve()
    sizes = {0: 1_655_458, 1: 1_634_609, 2: 1_647_375}
    return {
        "kind": "r0a_deploy_seeded1200_v2_exact_baseline",
        "root": str(root),
        "files": {
            str(seed): {
                "path": str(root / f"seed{seed}" / "results.json"),
                "sha256": common.BASELINE_RESULT_SHA256[seed],
                "bytes": sizes[seed],
                "episodes": 400,
                "successes": common.BASELINE_SUCCESS_PER_SEED,
            }
            for seed in SEEDS
        },
        "episodes": common.EXPECTED_EPISODES_TOTAL,
        "successes": common.BASELINE_SUCCESS_TOTAL,
        "success_rate_percent": (
            100.0 * common.BASELINE_SUCCESS_TOTAL / common.EXPECTED_EPISODES_TOTAL
        ),
        "checkpoint_step": common.BASELINE_CHECKPOINT_STEP,
    }


def _baseline_pairing_snapshot(
    rows: Mapping[tuple[str, str, int, int, int], Mapping[str, Any]],
) -> dict[str, Any]:
    compact = []
    for key in sorted(rows):
        row = rows[key]
        compact.append({
            "key": list(key),
            "env_seed": row.get("env_seed"),
            "policy_seed": (row.get("extra") or {}).get("policy_seed"),
            "success": row.get("success"),
        })
    encoded = _canonical_json(compact).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    if not (
        len(compact) == common.EXPECTED_EPISODES_TOTAL
        and sum(int(row["success"]) for row in compact)
        == common.BASELINE_SUCCESS_TOTAL
        and len(encoded) == BASELINE_PAIRING_ROWS_BYTES
        and digest == BASELINE_PAIRING_ROWS_SHA256
    ):
        raise OperatorRepairError("canonical baseline pairing snapshot changed")
    return {
        "kind": "r0a_deploy_seeded1200_compact_pairing_snapshot_v1",
        "serialization": "canonical_json_sort_keys_compact_utf8_v1",
        "rows": compact,
        "rows_sha256": digest,
        "encoded_bytes": len(encoded),
        "episodes": len(compact),
        "successes": sum(int(row["success"]) for row in compact),
        "source_file_sha256": {
            str(seed): common.BASELINE_RESULT_SHA256[seed] for seed in SEEDS
        },
    }


def _baseline_contract() -> dict[str, Any]:
    """Authenticate live baseline once, then freeze all pairing inputs in-plan."""
    try:
        baseline = common._authenticate_baseline(common.CANONICAL_BASELINE_ROOT)
        rows = common._baseline_rows({"baseline_comparison": {"baseline": baseline}})
    except common.ChainError as exc:
        raise OperatorRepairError(str(exc)) from exc
    if baseline != _expected_baseline_receipt():
        raise OperatorRepairError("canonical baseline receipt differs from frozen identity")
    return {
        "role": "descriptive_only_never_controls_training_or_evaluation",
        "baseline": baseline,
        "pairing_snapshot": _baseline_pairing_snapshot(rows),
        "live_baseline_files_required_after_plan_creation": False,
        "pairing_key": ["bench", "suite", "task_id", "episode", "seed"],
        "pairing_requires_equal_env_seed_and_policy_seed": True,
        "bootstrap": {
            "kind": "fixed_suite_stratified_task_resample_matrix_v1",
            "samples": common.BOOTSTRAP_SAMPLES,
            "seed": common.BOOTSTRAP_SEED,
            "confidence": common.BOOTSTRAP_CONFIDENCE,
            "lower_quantile": 0.025,
            "upper_quantile": 0.975,
            "lower_interpolation": "lower",
            "upper_interpolation": "higher",
            "matrix_sha256": common.BOOTSTRAP_MATRIX_SHA256,
        },
    }


def _validate_frozen_baseline_contract(
    value: Any,
) -> dict[tuple[str, str, int, int, int], dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise OperatorRepairError("descriptive baseline contract is not an object")
    expected_without_snapshot = {
        "role": "descriptive_only_never_controls_training_or_evaluation",
        "baseline": _expected_baseline_receipt(),
        "live_baseline_files_required_after_plan_creation": False,
        "pairing_key": ["bench", "suite", "task_id", "episode", "seed"],
        "pairing_requires_equal_env_seed_and_policy_seed": True,
        "bootstrap": {
            "kind": "fixed_suite_stratified_task_resample_matrix_v1",
            "samples": common.BOOTSTRAP_SAMPLES,
            "seed": common.BOOTSTRAP_SEED,
            "confidence": common.BOOTSTRAP_CONFIDENCE,
            "lower_quantile": 0.025,
            "upper_quantile": 0.975,
            "lower_interpolation": "lower",
            "upper_interpolation": "higher",
            "matrix_sha256": common.BOOTSTRAP_MATRIX_SHA256,
        },
    }
    without_snapshot = {
        key: copy.deepcopy(item) for key, item in value.items()
        if key != "pairing_snapshot"
    }
    if without_snapshot != expected_without_snapshot:
        raise OperatorRepairError("descriptive baseline metadata contract changed")
    snapshot = value.get("pairing_snapshot")
    if not isinstance(snapshot, Mapping):
        raise OperatorRepairError("descriptive baseline snapshot is absent")
    rows = snapshot.get("rows")
    if not isinstance(rows, list):
        raise OperatorRepairError("descriptive baseline snapshot rows are absent")
    expected_snapshot_metadata = {
        "kind": "r0a_deploy_seeded1200_compact_pairing_snapshot_v1",
        "serialization": "canonical_json_sort_keys_compact_utf8_v1",
        "rows_sha256": BASELINE_PAIRING_ROWS_SHA256,
        "encoded_bytes": BASELINE_PAIRING_ROWS_BYTES,
        "episodes": common.EXPECTED_EPISODES_TOTAL,
        "successes": common.BASELINE_SUCCESS_TOTAL,
        "source_file_sha256": {
            str(seed): common.BASELINE_RESULT_SHA256[seed] for seed in SEEDS
        },
    }
    if {key: item for key, item in snapshot.items() if key != "rows"} != (
        expected_snapshot_metadata
    ):
        raise OperatorRepairError("descriptive baseline snapshot metadata changed")
    encoded = _canonical_json(rows).encode()
    if (
        len(encoded) != BASELINE_PAIRING_ROWS_BYTES
        or hashlib.sha256(encoded).hexdigest() != BASELINE_PAIRING_ROWS_SHA256
    ):
        raise OperatorRepairError("descriptive baseline snapshot bytes changed")
    parsed: dict[tuple[str, str, int, int, int], dict[str, Any]] = {}
    for index, item in enumerate(rows):
        if not isinstance(item, Mapping) or set(item) != {
            "key", "env_seed", "policy_seed", "success",
        }:
            raise OperatorRepairError(f"baseline snapshot row {index} is malformed")
        key_value = item["key"]
        if not isinstance(key_value, list) or len(key_value) != 5:
            raise OperatorRepairError(f"baseline snapshot row {index} key is malformed")
        key = tuple(key_value)
        if not (
            isinstance(key[0], str) and isinstance(key[1], str)
            and all(isinstance(key[position], int) for position in (2, 3, 4))
            and isinstance(item["env_seed"], int)
            and isinstance(item["policy_seed"], int)
            and isinstance(item["success"], bool)
            and key not in parsed
        ):
            raise OperatorRepairError(f"baseline snapshot row {index} types changed")
        parsed[key] = {
            "env_seed": item["env_seed"],
            "success": item["success"],
            "extra": {"policy_seed": item["policy_seed"]},
        }
    if not (
        len(parsed) == common.EXPECTED_EPISODES_TOTAL
        and sum(int(row["success"]) for row in parsed.values())
        == common.BASELINE_SUCCESS_TOTAL
    ):
        raise OperatorRepairError("descriptive baseline snapshot closure changed")
    return parsed


def _run_identity_command(command: Sequence[str], *, label: str) -> bytes:
    try:
        completed = subprocess.run(
            list(command), check=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise OperatorRepairError(f"failed to authenticate {label}") from exc
    return completed.stdout


def _expected_siglip_eval_snapshot_receipt() -> dict[str, Any]:
    model_root = (
        DEFAULT_HF_HOME / "hub" / "models--google--siglip-so400m-patch14-384"
    )
    snapshot = model_root / "snapshots" / SIGLIP_COMMIT
    return {
        "scheme": SIGLIP_EVAL_SNAPSHOT_SCHEME,
        "model": SIGLIP_MODEL,
        "image_size": SIGLIP_EVAL_IMAGE_SIZE,
        "cache_n_patches": 256,
        "snapshot": str(snapshot),
        "commit": SIGLIP_COMMIT,
        "sha256": SIGLIP_EVAL_SNAPSHOT_SHA256,
        "files": len(SIGLIP_EVAL_SNAPSHOT_FILES),
        "bytes": SIGLIP_EVAL_SNAPSHOT_BYTES,
        "rows": {
            name: {"bytes": size, "sha256": digest}
            for name, (size, digest) in sorted(SIGLIP_EVAL_SNAPSHOT_FILES.items())
        },
        "snapshot_entries_are_symlinks_to_regular_blobs": True,
        "final_target_stat_sweep": True,
    }


def _siglip_eval_snapshot_receipt() -> dict[str, Any]:
    model_root = (
        DEFAULT_HF_HOME / "hub" / "models--google--siglip-so400m-patch14-384"
    ).resolve()
    snapshot = model_root / "snapshots" / SIGLIP_COMMIT
    if not snapshot.is_dir() or snapshot.is_symlink():
        raise OperatorRepairError("SigLIP eval snapshot is absent/non-directory")
    entries = sorted(snapshot.iterdir(), key=lambda path: path.name)
    if [path.name for path in entries] != sorted(SIGLIP_EVAL_SNAPSHOT_FILES):
        raise OperatorRepairError("SigLIP eval snapshot file set changed")
    rows: dict[str, dict[str, Any]] = {}
    targets: dict[Path, tuple[int, int, int, int, int]] = {}
    links: dict[Path, str] = {}
    digest = hashlib.sha256()
    digest.update((SIGLIP_EVAL_SNAPSHOT_SCHEME + "\0").encode())
    digest.update(struct.pack("<Q", len(entries)))
    total = 0
    blobs = model_root / "blobs"
    for path in entries:
        if not path.is_symlink():
            raise OperatorRepairError("SigLIP snapshot entry is not an exact symlink")
        links[path] = os.readlink(path)
        target = path.resolve(strict=True)
        if target.parent != blobs or not target.is_file() or target.is_symlink():
            raise OperatorRepairError("SigLIP snapshot target is not a regular blob")
        targets[target] = _stable_stat(target)
        size = target.stat().st_size
        file_sha = sha256_file(target)
        digest.update(path.name.encode()); digest.update(b"\0")
        digest.update(str(size).encode("ascii")); digest.update(b"\0")
        digest.update(file_sha.encode("ascii")); digest.update(b"\0")
        rows[path.name] = {"bytes": size, "sha256": file_sha}
        total += size
    if [path.name for path in sorted(snapshot.iterdir(), key=lambda item: item.name)] != (
        sorted(SIGLIP_EVAL_SNAPSHOT_FILES)
    ):
        raise OperatorRepairError("SigLIP eval snapshot changed during hashing")
    if any(os.readlink(path) != before for path, before in links.items()):
        raise OperatorRepairError("SigLIP eval snapshot symlink target changed")
    _require_unchanged_final_stats(targets, label="SigLIP eval blob")
    receipt = {
        **_expected_siglip_eval_snapshot_receipt(),
        "sha256": digest.hexdigest(), "files": len(entries),
        "bytes": total, "rows": rows,
    }
    if receipt != _expected_siglip_eval_snapshot_receipt():
        raise OperatorRepairError("SigLIP eval snapshot content receipt changed")
    return receipt


def _expected_eval_environment_receipt() -> dict[str, Any]:
    bddl = LIBERO_REPOSITORY / "libero/libero/bddl_files"
    init = LIBERO_REPOSITORY / "libero/libero/init_files"
    editable_root = LIBERO_REPOSITORY / "libero"
    return {
        "kind": "loom-libero-fixed-evaluation-environment-v1",
        "python": {
            "configured": str(LIBERO_EVAL_PYTHON),
            "resolved": str(LIBERO_EVAL_PYTHON_RESOLVED),
            "sha256": LIBERO_EVAL_PYTHON_SHA256,
            "bytes": LIBERO_EVAL_PYTHON_BYTES,
            "pip_freeze": {
                "normalization": "sorted_nonempty_lines_lf_terminated_utf8_v1",
                "sha256": LIBERO_EVAL_PIP_FREEZE_SHA256,
                "lines": LIBERO_EVAL_PIP_FREEZE_LINES,
                "bytes": LIBERO_EVAL_PIP_FREEZE_BYTES,
                "key_packages": dict(LIBERO_EVAL_KEY_PACKAGES),
                "editable_libero": (
                    "-e git+https://github.com/Lifelong-Robot-Learning/LIBERO.git@"
                    f"{LIBERO_REPOSITORY_HEAD}#egg=libero"
                ),
            },
        },
        "libero_repository": {
            "path": str(LIBERO_REPOSITORY),
            "head": LIBERO_REPOSITORY_HEAD,
            "clean_status_sha256": LIBERO_REPOSITORY_CLEAN_SHA256,
            "clean_status_bytes": 0,
            "libero_tree": LIBERO_TREE_SHA256,
            "bddl_tree": LIBERO_BDDL_TREE_SHA256,
            "init_tree": LIBERO_INIT_TREE_SHA256,
            "editable_import_root": str(editable_root),
            "bddl_dir": str(bddl),
            "bddl_files": LIBERO_BDDL_FILES,
            "init_states_dir": str(init),
            "init_state_files": LIBERO_INIT_FILES,
        },
        "siglip_snapshot": _expected_siglip_eval_snapshot_receipt(),
        "child_environment": {
            "LOOM_DATA_ROOT": DEFAULT_RAW_DATA_ROOT_TEXT,
            "LOOM_LIBERO_BDDL_DIR": str(bddl),
            "LOOM_LIBERO_INIT_STATES_DIR": str(init),
            "LOOM_LIBERO_IMAGE_SIZE": "256",
            "LOOM_LIBERO_IMAGE_CONVENTION": "opengl",
            "LOOM_LIBERO_PYTHON": str(LIBERO_EVAL_PYTHON),
            "LOOM_TOWER_MODEL": SIGLIP_MODEL,
            "LOOM_TOWER_IMAGE_SIZE": str(SIGLIP_EVAL_IMAGE_SIZE),
            "HF_HOME": str(DEFAULT_HF_HOME),
            "HF_HUB_CACHE": str(DEFAULT_HF_HOME / "hub"),
            "HUGGINGFACE_HUB_CACHE": str(DEFAULT_HF_HOME / "hub"),
            "TRANSFORMERS_CACHE": str(DEFAULT_HF_HOME / "hub"),
        },
    }


def _eval_environment_receipt() -> dict[str, Any]:
    if (
        LIBERO_EVAL_PYTHON.resolve() != LIBERO_EVAL_PYTHON_RESOLVED
        or not LIBERO_EVAL_PYTHON_RESOLVED.is_file()
        or LIBERO_EVAL_PYTHON_RESOLVED.is_symlink()
    ):
        raise OperatorRepairError("pinned LIBERO evaluation Python identity changed")
    python_identity = _identity(LIBERO_EVAL_PYTHON_RESOLVED)
    if (
        python_identity["sha256"] != LIBERO_EVAL_PYTHON_SHA256
        or python_identity["bytes"] != LIBERO_EVAL_PYTHON_BYTES
    ):
        raise OperatorRepairError("pinned LIBERO evaluation Python bytes changed")
    if not LIBERO_REPOSITORY.is_dir() or LIBERO_REPOSITORY.is_symlink():
        raise OperatorRepairError("pinned LIBERO repository is absent/non-directory")
    git = ["git", "-C", str(LIBERO_REPOSITORY)]
    head = _run_identity_command(git + ["rev-parse", "HEAD"], label="LIBERO HEAD")
    status = _run_identity_command(
        git + ["status", "--porcelain=v1", "--untracked-files=all"],
        label="LIBERO worktree",
    )
    trees = {
        "libero_tree": "HEAD:libero/libero",
        "bddl_tree": "HEAD:libero/libero/bddl_files",
        "init_tree": "HEAD:libero/libero/init_files",
    }
    tree_values = {
        key: _run_identity_command(
            git + ["rev-parse", spec], label=f"LIBERO {key}",
        ).decode().strip()
        for key, spec in trees.items()
    }
    bddl = LIBERO_REPOSITORY / "libero/libero/bddl_files"
    init = LIBERO_REPOSITORY / "libero/libero/init_files"
    bddl_files = sorted(path for path in bddl.rglob("*") if path.is_file())
    init_files = sorted(path for path in init.rglob("*") if path.is_file())
    if any(path.is_symlink() for path in (*bddl_files, *init_files)):
        raise OperatorRepairError("LIBERO BDDL/init inputs contain symlinks")
    freeze_stdout = _run_identity_command(
        [str(LIBERO_EVAL_PYTHON), "-m", "pip", "freeze", "--all"],
        label="LIBERO Python package closure",
    )
    freeze_lines = sorted(
        line.strip() for line in freeze_stdout.decode("utf-8").splitlines()
        if line.strip()
    )
    freeze = ("\n".join(freeze_lines) + "\n").encode("utf-8")
    package_versions = {}
    for line in freeze_lines:
        if "==" in line and not line.startswith("#"):
            name, version = line.split("==", 1)
            package_versions[name.lower()] = version
    key_packages = {
        name: package_versions.get(name) for name in LIBERO_EVAL_KEY_PACKAGES
    }
    spec_code = (
        "import importlib.util,json; s=importlib.util.find_spec('libero'); "
        "print(json.dumps({'origin':s.origin,'locations':"
        "list(s.submodule_search_locations or [])},sort_keys=True))"
    )
    try:
        spec = json.loads(_run_identity_command(
            [str(LIBERO_EVAL_PYTHON), "-c", spec_code],
            label="LIBERO editable import path",
        ))
    except json.JSONDecodeError as exc:
        raise OperatorRepairError("LIBERO editable import receipt is malformed") from exc
    actual = _expected_eval_environment_receipt()
    actual["python"]["pip_freeze"] = {
        **actual["python"]["pip_freeze"],
        "sha256": hashlib.sha256(freeze).hexdigest(),
        "lines": len(freeze_lines),
        "bytes": len(freeze),
        "key_packages": key_packages,
    }
    actual["libero_repository"] = {
        **actual["libero_repository"],
        "head": head.decode().strip(),
        "clean_status_sha256": hashlib.sha256(status).hexdigest(),
        "clean_status_bytes": len(status),
        **tree_values,
        "editable_import_root": (
            spec.get("locations", [None])[0]
            if spec.get("origin") is None and len(spec.get("locations", [])) == 1
            else None
        ),
        "bddl_files": len(bddl_files),
        "init_state_files": len(init_files),
    }
    actual["siglip_snapshot"] = _siglip_eval_snapshot_receipt()
    expected = _expected_eval_environment_receipt()
    if actual != expected:
        raise OperatorRepairError("live LIBERO evaluation environment changed")
    return actual


def build_plan(
    *, run_dir: Path, control_dir: Path, artifact_root: Path,
    group: str, project: str = PROJECT,
) -> dict[str, Any]:
    if CANONICAL_CONFIG.resolve() != CANONICAL_CONFIG or (
        sha256_file(CANONICAL_CONFIG) != CANONICAL_CONFIG_SHA256
    ):
        raise OperatorRepairError("canonical dual-code config identity changed")
    if project != PROJECT:
        raise OperatorRepairError(f"W&B project must be exact {PROJECT}")
    if not group or re.fullmatch(r"[A-Za-z0-9_.-]+", group) is None:
        raise OperatorRepairError("W&B group must be a non-empty scheduler-safe token")
    if "operator-repair" not in group or "fixed32k" not in group:
        raise OperatorRepairError("W&B group must explicitly say operator-repair and fixed32k")
    run_dir = _require_clean_absolute(run_dir, field="run-dir")
    control_dir = _require_clean_absolute(control_dir, field="control-dir")
    artifact_root = _require_clean_absolute(artifact_root, field="artifact-root")
    _require_isolated(run_dir, control_dir, artifact_root)
    for path in (run_dir, control_dir, artifact_root):
        if path.exists():
            raise OperatorRepairError(f"fresh output root already exists: {path}")
    cfg, digest = _resolved_training_config(run_dir)
    _validate_training_config(cfg, digest)
    return {
        "format_version": FORMAT_VERSION,
        "kind": KIND,
        "eligibility": {
            "fixed_endpoint_full_run": True,
            "formal_convergence_gate": False,
            "checkpoint_selection_by_metrics_or_evaluation": False,
            "evaluation_unconditional_after_integrity": True,
            "promotion_authority": False,
        },
        "method": {
            "fresh_loom_modules": True,
            "frozen_siglip_cached_tower": True,
            "dual_action_mode": "dual_q_action_proposal",
            "fixed_endpoint": FIXED_STEP,
            "endpoint_predeclared_before_training": True,
            "health_metrics_role": "observational_only",
            "metrics_ledger": {
                "format": "loom-fresh-metrics-rollback-v1",
                "reconcile_crash_tail_to_latest_checkpoint": True,
                "checkpoint_boundary_fsync": True,
                "direct_formal_decisions": False,
            },
        },
        "source_closure": _source_closure(),
        "assets": _asset_receipt(),
        "config": {
            "path": str(CANONICAL_CONFIG),
            "raw_sha256": CANONICAL_CONFIG_SHA256,
            "overrides": list(TRAIN_OVERRIDES),
            "resolved_config_hash": digest,
            "resolved_experiment": cfg,
        },
        "lineage": {
            "run_name": RUN_NAME,
            "run_dir": str(run_dir),
            "control_dir": str(control_dir),
            "artifact_root": str(artifact_root),
        },
        "schedule": {
            "fixed_updates": FIXED_STEP,
            "optimizer_schedule_horizon": FIXED_STEP,
            "links": TRAIN_LINKS,
            "link_walltime": "04:00:00",
            "link_budget_seconds": 13_800,
            "world_size": WORLD_SIZE,
            "checkpoint_every": 500,
            "selection_rule": "exact_predeclared_step_32000_only",
        },
        "evaluation": {
            "seeds": list(SEEDS),
            "suites": list(DEFAULT_LIBERO_SUITES),
            "tasks_per_suite": 10,
            "episodes_per_task": 10,
            "max_steps": 512,
            "episodes_per_seed": common.EXPECTED_EPISODES_PER_SEED,
            "total_episodes": common.EXPECTED_EPISODES_TOTAL,
            "workers": 8,
            "gripper_dwell": 1,
            "decoder_samples": 1,
            "duration_normalize_segments": False,
            "embodiment": "libero_franka",
            "python": str(LIBERO_EVAL_PYTHON),
            "environment": _eval_environment_receipt(),
            "result_store_resume": True,
            "max_process_attempts": EVAL_MAX_PROCESS_ATTEMPTS,
            "error_row_policy": "content_address_quarantine_then_fresh_retry",
            "attempt_identity": (
                "immutable_pre_environment_and_checkpoint_plus_post_reauthentication"
            ),
            "attempt_identity_mismatch_policy": (
                "content_address_quarantine_before_fail_no_episode_laundering"
            ),
            "runs_even_when_observational_health_is_poor": True,
        },
        "baseline_comparison": _baseline_contract(),
        "wandb": {
            "project": project,
            "group": group,
            "require_online": True,
            "tags": list(EXPECTED_TRAIN_TAGS),
            "training_job_type": "operator-repair-train",
            "training_resume_policy": {
                "fresh_initial": "never",
                "no_latest_bootstrap_requeue": "allow",
                "post_checkpoint": "must",
            },
            "training_run_id": uuid.uuid4().hex[:16],
            "stage_run_ids": {
                stage: uuid.uuid4().hex[:16]
                for stage in (
                    "consolidate", "eval-seed-0", "eval-seed-1",
                    "eval-seed-2", "eval-summary",
                )
            },
            "artifact_policy": "receipts_and_eval_results_only_no_checkpoint_upload",
            "training_log_failure_policy": {
                "kind": "consecutive_failures",
                "max_consecutive_failures": (
                    train_entry.MAX_CONSECUTIVE_LOG_FAILURES
                ),
                "log_every_updates": train_entry.LOG_EVERY_UPDATES,
                "failure_window_updates": train_entry.LOG_FAILURE_WINDOW_UPDATES,
                "success_resets_counter": True,
                "all_rank_outcome_broadcast": True,
            },
        },
        "failure_policy": {
            "only_integrity_or_execution_failure_blocks_afterok": True,
            "observational_metrics": "record_only_never_control_execution",
            "poor_success_rate": "publish_descriptive_result_return_zero",
            "training_requeue": "same_job_and_exact_checkpoint_resume",
            "partial_evaluation": "atomic_result_store_resume",
            "partial_merge": "exact_recompute_then_adopt",
            "persistent_online_wandb_failure": "execution_failure",
            "persistent_online_wandb_failure_definition": (
                "five_consecutive_log_calls_over_100_updates"
            ),
            "post_training_asset_verification_failure": (
                "durable_terminal_marker_never_reauthorize"
            ),
            "training_asset_post_transaction": (
                "pending_before_post_verification_exact_complete_required"
            ),
        },
        "diary": {
            "path": str(ROOT / "DIARY.md"),
            "schema": "loom-experiment-diary-v1",
            "runtime_mutation": False,
            "receipt_is_authoritative_diary_is_human_reasoning": True,
        },
        "paths": _expected_paths(control_dir, artifact_root),
    }


def _assert_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("format_version") != FORMAT_VERSION or plan.get("kind") != KIND:
        raise OperatorRepairError("unsupported operator-repair plan")
    if plan.get("source_closure") != _source_closure():
        raise OperatorRepairError("operator-repair executable source closure changed")
    if plan.get("assets") != _expected_asset_receipt():
        raise OperatorRepairError("cache/frozen-tower asset receipt changed")
    config = plan.get("config", {})
    run_dir = Path(plan.get("lineage", {}).get("run_dir", "")).resolve()
    cfg, digest = _resolved_training_config(run_dir)
    _validate_training_config(cfg, digest)
    expected_config = {
        "path": str(CANONICAL_CONFIG), "raw_sha256": CANONICAL_CONFIG_SHA256,
        "overrides": list(TRAIN_OVERRIDES), "resolved_config_hash": digest,
        "resolved_experiment": cfg,
    }
    if config != expected_config:
        raise OperatorRepairError("operator-repair config receipt changed")
    lineage = plan["lineage"]
    raw_paths = tuple(
        Path(lineage[name]) for name in ("run_dir", "control_dir", "artifact_root")
    )
    for path in raw_paths:
        _reject_existing_symlink_components(path)
    control_dir = raw_paths[1].resolve()
    artifact_root = raw_paths[2].resolve()
    if lineage.get("run_name") != RUN_NAME:
        raise OperatorRepairError("run name changed")
    _require_isolated(run_dir, control_dir, artifact_root)
    if plan.get("paths") != _expected_paths(control_dir, artifact_root):
        raise OperatorRepairError("derived output paths changed")
    derived_output_paths = [
        Path(plan["paths"]["checkpoint"]),
        Path(plan["paths"]["merged_results"]),
        Path(plan["paths"]["merged_table"]),
    ]
    for seed in SEEDS:
        out_dir = Path(plan["paths"]["eval"][str(seed)]["out_dir"])
        derived_output_paths.extend((out_dir, out_dir / "runtime", out_dir / "runtime/triton_cache"))
    for path in derived_output_paths:
        _reject_existing_symlink_components(path)
    _validate_frozen_baseline_contract(plan.get("baseline_comparison"))
    evaluation = plan.get("evaluation", {})
    if evaluation != {
        "seeds": list(SEEDS), "suites": list(DEFAULT_LIBERO_SUITES),
        "tasks_per_suite": 10, "episodes_per_task": 10, "max_steps": 512,
        "episodes_per_seed": common.EXPECTED_EPISODES_PER_SEED,
        "total_episodes": common.EXPECTED_EPISODES_TOTAL, "workers": 8,
        "gripper_dwell": 1, "decoder_samples": 1,
        "duration_normalize_segments": False, "embodiment": "libero_franka",
        "python": str(LIBERO_EVAL_PYTHON),
        "environment": _expected_eval_environment_receipt(),
        "result_store_resume": True,
        "max_process_attempts": EVAL_MAX_PROCESS_ATTEMPTS,
        "error_row_policy": "content_address_quarantine_then_fresh_retry",
        "attempt_identity": (
            "immutable_pre_environment_and_checkpoint_plus_post_reauthentication"
        ),
        "attempt_identity_mismatch_policy": (
            "content_address_quarantine_before_fail_no_episode_laundering"
        ),
        "runs_even_when_observational_health_is_poor": True,
    }:
        raise OperatorRepairError("fixed evaluation protocol changed")
    if not LIBERO_EVAL_PYTHON.is_file():
        raise OperatorRepairError(f"pinned LIBERO Python is absent: {LIBERO_EVAL_PYTHON}")
    if plan.get("schedule") != {
        "fixed_updates": FIXED_STEP, "optimizer_schedule_horizon": FIXED_STEP,
        "links": TRAIN_LINKS, "link_walltime": "04:00:00",
        "link_budget_seconds": 13_800, "world_size": WORLD_SIZE,
        "checkpoint_every": 500, "selection_rule": "exact_predeclared_step_32000_only",
    }:
        raise OperatorRepairError("fixed training schedule changed")
    if plan.get("eligibility") != {
        "fixed_endpoint_full_run": True,
        "formal_convergence_gate": False,
        "checkpoint_selection_by_metrics_or_evaluation": False,
        "evaluation_unconditional_after_integrity": True,
        "promotion_authority": False,
    }:
        raise OperatorRepairError("no-gate/no-promotion contract changed")
    if plan.get("method") != {
        "fresh_loom_modules": True,
        "frozen_siglip_cached_tower": True,
        "dual_action_mode": "dual_q_action_proposal",
        "fixed_endpoint": FIXED_STEP,
        "endpoint_predeclared_before_training": True,
        "health_metrics_role": "observational_only",
        "metrics_ledger": {
            "format": "loom-fresh-metrics-rollback-v1",
            "reconcile_crash_tail_to_latest_checkpoint": True,
            "checkpoint_boundary_fsync": True,
            "direct_formal_decisions": False,
        },
    }:
        raise OperatorRepairError("operator-repair method contract changed")
    if plan.get("failure_policy") != {
        "only_integrity_or_execution_failure_blocks_afterok": True,
        "observational_metrics": "record_only_never_control_execution",
        "poor_success_rate": "publish_descriptive_result_return_zero",
        "training_requeue": "same_job_and_exact_checkpoint_resume",
        "partial_evaluation": "atomic_result_store_resume",
        "partial_merge": "exact_recompute_then_adopt",
        "persistent_online_wandb_failure": "execution_failure",
        "persistent_online_wandb_failure_definition": (
            "five_consecutive_log_calls_over_100_updates"
        ),
        "post_training_asset_verification_failure": (
            "durable_terminal_marker_never_reauthorize"
        ),
        "training_asset_post_transaction": (
            "pending_before_post_verification_exact_complete_required"
        ),
    }:
        raise OperatorRepairError("operator-repair failure policy changed")
    if plan.get("diary") != {
        "path": str(ROOT / "DIARY.md"),
        "schema": "loom-experiment-diary-v1",
        "runtime_mutation": False,
        "receipt_is_authoritative_diary_is_human_reasoning": True,
    }:
        raise OperatorRepairError("operator-repair diary contract changed")
    wandb = plan.get("wandb", {})
    stage_ids = wandb.get("stage_run_ids", {})
    if not (
        wandb.get("project") == PROJECT
        and isinstance(wandb.get("group"), str)
        and re.fullmatch(r"[A-Za-z0-9_.-]+", wandb["group"]) is not None
        and "operator-repair" in wandb["group"] and "fixed32k" in wandb["group"]
        and wandb.get("require_online") is True
        and wandb.get("tags") == EXPECTED_STAGE_TAGS
        and wandb.get("training_job_type") == "operator-repair-train"
        and wandb.get("training_resume_policy") == {
            "fresh_initial": "never",
            "no_latest_bootstrap_requeue": "allow",
            "post_checkpoint": "must",
        }
        and wandb.get("artifact_policy")
        == "receipts_and_eval_results_only_no_checkpoint_upload"
        and wandb.get("training_log_failure_policy") == {
            "kind": "consecutive_failures",
            "max_consecutive_failures": train_entry.MAX_CONSECUTIVE_LOG_FAILURES,
            "log_every_updates": train_entry.LOG_EVERY_UPDATES,
            "failure_window_updates": train_entry.LOG_FAILURE_WINDOW_UPDATES,
            "success_resets_counter": True,
            "all_rank_outcome_broadcast": True,
        }
        and re.fullmatch(r"[0-9a-f]{16}", str(wandb.get("training_run_id")))
        and set(stage_ids) == {
            "consolidate", "eval-seed-0", "eval-seed-1", "eval-seed-2", "eval-summary",
        }
        and len(set(stage_ids.values())) == 5
        and all(re.fullmatch(r"[0-9a-f]{16}", str(value)) for value in stage_ids.values())
        and wandb["training_run_id"] not in stage_ids.values()
    ):
        raise OperatorRepairError("operator-repair W&B contract changed")


def load_plan(path: str | Path, expected_sha256: str | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    if expected_sha256 and sha256_file(path) != expected_sha256:
        raise OperatorRepairError("operator-repair plan SHA-256 mismatch")
    try:
        plan = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise OperatorRepairError("operator-repair plan is unreadable") from exc
    if not isinstance(plan, Mapping):
        raise OperatorRepairError("operator-repair plan is not an object")
    plan = dict(plan)
    _assert_plan(plan)
    return plan


def _required_plan() -> tuple[dict[str, Any], str, Path]:
    value = os.environ.get("OPERATOR_REPAIR_PLAN")
    digest = os.environ.get("OPERATOR_REPAIR_PLAN_SHA256")
    stage = os.environ.get("OPERATOR_REPAIR_STAGE")
    if not value or not digest or not stage:
        raise OperatorRepairError(
            "OPERATOR_REPAIR_PLAN, OPERATOR_REPAIR_PLAN_SHA256, and "
            "OPERATOR_REPAIR_STAGE are required"
        )
    path = Path(value).resolve()
    return load_plan(path, digest), stage, path


def _plan_sha() -> str:
    value = os.environ.get("OPERATOR_REPAIR_PLAN")
    if not value:
        raise OperatorRepairError("OPERATOR_REPAIR_PLAN is required")
    return sha256_file(value)


def _read_receipt(path: str | Path, *, kind: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise OperatorRepairError(f"invalid {kind} receipt") from exc
    if not isinstance(value, Mapping) or value.get("format_version") != FORMAT_VERSION or value.get("kind") != kind:
        raise OperatorRepairError(f"invalid {kind} receipt")
    return dict(value)


def _stage_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    previous: list[str] = []
    for index in range(1, TRAIN_LINKS + 1):
        name = f"train_{index:02d}"
        specs.append({
            "name": name,
            "sbatch": "scripts/r0_e2e_operator_repair_train.sbatch",
            "depends_on": list(previous),
        })
        previous = [name]
    specs.append({
        "name": "consolidate",
        "sbatch": "scripts/r0_e2e_operator_repair_consolidate.sbatch",
        "depends_on": list(previous),
    })
    specs.extend({
        "name": f"eval_seed{seed}",
        "sbatch": "scripts/r0_e2e_operator_repair_eval_seed.sbatch",
        "depends_on": ["consolidate"],
    } for seed in SEEDS)
    specs.append({
        "name": "merge",
        "sbatch": "scripts/r0_e2e_operator_repair_control.sbatch",
        "depends_on": [f"eval_seed{seed}" for seed in SEEDS],
    })
    return specs


def _sbatch_command(
    *, spec: Mapping[str, Any], plan_path: Path, plan_sha: str,
    dependencies: Sequence[str], group: str,
) -> list[str]:
    label = re.sub(r"[^A-Za-z0-9_-]", "_", f"r0repair_{group}_{spec['name']}")[:120]
    command = [
        "sbatch", "--parsable", "--hold", "--kill-on-invalid-dep=yes",
        f"--job-name={label}",
    ]
    if dependencies:
        command.append("--dependency=afterok:" + ":".join(dependencies))
    export = ",".join((
        "ALL", f"OPERATOR_REPAIR_PLAN={plan_path}",
        f"OPERATOR_REPAIR_PLAN_SHA256={plan_sha}",
        f"OPERATOR_REPAIR_STAGE={spec['name']}",
    ))
    command.extend((f"--export={export}", str(ROOT / str(spec["sbatch"]))))
    return command


def _parse_job_id(stdout: str) -> str:
    value = stdout.strip().split(";", 1)[0]
    if re.fullmatch(r"[0-9]+(?:_[0-9]+)?", value) is None:
        raise OperatorRepairError(f"invalid sbatch job id: {stdout!r}")
    return value


def submit_plan(
    plan: Mapping[str, Any], *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    _assert_plan(plan)
    if plan.get("assets") != _asset_receipt():
        raise OperatorRepairError("training inputs changed at plan submission")
    if plan.get("evaluation", {}).get("environment") != _eval_environment_receipt():
        raise OperatorRepairError("LIBERO evaluation environment changed at submission")
    lineage = plan["lineage"]
    run_dir = Path(lineage["run_dir"])
    control_dir = Path(lineage["control_dir"])
    artifact_root = Path(lineage["artifact_root"])
    if run_dir.exists() or control_dir.exists() or artifact_root.exists():
        raise OperatorRepairError("fresh run/control/artifact roots are required")
    control_dir.mkdir(parents=True, exist_ok=False)
    plan_path = control_dir / "plan.json"
    common.exclusive_json_write(plan_path, plan)
    plan_sha = sha256_file(plan_path)
    jobs: dict[str, str] = {}
    commands: dict[str, list[str]] = {}
    submitted: list[str] = []
    try:
        for spec in _stage_specs():
            command = _sbatch_command(
                spec=spec, plan_path=plan_path, plan_sha=plan_sha,
                dependencies=[jobs[name] for name in spec["depends_on"]],
                group=plan["wandb"]["group"],
            )
            completed = run(
                command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, check=True,
            )
            job_id = _parse_job_id(completed.stdout)
            jobs[str(spec["name"])] = job_id
            commands[str(spec["name"])] = command
            submitted.append(job_id)
        receipt = {
            "format_version": FORMAT_VERSION,
            "kind": "r0_e2e_operator_repair_jobs",
            "plan": str(plan_path), "plan_sha256": plan_sha,
            "jobs": jobs, "commands": commands, "released": False,
            "fixed_endpoint": FIXED_STEP, "decision_gate_jobs": [],
        }
        common.exclusive_json_write(control_dir / "jobs.json", receipt)
        run(
            ["scontrol", "release", ",".join(submitted)], cwd=ROOT,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
        )
        common.exclusive_json_write(control_dir / "released.json", {
            "format_version": FORMAT_VERSION,
            "kind": "r0_e2e_operator_repair_release",
            "plan_sha256": plan_sha,
            "jobs_sha256": sha256_file(control_dir / "jobs.json"),
            "job_ids": submitted, "released": True,
        })
        return {**receipt, "released": True}
    except Exception:
        if submitted:
            run(
                ["scancel", *submitted], cwd=ROOT, check=False,
                text=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        raise


def _latest_step(run_dir: Path) -> int:
    try:
        value = int((run_dir / "LATEST").read_text().strip())
    except (OSError, ValueError) as exc:
        raise OperatorRepairError(f"missing/invalid LATEST under {run_dir}") from exc
    if not 0 < value <= FIXED_STEP:
        raise OperatorRepairError(f"LATEST={value} is outside 1..{FIXED_STEP}")
    return value


def _run_config_identity(plan: Mapping[str, Any]) -> dict[str, Any]:
    from loom.train.loop import config_hash  # noqa: PLC0415

    path = Path(plan["lineage"]["run_dir"]) / "config.json"
    try:
        cfg = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise OperatorRepairError("run config.json is absent or unreadable") from exc
    if not isinstance(cfg, Mapping):
        raise OperatorRepairError("run config.json is not an object")
    experiment = {key: value for key, value in cfg.items() if key != "link"}
    if (
        config_hash(dict(cfg)) != plan["config"]["resolved_config_hash"]
        or experiment != plan["config"]["resolved_experiment"]
    ):
        raise OperatorRepairError("materialized run config differs from fixed plan")
    return _identity(path)


def _training_asset_verification_path(
    plan: Mapping[str, Any], *, stage: str, phase: str,
) -> Path:
    if re.fullmatch(r"train_[0-9]{2}", stage) is None or phase not in {"pre", "post"}:
        raise OperatorRepairError("invalid training asset verification stage/phase")
    return Path(plan["paths"]["training_asset_verification_dir"]) / (
        f"{stage}_{phase}.json"
    )


def _training_asset_verification_payload(
    plan: Mapping[str, Any], *, stage: str, phase: str,
) -> dict[str, Any]:
    assets = _asset_receipt()
    if assets != plan["assets"]:
        raise OperatorRepairError(f"training assets changed at {stage}/{phase}")
    return {
        "format_version": FORMAT_VERSION,
        "kind": "r0_e2e_operator_repair_training_asset_verification",
        "plan_sha256": _plan_sha(),
        "stage": stage,
        "phase": phase,
        "assets": assets,
        "raw_and_cache_full_semantic_content_rehashed": True,
        "controls_training_integrity_not_scientific_selection": True,
    }


def _publish_training_asset_verification(
    plan: Mapping[str, Any], *, stage: str, phase: str,
) -> dict[str, Any]:
    path = _training_asset_verification_path(plan, stage=stage, phase=phase)
    payload = _training_asset_verification_payload(plan, stage=stage, phase=phase)
    if path.exists():
        if not path.is_file() or path.is_symlink():
            raise OperatorRepairError("training asset verification is nonregular")
    else:
        try:
            common.exclusive_json_write(path, payload)
        except FileExistsError:
            pass
    if not path.is_file() or path.is_symlink():
        raise OperatorRepairError("training asset verification is nonregular")
    try:
        actual = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise OperatorRepairError("training asset verification is unreadable") from exc
    if actual != payload:
        raise OperatorRepairError("training asset verification replay differs")
    return payload


def _read_training_asset_verification(
    plan: Mapping[str, Any], *, stage: str, phase: str,
) -> tuple[dict[str, Any], str]:
    path = _training_asset_verification_path(plan, stage=stage, phase=phase)
    try:
        receipt = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise OperatorRepairError("training asset verification is absent/unreadable") from exc
    expected = {
        "format_version": FORMAT_VERSION,
        "kind": "r0_e2e_operator_repair_training_asset_verification",
        "plan_sha256": _plan_sha(),
        "stage": stage,
        "phase": phase,
        "assets": plan["assets"],
        "raw_and_cache_full_semantic_content_rehashed": True,
        "controls_training_integrity_not_scientific_selection": True,
    }
    if receipt != expected or path.is_symlink():
        raise OperatorRepairError("training asset verification receipt changed")
    return receipt, sha256_file(path)


def _read_final_training_asset_verification(
    plan: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    return _read_training_asset_verification(
        plan, stage=f"train_{TRAIN_LINKS:02d}", phase="post",
    )


def _training_asset_post_transaction_paths(
    plan: Mapping[str, Any], *, stage: str, restart_count: int,
) -> tuple[Path, Path]:
    if (
        re.fullmatch(r"train_0[1-6]", stage) is None
        or not isinstance(restart_count, int) or isinstance(restart_count, bool)
        or restart_count < 0
    ):
        raise OperatorRepairError("invalid training-asset transaction identity")
    directory = Path(plan["paths"]["training_asset_verification_dir"])
    stem = f"{stage}_restart_{restart_count:06d}_post"
    return directory / f"{stem}_pending.json", directory / f"{stem}_complete.json"


def _training_asset_post_pending_payload(
    plan: Mapping[str, Any], *, stage: str, restart_count: int,
    starting_checkpoint_step: int, pre_receipt_sha256: str,
) -> dict[str, Any]:
    if (
        not isinstance(starting_checkpoint_step, int)
        or isinstance(starting_checkpoint_step, bool)
        or not 0 <= starting_checkpoint_step <= FIXED_STEP
    ):
        raise OperatorRepairError("invalid training-asset starting checkpoint")
    pending_path, _ = _training_asset_post_transaction_paths(
        plan, stage=stage, restart_count=restart_count,
    )
    pre_path = _training_asset_verification_path(
        plan, stage=stage, phase="pre",
    )
    pre_receipt, actual_pre_sha = _read_training_asset_verification(
        plan, stage=stage, phase="pre",
    )
    if actual_pre_sha != pre_receipt_sha256:
        raise OperatorRepairError(
            "training-asset transaction pre-receipt identity changed"
        )
    return {
        "format_version": FORMAT_VERSION,
        "kind": "r0_e2e_operator_repair_training_asset_post_pending",
        "plan_sha256": _plan_sha(),
        "stage": stage,
        "restart_count": restart_count,
        "starting_checkpoint_step": starting_checkpoint_step,
        "phase": "post",
        "path": str(pending_path),
        "pre_verification": {
            "path": str(pre_path),
            "sha256": actual_pre_sha,
            "receipt_kind": pre_receipt["kind"],
            "assets_sha256": hashlib.sha256(
                _canonical_json(pre_receipt["assets"]).encode()
            ).hexdigest(),
        },
        "must_close_before_this_invocation_can_authorize_continuation": True,
    }


def _begin_training_asset_post_transaction(
    plan: Mapping[str, Any], *, stage: str, restart_count: int,
    starting_checkpoint_step: int, pre_receipt_sha256: str,
) -> dict[str, Any]:
    path, _ = _training_asset_post_transaction_paths(
        plan, stage=stage, restart_count=restart_count,
    )
    expected = _training_asset_post_pending_payload(
        plan, stage=stage, restart_count=restart_count,
        starting_checkpoint_step=starting_checkpoint_step,
        pre_receipt_sha256=pre_receipt_sha256,
    )
    if path.exists():
        if not path.is_file() or path.is_symlink():
            raise OperatorRepairError("training-asset pending transaction is nonregular")
    else:
        try:
            common.exclusive_json_write(path, expected)
        except FileExistsError:
            pass
    if not path.is_file() or path.is_symlink():
        raise OperatorRepairError("training-asset pending transaction was not published")
    try:
        actual = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise OperatorRepairError("training-asset pending transaction is unreadable") from exc
    if actual != expected:
        raise OperatorRepairError("training-asset pending transaction differs on replay")
    return expected


def _training_asset_post_complete_payload(
    plan: Mapping[str, Any], *, pending: Mapping[str, Any],
) -> dict[str, Any]:
    stage = pending.get("stage")
    restart_count = pending.get("restart_count")
    starting_step = pending.get("starting_checkpoint_step")
    pre = pending.get("pre_verification")
    pre_sha = pre.get("sha256") if isinstance(pre, Mapping) else None
    if not isinstance(stage, str) or not isinstance(pre_sha, str):
        raise OperatorRepairError("training-asset pending transaction is malformed")
    expected_pending = _training_asset_post_pending_payload(
        plan, stage=stage, restart_count=restart_count,
        starting_checkpoint_step=starting_step,
        pre_receipt_sha256=pre_sha,
    )
    if dict(pending) != expected_pending:
        raise OperatorRepairError("training-asset pending transaction changed")
    pending_path, complete_path = _training_asset_post_transaction_paths(
        plan, stage=stage, restart_count=restart_count,
    )
    post_path = _training_asset_verification_path(
        plan, stage=stage, phase="post",
    )
    _, post_sha = _read_training_asset_verification(
        plan, stage=stage, phase="post",
    )
    return {
        "format_version": FORMAT_VERSION,
        "kind": "r0_e2e_operator_repair_training_asset_post_complete",
        "plan_sha256": _plan_sha(),
        "stage": stage,
        "restart_count": restart_count,
        "starting_checkpoint_step": starting_step,
        "path": str(complete_path),
        "pending": {"path": str(pending_path), "sha256": sha256_file(pending_path)},
        "post_verification": {"path": str(post_path), "sha256": post_sha},
        "raw_and_cache_full_semantic_content_rehashed_after_invocation": True,
    }


def _complete_training_asset_post_transaction(
    plan: Mapping[str, Any], *, pending: Mapping[str, Any],
) -> dict[str, Any]:
    stage = str(pending.get("stage"))
    restart_count = pending.get("restart_count")
    _, path = _training_asset_post_transaction_paths(
        plan, stage=stage, restart_count=restart_count,
    )
    expected = _training_asset_post_complete_payload(plan, pending=pending)
    if path.exists():
        if not path.is_file() or path.is_symlink():
            raise OperatorRepairError("training-asset completion is nonregular")
    else:
        try:
            common.exclusive_json_write(path, expected)
        except FileExistsError:
            pass
    if not path.is_file() or path.is_symlink():
        raise OperatorRepairError("training-asset completion was not published")
    try:
        actual = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise OperatorRepairError("training-asset completion is unreadable") from exc
    if actual != expected:
        raise OperatorRepairError("training-asset completion differs on replay")
    return expected


def _training_asset_failure_payload(
    plan: Mapping[str, Any], *, pending: Mapping[str, Any],
) -> dict[str, Any]:
    stage = pending.get("stage")
    restart_count = pending.get("restart_count")
    starting_step = pending.get("starting_checkpoint_step")
    pre = pending.get("pre_verification")
    pre_sha = pre.get("sha256") if isinstance(pre, Mapping) else None
    if not isinstance(stage, str) or not isinstance(pre_sha, str):
        raise OperatorRepairError("training-asset pending transaction is malformed")
    expected_pending = _training_asset_post_pending_payload(
        plan, stage=stage, restart_count=restart_count,
        starting_checkpoint_step=starting_step,
        pre_receipt_sha256=pre_sha,
    )
    if dict(pending) != expected_pending:
        raise OperatorRepairError("training-asset pending transaction changed")
    pending_path, _ = _training_asset_post_transaction_paths(
        plan, stage=stage, restart_count=restart_count,
    )
    return {
        "format_version": FORMAT_VERSION,
        "kind": "r0_e2e_operator_repair_training_asset_failure",
        "plan_sha256": _plan_sha(),
        "stage": stage,
        "restart_count": restart_count,
        "phase": "post",
        "error_category": "post_training_asset_verification_failed",
        "pending": {"path": str(pending_path), "sha256": sha256_file(pending_path)},
        "pre_verification": copy.deepcopy(pending["pre_verification"]),
        "terminal_for_lineage": True,
        "restored_assets_cannot_reauthorize_this_lineage": True,
    }


def _publish_training_asset_failure(
    plan: Mapping[str, Any], *, pending: Mapping[str, Any],
) -> dict[str, Any]:
    path = Path(plan["paths"]["training_asset_failure"])
    expected = _training_asset_failure_payload(plan, pending=pending)
    if path.exists():
        if not path.is_file() or path.is_symlink():
            raise OperatorRepairError("training-asset failure marker is nonregular")
    else:
        try:
            common.exclusive_json_write(path, expected)
        except FileExistsError:
            pass
    if not path.is_file() or path.is_symlink():
        raise OperatorRepairError("training-asset failure marker was not published")
    try:
        actual = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise OperatorRepairError("training-asset failure marker is unreadable") from exc
    if actual != expected:
        raise OperatorRepairError("training-asset failure marker differs on replay")
    return expected


def _read_training_asset_transaction(path: Path, *, kind: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise OperatorRepairError("training-asset transaction is nonregular")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise OperatorRepairError("training-asset transaction is unreadable") from exc
    if not isinstance(value, Mapping) or value.get("kind") != kind:
        raise OperatorRepairError("training-asset transaction is malformed")
    return dict(value)


def _reject_training_asset_failure(plan: Mapping[str, Any]) -> None:
    failure_path = Path(plan["paths"]["training_asset_failure"])
    if failure_path.exists():
        marker = _read_training_asset_transaction(
            failure_path, kind="r0_e2e_operator_repair_training_asset_failure",
        )
        pending_ref = marker.get("pending")
        stage = marker.get("stage")
        restart_count = marker.get("restart_count")
        pending_path, _ = _training_asset_post_transaction_paths(
            plan, stage=stage, restart_count=restart_count,
        )
        if not isinstance(pending_ref, Mapping) or pending_ref.get("path") != str(
            pending_path
        ):
            raise OperatorRepairError("training-asset failure marker is malformed")
        pending = _read_training_asset_transaction(
            pending_path, kind="r0_e2e_operator_repair_training_asset_post_pending",
        )
        if marker != _training_asset_failure_payload(plan, pending=pending):
            raise OperatorRepairError("training-asset failure marker changed")
        raise OperatorRepairError(
            "training lineage is terminal after post-link asset verification failure "
            f"in {marker['stage']}"
        )

    directory = Path(plan["paths"]["training_asset_verification_dir"])
    if not directory.exists():
        return
    if not directory.is_dir() or directory.is_symlink():
        raise OperatorRepairError("training-asset verification directory is nonregular")
    pending_paths = sorted(directory.glob("train_*_post_pending.json"))
    expected_completions: set[Path] = set()
    for pending_path in pending_paths:
        pending = _read_training_asset_transaction(
            pending_path, kind="r0_e2e_operator_repair_training_asset_post_pending",
        )
        stage = pending.get("stage")
        restart_count = pending.get("restart_count")
        expected_pending_path, complete_path = _training_asset_post_transaction_paths(
            plan, stage=stage, restart_count=restart_count,
        )
        if pending_path != expected_pending_path:
            raise OperatorRepairError("training-asset pending path changed")
        expected_completions.add(complete_path)
        if not complete_path.exists():
            raise OperatorRepairError(
                "training lineage is terminal after an incomplete post-link asset "
                f"verification transaction in {stage}"
            )
        complete = _read_training_asset_transaction(
            complete_path, kind="r0_e2e_operator_repair_training_asset_post_complete",
        )
        if complete != _training_asset_post_complete_payload(plan, pending=pending):
            raise OperatorRepairError("training-asset completion changed")
    actual_completions = set(directory.glob("train_*_post_complete.json"))
    if actual_completions != expected_completions:
        raise OperatorRepairError("orphan training-asset completion exists")


def _publish_post_training_asset_verification(
    plan: Mapping[str, Any], *, pending: Mapping[str, Any],
) -> dict[str, Any]:
    stage = str(pending.get("stage"))
    try:
        result = _publish_training_asset_verification(
            plan, stage=stage, phase="post",
        )
        _complete_training_asset_post_transaction(plan, pending=pending)
        return result
    except BaseException as verification_error:
        try:
            _publish_training_asset_failure(plan, pending=pending)
        except BaseException as marker_error:
            raise OperatorRepairError(
                "post-link asset verification failed and its durable terminal "
                "marker could not be published"
            ) from marker_error
        raise OperatorRepairError(
            "post-link asset verification failed; lineage is durably terminal"
        ) from verification_error


def _fresh_lineage_marker_payload(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "format": "loom-fresh-training-lineage-marker-v1",
        "config_hash": plan["config"]["resolved_config_hash"],
        "run_name": RUN_NAME,
        "metrics_rollback_format": "loom-fresh-metrics-rollback-v1",
    }


def _materialize_training_bootstrap(
    plan: Mapping[str, Any], *, run_dir: Path, allow_create_missing: bool,
) -> tuple[Path, Path]:
    plan_sha = _plan_sha()
    marker_path = run_dir / "fresh_lineage_marker.json"
    health_path = run_dir / "operator_repair_wandb_health.json"
    wandb_id_path = run_dir / "wandb_id"
    marker = _fresh_lineage_marker_payload(plan)
    health = train_entry.initial_wandb_health_state(plan_sha)
    expected_id = str(plan["wandb"]["training_run_id"])
    if marker_path.exists():
        try:
            actual_marker = json.loads(marker_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise OperatorRepairError("training lineage marker is unreadable") from exc
        if marker_path.is_symlink() or actual_marker != marker:
            raise OperatorRepairError("resumable training bootstrap identity changed")
    elif allow_create_missing:
        common.exclusive_json_write(marker_path, marker)
    else:
        raise OperatorRepairError("resumable link lacks its training lineage marker")

    if health_path.exists():
        try:
            actual_health = json.loads(health_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise OperatorRepairError("W&B health state is unreadable") from exc
        if health_path.is_symlink():
            raise OperatorRepairError("W&B health state cannot be a symlink")
        try:
            train_entry._validate_wandb_health_state(
                actual_health, lineage_sha256=plan_sha,
            )
        except train_entry.OperatorRepairWandbError as exc:
            raise OperatorRepairError(str(exc)) from exc
    elif allow_create_missing:
        common.exclusive_json_write(health_path, health)
    else:
        raise OperatorRepairError("resumable link lacks its W&B health state")

    if wandb_id_path.exists():
        if (
            wandb_id_path.is_symlink()
            or not wandb_id_path.is_file()
            or wandb_id_path.read_text().strip() != expected_id
        ):
            raise OperatorRepairError("continuation W&B identity differs from plan")
    elif allow_create_missing:
        common._exclusive_text_write(wandb_id_path, expected_id + "\n")
    else:
        raise OperatorRepairError("resumable link lacks its W&B identity")
    return marker_path, health_path


def _require_nonterminal_wandb_health(
    health_path: Path, *, lineage_sha256: str,
) -> dict[str, Any]:
    try:
        state = json.loads(health_path.read_text())
        state = train_entry._validate_wandb_health_state(
            state, lineage_sha256=lineage_sha256,
        )
    except (OSError, json.JSONDecodeError, train_entry.OperatorRepairWandbError) as exc:
        raise OperatorRepairError("persistent W&B health state is invalid") from exc
    failures = train_entry._consecutive_failures(state["events"])
    if failures >= train_entry.MAX_CONSECUTIVE_LOG_FAILURES:
        raise OperatorRepairError(
            "persistent W&B logging reached the durable five-failure execution limit"
        )
    return {"consecutive_failures": failures, "last_event_step": (
        state["events"][-1]["global_step"] if state["events"] else 0
    )}


def _reject_durable_execution_failure(plan: Mapping[str, Any]) -> None:
    from loom.train.loop import _read_execution_failure  # noqa: PLC0415

    run_dir = Path(plan["lineage"]["run_dir"])
    try:
        marker = _read_execution_failure(
            run_dir, config_hash_value=plan["config"]["resolved_config_hash"],
        )
    except (RuntimeError, ValueError) as exc:
        raise OperatorRepairError("durable execution-failure marker is invalid") from exc
    if marker is not None:
        raise OperatorRepairError(
            "training lineage is terminal after durable execution failure at "
            f"step {marker['global_step']}: {marker['reason']}"
        )


def _metrics_receipt(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "metrics.jsonl"
    if not path.is_file() or path.is_symlink():
        raise OperatorRepairError("metrics.jsonl is absent/non-regular")
    count = 0
    last_step = 0
    last: dict[str, Any] = {}
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise OperatorRepairError(
                    f"metrics.jsonl line {line_number} is invalid JSON"
                ) from exc
            if not isinstance(row, Mapping) or row.get("global_step") != line_number:
                raise OperatorRepairError(
                    f"metrics ledger is not exact contiguous 1..N at line {line_number}"
                )
            count = line_number
            last_step = int(row["global_step"])
            last = dict(row)
    if count != FIXED_STEP or last_step != FIXED_STEP:
        raise OperatorRepairError(
            f"metrics ledger closes at {last_step}/{count}, expected {FIXED_STEP}"
        )
    observed_keys = (
        "loss/dyn", "act/decode_teacher", "act/decode_deploy", "act/align",
        "loss/proposal", "delta_op", "delta_sel/h1", "delta_sel/h2",
        "delta_sel/h3", "delta_sel/h4", "bank/live_ops_q_delta",
        "bank/live_ops_q_a", "grad_norm", "grad_skipped",
    )
    snapshot: dict[str, Any] = {}
    missing: list[str] = []
    for key in observed_keys:
        value = last.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            missing.append(key)
            continue
        if not math.isfinite(float(value)):
            raise OperatorRepairError(f"final observational metric {key} is nonfinite")
        snapshot[key] = value
    return {
        **_identity(path),
        "rows": count,
        "first_step": 1,
        "last_step": last_step,
        "contiguous_unique_steps": True,
        "role": "observational_only_never_a_dependency_decision",
        "final_snapshot": snapshot,
        "missing_observational_keys": missing,
    }


def _fixed_endpoint_payload(plan: Mapping[str, Any]) -> dict[str, Any]:
    run_dir = Path(plan["lineage"]["run_dir"])
    _reject_training_asset_failure(plan)
    _reject_durable_execution_failure(plan)
    if _latest_step(run_dir) != FIXED_STEP:
        raise OperatorRepairError("training did not reach the exact fixed endpoint")
    try:
        shards = common._checkpoint_shards(run_dir, FIXED_STEP, WORLD_SIZE)
    except common.ChainError as exc:
        raise OperatorRepairError(str(exc)) from exc
    direct_receipts = sorted(path.name for path in run_dir.glob("direct_formal_*.json"))
    if direct_receipts:
        raise OperatorRepairError(
            f"no-gate lineage unexpectedly contains direct-formal receipts: {direct_receipts}"
        )
    wandb_id = (run_dir / "wandb_id").read_text().strip()
    if wandb_id != plan["wandb"]["training_run_id"]:
        raise OperatorRepairError("fixed endpoint W&B identity differs from plan")
    asset_verification, asset_verification_sha = (
        _read_final_training_asset_verification(plan)
    )
    return {
        "format_version": FORMAT_VERSION,
        "kind": "r0_e2e_operator_repair_fixed_endpoint",
        "plan_sha256": _plan_sha(),
        "step": FIXED_STEP,
        "selection": "predeclared_fixed_step_no_metric_or_eval_selection",
        "optimizer_updates": FIXED_STEP,
        "run_config": _run_config_identity(plan),
        "metrics": _metrics_receipt(run_dir),
        "checkpoint_shards": common._checkpoint_shard_receipt(shards),
        "training_asset_verification": {
            "stage": asset_verification["stage"],
            "phase": asset_verification["phase"],
            "path": str(_training_asset_verification_path(
                plan, stage=asset_verification["stage"],
                phase=asset_verification["phase"],
            )),
            "sha256": asset_verification_sha,
        },
        "training_wandb_run_id": wandb_id,
        "direct_formal_receipts": [],
        "health_metrics_used_as_gate": False,
        "evaluation_required_after_integrity": True,
    }


def _read_fixed_endpoint(
    plan: Mapping[str, Any], *, rehash_source: bool = False,
) -> tuple[dict[str, Any], str]:
    path = Path(plan["paths"]["fixed_endpoint"])
    receipt = _read_receipt(path, kind="r0_e2e_operator_repair_fixed_endpoint")
    shards = receipt.get("checkpoint_shards")
    expected_names = {
        f"ckpt_{FIXED_STEP:09d}_rank{rank:05d}.pt" for rank in range(WORLD_SIZE)
    }
    if not (
        receipt.get("plan_sha256") == _plan_sha()
        and receipt.get("step") == FIXED_STEP
        and receipt.get("selection") == "predeclared_fixed_step_no_metric_or_eval_selection"
        and receipt.get("optimizer_updates") == FIXED_STEP
        and receipt.get("training_wandb_run_id") == plan["wandb"]["training_run_id"]
        and receipt.get("direct_formal_receipts") == []
        and receipt.get("health_metrics_used_as_gate") is False
        and receipt.get("evaluation_required_after_integrity") is True
        and receipt.get("training_asset_verification", {}).get("stage")
        == f"train_{TRAIN_LINKS:02d}"
        and receipt.get("training_asset_verification", {}).get("phase") == "post"
        and receipt.get("training_asset_verification", {}).get("path")
        == str(_training_asset_verification_path(
            plan, stage=f"train_{TRAIN_LINKS:02d}", phase="post",
        ))
        and re.fullmatch(
            r"[0-9a-f]{64}",
            str(receipt.get("training_asset_verification", {}).get("sha256")),
        )
        and isinstance(shards, Mapping) and set(shards) == expected_names
        and all(
            isinstance(row, Mapping)
            and isinstance(row.get("bytes"), int) and row["bytes"] > 0
            and re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256")))
            for row in shards.values()
        )
        and receipt.get("run_config", {}).get("path")
        == str((Path(plan["lineage"]["run_dir"]) / "config.json").resolve())
        and isinstance(receipt.get("run_config", {}).get("bytes"), int)
        and receipt["run_config"]["bytes"] > 0
        and re.fullmatch(r"[0-9a-f]{64}", str(receipt["run_config"].get("sha256")))
        and receipt.get("metrics", {}).get("path")
        == str((Path(plan["lineage"]["run_dir"]) / "metrics.jsonl").resolve())
        and re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("metrics", {}).get("sha256")))
        and receipt.get("metrics", {}).get("rows") == FIXED_STEP
        and receipt.get("metrics", {}).get("role")
        == "observational_only_never_a_dependency_decision"
    ):
        raise OperatorRepairError("fixed endpoint receipt failed immutable shape/plan closure")
    if rehash_source:
        expected = _fixed_endpoint_payload(plan)
        if receipt != expected:
            raise OperatorRepairError(
                "fixed endpoint receipt differs from live exact recomputation"
            )
    return receipt, sha256_file(path)


def _train_stage_policy(
    stage: str, restart_count: int, *, has_latest: bool,
) -> dict[str, Any]:
    match = re.fullmatch(r"train_([0-9]{2})", stage)
    if match is None:
        raise OperatorRepairError(f"invalid train stage {stage!r}")
    index = int(match.group(1))
    if not 1 <= index <= TRAIN_LINKS or restart_count < 0:
        raise OperatorRepairError("invalid train link/restart count")
    fresh = index == 1 and restart_count == 0
    return {
        "index": index,
        "fresh": fresh,
        "has_latest": has_latest,
        "resume": "never" if fresh else ("must" if has_latest else "allow"),
        "require_endpoint": index == TRAIN_LINKS,
    }


def _materialize_run_plan_copy(
    *, run_dir: Path, plan_path: Path, policy: Mapping[str, Any],
) -> Path:
    plan_copy = run_dir / "operator_repair_plan.json"

    def _validated() -> Path:
        if (
            not plan_copy.is_file() or plan_copy.is_symlink()
            or sha256_file(plan_copy) != sha256_file(plan_path)
        ):
            raise OperatorRepairError("run-local plan differs from submission plan")
        return plan_copy

    if policy["fresh"]:
        if run_dir.exists():
            raise OperatorRepairError(f"fresh run_dir already exists: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=False)
        common._exclusive_text_write(plan_copy, plan_path.read_text())
        return _validated()

    step0_train01_recovery = policy["index"] == 1 and not policy["has_latest"]
    if not run_dir.exists():
        if not step0_train01_recovery:
            raise OperatorRepairError("resumable link lacks its run directory")
        run_dir.mkdir(parents=True, exist_ok=False)
        common._exclusive_text_write(plan_copy, plan_path.read_text())
        return _validated()
    if not run_dir.is_dir() or run_dir.is_symlink():
        raise OperatorRepairError("resumable run path is not a real directory")
    if not plan_copy.exists():
        if not step0_train01_recovery or any(run_dir.iterdir()):
            raise OperatorRepairError("resumable link lacks its run-local immutable plan")
        common._exclusive_text_write(plan_copy, plan_path.read_text())
    return _validated()


def _stage_train(plan: Mapping[str, Any], stage: str, plan_path: Path) -> int:
    try:
        restart_count = int(os.environ.get("SLURM_RESTART_COUNT", "0"))
    except ValueError as exc:
        raise OperatorRepairError("SLURM_RESTART_COUNT must be an integer") from exc
    run_dir = Path(plan["lineage"]["run_dir"])
    policy = _train_stage_policy(
        stage, restart_count, has_latest=(run_dir / "LATEST").is_file(),
    )
    plan_copy = _materialize_run_plan_copy(
        run_dir=run_dir, plan_path=plan_path, policy=policy,
    )
    if sha256_file(plan_copy) != sha256_file(plan_path):
        raise OperatorRepairError("run-local plan differs from submission plan")
    if (run_dir / "STOP").exists():
        raise OperatorRepairError("STOP exists; fixed operator-repair chain never ignores it")
    _reject_training_asset_failure(plan)
    _reject_durable_execution_failure(plan)

    _, health_path = _materialize_training_bootstrap(
        plan, run_dir=run_dir,
        allow_create_missing=policy["fresh"] or not policy["has_latest"],
    )
    _publish_training_asset_verification(plan, stage=stage, phase="pre")
    _, pre_asset_receipt_sha = _read_training_asset_verification(
        plan, stage=stage, phase="pre",
    )

    # Requeue/link idempotence: once the exact endpoint exists, all remaining
    # training links are authenticated zero-step no-ops.
    if (run_dir / "LATEST").is_file() and _latest_step(run_dir) == FIXED_STEP:
        _require_nonterminal_wandb_health(
            health_path, lineage_sha256=_plan_sha(),
        )
        _run_config_identity(plan)
        common._checkpoint_shards(run_dir, FIXED_STEP, WORLD_SIZE)
        _metrics_receipt(run_dir)
        pending = _begin_training_asset_post_transaction(
            plan, stage=stage, restart_count=restart_count,
            starting_checkpoint_step=FIXED_STEP,
            pre_receipt_sha256=pre_asset_receipt_sha,
        )
        _publish_post_training_asset_verification(
            plan, pending=pending,
        )
        print(f"[operator-repair] {stage}: exact step {FIXED_STEP} already complete", flush=True)
        return 0

    from loom.train import wandb_util  # noqa: PLC0415

    expected_id = str(plan["wandb"]["training_run_id"])
    wandb_id_path = run_dir / "wandb_id"
    if policy["fresh"]:
        os.environ["WANDB_RUN_ID"] = expected_id
    materialized_id = wandb_util.stable_run_id(run_dir)
    if materialized_id != expected_id:
        raise OperatorRepairError("materialized W&B run id differs from plan")

    env = dict(os.environ)
    env.update({
        "WANDB_RUN_ID": materialized_id,
        "WANDB_RESUME": str(policy["resume"]),
        "WANDB_DIR": str(run_dir),
        "WANDB_MODE": "online",
        "WANDB_RUN_GROUP": plan["wandb"]["group"],
        "WANDB_JOB_TYPE": plan["wandb"]["training_job_type"],
        "WANDB_TAGS": ",".join(EXPECTED_TRAIN_TAGS),
        "LOOM_WANDB_PROJECT": PROJECT,
        "LOOM_WANDB_GROUP": plan["wandb"]["group"],
        "LOOM_WANDB_JOB_TYPE": plan["wandb"]["training_job_type"],
        "LOOM_WANDB_TAGS": ",".join(EXPECTED_TRAIN_TAGS),
        "LOOM_WANDB_REQUIRE_ONLINE": "1",
        "LOOM_WANDB_RESUME": str(policy["resume"]),
        "LOOM_WANDB_HEALTH_STATE": str(health_path.resolve()),
        "LOOM_WANDB_LINEAGE_SHA256": _plan_sha(),
        "LOOM_WANDB_COMMITTED_STEP": str(
            _latest_step(run_dir) if (run_dir / "LATEST").is_file() else 0
        ),
        "LOOM_RESTART_COUNT": str(restart_count),
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "8",
    })
    train_command = [
        "python3", "scripts/r0_e2e_operator_repair_train_entry.py",
        *_training_argv(run_dir, include_link=True),
    ]
    inner = (
        'export RANK="$SLURM_PROCID" WORLD_SIZE="$SLURM_NTASKS" '
        'LOCAL_RANK="$SLURM_LOCALID"; exec ' + shlex.join(train_command)
    )
    starting_checkpoint_step = (
        _latest_step(run_dir) if (run_dir / "LATEST").is_file() else 0
    )
    training_error: BaseException | None = None
    try:
        subprocess.run(
            ["srun", "--kill-on-bad-exit=1", "bash", "-c", inner],
            cwd=ROOT, env=env, check=True,
        )
    except BaseException as error:  # includes scheduler interruption
        training_error = error
    # Start the fail-closed transaction only after srun returns. An ordinary
    # in-srun preemption therefore remains resumable; once the post-check begins,
    # however, PENDING without exact COMPLETE is terminal on every replay.
    pending = _begin_training_asset_post_transaction(
        plan, stage=stage, restart_count=restart_count,
        starting_checkpoint_step=starting_checkpoint_step,
        pre_receipt_sha256=pre_asset_receipt_sha,
    )
    try:
        _publish_post_training_asset_verification(
            plan, pending=pending,
        )
    except BaseException as verification_error:
        if training_error is not None:
            raise OperatorRepairError(
                "training failed and post-link asset verification also failed: "
                f"{type(verification_error).__name__}: {verification_error}"
            ) from training_error
        raise
    if training_error is not None:
        raise training_error
    if (run_dir / "STOP").exists():
        raise OperatorRepairError("training link produced/observed STOP")
    latest = _latest_step(run_dir)
    _run_config_identity(plan)
    if policy["require_endpoint"] and latest != FIXED_STEP:
        raise OperatorRepairError(
            f"final link ended at {latest}, exact endpoint {FIXED_STEP} is required"
        )
    if latest == FIXED_STEP:
        common._checkpoint_shards(run_dir, FIXED_STEP, WORLD_SIZE)
        _metrics_receipt(run_dir)
    return 0


def _wandb_publish_once(
    plan: Mapping[str, Any], *, stage: str, path: Path,
    artifact_type: str, summary: Mapping[str, Any],
) -> None:
    try:
        import wandb  # noqa: PLC0415
    except ImportError as exc:
        raise OperatorRepairError("online W&B is required but unavailable") from exc
    run_dir = Path(plan["lineage"]["control_dir"]) / "wandb" / stage
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(plan["wandb"]["stage_run_ids"][stage])
    run_id_path = run_dir / "wandb_id"
    if run_id_path.exists():
        if run_id_path.read_text().strip() != run_id:
            raise OperatorRepairError(f"W&B id changed for {stage}")
    else:
        common._exclusive_text_write(run_id_path, run_id + "\n")
    run = wandb.init(
        project=PROJECT, id=run_id,
        name=f"{plan['wandb']['group']}-{stage}",
        group=plan["wandb"]["group"], job_type=stage,
        tags=EXPECTED_STAGE_TAGS, resume="allow", mode="online", dir=str(run_dir),
        config={
            "operator_repair_plan_sha256": _plan_sha(),
            "fixed_endpoint": FIXED_STEP,
            "decision_gate": False,
            "stage": stage,
        },
        settings=wandb.Settings(init_timeout=90),
    )
    if bool(getattr(run, "offline", False)):
        run.finish()
        raise OperatorRepairError("W&B returned offline for required-online stage")
    try:
        for key, value in summary.items():
            run.summary[key] = value
        artifact = wandb.Artifact(
            name=f"{plan['wandb']['group']}-{stage}", type=artifact_type,
            metadata={**dict(summary), "sha256": sha256_file(path)},
        )
        artifact.add_file(str(path), name=path.name)
        run.log_artifact(artifact)
    finally:
        run.finish()


def _wandb_publish(
    plan: Mapping[str, Any], *, stage: str, path: Path,
    artifact_type: str, summary: Mapping[str, Any],
) -> None:
    errors: list[str] = []
    for attempt in range(1, WANDB_ATTEMPTS + 1):
        try:
            _wandb_publish_once(
                plan, stage=stage, path=path,
                artifact_type=artifact_type, summary=summary,
            )
            return
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")
            if attempt == WANDB_ATTEMPTS:
                raise OperatorRepairError(
                    "required online W&B publication failed: " + " | ".join(errors)
                ) from exc
            time.sleep(WANDB_RETRY_SECONDS)


def _publish_or_read_fixed_endpoint(plan: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    path = Path(plan["paths"]["fixed_endpoint"])
    expected = _fixed_endpoint_payload(plan)
    if path.exists():
        receipt = _read_receipt(path, kind="r0_e2e_operator_repair_fixed_endpoint")
        if receipt != expected:
            raise OperatorRepairError("existing fixed endpoint differs from recomputation")
    else:
        common.exclusive_json_write(path, expected)
    return expected, sha256_file(path)


def _checkpoint_receipt_payload(
    plan: Mapping[str, Any], *, endpoint: Mapping[str, Any], endpoint_sha: str,
    report: Path, checkpoint: Path, pinned: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "kind": "r0_e2e_operator_repair_consolidated_checkpoint_receipt",
        "plan_sha256": _plan_sha(),
        "step": FIXED_STEP,
        "selection": "predeclared_fixed_step_no_metric_or_eval_selection",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "verification_report": str(report),
        "verification_report_sha256": sha256_file(report),
        "fixed_endpoint_sha256": endpoint_sha,
        "run_config_sha256": endpoint["run_config"]["sha256"],
        "metrics_sha256": endpoint["metrics"]["sha256"],
        "source_checkpoint_shards": endpoint["checkpoint_shards"],
        "pinned_checkpoint_shards": dict(pinned),
        "checkpoint_bytes_uploaded_to_wandb": False,
        "metric_or_eval_gate": False,
    }


def _validate_checkpoint_receipt(
    plan: Mapping[str, Any], *, rehash_pinned: bool = False,
) -> dict[str, Any]:
    _reject_training_asset_failure(plan)
    _reject_durable_execution_failure(plan)
    endpoint, endpoint_sha = _read_fixed_endpoint(
        plan, rehash_source=rehash_pinned,
    )
    path = Path(plan["paths"]["checkpoint_receipt"])
    receipt = _read_receipt(
        path, kind="r0_e2e_operator_repair_consolidated_checkpoint_receipt",
    )
    checkpoint = Path(plan["paths"]["checkpoint"])
    report = Path(plan["paths"]["checkpoint_report"])
    pinned_dir = checkpoint.parent / f"shards_{FIXED_STEP:09d}"
    if rehash_pinned:
        try:
            pinned = common._checkpoint_shard_receipt(
                common._checkpoint_shards(
                    pinned_dir, FIXED_STEP, WORLD_SIZE, require_latest=False,
                )
            )
        except common.ChainError as exc:
            raise OperatorRepairError(str(exc)) from exc
    else:
        pinned = receipt.get("pinned_checkpoint_shards")
        if not isinstance(pinned, Mapping):
            raise OperatorRepairError("checkpoint receipt omitted pinned shard identities")
    try:
        verification = json.loads(report.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise OperatorRepairError("checkpoint verification report is unreadable") from exc
    if verification.get("pass") is not True:
        raise OperatorRepairError("checkpoint verification report did not pass")
    if not checkpoint.is_file() or checkpoint.is_symlink():
        raise OperatorRepairError("consolidated checkpoint is absent/non-regular")
    expected = _checkpoint_receipt_payload(
        plan, endpoint=endpoint, endpoint_sha=endpoint_sha,
        report=report, checkpoint=checkpoint, pinned=pinned,
    )
    if receipt != expected or pinned != endpoint["checkpoint_shards"]:
        raise OperatorRepairError("checkpoint receipt differs from exact recomputation")
    return receipt


def _stage_consolidate(plan: Mapping[str, Any]) -> int:
    _reject_training_asset_failure(plan)
    _reject_durable_execution_failure(plan)
    endpoint, endpoint_sha = _publish_or_read_fixed_endpoint(plan)
    checkpoint = Path(plan["paths"]["checkpoint"])
    report = Path(plan["paths"]["checkpoint_report"])
    receipt_path = Path(plan["paths"]["checkpoint_receipt"])
    if receipt_path.exists():
        receipt = _validate_checkpoint_receipt(plan, rehash_pinned=True)
        _wandb_publish(
            plan, stage="consolidate", path=receipt_path,
            artifact_type="fixed-endpoint-checkpoint-receipt",
            summary={
                "checkpoint_step": FIXED_STEP,
                "fixed_endpoint": True,
                "metric_gate": False,
            },
        )
        return 0
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    attempt_report = report.with_name(
        f".{report.name}.attempt-{os.getpid()}-{uuid.uuid4().hex}"
    )
    command = [
        sys.executable, "-m", "loom.train.consolidate",
        "--run_dir", plan["lineage"]["run_dir"],
        "--step", str(FIXED_STEP),
        "--out", str(checkpoint),
        "--config", str(Path(plan["lineage"]["run_dir"]) / "config.json"),
        "--pin", "--report", str(attempt_report),
    ]
    if checkpoint.exists():
        command.append("--verify_only")
    try:
        subprocess.run(command, cwd=ROOT, check=True)
        try:
            verification = json.loads(attempt_report.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise OperatorRepairError("attempt verification report is unreadable") from exc
        if verification.get("pass") is not True:
            raise OperatorRepairError("checkpoint consolidation verification failed")
        if report.exists():
            if not report.is_file() or report.read_bytes() != attempt_report.read_bytes():
                raise OperatorRepairError("existing checkpoint report differs from recomputation")
        else:
            common._exclusive_text_write(report, attempt_report.read_text())
    finally:
        attempt_report.unlink(missing_ok=True)
    pinned_dir = checkpoint.parent / f"shards_{FIXED_STEP:09d}"
    try:
        pinned = common._checkpoint_shard_receipt(
            common._checkpoint_shards(
                pinned_dir, FIXED_STEP, WORLD_SIZE, require_latest=False,
            )
        )
    except common.ChainError as exc:
        raise OperatorRepairError(str(exc)) from exc
    if pinned != endpoint["checkpoint_shards"]:
        raise OperatorRepairError("pinned shards differ from fixed endpoint source shards")
    endpoint_after, endpoint_sha_after = _read_fixed_endpoint(
        plan, rehash_source=True,
    )
    if endpoint_after != endpoint or endpoint_sha_after != endpoint_sha:
        raise OperatorRepairError("fixed endpoint changed during consolidation")
    receipt = _checkpoint_receipt_payload(
        plan, endpoint=endpoint, endpoint_sha=endpoint_sha,
        report=report, checkpoint=checkpoint, pinned=pinned,
    )
    common.exclusive_json_write(receipt_path, receipt)
    _validate_checkpoint_receipt(plan, rehash_pinned=False)
    _wandb_publish(
        plan, stage="consolidate", path=receipt_path,
        artifact_type="fixed-endpoint-checkpoint-receipt",
        summary={"checkpoint_step": FIXED_STEP, "fixed_endpoint": True, "metric_gate": False},
    )
    return 0


def _eval_command(
    plan: Mapping[str, Any], *, seed: int, checkpoint: Path, out_dir: Path,
) -> tuple[list[str], dict[str, str]]:
    evaluation = plan["evaluation"]
    command = [
        str(LIBERO_EVAL_PYTHON), "-m", "loom.eval",
        "--bench", "libero", "--backend", "libero",
        "--embodiment", "libero_franka",
        "--ckpt", str(checkpoint), "--require-real", "--op-stats",
        "--suites", ",".join(evaluation["suites"]),
        "--n-tasks", str(evaluation["tasks_per_suite"]),
        "--episodes-per-task", str(evaluation["episodes_per_task"]),
        "--max-steps", str(evaluation["max_steps"]),
        "--seeds", str(seed), "--workers", str(evaluation["workers"]),
        "--gripper-dwell", str(evaluation["gripper_dwell"]),
        "--decoder-samples", str(evaluation["decoder_samples"]),
        "--row-label", "**LOOM · R0 operator repair**",
        "--out", str(out_dir / "results.json"),
        "--md", str(out_dir / "table.md"),
    ]
    runtime_dir = out_dir / "runtime"
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1",
        "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl",
        "MUJOCO_EGL_DEVICE_ID": "0", "HF_HOME": str(DEFAULT_HF_HOME),
        "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "TRANSFORMERS_VERBOSITY": "error", "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "8",
        "TRITON_CACHE_DIR": str(runtime_dir / "triton_cache"),
        **plan["evaluation"]["environment"]["child_environment"],
    })
    return command, env


def _validate_seed_result(
    plan: Mapping[str, Any], *, seed: int, result_path: Path,
) -> tuple[dict[str, Any], Any]:
    from loom.eval import EpisodeResult  # noqa: PLC0415
    from loom.eval.runner import aggregate  # noqa: PLC0415

    checkpoint_receipt = _validate_checkpoint_receipt(plan)
    checkpoint = Path(checkpoint_receipt["checkpoint"])
    if sha256_file(checkpoint) != checkpoint_receipt["checkpoint_sha256"]:
        raise OperatorRepairError("checkpoint bytes changed before/during evaluation")
    try:
        blob = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise OperatorRepairError(f"seed {seed} result is unreadable") from exc
    if not isinstance(blob, Mapping):
        raise OperatorRepairError(f"seed {seed} result is not an object")
    blob = dict(blob)
    try:
        protocol = common._result_protocol(blob)
        rows = common._validate_exact_eval_blob(
            blob, seed=seed, label=f"operator-repair seed {seed}",
            identity_profile="current_candidate",
        )
        common._validate_eval_method_identity(
            blob, label=f"operator-repair seed {seed}",
            identity_profile="current_candidate", checkpoint_step=FIXED_STEP,
            checkpoint_path=str(checkpoint.resolve()),
        )
    except common.ChainError as exc:
        raise OperatorRepairError(str(exc)) from exc
    actual = {
        "bench": protocol.bench, "seeds": protocol.seeds,
        "suites": protocol.suites, "n_tasks": protocol.n_tasks,
        "episodes_per_task": protocol.episodes_per_task,
        "max_steps": protocol.max_steps,
    }
    expected = {
        "bench": "libero", "seeds": (seed,),
        "suites": tuple(plan["evaluation"]["suites"]), "n_tasks": 10,
        "episodes_per_task": 10, "max_steps": 512,
    }
    if actual != expected or len(rows) != common.EXPECTED_EPISODES_PER_SEED:
        raise OperatorRepairError(f"seed {seed} protocol/result count changed")
    oracle = aggregate(
        [EpisodeResult.from_dict(dict(rows[key])) for key in sorted(rows)], protocol,
    )
    if blob.get("summary") != oracle:
        raise OperatorRepairError(f"seed {seed} summary differs from exact episode oracle")
    return blob, protocol


def _seed_markdown(blob: Mapping[str, Any]) -> str:
    from loom.eval.table import render_report  # noqa: PLC0415

    return render_report(blob, row_label="**LOOM · R0 operator repair**")


def _prepare_eval_runtime(out_dir: Path) -> Path:
    _reject_existing_symlink_components(out_dir)
    if out_dir.exists() and (not out_dir.is_dir() or out_dir.is_symlink()):
        raise OperatorRepairError("evaluation output path is not a real directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime = out_dir / "runtime"
    triton = runtime / "triton_cache"
    for path in (runtime, triton):
        _reject_existing_symlink_components(path)
        if path.exists() and (not path.is_dir() or path.is_symlink()):
            raise OperatorRepairError("evaluation runtime path is not a real directory")
        path.mkdir(parents=True, exist_ok=True)
    return runtime


def _eval_attempt_paths(out_dir: Path) -> tuple[Path, Path]:
    return out_dir / "active_attempt.json", out_dir / "completed_attempt.json"


def _eval_attempt_ordinal(out_dir: Path) -> int:
    recovery = out_dir / "recovery"
    if not recovery.exists():
        return 1
    if not recovery.is_dir() or recovery.is_symlink():
        raise OperatorRepairError("evaluation recovery path is not a real directory")
    return 1 + sum(
        path.name.startswith("active_attempt.json.") for path in recovery.iterdir()
    )


def _eval_attempt_payload(
    plan: Mapping[str, Any], *, seed: int,
    environment: Mapping[str, Any], checkpoint_receipt: Mapping[str, Any],
    ordinal: int,
) -> dict[str, Any]:
    receipt_path = Path(plan["paths"]["checkpoint_receipt"])
    out_dir = Path(plan["paths"]["eval"][str(seed)]["out_dir"])
    return {
        "format_version": FORMAT_VERSION,
        "kind": "r0_e2e_operator_repair_eval_attempt_v1",
        "plan_sha256": _plan_sha(), "seed": seed, "ordinal": ordinal,
        "environment": copy.deepcopy(environment),
        "environment_sha256": hashlib.sha256(
            _canonical_json(environment).encode()
        ).hexdigest(),
        "checkpoint_receipt_sha256": sha256_file(receipt_path),
        "checkpoint": checkpoint_receipt["checkpoint"],
        "checkpoint_sha256": checkpoint_receipt["checkpoint_sha256"],
        "checkpoint_bytes": checkpoint_receipt["checkpoint_bytes"],
        "result": str(out_dir / "results.json"),
        "table": str(out_dir / "table.md"),
    }


def _ensure_eval_attempt(
    plan: Mapping[str, Any], *, seed: int, out_dir: Path,
    environment: Mapping[str, Any], checkpoint_receipt: Mapping[str, Any],
    create: bool,
) -> dict[str, Any]:
    active, _ = _eval_attempt_paths(out_dir)
    expected = _eval_attempt_payload(
        plan, seed=seed, environment=environment,
        checkpoint_receipt=checkpoint_receipt,
        ordinal=_eval_attempt_ordinal(out_dir),
    )
    if not active.exists():
        if not create:
            raise EvalAttemptMismatch("evaluation result has no authenticated attempt")
        common.exclusive_json_write(active, expected)
    if not active.is_file() or active.is_symlink():
        raise OperatorRepairError("evaluation attempt receipt is nonregular")
    try:
        actual = json.loads(active.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise OperatorRepairError("evaluation attempt receipt is unreadable") from exc
    if actual != expected:
        if isinstance(actual, Mapping) and actual.get("kind") == expected["kind"]:
            raise EvalAttemptMismatch(
                "evaluation attempt environment/checkpoint differs from current"
            )
        raise OperatorRepairError("evaluation attempt receipt is malformed")
    return expected


def _completed_eval_attempt_payload(
    active: Mapping[str, Any], *, result: Path,
) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "kind": "r0_e2e_operator_repair_completed_eval_attempt_v1",
        "active_attempt_sha256": hashlib.sha256(
            _canonical_json(active).encode()
        ).hexdigest(),
        "seed": active["seed"], "ordinal": active["ordinal"],
        "environment_sha256": active["environment_sha256"],
        "checkpoint_receipt_sha256": active["checkpoint_receipt_sha256"],
        "checkpoint_sha256": active["checkpoint_sha256"],
        "result_sha256": sha256_file(result),
        "result_bytes": result.stat().st_size,
        "post_environment_and_checkpoint_reauthenticated": True,
    }


def _publish_or_validate_completed_eval_attempt(
    out_dir: Path, *, active: Mapping[str, Any], result: Path,
    create: bool,
) -> dict[str, Any]:
    _, completed = _eval_attempt_paths(out_dir)
    expected = _completed_eval_attempt_payload(active, result=result)
    if not completed.exists():
        if not create:
            raise EvalAttemptMismatch(
                "complete evaluation lacks post-attempt authentication"
            )
        common.exclusive_json_write(completed, expected)
    if not completed.is_file() or completed.is_symlink():
        raise OperatorRepairError("completed evaluation attempt is nonregular")
    try:
        actual = json.loads(completed.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise OperatorRepairError("completed evaluation attempt is unreadable") from exc
    if actual != expected:
        raise OperatorRepairError("completed evaluation attempt differs from result")
    return expected


def _quarantine_eval_file(path: Path, *, reason: str) -> Path:
    from loom.train import atomic as atomic_mod  # noqa: PLC0415

    if not path.is_file() or path.is_symlink():
        raise OperatorRepairError(f"cannot quarantine nonregular evaluation file: {path}")
    digest = sha256_file(path)
    recovery = path.parent / "recovery"
    _reject_existing_symlink_components(recovery)
    if recovery.exists() and (not recovery.is_dir() or recovery.is_symlink()):
        raise OperatorRepairError("evaluation recovery path is not a real directory")
    recovery.mkdir(parents=True, exist_ok=True)
    target = recovery / f"{path.name}.{reason}.sha256-{digest}"
    if target.exists():
        if (
            not target.is_file() or target.is_symlink()
            or sha256_file(target) != digest
            or target.stat().st_size != path.stat().st_size
        ):
            raise OperatorRepairError("evaluation quarantine collision changed bytes")
    else:
        try:
            os.link(path, target)
        except FileExistsError:
            if not target.is_file() or target.is_symlink() or sha256_file(target) != digest:
                raise OperatorRepairError("evaluation quarantine publish raced")
        atomic_mod.fsync_dir(recovery)
    path.unlink()
    atomic_mod.fsync_dir(path.parent)
    return target


def _quarantine_eval_attempt_outputs(out_dir: Path, *, reason: str) -> None:
    active, completed = _eval_attempt_paths(out_dir)
    for path in (out_dir / "results.json", out_dir / "table.md", completed, active):
        if path.exists():
            _quarantine_eval_file(path, reason=reason)


def _eval_recovery_receipt(out_dir: Path) -> dict[str, Any]:
    recovery = out_dir / "recovery"
    if not recovery.exists():
        return {"kind": "content_addressed_eval_recovery_v1", "files": []}
    if not recovery.is_dir() or recovery.is_symlink():
        raise OperatorRepairError("evaluation recovery path is not a real directory")
    files = []
    for path in sorted(recovery.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.is_symlink():
            raise OperatorRepairError("evaluation recovery contains nonregular entry")
        files.append({
            "name": path.name, "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return {"kind": "content_addressed_eval_recovery_v1", "files": files}


def _inspect_seed_result_for_resume(
    plan: Mapping[str, Any], *, seed: int, result_path: Path,
) -> tuple[str, dict[str, Any]]:
    from loom.eval import EpisodeResult  # noqa: PLC0415
    from loom.eval.runner import iter_work  # noqa: PLC0415

    checkpoint_receipt = _validate_checkpoint_receipt(plan)
    checkpoint = Path(checkpoint_receipt["checkpoint"]).resolve()
    try:
        blob = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise OperatorRepairError(f"seed {seed} resumable result is unreadable") from exc
    if not isinstance(blob, Mapping):
        raise OperatorRepairError(f"seed {seed} resumable result is not an object")
    blob = dict(blob)
    try:
        protocol = common._result_protocol(blob)
        common._validate_eval_method_identity(
            blob, label=f"operator-repair seed {seed} partial",
            identity_profile="current_candidate", checkpoint_step=FIXED_STEP,
            checkpoint_path=str(checkpoint),
        )
    except common.ChainError as exc:
        raise OperatorRepairError(str(exc)) from exc
    actual_protocol = {
        "bench": protocol.bench, "seeds": protocol.seeds,
        "suites": protocol.suites, "n_tasks": protocol.n_tasks,
        "episodes_per_task": protocol.episodes_per_task,
        "max_steps": protocol.max_steps,
    }
    expected_protocol = {
        "bench": "libero", "seeds": (seed,),
        "suites": tuple(plan["evaluation"]["suites"]), "n_tasks": 10,
        "episodes_per_task": 10, "max_steps": 512,
    }
    if actual_protocol != expected_protocol:
        raise OperatorRepairError(f"seed {seed} partial protocol changed")
    episode_rows = blob.get("episodes")
    if not isinstance(episode_rows, list) or len(episode_rows) > 400:
        raise OperatorRepairError(f"seed {seed} partial episodes are malformed")
    expected_work = {item.key(): item for item in iter_work(protocol)}
    seen = set()
    error_rows = 0
    for index, row in enumerate(episode_rows):
        if not isinstance(row, Mapping):
            raise OperatorRepairError(f"seed {seed} partial row {index} is malformed")
        try:
            record = EpisodeResult.from_dict(dict(row))
        except (TypeError, ValueError) as exc:
            raise OperatorRepairError(
                f"seed {seed} partial row {index} cannot be decoded"
            ) from exc
        key = record.key()
        work = expected_work.get(key)
        if key in seen or work is None:
            raise OperatorRepairError(f"seed {seed} partial key changed: {key}")
        if (
            record.env_seed != work.env_seed
            or (row.get("extra") or {}).get("policy_seed") != work.policy_seed
            or not isinstance(row.get("success"), bool)
        ):
            raise OperatorRepairError(f"seed {seed} partial RNG/outcome changed: {key}")
        seen.add(key)
        error_rows += int(record.error is not None)
    if error_rows:
        return "ERROR_ROWS", blob
    if len(seen) < common.EXPECTED_EPISODES_PER_SEED:
        return "RESUMABLE_PARTIAL", blob
    exact, _ = _validate_seed_result(plan, seed=seed, result_path=result_path)
    return "COMPLETE", exact


def _ensure_seed_table(
    table: Path, blob: Mapping[str, Any], *, allow_create: bool,
    recover_corrupt: bool = False,
) -> dict[str, Any]:
    from loom.train import atomic as atomic_mod  # noqa: PLC0415

    expected = _seed_markdown(blob)
    encoded = expected.encode("utf-8")
    if table.exists():
        if not table.is_file() or table.is_symlink():
            raise OperatorRepairError("seed table is not a regular file")
        if table.read_bytes() == encoded:
            return {"action": "NONE", "quarantine": None}
        if not recover_corrupt:
            raise OperatorRepairError("seed table differs from authenticated result")
        quarantine = _quarantine_eval_file(table, reason="corrupt_table")
        atomic_mod.atomic_write_bytes(table, encoded)
        return {"action": "RECOVERED_CORRUPT", "quarantine": str(quarantine)}
    if allow_create:
        table.parent.mkdir(parents=True, exist_ok=True)
        atomic_mod.atomic_write_bytes(table, encoded)
        return {"action": "CREATED", "quarantine": None}
    raise OperatorRepairError("seed receipt references missing table")


def _eval_receipt_payload(
    plan: Mapping[str, Any], *, seed: int, blob: Mapping[str, Any],
    result: Path, table: Path, checkpoint_receipt: Mapping[str, Any],
    environment_receipt: Mapping[str, Any], active_attempt: Mapping[str, Any],
    completed_attempt: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "kind": "r0_e2e_operator_repair_single_seed_eval_receipt",
        "plan_sha256": _plan_sha(), "seed": seed,
        "result": str(result), "result_sha256": sha256_file(result),
        "table": str(table), "table_sha256": sha256_file(table),
        "episodes": common.EXPECTED_EPISODES_PER_SEED,
        "errors": 0, "avg": blob["summary"]["avg"],
        "checkpoint_step": FIXED_STEP,
        "checkpoint_sha256": checkpoint_receipt["checkpoint_sha256"],
        "checkpoint_receipt_sha256": sha256_file(plan["paths"]["checkpoint_receipt"]),
        "evaluation_environment": copy.deepcopy(environment_receipt),
        "evaluation_environment_sha256": hashlib.sha256(
            _canonical_json(environment_receipt).encode()
        ).hexdigest(),
        "active_attempt": copy.deepcopy(active_attempt),
        "completed_attempt": copy.deepcopy(completed_attempt),
        "recovery": _eval_recovery_receipt(result.parent),
        "evaluation_was_unconditional_on_health": True,
    }


def _validate_eval_receipt(plan: Mapping[str, Any], seed: int) -> dict[str, Any]:
    environment_before = _eval_environment_receipt()
    if environment_before != plan["evaluation"]["environment"]:
        raise OperatorRepairError("LIBERO evaluation environment differs from plan")
    path = Path(plan["paths"]["eval"][str(seed)]["receipt"])
    receipt = _read_receipt(path, kind="r0_e2e_operator_repair_single_seed_eval_receipt")
    out_dir = Path(plan["paths"]["eval"][str(seed)]["out_dir"])
    result, table = out_dir / "results.json", out_dir / "table.md"
    checkpoint_before = _validate_checkpoint_receipt(plan)
    active = _ensure_eval_attempt(
        plan, seed=seed, out_dir=out_dir, environment=environment_before,
        checkpoint_receipt=checkpoint_before, create=False,
    )
    blob, _ = _validate_seed_result(plan, seed=seed, result_path=result)
    completed = _publish_or_validate_completed_eval_attempt(
        out_dir, active=active, result=result, create=False,
    )
    _ensure_seed_table(table, blob, allow_create=False)
    environment_after = _eval_environment_receipt()
    if environment_after != environment_before:
        raise OperatorRepairError("LIBERO evaluation environment changed during adoption")
    checkpoint_after = _validate_checkpoint_receipt(plan)
    if checkpoint_after != checkpoint_before:
        raise OperatorRepairError("checkpoint changed during evaluation adoption")
    expected = _eval_receipt_payload(
        plan, seed=seed, blob=blob, result=result, table=table,
        checkpoint_receipt=checkpoint_after,
        environment_receipt=environment_after,
        active_attempt=active, completed_attempt=completed,
    )
    if receipt != expected:
        raise OperatorRepairError(f"seed {seed} receipt differs from recomputation")
    return receipt


def _stage_eval(plan: Mapping[str, Any], stage: str) -> int:
    _reject_training_asset_failure(plan)
    _reject_durable_execution_failure(plan)
    match = re.fullmatch(r"eval_seed([0-9]+)", stage)
    if match is None or int(match.group(1)) not in SEEDS:
        raise OperatorRepairError(f"invalid eval stage {stage!r}")
    seed = int(match.group(1))
    environment_before = _eval_environment_receipt()
    if environment_before != plan["evaluation"]["environment"]:
        raise OperatorRepairError("LIBERO evaluation environment differs from plan")
    checkpoint_receipt = _validate_checkpoint_receipt(plan)
    checkpoint = Path(checkpoint_receipt["checkpoint"])
    out_dir = Path(plan["paths"]["eval"][str(seed)]["out_dir"])
    receipt_path = Path(plan["paths"]["eval"][str(seed)]["receipt"])
    result, table = out_dir / "results.json", out_dir / "table.md"
    if receipt_path.exists():
        receipt = _validate_eval_receipt(plan, seed)
        _wandb_publish(
            plan, stage=f"eval-seed-{seed}", path=result,
            artifact_type="operator-repair-evaluation-results",
            summary={
                "seed": seed, "episodes": common.EXPECTED_EPISODES_PER_SEED,
                "success_rate": receipt["avg"], "checkpoint_step": FIXED_STEP,
                "evaluation_unconditional": True,
            },
        )
        return 0
    runtime_dir = _prepare_eval_runtime(out_dir)
    if result.is_symlink() or table.is_symlink():
        raise OperatorRepairError("evaluation outputs must not be symlinks")
    blob: dict[str, Any] | None = None
    active_attempt: dict[str, Any] | None = None
    completed_attempt: dict[str, Any] | None = None
    process_attempts = 0
    while blob is None:
        state = "MISSING"
        if result.exists():
            if not result.is_file() or result.is_symlink():
                raise OperatorRepairError("evaluation result is not a regular file")
            try:
                active_attempt = _ensure_eval_attempt(
                    plan, seed=seed, out_dir=out_dir,
                    environment=environment_before,
                    checkpoint_receipt=checkpoint_receipt, create=False,
                )
            except EvalAttemptMismatch:
                _quarantine_eval_attempt_outputs(
                    out_dir, reason="attempt_identity_mismatch",
                )
                state = "MISSING"
            else:
                state, inspected = _inspect_seed_result_for_resume(
                    plan, seed=seed, result_path=result,
                )
                if state == "COMPLETE":
                    try:
                        completed_attempt = _publish_or_validate_completed_eval_attempt(
                            out_dir, active=active_attempt, result=result,
                            create=False,
                        )
                    except EvalAttemptMismatch:
                        _quarantine_eval_attempt_outputs(
                            out_dir, reason="missing_post_authentication",
                        )
                        state = "MISSING"
                    else:
                        blob = inspected
                        break
                if state == "ERROR_ROWS":
                    _quarantine_eval_attempt_outputs(
                        out_dir, reason="error_rows",
                    )
                    state = "MISSING"
        else:
            active_path, completed_path = _eval_attempt_paths(out_dir)
            if active_path.exists() or completed_path.exists() or table.exists():
                _quarantine_eval_attempt_outputs(
                    out_dir, reason="incomplete_quarantine_replay",
                )
        if process_attempts >= EVAL_MAX_PROCESS_ATTEMPTS:
            raise OperatorRepairError(
                f"seed {seed} exceeded {EVAL_MAX_PROCESS_ATTEMPTS} eval attempts"
            )
        command, env = _eval_command(
            plan, seed=seed, checkpoint=checkpoint, out_dir=out_dir,
        )
        process_attempts += 1
        attempt_environment_before = _eval_environment_receipt()
        if attempt_environment_before != environment_before:
            raise OperatorRepairError(
                "LIBERO evaluation environment changed before process attempt"
            )
        attempt_checkpoint_before = _validate_checkpoint_receipt(plan)
        if attempt_checkpoint_before != checkpoint_receipt:
            raise OperatorRepairError("checkpoint changed before evaluation attempt")
        active_attempt = _ensure_eval_attempt(
            plan, seed=seed, out_dir=out_dir,
            environment=attempt_environment_before,
            checkpoint_receipt=attempt_checkpoint_before, create=True,
        )
        process_error: subprocess.CalledProcessError | None = None
        try:
            subprocess.run(command, cwd=runtime_dir, env=env, check=True)
        except subprocess.CalledProcessError as error:
            process_error = error
        try:
            attempt_environment_after = _eval_environment_receipt()
            attempt_checkpoint_after = _validate_checkpoint_receipt(plan)
            if (
                attempt_environment_after != attempt_environment_before
                or attempt_checkpoint_after != attempt_checkpoint_before
            ):
                raise EvalAttemptMismatch(
                    "evaluation environment/checkpoint changed during process attempt"
                )
            _ensure_eval_attempt(
                plan, seed=seed, out_dir=out_dir,
                environment=attempt_environment_after,
                checkpoint_receipt=attempt_checkpoint_after, create=False,
            )
        except BaseException as authentication_error:
            try:
                _quarantine_eval_attempt_outputs(
                    out_dir, reason="post_attempt_authentication_failure",
                )
            except BaseException as quarantine_error:
                raise OperatorRepairError(
                    "evaluation authentication failed and contaminated outputs "
                    "could not be quarantined"
                ) from authentication_error
            raise
        if process_error is not None:
            if result.is_file() and not result.is_symlink():
                failed_state, _ = _inspect_seed_result_for_resume(
                    plan, seed=seed, result_path=result,
                )
                if failed_state == "ERROR_ROWS":
                    _quarantine_eval_attempt_outputs(
                        out_dir, reason="error_rows",
                    )
                    continue
            raise process_error
        if not result.is_file() or result.is_symlink():
            raise OperatorRepairError("evaluation process returned without a result")
        state, inspected = _inspect_seed_result_for_resume(
            plan, seed=seed, result_path=result,
        )
        if state == "ERROR_ROWS":
            _quarantine_eval_attempt_outputs(out_dir, reason="error_rows")
            continue
        if state == "COMPLETE":
            completed_attempt = _publish_or_validate_completed_eval_attempt(
                out_dir, active=active_attempt, result=result, create=True,
            )
            blob = inspected
            break
    _ensure_seed_table(
        table, blob, allow_create=True, recover_corrupt=True,
    )
    try:
        environment_after = _eval_environment_receipt()
        checkpoint_after = _validate_checkpoint_receipt(plan)
        if (
            environment_after != environment_before
            or checkpoint_after != checkpoint_receipt
        ):
            raise EvalAttemptMismatch(
                "evaluation environment/checkpoint changed before publication"
            )
        if not isinstance(active_attempt, Mapping) or not isinstance(
            completed_attempt, Mapping
        ):
            raise OperatorRepairError(
                "complete evaluation lacks authenticated attempt closure"
            )
        _ensure_eval_attempt(
            plan, seed=seed, out_dir=out_dir,
            environment=environment_after,
            checkpoint_receipt=checkpoint_after, create=False,
        )
        _publish_or_validate_completed_eval_attempt(
            out_dir, active=active_attempt, result=result, create=False,
        )
    except BaseException as authentication_error:
        try:
            _quarantine_eval_attempt_outputs(
                out_dir, reason="pre_publication_authentication_failure",
            )
        except BaseException as quarantine_error:
            raise OperatorRepairError(
                "pre-publication authentication failed and contaminated outputs "
                "could not be quarantined"
            ) from authentication_error
        raise
    receipt = _eval_receipt_payload(
        plan, seed=seed, blob=blob, result=result, table=table,
        checkpoint_receipt=checkpoint_after,
        environment_receipt=environment_after,
        active_attempt=active_attempt,
        completed_attempt=completed_attempt,
    )
    common.exclusive_json_write(receipt_path, receipt)
    _validate_eval_receipt(plan, seed)
    _wandb_publish(
        plan, stage=f"eval-seed-{seed}", path=result,
        artifact_type="operator-repair-evaluation-results",
        summary={
            "seed": seed, "episodes": common.EXPECTED_EPISODES_PER_SEED,
            "success_rate": blob["summary"]["avg"], "checkpoint_step": FIXED_STEP,
            "evaluation_unconditional": True,
        },
    )
    return 0


def _descriptive_bootstrap_matrix(task_keys: Sequence[str]) -> tuple[Any, dict[str, Any]]:
    import torch  # noqa: PLC0415

    keys = tuple(str(key) for key in task_keys)
    if len(keys) != 40 or len(set(keys)) != 40:
        raise OperatorRepairError("paired bootstrap requires 40 unique task keys")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(common.BOOTSTRAP_SEED)
    blocks = []
    for suite in sorted(DEFAULT_LIBERO_SUITES):
        indices = [index for index, key in enumerate(keys) if key.startswith(f"{suite}/")]
        if len(indices) != 10:
            raise OperatorRepairError(f"paired bootstrap suite {suite} lacks 10 tasks")
        choices = torch.randint(
            0, 10, (common.BOOTSTRAP_SAMPLES, 10), generator=generator,
            dtype=torch.int64,
        )
        lookup = torch.tensor(indices, dtype=torch.int64)
        blocks.append(lookup.index_select(0, choices.reshape(-1)).reshape_as(choices))
    matrix = torch.cat(blocks, dim=1).contiguous()
    digest = hashlib.sha256(matrix.numpy().tobytes(order="C")).hexdigest()
    if digest != common.BOOTSTRAP_MATRIX_SHA256:
        raise OperatorRepairError(f"descriptive bootstrap matrix changed: {digest}")
    return matrix, {
        "kind": "fixed_suite_stratified_task_resample_matrix_v1",
        "shape": list(matrix.shape), "dtype": "int64", "byte_order": "native-little",
        "seed": common.BOOTSTRAP_SEED, "samples": common.BOOTSTRAP_SAMPLES,
        "suite_order": sorted(DEFAULT_LIBERO_SUITES),
        "tasks_per_suite_per_replicate": 10, "sha256": digest,
    }


def _descriptive_comparison(
    plan: Mapping[str, Any],
    candidate_rows: Mapping[tuple[str, str, int, int, int], Mapping[str, Any]],
) -> dict[str, Any]:
    import torch  # noqa: PLC0415

    baseline_rows = _validate_frozen_baseline_contract(
        plan.get("baseline_comparison")
    )
    if set(candidate_rows) != set(baseline_rows) or len(candidate_rows) != 1_200:
        raise OperatorRepairError("candidate/baseline episode pairing is not exact 1,200")
    task_differences: dict[str, list[int]] = {}
    candidate_successes = baseline_successes = 0
    new_only = old_only = tie_success = tie_failure = 0
    candidate_by_seed = {seed: 0 for seed in SEEDS}
    suites = {
        suite: {"candidate": 0, "baseline": 0, "episodes": 0}
        for suite in DEFAULT_LIBERO_SUITES
    }
    for key in sorted(candidate_rows):
        candidate, baseline = candidate_rows[key], baseline_rows[key]
        if (
            candidate.get("env_seed") != baseline.get("env_seed")
            or (candidate.get("extra") or {}).get("policy_seed")
            != (baseline.get("extra") or {}).get("policy_seed")
        ):
            raise OperatorRepairError(f"paired RNG identity differs for {key}")
        new, old = bool(candidate["success"]), bool(baseline["success"])
        _, suite, task_id, _, seed = key
        candidate_successes += int(new)
        baseline_successes += int(old)
        candidate_by_seed[seed] += int(new)
        suites[suite]["candidate"] += int(new)
        suites[suite]["baseline"] += int(old)
        suites[suite]["episodes"] += 1
        task_differences.setdefault(f"{suite}/task={task_id:02d}", []).append(
            int(new) - int(old)
        )
        if new and not old:
            new_only += 1
        elif old and not new:
            old_only += 1
        elif new:
            tie_success += 1
        else:
            tie_failure += 1
    if baseline_successes != common.BASELINE_SUCCESS_TOTAL:
        raise OperatorRepairError("paired baseline no longer has exact 447 successes")
    task_keys = sorted(task_differences)
    if len(task_keys) != 40 or any(len(task_differences[key]) != 30 for key in task_keys):
        raise OperatorRepairError("paired reduction is not exact 40 tasks x 30 episodes")
    task_rows = [
        {
            "task_key": key, "paired_episodes": 30,
            "delta_successes": sum(task_differences[key]),
            "delta_percentage_points": 100.0 * sum(task_differences[key]) / 30,
        }
        for key in task_keys
    ]
    values = torch.tensor(
        [row["delta_percentage_points"] for row in task_rows], dtype=torch.float64,
    )
    matrix, matrix_receipt = _descriptive_bootstrap_matrix(task_keys)
    draws = values.index_select(0, matrix.reshape(-1)).reshape(matrix.shape).mean(dim=1)
    lower = float(torch.quantile(draws, 0.025, interpolation="lower"))
    upper = float(torch.quantile(draws, 0.975, interpolation="higher"))
    exact_delta = 100.0 * (candidate_successes - baseline_successes) / 1_200
    if abs(float(values.mean()) - exact_delta) > 1.0e-12:
        raise OperatorRepairError("balanced task point differs from exact protocol delta")
    per_suite: dict[str, Any] = {}
    for suite, counts in suites.items():
        if counts["episodes"] != 300:
            raise OperatorRepairError(f"suite {suite} does not contain 300 pairs")
        candidate_rate = 100.0 * counts["candidate"] / 300
        baseline_rate = 100.0 * counts["baseline"] / 300
        per_suite[suite] = {
            "episodes": 300,
            "candidate_successes": counts["candidate"],
            "baseline_successes": counts["baseline"],
            "candidate_success_rate_percent": candidate_rate,
            "baseline_success_rate_percent": baseline_rate,
            "delta_percentage_points": candidate_rate - baseline_rate,
        }
    return {
        "role": "descriptive_only_never_controls_scheduler_or_eligibility",
        "available": True,
        "evaluation_already_completed_before_comparison": True,
        "fixed_endpoint_unchanged_by_result": True,
        "pairing": {
            "key_fields": ["bench", "suite", "task_id", "episode", "seed"],
            "paired_episodes": 1_200, "new_only": new_only, "old_only": old_only,
            "tie_success": tie_success, "tie_failure": tie_failure,
            "rng_identity_equal": True,
        },
        "overall": {
            "candidate_successes": candidate_successes,
            "baseline_successes": baseline_successes,
            "episodes": 1_200,
            "candidate_success_rate_percent": 100.0 * candidate_successes / 1_200,
            "baseline_success_rate_percent": 100.0 * baseline_successes / 1_200,
            "delta_percentage_points": exact_delta,
        },
        "per_suite": per_suite,
        "per_seed_candidate_successes": {
            str(seed): candidate_by_seed[seed] for seed in SEEDS
        },
        "task_deltas": task_rows,
        "task_deltas_sha256": hashlib.sha256(_canonical_json(task_rows).encode()).hexdigest(),
        "paired_task_bootstrap": {
            "method": "suite_stratified_40_task_paired_percentile_two_sided",
            "point_delta_percentage_points": exact_delta,
            "ci_low_percentage_points": lower,
            "ci_high_percentage_points": upper,
            "resample_matrix": matrix_receipt,
        },
    }


def merge_seed_results(plan: Mapping[str, Any]) -> dict[str, Any]:
    from loom.eval import EpisodeResult  # noqa: PLC0415
    from loom.eval.runner import aggregate, iter_work  # noqa: PLC0415

    blobs: list[dict[str, Any]] = []
    protocols = []
    receipts = []
    candidate_rows: dict[tuple[str, str, int, int, int], dict[str, Any]] = {}
    for seed in SEEDS:
        receipt = _validate_eval_receipt(plan, seed)
        receipts.append({
            "seed": seed,
            "path": plan["paths"]["eval"][str(seed)]["receipt"],
            "sha256": sha256_file(plan["paths"]["eval"][str(seed)]["receipt"]),
        })
        result = Path(plan["paths"]["eval"][str(seed)]["out_dir"]) / "results.json"
        blob, protocol = _validate_seed_result(plan, seed=seed, result_path=result)
        blobs.append(blob)
        protocols.append(protocol)
        try:
            rows = common._validate_exact_eval_blob(
                blob, seed=seed, label=f"operator-repair seed {seed}",
                identity_profile="current_candidate",
            )
        except common.ChainError as exc:
            raise OperatorRepairError(str(exc)) from exc
        if candidate_rows.keys() & rows.keys():
            raise OperatorRepairError("singleton seed evaluations overlap")
        candidate_rows.update(rows)
    reference = common._protocol_without_seeds(protocols[0])
    if any(common._protocol_without_seeds(value) != reference for value in protocols[1:]):
        raise OperatorRepairError("singleton seed protocols differ")
    target_protocol = protocols[0].replace(seeds=SEEDS)
    records = [
        EpisodeResult.from_dict(row)
        for blob in blobs for row in blob.get("episodes", [])
    ]
    keys = [record.key() for record in records]
    expected = {item.key() for item in iter_work(target_protocol)}
    if len(keys) != len(set(keys)) or set(keys) != expected or len(keys) != 1_200:
        raise OperatorRepairError("seed results do not form the exact disjoint 1,200 union")
    common_ckpt = blobs[0]["meta"]["ckpt"]
    common_identity = blobs[0]["meta"]["eval_identity"]
    for blob in blobs[1:]:
        if blob["meta"]["ckpt"] != common_ckpt or blob["meta"]["eval_identity"] != common_identity:
            raise OperatorRepairError("seed checkpoint/evaluation identities differ")
    summary = aggregate(records, target_protocol)
    if not (
        summary.get("complete") is True
        and summary.get("n_episodes") == 1_200
        and summary.get("n_expected") == 1_200
        and summary.get("n_errors") == 0
    ):
        raise OperatorRepairError("merged exact-1,200 closure failed")
    endpoint, endpoint_sha = _read_fixed_endpoint(plan)
    comparison = _descriptive_comparison(plan, candidate_rows)
    return {
        "version": blobs[0].get("version", 1),
        "bench": target_protocol.bench,
        "protocol": target_protocol.to_dict(),
        "meta": {
            "ckpt": common_ckpt,
            "eval_identity": common_identity,
            "policy": blobs[0]["meta"]["policy"],
            "source_singleton_seed_receipts": receipts,
            "merge_provenance": {
                "kind": "r0_e2e_operator_repair_stable_merge_v1",
                "plan_sha256": _plan_sha(),
                "source_closure_sha256": plan["source_closure"]["sha256"],
                "config_resolved_hash": plan["config"]["resolved_config_hash"],
                "fixed_endpoint_sha256": endpoint_sha,
            },
            "checkpoint_selection": "predeclared_fixed_step_32000",
            "metric_or_evaluation_selection": False,
            "observational_training_metrics": endpoint["metrics"],
        },
        "summary": summary,
        "descriptive_baseline_comparison": comparison,
        "episodes": [record.to_dict() for record in sorted(records, key=lambda row: row.key())],
    }


def _merged_markdown(merged: Mapping[str, Any]) -> str:
    comparison = merged["descriptive_baseline_comparison"]
    rows = [
        "# R0 operator-repair fixed-step evaluation",
        "",
        "Endpoint: exact predeclared step 32,000. No metric or evaluation selected this checkpoint.",
        "The historical baseline delta and paired confidence interval, when available, are descriptive only.",
        "",
    ]
    rows.extend((
        "| suite | candidate | baseline | delta (pp) | episodes |",
        "|---|---:|---:|---:|---:|",
    ))
    for suite, value in merged["summary"]["per_suite"].items():
        paired = comparison["per_suite"][suite]
        rows.append(
            f"| {suite} | {float(value['success_rate']):.2f}% | "
            f"{float(paired['baseline_success_rate_percent']):.2f}% | "
            f"{float(paired['delta_percentage_points']):+.2f} | "
            f"{int(value['n_episodes'])} |"
        )
    overall = comparison["overall"]
    rows.extend((
        f"| **average** | **{float(merged['summary']['avg']):.2f}%** | "
        f"**{float(overall['baseline_success_rate_percent']):.2f}%** | "
        f"**{float(overall['delta_percentage_points']):+.2f}** | **1200** |",
        "",
        "Paired task-bootstrap 95% CI: "
        f"[{float(comparison['paired_task_bootstrap']['ci_low_percentage_points']):+.3f}, "
        f"{float(comparison['paired_task_bootstrap']['ci_high_percentage_points']):+.3f}] pp.",
        "No threshold, pass/fail, checkpoint-selection, or promotion rule is applied.",
    ))
    return "\n".join(rows) + "\n"


def _merged_receipt_payload(
    plan: Mapping[str, Any], merged: Mapping[str, Any], *,
    result: Path, table: Path,
) -> dict[str, Any]:
    comparison = merged["descriptive_baseline_comparison"]
    return {
        "format_version": FORMAT_VERSION,
        "kind": "r0_e2e_operator_repair_merged_eval_receipt",
        "plan_sha256": _plan_sha(),
        "fixed_endpoint": FIXED_STEP,
        "checkpoint_selection": "predeclared_no_metric_or_eval_selection",
        "result": str(result), "result_sha256": sha256_file(result),
        "table": str(table), "table_sha256": sha256_file(table),
        "episodes": 1_200, "errors": 0,
        "avg": merged["summary"]["avg"], "complete": True,
        "descriptive_comparison_sha256": hashlib.sha256(
            _canonical_json(comparison).encode()
        ).hexdigest(),
        "descriptive_comparison_available": True,
        "baseline_delta_pp": comparison["overall"]["delta_percentage_points"],
        "paired_ci_low_pp": comparison["paired_task_bootstrap"]["ci_low_percentage_points"],
        "paired_ci_high_pp": comparison["paired_task_bootstrap"]["ci_high_percentage_points"],
        "per_suite": comparison["per_suite"],
        "per_seed_successes": comparison["per_seed_candidate_successes"],
        "outcome_never_blocks_publication": True,
        "promotion_authority": False,
    }


def _validate_merged_receipt(plan: Mapping[str, Any]) -> dict[str, Any]:
    receipt = _read_receipt(
        plan["paths"]["merged_receipt"],
        kind="r0_e2e_operator_repair_merged_eval_receipt",
    )
    result = Path(plan["paths"]["merged_results"])
    table = Path(plan["paths"]["merged_table"])
    merged = merge_seed_results(plan)
    if (
        not result.is_file() or result.read_text() != _pretty_json(merged)
        or not table.is_file() or table.read_text() != _merged_markdown(merged)
    ):
        raise OperatorRepairError("merged artifacts differ from exact recomputation")
    expected = _merged_receipt_payload(plan, merged, result=result, table=table)
    if receipt != expected:
        raise OperatorRepairError("merged receipt differs from exact recomputation")
    return receipt


def _summary_fields(merged: Mapping[str, Any]) -> dict[str, Any]:
    comparison = merged["descriptive_baseline_comparison"]
    fields = {
        "fixed_endpoint": FIXED_STEP,
        "descriptive_comparison_available": True,
    }
    for suite, row in merged["summary"]["per_suite"].items():
        fields[f"candidate_success_rate/{suite}"] = row["success_rate"]
    fields.update({
        "baseline_delta_pp": comparison["overall"]["delta_percentage_points"],
        "paired_ci_low_pp": comparison["paired_task_bootstrap"]["ci_low_percentage_points"],
        "paired_ci_high_pp": comparison["paired_task_bootstrap"]["ci_high_percentage_points"],
        "seed0_successes": comparison["per_seed_candidate_successes"]["0"],
    })
    for suite, row in comparison["per_suite"].items():
        fields[f"baseline_delta_pp/{suite}"] = row["delta_percentage_points"]
    return fields


def _stage_merge(plan: Mapping[str, Any]) -> int:
    _reject_training_asset_failure(plan)
    _reject_durable_execution_failure(plan)
    result = Path(plan["paths"]["merged_results"])
    table = Path(plan["paths"]["merged_table"])
    receipt_path = Path(plan["paths"]["merged_receipt"])
    if receipt_path.exists():
        receipt = _validate_merged_receipt(plan)
        merged = json.loads(result.read_text())
        comparison = merged["descriptive_baseline_comparison"]
        _wandb_publish(
            plan, stage="eval-summary", path=result,
            artifact_type="operator-repair-evaluation-results",
            summary={
                "episodes": 1_200, "success_rate": receipt["avg"], "n_errors": 0,
                **_summary_fields(merged),
            },
        )
        return 0
    merged = merge_seed_results(plan)
    expected_result = _pretty_json(merged)
    expected_table = _merged_markdown(merged)
    if result.exists():
        if not result.is_file() or result.read_text() != expected_result:
            raise OperatorRepairError("partial merged result differs from recomputation")
    else:
        common.exclusive_json_write(result, merged)
    if table.exists():
        if not table.is_file() or table.read_text() != expected_table:
            raise OperatorRepairError("partial merged table differs from recomputation")
    else:
        common._exclusive_text_write(table, expected_table)
    receipt = _merged_receipt_payload(plan, merged, result=result, table=table)
    common.exclusive_json_write(receipt_path, receipt)
    _validate_merged_receipt(plan)
    comparison = merged["descriptive_baseline_comparison"]
    _wandb_publish(
        plan, stage="eval-summary", path=result,
        artifact_type="operator-repair-evaluation-results",
        summary={
            "episodes": 1_200, "success_rate": merged["summary"]["avg"], "n_errors": 0,
            **_summary_fields(merged),
        },
    )
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    return 0


def run_environment_stage() -> int:
    plan, stage, plan_path = _required_plan()
    if stage.startswith("train_"):
        return _stage_train(plan, stage, plan_path)
    # Downstream jobs do not reread 76 GB of cache bytes. The afterok edge is
    # closed by the final train link's immutable post-verification receipt,
    # whose full semantic raw/cache aggregates equal the plan-owned constants.
    _reject_training_asset_failure(plan)
    _reject_durable_execution_failure(plan)
    _read_final_training_asset_verification(plan)
    if stage == "consolidate":
        return _stage_consolidate(plan)
    if stage.startswith("eval_seed"):
        return _stage_eval(plan, stage)
    if stage == "merge":
        return _stage_merge(plan)
    raise OperatorRepairError(f"unknown operator-repair stage {stage!r}")


def _dry_run_payload(plan: Mapping[str, Any]) -> dict[str, Any]:
    plan_path = Path(plan["lineage"]["control_dir"]) / "plan.json"
    placeholder_ids: dict[str, str] = {}
    commands: dict[str, list[str]] = {}
    for index, spec in enumerate(_stage_specs(), 1):
        placeholder_ids[str(spec["name"])] = str(900_000 + index)
        commands[str(spec["name"])] = _sbatch_command(
            spec=spec, plan_path=plan_path, plan_sha="<plan-sha256>",
            dependencies=[placeholder_ids[name] for name in spec["depends_on"]],
            group=plan["wandb"]["group"],
        )
    return {
        "plan": plan,
        "dag": [
            {"name": spec["name"], "depends_on": spec["depends_on"], "sbatch": spec["sbatch"]}
            for spec in _stage_specs()
        ],
        "commands": commands,
        "job_count": len(_stage_specs()),
        "submission_performed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    submit = subparsers.add_parser("submit")
    submit.add_argument("--run-dir", required=True)
    submit.add_argument("--control-dir", required=True)
    submit.add_argument("--artifact-root", required=True)
    submit.add_argument("--group", required=True)
    submit.add_argument("--project", default=PROJECT)
    submit.add_argument("--dry-run", action="store_true")
    subparsers.add_parser("run-stage")
    args = parser.parse_args(argv)
    try:
        if args.command == "run-stage":
            return run_environment_stage()
        plan = build_plan(
            run_dir=Path(args.run_dir), control_dir=Path(args.control_dir),
            artifact_root=Path(args.artifact_root), group=args.group,
            project=args.project,
        )
        if args.dry_run:
            print(json.dumps(_dry_run_payload(plan), indent=2, sort_keys=True))
            return 0
        receipt = submit_plan(plan)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    except (OperatorRepairError, common.ChainError, subprocess.CalledProcessError) as exc:
        print(f"OPERATOR_REPAIR_INVALID: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
