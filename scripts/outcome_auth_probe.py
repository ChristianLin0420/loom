#!/usr/bin/env python3
"""Exact, read-only replay probe for outcome-recovery train0/group0.

This probe is intentionally narrower than the trainer authenticator.  It opens
the pinned step-49,666 parent and the immutable first train0 sidecar, replays
the parent proposal under a small precision/batching matrix, and writes one
exclusive JSON report.  It never changes either input.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch import Tensor  # noqa: E402

from contracts import M, TOPK  # noqa: E402
from loom.eval import outcome_recovery as recovery  # noqa: E402
from loom.heads.proposal import pl_log_prob, weights_from_logits  # noqa: E402
from loom.train import outcome_grpo as grpo  # noqa: E402


CHECKPOINT_REL = "runs/r0a_deploy_s1_eval/ckpt_000049666.pt"
CHECKPOINT_SHA256 = "15f286c268caa5327d5aa3abf1f67ebd0555c426a509fef22cb7f537bf6ab4e1"
SIDECAR_REL = (
    "runs/outcome_recovery_s49666_train0/groups/"
    "libero_spatial__task00__trial10__seed0.pt"
)
SIDECAR_SHA256 = "441004267ecca795e3a0e1ecdd7ec4efaa4a1503ee339e260694b046a956d961"
GROUP_ID = "libero_spatial/task=00/trial=10/seed=0"
EXPECTED_REPLANS = (97, 30, 16, 97, 97, 18, 17, 97)


class ProbeError(RuntimeError):
    """The fixed input, replay, or output contract was violated."""


def _require(condition: Any, message: str) -> None:
    if not bool(condition):
        raise ProbeError(message)


def sha256_file(path: str | os.PathLike[str], chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ReplayMode:
    name: str
    batch_size: int
    lang_layout: str
    deterministic: bool
    allow_tf32: bool
    autocast_dtype: str | None


def replay_modes() -> tuple[ReplayMode, ...]:
    """Small factorial isolating collector shape, trainer shape, and precision."""
    return (
        ReplayMode("collector_b1_fp32", 1, "single", False, False, None),
        ReplayMode("b1_expanded_fp32", 1, "expanded", False, False, None),
        ReplayMode("trainer_b1_fp32", 1, "expanded", True, False, None),
        ReplayMode("trainer_chunk32_fp32", 32, "expanded", True, False, None),
        ReplayMode("trainer_b1_tf32", 1, "expanded", True, True, None),
        ReplayMode("trainer_chunk32_tf32", 32, "expanded", True, True, None),
        ReplayMode("trainer_b1_bf16", 1, "expanded", True, False, "bfloat16"),
        ReplayMode("trainer_chunk32_bf16", 32, "expanded", True, False, "bfloat16"),
    )


def _linear_p95(values: Tensor) -> float:
    flat = values.detach().to(dtype=torch.float64, device="cpu").reshape(-1)
    _require(flat.numel() > 0, "cannot summarize an empty error vector")
    return float(torch.quantile(flat, 0.95, interpolation="linear"))


def _error_summary(values: Sequence[float]) -> dict[str, Any]:
    tensor = torch.tensor(list(values), dtype=torch.float64)
    _require(tensor.numel() > 0, "cannot summarize zero errors")
    _require(bool(torch.isfinite(tensor).all()), "error vector contains nan/inf")
    return {
        "n": int(tensor.numel()),
        "max": float(tensor.max()),
        "p95_linear": _linear_p95(tensor),
        "mean": float(tensor.mean()),
    }


def summarize_replay(
    rows: Sequence[Mapping[str, Any]],
    *,
    reference_logits: Sequence[Tensor] | None,
) -> dict[str, Any]:
    """Summarize per-replan errors; old-logprob excludes diagnostic arm zero."""
    _require(bool(rows), "replay produced no rows")
    if reference_logits is not None:
        _require(len(reference_logits) == len(rows), "reference row count differs")

    coeff_errors: list[float] = []
    sampled_lp_errors: list[float] = []
    direct_order = direct_support = 0
    replay_support = 0
    reference_order = reference_support = 0
    worst_coeff: dict[str, Any] | None = None
    worst_lp: dict[str, Any] | None = None

    for index, row in enumerate(rows):
        logits = torch.as_tensor(row["logits"]).float().cpu()
        coeff = torch.as_tensor(row["coeff"]).float().cpu()
        stored_coeff = torch.as_tensor(row["stored_coeff"]).float().cpu()
        order = torch.as_tensor(row["order"]).to(torch.int64).cpu()
        arm = int(row["arm"])
        replan = int(row["replan"])

        delta = (coeff - stored_coeff).abs()
        atom_error = float(delta.max())
        coeff_errors.append(atom_error)
        if worst_coeff is None or atom_error > float(worst_coeff["abs_error"]):
            operator = int(delta.argmax())
            worst_coeff = {
                "arm": arm, "replan": replan, "operator": operator,
                "abs_error": atom_error,
                "current": float(coeff[operator]),
                "stored": float(stored_coeff[operator]),
            }

        support = set(torch.nonzero(coeff != 0, as_tuple=False).flatten().tolist())
        if support == set(order.tolist()):
            replay_support += 1
        top = logits.topk(TOPK, dim=-1).indices
        if arm == 0:
            direct_order += int(torch.equal(top, order))
            direct_support += int(set(top.tolist()) == set(order.tolist()))

        if arm > 0:
            lp_error = abs(float(row["old_logprob"]) - float(row["stored_old_logprob"]))
            sampled_lp_errors.append(lp_error)
            if worst_lp is None or lp_error > float(worst_lp["abs_error"]):
                worst_lp = {
                    "arm": arm, "replan": replan, "abs_error": lp_error,
                    "current": float(row["old_logprob"]),
                    "stored": float(row["stored_old_logprob"]),
                }

        if reference_logits is not None:
            reference = torch.as_tensor(reference_logits[index]).float().cpu()
            ref_top = reference.topk(TOPK, dim=-1).indices
            reference_order += int(torch.equal(top, ref_top))
            reference_support += int(set(top.tolist()) == set(ref_top.tolist()))

    n_direct = sum(int(row["arm"]) == 0 for row in rows)
    result = {
        "coeff_abs_error_per_replan_linf": _error_summary(coeff_errors),
        "sampled_old_logprob_abs_error": _error_summary(sampled_lp_errors),
        "worst_coeff": worst_coeff,
        "worst_sampled_old_logprob": worst_lp,
        "support_identity": {
            "n_replans": len(rows),
            "recomputed_coeff_support_equals_stored_order": replay_support,
            "arm0_replans": n_direct,
            "arm0_argmax_order_equals_stored": direct_order,
            "arm0_argmax_support_equals_stored": direct_support,
            "vs_collector_b1_argmax_order": (
                reference_order if reference_logits is not None else len(rows)
            ),
            "vs_collector_b1_argmax_support": (
                reference_support if reference_logits is not None else len(rows)
            ),
        },
    }
    return result


@contextlib.contextmanager
def _precision(mode: ReplayMode, device: torch.device):
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    warn_fn = getattr(torch, "is_deterministic_algorithms_warn_only_enabled", None)
    old_warn_only = bool(warn_fn()) if callable(warn_fn) else False
    old_matmul_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    old_cudnn_tf32 = bool(torch.backends.cudnn.allow_tf32)
    old_precision = torch.get_float32_matmul_precision()
    try:
        torch.use_deterministic_algorithms(mode.deterministic, warn_only=True)
        torch.backends.cuda.matmul.allow_tf32 = mode.allow_tf32
        torch.backends.cudnn.allow_tf32 = mode.allow_tf32
        torch.set_float32_matmul_precision("high" if mode.allow_tf32 else "highest")
        dtype = torch.bfloat16 if mode.autocast_dtype == "bfloat16" else None
        with torch.autocast(
            device_type=device.type,
            dtype=dtype,
            enabled=dtype is not None,
        ):
            yield
    finally:
        torch.use_deterministic_algorithms(old_deterministic, warn_only=old_warn_only)
        torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
        torch.set_float32_matmul_precision(old_precision)


@torch.no_grad()
def replay_mode(
    proposal: torch.nn.Module,
    payload: Mapping[str, Any],
    *,
    mode: ReplayMode,
    device: torch.device,
) -> list[dict[str, Any]]:
    proposal.eval()
    dtype = next(proposal.parameters()).dtype
    rows: list[dict[str, Any]] = []
    with _precision(mode, device):
        for arm_index, arm in enumerate(payload["arms"]):
            z = arm["z"].to(device=device, dtype=dtype)
            lang_single = arm["lang"].to(device=device, dtype=dtype)
            if lang_single.ndim == 2:
                lang_single = lang_single.unsqueeze(0)
            _require(lang_single.ndim == 3 and lang_single.shape[0] == 1,
                     "stored language is not one collector batch")
            n = int(z.shape[0])
            lang_expanded = lang_single.expand(n, -1, -1)
            for lo in range(0, n, mode.batch_size):
                hi = min(n, lo + mode.batch_size)
                if mode.lang_layout == "single":
                    _require(mode.batch_size == 1,
                             "single collector language layout requires batch one")
                    lang = lang_single
                else:
                    lang = lang_expanded[lo:hi]
                logits_batch = proposal.logits(z[lo:hi], lang)
                _require(tuple(logits_batch.shape) == (hi - lo, M),
                         f"proposal output shape drifted: {tuple(logits_batch.shape)}")
                order_batch = arm["ordered_support"][lo:hi].to(device=device)
                coeff_batch = weights_from_logits(
                    logits_batch.float(), order_batch.to(torch.int64), M,
                ).float()
                lp_batch = pl_log_prob(
                    logits_batch.float(), order_batch.to(torch.int64),
                ).float()
                for offset in range(hi - lo):
                    position = lo + offset
                    rows.append({
                        "arm": arm_index,
                        "replan": position,
                        "logits": logits_batch[offset].detach().float().cpu(),
                        "order": arm["ordered_support"][position].detach().cpu(),
                        "coeff": coeff_batch[offset].detach().cpu(),
                        "stored_coeff": arm["coeff"][position].detach().float().cpu(),
                        "old_logprob": float(lp_batch[offset]),
                        "stored_old_logprob": float(arm["old_logprob"][position]),
                    })
            del z, lang_single, lang_expanded
    torch.cuda.synchronize(device)
    return rows


def _load_fixed_inputs(device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = ROOT / CHECKPOINT_REL
    sidecar = ROOT / SIDECAR_REL
    _require(checkpoint.is_file(), f"missing fixed checkpoint: {checkpoint}")
    _require(sidecar.is_file(), f"missing fixed sidecar: {sidecar}")
    _require(sha256_file(checkpoint) == CHECKPOINT_SHA256,
             "fixed parent checkpoint SHA-256 differs")
    _require(sha256_file(sidecar) == SIDECAR_SHA256,
             "fixed train0/group0 sidecar SHA-256 differs")

    try:
        parent = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        parent = torch.load(checkpoint, map_location="cpu", weights_only=True)
    _require(isinstance(parent, dict), "parent checkpoint is not a mapping")
    _require(int(parent.get("global_step", -1)) == recovery.SEED_GLOBAL_STEP,
             "parent checkpoint step differs")
    _require(parent.get("config_hash") == recovery.SEED_CONFIG_HASH,
             "parent checkpoint config hash differs")
    proposal = grpo._load_proposal(parent, device=device)
    del parent

    try:
        payload = torch.load(sidecar, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        payload = torch.load(sidecar, map_location="cpu", weights_only=False)
    _require(isinstance(payload, dict), "sidecar is not a mapping")
    _require(payload.get("group_id") == GROUP_ID, "sidecar group identity differs")
    arms = payload.get("arms")
    _require(isinstance(arms, list) and len(arms) == recovery.GROUP_SIZE,
             "sidecar does not have eight arms")
    replans = tuple(int(arm["z"].shape[0]) for arm in arms)
    _require(replans == EXPECTED_REPLANS, f"sidecar replan vector differs: {replans}")
    return proposal, payload


def _runtime() -> dict[str, Any]:
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": properties.name,
        "gpu_capability": list(torch.cuda.get_device_capability(device)),
        "startup": {
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "flash_sdp": bool(torch.backends.cuda.flash_sdp_enabled()),
            "mem_efficient_sdp": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
            "math_sdp": bool(torch.backends.cuda.math_sdp_enabled()),
        },
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


def run_probe() -> dict[str, Any]:
    _require(torch.cuda.is_available(), "exact replay probe requires one CUDA GPU")
    _require(torch.cuda.device_count() == 1,
             f"probe requires exactly one visible GPU, saw {torch.cuda.device_count()}")
    device = torch.device("cuda:0")
    runtime = _runtime()
    proposal, payload = _load_fixed_inputs(device)
    results: dict[str, Any] = {}
    reference_logits: list[Tensor] | None = None
    started = time.monotonic()
    for mode in replay_modes():
        before = time.monotonic()
        rows = replay_mode(proposal, payload, mode=mode, device=device)
        summary = summarize_replay(rows, reference_logits=reference_logits)
        summary["mode"] = asdict(mode)
        summary["wall_seconds"] = float(time.monotonic() - before)
        results[mode.name] = summary
        if reference_logits is None:
            reference_logits = [row["logits"].clone() for row in rows]
        del rows

    return {
        "format_version": 1,
        "kind": "loom_outcome_behaviour_auth_probe",
        "status": "COMPLETE",
        "inputs": {
            "checkpoint": {
                "path": str((ROOT / CHECKPOINT_REL).resolve()),
                "sha256": CHECKPOINT_SHA256,
                "global_step": recovery.SEED_GLOBAL_STEP,
                "config_hash": recovery.SEED_CONFIG_HASH,
            },
            "sidecar": {
                "path": str((ROOT / SIDECAR_REL).resolve()),
                "sha256": SIDECAR_SHA256,
                "group_id": GROUP_ID,
                "replans_by_arm": list(EXPECTED_REPLANS),
            },
        },
        "definitions": {
            "coeff_error": "per-replan L_inf across M, then linear p95/max",
            "old_logprob_error": "absolute scalar error over sampled arms 1..7 only",
            "support_identity": "raw-logit top4 order/set; sampled PL order is replayed exactly",
            "configured_tolerances": {
                "coeff_atol": grpo.BEHAVIOUR_COEFF_ATOL,
                "coeff_rtol": grpo.BEHAVIOUR_COEFF_RTOL,
                "old_logprob_atol": grpo.BEHAVIOUR_LOGPROB_ATOL,
                "old_logprob_rtol": grpo.BEHAVIOUR_LOGPROB_RTOL,
            },
        },
        "runtime": runtime,
        "modes": results,
        "wall_seconds": float(time.monotonic() - started),
        "source": {
            "probe_sha256": sha256_file(__file__),
            "proposal_sha256": sha256_file(ROOT / "loom/heads/proposal.py"),
            "trainer_sha256": sha256_file(ROOT / "loom/train/outcome_grpo.py"),
            "collector_sha256": sha256_file(ROOT / "loom/eval/outcome_recovery.py"),
        },
    }


def atomic_publish(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> str:
    target = Path(path).expanduser().resolve()
    _require(not target.exists(), f"refusing to overwrite probe result: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    fd, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(raw).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="new exclusive JSON report path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_probe()
        digest = atomic_publish(args.out, report)
    except Exception as exc:  # noqa: BLE001
        print(f"OUTCOME_AUTH_PROBE_FAILED: {type(exc).__name__}: {exc}", flush=True)
        return 2
    compact = {
        name: {
            "coeff": row["coeff_abs_error_per_replan_linf"],
            "sampled_old_logprob": row["sampled_old_logprob_abs_error"],
            "support": row["support_identity"],
        }
        for name, row in report["modes"].items()
    }
    print(json.dumps({
        "status": "COMPLETE", "out": str(Path(args.out).resolve()),
        "sha256": digest, "modes": compact,
    }, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
