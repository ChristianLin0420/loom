#!/usr/bin/env python3
"""Exact no-update gate for prospective positive-advantage imitation.

This isolated diagnostic is reachable only after authenticating the immutable
ABORT report from the direct outcome-GRPO audit.  At the unchanged step-49,666
seed it measures three exact round-robin TRAIN offsets (24 rank-local groups),
twice differentiates direct positive-advantage (PA) and direct
PA+unit-reference objectives, and projects production-clipped SGD/reset-AdamW
clone directions onto the unchanged outcome-blind 48-group development panel.

The TRAIN objective contains no GRPO, Switch balance, or sparse CE term.  The
heldout derivative remains the signed GRPO negative-clipped-surrogate loss, so
negative projected loss deltas retain the previous gate's meaning.  No live
optimizer step, proposal perturbation, checkpoint, candidate, formal
evaluation, or promotion artifact is possible.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import sys
import tempfile
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch import Tensor, nn  # noqa: E402

from loom.eval import outcome_recovery as recovery  # noqa: E402
from loom.train import outcome_grpo as v1  # noqa: E402
from loom.train import outcome_grpo_v2 as v2  # noqa: E402
from loom.train import outcome_positive_advantage as pa  # noqa: E402
from loom.train.determinism import set_global_seed, set_step_seed  # noqa: E402
from loom.train.loop import read_config  # noqa: E402
from scripts import outcome_round_robin_direction_audit as direct_v2  # noqa: E402


FORMAT_VERSION = 2
KIND = "loom_outcome_positive_advantage_round_robin_direction_audit"
EXPECTED_WORLD_SIZE = direct_v2.EXPECTED_WORLD_SIZE
AUDIT_OFFSETS = direct_v2.AUDIT_OFFSETS
AUDIT_STEPS = direct_v2.AUDIT_STEPS
EXPECTED_TRAIN_DRAWS = direct_v2.EXPECTED_TRAIN_DRAWS
EXPECTED_FOLD_DRAWS = direct_v2.EXPECTED_FOLD_DRAWS
PANEL_GROUPS = direct_v2.PANEL_GROUPS
PANEL_TASKS = direct_v2.PANEL_TASKS
PANEL_TASKS_PER_RANK = direct_v2.PANEL_TASKS_PER_RANK
EXPECTED_SEED_CHECKPOINT = direct_v2.EXPECTED_SEED_CHECKPOINT
EXPECTED_VALIDATION_IDENTITY_DIGEST = direct_v2.EXPECTED_VALIDATION_IDENTITY_DIGEST
EXPECTED_PANEL_GROUP_RECEIPT_SHA256 = direct_v2.EXPECTED_PANEL_GROUP_RECEIPT_SHA256
EXPECTED_BOOTSTRAP_MATRIX_SHA256 = direct_v2.EXPECTED_BOOTSTRAP_MATRIX_SHA256

PA_WEIGHT = 1.0
RECOVERY_REFERENCE_WEIGHT = 1.0
DEMO_REFERENCE_WEIGHT = 1.0
GRPO_TRAIN_WEIGHT = 0.0
BALANCE_WEIGHT = 0.0
SPARSE_CE_WEIGHT = 0.0

EXPECTED_PA_CORE_FILE_SHA256 = (
    "c70dfa6239ff8ef1eda9cf16b167c7bf5d3b7de00c918b8fa3c0b69faa69a358"
)
EXPECTED_PA_CORE_SOURCE_SHA256 = (
    "caf8d616e93d21522c6ebbaf448b86775ff4f76c83e9a421190f64c9b8121366"
)
EXPECTED_DIRECT_V2_FILE_SHA256 = (
    "0e9a163f288ec57183493294e829a2d02960fa3af0db0d77aed629e93bc19977"
)
EXPECTED_DIRECT_V2_DIAGNOSTIC_SHA256 = (
    "5fc13990ff04793d4eccfeeca01daf0a72bae60b98bb4e34a5f5a718950df9b7"
)
EXPECTED_V2_SOURCE_SHA256 = direct_v2.EXPECTED_V2_SOURCE_SHA256

EXPECTED_CONFIG_REL = "configs/r0a_outcome_positive_advantage_audit.yaml"
EXPECTED_CONFIG_FILE_SHA256 = (
    "2e375a3db095006d8b4dbe972f9b938da534bbc8b1bd7b7c1dfb4e186d2e9dd4"
)
EXPECTED_RESOLVED_CONFIG_HASH = "caaec7d8ecae82ec"
INHERITED_CONFIG_REL = direct_v2.EXPECTED_CONFIG_REL
INHERITED_CONFIG_FILE_SHA256 = direct_v2.EXPECTED_CONFIG_FILE_SHA256
INHERITED_RESOLVED_CONFIG_HASH = direct_v2.EXPECTED_RESOLVED_CONFIG_HASH

TRIGGER_REPORT_REL = (
    "runs/diagnostics/outcome_round_robin_direction_audit/"
    "outcome_round_robin_direction_audit_v2_s49666_32577492.json"
)
TRIGGER_REPORT_SHA256 = (
    "1f3c4af95ba97a5976e2cdacee8fc50fda6ff88f725886fca25b9304e4ef9e1f"
)
TRIGGER_STATUS = "ABORT_OUTCOME_OBJECTIVE"

OUTPUT_DIR_REL = "runs/diagnostics/outcome_positive_advantage_direction_audit"
OUTPUT_NAME_PREFIX = "outcome_positive_advantage_direction_audit_v2_s49666_"
_AUDIT_SOURCE_FILES = (
    EXPECTED_CONFIG_REL,
    "scripts/outcome_positive_advantage_direction_audit.py",
    "scripts/outcome_positive_advantage_direction_audit.sbatch",
)

ELIGIBILITY = {
    "diagnostic_only": True,
    "exposed_development_only": True,
    "full_training_eligible": False,
    "candidate_eligible": False,
    "official_evaluation_eligible": False,
    "promotion_eligible": False,
    "maximum_authority_on_pass": (
        "separate_64_update_ineligible_pa_pilot_freeze_only"
    ),
    "live_optimizer_steps": 0,
    "live_parameter_perturbations": 0,
    "checkpoint_emitted": False,
    "candidate_emitted": False,
}

INVALID_INSTRUMENTATION_HISTORY = ({
    "job_id": 32580600,
    "format_version": 1,
    "terminal_state": "FAILED",
    "exit_code": "2:0",
    "classification": "INVALID_NO_SCIENTIFIC_EVIDENCE",
    "cause": (
        "canonical_PA_disabled_sparse_target_was_passed_to_demo_anchor_constructor"
    ),
    "report_emitted": False,
    "panel_statistics_reached": False,
},)

DEMO_ANCHOR_CONSTRUCTION_RECEIPT = {
    "config_source": "authenticated_inherited_v2_resolved_config",
    "source_path": INHERITED_CONFIG_REL,
    "source_raw_sha256": INHERITED_CONFIG_FILE_SHA256,
    "source_resolved_hash": INHERITED_RESOLVED_CONFIG_HASH,
    "canonical_pa_proposal_target": {"enabled": False, "weight": 0.0},
    "inherited_construction_target": {
        "enabled": True,
        "weight": None,
        "mode": "sparse_ce",
        "temperature": 1.0,
        "detach_belief": True,
    },
    "v2_core_internal_target_producer_weight": 1.0,
    "scope": "construction_only_state_and_target_producer",
    "canonical_pa_config_mutated": False,
    "sparse_ce_scalar_computed": False,
    "sparse_ce_graph_constructed": False,
}


class PositiveAdvantageDirectionAuditError(RuntimeError):
    """Authentication/numerical/mutation failure; publish no report."""


def _require(condition: Any, message: str) -> None:
    if not bool(condition):
        raise PositiveAdvantageDirectionAuditError(message)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    before = resolved.stat()
    digest = recovery.sha256_file(resolved)
    after = resolved.stat()
    _require(
        (int(before.st_dev), int(before.st_ino), int(before.st_size),
         int(before.st_mtime_ns))
        == (int(after.st_dev), int(after.st_ino), int(after.st_size),
            int(after.st_mtime_ns)),
        f"file changed while hashing: {resolved}",
    )
    return {
        "path": str(resolved),
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "size": int(after.st_size),
        "mtime_ns": int(after.st_mtime_ns),
        "sha256": digest,
    }


def _read_json_with_identity(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Parse exactly the bytes whose SHA/stat identity is returned."""
    resolved = path.expanduser().resolve()
    before = resolved.stat()
    encoded = resolved.read_bytes()
    after = resolved.stat()
    _require(
        (int(before.st_dev), int(before.st_ino), int(before.st_size),
         int(before.st_mtime_ns))
        == (int(after.st_dev), int(after.st_ino), int(after.st_size),
            int(after.st_mtime_ns)),
        f"JSON file changed while reading: {resolved}",
    )
    _require(len(encoded) == int(after.st_size),
             f"JSON byte count differs from stat size: {resolved}")
    digest = hashlib.sha256(encoded).hexdigest()
    payload = json.loads(encoded.decode("utf-8"))
    _require(isinstance(payload, dict), f"JSON root is not an object: {resolved}")
    return dict(payload), {
        "path": str(resolved),
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "size": int(after.st_size),
        "mtime_ns": int(after.st_mtime_ns),
        "sha256": digest,
    }


def _validate_output_path(path: str | os.PathLike[str]) -> Path:
    output = Path(path).expanduser().resolve()
    directory = (ROOT / OUTPUT_DIR_REL).resolve()
    _require(output.parent == directory,
             f"diagnostic output must be directly inside {directory}: {output}")
    _require(output.suffix == ".json", "PA direction output must be JSON")
    _require(output.name.startswith(OUTPUT_NAME_PREFIX),
             f"PA direction output must start with {OUTPUT_NAME_PREFIX!r}")
    _require(not output.exists(), f"refusing existing diagnostic output: {output}")
    return output


def exclusive_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish with an exclusive hard link; never replace an existing path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False,
    ) + "\n").encode("utf-8")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise PositiveAdvantageDirectionAuditError(
                f"refusing existing diagnostic output: {path}"
            ) from exc
        from loom.train.atomic import fsync_dir
        fsync_dir(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _config_file_identity(path: Path) -> dict[str, Any]:
    expected = (ROOT / EXPECTED_CONFIG_REL).resolve()
    resolved = path.expanduser().resolve()
    _require(resolved == expected,
             f"PA audit config must be canonical file {expected}")
    identity = _file_identity(resolved)
    _require(identity["sha256"] == EXPECTED_CONFIG_FILE_SHA256,
             f"canonical PA config bytes drifted: {identity['sha256']}")
    return identity


def _validate_config(path: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    config_identity = _config_file_identity(path)
    cfg = read_config(path)
    resolved_hash = v1._config_hash(cfg)
    _require(resolved_hash == EXPECTED_RESOLVED_CONFIG_HASH,
             f"resolved PA config hash drifted: {resolved_hash}")

    inherited_path = (ROOT / INHERITED_CONFIG_REL).resolve()
    inherited_identity = _file_identity(inherited_path)
    _require(inherited_identity["sha256"] == INHERITED_CONFIG_FILE_SHA256,
             "inherited authenticated v2 config bytes drifted")
    inherited = read_config(inherited_path)
    _require(v1._config_hash(inherited) == INHERITED_RESOLVED_CONFIG_HASH,
             "inherited authenticated v2 resolved config drifted")
    inherited_validation = v2.validate_scaffold_config(inherited)
    for section in ("data", "optim", "fsdp", "slurm"):
        _require(cfg[section] == inherited[section],
                 f"PA receipt changed inherited {section} geometry")
    expected_inherited_outcome = copy.deepcopy(inherited["outcome_grpo_v2"])
    expected_inherited_outcome.update({
        "method_status": "RETIRED_TRAIN_RECIPE_AUTHENTICATED_INPUTS_ONLY",
        "stop_step": None,
        "snapshot_steps": [],
    })
    expected_inherited_outcome["sampler"]["total_updates"] = None
    expected_inherited_outcome["train_trust_panel"]["every"] = None
    expected_inherited_outcome["validation_lineage"][
        "current_development_collection"
    ]["exposures"] = [
        "v1_terminal_selection",
        "early_curve_diagnostic",
        "component_gradient_projection",
        "round_robin_direction_audit",
        "positive_advantage_direction_audit",
    ]
    expected_inherited_outcome["artifact_policy"] = {
        "role": "inherited_authenticated_inputs_only",
        "candidate_emission": "forbidden",
        "promotion": "forbidden",
        "official_evaluation": "forbidden",
        "pilot_checkpoint_only": False,
    }
    _require(cfg["outcome_grpo_v2"] == expected_inherited_outcome,
             "PA receipt changed or failed to revoke inherited v2 permissions")

    _require(cfg["run"] == {
        "name": "r0a_outcome_positive_advantage_audit_NONLAUNCHABLE",
        "project": "loom", "seed": v1.TRAIN_SEED, "steps": None,
        "log_every": None, "ckpt_every": None, "keep_last": 0,
        "deterministic": True, "wandb_mode": "disabled",
    }, "PA audit run receipt is not exactly nonlaunchable")
    _require(cfg["train_modules"] == [],
             "PA audit receipt must expose no train modules")
    losses = cfg["losses"]
    for name in ("dyn", "act", "potential"):
        _require(not bool(losses[name]["enabled"])
                 and float(losses[name]["weight"]) == 0.0,
                 f"inherited loss {name} must remain disabled")
    for name in ("grpo", "proposal", "balance"):
        _require(not bool(losses[name]["enabled"])
                 and float(losses[name]["weight"]) == 0.0,
                 f"forbidden PA TRAIN term {name} is enabled")
    positive = dict(losses["positive_advantage"])
    _require(positive == {
        "enabled": True,
        "weight": PA_WEIGHT,
        "kind": "clipped_positive_advantage_conditional_imitation",
        "rewards": "all_8_terminal_rewards",
        "standardisation": "subtract_mean_divide_population_rms",
        "positive_weight": "max_standardised_advantage_0",
        "scored_arms": list(range(1, recovery.GROUP_SIZE)),
        "arm0": "baseline_and_scale_only_never_scored_or_weighted",
        "aggregation": "equal_context_within_arm_then_equal_7_arms",
        "zero_weight_arms": "retained_as_explicit_terms",
        "ratio_clip": [pa.RATIO_CLIP_LOW, pa.RATIO_CLIP_HIGH],
        "minimized_atom": "-min(rho*w,clamp(rho,0.8,1.2)*w)",
    }, "positive-advantage objective receipt differs")
    recovery_ref = dict(losses["recovery_reference"])
    _require(
        bool(recovery_ref["enabled"])
        and float(recovery_ref["weight"]) == RECOVERY_REFERENCE_WEIGHT
        and recovery_ref["coefficient_status"] == "FIXED_FOR_DIRECTION_AUDIT"
        and recovery_ref["kind"] == "sampled_pl_forward_kl_control_variate"
        and recovery_ref["arms"] == list(range(1, recovery.GROUP_SIZE))
        and recovery_ref["arm0"] == "forbidden"
        and recovery_ref["controller_weight"] is None,
        "unit recovery-reference receipt differs",
    )
    demo_ref = dict(losses["demo_reference"])
    _require(
        bool(demo_ref["enabled"])
        and float(demo_ref["weight"]) == DEMO_REFERENCE_WEIGHT
        and demo_ref["coefficient_status"] == "FIXED_FOR_DIRECTION_AUDIT"
        and demo_ref["kind"]
            == "exact_analytic_vjp_dense_categorical_forward_kl"
        and demo_ref["current_vjp"] == "p_current_minus_p_seed"
        and demo_ref["horizon_aggregation"] == "equal_mean"
        and not bool(demo_ref["seed_requires_grad"])
        and not bool(demo_ref["seed_in_optimizer"])
        and not bool(demo_ref["seed_in_checkpoint"]),
        "unit exact-VJP demo-reference receipt differs",
    )

    receipt = dict(cfg["outcome_positive_advantage_audit"])
    _require(int(receipt["format_version"]) == FORMAT_VERSION
             and receipt["audit_kind"] == KIND
             and receipt["method_status"] == "DIAGNOSTIC_ONLY_NONLAUNCHABLE"
             and receipt["objective_name"]
                == "positive_advantage_conditional_imitation"
             and not bool(receipt["training_loop_present"]),
             "PA audit namespace identity differs")
    _require(receipt["seed_checkpoint"] == EXPECTED_SEED_CHECKPOINT
             and int(receipt["seed_global_step"]) == v2.START_STEP
             and int(receipt["world_size"]) == EXPECTED_WORLD_SIZE
             and tuple(receipt["round_robin_offsets"]) == AUDIT_OFFSETS
             and int(receipt["train_draws"]) == EXPECTED_TRAIN_DRAWS
             and int(receipt["contexts_per_arm"])
                == v2.EXPECTED_CONTEXTS_PER_ARM,
             "PA audit seed/geometry receipt differs")
    _require(
        tuple(receipt["instrumentation_history"])
            == INVALID_INSTRUMENTATION_HISTORY,
        "PA invalid-instrumentation history differs",
    )
    trigger = dict(receipt["trigger"])
    _require(trigger == {
        "required": True,
        "report_path": TRIGGER_REPORT_REL,
        "report_sha256": TRIGGER_REPORT_SHA256,
        "required_kind": direct_v2.KIND,
        "required_format_version": direct_v2.FORMAT_VERSION,
        "required_status": TRIGGER_STATUS,
        "required_execution_validated": True,
        "required_decision_passed": False,
    }, "PA trigger receipt differs")
    _require(receipt["source_provenance"] == {
        "positive_advantage_core_file_sha256": EXPECTED_PA_CORE_FILE_SHA256,
        "positive_advantage_core_closure_sha256": EXPECTED_PA_CORE_SOURCE_SHA256,
        "immutable_direct_v2_helper_file_sha256": EXPECTED_DIRECT_V2_FILE_SHA256,
        "immutable_direct_v2_diagnostic_closure_sha256": (
            EXPECTED_DIRECT_V2_DIAGNOSTIC_SHA256
        ),
        "own_diagnostic_closure": {
            "scheme": "sha256(path-nul-sha256-nul)-v1",
            "files": list(_AUDIT_SOURCE_FILES),
            "binding": (
                "computed_at_authenticated_start_rechecked_after_use_and_"
                "frozen_before_submission"
            ),
        },
    }, "PA source-provenance receipt differs")
    recipe = dict(receipt["frozen_direction_recipe"])
    _require(
        float(recipe["positive_advantage_weight"]) == PA_WEIGHT
        and float(recipe["recovery_reference_weight"])
            == RECOVERY_REFERENCE_WEIGHT
        and float(recipe["demo_reference_weight"]) == DEMO_REFERENCE_WEIGHT
        and float(recipe["grpo_train_weight"]) == GRPO_TRAIN_WEIGHT
        and float(recipe["switch_balance_weight"]) == BALANCE_WEIGHT
        and float(recipe["sparse_ce_weight"]) == SPARSE_CE_WEIGHT
        and recipe["tuning_or_sweep"] == "forbidden"
        and recipe["coefficient_switching"] == "forbidden"
        and recipe["train_scoring_geometry"]
            == "read_only_authentication_replay_plus_two_objective_graph_exact_B1_passes_per_train_point"
        and recipe["authentication_replay"]
            == "exact_selected_context_identity_no_objective_graph"
        and recipe["first_objective_graph_scorer_pass"]
            == "authoritative_positive_advantage_same_selected_rows_order_old"
        and recipe["second_objective_graph_scorer_pass"]
            == "recovery_k3_only_same_selected_rows_order_old"
        and recipe["second_pass_scorer_authentication"]
            == "exact_current_float_equals_stored_old_per_selected_row_or_INVALID"
        and recipe["second_pass_identity_evidence"]
            == "exact_zero_recovery_value_and_bitwise_zero_gradient"
        and recipe["demo_logit_authentication"]
            == "exact_live_seed_logits_per_horizon_or_INVALID"
        and recipe["demo_anchor_construction"]
            == DEMO_ANCHOR_CONSTRUCTION_RECEIPT
        and recipe["reference_identity_classification"] == {
            "missing_disconnected_nonfinite_auth_or_ratio_failure": (
                "INVALID_NO_REPORT"
            ),
            "complete_finite_nonzero_recovery_or_demo_vjp": (
                "SCIENTIFIC_ABORT"
            ),
            "pass_requires_local_and_synchronised_bitwise_zero": True,
        }
        and recipe["heldout_gradient"]
            == "signed_grpo_negative_clipped_surrogate_loss",
        "PA direction recipe differs",
    )
    _require(receipt["eligibility"] == ELIGIBILITY,
             "PA eligibility receipt differs")
    _require(all(value is None for value in receipt["unresolved_pilot"].values())
             and all(value is None for value in receipt["unresolved_formal"].values()),
             "PA pilot/formal fields must remain unresolved")
    _require(cfg["artifact_policy"] == {
        "role": "exposed_development_diagnostic_only",
        "checkpoint_emission": "forbidden",
        "candidate_emission": "forbidden",
        "promotion": "forbidden",
        "official_evaluation": "forbidden",
    }, "PA artifact policy differs")
    return cfg, resolved_hash, {
        "passed": True,
        "config_file": config_identity,
        "inherited_config_file": inherited_identity,
        "inherited_v2_scaffold": inherited_validation,
        "training_loop_present": False,
        "train_modules": [],
    }


def _authenticated_demo_anchor_construction_config(
    pa_cfg: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the authenticated v2 target-producer config without PA mutation."""
    canonical_hash_before = v1._config_hash(pa_cfg)
    canonical_target = dict(pa_cfg["losses"]["proposal"])
    _require(
        canonical_target["enabled"] is False
        and float(canonical_target["weight"]) == 0.0
        and canonical_target["mode"] == "sparse_ce"
        and float(canonical_target["temperature"]) == 1.0
        and canonical_target["detach_belief"] is True,
        "canonical PA sparse target is not disabled false/0 with retained recipe",
    )
    source_path = (ROOT / INHERITED_CONFIG_REL).resolve()
    source_identity = _file_identity(source_path)
    _require(
        source_identity["sha256"] == INHERITED_CONFIG_FILE_SHA256,
        "demo-anchor construction source bytes differ from authenticated v2 config",
    )
    construction_cfg = copy.deepcopy(read_config(source_path))
    construction_hash = v1._config_hash(construction_cfg)
    _require(
        construction_hash == INHERITED_RESOLVED_CONFIG_HASH,
        "demo-anchor construction resolved v2 config differs",
    )
    construction_target = dict(construction_cfg["losses"]["proposal"])
    expected_target = DEMO_ANCHOR_CONSTRUCTION_RECEIPT[
        "inherited_construction_target"
    ]
    _require(
        all(construction_target.get(key) == value
            for key, value in expected_target.items()),
        "authenticated v2 sparse target-producer fields differ",
    )
    _require(
        construction_cfg["data"] == pa_cfg["data"],
        "PA and authenticated v2 demo-anchor data geometry differ",
    )
    _require(
        construction_cfg["outcome_grpo_v2"]["anchor_manifest"]
            == pa_cfg["outcome_grpo_v2"]["anchor_manifest"],
        "PA and authenticated v2 demo-anchor manifest pins differ",
    )
    _require(
        int(construction_cfg["run"]["seed"]) == int(pa_cfg["run"]["seed"]),
        "PA and authenticated v2 demo-anchor sampler seeds differ",
    )
    _require(
        v1._config_hash(pa_cfg) == canonical_hash_before,
        "canonical PA config mutated while deriving demo-anchor construction",
    )
    return construction_cfg, {
        **copy.deepcopy(DEMO_ANCHOR_CONSTRUCTION_RECEIPT),
        "source_file_identity": source_identity,
        "canonical_pa_resolved_hash": canonical_hash_before,
        "construction_resolved_hash": construction_hash,
        "authenticated_geometry_equal": {
            "data": True,
            "anchor_manifest": True,
            "sampler_seed": True,
        },
    }


def _construct_demo_reference_anchor(
    parent: Mapping[str, Any],
    live_proposal: nn.Module,
    *,
    pa_cfg: Mapping[str, Any],
    device: torch.device,
    rank: int,
    world: int,
) -> tuple[v2.DemoReferenceAnchorV2, dict[str, Any]]:
    """Construct only the demo state/target producer from authenticated v2."""
    canonical_hash_before = v1._config_hash(pa_cfg)
    construction_cfg, receipt = _authenticated_demo_anchor_construction_config(
        pa_cfg,
    )
    construction_hash_before = v1._config_hash(construction_cfg)
    anchor = v2.DemoReferenceAnchorV2.from_parent(
        parent, live_proposal, trainer_cfg=construction_cfg, device=device,
        rank=rank, world_size=world,
    )
    _require(
        v1._config_hash(pa_cfg) == canonical_hash_before,
        "canonical PA config mutated during demo-anchor construction",
    )
    _require(
        v1._config_hash(construction_cfg) == construction_hash_before,
        "authenticated v2 construction config escaped core-local adaptation",
    )
    return anchor, {
        **receipt,
        "constructor_completed": True,
        "canonical_pa_config_unchanged_after_constructor": True,
        "construction_config_unchanged_after_constructor": True,
        "sparse_ce_scalar_computed": False,
        "sparse_ce_graph_constructed": False,
    }


def _source_identity() -> dict[str, Any]:
    direct_file = _file_identity(
        ROOT / "scripts/outcome_round_robin_direction_audit.py"
    )
    _require(direct_file["sha256"] == EXPECTED_DIRECT_V2_FILE_SHA256,
             "immutable direct-v2 helper module bytes drifted")
    direct_identity = direct_v2._source_identity()
    _require(direct_identity["diagnostic"]["sha256"]
             == EXPECTED_DIRECT_V2_DIAGNOSTIC_SHA256,
             "immutable direct-v2 diagnostic closure drifted")
    _require(direct_identity["v2_trainer"]["sha256"]
             == EXPECTED_V2_SOURCE_SHA256,
             "inherited v2 executable closure drifted")
    pa_file = _file_identity(ROOT / "loom/train/outcome_positive_advantage.py")
    _require(pa_file["sha256"] == EXPECTED_PA_CORE_FILE_SHA256,
             "positive-advantage core bytes drifted")
    pa_identity = pa.core_source_identity()
    _require(pa_identity["sha256"] == EXPECTED_PA_CORE_SOURCE_SHA256,
             "positive-advantage core closure drifted")
    pa.assert_core_source_identity(pa_identity)
    diagnostic = v1._trainer_source_identity(
        root=ROOT, files=_AUDIT_SOURCE_FILES,
    )
    _require(diagnostic["scheme"] == "sha256(path-nul-sha256-nul)-v1",
             "PA diagnostic source-closure scheme differs from receipt")
    return {
        "positive_advantage_core": pa.core_provenance(pa_identity),
        "immutable_direct_v2_helpers": direct_identity,
        "direct_v2_helper_file": direct_file,
        "positive_advantage_core_file": pa_file,
        "diagnostic": diagnostic,
    }


def _validate_trigger_payload(report: Mapping[str, Any], path: Path) -> dict[str, Any]:
    _require(int(report.get("format_version", -1)) == direct_v2.FORMAT_VERSION,
             "trigger report format differs")
    _require(report.get("kind") == direct_v2.KIND,
             "trigger report kind differs")
    _require(report.get("status") == TRIGGER_STATUS,
             "PA audit requires the exact GRPO ABORT status")
    _require(report.get("execution_validated") is True,
             "trigger GRPO audit was not execution-valid")
    decision = dict(report.get("decision") or {})
    _require(decision.get("passed") is False
             and decision.get("status") == TRIGGER_STATUS,
             "trigger GRPO decision is not the locked ABORT")
    _require(Path(str(report.get("output"))).resolve() == path,
             "trigger report self-path differs")
    source = dict(report.get("source_identity") or {})
    _require(source["diagnostic"]["sha256"]
             == EXPECTED_DIRECT_V2_DIAGNOSTIC_SHA256
             and source["v2_trainer"]["sha256"] == EXPECTED_V2_SOURCE_SHA256,
             "trigger source closure differs")
    config = dict(report.get("config") or {})
    _require(config["resolved_hash"] == INHERITED_RESOLVED_CONFIG_HASH
             and config["raw_file_identity"]["sha256"]
                == INHERITED_CONFIG_FILE_SHA256,
             "trigger config identity differs")
    panel = dict(report.get("outcome_blind_panel") or {})
    _require(panel["group_receipt"]["sha256"]
             == EXPECTED_PANEL_GROUP_RECEIPT_SHA256,
             "trigger panel receipt differs")
    _require(report["bootstrap_resample_matrix"]["sha256"]
             == EXPECTED_BOOTSTRAP_MATRIX_SHA256,
             "trigger bootstrap receipt differs")
    no_mutation = dict(report.get("no_mutation") or {})
    _require(no_mutation.get("passed") is True
             and int(no_mutation.get("live_optimizer_steps", -1)) == 0
             and no_mutation.get("proposal_digest_before")
                == no_mutation.get("proposal_digest_after"),
             "trigger no-mutation closure differs")
    sidecars = dict(report.get("selected_sidecar_post_use_closure") or {})
    _require(sidecars.get("passed") is True
             and int(sidecars.get("sidecars", -1)) == 72,
             "trigger 72-sidecar closure differs")
    _require(report.get("eligibility", {}).get("full_training_eligible") is False
             and report.get("eligibility", {}).get("official_evaluation_eligible")
                is False,
             "trigger report claims forbidden eligibility")
    return {
        "kind": report["kind"],
        "format_version": int(report["format_version"]),
        "status": report["status"],
        "execution_validated": True,
        "decision_passed": False,
        "scientific_meaning": (
            "authenticated outcome-GRPO objective aborted; PA is a new "
            "prospective method, not a threshold relaxation"
        ),
    }


def authenticate_trigger_once(*, rank: int, world: int) -> dict[str, Any]:
    expected = (ROOT / TRIGGER_REPORT_REL).resolve()
    box: list[Any] = [None]
    if rank == 0:
        try:
            report, identity = _read_json_with_identity(expected)
            _require(identity["sha256"] == TRIGGER_REPORT_SHA256,
                     f"trigger report SHA differs: {identity['sha256']}")
            receipt = _validate_trigger_payload(report, expected)
            box[0] = {"ok": True, "identity": identity, "receipt": receipt}
        except Exception as exc:  # noqa: BLE001
            box[0] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if world > 1:
        torch.distributed.broadcast_object_list(box, src=0)
    _require(bool(box[0].get("ok")),
             f"PA trigger authentication failed: {box[0].get('error')}")
    return {
        "file": dict(box[0]["identity"]),
        "receipt": dict(box[0]["receipt"]),
    }


def assert_trigger_unchanged(trigger: Mapping[str, Any], *, rank: int, world: int) -> None:
    local_error = ""
    if rank == 0:
        try:
            report, current = _read_json_with_identity(
                Path(trigger["file"]["path"])
            )
            _require(current == trigger["file"],
                     "trigger report stat/SHA changed during PA audit")
            _require(_validate_trigger_payload(report, Path(current["path"]))
                     == trigger["receipt"],
                     "trigger report semantic receipt changed")
        except Exception as exc:  # noqa: BLE001
            local_error = f"{type(exc).__name__}: {exc}"
    v1._raise_if_any_rank_failed(local_error, world, "final PA trigger closure")


def _recovery_reference_only(
    proposal: nn.Module,
    payload: Mapping[str, Any],
    replan_indices: Mapping[int, Sequence[int]],
    *,
    device: torch.device,
) -> tuple[Tensor, dict[str, Any]]:
    """Score only the sampled PL forward-KL control variate, no GRPO/balance."""
    expected = set(range(1, recovery.GROUP_SIZE))
    _require(set(replan_indices) == expected,
             "recovery reference requires exactly arms 1..7")
    arms = list(payload.get("arms") or ())
    _require(len(arms) == recovery.GROUP_SIZE,
             "recovery reference group does not contain eight arms")
    dtype = next(proposal.parameters()).dtype
    current_logprobs: list[Tensor] = []
    old_logprobs: list[Tensor] = []
    atoms = 0
    max_abs_current_old_error = 0.0
    for arm_index in range(1, recovery.GROUP_SIZE):
        arm = arms[arm_index]
        index = pa._selected_indices(
            replan_indices, arm_index, int(arm["z"].shape[0]),
        )
        z = arm["z"].index_select(0, index).to(
            device=device, dtype=dtype, non_blocking=True,
        )
        order = arm["ordered_support"].index_select(0, index).to(device=device)
        old = arm["old_logprob"].detach().index_select(0, index).to(
            device=device, dtype=torch.float32,
        )
        lang = v1._batched_lang(arm["lang"], int(index.numel()), device, dtype)
        current, _logits = v1.stored_order_logprob(proposal, z, lang, order)
        current32 = current.float()
        max_abs_current_old_error = max(
            max_abs_current_old_error,
            float((current32.detach() - old).abs().max()),
        )
        _require(
            torch.equal(current32.detach(), old),
            f"recovery second-pass current/old identity differs in arm {arm_index}",
        )
        current_logprobs.append(current)
        old_logprobs.append(old)
        atoms += int(current.numel())
    reference = v2.recovery_pl_forward_kl(current_logprobs, old_logprobs)
    return reference, {
        "selected_atoms": atoms,
        "max_abs_current_old_logprob_error": max_abs_current_old_error,
        "current_old_logprobs_bitwise_identical": True,
        "all_seed_ratios_exactly_one": True,
        "scorer_authentication_failure_classification": "INVALID_NO_REPORT",
    }


def _analytic_demo_reference_only(
    anchor: v2.DemoReferenceAnchorV2,
    global_step: int,
    *,
    cache_prepared_for_reuse: bool = False,
) -> tuple[Tensor, dict[str, Any]]:
    """Exact-value demo KL with PA core's exact analytic current-logit VJP."""
    anchor.assert_seed_unchanged()
    prepared = anchor.anchor._prepare(global_step)
    beliefs, lang, targets, _embodiment = prepared
    horizons = len(targets)
    _require(horizons > 0 and len(beliefs) >= horizons,
             "analytic demo anchor has no complete horizons")
    terms: list[Tensor] = []
    forward_autocast = anchor.anchor.device.type == "cuda"
    logits_bitwise_identical = True
    for horizon in range(horizons):
        belief = beliefs[horizon].detach()
        batched_lang = v1._batched_lang(
            lang, int(belief.shape[0]), belief.device, belief.dtype,
        )
        with torch.autocast(
            device_type=belief.device.type,
            dtype=torch.bfloat16,
            enabled=forward_autocast,
        ):
            current_logits = anchor.anchor.proposal.logits(belief, batched_lang)
            with torch.no_grad():
                seed_logits = anchor.seed_proposal.logits(belief, batched_lang)
        _require(current_logits.shape == seed_logits.shape,
                 f"analytic demo h{horizon + 1} logits shape differs")
        logits_bitwise_identical = (
            logits_bitwise_identical and torch.equal(current_logits, seed_logits)
        )
        terms.append(pa.analytic_categorical_forward_kl(
            current_logits, seed_logits, reduction="mean",
        ))
    reference = torch.stack(terms).mean()
    _require(bool(torch.isfinite(reference)),
             f"nonfinite analytic demo reference at step {global_step}")
    anchor.assert_seed_unchanged()
    if cache_prepared_for_reuse:
        _require(int(global_step) not in anchor.anchor._cache,
                 "analytic demo prepared-step cache unexpectedly occupied")
        anchor.anchor._cache[int(global_step)] = prepared
    return reference, {
        "demo_exact_analytic_vjp_forward_kl": float(reference.detach()),
        "demo_reference_horizons": horizons,
        "live_seed_logits_bitwise_identical": logits_bitwise_identical,
        "demo_reference_seed_trainable": False,
        "demo_reference_forward_bf16_autocast": forward_autocast,
        "demo_reference_probability_math_fp32": True,
        "current_logit_vjp": "p_current_minus_p_seed",
        "equal_horizon_aggregation": True,
        "sparse_ce_computed": False,
        "sparse_ce_graph_constructed": False,
    }


def _complete_reference_gradient_evidence(
    loss: Tensor,
    proposal: nn.Module,
    *,
    objective_norm: float,
    world: int,
    retain_graph: bool,
    label: str,
) -> dict[str, Any]:
    """Differentiate a complete reference graph and preserve local-zero proof."""
    local_error = ""
    try:
        named = direct_v2._named_live_parameters(proposal)
        local, missing = direct_v2.component_audit._local_gradient_vector(
            loss, named, retain_graph=retain_graph,
        )
        _require(not missing,
                 f"{label} reference gradient missing parameters: {missing[:8]}")
        reference_value = float(loss.detach())
        _require(math.isfinite(reference_value),
                 f"{label} reference value is nonfinite")
        _require(reference_value == 0.0,
                 f"{label} reference value is not exactly zero")
        local_bitwise_zero = int(torch.count_nonzero(local)) == 0
        local_norm = direct_v2.component_audit._vector_norm(local)
    except Exception as exc:  # noqa: BLE001
        local_error = f"{type(exc).__name__}: {exc}"
    v1._raise_if_any_rank_failed(local_error, world, f"{label} local reference graph")
    post_sync_error = ""
    try:
        synchronised = direct_v2.component_audit._synchronise_gradient(
            local, world=world,
        )
        norm = direct_v2.component_audit._vector_norm(synchronised)
        synchronised_bitwise_zero = int(torch.count_nonzero(synchronised)) == 0
        bound = direct_v2.REFERENCE_GRADIENT_RELATIVE_BOUND * max(
            float(objective_norm), 1.0,
        )
        bound_passed = norm <= bound
        value = reference_value
    except Exception as exc:  # noqa: BLE001
        post_sync_error = f"{type(exc).__name__}: {exc}"
    v1._raise_if_any_rank_failed(
        post_sync_error, world, f"{label} synchronised reference evidence",
    )
    return {
        "value": value,
        "value_exactly_zero": True,
        "local_gradient_norm": local_norm,
        "local_gradient_bitwise_all_zero": local_bitwise_zero,
        "synchronised_gradient_norm": norm,
        "synchronised_gradient_bitwise_all_zero": synchronised_bitwise_zero,
        "relative_bound": direct_v2.REFERENCE_GRADIENT_RELATIVE_BOUND,
        "absolute_bound": bound,
        "bound_passed": bound_passed,
        "missing_parameter_gradients": [],
        "complete_reference_graph_required": True,
    }


def _coordinated_synchronised_loss_gradient(
    loss: Tensor,
    named: Sequence[tuple[str, nn.Parameter]],
    *,
    world: int,
    retain_graph: bool,
    label: str,
) -> tuple[Tensor, list[str]]:
    """Coordinate both pre-reduce and post-reduce gradient failures."""
    local_error = ""
    try:
        vector, missing = direct_v2._synchronised_loss_gradient(
            loss, named, world=world, retain_graph=retain_graph,
            label=label, require_complete=True,
        )
    except Exception as exc:  # noqa: BLE001
        local_error = f"{type(exc).__name__}: {exc}"
    v1._raise_if_any_rank_failed(
        local_error, world, f"{label} synchronised gradient evidence",
    )
    return vector, missing


def benefit_cosine_evidence(
    heldout_gradient: Tensor,
    update_delta: Tensor,
) -> dict[str, Any]:
    """Return a gate-safe benefit cosine; zero directions scientifically fail."""
    heldout_norm = direct_v2.component_audit._vector_norm(heldout_gradient)
    update_norm = direct_v2.component_audit._vector_norm(update_delta)
    if heldout_norm == 0.0 or update_norm == 0.0:
        return {
            "defined": False,
            "value": None,
            "gate_value": 0.0,
            "heldout_gradient_norm": heldout_norm,
            "direction_norm": update_norm,
            "undefined_reason": (
                "zero_heldout_gradient" if heldout_norm == 0.0
                else "zero_update_direction"
            ),
        }
    value = direct_v2.aggregate_benefit_cosine(heldout_gradient, update_delta)
    return {
        "defined": True,
        "value": value,
        "gate_value": value,
        "heldout_gradient_norm": heldout_norm,
        "direction_norm": update_norm,
        "undefined_reason": None,
    }


def exact_float32_vector_sha256(vector: Tensor) -> str:
    """Hash every canonical fp32 vector byte with bounded host staging."""
    flat = vector.detach().to(dtype=torch.float32).contiguous().reshape(-1)
    digest = hashlib.sha256()
    digest.update(json.dumps({
        "dtype": "torch.float32",
        "numel": int(flat.numel()),
        "shape": list(vector.shape),
    }, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\0")
    chunk_elements = 4 * 1024 * 1024
    for start in range(0, int(flat.numel()), chunk_elements):
        chunk = flat[start:start + chunk_elements].cpu().numpy()
        digest.update(chunk.tobytes(order="C"))
    return digest.hexdigest()


def decide_pa_direction_gate(
    *,
    primary_names: Sequence[str],
    endpoint_bounds: Mapping[str, Any],
    endpoint_cosines: Sequence[Mapping[str, Any]],
    increment_names: Sequence[str],
    increment_bounds: Mapping[str, Any],
    increment_cosines: Sequence[Mapping[str, Any]],
    all_reference_relative_bounds_passed: bool,
    all_recovery_second_pass_gradients_bitwise_zero: bool,
    all_demo_analytic_vjp_gradients_bitwise_zero: bool,
) -> dict[str, Any]:
    """Apply the unchanged direct-v2 thresholds with PA-specific semantics."""
    _require(len(primary_names) == len(endpoint_cosines) == 3,
             "PA gate requires exactly three primary endpoints")
    _require(len(increment_names) == len(increment_cosines) == len(AUDIT_STEPS),
             "PA gate requires exactly three AdamW increments")
    endpoint_rows: list[dict[str, Any]] = []
    for index, name in enumerate(primary_names):
        point = float(endpoint_bounds["point_means"][index])
        upper = float(endpoint_bounds["upper_confidence_bounds"][index])
        cosine = dict(endpoint_cosines[index])
        row = {
            "name": str(name),
            "point_mean_loss_delta": point,
            "bonferroni_one_sided_95_ucb": upper,
            "aggregate_benefit_cosine": cosine,
            "point_mean_strictly_beneficial": point < 0.0,
            "ucb_strictly_beneficial": upper < 0.0,
            "benefit_cosine_defined": bool(cosine["defined"]),
            "benefit_cosine_at_least_0p01": (
                bool(cosine["defined"])
                and float(cosine["gate_value"])
                    >= direct_v2.MIN_ENDPOINT_BENEFIT_COSINE
            ),
        }
        row["passed"] = all((
            row["point_mean_strictly_beneficial"],
            row["ucb_strictly_beneficial"],
            row["benefit_cosine_at_least_0p01"],
        ))
        endpoint_rows.append(row)
    increment_rows: list[dict[str, Any]] = []
    for offset, (name, cosine_raw) in enumerate(zip(
        increment_names, increment_cosines, strict=True,
    )):
        lower = float(increment_bounds["lower_confidence_bounds"][offset])
        cosine = dict(cosine_raw)
        row = {
            "name": str(name),
            "offset": offset,
            "global_step": int(AUDIT_STEPS[offset]),
            "point_mean_loss_delta": float(
                increment_bounds["point_means"][offset]
            ),
            "bonferroni_one_sided_95_lcb": lower,
            "aggregate_benefit_cosine": cosine,
            "benefit_cosine_below_minus_0p01": (
                bool(cosine["defined"])
                and float(cosine["gate_value"])
                    < direct_v2.MAX_CATASTROPHIC_WRONG_WAY_BENEFIT_COSINE
            ),
            "lcb_strictly_harmful": lower > 0.0,
        }
        row["catastrophic"] = (
            row["benefit_cosine_below_minus_0p01"]
            or row["lcb_strictly_harmful"]
        )
        increment_rows.append(row)
    reference_gate = (
        bool(all_reference_relative_bounds_passed)
        and bool(all_recovery_second_pass_gradients_bitwise_zero)
        and bool(all_demo_analytic_vjp_gradients_bitwise_zero)
    )
    passed = (
        all(row["passed"] for row in endpoint_rows)
        and not any(row["catastrophic"] for row in increment_rows)
        and reference_gate
    )
    return {
        "status": (
            "PASS_TO_SEPARATE_64_UPDATE_PA_INELIGIBLE_PILOT_FREEZE"
            if passed else "ABORT_POSITIVE_ADVANTAGE_OBJECTIVE"
        ),
        "passed": passed,
        "primary_endpoints": endpoint_rows,
        "production_adamw_increment_catastrophes": increment_rows,
        "all_six_reference_relative_bounds_passed": bool(
            all_reference_relative_bounds_passed
        ),
        "all_three_recovery_second_pass_gradients_bitwise_zero": bool(
            all_recovery_second_pass_gradients_bitwise_zero
        ),
        "all_three_demo_analytic_vjp_gradients_bitwise_zero": bool(
            all_demo_analytic_vjp_gradients_bitwise_zero
        ),
        "reference_gate_passed": reference_gate,
        "selection_rule": (
            "all three predeclared PA endpoints require point<0, Bonferroni "
            "one-sided 95% UCB<0, and defined benefit cosine>=0.01; no "
            "production-AdamW increment may have defined benefit cosine<-0.01 "
            "or Bonferroni LCB>0; all six unit-reference relative bounds and "
            "all three recovery-second-pass plus all three exact analytic-demo-"
            "VJP bitwise-zero gradients must pass"
        ),
        "threshold_inheritance": {
            "source": "immutable_direct_v2_direction_audit",
            "minimum_endpoint_benefit_cosine": (
                direct_v2.MIN_ENDPOINT_BENEFIT_COSINE
            ),
            "maximum_catastrophic_wrong_way_benefit_cosine": (
                direct_v2.MAX_CATASTROPHIC_WRONG_WAY_BENEFIT_COSINE
            ),
            "reference_gradient_relative_bound": (
                direct_v2.REFERENCE_GRADIENT_RELATIVE_BOUND
            ),
            "confidence": direct_v2.CONFIDENCE,
            "bootstrap_samples": direct_v2.BOOTSTRAP_SAMPLES,
            "bootstrap_seed": direct_v2.BOOTSTRAP_SEED,
        },
        "pass_authority": (
            "separate_64_update_ineligible_pa_pilot_freeze_only; this report "
            "is not a launchable recipe"
        ),
        "full_training_authorized": False,
    }


def analyse_panel_directions(
    *,
    panel_task_rows: Sequence[Mapping[str, Any]],
    gathered_panel_selections: Sequence[Sequence[Mapping[str, Any]]],
    primary_vectors: Mapping[str, Tensor],
    increment_vectors: Mapping[str, Tensor],
    heldout_gradient: Tensor,
    all_reference_relative_bounds_passed: bool,
    all_recovery_second_pass_gradients_bitwise_zero: bool,
    all_demo_analytic_vjp_gradients_bitwise_zero: bool,
) -> dict[str, Any]:
    """Complete all post-panel local analysis before another collective."""
    panel_selection_rows = [
        dict(row) for rank_rows in gathered_panel_selections for row in rank_rows
    ]
    _require(len(panel_selection_rows) == PANEL_GROUPS,
             "PA panel did not authenticate exactly 48 sidecars")
    resample_matrix, bootstrap_receipt = (
        direct_v2.make_suite_stratified_resample_matrix(
            [str(row["task_key"]) for row in panel_task_rows]
        )
    )
    _require(bootstrap_receipt["sha256"] == EXPECTED_BOOTSTRAP_MATRIX_SHA256,
             "PA fixed bootstrap matrix drifted")
    primary_names = tuple(primary_vectors)
    endpoint_values = torch.tensor([
        [float(row["projections"][name]) for row in panel_task_rows]
        for name in primary_names
    ], dtype=torch.float64)
    endpoint_bounds = direct_v2.bonferroni_task_bounds(
        endpoint_values, resample_matrix,
    )
    increment_names = tuple(increment_vectors)
    increment_values = torch.tensor([
        [float(row["projections"][name]) for row in panel_task_rows]
        for name in increment_names
    ], dtype=torch.float64)
    increment_bounds = direct_v2.bonferroni_task_bounds(
        increment_values, resample_matrix,
    )
    endpoint_cosines = [
        benefit_cosine_evidence(heldout_gradient, primary_vectors[name])
        for name in primary_names
    ]
    increment_cosines = [
        benefit_cosine_evidence(heldout_gradient, increment_vectors[name])
        for name in increment_names
    ]
    decision = decide_pa_direction_gate(
        primary_names=primary_names,
        endpoint_bounds=endpoint_bounds,
        endpoint_cosines=endpoint_cosines,
        increment_names=increment_names,
        increment_bounds=increment_bounds,
        increment_cosines=increment_cosines,
        all_reference_relative_bounds_passed=all_reference_relative_bounds_passed,
        all_recovery_second_pass_gradients_bitwise_zero=(
            all_recovery_second_pass_gradients_bitwise_zero
        ),
        all_demo_analytic_vjp_gradients_bitwise_zero=(
            all_demo_analytic_vjp_gradients_bitwise_zero
        ),
    )
    all_directions = {**primary_vectors, **increment_vectors}
    projection_closure: dict[str, Any] = {}
    for name, direction in all_directions.items():
        task_mean = sum(float(row["projections"][name])
                        for row in panel_task_rows) / PANEL_TASKS
        aggregate_dot = direct_v2.component_audit._dot(
            heldout_gradient, direction,
        )
        residual = aggregate_dot - task_mean
        mean_abs = sum(abs(float(row["projections"][name]))
                       for row in panel_task_rows) / PANEL_TASKS
        scale = max(abs(aggregate_dot), abs(task_mean), mean_abs, 1e-30)
        relative = abs(residual) / scale
        _require(relative <= 2e-4,
                 f"PA equal-task projection residual too large for {name}: {relative}")
        projection_closure[name] = {
            "task_mean": task_mean,
            "aggregate_gradient_dot": aggregate_dot,
            "absolute_residual": abs(residual),
            "mean_absolute_task_projection": mean_abs,
            "relative_residual": relative,
            "max_relative_residual": 2e-4,
            "passed": True,
        }
    return {
        "panel_selection_rows": panel_selection_rows,
        "bootstrap_receipt": bootstrap_receipt,
        "primary_names": primary_names,
        "endpoint_bounds": endpoint_bounds,
        "increment_names": increment_names,
        "increment_bounds": increment_bounds,
        "endpoint_cosines": endpoint_cosines,
        "increment_cosines": increment_cosines,
        "decision": decision,
        "projection_closure": projection_closure,
    }


def _without_vectors(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "vectors"}


def _runtime_local(
    *, rank: int, world: int, local_rank: int, device: torch.device,
) -> dict[str, Any]:
    base = direct_v2._runtime_local(
        rank=rank, world=world, local_rank=local_rank, device=device,
    )
    return {**base, "audit_kind": KIND}


def run_audit(
    *,
    config_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
) -> dict[str, Any] | None:
    """Execute the fixed eight-rank read-only PA direction diagnostic."""
    rank, world, local_rank, device = v1._dist_info()
    _require(world == EXPECTED_WORLD_SIZE,
             f"PA direction audit requires world=8, got {world}")
    local_error = ""
    try:
        output = _validate_output_path(output_path)
    except Exception as exc:  # noqa: BLE001
        local_error = f"{type(exc).__name__}: {exc}"
    v1._raise_if_any_rank_failed(local_error, world, "exclusive PA diagnostic output")
    started = time.monotonic()

    local_error = ""
    try:
        strict = v1._configure_strict_outcome_determinism()
        scoring_config = v1._configure_exact_proposal_scoring(device)
        runtime_local = _runtime_local(
            rank=rank, world=world, local_rank=local_rank, device=device,
        )
    except Exception as exc:  # noqa: BLE001
        local_error = f"{type(exc).__name__}: {exc}"
    v1._raise_if_any_rank_failed(local_error, world, "A100 deterministic geometry")
    runtimes = direct_v2.component_audit._all_gather_object(runtime_local, world)
    _require(sorted(int(row["rank"]) for row in runtimes) == list(range(world)),
             "distributed runtime rank evidence is incomplete")
    _require(len({(row["hostname"], row["local_rank"]) for row in runtimes}) == world,
             "ranks do not map to eight distinct local A100s")
    _require(len({row["hostname"] for row in runtimes}) == 1,
             "PA audit requires one-node first-production geometry")

    config = Path(config_path).expanduser().resolve()
    local_error = ""
    try:
        cfg, config_hash, scaffold_validation = _validate_config(config)
        config_file_identity = _config_file_identity(config)
        source_identity = _source_identity()
    except Exception as exc:  # noqa: BLE001
        local_error = f"{type(exc).__name__}: {exc}"
    v1._raise_if_any_rank_failed(
        local_error, world,
        "PA source/config",
    )
    # Trigger authentication broadcasts rank-zero bytes. Enter it only after
    # every rank has cleared the preceding local source/config stage.
    trigger_identity = authenticate_trigger_once(rank=rank, world=world)
    # Only the exact authenticated ABORT can unlock any recovery-manifest read.
    local_error = ""
    try:
        group_receipt, validation_root = direct_v2.pre_reward_panel_receipt(cfg)
    except Exception as exc:  # noqa: BLE001
        local_error = f"{type(exc).__name__}: {exc}"
    v1._raise_if_any_rank_failed(
        local_error, world, "PA post-trigger pre-reward panel receipt",
    )
    source_rows = direct_v2.component_audit._all_gather_object(source_identity, world)
    _require(all(row == source_rows[0] for row in source_rows),
             "PA source identity differs across ranks")
    trigger_rows = direct_v2.component_audit._all_gather_object(trigger_identity, world)
    _require(all(row == trigger_rows[0] for row in trigger_rows),
             "PA trigger identity differs across ranks")
    panel_rows = direct_v2.component_audit._all_gather_object(group_receipt, world)
    _require(all(row == panel_rows[0] for row in panel_rows),
             "PA pre-reward panel receipt differs across ranks")

    seed = int(cfg["run"].get("seed", v1.TRAIN_SEED))
    set_global_seed(seed, rank)
    checkpoint = ROOT / EXPECTED_SEED_CHECKPOINT
    parent_identity = direct_v2._authenticate_parent_once(
        checkpoint, rank=rank, world=world,
    )
    local_error = ""
    try:
        v1._assert_seed_stat(parent_identity)
        parent = v1._load_parent_from_identity(parent_identity)
        v1._assert_seed_stat(parent_identity)
        proposal = v1._load_proposal(parent, device=device)
        proposal.eval()
        scoring_evidence = direct_v2._exact_scoring_evidence(proposal, device)
        proposal_digest_before = v1.proposal_module_digest(proposal.state_dict())
        _require(proposal_digest_before == v1.proposal_model_digest(parent["model"]),
                 "runtime proposal differs from authenticated seed proposal")
        demo_anchor, demo_anchor_construction = _construct_demo_reference_anchor(
            parent, proposal, pa_cfg=cfg, device=device,
            rank=rank, world=world,
        )
        with torch.no_grad():
            preflight_reference, preflight_metrics = _analytic_demo_reference_only(
                demo_anchor, v2.START_STEP, cache_prepared_for_reuse=True,
            )
        _require(float(preflight_reference) == 0.0,
                 "analytic demo preflight is not exactly zero")
        _require(bool(preflight_metrics["live_seed_logits_bitwise_identical"]),
                 "analytic demo preflight live/seed logits differ")
        anchor_preflight = {
            "passed": True,
            "objective": "exact_analytic_vjp_dense_categorical_forward_kl_only",
            "construction": demo_anchor_construction,
            "global_step": v2.START_STEP,
            "prepared_batch_cached_for_first_audit_point": True,
            "data": dict(demo_anchor.anchor.data_provenance),
            **preflight_metrics,
        }
        live_wrapper = v1._ProposalOnly(proposal)
        live_optimizer_sentinel = v1.build_optimizer(
            live_wrapper,
            lr=float(cfg["optim"]["lr"]),
            weight_decay=float(cfg["optim"]["weight_decay"]),
            betas=tuple(float(value) for value in cfg["optim"]["betas"]),
            lr_scales={
                "proposal": float(cfg["optim"]["lr_scales"]["proposal"])
            },
            module_names=["proposal"],
        )
        _require(len(live_optimizer_sentinel.state) == 0,
                 "live optimizer sentinel did not begin empty")
    except Exception as exc:  # noqa: BLE001
        local_error = f"{type(exc).__name__}: {exc}"
    v1._raise_if_any_rank_failed(local_error, world, "PA seed/demo-anchor construction")
    anchor_preflight_rows = direct_v2.component_audit._all_gather_object(
        anchor_preflight, world,
    )
    del parent

    train_collections: list[v1.ValidatedRecoveryCollection] = []
    local_error = ""
    try:
        for spec in cfg["outcome_grpo_v2"]["folds"]:
            train_collections.append(v1.ValidatedRecoveryCollection.open(
                ROOT / str(spec["path"]), checkpoint_identity=parent_identity,
                expected_split=str(spec["split"]), deep=False,
                verify_sidecars=False,
            ))
        lineage_rows = cfg["outcome_grpo_v2"]["authenticated_data_lineage"][
            "training"
        ]
        _require(len(lineage_rows) == len(train_collections),
                 "PA TRAIN lineage row count changed")
        for collection, pinned in zip(train_collections, lineage_rows, strict=True):
            _require(
                collection.split == str(pinned["split"])
                and str(collection.root)
                    == str((ROOT / str(pinned["path"])).resolve())
                and collection.manifest_sha256 == str(pinned["manifest_sha256"])
                and collection.identity_digest == str(pinned["identity_digest"]),
                f"opened {collection.split} differs from authenticated lineage",
            )
        validation = v1.ValidatedRecoveryCollection.open(
            validation_root, checkpoint_identity=parent_identity,
            expected_split="validation", deep=False, verify_sidecars=False,
        )
        _require(validation.identity_digest == EXPECTED_VALIDATION_IDENTITY_DIGEST,
                 "PA development collection identity differs from panel receipt")
        development = cfg["outcome_grpo_v2"]["validation_lineage"][
            "current_development_collection"
        ]
        _require(validation.manifest_sha256 == str(development["manifest_sha256"]),
                 "PA development manifest differs from authenticated lineage")
        receipt_by_group = {
            str(receipt["group_id"]): receipt for receipt in validation.receipts
        }
        sampling_receipt = direct_v2.attach_panel_sampling_receipt(
            group_receipt, receipt_by_group,
        )
        sampler = direct_v2._build_round_robin_sampler(
            train_collections, seed=seed, rank=rank,
            contexts_per_arm=int(
                cfg["outcome_positive_advantage_audit"]["contexts_per_arm"]
            ),
        )
    except Exception as exc:  # noqa: BLE001
        local_error = f"{type(exc).__name__}: {exc}"
    v1._raise_if_any_rank_failed(local_error, world, "PA recovery collections")

    train_points: list[dict[str, Any]] = []
    train_selection_rows: list[dict[str, Any]] = []
    step_gradients: list[dict[str, Any]] = []
    nondeterminism_warnings: list[str] = []
    for offset, step in enumerate(AUDIT_STEPS):
        set_step_seed(seed, step, rank)
        local_error = ""
        try:
            fold, group_index, visit = sampler.group_at(step)
            collection = train_collections[fold]
            receipt = collection.receipts[group_index]
            replan_indices = sampler.replans_at(step, receipt["n_replans_by_arm"])
            payload = collection.load(group_index)
            authentication = direct_v2.component_audit.authenticate_selected_contexts(
                proposal, payload, replan_indices, device=device,
            )
            pa_objective = pa.sampled_positive_advantage_objective(
                proposal, payload, replan_indices, device=device,
            )
            ratio_identity = v1._require_initial_behavior_ratio_identity(
                pa_objective.metrics, device=device,
            )
            recovery_reference, recovery_second_pass_identity = (
                _recovery_reference_only(
                proposal, payload, replan_indices, device=device,
                )
            )
            demo_reference, demo_metrics = _analytic_demo_reference_only(
                demo_anchor, step,
            )
            _require(bool(demo_metrics["live_seed_logits_bitwise_identical"]),
                     f"PA offset {offset} analytic demo live/seed logits differ")
        except Exception as exc:  # noqa: BLE001
            local_error = f"{type(exc).__name__}: {exc}"
        v1._raise_if_any_rank_failed(
            local_error, world, f"PA round-robin offset {offset} forward/authentication",
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            local_error = ""
            try:
                named = direct_v2._named_live_parameters(proposal)
            except Exception as exc:  # noqa: BLE001
                local_error = f"{type(exc).__name__}: {exc}"
            v1._raise_if_any_rank_failed(
                local_error, world,
                f"PA round-robin offset {offset} gradient preflight",
            )
            direct_pa, direct_pa_missing = (
                _coordinated_synchronised_loss_gradient(
                    pa_objective.loss, named, world=world, retain_graph=True,
                    label=f"offset {offset} first direct PA",
                )
            )
            repeated_pa, repeated_pa_missing = (
                _coordinated_synchronised_loss_gradient(
                    pa_objective.loss, named, world=world, retain_graph=True,
                    label=f"offset {offset} repeated direct PA",
                )
            )
            local_error = ""
            try:
                pa_repeat = direct_v2.direct_repeat_consistency(
                    direct_pa, repeated_pa, label=f"offset {offset} PA",
                )
                pa_norm = direct_v2.component_audit._vector_norm(direct_pa)
            except Exception as exc:  # noqa: BLE001
                local_error = f"{type(exc).__name__}: {exc}"
            v1._raise_if_any_rank_failed(
                local_error, world,
                f"PA round-robin offset {offset} PA repeat evidence",
            )
            recovery_evidence = _complete_reference_gradient_evidence(
                recovery_reference, proposal, objective_norm=pa_norm,
                world=world, retain_graph=True, label="recovery PL",
            )
            demo_evidence = _complete_reference_gradient_evidence(
                demo_reference, proposal, objective_norm=pa_norm,
                world=world, retain_graph=True, label="analytic demo categorical",
            )
            local_error = ""
            try:
                direct_full_loss = (
                    pa_objective.loss + recovery_reference + demo_reference
                )
            except Exception as exc:  # noqa: BLE001
                local_error = f"{type(exc).__name__}: {exc}"
            v1._raise_if_any_rank_failed(
                local_error, world,
                f"PA round-robin offset {offset} full-scalar construction",
            )
            direct_full, direct_full_missing = (
                _coordinated_synchronised_loss_gradient(
                    direct_full_loss, named, world=world, retain_graph=True,
                    label=f"offset {offset} first direct PA full",
                )
            )
            repeated_full, repeated_full_missing = (
                _coordinated_synchronised_loss_gradient(
                    direct_full_loss, named, world=world, retain_graph=False,
                    label=f"offset {offset} repeated direct PA full",
                )
            )
            local_error = ""
            try:
                full_repeat = direct_v2.direct_repeat_consistency(
                    direct_full, repeated_full,
                    label=f"offset {offset} PA full beta1 lambda1",
                )
                _require(not demo_anchor.unexpected_gradients(),
                         "frozen demo-anchor modules retained gradients")
                torch.cuda.synchronize(device)
            except Exception as exc:  # noqa: BLE001
                local_error = f"{type(exc).__name__}: {exc}"
            v1._raise_if_any_rank_failed(
                local_error, world,
                f"PA round-robin offset {offset} full repeat evidence",
            )
        nondeterminism_warnings.extend(direct_v2._warnings_checked(
            caught, world=world, label=f"PA round-robin offset {offset} backward",
        ))
        reference_rank_rows = direct_v2.component_audit._all_gather_object({
            "rank": rank,
            "recovery_local_gradient_bitwise_zero": bool(
                recovery_evidence["local_gradient_bitwise_all_zero"]
            ),
            "recovery_synchronised_gradient_bitwise_zero": bool(
                recovery_evidence["synchronised_gradient_bitwise_all_zero"]
            ),
            "demo_local_gradient_bitwise_zero": bool(
                demo_evidence["local_gradient_bitwise_all_zero"]
            ),
            "demo_synchronised_gradient_bitwise_zero": bool(
                demo_evidence["synchronised_gradient_bitwise_all_zero"]
            ),
            "recovery_missing_parameter_gradients": list(
                recovery_evidence["missing_parameter_gradients"]
            ),
            "demo_missing_parameter_gradients": list(
                demo_evidence["missing_parameter_gradients"]
            ),
        }, world)
        _require(sorted(int(row["rank"]) for row in reference_rank_rows)
                 == list(range(world)),
                 f"PA offset {offset} reference rank evidence is incomplete")
        offset_recovery_bitwise_zero = all(
            row["recovery_local_gradient_bitwise_zero"]
            and row["recovery_synchronised_gradient_bitwise_zero"]
            for row in reference_rank_rows
        )
        offset_demo_bitwise_zero = all(
            row["demo_local_gradient_bitwise_zero"]
            and row["demo_synchronised_gradient_bitwise_zero"]
            for row in reference_rank_rows
        )
        local_error = ""
        try:
            local_selection = direct_v2._receipt_row(
                collection, group_index, rank=rank, replan_indices=replan_indices,
                extra={
                    "global_step": int(step), "offset": offset,
                    "fold": int(fold), "visit": int(visit),
                    "selected_context_authentication": authentication,
                    "ratio_identity": ratio_identity,
                },
            )
        except Exception as exc:  # noqa: BLE001
            local_error = f"{type(exc).__name__}: {exc}"
        v1._raise_if_any_rank_failed(
            local_error, world, f"PA round-robin offset {offset} receipt",
        )
        selections = direct_v2.component_audit._all_gather_object(
            local_selection, world,
        )
        _require(len({(row["split"], row["group_index"]) for row in selections})
                 == world,
                 f"PA round-robin offset {offset} selected duplicate groups")
        train_selection_rows.extend(selections)
        reduced_metrics = v1._reduce_training_metrics(
            pa_objective.metrics, world, device,
        )
        _require(reduced_metrics["ratio_min"] == 1.0
                 and reduced_metrics["ratio_mean"] == 1.0
                 and reduced_metrics["ratio_max"] == 1.0
                 and reduced_metrics["max_abs_logratio"] == 0.0,
                 f"PA offset {offset} seed ratio is not exact identity")
        mean_losses = {
            key: direct_v2._mean_scalar(value, world=world, device=device)
            for key, value in {
                "positive_advantage": float(pa_objective.loss.detach()),
                "recovery_reference": float(recovery_reference.detach()),
                "analytic_demo_reference": float(demo_reference.detach()),
                "direct_full_pa_beta1_lambda1": float(direct_full_loss.detach()),
            }.items()
        }
        train_point_error = ""
        if rank == 0:
            try:
                train_points.append({
                "offset": offset,
                "global_step": int(step),
                "rank_local_groups": selections,
                "fold_counts": {
                    str(fold_index): sum(
                        int(row["fold"]) == fold_index for row in selections
                    ) for fold_index in range(v1.N_FOLDS)
                },
                "global_mean_losses": mean_losses,
                "global_ratio_identity": {
                    "ratio_atoms": int(reduced_metrics["ratio_atoms"]),
                    "ratio_min": reduced_metrics["ratio_min"],
                    "ratio_mean": reduced_metrics["ratio_mean"],
                    "ratio_max": reduced_metrics["ratio_max"],
                    "max_abs_logratio": reduced_metrics["max_abs_logratio"],
                    "all_exact": True,
                },
                "positive_advantage_metrics_rank0": dict(pa_objective.metrics),
                "reference_gradients": {
                    "recovery_pl_forward_kl": recovery_evidence,
                    "demo_exact_analytic_vjp_categorical_forward_kl": demo_evidence,
                    "all_rank_local_and_synchronised_zero_evidence": (
                        reference_rank_rows
                    ),
                    "recovery_all_rank_local_and_synchronised_bitwise_zero": (
                        offset_recovery_bitwise_zero
                    ),
                    "demo_all_rank_local_and_synchronised_bitwise_zero": (
                        offset_demo_bitwise_zero
                    ),
                    "recovery_bitwise_zero_required_for_pass": True,
                    "demo_bitwise_zero_required_for_pass": True,
                    "complete_finite_nonzero_reference_vjp_classification": (
                        "scientific_abort_not_execution_invalid"
                    ),
                },
                "two_pass_train_scoring_identity": {
                    "geometry": (
                        "read-only authentication replay plus two objective-"
                        "graph exact B1 scorer passes per train point"
                    ),
                    "authentication_replay": (
                        "exact selected-context identity; no objective graph"
                    ),
                    "first_objective_graph_pass": (
                        "authoritative_positive_advantage"
                    ),
                    "second_objective_graph_pass": "recovery_k3_only",
                    "same_selected_rows_order_and_stored_old": True,
                    "second_pass_current_old_identity": (
                        recovery_second_pass_identity
                    ),
                    "second_pass_recovery_value_exactly_zero": (
                        recovery_evidence["value_exactly_zero"]
                    ),
                    "second_pass_recovery_gradient_bitwise_zero": (
                        offset_recovery_bitwise_zero
                    ),
                },
                "direct_positive_advantage_gradient": {
                    "authoritative_vector": "first_direct_backward",
                    "scalar": "positive_advantage_conditional_imitation",
                    "preclip_norm": pa_norm,
                    "missing_parameter_gradients": direct_pa_missing,
                    "repeated_missing_parameter_gradients": repeated_pa_missing,
                    "repeat_consistency": pa_repeat,
                },
                "direct_full_pa_beta1_lambda1_gradient": {
                    "authoritative_vector": "first_direct_backward",
                    "authoritative_for_full_gated_endpoints": True,
                    "scalar": (
                        "positive_advantage + 1*recovery_reference + "
                        "1*exact_analytic_vjp_demo_reference"
                    ),
                    "forbidden_terms_absent": {
                        "grpo_train": True,
                        "switch_balance": True,
                        "sparse_ce": True,
                    },
                    "preclip_norm": direct_v2.component_audit._vector_norm(
                        direct_full
                    ),
                    "missing_parameter_gradients": direct_full_missing,
                    "repeated_missing_parameter_gradients": repeated_full_missing,
                    "repeat_consistency": full_repeat,
                    "reference_relative_bounds_passed": (
                        recovery_evidence["bound_passed"]
                        and demo_evidence["bound_passed"]
                    ),
                    "recovery_second_pass_gradient_bitwise_zero": (
                        offset_recovery_bitwise_zero
                    ),
                    "analytic_demo_vjp_gradient_bitwise_zero": (
                        offset_demo_bitwise_zero
                    ),
                },
                "analytic_demo_reference_metrics_rank0": demo_metrics,
                })
            except Exception as exc:  # noqa: BLE001
                train_point_error = f"{type(exc).__name__}: {exc}"
        v1._raise_if_any_rank_failed(
            train_point_error, world, f"PA offset {offset} report-row construction",
        )
        local_error = ""
        try:
            step_gradient_row = {
                "direct_pa": direct_pa.detach().clone(),
                "direct_full": direct_full.detach().clone(),
                "recovery_bound_passed": bool(recovery_evidence["bound_passed"]),
                "recovery_bitwise_zero": bool(offset_recovery_bitwise_zero),
                "demo_bound_passed": bool(demo_evidence["bound_passed"]),
                "demo_bitwise_zero": bool(offset_demo_bitwise_zero),
            }
        except Exception as exc:  # noqa: BLE001
            local_error = f"{type(exc).__name__}: {exc}"
        v1._raise_if_any_rank_failed(
            local_error, world, f"PA offset {offset} retained-gradient clone",
        )
        local_error = ""
        try:
            step_gradients.append(step_gradient_row)
            del payload, pa_objective, recovery_reference, demo_reference
            del direct_full_loss, repeated_pa, repeated_full
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            local_error = f"{type(exc).__name__}: {exc}"
        v1._raise_if_any_rank_failed(
            local_error, world, f"PA offset {offset} post-gradient bookkeeping",
        )

    local_error = ""
    try:
        _require(len(train_selection_rows) == EXPECTED_TRAIN_DRAWS,
                 "PA audit did not retain exactly 24 TRAIN draws")
        _require(len({(row["split"], int(row["group_index"]))
                      for row in train_selection_rows}) == EXPECTED_TRAIN_DRAWS,
                 "PA TRAIN draws are not all distinct")
        fold_totals = {
            fold: sum(int(row["fold"]) == fold for row in train_selection_rows)
            for fold in range(v1.N_FOLDS)
        }
        _require(all(value == EXPECTED_FOLD_DRAWS for value in fold_totals.values()),
                 f"PA three-offset fold mixture differs: {fold_totals}")
        pa_gradients = [row["direct_pa"] for row in step_gradients]
        full_gradients = [row["direct_full"] for row in step_gradients]
        all_reference_bounds = all(
            row["recovery_bound_passed"] and row["demo_bound_passed"]
            for row in step_gradients
        )
        all_demo_bitwise_zero = all(
            row["demo_bitwise_zero"] for row in step_gradients
        )
        all_recovery_bitwise_zero = all(
            row["recovery_bitwise_zero"] for row in step_gradients
        )
    except Exception as exc:  # noqa: BLE001
        local_error = f"{type(exc).__name__}: {exc}"
    v1._raise_if_any_rank_failed(local_error, world, "PA 24-draw gradient closure")
    local_error = ""
    try:
        pa_sgd = direct_v2.cumulative_clipped_sgd_direction(
            proposal, pa_gradients, cfg=cfg,
        )
        full_sgd = direct_v2.cumulative_clipped_sgd_direction(
            proposal, full_gradients, cfg=cfg,
        )
        adamw = direct_v2.virtual_adamw_clone_replay(
            proposal, full_gradients, cfg=cfg,
        )
        for expected, actual in zip(
            full_sgd["clipped_gradients"],
            adamw["vectors"]["clipped_gradients"], strict=True,
        ):
            _require(torch.equal(expected, actual),
                     "PA SGD and AdamW replay clipped gradients differ")

        primary_vectors = {
            "pa_only_cumulative_clipped_sgd": pa_sgd["delta"],
            "pa_full_beta1_lambda1_cumulative_clipped_sgd": full_sgd["delta"],
            "pa_full_beta1_lambda1_reset_adamw_with_production_decay": (
                adamw["vectors"]["cumulative"][-1]
            ),
        }
        increment_vectors = {
            f"pa_full_beta1_lambda1_adamw_decay_increment_t{offset}": value
            for offset, value in enumerate(adamw["vectors"]["increments"])
        }
        all_directions = {**primary_vectors, **increment_vectors}
        direction_norms = {
            key: direct_v2.component_audit._vector_norm(value)
            for key, value in all_directions.items()
        }
        direction_vector_sha256 = {
            key: exact_float32_vector_sha256(value)
            for key, value in all_directions.items()
        }
        direction_digest = _canonical_sha256({
            key: {
                "norm": direction_norms[key],
                "numel": int(value.numel()),
                "exact_float32_vector_sha256": direction_vector_sha256[key],
            }
            for key, value in all_directions.items()
        })
    except Exception as exc:  # noqa: BLE001
        local_error = f"{type(exc).__name__}: {exc}"
    v1._raise_if_any_rank_failed(
        local_error, world, "PA clipped-SGD/AdamW/direction construction",
    )
    direction_rows = direct_v2.component_audit._all_gather_object({
        "rank": rank,
        "digest": direction_digest,
        "norms": direction_norms,
        "exact_float32_vector_sha256": direction_vector_sha256,
    }, world)
    _require(all(row["digest"] == direction_digest for row in direction_rows),
             "PA exact direction vectors differ across ranks")

    local_error = ""
    try:
        sampling_by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in sampling_receipt["rows"]:
            if int(row["rank"]) == rank:
                sampling_by_task[str(row["task_key"])].append(row)
        _require(len(sampling_by_task) == PANEL_TASKS_PER_RANK,
                 f"rank {rank} did not receive five panel tasks")
        _require(sum(len(rows) for rows in sampling_by_task.values()) == 6,
                 f"rank {rank} did not receive exactly six panel groups")
        _require(all(len(rows) in {1, 2} for rows in sampling_by_task.values())
                 and sum(len(rows) == 2 for rows in sampling_by_task.values()) == 1,
                 f"rank {rank} panel assignment is not four singles plus one double")
        panel_local_task_rows: list[dict[str, Any]] = []
        panel_local_selection_rows: list[dict[str, Any]] = []
        heldout_local = torch.zeros_like(next(iter(primary_vectors.values())))
        group_index_by_id = {
            str(receipt["group_id"]): index
            for index, receipt in enumerate(validation.receipts)
        }
    except Exception as exc:  # noqa: BLE001
        local_error = f"{type(exc).__name__}: {exc}"
    v1._raise_if_any_rank_failed(local_error, world, "PA panel six-slot setup")
    panel_aggregation_error = ""
    for task_index, key in enumerate(sorted(sampling_by_task)):
        group_gradients: list[Tensor] = []
        group_losses: list[float] = []
        task_authentication: list[dict[str, Any]] = []
        for selected in sampling_by_task[key]:
            group_id = str(selected["group_id"])
            local_error = ""
            try:
                group_index = group_index_by_id[group_id]
                receipt = validation.receipts[group_index]
                _require(
                    str(receipt["sha256"]) == str(selected["sidecar_sha256"]),
                    f"PA panel sidecar SHA differs at use: {group_id}",
                )
                replans = {
                    int(arm): tuple(int(value) for value in values)
                    for arm, values in selected["replan_indices"].items()
                }
                set_step_seed(seed, v2.START_STEP + 100 + task_index, rank)
                payload = validation.load(group_index)
                authentication = direct_v2.component_audit.authenticate_selected_contexts(
                    proposal, payload, replans, device=device,
                )
                heldout_objective = v2.sampled_group_objectives_v2(
                    proposal, payload, replans, device=device,
                )
                ratio_identity = v1._require_initial_behavior_ratio_identity(
                    heldout_objective.metrics, device=device,
                )
                local_gradient, missing = direct_v2.component_audit._local_gradient_vector(
                    heldout_objective.grpo,
                    direct_v2._named_live_parameters(proposal),
                    retain_graph=False,
                )
                _require(not missing,
                         f"panel signed-GRPO gradient missing: {missing[:8]}")
                _require(all(parameter.grad is None for parameter in proposal.parameters()),
                         "PA panel autograd populated live gradients")
                group_loss = float(heldout_objective.grpo.detach())
                authentication_row = {
                    "group_id": group_id,
                    "selected_context_authentication": authentication,
                    "ratio_identity": ratio_identity,
                }
                selection_row = direct_v2._receipt_row(
                    validation, group_index, rank=rank, replan_indices=replans,
                    extra={
                        "task_key": key, "panel_role": selected["panel_role"]
                    },
                )
            except Exception as exc:  # noqa: BLE001
                local_error = f"{type(exc).__name__}: {exc}"
            v1._raise_if_any_rank_failed(
                local_error, world,
                f"PA panel task {key} group {group_id} authentication/gradient",
            )
            group_gradients.append(local_gradient)
            group_losses.append(group_loss)
            task_authentication.append(authentication_row)
            panel_local_selection_rows.append(selection_row)
            del payload, heldout_objective, local_gradient
        task_gradient: Tensor | None = None
        try:
            task_gradient, task_contribution = (
                direct_v2.equal_group_within_task_contribution(group_gradients)
            )
            heldout_local.add_(task_contribution)
            projections = {
                name: direct_v2.component_audit._dot(task_gradient, direction)
                for name, direction in all_directions.items()
            }
            panel_local_task_rows.append({
                "task_key": key,
                "suite": direct_v2.task_suite(key),
                "rank": rank,
                "groups": len(group_gradients),
                "equal_group_within_task_signed_grpo_loss": (
                    sum(group_losses) / len(group_losses)
                ),
                "task_signed_grpo_gradient_norm": (
                    direct_v2.component_audit._vector_norm(task_gradient)
                ),
                "authentication": task_authentication,
                "projections": projections,
            })
        except Exception as exc:  # noqa: BLE001
            if not panel_aggregation_error:
                panel_aggregation_error = f"{type(exc).__name__}: {exc}"
        del group_gradients
        if task_gradient is not None:
            del task_gradient
        gc.collect()
        torch.cuda.empty_cache()

    v1._raise_if_any_rank_failed(
        panel_aggregation_error, world, "PA panel local task aggregation/dots",
    )
    local_error = ""
    try:
        heldout_gradient = direct_v2._sum_synchronised_gradient(
            heldout_local, world=world,
        )
    except Exception as exc:  # noqa: BLE001
        local_error = f"{type(exc).__name__}: {exc}"
    v1._raise_if_any_rank_failed(
        local_error, world, "PA synchronised heldout gradient",
    )
    gathered_tasks = direct_v2.component_audit._all_gather_object(
        panel_local_task_rows, world,
    )
    local_error = ""
    try:
        panel_task_rows = [
            row for rank_rows in gathered_tasks for row in rank_rows
        ]
        panel_task_rows.sort(key=lambda row: str(row["task_key"]))
        _require(
            len(panel_task_rows) == PANEL_TASKS
            and len({row["task_key"] for row in panel_task_rows}) == PANEL_TASKS,
            "PA panel does not cover all 40 tasks exactly once",
        )
        _require(
            sum(int(row["groups"]) for row in panel_task_rows) == PANEL_GROUPS,
            "PA panel group-within-task multiplicity differs",
        )
    except Exception as exc:  # noqa: BLE001
        local_error = f"{type(exc).__name__}: {exc}"
    v1._raise_if_any_rank_failed(
        local_error, world, "PA panel gathered-task closure",
    )
    gathered_selections = direct_v2.component_audit._all_gather_object(
        panel_local_selection_rows, world,
    )
    local_error = ""
    try:
        analysis = analyse_panel_directions(
            panel_task_rows=panel_task_rows,
            gathered_panel_selections=gathered_selections,
            primary_vectors=primary_vectors,
            increment_vectors=increment_vectors,
            heldout_gradient=heldout_gradient,
            all_reference_relative_bounds_passed=all_reference_bounds,
            all_recovery_second_pass_gradients_bitwise_zero=(
                all_recovery_bitwise_zero
            ),
            all_demo_analytic_vjp_gradients_bitwise_zero=all_demo_bitwise_zero,
        )
    except Exception as exc:  # noqa: BLE001
        local_error = f"{type(exc).__name__}: {exc}"
    v1._raise_if_any_rank_failed(
        local_error, world, "PA post-panel bootstrap/projection analysis",
    )
    panel_selection_rows = analysis["panel_selection_rows"]
    bootstrap_receipt = analysis["bootstrap_receipt"]
    primary_names = analysis["primary_names"]
    endpoint_bounds = analysis["endpoint_bounds"]
    increment_names = analysis["increment_names"]
    increment_bounds = analysis["increment_bounds"]
    endpoint_cosines = analysis["endpoint_cosines"]
    increment_cosines = analysis["increment_cosines"]
    decision = analysis["decision"]
    projection_closure = analysis["projection_closure"]

    local_error = ""
    try:
        collection_map = {
            collection.split: collection for collection in train_collections
        }
        collection_map[validation.split] = validation
        all_selection_rows = [*train_selection_rows, *panel_selection_rows]
        rehash_local = direct_v2.rehash_rank_sidecars(
            collection_map, all_selection_rows, rank=rank,
        )
    except Exception as exc:  # noqa: BLE001
        local_error = f"{type(exc).__name__}: {exc}"
    v1._raise_if_any_rank_failed(local_error, world, "PA 72-sidecar rehash")
    rehash_rows = direct_v2.component_audit._all_gather_object(rehash_local, world)
    _require(sum(int(row["selected_sidecars"]) for row in rehash_rows) == 72,
             "PA post-use closure did not rehash all 72 sidecars")

    local_error = ""
    try:
        proposal_digest_after = v1.proposal_module_digest(proposal.state_dict())
        demo_anchor.assert_seed_unchanged()
        _require(proposal_digest_after == proposal_digest_before,
                 "live proposal changed during PA diagnostic")
        _require(len(live_optimizer_sentinel.state) == 0,
                 "live PA optimizer sentinel accumulated state")
        _require(all(parameter.grad is None for parameter in proposal.parameters()),
                 "live proposal retained PA gradient buffers")
        _require(not demo_anchor.unexpected_gradients(),
                 "frozen PA anchor/reference modules retained gradients")
        for collection in train_collections:
            collection.assert_unchanged()
        validation.assert_unchanged()
        v1._assert_seed_stat(parent_identity)
        _require(_source_identity() == source_identity,
                 "PA source closure changed during diagnostic")
        _require(_config_file_identity(config) == config_file_identity,
                 "PA config stat/SHA changed during diagnostic")
        construction_cfg_after, construction_receipt_after = (
            _authenticated_demo_anchor_construction_config(cfg)
        )
        _require(
            all(demo_anchor_construction.get(key) == value
                for key, value in construction_receipt_after.items()),
            "demo-anchor construction source/config changed during diagnostic",
        )
        del construction_cfg_after
        _require(v1._strict_outcome_determinism_state()
                 == v1.STRICT_OUTCOME_DETERMINISM,
                 "strict deterministic flags changed during PA diagnostic")
        direct_v2._exact_scoring_evidence(proposal, device)
        torch.cuda.synchronize(device)
    except Exception as exc:  # noqa: BLE001
        local_error = f"{type(exc).__name__}: {exc}"
    v1._raise_if_any_rank_failed(local_error, world, "final PA no-mutation closure")
    # This helper contains its own coordinated error gather and therefore must
    # not sit inside a rank-local try that another rank could leave early.
    assert_trigger_unchanged(trigger_identity, rank=rank, world=world)
    local_error = ""
    try:
        mutation_local = {
            "rank": rank,
            "proposal_digest_before": proposal_digest_before,
            "proposal_digest_after": proposal_digest_after,
            "live_optimizer_state_entries": len(live_optimizer_sentinel.state),
            "live_parameter_grad_buffers": sum(
                parameter.grad is not None for parameter in proposal.parameters()
            ),
        }
    except Exception as exc:  # noqa: BLE001
        local_error = f"{type(exc).__name__}: {exc}"
    v1._raise_if_any_rank_failed(local_error, world, "PA mutation rank receipt")
    mutation_rows = direct_v2.component_audit._all_gather_object(
        mutation_local, world,
    )

    report: dict[str, Any] | None = None
    publication_error = ""
    if rank == 0:
        try:
            _require(len(train_points) == len(AUDIT_STEPS),
                     "rank zero did not retain three PA TRAIN points")
            report = {
        "format_version": FORMAT_VERSION,
        "kind": KIND,
        "status": decision["status"],
        "execution_validated": True,
        "decision": decision,
        "eligibility": dict(ELIGIBILITY),
        "instrumentation_history": list(INVALID_INSTRUMENTATION_HISTORY),
        "trigger": {
            **trigger_identity,
            "decision_input": True,
            "required_sha256": TRIGGER_REPORT_SHA256,
            "required_status": TRIGGER_STATUS,
            "authenticated_before_recovery_manifest_or_sidecar_read": True,
            "post_use_stat_sha_semantics_rechecked": True,
        },
        "source_identity": source_identity,
        "config": {
            "path": str(config),
            "resolved_hash": config_hash,
            "raw_file_identity": config_file_identity,
            "scaffold_validation": scaffold_validation,
            "nonlaunchable": True,
            "objective": "positive_advantage_conditional_imitation",
            "weights": {
                "positive_advantage": PA_WEIGHT,
                "recovery_reference": RECOVERY_REFERENCE_WEIGHT,
                "exact_analytic_vjp_demo_reference": DEMO_REFERENCE_WEIGHT,
                "grpo_train": GRPO_TRAIN_WEIGHT,
                "switch_balance": BALANCE_WEIGHT,
                "sparse_ce": SPARSE_CE_WEIGHT,
            },
            "seed_checkpoint": EXPECTED_SEED_CHECKPOINT,
            "seed_global_step": v2.START_STEP,
            "demo_anchor_construction": demo_anchor_construction,
        },
        "parent": parent_identity,
        "runtime_by_rank": runtimes,
        "strict_determinism": strict,
        "exact_scoring": {
            "configured": scoring_config,
            "validated": scoring_evidence,
            "all_train_and_panel_selected_contexts_bitwise_authenticated": True,
            "all_seed_ratios_exactly_one": True,
        },
        "geometry": {
            "world_size": world,
            "nodes": 1,
            "gpus_per_node": world,
            "global_steps": list(AUDIT_STEPS),
            "round_robin_offsets": list(AUDIT_OFFSETS),
            "train_draws": EXPECTED_TRAIN_DRAWS,
            "train_draws_per_fold": EXPECTED_FOLD_DRAWS,
            "fold_draw_counts": {str(key): value for key, value in fold_totals.items()},
            "panel_tasks": PANEL_TASKS,
            "panel_groups": PANEL_GROUPS,
            "panel_extra_second_groups": direct_v2.PANEL_EXTRA_GROUPS,
            "panel_contexts_per_sampled_arm": direct_v2.PANEL_REPLANS_PER_ARM,
            "equal_group_within_task_then_equal_task": True,
        },
        "outcome_blind_panel": {
            "terminal_rewards_not_used_or_accessed_by_selection_logic": True,
            "manifest_parser_materializes_unprojected_fields": True,
            "group_receipt": group_receipt,
            "sampling_receipt": sampling_receipt,
            "collection": validation.provenance(),
            "task_rows": panel_task_rows,
        },
        "train_collections": [
            collection.provenance() for collection in train_collections
        ],
        "train_points": train_points,
        "anchor_preflight_by_rank": anchor_preflight_rows,
        "direction_construction": {
            "measurement_scope": "authoritative_direct_pa_gradients_only",
            "direct_pa_scalar": "positive_advantage_conditional_imitation",
            "direct_full_scalar": (
                "positive_advantage + 1*recovery_reference + "
                "1*exact_analytic_vjp_demo_reference"
            ),
            "first_direct_vectors_authoritative": True,
            "train_scoring_geometry": (
                "read-only authentication replay plus two objective-graph "
                "exact B1 scorer passes per train point"
            ),
            "authentication_replay": (
                "exact selected-context identity; no objective graph"
            ),
            "first_objective_graph_scorer_pass": (
                "frozen_PA_API_authoritative_same_selected_rows_order_old"
            ),
            "second_objective_graph_scorer_pass": (
                "recovery_k3_only_same_selected_rows_order_old"
            ),
            "second_pass_scorer_authentication": (
                "exact current.float equals stored old for every selected row; "
                "any mismatch is INVALID_NO_REPORT"
            ),
            "second_pass_identity_evidence": (
                "per-offset exact-zero recovery value and bitwise-zero "
                "synchronised gradient reported; no component-sum PA/full "
                "equality gate"
            ),
            "independent_repeat_checks": {
                "direct_positive_advantage": True,
                "direct_full_pa_beta1_lambda1": True,
                "max_relative_residual": (
                    direct_v2.MAX_DIRECT_REPEAT_RELATIVE_RESIDUAL
                ),
            },
            "forbidden_train_terms": {
                "grpo": {"weight": 0.0, "computed": False, "graph_included": False},
                "switch_balance": {
                    "weight": 0.0, "computed": False, "graph_included": False,
                },
                "sparse_ce": {
                    "weight": 0.0, "computed": False, "graph_included": False,
                },
            },
            "heldout_signed_grpo_is_evaluation_derivative_only": True,
            "coefficient_tuning_or_switching": False,
            "reference_relative_bounds_passed": all_reference_bounds,
            "recovery_second_pass_gradients_bitwise_zero": (
                all_recovery_bitwise_zero
            ),
            "analytic_demo_vjp_gradients_bitwise_zero": all_demo_bitwise_zero,
            "reference_identity_classification": {
                "second_pass_current_old_or_demo_live_seed_logit_mismatch": (
                    "INVALID_NO_REPORT"
                ),
                "missing_disconnected_nonfinite_auth_or_ratio_failure": (
                    "INVALID_NO_REPORT"
                ),
                "complete_finite_nonzero_recovery_or_demo_vjp": (
                    "SCIENTIFIC_ABORT"
                ),
                "pass_requires_local_and_synchronised_bitwise_zero": True,
            },
            "cross_rank_exact_direction_identity": {
                "passed": True,
                "canonical_digest": direction_digest,
                "exact_float32_vector_sha256": direction_vector_sha256,
                "rank_evidence": direction_rows,
            },
            "primary_endpoint_order": list(primary_names),
            "primary_endpoint_norms": {
                name: direction_norms[name] for name in primary_names
            },
            "pa_only_clipped_sgd": {
                key: value for key, value in pa_sgd.items()
                if key not in {"delta", "clipped_gradients"}
            },
            "pa_full_beta1_lambda1_clipped_sgd": {
                key: value for key, value in full_sgd.items()
                if key not in {"delta", "clipped_gradients"}
            },
            "reset_adamw_with_production_decay": _without_vectors(adamw),
            "live_optimizer_steps": 0,
            "virtual_clone_optimizer_steps": 3,
            "frozen_gradient_replay_caveat": (
                "all three gradients are measured at the unchanged seed; clone "
                "AdamW does not recompute gradients at virtual parameters"
            ),
        },
        "primary_endpoint_task_bootstrap": endpoint_bounds,
        "production_adamw_increment_task_bootstrap": increment_bounds,
        "aggregate_heldout_gradient": {
            "objective": "signed_grpo_negative_clipped_surrogate_loss",
            "role": "development_projection_only_not_train_direction",
            "definition": "equal group within task, then equal mean over 40 tasks",
            "norm": direct_v2.component_audit._vector_norm(heldout_gradient),
            "primary_endpoint_benefit_cosines": {
                name: endpoint_cosines[index]
                for index, name in enumerate(primary_names)
            },
            "production_adamw_increment_benefit_cosines": {
                name: increment_cosines[index]
                for index, name in enumerate(increment_names)
            },
            "projection_closure": projection_closure,
        },
        "bootstrap_resample_matrix": bootstrap_receipt,
        "selected_sidecar_post_use_closure": {
            "passed": True,
            "sidecars": 72,
            "rank_evidence": rehash_rows,
            "bytes": sum(int(row["selected_bytes"]) for row in rehash_rows),
        },
        "no_mutation": {
            "passed": True,
            "live_optimizer_constructed_as_empty_state_sentinel": True,
            "live_optimizer_steps": 0,
            "virtual_clone_optimizer_steps": 3,
            "live_parameter_perturbations": 0,
            "proposal_digest_before": proposal_digest_before,
            "proposal_digest_after": proposal_digest_after,
            "rank_evidence": mutation_rows,
            "checkpoint_emitted": False,
            "candidate_emitted": False,
        },
        "warnings": sorted(set(nondeterminism_warnings)),
        "wall_seconds": float(time.monotonic() - started),
        "output": str(output),
            }
            exclusive_json_write(output, report)
            print(json.dumps(
                report, indent=2, sort_keys=True, allow_nan=False,
            ), flush=True)
        except Exception as exc:  # noqa: BLE001
            publication_error = f"{type(exc).__name__}: {exc}"
    v1._raise_if_any_rank_failed(
        publication_error, world, "rank-zero PA report publication",
    )
    torch.distributed.barrier()
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / EXPECTED_CONFIG_REL,
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    exit_code = 0
    try:
        run_audit(config_path=args.config, output_path=args.out)
    except Exception as exc:  # noqa: BLE001
        print(
            "OUTCOME_POSITIVE_ADVANTAGE_DIRECTION_AUDIT_INVALID: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        exit_code = 2
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
