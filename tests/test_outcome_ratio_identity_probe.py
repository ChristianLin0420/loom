"""CPU contracts for the read-only real-A100 outcome ratio probe."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

import contracts as C
from loom.eval import outcome_recovery as recovery
from loom.heads.proposal import pl_log_prob
from loom.train import outcome_grpo as grpo
from scripts import outcome_ratio_identity_probe as probe


class TinyProposal(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.operator_logits = nn.Parameter(torch.linspace(-2.0, 2.0, C.M))

    def logits(self, z: Tensor, lang: Tensor) -> Tensor:  # noqa: ARG002
        return self.operator_logits.unsqueeze(0).expand(z.shape[0], -1)


def _identity_payload(proposal: TinyProposal, *, n: int = 3) -> dict:
    proposal.eval()
    z = torch.arange(n, dtype=torch.float32).reshape(n, 1, 1)
    lang = torch.zeros(1, 1, dtype=torch.float32)
    with torch.no_grad():
        logits = torch.cat([
            proposal.logits(z[row:row + 1], lang.reshape(1, 1, 1))
            for row in range(n)
        ])
        order = logits.topk(C.TOPK, dim=-1).indices
        old = pl_log_prob(logits.float(), order)
        coeff = grpo.weights_from_logits(logits.float(), order, C.M)
    return {
        "group_id": "cpu-identity",
        "arms": [{
            "z": z.clone(),
            "lang": lang.clone(),
            "ordered_support": order.clone(),
            "old_logprob": old.clone(),
            "coeff": coeff.clone(),
            "terminal_reward": torch.tensor(float(arm == 0)),
        } for arm in range(recovery.GROUP_SIZE)],
    }


class ScanCollection:
    def __init__(self, *payloads: dict) -> None:
        self.split = "train0"
        self.payloads = payloads
        self.loaded: list[int] = []
        self.receipts = tuple({
            "group_id": payload["group_id"],
            "sidecar": f"groups/group{index}.pt",
            "sha256": f"{index + 1:064x}",
        } for index, payload in enumerate(payloads))

    def load(self, index: int) -> dict:
        self.loaded.append(index)
        return self.payloads[index]


def _fixed_receipt() -> dict:
    return {
        "group_id": probe.EXPECTED_GROUP_ID,
        "sidecar": probe.EXPECTED_SIDECAR,
        "sha256": probe.EXPECTED_SIDECAR_SHA256,
        "size": probe.EXPECTED_SIDECAR_SIZE,
        "n_arms": recovery.GROUP_SIZE,
        "n_replans_by_arm": list(probe.EXPECTED_REPLANS),
        "terminal_rewards": list(probe.EXPECTED_REWARDS),
        "worker": {"test": True},
    }


def test_fixed_train0_group_and_rank0_indices_are_pinned():
    assert probe.CHECKPOINT_REL.endswith("ckpt_000049666.pt")
    assert probe.COLLECTION_REL.endswith("outcome_recovery_s49666_train0")
    assert probe.EXPECTED_GROUP_INDEX == 19
    assert probe.EXPECTED_GROUP_ID == "libero_spatial/task=03/trial=14/seed=0"
    assert probe.EXPECTED_SIDECAR_SHA256 == (
        "6cdb7ac21d2469f2c104ed47dd92903029d9569d2bedc2ba0def5aecb00cb2ef"
    )
    assert probe.EXPECTED_REPLAN_INDICES == {
        1: (88, 93), 2: (69, 11), 3: (67, 15), 4: (50, 2),
        5: (93, 0), 6: (69, 59), 7: (68, 65),
    }
    assert set(probe.EXPECTED_REPLAN_INDICES) == set(range(1, 8))
    assert sum(map(len, probe.EXPECTED_REPLAN_INDICES.values())) == 14
    assert probe.EXPECTED_CONFIG_HASH == "25afdedfc9deea5e"
    assert probe.EXPECTED_TRAINER_SOURCE_SHA256 == (
        "d5ef53e9f2e276f17d68f80b4c081c8f09b0d89ea9a966214fc3b63387364a52"
    )


def test_probe_is_bound_to_current_recipe_and_trainer_source():
    identity = probe.validated_source_identity()
    assert identity["config_hash"] == probe.EXPECTED_CONFIG_HASH
    assert identity["trainer_source"]["sha256"] == (
        probe.EXPECTED_TRAINER_SOURCE_SHA256
    )
    assert identity["probe_source"]["path"].endswith(
        "scripts/outcome_ratio_identity_probe.py"
    )
    assert len(identity["probe_source"]["sha256"]) == 64


def test_selection_uses_production_step_rank_world_and_checks_receipt(monkeypatch):
    seen = {}

    class FakeSampler:
        def __init__(self, groups, **kwargs):
            seen["groups"] = groups
            seen.update(kwargs)

        def group_at(self, step):
            seen["group_step"] = step
            return 0, probe.EXPECTED_GROUP_INDEX, 0

        def replans_at(self, step, replans):
            seen["replan_step"] = step
            seen["replans"] = tuple(replans)
            return dict(probe.EXPECTED_REPLAN_INDICES)

    monkeypatch.setattr(grpo, "DeterministicOutcomeSampler", FakeSampler)
    receipts = [{} for _ in range(probe.EXPECTED_GROUP_INDEX)] + [_fixed_receipt()]
    collection = SimpleNamespace(
        identity_digest=probe.EXPECTED_COLLECTION_IDENTITY_DIGEST,
        receipts=receipts,
        informative_indices=lambda: (0, 1, 2),
    )
    group, visit, indices, receipt = probe.deterministic_rank0_selection(collection)
    assert (group, visit) == (probe.EXPECTED_GROUP_INDEX, 0)
    assert indices == probe.EXPECTED_REPLAN_INDICES
    assert receipt["sha256"] == probe.EXPECTED_SIDECAR_SHA256
    assert seen["seed"] == grpo.TRAIN_SEED
    assert seen["rank"] == 0 and seen["world_size"] == grpo.EXPECTED_WORLD_SIZE
    assert seen["start_step"] == seen["group_step"] == seen["replan_step"] == 49_666
    assert seen["contexts_per_arm"] == 2

    receipts[-1] = {**_fixed_receipt(), "sha256": "0" * 64}
    with pytest.raises(probe.ProbeError, match="receipt sha256 drifted"):
        probe.deterministic_rank0_selection(collection)


def test_pl_witness_scan_transfers_multiple_rows_but_scores_production_pl_b1(
    monkeypatch,
):
    device = torch.device("cpu")
    grpo._configure_exact_proposal_scoring(device)
    proposal = TinyProposal().eval()
    payload = _identity_payload(proposal)
    collection = ScanCollection(payload)
    real_pl = grpo.pl_log_prob
    observed_batches: list[int] = []

    def batch_sensitive_pl(logits: Tensor, order: Tensor) -> Tensor:
        batch = int(logits.shape[0])
        observed_batches.append(batch)
        score = real_pl(logits, order)
        return score + (0.25 if batch > 1 else 0.0)

    monkeypatch.setattr(grpo, "pl_log_prob", batch_sensitive_pl)
    report = probe.scan_train0_pl_witness(
        proposal, collection, device=device,
    )

    assert collection.loaded == [0]
    assert report["passed"] is True
    assert report["scan_order"] == "manifest_group_then_arm1_to_7_then_replan"
    assert report["stop_condition"] == "first_transfer_chunk_with_legacy_mismatch"
    assert report["groups_scanned"] == report["chunks_scanned"] == 1
    assert report["rows_scanned"] == 3
    assert report["geometry"] == {
        "transfer_chunk_replans": 32,
        "transfer_chunk_gt_one": True,
        "actual_witness_transfer_rows": 3,
        "proposal_batch_size": 1,
        "production_pl_batch_size": 1,
        "observed_production_pl_batch_sizes": [1],
    }
    assert observed_batches == [1, 1, 1, 3]
    assert grpo.pl_log_prob is batch_sensitive_pl
    assert report["legacy_batched_pl"] == {
        "batch_size": 3,
        "mismatch_count": 3,
        "max_abs_old_logprob_error": 0.25,
    }
    assert report["fixed_rowwise"] == {
        "rows_checked": 3,
        "max_abs_coeff_error": 0.0,
        "max_abs_old_logprob_error": 0.0,
        "all_exact": True,
    }
    assert [(row["group_index"], row["arm"], row["replan"])
            for row in report["witness_rows"]] == [
        (0, 1, 0), (0, 1, 1), (0, 1, 2),
    ]
    assert all(row["legacy_abs_error"] == 0.25
               and row["fixed_abs_error"] == 0.0
               and row["fixed_coeff_max_abs_error"] == 0.0
               for row in report["witness_rows"])


@pytest.mark.parametrize("field", ["coeff", "old_logprob"])
def test_pl_witness_scan_requires_bitwise_exact_production_replay(
    monkeypatch, field,
):
    device = torch.device("cpu")
    grpo._configure_exact_proposal_scoring(device)
    proposal = TinyProposal().eval()
    payload = _identity_payload(proposal)
    payload["arms"][1][field][0].add_(0.25)
    real_pl = grpo.pl_log_prob

    def batch_sensitive_pl(logits: Tensor, order: Tensor) -> Tensor:
        score = real_pl(logits, order)
        return score + (0.25 if logits.shape[0] > 1 else 0.0)

    monkeypatch.setattr(grpo, "pl_log_prob", batch_sensitive_pl)
    message = "coefficient replay was not exact" if field == "coeff" \
        else "old-logprob replay was not exact"
    with pytest.raises(probe.ProbeError, match=message):
        probe.scan_train0_pl_witness(
            proposal, ScanCollection(payload), device=device,
        )


def test_pl_witness_scan_never_passes_without_a_legacy_mismatch():
    device = torch.device("cpu")
    grpo._configure_exact_proposal_scoring(device)
    proposal = TinyProposal().eval()
    with pytest.raises(probe.ProbeError, match="did not reproduce a legacy"):
        probe.scan_train0_pl_witness(
            proposal, ScanCollection(_identity_payload(proposal)), device=device,
        )


def test_cpu_backward_proves_outer_autocast_exact_ratios_gradient_and_no_update():
    device = torch.device("cpu")
    geometry = grpo._configure_exact_proposal_scoring(device)
    proposal = TinyProposal().eval()
    report = probe.execute_identity_backward(
        proposal,
        _identity_payload(proposal),
        {arm: (0, 2) for arm in range(1, recovery.GROUP_SIZE)},
        device=device,
    )

    assert geometry["autocast"] is False and geometry["dtype"] == "float32"
    assert report["scoring"] == {
        "outer_autocast": True,
        "outer_autocast_dtype": "bfloat16",
        "inner_autocast": False,
        "inner_batch_size": 1,
        "grpo_loss_dtype": "torch.float32",
        "switch_loss_dtype": "torch.float32",
    }
    assert report["ratios"]["passed"] is True
    assert report["ratios"]["ratio_atoms"] == 14
    assert report["ratios"]["all_ratio_atoms_exactly_one"] is True
    assert report["ratios"]["max_abs_logratio"] == 0.0
    assert report["ratios"]["clip_fraction"] == 0.0
    assert report["differentiability_witness"]["grpo_requires_grad"] is True
    assert report["differentiability_witness"]["switch_requires_grad"] is True
    assert report["backward"]["proposal_grad_norm"] > 0.0
    assert report["backward"]["warnings"] == []
    assert report["no_mutation"]["optimizer_steps"] == 0
    assert report["no_mutation"]["optimizer_state_entries_before"] == 0
    assert report["no_mutation"]["optimizer_state_entries_after"] == 0
    assert (
        report["no_mutation"]["proposal_digest_before"]
        == report["no_mutation"]["proposal_digest_after"]
    )


def test_ratio_gate_rejects_any_nonidentity_or_wrong_atom_count():
    exact = {
        "max_abs_logratio": 0.0,
        "ratio_min": 1.0,
        "ratio_mean": 1.0,
        "ratio_max": 1.0,
        "clip_fraction": 0.0,
        "ratio_atoms": 14.0,
        "ratio_sum": 14.0,
        "ratio_square_sum": 14.0,
        "ratio_ess_fraction": 1.0,
        "clipped_atoms": 0.0,
    }
    assert probe._require_ratio_identity(
        exact, device=torch.device("cpu"),
    )["all_ratio_atoms_exactly_one"] is True
    wrong_count = {**exact, "ratio_atoms": 7.0, "ratio_sum": 7.0,
                   "ratio_square_sum": 7.0}
    with pytest.raises(probe.ProbeError, match="fourteen ratio atoms"):
        probe._require_ratio_identity(wrong_count, device=torch.device("cpu"))
    nonidentity = {
        **exact,
        "max_abs_logratio": 1e-8,
    }
    with pytest.raises(grpo.OutcomeGRPOError, match="before optimizer step"):
        probe._require_ratio_identity(nonidentity, device=torch.device("cpu"))


def test_probe_rejects_any_nondeterministic_backward_warning():
    with pytest.raises(probe.ProbeError, match="nondeterministic-algorithm"):
        probe._require_no_nondeterminism_warning(
            [SimpleNamespace(message="Flash Attention is non-deterministic")],
            label="test backward",
        )


def test_probe_source_is_read_only_and_calls_exact_loss_under_autocast():
    source = Path(probe.__file__).read_text(encoding="utf-8")
    run_source = inspect.getsource(probe.run_probe)
    execute_source = inspect.getsource(probe.execute_identity_backward)
    scan_source = inspect.getsource(probe.scan_train0_pl_witness)
    bf16_source = inspect.getsource(probe.execute_strict_bf16_proposal_backward)

    assert "_load_authenticated_parent" in run_source
    assert "ValidatedRecoveryCollection.open" in run_source
    assert "deep=False" in run_source and "verify_sidecars=False" in run_source
    assert "scan_train0_pl_witness(" in run_source
    assert "enumerate(collection.receipts)" in scan_source
    assert "range(1, recovery.GROUP_SIZE)" in scan_source
    assert scan_source.count("collection.load(") == 1
    assert "_configure_exact_proposal_scoring" in run_source
    assert "_configure_strict_outcome_determinism" in run_source
    assert "execute_strict_bf16_proposal_backward(" in run_source
    assert "dtype=torch.bfloat16" in bf16_source
    assert "loss.backward()" in bf16_source
    assert "_require_no_nondeterminism_warning(" in bf16_source
    assert execute_source.index("with torch.autocast(") < execute_source.index(
        "grpo.sampled_group_losses("
    )
    assert "grpo.SWITCH_BALANCE_WEIGHT * switch_loss" in execute_source
    assert "total.backward()" in execute_source
    assert ".step(" not in source
    actions = {action.dest: action for action in probe.build_parser()._actions}
    assert set(actions) == {"help", "out"}
    assert actions["out"].required is True


def test_optional_launcher_requests_one_gpu_and_only_runs_the_probe():
    launcher = Path("scripts/outcome_ratio_identity_probe.sbatch").read_text(
        encoding="utf-8",
    )
    assert "#SBATCH --nodes=1" in launcher
    assert "#SBATCH --ntasks=1" in launcher
    assert "#SBATCH --gpus=1" in launcher
    assert "scripts/outcome_ratio_identity_probe.py" in launcher
    assert "optimizer_steps=0" in launcher
    assert "export CUBLAS_WORKSPACE_CONFIG=:4096:8" in launcher
    assert 'report["pl_rowwise_replay"]' in launcher
    assert 'geometry["transfer_chunk_replans"] == 32' in launcher
    assert 'geometry["observed_production_pl_batch_sizes"] == [1]' in launcher
    assert 'legacy["mismatch_count"] == len(rows) > 0' in launcher
    assert 'fixed["max_abs_coeff_error"] == 0.0' in launcher
    assert 'fixed["max_abs_old_logprob_error"] == 0.0' in launcher
    commands = [line.strip() for line in launcher.splitlines()
                if line.strip() and not line.lstrip().startswith("#")]
    assert not any(command.startswith("sbatch ") for command in commands)
