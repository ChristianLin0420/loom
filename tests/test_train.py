"""Team D — training loop, FSDP, schedules, checkpointing, preemption.

PLAN 4.D "done when": a 50-step stub run completes; FSDP works on 2 GPUs;
**survives an artificial SIGTERM and resumes with continuous loss**; the memory
profile fits at the configured batch size.

Everything here runs on CPU against ``stubs.*``. Anything needing a device is
behind ``@pytest.mark.gpu``; multi-GPU tests additionally skip on device count
(``pyproject.toml`` registers ``gpu``/``slow``/``bench`` and ``--strict-markers``
is on, so there is no ``multigpu`` marker to use -- see the note on
``test_fsdp_two_gpu``).

The tests that matter most are the ones nobody would notice failing:
``test_atomic_ckpt_survives_sigkill``, ``test_state_coverage_by_reflection`` and
``test_link_local_knobs_change_neither_identity_nor_schedule``. Silent corruption
lives there.
"""

from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import contracts as C
import stubs as S
from loom.train import atomic, ckpt as ckpt_mod, fsdp as fsdp_mod
from loom.train.determinism import (
    capture_rng_state, restore_rng_state, rng_fingerprint, set_step_seed,
)
from loom.train.loop import (
    LINK_LOCAL_KEYS, MODULE_NAMES, TrainState, build_model, config_hash,
    load_config, parse_args, read_config, WindowSampler,
)
from loom.train.preempt import (
    DEFAULT_SAFETY_S, PreemptGuard, decide_local, read_heartbeat, write_heartbeat,
)
from loom.train.schedule import (
    BANK_LR_MULT, CosineWithWarmup, EMATarget, FreezeSchedule, SpikeGuard,
    build_optimizer, clip_grad, module_grad_norms, param_groups,
)

CONFIGS = ROOT / "configs"
SLURM = ROOT / "loom" / "train" / "slurm"
STAGES = ("r0a", "r0b", "r1", "r2", "r3")


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def write_test_config(tmp_path: Path, **over) -> Path:
    """A stub-only config extending the real configs/base.yaml.

    ``use_stubs: true`` is not optional here: with ``auto`` this would build
    Team B's 144 M Perceiver and one CPU step would take minutes.
    """
    import yaml

    cfg = {
        "extends": str(CONFIGS / "base.yaml"),
        "run": {"name": "test", "steps": 6, "log_every": 1, "ckpt_every": 1000,
                "seed": 7, "project": "loom-test"},
        "model": {"use_stubs": True},
        "data": {"batch_per_gpu": 1, "embodiments": ["libero_franka"], "source": "stub"},
        "optim": {"lr": 1e-3, "warmup": 2},
    }
    for k, v in over.items():
        cfg[k] = {**cfg.get(k, {}), **v} if isinstance(v, dict) else v
    p = tmp_path / "test.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def _env() -> dict:
    return dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1",
                OMP_NUM_THREADS="8", WANDB_MODE="offline")


def link_cmd(cfg: Path, run_dir: Path, *extra: str) -> list[str]:
    return [sys.executable, "-m", "loom.train.loop", "--config", str(cfg),
            "--run_dir", str(run_dir), "--no_wandb", *extra]


def run_link(cfg: Path, run_dir: Path, *extra: str, timeout: int = 900):
    r = subprocess.run(link_cmd(cfg, run_dir, *extra), cwd=ROOT, env=_env(),
                       capture_output=True, text=True, timeout=timeout)
    assert r.returncode == 0, f"link failed\nSTDOUT\n{r.stdout}\nSTDERR\n{r.stderr}"
    return r


def curve(run_dir: Path) -> list[dict]:
    p = Path(run_dir) / "metrics.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def sigterm_mid_run(proc, run_dir: Path, kill_at: int, *, poll_s: float = 0.01,
                    timeout: float = 900.0) -> int:
    """Deliver SIGTERM while the loop is *provably* still running. Returns the
    step the child was frozen at.

    Polling the heartbeat and then signalling is a one-way handshake, and it is a
    race: on an A100 the child finishes the whole run in the time the parent
    takes to notice step ``kill_at``. The link then exits 0 having completed
    every step -- correct behaviour, but it tests nothing about preemption.
    Widening the band to accept it would be worse than failing, because a broken
    save-on-signal path would go green too.

    SIGSTOP closes the race: it cannot be caught, blocked or ignored, so the
    child freezes inside one scheduler slice and cannot advance another step. A
    SIGTERM sent to a stopped process stays pending and is delivered on SIGCONT.
    The signal therefore lands mid-run on any hardware, and the assertion after
    it keeps its original meaning: the signal interrupted a running loop.
    """
    end = time.time() + timeout
    while True:
        if proc.poll() is not None:
            raise AssertionError(
                f"the link exited (rc={proc.returncode}) before reaching step "
                f"{kill_at}; raise run.steps so there is a run left to interrupt"
            )
        hb = read_heartbeat(run_dir)
        if hb is not None and hb[1] >= kill_at:
            os.kill(proc.pid, signal.SIGSTOP)      # frozen: no further steps run
            break
        if time.time() > end:
            raise AssertionError(f"heartbeat never reached step {kill_at} in {timeout}s")
        time.sleep(poll_s)

    assert proc.poll() is None, "child exited between the heartbeat and the SIGSTOP"
    frozen_at = read_heartbeat(run_dir)[1]
    os.kill(proc.pid, signal.SIGTERM)              # pending while stopped
    os.kill(proc.pid, signal.SIGCONT)              # ... delivered here
    return frozen_at


def tiny_model() -> nn.Module:
    """A synthetic stand-in with the same child names the loop addresses."""
    m = nn.Module()
    m.estimator = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
    m.bank = nn.Sequential(nn.Linear(8, 8))
    m.q_delta = nn.Linear(8, 8)
    return m


def stub_train_state() -> TrainState:
    model = tiny_model()
    opt = build_optimizer(model, lr=1e-3, module_names=MODULE_NAMES)
    return TrainState(
        model=model, optimizer=opt,
        scheduler=CosineWithWarmup(1e-3, 2, 10),
        ema=EMATarget(model.estimator, tau=C.EMA_TAU),
        sampler=WindowSampler({"data": {"batch_per_gpu": 1}}, 0, 1, 0),
    )


# ═══════════════════════════════════════════════════════════════════════════
#  1 · A 50-STEP STUB RUN COMPLETES  (PLAN 4.D done-when)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
def test_50_step_stub_run_completes(tmp_path):
    """End to end on CPU with stubs: checkpoint, LATEST pointer, full curve."""
    cfg = write_test_config(tmp_path, run={"steps": 50, "log_every": 10,
                                           "ckpt_every": 25, "name": "fifty"})
    run_dir = tmp_path / "run"
    run_link(cfg, run_dir)

    assert ckpt_mod.latest_step(run_dir) == 50, "LATEST pointer not at the final step"
    payload = ckpt_mod.load_latest(run_dir)
    assert payload is not None and payload["global_step"] == 50
    assert (run_dir / ckpt_mod.shard_name(50, 0)).exists()

    rows = curve(run_dir)
    assert [r["global_step"] for r in rows] == list(range(1, 51))
    assert all(math.isfinite(r["loss"]) for r in rows)
    # Delta_op is logged on EVERY step, not every log_every: it is a build assert.
    assert all("delta_op" in r and math.isfinite(r["delta_op"]) for r in rows)
    # and it rides on the heartbeat line the watchdog reads
    hb = read_heartbeat(run_dir)
    assert hb is not None and hb[1] == 50


def test_stub_run_uses_all_four_r0_losses(tmp_path):
    """R0-A/R0-B = dyn + act + proposal + balance (PLAN 7)."""
    cfg = write_test_config(tmp_path, run={"steps": 2, "name": "losses"})
    run_dir = tmp_path / "run"
    run_link(cfg, run_dir)
    row = curve(run_dir)[-1]
    for term in ("dyn", "act", "proposal", "balance"):
        assert f"loss/{term}" in row, f"{term} is off in an R0 config"
    assert "loss/potential" not in row and "loss/grpo" not in row, "R3 terms leaked into R0"


def test_r1_config_disables_action_losses():
    """R1 is action-free: dyn + balance only, and no q_a / D_e / pi_c training."""
    cfg = read_config(CONFIGS / "r1.yaml")
    assert cfg["data"]["action_free"] is True
    assert cfg["losses"]["dyn"]["enabled"] and cfg["losses"]["balance"]["enabled"]
    assert not cfg["losses"]["act"]["enabled"]
    assert not cfg["losses"]["proposal"]["enabled"]
    assert set(cfg["train_modules"]) == {"estimator", "bank", "q_delta"}


def test_a_real_data_source_never_falls_back_to_stub_windows(monkeypatch):
    """`source: libero` + a stub fallback is a WASTED run, not a degraded one.

    R0-A on 16 GPUs would have trained for eight hours on torch.randn and
    produced a first score that reads like a modelling result. Only
    `source: stub` may fall back, and then it is an explicit choice.
    """
    from loom.train import loop as loop_mod

    def boom(name, *a, **kw):
        raise ImportError(f"no {name}")

    monkeypatch.setattr(loop_mod.importlib, "import_module", boom)

    stub_cfg = {"data": {"source": "stub", "batch_per_gpu": 1,
                         "embodiments": ["libero_franka"]}}
    assert isinstance(loop_mod.build_sampler(stub_cfg, 0, 1, 0, "cpu"), WindowSampler)

    for source in ("libero", "robotwin", "mixed"):
        with pytest.raises(RuntimeError, match=r"data.source is"):
            loop_mod.build_sampler({"data": {"source": source}}, 0, 1, 0, "cpu")


def test_missing_loader_factory_names_everything_it_tried(monkeypatch):
    from loom.train import loop as loop_mod

    empty = type("mod", (), {})()
    monkeypatch.setattr(loop_mod.importlib, "import_module", lambda *a, **k: empty)
    with pytest.raises(RuntimeError, match="build_loader") as exc:
        loop_mod.build_sampler({"data": {"source": "libero"}}, 0, 1, 0, "cpu")
    assert "absent" in str(exc.value)


def test_loader_factories_use_world_size_not_world():
    """Team A's constructors spell it `world_size`; only build_loader takes `world`."""
    from loom.train.loop import _LOADER_FACTORIES

    names = [n for n, _ in _LOADER_FACTORIES]
    assert names[0] == "build_loader", "the agreed factory must be tried first"
    for name, kw in _LOADER_FACTORIES:
        got = kw(0, 8, 1, "cuda")
        assert ("world" in got) == (name == "build_loader"), got


def test_batches_are_embodiment_homogeneous_and_dispatch(tmp_path):
    """PLAN 9: one embodiment per batch; q_a/D_e dispatched by window["embodiment"]."""
    cfg = {"data": {"batch_per_gpu": 2, "embodiments": ["libero_franka"]},
           "model": {"use_stubs": True}, "losses": {"dyn": {"enabled": True}}}
    sampler = WindowSampler(cfg, rank=0, world=1, seed=0)
    w = sampler.next(0)
    assert isinstance(w["embodiment"], str)
    assert w["actions"].shape[-1] == C.EMBODIMENTS[w["embodiment"]].dof

    model = build_model(cfg)
    assert w["embodiment"] in model.q_action and w["embodiment"] in model.decoder
    with pytest.raises(KeyError, match="embodiment-homogeneous"):
        model.compute_losses({**w, "embodiment": "no_such_body"}, 0, 0, 0)


# ═══════════════════════════════════════════════════════════════════════════
#  2 · FSDP ON 2 GPUs
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2,
                    reason="needs 2 CUDA devices; the login node has none")
def test_fsdp_two_gpu(tmp_path):
    """FSDP full-shard on E, bank+heads replicated, on 2 GPUs.

    Runs the same entry point the sbatch runs, with SLURM's rank plumbing faked.
    """
    cfg = write_test_config(tmp_path, run={"steps": 4, "name": "fsdp2"},
                            model={"use_stubs": True})
    run_dir = tmp_path / "run"
    procs = []
    for rank in range(2):
        env = _env() | {"RANK": str(rank), "WORLD_SIZE": "2", "LOCAL_RANK": str(rank),
                        "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29517"}
        procs.append(subprocess.Popen(link_cmd(cfg, run_dir), cwd=ROOT, env=env))
    for p in procs:
        assert p.wait(timeout=900) == 0
    # one shard per rank, and they must be distinct files
    assert (run_dir / ckpt_mod.shard_name(4, 0)).exists()
    assert (run_dir / ckpt_mod.shard_name(4, 1)).exists()


def test_fsdp_wrapping_is_a_noop_without_distributed():
    """The CPU path must be the identical code path, not a second implementation."""
    model = tiny_model()
    out, sync = fsdp_mod.wrap_for_training(model, {}, device="cpu", verbose=False)
    assert out is model
    assert isinstance(model.estimator, nn.Sequential)
    assert not sync.enabled
    sync.broadcast()                       # no-ops, never raises
    sync.all_reduce_grads()


def test_fsdp_config_shards_only_the_estimator():
    """PLAN 4.D: full-shard on E; bank and heads REPLICATED.

    Every rank needs the whole operator bank on every step, so sharding 25 M
    parameters buys nothing and adds an all-gather to the hot path.
    """
    cfg = fsdp_mod.FSDPConfig.from_dict(read_config(CONFIGS / "base.yaml")["fsdp"])
    assert "estimator" in cfg.shard
    assert "bank" not in cfg.shard and "bank" in cfg.replicate
    for head in ("q_delta", "q_action", "decoder", "proposal"):
        assert head in cfg.replicate
    assert cfg.activation_checkpointing


def test_frozen_tower_is_rejected():
    """PLAN 9: the frozen tower never enters the training graph."""
    class SiglipVisionTower(nn.Module):
        pass

    model = tiny_model()
    fsdp_mod.assert_frozen_tower_absent(model)          # clean
    model.estimator.tower = SiglipVisionTower()
    with pytest.raises(RuntimeError, match="frozen vision/text tower"):
        fsdp_mod.assert_frozen_tower_absent(model)


def test_cached_features_carry_no_autograd_history():
    w = S.make_window(b=1)
    fsdp_mod.assert_features_are_cached(w)              # clean
    w["feats"][0]["views"] = w["feats"][0]["views"].clone().requires_grad_(True)
    with pytest.raises(RuntimeError, match="frozen tower is in the training graph"):
        fsdp_mod.assert_features_are_cached(w)


def test_belief_is_pinned_to_the_compute_dtype():
    """The L_dyn rollout must not silently leave bf16.

    `E` is pre-LN, so its last op is a LayerNorm -- and layer_norm is in
    autocast's fp32 policy, so `z` comes back fp32 even with every matmul in bf16.
    `bank.step` then computes `a * x` with `a` bf16 and `x` fp32 and promotes the
    whole affine rollout. Found by a real R0-A smoke on one A100.
    """
    model = build_model({"data": {"embodiments": ["libero_franka"]},
                         "model": {"use_stubs": True},
                         "losses": {"dyn": {"enabled": True}}})
    w = S.make_window(b=1)
    assert model.compute_dtype is None
    assert model.beliefs(w)[0].dtype is torch.float32       # CPU path untouched

    model.compute_dtype = torch.bfloat16
    for z in model.beliefs(w):
        assert z.dtype is torch.bfloat16, "belief left the compute dtype"
    for z in model.target_beliefs(w):
        assert z.dtype is torch.bfloat16, "EMA target belief left the compute dtype"


def test_to_device_casts_only_float_tensors():
    from loom.train.loop import _to_device

    w = S.make_window(b=1)
    w["step_id"] = torch.tensor([3, 4])
    out = _to_device(w, "cpu", torch.bfloat16)
    assert out["feats"][0]["views"].dtype is torch.bfloat16
    assert out["lang"].dtype is torch.bfloat16
    assert out["actions"].dtype is torch.bfloat16
    assert out["step_id"].dtype is torch.int64, "an index tensor was cast to bf16"


def test_assert_bf16_is_a_real_build_assert():
    """The bank returns its PARAMETER dtype, not c's, so bf16 must be asserted."""
    fsdp_mod.assert_bf16(torch.zeros(2, dtype=torch.bfloat16), "ok")
    with pytest.raises(RuntimeError, match="not bfloat16"):
        fsdp_mod.assert_bf16(torch.zeros(2), "z")
    with pytest.raises(RuntimeError, match="complex"):
        fsdp_mod.assert_bf16(torch.zeros(2, dtype=torch.complex64), "z")


def test_replicated_modules_are_synced_by_hand_not_by_fsdp():
    """Bank/heads must NOT be FSDP- or DDP-wrapped.

    ``contracts.Bank`` is used as ``bank.step(...)``, ``Proposal`` as
    ``proposal.log_prob(...)``, ``Decoder`` as ``decoder.loss(...)`` -- none of
    which is ``forward``, and both DDP and FSDP install their gradient-sync hooks
    in the forward pre-hook. Wrapping them therefore trains with unsynchronised
    gradients on every rank while the loss curve looks perfect. Measured on 2
    A100s: an FSDP-wrapped head also stopped being subscriptable, because FSDP
    forwards __getitem__ but not __contains__.
    """
    model = tiny_model()
    _, sync = fsdp_mod.wrap_for_training(
        model, {"shard": ["estimator"], "replicate": ["bank", "q_delta"]},
        device="cpu", verbose=False)
    assert set(sync.replicated) == {"bank", "q_delta"}
    assert set(sync.sharded) == {"estimator"}

    ids_r = {id(p) for p in sync.replicated_params()}
    ids_s = {id(p) for p in sync.sharded_params()}
    assert not (ids_r & ids_s), "a parameter is both sharded and replicated"
    assert ids_r | ids_s == {id(p) for p in model.parameters()}


_GLOO_SYNC = r'''
import os, sys, torch, torch.distributed as dist
sys.path.insert(0, sys.argv[1])
import torch.nn as nn
from loom.train.fsdp import ReplicaSync
from loom.train.schedule import clip_grad

rank = int(os.environ["RANK"])
dist.init_process_group("gloo")
torch.manual_seed(rank)                       # deliberately DIFFERENT per rank
model = nn.Module()
model.estimator = nn.Linear(4, 4)
model.bank = nn.Linear(4, 4)
sync = ReplicaSync({"bank": model.bank}, {"estimator": model.estimator})
assert sync.enabled

sync.broadcast()
gathered = [None, None]
dist.all_gather_object(gathered, [p.detach().tolist() for p in model.bank.parameters()])
assert gathered[0] == gathered[1], "broadcast did not equalise the replicated weights"

for p in model.bank.parameters():
    p.grad = torch.full_like(p, float(rank + 1))      # rank0 -> 1.0, rank1 -> 2.0
for p in model.estimator.parameters():
    p.grad = torch.zeros_like(p)
sync.all_reduce_grads()
for p in model.bank.parameters():
    assert torch.allclose(p.grad, torch.full_like(p, 1.5)), p.grad
n = clip_grad(model, 1e9, sync=sync)
gathered = [None, None]
dist.all_gather_object(gathered, n)
assert abs(gathered[0] - gathered[1]) < 1e-9, f"ranks disagree on the grad norm: {gathered}"
print("GLOO_SYNC_OK", rank, flush=True)
dist.destroy_process_group()
'''


def test_replica_sync_broadcasts_and_averages_over_gloo(tmp_path):
    """The distributed half of ReplicaSync, exercised for real -- on CPU.

    Two ranks initialise differently on purpose (per-rank seeding), so if
    broadcast() were missing the replicas would silently diverge from step 0.
    """
    script = tmp_path / "gloo_sync.py"
    script.write_text(_GLOO_SYNC)
    procs = []
    for rank in range(2):
        env = _env() | {"RANK": str(rank), "WORLD_SIZE": "2", "LOCAL_RANK": str(rank),
                        "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29613"}
        procs.append(subprocess.Popen([sys.executable, str(script), str(ROOT)],
                                      cwd=ROOT, env=env, text=True,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT))
    outs = []
    for p in procs:
        out, _ = p.communicate(timeout=300)
        outs.append(out)
    for p, out in zip(procs, outs):
        assert p.returncode == 0, out
        assert "GLOO_SYNC_OK" in out


def test_clip_grad_without_a_sync_is_the_ordinary_global_norm():
    model = tiny_model()
    for p in model.parameters():
        p.grad = torch.full_like(p, 3.0)
    expected = torch.cat([p.grad.flatten() for p in model.parameters()]).norm()
    assert clip_grad(model, 1.0, sync=None) == pytest.approx(float(expected), rel=1e-5)


def test_no_fp8_path_anywhere():
    """A100 has no FP8 (PLAN 9). bf16 is the only reduced precision in this tree."""
    for p in sorted((ROOT / "loom" / "train").rglob("*.py")):
        text = p.read_text().lower()
        assert "float8" not in text and "fp8" not in text.replace("no fp8", ""), \
            f"{p} mentions fp8"


# ═══════════════════════════════════════════════════════════════════════════
#  3 · SIGTERM AND RESUME WITH A CONTINUOUS LOSS  (the PLAN 4.D headline)
# ═══════════════════════════════════════════════════════════════════════════

def _assert_continuous(baseline: list[dict], resumed: list[dict], *, rel: float,
                       n: int, boundary: int) -> dict:
    """Continuity, not bit-identity (PLAN 4.D). Returns the numbers for the log."""
    assert [r["global_step"] for r in resumed] == list(range(1, n + 1)), \
        f"resumed step sequence has a gap or a repeat: " \
        f"{[r['global_step'] for r in resumed]}"
    worst, worst_pair = 0.0, (0.0, 0.0)
    for a, b in zip(baseline, resumed):
        assert a["global_step"] == b["global_step"]
        d = abs(a["loss"] - b["loss"]) / max(1.0, abs(a["loss"]))
        if d > worst:
            worst, worst_pair = d, (a["loss"], b["loss"])
    assert worst <= rel, (
        f"loss curve discontinuous across resume: worst relative gap {worst:.3e} "
        f"(baseline {worst_pair[0]:.6f} vs resumed {worst_pair[1]:.6f})"
    )
    return {
        "worst_rel_gap": worst,
        # the step that ran last before the stop, and the first one after resume
        "pre_kill": resumed[boundary - 1]["loss"],
        "post_resume": resumed[boundary]["loss"] if boundary < len(resumed) else float("nan"),
        "baseline_at_post": baseline[boundary]["loss"] if boundary < len(baseline) else float("nan"),
    }


@pytest.mark.slow
def test_survives_sigterm_and_resumes_with_continuous_loss(tmp_path):
    """Kill a link with SIGTERM mid-run, restart it, and require continuity.

    NOT bit-identity: PLAN 4.D explicitly does not require it after a restart,
    and on 64 GPUs the reduction order alone breaks equality. What must hold is
    that ``global_step`` picks up exactly where it stopped and the loss lands on
    the pre-kill trend rather than jumping back to its initialisation.
    """
    # kill_at is deliberately early and n comfortably beyond it: the SIGSTOP
    # barrier in sigterm_mid_run makes the *delivery* device-independent, and the
    # remaining steps make it obvious in the failure message if it ever is not.
    n, kill_at = 8, 2
    cfg = write_test_config(tmp_path, run={"steps": n, "name": "sigterm"})

    base_dir = tmp_path / "baseline"
    run_link(cfg, base_dir)
    baseline = curve(base_dir)
    assert len(baseline) == n

    kill_dir = tmp_path / "killed"
    kill_dir.mkdir()
    proc = subprocess.Popen(link_cmd(cfg, kill_dir), cwd=ROOT, env=_env(),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        frozen_at = sigterm_mid_run(proc, kill_dir, kill_at)
        rc = proc.wait(timeout=300)
        out = proc.stdout.read()
    finally:
        if proc.poll() is None:
            proc.kill()                          # SIGKILL also reaps a stopped child
    assert rc == 0, f"SIGTERM did not produce a graceful exit:\n{out}"

    stopped_at = ckpt_mod.latest_step(kill_dir)
    assert stopped_at is not None and kill_at <= stopped_at < n, \
        f"SIGTERM link saved at {stopped_at}, expected {kill_at}..{n - 1}\n{out}"
    # The step in flight when the signal landed is allowed to finish, and nothing
    # beyond it may run: should_stop() is checked at the end of every step.
    assert frozen_at <= stopped_at <= frozen_at + 1, \
        f"frozen at {frozen_at} but the link ran on to {stopped_at}\n{out}"
    assert len(curve(kill_dir)) == stopped_at

    # It must have stopped for the RIGHT reason. A link that hit its wall-clock
    # budget at the right step would otherwise pass this test while the signal
    # path was dead.
    reason = ckpt_mod.load_latest(kill_dir).get("stop_reason", "")
    assert reason.startswith("signal"), \
        f"link stopped for {reason!r}, not the signal; SIGUSR1/SIGTERM handling is dead"
    assert f"({reason})" in out

    run_link(cfg, kill_dir)                      # fresh process, resumes from LATEST
    resumed = curve(kill_dir)
    assert ckpt_mod.latest_step(kill_dir) == n
    got = _assert_continuous(baseline, resumed, rel=5e-2, n=n, boundary=stopped_at)
    print(f"\nSIGTERM at step {stopped_at}/{n}: pre-kill loss {got['pre_kill']:.6f} -> "
          f"post-resume {got['post_resume']:.6f} "
          f"(uninterrupted at that step: {got['baseline_at_post']:.6f}; "
          f"worst relative gap over the whole curve {got['worst_rel_gap']:.2e})")


def test_sentinel_stop_is_recorded_as_a_different_reason(tmp_path):
    """`stop_reason` must discriminate, or asserting it above proves nothing.

    Also the operational contract: `touch runs/<name>/STOP` ends a run at the
    next step boundary with a durable checkpoint. Never scancel.
    """
    cfg = write_test_config(tmp_path, run={"steps": 8, "name": "sentinel"})
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "STOP").touch()                   # already there before step 1
    run_link(cfg, run_dir)

    stopped_at = ckpt_mod.latest_step(run_dir)
    assert stopped_at == 1, f"the sentinel did not stop the link promptly ({stopped_at})"
    assert ckpt_mod.load_latest(run_dir)["stop_reason"] == "sentinel"


@pytest.mark.slow
def test_single_process_cpu_resume_is_tight(tmp_path):
    """One process, one device: assert near-equality, not just continuity.

    Every step is reseeded from (seed, global_step, rank), so a resumed step is a
    pure function of the restored parameters. If this drifts, the checkpoint is
    missing a slot -- which is invisible in the loose test above.
    """
    n, stop = 6, 3
    cfg = write_test_config(tmp_path, run={"steps": n, "name": "tight"})

    base_dir = tmp_path / "baseline"
    run_link(cfg, base_dir)
    baseline = curve(base_dir)

    split_dir = tmp_path / "split"
    run_link(cfg, split_dir, "--stop_at", str(stop))
    assert ckpt_mod.latest_step(split_dir) == stop
    run_link(cfg, split_dir)
    resumed = curve(split_dir)

    got = _assert_continuous(baseline, resumed, rel=1e-6, n=n, boundary=stop)
    print(f"\nstop_at {stop}/{n}: pre-stop loss {got['pre_kill']:.6f} -> "
          f"post-resume {got['post_resume']:.6f} "
          f"(uninterrupted at that step: {got['baseline_at_post']:.6f}; "
          f"worst relative gap {got['worst_rel_gap']:.2e})")
    a, b = ckpt_mod.load_latest(base_dir), ckpt_mod.load_latest(split_dir)
    assert a["global_step"] == b["global_step"] == n
    assert a["samples_seen"] == b["samples_seen"]
    fa = torch.cat([v.flatten().float() for v in a["model"].values()])
    fb = torch.cat([v.flatten().float() for v in b["model"].values()])
    assert (fa - fb).abs().max().item() < 1e-5, "parameters diverged across resume"


def test_ema_target_survives_the_round_trip():
    """L_dyn's target must not silently reset to initialisation on every link."""
    src = tiny_model().estimator
    ema = EMATarget(src, tau=0.5)
    for _ in range(3):
        with torch.no_grad():
            for p in src.parameters():
                p.add_(torch.randn_like(p))
        ema.update(src)
    before = [p.clone() for p in ema.module.parameters()]

    restored = EMATarget(tiny_model().estimator, tau=0.9)
    restored.load_state_dict(ema.state_dict())
    assert restored.tau == pytest.approx(0.5)
    for a, b in zip(before, restored.module.parameters()):
        assert torch.equal(a, b), "EMA restarted across resume"


def test_ema_update_that_matches_nothing_raises():
    """A silent no-match would freeze the target forever while L_dyn looked fine."""
    ema = EMATarget(nn.Linear(4, 4))
    with pytest.raises(RuntimeError, match="matched no parameters"):
        ema.update(nn.Conv2d(1, 1, 1))


def test_rng_state_round_trips_and_survives_a_cuda_map_location():
    st = capture_rng_state()
    a = torch.randn(8)
    restore_rng_state(st)
    assert torch.allclose(a, torch.randn(8))
    # the GPU-resume defect: map_location="cuda" returns the RNG ByteTensor as a
    # non-uint8 tensor and set_rng_state raises. Restore must coerce.
    moved = dict(st, torch=st["torch"].to(torch.int64))
    restore_rng_state(moved)


def test_step_seeding_makes_a_step_a_pure_function():
    set_step_seed(3, 17, 0)
    a = torch.randn(4)
    set_step_seed(3, 17, 0)
    assert torch.equal(a, torch.randn(4))
    set_step_seed(3, 17, 1)
    assert not torch.equal(a, torch.randn(4)), "two ranks share an RNG stream"


def test_ranks_get_distinct_streams_and_shards():
    fps = [rng_fingerprint(seed=0, step=0, rank=r) for r in range(8)]
    assert len(set(fps)) == 8, "ranks share an RNG stream"
    shards = [ckpt_mod.shard_name(100, r) for r in range(8)]
    assert len(set(shards)) == 8, "two ranks target the same checkpoint file"
    assert rng_fingerprint(0, 0, 3) == rng_fingerprint(0, 0, 3)


# ═══════════════════════════════════════════════════════════════════════════
#  4 · MEMORY PROFILE FITS AT THE CONFIGURED BATCH SIZE
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("stage", STAGES)
def test_memory_profile_fits_at_the_configured_batch(stage):
    cfg = read_config(CONFIGS / f"{stage}.yaml")
    world = int(cfg["slurm"]["nodes"]) * int(cfg["slurm"]["gpus_per_node"])
    n_emb = len(cfg["data"]["embodiments"])
    spec = C.EMBODIMENTS.get(cfg["data"]["embodiments"][0])
    est = fsdp_mod.memory_estimate(
        world_size=world,
        batch_per_gpu=int(cfg["data"]["batch_per_gpu"]),
        # PLAN 2 budget: 150 M E, 105 M other shared, 50 M per embodiment.
        replicated_params=105e6 + 50e6 * n_emb,
        n_views=spec.n_views if spec else 2,
        activation_checkpointing=bool(cfg["fsdp"]["activation_checkpointing"]),
    )
    assert fsdp_mod.fits_on_a100_80gb(est), (
        f"{stage}: estimated {est['total_gib']:.1f} GiB/GPU at batch "
        f"{cfg['data']['batch_per_gpu']} on {world} GPUs; components {est}"
    )


def test_activation_checkpointing_is_what_makes_it_fit():
    """If AC were off the estimate must blow up -- otherwise the test is vacuous."""
    on = fsdp_mod.memory_estimate(world_size=16, batch_per_gpu=8,
                                  activation_checkpointing=True)
    off = fsdp_mod.memory_estimate(world_size=16, batch_per_gpu=8,
                                   activation_checkpointing=False)
    assert off["activations_gib"] > 3 * on["activations_gib"]


def test_sharding_the_estimator_actually_saves_memory():
    small = fsdp_mod.memory_estimate(world_size=64, batch_per_gpu=8)
    large = fsdp_mod.memory_estimate(world_size=1, batch_per_gpu=8)
    assert small["sharded_gib"] < large["sharded_gib"] / 8


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_measured_memory_matches_the_estimate(tmp_path):
    """The real measurement. The analytic estimate above is the CPU-side proxy."""
    cfg_path = write_test_config(tmp_path, run={"steps": 2, "name": "mem"})
    run_dir = tmp_path / "run"
    torch.cuda.reset_peak_memory_stats()
    run_link(cfg_path, run_dir)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    assert peak < 80.0 * 0.85, f"peak {peak:.1f} GiB on an 80 GiB card"


# ═══════════════════════════════════════════════════════════════════════════
#  5 & 6 · ATOMIC CHECKPOINTS
# ═══════════════════════════════════════════════════════════════════════════

_KILL_SCRIPT = r'''
import os, sys, time, torch
sys.path.insert(0, sys.argv[1])
from loom.train import atomic, ckpt
run = sys.argv[2]

# a durable earlier checkpoint
torch.save({"format": 1, "global_step": 10, "model": {"w": torch.ones(4)}},
           os.path.join(run, ckpt.shard_name(10, 0)))
atomic.write_pointer(run, 10)

# now a save that will never finish: write a partial .tmp, then block forever.
real_save = torch.save
def hang(obj, f, *a, **kw):
    with open(f, "wb") as fh:
        fh.write(b"\x00" * (1 << 20))
        fh.flush()
    open(os.path.join(run, "SAVING"), "w").close()
    time.sleep(600)
torch.save = hang
ckpt.save({"format": 1, "global_step": 20, "model": {}}, run, 20)
'''


def test_atomic_ckpt_survives_sigkill(tmp_path):
    """SIGKILL mid-save must leave a loadable EARLIER checkpoint.

    R1/R2 run 3-6 days across dozens of preemptible links. A half-written payload
    that the next link happily loads is a silent month-long corruption, so the
    payload is written to ``.tmp`` and only ``os.replace``d once durable, and
    LATEST is advanced only after that.
    """
    script = tmp_path / "kill.py"
    script.write_text(_KILL_SCRIPT)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    proc = subprocess.Popen([sys.executable, str(script), str(ROOT), str(run_dir)],
                            env=_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        end = time.time() + 300
        while not (run_dir / "SAVING").exists() and time.time() < end:
            assert proc.poll() is None, proc.stdout.read().decode()
            time.sleep(0.1)
        assert (run_dir / "SAVING").exists(), "child never reached the save"
        proc.kill()
        proc.wait(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()

    assert not (run_dir / ckpt_mod.shard_name(20, 0)).exists(), \
        "a half-written payload was published under its real name"
    assert (run_dir / (ckpt_mod.shard_name(20, 0) + ".tmp")).exists()
    assert ckpt_mod.latest_step(run_dir) == 10, "LATEST advanced past a partial payload"
    payload = ckpt_mod.load_latest(run_dir)
    assert payload is not None and payload["global_step"] == 10


def test_pointer_written_after_payload(tmp_path):
    atomic.atomic_write_text(tmp_path / "payload", "good")
    atomic.write_pointer(tmp_path, 5)
    assert atomic.read_pointer(tmp_path) == 5

    # an interrupted write leaves a .tmp and never replaces the good payload
    (tmp_path / "payload.tmp").write_text("half")
    assert (tmp_path / "payload").read_text() == "good"

    # a corrupt pointer degrades to None; it must never raise and take out a link
    (tmp_path / "LATEST").write_text("garbage")
    assert atomic.read_pointer(tmp_path) is None


def test_prune_keeps_the_last_n_and_the_permanent_ones(tmp_path):
    for step in (1000, 10000, 20000, 30000, 40000):
        (tmp_path / ckpt_mod.shard_name(step, 0)).write_bytes(b"x")
    ckpt_mod._prune(tmp_path, keep_last=2, permanent_every=10000)
    left = sorted(int(p.name.split("_")[1]) for p in tmp_path.glob("ckpt_*"))
    assert left == [10000, 20000, 30000, 40000]
    assert 1000 not in left


def test_resuming_onto_a_different_world_size_refuses_a_peer_shard(tmp_path):
    """FSDP shards are per-rank. Loading a peer's shard corrupts the estimator."""
    (tmp_path / ckpt_mod.shard_name(7, 0)).write_bytes(b"x")
    atomic.write_pointer(tmp_path, 7)
    os.environ["RANK"] = "3"
    try:
        with pytest.raises(RuntimeError, match="world size changed"):
            ckpt_mod.load_latest(tmp_path)
    finally:
        os.environ.pop("RANK", None)


# ═══════════════════════════════════════════════════════════════════════════
#  7 · STATE COVERAGE BY REFLECTION
# ═══════════════════════════════════════════════════════════════════════════

class _Stateful:
    """Stands in for the next object someone forgets to checkpoint."""

    def __init__(self):
        self.acc = torch.zeros(3)

    def update(self, x):
        self.acc += x

    def state_dict(self):
        return {"acc": self.acc}

    def load_state_dict(self, sd):
        self.acc = sd["acc"]


def _stateful_fields(state: TrainState) -> set[str]:
    return {n for n, o in vars(state).items()
            if n not in TrainState.NOT_STATE and hasattr(o, "state_dict")}


def test_state_coverage_by_reflection():
    """Every object in the training state with a state_dict() must be checkpointed.

    ``ckpt.build_state`` enumerates its slots explicitly; this walks the live
    ``TrainState`` and fails on anything it missed. Reflection-based on purpose:
    it is meant to fail on the *next* stateful object without anyone remembering
    to add a case, which is how the resume path silently goes wrong.
    """
    state = stub_train_state()
    payload = ckpt_mod.build_state(state, config_hash="x")

    missing = _stateful_fields(state) - set(payload)
    assert not missing, f"training state not in the checkpoint: {sorted(missing)}"
    for slot in ckpt_mod.STATE_SLOTS:
        assert slot in payload
    for scalar in ("global_step", "samples_seen", "rng", "config_hash", "git_sha",
                   "world_size", "wandb_run_id"):
        assert scalar in payload

    # prove the test can fail: plant an object build_state does not know about
    state.running_stats = _Stateful()
    assert _stateful_fields(state) - set(payload) == {"running_stats"}


def test_checkpoint_round_trips_the_whole_state():
    state = stub_train_state()
    for _ in range(3):
        loss = state.model.estimator(torch.randn(2, 8)).sum()
        state.optimizer.zero_grad()
        loss.backward()
        state.optimizer.step()
        state.ema.update(state.model.estimator)
    state.global_step, state.samples_seen = 3, 24
    state.sampler.cursor = 24
    payload = ckpt_mod.build_state(state, config_hash="abc", world_size=1)

    fresh = stub_train_state()
    ckpt_mod.restore(payload, fresh)
    assert fresh.global_step == 3 and fresh.samples_seen == 24
    assert fresh.sampler.cursor == 24
    for a, b in zip(state.model.parameters(), fresh.model.parameters()):
        assert torch.equal(a, b)
    for a, b in zip(state.ema.module.parameters(), fresh.ema.module.parameters()):
        assert torch.equal(a, b)
    assert (state.optimizer.state_dict()["state"].keys()
            == fresh.optimizer.state_dict()["state"].keys())


def test_every_module_buffer_reaches_the_checkpoint():
    """register_buffer(persistent=False) silently drops out of a state_dict."""
    model = build_model({"data": {"embodiments": ["libero_franka"]},
                         "model": {"use_stubs": True}, "losses": {}})
    sd = set(model.state_dict())
    missing = [n for n, _ in model.named_buffers() if n not in sd]
    assert not missing, f"non-persistent buffers absent from the checkpoint: {missing}"


# ═══════════════════════════════════════════════════════════════════════════
#  8 · CONFIG IDENTITY — LINK-LOCAL vs SCHEDULE
# ═══════════════════════════════════════════════════════════════════════════

LINK_LOCAL = [["--stop_at", "3"], ["--budget_s", "120"], ["--safety_s", "30"],
              ["--run_dir", "/tmp/loom-identity"], ["--no_wandb"],
              ["--allow_reshard"]]
SCHEDULE = [["--steps", "2000"], ["--seed", "1"], ["--lr", "1e-4"],
            ["--batch", "4"], ["--set", "optim.warmup=100"],
            ["--set", "losses.act.enabled=false"]]


def _cfg(tmp_path, extra):
    return load_config(parse_args(["--config", str(write_test_config(tmp_path))] + extra))


def _lr_curve(cfg, n=64):
    s = CosineWithWarmup(float(cfg["optim"]["lr"]), int(cfg["optim"]["warmup"]),
                         int(cfg["run"]["steps"]))
    return [s.lr_at(i) for i in range(0, 1000)][:n]


@pytest.mark.parametrize("extra", LINK_LOCAL, ids=lambda e: e[0])
def test_link_local_knobs_change_neither_identity_nor_schedule(tmp_path, extra):
    base, variant = _cfg(tmp_path, []), _cfg(tmp_path, extra)
    assert config_hash(base) == config_hash(variant), (
        f"{extra[0]} leaked into the run hash: every requeue would look like a "
        f"different experiment"
    )
    assert _lr_curve(base) == _lr_curve(variant), f"{extra[0]} changed the LR curve"


@pytest.mark.parametrize("extra", SCHEDULE, ids=lambda e: "".join(e))
def test_schedule_knobs_change_identity(tmp_path, extra):
    base, variant = _cfg(tmp_path, []), _cfg(tmp_path, extra)
    assert config_hash(base) != config_hash(variant), \
        f"{extra} defines the experiment but is excluded from the run hash"


def test_stop_at_truncates_the_link_and_never_moves_the_schedule(tmp_path):
    """The bug this rule comes from: a short link must follow the long schedule."""
    full = _cfg(tmp_path, ["--steps", "1000"])
    short = _cfg(tmp_path, ["--steps", "1000", "--stop_at", "200"])
    assert full["run"]["steps"] == short["run"]["steps"] == 1000
    assert short["link"]["stop_at"] == 200
    a = CosineWithWarmup(1e-3, 50, full["run"]["steps"])
    b = CosineWithWarmup(1e-3, 50, short["run"]["steps"])
    assert [a.lr_at(i) for i in range(200)] == [b.lr_at(i) for i in range(200)]


def test_link_section_is_excluded_from_the_hash_by_construction(tmp_path):
    cfg = _cfg(tmp_path, [])
    assert set(cfg["link"]) == set(LINK_LOCAL_KEYS)
    assert config_hash(cfg) == config_hash({k: v for k, v in cfg.items() if k != "link"})


@pytest.mark.parametrize("stage", STAGES)
def test_every_run_config_loads_and_is_self_consistent(stage):
    cfg = read_config(CONFIGS / f"{stage}.yaml")
    assert cfg["run"]["name"] == stage
    assert cfg["optim"]["warmup"] == 2000, "PLAN 4.D fixes warmup at 2k"
    assert cfg["optim"]["grad_clip"] == 1.0, "PLAN 4.D fixes grad clip at 1.0"
    assert cfg["optim"]["bank_lr_mult"] == 0.1, "PLAN 4.D: bank at 10x lower LR"
    assert cfg["optim"]["ema_tau"] == C.EMA_TAU
    assert cfg["losses"]["balance"]["weight"] == C.BALANCE_COEF
    assert cfg["data"]["embodiments"], "no embodiment: no head dispatch"
    assert cfg["run"]["steps"] > cfg["optim"]["warmup"]
    # every enabled loss must have a module that can produce it
    if cfg["losses"]["act"]["enabled"]:
        assert not cfg["data"]["action_free"], "act loss with no actions"
        assert {"q_action", "decoder"} <= set(cfg["train_modules"])
    if cfg["losses"]["proposal"]["enabled"]:
        assert "proposal" in cfg["train_modules"]
    if cfg["losses"]["potential"]["enabled"] or cfg["losses"]["grpo"]["enabled"]:
        assert "potential" in cfg["train_modules"]


# ═══════════════════════════════════════════════════════════════════════════
#  8b · THE R0-A RERUN: proprio-only decoder, real hinge, Switch balance
#
#  Three changes the project owner authorised, and the metrics that read them.
#  Each one has a measured failure behind it; the comments name it, because the
#  numbers are the only reason any of this is here.
# ═══════════════════════════════════════════════════════════════════════════

def _stub_cfg(**over) -> dict:
    cfg = {"data": {"batch_per_gpu": 4, "embodiments": ["libero_franka"]},
           "model": {"use_stubs": True},
           "losses": {"dyn": {"enabled": True, "negatives": "within_trajectory"},
                      "act": {"enabled": True},
                      "balance": {"enabled": True, "weight": C.BALANCE_COEF}}}
    for k, v in over.items():
        cfg[k] = {**cfg.get(k, {}), **v} if isinstance(v, dict) else v
    return cfg


def test_decoder_is_given_proprio_and_never_the_belief():
    """CHANGE 1. `D_e(proprio_t, c)`, so `c` is the only channel carrying task
    information into the action.

    With the belief in there, `L_act` is behaviour cloning: R0-A measured
    `act/decode` falling 0.2489 -> 0.0559 while `c_a` held 2-3 distinct top-4
    supports over 64 real training windows, i.e. the term put no pressure on
    the coefficient whatsoever.
    """
    cfg = _stub_cfg()
    model = build_model(cfg)
    sampler = WindowSampler(cfg, rank=0, world=1, seed=0)
    w = sampler.next(0)

    seen = []
    dec = model.decoder[w["embodiment"]]
    real_loss = dec.loss

    def spy(first, c, a_seg, *a, **kw):
        seen.append(tuple(first.shape))
        return real_loss(first, c, a_seg, *a, **kw)

    dec.loss = spy
    model.compute_losses(w, 0, 0, 0)

    dof = C.EMBODIMENTS[w["embodiment"]].dof
    assert seen == [(4, dof)] * C.DEPTH, seen
    assert all(sh != (4, C.K, C.D) for sh in seen), "the belief reached D_e"


def test_decoder_gets_the_proprio_at_the_start_of_its_own_segment():
    """`window["feats"][h]["proprio"]`, not feats[0] and not feats[h+1].

    Segment `h` covers canonical frames 8h..8h+7, so the conditioning state is
    the one at frame 8h. Off by one here trains fine and scores near zero.
    """
    cfg = _stub_cfg()
    model = build_model(cfg)
    w = WindowSampler(cfg, rank=0, world=1, seed=0).next(0)
    for h in range(C.N_STATES):                     # make each state identifiable
        w["feats"][h]["proprio"] = torch.full_like(w["feats"][h]["proprio"], float(h))

    seen = []
    dec = model.decoder[w["embodiment"]]
    real_loss = dec.loss
    dec.loss = lambda p, c, a, *x, **k: (seen.append(float(p[0, 0])),
                                         real_loss(p, c, a, *x, **k))[1]
    model.compute_losses(w, 0, 0, 0)
    assert seen == [0.0, 1.0, 2.0, 3.0], seen


def test_l_dyn_runs_the_configured_within_trajectory_hinge():
    """CHANGE 2. The loop used to compute a bare `1 - cos(A(c)z, z+)` with NO
    negatives, so `losses.dyn.negatives` was inert in every config that set it.

    `dyn/neg` is the hinge term. It must be reported, and it must be exactly
    zero when negatives are off -- otherwise the switch is not doing anything
    either way and the metric cannot be read.
    """
    on = build_model(_stub_cfg())
    w = WindowSampler(_stub_cfg(), rank=0, world=1, seed=0).next(0)
    m_on = on.compute_losses(w, 0, 0, 0)[1]
    assert "dyn/neg" in m_on
    assert m_on["dyn/neg"] > 0.0, "the hinge is wired but never fires"

    off_cfg = _stub_cfg(losses={"dyn": {"enabled": True, "negatives": "none"},
                                "act": {"enabled": True},
                                "balance": {"enabled": True}})
    off = build_model(off_cfg)
    m_off = off.compute_losses(w, 0, 0, 0)[1]
    assert m_off["dyn/neg"] == 0.0


def test_l_dyn_negatives_draw_on_a_cpu_generator_against_device_tensors():
    """The trap a previous attempt at this hit.

    `torch_generator` returns a CPU generator -- that is what makes a step a
    pure function of `(seed, global_step, rank)` on any device -- and
    `torch.multinomial(<cuda tensor>, generator=<cpu gen>)` is a hard error.
    The choice therefore has to be made where the generator lives and the
    indices moved. Checked on CPU by handing the sampler a generator and a
    coefficient tensor and requiring the result to land on the tensor's device.
    """
    from loom.losses.dyn import sample_within_trajectory_negatives
    from loom.train.determinism import torch_generator

    c_seq = S.sparse_simplex(3, C.DEPTH)
    g = torch_generator(0, 0, 0, tag="dyn")
    assert g.device.type == "cpu", "the generator contract changed"
    neg = sample_within_trajectory_negatives(c_seq, 2, g)
    assert neg.shape == c_seq.shape and neg.device == c_seq.device
    # min_gap=2 with DEPTH=4: 0->{2,3}, 1->{3}, 2->{0}, 3->{0,1}. Segment 1 has
    # exactly one legal partner, so that row is checkable outright.
    assert torch.equal(neg[:, 1], c_seq[:, 3])


def test_switch_balance_floor_ceiling_and_direction():
    """CHANGE 3. `M * sum_m f_m P_m`, with `f` the routing-slot fraction and `P`
    the mean DENSE router probability.

    Degenerate floor is exactly 1.0 (both uniform) -- know it before reading a
    flat curve as progress. Ceiling is `M / TOPK = 32`, every token on the same
    four operators with a router that is certain about it.
    """
    from loom.train.loop import _switch_balance

    m = C.M
    # uniform routing AND a uniform router
    c = torch.zeros(m, m)
    for i in range(m):
        c[i, [(i + j) % m for j in range(C.TOPK)]] = 1.0 / C.TOPK
    flat = torch.zeros(m, m)
    assert float(_switch_balance(c, flat)) == pytest.approx(1.0, abs=1e-5)

    # everyone on the same four, router certain about those four
    same = torch.zeros(8, m)
    same[:, : C.TOPK] = 1.0 / C.TOPK
    peaked = torch.full((8, m), -30.0)
    peaked[:, : C.TOPK] = 30.0
    assert float(_switch_balance(same, peaked)) == pytest.approx(m / C.TOPK, abs=1e-3)

    # same routing, uniform router: still above the floor is NOT claimed --
    # the term is a product, and a flat P is exactly the floor by construction.
    assert float(_switch_balance(same, flat)) == pytest.approx(1.0, abs=1e-5)


def test_switch_balance_pushes_an_operator_that_is_in_no_support_back_up():
    """A dead operator must still get gradient, and it must point UP."""
    from loom.heads.q_delta import topk_simplex_st
    from loom.train.loop import _switch_balance

    torch.manual_seed(0)
    logits = torch.randn(16, C.M)
    logits[:, 5] = -10.0                            # operator 5 is dead
    logits = logits.requires_grad_(True)
    c = topk_simplex_st(logits)
    assert c[:, 5].sum() == 0.0
    (C.BALANCE_COEF * _switch_balance(c, logits)).backward()
    assert logits.grad[:, 5].abs().sum() > 0, "a dead operator gets no gradient"
    assert (logits.grad[:, 5] < 0).all(), "gradient must push the dead operator UP"


def test_switch_balance_reads_the_dense_router_and_not_only_the_hard_support():
    """The substantive difference from the KL it replaces.

    `KL(mean_batch(c) || uniform)` is a function of `c` alone, and `c` is
    exactly zero outside the top-4 support: two routers that pick the same four
    operators are indistinguishable to it no matter how differently confident
    they are. The Switch form reads `P = mean_t softmax(logits)`, so moving an
    out-of-support logit moves the loss.

    Not a claim that the gradient is larger -- measured at random init it is
    not (0.10 vs 0.12 unselected:selected). The magnitude change is
    BALANCE_COEF 3e-3 -> 1e-2, and `grad_ratio/q_delta_logits` is logged every
    100 steps so the real answer comes from the run rather than from here.
    """
    from loom.heads.q_delta import topk_simplex_st
    from loom.losses.balance import balance_kl
    from loom.train.loop import _switch_balance

    torch.manual_seed(1)
    base = torch.randn(16, C.M)
    base[:, : C.TOPK] += 12.0                       # pin the support, both times
    other = base.clone()
    other[:, C.TOPK:] *= 3.0                        # only the LOSERS move

    c_a, c_b = topk_simplex_st(base), topk_simplex_st(other)
    assert torch.equal(c_a.topk(C.TOPK, -1).indices, c_b.topk(C.TOPK, -1).indices)

    assert float(balance_kl(c_a)) == pytest.approx(float(balance_kl(c_b)), abs=1e-3)
    assert abs(float(_switch_balance(c_a, base))
               - float(_switch_balance(c_b, other))) > 1e-4


def test_balance_utilization_is_reported_per_head_not_pooled():
    """One pooled `bank/live_ops` cannot say that q_a had 7 operators alive and
    q_Delta 19 (measured, ctrl, 64 real windows). Aggregate statistics hide the
    structure -- CLAUDE.md, and it has cost a run before.
    """
    model = build_model(_stub_cfg())
    w = WindowSampler(_stub_cfg(), rank=0, world=1, seed=0).next(0)
    m = model.compute_losses(w, 0, 0, 0)[1]
    for k in ("bank/live_ops", "bank/live_ops_q_delta", "bank/entropy_q_delta",
              "bank/live_ops_q_a", "bank/entropy_q_a"):
        assert k in m, f"{k} missing from {sorted(m)}"
    assert 0 < m["bank/live_ops_q_delta"] <= C.M
    assert 0.0 <= m["bank/entropy_q_delta"] <= math.log(C.M) + 1e-6


def test_delta_sel_is_the_discrimination_guard_and_is_reported_per_horizon():
    """`Delta_sel = d(A(c_other) z, z+) - d(A(c_true) z, z+)`, `c_other` a REAL
    coefficient from another window in the batch.

    `Delta_op` compares against a uniform random simplex point, so it only says
    the bank is alive. This asks whether the coefficient THIS window produced
    beats one another window produced, which is the question the method rests
    on. On the R0-A checkpoints it was +0.0002 (ctrl) / +0.0000 (zinit).

    Pinned here by construction rather than by value: with every window's `c`
    identical, `c.roll(1)` is `c` and the gap must be exactly zero.
    """
    model = build_model(_stub_cfg())
    zs = [torch.randn(4, C.K, C.D) for _ in range(C.N_STATES)]
    zts = [torch.randn(4, C.K, C.D) for _ in range(C.N_STATES)]

    one = S.sparse_simplex(1)
    same = [one.expand(4, C.M).contiguous() for _ in range(C.DEPTH)]
    m = model._delta_sel(zs, zts, same)
    assert set(m) == {"delta_sel", *(f"delta_sel/h{h + 1}" for h in range(C.DEPTH))}
    for k, v in m.items():
        assert abs(v) < 1e-5, f"{k} = {v} with an identical c in every row"

    # distinct coefficients: a real, finite, generally nonzero gap
    m2 = model._delta_sel(zs, zts, [S.sparse_simplex(4) for _ in range(C.DEPTH)])
    assert all(math.isfinite(v) for v in m2.values())


def test_grad_probe_reports_the_q_delta_logit_ratio_on_its_cadence(tmp_path):
    """The per-entry gradient ratio needs a backward, so it runs every
    `optim.grad_probe_every` steps and not every step.

    On the stub path q_Delta exposes no logits, so this exercises the cadence
    and the "return nothing rather than something wrong" branch; the real head
    fills the numbers in. `_probe_grad` is what `main` sets.
    """
    from loom.train.loop import GRAD_PROBE_EVERY

    assert GRAD_PROBE_EVERY == 100
    model = build_model(_stub_cfg())
    assert model._probe_grad is False
    assert model.grad_probe_metrics() == {}
    model._probe_grad = True
    w = WindowSampler(_stub_cfg(), rank=0, world=1, seed=0).next(0)
    model.compute_losses(w, 0, 0, 0)
    # stubs have no dense logits to hang retain_grad on -> nothing, not junk
    assert model.grad_probe_metrics() == {}


def test_a_stub_link_logs_every_new_metric(tmp_path):
    """End to end through `main`, so the metrics really land in metrics.jsonl."""
    cfg = write_test_config(tmp_path, run={"steps": 3, "log_every": 1},
                            optim={"lr": 1e-3, "warmup": 2, "grad_probe_every": 1})
    run_dir = tmp_path / "run"
    run_link(cfg, run_dir)
    rows = curve(run_dir)
    assert len(rows) == 3
    for k in ("delta_op", "delta_sel", "delta_sel/h1", "dyn/neg",
              "bank/live_ops_q_delta", "bank/entropy_q_a", "act/decode"):
        assert k in rows[-1], f"{k} missing from {sorted(rows[-1])}"


# ═══════════════════════════════════════════════════════════════════════════
#  9 · THE THREE STOP PATHS
# ═══════════════════════════════════════════════════════════════════════════

def test_decide_local_covers_all_three_paths():
    now, deadline, safety = 1000.0, 2000.0, 100.0
    assert not decide_local(now, deadline, safety, False, False).stop
    assert decide_local(now, deadline, safety, True, False) == \
        type(decide_local(now, deadline, safety, True, False))(True, "signal")
    assert decide_local(now, deadline, safety, False, True).reason == "sentinel"
    assert decide_local(1950.0, deadline, safety, False, False).reason == "budget"
    # the safety margin, not the deadline, is what ends the link
    assert not decide_local(1899.0, deadline, safety, False, False).stop
    assert decide_local(1901.0, deadline, safety, False, False).stop


def test_decide_local_is_torch_free():
    """Unit-testable without a GPU or a torch install; that is why it is separate."""
    src = (ROOT / "loom" / "train" / "preempt.py").read_text()
    head = src.split("class PreemptGuard")[0]
    assert "import torch" not in head


def test_sentinel_file_stops_the_guard(tmp_path):
    guard = PreemptGuard(tmp_path, budget_s=1e9, safety_s=1.0, install_handlers=False)
    assert not guard.should_stop()
    (tmp_path / "STOP").touch()
    assert guard.should_stop() and guard.reason == "sentinel"


def test_budget_stops_the_guard(tmp_path):
    guard = PreemptGuard(tmp_path, budget_s=1.0, safety_s=2.0, install_handlers=False)
    assert guard.should_stop() and guard.reason == "budget"
    assert guard.seconds_left == 0.0


def test_signal_stops_the_guard(tmp_path):
    guard = PreemptGuard(tmp_path, budget_s=1e9, safety_s=1.0, install_handlers=False)
    guard._on_signal(signal.SIGUSR1, None)
    assert guard.should_stop() and "signal" in guard.reason


def test_safety_margin_matches_the_sbatch_signal():
    """`--signal=USR1@N` and the loop's safety_s must be the same number."""
    for stage in STAGES:
        text = (SLURM / f"{stage}.sbatch").read_text()
        m = re.search(r"#SBATCH --signal=USR1@(\d+)", text)
        assert m, f"{stage}.sbatch has no --signal directive"
        assert float(m.group(1)) == DEFAULT_SAFETY_S


def test_heartbeat_is_rank_zero_only_and_carries_delta_op(tmp_path):
    write_heartbeat(tmp_path, 42, rank=1, delta_op=0.5)
    assert not (tmp_path / "HEARTBEAT").exists(), \
        "a non-zero rank wrote the heartbeat; ranks race on the shared .tmp"
    write_heartbeat(tmp_path, 42, rank=0, delta_op=0.5)
    ts, step, delta = read_heartbeat(tmp_path)
    assert step == 42 and delta == pytest.approx(0.5)
    assert ts <= time.time()
    # scripts/watchdog.sh reads field 2 with awk; keep the layout
    assert (tmp_path / "HEARTBEAT").read_text().split()[1] == "42"


# ═══════════════════════════════════════════════════════════════════════════
#  10 · SCHEDULES
# ═══════════════════════════════════════════════════════════════════════════

def test_warmup_length_and_cosine_shape():
    s = CosineWithWarmup(3e-4, warmup_steps=2000, total_steps=100000)
    assert s.lr_at(0) == pytest.approx(3e-4 / 2000)
    assert s.lr_at(1999) == pytest.approx(3e-4)              # warmup ends exactly here
    assert s.lr_at(2000) == pytest.approx(3e-4)
    mid = [s.lr_at(i) for i in range(2000, 100000, 5000)]
    assert all(a > b for a, b in zip(mid, mid[1:])), "cosine is not monotone decreasing"
    assert s.lr_at(100000) == pytest.approx(3e-4 * s.min_lr_ratio)
    assert s.lr_at(50000) == s.lr_at(50000)                  # pure


def test_schedule_never_reads_the_wall_clock():
    src = (ROOT / "loom" / "train" / "schedule.py").read_text()
    assert "time.time" not in src and "import time" not in src, \
        "a wall-clock schedule changes behaviour across a 4 h requeue boundary"


def test_bank_lr_is_exactly_one_tenth_of_the_estimator_lr():
    """PLAN 4.D. The spectral parameters are poles; they do not survive a full step."""
    model = tiny_model()
    opt = build_optimizer(model, lr=3e-4, lr_scales={"bank": BANK_LR_MULT},
                          module_names=MODULE_NAMES)
    sched = CosineWithWarmup(3e-4, 2000, 100000)
    for step in (0, 1000, 2000, 50000, 99999):
        lrs = sched.apply(opt, step)
        est = [v for k, v in lrs.items() if k.startswith("estimator/")]
        bank = [v for k, v in lrs.items() if k.startswith("bank/")]
        assert est and bank
        for b in bank:
            assert b == est[0] * 0.1, f"step {step}: bank lr {b} vs estimator {est[0]}"
        for k, v in lrs.items():                     # heads follow the estimator
            if k.startswith("q_delta/"):
                assert v == est[0]


def test_param_groups_cover_every_parameter_exactly_once():
    model = tiny_model()
    groups = param_groups(model, module_names=MODULE_NAMES)
    seen = [id(p) for g in groups for p in g["params"]]
    assert len(seen) == len(set(seen)), "a parameter is in two LR groups"
    assert set(seen) == {id(p) for p in model.parameters()}
    assert all(g["weight_decay"] == 0.0 for g in groups if "nodecay" in g["name"])


def test_spectral_parameters_are_not_weight_decayed():
    """log_r/omega are poles, not weights; decay pulls every operator to r=RHO/2."""
    bank = nn.Module()
    bank.log_r = nn.Parameter(torch.zeros(4, 4))
    bank.omega = nn.Parameter(torch.zeros(4, 4))
    model = nn.Module()
    model.bank = bank
    for g in param_groups(model, module_names=("bank",)):
        assert g["weight_decay"] == 0.0


def test_grad_clip_is_applied():
    model = tiny_model()
    for p in model.parameters():
        p.grad = torch.full_like(p, 100.0)
    before = torch.cat([p.grad.flatten() for p in model.parameters()]).norm()
    assert before > 1.0
    reported = clip_grad(model, 1.0)
    after = torch.cat([p.grad.flatten() for p in model.parameters()]).norm()
    assert reported == pytest.approx(float(before), rel=1e-5)
    assert float(after) == pytest.approx(1.0, rel=1e-5)


def test_grad_clip_with_no_grads_is_a_noop():
    assert clip_grad(tiny_model(), 1.0) == 0.0


def test_module_grad_norms_decompose_the_global_norm():
    """The per-module norms must reproduce clip_grad's total exactly.

    If they do not, the decomposition attributes a spike to the wrong module,
    which is worse than not having it at all.
    """
    model = tiny_model()
    torch.manual_seed(0)
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    names = tuple(n for n, _ in model.named_children())
    parts = module_grad_norms(model, sync=None, module_names=names)
    total = math.sqrt(sum(v * v for v in parts.values()))
    # clip_grad mutates the grads, so compare against the pre-clip report it returns
    assert total == pytest.approx(clip_grad(model, 1e9, sync=None), rel=1e-5)


def test_spike_guard_is_off_by_default_and_at_mult_zero():
    """A chain already in flight must not silently acquire a guard."""
    assert SpikeGuard(mult=0.0).enabled is False
    g = SpikeGuard(mult=0.0)
    assert g.check(1e9) is False        # nothing is ever skipped when disabled


def test_spike_guard_rejects_a_spike_but_not_the_ordinary_step():
    g = SpikeGuard(mult=10.0, beta=0.98, warmup=50)
    for _ in range(200):
        assert g.check(3.0) is False    # a stationary regime is never rejected
    assert g.check(3.5) is False
    assert g.check(7000.0) is True
    assert g.check(float("nan")) is True     # NaN must never reach the moments


def test_spike_guard_does_not_deadlock_on_a_sustained_regime_shift():
    """REGRESSION. Measured on run tdgradB, which halted for 457 steps.

    Updating the reference only on ACCEPTED steps is the obvious design and it
    deadlocks: once the gradient regime moves above the threshold nothing is
    accepted, so nothing updates the reference, so the threshold never follows
    and no step is ever taken again. The loss curve just goes flat -- there is
    no error and no warning. `min(gnorm, threshold)` un-sticks it geometrically.
    """
    g = SpikeGuard(mult=10.0, beta=0.98, warmup=50)
    for _ in range(200):
        g.check(3.0)
    stuck = [g.check(93.0) for _ in range(400)]
    assert stuck[0] is True, "a 30x jump should be rejected at first"
    assert False in stuck, "guard deadlocked: never re-admitted a sustained regime"
    assert stuck.index(False) < 100, "took too long to re-admit"
    assert stuck[-1] is False, "still rejecting a regime that is now the norm"


def test_spike_guard_lone_spike_barely_moves_the_threshold():
    """The other failure mode: a burst must not walk the bar up behind itself."""
    g = SpikeGuard(mult=10.0, beta=0.98, warmup=50)
    for _ in range(200):
        g.check(3.0)
    before = g.threshold
    g.check(7e4)
    assert g.threshold / before == pytest.approx(10.0 ** 0.02, rel=1e-6)


def test_spike_guard_state_survives_a_link_boundary():
    """Not persisting this makes the guard a schedule derived from wall clock:
    the first `warmup` steps of every 4 h link would run unguarded."""
    g = SpikeGuard(mult=10.0, beta=0.98, warmup=50)
    for _ in range(200):
        g.check(3.0)
    fresh = SpikeGuard(mult=10.0, beta=0.98, warmup=50)
    fresh.load_state_dict(g.state_dict())
    assert fresh.threshold == pytest.approx(g.threshold)
    assert fresh.n == g.n
    assert fresh.check(7000.0) is True       # guarded immediately, no re-warmup


# ═══════════════════════════════════════════════════════════════════════════
#  11 · R2 FREEZE SCHEDULE
# ═══════════════════════════════════════════════════════════════════════════

def _requires_grad(mod) -> set[bool]:
    return {p.requires_grad for p in mod.parameters()}


def test_r2_freezes_estimator_and_bank_together_for_the_first_30_percent():
    """PLAN 7. Freezing the bank alone leaves fixed operators in a drifting basis."""
    cfg = read_config(CONFIGS / "r2.yaml")
    assert set(cfg["freeze"]["modules"]) == {"estimator", "bank"}
    assert cfg["freeze"]["until_frac"] == 0.3

    total = int(cfg["run"]["steps"])
    sched = FreezeSchedule(tuple(cfg["freeze"]["modules"]),
                           float(cfg["freeze"]["until_frac"]), total)
    model = tiny_model()
    trainable = set(cfg["train_modules"])

    for step in (0, 1, total // 4, sched.until_step - 1):
        assert sched.apply(model, step, trainable) is True
        assert _requires_grad(model.estimator) == {False}, f"E unfrozen at {step}"
        assert _requires_grad(model.bank) == {False}, f"bank unfrozen at {step}"
        assert _requires_grad(model.q_delta) == {True}, "q_delta must keep training"

    for step in (sched.until_step, sched.until_step + 1, total - 1):
        assert sched.apply(model, step, trainable) is False
        assert _requires_grad(model.estimator) == {True}, f"E still frozen at {step}"
        assert _requires_grad(model.bank) == {True}, f"bank still frozen at {step}"


def test_freezing_the_bank_without_the_estimator_is_rejected():
    with pytest.raises(ValueError, match="drifting basis"):
        FreezeSchedule(("bank",), 0.3, 1000)


def test_no_freeze_schedule_is_a_noop():
    sched = FreezeSchedule((), 0.0, 1000)
    model = tiny_model()
    for step in (0, 500, 999):
        assert sched.apply(model, step, set(MODULE_NAMES)) is False
    assert _requires_grad(model.estimator) == {True}


def test_freeze_never_unfreezes_a_module_this_run_does_not_train():
    """R1 must not start training q_a just because the freeze window ended."""
    sched = FreezeSchedule(("estimator", "bank"), 0.3, 100)
    model = tiny_model()
    sched.apply(model, 50, trainable={"estimator"})
    assert _requires_grad(model.estimator) == {True}
    assert _requires_grad(model.bank) == {False}


def test_freeze_schedule_is_step_based_not_wall_clock():
    sched = FreezeSchedule(("estimator", "bank"), 0.3, 1000)
    assert sched.until_step == 300
    assert sched.frozen_at(299) and not sched.frozen_at(300)


# ═══════════════════════════════════════════════════════════════════════════
#  12 · SBATCH SANITY
# ═══════════════════════════════════════════════════════════════════════════

GHOST_PARTITIONS = re.compile(r"batch_block[123]")
REAL_PARTITIONS = "polar4,polar3,polar,grizzly"
ACCOUNT = "edgeai_tao-ptm_image-foundation-model-clip"


@pytest.mark.parametrize("stage", STAGES)
def test_sbatch_names_no_nonexistent_partition(stage):
    """batch_block1/2/3 do not exist on this cluster: a 0-second job failure."""
    text = (SLURM / f"{stage}.sbatch").read_text()
    assert not GHOST_PARTITIONS.search(text), \
        f"{stage}.sbatch names a partition that does not exist"
    assert f"#SBATCH --partition={REAL_PARTITIONS}" in text


@pytest.mark.parametrize("stage", STAGES)
def test_multinode_sbatch_never_names_a_single_node_partition(stage):
    """`batch_singlenode` carries QoS `..._1_node_per_job`: 8 GPUs, 1970G per job.

    A partition list is NOT a fallback chain. sbatch validates the request
    against it and rejects the whole submission with `QOSMaxMemoryPerJob`
    rather than scheduling onto a partition that fits, so naming it alongside
    polar4 failed all five stages at submit time -- every LOOM stage is
    multi-node. This asserts the rule rather than a literal string, so it keeps
    holding if the partition list is ever retuned.
    """
    text = (SLURM / f"{stage}.sbatch").read_text()
    m = re.search(r"^#SBATCH --nodes=(\d+)", text, re.M)
    assert m, f"{stage}.sbatch does not declare --nodes"
    # The --partition directive only, never the whole file: these sbatch files
    # carry a comment explaining why batch_singlenode must not be listed, and a
    # substring search flags the very comment that documents the rule.
    part = re.search(r"^#SBATCH --partition=(\S+)", text, re.M)
    assert part, f"{stage}.sbatch does not declare --partition"
    named = part.group(1).split(",")
    if int(m.group(1)) > 1:
        assert "batch_singlenode" not in named, (
            f"{stage}.sbatch is --nodes={m.group(1)} but names batch_singlenode; "
            "sbatch rejects the whole submission, it does not fall through"
        )


def test_no_file_in_this_workstream_names_a_nonexistent_partition():
    """Cheap, and it covers the scripts and configs too, not just the sbatch files."""
    targets = [*SLURM.glob("*.sbatch"), *(ROOT / "scripts").glob("*.sh"),
               *CONFIGS.glob("*.yaml"), *(ROOT / "loom" / "train").rglob("*.py")]
    offenders = [str(p) for p in targets if GHOST_PARTITIONS.search(p.read_text())]
    assert not offenders, f"nonexistent partition named in: {offenders}"


@pytest.mark.parametrize("stage", STAGES)
def test_sbatch_directives(stage):
    text = (SLURM / f"{stage}.sbatch").read_text()
    for directive in (f"#SBATCH --account={ACCOUNT}",
                      "#SBATCH --time=04:00:00",
                      "#SBATCH --requeue",
                      f"#SBATCH --signal=USR1@{int(DEFAULT_SAFETY_S)}",
                      "#SBATCH --ntasks-per-node=8",
                      "#SBATCH --gpus-per-node=8",
                      "#SBATCH --output=logs/%x_%j.out"):
        assert directive in text, f"{stage}.sbatch is missing {directive!r}"
    # SLURM stops parsing #SBATCH at the first executable line
    body_start = next(i for i, l in enumerate(text.splitlines())
                      if l.strip() and not l.startswith("#"))
    assert not any(l.startswith("#SBATCH") for l in text.splitlines()[body_start:]), \
        "an #SBATCH directive sits after the first executable line and is ignored"


@pytest.mark.parametrize("stage", STAGES)
def test_sbatch_matches_the_configs_gpu_count(stage):
    cfg = read_config(CONFIGS / f"{stage}.yaml")
    text = (SLURM / f"{stage}.sbatch").read_text()
    assert f"#SBATCH --nodes={cfg['slurm']['nodes']}" in text, \
        f"{stage}: sbatch node count disagrees with the config"
    assert f'configs/{stage}.yaml' in text


@pytest.mark.parametrize("stage,gpus", [("r0a", 16), ("r0b", 16), ("r1", 64),
                                        ("r2", 64), ("r3", 32)])
def test_gpu_counts_match_plan_section_7(stage, gpus):
    cfg = read_config(CONFIGS / f"{stage}.yaml")
    assert cfg["slurm"]["nodes"] * cfg["slurm"]["gpus_per_node"] == gpus


@pytest.mark.parametrize("stage,hours", [("r0a", 8), ("r0b", 24), ("r1", 96),
                                         ("r2", 144), ("r3", 72)])
def test_n_links_covers_the_planned_wall_time(stage, hours):
    """The 4 h cap means "~8 h" is >= 2 links and "~6 d" is >= 36."""
    n = read_config(CONFIGS / f"{stage}.yaml")["slurm"]["n_links"]
    assert n >= math.ceil(hours / 4), f"{stage}: {n} links cannot cover {hours} h"


def test_sbatch_sets_an_overridable_wandb_mode_and_the_stable_id():
    """Mode is online (compute nodes have a route, re-measured 2026-08-17) but must
    stay overridable; the stable id is what makes chained links one run.

    This used to assert `WANDB_MODE=offline` outright, encoding a measurement that
    had gone stale -- api.wandb.ai now answers from a compute node in 0.22 s and a
    real `wandb.init` completes in 1.8 s. Assert the *rule* instead: a mode is set,
    and `LOOM_WANDB_MODE` can override it without editing five files. Whether it
    resolves to online or offline is a cluster fact, not an invariant, and
    `wandb_util.init` degrades to offline by itself if the route disappears.
    """
    for stage in STAGES:
        text = (SLURM / f"{stage}.sbatch").read_text()
        assert 'WANDB_MODE="${LOOM_WANDB_MODE:-' in text, (
            f"{stage}.sbatch must set WANDB_MODE with a LOOM_WANDB_MODE override"
        )
        assert "wandb_id" in text and "WANDB_RUN_ID" in text
        assert 'WANDB_DIR="$RUN_DIR"' in text, "wandb appends wandb/ itself"
        assert "RANK=\"$SLURM_PROCID\"" in text, "rank must come from SLURM, not torchrun"
        assert "LOOM_TIME_BUDGET_S" in text
        assert "link start:" in text


def test_scripts_never_scancel():
    """A scancel breaks the chain; scontrol requeue costs one checkpoint interval."""
    for p in (ROOT / "scripts").glob("*.sh"):
        for line in p.read_text().splitlines():
            code = line.split("#", 1)[0].strip()      # drop comments, incl. "never scancel"
            assert "scancel" not in code, f"{p.name} calls scancel: {line.strip()}"
    assert "scontrol requeue" in (ROOT / "scripts" / "watchdog.sh").read_text()


def test_submit_uses_singleton_chaining_not_self_requeue():
    text = (ROOT / "scripts" / "submit.sh").read_text()
    assert "--dependency=singleton" in text
    assert "LOOM_RUN_NAME" in text and "--export=ALL" in text
