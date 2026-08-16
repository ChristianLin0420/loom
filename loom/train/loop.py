"""The only training loop. Every run (R0-A ... R3) is a config, not a new script.

    python -m loom.train       --config configs/r0a.yaml
    python -m loom.train.loop  --config configs/r0a.yaml

Design notes that are load bearing:

* ``build_model(cfg)`` is the **single integration point**. Until Teams A/B/C/E
  land, every module falls back to ``stubs.*``. Swapping a stub for a real module
  is a one-line change inside that function and nothing else in this file moves.

* ``--steps`` is the *schedule horizon* and is identical on every link of a
  chained run. ``--stop_at`` / ``--budget_s`` / ``--safety_s`` / ``--run_dir``
  end *this link* and are excluded from ``config_hash``. Mixing the two makes
  the LR curve depend on how long a job happened to run.

* Every step begins with ``set_step_seed(seed, global_step, rank)``, so a step is
  a pure function of ``(seed, step, rank, params)``. That is what makes resume
  continuity assertable rather than eyeballable.

* Batches are embodiment-homogeneous (PLAN 9). The loop reads
  ``window["embodiment"]`` and dispatches ``q_a`` / ``D_e`` through a ModuleDict.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

# `contracts` and `stubs` live at the repo root, not inside the package.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import contracts as C
import stubs as S
from loom.train import ckpt as ckpt_mod
from loom.train import fsdp as fsdp_mod
from loom.train import wandb_util
from loom.train.determinism import (
    enable_determinism, rank_identity, set_global_seed, set_step_seed, torch_generator,
)
from loom.train.preempt import PreemptGuard, write_heartbeat
from loom.train.schedule import (
    BANK_LR_MULT, CosineWithWarmup, EMATarget, FreezeSchedule, build_optimizer, clip_grad,
)

__all__ = [
    "LoomModel", "WindowSampler", "TrainState", "build_model", "load_config",
    "config_hash", "parse_args", "main", "LINK_LOCAL_KEYS", "MODULE_NAMES",
]

#: Top-level trainable modules, in PLAN order. Used for LR groups and freezing.
MODULE_NAMES = ("estimator", "bank", "q_delta", "q_action", "decoder",
                "proposal", "potential")

#: Knobs describing *this link*, not the experiment. Never in the config hash.
LINK_LOCAL_KEYS = ("run_dir", "stop_at", "budget_s", "safety_s", "no_wandb",
                   "allow_reshard", "config_path")

_CONFIG_DIR = _ROOT / "configs"


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════

def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _read_yaml(path: Path) -> dict:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f) or {}


def _resolve(path: str, relative_to: Path) -> Path:
    p = Path(path)
    for cand in (p, relative_to / p, _CONFIG_DIR / p, _ROOT / p):
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(f"config {path!r} not found (looked near {relative_to})")


def read_config(path: str | Path, _seen: tuple[str, ...] = ()) -> dict:
    """YAML with a single-parent ``extends:`` key, deep merged child-over-parent."""
    p = _resolve(str(path), Path.cwd())
    if str(p) in _seen:
        raise ValueError(f"circular extends: {' -> '.join(_seen + (str(p),))}")
    cfg = _read_yaml(p)
    parent = cfg.pop("extends", None)
    if parent:
        cfg = _deep_merge(read_config(_resolve(parent, p.parent), _seen + (str(p),)), cfg)
    return cfg


def _coerce(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def load_config(args) -> dict:
    """Merged config + link-local section. This dict is the whole experiment."""
    cfg = read_config(args.config)
    cfg.setdefault("run", {})
    cfg.setdefault("data", {})
    cfg.setdefault("model", {})
    cfg.setdefault("optim", {})
    cfg.setdefault("losses", {})
    cfg.setdefault("fsdp", {})
    cfg.setdefault("freeze", {})
    cfg.setdefault("train_modules", list(MODULE_NAMES))

    # generic --set a.b=value overrides, applied before the named ones
    for item in args.set or []:
        key, _, val = item.partition("=")
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _coerce(val)

    if args.steps is not None:
        cfg["run"]["steps"] = args.steps
    if args.seed is not None:
        cfg["run"]["seed"] = args.seed
    if args.lr is not None:
        cfg["optim"]["lr"] = args.lr
    if args.batch is not None:
        cfg["data"]["batch_per_gpu"] = args.batch
    if args.ckpt_every is not None:
        cfg["run"]["ckpt_every"] = args.ckpt_every
    if args.log_every is not None:
        cfg["run"]["log_every"] = args.log_every
    if args.deterministic:
        cfg["run"]["deterministic"] = True

    name = cfg["run"].get("name", "loom")
    cfg["link"] = {
        "run_dir": args.run_dir or str(_ROOT / "runs" / name),
        "stop_at": args.stop_at,
        "budget_s": args.budget_s,
        "safety_s": args.safety_s,
        "no_wandb": bool(args.no_wandb),
        "allow_reshard": bool(args.allow_reshard),
        "config_path": str(args.config),
    }
    return cfg


def config_hash(cfg: dict) -> str:
    """Identity of the *experiment*. ``cfg["link"]`` is excluded by construction."""
    d = {k: v for k, v in cfg.items() if k != "link"}
    return hashlib.blake2b(
        json.dumps(d, sort_keys=True, default=str).encode(), digest_size=8
    ).hexdigest()


# ═══════════════════════════════════════════════════════════════════════════
#  TRAINABLE STUB SHIMS
#
#  stubs.py is frozen and its modules are deliberately parameter-free, so a loop
#  built on them would carry no gradient at all: the optimizer would be a no-op,
#  the resume test would be vacuous and the R2 freeze schedule would have nothing
#  to freeze. These shims add one real parameter each and keep the stub's random
#  output, which is exactly enough to exercise every path.
# ═══════════════════════════════════════════════════════════════════════════

class _StubEstimator(S.StubEstimator):
    def __init__(self) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.ones(1))
        self.shift = nn.Parameter(torch.zeros(C.K, C.D))

    def forward(self, feats, z_prev):
        z = super().forward(feats, z_prev)
        return z * self.gain.to(z.dtype) + self.shift.to(z.dtype)


class _StubDecoder(S.StubDecoder):
    def __init__(self, embodiment: str) -> None:
        super().__init__(embodiment)
        self.scale = nn.Parameter(torch.ones(self.dof))

    def loss(self, z: Tensor, c: Tensor, a_seg: Tensor) -> Tensor:
        return ((self.forward(z, c) * self.scale.to(z.dtype) - a_seg) ** 2).mean()


class _StubQAction(S.StubQAction):
    def __init__(self) -> None:
        super().__init__()
        self.temp = nn.Parameter(torch.zeros(1))

    def forward(self, a_seg: Tensor, z: Tensor) -> Tensor:
        c = super().forward(a_seg, z)
        # straight-through: keeps c exactly on the simplex, carries a gradient
        return c + (self.temp.to(c.dtype) - self.temp.detach().to(c.dtype))


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL
# ═══════════════════════════════════════════════════════════════════════════

class _Bound:
    """Binds one embodiment of a shared per-embodiment head.

    Team C's real ``QAction`` / ``Decoder`` are themselves keyed by embodiment
    (``forward(..., embodiment=)``), while the stubs are one module per body.
    Both are presented to the loop as ``heads[embodiment]``, so the dispatch in
    ``compute_losses`` does not care which it got.
    """

    __slots__ = ("_inner", "_emb")

    def __init__(self, inner: nn.Module, emb: str):
        self._inner, self._emb = inner, emb

    def __call__(self, *a, **kw):
        return self._inner(*a, embodiment=self._emb, **kw)

    def loss(self, *a, **kw):
        return self._inner.loss(*a, embodiment=self._emb, **kw)


class EmbodimentHeads(nn.Module):
    """``heads[embodiment]`` over either a ModuleDict or one shared container."""

    def __init__(self, inner: nn.Module, names: Sequence[str], shared: bool):
        super().__init__()
        self.inner = inner
        self._names = tuple(names)
        self.shared = bool(shared)

    def keys(self):
        return self._names

    def __contains__(self, name: str) -> bool:
        return name in self._names

    def __getitem__(self, name: str):
        if name not in self._names:
            raise KeyError(f"no head for embodiment {name!r}; have {self._names}")
        return _Bound(self.inner, name) if self.shared else self.inner[name]


def _import(module_path: str):
    try:
        return importlib.import_module(module_path), None
    except Exception as e:              # not written yet, or broken mid-edit
        return None, e


def _try_build(module_path: str, class_names: Sequence[str], kwargs: dict,
               fallback, mode: str, what: str):
    """Import a real module if it exists, otherwise fall back to a stub.

    ``mode`` is ``model.use_stubs``:
      ``True``  always stubs -- no real import is attempted at all
      ``"auto"`` real if importable, stub otherwise (the default)
      ``False`` real required; a missing module is a hard error
    """
    if mode is True:
        return fallback(), "stub"
    mod, err = _import(module_path)
    if mod is not None:
        for cn in class_names:
            cls = getattr(mod, cn, None)
            if cls is None:
                continue
            try:
                return cls(**kwargs), "real"
            except TypeError as e:
                err = e
                try:
                    return cls(), "real"
                except Exception as e2:
                    err = e2
            except Exception as e:
                err = e
    if mode is False:
        raise RuntimeError(
            f"model.use_stubs is false but {what} could not be built from "
            f"{module_path}: {err!r}"
        )
    return fallback(), "stub"


def _build_heads(module_path: str, class_names: Sequence[str], kwargs: dict,
                 embodiments: Sequence[str], stub_factory, mode: str,
                 what: str) -> tuple[EmbodimentHeads, str]:
    """Per-embodiment heads (``q_a``, ``D_e``), whichever shape the real one has."""
    if mode is not True:
        mod, err = _import(module_path)
        if mod is not None:
            for cn in class_names:
                cls = getattr(mod, cn, None)
                if cls is None:
                    continue
                try:                     # Team C's container form
                    return EmbodimentHeads(
                        cls(embodiments=list(embodiments), **kwargs),
                        embodiments, shared=True), "real"
                except Exception as e:
                    err = e
        if mode is False:
            raise RuntimeError(
                f"model.use_stubs is false but {what} could not be built from "
                f"{module_path}: {err!r}"
            )
    d = nn.ModuleDict({e: stub_factory(e) for e in embodiments})
    return EmbodimentHeads(d, embodiments, shared=False), "stub"


def build_model(cfg: dict) -> "LoomModel":
    """THE integration point. Real modules if importable, ``stubs.*`` otherwise.

    Integration order is B -> C -> A -> E -> D -> F (PLAN 6.2); each team's module
    starts being used the moment it imports cleanly, with no edit here.

    ``model.use_stubs``
      ``true``   never import a real module. What ``tests/test_train.py`` uses:
                 a 150 M Perceiver takes minutes per CPU step and would make the
                 login-node suite unrunnable.
      ``"auto"`` real if importable, stub otherwise. The default.
      ``false``  a missing real module is a hard error. Set this in the R0-A
                 config once Phase 2 starts, so a typo cannot silently train a
                 random-output stub for eight hours.

    The four loss terms are computed in :meth:`LoomModel.compute_losses` straight
    from PLAN 4.C. ``loom/losses/{dyn,act,proposal_bc,balance}.py`` (Team C) has
    since landed with an equivalent API (``dyn_loss``, ``act_loss``,
    ``proposal_bc_loss``, ``balance_loss``); swapping to it is the remaining
    integration step and belongs in this function.
    """
    mcfg = dict(cfg.get("model", {}))
    mode = mcfg.get("use_stubs", "auto")
    embodiments = list(cfg.get("data", {}).get("embodiments", ["libero_franka"]))
    for e in embodiments:
        if e not in C.EMBODIMENTS:
            raise ValueError(f"unregistered embodiment {e!r}; adapters register at import")

    estimator, k_est = _try_build(
        "loom.model.estimator", ("Estimator", "PerceiverEstimator", "LoomEstimator"),
        dict(mcfg.get("estimator", {})), _StubEstimator, mode, "estimator")
    bank, k_bank = _try_build(
        "loom.model.bank", ("OperatorBank", "Bank", "LoomBank"),
        dict(mcfg.get("bank", {})), S.StubBank, mode, "bank")
    q_delta, k_qd = _try_build(
        "loom.heads.q_delta", ("QDelta", "QDeltaHead"),
        dict(mcfg.get("q_delta", {})), S.StubQDelta, mode, "q_delta")
    proposal, k_prop = _try_build(
        "loom.heads.proposal", ("Proposal", "ProposalHead", "PolicyProposal"),
        dict(mcfg.get("proposal", {})), S.StubProposal, mode, "proposal")

    q_action, k_qa = _build_heads(
        "loom.heads.q_action", ("QAction", "QActionHead"),
        dict(mcfg.get("q_action", {})), embodiments, lambda e: _StubQAction(),
        mode, "q_action")
    decoder, k_dec = _build_heads(
        "loom.heads.decoder", ("Decoder", "DecoderHead", "FlowDecoder"),
        dict(mcfg.get("decoder", {})), embodiments, _StubDecoder, mode, "decoder")

    potential = None
    k_pot = "off"
    if cfg.get("losses", {}).get("potential", {}).get("enabled", False):
        potential, k_pot = _try_build(
            "loom.heads.potential", ("Potential", "PotentialHead"),
            dict(mcfg.get("potential", {})), S.StubPotential, mode, "potential")

    print("[build] " + " ".join(
        f"{n}={k}" for n, k in [("E", k_est), ("bank", k_bank), ("q_delta", k_qd),
                                ("q_action", k_qa), ("decoder", k_dec),
                                ("proposal", k_prop), ("potential", k_pot)]), flush=True)

    return LoomModel(estimator=estimator, bank=bank, q_delta=q_delta,
                     q_action=q_action, decoder=decoder, proposal=proposal,
                     potential=potential, cfg=cfg)


def _accepts(module, name: str) -> bool:
    """Does ``module.forward`` take a keyword called ``name``?"""
    import inspect

    try:
        return name in inspect.signature(module.forward).parameters
    except (TypeError, ValueError):
        return False


def _cos_dist(a: Tensor, b: Tensor) -> Tensor:
    """1 - cos(LN(a), LN(b)), averaged over slots and batch. PLAN 4.C."""
    a = F.layer_norm(a.float(), (a.shape[-1],))
    b = F.layer_norm(b.float(), (b.shape[-1],))
    return (1.0 - F.cosine_similarity(a, b, dim=-1)).mean()


class LoomModel(nn.Module):
    """Everything that carries a gradient, plus the EMA target ``L_dyn`` needs.

    Child names are exactly :data:`MODULE_NAMES` plus ``ema``, because
    ``schedule.param_groups`` and ``FreezeSchedule`` address modules by name and
    ``fsdp.wrap_for_training`` wraps them by name.
    """

    def __init__(self, *, estimator, bank, q_delta, q_action, decoder, proposal,
                 potential=None, cfg: dict):
        super().__init__()
        self.estimator = estimator
        self.bank = bank
        self.q_delta = q_delta
        self.q_action = q_action
        self.decoder = decoder
        self.proposal = proposal
        if potential is not None:
            self.potential = potential
        self.ema = EMATarget(estimator, tau=float(cfg.get("optim", {}).get(
            "ema_tau", C.EMA_TAU)))
        self.cfg = cfg
        lc = cfg.get("losses", {})
        self.loss_cfg = {k: dict(v) for k, v in lc.items()}
        self.negatives = self.loss_cfg.get("dyn", {}).get("negatives", "within_trajectory")
        # Team B's estimator holds per-embodiment proprio projections and infers
        # the body from proprio.shape[-1] when not told. Two registered bodies
        # already share dof=7, so inference is ambiguous and picking the wrong
        # projection is a trains-fine-scores-zero bug. The window names the body;
        # pass it through whenever the estimator can accept it.
        self._est_takes_embodiment = _accepts(estimator, "embodiment")
        #: set by main() on the CUDA path; PLAN 9 build assert, see fsdp.assert_bf16
        self.check_bf16 = False

    # ── beliefs ────────────────────────────────────────────────────────────
    def _est_kw(self, window: dict) -> dict:
        return {"embodiment": window["embodiment"]} if self._est_takes_embodiment else {}

    def beliefs(self, window: dict) -> list[Tensor]:
        kw = self._est_kw(window)
        z, out = None, []
        for feats in window["feats"]:
            z = self.estimator(feats, z, **kw)
            out.append(z)
        return out

    @torch.no_grad()
    def target_beliefs(self, window: dict) -> list[Tensor]:
        kw = self._est_kw(window)
        z, out = None, []
        for feats in window["feats"]:
            z = self.ema(feats, z, **kw)
            out.append(z.detach())
        return out

    # ── the four losses ────────────────────────────────────────────────────
    def compute_losses(self, window: dict, step: int, rank: int,
                       seed: int) -> tuple[Tensor, dict[str, float]]:
        emb = window["embodiment"]
        # `.keys()` and not `emb in self.q_action`: an FSDP wrapper forwards
        # __getitem__ but not __contains__, so `in` falls back to the old
        # integer-iteration protocol and raises KeyError("... 0"). Measured on 2
        # A100s. The heads are unwrapped now, but the idiom stays safe either way.
        if emb not in tuple(self.q_action.keys()):
            raise KeyError(
                f"batch embodiment {emb!r} has no q_a/D_e; data.embodiments is "
                f"{list(self.q_action.keys())}. Batches are embodiment-homogeneous "
                f"(PLAN 9), so this is a loader/config mismatch, not a padding issue."
            )
        fsdp_mod.assert_features_are_cached(window)

        zs = self.beliefs(window)                      # online,  N_STATES
        zts = self.target_beliefs(window)              # EMA target, stop-grad
        dev = zs[0].device
        metrics: dict[str, float] = {}
        terms: dict[str, Tensor] = {}
        zero = torch.zeros((), device=dev)

        # coefficients from q_Delta -- action-free, available on every dataset
        c_delta = [self.q_delta(zs[h], zts[h + 1]) for h in range(C.DEPTH)]

        # ── L_dyn ──────────────────────────────────────────────────────────
        if self._on("dyn"):
            zhat, l_dyn = zs[0], zero
            for h in range(C.DEPTH):
                zhat = self.bank.step(c_delta[h], zhat)
                if h == 0 and self.check_bf16:
                    fsdp_mod.assert_bf16(zhat, "bank.step output (L_dyn rollout)")
                l_dyn = l_dyn + C.DYN_WEIGHTS[h] * _cos_dist(zhat, zts[h + 1])
            terms["dyn"] = l_dyn
            metrics["delta_op"] = self._delta_op(zs, zts, c_delta, step, rank, seed)

        # ── L_act ──────────────────────────────────────────────────────────
        c_act: list[Tensor] | None = None
        if self._on("act") and window.get("actions") is not None:
            qa, dec = self.q_action[emb], self.decoder[emb]
            c_act, l_act, l_align = [], zero, zero
            for h in range(C.DEPTH):
                a_seg = window["actions"][:, h]                 # (B, H_OP, dof_e)
                c_a = qa(a_seg, zs[h])
                c_act.append(c_a)
                l_act = l_act + dec.loss(zs[h], c_a, a_seg)
                # q_a regresses onto sg(q_Delta) -- one coefficient space by
                # construction, which is why there is no separate alignment loss.
                l_align = l_align + ((c_a - c_delta[h].detach()) ** 2).sum(-1).mean()
            terms["act"] = (l_act + l_align) / C.DEPTH
            metrics["act/decode"] = float(l_act.detach()) / C.DEPTH
            metrics["act/align"] = float(l_align.detach()) / C.DEPTH

        # ── L_proposal ─────────────────────────────────────────────────────
        if self._on("proposal"):
            src = c_act if c_act is not None else c_delta
            lp = zero
            for h in range(C.DEPTH):
                lp = lp + self.proposal.log_prob(zs[h], window["lang"], src[h].detach()).mean()
            terms["proposal"] = -lp / C.DEPTH

        # ── L_balance ──────────────────────────────────────────────────────
        if self._on("balance"):
            allc = torch.stack(c_delta + (c_act or []), dim=0).flatten(0, -2)
            cbar = allc.float().mean(0).clamp_min(1e-9)
            cbar = cbar / cbar.sum()
            terms["balance"] = (cbar * (cbar.log() + float(torch.log(torch.tensor(
                float(C.M)))))).sum()
            metrics["bank/live_ops"] = float((cbar > 1e-4).sum())

        # ── R3: potential + GRPO ───────────────────────────────────────────
        if self._on("potential") and getattr(self, "potential", None) is not None:
            reward = window.get("reward")
            if reward is None:
                reward = torch.zeros(zs[0].shape[0], device=dev, dtype=zs[0].dtype)
            phi = self.potential(zs[-1], window["lang"])
            terms["potential"] = F.mse_loss(phi.float(), reward.float())
        if self._on("grpo") and getattr(self, "potential", None) is not None:
            terms["grpo"] = self._grpo(zs[0], window, dev)

        total = zero
        for name, t in terms.items():
            w = float(self.loss_cfg.get(name, {}).get("weight", 1.0))
            total = total + w * t
            metrics[f"loss/{name}"] = float(t.detach())
        metrics["loss"] = float(total.detach())
        return total, metrics

    def _on(self, name: str) -> bool:
        return bool(self.loss_cfg.get(name, {}).get("enabled", False))

    @torch.no_grad()
    def _delta_op(self, zs, zts, c_true, step: int, rank: int, seed: int) -> float:
        """Delta_op = d(A(c_rand) z, z+) - d(A(c_true) z, z+), which must be > 0.

        A build assert, not a metric (PLAN 4.C). Latent states 8 steps apart are
        ~0.95 cosine-similar before training, so ``A(c) ~ I`` nearly satisfies
        ``L_dyn`` while ``c`` carries nothing. If this flatlines in the first few
        thousand steps the model has collapsed to a plain latent policy: flip
        ``losses.dyn.negatives`` to ``within_trajectory`` before burning the run.

        ``within_trajectory`` negatives are ``c`` from another segment of the SAME
        trajectory at least 2 segments away -- same scene, same body, genuinely
        different effect. Uncurated in-batch negatives would make two bodies
        producing the same world effect negatives for each other, which is the
        opposite of what a shared bank should learn.
        """
        g = torch_generator(seed, step, rank, tag="delta_op")
        negs = []
        for h in range(C.DEPTH):
            if self.negatives == "within_trajectory":
                far = [j for j in range(C.DEPTH) if abs(j - h) >= 2]
                negs.append(c_true[far[int(torch.randint(len(far), (1,), generator=g))]])
            else:
                negs.append(S.sparse_simplex(zs[h].shape[0], device=zs[h].device,
                                             dtype=zs[h].dtype))
        # One batched bank.step for all 2 * DEPTH probes. Delta_op is computed on
        # EVERY step, and the bank rebuilds its (M, K, D/2) lambda tables inside
        # every call, so 8 separate calls put ~30% of the step into a diagnostic.
        z_in = torch.cat([zs[h] for h in range(C.DEPTH)], 0)
        z_tgt = torch.cat([zts[h + 1] for h in range(C.DEPTH)], 0)
        out = self.bank.step(torch.cat(c_true + negs, 0), torch.cat([z_in, z_in], 0))
        n = z_in.shape[0]
        return float(_cos_dist(out[n:], z_tgt) - _cos_dist(out[:n], z_tgt))

    def _grpo(self, z: Tensor, window: dict, dev) -> Tensor:
        """Group-relative advantage on pi_c, scored by Phi. R3 only.

        PLAN 7 names "potential + GRPO" for R3 but does not specify the estimator,
        and nothing in this repo produces sim rollouts yet. This is the plumbing:
        sample a group, score the rollout leaf with Phi, centre the reward inside
        the group, and reinforce. Team E owns the search; revisit before R3.
        """
        n = int(self.loss_cfg.get("grpo", {}).get("group", 8))
        c_seq = self.proposal.sample(z, window["lang"], n=n)         # (B, n, M)
        with torch.no_grad():
            plan = c_seq.unsqueeze(2).expand(-1, -1, C.DEPTH, -1)
            leaf = self.bank.rollout(plan, z)                        # (B, n, K, D)
            r = self.potential(leaf, window["lang"]).float()         # (B, n)
            adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-6)
        b, k, d = z.shape
        zf = z.unsqueeze(1).expand(b, n, k, d).reshape(b * n, k, d)
        lf = window["lang"].unsqueeze(1).expand(-1, n, -1, -1).flatten(0, 1)
        lp = self.proposal.log_prob(zf, lf, c_seq.reshape(b * n, C.M)).view(b, n)
        return -(adv * lp).mean()


# ═══════════════════════════════════════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════════════════════════════════════

class WindowSampler:
    """Deterministic window source with a checkpointable cursor.

    Team A owns ``loom/data/loader.py``; ``source: stub`` here keeps Team D
    unblocked and is swapped by ``build_sampler``. Whatever replaces it must keep
    two properties: batches are embodiment-homogeneous, and the cursor is
    ``state_dict``-able so a resumed link does not re-see the same windows.
    """

    def __init__(self, cfg: dict, rank: int, world: int, seed: int, device: str = "cpu"):
        dcfg = cfg.get("data", {})
        self.embodiments = list(dcfg.get("embodiments", ["libero_franka"]))
        self.batch = int(dcfg.get("batch_per_gpu", 2))
        self.action_free = bool(dcfg.get("action_free", False))
        self.rank, self.world, self.seed = rank, world, seed
        self.device = device
        self.cursor = 0
        self.epoch = 0

    def embodiment_for(self, step: int) -> str:
        """One embodiment per batch, cycled between batches (PLAN 9)."""
        return self.embodiments[(step + self.rank) % len(self.embodiments)]

    def next(self, step: int) -> dict:
        emb = self.embodiment_for(step)
        w = S.make_window(b=self.batch, embodiment=emb, device=self.device,
                          action_free=self.action_free)
        self.cursor += self.batch
        return w

    def state_dict(self) -> dict:
        return {"cursor": self.cursor, "epoch": self.epoch,
                "embodiments": self.embodiments, "batch": self.batch}

    def load_state_dict(self, sd: dict) -> None:
        self.cursor = int(sd.get("cursor", 0))
        self.epoch = int(sd.get("epoch", 0))


def log_shm_headroom(cfg: dict) -> None:
    """/dev/shm is 64 MiB on this cluster. DataLoader workers pass tensors through
    it, so a LIBERO prefetch queue does not fit and the failure surfaces as
    ``DataLoader worker (pid N) exited unexpectedly`` or a bare bus error -- which
    reads like a code bug, not a resource limit. Both torch sharing strategies use
    /dev/shm, so switching strategy does not help.

    Team A's ``LoomLoader`` calls ``fit_workers()`` to shrink workers/prefetch (down
    to in-process) so the queue fits; this only makes the measured number visible in
    the log of every link, because a compute node's value may differ from the login
    node's.
    """
    try:
        from loom.data.loader import shm_free_bytes
    except Exception:
        return
    try:
        free = shm_free_bytes()
    except Exception as e:
        print(f"[data] /dev/shm unreadable ({e!r})", flush=True)
        return
    dcfg = cfg.get("data", {})
    print(f"[data] /dev/shm free {free / 2 ** 20:.0f} MiB, "
          f"num_workers={dcfg.get('num_workers', 0)} "
          f"prefetch_factor={dcfg.get('prefetch_factor', 2)} "
          f"(LoomLoader shrinks these with fit_workers to fit)", flush=True)


def build_sampler(cfg: dict, rank: int, world: int, seed: int, device: str):
    """Real loader if Team A's is importable, otherwise the stub sampler."""
    if cfg.get("data", {}).get("source", "stub") != "stub":
        try:
            mod = importlib.import_module("loom.data.loader")
            for cn in ("build_loader", "WindowLoader", "Loader", "LoomLoader"):
                fn = getattr(mod, cn, None)
                if fn is not None:
                    # Never hardcode num_workers: fit_workers/_fit_shared_memory
                    # inside the loader shrink it to what /dev/shm can hold.
                    return fn(cfg, rank=rank, world=world, seed=seed, device=device)
        except Exception as e:
            print(f"[data] real loader unavailable ({e!r}); using stub windows", flush=True)
    return WindowSampler(cfg, rank, world, seed, device)


def _to_device(window: dict, device: str) -> dict:
    if device == "cpu":
        return window
    out = dict(window)
    out["feats"] = [{k: v.to(device, non_blocking=True) for k, v in f.items()}
                    for f in window["feats"]]
    out["lang"] = window["lang"].to(device, non_blocking=True)
    if window.get("actions") is not None:
        out["actions"] = window["actions"].to(device, non_blocking=True)
    if window.get("reward") is not None:
        out["reward"] = window["reward"].to(device, non_blocking=True)
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  TRAIN STATE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TrainState:
    """Everything that must survive a requeue.

    ``tests/test_train.py::test_state_coverage_by_reflection`` walks this object
    and fails if any field exposing ``state_dict()`` is not saved by
    ``ckpt.build_state``. Add a field here without adding it there and a test
    goes red instead of a 37-link chain going silently wrong.
    """

    model: nn.Module
    optimizer: Any
    scheduler: CosineWithWarmup
    ema: Any
    sampler: Any
    global_step: int = 0
    samples_seen: int = 0

    #: fields that are configuration or a live handle, not mutable training state
    NOT_STATE = ("global_step", "samples_seen")


# ═══════════════════════════════════════════════════════════════════════════
#  LAUNCH ASSERTS
# ═══════════════════════════════════════════════════════════════════════════

def assert_ranks_distinct(ident: dict) -> None:
    """Log this rank's identity and prove no two ranks collide.

    Under plain ``srun`` nothing sets ``RANK``; SLURM exports ``SLURM_PROCID``.
    If the sbatch forgets to map it, every task reads rank 0, draws the same
    windows and writes the same checkpoint shard -- and the loss curve looks
    completely normal.
    """
    ident = dict(ident, ckpt_shard=ckpt_mod.shard_name(0, ident["rank"]))
    print(f"[rank{ident['rank']}] " + " ".join(f"{k}={v}" for k, v in ident.items()),
          flush=True)

    # A job allocated GPUs that cannot see them must not quietly train on CPU:
    # a +cu13x wheel on this CUDA-12.2 driver imports fine, reports
    # is_available()==False, and trains on CPU while holding 8 A100s.
    if os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_GPUS_ON_NODE"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "SLURM allocated GPUs but torch.cuda.is_available() is False. "
                "The node driver is CUDA 12.2 -- install torch==2.6.0+cu124 from "
                "https://download.pytorch.org/whl/cu124, not the default wheel."
            )

    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return
    gathered: list = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, ident)
    for f in ("rank", "rng_fingerprint", "ckpt_shard"):
        seen = [g[f] for g in gathered]
        if len(set(seen)) != len(seen):
            raise RuntimeError(
                f"ranks collide on {f!r}: {seen}. Every task is running as the same "
                f"rank -- check that the sbatch maps SLURM_PROCID into RANK."
            )


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args(argv=None):
    p = argparse.ArgumentParser("loom.train")
    p.add_argument("--config", required=True, help="configs/rX.yaml")
    # ── experiment-defining (all in config_hash) ──
    p.add_argument("--steps", type=int, default=None,
                   help="schedule horizon; IDENTICAL on every link of a chain")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--ckpt_every", type=int, default=None)
    p.add_argument("--log_every", type=int, default=None)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--set", action="append", default=[], metavar="a.b=value",
                   help="override any config key, JSON-parsed")
    # ── link-local (excluded from config_hash) ──
    p.add_argument("--run_dir", default=None)
    p.add_argument("--stop_at", type=int, default=None,
                   help="end THIS link at this global_step; --steps still sets the "
                        "schedule, so links are interchangeable")
    p.add_argument("--budget_s", type=float, default=None)
    p.add_argument("--safety_s", type=float, default=None)
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--allow_reshard", action="store_true")
    return p.parse_args(argv)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args)
    link = cfg["link"]
    run_dir = Path(link["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    rcfg, ocfg, dcfg = cfg["run"], cfg["optim"], cfg["data"]
    seed = int(rcfg.get("seed", 0))
    steps = int(rcfg.get("steps", 1000))
    log_every = int(rcfg.get("log_every", 20))
    ckpt_every = int(rcfg.get("ckpt_every", 500))

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if world > 1:
        import torch.distributed as dist

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank % torch.cuda.device_count())
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

    if rcfg.get("deterministic"):
        enable_determinism()
    set_global_seed(seed, rank)
    assert_ranks_distinct(rank_identity(seed, rank, local_rank, world))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    chash = config_hash(cfg)
    if rank == 0:
        (run_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
        print(f"[rank0] run={rcfg.get('name')} config_hash={chash} steps={steps} "
              f"world={world} device={device}", flush=True)

    # ── model ──────────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    trainable = set(cfg.get("train_modules", list(MODULE_NAMES)))
    for name in MODULE_NAMES:
        sub = getattr(model, name, None)
        if sub is not None:
            sub.requires_grad_(name in trainable)

    freeze = FreezeSchedule(
        modules=tuple(cfg["freeze"].get("modules", ())),
        until_frac=float(cfg["freeze"].get("until_frac", 0.0)),
        total_steps=steps,
    )
    model, sync = fsdp_mod.wrap_for_training(model, cfg.get("fsdp"), device=device,
                                             verbose=rank == 0)

    # bank_lr_mult is the PLAN 4.D rule; lr_scales lets R3 additionally tune E
    # and the bank "lightly" without touching the head LRs.
    lr_scales = {"bank": float(ocfg.get("bank_lr_mult", BANK_LR_MULT))}
    lr_scales.update({k: float(v) for k, v in (ocfg.get("lr_scales") or {}).items()})
    opt = build_optimizer(
        model, lr=float(ocfg.get("lr", 3e-4)),
        weight_decay=float(ocfg.get("weight_decay", 0.05)),
        betas=tuple(ocfg.get("betas", (0.9, 0.95))),
        lr_scales=lr_scales,
        module_names=MODULE_NAMES + ("ema",),
    )
    sched = CosineWithWarmup(float(ocfg.get("lr", 3e-4)),
                             int(ocfg.get("warmup", 2000)), steps,
                             float(ocfg.get("min_lr_ratio", 0.05)))
    if rank == 0:
        log_shm_headroom(cfg)
    sampler = build_sampler(cfg, rank, world, seed, "cpu")
    state = TrainState(model=model, optimizer=opt, scheduler=sched, ema=model.ema,
                       sampler=sampler)

    # bf16 throughout (PLAN 9). FSDP's MixedPrecision covers the wrapped modules;
    # autocast covers the single-GPU debug path where nothing is wrapped.
    amp = device == "cuda" and str(cfg["fsdp"].get("precision", "bf16")) == "bf16"
    # The stub estimator emits whatever dtype stubs.make_window produced (fp32,
    # and stubs.py is frozen), so the bf16 build assert only means something with
    # the real modules. Asserting on the stub path fails every GPU smoke test for
    # a reason that is not a bug.
    model.check_bf16 = amp and cfg["model"].get("use_stubs", "auto") is not True

    # ── resume ─────────────────────────────────────────────────────────────
    with fsdp_mod.sharded_state_dict(model):
        payload = ckpt_mod.load_latest(run_dir, map_location=device,
                                       allow_reshard=link["allow_reshard"])
        if payload is not None:
            got = ckpt_mod.restore(payload, state, world_size=world)
            if got["config_hash"] and got["config_hash"] != chash:
                print(f"[rank{rank}] WARNING config_hash changed "
                      f"{got['config_hash']} -> {chash}; this is a different "
                      f"experiment resuming into the same run dir", flush=True)
            print(f"[rank{rank}] resumed at step {state.global_step} "
                  f"(git {got['git_sha'][:8]})", flush=True)

    # ── logging ────────────────────────────────────────────────────────────
    run = None if link["no_wandb"] else wandb_util.init(
        run_dir, rcfg.get("project", "loom"), cfg, rank=rank, name=rcfg.get("name"))
    metrics_fp = None
    if rank == 0:
        metrics_fp = open(run_dir / "metrics.jsonl", "a", buffering=1)

    guard = PreemptGuard(run_dir, budget_s=link["budget_s"],
                         safety_s=link["safety_s"] if link["safety_s"] is not None else 420.0)
    stop_at = min(link["stop_at"], steps) if link["stop_at"] else steps
    batch = int(dcfg.get("batch_per_gpu", 2))
    grad_clip = float(ocfg.get("grad_clip", 1.0))
    t0, last_delta = time.time(), float("nan")

    def _save(step: int, stop_reason: str = "") -> None:
        # stop_reason rides in the payload so a chain of 38 links can be triaged
        # from its checkpoints alone: "signal" is SLURM preempting on schedule,
        # "budget" is the link running out of its own clock, "sentinel" is a
        # human, "" is a periodic save. A run that keeps stopping for "budget"
        # near step 0 is a startup that got slower, not a preemption problem.
        with fsdp_mod.sharded_state_dict(model):
            ckpt_mod.save(
                ckpt_mod.build_state(state, config_hash=chash, world_size=world,
                                     wandb_run_id=wandb_util.stable_run_id(run_dir),
                                     extra={"stop_reason": stop_reason}),
                run_dir, step, keep_last=int(rcfg.get("keep_last", 3)))

    while state.global_step < stop_at:
        step = state.global_step
        set_step_seed(seed, step, rank)
        is_frozen = freeze.apply(model, step, trainable)
        lrs = sched.apply(opt, step)

        window = _to_device(sampler.next(step), device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            loss, metrics = model.compute_losses(window, step, rank, seed)

        opt.zero_grad(set_to_none=True)
        gnorm = 0.0
        if loss.requires_grad:
            loss.backward()
            # Replicated modules are invoked through step()/log_prob()/loss(), not
            # forward(), so no DDP/FSDP hook ever fires for them. Sync by hand.
            sync.all_reduce_grads()
            gnorm = clip_grad(model, grad_clip, sync=sync)
            opt.step()
        state.ema.update(model.estimator)

        state.global_step += 1
        state.samples_seen += batch * world
        last_delta = metrics.get("delta_op", last_delta)

        if rank == 0 and metrics_fp is not None:
            metrics_fp.write(json.dumps({
                "global_step": state.global_step, "lr": lrs.get("estimator/decay",
                                                                sched.lr_at(step)),
                "grad_norm": gnorm, "frozen": is_frozen,
                "embodiment": window["embodiment"], **metrics}) + "\n")

        if state.global_step % log_every == 0:
            write_heartbeat(run_dir, state.global_step, rank, last_delta)
            if rank == 0:
                print(f"[rank0] step {state.global_step} loss={metrics['loss']:.4f} "
                      f"delta_op={last_delta:+.4f} lr={sched.lr_at(step):.3e} "
                      f"gnorm={gnorm:.3f} frozen={int(is_frozen)} "
                      f"emb={window['embodiment']} "
                      f"{state.global_step / max(1e-6, time.time() - t0):.2f} it/s",
                      flush=True)
            wandb_util.log(run, {
                **metrics, "grad_norm": gnorm, "samples_seen": state.samples_seen,
                "frozen": float(is_frozen),
                "seconds_to_budget": guard.seconds_left,
                **{f"lr/{k}": v for k, v in lrs.items()},
            }, state.global_step)

        # EVERY rank, EVERY step. One rank saving while the others train hangs
        # the next collective until SLURM kills the job.
        stop = guard.should_stop()
        if stop or state.global_step % ckpt_every == 0 or state.global_step >= stop_at:
            _save(state.global_step, guard.reason if stop else "")
        if stop:
            print(f"[rank{rank}] stopping at {state.global_step} ({guard.reason})",
                  flush=True)
            break

    if metrics_fp is not None:
        metrics_fp.close()
    wandb_util.finish(run)
    print(f"[rank{rank}] exit at {state.global_step}/{steps}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
