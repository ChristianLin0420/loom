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
import math
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
from loom.losses.dyn import dyn_loss, ln_cosine_distance
from loom.train import ckpt as ckpt_mod
from loom.train import fsdp as fsdp_mod
from loom.train import wandb_util
from loom.train.determinism import (
    enable_determinism, rank_identity, set_global_seed, set_step_seed, torch_generator,
)
from loom.train.preempt import PreemptGuard, write_heartbeat
from loom.train.schedule import (
    BANK_LR_MULT, CosineWithWarmup, EMATarget, FreezeSchedule, SpikeGuard,
    build_optimizer, clip_grad, module_grad_norms,
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

#: How often to measure the per-entry gradient ratio at q_Delta's logits
#: (unselected : selected). It needs the backward pass to have run, so it is not
#: free the way a forward-only statistic is; `optim.grad_probe_every` overrides,
#: 0 turns it off.
GRAD_PROBE_EVERY = 100

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
    """`stubs.StubDecoder` is frozen and predates the `(proprio, c)` contract.

    It never reads its first argument beyond `shape[0]` / `device` / `dtype`, so
    handing it `(B, dof_e)` proprio instead of `(B, K, D)` belief works
    unchanged; only the names here move, so the shape the loop passes matches
    what the REAL decoder now takes.
    """

    def __init__(self, embodiment: str) -> None:
        super().__init__(embodiment)
        self.scale = nn.Parameter(torch.ones(self.dof))

    def loss(self, proprio: Tensor, c: Tensor, a_seg: Tensor) -> Tensor:
        return ((self.forward(proprio, c) * self.scale.to(proprio.dtype)
                 - a_seg) ** 2).mean()


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
    from PLAN 4.C. ``L_dyn`` now calls Team C's ``loom/losses/dyn.py::dyn_loss``
    -- the inline version it replaced had no negatives at all, so the configured
    ``dyn.negatives: within_trajectory`` was inert and a constant ``z`` drove the
    term to zero for free. ``L_act`` / ``L_proposal`` stay inline: they are one
    call to ``Decoder.loss`` and one to ``Proposal.log_prob`` respectively, and
    the loop needs the per-horizon decomposition that the wrappers hide.
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
    """1 - cos(LN(a), LN(b)), averaged over slots and batch. PLAN 4.C.

    Exactly `losses.dyn.ln_cosine_distance(a, b, "per_slot").mean()`, kept as a
    local so the diagnostics below read the same as they did before `L_dyn`
    itself moved into `loom/losses/dyn.py`, and so `delta_op` stays comparable
    with every run logged before that.
    """
    return ln_cosine_distance(a, b, "per_slot").mean()


def _coeff_and_logits(head, *args, **kw) -> tuple[Tensor, Tensor | None]:
    """`c` and, when the head exposes them, the DENSE logits behind it.

    Team C's `QDelta` / `QAction` take `return_logits=True`; the frozen stubs do
    not. The Switch load-balancing term needs the dense router distribution
    (`P_m`), not the sparse `c` -- with `P_m` read off `c` the gradient on an
    operator that is in no support is exactly zero, which is the closed path the
    term exists to open. `None` here means "stub", and `_switch_balance` falls
    back to `c`.
    """
    try:
        out = head(*args, return_logits=True, **kw)
    except TypeError:
        return head(*args, **kw), None
    if isinstance(out, tuple):
        return out[0], out[1]
    return out, None


def _switch_balance(c: Tensor, logits: Tensor | None, topk: int = C.TOPK) -> Tensor:
    """The Switch auxiliary load-balancing loss,  `M * sum_m f_m P_m`.

    * `f_m` -- the fraction of ROUTING SLOTS that went to operator `m`. One
      token contributes `TOPK` slots (its hard top-4 support), so `sum_m f_m`
      is 1 by construction and this reduces to Switch's own definition at
      `TOPK = 1`. Non-differentiable and detached; it is a count.
    * `P_m` -- the mean DENSE router probability for `m`, `softmax(logits)`
      averaged over tokens. `sum_m P_m == 1`.

    Range: `1.0` when both are uniform (the degenerate floor -- read it as a
    plateau, not as progress) up to `M / TOPK = 32` when every token routes to
    the same four operators and the router is certain about it.

    What it replaces and why (owner's call; the measurement behind it):
    `KL(mean_batch(c) || uniform(M))` is a function of `c` alone, and `c` is
    exactly zero for every operator outside the top-4 support -- so the whole
    term sees only the hard routing decision, never how nearly an operator was
    chosen. Its gradient still reaches an unselected logit (`topk_simplex_st`
    returns `hard + soft - soft.detach()`, so the backward is dense) but
    measured on the R0-A checkpoints it arrived at 0.0006 (ctrl) / 0.0001
    (zinit) of a selected operator's per-entry size. The Switch form is a
    function of the DENSE router as well as of the routing, so the loss changes
    when an out-of-support logit moves at all -- and its coefficient went
    3e-3 -> 1e-2 with it, which is the part that is unambiguously a magnitude
    change. `grad_ratio/q_delta_logits` is logged so this is read and not
    assumed.

    Gradient, for the record:
        dL/dl_{t,m} = (M/T) * (f_m - sum_j f_j P_{t,j}) * P_{t,m}
    negative -- i.e. pushing the logit UP -- for every operator whose load is
    below the load-weighted average, including the ones at exactly zero.
    """
    m = c.shape[-1]
    k = min(topk, m)
    cf = c.detach().float().reshape(-1, m)          # `f` is a count: detached
    t = cf.shape[0]
    idx = cf.topk(k, dim=-1).indices
    f = torch.zeros_like(cf).scatter_(1, idx, 1.0).sum(0) / float(t * k)
    # `P` carries the gradient. The stub path has no dense router to read, so
    # it falls back to `c` -- which is the OLD behaviour and is why the real
    # heads are asked for `return_logits`.
    dense = (torch.softmax(logits.float().reshape(-1, m), dim=-1)
             if logits is not None else c.float().reshape(-1, m))
    return float(m) * (f * dense.mean(0)).sum()


def _usage(cs: Sequence[Tensor]) -> tuple[float, float]:
    """(operators used, usage entropy in nats) for one head's coefficients.

    Aggregate statistics hide structure (CLAUDE.md), which is why this is
    reported per HEAD rather than pooled: `q_a` and `q_Delta` had 7 and 19
    operators alive respectively on the R0-A checkpoints, and one pooled number
    cannot say that. `1e-4` is the same liveness threshold the pooled
    `bank/live_ops` used, so the two stay comparable.
    """
    u = torch.stack(list(cs), 0).detach().float().flatten(0, -2).mean(0)
    u = u / u.sum().clamp_min(1e-12)
    return (float((u > 1e-4).sum()),
            float(-(u * u.clamp_min(1e-12).log()).sum()))


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
        #: bf16 on CUDA, None on CPU. See `_cast`.
        self.compute_dtype: torch.dtype | None = None
        #: set by main() on the steps that measure the q_Delta logit-grad ratio
        self._probe_grad = False
        #: this step's dense q_Delta logits, one per horizon (None on the stub
        #: path). Rebuilt every `compute_losses`, read once after `backward`.
        self._qd_logits: list[Tensor | None] = []

    # ── beliefs ────────────────────────────────────────────────────────────
    def _cast(self, z: Tensor) -> Tensor:
        """Pin the belief to the compute dtype at the estimator boundary.

        Autocast is not enough on its own, and the reason is worth writing down.
        `E` is a pre-LN Perceiver, so its last op is a LayerNorm -- and
        `layer_norm` sits in autocast's **fp32** cast policy, so `z` comes back
        fp32 even with every matmul in the block running bf16. `bank.step` then
        does `a * x` with `a` bf16 (einsum, autocast-lowered) and `x` fp32, which
        promotes the whole affine rollout to fp32: 2x the activation memory for
        the one part of the model that is pure elementwise algebra.

        The cast belongs here rather than inside the bank because the bank casts
        `c` to its *parameter* dtype and hands back that, so the call site is the
        only place that controls the belief's dtype. It is also correct under
        FSDP, whose fp32 master weights are a storage detail: the forward already
        runs on bf16 shards, and the LayerNorm upcast happens either way.
        """
        return z if self.compute_dtype is None else z.to(self.compute_dtype)

    def _est_kw(self, window: dict) -> dict:
        return {"embodiment": window["embodiment"]} if self._est_takes_embodiment else {}

    def beliefs(self, window: dict) -> list[Tensor]:
        kw = self._est_kw(window)
        z, out = None, []
        for feats in window["feats"]:
            z = self._cast(self.estimator(feats, z, **kw))
            out.append(z)
        return out

    @torch.no_grad()
    def target_beliefs(self, window: dict) -> list[Tensor]:
        kw = self._est_kw(window)
        z, out = None, []
        for feats in window["feats"]:
            z = self._cast(self.ema(feats, z, **kw))
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

        # coefficients from q_Delta -- action-free, available on every dataset.
        # The dense logits come back too: `L_balance` needs the router
        # distribution, and the grad probe needs a tensor to hang `retain_grad`
        # on. `_qd_logits` is cleared every step so a stale graph cannot be read.
        self._qd_logits = []
        c_delta = []
        for h in range(C.DEPTH):
            c_h, lg_h = _coeff_and_logits(self.q_delta, zs[h], zts[h + 1])
            c_delta.append(c_h)
            self._qd_logits.append(lg_h)
        if self._probe_grad and all(lg is not None for lg in self._qd_logits):
            for lg in self._qd_logits:
                lg.retain_grad()

        # ── L_dyn ──────────────────────────────────────────────────────────
        #
        # `loom/losses/dyn.py`, not an inline cosine. The inline version had NO
        # negatives at all: a plain `1 - cos(A(c)z, z+)` that a constant `z` and
        # `A(c) ~ I` drive to zero for free while `c` carries nothing. Every
        # config has asked for `negatives: within_trajectory` since R0-A and
        # nothing was reading it.
        if self._on("dyn"):
            dcfg = self.loss_cfg.get("dyn", {})
            c_seq = torch.stack(c_delta, dim=1)                       # (B,DEPTH,M)
            z_tgt = torch.stack([zts[h + 1] for h in range(C.DEPTH)], dim=1)
            out = dyn_loss(
                self.bank, zs[0], c_seq, z_tgt,
                negatives=self.negatives,
                min_gap=int(dcfg.get("min_gap", 2)),
                neg_weight=float(dcfg.get("neg_weight", 1.0)),
                neg_margin=float(dcfg.get("neg_margin", 0.1)),
                weights=C.DYN_WEIGHTS,
                cosine=str(dcfg.get("cosine", "per_slot")),
                # CPU generator: the negatives' multinomial and the delta_op
                # draw both happen where it lives and the indices move to the
                # coefficients' device. See `losses.dyn._draw`.
                generator=torch_generator(seed, step, rank, tag="dyn"),
            )
            if self.check_bf16:
                fsdp_mod.assert_bf16(out["z_hat1"], "bank.step output (L_dyn rollout)")
            terms["dyn"] = out["loss"]
            metrics["dyn/pos"] = float(out["dyn"])
            metrics["dyn/neg"] = float(out["neg"])
            metrics["dyn/cos_pos"] = float(out["cos_pos"])
            # The build assert, unchanged, so the number stays comparable with
            # every prior run. Delta_op says the BANK is alive; it is not the
            # discrimination test -- `Delta_sel` below is.
            metrics["delta_op"] = self._delta_op(zs, zts, c_delta, step, rank, seed)
            metrics.update(self._delta_sel(zs, zts, c_delta))

        # ── L_act ──────────────────────────────────────────────────────────
        c_act: list[Tensor] | None = None
        c_act_lg: list[Tensor | None] = []
        if self._on("act") and window.get("actions") is not None:
            qa, dec = self.q_action[emb], self.decoder[emb]
            c_act, c_act_lg, l_act, l_align = [], [], zero, zero
            for h in range(C.DEPTH):
                a_seg = window["actions"][:, h]                 # (B, H_OP, dof_e)
                c_a, lg_a = _coeff_and_logits(qa, a_seg, zs[h])
                c_act.append(c_a)
                c_act_lg.append(lg_a)
                # D_e(proprio_t, c) -- NOT D_e(z, c). Given the whole belief the
                # decoder is a behaviour-cloning head and needs nothing from `c`,
                # so `L_act` exerts no pressure on the coefficient (measured:
                # act/decode 0.2489 -> 0.0559 while c_a held 2-3 distinct top-4
                # supports over 64 real windows). `feats[h]["proprio"]` is
                # (B, dof_e) -- ONE timestep, at the START of segment h.
                proprio = window["feats"][h]["proprio"]
                l_act = l_act + dec.loss(proprio, c_a, a_seg)
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
            all_lg = self._qd_logits + (c_act_lg if c_act is not None else [])
            lg = (torch.stack(all_lg, 0).flatten(0, -2)
                  if all(x is not None for x in all_lg) else None)
            terms["balance"] = _switch_balance(allc, lg)
            # pooled, unchanged, so it stays comparable with the prior runs
            cbar = allc.detach().float().mean(0).clamp_min(1e-9)
            cbar = cbar / cbar.sum()
            metrics["bank/live_ops"] = float((cbar > 1e-4).sum())
            # ...and split by head, which is the number that carries information
            ops, ent = _usage(c_delta)
            metrics["bank/live_ops_q_delta"], metrics["bank/entropy_q_delta"] = ops, ent
            if c_act is not None:
                ops, ent = _usage(c_act)
                metrics["bank/live_ops_q_a"], metrics["bank/entropy_q_a"] = ops, ent

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
        ``L_dyn`` while ``c`` carries nothing.

        **This is NOT the discrimination test.** It compares the true operator
        against a *uniform random* simplex point, so it says the bank is alive
        and nothing more; a collapsed ``c`` makes the comparison vacuous in the
        other direction too (CLAUDE.md). ``_delta_sel`` is the real guard.

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

    @torch.no_grad()
    def _delta_sel(self, zs, zts, c_true) -> dict[str, float]:
        """THE discrimination guard.  `Delta_sel > 0` or the coefficient is decoration.

            c_other   = c.roll(1, dims=0)          # a REAL c, from another window
            Delta_sel = d(A(c_other) z, z+) - d(A(c_true) z, z+)

        Same distance `L_dyn` uses, same batched single `bank.step` as
        `_delta_op`, reported per horizon and as the mean.

        `Delta_op` compares the true operator against a *uniform random* point
        on the simplex, so it answers "is the bank alive". This asks the only
        question that matters for the method: does the coefficient this window
        produced predict this window's transition BETTER THAN a coefficient a
        different window produced? On the R0-A checkpoints the answer was
        +0.0002 (ctrl) / +0.0000 (zinit) -- any other window's operator
        predicted the transition exactly as well, which means `c` was not
        carrying the transition at all.

        `roll(1, dims=0)` needs B >= 2 to be a different window; at B=1 it is the
        identity and this is identically zero by construction.
        """
        z_in = torch.cat([zs[h] for h in range(C.DEPTH)], 0)
        z_tgt = torch.cat([zts[h + 1] for h in range(C.DEPTH)], 0)
        c_pos = torch.cat(list(c_true), 0)
        c_oth = torch.cat([c.roll(1, dims=0) for c in c_true], 0)
        out = self.bank.step(torch.cat([c_pos, c_oth], 0), torch.cat([z_in, z_in], 0))
        n = z_in.shape[0]
        gap = (ln_cosine_distance(out[n:], z_tgt)
               - ln_cosine_distance(out[:n], z_tgt)).view(C.DEPTH, -1).mean(1)
        m = {f"delta_sel/h{h + 1}": float(gap[h]) for h in range(C.DEPTH)}
        m["delta_sel"] = float(gap.mean())
        return m

    def grad_probe_metrics(self) -> dict[str, float]:
        """Per-entry |grad| at q_Delta's logits, unselected : selected.

        Called by `main` AFTER `loss.backward()` and BEFORE `zero_grad`, on the
        steps `_probe_grad` marked. Returns `{}` on every other step and on the
        stub path.

        This is the number `L_balance` exists to move. `topk_simplex_st` returns
        `hard + soft - soft.detach()`, so the backward into an out-of-support
        logit is not blocked -- it is simply small, and a ratio near zero means
        an operator that has fallen out of every support is receiving no useful
        signal to come back with.
        """
        if not self._probe_grad:
            return {}
        num = den = 0.0
        n_num = n_den = 0
        for lg in self._qd_logits:
            g = None if lg is None else lg.grad
            if g is None:
                continue
            g = g.detach().float().abs()
            # selected == in the hard top-4 support, read off the logits
            # themselves so this does not depend on `c` still being alive.
            sel = torch.zeros_like(g).scatter_(
                -1, lg.detach().float().topk(C.TOPK, dim=-1).indices, 1.0).bool()
            num += float(g[~sel].sum()); n_num += int((~sel).sum())
            den += float(g[sel].sum()); n_den += int(sel.sum())
        if n_num == 0 or n_den == 0 or den == 0.0:
            return {}
        per_unsel, per_sel = num / n_num, den / n_den
        return {"grad_ratio/q_delta_logits": per_unsel / per_sel if per_sel else 0.0,
                "grad_per_entry/q_delta_unselected": per_unsel,
                "grad_per_entry/q_delta_selected": per_sel}

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


#: (factory name, kwargs builder). `build_loader` is the agreed Team A factory;
#: the class constructors are the fallback and spell `world_size`, not `world`.
_LOADER_FACTORIES = (
    ("build_loader", lambda r, w, s, d: dict(rank=r, world=w, seed=s, device=d)),
    ("WindowLoader", lambda r, w, s, d: dict(rank=r, world_size=w, seed=s, device=d)),
    ("LoomLoader", lambda r, w, s, d: dict(rank=r, world_size=w, seed=s)),
)


def build_sampler(cfg: dict, rank: int, world: int, seed: int, device: str):
    """Team A's loader for a real source; the stub sampler ONLY for `source: stub`.

    Falling back to stub windows when the config asked for LIBERO is a wasted run,
    not a degraded one: R0-A would have trained 16 GPUs for eight hours on
    `torch.randn` and produced a first score that reads like a modelling result.
    A missing loader is therefore fatal here, and the traceback names every
    factory that was tried and why each failed.
    """
    source = cfg.get("data", {}).get("source", "stub")
    if source == "stub":
        return WindowSampler(cfg, rank, world, seed, device)

    tried: list[str] = []
    try:
        mod = importlib.import_module("loom.data.loader")
    except Exception as e:
        raise RuntimeError(
            f"data.source is {source!r} but loom.data.loader could not be imported "
            f"({e!r}). Set data.source: stub to train on random windows on purpose."
        ) from e

    for name, kwargs_for in _LOADER_FACTORIES:
        fn = getattr(mod, name, None)
        if fn is None:
            tried.append(f"{name}: absent")
            continue
        try:
            # Never hardcode num_workers here: fit_workers/_fit_shared_memory
            # inside the loader shrink it to what /dev/shm actually holds, and
            # that differs between login (64 MiB) and compute (1008 GiB) nodes.
            return fn(cfg, **kwargs_for(rank, world, seed, device))
        except Exception as e:
            tried.append(f"{name}: {e!r}")

    raise RuntimeError(
        f"data.source is {source!r} but no usable loader factory was found in "
        f"loom.data.loader:\n  " + "\n  ".join(tried) +
        "\nTeam A owns build_loader(cfg, *, rank, world, seed, device). "
        "Set data.source: stub to train on random windows on purpose."
    )


def _to_device(window: dict, device: str, dtype=None) -> dict:
    """Move and, on the CUDA path, pin every float tensor to the compute dtype.

    Cached tower features arrive in whatever Team A's cache stored (fp16 or
    fp32); leaving them fp32 drags the heads out of bf16 the same way the belief
    did, and doubles the resident size of the largest tensor in the batch.
    Integer tensors are left alone.
    """
    if device == "cpu" and dtype is None:
        return window

    def _m(v):
        v = v.to(device, non_blocking=True)
        return v.to(dtype) if dtype is not None and v.is_floating_point() else v

    out = dict(window)
    out["feats"] = [{k: _m(v) for k, v in f.items()} for f in window["feats"]]
    out["lang"] = _m(window["lang"])
    if window.get("actions") is not None:
        out["actions"] = _m(window["actions"])
    if window.get("reward") is not None:
        out["reward"] = _m(window["reward"])
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
    #: spike-rejection reference. Optimizer state in every sense that matters:
    #: left out of the checkpoint it would reset at every 4 h link boundary, and
    #: the first ``warmup`` steps of every link would run unguarded.
    guard: Any = None
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
    # `spike_mult: 0` is OFF and is the default: a guard is an intervention, and a
    # chain already in flight must not silently acquire one at a link boundary.
    spike = SpikeGuard(mult=float(ocfg.get("spike_mult", 0.0)),
                       beta=float(ocfg.get("spike_beta", 0.98)),
                       warmup=int(ocfg.get("spike_warmup", 100)))
    state = TrainState(model=model, optimizer=opt, scheduler=sched, ema=model.ema,
                       sampler=sampler, guard=spike)
    if rank == 0 and spike.enabled:
        print(f"[rank0] spike guard ON: skip when gnorm > {spike.mult}x the "
              f"running geometric mean (beta={spike.beta}, warmup={spike.warmup})",
              flush=True)

    # bf16 throughout (PLAN 9). FSDP's MixedPrecision covers the wrapped modules;
    # autocast covers the single-GPU debug path where nothing is wrapped.
    amp = device == "cuda" and str(cfg["fsdp"].get("precision", "bf16")) == "bf16"
    # LoomModel._cast pins the belief to bf16 at the estimator boundary, so the
    # rollout is bf16 on the stub path too and the assert is checked everywhere
    # amp is on. It was previously skipped for stubs, which is exactly why a GPU
    # smoke could not have caught the fp32 rollout.
    model.compute_dtype = torch.bfloat16 if amp else None
    model.check_bf16 = amp

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
    grad_report = bool(ocfg.get("grad_report", True))
    probe_every = int(ocfg.get("grad_probe_every", GRAD_PROBE_EVERY))
    t0, last_delta, last_sel = time.time(), float("nan"), float("nan")

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

        window = _to_device(sampler.next(step), device, model.compute_dtype)
        # `retain_grad` on q_Delta's logits, on this step only. Cheap (a (B, M)
        # tensor per horizon) but it is still a diagnostic, and it needs the
        # backward to have run, so it is not on every step.
        model._probe_grad = probe_every > 0 and step % probe_every == 0
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            loss, metrics = model.compute_losses(window, step, rank, seed)

        opt.zero_grad(set_to_none=True)
        gnorm, skipped, gparts = 0.0, False, {}
        if loss.requires_grad:
            loss.backward()
            # BEFORE any all-reduce or clip: these are activation gradients on
            # THIS rank's own logits, and the question is their relative size.
            metrics.update(model.grad_probe_metrics())
            # Replicated modules are invoked through step()/log_prob()/loss(), not
            # forward(), so no DDP/FSDP hook ever fires for them. Sync by hand.
            sync.all_reduce_grads()
            # BEFORE clip_grad: afterwards every number carries the same `coef`
            # and the decomposition is no longer in the units the spike happened
            # in. One extra all-reduce of ~7 floats, unconditional on every rank.
            if grad_report:
                gparts = module_grad_norms(model, sync=sync,
                                           module_names=MODULE_NAMES)
            gnorm = clip_grad(model, grad_clip, sync=sync)
            # `gnorm` is the globally reduced pre-clip norm, so every rank feeds
            # the guard the identical number and reaches the identical verdict
            # without another collective. It must stay that way: a guard that
            # skipped on one rank only would desynchronise the optimizer.
            skipped = state.guard.check(gnorm)
            if not skipped:
                opt.step()
        state.ema.update(model.estimator)

        state.global_step += 1
        state.samples_seen += batch * world
        last_delta = metrics.get("delta_op", last_delta)
        last_sel = metrics.get("delta_sel", last_sel)

        if rank == 0 and metrics_fp is not None:
            metrics_fp.write(json.dumps({
                "global_step": state.global_step, "lr": lrs.get("estimator/decay",
                                                                sched.lr_at(step)),
                "grad_norm": gnorm, "frozen": is_frozen,
                "grad_skipped": int(skipped),
                # null, not Infinity: json.dumps emits a bare `Infinity` for the
                # disabled/warming-up guard, which Python reads back but jq and
                # every other JSON reader rejects.
                "grad_thresh": (state.guard.threshold
                                if math.isfinite(state.guard.threshold) else None),
                **{f"gnorm/{k}": v for k, v in gparts.items()},
                "embodiment": window["embodiment"], **metrics}) + "\n")

        if state.global_step % log_every == 0:
            write_heartbeat(run_dir, state.global_step, rank, last_delta)
            if rank == 0:
                parts = " ".join(f"{k[:3]}={v:.1f}" for k, v in gparts.items())
                print(f"[rank0] step {state.global_step} loss={metrics['loss']:.4f} "
                      f"delta_op={last_delta:+.4f} delta_sel={last_sel:+.4f} "
                      f"lr={sched.lr_at(step):.3e} "
                      f"gnorm={gnorm:.3f}{' SKIP' if skipped else ''} "
                      f"[{parts}] frozen={int(is_frozen)} "
                      f"emb={window['embodiment']} "
                      f"{state.global_step / max(1e-6, time.time() - t0):.2f} it/s",
                      flush=True)
            wandb_util.log(run, {
                **metrics, "grad_norm": gnorm, "samples_seen": state.samples_seen,
                "frozen": float(is_frozen),
                "grad_skipped": float(skipped),
                **{f"gnorm/{k}": v for k, v in gparts.items()},
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
