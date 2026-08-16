"""LOOM — the R0 inference policy, and the rate conversion that protects the score.

    z <- E(o, l, z_prev)
    c <- argmax pi_c(. | z, l)
    a <- D_e(z, c)                 # (H_OP, dof) at FPS_CANONICAL
    execute the segment, re-filter                                  (PLAN 4.F)

No search in R0.

**The failure mode this file exists to prevent.** The decoder emits `H_OP = 8`
steps at `FPS_CANONICAL = 30`. LIBERO's environment runs at **20 Hz**, so one
operator is `contracts.env_steps_per_segment(20.0) == 5.333...` environment
steps — deliberately not an integer. Rounding each segment to 5 loses 1/16 of
wall-clock time per segment, which over a 512-step episode is 32 steps, more
than a second and a half of drift. `SegmentClock` therefore carries the
remainder in a fractional accumulator across segments, exactly as
`stubs.StubPolicy` does and as
`tests/test_contracts.py::test_policy_replans_at_segment_boundaries` pins.
PLAN 7 names this as one of the two failure modes to check first when a trained
model scores near zero.

**How many env steps** is this file's problem (`SegmentClock`). **What those
steps contain** is Team A's (`loom.data.canonical.to_env_rate`), and this module
must not have a second opinion about it. `canonical` does not interpolate
actions: semantics are per dimension and registered with no default. LIBERO is
`(delta,) * 6 + (hold,)`, so

* **delta** channels integrate to cumulative displacement, resample *that*, and
  difference back. Integrated motion is preserved exactly and per-step
  magnitude rescales by `src/dst` — a `0.3` command at 20 Hz canonicalises to
  `0.2` at 30 Hz, and the inverse here turns it back into `0.3`. Interpolating
  the delta stream instead would execute two-thirds of the commanded motion
  forever, with no other symptom.
* the **gripper** channel is zero-order held. It only ever takes `{-1, +1}` in
  the demonstrations; interpolating it emits fractional values mid-segment, so
  the grasp is never decisive while everything else looks healthy.

`tests/test_eval.py::test_action_round_trip_*` is the guard on both.

**Import discipline.** This module imports `contracts`, `numpy`, `torch` and
`loom.data.canonical` — nothing from `loom.model` / `loom.heads` / `loom.train`,
which carry the training stack. `canonical` is numpy-level, safe across
interpreters, and is imported precisely so that eval and training share one
rate-conversion code path and cannot drift apart. `load_policy` imports the real
model modules lazily, inside the function, and falls back to `stubs` when they
are absent, so eval stays importable while the other teams are mid-flight.

**Interpreter discipline.** Eval will not run under the training venv: LIBERO
needs python 3.10 + robosuite 1.4 + numpy<2 in a separate conda env, while
training is python 3.13 + robosuite 1.5.2. Everything in `loom.eval` must
therefore stay importable with only `torch`, `numpy` and `contracts`/`stubs`.
Do not couple this package to a py3.13-only dependency, and load checkpoints
with `map_location="cpu"`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor

from contracts import (
    EMBODIMENTS,
    FPS_CANONICAL,
    H_OP,
    EmbodimentSpec,
    ObsFeats,
    env_steps_per_segment,
)

# The one sanctioned cross-team import. Team A owns the forward resampler; this
# is its inverse, from the same file, dispatching on the same registered
# per-dimension semantics. Do not reimplement it here — two implementations of
# one transform is exactly how train and eval come to disagree.
from loom.data.canonical import to_env_rate

__all__ = [
    "to_env_rate",
    "SegmentClock",
    "PolicyModules",
    "LoomPolicy",
    "load_policy",
    "make_policy",
    "PLACEHOLDER_FEATURES",
]


# ═══════════════════════════════════════════════════════════════════════════
#  PLACEHOLDER FEATURE GEOMETRY
#
#  Team A owns the frozen tower and has not finalised it. Until then the
#  default featuriser emits correctly-shaped zeros so the whole harness runs.
#  These are the only guesses in this package and they live in one block.
# ═══════════════════════════════════════════════════════════════════════════

PLACEHOLDER_FEATURES = {
    "patches": 196,      # 14x14 ViT grid
    "dim": 1152,         # SigLIP-so400m width
    "lang_tokens": 16,
}


# ═══════════════════════════════════════════════════════════════════════════
#  RATE CONVERSION
#
#  There is no resampler in this file. `to_env_rate` is re-exported from
#  `loom.data.canonical` so that anyone reaching for one from `loom.eval.policy`
#  gets Team A's, and `SegmentClock` below decides only *how many* env steps a
#  segment becomes — never what is in them.
# ═══════════════════════════════════════════════════════════════════════════

class SegmentClock:
    """The fractional accumulator. One operator -> a variable number of env steps.

    `env_steps_per_segment(20.0)` is 5.333..., so segments alternate 5, 5, 6,
    5, 5, 6, ... The remainder is carried, never rounded away:

        accum += steps_per_segment
        n      = floor(accum)
        accum -= n

    which keeps `|n_replans * steps_per_segment - n_steps_dispatched| < 1` for
    the whole episode. This mirrors `stubs.StubPolicy` step for step, including
    its `max(n, 1)` floor: an env slower than `FPS_CANONICAL / H_OP` = 3.75 Hz
    would want less than one env step per operator, and there is no way to
    execute a fraction of a step, so the segment is truncated and the drift
    bound does not apply below 3.75 Hz. LIBERO at 20 Hz is nowhere near it.

    The accumulator is a float, like the stub's. `16/3` is not representable,
    so the segment pattern is 5,5,5,6,5,5,6,... rather than the exact
    5,5,6,5,5,6,...; the phase differs by at most one env step and the error
    stays at the 1e-13 level over a full episode, which is 12 orders of
    magnitude below the 32-step error that rounding each segment would cause.
    """

    def __init__(self, env_fps: float) -> None:
        if env_fps <= 0:
            raise ValueError(f"env_fps must be positive, got {env_fps}")
        self.env_fps = float(env_fps)
        self.steps_per_segment = env_steps_per_segment(env_fps)
        self.reset()

    def reset(self) -> None:
        self._accum = 0.0
        self.n_replans = 0
        self.n_steps_dispatched = 0

    def next_segment_len(self) -> int:
        """How many env steps this operator gets. Advances the accumulator."""
        self._accum += self.steps_per_segment
        n = math.floor(self._accum)
        self._accum -= n                 # remainder carries forward, never rounded
        n = max(n, 1)                    # sub-3.75 Hz only; see the class docstring
        self.n_replans += 1
        self.n_steps_dispatched += n
        return n

    @property
    def accum(self) -> float:
        return self._accum

    @property
    def drift(self) -> float:
        """`replans * steps_per_segment - dispatched`. Must stay in (-1, 1)."""
        return self.n_replans * self.steps_per_segment - self.n_steps_dispatched


# ═══════════════════════════════════════════════════════════════════════════
#  MODULE BUNDLE
# ═══════════════════════════════════════════════════════════════════════════

def default_featurizer(spec: EmbodimentSpec) -> Callable[[dict, str], ObsFeats]:
    """`obs dict -> ObsFeats` with correct shapes and zero content.

    Placeholder for Team A's frozen tower. It keeps the harness runnable today;
    `load_policy` swaps in the real encoder as soon as one exists. Proprio is
    read from the observation when present so that the one field the policy
    actually needs at R0 is not fabricated.
    """
    p = PLACEHOLDER_FEATURES
    # constant while the tower is a placeholder — allocated once, not per segment
    views = torch.zeros(1, spec.n_views, p["patches"], p["dim"])
    lang = torch.zeros(1, p["lang_tokens"], p["dim"])

    def featurize(obs: dict, instruction: str) -> ObsFeats:
        proprio = obs.get("state", obs.get("proprio"))
        if proprio is None:
            prop = torch.zeros(1, spec.dof)
        else:
            v = np.asarray(proprio, dtype=np.float32).reshape(-1)
            if v.shape[0] < spec.dof:
                v = np.pad(v, (0, spec.dof - v.shape[0]))
            prop = torch.from_numpy(v[: spec.dof].copy()).unsqueeze(0)
        return ObsFeats(views=views, proprio=prop, lang=lang)

    return featurize


@dataclass
class PolicyModules:
    """Everything `LoomPolicy` needs, injected. Stubs or real, same surface.

    Kept as a plain bundle rather than a checkpoint-shaped object so tests can
    hand in `stubs.StubEstimator()` etc. without touching disk.
    """

    estimator: Any
    proposal:  Any
    decoder:   Any
    featurize: Callable[[dict, str], ObsFeats]
    embodiment: str = "libero_franka"
    device: str = "cpu"
    is_stub: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
#  THE POLICY
# ═══════════════════════════════════════════════════════════════════════════

class LoomPolicy:
    """`contracts.Policy` — the R0-A inference path, no search.

    One `act()` call returns one environment action. A new belief and a new
    operator are computed only at segment boundaries, which `SegmentClock`
    places at the true 3.75 Hz operator rate rather than every 5 or every 6
    steps.
    """

    def __init__(
        self,
        modules: PolicyModules,
        *,
        n_candidates: int = 16,
        clip_actions: bool = True,
        env_fps: float | None = None,
    ) -> None:
        self.modules = modules
        self.embodiment = modules.embodiment
        self.spec = EMBODIMENTS[modules.embodiment]
        self.device = modules.device
        self.n_candidates = n_candidates
        self.clip_actions = clip_actions
        self._low = np.asarray(self.spec.action_low, dtype=np.float32)
        self._high = np.asarray(self.spec.action_high, dtype=np.float32)
        # ONE env rate, feeding both how many steps a segment becomes and what
        # they contain. Two sources of truth here would desynchronise silently.
        self.env_fps = float(self.spec.env_fps if env_fps is None else env_fps)
        self.clock = SegmentClock(self.env_fps)
        self.reset()

    # ── contracts.Policy ──────────────────────────────────────────────────

    def reset(self) -> None:
        self._z: Tensor | None = None
        self._queue: list[np.ndarray] = []
        self.clock.reset()
        self.last_coeff: Tensor | None = None

    def act(self, obs: dict, instruction: str) -> np.ndarray:
        if not self._queue:
            n_env = self.clock.next_segment_len()           # how many steps
            seg = self._plan(obs, instruction)              # (H_OP, dof) @ 30 Hz
            resampled = to_env_rate(                        # what is in them
                seg, self.embodiment, n_env, src_fps=self.env_fps,
            )                                               # (n_env, dof) @ env_fps
            self._queue = [row for row in resampled]
        a = self._queue.pop(0)
        if self.clip_actions:
            a = np.clip(a, self._low, self._high)
        return np.asarray(a, dtype=np.float32)

    # ── mirrors stubs.StubPolicy so tests can treat them alike ────────────

    @property
    def replans(self) -> int:
        return self.clock.n_replans

    # ── inference ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def _plan(self, obs: dict, instruction: str) -> np.ndarray:
        m = self.modules
        feats = m.featurize(obs, instruction)
        feats = ObsFeats(**{k: _to_device(v, self.device) for k, v in feats.items()})

        z = _call(m.estimator, feats, self._z)                       # (1, K, D)
        self._z = z

        c = _argmax_coeff(m.proposal, z, feats["lang"], self.n_candidates)
        self.last_coeff = c

        a = _call(m.decoder, z, c)                                   # (1, H_OP, dof)
        a = a.detach().to(torch.float32).cpu().numpy()
        if a.ndim == 3:
            a = a[0]
        if a.shape != (H_OP, self.spec.dof):
            raise ValueError(
                f"decoder must emit ({H_OP}, {self.spec.dof}) at {FPS_CANONICAL} Hz "
                f"— one operator, never H_PLAN — got {a.shape}"
            )
        return a


def _to_device(v: Any, device: str) -> Any:
    return v.to(device) if isinstance(v, Tensor) else v


def _call(mod: Any, *args: Any) -> Tensor:
    """`nn.Module.__call__` when available, else the bare `forward`."""
    fn = mod if callable(mod) else getattr(mod, "forward")
    return fn(*args)


def _argmax_coeff(proposal: Any, z: Tensor, lang: Tensor, n_candidates: int) -> Tensor:
    """`c <- argmax pi_c(. | z, l)`, the R0 inference path. No search.

    Team E's `Proposal` exposes `argmax` (the unperturbed top-k of the logits,
    which is the mode of the Plackett-Luce support distribution). When it is
    absent — the stub path — fall back to the sampled-mode approximation:
    draw candidates and keep the highest log-probability one.

    **R3 (Phase 1B) replaces this function, and nothing else.** Notes from
    Team E, recorded here because this is where the swap happens:

    * `search.shooting()` returns `(c_root, info)`. Put `info["n_rejected"]`
      and `info["gate_exhausted"]` into `EpisodeResult.extra` — a flat 100%
      rejection rate is the diagnostic that `q_a` and `D_e` never converged
      into a shared coefficient space. `REALIZABILITY_TAU = 0.5` is provisional
      and is tight for 4-sparse simplex points (two random ones sit 0.7-1.0
      apart), so at R3 init the gate may reject nearly everything and fall
      through to rank 0.
    * Use `proposal.sample_seq(z, lang, n, depth) -> (B, N, DEPTH, M)` for the
      planner. PLAN 5 writes that signature for `sample`, which contradicts
      `contracts.Proposal.sample -> (B, n, M)`; `sample` is contract-exact and
      is not the planner entry point.
    """
    fn = getattr(proposal, "argmax", None)
    if callable(fn):
        return fn(z, lang)

    cand = proposal.sample(z, lang, n_candidates)                # (B, n, M)
    best, best_lp = None, None
    for j in range(cand.shape[1]):
        cj = cand[:, j]
        lp = proposal.log_prob(z, lang, cj)
        if best is None or float(lp.sum()) > best_lp:
            best, best_lp = cj, float(lp.sum())
    return best


# ═══════════════════════════════════════════════════════════════════════════
#  FACTORY  —  the ONLY place real modules are touched, and lazily
# ═══════════════════════════════════════════════════════════════════════════

def load_policy(
    ckpt: str | None = None,
    *,
    embodiment: str = "libero_franka",
    device: str = "cpu",
    allow_stub: bool = True,
    n_candidates: int = 16,
) -> LoomPolicy:
    """Build a `LoomPolicy` from a checkpoint, falling back to stubs.

    The imports of `loom.model` / `loom.heads` happen **inside this function**.
    That is what keeps `loom.eval` importable while Teams B/C/E are mid-flight,
    and what `tests/test_eval.py` enforces by reading the source.

    `ckpt=None`, or a repo where the real modules do not import, yields a
    stub-backed policy: the harness runs end to end and emits a correctly
    shaped table with random success rates, which is the Phase 1A deliverable
    (PLAN 4.F). Set `allow_stub=False` to make that an error instead.
    """
    spec = EMBODIMENTS[embodiment]
    modules, err = _try_real_modules(ckpt, embodiment, device)

    if modules is None:
        if not allow_stub:
            raise RuntimeError(f"real modules unavailable and allow_stub=False: {err}")
        modules = _stub_modules(embodiment, device, reason=str(err))

    modules.featurize = modules.featurize or default_featurizer(spec)
    return LoomPolicy(modules, n_candidates=n_candidates)


def _try_real_modules(
    ckpt: str | None, embodiment: str, device: str
) -> tuple[PolicyModules | None, Exception | None]:
    """Lazy import + checkpoint load. Any failure degrades to stubs."""
    if ckpt is None:
        return None, RuntimeError("no checkpoint given")
    try:
        from loom.heads.decoder import Decoder            # noqa: PLC0415  (lazy by design)
        from loom.heads.proposal import Proposal          # noqa: PLC0415
        from loom.model.estimator import Estimator        # noqa: PLC0415

        # `weights_only=False` and `map_location="cpu"`: eval may run under a
        # different interpreter than training, and CPU-first avoids dragging
        # saved RNG ByteTensors onto a GPU (CLAUDE.md gotcha).
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        state = payload.get("model", payload) if isinstance(payload, dict) else payload

        estimator, proposal, decoder = Estimator(), Proposal(), Decoder()
        for name, mod in (("estimator", estimator), ("proposal", proposal),
                          ("decoder", decoder)):
            sd = state.get(name) if isinstance(state, dict) else None
            if sd is not None:
                mod.load_state_dict(sd, strict=False)
            mod.eval().to(device)

        return PolicyModules(
            estimator=estimator,
            proposal=proposal,
            decoder=_bind_embodiment(decoder, embodiment),
            featurize=None,                              # filled by load_policy
            embodiment=embodiment,
            device=device,
            is_stub=False,
            meta={"ckpt": str(ckpt)},
        ), None
    except Exception as e:                               # noqa: BLE001 — degrade, never crash eval
        return None, e


def _bind_embodiment(decoder: Any, embodiment: str) -> Any:
    """Team C's `Decoder` is a `ModuleDict` dispatched by embodiment name."""
    try:
        body = decoder.body(embodiment)
        return body
    except Exception:                                    # noqa: BLE001
        return decoder


def _stub_modules(embodiment: str, device: str, reason: str = "") -> PolicyModules:
    import stubs                                          # noqa: PLC0415  (lazy by design)

    spec = EMBODIMENTS[embodiment]
    return PolicyModules(
        estimator=stubs.StubEstimator(),
        proposal=stubs.StubProposal(),
        decoder=stubs.StubDecoder(embodiment),
        featurize=default_featurizer(spec),
        embodiment=embodiment,
        device=device,
        is_stub=True,
        meta={"stub_reason": reason},
    )


def make_policy(ckpt: str | None = None, **kw: Any) -> LoomPolicy:
    """Alias used by `runner.py` and the CLI, so the seam has one name."""
    return load_policy(ckpt, **kw)
