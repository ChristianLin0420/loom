"""LOOM — the R0 inference policy, and the rate conversion that protects the score.

    z <- E(o, l, z_prev)
    c <- argmax pi_c(. | z, l)
    a <- D_e(proprio_t, c)         # (H_OP, dof) at FPS_CANONICAL
    execute the segment, re-filter                                  (PLAN 4.F)

`D_e` takes the body's proprio and the coefficient — **not** the belief. See
`loom/heads/decoder.py`; the belief made `L_act` a behaviour-cloning objective
that put no pressure on `c` at all.

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

**Import discipline.** This module imports `contracts`, `numpy`, `torch`,
`loom.data.canonical` and `loom.data.tower` — nothing from `loom.model` /
`loom.heads` / `loom.train`, which carry the training stack. Both `loom.data`
imports are there for the same reason: they are the *single* implementation of a
transform that training also applies, and a second copy on this side is how eval
comes to disagree with training. `canonical` owns rate conversion; `tower` owns
image preprocessing and the frozen encoder. Both are numpy/torch-level at module
scope (`tower` imports `transformers` lazily, inside the loader), so both are
safe in the py3.10 LIBERO interpreter. `load_policy` imports the real model
modules lazily, inside the function, and falls back to `stubs` when they are
absent, so eval stays importable while the other teams are mid-flight.

**Interpreter discipline.** Eval will not run under the training venv: LIBERO
needs python 3.10 + robosuite 1.4 + numpy<2 in a separate conda env, while
training is python 3.13 + robosuite 1.5.2. Everything in `loom.eval` must
therefore stay importable with only `torch`, `numpy` and `contracts`/`stubs`.
Do not couple this package to a py3.13-only dependency, and load checkpoints
with `map_location="cpu"`.
"""

from __future__ import annotations

import hashlib
import inspect
import json
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
from loom.data.canonical import HOLD, action_semantics, to_env_rate

# The frozen SigLIP tower (Team I). Also `loom.data`, also numpy/torch-only at
# module scope — it imports `transformers` lazily, inside the loader — so this
# stays importable in the py3.10 LIBERO interpreter. It is the ONE encoder, used
# by `encode_to_cache` on the training side and by `default_featurizer` here, so
# train and eval cannot preprocess differently.
from loom.data.tower import (
    EVAL_VIEW_KEYS as TOWER_EVAL_VIEW_KEYS,
    FEAT_DIM,
    IMAGE_SIZE,
    LANG_LEN,
    N_PATCHES,
    TOWER_MODEL_ID,
    obs_featurizer,
)

__all__ = [
    "to_env_rate",
    "SegmentClock",
    "PolicyModules",
    "LoomPolicy",
    "load_policy",
    "make_policy",
    "VIEW_KEY_SOURCES",
    "view_keys_for",
    "default_featurizer",
    "zeros_featurizer",
    "feats_to",
    "submodule_state",
    "policy_provenance",
    "PLACEHOLDER_FEATURES",
    "GRIPPER_DWELL_OFF",
]

# Require this many consecutive segment-level polarity proposals before a HOLD
# channel reverses. One is exactly the original, ungated inference path.
GRIPPER_DWELL_OFF = 1


# ═══════════════════════════════════════════════════════════════════════════
#  FEATURE GEOMETRY
#
#  No longer a guess: read from `loom.data.tower`, which reads it from the
#  checkpoint. `zeros_featurizer` uses it so the stub path has exactly the
#  geometry the real cache has.
# ═══════════════════════════════════════════════════════════════════════════

PLACEHOLDER_FEATURES = {
    "patches": N_PATCHES,        # (IMAGE_SIZE // 14)^2
    "dim": FEAT_DIM,             # SigLIP-so400m width, both towers
    "lang_tokens": LANG_LEN,     # tokenizer.model_max_length
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

    which keeps `|n_replans * steps_per_segment - n_steps_dispatched| <= 1` for
    the whole episode. Measured over a full 512-step episode (96 segments) the
    maximum is exactly 1.0: repeated addition of `float(16/3)` accumulates a
    ~1e-13 deficit, so the 96th segment is still a 5 and the accumulator tips one
    segment later. That is 50 ms of phase over 25.6 s, against the 32 steps
    (1.6 s) that rounding each segment to 5 would cost. This mirrors
    `stubs.StubPolicy` step for step, including
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
        """`replans * steps_per_segment - dispatched`. Must stay in [-1, 1]."""
        return self.n_replans * self.steps_per_segment - self.n_steps_dispatched


# ═══════════════════════════════════════════════════════════════════════════
#  MODULE BUNDLE
# ═══════════════════════════════════════════════════════════════════════════

#: Where each embodiment's **eval-side** observation keys come from, in V order.
#:
#: `tower.obs_featurizer`'s own default is LIBERO's `("full_image",
#: "wrist_image")`, and that default silently applied to every body: calling
#: `default_featurizer(EMBODIMENTS["robotwin_aloha"])` asked a 4-view body for
#: two views and died with `n_views=4 but view_keys=('full_image',
#: 'wrist_image')`. The keys are owned by the adapter that also produced the
#: cache — one place, so the V axis eval feeds the estimator is by construction
#: the V axis training encoded. `None` means "the tower default", which is
#: LIBERO's pair and is correct *only* for LIBERO.
VIEW_KEY_SOURCES: dict[str, tuple[str, str] | None] = {
    "libero_franka": None,                       # tower.EVAL_VIEW_KEYS
    "robotwin_aloha": ("loom.data.adapters.robotwin", "EVAL_VIEW_KEYS"),
}


def view_keys_for(spec: EmbodimentSpec | str) -> tuple[str, ...]:
    """The observation keys `spec`'s featuriser reads, in the cache's V order.

    Never guesses. An unregistered embodiment whose `n_views` happens to match
    LIBERO's is allowed through on the tower default (the keys are then the only
    plausible ones); anything else raises, because the alternative is a featuriser
    reading the wrong cameras in the wrong order and a score that looks like a bad
    policy.
    """
    if isinstance(spec, str):
        spec = EMBODIMENTS[spec]
    src = VIEW_KEY_SOURCES.get(spec.name, ...)
    if src is not None and src is not ...:
        import importlib                                  # noqa: PLC0415

        module, attr = src
        keys = tuple(getattr(importlib.import_module(module), attr))
    else:
        keys = tuple(TOWER_EVAL_VIEW_KEYS)
        if src is ... and len(keys) != spec.n_views:
            raise KeyError(
                f"no eval view keys registered for embodiment {spec.name!r} "
                f"(n_views={spec.n_views}); add it to "
                f"loom.eval.policy.VIEW_KEY_SOURCES. The tower default "
                f"{keys} is LIBERO's and must not be applied to another body."
            )
    if len(keys) != spec.n_views:
        raise ValueError(
            f"{spec.name} has n_views={spec.n_views} but view_keys={keys}"
        )
    return keys


def default_featurizer(
    spec: EmbodimentSpec, *, device: str = "cpu", **kw: Any
) -> Callable[[dict, str], ObsFeats]:
    """`obs dict -> ObsFeats` through the **real frozen SigLIP tower**.

    A thin alias for `loom.data.tower.obs_featurizer`, which is the same object
    the cache builder encodes with. Two featurisers would be two preprocessing
    pipelines, and the second one is always the one that is subtly wrong.

    **`view_keys` is resolved per embodiment** (`view_keys_for`) rather than left
    to the tower's LIBERO default. It used not to be, and the consequence was
    that no RoboTwin policy could be constructed at all.

    Raises `tower.TowerUnavailable` when the checkpoint or `transformers` is
    missing. That is deliberate: a real evaluation must never silently fall back
    to zero features and report a chance-level score as a result. `load_policy`
    catches it and degrades to the explicitly-marked stub path instead.
    """
    kw.setdefault("view_keys", view_keys_for(spec))
    return obs_featurizer(spec, device=device, **kw)


def zeros_featurizer(spec: EmbodimentSpec) -> Callable[[dict, str], ObsFeats]:
    """Correctly-shaped zeros. **Stub path only.**

    A policy on this featuriser scores at chance by construction, so it is
    reachable only through `_stub_modules`, where `PolicyModules.is_stub` is
    True and the results JSON says so. It exists because the Phase 1A
    deliverable is the plumbing — a runnable harness with random success rates
    — and because the login node has no GPU to run an 878 M tower on.
    """
    p = PLACEHOLDER_FEATURES
    # content-free, so allocated once rather than per segment
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
    # Loaded only for opt-in operator-oracle / search paths.  R0 does not touch
    # this head, so keeping it optional avoids another ~30 M parameters in every
    # ordinary evaluation worker.
    q_action: Any = None


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
        op_stats: bool = False,
        gripper_dwell: int = GRIPPER_DWELL_OFF,
        decoder_samples: int = 1,
        duration_normalize_segments: bool = False,
    ) -> None:
        self.modules = modules
        self.embodiment = modules.embodiment
        self.spec = EMBODIMENTS[modules.embodiment]
        self.device = modules.device
        self.n_candidates = n_candidates
        self.clip_actions = clip_actions
        # Diagnostics only. Off by default; when off, `_plan` is byte-identical
        # to what it was and `op_stats_summary()` is empty, so a normal run puts
        # nothing extra in the results JSON.
        self.op_stats = bool(op_stats)
        self._low = np.asarray(self.spec.action_low, dtype=np.float32)
        self._high = np.asarray(self.spec.action_high, dtype=np.float32)
        # ONE env rate owns how many steps a segment becomes and the legacy
        # fixed-rate content mapping. The opt-in duration-normalized arm below
        # still takes its bin count exclusively from this clock.
        self.env_fps = float(self.spec.env_fps if env_fps is None else env_fps)
        self.clock = SegmentClock(self.env_fps)
        # Opt-in A/B arm. False takes the exact legacy call path in ``act``;
        # true fills every fractional-clock chunk from the complete 8-step
        # decoded segment instead of dropping its tail on a 5-step chunk.
        self.duration_normalize_segments = bool(duration_normalize_segments)
        self.decoder_samples = int(decoder_samples)
        if self.decoder_samples < 1:
            raise ValueError(f"decoder_samples must be >= 1, got {decoder_samples}")
        self.gripper_dwell = int(gripper_dwell)
        if self.gripper_dwell < 1:
            raise ValueError(
                f"gripper_dwell must be >= 1 ({GRIPPER_DWELL_OFF} = off), "
                f"got {gripper_dwell}"
            )
        kinds = action_semantics(self.embodiment)
        self._hold_dims = tuple(d for d, kind in enumerate(kinds) if kind == HOLD)
        self._hold_mid = np.asarray(
            [(self._low[d] + self._high[d]) / 2.0 for d in self._hold_dims],
            dtype=np.float32,
        )
        self._policy_seed: int | None = None
        self._decoder_accepts_generator = _accepts_keyword(modules.decoder, "generator")
        self.reset()

    # ── contracts.Policy ──────────────────────────────────────────────────

    def reset(self) -> None:
        self._z: Tensor | None = None
        self._queue: list[np.ndarray] = []
        self.clock.reset()
        self.last_coeff: Tensor | None = None
        self._op_log: list[dict[str, Any]] = []
        n_hold = len(self._hold_dims)
        self._latch: np.ndarray | None = None
        self._prev_prop = np.zeros(n_hold, dtype=np.float32)
        self._dwell = np.zeros(n_hold, dtype=np.int64)
        self._n_prop_flip = 0
        self._n_exec_flip = 0
        self._n_suppressed = 0
        self._decoder_generator: torch.Generator | None = None
        if self._policy_seed is not None:
            self._decoder_generator = torch.Generator(device=self.device)
            self._decoder_generator.manual_seed(self._policy_seed)

    def set_policy_seed(self, seed: int) -> None:
        """Select and reset this episode's private decoder-noise stream.

        The runner calls this before the benchmark calls ``reset()``.  Keeping
        the RNG on the policy avoids perturbing torch's process-global stream,
        and recreating it in ``reset`` makes an episode reproducible regardless
        of worker assignment or prior episodes in that worker.
        """
        self._policy_seed = int(seed)
        self._decoder_generator = torch.Generator(device=self.device)
        self._decoder_generator.manual_seed(self._policy_seed)

    def act(self, obs: dict, instruction: str) -> np.ndarray:
        if not self._queue:
            n_env = self.clock.next_segment_len()           # how many steps
            seg = self._plan(obs, instruction)              # (H_OP, dof) @ 30 Hz
            seg = self._gate_gripper(seg)                   # no-op when k=1
            if self.duration_normalize_segments:
                resampled = to_env_rate(                    # what is in them
                    seg, self.embodiment, n_env, src_fps=self.env_fps,
                    duration_normalized=True,
                )
            else:
                # Keep the default path byte-for-byte compatible with the
                # pre-A/B policy, including the historical call signature.
                resampled = to_env_rate(
                    seg, self.embodiment, n_env, src_fps=self.env_fps,
                )
            self._queue = [row for row in resampled]
        a = self._queue.pop(0)
        if self.clip_actions:
            a = np.clip(a, self._low, self._high)
        return np.asarray(a, dtype=np.float32)

    # ── mirrors stubs.StubPolicy so tests can treat them alike ────────────

    @property
    def replans(self) -> int:
        return self.clock.n_replans

    def _gate_gripper(self, seg: np.ndarray) -> np.ndarray:
        """Debounce HOLD-channel polarity reversals across operator replans.

        A segment proposes the side of the action-range midpoint containing its
        mean HOLD value. A reversal executes only after ``gripper_dwell``
        consecutive proposals; until then values are reflected about the
        midpoint, preserving magnitude and changing only polarity.
        """
        if not self._hold_dims:
            return seg
        seg = np.array(seg, dtype=np.float32, copy=True)
        for j, d in enumerate(self._hold_dims):
            mid = float(self._hold_mid[j])
            proposed = 1.0 if float(seg[:, d].mean()) >= mid else -1.0
            if self._latch is None:
                self._latch = np.ones(len(self._hold_dims), dtype=np.float32)
            if self._prev_prop[j] != 0.0 and proposed != self._prev_prop[j]:
                self._n_prop_flip += 1
            self._prev_prop[j] = proposed
            if self.clock.n_replans <= 1:
                self._latch[j] = proposed
                continue
            if proposed == self._latch[j]:
                self._dwell[j] = 0
                continue
            self._dwell[j] += 1
            if self._dwell[j] >= self.gripper_dwell:
                self._latch[j] = proposed
                self._dwell[j] = 0
                self._n_exec_flip += 1
            else:
                seg[:, d] = 2.0 * mid - seg[:, d]
                self._n_suppressed += 1
        return seg

    def gripper_summary(self) -> dict[str, Any]:
        if not self._hold_dims:
            return {}
        return {
            "grip_dwell_k": self.gripper_dwell,
            "grip_hold_dims": list(self._hold_dims),
            "grip_prop_flips": self._n_prop_flip,
            "grip_exec_flips": self._n_exec_flip,
            "grip_suppressed": self._n_suppressed,
        }

    # ── inference ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def _plan(self, obs: dict, instruction: str) -> np.ndarray:
        m = self.modules
        feats = m.featurize(obs, instruction)
        feats = feats_to(feats, self.device, _module_dtype(m.estimator))

        z = _call(m.estimator, feats, self._z)                       # (1, K, D)
        self._z = z

        c = _argmax_coeff(m.proposal, z, feats["lang"], self.n_candidates)
        self.last_coeff = c
        if self.op_stats:
            self._log_operator(m.proposal, z, feats["lang"], c)

        # D_e(proprio_t, c), NOT D_e(z, c). The belief is deliberately not an
        # input to the realizer -- with it the decoder is a behaviour-cloning
        # head and `c` is decorative (loom/heads/decoder.py). `feats["proprio"]`
        # is (1, dof_e) here, already on the estimator's device and dtype via
        # `feats_to` above, and it is the same quantity training reads out of
        # `window["feats"][h]["proprio"]`.
        decoder_kw = ({"generator": self._decoder_generator}
                      if self._decoder_accepts_generator else {})
        proprio = feats["proprio"]
        if self.decoder_samples > 1:
            proprio = proprio.expand(self.decoder_samples, *proprio.shape[1:])
            c = c.expand(self.decoder_samples, *c.shape[1:])
        a = _call(m.decoder, proprio, c, **decoder_kw)               # (S, H_OP, dof)
        if self.decoder_samples > 1:
            a = a.mean(dim=0, keepdim=True)
        a = a.detach().to(torch.float32).cpu().numpy()
        if a.ndim == 3:
            a = a[0]
        if a.shape != (H_OP, self.spec.dof):
            raise ValueError(
                f"decoder must emit ({H_OP}, {self.spec.dof}) at {FPS_CANONICAL} Hz "
                f"— one operator, never H_PLAN — got {a.shape}"
            )
        return a

    # ── diagnostics · WHICH operator was chosen, not whether it worked ─────
    #
    # A `pi_c` sitting at its uniform Plackett-Luce baseline makes
    # `c <- argmax pi_c` an effectively random draw over the M operators, and
    # the only symptom in the score is a zero that looks exactly like an
    # undertrained decoder. These two methods make the difference visible: the
    # selected index per replan, how often it changes, and how peaked the
    # logits are. Opt-in (`op_stats=True`), never on the scoring path.

    def _log_operator(self, proposal: Any, z: Tensor, lang: Tensor, c: Tensor) -> None:
        """One row per replan. `proposal.logits` is a second, diagnostic-only call.

        Deliberately not folded into `_argmax_coeff`: the inference path must be
        identical with and without diagnostics, and a ~50 M head re-run 96 times
        an episode is noise against an 878 M tower plus MuJoCo.
        """
        row: dict[str, Any] = {}
        cv = c.detach().to(torch.float32).reshape(-1)
        top = torch.topk(cv, min(int(cv.numel()), 4))
        row["m"] = int(cv.numel())
        row["top1"] = int(top.indices[0])
        row["support"] = [int(i) for i in top.indices]
        row["w1"] = float(top.values[0])
        fn = getattr(proposal, "logits", None)
        if callable(fn):
            lg = fn(z, lang).detach().to(torch.float32).reshape(-1)
            p = torch.softmax(lg, dim=-1)
            row["ent"] = float(-(p * p.clamp_min(1e-12).log()).sum())
            two = torch.topk(lg, 2).values
            row["gap"] = float(two[0] - two[1])
        self._op_log.append(row)

    def op_stats_summary(self) -> dict[str, Any]:
        """Per-episode operator-selection summary, for `EpisodeResult.extra`.

        Empty when `op_stats` is off, so `run_episode` adds nothing to a normal
        results JSON. `op_top1` is kept in full — whether the selected operator
        changes *between replans within one episode* is the question, and a mean
        cannot answer it.
        """
        log = self._op_log
        if not log:
            return self.gripper_summary()
        top1 = [r["top1"] for r in log]
        support = {i for r in log for i in r["support"]}
        # The *set* the top-4 picks out, not just which indices ever appear in
        # one. `op_support_unique` counts operators; two replans on supports
        # {1,2,3,4} and {1,2,3,5} share three of them and are still different
        # actions. Counted as sorted "a-b-c-d" keys so the run-wide number of
        # distinct supports is recoverable from the results JSON without
        # keeping every replan's tuple.
        support_counts: dict[str, int] = {}
        for r in log:
            k = "-".join(str(i) for i in sorted(r["support"]))
            support_counts[k] = support_counts.get(k, 0) + 1
        ents = [r["ent"] for r in log if "ent" in r]
        gaps = [r["gap"] for r in log if "gap" in r]
        out: dict[str, Any] = {
            "op_m": log[0]["m"],
            "op_n_replans": len(log),
            "op_top1": top1,
            "op_top1_unique": len(set(top1)),
            "op_top1_switches": sum(a != b for a, b in zip(top1, top1[1:])),
            "op_support_unique": len(support),
            "op_support_counts": support_counts,
            "op_support_set_unique": len(support_counts),
            "op_w1_mean": round(sum(r["w1"] for r in log) / len(log), 5),
        }
        if ents:
            out["op_ent_mean"] = round(sum(ents) / len(ents), 5)
            out["op_ent_min"] = round(min(ents), 5)
            out["op_ent_max"] = round(max(ents), 5)
            out["op_ent_head"] = [round(e, 5) for e in ents[:5]]
        if gaps:
            out["op_logit_gap_mean"] = round(sum(gaps) / len(gaps), 5)
        out.update(self.gripper_summary())
        return out


def _to_device(v: Any, device: str) -> Any:
    return v.to(device) if isinstance(v, Tensor) else v


def _module_dtype(mod: Any) -> torch.dtype | None:
    """The dtype of a module's first parameter, or None for a stub."""
    params = getattr(mod, "parameters", None)
    if not callable(params):
        return None
    for p in params():
        return p.dtype
    return None


def feats_to(feats: ObsFeats, device: str, dtype: torch.dtype | None = None) -> ObsFeats:
    """Move an `ObsFeats` onto the estimator's device **and dtype**.

    The frozen tower emits bf16 (PLAN §9) and reads proprio as float32, exactly
    as the fp16 feature cache hands them to training. Training then runs under
    `torch.autocast("cuda", bfloat16)`, which reconciles the two silently; eval
    has no autocast, so a float32 checkpoint meeting bf16 features is a hard
    `mat1 and mat2 must have the same dtype`. This is where the two meet, and it
    is the only place that should have an opinion about it. `dtype=None` (a
    stub, which has no parameters) leaves everything alone.
    """
    out = {}
    for k, v in feats.items():
        if not isinstance(v, Tensor):
            out[k] = v
        elif dtype is not None and v.is_floating_point():
            out[k] = v.to(device=device, dtype=dtype)
        else:
            out[k] = v.to(device)
    return ObsFeats(**out)


def _call(mod: Any, *args: Any, **kw: Any) -> Tensor:
    """`nn.Module.__call__` when available, else the bare `forward`."""
    fn = mod if callable(mod) else getattr(mod, "forward")
    return fn(*args, **kw)


def _accepts_keyword(mod: Any, name: str) -> bool:
    """Whether a module's forward surface accepts ``name``.

    Stub and test decoders predate the optional generator argument.  Inspecting
    ``forward`` (rather than ``nn.Module.__call__``, whose signature is generic)
    preserves those injected-policy seams without catching and masking a real
    TypeError raised inside a decoder.
    """
    fn = getattr(mod, "forward", mod)
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == name or p.kind == inspect.Parameter.VAR_KEYWORD
               for p in params)


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
    allow_stub: bool | None = None,
    n_candidates: int = 16,
    op_stats: bool = False,
    gripper_dwell: int = GRIPPER_DWELL_OFF,
    decoder_samples: int = 1,
    duration_normalize_segments: bool = False,
    _include_q_action: bool = False,
) -> LoomPolicy:
    """Build a `LoomPolicy` from a checkpoint, falling back to stubs.

    The imports of `loom.model` / `loom.heads` happen **inside this function**.
    That is what keeps `loom.eval` importable while Teams B/C/E are mid-flight,
    and what `tests/test_eval.py` enforces by reading the source.

    `ckpt=None` yields a stub-backed policy: the harness runs end to end and
    emits a correctly shaped table with random success rates, which is the
    Phase 1A deliverable (PLAN 4.F).

    **`allow_stub` defaults to `ckpt is None`.** If a checkpoint was named, a
    failure to load it — bad path, missing tower, key layout the loader does not
    understand — raises instead of quietly substituting zero features and stub
    modules. That fallback scores ~0 and is indistinguishable from an untrained
    model, which is precisely the confusion that would poison a real evaluation.
    Pass `allow_stub=True` explicitly to opt back into it.
    """
    if allow_stub is None:
        allow_stub = ckpt is None
    spec = EMBODIMENTS[embodiment]
    modules, err = _try_real_modules(
        ckpt, embodiment, device, include_q_action=_include_q_action,
    )

    if modules is None:
        if not allow_stub:
            raise RuntimeError(f"real modules unavailable and allow_stub=False: {err}")
        modules = _stub_modules(embodiment, device, reason=str(err))

    if modules.featurize is None:
        modules.featurize = (zeros_featurizer(spec) if modules.is_stub
                             else default_featurizer(spec, device=device))
    return LoomPolicy(
        modules, n_candidates=n_candidates, op_stats=op_stats,
        gripper_dwell=gripper_dwell, decoder_samples=decoder_samples,
        duration_normalize_segments=duration_normalize_segments,
    )


def _resolved_config_hash(cfg: dict[str, Any]) -> str:
    """Validate embedded provenance without importing the training stack."""
    experiment = {k: v for k, v in cfg.items() if k != "link"}
    return hashlib.blake2b(
        json.dumps(experiment, sort_keys=True, default=str).encode(), digest_size=8,
    ).hexdigest()


def _embedded_model_kwargs(payload: Any, module: str) -> dict | None:
    """Authenticated model kwargs, or ``None`` for a legacy checkpoint."""
    if not isinstance(payload, dict) or "resolved_config" not in payload:
        return None
    cfg = payload["resolved_config"]
    if not isinstance(cfg, dict):
        raise RuntimeError("checkpoint resolved_config is not a mapping")
    expected = str(payload.get("config_hash", ""))
    if not expected:
        raise RuntimeError(
            "checkpoint embeds resolved_config without the config_hash that "
            "authenticates it"
        )
    got = _resolved_config_hash(cfg)
    if got != expected:
        raise RuntimeError(
            f"checkpoint resolved_config hash {got} does not match saved "
            f"config_hash {expected}; refusing mutable architecture provenance"
        )
    model = cfg.get("model", {})
    if not isinstance(model, dict):
        raise RuntimeError("checkpoint resolved_config.model is not a mapping")
    kw = model.get(module, {}) or {}
    if not isinstance(kw, dict):
        raise RuntimeError(f"checkpoint resolved_config.model.{module} is not a mapping")
    return dict(kw)


def _run_model_kwargs(ckpt: str | Any, module: str, payload: Any = None) -> dict:
    """Architecture kwargs for `module`, read from the run that produced `ckpt`.

    New consolidated checkpoints embed the resolved, config-hash-authenticated
    experiment config. It is the source of truth even if an adjacent
    ``config.json`` is later edited. Older checkpoints fall back to
    ``runs/<run>/config.json`` -> ``model.<module>``; a consolidated checkpoint
    lives in ``runs/<run>_eval/``, so try that sibling first, then the
    checkpoint's own directory and its parent.

    Returns {} when nothing is found, which reproduces the shipped defaults --
    correct for every checkpoint trained before these flags existed.
    """
    from pathlib import Path as _Path                        # noqa: PLC0415

    if payload is None:
        try:
            payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        except Exception:                                    # noqa: BLE001
            # Legacy tests and raw/sharded checkpoints may not be loadable here;
            # their adjacent config fallback remains supported.
            payload = None
    embedded = _embedded_model_kwargs(payload, module)
    if embedded is not None:
        return embedded

    p = _Path(str(ckpt)).resolve()
    d = p.parent
    cands = []
    if d.name.endswith("_eval"):
        cands.append(d.parent / d.name[: -len("_eval")] / "config.json")
    cands += [d / "config.json", d.parent / "config.json"]
    for c in cands:
        try:
            if c.is_file():
                cfg = json.loads(c.read_text())
                kw = ((cfg or {}).get("model") or {}).get(module) or {}
                if isinstance(kw, dict):
                    return dict(kw)
        except (OSError, ValueError):
            continue
    return {}


def submodule_state(state: Any, name: str) -> dict[str, Tensor] | None:
    """One submodule's `state_dict` out of a training checkpoint. Both layouts.

    `loom.train.ckpt.build_state` stores ``payload["model"] =
    LoomModel.state_dict()``, which is **flat and dotted** —
    ``estimator.latents``, ``proposal.query``,
    ``decoder.inner.bodies.libero_franka.step_emb`` — not a dict of per-module
    state dicts. `state.get("estimator")` on that returns `None`.

    That is the bug this function exists to close, and it is the worst kind:
    eval built `Estimator()`, loaded nothing into it, and evaluated **randomly
    initialised weights** with no error, no warning and a plausible-looking
    near-zero score. Verified against `runs/r0a_smoke/ckpt_000000030_rank0.pt`:
    929 flat keys, zero of which `state.get(name)` reaches.

    The ``inner.`` hop is `loom.train.loop.EmbodimentHeads`, which wraps Team
    C's per-embodiment container (`q_action`, `decoder`) so the loop can write
    ``heads[embodiment]``. Eval instantiates the container directly, so that
    level has to come off.

    Returns `None` when the name is absent entirely — the caller raises rather
    than proceeding with random weights.
    """
    if not isinstance(state, dict):
        return None
    sub = state.get(name)
    if isinstance(sub, dict) and sub:
        return sub                                       # nested layout
    prefix = name + "."
    flat = {k[len(prefix):]: v for k, v in state.items()
            if isinstance(k, str) and k.startswith(prefix)}
    if not flat:
        return None
    if all(k.startswith("inner.") for k in flat):         # EmbodimentHeads
        flat = {k[len("inner."):]: v for k, v in flat.items()}
    return flat


def _try_real_modules(
    ckpt: str | None, embodiment: str, device: str, *,
    include_q_action: bool = False,
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
        try:
            payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        except RuntimeError as e:
            # A raw per-rank FSDP shard. `sharded_state_dict` saves
            # SHARDED_STATE_DICT, so ckpt_*_rank*.pt hold ShardedTensor objects
            # that will not even unpickle without a process group -- and the
            # bare torch message reads like a distributed-setup bug in eval.
            if "process group" in str(e) or "ShardedTensor" in str(e):
                raise RuntimeError(
                    f"{ckpt} is a per-rank FSDP shard, not a whole model. "
                    "Reassemble the run's shards first:\n"
                    "    python -m loom.train.consolidate --run_dir <run> --pin\n"
                    "and pass the consolidated file to --ckpt. (--pin hardlinks "
                    "the shards out of the way of ckpt._prune, which keeps only "
                    "the last 3 steps.)"
                ) from e
            raise
        state = payload.get("model", payload) if isinstance(payload, dict) else payload

        # Only the body under evaluation. Building every registered embodiment
        # would make a body the checkpoint has never seen look like a *missing*
        # key, and the missing-key check below is the whole guard.
        #
        # The estimator's architecture flags must come from the RUN, not from
        # this file's defaults. `z_prev_residual` is not a parameter, so a
        # checkpoint trained with it off produces NO missing and NO unexpected
        # key -- eval would silently score a different model than the one that
        # trained, and the per-module guard below cannot see it. (`learned_z_init`
        # at least shows up as an unexpected `z_init`.) Measured: that difference
        # is worth 1.0 vs 18.0 LIBERO avg between two arms of the same run.
        est_kw = _run_model_kwargs(ckpt, "estimator", payload)
        if est_kw:
            print(f"[policy] estimator kwargs from checkpoint config: {est_kw}", flush=True)
        estimator = Estimator(embodiments=[embodiment], **est_kw)
        prop_kw = _run_model_kwargs(ckpt, "proposal", payload)
        if prop_kw:
            print(f"[policy] proposal kwargs from checkpoint config: {prop_kw}", flush=True)
        proposal = Proposal(**prop_kw)
        # Same argument for the decoder, and it is sharper here: `residual` is
        # not a parameter either, so a body trained on the proprio-relative
        # target and rebuilt without the flag loads with 0 missing and 0
        # unexpected keys and then emits ~0.03 rad residuals as ABSOLUTE joint
        # targets. Half of that change is far worse than none.
        dec_kw = _run_model_kwargs(ckpt, "decoder", payload)
        if dec_kw:
            print(f"[policy] decoder kwargs from checkpoint config: {dec_kw}", flush=True)
        decoder = Decoder(embodiments=[embodiment], default_embodiment=embodiment,
                          **dec_kw)

        q_action = None
        if include_q_action:
            from loom.heads.q_action import QAction        # noqa: PLC0415

            qa_kw = _run_model_kwargs(ckpt, "q_action", payload)
            if qa_kw:
                print(f"[policy] q_action kwargs from checkpoint config: {qa_kw}",
                      flush=True)
            q_action = QAction(
                embodiments=[embodiment], default_embodiment=embodiment, **qa_kw,
            )

        loaded: dict[str, Any] = {}
        wanted = [("estimator", estimator), ("proposal", proposal),
                  ("decoder", decoder)]
        if q_action is not None:
            wanted.append(("q_action", q_action))
        for name, mod in wanted:
            sd = submodule_state(state, name)
            if sd is None:
                raise KeyError(
                    f"{ckpt} has no weights for {name!r}. Keys look like "
                    f"{sorted(state)[:4] if isinstance(state, dict) else type(state)}. "
                    f"Refusing to evaluate a randomly initialised {name}."
                )
            incompatible = mod.load_state_dict(sd, strict=False)
            missing = list(getattr(incompatible, "missing_keys", []) or [])
            unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
            if missing:
                raise RuntimeError(
                    f"{ckpt}: {len(missing)} parameters of {name} are not in the "
                    f"checkpoint (e.g. {missing[:3]}). A partly-loaded module is a "
                    f"partly-random module; refusing to score it."
                )
            loaded[name] = {"tensors_loaded": len(sd) - len(unexpected),
                            "unexpected": len(unexpected)}
            mod.eval().to(device)
        if include_q_action:
            # The operator oracle is only eligible on the promoted bank-stage
            # checkpoint.  It does not execute the bank, but absence of the
            # trained bank state means this is not that artifact.
            bank_sd = submodule_state(state, "bank")
            if bank_sd is None:
                raise KeyError(
                    f"{ckpt} has no operator-bank weights; refusing to label it "
                    "as the promoted bank-stage oracle checkpoint"
                )
            loaded["bank"] = {"tensors_present": len(bank_sd), "loaded": False}

        # The frozen tower is part of "real modules": if the checkpoint loaded
        # but the tower cannot, this whole path must fail and degrade to the
        # stub rather than run a trained policy on zero features.
        featurize = default_featurizer(EMBODIMENTS[embodiment], device=device)

        return PolicyModules(
            estimator=estimator,
            proposal=proposal,
            decoder=_bind_embodiment(decoder, embodiment),
            q_action=(_bind_embodiment(q_action, embodiment)
                      if q_action is not None else None),
            featurize=featurize,
            embodiment=embodiment,
            device=device,
            is_stub=False,
            meta={"ckpt": str(ckpt), "tower": TOWER_MODEL_ID,
                  "tower_image_size": IMAGE_SIZE,
                  "featurizer": "loom.data.tower.obs_featurizer",
                  "view_keys": list(view_keys_for(EMBODIMENTS[embodiment])),
                  "ckpt_global_step": (payload.get("global_step")
                                       if isinstance(payload, dict) else None),
                  "ckpt_config_hash": (payload.get("config_hash")
                                        if isinstance(payload, dict) else None),
                  "state_dict": loaded},
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
        featurize=zeros_featurizer(spec),
        embodiment=embodiment,
        device=device,
        is_stub=True,
        meta={"stub_reason": reason,
              "featurizer": "loom.eval.policy.zeros_featurizer (ZEROS, not the tower)"},
    )


def make_policy(ckpt: str | None = None, **kw: Any) -> LoomPolicy:
    """Alias used by `runner.py` and the CLI, so the seam has one name."""
    oracle = kw.pop("operator_oracle", None)
    if oracle is not None:
        # Lazy for the same reason as the model imports above: the standard R0
        # evaluator must not import or construct oracle-only machinery.
        from loom.eval.operator_oracle import load_operator_oracle_policy  # noqa: PLC0415

        return load_operator_oracle_policy(ckpt, oracle=oracle, **kw)
    return load_policy(ckpt, **kw)


def policy_provenance(policy: Any) -> dict[str, Any]:
    """What actually ran, for the results JSON. Never inferred from the score.

    `is_stub` is the field that matters: a run that asked for `--ckpt` and got
    stubs (bad path, missing tower, unreadable checkpoint) scores ~0 and is
    indistinguishable from an untrained model unless this is written down.
    """
    m = getattr(policy, "modules", None)
    if m is None:
        return {"is_stub": None, "policy": type(policy).__name__}
    out: dict[str, Any] = {
        "policy": type(policy).__name__,
        "is_stub": bool(m.is_stub),
        "embodiment": m.embodiment,
        "device": str(m.device),
        "env_fps": getattr(policy, "env_fps", None),
        "env_steps_per_segment": getattr(getattr(policy, "clock", None),
                                         "steps_per_segment", None),
        "gripper_dwell": getattr(policy, "gripper_dwell", None),
        "decoder_samples": getattr(policy, "decoder_samples", None),
        "duration_normalize_segments": getattr(
            policy, "duration_normalize_segments", None,
        ),
        "h_op": H_OP,
        "fps_canonical": FPS_CANONICAL,
        "resampler": f"{to_env_rate.__module__}.{to_env_rate.__name__}",
    }
    out.update(m.meta or {})
    return out
