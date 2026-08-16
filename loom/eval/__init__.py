"""LOOM — evaluation harness (Team F).

`loom.eval` depends on **nothing but `contracts.Policy`** (PLAN 4.F). No module
in this package may import `loom.model`, `loom.heads`, `loom.losses`,
`loom.data` or `loom.train` at module scope; the one place real modules are
touched is a lazy factory inside `loom.eval.policy`, which falls back to
`stubs` when they are unavailable. `tests/test_eval.py::test_eval_does_not_import_model_at_module_scope`
reads the source and enforces this.

This file holds only the bench-agnostic vocabulary — the evaluation protocol,
the per-episode record, and deterministic seeding — so that `libero`,
`robotwin` and `libero_plus` can share it without importing `runner` (which
imports them).
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field, replace
from typing import Any

__all__ = [
    "DEFAULT_LIBERO_SUITES",
    "EvalProtocol",
    "EpisodeResult",
    "episode_seed",
    "PROTOCOL_NOTE",
]


#: The four standard LIBERO suites, in the column order of PLAN 8.
#: `libero_long` is the same suite as `libero_10`.
DEFAULT_LIBERO_SUITES: tuple[str, ...] = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_long",
)

#: Why the default protocol is what it is. Emitted inline with every table.
PROTOCOL_NOTE = (
    "Light-WAM Table 1 does not state its episode count; PLAN 4.F therefore "
    "fixes the protocol at 10 episodes/task x 10 tasks x 4 suites over 3 seeds "
    "and requires it to be stated with every number. max_steps=512 is not "
    "invented either: it is LIBERO_MAX_STEPS_MAP, identical for every suite."
)


# ═══════════════════════════════════════════════════════════════════════════
#  PROTOCOL
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class EvalProtocol:
    """The evaluation protocol, logged with every result.

    PLAN 4.F: *"Replicate the evaluation protocol of the paper whose table we
    compare against (Light-WAM Table 1). Do not invent an episode count. If the
    protocol is unstated, use 10 episodes/task x 10 tasks x 4 suites over 3
    seeds and state it."* The protocol is never hardcoded in a loop — it is
    this object, it is written into the results JSON, and it is printed inline
    above every emitted table.

    `max_steps` is the per-episode environment step cap. Episodes that reach it
    without the env raising its success flag are failures, and the fraction
    that hit the cap is reported so a timeout-bound policy is distinguishable
    from a policy that fails outright. 512 is not invented either — it is
    `LIBERO_MAX_STEPS_MAP`, which is 512 for every suite.
    """

    bench:             str = "libero"
    episodes_per_task: int = 10
    n_tasks:           int = 10
    suites:            tuple[str, ...] = DEFAULT_LIBERO_SUITES
    seeds:             tuple[int, ...] = (0, 1, 2)
    max_steps:         int = 512
    notes:             str = PROTOCOL_NOTE

    def __post_init__(self) -> None:
        if self.episodes_per_task <= 0:
            raise ValueError(f"episodes_per_task must be positive, got {self.episodes_per_task}")
        if self.n_tasks <= 0:
            raise ValueError(f"n_tasks must be positive, got {self.n_tasks}")
        if not self.suites:
            raise ValueError("protocol needs at least one suite")
        if not self.seeds:
            raise ValueError("protocol needs at least one seed")
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")
        # tuples, not lists, so the protocol stays hashable and comparable
        object.__setattr__(self, "suites", tuple(self.suites))
        object.__setattr__(self, "seeds", tuple(int(s) for s in self.seeds))

    @property
    def total_episodes(self) -> int:
        return (
            self.episodes_per_task
            * self.n_tasks
            * len(self.suites)
            * len(self.seeds)
        )

    def describe(self) -> str:
        """One line, pasted above every table. Changing the protocol changes it."""
        seeds = ",".join(str(s) for s in self.seeds)
        return (
            f"{self.episodes_per_task} episodes/task x {self.n_tasks} tasks x "
            f"{len(self.suites)} suites over {len(self.seeds)} seeds "
            f"(seeds {seeds}), max {self.max_steps} env steps/episode "
            f"-> {self.total_episodes} episodes total"
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["suites"] = list(self.suites)
        d["seeds"] = list(self.seeds)
        d["total_episodes"] = self.total_episodes
        d["description"] = self.describe()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EvalProtocol":
        fields = {"bench", "episodes_per_task", "n_tasks", "suites", "seeds",
                  "max_steps", "notes"}
        kw = {k: v for k, v in d.items() if k in fields}
        if "suites" in kw:
            kw["suites"] = tuple(kw["suites"])
        if "seeds" in kw:
            kw["seeds"] = tuple(kw["seeds"])
        return cls(**kw)

    def replace(self, **kw: Any) -> "EvalProtocol":
        return replace(self, **kw)


# ═══════════════════════════════════════════════════════════════════════════
#  EPISODE RECORD
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EpisodeResult:
    """One episode. The unit of work, of resumption, and of aggregation.

    `success` follows LIBERO's own convention: the task counts as solved if the
    environment raises its success flag at **any** point in the episode, not
    only on the final step (see `loom.eval.libero.run_episode`).

    `extra` is the home for per-episode diagnostics that are not part of the
    score. At R3 it carries the shooting gate's `n_rejected` and
    `gate_exhausted` (see `loom.eval.policy._argmax_coeff`): a flat 100%
    rejection rate is the diagnostic that `q_a` and `D_e` never converged into
    a shared coefficient space, and it must be visible in the results JSON
    rather than inferred from a low score.
    """

    bench:        str
    suite:        str
    task_id:      int
    episode:      int
    seed:         int
    env_seed:     int
    success:      bool = False
    steps:        int = 0
    hit_step_cap: bool = False
    task_name:    str = ""
    n_replans:    int | None = None
    wall_s:       float = 0.0
    error:        str | None = None          # traceback of a crashed episode
    extra:        dict[str, Any] = field(default_factory=dict)

    def key(self) -> tuple[str, str, int, int, int]:
        """Identity for resumption. Stable across processes and restarts."""
        return (self.bench, self.suite, self.task_id, self.episode, self.seed)

    @staticmethod
    def key_of(bench: str, suite: str, task_id: int, episode: int, seed: int):
        return (bench, suite, int(task_id), int(episode), int(seed))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EpisodeResult":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


# ═══════════════════════════════════════════════════════════════════════════
#  DETERMINISTIC SEEDING
# ═══════════════════════════════════════════════════════════════════════════

def episode_seed(seed: int, bench: str, suite: str, task_id: int, episode: int) -> int:
    """Episode `k` of task `t` under protocol seed `s` always gets this env seed.

    `hash()` is salted per process and would give a different env to every
    worker, so this is a SHA-256 of the tuple instead. Stable across processes,
    machines and restarts.
    """
    raw = f"{bench}|{suite}|{int(task_id)}|{int(episode)}|{int(seed)}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big") & 0x7FFF_FFFF
