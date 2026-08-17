"""LOOM — the evaluation runner.

    python -m loom.eval --bench libero --ckpt <path> --out results.json

Four properties, in the order they matter:

* **Deterministic.** Episode `k` of task `t` under protocol seed `s` always
  gets the same env seed, computed by `episode_seed` from a SHA-256 of the
  tuple. `hash()` is salted per process and would hand a different scene to
  every worker.
* **Robust.** One crashed episode is recorded as a failure with its traceback
  in the results JSON; the run continues. Only `BaseException`
  (KeyboardInterrupt, SIGTERM-as-SystemExit) stops a run, and even then the
  results file has already been flushed.
* **Resumable.** The results JSON is rewritten atomically after every episode.
  Restarting with the same `--out` skips everything already recorded.
* **Parallel.** Work is sharded across visible GPUs, one worker process per
  device, each building its own policy once. On a CPU-only login node this
  degrades to a single in-process loop, which is also what the tests use.

Imports `contracts`, `numpy`, `torch` and its own package. Nothing from
`loom.model` / `loom.heads` / `loom.train`.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import torch

from loom.eval import EpisodeResult, EvalProtocol, episode_seed

__all__ = [
    "WorkItem", "iter_work", "shard", "n_devices", "seed_fn_for",
    "ResultStore", "aggregate", "run_eval", "bench_module", "ensure_runtime",
    "claim_device",
]

RESULTS_VERSION = 1


# ═══════════════════════════════════════════════════════════════════════════
#  WORK
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class WorkItem:
    bench:    str
    suite:    str
    task_id:  int
    episode:  int
    seed:     int
    env_seed: int
    max_steps: int

    def key(self) -> tuple[str, str, int, int, int]:
        return EpisodeResult.key_of(self.bench, self.suite, self.task_id,
                                    self.episode, self.seed)

    def to_dict(self) -> dict[str, Any]:
        return dict(bench=self.bench, suite=self.suite, task_id=self.task_id,
                    episode=self.episode, seed=self.seed, env_seed=self.env_seed,
                    max_steps=self.max_steps)


def seed_fn_for(bench: str):
    """The bench's own env-seed rule, or the SHA-256 default.

    LIBERO has no seed convention of its own, so `episode_seed`'s hash of the
    tuple is as good as any. RoboTwin does: its eval script starts at
    `st_seed = 100000 * (1 + seed)` and walks forward, and a released `seed.txt`
    is the list of seeds that survived that walk — so reproducing the origin is
    part of reproducing the protocol. A bench module that defines `episode_seed`
    gets to say.
    """
    try:
        return getattr(bench_module(bench), "episode_seed", episode_seed)
    except Exception:                                    # noqa: BLE001
        return episode_seed


def iter_work(protocol: EvalProtocol) -> list[WorkItem]:
    """Every (suite, task, episode, seed) the protocol asks for, in a fixed order.

    Nothing in this function is hardcoded — the counts come from the protocol
    object, which is also what gets written into the results JSON.
    """
    items: list[WorkItem] = []
    env_seed_of = seed_fn_for(protocol.bench)
    for seed in protocol.seeds:
        for suite in protocol.suites:
            for task_id in range(protocol.n_tasks):
                for ep in range(protocol.episodes_per_task):
                    items.append(WorkItem(
                        bench=protocol.bench,
                        suite=suite,
                        task_id=task_id,
                        episode=ep,
                        seed=seed,
                        env_seed=env_seed_of(seed, protocol.bench, suite, task_id, ep),
                        max_steps=protocol.max_steps,
                    ))
    return items


def shard(items: Sequence[WorkItem], n: int) -> list[list[WorkItem]]:
    """Round-robin so every shard sees a mix of suites, not one suite each."""
    n = max(1, int(n))
    out: list[list[WorkItem]] = [[] for _ in range(n)]
    for i, it in enumerate(items):
        out[i % n].append(it)
    return out


def n_devices() -> int:
    """Visible CUDA devices, or 1 on a CPU-only node."""
    try:
        return max(1, torch.cuda.device_count() if torch.cuda.is_available() else 1)
    except Exception:                                    # noqa: BLE001
        return 1


def bench_module(bench: str):
    """`--bench` -> the module that owns `make_env` / `run_episode`."""
    import importlib                                     # noqa: PLC0415

    if bench not in ("libero", "robotwin", "libero_plus"):
        raise ValueError(f"unknown bench {bench!r}")
    return importlib.import_module(f"loom.eval.{bench}")


#: Per-bench one-time process setup, run before any env is built. LIBERO needs
#: the headless render vars and the `torch.load` shim for its init states;
#: RoboTwin needs the Vulkan ICD, the chdir into its checkout and the SAPIEN
#: render-device pin. Both are idempotent and both report a status string, which
#: `_run_item` carries into a failed episode's record so a run that died at
#: reset says why instead of leaving 1200 identical tracebacks.
_RUNTIME_SETUP = ("ensure_libero_runtime", "ensure_robotwin_runtime")
_RUNTIME_STATUS = ("LIBERO_RUNTIME_STATUS", "ROBOTWIN_RUNTIME_STATUS")


def ensure_runtime(mod: Any) -> str | None:
    for name in _RUNTIME_SETUP:
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn()
    return None


def runtime_status(mod: Any) -> str | None:
    for name in _RUNTIME_STATUS:
        if hasattr(mod, name):
            return getattr(mod, name)
    return None


def env_available(mod: Any) -> bool:
    """Did the bench's REAL simulator import? Never raises."""
    for name in ("libero_available", "robotwin_available"):
        fn = getattr(mod, name, None)
        if callable(fn):
            try:
                return bool(fn())
            except Exception:                            # noqa: BLE001
                return False
    return False


# ═══════════════════════════════════════════════════════════════════════════
#  RESULT STORE  —  incremental, atomic, resumable
# ═══════════════════════════════════════════════════════════════════════════

class ResultStore:
    """Holds every `EpisodeResult` and rewrites the JSON after each one.

    Rewriting the whole file is deliberate: at 1200 episodes it costs
    milliseconds, and it keeps the file a single valid JSON document at every
    instant, which a JSONL sidecar plus a merge step would not.
    """

    def __init__(
        self,
        path: str | os.PathLike | None,
        protocol: EvalProtocol,
        *,
        resume: bool = True,
        meta: dict[str, Any] | None = None,
    ) -> None:
        # Absolute, always. RoboTwin's env setup `chdir`s into its own checkout
        # (its configs store relative asset paths), and a single-worker run does
        # that in *this* process — so a relative `--out` would land inside the
        # simulator's source tree halfway through the run.
        self.path = Path(path).resolve() if path is not None else None
        self.protocol = protocol
        self.meta = dict(meta or {})
        self.records: dict[tuple, EpisodeResult] = {}
        self.n_resumed = 0
        if resume and self.path is not None and self.path.exists():
            self._load()

    def _load(self) -> None:
        try:
            blob = json.loads(self.path.read_text())
        except Exception as e:                           # noqa: BLE001
            raise ValueError(
                f"{self.path} exists but is not readable JSON ({e}); "
                f"delete it or pass --no-resume"
            ) from e
        old = EvalProtocol.from_dict(blob.get("protocol", {}))
        if old.max_steps != self.protocol.max_steps or old.bench != self.protocol.bench:
            raise ValueError(
                f"{self.path} was produced under a different protocol "
                f"(bench={old.bench}, max_steps={old.max_steps}); resuming would "
                f"mix incomparable episodes. Use a different --out or --no-resume."
            )
        for d in blob.get("episodes", []):
            r = EpisodeResult.from_dict(d)
            self.records[r.key()] = r
        self.n_resumed = len(self.records)

    # ── mutation ──────────────────────────────────────────────────────────

    def has(self, key: tuple) -> bool:
        return key in self.records

    def add(self, record: EpisodeResult, *, flush: bool = True) -> None:
        self.records[record.key()] = record
        if flush:
            self.flush()

    def flush(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2))
        os.replace(tmp, self.path)                       # atomic; never a half file

    # ── output ────────────────────────────────────────────────────────────

    def episodes(self) -> list[EpisodeResult]:
        return [self.records[k] for k in sorted(self.records)]

    def to_dict(self) -> dict[str, Any]:
        eps = self.episodes()
        return {
            "version": RESULTS_VERSION,
            "bench": self.protocol.bench,
            "protocol": self.protocol.to_dict(),
            "meta": self.meta,
            "summary": aggregate(eps, self.protocol),
            "episodes": [e.to_dict() for e in eps],
        }


# ═══════════════════════════════════════════════════════════════════════════
#  AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════

def _rate(n_success: int, n: int) -> float:
    return 100.0 * n_success / n if n else 0.0


def aggregate(records: Iterable[EpisodeResult], protocol: EvalProtocol) -> dict[str, Any]:
    """Per-task and per-suite success rates, plus the diagnostics PLAN 4.F asks for.

    Suite rate is the mean over tasks of the per-task rate, then x100 — the
    reference convention (`_pack_grpo_results`: successes / n_generation per
    task, then mean over tasks). With a uniform episode count per task this is
    the same as pooling; it is not once an episode crashes, so it is done the
    documented way.
    """
    recs = list(records)
    per_task: dict[str, dict[str, Any]] = {}
    for r in recs:
        d = per_task.setdefault(r.suite, {}).setdefault(str(r.task_id), {
            "n": 0, "n_success": 0, "n_error": 0, "n_hit_cap": 0, "steps": 0,
            "task_name": r.task_name,
        })
        d["n"] += 1
        d["n_success"] += int(bool(r.success))
        d["n_error"] += int(r.error is not None)
        d["n_hit_cap"] += int(bool(r.hit_step_cap))
        d["steps"] += int(r.steps)
    for suite in per_task:
        for d in per_task[suite].values():
            d["success_rate"] = _rate(d["n_success"], d["n"])
            d["mean_steps"] = d.pop("steps") / d["n"] if d["n"] else 0.0

    # protocol order, so the emitted table reads spatial/object/goal/long
    suite_order = [s for s in protocol.suites if s in per_task]
    suite_order += [s for s in per_task if s not in suite_order]

    per_suite: dict[str, dict[str, Any]] = {}
    for suite in suite_order:
        tasks = per_task[suite]
        rates = [d["success_rate"] for d in tasks.values()]
        sub = [r for r in recs if r.suite == suite]
        per_seed = {}
        for s in sorted({r.seed for r in sub}):
            ss = [r for r in sub if r.seed == s]
            per_seed[str(s)] = _rate(sum(bool(r.success) for r in ss), len(ss))
        per_suite[suite] = {
            "success_rate": sum(rates) / len(rates) if rates else 0.0,
            "n_tasks": len(tasks),
            "n_episodes": len(sub),
            "n_errors": sum(r.error is not None for r in sub),
            "n_hit_step_cap": sum(bool(r.hit_step_cap) for r in sub),
            "mean_episode_len": (sum(r.steps for r in sub) / len(sub)) if sub else 0.0,
            "per_seed": per_seed,
        }

    ordered = [s for s in protocol.suites if s in per_suite]
    avg = (sum(per_suite[s]["success_rate"] for s in ordered) / len(ordered)) if ordered else 0.0
    return {
        "per_task": {s: per_task[s] for s in suite_order},
        "per_suite": per_suite,
        "avg": avg,
        "n_episodes": len(recs),
        "n_expected": protocol.total_episodes,
        "n_errors": sum(r.error is not None for r in recs),
        "n_hit_step_cap": sum(bool(r.hit_step_cap) for r in recs),
        "mean_episode_len": (sum(r.steps for r in recs) / len(recs)) if recs else 0.0,
        "complete": len(recs) >= protocol.total_episodes,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  EPISODE EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

def _run_item(
    item: WorkItem,
    policy: Any,
    mod: Any,
    env_factory: Callable[..., Any] | None,
    backend: str | None,
) -> EpisodeResult:
    """One episode, crash-contained. Never raises for an env or policy fault."""
    rec = EpisodeResult(
        bench=item.bench, suite=item.suite, task_id=item.task_id,
        episode=item.episode, seed=item.seed, env_seed=item.env_seed,
        task_name=mod.task_name(item.suite, item.task_id),
    )
    t0 = time.time()
    factory = env_factory or mod.make_env

    def build():
        return factory(item.suite, item.task_id, item.env_seed,
                       trial_id=item.episode, backend=backend,
                       max_steps=item.max_steps)

    mod.run_episode_safe(
        policy, build,
        mod.task_instruction(item.suite, item.task_id),
        item.max_steps,
        record=rec,
    )
    rec.wall_s = time.time() - t0
    if rec.error is not None:
        # The overwhelmingly likely cause of a failure at reset is a missing
        # torch.load shim (torch >= 2.6 refuses LIBERO's .pruned_init files) or,
        # on RoboTwin, an unpinned Vulkan ICD, so carry the runtime status
        # alongside the traceback rather than making someone reconstruct it from
        # a run of 1200 identical failures.
        rec.extra["runtime_status"] = runtime_status(mod)
    return rec


# ═══════════════════════════════════════════════════════════════════════════
#  WORKER PROCESSES
# ═══════════════════════════════════════════════════════════════════════════

_WORKER: dict[str, Any] = {}


def claim_device(device_queue) -> str:
    """Take this worker's GPU off the queue and pin **both** device selectors.

    Order is the whole point, and getting it wrong is silent. The two selectors
    are read at different moments by different libraries:

    * `CUDA_VISIBLE_DEVICES` is latched by the CUDA driver at `cuInit`, which
      `torch.cuda.is_available()` triggers via `cudaGetDeviceCount`. Setting it
      **after** any torch.cuda call is a no-op for the rest of the process.
    * `MUJOCO_EGL_DEVICE_ID` is read later, when the render context is created,
      and EGL enumerates physical devices — it does not honour
      `CUDA_VISIBLE_DEVICES`.

    This function used to call `default_device()` first and set the variables
    second, so only the EGL half took effect. Measured mid-run on an 8-worker
    LIBERO job (`srun --overlap --jobid=32301529 nvidia-smi`): all eight worker
    processes held 3.2 GiB each on the *same* physical GPU, which sat at 100%
    while GPUs 1-7 carried only their 277 MiB EGL context. Nothing in the
    results said so — the score is identical either way, which is exactly why
    it survived two evaluations.

    Returns the torch device string. After the pin, `cuda:0` **is** physical
    `dev`, because the process can no longer see anything else.
    """
    dev = None
    if device_queue is not None:
        try:
            dev = device_queue.get_nowait()
        except Exception:                                # noqa: BLE001
            dev = None
    if dev is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(dev)
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(dev)
    return default_device()                              # first torch.cuda call


def _init_worker(device_queue, bench: str, ckpt: str | None,
                 backend: str | None, policy_kw: dict) -> None:
    """One policy per process, pinned to one device. Runs in the child."""
    device = claim_device(device_queue)
    from loom.eval.policy import make_policy, policy_provenance  # noqa: PLC0415

    mod = bench_module(bench)
    # once per worker process, before any env is built: the bench's own headless
    # render setup (see `ensure_runtime`)
    if backend != "fake":
        ensure_runtime(mod)
    _WORKER["mod"] = mod
    _WORKER["policy"] = make_policy(ckpt, device=device, **(policy_kw or {}))
    _WORKER["backend"] = backend
    # Carried out of the child on the first record only. Which modules actually
    # ran is not recoverable from the score, and a silent stub fallback under
    # `--ckpt` looks exactly like an untrained model.
    _WORKER["provenance"] = policy_provenance(_WORKER["policy"])


def _worker_run(item_dict: dict) -> dict:
    item = WorkItem(**item_dict)
    try:
        rec = _run_item(item, _WORKER["policy"], _WORKER["mod"], None,
                        _WORKER["backend"])
    except Exception:                                    # noqa: BLE001
        rec = EpisodeResult(
            bench=item.bench, suite=item.suite, task_id=item.task_id,
            episode=item.episode, seed=item.seed, env_seed=item.env_seed,
            error=traceback.format_exc(),
        )
    prov = _WORKER.pop("provenance", None)
    if prov is not None:
        rec.extra["_policy"] = prov
    return rec.to_dict()


# ═══════════════════════════════════════════════════════════════════════════
#  RUN
# ═══════════════════════════════════════════════════════════════════════════

def run_eval(
    protocol: EvalProtocol | None = None,
    *,
    bench: str = "libero",
    ckpt: str | None = None,
    out: str | os.PathLike | None = None,
    workers: int | None = None,
    resume: bool = True,
    backend: str | None = None,
    policy: Any = None,
    policy_factory: Callable[[], Any] | None = None,
    env_factory: Callable[..., Any] | None = None,
    policy_kw: dict | None = None,
    on_episode: Callable[[EpisodeResult], None] | None = None,
) -> dict[str, Any]:
    """Run the protocol and return the results dict (also written to `out`).

    `policy` / `policy_factory` / `env_factory` are the injection seams the
    tests use; a plain CLI run supplies none of them and gets
    `loom.eval.policy.make_policy(ckpt)` plus the bench's own `make_env`.
    """
    mod = bench_module(bench)
    protocol = protocol or getattr(mod, "DEFAULT_PROTOCOL")
    if protocol.bench != bench:
        protocol = protocol.replace(bench=bench)

    # The body under evaluation follows the bench unless the caller says
    # otherwise. `load_policy`'s default is `libero_franka`, and inheriting it
    # for RoboTwin builds a 7-dof decoder for a 14-dof arm — every episode then
    # dies on the first action with a shape error, which is loud but wastes a
    # whole GPU allocation to discover.
    body = getattr(mod, "EMBODIMENT", None)
    if body is not None:
        policy_kw = dict(policy_kw or {})
        policy_kw.setdefault("embodiment", body)

    items = iter_work(protocol)
    store = ResultStore(out, protocol, resume=resume, meta={
        "ckpt": str(ckpt) if ckpt else None,
        "bench": bench,
        "backend": backend or ("auto"),
        # Whether the REAL simulator was importable. A fake-env run and a real
        # one produce identically shaped tables; this is the only field that
        # distinguishes them, so it is recorded per bench rather than only for
        # LIBERO (`libero_available` is kept for older results files).
        "env_available": env_available(mod),
        "libero_available": bool(getattr(mod, "libero_available", lambda: False)()),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    todo = [it for it in items if not store.has(it.key())]

    n_workers = workers if workers is not None else n_devices()
    n_workers = max(1, min(int(n_workers), max(1, len(todo))))
    parallel = n_workers > 1 and policy is None and env_factory is None

    if not todo:
        store.flush()
        return store.to_dict()

    if parallel:
        _run_parallel(todo, store, bench, ckpt, backend, policy_kw, n_workers, on_episode)
    else:
        pol = policy if policy is not None else (
            policy_factory() if policy_factory is not None
            else _default_policy(ckpt, policy_kw)
        )
        store.meta["policy"] = _provenance(pol)
        for item in todo:
            rec = _run_item(item, pol, mod, env_factory, backend)
            store.add(rec)                               # atomic write per episode
            if on_episode is not None:
                on_episode(rec)

    store.flush()
    return store.to_dict()


def default_device() -> str:
    """`cuda:0` when there is a GPU, else `cpu`. Never a silent CPU run."""
    try:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:                                    # noqa: BLE001
        return "cpu"


def _default_policy(ckpt: str | None, policy_kw: dict | None) -> Any:
    """The single-worker policy. **Must pick a device**, or it lands on CPU.

    `_init_worker` sets `device=` for every parallel worker; this branch did
    not, so `--workers 1` on an 8-GPU node built the 878 M tower and the
    estimator on CPU and ran the whole evaluation there. Measured: 64 env steps
    took 85-90 s on CPU against ~3 s on an A100, and nothing in the output said
    so except `meta.policy.device`.
    """
    from loom.eval.policy import make_policy             # noqa: PLC0415

    kw = dict(policy_kw or {})
    kw.setdefault("device", default_device())
    return make_policy(ckpt, **kw)


def _provenance(pol: Any) -> dict[str, Any]:
    """`policy_provenance`, but never a reason the run cannot start."""
    try:
        from loom.eval.policy import policy_provenance   # noqa: PLC0415

        return policy_provenance(pol)
    except Exception:                                    # noqa: BLE001
        return {"policy": type(pol).__name__}


def _mp_context():
    """spawn when CUDA is live (fork + CUDA is undefined), fork otherwise.

    The real LIBERO env additionally *requires* spawn — mujoco does not survive
    fork — so a GPU run is on the correct context by construction. The fork
    branch only ever serves CPU-only multi-worker runs, which are a developer
    convenience; forkserver is not used because it re-imports the parent's
    `__main__`, which fails under pytest and under `python -`.
    """
    import multiprocessing as mp                         # noqa: PLC0415

    try:
        if torch.cuda.is_available():
            return mp.get_context("spawn")
    except Exception:                                    # noqa: BLE001
        pass
    return mp.get_context("fork" if hasattr(os, "fork") else "spawn")


def _run_parallel(todo, store, bench, ckpt, backend, policy_kw, n_workers, on_episode):
    from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: PLC0415

    ctx = _mp_context()
    q = ctx.Queue()
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    for i in range(n_workers):
        q.put(i % n_gpu if n_gpu else None)

    with ProcessPoolExecutor(
        max_workers=n_workers, mp_context=ctx,
        initializer=_init_worker,
        initargs=(q, bench, ckpt, backend, policy_kw or {}),
    ) as pool:
        futures = [pool.submit(_worker_run, it.to_dict()) for it in todo]
        for fut in as_completed(futures):
            rec = EpisodeResult.from_dict(fut.result())
            prov = rec.extra.pop("_policy", None)        # one per worker process
            if prov is not None:
                store.meta.setdefault("policy", prov)
            store.add(rec)                               # parent owns the file
            if on_episode is not None:
                on_episode(rec)
