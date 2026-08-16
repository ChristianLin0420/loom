"""LOOM — Team F done-when gate (PLAN 4.F).

    "the harness runs end to end on stubs.Policy and emits a correctly-shaped
     table with random success rates. Plumbing is the deliverable."

Everything here runs on CPU, with no LIBERO installed, in a few seconds.

The load-bearing test is `test_resampling_*`: the decoder emits 8 steps at
30 Hz, LIBERO runs at 20, and `env_steps_per_segment(20) == 5.333...` is
deliberately fractional. Rounding each segment independently drifts by more
than a second over an episode, and PLAN 7 names that as one of the two failure
modes that produce a trained model with a near-zero score.
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

# Batch-1 stub tensors on a 64-core login node: torch's default 32-thread pool
# spends ~70 ms per op in thread wakeup when the node is contended, and this
# suite does ~250 of them. Nothing in eval depends on the thread count.
torch.set_num_threads(1)

import contracts as C  # noqa: E402
import stubs as S  # noqa: E402
from loom import eval as E  # noqa: E402
from loom.data import canonical  # noqa: E402  (the shared rate-conversion path)
from loom.eval import EpisodeResult, EvalProtocol, episode_seed  # noqa: E402
from loom.eval import libero, libero_plus, policy as pol, robotwin, runner, table  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

def tiny_protocol(**kw) -> EvalProtocol:
    base = dict(bench="libero", episodes_per_task=2, n_tasks=2,
                suites=("libero_spatial", "libero_object"), seeds=(0,),
                max_steps=24)
    base.update(kw)
    return EvalProtocol(**base)


def fake_env_factory(**overrides):
    """`make_env` seam pinned to the fake backend, whatever is installed."""
    def factory(suite, task_id, seed, **kw):
        kw.pop("backend", None)
        kw.update(overrides)
        return libero.FakeLiberoEnv(suite, task_id, seed, **kw)
    return factory


def stub_policy_factory():
    return S.StubPolicy()


def stub_loom_policy(**kw) -> pol.LoomPolicy:
    """`LoomPolicy` wired to stub modules — the real class, no real weights."""
    kw.setdefault("n_candidates", 2)
    return pol.LoomPolicy(pol._stub_modules("libero_franka", "cpu"), **kw)


# ═══════════════════════════════════════════════════════════════════════════
#  1 · END TO END ON stubs.Policy   —   the headline deliverable
# ═══════════════════════════════════════════════════════════════════════════

def test_end_to_end_on_stub_policy_emits_json_and_table(tmp_path):
    out = tmp_path / "results.json"
    protocol = tiny_protocol()

    results = runner.run_eval(
        protocol, bench="libero", out=out,
        policy_factory=stub_policy_factory,
        env_factory=fake_env_factory(),
    )

    # results JSON exists, is valid, and holds every episode the protocol asked for
    assert out.exists()
    blob = json.loads(out.read_text())
    assert blob == results
    assert len(blob["episodes"]) == protocol.total_episodes == 8
    assert blob["summary"]["complete"]

    # random success rates, per task and per suite, in [0, 100]
    per_suite = blob["summary"]["per_suite"]
    assert list(per_suite) == list(protocol.suites), "suites must stay in PLAN 8 order"
    for d in per_suite.values():
        assert 0.0 <= d["success_rate"] <= 100.0
        assert d["n_episodes"] == 4
    assert 0.0 <= blob["summary"]["avg"] <= 100.0

    # episode lengths and step-cap accounting are recorded
    assert blob["summary"]["n_errors"] == 0, blob["episodes"][0]["error"]
    assert blob["summary"]["mean_episode_len"] > 0
    assert "n_hit_step_cap" in blob["summary"]
    assert all(0 < e["steps"] <= protocol.max_steps for e in blob["episodes"])

    # ... and a correctly shaped markdown table falls out of it
    md = table.render_report(results)
    assert table.LIBERO_HEADER in md
    assert "LOOM · R0-A" in md
    header_i = md.index(table.LIBERO_HEADER)
    body = md[header_i:].splitlines()
    assert body[1].count("---") == len(table.LIBERO_COLUMNS)
    # 8 baselines + 2 LOOM rows, all with the right number of cells
    for line in body[2:12]:
        assert line.count("|") == len(table.LIBERO_COLUMNS) + 1


def test_end_to_end_with_the_real_policy_class(tmp_path):
    """Same harness, `LoomPolicy` on stub modules, with bounds enforced."""
    results = runner.run_eval(
        tiny_protocol(episodes_per_task=1, n_tasks=1, suites=("libero_spatial",)),
        bench="libero", out=tmp_path / "r.json",
        policy=stub_loom_policy(),
        env_factory=fake_env_factory(strict_bounds=True),
    )
    assert results["summary"]["n_episodes"] == 1
    assert results["summary"]["n_errors"] == 0


def test_cli_runs_end_to_end(tmp_path, capsys):
    from loom.eval.__main__ import main

    out, md = tmp_path / "r.json", tmp_path / "r.md"
    rc = main([
        "--bench", "libero", "--out", str(out), "--md", str(md),
        "--episodes-per-task", "1", "--n-tasks", "1",
        "--suites", "libero_spatial", "--seeds", "0",
        "--max-steps", "16", "--backend", "fake", "--quiet",
    ])
    assert rc == 0
    assert json.loads(out.read_text())["summary"]["n_episodes"] == 1
    assert table.LIBERO_HEADER in md.read_text()


# ═══════════════════════════════════════════════════════════════════════════
#  2 · RESAMPLING — the test that protects the score
# ═══════════════════════════════════════════════════════════════════════════

def test_resampling_replan_count_over_a_600_step_episode():
    """600 env steps at 20 Hz is 600 / 5.333... = 112.5 operators, not 120."""
    p = stub_loom_policy()
    p.reset()
    for _ in range(600):
        p.act({}, "put the bowl on the plate")

    expected = 600 / C.env_steps_per_segment(20.0)
    assert math.isclose(expected, 112.5)
    assert abs(p.replans - expected) <= 1.5, (
        f"{p.replans} replans over 600 steps, expected ~{expected:.1f}. "
        f"120 means the segment was rounded to 5 steps; 100 means it was "
        f"executed at the canonical rate."
    )


def test_resampling_accumulator_does_not_drift():
    """Cumulative env steps must track `n_replans * H_OP * 20/30` to within one step.

    This is the whole point of the fractional accumulator: independently
    rounding 5.333 to 5 loses 1/16 of wall-clock time per segment, which is
    ~37 steps — nearly two seconds — over a 600-step episode.
    """
    p = stub_loom_policy()
    p.reset()
    for _ in range(600):
        p.act({}, "task")

    clock = p.clock
    ideal = clock.n_replans * C.H_OP * 20.0 / C.FPS_CANONICAL
    assert abs(ideal - clock.n_steps_dispatched) <= 1.0, (
        f"drift {ideal - clock.n_steps_dispatched:.3f} steps after "
        f"{clock.n_replans} replans"
    )
    assert abs(clock.drift) <= 1.0
    assert 0.0 <= clock.accum < 1.0
    # every env step of the episode was covered, with at most one segment spare
    assert 600 <= clock.n_steps_dispatched <= 600 + C.H_OP


def test_resampling_accumulator_matches_the_stub_contract():
    """`stubs.StubPolicy` pins the accumulator; the real policy must match it."""
    ref = S.StubPolicy()
    ref.reset()
    lens_ref: list[int] = []
    for _ in range(200):
        before = ref.replans
        ref.act({}, "t")
        if ref.replans != before:
            lens_ref.append(1)
        else:
            lens_ref[-1] += 1

    mine = pol.SegmentClock(20.0)
    lens_mine = [mine.next_segment_len() for _ in range(len(lens_ref))]

    # identical to the stub, step for step (the last ref segment is truncated
    # by the 200-step budget)
    assert lens_mine[:-1] == lens_ref[:-1], "the real clock must match the stub contract"
    # segments alternate 5 and 6 env steps and never round to a constant
    assert set(lens_mine) == {5, 6}
    # and every prefix tracks the exact 16/3 rate to within one env step
    for k in range(1, len(lens_mine) + 1):
        assert abs(sum(lens_mine[:k]) - k * 16 / 3) <= 1.0


# ── the semantics-aware inverse, owned by Team A ──────────────────────────
#
# `loom.eval` has no resampler of its own. These tests pin the behaviour eval
# depends on, through the exact entry point eval calls, so that a change on
# either side of the train/eval boundary breaks here rather than in a score.

def test_eval_has_no_second_resampler():
    """One transform, one implementation. `to_env_rate` comes from `canonical`."""
    assert pol.to_env_rate is canonical.to_env_rate
    assert not hasattr(pol, "resample_segment"), (
        "a local resampler reappeared in loom.eval.policy; train and eval must "
        "share one code path or they drift apart silently"
    )


def test_libero_action_semantics_are_delta_delta_hold():
    kinds = canonical.action_semantics("libero_franka")
    assert kinds == (canonical.DELTA,) * 6 + (canonical.HOLD,)


def test_action_round_trip_preserves_integrated_motion_and_the_gripper():
    """20 Hz -> canonical 30 Hz -> back to 20 Hz. The guard on the score.

    Delta channels must return the same *integrated* displacement (per-step
    magnitude legitimately rescales by src/dst); the gripper must still be
    latched at exactly +/-1, never an interpolated value in between.
    """
    rng = np.random.default_rng(0)
    n_src = 24                                   # 1.2 s at 20 Hz
    src = np.empty((n_src, 7), dtype=np.float32)
    src[:, :6] = rng.uniform(-0.4, 0.4, (n_src, 6))
    src[:, 6] = np.where(np.arange(n_src) < 10, -1.0, 1.0)      # latched grasp

    kinds = canonical.action_semantics("libero_franka")
    canon = canonical.resample_actions(src, 20.0, C.FPS_CANONICAL, kinds)
    assert canon.shape[0] == 36                  # 1.2 s at 30 Hz
    back = canonical.resample_actions(canon, C.FPS_CANONICAL, 20.0, kinds, n_dst=n_src)

    assert np.allclose(back[:, :6].sum(0), src[:, :6].sum(0), atol=1e-4), (
        "integrated motion changed across the round trip"
    )
    assert set(np.unique(canon[:, 6]).tolist()) <= {-1.0, 1.0}
    assert set(np.unique(back[:, 6]).tolist()) <= {-1.0, 1.0}, (
        "the gripper was interpolated; it is a latched channel"
    )


def test_delta_magnitude_rescales_by_the_rate_ratio():
    """Team A's worked number: 0.3 at 20 Hz canonicalises to 0.2 at 30 Hz."""
    kinds = canonical.action_semantics("libero_franka")
    src = np.full((8, 7), 0.3, dtype=np.float32)
    src[:, 6] = -1.0
    canon = canonical.resample_actions(src, 20.0, C.FPS_CANONICAL, kinds)
    assert np.allclose(canon[:, :6], 0.2, atol=1e-6)


def test_to_env_rate_restores_the_env_rate_magnitude():
    """... and the eval-side inverse turns that 0.2 back into 0.3.

    Interpolating instead would send 0.2 to a 20 Hz env and under-actuate every
    delta channel by a third, forever, with no other symptom.
    """
    seg = np.full((C.H_OP, 7), 0.2, dtype=np.float32)
    seg[:, 6] = 1.0
    out = pol.to_env_rate(seg, "libero_franka", 5)
    assert out.shape == (5, 7)
    assert np.allclose(out[:, :6], 0.3, atol=1e-6)
    assert set(np.unique(out[:, 6]).tolist()) == {1.0}


def test_to_env_rate_holds_the_gripper_through_a_switch():
    seg = np.zeros((C.H_OP, 7), dtype=np.float32)
    seg[:, 6] = [-1, -1, -1, -1, 1, 1, 1, 1]
    for n_env in (5, 6):
        out = pol.to_env_rate(seg, "libero_franka", n_env)
        assert set(np.unique(out[:, 6]).tolist()) <= {-1.0, 1.0}
        # and the switch is not reordered in time
        g = out[:, 6]
        assert np.all(np.diff(g) >= 0)


def test_to_env_rate_is_identity_at_the_canonical_rate():
    seg = np.random.default_rng(1).normal(size=(C.H_OP, 7)).astype(np.float32)
    seg[:, 6] = 1.0
    out = pol.to_env_rate(seg, "libero_franka", C.H_OP, src_fps=C.FPS_CANONICAL)
    assert np.allclose(out, seg, atol=1e-6)
    # 30 Hz env => 8 env steps per segment, exactly
    assert C.env_steps_per_segment(C.FPS_CANONICAL) == C.H_OP


def test_policy_executes_the_semantics_aware_segment():
    """End of the chain: what the decoder emits is what reaches the env."""
    class FixedDecoder:
        def forward(self, z, c):
            seg = np.zeros((C.H_OP, 7), dtype=np.float32)
            seg[:, :6] = 0.2
            seg[:, 6] = [-1, -1, -1, -1, 1, 1, 1, 1]
            return torch.from_numpy(seg).unsqueeze(0)

    mods = pol._stub_modules("libero_franka", "cpu")
    mods.decoder = FixedDecoder()
    p = pol.LoomPolicy(mods, n_candidates=2)
    p.reset()
    acted = np.stack([p.act({}, "task") for _ in range(5)])

    assert p.replans == 1, "one segment should cover the first 5 env steps"
    assert np.allclose(acted[:, :6], 0.3, atol=1e-6), "delta magnitude not restored"
    assert set(np.unique(acted[:, 6]).tolist()) <= {-1.0, 1.0}, "gripper interpolated"


def test_policy_emits_one_action_per_env_step_with_the_right_shape():
    p = stub_loom_policy()
    p.reset()
    a = p.act({}, "task")
    assert isinstance(a, np.ndarray)
    assert a.shape == (C.EMBODIMENTS["libero_franka"].dof,)
    assert np.isfinite(a).all()
    assert (a >= -1.0).all() and (a <= 1.0).all(), "actions must respect the spec bounds"


def test_policy_reset_clears_the_accumulator():
    p = stub_loom_policy()
    for _ in range(20):
        p.act({}, "task")
    p.reset()
    assert p.replans == 0 and p.clock.accum == 0.0 and p._queue == []


def test_segment_clock_rejects_nonsense_rates():
    with pytest.raises(ValueError):
        pol.SegmentClock(0.0)
    with pytest.raises(ValueError):
        pol.SegmentClock(-20.0)
    # below FPS_CANONICAL / H_OP = 3.75 Hz an operator wants less than one env
    # step; the segment is truncated (same as the stub) and the drift bound
    # does not apply. LIBERO at 20 Hz is nowhere near this.
    slow = pol.SegmentClock(1.0)
    assert slow.steps_per_segment < 1.0
    assert all(slow.next_segment_len() >= 1 for _ in range(20))


# ═══════════════════════════════════════════════════════════════════════════
#  3 · DECOUPLING — eval knows only contracts.Policy
# ═══════════════════════════════════════════════════════════════════════════

EVAL_DIR = Path(__file__).resolve().parents[1] / "loom" / "eval"

#: The training stack. Eval must never import these at module scope.
FORBIDDEN_ROOTS = ("loom.model", "loom.heads", "loom.losses", "loom.train",
                   "loom.search")

#: The sanctioned cross-team imports, each because eval and training must share
#: ONE implementation of a transform rather than two that can drift: rate
#: conversion, image orientation, and (Team I) the frozen tower's image
#: preprocessing and encoder. All three are numpy/torch-level at module scope —
#: `tower` imports `transformers` lazily inside its loader — and so are safe in
#: the separate LIBERO interpreter. Nothing else from `loom.data` (loader,
#: cache, the HDF5 reader) may be imported by eval.
ALLOWED_DATA_IMPORTS = {"loom.data.canonical", "loom.data.adapters.libero",
                        "loom.data.tower"}


def _module_scope_imports(path: Path) -> list[str]:
    """Every module imported at module scope, i.e. not inside a function body."""
    tree = ast.parse(path.read_text())
    found: list[str] = []

    def walk(node: ast.AST, in_func: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                walk(child, True)
                continue
            if not in_func:
                if isinstance(child, ast.Import):
                    found.extend(a.name for a in child.names)
                elif isinstance(child, ast.ImportFrom) and child.module:
                    found.append(child.module)
            walk(child, in_func)

    walk(tree, False)
    return found


def test_eval_does_not_import_model_at_module_scope():
    """PLAN 4.F: eval depends on nothing but `contracts.Policy`.

    Real model modules are reachable only through
    `loom.eval.policy.load_policy`, which imports them inside the function and
    falls back to stubs. This is what lets Team F land before Teams B/C/E do.
    """
    files = sorted(EVAL_DIR.glob("*.py"))
    assert len(files) >= 7, f"expected the whole eval package, found {files}"
    for f in files:
        for mod in _module_scope_imports(f):
            assert not any(mod == r or mod.startswith(r + ".") for r in FORBIDDEN_ROOTS), (
                f"{f.name} imports {mod} at module scope; eval must stay "
                f"importable while the other teams are mid-flight"
            )


def test_only_the_shared_transforms_are_imported_from_loom_data():
    """`loom.data` is not open season — only the shared transforms are sanctioned."""
    seen = set()
    for f in sorted(EVAL_DIR.glob("*.py")):
        for mod in _module_scope_imports(f):
            if mod == "loom.data" or mod.startswith("loom.data."):
                assert mod in ALLOWED_DATA_IMPORTS, (
                    f"{f.name} imports {mod}; only {sorted(ALLOWED_DATA_IMPORTS)} "
                    f"are shared with eval"
                )
                seen.add(mod)
    assert seen == ALLOWED_DATA_IMPORTS, (
        f"expected eval to use both shared transforms, found {sorted(seen)}"
    )


def test_policy_imports_real_modules_lazily():
    src = (EVAL_DIR / "policy.py").read_text()
    assert "from loom.heads.decoder import" in src, "the lazy factory disappeared"
    for f in ("_try_real_modules", "_stub_modules"):
        assert f in src


def test_eval_imports_with_only_torch_numpy_and_contracts():
    """Eval runs under the LIBERO interpreter, not the training venv."""
    import importlib

    for name in ("loom.eval", "loom.eval.policy", "loom.eval.libero",
                 "loom.eval.runner", "loom.eval.table", "loom.eval.robotwin",
                 "loom.eval.libero_plus", "loom.eval.__main__"):
        importlib.import_module(name)


def test_load_policy_falls_back_to_stubs_without_a_checkpoint():
    p = pol.load_policy(None)
    assert isinstance(p, pol.LoomPolicy)
    assert p.modules.is_stub
    with pytest.raises(RuntimeError):
        pol.load_policy(None, allow_stub=False)


# ═══════════════════════════════════════════════════════════════════════════
#  4 · PROTOCOL IS LOGGED, NEVER HARDCODED
# ═══════════════════════════════════════════════════════════════════════════

def test_default_protocol_is_the_stated_one():
    """PLAN 4.F: 10 episodes/task x 10 tasks x 4 suites over 3 seeds, stated."""
    p = libero.DEFAULT_PROTOCOL
    assert p.episodes_per_task == 10
    assert p.n_tasks == 10
    assert p.suites == ("libero_spatial", "libero_object", "libero_goal", "libero_long")
    assert len(p.seeds) == 3
    assert p.max_steps == 512                    # LIBERO_MAX_STEPS_MAP, not invented
    assert p.total_episodes == 10 * 10 * 4 * 3 == 1200


def test_protocol_is_written_into_the_results_json(tmp_path):
    protocol = tiny_protocol()
    results = runner.run_eval(protocol, out=tmp_path / "r.json",
                              policy_factory=stub_policy_factory,
                              env_factory=fake_env_factory())
    logged = results["protocol"]
    assert logged["episodes_per_task"] == protocol.episodes_per_task
    assert logged["n_tasks"] == protocol.n_tasks
    assert logged["suites"] == list(protocol.suites)
    assert logged["seeds"] == list(protocol.seeds)
    assert logged["max_steps"] == protocol.max_steps
    assert logged["total_episodes"] == protocol.total_episodes
    assert EvalProtocol.from_dict(logged) == protocol


def test_changing_the_protocol_changes_the_emitted_header(tmp_path):
    def emit(protocol):
        r = runner.run_eval(protocol, out=tmp_path / f"{protocol.episodes_per_task}.json",
                            policy_factory=stub_policy_factory,
                            env_factory=fake_env_factory())
        return table.render_report(r)

    a = emit(tiny_protocol())
    b = emit(tiny_protocol(episodes_per_task=3, seeds=(0, 7)))

    assert "2 episodes/task" in a and "1 seeds" in a
    assert "3 episodes/task" in b and "2 seeds" in b and "seeds 0,7" in b
    assert a.splitlines()[2] != b.splitlines()[2]
    # the protocol statement sits above the table, not buried in the JSON
    assert a.index("Protocol") < a.index(table.LIBERO_HEADER)
    assert str(tiny_protocol().max_steps) in a


def test_protocol_rejects_impossible_values():
    for kw in ({"episodes_per_task": 0}, {"n_tasks": -1}, {"suites": ()},
               {"seeds": ()}, {"max_steps": 0}):
        with pytest.raises(ValueError):
            tiny_protocol(**kw)


# ═══════════════════════════════════════════════════════════════════════════
#  5 · TABLES — PLAN 8 column order, verbatim baselines
# ═══════════════════════════════════════════════════════════════════════════

def test_libero_table_column_order_matches_plan_8():
    assert table.LIBERO_HEADER == \
        "| method | params | emb. PT | spatial | object | goal | long | avg |"
    assert table.libero_table().splitlines()[0] == table.LIBERO_HEADER


def test_robotwin_table_column_order_matches_plan_8():
    assert table.ROBOTWIN_HEADER == \
        "| method | clean | rand | hanging mug | turn switch | place can basket | handover block |"
    assert table.robotwin_table().splitlines()[0] == table.ROBOTWIN_HEADER


def test_libero_plus_table_column_order_matches_plan_8():
    assert table.LIBERO_PLUS_HEADER == \
        "| method | camera | robot init | layout | geo avg | light | backgnd | language | noise | total |"
    assert table.libero_plus_table().splitlines()[0] == table.LIBERO_PLUS_HEADER


def test_geo_avg_is_the_mean_of_camera_robot_init_layout():
    assert table.geo_avg(80.5, 89.6, 82.8) == pytest.approx(84.3, abs=0.05)
    md = table.libero_plus_table({"**LOOM · R2**": {
        "camera": 60.0, "robot init": 70.0, "layout": 80.0, "total": 75.0}})
    row = [l for l in md.splitlines() if "LOOM · R2" in l][0]
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[1:5] == ["60.0", "70.0", "80.0", "70.0"]


def test_baselines_are_carried_verbatim_from_one_source_per_table():
    """Never assemble a table across papers (PLAN 8)."""
    md = table.libero_table()
    assert "| Fast-WAM | 6 B | ✗ | 97.0 | 99.4 | 96.6 | 94.8 | 97.0 |" in md
    assert "| π0.5 | 3 B | ✓ | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |" in md
    assert table.LIBERO_SOURCE == "Light-WAM Table 1"
    assert "Fast-WAM" in table.ROBOTWIN_SOURCE and "OA-WAM" in table.LIBERO_PLUS_SOURCE
    assert len(table.LIBERO_BASELINES) == 8
    assert all(len(r) == len(table.LIBERO_COLUMNS) for r in table.LIBERO_BASELINES)
    assert all(len(r) == len(table.ROBOTWIN_COLUMNS) for r in table.ROBOTWIN_BASELINES)
    assert all(len(r) == len(table.LIBERO_PLUS_COLUMNS) for r in table.LIBERO_PLUS_BASELINES)


def test_loom_rows_are_blank_until_measured():
    md = table.libero_table()
    row = [l for l in md.splitlines() if "LOOM · R0-A" in l][0]
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[:3] == ["**LOOM · R0-A**", "0.3 B", "✗"]
    assert cells[3:] == ["", "", "", "", ""], "an unmeasured cell must stay empty"


def test_libero_row_avg_is_the_mean_of_the_four_suites():
    results = {"summary": {"per_suite": {
        "libero_spatial": {"success_rate": 90.0},
        "libero_object": {"success_rate": 80.0},
        "libero_goal": {"success_rate": 70.0},
        "libero_long": {"success_rate": 60.0},
    }}}
    row = table.libero_row_from_results(results)
    assert row["avg"] == pytest.approx(75.0)
    md = table.libero_table({"**LOOM · R0-A**": row})
    assert "| **LOOM · R0-A** | 0.3 B | ✗ | 90.0 | 80.0 | 70.0 | 60.0 | 75.0 |" in md


def test_libero_10_is_the_long_column():
    results = {"summary": {"per_suite": {"libero_10": {"success_rate": 42.0}}}}
    assert table.libero_row_from_results(results)["long"] == 42.0


# ═══════════════════════════════════════════════════════════════════════════
#  6 · A CRASHED EPISODE IS A FAILURE, NOT A DEAD RUN
# ═══════════════════════════════════════════════════════════════════════════

def test_a_crashed_episode_is_recorded_and_the_run_continues(tmp_path):
    def factory(suite, task_id, seed, **kw):
        kw.pop("backend", None)
        if suite == "libero_spatial" and task_id == 1:
            # p_success=0 so the episode cannot finish before the crash lands:
            # 15 settle steps + 5 policy steps
            return libero.FakeLiberoEnv(suite, task_id, seed, crash_at=20,
                                        p_success=0.0, **kw)
        return libero.FakeLiberoEnv(suite, task_id, seed, **kw)

    protocol = tiny_protocol()
    results = runner.run_eval(protocol, out=tmp_path / "r.json",
                              policy_factory=stub_policy_factory,
                              env_factory=factory)

    assert len(results["episodes"]) == protocol.total_episodes, "the run stopped"
    crashed = [e for e in results["episodes"] if e["error"]]
    assert len(crashed) == protocol.episodes_per_task, "the injected crash was swallowed"
    assert all(e["success"] is False for e in crashed)
    assert "injected crash" in crashed[0]["error"]
    assert "Traceback" in crashed[0]["error"], "the traceback must reach the JSON"
    assert results["summary"]["n_errors"] == len(crashed)


def test_a_crashing_policy_does_not_kill_the_run(tmp_path):
    class BadPolicy:
        def reset(self):
            pass

        def act(self, obs, instruction):
            raise ZeroDivisionError("policy blew up")

    results = runner.run_eval(tiny_protocol(), out=tmp_path / "r.json",
                              policy=BadPolicy(), env_factory=fake_env_factory())
    assert results["summary"]["n_errors"] == 8
    assert results["summary"]["avg"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  7 · RESUMABLE
# ═══════════════════════════════════════════════════════════════════════════

def test_results_json_is_resumable(tmp_path):
    out = tmp_path / "results.json"
    protocol = tiny_protocol()
    ran: list[tuple] = []

    class CountingPolicy(S.StubPolicy):
        pass

    def run(stop_after=None):
        seen = []

        def tick(rec):
            seen.append(rec.key())
            ran.append(rec.key())
            if stop_after is not None and len(seen) >= stop_after:
                raise KeyboardInterrupt("preempted")

        try:
            return runner.run_eval(protocol, out=out, policy=CountingPolicy(),
                                   env_factory=fake_env_factory(), on_episode=tick)
        except KeyboardInterrupt:
            return None

    run(stop_after=3)
    partial = json.loads(out.read_text())
    assert len(partial["episodes"]) == 3, "the file was not written incrementally"
    first_pass = list(ran)

    ran.clear()
    results = run()
    assert len(results["episodes"]) == protocol.total_episodes
    assert len(ran) == protocol.total_episodes - 3, "completed episodes were repeated"
    assert not set(ran) & set(first_pass)
    # the resumed episodes survived verbatim
    keep = {tuple(e[k] for k in ("bench", "suite", "task_id", "episode", "seed")): e
            for e in partial["episodes"]}
    for e in results["episodes"]:
        k = tuple(e[k] for k in ("bench", "suite", "task_id", "episode", "seed"))
        if k in keep:
            assert e["success"] == keep[k]["success"] and e["steps"] == keep[k]["steps"]


def test_resume_refuses_to_mix_incomparable_protocols(tmp_path):
    out = tmp_path / "r.json"
    runner.run_eval(tiny_protocol(), out=out, policy_factory=stub_policy_factory,
                    env_factory=fake_env_factory())
    with pytest.raises(ValueError, match="different protocol"):
        runner.run_eval(tiny_protocol(max_steps=99), out=out,
                        policy_factory=stub_policy_factory,
                        env_factory=fake_env_factory())


def test_no_resume_starts_over(tmp_path):
    out = tmp_path / "r.json"
    protocol = tiny_protocol()
    runner.run_eval(protocol, out=out, policy_factory=stub_policy_factory,
                    env_factory=fake_env_factory())
    n = []
    runner.run_eval(protocol, out=out, resume=False, policy_factory=stub_policy_factory,
                    env_factory=fake_env_factory(), on_episode=lambda r: n.append(r))
    assert len(n) == protocol.total_episodes


# ═══════════════════════════════════════════════════════════════════════════
#  8 · PROTOCOL CONFORMANCE
# ═══════════════════════════════════════════════════════════════════════════

def test_loom_policy_satisfies_contracts_policy():
    assert isinstance(stub_loom_policy(), C.Policy)
    assert isinstance(S.StubPolicy(), C.Policy)
    assert isinstance(pol.load_policy(None), C.Policy)


# ═══════════════════════════════════════════════════════════════════════════
#  DETERMINISM AND SHARDING
# ═══════════════════════════════════════════════════════════════════════════

def test_episode_seed_is_deterministic_and_distinct():
    a = episode_seed(0, "libero", "libero_spatial", 3, 7)
    assert a == episode_seed(0, "libero", "libero_spatial", 3, 7)
    assert a != episode_seed(1, "libero", "libero_spatial", 3, 7)
    assert a != episode_seed(0, "libero", "libero_object", 3, 7)
    assert a != episode_seed(0, "libero", "libero_spatial", 4, 7)
    assert a != episode_seed(0, "libero", "libero_spatial", 3, 8)
    assert 0 <= a < 2 ** 31


def test_work_is_deterministic_and_covers_the_protocol():
    protocol = tiny_protocol()
    w1, w2 = runner.iter_work(protocol), runner.iter_work(protocol)
    assert [i.to_dict() for i in w1] == [i.to_dict() for i in w2]
    assert len(w1) == protocol.total_episodes
    assert len({i.key() for i in w1}) == len(w1)


def test_shard_is_a_partition():
    items = runner.iter_work(tiny_protocol())
    shards = runner.shard(items, 3)
    assert sum(len(s) for s in shards) == len(items)
    assert {i.key() for s in shards for i in s} == {i.key() for i in items}
    assert max(len(s) for s in shards) - min(len(s) for s in shards) <= 1


def test_same_seed_gives_the_same_episode(tmp_path):
    protocol = tiny_protocol()
    a = runner.run_eval(protocol, out=tmp_path / "a.json",
                        policy_factory=stub_policy_factory, env_factory=fake_env_factory())
    b = runner.run_eval(protocol, out=tmp_path / "b.json",
                        policy_factory=stub_policy_factory, env_factory=fake_env_factory())
    assert [e["env_seed"] for e in a["episodes"]] == [e["env_seed"] for e in b["episodes"]]
    # the fake env is a pure function of its seed, so outcomes match too
    assert [e["success"] for e in a["episodes"]] == [e["success"] for e in b["episodes"]]


def test_degrades_to_one_process_on_cpu():
    assert runner.n_devices() >= 1


@pytest.mark.slow
def test_parallel_workers_produce_the_same_episodes(tmp_path):
    """Sharding across processes must not change the result set."""
    protocol = tiny_protocol(max_steps=16)
    serial = runner.run_eval(protocol, out=tmp_path / "s.json", workers=1,
                             backend="fake")
    par = runner.run_eval(protocol, out=tmp_path / "p.json", workers=2,
                          backend="fake")
    key = lambda e: (e["suite"], e["task_id"], e["episode"], e["seed"])   # noqa: E731
    assert sorted(map(key, serial["episodes"])) == sorted(map(key, par["episodes"]))
    assert {key(e): e["env_seed"] for e in serial["episodes"]} == \
           {key(e): e["env_seed"] for e in par["episodes"]}


# ═══════════════════════════════════════════════════════════════════════════
#  LIBERO SPECIFICS
# ═══════════════════════════════════════════════════════════════════════════

def test_success_is_latched_anywhere_in_the_episode():
    """LIBERO counts a task solved if the env raises success at any point."""
    class LateThenNeverEnv:
        def __init__(self):
            self.t = 0

        def reset(self):
            return {}

        def step(self, a):
            self.t += 1
            success = self.t == 5
            return {}, 0.0, success, {"success": success}

        def close(self):
            pass

    out = libero.run_episode(S.StubPolicy(), LateThenNeverEnv(), "t", 50,
                             settle_steps=0, obs_fn=lambda o: o)
    assert out["success"] is True
    assert out["steps"] == 5, "the episode must stop at success, not run to the cap"
    assert out["hit_step_cap"] is False


def test_step_cap_is_recorded():
    env = libero.FakeLiberoEnv("libero_spatial", 0, 0, p_success=0.0, max_steps=10)
    out = libero.run_episode(S.StubPolicy(), env, "t", 10, obs_fn=lambda o: o)
    assert out["success"] is False
    assert out["steps"] == 10
    assert out["hit_step_cap"] is True


def test_settle_steps_run_before_the_policy_and_do_not_count():
    """15 dummy [0,0,0,0,0,0,-1] actions let the scene settle first."""
    seen: list[np.ndarray] = []

    class RecordingEnv(libero.FakeLiberoEnv):
        def step(self, a):
            seen.append(np.asarray(a).copy())
            return super().step(a)

    env = RecordingEnv("libero_spatial", 0, 0, p_success=0.0)
    out = libero.run_episode(S.StubPolicy(), env, "t", 5, obs_fn=lambda o: o)
    assert len(seen) == libero.SETTLE_STEPS + 5
    assert out["steps"] == 5, "settle steps must not count against max_steps"
    for a in seen[:libero.SETTLE_STEPS]:
        assert np.array_equal(a, libero.DUMMY_ACTION)


def test_image_orientation_is_delegated_not_reimplemented():
    """opengl-convention HDF5s vs the live env: a silent near-zero-score bug.

    Team G is settling the transform empirically (the in-tree reference does
    `[::-1, ::-1]`, a 180 rotation, while the stored convention implies a
    vertical flip). Eval must therefore hold no opinion of its own: it declares
    which convention the simulator is configured with and calls Team A's
    helper, so whatever Team G measures propagates here without an edit.
    """
    from loom.data.adapters.libero import orient_env_image

    assert libero.orient_env_image is orient_env_image

    # no reversing slice anywhere in eval's *code* (docstrings may discuss it)
    for f in sorted(EVAL_DIR.glob("*.py")):
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.Slice) and isinstance(node.step, ast.UnaryOp):
                assert not isinstance(node.step.op, ast.USub), (
                    f"{f.name}:{node.lineno} reimplements the flip instead of "
                    f"delegating to orient_env_image"
                )

    img = np.arange(2 * 3 * 3).reshape(2, 3, 3).astype(np.uint8)
    assert np.array_equal(
        libero._orient(img),
        orient_env_image(img, libero.LIBERO_ENV_IMAGE_CONVENTION),
    )
    # "opencv" means robosuite already flipped; eval must not flip twice
    assert np.array_equal(orient_env_image(img, "opencv"), img)


def test_extract_obs_builds_the_reference_state_vector():
    raw = libero.FakeLiberoEnv("libero_spatial", 0, 0).reset()
    obs = libero.extract_obs(raw)
    assert obs["full_image"].shape == raw["agentview_image"].shape
    assert np.array_equal(obs["full_image"], libero._orient(raw["agentview_image"]))
    # eef_pos (3) + axis-angle (3) + gripper_qpos (2)
    assert obs["state"].shape == (8,)


def test_quat2axisangle_matches_the_reference():
    assert np.allclose(libero.quat2axisangle(np.array([0.0, 0.0, 0.0, 1.0])), np.zeros(3))
    q = np.array([np.sin(0.25), 0.0, 0.0, np.cos(0.25)])
    assert np.allclose(libero.quat2axisangle(q), [0.5, 0.0, 0.0], atol=1e-6)


def test_torch_load_shim_unblocks_libero_init_states(tmp_path):
    """Without this, every episode dies at reset and there is no score at all.

    LIBERO reads `.pruned_init` with a bare `torch.load(path)`; torch >= 2.6
    defaults that to `weights_only=True` and refuses the plain-python payload.
    """
    payload = {"states": np.zeros((50, 71)), "meta": {"note": "plain python"}}
    p = tmp_path / "task.pruned_init"
    torch.save(payload, p)

    orig = torch.load
    try:
        # the bare call LIBERO makes, unpatched
        with pytest.raises(Exception) as e:
            orig(p)
        assert "weights_only" in str(e.value).lower() or "unpickl" in str(e.value).lower()

        status = libero.patch_torch_load_for_init_states()
        assert "weights_only=False" in status
        loaded = torch.load(p)                    # same bare call, now fine
        assert loaded["meta"]["note"] == "plain python"
    finally:
        torch.load = orig


def test_runtime_setup_is_idempotent_and_not_applied_at_import():
    """`torch.load` is process-global; eval is imported by the training venv too."""
    src = (EVAL_DIR / "libero.py").read_text()
    tree = ast.parse(src)
    for node in ast.iter_child_nodes(tree):
        assert not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call), (
            "libero.py calls something at import time; the torch.load patch must "
            "only happen when a real env is actually constructed"
        )
    orig = torch.load
    try:
        first = libero.ensure_libero_runtime()
        assert libero.ensure_libero_runtime() is first, "shim applied twice"
        assert libero.LIBERO_RUNTIME_STATUS == first
    finally:
        torch.load = orig
        libero.LIBERO_RUNTIME_STATUS = None


def test_verified_environment_facts_are_pinned():
    """Measured by Team G's smoke test on an A100; green for all four suites."""
    assert libero.MEASURED_CONTROL_FREQ == C.EMBODIMENTS["libero_franka"].env_fps == 20.0
    assert libero.LIBERO_ENV_IMAGE_CONVENTION == "opengl"    # 8/8 vote identity
    assert libero.HEADLESS_ENV["MUJOCO_GL"] == "egl"         # no osmesa fallback
    assert libero.SETTLE_STEPS == 15
    assert libero.LIBERO_PYTHON.endswith("/loom-libero/bin/python")
    assert all(v == 512 for v in libero.MAX_STEPS_BY_SUITE.values())


def test_default_protocol_fits_the_walltime_cap_when_sharded():
    """1200 episodes at 10-15 s each is 4-5 GPU-hours; the cap is 4 h per link."""
    lo, hi = libero.EPISODE_SECONDS
    serial_h = libero.DEFAULT_PROTOCOL.total_episodes * hi / 3600.0
    assert 4.0 <= serial_h <= 5.0, f"{serial_h:.1f} GPU-hours single-process"
    assert serial_h / 8 < 4.0, "must fit the 4 h walltime cap on one 8-GPU node"


def test_suite_names_and_aliases():
    assert libero.SUITES == E.DEFAULT_LIBERO_SUITES
    assert libero.benchmark_name("libero_long") == "libero_10"
    assert libero.benchmark_name("libero_10") == "libero_10"
    assert all(libero.n_tasks(s) == 10 for s in libero.SUITES)
    assert all(libero.MAX_STEPS_BY_SUITE[s] == 512 for s in libero.SUITES)
    with pytest.raises(KeyError):
        libero.benchmark_name("libero_nope")


def test_env_seam_falls_back_to_the_fake_env():
    env = libero.make_env("libero_spatial", 0, 123)
    assert isinstance(env, libero.FakeLiberoEnv) or libero.libero_available()
    with pytest.raises(ValueError):
        libero.make_env("libero_spatial", 0, 0, backend="nope")


def test_fake_env_is_a_pure_function_of_its_seed():
    a = libero.FakeLiberoEnv("libero_goal", 2, 7)
    b = libero.FakeLiberoEnv("libero_goal", 2, 7)
    assert a._will_succeed == b._will_succeed and a._solve_step == b._solve_step


def test_fake_env_rejects_a_malformed_action():
    env = libero.FakeLiberoEnv("libero_spatial", 0, 0)
    env.reset()
    with pytest.raises(ValueError):
        env.step(np.zeros(3))
    with pytest.raises(ValueError):
        env.step(np.full(7, np.nan))


# ═══════════════════════════════════════════════════════════════════════════
#  ENV FPS — the frozen constant the accumulator depends on
# ═══════════════════════════════════════════════════════════════════════════

def test_libero_env_fps_is_20_hz():
    """If the real env measures anything but 20 Hz this is a frozen-contract problem.

    `contracts.EMBODIMENTS["libero_franka"].env_fps` is frozen at 20.0 and the
    whole fractional-accumulator path is derived from it. robosuite's
    `control_freq` for LIBERO is 20; if Team G measures otherwise, contracts.py
    has to change and every eval number moves.
    """
    assert C.EMBODIMENTS["libero_franka"].env_fps == 20.0
    assert C.env_steps_per_segment(20.0) == pytest.approx(16 / 3)
    assert not float(C.env_steps_per_segment(20.0)).is_integer()


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 1B SEAMS
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("mod", [robotwin, libero_plus])
def test_phase_1b_modules_have_the_same_seam(mod):
    assert isinstance(mod.DEFAULT_PROTOCOL, EvalProtocol)
    for name in ("make_env", "task_name", "task_instruction", "n_tasks",
                 "run_episode", "run_episode_safe", "SUITES"):
        assert hasattr(mod, name), f"{mod.__name__} is missing {name}"
    with pytest.raises(NotImplementedError):
        mod.make_env(mod.SUITES[0], 0, 0)


def test_phase_1b_columns_line_up_with_the_tables():
    assert set(robotwin.TASKS) == {"hanging mug", "turn switch",
                                   "place can basket", "handover block"}
    assert robotwin.SUITES[:2] == ("clean", "randomized")
    assert libero_plus.GEO_AXES == ("camera", "robot_init", "layout")
    for axis in libero_plus.SUITES:
        assert axis.replace("_", " ") in table.LIBERO_PLUS_COLUMNS or \
            axis in ("background",)          # PLAN 8 abbreviates it to `backgnd`


def test_bench_module_dispatch():
    assert runner.bench_module("libero") is libero
    assert runner.bench_module("robotwin") is robotwin
    assert runner.bench_module("libero_plus") is libero_plus
    with pytest.raises(ValueError):
        runner.bench_module("nope")


# ═══════════════════════════════════════════════════════════════════════════
#  AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════

def _rec(suite, task, ep, seed, success):
    return EpisodeResult(bench="libero", suite=suite, task_id=task, episode=ep,
                         seed=seed, env_seed=0, success=success, steps=10)


def test_suite_rate_is_the_mean_over_tasks():
    """successes / n per task, then mean over tasks x 100 (reference convention)."""
    recs = [_rec("libero_spatial", 0, 0, 0, True), _rec("libero_spatial", 0, 1, 0, True),
            _rec("libero_spatial", 1, 0, 0, True), _rec("libero_spatial", 1, 1, 0, False)]
    s = runner.aggregate(recs, tiny_protocol(suites=("libero_spatial",)))
    assert s["per_task"]["libero_spatial"]["0"]["success_rate"] == 100.0
    assert s["per_task"]["libero_spatial"]["1"]["success_rate"] == 50.0
    assert s["per_suite"]["libero_spatial"]["success_rate"] == 75.0
    assert s["avg"] == 75.0


def test_per_seed_rates_are_reported():
    recs = [_rec("libero_spatial", 0, 0, 0, True), _rec("libero_spatial", 0, 0, 1, False)]
    s = runner.aggregate(recs, tiny_protocol(suites=("libero_spatial",), seeds=(0, 1)))
    assert s["per_suite"]["libero_spatial"]["per_seed"] == {"0": 100.0, "1": 0.0}


def test_aggregate_on_no_episodes_is_not_a_crash():
    s = runner.aggregate([], tiny_protocol())
    assert s["avg"] == 0.0 and s["n_episodes"] == 0 and not s["complete"]


# ═══════════════════════════════════════════════════════════════════════════
#  CHECKPOINT PROVENANCE
#
#  `loom.train.ckpt.build_state` stores `payload["model"] =
#  LoomModel.state_dict()`, which is FLAT and dotted. `state.get("estimator")`
#  on that returns None, and the old loader treated that as "nothing to load"
#  and scored randomly initialised weights with no error and no warning.
#  Measured against runs/r0a_smoke/ckpt_000000030_rank0.pt: 929 flat keys, none
#  of them reachable by name.
# ═══════════════════════════════════════════════════════════════════════════

def test_submodule_state_reads_the_flat_training_layout():
    """The layout `loom.train.ckpt` actually writes, including the `inner.` hop."""
    flat = {
        "estimator.latents": torch.zeros(2),
        "estimator.blocks.0.w": torch.zeros(2),
        "proposal.query": torch.zeros(2),
        "decoder.inner.bodies.libero_franka.step_emb": torch.zeros(2),
        "bank.log_r": torch.zeros(2),
    }
    est = pol.submodule_state(flat, "estimator")
    assert set(est) == {"latents", "blocks.0.w"}, est
    assert set(pol.submodule_state(flat, "proposal")) == {"query"}
    # EmbodimentHeads wraps Team C's container; that level has to come off
    assert set(pol.submodule_state(flat, "decoder")) == {"bodies.libero_franka.step_emb"}
    assert pol.submodule_state(flat, "q_delta") is None


def test_submodule_state_still_reads_a_nested_layout():
    nested = {"estimator": {"latents": torch.zeros(2)}}
    assert set(pol.submodule_state(nested, "estimator")) == {"latents"}


def test_a_checkpoint_with_no_matching_weights_never_yields_a_scored_policy():
    """Silence here is the one failure mode that cannot be seen in the number."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "wrong.pt"
        torch.save({"model": {"totally.unrelated.key": torch.zeros(2)}}, p)
        mods, err = pol._try_real_modules(str(p), "libero_franka", "cpu")
        assert mods is None
        assert err is not None and "estimator" in str(err)
        with pytest.raises(RuntimeError):
            pol.load_policy(str(p), allow_stub=False)


def test_provenance_records_whether_stubs_ran():
    prov = pol.policy_provenance(pol.load_policy(None))
    assert prov["is_stub"] is True
    assert "zeros_featurizer" in prov["featurizer"]
    assert prov["resampler"] == "loom.data.canonical.to_env_rate"
    assert prov["env_steps_per_segment"] == C.env_steps_per_segment(20.0)


def test_results_json_records_the_policy_that_actually_ran(tmp_path):
    out = tmp_path / "r.json"
    runner.run_eval(tiny_protocol(), bench="libero", out=out, backend="fake")
    meta = json.loads(out.read_text())["meta"]
    assert "policy" in meta, "the results JSON must say which modules ran"
    assert meta["policy"]["is_stub"] is True


def test_a_named_checkpoint_never_degrades_to_stubs_by_default(tmp_path):
    """`--ckpt` means it. The stub path is opt-in once a checkpoint is named."""
    missing = tmp_path / "does_not_exist.pt"
    with pytest.raises(RuntimeError):
        pol.load_policy(str(missing))
    # still explicitly available for anyone who wants it
    p = pol.load_policy(str(missing), allow_stub=True)
    assert p.modules.is_stub
