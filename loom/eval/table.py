"""LOOM — markdown emitters, in PLAN 8 column order so numbers paste directly.

Three tables, three sources, one rule:

**Baseline rows are copied verbatim from a single source per table and are
never re-run and never assembled across papers.** Fast-WAM's LIBERO average
appears in print as 97.6, 97.0 and 97.60 in three different papers; a table
stitched from three sources is not a comparison. Every baseline row below sits
in a constants block that names its one source, and only the `LOOM ...` rows
are ever filled from measurement.

The header strings are literals, asserted character-for-character by
`tests/test_eval.py`, because the whole point is that the emitted block can be
pasted into PLAN 8 without editing.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from loom.eval import EvalProtocol

__all__ = [
    "LIBERO_HEADER", "ROBOTWIN_HEADER", "LIBERO_PLUS_HEADER",
    "LIBERO_SOURCE", "ROBOTWIN_SOURCE", "LIBERO_PLUS_SOURCE",
    "LIBERO_BASELINES", "ROBOTWIN_BASELINES", "LIBERO_PLUS_BASELINES",
    "LIBERO_COLUMNS", "ROBOTWIN_COLUMNS", "LIBERO_PLUS_COLUMNS",
    "geo_avg", "fmt",
    "libero_table", "robotwin_table", "libero_plus_table",
    "libero_row_from_results", "render_report",
]


# ═══════════════════════════════════════════════════════════════════════════
#  COLUMN ORDER  —  PLAN 8, verbatim
# ═══════════════════════════════════════════════════════════════════════════

LIBERO_COLUMNS = ("method", "params", "emb. PT", "spatial", "object", "goal",
                  "long", "avg")

ROBOTWIN_COLUMNS = ("method", "clean", "rand", "hanging mug", "turn switch",
                    "place can basket", "handover block")

LIBERO_PLUS_COLUMNS = ("method", "camera", "robot init", "layout", "geo avg",
                       "light", "backgnd", "language", "noise", "total")


def _header(cols: Sequence[str]) -> str:
    return "| " + " | ".join(cols) + " |"


LIBERO_HEADER = _header(LIBERO_COLUMNS)
ROBOTWIN_HEADER = _header(ROBOTWIN_COLUMNS)
LIBERO_PLUS_HEADER = _header(LIBERO_PLUS_COLUMNS)


# ═══════════════════════════════════════════════════════════════════════════
#  BASELINES  —  ONE source per table. Do not add a row from another paper.
# ═══════════════════════════════════════════════════════════════════════════

LIBERO_SOURCE = "Light-WAM Table 1"
ROBOTWIN_SOURCE = "Fast-WAM Table 1 + per-task appendix (randomized column)"
LIBERO_PLUS_SOURCE = "OA-WAM Table 2"

#: method, params, emb. PT, spatial, object, goal, long, avg   (Light-WAM Table 1)
LIBERO_BASELINES: tuple[tuple[str, ...], ...] = (
    ("Diffusion Policy", "—", "✗", "78.3", "92.5", "68.3", "50.5", "72.4"),
    ("OpenVLA",          "7 B", "✓", "84.7", "88.4", "79.2", "53.7", "76.5"),
    ("π0",               "3 B", "✓", "96.8", "98.8", "95.8", "85.2", "94.1"),
    ("VLA-Adapter",      "0.6 B", "✗", "96.0", "96.8", "97.4", "94.4", "96.2"),
    ("π0.5",             "3 B", "✓", "98.8", "98.2", "98.0", "92.4", "96.9"),
    ("Fast-WAM",         "6 B", "✗", "97.0", "99.4", "96.6", "94.8", "97.0"),
    ("Motus",            "8 B", "✓", "96.8", "99.8", "96.6", "97.6", "97.7"),
    ("LingBot-VA",       "5.3 B", "✓", "98.5", "99.6", "97.2", "98.5", "98.5"),
)

#: method, clean, rand, hanging mug, turn switch, place can basket, handover block
ROBOTWIN_BASELINES: tuple[tuple[str, ...], ...] = (
    ("π0",         "65.9", "58.4", "—", "—", "—", "—"),
    ("π0.5",       "82.7", "76.8", "17", "54", "62", "57"),
    ("Motus",      "88.7", "87.0", "38", "78", "76", "73"),
    ("Fast-WAM",   "91.9", "91.8", "62", "59", "69", "81"),
    ("LingBot-VA", "92.9", "91.5", "28", "45", "84", "78"),
)

#: method, camera, robot init, layout, geo avg, light, backgnd, language, noise, total
LIBERO_PLUS_BASELINES: tuple[tuple[str, ...], ...] = (
    ("HoloBrain-0",    "65.5", "58.2", "79.5", "67.7", "88.1", "90.3", "78.7", "66.9", "74.0"),
    ("GE-Act",         "60.7", "77.0", "80.2", "72.6", "95.8", "86.0", "77.4", "90.9", "80.3"),
    ("π0.5",           "—", "—", "—", "79.5", "—", "—", "—", "—", "—"),
    ("Cosmos-Policy",  "75.8", "63.3", "82.2", "73.8", "96.5", "88.9", "81.7", "92.7", "82.2"),
    ("OA-WAM",         "80.5", "89.6", "82.8", "84.3", "96.5", "95.9", "85.3", "75.6", "83.9"),
)

#: The LOOM rows PLAN 8 leaves blank, with the fixed cells it already states.
LIBERO_LOOM_ROWS = (
    ("**LOOM · R0-A**", "0.3 B", "✗"),
    ("**LOOM · R2**",   "0.3 B", "✓"),
)
ROBOTWIN_LOOM_ROWS = ("**LOOM · R0-B**", "**LOOM · R2**", "**LOOM · R3**")
LIBERO_PLUS_LOOM_ROWS = ("**LOOM · R2**", "**LOOM · R3**")


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def fmt(v: Any, nd: int = 1) -> str:
    """One decimal, or an empty cell when unmeasured. Never invents a number."""
    if v is None or v == "":
        return ""
    if isinstance(v, str):
        return v
    return f"{float(v):.{nd}f}"


def geo_avg(camera: float, robot_init: float, layout: float) -> float:
    """LIBERO-Plus Geo Avg = mean of camera / robot-init / layout (OA-WAM Table 2)."""
    return (float(camera) + float(robot_init) + float(layout)) / 3.0


def _rows_to_md(cols: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    lines = [_header(cols), "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        cells = list(r) + [""] * (len(cols) - len(r))
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    return "\n".join(lines)


def _measured(row: Mapping[str, Any] | None, key: str) -> str:
    return fmt(row.get(key)) if row else ""


# ═══════════════════════════════════════════════════════════════════════════
#  TABLE 1  —  STANDARD LIBERO
# ═══════════════════════════════════════════════════════════════════════════

def libero_table(measured: Mapping[str, Mapping[str, Any]] | None = None) -> str:
    """`measured` maps a LOOM row label -> {spatial, object, goal, long[, avg]}.

    `avg` is the mean of the four suites when not supplied. Baselines are
    carried verbatim from `LIBERO_BASELINES` and are never recomputed.
    """
    measured = measured or {}
    rows: list[Sequence[str]] = [list(r) for r in LIBERO_BASELINES]
    for label, params, pt in LIBERO_LOOM_ROWS:
        m = measured.get(label) or measured.get(label.strip("*"))
        suites = ["spatial", "object", "goal", "long"]
        cells = [_measured(m, s) for s in suites]
        avg = ""
        if m is not None:
            if m.get("avg") is not None:
                avg = fmt(m["avg"])
            else:
                vals = [m.get(s) for s in suites]
                if all(v is not None for v in vals):
                    avg = fmt(sum(float(v) for v in vals) / len(vals))
        rows.append([label, params, pt, *cells, avg])
    return _rows_to_md(LIBERO_COLUMNS, rows)


def libero_row_from_results(results: Mapping[str, Any]) -> dict[str, Any]:
    """Results JSON -> the four suite cells of a LIBERO row, in PLAN 8 order.

    `libero_long` and `libero_10` are the same suite; the column is `long`.
    """
    per_suite = results.get("summary", {}).get("per_suite", {})

    def get(*names: str) -> float | None:
        for n in names:
            if n in per_suite:
                return per_suite[n]["success_rate"]
        return None

    row = {
        "spatial": get("libero_spatial"),
        "object": get("libero_object"),
        "goal": get("libero_goal"),
        "long": get("libero_long", "libero_10"),
    }
    vals = [v for v in row.values() if v is not None]
    row["avg"] = sum(vals) / len(vals) if len(vals) == 4 else None
    return row


# ═══════════════════════════════════════════════════════════════════════════
#  TABLE 2  —  ROBOTWIN 2.0
# ═══════════════════════════════════════════════════════════════════════════

def robotwin_table(measured: Mapping[str, Mapping[str, Any]] | None = None) -> str:
    measured = measured or {}
    keys = ("clean", "rand", "hanging mug", "turn switch",
            "place can basket", "handover block")
    rows: list[Sequence[str]] = [list(r) for r in ROBOTWIN_BASELINES]
    for label in ROBOTWIN_LOOM_ROWS:
        m = measured.get(label) or measured.get(label.strip("*"))
        rows.append([label, *[_measured(m, k) for k in keys]])
    return _rows_to_md(ROBOTWIN_COLUMNS, rows)


# ═══════════════════════════════════════════════════════════════════════════
#  TABLE 3  —  LIBERO-PLUS
# ═══════════════════════════════════════════════════════════════════════════

def libero_plus_table(measured: Mapping[str, Mapping[str, Any]] | None = None) -> str:
    """`geo avg` is computed here as the mean of camera/robot-init/layout.

    It is never read from `measured` unless all three components are missing,
    so a hand-typed geo avg can never disagree with its own columns.
    """
    measured = measured or {}
    keys = ("camera", "robot init", "layout", "geo avg", "light", "backgnd",
            "language", "noise", "total")
    rows: list[Sequence[str]] = [list(r) for r in LIBERO_PLUS_BASELINES]
    for label in LIBERO_PLUS_LOOM_ROWS:
        m = dict(measured.get(label) or measured.get(label.strip("*")) or {})
        trio = [m.get("camera"), m.get("robot init"), m.get("layout")]
        if all(v is not None for v in trio):
            m["geo avg"] = geo_avg(*trio)
        rows.append([label, *[_measured(m, k) for k in keys]])
    return _rows_to_md(LIBERO_PLUS_COLUMNS, rows)


# ═══════════════════════════════════════════════════════════════════════════
#  REPORT
# ═══════════════════════════════════════════════════════════════════════════

def protocol_block(protocol: EvalProtocol, results: Mapping[str, Any] | None = None) -> str:
    """The protocol, stated inline above every table. PLAN 4.F requires this."""
    lines = [f"**Protocol** (`{protocol.bench}`): {protocol.describe()}."]
    if protocol.notes:
        lines.append(f"<sub>{protocol.notes}</sub>")
    if results is not None:
        s = results.get("summary", {})
        meta = results.get("meta", {})
        env = "real LIBERO" if meta.get("libero_available") else "FakeLiberoEnv (no LIBERO installed)"
        lines.append(
            f"<sub>Ran {s.get('n_episodes', 0)}/{s.get('n_expected', 0)} episodes, "
            f"{s.get('n_errors', 0)} crashed, {s.get('n_hit_step_cap', 0)} hit the "
            f"{protocol.max_steps}-step cap, mean episode length "
            f"{s.get('mean_episode_len', 0.0):.1f}. env: {env}. "
            f"ckpt: {meta.get('ckpt')}.</sub>"
        )
    return "\n\n".join(lines)


def suite_detail_table(results: Mapping[str, Any]) -> str:
    """Per-suite diagnostics. Not a paper table — the numbers behind the row."""
    cols = ("suite", "success", "episodes", "crashed", "hit cap", "mean len", "per-seed")
    rows = []
    for suite, d in results.get("summary", {}).get("per_suite", {}).items():
        per_seed = " / ".join(f"{k}:{v:.1f}" for k, v in d.get("per_seed", {}).items())
        rows.append([suite, fmt(d["success_rate"]), str(d["n_episodes"]),
                     str(d["n_errors"]), str(d["n_hit_step_cap"]),
                     f"{d['mean_episode_len']:.1f}", per_seed])
    return _rows_to_md(cols, rows)


def render_report(results: Mapping[str, Any], *, row_label: str = "**LOOM · R0-A**") -> str:
    """The full markdown block: protocol, the PLAN 8 table, then the diagnostics."""
    protocol = EvalProtocol.from_dict(results.get("protocol", {}))
    bench = results.get("bench", protocol.bench)

    parts = [f"## LOOM — {bench} results", protocol_block(protocol, results)]
    if bench == "libero":
        parts += [
            f"**Standard LIBERO** (baselines source: {LIBERO_SOURCE} — carried verbatim, "
            f"never re-run)",
            libero_table({row_label: libero_row_from_results(results)}),
        ]
    elif bench == "robotwin":
        parts += [
            f"**RoboTwin 2.0** (baselines source: {ROBOTWIN_SOURCE})",
            robotwin_table(),
        ]
    else:
        parts += [
            f"**LIBERO-Plus** (baselines source: {LIBERO_PLUS_SOURCE}; "
            f"geo avg = mean of camera/robot-init/layout)",
            libero_plus_table(),
        ]
    parts += ["**Per-suite detail**", suite_detail_table(results)]
    return "\n\n".join(parts) + "\n"
