#!/usr/bin/env python3
# v0.9.25f-a — extract 10 pure helpers to cadw/dataframe_helpers.py.
#
# DIAGNOSTIC INSIGHT (v0.9.25f):
#   - CadwDataFrame: 5960 L (94 methods, 5773 LOC method bodies)
#   - Module helpers: 12 (450 L)
#   - Helper classes: 6 (648 L) — 4 leaf + _ColumnProxy/_GroupBy
#   - 48 AST-guard tests scan inspect.getsource(cadw.dataframe)
#   - External imports: tests/ 6 sym, scripts/ 6 sym
#
# FULL SPLIT (7-10 modules) — REJECTED. R1 = HARD blocker:
#   After split, cadw/dataframe/__init__.py = 60 L; 48 AST guards
#   trivially PASS scanning only facade re-exports → SILENT coverage
#   loss for facade invariants (no as_py, no row loop, etc.).
#   Retargeting cost: 48 apply_or_fail (~5-8h) for 0 user value.
#
# v0.9.25f-a: PARTIAL extraction — 10 pure helpers, 433 L total.
#   No CadwDataFrame cross-refs. dataframe.py 7172 → ~6740 L.
#   AST guards intact (single-file dataframe.py still scanned).
#
# Extract:
#   Classes: _ArrowTail(45), _RollState(85), _ExpandingState(65),
#            _EwmState(48) = 243 L
#   Funcs:   _group_boundaries_arrow(41), _convert_batch(31),
#            _concat_batches(13), _empty_value_counts_table(13),
#            _reservoir_positions(41), _weighted_reservoir_positions(51)
#            = 190 L
#
# AST extraction (§13.4/253 per-container heuristic — applied properly):
#   Per helper: find def in dataframe.py AST (exact node), get
#   lineno..end_lineno, extract source, remove with PEP8 blank
#   normalization, append to helpers.py (compile-before-write both).
#   Not per-file transform, not string anchors.
#
# Backward compat: `from cadw.dataframe import _RollState` still
# works — `from cadw.dataframe_helpers import (...)` in dataframe.py
# brings names into module namespace. Same for scripts/tests imports.
#
# §13.4/144 (7th occurrence this session — PROACTIVE grep):
#   Checked all 48 AST guards — they scan per-method via
#   inspect.getsource(CadwDataFrame.method) or the whole
#   dataframe.py for facade invariants (no as_py, no row loop,
#   no to_pandas, delegated methods). None reference the 10
#   extracted helpers by name (helpers had no per-method guards
#   — they're not facade methods). Zero stale test updates needed.
#   Verified by test_helpers_not_defined_in_facade in new tests.
#
# New tests (tests/test_v0925f_a.py):
#   - test_helpers_defined_in_helpers_module (10 names in AST of
#     cadw/dataframe_helpers.py)
#   - test_helpers_re_exported_from_dataframe (importable from
#     cadw.dataframe — backward compat)
#   - test_helpers_not_defined_in_facade (no `class _RollState` /
#     `def _group_boundaries_arrow` in dataframe.py AST)
#   - test_helpers_module_no_cadwdataframe_dep (helpers.py AST
#     has no `from cadw.dataframe import` — pure module invariant)
#
# Expected cycles: 0-1. No circular imports (helpers.py imports
# only pyarrow/numpy/collections). If a helper has missed cross-ref,
# extract aborts with AST fail-fast on that specific helper → fix1.
#
# Run: uv run python scripts/patch_v0925f_a_extract_helpers.py

from __future__ import annotations
import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import (
    assert_compiles, commit_with_guards, run_verify_in_subprocess,
    check_no_invalid_escape,
)


HELPERS_PATH = "cadw/dataframe_helpers.py"
FACADE_PATH = "cadw/dataframe.py"

EXTRACT_CLASSES = [
    "_ArrowTail", "_RollState", "_ExpandingState", "_EwmState",
]
EXTRACT_FUNCS = [
    "_group_boundaries_arrow", "_convert_batch", "_concat_batches",
    "_empty_value_counts_table", "_reservoir_positions",
    "_weighted_reservoir_positions",
]
ALL_EXTRACTS = EXTRACT_CLASSES + EXTRACT_FUNCS


# ============ 0. verify state ============

def verify_state():
    print("\n=== verify: HEAD=v0.9.25f-diag-fix1; helpers in dataframe.py ===")
    log = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout
    if "v0.9.25f-diagnostic-fix1" not in log:
        raise SystemExit(f"expected HEAD=v0.9.25f-diagnostic-fix1; "
                          f"got {log[:200]!r}")
    if Path(HELPERS_PATH).exists():
        raise SystemExit(
            f"{HELPERS_PATH} already exists — f-a partially applied?"
        )
    src = Path(FACADE_PATH).read_text()
    tree = ast.parse(src)
    defs = {
        node.name for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef))
    }
    missing = [n for n in ALL_EXTRACTS if n not in defs]
    if missing:
        raise SystemExit(
            f"missing helpers in dataframe.py: {missing}"
        )
    print(f"  all {len(ALL_EXTRACTS)} helper defs present — OK")


# ============ 1. extract per-helper (AST, fail-fast) ============

HELPERS_HEADER = '''"""Pure helpers extracted from cadw/dataframe.py (v0.9.25f-a).

No CadwDataFrame dependency (pure module invariant — see
test_v0925f_a.py::test_helpers_module_no_cadwdataframe_dep).

Imported + re-exported by cadw.dataframe for backward compat:
`from cadw.dataframe import _RollState` still works.

Contents (10 items, 433 L total):
  - _ArrowTail (45 L): Arrow-native sliding tail buffer (v0.9.18).
  - _RollState (85 L): incremental rolling state (v0.6.2c).
  - _ExpandingState (65 L): cumulative window state (v0.8.0).
  - _EwmState (48 L): EWMA state (v0.8.0).
  - _group_boundaries_arrow (41 L): Arrow-native group boundaries
    (v0.9.21, §1.4).
  - _convert_batch (31 L): format='arrow'|'pandas'|'numpy'|'torch'.
  - _concat_batches (13 L): per-column concat (v0.9.10).
  - _empty_value_counts_table (13 L): empty schema (v0.9.23g-fix2).
  - _reservoir_positions (41 L): Algorithm R reservoir.
  - _weighted_reservoir_positions (51 L): Efraimidis-Spirakis A-Res.

This module can be imported independently (no cadw.dataframe dep);
useful for tests that want just the state classes without the full
facade.
"""
from __future__ import annotations

import pyarrow as pa


'''


def _extract_one(src: str, name: str, kind: str):
    """Return (new_src, extracted_source) or (src, None) if not found."""
    tree = ast.parse(src)
    target = None
    for node in tree.body:
        if kind == "class" and isinstance(node, ast.ClassDef) \
                and node.name == name:
            target = node
            break
        if kind == "func" and isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef),
        ) and node.name == name:
            target = node
            break
    if target is None:
        return src, None

    lines = src.split("\n")
    extracted = "\n".join(lines[target.lineno - 1:target.end_lineno])

    before = lines[:target.lineno - 1]
    after = lines[target.end_lineno:]
    while before and before[-1].strip() == "":
        before.pop()
    while after and after[0].strip() == "":
        after.pop(0)
    new_src = "\n".join(before + ["", ""] + after)
    return new_src, extracted


def apply_extract():
    print("\n=== extract helpers → cadw/dataframe_helpers.py ===")
    src = Path(FACADE_PATH).read_text()
    extracted_sources = []

    for name in EXTRACT_CLASSES:
        new_src, ext = _extract_one(src, name, "class")
        if ext is None:
            raise SystemExit(f"class {name!r} not found — already extracted?")
        src = new_src
        extracted_sources.append(ext)
        print(f"  class {name}: extracted")

    for name in EXTRACT_FUNCS:
        new_src, ext = _extract_one(src, name, "func")
        if ext is None:
            raise SystemExit(f"func {name!r} not found — already extracted?")
        src = new_src
        extracted_sources.append(ext)
        print(f"  func  {name}: extracted")

    helpers_src = HELPERS_HEADER + "\n\n\n".join(extracted_sources) + "\n"

    # Compile-before-write both (§13.4/211)
    try:
        ast.parse(helpers_src)
    except SyntaxError as e:
        raise SystemExit(f"helpers.py compile fail: {e}")

    # Insert import in dataframe.py after existing imports.
    # Anchor: last import line before class definitions. Use AST to
    # find the last ImportFrom at top level, insert after it.
    tree = ast.parse(src)
    last_import_line = 0
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            last_import_line = max(last_import_line, node.end_lineno)
    if last_import_line == 0:
        raise SystemExit("no imports found in dataframe.py — unexpected")

    import_block = (
        "\n"
        "from cadw.dataframe_helpers import (  # v0.9.25f-a\n"
        "    _ArrowTail, _RollState, _ExpandingState, _EwmState,\n"
        "    _group_boundaries_arrow, _convert_batch, _concat_batches,\n"
        "    _empty_value_counts_table, _reservoir_positions,\n"
        "    _weighted_reservoir_positions,\n"
        ")\n"
    )
    lines = src.split("\n")
    new_src = (
        "\n".join(lines[:last_import_line])
        + import_block
        + "\n".join(lines[last_import_line:])
    )
    try:
        ast.parse(new_src)
    except SyntaxError as e:
        raise SystemExit(f"dataframe.py compile fail after edit: {e}")

    Path(FACADE_PATH).write_text(new_src)
    Path(HELPERS_PATH).write_text(helpers_src)
    assert_compiles(FACADE_PATH)
    assert_compiles(HELPERS_PATH)
    check_no_invalid_escape(FACADE_PATH)
    check_no_invalid_escape(HELPERS_PATH)
    print(f"  wrote {HELPERS_PATH} ({len(helpers_src.splitlines())} L)")
    print(f"  edited {FACADE_PATH} (import block at line "
          f"{last_import_line})")


# ============ 2. new tests ============

NEW_TESTS = '''"""v0.9.25f-a: 10 pure helpers extracted to cadw/dataframe_helpers.py.

Phase B diagnostic revealed 48 AST guards block full split (silent
coverage loss). Partial extract: 433 L of pure helpers with no
CadwDataFrame cross-refs. dataframe.py 7172 → ~6740 L. AST guards
intact (single-file facade still scanned).
"""
import ast as _ast
import importlib as _importlib
import inspect as _inspect


HELPER_NAMES = [
    "_ArrowTail", "_RollState", "_ExpandingState", "_EwmState",
    "_group_boundaries_arrow", "_convert_batch", "_concat_batches",
    "_empty_value_counts_table", "_reservoir_positions",
    "_weighted_reservoir_positions",
]

HELPER_CLASSES = {
    "_ArrowTail", "_RollState", "_ExpandingState", "_EwmState",
}


def test_helpers_defined_in_helpers_module():
    """v0.9.25f-a: all 10 helpers defined in cadw/dataframe_helpers.py."""
    from cadw import dataframe_helpers as hm
    src = _inspect.getsource(hm)
    tree = _ast.parse(src)
    defs = {
        node.name for node in tree.body
        if isinstance(node, (_ast.ClassDef, _ast.FunctionDef,
                             _ast.AsyncFunctionDef))
    }
    missing = [n for n in HELPER_NAMES if n not in defs]
    assert not missing, (
        f"helpers missing from dataframe_helpers.py: {missing}"
    )


def test_helpers_re_exported_from_dataframe():
    """Backward compat: `from cadw.dataframe import _RollState` works."""
    import cadw.dataframe as df_mod
    for name in HELPER_NAMES:
        assert hasattr(df_mod, name), (
            f"cadw.dataframe missing re-export of {name!r} — "
            f"backward-compat broken (scripts/tests import these)"
        )


def test_helpers_not_defined_in_facade():
    """v0.9.25f-a belt-and-suspenders: helper definitions must NOT
    reappear in cadw/dataframe.py (would double-define and defeat
    extraction)."""
    import cadw.dataframe as df_mod
    src = _inspect.getsource(df_mod)
    tree = _ast.parse(src)
    for name in HELPER_NAMES:
        if name in HELPER_CLASSES:
            hits = [
                n for n in tree.body
                if isinstance(n, _ast.ClassDef) and n.name == name
            ]
        else:
            hits = [
                n for n in tree.body
                if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef))
                and n.name == name
            ]
        assert not hits, (
            f"{name!r} re-defined in cadw/dataframe.py — "
            f"extraction regressed. Move back to "
            f"cadw/dataframe_helpers.py or remove duplicate."
        )


def test_helpers_module_no_cadwdataframe_dep():
    """Pure module invariant: helpers.py imports no CadwDataFrame
    or cadw.dataframe (else circular import)."""
    from cadw import dataframe_helpers as hm
    src = _inspect.getsource(hm)
    tree = _ast.parse(src)
    for node in _ast.walk(tree):
        if isinstance(node, _ast.ImportFrom):
            assert node.module != "cadw.dataframe", (
                "helpers.py imports from cadw.dataframe — circular"
            )
            if node.module and "CadwDataFrame" in str(node.module):
                raise AssertionError(
                    f"helpers.py imports {node.module!r} — must be pure"
                )
'''

def add_tests():
    print("\n=== add tests/test_v0925f_a.py ===")
    p = Path("tests/test_v0925f_a.py")
    if p.exists() and "test_helpers_defined_in_helpers_module" in p.read_text():
        print("  already present — skip")
        return
    p.write_text(NEW_TESTS)
    assert_compiles(p)
    check_no_invalid_escape(p)
    print("  OK")


# ============ 3. dogfood ============

def dogfood():
    print("\n=== dogfood: 10 helpers importable + facade unchanged ===")

    def _verify():
        import ast as _ast
        import inspect as _inspect
        import cadw.dataframe as df_mod
        from cadw import dataframe_helpers as hm

        names = [
            "_ArrowTail", "_RollState", "_ExpandingState", "_EwmState",
            "_group_boundaries_arrow", "_convert_batch", "_concat_batches",
            "_empty_value_counts_table", "_reservoir_positions",
            "_weighted_reservoir_positions",
        ]
        for name in names:
            assert hasattr(hm, name), (
                f"helpers missing {name!r}"
            )
            assert hasattr(df_mod, name), (
                f"cadw.dataframe missing re-export of {name!r}"
            )

        # Facade no longer defines them
        df_src = _inspect.getsource(df_mod)
        df_tree = _ast.parse(df_src)
        defs_in_facade = {
            node.name for node in df_tree.body
            if isinstance(node, (_ast.ClassDef, _ast.FunctionDef,
                                  _ast.AsyncFunctionDef))
        }
        redefined = [n for n in names if n in defs_in_facade]
        assert not redefined, (
            f"helpers still defined in facade: {redefined}"
        )

        # Helpers module has no cadw.dataframe dep
        hm_src = _inspect.getsource(hm)
        hm_tree = _ast.parse(hm_src)
        for node in _ast.walk(hm_tree):
            if isinstance(node, _ast.ImportFrom):
                assert node.module != "cadw.dataframe", (
                    "helpers.py imports from cadw.dataframe"
                )

        # Sanity: use _RollState like old code did
        st = hm._RollState(2, 2, "sum")
        assert st.push(1.0) is None  # min_periods=2 not met
        assert st.push(2.0) == 3.0

        print(
            f"  10 helpers extracted + re-exported; facade clean; "
            f"_RollState smoke OK"
        )

    run_verify_in_subprocess(_verify)


# ============ 4. pytest ============

def run_pytest():
    print("\n=== pre-flight: 48 AST guards in dataframe.py ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_dataframe.py",
         "tests/test_dataframe_filters.py",
         "tests/test_v0925f_a.py",
         "-q", "-W", "error::SyntaxWarning"],
        timeout=180,
    )
    if r.returncode != 0:
        print("\n*** AST GUARDS OR v0925f_a FAILED — ABORT ***")
        sys.exit(1)

    print("\n=== pytest: full dataframe + helper consumers ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_v0925f_a.py",
         "tests/test_dataframe.py",
         "tests/test_dataframe_filters.py",
         "tests/test_dataframe_v094b.py",
         "tests/test_dataframe_v094b2.py",
         "tests/test_pylist_fixes.py",
         "tests/test_apply.py",
         "tests/test_cum_series.py",
         "tests/test_expanding_ewm.py",
         "tests/test_rolling.py",
         "tests/test_rolling_on_col.py",
         "tests/test_shift_diff.py",
         "tests/test_shift_on_col.py",
         "tests/test_groupby_per_group.py",
         "tests/test_eda_basics.py",
         "tests/test_sample.py",
         "-q", "-m", "not slow",
         "-W", "error::SyntaxWarning"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** TARGETED FAILURE ***")
        sys.exit(1)

    print("\n=== pytest fast (full) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)


# ============ 5. main ============

def main():
    print("=== v0.9.25f-a: partial extract (10 helpers → helpers.py) ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"  branch: {br}")

    verify_state()
    apply_extract()
    add_tests()
    dogfood()
    run_pytest()

    commit_with_guards(
        "v0.9.25f-a: partial extract — 10 pure helpers → "
        "cadw/dataframe_helpers.py (Phase B risk-mitigated). "
        "Phase B diagnostic (v0.9.25f) found 48 AST-guard tests "
        "scanning inspect.getsource(cadw.dataframe) — HARD blocker "
        "for full split (was estimated 4-6, actual 48). After split, "
        "cadw/dataframe/__init__.py = 60 L; 48 guards would trivially "
        "PASS scanning only facade re-exports → silent coverage loss "
        "for facade invariants (no as_py, no row loop, no to_pandas, "
        "delegated methods). Retargeting: 48 apply_or_fail (5-8h) for "
        "0 user value. Full split DEFERRED to v0.9.26. "
        "This patch: extract 10 helpers (433 L) with no CadwDataFrame "
        "cross-refs — 4 leaf state classes (_ArrowTail, _RollState, "
        "_ExpandingState, _EwmState) + 6 pure functions "
        "(_group_boundaries_arrow, _convert_batch, _concat_batches, "
        "_empty_value_counts_table, _reservoir_positions, "
        "_weighted_reservoir_positions). dataframe.py 7172 → ~6740 L. "
        "AST extraction per helper (§13.4/253 per-container heuristic "
        "applied properly: iterate per helper, not per file). "
        "Compile-before-write both files (§13.4/211). "
        "Backward compat: `from cadw.dataframe import _RollState` "
        "still works via `from cadw.dataframe_helpers import (...)` "
        "in dataframe.py (6 external imports from tests/ + 6 from "
        "scripts/ preserved). "
        "AST guards intact (§13.4/144 7th occurrence — PROACTIVE): "
        "checked all 48 — they scan per-method (inspect.getsource("
        "CadwDataFrame.method)) or whole-file for facade invariants. "
        "None reference the 10 extracted helpers by name. Zero stale "
        "test updates. "
        "New tests/test_v0925f_a.py: 4 — helpers defined in "
        "helpers.py, re-exported from dataframe, NOT re-defined in "
        "facade (belt-and-suspenders), helpers.py no cadw.dataframe "
        "dep (pure module). "
        "Full split roadmap: v0.9.26, single-purpose session — "
        "retarget 48 AST guards on rglob, then split. "
        "Next: Phase C (Pandas P0 — 20 features, HIGH user value). "
        "Fast suite: ~2306 → ~2310 (+4).",
        self_path="scripts/patch_v0925f_a_extract_helpers.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.25f-a applied. Phase B deferred to v0.9.26. "
          "Next: Phase C — Pandas P0.")


if __name__ == "__main__":
    main()