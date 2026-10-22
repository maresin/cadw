#!/usr/bin/env python3
# v0.9.25f-a-fix1 — missing `import numpy as np` in extracted helpers.
#
# BUG: v0.9.25f-a extracted 10 helpers to cadw/dataframe_helpers.py.
# Header had `import pyarrow as pa` only. But `_group_boundaries_arrow`
# uses `np.flatnonzero` (line 321); in the original dataframe.py `np`
# was available from module-level `import numpy as np`. Extraction
# missed it.
#
# 37 tests failing:
#   - All test_cum_series.py + test_groupby_per_group.py per-group
#     ops (shift/diff/rolling/cum*) — they call _group_boundaries_arrow
#     transitively via _per_group_shift/_per_group_rolling/_per_group_
#     cumulative.
#   - test_group_boundaries_arrow_correctness — direct call.
# All NameError: name 'np' is not defined at dataframe_helpers.py:321.
#
# Verified 9 other helpers manually (import audit):
#   - _ArrowTail: only `pa` ✓
#   - _RollState: `from collections import deque` local in __init__ ✓
#   - _ExpandingState, _EwmState: pure floats ✓
#   - _convert_batch, _concat_batches: `pa` only ✓
#   - _empty_value_counts_table: `import pyarrow as _pa` local ✓
#   - _reservoir_positions, _weighted_reservoir_positions:
#     `import numpy as np` local in function body ✓
# Only `_group_boundaries_arrow` needs module-level numpy.
#
# WHY dogfood in v0.9.25f-a PASSED despite bug (§13.4/257 NEW):
#   v0.9.25f-a dogfood checked `hasattr(hm, name)` (10 imports) + smoke
#   `_RollState` push (2 calls). `_group_boundaries_arrow` not smoke-
#   tested. `_RollState` doesn't use numpy → dogfood green. pytest
#   caught it (37 failures) → verifies AST guards + helper consumers.
#
# Lesson §13.4/257: extracted module needs runtime smoke test of
# EVERY extracted symbol, not just import-check + 1 sample. Otherwise
# missing module-level imports stay invisible until consumer path
# hits the specific helper. Fix1 adds smoke test for
# `_group_boundaries_arrow` (the actual failure mode) + AST import
# guard test (prevents future missing imports in extracted modules).
#
# STATE: HEAD = v0.9.25f-diagnostic-fix1 (v0.9.25f-a NOT committed —
# pytest failed). On-disk: extract applied, helpers.py missing numpy
# import, dataframe.py has import block, tests/test_v0925f_a.py
# exists (4 passing tests; production bug out of their scope).
#
# Fix scope: 1-line production edit (add `import numpy as np`) +
# 2 regression tests. Single commit v0.9.25f-a (extract + fix1).
#
# Run: uv run python scripts/patch_v0925f_a_fix1_numpy.py

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
    apply_or_fail, assert_compiles, commit_with_guards,
    run_verify_in_subprocess, check_no_invalid_escape,
)


HELPERS_PATH = "cadw/dataframe_helpers.py"
FACADE_PATH = "cadw/dataframe.py"
TESTS_PATH = "tests/test_v0925f_a.py"


# ============ 0. verify state ============

def verify_state():
    print("\n=== verify: HEAD=v0.9.25f-diagnostic-fix1; "
          "helpers.py missing numpy ===")
    log = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout
    if "v0.9.25f-diagnostic-fix1" not in log:
        raise SystemExit(
            f"expected HEAD=v0.9.25f-diagnostic-fix1; got {log[:200]!r}"
        )
    if "v0.9.25f-a" in log:
        raise SystemExit("v0.9.25f-a already committed — fix1 misplaced")

    hp = Path(HELPERS_PATH)
    if not hp.exists():
        raise SystemExit(
            f"{HELPERS_PATH} missing — v0.9.25f-a extract not applied?"
        )
    h_src = hp.read_text()
    if "import numpy as np" in h_src:
        raise SystemExit(
            "helpers.py already has `import numpy as np` — fix1 misplaced"
        )
    if "import pyarrow as pa" not in h_src:
        raise SystemExit("helpers.py missing `import pyarrow as pa`?!")

    # Extract was applied (import block in dataframe.py)
    f_src = Path(FACADE_PATH).read_text()
    if "from cadw.dataframe_helpers import" not in f_src:
        raise SystemExit(
            "dataframe.py missing import block from helpers — "
            "v0.9.25f-a extract not applied?"
        )

    # tests/test_v0925f_a.py exists
    if not Path(TESTS_PATH).exists():
        raise SystemExit(f"{TESTS_PATH} missing — v0.9.25f-a add_tests?")

    print("  state matches v0.9.25f-a uncommitted; helpers.py lacks numpy — OK")


# ============ 1. add numpy import to helpers header ============

OLD_HEADER = (
    "from __future__ import annotations\n"
    "\n"
    "import pyarrow as pa\n"
)

NEW_HEADER = (
    "from __future__ import annotations\n"
    "\n"
    "import numpy as np\n"
    "import pyarrow as pa\n"
)


def add_numpy_import():
    print("\n=== add `import numpy as np` to helpers header ===")
    applied = apply_or_fail(
        HELPERS_PATH, OLD_HEADER, NEW_HEADER,
        label="numpy_import",
    )
    if applied:
        assert_compiles(HELPERS_PATH)
        check_no_invalid_escape(HELPERS_PATH)
    print("  applied" if applied else "  already applied — skip")


# ============ 2. add regression tests ============

NEW_TESTS = '''

def test_group_boundaries_arrow_smoke():
    """v0.9.25f-a-fix1 (§13.4/257): runtime smoke test of every extracted
    symbol. Prior dogfood only tested _RollState (numpy-free) — missed
    that _group_boundaries_arrow needs `import numpy as np` at module
    level. This directly exercises the bug.
    """
    import pyarrow as pa
    from cadw.dataframe_helpers import _group_boundaries_arrow
    b = pa.RecordBatch.from_arrays(
        [pa.array([1, 1, 2, 2, 2, 3])], names=["g"],
    )
    assert _group_boundaries_arrow(b, ["g"]) == [0, 2, 5, 6]


def test_helpers_module_has_required_imports():
    """v0.9.25f-a-fix1: AST check — module-level imports cover all
    Name loads inside helper defs (excluding locals + builtins).

    Prevents missing module-level imports like the missing numpy
    that v0.9.25f-a shipped. `_group_boundaries_arrow` used np.
    flatnonzero; helpers.py had no numpy import -> NameError at
    runtime.
    """
    import ast as _ast
    import builtins as _builtins
    import inspect as _inspect
    from cadw import dataframe_helpers as hm

    src = _inspect.getsource(hm)
    tree = _ast.parse(src)

    module_imports: set = set()
    for node in tree.body:
        if isinstance(node, _ast.Import):
            for a in node.names:
                module_imports.add(a.asname or a.name.split(".")[0])
        elif isinstance(node, _ast.ImportFrom):
            for a in node.names:
                module_imports.add(a.asname or a.name)

    module_defs = {
        node.name for node in tree.body
        if isinstance(node, (_ast.ClassDef, _ast.FunctionDef,
                              _ast.AsyncFunctionDef))
    }

    builtin_names = set(dir(_builtins))

    missing_by_fn: dict = {}
    for fn in [
        n for n in tree.body
        if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef))
    ]:
        local: set = set()
        for a in fn.args.args:
            local.add(a.arg)
        for a in getattr(fn.args, "posonlyargs", ()):
            local.add(a.arg)
        for a in getattr(fn.args, "kwonlyargs", ()):
            local.add(a.arg)
        if fn.args.vararg:
            local.add(fn.args.vararg.arg)
        if fn.args.kwarg:
            local.add(fn.args.kwarg.arg)

        for n in _ast.walk(fn):
            if isinstance(n, _ast.Assign):
                for t in n.targets:
                    for nn in _ast.walk(t):
                        if isinstance(nn, _ast.Name):
                            local.add(nn.id)
            elif isinstance(n, _ast.Import):
                for a in n.names:
                    local.add(a.asname or a.name.split(".")[0])
            elif isinstance(n, _ast.ImportFrom):
                for a in n.names:
                    local.add(a.asname or a.name)
            elif isinstance(n, (_ast.For, _ast.AsyncFor)):
                for nn in _ast.walk(n.target):
                    if isinstance(nn, _ast.Name):
                        local.add(nn.id)
            elif isinstance(n, _ast.withitem):
                if n.optional_vars:
                    for nn in _ast.walk(n.optional_vars):
                        if isinstance(nn, _ast.Name):
                            local.add(nn.id)

        missing = []
        for n in _ast.walk(fn):
            if not isinstance(n, _ast.Name):
                continue
            if not isinstance(n.ctx, _ast.Load):
                continue
            if n.id in local:
                continue
            if n.id in module_imports:
                continue
            if n.id in module_defs:
                continue
            if n.id in builtin_names:
                continue
            missing.append(n.id)
        if missing:
            missing_by_fn[fn.name] = sorted(set(missing))

    assert not missing_by_fn, (
        f"helpers.py functions reference names without module-level "
        f"imports (would NameError at runtime): {missing_by_fn}. "
        f"Add missing imports to cadw/dataframe_helpers.py header. "
        f"§13.4/257: extracted modules need imports that were "
        f"module-level in the source file."
    )
'''


def add_tests():
    print("\n=== add regression tests ===")
    p = Path(TESTS_PATH)
    s = p.read_text()
    if "test_group_boundaries_arrow_smoke" in s:
        print("  already present — skip")
        return
    p.write_text(s + NEW_TESTS)
    assert_compiles(p)
    check_no_invalid_escape(p)
    print("  OK")


# ============ 3. dogfood ============

def dogfood():
    print("\n=== dogfood: all 10 helpers smoke + boundaries works ===")

    def _verify():
        import ast as _ast
        import inspect as _inspect
        import pyarrow as pa
        from cadw import dataframe_helpers as hm

        # 1. Runtime smoke: all 4 state classes + 1 boundaries fn
        st = hm._RollState(2, 2, "sum")
        assert st.push(1.0) is None
        assert st.push(2.0) == 3.0

        exp = hm._ExpandingState("sum", 1)
        assert exp.push(5.0) == 5.0
        assert exp.push(3.0) == 8.0

        ewm = hm._EwmState(0.5, False, 1)
        assert ewm.push(1.0) == 1.0
        assert ewm.push(2.0) == 1.5

        at = hm._ArrowTail(2)
        at.push(pa.array([1, 2, 3]))
        tail = at.get()
        assert tail.to_pylist() == [2, 3]

        # **The buggy one**: _group_boundaries_arrow uses numpy
        b = pa.RecordBatch.from_arrays(
            [pa.array([1, 1, 2, 2, 2, 3])], names=["g"],
        )
        got = hm._group_boundaries_arrow(b, ["g"])
        assert got == [0, 2, 5, 6], (
            f"_group_boundaries_arrow returned {got}, expected [0,2,5,6]"
        )

        # 2. Every helper has runtime smoke OR pure function test above.
        #    Remaining (convert_batch, concat_batches, empty_value_counts,
        #    reservoir_positions, weighted_reservoir_positions) are
        #    tested via their consumers in test_dataframe*.py + test_*
        #    (transitive coverage) — not skipped, just not duplicated
        #    in dogfood.

        # 3. AST import guard: no missing module-level names
        src = _inspect.getsource(hm)
        tree = _ast.parse(src)
        module_imports = set()
        for node in tree.body:
            if isinstance(node, _ast.Import):
                for a in node.names:
                    module_imports.add(a.asname or a.name.split(".")[0])
            elif isinstance(node, _ast.ImportFrom):
                for a in node.names:
                    module_imports.add(a.asname or a.name)
        assert "np" in module_imports, (
            "numpy not in module-level imports of helpers.py"
        )

        print(
            "  10 helpers defined; _RollState/_ExpandingState/_EwmState/"
            "_ArrowTail/_group_boundaries_arrow all smoke-pass; "
            "numpy import present"
        )

    run_verify_in_subprocess(_verify)


# ============ 4. pytest ============

def run_pytest():
    print("\n=== pre-flight: AST guards (48 tests scan dataframe.py) ===")
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

    print("\n=== pytest: full targeted (cum + per_group + dataframe) ===")
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
         "tests/test_metrics.py",
         "tests/test_metrics_e2e.py",
         "tests/test_ts_api.py",
         "tests/test_ts_features.py",
         "tests/test_ts_stats.py",
         "tests/test_ts_models.py",
         "tests/test_ts_thf.py",
         "tests/test_ts_backtest.py",
         "tests/test_ts_readme_example.py",
         "tests/test_ts_stage8.py",
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
    print("=== v0.9.25f-a-fix1: add numpy import + regression guards ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"  branch: {br}")

    verify_state()
    add_numpy_import()
    add_tests()
    dogfood()
    run_pytest()

    commit_with_guards(
        "v0.9.25f-a: partial extract — 10 pure helpers → "
        "cadw/dataframe_helpers.py + fix1 (missing numpy). "
        "Phase B diagnostic (v0.9.25f) found 48 AST-guard tests "
        "scanning inspect.getsource(cadw.dataframe) — HARD blocker "
        "for full split (was estimated 4-6, actual 48). Full split "
        "DEFERRED to v0.9.26 with retargeting plan. "
        "This patch: extract 10 helpers (433 L) with no CadwDataFrame "
        "cross-refs — 4 leaf state classes (_ArrowTail, _RollState, "
        "_ExpandingState, _EwmState) + 6 pure functions "
        "(_group_boundaries_arrow, _convert_batch, _concat_batches, "
        "_empty_value_counts_table, _reservoir_positions, "
        "_weighted_reservoir_positions). dataframe.py 7172 → ~6740 L. "
        "AST extraction per helper (§13.4/253 applied properly: "
        "iterate per helper, not per file). Compile-before-write "
        "both files (§13.4/211). Backward compat: `from "
        "cadw.dataframe import _RollState` still works via "
        "`from cadw.dataframe_helpers import (...)` re-export "
        "(6 external imports from tests/ + 6 from scripts/ preserved). "
        "AST guards intact (§13.4/144 7th occurrence — PROACTIVE): "
        "all 48 scan per-method or whole-file facade invariants; "
        "none reference the 10 extracted helpers by name. Zero stale "
        "test updates. "
        "fix1: my extraction header had `import pyarrow as pa` only; "
        "`_group_boundaries_arrow` uses `np.flatnonzero`. In original "
        "dataframe.py, numpy was available from module-level import; "
        "extraction missed it. 37 tests failed (all per-group shift/"
        "diff/rolling/cum* + direct boundary test) with NameError: "
        "name 'np' is not defined at dataframe_helpers.py:321. "
        "Verified 9 other helpers manually — only _group_boundaries_"
        "arrow needs module-level numpy (_RollState has local deque "
        "import; _reservoir_positions + _weighted_reservoir_positions "
        "have local numpy; _ArrowTail only pa). "
        "§13.4/257 (NEW): extracted module needs runtime smoke test of "
        "EVERY extracted symbol, not just import-check + 1 sample. "
        "v0.9.25f-a dogfood tested _RollState (numpy-free) only; "
        "_group_boundaries_arrow not exercised -> bug invisible to "
        "dogfood, caught by pytest. Fix1 adds: (a) runtime smoke "
        "test of _group_boundaries_arrow, (b) AST import guard "
        "test_helpers_module_has_required_imports — checks every "
        "function body's Name loads against module-level imports + "
        "locals + builtins, prevents future extraction imports drift. "
        "New tests/test_v0925f_a.py: 6 total (4 original + 2 fix1). "
        "Full split roadmap: v0.9.26, single-purpose session — "
        "retarget 48 AST guards via _all_dataframe_sources() helper + "
        "meta-guard test_all_ast_guards_use_rglob, then split. "
        "Next: Phase C (Pandas P0 — 20 features, HIGH user value). "
        "Fast suite: ~2306 → ~2312 (+6 new tests in v0925f_a.py).",
        self_path="scripts/patch_v0925f_a_fix1_numpy.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.25f-a COMPLETE (extract + fix1). "
          "Phase B deferred to v0.9.26. Next: Phase C — Pandas P0.")


if __name__ == "__main__":
    main()