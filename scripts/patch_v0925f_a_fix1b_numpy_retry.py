#!/usr/bin/env python3
# v0.9.25f-a-fix1b — retry of fix1 with corrected verify_state.
#
# STATE: HEAD = v0.9.25f-diagnostic-fix1. v0.9.25f-a NOT committed:
#   - extract applied (10 helpers → cadw/dataframe_helpers.py)
#   - helpers.py header = `import pyarrow as pa` only (no module-level
#     numpy)
#   - `_group_boundaries_arrow` uses `np.flatnonzero` -> NameError at
#     runtime (37 tests fail: all per-group shift/diff/rolling/cum*
#     + direct boundary test)
#   - two helpers (`_reservoir_positions`, `_weighted_reservoir_
#     positions`) have function-local `import numpy as np` (from
#     extract — correct, defs self-contained)
#   - tests/test_v0925f_a.py (4 tests, passing)
#
# BUG IN fix1's verify_state: checked `"import numpy as np" in
# helpers_src` — substring. This matches:
#   (a) module-level import (what fix1 wanted to detect as "already
#       applied" → fix1 misplaced)
#   (b) function-local import (what v0.9.25f-a's extract correctly
#       produced for 2 helpers)
#   (c) docstring mentions
# Result: verify_state saw (b) → false-positive "already has numpy" →
# SystemExit BEFORE any mutation. fix1 didn't run. State unchanged.
#
# §13.4/258 (NEW): "module-level import present/absent" as substring
# check is wrong — function-local imports and docstrings match too.
# Rule: AST-based module-level check (iterate tree.body, filter
# ast.Import / ast.ImportFrom, build set of names, test membership).
# Substring `"import X" in src` is only valid when src has no other
# occurrences of that text — never true for arbitrary code.
#
# Fix scope: only verify_state (AST-based module-level check).
# Everything else in fix1 (add numpy header, add 2 regression tests,
# dogfood, pytest) is correct and unchanged.
#
# Fix-cycle count v0.9.25f-a: extract(1) + fix1(verify bug, no
# mutation, catches pre-mutation) + fix1b(real fix) = 2 effective
# cycles, budget 2. Zero bad-commits (fix1's own verify_state was
# the gate that caught my fix1 verify bug — meta-safety).
#
# Run: uv run python scripts/patch_v0925f_a_fix1b_numpy_retry.py

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


# ============ 0. verify state (AST module-level check) ============

def _module_level_imports(src: str) -> set:
    """Names imported at module level (top-level Import/ImportFrom)."""
    tree = ast.parse(src)
    out = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for a in node.names:
                out.add(a.asname or a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                out.add(a.asname or a.name)
    return out


def verify_state():
    print("\n=== verify: HEAD=v0.9.25f-diagnostic-fix1; "
          "helpers.py missing MODULE-LEVEL numpy ===")
    log = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout
    if "v0.9.25f-diagnostic-fix1" not in log:
        raise SystemExit(
            f"expected HEAD=v0.9.25f-diagnostic-fix1; got {log[:200]!r}"
        )
    if "v0.9.25f-a" in log:
        raise SystemExit("v0.9.25f-a already committed — fix1b misplaced")

    hp = Path(HELPERS_PATH)
    if not hp.exists():
        raise SystemExit(
            f"{HELPERS_PATH} missing — v0.9.25f-a extract not applied?"
        )
    h_src = hp.read_text()

    # AST module-level check (fix1's bug was substring check here)
    module_imports = _module_level_imports(h_src)
    if "np" in module_imports:
        raise SystemExit(
            "helpers.py already has MODULE-LEVEL `import numpy as np` — "
            "fix1b misplaced (v0.9.25f-a extract must have run without "
            "numpy; if user manually added numpy, state diverged)"
        )
    if "pa" not in module_imports:
        raise SystemExit(
            "helpers.py missing MODULE-LEVEL `import pyarrow as pa`?!"
        )

    # Sanity: np IS used somewhere (verify bug exists to fix)
    tree = ast.parse(h_src)
    uses_np = any(
        isinstance(n, ast.Name) and n.id == "np"
        for n in ast.walk(tree)
    )
    if not uses_np:
        raise SystemExit(
            "helpers.py doesn't reference `np` at all — the reported "
            "bug (NameError at _group_boundaries_arrow) doesn't match "
            "current file. Inspect manually."
        )

    # Extract was applied (import block in dataframe.py)
    f_src = Path(FACADE_PATH).read_text()
    if "from cadw.dataframe_helpers import" not in f_src:
        raise SystemExit(
            "dataframe.py missing import block from helpers — "
            "v0.9.25f-a extract not applied?"
        )

    # tests file exists
    if not Path(TESTS_PATH).exists():
        raise SystemExit(f"{TESTS_PATH} missing — v0.9.25f-a add_tests?")

    print("  state matches v0.9.25f-a uncommitted; helpers.py lacks "
          "MODULE-LEVEL numpy — OK")


# ============ 1. add numpy import to header ============

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
    print("\n=== add `import numpy as np` to helpers header (module-level) ===")
    applied = apply_or_fail(
        HELPERS_PATH, OLD_HEADER, NEW_HEADER,
        label="numpy_import",
    )
    if applied:
        assert_compiles(HELPERS_PATH)
        check_no_invalid_escape(HELPERS_PATH)
    print("  applied" if applied else "  already applied — skip")

    # Belt-and-suspenders: re-check AST module-level after edit
    h_src = Path(HELPERS_PATH).read_text()
    assert "np" in _module_level_imports(h_src), (
        "numpy still not module-level after edit — header anchor missed"
    )


# ============ 2. add regression tests ============

NEW_TESTS = '''

def test_group_boundaries_arrow_smoke():
    """v0.9.25f-a-fix1b (§13.4/257): runtime smoke test of every extracted
    symbol. Prior dogfood in v0.9.25f-a tested _RollState (numpy-free)
    only — missed that _group_boundaries_arrow needs module-level
    `import numpy as np`. This directly exercises the bug: would
    NameError if numpy not in helpers module scope.
    """
    import pyarrow as pa
    from cadw.dataframe_helpers import _group_boundaries_arrow
    b = pa.RecordBatch.from_arrays(
        [pa.array([1, 1, 2, 2, 2, 3])], names=["g"],
    )
    assert _group_boundaries_arrow(b, ["g"]) == [0, 2, 5, 6]


def test_helpers_module_has_required_imports():
    """v0.9.25f-a-fix1b (§13.4/258): AST check — module-level imports
    cover all Name loads inside helper defs (excluding locals +
    builtins).

    Prevents missing module-level imports like the missing numpy that
    v0.9.25f-a shipped. `_group_boundaries_arrow` used `np.flatnonzero`;
    helpers.py had no module-level numpy import -> NameError at runtime.

    Function-local imports (e.g. `import numpy as np` inside
    `_reservoir_positions`) are correctly treated as local — NOT a
    substitute for module-level import in other functions. This test
    walks each function separately and checks `Load` Name against
    that function's locals + module-level imports + builtins.
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
    print("\n=== add regression tests (if not present) ===")
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
    print("\n=== dogfood: all 5 runtime smoke + numpy module-level ===")

    def _verify():
        import ast as _ast
        import inspect as _inspect
        import pyarrow as pa
        from cadw import dataframe_helpers as hm

        # Runtime smoke of 5 (state classes + boundaries):
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

        # **The buggy one**: _group_boundaries_arrow uses np.flatnonzero
        b = pa.RecordBatch.from_arrays(
            [pa.array([1, 1, 2, 2, 2, 3])], names=["g"],
        )
        got = hm._group_boundaries_arrow(b, ["g"])
        assert got == [0, 2, 5, 6], (
            f"_group_boundaries_arrow returned {got}, expected [0,2,5,6]"
        )

        # AST: numpy is module-level
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
            "  10 helpers; _RollState/_ExpandingState/_EwmState/"
            "_ArrowTail/_group_boundaries_arrow smoke-pass; "
            "numpy module-level"
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

    print("\n=== pytest: full targeted (cum + per_group + dataframe + ts) ===")
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
    print("=== v0.9.25f-a-fix1b: numpy header (retry, AST verify) ===")
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
        "cadw/dataframe_helpers.py + fix1b (missing numpy). "
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
        "AST extraction per helper (§13.4/253 per-container heuristic "
        "applied properly: iterate per helper, not per file). "
        "Compile-before-write both files (§13.4/211). "
        "Backward compat: `from cadw.dataframe import _RollState` "
        "still works via `from cadw.dataframe_helpers import (...)` "
        "re-export (6 external imports from tests/ + 6 from scripts/ "
        "preserved). "
        "AST guards intact (§13.4/144 7th occurrence — PROACTIVE): "
        "all 48 scan per-method or whole-file facade invariants; "
        "none reference the 10 extracted helpers by name. "
        "fix1b: my extraction header had `import pyarrow as pa` only; "
        "`_group_boundaries_arrow` uses `np.flatnonzero`. In original "
        "dataframe.py numpy was available from module-level import; "
        "extraction missed it. 37 tests failed (all per-group shift/"
        "diff/rolling/cum* + direct boundary test) with NameError: "
        "name 'np' is not defined at dataframe_helpers.py:321. "
        "Verified 9 other helpers manually — only _group_boundaries_"
        "arrow needs module-level numpy (2 helpers have function-"
        "local numpy from extract; correct). "
        "§13.4/257 (NEW): extracted module needs runtime smoke test of "
        "EVERY extracted symbol, not just import-check + 1 sample. "
        "v0.9.25f-a dogfood tested _RollState (numpy-free) only -> "
        "bug invisible to dogfood, caught by pytest. fix1b adds: "
        "(a) runtime smoke test of _group_boundaries_arrow, (b) AST "
        "import guard test_helpers_module_has_required_imports — "
        "checks every function body's Name loads against module-level "
        "imports + locals + builtins. "
        "§13.4/258 (NEW): fix1's own verify_state used substring "
        "`\"import numpy as np\" in helpers_src` to detect module-level "
        "import — false-positived on function-LOCAL import in "
        "_reservoir_positions. Fix1 aborted at verify_state (before "
        "any mutation) -> state unchanged -> fix1b retries with "
        "AST-based module-level check (iterate tree.body, filter "
        "ast.Import/ImportFrom). Rule: substring only valid when "
        "src has no other occurrences of that text — never true for "
        "arbitrary code. "
        "Fix-cycle count v0.9.25f-a: extract(1) + fix1(verify bug, "
        "no mutation) + fix1b(real fix) = 2 effective cycles, budget "
        "2. Zero bad-commits — fix1's own verify_state was the gate "
        "that caught my verify bug (meta-safety). "
        "New tests/test_v0925f_a.py: 6 total (4 original + 2 fix1b). "
        "Full split roadmap: v0.9.26, single-purpose session — "
        "retarget 48 AST guards via _all_dataframe_sources() helper + "
        "meta-guard test_all_ast_guards_use_rglob, then split. "
        "Next: Phase C (Pandas P0 — 20 features, HIGH user value). "
        "Fast suite: ~2306 → ~2312 (+6 new tests in v0925f_a.py).",
        self_path="scripts/patch_v0925f_a_fix1b_numpy_retry.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.25f-a COMPLETE (extract + fix1b). "
          "Phase B deferred to v0.9.26. Next: Phase C — Pandas P0.")


if __name__ == "__main__":
    main()