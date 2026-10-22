#!/usr/bin/env python3
# v0.9.25f-a-fix2 — replace broken ast.walk scope test with symtable.
#
# STATE: HEAD = v0.9.25f-diagnostic-fix1. v0.9.25f-a uncommitted:
#   - extract applied (10 helpers → cadw/dataframe_helpers.py)
#   - numpy module-level import added (fix1b)
#   - tests/test_v0925f_a.py has 6 tests, 5 passing, 1 failing
#   - failing test = test_helpers_module_has_required_imports (my own,
#     v0.9.25f-a-fix1b) — 3 false positives from ast.walk:
#       _convert_batch: ['e', 'i', 'name']
#       _concat_batches: ['b', 'i']
#       _weighted_reservoir_positions: ['heap']
#
# ROOT: ast.walk + manual local-collection cannot handle:
#   - comprehension targets (i, name, b) — own scope, not local to fn
#   - except-as variables (e) — ExceptHandler binding
#   - AnnAssign targets (heap: list = []) — annotated, not plain Assign
#   - lambda params, walrus (:=) targets
# The existing assert_test_imports_defined helper in patch_utils.py
# also misses ExceptHandler + AnnAssign + comprehension, but it
# ONLY checks Call(Name) targets (bare function calls), not all Name
# Loads. My new test needs all Name Loads (to catch `np.flatnonzero`
# where np is Name inside Attribute.value), so scope analysis must be
# complete — hence symtable, not ast.
#
# §13.4/259 (NEW): ast.walk is not a scope analyzer. For "name X is
# missing module-level import" use symtable.symtable(src, '<name>',
# 'exec').get_children() recursively + sym.is_referenced() +
# sym.is_global(); filter name in module_names or builtins. Handles
# comprehensions, except-as, AnnAssign, lambdas, walrus — all
# correctly, no manual local-collection whack-a-mole.
#
# Fix scope: replace ONLY the buggy test body with symtable version.
# Production numpy import (fix1b) is correct + dogfood-passed;
# runtime smoke test (fix1b) is correct + passes; verify_state was
# correct after fix1b's AST-based module-level check. Only the
# AST-walk guard needs replacement.
#
# Fix-cycle count v0.9.25f-a: extract(1) → fix1(verify bug, no
# mutation) → fix1b(AST test false-positives, mutation applied,
# pre-flight aborted) → fix2(this, replaces broken test). 4 attempts,
# budget 2 = 2× over. Zero bad-commits — each attempt caught at
# specific gate (pytest after extract; verify_state after fix1;
# pre-flight after fix1b). Production changes intact all along;
# churn was 100% in my own test tooling, not in extract logic.
#
# Run: uv run python scripts/patch_v0925f_a_fix2_symtable.py

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

def _module_level_imports(src: str) -> set:
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
          "numpy applied; broken ast-walk test present ===")
    log = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout
    if "v0.9.25f-diagnostic-fix1" not in log:
        raise SystemExit(
            f"expected HEAD=v0.9.25f-diagnostic-fix1; got {log[:200]!r}"
        )
    if "v0.9.25f-a" in log:
        raise SystemExit("v0.9.25f-a already committed — fix2 misplaced")

    # fix1b's numpy import applied
    h_src = Path(HELPERS_PATH).read_text()
    module_imports = _module_level_imports(h_src)
    if "np" not in module_imports:
        raise SystemExit(
            "helpers.py missing MODULE-LEVEL numpy — fix1b didn't apply? "
            "Run patch_v0925f_a_fix1b_numpy_retry.py first."
        )

    # test_v0925f_a.py has BOTH new tests from fix1b, and the guard
    # test is the broken ast-walk version (uses `_ast.walk(fn)` +
    # manual local collection — the false-positive class)
    t_src = Path(TESTS_PATH).read_text()
    if "def test_group_boundaries_arrow_smoke" not in t_src:
        raise SystemExit(
            "missing test_group_boundaries_arrow_smoke — fix1b add_tests "
            "didn't apply?"
        )
    if "def test_helpers_module_has_required_imports" not in t_src:
        raise SystemExit(
            "missing test_helpers_module_has_required_imports — fix1b "
            "add_tests didn't apply?"
        )
    # Broken guard uses ast.walk; symtable version does not.
    if "import symtable as _symtable" in t_src:
        raise SystemExit(
            "test file already has symtable — fix2 might be partially "
            "applied; inspect manually"
        )
    if "missing_by_fn" not in t_src:
        raise SystemExit(
            "broken test doesn't have expected `missing_by_fn` — "
            "inspect manually"
        )

    # Import block in dataframe.py
    f_src = Path(FACADE_PATH).read_text()
    if "from cadw.dataframe_helpers import" not in f_src:
        raise SystemExit(
            "dataframe.py missing import block from helpers"
        )

    print("  state matches v0.9.25f-a-fix1b (numpy applied; broken "
          "ast-walk test present) — OK")


# ============ 1. replace broken test body ============

OLD_TEST = '''def test_helpers_module_has_required_imports():
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
    )'''


NEW_TEST = '''def test_helpers_module_has_required_imports():
    """v0.9.25f-a-fix2 (§13.4/259): module-level imports cover all
    global Name references in helper defs.

    Uses Python's canonical scope analyzer `symtable`, NOT ast.walk.
    ast.walk + manual local-collection misses:
      - comprehension targets (i, name, b in `[b.col(i) for ...]`)
      - except-as variables (e in `except X as e: ... e.foo()`)
      - AnnAssign targets (heap in `heap: list = []`)
      - lambda parameters
      - walrus (:=) targets
    Each is a real local binding, not a module-level import. Prior
    attempt (v0.9.25f-a-fix1b) false-positived on all 4 kinds;
    symtable handles them all correctly.

    Detection: Name with (a) is_referenced(), (b) is_global() = True
    (looked up in module globals / builtins at runtime), (c) NOT
    bound at module level, (d) NOT a builtin -> missing import.

    Function-local imports (`import numpy as np` inside
    `_reservoir_positions`) don't count as module-level: symtable
    reports them as local within that function only; other functions
    referencing np would still be flagged.

    Recursion into nested scopes catches missing imports at any
    depth — symtable.get_children() gives each nested function /
    class / comprehension its own scope.
    """
    import builtins as _builtins
    import inspect as _inspect
    import symtable as _symtable
    from cadw import dataframe_helpers as hm

    src = _inspect.getsource(hm)
    st = _symtable.symtable(src, "<helpers>", "exec")

    module_names = {sym.get_name() for sym in st.get_symbols()}
    builtin_names = set(dir(_builtins))

    missing: list = []

    def _walk_table(table, path):
        for child in table.get_children():
            child_path = (
                f"{path}.{child.get_name()}" if path else child.get_name()
            )
            for sym in child.get_symbols():
                if not sym.is_referenced():
                    continue
                if not sym.is_global():
                    continue
                name = sym.get_name()
                if name in module_names or name in builtin_names:
                    continue
                missing.append((child_path, name))
            _walk_table(child, child_path)

    _walk_table(st, "")

    assert not missing, (
        f"helpers.py references names not bound at module level "
        f"(would NameError at runtime): {sorted(set(missing))}. "
        f"Add missing imports to cadw/dataframe_helpers.py header. "
        f"§13.4/259: symtable.get_children() + is_global() correctly "
        f"handles comprehensions, except-as, AnnAssign, lambdas, "
        f"walrus — all of which ast.walk misses."
    )'''


def replace_test():
    print("\n=== replace broken AST-walk test with symtable version ===")
    applied = apply_or_fail(
        TESTS_PATH, OLD_TEST, NEW_TEST,
        label="test_symtable",
    )
    if applied:
        assert_compiles(TESTS_PATH)
        check_no_invalid_escape(TESTS_PATH)
    print("  applied" if applied else "  already applied — skip")


# ============ 2. dogfood ============

def dogfood():
    print("\n=== dogfood: symtable-based guard detects no false positives "
          "+ catches real missing import ===")

    def _verify():
        import ast as _ast
        import builtins as _builtins
        import inspect as _inspect
        import symtable as _symtable
        import pyarrow as pa
        from cadw import dataframe_helpers as hm

        # 1. Real guard on helpers.py: must be clean now
        src = _inspect.getsource(hm)
        st = _symtable.symtable(src, "<helpers>", "exec")
        module_names = {sym.get_name() for sym in st.get_symbols()}
        builtin_names = set(dir(_builtins))

        def _walk(table, path):
            issues = []
            for child in table.get_children():
                child_path = (
                    f"{path}.{child.get_name()}" if path else child.get_name()
                )
                for sym in child.get_symbols():
                    if not sym.is_referenced():
                        continue
                    if not sym.is_global():
                        continue
                    name = sym.get_name()
                    if name in module_names or name in builtin_names:
                        continue
                    issues.append((child_path, name))
                issues.extend(_walk(child, child_path))
            return issues

        issues = _walk(st, "")
        assert not issues, (
            f"helpers.py still has missing imports: {sorted(set(issues))}"
        )

        # 2. False-positive class: comprehension / except-as / AnnAssign
        fake = '''
import pyarrow as pa

def with_comprehension(x, n):
    return [x * i for i in range(n)]

def with_except():
    try:
        pass
    except ValueError as e:
        return str(e)

def with_annassign():
    heap: list = []
    return len(heap)
'''
        st2 = _symtable.symtable(fake, "<fake>", "exec")
        mod_names2 = {sym.get_name() for sym in st2.get_symbols()}
        issues2 = _walk(st2, "")
        assert not issues2, (
            f"false positives on comp/except/annassign: {issues2}"
        )

        # 3. Real detection: missing numpy
        fake_missing = '''
import pyarrow as pa

def uses_numpy(x):
    return np.array([1, 2, 3])
'''
        st3 = _symtable.symtable(fake_missing, "<fake2>", "exec")
        mod_names3 = {sym.get_name() for sym in st3.get_symbols()}
        issues3 = []

        def _walk2(table, path):
            for child in table.get_children():
                cp = f"{path}.{child.get_name()}" if path else child.get_name()
                for sym in child.get_symbols():
                    if not sym.is_referenced() or not sym.is_global():
                        continue
                    n = sym.get_name()
                    if n in mod_names3 or n in builtin_names:
                        continue
                    issues3.append((cp, n))
                _walk2(child, cp)

        _walk2(st3, "")
        # Should flag uses_numpy → np
        assert any(name == "np" for _, name in issues3), (
            f"failed to detect missing numpy: {issues3}"
        )

        print(
            "  helpers.py: 0 issues; "
            "comp/except/annassign: 0 false positives; "
            "missing numpy: correctly detected"
        )

    run_verify_in_subprocess(_verify)


# ============ 3. pytest ============

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


# ============ 4. main ============

def main():
    print("=== v0.9.25f-a-fix2: symtable for scope analysis ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"  branch: {br}")

    verify_state()
    replace_test()
    dogfood()
    run_pytest()

    commit_with_guards(
        "v0.9.25f-a: partial extract — 10 pure helpers → "
        "cadw/dataframe_helpers.py + fix1b (missing numpy) + "
        "fix2 (symtable scope check). "
        "Phase B diagnostic (v0.9.25f) found 48 AST-guard tests "
        "scanning inspect.getsource(cadw.dataframe) — HARD blocker "
        "for full split (was estimated 4-6, actual 48). Full split "
        "DEFERRED to v0.9.26 with retargeting plan. "
        "Extract: 10 helpers (433 L) with no CadwDataFrame "
        "cross-refs — 4 leaf state classes (_ArrowTail, _RollState, "
        "_ExpandingState, _EwmState) + 6 pure functions "
        "(_group_boundaries_arrow, _convert_batch, _concat_batches, "
        "_empty_value_counts_table, _reservoir_positions, "
        "_weighted_reservoir_positions). dataframe.py 7172 → ~6740 L. "
        "Backward compat via re-export: `from cadw.dataframe import "
        "_RollState` still works (6 external imports from tests/ + 6 "
        "from scripts/ preserved). AST guards intact (all 48 scan "
        "per-method or whole-file facade invariants; none reference "
        "extracted helpers). "
        "fix1b: extraction header had `import pyarrow as pa` only; "
        "`_group_boundaries_arrow` uses `np.flatnonzero`. In original "
        "dataframe.py numpy was available from module-level import; "
        "extraction missed it. 37 tests failed with NameError: name "
        "'np' is not defined. Fixed by adding `import numpy as np` to "
        "helpers header. Verified 9 other helpers manually — only "
        "_group_boundaries_arrow needs module-level numpy (2 helpers "
        "have function-local numpy from extract; correct). "
        "fix2: my own guard test test_helpers_module_has_required_"
        "imports (v0.9.25f-a-fix1b) used ast.walk + manual local-"
        "collection — false-positived on 3 classes: comprehension "
        "targets (_convert_batch: i, name; _concat_batches: b, i), "
        "except-as variables (_convert_batch: e), AnnAssign targets "
        "(_weighted_reservoir_positions: heap). Replaced with "
        "symtable.symtable + recursive get_children + is_global() + "
        "is_referenced() check — Python's own scope analyzer, handles "
        "ALL edge cases correctly (comprehensions, except-as, "
        "AnnAssign, lambdas, walrus). "
        "§13.4/257 (NEW): extracted module needs runtime smoke test of "
        "EVERY extracted symbol (not import-check + 1 sample). "
        "v0.9.25f-a dogfood tested _RollState only; numpy bug invisible "
        "until pytest. fix1b added test_group_boundaries_arrow_smoke. "
        "§13.4/259 (NEW): ast.walk is not a scope analyzer. For "
        "'name X is missing module-level import' use symtable — "
        "ast.walk + manual local-collection misses comprehension "
        "targets, except-as, AnnAssign, lambda params, walrus. "
        "Fix-cycle count v0.9.25f-a: extract(1) + fix1(verify bug, "
        "no mutation) + fix1b(mutation, pre-flight abort) + fix2 "
        "(real fix) = 4 attempts, budget 2 (2× over). Zero bad-"
        "commits — each attempt caught at specific gate (pytest "
        "after extract; verify_state after fix1; pre-flight after "
        "fix1b). Production changes intact all along; churn was 100% "
        "in my own test tooling. "
        "New tests/test_v0925f_a.py: 6 (4 original + 2 fix1b/fix2: "
        "runtime smoke of _group_boundaries_arrow; symtable-based "
        "import guard). "
        "Next: Phase C — Pandas P0 (20 features, HIGH user value). "
        "Fast suite: ~2306 → ~2312 (+6).",
        self_path="scripts/patch_v0925f_a_fix2_symtable.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.25f-a COMPLETE (extract + fix1b + fix2). "
          "Phase B deferred to v0.9.26. Next: Phase C — Pandas P0.")


if __name__ == "__main__":
    main()