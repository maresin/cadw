#!/usr/bin/env python3
# v0.9.25f-a-fix3 — finish pipeline; fix2's dogfood blocked on
#                     triple-quoted dedent no-op.
#
# STATE: HEAD = v0.9.25f-diagnostic-fix1. v0.9.25f-a uncommitted:
#   - extract applied (helpers.py has 10 helpers + numpy; dataframe.py
#     has re-export block)
#   - tests/test_v0925f_a.py has symtable test replacement (fix2's
#     replace_test succeeded before dogfood abort)
#   - fix2's dogfood aborted: IndentationError in subprocess
#   - pytest after fix2 never ran (SystemExit from dogfood)
#
# ROOT (§13.4/260 NEW): fix2's `_verify` in dogfood() contains
# triple-quoted string literals with 0-indent content (synthetic
# Python source for false-positive tests):
#
#     fake = '''
# import pyarrow as pa
# def with_comprehension(...): ...
# '''
#
# textwrap.dedent() computes common prefix across ALL non-blank lines.
# Inside `_verify` (nested 4-space in dogfood):
#   - line `        fake = '''`      → 8-space indent
#   - line `import pyarrow as pa`    → 0-space indent (inside '''...''')
#   → common prefix = "" (empty) → dedent does NOTHING.
# `def _verify():` stays at 4-space indent → temp file line 1 =
# `    def _verify():` → subprocess `python file.py` fails with
# IndentationError at line 1.
#
# Prior v0.9.25e / fix1b passed because their `_verify` had NO
# triple-quoted strings with 0-indent content (only docstrings with
# matching indentation → dedent worked).
#
# §13.4/260 (NEW): textwrap.dedent fails when function body contains
# triple-quoted strings with unindented content (SQL / JSON / YAML /
# Python code templates). Rule for dogfood `_verify`:
#   (a) indent template content to match (`    import pyarrow...`),
#   (b) move `_verify` to module level (no outer indent → dedent moot),
#   (c) skip subprocess dogfood when pytest provides equivalent coverage.
# Choice (c) here — pytest covers all false-positive classes via
# permanent test_symtable_guard_no_false_positives added below.
#
# Fix3 scope:
#   1. Verify test replacement applied (fix2 mutated before abort).
#   2. Add test_symtable_guard_no_false_positives — permanent coverage
#      of comp/except/annassign false-positives (was only in dogfood).
#   3. Run pytest (authoritative, no subprocess isolation needed).
#   4. Single commit v0.9.25f-a (extract + fix1b + fix2 + fix3).
#
# Fix-cycle count v0.9.25f-a: extract → fix1 (verify bug) → fix1b
# (mutation, pre-flight abort on ast-walk false positives) → fix2
# (test replacement, dogfood abort on dedent) → fix3 (final). 5
# attempts, budget 2 (2.5× over). Zero bad-commits. Churn was 100%
# in test-tooling / dogfood infrastructure, NOT in extract logic
# (dataframe.py + helpers.py correct since extract, numpy since fix1b).
#
# Run: uv run python scripts/patch_v0925f_a_fix3_finish.py

from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import (
    apply_or_fail, assert_compiles, commit_with_guards,
    check_no_invalid_escape,
)


HELPERS_PATH = "cadw/dataframe_helpers.py"
FACADE_PATH = "cadw/dataframe.py"
TESTS_PATH = "tests/test_v0925f_a.py"


# ============ 0. verify state ============

def verify_state():
    print("\n=== verify: HEAD=v0.9.25f-diagnostic-fix1; extract + "
          "symtable replacement applied; pytest not yet run ===")
    log = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout
    if "v0.9.25f-diagnostic-fix1" not in log:
        raise SystemExit(
            f"expected HEAD=v0.9.25f-diagnostic-fix1; got {log[:200]!r}"
        )
    if "v0.9.25f-a" in log:
        raise SystemExit("v0.9.25f-a already committed — fix3 misplaced")

    # helpers.py + numpy
    h_src = Path(HELPERS_PATH).read_text()
    if "import numpy as np" not in h_src:
        raise SystemExit("helpers.py missing numpy — fix1b didn't apply?")
    # Confirm 10 helpers exist
    import ast
    tree = ast.parse(h_src)
    defined = {
        n.name for n in tree.body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    expected = {
        "_ArrowTail", "_RollState", "_ExpandingState", "_EwmState",
        "_group_boundaries_arrow", "_convert_batch", "_concat_batches",
        "_empty_value_counts_table", "_reservoir_positions",
        "_weighted_reservoir_positions",
    }
    missing = expected - defined
    if missing:
        raise SystemExit(f"helpers.py missing {sorted(missing)}")

    # dataframe.py has re-export block
    f_src = Path(FACADE_PATH).read_text()
    if "from cadw.dataframe_helpers import" not in f_src:
        raise SystemExit(
            "dataframe.py missing import block — extract not applied?"
        )

    # Symtable replacement applied (fix2's replace_test succeeded)
    t_src = Path(TESTS_PATH).read_text()
    if "import symtable as _symtable" not in t_src:
        raise SystemExit(
            "test file missing symtable — fix2's replace_test didn't "
            "apply? Re-run patch_v0925f_a_fix2_symtable.py."
        )
    if "test_symtable_guard_no_false_positives" in t_src:
        raise SystemExit(
            "test_symtable_guard_no_false_positives already present — "
            "fix3 add_test misplaced"
        )

    print("  state OK: extract + numpy + symtable replacement, no "
          "pytest run yet, no synthetic test yet")


# ============ 1. add synthetic false-positive pytest test ============

SYNTHETIC_TEST = '''

def test_symtable_guard_no_false_positives():
    """v0.9.25f-a-fix3 (§13.4/259/260): symtable guard produces no
    false positives on comprehension / except-as / AnnAssign — the
    3 classes that broke v0.9.25f-a-fix1b's ast.walk guard.

    v0.9.25f-a-fix1b's `test_helpers_module_has_required_imports` used
    ast.walk + manual local-collection → false-positived on:
      - comprehension targets (`[b.col(i) for i in ...]` → i, b)
      - except-as variables (`except X as e: ... e.foo()` → e)
      - AnnAssign targets (`heap: list = []` → heap)
    v0.9.25f-a-fix2 replaced with symtable — this test proves it
    handles all 3 (plus synthetic missing-import detection).

    Previously covered only by fix2's dogfood (subprocess isolation,
    triple-quoted dedent bug aborted it). Now permanent in pytest —
    no subprocess, no dedent issue.
    """
    import builtins as _builtins
    import symtable as _symtable

    builtin_names = set(dir(_builtins))

    def _collect_missing(src: str) -> list:
        st = _symtable.symtable(src, "<test>", "exec")
        module_names = {sym.get_name() for sym in st.get_symbols()}
        missing: list = []

        def _walk(table, path):
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
                _walk(child, child_path)

        _walk(st, "")
        return missing

    # Case 1: comprehension target
    fake_comp = (
        "import pyarrow as pa\\n"
        "def with_comprehension(x, n):\\n"
        "    return [x * i for i in range(n)]\\n"
    )
    assert _collect_missing(fake_comp) == [], (
        "false positive on comprehension target"
    )

    # Case 2: except-as variable
    fake_except = (
        "def with_except():\\n"
        "    try:\\n"
        "        pass\\n"
        "    except ValueError as e:\\n"
        "        return str(e)\\n"
    )
    assert _collect_missing(fake_except) == [], (
        "false positive on except-as variable"
    )

    # Case 3: AnnAssign target
    fake_ann = (
        "def with_annassign():\\n"
        "    heap: list = []\\n"
        "    return len(heap)\\n"
    )
    assert _collect_missing(fake_ann) == [], (
        "false positive on AnnAssign target"
    )

    # Case 4: real missing import MUST be detected
    fake_missing = (
        "import pyarrow as pa\\n"
        "def uses_numpy(x):\\n"
        "    return np.array([1, 2, 3])\\n"
    )
    issues = _collect_missing(fake_missing)
    assert any(name == "np" for _, name in issues), (
        f"missing numpy not detected: {issues}"
    )
'''


def add_synthetic_test():
    print("\n=== add test_symtable_guard_no_false_positives ===")
    p = Path(TESTS_PATH)
    s = p.read_text()
    if "test_symtable_guard_no_false_positives" in s:
        print("  already present — skip")
        return
    p.write_text(s + SYNTHETIC_TEST)
    assert_compiles(p)
    check_no_invalid_escape(p)
    print("  OK")


# ============ 2. pytest ============

def run_pytest():
    print("\n=== pre-flight: 48 AST guards on dataframe.py + v0925f_a ===")
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

    print("\n=== pytest: full targeted (dataframe + helpers consumers "
          "+ per_group + ts) ===")
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


# ============ 3. main ============

def main():
    print("=== v0.9.25f-a-fix3: finish (skip dogfood, pytest-authoritative) ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"  branch: {br}")

    verify_state()
    add_synthetic_test()
    run_pytest()

    commit_with_guards(
        "v0.9.25f-a: partial extract — 10 pure helpers → "
        "cadw/dataframe_helpers.py + fix1b (missing numpy) + "
        "fix2 (symtable scope check) + fix3 (finish). "
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
        "from scripts/ preserved). AST guards intact. "
        "fix1b: extraction header had `import pyarrow as pa` only; "
        "`_group_boundaries_arrow` uses `np.flatnonzero`. Fixed with "
        "module-level numpy import. §13.4/257 (NEW): extracted module "
        "needs runtime smoke test of EVERY extracted symbol — "
        "added test_group_boundaries_arrow_smoke. "
        "fix2: my own guard test test_helpers_module_has_required_"
        "imports used ast.walk + manual local-collection — false-"
        "positived on 3 classes (comprehension targets, except-as, "
        "AnnAssign). Replaced with symtable.symtable + recursive "
        "get_children + is_global() + is_referenced() — Python's "
        "canonical scope analyzer, handles ALL edge cases. "
        "§13.4/259 (NEW): ast.walk is not a scope analyzer. Use "
        "symtable for 'name X missing module-level import'. "
        "fix3: fix2's dogfood had `_verify` with triple-quoted string "
        "literals (synthetic fake-source templates for false-positive "
        "coverage) whose content was 0-indent → textwrap.dedent "
        "common-prefix calculation saw the 0-indent line → dedent "
        "no-op → `def _verify():` stayed at 4-space indent → subprocess "
        "got `    def _verify():` at line 1 → IndentationError. "
        "Fix3 skips subprocess dogfood entirely (pytest authoritative) "
        "and adds test_symtable_guard_no_false_positives as PERMANENT "
        "pytest test (covers comp/except/annassign false-positives + "
        "synthetic missing-import detection — 4 cases). Prior fix2 "
        "test replacement was correct; fix3 just adds synthetic "
        "coverage and runs pytest. "
        "§13.4/260 (NEW): textwrap.dedent fails when function body "
        "contains triple-quoted strings with unindented content (SQL/"
        "JSON/YAML/Python templates). Rule: (a) indent template "
        "content, (b) module-level `_verify`, or (c) skip subprocess "
        "dogfood when pytest provides equivalent coverage. Choice (c). "
        "Fix-cycle count v0.9.25f-a: extract(1) → fix1(verify bug, "
        "no mutation) → fix1b(mutation, pre-flight abort on ast-walk "
        "false positives) → fix2(test replacement, dogfood abort on "
        "dedent) → fix3(final) = 5 attempts, budget 2 (2.5× over). "
        "Zero bad-commits — each attempt caught at specific gate "
        "(pytest after extract; verify_state after fix1; pre-flight "
        "after fix1b; dogfood after fix2). Churn 100% in test-tooling "
        "and dogfood infrastructure — NOT in extract logic "
        "(dataframe.py + helpers.py correct since extract; numpy since "
        "fix1b). Pattern: v0.9.26 full split should be b1..b7 sequence "
        "(1 concern per patch) to avoid multi-fix cascade. "
        "New tests/test_v0925f_a.py: 7 (4 original + 2 fix1b "
        "smoke/guard + 1 fix3 synthetic false-positive). "
        "Next: Phase C — Pandas P0 (20 features, HIGH user value). "
        "Fast suite: ~2306 → ~2313 (+7).",
        self_path="scripts/patch_v0925f_a_fix3_finish.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.25f-a COMPLETE (extract + fix1b + fix2 + fix3). "
          "Phase B deferred to v0.9.26. Next: Phase C — Pandas P0.")


if __name__ == "__main__":
    main()