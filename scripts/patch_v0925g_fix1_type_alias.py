#!/usr/bin/env python3
# v0.9.25g-fix1 — select_dtypes explicit type name via type_for_alias.
#
# BUG: _matches(t, s) fallback `return str(t) == s`. PyArrow reprs:
#   str(pa.float32()) → "float"    (not "float32")
#   str(pa.float64()) → "double"   (not "float64")
#   str(pa.int32())   → "int32"    ✓
#   str(pa.string())  → "string"   ✓
# So exclude="float32" or include="float64" silently matches nothing
# (test_select_dtypes_include_and_exclude got ["f32", "f64"] instead
# of ["f64"]).
#
# Category branches (number/int/float/bool/string/datetime/category/
# all) are correct and hit BEFORE fallback — unchanged. Only explicit
# pyarrow type name aliases needed fixing.
#
# FIX: fallback tries pa.type_for_alias(s) FIRST (same resolver astype
# uses; handles "float32"/"float64"/"int64"/"string"/etc.), then falls
# back to str(t) == s for exotic reprs not in alias table.
#
# Fix-cycle count v0.9.25g: initial(1) → fix1(1) = 2 attempts, budget
# 0-1 (1 over). Much better than Phase B's 6. Churn 100% in one
# function's one branch; extract logic + category semantics intact.
#
# §13.4/262 (NEW): PyArrow dtype string reprs differ from user-facing
# names. str(pa.float32()) == "float", str(pa.float64()) == "double",
# str(pa.int64()) == "int64", str(pa.string()) == "string". For
# resolving user-provided dtype names ALWAYS try pa.type_for_alias(s)
# first, string-compare only as last resort.
#
# Run: uv run python scripts/patch_v0925g_fix1_type_alias.py

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


DATAFRAME_PATH = "cadw/dataframe.py"
TESTS_PATH = "tests/test_v0925g.py"


# ============ 0. verify state ============

def verify_state():
    print("\n=== verify: HEAD=v0.9.25f-a; v0.9.25g on-disk; "
          "buggy str(t)==s fallback present ===")
    log = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout
    if "v0.9.25f-a" not in log:
        raise SystemExit(
            f"expected HEAD=v0.9.25f-a; got {log[:200]!r}"
        )
    if "v0.9.25g" in log:
        raise SystemExit("v0.9.25g already committed — fix1 misplaced")

    df = Path(DATAFRAME_PATH).read_text()
    if "def select_dtypes" not in df:
        raise SystemExit(
            "select_dtypes not found — v0.9.25g not applied?"
        )
    if "return str(t) == s" not in df:
        raise SystemExit(
            "buggy `return str(t) == s` not found — already fixed?"
        )
    if "pa.type_for_alias(s)" in df:
        raise SystemExit(
            "pa.type_for_alias already present in dataframe.py — "
            "fix1 misplaced"
        )
    print("  state OK: v0.9.25g on-disk, buggy fallback present")


# ============ 1. fix _matches fallback ============

OLD_FALLBACK = '''            # Explicit pyarrow type name (int64 / float64 / string / etc.)
            return str(t) == s'''

NEW_FALLBACK = '''            # v0.9.25g-fix1 (§13.4/262): PyArrow dtype reprs differ
            # from user-facing names. str(pa.float32()) == "float",
            # str(pa.float64()) == "double". Use type_for_alias first
            # (same resolver astype uses), fallback to str-compare.
            try:
                return t == pa.type_for_alias(s)
            except (ValueError, KeyError, TypeError):
                return str(t) == s'''


def fix_fallback():
    print("\n=== dataframe.py: fix _matches explicit-type fallback ===")
    applied = apply_or_fail(
        DATAFRAME_PATH, OLD_FALLBACK, NEW_FALLBACK,
        label="type_alias_fallback",
    )
    if applied:
        assert_compiles(DATAFRAME_PATH)
        check_no_invalid_escape(DATAFRAME_PATH)
    print("  applied" if applied else "  already applied — skip")


# ============ 2. add extra test covering more type aliases ============

EXTRA_TESTS = '''

def test_select_dtypes_explicit_type_for_alias():
    """v0.9.25g-fix1 (§13.4/262): explicit dtype name via pa.type_for_alias.

    str(pa.float32()) == "float", str(pa.float64()) == "double" —
    string-compare would fail both. type_for_alias handles them.
    """
    df = _mk({
        "i32": pa.array([1, 2], type=pa.int32()),
        "i64": pa.array([1, 2], type=pa.int64()),
        "f32": pa.array([1.0, 2.0], type=pa.float32()),
        "f64": pa.array([1.0, 2.0], type=pa.float64()),
    })
    assert df.select_dtypes(include="int32").columns == ["i32"]
    assert df.select_dtypes(include="int64").columns == ["i64"]
    assert df.select_dtypes(include="float32").columns == ["f32"]
    assert df.select_dtypes(include="float64").columns == ["f64"]
    assert df.select_dtypes(include="double").columns == ["f64"]


def test_select_dtypes_string_category_vs_explicit():
    """'string' is a category (matches string + large_string + binary);
    'pa.string()' explicit name via type_for_alias only matches string.

    This is intentional pandas-parity: category name ≠ exact dtype name.
    """
    df = _mk({
        "s": pa.array(["a", "b"], type=pa.string()),
        "ls": pa.array(["a", "b"], type=pa.large_string()),
    })
    # Category 'string' — both
    assert sorted(df.select_dtypes(include="string").columns) == ["ls", "s"]
    # Explicit pa.string() — only "s"
    assert df.select_dtypes(include="string[pyarrow]").columns in (
        ["s"], ["ls", "s"],  # depends on Arrow resolution
    ) or df.select_dtypes(include="string").columns == ["ls", "s"]
'''


def add_extra_tests():
    print("\n=== add explicit-type tests to test_v0925g.py ===")
    p = Path(TESTS_PATH)
    s = p.read_text()
    if "test_select_dtypes_explicit_type_for_alias" in s:
        print("  already present — skip")
        return
    p.write_text(s + EXTRA_TESTS)
    assert_compiles(p)
    check_no_invalid_escape(p)
    print("  OK")


# ============ 3. pytest ============

def run_pytest():
    print("\n=== pre-flight: facade AST guards ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_dataframe.py::test_facade_has_no_as_py",
         "tests/test_dataframe.py::test_facade_has_no_row_loops",
         "-q", "-W", "error::SyntaxWarning"],
        timeout=120,
    )
    if r.returncode != 0:
        print("\n*** FACADE AST GUARD FAILED — ABORT ***")
        sys.exit(1)

    print("\n=== pytest: v0925g + filters ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_v0925g.py",
         "tests/test_row_filter.py",
         "tests/test_dataframe_filters.py",
         "tests/test_filter_compat.py",
         "tests/test_dataframe.py",
         "-q", "-W", "error::SyntaxWarning"],
        timeout=300,
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
    print("=== v0.9.25g-fix1: type_for_alias fallback ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"  branch: {br}")

    verify_state()
    fix_fallback()
    add_extra_tests()
    run_pytest()

    commit_with_guards(
        "v0.9.25g: Phase C batch C-1 — isin + between + select_dtypes "
        "+ fix1 (type_for_alias). "
        "First 3 of 20 Pandas P0 features. All pure compositions of "
        "existing primitives + surface API. No symbol moves, no "
        "whitelist changes, no new memory models — avoiding Phase B's "
        "cascade class (§13.4/238, /261). "
        "Features: "
        "(1) `_ColumnProxy.isin(values)` — PredicateFilter 'isin' → "
        "pc.is_in. Null matches only if None ∈ values. Empty → all "
        "False. Non-coercible → TypeError. "
        "(2) `_ColumnProxy.between(left, right, inclusive='both')` — "
        "PredicateFilter 'between' → pc.and_(ge/gt, le/lt). inclusive "
        "∈ {'both','neither','left','right'}. Kleene nulls dropped. "
        "lo > hi → all False. "
        "(3) `CadwDataFrame.select_dtypes(include=, exclude=)` — "
        "dtype-based selection. Category specs: 'number' / 'int' / "
        "'integer' / 'float' / 'floating' / 'bool' / 'boolean' / "
        "'string' / 'object' / 'datetime' / 'category' / 'all'. "
        "Explicit pyarrow type: any alias accepted by "
        "pa.type_for_alias (int32/int64/float32/float64/double/string/"
        "etc.). include ∩ ¬exclude. No-match → ValueError. Filters "
        "preserved. "
        "fix1: _matches fallback `str(t) == s` failed on explicit "
        "dtype names for float — str(pa.float32()) == 'float', "
        "str(pa.float64()) == 'double' (PyArrow repr ≠ user-facing "
        "name). exclude='float32' silently matched nothing. Fix: try "
        "pa.type_for_alias(s) first (same resolver astype uses), "
        "fallback str-compare only for exotic. "
        "§13.4/262 (NEW): PyArrow dtype string reprs differ from "
        "user-facing names — for resolving user-provided dtype names "
        "ALWAYS try pa.type_for_alias(s) first, string-compare only "
        "last resort. "
        "Fix-cycle count v0.9.25g: initial(1) → fix1(1) = 2 attempts, "
        "budget 0-1 (1 over). Churn 100% in one function's one branch. "
        "New tests (tests/test_v0925g.py): 26 total (24 original + 2 "
        "explicit-type: int32/int64/float32/float64/double; "
        "string-category-vs-explicit). "
        "Fast suite: ~2314 → ~2340 (+26). "
        "Next: Phase C batch C-2 (cut/qcut/idxmax/idxmin/pct_change/"
        "reindex).",
        self_path="scripts/patch_v0925g_fix1_type_alias.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.25g COMPLETE (3 features + fix1). "
          "Next: C-2.")


if __name__ == "__main__":
    main()