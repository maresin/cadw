#!/usr/bin/env python3
# v0.9.25g-fix3 — test-only: document pa.type_for_alias behavior.
#
# STATE: HEAD = v0.9.25f-a. v0.9.25g uncommitted. Production
# select_dtypes + fix1 type_for_alias correct. test_v0925g.py has 26
# tests, 25 pass + 1 fails:
#
#   test_select_dtypes_rejects_pandas_style_names:
#     with pytest.raises(ValueError, match="no columns matched"):
#         df_int.select_dtypes(include="Int64")
#     → DID NOT RAISE
#
# ROOT: pa.type_for_alias("Int64") RESOLVES to pa.int64() —
# case-insensitive alias resolution in PyArrow 26. My fix2 test
# wrongly assumed "Int64" (pandas nullable int syntax) would be
# rejected by type_for_alias. It's accepted — production behavior
# is CORRECT; test assumption was wrong.
#
# Empirically:
#   pa.type_for_alias("Int64")           → pa.int64()   (case-insensitive)
#   pa.type_for_alias("int64")           → pa.int64()
#   pa.type_for_alias("string[pyarrow]") → KeyError      (bracket syntax)
#   pa.type_for_alias("Float64")         → pa.float64()
#
# So `Int64` on int64 column correctly MATCHES (feature — pandas
# users pass Int64, CADW handles it). `Float64` on int64 does NOT
# match (correct — different dtype). `string[pyarrow]` does NOT
# resolve (correct — bracket is not an alias).
#
# Fix scope: test-only. Replace broken test with 2 honest:
#   (1) test_select_dtypes_bracket_syntax_rejected — only
#       string[pyarrow] raises (empirically unparseable).
#   (2) test_select_dtypes_alias_case_insensitive — Int64 matches
#       int64 (documents actual PyArrow behavior); Float64 on int64
#       raises (different type); Int32 on int64 raises (different
#       bit-width).
#
# Production code change: NONE. Only test_v0925g.py.
#
# §13.4/264 (NEW): before writing a test for vocabulary / parser
# behavior (aliases, dtype names, magic strings), do a 5-second
# REPL check of actual behavior:
#   python -c "import pyarrow as pa; print(pa.type_for_alias('Int64'))"
# Avoids assumption-based test overreach (this is 2nd in v0.9.25g:
# fix1 real type_for_alias bug, fix2 wrongly assumed rejection,
# fix3 corrects).
#
# Fix-cycle count v0.9.25g: initial(1) → fix1(1) → fix2(1) →
# fix3(1) = 4 attempts, budget 0-1 (3 over). Zero bad-commits —
# production correct since fix1; test overreach 100% of churn
# after fix1. Splitting C-1 from C-2 kept cascade radius small
# (1-branch fixes, not module-wide).
#
# Run: uv run python scripts/patch_v0925g_fix3_alias_behavior.py

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


TESTS_PATH = "tests/test_v0925g.py"


# ============ 0. verify state ============

def verify_state():
    print("\n=== verify: HEAD=v0.9.25f-a; v0.9.25g+fix1+fix2 "
          "on-disk; broken rejects-pandas test present ===")
    log = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout
    if "v0.9.25f-a" not in log:
        raise SystemExit(
            f"expected HEAD=v0.9.25f-a; got {log[:200]!r}"
        )
    if "v0.9.25g" in log:
        raise SystemExit("v0.9.25g already committed — fix3 misplaced")

    src = Path(TESTS_PATH).read_text()
    if "test_select_dtypes_rejects_pandas_style_names" not in src:
        raise SystemExit(
            "fix2's test not found — fix2 didn't apply? "
            "Run patch_v0925g_fix2_test_cleanup.py first."
        )
    if "test_select_dtypes_bracket_syntax_rejected" in src:
        raise SystemExit(
            "fix3 replacement already present — misplaced"
        )

    # Verify PyArrow behavior matches what we assert
    import pyarrow as pa
    try:
        a = pa.type_for_alias("Int64")
        assert a == pa.int64(), (
            f"pa.type_for_alias('Int64') = {a}, expected pa.int64(); "
            f"fix3 test assumptions stale"
        )
    except (KeyError, ValueError) as e:
        raise SystemExit(
            f"pa.type_for_alias('Int64') raised {type(e).__name__}: {e}; "
            f"fix3 assumes it resolves — inspect PyArrow version"
        )
    try:
        pa.type_for_alias("string[pyarrow]")
        raise SystemExit(
            "pa.type_for_alias('string[pyarrow]') unexpectedly resolved; "
            "fix3 assumes KeyError — inspect PyArrow version"
        )
    except (KeyError, ValueError):
        pass  # expected

    print("  state OK: PyArrow behavior verified; broken test present")


# ============ 1. replace broken test ============

OLD_TEST = '''def test_select_dtypes_rejects_pandas_style_names():
    """v0.9.25g-fix2 (§13.4/263): pandas-only dtype syntax raises
    ValueError.

    CADW select_dtypes vocabulary:
      - CADW category names: 'number', 'int', 'float', 'bool',
        'string', 'object', 'datetime', 'category', 'all'.
      - Any alias resolvable by pa.type_for_alias: 'int32',
        'int64', 'float32', 'float64', 'double', 'string', 'bool',
        'date32', 'timestamp[ns]', etc.

    Pandas-only syntax is NOT supported:
      - 'string[pyarrow]' (pandas 3 ArrowStringArray)
      - 'Int64', 'Int32', 'Float64' (pandas nullable int/float)
      - 'category' as pandas CategoricalDtype spec

    These raise ValueError 'no columns matched' — clear signal to
    user to use plain CADW category names or Arrow aliases. This
    test also verifies fix1's `except (ValueError, KeyError,
    TypeError)` correctly catches pa.type_for_alias's KeyError for
    unparseable aliases.
    """
    df = _mk({"s": pa.array(["a", "b"], type=pa.string())})

    # pandas 3 pyarrow-backed syntax
    with pytest.raises(ValueError, match="no columns matched"):
        df.select_dtypes(include="string[pyarrow]")

    # pandas nullable int syntax
    df_int = _mk({"i": pa.array([1, 2], type=pa.int64())})
    with pytest.raises(ValueError, match="no columns matched"):
        df_int.select_dtypes(include="Int64")

    # 'Int32'/'Float64' variants
    with pytest.raises(ValueError, match="no columns matched"):
        df_int.select_dtypes(include="Float64")'''


NEW_TESTS_BLOCK = '''def test_select_dtypes_bracket_syntax_rejected():
    """v0.9.25g-fix3 (§13.4/264): pandas BRACKET syntax
    (`string[pyarrow]`) not resolvable by pa.type_for_alias → CADW
    select_dtypes raises ValueError.

    Only bracket syntax is rejected. Case-insensitive aliases (`Int64`,
    `Float64`) resolve via pa.type_for_alias — see
    test_select_dtypes_alias_case_insensitive.

    Why: `pa.type_for_alias("string[pyarrow]")` raises KeyError
    (bracket syntax isn't in PyArrow's alias table). fix1's
    `except (ValueError, KeyError, TypeError)` catches it, falls
    through to `str(t) == s` → False for all → ValueError from
    select_dtypes "no columns matched".
    """
    df = _mk({"s": pa.array(["a", "b"], type=pa.string())})
    with pytest.raises(ValueError, match="no columns matched"):
        df.select_dtypes(include="string[pyarrow]")


def test_select_dtypes_alias_case_insensitive():
    """v0.9.25g-fix3 (§13.4/264): pa.type_for_alias is case-
    insensitive for most aliases. `Int64` resolves to `pa.int64()`
    — CADW naturally accepts pandas-style capitalized type names
    without any special code.

    Empirically (PyArrow 26):
        pa.type_for_alias("Int64")  → pa.int64()    ← case-insensitive
        pa.type_for_alias("int64")  → pa.int64()
        pa.type_for_alias("Float64") → pa.float64()

    Type-mismatch (different bit-width or int vs float) → no match
    → ValueError.

    (v0.9.25g-fix2 wrongly assumed Int64 would be rejected; this
    test documents actual behavior.)
    """
    df_int64 = _mk({"i": pa.array([1, 2], type=pa.int64())})
    df_int32 = _mk({"i": pa.array([1, 2], type=pa.int32())})
    df_float64 = _mk({"f": pa.array([1.0, 2.0], type=pa.float64())})

    # Case-insensitive: Int64 == int64 (matches)
    assert df_int64.select_dtypes(include="Int64").columns == ["i"]
    assert df_float64.select_dtypes(include="Float64").columns == ["f"]

    # Different bit-width: no match → ValueError
    with pytest.raises(ValueError, match="no columns matched"):
        df_int32.select_dtypes(include="Int64")

    # Float vs int: no match → ValueError
    with pytest.raises(ValueError, match="no columns matched"):
        df_int64.select_dtypes(include="Float64")'''


def replace_test():
    print("\n=== replace broken rejects-pandas test with 2 honest tests ===")
    applied = apply_or_fail(
        TESTS_PATH, OLD_TEST, NEW_TESTS_BLOCK,
        label="alias_behavior_tests",
    )
    if applied:
        assert_compiles(TESTS_PATH)
        check_no_invalid_escape(TESTS_PATH)
    print("  applied" if applied else "  already applied — skip")


# ============ 2. pytest ============

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

    print("\n=== pytest: v0925g + filters + row_filter ===")
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


# ============ 3. main ============

def main():
    print("=== v0.9.25g-fix3: pa.type_for_alias case-insensitivity ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"  branch: {br}")

    verify_state()
    replace_test()
    run_pytest()

    commit_with_guards(
        "v0.9.25g: Phase C batch C-1 — isin + between + select_dtypes "
        "+ fix1 (type_for_alias) + fix2 (test cleanup) + fix3 "
        "(alias behavior). "
        "First 3 of 20 Pandas P0 features. All pure compositions of "
        "existing primitives + surface API. No symbol moves, no "
        "whitelist changes, no new memory models — avoiding Phase B's "
        "cascade class. "
        "Features: "
        "(1) `_ColumnProxy.isin(values)` — PredicateFilter 'isin' → "
        "pc.is_in. Null matches only if None ∈ values. Empty → all "
        "False. Non-coercible → TypeError. "
        "(2) `_ColumnProxy.between(left, right, inclusive='both')` — "
        "PredicateFilter 'between' → pc.and_(ge/gt, le/lt). inclusive "
        "∈ {'both','neither','left','right'}. Kleene nulls dropped. "
        "lo > hi → all False. "
        "(3) `CadwDataFrame.select_dtypes(include=, exclude=)` — "
        "dtype-based selection. Category names: 'number' / 'int' / "
        "'integer' / 'float' / 'floating' / 'bool' / 'boolean' / "
        "'string' / 'object' / 'datetime' / 'category' / 'all'. "
        "Explicit pyarrow type: any alias resolvable by "
        "pa.type_for_alias (case-insensitive: int32/Int32 both work; "
        "float32/float64/double/string/etc.). Bracket syntax "
        "('string[pyarrow]') NOT resolvable → ValueError. "
        "include ∩ ¬exclude. No-match → ValueError. Filters preserved. "
        "fix1: _matches fallback `str(t) == s` failed on float — "
        "str(pa.float32()) == 'float', str(pa.float64()) == 'double'. "
        "Fix: try pa.type_for_alias(s) first (same resolver astype "
        "uses), fallback str-compare only for exotic. "
        "§13.4/262 (NEW): PyArrow dtype string reprs differ from "
        "user-facing names — use pa.type_for_alias(s) first. "
        "fix2: my own test used `string[pyarrow]` (pandas 3 dtype "
        "syntax, not Arrow alias) → TypeError expected but pa.type_"
        "for_alias raised KeyError → ValueError (correct). Replaced "
        "with 2 tests: bracket-syntax-rejected + alias-case-"
        "insensitive. §13.4/263: user-provided type aliases may use "
        "foreign vocabulary; select_dtypes accepts CADW category "
        "names + pa.type_for_alias-resolvable names only. "
        "fix3: fix2's test asserted `Int64` (uppercase) rejected — "
        "wrong; pa.type_for_alias('Int64') resolves to pa.int64() "
        "(case-insensitive). This is a FEATURE (pandas users pass "
        "Int64/Float64, CADW handles naturally). Replaced broken "
        "test with 2 honest: bracket-syntax-rejected (only "
        "string[pyarrow] unparseable); alias-case-insensitive "
        "(Int64/INT64 work via type_for_alias; Float64 on int64 "
        "raises — different type). "
        "§13.4/264 (NEW): verify vocabulary/parser behavior via "
        "5-second REPL check BEFORE writing test assumptions. "
        "python -c \"import pyarrow as pa; print(pa.type_for_alias("
        "'Int64'))\" would have caught this. This is 2nd test-side "
        "overreach in v0.9.25g (fix2 wrongly assumed rejection, "
        "fix3 corrects). "
        "Fix-cycle count v0.9.25g: initial(1) → fix1(1) → fix2(1) "
        "→ fix3(1) = 4 attempts, budget 0-1 (3 over). Zero bad-"
        "commits — production correct since fix1; test overreach "
        "100% of churn after fix1. Splitting C-1 from C-2 kept "
        "cascade radius small (1-branch fixes, not module-wide). "
        "New tests (tests/test_v0925g.py): 26 total (fix3 replaces "
        "test_select_dtypes_rejects_pandas_style_names with "
        "test_select_dtypes_bracket_syntax_rejected + "
        "test_select_dtypes_alias_case_insensitive — net +1). "
        "Fast suite: ~2314 → ~2340 (+26). "
        "Next: Phase C batch C-2 (idxmax/idxmin/pct_change/"
        "duplicated).",
        self_path="scripts/patch_v0925g_fix3_alias_behavior.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.25g COMPLETE (3 features + 3 fixes). "
          "Next: C-2 (idxmax/idxmin/pct_change/duplicated).")


if __name__ == "__main__":
    main()