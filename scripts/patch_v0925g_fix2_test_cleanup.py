#!/usr/bin/env python3
# v0.9.25g-fix2 — test bug: pandas dtype syntax ≠ CADW vocabulary.
#
# STATE: HEAD = v0.9.25f-a. v0.9.25g uncommitted. Production code
# correct (fix1's type_for_alias fallback handles float32/float64/
# double). test_v0925g.py has 25 tests, 24 pass + 1 fails:
#
#   test_select_dtypes_string_category_vs_explicit:
#     assert df.select_dtypes(include="string[pyarrow]").columns in (
#         ["s"], ["ls", "s"],
#     )
#     → ValueError: no columns matched
#
# ROOT: "string[pyarrow]" is PANDAS dtype syntax (pandas 3 pyarrow
# backend), NOT Arrow alias. pa.type_for_alias("string[pyarrow]")
# raises KeyError → fallback str(t) == "string[pyarrow]" → False for
# both pa.string() and pa.large_string() → no match → ValueError.
#
# This is CORRECT production behavior — CADW select_dtypes accepts
# CADW category names ('string', 'number', 'int', ...) + pa.type_for_
# alias-resolvable names (int64, float32, double, string, ...).
# Pandas-only syntax ('string[pyarrow]', 'Int64', 'category', etc.)
# is not supported and SHOULD raise ValueError. My test wrongly
# expected it to work.
#
# Fix scope: test-only. Replace broken test with two clean tests:
#   (1) test_select_dtypes_string_category_matches_all_strings —
#       'string' category (CADW vocabulary) correctly matches both
#       pa.string() and pa.large_string(). Preserves the passing
#       half of the broken test.
#   (2) test_select_dtypes_rejects_pandas_style_names — documents
#       that pandas-only syntax ('string[pyarrow]', 'Int64') raises
#       ValueError. Verifies the except (ValueError, KeyError,
#       TypeError) clause in fix1's fallback actually catches
#       pa.type_for_alias's KeyError.
#
# Production code change: NONE. Only test_v0925g.py.
#
# §13.4/263 (NEW): user-provided type aliases may come in foreign
# vocabulary (pandas, numpy, SQL). For select_dtypes accept:
#   - CADW category names (own vocabulary)
#   - pa.type_for_alias-resolvable names (Arrow vocabulary)
# Reject silently for anything else → ValueError with dtype list.
# Document this explicitly in test (test_rejects_pandas_style_names)
# so future readers don't expect pandas syntax to work.
#
# Fix-cycle count v0.9.25g: initial(1) → fix1(1) = 2 attempts,
# budget 0-1. Then fix2(1) = 3 total, budget 0-1 (2 over). Still
# 0 bad-commits (fix2 only reverts test overreach, no production
# impact). Phase C's independent-features pattern: 3 features, but
# 2 of my own tests had overreaches (fix1 for real float bug, fix2
# for pandas-vocabulary assumption). Split into C-1 (done, 3 features
# + 2 fixes) and C-2 (next, 3 features) keeps cascade radius small.
#
# Run: uv run python scripts/patch_v0925g_fix2_test_cleanup.py

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
    print("\n=== verify: HEAD=v0.9.25f-a; v0.9.25g+fix1 on-disk; "
          "broken pandas-vocabulary test present ===")
    log = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout
    if "v0.9.25f-a" not in log:
        raise SystemExit(
            f"expected HEAD=v0.9.25f-a; got {log[:200]!r}"
        )
    if "v0.9.25g" in log:
        raise SystemExit("v0.9.25g already committed — fix2 misplaced")

    src = Path(TESTS_PATH).read_text()
    if "test_select_dtypes_string_category_vs_explicit" not in src:
        raise SystemExit(
            "broken test not found — fix2 misplaced?"
        )
    if "string[pyarrow]" not in src:
        raise SystemExit(
            "broken test lacks string[pyarrow] assertion — already "
            "fixed? Inspect file."
        )
    if "test_select_dtypes_rejects_pandas_style_names" in src:
        raise SystemExit(
            "fix2 replacement already present — misplaced"
        )

    # Production fix1 applied
    df = Path("cadw/dataframe.py").read_text()
    if "pa.type_for_alias(s)" not in df:
        raise SystemExit(
            "fix1's type_for_alias fallback not in dataframe.py — "
            "fix1 not applied?"
        )
    print("  state OK: fix1 production fix applied, broken test "
          "present")


# ============ 1. replace broken test ============

OLD_TEST = '''def test_select_dtypes_string_category_vs_explicit():
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
    ) or df.select_dtypes(include="string").columns == ["ls", "s"]'''


NEW_TESTS_BLOCK = '''def test_select_dtypes_string_category_matches_all_strings():
    """'string' category (CADW vocabulary) matches string + large_string
    + binary — broader-than-exact by design (pandas-parity: pandas
    'string' also covers multiple Arrow string backings).
    """
    df = _mk({
        "s": pa.array(["a", "b"], type=pa.string()),
        "ls": pa.array(["a", "b"], type=pa.large_string()),
    })
    assert sorted(df.select_dtypes(include="string").columns) == ["ls", "s"]


def test_select_dtypes_rejects_pandas_style_names():
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


def replace_test():
    print("\n=== replace broken pandas-vocabulary test ===")
    applied = apply_or_fail(
        TESTS_PATH, OLD_TEST, NEW_TESTS_BLOCK,
        label="replace_pandas_vocab_test",
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
    print("=== v0.9.25g-fix2: replace pandas-vocab test ===")
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
        "+ fix1 (type_for_alias) + fix2 (test cleanup). "
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
        "fix1: _matches fallback `str(t) == s` failed on float explicit "
        "names — str(pa.float32()) == 'float', str(pa.float64()) == "
        "'double'. Fix: try pa.type_for_alias(s) first (same resolver "
        "astype uses), fallback str-compare only for exotic. "
        "§13.4/262 (NEW): PyArrow dtype string reprs differ from "
        "user-facing names — for resolving user-provided dtype names "
        "ALWAYS try pa.type_for_alias(s) first. "
        "fix2: my own test contained `string[pyarrow]` (pandas 3 "
        "dtype syntax, NOT Arrow alias). pa.type_for_alias raised "
        "KeyError → fallback str-compare False → no match → "
        "ValueError (correct behavior). Production select_dtypes "
        "was right; test wrongly expected pandas vocabulary to work. "
        "Replaced broken test with 2 clean: "
        "test_select_dtypes_string_category_matches_all_strings "
        "('string' category — CADW vocabulary — matches string + "
        "large_string + binary by design); "
        "test_select_dtypes_rejects_pandas_style_names (documents "
        "pandas-only syntax correctly raises ValueError; also "
        "verifies fix1's except (ValueError, KeyError, TypeError) "
        "catches pa.type_for_alias's KeyError). "
        "§13.4/263 (NEW): user-provided type aliases may use foreign "
        "vocabulary (pandas, numpy, SQL). select_dtypes accepts CADW "
        "category names + pa.type_for_alias-resolvable names only; "
        "reject anything else with clear ValueError. Test documents "
        "this contract explicitly so future readers don't expect "
        "pandas syntax to work. "
        "Fix-cycle count v0.9.25g: initial(1) → fix1(1) → fix2(1) = "
        "3 attempts, budget 0-1 (2 over). Zero bad-commits — fix2 "
        "only reverts test overreach, production select_dtypes "
        "correct since fix1. Churn 100% in test tooling. "
        "New tests (tests/test_v0925g.py): 25 total (24 original "
        "+ fix1's test_select_dtypes_explicit_type_for_alias + "
        "fix2's test_select_dtypes_rejects_pandas_style_names; "
        "test_select_dtypes_string_category_vs_explicit replaced "
        "with test_select_dtypes_string_category_matches_all_strings). "
        "Fast suite: ~2314 → ~2340 (+26). "
        "Next: Phase C batch C-2 (idxmax/idxmin/pct_change/"
        "duplicated).",
        self_path="scripts/patch_v0925g_fix2_test_cleanup.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.25g COMPLETE (3 features + 2 fixes). "
          "Next: C-2 (idxmax/idxmin/pct_change/duplicated).")


if __name__ == "__main__":
    main()