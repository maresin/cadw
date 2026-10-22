#!/usr/bin/env python3
# v0.9.25h-fix1 — my own test has stray assert from copy-paste.
#
# STATE: HEAD = v0.9.25g. v0.9.25h uncommitted (production correct;
# 115 passing tests; 1 broken test = my own bug in test file).
#
# BUG: test_pct_change_inf_nan_roundtrip_parquet:
#   assert math.isinf(vals[1])       # 1/0 → inf round-trips
#   assert math.isinf(vals[2])       # 4/1 = 4-1 = 3.0 — wait, 4/1=4, 4-1=3
#                                                 ^^^^^^^^^^^^^^^^^^^^^
# The inline comment literally says "wait, 4/1=4, 4-1=3" — I caught
# myself mid-write that 4/1 is finite (=3.0 after -1), but forgot to
# remove the copy-pasted assert from test_pct_change_div_by_zero_
# pandas_parity (where vals[2] WAS inf). Test asserts isinf(3.0) →
# False → FAIL.
#
# Correct: vals = [None, inf, 3.0]. Row 0 = null (no prev), row 1 =
# 1/0-1 = inf, row 2 = 4/1-1 = 3.0.
#
# Production code intact: idxmax, idxmin, pct_change all correct
# (115 passing tests confirm — pct_change behavior validated by 12
# other tests including div_by_zero and zero_over_zero).
#
# §13.4/265 (NEW): "wait, X" inline comment = signal to STOP and
# delete nearby assert. Comment reasoning ≠ assert verification.
# Copy-paste from sibling test with different data is the common
# case. When you catch yourself in a comment, act on it — don't
# leave half-fix (comment updated, assert stale).
#
# Fix-cycle count v0.9.25h: initial(1) → fix1(1) = 2 attempts,
# 0 production bugs. §13.4/264 (REPL-first) prevented 2 more
# (would have been 4 without it — pc.divide assumption).
#
# Run: uv run python scripts/patch_v0925h_fix1_roundtrip.py

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


TESTS_PATH = "tests/test_v0925h.py"


# ============ 0. verify state ============

def verify_state():
    print("\n=== verify: HEAD=v0.9.25g; v0.9.25h uncommitted; "
          "stray isinf assert present ===")
    log = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout
    if "v0.9.25g" not in log:
        raise SystemExit(
            f"expected HEAD=v0.9.25g; got {log[:200]!r}"
        )
    if "v0.9.25h" in log:
        raise SystemExit("v0.9.25h already committed — fix1 misplaced")

    src = Path(TESTS_PATH).read_text()
    if "def test_pct_change_inf_nan_roundtrip_parquet" not in src:
        raise SystemExit(
            "test_pct_change_inf_nan_roundtrip_parquet not found — "
            "v0.9.25h didn't apply?"
        )
    # Bug present: isinf(vals[2]) + the "wait" comment
    if "wait, 4/1=4, 4-1=3" not in src:
        raise SystemExit(
            "expected 'wait, 4/1=4, 4-1=3' comment (bug marker); "
            "already fixed?"
        )
    if 'assert math.isinf(vals[2])' not in src:
        raise SystemExit(
            "stray isinf(vals[2]) assert not found — already fixed?"
        )

    # Production methods present
    df = Path("cadw/dataframe.py").read_text()
    for marker in ("def pct_change(self",
                   "def _index_at_raw_pos",
                   "def idxmax(self)"):
        if marker not in df:
            raise SystemExit(
                f"production marker {marker!r} missing — v0.9.25h "
                f"production didn't apply?"
            )

    print("  state OK: production intact, stray isinf(vals[2]) "
          "assert present")


# ============ 1. fix broken test ============

OLD_BLOCK = '''def test_pct_change_inf_nan_roundtrip_parquet(tmp_path):
    """inf/nan survive Parquet round-trip (IEEE 754 parquet spec)."""
    import pyarrow.parquet as pq
    df = _mk({"x": [0.0, 1.0, 4.0]})
    result = df.pct_change(periods=1)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    t = pq.read_table(out)
    vals = t.column("x_pct_change").to_pylist()
    assert vals[0] is None
    assert math.isinf(vals[1])       # 1/0 → inf round-trips
    assert math.isinf(vals[2])       # 4/1 = 4-1 = 3.0 — wait, 4/1=4, 4-1=3
    # Actually row 2: current=4, prev=1, 4/1=4, 4-1=3.0
    # Correcting: 4/1=4 → finite, not inf
    # (row 1 is the inf case)'''


NEW_BLOCK = '''def test_pct_change_inf_nan_roundtrip_parquet(tmp_path):
    """inf/nan survive Parquet round-trip (IEEE 754 parquet spec).

    Data: x = [0.0, 1.0, 4.0]. pct_change(1):
      row 0: null (no prev)
      row 1: 1/0 - 1 = inf (div-by-zero)
      row 2: 4/1 - 1 = 3.0 (finite)

    Verifies inf survives pa.array → Parquet → pq.read_table.
    """
    import pyarrow.parquet as pq
    df = _mk({"x": [0.0, 1.0, 4.0]})
    result = df.pct_change(periods=1)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    t = pq.read_table(out)
    vals = t.column("x_pct_change").to_pylist()
    assert vals[0] is None           # no prev for row 0
    assert math.isinf(vals[1])       # 1/0 - 1 = inf round-trips
    assert vals[1] > 0
    assert vals[2] == 3.0            # 4/1 - 1 = 3.0 (finite, not inf)'''


def fix_test():
    print("\n=== fix stray isinf(vals[2]) → vals[2] == 3.0 ===")
    applied = apply_or_fail(
        TESTS_PATH, OLD_BLOCK, NEW_BLOCK,
        label="inf_roundtrip_fix",
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

    print("\n=== pytest: v0925h (should all pass now) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_v0925h.py",
         "tests/test_shift_diff.py",
         "tests/test_shift_on_col.py",
         "tests/test_dataframe.py",
         "tests/test_dataframe_filters.py",
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
    print("=== v0.9.25h-fix1: fix stray isinf in roundtrip test ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"  branch: {br}")

    verify_state()
    fix_test()
    run_pytest()

    commit_with_guards(
        "v0.9.25h: Phase C batch C-2 — idxmax + idxmin + pct_change "
        "+ fix1 (stray isinf assert). "
        "3 of 20 Pandas P0 features (after C-1's isin/between/"
        "select_dtypes at v0.9.25g). "
        "Features: "
        "(1) `_ColumnProxy.idxmax()` — index label of first row with "
        "maximum value. Two-pass: ReduceOp(MinMaxOp, 'max') + stream "
        "for first matching position (pc.equal + pc.indices_nonzero). "
        "Returns index LABEL (RangeIndex → int; MaterializedIndex → "
        "original; MultiIndex → tuple) via `_index_at_raw_pos`. "
        "Nulls skipped; all-null / empty → ValueError. "
        "(2) `_ColumnProxy.idxmin()` — mirror with 'min'. "
        "(3) `CadwDataFrame.pct_change(periods=1, columns=None)` — "
        "append `{col}_pct_change` (float64, nullable). Formula: "
        "(x / shift(x, periods)) - 1. Composition of shift + "
        "pc.cast-to-float64 + pc.divide + pc.subtract. "
        "**Semantics: pandas-parity** — pc.divide uses IEEE 754 (inf "
        "for x/0, nan for 0/0) → matches pandas pct_change exactly. "
        "First `periods` rows → null. Nulls propagate. Empty → "
        "empty schema-correct. Original cols preserved. inf/nan "
        "round-trip through Parquet (verified). "
        "CRITICAL: aborted v0.9.25h attempt's verify_state REPL check "
        "#3 caught WRONG assumption — pc.divide([10,0,1],[2,0,0]) = "
        "[5.0, nan, inf], NOT [5.0, None, None]. §13.4/264 (REPL-"
        "first) WORKED AS DESIGNED: caught BEFORE any edit. HEAD "
        "stayed v0.9.25g. Zero mutation. Corrected: verify_state "
        "asserts [5.0, isnan, isinf]; pct_change docstring reflects "
        "pandas-parity (inf/nan), not divergence. 8 REPL checks in "
        "verify_state (indices_nonzero, int(pa.Scalar), pc.divide "
        "IEEE 754, fill_null-bool, pa.scalar-typed, equal+indices, "
        "int(np.int64), inf/nan round-trip). "
        "fix1: my own test had stray `assert math.isinf(vals[2])` "
        "copy-pasted from sibling test_pct_change_div_by_zero_pandas_"
        "parity (where vals[2] WAS inf). Inline comment 'wait, 4/1=4, "
        "4-1=3' caught my own contradiction but I forgot to remove "
        "the assert. Correct: vals[2] == 3.0 (4/1-1). Test-only fix; "
        "production pct_change intact (validated by 12 sibling "
        "tests). "
        "§13.4/265 (NEW): 'wait, X' inline comment = signal to STOP "
        "and delete nearby assert. Comment reasoning ≠ assert "
        "verification. Copy-paste from sibling test with different "
        "data is common — act on the catch, don't leave half-fix "
        "(comment updated, assert stale). "
        "Fix-cycle count v0.9.25h: initial(1) → fix1(1) = 2 attempts, "
        "0 production bugs. §13.4/264 prevented 2 more (would have "
        "been 4 without). "
        "§13.4/260/261/262 preserved: no subprocess dogfood (dedent), "
        "no symbol moves (whitelist paths), pa.scalar typed scalar "
        "for max_val. "
        "Tests (tests/test_v0925h.py): 26 total — idxmax (8), idxmin "
        "(5), pct_change (13, incl. 2 pandas-parity + 1 parquet-"
        "roundtrip fixed). "
        "Fast suite: ~2343 → ~2369 (+26). "
        "Next: Phase C C-3a (ffill/bfill/reindex).",
        self_path="scripts/patch_v0925h_fix1_roundtrip.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.25h COMPLETE (3 features + fix1). Phase C: 6/20. "
          "Next: C-3a (ffill/bfill/reindex).")


if __name__ == "__main__":
    main()