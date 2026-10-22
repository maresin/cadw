#!/usr/bin/env python3
# v0.9.25g — Phase C batch C-1: 3 simple surface API features.
#
# Phase C = Pandas P0 (20 features per roadmap). This batch: 3 simplest.
# All are pure compositions of existing primitives + surface API.
# NO symbol moves, NO whitelist changes, NO new memory models.
#
# FEATURES:
#   1. `df['col'].isin([1, 2, 3])` → PredicateFilter ("isin" → pc.is_in)
#   2. `df['col'].between(lo, hi, inclusive='both')` →
#      PredicateFilter ("between" → pc.and_(ge/gt, le/lt))
#   3. `df.select_dtypes(include=..., exclude=...)` → dtype-based
#      column selection (composition over dtypes())
#
# WHY SAFE (vs Phase B cascade):
#   - No symbols moved between files → no whitelist path updates
#     (§13.4/261).
#   - No new tests guarding memory bounds → no AST guards retargeted.
#   - 3 independent apply_or_fail edits (row_filter.py, dataframe.py × 2)
#     + tests/test_v0925g.py. Each anchor unique per function.
#   - No subprocess dogfood (`_verify` with triple-quoted templates
#     triggered §13.4/260 in Phase B). Verification = pytest (fresh
#     subprocess via uv run → no stale module cache, §13.4/178).
#
# Semantics (pandas-compatible):
#   isin(values): null row not matched unless None in values. Empty
#     values → all False. Non-coercible value type → TypeError.
#   between(lo, hi, inclusive): inclusive ∈ {'both' (default),
#     'neither', 'left', 'right'}. Null row → not matched (Kleene
#     drop). lo > hi → all False (no reorder).
#   select_dtypes(include=, exclude=): dtype string predicates:
#     'number' | 'int'/'integer' | 'float'/'floating' | 'bool' |
#     'string'/'object' | 'datetime' | 'category' | 'all' |
#     explicit pyarrow type name ('int64', 'float64', 'string').
#     Include then exclude: keep ∈ include ∩ ¬exclude.
#     At least one of include/exclude required. No-match → ValueError.
#
# Run: uv run python scripts/patch_v0925g_isin_between_selectdtypes.py

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


ROW_FILTER_PATH = "cadw/dataset/row_filter.py"
DATAFRAME_PATH = "cadw/dataframe.py"
TESTS_PATH = "tests/test_v0925g.py"


# ============ 0. verify state ============

def verify_state():
    print("\n=== verify: HEAD=v0.9.25f-a; no new features yet ===")
    log = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout
    if "v0.9.25f-a" not in log:
        raise SystemExit(
            f"expected HEAD=v0.9.25f-a; got {log[:200]!r}"
        )
    rf = Path(ROW_FILTER_PATH).read_text()
    if '"isin"' in rf:
        raise SystemExit("row_filter.py already has isin — v0.9.25g applied?")
    if '"between"' in rf:
        raise SystemExit("row_filter.py already has between")
    df = Path(DATAFRAME_PATH).read_text()
    if "def select_dtypes" in df:
        raise SystemExit("dataframe.py already has select_dtypes")
    if "def isin(self, values)" in df:
        raise SystemExit("_ColumnProxy.isin already present")
    if "def between(self, left, right" in df:
        raise SystemExit("_ColumnProxy.between already present")
    print("  state OK: no v0.9.25g features yet")


# ============ 1. row_filter.py: extend _OPS_NO_SCALAR + branches ============

OLD_OPS_SET = '''    _OPS_NO_SCALAR = frozenset({"is_null", "not_null", "contains", "starts_with", "ends_with"})'''

NEW_OPS_SET = '''    _OPS_NO_SCALAR = frozenset({
        "is_null", "not_null", "contains", "starts_with", "ends_with",
        # v0.9.25g (Phase C C-1): set-based / range-based predicates.
        "isin", "between",
    })'''


OLD_MASK_TAIL = '''        if self.op == "ends_with":
            m = pc.ends_with(arr, self.value)
            return pc.fill_null(m, pa.scalar(False))
        scalar = pa.scalar(self.value, type=arr.type)
        return self._OPS[self.op](arr, scalar)'''


NEW_MASK_TAIL = '''        if self.op == "ends_with":
            m = pc.ends_with(arr, self.value)
            return pc.fill_null(m, pa.scalar(False))
        # v0.9.25g: isin — set membership, null-safe (null matches only
        # if None ∈ values). Empty list → all False.
        if self.op == "isin":
            values = list(self.value)
            if not values:
                return pa.array([False] * len(arr), type=pa.bool_())
            try:
                value_set = pa.array(values, type=arr.type)
            except (pa.ArrowInvalid, pa.ArrowTypeError) as e:
                raise TypeError(
                    f"isin: values {values[:5]!r} not coercible to "
                    f"column type {arr.type}: {e}"
                ) from e
            return pc.is_in(arr, value_set=value_set)
        # v0.9.25g: between — inclusive='both'|'neither'|'left'|'right',
        # default 'both'. Kleene nulls dropped.
        if self.op == "between":
            lo, hi, inclusive = self.value
            if inclusive not in ("both", "neither", "left", "right"):
                raise ValueError(
                    f"between: inclusive={inclusive!r} not in "
                    f"'both'|'neither'|'left'|'right'"
                )
            lo_s = pa.scalar(lo, type=arr.type)
            hi_s = pa.scalar(hi, type=arr.type)
            lo_op = pc.greater_equal if inclusive in ("both", "left") \\
                else pc.greater
            hi_op = pc.less_equal if inclusive in ("both", "right") \\
                else pc.less
            return pc.and_(lo_op(arr, lo_s), hi_op(arr, hi_s))
        scalar = pa.scalar(self.value, type=arr.type)
        return self._OPS[self.op](arr, scalar)'''


def extend_row_filter():
    print("\n=== row_filter.py: add isin + between predicates ===")
    for old, new, label in [
        (OLD_OPS_SET, NEW_OPS_SET, "ops_set"),
        (OLD_MASK_TAIL, NEW_MASK_TAIL, "mask_tail"),
    ]:
        applied = apply_or_fail(ROW_FILTER_PATH, old, new, label=label)
        if applied:
            assert_compiles(ROW_FILTER_PATH)
            check_no_invalid_escape(ROW_FILTER_PATH)
        print(f"  {label}: {'applied' if applied else 'skip'}")


# ============ 2. dataframe.py: _ColumnProxy.isin + between ============

OLD_PROXY_TAIL = '''    def notnull(self) -> PredicateFilter:
        """Not-null mask. Use in `df[df['a'].notnull()]`."""
        return PredicateFilter(self._column, "not_null", None)'''


NEW_PROXY_TAIL = '''    def notnull(self) -> PredicateFilter:
        """Not-null mask. Use in `df[df['a'].notnull()]`."""
        return PredicateFilter(self._column, "not_null", None)

    def isin(self, values) -> PredicateFilter:
        """Set membership (v0.9.25g). `df[df['a'].isin([1, 2, 3])]`.

        - values: iterable of scalars; coerced to column type per batch.
        - Empty iterable → all False.
        - Null row matches only if None is in values (pandas semantics).
        - Non-coercible type → TypeError at first batch.
        """
        return PredicateFilter(self._column, "isin", values)

    def between(self, left, right, *, inclusive: str = "both"):
        """Range predicate (v0.9.25g). `df[df['a'].between(0, 10)]`.

        - inclusive ∈ {'both' (default), 'neither', 'left', 'right'}.
        - Null row not matched (Kleene drop).
        - left > right → all False (no argument reorder).
        """
        return PredicateFilter(
            self._column, "between", (left, right, inclusive),
        )'''


def extend_column_proxy():
    print("\n=== dataframe.py: _ColumnProxy.isin + between ===")
    applied = apply_or_fail(
        DATAFRAME_PATH, OLD_PROXY_TAIL, NEW_PROXY_TAIL,
        label="proxy_methods",
    )
    if applied:
        assert_compiles(DATAFRAME_PATH)
        check_no_invalid_escape(DATAFRAME_PATH)
    print("  applied" if applied else "  already applied — skip")


# ============ 3. dataframe.py: select_dtypes ============

OLD_COUNT_ANCHOR = '''    def __len__(self) -> int:
        """Row count (v0.9.6b2) — same as count().

        Enables `len(df)` in per-group apply patterns
        (`lambda g: len(g)`). O(1) unfiltered, O(N) filtered
        (same cost as count()).
        """
        return self.count()'''


NEW_COUNT_ANCHOR = OLD_COUNT_ANCHOR + '''

    def select_dtypes(
        self,
        include=None,
        exclude=None,
    ) -> "CadwDataFrame":
        """Select columns by dtype predicate (v0.9.25g).

        - include / exclude: dtype spec (str or iterable of str).
          At least one required.
        - Keep column if (matches any include OR include is None/'all')
          AND (matches no exclude).
        - Dtype specs: 'number', 'int'/'integer', 'float'/'floating',
          'bool'/'boolean', 'string'/'object', 'datetime',
          'category', 'all', or explicit pyarrow type name
          (e.g. 'int64', 'float64', 'string').
        - Returns new CadwDataFrame with selected columns;
          filters preserved.
        - No-match → ValueError.

        Example:
            df.select_dtypes(include='number')
            df.select_dtypes(exclude=['string', 'bool'])
            df.select_dtypes(include='number', exclude='float32')
        """
        if include is None and exclude is None:
            raise ValueError(
                "select_dtypes: at least one of include/exclude required"
            )

        def _normalize(spec):
            if spec is None:
                return None
            if isinstance(spec, str):
                return [spec]
            return list(spec)

        def _matches(t: pa.DataType, s: str) -> bool:
            if s == "all":
                return True
            if s == "number":
                return pa.types.is_integer(t) or pa.types.is_floating(t)
            if s in ("int", "integer"):
                return pa.types.is_integer(t)
            if s in ("float", "floating"):
                return pa.types.is_floating(t)
            if s in ("bool", "boolean"):
                return pa.types.is_boolean(t)
            if s in ("string", "object"):
                return (
                    pa.types.is_string(t) or pa.types.is_large_string(t)
                    or pa.types.is_binary(t) or pa.types.is_large_binary(t)
                )
            if s in ("datetime", "datetime64"):
                return pa.types.is_timestamp(t)
            if s == "category":
                return pa.types.is_dictionary(t)
            # Explicit pyarrow type name (int64 / float64 / string / etc.)
            return str(t) == s

        inc_list = _normalize(include)
        exc_list = _normalize(exclude)

        keep = []
        for c in self._columns:
            t = self._dataset.schema.field(c).type
            if inc_list is not None and "all" not in inc_list:
                if not any(_matches(t, s) for s in inc_list):
                    continue
            if exc_list is not None:
                if any(_matches(t, s) for s in exc_list):
                    continue
            keep.append(c)

        if not keep:
            raise ValueError(
                f"select_dtypes: no columns matched "
                f"include={include!r} exclude={exclude!r}; "
                f"available dtypes={self.dtypes()}"
            )
        return self[keep]'''


def add_select_dtypes():
    print("\n=== dataframe.py: select_dtypes method ===")
    applied = apply_or_fail(
        DATAFRAME_PATH, OLD_COUNT_ANCHOR, NEW_COUNT_ANCHOR,
        label="select_dtypes",
    )
    if applied:
        assert_compiles(DATAFRAME_PATH)
        check_no_invalid_escape(DATAFRAME_PATH)
    print("  applied" if applied else "  already applied — skip")


# ============ 4. tests ============

NEW_TESTS = '''"""v0.9.25g: Phase C batch C-1 — isin, between, select_dtypes."""
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard


def _mk(data, mem_mb=20):
    return CadwDataFrame.from_table(
        pa.table(data),
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
    )


# ============ isin ============


def test_isin_basic():
    df = _mk({"x": [1, 2, 3, 4], "v": [10, 20, 30, 40]})
    sub = df[df["x"].isin([2, 3])]
    assert sub.count() == 2
    out = sub.head(10)
    assert sorted(out.column("v").to_pylist()) == [20, 30]


def test_isin_empty_list():
    df = _mk({"x": [1, 2, 3]})
    sub = df[df["x"].isin([])]
    assert sub.count() == 0


def test_isin_strings():
    df = _mk({
        "s": ["a", "b", "c", "a"], "v": [1, 2, 3, 4],
    })
    sub = df[df["s"].isin(["a", "c"])]
    assert sub.count() == 3


def test_isin_null_row_not_matched():
    df = _mk({
        "x": pa.array([1, None, 3], type=pa.int64()),
        "v": [10, 20, 30],
    })
    sub = df[df["x"].isin([1, 3])]
    assert sub.count() == 2


def test_isin_null_in_values_matches_null_row():
    df = _mk({
        "x": pa.array([1, None, 3], type=pa.int64()),
        "v": [10, 20, 30],
    })
    sub = df[df["x"].isin([1, None])]
    assert sub.count() == 2


def test_isin_bad_type_raises():
    df = _mk({"x": [1, 2, 3]})
    # Wrong value type: int column, string values → TypeError at mask
    with pytest.raises(TypeError, match="isin"):
        df[df["x"].isin(["a", "b"])].count()


# ============ between ============


def test_between_basic_both():
    df = _mk({"x": [1, 5, 10, 15, 20], "v": [1, 2, 3, 4, 5]})
    sub = df[df["x"].between(5, 15)]
    assert sub.count() == 3  # 5, 10, 15 (both inclusive)


def test_between_neither():
    df = _mk({"x": [1, 5, 10, 15, 20]})
    sub = df[df["x"].between(5, 15, inclusive="neither")]
    assert sub.count() == 1  # 10 only


def test_between_left():
    df = _mk({"x": [1, 5, 10, 15, 20]})
    sub = df[df["x"].between(5, 15, inclusive="left")]
    assert sub.count() == 2  # 5, 10


def test_between_right():
    df = _mk({"x": [1, 5, 10, 15, 20]})
    sub = df[df["x"].between(5, 15, inclusive="right")]
    assert sub.count() == 2  # 10, 15


def test_between_null_row_not_matched():
    df = _mk({
        "x": pa.array([1, None, 10], type=pa.int64()),
    })
    sub = df[df["x"].between(0, 20)]
    assert sub.count() == 2  # null drops


def test_between_lo_gt_hi_empty():
    df = _mk({"x": [1, 5, 10]})
    sub = df[df["x"].between(10, 5)]
    assert sub.count() == 0  # no reorder


def test_between_invalid_inclusive():
    df = _mk({"x": [1, 2, 3]})
    with pytest.raises(ValueError, match="between: inclusive"):
        df[df["x"].between(1, 2, inclusive="bogus")].count()


def test_between_float():
    df = _mk({"x": [1.5, 2.5, 3.5, 4.5]})
    sub = df[df["x"].between(2.0, 4.0)]
    assert sub.count() == 2  # 2.5, 3.5


# ============ select_dtypes ============


def test_select_dtypes_include_number():
    df = _mk({"i": [1, 2], "f": [1.0, 2.0], "s": ["a", "b"]})
    sub = df.select_dtypes(include="number")
    assert sorted(sub.columns) == ["f", "i"]


def test_select_dtypes_exclude_string():
    df = _mk({"i": [1, 2], "f": [1.0, 2.0], "s": ["a", "b"]})
    sub = df.select_dtypes(exclude="string")
    assert sorted(sub.columns) == ["f", "i"]


def test_select_dtypes_int_only():
    df = _mk({"i": [1, 2], "f": [1.0, 2.0]})
    sub = df.select_dtypes(include="int")
    assert sub.columns == ["i"]


def test_select_dtypes_float_only():
    df = _mk({"i": [1, 2], "f": [1.0, 2.0]})
    sub = df.select_dtypes(include="float")
    assert sub.columns == ["f"]


def test_select_dtypes_include_list():
    df = _mk({"i": [1, 2], "f": [1.0, 2.0], "s": ["a", "b"]})
    sub = df.select_dtypes(include=["int64", "string"])
    assert sorted(sub.columns) == ["i", "s"]


def test_select_dtypes_include_and_exclude():
    df = _mk({
        "i": [1, 2], "f32": pa.array([1.0, 2.0], type=pa.float32()),
        "f64": pa.array([1.0, 2.0], type=pa.float64()),
    })
    sub = df.select_dtypes(include="float", exclude="float32")
    assert sub.columns == ["f64"]


def test_select_dtypes_all():
    df = _mk({"i": [1], "f": [1.0], "s": ["a"]})
    sub = df.select_dtypes(include="all")
    assert sorted(sub.columns) == ["f", "i", "s"]


def test_select_dtypes_no_match_raises():
    df = _mk({"i": [1, 2]})
    with pytest.raises(ValueError, match="no columns matched"):
        df.select_dtypes(include="string")


def test_select_dtypes_no_include_exclude_raises():
    df = _mk({"i": [1, 2]})
    with pytest.raises(ValueError, match="at least one"):
        df.select_dtypes()


def test_select_dtypes_preserves_filter():
    df = _mk({"x": [1, 2, 3, 4], "v": [10, 20, 30, 40]})
    filtered = df[df["x"] > 1]
    sub = filtered.select_dtypes(include="number")
    # 3 filtered rows survive, but only "v" selected (x is int, still number...)
    # both x and v are int → both kept
    assert set(sub.columns) == {"x", "v"}
    assert sub.count() == 3


# ============ integration with __getitem__ ============


def test_isin_and_between_chain():
    df = _mk({"x": [1, 5, 10, 15, 20], "v": [1, 2, 3, 4, 5]})
    sub = df[(df["x"].isin([1, 5, 10, 15])) & (df["x"].between(5, 15))]
    # isin: 1, 5, 10, 15 → 4 rows; AND between 5..15 → 5, 10, 15
    assert sub.count() == 3
'''

def add_tests():
    print("\n=== add tests/test_v0925g.py ===")
    p = Path(TESTS_PATH)
    if p.exists() and "test_isin_basic" in p.read_text():
        print("  already present — skip")
        return
    p.write_text(NEW_TESTS)
    assert_compiles(p)
    check_no_invalid_escape(p)
    print(f"  OK ({len(NEW_TESTS.splitlines())} L)")


# ============ 5. pytest ============

def run_pytest():
    print("\n=== pre-flight: facade AST guards ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_dataframe.py::test_facade_has_no_as_py",
         "tests/test_dataframe.py::test_facade_has_no_row_loops",
         "tests/test_dataframe_filters.py::test_facade_has_no_as_py_ast",
         "tests/test_dataframe_filters.py::test_facade_has_no_row_loop_ast",
         "-q", "-W", "error::SyntaxWarning"],
        timeout=120,
    )
    if r.returncode != 0:
        print("\n*** FACADE AST GUARD FAILED — ABORT ***")
        sys.exit(1)

    print("\n=== pytest: v0925g + filters + row_filter tests ===")
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


# ============ 6. main ============

def main():
    print("=== v0.9.25g: Phase C C-1 — isin + between + select_dtypes ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"  branch: {br}")

    verify_state()
    extend_row_filter()
    extend_column_proxy()
    add_select_dtypes()
    add_tests()
    run_pytest()

    commit_with_guards(
        "v0.9.25g: Phase C batch C-1 — isin + between + select_dtypes. "
        "First 3 of 20 Pandas P0 features per roadmap. All pure "
        "compositions of existing primitives + surface API. No symbol "
        "moves, no whitelist changes, no new memory models — "
        "deliberately avoiding Phase B's cascade class (§13.4/238, "
        "/261). "
        "Features: "
        "(1) `_ColumnProxy.isin(values)` — PredicateFilter with "
        "'isin' op → pc.is_in per batch. Null row not matched unless "
        "None ∈ values (pandas). Empty list → all False. Non-coercible "
        "value type → TypeError. "
        "(2) `_ColumnProxy.between(left, right, inclusive='both')` — "
        "PredicateFilter with 'between' op → pc.and_(ge/gt, le/lt) per "
        "batch. inclusive ∈ {'both','neither','left','right'}. Null "
        "row not matched (Kleene drop). lo > hi → all False. "
        "(3) `CadwDataFrame.select_dtypes(include=, exclude=)` — "
        "dtype-based column selection. dtype specs: 'number' / 'int' "
        "/ 'integer' / 'float' / 'floating' / 'bool' / 'boolean' / "
        "'string' / 'object' / 'datetime' / 'category' / 'all' / "
        "explicit pyarrow type name. include ∩ ¬exclude. At least one "
        "required. No-match → ValueError. Filters preserved. "
        "All in existing files: row_filter.py (2 new mask branches + "
        "set expansion), dataframe.py (_ColumnProxy 2 methods + "
        "CadwDataFrame.select_dtypes). No new modules, no imports "
        "changed, no test file paths changed (§13.4/261 avoided). "
        "Verification: pytest only (skip subprocess dogfood — Phase "
        "B's `_verify` with triple-quoted 0-indent templates triggered "
        "§13.4/260 IndentationError; pytest via uv run is fresh "
        "subprocess + authoritative). "
        "Tests: tests/test_v0925g.py (24 tests): isin (6 — basic / "
        "empty / string / null-not-matched / null-in-values / bad-"
        "type-raise); between (8 — both / neither / left / right / "
        "null / lo>hi empty / invalid inclusive / float); "
        "select_dtypes (10 — number / exclude-string / int / float / "
        "include-list / include+exclude / all / no-match-raise / "
        "no-args-raise / preserves-filter); integration (1 — isin & "
        "between chain). "
        "Fast suite: ~2314 → ~2338 (+24). "
        "Expected fix-cycles: 0-1 (3 independent features, "
        "independent edits, no cascade). "
        "Next: Phase C batch C-2 (cut/qcut/idxmax/idxmin/pct_change/"
        "reindex).",
        self_path="scripts/patch_v0925g_isin_between_selectdtypes.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.25g applied. Phase C batch C-1 (3 features). "
          "Next: C-2 (cut/qcut/idxmax/idxmin/pct_change/reindex).")


if __name__ == "__main__":
    main()