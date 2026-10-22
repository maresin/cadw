#!/usr/bin/env python3
# v0.9.25h — Phase C batch C-2: idxmax + idxmin + pct_change.
#
# 3 independent surface API features. Composition of existing
# primitives + Arrow kernels. No symbol moves, no whitelist changes,
# no new memory models, no new modules.
#
# FEATURES:
#   1. `_ColumnProxy.idxmax()` — index label of first max position.
#   2. `_ColumnProxy.idxmin()` — index label of first min position.
#   3. `CadwDataFrame.pct_change(periods=1, columns=None)` — append
#      `{col}_pct_change` (float64, nullable). Composition of
#      shift(periods) + pc.divide + pc.subtract.
#
# DEFERRED: `duplicated(subset, keep)` — order-dependent (keep='first'/
# 'last') requires row_id-aware composition with stable order
# tracking. Set-only keep=False doable, but pandas contract includes
# order. Move to v0.9.25i with careful design.
#
# SEMANTICS:
#   idxmax/idxmin:
#     - Returns index LABEL (not position). RangeIndex → int;
#       MaterializedIndex → original label; MultiIndex → tuple.
#     - First occurrence wins (pandas convention, row_id order).
#     - Nulls skipped (skipna=True). All-null → ValueError.
#     - Empty filtered view → ValueError.
#     - Uses _iter_filtered_batches_with_positions to correctly
#       map filtered positions → raw → index label.
#
#   pct_change:
#     - `(x / shift(x, periods)) - 1` per row; null if either null.
#     - Div-by-zero: pc.divide returns null (Arrow-safe). Differs
#       from pandas which gives inf for x != 0 / 0. Documented.
#     - First `periods` rows → null. Empty → empty schema-correct.
#     - Output float64 always (int cols → float64 result).
#     - columns=None → all numeric (int/float); non-numeric → TypeError.
#
# §13.4/264 APPLIED (REPL-first): verify_state does 7 pyarrow REPL
# checks BEFORE any edit. If PyArrow behavior differs from assumed
# (indices_nonzero, int(Scalar), divide-by-zero semantics, fill_null
# bool) → abort with clear message. Would have caught fix2+fix3
# overreach in v0.9.25g if applied there.
#
# §13.4/262/263 (imports/dtype): idxmax uses `pa.scalar(max_val,
# type=arr.type)` — max_val from MinMaxOp finalize is Python scalar
# (from .as_py inside reduce_op, not facade), reconstructs typed
# scalar correctly for int64/float64/timestamp. Tests verify int64 +
# float64 (float32 edge case documented as limitation).
#
# §13.4/260 (dedent): no subprocess dogfood with triple-quoted
# templates. pytest is authoritative.
#
# §13.4/261 (whitelist paths): no symbol moves → no path updates.
# `_index_at_raw_pos` added inside `_ColumnProxy` in existing file;
# no whitelist entry to update.
#
# Fix-cycle budget: 0-1 (3 features, independent anchors, REPL-first).
# Churn defense: each edit independent; anchor miss → 1-branch fix.
#
# Run: uv run python scripts/patch_v0925h_idxmax_pctchange.py

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
TESTS_PATH = "tests/test_v0925h.py"


# ============ 0. verify state (REPL-first, §13.4/264) ============

def verify_state():
    print("\n=== verify: HEAD=v0.9.25g; REPL-check pyarrow behavior; "
          "no v0.9.25h features yet ===")
    log = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout
    if "v0.9.25g" not in log:
        raise SystemExit(
            f"expected HEAD=v0.9.25g; got {log[:200]!r}"
        )
    if "v0.9.25h" in log:
        raise SystemExit("v0.9.25h already committed — patch misplaced")

    # Feature absence
    df = Path(DATAFRAME_PATH).read_text()
    if "def idxmax(self" in df and "PredicateFilter(self._column, \"isin\"" not in df:
        # crude check — but idxmax is _ColumnProxy method
        pass
    if 'def _index_at_raw_pos' in df:
        raise SystemExit("_index_at_raw_pos already present — v0.9.25h applied?")
    if "def pct_change(self" in df:
        raise SystemExit("pct_change already present — v0.9.25h applied?")

    # REPL checks (§13.4/264) — actual pyarrow behavior for our assumptions
    import pyarrow as pa
    import pyarrow.compute as _pc

    # (1) indices_nonzero exists and returns Array
    arr = pa.array([False, True, False, True])
    nz = _pc.indices_nonzero(arr)
    if not isinstance(nz, pa.Array):
        raise SystemExit(
            f"pc.indices_nonzero returned {type(nz)}, expected pa.Array"
        )
    if nz.to_pylist() != [1, 3]:
        raise SystemExit(
            f"pc.indices_nonzero([F,T,F,T]) = {nz.to_pylist()}, "
            f"expected [1, 3]"
        )

    # (2) int(pa.Scalar) works without .as_py()
    int_val = int(nz[0])
    if int_val != 1:
        raise SystemExit(f"int(nz[0]) = {int_val}, expected 1")

    # (3) pc.divide(x, 0) = null (Arrow-safe, not inf)
    div = _pc.divide(pa.array([10.0, 0.0, 1.0]), pa.array([2.0, 0.0, 0.0]))
    div_list = div.to_pylist()
    if div_list != [5.0, None, None]:
        raise SystemExit(
            f"pc.divide([10,0,1], [2,0,0]) = {div_list}, "
            f"expected [5.0, None, None]. PyArrow behavior differs from "
            f"assumption (Arrow-safe div returns null where divisor==0, "
            f"not inf)."
        )

    # (4) pc.fill_null on bool
    filled = _pc.fill_null(
        pa.array([True, None, False]), pa.scalar(False, type=pa.bool_())
    )
    if filled.to_pylist() != [True, False, False]:
        raise SystemExit(
            f"pc.fill_null([T,None,F], False) = {filled.to_pylist()}"
        )

    # (5) pa.scalar(max_val, type=arr.type) works for int64/float64
    try:
        _ = pa.scalar(5, type=pa.int64())
        _ = pa.scalar(2.5, type=pa.float64())
    except Exception as e:
        raise SystemExit(
            f"pa.scalar with explicit type failed: {e}"
        )

    # (6) pc.equal + pc.indices_nonzero on int64 with max value
    col = pa.array([1, 3, 2, 3])
    eq = _pc.equal(col, pa.scalar(3, type=pa.int64()))
    nz2 = _pc.indices_nonzero(eq)
    if nz2.to_pylist() != [1, 3]:
        raise SystemExit(
            f"max-position check: got {nz2.to_pylist()}, expected [1, 3]"
        )

    # (7) int(np.int64) works (positions array from np.arange)
    import numpy as np
    np_pos = np.arange(0, 10, dtype=np.int64)
    if int(np_pos[3]) != 3:
        raise SystemExit(f"int(np.int64) = {int(np_pos[3])}, expected 3")

    print("  7 REPL checks passed; state OK")


# ============ 1. add _ColumnProxy.idxmax + idxmin + _index_at_raw_pos ============

OLD_PROXY_END = '''    def between(self, left, right, *, inclusive: str = "both"):
        """Range predicate (v0.9.25g). `df[df['a'].between(0, 10)]`.

        - inclusive ∈ {'both' (default), 'neither', 'left', 'right'}.
        - Null row not matched (Kleene drop).
        - left > right → all False (no argument reorder).
        """
        return PredicateFilter(
            self._column, "between", (left, right, inclusive),
        )'''


NEW_PROXY_END = OLD_PROXY_END + '''

    def idxmax(self):
        """Index label of first row with maximum value (v0.9.25h).

        - Returns index LABEL (RangeIndex → int; MaterializedIndex →
          original label; MultiIndex → tuple). Not positional.
        - First occurrence in row_id order (pandas `idxmax` semantics).
        - Nulls skipped. All-null column → ValueError.
        - Empty filtered view → ValueError.

        Two passes: ReduceOp(MinMaxOp) for max value, then stream to
        find first position. Peak = batch + O(1) state.
        """
        from cadw.ops.reduce import MinMaxOp, reduce
        import pyarrow.compute as _pc

        max_val = reduce(
            self._parent._iter_filtered_batches(),
            MinMaxOp(self._column, "max"),
            self._parent._memory_guard,
            log=False,
        )
        if max_val is None:
            raise ValueError(
                f"idxmax: column {self._column!r} has no non-null values"
            )
        for batch, positions in (
            self._parent._iter_filtered_batches_with_positions()
        ):
            arr = batch.column(self._column)
            eq = _pc.equal(arr, pa.scalar(max_val, type=arr.type))
            eq = _pc.fill_null(eq, pa.scalar(False, type=pa.bool_()))
            nonzero = _pc.indices_nonzero(eq)
            if len(nonzero) > 0:
                local_pos = int(nonzero[0])
                raw_pos = int(positions[local_pos])
                return self._index_at_raw_pos(raw_pos)
        raise RuntimeError("idxmax: value not found (unreachable)")

    def idxmin(self):
        """Index label of first row with minimum value (v0.9.25h).

        Mirror of `idxmax` (see docstring). Nulls skipped.
        """
        from cadw.ops.reduce import MinMaxOp, reduce
        import pyarrow.compute as _pc

        min_val = reduce(
            self._parent._iter_filtered_batches(),
            MinMaxOp(self._column, "min"),
            self._parent._memory_guard,
            log=False,
        )
        if min_val is None:
            raise ValueError(
                f"idxmin: column {self._column!r} has no non-null values"
            )
        for batch, positions in (
            self._parent._iter_filtered_batches_with_positions()
        ):
            arr = batch.column(self._column)
            eq = _pc.equal(arr, pa.scalar(min_val, type=arr.type))
            eq = _pc.fill_null(eq, pa.scalar(False, type=pa.bool_()))
            nonzero = _pc.indices_nonzero(eq)
            if len(nonzero) > 0:
                local_pos = int(nonzero[0])
                raw_pos = int(positions[local_pos])
                return self._index_at_raw_pos(raw_pos)
        raise RuntimeError("idxmin: value not found (unreachable)")

    def _index_at_raw_pos(self, raw_pos: int):
        """Look up index label at raw (unfiltered) position.

        Delegates to MaterializedIndex.__getitem__ (in dataset/index.py)
        for label extraction — avoids .as_py() in facade (AST guard).
        """
        if self._parent._index_kind == "range":
            return self._parent._index_start + int(raw_pos)
        if self._parent._index_kind == "materialized":
            idx = MaterializedIndex(self._parent._index_values)
            return idx[raw_pos]
        if self._parent._index_kind == "multi":
            idx = MaterializedMultiIndex(self._parent._index_levels)
            return idx[raw_pos]
        raise RuntimeError(
            f"idxmax/idxmin: unknown index_kind "
            f"{self._parent._index_kind!r}"
        )'''


def add_idx_methods():
    print("\n=== _ColumnProxy: idxmax + idxmin + _index_at_raw_pos ===")
    applied = apply_or_fail(
        DATAFRAME_PATH, OLD_PROXY_END, NEW_PROXY_END,
        label="proxy_idx",
    )
    if applied:
        assert_compiles(DATAFRAME_PATH)
        check_no_invalid_escape(DATAFRAME_PATH)
    print("  applied" if applied else "  already applied — skip")


# ============ 2. CadwDataFrame.pct_change ============

# Anchor: insert right after select_dtypes method (added v0.9.25g), before
# _repr_html_ or next method. Use end of select_dtypes body as anchor.
PCT_ANCHOR_TAIL = '''        if not keep:
            raise ValueError(
                f"select_dtypes: no columns matched "
                f"include={include!r} exclude={exclude!r}; "
                f"available dtypes={self.dtypes()}"
            )
        return self[keep]'''


PCT_NEW_TAIL = PCT_ANCHOR_TAIL + '''

    def pct_change(self, periods: int = 1, columns=None) -> "CadwDataFrame":
        """Percent change between rows (v0.9.25h).

        Formula: `(x / shift(x, periods)) - 1` per row. Composition
        of `shift(periods)` + `pc.divide` + `pc.subtract`.

        - periods: int >= 1. Default 1 (adjacent rows).
        - columns: None → all numeric (int/float). Else str or
          Sequence[str]; non-numeric → TypeError.
        - Output: original + `{col}_pct_change` (float64, nullable).
        - First `periods` rows → null (no previous value).
        - Null in current or shifted → null row (skipna).
        - Div-by-zero: pc.divide returns null (Arrow-safe); pandas
          gives inf for x != 0 / 0. Documented limitation: for
          pandas-inf behavior use `.shift()` + `pc.divide_checked`
          composition externally.
        - Empty filtered view → empty CadwDataFrame with full schema.
        - Original `{col}` columns preserved unchanged; new
          `{col}_pct_change` appended.

        Peak RAM = batch × (1 + #cols) × 8 B + _ArrowTail(k × 8 B) per
        col (from shift) — O(batch + k × #cols). Independent of N.
        """
        import pyarrow as pa  # already imported; redundant for clarity
        import pyarrow.compute as _pc
        from cadw.dataset import Dataset

        if not isinstance(periods, int) or periods < 1:
            raise ValueError(
                f"pct_change: periods must be int >= 1; got {periods!r}"
            )
        if columns is None:
            cols = [
                c for c in self._columns
                if pa.types.is_integer(
                    self._dataset.schema.field(c).type
                ) or pa.types.is_floating(
                    self._dataset.schema.field(c).type
                )
            ]
        elif isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)
        if not cols:
            raise ValueError(
                "pct_change: no numeric columns selected "
                "(pass columns=None for auto int/float, or explicit)"
            )
        for c in cols:
            if c not in self._columns:
                raise KeyError(
                    f"pct_change: column {c!r} not in {self._columns}"
                )
            t = self._dataset.schema.field(c).type
            if not (pa.types.is_integer(t) or pa.types.is_floating(t)):
                raise TypeError(
                    f"pct_change: column {c!r} has type {t}; numeric "
                    f"required"
                )

        shifted = self.shift(periods, columns=cols)

        out_fields = [
            self._dataset.schema.field(c) for c in self._columns
        ]
        for c in cols:
            out_fields.append(
                pa.field(f"{c}_pct_change", pa.float64(), nullable=True)
            )
        out_schema = pa.schema(out_fields)

        batch_size = 10_000

        def _iter():
            orig_it = self._iter_arrow_batches(batch_size=batch_size)
            shifted_it = shifted._iter_arrow_batches(batch_size=batch_size)
            for ob, sb in zip(orig_it, shifted_it):
                if ob.num_rows != sb.num_rows:
                    raise RuntimeError(
                        f"pct_change: batch size mismatch "
                        f"(orig={ob.num_rows}, shifted={sb.num_rows})"
                    )
                arrays = [ob.column(c) for c in self._columns]
                for c in cols:
                    curr_f = _pc.cast(ob.column(c), pa.float64())
                    prev_f = _pc.cast(sb.column(c), pa.float64())
                    ratio = _pc.divide(curr_f, prev_f)
                    pct = _pc.subtract(
                        ratio, pa.scalar(1.0, type=pa.float64())
                    )
                    arrays.append(pct)
                yield pa.RecordBatch.from_arrays(
                    arrays, schema=out_schema,
                )

        new_ds = Dataset.from_batches(
            out_schema, _iter(), self._new_dataset_dir(),
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )'''


def add_pct_change():
    print("\n=== CadwDataFrame: pct_change ===")
    applied = apply_or_fail(
        DATAFRAME_PATH, PCT_ANCHOR_TAIL, PCT_NEW_TAIL,
        label="pct_change",
    )
    if applied:
        assert_compiles(DATAFRAME_PATH)
        check_no_invalid_escape(DATAFRAME_PATH)
    print("  applied" if applied else "  already applied — skip")


# ============ 3. tests ============

NEW_TESTS = '''"""v0.9.25h: Phase C batch C-2 — idxmax, idxmin, pct_change."""
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard


def _mk(data, mem_mb=20):
    return CadwDataFrame.from_table(
        pa.table(data),
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
    )


# ============ idxmax ============


def test_idxmax_basic():
    df = _mk({"x": [1, 5, 3, 2]})
    # max=5 at position 1, RangeIndex → label 1
    assert df["x"].idxmax() == 1


def test_idxmax_first_of_multiple():
    df = _mk({"x": [3, 5, 1, 5, 2]})
    # Two maxes at pos 1 and 3; first wins
    assert df["x"].idxmax() == 1


def test_idxmax_with_nulls():
    df = _mk({"x": pa.array([1, None, 5, None], type=pa.int64())})
    # max=5 at position 2, nulls skipped
    assert df["x"].idxmax() == 2


def test_idxmax_all_null_raises():
    df = _mk({"x": pa.array([None, None, None], type=pa.int64())})
    with pytest.raises(ValueError, match="no non-null values"):
        df["x"].idxmax()


def test_idxmax_empty_raises():
    df = _mk({"x": [1, 2, 3], "v": [10, 20, 30]})
    empty = df[df["v"] > 100]
    with pytest.raises(ValueError, match="no non-null values"):
        empty["x"].idxmax()


def test_idxmax_with_filter():
    df = _mk({"x": [1, 5, 3, 9, 2], "v": [10, 20, 30, 40, 50]})
    filtered = df[df["v"] >= 20]  # rows at raw positions 1, 2, 3, 4
    # After filter: x = [5, 3, 9, 2]; max=9 at filtered position 2
    # → raw position 3 → RangeIndex label 3
    assert filtered["x"].idxmax() == 3


def test_idxmax_materialized_index():
    df = _mk({"x": [1, 5, 3], "k": ["a", "b", "c"]}).set_index("k", drop=False)
    # Index = ["a", "b", "c"]; max x=5 at position 1 → label "b"
    assert df["x"].idxmax() == "b"


def test_idxmax_float():
    df = _mk({"x": [1.5, 2.5, 0.5]})
    assert df["x"].idxmax() == 1


# ============ idxmin ============


def test_idxmin_basic():
    df = _mk({"x": [3, 1, 5, 2]})
    assert df["x"].idxmin() == 1


def test_idxmin_first_of_multiple():
    df = _mk({"x": [5, 1, 3, 1, 2]})
    assert df["x"].idxmin() == 1


def test_idxmin_with_nulls():
    df = _mk({"x": pa.array([5, None, 1, None], type=pa.int64())})
    assert df["x"].idxmin() == 2


def test_idxmin_all_null_raises():
    df = _mk({"x": pa.array([None, None], type=pa.int64())})
    with pytest.raises(ValueError, match="no non-null values"):
        df["x"].idxmin()


def test_idxmin_materialized_index():
    df = _mk({"x": [5, 1, 3], "k": ["a", "b", "c"]}).set_index("k", drop=False)
    assert df["x"].idxmin() == "b"


# ============ pct_change ============


def test_pct_change_basic():
    df = _mk({"x": [1.0, 2.0, 4.0, 8.0]})
    result = df.pct_change(periods=1)
    out = result.head(10)
    vals = out.column("x_pct_change").to_pylist()
    # (1/null) null, (2/1)-1=1, (4/2)-1=1, (8/4)-1=1
    assert vals == [None, 1.0, 1.0, 1.0]


def test_pct_change_periods_2():
    df = _mk({"x": [1.0, 2.0, 4.0, 8.0, 16.0]})
    result = df.pct_change(periods=2)
    out = result.head(10)
    vals = out.column("x_pct_change").to_pylist()
    # (4/1)-1=3, (8/2)-1=3, (16/4)-1=3; first two null
    assert vals == [None, None, 3.0, 3.0, 3.0]


def test_pct_change_with_nulls():
    df = _mk({"x": pa.array([1.0, None, 4.0, 8.0], type=pa.float64())})
    result = df.pct_change(periods=1)
    out = result.head(10)
    vals = out.column("x_pct_change").to_pylist()
    # null row propagates null to self and next
    # row 0: prev=null → null
    # row 1: current=null → null
    # row 2: current=4, prev=null → null
    # row 3: current=8, prev=4 → 1.0
    assert vals == [None, None, None, 1.0]


def test_pct_change_div_by_zero_null():
    """pc.divide returns null where divisor==0 (Arrow-safe). Differs
    from pandas which gives inf for nonzero/0. Documented limitation.
    """
    df = _mk({"x": [0.0, 1.0, 2.0]})
    result = df.pct_change(periods=1)
    out = result.head(10)
    vals = out.column("x_pct_change").to_pylist()
    # row 1: current=1, prev=0 → null (Arrow-safe; pandas: inf)
    # row 2: current=2, prev=1 → 1.0
    assert vals == [None, None, 1.0]


def test_pct_change_zero_over_zero_null():
    df = _mk({"x": [0.0, 0.0, 4.0]})
    result = df.pct_change(periods=1)
    out = result.head(10)
    vals = out.column("x_pct_change").to_pylist()
    # row 1: 0/0 → null; row 2: 4/0 → null (Arrow-safe)
    assert vals == [None, None, None]


def test_pct_change_preserves_original():
    df = _mk({"x": [1.0, 2.0, 4.0]})
    result = df.pct_change(periods=1)
    out = result.head(10)
    # Both original x and new pct column present
    assert set(out.schema.names) == {"x", "x_pct_change"}
    assert out.column("x").to_pylist() == [1.0, 2.0, 4.0]


def test_pct_change_multiple_columns():
    df = _mk({"a": [1.0, 2.0, 4.0], "b": [10.0, 20.0, 40.0]})
    result = df.pct_change(periods=1)
    out = result.head(10)
    a_vals = out.column("a_pct_change").to_pylist()
    b_vals = out.column("b_pct_change").to_pylist()
    assert a_vals == [None, 1.0, 1.0]
    assert b_vals == [None, 1.0, 1.0]


def test_pct_change_columns_subset():
    df = _mk({"a": [1.0, 2.0, 4.0], "b": [10.0, 20.0, 40.0]})
    result = df.pct_change(periods=1, columns=["a"])
    out = result.head(10)
    assert "a_pct_change" in out.schema.names
    assert "b_pct_change" not in out.schema.names


def test_pct_change_int_to_float():
    df = _mk({"x": [1, 2, 4]})  # int64 column
    result = df.pct_change(periods=1)
    out = result.head(10)
    # Output must be float64 (int inputs → float output)
    assert out.schema.field("x_pct_change").type == pa.float64()


def test_pct_change_invalid_periods():
    df = _mk({"x": [1.0, 2.0]})
    with pytest.raises(ValueError, match="periods must be int"):
        df.pct_change(periods=0)
    with pytest.raises(ValueError, match="periods must be int"):
        df.pct_change(periods=-1)


def test_pct_change_non_numeric_raises():
    df = _mk({"x": [1.0, 2.0], "s": ["a", "b"]})
    with pytest.raises(TypeError, match="numeric required"):
        df.pct_change(periods=1, columns=["s"])


def test_pct_change_no_numeric_raises():
    df = _mk({"s": ["a", "b"]})
    with pytest.raises(ValueError, match="no numeric columns"):
        df.pct_change(periods=1)


def test_pct_change_filtered():
    df = _mk({"x": [1.0, 2.0, 4.0, 8.0], "v": [10, 20, 30, 40]})
    filtered = df[df["v"] > 15]  # rows at positions 1, 2, 3
    result = filtered.pct_change(periods=1)
    out = result.head(10)
    # filtered x = [2, 4, 8]; pct: [null, 1.0, 1.0]
    assert out.column("x_pct_change").to_pylist() == [None, 1.0, 1.0]


def test_pct_change_empty():
    df = _mk({"x": [1.0, 2.0], "v": [10, 20]})
    empty = df[df["v"] > 100]
    result = empty.pct_change(periods=1)
    out = result.head(10)
    assert out.num_rows == 0
    assert "x_pct_change" in out.schema.names
'''


def add_tests():
    print("\n=== add tests/test_v0925h.py ===")
    p = Path(TESTS_PATH)
    if p.exists() and "test_idxmax_basic" in p.read_text():
        print("  already present — skip")
        return
    p.write_text(NEW_TESTS)
    assert_compiles(p)
    check_no_invalid_escape(p)
    print(f"  OK ({len(NEW_TESTS.splitlines())} L)")


# ============ 4. pytest ============

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

    print("\n=== pytest: v0925h + filters + shift ===")
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


# ============ 5. main ============

def main():
    print("=== v0.9.25h: Phase C C-2 — idxmax + idxmin + pct_change ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"  branch: {br}")

    verify_state()
    add_idx_methods()
    add_pct_change()
    add_tests()
    run_pytest()

    commit_with_guards(
        "v0.9.25h: Phase C batch C-2 — idxmax + idxmin + pct_change. "
        "3 of 20 Pandas P0 features (after C-1's isin/between/"
        "select_dtypes at v0.9.25g). All pure compositions of "
        "existing primitives + Arrow kernels. No symbol moves, no "
        "whitelist changes, no new modules, no new memory models. "
        "Features: "
        "(1) `_ColumnProxy.idxmax()` — index label of first row with "
        "maximum value. Two-pass: ReduceOp(MinMaxOp, 'max') + stream "
        "to find first matching position via pc.equal + pc."
        "indices_nonzero. Returns index LABEL (RangeIndex → int; "
        "MaterializedIndex → original label; MultiIndex → tuple) via "
        "`_index_at_raw_pos` (delegates to MaterializedIndex.__"
        "getitem__, avoiding .as_py() in facade — AST guard clean). "
        "Nulls skipped; all-null → ValueError; empty filter → "
        "ValueError. Peak = batch + O(1). "
        "(2) `_ColumnProxy.idxmin()` — mirror with MinMaxOp('min'). "
        "(3) `CadwDataFrame.pct_change(periods=1, columns=None)` — "
        "append `{col}_pct_change` (float64, nullable). Formula: "
        "(x / shift(x, periods)) - 1 per row. Composition of "
        "shift(periods) + pc.cast to float64 + pc.divide + pc."
        "subtract. Nulls propagate (skipna); div-by-zero → null "
        "(pc.divide Arrow-safe, differs from pandas inf — documented). "
        "First `periods` rows → null. Int cols → float64 output. "
        "Multiple columns via columns=None (auto numeric) or "
        "explicit list. Peak = batch + k×8 B × cols (from shift's "
        "_ArrowTail). Original cols preserved. "
        "DEFERRED: `duplicated(subset, keep)` — order-dependent "
        "(keep='first'/'last') requires row_id-aware stable-order "
        "tracking; set-only keep=False doable via groupby+count but "
        "pandas contract includes order. Move to v0.9.25i with "
        "careful design. "
        "§13.4/264 APPLIED (REPL-first, first patch after fix2/fix3 "
        "lesson): verify_state does 7 pyarrow REPL checks BEFORE any "
        "edit — indices_nonzero exists + returns Array; int(pa."
        "Scalar) works without .as_py(); pc.divide(x,0) = null (not "
        "inf — verified, wouldn't have guessed); pc.fill_null on "
        "bool; pa.scalar(int, int64-type) + pa.scalar(float, "
        "float64-type); pc.equal+indices_nonzero on int64; int(np."
        "int64). If any pyarrow behavior differs from assumption → "
        "abort with clear message. Would have caught v0.9.25g's "
        "fix2+fix3 overreach. "
        "§13.4/260: no subprocess dogfood (dedent trap). pytest "
        "authoritative. §13.4/261: no symbol moves → no whitelist "
        "path updates. §13.4/262: pa.scalar(max_val, type=arr.type) "
        "reconstructs typed scalar; max_val from MinMaxOp finalize "
        "is Python scalar (from .as_py in reduce_op, not facade). "
        "Tests (tests/test_v0925h.py): 26 total — "
        "idxmax (8: basic / first-of-multiple / with-nulls / "
        "all-null-raise / empty-raise / filter / materialized-index "
        "/ float); "
        "idxmin (5: basic / first-of-multiple / with-nulls / "
        "all-null / materialized-index); "
        "pct_change (13: basic / periods=2 / with-nulls / div-by-"
        "zero-null / zero-over-zero-null / preserves-original / "
        "multiple-cols / columns-subset / int-to-float / invalid-"
        "periods / non-numeric-raise / no-numeric-raise / filtered "
        "/ empty). "
        "Fast suite: ~2343 → ~2369 (+26). "
        "Next: Phase C C-3 (cut/qcut/reindex/ffill/bfill/dup — 6 "
        "features, more complex, likely split C-3a/C-3b).",
        self_path="scripts/patch_v0925h_idxmax_pctchange.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.25h applied. Phase C C-2 done. "
          "Next: C-3 (cut/qcut/reindex/ffill/bfill/dup).")


if __name__ == "__main__":
    main()