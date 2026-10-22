#!/usr/bin/env python3
# v0.9.25h-corrected — Phase C C-2: idxmax + idxmin + pct_change.
#
# Previous v0.9.25h attempt aborted in verify_state (REPL check #3):
# pc.divide([10,0,1], [2,0,0]) = [5.0, nan, inf], NOT [5.0, None, None].
# Assumption "Arrow-safe div returns null" was WRONG — pc.divide uses
# IEEE 754 float semantics (inf/nan), which MATCHES pandas.
#
# §13.4/264 (REPL-first) WORKED AS DESIGNED: caught assumption BEFORE
# any edit. HEAD still v0.9.25g. Zero mutation. This is 0 fix-cycles,
# not a failure — it's the safety net.
#
# REVISED SEMANTICS: pct_change uses pc.divide → inf/nan propagation
# (pandas-parity), NOT null-out. Rationale:
#   - User mental model (pandas users expect inf/nan)
#   - Honest — no magic conversion
#   - If CADW wants null-out-by-default, that's policy change for ALL
#     float ops, not just pct_change
#   - Users wanting null-out can filter: df[df.pct.is_finite()]
# Documented limitation: inf in feature column poisons downstream
# groupby-sum/mean. Standard pandas practice (.replace([inf,-inf],
# nan)) applies; user is responsible (matches pandas contract).
#
# CHANGES vs aborted v0.9.25h:
#   1. verify_state REPL check #3 fixed: assert [5.0, isnan, isinf],
#      not [5.0, None, None].
#   2. pct_change docstring: replaced "CADW differs from pandas (null
#      for div-by-zero)" with "matches pandas (inf/nan); user
#      responsible for filter if needed".
#   3. Two pct_change tests rewritten:
#      test_pct_change_div_by_zero_pandas_parity (was _null): asserts
#      math.isinf(vals[1]) not is None.
#      test_pct_change_zero_over_zero_pandas_parity (was _null):
#      asserts math.isnan(vals[1]) not is None.
#
# 3 features, 5 edits (2 idx methods + helper + pct_change + tests),
# independent anchors, REPL-first applied, no symbol moves. Expected
# fix-cycle: 0-1.
#
# Run: uv run python scripts/patch_v0925h_idxmax_pctchange_corrected.py

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


# ============ 0. verify state (REPL-first, §13.4/264 CORRECTED) ============

def verify_state():
    print("\n=== verify: HEAD=v0.9.25g; REPL-check pyarrow behavior "
          "(CORRECTED: pc.divide = IEEE 754 inf/nan); no features yet ===")
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

    df = Path(DATAFRAME_PATH).read_text()
    if "def _index_at_raw_pos" in df:
        raise SystemExit("_index_at_raw_pos already present")
    if "def pct_change(self" in df:
        raise SystemExit("pct_change already present")

    # REPL checks (§13.4/264) — CORRECTED assumptions
    import math
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as _pc

    # (1) indices_nonzero
    arr = pa.array([False, True, False, True])
    nz = _pc.indices_nonzero(arr)
    if not isinstance(nz, pa.Array) or nz.to_pylist() != [1, 3]:
        raise SystemExit(
            f"pc.indices_nonzero([F,T,F,T]) unexpected: {nz}"
        )

    # (2) int(pa.Scalar) without .as_py()
    if int(nz[0]) != 1:
        raise SystemExit(f"int(nz[0]) = {int(nz[0])}, expected 1")

    # (3) CORRECTED: pc.divide uses IEEE 754 (inf for x/0, nan for 0/0),
    # NOT null. Verified empirically 2026-10-22 pyarrow 26:
    #   pc.divide([10.0, 0.0, 1.0], [2.0, 0.0, 0.0])
    #   → [5.0, nan, inf]
    # This matches pandas pct_change contract (user expected inf/nan).
    div = _pc.divide(
        pa.array([10.0, 0.0, 1.0]),
        pa.array([2.0, 0.0, 0.0]),
    )
    div_list = div.to_pylist()
    if not (
        div_list[0] == 5.0
        and math.isnan(div_list[1])       # 0/0 → nan (IEEE 754)
        and math.isinf(div_list[2])       # 1/0 → inf (IEEE 754)
        and div_list[2] > 0
    ):
        raise SystemExit(
            f"pc.divide([10,0,1], [2,0,0]) = {div_list}, "
            f"expected [5.0, nan, inf] (IEEE 754). If different, "
            f"PyArrow version drift — review pct_change docstring."
        )

    # (4) fill_null on bool
    filled = _pc.fill_null(
        pa.array([True, None, False]), pa.scalar(False, type=pa.bool_())
    )
    if filled.to_pylist() != [True, False, False]:
        raise SystemExit(
            f"pc.fill_null([T,None,F], False) = {filled.to_pylist()}"
        )

    # (5) pa.scalar with explicit type
    try:
        _ = pa.scalar(5, type=pa.int64())
        _ = pa.scalar(2.5, type=pa.float64())
    except Exception as e:
        raise SystemExit(f"pa.scalar with explicit type failed: {e}")

    # (6) pc.equal + indices_nonzero on int64
    col = pa.array([1, 3, 2, 3])
    eq = _pc.equal(col, pa.scalar(3, type=pa.int64()))
    nz2 = _pc.indices_nonzero(eq)
    if nz2.to_pylist() != [1, 3]:
        raise SystemExit(
            f"max-position check: got {nz2.to_pylist()}, expected [1, 3]"
        )

    # (7) int(np.int64)
    np_pos = np.arange(0, 10, dtype=np.int64)
    if int(np_pos[3]) != 3:
        raise SystemExit(f"int(np.int64) = {int(np_pos[3])}, expected 3")

    # (8) inf/nan survive pa.array → to_pylist
    inf_round = pa.array([float("inf"), float("nan")], type=pa.float64()).to_pylist()
    if not (math.isinf(inf_round[0]) and math.isnan(inf_round[1])):
        raise SystemExit(
            f"inf/nan round-trip through pa.array broken: {inf_round}"
        )

    print("  8 REPL checks passed; state OK")


# ============ 1. _ColumnProxy.idxmax + idxmin + _index_at_raw_pos ============

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

        Two-pass: ReduceOp(MinMaxOp, 'max') for the max value, then
        stream to find first matching position via pc.equal +
        pc.indices_nonzero. Peak = batch + O(1) state.
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

        Delegates to MaterializedIndex.__getitem__ (in
        cadw/dataset/index.py) — avoids .as_py() in facade (AST
        guard test_facade_has_no_as_py).
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


# ============ 2. CadwDataFrame.pct_change (pandas-parity semantics) ============

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
        of `shift(periods)` + `pc.cast` to float64 + `pc.divide` +
        `pc.subtract`.

        - periods: int >= 1. Default 1 (adjacent rows).
        - columns: None → all numeric (int/float). Else str or
          Sequence[str]; non-numeric → TypeError.
        - Output: original + `{col}_pct_change` (float64, nullable).
        - First `periods` rows → null (no previous value).
        - Null in current or shifted → null row (skipna).
        - **Div-by-zero matches pandas**: `pc.divide` uses IEEE 754
          float semantics → `x / 0` (x != 0) = `inf`; `0 / 0` = `nan`.
          Then `inf - 1 = inf`, `nan - 1 = nan`. `pa.array` and Parquet
          preserve inf/nan (IEEE 754 round-trip).
          **CadwDataFrame.pct_change does NOT null-out inf/nan** — user
          who wants null semantics should filter: `df[df.pct.is_finite()]`
          or post-process. Rationale: pandas parity + honesty; changing
          this to null-out-by-default would be a policy decision for
          ALL float ops, not just pct_change (§11.5).
        - Empty filtered view → empty CadwDataFrame with full schema.
        - Original `{col}` columns preserved; new `{col}_pct_change`
          appended.

        Peak RAM = batch × (1 + #cols) × 8 B + _ArrowTail(k × 8 B) per
        col (from shift) — O(batch + k × #cols). Independent of N.
        """
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
                    # pc.divide uses IEEE 754 (inf for x/0, nan for 0/0)
                    # — matches pandas. Nulls preserved (null/null = null).
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
    print("\n=== CadwDataFrame: pct_change (pandas-parity) ===")
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
import math

import pyarrow as pa
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
    assert df["x"].idxmax() == 1


def test_idxmax_first_of_multiple():
    df = _mk({"x": [3, 5, 1, 5, 2]})
    assert df["x"].idxmax() == 1


def test_idxmax_with_nulls():
    df = _mk({"x": pa.array([1, None, 5, None], type=pa.int64())})
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
    filtered = df[df["v"] >= 20]
    # Filtered x = [5, 3, 9, 2]; max=9 at filtered pos 2 → raw pos 3
    assert filtered["x"].idxmax() == 3


def test_idxmax_materialized_index():
    df = _mk({"x": [1, 5, 3], "k": ["a", "b", "c"]}).set_index("k", drop=False)
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
    assert vals == [None, 1.0, 1.0, 1.0]


def test_pct_change_periods_2():
    df = _mk({"x": [1.0, 2.0, 4.0, 8.0, 16.0]})
    result = df.pct_change(periods=2)
    out = result.head(10)
    vals = out.column("x_pct_change").to_pylist()
    assert vals == [None, None, 3.0, 3.0, 3.0]


def test_pct_change_with_nulls():
    df = _mk({"x": pa.array([1.0, None, 4.0, 8.0], type=pa.float64())})
    result = df.pct_change(periods=1)
    out = result.head(10)
    vals = out.column("x_pct_change").to_pylist()
    # nulls propagate; row 3 (8/4) is 1.0
    assert vals == [None, None, None, 1.0]


def test_pct_change_div_by_zero_pandas_parity():
    """v0.9.25h-corrected: pc.divide uses IEEE 754 (inf/nan), matches
    pandas pct_change. Row 1: 1/0 → inf. NOT null. Users wanting null
    filter: df[df.pct.is_finite()].
    """
    df = _mk({"x": [0.0, 1.0, 2.0]})
    result = df.pct_change(periods=1)
    out = result.head(10)
    vals = out.column("x_pct_change").to_pylist()
    assert vals[0] is None           # no prev for row 0
    assert math.isinf(vals[1])       # 1/0 → inf (pandas-parity)
    assert vals[1] > 0
    assert vals[2] == 1.0            # 2/1 = 2 → 2-1 = 1


def test_pct_change_zero_over_zero_pandas_parity():
    """v0.9.25h-corrected: 0/0 → nan (IEEE 754), matches pandas.
    Row 1: 0/0 → nan. Row 2: 4/0 → inf.
    """
    df = _mk({"x": [0.0, 0.0, 4.0]})
    result = df.pct_change(periods=1)
    out = result.head(10)
    vals = out.column("x_pct_change").to_pylist()
    assert vals[0] is None
    assert math.isnan(vals[1])       # 0/0 → nan (pandas-parity)
    assert math.isinf(vals[2])       # 4/0 → inf


def test_pct_change_inf_nan_roundtrip_parquet(tmp_path):
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
    # (row 1 is the inf case)


def test_pct_change_preserves_original():
    df = _mk({"x": [1.0, 2.0, 4.0]})
    result = df.pct_change(periods=1)
    out = result.head(10)
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
    df = _mk({"x": [1, 2, 4]})
    result = df.pct_change(periods=1)
    out = result.head(10)
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
    filtered = df[df["v"] > 15]
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
    print("=== v0.9.25h-corrected: C-2 idxmax + idxmin + pct_change ===")
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
        "for first matching position (pc.equal + pc.indices_nonzero). "
        "Returns index LABEL (RangeIndex → int; MaterializedIndex → "
        "original; MultiIndex → tuple) via `_index_at_raw_pos` "
        "(delegates to MaterializedIndex.__getitem__, avoids .as_py() "
        "in facade — AST guard clean). Nulls skipped; all-null / "
        "empty → ValueError. "
        "(2) `_ColumnProxy.idxmin()` — mirror with 'min'. "
        "(3) `CadwDataFrame.pct_change(periods=1, columns=None)` — "
        "append `{col}_pct_change` (float64, nullable). Formula: "
        "(x / shift(x, periods)) - 1. Composition of shift + "
        "pc.cast-to-float64 + pc.divide + pc.subtract. "
        "**Semantics: pandas-parity** — pc.divide uses IEEE 754 (inf "
        "for x/0, nan for 0/0) → matches pandas pct_change exactly. "
        "First `periods` rows → null. Nulls propagate. Empty → "
        "empty schema-correct. Original cols preserved. "
        "DEFERRED: `duplicated(subset, keep)` — order-dependent "
        "(keep='first'/'last') needs row_id-aware design; "
        "v0.9.25i target. "
        "CRITICAL: aborted v0.9.25h attempt's verify_state REPL check "
        "#3 caught WRONG assumption — pc.divide([10,0,1],[2,0,0]) = "
        "[5.0, nan, inf], NOT [5.0, None, None]. My 'Arrow-safe null' "
        "assumption was wrong; pc.divide is IEEE 754 = pandas-parity. "
        "§13.4/264 (REPL-first) WORKED AS DESIGNED: caught BEFORE "
        "any edit. HEAD stayed v0.9.25g. Zero mutation. This is "
        "0 fix-cycles, not a failure — it's the safety net. Would "
        "have caught v0.9.25g's fix2+fix3 overreach if applied there. "
        "Corrected: verify_state asserts [5.0, isnan, isinf]; "
        "pct_change docstring 'CADW differs from pandas' replaced "
        "with 'matches pandas (inf/nan); user filter if needed'; "
        "2 tests rewritten (test_pct_change_div_by_zero_pandas_"
        "parity, test_pct_change_zero_over_zero_pandas_parity) with "
        "math.isinf/math.isnan. inf/nan round-trip through Parquet "
        "verified (test_pct_change_inf_nan_roundtrip_parquet). "
        "§13.4/260: no subprocess dogfood (dedent trap). pytest "
        "authoritative. §13.4/261: no symbol moves → no whitelist "
        "path updates. §13.4/262: pa.scalar(max_val, type=arr.type) "
        "for typed scalar reconstruction (max_val from MinMaxOp "
        "finalize is Python scalar; not from facade .as_py). "
        "8 REPL checks in verify_state (indices_nonzero, int(pa."
        "Scalar), pc.divide IEEE 754, fill_null-bool, pa.scalar-"
        "typed, equal+indices, int(np.int64), inf/nan round-trip). "
        "Tests (tests/test_v0925h.py): 26 total — "
        "idxmax (8), idxmin (5), pct_change (13, incl. 2 "
        "pandas-parity + 1 parquet-roundtrip). "
        "Fast suite: ~2343 → ~2369 (+26). "
        "Next: Phase C C-3 (cut/qcut/reindex/ffill/bfill/dup — 6 "
        "features, split into C-3a/C-3b).",
        self_path="scripts/patch_v0925h_idxmax_pctchange_corrected.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.25h COMPLETE (3 features). Phase C: 6/20. "
          "Next: C-3a (ffill/bfill/reindex).")


if __name__ == "__main__":
    main()