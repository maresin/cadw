"""v0.9.25h: Phase C batch C-2 — idxmax, idxmin, pct_change."""
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
    assert vals[2] == 3.0            # 4/1 - 1 = 3.0 (finite, not inf)


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
