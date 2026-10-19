"""v0.6.2c: rolling(window, agg)."""
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard


def make_df(tmp_path, name="L", data=None, chunk=2):
    if data is None:
        data = {"v": [1.0, 2.0, 3.0, 4.0, 5.0]}
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def read_col(df, tmp_path, col):
    out = tmp_path / "o.parquet"
    df.write_parquet(out)
    return pq.read_table(out).column(col).to_pylist()


def approx(a, b, tol=1e-9):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(a - b) < tol


# ---------- basic windows ----------

def test_rolling_sum_w3(tmp_path):
    df = make_df(tmp_path)
    res = df.rolling(3, "sum")
    # [null, null, 1+2+3=6, 2+3+4=9, 3+4+5=12]
    assert [x is None for x in read_col(res, tmp_path, "v_rolling_sum")] == [
        True, True, False, False, False,
    ]
    vals = read_col(res, tmp_path, "v_rolling_sum")
    assert approx(vals[2], 6.0)
    assert approx(vals[3], 9.0)
    assert approx(vals[4], 12.0)


def test_rolling_mean_w3(tmp_path):
    df = make_df(tmp_path)
    res = df.rolling(3, "mean")
    vals = read_col(res, tmp_path, "v_rolling_mean")
    assert vals[0] is None and vals[1] is None
    assert approx(vals[2], 2.0)
    assert approx(vals[3], 3.0)
    assert approx(vals[4], 4.0)


def test_rolling_min_w3(tmp_path):
    df = make_df(tmp_path, data={"v": [3.0, 1.0, 4.0, 2.0, 5.0]})
    res = df.rolling(3, "min")
    vals = read_col(res, tmp_path, "v_rolling_min")
    assert vals[0] is None and vals[1] is None
    assert approx(vals[2], 1.0)
    assert approx(vals[3], 1.0)
    assert approx(vals[4], 2.0)


def test_rolling_max_w3(tmp_path):
    df = make_df(tmp_path, data={"v": [3.0, 1.0, 4.0, 2.0, 5.0]})
    res = df.rolling(3, "max")
    vals = read_col(res, tmp_path, "v_rolling_max")
    assert vals[0] is None and vals[1] is None
    assert approx(vals[2], 4.0)
    assert approx(vals[3], 4.0)
    assert approx(vals[4], 5.0)


def test_rolling_count_w3(tmp_path):
    df = make_df(tmp_path, data={"v": [1.0, 2.0, 3.0, 4.0, 5.0]})
    res = df.rolling(3, "count")
    vals = read_col(res, tmp_path, "v_rolling_count")
    assert vals[0] is None and vals[1] is None
    assert vals[2] == 3 and vals[3] == 3 and vals[4] == 3


# ---------- min_periods ----------

def test_rolling_min_periods_1(tmp_path):
    df = make_df(tmp_path)
    res = df.rolling(3, "mean", min_periods=1)
    vals = read_col(res, tmp_path, "v_rolling_mean")
    assert approx(vals[0], 1.0)
    assert approx(vals[1], 1.5)
    assert approx(vals[2], 2.0)
    assert approx(vals[3], 3.0)
    assert approx(vals[4], 4.0)


def test_rolling_min_periods_2(tmp_path):
    df = make_df(tmp_path)
    res = df.rolling(3, "mean", min_periods=2)
    vals = read_col(res, tmp_path, "v_rolling_mean")
    assert vals[0] is None
    assert approx(vals[1], 1.5)
    assert approx(vals[2], 2.0)


def test_rolling_min_periods_invalid(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(ValueError, match="min_periods"):
        df.rolling(3, "sum", min_periods=0)
    with pytest.raises(ValueError, match="min_periods"):
        df.rolling(3, "sum", min_periods=4)


# ---------- window=1 ----------

def test_rolling_window_1(tmp_path):
    df = make_df(tmp_path)
    res = df.rolling(1, "sum")
    vals = read_col(res, tmp_path, "v_rolling_sum")
    assert [round(x, 9) for x in vals] == [1.0, 2.0, 3.0, 4.0, 5.0]


# ---------- window > n ----------

def test_rolling_window_gt_n(tmp_path):
    df = make_df(tmp_path, data={"v": [1.0, 2.0]})
    res = df.rolling(10, "sum")
    vals = read_col(res, tmp_path, "v_rolling_sum")
    # all null (min_periods=10 > n)
    assert vals == [None, None]


def test_rolling_window_gt_n_min_periods_1(tmp_path):
    df = make_df(tmp_path, data={"v": [1.0, 2.0]})
    res = df.rolling(10, "sum", min_periods=1)
    vals = read_col(res, tmp_path, "v_rolling_sum")
    assert approx(vals[0], 1.0)
    assert approx(vals[1], 3.0)


# ---------- null handling ----------

def test_rolling_with_nulls(tmp_path):
    df = make_df(tmp_path, data={"v": [1.0, None, 3.0, 4.0, None]})
    res = df.rolling(3, "sum", min_periods=1)
    vals = read_col(res, tmp_path, "v_rolling_sum")
    assert approx(vals[0], 1.0)
    assert approx(vals[1], 1.0)
    assert approx(vals[2], 4.0)
    assert approx(vals[3], 7.0)
    assert approx(vals[4], 7.0)


def test_rolling_count_skips_nulls(tmp_path):
    df = make_df(tmp_path, data={"v": [1.0, None, 3.0, None, 5.0]})
    res = df.rolling(3, "count", min_periods=1)
    vals = read_col(res, tmp_path, "v_rolling_count")
    assert vals[0] == 1
    assert vals[1] == 1
    assert vals[2] == 2
    assert vals[3] == 1
    assert vals[4] == 2


# ---------- multi columns ----------

def test_rolling_two_columns_default(tmp_path):
    df = make_df(tmp_path, data={
        "a": [1.0, 2.0, 3.0],
        "b": [10.0, 20.0, 30.0],
    })
    res = df.rolling(2, "sum")
    assert approx(read_col(res, tmp_path, "a_rolling_sum")[1], 3.0)
    assert approx(read_col(res, tmp_path, "b_rolling_sum")[1], 30.0)


def test_rolling_subset(tmp_path):
    df = make_df(tmp_path, data={
        "a": [1.0, 2.0, 3.0],
        "b": [10.0, 20.0, 30.0],
    })
    res = df.rolling(2, "sum", columns=["a"])
    out = tmp_path / "o.parquet"
    res.write_parquet(out)
    names = pq.read_schema(out).names
    assert "a_rolling_sum" in names
    assert "b_rolling_sum" not in names


# ---------- batch boundaries ----------

def test_rolling_across_batches(tmp_path):
    """window > batch_size: state carried across batches."""
    df = make_df(tmp_path, chunk=1)  # 1 row per batch
    res = df.rolling(3, "mean", min_periods=1)
    vals = read_col(res, tmp_path, "v_rolling_mean")
    assert approx(vals[0], 1.0)
    assert approx(vals[1], 1.5)
    assert approx(vals[2], 2.0)
    assert approx(vals[3], 3.0)
    assert approx(vals[4], 4.0)


# ---------- filtered ----------

def test_rolling_filtered(tmp_path):
    df = make_df(tmp_path, data={"v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    filtered = df[df["v"] > 2]  # [3, 4, 5, 6]
    res = filtered.rolling(2, "sum")
    vals = read_col(res, tmp_path, "v_rolling_sum")
    assert vals[0] is None
    assert approx(vals[1], 7.0)
    assert approx(vals[2], 9.0)
    assert approx(vals[3], 11.0)


# ---------- errors ----------

def test_rolling_invalid_window(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(ValueError, match="window must be >= 1"):
        df.rolling(0, "sum")
    with pytest.raises(TypeError, match="window must be int"):
        df.rolling(1.5, "sum")


def test_rolling_invalid_agg(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(ValueError, match="agg must be one of"):
        df.rolling(3, "std")


def test_rolling_missing_column(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(KeyError, match="nope"):
        df.rolling(3, "sum", columns=["nope"])


def test_rolling_non_numeric_for_sum_raises(tmp_path):
    df = make_df(tmp_path, data={
        "v": [1.0, 2.0, 3.0], "s": ["a", "b", "c"],
    })
    with pytest.raises(TypeError, match="numeric"):
        df.rolling(2, "sum", columns=["s"])


def test_rolling_count_on_string_works(tmp_path):
    df = make_df(tmp_path, data={
        "v": [1.0, 2.0, 3.0], "s": ["a", "b", "c"],
    })
    res = df.rolling(2, "count", columns=["s"])
    vals = read_col(res, tmp_path, "s_rolling_count")
    assert vals[0] is None
    assert vals[1] == 2 and vals[2] == 2


# ---------- preserves original ----------

def test_rolling_preserves_original(tmp_path):
    df = make_df(tmp_path)
    _ = df.rolling(3, "sum")
    assert read_col(df, tmp_path, "v") == [1.0, 2.0, 3.0, 4.0, 5.0]


# ---------- dtype ----------

def test_rolling_sum_dtype_float(tmp_path):
    df = make_df(tmp_path, data={"v": [1, 2, 3, 4, 5]})  # int
    res = df.rolling(2, "sum")
    out = tmp_path / "o.parquet"
    res.write_parquet(out)
    schema = pq.read_schema(out)
    assert schema.field("v_rolling_sum").type == pa.float64()


def test_rolling_count_dtype_int(tmp_path):
    df = make_df(tmp_path)
    res = df.rolling(3, "count")
    out = tmp_path / "o.parquet"
    res.write_parquet(out)
    schema = pq.read_schema(out)
    assert schema.field("v_rolling_count").type == pa.int64()


def test_rolling_min_dtype_same_as_input(tmp_path):
    df = make_df(tmp_path, data={"v": [1, 2, 3, 4, 5]})  # int
    res = df.rolling(2, "min")
    out = tmp_path / "o.parquet"
    res.write_parquet(out)
    schema = pq.read_schema(out)
    assert schema.field("v_rolling_min").type == pa.int64()

# ---------- v0.8.4-fix (S1): min/max + nulls (was TypeError crash) ----------

def test_rolling_min_with_nulls(tmp_path):
    """S1 regression (min + null) + reference-derived (§13.4/128)."""
    from tests._helpers.rolling_reference import expected_rolling
    vals_in = [3.0, None, 1.0, None, 5.0]
    df = make_df(tmp_path, data={"v": vals_in})
    res = df.rolling(2, "min", min_periods=1)
    got = read_col(res, tmp_path, "v_rolling_min")
    assert got == expected_rolling(vals_in, 2, "min", min_periods=1)

def test_rolling_max_with_nulls(tmp_path):
    """S1 regression: max + null must not crash.

    Positional window=2, min_periods=1, skipna. Walk:
      pos0 window=[3.0]         → max=3.0
      pos1 window=[3.0, None]   → max=3.0
      pos2 window=[None, 1.0]   → max=1.0  (3.0 evicted)
      pos3 window=[1.0, None]   → max=1.0
      pos4 window=[None, 5.0]   → max=5.0
    pandas: pd.Series([3.0,None,1.0,None,5.0]).rolling(2,min_periods=1).max()
            → [3.0, 3.0, 1.0, 1.0, 5.0]
    (min gives same result here — after pos 0 only one non-null per window.)
    """
    df = make_df(tmp_path, data={"v": [3.0, None, 1.0, None, 5.0]})
    res = df.rolling(2, "max", min_periods=1)
    vals = read_col(res, tmp_path, "v_rolling_max")
    assert vals == [3.0, 3.0, 1.0, 1.0, 5.0]


def test_rolling_min_all_nulls(tmp_path):
    """All-null: no crash (`_dq` stays empty throughout)."""
    df = make_df(tmp_path, data={
        "v": pa.array([None, None, None], type=pa.float64()),
    })
    res = df.rolling(2, "min", min_periods=1)
    vals = read_col(res, tmp_path, "v_rolling_min")
    assert vals == [None, None, None]

