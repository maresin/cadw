"""v0.9.23e: cumsum/cumprod/cummax/cummin per group (7.A.6).

Reference: pandas .cumsum() / .cumprod() / .cummax() / .cummin() on
Series and GroupBy. Nulls preserved (state frozen, matches CADW
expanding semantics — NOT pandas NaN-propagate).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard


def make_df(tmp_path, data, chunk=3, name="L", mem_mb=50):
    t = pa.table(data)
    p = tmp_path / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, tmp_path / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
        spill_dir=tmp_path / f"spill_{name}",
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


# ============ GLOBAL cum* (df.cumsum) ============

def test_cumsum_global_basic(tmp_path):
    df = make_df(tmp_path, {"v": [1, 2, 3, 4, 5]})
    res = df.cumsum()
    assert read_col(res, tmp_path, "v_cumsum") == [1, 3, 6, 10, 15]


def test_cumprod_global_basic(tmp_path):
    df = make_df(tmp_path, {"v": [1, 2, 3, 4, 5]})
    res = df.cumprod()
    assert read_col(res, tmp_path, "v_cumprod") == [1, 2, 6, 24, 120]


def test_cummax_global_basic(tmp_path):
    df = make_df(tmp_path, {"v": [3, 1, 4, 1, 5]})
    res = df.cummax()
    assert read_col(res, tmp_path, "v_cummax") == [3, 3, 4, 4, 5]


def test_cummin_global_basic(tmp_path):
    df = make_df(tmp_path, {"v": [3, 1, 4, 1, 5]})
    res = df.cummin()
    assert read_col(res, tmp_path, "v_cummin") == [3, 1, 1, 1, 1]


def test_cumsum_global_pandas_equiv(tmp_path):
    rng = np.random.default_rng(0)
    vals = rng.normal(size=100).tolist()
    df = make_df(tmp_path, {"v": vals}, chunk=10)
    res = df.cumsum()
    got = read_col(res, tmp_path, "v_cumsum")
    expected = pd.Series(vals).cumsum().tolist()
    np.testing.assert_allclose(got, expected, rtol=1e-10)


def test_cumprod_global_pandas_equiv(tmp_path):
    rng = np.random.default_rng(0)
    vals = (rng.random(size=50) + 0.5).tolist()  # avoid 0
    df = make_df(tmp_path, {"v": vals}, chunk=10)
    res = df.cumprod()
    got = read_col(res, tmp_path, "v_cumprod")
    expected = pd.Series(vals).cumprod().tolist()
    np.testing.assert_allclose(got, expected, rtol=1e-10)


def test_cumsum_global_preserves_original(tmp_path):
    df = make_df(tmp_path, {"v": [1, 2, 3], "s": ["a", "b", "c"]})
    res = df.cumsum()
    assert "v" in res.columns
    assert "s" in res.columns
    assert "v_cumsum" in res.columns
    # Original unchanged
    assert read_col(res, tmp_path, "v") == [1, 2, 3]


def test_cumsum_with_nulls_freeze_state(tmp_path):
    df = make_df(tmp_path, {
        "v": pa.array([1.0, None, 3.0, None, 5.0], type=pa.float64()),
    })
    res = df.cumsum()
    got = read_col(res, tmp_path, "v_cumsum")
    # Null row -> null output; state frozen
    assert approx(got[0], 1.0)
    assert got[1] is None
    assert approx(got[2], 4.0)  # 1 + 3
    assert got[3] is None
    assert approx(got[4], 9.0)  # 4 + 5


def test_cumsum_on_col(tmp_path):
    df = make_df(tmp_path, {
        "ts": [3, 1, 2, 5, 4],
        "v": [30, 10, 20, 50, 40],
    })
    res = df.cumsum("v", on="ts")
    got = read_col(res, tmp_path, "v_cumsum")
    # Sorted by ts: v=[10, 20, 30, 40, 50] -> cumsum=[10, 30, 60, 100, 150]
    # orig: [3->60, 1->10, 2->30, 5->150, 4->100]
    assert got == [60, 10, 30, 150, 100]


def test_cummax_on_col(tmp_path):
    df = make_df(tmp_path, {
        "ts": [1, 2, 3, 4, 5],
        "v": [3, 1, 4, 1, 5],
    })
    res = df.cummax("v", on="ts")
    got = read_col(res, tmp_path, "v_cummax")
    assert got == [3, 3, 4, 4, 5]


def test_cumsum_subset_columns(tmp_path):
    df = make_df(tmp_path, {
        "a": [1, 2, 3], "b": [10, 20, 30], "s": ["x", "y", "z"],
    })
    res = df.cumsum(columns=["a"])
    out = tmp_path / "o.parquet"
    res.write_parquet(out)
    names = pq.read_schema(out).names
    assert "a_cumsum" in names
    assert "b_cumsum" not in names


# ============ PER-GROUP cum* ============

def test_cumsum_per_group_basic(tmp_path):
    df = make_df(tmp_path, {
        "g": ["a", "a", "a", "b", "b", "b"],
        "v": [1, 2, 3, 10, 20, 30],
    })
    res = df.groupby("g").cumsum()
    got = read_col(res, tmp_path, "v_cumsum")
    # group a: 1, 3, 6; group b: 10, 30, 60
    assert got == [1, 3, 6, 10, 30, 60]


def test_cumprod_per_group_basic(tmp_path):
    df = make_df(tmp_path, {
        "g": ["a", "a", "a", "b", "b"],
        "v": [1, 2, 3, 10, 20],
    })
    res = df.groupby("g").cumprod()
    got = read_col(res, tmp_path, "v_cumprod")
    assert got == [1, 2, 6, 10, 200]


def test_cummax_per_group_basic(tmp_path):
    df = make_df(tmp_path, {
        "g": ["a", "a", "a", "b", "b", "b"],
        "v": [3, 1, 4, 1, 5, 2],
    })
    res = df.groupby("g").cummax()
    got = read_col(res, tmp_path, "v_cummax")
    assert got == [3, 3, 4, 1, 5, 5]


def test_cummin_per_group_basic(tmp_path):
    df = make_df(tmp_path, {
        "g": ["a", "a", "a", "b", "b"],
        "v": [3, 1, 4, 5, 2],
    })
    res = df.groupby("g").cummin()
    got = read_col(res, tmp_path, "v_cummin")
    assert got == [3, 1, 1, 5, 2]


def test_cumsum_per_group_on_col(tmp_path):
    """Sort-aware per-group: sort by [g, ts], reset per group."""
    df = make_df(tmp_path, {
        "g": ["a", "a", "a", "b", "b"],
        "ts": [3, 1, 2, 2, 1],
        "v": [30, 10, 20, 20, 10],
    })
    res = df.groupby("g").cumsum(on="ts")
    got = read_col(res, tmp_path, "v_cumsum")
    # group a: sorted ts=1(v=10), 2(v=20), 3(v=30) -> cumsum=10,30,60
    #   orig order: ts=3->60, ts=1->10, ts=2->30
    # group b: sorted ts=1(v=10), 2(v=20) -> cumsum=10,30
    #   orig order: ts=2->30, ts=1->10
    assert got == [60, 10, 30, 30, 10]


def test_cumsum_per_group_pandas_equiv(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    groups = [f"g{i % 5}" for i in range(n)]
    vals = rng.normal(size=n).tolist()
    df = make_df(tmp_path, {"g": groups, "v": vals}, chunk=20)
    res = df.groupby("g").cumsum()
    got = read_col(res, tmp_path, "v_cumsum")

    pdf = pd.DataFrame({"g": groups, "v": vals})
    expected = pdf.groupby("g")["v"].cumsum().tolist()
    np.testing.assert_allclose(got, expected, rtol=1e-10)


def test_cumsum_per_group_with_nulls(tmp_path):
    df = make_df(tmp_path, {
        "g": ["a", "a", "a", "a"],
        "v": pa.array([1.0, None, 3.0, 5.0], type=pa.float64()),
    })
    res = df.groupby("g").cumsum()
    got = read_col(res, tmp_path, "v_cumsum")
    # Null row -> null output; state frozen within group
    assert approx(got[0], 1.0)
    assert got[1] is None
    assert approx(got[2], 4.0)
    assert approx(got[3], 9.0)


def test_cumsum_per_group_null_group(tmp_path):
    df = make_df(tmp_path, {
        "g": ["a", "a", None, None],
        "v": [1, 2, 10, 20],
    })
    res = df.groupby("g").cumsum()
    got = read_col(res, tmp_path, "v_cumsum")
    # Null group -> own group
    assert got == [1, 3, 10, 30]


def test_cumsum_per_group_append_not_replace(tmp_path):
    df = make_df(tmp_path, {
        "g": ["a", "a", "b"], "v": [1, 2, 3],
    })
    res = df.groupby("g").cumsum()
    out = tmp_path / "o.parquet"
    res.write_parquet(out)
    names = pq.read_schema(out).names
    assert "v" in names and "v_cumsum" in names


def test_cumsum_per_group_multibatch(tmp_path):
    """chunk=1: many batches, per-group state persists across batches."""
    df = make_df(tmp_path, {
        "g": ["a", "b", "a", "b", "a", "b"],
        "v": [1, 10, 2, 20, 3, 30],
    }, chunk=1)
    res = df.groupby("g").cumsum()
    got = read_col(res, tmp_path, "v_cumsum")
    # Interleaved groups: a: 1, 3, 6; b: 10, 30, 60
    # Output row_id order: [1,10,2,20,3,30] -> [1,10,3,30,6,60]
    assert got == [1, 10, 3, 30, 6, 60]


def test_cumsum_per_group_filtered(tmp_path):
    df = make_df(tmp_path, {
        "g": ["a", "a", "a", "b", "b"],
        "v": [1, 2, 3, 10, 20],
        "f": list(range(5)),
    })
    filtered = df[df["f"] > 1]
    res = filtered.groupby("g").cumsum()
    got = read_col(res, tmp_path, "v_cumsum")
    # After filter: a:v=[3], b:v=[10, 20] -> cumsum=[3], [10, 30]
    assert got == [3, 10, 30]


# ============ Validation ============

def test_cumsum_non_numeric_raises(tmp_path):
    df = make_df(tmp_path, {"s": ["a", "b", "c"], "v": [1, 2, 3]})
    with pytest.raises(TypeError, match="numeric"):
        df.cumsum(columns=["s"])


def test_cumsum_missing_column_raises(tmp_path):
    df = make_df(tmp_path, {"v": [1, 2, 3]})
    with pytest.raises(KeyError, match="nope"):
        df.cumsum(columns=["nope"])


def test_cumsum_empty_columns_raises(tmp_path):
    df = make_df(tmp_path, {"s": ["a", "b"]})
    with pytest.raises(ValueError, match="no numeric columns"):
        df.cumsum()


def test_cumsum_on_missing_raises(tmp_path):
    df = make_df(tmp_path, {"v": [1, 2], "ts": [1, 2]})
    with pytest.raises(KeyError, match="nope"):
        df.cumsum("v", on="nope")


def test_per_group_cumsum_group_col_in_columns_raises(tmp_path):
    df = make_df(tmp_path, {"g": ["a", "a"], "v": [1, 2]})
    with pytest.raises(ValueError, match="cannot be accumulated"):
        df.groupby("g").cumsum(columns=["g"])


# ============ AST guards ============

def test_cumulative_impl_no_to_pandas_ast():
    import inspect
    from cadw.patch_utils import assert_no_call_in_source
    from cadw.dataframe import CadwDataFrame
    src = inspect.getsource(CadwDataFrame._cumulative_impl)
    assert_no_call_in_source(
        src, "to_pandas", "from_pandas",
        fn_name="_cumulative_impl",
    )


def test_per_group_cumulative_no_to_pandas_ast():
    import inspect
    from cadw.patch_utils import assert_no_call_in_source
    from cadw.dataframe import CadwDataFrame
    src = inspect.getsource(CadwDataFrame._per_group_cumulative)
    assert_no_call_in_source(
        src, "to_pandas", "from_pandas",
        fn_name="_per_group_cumulative",
    )


def test_per_group_cumulative_uses_boundaries():
    """Reuses _group_boundaries_arrow (Arrow-native, v0.9.21)."""
    import inspect
    from cadw.dataframe import CadwDataFrame
    src = inspect.getsource(CadwDataFrame._per_group_cumulative)
    assert "_group_boundaries_arrow" in src


def test_expanding_state_has_prod():
    """_ExpandingState supports 'prod' after v0.9.23e."""
    from cadw.dataframe import _ExpandingState
    st = _ExpandingState("prod", 1)
    assert st.push(2) == 2
    assert st.push(3) == 6
    assert st.push(4) == 24


# ============ v0.9.23e-fix1: _ExpandingState.reset() ============

def test_expanding_state_reset():
    """_ExpandingState.reset() zeroes state; _agg/_min_periods preserved.

    v0.9.23e-fix1: _per_group_cumulative calls reset() on group change;
    prior AttributeError because _ExpandingState (for global expanding)
    had no reset. Mirrors _RollState.reset (v0.8.10).
    """
    from cadw.dataframe import _ExpandingState

    st = _ExpandingState("sum", min_periods=1)
    assert st.push(5) == 5
    assert st.push(10) == 15
    st.reset()
    # Config preserved
    assert st._agg == "sum"
    assert st._min_periods == 1
    # State reset: first push after reset starts from 0
    assert st.push(3) == 3
    assert st.push(4) == 7


def test_expanding_state_reset_prod():
    """reset() resets _prod to 1.0 (identity for multiplication)."""
    from cadw.dataframe import _ExpandingState

    st = _ExpandingState("prod", min_periods=1)
    assert st.push(2) == 2
    assert st.push(3) == 6
    st.reset()
    assert st.push(5) == 5   # starts from 1.0 * 5 = 5


def test_expanding_state_reset_min_max():
    """reset() clears _min/_max to None."""
    from cadw.dataframe import _ExpandingState

    st = _ExpandingState("max", min_periods=1)
    assert st.push(10) == 10
    st.reset()
    assert st.push(3) == 3   # new max from reset

    st2 = _ExpandingState("min", min_periods=1)
    assert st2.push(1) == 1
    st2.reset()
    assert st2.push(100) == 100


def test_per_group_cumsum_uses_reset():
    """AST: _per_group_cumulative calls st.reset() on group change."""
    import inspect
    from cadw.dataframe import CadwDataFrame
    src = inspect.getsource(CadwDataFrame._per_group_cumulative)
    assert "st.reset()" in src, (
        "_per_group_cumulative must call _ExpandingState.reset() "
        "on group change"
    )
