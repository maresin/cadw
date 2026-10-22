"""v0.9.23d: CadwDataFrame.corr(method='pearson') (AUDIT_FULL_v0922 7.A.4).

Pearson via CovarianceMeanOp (Reduce, O(d²) state). Reference: numpy
corrcoef (rowvar=False, ddof=1 like CADW).
"""
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.core.memory_guard import MemoryLimitExceeded


def make_df(tmp_path, data, chunk=10, name="L", mem_mb=50):
    t = pa.table(data)
    p = tmp_path / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, tmp_path / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
        spill_dir=tmp_path / f"spill_{name}",
    )


# ============ Basic correctness ============


def test_corr_perfect_positive(tmp_path):
    n = 100
    x = np.arange(n, dtype=np.float64)
    y = 2.0 * x + 3.0
    df = make_df(tmp_path, {"x": x.tolist(), "y": y.tolist()})
    c = df.corr()
    assert c["x"]["y"] == pytest.approx(1.0)
    assert c["y"]["x"] == pytest.approx(1.0)
    assert c["x"]["x"] == pytest.approx(1.0)


def test_corr_perfect_negative(tmp_path):
    n = 100
    x = np.arange(n, dtype=np.float64)
    y = -2.0 * x + 3.0
    df = make_df(tmp_path, {"x": x.tolist(), "y": y.tolist()})
    c = df.corr()
    assert c["x"]["y"] == pytest.approx(-1.0)
    assert c["y"]["x"] == pytest.approx(-1.0)


def test_corr_independent_is_zero(tmp_path):
    """Large independent sample -> |r| small."""
    rng = np.random.default_rng(42)
    n = 10_000
    df = make_df(
        tmp_path,
        {"x": rng.normal(size=n).tolist(), "y": rng.normal(size=n).tolist()},
        chunk=1000,
    )
    c = df.corr()
    assert abs(c["x"]["y"]) < 0.05


def test_corr_matches_numpy(tmp_path):
    """Exact parity with np.corrcoef over 3 columns."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(200, 3))
    df = make_df(
        tmp_path,
        {f"x{i}": X[:, i].tolist() for i in range(3)},
        chunk=20,
    )
    c = df.corr()
    ref = np.corrcoef(X, rowvar=False)
    for i in range(3):
        for j in range(3):
            assert c[f"x{i}"][f"x{j}"] == pytest.approx(ref[i, j], rel=1e-9)


def test_corr_symmetric(tmp_path):
    rng = np.random.default_rng(11)
    X = rng.normal(size=(100, 3))
    df = make_df(
        tmp_path,
        {f"x{i}": X[:, i].tolist() for i in range(3)},
    )
    c = df.corr()
    for i in range(3):
        for j in range(3):
            assert c[f"x{i}"][f"x{j}"] == pytest.approx(c[f"x{j}"][f"x{i}"])


# ============ Constant / null edge cases ============


def test_corr_constant_column_nan(tmp_path):
    df = make_df(tmp_path, {
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": [5.0, 5.0, 5.0, 5.0],   # constant
    })
    c = df.corr()
    assert math.isnan(c["a"]["b"])
    assert math.isnan(c["b"]["a"])
    assert math.isnan(c["b"]["b"])
    assert c["a"]["a"] == pytest.approx(1.0)


def test_corr_all_null_column_nan(tmp_path):
    df = make_df(tmp_path, {
        "a": [1.0, 2.0, 3.0],
        "b": pa.array([None, None, None], type=pa.float64()),
    })
    c = df.corr()
    assert math.isnan(c["a"]["b"])
    assert math.isnan(c["b"]["b"])


def test_corr_no_rows_nan(tmp_path):
    df = make_df(tmp_path, {
        "a": pa.array([], type=pa.float64()),
        "b": pa.array([], type=pa.float64()),
    })
    c = df.corr()
    assert math.isnan(c["a"]["a"])
    assert math.isnan(c["a"]["b"])


def test_corr_single_row_nan(tmp_path):
    """n=1 -> not enough for variance (ddof=1)."""
    df = make_df(tmp_path, {
        "a": [1.0], "b": [2.0],
    })
    c = df.corr()
    assert math.isnan(c["a"]["a"])


def test_corr_nulls_skipped(tmp_path):
    """Nulls skipped per row (pairwise, matches CovarianceOp's mask)."""
    df = make_df(tmp_path, {
        "a": pa.array([1.0, 2.0, None, 4.0, 5.0], type=pa.float64()),
        "b": pa.array([2.0, 4.0, 6.0, None, 10.0], type=pa.float64()),
    })
    c = df.corr()
    # Valid rows: (1,2), (2,4), (5,10) — perfectly linear -> r=1.0
    assert c["a"]["b"] == pytest.approx(1.0)


# ============ column selection ============


def test_corr_auto_detect_numeric(tmp_path):
    df = make_df(tmp_path, {
        "a": [1.0, 2.0, 3.0],
        "b": [2.0, 4.0, 6.0],
        "s": ["x", "y", "z"],   # string -> skipped
    })
    c = df.corr()
    assert set(c.keys()) == {"a", "b"}
    assert "s" not in c


def test_corr_columns_subset(tmp_path):
    df = make_df(tmp_path, {
        "a": [1.0, 2.0, 3.0],
        "b": [2.0, 4.0, 6.0],
        "s": ["x", "y", "z"],
    })
    c = df.corr(columns=["a"])
    assert list(c.keys()) == ["a"]
    assert "b" not in c


def test_corr_columns_string(tmp_path):
    df = make_df(tmp_path, {"a": [1.0, 2.0, 3.0]})
    c = df.corr(columns="a")
    assert c["a"]["a"] == pytest.approx(1.0)


def test_corr_empty_columns_raises(tmp_path):
    df = make_df(tmp_path, {"s": ["x", "y", "z"]})
    with pytest.raises(ValueError, match="no numeric columns"):
        df.corr()


def test_corr_missing_column_raises(tmp_path):
    df = make_df(tmp_path, {"a": [1.0, 2.0]})
    with pytest.raises(KeyError, match="nope"):
        df.corr(columns=["nope"])


def test_corr_non_numeric_raises(tmp_path):
    df = make_df(tmp_path, {"a": [1.0, 2.0], "s": ["x", "y"]})
    with pytest.raises(TypeError, match="numeric"):
        df.corr(columns=["s"])


# ============ method ============


def test_corr_spearman_not_implemented(tmp_path):
    df = make_df(tmp_path, {"a": [1.0, 2.0]})
    with pytest.raises(NotImplementedError, match="spearman"):
        df.corr(method="spearman")


# ============ multi-batch / filtered ============


def test_corr_multibatch_consistent(tmp_path):
    rng = np.random.default_rng(5)
    X = rng.normal(size=(200, 2))
    data = {"x0": X[:, 0].tolist(), "x1": X[:, 1].tolist()}
    df1 = make_df(tmp_path, data, chunk=20, name="a")
    df2 = make_df(tmp_path, data, chunk=200, name="b")
    c1 = df1.corr()
    c2 = df2.corr()
    assert c1["x0"]["x1"] == pytest.approx(c2["x0"]["x1"], rel=1e-12)


def test_corr_filtered(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 2))
    df = make_df(tmp_path, {
        "x0": X[:, 0].tolist(),
        "x1": X[:, 1].tolist(),
        "f": list(range(100)),
    }, chunk=20)
    filtered = df[df["f"] > 50]
    c = filtered.corr()
    ref = np.corrcoef(X[51:], rowvar=False)
    assert c["x0"]["x1"] == pytest.approx(ref[0, 1], rel=1e-9)


# ============ memory guard ============


def test_corr_large_d_fails_fast(tmp_path):
    """d=500 -> state ~2 MB; budget 100 KB -> fail-fast BEFORE init."""
    rng = np.random.default_rng(0)
    data = {f"c{i}": rng.normal(size=100).tolist() for i in range(500)}
    df = make_df(tmp_path, data, chunk=20)
    df._memory_guard = MemoryGuard(max_memory_bytes=100_000)
    with pytest.raises(MemoryLimitExceeded):
        df.corr()


# ============ AST guards ============


def test_corr_no_pandas_ast():
    """AST: no to_pandas/from_pandas Call in corr body."""
    import inspect
    from cadw.patch_utils import assert_no_call_in_source
    from cadw.dataframe import CadwDataFrame
    src = inspect.getsource(CadwDataFrame.corr)
    assert_no_call_in_source(
        src, "to_pandas", "from_pandas",
        fn_name="CadwDataFrame.corr",
    )


def test_corr_uses_covariance_mean_op():
    """AST: corr uses CovarianceMeanOp (not a custom Reduce loop)."""
    import inspect
    from cadw.dataframe import CadwDataFrame
    src = inspect.getsource(CadwDataFrame.corr)
    assert "CovarianceMeanOp" in src
    assert "reduce(" in src


def test_corr_returns_nested_dict_shape(tmp_path):
    """Return type: dict[str, dict[str, float]] (describe-consistent)."""
    df = make_df(tmp_path, {
        "a": [1.0, 2.0, 3.0],
        "b": [2.0, 4.0, 6.0],
    })
    c = df.corr()
    assert isinstance(c, dict)
    assert set(c.keys()) == {"a", "b"}
    for row in c.values():
        assert isinstance(row, dict)
        for v in row.values():
            assert isinstance(v, float)
