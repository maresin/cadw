"""v0.8.0: expanding window + EWM + PCA single-pass (CovarianceMeanOp)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard, PCA, fit_pca
from cadw.ops.reduce import CovarianceMeanOp, reduce


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


# ============ expanding ============

def test_expanding_sum(tmp_path):
    df = make_df(tmp_path)
    res = df.expanding("sum")
    assert read_col(res, tmp_path, "v_expanding_sum") == [1.0, 3.0, 6.0, 10.0, 15.0]


def test_expanding_mean(tmp_path):
    df = make_df(tmp_path)
    res = df.expanding("mean")
    vals = read_col(res, tmp_path, "v_expanding_mean")
    assert vals == [1.0, 1.5, 2.0, 2.5, 3.0]


def test_expanding_min(tmp_path):
    df = make_df(tmp_path, data={"v": [3.0, 1.0, 4.0, 2.0, 5.0]})
    res = df.expanding("min")
    vals = read_col(res, tmp_path, "v_expanding_min")
    assert vals == [3.0, 1.0, 1.0, 1.0, 1.0]


def test_expanding_max(tmp_path):
    df = make_df(tmp_path, data={"v": [3.0, 1.0, 4.0, 2.0, 5.0]})
    res = df.expanding("max")
    vals = read_col(res, tmp_path, "v_expanding_max")
    assert vals == [3.0, 3.0, 4.0, 4.0, 5.0]


def test_expanding_count(tmp_path):
    df = make_df(tmp_path)
    res = df.expanding("count")
    assert read_col(res, tmp_path, "v_expanding_count") == [1, 2, 3, 4, 5]


def test_expanding_min_periods(tmp_path):
    df = make_df(tmp_path)
    res = df.expanding("mean", min_periods=3)
    vals = read_col(res, tmp_path, "v_expanding_mean")
    assert vals[0] is None
    assert vals[1] is None
    assert approx(vals[2], 2.0)
    assert approx(vals[3], 2.5)
    assert approx(vals[4], 3.0)


def test_expanding_with_nulls(tmp_path):
    df = make_df(tmp_path, data={"v": [1.0, None, 3.0, None, 5.0]})
    res = df.expanding("sum")
    vals = read_col(res, tmp_path, "v_expanding_sum")
    assert approx(vals[0], 1.0)
    assert vals[1] is None  # null row -> null
    assert approx(vals[2], 4.0)  # 1 + 3
    assert vals[3] is None
    assert approx(vals[4], 9.0)


def test_expanding_across_batches(tmp_path):
    df = make_df(tmp_path, chunk=1)
    res = df.expanding("sum")
    vals = read_col(res, tmp_path, "v_expanding_sum")
    assert vals == [1.0, 3.0, 6.0, 10.0, 15.0]


def test_expanding_dtype_sum_float(tmp_path):
    df = make_df(tmp_path, data={"v": [1, 2, 3]})
    res = df.expanding("sum")
    out = tmp_path / "o.parquet"
    res.write_parquet(out)
    assert pq.read_schema(out).field("v_expanding_sum").type == pa.float64()


def test_expanding_dtype_min_same_as_input(tmp_path):
    df = make_df(tmp_path, data={"v": [1, 2, 3]})
    res = df.expanding("min")
    out = tmp_path / "o.parquet"
    res.write_parquet(out)
    assert pq.read_schema(out).field("v_expanding_min").type == pa.int64()


def test_expanding_invalid_agg(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(ValueError, match="agg must be one of"):
        df.expanding("std")


def test_expanding_invalid_min_periods(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(ValueError, match="min_periods"):
        df.expanding("sum", min_periods=0)


def test_expanding_missing_column(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(KeyError, match="nope"):
        df.expanding("sum", columns=["nope"])


def test_expanding_non_numeric_for_sum_raises(tmp_path):
    df = make_df(tmp_path, data={
        "v": [1.0, 2.0, 3.0], "s": ["a", "b", "c"],
    })
    with pytest.raises(TypeError, match="numeric"):
        df.expanding("sum", columns=["s"])


def test_expanding_pandas_equivalence(tmp_path):
    n = 50
    rng = np.random.default_rng(0)
    v = rng.normal(size=n).tolist()
    df = make_df(tmp_path, data={"v": v}, chunk=10)
    for agg in ("sum", "mean", "min", "max", "count"):
        res = df.expanding(agg)
        col = f"v_expanding_{agg}"
        got = read_col(res, tmp_path, col)
        exp = getattr(pd.Series(v).expanding(min_periods=1), agg)().tolist()
        np.testing.assert_allclose(got, exp, rtol=1e-10, equal_nan=True)


def test_expanding_subset_columns(tmp_path):
    df = make_df(tmp_path, data={
        "a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0],
    })
    res = df.expanding("sum", columns=["a"])
    out = tmp_path / "o.parquet"
    res.write_parquet(out)
    names = pq.read_schema(out).names
    assert "a_expanding_sum" in names
    assert "b_expanding_sum" not in names


# ============ ewm ============

def test_ewm_adjust_true_alpha(tmp_path):
    df = make_df(tmp_path)
    res = df.ewm(alpha=0.5, adjust=True)
    vals = read_col(res, tmp_path, "v_ewm")
    # Manually: [1, 2.5/1.5, 4.25/1.75, 6.125/1.875, 8.0625/1.9375]
    expected = [
        1.0,
        2.5 / 1.5,
        4.25 / 1.75,
        6.125 / 1.875,
        8.0625 / 1.9375,
    ]
    np.testing.assert_allclose(vals, expected, rtol=1e-12)


def test_ewm_adjust_false_alpha(tmp_path):
    df = make_df(tmp_path)
    res = df.ewm(alpha=0.5, adjust=False)
    vals = read_col(res, tmp_path, "v_ewm")
    expected = [1.0, 1.5, 2.25, 3.125, 4.0625]
    np.testing.assert_allclose(vals, expected, rtol=1e-12)


def test_ewm_span(tmp_path):
    """span=3 → alpha=0.5 (same as alpha=0.5)."""
    df = make_df(tmp_path)
    r1 = df.ewm(alpha=0.5).ewm  # not called, just checking API allows
    df_a = df.ewm(alpha=0.5)
    df_s = df.ewm(span=3)  # 2/(3+1) = 0.5
    a = read_col(df_a, tmp_path, "v_ewm")
    s = read_col(df_s, tmp_path, "v_ewm")
    np.testing.assert_allclose(a, s, rtol=1e-12)


def test_ewm_halflife(tmp_path):
    df = make_df(tmp_path)
    import math
    hl = 2.0
    df_h = df.ewm(halflife=hl, adjust=False)
    alpha_equiv = 1.0 - math.exp(-math.log(2.0) / hl)
    df_a = df.ewm(alpha=alpha_equiv, adjust=False)
    h = read_col(df_h, tmp_path, "v_ewm")
    a = read_col(df_a, tmp_path, "v_ewm")
    np.testing.assert_allclose(h, a, rtol=1e-12)


def test_ewm_com(tmp_path):
    df = make_df(tmp_path)
    # com=1 → alpha=0.5
    df_c = df.ewm(com=1, adjust=False)
    df_a = df.ewm(alpha=0.5, adjust=False)
    c = read_col(df_c, tmp_path, "v_ewm")
    a = read_col(df_a, tmp_path, "v_ewm")
    np.testing.assert_allclose(c, a, rtol=1e-12)


def test_ewm_min_periods(tmp_path):
    df = make_df(tmp_path)
    res = df.ewm(alpha=0.5, min_periods=3)
    vals = read_col(res, tmp_path, "v_ewm")
    assert vals[0] is None
    assert vals[1] is None
    assert vals[2] is not None
    assert vals[3] is not None
    assert vals[4] is not None


def test_ewm_with_nulls(tmp_path):
    df = make_df(tmp_path, data={"v": [1.0, None, 3.0, None, 5.0]})
    res = df.ewm(alpha=0.5, adjust=True)
    vals = read_col(res, tmp_path, "v_ewm")
    # Null rows -> null; state frozen
    assert approx(vals[0], 1.0)
    assert vals[1] is None
    # After row 2: s_num=(0.5)*1+3=3.5, s_den=(0.5)*1+1=1.5, y=2.333
    assert approx(vals[2], 3.5 / 1.5)
    assert vals[3] is None
    # Row 4: s_num=(0.5)*3.5+5=6.75, s_den=(0.5)*1.5+1=1.75, y=3.857
    assert approx(vals[4], 6.75 / 1.75)


def test_ewm_across_batches(tmp_path):
    df = make_df(tmp_path, chunk=1)
    res = df.ewm(alpha=0.5, adjust=False)
    vals = read_col(res, tmp_path, "v_ewm")
    expected = [1.0, 1.5, 2.25, 3.125, 4.0625]
    np.testing.assert_allclose(vals, expected, rtol=1e-12)


def test_ewm_pandas_equivalence_adjust_true(tmp_path):
    n = 50
    rng = np.random.default_rng(0)
    v = rng.normal(size=n).tolist()
    df = make_df(tmp_path, data={"v": v}, chunk=10)
    res = df.ewm(alpha=0.3, adjust=True)
    got = read_col(res, tmp_path, "v_ewm")
    exp = pd.Series(v).ewm(alpha=0.3, adjust=True).mean().tolist()
    np.testing.assert_allclose(got, exp, rtol=1e-10)


def test_ewm_pandas_equivalence_adjust_false(tmp_path):
    n = 50
    rng = np.random.default_rng(0)
    v = rng.normal(size=n).tolist()
    df = make_df(tmp_path, data={"v": v}, chunk=10)
    res = df.ewm(alpha=0.3, adjust=False)
    got = read_col(res, tmp_path, "v_ewm")
    exp = pd.Series(v).ewm(alpha=0.3, adjust=False).mean().tolist()
    np.testing.assert_allclose(got, exp, rtol=1e-10)


def test_ewm_dtype_float(tmp_path):
    df = make_df(tmp_path, data={"v": [1, 2, 3]})
    res = df.ewm(alpha=0.5)
    out = tmp_path / "o.parquet"
    res.write_parquet(out)
    assert pq.read_schema(out).field("v_ewm").type == pa.float64()


def test_ewm_requires_exactly_one_param(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(ValueError, match="exactly one"):
        df.ewm()  # none
    with pytest.raises(ValueError, match="exactly one"):
        df.ewm(alpha=0.5, span=3)  # two


def test_ewm_invalid_alpha(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(ValueError, match="alpha"):
        df.ewm(alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        df.ewm(alpha=1.5)


def test_ewm_invalid_span(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(ValueError, match="span"):
        df.ewm(span=0.5)


def test_ewm_invalid_halflife(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(ValueError, match="halflife"):
        df.ewm(halflife=-1)


def test_ewm_invalid_com(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(ValueError, match="com"):
        df.ewm(com=-1)


def test_ewm_non_numeric_raises(tmp_path):
    df = make_df(tmp_path, data={"s": ["a", "b", "c"]})
    with pytest.raises(TypeError, match="numeric"):
        df.ewm(alpha=0.5)


def test_ewm_missing_column(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(KeyError, match="nope"):
        df.ewm(alpha=0.5, columns=["nope"])


def test_ewm_subset_columns(tmp_path):
    df = make_df(tmp_path, data={
        "a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0],
    })
    res = df.ewm(alpha=0.5, columns=["a"])
    out = tmp_path / "o.parquet"
    res.write_parquet(out)
    names = pq.read_schema(out).names
    assert "a_ewm" in names
    assert "b_ewm" not in names


# ============ PCA single-pass: CovarianceMeanOp ============

def test_covariance_mean_op_values(tmp_path):
    """CovarianceMeanOp: {cov, mean, n} matches 2-pass reference."""
    n = 200
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 3))
    batches = []
    for i in range(0, n, 20):
        Xb = X[i:i+20]
        batches.append(pa.RecordBatch.from_arrays(
            [pa.array(Xb[:, j]) for j in range(3)],
            names=["x0", "x1", "x2"],
        ))
    op = CovarianceMeanOp(["x0", "x1", "x2"])
    mg = MemoryGuard(max_memory_mb=10)
    res = reduce(iter(batches), op, mg, log=False)
    # Reference
    mean_ref = X.mean(axis=0)
    cov_ref = np.cov(X, rowvar=False, ddof=1)
    np.testing.assert_allclose(res["mean"], mean_ref, rtol=1e-12)
    np.testing.assert_allclose(res["cov"], cov_ref, rtol=1e-12)
    assert res["n"] == n


def test_covariance_mean_op_estimate_state():
    """CovarianceMeanOp inherits estimate_state_nbytes from CovarianceOp."""
    op = CovarianceMeanOp(["a", "b", "c"])
    # Same as CovarianceOp: 8 + 8d + 8d²
    assert op.estimate_state_nbytes() == 8 + 8*3 + 8*3*3


def test_fit_pca_matches_v073(tmp_path):
    """fit_pca still produces same result after CovarianceMeanOp switch."""
    n, d = 200, 3
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, d))
    data = {f"x{i}": X[:, i].tolist() for i in range(d)}
    df = make_df(tmp_path, data=data, chunk=20)
    pca = fit_pca(df, [f"x{i}" for i in range(d)], n_components=2)
    # Reference numpy
    Xc = X - X.mean(axis=0)
    cov = (Xc.T @ Xc) / (n - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    comp_ref = eigvecs[:, :2].T
    dot = np.abs(pca.components_ @ comp_ref.T)
    np.testing.assert_allclose(dot, np.eye(2), atol=1e-6)
    np.testing.assert_allclose(
        pca.explained_variance_, eigvals[:2], rtol=1e-9,
    )
