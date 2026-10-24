"""Tests for D.6.4 randomized SVD (v0.9.40)."""
from __future__ import annotations
import inspect, numpy as np, pyarrow as pa, pytest
from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ml import randsvd as rmod
from cadw.ml.randsvd import fit_randomized_svd, RandomizedSVDModel


def _low_rank(n=800, k=3, d=10, seed=0):
    rng = np.random.RandomState(seed)
    Z = rng.normal(size=(n, k))
    X = Z @ rng.normal(size=(k, d)) + 0.05 * rng.normal(size=(n, d))
    return X


def _df(X):
    cols = [f"x{j}" for j in range(X.shape[1])]
    d = {c: pa.array(X[:, j]) for j, c in enumerate(cols)}
    return CadwDataFrame.from_table(pa.table(d)), cols


def test_synthetic_singular_values_match_pca():
    X = _low_rank(seed=0)
    df, cols = _df(X)
    m = fit_randomized_svd(df, cols, n_components=3)
    from sklearn.decomposition import PCA
    p = PCA(n_components=3).fit(X)
    pca_sv = np.sqrt(p.explained_variance_ * (X.shape[0] - 1))
    assert np.allclose(m.singular_values_, pca_sv, rtol=1e-6)
    assert np.allclose(m.explained_variance_ratio(df), p.explained_variance_ratio_, rtol=1e-6)


def test_transform_shape_and_projection_quality():
    X = _low_rank(seed=1)
    df, cols = _df(X)
    m = fit_randomized_svd(df, cols, n_components=3)
    Z = m.transform(df)
    assert Z.shape == (X.shape[0], 3)
    # reconstruct: Z @ components + mean should be close to X on the
    # rank-3 signal (noise 0.05)
    Xrec = Z @ m.components_ + m.mean_
    assert np.mean((Xrec - X) ** 2) < 0.01


def test_determinism():
    X = _low_rank(n=300, seed=2)
    df, cols = _df(X)
    a = fit_randomized_svd(df, cols, n_components=3).singular_values_
    b = fit_randomized_svd(df, cols, n_components=3).singular_values_
    assert np.array_equal(a, b)


def test_edge_guards():
    X = _low_rank(n=200, seed=3)
    df, cols = _df(X)
    with pytest.raises(ValueError, match="n_components"):
        fit_randomized_svd(df, cols, n_components=0)
    with pytest.raises(ValueError, match="non-empty"):
        fit_randomized_svd(df, [], n_components=2)
    with pytest.raises(KeyError):
        fit_randomized_svd(df, ["nope"], n_components=2)


def test_guard_nulls():
    X = _low_rank(n=100, seed=4); X[2, 0] = np.nan
    df, cols = _df(X)
    with pytest.raises(ValueError, match="nulls"):
        fit_randomized_svd(df, cols, n_components=2)


def test_ram_guard_raises_when_tight():
    X = _low_rank(n=500, seed=5)
    df, cols = _df(X)
    mg = MemoryGuard(max_memory_bytes=512)
    with pytest.raises(MemoryLimitExceeded):
        fit_randomized_svd(df, cols, n_components=2, memory_guard=mg)


def test_multi_batch_consistency(tmp_path):
    import pyarrow.parquet as pq
    X = _low_rank(n=400, seed=6)
    df, cols = _df(X)
    tbl = pa.Table.from_batches(list(df._iter_filtered_batches()))
    path = str(tmp_path / "r.parquet")
    pq.write_table(tbl, path, row_group_size=71)
    dmb = CadwDataFrame.from_parquet(path, base_dir="/tmp", chunk_size=100)
    m1 = fit_randomized_svd(df, cols, n_components=3)
    m2 = fit_randomized_svd(dmb, cols, n_components=3)
    assert np.allclose(m1.singular_values_, m2.singular_values_, rtol=1e-6)


def test_source_contract_guard_and_log_fit():
    src = inspect.getsource(rmod)
    assert "check_batch" in src and "extra_factor=3.0" in src
    assert "log_fit" in src
    assert "import pandas" not in src
