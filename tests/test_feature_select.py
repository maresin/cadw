"""Tests for D.6.1 mutual_info / variance / Boruta (v0.9.40)."""
from __future__ import annotations
import inspect
import numpy as np
import pyarrow as pa
import pytest
from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ml import feature_select as fsmod
from cadw.ml.feature_select import fit_feature_select, FeatureSelectorModel


def _data(n=1000, seed=0, d=4):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] + X[:, 1] + 0.3 * rng.normal(size=n) > 0).astype(np.float64)
    return X, y


def _df(X, y):
    d = {f"f{j}": pa.array(X[:, j]) for j in range(X.shape[1])}
    d["y"] = pa.array(y)
    return CadwDataFrame.from_table(pa.table(d))


def test_mutual_info_top_k_matches_sklearn():
    X, y = _data(seed=0)
    df = _df(X, y)
    m = fit_feature_select(df, [f"f{j}" for j in range(4)], "y",
                           method="mutual_info", task="classification")
    from sklearn.feature_selection import mutual_info_classif
    ref = mutual_info_classif(X, y, random_state=0)
    assert np.allclose(m.scores_, ref, atol=1e-9), (m.scores_, ref)
    # top-2 must be f0, f1
    top2 = set(np.array(m.feature_cols_)[np.argsort(-m.scores_)[:2]])
    assert top2 == {"f0", "f1"}


def test_mutual_info_regression_matches_sklearn():
    X, y = _data(seed=1)
    y = X[:, 0] * 2.0 + 0.1 * np.random.RandomState(1).normal(size=X.shape[0])
    df = _df(X, y)
    m = fit_feature_select(df, [f"f{j}" for j in range(4)], "y",
                           method="mutual_info", task="regression")
    from sklearn.feature_selection import mutual_info_regression
    ref = mutual_info_regression(X, y, random_state=0)
    assert np.allclose(m.scores_, ref, rtol=0.02, atol=0.03), (m.scores_, ref)


def test_variance_drops_constant_columns():
    rng = np.random.RandomState(2)
    X = rng.normal(size=(300, 4))
    X[:, 2] = 1.0
    y = (X[:, 0] > 0).astype(float)
    m = fit_feature_select(_df(X, y), [f"f{j}" for j in range(4)], "y",
                           method="variance", variance_threshold=0.0)
    assert "f2" not in m.selected_
    assert set(m.selected_) == {"f0", "f1", "f3"}


def test_boruta_keeps_informative_features():
    X, y = _data(n=800, seed=3)
    m = fit_feature_select(_df(X, y), [f"f{j}" for j in range(4)], "y",
                           method="boruta", max_iter=5, n_estimators=20)
    assert set(m.selected_) >= {"f0", "f1"}


def test_transform_returns_selected_columns():
    X, y = _data(n=300, seed=4)
    df = _df(X, y)
    m = fit_feature_select(df, [f"f{j}" for j in range(4)], "y",
                           method="variance", variance_threshold=0.0)
    out = m.transform(df)
    assert set(out.columns) <= set(m.selected_) | {"y"}
    for c in m.selected_:
        assert c in out.columns


def test_determinism():
    X, y = _data(n=300, seed=5)
    df = _df(X, y)
    a = fit_feature_select(df, [f"f{j}" for j in range(4)], "y",
                           method="mutual_info").scores_
    b = fit_feature_select(df, [f"f{j}" for j in range(4)], "y",
                           method="mutual_info").scores_
    assert np.array_equal(a, b)


def test_edge_guards():
    X, y = _data(n=200, seed=6)
    df = _df(X, y)
    with pytest.raises(ValueError, match="method must be"):
        fit_feature_select(df, ["f0"], "y", method="foo")
    with pytest.raises(ValueError, match="non-empty"):
        fit_feature_select(df, [], "y", method="variance")
    with pytest.raises(KeyError):
        fit_feature_select(df, ["nope"], "y", method="variance")
    with pytest.raises(ValueError, match="task must be"):
        fit_feature_select(df, ["f0"], "y", method="mutual_info", task="foo")


def test_ram_guard_raises_when_tight():
    X, y = _data(n=1000, seed=7)
    mg = MemoryGuard(max_memory_bytes=512)
    with pytest.raises(MemoryLimitExceeded):
        fit_feature_select(_df(X, y), [f"f{j}" for j in range(4)], "y",
                           method="variance", memory_guard=mg)


def test_multi_batch_consistency(tmp_path):
    import pyarrow.parquet as pq
    X, y = _data(n=600, seed=8)
    df = _df(X, y)
    tbl = pa.Table.from_batches(list(df._iter_filtered_batches()))
    path = str(tmp_path / "fs.parquet")
    pq.write_table(tbl, path, row_group_size=71)
    dmb = CadwDataFrame.from_parquet(path, base_dir="/tmp", chunk_size=100)
    m1 = fit_feature_select(df, [f"f{j}" for j in range(4)], "y", method="mutual_info")
    m2 = fit_feature_select(dmb, [f"f{j}" for j in range(4)], "y", method="mutual_info")
    assert np.allclose(m1.scores_, m2.scores_, atol=1e-9)


def test_source_contract_guard_and_log_fit():
    src = inspect.getsource(fsmod)
    assert "check_batch" in src and "extra_factor=3.0" in src
    assert "log_fit" in src
    assert "import pandas" not in src
