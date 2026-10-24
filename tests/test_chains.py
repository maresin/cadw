"""Tests for D.5.2 classifier chains (v0.9.39)."""
from __future__ import annotations
import inspect
import numpy as np
import pyarrow as pa
import pytest
from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ml import chains as chmod
from cadw.ml.chains import fit_chains, ChainsModel


def _dependent(n=2500, seed=0):
    """Labels with a strong conditional dependency (b mostly copies a)."""
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, 3))
    a = (X[:, 0] + 0.3 * rng.normal(size=n) > 0).astype(float)
    b = np.where(rng.uniform(size=n) < 0.85, a,
                 1.0 - a).astype(float)
    c = (-0.4 * X[:, 0] + 0.5 * X[:, 2] + 0.3 * rng.normal(size=n) > 0).astype(float)
    return X, np.column_stack([a, b, c])


def _df(X, Y):
    return CadwDataFrame.from_table(pa.table({
        "x0": pa.array(X[:, 0]), "x1": pa.array(X[:, 1]),
        "x2": pa.array(X[:, 2]),
        "a": pa.array(Y[:, 0]), "b": pa.array(Y[:, 1]),
        "c": pa.array(Y[:, 2])}))


def test_synthetic_dependent_labels_chain_beats_independent():
    X, Y = _dependent(seed=0)
    df = _df(X, Y)
    chain = fit_chains(df, ["x0", "x1", "x2"], ["a", "b", "c"],
                       chain=True, n_estimators=20)
    indep = fit_chains(df, ["x0", "x1", "x2"], ["a", "b", "c"],
                       chain=False, n_estimators=20)
    sa_chain = chain.subset_accuracy(df)
    sa_indep = indep.subset_accuracy(df)
    assert sa_chain >= sa_indep - 1e-9, (sa_chain, sa_indep)
    assert sa_chain > 0.60, sa_chain


def test_reference_sklearn_classifier_chain_comparable():
    sk = pytest.importorskip("sklearn.multioutput")
    X, Y = _dependent(seed=1)
    df = _df(X, Y)
    ours = fit_chains(df, ["x0", "x1", "x2"], ["a", "b", "c"],
                      chain=True, n_estimators=20).subset_accuracy(df)
    from sklearn.ensemble import HistGradientBoostingClassifier
    cc = sk.ClassifierChain(
        HistGradientBoostingClassifier(max_iter=20), order=[0, 1, 2]).fit(X, Y)
    ref = float(np.mean(np.all(cc.predict(X) == Y, axis=1)))
    assert ours > ref * 0.85, (ours, ref)


def test_shapes_and_threshold():
    X, Y = _dependent(n=500, seed=2)
    df = _df(X, Y)
    m = fit_chains(df, ["x0", "x1"], ["a", "b"], chain=True, n_estimators=10)
    P = m.predict_proba(df)
    assert P.shape == (500, 2)
    assert np.all(P >= 0) and np.all(P <= 1)
    hard = m.predict(df, threshold=0.5)
    assert hard.shape == (500, 2)
    assert set(np.unique(hard)).issubset({0, 1})


def test_determinism():
    X, Y = _dependent(n=400, seed=3)
    df = _df(X, Y)
    a = fit_chains(df, ["x0", "x1"], ["a", "b"], chain=True, n_estimators=10).predict_proba(df)
    b = fit_chains(df, ["x0", "x1"], ["a", "b"], chain=True, n_estimators=10).predict_proba(df)
    assert np.array_equal(a, b)


def test_edge_guards():
    X, Y = _dependent(n=200, seed=4)
    df = _df(X, Y)
    with pytest.raises(ValueError, match="X_cols must be non-empty"):
        fit_chains(df, [], ["a"], n_estimators=5)
    with pytest.raises(ValueError, match="label_cols must be non-empty"):
        fit_chains(df, ["x0"], [], n_estimators=5)
    with pytest.raises(KeyError):
        fit_chains(df, ["nope"], ["a"], n_estimators=5)
    Yn = Y.copy(); Yn[0, 0] = 0.5
    with pytest.raises(ValueError, match="binary"):
        fit_chains(_df(X, Yn), ["x0"], ["a"], n_estimators=5)


def test_ram_guard_raises_when_tight():
    X, Y = _dependent(n=1000, seed=5)
    mg = MemoryGuard(max_memory_bytes=512)
    with pytest.raises(MemoryLimitExceeded):
        fit_chains(_df(X, Y), ["x0", "x1"], ["a", "b"], n_estimators=5,
                   memory_guard=mg)


def test_multi_batch_consistency(tmp_path):
    import pyarrow.parquet as pq
    X, Y = _dependent(n=800, seed=6)
    df = _df(X, Y)
    tbl = pa.Table.from_batches(list(df._iter_filtered_batches()))
    path = str(tmp_path / "ch.parquet")
    pq.write_table(tbl, path, row_group_size=71)
    dmb = CadwDataFrame.from_parquet(path, base_dir="/tmp", chunk_size=100)
    m = fit_chains(df, ["x0", "x1", "x2"], ["a", "b", "c"], chain=True,
                   n_estimators=10)
    assert np.allclose(m.predict_proba(df), m.predict_proba(dmb), atol=1e-9)


def test_source_contract_guard_and_log_fit():
    src = inspect.getsource(chmod)
    assert "check_batch" in src and "extra_factor=3.0" in src
    assert "log_fit" in src
    assert "import pandas" not in src
