"""Tests for D.5.3 MiniBatch KMeans + GMM (v0.9.39)."""
from __future__ import annotations
import inspect
import numpy as np
import pyarrow as pa
import pytest
from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ml import minibatch as mbmod
from cadw.ml.minibatch import (
    fit_minibatch_kmeans, fit_minibatch_gmm,
    MiniBatchKMeansModel, MiniBatchGMMModel,
)


def _blobs(n=4000, seed=0, k=3):
    rng = np.random.RandomState(seed)
    centers = rng.normal(scale=5.0, size=(k, 2))
    Y = rng.randint(0, k, size=n)
    X = centers[Y] + rng.normal(0, 0.5, size=(n, 2))
    return X, Y, centers


def _df(X):
    return CadwDataFrame.from_table(pa.table({
        "a": pa.array(X[:, 0]), "b": pa.array(X[:, 1])}))


def test_kmeans_synthetic_inertia_and_purity():
    X, Y, centers = _blobs(seed=0)
    m = fit_minibatch_kmeans(_df(X), ["a", "b"], n_clusters=3,
                             batch_size=256, max_iter=10)
    lab = m.predict(_df(X))
    # purity: each true blob maps to a single fitted cluster
    from collections import Counter
    by_blob = {}
    for b, l in zip(Y, lab):
        by_blob.setdefault(int(b), Counter())[int(l)] += 1
    purity = sum(c.most_common(1)[0][1] for c in by_blob.values()) / len(Y)
    assert purity > 0.95, purity


def test_kmeans_reference_sklearn_inertia():
    sk = pytest.importorskip("sklearn.cluster")
    X, _, _ = _blobs(seed=1)
    ours = fit_minibatch_kmeans(_df(X), ["a", "b"], n_clusters=3,
                                batch_size=256, max_iter=10).inertia(_df(X))
    ref = sk.MiniBatchKMeans(n_clusters=3, batch_size=256, random_state=0,
                             n_init=3).fit(X).inertia_
    assert ours < ref * 1.15, (ours, ref)


def test_kmeans_determinism():
    X, _, _ = _blobs(n=1500, seed=2)
    a = fit_minibatch_kmeans(_df(X), ["a", "b"], n_clusters=3, batch_size=128,
                             max_iter=5).predict(_df(X))
    b = fit_minibatch_kmeans(_df(X), ["a", "b"], n_clusters=3, batch_size=128,
                             max_iter=5).predict(_df(X))
    assert np.array_equal(a, b)


def test_gmm_synthetic_recovery():
    X, Y, centers = _blobs(seed=3)
    m = fit_minibatch_gmm(_df(X), ["a", "b"], n_components=3, batch_size=256,
                          max_iter=25)
    assert m.weights_.shape == (3,)
    assert abs(float(m.weights_.sum()) - 1.0) < 1e-9
    assert np.all(m.covars_ > 0)
    lab = m.predict(_df(X))
    assert lab.shape == (X.shape[0],)
    assert set(np.unique(lab)).issubset({0, 1, 2})


def test_gmm_reference_sklearn_likelihood():
    sk = pytest.importorskip("sklearn.mixture")
    X, _, _ = _blobs(seed=4)
    ours = fit_minibatch_gmm(_df(X), ["a", "b"], n_components=3,
                             batch_size=256, max_iter=25).score(_df(X))
    ref = sk.GaussianMixture(n_components=3, covariance_type="diag",
                             random_state=0).fit(X).score(X)
    assert ours > ref - 0.15, (ours, ref)


def test_gmm_proba_sums_to_one():
    X, _, _ = _blobs(n=1200, seed=5)
    m = fit_minibatch_gmm(_df(X), ["a", "b"], n_components=3, batch_size=256,
                          max_iter=10)
    R = m.predict_proba(_df(X))
    assert R.shape == (1200, 3)
    assert np.allclose(R.sum(axis=1), 1.0, atol=1e-9)


def test_gmm_determinism():
    X, _, _ = _blobs(n=1200, seed=6)
    a = fit_minibatch_gmm(_df(X), ["a", "b"], n_components=3, batch_size=256,
                          max_iter=10).score(_df(X))
    b = fit_minibatch_gmm(_df(X), ["a", "b"], n_components=3, batch_size=256,
                          max_iter=10).score(_df(X))
    assert a == b


def test_kmeans_edge_guards():
    X, _, _ = _blobs(n=200, seed=7)
    with pytest.raises(ValueError, match="n_clusters"):
        fit_minibatch_kmeans(_df(X), ["a"], n_clusters=0)
    with pytest.raises(ValueError, match="batch_size"):
        fit_minibatch_kmeans(_df(X), ["a"], batch_size=1)
    with pytest.raises(ValueError, match="non-empty"):
        fit_minibatch_kmeans(_df(X), [], n_clusters=2)
    with pytest.raises(KeyError):
        fit_minibatch_kmeans(_df(X), ["nope"], n_clusters=2)


def test_gmm_edge_guards():
    X, _, _ = _blobs(n=200, seed=8)
    with pytest.raises(ValueError, match="n_components"):
        fit_minibatch_gmm(_df(X), ["a"], n_components=0)
    with pytest.raises(ValueError, match="non-empty"):
        fit_minibatch_gmm(_df(X), [], n_components=2)
    with pytest.raises(KeyError):
        fit_minibatch_gmm(_df(X), ["nope"], n_components=2)


def test_ram_guard_raises_when_tight():
    X, _, _ = _blobs(n=1000, seed=9)
    mg = MemoryGuard(max_memory_bytes=512)
    with pytest.raises(MemoryLimitExceeded):
        fit_minibatch_kmeans(_df(X), ["a", "b"], n_clusters=2,
                             batch_size=64, memory_guard=mg)


def test_multi_batch_consistency(tmp_path):
    import pyarrow.parquet as pq
    X, _, _ = _blobs(n=1200, seed=10)
    df = _df(X)
    tbl = pa.Table.from_batches(list(df._iter_filtered_batches()))
    path = str(tmp_path / "mb.parquet")
    pq.write_table(tbl, path, row_group_size=97)
    dmb = CadwDataFrame.from_parquet(path, base_dir="/tmp", chunk_size=100)
    m = fit_minibatch_kmeans(df, ["a", "b"], n_clusters=3, batch_size=128,
                             max_iter=5)
    assert np.allclose(m.predict(df), m.predict(dmb), atol=0)


def test_source_contract_guard_and_log_fit():
    src = inspect.getsource(mbmod)
    assert "check_batch" in src and "extra_factor=3.0" in src
    assert "log_fit" in src
    assert "import pandas" not in src
