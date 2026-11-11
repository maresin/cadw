"""v0.9.215-e: fit_agglomerative verification (scipy dense).

Sklearn parity (4 linkages) + refuse_at_guard + validation + modes.
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ml import fit_agglomerative


def _mk_df(X):
    return CadwDataFrame.from_table(pa.table({
        f"x{j}": pa.array(X[:, j], type=pa.float64())
        for j in range(X.shape[1])
    }), memory_guard=MemoryGuard(max_memory_mb=500))


def _two_clusters(seed=0):
    rng = np.random.RandomState(seed)
    X1 = rng.normal(0, 0.3, size=(30, 2))
    X2 = rng.normal(5, 0.3, size=(30, 2))
    return np.vstack([X1, X2])


def _warmup():
    X = _two_clusters(seed=999)
    fit_agglomerative(_mk_df(X), ["x0", "x1"], n_clusters=2, linkage="ward",
                       memory_guard=MemoryGuard(max_memory_mb=500))


_warmup()


# ---- CW-1..4: sklearn ARI parity (4 linkages, parametrized) ----
@pytest.mark.parametrize("linkage", ["ward", "complete", "average", "single"])
def test_sklearn_parity(linkage):
    from sklearn.cluster import AgglomerativeClustering as SKAGG
    from sklearn.metrics import adjusted_rand_score
    X = _two_clusters(seed=0)
    sk = SKAGG(n_clusters=2, linkage=linkage).fit(X)
    r = fit_agglomerative(_mk_df(X), ["x0", "x1"], n_clusters=2,
                           linkage=linkage,
                           memory_guard=MemoryGuard(max_memory_mb=500))
    ari = adjusted_rand_score(sk.labels_, r.labels_)
    assert ari > 0.99, f"linkage={linkage} ARI={ari:.4f}"
    assert r.n_clusters_ == 2


# ---- CW-5: 3 clusters ----
def test_three_clusters():
    from sklearn.cluster import AgglomerativeClustering as SKAGG
    from sklearn.metrics import adjusted_rand_score
    rng = np.random.RandomState(0)
    X = np.vstack([
        rng.normal(0, 0.3, size=(20, 2)),
        rng.normal(5, 0.3, size=(20, 2)),
        rng.normal(10, 0.3, size=(20, 2)),
    ])
    sk = SKAGG(n_clusters=3, linkage="ward").fit(X)
    r = fit_agglomerative(_mk_df(X), ["x0", "x1"], n_clusters=3,
                           linkage="ward",
                           memory_guard=MemoryGuard(max_memory_mb=500))
    ari = adjusted_rand_score(sk.labels_, r.labels_)
    assert ari > 0.99
    assert r.n_clusters_ == 3


# ---- CW-6: refuse_at_guard n > 5000 ----
def test_refuse_n_above_bound():
    X = np.zeros((6000, 2))
    with pytest.raises(MemoryLimitExceeded, match="n=6000"):
        fit_agglomerative(_mk_df(X), ["x0", "x1"], n_clusters=2,
                           linkage="ward",
                           memory_guard=MemoryGuard(max_memory_mb=10000))


# ---- CW-7: guard refuses tight ----
def test_guard_refuses_tight():
    X = _two_clusters(seed=0)
    with pytest.raises(MemoryLimitExceeded):
        fit_agglomerative(_mk_df(X), ["x0", "x1"], n_clusters=2,
                           linkage="ward",
                           memory_guard=MemoryGuard(max_memory_mb=0.001))


# ---- CW-8: input validation ----
def test_input_validation():
    X = _two_clusters(seed=0)
    df = _mk_df(X)
    with pytest.raises(ValueError, match="X_cols is empty"):
        fit_agglomerative(df, [], memory_guard=MemoryGuard(max_memory_mb=500))
    with pytest.raises(ValueError, match="n_clusters must be"):
        fit_agglomerative(df, ["x0", "x1"], n_clusters=1,
                           memory_guard=MemoryGuard(max_memory_mb=500))
    with pytest.raises(ValueError, match="linkage must be"):
        fit_agglomerative(df, ["x0", "x1"], linkage="bogus",
                           memory_guard=MemoryGuard(max_memory_mb=500))
    with pytest.raises(ValueError, match="metric"):
        fit_agglomerative(df, ["x0", "x1"], metric="cosine",
                           memory_guard=MemoryGuard(max_memory_mb=500))
    with pytest.raises(KeyError, match="columns missing"):
        fit_agglomerative(df, ["missing"],
                           memory_guard=MemoryGuard(max_memory_mb=500))


# ---- CW-9: state_nbytes ----
def test_state_nbytes():
    X = _two_clusters(seed=0)
    r = fit_agglomerative(_mk_df(X), ["x0", "x1"], n_clusters=2,
                           linkage="ward",
                           memory_guard=MemoryGuard(max_memory_mb=500))
    # 256 header + labels (n int64)
    expected = 256 + len(X) * 8
    assert r.state_nbytes() == expected


# ---- CW-10: to_dict ----
def test_to_dict():
    X = _two_clusters(seed=0)
    r = fit_agglomerative(_mk_df(X), ["x0", "x1"], n_clusters=2,
                           linkage="complete",
                           memory_guard=MemoryGuard(max_memory_mb=500))
    d = r.to_dict()
    assert d["n_clusters"] == 2
    assert d["linkage"] == "complete"
    assert d["n_train"] == len(X)


# ---- CW-11: reproducible ----
def test_reproducible():
    X = _two_clusters(seed=0)
    r1 = fit_agglomerative(_mk_df(X), ["x0", "x1"], n_clusters=2,
                            linkage="ward",
                            memory_guard=MemoryGuard(max_memory_mb=500))
    r2 = fit_agglomerative(_mk_df(X), ["x0", "x1"], n_clusters=2,
                            linkage="ward",
                            memory_guard=MemoryGuard(max_memory_mb=500))
    np.testing.assert_array_equal(r1.labels_, r2.labels_)


# ---- CW-12: labels compacted (0..k-1) ----
def test_labels_compacted():
    X = _two_clusters(seed=0)
    r = fit_agglomerative(_mk_df(X), ["x0", "x1"], n_clusters=2,
                           linkage="ward",
                           memory_guard=MemoryGuard(max_memory_mb=500))
    assert set(r.labels_) == {0, 1}, f"labels={set(r.labels_)}"
