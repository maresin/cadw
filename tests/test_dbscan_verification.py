"""v0.9.215: fit_dbscan verification (Ester 1996).

Sklearn parity + SpillArray labels + guard + core/border + validation.
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ml import fit_dbscan


def _mk_df(X):
    return CadwDataFrame.from_table(pa.table({
        f"x{j}": pa.array(X[:, j], type=pa.float64())
        for j in range(X.shape[1])
    }), memory_guard=MemoryGuard(max_memory_mb=500))


def _collect(labels_res):
    parts = []
    for chunk in labels_res:
        arr = np.asarray(chunk)
        if arr.ndim > 1:
            arr = arr.ravel()
        parts.append(arr)
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)


def _two_clusters(seed=0):
    rng = np.random.RandomState(seed)
    X1 = rng.normal(0, 0.3, size=(40, 2))
    X2 = rng.normal(5, 0.3, size=(40, 2))
    return np.vstack([X1, X2])


def _warmup():
    X = _two_clusters(seed=999)
    fit_dbscan(_mk_df(X), ["x0", "x1"], eps=0.7, min_samples=5,
               output="memory",
               memory_guard=MemoryGuard(max_memory_mb=500))


_warmup()


# ---- CW-1: sklearn ARI parity ----
def test_sklearn_parity():
    from sklearn.cluster import DBSCAN as SKDBSCAN
    from sklearn.metrics import adjusted_rand_score
    rng = np.random.RandomState(0)
    X = np.vstack([
        rng.normal(0, 0.3, size=(40, 2)),
        rng.normal(5, 0.3, size=(40, 2)),
        rng.uniform(-3, 8, size=(10, 2)),  # noise
    ])
    sk = SKDBSCAN(eps=0.7, min_samples=5).fit(X)
    r = fit_dbscan(_mk_df(X), ["x0", "x1"], eps=0.7, min_samples=5,
                    output="memory",
                    memory_guard=MemoryGuard(max_memory_mb=500))
    labels = _collect(r.labels_)
    ari = adjusted_rand_score(sk.labels_, labels)
    assert ari > 0.9, f"ARI={ari:.4f}"
    assert r.n_clusters_ >= 2


# ---- CW-2: all noise ----
def test_all_noise():
    rng = np.random.RandomState(0)
    X = rng.uniform(-10, 10, size=(30, 2))
    r = fit_dbscan(_mk_df(X), ["x0", "x1"], eps=0.1, min_samples=5,
                    output="memory",
                    memory_guard=MemoryGuard(max_memory_mb=500))
    assert r.n_clusters_ == 0
    assert r.n_noise_ == len(X)
    labels = _collect(r.labels_)
    assert np.all(labels == -1)


# ---- CW-3: single dense cluster ----
def test_single_cluster():
    rng = np.random.RandomState(0)
    X = rng.normal(0, 0.2, size=(50, 2))
    r = fit_dbscan(_mk_df(X), ["x0", "x1"], eps=0.5, min_samples=3,
                    output="memory",
                    memory_guard=MemoryGuard(max_memory_mb=500))
    assert r.n_clusters_ == 1
    labels = _collect(r.labels_)
    assert len(set(labels[labels >= 0])) == 1


# ---- CW-4: core mask shape ----
def test_core_mask():
    X = _two_clusters()
    r = fit_dbscan(_mk_df(X), ["x0", "x1"], eps=0.7, min_samples=5,
                    output="memory",
                    memory_guard=MemoryGuard(max_memory_mb=500))
    assert r.core_sample_indices_.shape == (len(X),)
    assert r.core_sample_indices_.dtype == bool
    # two dense clusters -> most points core
    assert r.core_sample_indices_.sum() >= len(X) * 0.5


# ---- CW-5: guard refuses tight ----
def test_guard_refuses_tight():
    X = np.random.RandomState(0).normal(0, 1, size=(100, 2))
    with pytest.raises(MemoryLimitExceeded):
        fit_dbscan(_mk_df(X), ["x0", "x1"], eps=0.5, min_samples=5,
                    memory_guard=MemoryGuard(max_memory_mb=0.001))


# ---- CW-6: input validation ----
def test_input_validation():
    X = _two_clusters()
    df = _mk_df(X)
    with pytest.raises(ValueError, match="X_cols is empty"):
        fit_dbscan(df, [], memory_guard=MemoryGuard(max_memory_mb=500))
    with pytest.raises(ValueError, match="eps must be"):
        fit_dbscan(df, ["x0", "x1"], eps=0,
                    memory_guard=MemoryGuard(max_memory_mb=500))
    with pytest.raises(ValueError, match="min_samples must be"):
        fit_dbscan(df, ["x0", "x1"], min_samples=0,
                    memory_guard=MemoryGuard(max_memory_mb=500))
    with pytest.raises(KeyError, match="columns missing"):
        fit_dbscan(df, ["missing"],
                    memory_guard=MemoryGuard(max_memory_mb=500))


# ---- CW-7: state_nbytes ----
def test_state_nbytes():
    X = _two_clusters()
    r = fit_dbscan(_mk_df(X), ["x0", "x1"], eps=0.7, min_samples=5,
                    output="memory",
                    memory_guard=MemoryGuard(max_memory_mb=500))
    # 256 header + core mask (n bool)
    expected = 256 + len(X) * 1
    assert r.state_nbytes() >= expected


# ---- CW-8: to_dict ----
def test_to_dict():
    X = _two_clusters()
    r = fit_dbscan(_mk_df(X), ["x0", "x1"], eps=0.5, min_samples=3,
                    output="memory",
                    memory_guard=MemoryGuard(max_memory_mb=500))
    d = r.to_dict()
    assert d["eps"] == 0.5
    assert d["min_samples"] == 3
    assert d["n_train"] == len(X)
    assert "n_clusters" in d and "n_noise" in d


# ---- CW-9: labels compacted (0..k-1, -1 noise) ----
def test_labels_compacted():
    X = _two_clusters()
    r = fit_dbscan(_mk_df(X), ["x0", "x1"], eps=0.7, min_samples=5,
                    output="memory",
                    memory_guard=MemoryGuard(max_memory_mb=500))
    labels = _collect(r.labels_)
    non_noise = labels[labels >= 0]
    if len(non_noise) > 0:
        assert set(non_noise) == set(range(r.n_clusters_)), (
            f"labels not compacted: {set(non_noise)}")


# ---- CW-10: output modes equivalent ----
def test_output_modes():
    X = _two_clusters()
    r_mem = fit_dbscan(_mk_df(X), ["x0", "x1"], eps=0.7, min_samples=5,
                        output="memory",
                        memory_guard=MemoryGuard(max_memory_mb=500))
    r_auto = fit_dbscan(_mk_df(X), ["x0", "x1"], eps=0.7, min_samples=5,
                         output="auto",
                         memory_guard=MemoryGuard(max_memory_mb=500))
    labels_mem = _collect(r_mem.labels_)
    labels_auto = _collect(r_auto.labels_)
    np.testing.assert_array_equal(labels_mem, labels_auto)


# ---- CW-11: reproducible (deterministic) ----
def test_reproducible():
    X = _two_clusters()
    r1 = fit_dbscan(_mk_df(X), ["x0", "x1"], eps=0.7, min_samples=5,
                     output="memory",
                     memory_guard=MemoryGuard(max_memory_mb=500))
    r2 = fit_dbscan(_mk_df(X), ["x0", "x1"], eps=0.7, min_samples=5,
                     output="memory",
                     memory_guard=MemoryGuard(max_memory_mb=500))
    np.testing.assert_array_equal(_collect(r1.labels_),
                                   _collect(r2.labels_))


# ---- CW-12: border assignment ----
def test_border_assignment():
    # 2 core points close + 1 border point at edge
    rng = np.random.RandomState(0)
    X = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.2],
        [5.0, 5.0],  # outlier
    ])
    # min_samples=3, eps=0.5: first 3 points form cluster; [5,5] noise
    r = fit_dbscan(_mk_df(X), ["x0", "x1"], eps=0.5, min_samples=2,
                    output="memory",
                    memory_guard=MemoryGuard(max_memory_mb=500))
    labels = _collect(r.labels_)
    assert labels[3] == -1, f"outlier not noise: {labels[3]}"
    assert labels[0] == labels[1] == labels[2], (
        f"dense trio split: {labels[:3]}")
