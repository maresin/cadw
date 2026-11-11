"""v0.9.144: tests for cadw.ops.linalg.knn_query + core_distances (L1, L2)."""
import ctypes
import gc

import numpy as np
import pytest

from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ops.linalg import (
    knn_query, core_distances, sq_dists,
)


def _malloc_trim():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _brute_knn(X, Q, k):
    """Brute-force kNN via sq_dists + argsort (reference)."""
    D = sq_dists(Q, X)
    order = np.argsort(D, axis=1, kind="stable")
    return order[:, :k], np.sqrt(np.maximum(
        np.take_along_axis(D, order[:, :k], axis=1), 0.0))


# ---- correctness vs brute force ----

@pytest.mark.parametrize("n_x,n_q,d,k", [
    (100, 10, 3, 5),
    (500, 20, 5, 1),
    (1000, 50, 10, 10),
])
def test_knn_matches_bruteforce(n_x, n_q, d, k):
    rng = np.random.RandomState(0)
    X = rng.normal(size=(n_x, d))
    Q = rng.normal(size=(n_q, d))
    idx, dist = knn_query(X, Q, k, return_dist=True)
    idx_ref, dist_ref = _brute_knn(X, Q, k)
    np.testing.assert_array_equal(idx, idx_ref)
    np.testing.assert_allclose(dist, dist_ref, rtol=1e-10, atol=1e-10)


def test_knn_index_only():
    rng = np.random.RandomState(1)
    X = rng.normal(size=(100, 3))
    Q = rng.normal(size=(10, 3))
    idx = knn_query(X, Q, 5)
    idx_ref, _ = _brute_knn(X, Q, 5)
    np.testing.assert_array_equal(idx, idx_ref)


def test_knn_k_equals_n():
    """k == n_x returns all points sorted by distance."""
    rng = np.random.RandomState(2)
    X = rng.normal(size=(50, 3))
    Q = rng.normal(size=(5, 3))
    idx, dist = knn_query(X, Q, 50, return_dist=True)
    assert idx.shape == (5, 50)
    # distances should be non-decreasing per row
    assert (np.diff(dist, axis=1) >= -1e-10).all()


def test_knn_chunk_invariant():
    """Same result regardless of chunk_size."""
    rng = np.random.RandomState(3)
    X = rng.normal(size=(500, 5))
    Q = rng.normal(size=(10, 5))
    idx_a, dist_a = knn_query(X, Q, 7, chunk_size=17, return_dist=True)
    idx_b, dist_b = knn_query(X, Q, 7, chunk_size=1000, return_dist=True)
    np.testing.assert_array_equal(idx_a, idx_b)
    np.testing.assert_allclose(dist_a, dist_b, rtol=1e-12)


def test_knn_cosine_metric():
    rng = np.random.RandomState(4)
    X = rng.normal(size=(200, 4))
    Q = rng.normal(size=(20, 4))
    idx, dist = knn_query(X, Q, 5, metric="cosine", return_dist=True)
    # brute cosine
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
    D = 1.0 - Qn @ Xn.T
    order = np.argsort(D, axis=1, kind="stable")[:, :5]
    dist_ref = np.take_along_axis(D, order, axis=1)
    np.testing.assert_array_equal(idx, order)
    np.testing.assert_allclose(dist, dist_ref, rtol=1e-10)


def test_knn_invalid_args():
    X = np.zeros((10, 3))
    Q = np.zeros((2, 3))
    with pytest.raises(ValueError, match="k must be"):
        knn_query(X, Q, 0)
    with pytest.raises(ValueError, match="k must be"):
        knn_query(X, Q, 11)
    with pytest.raises(ValueError, match="metric must be"):
        knn_query(X, Q, 3, metric="hamming")
    with pytest.raises(ValueError, match="X is empty"):
        knn_query(np.zeros((0, 3)), Q, 3)


def test_knn_guard_refuses_at_tight():
    """Large n_q * k with tiny guard -> MemoryLimitExceeded."""
    X = np.zeros((100, 3))
    Q = np.zeros((10_000, 3))
    with pytest.raises(MemoryLimitExceeded):
        knn_query(X, Q, 100, memory_guard=MemoryGuard(max_memory_mb=0.1))


def test_knn_ooc_in_n_x():
    """State independent of n_x (bounded by n_q*k)."""
    rng = np.random.RandomState(5)
    Q = rng.normal(size=(20, 5))
    for n_x in (100, 1000):
        X = rng.normal(size=(n_x, 5))
        idx = knn_query(X, Q, 5)
        # same result as brute
        idx_ref, _ = _brute_knn(X, Q, 5)
        np.testing.assert_array_equal(idx, idx_ref)


# ---- core_distances ----

@pytest.mark.parametrize("n,k", [(50, 1), (100, 5), (200, 10)])
def test_core_distances_matches_brute(n, k):
    rng = np.random.RandomState(6)
    X = rng.normal(size=(n, 3))
    core = core_distances(X, k)
    # brute: sort distances per row, exclude self (dist 0 at position 0),
    # take k-th NN (position k in sorted including self)
    D = sq_dists(X, X)
    Ds = np.sort(D, axis=1)
    ref = np.sqrt(np.maximum(Ds[:, k], 0.0))
    np.testing.assert_allclose(core, ref, rtol=1e-10)


def test_core_distances_invalid_k():
    X = np.zeros((10, 3))
    with pytest.raises(ValueError, match="k must be"):
        core_distances(X, 0)
    with pytest.raises(ValueError, match="k must be"):
        core_distances(X, 10)
    with pytest.raises(ValueError, match="k must be"):
        core_distances(X, 15)


def test_core_distances_chunk_invariant():
    rng = np.random.RandomState(7)
    X = rng.normal(size=(300, 4))
    a = core_distances(X, 5, chunk_size=23)
    b = core_distances(X, 5, chunk_size=1000)
    np.testing.assert_allclose(a, b, rtol=1e-12)

# ---- v0.9.197: knn_query_stream (bit-identical to knn_query) ----

def test_knn_stream_ndarray_batches_bit_identical():
    """X_iter yields ndarray batches; == knn_query(X, Q, k)."""
    from cadw.ops.linalg import knn_query_stream
    rng = np.random.RandomState(0)
    n, d, n_q, k = 200, 4, 20, 5
    X = rng.randn(n, d)
    Q = rng.randn(n_q, d)
    idx_a, dist_a = knn_query(X, Q, k, return_dist=True,
                               memory_guard=MemoryGuard(max_memory_mb=500))
    batches = [X[i:i + 50] for i in range(0, n, 50)]
    idx_b, dist_b = knn_query_stream(iter(batches), Q, k, return_dist=True,
                                      memory_guard=MemoryGuard(max_memory_mb=500))
    np.testing.assert_array_equal(idx_a, idx_b)
    np.testing.assert_allclose(dist_a, dist_b, atol=1e-12)


def test_knn_stream_recordbatch_bit_identical():
    """X_iter yields pa.RecordBatch; == knn_query(X, Q, k)."""
    import pyarrow as pa
    from cadw.ops.linalg import knn_query_stream
    rng = np.random.RandomState(1)
    n, d, n_q, k = 150, 3, 15, 4
    X = rng.randn(n, d)
    Q = rng.randn(n_q, d)
    idx_a, dist_a = knn_query(X, Q, k, return_dist=True,
                               memory_guard=MemoryGuard(max_memory_mb=500))
    batches = []
    for i in range(0, n, 30):
        chunk = X[i:i + 30]
        cols = [pa.array(chunk[:, j], type=pa.float64())
                for j in range(d)]
        batches.append(pa.RecordBatch.from_arrays(cols,
                     names=[f"x{j}" for j in range(d)]))
    idx_b, dist_b = knn_query_stream(iter(batches), Q, k,
                                      return_dist=True,
                                      memory_guard=MemoryGuard(max_memory_mb=500))
    np.testing.assert_array_equal(idx_a, idx_b)
    np.testing.assert_allclose(dist_a, dist_b, atol=1e-12)


def test_knn_stream_single_batch_k_eq_n():
    """Edge: k=n_x, single batch."""
    from cadw.ops.linalg import knn_query_stream
    rng = np.random.RandomState(2)
    X = rng.randn(10, 3)
    Q = rng.randn(2, 3)
    idx, dist = knn_query_stream(iter([X]), Q, 10, return_dist=True,
                                  memory_guard=MemoryGuard(max_memory_mb=500))
    assert idx.shape == (2, 10)
    assert (idx >= 0).all() and (idx < 10).all()


def test_knn_stream_cosine_metric():
    """Cosine metric bit-identical."""
    from cadw.ops.linalg import knn_query_stream
    rng = np.random.RandomState(3)
    X = rng.randn(80, 4)
    Q = rng.randn(10, 4)
    idx_a, dist_a = knn_query(X, Q, 5, return_dist=True, metric="cosine",
                               memory_guard=MemoryGuard(max_memory_mb=500))
    batches = [X[i:i + 20] for i in range(0, 80, 20)]
    idx_b, dist_b = knn_query_stream(iter(batches), Q, 5,
                                      return_dist=True, metric="cosine",
                                      memory_guard=MemoryGuard(max_memory_mb=500))
    np.testing.assert_array_equal(idx_a, idx_b)
    np.testing.assert_allclose(dist_a, dist_b, atol=1e-12)


def test_knn_stream_k_gt_n_raises():
    """k > n_x raises (checked at end of iteration)."""
    from cadw.ops.linalg import knn_query_stream
    X = np.random.RandomState(4).randn(5, 2)
    Q = np.random.RandomState(5).randn(2, 2)
    with pytest.raises(ValueError, match="k must be in"):
        knn_query_stream(iter([X]), Q, 10)

# ---- v0.9.199: nan_sq_dists (sklearn nan_euclidean parity) ----

def test_nan_sq_dists_sklearn_parity_no_nan():
    """No NaN -> matches sklearn nan_euclidean_distances(squared=True)."""
    from sklearn.metrics.pairwise import nan_euclidean_distances
    from cadw.ops.linalg import nan_sq_dists
    rng = np.random.RandomState(0)
    Q = rng.randn(10, 4)
    X = rng.randn(20, 4)
    d_ref = nan_euclidean_distances(Q, X, squared=True)
    d_cadw = nan_sq_dists(Q, X)
    np.testing.assert_allclose(d_cadw, d_ref, atol=1e-12, equal_nan=True)


def test_nan_sq_dists_sklearn_parity_with_nan():
    """With NaN -> matches sklearn bit-for-bit."""
    from sklearn.metrics.pairwise import nan_euclidean_distances
    from cadw.ops.linalg import nan_sq_dists
    rng = np.random.RandomState(1)
    Q = rng.randn(8, 4)
    X = rng.randn(15, 4)
    # inject NaN patterns
    Q[rng.rand(*Q.shape) < 0.2] = np.nan
    X[rng.rand(*X.shape) < 0.2] = np.nan
    d_ref = nan_euclidean_distances(Q, X, squared=True)
    d_cadw = nan_sq_dists(Q, X)
    np.testing.assert_allclose(d_cadw, d_ref, atol=1e-12, equal_nan=True)


def test_nan_sq_dists_no_common_dims_nan():
    """Pair with no common non-NaN dims -> NaN."""
    from cadw.ops.linalg import nan_sq_dists
    Q = np.array([[1.0, np.nan, np.nan]])
    X = np.array([[np.nan, 2.0, np.nan]])
    d = nan_sq_dists(Q, X)
    assert np.isnan(d).all()


def test_nan_sq_dists_partial_common():
    """Pair with 1 of 3 common dims: sum_sq * d / n_common = sum_sq * 3."""
    from cadw.ops.linalg import nan_sq_dists
    Q = np.array([[1.0, 2.0, np.nan]])     # dims 0,1 valid
    X = np.array([[3.0, np.nan, 5.0]])     # dim 0 valid
    # common: only dim 0; diff^2 = (1-3)^2 = 4; n_common=1, d=3
    # → 4 * 3/1 = 12
    d = nan_sq_dists(Q, X)
    assert d[0, 0] == 12.0


def test_nan_sq_dists_empty():
    """Empty X -> (n_q, 0)."""
    from cadw.ops.linalg import nan_sq_dists
    Q = np.random.RandomState(2).randn(5, 3)
    X = np.empty((0, 3))
    d = nan_sq_dists(Q, X)
    assert d.shape == (5, 0)


def test_nan_sq_dists_d_mismatch_raises():
    from cadw.ops.linalg import nan_sq_dists
    Q = np.zeros((3, 2))
    X = np.zeros((4, 3))
    with pytest.raises(ValueError, match="nan_sq_dists: Q.d"):
        nan_sq_dists(Q, X)

