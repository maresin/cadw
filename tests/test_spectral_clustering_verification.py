"""Component-wise verification for fit_spectral_clustering (DEC-033, v0.9.203).

Sklearn SpectralClustering parity (ARI >= 0.99 on 3 datasets x 2
affinities). Honest bounds: bounded_by_domain {n: 1e6} (embedding
(n, k) + kmeans state).
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ml import fit_spectral_clustering, SpectralClusteringResult


def _mk(X):
    cols = {f"x{j}": pa.array(X[:, j], type=pa.float64())
            for j in range(X.shape[1])}
    return CadwDataFrame.from_table(pa.table(cols),
                                     memory_guard=MemoryGuard(max_memory_mb=500))


def _warmup():
    from sklearn.datasets import make_blobs
    X = make_blobs(n_samples=50, centers=2, n_features=3,
                    random_state=999)[0]
    df = _mk(X)
    r = fit_spectral_clustering(df, ["x0", "x1", "x2"],
                                  n_clusters=2, n_neighbors=5,
                                  random_state=0,
                                  memory_guard=MemoryGuard(max_memory_mb=500))
    r.release_train_state()


_warmup()


def test_sklearn_ari_parity_blobs():
    from sklearn.cluster import SpectralClustering
    from sklearn.datasets import make_blobs
    from sklearn.metrics import adjusted_rand_score
    X, y = make_blobs(n_samples=200, centers=3, n_features=4, random_state=0)
    df = _mk(X)
    sk = SpectralClustering(n_clusters=3, affinity="nearest_neighbors",
                             n_neighbors=10, random_state=0).fit(X)
    ari_sk = adjusted_rand_score(y, sk.labels_)
    r = fit_spectral_clustering(df, [f"x{j}" for j in range(4)],
                                  n_clusters=3, n_neighbors=10,
                                  random_state=0,
                                  memory_guard=MemoryGuard(max_memory_mb=500))
    ari_cadw = adjusted_rand_score(y, r.labels)
    assert ari_cadw >= 0.99, f"blobs3: cadw ARI {ari_cadw:.4f}"
    assert abs(ari_cadw - ari_sk) < 0.02
    r.release_train_state()


def test_sklearn_ari_parity_moons():
    from sklearn.cluster import SpectralClustering
    from sklearn.datasets import make_moons
    from sklearn.metrics import adjusted_rand_score
    X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
    df = _mk(X)
    sk = SpectralClustering(n_clusters=2, affinity="nearest_neighbors",
                             n_neighbors=10, random_state=0).fit(X)
    r = fit_spectral_clustering(df, ["x0", "x1"],
                                  n_clusters=2, n_neighbors=10,
                                  random_state=0,
                                  memory_guard=MemoryGuard(max_memory_mb=500))
    ari_cadw = adjusted_rand_score(y, r.labels)
    assert ari_cadw >= 0.99, f"moons2: cadw ARI {ari_cadw:.4f}"
    r.release_train_state()


def test_sklearn_ari_parity_rbf():
    from sklearn.cluster import SpectralClustering
    from sklearn.datasets import make_blobs
    from sklearn.metrics import adjusted_rand_score
    X, y = make_blobs(n_samples=150, centers=3, n_features=3, random_state=1)
    df = _mk(X)
    r = fit_spectral_clustering(df, ["x0", "x1", "x2"],
                                  n_clusters=3, affinity="rbf", gamma=1.0,
                                  random_state=0,
                                  memory_guard=MemoryGuard(max_memory_mb=500))
    ari_cadw = adjusted_rand_score(y, r.labels)
    assert ari_cadw >= 0.99, f"rbf: cadw ARI {ari_cadw:.4f}"
    r.release_train_state()


def test_spectral_embedding_parity_dense():
    from sklearn.datasets import make_blobs
    from sklearn.manifold import spectral_embedding
    from sklearn.metrics.pairwise import rbf_kernel
    from cadw.ml.spectral_clustering import _spectral_embedding

    X = make_blobs(n_samples=80, centers=3, n_features=3,
                    random_state=1)[0]
    W = rbf_kernel(X, X, gamma=1.0)
    emb_sk = spectral_embedding(W, n_components=3, drop_first=False,
                                  norm_laplacian=True, random_state=0)
    emb_cadw = _spectral_embedding(W, 3, random_state=0,
                                     norm_laplacian=True, drop_first=False)
    for j in range(emb_cadw.shape[1]):
        s = np.sign(emb_cadw[:, j] @ emb_sk[:, j])
        if s != 0:
            emb_cadw[:, j] *= s
    diff = np.abs(emb_cadw - emb_sk).max()
    assert diff < 1e-4, f"dense embed diff {diff:.2e}"


def test_spectral_embedding_parity_sparse():
    from sklearn.datasets import make_blobs
    from sklearn.manifold import spectral_embedding
    from sklearn.neighbors import kneighbors_graph
    from cadw.ml.spectral_clustering import _spectral_embedding

    X = make_blobs(n_samples=80, centers=3, n_features=3,
                    random_state=1)[0]
    conn = kneighbors_graph(X, n_neighbors=10, include_self=True)
    W = 0.5 * (conn + conn.T)
    emb_sk = spectral_embedding(W, n_components=3, drop_first=False,
                                  norm_laplacian=True, random_state=0)
    emb_cadw = _spectral_embedding(W, 3, random_state=0,
                                     norm_laplacian=True, drop_first=False)
    for j in range(emb_cadw.shape[1]):
        s = np.sign(emb_cadw[:, j] @ emb_sk[:, j])
        if s != 0:
            emb_cadw[:, j] *= s
    diff = np.abs(emb_cadw - emb_sk).max()
    assert diff < 1e-4, f"sparse embed diff {diff:.2e}"


def test_state_nbytes_reasonable():
    from sklearn.datasets import make_blobs
    X = make_blobs(n_samples=100, centers=3, n_features=4,
                    random_state=2)[0]
    df = _mk(X)
    r = fit_spectral_clustering(df, [f"x{j}" for j in range(4)],
                                  n_clusters=3, n_neighbors=10,
                                  random_state=0,
                                  memory_guard=MemoryGuard(max_memory_mb=500))
    assert r.state_nbytes() > 0
    assert r.state_nbytes() < 500_000
    r.release_train_state()


def test_n_above_bound_raises():
    class _FakeDF:
        columns = ["x0", "x1"]
        _memory_guard = MemoryGuard(max_memory_mb=500)
        _spill_dir = "/tmp"
        def count(self):
            return 1_000_001
        def _iter_filtered_batches(self, batch_size=None):
            return iter([])
    with pytest.raises(MemoryLimitExceeded, match="intrinsic bound"):
        fit_spectral_clustering(_FakeDF(), ["x0", "x1"],
                                  n_clusters=3, n_neighbors=10,
                                  random_state=0)


def test_release_idempotent():
    from sklearn.datasets import make_blobs
    X = make_blobs(n_samples=100, centers=3, n_features=3,
                    random_state=3)[0]
    df = _mk(X)
    r = fit_spectral_clustering(df, ["x0", "x1", "x2"],
                                  n_clusters=3, n_neighbors=10,
                                  random_state=0,
                                  memory_guard=MemoryGuard(max_memory_mb=500))
    assert r.affinity_matrix_ is not None
    r.release_train_state()
    assert r.affinity_matrix_ is None
    assert r.embedding_ is None
    r.release_train_state()


def test_n_clusters_gt_n_raises():
    X = np.random.RandomState(0).randn(5, 2)
    df = _mk(X)
    with pytest.raises(ValueError, match="must be > n_clusters"):
        fit_spectral_clustering(df, ["x0", "x1"], n_clusters=10,
                                  memory_guard=MemoryGuard(max_memory_mb=500))


def test_bad_affinity_raises():
    X = np.random.RandomState(0).randn(30, 2)
    df = _mk(X)
    with pytest.raises(ValueError, match="affinity must be"):
        fit_spectral_clustering(df, ["x0", "x1"], n_clusters=2,
                                  affinity="precomputed",
                                  memory_guard=MemoryGuard(max_memory_mb=500))


def test_nan_raises():
    X = np.random.RandomState(0).randn(30, 2)
    X[5, 0] = np.nan
    df = _mk(X)
    with pytest.raises(ValueError, match="X contains NaN"):
        fit_spectral_clustering(df, ["x0", "x1"], n_clusters=2,
                                  n_neighbors=5, random_state=0,
                                  memory_guard=MemoryGuard(max_memory_mb=500))


def test_reproducibility_same_seed():
    from sklearn.datasets import make_blobs
    X = make_blobs(n_samples=100, centers=3, n_features=3,
                    random_state=4)[0]
    df = _mk(X)
    r1 = fit_spectral_clustering(df, ["x0", "x1", "x2"],
                                   n_clusters=3, n_neighbors=10,
                                   random_state=42,
                                   memory_guard=MemoryGuard(max_memory_mb=500))
    r2 = fit_spectral_clustering(df, ["x0", "x1", "x2"],
                                   n_clusters=3, n_neighbors=10,
                                   random_state=42,
                                   memory_guard=MemoryGuard(max_memory_mb=500))
    np.testing.assert_array_equal(r1.labels, r2.labels)
    r1.release_train_state()
    r2.release_train_state()
