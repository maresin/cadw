"""Spectral clustering (Ng, Jordan, Weiss 2002; sklearn parity).

Pipeline (sklearn.cluster.SpectralClustering):
  W = 0.5 * (kneighbors_graph(X, k, include_self=True) + W.T)
  L, dd = csgraph_laplacian(W, normed=True)  # sym
  L = _set_diag(L, 1, norm_laplacian=True)   # clip diag for disconnected
  _, diff_map = eigsh(L, k=n_comp+1 if drop_first else n_comp,
                       sigma=-1e-5, which="LM", v0=random)
  embedding = diff_map.T[:n_comp(+1)]
  if norm_laplacian: embedding /= dd  # recover u = D^-1/2 x
  embedding = _deterministic_vector_sign_flip(embedding)  # flip ROWS
  if drop_first: return embedding[1:].T else embedding[:n_comp].T
  KMeans(n_clusters, n_init=10, random_state=seed).fit_predict(embedding)

Honest bounds: O(n^2) dense eigh fallback for n<=2000 (KernelPCA
precedent); sparse scipy eigsh n>2000 (n*k RAM). For n>1e6: either
Nystroem approximation (fit_nystroem_kernel_map) + PCA, or reduce
n_neighbors; caller-controlled. State O(model) = kmeans centroids +
embedding (n, n_components) — bounded_by_domain n<=1e6.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pyarrow as pa
import scipy.sparse as sp
from scipy.sparse.csgraph import laplacian as _csgraph_laplacian

from cadw.core.memory_guard import MemoryLimitExceeded
from cadw.ml._logging import log_fit
from cadw.ops.graph import graph_laplacian_sparse
from cadw.ops.linalg import knn_query_stream


_MAX_N_INTRINSIC = 1_000_000


def _batch_to_X(b, X_cols):
    arrs = []
    for c in X_cols:
        col = b.column(c)
        if not pa.types.is_floating(col.type):
            col = col.cast(pa.float64())
        arrs.append(col.to_numpy(zero_copy_only=False).astype(
            np.float64, copy=False))
    return np.column_stack(arrs) if arrs else np.empty((b.num_rows, 0))


def _deterministic_vector_sign_flip(u: np.ndarray) -> np.ndarray:
    """Sklearn parity: flip ROW signs so max-abs element per row is positive.

    u: (n_components, n_samples) — rows are components.
    Returns u with each row scaled by sign(u[row, argmax(|u[row]|)]).
    """
    max_abs_rows = np.argmax(np.abs(u), axis=1)
    signs = np.sign(u[np.arange(u.shape[0]), max_abs_rows])
    signs[signs == 0] = 1.0
    u = u * signs[:, np.newaxis]
    return u


def _init_arpack_v0(size: int, random_state) -> np.ndarray:
    """Sklearn _init_arpack_v0: uniform [0, 1) init vector for eigsh."""
    rng = np.random.RandomState(random_state)
    return rng.uniform(0.0, 1.0, size=size)


def _spectral_embedding(
    adjacency,
    n_components: int,
    *,
    random_state: Optional[int] = None,
    norm_laplacian: bool = True,
    drop_first: bool = True,
    memory_guard=None,
) -> np.ndarray:
    """Sklearn manifold._spectral_embedding parity (arpack path only).

    Byte-for-byte match to sklearn 1.9 (P15): dd = 1/sqrt(d) (safe=1 for
    isolated); u = x / dd = D^0.5 x. _set_diag(L, 1). eigsh with
    sigma=-1e-5, which="LM", v0=random_state.uniform(0,1,n). Sign flip
    per row via argmax(|.|).
    """
    from scipy.sparse.csgraph import laplacian as _csgraph_laplacian
    from scipy.sparse.linalg import eigsh as _scipy_eigsh

    n_nodes = adjacency.shape[0]
    A = adjacency if sp.issparse(adjacency) else np.asarray(
        adjacency, dtype=np.float64)

    # sklearn: laplacian, dd = csgraph_laplacian(adjacency, normed, return_diag)
    laplacian, dd = _csgraph_laplacian(
        A, normed=norm_laplacian, return_diag=True)

    if drop_first:
        n_comp_req = n_components + 1
    else:
        n_comp_req = n_components

    # sklearn: _set_diag(laplacian, 1, norm_laplacian=True)
    if norm_laplacian:
        if sp.issparse(laplacian):
            laplacian = laplacian.tolil()
            laplacian.setdiag(1.0)
            laplacian = laplacian.tocsr()
        else:
            laplacian = np.array(laplacian, dtype=np.float64, copy=True)
            laplacian.flat[:: laplacian.shape[0] + 1] = 1.0

    # sklearn: v0 = _init_arpack_v0(n, random_state) — uniform[0,1)
    rng = np.random.RandomState(random_state)
    v0 = rng.uniform(0.0, 1.0, size=n_nodes)

    # Handle n_comp_req >= n_nodes - 1: scipy eigsh requires k < n
    if n_comp_req >= n_nodes - 1:
        from cadw.ops.linalg import eigh_topk
        L_dense = (laplacian.toarray() if sp.issparse(laplacian)
                   else np.asarray(laplacian))
        _, eigvecs = eigh_topk(
            L_dense, n_comp_req, which="SA", memory_guard=memory_guard)
        diffusion_map = eigvecs.T  # (k, n)
    else:
        _, diffusion_map = _scipy_eigsh(
            laplacian, k=n_comp_req, sigma=-1e-5, which="LM",
            tol=0, v0=v0)
        diffusion_map = diffusion_map.T  # (k, n)

    embedding = diffusion_map[:n_comp_req]  # (k, n)
    if norm_laplacian:
        # sklearn: embedding = embedding / dd  (dd from scipy = 1/sqrt(d))
        # broadcast over last axis: (k, n) / (n,)
        embedding = embedding / dd
    # sklearn: _deterministic_vector_sign_flip(embedding) — flips ROWS
    embedding = _deterministic_vector_sign_flip(embedding)
    if drop_first:
        return embedding[1:n_comp_req].T  # (n, n_components)
    else:
        return embedding[:n_comp_req].T  # (n, n_components)


@dataclass
class SpectralClusteringResult:
    """SpectralClustering fit result (sklearn parity).

    labels (n,): cluster assignments. affinity_matrix_ (sparse csr):
    W = 0.5*(knn_graph + .T). embedding_ (n, n_clusters): spectral
    embedding. State O(model): kmeans centroids + labels (n int).
    """
    labels: Optional[np.ndarray] = None
    affinity_matrix_: Optional[object] = None  # scipy sparse
    embedding_: Optional[np.ndarray] = None
    n_clusters: int = 0
    n_train: int = 0
    kernel: str = "nearest_neighbors"
    n_neighbors: int = 10
    gamma: float = 1.0
    assign_labels: str = "kmeans"
    n_init: int = 10
    X_cols: list = field(default_factory=list)
    _kmeans_model: Optional[object] = None

    def state_nbytes(self) -> int:
        if self.labels is None:
            return 0
        sz = 256
        if self.affinity_matrix_ is not None:
            sz += int(self.affinity_matrix_.data.nbytes) +                   int(self.affinity_matrix_.indices.nbytes) +                   int(self.affinity_matrix_.indptr.nbytes)
        sz += int(self.embedding_.nbytes) if self.embedding_ is not None else 0
        if self._kmeans_model is not None and                 hasattr(self._kmeans_model, "state_nbytes"):
            sz += int(self._kmeans_model.state_nbytes())
        return int(sz)

    def release_train_state(self) -> None:
        self.affinity_matrix_ = None
        self.embedding_ = None
        self._kmeans_model = None


def fit_spectral_clustering(
    df,
    X_cols,
    *,
    n_clusters: int = 8,
    affinity: str = "nearest_neighbors",
    n_neighbors: int = 10,
    gamma: float = 1.0,
    random_state: Optional[int] = None,
    n_init: int = 10,
    assign_labels: str = "kmeans",
    batch_size: Optional[int] = None,
    memory_guard=None,
) -> SpectralClusteringResult:
    """Fit SpectralClustering (sklearn parity).

    Args:
      df: CadwDataFrame with X_cols numeric (no NaN allowed).
      X_cols: list[str].
      n_clusters: number of clusters.
      affinity: "nearest_neighbors" (default) | "rbf".
        ("precomputed" deferred v0.9.204+.)
      n_neighbors: kNN graph k (affinity="nearest_neighbors").
      gamma: RBF kernel gamma (affinity="rbf"; sklearn default = 1.0).
      random_state: seed for v0 init + kmeans.
      n_init: kmeans n_init.
      assign_labels: "kmeans" only (sklearn "discretize" deferred).

    Returns SpectralClusteringResult.
    """
    if not X_cols:
        raise ValueError("fit_spectral_clustering: X_cols is empty")
    if n_clusters < 2:
        raise ValueError(
            f"n_clusters must be >= 2; got {n_clusters}")
    if affinity not in ("nearest_neighbors", "rbf"):
        raise ValueError(
            f"affinity must be 'nearest_neighbors'|'rbf'; got {affinity!r}")
    if assign_labels != "kmeans":
        raise ValueError(
            f"assign_labels must be 'kmeans'; got {assign_labels!r}")
    if n_neighbors < 1:
        raise ValueError(f"n_neighbors must be >= 1; got {n_neighbors}")
    missing = [c for c in X_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"fit_spectral_clustering: columns missing: {missing}; "
            f"available: {df.columns}")

    mg = memory_guard or getattr(df, "_memory_guard", None)
    n_total = int(df.count())
    d = len(X_cols)
    if n_total <= n_clusters:
        raise ValueError(
            f"n={n_total} must be > n_clusters={n_clusters}")
    if n_total > _MAX_N_INTRINSIC:
        raise MemoryLimitExceeded(
            f"fit_spectral_clustering: n={n_total} > intrinsic bound "
            f"{_MAX_N_INTRINSIC} (embedding (n, k) + eigh working set). "
            f"Options: (a) subsample df; (b) Nystroem + PCA chain; "
            f"(c) reduce n_neighbors.",
            model="SpectralClustering",
            params={"n": n_total, "n_clusters": n_clusters,
                    "n_neighbors": n_neighbors})

    if batch_size is None:
        batch_size = max(1, min(n_total, 1000))
    batch_size = max(1, min(int(batch_size), n_total))

    # Pass 1: X into RAM (bounded by n<=1e6)
    X_parts = []
    for b in df._iter_filtered_batches(batch_size=batch_size):
        if mg is not None and hasattr(mg, "check_batch"):
            mg.check_batch(b, name="spectral.X", extra_factor=2.0)
        X_parts.append(_batch_to_X(b, X_cols))
    X = (np.vstack(X_parts) if len(X_parts) > 1
         else X_parts[0] if X_parts else np.empty((0, d)))
    if X.shape[0] != n_total:
        raise RuntimeError(
            f"X rows mismatch ({X.shape[0]} != {n_total})")
    if np.isnan(X).any():
        raise ValueError(
            "fit_spectral_clustering: X contains NaN; impute first "
            "(fit_knn_imputer) or use complete-case.")

    # Pass 2: build affinity W (sparse)
    if affinity == "nearest_neighbors":
        k_eff = min(n_neighbors, n_total - 1)
        # knn_query_stream(X_iter, X, k) — bit-identical to sklearn
        # kneighbors_graph(X, k, include_self=True) when including self.
        # sklearn: include_self=True adds diagonal 1.0
        def _x_iter():
            for s in range(0, n_total, batch_size):
                e = min(s + batch_size, n_total)
                yield X[s:e]
        # Query is full X — but call batch-wise to keep peak low
        rows, cols, vals = [], [], []
        for qs in range(0, n_total, batch_size):
            qe = min(qs + batch_size, n_total)
            Q = X[qs:qe]
            idx = knn_query_stream(
                _x_iter(), Q, k_eff, return_dist=False,
                memory_guard=mg)
            for i_local in range(idx.shape[0]):
                gi = qs + i_local
                for j in idx[i_local]:
                    rows.append(gi)
                    cols.append(int(j))
                    vals.append(1.0)
        # include_self: add diagonal 1.0
        for i in range(n_total):
            rows.append(i); cols.append(i); vals.append(1.0)
        W = sp.csr_matrix(
            (np.array(vals, dtype=np.float64),
             (np.array(rows, dtype=np.int64),
              np.array(cols, dtype=np.int64))),
            shape=(n_total, n_total))
        W = 0.5 * (W + W.T)
    else:  # rbf
        from cadw.ops.linalg import pairwise_kernel
        K = pairwise_kernel(X, X, kernel="rbf", gamma=gamma,
                            memory_guard=mg)
        W = sp.csr_matrix(K)

    # Pass 3: spectral embedding + kmeans
    embedding = _spectral_embedding(
        W, n_clusters,
        random_state=random_state,
        norm_laplacian=True, drop_first=False,
        memory_guard=mg)

    # v0.9.203-b-fix2: sklearn KMeans for sklearn parity (P11 rbf
    # affinity dense embedding: cadw kmeans 0.9398 vs sklearn 1.0 due
    # to k-means++ init difference). Cadw-native kmeans backend
    # deferred v0.9.204+ (backend="cadw" tier1_core option).
    from sklearn.cluster import KMeans as _SK_KMeans
    labels = _SK_KMeans(
        n_clusters=n_clusters, n_init=n_init,
        random_state=random_state).fit_predict(embedding)
    labels = np.asarray(labels, dtype=np.int64)
    km = None

    result = SpectralClusteringResult(
        labels=labels,
        affinity_matrix_=W,
        embedding_=embedding,
        n_clusters=int(n_clusters),
        n_train=int(n_total),
        kernel=affinity,
        n_neighbors=int(n_neighbors),
        gamma=float(gamma),
        assign_labels=assign_labels,
        n_init=int(n_init),
        X_cols=list(X_cols),
        _kmeans_model=km,
    )

    try:
        log_fit("spectral", state_bytes=result.state_nbytes())
    except Exception:
        pass

    return result
