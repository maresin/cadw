"""Spectral biclustering (Kluger 2003).

Generalized spectral biclustering on a matrix A (r x c). Rows and
columns are clustered independently after normalization + SVD.

Reference: Kluger et al. 2003, "Spectral biclustering of microarray
data: coclustering genes and conditions", Genome Research 13(4):703-716.

Sklearn parity: sklearn.cluster.SpectralBiclustering (svd_method=
'randomized', n_init=10, init='k-means++'). CADW uses exact svd_topk
(always sorted descending; sklearn randomized approximate on k>=3).

OOC patterns (ROADMAP §OOC by primitive):
  - svd_topk: Pattern #1 batch-layer (chunked Gram), state O(c^2).
    bounded_by_window {gram_dim=min(r,c): 1e4}.
  - A (r*c) materialized: sklearn _project_and_cluster intrinsic
    (A @ best_vt.T for rows, A.T @ best_ut.T for cols). Target
    streaming refactor v0.9.213+.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.ml._logging import log_fit
from cadw.ops.linalg import svd_topk


def _batch_to_X(b, X_cols):
    arrs = []
    for c in X_cols:
        col = b.column(c)
        if not pa.types.is_floating(col.type):
            col = col.cast(pa.float64())
        arrs.append(col.to_numpy(zero_copy_only=False).astype(
            np.float64, copy=False))
    return np.column_stack(arrs) if arrs else np.empty((b.num_rows, 0))


def _scale_normalize(A):
    """sklearn _scale_normalize. Returns (A_norm, row_diag, col_diag)."""
    A = np.maximum(A, 0.0)  # make_nonnegative
    row_sum = A.sum(axis=1)
    col_sum = A.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rd = 1.0 / np.sqrt(row_sum)
        cd = 1.0 / np.sqrt(col_sum)
    rd = np.where(np.isfinite(rd), rd, 0.0)
    cd = np.where(np.isfinite(cd), cd, 0.0)
    A_norm = rd[:, None] * A * cd[None, :]
    return A_norm, rd, cd


def _bistochastic_normalize(A, max_iter=1000, tol=1e-5):
    """sklearn _bistochastic_normalize. Iterative Sinkhorn."""
    A = np.maximum(A, 0.0)
    A_scaled = A
    for _ in range(max_iter):
        A_new, _, _ = _scale_normalize(A_scaled)
        dist = float(np.linalg.norm(A_scaled - A_new))
        A_scaled = A_new
        if dist < tol:
            break
    return A_scaled


def _log_normalize(A):
    """sklearn _log_normalize. Kluger log-interactions."""
    A = np.maximum(A, 1.0)  # make_nonnegative(min_value=1)
    L = np.log(A)
    row_avg = L.mean(axis=1, keepdims=True)
    col_avg = L.mean(axis=0, keepdims=True)
    avg = L.mean()
    return L - row_avg - col_avg + avg


def _kmeans_1d(v, n_clusters, random_state):
    """sklearn _k_means on 1D: returns centroid[labels] (piecewise approx)."""
    from sklearn.cluster import KMeans as _SK_KMeans
    km = _SK_KMeans(n_clusters=n_clusters, n_init=10,
                    init="k-means++", random_state=random_state)
    labels = km.fit_predict(v.reshape(-1, 1))
    return km.cluster_centers_[labels].ravel()


def _fit_best_piecewise(vectors, n_best, n_clusters, random_state):
    """sklearn _fit_best_piecewise. vectors (k_sv, n_entities)."""
    n_rows = vectors.shape[0]
    piecewise = np.empty_like(vectors)
    for i in range(n_rows):
        piecewise[i] = _kmeans_1d(vectors[i], n_clusters, random_state)
    dists = np.linalg.norm(vectors - piecewise, axis=1)
    order = np.argsort(dists)
    return vectors[order[:n_best]]


def _project_and_cluster(data, vectors, n_clusters, random_state):
    """sklearn _project_and_cluster. data (n, d), vectors (d, n_best)."""
    from sklearn.cluster import KMeans as _SK_KMeans
    projected = data @ vectors  # (n, n_best)
    km = _SK_KMeans(n_clusters=n_clusters, n_init=10,
                    init="k-means++", random_state=random_state)
    return km.fit_predict(projected).astype(np.int64, copy=False)


@dataclass
class SpectralBiclusteringResult:
    """SpectralBiclustering fit result (Kluger 2003).

    row_labels_: (r,) int64
    column_labels_: (c,) int64
    rows_: (n_row_clusters*n_col_clusters, r) bool indicator
    columns_: (n_row_clusters*n_col_clusters, c) bool indicator
    """
    row_labels_: Optional[np.ndarray] = None
    column_labels_: Optional[np.ndarray] = None
    rows_: Optional[np.ndarray] = None
    columns_: Optional[np.ndarray] = None
    A_shape: tuple = ()
    method: str = "bistochastic"
    n_row_clusters: int = 0
    n_col_clusters: int = 0
    n_best: int = 3
    n_sv: int = 0
    X_cols: list = field(default_factory=list)

    def state_nbytes(self) -> int:
        if self.row_labels_ is None:
            return 0
        sz = 256
        sz += int(self.row_labels_.nbytes)
        sz += int(self.column_labels_.nbytes)
        if self.rows_ is not None:
            sz += int(self.rows_.nbytes)
        if self.columns_ is not None:
            sz += int(self.columns_.nbytes)
        return int(sz)

    def release_train_state(self) -> None:
        self.row_labels_ = None
        self.column_labels_ = None
        self.rows_ = None
        self.columns_ = None

    def to_dict(self) -> dict:
        return {
            "A_shape": tuple(self.A_shape),
            "method": self.method,
            "n_row_clusters": int(self.n_row_clusters),
            "n_col_clusters": int(self.n_col_clusters),
            "n_best": int(self.n_best),
            "n_sv": int(self.n_sv),
        }


def fit_spectral_biclustering(
    df,
    X_cols,
    *,
    n_clusters=3,
    method: str = "bistochastic",
    n_components: int = 6,
    n_best: int = 3,
    n_init: int = 10,
    random_state: Optional[int] = None,
    batch_size: Optional[int] = None,
    memory_guard=None,
) -> SpectralBiclusteringResult:
    """Spectral biclustering (Kluger 2003). Sklearn parity.

    Args:
      df: CadwDataFrame; A = df[X_cols], rows x cols bipartite matrix.
      X_cols: list of numeric columns; r = df.count(), c = len(X_cols).
      n_clusters: int or tuple (n_row, n_col). sklearn default 3.
      method: "bistochastic" | "scale" | "log".
      n_components: base SVD components (sklearn default 6).
      n_best: best singular vector count for clustering (default 3).
      n_init: kmeans n_init.
      random_state: for kmeans + (unused) svd_topk (exact).
      batch_size, memory_guard: standard.

    Returns SpectralBiclusteringResult.

    OOC:
      - svd_topk Pattern #1 (chunked Gram); bounded_by_window
        {gram_dim=min(r,c): 1e4}.
      - A materialized O(r*c); bounded_by_domain {r*c*8: 1e9}.
        sklearn _project_and_cluster intrinsic. Streaming plan
        v0.9.213+.
    """
    if not X_cols:
        raise ValueError("fit_spectral_biclustering: X_cols is empty")
    X_cols = list(X_cols)
    if isinstance(n_clusters, (list, tuple)):
        if len(n_clusters) != 2:
            raise ValueError(
                f"n_clusters tuple must have 2 elements; got {n_clusters}")
        n_row_clusters, n_col_clusters = int(n_clusters[0]), int(n_clusters[1])
    else:
        n_row_clusters = n_col_clusters = int(n_clusters)
    if n_row_clusters < 2 or n_col_clusters < 2:
        raise ValueError(
            f"n_clusters must be >= 2; got ({n_row_clusters}, "
            f"{n_col_clusters})")
    if method not in ("bistochastic", "scale", "log"):
        raise ValueError(
            f"method must be 'bistochastic'|'scale'|'log'; got {method!r}")
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1; got {n_components}")
    if n_best < 1:
        raise ValueError(f"n_best must be >= 1; got {n_best}")

    missing = [c for c in X_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"fit_spectral_biclustering: columns missing: {missing}")

    mg = memory_guard or getattr(df, "_memory_guard", None)
    r = int(df.count())
    c = len(X_cols)
    if r < 2 or c < 2:
        raise ValueError(f"need r>=2 and c>=2; got r={r}, c={c}")

    # A materialization bound: r*c*8 (A) + r*c*8 (A_norm) = 2*r*c*8
    A_bytes = r * c * 8
    if mg is not None and hasattr(mg, "check_state"):
        mg.check_state(2 * A_bytes + r * 8 + c * 8 + 256,
                       name="biclust.A_state")

    if batch_size is None:
        batch_size = max(1, min(r, 1000))
    batch_size = max(1, min(int(batch_size), r))

    # Pass 1: materialize A (r, c)
    A_parts = []
    for b in df._iter_filtered_batches(batch_size=batch_size):
        if mg is not None and hasattr(mg, "check_batch"):
            mg.check_batch(b, name="biclust.A", extra_factor=2.0)
        A_parts.append(_batch_to_X(b, X_cols))
    A = (np.vstack(A_parts) if len(A_parts) > 1
         else A_parts[0] if A_parts else np.empty((0, c)))
    if A.shape != (r, c):
        raise RuntimeError(f"A shape mismatch: {A.shape} != ({r},{c})")
    if np.isnan(A).any():
        raise ValueError(
            "fit_spectral_biclustering: A contains NaN; impute first")

    # Pass 2: normalize
    if method == "bistochastic":
        A_norm = _bistochastic_normalize(A)
        n_discard = 1
    elif method == "scale":
        A_norm, _, _ = _scale_normalize(A)
        n_discard = 1
    else:  # log
        A_norm = _log_normalize(A)
        n_discard = 0

    n_sv = n_components + (1 if method != "log" else 0)
    # svd_topk requires k <= min(r, c)
    if n_sv > min(r, c):
        raise ValueError(
            f"n_sv={n_sv} > min(r={r}, c={c}); reduce n_components")

    # Pass 3: SVD via svd_topk Pattern #1 (chunked Gram, exact)
    # A_norm may be non-integer/non-square; svd_topk(df) requires df
    # numeric. Wrap A_norm as df.
    from cadw.dataset.dataset import Dataset
    from cadw import CadwDataFrame
    a_schema = pa.schema([pa.field(f"a{j}", pa.float64())
                          for j in range(c)])

    def _iter_a_norm():
        for s in range(0, r, batch_size):
            e = min(s + batch_size, r)
            yield pa.RecordBatch.from_arrays(
                [pa.array(A_norm[s:e, j], pa.float64()) for j in range(c)],
                schema=a_schema)

    a_ds = Dataset.from_batches(a_schema, _iter_a_norm(),
                                 df._new_dataset_dir())
    a_df = CadwDataFrame(a_ds, memory_guard=mg,
                          spill_dir=getattr(df, "_spill_dir", None))

    U, sigma, V = svd_topk(
        a_df, [f"a{j}" for j in range(c)], n_sv,
        compute_u=True, batch_size=batch_size, memory_guard=mg)

    # Collect U (r, n_sv) from AutoArrayResult
    u_parts = []
    for chunk in U:
        arr = np.asarray(chunk)
        if arr.ndim == 1:
            arr = arr[:, None]
        u_parts.append(arr)
    U_arr = np.vstack(u_parts) if u_parts else np.empty((0, n_sv))
    if U_arr.shape[0] != r:
        raise RuntimeError(f"U rows mismatch: {U_arr.shape[0]} != {r}")

    # Discard (n_discard=1 for bistochastic/scale; 0 for log)
    U_red = U_arr[:, n_discard:]  # (r, k_used)
    V_red = V[:, n_discard:]      # (c, k_used)
    k_used = U_red.shape[1]

    # svd_flip parity with sklearn (u_based_decision=False):
    # per-column argmax sign
    for j in range(k_used):
        col_v = V_red[:, j]
        idx = int(np.argmax(np.abs(col_v)))
        if col_v[idx] < 0:
            V_red[:, j] *= -1
            U_red[:, j] *= -1

    ut = U_red.T  # (k_used, r)
    vt = V_red.T  # (k_used, c)

    # Pass 4: piecewise selection + project+cluster (sklearn parity)
    best_ut = _fit_best_piecewise(ut, n_best, n_row_clusters, random_state)
    best_vt = _fit_best_piecewise(vt, n_best, n_col_clusters, random_state)

    row_labels = _project_and_cluster(A, best_vt.T, n_row_clusters,
                                        random_state)
    col_labels = _project_and_cluster(A.T, best_ut.T, n_col_clusters,
                                        random_state)

    # Indicator matrices (sklearn rows_/columns_)
    rows_ind = np.vstack([
        row_labels == label
        for label in range(n_row_clusters)
        for _ in range(n_col_clusters)
    ])
    columns_ind = np.vstack([
        col_labels == label
        for _ in range(n_row_clusters)
        for label in range(n_col_clusters)
    ])

    result = SpectralBiclusteringResult(
        row_labels_=row_labels,
        column_labels_=col_labels,
        rows_=rows_ind,
        columns_=columns_ind,
        A_shape=(r, c),
        method=method,
        n_row_clusters=n_row_clusters,
        n_col_clusters=n_col_clusters,
        n_best=int(n_best),
        n_sv=int(n_sv),
        X_cols=X_cols,
    )

    try:
        log_fit("spectral_biclustering",
                state_bytes=result.state_nbytes())
    except Exception:
        pass

    return result
