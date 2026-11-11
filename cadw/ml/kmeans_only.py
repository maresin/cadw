"""KMeansOnly baseline (B6 consensus baseline).

Single-run KMeans baseline — what consensus must beat. Thin wrapper
over fit_kmeans (verified). No cross-algorithm agreement; just the
plain KMeans result.

OOC (Pattern #1 batch-layer, axis=n):
  state = O(k*d) (centroids + inertia); independent of n.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from cadw.core.memory_guard import MemoryGuard
from cadw.ml import fit_kmeans


@dataclass
class KMeansOnlyResult:
    """KMeansOnly baseline result.

    labels_: (n,) int64 (via fit_kmeans.predict)
    cluster_centers_: (k, d) f64
    k: int
    X_cols: list
    """
    labels_: Optional[np.ndarray] = None
    cluster_centers_: Optional[np.ndarray] = None
    k: int = 0
    X_cols: list = field(default_factory=list)
    n_train_: int = 0

    def state_nbytes(self) -> int:
        sz = 256
        if self.cluster_centers_ is not None:
            sz += int(self.cluster_centers_.nbytes)
        if self.labels_ is not None:
            sz += int(self.labels_.nbytes)
        return int(sz)

    def to_dict(self) -> dict:
        return {"k": int(self.k), "n_train": int(self.n_train_)}


def fit_kmeans_only(
    df,
    X_cols,
    *,
    k: int = 8,
    n_init: int = 10,
    max_iter: int = 300,
    tol: float = 1e-4,
    seed: int = 42,
    batch_size: Optional[int] = None,
    memory_guard: Optional[MemoryGuard] = None,
) -> KMeansOnlyResult:
    """KMeansOnly baseline (thin wrapper over fit_kmeans).

    OOC: state = O(k*d) via fit_kmeans (Pattern #1).
    """
    if k < 1:
        raise ValueError(f"fit_kmeans_only: k must be >= 1; got {k}")
    if not X_cols:
        raise ValueError("fit_kmeans_only: X_cols is empty")

    mg = memory_guard or getattr(df, "_memory_guard", None)
    km = fit_kmeans(df, X_cols, k=k, n_init=n_init, max_iter=max_iter,
                     tol=tol, seed=seed, batch_size=batch_size,
                     memory_guard=mg)

    # Extract labels via chunked predict
    labels_parts = []
    for b in df._iter_filtered_batches(batch_size=batch_size):
        X_b = np.column_stack([
            b.column(c).to_numpy(zero_copy_only=False).astype(
                np.float64, copy=False)
            for c in X_cols
        ])
        labels_parts.append(km.predict(X_b))
    labels = (np.concatenate(labels_parts) if labels_parts
              else np.empty(0, dtype=np.int64))

    return KMeansOnlyResult(
        labels_=labels,
        cluster_centers_=km.cluster_centers_.copy(),
        k=int(k),
        X_cols=list(X_cols),
        n_train_=int(df.count()),
    )
