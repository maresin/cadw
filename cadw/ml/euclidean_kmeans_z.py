"""EuclideanKMeansZ (shape-based clustering, z-normalized).

KMeans on per-row z-normalized series with Euclidean distance.
Shape-aware baseline; ignores temporal shifts (unlike KShape which
uses SBD / cross-correlation).

OOC (Pattern #1 batch-layer, axis=n):
  z-normalize per row (in-memory transform per batch); fit_kmeans state
  O(k*d). Independent of n.

Reference: Paparrizos & Gravano 2015 (KShape paper §2.3; Euclidean z-norm
baseline for comparison).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.ml import fit_kmeans
from cadw.ops.stats import znormalize_rows


@dataclass
class EuclideanKMeansZResult:
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


def _batch_to_X(b, X_cols):
    arrs = []
    for c in X_cols:
        col = b.column(c)
        if not pa.types.is_floating(col.type):
            col = col.cast(pa.float64())
        arrs.append(col.to_numpy(zero_copy_only=False).astype(
            np.float64, copy=False))
    return np.column_stack(arrs) if arrs else np.empty((b.num_rows, 0))


def fit_euclidean_kmeans_z(
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
) -> EuclideanKMeansZResult:
    """EuclideanKMeansZ (KMeans on per-row z-normalized series).

    OOC: z-normalize per batch (O(batch*d)); fit_kmeans state O(k*d).
    """
    if k < 1:
        raise ValueError(
            f"fit_euclidean_kmeans_z: k must be >= 1; got {k}")
    if not X_cols:
        raise ValueError("fit_euclidean_kmeans_z: X_cols is empty")
    missing = [c for c in X_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"fit_euclidean_kmeans_z: columns missing: {missing}")

    mg = memory_guard or getattr(df, "_memory_guard", None)
    if batch_size is None:
        batch_size = max(1, min(int(df.count()), 1000))
    batch_size = max(1, min(int(batch_size), int(df.count())))

    # Materialize z-normalized X (state = n*d f64). For true OOC, this
    # would need SpillArray; but fit_kmeans needs df API. Note: n*d f64
    # at n=1e6, d=100 = 800 MB — bounded_by_domain {n*d: 1e8}.
    zn_parts = []
    for b in df._iter_filtered_batches(batch_size=batch_size):
        X_b = _batch_to_X(b, X_cols)
        zn_parts.append(znormalize_rows(X_b))
    X_zn = (np.vstack(zn_parts) if len(zn_parts) > 1
            else zn_parts[0] if zn_parts else np.empty((0, len(X_cols))))
    n = X_zn.shape[0]
    d = X_zn.shape[1]
    if n * d > 100_000_000:
        from cadw.core.memory_guard import MemoryLimitExceeded
        raise MemoryLimitExceeded(
            f"fit_euclidean_kmeans_z: n*d={n*d} > 1e8 (z-normalized X "
            f"materialized for fit_kmeans df API). Options: (a) reduce d; "
            f"(b) subsample n; (c) reduce batch_size — but X still fully "
            f"materialized.",
            model="EuclideanKMeansZ", params={"n": n, "d": d})

    # Build df for fit_kmeans
    zn_table = pa.table({
        f"zn_{j}": pa.array(X_zn[:, j], type=pa.float64())
        for j in range(d)
    })
    from cadw import CadwDataFrame
    zn_df = CadwDataFrame.from_table(
        zn_table, memory_guard=mg,
        spill_dir=getattr(df, "_spill_dir", None))

    km = fit_kmeans(zn_df, [f"zn_{j}" for j in range(d)], k=k,
                     n_init=n_init, max_iter=max_iter, tol=tol, seed=seed,
                     batch_size=batch_size, memory_guard=mg)

    # Extract labels via chunked predict
    labels_parts = []
    for b in zn_df._iter_filtered_batches(batch_size=batch_size):
        X_b = np.column_stack([
            b.column(f"zn_{j}").to_numpy(zero_copy_only=False)
            for j in range(d)
        ])
        labels_parts.append(km.predict(X_b))
    labels = (np.concatenate(labels_parts) if labels_parts
              else np.empty(0, dtype=np.int64))

    return EuclideanKMeansZResult(
        labels_=labels,
        cluster_centers_=km.cluster_centers_.copy(),
        k=int(k),
        X_cols=list(X_cols),
        n_train_=int(n),
    )
