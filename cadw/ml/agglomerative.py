"""Agglomerative hierarchical clustering (bottom-up).

Sklearn parity: sklearn.cluster.AgglomerativeClustering for all four
linkages (ward, complete, average, single) via scipy.cluster.hierarchy
dense linkage.

OOC (Pattern #3 intrinsic — dense linkage O(n^2)):
  - All linkages: scipy dense linkage matrix O(n^2).
  - refuse_at_guard {n: 5000}.
  - Approximate single-linkage via k-NN graph + MST deferred to
    v0.9.216+ (opt-in mode='approx'; HDBSCAN-style approximation,
    loses exact sklearn parity for high k_neighbors).

Honest status: debt (dense O(n^2) intrinsic).

Reference: Ward 1963; scipy.cluster.hierarchy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ml._logging import log_fit


_MAX_N_DENSE = 5000  # scipy dense linkage O(n^2)


def _batch_to_X(b, X_cols):
    arrs = []
    for c in X_cols:
        col = b.column(c)
        if not pa.types.is_floating(col.type) and \
           not pa.types.is_integer(col.type):
            col = col.cast(pa.float64())
        arrs.append(col.to_numpy(zero_copy_only=False).astype(
            np.float64, copy=False))
    return np.column_stack(arrs) if arrs else np.empty((b.num_rows, 0))


@dataclass
class AgglomerativeResult:
    labels_: Optional[np.ndarray] = None
    n_clusters_: int = 0
    linkage_: str = "ward"
    n_train_: int = 0
    X_cols: list = field(default_factory=list)

    def state_nbytes(self) -> int:
        sz = 256
        if self.labels_ is not None:
            sz += int(self.labels_.nbytes)
        return int(sz)

    def to_dict(self) -> dict:
        return {
            "n_clusters": int(self.n_clusters_),
            "linkage": self.linkage_,
            "n_train": int(self.n_train_),
        }


def fit_agglomerative(
    df,
    X_cols,
    *,
    n_clusters: int = 2,
    linkage: str = "ward",
    metric: str = "euclidean",
    batch_size: Optional[int] = None,
    memory_guard: Optional[MemoryGuard] = None,
) -> AgglomerativeResult:
    """Agglomerative hierarchical clustering (dense scipy linkage).

    Args:
      df: CadwDataFrame with numeric X_cols.
      X_cols: list[str].
      n_clusters: target cluster count (>= 2).
      linkage: 'single' | 'ward' | 'complete' | 'average'.
      metric: 'euclidean' only (others deferred).
      batch_size, memory_guard: standard.

    Returns AgglomerativeResult with labels_.

    OOC (Pattern #3 intrinsic — dense linkage O(n^2)):
      - refuse_at_guard {n: 5000}.
      - Approximate single-linkage via k-NN graph + MST v0.9.216+.
    """
    if not X_cols:
        raise ValueError("fit_agglomerative: X_cols is empty")
    X_cols = list(X_cols)
    if n_clusters < 2:
        raise ValueError(
            f"fit_agglomerative: n_clusters must be >= 2; got {n_clusters}")
    if linkage not in ("single", "ward", "complete", "average"):
        raise ValueError(
            f"fit_agglomerative: linkage must be single|ward|complete|"
            f"average; got {linkage!r}")
    if metric != "euclidean":
        raise ValueError(
            f"fit_agglomerative: metric 'euclidean' only (others deferred); "
            f"got {metric!r}")
    missing = [c for c in X_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"fit_agglomerative: columns missing: {missing}")

    mg = memory_guard or getattr(df, "_memory_guard", None)
    n = int(df.count())
    d = len(X_cols)
    if n < 1:
        raise ValueError("fit_agglomerative: empty df")
    if n <= n_clusters:
        raise ValueError(
            f"fit_agglomerative: n={n} must be > n_clusters={n_clusters}")

    if n > _MAX_N_DENSE:
        raise MemoryLimitExceeded(
            f"fit_agglomerative: n={n} > {_MAX_N_DENSE} (dense scipy "
            f"linkage O(n^2) intrinsic). Options: (a) subsample df; "
            f"(b) use MiniBatchKMeans (approximate); "
            f"(c) wait for v0.9.216+ approximate single-linkage "
            f"(k-NN graph + MST, opt-in mode='approx').",
            model="AgglomerativeClustering",
            params={"n": n, "linkage": linkage})

    if batch_size is None:
        batch_size = max(1, min(n, 1000))
    batch_size = max(1, min(int(batch_size), n))

    # State: X (n*d) + linkage (n^2)
    state_est = n * d * 8 + n * n * 8 + 256
    if mg is not None and hasattr(mg, "check_state"):
        mg.check_state(state_est, name="agglomerative.state")

    # Materialize X (dense)
    X_parts = []
    for b in df._iter_filtered_batches(batch_size=batch_size):
        if mg is not None and hasattr(mg, "check_batch"):
            mg.check_batch(b, name="agglomerative.X", extra_factor=2.0)
        X_parts.append(_batch_to_X(b, X_cols))
    X = (np.vstack(X_parts) if len(X_parts) > 1
         else X_parts[0] if X_parts else np.empty((0, d)))
    if np.isnan(X).any():
        raise ValueError(
            "fit_agglomerative: X contains NaN; impute first")

    from scipy.cluster.hierarchy import linkage as _scipy_linkage
    from scipy.cluster.hierarchy import fcluster as _scipy_fcluster
    Z = _scipy_linkage(X, method=linkage, metric=metric)
    labels_arr = _scipy_fcluster(Z, t=n_clusters, criterion="maxclust")
    # scipy returns 1-based labels; convert to 0-based
    labels_arr = labels_arr.astype(np.int64) - 1

    n_clusters_actual = int(len(np.unique(labels_arr)))

    result = AgglomerativeResult(
        labels_=labels_arr,
        n_clusters_=n_clusters_actual,
        linkage_=linkage,
        n_train_=int(n),
        X_cols=list(X_cols),
    )

    try:
        log_fit("agglomerative", state_bytes=result.state_nbytes())
    except Exception:
        pass

    return result
