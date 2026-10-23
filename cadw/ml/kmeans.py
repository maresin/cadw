"""k-means clustering (v0.7.1).

Composition: iter-Reduce (assign + accumulate) + update.
Peak RAM = batch + O(k×d). Independent of N.

Algorithm:
    1. k-means++ init from first sample batch.
    2. Repeat until shift < tol or max_iter:
        a. Reduce: assign each row to nearest centroid; accumulate
           sums (k×d), counts (k), inertia = Σ ||x - c||².
        b. Update: centroids = sums / counts (empty → keep old).

    shift = max ||centroid_new - centroid_old||_inf.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.ops.reduce import ReduceOp, reduce


class KMeansStepOp(ReduceOp):
    """One Reduce pass: assign rows to nearest centroid + accumulate.

    State: sums (k, d), counts (k), inertia (float). O(k×d).
    Centroid-array passed from previous iteration.
    Nulls in any feature → row skipped.
    """

    def __init__(
        self,
        feature_cols: Sequence[str],
        centroids: np.ndarray,
    ) -> None:
        if not feature_cols:
            raise ValueError("KMeansStepOp: feature_cols must be non-empty")
        self.feature_cols = list(feature_cols)
        self.centroids = np.asarray(centroids, dtype=np.float64)
        if self.centroids.ndim != 2:
            raise ValueError(
                f"KMeansStepOp: centroids must be 2D; "
                f"got shape {self.centroids.shape}"
            )
        self.k, self.d = self.centroids.shape

    def batch_memory_factor(self) -> float:
        """v0.9.22 (§13.4/243, AUDIT_ML §1.4): update() creates two
        (n, k, d) float64 tensors (diff then diff*diff) — 2·k·
        batch.nbytes transient. Prior inherited 1.0 → k=100, d=50,
        batch=10k: real peak ~800 MB vs check_batch 4 MB. Override
        to 2k. Output identity unchanged (guard-only edit)."""
        return 2.0 * float(self.k)

    def init(self):
        return (
            np.zeros((self.k, self.d), dtype=np.float64),
            np.zeros(self.k, dtype=np.int64),
            0.0,
        )

    def update(self, state, batch):
        sums, counts, inertia = state
        X = _stack_numeric(batch, self.feature_cols)
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        if X.shape[0] == 0:
            return state
        # Squared distances (n, k)
        # (n, k) = sum over d of (x - c)²
        diff = X[:, None, :] - self.centroids[None, :, :]
        d2 = (diff * diff).sum(axis=2)
        labels = d2.argmin(axis=1)
        inertia += float(d2[np.arange(X.shape[0]), labels].sum())
        # Accumulate per-cluster sums / counts
        for j in range(self.k):
            m = labels == j
            if m.any():
                sums[j] += X[m].sum(axis=0)
                counts[j] += int(m.sum())
        return (sums, counts, inertia)

    def merge(self, a, b):
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

    def state_nbytes(self, state):
        sums, counts, _ = state
        return int(sums.nbytes + counts.nbytes + 8)

    def estimate_state_nbytes(self):
        k, d = self.k, self.d
        return 8 * k * d + 8 * k + 8

    def finalize(self, state):
        sums, counts, inertia = state
        return {
            "sums": sums, "counts": counts, "inertia": float(inertia),
        }


def _stack_numeric(
    batch: pa.RecordBatch, cols: Sequence[str],
) -> np.ndarray:
    parts = [_column_to_f64(batch.column(c)) for c in cols]
    return np.column_stack(parts)


def _column_to_f64(arr: pa.Array) -> np.ndarray:
    np_arr = arr.to_numpy(zero_copy_only=False)
    if np_arr.dtype == object:
        out = np.empty(len(np_arr), dtype=np.float64)
        for i, v in enumerate(np_arr):
            out[i] = np.nan if v is None else float(v)
        return out
    return np_arr.astype(np.float64, copy=False)


def _kmeans_plus_plus_init(
    df, X_cols: Sequence[str], k: int, seed: int,
) -> np.ndarray:
    """k-means++ init from first sample batch. Deterministic with seed."""
    rng = np.random.default_rng(seed)
    # Grab first batch (small enough; if df huge — still 1 batch = bounded)
    sample: np.ndarray | None = None
    for b in df._iter_filtered_batches():
        X = _stack_numeric(b, X_cols)
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        if X.shape[0] > 0:
            sample = X
            break
    if sample is None or sample.shape[0] == 0:
        raise ValueError("fit_kmeans: no valid (non-null) rows in df")

    n, d = sample.shape
    if n < k:
        raise ValueError(
            f"fit_kmeans: k={k} > available rows={n}"
        )

    # First centroid: random
    centers = np.empty((k, d), dtype=np.float64)
    first = rng.integers(0, n)
    centers[0] = sample[first]

    # Subsequent: weighted by D²
    closest_sq = ((sample - centers[0]) ** 2).sum(axis=1)
    for j in range(1, k):
        probs = closest_sq / closest_sq.sum() if closest_sq.sum() > 0 else None
        idx = rng.choice(n, p=probs) if probs is not None else rng.integers(0, n)
        centers[j] = sample[idx]
        new_sq = ((sample - centers[j]) ** 2).sum(axis=1)
        closest_sq = np.minimum(closest_sq, new_sq)
    return centers


@dataclass
class KMeansModel:
    """Fitted k-means model."""

    cluster_centers_: np.ndarray  # (k, d)
    feature_names_: list[str]
    inertia_: float
    n_iter_: int
    converged_: bool
    n_rows_: int
    fitted_: bool = True

    @property
    def k(self) -> int:
        return self.cluster_centers_.shape[0]

    def to_dict(self) -> dict:
        return {
            "cluster_centers": self.cluster_centers_.tolist(),
            "feature_names": self.feature_names_,
            "inertia": float(self.inertia_),
            "n_iter": int(self.n_iter_),
            "converged": bool(self.converged_),
            "n_rows": int(self.n_rows_),
        }

    def _labels_for_array(self, X: np.ndarray) -> np.ndarray:
        diff = X[:, None, :] - self.cluster_centers_[None, :, :]
        d2 = (diff * diff).sum(axis=2)
        return d2.argmin(axis=1)

    def predict(self, X):
        """Predict cluster labels.

        - np.ndarray (n, d) → np.ndarray (n,) int64
        - CadwDataFrame → CadwDataFrame with `cluster` column (int64)
        """
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            if X.shape[1] != len(self.feature_names_):
                raise ValueError(
                    f"predict: X has {X.shape[1]} features, "
                    f"expected {len(self.feature_names_)}"
                )
            return self._labels_for_array(X).astype(np.int64)

        from cadw.dataframe import CadwDataFrame
        from cadw.dataset import Dataset
        if not isinstance(X, CadwDataFrame):
            raise TypeError(
                f"predict: X must be np.ndarray or CadwDataFrame; "
                f"got {type(X).__name__}"
            )
        for c in self.feature_names_:
            if c not in X.columns:
                raise KeyError(f"predict: column {c!r} not in df")

        out_col = "cluster"
        orig_schema = X._dataset.schema
        out_fields = [orig_schema.field(c) for c in X._columns]
        out_fields.append(pa.field(out_col, pa.int64(), nullable=True))
        out_schema = pa.schema(out_fields)

        centers = self.cluster_centers_

        def _iter_pred():
            for b in X._iter_filtered_batches():
                Xm = _stack_numeric(b, self.feature_names_)
                n = Xm.shape[0]
                labels = np.full(n, -1, dtype=np.int64)  # -1 = null
                mask = ~np.isnan(Xm).any(axis=1)
                if mask.any():
                    Xv = Xm[mask]
                    diff = Xv[:, None, :] - centers[None, :, :]
                    d2 = (diff * diff).sum(axis=2)
                    labels[mask] = d2.argmin(axis=1)
                arrays = [b.column(c) for c in X._columns]
                # -1 → null
                arr = pa.array(
                    [None if v < 0 else int(v) for v in labels],
                    type=pa.int64(),
                )
                arrays.append(arr)
                yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

        new_ds = Dataset.from_batches(
            out_schema, _iter_pred(), X._new_dataset_dir()
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=X._memory_guard,
            spill_dir=X._spill_dir,
        )


def fit_kmeans(
    df,
    X_cols,
    k: int,
    *,
    n_init: int = 3,
    max_iter: int = 300,
    tol: float = 1e-4,
    seed: int = 42,
    memory_guard: Optional[MemoryGuard] = None,
) -> KMeansModel:
    """Fit k-means clustering via iter-Reduce.

    - X_cols: numeric feature columns.
    - k: number of clusters.
    - n_init: number of random inits (k-means++ seeded). Best inertia kept.
    - max_iter, tol: stopping criteria.
    - seed: for reproducibility.
    - memory_guard: default df._memory_guard.

    Peak RAM = batch + O(k×d). Independent of N.

    Raises:
        ValueError: empty X, k < 1, k > valid rows, all-null data.
        KeyError: missing column.
        TypeError: non-numeric column.
    """
    if isinstance(X_cols, str):
        X_cols = [X_cols]
    else:
        X_cols = list(X_cols)
    if not X_cols:
        raise ValueError("fit_kmeans: X_cols must be non-empty")
    if not isinstance(k, int):
        raise TypeError(f"fit_kmeans: k must be int; got {type(k).__name__}")
    if k < 1:
        raise ValueError(f"fit_kmeans: k must be >= 1; got {k}")
    if n_init < 1:
        raise ValueError(f"fit_kmeans: n_init must be >= 1; got {n_init}")
    if max_iter < 1:
        raise ValueError(f"fit_kmeans: max_iter must be >= 1; got {max_iter}")
    if tol <= 0:
        raise ValueError(f"fit_kmeans: tol must be > 0; got {tol}")
    for c in X_cols:
        if c not in df.columns:
            raise KeyError(f"fit_kmeans: column {c!r} not in df")

    schema = df._dataset.schema
    for c in X_cols:
        t = schema.field(c).type
        if not (pa.types.is_integer(t) or pa.types.is_floating(t)):
            raise TypeError(
                f"fit_kmeans: column {c!r} has type {t}; "
                f"numeric required (use encode_categorical for strings)"
            )

    mg = memory_guard or df._memory_guard
    d = len(X_cols)

    best: Optional[dict] = None
    for init_idx in range(n_init):
        init_seed = seed + init_idx
        centers = _kmeans_plus_plus_init(df, X_cols, k, init_seed)
        n_iter = 0
        converged = False
        inertia = float("inf")
        n_valid = 0

        for it in range(max_iter):
            n_iter = it + 1
            op = KMeansStepOp(X_cols, centers)
            res = reduce(df._iter_filtered_batches(), op, mg, log=False)
            n_valid = res["counts"].sum()
            if n_valid == 0:
                raise ValueError("fit_kmeans: no valid (non-null) rows")
            sums = res["sums"]
            counts = res["counts"]
            inertia = res["inertia"]

            # Update: mean per cluster; empty cluster keeps old
            new_centers = centers.copy()
            for j in range(k):
                if counts[j] > 0:
                    new_centers[j] = sums[j] / counts[j]

            shift = float(np.max(np.abs(new_centers - centers)))
            centers = new_centers

            # v0.9.31: uniform training log (opt-in via CADW_LOG=1).
            try:
                from cadw.ml._logging import log_fit
                log_fit("kmeans", iter_i=n_iter, iter_n=max_iter,
                        loss=float(inertia), state_bytes=int(k*centers.shape[1]*8))
            except Exception:
                pass

            if shift < tol:
                converged = True
                break

        if best is None or inertia < best["inertia"]:
            best = {
                "centers": centers.copy(),
                "inertia": float(inertia),
                "n_iter": int(n_iter),
                "converged": bool(converged),
                "n_rows": int(n_valid),
            }

    assert best is not None
    return KMeansModel(
        cluster_centers_=best["centers"],
        feature_names_=list(X_cols),
        inertia_=best["inertia"],
        n_iter_=best["n_iter"],
        converged_=best["converged"],
        n_rows_=best["n_rows"],
    )
