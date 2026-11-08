"""Sampling primitives (v0.9.143-batch5, DEC-058).

Reservoir sampling (Algorithm R/L) + feature extraction helper.
No I/O; numpy + Arrow batches.

History:
- v0.9.143-batch5: created. Extracted _batch_to_X + _reservoir_k from
  cadw.ml.kernel_density (inlined v0.9.143-batch4b). Consumers:
  KernelDensity + ReservoirSample (2 -> P2 extraction, AP10 rule).
"""
from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np
import pyarrow as pa


def batch_to_features(
    batch: pa.RecordBatch, feature_cols: Sequence[str],
) -> np.ndarray:
    """Convert feature columns to (n, d) float64, dropping NaN rows."""
    cols = []
    for c in feature_cols:
        arr = batch.column(c).to_numpy(zero_copy_only=False)
        cols.append(np.asarray(arr, dtype=np.float64))
    X = np.column_stack(cols)
    mask = ~np.isnan(X).any(axis=1)
    return X[mask]


def reservoir_sample(
    batch_iter: Iterator[pa.RecordBatch],
    feature_cols: Sequence[str],
    k: int,
    *,
    seed: int = 0,
    memory_guard=None,
) -> tuple[np.ndarray, int]:
    """Algorithm R reservoir sample of k rows, vectorized per batch.

    Returns (reservoir_ndarray, n_seen):
      reservoir: (k', d) float64, k' = min(k, n_seen)
      n_seen:    total rows seen (before NaN filtering)

    Guard: state (k*d*f64) checked before np.empty (AI_CONSTITUTION §3.1).
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    rng = np.random.default_rng(seed)
    reservoir = None
    n_seen = 0
    for batch in batch_iter:
        X = batch_to_features(batch, feature_cols)
        if X.shape[0] == 0:
            continue
        b, d = X.shape
        if reservoir is None:
            if memory_guard is not None:
                memory_guard.check_state(
                    k * d * 8, name="reservoir_sample.init")
            reservoir = np.empty((k, d), dtype=np.float64)
        take = min(k - n_seen, b)
        if take > 0:
            reservoir[n_seen:n_seen + take] = X[:take]
            n_seen += take
            X = X[take:]
            b = X.shape[0]
        if b == 0:
            continue
        stream_idx = np.arange(n_seen + 1, n_seen + b + 1, dtype=np.float64)
        probs = k / stream_idx
        accept = rng.random(b) < probs
        n_acc = int(accept.sum())
        if n_acc > 0:
            slots = rng.integers(0, k, size=n_acc)
            reservoir[slots] = X[accept]
        n_seen += b
    if reservoir is None or n_seen == 0:
        raise ValueError("reservoir_sample: empty input")
    n_kept = min(n_seen, k)
    return reservoir[:n_kept].copy(), n_seen
