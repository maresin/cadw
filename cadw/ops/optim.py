"""Streaming optimization primitives (v0.9.89).

CADW-native streaming versions of classical training loops: state is
O(d * K) bytes (parameters + accumulators), independent of n.

Contrast with closed-form solutions (e.g. ridge via XTX = O(d^2)),
which force an arbitrary memory-driven cutoff on feature dimension d.
For wide-feature models (ROCKET with n_kernels=10000 → d=20000 →
XTX=3.2 GB), closed-form is a base-case violation
(METHODOLOGY_ALGORITHM_SELECTION.md §1.1).

See docs/ml_tools/METHODOLOGY_ALGORITHM_SELECTION.md §5 (planned
pattern 4: streaming SGD).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from cadw.core.memory_guard import MemoryGuard


def sgd_streaming(
    batch_stream,
    n_features: int,
    n_outputs: int,
    *,
    lam: float = 1.0,
    lr: float = 0.01,
    epochs: int = 30,
    beta: float = 0.9,
    eps: float = 1e-8,
    mg: Optional[MemoryGuard] = None,
    name: str = "sgd",
) -> np.ndarray:
    """Streaming SGD + RMSprop on ridge loss. Returns W (n_outputs, d).

    Parameters
    ----------
    batch_stream : callable() -> iterator of (X_b, Y_b)
        X_b: (b, d) float64; Y_b: (b, K) float64 (one-hot or ±1).
        Called once for warm start, then `epochs` times for gradient
        steps. Each call must produce a fresh iterator.
    n_features : int
        Feature dimension d (user-chosen hyperparameter of the caller,
        NOT derived from n). SGD state is O(d), independent of d².
    n_outputs : int
        1 for binary (±1 target), K for one-vs-rest multiclass.

    Loss
    ----
    0.5 * mean((X W^T - Y)^2) + 0.5 * lam * ||W||^2

    Update
    ------
    pred = X_b @ W^T                           (b, K)
    grad = (pred - Y_b).T @ X_b / b + lam * W  (K, d)
    h    = beta * h + (1-beta) * grad**2       (K, d)  [RMSprop]
    W   -= lr * grad / (sqrt(h) + eps)

    State
    -----
    W (K, d) + h (K, d) = 2 * K * d * 8 bytes.
    For d=20000, K=1: 320 KB. Independent of n and n_series.

    Determinism
    -----------
    Batch order in batch_stream determines SGD order. Same batch_stream
    → bit-identical result across runs (no random seed involved).
    """
    d = n_features
    W = np.zeros((n_outputs, d), dtype=np.float64)
    h = np.zeros((n_outputs, d), dtype=np.float64)
    if mg is not None:
        # Classifier state: W + h + small numpy overhead.
        mg.check_state(2 * d * n_outputs * 8 + d * 8,
                       name=name + ".classifier.state")

    # Warm start: W_0 ≈ (1/n) X^T Y (one gradient step at W=0, lr=1).
    n_seen = 0
    for X_b, Y_b in batch_stream():
        if mg is not None:
            # Intermediate: Y_b.T @ X_b → (K, d) temp; account for that
            # plus X_b read view. check_state (raw byte count), not
            # check_batch (which expects a pa.RecordBatch).
            mg.check_state(len(X_b) * d * 8 + n_outputs * d * 8,
                           name=name + ".warm")
        # X_b: (b, d), Y_b: (b, K). Y_b.T @ X_b = (K, d). Matches W shape.
        W += Y_b.T @ X_b
        n_seen += X_b.shape[0]
    if n_seen == 0:
        raise ValueError(f"{name}: empty feature stream")
    W /= float(n_seen)

    for _epoch in range(epochs):
        for X_b, Y_b in batch_stream():
            if mg is not None:
                # Transients: pred (b,K), grad (K,d), h update (K,d),
                # masked (K,d). Estimate 3 * (K*d + b*K) * 8 bytes.
                mg.check_state(3 * (n_outputs * d + len(X_b) * n_outputs) * 8,
                               name=name + ".batch")
            pred = X_b @ W.T                           # (b, K)
            grad = ((pred - Y_b).T @ X_b) / max(len(X_b), 1) + lam * W
            h = beta * h + (1.0 - beta) * grad * grad
            W -= lr * grad / (np.sqrt(h) + eps)
    return W
