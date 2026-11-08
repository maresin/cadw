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
from cadw.ops.numeric import sigmoid




def sgd_streaming(
    batch_stream,
    n_features: int,
    n_outputs: int,
    *,
    loss: str = "ridge",
    lam: float = 1.0,
    lr: float = 0.01,
    epochs: int = 30,
    beta: float = 0.9,
    eps: float = 1e-8,
    penalty_mask: Optional[np.ndarray] = None,
    use_rmsprop: bool = True,
    mg: Optional[MemoryGuard] = None,
    name: str = "sgd",
) -> np.ndarray:
    """Streaming SGD + RMSprop. Supports ridge and logistic loss.

    Returns W (n_outputs, d).

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
    loss="ridge" (default):
        0.5 * mean((X W^T - Y)^2) + 0.5 * lam * ||W||^2
    loss="logistic":
        mean(-[Y * log(sigmoid(X W^T))
              + (1-Y) * log(1 - sigmoid(X W^T))])
        + 0.5 * lam * ||W||^2
        (Y_b expected in {0, 1}; for ±1 targets, caller transforms
        to 0/1 before passing to batch_stream.)

    Update (gradient-only difference; optimizer identical)
    ------
    z    = X_b @ W^T                           (b, K)
    pred = z (ridge) | sigmoid(z) (logistic)
    grad = (pred - Y_b).T @ X_b / b + lam * W  (K, d)
    h    = beta * h + (1-beta) * grad**2       (K, d)  [RMSprop]
    W   -= lr * grad / (sqrt(h) + eps)

    Warm start
    ----------
    ridge: W_0 = mean(Y^T X) over stream (one gradient step at W=0).
    logistic: W_0 = 0 (standard; mean(Y^T X) would saturate sigmoid
      for ±1 targets and 0/1 targets where class imbalance is high).

    Caller responsibility for lam
    -----------------------------
    To match a closed-form objective `sum_loss + α·||W||²`:
      ridge:    lam = α / n
      logistic: lam = α / n
    (Derivation: f/n where f = -sum_loss + α/2·||W||² gives
    -mean_loss + α/(2n)·||W||². SGD's lam/2·||W||² matches when
    lam = α/n in both cases. Earlier note "2·α/n" was incorrect
    — the factor-2 in logistic's objective is already folded into
    α/2 penalty scale; linear Ridge's alpha is on sum_sq not
    sum_sq/2, so both reduce to lam = α/n.)

    penalty_mask : np.ndarray, optional
        Shape (n_outputs, n_features) or (1, n_features). Multiplicative
        mask on the L2 penalty: grad += lam * W * penalty_mask.
        Use to exempt intercept (mask[:, 0] = 0). Default None → all
        parameters penalized.

    use_rmsprop : bool, default True
        True: RMSprop (h accumulator, per-param adaptive lr). Good for
        ridge (quadratic, well-scaled gradients) and default rocket.
        False: plain GD W -= lr*grad. Required for strongly-regularized
        logistic (RMSprop oscillates at amplitude ~lr when penalty
        dominates gradient; plain GD contracts geometrically at rate
        1 - lr*lam). See _fit_sgd_logistic.

    State
    -----
    W (K, d) + h (K, d) = 2 * K * d * 8 bytes.
    For d=20000, K=1: 320 KB. Independent of n and n_series.

    Determinism
    -----------
    Batch order in batch_stream determines SGD order. Same batch_stream
    → bit-identical result across runs (no random seed involved).
    """
    if loss not in ("ridge", "logistic"):
        raise ValueError(
            f"{name}: loss must be 'ridge' or 'logistic'; got {loss!r}")

    d = n_features
    W = np.zeros((n_outputs, d), dtype=np.float64)
    h = np.zeros((n_outputs, d), dtype=np.float64)
    if mg is not None:
        # Classifier state: W + h + small numpy overhead.
        mg.check_state(2 * d * n_outputs * 8 + d * 8,
                       name=name + ".classifier.state")

    if loss == "ridge":
        # Warm start: W_0 ≈ (1/n) X^T Y (one gradient step at W=0).
        n_seen = 0
        for X_b, Y_b in batch_stream():
            if mg is not None:
                mg.check_state(len(X_b) * d * 8 + n_outputs * d * 8,
                               name=name + ".warm")
            W += Y_b.T @ X_b
            n_seen += X_b.shape[0]
        if n_seen == 0:
            raise ValueError(f"{name}: empty feature stream")
        W /= float(n_seen)
    else:  # logistic — no warm start (Y^T X would saturate sigmoid
           # for ±1 targets or imbalanced 0/1 targets)
        n_seen = 0
        for X_b, _ in batch_stream():
            n_seen += X_b.shape[0]
        if n_seen == 0:
            raise ValueError(f"{name}: empty feature stream")

    for _epoch in range(epochs):
        for X_b, Y_b in batch_stream():
            if mg is not None:
                # Transients: pred (b,K), grad (K,d), h update (K,d).
                # Estimate 3 * (K*d + b*K) * 8 bytes.
                mg.check_state(3 * (n_outputs * d + len(X_b) * n_outputs) * 8,
                               name=name + ".batch")
            z = X_b @ W.T                              # (b, K)
            if loss == "ridge":
                pred = z
            else:  # logistic
                pred = sigmoid(z)
            grad = (pred - Y_b).T @ X_b / max(len(X_b), 1)
            if penalty_mask is not None:
                grad = grad + lam * W * penalty_mask
            else:
                grad = grad + lam * W
            if use_rmsprop:
                h = beta * h + (1.0 - beta) * grad * grad
                W -= lr * grad / (np.sqrt(h) + eps)
            else:
                # Plain GD. Strongly-regularized logistic: RMSprop
                # oscillates at amplitude ~lr (adaptive denominator
                # can't resolve small grads). Plain GD contracts
                # geometrically (rate 1 - lr*lam).
                W -= lr * grad
    return W
