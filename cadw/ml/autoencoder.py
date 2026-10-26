"""Autoencoder (MLP): reconstruction-based DL anomaly detection.

Simple MLP encoder-decoder trained on sliding windows with SGD + momentum.
Anomaly score per window = reconstruction MSE; aggregated to timestamps
via max/mean.

Closes the D3.1 cell "DL from scratch + reconstruction" with a trainable
neural network, complementing Sub-PCA (linear) and EIF (tree-based).
Not PaAno (patch-CNN + memory bank, torch-based SOTA); PaAno-like is
deferred (requires backprop through 1D conv + triplet loss + memory
bank, ~3x implementation risk).

Trained in-sample (unsupervised): no labels needed. Window z-normalised
to zero mean unit std before encoding; reconstruction evaluated on
z-normalised windows to remove scale bias.

Pure numpy — no sklearn/scipy/torch in cadw/ (AST-guard
test_no_bench_deps_in_core.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sub_pca import _map_max, _map_mean


@dataclass
class AutoencoderResult:
    decision_scores: np.ndarray
    labels: np.ndarray
    threshold: float
    window_scores: np.ndarray
    training_losses: list
    window_size: int
    hidden_size: int
    latent_size: int
    n_epochs: int
    lr: float
    batch_size: int
    contamination: float
    aggregate: str
    reconstruction_error: str
    encoder_weights: list
    decoder_weights: list


def _probe_memory_limit_mb(guard) -> float | None:
    def _is_pos(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0
    MB = ("max_memory_mb", "_max_memory_mb", "limit_mb", "budget_mb",
          "_budget_mb", "max_mb", "_max_mb", "memory_limit_mb",
          "_memory_limit_mb")
    for a in MB:
        v = getattr(guard, a, None)
        if _is_pos(v):
            return float(v)
    BYTES = ("max_memory_bytes", "_max_memory_bytes", "max_bytes",
             "_max_bytes", "limit_bytes", "_limit_bytes", "budget_bytes",
             "_budget_bytes")
    for a in BYTES:
        v = getattr(guard, a, None)
        if _is_pos(v):
            return float(v) / (1024.0 * 1024.0)
    try:
        d = guard.__dict__
    except AttributeError:
        return None
    for k, v in d.items():
        kl = k.lower()
        if _is_pos(v) and "mb" in kl and any(
                t in kl for t in ("mem", "budget", "limit", "max")):
            return float(v)
    return None


def _xavier_init(fan_in: int, fan_out: int,
                 rng: np.random.Generator) -> np.ndarray:
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out))


def _forward(X: np.ndarray, W1, b1, W2, b2, W3, b3, W4, b4):
    h1 = np.maximum(X @ W1 + b1, 0.0)
    z = h1 @ W2 + b2
    h2 = np.maximum(z @ W3 + b3, 0.0)
    x_hat = h2 @ W4 + b4
    return x_hat, (h1, z, h2)


def _backward(X, x_hat, cache, W1, W2, W3, W4):
    h1, z, h2 = cache
    B, w = X.shape
    d_xhat = 2.0 * (x_hat - X) / (B * w)
    dW4 = h2.T @ d_xhat
    db4 = d_xhat.sum(axis=0)
    dh2 = d_xhat @ W4.T
    dh2 = dh2 * (h2 > 0.0)
    dW3 = z.T @ dh2
    db3 = dh2.sum(axis=0)
    dz = dh2 @ W3.T
    dW2 = h1.T @ dz
    db2 = dz.sum(axis=0)
    dh1 = dz @ W2.T
    dh1 = dh1 * (h1 > 0.0)
    dW1 = X.T @ dh1
    db1 = dh1.sum(axis=0)
    return dW1, db1, dW2, db2, dW3, db3, dW4, db4


def _znormalize_rows(X: np.ndarray) -> np.ndarray:
    """Per-row (window) z-normalization with zero-std guard."""
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu) / sd


def fit_autoencoder(
    series: np.ndarray,
    *,
    window_size: int = 50,
    hidden_size: int = 32,
    latent_size: int = 8,
    n_epochs: int = 30,
    lr: float = 0.01,
    batch_size: int = 64,
    momentum: float = 0.9,
    contamination: float = 0.1,
    aggregate: str = "max",
    reconstruction_error: str = "mean",
    random_state: int | None = 42,
    guard=None,
) -> AutoencoderResult:
    """MLP autoencoder for TS anomaly detection.

    Parameters
    ----------
    series : np.ndarray, shape (n,)
    window_size : int, default=50
    hidden_size : int, default=32
        Hidden layer width (encoder and decoder symmetric).
    latent_size : int, default=8
        Bottleneck dimension.
    n_epochs : int, default=30
    lr : float, default=0.01
    batch_size : int, default=64
    momentum : float, default=0.9
    contamination : float, default=0.1
    aggregate : str, default="max"
        "max" or "mean".
    random_state : int or None, default=42
    guard : MemoryGuard | None

    Returns
    -------
    AutoencoderResult
    """
    x = np.asarray(series, dtype=np.float64).ravel()
    n = len(x)
    if n < window_size:
        raise ValueError(
            f"series length {n} < window_size {window_size}")
    if hidden_size < 1 or latent_size < 1:
        raise ValueError(
            f"hidden_size and latent_size must be >= 1 "
            f"(got {hidden_size}, {latent_size})")
    if aggregate not in ("max", "mean"):
        raise ValueError(
            f"aggregate must be 'max' or 'mean', got {aggregate!r}")
    if n_epochs < 1 or batch_size < 1:
        raise ValueError(
            f"n_epochs and batch_size must be >= 1 "
            f"(got {n_epochs}, {batch_size})")

    n_w = n - window_size + 1
    if guard is not None:
        est_bytes = 8 * n_w * window_size * 4
        limit_mb = _probe_memory_limit_mb(guard)
        if limit_mb is not None and limit_mb > 0:
            if est_bytes > limit_mb * 1024 * 1024:
                from cadw.core.memory_guard import MemoryLimitExceeded
                raise MemoryLimitExceeded(
                    f"Autoencoder estimated {est_bytes} bytes > "
                    f"guard {limit_mb * 1024 * 1024:.0f} bytes")

    windows = np.lib.stride_tricks.sliding_window_view(x, window_size).copy()
    X = _znormalize_rows(windows)

    rng = np.random.default_rng(random_state)
    W1 = _xavier_init(window_size, hidden_size, rng)
    b1 = np.zeros(hidden_size)
    W2 = _xavier_init(hidden_size, latent_size, rng)
    b2 = np.zeros(latent_size)
    W3 = _xavier_init(latent_size, hidden_size, rng)
    b3 = np.zeros(hidden_size)
    W4 = _xavier_init(hidden_size, window_size, rng)
    b4 = np.zeros(window_size)

    params = [W1, b1, W2, b2, W3, b3, W4, b4]
    velocities = [np.zeros_like(p) for p in params]

    training_losses = []
    n_batches = max(1, n_w // batch_size)
    for epoch in range(n_epochs):
        indices = rng.permutation(n_w)
        epoch_loss = 0.0
        for b in range(n_batches):
            batch_idx = indices[b * batch_size:(b + 1) * batch_size]
            if len(batch_idx) == 0:
                continue
            Xb = X[batch_idx]
            x_hat, cache = _forward(Xb, *params)
            loss = float(np.mean((x_hat - Xb) ** 2))
            epoch_loss += loss
            grads = _backward(Xb, x_hat, cache, W1, W2, W3, W4)
            for p, g, v in zip(params, grads, velocities):
                v_new = momentum * v - lr * g
                p += v_new
                v[:] = v_new
        training_losses.append(epoch_loss / n_batches)

    # Inference: per-window reconstruction error.
    # 'mean' = MSE (classic). 'median' = median(|residual|^2) — more
    # robust to isolated position-wise memorization (MLP may learn to
    # reconstruct 1-2 spike positions within an otherwise-normal window;
    # median ignores those, focaressing structural errors).
    x_hat_all, _ = _forward(X, *params)
    resid_sq = (x_hat_all - X) ** 2
    if reconstruction_error == "mean":
        window_scores = np.mean(resid_sq, axis=1)
    elif reconstruction_error == "median":
        window_scores = np.median(resid_sq, axis=1)
    else:
        raise ValueError(
            f"reconstruction_error must be 'mean' or 'median', "
            f"got {reconstruction_error!r}")

    if aggregate == "max":
        scores = _map_max(window_scores, window_size, n, stride=1)
    else:
        scores = _map_mean(window_scores, window_size, n, stride=1)

    threshold = float(np.percentile(
        scores, 100.0 * (1.0 - contamination)))
    labels = (scores > threshold).astype(int)

    return AutoencoderResult(
        decision_scores=scores,
        labels=labels,
        threshold=threshold,
        window_scores=window_scores,
        training_losses=training_losses,
        window_size=window_size,
        hidden_size=hidden_size,
        latent_size=latent_size,
        n_epochs=n_epochs,
        lr=lr,
        batch_size=batch_size,
        contamination=contamination,
        aggregate=aggregate,
        reconstruction_error=reconstruction_error,
        encoder_weights=[W1.copy(), b1.copy(), W2.copy(), b2.copy()],
        decoder_weights=[W3.copy(), b3.copy(), W4.copy(), b4.copy()],
    )

