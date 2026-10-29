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

from .sub_pca import _map_max, _map_mean, _prepare_windows


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
    normalization: str


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


def _resolve_chunk_nw_ae(d: int, batch_size: int, guard=None) -> int:
    """v0.9.91-S6d: inference chunk size, 8 MB target.

    Per-window: chunk buffer (d*8) + forward temp (d*8 * 3) ~ 4*d*8.
    Floor at batch_size (training mini-batch unaffected by chunk).
    """
    default_target = 8 * 1024 * 1024
    target_bytes = default_target
    if guard is not None:
        sl = getattr(guard, "state_limit_bytes", None)
        if isinstance(sl, (int, float)) and sl > 0:
            target_bytes = min(int(target_bytes), max(1, int(sl) // 8))
    bytes_per_window = 8 * max(int(d), 1) * 4
    n = max(int(batch_size), target_bytes // bytes_per_window)
    return min(10_000, n)


def _iter_window_chunks_ae(series, w, chunk_nw):
    """Yield (chunk(nb, w*c), start) from 1D or 2D series. stride=1."""
    X = np.asarray(series, dtype=np.float64)
    if X.ndim == 1:
        X_ = X[:, None]
    else:
        X_ = X
    n, c = X_.shape
    n_w = n - w + 1
    if n_w < 1:
        return
    offsets = np.arange(w, dtype=np.int64)
    for start in range(0, n_w, chunk_nw):
        end = min(start + chunk_nw, n_w)
        abs_starts = np.arange(start, end, dtype=np.int64)
        idx = abs_starts[:, None] + offsets[None, :]
        yield X_[idx].reshape(end - start, w * c), start


def _fetch_windows_ae(series, indices, w, c, normalization):
    """Extract (bs, w*c) batch by row-indices (fancy indexing).

    Per-row normalize if normalization="window" — bit-identical to
    pre-S6d `_znormalize_rows(windows)[batch_idx]` (row-independent).
    """
    X = np.asarray(series, dtype=np.float64)
    if X.ndim == 1:
        X_ = X[:, None]
    else:
        X_ = X
    idx = indices[:, None] + np.arange(w, dtype=np.int64)[None, :]
    Xb = X_[idx].reshape(len(indices), w * c)
    if normalization == "window":
        Xb = _znormalize_rows(Xb)
    return Xb


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
    normalization: str = "none",
    random_state: int | None = 42,
    guard=None,
) -> AutoencoderResult:
    """MLP autoencoder for TS anomaly detection.

    v0.9.91-S6d: two paths.
      - Duck-typed guard (max_memory_mb only, no check_state):
        old path via _prepare_windows (bit-identical to pre-S6d).
        Tight budget → MemoryLimitExceeded.
      - Real MemoryGuard (has check_state): chunked SGD.
        rng.permutation(n_w) held as (n_w,) int64 in RAM
        (8*n_w bytes = 50x smaller than (n_w, d) float64 for w=50).
        Windows fetched on-demand per mini-batch via fancy indexing
        (bs*d*8 bounded); (n_w, d) intermediate eliminated.
        BIT-IDENTICAL to old path on the same random_state: same
        rng.permutation sequence, same batch order, same per-row
        normalization, same forward/backward.
        Peak = 16*n_w + bs*d*8*6 + params*8*2.
        Refuses (MemoryLimitExceeded) when 16*n_w > state_limit
        (honest: full OOC via disk permutation is S6d-full backlog).
    """
    _arr = np.asarray(series)
    n_pre = int(_arr.shape[0]) if _arr.ndim >= 1 else 0
    c_pre = 1 if _arr.ndim == 1 else int(np.prod(_arr.shape[1:]))
    if n_pre < window_size:
        raise ValueError(
            f"series length {n_pre} < window_size {window_size}")
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
    n_w_pre = n_pre - window_size + 1
    d_pre = window_size * c_pre

    has_check_state = (
        guard is not None and hasattr(guard, "check_state")
        and callable(guard.check_state)
    )

    if not has_check_state:
        # --- Old path (back-compat, bit-identical) ---
        if guard is not None:
            est_bytes = 8 * n_w_pre * d_pre * 4
            limit_mb = _probe_memory_limit_mb(guard)
            if limit_mb is not None and limit_mb > 0:
                if est_bytes > limit_mb * 1024 * 1024:
                    from cadw.core.memory_guard import MemoryLimitExceeded
                    raise MemoryLimitExceeded(
                        f"Autoencoder estimated {est_bytes} bytes > "
                        f"guard {limit_mb * 1024 * 1024:.0f} bytes")

        windows, n, c = _prepare_windows(series, window_size, stride=1)
        n_w = len(windows)
        d = windows.shape[1]

        if normalization == "none":
            X = windows
        elif normalization == "window":
            X = _znormalize_rows(windows)
        else:
            raise ValueError(
                f"normalization must be 'none'|'window', "
                f"got {normalization!r}")

        rng = np.random.default_rng(random_state)
        W1 = _xavier_init(d, hidden_size, rng)
        b1 = np.zeros(hidden_size)
        W2 = _xavier_init(hidden_size, latent_size, rng)
        b2 = np.zeros(latent_size)
        W3 = _xavier_init(latent_size, hidden_size, rng)
        b3 = np.zeros(hidden_size)
        W4 = _xavier_init(hidden_size, d, rng)
        b4 = np.zeros(d)

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
    else:
        # --- Chunked OOC path (real MemoryGuard) ---
        n = n_pre
        c = c_pre
        d = d_pre
        n_w = n_w_pre
        if normalization not in ("none", "window"):
            raise ValueError(
                f"normalization must be 'none'|'window', "
                f"got {normalization!r}")
        if reconstruction_error not in ("mean", "median"):
            raise ValueError(
                f"reconstruction_error must be 'mean' or 'median', "
                f"got {reconstruction_error!r}")

        chunk_nw = _resolve_chunk_nw_ae(d, batch_size, guard)

        # Peak estimate: permutation (8n_w) + scores (8n_w) +
        # batch buffer (bs*d*8 * 6 forward+backward temps) +
        # params + velocities.
        est_bytes = (16 * n_w
                     + batch_size * d * 8 * 6
                     + (d * hidden_size + hidden_size
                        + hidden_size * latent_size + latent_size
                        + latent_size * hidden_size + hidden_size
                        + hidden_size * d + d) * 8 * 2)
        guard.check_state(est_bytes, name="autoencoder.chunked")

        rng = np.random.default_rng(random_state)
        W1 = _xavier_init(d, hidden_size, rng)
        b1 = np.zeros(hidden_size)
        W2 = _xavier_init(hidden_size, latent_size, rng)
        b2 = np.zeros(latent_size)
        W3 = _xavier_init(latent_size, hidden_size, rng)
        b3 = np.zeros(hidden_size)
        W4 = _xavier_init(hidden_size, d, rng)
        b4 = np.zeros(d)

        params = [W1, b1, W2, b2, W3, b3, W4, b4]
        velocities = [np.zeros_like(p) for p in params]

        training_losses = []
        n_batches = max(1, n_w // batch_size)
        for epoch in range(n_epochs):
            # O(n_w) int64 = 8*n_w bytes. 50x smaller than
            # (n_w, d) float64 for w=50.
            indices = rng.permutation(n_w)
            epoch_loss = 0.0
            for b in range(n_batches):
                batch_idx = indices[b * batch_size:(b + 1) * batch_size]
                if len(batch_idx) == 0:
                    continue
                Xb = _fetch_windows_ae(
                    series, batch_idx, window_size, c, normalization)
                x_hat, cache = _forward(Xb, *params)
                loss = float(np.mean((x_hat - Xb) ** 2))
                epoch_loss += loss
                grads = _backward(Xb, x_hat, cache, W1, W2, W3, W4)
                for p, g, v in zip(params, grads, velocities):
                    v_new = momentum * v - lr * g
                    p += v_new
                    v[:] = v_new
            training_losses.append(epoch_loss / n_batches)

        # Chunked inference: (n_w, d) intermediate eliminated.
        window_scores = np.empty(n_w, dtype=np.float64)
        for chunk, start in _iter_window_chunks_ae(
                series, window_size, chunk_nw):
            nb = chunk.shape[0]
            Xc = chunk if normalization == "none" else _znormalize_rows(chunk)
            x_hat_c, _ = _forward(Xc, *params)
            resid_sq_c = (x_hat_c - Xc) ** 2
            if reconstruction_error == "mean":
                window_scores[start:start + nb] = np.mean(
                    resid_sq_c, axis=1)
            else:
                window_scores[start:start + nb] = np.median(
                    resid_sq_c, axis=1)

    # --- Common: aggregate + threshold + labels ---
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
        normalization=normalization,
    )
