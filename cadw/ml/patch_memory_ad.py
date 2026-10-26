"""PatchMemoryAD: PaAno-inspired patch+memory-bank anomaly detection.

Inspired by PaAno (ICLR 2026): patch-based representation + memory bank
of normal patterns + triplet loss for anomaly discrimination.

Architecture (documented deviations from PaAno paper):
  1. Windowing: (n,) -> (n_w, window_size).
  2. Patch extraction: split window into non-overlapping patches of
     length patch_len; flatten -> (n_w, n_patches*patch_len).
  3. Encoder: 2-layer MLP (flattened -> hidden ReLU -> latent).
     DEVIATION: PaAno uses patch-wise 1D-CNN; we use MLP on flattened
     patches — same "patch-based representation" but no conv backprop
     (which is bug-prone in pure numpy; MLP backward already validated
     by v0.9.61 autoencoder).
  4. Memory bank: M=100 slots x latent_dim, initialized to random
     normal-window latents (after first epoch of encoder warmup).
  5. Training: triplet loss per window:
       z = encode(x)
       pos_idx = argmin_k ||z - mem[k]||²
       neg_idx = argmax_k ||z - mem[k]||²
       loss = max(0, ||z - pos||² - ||z - neg||² + margin)
     Backprop to encoder; update mem[pos_idx] with z (momentum α=0.1).
  6. Scoring: anomaly_score(x) = min_k ||encode(x) - mem[k]||².
     Aggregate to timestamps via max/mean.

Other deviations:
  - Single-scale patches (PaAno: multi-scale 8/16/32).
  - Single head (PaAno: multi-head).
  - Trained in-sample (unsupervised).

USE CASE: point anomaly specialist. Bench at n=30000 shows recall 1.0
on point spikes (20/20) but recall ~0.04 on segment anomalies. Segment
anomalies produce flat patches indistinguishable from normal flat
regions; point spikes produce unique patches far from all memory
slots. This matches TSB-AD: PaAno is SOTA on point, not on segment.
For segments prefer KMeansAD, Sub-PCA (mean aggregation), or CUSUM.

Pure numpy — no sklearn/scipy/torch in cadw/ (AST-guard
test_no_bench_deps_in_core.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sub_pca import _map_max, _map_mean


@dataclass
class PatchMemoryADResult:
    decision_scores: np.ndarray
    labels: np.ndarray
    threshold: float
    window_scores: np.ndarray
    training_losses: list
    memory_bank: np.ndarray
    window_size: int
    patch_len: int
    hidden_size: int
    latent_size: int
    n_memory: int
    n_epochs: int
    lr: float
    momentum: float
    triplet_margin: float
    contamination: float
    aggregate: str
    encoder_weights: list


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


def _extract_patches(windows: np.ndarray, patch_len: int) -> np.ndarray:
    """(n_w, w) -> (n_w, n_patches*patch_len). Patches concatenated."""
    n_w, w = windows.shape
    if w % patch_len != 0:
        # Truncate to multiple of patch_len
        w_used = (w // patch_len) * patch_len
        windows = windows[:, :w_used]
        w = w_used
    n_patches = w // patch_len
    reshaped = windows.reshape(n_w, n_patches, patch_len)
    return reshaped.reshape(n_w, n_patches * patch_len)


def _encode(X: np.ndarray, W1, b1, W2, b2):
    h = np.maximum(X @ W1 + b1, 0.0)
    z = h @ W2 + b2
    return z, h


def _encode_backward(dz: np.ndarray, X: np.ndarray, h: np.ndarray,
                     W2: np.ndarray) -> tuple:
    """Backprop through encoder given dz (grad wrt z). Returns param grads."""
    dW2 = h.T @ dz
    db2 = dz.sum(axis=0)
    dh = dz @ W2.T
    dh = dh * (h > 0.0)
    dW1 = X.T @ dh
    db1 = dh.sum(axis=0)
    return dW1, db1, dW2, db2


def _xavier_init(fan_in: int, fan_out: int,
                 rng: np.random.Generator) -> np.ndarray:
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out))


def _sq_dists(z: np.ndarray, mem: np.ndarray) -> np.ndarray:
    """Squared distances (n, M) between z rows and memory slots."""
    # ||z_i - mem_k||² = z_i·z_i - 2·z_i·mem_k + mem_k·mem_k
    z_sq = (z * z).sum(axis=1)
    m_sq = (mem * mem).sum(axis=1)
    return z_sq[:, None] - 2.0 * (z @ mem.T) + m_sq[None, :]


def _znormalize_rows(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu) / sd


def fit_patch_memory_ad(
    series: np.ndarray,
    *,
    window_size: int = 50,
    patch_len: int = 10,
    hidden_size: int = 32,
    latent_size: int = 16,
    n_memory: int = 100,
    n_epochs: int = 10,
    lr: float = 0.005,
    memory_momentum: float = 0.1,
    triplet_margin: float = 0.5,
    contamination: float = 0.1,
    aggregate: str = "max",
    random_state: int | None = 42,
    guard=None,
) -> PatchMemoryADResult:
    """PatchMemoryAD anomaly detection.

    See module docstring for architecture and deviations.
    """
    x = np.asarray(series, dtype=np.float64).ravel()
    n = len(x)
    if n < window_size:
        raise ValueError(
            f"series length {n} < window_size {window_size}")
    if window_size % patch_len != 0:
        raise ValueError(
            f"window_size ({window_size}) must be a multiple of "
            f"patch_len ({patch_len})")
    if n_memory < 2:
        raise ValueError(f"n_memory must be >= 2, got {n_memory}")
    if aggregate not in ("max", "mean"):
        raise ValueError(
            f"aggregate must be 'max' or 'mean', got {aggregate!r}")

    n_w = n - window_size + 1
    if n_w < 10:
        raise ValueError(f"too few windows ({n_w}) for training")

    if guard is not None:
        est_bytes = 8 * n_w * window_size * 2
        limit_mb = _probe_memory_limit_mb(guard)
        if limit_mb is not None and limit_mb > 0:
            if est_bytes > limit_mb * 1024 * 1024:
                from cadw.core.memory_guard import MemoryLimitExceeded
                raise MemoryLimitExceeded(
                    f"PatchMemoryAD estimated {est_bytes} bytes > "
                    f"guard {limit_mb * 1024 * 1024:.0f} bytes")

    windows = np.lib.stride_tricks.sliding_window_view(
        x, window_size).copy()
    X_patches = _extract_patches(windows, patch_len)  # (n_w, d)
    X_norm = _znormalize_rows(X_patches)
    d = X_norm.shape[1]

    rng = np.random.default_rng(random_state)

    # Initialize encoder
    W1 = _xavier_init(d, hidden_size, rng)
    b1 = np.zeros(hidden_size)
    W2 = _xavier_init(hidden_size, latent_size, rng)
    b2 = np.zeros(latent_size)

    # Initialize memory: random latent vectors (small scale)
    mem = rng.standard_normal((n_memory, latent_size)) * 0.1

    # Warmup: encode a few windows and use their latents as initial memory
    warmup_idx = rng.choice(n_w, size=min(n_memory, n_w), replace=False)
    z_warmup, _ = _encode(X_norm[warmup_idx], W1, b1, W2, b2)
    if len(warmup_idx) < n_memory:
        pad = rng.standard_normal((n_memory - len(warmup_idx), latent_size)) * 0.1
        mem = np.vstack([z_warmup, pad])
    else:
        mem = z_warmup.copy()

    velocities = [np.zeros_like(p) for p in (W1, b1, W2, b2)]
    training_losses = []

    # Training with triplet loss
    for epoch in range(n_epochs):
        idx = rng.permutation(n_w)
        epoch_loss = 0.0
        n_updates = 0
        for i in idx:
            xi = X_norm[i:i + 1]
            z, h = _encode(xi, W1, b1, W2, b2)  # (1, latent)
            z_flat = z[0]
            # distances to memory
            dists = _sq_dists(z, mem)[0]  # (M,)
            pos_idx = int(np.argmin(dists))
            neg_idx = int(np.argmax(dists))
            pos_vec = mem[pos_idx]
            neg_vec = mem[neg_idx]
            d_pos = float(np.sum((z_flat - pos_vec) ** 2))
            d_neg = float(np.sum((z_flat - neg_vec) ** 2))
            loss = max(0.0, d_pos - d_neg + triplet_margin)
            epoch_loss += loss

            if loss > 0.0:
                # Gradient of triplet loss wrt z
                # loss = ||z-pos||² - ||z-neg||² + margin
                # d loss / dz = 2*(z-pos) - 2*(z-neg) = 2*(neg-pos)
                grad_z = 2.0 * (neg_vec - pos_vec)  # (latent,)
                dz = grad_z.reshape(1, -1)
                dW1, db1, dW2, db2 = _encode_backward(dz, xi, h, W2)
                for p, g, v in zip((W1, b1, W2, b2),
                                   (dW1, db1, dW2, db2),
                                   velocities):
                    v_new = 0.9 * v - lr * g
                    p += v_new
                    v[:] = v_new
                n_updates += 1

            # Update memory slot (momentum)
            mem[pos_idx] = ((1.0 - memory_momentum) * mem[pos_idx]
                            + memory_momentum * z_flat)

        training_losses.append(epoch_loss / max(n_updates, 1))

    # Inference
    z_all, _ = _encode(X_norm, W1, b1, W2, b2)
    dists_all = _sq_dists(z_all, mem)  # (n_w, M)
    window_scores = dists_all.min(axis=1)  # min distance to memory

    if aggregate == "max":
        scores = _map_max(window_scores, window_size, n, stride=1)
    else:
        scores = _map_mean(window_scores, window_size, n, stride=1)

    threshold = float(np.percentile(
        scores, 100.0 * (1.0 - contamination)))
    out_labels = (scores > threshold).astype(int)

    return PatchMemoryADResult(
        decision_scores=scores,
        labels=out_labels,
        threshold=threshold,
        window_scores=window_scores,
        training_losses=training_losses,
        memory_bank=mem.copy(),
        window_size=window_size,
        patch_len=patch_len,
        hidden_size=hidden_size,
        latent_size=latent_size,
        n_memory=n_memory,
        n_epochs=n_epochs,
        lr=lr,
        momentum=memory_momentum,
        triplet_margin=triplet_margin,
        contamination=contamination,
        aggregate=aggregate,
        encoder_weights=[W1.copy(), b1.copy(), W2.copy(), b2.copy()],
    )

