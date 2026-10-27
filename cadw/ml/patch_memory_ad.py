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

NORMALIZATION (v0.9.63-fix2, per HYDRA SIGMOD 2026 + PaAno ICLR 2026):
  HYDRA §1: "while normalization reduces scale bias and can enhance
  anomaly detectability, it may also suppress amplitude-driven
  anomalies (e.g., spikes), such normalization inadvertently attenuates
  the large magnitude deviations that signal anomalous behavior."
  PaAno normalises each PATCH (instance norm, patch_len 8-16), NOT the
  whole window — a spike in a 10-point patch stays an outlier; a spike
  in a 50-point flattened window is diluted.
  Default "none": preserves amplitude spikes (ablation: recall 1.0 on
  noisy sigma=0.5; z-norm "window" gave recall 0.0).
  Options: "none" | "window" (old) | "patch" (PaAno-faithful, best for
  drift/scale-shifted series).

USE CASE (updated): point/amplitude anomaly specialist with
normalization="none"; with "patch" normalization also handles
scale-shifted series. Segment anomalies: earlier version (with window
z-norm) missed them; "none" improves segment detection too. Prefer
KMeansAD, Sub-PCA (mean aggregation), or CUSUM when segment
boundaries matter.

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
    normalization: str


from cadw.core.spill import SpillDir


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


# v0.9.87-P4d (audit HIGH-1 full fix): streaming patch_memory_ad.
# X_norm(n_w, d) written to SpillDir (Arrow IPC, 1 row per batch for
# random access via ArrowIPCBatchReader.get_batch(i)). Peak RAM
# O(chunk*d) for write, O(d) per read in training, O(chunk*d) for
# inference chunks, O(n) output. See METHODOLOGY §5, PRIMITIVES §5.


def _iter_x_window_chunks_pma(x: np.ndarray, window_size: int,
                              chunk_nw: int):
    """Yield (chunk(nb, w), start) chunks from 1D x — bounded."""
    n = len(x)
    n_w = n - window_size + 1
    if n_w < 1:
        return
    offsets = np.arange(window_size, dtype=np.int64)
    for start in range(0, n_w, chunk_nw):
        end = min(start + chunk_nw, n_w)
        abs_starts = np.arange(start, end, dtype=np.int64)
        idx = abs_starts[:, None] + offsets[None, :]
        yield x[idx], start


def _resolve_chunk_nw_pma(d_patches: int, guard=None) -> int:
    """Chunk size so per-chunk peak RAM stays within budget."""
    default_target = 8 * 1024 * 1024
    target_bytes = default_target
    if guard is not None:
        state_limit = getattr(guard, "state_limit_bytes", None)
        if (isinstance(state_limit, (int, float))
                and state_limit > 0):
            target_bytes = min(int(target_bytes),
                               max(1, int(state_limit) // 8))
    bytes_per_window = 8 * max(int(d_patches), 1) * 6
    n = max(100, target_bytes // bytes_per_window)
    return min(10_000, n)


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
    normalization: str = "none",
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

    # v0.9.87-P4d (audit HIGH-1 full fix): per-chunk buffers +
    # output arrays (bounded) + model params. X_norm is memmap-backed
    # (OS page cache); Python-side peak is O(chunk_nw*d).
    _npw = max(1, window_size // patch_len)
    _d_patches = _npw * patch_len
    _chunk_nw = _resolve_chunk_nw_pma(_d_patches, guard)
    if guard is not None:
        per_chunk = _chunk_nw * _d_patches * 8 * 6
        output_phase = (n_w + n) * 8 * 2
        model_params = (_d_patches * hidden_size + hidden_size
                        + hidden_size * latent_size + latent_size) * 8 * 2
        mem_params = n_memory * latent_size * 8 * 2
        est_bytes = int(max(per_chunk, output_phase)
                        + model_params + mem_params)
        if hasattr(guard, "check_state") and callable(guard.check_state):
            guard.check_state(est_bytes, name="patch_memory_ad.state")
        else:
            limit_mb = _probe_memory_limit_mb(guard)
            if limit_mb is not None and limit_mb > 0:
                if est_bytes > limit_mb * 1024 * 1024:
                    from cadw.core.memory_guard import MemoryLimitExceeded
                    raise MemoryLimitExceeded(
                        f"PatchMemoryAD estimated {est_bytes} bytes > "
                        f"guard {limit_mb * 1024 * 1024:.0f} bytes")

    # v0.9.87-P4d: X_norm is memmap-backed (OS page cache, O(1)
    # random row access for the training loop). Python-side peak RAM
    # is O(chunk_nw*d). Bit-identical output vs pre-P4d (same RNG
    # sequence, same per-row math, same order).
    d = _d_patches

    with SpillDir(guard=guard, name="patch_memory_ad") as _sdir:
        _mmap_path = _sdir.path("X_norm.bin")
        X_norm_mmap = np.memmap(
            str(_mmap_path), dtype=np.float64, mode="w+",
            shape=(n_w, d))

        # Fill memmap via streaming chunks (no full (n_w, w) alloc).
        for _chunk, _start in _iter_x_window_chunks_pma(
                x, window_size, _chunk_nw):
            X_patches_chunk = _extract_patches(_chunk, patch_len)
            if X_patches_chunk.shape[1] != d:
                raise RuntimeError(
                    f"_extract_patches returned d={X_patches_chunk.shape[1]}"
                    f" != {d} (internal inconsistency)")
            # v0.9.63-fix2: normalization semantics preserved.
            if normalization == "none":
                X_norm_chunk = X_patches_chunk
            elif normalization == "window":
                X_norm_chunk = _znormalize_rows(X_patches_chunk)
            elif normalization == "patch":
                npw = window_size // patch_len
                p2 = X_patches_chunk.reshape(
                    X_patches_chunk.shape[0], npw, patch_len)
                mu = p2.mean(axis=2, keepdims=True)
                sd = p2.std(axis=2, keepdims=True)
                sd = np.where(sd < 1e-12, 1.0, sd)
                X_norm_chunk = ((p2 - mu) / sd).reshape(
                    X_patches_chunk.shape)
            else:
                raise ValueError(
                    f"normalization must be 'none'|'window'|'patch', "
                    f"got {normalization!r}")
            nb = X_norm_chunk.shape[0]
            X_norm_mmap[_start:_start + nb] = X_norm_chunk
        X_norm_mmap.flush()

        rng = np.random.default_rng(random_state)

        W1 = _xavier_init(d, hidden_size, rng)
        b1 = np.zeros(hidden_size)
        W2 = _xavier_init(hidden_size, latent_size, rng)
        b2 = np.zeros(latent_size)

        mem = rng.standard_normal((n_memory, latent_size)) * 0.1

        warmup_idx = rng.choice(
            n_w, size=min(n_memory, n_w), replace=False)
        X_warmup = np.asarray(X_norm_mmap[warmup_idx])
        z_warmup, _ = _encode(X_warmup, W1, b1, W2, b2)
        if len(warmup_idx) < n_memory:
            pad = rng.standard_normal(
                (n_memory - len(warmup_idx), latent_size)) * 0.1
            mem = np.vstack([z_warmup, pad])
        else:
            mem = z_warmup.copy()

        velocities = [np.zeros_like(p) for p in (W1, b1, W2, b2)]
        training_losses = []

        for epoch in range(n_epochs):
            idx = rng.permutation(n_w)
            epoch_loss = 0.0
            n_updates = 0
            for i in idx:
                xi = X_norm_mmap[int(i):int(i) + 1]  # (1, d) view
                z, h = _encode(xi, W1, b1, W2, b2)
                z_flat = z[0]
                dists = _sq_dists(z, mem)[0]
                pos_idx = int(np.argmin(dists))
                neg_idx = int(np.argmax(dists))
                pos_vec = mem[pos_idx]
                neg_vec = mem[neg_idx]
                d_pos = float(np.sum((z_flat - pos_vec) ** 2))
                d_neg = float(np.sum((z_flat - neg_vec) ** 2))
                loss = max(0.0, d_pos - d_neg + triplet_margin)
                epoch_loss += loss

                if loss > 0.0:
                    grad_z = 2.0 * (neg_vec - pos_vec)
                    dz = grad_z.reshape(1, -1)
                    dW1, db1, dW2, db2 = _encode_backward(
                        dz, xi, h, W2)
                    for p, g, v in zip((W1, b1, W2, b2),
                                       (dW1, db1, dW2, db2),
                                       velocities):
                        v_new = 0.9 * v - lr * g
                        p += v_new
                        v[:] = v_new
                    n_updates += 1

                mem[pos_idx] = ((1.0 - memory_momentum) * mem[pos_idx]
                                + memory_momentum * z_flat)

            training_losses.append(epoch_loss / max(n_updates, 1))

        # Inference: streaming chunks over memmap.
        window_scores = np.empty(n_w, dtype=np.float64)
        for _start in range(0, n_w, _chunk_nw):
            _end = min(_start + _chunk_nw, n_w)
            xi = np.asarray(X_norm_mmap[_start:_end])
            z_b, _ = _encode(xi, W1, b1, W2, b2)
            dists_b = _sq_dists(z_b, mem)
            window_scores[_start:_end] = dists_b.min(axis=1)

        del X_norm_mmap  # flush + release memmap handle before cleanup

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
        normalization=normalization,
    )

