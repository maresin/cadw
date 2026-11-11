"""Kernel PCA (Scholkopf, Smola, Muller 1998; sklearn parity).

Intrinsic O(n^2) fit transient (K = kernel(X, X), eigh). Honest
bounded_by_domain {n: 1e4} per DEC-075: at n > 1e4, fit refuses with
MemoryLimitExceeded (K = 2 * n^2 * f64 bytes; 1.6 GB at n=1e4 for
K + eigh copy).

transform: OOC in n_q via streaming (pairwise_kernel(X_fit, Q_batch)).

Sklearn reference: sklearn.decomposition.KernelPCA (P6 formulas).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryLimitExceeded
from cadw.ops.linalg import pairwise_kernel, eigh_topk
from cadw.ml._logging import log_fit


# Intrinsic bound (DEC-075). n=1e4 -> K = 800 MB; + eigh copy ~1.6 GB.
_MAX_N_INTRINSIC = 10000

_SUPPORTED_KERNELS = ("linear", "rbf", "poly", "laplacian")


def _svd_flip(u: np.ndarray) -> np.ndarray:
    """Deterministic eigenvector signs (sklearn.utils.extmath.svd_flip
    u_based_decision=True: largest-|u[i,j]| in each column becomes +).

    In-place on u (caller owns the array).
    """
    max_abs_rows = np.argmax(np.abs(u), axis=0)
    signs = np.sign(u[max_abs_rows, np.arange(u.shape[1])])
    signs[signs == 0] = 1.0  # all-zero column edge; keep sign
    u *= signs
    return u


class _KernelCenterer:
    """Kernel matrix double-centering (sklearn KernelCenterer parity).

    fit(K): K_fit_rows_ = sum(K, axis=0)/n; K_fit_all_ = sum(K_fit_rows)/n.
    transform(K_new): K_new - K_fit_rows_ - sum(K_new,axis=1)/n + K_fit_all_.
    """

    def __init__(self):
        self.K_fit_rows_: Optional[np.ndarray] = None
        self.K_fit_all_: Optional[float] = None

    def fit(self, K: np.ndarray) -> "_KernelCenterer":
        n = K.shape[0]
        if K.shape[1] != n:
            raise ValueError(
                f"_KernelCenterer.fit: K must be square; got {K.shape}")
        self.K_fit_rows_ = K.sum(axis=0) / n
        self.K_fit_all_ = float(self.K_fit_rows_.sum() / n)
        return self

    def transform(self, K_new: np.ndarray) -> np.ndarray:
        if self.K_fit_rows_ is None:
            raise RuntimeError("_KernelCenterer.transform before fit")
        n_fit = self.K_fit_rows_.shape[0]
        K_pred_cols = K_new.sum(axis=1, keepdims=True) / n_fit
        return K_new - self.K_fit_rows_ - K_pred_cols + self.K_fit_all_


def _batch_to_X(b, X_cols):
    """Extract (b, d) float64 from pa.RecordBatch (NaN preserved)."""
    arrs = []
    for c in X_cols:
        col = b.column(c)
        if not pa.types.is_floating(col.type):
            col = col.cast(pa.float64())
        arrs.append(col.to_numpy(zero_copy_only=False).astype(
            np.float64, copy=False))
    return np.column_stack(arrs) if arrs else np.empty((b.num_rows, 0))


@dataclass
class KernelPCAResult:
    """KernelPCA fit result (sklearn parity, intrinsic O(n^2) fit).

    Holds: X_fit (n, d), centerer stats (K_fit_rows (n,), K_fit_all scalar),
    eigenvalues (k,), eigenvectors (n, k). State O(n*d + n*k + n + k).
    `.transform(df)` streaming via pairwise_kernel(X_fit, Q_batch).
    `.fit_transform()` returns ndarray (n, k) — sklearn parity.

    release_train_state() clears X_fit + eigvecs; fit_transform/transform
    raise after.
    """
    X_fit: Optional[np.ndarray] = None
    X_cols: list = field(default_factory=list)
    K_fit_rows: Optional[np.ndarray] = None
    K_fit_all: float = 0.0
    eigenvalues: Optional[np.ndarray] = None
    eigenvectors: Optional[np.ndarray] = None
    n_components: int = 0
    kernel: str = "linear"
    gamma: float = 1.0
    degree: int = 3
    coef0: float = 1.0
    n_train: int = 0

    def state_nbytes(self) -> int:
        if self.X_fit is None:
            return 0
        sz = 256 + int(self.X_fit.nbytes) + int(self.K_fit_rows.nbytes)
        sz += 8  # K_fit_all scalar
        sz += int(self.eigenvalues.nbytes)
        sz += int(self.eigenvectors.nbytes)
        return int(sz)

    def release_train_state(self) -> None:
        """Clear X_fit + eigvecs (idempotent). transform raises after."""
        self.X_fit = None
        self.eigenvectors = None
        self.K_fit_rows = None

    def fit_transform(self) -> np.ndarray:
        """Training projection (n_train, n_components) = eigvecs * sqrt(eigvals).

        sklearn parity: KernelPCA.fit_transform returns this directly.
        """
        if self.X_fit is None:
            raise RuntimeError(
                "KernelPCAResult.fit_transform: model released "
                "(call fit_kernel_pca again)")
        return (self.eigenvectors
                * np.sqrt(self.eigenvalues)[None, :]).astype(np.float64)

    def transform(self, df, X_cols=None, *, batch_size=None,
                  memory_guard=None):
        """Project new data Q (OOC in n_q via streaming).

        Returns CadwDataFrame with cols kp_0..kp_{n_components-1}.
        """
        if self.X_fit is None:
            raise RuntimeError(
                "KernelPCAResult.transform: model released "
                "(call fit_kernel_pca again)")
        X_cols = list(X_cols) if X_cols is not None else list(self.X_cols)
        if X_cols != list(self.X_cols):
            # Column order must match; require exact set
            if set(X_cols) != set(self.X_cols):
                raise ValueError(
                    f"KernelPCAResult.transform: X_cols mismatch "
                    f"(fit: {self.X_cols}, got: {X_cols})")
            X_cols = list(self.X_cols)  # reorder

        k = self.n_components
        n_q = int(df.count())
        if n_q == 0:
            raise ValueError("KernelPCAResult.transform: df is empty")

        mg = memory_guard or getattr(df, "_memory_guard", None)
        d = self.X_fit.shape[1]
        if batch_size is None:
            try:
                avail = mg.available_bytes() if mg is not None else (8 << 20)
            except Exception:
                avail = (8 << 20)
            target = min(8 << 20, avail // 20)
            bsz = max(1, target // max(1, (d + k + 4) * 16))
            bsz = min(int(bsz), n_q)
        else:
            bsz = int(batch_size)
        bsz = max(1, bsz)

        # Precompute scaled alphas: eigenvectors / sqrt(eigvals) (skip 0)
        nonzero = np.flatnonzero(self.eigenvalues)
        scaled = np.zeros_like(self.eigenvectors)
        scaled[:, nonzero] = (self.eigenvectors[:, nonzero]
                              / np.sqrt(self.eigenvalues[nonzero]))

        centerer = _KernelCenterer()
        centerer.K_fit_rows_ = self.K_fit_rows
        centerer.K_fit_all_ = self.K_fit_all

        from cadw.dataset.dataset import Dataset
        from cadw import CadwDataFrame

        schema = pa.schema([pa.field(f"kp_{i}", pa.float64())
                            for i in range(k)])

        def _iter_q_batches():
            for qb in df._iter_filtered_batches(batch_size=bsz):
                if mg is not None and hasattr(mg, "check_batch"):
                    mg.check_batch(qb, name="kernel_pca.Q",
                                   extra_factor=3.0)
                Q = _batch_to_X(qb, X_cols)  # (b, d)
                if Q.shape[0] == 0:
                    continue
                # K_new = kernel(Q, X_fit): (b, n_fit)
                K_new = pairwise_kernel(
                    Q, self.X_fit, kernel=self.kernel,
                    gamma=self.gamma, degree=self.degree,
                    coef0=self.coef0, memory_guard=mg,
                )
                K_centered = centerer.transform(K_new)  # (b, n_fit)
                Z = K_centered @ scaled               # (b, k)
                cols = [pa.array(Z[:, i], type=pa.float64())
                        for i in range(k)]
                yield pa.RecordBatch.from_arrays(cols, schema=schema)

        ds = Dataset.from_batches(schema, _iter_q_batches(),
                                   df._new_dataset_dir())
        return CadwDataFrame(ds, memory_guard=mg,
                             spill_dir=getattr(df, "_spill_dir", None))


def fit_kernel_pca(
    df,
    X_cols,
    *,
    n_components: Optional[int] = None,
    kernel: str = "linear",
    gamma: Optional[float] = None,
    degree: int = 3,
    coef0: float = 1.0,
    remove_zero_eig: bool = False,
    batch_size: Optional[int] = None,
    memory_guard=None,
) -> KernelPCAResult:
    """Fit Kernel PCA (sklearn parity; intrinsic O(n^2) fit transient).

    Honest bounded_by_domain {n: 1e4} (DEC-075): K = n*n*f64; + eigh copy.
    Above 1e4, raises MemoryLimitExceeded with actionable options.

    Args:
      df: CadwDataFrame with X_cols numeric.
      X_cols: list[str].
      n_components: k (None -> use all; sklearn drops zero eig then).
      kernel: "linear" (default, sklearn parity) | "rbf" | "poly" |
        "laplacian". (sigmoid/cosine deferred v0.9.201+.)
      gamma: rbf/poly/laplacian; None -> 1/d (sklearn).
      degree, coef0: poly only.
      remove_zero_eig: drop eigenvectors with eigenvalue == 0.

    Returns KernelPCAResult (fit_transform, transform, release).
    """
    if not X_cols:
        raise ValueError("fit_kernel_pca: X_cols is empty")
    if kernel not in _SUPPORTED_KERNELS:
        raise ValueError(
            f"fit_kernel_pca: kernel must be one of {_SUPPORTED_KERNELS}; "
            f"got {kernel!r}")
    if degree < 1:
        raise ValueError(
            f"fit_kernel_pca: degree must be >= 1; got {degree}")
    if gamma is not None and gamma <= 0:
        raise ValueError(
            f"fit_kernel_pca: gamma must be > 0; got {gamma}")
    missing = [c for c in X_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"fit_kernel_pca: columns missing: {missing}; "
            f"available: {df.columns}")

    mg = memory_guard or getattr(df, "_memory_guard", None)
    n_total = int(df.count())
    d = len(X_cols)
    if n_total < 2:
        raise ValueError(
            f"fit_kernel_pca: need n >= 2; got n={n_total}")

    # Honest intrinsic bound (DEC-075)
    if n_total > _MAX_N_INTRINSIC:
        kernel_gb = 2.0 * n_total * n_total * 8.0 / 1e9
        raise MemoryLimitExceeded(
            f"fit_kernel_pca: n={n_total} > intrinsic bound "
            f"{_MAX_N_INTRINSIC} (O(n^2) kernel + eigh). "
            f"Options: (a) subsample df to n<={_MAX_N_INTRINSIC}; "
            f"(b) use Nystroem approximation (v0.9.201+); "
            f"(c) if physical RAM allows, expected fit transient "
            f"~{kernel_gb:.1f} GB (raise MemoryGuard budget).",
            model="KernelPCA", params={"n": n_total, "d": d},
        )

    if batch_size is None:
        batch_size = max(1, min(n_total, 1000))
    batch_size = max(1, min(int(batch_size), n_total))

    # Pass 1: X -> in-memory (bounded by n<=1e4)
    X_parts = []
    for b in df._iter_filtered_batches(batch_size=batch_size):
        if mg is not None and hasattr(mg, "check_batch"):
            mg.check_batch(b, name="kernel_pca.X", extra_factor=2.0)
        X_parts.append(_batch_to_X(b, X_cols))
    X_fit = (np.vstack(X_parts) if len(X_parts) > 1
             else X_parts[0] if X_parts else np.empty((0, d)))

    if X_fit.shape[0] != n_total:
        raise RuntimeError(
            f"fit_kernel_pca: X rows mismatch "
            f"({X_fit.shape[0]} != {n_total})")
    if np.isnan(X_fit).any():
        raise ValueError(
            "fit_kernel_pca: X contains NaN; impute first "
            "(fit_knn_imputer) or use complete-case.")

    # Effective gamma
    gamma_eff = (1.0 / d if gamma is None else float(gamma))

    # Pass 2: K = kernel(X, X) — intrinsic O(n^2)
    if mg is not None and hasattr(mg, "check_state"):
        mg.check_state(
            n_total * n_total * 8 * 2, name="kernel_pca.K_fit_transient")
    K = pairwise_kernel(
        X_fit, X_fit, kernel=kernel, gamma=gamma_eff,
        degree=degree, coef0=coef0, memory_guard=mg,
    )

    # Centerer
    centerer = _KernelCenterer().fit(K)
    K_centered = centerer.transform(K)
    del K

    # Determine n_components
    if n_components is None:
        n_comp = n_total  # all dims (zeros dropped below)
    else:
        n_comp = min(n_total, int(n_components))
        if n_comp < 1:
            raise ValueError(
                f"fit_kernel_pca: n_components must be >= 1; "
                f"got {n_components}")

    eigvals, eigvecs = eigh_topk(K_centered, n_comp, which="LA",
                                  memory_guard=mg)
    # sklearn _check_psd_eigenvalues(enable_warnings=False) parity (P7c):
    #   1. clip negatives to 0
    #   2. clip small positives (0 < λ < small_pos_ratio * max_λ) to 0
    #      small_pos_ratio = 1e-12 (float64)
    # This makes remove_zero_eig / n_components=None keep only real rank.
    eigvals = np.maximum(eigvals, 0.0)
    if eigvals.size > 0:
        max_eig = float(eigvals.max())
        small_pos_ratio = 1e-12
        tol = small_pos_ratio * max_eig
        eigvals = np.where(
            (eigvals > 0) & (eigvals < tol), 0.0, eigvals)
    # Deterministic sign (sklearn svd_flip)
    eigvecs = _svd_flip(eigvecs)
    # Sort desc (already from eigh_topk LA)
    # Drop zero eigenvalues (sklearn: remove_zero_eig OR n_components=None)
    if remove_zero_eig or n_components is None:
        mask = eigvals > 0
        eigvals = eigvals[mask]
        eigvecs = eigvecs[:, mask]
    k_eff = eigvals.shape[0]
    if k_eff < 1:
        raise ValueError(
            "fit_kernel_pca: all eigenvalues are zero; data may be "
            "constant (variance too low)")

    result = KernelPCAResult(
        X_fit=X_fit,
        X_cols=list(X_cols),
        K_fit_rows=centerer.K_fit_rows_,
        K_fit_all=float(centerer.K_fit_all_),
        eigenvalues=eigvals.astype(np.float64),
        eigenvectors=eigvecs.astype(np.float64),
        n_components=int(k_eff),
        kernel=kernel,
        gamma=float(gamma_eff),
        degree=int(degree),
        coef0=float(coef0),
        n_train=int(n_total),
    )

    try:
        log_fit("kernel_pca", state_bytes=result.state_nbytes())
    except Exception:
        pass

    return result
