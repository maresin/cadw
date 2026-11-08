"""STRAY: streaming TS anomaly detection via robust KNN + EVT.

Talagala, Hyndman, Smith-Miles (2021). "Anomaly Detection in High
Dimensional Data." JCGS 30(2).

Algorithm (STRAY-inspired pure-numpy reimplementation):
  1. Robust standardize: (x - median) / (1.4826 * MAD)
  2. Distance to k-th nearest neighbour (excluding self)
  3. Fit GPD to the exceedances above the (1-tail_fraction) quantile
  4. -log(p_value) per point where p_value = SF(exceedance) * n_tail/n
  5. Anomaly if score > -log(alpha)

Deviations from sktime STRAY (documented):
  - GPD fit: Method of Moments (closed form, numpy-only) instead of MLE
    via scipy.stats.genpareto.fit. Shape xi may differ by up to ~0.1
    for heavy-tailed distributions; ranking is preserved for typical
    cases (verified in tests).
  - KNN: brute-force pairwise (chunked, memory-bounded). Exact O(n^2)
    time. For n > ~20k, use EIF or Sub-PCA which scale better.

Reference: sktime.annotation.STRAY.
Pure numpy — no sklearn/scipy/torch in cadw/ (AST-guard).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class STRAYResult:
    decision_scores: np.ndarray
    labels: np.ndarray
    threshold: float
    k: int
    tail_fraction: float
    alpha: float
    contamination: float | None
    xi: float
    beta: float
    tail_threshold: float
    n_tail: int

    def state_nbytes(self) -> int:
        """Persistent state (bytes), v0.9.140.

        O(n) output arrays (decision_scores, labels = 2n*f64) + 256
        header. bounded_state: false, tier1_core_exception (n<=20k
        documented CPU bound at AP22).
        """
        return int(self.decision_scores.nbytes + self.labels.nbytes
                   + 256)


# v0.9.130 (DEC-054): _probe_memory_limit_mb -> 
# cadw.core.memory_guard.probe_limit_mb (single source of truth).
from cadw.core.memory_guard import probe_limit_mb as _probe_memory_limit_mb


def _kth_nn_distances(X: np.ndarray, k: int,
                      max_mb: float = 100.0) -> np.ndarray:
    """Distance from each row to its k-th nearest neighbour (excluding self).

    Brute-force pairwise, chunked over rows so peak memory stays under
    max_mb (MB). O(n^2 * d) time; exact.
    """
    n, d = X.shape
    if n < 2:
        raise ValueError(f"need n>=2, got {n}")
    k = min(k, n - 1)
    # chunk_size * n * 8 (bytes per dist matrix) <= max_mb * 1e6
    chunk_size = max(1, min(n, int(max_mb * 1e6 / (n * 8))))
    result = np.empty(n, dtype=np.float64)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        diff = X[start:end, None, :] - X[None, :, :]
        d_sq = np.einsum("ijk,ijk->ij", diff, diff)
        d_mat = np.sqrt(np.maximum(d_sq, 0.0))
        rows = np.arange(end - start)
        cols = np.arange(start, end)
        d_mat[rows, cols] = np.inf
        result[start:end] = np.partition(d_mat, k - 1, axis=1)[:, k - 1]
    return result


def _gpd_fit_mom(y: np.ndarray) -> tuple[float, float]:
    """Method of Moments GPD fit. Returns (xi, beta).

    MoM formulas:
      xi = 0.5 * (1 - m1^2 / m2)
      beta = m1 * (1 - xi)
    where m1 = mean, m2 = variance. Clipped to valid GPD range; falls
    back to exponential (xi=0) when degenerate.
    """
    if len(y) < 5:
        return 0.0, max(float(np.mean(y)) if len(y) > 0 else 1.0, 1e-12)
    m1 = float(y.mean())
    var = float(y.var())
    if var < 1e-12 or m1 < 1e-12:
        return 0.0, max(m1, 1e-12)
    ratio = m1 * m1 / var
    xi = 0.5 * (1.0 - ratio)
    xi = float(np.clip(xi, -0.5, 0.49))
    beta = m1 * (1.0 - xi)
    if beta <= 0:
        beta = max(m1, 1e-12)
    return xi, beta


def _gpd_sf(y: np.ndarray, xi: float, beta: float) -> np.ndarray:
    """GPD survival function SF(y) = P(Y > y)."""
    y = np.maximum(y, 0.0)
    if abs(xi) < 1e-9:
        return np.exp(-np.minimum(y / beta, 700.0))
    z = 1.0 + xi * y / beta
    if xi > 0:
        return np.where(z > 0, np.power(np.maximum(z, 1e-12), -1.0 / xi), 0.0)
    # xi < 0: bounded support at -beta/xi
    support = -beta / xi  # positive
    safe_z = np.maximum(z, 1e-12)
    return np.where(y < support, np.power(safe_z, -1.0 / xi), 0.0)


def fit_stray(
    series: np.ndarray,
    *,
    k: int = 50,
    tail_fraction: float = 0.1,
    alpha: float = 0.01,
    contamination: float | None = None,
    max_mb: float = 100.0,
    guard=None,
) -> STRAYResult:
    """STRAY anomaly detection.

    Parameters
    ----------
    series : np.ndarray, shape (n,) or (n, c)
        Univariate or multivariate time series (per-timestamp vectors).
    k : int, default=50
        k-th nearest neighbour used for distance ranking.
    tail_fraction : float, default=0.1
        Fraction of largest KNN distances used to fit GPD.
    alpha : float, default=0.01
        Significance level. Threshold for decision_scores = -log(alpha).
        Ignored if `contamination` is set.
    contamination : float | None, default=None
        If set, threshold = percentile(decision_scores, 100*(1-c)).
    max_mb : float, default=100.0
        Peak-memory bound (MB) for KNN pairwise chunking.
    guard : MemoryGuard | None

    Returns
    -------
    STRAYResult
    """
    x = np.asarray(series, dtype=np.float64)
    if x.ndim == 1:
        X = x.reshape(-1, 1)
    elif x.ndim == 2:
        X = x
    else:
        raise ValueError(f"series must be 1D or 2D, got ndim={x.ndim}")
    n = X.shape[0]
    if n < k + 1:
        raise ValueError(
            f"series length {n} too short for k={k} (need n>=k+1)")
    if not (0.0 < tail_fraction < 1.0):
        raise ValueError(
            f"tail_fraction must be in (0, 1), got {tail_fraction}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    if guard is not None:
        est_bytes = 8 * n * X.shape[1] * min(n, 4 * k)
        limit_mb = _probe_memory_limit_mb(guard)
        if limit_mb is not None and limit_mb > 0:
            if est_bytes > limit_mb * 1024 * 1024:
                from cadw.core.memory_guard import MemoryLimitExceeded
                raise MemoryLimitExceeded(
                    f"STRAY estimated {est_bytes} bytes > "
                    f"guard {limit_mb * 1024 * 1024:.0f} bytes")

    # Robust standardize (per-column)
    med = np.median(X, axis=0)
    mad = 1.4826 * np.median(np.abs(X - med), axis=0)
    # Fallback per column: mad -> std -> 1.0
    std = X.std(axis=0)
    fallback = np.where(mad < 1e-12, std, mad)
    fallback = np.where(fallback < 1e-12, 1.0, fallback)
    z = (X - med) / fallback

    kth = _kth_nn_distances(z, k=k, max_mb=max_mb)

    # GPD tail fit
    sorted_kth = np.sort(kth)
    n_tail = max(int(n * tail_fraction), 20)
    n_tail = min(n_tail, n - 1)
    tail_threshold = float(sorted_kth[-n_tail])
    exceedances = sorted_kth[-n_tail:] - tail_threshold
    exceedances = exceedances[exceedances > 1e-12]
    xi, beta = _gpd_fit_mom(exceedances)

    # -log(p_value) decision scores
    decision_scores = np.zeros(n, dtype=np.float64)
    in_tail = kth > tail_threshold
    if in_tail.any():
        excess = kth[in_tail] - tail_threshold
        sf = _gpd_sf(excess, xi, beta)
        p_value = sf * (n_tail / n)
        p_value = np.clip(p_value, 1e-10, 1.0)
        decision_scores[in_tail] = -np.log(p_value)

    # Threshold
    if contamination is not None:
        thr = float(np.percentile(
            decision_scores, 100.0 * (1.0 - contamination)))
        if thr <= 0:
            thr = float(-np.log(alpha))
    else:
        thr = float(-np.log(alpha))

    labels = (decision_scores > thr).astype(int)

    return STRAYResult(
        decision_scores=decision_scores,
        labels=labels,
        threshold=thr,
        k=k,
        tail_fraction=tail_fraction,
        alpha=alpha,
        contamination=contamination,
        xi=xi,
        beta=beta,
        tail_threshold=tail_threshold,
        n_tail=n_tail,
    )

