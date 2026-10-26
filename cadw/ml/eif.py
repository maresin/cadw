"""Extended Isolation Forest (EIF) anomaly detection.

Hariri, Carrasco Kind, Brunner (2019), IEEE TKDE 31(10).
Generalises Isolation Forest by replacing axis-parallel cuts with
random hyperplanes: at each node draw unit normal v (Gaussian normalized,
extension_level+1 random non-zero components) and random intercept
alpha in [min(x.v), max(x.v)]; split by sign of v.x - alpha.

Anomaly score: s(x) = 2 ** (-E[h(x)] / c(psi)), where h is path length,
c(psi) = 2*H(psi-1) - 2*(psi-1)/psi (H = harmonic).

Windowing (aeon convention): sliding windows of `window_size` timestamps
and `stride`; multivariate channels are stacked into one feature vector
of length window_size * n_channels. Per-window scores are aggregated to
per-timestamp scores (max or mean).

Pure numpy — no sklearn/aeon/torch/scipy in cadw/ (AST-guard
test_no_bench_deps_in_core.py).

References:
  - paper: https://arxiv.org/abs/1811.02141
  - reference impl: https://github.com/sahandha/eif
  - aeon wrapper: aeon.anomaly_detection.ExtendedIsolationForest
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sub_pca import _map_max, _map_mean


@dataclass
class EIFResult:
    """Result of fit_eif.

    Attributes
    ----------
    decision_scores : np.ndarray, shape (n,)
        Per-timestamp anomaly score in [0, 1]; higher = more anomalous.
    labels : np.ndarray, shape (n,), int
        0/1 binary labels (score > threshold).
    threshold : float
        Percentile(100*(1-contamination)) of decision_scores.
    window_scores : np.ndarray, shape (n_windows,)
        Raw per-window anomaly scores in [0, 1].
    n_estimators, max_samples, extension_level : int
        Forest hyperparameters.
    window_size, stride : int
        Windowing parameters.
    contamination : float
    aggregate : str
        "max" or "mean".
    n_channels : int
        Number of channels in input (1 for univariate).
    """
    decision_scores: np.ndarray
    labels: np.ndarray
    threshold: float
    window_scores: np.ndarray
    n_estimators: int
    max_samples: int
    extension_level: int
    window_size: int
    stride: int
    contamination: float
    aggregate: str
    n_channels: int


# ── MemoryGuard probe (same defensive logic as sub_pca) ───────────

def _probe_memory_limit_mb(guard) -> float | None:
    def _is_pos(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0
    MB = ("max_memory_mb", "_max_memory_mb", "limit_mb", "budget_mb",
          "_budget_mb", "max_mb", "_max_mb", "memory_limit_mb",
          "_memory_limit_mb")
    BYTES = ("max_memory_bytes", "_max_memory_bytes", "max_bytes",
             "_max_bytes", "limit_bytes", "_limit_bytes", "budget_bytes",
             "_budget_bytes", "memory_limit_bytes", "_memory_limit_bytes")
    for a in MB:
        v = getattr(guard, a, None)
        if _is_pos(v):
            return float(v)
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
        if not _is_pos(v):
            continue
        if "mb" in kl and any(t in kl for t in ("mem", "budget", "limit", "max")):
            return float(v)
    for k, v in d.items():
        kl = k.lower()
        if not _is_pos(v):
            continue
        if any(t in kl for t in ("mem", "budget", "limit", "max")) and (
                "byte" in kl or kl.endswith("_b")):
            return float(v) / (1024.0 * 1024.0)
    return None


# ── c(psi): expected path length of unsuccessful BST search ───────

def _c(psi: int) -> float:
    if psi <= 1:
        return 0.0
    if psi == 2:
        return 1.0
    h = np.log(psi - 1.0) + 0.5772156649015329  # H(psi-1) approx
    return 2.0 * h - 2.0 * (psi - 1.0) / psi


# ── Random normal vector with extension_level ─────────────────────

def _random_normal(d: int, extension_level: int,
                   rng: np.random.Generator) -> np.ndarray:
    """Unit normal with extension_level+1 non-zero components.

    extension_level=0 -> single +/-1 component (axis-parallel, IForest).
    extension_level=d-1 -> all d components ~ N(0,1)/||.|| (full EIF).
    """
    n_active = extension_level + 1
    v = np.zeros(d, dtype=np.float64)
    idx = rng.choice(d, size=n_active, replace=False)
    v[idx] = rng.standard_normal(n_active)
    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        v[idx[0]] = 1.0
        return v
    return v / norm


# ── Single EIF tree ──────────────────────────────────────────────

class _EIFTree:
    __slots__ = ("X", "v", "alpha", "left", "right", "leaf_size")

    def __init__(self) -> None:
        self.X = None
        self.v = None
        self.alpha = 0.0
        self.left = None
        self.right = None
        self.leaf_size = 0

    def fit(self, X: np.ndarray, indices: np.ndarray, depth: int,
            max_depth: int, extension_level: int,
            rng: np.random.Generator) -> "_EIFTree":
        n = len(indices)
        if depth >= max_depth or n <= 1:
            self.leaf_size = n
            return self
        d = X.shape[1]
        v = _random_normal(d, extension_level, rng)
        proj = X[indices] @ v
        lo, hi = float(proj.min()), float(proj.max())
        if hi - lo < 1e-12:
            self.leaf_size = n
            return self
        alpha = float(rng.uniform(lo, hi))
        left_idx = indices[proj < alpha]
        right_idx = indices[proj >= alpha]
        if len(left_idx) == 0 or len(right_idx) == 0:
            self.leaf_size = n
            return self
        self.v = v
        self.alpha = alpha
        self.left = _EIFTree().fit(X, left_idx, depth + 1, max_depth,
                                   extension_level, rng)
        self.right = _EIFTree().fit(X, right_idx, depth + 1, max_depth,
                                    extension_level, rng)
        return self

    def path_length(self, x: np.ndarray, depth: int = 0) -> float:
        if self.leaf_size > 0:
            return float(depth) + _c(self.leaf_size)
        if float(x @ self.v) < self.alpha:
            return self.left.path_length(x, depth + 1)
        return self.right.path_length(x, depth + 1)


# ── Forest ────────────────────────────────────────────────────────

class _EIFForest:
    def __init__(self, n_estimators: int, max_samples: int,
                 extension_level: int, random_state: int | None) -> None:
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.extension_level = extension_level
        self.random_state = random_state
        self.trees: list[_EIFTree] = []
        self.max_depth: int = 0
        self._c_psi: float = 0.0

    def fit(self, X: np.ndarray) -> "_EIFForest":
        n, d = X.shape
        psi = min(self.max_samples, n)
        self.max_depth = int(np.ceil(np.log2(max(psi, 2))))
        self._c_psi = _c(psi)
        rng = np.random.default_rng(self.random_state)
        self.trees = []
        for _ in range(self.n_estimators):
            idx = rng.choice(n, size=psi, replace=False)
            tree = _EIFTree().fit(X, idx, 0, self.max_depth,
                                  self.extension_level, rng)
            self.trees.append(tree)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        paths = np.zeros(n, dtype=np.float64)
        for tree in self.trees:
            for i in range(n):
                paths[i] += tree.path_length(X[i])
        paths /= self.n_estimators
        if self._c_psi <= 0.0:
            return np.zeros(n, dtype=np.float64)
        return np.power(2.0, -paths / self._c_psi)


# ── Windowing ─────────────────────────────────────────────────────

def _make_windows(X: np.ndarray, window_size: int,
                  stride: int) -> np.ndarray:
    """(n, n_channels) -> (n_windows, window_size*n_channels)."""
    if X.ndim == 1:
        X = X[:, None]
    n, c = X.shape
    n_w = (n - window_size) // stride + 1
    if n_w < 1:
        raise ValueError(
            f"series length {n} < window_size {window_size}")
    idx = np.arange(n_w)[:, None] * stride + np.arange(window_size)[None, :]
    return X[idx].reshape(n_w, window_size * c)


# ── Public API ────────────────────────────────────────────────────

def fit_eif(
    series: np.ndarray,
    *,
    n_estimators: int = 100,
    max_samples: int | str = "auto",
    extension_level: int | None = None,
    window_size: int = 50,
    stride: int = 1,
    contamination: float = 0.1,
    aggregate: str = "max",
    random_state: int | None = 42,
    guard=None,
) -> EIFResult:
    """Extended Isolation Forest anomaly detection with windowing.

    Parameters
    ----------
    series : np.ndarray, shape (n,) or (n, n_channels)
        Univariate or multivariate time series.
    n_estimators : int, default=100
        Number of isolation trees.
    max_samples : int or "auto", default="auto"
        Subsample size per tree. "auto" -> min(256, n_windows).
    extension_level : int or None, default=None
        Number of random non-axis components minus 1 (0 = standard IForest,
        d-1 = fully extended). None -> n_features - 1 (fully extended).
    window_size : int, default=50
        Window length in timestamps.
    stride : int, default=1
        Step between consecutive windows.
    contamination : float, default=0.1
        Fraction expected anomalous; used only for threshold.
    aggregate : str or default="max"
        "max" or "mean" aggregation of window scores to timestamps.
    random_state : int or None, default=42
    guard : MemoryGuard | None

    Returns
    -------
    EIFResult
    """
    X = np.asarray(series, dtype=np.float64)
    if X.ndim == 1:
        n = len(X)
        c = 1
    elif X.ndim == 2:
        n, c = X.shape
    else:
        raise ValueError(f"series must be 1D or 2D, got ndim={X.ndim}")
    if n < window_size:
        raise ValueError(
            f"series length {n} < window_size {window_size}")

    windows = _make_windows(X, window_size, stride)
    n_w, d = windows.shape

    if max_samples == "auto":
        max_samples = min(256, n_w)
    if max_samples > n_w:
        max_samples = n_w

    if extension_level is None:
        extension_level = d - 1
    if not (0 <= extension_level <= d - 1):
        raise ValueError(
            f"extension_level must be in [0, {d - 1}], "
            f"got {extension_level}")

    if guard is not None:
        est_bytes = 8 * n_w * d
        limit_mb = _probe_memory_limit_mb(guard)
        if limit_mb is not None and limit_mb > 0:
            if est_bytes > limit_mb * 1024 * 1024:
                from cadw.core.memory_guard import MemoryLimitExceeded
                raise MemoryLimitExceeded(
                    f"EIF estimated {est_bytes} bytes > "
                    f"guard {limit_mb * 1024 * 1024:.0f} bytes")

    forest = _EIFForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        extension_level=extension_level,
        random_state=random_state,
    ).fit(windows)
    window_scores = forest.score(windows)

    if aggregate == "max":
        scores = _map_max(window_scores, window_size, n, stride=stride)
    elif aggregate == "mean":
        scores = _map_mean(window_scores, window_size, n, stride=stride)
    else:
        raise ValueError(
            f"aggregate must be 'max' or 'mean', got {aggregate!r}")

    threshold = float(np.percentile(
        scores, 100.0 * (1.0 - contamination)))
    labels = (scores > threshold).astype(int)

    return EIFResult(
        decision_scores=scores,
        labels=labels,
        threshold=threshold,
        window_scores=window_scores,
        n_estimators=n_estimators,
        max_samples=max_samples,
        extension_level=extension_level,
        window_size=window_size,
        stride=stride,
        contamination=contamination,
        aggregate=aggregate,
        n_channels=c,
    )

