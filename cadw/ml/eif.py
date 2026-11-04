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

    def state_nbytes(self):
        """Persistent output state: 3 n-sized arrays + 256 header.

        v0.9.114: honest O(n). decision_scores (n)*8 + labels (n)*8 +
        window_scores (n_w~n)*8 + 256. Forest is hyperparameter-bounded
        (n_estimators*psi nodes, Python objects, ~O(model) not O(n))
        and is not included here — see class docstring. bounded_by_domain
        n<=1e6 (CoxPH v0.9.114 precedent) accepts the O(n) output.
        """
        return int(
            self.decision_scores.nbytes
            + self.labels.nbytes
            + self.window_scores.nbytes
            + 256
        )


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


# ── Streaming helpers (v0.9.91-S6b) ──────────────────────────────

def _resolve_chunk_nw_eif(d: int, guard=None) -> int:
    """Chunk size so peak RAM per chunk stays within budget.

    8 MB default; if guard exposes state_limit_bytes, cap to
    state_limit // 8. bytes_per_window ≈ 8 * d * 2 (window + score
    temp).
    """
    default_target = 8 * 1024 * 1024
    target_bytes = default_target
    if guard is not None:
        sl = getattr(guard, "state_limit_bytes", None)
        if isinstance(sl, (int, float)) and sl > 0:
            target_bytes = min(int(target_bytes), max(1, int(sl) // 8))
    bytes_per_window = 8 * max(int(d), 1) * 2
    n = max(100, target_bytes // bytes_per_window)
    return min(10_000, n)


def _iter_window_chunks_eif(X: np.ndarray, w: int, stride: int,
                            chunk_nw: int):
    """Yield (chunk(nb, w*c), start). No full (n_w, w*c) materialize."""
    if X.ndim == 1:
        X = X[:, None]
    n, c = X.shape
    n_w = (n - w) // stride + 1
    if n_w < 1:
        return
    offsets = np.arange(w, dtype=np.int64)
    for start in range(0, n_w, chunk_nw):
        end = min(start + chunk_nw, n_w)
        abs_starts = np.arange(start, end, dtype=np.int64) * stride
        idx = abs_starts[:, None] + offsets[None, :]
        yield X[idx].reshape(end - start, w * c), start


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

    # v0.9.87-fix1b (audit HIGH-1): guard check BEFORE
    # _make_windows allocation. Analytic shape only.
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    n_w_pre = (n - window_size) // stride + 1
    d_pre = window_size * c
    if n_w_pre < 1:
        raise ValueError(
            f"n_windows={n_w_pre} < 1 (n={n}, window_size={window_size}, "
            f"stride={stride})")
    if max_samples == "auto":
        _ms_pre = min(256, n_w_pre)
    else:
        _ms_pre = min(int(max_samples), n_w_pre)
    if extension_level is None:
        _ext_pre = d_pre - 1
    else:
        _ext_pre = int(extension_level)
    if not (0 <= _ext_pre <= d_pre - 1):
        raise ValueError(
            f"extension_level must be in [0, {d_pre - 1}], "
            f"got {extension_level}")

    if guard is not None:
        # v0.9.91-S6b-fix3: peak under chunked fit.
        #   gathered  = n_est * psi * d * 8   (unique fit windows)
        #   w_scores  = n_w * 8               (output)
        #   chunk buf = chunk_nw * d * 8 * 2
        _psi_est = max_samples if isinstance(max_samples, int) else 256
        _psi_est = min(_psi_est, n_w_pre)
        _chunk_est = _resolve_chunk_nw_eif(d_pre, guard)
        est_bytes = (int(n_estimators) * _psi_est * d_pre * 8
                     + n_w_pre * 8
                     + _chunk_est * d_pre * 8 * 2)
        if hasattr(guard, "check_state") and callable(guard.check_state):
            guard.check_state(est_bytes, name="eif.windows")
        else:
            limit_mb = _probe_memory_limit_mb(guard)
            if limit_mb is not None and limit_mb > 0:
                if est_bytes > limit_mb * 1024 * 1024:
                    from cadw.core.memory_guard import MemoryLimitExceeded
                    raise MemoryLimitExceeded(
                        f"EIF estimated {est_bytes} bytes > "
                        f"guard {limit_mb * 1024 * 1024:.0f} bytes")

    # v0.9.91-S6b-fix3: bit-identical chunked fit.
    # Pre-sample per-tree indices via rng.choice(n_w, size=psi,
    # replace=False) with same rng sequence as _EIFForest.fit
    # (one choice call per tree); gather all unique sampled windows
    # in a single chunked pass; build trees manually with rng state
    # advancing exactly as _EIFForest.fit does (choice, then pass
    # same rng to _EIFTree.fit).
    # Peak = n_est*psi*d*8 (unique gathered windows, hyperparameter-
    # bound) + n_w*8 (scores) + chunk*d*8*2.
    n_w = n_w_pre
    d = d_pre
    if max_samples == "auto":
        max_samples = min(256, n_w)
    if max_samples > n_w:
        max_samples = n_w
    if extension_level is None:
        extension_level = d - 1

    chunk_nw = _resolve_chunk_nw_eif(d, guard)
    psi_actual = min(max_samples, n_w)

    # Phase 1: pre-sample per-tree indices (same rng sequence as
    # _EIFForest.fit: fresh default_rng(random_state), one choice()
    # per tree, size=psi, replace=False).
    rng_fit = np.random.default_rng(random_state)
    tree_indices: list = []
    unique_set: set = set()
    for _ in range(int(n_estimators)):
        idx = rng_fit.choice(n_w, size=psi_actual, replace=False)
        tree_indices.append(idx)
        unique_set.update(int(i) for i in idx)
    unique_sorted = np.array(sorted(unique_set), dtype=np.int64)
    pos_map = {int(v): j for j, v in enumerate(unique_sorted)}

    # Phase 2: one chunked pass, gather all unique sampled windows.
    gathered = np.empty((len(unique_sorted), d), dtype=np.float64)
    for chunk, start in _iter_window_chunks_eif(
            X, window_size, stride, chunk_nw):
        nb = chunk.shape[0]
        end = start + nb
        lo = int(np.searchsorted(unique_sorted, start, side='left'))
        hi = int(np.searchsorted(unique_sorted, end, side='left'))
        for k in range(lo, hi):
            g = int(unique_sorted[k])
            gathered[k] = chunk[g - start]

    # Phase 3: build trees with rng state identical to _EIFForest.fit.
    forest = _EIFForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        extension_level=extension_level,
        random_state=random_state,
    )
    forest.max_depth = int(np.ceil(np.log2(max(psi_actual, 2))))
    forest._c_psi = _c(psi_actual)
    forest.trees = []
    rng_build = np.random.default_rng(random_state)
    for t in range(int(n_estimators)):
        # Advance rng_build exactly as _EIFForest.fit would
        # (result discarded — we already have tree_indices[t]).
        _ = rng_build.choice(n_w, size=psi_actual, replace=False)
        local_idx = np.fromiter(
            (pos_map[int(i)] for i in tree_indices[t]),
            dtype=np.int64, count=psi_actual)
        sample_t = gathered[local_idx]
        tree = _EIFTree().fit(
            sample_t, np.arange(psi_actual, dtype=np.int64), 0,
            forest.max_depth, extension_level, rng_build)
        forest.trees.append(tree)

    # Chunked scoring: window_scores is (n_w,) output — acceptable.
    window_scores = np.empty(n_w, dtype=np.float64)
    for chunk, start in _iter_window_chunks_eif(
            X, window_size, stride, chunk_nw):
        cs = forest.score(chunk)
        window_scores[start:start + len(cs)] = cs

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

