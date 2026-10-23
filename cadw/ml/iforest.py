"""Isolation Forest (D.1.1, Phase D, v0.9.31).

Reference: Liu, Ting, Zhou. *Isolation Forest*. ICDM 2008.
DOI:10.1109/ICDM.2008.17. See sklearn.ensemble.IsolationForest (BSD)
for a reference implementation, used only for validation in tests.

Design: docs/decisions/d1_1_iforest.md.
State: O(n_trees * subsample_size) nodes, independent of N.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard


def _harmonic(n: int) -> float:
    if n <= 0:
        return 0.0
    if n < 10:
        return float(sum(1.0 / i for i in range(1, n + 1)))
    return float(math.log(n) + 0.5772156649015329)


def _c(n: int) -> float:
    if n <= 1:
        return 0.0
    return 2.0 * _harmonic(n - 1) - 2.0 * (n - 1) / n


class _Tree:
    """Isolation tree stored as flat lists (compact state)."""
    __slots__ = ("feature", "threshold", "left", "right", "size")

    def __init__(self) -> None:
        self.feature = []
        self.threshold = []
        self.left = []
        self.right = []
        self.size = []

    def _add(self, size: int) -> int:
        i = len(self.feature)
        self.feature.append(-1)
        self.threshold.append(0.0)
        self.left.append(-1)
        self.right.append(-1)
        self.size.append(int(size))
        return i

    def nbytes_estimate(self) -> int:
        return len(self.feature) * 40

    def build(self, X: np.ndarray, depth: int, max_depth: int, rng) -> int:
        n, d = X.shape
        idx = self._add(n)
        if n <= 1 or depth >= max_depth or d == 0:
            return idx
        f = -1; cmin = cmax = 0.0
        for _ in range(5):
            f = int(rng.randint(0, d))
            col = X[:, f]
            cmin = float(col.min()); cmax = float(col.max())
            if cmax > cmin:
                break
        else:
            return idx
        thr = float(rng.uniform(cmin, cmax))
        mask = X[:, f] < thr
        if mask.all() or not mask.any():
            return idx
        self.feature[idx] = f
        self.threshold[idx] = thr
        li = self.build(X[mask], depth + 1, max_depth, rng)
        ri = self.build(X[~mask], depth + 1, max_depth, rng)
        self.left[idx] = li
        self.right[idx] = ri
        return idx

    def path_length(self, x: np.ndarray) -> float:
        node = 0; depth = 0.0
        while True:
            f = self.feature[node]
            if f < 0:
                return depth + _c(self.size[node])
            node = self.left[node] if x[f] < self.threshold[node] else self.right[node]
            depth += 1.0

    def path_length_batch(self, X: np.ndarray) -> np.ndarray:
        """Vectorized path lengths for a batch of rows.

        Traverses level-by-level: at each step, partition rows by the
        current node's split, updating a node-index array. Replaces a
        per-row Python loop (was ~220 s for 1M rows) with O(N * depth)
        vectorized NumPy ops.
        """
        n = X.shape[0]
        nodes = np.zeros(n, dtype=np.int64)
        depths = np.zeros(n, dtype=np.float64)
        active = np.ones(n, dtype=bool)
        while True:
            if not active.any():
                break
            # Look up feature/threshold for each row's current node.
            fs = np.array(self.feature, dtype=np.int64)[nodes]
            ths = np.array(self.threshold, dtype=np.float64)[nodes]
            leaves = fs < 0
            # Terminal rows: accumulate c(size[node]).
            if leaves.any():
                leaf_nodes = nodes[leaves]
                sizes = np.array(self.size, dtype=np.int64)[leaf_nodes]
                # c(n) scalar per node — compute via python but on leaves only
                cs = np.array([_c(int(s)) for s in sizes], dtype=np.float64)
                depths[leaves] += cs
                active &= ~leaves
            # Non-leaf rows: descend.
            if not active.any():
                break
            xv = X[np.arange(n), np.where(active, fs, 0)]
            go_left = xv < ths
            left = np.array(self.left, dtype=np.int64)[nodes]
            right = np.array(self.right, dtype=np.int64)[nodes]
            new_nodes = np.where(go_left, left, right)
            nodes = np.where(active, new_nodes, nodes)
            depths += active.astype(np.float64)
        return depths




def _path_batch(X, feature, threshold, left, right, size, c_cache):
    """Vectorized path length using pre-cached per-tree numpy arrays."""
    n = X.shape[0]
    nodes = np.zeros(n, dtype=np.int64)
    depths = np.zeros(n, dtype=np.float64)
    active = np.ones(n, dtype=bool)
    while active.any():
        fs = feature[nodes]
        ths = threshold[nodes]
        leaves = fs < 0
        if leaves.any():
            leaf_nodes = nodes[leaves]
            sizes = size[leaf_nodes]
            cs = np.array(
                [c_cache.setdefault(int(s), _c(int(s))) for s in sizes],
                dtype=np.float64,
            )
            depths[leaves] += cs
            active &= ~leaves
            if not active.any():
                break
        xv = X[np.arange(n), fs]
        go_left = xv < ths
        nodes = np.where(go_left, left[nodes], right[nodes])
        depths += active.astype(np.float64)
    return depths

class IsolationForestModel:
    """Fitted Isolation Forest. Immutable state."""

    def __init__(self, trees, feature_names, subsample_size, threshold=0.5) -> None:
        self.trees_ = list(trees)
        self.feature_names_ = list(feature_names)
        self.subsample_size_ = int(subsample_size)
        self.n_trees_ = len(self.trees_)
        self.c_norm_ = _c(self.subsample_size_)
        self.threshold_ = float(threshold)
        # v0.9.31-fix2: cache per-tree arrays as numpy once. The
        # scoring loop recreated np.array(self.feature) etc. per level —
        # dominant overhead at N=1M. Arrays never change post-fit.
        self._feature = [np.asarray(t.feature, dtype=np.int64) for t in self.trees_]
        self._threshold = [np.asarray(t.threshold, dtype=np.float64) for t in self.trees_]
        self._left = [np.asarray(t.left, dtype=np.int64) for t in self.trees_]
        self._right = [np.asarray(t.right, dtype=np.int64) for t in self.trees_]
        self._size = [np.asarray(t.size, dtype=np.int64) for t in self.trees_]

    def state_nbytes(self) -> int:
        return sum(t.nbytes_estimate() for t in self.trees_)

    def _score_matrix(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        acc = np.zeros(n, dtype=np.float64)
        c_cache: dict[int, float] = {}
        for ti in range(self.n_trees_):
            acc += _path_batch(
                X, self._feature[ti], self._threshold[ti],
                self._left[ti], self._right[ti], self._size[ti], c_cache,
            )
        acc /= max(1, self.n_trees_)
        if self.c_norm_ <= 0:
            return np.full(n, 0.5, dtype=np.float64)
        return np.power(2.0, -acc / self.c_norm_)

    def score_samples(self, df, X_cols=None, memory_guard=None) -> np.ndarray:
        """Per-batch scoring. Peak RAM O(batch) (v0.9.31-fix1).

        Prior version concatenated ALL batches to a single numpy array
        (O(N) RAM, 227 MB @1M rows) — a hard violation of the
        <=1-batch-in-RAM contract. Now scores each batch independently
        with check_batch guard.
        """
        cols = list(X_cols) if X_cols is not None else self.feature_names_
        mg = memory_guard
        if mg is None:
            try:
                mg = df._memory_guard
            except Exception:
                mg = MemoryGuard(max_memory_mb=64)
        parts: list[np.ndarray] = []
        for b in df._iter_filtered_batches():
            mg.check_batch(b, name="iforest.score", extra_factor=3.0)
            arrays = []
            for c in cols:
                arr = b.column(c)
                if not (pa.types.is_integer(arr.type) or pa.types.is_floating(arr.type)):
                    raise TypeError(
                        f"iforest: column {c!r} must be numeric; got {arr.type}"
                    )
                arrays.append(arr.cast(pa.float64(), safe=False).to_numpy(zero_copy_only=False))
            X_b = np.column_stack(arrays) if arrays else np.empty((b.num_rows, 0))
            mask = ~np.isnan(X_b).any(axis=1)
            out = np.full(X_b.shape[0], np.nan, dtype=np.float64)
            if mask.any():
                out[mask] = self._score_matrix(X_b[mask])
            parts.append(out)
        if not parts:
            return np.zeros(0, dtype=np.float64)
        return np.concatenate(parts, axis=0)

    def score(self, df, X_cols=None) -> pa.Table:
        s = self.score_samples(df, X_cols)
        return pa.table({"score": pa.array(s, type=pa.float64())})

    def predict(self, df, X_cols=None, threshold=None) -> pa.Table:
        thr = float(threshold) if threshold is not None else self.threshold_
        s = self.score_samples(df, X_cols)
        labels = np.where(np.isnan(s), False, s > thr)
        return pa.table({"is_outlier": pa.array(labels, type=pa.bool_())})


def _dense_from_df(df, cols):
    parts = []
    for b in df._iter_filtered_batches():
        arrays = []
        for c in cols:
            arr = b.column(c)
            if not (pa.types.is_integer(arr.type) or pa.types.is_floating(arr.type)):
                raise TypeError(f"iforest: column {c!r} must be numeric; got {arr.type}")
            arrays.append(arr.cast(pa.float64(), safe=False).to_numpy(zero_copy_only=False))
        parts.append(np.column_stack(arrays))
    if not parts:
        return np.empty((0, len(cols))), np.zeros(0, dtype=bool)
    X = np.concatenate(parts, axis=0)
    return X, ~np.isnan(X).any(axis=1)


def fit_isolation_forest(df, X_cols, *, n_trees=100, subsample_size=256,
                        random_state=None, contamination=0.1,
                        memory_guard=None) -> IsolationForestModel:
    """Fit an Isolation Forest on df[X_cols].

    Reservoir sample of size subsample_size over the streaming batches
    (state independent of N); n_trees trees each on a bootstrap
    subsample of size psi=min(subsample_size, N); scoring is
    O(batch * n_trees * max_depth).
    """
    if not X_cols:
        raise ValueError("fit_isolation_forest: X_cols must be non-empty")
    if n_trees < 1:
        raise ValueError("fit_isolation_forest: n_trees must be >= 1")
    if subsample_size < 2:
        raise ValueError("fit_isolation_forest: subsample_size must be >= 2")
    if not (0.0 < contamination <= 0.5):
        raise ValueError("fit_isolation_forest: 0 < contamination <= 0.5")

    mg = memory_guard or MemoryGuard(max_memory_mb=64)
    cols = list(X_cols)
    rng = np.random.RandomState(0 if random_state is None else int(random_state))

    S = int(subsample_size)
    sample = np.empty((S, len(cols)), dtype=np.float64)
    n_seen = 0; n_filled = 0
    for b in df._iter_filtered_batches():
        mg.check_batch(b, name="iforest.sample", extra_factor=2.0)
        arrays = []
        for c in cols:
            arr = b.column(c)
            if not (pa.types.is_integer(arr.type) or pa.types.is_floating(arr.type)):
                raise TypeError(f"iforest: column {c!r} must be numeric; got {arr.type}")
            arrays.append(arr.cast(pa.float64(), safe=False).to_numpy(zero_copy_only=False))
        X_b = np.column_stack(arrays)
        mask = ~np.isnan(X_b).any(axis=1)
        for row in X_b[mask]:
            if n_filled < S:
                sample[n_filled] = row; n_filled += 1
            else:
                j = int(rng.randint(0, n_seen + 1))
                if j < S:
                    sample[j] = row
            n_seen += 1

    if n_filled < 2:
        raise ValueError("fit_isolation_forest: fewer than 2 valid rows")
    X_fit = sample[:n_filled]
    psi = min(n_filled, S)
    max_depth = max(1, int(math.ceil(math.log2(psi)))) if psi > 1 else 1

    trees = []
    for _ in range(int(n_trees)):
        idx = rng.randint(0, n_filled, size=psi)
        t = _Tree()
        t.build(X_fit[idx], 0, max_depth, rng)
        trees.append(t)

    tmp = IsolationForestModel(trees, cols, psi, threshold=0.5)
    s_fit = tmp._score_matrix(X_fit)
    tmp.threshold_ = float(np.quantile(s_fit, 1.0 - contamination))
    mg.check_state(tmp.state_nbytes(), name="iforest.state")
    return tmp
