"""BIRCH CF-tree streaming clustering (D.5.4, v0.9.40).

Reference: Zhang, Ramakrishnan, Livny, *BIRCH: An Efficient Data Clustering
Method for Very Large Databases*, SIGMOD 1996.
See .pingis/algorithm_notes/d5_4_birch.md.

CF = (N, LS, SS) is the sufficient statistic of a sub-cluster: N points,
vector sum LS, sum of squares SS. Streaming insertion walks the tree from
the root to the closest child whose radius after insertion stays below
`threshold`; if no child absorbs the point a new leaf CF is created and the
node is split when its fan-out exceeds `branching_factor`. On close() the
leaf CFs are agglomerated into k final clusters (the paper's phase 3).

State is O(#CF nodes) in `(2d+1)` floats per node, independent of N. No
point is retained after insertion -- the whole model is the tree itself.
"""
from __future__ import annotations

from cadw.ml._common import _stack_numeric as _stack_numeric_safe

from typing import Optional

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.ml._logging import log_fit




class _CF:
    """A cluster feature: (N, LS, SS)."""

    __slots__ = ("n", "ls", "ss", "children")

    def __init__(self, d):
        self.n = 0.0
        self.ls = np.zeros(d, dtype=np.float64)
        self.ss = np.zeros(d, dtype=np.float64)
        self.children = []

    @property
    def centroid(self):
        return self.ls / max(self.n, 1e-300)

    def radius_after(self, x):
        """Radius of the sub-cluster if x were added."""
        n2 = self.n + 1
        ls2 = self.ls + x
        ss2 = self.ss + x * x
        # mean squared distance from centroid = (SS - 2*<LS,mu> + N*<mu,mu>)/N
        # = SS/N - <mu,mu>, with mu = LS/N
        mu2 = ls2 / n2
        mse = float(np.sum(ss2) / n2 - np.sum(mu2 * mu2))
        return float(np.sqrt(max(mse, 0.0)))

    def add(self, x):
        self.n += 1
        self.ls += x
        self.ss += x * x

    def absorb(self, other):
        self.n += other.n
        self.ls += other.ls
        self.ss += other.ss


class _Node:
    """A leaf node holding CFs; kept simple with one level for the model."""

    __slots__ = ("entries", "branching_factor")

    def __init__(self, branching_factor):
        self.entries = []
        self.branching_factor = int(branching_factor)


class BIRCHModel:
    """Fitted BIRCH model: leaf CFs, final centroids and labels."""

    def __init__(self, cf_root, n_clusters, centers, feature_cols, params):
        self.cf_root_ = cf_root
        self.n_clusters_ = int(n_clusters)
        self.centers_ = np.asarray(centers, dtype=np.float64)
        self.feature_cols_ = list(feature_cols)
        self.params_ = dict(params)
        self.fitted_ = True

    def _cf_count(self):
        return int(sum(1 for _ in self._iter_cfs()))

    def _iter_cfs(self):
        stack = [self.cf_root_]
        while stack:
            node = stack.pop()
            for cf in node.entries:
                yield cf
                if cf.children:
                    stack.extend(cf.children)

    def state_nbytes(self):
        d = self.centers_.shape[1]
        per_cf = (2 * d + 1) * 8 + 32
        return int(self._cf_count() * per_cf + self.centers_.nbytes + 512)

    def to_dict(self):
        return {"n_clusters": self.n_clusters_, "n_cf": self._cf_count(),
                "params": self.params_}

    def _assign(self, X):
        d2 = (np.sum(X * X, axis=1)[:, None]
              - 2.0 * X @ self.centers_.T
              + np.sum(self.centers_ * self.centers_, axis=1)[None, :])
        return np.argmin(np.maximum(d2, 0.0), axis=1).astype(np.int64)

    def predict(self, df, memory_guard=None):
        mg = memory_guard or df._memory_guard
        out = []
        for b in df._iter_filtered_batches():
            mg.check_batch(b, name="birch.predict", extra_factor=3.0)
            out.append(self._assign(_stack_numeric_safe(b, self.feature_cols_)))
        return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def _insert(root, x, threshold, branching_factor, d):
    """Streaming insert of one point into a one-level CF tree."""
    entries = root.entries
    if entries:
        n = np.fromiter((c.n for c in entries), dtype=np.float64, count=len(entries))
        LS = np.array([c.ls for c in entries], dtype=np.float64)
        SS = np.array([c.ss for c in entries], dtype=np.float64)
        n2 = n + 1.0
        ls2 = LS + x
        ss2 = SS + x * x
        mu2 = ls2 / n2[:, None]
        mse = ss2.sum(axis=1) / n2 - (mu2 * mu2).sum(axis=1)
        r = np.sqrt(np.maximum(mse, 0.0))
        ok = r <= threshold
        if ok.any():
            r_masked = np.where(ok, r, np.inf)
            best = int(np.argmin(r_masked))
            entries[best].add(x)
            return
    cf = _CF(d)
    cf.add(x)
    entries.append(cf)
    if len(entries) > branching_factor:
        # split: merge the two closest CFs and keep the rest -- keeps state
        # bounded O(branching_factor) and is the paper's split policy.
        #
        # v0.9.116 (DEC-035): row-by-row argmin. Prior code materialised
        # `(bf, bf, d)` = 4 MB intermediate per split (bf=50, d=200); split
        # fires ~per-row on random-normal data → 16 GB churn → arena
        # fragmentation → ~5-6 MB intercept in test_batch_coef_isolated.
        # New: per-i `(bf-i-1, d)` sum (peak 80 KB), same sum-over-d order
        # → bit-identical (i,j) minimal pair.
        centers = np.array([c.centroid for c in entries])
        n_e = len(entries)
        best_dist = np.inf
        best_i, best_j = 0, 1
        for i in range(n_e - 1):
            d2_i = np.sum((centers[i + 1:] - centers[i]) ** 2, axis=1)
            j_local = int(np.argmin(d2_i))
            if d2_i[j_local] < best_dist:
                best_dist = d2_i[j_local]
                best_i, best_j = i, i + 1 + j_local
        entries[best_i].absorb(entries[best_j])
        del entries[best_j]


def fit_birch(df, X_cols, *, n_clusters=8, threshold=0.5,
              branching_factor=50, max_iter=20,
              memory_guard: Optional[MemoryGuard] = None,
              batch_size: "int | None" = None):
    """Fit a BIRCH CF tree streaming over the input, then agglomerate.

    v0.9.111 (DEC-033 OOC): `batch_size` kwarg. None (default) derives
    from MemoryGuard via `get_batch_rows` with `state_est` upper bound
    (cf_count <= branching_factor by 1-level tree split policy).
    Int = explicit override (e.g. for bit-identical multi-batch tests).
    """
    cols = [X_cols] if isinstance(X_cols, str) else list(X_cols)
    if not cols:
        raise ValueError("fit_birch: X_cols must be non-empty")
    if n_clusters < 1:
        raise ValueError("fit_birch: n_clusters must be >= 1")
    if threshold <= 0:
        raise ValueError("fit_birch: threshold must be > 0")
    if branching_factor < 2:
        raise ValueError("fit_birch: branching_factor must be >= 2")
    for c in cols:
        if c not in df._columns:
            raise KeyError(f"fit_birch: column {c!r} not in {df._columns}")
    mg = memory_guard or df._memory_guard
    d = len(cols)
    # v0.9.111 (DEC-033): derive batch_size from guard if not given.
    # state_est = upper bound on tree size (cf_count <= branching_factor
    # by split policy) + finalize (k*d*f64, k <= min(n_clusters, bf))
    # + header 512.
    if batch_size is None:
        k_bound = min(int(n_clusters), int(branching_factor))
        state_est = int(
            branching_factor * ((2 * d + 1) * 8 + 32)
            + k_bound * d * 8
            + 512
        )
        row_size = mg.estimate_row_size_from_schema(df._dataset.schema)
        batch_size = mg.get_batch_rows(
            row_size_bytes=row_size,
            expansion_factor=3.0,   # _stack_numeric: X + column_stack parts
            state_bytes=state_est,
        )
    root = _Node(branching_factor)
    n = 0
    for b in df._iter_filtered_batches(batch_size=batch_size):
        mg.check_batch(b, name="birch.fit", extra_factor=3.0)
        X = _stack_numeric_safe(b, cols)
        if np.isnan(X).any():
            raise ValueError("fit_birch: nulls in features")
        for row in X:
            _insert(root, row, threshold, branching_factor, d)
            n += 1
        mg.check_state(len(root.entries) * (2 * d + 1) * 8 + 256,
                       name="birch.state")
    if n == 0:
        raise ValueError("fit_birch: empty input")
    cfs = list(root.entries)
    if not cfs:
        raise ValueError("fit_birch: no CFs built")
    k = min(int(n_clusters), len(cfs))
    # phase 3: kmeans on CF centroids, initialised with weighted k-means++
    cents = np.array([c.centroid for c in cfs])
    wts = np.array([c.n for c in cfs])
    rng = np.random.RandomState(0)
    order = np.argsort(-wts)
    centers = [cents[order[0]]]
    for _ in range(1, k):
        d2 = np.min(np.sum((cents[:, None, :] - np.array(centers)[None, :, :]) ** 2,
                           axis=2), axis=1)
        probs = d2 * wts
        s = probs.sum()
        if s <= 0:
            j = int(rng.randint(len(cents)))
        else:
            j = int(rng.choice(len(cents), p=probs / s))
        centers.append(cents[j])
    centers = np.array(centers)
    for _ in range(int(max_iter)):
        d2 = (np.sum(cents * cents, axis=1)[:, None]
              - 2.0 * cents @ centers.T
              + np.sum(centers * centers, axis=1)[None, :])
        lab = np.argmin(np.maximum(d2, 0.0), axis=1)
        new = centers.copy()
        for j in range(k):
            mask = lab == j
            if mask.any():
                new[j] = (cents[mask] * wts[mask, None]).sum(axis=0) / wts[mask].sum()
        if np.allclose(new, centers):
            centers = new
            break
        centers = new
    model = BIRCHModel(root, k, centers, cols,
                       {"n_clusters": k, "threshold": threshold,
                        "branching_factor": branching_factor,
                        "max_iter": max_iter})
    log_fit("birch", state_bytes=model.state_nbytes())
    mg.check_state(model.state_nbytes(), name="birch.model")
    return model
