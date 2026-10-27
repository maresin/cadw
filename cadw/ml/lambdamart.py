"""LambdaMART / lambdarank (D.2.1, Phase D, v0.9.35).

Reference: Burges, *From RankNet to LambdaRank to LambdaMART: An
Overview*, MSR-TR-2010-82; Wu, Burges, Svore, Gao, *Adapting Boosting
for Rank: A Progress Report*, ECIR 2010. LightGBM `objective=lambdarank`
(MIT) is the API reference; LETOR 4.0 / MSLR-WEB10K are the benchmark
sets. See .pingis/algorithm_notes/d2_1_lambdamart.md.

Implementation notes:
- Graded relevance y in {0,1,...,k}; queries given by a query column.
- Per iteration: compute lambda gradients and hessians for every document
  from all pairs within a query (O(n_q^2) per query — LETOR queries are
  tens of documents), then fit one regression tree to (lambda, hessian)
  on quantile-binned features.
- State: scores (N), binned X (N*d uint8), lambda/hess (N each) — O(N*d),
  the same class as the existing GBDT (Newton boosting needs per-row
  gradients). Guarded via check_state before allocation.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.ml._logging import log_fit


def _dcg(gains, k):
    gains = np.asarray(gains, dtype=np.float64)[:k]
    if gains.size == 0:
        return 0.0
    disc = 1.0 / np.log2(np.arange(2, gains.size + 2))
    return float((gains * disc).sum())


def ndcg_at_k(y, scores, k=10):
    """NDCG@k for one query."""
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    gains = 2.0 ** y - 1.0
    ideal = _dcg(np.sort(gains)[::-1], k)
    if ideal <= 0:
        return 1.0
    return _dcg(gains[order], k) / ideal


def _lambda_gradients(scores, y, k, sigma):
    """LambdaRank gradients + hessians for one query, vectorized O(n^2).

    For every pair i<j (only the upper triangle is used):
        dNDCG_ij = |(2^y_i - 2^y_j) * (disc[rank_i] - disc[rank_j])|
        p_ij     = 1 / (1 + exp(s_i - s_j))
        rho_ij   = sigma * p_ij * dNDCG_ij * sign(y_i - y_j)
    lambda_i = sum_j rho_ij - sum_j rho_ji ; hess_i = sum_j w_ij.
    """
    n = scores.shape[0]
    if n < 2:
        return np.zeros(n), np.zeros(n)
    gains = 2.0 ** np.asarray(y, dtype=np.float64) - 1.0
    order = np.argsort(-scores, kind="stable")
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    disc = 1.0 / np.log2(np.arange(2, n + 2))
    d = disc[rank]
    diff = scores[:, None] - scores[None, :]
    P = 1.0 / (1.0 + np.exp(np.clip(diff, -50.0, 50.0)))
    D = np.abs(gains[:, None] - gains[None, :]) * np.abs(d[:, None] - d[None, :])
    Y = np.sign(np.asarray(y, dtype=np.float64)[:, None]
                - np.asarray(y, dtype=np.float64)[None, :])
    upper = np.triu(np.ones((n, n), dtype=bool), 1)
    rho = np.where(upper, sigma * P * D * Y, 0.0)
    w = np.where(upper, sigma * sigma * P * (1.0 - P) * D, 0.0)
    lam = rho.sum(axis=1) - rho.sum(axis=0)
    hess = w.sum(axis=1) + w.sum(axis=0)
    return lam, hess


def _bin_features(X, bins, memory_guard=None):
    """Quantile binning -> uint8 codes (n, d) + bin edges.

    `np.quantile` is O(n log n) in a copy of one column at a time; the
    state check runs first so an oversized X raises instead of allocating.
    """
    n, d = X.shape
    if memory_guard is not None:
        memory_guard.check_state(n * d * 8, name="lambdamart.binning")
    codes = np.empty((n, d), dtype=np.uint8)
    edges = []
    for j in range(d):
        col = X[:, j]
        qs = np.unique(np.quantile(col, np.linspace(0, 1, bins + 1)[1:-1]))
        if qs.size > 254:
            qs = qs[:254]
        e = np.concatenate(([-np.inf], qs, [np.inf]))
        edges.append(e)
        codes[:, j] = np.clip(np.searchsorted(e, col, side="right") - 1,
                              0, len(e) - 2).astype(np.uint8)
    return codes, edges


def _apply_edges(col, e):
    return np.clip(np.searchsorted(e, col, side="right") - 1,
                   0, len(e) - 2).astype(np.uint8)


class _Node:
    __slots__ = ("feat", "thr", "left", "right", "value")

    def __init__(self):
        self.feat = -1
        self.thr = 0
        self.left = None
        self.right = None
        self.value = 0.0


def _build_tree(codes, grad, hess, idx, depth, max_depth, min_child_weight,
                reg_lambda, nbins):
    node = _Node()
    G = grad[idx].sum()
    H = hess[idx].sum()
    node.value = -G / (H + reg_lambda)
    if depth >= max_depth or idx.size < 2 * min_child_weight:
        return node
    d = codes.shape[1]
    best = (0.0, -1, -1)
    parent = G * G / (H + reg_lambda)
    for j in range(d):
        gsum = np.bincount(codes[idx, j], weights=grad[idx], minlength=nbins)
        hsum = np.bincount(codes[idx, j], weights=hess[idx], minlength=nbins)
        gl = np.cumsum(gsum)[:-1]
        hl = np.cumsum(hsum)[:-1]
        gr = G - gl
        hr = H - hl
        gain = gl * gl / (hl + reg_lambda) + gr * gr / (hr + reg_lambda) - parent
        gain = np.where((hl >= min_child_weight) & (hr >= min_child_weight),
                        gain, -np.inf)
        k = int(np.argmax(gain))
        if gain[k] > best[0]:
            best = (float(gain[k]), j, k)
    if best[1] < 0:
        return node
    _, j, k = best
    mask = codes[idx, j] <= k
    if not mask.any() or mask.all():
        return node
    node.feat = j
    node.thr = k
    node.left = _build_tree(codes, grad, hess, idx[mask], depth + 1,
                            max_depth, min_child_weight, reg_lambda, nbins)
    node.right = _build_tree(codes, grad, hess, idx[~mask], depth + 1,
                             max_depth, min_child_weight, reg_lambda, nbins)
    return node


def _predict_tree(node, codes):
    out = np.empty(codes.shape[0], dtype=np.float64)
    stack = [(node, np.arange(codes.shape[0]))]
    while stack:
        n, rows = stack.pop()
        if n.feat < 0:
            out[rows] = n.value
            continue
        go_left = codes[rows, n.feat] <= n.thr
        stack.append((n.left, rows[go_left]))
        stack.append((n.right, rows[~go_left]))
    return out


class LambdaMARTModel:
    """Gradient-boosted ranker with the LambdaRank gradient."""

    def __init__(self, trees, edges, base_score, n_features, params):
        self.trees_ = trees
        self.edges_ = edges
        self.base_score_ = float(base_score)
        self.n_features_ = int(n_features)
        self.params_ = dict(params)
        self.fitted_ = True

    def state_nbytes(self):
        n_nodes = 0
        for t in self.trees_:
            stack = [t]
            while stack:
                nd = stack.pop()
                n_nodes += 1
                if nd.left is not None:
                    stack.append(nd.left)
                    stack.append(nd.right)
        return int(n_nodes * 96 + self.n_features_ * 4096)

    def _raw(self, X):
        X = np.asarray(X, dtype=np.float64)
        codes = np.empty(X.shape, dtype=np.uint8)
        for j, e in enumerate(self.edges_):
            codes[:, j] = _apply_edges(X[:, j], e)
        s = np.full(X.shape[0], self.base_score_, dtype=np.float64)
        for t in self.trees_:
            s += _predict_tree(t, codes)
        return s

    def predict(self, df, X_cols=None):
        from cadw.ml.gbdt import _stack_numeric
        cols = list(X_cols) if X_cols is not None else list(self.feature_names_)
        parts = []
        for b in df._iter_filtered_batches():
            parts.append(self._raw(_stack_numeric(b, cols)))
        arr = np.concatenate(parts) if parts else np.zeros(0)
        return pa.table({"score": pa.array(arr, type=pa.float64())})

    def to_dict(self):
        return {"n_features": self.n_features_, "base_score": self.base_score_,
                "params": self.params_, "edges": [e.tolist() for e in self.edges_]}


def fit_lambdamart(df, X_cols, y_col, query_col, *, n_estimators=100,
                   max_depth=3, learning_rate=0.1, bins=32, sigma=1.0,
                   min_child_weight=1.0, reg_lambda=1.0, ndcg_k=10,
                   memory_guard: Optional[MemoryGuard] = None):
    """Fit a LambdaMART ranker with a bounded streaming pre-pass.

    Pass 1 collects features, graded labels and query ids into memory;
    check_state(N*d*8) runs first so an oversized input raises
    MemoryLimitExceeded rather than OOM. Boosting then runs per query.
    """
    from cadw.ml.gbdt import _stack_numeric
    cols = [X_cols] if isinstance(X_cols, str) else list(X_cols)
    if not cols:
        raise ValueError("fit_lambdamart: X_cols must be non-empty")
    for c in cols + [y_col, query_col]:
        if c not in df._columns:
            raise KeyError(f"fit_lambdamart: column {c!r} not in {df._columns}")
    mg = memory_guard or df._memory_guard

    # v0.9.87-P2a (audit MEDIUM): pre-allocate X + y from df.count();
    # no Xs list, no np.vstack -> eliminate transient 2*n*d*8 peak.
    # Peak = n*(d*8+16). Bit-identical output (same values, same
    # order; only the transient duplicates are gone).
    # q kept as small per-batch list (~8 bytes/row across concat; not
    # the audit concern, 25x smaller than X at d=10).
    n_total = df.count()
    if n_total == 0:
        raise ValueError("fit_lambdamart: empty input")
    d_cols = len(cols)
    mg.check_state(n_total * (d_cols * 8 + 16),
                   name="lambdamart.state")

    X = np.empty((n_total, d_cols), dtype=np.float64)
    y = np.empty(n_total, dtype=np.float64)
    qs = []
    off = 0
    for b in df._iter_filtered_batches():
        mg.check_batch(b, name="lambdamart.pass1", extra_factor=2.0)
        nb = b.num_rows
        X[off:off + nb] = _stack_numeric(b, cols)
        y[off:off + nb] = b.column(y_col).cast(
            pa.float64(), safe=False).to_numpy(zero_copy_only=False)
        qs.append(b.column(query_col).to_numpy(zero_copy_only=False))
        off += nb
    n_rows = off
    if n_rows != n_total:
        # df changed between count and iterate - defensive truncate.
        X = X[:n_rows]
        y = y[:n_rows]
        n_total = n_rows
    q = np.concatenate(qs) if qs else np.array([], dtype=np.int64)
    if np.isnan(X).any() or np.isnan(y).any():
        raise ValueError("fit_lambdamart: nulls in features or labels")
    if y.min() < 0:
        raise ValueError("fit_lambdamart: relevance labels must be >= 0")

    codes, edges = _bin_features(X, int(bins), memory_guard=mg)
    nbins = int(codes.max()) + 1
    uq = sorted(set(q.tolist()), key=str)
    groups = [(u, np.where(q == u)[0]) for u in uq]
    if len(groups) < 2:
        raise ValueError("fit_lambdamart: need >= 2 queries to rank against")

    scores = np.zeros(n_rows, dtype=np.float64)
    trees = []
    base = 0.0
    for it in range(int(n_estimators)):
        grad = np.zeros(n_rows, dtype=np.float64)
        hess = np.zeros(n_rows, dtype=np.float64)
        for _, idx in groups:
            lam, h = _lambda_gradients(scores[idx], y[idx], ndcg_k, float(sigma))
            grad[idx] = -lam
            hess[idx] = np.maximum(h, 1e-6)
        tree = _build_tree(codes, grad, hess, np.arange(n_rows), 0,
                           int(max_depth), float(min_child_weight),
                           float(reg_lambda), nbins)
        scores += float(learning_rate) * _predict_tree(tree, codes)
        trees.append(tree)
        nd = float(np.mean([ndcg_at_k(y[i], scores[i], ndcg_k) for _, i in groups]))
        log_fit("lambdamart", iter_i=it + 1, iter_n=int(n_estimators),
                loss=-nd, state_bytes=n_rows * len(cols) * 8)

    model = LambdaMARTModel(trees, edges, base, len(cols),
                            {"n_estimators": n_estimators, "max_depth": max_depth,
                             "learning_rate": learning_rate, "bins": bins,
                             "sigma": sigma, "ndcg_k": ndcg_k})
    model.feature_names_ = cols
    mg.check_state(model.state_nbytes(), name="lambdamart.model")
    return model
