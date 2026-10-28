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
from cadw.core.spill import SpillDir
from cadw.ops.reduce import reduce, ReduceOp
from cadw.ml.gbdt import (
    _binning_pass,
    _stack_numeric,
    _column_to_f64,
    _find_best_split,
)


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


def _predict_tree_chunked(node, codes_mm, chunk_size=10000):
    """P2b-full: chunked predict over memmap codes. Bit-identical to
    _predict_tree (same values, same leaf assignment order per chunk)."""
    n = codes_mm.shape[0]
    out = np.empty(n, dtype=np.float64)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        out[start:end] = _predict_tree(
            node, np.asarray(codes_mm[start:end]))
    return out


def _accumulate_tree_into_memmap(node, codes_mm, scores_mm, lr,
                                  chunk_size=10000):
    """P2b-full: scores_mm += lr * predict(codes_mm) in chunks."""
    n = codes_mm.shape[0]
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        codes_c = np.asarray(codes_mm[start:end])
        out_c = _predict_tree(node, codes_c)
        scores_c = np.asarray(scores_mm[start:end]).copy()
        scores_c += lr * out_c
        scores_mm[start:end] = scores_c


def _dict_to_node(tree_struct, leaf_values, nid):
    """P2b-full: reconstruct _Node tree from dict + leaf_values."""
    node = _Node()
    spec = tree_struct.get(nid)
    if spec is None:
        node.value = float(leaf_values.get(nid, 0.0))
        return node
    j, thr, left_id, right_id = spec[:4]
    node.feat = int(j)
    node.thr = int(thr)
    node.left = _dict_to_node(tree_struct, leaf_values, int(left_id))
    node.right = _dict_to_node(tree_struct, leaf_values, int(right_id))
    return node


# ============================================================
# P2b-full: ReduceOps + BFS tree build (level-synchronous).
# Adapted from cadw/ml/gbdt.py _AllNodeHistOpCodes / _AssignLeavesOpCodes
# / _LeafValuesOp for single-class LambdaMART (no per-class loop).
# ============================================================


class _AllNodeHistOpCodes_LM(ReduceOp):
    """Per-node histogram from pre-binned codes (single-class LM)."""

    def __init__(self, feature_cols, node_ids, bins):
        self.feature_cols = list(feature_cols)
        self.f = len(feature_cols)
        self.node_ids = list(node_ids)
        self.node_to_slot = {nid: i for i, nid in enumerate(node_ids)}
        self.bins = int(bins)
        self.S = len(node_ids)

    def init(self):
        return (
            np.zeros((self.S, self.f, self.bins), dtype=np.float64),
            np.zeros((self.S, self.f, self.bins), dtype=np.float64),
            np.zeros(self.S, dtype=np.float64),
            np.zeros(self.S, dtype=np.float64),
            np.zeros(self.S, dtype=np.int64),
        )

    def update(self, state, batch):
        gh, hh, gt, ht, nt = state
        n = batch.num_rows
        if n == 0:
            return state
        codes = np.stack(
            [batch.column(f"__code_{j}").to_numpy(zero_copy_only=False)
             for j in range(self.f)], axis=1)
        nan_mask = batch.column("__nan_mask").to_numpy(zero_copy_only=False)
        g = _column_to_f64(batch.column("__g"))
        h = _column_to_f64(batch.column("__h"))
        nid_arr = batch.column("__node_id").to_numpy(zero_copy_only=False)
        for nid, slot in self.node_to_slot.items():
            m = nid_arr == nid
            if not m.any():
                continue
            m_valid = m & ~nan_mask
            if not m_valid.any():
                continue
            gn = g[m_valid]
            hn = h[m_valid]
            gt[slot] += float(gn.sum())
            ht[slot] += float(hn.sum())
            nt[slot] += int(m_valid.sum())
            codes_valid = codes[m_valid]
            for j in range(self.f):
                b_idx = codes_valid[:, j]
                gh[slot, j] += np.bincount(
                    b_idx, weights=gn, minlength=self.bins)
                hh[slot, j] += np.bincount(
                    b_idx, weights=hn, minlength=self.bins)
        return state

    def merge(self, a, b):
        return tuple(x + y for x, y in zip(a, b))

    def state_nbytes(self, state):
        return int(sum(x.nbytes for x in state) + 128)

    def estimate_state_nbytes(self):
        return int(2 * self.S * self.f * self.bins * 8 + 256)

    def batch_memory_factor(self):
        return 4.0

    def finalize(self, state):
        gh, hh, gt, ht, nt = state
        return {"g_hist": gh, "h_hist": hh,
                "g_tot": gt, "h_tot": ht, "n_tot": nt}


class _AssignLeavesOpCodes_LM(ReduceOp):
    """Assign node_id per row from codes + current tree structure."""

    def __init__(self, feature_cols, tree, root_id=0):
        self.feature_cols = list(feature_cols)
        self.f = len(feature_cols)
        self.tree = dict(tree)
        self.root_id = int(root_id)

    def init(self):
        return []

    def update(self, state, batch):
        n = batch.num_rows
        codes = np.stack(
            [batch.column(f"__code_{j}").to_numpy(zero_copy_only=False)
             for j in range(self.f)], axis=1)
        nan_mask = batch.column("__nan_mask").to_numpy(zero_copy_only=False)
        node_id = np.full(n, self.root_id, dtype=np.int32)
        for _ in range(32):
            any_change = False
            for nid, spec in self.tree.items():
                if spec is None:
                    continue
                j, thr, left, right = spec[:4]
                m = (node_id == nid) & ~nan_mask
                if not m.any():
                    continue
                new_ids = np.where(
                    codes[m, j] <= thr, int(left), int(right)).astype(np.int32)
                node_id[m] = new_ids
                any_change = True
            if not any_change:
                break
        state.append(node_id)
        return state

    def merge(self, a, b):
        return a + b

    def state_nbytes(self, state):
        return int(sum(x.nbytes for x in state) + 8)

    def estimate_state_nbytes(self):
        return 0

    def batch_memory_factor(self):
        return 4.0

    def finalize(self, state):
        if not state:
            return np.zeros(0, dtype=np.int32)
        return np.concatenate(state).astype(np.int32)


class _LeafValuesOp_LM(ReduceOp):
    """Per-leaf gradient/hessian sum -> Newton step."""

    def __init__(self, reg_lambda=1.0):
        self.reg_lambda = float(reg_lambda)

    def init(self):
        return {}

    def update(self, state, batch):
        g = _column_to_f64(batch.column("__g"))
        h = _column_to_f64(batch.column("__h"))
        nid_arr = batch.column("__node_id").to_numpy(zero_copy_only=False)
        for nid in np.unique(nid_arr):
            m = nid_arr == nid
            key = int(nid)
            sg, sh = state.get(key, (0.0, 0.0))
            state[key] = (sg + float(g[m].sum()), sh + float(h[m].sum()))
        return state

    def merge(self, a, b):
        result = dict(a)
        for k, (g, h) in b.items():
            og, oh = result.get(k, (0.0, 0.0))
            result[k] = (og + g, oh + h)
        return result

    def state_nbytes(self, state):
        return 16 * max(1, len(state))

    def estimate_state_nbytes(self):
        return 0

    def batch_memory_factor(self):
        return 3.5

    def finalize(self, state):
        return {k: -g / (h + self.reg_lambda)
                for k, (g, h) in state.items()}


def _iter_lm_enriched(df, X_cols, grad_mm, hess_mm, node_id_mm,
                      codes_mm, nan_mask):
    """Yield batches with __code_*, __g, __h, __node_id, __nan_mask."""
    off = 0
    d = len(X_cols)
    for b in df._iter_filtered_batches():
        n = b.num_rows
        codes_b = np.asarray(codes_mm[off:off + n])
        nm_b = np.asarray(nan_mask[off:off + n])
        g = np.where(~nm_b, np.asarray(grad_mm[off:off + n]), 0.0)
        h = np.where(~nm_b, np.asarray(hess_mm[off:off + n]), 0.0)
        nid = np.asarray(node_id_mm[off:off + n])
        cols = [pa.array(codes_b[:, j], type=pa.uint8())
                for j in range(d)]
        cols += [pa.array(g, type=pa.float64()),
                 pa.array(h, type=pa.float64()),
                 pa.array(nid, type=pa.int32()),
                 pa.array(nm_b)]
        names = ([f"__code_{j}" for j in range(d)]
                 + ["__g", "__h", "__node_id", "__nan_mask"])
        yield pa.RecordBatch.from_arrays(cols, names=names)
        off += n


def _iter_x_with_codes_lm(df, X_cols, codes_mm, nan_mask):
    """Codes + nan_mask only (for _AssignLeavesOpCodes_LM)."""
    off = 0
    d = len(X_cols)
    for b in df._iter_filtered_batches():
        n = b.num_rows
        codes_b = np.asarray(codes_mm[off:off + n])
        nm_b = np.asarray(nan_mask[off:off + n])
        code_cols = [pa.array(codes_b[:, j], type=pa.uint8())
                     for j in range(d)]
        code_names = [f"__code_{j}" for j in range(d)]
        yield pa.RecordBatch.from_arrays(
            code_cols + [pa.array(nm_b)],
            names=code_names + ["__nan_mask"],
        )
        off += n


def _iter_lm_for_leaves(df, X_cols, grad_mm, hess_mm, node_id_mm):
    """__g, __h, __node_id only (for _LeafValuesOp_LM)."""
    off = 0
    for b in df._iter_filtered_batches():
        n = b.num_rows
        g = np.asarray(grad_mm[off:off + n])
        h = np.asarray(hess_mm[off:off + n])
        nid = np.asarray(node_id_mm[off:off + n])
        yield pa.RecordBatch.from_arrays(
            [pa.array(g, type=pa.float64()),
             pa.array(h, type=pa.float64()),
             pa.array(nid, type=pa.int32())],
            names=["__g", "__h", "__node_id"],
        )
        off += n


def _build_tree_bfs(df, X_cols, codes_mm, grad_mm, hess_mm,
                    nan_mask, node_id_mm, mg, *, max_depth,
                    min_child_weight, reg_lambda, nbins):
    """Level-synchronous tree build via ReduceOps (P2b-full).

    Replaces DFS _build_tree. Same math (bincount-based histogram +
    _find_best_split per node), bit-identical to DFS within float
    summation order (~1e-15). True OOC: RAM = O(batch·d + n_nodes·d·bins).
    """
    tree = {0: None}
    next_node_id = 1

    # Reset node_id_mm to root=0 (all rows start at root)
    node_id_mm[:] = 0
    node_id_mm.flush()

    for level in range(int(max_depth)):
        internal = [nid for nid, t in tree.items() if t is None]
        if not internal:
            break

        hist_op = _AllNodeHistOpCodes_LM(X_cols, internal, nbins)
        hist_res = reduce(
            _iter_lm_enriched(
                df, X_cols, grad_mm, hess_mm, node_id_mm,
                codes_mm, nan_mask),
            hist_op, mg, log=False,
        )
        gh = hist_res["g_hist"]
        hh = hist_res["h_hist"]
        gt = hist_res["g_tot"]
        ht = hist_res["h_tot"]
        nt = hist_res["n_tot"]

        for slot, nid in enumerate(internal):
            if ht[slot] < float(min_child_weight) * 2:
                continue
            best = _find_best_split(
                slot, gh[slot], hh[slot],
                float(gt[slot]), float(ht[slot]), int(nt[slot]),
                nbins, float(min_child_weight), 0.0,
                float(reg_lambda))
            if best is None:
                continue
            j, thr, gain = best
            if gain <= 0.0:
                continue
            left_id = next_node_id
            right_id = next_node_id + 1
            next_node_id += 2
            tree[nid] = (int(j), int(thr), int(left_id), int(right_id),
                         float(gain))
            tree[left_id] = None
            tree[right_id] = None

        # Reassign node_id per row using current tree state
        assign_op = _AssignLeavesOpCodes_LM(X_cols, tree)
        leaf_list = reduce(
            _iter_x_with_codes_lm(df, X_cols, codes_mm, nan_mask),
            assign_op, mg, log=False,
        )
        new_nid = (np.concatenate(leaf_list) if isinstance(leaf_list, list)
                   and leaf_list else np.asarray(leaf_list, dtype=np.int32))
        node_id_mm[:] = new_nid.astype(np.int32)
        node_id_mm.flush()

    # Finalize leaf values
    lv_op = _LeafValuesOp_LM(reg_lambda=float(reg_lambda))
    leaf_values = reduce(
        _iter_lm_for_leaves(df, X_cols, grad_mm, hess_mm, node_id_mm),
        lv_op, mg, log=False,
    )
    return tree, leaf_values


class LambdaMARTModel:
    """Gradient-boosted ranker with the LambdaRank gradient.

    v0.9.87-P2b-full: uses uniform binning (mn/mx/bins) not quantile
    edges. Predict path matches gbdt _binning_pass formula. Documented
    deviation from pre-P2b (which used quantile bins); same-version
    determinism preserved.
    """

    def __init__(self, trees, mn, mx, bins, base_score, n_features, params):
        self.trees_ = trees
        self.mn_ = np.asarray(mn, dtype=np.float64)
        self.mx_ = np.asarray(mx, dtype=np.float64)
        self.range_ = np.where(self.mx_ - self.mn_ > 0,
                               self.mx_ - self.mn_, 1.0)
        self.bins_ = int(bins)
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
        """Predict via uniform binning (v0.9.87-P2b-full)."""
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        with np.errstate(divide="ignore", invalid="ignore"):
            codes = np.clip(
                ((X - self.mn_) / self.range_ * self.bins_).astype(np.int64),
                0, self.bins_ - 1).astype(np.uint8)
        s = np.full(n, self.base_score_, dtype=np.float64)
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
        return {"n_features": self.n_features_,
                "base_score": self.base_score_,
                "params": self.params_,
                "mn": self.mn_.tolist(), "mx": self.mx_.tolist(),
                "bins": self.bins_}


def fit_lambdamart(df, X_cols, y_col, query_col, *, n_estimators=100,
                   max_depth=3, learning_rate=0.1, bins=32, sigma=1.0,
                   min_child_weight=1.0, reg_lambda=1.0, ndcg_k=10,
                   memory_guard: Optional[MemoryGuard] = None):
    """Fit a LambdaMART ranker, true out-of-core (v0.9.87-P2b-full).

    All n-sized state is SpillDir-backed memmap: codes (uint8), y
    (float64), scores/grad/hess (float64), node_id (int32). Tree
    build is level-synchronous BFS via ReduceOps (_AllNodeHistOpCodes_LM,
    _AssignLeavesOpCodes_LM, _LeafValuesOp_LM); no per-row arange
    over full n, no codes[idx, j] fancy indexing on full-n.
    RAM = O(batch*d + n_nodes*d*bins).

    Uniform binning (mn/mx/bins) replaces quantile edges — deviation
    from pre-P2b version; documented in CHANGELOG. Same-version
    determinism preserved (no cross-version bit-identical contract).

    Query grouping is dict-based (Python lists) — bounded by n_q.
    Pathological n_q > ~1M requires external_sort (future P2b-full-2).
    """
    cols = [X_cols] if isinstance(X_cols, str) else list(X_cols)
    if not cols:
        raise ValueError("fit_lambdamart: X_cols must be non-empty")
    for c in cols + [y_col, query_col]:
        if c not in df._columns:
            raise KeyError(f"fit_lambdamart: column {c!r} not in {df._columns}")
    mg = memory_guard or df._memory_guard

    n_total = df.count()
    if n_total == 0:
        raise ValueError("fit_lambdamart: empty input")

    # Batch-bound guard only — all persistent n-sized state on disk.
    mg.check_state(n_total * 4, name="lambdamart.batch_bound")

    mn, mx, range_, codes_mm, nan_mask, binning_spill = _binning_pass(
        df, cols, bins=int(bins), memory_guard=mg, guard=mg)
    nbins = int(bins)

    try:
        # y_mm
        _y_path = binning_spill.path("y.bin")
        y_mm = np.memmap(str(_y_path), dtype=np.float64, mode="w+",
                         shape=(n_total,))

        # Group rows by query (dict-based; bounded by n_q)
        q_groups = {}
        off = 0
        for b in df._iter_filtered_batches():
            mg.check_batch(b, name="lambdamart.pass1", extra_factor=2.0)
            q_arr = b.column(query_col).to_numpy(zero_copy_only=False)
            y_arr = b.column(y_col).cast(
                pa.float64(), safe=False).to_numpy(zero_copy_only=False)
            nb = b.num_rows
            y_mm[off:off + nb] = y_arr
            for i, qv in enumerate(q_arr.tolist()):
                q_groups.setdefault(qv, []).append(off + i)
            off += nb
        y_mm.flush()
        if off != n_total:
            raise RuntimeError(
                f"fit_lambdamart: row count changed between count() "
                f"({n_total}) and iterate ({off})")

        if len(q_groups) < 2:
            raise ValueError(
                "fit_lambdamart: need >= 2 queries to rank against")
        if np.isnan(y_mm).any():
            raise ValueError("fit_lambdamart: nulls in labels")
        if y_mm.min() < 0:
            raise ValueError("fit_lambdamart: relevance labels must be >= 0")

        q_groups = {q: np.array(idx, dtype=np.int64)
                    for q, idx in q_groups.items()}

        # scores_mm, node_id_mm (both persistent across rounds)
        _scores_path = binning_spill.path("scores.bin")
        scores_mm = np.memmap(str(_scores_path), dtype=np.float64,
                              mode="w+", shape=(n_total,))
        _nid_path = binning_spill.path("node_id.bin")
        node_id_mm = np.memmap(str(_nid_path), dtype=np.int32,
                               mode="w+", shape=(n_total,))

        trees = []
        base = 0.0
        for it in range(int(n_estimators)):
            _g_path = binning_spill.path(f"g_r{it:04d}.bin")
            _h_path = binning_spill.path(f"h_r{it:04d}.bin")
            grad_mm = np.memmap(str(_g_path), dtype=np.float64,
                                mode="w+", shape=(n_total,))
            hess_mm = np.memmap(str(_h_path), dtype=np.float64,
                                mode="w+", shape=(n_total,))

            # Compute grad/hess per query
            for q, idx in q_groups.items():
                scores_q = np.asarray(scores_mm[idx])
                y_q = np.asarray(y_mm[idx])
                lam, h = _lambda_gradients(
                    scores_q, y_q, ndcg_k, float(sigma))
                grad_mm[idx] = -lam
                hess_mm[idx] = np.maximum(h, 1e-6)
            grad_mm.flush()
            hess_mm.flush()

            # BFS tree build
            tree_struct, leaf_values = _build_tree_bfs(
                df, cols, codes_mm, grad_mm, hess_mm,
                nan_mask, node_id_mm, mg,
                max_depth=int(max_depth),
                min_child_weight=float(min_child_weight),
                reg_lambda=float(reg_lambda), nbins=nbins)

            # Convert to _Node and update scores
            root = _dict_to_node(tree_struct, leaf_values, 0)
            _accumulate_tree_into_memmap(
                root, codes_mm, scores_mm, float(learning_rate))
            scores_mm.flush()
            trees.append(root)

            # NDCG for logging (all queries, bounded per-query arrays)
            nd_vals = []
            for q, idx in q_groups.items():
                y_q = np.asarray(y_mm[idx])
                s_q = np.asarray(scores_mm[idx])
                nd_vals.append(ndcg_at_k(y_q, s_q, ndcg_k))
            nd = float(np.mean(nd_vals)) if nd_vals else 0.0
            log_fit("lambdamart", iter_i=it + 1,
                    iter_n=int(n_estimators),
                    loss=-nd, state_bytes=nbins * max_depth * 96)

            # Cleanup per-round grad/hess
            grad_mm.flush()
            hess_mm.flush()
            del grad_mm, hess_mm
            binning_spill.cleanup_file(_g_path)
            binning_spill.cleanup_file(_h_path)

        model = LambdaMARTModel(
            trees, mn, mx, int(bins), base, len(cols),
            {"n_estimators": int(n_estimators),
             "max_depth": int(max_depth),
             "learning_rate": float(learning_rate),
             "bins": int(bins), "sigma": float(sigma),
             "ndcg_k": int(ndcg_k)})
        model.feature_names_ = cols
        mg.check_state(model.state_nbytes(), name="lambdamart.model")
        return model
    finally:
        binning_spill.__exit__(None, None, None)
