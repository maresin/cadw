"""Tree builders: symmetric + leaf_wise. Constant + linear leaf predictors.

Symmetric (CatBoost): 2^depth leaves, one split per level.
Leaf-wise (LightGBM): best-first growth, per-node splits.

Both support: default_left per split, min_child_weight, path_smoothing,
max_delta_step, colsample_bytree, constant/linear leaves.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .histogram import build_histogram
from .split_finder import (
    SplitResult,
    best_split_per_node,
    best_split_symmetric,
)


def _constant_leaf_values(
    G_leaf, H_leaf, reg_lambda, min_child_weight,
    max_delta_step, path_smoothing,
):
    eps = 1e-12
    leaves = -G_leaf / (H_leaf + reg_lambda + eps)
    if path_smoothing > 0:
        leaves = leaves / (1.0 + path_smoothing * H_leaf)
    if max_delta_step > 0:
        leaves = np.clip(leaves, -max_delta_step, max_delta_step)
    if min_child_weight > 0:
        leaves[H_leaf < min_child_weight] = 0.0
    return leaves


def _linear_leaf_fit(
    grad, hess, X, leaf_idx, n_leaves, G_leaf, H_leaf,
    reg_lambda, min_child_weight,
):
    """Per-leaf weighted ridge regression. Returns (coefs, intercepts)."""
    n, d = X.shape
    coefs = np.zeros((n_leaves, d))
    intercepts = np.zeros(n_leaves)
    for L in range(n_leaves):
        if H_leaf[L] < min_child_weight:
            continue
        mask = leaf_idx == L
        n_L = int(mask.sum())
        if n_L < 2:
            intercepts[L] = -G_leaf[L] / (H_leaf[L] + reg_lambda + 1e-12)
            continue
        X_L = X[mask]
        g_L = grad[mask]
        h_L = hess[mask]
        XtHX = X_L.T @ (X_L * h_L[:, None]) + reg_lambda * np.eye(d)
        Xtg = -(X_L.T @ g_L)
        try:
            w = np.linalg.solve(XtHX, Xtg)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(XtHX, Xtg, rcond=None)[0]
        coefs[L] = w
        pred_L = X_L @ w
        denom = h_L.sum() + reg_lambda + 1e-12
        intercepts[L] = float(np.sum(h_L * (-g_L / h_L - pred_L)) / denom)
    return coefs, intercepts


@dataclass
class Tree:
    mode: str
    max_depth: int
    linear: bool = False
    split_gains: Optional[np.ndarray] = None   # per-split gain (for importance)
    # Symmetric state
    splits: Optional[np.ndarray] = None          # (depth, 3) [feat, thresh, dl]
    # Leaf-wise state
    feature: Optional[np.ndarray] = None
    threshold: Optional[np.ndarray] = None
    default_left: Optional[np.ndarray] = None
    left_child: Optional[np.ndarray] = None
    right_child: Optional[np.ndarray] = None
    is_leaf: Optional[np.ndarray] = None
    # Common
    values: np.ndarray = None
    coefs: Optional[np.ndarray] = None

    def predict(self, codes, X_raw=None):
        if self.mode == "symmetric":
            return self._predict_symmetric(codes, X_raw)
        return self._predict_leafwise(codes, X_raw)

    def _leaf_idx_symmetric(self, codes):
        n = codes.shape[0]
        leaf_idx = np.zeros(n, dtype=np.int64)
        for level in range(self.max_depth):
            feat = int(self.splits[level, 0])
            if feat < 0:
                continue
            thresh = int(self.splits[level, 1])
            go_right = codes[:, feat] >= thresh
            leaf_idx = leaf_idx | (go_right.astype(np.int64) << level)
        return leaf_idx

    def _predict_symmetric(self, codes, X_raw):
        leaf_idx = self._leaf_idx_symmetric(codes)
        if not self.linear:
            return self.values[leaf_idx]
        if X_raw is None:
            raise ValueError("linear leaves require X_raw at predict")
        return self.values[leaf_idx] + np.einsum(
            "ij,ij->i", X_raw, self.coefs[leaf_idx])

    def _predict_leafwise(self, codes, X_raw):
        n = codes.shape[0]
        node = np.zeros(n, dtype=np.int64)
        for _ in range(len(self.feature) + 1):
            active = ~self.is_leaf[node]
            if not active.any():
                break
            feat = np.where(active, self.feature[node], 0)
            thresh = self.threshold[node]
            col = codes[np.arange(n), feat]
            go_right = col >= thresh
            nxt = np.where(go_right,
                           self.right_child[node],
                           self.left_child[node])
            node = np.where(active, nxt, node)
        if not self.linear:
            return self.values[node]
        if X_raw is None:
            raise ValueError("linear leaves require X_raw at predict")
        return self.values[node] + np.einsum(
            "ij,ij->i", X_raw, self.coefs[node])


def fit_symmetric_tree(
    codes, grad, hess, X_raw, n_bins_per_feature, params, rng,
    *, missing_bin=False,
):
    n, d = codes.shape
    depth = int(params["max_depth"])
    splits = np.full((depth, 3), -1, dtype=np.int64)
    splits[:, 2] = 1
    split_gains = np.zeros(depth, dtype=np.float64)
    leaf_idx = np.zeros(n, dtype=np.int64)
    history = []

    for level in range(depth):
        n_leaves = 2 ** level
        hist = build_histogram(
            codes, grad, hess, leaf_idx, n_leaves, n_bins_per_feature)
        fmask = None
        if params.get("colsample_bytree", 1.0) < 1.0:
            fmask = rng.rand(d) < params["colsample_bytree"]
        sp = best_split_symmetric(
            hist, params["reg_lambda"], params["min_child_weight"],
            params["min_gain"], missing_bin=missing_bin,
            feature_mask=fmask)
        if sp.feature < 0:
            history.append((level, -1, -1, 0.0))
            break
        splits[level, 0] = sp.feature
        splits[level, 1] = sp.threshold
        splits[level, 2] = 1 if sp.default_left else 0
        split_gains[level] = sp.gain
        history.append((level, sp.feature, sp.threshold, sp.gain))
        go_right = (codes[:, sp.feature] >= sp.threshold).astype(np.int64)
        leaf_idx = leaf_idx | (go_right << level)

    n_leaves_final = 2 ** depth
    G_leaf = np.zeros(n_leaves_final)
    H_leaf = np.zeros(n_leaves_final)
    np.add.at(G_leaf, leaf_idx, grad)
    np.add.at(H_leaf, leaf_idx, hess)

    if (params.get("leaf_predictor", "constant") == "linear"
            and X_raw is not None):
        coefs, intercepts = _linear_leaf_fit(
            grad, hess, X_raw, leaf_idx, n_leaves_final,
            G_leaf, H_leaf, params["reg_lambda"],
            params["min_child_weight"])
        values = intercepts
        linear = True
    else:
        values = _constant_leaf_values(
            G_leaf, H_leaf, params["reg_lambda"],
            params["min_child_weight"],
            params.get("max_delta_step", 0.0),
            params.get("path_smoothing", 0.0))
        coefs = None
        linear = False

    tree = Tree(
        mode="symmetric", max_depth=depth, linear=linear,
        splits=splits, values=values, coefs=coefs,
        split_gains=split_gains,
    )
    return tree, history


def fit_leafwise_tree(
    codes, grad, hess, X_raw, n_bins_per_feature, params, rng,
    *, missing_bin=False,
):
    n, d = codes.shape
    max_leaves = 2 ** int(params["max_depth"])
    node_rows = [np.arange(n)]
    is_leaf = [True]
    feat_arr = [-1]
    thresh_arr = [-1]
    dl_arr = [True]
    left_arr = [-1]
    right_arr = [-1]

    fmask = None
    if params.get("colsample_bytree", 1.0) < 1.0:
        fmask = rng.rand(d) < params["colsample_bytree"]

    # Per-node histogram cache (LightGBM-style)
    hist_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def _node_hist(node_id):
        if node_id in hist_cache:
            return hist_cache[node_id]
        rows = node_rows[node_id]
        local = np.zeros(len(rows), dtype=np.int64)
        hist = build_histogram(
            codes[rows], grad[rows], hess[rows], local, 1,
            n_bins_per_feature)
        result = (hist.G[0].copy(), hist.H[0].copy())
        hist_cache[node_id] = result
        return result

    for split_i in range(max_leaves - 1):
        best_node = -1
        best_sp = SplitResult(-1, -1, True, -np.inf)
        for node_i in range(len(is_leaf)):
            if not is_leaf[node_i]:
                continue
            rows = node_rows[node_i]
            if len(rows) < 2:
                continue
            G_n, H_n = _node_hist(node_i)
            sp = best_split_per_node(
                G_n, H_n, n_bins_per_feature,
                params["reg_lambda"], params["min_child_weight"],
                params["min_gain"], missing_bin=missing_bin,
                feature_mask=fmask)
            if sp.feature >= 0 and sp.gain > best_sp.gain:
                best_sp = sp
                best_node = node_i
        if best_node < 0:
            break
        rows = node_rows[best_node]
        go_right = codes[rows, best_sp.feature] >= best_sp.threshold
        left_rows = rows[~go_right]
        right_rows = rows[go_right]
        if len(left_rows) == 0 or len(right_rows) == 0:
            break

        # Get parent hist before we invalidate best_node
        parent_G, parent_H = _node_hist(best_node)

        is_leaf[best_node] = False
        feat_arr[best_node] = best_sp.feature
        thresh_arr[best_node] = best_sp.threshold
        dl_arr[best_node] = best_sp.default_left
        for rows_side in (left_rows, right_rows):
            is_leaf.append(True)
            node_rows.append(rows_side)
            feat_arr.append(-1)
            thresh_arr.append(-1)
            dl_arr.append(True)
            left_arr.append(-1)
            right_arr.append(-1)
        li = len(is_leaf) - 2
        ri = len(is_leaf) - 1
        left_arr[best_node] = li
        right_arr[best_node] = ri

        # Histogram subtraction (LightGBM trick):
        # build smaller child, derive larger = parent - smaller.
        if len(left_rows) <= len(right_rows):
            local = np.zeros(len(left_rows), dtype=np.int64)
            hist_l = build_histogram(
                codes[left_rows], grad[left_rows], hess[left_rows], local, 1,
                n_bins_per_feature)
            hist_cache[li] = (hist_l.G[0].copy(), hist_l.H[0].copy())
            hist_cache[ri] = (parent_G - hist_l.G[0],
                              parent_H - hist_l.H[0])
        else:
            local = np.zeros(len(right_rows), dtype=np.int64)
            hist_r = build_histogram(
                codes[right_rows], grad[right_rows], hess[right_rows], local, 1,
                n_bins_per_feature)
            hist_cache[ri] = (hist_r.G[0].copy(), hist_r.H[0].copy())
            hist_cache[li] = (parent_G - hist_r.G[0],
                              parent_H - hist_r.H[0])
        # Free parent cache (no longer needed)
        del hist_cache[best_node]

    n_nodes = len(is_leaf)
    leaf_global = np.full(n, -1, dtype=np.int64)
    for i in range(n_nodes):
        if is_leaf[i]:
            leaf_global[node_rows[i]] = i
    G_leaf = np.zeros(n_nodes)
    H_leaf = np.zeros(n_nodes)
    ok = leaf_global >= 0
    np.add.at(G_leaf, leaf_global[ok], grad[ok])
    np.add.at(H_leaf, leaf_global[ok], hess[ok])

    if (params.get("leaf_predictor", "constant") == "linear"
            and X_raw is not None):
        coefs, intercepts = _linear_leaf_fit(
            grad, hess, X_raw, leaf_global, n_nodes, G_leaf, H_leaf,
            params["reg_lambda"], params["min_child_weight"])
        values = intercepts
        linear = True
    else:
        values = _constant_leaf_values(
            G_leaf, H_leaf, params["reg_lambda"],
            params["min_child_weight"],
            params.get("max_delta_step", 0.0),
            params.get("path_smoothing", 0.0))
        for i in range(n_nodes):
            if not is_leaf[i]:
                values[i] = 0.0
        coefs = None
        linear = False

    return Tree(
        mode="leaf_wise", max_depth=int(params["max_depth"]),
        linear=linear,
        feature=np.array(feat_arr, dtype=np.int16),
        threshold=np.array(thresh_arr, dtype=np.int16),
        default_left=np.array(dl_arr, dtype=bool),
        left_child=np.array(left_arr, dtype=np.int32),
        right_child=np.array(right_arr, dtype=np.int32),
        is_leaf=np.array(is_leaf, dtype=bool),
        values=values, coefs=coefs,
    )
