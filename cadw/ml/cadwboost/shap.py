"""SHAP-style feature contributions via tree path tracing.

Not exact TreeSHAP (which requires O(2^depth) interventional estimates
per sample). Simplified attribution:
- For each tree, each sample: trace path from root to leaf
- Get leaf_value
- Attribute leaf_value equally among features that appeared in path
- Sum across trees

Properties:
- Sum of contributions + init_pred ≈ prediction (honest baseline)
- Feature appearing more often in paths (deeper trees) gets more credit
- Deterministic, O(n × n_trees × depth) — fast

For exact Shapley values with interactions → use `shap` library with
TreeExplainer (requires export to external format, P2 backlog).
"""
from __future__ import annotations

import numpy as np


def _path_features_symmetric(tree, codes_row):
    """Return (features_in_path, leaf_idx) for one row."""
    n = 1
    leaf_idx = 0
    features = []
    for lvl in range(tree.max_depth):
        feat = int(tree.splits[lvl, 0])
        if feat < 0:
            continue
        thresh = int(tree.splits[lvl, 1])
        go_right = int(codes_row[feat]) >= thresh
        if go_right:
            leaf_idx |= (1 << lvl)
        features.append(feat)
    return features, leaf_idx


def _path_features_leafwise(tree, codes_row):
    """Return (features_in_path, leaf_node) for one row."""
    node = 0
    features = []
    for _ in range(len(tree.feature) + 1):
        if tree.is_leaf[node]:
            break
        feat = int(tree.feature[node])
        features.append(feat)
        thresh = int(tree.threshold[node])
        if codes_row[feat] >= thresh:
            node = int(tree.right_child[node])
        else:
            node = int(tree.left_child[node])
    return features, node


def feature_contributions(model, X, feature_names=None):
    """Per-sample, per-feature contributions (simplified TreeSHAP).

    Parameters
    ----------
    model : CADWBoost (fitted)
    X : (n, d) numpy array
    feature_names : optional list, default = model.feature_names_

    Returns
    -------
    (n, d) ndarray: contributions[i, j] = contribution of feature j
    to prediction of sample i.

    Interpretation:
    - prediction ≈ init_pred + sum_j contributions[i, j]  (for loss
      where predict_from_raw = identity, e.g. mse)
    - For classification (logistic), sum is on logit scale; apply
      sigmoid for probability-level interpretation.
    """
    if not model.fitted_:
        raise RuntimeError("model not fitted")
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    d = model.n_features_in_
    if feature_names is None:
        feature_names = model.feature_names_

    from .binning import apply_binning
    codes = apply_binning(X, model.thresholds_, model.missing_bin)
    if model.bundles_ is not None:
        from .efb import apply_bundles
        codes, _ = apply_bundles(codes, model.bundles_)

    contrib = np.zeros((n, d), dtype=np.float64)

    for tree, w in zip(model.trees_, model.tree_weights_):
        if w == 0:
            continue
        for i in range(n):
            codes_row = codes[i]
            if tree.mode == "symmetric":
                feats, leaf_idx = _path_features_symmetric(tree, codes_row)
                leaf_val = float(tree.values[leaf_idx])
            else:
                feats, leaf_node = _path_features_leafwise(tree, codes_row)
                leaf_val = float(tree.values[leaf_node])

            if not feats:
                continue
            per_feat = (w * leaf_val) / len(feats)
            for f in feats:
                if 0 <= f < d:
                    contrib[i, f] += per_feat

    return contrib


def summarize_contributions(contrib, feature_names):
    """Aggregate per-sample contributions → global importance.

    Returns dict feature_name -> {
        mean_abs: mean(|contribution|),   # global importance
        mean: mean(contribution),          # bias direction
        std: std(contribution),
    }
    """
    n, d = contrib.shape
    out = {}
    for j, name in enumerate(feature_names):
        col = contrib[:, j]
        out[name] = {
            "mean_abs": float(np.mean(np.abs(col))),
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
        }
    return out


def top_k_contributors(contrib, feature_names, k=5):
    """For each sample, top-k contributing features (positive + negative)."""
    n, d = contrib.shape
    out = []
    for i in range(n):
        row = contrib[i]
        # Sort by |contribution|
        order = np.argsort(-np.abs(row))[:k]
        out.append([
            (feature_names[j], float(row[j])) for j in order
        ])
    return out
