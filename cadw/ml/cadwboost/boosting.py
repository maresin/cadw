"""Boosting: standard GBDT + DART."""
from __future__ import annotations

import numpy as np

from .sampling import sample_goss, sample_random
from .tree import fit_leafwise_tree, fit_symmetric_tree


def _tree_params(model):
    return dict(
        max_depth=model.max_depth,
        reg_lambda=model.reg_lambda,
        min_child_weight=model.min_child_weight,
        min_gain=model.min_gain,
        colsample_bytree=model.colsample_bytree,
        leaf_predictor=model.leaf_predictor,
        max_delta_step=model.max_delta_step,
        path_smoothing=model.path_smoothing,
    )


def _fit_one_tree(model, codes_sub, grad_sub, hess_sub, X_sub,
                  n_bins_per_feature, rng, missing_bin):
    tp = _tree_params(model)
    if model.tree_mode == "symmetric":
        tree, _ = fit_symmetric_tree(
            codes_sub, grad_sub, hess_sub, X_sub,
            n_bins_per_feature, tp, rng, missing_bin=missing_bin)
        return tree
    return fit_leafwise_tree(
        codes_sub, grad_sub, hess_sub, X_sub,
        n_bins_per_feature, tp, rng, missing_bin=missing_bin)


def boost_standard(model, codes, X_raw, y, init_pred,
                   n_bins_per_feature, rng, missing_bin):
    n = codes.shape[0]
    pred = np.full(n, init_pred)
    trees = []
    weights = []
    for it in range(model.n_estimators):
        grad, hess = model.loss_.grad_hess(y, pred)
        if model.sampling == "none" and model.subsample >= 1.0:
            mask = np.ones(n, dtype=bool)
            w = np.ones(n)
        elif model.sampling == "goss":
            sm = sample_goss(n, grad, model.goss_top_rate,
                             model.goss_other_rate, rng)
            mask, w = sm.mask, sm.weights
        else:
            sm = sample_random(n, model.subsample, rng)
            mask, w = sm.mask, sm.weights
        g_fit = grad * w
        h_fit = hess * w
        X_sub = X_raw[mask] if X_raw is not None else None
        tree = _fit_one_tree(model, codes[mask], g_fit[mask], h_fit[mask],
                             X_sub, n_bins_per_feature, rng, missing_bin)
        update = tree.predict(codes, X_raw if tree.linear else None)
        pred = pred + model.learning_rate * update
        trees.append(tree)
        weights.append(model.learning_rate)
    return trees, weights, pred


def boost_dart(model, codes, X_raw, y, init_pred,
               n_bins_per_feature, rng, missing_bin):
    """DART: training-time dropout (uniform weight-drop variant).

    Simplified from Rashmi & Gilad-Bachrach (2015) "tree normalization":
    - Fit new tree on residual of (dropout mask on previous trees)
    - New tree weight = lr (same as standard GBDT)
    - No rescale of existing trees (avoids ln(N) underfit)
    - At inference: all trees at lr weight, no dropout

    This gives ~DART regularization without needing 4-7× more estimators.
    """
    n = codes.shape[0]
    dart_rate = model.dart_rate
    trees = []
    weights = []
    pred = np.full(n, init_pred)
    rng = np.random.RandomState(model.random_state)

    for it in range(model.n_estimators):
        # Per-row dropout mask for existing trees
        if it == 0:
            pred_drop = np.full(n, init_pred)
        else:
            # Random keep mask per (tree, row)
            keep = rng.rand(it, n) >= dart_rate   # True = use tree
            # Build pred_drop per row
            pred_drop = np.full(n, init_pred)
            for i, tree in enumerate(trees):
                tree_pred = tree.predict(
                    codes, X_raw if tree.linear else None)
                w_i = weights[i]
                contrib = w_i * tree_pred
                pred_drop = np.where(keep[i], pred_drop + contrib, pred_drop)

        # Fit new tree on residual
        grad, hess = model.loss_.grad_hess(y, pred_drop)
        if model.sampling == "goss":
            sm = sample_goss(n, grad, model.goss_top_rate,
                             model.goss_other_rate, rng)
            mask, w = sm.mask, sm.weights
        elif model.subsample < 1.0:
            sm = sample_random(n, model.subsample, rng)
            mask, w = sm.mask, sm.weights
        else:
            mask = np.ones(n, dtype=bool)
            w = np.ones(n)
        X_sub = X_raw[mask] if X_raw is not None else None
        tree = _fit_one_tree(model, codes[mask], (grad * w)[mask],
                             (hess * w)[mask], X_sub,
                             n_bins_per_feature, rng, missing_bin)

        trees.append(tree)
        weights.append(model.learning_rate)   # uniform, same as standard

        # Full pred update (no dropout) for training loss tracking
        pred = np.full(n, init_pred)
        for i, t in enumerate(trees):
            pred = pred + weights[i] * t.predict(
                codes, X_raw if t.linear else None)

    return trees, weights, pred
