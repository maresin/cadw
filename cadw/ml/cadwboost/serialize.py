"""JSON serialization: save/load CADWBoost."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _tree_to_dict(tree) -> dict:
    d = {
        "mode": tree.mode,
        "max_depth": int(tree.max_depth),
        "linear": bool(tree.linear),
        "values": tree.values.tolist() if tree.values is not None else None,
    }
    if tree.mode == "symmetric":
        d["splits"] = tree.splits.tolist() if tree.splits is not None else None
    else:
        d["feature"] = tree.feature.tolist() if tree.feature is not None else None
        d["threshold"] = tree.threshold.tolist() if tree.threshold is not None else None
        d["default_left"] = tree.default_left.tolist() if tree.default_left is not None else None
        d["left_child"] = tree.left_child.tolist() if tree.left_child is not None else None
        d["right_child"] = tree.right_child.tolist() if tree.right_child is not None else None
        d["is_leaf"] = tree.is_leaf.tolist() if tree.is_leaf is not None else None
    if tree.coefs is not None:
        d["coefs"] = tree.coefs.tolist()
    return d


def _dict_to_tree(d: dict):
    from .tree import Tree
    mode = d["mode"]
    values = np.array(d["values"], dtype=np.float64) if d["values"] else None
    coefs = np.array(d["coefs"], dtype=np.float64) if d.get("coefs") else None
    if mode == "symmetric":
        splits = np.array(d["splits"], dtype=np.int64) if d["splits"] else None
        return Tree(mode=mode, max_depth=d["max_depth"],
                    linear=d["linear"], splits=splits,
                    values=values, coefs=coefs)
    return Tree(
        mode=mode, max_depth=d["max_depth"], linear=d["linear"],
        feature=np.array(d["feature"], dtype=np.int16) if d["feature"] else None,
        threshold=np.array(d["threshold"], dtype=np.int16) if d["threshold"] else None,
        default_left=np.array(d["default_left"], dtype=bool) if d["default_left"] else None,
        left_child=np.array(d["left_child"], dtype=np.int32) if d["left_child"] else None,
        right_child=np.array(d["right_child"], dtype=np.int32) if d["right_child"] else None,
        is_leaf=np.array(d["is_leaf"], dtype=bool) if d["is_leaf"] else None,
        values=values, coefs=coefs)


def model_to_dict(model) -> dict:
    """Serialize fitted CADWBoost to plain dict."""
    if not model.fitted_:
        raise RuntimeError("model not fitted")
    return {
        "version": "cadwboost-0.1.0-dev",
        "hyperparams": {
            "n_estimators": int(model.n_estimators),
            "learning_rate": float(model.learning_rate),
            "tree_mode": model.tree_mode,
            "max_depth": int(model.max_depth),
            "split_mode": model.split_mode,
            "binning": model.binning,
            "max_bins": int(model.max_bins),
            "leaf_predictor": model.leaf_predictor,
            "loss": model.loss,
            "loss_kwargs": model.loss_kwargs,
            "subsample": float(model.subsample),
            "colsample_bytree": float(model.colsample_bytree),
            "sampling": model.sampling,
            "goss_top_rate": float(model.goss_top_rate),
            "goss_other_rate": float(model.goss_other_rate),
            "reg_lambda": float(model.reg_lambda),
            "min_gain": float(model.min_gain),
            "min_child_weight": float(model.min_child_weight),
            "path_smoothing": float(model.path_smoothing),
            "max_delta_step": float(model.max_delta_step),
            "dart_rate": float(model.dart_rate),
            "missing_bin": bool(model.missing_bin),
            "random_state": int(model.random_state),
        },
        "fitted": {
            "init_pred": float(model.init_pred_),
            "n_features_in": int(model.n_features_in_),
            "feature_names": list(model.feature_names_),
            "thresholds": [t.tolist() for t in model.thresholds_],
            "n_bins_per_feature": model.n_bins_per_feature_.tolist(),
            "has_missing": model.has_missing_.tolist(),
            "tree_weights": model.tree_weights_.tolist(),
            "trees": [_tree_to_dict(t) for t in model.trees_],
        },
    }


def dict_to_model(d: dict, model_cls):
    """Deserialize dict to fitted CADWBoost."""
    hp = d["hyperparams"]
    model = model_cls(**hp)
    fit = d["fitted"]
    model.init_pred_ = fit["init_pred"]
    model.n_features_in_ = fit["n_features_in"]
    model.feature_names_ = fit["feature_names"]
    model.thresholds_ = [np.array(t, dtype=np.float64) for t in fit["thresholds"]]
    model.n_bins_per_feature_ = np.array(fit["n_bins_per_feature"], dtype=np.uint16)
    model.has_missing_ = np.array(fit["has_missing"], dtype=bool)
    model.tree_weights_ = np.array(fit["tree_weights"], dtype=np.float64)
    model.trees_ = [_dict_to_tree(t) for t in fit["trees"]]
    model.loss_ = model._loss_instance()
    model.fitted_ = True
    model._compute_importance()
    return model


def save_json(model, path):
    Path(path).write_text(json.dumps(model_to_dict(model), indent=2))


def load_json(path, model_cls):
    return dict_to_model(json.loads(Path(path).read_text()), model_cls)
