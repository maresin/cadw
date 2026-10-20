"""Serialization helpers for Pipeline save/load (v0.9.1c).

Explicit to_dict / from_dict per model type (no pickle — safety).
JSON for everything; nested lists for numpy arrays / tree structures.

Serialized model sizes:
  - GBDT (n_estimators=100, max_depth=3): ~100 trees × 15 nodes × ~200 B
    = ~30 KB JSON. Acceptable for reproducibility.
  - Linear (d features): d floats + intercept = ~100 B × d.
  - PCA: (k, d) components + mean.
  - KMeans: (k, d) centers.
"""
from __future__ import annotations

import numpy as np


# ============ GBDT (regressor / classifier) ============

def gbdt_regressor_to_dict(model) -> dict:
    """Serialize GBDTRegressor (regression, L2 loss)."""
    trees = []
    for tree in model.trees_:
        trees.append({
            "structure": {
                int(nid): (
                    None if spec is None else list(spec)
                )
                for nid, spec in tree["structure"].items()
            },
            "leaf_values": {
                int(lid): float(v)
                for lid, v in tree["leaf_values"].items()
            },
        })
    return {
        "base_prediction": float(model.base_prediction_),
        "feature_names": list(model.feature_names_),
        "target_col": model.target_col_,
        "mn": [float(x) for x in model.mn_],
        "mx": [float(x) for x in model.mx_],
        "bins": int(model.bins_),
        "learning_rate": float(model.learning_rate_),
        "n_estimators": int(model.n_estimators_),
        "max_depth": int(model.max_depth_),
        "n_features_in": int(model.n_features_in_),
        "trees": trees,
    }


def gbdt_regressor_from_dict(d: dict):
    """Deserialize GBDTRegressor from dict."""
    from cadw.ml.gbdt import GBDTRegressor
    trees = []
    for td in d["trees"]:
        structure = {
            int(nid): (None if spec is None else tuple(spec))
            for nid, spec in td["structure"].items()
        }
        leaf_values = {
            int(lid): float(v) for lid, v in td["leaf_values"].items()
        }
        trees.append({
            "structure": structure,
            "leaf_values": leaf_values,
        })
    return GBDTRegressor(
        trees_=trees,
        base_prediction_=float(d["base_prediction"]),
        feature_names_=list(d["feature_names"]),
        target_col_=d["target_col"],
        mn_=np.asarray(d["mn"], dtype=np.float64),
        mx_=np.asarray(d["mx"], dtype=np.float64),
        bins_=int(d["bins"]),
        learning_rate_=float(d["learning_rate"]),
        n_estimators_=int(d["n_estimators"]),
        max_depth_=int(d["max_depth"]),
        n_features_in_=int(d["n_features_in"]),
    )


# ============ Linear ============

def linear_to_dict(model) -> dict:
    return {
        "coef": [float(x) for x in model.coef_],
        "intercept": float(model.intercept_),
        "feature_names": list(model.feature_names_),
        "target_col": model.target_col_,
        "alpha": float(model.alpha_),
        "n_rows": int(model.n_rows_),
        "penalty": model.penalty_,
        "l1_ratio": float(model.l1_ratio_),
        "n_iter_cd": int(model.n_iter_cd_),
        "converged": bool(model.converged_),
    }


def linear_from_dict(d: dict):
    from cadw.ml.linear import LinearModel
    return LinearModel(
        coef_=np.asarray(d["coef"], dtype=np.float64),
        intercept_=float(d["intercept"]),
        feature_names_=list(d["feature_names"]),
        target_col_=d["target_col"],
        alpha_=float(d["alpha"]),
        n_rows_=int(d["n_rows"]),
        penalty_=d["penalty"],
        l1_ratio_=float(d["l1_ratio"]),
        n_iter_cd_=int(d["n_iter_cd"]),
        converged_=bool(d["converged"]),
    )
