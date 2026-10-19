#!/usr/bin/env python3
"""v0.7.5: GBDT binary classification (logloss).

Refactor gbdt.py:
    - Extract `_fit_boosting(df, X, y, loss=...)` — common level-wise loop.
    - loss='l2'             : GBDTRegressor (existing)
    - loss='logloss_binary' : GBDTClassifier (new)

Loss-specific parts:
    _grad_hess(pred, y, loss) -> (grad, hess)
        l2      : pred - y, 1.0
        logloss : p - y, p*(1-p), p = sigmoid(pred)
    _base_prediction(y, loss) -> float
        l2      : mean(y)
        logloss : log(p_mean / (1-p_mean)), clip p_mean to [eps, 1-eps]
    _link(pred, loss) -> final
        l2      : pred (regression value)
        logloss : sigmoid(pred) (probability)

Unchanged: _AllNodeHistOp, _AssignLeavesOp, _LeafValuesOp (loss-agnostic;
they receive precomputed grad,hess).

GBDTClassifier:
    - predict_proba(df|np) -> probabilities / {target}_prob column
    - predict(df|np, threshold=0.5) -> {0,1} / {target}_pred column
    - Logloss, accuracy verified in tests.

Not in v0.7.5:
    - Multiclass softmax (v0.7.6).
    - Categorical splits.
    - Early stopping.

Run: uv run python scripts/patch_v075_gbdt_classification.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)


def sh(cmd, check=True):
    print(f"  $ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    r = subprocess.run(
        cmd, shell=isinstance(cmd, str), capture_output=True, text=True,
    )
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines()[:10]:
            print(f"    {line}")
    if check and r.returncode != 0:
        print(f"    STDERR: {r.stderr.strip()[-800:]}")
        sys.exit(r.returncode)
    return r.returncode


def compile_check(path):
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         f"import py_compile; py_compile.compile('{path}', doraise=True)"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"COMPILE FAIL: {path}")
        print(r.stderr[-800:])
        sys.exit(1)
    print(f"  compile OK: {path}")


def ensure_branch(name):
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    if r.stdout.strip() == name:
        print(f"  branch: already on {name}")
        return
    r = subprocess.run(["git", "rev-parse", "--verify", name],
                       capture_output=True)
    if r.returncode == 0:
        sh(["git", "checkout", name])
    else:
        sh(["git", "checkout", "-b", name])


def require_clean():
    r = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True,
    )
    if r.stdout.strip():
        print("ERROR: tracked files dirty:")
        print(r.stdout)
        sys.exit(1)


# ============ new gbdt.py (full rewrite — refactored) ============

GBDT_SRC = '''"""Gradient Boosting Decision Trees — regression + classification (v0.7.5).

Composition: Reduce (histograms) + Reduce (assign leaves) + Reduce
(leaf values). Level-wise tree growth.

Losses:
    - 'l2'              : regression (GBDTRegressor)
    - 'logloss_binary'  : binary classification (GBDTClassifier)

Peak RAM = batch + O(n × 32B) internal state (pred, cur_leaf, grad, hess)
+ O(nodes × feat × bins) histograms. Guarded via MemoryGuard.
The O(n) state is inherent to Newton-based boosting (XGBoost/LightGBM).

For multiclass (softmax) — v0.7.6.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ops.reduce import ReduceOp, reduce


# ============ helpers: type conversion, gradient ============

def _column_to_f64(arr: pa.Array) -> np.ndarray:
    np_arr = arr.to_numpy(zero_copy_only=False)
    if np_arr.dtype == object:
        out = np.empty(len(np_arr), dtype=np.float64)
        for i, v in enumerate(np_arr):
            out[i] = np.nan if v is None else float(v)
        return out
    return np_arr.astype(np.float64, copy=False)


def _stack_numeric(
    batch: pa.RecordBatch, cols: Sequence[str],
) -> np.ndarray:
    return np.column_stack([_column_to_f64(batch.column(c)) for c in cols])


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return 0.5 * (1.0 + np.tanh(0.5 * z))


_LOSSES = ("l2", "logloss_binary")


def _grad_hess(
    pred: np.ndarray, y: np.ndarray, loss: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Grad and hess per row for given loss."""
    if loss == "l2":
        return pred - y, np.ones_like(y, dtype=np.float64)
    if loss == "logloss_binary":
        p = _sigmoid(pred)
        return p - y, np.maximum(p * (1.0 - p), 1e-16)
    raise ValueError(f"_grad_hess: unknown loss {loss!r}")


def _base_prediction(y_sum: float, n: int, loss: str) -> float:
    if loss == "l2":
        return y_sum / n
    if loss == "logloss_binary":
        p = y_sum / n
        eps = 1e-6
        p = min(max(p, eps), 1.0 - eps)
        return float(np.log(p / (1.0 - p)))
    raise ValueError(f"_base_prediction: unknown loss {loss!r}")


def _link(pred: np.ndarray, loss: str) -> np.ndarray:
    if loss == "l2":
        return pred
    if loss == "logloss_binary":
        return _sigmoid(pred)
    raise ValueError(f"_link: unknown loss {loss!r}")


# ============ ReduceOps (loss-agnostic) ============

class _MinMaxOp(ReduceOp):
    """Global min/max per feature (ignore nulls)."""

    def __init__(self, feature_cols: Sequence[str]) -> None:
        self.feature_cols = list(feature_cols)
        self.d = len(feature_cols)

    def init(self):
        return (np.full(self.d, np.inf), np.full(self.d, -np.inf))

    def update(self, state, batch):
        mn, mx = state
        X = _stack_numeric(batch, self.feature_cols)
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        if X.shape[0] == 0:
            return state
        return (np.minimum(mn, X.min(axis=0)),
                np.maximum(mx, X.max(axis=0)))

    def merge(self, a, b):
        return (np.minimum(a[0], b[0]), np.maximum(a[1], b[1]))

    def state_nbytes(self, state):
        return int(state[0].nbytes + state[1].nbytes)

    def estimate_state_nbytes(self):
        return 8 * self.d * 2

    def finalize(self, state):
        return {"min": state[0], "max": state[1]}


class _AllNodeHistOp(ReduceOp):
    """Accumulate (grad, hess) histograms for all nodes in one pass.

    State: g_hist (S, F, B), h_hist (S, F, B), g_tot, h_tot, n_tot per slot.
    """

    def __init__(
        self,
        feature_cols: Sequence[str],
        node_ids: Sequence[int],
        mn: np.ndarray,
        mx: np.ndarray,
        bins: int,
    ) -> None:
        self.feature_cols = list(feature_cols)
        self.f = len(feature_cols)
        self.node_ids = list(node_ids)
        self.node_to_slot = {nid: i for i, nid in enumerate(self.node_ids)}
        self.mn = np.asarray(mn, dtype=np.float64)
        self.mx = np.asarray(mx, dtype=np.float64)
        self.range = np.where(self.mx - self.mn > 0,
                              self.mx - self.mn, 1.0)
        self.bins = int(bins)
        self.S = len(self.node_ids)

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
        X = _stack_numeric(batch, self.feature_cols)
        g = _column_to_f64(batch.column("__g"))
        h = _column_to_f64(batch.column("__h"))
        lf = batch.column("__leaf").to_numpy(zero_copy_only=False)
        n = X.shape[0]
        if n == 0:
            return state
        for nid, slot in self.node_to_slot.items():
            m = lf == nid
            if not m.any():
                continue
            Xn = X[m]
            gn = g[m]
            hn = h[m]
            valid = ~np.isnan(Xn).any(axis=1)
            if not valid.any():
                continue
            Xn = Xn[valid]
            gn = gn[valid]
            hn = hn[valid]
            gt[slot] += gn.sum()
            ht[slot] += hn.sum()
            nt[slot] += Xn.shape[0]
            for j in range(self.f):
                b_idx = np.clip(
                    ((Xn[:, j] - self.mn[j]) / self.range[j] * self.bins)
                    .astype(np.int64),
                    0, self.bins - 1,
                )
                np.add.at(gh[slot, j], b_idx, gn)
                np.add.at(hh[slot, j], b_idx, hn)
        return (gh, hh, gt, ht, nt)

    def merge(self, a, b):
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3], a[4] + b[4])

    def state_nbytes(self, state):
        return int(sum(x.nbytes for x in state))

    def estimate_state_nbytes(self):
        per_slot = self.f * self.bins * 8 * 2 + 8 * 3
        return self.S * per_slot

    def finalize(self, state):
        gh, hh, gt, ht, nt = state
        return {
            "g_hist": gh, "h_hist": hh,
            "g_tot": gt, "h_tot": ht, "n_tot": nt,
        }


class _AssignLeavesOp(ReduceOp):
    """Assign rows to leaf_ids per current tree structure."""

    def __init__(
        self,
        feature_cols: Sequence[str],
        mn: np.ndarray,
        mx: np.ndarray,
        bins: int,
        tree: dict,
        root_id: int = 0,
    ) -> None:
        self.feature_cols = list(feature_cols)
        self.mn = np.asarray(mn, dtype=np.float64)
        self.mx = np.asarray(mx, dtype=np.float64)
        self.range = np.where(self.mx - self.mn > 0,
                              self.mx - self.mn, 1.0)
        self.bins = int(bins)
        self.tree = dict(tree)
        self.root_id = int(root_id)

    def init(self):
        return []

    def update(self, state, batch):
        X = _stack_numeric(batch, self.feature_cols)
        n = X.shape[0]
        leaf = np.full(n, self.root_id, dtype=np.int64)
        nan_rows = np.isnan(X).any(axis=1)
        max_iter = 32
        with np.errstate(divide="ignore", invalid="ignore"):
            for _ in range(max_iter):
                any_change = False
                for nid, spec in self.tree.items():
                    if spec is None:
                        continue
                    j, thr, left, right = spec
                    m = (leaf == nid) & ~nan_rows
                    if not m.any():
                        continue
                    b_idx = np.clip(
                        ((X[m, j] - self.mn[j]) / self.range[j] * self.bins)
                        .astype(np.int64),
                        0, self.bins - 1,
                    )
                    new_ids = np.where(b_idx <= thr, left, right)
                    leaf[m] = new_ids
                    any_change = True
                if not any_change:
                    break
        state.append(leaf)
        return state

    def merge(self, a, b):
        return a + b

    def state_nbytes(self, state):
        return int(sum(x.nbytes for x in state) + 8)

    def estimate_state_nbytes(self):
        return 0

    def finalize(self, state):
        return np.concatenate(state) if state else np.array([], dtype=np.int64)


class _LeafValuesOp(ReduceOp):
    """Per-leaf: sum_grad, sum_hess → Newton step for any loss."""

    def __init__(self, reg_lambda: float = 1.0) -> None:
        self.reg_lambda = float(reg_lambda)

    def init(self):
        return {}

    def update(self, state, batch):
        g = _column_to_f64(batch.column("__g"))
        h = _column_to_f64(batch.column("__h"))
        lf = batch.column("__leaf").to_numpy(zero_copy_only=False)
        for nid in np.unique(lf):
            m = lf == nid
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

    def finalize(self, state):
        return {
            k: -g / (h + self.reg_lambda)
            for k, (g, h) in state.items()
        }


# ============ tree utility ============

def _find_best_split(
    slot: int,
    gh: np.ndarray, hh: np.ndarray,
    gt: float, ht: float, n_tot: int,
    bins: int,
    min_child_weight: float,
    min_gain: float,
    reg_lambda: float,
) -> Optional[tuple[int, int, float]]:
    if ht <= 1e-12:
        return None
    parent_denom = ht + reg_lambda
    parent_gain = gt * gt / parent_denom
    best = None
    F = gh.shape[0]
    for j in range(F):
        g_l = np.cumsum(gh[j])[:-1]
        h_l = np.cumsum(hh[j])[:-1]
        g_r = gt - g_l
        h_r = ht - h_l
        valid = (h_l >= min_child_weight) & (h_r >= min_child_weight)
        if not valid.any():
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            g_l_sq = g_l * g_l / (h_l + reg_lambda)
            g_r_sq = g_r * g_r / (h_r + reg_lambda)
        gains = np.where(
            valid, g_l_sq + g_r_sq - parent_gain, -np.inf,
        )
        k = int(np.argmax(gains))
        if gains[k] > (best[2] if best is not None else min_gain):
            best = (j, k, float(gains[k]))
    return best


def _assign_leaf_from_array(
    X: np.ndarray, tree_struct: dict,
    mn: np.ndarray, mx: np.ndarray, bins: int,
) -> np.ndarray:
    n = X.shape[0]
    leaf = np.zeros(n, dtype=np.int64)
    nan_rows = np.isnan(X).any(axis=1)
    rng = np.where(mx - mn > 0, mx - mn, 1.0)
    for nid, spec in tree_struct.items():
        if spec is None:
            continue
        j, thr, left, right = spec
        m = (leaf == nid) & ~nan_rows
        if not m.any():
            continue
        b_idx = np.clip(
            ((X[m, j] - mn[j]) / rng[j] * bins)
            .astype(np.int64), 0, bins - 1,
        )
        leaf[m] = np.where(b_idx <= thr, left, right)
    return leaf


def _leaf_vector(lv_map: dict, leaf: np.ndarray) -> np.ndarray:
    if not lv_map:
        return np.zeros_like(leaf, dtype=np.float64)
    max_lid = max(lv_map.keys())
    arr = np.zeros(max_lid + 1, dtype=np.float64)
    for lid, v in lv_map.items():
        arr[lid] = v
    return arr[leaf]


# ============ models ============

@dataclass
class GBDTRegressor:
    """GBDT regression (L2 loss)."""

    trees_: list
    base_prediction_: float
    feature_names_: list
    target_col_: str
    mn_: np.ndarray
    mx_: np.ndarray
    bins_: int
    learning_rate_: float
    n_estimators_: int
    max_depth_: int
    n_features_in_: int
    fitted_: bool = True

    def predict(self, X):
        if isinstance(X, np.ndarray):
            return self._predict_np(X)
        return self._predict_df(X)

    def _predict_np(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"predict: X has {X.shape[1]} features, "
                f"expected {self.n_features_in_}"
            )
        pred = np.full(X.shape[0], self.base_prediction_, dtype=np.float64)
        for tree in self.trees_:
            leaf = _assign_leaf_from_array(
                X, tree["structure"], self.mn_, self.mx_, self.bins_,
            )
            pred += self.learning_rate_ * _leaf_vector(
                tree["leaf_values"], leaf,
            )
        return pred

    def _predict_df(self, X):
        from cadw.dataframe import CadwDataFrame
        from cadw.dataset import Dataset
        if not isinstance(X, CadwDataFrame):
            raise TypeError(
                f"predict: X must be np.ndarray or CadwDataFrame; "
                f"got {type(X).__name__}"
            )
        for c in self.feature_names_:
            if c not in X.columns:
                raise KeyError(f"predict: column {c!r} not in df")

        out_col = f"{self.target_col_}_pred"
        orig_schema = X._dataset.schema
        out_fields = [orig_schema.field(c) for c in X._columns]
        out_fields.append(pa.field(out_col, pa.float64(), nullable=True))
        out_schema = pa.schema(out_fields)
        trees = self.trees_
        lr = self.learning_rate_
        base = self.base_prediction_
        mn = self.mn_
        mx = self.mx_
        bins = self.bins_
        feat = self.feature_names_

        def _iter_pred():
            for b in X._iter_filtered_batches():
                Xm = _stack_numeric(b, feat)
                n = Xm.shape[0]
                nan_rows = np.isnan(Xm).any(axis=1)
                pred = np.full(n, base, dtype=np.float64)
                for tree in trees:
                    leaf = _assign_leaf_from_array(
                        Xm, tree["structure"], mn, mx, bins,
                    )
                    pred += lr * _leaf_vector(tree["leaf_values"], leaf)
                arrays = [b.column(c) for c in X._columns]
                arrays.append(pa.array(
                    pred, type=pa.float64(),
                    mask=nan_rows if nan_rows.any() else None,
                ))
                yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

        new_ds = Dataset.from_batches(
            out_schema, _iter_pred(), X._new_dataset_dir()
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=X._memory_guard,
            spill_dir=X._spill_dir,
        )

    def to_dict(self) -> dict:
        return {
            "base_prediction": float(self.base_prediction_),
            "feature_names": self.feature_names_,
            "target_col": self.target_col_,
            "n_estimators": int(self.n_estimators_),
            "max_depth": int(self.max_depth_),
            "learning_rate": float(self.learning_rate_),
            "bins": int(self.bins_),
            "n_trees": len(self.trees_),
            "loss": "l2",
        }


@dataclass
class GBDTClassifier:
    """GBDT binary classification (logloss).

    predict_proba -> P(y=1|x); predict -> {0,1} by threshold.
    """

    trees_: list
    base_prediction_: float
    feature_names_: list
    target_col_: str
    mn_: np.ndarray
    mx_: np.ndarray
    bins_: int
    learning_rate_: float
    n_estimators_: int
    max_depth_: int
    n_features_in_: int
    fitted_: bool = True

    def _raw_score_np(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"predict: X has {X.shape[1]} features, "
                f"expected {self.n_features_in_}"
            )
        pred = np.full(X.shape[0], self.base_prediction_, dtype=np.float64)
        for tree in self.trees_:
            leaf = _assign_leaf_from_array(
                X, tree["structure"], self.mn_, self.mx_, self.bins_,
            )
            pred += self.learning_rate_ * _leaf_vector(
                tree["leaf_values"], leaf,
            )
        return pred

    def predict_proba(self, X):
        if isinstance(X, np.ndarray):
            return _sigmoid(self._raw_score_np(X))
        return self._predict_df(X, proba_only=True, threshold=0.5)

    def predict(self, X, threshold: float = 0.5):
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(
                f"predict: threshold must be in [0, 1]; got {threshold}"
            )
        if isinstance(X, np.ndarray):
            proba = self.predict_proba(X)
            return (proba >= threshold).astype(np.int64)
        return self._predict_df(X, proba_only=False, threshold=threshold)

    def _predict_df(self, X, *, proba_only: bool, threshold: float):
        from cadw.dataframe import CadwDataFrame
        from cadw.dataset import Dataset
        if not isinstance(X, CadwDataFrame):
            raise TypeError(
                f"predict: X must be np.ndarray or CadwDataFrame; "
                f"got {type(X).__name__}"
            )
        for c in self.feature_names_:
            if c not in X.columns:
                raise KeyError(f"predict: column {c!r} not in df")

        prob_col = f"{self.target_col_}_prob"
        pred_col = f"{self.target_col_}_pred"
        orig_schema = X._dataset.schema
        out_fields = [orig_schema.field(c) for c in X._columns]
        out_fields.append(pa.field(prob_col, pa.float64(), nullable=True))
        if not proba_only:
            out_fields.append(pa.field(pred_col, pa.int64(), nullable=True))
        out_schema = pa.schema(out_fields)
        trees = self.trees_
        lr = self.learning_rate_
        base = self.base_prediction_
        mn = self.mn_
        mx = self.mx_
        bins = self.bins_
        feat = self.feature_names_

        def _iter_pred():
            for b in X._iter_filtered_batches():
                Xm = _stack_numeric(b, feat)
                n = Xm.shape[0]
                nan_rows = np.isnan(Xm).any(axis=1)
                pred = np.full(n, base, dtype=np.float64)
                for tree in trees:
                    leaf = _assign_leaf_from_array(
                        Xm, tree["structure"], mn, mx, bins,
                    )
                    pred += lr * _leaf_vector(tree["leaf_values"], leaf)
                proba = _sigmoid(pred)
                arrays = [b.column(c) for c in X._columns]
                arrays.append(pa.array(
                    proba, type=pa.float64(),
                    mask=nan_rows if nan_rows.any() else None,
                ))
                if not proba_only:
                    pred01 = (proba >= threshold).astype(np.int64)
                    arrays.append(pa.array(
                        pred01, type=pa.int64(),
                        mask=nan_rows if nan_rows.any() else None,
                    ))
                yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

        new_ds = Dataset.from_batches(
            out_schema, _iter_pred(), X._new_dataset_dir()
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=X._memory_guard,
            spill_dir=X._spill_dir,
        )

    def to_dict(self) -> dict:
        return {
            "base_prediction": float(self.base_prediction_),
            "feature_names": self.feature_names_,
            "target_col": self.target_col_,
            "n_estimators": int(self.n_estimators_),
            "max_depth": int(self.max_depth_),
            "learning_rate": float(self.learning_rate_),
            "bins": int(self.bins_),
            "n_trees": len(self.trees_),
            "loss": "logloss_binary",
        }


# ============ shared fit ============

def _validate_fit_args(
    df, X_cols, y_col, *, n_estimators, max_depth, learning_rate, bins,
    min_child_weight, reg_lambda,
):
    if isinstance(X_cols, str):
        X_cols = [X_cols]
    else:
        X_cols = list(X_cols)
    if not X_cols:
        raise ValueError("fit_gbdt: X_cols must be non-empty")
    if y_col in X_cols:
        raise ValueError(f"fit_gbdt: target {y_col!r} must not be in X_cols")
    for c in X_cols + [y_col]:
        if c not in df.columns:
            raise KeyError(f"fit_gbdt: column {c!r} not in df")
    schema = df._dataset.schema
    for c in X_cols + [y_col]:
        t = schema.field(c).type
        if not (pa.types.is_integer(t) or pa.types.is_floating(t)):
            raise TypeError(
                f"fit_gbdt: column {c!r} has type {t}; numeric required"
            )
    if n_estimators < 1:
        raise ValueError(f"n_estimators must be >= 1; got {n_estimators}")
    if max_depth < 1:
        raise ValueError(f"max_depth must be >= 1; got {max_depth}")
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0; got {learning_rate}")
    if bins < 2:
        raise ValueError(f"bins must be >= 2; got {bins}")
    if min_child_weight < 0:
        raise ValueError(f"min_child_weight must be >= 0; got {min_child_weight}")
    if reg_lambda < 0:
        raise ValueError(f"reg_lambda must be >= 0; got {reg_lambda}")
    return X_cols


def _validate_binary_y(df, y_col: str) -> None:
    """Ensure y has both classes {0, 1}."""
    seen: set = set()
    for b in df._iter_filtered_batches():
        col = b.column(y_col)
        try:
            uniq = col.unique()
            vals = uniq.to_pylist() if uniq is not None else col.to_pylist()
        except Exception:
            vals = col.to_pylist()
        for v in vals:
            if v is None:
                continue
            fv = float(v)
            if fv != 0.0 and fv != 1.0:
                raise ValueError(
                    f"fit_gbdt_classifier: y must be binary (0/1); got {v!r}"
                )
            seen.add(fv)
    if len(seen) < 2:
        raise ValueError(
            f"fit_gbdt_classifier: y must have both classes 0 and 1; got {seen}"
        )


def _fit_boosting(
    df, X_cols, y_col, *,
    loss: str,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    bins: int,
    min_child_weight: float,
    reg_lambda: float,
    min_gain: float,
    memory_guard: Optional[MemoryGuard],
):
    """Common level-wise boosting fit for l2 and logloss_binary."""
    if loss not in _LOSSES:
        raise ValueError(f"_fit_boosting: unknown loss {loss!r}")

    X_cols = _validate_fit_args(
        df, X_cols, y_col,
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, bins=bins,
        min_child_weight=min_child_weight, reg_lambda=reg_lambda,
    )

    if loss == "logloss_binary":
        _validate_binary_y(df, y_col)

    mg = memory_guard or df._memory_guard
    n_total = df.count()
    if n_total == 0:
        raise ValueError("fit_gbdt: no valid (non-null) rows (filtered count 0)")
    est_n_state = n_total * 32
    try:
        mg.check_state(est_n_state, name="gbdt.internal_state")
    except MemoryLimitExceeded:
        raise MemoryLimitExceeded(
            f"fit_gbdt: internal state O(n) = {est_n_state} B "
            f"({n_total} rows × 32B) > state_limit. GBDT requires "
            f"materializing gradients. See §13.4/92."
        )

    # 1. Global min/max
    mm_op = _MinMaxOp(X_cols)
    mm_res = reduce(df._iter_filtered_batches(), mm_op, mg, log=False)
    mn = mm_res["min"]
    mx = mm_res["max"]
    if np.any(np.isinf(mn)) or np.any(np.isinf(mx)):
        raise ValueError("fit_gbdt: no valid (non-null) rows")

    # 2. Base prediction (loss-specific) — needs y_sum and n_valid
    y_sum = 0.0
    n_valid = 0
    offset = 0
    for b in df._iter_filtered_batches():
        n = b.num_rows
        X = _stack_numeric(b, X_cols)
        y = _column_to_f64(b.column(y_col))
        valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        y_v = y[valid]
        y_sum += float(y_v.sum())
        n_valid += int(valid.sum())
        offset += n
    if n_valid == 0:
        raise ValueError("fit_gbdt: no valid rows")
    base_pred = _base_prediction(y_sum, n_valid, loss)

    # 3. Init pred
    pred = np.full(n_total, base_pred, dtype=np.float64)

    trees = []

    def _iter_with_state():
        """Yield enriched batches with __g, __h, __leaf."""
        off = 0
        for b in df._iter_filtered_batches():
            n = b.num_rows
            X = _stack_numeric(b, X_cols)
            y = _column_to_f64(b.column(y_col))
            p = pred[off:off + n]
            lf = cur_leaf[off:off + n]
            valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
            g, h = _grad_hess(p, y, loss)
            g = np.where(valid, g, 0.0)
            h = np.where(valid, h, 0.0)
            cols = [b.column(c) for c in X_cols]
            cols.append(pa.array(g, type=pa.float64()))
            cols.append(pa.array(h, type=pa.float64()))
            cols.append(pa.array(lf, type=pa.int64()))
            names = list(X_cols) + ["__g", "__h", "__leaf"]
            yield pa.RecordBatch.from_arrays(cols, names=names)
            off += n

    def _iter_for_leaves():
        off = 0
        for b in df._iter_filtered_batches():
            n = b.num_rows
            X = _stack_numeric(b, X_cols)
            y = _column_to_f64(b.column(y_col))
            p = pred[off:off + n]
            lf = cur_leaf[off:off + n]
            valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
            g, h = _grad_hess(p, y, loss)
            g = np.where(valid, g, 0.0)
            h = np.where(valid, h, 0.0)
            yield pa.RecordBatch.from_arrays(
                [pa.array(g, type=pa.float64()),
                 pa.array(h, type=pa.float64()),
                 pa.array(lf, type=pa.int64())],
                names=["__g", "__h", "__leaf"],
            )
            off += n

    for round_i in range(n_estimators):
        tree: dict = {0: None}
        cur_leaf = np.zeros(n_total, dtype=np.int64)
        next_node_id = 1

        for level in range(max_depth):
            internal = [nid for nid, t in tree.items() if t is None]
            if not internal:
                break

            hist_op = _AllNodeHistOp(X_cols, internal, mn, mx, bins)
            hist_res = reduce(_iter_with_state(), hist_op, mg, log=False)
            gh = hist_res["g_hist"]
            hh = hist_res["h_hist"]
            gt = hist_res["g_tot"]
            ht = hist_res["h_tot"]
            nt = hist_res["n_tot"]

            for slot, nid in enumerate(internal):
                if ht[slot] < min_child_weight * 2:
                    continue
                best = _find_best_split(
                    slot, gh[slot], hh[slot],
                    float(gt[slot]), float(ht[slot]), int(nt[slot]),
                    bins, min_child_weight, min_gain, reg_lambda,
                )
                if best is None:
                    continue
                j, thr, gain = best
                left_id = next_node_id
                right_id = next_node_id + 1
                next_node_id += 2
                tree[nid] = (j, thr, left_id, right_id)
                tree[left_id] = None
                tree[right_id] = None

            assign_op = _AssignLeavesOp(X_cols, mn, mx, bins, tree)
            cur_leaf = reduce(
                df._iter_filtered_batches(), assign_op, mg, log=False,
            )

        lv_op = _LeafValuesOp(reg_lambda=reg_lambda)
        leaf_values = reduce(_iter_for_leaves(), lv_op, mg, log=False)

        trees.append({
            "structure": dict(tree),
            "leaf_values": dict(leaf_values),
        })

        if leaf_values:
            max_lid = max(leaf_values.keys())
            lv_arr = np.zeros(max_lid + 1, dtype=np.float64)
            for lid, v in leaf_values.items():
                lv_arr[lid] = v
            pred += learning_rate * lv_arr[cur_leaf]

    return {
        "trees_": trees,
        "base_prediction_": float(base_pred),
        "feature_names_": list(X_cols),
        "target_col_": y_col,
        "mn_": mn,
        "mx_": mx,
        "bins_": int(bins),
        "learning_rate_": float(learning_rate),
        "n_estimators_": int(n_estimators),
        "max_depth_": int(max_depth),
        "n_features_in_": len(X_cols),
    }


def fit_gbdt_regressor(
    df, X_cols, y_col: str,
    *, n_estimators: int = 100, max_depth: int = 3,
    learning_rate: float = 0.1, bins: int = 32,
    min_child_weight: float = 1.0, reg_lambda: float = 1.0,
    min_gain: float = 0.0,
    memory_guard: Optional[MemoryGuard] = None,
) -> GBDTRegressor:
    """GBDT regression (L2 loss) — v0.7.3 (unchanged API)."""
    out = _fit_boosting(
        df, X_cols, y_col, loss="l2",
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, bins=bins,
        min_child_weight=min_child_weight, reg_lambda=reg_lambda,
        min_gain=min_gain, memory_guard=memory_guard,
    )
    return GBDTRegressor(**out)


def fit_gbdt_classifier(
    df, X_cols, y_col: str,
    *, n_estimators: int = 100, max_depth: int = 3,
    learning_rate: float = 0.1, bins: int = 32,
    min_child_weight: float = 1.0, reg_lambda: float = 1.0,
    min_gain: float = 0.0,
    memory_guard: Optional[MemoryGuard] = None,
) -> GBDTClassifier:
    """GBDT binary classification (logloss) — v0.7.5.

    y must be {0, 1}, both classes present. predict_proba → P(y=1|x);
    predict → {0,1} by threshold (default 0.5).
    """
    out = _fit_boosting(
        df, X_cols, y_col, loss="logloss_binary",
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, bins=bins,
        min_child_weight=min_child_weight, reg_lambda=reg_lambda,
        min_gain=min_gain, memory_guard=memory_guard,
    )
    return GBDTClassifier(**out)
'''


# ============ patch dataframe.py ============

def patch_dataframe():
    print("\n=== cadw/dataframe.py: fit_gbdt_classifier ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "def fit_gbdt_classifier" in s:
        print("  already present — skip")
        return

    anchor = "    # ---------------- merge ----------------"
    if anchor not in s:
        raise SystemExit("merge anchor not found")

    method = '''    # ---------------- ML: fit_gbdt_classifier ----------------

    def fit_gbdt_classifier(
        self,
        X_cols: "str | Sequence[str]",
        y_col: str,
        *,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        bins: int = 32,
        min_child_weight: float = 1.0,
        reg_lambda: float = 1.0,
        min_gain: float = 0.0,
    ):
        """Fit GBDT binary classification (logloss) — v0.7.5.

        y must be {0, 1}, both classes present.
        Returns GBDTClassifier (predict_proba, predict).
        """
        from cadw.ml.gbdt import fit_gbdt_classifier
        return fit_gbdt_classifier(
            self, X_cols, y_col,
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, bins=bins,
            min_child_weight=min_child_weight, reg_lambda=reg_lambda,
            min_gain=min_gain,
            memory_guard=self._memory_guard,
        )

'''
    s = s.replace(anchor, method + anchor, 1)
    p.write_text(s)
    compile_check(p)
    print("  OK: fit_gbdt_classifier method added")


# ============ exports ============

def patch_exports():
    print("\n=== cadw/ml/__init__.py + cadw/__init__.py ===")
    p = Path("cadw/ml/__init__.py")
    p.write_text('''"""CADW ML layer (v0.7.5)."""
from .linear import XTXOp, LinearModel, fit_linear
from .logistic import LogisticGradOp, LogisticModel, fit_logistic
from .kmeans import KMeansStepOp, KMeansModel, fit_kmeans
from .pca import PCA, fit_pca
from .gbdt import (
    GBDTRegressor, fit_gbdt_regressor,
    GBDTClassifier, fit_gbdt_classifier,
)

__all__ = [
    "XTXOp", "LinearModel", "fit_linear",
    "LogisticGradOp", "LogisticModel", "fit_logistic",
    "KMeansStepOp", "KMeansModel", "fit_kmeans",
    "PCA", "fit_pca",
    "GBDTRegressor", "fit_gbdt_regressor",
    "GBDTClassifier", "fit_gbdt_classifier",
]
''')
    compile_check(p)

    p2 = Path("cadw/__init__.py")
    s = p2.read_text()
    if "GBDTClassifier" in s:
        print("  top-level already updated")
        return
    s = s.replace(
        "    GBDTRegressor, fit_gbdt_regressor,\n)",
        "    GBDTRegressor, fit_gbdt_regressor,\n"
        "    GBDTClassifier, fit_gbdt_classifier,\n)",
        1,
    )
    s = s.replace(
        '    "GBDTRegressor", "fit_gbdt_regressor",\n',
        '    "GBDTRegressor", "fit_gbdt_regressor",\n'
        '    "GBDTClassifier", "fit_gbdt_classifier",\n',
        1,
    )
    p2.write_text(s)
    compile_check(p2)
    print("  OK: exports updated")


# ============ tests ============

TESTS_SRC = '''"""v0.7.5: GBDT binary classification (logloss)."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard, fit_gbdt_classifier
from cadw.core.memory_guard import MemoryLimitExceeded
from cadw.ml.gbdt import GBDTClassifier, fit_gbdt_regressor


def make_df(tmp_path, data, chunk=50, name="L"):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def _synth_binary(n, coefs, intercept=0.0, noise=0.1, seed=42):
    rng = np.random.default_rng(seed)
    d = len(coefs)
    X = rng.normal(size=(n, d))
    z = X @ np.array(coefs) + intercept + noise * rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-z))
    y = (rng.random(n) < p).astype(np.int64)
    return X, y


def _make_binary_df(tmp_path, n=500, coefs=(2.0, -1.5), intercept=0.5, seed=42):
    X, y = _synth_binary(n, coefs, intercept=intercept, seed=seed)
    data = {f"x{i}": X[:, i] for i in range(X.shape[1])}
    data["y"] = y.tolist()
    return make_df(tmp_path, data), X, y


# ---------- basic ----------

def test_fit_gbdt_classifier_accuracy(tmp_path):
    """Clean signal: accuracy > 0.85."""
    df, X, y = _make_binary_df(tmp_path, n=500, coefs=(3.0,), intercept=0.0)
    model = fit_gbdt_classifier(
        df, ["x0"], "y",
        n_estimators=30, max_depth=3, learning_rate=0.1, bins=16,
    )
    assert isinstance(model, GBDTClassifier)
    pred = model.predict(X)
    acc = (pred == y).mean()
    assert acc > 0.85, f"accuracy={acc:.3f}"


def test_fit_gbdt_classifier_predict_proba_range(tmp_path):
    df, X, y = _make_binary_df(tmp_path)
    model = fit_gbdt_classifier(df, ["x0", "x1"], "y", n_estimators=10)
    proba = model.predict_proba(X[:20])
    assert proba.shape == (20,)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_fit_gbdt_classifier_n_estimators_improves(tmp_path):
    """More trees → higher accuracy (monotone-ish)."""
    df, X, y = _make_binary_df(tmp_path, n=800, coefs=(2.0,), intercept=0.0)
    m5 = fit_gbdt_classifier(df, ["x0"], "y", n_estimators=5, learning_rate=0.1)
    m50 = fit_gbdt_classifier(df, ["x0"], "y", n_estimators=50, learning_rate=0.1)
    acc5 = (m5.predict(X) == y).mean()
    acc50 = (m50.predict(X) == y).mean()
    assert acc50 >= acc5, f"acc(5)={acc5:.3f} vs acc(50)={acc50:.3f}"


def test_fit_gbdt_classifier_deterministic(tmp_path):
    df, X, y = _make_binary_df(tmp_path)
    m1 = fit_gbdt_classifier(df, ["x0", "x1"], "y", n_estimators=10)
    m2 = fit_gbdt_classifier(df, ["x0", "x1"], "y", n_estimators=10)
    np.testing.assert_allclose(
        m1.predict_proba(X), m2.predict_proba(X), rtol=1e-9,
    )


def test_fit_gbdt_classifier_multibatch_consistent(tmp_path):
    X, y = _synth_binary(500, (2.0,), seed=42)
    data = {"x0": X[:, 0].tolist(), "y": y.tolist()}
    df1 = make_df(tmp_path, data, chunk=10, name="a")
    df2 = make_df(tmp_path, data, chunk=100, name="b")
    m1 = fit_gbdt_classifier(df1, ["x0"], "y", n_estimators=10)
    m2 = fit_gbdt_classifier(df2, ["x0"], "y", n_estimators=10)
    np.testing.assert_allclose(
        m1.predict_proba(X), m2.predict_proba(X), rtol=1e-9,
    )


# ---------- predict / threshold ----------

def test_predict_threshold(tmp_path):
    df, X, y = _make_binary_df(tmp_path)
    model = fit_gbdt_classifier(df, ["x0", "x1"], "y", n_estimators=20)
    pred_05 = model.predict(X, threshold=0.5)
    pred_low = model.predict(X, threshold=0.1)
    pred_high = model.predict(X, threshold=0.9)
    assert pred_low.sum() >= pred_05.sum()
    assert pred_high.sum() <= pred_05.sum()


def test_predict_invalid_threshold(tmp_path):
    df, X, y = _make_binary_df(tmp_path)
    model = fit_gbdt_classifier(df, ["x0", "x1"], "y", n_estimators=5)
    with pytest.raises(ValueError, match="threshold"):
        model.predict(X, threshold=1.5)
    with pytest.raises(ValueError, match="threshold"):
        model.predict(X, threshold=-0.1)


def test_predict_df_adds_columns(tmp_path):
    df, X, y = _make_binary_df(tmp_path)
    model = fit_gbdt_classifier(df, ["x0", "x1"], "y", n_estimators=5)
    result = model.predict(df)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    names = pq.read_schema(out).names
    assert "y_prob" in names
    assert "y_pred" in names


def test_predict_proba_df_adds_only_prob(tmp_path):
    df, X, y = _make_binary_df(tmp_path)
    model = fit_gbdt_classifier(df, ["x0", "x1"], "y", n_estimators=5)
    result = model.predict_proba(df)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    names = pq.read_schema(out).names
    assert "y_prob" in names
    assert "y_pred" not in names


def test_predict_null_rows_yield_null(tmp_path):
    df, X, y = _make_binary_df(tmp_path)
    # Introduce null
    df_np = df.head(df.count())  # materialize
    # Easier: rebuild df with one null
    data = {f"x{i}": X[:, i].tolist() for i in range(X.shape[1])}
    data["y"] = y.tolist()
    data["x0"][5] = None
    df2 = make_df(tmp_path, data, name="with_null")
    model = fit_gbdt_classifier(df2, ["x0", "x1"], "y", n_estimators=5)
    result = model.predict(df2)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert rows[5]["y_prob"] is None
    assert rows[5]["y_pred"] is None


# ---------- validation ----------

def test_fit_gbdt_classifier_non_binary_y_raises(tmp_path):
    rng = np.random.default_rng(0)
    n = 100
    df = make_df(tmp_path, {
        "x0": rng.normal(size=n).tolist(),
        "y": [2 if i % 2 else 3 for i in range(n)],
    })
    with pytest.raises(ValueError, match="binary"):
        fit_gbdt_classifier(df, ["x0"], "y", n_estimators=5)


def test_fit_gbdt_classifier_single_class_raises(tmp_path):
    rng = np.random.default_rng(0)
    df = make_df(tmp_path, {
        "x0": rng.normal(size=50).tolist(),
        "y": [1] * 50,
    })
    with pytest.raises(ValueError, match="both classes"):
        fit_gbdt_classifier(df, ["x0"], "y", n_estimators=5)


def test_fit_gbdt_classifier_target_in_X(tmp_path):
    df, X, y = _make_binary_df(tmp_path)
    with pytest.raises(ValueError, match="must not be in X_cols"):
        fit_gbdt_classifier(df, ["x0", "y"], "y", n_estimators=5)


def test_fit_gbdt_classifier_missing_column(tmp_path):
    df, X, y = _make_binary_df(tmp_path)
    with pytest.raises(KeyError, match="nope"):
        fit_gbdt_classifier(df, ["nope"], "y")


def test_fit_gbdt_classifier_non_numeric(tmp_path):
    df = make_df(tmp_path, {
        "x0": ["a", "b", "c", "d"],
        "y": [0, 1, 0, 1],
    })
    with pytest.raises(TypeError, match="numeric"):
        fit_gbdt_classifier(df, ["x0"], "y")


def test_fit_gbdt_classifier_memory_guard(tmp_path):
    rng = np.random.default_rng(0)
    n = 10_000
    X = rng.normal(size=n)
    y = (rng.random(n) < 0.5).astype(np.int64)
    # Ensure both classes
    y[0] = 0; y[1] = 1
    df = make_df(tmp_path, {"x0": X.tolist(), "y": y.tolist()})
    df._memory_guard = MemoryGuard(max_memory_bytes=100_000)
    with pytest.raises(MemoryLimitExceeded, match="internal state"):
        fit_gbdt_classifier(df, ["x0"], "y", n_estimators=5)


# ---------- convenience / to_dict ----------

def test_fit_gbdt_classifier_dataframe_method(tmp_path):
    df, X, y = _make_binary_df(tmp_path)
    model = df.fit_gbdt_classifier(["x0", "x1"], "y", n_estimators=5)
    assert isinstance(model, GBDTClassifier)


def test_classifier_to_dict(tmp_path):
    df, X, y = _make_binary_df(tmp_path)
    model = fit_gbdt_classifier(df, ["x0", "x1"], "y", n_estimators=5)
    d = model.to_dict()
    assert d["loss"] == "logloss_binary"
    assert d["n_trees"] == 5


# ---------- regression still works ----------

def test_regressor_still_works(tmp_path):
    """Regression API unchanged after refactor."""
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    y = 2.0 * x + 0.1 * rng.normal(size=n)
    df = make_df(tmp_path, {"x": x.tolist(), "y": y.tolist()})
    model = fit_gbdt_regressor(
        df, ["x"], "y", n_estimators=10, max_depth=2, learning_rate=0.2,
    )
    pred = model.predict(np.column_stack([x]))
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot
    assert r2 > 0.9
'''


def write_tests():
    print("\n=== tests/test_gbdt_classification.py ===")
    p = Path("tests/test_gbdt_classification.py")
    p.write_text(TESTS_SRC)
    compile_check(p)
    print("  OK: tests written")


# ============ main ============

def main():
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    branch = "patch/v075-gbdt-classification"

    print("=== v0.7.5: GBDT classification (logloss) ===")
    ensure_branch(branch)
    require_clean()

    backup_dir = Path(f".backups/backup-pre-v075-{ts_str}")
    backup_dir.mkdir(parents=True, exist_ok=True)
    sh(["git", "archive", "--format=tar.gz",
        "-o", str(backup_dir / "repo.tgz"), "HEAD"])
    sh(["git", "tag", "-a", f"backup-pre-v075-{ts_str}",
        "-m", "before patch v0.7.5 GBDT classification"])

    print("\n=== cadw/ml/gbdt.py (refactor: l2 + logloss_binary) ===")
    Path("cadw/ml/gbdt.py").write_text(GBDT_SRC)
    compile_check("cadw/ml/gbdt.py")
    print("  OK: gbdt.py refactored (l2 + logloss_binary)")

    patch_dataframe()
    patch_exports()
    write_tests()

    print("\n=== pytest tests/test_gbdt_classification.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_gbdt_classification.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** FAILURE — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.7.5 (test red)"], check=False)
        sys.exit(1)

    print("\n=== pytest tests/test_gbdt_regression.py -v (regression) ===")
    r2 = subprocess.run(
        ["uv", "run", "pytest", "tests/test_gbdt_regression.py", "-q"],
        timeout=300,
    )
    if r2.returncode != 0:
        print("\n*** REGRESSION FAILURE — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.7.5 (regression red)"], check=False)
        sys.exit(1)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.7.5: GBDT binary classification (logloss); gbdt.py refactored "
        "(l2 + logloss_binary via common _fit_boosting)"])

    print("\n==> v0.7.5 applied on", branch)


if __name__ == "__main__":
    main()