#!/usr/bin/env python3
"""v0.7.6: Naive Bayes — Gaussian, Multinomial, Bernoulli.

Design:
    Composition: one Reduce (per-class statistics) + streaming predict.
    Peak RAM = batch + O(C×d) state. Independent of N.

Variants:
    gaussian    : X continuous; per (class, feature) mean/var.
    multinomial : X counts (>=0); per (class, feature) sum → prob via
                  Laplace smoothing (alpha).
    bernoulli   : X in {0,1}; per (class, feature) count of ones →
                  Bernoulli probability via Laplace smoothing (alpha).

y must be numeric (int/float). For categorical target: encode_categorical
first. Documented §11.5.

Prior: empirical (n_c / n_total). No user prior in MVP.

Predict:
    - predict(X)          → class values (float64)
    - predict_proba(X)    → (n, C) matrix of probabilities
    - for CadwDataFrame: adds {target}_pred column(s);
      predict_proba adds {target}_prob_0..{target}_prob_{C-1}
      (index-based; classes_ maps index -> actual class value)

Run: uv run python scripts/patch_v076_naive_bayes.py
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


# ============ cadw/ml/naive_bayes.py ============

NB_SRC = '''"""Naive Bayes (v0.7.6) — Gaussian, Multinomial, Bernoulli.

Composition: one Reduce (per-class stats) + streaming predict.
Peak RAM = batch + O(C×d) state. Independent of N.

y must be numeric (int/float). For categorical target — encode_categorical
first (documented §11.5).

Prior: empirical (n_c / n_total).

Multinomial: X >= 0 (counts). alpha (Laplace smoothing) >= 0.
Bernoulli: X in {0, 1}. alpha (Laplace smoothing) >= 0.
Gaussian: any continuous X. var_smoothing adds fraction of max variance
          to every variance (numerical stability for constant features).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.ops.reduce import ReduceOp, reduce


_VARIANTS = ("gaussian", "multinomial", "bernoulli")


# ============ helpers ============

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


def _softmax(log_joint: np.ndarray) -> np.ndarray:
    m = np.max(log_joint, axis=1, keepdims=True)
    exp_d = np.exp(log_joint - m)
    return exp_d / np.sum(exp_d, axis=1, keepdims=True)


# ============ ReduceOps ============

class _GaussianStatsOp(ReduceOp):
    """Per-class: (n_c, sum_x[d], sum_xx[d]). Dynamic class discovery."""

    def __init__(self, feature_cols: Sequence[str], target_col: str):
        self.feature_cols = list(feature_cols)
        self.target_col = target_col
        self.d = len(feature_cols)

    def init(self):
        return {}

    def update(self, state, batch):
        X = _stack_numeric(batch, self.feature_cols)
        y = _column_to_f64(batch.column(self.target_col))
        valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[valid]
        y = y[valid]
        if X.shape[0] == 0:
            return state
        for cls in np.unique(y):
            m = y == cls
            Xc = X[m]
            key = float(cls)
            if key not in state:
                state[key] = [0, np.zeros(self.d), np.zeros(self.d)]
            state[key][0] += int(Xc.shape[0])
            state[key][1] += Xc.sum(axis=0)
            state[key][2] += (Xc * Xc).sum(axis=0)
        return state

    def merge(self, a, b):
        for k, (n, s, ss) in b.items():
            if k in a:
                a[k][0] += n
                a[k][1] = a[k][1] + s
                a[k][2] = a[k][2] + ss
            else:
                a[k] = [n, s.copy(), ss.copy()]
        return a

    def state_nbytes(self, state):
        return sum(8 + 16 * self.d for _ in state.values())

    def finalize(self, state):
        classes = sorted(state.keys())
        C = len(classes)
        n_c = np.zeros(C, dtype=np.int64)
        mean = np.zeros((C, self.d), dtype=np.float64)
        var = np.zeros((C, self.d), dtype=np.float64)
        for i, k in enumerate(classes):
            n, s, ss = state[k]
            n_c[i] = n
            if n > 0:
                mean[i] = s / n
                var[i] = ss / n - mean[i] ** 2
        return {
            "classes": np.array(classes, dtype=np.float64),
            "n_c": n_c, "mean": mean, "var": var,
        }


class _MultinomialStatsOp(ReduceOp):
    """Per-class: (n_c, sum_x[d]). X assumed >= 0 (counts)."""

    def __init__(self, feature_cols: Sequence[str], target_col: str):
        self.feature_cols = list(feature_cols)
        self.target_col = target_col
        self.d = len(feature_cols)

    def init(self):
        return {}

    def update(self, state, batch):
        X = _stack_numeric(batch, self.feature_cols)
        y = _column_to_f64(batch.column(self.target_col))
        valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[valid]
        y = y[valid]
        if X.shape[0] == 0:
            return state
        for cls in np.unique(y):
            m = y == cls
            Xc = X[m]
            key = float(cls)
            if key not in state:
                state[key] = [0, np.zeros(self.d)]
            state[key][0] += int(Xc.shape[0])
            state[key][1] += Xc.sum(axis=0)
        return state

    def merge(self, a, b):
        for k, (n, s) in b.items():
            if k in a:
                a[k][0] += n
                a[k][1] = a[k][1] + s
            else:
                a[k] = [n, s.copy()]
        return a

    def state_nbytes(self, state):
        return sum(8 + 8 * self.d for _ in state.values())

    def finalize(self, state):
        classes = sorted(state.keys())
        C = len(classes)
        n_c = np.zeros(C, dtype=np.int64)
        feature_sum = np.zeros((C, self.d), dtype=np.float64)
        for i, k in enumerate(classes):
            n, s = state[k]
            n_c[i] = n
            feature_sum[i] = s
        return {
            "classes": np.array(classes, dtype=np.float64),
            "n_c": n_c, "feature_sum": feature_sum,
        }


class _BernoulliStatsOp(ReduceOp):
    """Per-class: (n_c, count_ones[d]). X assumed in {0, 1}."""

    def __init__(self, feature_cols: Sequence[str], target_col: str):
        self.feature_cols = list(feature_cols)
        self.target_col = target_col
        self.d = len(feature_cols)

    def init(self):
        return {}

    def update(self, state, batch):
        X = _stack_numeric(batch, self.feature_cols)
        y = _column_to_f64(batch.column(self.target_col))
        valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[valid]
        y = y[valid]
        if X.shape[0] == 0:
            return state
        Xb = (X > 0.5).astype(np.float64)
        for cls in np.unique(y):
            m = y == cls
            Xc = Xb[m]
            key = float(cls)
            if key not in state:
                state[key] = [0, np.zeros(self.d)]
            state[key][0] += int(Xc.shape[0])
            state[key][1] += Xc.sum(axis=0)
        return state

    def merge(self, a, b):
        for k, (n, c) in b.items():
            if k in a:
                a[k][0] += n
                a[k][1] = a[k][1] + c
            else:
                a[k] = [n, c.copy()]
        return a

    def state_nbytes(self, state):
        return sum(8 + 8 * self.d for _ in state.values())

    def finalize(self, state):
        classes = sorted(state.keys())
        C = len(classes)
        n_c = np.zeros(C, dtype=np.int64)
        count_ones = np.zeros((C, self.d), dtype=np.float64)
        for i, k in enumerate(classes):
            n, c = state[k]
            n_c[i] = n
            count_ones[i] = c
        return {
            "classes": np.array(classes, dtype=np.float64),
            "n_c": n_c, "count_ones": count_ones,
        }


# ============ models ============

@dataclass
class GaussianNB:
    """Gaussian Naive Bayes."""

    classes_: np.ndarray
    class_prior_: np.ndarray
    theta_: np.ndarray
    var_: np.ndarray
    feature_names_: list
    target_col_: str
    variant_: str = "gaussian"
    fitted_: bool = True

    def _log_joint_np(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        C = len(self.classes_)
        log_prior = np.log(self.class_prior_ + 1e-300)
        log_joint = np.empty((n, C), dtype=np.float64)
        for c in range(C):
            theta = self.theta_[c]
            var = self.var_[c]
            diff = X - theta
            log_joint[:, c] = (
                log_prior[c]
                - 0.5 * np.sum(np.log(2.0 * np.pi * var))
                - 0.5 * np.sum(diff * diff / var, axis=1)
            )
        return log_joint

    def predict_proba(self, X):
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            return _softmax(self._log_joint_np(X))
        return self._predict_df(X, proba=True)

    def predict(self, X):
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            return self.classes_[self._log_joint_np(X).argmax(axis=1)]
        return self._predict_df(X, proba=False)

    def _predict_df(self, X, *, proba: bool):
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

        C = len(self.classes_)
        target = self.target_col_
        pred_col = f"{target}_pred"
        prob_cols = [f"{target}_prob_{i}" for i in range(C)]
        orig_schema = X._dataset.schema
        out_fields = [orig_schema.field(c) for c in X._columns]
        out_fields.append(pa.field(pred_col, pa.float64(), nullable=True))
        if proba:
            for pc in prob_cols:
                out_fields.append(pa.field(pc, pa.float64(), nullable=True))
        out_schema = pa.schema(out_fields)

        model = self

        def _iter_pred():
            for b in X._iter_filtered_batches():
                Xm = _stack_numeric(b, model.feature_names_)
                n = Xm.shape[0]
                nan_rows = np.isnan(Xm).any(axis=1)
                log_joint = np.full((n, C), -np.inf, dtype=np.float64)
                if (~nan_rows).any():
                    log_joint[~nan_rows] = model._log_joint_np(Xm[~nan_rows])
                # Pred: argmax among finite rows; NaN rows → null
                idx = log_joint.argmax(axis=1).astype(np.int64)
                pred_vals = model.classes_[idx]
                arrays = [b.column(c) for c in X._columns]
                arrays.append(pa.array(
                    pred_vals, type=pa.float64(),
                    mask=nan_rows if nan_rows.any() else None,
                ))
                if proba:
                    probs = np.zeros((n, C), dtype=np.float64)
                    if (~nan_rows).any():
                        probs[~nan_rows] = _softmax(log_joint[~nan_rows])
                    for i in range(C):
                        arrays.append(pa.array(
                            probs[:, i], type=pa.float64(),
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
            "variant": "gaussian",
            "classes": self.classes_.tolist(),
            "class_prior": self.class_prior_.tolist(),
            "theta": self.theta_.tolist(),
            "var": self.var_.tolist(),
            "feature_names": self.feature_names_,
            "target_col": self.target_col_,
        }


@dataclass
class MultinomialNB:
    """Multinomial Naive Bayes (for counts; e.g., bag-of-words)."""

    classes_: np.ndarray
    class_prior_: np.ndarray
    feature_log_prob_: np.ndarray  # (C, d) log P(x_j | c)
    feature_count_: np.ndarray     # (C, d) sum x_j | c
    feature_names_: list
    target_col_: str
    alpha_: float
    variant_: str = "multinomial"
    fitted_: bool = True

    def _log_joint_np(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        C = len(self.classes_)
        log_prior = np.log(self.class_prior_ + 1e-300)
        log_joint = np.empty((n, C), dtype=np.float64)
        for c in range(C):
            log_joint[:, c] = log_prior[c] + X @ self.feature_log_prob_[c]
        return log_joint

    def predict_proba(self, X):
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            return _softmax(self._log_joint_np(X))
        return _MultinomialPredictMixin._predict_df(self, X, proba=True)

    def predict(self, X):
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            return self.classes_[self._log_joint_np(X).argmax(axis=1)]
        return _MultinomialPredictMixin._predict_df(self, X, proba=False)

    def to_dict(self) -> dict:
        return {
            "variant": "multinomial",
            "classes": self.classes_.tolist(),
            "class_prior": self.class_prior_.tolist(),
            "feature_log_prob": self.feature_log_prob_.tolist(),
            "feature_count": self.feature_count_.tolist(),
            "feature_names": self.feature_names_,
            "target_col": self.target_col_,
            "alpha": self.alpha_,
        }


@dataclass
class BernoulliNB:
    """Bernoulli Naive Bayes (for binary features)."""

    classes_: np.ndarray
    class_prior_: np.ndarray
    feature_log_prob_: np.ndarray
    feature_log_neg_prob_: np.ndarray
    feature_count_: np.ndarray
    feature_names_: list
    target_col_: str
    alpha_: float
    variant_: str = "bernoulli"
    fitted_: bool = True

    def _log_joint_np(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        C = len(self.classes_)
        log_prior = np.log(self.class_prior_ + 1e-300)
        Xb = (X > 0.5).astype(np.float64)
        log_joint = np.empty((n, C), dtype=np.float64)
        for c in range(C):
            log_joint[:, c] = (
                log_prior[c]
                + Xb @ self.feature_log_prob_[c]
                + (1.0 - Xb) @ self.feature_log_neg_prob_[c]
            )
        return log_joint

    def predict_proba(self, X):
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            return _softmax(self._log_joint_np(X))
        return _MultinomialPredictMixin._predict_df(self, X, proba=True)

    def predict(self, X):
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            return self.classes_[self._log_joint_np(X).argmax(axis=1)]
        return _MultinomialPredictMixin._predict_df(self, X, proba=False)

    def to_dict(self) -> dict:
        return {
            "variant": "bernoulli",
            "classes": self.classes_.tolist(),
            "class_prior": self.class_prior_.tolist(),
            "feature_log_prob": self.feature_log_prob_.tolist(),
            "feature_log_neg_prob": self.feature_log_neg_prob_.tolist(),
            "feature_names": self.feature_names_,
            "target_col": self.target_col_,
            "alpha": self.alpha_,
        }


class _MultinomialPredictMixin:
    """Shared predict_df for Multinomial/Bernoulli."""

    @staticmethod
    def _predict_df(model, X, *, proba: bool):
        from cadw.dataframe import CadwDataFrame
        from cadw.dataset import Dataset
        if not isinstance(X, CadwDataFrame):
            raise TypeError(
                f"predict: X must be np.ndarray or CadwDataFrame; "
                f"got {type(X).__name__}"
            )
        for c in model.feature_names_:
            if c not in X.columns:
                raise KeyError(f"predict: column {c!r} not in df")

        C = len(model.classes_)
        target = model.target_col_
        pred_col = f"{target}_pred"
        prob_cols = [f"{target}_prob_{i}" for i in range(C)]
        orig_schema = X._dataset.schema
        out_fields = [orig_schema.field(c) for c in X._columns]
        out_fields.append(pa.field(pred_col, pa.float64(), nullable=True))
        if proba:
            for pc in prob_cols:
                out_fields.append(pa.field(pc, pa.float64(), nullable=True))
        out_schema = pa.schema(out_fields)

        def _iter_pred():
            for b in X._iter_filtered_batches():
                Xm = _stack_numeric(b, model.feature_names_)
                n = Xm.shape[0]
                nan_rows = np.isnan(Xm).any(axis=1)
                log_joint = np.full((n, C), -np.inf, dtype=np.float64)
                if (~nan_rows).any():
                    log_joint[~nan_rows] = model._log_joint_np(Xm[~nan_rows])
                idx = log_joint.argmax(axis=1).astype(np.int64)
                pred_vals = model.classes_[idx]
                arrays = [b.column(c) for c in X._columns]
                arrays.append(pa.array(
                    pred_vals, type=pa.float64(),
                    mask=nan_rows if nan_rows.any() else None,
                ))
                if proba:
                    probs = np.zeros((n, C), dtype=np.float64)
                    if (~nan_rows).any():
                        probs[~nan_rows] = _softmax(log_joint[~nan_rows])
                    for i in range(C):
                        arrays.append(pa.array(
                            probs[:, i], type=pa.float64(),
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


# ============ fit entry points ============

def _validate_common(df, X_cols, y_col):
    if isinstance(X_cols, str):
        X_cols = [X_cols]
    else:
        X_cols = list(X_cols)
    if not X_cols:
        raise ValueError("fit_naive_bayes: X_cols must be non-empty")
    if y_col in X_cols:
        raise ValueError(
            f"fit_naive_bayes: target {y_col!r} must not be in X_cols"
        )
    for c in X_cols + [y_col]:
        if c not in df.columns:
            raise KeyError(f"fit_naive_bayes: column {c!r} not in df")
    schema = df._dataset.schema
    for c in X_cols + [y_col]:
        t = schema.field(c).type
        if not (pa.types.is_integer(t) or pa.types.is_floating(t)):
            raise TypeError(
                f"fit_naive_bayes: column {c!r} has type {t}; "
                f"numeric required (use encode_categorical for strings)"
            )
    return X_cols


def _class_prior(n_c: np.ndarray) -> np.ndarray:
    total = int(n_c.sum())
    if total == 0:
        raise ValueError("fit_naive_bayes: no valid rows")
    return n_c.astype(np.float64) / total


def fit_gaussian_nb(
    df, X_cols, y_col: str, *,
    var_smoothing: float = 1e-9,
    memory_guard: Optional[MemoryGuard] = None,
) -> GaussianNB:
    """Gaussian Naive Bayes (continuous X)."""
    X_cols = _validate_common(df, X_cols, y_col)
    if var_smoothing < 0:
        raise ValueError(
            f"var_smoothing must be >= 0; got {var_smoothing}"
        )
    mg = memory_guard or df._memory_guard
    op = _GaussianStatsOp(X_cols, y_col)
    res = reduce(df._iter_filtered_batches(), op, mg, log=False)
    classes = res["classes"]
    n_c = res["n_c"]
    mean = res["mean"]
    var = res["var"]
    if len(classes) == 0 or n_c.sum() == 0:
        raise ValueError("fit_gaussian_nb: no valid rows")
    # Numerical stability: add fraction of max variance
    max_var = float(var.max()) if var.size > 0 else 0.0
    epsilon = var_smoothing * (max_var if max_var > 0 else 1.0)
    var = var + epsilon
    return GaussianNB(
        classes_=classes,
        class_prior_=_class_prior(n_c),
        theta_=mean,
        var_=var,
        feature_names_=list(X_cols),
        target_col_=y_col,
    )


def fit_multinomial_nb(
    df, X_cols, y_col: str, *,
    alpha: float = 1.0,
    memory_guard: Optional[MemoryGuard] = None,
) -> MultinomialNB:
    """Multinomial Naive Bayes (X >= 0 counts)."""
    X_cols = _validate_common(df, X_cols, y_col)
    if alpha < 0:
        raise ValueError(f"alpha must be >= 0; got {alpha}")
    mg = memory_guard or df._memory_guard
    op = _MultinomialStatsOp(X_cols, y_col)
    res = reduce(df._iter_filtered_batches(), op, mg, log=False)
    classes = res["classes"]
    n_c = res["n_c"]
    feature_sum = res["feature_sum"]
    if len(classes) == 0 or n_c.sum() == 0:
        raise ValueError("fit_multinomial_nb: no valid rows")
    d = len(X_cols)
    # feature_log_prob[c, j] = log((sum_c_j + alpha) / (sum_j sum_c_j + alpha*d))
    smoothed = feature_sum + alpha
    denom = smoothed.sum(axis=1, keepdims=True)
    feature_log_prob = np.log(smoothed / denom)
    return MultinomialNB(
        classes_=classes,
        class_prior_=_class_prior(n_c),
        feature_log_prob_=feature_log_prob,
        feature_count_=feature_sum,
        feature_names_=list(X_cols),
        target_col_=y_col,
        alpha_=float(alpha),
    )


def fit_bernoulli_nb(
    df, X_cols, y_col: str, *,
    alpha: float = 1.0,
    memory_guard: Optional[MemoryGuard] = None,
) -> BernoulliNB:
    """Bernoulli Naive Bayes (X in {0,1}; X > 0.5 → 1)."""
    X_cols = _validate_common(df, X_cols, y_col)
    if alpha < 0:
        raise ValueError(f"alpha must be >= 0; got {alpha}")
    mg = memory_guard or df._memory_guard
    op = _BernoulliStatsOp(X_cols, y_col)
    res = reduce(df._iter_filtered_batches(), op, mg, log=False)
    classes = res["classes"]
    n_c = res["n_c"]
    count_ones = res["count_ones"]
    if len(classes) == 0 or n_c.sum() == 0:
        raise ValueError("fit_bernoulli_nb: no valid rows")
    # P(x_j=1|c) = (count_ones + alpha) / (n_c + 2*alpha)
    denom = n_c[:, None] + 2.0 * alpha
    p = (count_ones + alpha) / denom
    feature_log_prob = np.log(p)
    feature_log_neg_prob = np.log(1.0 - p)
    return BernoulliNB(
        classes_=classes,
        class_prior_=_class_prior(n_c),
        feature_log_prob_=feature_log_prob,
        feature_log_neg_prob_=feature_log_neg_prob,
        feature_count_=count_ones,
        feature_names_=list(X_cols),
        target_col_=y_col,
        alpha_=float(alpha),
    )
'''


# ============ patch dataframe ============

def patch_dataframe():
    print("\n=== cadw/dataframe.py: fit_naive_bayes + variants ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "def fit_naive_bayes" in s:
        print("  already present — skip")
        return

    anchor = "    # ---------------- merge ----------------"
    if anchor not in s:
        raise SystemExit("merge anchor not found")

    methods = '''    # ---------------- ML: fit_naive_bayes ----------------

    def fit_naive_bayes(
        self,
        X_cols: "str | Sequence[str]",
        y_col: str,
        *,
        variant: str = "gaussian",
        alpha: float = 1.0,
        var_smoothing: float = 1e-9,
    ):
        """Fit Naive Bayes (v0.7.6).

        - variant='gaussian'    : continuous X; params: var_smoothing.
        - variant='multinomial' : X >= 0 counts; params: alpha.
        - variant='bernoulli'   : X in {0,1}; params: alpha.

        y must be numeric (int/float). For categorical target — encode.
        Returns GaussianNB | MultinomialNB | BernoulliNB.
        """
        from cadw.ml.naive_bayes import (
            _VARIANTS,
            fit_gaussian_nb, fit_multinomial_nb, fit_bernoulli_nb,
        )
        if variant not in _VARIANTS:
            raise ValueError(
                f"fit_naive_bayes: variant must be one of {_VARIANTS}; "
                f"got {variant!r}"
            )
        if variant == "gaussian":
            return fit_gaussian_nb(
                self, X_cols, y_col, var_smoothing=var_smoothing,
                memory_guard=self._memory_guard,
            )
        if variant == "multinomial":
            return fit_multinomial_nb(
                self, X_cols, y_col, alpha=alpha,
                memory_guard=self._memory_guard,
            )
        return fit_bernoulli_nb(
            self, X_cols, y_col, alpha=alpha,
            memory_guard=self._memory_guard,
        )

    def fit_gaussian_nb(
        self, X_cols: "str | Sequence[str]", y_col: str,
        *, var_smoothing: float = 1e-9,
    ):
        from cadw.ml.naive_bayes import fit_gaussian_nb
        return fit_gaussian_nb(
            self, X_cols, y_col, var_smoothing=var_smoothing,
            memory_guard=self._memory_guard,
        )

    def fit_multinomial_nb(
        self, X_cols: "str | Sequence[str]", y_col: str,
        *, alpha: float = 1.0,
    ):
        from cadw.ml.naive_bayes import fit_multinomial_nb
        return fit_multinomial_nb(
            self, X_cols, y_col, alpha=alpha,
            memory_guard=self._memory_guard,
        )

    def fit_bernoulli_nb(
        self, X_cols: "str | Sequence[str]", y_col: str,
        *, alpha: float = 1.0,
    ):
        from cadw.ml.naive_bayes import fit_bernoulli_nb
        return fit_bernoulli_nb(
            self, X_cols, y_col, alpha=alpha,
            memory_guard=self._memory_guard,
        )

'''
    s = s.replace(anchor, methods + anchor, 1)
    p.write_text(s)
    compile_check(p)
    print("  OK: fit_naive_bayes (+3 variants) added")


# ============ exports ============

def patch_exports():
    print("\n=== cadw/ml/__init__.py + cadw/__init__.py ===")
    p = Path("cadw/ml/__init__.py")
    p.write_text('''"""CADW ML layer (v0.7.6)."""
from .linear import XTXOp, LinearModel, fit_linear
from .logistic import LogisticGradOp, LogisticModel, fit_logistic
from .kmeans import KMeansStepOp, KMeansModel, fit_kmeans
from .pca import PCA, fit_pca
from .gbdt import (
    GBDTRegressor, fit_gbdt_regressor,
    GBDTClassifier, fit_gbdt_classifier,
)
from .naive_bayes import (
    GaussianNB, fit_gaussian_nb,
    MultinomialNB, fit_multinomial_nb,
    BernoulliNB, fit_bernoulli_nb,
)

__all__ = [
    "XTXOp", "LinearModel", "fit_linear",
    "LogisticGradOp", "LogisticModel", "fit_logistic",
    "KMeansStepOp", "KMeansModel", "fit_kmeans",
    "PCA", "fit_pca",
    "GBDTRegressor", "fit_gbdt_regressor",
    "GBDTClassifier", "fit_gbdt_classifier",
    "GaussianNB", "fit_gaussian_nb",
    "MultinomialNB", "fit_multinomial_nb",
    "BernoulliNB", "fit_bernoulli_nb",
]
''')
    compile_check(p)

    p2 = Path("cadw/__init__.py")
    s = p2.read_text()
    if "GaussianNB" in s:
        print("  top-level already updated")
        return
    s = s.replace(
        "    GBDTClassifier, fit_gbdt_classifier,\n)",
        "    GBDTClassifier, fit_gbdt_classifier,\n"
        "    GaussianNB, fit_gaussian_nb,\n"
        "    MultinomialNB, fit_multinomial_nb,\n"
        "    BernoulliNB, fit_bernoulli_nb,\n)",
        1,
    )
    s = s.replace(
        '    "GBDTClassifier", "fit_gbdt_classifier",\n',
        '    "GBDTClassifier", "fit_gbdt_classifier",\n'
        '    "GaussianNB", "fit_gaussian_nb",\n'
        '    "MultinomialNB", "fit_multinomial_nb",\n'
        '    "BernoulliNB", "fit_bernoulli_nb",\n',
        1,
    )
    p2.write_text(s)
    compile_check(p2)
    print("  OK: exports updated")


# ============ tests ============

TESTS_SRC = '''"""v0.7.6: Naive Bayes — Gaussian, Multinomial, Bernoulli."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.ml.naive_bayes import (
    GaussianNB, MultinomialNB, BernoulliNB,
    fit_gaussian_nb, fit_multinomial_nb, fit_bernoulli_nb,
)


def make_df(tmp_path, data, chunk=50, name="L"):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


# ============ Gaussian NB ============

def _gauss_data(n_per_class=200, seed=42):
    rng = np.random.default_rng(seed)
    # Two classes with distinct means
    X0 = rng.normal(loc=[-2.0, -2.0], scale=1.0, size=(n_per_class, 2))
    X1 = rng.normal(loc=[2.0, 2.0], scale=1.0, size=(n_per_class, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * n_per_class + [1] * n_per_class, dtype=np.float64)
    return {"x0": X[:, 0].tolist(), "x1": X[:, 1].tolist(), "y": y.tolist()}, X, y


def test_gaussian_nb_basic(tmp_path):
    data, X, y = _gauss_data()
    df = make_df(tmp_path, data)
    model = fit_gaussian_nb(df, ["x0", "x1"], "y")
    assert isinstance(model, GaussianNB)
    assert model.classes_.tolist() == [0.0, 1.0]
    assert model.theta_.shape == (2, 2)


def test_gaussian_nb_predict_accuracy(tmp_path):
    data, X, y = _gauss_data(n_per_class=300)
    df = make_df(tmp_path, data)
    model = fit_gaussian_nb(df, ["x0", "x1"], "y")
    pred = model.predict(X)
    acc = (pred == y).mean()
    assert acc > 0.9, f"accuracy={acc:.3f}"


def test_gaussian_nb_predict_proba_range(tmp_path):
    data, X, y = _gauss_data()
    df = make_df(tmp_path, data)
    model = fit_gaussian_nb(df, ["x0", "x1"], "y")
    proba = model.predict_proba(X[:10])
    assert proba.shape == (10, 2)
    assert (proba >= 0).all() and (proba <= 1).all()
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_gaussian_nb_multiclass(tmp_path):
    rng = np.random.default_rng(0)
    n = 100
    data = {"x0": [], "x1": [], "y": []}
    for cls, center in enumerate([[-3.0, 0.0], [0.0, 0.0], [3.0, 0.0]]):
        X = rng.normal(loc=center, scale=0.5, size=(n, 2))
        data["x0"].extend(X[:, 0].tolist())
        data["x1"].extend(X[:, 1].tolist())
        data["y"].extend([float(cls)] * n)
    df = make_df(tmp_path, data)
    model = fit_gaussian_nb(df, ["x0", "x1"], "y")
    assert len(model.classes_) == 3
    assert model.theta_.shape == (3, 2)


def test_gaussian_nb_predict_df(tmp_path):
    data, X, y = _gauss_data()
    df = make_df(tmp_path, data)
    model = fit_gaussian_nb(df, ["x0", "x1"], "y")
    result = model.predict(df)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    names = pq.read_schema(out).names
    assert "y_pred" in names


def test_gaussian_nb_predict_proba_df_adds_columns(tmp_path):
    data, X, y = _gauss_data()
    df = make_df(tmp_path, data)
    model = fit_gaussian_nb(df, ["x0", "x1"], "y")
    result = model.predict_proba(df)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    names = pq.read_schema(out).names
    assert "y_prob_0" in names
    assert "y_prob_1" in names


def test_gaussian_nb_with_nulls(tmp_path):
    data, X, y = _gauss_data()
    data["x0"][5] = None
    df = make_df(tmp_path, data)
    model = fit_gaussian_nb(df, ["x0", "x1"], "y")
    result = model.predict(df)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert rows[5]["y_pred"] is None


def test_gaussian_nb_var_smoothing(tmp_path):
    """Constant feature (var=0) must not divide by zero."""
    data = {"x0": [1.0, 1.0, 2.0, 2.0], "x1": [3.0, 4.0, 3.0, 4.0], "y": [0.0, 0.0, 1.0, 1.0]}
    df = make_df(tmp_path, data)
    model = fit_gaussian_nb(df, ["x0", "x1"], "y", var_smoothing=1e-6)
    # Predict with constant x0
    pred = model.predict(np.array([[1.0, 3.0], [2.0, 3.0]]))
    assert len(pred) == 2
    assert not np.any(np.isnan(pred))


def test_gaussian_nb_validation_empty_X(tmp_path):
    df = make_df(tmp_path, {"x0": [1.0], "y": [0.0]})
    with pytest.raises(ValueError, match="X_cols"):
        fit_gaussian_nb(df, [], "y")


def test_gaussian_nb_validation_target_in_X(tmp_path):
    df = make_df(tmp_path, {"x0": [1.0], "y": [0.0]})
    with pytest.raises(ValueError, match="must not be in X_cols"):
        fit_gaussian_nb(df, ["x0", "y"], "y")


def test_gaussian_nb_validation_non_numeric(tmp_path):
    df = make_df(tmp_path, {"x0": ["a", "b"], "y": [0.0, 1.0]})
    with pytest.raises(TypeError, match="numeric"):
        fit_gaussian_nb(df, ["x0"], "y")


def test_gaussian_nb_dataframe_method(tmp_path):
    data, X, y = _gauss_data()
    df = make_df(tmp_path, data)
    model = df.fit_gaussian_nb(["x0", "x1"], "y")
    assert isinstance(model, GaussianNB)


def test_gaussian_nb_to_dict(tmp_path):
    data, X, y = _gauss_data()
    df = make_df(tmp_path, data)
    model = fit_gaussian_nb(df, ["x0", "x1"], "y")
    d = model.to_dict()
    assert d["variant"] == "gaussian"
    assert "classes" in d and "theta" in d and "var" in d


# ============ Multinomial NB ============

def _counts_data(n_per_class=100, seed=42):
    rng = np.random.default_rng(seed)
    # Class 0: high x0, low x1; Class 1: low x0, high x1
    X0 = rng.poisson(lam=[10, 1], size=(n_per_class, 2)).astype(np.float64)
    X1 = rng.poisson(lam=[1, 10], size=(n_per_class, 2)).astype(np.float64)
    X = np.vstack([X0, X1])
    y = np.array([0.0] * n_per_class + [1.0] * n_per_class)
    return {"x0": X[:, 0].tolist(), "x1": X[:, 1].tolist(), "y": y.tolist()}, X, y


def test_multinomial_nb_basic(tmp_path):
    data, X, y = _counts_data()
    df = make_df(tmp_path, data)
    model = fit_multinomial_nb(df, ["x0", "x1"], "y")
    assert isinstance(model, MultinomialNB)
    assert model.feature_log_prob_.shape == (2, 2)


def test_multinomial_nb_accuracy(tmp_path):
    data, X, y = _counts_data(n_per_class=150)
    df = make_df(tmp_path, data)
    model = fit_multinomial_nb(df, ["x0", "x1"], "y")
    pred = model.predict(X)
    acc = (pred == y).mean()
    assert acc > 0.9, f"accuracy={acc:.3f}"


def test_multinomial_nb_predict_proba(tmp_path):
    data, X, y = _counts_data()
    df = make_df(tmp_path, data)
    model = fit_multinomial_nb(df, ["x0", "x1"], "y")
    proba = model.predict_proba(X[:5])
    assert proba.shape == (5, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_multinomial_nb_alpha(tmp_path):
    """alpha=0 vs alpha=1: both work (alpha=0 may warn on zero counts)."""
    data, X, y = _counts_data()
    df = make_df(tmp_path, data)
    model = fit_multinomial_nb(df, ["x0", "x1"], "y", alpha=1.0)
    assert model.alpha_ == 1.0


def test_multinomial_nb_negative_alpha_raises(tmp_path):
    df = make_df(tmp_path, {"x0": [1.0], "x1": [0.0], "y": [0.0]})
    with pytest.raises(ValueError, match="alpha"):
        fit_multinomial_nb(df, ["x0", "x1"], "y", alpha=-0.1)


def test_multinomial_nb_predict_df(tmp_path):
    data, X, y = _counts_data()
    df = make_df(tmp_path, data)
    model = fit_multinomial_nb(df, ["x0", "x1"], "y")
    result = model.predict(df)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    names = pq.read_schema(out).names
    assert "y_pred" in names


# ============ Bernoulli NB ============

def _binary_data(n_per_class=150, seed=42):
    rng = np.random.default_rng(seed)
    # Class 0: features mostly 0; Class 1: features mostly 1
    X0 = (rng.random((n_per_class, 3)) < 0.2).astype(np.float64)
    X1 = (rng.random((n_per_class, 3)) < 0.8).astype(np.float64)
    X = np.vstack([X0, X1])
    y = np.array([0.0] * n_per_class + [1.0] * n_per_class)
    return {
        "x0": X[:, 0].tolist(), "x1": X[:, 1].tolist(),
        "x2": X[:, 2].tolist(), "y": y.tolist(),
    }, X, y


def test_bernoulli_nb_basic(tmp_path):
    data, X, y = _binary_data()
    df = make_df(tmp_path, data)
    model = fit_bernoulli_nb(df, ["x0", "x1", "x2"], "y")
    assert isinstance(model, BernoulliNB)
    assert model.feature_log_prob_.shape == (2, 3)


def test_bernoulli_nb_accuracy(tmp_path):
    data, X, y = _binary_data(n_per_class=300)
    df = make_df(tmp_path, data)
    model = fit_bernoulli_nb(df, ["x0", "x1", "x2"], "y")
    pred = model.predict(X)
    acc = (pred == y).mean()
    assert acc > 0.9, f"accuracy={acc:.3f}"


def test_bernoulli_nb_predict_proba(tmp_path):
    data, X, y = _binary_data()
    df = make_df(tmp_path, data)
    model = fit_bernoulli_nb(df, ["x0", "x1", "x2"], "y")
    proba = model.predict_proba(X[:5])
    assert proba.shape == (5, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_bernoulli_nb_predict_df(tmp_path):
    data, X, y = _binary_data()
    df = make_df(tmp_path, data)
    model = fit_bernoulli_nb(df, ["x0", "x1", "x2"], "y")
    result = model.predict_proba(df)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    names = pq.read_schema(out).names
    assert "y_prob_0" in names
    assert "y_prob_1" in names


# ============ unified entry point ============

def test_fit_naive_bayes_variant_gaussian(tmp_path):
    data, X, y = _gauss_data()
    df = make_df(tmp_path, data)
    model = df.fit_naive_bayes(["x0", "x1"], "y", variant="gaussian")
    assert isinstance(model, GaussianNB)


def test_fit_naive_bayes_variant_multinomial(tmp_path):
    data, X, y = _counts_data()
    df = make_df(tmp_path, data)
    model = df.fit_naive_bayes(["x0", "x1"], "y", variant="multinomial")
    assert isinstance(model, MultinomialNB)


def test_fit_naive_bayes_variant_bernoulli(tmp_path):
    data, X, y = _binary_data()
    df = make_df(tmp_path, data)
    model = df.fit_naive_bayes(["x0", "x1", "x2"], "y", variant="bernoulli")
    assert isinstance(model, BernoulliNB)


def test_fit_naive_bayes_invalid_variant(tmp_path):
    data, X, y = _gauss_data()
    df = make_df(tmp_path, data)
    with pytest.raises(ValueError, match="variant"):
        df.fit_naive_bayes(["x0", "x1"], "y", variant="unknown")
'''


def write_tests():
    print("\n=== tests/test_naive_bayes.py ===")
    p = Path("tests/test_naive_bayes.py")
    p.write_text(TESTS_SRC)
    compile_check(p)
    print("  OK: tests written")


# ============ main ============

def main():
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    branch = "patch/v076-naive-bayes"

    print("=== v0.7.6: Naive Bayes ===")
    ensure_branch(branch)
    require_clean()

    backup_dir = Path(f".backups/backup-pre-v076-{ts_str}")
    backup_dir.mkdir(parents=True, exist_ok=True)
    sh(["git", "archive", "--format=tar.gz",
        "-o", str(backup_dir / "repo.tgz"), "HEAD"])
    sh(["git", "tag", "-a", f"backup-pre-v076-{ts_str}",
        "-m", "before patch v0.7.6 Naive Bayes"])

    print("\n=== cadw/ml/naive_bayes.py ===")
    Path("cadw/ml/naive_bayes.py").write_text(NB_SRC)
    compile_check("cadw/ml/naive_bayes.py")
    print("  OK: naive_bayes.py created")

    patch_dataframe()
    patch_exports()
    write_tests()

    print("\n=== pytest tests/test_naive_bayes.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_naive_bayes.py", "-v"]
    )
    if r.returncode != 0:
        print("\n*** FAILURE — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.7.6 (test red)"], check=False)
        sys.exit(1)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.7.6: Naive Bayes (Gaussian, Multinomial, Bernoulli); ML core complete"])

    print("\n==> v0.7.6 applied on", branch)


if __name__ == "__main__":
    main()