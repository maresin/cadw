"""Naive Bayes (v0.7.6) — Gaussian, Multinomial, Bernoulli.

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

    def batch_memory_factor(self) -> float:
        # 2*batch (X_parts + X_masked) + 1*batch allocator;
        # state += in-place, no double
        return 3.0


    def __init__(self, feature_cols: Sequence[str], target_col: str):
        self.feature_cols = list(feature_cols)
        self.target_col = target_col
        self.d = len(feature_cols)

    def estimate_state_nbytes(self) -> int:
        # v0.7.6-fix: pre-init bound for reduce() check_state.
        # Real state is O(C×d), C discovered dynamically from y.
        # v0.9.22 (§13.4/242, AUDIT_ML §1.3): C_ASSUMED=64 (was 32).
        # Audit: K=100 gives 3.125× drift, K=500 gives 15.6×; but
        # measured RSS is tiny (2 MB @ K=100) — not a RAM bug,
        # per-update check_state enforces actual bound. Bumping to 64
        # doubles the assumption (covers K≤64) without breaking
        # K>32 users (audit's ValueError K>32 too aggressive for 0
        # RAM win).
        # Per class: 8 (n) + 8d (sum_x) + 8d (sum_xx) = 8 + 16d.
        return 64 * (8 + 16 * self.d)

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

    def batch_memory_factor(self) -> float:
        # same pattern as Gaussian; state += in-place, no double
        return 3.0


    def __init__(self, feature_cols: Sequence[str], target_col: str):
        self.feature_cols = list(feature_cols)
        self.target_col = target_col
        self.d = len(feature_cols)

    def estimate_state_nbytes(self) -> int:
        # Per class: 8 (n) + 8d (sum_x) = 8 + 8d.
        # v0.9.22 (§13.4/242): C_ASSUMED=64 (was 32). See Gaussian.
        return 64 * (8 + 8 * self.d)

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

    def batch_memory_factor(self) -> float:
        # Post-O3: Xb is bool (1/8 memory), Xc small. Peak = X + Xc + Xb_bool
        # + allocator. Coef measured = 2.006 (iter-5). Guard = ceil to 3.0.
        return 3.0


    def __init__(self, feature_cols: Sequence[str], target_col: str):
        self.feature_cols = list(feature_cols)
        self.target_col = target_col
        self.d = len(feature_cols)

    def estimate_state_nbytes(self) -> int:
        # Per class: 8 (n) + 8d (count_ones) = 8 + 8d.
        # v0.9.22 (§13.4/242): C_ASSUMED=64 (was 32). See Gaussian.
        return 64 * (8 + 8 * self.d)

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
        # O3 (DEC-020): keep bool, .sum(axis=0) returns float64 anyway.
        # Saves one full (B, d) f64 copy per update (coef 2.94 -> ~2.0).
        Xb = (X > 0.5)
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

def _check_X_non_negative(df, X_cols) -> None:
    """Multinomial NB requires X >= 0 (counts). 1-pass, cheap.

    Raises ValueError on first negative found. Silent NaN otherwise
    (feature_sum + alpha may go negative → log → nan).
    """
    import pyarrow.compute as pc
    for b in df._iter_filtered_batches():
        for c in X_cols:
            col = b.column(c)
            min_val = pc.min(col, skip_nulls=True).as_py()
            if min_val is not None and min_val < 0:
                raise ValueError(
                    f"fit_multinomial_nb: X column {c!r} has negative "
                    f"value {min_val}; counts must be >= 0"
                )


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
    # v0.9.34 (A9): uniform ML log.
    from cadw.ml._logging import log_fit
    log_fit("naive_bayes.gaussian", state_bytes=int(mean.nbytes + var.nbytes + n_c.nbytes))
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
    _check_X_non_negative(df, X_cols)
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
    # v0.9.34 (A9): uniform ML log.
    from cadw.ml._logging import log_fit
    log_fit("naive_bayes.multinomial", state_bytes=int(feature_sum.nbytes + n_c.nbytes))
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
    # v0.9.34 (A9): uniform ML log.
    from cadw.ml._logging import log_fit
    log_fit("naive_bayes.bernoulli", state_bytes=int(count_ones.nbytes + n_c.nbytes))
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
