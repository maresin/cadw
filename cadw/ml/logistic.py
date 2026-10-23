"""Logistic regression (v0.7.0b).

Composition: iter-Reduce Newton IRLS + solve. Peak RAM = batch + O(d²).

Model:
    P(y=1 | x) = sigmoid(x · coef + intercept)

Optimization (Newton-Raphson on log-likelihood):
    grad = X^T (p - y)                    # d
    hess = X^T diag(p*(1-p)) X            # d×d
    step = solve(hess + α·I, grad)        # α on non-intercept
    β ← β - step

    Each iteration: one Reduce pass over df.

Stopping: ||step||_inf < tol, or max_iter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.ops.reduce import ReduceOp, reduce


class LogisticGradOp(ReduceOp):
    """One Newton iteration: accumulate (grad, hess, log-likelihood).

    State: O(d²). Independent of N.

    Beta is passed from previous iteration; y must be binary {0,1}.
    Rows with null in X or y are skipped.
    """

    def __init__(
        self,
        feature_cols: Sequence[str],
        target_col: str,
        beta: np.ndarray,
    ) -> None:
        if not feature_cols:
            raise ValueError("LogisticGradOp: feature_cols must be non-empty")
        self.feature_cols = list(feature_cols)
        self.target_col = target_col
        self.d = len(self.feature_cols) + 1  # + intercept
        self.beta = np.asarray(beta, dtype=np.float64)
        if self.beta.shape != (self.d,):
            raise ValueError(
                f"LogisticGradOp: beta shape {self.beta.shape}, "
                f"expected ({self.d},)"
            )

    def init(self):
        return (
            0,                                # n_valid
            np.zeros(self.d, dtype=np.float64),
            np.zeros((self.d, self.d), dtype=np.float64),
            0.0,                              # log-likelihood
        )

    def update(self, state, batch):
        n, grad, hess, ll = state
        X, y = _stack_and_validate(batch, self.feature_cols, self.target_col)
        if X.shape[0] == 0:
            return state
        X_aug = np.column_stack([np.ones(X.shape[0]), X])  # (B, d)
        z = X_aug @ self.beta                              # (B,)
        # Numerically stable sigmoid
        p = 0.5 * (1.0 + np.tanh(0.5 * z))                 # (B,)
        w = p * (1.0 - p)                                  # (B,)
        # Grad / hess
        grad_new = X_aug.T @ (p - y)
        hess_new = (X_aug * w[:, None]).T @ X_aug
        # Log-likelihood (clip for numerical stability)
        eps = 1e-15
        p_clip = np.clip(p, eps, 1.0 - eps)
        ll_new = float(
            (y * np.log(p_clip) + (1.0 - y) * np.log(1.0 - p_clip)).sum()
        )
        return (
            n + X.shape[0],
            grad + grad_new,
            hess + hess_new,
            ll + ll_new,
        )

    def merge(self, a, b):
        return (
            a[0] + b[0],
            a[1] + b[1],
            a[2] + b[2],
            a[3] + b[3],
        )

    def state_nbytes(self, state):
        n, grad, hess, ll = state
        return int(grad.nbytes + hess.nbytes + 8 + 8)

    def estimate_state_nbytes(self):
        d = self.d
        return 8 + 8 * d + 8 * d * d + 8

    def finalize(self, state):
        n, grad, hess, ll = state
        return {
            "n": int(n), "grad": grad, "hess": hess, "ll": float(ll),
        }


def _stack_and_validate(
    batch: pa.RecordBatch,
    feature_cols: Sequence[str],
    target_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract numeric X, y; mask rows with null or non-binary y."""
    X_parts = [_column_to_f64(batch.column(c)) for c in feature_cols]
    X = np.column_stack(X_parts)
    y = _column_to_f64(batch.column(target_col))
    # Mask: no NaN in X or y; y in {0, 1}
    mask = (
        ~np.isnan(X).any(axis=1)
        & ~np.isnan(y)
        & ((y == 0.0) | (y == 1.0))
    )
    return X[mask], y[mask]


def _column_to_f64(arr: pa.Array) -> np.ndarray:
    np_arr = arr.to_numpy(zero_copy_only=False)
    if np_arr.dtype == object:
        out = np.empty(len(np_arr), dtype=np.float64)
        for i, v in enumerate(np_arr):
            out[i] = np.nan if v is None else float(v)
        return out
    return np_arr.astype(np.float64, copy=False)


def _validate_binary_y(df, y_col: str) -> set:
    """Ensure y has exactly values in {0, 1}, both present. O(distinct)."""
    seen: set = set()
    for b in df._iter_filtered_batches():
        col = b.column(y_col)
        # ChunkedArray.unique returns Array
        try:
            uniq = col.unique()
        except Exception:
            uniq = None
        if uniq is not None:
            for v in uniq.to_pylist():
                if v is None:
                    continue
                fv = float(v)
                if fv != 0.0 and fv != 1.0:
                    raise ValueError(
                        f"fit_logistic: y must be binary (0/1); "
                        f"got {v!r}"
                    )
                seen.add(fv)
                if len(seen) == 2:
                    # Both classes present, but still scan rest for invalid
                    pass
        else:
            for v in col.to_pylist():
                if v is None:
                    continue
                fv = float(v)
                if fv != 0.0 and fv != 1.0:
                    raise ValueError(
                        f"fit_logistic: y must be binary (0/1); "
                        f"got {v!r}"
                    )
                seen.add(fv)
    if len(seen) < 2:
        raise ValueError(
            f"fit_logistic: y must have both classes 0 and 1; got {seen}"
        )
    return seen


@dataclass
class LogisticModel:
    """Fitted logistic regression model."""

    coef_: np.ndarray
    intercept_: float
    feature_names_: list[str]
    target_col_: str
    alpha_: float
    n_rows_: int
    n_iter_: int
    converged_: bool
    log_likelihood_: float
    fitted_: bool = True

    def to_dict(self) -> dict:
        return {
            "coefficients": dict(zip(self.feature_names_, self.coef_.tolist())),
            "intercept": float(self.intercept_),
            "alpha": float(self.alpha_),
            "n_rows": int(self.n_rows_),
            "n_iter": int(self.n_iter_),
            "converged": bool(self.converged_),
            "log_likelihood": float(self.log_likelihood_),
        }

    def predict_proba(self, X):
        """P(y=1|x). Returns np.ndarray for np.ndarray input; for
        CadwDataFrame adds `{target}_prob` column and returns CadwDataFrame.
        """
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            if X.shape[1] != len(self.coef_):
                raise ValueError(
                    f"predict_proba: X has {X.shape[1]} features, "
                    f"expected {len(self.coef_)}"
                )
            z = X @ self.coef_ + self.intercept_
            return 0.5 * (1.0 + np.tanh(0.5 * z))

        return self._predict_df(X, proba_only=True)

    def predict(self, X, threshold: float = 0.5):
        """Binary prediction (>= threshold → 1).

        For CadwDataFrame: adds BOTH `{target}_prob` and `{target}_pred`.
        """
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(
                f"predict: threshold must be in [0, 1]; got {threshold}"
            )
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            proba = self.predict_proba(X)
            return (proba >= threshold).astype(np.int64)
        return self._predict_df(X, proba_only=False, threshold=threshold)

    def _predict_df(self, X, *, proba_only: bool, threshold: float = 0.5):
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

        coef = self.coef_
        intercept = self.intercept_

        def _iter_pred():
            for b in X._iter_filtered_batches():
                Xm = np.column_stack(
                    [_column_to_f64(b.column(c)) for c in self.feature_names_]
                )
                z = Xm @ coef + intercept
                prob = 0.5 * (1.0 + np.tanh(0.5 * z))
                arrays = [b.column(c) for c in X._columns]
                arrays.append(pa.array(prob, type=pa.float64()))
                if not proba_only:
                    pred = (prob >= threshold).astype(np.int64)
                    arrays.append(pa.array(pred, type=pa.int64()))
                yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

        new_ds = Dataset.from_batches(
            out_schema, _iter_pred(), X._new_dataset_dir()
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=X._memory_guard,
            spill_dir=X._spill_dir,
        )


def fit_logistic(
    df,
    X_cols,
    y_col: str,
    *,
    alpha: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-6,
    memory_guard: Optional[MemoryGuard] = None,
) -> LogisticModel:
    """Fit binary logistic regression via Newton-Raphson IRLS.

    - X_cols: feature columns (numeric). Use encode_categorical() for strings.
    - y_col: target column, must be binary {0,1}, both classes required.
    - alpha: L2 penalty (non-intercept). Default 1.0 (stability).
    - max_iter, tol: stopping criteria.
    - memory_guard: default df._memory_guard.

    Iterations: typically 5-15; each is one full Reduce pass.
    Peak RAM = batch + O(d²). Independent of N.

    Raises:
        ValueError: empty X, target in X, negative alpha, all-null rows,
                    non-binary y, only one class in y.
        KeyError: missing column.
        TypeError: non-numeric column.
    """
    if isinstance(X_cols, str):
        X_cols = [X_cols]
    else:
        X_cols = list(X_cols)
    if not X_cols:
        raise ValueError("fit_logistic: X_cols must be non-empty")
    if y_col in X_cols:
        raise ValueError(
            f"fit_logistic: target {y_col!r} must not be in X_cols"
        )
    if alpha < 0:
        raise ValueError(f"fit_logistic: alpha must be >= 0; got {alpha}")
    if max_iter < 1:
        raise ValueError(f"fit_logistic: max_iter must be >= 1; got {max_iter}")
    if tol <= 0:
        raise ValueError(f"fit_logistic: tol must be > 0; got {tol}")
    for c in X_cols:
        if c not in df.columns:
            raise KeyError(f"fit_logistic: column {c!r} not in df")
    if y_col not in df.columns:
        raise KeyError(f"fit_logistic: target {y_col!r} not in df")

    schema = df._dataset.schema
    for c in X_cols + [y_col]:
        t = schema.field(c).type
        if not (pa.types.is_integer(t) or pa.types.is_floating(t)):
            raise TypeError(
                f"fit_logistic: column {c!r} has type {t}; "
                f"numeric required (use encode_categorical for strings)"
            )

    # Validate binary y + both classes
    _validate_binary_y(df, y_col)

    mg = memory_guard or df._memory_guard
    d = len(X_cols) + 1  # + intercept
    beta = np.zeros(d, dtype=np.float64)
    ll_prev = -np.inf
    converged = False
    n_iter = 0
    n_valid = 0
    ll = 0.0

    for it in range(max_iter):
        n_iter = it + 1
        op = LogisticGradOp(X_cols, y_col, beta)
        res = reduce(df._iter_filtered_batches(), op, mg, log=False)
        n_valid = res["n"]
        if n_valid == 0:
            raise ValueError("fit_logistic: no valid (non-null, binary) rows")
        grad = res["grad"]
        hess = res["hess"]
        ll = res["ll"]

        reg = alpha * np.eye(d, dtype=np.float64)
        reg[0, 0] = 0.0  # intercept not penalized
        try:
            step = np.linalg.solve(hess + reg, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess + reg, grad, rcond=None)[0]

        step_inf = float(np.max(np.abs(step)))
        beta = beta - step

        # v0.9.31: uniform training log (opt-in via CADW_LOG=1).
        try:
            from cadw.ml._logging import log_fit
            log_fit("logistic", iter_i=n_iter, iter_n=max_iter,
                    loss=-float(ll), state_bytes=int((d*d + d) * 8))
        except Exception:
            pass

        if step_inf < tol:
            converged = True
            break

        # Δll check — safety net if Newton oscillates
        if abs(ll - ll_prev) < tol * 0.1 and it >= 1:
            converged = True
            break
        ll_prev = ll

    return LogisticModel(
        coef_=beta[1:],
        intercept_=float(beta[0]),
        feature_names_=list(X_cols),
        target_col_=y_col,
        alpha_=float(alpha),
        n_rows_=int(n_valid),
        n_iter_=int(n_iter),
        converged_=bool(converged),
        log_likelihood_=float(ll),
    )
