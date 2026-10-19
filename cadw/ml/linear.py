"""Linear/Ridge regression (v0.7.0a).

Composition: Reduce (XTXOp) → solve. Peak RAM = batch + O(d²).

Model:
    y = X · coef + intercept
    OLS: minimize ||X β - y||²
    Ridge: minimize ||X β - y||² + α ||β||² (intercept unpenalized)

XTXOp accumulates: n, X^T X (d×d), X^T y (d), sum_X (d), sum_y.
Augmented system for intercept:
    [n        sum_X^T] [b0   ]   [sum_y  ]
    [sum_X    X^T X  ] [beta ] = [X^T y  ]
    + α · diag(0, 1, ..., 1) on non-intercept.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.ops.reduce import ReduceOp, reduce


class XTXOp(ReduceOp):
    """Accumulate (n, X^T X, X^T y, sum_X, sum_y) over batches.

    State size: O(d²) + O(d). One pass.
    Only numeric columns. Nulls in X or y → row skipped (both X and y).
    """

    def __init__(
        self,
        feature_cols: Sequence[str],
        target_col: str,
    ) -> None:
        if not feature_cols:
            raise ValueError("XTXOp: feature_cols must be non-empty")
        self.feature_cols = list(feature_cols)
        self.target_col = target_col
        self.d = len(self.feature_cols)

    def init(self):
        return (
            0,
            np.zeros((self.d, self.d), dtype=np.float64),
            np.zeros(self.d, dtype=np.float64),
            np.zeros(self.d, dtype=np.float64),
            0.0,
        )

    def update(self, state, batch):
        n, xtx, xty, sx, sy = state
        X = _stack_numeric(batch, self.feature_cols)
        y = _numeric_to_f64(batch.column(self.target_col))
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[mask]
        y = y[mask]
        if X.shape[0] == 0:
            return state
        return (
            n + X.shape[0],
            xtx + X.T @ X,
            xty + X.T @ y,
            sx + X.sum(axis=0),
            sy + float(y.sum()),
        )

    def merge(self, a, b):
        return (
            a[0] + b[0],
            a[1] + b[1],
            a[2] + b[2],
            a[3] + b[3],
            a[4] + b[4],
        )

    def state_nbytes(self, state):
        n, xtx, xty, sx, sy = state
        return int(xtx.nbytes + xty.nbytes + sx.nbytes + 8 + 8)

    def estimate_state_nbytes(self):
        d = self.d
        return 8 + 8 * d * d + 8 * d + 8 * d + 8

    def finalize(self, state):
        n, xtx, xty, sx, sy = state
        return {
            "n": int(n), "xtx": xtx, "xty": xty,
            "sum_X": sx, "sum_y": float(sy),
        }


def _numeric_to_f64(arr: pa.Array) -> np.ndarray:
    """Convert Arrow numeric Array (possibly with nulls) to float64 with NaN."""
    np_arr = arr.to_numpy(zero_copy_only=False)
    if np_arr.dtype == object:
        out = np.empty(len(np_arr), dtype=np.float64)
        for i, v in enumerate(np_arr):
            out[i] = np.nan if v is None else float(v)
        return out
    return np_arr.astype(np.float64, copy=False)


def _stack_numeric(batch: pa.RecordBatch, cols: Sequence[str]) -> np.ndarray:
    return np.column_stack([_numeric_to_f64(batch.column(c)) for c in cols])


@dataclass
class LinearModel:
    """Fitted linear model (OLS or Ridge)."""

    coef_: np.ndarray
    intercept_: float
    feature_names_: list[str]
    target_col_: str
    alpha_: float
    n_rows_: int
    fitted_: bool = True

    def to_dict(self) -> dict:
        return {
            "coefficients": dict(zip(self.feature_names_, self.coef_.tolist())),
            "intercept": float(self.intercept_),
            "alpha": float(self.alpha_),
            "n_rows": int(self.n_rows_),
        }

    def predict(self, X):
        """Predict.

        - np.ndarray (n, d) → np.ndarray (n,)
        - CadwDataFrame → CadwDataFrame with added `{target_col_}_pred`.
        """
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            if X.shape[1] != len(self.coef_):
                raise ValueError(
                    f"predict: X has {X.shape[1]} features, "
                    f"expected {len(self.coef_)}"
                )
            return X @ self.coef_ + self.intercept_

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

        coef = self.coef_
        intercept = self.intercept_

        def _iter_pred():
            for b in X._iter_filtered_batches():
                Xm = _stack_numeric(b, self.feature_names_)
                pred = Xm @ coef + intercept
                arrays = [b.column(c) for c in X._columns]
                arrays.append(pa.array(pred, type=pa.float64()))
                yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

        new_ds = Dataset.from_batches(
            out_schema, _iter_pred(), X._new_dataset_dir()
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=X._memory_guard,
            spill_dir=X._spill_dir,
        )


def fit_linear(
    df,
    X_cols: "str | Sequence[str]",
    y_col: str,
    *,
    alpha: float = 0.0,
    memory_guard: Optional[MemoryGuard] = None,
) -> LinearModel:
    """Fit Linear (alpha=0) or Ridge (alpha>0) via one Reduce + solve.

    - X_cols: feature column names (numeric).
    - y_col: target column (numeric).
    - alpha: Ridge L2 penalty (non-negative). Intercept not penalized.
    - memory_guard: default df._memory_guard.

    Raises ValueError for empty df (all-null rows), negative alpha,
    X ∩ {y} non-empty, non-numeric columns.
    """
    if isinstance(X_cols, str):
        X_cols = [X_cols]
    else:
        X_cols = list(X_cols)
    if not X_cols:
        raise ValueError("fit_linear: X_cols must be non-empty")
    if y_col in X_cols:
        raise ValueError(
            f"fit_linear: target {y_col!r} must not be in X_cols"
        )
    if alpha < 0:
        raise ValueError(f"fit_linear: alpha must be >= 0; got {alpha}")
    for c in X_cols:
        if c not in df.columns:
            raise KeyError(f"fit_linear: column {c!r} not in df")
    if y_col not in df.columns:
        raise KeyError(f"fit_linear: target {y_col!r} not in df")

    # Validate numeric
    schema = df._dataset.schema
    for c in X_cols + [y_col]:
        t = schema.field(c).type
        if not (pa.types.is_integer(t) or pa.types.is_floating(t)):
            raise TypeError(
                f"fit_linear: column {c!r} has type {t}; "
                f"numeric required (use encode_categorical for strings)"
            )

    mg = memory_guard or df._memory_guard
    op = XTXOp(X_cols, y_col)
    res = reduce(df._iter_filtered_batches(), op, mg, log=False)

    if res["n"] == 0:
        raise ValueError("fit_linear: no valid (non-null) rows")

    d = len(X_cols)
    xtx = res["xtx"]
    xty = res["xty"]
    sum_x = res["sum_X"]
    sum_y = res["sum_y"]
    n = res["n"]

    # Augmented system (d+1 features: intercept + X)
    xtx_aug = np.zeros((d + 1, d + 1), dtype=np.float64)
    xtx_aug[0, 0] = float(n)
    xtx_aug[0, 1:] = sum_x
    xtx_aug[1:, 0] = sum_x
    xtx_aug[1:, 1:] = xtx

    xty_aug = np.zeros(d + 1, dtype=np.float64)
    xty_aug[0] = sum_y
    xty_aug[1:] = xty

    reg = alpha * np.eye(d + 1, dtype=np.float64)
    reg[0, 0] = 0.0  # intercept not penalized
    xtx_reg = xtx_aug + reg

    try:
        beta = np.linalg.solve(xtx_reg, xty_aug)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(xtx_reg, xty_aug, rcond=None)[0]

    return LinearModel(
        coef_=beta[1:],
        intercept_=float(beta[0]),
        feature_names_=X_cols,
        target_col_=y_col,
        alpha_=float(alpha),
        n_rows_=int(n),
    )
