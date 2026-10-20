"""Regression + log_loss metrics via ReduceOp (v0.9.2a).

Single-pass accumulation, O(1) state (7 floats + 2 ints). Independent
of N. Nulls skipped per row (skipna=True).

Metrics:
  - r2        : coefficient of determination
  - mse,rmse  : mean squared error, root
  - mae       : mean absolute error
  - log_loss  : binary cross-entropy (requires proba in [0,1])

Regressions:
  - ss_tot == 0 → r2 = 1.0 if ss_res == 0 (perfect), else 0.0
    (matches sklearn.metrics.r2_score)
  - Empty input → NaN (honest "no data")

Classification metrics (accuracy, precision, recall, f1, roc_auc) —
v0.9.2b (require confusion matrix, K×K state; roc_auc needs sort).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pyarrow as pa

from cadw.ops.reduce import ReduceOp


REGRESSION_METRICS = frozenset({"r2", "mse", "rmse", "mae"})
PROBA_METRICS = frozenset({"log_loss"})
SUPPORTED_METRICS = REGRESSION_METRICS | PROBA_METRICS


def _to_f64(arr: pa.Array) -> np.ndarray:
    """Arrow Array → np.float64 with NaN for nulls."""
    np_arr = arr.to_numpy(zero_copy_only=False)
    if np_arr.dtype == object:
        out = np.empty(len(np_arr), dtype=np.float64)
        for i, v in enumerate(np_arr):
            out[i] = np.nan if v is None else float(v)
        return out
    return np_arr.astype(np.float64, copy=False)


class MetricsOp(ReduceOp):
    """Regression + log_loss metrics in one Reduce pass.

    Usage:
        op = MetricsOp("y", y_pred_col="yp", metrics=["r2", "rmse"])
        result = reduce(batches, op, guard)
        # {"r2": 0.85, "rmse": 0.32}
    """

    def __init__(
        self,
        y_true_col: str,
        *,
        y_pred_col: Optional[str] = None,
        y_proba_col: Optional[str] = None,
        metrics: Sequence[str] | str,
    ) -> None:
        if isinstance(metrics, str):
            metrics = [metrics]
        self.metrics = list(metrics)
        unknown = [m for m in self.metrics if m not in SUPPORTED_METRICS]
        if unknown:
            raise ValueError(
                f"MetricsOp: unknown metric(s) {unknown}; "
                f"supported: {sorted(SUPPORTED_METRICS)}"
            )
        self.y_true_col = y_true_col
        self.y_pred_col = y_pred_col
        self.y_proba_col = y_proba_col

        needs_pred = bool(set(self.metrics) & REGRESSION_METRICS)
        needs_proba = bool(set(self.metrics) & PROBA_METRICS)
        if needs_pred and y_pred_col is None:
            raise ValueError(
                f"MetricsOp: y_pred_col required for "
                f"{sorted(set(self.metrics) & REGRESSION_METRICS)}"
            )
        if needs_proba and y_proba_col is None:
            raise ValueError(
                f"MetricsOp: y_proba_col required for "
                f"{sorted(set(self.metrics) & PROBA_METRICS)}"
            )

    def estimate_state_nbytes(self) -> int:
        # 7 counters (n, sum_y, sum_yy, sum_res_sq, sum_abs_res, sum_ll,
        # n_proba) × 8 B = 56, rounded to 64.
        return 64

    def init(self):
        return {
            "n": 0,
            "sum_y": 0.0,
            "sum_yy": 0.0,
            "sum_residual_sq": 0.0,
            "sum_abs_residual": 0.0,
            "sum_log_loss": 0.0,
            "n_proba": 0,
        }

    def update(self, state, batch):
        # Regression
        if self.y_pred_col is not None:
            y = _to_f64(batch.column(self.y_true_col))
            yp = _to_f64(batch.column(self.y_pred_col))
            mask = ~np.isnan(y) & ~np.isnan(yp)
            if mask.any():
                y_m = y[mask]
                yp_m = yp[mask]
                resid = y_m - yp_m
                state["n"] += int(mask.sum())
                state["sum_y"] += float(y_m.sum())
                state["sum_yy"] += float((y_m * y_m).sum())
                state["sum_residual_sq"] += float((resid * resid).sum())
                state["sum_abs_residual"] += float(np.abs(resid).sum())

        # Log loss (binary; y must be {0,1}, p in [0,1])
        if self.y_proba_col is not None:
            y = _to_f64(batch.column(self.y_true_col))
            p = _to_f64(batch.column(self.y_proba_col))
            mask = ~np.isnan(y) & ~np.isnan(p)
            if mask.any():
                y_m = y[mask]
                p_m = p[mask]
                # §13.4/163: validate y ∈ {0, 1} — log_loss formula
                # assumes binary; continuous y gives mathematically
                # invalid (often negative) output silently.
                bad_mask = (y_m != 0.0) & (y_m != 1.0)
                if bad_mask.any():
                    bad_vals = y_m[bad_mask][:5].tolist()
                    raise ValueError(
                        f"MetricsOp: log_loss requires binary y_true "
                        f"∈ {{0, 1}}; got values {bad_vals} "
                        f"(first 5 of {int(bad_mask.sum())} invalid). "
                        f"Use `encode_categorical` for multiclass or "
                        f"threshold predictions. §13.4/163."
                    )
                eps = 1e-15
                p_c = np.clip(p_m, eps, 1.0 - eps)
                ll = -(y_m * np.log(p_c)
                       + (1.0 - y_m) * np.log(1.0 - p_c))
                state["sum_log_loss"] += float(ll.sum())
                state["n_proba"] += int(mask.sum())

        return state

    def merge(self, a, b):
        return {k: a[k] + b[k] for k in a}

    def state_nbytes(self, state):
        return 8 * 7  # 7 counters × 8 B

    def finalize(self, state):
        results = {}
        nan = float("nan")
        n = state["n"]

        if "r2" in self.metrics:
            if n > 0:
                y_mean = state["sum_y"] / n
                ss_tot = state["sum_yy"] - n * y_mean * y_mean
                ss_res = state["sum_residual_sq"]
                if ss_tot <= 0:
                    results["r2"] = 1.0 if ss_res <= 0 else 0.0
                else:
                    results["r2"] = 1.0 - ss_res / ss_tot
            else:
                results["r2"] = nan

        if "mse" in self.metrics:
            results["mse"] = (
                state["sum_residual_sq"] / n if n > 0 else nan
            )
        if "rmse" in self.metrics:
            results["rmse"] = (
                float(np.sqrt(state["sum_residual_sq"] / n))
                if n > 0 else nan
            )
        if "mae" in self.metrics:
            results["mae"] = (
                state["sum_abs_residual"] / n if n > 0 else nan
            )

        if "log_loss" in self.metrics:
            results["log_loss"] = (
                state["sum_log_loss"] / state["n_proba"]
                if state["n_proba"] > 0 else nan
            )

        return results
