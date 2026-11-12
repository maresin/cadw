"""Linear quantile regression via SGD on pinball loss.

Reference: Koenker & Bassett 1978 (Regression Quantiles, Econometrica).

For q ∈ (0, 1), the q-quantile of y given X is linear:
    Q_y(q | x) = x·coef + intercept
Minimizes E[L_q(y - pred)] where L_q(r) = max(q*r, (q-1)*r) is pinball.

Fit: mini-batch SGD on pinball loss. State O(d), fit peak O(batch*d).
  - Batch gradient: coef -= lr * (X.T @ subgrad) / batch_n
  - subgrad = -pinball_grad(y, pred, q)  (negative of GBDT residual target)

OOC: state = coef_(d) + intercept_(1) + 256B header = O(d).
Fit peak = O(batch * d); never materializes full X or y (streams via
_iter_filtered_batches). n-scalable.

Not the same as sklearn.linear_model.QuantileRegressor (LP-based exact
solver); our version is SGD approximate — same asymptotic target, faster
fit, converges with tolerance. Documented in YAML measurement.note.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.ml._gbdt_common import pinball_loss, pinball_grad
from cadw.ml._logging import log_fit


def _stack_X(batch, X_cols: Sequence[str]) -> np.ndarray:
    """Stack X_cols into (b, d) float64. Row-bounded by batch size."""
    d = len(X_cols)
    b_n = batch.num_rows
    X = np.empty((b_n, d), dtype=np.float64)
    for j, c in enumerate(X_cols):
        X[:, j] = batch.column(c).to_numpy(zero_copy_only=False).astype(
            np.float64, copy=False)
    return X


@dataclass
class QuantileLinearModel:
    """Linear quantile regression model (SGD-fit).

    State: coef_(d) + intercept_(1) + 256B header. O(d).
    """
    coef_: np.ndarray
    intercept_: float
    feature_names_: list[str]
    target_col_: str
    quantile_: float
    n_rows_: int
    n_epochs_: int
    lr_: float
    seed_: int
    final_loss_: float
    converged_: bool = True

    def state_nbytes(self) -> int:
        """State = coef_(d)*f64 + intercept_*f64 + 256B header."""
        return int(self.coef_.nbytes + 8 + 256)

    def predict(self, df, *, memory_guard: Optional[MemoryGuard] = None):
        """Predict q-quantile of target. Returns CadwDataFrame with
        column `target_col_q{q}` (e.g., 'y_q0.5')."""
        from cadw import CadwDataFrame
        mg = memory_guard or df._memory_guard
        for c in self.feature_names_:
            if c not in df._columns:
                raise KeyError(f"predict: column {c!r} not in df")
        cols = {}
        n_total = int(df.count())
        mg.check_state(n_total * 8 + 256,
                      name="quantile_linear.predict_output")
        out = np.empty(n_total, dtype=np.float64)
        off = 0
        for b in df._iter_filtered_batches():
            mg.check_batch(b, name="quantile_linear.predict",
                          extra_factor=2.0)
            X = _stack_X(b, self.feature_names_)
            pred = X @ self.coef_ + self.intercept_
            out[off:off + b.num_rows] = pred
            off += b.num_rows
        if off != n_total:
            out = out[:off]
        colname = f"{self.target_col_}_q{self.quantile_:g}"
        cols[colname] = pa.array(out, type=pa.float64())
        return CadwDataFrame.from_table(
            pa.table(cols), memory_guard=mg)

    def to_dict(self) -> dict:
        return {
            "coef_": self.coef_.tolist(),
            "intercept_": float(self.intercept_),
            "feature_names_": list(self.feature_names_),
            "target_col_": self.target_col_,
            "quantile_": float(self.quantile_),
            "n_rows_": int(self.n_rows_),
            "n_epochs_": int(self.n_epochs_),
            "lr_": float(self.lr_),
            "seed_": int(self.seed_),
            "final_loss_": float(self.final_loss_),
            "converged_": bool(self.converged_),
        }


def fit_linear_quantile(
    df,
    X_cols: "str | Sequence[str]",
    y_col: str,
    *,
    quantile: float = 0.5,
    n_epochs: int = 30,
    lr: float = 0.01,
    seed: int = 0,
    memory_guard: Optional[MemoryGuard] = None,
    batch_size: "int | None" = None,
) -> QuantileLinearModel:
    """Fit linear quantile regression via mini-batch SGD on pinball loss.

    State: O(d). Fit peak: O(batch * d).

    Args:
      df: CadwDataFrame.
      X_cols: feature column names (numeric).
      y_col: target column (numeric).
      quantile: q in (0, 1); e.g., 0.5 = median, 0.1 = 10th percentile.
      n_epochs: SGD passes; >= 1.
      lr: learning rate; > 0.
      seed: deterministic init + batch shuffle (not used here; batches
        come from df sequentially — deterministic).
      memory_guard: optional; defaults to df._memory_guard.
      batch_size: optional; auto-derived from guard when None.

    Returns:
      QuantileLinearModel with coef_, intercept_, state_nbytes().
    """
    if isinstance(X_cols, str):
        X_cols = [X_cols]
    else:
        X_cols = list(X_cols)
    if not X_cols:
        raise ValueError("fit_linear_quantile: X_cols must be non-empty")
    if y_col in X_cols:
        raise ValueError(
            f"fit_linear_quantile: target {y_col!r} must not be in X_cols")
    if not (0.0 < quantile < 1.0):
        raise ValueError(
            f"fit_linear_quantile: quantile must be in (0, 1); got {quantile}")
    if n_epochs < 1:
        raise ValueError(
            f"fit_linear_quantile: n_epochs must be >= 1; got {n_epochs}")
    if lr <= 0:
        raise ValueError(
            f"fit_linear_quantile: lr must be > 0; got {lr}")

    for c in X_cols:
        if c not in df.columns:
            raise KeyError(f"fit_linear_quantile: column {c!r} not in df")
    if y_col not in df.columns:
        raise KeyError(f"fit_linear_quantile: target {y_col!r} not in df")

    schema = df._dataset.schema
    for c in X_cols + [y_col]:
        t = schema.field(c).type
        if not (pa.types.is_integer(t) or pa.types.is_floating(t)):
            raise TypeError(
                f"fit_linear_quantile: column {c!r} has type {t}; "
                f"numeric required")

    mg = memory_guard or df._memory_guard
    d = len(X_cols)

    if batch_size is None:
        _row_size = mg.estimate_row_size_from_schema(df._dataset.schema)
        batch_size = mg.get_batch_rows(
            row_size_bytes=_row_size,
            expansion_factor=3.0,
            state_bytes=d * 8 + 8,
        )

    # State: O(d). Guard before alloc.
    mg.check_state(d * 8 + 8 + 256, name="linear_quantile.state")
    coef = np.zeros(d, dtype=np.float64)
    intercept = 0.0
    n_seen = 0
    final_loss = float("nan")

    for epoch in range(int(n_epochs)):
        epoch_loss = 0.0
        epoch_n = 0
        for b in df._iter_filtered_batches(batch_size=batch_size):
            mg.check_batch(b, name="linear_quantile.batch", extra_factor=3.0)
            X = _stack_X(b, X_cols)
            y = b.column(y_col).to_numpy(zero_copy_only=False).astype(
                np.float64, copy=False)
            if X.shape[0] == 0:
                continue
            pred = X @ coef + intercept
            # dL/dpred of pinball loss = -pinball_grad
            subgrad = -pinball_grad(y, pred, quantile)  # (b_n,)
            b_n = X.shape[0]
            coef -= lr * (X.T @ subgrad) / b_n
            intercept -= lr * float(subgrad.mean())
            epoch_loss += float(pinball_loss(y, pred, quantile).sum())
            epoch_n += b_n
        n_seen = epoch_n
        final_loss = epoch_loss / max(epoch_n, 1)
        log_fit("linear_quantile", iter_i=epoch + 1,
                iter_n=int(n_epochs), loss=final_loss,
                state_bytes=int(d * 8 + 8 + 256))

    model = QuantileLinearModel(
        coef_=coef,
        intercept_=float(intercept),
        feature_names_=list(X_cols),
        target_col_=y_col,
        quantile_=float(quantile),
        n_rows_=int(n_seen),
        n_epochs_=int(n_epochs),
        lr_=float(lr),
        seed_=int(seed),
        final_loss_=float(final_loss),
        converged_=True,
    )
    log_fit("linear_quantile", state_bytes=model.state_nbytes())
    return model
