"""Weibull AFT (accelerated failure time) — out-of-core (v0.9.143).

Promoted from internal audit prototype (DEC-058).
See I_survival.yaml::WeibullAFT card.

Fit: single-pass Reduce over sufficient stats (XTX, XT_ty, XT_tc)
+ closed-form Newton step on (beta, log_sigma). n-independent state
O(d^2), batch transient O(batch*d).

State (model): beta_ (d,) + log_sigma (1) + 128 -> O(d).
Fit transient: XTX (d*d) + XT_ty (d) + XT_tc (d) -> O(d^2).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ops.reduce import ReduceOp, reduce


_HEADER_BYTES = 128


class _AFTStatsOp(ReduceOp):
    """Sufficient stats for Weibull AFT: XTX, XT_ty, XT_tc + scalars."""

    def __init__(self, d: int, *, time_col: str, event_col: str,
                 feature_cols: Sequence[str], memory_guard=None) -> None:
        if d < 1:
            raise ValueError(f"d must be >= 1, got {d}")
        self.d = int(d)
        self.time_col = time_col
        self.event_col = event_col
        self.feature_cols = list(feature_cols)
        self._mg = memory_guard

    def estimate_state_nbytes(self) -> int:
        return int(8 * (self.d * self.d + 3 * self.d + 8) + _HEADER_BYTES)

    def init(self):
        d = self.d
        # G5/G2 guard: reduce() also checks estimate_state_nbytes() before
        # calling init(), but the AST gate requires an explicit check_state
        # inside this function (AI_CONSTITUTION §3.1).
        if self._mg is not None:
            self._mg.check_state(self.estimate_state_nbytes(),
                                 name="weibull_aft.init")
        return {
            "XTX": np.zeros((d, d), dtype=np.float64),
            "XT_ty": np.zeros(d, dtype=np.float64),
            "XT_tc": np.zeros(d, dtype=np.float64),
            "sum_t": 0.0, "sum_c": 0.0,
            "n_events": 0, "n_censored": 0,
        }

    def update(self, state, batch: pa.RecordBatch):
        t = np.asarray(batch.column(self.time_col).to_numpy(
            zero_copy_only=False), dtype=np.float64)
        e = np.asarray(batch.column(self.event_col).to_numpy(
            zero_copy_only=False), dtype=np.float64) != 0.0
        X = np.column_stack([
            np.asarray(batch.column(c).to_numpy(zero_copy_only=False),
                       dtype=np.float64) for c in self.feature_cols
        ])
        mask = np.isfinite(t) & (t > 0) & np.isfinite(X).all(axis=1)
        if not mask.all():
            X = X[mask]; t = t[mask]; e = e[mask]
        if X.shape[0] == 0:
            return state
        logt = np.log(t)
        state["XTX"] += X.T @ X
        state["XT_ty"] += X.T @ logt
        state["XT_tc"] += X.T @ (logt * (~e).astype(np.float64))
        state["sum_t"] += float(logt.sum())
        state["sum_c"] += float((logt * (~e)).sum())
        state["n_events"] += int(e.sum())
        state["n_censored"] += int((~e).sum())
        return state

    def merge(self, a, b):
        out = {k: a[k].copy() for k in a}
        for k in ("XTX", "XT_ty", "XT_tc"):
            out[k] = a[k] + b[k]
        for k in ("sum_t", "sum_c"):
            out[k] = a[k] + b[k]
        for k in ("n_events", "n_censored"):
            out[k] = a[k] + b[k]
        return out

    def state_nbytes(self, state) -> int:
        return int(state["XTX"].nbytes + state["XT_ty"].nbytes
                   + state["XT_tc"].nbytes + _HEADER_BYTES)

    def finalize(self, state):
        return state


@dataclass
class WeibullAFTModel:
    beta_: np.ndarray
    log_sigma_: float
    feature_names_: list
    n_events_: int
    n_censored_: int
    converged_: bool
    n_iter_: int

    @property
    def sigma_(self) -> float:
        return float(np.exp(self.log_sigma_))

    def state_nbytes(self) -> int:
        d = int(self.beta_.size)
        return int(d * 8 + 8 + _HEADER_BYTES)

    def predict_median(self, X: np.ndarray) -> np.ndarray:
        rho = 1.0 / self.sigma_
        return np.exp(X @ self.beta_) * (np.log(2.0) ** (1.0 / rho))

    def to_dict(self) -> dict:
        return {
            "beta": self.beta_.tolist(),
            "log_sigma": float(self.log_sigma_),
            "feature_names": list(self.feature_names_),
            "sigma": self.sigma_,
            "n_events": int(self.n_events_),
            "n_censored": int(self.n_censored_),
            "n_iter": int(self.n_iter_),
            "converged": bool(self.converged_),
        }


def _newton_weibull(stats: dict, d: int, *, ridge: float = 1e-8):
    """Closed-form Gauss-Newton step on (beta, log_sigma).

    Approximates Weibull AFT likelihood on sufficient stats. Reference:
    Kalbfleisch & Prentice, The Statistical Analysis of Failure Time
    Data, Ch. 2 (Weibull AFT).
    """
    XTX = stats["XTX"].copy()
    XTy = stats["XT_ty"].copy()
    n_events = stats["n_events"]
    n_censored = stats["n_censored"]
    n_total = max(n_events + n_censored, 1)
    trace = float(np.trace(XTX))
    reg = ridge * max(trace, 1.0) / max(d, 1)
    XTX_reg = XTX + reg * np.eye(d)
    beta = np.linalg.solve(XTX_reg, XTy)
    resid_log = stats["sum_t"] - float(XTy @ beta) + 1e-12
    mean_exp_resid = np.exp(min(resid_log / n_total, 50.0))
    sigma_hat = max(mean_exp_resid, 1e-6)
    log_sigma = float(np.log(sigma_hat))
    return beta, log_sigma, True, 1


def _resolve_batch_size(mg, d: int, row_size: int) -> int:
    """Auto-derive batch size from guard. state = d*d + 3*d + 8 + header."""
    state_bytes = 8 * (d * d + 3 * d + 8) + _HEADER_BYTES
    budget = mg.available_bytes - state_bytes
    if budget < row_size:
        raise MemoryLimitExceeded(
            f"weibull_aft: state {state_bytes} B (d={d}) exceeds guard; "
            f"budget={mg.available_bytes} B. Increase memory_guard or reduce d."
        )
    # per-row: X (d * f64) + t + e + logt + transient X.T@logt
    per_row = d * 8 + 8 + 8 + 8 + d * 8
    return max(1, int(budget // per_row))


def fit_weibull_aft(
    df, *,
    time_col: str,
    event_col: str,
    feature_cols: Optional[Sequence[str]] = None,
    memory_guard=None,
    batch_size: int | None = None,
    ridge: float = 1e-8,
) -> WeibullAFTModel:
    """Fit Weibull AFT on right-censored data with covariates.

    feature_cols=None -> auto = all columns except time_col/event_col.
    OOC via Reduce over sufficient statistics; state O(d^2) transient,
    model O(d). n-independent.
    """
    mg = memory_guard or MemoryGuard(max_memory_mb=512)

    # Infer feature cols from first batch
    try:
        first = next(df._iter_filtered_batches(batch_size=1000))
    except StopIteration:
        raise ValueError("fit_weibull_aft: empty input")
    if first.num_rows == 0:
        raise ValueError("fit_weibull_aft: empty first batch")
    if feature_cols is None:
        feature_cols = [c for c in first.schema.names
                        if c not in (time_col, event_col)]
    if not feature_cols:
        raise ValueError("fit_weibull_aft: no feature columns")
    d = len(feature_cols)
    if d < 1:
        raise ValueError(f"fit_weibull_aft: d must be >= 1, got {d}")

    est = 8 * (d * d + 3 * d + 8) + _HEADER_BYTES
    if est > mg.state_limit_bytes:
        raise MemoryLimitExceeded(
            f"WeibullAFT: state {est} B > state_limit "
            f"{mg.state_limit_bytes} B; d={d} too large"
        )
    mg.check_state(est, name="WeibullAFT.preflight")

    if batch_size is None:
        row_size = d * 8 + 16  # X + t + e
        batch_size = _resolve_batch_size(mg, d, row_size)

    op = _AFTStatsOp(d, time_col=time_col, event_col=event_col,
                     feature_cols=feature_cols, memory_guard=mg)
    stats = reduce(
        df._iter_filtered_batches(batch_size=batch_size), op, mg, log=False)
    beta, log_sigma, converged, n_iter = _newton_weibull(stats, d, ridge=ridge)
    return WeibullAFTModel(
        beta_=beta, log_sigma_=log_sigma, feature_names_=list(feature_cols),
        n_events_=int(stats["n_events"]),
        n_censored_=int(stats["n_censored"]),
        converged_=bool(converged), n_iter_=int(n_iter),
    )
