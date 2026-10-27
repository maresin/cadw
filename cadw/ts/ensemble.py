"""EnsembleForecaster: per-model pipeline + weighted averaging (v0.9.81a).

Wraps each member ForecasterBase in its own AutoForecastPipeline
(capability-based per-model transforms from v0.9.80), so members see
data in their native spaces:

  - SARIMA / ETS / AutoETS / AutoTheta: naked (handles trend+season
    natively); base model trained on raw y.
  - NaiveForecaster / GBDTForecaster (no exog 't'): wrapped with
    STL/MSTL (if seasonal_periods given) + AutoStationarizer (auto
    differencing). Base model trained on stationary residual.

Predictions are inverse-transformed per member (back to original
scale) and then weighted-averaged. This preserves diversity: if the
same detrended series were fed to all members, they would all produce
the same flat forecast; inverse adds back the same trend — zero
ensemble gain from diversity.

Weight strategies
-----------------
  'equal'        : simple mean of member predictions.
  'inverse_mase' : weights ∝ 1 / val_MASE; best members get higher
                   weight. Holdout = last `val_steps` points.
                   Default.
  'median'       : median of predictions (robust to one outlier
                   member). No weights; ignores val.
  'custom'       : explicit weight list (length == n_members).

Validation protocol for 'inverse_mase'
--------------------------------------
  1. Split y into train = y[:-val_steps], val = y[-val_steps:].
     Default val_steps = max(seasonal_period_or_1, ceil(0.2 * n)).
  2. Fit each member pipeline on train; predict val.
  3. MASE = mean(|y_val - pred_val|) / mean(|Δ y_train|).
  4. weights = (1/MASE) / sum(1/MASE); MASE capped to avoid inf.
  5. Refit each member pipeline on full y.

References
----------
  - Bates & Granger (1969): "The combination of forecasts" —
    inverse-variance weighting rationale.
  - Hyndman & Athanasopoulos, FPP3, ch. 12: forecast combination
    beats best individual on average; median robust to outliers.
  - Makridakis et al. M4: simple average or inverse-error weighting
    often matches sophisticated stacking.

Pure numpy; no sklearn/statsmodels/scipy/torch.
"""
from __future__ import annotations

import math
import warnings

import numpy as np

from cadw.ts.models import (
    ForecasterBase, _check_df, _extract_series,
)
from cadw.ts.transforms import AutoForecastPipeline


# ── Helpers ───────────────────────────────────────────────────────

def _naive_mae_train(y: np.ndarray, s: int) -> float:
    """MASE denominator: naive seasonal MAE on y (or diff if s<=1)."""
    y = np.asarray(y, dtype=np.float64)
    if s > 1 and y.size > s:
        return float(np.mean(np.abs(y[s:] - y[:-s])))
    if y.size > 1:
        return float(np.mean(np.abs(np.diff(y))))
    return 1.0


def _mase(y_true: np.ndarray, y_pred: np.ndarray,
          denom: float) -> float:
    return float(np.mean(np.abs(y_true - y_pred))
                 / max(abs(denom), 1e-12))


# ── EnsembleForecaster ───────────────────────────────────────────

class EnsembleForecaster(ForecasterBase):
    """Weighted ensemble of forecasters with per-member pipelines.

    Parameters
    ----------
    members : list[ForecasterBase]
        At least 2. Each will be wrapped in its own
        AutoForecastPipeline (per-model transforms). Members that are
        already AutoForecastPipeline instances are used as-is.
    weights : str or list[float], default='inverse_mase'
        'equal', 'inverse_mase', 'median', or explicit list.
    seasonal_periods : list[int] | None, default=None
        Passed to AutoForecastPipeline for each member. Members that
        handle seasonality natively ignore this. None -> empty list.
    val_steps : int | None, default=None
        Holdout size for 'inverse_mase'. None ->
        max(max(seasonal_periods) if any else 1, ceil(0.2*n)).
    max_d : int, default=2
        Passed to AutoStationarizer via AutoForecastPipeline.
    """

    def __init__(self, members, *,
                 weights="inverse_mase",
                 seasonal_periods=None,
                 val_steps=None,
                 max_d=2):
        super().__init__()
        if not members or len(members) < 2:
            raise ValueError(
                "EnsembleForecaster: need at least 2 members")
        for m in members:
            if not isinstance(m, ForecasterBase):
                raise ValueError(
                    f"EnsembleForecaster: member {type(m).__name__} "
                    f"is not a ForecasterBase")
        self.members = list(members)
        self.weights_spec = weights
        self.seasonal_periods = (list(seasonal_periods)
                                 if seasonal_periods is not None else [])
        self.val_steps = val_steps
        self.max_d = int(max_d)

        # Wrap each member in its own pipeline
        self._pipelines = [
            (m if isinstance(m, AutoForecastPipeline)
             else AutoForecastPipeline(
                 m, seasonal_periods=self.seasonal_periods,
                 max_d=self.max_d))
            for m in self.members
        ]

        self.member_weights_ = None
        self.member_val_mase_ = None
        self.y_ = None

        # Capability flags: OR across members
        self.handles_trend = any(
            getattr(p, "handles_trend", False)
            for p in self._pipelines)
        self.handles_seasonality = any(
            getattr(p, "handles_seasonality", False)
            for p in self._pipelines)
        self.handles_multi_seasonality = any(
            getattr(p, "handles_multi_seasonality", False)
            for p in self._pipelines)

        # v0.9.81a-fix1: validate weights_spec at construction (fail
        # fast on bad spec instead of waiting for fit()).
        self._resolve_weights_spec(len(self._pipelines))

    # ── Weight computation ────────────────────────────────────

    def _resolve_weights_spec(self, n: int) -> str:
        """Normalize weights_spec to a strategy name or None (custom)."""
        w = self.weights_spec
        if isinstance(w, str):
            if w not in ("equal", "inverse_mase", "median"):
                raise ValueError(
                    f"EnsembleForecaster: weights={w!r} not in "
                    f"'equal' | 'inverse_mase' | 'median'")
            return w
        if isinstance(w, (list, tuple, np.ndarray)):
            if len(w) != n:
                raise ValueError(
                    f"EnsembleForecaster: custom weights length "
                    f"{len(w)} != n_members {n}")
            return None
        raise ValueError(
            f"EnsembleForecaster: weights must be str or list, "
            f"got {type(w).__name__}")

    # ── Fit ───────────────────────────────────────────────────

    def fit(self, df, target_col, *, time_col=None, group_col=None):
        _check_df(df, target_col, time_col, group_col)
        y = _extract_series(df, target_col, time_col=time_col)
        n = y.size
        if n < 4:
            raise ValueError(
                "EnsembleForecaster: need at least 4 observations")
        if np.any(~np.isfinite(y)):
            raise ValueError(
                "EnsembleForecaster: series contains non-finite values")

        strategy = self._resolve_weights_spec(len(self._pipelines))

        # Custom weights: no validation, refit all members on full y.
        if strategy is None:
            w = np.asarray(self.weights_spec, dtype=np.float64)
            s = w.sum()
            if s <= 0:
                raise ValueError(
                    "EnsembleForecaster: custom weights sum <= 0")
            self.member_weights_ = w / s
            for p in self._pipelines:
                p.fit(df, target_col,
                      time_col=time_col, group_col=group_col)
            self.y_ = y
            self._is_fitted = True
            return self

        # 'equal' or 'median': no validation needed; equal weights
        # (median uses weights only for display).
        if strategy in ("equal", "median"):
            self.member_weights_ = np.full(
                len(self._pipelines), 1.0 / len(self._pipelines))
            for p in self._pipelines:
                p.fit(df, target_col,
                      time_col=time_col, group_col=group_col)
            self.y_ = y
            self._is_fitted = True
            return self

        # 'inverse_mase': holdout validation
        if self.val_steps is None:
            base = max(self.seasonal_periods) if self.seasonal_periods else 1
            val_steps = max(base, math.ceil(0.2 * n))
        else:
            val_steps = int(self.val_steps)
        val_steps = max(1, min(val_steps, n - 4))
        n_train = n - val_steps
        y_train = y[:n_train]
        y_val = y[n_train:]
        s_denom = max(self.seasonal_periods) if self.seasonal_periods else 1
        denom = _naive_mae_train(y_train, s_denom)

        # Import CadwDataFrame for train split
        import pyarrow as pa
        from cadw import CadwDataFrame
        train_df = CadwDataFrame.from_table(
            pa.table({target_col: pa.array(y_train)}))

        val_mase = []
        for p in self._pipelines:
            try:
                p.fit(train_df, target_col)
                f_val = p.predict(val_steps)
                m = _mase(y_val, f_val, denom)
                val_mase.append(max(m, 1e-6))
            except Exception as e:
                warnings.warn(
                    f"EnsembleForecaster: member "
                    f"{type(p.forecaster).__name__} failed on val "
                    f"({type(e).__name__}: {e}); MASE set to inf",
                    stacklevel=2,
                )
                val_mase.append(float("inf"))
        val_mase = np.array(val_mase)
        self.member_val_mase_ = val_mase

        # Weights: 1/MASE, normalized; inf -> 0
        finite = np.isfinite(val_mase)
        if not finite.any():
            warnings.warn(
                "EnsembleForecaster: all members failed on val; "
                "falling back to equal weights",
                stacklevel=2,
            )
            weights = np.full(len(self._pipelines),
                              1.0 / len(self._pipelines))
        else:
            inv = np.zeros_like(val_mase)
            inv[finite] = 1.0 / val_mase[finite]
            weights = inv / inv.sum()
        self.member_weights_ = weights

        # Refit all members on full series
        for p in self._pipelines:
            p.fit(df, target_col,
                  time_col=time_col, group_col=group_col)

        self.y_ = y
        self._is_fitted = True
        return self

    # ── Predict ───────────────────────────────────────────────

    def predict(self, steps):
        if not self._is_fitted:
            raise RuntimeError(
                "EnsembleForecaster.predict: call fit first")
        preds = np.stack([p.predict(steps)
                          for p in self._pipelines], axis=0)  # (m, h)
        if self._resolve_weights_spec(len(self._pipelines)) == "median":
            return np.median(preds, axis=0)
        return preds.T @ self.member_weights_

    def predict_in_sample(self):
        if not self._is_fitted:
            raise RuntimeError(
                "EnsembleForecaster.predict_in_sample: call fit first")
        ins = [p.predict_in_sample() for p in self._pipelines]
        # In-sample may have different lengths across members
        # (differencing shortens); use min length, truncate each from
        # start? No — align from end (most recent is most important).
        min_len = min(x.size for x in ins)
        aligned = np.stack(
            [x[-min_len:] for x in ins], axis=0)
        return aligned.T @ self.member_weights_

    def prediction_interval(self, steps, level=0.95):
        if not self._is_fitted:
            raise RuntimeError(
                "EnsembleForecaster.prediction_interval: call fit first")
        los, his = [], []
        for p in self._pipelines:
            if hasattr(p, "prediction_interval"):
                lo, hi = p.prediction_interval(steps, level)
            else:
                f = p.predict(steps)
                lo, hi = f, f
            los.append(lo)
            his.append(hi)
        los = np.stack(los, axis=0)
        his = np.stack(his, axis=0)
        return los.T @ self.member_weights_, his.T @ self.member_weights_

    def state_nbytes(self):
        return int(sum(p.state_nbytes() for p in self._pipelines))

    def get_params(self) -> dict:
        return {
            "members": [type(p.forecaster).__name__
                        if isinstance(p, AutoForecastPipeline)
                        else type(p).__name__
                        for p in self._pipelines],
            "weights_spec": (self.weights_spec
                             if isinstance(self.weights_spec, str)
                             else list(self.weights_spec)),
            "member_weights": (None if self.member_weights_ is None
                               else self.member_weights_.tolist()),
            "seasonal_periods": list(self.seasonal_periods),
        }

