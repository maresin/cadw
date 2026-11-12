"""CADWBoostForecaster — TimeSeries wrapper matching cadw.ts.models.GBDTForecaster.

API: fit(df, target_col, time_col=None, group_col=None),
     predict(horizon) → np.ndarray (recursive multi-step) or list[array].

Modes:
- "recursive" (default): fit one model on lag features, predict step-by-step,
  feed prediction back as lag for next step.
- "direct": fit one model per horizon h ∈ 1..H. Predict independently.
- "multi_output": fit one model, predict all H at once (H outputs via multiclass).
"""
from __future__ import annotations

import numpy as np

from .model import CADWBoost


def _make_lag_features(series: np.ndarray, lags: tuple[int, ...]) -> np.ndarray:
    """Given 1D series, build (n - max_lag, len(lags)) lag features."""
    n = len(series)
    max_lag = max(lags)
    n_out = n - max_lag
    if n_out <= 0:
        raise ValueError(
            f"series too short ({n}) for max lag {max_lag}")
    X = np.empty((n_out, len(lags)))
    for j, lag in enumerate(lags):
        X[:, j] = series[max_lag - lag:max_lag - lag + n_out]
    return X


def _make_y_direct(series: np.ndarray, lags: tuple[int, ...], horizon: int):
    """For direct strategy: X = lag features, y = series[t + horizon]."""
    n = len(series)
    max_lag = max(lags)
    n_out = n - max_lag - horizon + 1
    if n_out <= 0:
        raise ValueError(
            f"series too short ({n}) for max_lag={max_lag} + horizon={horizon}")
    X = np.empty((n_out, len(lags)))
    for j, lag in enumerate(lags):
        X[:, j] = series[max_lag - lag:max_lag - lag + n_out]
    y = series[max_lag + horizon - 1:max_lag + horizon - 1 + n_out]
    return X, y


class CADWBoostForecaster:
    """CADWBoost-based forecaster.

    Same semantics as cadw.ts.models.GBDTForecaster for basic cases
    (recursive + direct modes; multi_output → P1).
    """

    def __init__(
        self, lags: tuple = (1, 2, 3, 7),
        mode: str = "recursive",
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.1,
        random_state: int = 0,
        **boost_kwargs,
    ):
        if mode not in ("recursive", "direct"):
            raise ValueError(f"mode must be 'recursive'|'direct'; got {mode!r}")
        self.lags = tuple(int(l) for l in lags)
        self.mode = mode
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.boost_kwargs = boost_kwargs
        self.series_ = None
        self.model_ = None          # recursive: single model
        self.models_ = None         # direct: [model_h1, ..., model_hH]
        self.max_horizon_ = 0
        self.fitted_ = False

    def fit(self, series, horizon: int = 1):
        """series: 1D array-like. horizon: needed for direct mode."""
        series = np.asarray(series, dtype=np.float64).ravel()
        self.series_ = series.copy()

        if self.mode == "recursive":
            X = _make_lag_features(series, self.lags)
            y = series[max(self.lags):]
            self.model_ = CADWBoost(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                loss="mse", random_state=self.random_state, **self.boost_kwargs)
            self.model_.fit(X, y)
        else:  # direct
            if horizon < 1:
                raise ValueError(f"horizon must be >=1; got {horizon}")
            self.max_horizon_ = int(horizon)
            self.models_ = []
            for h in range(1, horizon + 1):
                X, y = _make_y_direct(series, self.lags, h)
                m = CADWBoost(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    loss="mse",
                    random_state=self.random_state + h,
                    **self.boost_kwargs)
                m.fit(X, y)
                self.models_.append(m)

        self.fitted_ = True
        return self

    def predict(self, horizon: int = 1):
        if not self.fitted_:
            raise RuntimeError("not fitted")
        if self.mode == "direct" and horizon > self.max_horizon_:
            raise ValueError(
                f"direct mode fitted for horizon {self.max_horizon_}, "
                f"cannot predict {horizon}")

        if self.mode == "direct":
            # Use model for step h; if h > trained horizon, use last model
            h = min(horizon, self.max_horizon_)
            # Reconstruct last X (lags of series)
            series = self.series_
            max_lag = max(self.lags)
            # Need one X per step (the last-known lags); for h-step,
            # model_h was trained on t where y = series[t+h-1], X = lags at t
            # So predict for t=len(series)-max_lag is "h steps ahead"
            X_last = np.empty((1, len(self.lags)))
            for j, lag in enumerate(self.lags):
                # lag L means value from L steps ago = series[-L]
                X_last[0, j] = series[-lag]
            m = self.models_[h - 1]
            return np.array([m.predict(X_last)[0]])

        # recursive: predict step-by-step, feed back
        series = list(self.series_)
        preds = []
        max_lag = max(self.lags)
        for _ in range(horizon):
            X = np.empty((1, len(self.lags)))
            for j, lag in enumerate(self.lags):
                X[0, j] = series[-lag]
            p = float(self.model_.predict(X)[0])
            preds.append(p)
            series.append(p)
        return np.array(preds)

    def state_nbytes(self):
        if self.model_ is not None:
            return self.model_.state_nbytes()
        if self.models_ is not None:
            return sum(m.state_nbytes() for m in self.models_)
        return 0


def fit_forecaster(series, *, horizon: int = 1, lags=(1, 2, 3, 7),
                   mode: str = "recursive", **kwargs):
    """Convenience: fit and return CADWBoostForecaster."""
    f = CADWBoostForecaster(lags=lags, mode=mode, **kwargs)
    f.fit(series, horizon=horizon)
    return f
