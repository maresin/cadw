"""Target transforms and pipeline wrappers for TS forecasters (v0.9.79a).

BaseTransformer + Differencer + TransformedTargetForecaster.
STLTransformer, MSTLTransformer, AutoStationarizer in v0.9.79b (added
to this module).

Design rationale (why per-model pipeline, not global transform):
  - Some models handle trend/seasonality natively (SARIMA via d/D,
    ETS/AutoETS via state-space, AutoTheta via STL decomposition inside
    fit). Applying external differencing to them would double-
    stationarize -> inverse transform reconstructs wrong scale.
  - Models without native handling (GBDTForecaster without exog 't',
    NaiveForecaster, plain AR without differencing) need external
    differencing/STL to remove trend/seasonality before fit, with
    inverse transform applied to predictions.
  - Global ensemble pipeline (apply one transform to all members)
    destroys diversity: all models see the same detrended series,
    so all produce the same flat forecast; inverse adds back the same
    trend -> zero ensemble gain from diversity.
  - Per-model TransformedTargetForecaster respects each model's
    capabilities. Ensemble wraps each member in its own pipeline.

Transformer contract:
  - fit(y): learn state (last values, seasonal component, etc.)
  - transform(y): return transformed series (same length or shorter;
    length reduction is tracked in `n_obs_used_`)
  - inverse_transform(y_transformed, steps, original_tail=None):
    map forecast back to original scale. `original_tail` may be
    passed explicitly, or the transformer may store tail from fit.
  - get_params / set_params: for saving/loading via cv.py.

TransformedTargetForecaster:
  - applies transformer(s) in sequence: y -> t1 -> t2 -> ... -> base
  - fit: transform y; fit base on transformed; store inverse state.
  - predict: base.predict(steps); inverse_transform in reverse order.
  - predict_in_sample: base.predict_in_sample; inverse_transform.

Pure numpy (no sklearn/statsmodels/torch/scipy at module level; scipy
is allowed via lazy import if needed, per project conventions).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from math import comb

import numpy as np

from cadw.ts.models import (
    ForecasterBase, _check_df, _extract_series,
)


# ── Base ──────────────────────────────────────────────────────────

class BaseTransformer(ABC):
    """Base class for target transforms.

    All transformers are stateful: fit stores state; transform uses it;
    inverse_transform applies inverse with the same state.
    """

    def __init__(self):
        self._is_fitted = False

    @abstractmethod
    def fit(self, y: np.ndarray) -> "BaseTransformer":
        """Learn state from y. Returns self."""

    @abstractmethod
    def transform(self, y: np.ndarray) -> np.ndarray:
        """Apply forward transform. May reduce series length."""

    @abstractmethod
    def inverse_transform(self, y_transformed: np.ndarray,
                          steps: int,
                          original_tail: np.ndarray | None = None
                          ) -> np.ndarray:
        """Apply inverse transform to `y_transformed` (forecast of length
        `steps`). Returns `steps`-length predictions on original scale.

        `original_tail` is optional context for inverse (e.g. the last
        `lags` values of original series); if None, uses state from fit.
        """

    def get_params(self) -> dict:
        return {}


# ── Differencer ───────────────────────────────────────────────────

class Differencer(BaseTransformer):
    """Regular and/or seasonal differencing with inverse.

    Parameters
    ----------
    lags : int, default=1
        Regular differencing order d (0 = no differencing, 1 = first
        difference, 2 = second difference).
    seasonal_period : int | None, default=None
        Seasonal period s. None -> no seasonal differencing.
    D : int, default=0
        Seasonal differencing order (0, 1, 2).

    Fit / transform semantics:
      - fit(y) stores the last `lags` and last `D * s` values needed
        for inverse (plus one step for seasonal diff alignment).
      - transform(y): apply d-times regular diff, then D-times seasonal
        diff. Result length = n - lags - D * s.
      - inverse_transform(y_t, steps): reconstruct original scale from
        forecast y_t. Requires the tail stored at fit.

    Order of operations (matches statsmodels/ARIMA convention):
      1. Regular diff d times: y -> Δy -> Δ²y -> ...
      2. Seasonal diff D times on top: Δy -> Δ_s Δy -> ...
      Inverse in reverse order.
    """

    def __init__(self, lags: int = 1, seasonal_period: int | None = None,
                 D: int = 0):
        super().__init__()
        if lags < 0:
            raise ValueError(
                f"Differencer: lags must be >= 0, got {lags}")
        if D < 0:
            raise ValueError(
                f"Differencer: D must be >= 0, got {D}")
        if D > 0 and (seasonal_period is None or seasonal_period < 2):
            raise ValueError(
                f"Differencer: D>0 requires seasonal_period >= 2, "
                f"got seasonal_period={seasonal_period}")
        self.lags = int(lags)
        self.seasonal_period = (int(seasonal_period)
                                if seasonal_period is not None else None)
        self.D = int(D)
        # State
        self.tail_regular_ = None   # last `lags` values of y before diff
        self.tail_seasonal_ = None  # last `D*s` values of Δ^d y before seasonal diff
        self.n_obs_used_ = 0

    def fit(self, y: np.ndarray) -> "Differencer":
        y = np.asarray(y, dtype=np.float64).ravel()
        n = y.size
        # Step 1: store regular tail
        if self.lags > 0:
            if n < self.lags + 1:
                raise ValueError(
                    f"Differencer.fit: n={n} < lags+1={self.lags+1}")
            self.tail_regular_ = y[-self.lags:].copy()
            # Apply d regular diffs to know seasonal tail reference
            y_after_reg = y.copy()
            for _ in range(self.lags):
                y_after_reg = np.diff(y_after_reg)
        else:
            y_after_reg = y.copy()
        # Step 2: store seasonal tail (on regular-diffed series)
        if self.D > 0:
            s = self.seasonal_period
            if y_after_reg.size < self.D * s + 1:
                raise ValueError(
                    f"Differencer.fit: n after regular diff = "
                    f"{y_after_reg.size} < D*s+1={self.D*s+1}")
            self.tail_seasonal_ = y_after_reg[-self.D * s:].copy()
        else:
            self.tail_seasonal_ = None
        self._is_fitted = True
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Differencer.transform: call fit first")
        y = np.asarray(y, dtype=np.float64).ravel()
        out = y.copy()
        for _ in range(self.lags):
            out = np.diff(out)
        for _ in range(self.D):
            out = out[self.seasonal_period:] - out[:-self.seasonal_period]
        self.n_obs_used_ = out.size
        return out

    def inverse_transform(self, y_transformed: np.ndarray,
                          steps: int,
                          original_tail: np.ndarray | None = None
                          ) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError(
                "Differencer.inverse_transform: call fit first")
        y_t = np.asarray(y_transformed, dtype=np.float64).ravel()
        if y_t.size != steps:
            raise ValueError(
                f"Differencer.inverse_transform: y_transformed.size="
                f"{y_t.size} != steps={steps}")
        out = y_t.copy()
        # v0.9.79a-fix2: closed-form binomial inverse.
        #   d-th diff:  y[t] = y_t[t] + sum_{k=1..d} (-1)^(k+1) C(d,k) y[t-k]
        #   D-th seasonal: y[t] = y_t[t] + sum_{k=1..D} (-1)^(k+1) C(D,k) y[t-k*s]
        # Inverse seasonal first (it was applied last in forward transform).
        if self.D > 0:
            s = self.seasonal_period
            tail = (self.tail_seasonal_ if original_tail is None
                    else np.asarray(original_tail, dtype=np.float64))
            need = self.D * s
            if tail.size < need:
                raise ValueError(
                    f"Differencer.inverse_transform: tail size="
                    f"{tail.size} < D*s={need}")
            y_tail = list(tail[-need:])
            result = []
            for v in out:
                y_new = float(v)
                for k in range(1, self.D + 1):
                    coef = ((-1) ** (k + 1)) * comb(self.D, k)
                    y_new += coef * y_tail[-k * s]
                result.append(y_new)
                y_tail.append(y_new)
                if len(y_tail) > need:
                    y_tail.pop(0)
            out = np.array(result, dtype=np.float64)
        # Inverse regular diff (out is now Delta^d y forecast).
        if self.lags > 0:
            d = self.lags
            tail = (self.tail_regular_ if original_tail is None
                    else np.asarray(original_tail, dtype=np.float64))
            if tail.size < d:
                raise ValueError(
                    f"Differencer.inverse_transform: tail size="
                    f"{tail.size} < lags={d}")
            y_tail = list(tail[-d:])
            result = []
            for v in out:
                y_new = float(v)
                for k in range(1, d + 1):
                    coef = ((-1) ** (k + 1)) * comb(d, k)
                    y_new += coef * y_tail[-k]
                result.append(y_new)
                y_tail.append(y_new)
                if len(y_tail) > d:
                    y_tail.pop(0)
            out = np.array(result, dtype=np.float64)
        return out

    def get_params(self) -> dict:
        return {"lags": self.lags,
                "seasonal_period": self.seasonal_period,
                "D": self.D}


# ── TransformedTargetForecaster ──────────────────────────────────

class TransformedTargetForecaster(ForecasterBase):
    """Pipeline: transformers (in order) then forecaster.

    At fit: y -> t1 -> t2 -> ... -> tN -> base.fit(tN_result)
    At predict: base.predict(steps) -> tN.inverse -> ... -> t1.inverse

    Parameters
    ----------
    transformers : list[BaseTransformer]
        Applied in list order to y. Inverted in reverse order at predict.
    forecaster : ForecasterBase
        Base model fit on the fully-transformed series.
    """

    def __init__(self, transformers: list[BaseTransformer],
                 forecaster: ForecasterBase):
        super().__init__()
        if not transformers:
            raise ValueError(
                "TransformedTargetForecaster: at least one transformer")
        if forecaster is None:
            raise ValueError(
                "TransformedTargetForecaster: forecaster is required")
        self.transformers = list(transformers)
        self.forecaster = forecaster
        # Forward capability: pipeline handles trend/seasonality only if
        # EITHER the base forecaster does OR a transformer covers it.
        # Simplification: inherit base forecaster's flags; differencing
        # makes any model "handles trend", so if first transformer is
        # Differencer with lags>0 -> handles_trend = True. Similarly for
        # seasonality.
        # v0.9.79a-fix1: capability flags are optional on models
        # (Theta/OTM/AutoTheta/ETS/AutoETS/SARIMA have them;
        # NaiveForecaster and legacy models don't). Default to False
        # via getattr.
        self.handles_trend = (
            getattr(forecaster, "handles_trend", False)
            or any(isinstance(t, Differencer) and t.lags > 0
                   for t in self.transformers))
        self.handles_seasonality = (
            getattr(forecaster, "handles_seasonality", False)
            or any(isinstance(t, Differencer) and t.D > 0
                   for t in self.transformers))
        self.handles_multi_seasonality = getattr(
            forecaster, "handles_multi_seasonality", False)

    def fit(self, df, target_col, *, time_col=None, group_col=None):
        _check_df(df, target_col, time_col, group_col)
        y = _extract_series(df, target_col, time_col=time_col)
        y_current = y.copy()
        for t in self.transformers:
            t.fit(y_current)
            y_current = t.transform(y_current)
        # Build a CadwDataFrame of the transformed series for the base
        import pyarrow as pa
        from cadw import CadwDataFrame
        transformed_df = CadwDataFrame.from_table(
            pa.table({target_col: pa.array(y_current)}))
        self.forecaster.fit(transformed_df, target_col,
                            time_col=None, group_col=None)
        self.y_ = y
        self._is_fitted = True
        return self

    def predict(self, steps):
        out = self.forecaster.predict(steps)
        for t in reversed(self.transformers):
            out = t.inverse_transform(out, steps)
        return out

    def predict_in_sample(self):
        out = self.forecaster.predict_in_sample()
        # In-sample: inverse_transform with `steps = len(out)`; no
        # original_tail needed because state from fit is available.
        for t in reversed(self.transformers):
            out = t.inverse_transform(out, out.size)
        return out

    def prediction_interval(self, steps, level=0.95):
        # Approximate: run PI through forecaster, then inverse_transform
        # endpoints. Exact interval transform only for linear inverters
        # (differencing is linear); STL inverse is additive so endpoints
        # stay valid.
        if hasattr(self.forecaster, "prediction_interval"):
            lo, hi = self.forecaster.prediction_interval(steps, level)
        else:
            f = self.forecaster.predict(steps)
            lo, hi = f, f
        for t in reversed(self.transformers):
            lo = t.inverse_transform(lo, steps)
            hi = t.inverse_transform(hi, steps)
        return lo, hi

    def state_nbytes(self):
        base = self.forecaster.state_nbytes()
        for t in self.transformers:
            # Conservative: no exact size available
            base += 1024
        return int(base)

    def get_params(self) -> dict:
        return {
            "transformers": [t.get_params() for t in self.transformers],
            "forecaster": type(self.forecaster).__name__,
        }

