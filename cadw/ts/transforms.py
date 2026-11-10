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

from cadw.ts._decomposition import stl_decompose
from cadw.ops.stats import acf as _acf_fn


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

    def state_nbytes(self) -> int:
        """Persistent-state RAM after fit (bytes). Override in
        subclasses. Base returns 256 (header)."""
        return 256

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

    def state_nbytes(self) -> int:
        total = 256
        if self.tail_regular_ is not None:
            total += int(self.tail_regular_.nbytes)
        if self.tail_seasonal_ is not None:
            total += int(self.tail_seasonal_.nbytes)
        return total

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
        # v0.9.129: guard before O(n) transforms + base forecaster fit.
        # Peak = (3 + n_transformers) * n * 8 + 4096. bounded_by_domain
        # n <= 1e6.
        _mg = getattr(df, "_memory_guard", None)
        if _mg is not None:
            _mg.check_state(
                (3 + 8 * len(self.transformers)) * y.size * 8 + 4096,
                name="ttf.fit",
            )
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

    def predict_in_sample(self, df=None):
        """v0.9.182: accepts df for OOC inner forecasters (Naive v0.9.182+).

        - df=None: legacy no-df call (inner forecaster must support it).
        - df=given: passes df to inner forecaster if its
          predict_in_sample accepts df (signature-inspected); legacy
          inner uses no-df path.
        """
        import inspect
        accepts_df = False
        if df is not None:
            try:
                sig = inspect.signature(
                    self.forecaster.predict_in_sample)
                accepts_df = "df" in sig.parameters
            except (ValueError, TypeError):
                accepts_df = False
        if accepts_df:
            out = self.forecaster.predict_in_sample(df)
        else:
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
        # v0.9.129: per-transformer exact state (AP22 fix: was 1024
        # invented constant).
        base = (self.forecaster.state_nbytes()
                if hasattr(self.forecaster, "state_nbytes") else 4096)
        for t in self.transformers:
            base += t.state_nbytes()
        return int(base + 256)

    def get_params(self) -> dict:
        return {
            "transformers": [t.get_params() for t in self.transformers],
            "forecaster": type(self.forecaster).__name__,
        }

# ── STLTransformer ────────────────────────────────────────────────

class STLTransformer(BaseTransformer):
    """Remove seasonal component via STL; add back at forecast.

    fit(y) learns trend/seasonal/remainder via stl_decompose(y, period).
    transform(y) = y - seasonal.
    inverse_transform(forecast, steps) = forecast + seasonal_forecast,
    where seasonal_forecast is the last cycle of learned seasonal
    pattern, repeated.

    After transform the base forecaster sees y - seasonal (trend +
    remainder). Trend is still present; combine with Differencer for
    full stationarization.
    """

    def __init__(self, period: int, n_iter: int = 2):
        super().__init__()
        if period < 2:
            raise ValueError("STLTransformer: period must be >= 2")
        self.period = int(period)
        self.n_iter = int(n_iter)
        self.seasonal_ = None
        self.trend_ = None
        self.remainder_ = None

    def state_nbytes(self) -> int:
        if self.seasonal_ is None:
            return 256
        return int(self.seasonal_.nbytes + self.trend_.nbytes
                   + self.remainder_.nbytes + 256)

    def fit(self, y: np.ndarray) -> "STLTransformer":
        y = np.asarray(y, dtype=np.float64).ravel()
        if y.size < 2 * self.period:
            raise ValueError(
                f"STLTransformer: need at least 2*period="
                f"{2 * self.period} obs, got {y.size}")
        result = stl_decompose(y, period=self.period,
                                n_iter=self.n_iter)
        self.seasonal_ = result.seasonal
        self.trend_ = result.trend
        self.remainder_ = result.remainder
        self._is_fitted = True
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("STLTransformer.transform: call fit first")
        y = np.asarray(y, dtype=np.float64).ravel()
        if y.size != self.seasonal_.size:
            raise ValueError(
                f"STLTransformer.transform: y.size={y.size} != "
                f"seasonal.size={self.seasonal_.size}")
        return y - self.seasonal_

    def inverse_transform(self, y_transformed: np.ndarray,
                          steps: int,
                          original_tail: np.ndarray | None = None
                          ) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError(
                "STLTransformer.inverse_transform: call fit first")
        y_t = np.asarray(y_transformed, dtype=np.float64).ravel()
        if y_t.size != steps:
            raise ValueError(
                f"STLTransformer.inverse_transform: y_transformed.size="
                f"{y_t.size} != steps={steps}")
        s = self.period
        # v0.9.79b-fix1: if steps == full seasonal length, this is
        # in-sample inverse (roundtrip) — use full seasonal array.
        # Forecast (steps < n or steps > n) uses last cycle tiled.
        if steps == self.seasonal_.size:
            return y_t + self.seasonal_
        last_cycle = self.seasonal_[-s:]
        idx = np.arange(steps) % s
        return y_t + last_cycle[idx]

    def get_params(self) -> dict:
        return {"period": self.period, "n_iter": self.n_iter}


# ── MSTLTransformer ───────────────────────────────────────────────

class MSTLTransformer(BaseTransformer):
    """Multi-seasonal STL: iteratively remove each seasonal period.

    fit(y): for each period s in `periods` (ascending), run
    stl_decompose on the current residual, subtract seasonal, continue.
    transform(y) = y - sum(seasonals).
    inverse_transform(forecast) = forecast + sum(last-cycle patterns).
    """

    def __init__(self, periods: list, n_iter: int = 2):
        super().__init__()
        if not periods or len(periods) < 1:
            raise ValueError(
                "MSTLTransformer: at least one period required")
        periods_sorted = sorted(int(p) for p in periods)
        if periods_sorted[0] < 2:
            raise ValueError("MSTLTransformer: periods must be >= 2")
        self.periods = periods_sorted
        self.n_iter = int(n_iter)
        self.seasonals_ = None
        self.deseasonalized_ = None

    def state_nbytes(self) -> int:
        if self.seasonals_ is None:
            return 256
        total = 256 + int(self.deseasonalized_.nbytes)
        for s in self.seasonals_:
            total += int(s.nbytes)
        return total

    def fit(self, y: np.ndarray) -> "MSTLTransformer":
        y = np.asarray(y, dtype=np.float64).ravel()
        s_max = max(self.periods)
        if y.size < 2 * s_max:
            raise ValueError(
                f"MSTLTransformer: need at least 2*max_period="
                f"{2 * s_max} obs, got {y.size}")
        current = y.copy()
        seasonals = []
        for s in self.periods:
            result = stl_decompose(current, period=s,
                                    n_iter=self.n_iter)
            seasonals.append(result.seasonal)
            current = current - result.seasonal
        self.seasonals_ = seasonals
        self.deseasonalized_ = current
        self._is_fitted = True
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError(
                "MSTLTransformer.transform: call fit first")
        y = np.asarray(y, dtype=np.float64).ravel()
        out = y.copy()
        for seasonal in self.seasonals_:
            out = out - seasonal
        return out

    def inverse_transform(self, y_transformed: np.ndarray,
                          steps: int,
                          original_tail: np.ndarray | None = None
                          ) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError(
                "MSTLTransformer.inverse_transform: call fit first")
        y_t = np.asarray(y_transformed, dtype=np.float64).ravel()
        if y_t.size != steps:
            raise ValueError(
                f"MSTLTransformer.inverse_transform: y_transformed.size="
                f"{y_t.size} != steps={steps}")
        out = y_t.copy()
        # v0.9.79b-fix1: in-sample inverse (roundtrip) uses full seasonal
        # arrays; forecast uses last cycle tiled per period.
        in_sample = (len(self.seasonals_) > 0
                     and steps == self.seasonals_[0].size)
        for s, seasonal in zip(self.periods, self.seasonals_):
            if seasonal.size < s:
                continue
            if in_sample:
                out = out + seasonal
            else:
                last_cycle = seasonal[-s:]
                idx = np.arange(steps) % s
                out = out + last_cycle[idx]
        return out

    def get_params(self) -> dict:
        return {"periods": list(self.periods), "n_iter": self.n_iter}


# ── AutoStationarizer ────────────────────────────────────────────

class AutoStationarizer(Differencer):
    """Auto-select d (regular) and D (seasonal) differencing orders.

    Heuristics (fast; formal ADF/KPSS in backlog):
      d: iteratively check |acf(y, lag=1)|. If < acf_threshold,
         stop (d = current). Else diff, repeat; cap at max_d.
      D: after d regular diffs, compute STL seasonal strength
         F_S = max(0, 1 - Var(R) / Var(S + R)). If F_S > threshold,
         D = 1. Cap at max_D (currently supports D=0 or 1).

    Subclass of Differencer: sets self.lags, self.D in fit and
    delegates transform/inverse to parent.
    """

    def __init__(self, seasonal_period: int | None = None, *,
                 max_d: int = 2, max_D: int = 1,
                 acf_threshold: float = 0.8,
                 seasonal_strength_threshold: float = 0.5,
                 n_iter_stl: int = 2):
        if max_d < 0:
            raise ValueError(
                f"AutoStationarizer: max_d must be >= 0, got {max_d}")
        if max_D < 0:
            raise ValueError(
                f"AutoStationarizer: max_D must be >= 0, got {max_D}")
        super().__init__(lags=0, seasonal_period=seasonal_period, D=0)
        self._max_d = int(max_d)
        self._max_D = int(max_D)
        self._acf_threshold = float(acf_threshold)
        self._strength_threshold = float(seasonal_strength_threshold)
        self._n_iter_stl = int(n_iter_stl)
        self.chosen_d_ = 0
        self.chosen_D_ = 0
        self.seasonal_strength_ = None

    def state_nbytes(self) -> int:
        return super().state_nbytes() + 8  # seasonal_strength_ scalar

    def _acf(self, y: np.ndarray, lag: int) -> float:
        return _acf_fn(y, lag)

    def _choose_d(self, y: np.ndarray) -> int:
        cur = y.copy()
        for d in range(self._max_d):
            if cur.size < 20:
                return d
            if abs(self._acf(cur, 1)) < self._acf_threshold:
                return d
            cur = np.diff(cur)
        return self._max_d

    def _choose_D(self, y_after_d: np.ndarray) -> int:
        s = self.seasonal_period
        if s is None or s < 2:
            return 0
        if self._max_D < 1:
            return 0
        if y_after_d.size < 2 * s:
            return 0
        try:
            result = stl_decompose(y_after_d, period=s,
                                    n_iter=self._n_iter_stl)
        except ValueError:
            return 0
        var_sr = float(np.var(result.seasonal + result.remainder))
        if var_sr < 1e-12:
            return 0
        strength = max(0.0, 1.0 - float(np.var(result.remainder)) / var_sr)
        self.seasonal_strength_ = strength
        return 1 if strength > self._strength_threshold else 0

    def fit(self, y: np.ndarray) -> "AutoStationarizer":
        y = np.asarray(y, dtype=np.float64).ravel()
        if y.size < 4:
            raise ValueError(
                "AutoStationarizer.fit: need at least 4 observations")
        chosen_d = self._choose_d(y)
        y_after_d = y.copy()
        for _ in range(chosen_d):
            y_after_d = np.diff(y_after_d)
        chosen_D = self._choose_D(y_after_d)
        self.lags = int(chosen_d)
        self.D = int(chosen_D)
        self.chosen_d_ = int(chosen_d)
        self.chosen_D_ = int(chosen_D)
        return super().fit(y)

    def get_params(self) -> dict:
        return {
            "seasonal_period": self.seasonal_period,
            "max_d": self._max_d,
            "max_D": self._max_D,
            "acf_threshold": self._acf_threshold,
            "seasonal_strength_threshold": self._strength_threshold,
            "chosen_d": self.chosen_d_,
            "chosen_D": self.chosen_D_,
        }

# ── AutoForecastPipeline ─────────────────────────────────────────

class AutoForecastPipeline(ForecasterBase):
    """Capability-based per-model transform pipeline.

    Wraps any ForecasterBase. Reads its capability flags
    (handles_trend, handles_seasonality, handles_multi_seasonality);
    applies transforms only where the underlying model cannot handle
    that component natively.

    Transform ordering (v0.9.80):
      1. Seasonal removal (if model cannot handle): STLTransformer
         (single period) or MSTLTransformer (multiple periods).
      2. Trend removal (if model cannot handle): AutoStationarizer
         (differencing with auto d; seasonal_period=None because
         seasonality was handled by STL above).

    Forward: y -> [STL: y - S] -> [AutoStat: Delta^d(y - S)]
    Inverse: forecast -> [AutoStat: cumsum] -> [STL: + S]
    Order matches literature convention (STL+ARIMA: STL first,
    differencing on deseasonalized series).

    If the base model already handles both trend and seasonality
    (e.g. SARIMA, ETS, AutoETS, AutoTheta), the pipeline stays
    "naked" (no transforms added) and delegates directly to the base.

    Parameters
    ----------
    forecaster : ForecasterBase
    seasonal_periods : list[int] | None
        Seasonal period(s). If None/empty and base needs seasonality,
        STL is not added (differencing still applied for trend). User
        may accept degraded seasonality or set these explicitly.
    max_d : int, default=2
    max_D : int, default=1  (only used by AutoStationarizer's D
        heuristic; here AutoStationarizer is created with
        seasonal_period=None so D=0)
    acf_threshold, seasonal_strength_threshold : float
        Passed to AutoStationarizer.
    """

    def __init__(self, forecaster: ForecasterBase, *,
                 seasonal_periods: list | None = None,
                 max_d: int = 2, max_D: int = 1,
                 acf_threshold: float = 0.8,
                 seasonal_strength_threshold: float = 0.5):
        super().__init__()
        if forecaster is None:
            raise ValueError(
                "AutoForecastPipeline: forecaster is required")
        self.forecaster = forecaster
        self.seasonal_periods = (list(seasonal_periods)
                                 if seasonal_periods is not None else [])
        self.max_d = int(max_d)
        self.max_D = int(max_D)
        self.acf_threshold = float(acf_threshold)
        self.seasonal_strength_threshold = float(
            seasonal_strength_threshold)

        transformers = self._build_transformers()
        self.transformers = transformers
        self._pipeline = (
            TransformedTargetForecaster(transformers, forecaster)
            if transformers else None)

        # Capability flags: base handles OR transformer covers
        self.handles_trend = (
            getattr(forecaster, "handles_trend", False)
            or any(isinstance(t, Differencer) and t.lags > 0
                   for t in transformers)
            or any(isinstance(t, AutoStationarizer)
                   for t in transformers))
        self.handles_seasonality = (
            getattr(forecaster, "handles_seasonality", False)
            or any(isinstance(t, STLTransformer) for t in transformers)
            or any(isinstance(t, MSTLTransformer) for t in transformers))
        self.handles_multi_seasonality = (
            getattr(forecaster, "handles_multi_seasonality", False)
            or any(isinstance(t, MSTLTransformer)
                   for t in transformers))

    def _build_transformers(self) -> list:
        transformers = []
        needs_seasonal = not getattr(self.forecaster,
                                     "handles_seasonality", False)
        needs_multi = not getattr(self.forecaster,
                                  "handles_multi_seasonality", False)
        needs_trend = not getattr(self.forecaster,
                                  "handles_trend", False)

        # 1. Seasonal (STL / MSTL)
        if needs_seasonal and self.seasonal_periods:
            if len(self.seasonal_periods) >= 2 and needs_multi:
                transformers.append(MSTLTransformer(self.seasonal_periods))
            else:
                # If multi-period requested but base handles multi,
                # just use first period (base handles rest natively).
                transformers.append(
                    STLTransformer(self.seasonal_periods[0]))
        # 2. Trend (differencing on deseasonalized)
        if needs_trend:
            transformers.append(AutoStationarizer(
                seasonal_period=None,  # seasonality handled by STL
                max_d=self.max_d, max_D=0,
                acf_threshold=self.acf_threshold,
                seasonal_strength_threshold=(
                    self.seasonal_strength_threshold)))
        return transformers

    def fit(self, df, target_col, *, time_col=None, group_col=None):
        if self._pipeline is not None:
            self._pipeline.fit(df, target_col,
                               time_col=time_col, group_col=group_col)
        else:
            self.forecaster.fit(df, target_col,
                                time_col=time_col, group_col=group_col)
        # Extract y for state/roundtrip.
        from cadw.ts.models import _extract_series
        self.y_ = _extract_series(df, target_col, time_col=time_col)
        self._is_fitted = True
        return self

    def predict(self, steps):
        if self._pipeline is not None:
            return self._pipeline.predict(steps)
        return self.forecaster.predict(steps)

    def predict_in_sample(self, df=None):
        """v0.9.182: accepts df, passes through to inner pipeline/forecaster.

        - df=None: legacy no-df call.
        - df=given: passes df to target (self._pipeline or self.forecaster)
          if its predict_in_sample accepts df (signature-inspected).
        """
        import inspect
        target = (self._pipeline if self._pipeline is not None
                  else self.forecaster)
        accepts_df = False
        if df is not None:
            try:
                sig = inspect.signature(target.predict_in_sample)
                accepts_df = "df" in sig.parameters
            except (ValueError, TypeError):
                accepts_df = False
        if accepts_df:
            return target.predict_in_sample(df)
        return target.predict_in_sample()

    def prediction_interval(self, steps, level=0.95):
        target = (self._pipeline if self._pipeline is not None
                  else self.forecaster)
        if hasattr(target, "prediction_interval"):
            return target.prediction_interval(steps, level)
        f = self.predict(steps)
        return f, f

    def state_nbytes(self):
        target = (self._pipeline if self._pipeline is not None
                  else self.forecaster)
        return int(target.state_nbytes())

    def get_params(self) -> dict:
        return {
            "forecaster": type(self.forecaster).__name__,
            "seasonal_periods": list(self.seasonal_periods),
            "n_transformers": len(self.transformers),
            "transformers": [t.get_params()
                             for t in self.transformers],
        }
