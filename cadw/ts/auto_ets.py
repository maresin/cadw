"""AutoETS: AICc-selected ETS model (v0.9.76b).

Wrapper over ETSForecaster that evaluates all valid ETS configurations
(trend x season) and picks the one with minimum AICc.

Configurations evaluated (max 9):
  trend in  {N (none), A (additive), Ad (damped additive)}
  season in {N (none), A (additive), M (multiplicative)}

Skipped configurations:
  - any seasonal variant if n < 2 * seasonal_period
  - multiplicative seasonal if any y <= 0 (log undefined)
  - already-invalid combinations (e.g. damped without trend)

AICc = n * ln(SSE / n) + 2k + 2k(k+1) / (n - k - 1)

where k = #optimised parameters:
  1 (sigma) + 1 (alpha) + 1 if trend + 1 if season + 1 if damped
Seasonal indices (s-1 values) are treated as initial state, not
optimised, so are NOT counted in k (statsmodels convention — matches
sm.tsa.ExponentialSmoothing).

Fallback: if no ETS config is valid (all skipped or all raise), the
wrapper falls back to NaiveForecaster(strategy='last') so it always
returns a forecaster. A warning is emitted.

Reference: Hyndman, Koehler, Snyder, Grose (2002) "A state space
framework for automatic forecasting using exponential smoothing";
statsforecast AutoETS / statsmodels ExponentialSmoothing.

Pure numpy (via ETSForecaster); no sklearn/scipy/torch imports in this
module (AST-guard test_no_bench_deps_in_core.py).
"""
from __future__ import annotations

import time
import warnings

import numpy as np

from cadw.ts.ets import ETSForecaster
from cadw.ts.models import (
    ForecasterBase, NaiveForecaster, _check_df, _extract_series,
)


class AutoETSForecaster(ForecasterBase):
    """AICc-selected ETS.

    Attributes after fit
    --------------------
    best_model_ : ETSForecaster or NaiveForecaster
        Selected model.
    selected_config_ : dict
        {'label', 'trend', 'seasonal', 'damped'} for ETS; {'fallback': True}
        for Naive.
    aicc_scores_ : list[dict]
        Per-config AICc: {'label', 'trend', 'seasonal', 'damped',
        'aicc', 'error' (optional)}.
    best_aicc_ : float
        AICc of the selected model (inf for fallback).
    fit_duration_s_ : float
        Wall-clock fit time.
    """

    handles_trend: bool = True
    handles_seasonality: bool = True
    handles_multi_seasonality: bool = False

    def __init__(self, seasonal_period: int = 1, *,
                 warn_slow_fit: bool = True,
                 slow_fit_threshold_s: float = 30.0):
        super().__init__()
        if seasonal_period < 1:
            raise ValueError(
                "AutoETSForecaster: seasonal_period must be >= 1")
        self.seasonal_period = int(seasonal_period)
        self.warn_slow_fit = bool(warn_slow_fit)
        self.slow_fit_threshold_s = float(slow_fit_threshold_s)
        self.best_model_ = None
        self.selected_config_ = None
        self.aicc_scores_ = []
        self.best_aicc_ = float("inf")
        self.fit_duration_s_ = 0.0
        self.y_ = None

    # ── Config enumeration ────────────────────────────────────────

    def _enumerate_configs(self, y: np.ndarray, n: int) -> list[dict]:
        s = self.seasonal_period
        has_season = s > 1 and n >= 2 * s
        has_positive = bool((y > 0).all()) if has_season else True

        trend_opts = [
            ("N",  False, False),   # no trend
            ("A",  True,  False),   # additive trend
            ("Ad", True,  True),    # damped additive trend
        ]
        if has_season:
            if has_positive:
                season_opts = [
                    ("N", "additive"),        # season formula: none (s>1 but
                                              # grid override; see note)
                    ("A", "additive"),        # additive season
                    ("M", "multiplicative"),  # multiplicative season
                ]
            else:
                season_opts = [
                    ("N", "additive"),
                    ("A", "additive"),
                ]
        else:
            season_opts = [("N", "additive")]

        # NOTE on (trend, season, s) combinations:
        #   - ETS(A,N,N) with s==1 -> ETSForecaster SES path (alpha-only)
        #   - ETS(A,N,A) with s>1 -> seasonal-without-trend (still valid)
        #   - season_label 'N' with s>1 means: use additive formula but
        #     d/s smaller; effectively passthrough. ETSForecaster(s=1) case.

        configs = []
        for tl, trend, damped in trend_opts:
            if not trend and damped:
                continue  # impossible
            for sl, season_str in season_opts:
                configs.append({
                    "label": f"ETS(A,{tl},{sl})",
                    "trend": trend,
                    "seasonal": season_str,
                    "damped": damped,
                    "trend_label": tl,
                    "season_label": sl,
                    "has_season": sl != "N",
                })
        return configs

    # ── AICc ──────────────────────────────────────────────────────

    def _aicc(self, y: np.ndarray, model: ETSForecaster,
              has_season: bool) -> float:
        n = y.size
        resid = y - model.fitted_
        sse = float(np.sum(resid * resid))
        sse = max(sse, 1e-12)
        k = 1  # sigma^2
        k += 1  # alpha
        if model.trend:
            k += 1  # beta
        if has_season:
            k += 1  # gamma
        if model.damped:
            k += 1  # phi
        # Seasonal indices (s-1) are initial state, not optimised;
        # excluded from k (statsmodels convention).
        aic = n * np.log(sse / n) + 2.0 * k
        denom = n - k - 1
        if denom > 0:
            aicc = aic + 2.0 * k * (k + 1) / denom
        else:
            aicc = float("inf")
        return float(aicc)

    # ── fit ───────────────────────────────────────────────────────

    def fit(self, df, target_col, *, time_col=None, group_col=None):
        _check_df(df, target_col, time_col, group_col)
        y = _extract_series(df, target_col, time_col=time_col)
        if y.size < 4:
            raise ValueError(
                "AutoETSForecaster: need at least 4 observations")
        if np.any(~np.isfinite(y)):
            raise ValueError(
                "AutoETSForecaster: series contains non-finite values")

        n = y.size
        configs = self._enumerate_configs(y, n)
        if not configs:
            raise ValueError(
                "AutoETSForecaster: no valid ETS configs for this data "
                f"(n={n}, s={self.seasonal_period})")

        t_start = time.perf_counter()
        best_aicc = float("inf")
        best_cfg = None
        best_model = None
        aicc_scores = []

        for cfg in configs:
            entry = {
                "label": cfg["label"],
                "trend": cfg["trend_label"],
                "seasonal": cfg["season_label"],
                "damped": cfg["damped"],
            }
            try:
                # v0.9.76b-fix2: for season_label='N', pass
                # seasonal_period=1 (no seasonality), not self.seasonal_period.
                # Old code passed self.seasonal_period with seasonal='additive'
                # for BOTH 'N' and 'A' labels, so ETS(A,N,N) was actually a
                # seasonal model with a small k penalty — AICc preferred it
                # on seasonal data, defeating auto-selection.
                cfg_s = 1 if cfg["season_label"] == "N" else self.seasonal_period
                m = ETSForecaster(
                    seasonal_period=cfg_s,
                    seasonal=cfg["seasonal"],
                    trend=cfg["trend"],
                    damped=cfg["damped"],
                )
                m.fit(df, target_col, time_col=time_col,
                      group_col=group_col)
                aicc = self._aicc(y, m, cfg["has_season"])
                entry["aicc"] = aicc
                if aicc < best_aicc:
                    best_aicc = aicc
                    best_cfg = cfg
                    best_model = m
            except Exception as e:
                entry["aicc"] = None
                entry["error"] = f"{type(e).__name__}: {e}"
            aicc_scores.append(entry)

        elapsed = time.perf_counter() - t_start

        if (self.warn_slow_fit
                and elapsed > self.slow_fit_threshold_s):
            warnings.warn(
                f"AutoETSForecaster: fit took {elapsed:.1f}s "
                f"(> {self.slow_fit_threshold_s}s threshold). "
                f"Consider fewer configs or shorter series.",
                stacklevel=2,
            )

        if best_model is None:
            warnings.warn(
                "AutoETSForecaster: no ETS config valid; "
                "falling back to NaiveForecaster(strategy='last').",
                stacklevel=2,
            )
            best_model = NaiveForecaster(strategy="last")
            best_model.fit(df, target_col, time_col=time_col,
                           group_col=group_col)
            selected = {"label": "fallback:NaiveForecaster(last)",
                        "trend": "N",
                        "seasonal": "N",
                        "damped": False,
                        "fallback": True}
            best_aicc = float("inf")
        else:
            # v0.9.76b-fix1: normalize selected_config_ keys to use
            # human-readable labels ('N'|'A'|'M'|'Ad'), not ETS
            # constructor params ('additive'|'multiplicative').
            selected = {
                "label": best_cfg["label"],
                "trend": best_cfg["trend_label"],
                "seasonal": best_cfg["season_label"],
                "damped": best_cfg["damped"],
            }

        self.best_model_ = best_model
        self.selected_config_ = selected
        self.aicc_scores_ = aicc_scores
        self.best_aicc_ = best_aicc
        self.fit_duration_s_ = elapsed
        self.y_ = y
        self._is_fitted = True
        return self

    # ── predict / PI / state ─────────────────────────────────────

    def predict(self, steps):
        return self.best_model_.predict(steps)

    def predict_in_sample(self):
        return self.best_model_.predict_in_sample()

    def prediction_interval(self, steps, level=0.95):
        if hasattr(self.best_model_, "prediction_interval"):
            return self.best_model_.prediction_interval(steps, level)
        # Fallback: no interval from NaiveForecaster; return point twice.
        f = self.predict(steps)
        return f, f

    def state_nbytes(self):
        if self.best_model_ is None:
            return 0
        return int(self.best_model_.state_nbytes())

