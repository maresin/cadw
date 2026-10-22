"""Time-series accessor (v0.9.24): df.ts.* thin wrappers.

All methods delegate directly to cadw.ts functions/classes; no logic
lives here. Registered as the `ts` property on CadwDataFrame, matching
the .dt/.str/.plot accessor pattern.
"""
from __future__ import annotations

from typing import Sequence


class _TsAccessor:
    """Pandas-style `df.ts.*` — thin wrappers over cadw.ts."""

    __slots__ = ("_df",)

    def __init__(self, parent) -> None:
        self._df = parent

    # ----- features -----
    def lag(self, column, lags=(1,), *, group=None, time_col=None, fill_value=None):
        from cadw.ts import lag_features
        return lag_features(
            self._df, column, lags,
            group_col=group, time_col=time_col, fill_value=fill_value,
        )

    def lead(self, column, leads=(1,), *, group=None, time_col=None, fill_value=None):
        from cadw.ts import lead_features
        return lead_features(
            self._df, column, leads,
            group_col=group, time_col=time_col, fill_value=fill_value,
        )

    def diff(self, column, lags=(1,), *, group=None, time_col=None,
             seasonal_periods=None):
        from cadw.ts import diff_features
        return diff_features(
            self._df, column, lags,
            group_col=group, time_col=time_col,
            seasonal_periods=seasonal_periods,
        )

    def calendar(self, ts_col, features=None):
        from cadw.ts import calendar_features
        kw = {} if features is None else {"features": features}
        return calendar_features(self._df, ts_col, **kw)

    def cyclical(self, ts_col, features=None):
        from cadw.ts import cyclical_features
        kw = {} if features is None else {"features": features}
        return cyclical_features(self._df, ts_col, **kw)

    def fourier(self, time_col, period, order, prefix=None):
        from cadw.ts import fourier_features
        return fourier_features(
            self._df, time_col, period, order, prefix=prefix,
        )

    def holiday(self, ts_col):
        from cadw.ts import holiday_features
        return holiday_features(self._df, ts_col)

    def make_features(self, **kwargs):
        from cadw.ts import make_features
        return make_features(self._df, **kwargs)

    # ----- window stats -----
    def rolling_features(self, column, windows, *, stats=("mean", "std"),
                         group=None, time_col=None, min_periods=None):
        from cadw.ts import rolling_stats
        return rolling_stats(
            self._df, column, windows, stats=stats,
            group_col=group, time_col=time_col, min_periods=min_periods,
        )

    def crosscorr(self, col_x, col_y, windows, *, lags=(0,),
                  group=None, time_col=None, min_periods=None):
        from cadw.ts import rolling_crosscorr
        return rolling_crosscorr(
            self._df, col_x, col_y, windows, lags=lags,
            group_col=group, time_col=time_col, min_periods=min_periods,
        )

    # ----- forecasters -----
    def fit_forecaster(self, forecaster, target_col, *, time_col=None,
                       group=None, **fit_kwargs):
        # forecaster: ForecasterBase instance or short name.
        from cadw.ts import get_forecaster
        if isinstance(forecaster, str):
            f = get_forecaster(forecaster)
        else:
            f = forecaster
        if group is not None:
            fit_kwargs.setdefault("group_col", group)
        return f.fit(self._df, target_col, time_col=time_col, **fit_kwargs)

    def backtest(self, forecaster, *, target_col, time_col, **kwargs):
        from cadw.ts import backtest_forecaster
        return backtest_forecaster(
            self._df, forecaster, target_col=target_col,
            time_col=time_col, **kwargs,
        )

    def fit_temporal_hierarchy(self, levels, forecaster_factory, *,
                               target_col, time_col=None, group=None,
                               reconciliation="optimal_ols", agg="sum"):
        from cadw.ts import TemporalHierarchy
        th = TemporalHierarchy(
            levels, forecaster_factory, reconciliation=reconciliation,
            agg=agg,
        )
        if group is not None:
            return th.fit(
                self._df, target_col, time_col=time_col, group_col=group,
            )
        return th.fit(self._df, target_col, time_col=time_col)


__all__ = ["_TsAccessor"]
