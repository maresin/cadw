"""CADW time-series toolkit (v0.9.24).

See README.md Time series section and DECISIONS.md for design notes.
"""
from .features import (
    calendar_features, cyclical_features, diff_features,
    fourier_features, holiday_features, lag_features, lead_features,
    make_features,
)
from .stats import rolling_crosscorr, rolling_stats
from .models import (
    ForecasterBase, GBDTForecaster, NaiveForecaster,
    SARIMAForecaster, get_forecaster,
)
from .backtest import backtest_forecaster
from .thf import TemporalHierarchy
from .ets import ETSForecaster
from .theta import ThetaForecaster
from .var import VARModel

__all__ = sorted([
    # features
    "calendar_features", "cyclical_features", "diff_features",
    "fourier_features", "holiday_features", "lag_features",
    "lead_features", "make_features",
    # window stats
    "rolling_crosscorr", "rolling_stats",
    # forecasters
    "ForecasterBase", "NaiveForecaster", "SARIMAForecaster",
    "GBDTForecaster", "get_forecaster",
    # backtesting
    "backtest_forecaster",
    # temporal hierarchy
    "TemporalHierarchy",
    "ETSForecaster", "ThetaForecaster", "VARModel",
])
