"""cadw.ts.auto_theta — DEPRECATED shim (v0.9.194).

Merged into ThetaForecaster(params="auto"). Kept for backward compat;
removed in v1.0.0. Re-exports helpers from cadw.ts._theta_common so
existing imports (`from cadw.ts.auto_theta import _MultiplicativeWrapper`)
keep working.
"""
from __future__ import annotations

import warnings

from cadw.ts._theta_common import (  # noqa: F401
    _naive_seasonal_mae,
    _mase,
    _log_transform_batches,
    _mk_log_df,
    _MultiplicativeWrapper,
)
from cadw.ts.theta import ThetaForecaster


class AutoThetaForecaster(ThetaForecaster):
    """DEPRECATED in v0.9.194.

    Alias for ThetaForecaster(params="auto"). Emits DeprecationWarning
    on construction. Removed in v1.0.0.
    """

    def __init__(self, seasonal_period: int = 1, *,
                 val_steps=None, n_iter: int = 2):
        warnings.warn(
            "AutoThetaForecaster is deprecated in v0.9.194; "
            "use ThetaForecaster(params='auto'). Removed in v1.0.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            seasonal_period=seasonal_period, n_iter=n_iter,
            params="auto", val_steps=val_steps,
        )
