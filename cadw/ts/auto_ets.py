"""AutoETSForecaster — DEPRECATED (v0.9.190).

Use ``ETSForecaster(params="auto")`` instead. Per DEC-068 §5 / DEC-073:
batch-family merge. Streaming separate (v0.9.192+).

Kept as backward-compat shim; removed in v1.0.0.
"""
from __future__ import annotations

import warnings

from cadw.ts.ets import ETSForecaster


class AutoETSForecaster(ETSForecaster):
    """DEPRECATED: use ETSForecaster(params="auto")."""

    def __init__(self, seasonal_period: int = 1, *,
                 warn_slow_fit: bool = True,
                 slow_fit_threshold_s: float = 30.0):
        warnings.warn(
            "AutoETSForecaster is deprecated since v0.9.190; "
            "use ETSForecaster(params='auto'). "
            "Will be removed in v1.0.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            seasonal_period=seasonal_period,
            params="auto",
            warn_slow_fit=warn_slow_fit,
            slow_fit_threshold_s=slow_fit_threshold_s,
        )


__all__ = ["AutoETSForecaster"]
