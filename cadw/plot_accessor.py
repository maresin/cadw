"""Plot accessor for CadwDataFrame (v0.9.0c).

`df.plot.hist(col)` — matplotlib default.
`df.plot.plotly.hist(col)` — plotly explicit.
`df.plot.seaborn.hist(col)` — seaborn explicit.

All methods delegate to `cadw.viz.*` adapters (aggregate-first: adapters
receive small pre-computed aggregates, not raw data — see §7.4 discipline).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cadw.dataframe import CadwDataFrame


class _MatplotlibAccessor:
    """`df.plot.*` — matplotlib backend (default)."""

    __slots__ = ("_df",)

    def __init__(self, df: "CadwDataFrame") -> None:
        self._df = df

    def hist(self, column: str, bins: int = 50, **kw):
        from cadw.viz.matplotlib_adapter import hist_plot
        return hist_plot(self._df, column, bins=bins, **kw)

    def hist2d(self, x: str, y: str, bins=(200, 200), **kw):
        from cadw.viz.matplotlib_adapter import hist2d_plot
        return hist2d_plot(self._df, x, y, bins=bins, **kw)

    def line(self, by: str, agg_col: str, agg: str = "mean", **kw):
        from cadw.viz.matplotlib_adapter import line_plot
        return line_plot(self._df, by, agg_col, agg=agg, **kw)

    def box(self, column: str, **kw):
        from cadw.viz.matplotlib_adapter import box_plot
        return box_plot(self._df, column, **kw)

    def scatter(self, x: str, y: str, n: int = 10_000, **kw):
        from cadw.viz.matplotlib_adapter import scatter
        return scatter(self._df, x, y, n=n, **kw)

    def hexbin(self, x: str, y: str, gridsize: int = 30,
               n: int = 10_000, **kw):
        from cadw.viz.matplotlib_adapter import hexbin_plot
        return hexbin_plot(self._df, x, y, gridsize=gridsize, n=n, **kw)


class _PlotlyAccessor:
    """`df.plot.plotly.*` — plotly backend."""

    __slots__ = ("_df",)

    def __init__(self, df: "CadwDataFrame") -> None:
        self._df = df

    def hist(self, column: str, bins: int = 50, **kw):
        from cadw.viz.plotly_adapter import hist_plot
        return hist_plot(self._df, column, bins=bins, **kw)

    def hist2d(self, x: str, y: str, bins=(200, 200), **kw):
        from cadw.viz.plotly_adapter import hist2d_plot
        return hist2d_plot(self._df, x, y, bins=bins, **kw)

    def line(self, by: str, agg_col: str, agg: str = "mean", **kw):
        from cadw.viz.plotly_adapter import line_plot
        return line_plot(self._df, by, agg_col, agg=agg, **kw)

    def box(self, column: str, **kw):
        from cadw.viz.plotly_adapter import box_plot
        return box_plot(self._df, column, **kw)

    def scatter(self, x: str, y: str, n: int = 10_000, **kw):
        from cadw.viz.plotly_adapter import scatter
        return scatter(self._df, x, y, n=n, **kw)


class _SeabornAccessor:
    """`df.plot.seaborn.*` — seaborn backend."""

    __slots__ = ("_df",)

    def __init__(self, df: "CadwDataFrame") -> None:
        self._df = df

    def hist(self, column: str, bins: int = 50, **kw):
        from cadw.viz.seaborn_adapter import hist_plot
        return hist_plot(self._df, column, bins=bins, **kw)

    def heatmap(self, x: str, y: str, bins=(50, 50), **kw):
        from cadw.viz.seaborn_adapter import heatmap_plot
        return heatmap_plot(self._df, x, y, bins=bins, **kw)

    def line(self, by: str, agg_col: str, agg: str = "mean", **kw):
        from cadw.viz.seaborn_adapter import line_plot
        return line_plot(self._df, by, agg_col, agg=agg, **kw)

    def box(self, column: str, n: int = 1000, **kw):
        from cadw.viz.seaborn_adapter import box_plot
        return box_plot(self._df, column, n=n, **kw)

    def scatter(self, x: str, y: str, n: int = 1000, **kw):
        from cadw.viz.seaborn_adapter import scatter
        return scatter(self._df, x, y, n=n, **kw)


class _PlotAccessor(_MatplotlibAccessor):
    """`df.plot.*` — matplotlib by default; `.plotly`, `.seaborn` explicit."""

    __slots__ = ("_df",)

    @property
    def plotly(self) -> _PlotlyAccessor:
        return _PlotlyAccessor(self._df)

    @property
    def seaborn(self) -> _SeabornAccessor:
        return _SeabornAccessor(self._df)
