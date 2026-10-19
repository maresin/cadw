"""CADW viz layer (v0.7.4a): aggregate-first plotting.

Public API:
    cadw.viz.hist          — 1D histogram (Reduce -> counts, edges)
    cadw.viz.hist2d        — 2D histogram (Reduce -> counts_2d, edges)
    cadw.viz.box_stats     — {min, q1, median, q3, max} via quantile
    cadw.viz.line_agg      — groupby aggregate (K rows)
    cadw.viz.sample        — bounded df.head(n)

matplotlib adapter (requires matplotlib):
    from cadw.viz.matplotlib_adapter import (
        hist_plot, hist2d_plot, box_plot, line_plot, scatter,
        feature_importance,
    )

Rule: renderers never receive raw CadwDataFrame; only aggregates
from _aggregate. See docs in DECISIONS §13.4/92.
"""
from ._aggregate import (
    hist,
    hist2d,
    box_stats,
    line_agg,
    sample,
)

__all__ = [
    "hist", "hist2d", "box_stats", "line_agg", "sample",
]
