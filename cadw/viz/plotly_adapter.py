"""Plotly adapter (v0.7.4b).

Aggregate-first: hist/hist2d/line/box/feature_importance use precomputed
aggregates. Scatter auto-mode: Scattergl (<=n, bounded) / Heatmap (>n).

Requires: plotly>=5.18 (lazy import).
"""
from __future__ import annotations

import numpy as np

from cadw.viz import _aggregate


def _go():
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        raise ImportError(
            "plotly is required for cadw.viz.plotly_adapter; "
            "install with: uv sync --group viz"
        ) from e
    return go


def _to_np(arr):
    a = arr.to_numpy(zero_copy_only=False)
    if a.dtype == object:
        return np.array(
            [np.nan if v is None else float(v) for v in a],
            dtype=np.float64,
        )
    return a.astype(np.float64, copy=False)


def _sample_pair(df, x: str, y: str, n: int):
    """Pull (x, y) numpy arrays for at most n rows via _aggregate.sample."""
    smp = _aggregate.sample(df, n=n)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for b in smp._iter_filtered_batches():
        xs.append(_to_np(b.column(x)))
        ys.append(_to_np(b.column(y)))
    if not xs:
        return np.empty(0), np.empty(0)
    return np.concatenate(xs), np.concatenate(ys)


def hist_plot(df, column: str, bins: int = 50, **kwargs):
    """1D histogram as go.Figure (Bar)."""
    go = _go()
    counts, edges = _aggregate.hist(df, column, bins=bins)
    widths = np.diff(edges)
    fig = go.Figure(go.Bar(
        x=edges[:-1], y=counts, width=widths,
        marker_color=kwargs.pop("color", "steelblue"),
        name=column,
    ))
    fig.update_layout(
        title=f"hist({column}, bins={bins})",
        xaxis_title=column, yaxis_title="count",
        **kwargs,
    )
    return fig


def hist2d_plot(
    df, x: str, y: str,
    bins: tuple[int, int] = (200, 200),
    **kwargs,
):
    """2D histogram as go.Figure (Heatmap)."""
    go = _go()
    counts, x_edges, y_edges = _aggregate.hist2d(df, x, y, bins=bins)
    fig = go.Figure(go.Heatmap(
        z=counts.T,
        x=x_edges, y=y_edges,
        colorscale=kwargs.pop("colorscale", "Viridis"),
        colorbar=dict(title="count"),
    ))
    fig.update_layout(
        title=f"hist2d({x}, {y}, {bins[0]}x{bins[1]})",
        xaxis_title=x, yaxis_title=y,
        **kwargs,
    )
    return fig


def box_plot(df, column: str, **kwargs):
    """Box plot using precomputed quartiles (no raw data)."""
    go = _go()
    stats = _aggregate.box_stats(df, column)
    fig = go.Figure(go.Box(
        q1=[stats["q1"]],
        median=[stats["median"]],
        q3=[stats["q3"]],
        lowerfence=[stats["min"]],
        upperfence=[stats["max"]],
        name=column,
        boxmean=False,
    ))
    fig.update_layout(
        title=f"box({column})",
        yaxis_title=column,
        showlegend=False,
        **kwargs,
    )
    return fig


def line_plot(
    df, by: str, agg_col: str, agg: str = "mean", **kwargs,
):
    """Line plot from groupby aggregate."""
    go = _go()
    t = _aggregate.line_agg(df, by, agg_col, agg=agg)
    xs = t.column(by).to_pylist()
    out_col = f"{agg_col}_{agg}"
    if out_col not in t.schema.names:
        raise KeyError(
            f"line_plot: expected column {out_col!r}; got {t.schema.names}"
        )
    ys = t.column(out_col).to_pylist()
    fig = go.Figure(go.Scatter(
        x=xs, y=ys, mode="lines+markers",
        name=out_col,
    ))
    fig.update_layout(
        title=f"{out_col} by {by}",
        xaxis_title=by, yaxis_title=out_col,
        **kwargs,
    )
    return fig


def scatter(
    df, x: str, y: str,
    *,
    n: int = 10_000,
    bins: tuple[int, int] = (200, 200),
    **kwargs,
):
    """Scatter with auto-mode.

    - count() <= n: go.Scattergl (WebGL, bounded n)
    - count() >  n: go.Heatmap (from hist2d, O(pixels))
    """
    cnt = df.count()
    if cnt <= n:
        go = _go()
        xs, ys = _sample_pair(df, x, y, n)
        mask = ~(np.isnan(xs) | np.isnan(ys))
        xs, ys = xs[mask], ys[mask]
        fig = go.Figure(go.Scattergl(
            x=xs, y=ys, mode="markers",
            marker=dict(size=kwargs.pop("marker_size", 4)),
            name=f"{y} vs {x}",
        ))
        fig.update_layout(
            title=f"scatter({x}, {y}, n={len(xs)})",
            xaxis_title=x, yaxis_title=y,
            **kwargs,
        )
        return fig
    return hist2d_plot(df, x, y, bins=bins, **kwargs)


def feature_importance(
    model,
    *,
    top_k: int | None = None,
    **kwargs,
):
    """Horizontal bar of split counts per feature."""
    go = _go()
    trees = getattr(model, "trees_", None)
    names = getattr(model, "feature_names_", None)
    if trees is None or names is None:
        raise TypeError(
            "feature_importance: model must be a GBDT (trees_, "
            "feature_names_)"
        )
    imp = {n: 0 for n in names}
    for tree in trees:
        for _nid, spec in tree["structure"].items():
            if spec is None:
                continue
            j = spec[0]
            imp[names[j]] += 1
    items = sorted(imp.items(), key=lambda kv: kv[1])
    if top_k is not None:
        items = items[-top_k:]
    names_list = [n for n, _ in items]
    values = [v for _, v in items]
    fig = go.Figure(go.Bar(
        x=values, y=names_list, orientation="h",
        marker_color=kwargs.pop("color", "steelblue"),
    ))
    fig.update_layout(
        title="feature importance (split count)",
        xaxis_title="split count", yaxis_title="feature",
        **kwargs,
    )
    return fig
