"""Seaborn adapter (v0.7.4b).

Seaborn is built on matplotlib. Adapters use it where precomputed stats
are supported (hist via barplot, heatmap, line via lineplot). For box
and scatter — sample-based (n=1000, documented as approximate).

Requires: seaborn>=0.13 (lazy import), matplotlib (already via viz group).
"""
from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

from cadw.viz import _aggregate


def _sns():
    try:
        import seaborn as sns
    except ImportError as e:
        raise ImportError(
            "seaborn is required for cadw.viz.seaborn_adapter; "
            "install with: uv sync --group viz"
        ) from e
    return sns


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for cadw.viz.seaborn_adapter; "
            "install with: uv sync --group viz"
        ) from e
    return plt


def _sample_to_pandas(df, cols, n: int = 1000):
    """Pull sample(n) as pandas DataFrame with specified columns."""
    smp = _aggregate.sample(df, n=n)
    parts = {}
    for b in smp._iter_filtered_batches():
        for c in cols:
            arr = b.column(c).to_numpy(zero_copy_only=False)
            parts.setdefault(c, []).append(arr)
    if not parts:
        import pandas as pd
        return pd.DataFrame({c: [] for c in cols})
    import pandas as pd
    data = {}
    for c in cols:
        if c in parts:
            flat = np.concatenate(parts[c])
            data[c] = flat
    return pd.DataFrame(data)


def hist_plot(
    df, column: str, bins: int = 50,
    *, ax=None, **kwargs,
):
    """Histogram via sns.barplot on precomputed counts."""
    sns = _sns()
    plt = _plt()
    counts, edges = _aggregate.hist(df, column, bins=bins)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3))
    else:
        fig = ax.figure
    xs = (edges[:-1] + edges[1:]) / 2.0
    sns.barplot(
        x=xs, y=counts, ax=ax,
        color=kwargs.pop("color", "steelblue"),
        **kwargs,
    )
    ax.set_xlabel(column)
    ax.set_ylabel("count")
    ax.set_title(f"hist({column}, bins={bins})")
    return fig, ax


def heatmap_plot(
    df, x: str, y: str,
    bins: tuple[int, int] = (50, 50),
    *, ax=None, **kwargs,
):
    """2D histogram as sns.heatmap on counts_2d."""
    sns = _sns()
    plt = _plt()
    counts, x_edges, y_edges = _aggregate.hist2d(df, x, y, bins=bins)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure
    sns.heatmap(
        counts.T, ax=ax,
        cmap=kwargs.pop("cmap", "viridis"),
        cbar_kws={"label": "count"},
        xticklabels=False, yticklabels=False,
        **kwargs,
    )
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"heatmap({x}, {y}, {bins[0]}x{bins[1]})")
    return fig, ax


def line_plot(
    df, by: str, agg_col: str, agg: str = "mean",
    *, ax=None, **kwargs,
):
    """Line plot via sns.lineplot on precomputed aggregate."""
    sns = _sns()
    plt = _plt()
    t = _aggregate.line_agg(df, by, agg_col, agg=agg)
    out_col = f"{agg_col}_{agg}"
    if out_col not in t.schema.names:
        raise KeyError(
            f"line_plot: expected column {out_col!r}; got {t.schema.names}"
        )
    import pandas as pd
    pdf = pd.DataFrame({
        by: t.column(by).to_pylist(),
        out_col: t.column(out_col).to_pylist(),
    })
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3))
    else:
        fig = ax.figure
    sns.lineplot(data=pdf, x=by, y=out_col, marker="o", ax=ax, **kwargs)
    ax.set_xlabel(by)
    ax.set_ylabel(out_col)
    ax.set_title(f"{out_col} by {by}")
    return fig, ax


def box_plot(
    df, column: str,
    *, n: int = 1000, ax=None, **kwargs,
):
    """Box plot via sns.boxplot on sample(n=1000).

    NOTE: precomputed quartiles not supported by seaborn.boxplot (it
    requires raw values). Sample-based — quartiles approximate.
    For exact, use matplotlib_adapter.box_plot (precomputed via bxp).
    """
    sns = _sns()
    plt = _plt()
    pdf = _sample_to_pandas(df, [column], n=n)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))
    else:
        fig = ax.figure
    sns.boxplot(data=pdf, y=column, ax=ax, **kwargs)
    ax.set_ylabel(column)
    ax.set_title(f"box({column}) — sample n={len(pdf)}")
    return fig, ax


def scatter(
    df, x: str, y: str,
    *, n: int = 1000, ax=None, **kwargs,
):
    """Scatter via sns.scatterplot on sample(n=1000).

    Sample-based; for large N use matplotlib_adapter.scatter(auto) or
    plotly_adapter.scatter(auto) → hist2d.
    """
    sns = _sns()
    plt = _plt()
    pdf = _sample_to_pandas(df, [x, y], n=n)
    pdf = pdf.dropna(subset=[x, y])
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure
    sns.scatterplot(
        data=pdf, x=x, y=y, ax=ax,
        s=kwargs.pop("s", 8), alpha=kwargs.pop("alpha", 0.5),
        **kwargs,
    )
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"scatter({x}, {y}) — sample n={len(pdf)}")
    return fig, ax
