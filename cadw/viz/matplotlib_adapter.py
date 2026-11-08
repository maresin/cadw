"""Matplotlib adapter (v0.7.4a).

Aggregate-first: matplotlib receives small numpy arrays or Tables,
never raw CadwDataFrame.

Auto-mode scatter:
    count() <= n  → ax.scatter(sample)        (n bounded, OK)
    count() >  n  → hist2d + imshow            (O(pixels), not O(points))
"""
from __future__ import annotations

import os

# Set Agg backend by default — safe for headless & CI. User can override
# with MPLBACKEND env var (e.g., "TkAgg" in Jupyter).
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

from cadw.viz import _aggregate

from cadw.viz._lazy import lazy_plt


def _plt():
    return lazy_plt("cadw.viz.matplotlib_adapter")


def _sample_pair(df, x: str, y: str, n: int, *, random: bool = False):
    """Pull (x, y) numpy arrays for at most n rows.

    random=False (default): head (fast, may be biased on ordered data).
    random=True: uniform random sample (full O(N) scan).
    """
    smp = _aggregate.sample(df, n=n, random=random)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for b in smp._iter_filtered_batches():
        xs.append(_to_np(b.column(x)))
        ys.append(_to_np(b.column(y)))
    if not xs:
        return np.empty(0), np.empty(0)
    return np.concatenate(xs), np.concatenate(ys)


def _to_np(arr):
    a = arr.to_numpy(zero_copy_only=False)
    if a.dtype == object:
        return np.array(
            [np.nan if v is None else float(v) for v in a],
            dtype=np.float64,
        )
    return a.astype(np.float64, copy=False)


def hist_plot(df, column: str, bins: int = 50, *, ax=None, **kwargs):
    """1D histogram. Returns (fig, ax)."""
    plt = _plt()
    counts, edges = _aggregate.hist(df, column, bins=bins)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3))
    else:
        fig = ax.figure
    ax.bar(
        edges[:-1], counts,
        width=np.diff(edges),
        align="edge",
        edgecolor=kwargs.pop("edgecolor", "k"),
        **kwargs,
    )
    ax.set_xlabel(column)
    ax.set_ylabel("count")
    ax.set_title(f"hist({column}, bins={bins})")
    return fig, ax


def hist2d_plot(
    df, x: str, y: str,
    bins: tuple[int, int] = (200, 200),
    *,
    ax=None,
    log_scale: bool = True,
    cmap: str = "viridis",
    **kwargs,
):
    """2D histogram as imshow. Returns (fig, ax)."""
    plt = _plt()
    import matplotlib.colors as mcolors
    counts, x_edges, y_edges = _aggregate.hist2d(df, x, y, bins=bins)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure
    img = counts.T
    norm = mcolors.LogNorm(vmin=max(1, img[img > 0].min() if (img > 0).any() else 1),
                           vmax=img.max()) if log_scale else None
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    im = ax.imshow(
        img, origin="lower", aspect="auto", extent=extent,
        cmap=cmap, norm=norm, **kwargs,
    )
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"hist2d({x}, {y}, {bins[0]}x{bins[1]})")
    fig.colorbar(im, ax=ax, label="count")
    return fig, ax


def box_plot(df, column: str, *, ax=None, **kwargs):
    """Box plot from quantile stats. Returns (fig, ax)."""
    plt = _plt()
    stats = _aggregate.box_stats(df, column)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))
    else:
        fig = ax.figure
    ax.bxp(
        [{
            "med": stats["median"],
            "q1": stats["q1"],
            "q3": stats["q3"],
            "whislo": stats["min"],
            "whishi": stats["max"],
            "fliers": [],
            "label": column,
        }],
        showfliers=False,
        **kwargs,
    )
    ax.set_ylabel(column)
    ax.set_title(f"box({column})")
    return fig, ax


def line_plot(
    df, by: str, agg_col: str, agg: str = "mean",
    *, ax=None, **kwargs,
):
    """Line plot from groupby aggregate. Returns (fig, ax)."""
    plt = _plt()
    t = _aggregate.line_agg(df, by, agg_col, agg=agg)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3))
    else:
        fig = ax.figure
    xs = t.column(by).to_pylist()
    out_col = f"{agg_col}_{agg}"
    if out_col not in t.schema.names:
        raise KeyError(
            f"line_plot: expected column {out_col!r} in aggregated "
            f"table; got {t.schema.names}"
        )
    ys = t.column(out_col).to_pylist()
    ax.plot(xs, ys, marker=kwargs.pop("marker", "o"), **kwargs)
    ax.set_xlabel(by)
    ax.set_ylabel(out_col)
    ax.set_title(f"{out_col} by {by}")
    return fig, ax


def scatter(
    df, x: str, y: str,
    *,
    n: int = 10_000,
    bins: tuple[int, int] = (200, 200),
    ax=None,
    cmap: str = "viridis",
    **kwargs,
):
    """Scatter with auto-mode.

    - If df.count() <= n: honest scatter (bounded n, safe).
    - Else: hist2d via imshow (O(pixels), not O(points)).
    Returns (fig, ax).
    """
    plt = _plt()
    if df.count() <= n:
        xs, ys = _sample_pair(df, x, y, n)
        mask = ~(np.isnan(xs) | np.isnan(ys))
        xs, ys = xs[mask], ys[mask]
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
        else:
            fig = ax.figure
        ax.scatter(
            xs, ys,
            s=kwargs.pop("s", 4),
            alpha=kwargs.pop("alpha", 0.5),
            **kwargs,
        )
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"scatter({x}, {y}, n={len(xs)})")
        return fig, ax
    # Aggregate mode
    return hist2d_plot(df, x, y, bins=bins, ax=ax, cmap=cmap, **kwargs)


def hexbin_plot(
    df, x: str, y: str,
    *,
    gridsize: int = 30,
    n: int = 10_000,
    random: bool = True,
    ax=None,
    cmap: str = "viridis",
    log_scale: bool = True,
    **kwargs,
):
    """Hexbin via sample(n) + plt.hexbin (v0.7.4c, matplotlib native).

    **Sample-based, not aggregate-first**: hexbin aggregation on a
    hexagonal grid is non-trivial (matplotlib does its own binning
    internally via plt.hexbin). We pass sampled (xs, ys), not counts.

    - n=10_000: bounded sample (same pattern as scatter).
    - random=True (default): unbiased. Full O(N) scan — set False for
      fast preview on ordered data (head, may be biased).
    - gridsize: matplotlib hexagon count.
    - log_scale=True: LogNorm for colorbar (typical scatter density).

    Plotly/Seaborn don't have native hexbin — use hist2d_plot instead.
    """
    plt = _plt()
    import matplotlib.colors as mcolors
    if gridsize < 1:
        raise ValueError(f"hexbin_plot: gridsize must be >= 1; got {gridsize}")
    xs, ys = _sample_pair(df, x, y, n, random=random)
    # Drop NaN rows (both x and y must be valid)
    mask = ~(np.isnan(xs) | np.isnan(ys))
    xs = xs[mask]
    ys = ys[mask]
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure
    norm = mcolors.LogNorm() if log_scale and len(xs) > 0 else None
    hb = ax.hexbin(
        xs, ys,
        gridsize=gridsize,
        cmap=cmap,
        norm=norm,
        **kwargs,
    )
    fig.colorbar(hb, ax=ax, label="count")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"hexbin({x}, {y}, gridsize={gridsize}, n={len(xs)})")
    return fig, ax


def feature_importance(
    model,
    *,
    importance_type: str = "gain",
    top_k: int | None = None,
    ax=None,
    **kwargs,
):
    """GBDT feature importance (gain or split count).

    - importance_type='gain' (default): sum of split gains per feature
      (XGBoost-style). Values are raw gain sums (varies by tree scale).
    - importance_type='split': number of times feature used in a split
      (v0.7.4a behavior).

    Returns (fig, ax) or (fig, ax) with title indicating constant model.
    """
    plt = _plt()
    from cadw.ml.gbdt import compute_feature_importance
    importance = compute_feature_importance(
        model, importance_type=importance_type, top_k=top_k,
    )
    if sum(importance.values()) == 0:
        if ax is not None:
            ax.set_title(
                f"feature_importance ({importance_type}) — no splits"
            )
            return ax.figure, ax
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.set_title(
            f"feature_importance ({importance_type}) — no splits"
        )
        return fig, ax
    names = [n for n, _ in importance.items()]
    values = [v for _, v in importance.items()]
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, max(3, 0.3 * len(importance))))
    else:
        fig = ax.figure
    ax.barh(names, values, **kwargs)
    ax.set_xlabel(importance_type)
    ax.set_title(f"feature importance ({importance_type})")
    return fig, ax
