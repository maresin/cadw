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


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for cadw.viz.matplotlib_adapter; "
            "install with: uv sync --group viz"
        ) from e
    return plt


def _sample_pair(df, x: str, y: str, n: int):
    """Pull (x, y) numpy arrays for at most n rows."""
    smp = _aggregate.sample(df, n=n)
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


def feature_importance(
    model,
    *,
    top_k: int | None = None,
    ax=None,
    **kwargs,
):
    """GBDT feature importance (split count, not gain).

    Accumulates per-feature split counts across all trees. Returns
    (fig, ax) or (None, None) if model has no trees.
    """
    plt = _plt()
    trees = getattr(model, "trees_", None)
    feature_names = getattr(model, "feature_names_", None)
    if trees is None or feature_names is None:
        raise TypeError(
            "feature_importance: model must be a GBDT (trees_, "
            "feature_names_)"
        )
    importance = {name: 0 for name in feature_names}
    for tree in trees:
        for nid, spec in tree["structure"].items():
            if spec is None:
                continue
            j, _thr, _l, _r = spec
            importance[feature_names[j]] += 1
    if sum(importance.values()) == 0:
        if ax is not None:
            ax.set_title("feature_importance (no splits — constant model)")
            return ax.figure, ax
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.set_title("feature_importance (no splits — constant model)")
        return fig, ax
    items = sorted(importance.items(), key=lambda kv: kv[1])
    if top_k is not None:
        items = items[-top_k:]
    names = [n for n, _ in items]
    values = [v for _, v in items]
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, max(3, 0.3 * len(items))))
    else:
        fig = ax.figure
    ax.barh(names, values, **kwargs)
    ax.set_xlabel("split count")
    ax.set_title("feature importance (split count)")
    return fig, ax
