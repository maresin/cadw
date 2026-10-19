#!/usr/bin/env python3
"""v0.7.4a: aggregate-first viz layer + Histogram2DOp.

Design:
    Viz NEVER receives raw CadwDataFrame with millions of rows.
    All data flows through `cadw.viz._aggregate` (hist, hist2d, box_stats,
    line_agg, sample) — bounded state, matplotlib gets only aggregates.

    - hist     : Reduce(HistogramOp)      → (counts, edges)
    - hist2d   : Reduce(Histogram2DOp)    → (counts_2d, x_edges, y_edges)
    - box_stats: df.quantile()           → {min, q1, median, q3, max}
    - line_agg : df.groupby().agg()      → small Table (K rows)
    - sample   : df.head(n)              → CadwDataFrame (bounded n)

    matplotlib_adapter: aggregate -> render. Auto-mode scatter:
    count() <= n  → honest ax.scatter (bounded n, OK)
    count() >  n  → hist2d via imshow (O(pixels), not O(points))

Not in v0.7.4a:
    - Datashader, hexbin (v0.7.4b)
    - plotly / seaborn adapters (v0.7.4c/d)
    - Random sampling (v0.7.4b — needs row_id stride)

Run: uv run python scripts/patch_v074a_viz.py
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)


def sh(cmd, check=True):
    print(f"  $ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    r = subprocess.run(
        cmd, shell=isinstance(cmd, str), capture_output=True, text=True,
    )
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines()[:10]:
            print(f"    {line}")
    if check and r.returncode != 0:
        print(f"    STDERR: {r.stderr.strip()[-800:]}")
        sys.exit(r.returncode)
    return r.returncode


def compile_check(path):
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         f"import py_compile; py_compile.compile('{path}', doraise=True)"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"COMPILE FAIL: {path}")
        print(r.stderr[-800:])
        sys.exit(1)
    print(f"  compile OK: {path}")


def ensure_branch(name):
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    if r.stdout.strip() == name:
        print(f"  branch: already on {name}")
        return
    r = subprocess.run(["git", "rev-parse", "--verify", name],
                       capture_output=True)
    if r.returncode == 0:
        sh(["git", "checkout", name])
    else:
        sh(["git", "checkout", "-b", name])


def require_clean():
    r = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True,
    )
    if r.stdout.strip():
        print("ERROR: tracked files dirty:")
        print(r.stdout)
        sys.exit(1)


# ============ ops/histogram.py ============

HISTOGRAM_OPS = '''"""Histogram reduction ops — 1D/2D (v0.7.4a).

`Histogram2DOp`: 2D binning via Reduce. State O(bins_x × bins_y).
Used by cadw.viz._aggregate.hist2d for aggregate-first scatter plots.

For 1D, use `cadw.ops.reduce.HistogramOp` (already exists).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pyarrow as pa

from cadw.ops.reduce import ReduceOp


class Histogram2DOp(ReduceOp):
    """2D histogram: counts[bins_x, bins_y].

    Requires fixed x_min/max, y_min/max (two-pass or sample-based).
    Nulls in either column → row skipped.
    """

    def __init__(
        self,
        x_col: str,
        y_col: str,
        *,
        bins_x: int,
        bins_y: int,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> None:
        if bins_x < 1 or bins_y < 1:
            raise ValueError("Histogram2DOp: bins must be >= 1")
        if x_max <= x_min:
            raise ValueError("Histogram2DOp: x_max must be > x_min")
        if y_max <= y_min:
            raise ValueError("Histogram2DOp: y_max must be > y_min")
        self.x_col = x_col
        self.y_col = y_col
        self.bins_x = int(bins_x)
        self.bins_y = int(bins_y)
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)

    def init(self):
        return np.zeros((self.bins_x, self.bins_y), dtype=np.int64)

    def update(self, state, batch):
        x = _to_f64(batch.column(self.x_col))
        y = _to_f64(batch.column(self.y_col))
        mask = ~(np.isnan(x) | np.isnan(y))
        x = x[mask]
        y = y[mask]
        if x.shape[0] == 0:
            return state
        x_idx = np.clip(
            ((x - self.x_min) / (self.x_max - self.x_min)
             * self.bins_x).astype(np.int64),
            0, self.bins_x - 1,
        )
        y_idx = np.clip(
            ((y - self.y_min) / (self.y_max - self.y_min)
             * self.bins_y).astype(np.int64),
            0, self.bins_y - 1,
        )
        np.add.at(state, (x_idx, y_idx), 1)
        return state

    def merge(self, a, b):
        return a + b

    def state_nbytes(self, state):
        return int(state.nbytes)

    def estimate_state_nbytes(self):
        return self.bins_x * self.bins_y * 8

    def finalize(self, state):
        return {
            "counts": state,
            "x_edges": np.linspace(self.x_min, self.x_max, self.bins_x + 1),
            "y_edges": np.linspace(self.y_min, self.y_max, self.bins_y + 1),
        }


def _to_f64(arr: pa.Array) -> np.ndarray:
    np_arr = arr.to_numpy(zero_copy_only=False)
    if np_arr.dtype == object:
        out = np.empty(len(np_arr), dtype=np.float64)
        for i, v in enumerate(np_arr):
            out[i] = np.nan if v is None else float(v)
        return out
    return np_arr.astype(np.float64, copy=False)
'''


# ============ cadw/viz/__init__.py ============

VIZ_INIT = '''"""CADW viz layer (v0.7.4a): aggregate-first plotting.

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
'''


# ============ cadw/viz/_aggregate.py ============

AGGREGATE_SRC = '''"""Aggregate layer for viz (v0.7.4a).

Pure functions: input = CadwDataFrame, output = small numpy/Table.
No matplotlib dependency.

- hist     : Reduce(HistogramOp)     → (counts_np, edges_np)
- hist2d   : Reduce(Histogram2DOp)   → (counts_2d, x_edges, y_edges)
- box_stats: df.quantile             → dict
- line_agg : df.groupby().agg        → pa.Table (K rows)
- sample   : df.head(n)              → CadwDataFrame

Peak RAM: bounded by Reduce state (bins, k×d) or by sample size n.
Independent of N (for hist/hist2d), or O(n) small (for sample).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.ops.histogram import Histogram2DOp
from cadw.ops.reduce import HistogramOp, reduce


def _minmax_from_sample(
    df, column: str, *, pad_ratio: float = 0.01,
) -> tuple[float, float]:
    """Sample-based min/max from first batch. Fast but approximate.

    Padding ±pad_ratio handles out-of-sample outliers (they bin into
    edge buckets). For exact min/max — use Reduce(MinMaxOp) explicitly.
    """
    first = None
    for b in df._iter_filtered_batches():
        first = b
        break
    if first is None or first.num_rows == 0:
        raise ValueError(f"_minmax_from_sample: empty df for {column!r}")
    arr = first.column(column).to_numpy(zero_copy_only=False)
    if arr.dtype == object:
        vals = np.array(
            [np.nan if v is None else float(v) for v in arr],
            dtype=np.float64,
        )
    else:
        vals = arr.astype(np.float64, copy=False)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        raise ValueError(f"_minmax_from_sample: no valid values in {column!r}")
    lo = float(vals.min())
    hi = float(vals.max())
    if hi == lo:
        hi = lo + 1.0
    pad = (hi - lo) * pad_ratio
    return lo - pad, hi + pad


def hist(
    df,
    column: str,
    bins: int = 50,
    *,
    memory_guard: MemoryGuard | None = None,
):
    """1D histogram → (counts: np.ndarray, edges: np.ndarray)."""
    if column not in df.columns:
        raise KeyError(f"hist: column {column!r} not in df")
    if bins < 1:
        raise ValueError(f"hist: bins must be >= 1; got {bins}")
    mg = memory_guard or df._memory_guard
    lo, hi = _minmax_from_sample(df, column)
    op = HistogramOp(column, bins=bins, min_val=lo, max_val=hi)
    counts = reduce(df._iter_filtered_batches(), op, mg, log=False)
    edges = np.linspace(lo, hi, bins + 1)
    return counts.astype(np.int64), edges


def hist2d(
    df,
    x: str,
    y: str,
    bins: tuple[int, int] = (200, 200),
    *,
    memory_guard: MemoryGuard | None = None,
):
    """2D histogram → (counts: (bx, by) np.ndarray, x_edges, y_edges)."""
    for c in (x, y):
        if c not in df.columns:
            raise KeyError(f"hist2d: column {c!r} not in df")
    bx, by = int(bins[0]), int(bins[1])
    if bx < 1 or by < 1:
        raise ValueError(f"hist2d: bins must be >= 1; got {bins}")
    mg = memory_guard or df._memory_guard
    x_lo, x_hi = _minmax_from_sample(df, x)
    y_lo, y_hi = _minmax_from_sample(df, y)
    op = Histogram2DOp(
        x, y,
        bins_x=bx, bins_y=by,
        x_min=x_lo, x_max=x_hi, y_min=y_lo, y_max=y_hi,
    )
    result = reduce(df._iter_filtered_batches(), op, mg, log=False)
    return result["counts"], result["x_edges"], result["y_edges"]


def box_stats(
    df,
    column: str,
    *,
    quantiles: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> dict:
    """Boxplot stats via df.quantile. Returns dict {min, q1, median, q3, max}."""
    if column not in df.columns:
        raise KeyError(f"box_stats: column {column!r} not in df")
    if tuple(quantiles) != (0.0, 0.25, 0.5, 0.75, 1.0):
        raise ValueError(
            f"box_stats: quantiles must be exactly (0, .25, .5, .75, 1); "
            f"got {tuple(quantiles)}"
        )
    q = df.quantile(column, [0.0, 0.25, 0.5, 0.75, 1.0])
    return {
        "min": float(q[0]),
        "q1": float(q[1]),
        "median": float(q[2]),
        "q3": float(q[3]),
        "max": float(q[4]),
    }


def line_agg(
    df,
    by: str,
    agg_col: str,
    agg: str = "mean",
) -> pa.Table:
    """Groupby aggregate for line plot. Returns small Table (K rows)."""
    for c in (by, agg_col):
        if c not in df.columns:
            raise KeyError(f"line_agg: column {c!r} not in df")
    if agg not in {"mean", "sum", "count", "min", "max"}:
        raise ValueError(f"line_agg: agg must be sum/mean/count/min/max; got {agg}")
    result_df = df.groupby(by).agg(agg_col, agg)
    # Materialize as pa.Table (K rows — safe)
    batches = list(result_df._iter_filtered_batches())
    if not batches:
        return pa.Table.from_batches([], schema=result_df._dataset.schema)
    return pa.Table.from_batches(batches)


def sample(df, n: int = 10_000):
    """Bounded sample: returns df if count() <= n, else df.head(n)."""
    if n < 1:
        raise ValueError(f"sample: n must be >= 1; got {n}")
    cnt = df.count()
    if cnt <= n:
        return df
    return df.head(n)
'''


# ============ cadw/viz/matplotlib_adapter.py ============

MATPLOTLIB_ADAPTER = '''"""Matplotlib adapter (v0.7.4a).

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
'''


# ============ patch pyproject.toml ============

def patch_pyproject():
    print("\n=== pyproject.toml: matplotlib in [dependency-groups].viz ===")
    p = Path("pyproject.toml")
    s = p.read_text()
    if 'matplotlib' in s:
        print("  already present — skip")
        return
    # Add viz group; keep dev unchanged (matplotlib also in dev for tests)
    old = "[dependency-groups]\ndev = ["
    if old not in s:
        raise SystemExit("[dependency-groups] dev not found")
    s = s.replace(
        old,
        '[dependency-groups]\nviz = [\n    "matplotlib>=3.8",\n]\ndev = [\n    "matplotlib>=3.8",',
        1,
    )
    p.write_text(s)
    compile_check("pyproject.toml") if False else None  # toml, not py
    print("  OK: [dependency-groups].viz added; matplotlib in dev")


# ============ tests ============

AGGREGATE_TESTS = '''"""v0.7.4a: viz aggregate layer (hist, hist2d, box_stats, line_agg, sample)."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.viz import hist, hist2d, box_stats, line_agg, sample


def make_df(tmp_path, data, chunk=200, name="L"):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


# ---------- hist ----------

def test_hist_basic(tmp_path):
    rng = np.random.default_rng(0)
    n = 10_000
    x = rng.normal(size=n)
    df = make_df(tmp_path, {"x": x.tolist()})
    counts, edges = hist(df, "x", bins=20)
    assert counts.sum() == n
    assert len(counts) == 20
    assert len(edges) == 21


def test_hist_skips_nulls(tmp_path):
    df = make_df(tmp_path, {
        "x": [1.0, None, 2.0, None, 3.0],
    })
    counts, _ = hist(df, "x", bins=10)
    assert counts.sum() == 3


def test_hist_bad_column(tmp_path):
    df = make_df(tmp_path, {"x": [1.0, 2.0]})
    with pytest.raises(KeyError, match="nope"):
        hist(df, "nope")


# ---------- hist2d ----------

def test_hist2d_basic(tmp_path):
    rng = np.random.default_rng(0)
    n = 20_000
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    df = make_df(tmp_path, {"x": x.tolist(), "y": y.tolist()})
    counts, x_edges, y_edges = hist2d(df, "x", "y", bins=(50, 50))
    assert counts.sum() == n
    assert counts.shape == (50, 50)
    assert len(x_edges) == 51 and len(y_edges) == 51


def test_hist2d_skips_nulls(tmp_path):
    df = make_df(tmp_path, {
        "x": [1.0, None, 2.0, 3.0],
        "y": [1.0, 2.0, None, 3.0],
    })
    counts, _, _ = hist2d(df, "x", "y", bins=(5, 5))
    # Only (1.0,1.0) and (3.0,3.0) survive — both have valid x,y
    assert counts.sum() == 2


# ---------- box_stats ----------

def test_box_stats_basic(tmp_path):
    n = 1000
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    df = make_df(tmp_path, {"x": x.tolist()})
    stats = box_stats(df, "x")
    assert set(stats.keys()) == {"min", "q1", "median", "q3", "max"}
    assert stats["min"] <= stats["q1"] <= stats["median"]
    assert stats["median"] <= stats["q3"] <= stats["max"]


# ---------- line_agg ----------

def test_line_agg_basic(tmp_path):
    df = make_df(tmp_path, {
        "g": ["A", "A", "B", "B"],
        "v": [1.0, 2.0, 3.0, 4.0],
    })
    t = line_agg(df, "g", "v", agg="mean")
    assert t.num_rows == 2
    rows = {r["g"]: r["v_mean"] for r in t.to_pylist()}
    assert abs(rows["A"] - 1.5) < 1e-9
    assert abs(rows["B"] - 3.5) < 1e-9


# ---------- sample ----------

def test_sample_small_df_returns_self(tmp_path):
    df = make_df(tmp_path, {"x": [1.0, 2.0, 3.0]})
    s = sample(df, n=10)
    assert s is df


def test_sample_large_df_bounded(tmp_path):
    n = 1000
    df = make_df(tmp_path, {"x": list(range(n))})
    s = sample(df, n=100)
    assert s.count() == 100


def test_sample_bad_n(tmp_path):
    df = make_df(tmp_path, {"x": [1.0]})
    with pytest.raises(ValueError, match="n must be >= 1"):
        sample(df, n=0)
'''


MATPLOTLIB_TESTS = '''"""v0.7.4a: matplotlib adapter (aggregate-first, auto-mode scatter)."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# Force Agg for headless tests
import os
os.environ["MPLBACKEND"] = "Agg"

pytest.importorskip("matplotlib")

from cadw import CadwDataFrame, MemoryGuard
from cadw.viz.matplotlib_adapter import (
    hist_plot, hist2d_plot, box_plot, line_plot, scatter, feature_importance,
)


def make_df(tmp_path, data, chunk=500, name="L"):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def _save_and_check(fig, path: Path) -> int:
    fig.savefig(str(path), dpi=80, bbox_inches="tight")
    return path.stat().st_size


# ---------- hist_plot ----------

def test_hist_plot_writes_png(tmp_path):
    rng = np.random.default_rng(0)
    n = 5000
    df = make_df(tmp_path, {"x": rng.normal(size=n).tolist()})
    fig, ax = hist_plot(df, "x", bins=20)
    size = _save_and_check(fig, tmp_path / "hist.png")
    assert size > 1000


# ---------- hist2d_plot ----------

def test_hist2d_plot_writes_png(tmp_path):
    rng = np.random.default_rng(0)
    n = 20_000
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    df = make_df(tmp_path, {"x": x.tolist(), "y": y.tolist()})
    fig, ax = hist2d_plot(df, "x", "y", bins=(50, 50))
    size = _save_and_check(fig, tmp_path / "hist2d.png")
    assert size > 2000


# ---------- box_plot ----------

def test_box_plot_writes_png(tmp_path):
    rng = np.random.default_rng(0)
    df = make_df(tmp_path, {"x": rng.normal(size=1000).tolist()})
    fig, ax = box_plot(df, "x")
    size = _save_and_check(fig, tmp_path / "box.png")
    assert size > 500


# ---------- line_plot ----------

def test_line_plot_writes_png(tmp_path):
    df = make_df(tmp_path, {
        "g": ["A", "A", "B", "B", "C", "C"],
        "v": [1.0, 3.0, 2.0, 4.0, 3.0, 5.0],
    })
    fig, ax = line_plot(df, "g", "v", agg="mean")
    size = _save_and_check(fig, tmp_path / "line.png")
    assert size > 1000


# ---------- scatter auto-mode ----------

def test_scatter_small_uses_honest_scatter(tmp_path):
    """count() <= n → PathCollection (not AxesImage)."""
    rng = np.random.default_rng(0)
    n = 500
    df = make_df(tmp_path, {
        "x": rng.normal(size=n).tolist(),
        "y": rng.normal(size=n).tolist(),
    })
    fig, ax = scatter(df, "x", "y", n=1000)
    # Honest scatter → collections
    assert len(ax.collections) > 0
    # No imshow image
    assert len(ax.images) == 0
    size = _save_and_check(fig, tmp_path / "scatter.png")
    assert size > 500


def test_scatter_large_uses_hist2d(tmp_path):
    """count() > n → AxesImage (hist2d), NOT PathCollection with n+ points."""
    rng = np.random.default_rng(0)
    n = 5000
    df = make_df(tmp_path, {
        "x": rng.normal(size=n).tolist(),
        "y": rng.normal(size=n).tolist(),
    })
    fig, ax = scatter(df, "x", "y", n=1000, bins=(50, 50))
    # Aggregate mode → AxesImage
    assert len(ax.images) == 1
    # No scatter collection (or empty)
    assert sum(len(c.get_offsets()) for c in ax.collections) == 0
    size = _save_and_check(fig, tmp_path / "scatter_big.png")
    assert size > 2000


# ---------- feature_importance ----------

def test_feature_importance_render(tmp_path):
    """Train small GBDT and render feature importance."""
    from cadw import fit_gbdt_regressor
    rng = np.random.default_rng(0)
    n = 300
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    # Only x1 matters
    y = x1 * 3.0
    df = make_df(tmp_path, {
        "x1": x1.tolist(), "x2": x2.tolist(), "y": y.tolist(),
    })
    model = fit_gbdt_regressor(
        df, ["x1", "x2"], "y",
        n_estimators=10, max_depth=2, learning_rate=0.3, bins=16,
    )
    fig, ax = feature_importance(model)
    size = _save_and_check(fig, tmp_path / "fi.png")
    assert size > 500


def test_feature_importance_non_gbdt_raises():
    class FakeModel:
        pass
    with pytest.raises(TypeError, match="GBDT"):
        feature_importance(FakeModel())
'''


def write_tests():
    print("\n=== tests ===")
    for name, src in (
        ("tests/test_viz_aggregate.py", AGGREGATE_TESTS),
        ("tests/test_viz_matplotlib.py", MATPLOTLIB_TESTS),
    ):
        Path(name).write_text(src)
        compile_check(name)
    print("  OK: tests written")


# ============ exports in cadw/__init__.py ============

def patch_top_init():
    print("\n=== cadw/__init__.py: viz exports ===")
    p = Path("cadw/__init__.py")
    s = p.read_text()
    if "from .viz import" in s:
        print("  already present — skip")
        return
    if "__all__ = [" not in s:
        raise SystemExit("__all__ not found")
    idx = s.find("__all__ = [")
    s = s[:idx] + (
        "from .viz import (\n"
        "    hist as viz_hist,\n"
        "    hist2d as viz_hist2d,\n"
        "    box_stats as viz_box_stats,\n"
        "    line_agg as viz_line_agg,\n"
        "    sample as viz_sample,\n"
        ")\n\n"
    ) + s[idx:]
    s = s.replace(
        '__all__ = [',
        '__all__ = [\n'
        '    "viz_hist", "viz_hist2d", "viz_box_stats",\n'
        '    "viz_line_agg", "viz_sample",',
        1,
    )
    p.write_text(s)
    compile_check(p)
    print("  OK: viz exports added (prefixed with viz_)")


# ============ main ============

def main():
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    branch = "patch/v074a-viz"

    print("=== v0.7.4a: viz aggregate + Histogram2DOp ===")
    ensure_branch(branch)
    require_clean()

    backup_dir = Path(f".backups/backup-pre-v074a-{ts_str}")
    backup_dir.mkdir(parents=True, exist_ok=True)
    sh(["git", "archive", "--format=tar.gz",
        "-o", str(backup_dir / "repo.tgz"), "HEAD"])
    sh(["git", "tag", "-a", f"backup-pre-v074a-{ts_str}",
        "-m", "before patch v0.7.4a viz"])

    print("\n=== cadw/ops/histogram.py ===")
    Path("cadw/ops/histogram.py").write_text(HISTOGRAM_OPS)
    compile_check("cadw/ops/histogram.py")
    print("  OK: Histogram2DOp created")

    print("\n=== cadw/viz/ ===")
    Path("cadw/viz").mkdir(exist_ok=True)
    Path("cadw/viz/__init__.py").write_text(VIZ_INIT)
    Path("cadw/viz/_aggregate.py").write_text(AGGREGATE_SRC)
    Path("cadw/viz/matplotlib_adapter.py").write_text(MATPLOTLIB_ADAPTER)
    for f in ("__init__.py", "_aggregate.py", "matplotlib_adapter.py"):
        compile_check(f"cadw/viz/{f}")
    print("  OK: cadw/viz package created")

    patch_pyproject()
    patch_top_init()
    write_tests()

    # Install matplotlib (viz group + dev group)
    print("\n=== uv lock + sync ===")
    sh(["uv", "lock"])
    sh(["uv", "sync", "--all-groups"])

    print("\n=== pytest tests/test_viz_aggregate.py tests/test_viz_matplotlib.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_viz_aggregate.py",
         "tests/test_viz_matplotlib.py",
         "-v"]
    )
    if r.returncode != 0:
        print("\n*** FAILURE — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.7.4a (test red)"], check=False)
        sys.exit(1)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.7.4a: aggregate-first viz layer (hist/hist2d/box/line/sample) "
        "+ Histogram2DOp + matplotlib adapter (auto-mode scatter)"])

    print("\n==> v0.7.4a applied on", branch)


if __name__ == "__main__":
    main()