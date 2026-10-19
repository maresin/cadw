#!/usr/bin/env python3
"""v0.7.4b: Plotly + Seaborn adapters (aggregate-first).

Design:
    Plotly:
        - hist       : go.Bar         (from counts, edges)
        - hist2d     : go.Heatmap     (from counts_2d, edges)
        - box        : go.Box         (precomputed q1/median/q3/lower/upper)
        - line       : go.Scatter     (from line_agg)
        - scatter    : auto-mode      (<=n: Scattergl; >n: Heatmap)
        - feature_imp: go.Bar horizontal

    Seaborn:
        - hist       : sns.barplot    (precomputed)
        - heatmap    : sns.heatmap    (from counts_2d)
        - line       : sns.lineplot   (from line_agg → pandas)
        - box        : sns.boxplot    (sample-based, n=1000; documented)
        - scatter    : sns.scatterplot (sample-based; documented)

    Rule: no renderer receives raw CadwDataFrame with millions of rows.
    hist/hist2d/line: aggregate → precomputed data (0 raw rows)
    box/scatter (seaborn): sample(n=1000) — bounded, approximate

Both adapters require optional deps (plotly / seaborn), lazy-imported.
Tests use pytest.importorskip.

Run: uv run python scripts/patch_v074b_viz_plotly_seaborn.py
"""
from __future__ import annotations

import os
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


# ============ plotly_adapter.py ============

PLOTLY_SRC = '''"""Plotly adapter (v0.7.4b).

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
'''


# ============ seaborn_adapter.py ============

SEABORN_SRC = '''"""Seaborn adapter (v0.7.4b).

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
'''


# ============ pyproject.toml ============

def patch_pyproject():
    print("\n=== pyproject.toml: plotly + seaborn ===")
    p = Path("pyproject.toml")
    s = p.read_text()
    if '"plotly>=5.18"' in s and '"seaborn>=0.13"' in s:
        print("  already present — skip")
        return
    old_viz = 'viz = [\n    "matplotlib>=3.8",\n]'
    new_viz = (
        'viz = [\n'
        '    "matplotlib>=3.8",\n'
        '    "plotly>=5.18",\n'
        '    "seaborn>=0.13",\n'
        ']'
    )
    if old_viz not in s:
        raise SystemExit("viz group not found in pyproject")
    s = s.replace(old_viz, new_viz, 1)
    # Add to dev
    old_dev = 'dev = [\n    "matplotlib>=3.8",'
    new_dev = (
        'dev = [\n'
        '    "matplotlib>=3.8",\n'
        '    "plotly>=5.18",\n'
        '    "seaborn>=0.13",'
    )
    if old_dev not in s:
        raise SystemExit("dev group start not found")
    s = s.replace(old_dev, new_dev, 1)
    p.write_text(s)
    print("  OK: plotly + seaborn in viz and dev groups")


# ============ tests: plotly ============

PLOTLY_TESTS = '''"""v0.7.4b: Plotly adapter (aggregate-first, auto-mode scatter)."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("plotly")

from cadw import CadwDataFrame, MemoryGuard
from cadw.viz.plotly_adapter import (
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


def test_hist_plot_returns_bar(tmp_path):
    rng = np.random.default_rng(0)
    n = 2000
    df = make_df(tmp_path, {"x": rng.normal(size=n).tolist()})
    fig = hist_plot(df, "x", bins=20)
    assert len(fig.data) == 1
    assert fig.data[0].type == "bar"
    assert sum(fig.data[0].y) == n  # totality invariant


def test_hist2d_plot_returns_heatmap(tmp_path):
    rng = np.random.default_rng(0)
    n = 5000
    df = make_df(tmp_path, {
        "x": rng.normal(size=n).tolist(),
        "y": rng.normal(size=n).tolist(),
    })
    fig = hist2d_plot(df, "x", "y", bins=(30, 30))
    assert len(fig.data) == 1
    assert fig.data[0].type == "heatmap"
    counts = np.array(fig.data[0].z)
    assert counts.sum() == n


def test_box_plot_precomputed(tmp_path):
    rng = np.random.default_rng(0)
    df = make_df(tmp_path, {"x": rng.normal(size=1000).tolist()})
    fig = box_plot(df, "x")
    assert len(fig.data) == 1
    assert fig.data[0].type == "box"
    # Precomputed quartiles present
    assert fig.data[0].q1 is not None


def test_line_plot_returns_scatter(tmp_path):
    df = make_df(tmp_path, {
        "g": ["A", "A", "B", "B", "C", "C"],
        "v": [1.0, 3.0, 2.0, 4.0, 3.0, 5.0],
    })
    fig = line_plot(df, "g", "v", agg="mean")
    assert len(fig.data) == 1
    assert fig.data[0].type == "scatter"
    assert list(fig.data[0].x) == ["A", "B", "C"]


def test_scatter_small_uses_scattergl(tmp_path):
    rng = np.random.default_rng(0)
    n = 500
    df = make_df(tmp_path, {
        "x": rng.normal(size=n).tolist(),
        "y": rng.normal(size=n).tolist(),
    })
    fig = scatter(df, "x", "y", n=1000)
    assert len(fig.data) == 1
    assert fig.data[0].type == "scattergl"


def test_scatter_large_uses_heatmap(tmp_path):
    rng = np.random.default_rng(0)
    n = 5000
    df = make_df(tmp_path, {
        "x": rng.normal(size=n).tolist(),
        "y": rng.normal(size=n).tolist(),
    })
    fig = scatter(df, "x", "y", n=1000, bins=(30, 30))
    assert len(fig.data) == 1
    assert fig.data[0].type == "heatmap"
    counts = np.array(fig.data[0].z)
    assert counts.sum() == n


def test_feature_importance_renders(tmp_path):
    from cadw import fit_gbdt_regressor
    rng = np.random.default_rng(0)
    n = 300
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = x1 * 3.0
    df = make_df(tmp_path, {
        "x1": x1.tolist(), "x2": x2.tolist(), "y": y.tolist(),
    })
    model = fit_gbdt_regressor(
        df, ["x1", "x2"], "y",
        n_estimators=10, max_depth=2, learning_rate=0.3, bins=16,
    )
    fig = feature_importance(model)
    assert len(fig.data) == 1
    assert fig.data[0].type == "bar"
    assert fig.data[0].orientation == "h"


def test_feature_importance_non_gbdt_raises():
    class FakeModel:
        pass
    with pytest.raises(TypeError, match="GBDT"):
        feature_importance(FakeModel())


def test_plotly_fig_to_json(tmp_path):
    """Smoke: full plotly serialization (dict) works."""
    rng = np.random.default_rng(0)
    df = make_df(tmp_path, {"x": rng.normal(size=100).tolist()})
    fig = hist_plot(df, "x", bins=10)
    d = fig.to_dict()
    assert "data" in d and "layout" in d
'''


SEABORN_TESTS = '''"""v0.7.4b: Seaborn adapter (aggregate-first where possible)."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import os
os.environ["MPLBACKEND"] = "Agg"

pytest.importorskip("seaborn")

from cadw import CadwDataFrame, MemoryGuard
from cadw.viz.seaborn_adapter import (
    hist_plot, heatmap_plot, line_plot, box_plot, scatter,
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


def _save(fig, path):
    fig.savefig(str(path), dpi=80, bbox_inches="tight")
    return path.stat().st_size


def test_hist_plot_writes_png(tmp_path):
    rng = np.random.default_rng(0)
    df = make_df(tmp_path, {"x": rng.normal(size=2000).tolist()})
    fig, ax = hist_plot(df, "x", bins=20)
    assert _save(fig, tmp_path / "h.png") > 1000


def test_heatmap_plot_writes_png(tmp_path):
    rng = np.random.default_rng(0)
    n = 5000
    df = make_df(tmp_path, {
        "x": rng.normal(size=n).tolist(),
        "y": rng.normal(size=n).tolist(),
    })
    fig, ax = heatmap_plot(df, "x", "y", bins=(30, 30))
    assert _save(fig, tmp_path / "hm.png") > 2000


def test_line_plot_writes_png(tmp_path):
    df = make_df(tmp_path, {
        "g": ["A", "A", "B", "B", "C", "C"],
        "v": [1.0, 3.0, 2.0, 4.0, 3.0, 5.0],
    })
    fig, ax = line_plot(df, "g", "v", agg="mean")
    assert _save(fig, tmp_path / "l.png") > 1000


def test_box_plot_sample_based(tmp_path):
    rng = np.random.default_rng(0)
    df = make_df(tmp_path, {"x": rng.normal(size=5000).tolist()})
    fig, ax = box_plot(df, "x", n=500)
    assert _save(fig, tmp_path / "b.png") > 500


def test_scatter_sample_based(tmp_path):
    rng = np.random.default_rng(0)
    n = 5000
    df = make_df(tmp_path, {
        "x": rng.normal(size=n).tolist(),
        "y": rng.normal(size=n).tolist(),
    })
    fig, ax = scatter(df, "x", "y", n=500)
    assert _save(fig, tmp_path / "s.png") > 1000


def test_heatmap_totality_invariant(tmp_path):
    """counts from hist2d are exact (sum == n)."""
    from cadw.viz import _aggregate
    rng = np.random.default_rng(0)
    n = 3000
    df = make_df(tmp_path, {
        "x": rng.normal(size=n).tolist(),
        "y": rng.normal(size=n).tolist(),
    })
    counts, _, _ = _aggregate.hist2d(df, "x", "y", bins=(20, 20))
    assert counts.sum() == n


def test_hist_totality_invariant(tmp_path):
    from cadw.viz import _aggregate
    rng = np.random.default_rng(0)
    n = 3000
    df = make_df(tmp_path, {"x": rng.normal(size=n).tolist()})
    counts, _ = _aggregate.hist(df, "x", bins=30)
    assert counts.sum() == n
'''


def write_tests():
    print("\n=== tests ===")
    for name, src in (
        ("tests/test_viz_plotly.py", PLOTLY_TESTS),
        ("tests/test_viz_seaborn.py", SEABORN_TESTS),
    ):
        Path(name).write_text(src)
        compile_check(name)
    print("  OK: plotly + seaborn tests written")


def main():
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    branch = "patch/v074b-viz-plotly-seaborn"

    print("=== v0.7.4b: Plotly + Seaborn adapters ===")
    ensure_branch(branch)
    require_clean()

    backup_dir = Path(f".backups/backup-pre-v074b-{ts_str}")
    backup_dir.mkdir(parents=True, exist_ok=True)
    sh(["git", "archive", "--format=tar.gz",
        "-o", str(backup_dir / "repo.tgz"), "HEAD"])
    sh(["git", "tag", "-a", f"backup-pre-v074b-{ts_str}",
        "-m", "before patch v0.7.4b plotly+seaborn"])

    print("\n=== cadw/viz/plotly_adapter.py + seaborn_adapter.py ===")
    Path("cadw/viz/plotly_adapter.py").write_text(PLOTLY_SRC)
    Path("cadw/viz/seaborn_adapter.py").write_text(SEABORN_SRC)
    compile_check("cadw/viz/plotly_adapter.py")
    compile_check("cadw/viz/seaborn_adapter.py")
    print("  OK: adapters created")

    patch_pyproject()
    write_tests()

    print("\n=== uv lock + sync ===")
    sh(["uv", "lock"])
    sh(["uv", "sync", "--all-groups"])

    print("\n=== pytest tests/test_viz_plotly.py tests/test_viz_seaborn.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_viz_plotly.py",
         "tests/test_viz_seaborn.py",
         "-v"]
    )
    if r.returncode != 0:
        print("\n*** FAILURE — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.7.4b (test red)"], check=False)
        sys.exit(1)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.7.4b: Plotly + Seaborn adapters (aggregate-first; auto-mode scatter)"])

    print("\n==> v0.7.4b applied on", branch)


if __name__ == "__main__":
    main()