"""v0.7.4b: Plotly adapter (aggregate-first, auto-mode scatter)."""
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
