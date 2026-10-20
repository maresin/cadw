"""v0.9.0c: df.plot accessor tests."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard


def make_df(tmp_path, data, chunk=200, name="L"):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def _make_numeric_df(tmp_path):
    rng = np.random.default_rng(0)
    return make_df(tmp_path, {
        "x": rng.normal(size=200).tolist(),
        "y": rng.normal(size=200).tolist(),
        "g": [f"a{i % 4}" for i in range(200)],
    })


# ============ matplotlib (default backend) ============

def test_plot_hist_matplotlib(tmp_path):
    pytest.importorskip("matplotlib")
    df = _make_numeric_df(tmp_path)
    fig, ax = df.plot.hist("x", bins=10)
    assert fig is not None
    assert ax is not None


def test_plot_line_matplotlib(tmp_path):
    pytest.importorskip("matplotlib")
    df = _make_numeric_df(tmp_path)
    fig, ax = df.plot.line("g", "x")
    assert fig is not None


def test_plot_box_matplotlib(tmp_path):
    pytest.importorskip("matplotlib")
    df = _make_numeric_df(tmp_path)
    fig, ax = df.plot.box("x")
    assert fig is not None


def test_plot_scatter_matplotlib(tmp_path):
    pytest.importorskip("matplotlib")
    df = _make_numeric_df(tmp_path)
    fig, ax = df.plot.scatter("x", "y", n=100)
    assert fig is not None


def test_plot_hist2d_matplotlib(tmp_path):
    pytest.importorskip("matplotlib")
    df = _make_numeric_df(tmp_path)
    fig, ax = df.plot.hist2d("x", "y", bins=(10, 10))
    assert fig is not None


# ============ plotly (explicit backend) ============

def test_plot_plotly_hist(tmp_path):
    pytest.importorskip("plotly")
    df = _make_numeric_df(tmp_path)
    fig = df.plot.plotly.hist("x", bins=10)
    # plotly Figure has .data and .layout
    assert hasattr(fig, "data")
    assert hasattr(fig, "layout")


def test_plot_plotly_scatter(tmp_path):
    pytest.importorskip("plotly")
    df = _make_numeric_df(tmp_path)
    fig = df.plot.plotly.scatter("x", "y", n=100)
    assert hasattr(fig, "data")


# ============ seaborn (explicit backend) ============

def test_plot_seaborn_hist(tmp_path):
    pytest.importorskip("seaborn")
    df = _make_numeric_df(tmp_path)
    fig, ax = df.plot.seaborn.hist("x", bins=10)
    assert fig is not None


# ============ errors ============

def test_plot_missing_column_raises(tmp_path):
    pytest.importorskip("matplotlib")
    df = _make_numeric_df(tmp_path)
    with pytest.raises(KeyError, match="nope"):
        df.plot.hist("nope")
