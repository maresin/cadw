"""v0.7.4b: Seaborn adapter (aggregate-first where possible)."""
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
