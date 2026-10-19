"""v0.7.4a: viz aggregate layer (hist, hist2d, box_stats, line_agg, sample)."""
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
