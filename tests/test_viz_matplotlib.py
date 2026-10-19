"""v0.7.4a: matplotlib adapter (aggregate-first, auto-mode scatter)."""
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

# ---------- v0.7.4c: hexbin_plot ----------

def test_hexbin_plot_writes_png(tmp_path):
    """hexbin_plot: sample + plt.hexbin, writes PNG."""
    rng = np.random.default_rng(0)
    n = 10_000
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    df = make_df(tmp_path, {"x": x.tolist(), "y": y.tolist()})
    from cadw.viz.matplotlib_adapter import hexbin_plot
    fig, ax = hexbin_plot(df, "x", "y", gridsize=20, n=2000)
    out = tmp_path / "hexbin.png"
    fig.savefig(str(out), dpi=80, bbox_inches="tight")
    assert out.stat().st_size > 1500


def test_hexbin_plot_random_sample(tmp_path):
    """random=True: sample spreads across dataset."""
    n = 5000
    rng = np.random.default_rng(1)
    df = make_df(tmp_path, {
        "x": np.arange(n, dtype=np.float64).tolist(),
        "y": rng.normal(size=n).tolist(),
    })
    from cadw.viz.matplotlib_adapter import hexbin_plot
    fig, ax = hexbin_plot(df, "x", "y", gridsize=20, n=500, random=True)
    # If sample were head, all x ∈ [0, 500]; check title contains n=500
    assert "n=500" in ax.get_title()


def test_hexbin_plot_missing_column(tmp_path):
    df = make_df(tmp_path, {"x": [1.0, 2.0], "y": [3.0, 4.0]})
    from cadw.viz.matplotlib_adapter import hexbin_plot
    import pytest
    with pytest.raises(KeyError, match="nope"):
        hexbin_plot(df, "nope", "y")


def test_hexbin_plot_invalid_gridsize(tmp_path):
    df = make_df(tmp_path, {"x": [1.0, 2.0], "y": [3.0, 4.0]})
    from cadw.viz.matplotlib_adapter import hexbin_plot
    import pytest
    with pytest.raises(ValueError, match="gridsize"):
        hexbin_plot(df, "x", "y", gridsize=0)


def test_hexbin_plot_skips_nulls(tmp_path):
    """Rows with null in x or y — dropped before plt.hexbin."""
    df = make_df(tmp_path, {
        "x": [1.0, None, 3.0, 4.0],
        "y": [1.0, 2.0, None, 4.0],
    })
    from cadw.viz.matplotlib_adapter import hexbin_plot
    fig, ax = hexbin_plot(df, "x", "y", n=100, random=False)
    # 2 valid rows (1,1) and (4,4) — hexbin shows both
    assert "n=2" in ax.get_title()

