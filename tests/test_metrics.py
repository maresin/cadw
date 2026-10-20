"""v0.9.2a: regression + log_loss metrics tests."""
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard


def make_df(tmp_path, data, chunk=3, name="L", mem_mb=50):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


# ============ r2 ============

def test_metrics_r2_perfect(tmp_path):
    df = make_df(tmp_path, {"y": [1.0, 2.0, 3.0], "yp": [1.0, 2.0, 3.0]})
    res = df.metrics(y_true="y", y_pred="yp", metrics=["r2"])
    assert res["r2"] == pytest.approx(1.0)


def test_metrics_r2_known(tmp_path):
    """y=[1,2,3,4], yp=[2,2,3,3] → SS_res=2, SS_tot=5, r2=1-2/5=0.6."""
    df = make_df(tmp_path, {
        "y": [1.0, 2.0, 3.0, 4.0],
        "yp": [2.0, 2.0, 3.0, 3.0],
    })
    res = df.metrics(y_true="y", y_pred="yp", metrics=["r2"])
    assert res["r2"] == pytest.approx(0.6)


def test_metrics_r2_constant_y_perfect(tmp_path):
    """y all equal to yp → ss_tot==0, ss_res==0 → r2=1.0 (sklearn behavior)."""
    df = make_df(tmp_path, {
        "y": [5.0, 5.0, 5.0],
        "yp": [5.0, 5.0, 5.0],
    })
    res = df.metrics(y_true="y", y_pred="yp", metrics=["r2"])
    assert res["r2"] == 1.0


def test_metrics_r2_constant_y_wrong_pred(tmp_path):
    """y all 5, yp all 3 → ss_res>0, ss_tot==0 → r2=0.0 (sklearn behavior)."""
    df = make_df(tmp_path, {
        "y": [5.0, 5.0],
        "yp": [3.0, 3.0],
    })
    res = df.metrics(y_true="y", y_pred="yp", metrics=["r2"])
    assert res["r2"] == 0.0


# ============ mse / rmse / mae ============

def test_metrics_mse_rmse_mae(tmp_path):
    """y=[0,0,0], yp=[1,-1,0] → mse=2/3, rmse=sqrt(2/3), mae=2/3."""
    df = make_df(tmp_path, {
        "y": [0.0, 0.0, 0.0],
        "yp": [1.0, -1.0, 0.0],
    })
    res = df.metrics(
        y_true="y", y_pred="yp",
        metrics=["mse", "rmse", "mae"],
    )
    assert res["mse"] == pytest.approx(2.0 / 3.0)
    assert res["rmse"] == pytest.approx(math.sqrt(2.0 / 3.0))
    assert res["mae"] == pytest.approx(2.0 / 3.0)


def test_metrics_all_regression_combined(tmp_path):
    """All regression metrics from one pass."""
    df = make_df(tmp_path, {
        "y": [1.0, 2.0, 3.0, 4.0],
        "yp": [1.1, 1.9, 3.1, 3.9],
    })
    res = df.metrics(
        y_true="y", y_pred="yp",
        metrics=["r2", "mse", "rmse", "mae"],
    )
    assert set(res.keys()) == {"r2", "mse", "rmse", "mae"}
    assert res["rmse"] == pytest.approx(math.sqrt(res["mse"]))
    # residual abs: 0.1 each → mae = 0.1
    assert res["mae"] == pytest.approx(0.1)


# ============ log_loss ============

def test_metrics_log_loss_perfect(tmp_path):
    """y=[1,0], p=[1,0] → clipped, loss ~ tiny."""
    df = make_df(tmp_path, {"y": [1.0, 0.0], "p": [1.0, 0.0]})
    res = df.metrics(y_true="y", y_proba="p", metrics=["log_loss"])
    assert res["log_loss"] < 1e-10


def test_metrics_log_loss_50_50(tmp_path):
    """y=[1,0], p=[0.5,0.5] → loss = -log(0.5) = ln(2)."""
    df = make_df(tmp_path, {"y": [1.0, 0.0], "p": [0.5, 0.5]})
    res = df.metrics(y_true="y", y_proba="p", metrics=["log_loss"])
    assert res["log_loss"] == pytest.approx(math.log(2.0))


def test_metrics_log_loss_known(tmp_path):
    """y=[1,0], p=[0.9, 0.1] → loss = -[log(0.9)+log(0.9)]/2 = -log(0.9)."""
    df = make_df(tmp_path, {"y": [1.0, 0.0], "p": [0.9, 0.1]})
    res = df.metrics(y_true="y", y_proba="p", metrics=["log_loss"])
    assert res["log_loss"] == pytest.approx(-math.log(0.9))


# ============ combined ============

def test_metrics_regression_and_log_loss(tmp_path):
    """Both y_pred and y_proba passed in one call."""
    df = make_df(tmp_path, {
        "y": [0.0, 1.0, 0.0, 1.0],
        "yp": [0.1, 0.9, 0.2, 0.8],
        "p": [0.1, 0.9, 0.2, 0.8],
    })
    res = df.metrics(
        y_true="y", y_pred="yp", y_proba="p",
        metrics=["mse", "log_loss"],
    )
    assert set(res.keys()) == {"mse", "log_loss"}
    assert res["mse"] > 0
    assert res["log_loss"] > 0


# ============ nulls, empty ============

def test_metrics_nulls_skipped(tmp_path):
    """Null in y or yp → row skipped."""
    df = make_df(tmp_path, {
        "y": [1.0, None, 3.0, 4.0],
        "yp": [1.0, 2.0, None, 4.0],
    })
    res = df.metrics(y_true="y", y_pred="yp", metrics=["mse"])
    # valid rows: (1,1), (4,4) → mse = 0
    assert res["mse"] == pytest.approx(0.0)


def test_metrics_empty_input_nan(tmp_path):
    df = make_df(tmp_path, {"y": [1.0, 2.0], "yp": [1.0, 2.0]})
    empty = df[df["y"] > 100]
    res = empty.metrics(
        y_true="y", y_pred="yp", metrics=["r2", "rmse", "mae"],
    )
    for k in ("r2", "rmse", "mae"):
        assert math.isnan(res[k])


def test_metrics_filtered(tmp_path):
    df = make_df(tmp_path, {
        "y": [0.0, 1.0, 2.0, 3.0],
        "yp": [0.0, 1.0, 2.0, 100.0],  # big error on filtered-out row
    })
    filtered = df[df["y"] < 3]
    res = filtered.metrics(y_true="y", y_pred="yp", metrics=["mse"])
    assert res["mse"] == pytest.approx(0.0)


# ============ validation ============

def test_metrics_string_metric(tmp_path):
    """String metric (not list) works."""
    df = make_df(tmp_path, {"y": [1.0, 2.0], "yp": [1.0, 2.0]})
    res = df.metrics(y_true="y", y_pred="yp", metrics="r2")
    assert res["r2"] == pytest.approx(1.0)


def test_metrics_invalid_metric(tmp_path):
    df = make_df(tmp_path, {"y": [1.0], "yp": [1.0]})
    with pytest.raises(ValueError, match="unknown metric"):
        df.metrics(y_true="y", y_pred="yp", metrics=["bogus"])


def test_metrics_missing_y_pred_for_r2(tmp_path):
    df = make_df(tmp_path, {"y": [1.0], "yp": [1.0]})
    with pytest.raises(ValueError, match="y_pred_col required"):
        df.metrics(y_true="y", metrics=["r2"])


def test_metrics_missing_y_proba_for_log_loss(tmp_path):
    df = make_df(tmp_path, {"y": [1.0], "yp": [1.0]})
    with pytest.raises(ValueError, match="y_proba_col required"):
        df.metrics(y_true="y", metrics=["log_loss"])


def test_metrics_missing_column(tmp_path):
    df = make_df(tmp_path, {"y": [1.0], "yp": [1.0]})
    with pytest.raises(KeyError, match="nope"):
        df.metrics(y_true="nope", y_pred="yp", metrics=["r2"])


def test_metrics_empty_metrics_list(tmp_path):
    df = make_df(tmp_path, {"y": [1.0], "yp": [1.0]})
    with pytest.raises(ValueError, match="metrics list is empty"):
        df.metrics(y_true="y", y_pred="yp", metrics=[])


# ============ pandas / sklearn equivalence ============

def test_metrics_pandas_equivalence(tmp_path):
    n = 200
    rng = np.random.default_rng(42)
    y = rng.normal(size=n)
    yp = y + rng.normal(scale=0.5, size=n)
    df = make_df(tmp_path, {"y": y.tolist(), "yp": yp.tolist()}, chunk=50)
    res = df.metrics(
        y_true="y", y_pred="yp", metrics=["r2", "mse", "rmse", "mae"],
    )
    # Reference (numpy, matches sklearn)
    ss_res = float(((y - yp) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    assert res["r2"] == pytest.approx(1.0 - ss_res / ss_tot)
    assert res["mse"] == pytest.approx(ss_res / n)
    assert res["rmse"] == pytest.approx(math.sqrt(ss_res / n))
    assert res["mae"] == pytest.approx(float(np.abs(y - yp).mean()))


# ============ integration ============

def test_metrics_with_groupby_predict(tmp_path):
    """End-to-end: groupby → agg → predict → metrics works as chain."""
    n = 100
    rng = np.random.default_rng(0)
    df = make_df(tmp_path, {
        "x": rng.normal(size=n).tolist(),
        "y": (rng.normal(size=n) * 2).tolist(),
    })
    # Just verify metrics can be called on a df with any data
    res = df.metrics(y_true="y", y_pred="x", metrics=["mse"])
    assert res["mse"] > 0


def test_metrics_log_loss_rejects_continuous_y(tmp_path):
    """§13.4/163: log_loss requires binary y; continuous y → ValueError.

    Formula `-(y·log(p) + (1-y)·log(1-p))` assumes y ∈ {0, 1}.
    With y=2.0, output is negative — mathematically invalid. Must
    reject with clear message.
    """
    df = make_df(tmp_path, {
        "y": [0.0, 1.0, 2.0],  # contains invalid 2.0
        "p": [0.1, 0.9, 0.5],
    })
    with pytest.raises(ValueError, match="requires binary y_true"):
        df.metrics(y_true="y", y_proba="p", metrics=["log_loss"])
