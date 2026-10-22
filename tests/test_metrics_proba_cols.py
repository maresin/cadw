"""v0.9.23b: metrics(y_proba_cols=[...]) alias (AUDIT_FULL_v0922 7.A.11.3).

y_proba_cols is an alias for y_proba when passing a list of columns,
matching MetricsOp's parameter name. Both forms work; both together →
ValueError.
"""
from pathlib import Path

import math
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard


def make_df(tmp_path, data, chunk=3, name="L", mem_mb=50):
    t = pa.table(data)
    p = tmp_path / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, tmp_path / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
        spill_dir=tmp_path / f"spill_{name}",
    )


# ============ y_proba_cols alias: positive cases ============


def test_metrics_y_proba_cols_binary(tmp_path):
    """y_proba_cols=['p'] on binary y → same as y_proba='p'."""
    df = make_df(tmp_path, {
        "y": [0.0, 0.0, 1.0, 1.0],
        "p": [0.1, 0.4, 0.35, 0.8],
    })
    via_kwarg = df.metrics(
        y_true="y", y_proba="p", metrics=["roc_auc"],
    )
    via_alias = df.metrics(
        y_true="y", y_proba_cols=["p"], metrics=["roc_auc"],
    )
    assert via_kwarg == via_alias
    assert via_alias["roc_auc"] == pytest.approx(0.75)


def test_metrics_y_proba_cols_multiclass_log_loss(tmp_path):
    """y_proba_cols=[3 cols] on multiclass y → same as y_proba list."""
    df = make_df(tmp_path, {
        "y": [0, 1, 2, 0, 1, 2],
        "p0": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "p1": [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        "p2": [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    })
    via_list = df.metrics(
        y_true="y", y_proba=["p0", "p1", "p2"],
        metrics=["log_loss"], classes=[0, 1, 2],
    )
    via_alias = df.metrics(
        y_true="y", y_proba_cols=["p0", "p1", "p2"],
        metrics=["log_loss"], classes=[0, 1, 2],
    )
    assert via_alias["log_loss"] == pytest.approx(via_list["log_loss"])
    assert via_alias["log_loss"] == pytest.approx(0.0, abs=1e-10)


def test_metrics_y_proba_cols_multiclass_roc_auc(tmp_path):
    """y_proba_cols works for macro roc_auc (K ovr passes)."""
    df = make_df(tmp_path, {
        "y": [0, 0, 1, 1, 2, 2],
        "p0": [0.9, 0.8, 0.05, 0.1, 0.05, 0.1],
        "p1": [0.05, 0.1, 0.9, 0.8, 0.1, 0.05],
        "p2": [0.05, 0.1, 0.05, 0.1, 0.9, 0.85],
    })
    res = df.metrics(
        y_true="y", y_proba_cols=["p0", "p1", "p2"],
        metrics=["roc_auc"], classes=[0, 1, 2],
    )
    assert res["roc_auc"] == pytest.approx(1.0)


# ============ y_proba_cols alias: errors ============


def test_metrics_y_proba_cols_both_given_raises(tmp_path):
    """Both y_proba= and y_proba_cols= → ValueError (ambiguous)."""
    df = make_df(tmp_path, {
        "y": [0, 1], "p": [0.1, 0.9],
    })
    with pytest.raises(ValueError, match="not both"):
        df.metrics(
            y_true="y",
            y_proba="p",
            y_proba_cols=["p"],
            metrics=["roc_auc"],
        )


def test_metrics_y_proba_cols_string_raises(tmp_path):
    """y_proba_cols='p' (single string) → TypeError with hint."""
    df = make_df(tmp_path, {
        "y": [0, 1], "p": [0.1, 0.9],
    })
    with pytest.raises(TypeError, match="must be a sequence"):
        df.metrics(
            y_true="y", y_proba_cols="p", metrics=["roc_auc"],
        )


def test_metrics_y_proba_cols_empty_raises(tmp_path):
    """y_proba_cols=[] → ValueError."""
    df = make_df(tmp_path, {
        "y": [0, 1], "p": [0.1, 0.9],
    })
    with pytest.raises(ValueError, match="must be non-empty"):
        df.metrics(
            y_true="y", y_proba_cols=[], metrics=["roc_auc"],
        )


# ============ backward compat: y_proba unchanged ============


def test_metrics_y_proba_str_unchanged(tmp_path):
    """Existing y_proba='p' still works."""
    df = make_df(tmp_path, {
        "y": [0.0, 0.0, 1.0, 1.0],
        "p": [0.1, 0.4, 0.35, 0.8],
    })
    res = df.metrics(y_true="y", y_proba="p", metrics=["roc_auc"])
    assert res["roc_auc"] == pytest.approx(0.75)


def test_metrics_y_proba_list_unchanged(tmp_path):
    """Existing y_proba=[...] (list) still works."""
    df = make_df(tmp_path, {
        "y": [0, 1, 2],
        "p0": [0.9, 0.1, 0.2],
        "p1": [0.05, 0.7, 0.2],
        "p2": [0.05, 0.2, 0.6],
    })
    res = df.metrics(
        y_true="y", y_proba=["p0", "p1", "p2"],
        metrics=["log_loss"], classes=[0, 1, 2],
    )
    assert not math.isnan(res["log_loss"])
    assert res["log_loss"] > 0


def test_metrics_y_proba_none_unchanged(tmp_path):
    """Existing y_proba=None + regression metric still works."""
    df = make_df(tmp_path, {"y": [1.0, 2.0], "yp": [1.0, 2.0]})
    res = df.metrics(y_true="y", y_pred="yp", metrics=["r2"])
    assert res["r2"] == pytest.approx(1.0)
