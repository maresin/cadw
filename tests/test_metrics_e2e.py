"""v0.9.2a: metrics E2E (slow)."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard


@pytest.mark.slow
def test_metrics_100k_regression(tmp_path):
    """100K rows — regression metrics (r2/rmse/mae), continuous y.

    §13.4/163: separate regression semantics (continuous y) from
    log_loss semantics (binary y). Prior test mixed both in one call —
    log_loss formula produced invalid negative output for continuous y.
    """
    n = 100_000
    rng = np.random.default_rng(42)
    y = rng.normal(size=n)
    yp = y + rng.normal(scale=0.3, size=n)

    src = tmp_path / "src.parquet"
    pq.write_table(
        pa.table({"y": y.tolist(), "yp": yp.tolist()}),
        src, row_group_size=5000,
    )
    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=5000,
        memory_guard=MemoryGuard(max_memory_mb=30),
        spill_dir=tmp_path / "spill",
    )
    res = df.metrics(
        y_true="y", y_pred="yp", metrics=["r2", "rmse", "mae"],
    )

    ss_res = float(((y - yp) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    ref_r2 = 1.0 - ss_res / ss_tot
    ref_rmse = float(np.sqrt(ss_res / n))
    ref_mae = float(np.abs(y - yp).mean())

    assert res["r2"] == pytest.approx(ref_r2, rel=1e-9)
    assert res["rmse"] == pytest.approx(ref_rmse, rel=1e-9)
    assert res["mae"] == pytest.approx(ref_mae, rel=1e-9)


@pytest.mark.slow
def test_metrics_100k_log_loss_binary(tmp_path):
    """100K rows — log_loss with binary y.

    Binary y_bin = (y > 0); proba p = sigmoid(y) — calibrated-ish.
    """
    n = 100_000
    rng = np.random.default_rng(42)
    y = rng.normal(size=n)
    y_bin = (y > 0).astype(np.float64)
    p = 1.0 / (1.0 + np.exp(-y))  # sigmoid(y) — well-calibrated

    src = tmp_path / "src.parquet"
    pq.write_table(
        pa.table({"yb": y_bin.tolist(), "p": p.tolist()}),
        src, row_group_size=5000,
    )
    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=5000,
        memory_guard=MemoryGuard(max_memory_mb=30),
        spill_dir=tmp_path / "spill",
    )
    res = df.metrics(y_true="yb", y_proba="p", metrics=["log_loss"])

    # Reference (numpy, matches sklearn)
    eps = 1e-15
    p_c = np.clip(p, eps, 1.0 - eps)
    ref_ll = float(-(y_bin * np.log(p_c)
                     + (1 - y_bin) * np.log(1 - p_c)).mean())

    assert res["log_loss"] == pytest.approx(ref_ll, rel=1e-9)
    assert 0 < res["log_loss"] < 1.0  # binary CE ∈ (0, ∞); sigmoid-calibrated ~0.35
