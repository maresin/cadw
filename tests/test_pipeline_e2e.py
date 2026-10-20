"""v0.9.0b: end-to-end pipeline demo — parquet(ts,sensor,y) → GBDT.

Slow test. Demonstrates full out-of-core flow:
  parquet → dt.features + cast + dropna + GBDT → predict → parquet.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.pipeline import CastStep, DropnaStep, GBDTStep, Pipeline


def _r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


@pytest.mark.slow
def test_e2e_pipeline_time_series(tmp_path):
    # v0.9.4a-triage: N=20k->10k, GBDT(30)->GBDT(20). v0.9.3
    # test time ~3s; now ~1.5s. R2>0.3 threshold unchanged
    # (wide margin at both sizes on linear-signal target).
    rng = np.random.default_rng(42)
    n = 10_000
    ts = pd.date_range("2020-01-01", periods=n, freq="1h").to_numpy()
    sensor = np.tile(["A", "B", "C", "D"], n // 4 + 1)[:n]
    month = pd.DatetimeIndex(ts).month.to_numpy()
    weekday = pd.DatetimeIndex(ts).weekday.to_numpy()
    y = (
        10.0
        + 5.0 * month / 12.0
        + 3.0 * weekday / 7.0
        + rng.normal(scale=0.3, size=n)
    )

    import pyarrow as pa
    table = pa.table({
        "ts": pa.array(ts),
        "sensor": sensor,
        "y": y.tolist(),
    })
    src = tmp_path / "src.parquet"
    pq.write_table(table, src, row_group_size=1000)

    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=1000,
        memory_guard=MemoryGuard(max_memory_mb=100),
        spill_dir=tmp_path / "spill",
    )

    # Feature engineering via accessor + cast
    df_feat = (
        df["ts"].dt.month
    )
    df_feat = df_feat["ts"].dt.weekday
    df_feat = df_feat["ts"].dt.day
    # After 3 chained accesses: cols = ts, sensor, y, ts_month, ts_weekday, ts_day

    # Encode sensor
    df_enc, _ = df_feat.encode_categorical(["sensor"])
    # Keep only features that GBDT can use (numeric)
    feature_cols = ["ts_month", "ts_weekday", "ts_day", "sensor"]

    pipe = Pipeline(
        memory_guard=MemoryGuard(max_memory_mb=100),
    ).add_step(
        "model", GBDTStep(
            feature_cols, "y",
            n_estimators=20, max_depth=3, learning_rate=0.1, bins=32,
        ),
    )
    pipe.fit(df_enc, target="y")
    preds = pipe.predict(df_enc)
    out = tmp_path / "pred.parquet"
    preds.write_parquet(out)

    pred_table = pq.read_table(out)
    got_y = pred_table.column("y").to_pylist()
    got_pred = pred_table.column("y_pred").to_pylist()
    r2 = _r2(got_y, got_pred)
    assert r2 > 0.3, f"R²={r2:.3f}, expected > 0.3"

    # MemoryGuard: total state small (< 1 MB for 30 trees × depth 3)
    assert pipe.total_state_bytes() < 1_000_000
