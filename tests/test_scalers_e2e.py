"""v0.9.2f: scalers E2E (slow)."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import (
    CadwDataFrame, MemoryGuard, cv_score, KFold,
    StandardScaler,
)
from cadw.pipeline import GBDTStep, Pipeline


@pytest.mark.slow
def test_e2e_standard_scaler_100k(tmp_path):
    """100K rows, StandardScaler → GBDTStep, memory 50 MB, R² > 0.5."""
    n = 100_000
    rng = np.random.default_rng(42)
    x1 = rng.normal(loc=100, scale=10, size=n)
    x2 = rng.normal(loc=0.5, scale=0.1, size=n)
    y = (
        2.0 * (x1 - 100) / 10
        + 3.0 * (x2 - 0.5) / 0.1
        + rng.normal(scale=0.3, size=n)
    )

    src = tmp_path / "src.parquet"
    pq.write_table(
        pa.table({
            "x1": x1.tolist(), "x2": x2.tolist(), "y": y.tolist(),
        }),
        src, row_group_size=5000,
    )
    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=5000,
        memory_guard=MemoryGuard(max_memory_mb=50),
        spill_dir=tmp_path / "spill",
    )

    pipe = Pipeline(memory_guard=MemoryGuard(max_memory_mb=50))
    pipe.add_step("scale", StandardScaler(columns=["x1", "x2"]))
    pipe.add_step(
        "model", GBDTStep(
            ["x1", "x2"], "y",
            n_estimators=30, max_depth=2, learning_rate=0.2,
        ),
    )
    pipe.fit(df, target="y")
    preds = pipe.predict(df)
    out = tmp_path / "p.parquet"
    preds.write_parquet(out)

    t = pq.read_table(out)
    y_true = np.array(t.column("y").to_pylist(), dtype=np.float64)
    y_pred = np.array(t.column("y_pred").to_pylist(), dtype=np.float64)
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot
    assert r2 > 0.5, f"R²={r2:.3f}"
