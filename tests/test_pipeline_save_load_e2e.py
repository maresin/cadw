"""v0.9.1c: E2E save/load — fit → save → load → predict equal."""
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.pipeline import GBDTStep, Pipeline


def _r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


@pytest.mark.slow
def test_e2e_save_load_predictions_equal(tmp_path):
    """Fit GBDT pipeline, save, load, predict. Predictions must match."""
    n = 5000
    rng = np.random.default_rng(42)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = x1 * 3.0 + x2 * 1.5 + rng.normal(scale=0.2, size=n)

    import pyarrow as pa
    src = tmp_path / "src.parquet"
    pq.write_table(
        pa.table({"x1": x1.tolist(), "x2": x2.tolist(), "y": y.tolist()}),
        src, row_group_size=500,
    )
    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=500,
        memory_guard=MemoryGuard(max_memory_mb=100),
        spill_dir=tmp_path / "spill",
    )

    pipe = Pipeline(memory_guard=MemoryGuard(max_memory_mb=100))
    pipe.add_step(
        "model", GBDTStep(
            ["x1", "x2"], "y",
            n_estimators=50, max_depth=3, learning_rate=0.1,
        ),
    )
    pipe.fit(df, target="y")

    # Save
    pipe_dir = tmp_path / "saved_pipeline"
    pipe.to_json(pipe_dir)

    # Load in fresh object
    restored = Pipeline.from_json(
        pipe_dir, memory_guard=MemoryGuard(max_memory_mb=100),
    )

    # Predict with both
    preds_orig = pipe.predict(df)
    preds_restored = restored.predict(df)

    out1 = tmp_path / "p1.parquet"
    out2 = tmp_path / "p2.parquet"
    preds_orig.write_parquet(out1)
    preds_restored.write_parquet(out2)

    t1 = pq.read_table(out1)
    t2 = pq.read_table(out2)
    p1 = t1.column("y_pred").to_pylist()
    p2 = t2.column("y_pred").to_pylist()
    np.testing.assert_allclose(p1, p2, rtol=1e-9, equal_nan=True)

    # Sanity: R² > 0.9 on training
    r2 = _r2(t1.column("y").to_pylist(), p1)
    assert r2 > 0.9
