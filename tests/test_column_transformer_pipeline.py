"""v0.9.1d: ColumnTransformer inside Pipeline + save/load."""
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
from pathlib import Path

from cadw import CadwDataFrame, MemoryGuard
from cadw.pipeline import (
    CastStep, ColumnTransformer, GBDTStep, LambdaTransform, Pipeline,
)


def make_df(tmp_path, data, chunk=50, name="L", mem_mb=200):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def test_ct_inside_pipeline(tmp_path):
    n = 100
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = x1 * 2.0 + x2 * 0.5
    df = make_df(
        tmp_path,
        {"x1": x1.tolist(), "x2": x2.tolist(), "y": y.tolist()},
    )
    ct = ColumnTransformer([
        ("feat1", ["x1"], LambdaTransform(
            "x1", "sq", lambda arr: pc.multiply(arr, arr), pa.float64(),
        )),
        ("feat2", ["x2"], CastStep({"x2": "float64"})),
        ("target", ["y"], CastStep({"y": "float64"})),
    ], remainder="drop")
    pipe = Pipeline(memory_guard=MemoryGuard(max_memory_mb=200))
    pipe.add_step("ct", ct)
    pipe.add_step(
        "model", GBDTStep(["x1_sq", "x2"], "y", n_estimators=10, max_depth=2),
    )
    pipe.fit(df, target="y")
    preds = pipe.predict(df)
    out = tmp_path / "p.parquet"
    preds.write_parquet(out)
    assert "y_pred" in pq.read_schema(out).names


def test_ct_pipeline_save_load(tmp_path):
    n = 100
    rng = np.random.default_rng(0)
    df = make_df(
        tmp_path,
        {"x1": rng.normal(size=n).tolist(),
         "x2": rng.normal(size=n).tolist(),
         "y": (rng.normal(size=n) * 2).tolist()},
    )
    ct = ColumnTransformer([
        ("f", ["x1", "x2"], CastStep({"x1": "float64", "x2": "float64"})),
        ("t", ["y"], CastStep({"y": "float64"})),
    ], remainder="drop")
    pipe = Pipeline(memory_guard=MemoryGuard(max_memory_mb=200))
    pipe.add_step("ct", ct)
    pipe.add_step(
        "model", GBDTStep(["x1", "x2"], "y", n_estimators=5),
    )
    pipe.fit(df, target="y")

    out_dir = tmp_path / "mypipe"
    pipe.to_json(out_dir)
    restored = Pipeline.from_json(out_dir)

    preds_orig = pipe.predict(df)
    preds_restored = restored.predict(df)
    o1 = tmp_path / "p1.parquet"
    o2 = tmp_path / "p2.parquet"
    preds_orig.write_parquet(o1)
    preds_restored.write_parquet(o2)
    p1 = pq.read_table(o1).column("y_pred").to_pylist()
    p2 = pq.read_table(o2).column("y_pred").to_pylist()
    np.testing.assert_allclose(p1, p2, rtol=1e-9, equal_nan=True)


def test_ct_state_bytes_propagates():
    """ColumnTransformer.state_bytes = sum of nested state_bytes."""
    ct = ColumnTransformer([
        ("a", ["x"], CastStep({"x": "float64"})),
        ("b", ["y"], CastStep({"y": "float64"})),
    ])
    # Stateless — 0 both before and after
    from cadw.core.models import DataChars
    dc = DataChars(
        schema=pa.schema([("x", pa.int64()), ("y", pa.int64())]),
        n_rows=100, row_size_bytes=16,
    )
    assert ct.estimate_state_bytes(dc) == 0
    assert ct.state_bytes() == 0


@pytest.mark.slow
def test_e2e_ct_pipeline_realistic(tmp_path):
    """Realistic: 5K rows, num+cat cols, ColumnTransformer + GBDT, R² > 0.5."""
    n = 5000
    rng = np.random.default_rng(42)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n) * 0.5
    cat_int = rng.integers(0, 5, size=n)
    y = x1 * 2.0 + x2 * 1.5 + cat_int * 0.3 + rng.normal(scale=0.2, size=n)

    src = tmp_path / "src.parquet"
    pq.write_table(
        pa.table({
            "x1": x1.tolist(),
            "x2": x2.tolist(),
            "cat": cat_int.tolist(),
            "y": y.tolist(),
        }),
        src, row_group_size=500,
    )
    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=500,
        memory_guard=MemoryGuard(max_memory_mb=100),
        spill_dir=tmp_path / "spill",
    )

    ct = ColumnTransformer([
        ("feat", ["x1", "x2"], CastStep({"x1": "float64", "x2": "float64"})),
        ("cat", ["cat"], CastStep({"cat": "float64"})),
        ("target", ["y"], CastStep({"y": "float64"})),
    ], remainder="drop")

    pipe = Pipeline(memory_guard=MemoryGuard(max_memory_mb=100))
    pipe.add_step("ct", ct)
    pipe.add_step(
        "model", GBDTStep(
            ["x1", "x2", "cat"], "y",
            n_estimators=30, max_depth=3, learning_rate=0.1,
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
