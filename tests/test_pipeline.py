"""v0.9.0b: Pipeline skeleton + plan."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.core.memory_guard import MemoryLimitExceeded
from cadw.core.planner import PipelinePlan, PipelineMemoryError
from cadw.pipeline import (
    CastStep, DropnaStep, GBDTStep, Pipeline, Transformer,
)


def make_df(tmp_path, data, chunk=3, name="L", mem_mb=100):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


# ============ construction ============

def test_empty_pipeline():
    p = Pipeline()
    assert p.steps == []
    assert not p.fitted
    assert "Pipeline([]" in repr(p)


def test_add_step_chaining():
    p = Pipeline()
    result = p.add_step("a", CastStep({"v": "float64"}))
    assert result is p
    assert len(p.steps) == 1


def test_add_step_duplicate_name_raises():
    p = Pipeline()
    p.add_step("a", CastStep({"v": "float64"}))
    with pytest.raises(ValueError, match="duplicate name"):
        p.add_step("a", CastStep({"v": "int64"}))


def test_add_step_bad_step_type_raises():
    p = Pipeline()
    with pytest.raises(TypeError, match="must have fit"):
        p.add_step("a", object())


# ============ wrappers (pure transforms) ============

def test_cast_step_transform(tmp_path):
    df = make_df(tmp_path, {"v": [1, 2, 3]})
    step = CastStep({"v": "float64"})
    out = step.transform(df)
    t = step_out_table(out, tmp_path)
    assert t.schema.field("v").type == pa.float64()
    assert t.column("v").to_pylist() == [1.0, 2.0, 3.0]


def step_out_table(df, tmp_path, name="o.parquet"):
    out = tmp_path / name
    df.write_parquet(out)
    return pq.read_table(out)


def test_dropna_step_transform(tmp_path):
    df = make_df(tmp_path, {
        "a": [1, None, 3],
        "v": [10, 20, 30],
    })
    step = DropnaStep(how="any", subset=["a"])
    out = step.transform(df)
    assert out.count() == 2


def test_compose_cast_dropna(tmp_path):
    df = make_df(tmp_path, {
        "a": [1, None, 3, None],
        "v": [1.0, 2.0, 3.0, 4.0],
    })
    pipe = (
        Pipeline()
        .add_step("cast", CastStep({"v": "float32"}))
        .add_step("dropna", DropnaStep(how="any", subset=["a"]))
    )
    pipe.fit(df)  # no model, so no target
    out = pipe.transform(df)
    assert out.count() == 2
    t = step_out_table(out, tmp_path)
    assert t.schema.field("v").type == pa.float32()


# ============ fit / transform / predict contract ============

def test_fit_returns_self(tmp_path):
    df = make_df(tmp_path, {"v": [1, 2, 3]})
    pipe = Pipeline().add_step("cast", CastStep({"v": "float64"}))
    assert pipe.fit(df) is pipe
    assert pipe.fitted


def test_transform_before_fit_raises(tmp_path):
    df = make_df(tmp_path, {"v": [1, 2, 3]})
    pipe = Pipeline().add_step("cast", CastStep({"v": "float64"}))
    with pytest.raises(RuntimeError, match="not fitted"):
        pipe.transform(df)


def test_predict_without_model_raises(tmp_path):
    df = make_df(tmp_path, {"v": [1, 2, 3]})
    pipe = Pipeline().add_step("cast", CastStep({"v": "float64"}))
    pipe.fit(df)
    with pytest.raises(TypeError, match="Predictor"):
        pipe.predict(df)


def test_fit_requires_target_with_model(tmp_path):
    n = 50
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    y = x * 2.0
    df = make_df(tmp_path, {
        "x": x.tolist(), "y": y.tolist(),
    })
    pipe = Pipeline().add_step(
        "model", GBDTStep(["x"], "y", n_estimators=5),
    )
    with pytest.raises(ValueError, match="target required"):
        pipe.fit(df)  # no target


def test_fit_transform_returns_features(tmp_path):
    df = make_df(tmp_path, {"v": [1, 2, 3]})
    pipe = Pipeline().add_step("cast", CastStep({"v": "float64"}))
    out = pipe.fit_transform(df)
    # No model → fit_transform returns transformed df
    assert out.count() == 3
    assert out.columns == ["v"]


# ============ plan ============

def test_plan_returns_pipeline_plan(tmp_path):
    df = make_df(tmp_path, {"v": [1, 2, 3]})
    pipe = (
        Pipeline()
        .add_step("cast", CastStep({"v": "float64"}))
        .add_step("dropna", DropnaStep())
    )
    plan = pipe.plan(df)
    assert isinstance(plan, PipelinePlan)
    assert plan.predicted_outcome == "fits"
    assert len(plan.steps) == 2
    assert plan.peak_bytes == 0  # pure transforms: 0 state


def test_plan_with_gbdt_state(tmp_path):
    n = 100
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    y = x * 2.0
    df = make_df(tmp_path, {"x": x.tolist(), "y": y.tolist()})
    pipe = Pipeline().add_step(
        "model", GBDTStep(["x"], "y", n_estimators=10, max_depth=3),
    )
    plan = pipe.plan(df)
    assert plan.predicted_outcome == "fits"
    # 10 trees × 15 nodes × 48 = 7200 + 1024
    assert plan.peak_bytes > 0
    assert plan.peak_bytes < 100_000


def test_plan_does_not_fit_raises(tmp_path):
    """Custom step with huge estimate → PipelineMemoryError."""
    df = make_df(tmp_path, {"v": [1, 2, 3]})

    class HugeStep:
        def fit(self, df, target=None):
            return self
        def transform(self, df):
            return df
        def estimate_state_bytes(self, dc):
            return 10 ** 12  # 1 TB
        def state_bytes(self):
            return 0

    pipe = Pipeline(
        memory_guard=MemoryGuard(max_memory_mb=100),
    ).add_step("huge", HugeStep())
    with pytest.raises(PipelineMemoryError):
        pipe.plan(df)
    with pytest.raises(PipelineMemoryError):
        pipe.fit(df)


# ============ GBDT E2E (fast) ============

def test_pipeline_gbdt_predict(tmp_path):
    """Small GBDT pipeline: fit → predict."""
    n = 200
    rng = np.random.default_rng(42)
    x = rng.normal(size=n)
    y = x * 2.0 + rng.normal(scale=0.05, size=n)
    df = make_df(
        tmp_path, {"x": x.tolist(), "y": y.tolist()},
        chunk=50, mem_mb=200,
    )
    pipe = Pipeline(
        memory_guard=MemoryGuard(max_memory_mb=200),
    ).add_step(
        "model", GBDTStep(
            ["x"], "y", n_estimators=20, max_depth=2, learning_rate=0.2,
        ),
    )
    pipe.fit(df, target="y")
    preds = pipe.predict(df)
    out = tmp_path / "pred.parquet"
    preds.write_parquet(out)
    t = pq.read_table(out)
    assert "y_pred" in t.schema.names
    # Simple sanity: predictions within expected range
    pred_vals = t.column("y_pred").to_pylist()
    assert max(abs(p) for p in pred_vals if p is not None) < 10


def test_pipeline_cast_then_gbdt(tmp_path):
    """Cast (int→float) + GBDT. Verify chain works."""
    n = 200
    rng = np.random.default_rng(1)
    x_int = (rng.normal(size=n) * 10).astype(int)
    y = x_int * 1.5
    df = make_df(
        tmp_path, {
            "x": x_int.tolist(),
            "y": y.tolist(),
        },
        chunk=50, mem_mb=200,
    )
    pipe = Pipeline(
        memory_guard=MemoryGuard(max_memory_mb=200),
    ).add_step(
        "cast", CastStep({"x": "float64", "y": "float64"}),
    ).add_step(
        "model", GBDTStep(
            ["x"], "y", n_estimators=10, max_depth=3,
        ),
    )
    pipe.fit(df, target="y")
    preds = pipe.predict(df)
    out = tmp_path / "pred.parquet"
    preds.write_parquet(out)
    t = pq.read_table(out)
    assert "y_pred" in t.schema.names


# ============ total_state_bytes ============

def test_total_state_bytes_zero_before_fit(tmp_path):
    df = make_df(tmp_path, {"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]})
    pipe = Pipeline().add_step(
        "model", GBDTStep(["x"], "y", n_estimators=5),
    )
    assert pipe.total_state_bytes() == 0


def test_total_state_bytes_grows_after_fit(tmp_path):
    n = 100
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    y = x * 2.0
    df = make_df(
        tmp_path, {"x": x.tolist(), "y": y.tolist()},
        chunk=50, mem_mb=200,
    )
    pipe = Pipeline(
        memory_guard=MemoryGuard(max_memory_mb=200),
    ).add_step(
        "model", GBDTStep(["x"], "y", n_estimators=10, max_depth=2),
    )
    before = pipe.total_state_bytes()
    pipe.fit(df, target="y")
    after = pipe.total_state_bytes()
    assert before == 0
    assert after > 0
