"""v0.9.1c: Pipeline save/load tests."""
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.pipeline import (
    CastStep, DropnaStep, GBDTStep, LambdaTransform, Pipeline,
)


def make_df(tmp_path, data, chunk=5, name="L", mem_mb=200):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


# ============ CastStep / DropnaStep round-trip ============

def test_caststep_round_trip():
    step = CastStep({"x": "float64", "y": "int32"}, safe=False)
    d = step.to_dict()
    restored = CastStep.from_dict(d)
    assert restored._mapping == step._mapping
    assert restored._safe is False


def test_dropnastep_round_trip():
    step = DropnaStep(how="all", subset=["a", "b"])
    d = step.to_dict()
    restored = DropnaStep.from_dict(d)
    assert restored._how == "all"
    assert restored._subset == ["a", "b"]


# ============ GBDT round-trip ============

def test_gbdtstep_round_trip_predictions(tmp_path):
    n = 200
    rng = np.random.default_rng(42)
    x = rng.normal(size=n)
    y = x * 2.0 + rng.normal(scale=0.05, size=n)
    df = make_df(
        tmp_path, {"x": x.tolist(), "y": y.tolist()},
        chunk=50, mem_mb=200,
    )
    step = GBDTStep(["x"], "y", n_estimators=10, max_depth=2)
    step.fit(df, target="y")

    d = step.to_dict()
    restored = GBDTStep.from_dict(d)

    preds_orig = step.predict(df)
    preds_restored = restored.predict(df)

    orig_list = preds_orig._dataset.schema  # just smoke
    out_orig = tmp_path / "o1.parquet"
    out_rest = tmp_path / "o2.parquet"
    preds_orig.write_parquet(out_orig)
    preds_restored.write_parquet(out_rest)
    t1 = pq.read_table(out_orig)
    t2 = pq.read_table(out_rest)
    p1 = t1.column("y_pred").to_pylist()
    p2 = t2.column("y_pred").to_pylist()
    np.testing.assert_allclose(p1, p2, rtol=1e-9, equal_nan=True)


def test_gbdtstep_unfitted_round_trip():
    step = GBDTStep(["x"], "y", n_estimators=5)
    d = step.to_dict()
    assert d["state"] is None
    restored = GBDTStep.from_dict(d)
    assert restored._model is None


# ============ Pipeline round-trip ============

def test_pipeline_to_json_creates_dir(tmp_path):
    pipe = Pipeline()
    pipe.add_step("cast", CastStep({"x": "float64"}))
    out_path = tmp_path / "mypipe"
    result = pipe.to_json(out_path)
    assert isinstance(result, Path)
    assert (out_path / "pipeline.json").exists()
    assert result == out_path / "pipeline.json"


def test_pipeline_round_trip_stateless(tmp_path):
    pipe = Pipeline()
    pipe.add_step("cast", CastStep({"x": "float64"}))
    pipe.add_step("dropna", DropnaStep(how="any", subset=["x"]))
    out_dir = tmp_path / "mypipe"
    pipe.to_json(out_dir)

    restored = Pipeline.from_json(out_dir)
    assert len(restored.steps) == 2
    assert restored._step_names == ["cast", "dropna"]
    # Params preserved
    cast_step = restored._steps[0]
    assert isinstance(cast_step, CastStep)
    assert cast_step._mapping == pipe._steps[0]._mapping


def test_pipeline_round_trip_dag_edges(tmp_path):
    pipe = Pipeline()
    pipe.add_step("a", CastStep({"x": "float64"}), inputs=[])
    pipe.add_step("b", CastStep({"x": "int64"}), inputs=[])
    pipe.add_step("merge", DropnaStep(), inputs=["a", "b"])
    out_dir = tmp_path / "mypipe"
    pipe.to_json(out_dir)

    restored = Pipeline.from_json(out_dir)
    graph = restored._build_graph()
    assert graph["a"] == []
    assert graph["b"] == []
    assert graph["merge"] == ["a", "b"]


def test_pipeline_round_trip_state_file_created(tmp_path):
    n = 100
    rng = np.random.default_rng(0)
    df = make_df(
        tmp_path, {"x": rng.normal(size=n).tolist(),
                   "y": (rng.normal(size=n) * 2).tolist()},
        chunk=50, mem_mb=200,
    )
    pipe = Pipeline(memory_guard=MemoryGuard(max_memory_mb=200))
    pipe.add_step("model", GBDTStep(["x"], "y", n_estimators=5))
    pipe.fit(df, target="y")

    out_dir = tmp_path / "mypipe"
    pipe.to_json(out_dir)
    assert (out_dir / "state" / "model.json").exists()


def test_pipeline_from_json_without_state_dir(tmp_path):
    """Stateless pipeline: state/ dir not created, from_json works."""
    pipe = Pipeline()
    pipe.add_step("cast", CastStep({"x": "float64"}))
    out_dir = tmp_path / "mypipe"
    pipe.to_json(out_dir)
    assert not (out_dir / "state").exists()
    restored = Pipeline.from_json(out_dir)
    assert restored._step_names == ["cast"]


def test_pipeline_round_trip_fitted_flag(tmp_path):
    n = 100
    rng = np.random.default_rng(0)
    df = make_df(
        tmp_path, {"x": rng.normal(size=n).tolist(),
                   "y": (rng.normal(size=n) * 2).tolist()},
        chunk=50, mem_mb=200,
    )
    pipe = Pipeline(memory_guard=MemoryGuard(max_memory_mb=200))
    pipe.add_step("model", GBDTStep(["x"], "y", n_estimators=5))
    pipe.fit(df, target="y")
    assert pipe.fitted

    out_dir = tmp_path / "mypipe"
    pipe.to_json(out_dir)
    restored = Pipeline.from_json(out_dir)
    assert restored.fitted


def test_pipeline_from_json_path_is_file(tmp_path):
    """from_json accepts pipeline.json file directly."""
    pipe = Pipeline()
    pipe.add_step("cast", CastStep({"x": "float64"}))
    out_dir = tmp_path / "mypipe"
    out_file = pipe.to_json(out_dir)

    restored = Pipeline.from_json(out_file)
    assert restored._step_names == ["cast"]


# ============ Errors ============

def test_pipeline_to_json_lambda_raises(tmp_path):
    pipe = Pipeline()
    pipe.add_step(
        "lambda", LambdaTransform(
            "x", "p", lambda arr: arr, pa.int64(),
        ),
        inputs=[],
    )
    out_dir = tmp_path / "mypipe"
    with pytest.raises(NotImplementedError, match="LambdaTransform"):
        pipe.to_json(out_dir)


def test_pipeline_from_json_corrupt_raises(tmp_path):
    out_dir = tmp_path / "mypipe"
    out_dir.mkdir()
    (out_dir / "pipeline.json").write_text("{not valid json")
    with pytest.raises(ValueError, match="not valid JSON"):
        Pipeline.from_json(out_dir)


def test_pipeline_from_json_missing_state_file_raises(tmp_path):
    n = 100
    rng = np.random.default_rng(0)
    df = make_df(
        tmp_path, {"x": rng.normal(size=n).tolist(),
                   "y": (rng.normal(size=n) * 2).tolist()},
        chunk=50, mem_mb=200,
    )
    pipe = Pipeline(memory_guard=MemoryGuard(max_memory_mb=200))
    pipe.add_step("model", GBDTStep(["x"], "y", n_estimators=5))
    pipe.fit(df, target="y")
    out_dir = tmp_path / "mypipe"
    pipe.to_json(out_dir)
    # Delete state file
    (out_dir / "state" / "model.json").unlink()
    with pytest.raises(FileNotFoundError, match="state file"):
        Pipeline.from_json(out_dir)


def test_pipeline_from_json_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="pipeline.json"):
        Pipeline.from_json(tmp_path / "nope")


def test_pipeline_to_json_path_is_file_raises(tmp_path):
    f = tmp_path / "existing.txt"
    f.write_text("hello")
    pipe = Pipeline()
    pipe.add_step("cast", CastStep({"x": "float64"}))
    with pytest.raises(ValueError, match="is a file; expected directory"):
        pipe.to_json(f)


def test_pipeline_version_mismatch_warns(tmp_path):
    pipe = Pipeline()
    pipe.add_step("cast", CastStep({"x": "float64"}))
    out_dir = tmp_path / "mypipe"
    pipe.to_json(out_dir)
    # Corrupt version
    meta = json.loads((out_dir / "pipeline.json").read_text())
    meta["version"] = "0.0.0"
    (out_dir / "pipeline.json").write_text(json.dumps(meta))

    with pytest.warns(RuntimeWarning, match="version"):
        Pipeline.from_json(out_dir)


def test_pipeline_version_mismatch_strict_raises(tmp_path):
    pipe = Pipeline()
    pipe.add_step("cast", CastStep({"x": "float64"}))
    out_dir = tmp_path / "mypipe"
    pipe.to_json(out_dir)
    meta = json.loads((out_dir / "pipeline.json").read_text())
    meta["version"] = "0.0.0"
    (out_dir / "pipeline.json").write_text(json.dumps(meta))

    with pytest.raises(ValueError, match="version"):
        Pipeline.from_json(out_dir, strict_version=True)
