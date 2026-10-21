"""v0.9.1b: DAG Pipeline tests."""
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
from pathlib import Path

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


def read_table(df, tmp_path, name="o.parquet"):
    out = tmp_path / name
    df.write_parquet(out)
    return pq.read_table(out)


# ============ backward compat: sequence still works ============

def test_sequence_still_works(tmp_path):
    """inputs=None → chain semantics (v0.9.0 behaviour preserved)."""
    df = make_df(tmp_path, {"v": [1, 2, 3]})
    pipe = (
        Pipeline()
        .add_step("cast", CastStep({"v": "float64"}))
        .add_step("dropna", DropnaStep())
    )
    pipe.fit(df)
    out = pipe.transform(df)
    assert out.count() == 3


def test_inputs_none_chains_to_prev(tmp_path):
    df = make_df(tmp_path, {"v": [1, 2, 3]})
    pipe = Pipeline()
    pipe.add_step("a", CastStep({"v": "float64"}))
    pipe.add_step("b", DropnaStep())  # implicitly inputs=['a']
    graph = pipe._build_graph()
    assert graph["a"] == []
    assert graph["b"] == ["a"]


# ============ fan-in basics ============

def test_dag_two_source_branches_merge(tmp_path):
    """a and b both source → merge fan-in → union columns."""
    df = make_df(tmp_path, {"x": [1, 2, 3, 4]})

    pipe = Pipeline()
    pipe.add_step(
        "a", LambdaTransform(
            "x", "plus1", lambda arr: pc.add(arr, 1), pa.int64(),
        ), inputs=[],
    )
    pipe.add_step(
        "b", LambdaTransform(
            "x", "times2", lambda arr: pc.multiply(arr, 2), pa.int64(),
        ), inputs=[],
    )
    pipe.add_step("merge", DropnaStep(), inputs=["a", "b"])
    pipe.fit(df)
    out = pipe.transform(df)
    t = read_table(out, tmp_path)
    # Columns: x, x_plus1, x_times2 (x dedup, source col once)
    cols = t.schema.names
    assert "x" in cols
    assert "x_plus1" in cols
    assert "x_times2" in cols
    assert cols.count("x") == 1
    # Verify values
    assert t.column("x_plus1").to_pylist() == [2, 3, 4, 5]
    assert t.column("x_times2").to_pylist() == [2, 4, 6, 8]


def test_dag_three_branches(tmp_path):
    df = make_df(tmp_path, {"x": [10, 20]})
    pipe = Pipeline()
    pipe.add_step(
        "a", LambdaTransform(
            "x", "add1", lambda arr: pc.add(arr, 1), pa.int64(),
        ), inputs=[],
    )
    pipe.add_step(
        "b", LambdaTransform(
            "x", "add2", lambda arr: pc.add(arr, 2), pa.int64(),
        ), inputs=[],
    )
    pipe.add_step(
        "c", LambdaTransform(
            "x", "add3", lambda arr: pc.add(arr, 3), pa.int64(),
        ), inputs=[],
    )
    pipe.add_step("merge", DropnaStep(), inputs=["a", "b", "c"])
    pipe.fit(df)
    t = read_table(pipe.transform(df), tmp_path)
    assert t.column("x_add1").to_pylist() == [11, 21]
    assert t.column("x_add2").to_pylist() == [12, 22]
    assert t.column("x_add3").to_pylist() == [13, 23]


def test_dag_diamond(tmp_path):
    """raw → a, raw → b, merge → model. Predict via full DAG."""
    n = 100
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    y = x * 2.0
    df = make_df(
        tmp_path, {"x": x.tolist(), "y": y.tolist()},
        chunk=20, mem_mb=200,
    )

    pipe = Pipeline(
        memory_guard=MemoryGuard(max_memory_mb=200),
    )
    pipe.add_step(
        "a", LambdaTransform(
            "x", "sq", lambda arr: pc.multiply(arr, arr), pa.float64(),
        ), inputs=[],
    )
    pipe.add_step(
        "b", LambdaTransform(
            "x", "abs", lambda arr: pc.abs(arr), pa.float64(),
        ), inputs=[],
    )
    pipe.add_step(
        "model", GBDTStep(
            ["x_sq", "x_abs"], "y",
            n_estimators=10, max_depth=2,
        ),
        inputs=["a", "b"],
    )
    pipe.fit(df, target="y")
    preds = pipe.predict(df)
    t = read_table(preds, tmp_path)
    assert "y_pred" in t.schema.names


# ============ fan-in row count mismatch ============

def test_dag_fan_in_row_mismatch_raises(tmp_path):
    """dropna in one branch drops rows → count mismatch → RuntimeError."""
    df = make_df(tmp_path, {
        "x": [1, 2, 3, 4],
        "y": [10, None, 20, None],   # dropna drops 2 rows
    })
    pipe = Pipeline()
    pipe.add_step("a", DropnaStep(how="any", subset=["y"]), inputs=[])
    pipe.add_step("b", CastStep({"x": "float64"}), inputs=[])
    pipe.add_step("merge", DropnaStep(), inputs=["a", "b"])
    with pytest.raises(RuntimeError, match="row count mismatch"):
        pipe.fit(df)


# ============ validation ============

def test_add_step_unknown_input_raises():
    pipe = Pipeline()
    pipe.add_step("a", CastStep({"v": "float64"}))
    with pytest.raises(ValueError, match="unknown step"):
        pipe.add_step("b", DropnaStep(), inputs=["nope"])


def test_add_step_predictor_as_input_raises(tmp_path):
    df = make_df(
        tmp_path,
        {"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.0, 6.0, 8.0]},
    )
    pipe = Pipeline(memory_guard=MemoryGuard(max_memory_mb=200))
    pipe.add_step("model", GBDTStep(["x"], "y", n_estimators=5))
    with pytest.raises(RuntimeError, match="cannot depend on Predictor"):
        pipe.add_step("post", DropnaStep(), inputs=["model"])


def test_add_step_bad_inputs_type_raises():
    pipe = Pipeline()
    pipe.add_step("a", CastStep({"v": "float64"}))
    with pytest.raises(TypeError, match="inputs must be list"):
        pipe.add_step("b", DropnaStep(), inputs="a")  # str, not list


def test_add_step_multiple_sources_allowed():
    """v0.9.1b-fix2 (§13.4/149): inputs=[] valid on ANY step.

    Multiple sources from raw is the DAG feature (fanout). Prior
    v0.9.1b encoded incorrect validation "explicit source only for
    first step" — removed in v0.9.1b-fix1. This test locks in the
    CORRECT behavior, derived from the API contract (docstring
    `add_step`: "inputs=[]: explicit source ... can be on ANY step"),
    not from implementation.
    """
    pipe = Pipeline()
    pipe.add_step("a", CastStep({"v": "float64"}))
    # No raise — inputs=[] on non-first step is valid (source).
    pipe.add_step("b", DropnaStep(), inputs=[])
    graph = pipe._build_graph()
    assert graph["a"] == []
    assert graph["b"] == []  # both source from raw df


# ============ topology ============

def test_topological_order_chain():
    pipe = Pipeline()
    pipe.add_step("a", CastStep({"v": "float64"}))
    pipe.add_step("b", DropnaStep())
    pipe.add_step("c", CastStep({"v": "int64"}))
    order = pipe._topological_order()
    assert order == ["a", "b", "c"]


def test_topological_order_fan_out():
    pipe = Pipeline()
    pipe.add_step(
        "a", LambdaTransform(
            "x", "p1", lambda arr: arr, pa.int64(),
        ), inputs=[],
    )
    pipe.add_step(
        "b", LambdaTransform(
            "x", "p2", lambda arr: arr, pa.int64(),
        ), inputs=[],
    )
    pipe.add_step("merge", DropnaStep(), inputs=["a", "b"])
    order = pipe._topological_order()
    # merge must come after a and b
    assert order.index("merge") > order.index("a")
    assert order.index("merge") > order.index("b")


# ============ transform/predict DAG semantics ============

def test_transform_stops_at_model(tmp_path):
    n = 50
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    y = x * 2.0
    df = make_df(tmp_path, {"x": x.tolist(), "y": y.tolist()})
    pipe = Pipeline(memory_guard=MemoryGuard(max_memory_mb=200))
    pipe.add_step(
        "feat", LambdaTransform(
            "x", "sq", lambda arr: pc.multiply(arr, arr), pa.float64(),
        ),
    )
    pipe.add_step("model", GBDTStep(["x_sq"], "y", n_estimators=5))
    pipe.fit(df, target="y")
    features = pipe.transform(df)
    # features: x, y, x_sq
    assert "x_sq" in features.columns
    assert "y_pred" not in features.columns


def test_predict_runs_full_dag(tmp_path):
    n = 50
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    y = x * 2.0
    df = make_df(tmp_path, {"x": x.tolist(), "y": y.tolist()})
    pipe = Pipeline(memory_guard=MemoryGuard(max_memory_mb=200))
    pipe.add_step(
        "feat", LambdaTransform(
            "x", "sq", lambda arr: pc.multiply(arr, arr), pa.float64(),
        ),
    )
    pipe.add_step("model", GBDTStep(["x_sq"], "y", n_estimators=5))
    pipe.fit(df, target="y")
    preds = pipe.predict(df)
    assert "y_pred" in preds.columns


# ============ repr_html DAG edges ============

def test_repr_html_shows_dag_edges(tmp_path):
    df = make_df(tmp_path, {"x": [1, 2, 3]})
    pipe = Pipeline()
    pipe.add_step("a", CastStep({"v": "float64"}), inputs=[])
    pipe.add_step("b", DropnaStep(), inputs=["a"])
    html = pipe._repr_html_()
    assert "a" in html
    assert "b" in html
    # HTML contains arrow direction marker
    assert "&larr;" in html or "<-" in html


def test_merge_fanout_duplicate_new_column_raises(tmp_path):
    """v0.9.9 (N5) + fix1c: duplicate new column among dfs[1:].

    F5 contract: _merge_fanout treats dfs[0] as base (chain semantics).
    Guard fires when a NEW column (not in base) appears in TWO branches
    among dfs[1:]. Honest limitation (§11.5): if the same feature-name
    is produced by dfs[0] and another branch, base wins silently —
    _merge_fanout can't distinguish "was in raw" from "added by dfs[0]"
    without explicit raw_columns (backlog v0.9.10).

    F5-covered scenario: a=base (passthrough x), b produces x_same,
    c produces x_same → x_same is new (not in a) and duplicated
    between b and c → RuntimeError.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    import pytest
    from cadw import CadwDataFrame, MemoryGuard
    from cadw.pipeline import LambdaTransform, Pipeline
    t = pa.table({'x': [1, 2, 3, 4]})
    p = tmp_path / 'src.parquet'
    pq.write_table(t, p)
    df = CadwDataFrame.from_parquet(
        p, tmp_path / 'ds', chunk_size=2,
        memory_guard=MemoryGuard(max_memory_mb=100),
        spill_dir=tmp_path / 'spill',
    )
    pipe = Pipeline()
    # Base branch: just passthrough (x only, no new cols).
    pipe.add_step(
        'base', __import__('cadw.pipeline', fromlist=['CastStep']).CastStep({'x': 'int64'}),
        inputs=[],
    )
    # Two siblings both produce x_same → duplicate new column.
    pipe.add_step('b', LambdaTransform(
        'x', 'same', lambda arr: pc.add(arr, 1), pa.int64(),
    ), inputs=[])
    pipe.add_step('c', LambdaTransform(
        'x', 'same', lambda arr: pc.add(arr, 2), pa.int64(),
    ), inputs=[])
    pipe.add_step('merge', __import__('cadw.pipeline', fromlist=['DropnaStep']).DropnaStep(),
                  inputs=['base', 'b', 'c'])
    # F5 raises in Pipeline.fit (line 603: _merge_fanout for fan-in),
    # not transform. Wrap fit in raises.
    with pytest.raises(RuntimeError, match='duplicate new column'):
        pipe.fit(df)