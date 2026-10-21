"""v0.9.1f: pivot_table / melt / crosstab tests."""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.core.memory_guard import MemoryLimitExceeded


def make_df(tmp_path, data, chunk=3, name="L", mem_mb=200):
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


# ============ melt ============

def test_melt_basic(tmp_path):
    df = make_df(tmp_path, {
        "id": [1, 2],
        "a": [10, 20],
        "b": [100, 200],
    })
    out = df.melt(id_vars=["id"], value_vars=["a", "b"])
    t = read_table(out, tmp_path)
    assert set(t.schema.names) == {"id", "variable", "value"}
    assert t.num_rows == 4  # 2 ids × 2 value_vars
    rows = t.to_pylist()
    # First two from a (id=1,2), next two from b
    assert [r["variable"] for r in rows] == ["a", "a", "b", "b"]
    assert [r["value"] for r in rows] == [10, 20, 100, 200]


def test_melt_custom_names(tmp_path):
    df = make_df(tmp_path, {"id": [1], "a": [10], "b": [20]})
    out = df.melt(
        id_vars=["id"], value_vars=["a", "b"],
        var_name="src", value_name="val",
    )
    t = read_table(out, tmp_path)
    assert set(t.schema.names) == {"id", "src", "val"}


def test_melt_default_value_vars(tmp_path):
    """value_vars=None → all except id_vars."""
    df = make_df(tmp_path, {"id": [1], "x": [10], "y": [20]})
    out = df.melt(id_vars=["id"])
    t = read_table(out, tmp_path)
    assert set(t.schema.names) == {"id", "variable", "value"}
    assert t.num_rows == 2


def test_melt_mixed_dtype_raises(tmp_path):
    df = make_df(tmp_path, {
        "id": [1], "i": [10], "f": [1.5],
    })
    with pytest.raises(TypeError, match="share one dtype"):
        df.melt(id_vars=["id"], value_vars=["i", "f"])


def test_melt_overlap_raises(tmp_path):
    df = make_df(tmp_path, {"id": [1], "a": [10]})
    with pytest.raises(ValueError, match="both id_vars and value_vars"):
        df.melt(id_vars=["id", "a"], value_vars=["a"])


def test_melt_empty_value_vars_raises(tmp_path):
    df = make_df(tmp_path, {"id": [1], "a": [10]})
    with pytest.raises(ValueError, match="value_vars must be non-empty"):
        df.melt(id_vars=["id", "a"])


# ============ pivot_table ============

def test_pivot_basic_mean(tmp_path):
    df = make_df(tmp_path, {
        "row": ["r1", "r1", "r2", "r2"],
        "col": ["c1", "c2", "c1", "c2"],
        "v": [10.0, 20.0, 30.0, 40.0],
    })
    out = df.pivot_table(
        index=["row"], columns="col", values="v", aggfunc="mean",
    )
    t = read_table(out, tmp_path)
    assert set(t.schema.names) == {"row", "c1", "c2"}
    rows = {r["row"]: (r["c1"], r["c2"]) for r in t.to_pylist()}
    assert rows == {"r1": (10.0, 20.0), "r2": (30.0, 40.0)}


def test_pivot_sum(tmp_path):
    df = make_df(tmp_path, {
        "row": ["r1", "r1", "r1"],
        "col": ["c1", "c1", "c2"],
        "v": [1.0, 2.0, 3.0],
    })
    out = df.pivot_table(
        index=["row"], columns="col", values="v", aggfunc="sum",
    )
    t = read_table(out, tmp_path)
    rows = t.to_pylist()
    assert rows[0]["c1"] == 3.0
    assert rows[0]["c2"] == 3.0


def test_pivot_count(tmp_path):
    df = make_df(tmp_path, {
        "row": ["r1", "r1", "r2"],
        "col": ["c1", "c1", "c2"],
        "v": [10, 20, 30],
    })
    out = df.pivot_table(
        index=["row"], columns="col", values="v", aggfunc="count",
    )
    t = read_table(out, tmp_path)
    rows = {r["row"]: r for r in t.to_pylist()}
    assert rows["r1"]["c1"] == 2
    # r1 c2 missing → null
    assert rows["r1"]["c2"] is None
    assert rows["r2"]["c2"] == 1


def test_pivot_missing_combo_null(tmp_path):
    """Missing (index, columns) combos → null cells."""
    df = make_df(tmp_path, {
        "row": ["r1", "r2"],
        "col": ["c1", "c2"],
        "v": [10.0, 20.0],
    })
    out = df.pivot_table(
        index=["row"], columns="col", values="v", aggfunc="sum",
    )
    t = read_table(out, tmp_path)
    rows = {r["row"]: r for r in t.to_pylist()}
    # r1 c1 present, c2 missing
    assert rows["r1"]["c1"] == 10.0
    assert rows["r1"]["c2"] is None
    # r2 c1 missing, c2 present
    assert rows["r2"]["c1"] is None
    assert rows["r2"]["c2"] == 20.0


def test_pivot_multi_index(tmp_path):
    df = make_df(tmp_path, {
        "a": ["x", "x", "y", "y"],
        "b": ["p", "p", "q", "q"],
        "c": ["c1", "c2", "c1", "c2"],
        "v": [1.0, 2.0, 3.0, 4.0],
    })
    out = df.pivot_table(
        index=["a", "b"], columns="c", values="v", aggfunc="mean",
    )
    t = read_table(out, tmp_path)
    assert set(t.schema.names) == {"a", "b", "c1", "c2"}
    assert t.num_rows == 2


def test_pivot_invalid_aggfunc(tmp_path):
    df = make_df(tmp_path, {"a": [1], "b": [1], "v": [1.0]})
    with pytest.raises(ValueError, match="aggfunc must be"):
        df.pivot_table(
            index=["a"], columns="b", values="v", aggfunc="median",
        )


def test_pivot_too_large_raises(tmp_path):
    """Tight budget → fail-fast BEFORE groupby."""
    n = 1000
    df = make_df(tmp_path, {
        "row": [f"r{i}" for i in range(n)],
        "col": [f"c{i % 100}" for i in range(n)],
        "v": [float(i) for i in range(n)],
    }, chunk=100, mem_mb=1)
    df._memory_guard = MemoryGuard(max_memory_bytes=2000)
    with pytest.raises(MemoryLimitExceeded, match="pivot"):
        df.pivot_table(
            index=["row"], columns="col", values="v", aggfunc="sum",
        )


# ============ crosstab ============

def test_crosstab_basic(tmp_path):
    df = make_df(tmp_path, {
        "g": ["a", "a", "b"],
        "c": ["x", "y", "x"],
    })
    out = df.crosstab(index="g", columns="c")
    t = read_table(out, tmp_path)
    assert set(t.schema.names) == {"g", "x", "y"}
    rows = {r["g"]: r for r in t.to_pylist()}
    assert rows["a"]["x"] == 1
    assert rows["a"]["y"] == 1
    assert rows["b"]["x"] == 1
    assert rows["b"]["y"] is None


def test_crosstab_matches_pivot_count(tmp_path):
    data = {
        "g": ["a", "a", "b", "b", "b"],
        "c": ["x", "y", "x", "y", "y"],
    }
    df = make_df(tmp_path, data)
    ct = read_table(
        df.crosstab(index="g", columns="c"),
        tmp_path, "ct.parquet",
    )
    pt = read_table(
        df.pivot_table(
            index=["g"], columns="c", values="g", aggfunc="count",
        ),
        tmp_path, "pt.parquet",
    )
    # Both should have same shape and values
    ct_rows = sorted(ct.to_pylist(), key=lambda r: r["g"])
    pt_rows = sorted(pt.to_pylist(), key=lambda r: r["g"])
    assert ct_rows == pt_rows


def test_crosstab_drops_nulls(tmp_path):
    df = make_df(tmp_path, {
        "g": ["a", "a", None],
        "c": ["x", "y", "x"],
    })
    out = df.crosstab(index="g", columns="c")
    t = read_table(out, tmp_path)
    # Row with null g is dropped → only "a" row
    assert t.num_rows == 1
    assert t.to_pylist()[0]["g"] == "a"


# ============ AST guards ============

def test_pivot_table_delegates_to_streaming():
    """v0.9.6c2: pivot_table delegates to streaming pivot_to_disk.

    Old contract (v0.9.1f): dense pivot_table had check_state before
    to_pandas. New streaming design has no `to_pandas` at all —
    facade delegates to `pivot_to_disk`, which guards per-step
    (`check_state` before external_sort, before output writer).
    Guard-by-delegation: pivot_table is thin validation + call.
    """
    import ast as _ast
    import inspect
    import textwrap

    from cadw.dataframe import CadwDataFrame

    src = textwrap.dedent(inspect.getsource(CadwDataFrame.pivot_table))
    tree = _ast.parse(src)

    # 1. Delegates to pivot_to_disk (bare Name Call).
    call_names = {
        n.func.id for n in _ast.walk(tree)
        if isinstance(n, _ast.Call) and isinstance(n.func, _ast.Name)
    }
    assert "pivot_to_disk" in call_names, (
        "pivot_table must delegate to cadw.ops.pivot.pivot_to_disk "
        "(v0.9.6c2 streaming)"
    )

    # 2. No pandas-pivot (attr Calls).
    attr_names = {
        n.func.attr for n in _ast.walk(tree)
        if isinstance(n, _ast.Call) and isinstance(n.func, _ast.Attribute)
    }
    assert "to_pandas" not in attr_names, (
        "pivot_table must not call to_pandas (v0.9.6c2 streaming)"
    )
    assert "pivot" not in attr_names, (
        "pivot_table must not call pd.pivot (v0.9.6c2 streaming)"
    )


def test_pivot_to_disk_guarded_before_output():
    """v0.9.6c2: pivot_to_disk has check_state Call before
    ArrowIPCBatchWriter Call (guard before output allocation).

    Uses patch_utils.assert_call_order (AST-based, v0.9.6d
    scaffold) — the helper is dogfooded here on the new streaming
    implementation (§13.4/183, /185).
    """
    import inspect

    from cadw.patch_utils import assert_call_order
    from cadw.ops.pivot import pivot_to_disk

    src = inspect.getsource(pivot_to_disk)
    assert_call_order(
        src, "check_state", "ArrowIPCBatchWriter",
        fn_name="pivot_to_disk",
    )



def test_melt_is_streaming():
    """AST: melt yields per batch (no materialize)."""
    import inspect
    from cadw.dataframe import CadwDataFrame
    src = inspect.getsource(CadwDataFrame.melt)
    assert "yield" in src
    # No to_pandas
    assert "to_pandas" not in src
