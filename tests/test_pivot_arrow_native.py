"""v0.9.6c2: pivot_table Arrow-native streaming (R6)."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.core.memory_guard import MemoryLimitExceeded


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


# ============ Behaviour preserved (pandas-equivalence) ============


def test_pivot_mean_regression(tmp_path):
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
    r = t.to_pylist()[0]
    assert r["c1"] == 3.0
    assert r["c2"] == 3.0


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
    assert rows["r1"]["c2"] is None
    assert rows["r2"]["c2"] == 1


def test_pivot_missing_combo_null(tmp_path):
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
    assert rows["r1"]["c1"] == 10.0
    assert rows["r1"]["c2"] is None
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


def test_pivot_drops_null_columns(tmp_path):
    df = make_df(tmp_path, {
        "row": ["r1", "r2", "r3"],
        "col": ["c1", None, "c2"],
        "v": [10.0, 20.0, 30.0],
    })
    out = df.pivot_table(
        index=["row"], columns="col", values="v", aggfunc="mean",
    )
    t = read_table(out, tmp_path)
    # r2 had null col -> dropped; only r1 and r3 remain.
    assert t.num_rows == 2


def test_pivot_int_columns_str_names(tmp_path):
    df = make_df(tmp_path, {
        "row": ["r1", "r1"],
        "col": [1, 2],
        "v": [10.0, 20.0],
    })
    out = df.pivot_table(
        index=["row"], columns="col", values="v", aggfunc="mean",
    )
    t = read_table(out, tmp_path)
    assert set(t.schema.names) == {"row", "1", "2"}


def test_pivot_sorted_row_order(tmp_path):
    df = make_df(tmp_path, {
        "row": ["z", "a", "m"],
        "col": ["c1", "c1", "c1"],
        "v": [1.0, 2.0, 3.0],
    })
    out = df.pivot_table(
        index=["row"], columns="col", values="v", aggfunc="mean",
    )
    t = read_table(out, tmp_path)
    assert [r["row"] for r in t.to_pylist()] == ["a", "m", "z"]


def test_pivot_sorted_column_order(tmp_path):
    df = make_df(tmp_path, {
        "row": ["r1", "r1", "r1"],
        "col": ["z", "a", "m"],
        "v": [1.0, 2.0, 3.0],
    })
    out = df.pivot_table(
        index=["row"], columns="col", values="v", aggfunc="mean",
    )
    t = read_table(out, tmp_path)
    # Columns: row, a, m, z (sorted).
    assert t.schema.names == ["row", "a", "m", "z"]


# ============ R6 scale (smaller) ============


def test_pivot_200k_300x300_bounded(tmp_path):
    """R6 scale-down: 20k rows / 300 x 300. Peak bounded."""
    n = 20_000
    rng = np.random.default_rng(42)
    df = make_df(tmp_path, {
        "r": [f"r{i % 300}" for i in range(n)],
        "c": [f"c{i % 300}" for i in range(n)],
        "v": rng.normal(size=n).tolist(),
    }, chunk=2000, mem_mb=100)
    out = df.pivot_table(
        index=["r"], columns="c", values="v", aggfunc="mean",
    )
    t = read_table(out, tmp_path)
    assert t.num_rows == 300
    assert len(t.schema.names) == 1 + 300


# ============ Fail-fast on huge K_columns ============


def test_pivot_huge_k_columns_fails_fast(tmp_path):
    """K_columns so large that output_batch check fails."""
    n = 5_000
    df = make_df(tmp_path, {
        "row": ["r1"] * n,
        "col": [f"c{i}" for i in range(n)],
        "v": [float(i) for i in range(n)],
    }, chunk=500, mem_mb=2)
    # 5k distinct cols × 32 B × 2 (dict) = 320 KB > state_limit (2 MB * 0.5 = 1 MB)
    # OK, that fits. Force smaller budget.
    df._memory_guard = MemoryGuard(max_memory_bytes=100_000)
    # 5k × 32 × 2 = 320 KB > 50 KB state_limit → fail
    with pytest.raises(MemoryLimitExceeded, match="pivot"):
        df.pivot_table(
            index=["row"], columns="col", values="v", aggfunc="mean",
        )


# ============ AST guards ============


def test_pivot_table_no_to_pandas_ast():
    """pivot_table must not call .to_pandas() — AST check."""
    import inspect
    from cadw.patch_utils import assert_no_call_in_source
    from cadw.dataframe import CadwDataFrame
    src = inspect.getsource(CadwDataFrame.pivot_table)
    assert_no_call_in_source(
        src, "to_pandas", fn_name="CadwDataFrame.pivot_table",
    )


def test_pivot_table_no_pd_pivot_ast():
    """pivot_table must not call pd.DataFrame.pivot/pivot_table."""
    import inspect
    from cadw.patch_utils import assert_no_call_in_source
    from cadw.dataframe import CadwDataFrame
    src = inspect.getsource(CadwDataFrame.pivot_table)
    assert_no_call_in_source(
        src, "pivot", "pivot_table",
        fn_name="CadwDataFrame.pivot_table",
    )


def test_pivot_to_disk_uses_external_sort():
    import inspect
    from cadw.ops import pivot as pv
    src = inspect.getsource(pv.pivot_to_disk)
    assert "external_sort" in src
    assert "groupby_aggregate" in src


def test_pivot_to_disk_no_dense_materialize_ast():
    """AST: no dense matrix shape (K_index × K_columns) in code."""
    import ast as _ast
    import inspect
    import textwrap
    from cadw.ops import pivot as pv
    src = inspect.getsource(pv.pivot_to_disk)
    tree = _ast.parse(textwrap.dedent(src))
    # No `to_pandas` call anywhere
    for n in _ast.walk(tree):
        if (isinstance(n, _ast.Call)
                and isinstance(n.func, _ast.Attribute)
                and n.func.attr in ("to_pandas", "from_pandas", "pivot")):
            raise AssertionError(
                f"forbidden Call {n.func.attr}() at line {n.lineno}"
            )
