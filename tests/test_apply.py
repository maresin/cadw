"""v0.9.1e: per-group apply(fn) tests."""
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


def test_apply_scalar_mean(tmp_path):
    df = make_df(tmp_path, {
        "g": ["a", "a", "b", "b"], "v": [10, 20, 100, 200],
    })
    out = df.groupby("g").apply(lambda g: g["v"].mean())
    t = read_table(out, tmp_path)
    rows = {r["g"]: r["apply_result"] for r in t.to_pylist()}
    assert rows == {"a": 15.0, "b": 150.0}
    assert t.schema.names == ["g", "apply_result"]


def test_apply_scalar_custom_result_col(tmp_path):
    df = make_df(tmp_path, {"g": ["a", "a", "b"], "v": [10, 20, 30]})
    out = df.groupby("g").apply(
        lambda g: g["v"].max(), result_col="max_v",
    )
    t = read_table(out, tmp_path)
    assert t.schema.names == ["g", "max_v"]


def test_apply_scalar_int_return(tmp_path):
    df = make_df(tmp_path, {
        "g": ["a", "a", "b", "b", "b"], "v": [1, 2, 3, 4, 5],
    })
    out = df.groupby("g").apply(lambda g: len(g))
    t = read_table(out, tmp_path)
    rows = {r["g"]: r["apply_result"] for r in t.to_pylist()}
    assert rows == {"a": 2, "b": 3}
    assert t.schema.field("apply_result").type == pa.int64()


def test_apply_series_multi_output(tmp_path):
    df = make_df(tmp_path, {
        "g": ["a", "a", "b", "b"], "v": [10, 20, 100, 200],
    })
    out = df.groupby("g").apply(
        lambda g: pd.Series({"mean": g["v"].mean(),
                             "max": g["v"].max()}),
    )
    t = read_table(out, tmp_path)
    assert set(t.schema.names) == {"g", "mean", "max"}
    rows = {r["g"]: (r["mean"], r["max"]) for r in t.to_pylist()}
    assert rows == {"a": (15.0, 20), "b": (150.0, 200)}


def test_apply_dataframe_expand(tmp_path):
    df = make_df(tmp_path, {
        "g": ["a", "a", "a", "b", "b"], "v": [1, 2, 3, 10, 20],
    })
    def fn(g):
        return pd.DataFrame({
            "v_rank": np.arange(1, len(g) + 1),
            "v_shift": g["v"].tolist(),
        })
    out = df.groupby("g").apply(fn)
    t = read_table(out, tmp_path)
    assert set(t.schema.names) == {"g", "v_rank", "v_shift"}
    assert t.num_rows == 5


def test_apply_null_group(tmp_path):
    df = make_df(tmp_path, {
        "g": ["a", "a", None, None], "v": [10, 20, 30, 40],
    })
    out = df.groupby("g").apply(lambda g: g["v"].sum())
    t = read_table(out, tmp_path)
    assert t.num_rows == 2


def test_apply_single_row_groups(tmp_path):
    df = make_df(tmp_path, {"g": ["a", "b", "c"], "v": [10, 20, 30]})
    out = df.groupby("g").apply(lambda g: g["v"].iloc[0])
    t = read_table(out, tmp_path)
    rows = {r["g"]: r["apply_result"] for r in t.to_pylist()}
    assert rows == {"a": 10, "b": 20, "c": 30}


def test_apply_multi_group(tmp_path):
    df = make_df(tmp_path, {
        "s1": ["a", "a", "b", "b"],
        "s2": ["x", "y", "x", "y"],
        "v": [1, 2, 3, 4],
    })
    out = df.groupby(["s1", "s2"]).apply(lambda g: g["v"].sum())
    t = read_table(out, tmp_path)
    assert t.num_rows == 4
    assert set(t.schema.names) == {"s1", "s2", "apply_result"}


def test_apply_after_filter(tmp_path):
    df = make_df(tmp_path, {
        "g": ["a", "a", "a", "b", "b"], "v": [1, 2, 3, 4, 5],
    })
    filtered = df[df["v"] > 1]
    out = filtered.groupby("g").apply(lambda g: g["v"].sum())
    t = read_table(out, tmp_path)
    rows = {r["g"]: r["apply_result"] for r in t.to_pylist()}
    assert rows == {"a": 5, "b": 9}


def test_apply_empty_input(tmp_path):
    df = make_df(tmp_path, {"g": ["a"], "v": [1]})
    empty = df[df["v"] > 100]
    out = empty.groupby("g").apply(lambda g: g["v"].sum())
    t = read_table(out, tmp_path)
    assert t.num_rows == 0


def test_apply_on_col_within_group(tmp_path):
    df = make_df(tmp_path, {
        "g": ["a", "a", "a"], "ts": [3, 1, 2], "v": [30, 10, 20],
    })
    out = df.groupby("g").apply(
        lambda g: g["v"].iloc[0], on="ts",
    )
    t = read_table(out, tmp_path)
    assert t.to_pylist()[0]["apply_result"] == 10


def test_apply_deterministic(tmp_path):
    df = make_df(tmp_path, {
        "g": ["b", "a", "b", "a"], "v": [1, 2, 3, 4],
    })
    t1 = read_table(
        df.groupby("g").apply(lambda g: g["v"].sum()),
        tmp_path, "o1.parquet",
    )
    t2 = read_table(
        df.groupby("g").apply(lambda g: g["v"].sum()),
        tmp_path, "o2.parquet",
    )
    assert t1.to_pylist() == t2.to_pylist()


def test_apply_fn_not_callable_raises(tmp_path):
    df = make_df(tmp_path, {"g": ["a", "a"], "v": [1, 2]})
    with pytest.raises(TypeError, match="must be callable"):
        df.groupby("g").apply("not_a_function")


def test_apply_mixed_modes_raises(tmp_path):
    df = make_df(tmp_path, {
        "g": ["a", "a", "b", "b"], "v": [1, 2, 3, 4],
    })
    call_count = [0]
    def fn(g):
        call_count[0] += 1
        if call_count[0] == 1:
            return g["v"].sum()  # scalar
        return pd.Series({"x": 1})  # series
    with pytest.raises(RuntimeError, match="mixed return modes"):
        df.groupby("g").apply(fn)


def test_apply_unsupported_return_type_raises(tmp_path):
    df = make_df(tmp_path, {"g": ["a", "a"], "v": [1, 2]})
    with pytest.raises(TypeError, match="expected scalar"):
        df.groupby("g").apply(lambda g: [1, 2, 3])


def test_apply_group_too_large_raises(tmp_path):
    df = make_df(
        tmp_path, {"g": ["a"] * 500, "v": list(range(500))},
        chunk=100, mem_mb=1,
    )
    df._memory_guard = MemoryGuard(max_memory_bytes=2000)
    with pytest.raises(MemoryLimitExceeded):
        df.groupby("g").apply(lambda g: g["v"].sum())


def test_apply_fn_exception_propagates(tmp_path):
    df = make_df(tmp_path, {"g": ["a", "a"], "v": [1, 2]})
    def bad_fn(g):
        raise ValueError("user error")
    with pytest.raises(ValueError, match="user error"):
        df.groupby("g").apply(bad_fn)


def test_apply_scalar_pandas_equivalence(tmp_path):
    n = 100
    rng = np.random.default_rng(42)
    data = {
        "g": [f"k{i % 5}" for i in range(n)],
        "v": rng.integers(0, 100, size=n).tolist(),
    }
    df = make_df(tmp_path, data, chunk=20)
    out = df.groupby("g").apply(lambda g: g["v"].mean())
    t = read_table(out, tmp_path)

    pdf = pd.DataFrame(data)
    expected = pdf.groupby("g")["v"].mean().reset_index()
    expected.columns = ["g", "apply_result"]
    got_df = t.to_pandas().sort_values("g").reset_index(drop=True)
    expected = expected.sort_values("g").reset_index(drop=True)
    np.testing.assert_allclose(
        got_df["apply_result"], expected["apply_result"], rtol=1e-9,
    )


def test_apply_ast_uses_composite_sort():
    """AST: _per_group_apply external_sort by group+on; check before to_pandas."""
    import inspect
    from cadw.dataframe import CadwDataFrame
    src = inspect.getsource(CadwDataFrame._per_group_apply)
    assert "external_sort" in src
    assert "to_pandas" in src
    assert "check_state" in src
    assert src.index("check_state") < src.index("to_pandas")
