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


def test_apply_dict_multi_output(tmp_path):
    """v0.9.6b2: dict[str,scalar] replaces pd.Series."""
    df = make_df(tmp_path, {
        "g": ["a", "a", "b", "b"], "v": [10, 20, 100, 200],
    })
    out = df.groupby("g").apply(
        lambda g: {"mean": g["v"].mean(), "max": g["v"].max()},
    )
    t = read_table(out, tmp_path)
    assert set(t.schema.names) == {"g", "mean", "max"}
    rows = {r["g"]: (r["mean"], r["max"]) for r in t.to_pylist()}
    assert rows == {"a": (15.0, 20), "b": (150.0, 200)}


def test_apply_cadwdf_expand(tmp_path):
    """v0.9.6b2: CadwDataFrame replaces pd.DataFrame."""
    import pyarrow as pa
    df = make_df(tmp_path, {
        "g": ["a", "a", "a", "b", "b"], "v": [1, 2, 3, 10, 20],
    })
    def fn(g):
        n = len(g)
        return CadwDataFrame.from_table(
            pa.table({
                "v_rank": np.arange(1, n + 1),
                "v_shift": g["v"].to_list(),
            }),
            memory_guard=g._memory_guard,
            spill_dir=g._spill_dir,
        )
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
    out = df.groupby("g").apply(lambda g: g["v"].to_list()[0])
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
        lambda g: g["v"].to_list()[0], on="ts",
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
        return {"x": 1}  # dict (was pd.Series)
    with pytest.raises(RuntimeError, match="mixed return modes"):
        df.groupby("g").apply(fn)


def test_apply_unsupported_return_type_raises(tmp_path):
    df = make_df(tmp_path, {"g": ["a", "a"], "v": [1, 2]})
    with pytest.raises(TypeError, match="expected scalar, dict"):
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
    """AST: _per_group_apply — external_sort + check_state before
    materialize, no .to_pandas() Call nodes.

    v0.9.6b2-fix4: ALL checks AST-based. Prior fix2 kept
    `src.index("check_state") < src.index("from_table")` — substring;
    docstring "CadwDataFrame.from_table(pa_table)" matched earlier than
    real check_state call → false failure (§13.4/161, third occurrence
    in v0.9.6b2 session: /182, /183, /184, /185).
    """
    import ast as _ast
    import inspect
    import textwrap
    from cadw.dataframe import CadwDataFrame

    src = inspect.getsource(CadwDataFrame._per_group_apply)
    tree = _ast.parse(textwrap.dedent(src))

    # Collect real Call-node linenos by attr/id. Docstrings and
    # comments are invisible to ast.walk over Call nodes.
    call_linenos: dict[str, list[int]] = {}
    for node in _ast.walk(tree):
        if not isinstance(node, _ast.Call):
            continue
        f = node.func
        if isinstance(f, _ast.Name):
            # bare function calls: external_sort(...)
            if f.id in ("external_sort",):
                call_linenos.setdefault(f.id, []).append(node.lineno)
        elif isinstance(f, _ast.Attribute):
            # method calls: obj.check_state(...), CadwDataFrame.from_table(...)
            if f.attr in ("check_state", "from_table", "to_pandas"):
                call_linenos.setdefault(f.attr, []).append(node.lineno)

    # 1. external_sort is called (composite sort by group+on).
    assert "external_sort" in call_linenos, (
        "v0.9.6b2: _per_group_apply must call external_sort"
    )

    # 2. No .to_pandas() Call nodes.
    assert "to_pandas" not in call_linenos, (
        f"_per_group_apply calls to_pandas at lines "
        f"{call_linenos.get('to_pandas')}"
    )

    # 3. Group materialized via CadwDataFrame.from_table.
    assert "from_table" in call_linenos, (
        "v0.9.6b2: group must be materialized via CadwDataFrame.from_table"
    )

    # 4. check_state Call precedes from_table Call (§3.10).
    assert "check_state" in call_linenos, (
        "_per_group_apply must call check_state"
    )
    first_guard = min(call_linenos["check_state"])
    first_materialize = min(call_linenos["from_table"])
    assert first_guard < first_materialize, (
        f"check_state call (line {first_guard}) must precede "
        f"CadwDataFrame.from_table call (line {first_materialize}) — "
        f"§3.10 guard before materialize"
    )




def test_apply_cadwdf_group_col_embedded(tmp_path):
    """User returns CadwDataFrame with group col already
    present — output has group col exactly once."""
    import pyarrow as pa
    df = make_df(tmp_path, {"g": ["a", "a", "b"], "v": [1, 2, 3]})
    def fn(g):
        vals = g["v"].to_list()
        gval = g["g"].to_list()[0]
        return CadwDataFrame.from_table(
            pa.table({"g": [gval] * len(vals), "val": vals}),
            memory_guard=g._memory_guard,
            spill_dir=g._spill_dir,
        )
    out = df.groupby("g").apply(fn)
    t = read_table(out, tmp_path)
    assert t.schema.names.count("g") == 1
    assert "val" in t.schema.names
    assert t.num_rows == 3


# ============ v0.9.10 (C1): _per_group_apply mixed int/float widen ============


def test_apply_mixed_int_float_widens_schema(tmp_path):
    """C1: first group int, later float — schema widens to float64.

    Prior: pa.Table.from_pylist(rows, schema=int64) truncated
    float(2.5) to int(2), silent corruption. Or raises ArrowInvalid.
    """
    import pyarrow.parquet as pq
    df = make_df(tmp_path, {
        'g': ['a', 'a', 'b', 'b'],
        'v': [1, 2, 3, 4],
    })
    def fn(g):
        if g['g'].to_list()[0] == 'a':
            return 5  # int (first group)
        return 2.5  # float (later)
    result = df.groupby('g').apply(fn)
    out = tmp_path / 'out.parquet'
    result.write_parquet(out)
    rows = {r['g']: r['apply_result'] for r in
            pq.read_table(out).to_pylist()}
    assert rows['a'] == 5.0
    assert rows['b'] == 2.5


def test_apply_all_int_stays_int(tmp_path):
    """C1 regression: all-int → schema stays int64 (no widen)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    df = make_df(tmp_path, {
        'g': ['a', 'a', 'b', 'b'],
        'v': [1, 2, 3, 4],
    })
    def fn(g):
        return 5
    result = df.groupby('g').apply(fn)
    out = tmp_path / 'out.parquet'
    result.write_parquet(out)
    schema = pq.read_schema(out)
    assert schema.field('apply_result').type == pa.int64()