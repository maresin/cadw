"""v0.9.23g: isna / isna_sum / fillna / value_counts / nunique (7.A.5 + 7.A.3).

EDA basics, streaming primitives. No pandas; pa.Table for value_counts.
"""
from pathlib import Path

import math
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.core.memory_guard import MemoryLimitExceeded


def make_df(tmp_path, data, chunk=5, name="L", mem_mb=50):
    t = pa.table(data)
    p = tmp_path / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, tmp_path / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
        spill_dir=tmp_path / f"spill_{name}",
    )


# ============ isna ============

def test_isna_basic(tmp_path):
    df = make_df(tmp_path, {
        "a": pa.array([1, None, 3, None], type=pa.int64()),
        "b": pa.array(["x", "y", None, "z"], type=pa.string()),
    })
    out = df.isna()
    t = tmp_path / "o.parquet"
    out.write_parquet(t)
    tbl = pq.read_table(t)
    assert tbl.schema.field("a").type == pa.bool_()
    assert tbl.schema.field("b").type == pa.bool_()
    assert tbl.column("a").to_pylist() == [False, True, False, True]
    assert tbl.column("b").to_pylist() == [False, False, True, False]


def test_isna_no_nulls(tmp_path):
    df = make_df(tmp_path, {"a": [1, 2, 3]})
    out = df.isna()
    t = tmp_path / "o.parquet"
    out.write_parquet(t)
    assert pq.read_table(t).column("a").to_pylist() == [False, False, False]


def test_isna_after_filter(tmp_path):
    df = make_df(tmp_path, {
        "a": pa.array([1, None, 3, None, 5], type=pa.int64()),
        "v": [10, 20, 30, 40, 50],
    })
    filtered = df[df["v"] > 25]
    out = filtered.isna()
    t = tmp_path / "o.parquet"
    out.write_parquet(t)
    assert pq.read_table(t).column("a").to_pylist() == [False, True, False]


# ============ isna_sum ============

def test_isna_sum_basic(tmp_path):
    df = make_df(tmp_path, {
        "a": pa.array([1, None, 3, None], type=pa.int64()),
        "b": pa.array([None, None, "x", "y"], type=pa.string()),
    })
    out = df.isna_sum()
    assert out == {"a": 2, "b": 2}


def test_isna_sum_no_nulls(tmp_path):
    df = make_df(tmp_path, {"a": [1, 2, 3], "b": ["x", "y", "z"]})
    assert df.isna_sum() == {"a": 0, "b": 0}


def test_isna_sum_after_filter(tmp_path):
    df = make_df(tmp_path, {
        "a": pa.array([1, None, 3, None, 5], type=pa.int64()),
        "v": [10, 20, 30, 40, 50],
    })
    filtered = df[df["v"] > 25]
    assert filtered.isna_sum()["a"] == 1


# ============ fillna ============

def test_fillna_basic(tmp_path):
    df = make_df(tmp_path, {
        "a": pa.array([1, None, 3, None], type=pa.int64()),
        "b": pa.array(["x", None, "z", None], type=pa.string()),
    })
    out = df.fillna(0, columns=["a"])
    t = tmp_path / "o.parquet"
    out.write_parquet(t)
    assert pq.read_table(t).column("a").to_pylist() == [1, 0, 3, 0]


def test_fillna_string(tmp_path):
    df = make_df(tmp_path, {
        "s": pa.array(["x", None, "z"], type=pa.string()),
    })
    out = df.fillna("MISSING", columns=["s"])
    t = tmp_path / "o.parquet"
    out.write_parquet(t)
    assert pq.read_table(t).column("s").to_pylist() == [
        "x", "MISSING", "z",
    ]


def test_fillna_all_columns(tmp_path):
    df = make_df(tmp_path, {
        "a": pa.array([1, None], type=pa.int64()),
        "b": pa.array([10, None], type=pa.int64()),
    })
    out = df.fillna(0)
    t = tmp_path / "o.parquet"
    out.write_parquet(t)
    tbl = pq.read_table(t)
    assert tbl.column("a").to_pylist() == [1, 0]
    assert tbl.column("b").to_pylist() == [10, 0]


def test_fillna_missing_column_raises(tmp_path):
    df = make_df(tmp_path, {"a": [1, 2]})
    with pytest.raises(KeyError, match="nope"):
        df.fillna(0, columns=["nope"])


def test_fillna_incompatible_type_raises(tmp_path):
    """Fill value with wrong type -> TypeError at entry (fail-fast)."""
    df = make_df(tmp_path, {
        "s": pa.array(["x", None], type=pa.string()),
    })
    with pytest.raises(TypeError, match="incompatible"):
        df.fillna(123, columns=["s"])


def test_fillna_empty_columns_raises(tmp_path):
    df = make_df(tmp_path, {"a": [1, 2]})
    with pytest.raises(ValueError, match="non-empty"):
        df.fillna(0, columns=[])


# ============ value_counts ============

def test_value_counts_basic(tmp_path):
    df = make_df(tmp_path, {
        "k": ["a", "b", "a", "c", "b", "a"],
    })
    out = df.value_counts("k")
    rows = out.to_pylist()
    # Sorted by count desc: a=3, b=2, c=1
    assert rows == [
        {"k": "a", "count": 3},
        {"k": "b", "count": 2},
        {"k": "c", "count": 1},
    ]


def test_value_counts_no_sort(tmp_path):
    df = make_df(tmp_path, {"k": ["a", "b", "a"]})
    out = df.value_counts("k", sort=False)
    # Total counts must sum to 3 regardless of order
    assert sum(r["count"] for r in out.to_pylist()) == 3


def test_value_counts_normalize(tmp_path):
    df = make_df(tmp_path, {"k": ["a", "a", "b", "b"]})
    out = df.value_counts("k", normalize=True)
    rows = out.to_pylist()
    assert out.schema.names == ["k", "fraction"]
    fracs = {r["k"]: r["fraction"] for r in rows}
    assert fracs["a"] == pytest.approx(0.5)
    assert fracs["b"] == pytest.approx(0.5)


def test_value_counts_with_nulls(tmp_path):
    """Nulls form their own group (like groupby dropna=False)."""
    df = make_df(tmp_path, {
        "k": pa.array(["a", None, "a", "b", None], type=pa.string()),
    })
    out = df.value_counts("k")
    rows = {(r["k"], r["count"]) for r in out.to_pylist()}
    assert ("a", 2) in rows
    assert ("b", 1) in rows
    assert (None, 2) in rows


def test_value_counts_missing_column_raises(tmp_path):
    df = make_df(tmp_path, {"k": ["a"]})
    with pytest.raises(KeyError, match="nope"):
        df.value_counts("nope")


def test_value_counts_after_filter(tmp_path):
    df = make_df(tmp_path, {
        "k": ["a", "b", "a", "c"],
        "v": [1, 2, 3, 4],
    })
    filtered = df[df["v"] > 2]
    out = filtered.value_counts("k")
    rows = {(r["k"], r["count"]) for r in out.to_pylist()}
    assert rows == {("a", 1), ("c", 1)}


# ============ nunique ============

def test_nunique_single_column(tmp_path):
    df = make_df(tmp_path, {"k": ["a", "b", "a", "c", "b"]})
    assert df.nunique("k") == 3


def test_nunique_all_columns(tmp_path):
    df = make_df(tmp_path, {
        "a": ["x", "y", "x"],
        "b": [1, 2, 3],
    })
    assert df.nunique() == {"a": 2, "b": 3}


def test_nunique_missing_column_raises(tmp_path):
    df = make_df(tmp_path, {"a": [1]})
    with pytest.raises(KeyError, match="nope"):
        df.nunique("nope")


def test_nunique_after_filter(tmp_path):
    df = make_df(tmp_path, {
        "k": ["a", "b", "a", "c"],
        "v": [1, 2, 3, 4],
    })
    filtered = df[df["v"] > 2]
    assert filtered.nunique("k") == 2


# ============ AST guards ============

def test_eda_methods_no_to_pandas():
    import inspect
    from cadw.patch_utils import assert_no_call_in_source
    from cadw.dataframe import CadwDataFrame
    for fn in ("isna", "isna_sum", "fillna",
               "value_counts", "nunique"):
        src = inspect.getsource(getattr(CadwDataFrame, fn))
        assert_no_call_in_source(
            src, "to_pandas", "from_pandas",
            fn_name=f"CadwDataFrame.{fn}",
        )


def test_isna_uses_pc_is_null():
    import inspect
    from cadw.dataframe import CadwDataFrame
    src = inspect.getsource(CadwDataFrame.isna)
    assert "is_null" in src


def test_fillna_uses_pc_fill_null():
    import inspect
    from cadw.dataframe import CadwDataFrame
    src = inspect.getsource(CadwDataFrame.fillna)
    assert "fill_null" in src


def test_value_counts_uses_groupby_aggregate():
    import inspect
    from cadw.dataframe import CadwDataFrame
    src = inspect.getsource(CadwDataFrame.value_counts)
    assert "groupby_aggregate" in src


# ============ v0.9.23g-fix1: multi-batch regression ============

def test_isna_multibatch(tmp_path):
    """isna works when chunk=1 (schema nullable=True per §13.4/169).

    Prior v0.9.23g had nullable=False in out_fields, which crashed
    with ArrowInvalid on write_batch (Dataset.from_batches
    reconstructs columns with nullable=True).
    """
    df = make_df(tmp_path, {
        "a": pa.array([1, None, 3, None], type=pa.int64()),
    }, chunk=1)  # 4 batches of 1 row
    out = df.isna()
    t = tmp_path / "o.parquet"
    out.write_parquet(t)
    assert pq.read_table(t).column("a").to_pylist() == [
        False, True, False, True,
    ]


def test_nunique_multibatch_overlap(tmp_path):
    """SILENT DATA CORRUPTION regression: multi-batch overlap.

    Prior v0.9.23g summed per-batch count_distinct. Values [a, a]
    with chunk=1: batch1=[a] -> count_distinct=1, batch2=[a] ->
    count_distinct=1, sum=2. But global nunique=1.

    Fix: groupby_aggregate (HashShuffle + count) — always correct.
    """
    df = make_df(tmp_path, {
        "k": ["a", "a", "a", "a"],
    }, chunk=1)  # 4 batches, all "a"
    assert df.nunique("k") == 1


def test_nunique_multibatch_partial_overlap(tmp_path):
    """Values spread across batches with partial overlap."""
    df = make_df(tmp_path, {
        "k": ["a", "b", "a", "c", "b", "d"],
    }, chunk=2)  # batches: [a,b], [a,c], [b,d]
    # Per-batch sum would be 2+2+2=6, actual=4 (a,b,c,d)
    assert df.nunique("k") == 4


def test_nunique_all_columns_multibatch(tmp_path):
    """All-columns nunique with chunk=1."""
    df = make_df(tmp_path, {
        "a": ["x", "x", "y"],
        "b": [1, 1, 1],
    }, chunk=1)
    assert df.nunique() == {"a": 2, "b": 1}


def test_value_counts_multibatch_overlap(tmp_path):
    """value_counts is already correct (HashShuffle), but pin it."""
    df = make_df(tmp_path, {
        "k": ["a", "a", "b"],
    }, chunk=1)
    out = df.value_counts("k")
    rows = {(r["k"], r["count"]) for r in out.to_pylist()}
    assert rows == {("a", 2), ("b", 1)}


# ============ v0.9.23g-fix2: value_counts null/dropna/empty ============

def test_value_counts_null_multibatch(tmp_path):
    """v0.9.23g-fix2 SILENT BUG regression: null count via size.

    Prior used groupby(count) — CountOp skips nulls -> null group
    count=0. Fix: ones-sum -> every row contributes 1 -> null group
    count = number of nulls. chunk=1 (worst case).
    """
    df = make_df(tmp_path, {
        "k": pa.array(
            ["a", None, "a", None, "b"], type=pa.string(),
        ),
    }, chunk=1)
    out = df.value_counts("k")
    rows = {(r["k"], r["count"]) for r in out.to_pylist()}
    assert ("a", 2) in rows
    assert ("b", 1) in rows
    assert (None, 2) in rows, f"null count must be 2, got {rows}"


def test_value_counts_dropna_true(tmp_path):
    """dropna=True excludes null group (pandas-parity)."""
    df = make_df(tmp_path, {
        "k": pa.array(
            ["a", None, "a", None, "b"], type=pa.string(),
        ),
    }, chunk=1)
    out = df.value_counts("k", dropna=True)
    rows = {(r["k"], r["count"]) for r in out.to_pylist()}
    assert rows == {("a", 2), ("b", 1)}


def test_value_counts_dropna_true_all_null(tmp_path):
    """dropna=True on all-null column -> empty table."""
    df = make_df(tmp_path, {
        "k": pa.array([None, None], type=pa.string()),
    })
    out = df.value_counts("k", dropna=True)
    assert out.num_rows == 0
    assert out.schema.names == ["k", "count"]


def test_value_counts_empty_df(tmp_path):
    """Empty df -> empty table with proper schema (no crash)."""
    df = make_df(tmp_path, {
        "k": pa.array([], type=pa.string()),
    })
    out = df.value_counts("k")
    assert out.num_rows == 0
    assert out.schema.names == ["k", "count"]


def test_value_counts_normalize_null_multibatch(tmp_path):
    """normalize with nulls: fraction includes null count in denominator."""
    df = make_df(tmp_path, {
        "k": pa.array(
            ["a", None, "a", None], type=pa.string(),
        ),
    }, chunk=1)
    out = df.value_counts("k", normalize=True)
    rows = {(r["k"], r["fraction"]) for r in out.to_pylist()}
    # Each value (a, null) has 2/4 = 0.5
    assert ("a", 0.5) in rows
    assert (None, 0.5) in rows


def test_nunique_dropna_true(tmp_path):
    """dropna=True excludes null (pandas-parity)."""
    df = make_df(tmp_path, {
        "k": pa.array(
            ["a", None, "a", None, "b"], type=pa.string(),
        ),
    }, chunk=1)
    assert df.nunique("k", dropna=True) == 2


def test_nunique_dropna_true_all_null(tmp_path):
    """dropna=True on all-null column -> 0."""
    df = make_df(tmp_path, {
        "k": pa.array([None, None], type=pa.string()),
    })
    assert df.nunique("k", dropna=True) == 0


def test_nunique_dropna_false_default(tmp_path):
    """Default dropna=False: null counted as distinct (CADW groupby)."""
    df = make_df(tmp_path, {
        "k": pa.array(
            ["a", None, "a", None, "b"], type=pa.string(),
        ),
    }, chunk=1)
    assert df.nunique("k") == 3  # a, b, null


# ============ v0.9.23g-fix5: is_empty_input_error AST guards ============

def test_eda_value_counts_uses_is_empty_input_error():
    """value_counts catches empty-input via helper (both languages).

    Prior bug: `"empty" in str(e).lower()` missed Russian
    "пустой вход" -> ValueError re-raised on all-null + dropna=True.
    """
    import inspect
    from cadw.dataframe import CadwDataFrame
    src = inspect.getsource(CadwDataFrame.value_counts)
    assert "is_empty_input_error" in src, (
        "value_counts must use patch_utils.is_empty_input_error "
        "for empty-input detection (English+Russian)"
    )


def test_eda_nunique_uses_is_empty_input_error():
    """nunique (_nunique_one) uses helper (would've caught fix4 bug).

    fix4's custom guard false-positived (marker in value_counts bled
    into _nunique_one check). Fix5 fixed it; this test prevents
    regression.
    """
    import inspect
    from cadw.dataframe import CadwDataFrame
    src = inspect.getsource(CadwDataFrame.nunique)
    assert "is_empty_input_error" in src, (
        "nunique must use patch_utils.is_empty_input_error "
        "inside _nunique_one (fix5 bug: custom guard skipped the "
        "actual fix)"
    )


# ============ v0.9.23g-fix6: no .as_py in isna_sum ============

def test_isna_sum_no_as_py_ast():
    """AST: isna_sum body has no .as_py() Call.

    test_facade_has_no_as_py catches facade-wide. This pins the
    specific fix (isna_sum uses NullCountOp, not pc.sum().as_py()).
    """
    import ast as _ast
    import inspect as _inspect
    import textwrap as _tw
    from cadw.dataframe import CadwDataFrame
    src = _tw.dedent(_inspect.getsource(CadwDataFrame.isna_sum))
    tree = _ast.parse(src)
    as_py_calls = [
        n for n in _ast.walk(tree)
        if isinstance(n, _ast.Call)
        and isinstance(n.func, _ast.Attribute)
        and n.func.attr == "as_py"
    ]
    assert not as_py_calls, (
        f"isna_sum calls .as_py() {len(as_py_calls)} time(s); "
        f"use NullCountOp (Reduce)"
    )
    assert "NullCountOp" in src, (
        "isna_sum must use NullCountOp"
    )
