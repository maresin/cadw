"""v0.9.23j: semi join (7.A.10 split 3/3 final).

Semi join = left rows with match in right. Output = left-schema only.
SQL null semantics: null left key -> NOT emitted (null never matches).
"""
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard


def make_df(tmp_path, data, chunk=3, name="L", mem_mb=50):
    t = pa.table(data)
    p = tmp_path / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, tmp_path / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
        spill_dir=tmp_path / f"spill_{name}",
    )


def _rows(df):
    out = []
    for b in df._iter_filtered_batches():
        out.extend(b.to_pylist())
    return out


# ============ basic semantics ============

def test_semi_basic(tmp_path):
    left = make_df(tmp_path, {
        "k": [1, 2, 3, 4], "a": ["w", "x", "y", "z"],
    }, name="L")
    right = make_df(tmp_path, {"k": [1, 3], "b": ["p", "q"]}, name="R")
    result = left.merge(right, on="k", how="semi")
    rows = sorted(_rows(result), key=lambda r: r["k"])
    # Emit matched: keys 1, 3.
    assert [r["k"] for r in rows] == [1, 3]
    # Output = left-schema only (no right column b).
    assert "b" not in rows[0]


def test_semi_all_match(tmp_path):
    left = make_df(tmp_path, {"k": [1, 2, 3]}, name="L")
    right = make_df(tmp_path, {"k": [1, 2, 3]}, name="R")
    result = left.merge(right, on="k", how="semi")
    assert result.count() == 3


def test_semi_none_match(tmp_path):
    left = make_df(tmp_path, {"k": [10, 20]}, name="L")
    right = make_df(tmp_path, {"k": [1, 2, 3]}, name="R")
    result = left.merge(right, on="k", how="semi")
    assert result.count() == 0


def test_semi_empty_right(tmp_path):
    left = make_df(tmp_path, {"k": [1, 2]}, name="L")
    right_data = {"k": pa.array([], type=pa.int64())}
    right = make_df(tmp_path, right_data, name="R")
    result = left.merge(right, on="k", how="semi")
    assert result.count() == 0


def test_semi_null_left_not_emitted(tmp_path):
    """SQL semantics: null left key never matches -> NOT emitted (semi)."""
    left = make_df(tmp_path, {
        "k": pa.array([1, None, 3], type=pa.int64()),
    }, name="L")
    right = make_df(tmp_path, {"k": [1, 2]}, name="R")
    result = left.merge(right, on="k", how="semi")
    rows = _rows(result)
    # Emit only k=1 (None never matches, 3 not in right).
    keys = {r["k"] for r in rows}
    assert keys == {1}
    assert None not in keys


# ============ multi-key ============

def test_semi_multi_key(tmp_path):
    left = make_df(tmp_path, {
        "k1": [1, 1, 2, 2], "k2": ["a", "b", "a", "b"],
        "v": [10, 20, 30, 40],
    }, name="L")
    right = make_df(tmp_path, {
        "k1": [1, 2], "k2": ["a", "a"],
    }, name="R")
    result = left.merge(right, on=["k1", "k2"], how="semi")
    rows = sorted(_rows(result), key=lambda r: (r["k1"], r["k2"]))
    # Emit matched: (1,a) and (2,a); (1,b),(2,b) unmatched.
    assert len(rows) == 2
    assert [(r["k1"], r["k2"]) for r in rows] == [(1, "a"), (2, "a")]


def test_semi_multi_key_crossed(tmp_path):
    """Canonical multi-key detector (v0.9.23i-fix3 lesson).

    left=(1,"a"), right={(1,"b"),(2,"a")}: cross-product match would
    falsely emit. Composite key: NOT emitted (correct).
    """
    left = make_df(tmp_path, {
        "k1": [1], "k2": ["a"], "v": [100],
    }, name="L")
    right = make_df(tmp_path, {
        "k1": [1, 2], "k2": ["b", "a"],
    }, name="R")
    result = left.merge(right, on=["k1", "k2"], how="semi")
    rows = _rows(result)
    # (1,"a") crossed: (1,*) and (*,"a") in right, but no (1,"a") row.
    # Semi must NOT emit.
    assert len(rows) == 0, (
        f"multi-key semi cross-product bug: expected 0 rows, got {rows}"
    )


def test_semi_multi_key_actual_match(tmp_path):
    """left-tuple IS a right row -> emitted."""
    left = make_df(tmp_path, {
        "k1": [1, 1], "k2": ["a", "b"], "v": [10, 20],
    }, name="L")
    right = make_df(tmp_path, {
        "k1": [1], "k2": ["a"],
    }, name="R")
    result = left.merge(right, on=["k1", "k2"], how="semi")
    rows = _rows(result)
    assert len(rows) == 1
    assert rows[0]["k1"] == 1 and rows[0]["k2"] == "a"


def test_semi_multi_key_null_not_emitted(tmp_path):
    """Null in one on-col -> not matched -> not emitted."""
    left = make_df(tmp_path, {
        "k1": [1, None], "k2": ["a", "b"], "v": [10, 20],
    }, name="L")
    right = make_df(tmp_path, {
        "k1": [1], "k2": ["a"],
    }, name="R")
    result = left.merge(right, on=["k1", "k2"], how="semi")
    rows = _rows(result)
    # (1, "a") matched -> emit. (None, "b") unmatched -> no emit.
    assert len(rows) == 1
    assert rows[0]["v"] == 10


# ============ AST guards ============

def test_semi_uses_hash_shuffle():
    import inspect
    from cadw.ops import join as j
    src = inspect.getsource(j.semi_join_streaming)
    assert "hash_shuffle" in src


def test_semi_uses_is_in():
    import inspect
    from cadw.ops import join as j
    src = inspect.getsource(j._process_partition_pair_semi)
    assert "is_in" in src


def test_semi_fill_null_false():
    """Semi: null left key must NOT be silently matched (fill_null False)."""
    import inspect
    from cadw.ops import join as j
    src = inspect.getsource(j._process_partition_pair_semi)
    assert "fill_null" in src
    assert "False" in src


def test_semi_uses_composite_key():
    """AST: multi-column via _combine_join_keys (fix3 lesson)."""
    import inspect
    from cadw.ops import join as j
    src = inspect.getsource(j._process_partition_pair_semi)
    assert "_combine_join_keys" in src, (
        "multi-column semi must use composite key, not per-column is_in"
    )


def test_semi_exported_from_cadw_ops():
    from cadw.ops import semi_join_streaming
    assert callable(semi_join_streaming)
