"""v0.9.23i: anti join + outer join (7.A.10 split 1/3)."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard


def make_df(tmp_path, data, chunk=3, name="L", mem_mb=20):
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


# ============ anti join ============

def test_anti_basic(tmp_path):
    left = make_df(tmp_path, {"k": [1, 2, 3, 4], "a": ["w", "x", "y", "z"]}, name="L")
    right = make_df(tmp_path, {"k": [1, 3], "b": ["p", "q"]}, name="R")
    result = left.merge(right, on="k", how="anti")
    rows = sorted(_rows(result), key=lambda r: r["k"])
    # Keys 2, 4 absent from right
    assert [r["k"] for r in rows] == [2, 4]
    assert [r["a"] for r in rows] == ["x", "z"]
    assert "b" not in rows[0]


def test_anti_all_match(tmp_path):
    left = make_df(tmp_path, {"k": [1, 2, 3]}, name="L")
    right = make_df(tmp_path, {"k": [1, 2, 3]}, name="R")
    result = left.merge(right, on="k", how="anti")
    assert result.count() == 0


def test_anti_none_match(tmp_path):
    left = make_df(tmp_path, {"k": [10, 20]}, name="L")
    right = make_df(tmp_path, {"k": [1, 2, 3]}, name="R")
    result = left.merge(right, on="k", how="anti")
    assert result.count() == 2


def test_anti_empty_right(tmp_path):
    left = make_df(tmp_path, {"k": [1, 2]}, name="L")
    right_data = {"k": pa.array([], type=pa.int64())}
    right = make_df(tmp_path, right_data, name="R")
    result = left.merge(right, on="k", how="anti")
    assert result.count() == 2


def test_anti_null_left_key_emitted(tmp_path):
    """SQL semantics: null left key never matches -> emitted in anti."""
    left = make_df(tmp_path, {
        "k": pa.array([1, None, 3], type=pa.int64()),
    }, name="L")
    right = make_df(tmp_path, {"k": [1, 2]}, name="R")
    result = left.merge(right, on="k", how="anti")
    rows = _rows(result)
    # 1 matched; None and 3 unmatched
    assert len(rows) == 2
    keys = {r["k"] for r in rows}
    assert 3 in keys
    assert None in keys


def test_anti_multi_key(tmp_path):
    left = make_df(tmp_path, {
        "k1": [1, 1, 2, 2], "k2": ["a", "b", "a", "b"],
        "v": [10, 20, 30, 40],
    }, name="L")
    right = make_df(tmp_path, {
        "k1": [1, 2], "k2": ["a", "a"],
    }, name="R")
    result = left.merge(right, on=["k1", "k2"], how="anti")
    rows = sorted(_rows(result), key=lambda r: (r["k1"], r["k2"]))
    assert len(rows) == 2
    assert [(r["k1"], r["k2"]) for r in rows] == [(1, "b"), (2, "b")]


# ============ outer join ============

def test_outer_basic(tmp_path):
    left = make_df(tmp_path, {"k": [1, 2, 3], "a": ["w", "x", "y"]}, name="L")
    right = make_df(tmp_path, {"k": [2, 3, 4], "b": ["p", "q", "r"]}, name="R")
    result = left.merge(right, on="k", how="outer")
    rows = sorted(_rows(result), key=lambda r: r["k"])
    assert [r["k"] for r in rows] == [1, 2, 3, 4]
    by_k = {r["k"]: r for r in rows}
    assert by_k[1]["a"] == "w" and by_k[1]["b"] is None
    assert by_k[2]["a"] == "x" and by_k[2]["b"] == "p"
    assert by_k[3]["a"] == "y" and by_k[3]["b"] == "q"
    assert by_k[4]["a"] is None and by_k[4]["b"] == "r"


def test_outer_all_match(tmp_path):
    left = make_df(tmp_path, {"k": [1, 2], "a": ["x", "y"]}, name="L")
    right = make_df(tmp_path, {"k": [1, 2], "b": ["p", "q"]}, name="R")
    result = left.merge(right, on="k", how="outer")
    assert result.count() == 2


def test_outer_disjoint(tmp_path):
    left = make_df(tmp_path, {"k": [1, 2], "a": ["x", "y"]}, name="L")
    right = make_df(tmp_path, {"k": [3, 4], "b": ["p", "q"]}, name="R")
    result = left.merge(right, on="k", how="outer")
    assert result.count() == 4
    rows = sorted(_rows(result), key=lambda r: r["k"])
    assert [r["k"] for r in rows] == [1, 2, 3, 4]


def test_outer_null_left_emitted(tmp_path):
    left = make_df(tmp_path, {
        "k": pa.array([1, None], type=pa.int64()),
        "a": ["x", "y"],
    }, name="L")
    right = make_df(tmp_path, {"k": [1, 2], "b": ["p", "q"]}, name="R")
    result = left.merge(right, on="k", how="outer")
    rows = _rows(result)
    assert len(rows) == 3
    by_k = {r["k"]: r for r in rows if r["k"] is not None}
    assert by_k[1]["b"] == "p"
    assert by_k[2]["a"] is None
    null_row = next(r for r in rows if r["k"] is None)
    assert null_row["a"] == "y"
    assert null_row["b"] is None


def test_outer_empty_left(tmp_path):
    left_data = {
        "k": pa.array([], type=pa.int64()),
        "a": pa.array([], type=pa.string()),
    }
    left = make_df(tmp_path, left_data, name="L")
    right = make_df(tmp_path, {"k": [1, 2], "b": ["p", "q"]}, name="R")
    result = left.merge(right, on="k", how="outer")
    rows = sorted(_rows(result), key=lambda r: r["k"])
    assert [r["k"] for r in rows] == [1, 2]
    assert all(r["a"] is None for r in rows)


def test_outer_empty_right(tmp_path):
    left = make_df(tmp_path, {"k": [1, 2], "a": ["x", "y"]}, name="L")
    right_data = {
        "k": pa.array([], type=pa.int64()),
        "b": pa.array([], type=pa.string()),
    }
    right = make_df(tmp_path, right_data, name="R")
    result = left.merge(right, on="k", how="outer")
    rows = sorted(_rows(result), key=lambda r: r["k"])
    assert [r["k"] for r in rows] == [1, 2]
    assert all(r["b"] is None for r in rows)


def test_outer_multi_key(tmp_path):
    left = make_df(tmp_path, {
        "k1": [1, 1, 2], "k2": ["a", "b", "a"], "v": [10, 20, 30],
    }, name="L")
    right = make_df(tmp_path, {
        "k1": [1, 2, 3], "k2": ["a", "a", "a"], "w": [100, 200, 300],
    }, name="R")
    result = left.merge(right, on=["k1", "k2"], how="outer")
    rows = sorted(_rows(result), key=lambda r: (r["k1"], r["k2"]))
    assert len(rows) == 4
    assert rows[0]["v"] == 10 and rows[0]["w"] == 100
    assert rows[1]["v"] == 20 and rows[1]["w"] is None
    assert rows[2]["v"] == 30 and rows[2]["w"] == 200
    assert rows[3]["v"] is None and rows[3]["w"] == 300


# ============ Dispatch / errors ============

def test_merge_how_unsupported_raises(tmp_path):
    left = make_df(tmp_path, {"k": [1]}, name="L")
    right = make_df(tmp_path, {"k": [1]}, name="R")
    with pytest.raises(NotImplementedError, match="right"):
        left.merge(right, on="k", how="right")


def test_merge_how_semi_works(tmp_path):
    """v0.9.23j: semi join now works (was NotImplementedError).

    Rename from test_merge_how_semi_raises; test semantics
    flipped: emit left rows with match in right.
    """
    left = make_df(tmp_path, {"k": [1, 2]}, name="L")
    right = make_df(tmp_path, {"k": [1]}, name="R")
    result = left.merge(right, on="k", how="semi")
    rows = _rows(result)
    assert len(rows) == 1
    assert rows[0]["k"] == 1


# ============ AST guards ============

def test_anti_uses_hash_shuffle():
    import inspect
    from cadw.ops import join as j
    src = inspect.getsource(j.anti_join_streaming)
    assert "hash_shuffle" in src


def test_outer_dispatches_via_merge():
    """AST: merge() dispatches how='outer' to left+anti composition."""
    import inspect
    from cadw.dataframe import CadwDataFrame
    src = inspect.getsource(CadwDataFrame.merge)
    assert '"outer"' in src
    assert '"anti"' in src


def test_anti_uses_is_in():
    import inspect
    from cadw.ops import join as j
    src = inspect.getsource(j._process_partition_pair_anti)
    assert "is_in" in src


def test_anti_fill_null_false():
    """Anti: null left key must NOT be silently matched (fill_null False)."""
    import inspect
    from cadw.ops import join as j
    src = inspect.getsource(j._process_partition_pair_anti)
    assert "fill_null" in src
    assert "False" in src


# ============ Export from cadw.ops ============

def test_anti_outer_exported_from_cadw_ops():
    """Public API: cadw.ops.anti_join_streaming / outer_join_streaming."""
    from cadw.ops import (
        anti_join_streaming, outer_join_streaming,
    )
    assert callable(anti_join_streaming)
    assert callable(outer_join_streaming)


# ============ v0.9.23i-fix2: planner how mapping ============

def test_anti_planner_how_mapped():
    """AST: merge() maps anti -> 'left' for planner (not 'anti')."""
    import inspect
    from cadw.dataframe import CadwDataFrame
    src = inspect.getsource(CadwDataFrame.merge)
    assert '_planner_how = (' in src, (
        "merge must compute _planner_how before plan() call"
    )
    # Right next to plan() call (avoid false positives from other code).
    idx_plan = src.find("self._memory_guard.plan(")
    idx_map = src.find('_planner_how = (')
    assert idx_map >= 0 and idx_map < idx_plan, (
        "_planner_how must be computed before plan() call"
    )


def test_anti_whole_op_warning_skip():
    """AST: predicted_output=0 for anti/semi (output ≤ left_n)."""
    import inspect
    from cadw.dataframe import CadwDataFrame
    src = inspect.getsource(CadwDataFrame.merge)
    assert (
        'if how in ("anti", "semi"):' in src
        and "predicted_output = 0" in src
    ), "anti/semi should skip predicted_output (per-batch guard sufficient)"


def test_semi_join_streaming_import_present():
    """v0.9.23j: semi_join_streaming now implemented and dispatched.

    Stale-test on enum extension (git-log §13.4 pattern): the previous
    assertion "semi_join_streaming not in merge source" was valid only
    while semi join was NotImplemented. Once v0.9.23j shipped the
    implementation, the invariant flipped: merge MUST reference it.
    """
    import inspect
    from cadw.dataframe import CadwDataFrame
    src = inspect.getsource(CadwDataFrame.merge)
    assert "semi_join_streaming" in src, (
        "v0.9.23j: semi join is implemented; merge should dispatch to "
        "semi_join_streaming"
    )


# ============ v0.9.23i-fix3: multi-column crossed-tuple bug ============
#
# Bug (silent data loss): prior code used AND of per-column is_in.
# For left=(1,"a"), right={(1,"b"),(2,"a")}:
#   is_in(k1:[1,2])=True, is_in(k2:["b","a"])=True -> AND=True ->
#   treated as matched -> NOT emitted. But no right row == (1,"a")
#   -> SHOULD emit. This is the class fixed by fix3 (composite key).

def test_anti_multi_key_crossed(tmp_path):
    """left-tuple appears in cross-product of right columns but not as
    a right row -> must be EMITTED (anti).

    Canonical detector for the AND-of-per-column is_in bug.
    """
    left = make_df(tmp_path, {
        "k1": [1], "k2": ["a"], "v": [100],
    }, name="L")
    right = make_df(tmp_path, {
        "k1": [1, 2], "k2": ["b", "a"],
    }, name="R")
    result = left.merge(right, on=["k1", "k2"], how="anti")
    rows = _rows(result)
    assert len(rows) == 1, (
        f"multi-key anti cross-product bug: expected 1 row, got {rows}"
    )
    assert rows[0]["k1"] == 1 and rows[0]["k2"] == "a"


def test_anti_multi_key_actual_match_not_emitted(tmp_path):
    """left-tuple IS a right row -> excluded from anti."""
    left = make_df(tmp_path, {
        "k1": [1, 1], "k2": ["a", "b"], "v": [10, 20],
    }, name="L")
    right = make_df(tmp_path, {
        "k1": [1], "k2": ["a"],
    }, name="R")
    result = left.merge(right, on=["k1", "k2"], how="anti")
    rows = _rows(result)
    # (1,"a") matched -> excluded; (1,"b") unmatched -> emitted.
    assert len(rows) == 1
    assert rows[0]["k1"] == 1 and rows[0]["k2"] == "b"


def test_outer_multi_key_crossed(tmp_path):
    """Cross-product case: outer must include both sides properly."""
    left = make_df(tmp_path, {
        "k1": [1], "k2": ["a"], "v": [100],
    }, name="L")
    right = make_df(tmp_path, {
        "k1": [1, 2], "k2": ["b", "a"], "w": [200, 300],
    }, name="R")
    result = left.merge(right, on=["k1", "k2"], how="outer")
    rows = _rows(result)
    # left(1,"a"): no match. right(1,"b"), right(2,"a"): no match.
    # Outer = 1 + 2 = 3 rows total, both sides padded where needed.
    assert len(rows) == 3
    by_key = {(r["k1"], r["k2"]): r for r in rows}
    assert by_key[(1, "a")]["v"] == 100
    assert by_key[(1, "a")]["w"] is None
    assert by_key[(1, "b")]["v"] is None
    assert by_key[(1, "b")]["w"] == 200
    assert by_key[(2, "a")]["v"] is None
    assert by_key[(2, "a")]["w"] == 300


def test_anti_multi_key_null_emitted(tmp_path):
    """Multi-key with null in one col -> never matched -> emitted."""
    left = make_df(tmp_path, {
        "k1": [1, None], "k2": ["a", "b"], "v": [10, 20],
    }, name="L")
    right = make_df(tmp_path, {
        "k1": [1], "k2": ["a"],
    }, name="R")
    result = left.merge(right, on=["k1", "k2"], how="anti")
    rows = _rows(result)
    # (1, "a") matched -> excluded. (None, "b") unmatched -> emitted.
    assert len(rows) == 1
    assert rows[0]["v"] == 20
