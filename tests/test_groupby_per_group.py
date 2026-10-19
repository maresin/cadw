"""v0.8.10: per-group shift/diff/rolling.

Prototype-validated (scripts/proto_per_group.py, 11 cases A==B).
Reference expectations from tests/_helpers/per_group_reference.py
(pandas ground truth, §13.4/128, /129, /139).
"""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from tests._helpers.per_group_reference import (
    expected_per_group_shift,
    expected_per_group_diff,
    expected_per_group_rolling,
)


def make_df(tmp_path, data, chunk=3, name="L"):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def read_col(df, tmp_path, col, name="o.parquet"):
    out = tmp_path / name
    df.write_parquet(out)
    return pq.read_table(out).column(col).to_pylist()


def _float_eq(a, b, tol=1e-9):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) < tol
    except (TypeError, ValueError):
        return a == b


def _assert_lists_close(got, expected, *, tol=1e-9, msg=""):
    assert len(got) == len(expected), f"length mismatch {msg}"
    for i, (g, e) in enumerate(zip(got, expected)):
        assert _float_eq(g, e, tol), (
            f"[{i}] got={g!r} expected={e!r} ({msg})"
        )


# ============ shift — basic ============

def test_shift_per_group_basic(tmp_path):
    data = {
        "sensor": ["a", "a", "a", "b", "b", "b"],
        "ts": [1, 2, 3, 1, 2, 3],
        "v": [10, 20, 30, 100, 200, 300],
    }
    df = make_df(tmp_path, data)
    res = df.groupby("sensor").shift(1, on="ts", columns=["v"])
    got = read_col(res, tmp_path, "v")
    expected = expected_per_group_shift(
        data["v"], data["sensor"], on=data["ts"], k=1,
    )
    _assert_lists_close(got, expected)


def test_shift_per_group_no_on(tmp_path):
    """on=None → within-group order = original row_id."""
    data = {
        "sensor": ["a", "a", "b", "b"],
        "v": [10, 20, 30, 40],
    }
    df = make_df(tmp_path, data)
    res = df.groupby("sensor").shift(1, columns=["v"])
    got = read_col(res, tmp_path, "v")
    expected = expected_per_group_shift(
        data["v"], data["sensor"], on=None, k=1,
    )
    _assert_lists_close(got, expected)


def test_shift_per_group_k2(tmp_path):
    data = {
        "sensor": ["a", "a", "a", "b", "b", "b"],
        "ts": [1, 2, 3, 1, 2, 3],
        "v": [10, 20, 30, 100, 200, 300],
    }
    df = make_df(tmp_path, data)
    res = df.groupby("sensor").shift(2, on="ts", columns=["v"])
    got = read_col(res, tmp_path, "v")
    expected = expected_per_group_shift(
        data["v"], data["sensor"], on=data["ts"], k=2,
    )
    _assert_lists_close(got, expected)


def test_shift_per_group_fill_value(tmp_path):
    data = {
        "sensor": ["a", "a", "b", "b"],
        "ts": [1, 2, 1, 2],
        "v": [10, 20, 30, 40],
    }
    df = make_df(tmp_path, data)
    res = df.groupby("sensor").shift(
        1, fill_value=0, on="ts", columns=["v"],
    )
    got = read_col(res, tmp_path, "v")
    expected = expected_per_group_shift(
        data["v"], data["sensor"], on=data["ts"], k=1, fill_value=0,
    )
    _assert_lists_close(got, expected)


def test_shift_per_group_shuffled_input(tmp_path):
    data = {
        "sensor": ["b", "a", "b", "a", "b", "a"],
        "ts": [2, 1, 1, 2, 3, 3],
        "v": [200, 10, 100, 20, 300, 30],
    }
    df = make_df(tmp_path, data)
    res = df.groupby("sensor").shift(1, on="ts", columns=["v"])
    got = read_col(res, tmp_path, "v")
    expected = expected_per_group_shift(
        data["v"], data["sensor"], on=data["ts"], k=1,
    )
    _assert_lists_close(got, expected)


# ============ shift — nulls ============

def test_shift_per_group_nulls_in_value(tmp_path):
    data = {
        "sensor": ["a", "a", "a", "a"],
        "ts": [1, 2, 3, 4],
        "v": [10, None, 30, None],
    }
    df = make_df(tmp_path, data)
    res = df.groupby("sensor").shift(1, on="ts", columns=["v"])
    got = read_col(res, tmp_path, "v")
    expected = expected_per_group_shift(
        data["v"], data["sensor"], on=data["ts"], k=1,
    )
    _assert_lists_close(got, expected)


def test_shift_per_group_nulls_in_group(tmp_path):
    """Null group = own group (SQL, not pandas dropna=True default)."""
    data = {
        "sensor": ["a", "a", None, None, "b"],
        "ts": [1, 2, 1, 2, 1],
        "v": [10, 20, 30, 40, 50],
    }
    df = make_df(tmp_path, data)
    res = df.groupby("sensor").shift(1, on="ts", columns=["v"])
    got = read_col(res, tmp_path, "v")
    expected = expected_per_group_shift(
        data["v"], data["sensor"], on=data["ts"], k=1,
    )
    _assert_lists_close(got, expected)


# ============ shift — edge ============

def test_shift_per_group_single_row_per_group(tmp_path):
    data = {
        "sensor": ["a", "b", "c"],
        "ts": [1, 1, 1],
        "v": [10, 20, 30],
    }
    df = make_df(tmp_path, data)
    res = df.groupby("sensor").shift(1, on="ts", columns=["v"])
    got = read_col(res, tmp_path, "v")
    assert got == [None, None, None]


def test_shift_per_group_ties_in_on(tmp_path):
    """Ties in on → tie-break by row_id (stable)."""
    data = {
        "sensor": ["a", "a", "a"],
        "ts": [1, 1, 1],
        "v": [10, 20, 30],
    }
    df = make_df(tmp_path, data)
    res = df.groupby("sensor").shift(1, on="ts", columns=["v"])
    got = read_col(res, tmp_path, "v")
    expected = expected_per_group_shift(
        data["v"], data["sensor"], on=data["ts"], k=1,
    )
    _assert_lists_close(got, expected)


def test_shift_per_group_multi_key(tmp_path):
    data = {
        "s1": ["a", "a", "a", "a"],
        "s2": ["x", "x", "y", "y"],
        "ts": [1, 2, 1, 2],
        "v": [10, 20, 30, 40],
    }
    df = make_df(tmp_path, data)
    # For multi-key group, use the string concat as one group in reference
    groups_combined = [f"{a}|{b}" for a, b in zip(data["s1"], data["s2"])]
    res = df.groupby(["s1", "s2"]).shift(1, on="ts", columns=["v"])
    got = read_col(res, tmp_path, "v")
    expected = expected_per_group_shift(
        data["v"], groups_combined, on=data["ts"], k=1,
    )
    _assert_lists_close(got, expected)


def test_shift_per_group_multi_value_columns(tmp_path):
    data = {
        "sensor": ["a", "a", "b", "b"],
        "ts": [1, 2, 1, 2],
        "v": [10, 20, 30, 40],
        "w": [100, 200, 300, 400],
    }
    df = make_df(tmp_path, data)
    res = df.groupby("sensor").shift(1, on="ts", columns=["v", "w"])
    got_v = read_col(res, tmp_path, "v")
    got_w = read_col(res, tmp_path, "w")
    exp_v = expected_per_group_shift(
        data["v"], data["sensor"], on=data["ts"], k=1,
    )
    exp_w = expected_per_group_shift(
        data["w"], data["sensor"], on=data["ts"], k=1,
    )
    _assert_lists_close(got_v, exp_v)
    _assert_lists_close(got_w, exp_w)


def test_shift_per_group_default_columns_excludes_group(tmp_path):
    """columns=None → all except group cols (sensor not shifted)."""
    data = {
        "sensor": ["a", "a", "b", "b"],
        "ts": [1, 2, 1, 2],
        "v": [10, 20, 30, 40],
    }
    df = make_df(tmp_path, data)
    res = df.groupby("sensor").shift(1, on="ts")
    # sensor NOT shifted
    got_sensor = read_col(res, tmp_path, "sensor")
    assert got_sensor == data["sensor"]
    # ts IS shifted (it's not a group col)
    got_ts = read_col(res, tmp_path, "ts", name="o2.parquet")
    exp_ts = expected_per_group_shift(
        data["ts"], data["sensor"], on=data["ts"], k=1,
    )
    _assert_lists_close(got_ts, exp_ts)


def test_shift_per_group_after_filter(tmp_path):
    data = {
        "sensor": ["a", "a", "a", "b", "b", "b"],
        "ts": [1, 2, 3, 1, 2, 3],
        "v": [10, 20, 30, 100, 200, 300],
    }
    df = make_df(tmp_path, data)
    filtered = df[df["ts"] >= 2]
    res = filtered.groupby("sensor").shift(1, on="ts", columns=["v"])
    out = tmp_path / "o.parquet"
    res.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    # Filtered rows: ts=2,3 per sensor → v=[20,30,200,300]
    # shift(1) per group: [None, 20, None, 200]
    assert [r["v"] for r in rows] == [None, 20, None, 200]


def test_shift_per_group_empty(tmp_path):
    data = {
        "sensor": ["a"],
        "ts": [1],
        "v": [10],
    }
    df = make_df(tmp_path, data)
    empty = df[df["v"] > 100]  # filtered to empty
    res = empty.groupby("sensor").shift(1, on="ts", columns=["v"])
    out = tmp_path / "o.parquet"
    res.write_parquet(out)
    t = pq.read_table(out)
    assert t.num_rows == 0
    assert t.schema.names == df.columns


# ============ shift — validation ============

def test_shift_per_group_missing_on_column(tmp_path):
    df = make_df(tmp_path, {"sensor": ["a"], "v": [10]})
    with pytest.raises(KeyError, match="nope"):
        df.groupby("sensor").shift(1, on="nope")


def test_shift_per_group_explicit_group_in_columns(tmp_path):
    df = make_df(tmp_path, {
        "sensor": ["a", "a"], "ts": [1, 2], "v": [10, 20],
    })
    with pytest.raises(ValueError, match="cannot be shifted"):
        df.groupby("sensor").shift(1, on="ts", columns=["sensor"])


def test_shift_per_group_invalid_k(tmp_path):
    df = make_df(tmp_path, {
        "sensor": ["a", "a"], "ts": [1, 2], "v": [10, 20],
    })
    with pytest.raises(ValueError, match="k must be int"):
        df.groupby("sensor").shift(0, on="ts")


# ============ diff ============

def test_diff_per_group_basic(tmp_path):
    data = {
        "sensor": ["a", "a", "a", "b", "b", "b"],
        "ts": [1, 2, 3, 1, 2, 3],
        "v": [10, 25, 40, 100, 130, 170],
    }
    df = make_df(tmp_path, data)
    res = df.groupby("sensor").diff(1, on="ts", columns=["v"])
    got = read_col(res, tmp_path, "v")
    expected = expected_per_group_diff(
        data["v"], data["sensor"], on=data["ts"], k=1,
    )
    _assert_lists_close(got, expected)


def test_diff_per_group_k2(tmp_path):
    data = {
        "sensor": ["a", "a", "a", "a"],
        "ts": [1, 2, 3, 4],
        "v": [10, 25, 40, 60],
    }
    df = make_df(tmp_path, data)
    res = df.groupby("sensor").diff(2, on="ts", columns=["v"])
    got = read_col(res, tmp_path, "v")
    expected = expected_per_group_diff(
        data["v"], data["sensor"], on=data["ts"], k=2,
    )
    _assert_lists_close(got, expected)


def test_diff_per_group_nulls(tmp_path):
    data = {
        "sensor": ["a", "a", "a", "a"],
        "ts": [1, 2, 3, 4],
        "v": [10.0, None, 30.0, 40.0],
    }
    df = make_df(tmp_path, data)
    res = df.groupby("sensor").diff(1, on="ts", columns=["v"])
    got = read_col(res, tmp_path, "v")
    expected = expected_per_group_diff(
        data["v"], data["sensor"], on=data["ts"], k=1,
    )
    _assert_lists_close(got, expected)


def test_diff_per_group_non_numeric_raises(tmp_path):
    df = make_df(tmp_path, {
        "sensor": ["a", "a"], "ts": [1, 2], "s": ["x", "y"],
    })
    with pytest.raises(TypeError, match="numeric"):
        df.groupby("sensor").diff(1, on="ts", columns=["s"])


# ============ rolling ============

@pytest.mark.parametrize("agg", ["sum", "mean", "min", "max", "count"])
def test_rolling_per_group_all_aggs(tmp_path, agg):
    data = {
        "sensor": ["a", "a", "a", "b", "b", "b"],
        "ts": [1, 2, 3, 1, 2, 3],
        "v": [10, 20, 30, 100, 200, 300],
    }
    df = make_df(tmp_path, data)
    res = df.groupby("sensor").rolling(2, agg, on="ts", columns=["v"])
    got = read_col(res, tmp_path, f"v_rolling_{agg}")
    expected = expected_per_group_rolling(
        data["v"], data["sensor"], on=data["ts"],
        window=2, agg=agg,
    )
    _assert_lists_close(got, expected)


def test_rolling_per_group_min_periods_1(tmp_path):
    data = {
        "sensor": ["a", "a", "a", "b", "b", "b"],
        "ts": [1, 2, 3, 1, 2, 3],
        "v": [10, 20, 30, 100, 200, 300],
    }
    df = make_df(tmp_path, data)
    res = df.groupby("sensor").rolling(
        3, "sum", min_periods=1, on="ts", columns=["v"],
    )
    got = read_col(res, tmp_path, "v_rolling_sum")
    expected = expected_per_group_rolling(
        data["v"], data["sensor"], on=data["ts"],
        window=3, agg="sum", min_periods=1,
    )
    _assert_lists_close(got, expected)


def test_rolling_per_group_nulls_in_value(tmp_path):
    data = {
        "sensor": ["a", "a", "a", "a"],
        "ts": [1, 2, 3, 4],
        "v": [10.0, None, 30.0, 40.0],
    }
    df = make_df(tmp_path, data)
    res = df.groupby("sensor").rolling(
        2, "sum", min_periods=1, on="ts", columns=["v"],
    )
    got = read_col(res, tmp_path, "v_rolling_sum")
    expected = expected_per_group_rolling(
        data["v"], data["sensor"], on=data["ts"],
        window=2, agg="sum", min_periods=1,
    )
    _assert_lists_close(got, expected)


def test_rolling_per_group_append_not_replace(tmp_path):
    data = {
        "sensor": ["a", "a", "b", "b"],
        "ts": [1, 2, 1, 2],
        "v": [10, 20, 30, 40],
    }
    df = make_df(tmp_path, data)
    res = df.groupby("sensor").rolling(2, "sum", on="ts", columns=["v"])
    out = tmp_path / "o.parquet"
    res.write_parquet(out)
    names = pq.read_schema(out).names
    # Original columns present + rolling column appended
    assert "v" in names
    assert "v_rolling_sum" in names


def test_rolling_per_group_window_gt_group_size(tmp_path):
    """window=10 > group size: within each group, min_periods=window → all null."""
    data = {
        "sensor": ["a", "a", "b", "b"],
        "ts": [1, 2, 1, 2],
        "v": [10, 20, 30, 40],
    }
    df = make_df(tmp_path, data)
    res = df.groupby("sensor").rolling(10, "sum", on="ts", columns=["v"])
    got = read_col(res, tmp_path, "v_rolling_sum")
    assert got == [None, None, None, None]


# ============ AST guards ============

def test_per_group_uses_composite_sort():
    """AST: _per_group_shift must sort by [*group_cols, *on_cols]."""
    import ast
    import inspect
    from cadw.dataframe import CadwDataFrame
    src = inspect.getsource(CadwDataFrame._per_group_shift)
    tree = ast.parse(inspect.cleandoc(src))
    has_composite = False
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "external_sort"):
            for kw in node.keywords:
                if kw.arg == "key_columns":
                    unparsed = ast.unparse(kw.value)
                    if "sort_keys" in unparsed:
                        has_composite = True
    assert has_composite, (
        "_per_group_shift must sort by composite key (sort_keys = "
        "group_cols + on_cols)"
    )


def test_per_group_resets_on_group_change():
    """AST: _per_group_shift clears deques on group change."""
    import inspect
    from cadw.dataframe import CadwDataFrame
    src = inspect.getsource(CadwDataFrame._per_group_shift)
    assert "dq.clear()" in src, (
        "_per_group_shift must clear deque on group change"
    )
    assert "prev_group" in src, (
        "_per_group_shift must track prev_group"
    )


def test_per_group_uses_second_sort_by_orig_rid():
    """AST: second sort must use __orig_rid (not raw row_id)."""
    import ast
    import inspect
    from cadw.dataframe import CadwDataFrame
    src = inspect.getsource(CadwDataFrame._per_group_shift)
    tree = ast.parse(inspect.cleandoc(src))
    sort_keys_seen: list = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "external_sort"):
            for kw in node.keywords:
                if kw.arg == "key_columns":
                    if isinstance(kw.value, ast.List):
                        keys = [
                            e.value for e in kw.value.elts
                            if isinstance(e, ast.Constant)
                        ]
                        sort_keys_seen.append(keys)
    # Second sort (last) must be __orig_rid
    assert sort_keys_seen, "no external_sort calls found"
    second = sort_keys_seen[-1]
    assert "__orig_rid" in second, (
        f"second sort must use __orig_rid; got {second}"
    )
    assert "row_id" not in second, (
        f"second sort must NOT use raw row_id; got {second}"
    )


def test_rollstate_has_reset():
    """_RollState.reset() must exist (used by per_group_rolling)."""
    from cadw.dataframe import _RollState
    assert hasattr(_RollState, "reset")
