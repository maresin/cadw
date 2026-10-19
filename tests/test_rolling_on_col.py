"""v0.8.4: rolling(on=col) — sort-aware rolling via 2 external_sort."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard


def make_df(tmp_path, data=None, chunk=3, name="L"):
    if data is None:
        data = {
            "ts": [5, 3, 1, 4, 2],
            "v": [50.0, 30.0, 10.0, 40.0, 20.0],
            "cat": ["e", "c", "a", "d", "b"],
        }
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def read_col(df, tmp_path, col):
    out = tmp_path / "o.parquet"
    df.write_parquet(out)
    return pq.read_table(out).column(col).to_pylist()


def _approx_eq_float(xs, ys, tol=1e-9):
    if len(xs) != len(ys):
        return False
    for x, y in zip(xs, ys):
        if x is None and y is None:
            continue
        if x is None or y is None:
            return False
        if abs(x - y) > tol:
            return False
    return True


# ---------- Basic aggs ----------

def test_rolling_on_sum(tmp_path):
    """Prototype expected: [90, 50, None, 70, 30]."""
    df = make_df(tmp_path)
    res = df.rolling(2, "sum", on="ts", columns=["v"])
    vals = read_col(res, tmp_path, "v_rolling_sum")
    assert vals == [90.0, 50.0, None, 70.0, 30.0]


def test_rolling_on_mean(tmp_path):
    """Sorted v: [10,20,30,40,50]; mean_w2 = [None,15,25,35,45]."""
    df = make_df(tmp_path)
    res = df.rolling(2, "mean", on="ts", columns=["v"])
    vals = read_col(res, tmp_path, "v_rolling_mean")
    assert vals == [45.0, 25.0, None, 35.0, 15.0]


def test_rolling_on_min(tmp_path):
    """Sorted v: [10,20,30,40,50]; min_w2 = [None,10,20,30,40]."""
    df = make_df(tmp_path)
    res = df.rolling(2, "min", on="ts", columns=["v"])
    vals = read_col(res, tmp_path, "v_rolling_min")
    assert vals == [40.0, 20.0, None, 30.0, 10.0]


def test_rolling_on_max(tmp_path):
    df = make_df(tmp_path)
    res = df.rolling(2, "max", on="ts", columns=["v"])
    vals = read_col(res, tmp_path, "v_rolling_max")
    assert vals == [50.0, 30.0, None, 40.0, 20.0]


def test_rolling_on_count(tmp_path):
    df = make_df(tmp_path)
    res = df.rolling(2, "count", on="ts", columns=["v"])
    vals = read_col(res, tmp_path, "v_rolling_count")
    assert vals == [2, 2, None, 2, 2]


# ---------- Edge cases ----------

def test_rolling_on_min_periods_1(tmp_path):
    df = make_df(tmp_path)
    res = df.rolling(2, "sum", on="ts", columns=["v"], min_periods=1)
    vals = read_col(res, tmp_path, "v_rolling_sum")
    assert vals == [90.0, 50.0, 10.0, 70.0, 30.0]


def test_rolling_on_with_nulls(tmp_path):
    """Nulls consume window slot (positional, matches pandas skipna=True).

    Sorted by ts: [10, 20, None, 30]. window=2, sum, min_periods=1:
        pos0: [10] → 10
        pos1: [10,20] → 30
        pos2: [20,None] → 20 (window evicts 10; null contributes 0 to sum)
        pos3: [None,30] → 30 (window evicts 20)
    Sorted: [10, 30, 20, 30]. Map by row_id [1,2,0,3]:
        orig 0 → pos2 → 20
        orig 1 → pos0 → 10
        orig 2 → pos1 → 30
        orig 3 → pos3 → 30
    Expected: [20, 10, 30, 30] (matches pandas rolling skipna=True).
    """
    df = make_df(tmp_path, data={
        "ts": [3, 1, 2, 4],
        "v": [None, 10.0, 20.0, 30.0],
    })
    res = df.rolling(2, "sum", on="ts", columns=["v"], min_periods=1)
    vals = read_col(res, tmp_path, "v_rolling_sum")
    assert vals == [20.0, 10.0, 30.0, 30.0]


def test_rolling_on_multibatch(tmp_path):
    """chunk=1: batch boundaries don't affect result."""
    df = make_df(tmp_path, chunk=1)
    res = df.rolling(2, "sum", on="ts", columns=["v"])
    vals = read_col(res, tmp_path, "v_rolling_sum")
    assert vals == [90.0, 50.0, None, 70.0, 30.0]


def test_rolling_on_preserves_original(tmp_path):
    df = make_df(tmp_path)
    _ = df.rolling(2, "sum", on="ts", columns=["v"])
    vals = read_col(df, tmp_path, "v")
    assert vals == [50.0, 30.0, 10.0, 40.0, 20.0]


def test_rolling_on_all_columns(tmp_path):
    """columns=None: all columns rolled (numeric only for agg=sum).

    Sorted by ts [1,2,3,4,5]:
      v: [10, 20, 30, 40, 50]  → v_sum w2 (min_periods=2): [None, 30, 50, 70, 90]
      w: [3, 5, 2, 4, 1]       → w_sum w2: [None, 8, 7, 6, 5]
    Map back to orig ts order [5, 3, 1, 4, 2]:
      v: orig 0(ts=5)→pos4→90, orig 1(ts=3)→pos2→50, orig 2(ts=1)→pos0→None,
         orig 3(ts=4)→pos3→70, orig 4(ts=2)→pos1→30
      w: orig 0(ts=5)→pos4→5,  orig 1(ts=3)→pos2→7,  orig 2(ts=1)→pos0→None,
         orig 3(ts=4)→pos3→6,  orig 4(ts=2)→pos1→8
    """
    df = make_df(tmp_path, data={
        "ts": [5, 3, 1, 4, 2],
        "v": [50.0, 30.0, 10.0, 40.0, 20.0],
        "w": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    res = df.rolling(2, "sum", on="ts", columns=["v", "w"])
    v = read_col(res, tmp_path, "v_rolling_sum")
    w = read_col(res, tmp_path, "w_rolling_sum")
    assert v == [90.0, 50.0, None, 70.0, 30.0]
    assert w == [5.0, 7.0, None, 6.0, 8.0]


def test_rolling_on_then_sort(tmp_path):
    """Explicit sort after works."""
    df = make_df(tmp_path)
    res = df.rolling(2, "sum", on="ts", columns=["v"]).sort_values("ts")
    out = tmp_path / "o.parquet"
    res.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["ts"] for r in rows] == [1, 2, 3, 4, 5]
    assert [r["v_rolling_sum"] for r in rows] == [None, 30.0, 50.0, 70.0, 90.0]


# ---------- Errors ----------

def test_rolling_on_missing_column_raises(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(KeyError, match="nope"):
        df.rolling(2, "sum", on="nope", columns=["v"])


def test_rolling_on_non_numeric_raises(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(TypeError, match="numeric"):
        df.rolling(2, "sum", on="ts", columns=["cat"])


# ---------- Backward compat ----------

def test_rolling_no_on_unchanged(tmp_path):
    """Backward compat: rolling() without on — positional."""
    df = make_df(tmp_path)
    res = df.rolling(2, "sum", columns=["v"])
    # positional: 50,30,10,40,20 → sum w2: None,80,40,50,60
    vals = read_col(res, tmp_path, "v_rolling_sum")
    assert vals == [None, 80.0, 40.0, 50.0, 60.0]


# ---------- AST guards ----------

def test_rolling_on_second_sort_renames_row_id():
    """AST guard: _rolling_on second external_sort must NOT sort by raw
    row_id (external_sort appends its own → duplicate → ArrowInvalid).

    Регрессия v0.8.2-fix3 (§13.4/122). Same fix for _rolling_on (v0.8.4).
    """
    import ast as _ast
    import inspect as _inspect
    from cadw.dataframe import CadwDataFrame

    src = _inspect.getsource(CadwDataFrame._rolling_on)
    tree = _ast.parse(_inspect.cleandoc(src))

    ext_calls: list[dict] = []
    for node in _ast.walk(tree):
        if (isinstance(node, _ast.Call)
                and isinstance(node.func, _ast.Name)
                and node.func.id == "external_sort"):
            keys = None
            for kw in node.keywords:
                if kw.arg == "key_columns":
                    if isinstance(kw.value, _ast.List):
                        keys = [
                            elt.value for elt in kw.value.elts
                            if isinstance(elt, _ast.Constant)
                        ]
            ext_calls.append({"lineno": node.lineno, "keys": keys})

    assert len(ext_calls) >= 2, (
        f"expected >=2 external_sort in _rolling_on; got {len(ext_calls)}"
    )
    second = ext_calls[-1]
    assert "row_id" not in (second["keys"] or []), (
        f"second external_sort (line {second['lineno']}) uses "
        f"key_columns={second['keys']}; must use __orig_rid"
    )
    assert "__orig_rid" in (second["keys"] or []), (
        f"second external_sort (line {second['lineno']}) must sort by "
        f"__orig_rid; got {second['keys']}"
    )


def test_rolling_on_iter_zipped_appends_not_replaces():
    """AST guard: _rolling_on `_iter_zipped` must APPEND rolling columns
    (rolling adds new {col}_rolling_{agg}, doesn't replace).

    Note: _shift_on REPLACES (shift replaces values); _rolling_on APPENDS
    (new column). Opposite invariants — both AST-guarded separately.
    """
    import ast as _ast
    import inspect as _inspect
    from cadw.dataframe import CadwDataFrame

    src = _inspect.getsource(CadwDataFrame._rolling_on)
    tree = _ast.parse(_inspect.cleandoc(src))
    iter_zipped_src = None
    for node in _ast.walk(tree):
        if (isinstance(node, _ast.FunctionDef)
                and node.name == "_iter_zipped"):
            iter_zipped_src = _ast.get_source_segment(
                _inspect.cleandoc(src), node,
            )
            break
    assert iter_zipped_src is not None
    # APPEND-style: `for out_name in rolling_out_cols:` + `arrays.append(take_batch.column(...))`
    assert "rolling_out_cols" in iter_zipped_src
    assert "arrays.append(take_batch.column(" in iter_zipped_src or            "arrays.append(take_batch.column" in iter_zipped_src, (
        "_iter_zipped must append take_batch.column(out_name) — "
        "rolling adds new columns (replace-style would be a bug)"
    )
    assert "combine_chunks()" in iter_zipped_src, (
        "_iter_zipped must use combine_chunks().to_batches()[0] "
        "(ChunkedArray → Array)"
    )

def test_rolling_on_min_with_nulls(tmp_path):
    """S1 regression (on=col): min + null must not crash."""
    df = make_df(tmp_path, data={
        "ts": [1, 2, 3, 4, 5],
        "v": [3.0, None, 1.0, None, 5.0],
    })
    res = df.rolling(2, "min", on="ts", columns=["v"], min_periods=1)
    vals = read_col(res, tmp_path, "v_rolling_min")
    # Sorted by ts is already 1..5; positional nulls consume slot:
    # [3, 3, 1, 1, 5]. Row_id = original position.
    assert vals == [3.0, 3.0, 1.0, 1.0, 5.0]


def test_rolling_on_max_with_nulls(tmp_path):
    """S1 regression (on=col): max + null must not crash.

    ts already sorted 1..5; same positional walk as positional test:
      pos0 [3.0]        → 3.0
      pos1 [3.0, None]  → 3.0
      pos2 [None, 1.0]  → 1.0  (3.0 evicted)
      pos3 [1.0, None]  → 1.0
      pos4 [None, 5.0]  → 5.0
    """
    df = make_df(tmp_path, data={
        "ts": [1, 2, 3, 4, 5],
        "v": [3.0, None, 1.0, None, 5.0],
    })
    res = df.rolling(2, "max", on="ts", columns=["v"], min_periods=1)
    vals = read_col(res, tmp_path, "v_rolling_max")
    assert vals == [3.0, 3.0, 1.0, 1.0, 5.0]

