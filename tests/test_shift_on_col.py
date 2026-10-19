"""v0.8.2: shift/diff on=col — time-aware via external_sort."""
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
            "v": [50, 30, 10, 40, 20],
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


def read_all(df, tmp_path):
    out = tmp_path / "o.parquet"
    df.write_parquet(out)
    return pq.read_table(out)


# ---------- shift(on=col) ----------

def test_shift_on_basic(tmp_path):
    """ts=1,2,3,4,5 order → shift(1) lag v."""
    df = make_df(tmp_path)
    res = df.shift(1, on="ts", columns=["v"])
    # Original order: ts=5,3,1,4,2. Shift by ts order:
    # sorted: ts=1(v=10),2(v=20),3(v=30),4(v=40),5(v=50)
    # shifted: None,10,20,30,40 → back to orig: ts=5→40, 3→20, 1→None, 4→30, 2→10
    vals = read_col(res, tmp_path, "v")
    assert vals == [40, 20, None, 30, 10]


def test_shift_on_fill_value(tmp_path):
    df = make_df(tmp_path)
    res = df.shift(1, fill_value=0, on="ts", columns=["v"])
    vals = read_col(res, tmp_path, "v")
    assert vals == [40, 20, 0, 30, 10]


def test_shift_on_k2(tmp_path):
    df = make_df(tmp_path)
    res = df.shift(2, on="ts", columns=["v"])
    # sorted v: 10,20,30,40,50; shift(2): None,None,10,20,30
    # orig: ts=5→30, 3→10, 1→None, 4→20, 2→None
    vals = read_col(res, tmp_path, "v")
    assert vals == [30, 10, None, 20, None]


def test_shift_on_string_column(tmp_path):
    df = make_df(tmp_path)
    res = df.shift(1, on="ts", columns=["cat"])
    # sorted cat by ts: a(1),b(2),c(3),d(4),e(5)
    # shifted: None,a,b,c,d
    # orig: ts=5(e)→d, 3(c)→b, 1(a)→None, 4(d)→c, 2(b)→a
    vals = read_col(res, tmp_path, "cat")
    assert vals == ["d", "b", None, "c", "a"]


def test_shift_on_preserves_original(tmp_path):
    df = make_df(tmp_path)
    _ = df.shift(1, on="ts", columns=["v"])
    vals = read_col(df, tmp_path, "v")
    assert vals == [50, 30, 10, 40, 20]


def test_shift_on_all_columns(tmp_path):
    df = make_df(tmp_path)
    res = df.shift(1, on="ts")  # all columns
    vals_v = read_col(res, tmp_path, "v")
    vals_ts = read_col(res, tmp_path, "ts")
    # sorted ts: 1,2,3,4,5; shifted: None,1,2,3,4
    # orig: ts=5→4, 3→2, 1→None, 4→3, 2→1
    assert vals_ts == [4, 2, None, 3, 1]


def test_shift_on_missing_column_raises(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(KeyError, match="nope"):
        df.shift(1, on="nope", columns=["v"])


def test_shift_on_with_nulls_in_shift_col(tmp_path):
    df = make_df(tmp_path, data={
        "ts": [3, 1, 2, 4],
        "v": [None, 10, 20, 30],
    })
    res = df.shift(1, on="ts", columns=["v"])
    # sorted ts: 1(v=10),2(v=20),3(v=None),4(v=30)
    # shifted: None,10,20,None
    # orig: ts=3→20, 1→None, 2→10, 4→None
    vals = read_col(res, tmp_path, "v")
    assert vals == [20, None, 10, None]


def test_shift_on_multibatch(tmp_path):
    """chunk=1: batch boundaries don't affect result."""
    df = make_df(tmp_path, chunk=1)
    res = df.shift(1, on="ts", columns=["v"])
    vals = read_col(res, tmp_path, "v")
    assert vals == [40, 20, None, 30, 10]


def test_shift_no_on_unchanged(tmp_path):
    """Backward compat: shift() without on — positional."""
    df = make_df(tmp_path)
    res = df.shift(1, columns=["v"])
    # original order: 50,30,10,40,20 → shifted: None,50,30,10,40
    vals = read_col(res, tmp_path, "v")
    assert vals == [None, 50, 30, 10, 40]


# ---------- diff(on=col) ----------

def test_diff_on_basic(tmp_path):
    df = make_df(tmp_path)
    res = df.diff(1, on="ts", columns=["v"])
    # sorted v: 10,20,30,40,50; diff: None,10,10,10,10
    # orig: ts=5→10, 3→10, 1→None, 4→10, 2→10
    vals = read_col(res, tmp_path, "v")
    assert vals == [10, 10, None, 10, 10]


def test_diff_on_k2(tmp_path):
    df = make_df(tmp_path)
    res = df.diff(2, on="ts", columns=["v"])
    # sorted diff(2): None,None,20,20,20
    # orig: ts=5→20, 3→20, 1→None, 4→20, 2→None
    vals = read_col(res, tmp_path, "v")
    assert vals == [20, 20, None, 20, None]


def test_diff_on_preserves_original(tmp_path):
    df = make_df(tmp_path)
    _ = df.diff(1, on="ts", columns=["v"])
    vals = read_col(df, tmp_path, "v")
    assert vals == [50, 30, 10, 40, 20]


def test_diff_on_non_numeric_raises(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(TypeError, match="numeric"):
        df.diff(1, on="ts", columns=["cat"])


def test_diff_on_unsorted_non_monotonic(tmp_path):
    """ts may be arbitrary; sort is stable, tie-break by row_id."""
    df = make_df(tmp_path, data={
        "ts": [3, 1, 3, 2, 1],  # non-monotonic with duplicates
        "v": [10, 20, 30, 40, 50],
    })
    res = df.diff(1, on="ts", columns=["v"])
    # Stable sort by ts (row_id asc for ties):
    # row 1 (ts=1, v=20), row 4 (ts=1, v=50), row 3 (ts=2, v=40),
    # row 0 (ts=3, v=10), row 2 (ts=3, v=30)
    # diff: None, 30, -10, -30, 20
    # orig: row0→-30, row1→None, row2→20, row3→-10, row4→30
    vals = read_col(res, tmp_path, "v")
    assert vals == [-30, None, 20, -10, 30]


def test_diff_no_on_unchanged(tmp_path):
    """Backward compat: diff() without on — positional."""
    df = make_df(tmp_path)
    res = df.diff(1, columns=["v"])
    # positional: 50,30,10,40,20 → diff: None,-20,-20,30,-20
    vals = read_col(res, tmp_path, "v")
    assert vals == [None, -20, -20, 30, -20]


def test_diff_on_with_nulls(tmp_path):
    df = make_df(tmp_path, data={
        "ts": [1, 2, 3, 4],
        "v": [10.0, None, 30.0, 40.0],
    })
    res = df.diff(1, on="ts", columns=["v"])
    # sorted: 10, None, 30, 40; diff: None, None(null prop), None(prop), 10
    vals = read_col(res, tmp_path, "v")
    assert vals[0] is None
    assert vals[1] is None
    assert vals[2] is None
    assert abs(vals[3] - 10.0) < 1e-9


# ---------- shift/diff chain with sort ----------

def test_shift_on_then_sort(tmp_path):
    """Result preserves original order; explicit sort after works."""
    df = make_df(tmp_path)
    res = df.shift(1, on="ts", columns=["v"]).sort_values("ts")
    out = tmp_path / "o.parquet"
    res.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    # after sort by ts: 1,2,3,4,5; shifted v: None,10,20,30,40
    assert [r["ts"] for r in rows] == [1, 2, 3, 4, 5]
    assert [r["v"] for r in rows] == [None, 10, 20, 30, 40]

def test_shift_on_second_sort_renames_row_id():
    """AST guard: _shift_on второй external_sort не должен использовать
    key_columns=["row_id"] — это вызовет duplicate row_id (ArrowInvalid).

    Регрессия v0.8.2-fix3: external_sort ВСЕГДА append_column("row_id", ...).
    Второй sort (by row_id) на потоке с уже-существующим row_id → два
    row_id → pc.sort_indices падает. Fix: переименовать row_id → __orig_rid.
    """
    import ast as _ast
    import inspect as _inspect
    from cadw.dataframe import CadwDataFrame

    src = _inspect.getsource(CadwDataFrame._shift_on)
    tree = _ast.parse(_inspect.cleandoc(src))

    # Find external_sort calls; extract key_columns arg
    external_sort_calls: list[dict] = []
    for node in _ast.walk(tree):
        if (isinstance(node, _ast.Call)
                and isinstance(node.func, _ast.Name)
                and node.func.id == "external_sort"):
            keys_arg = None
            for kw in node.keywords:
                if kw.arg == "key_columns":
                    # Expect List(Constant("__orig_rid"))
                    if isinstance(kw.value, _ast.List):
                        keys_arg = [
                            elt.value for elt in kw.value.elts
                            if isinstance(elt, _ast.Constant)
                        ]
            external_sort_calls.append({"lineno": node.lineno, "keys": keys_arg})

    assert len(external_sort_calls) >= 2, (
        f"expected >=2 external_sort in _shift_on; got {len(external_sort_calls)}"
    )
    # Second sort (last in list, since first-by-on is earlier)
    second = external_sort_calls[-1]
    assert "row_id" not in (second["keys"] or []), (
        f"second external_sort (line {second['lineno']}) uses key_columns="
        f"{second['keys']}; must use __orig_rid (not raw row_id — "
        f"external_sort appends its own row_id → duplicate column)"
    )
    assert "__orig_rid" in (second["keys"] or []), (
        f"second external_sort (line {second['lineno']}) must sort by "
        f"__orig_rid; got {second['keys']}"
    )

def test_shift_on_iter_zipped_replace_not_append():
    """AST guard: _shift_on `_iter_zipped` must REPLACE shifted columns,
    not append (shift replaces column values, doesn't add new column).

    Регрессия v0.8.2-fix4:
        Schema and number of arrays unequal — out_schema=3 (self._columns),
        arrays=4 (3 original + 1 shifted appended).

    Проверяем:
      - НЕТ `arrays = [ob.column(c) for c in self._columns]` (append-style).
      - ЕСТЬ `if c in shift_cols:` / `if c in cols:` (replace-style).
    """
    import ast as _ast
    import inspect as _inspect
    from cadw.dataframe import CadwDataFrame

    src = _inspect.getsource(CadwDataFrame._shift_on)
    # Find `_iter_zipped` inner function
    tree = _ast.parse(_inspect.cleandoc(src))
    iter_zipped_src = None
    for node in _ast.walk(tree):
        if (isinstance(node, _ast.FunctionDef)
                and node.name == "_iter_zipped"):
            iter_zipped_src = _ast.get_source_segment(
                _inspect.cleandoc(src), node,
            )
            break
    assert iter_zipped_src is not None, "_iter_zipped not found in _shift_on"

    # Check for append-style pattern (bad)
    assert "arrays = [ob.column(c) for c in self._columns]" not in iter_zipped_src, (
        "_iter_zipped uses append-style (arrays = [ob.column...]) — must "
        "REPLACE shifted columns via `if c in shift_cols:`"
    )
    # Check for replace-style pattern (good)
    assert "if c in shift_cols:" in iter_zipped_src or "if c in cols:" in iter_zipped_src, (
        "_iter_zipped must use `if c in shift_cols:` (replace semantics)"
    )
    # Check ChunkedArray handling
    assert "combine_chunks()" in iter_zipped_src, (
        "_iter_zipped must use .combine_chunks().to_batches()[0] for "
        "Table→Array conversion (Table.column returns ChunkedArray)"
    )

# ---------- v0.8.6: multi-column on ----------

def test_shift_on_multi_key(tmp_path):
    """on=['ts','sensor'] — lexicographic composite sort."""
    df = make_df(tmp_path, data={
        "ts": [2, 1, 1, 2],
        "sensor": ["a", "a", "b", "b"],
        "v": [40, 10, 20, 30],
    })
    res = df.shift(1, on=["ts", "sensor"], columns=["v"])
    # Sorted by (ts, sensor): row1(1,a)=10, row2(1,b)=20, row0(2,a)=40,
    # row3(2,b)=30. shift(1): [None, 10, 20, 40].
    # Back to original row_id order [0,1,2,3]: [20, None, 10, 40]
    vals = read_col(res, tmp_path, "v")
    assert vals == [20, None, 10, 40]


def test_diff_on_multi_key(tmp_path):
    df = make_df(tmp_path, data={
        "ts": [2, 1, 1, 2],
        "sensor": ["a", "a", "b", "b"],
        "v": [40, 10, 20, 30],
    })
    res = df.diff(1, on=["ts", "sensor"], columns=["v"])
    # Sorted v: [10, 20, 40, 30]. diff(1): [None, 10, 20, -10].
    # Back: [20, None, 10, -10]
    vals = read_col(res, tmp_path, "v")
    assert vals == [20, None, 10, -10]


def test_shift_on_single_element_list(tmp_path):
    """on=['ts'] behaves like on='ts'."""
    df = make_df(tmp_path)
    r1 = df.shift(1, on="ts", columns=["v"])
    r2 = df.shift(1, on=["ts"], columns=["v"])
    (tmp_path / "r1").mkdir()
    (tmp_path / "r2").mkdir()
    v1 = read_col(r1, tmp_path / "r1", "v")
    v2 = read_col(r2, tmp_path / "r2", "v")
    assert v1 == v2


def test_shift_on_empty_list_raises(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(ValueError, match="non-empty"):
        df.shift(1, on=[], columns=["v"])


def test_shift_on_multi_key_stable_tiebreak(tmp_path):
    """Equal on-values: row_id tie-break keeps original order."""
    df = make_df(tmp_path, data={
        "ts": [1, 1, 2],
        "sensor": ["a", "a", "b"],
        "v": [10, 20, 30],
    })
    res = df.shift(1, on=["ts", "sensor"], columns=["v"])
    # Sorted by (ts, sensor, row_id): row0(1,a)=10, row1(1,a)=20,
    # row2(2,b)=30. shift(1): [None, 10, 20].
    vals = read_col(res, tmp_path, "v")
    assert vals == [None, 10, 20]

