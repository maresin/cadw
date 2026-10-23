"""Тесты ленивых __getitem__ в CadwDataFrame (03c)."""
import inspect
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import cadw.dataframe as df_mod
from cadw import CadwDataFrame, MemoryGuard


def make_df(tmp_path: Path, n: int = 20) -> CadwDataFrame:
    t = pa.table({
        "cat": [f"g{i % 5}" for i in range(n)],
        "val": list(range(n)),
    })
    p = tmp_path / "src.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, tmp_path / "ds", chunk_size=5,
        memory_guard=MemoryGuard(max_memory_mb=50),
        spill_dir=tmp_path / "spill",
    )


def read_parquet_vals(path):
    return pq.read_table(path).to_pylist()


# ---------- column selection ----------


def _iter_dataframe_sources():
    """v0.9.26h2 (§13.4/292): iterate all dataframe package sources.
    Replaces single-file scans; new extracted modules auto-covered.
    """
    import pathlib as _p
    pkg = _p.Path("cadw/dataframe")
    if pkg.is_dir():
        for f in sorted(pkg.rglob("*.py")):
            yield f, f.read_text()
    else:
        # legacy single-file fallback (should not trigger post-v0.9.26g)
        legacy = _p.Path("cadw/dataframe.py")
        if legacy.exists():
            yield legacy, legacy.read_text()

def test_select_columns(tmp_path):
    df = make_df(tmp_path)
    sub = df[["val"]]
    assert sub.columns == ["val"]
    assert sub.shape == (20, 1)
    out = tmp_path / "o.parquet"
    sub.write_parquet(out)
    t = pq.read_table(out)
    assert t.schema.names == ["val"]


def test_select_columns_missing_raises(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(KeyError):
        df[["nope"]]


# ---------- slice ----------

def test_slice_basic(tmp_path):
    df = make_df(tmp_path)
    sliced = df[5:10]
    out = tmp_path / "o.parquet"
    sliced.write_parquet(out)
    vals = [r["val"] for r in read_parquet_vals(out)]
    assert vals == [5, 6, 7, 8, 9]


def test_slice_with_step(tmp_path):
    df = make_df(tmp_path)
    sliced = df[0:20:3]
    out = tmp_path / "o.parquet"
    sliced.write_parquet(out)
    vals = [r["val"] for r in read_parquet_vals(out)]
    assert vals == [0, 3, 6, 9, 12, 15, 18]


# ---------- predicate filters ----------

def test_filter_gt(tmp_path):
    df = make_df(tmp_path)
    filtered = df[df["val"] > 15]
    out = tmp_path / "o.parquet"
    filtered.write_parquet(out)
    vals = [r["val"] for r in read_parquet_vals(out)]
    assert vals == [16, 17, 18, 19]


def test_filter_eq(tmp_path):
    df = make_df(tmp_path)
    filtered = df[df["val"] == 7]
    out = tmp_path / "o.parquet"
    filtered.write_parquet(out)
    vals = [r["val"] for r in read_parquet_vals(out)]
    assert vals == [7]


def test_filter_and(tmp_path):
    df = make_df(tmp_path)
    filtered = df[(df["val"] >= 5) & (df["val"] < 10)]
    out = tmp_path / "o.parquet"
    filtered.write_parquet(out)
    vals = [r["val"] for r in read_parquet_vals(out)]
    assert vals == [5, 6, 7, 8, 9]


def test_filter_or(tmp_path):
    df = make_df(tmp_path)
    filtered = df[(df["val"] < 3) | (df["val"] > 17)]
    out = tmp_path / "o.parquet"
    filtered.write_parquet(out)
    vals = [r["val"] for r in read_parquet_vals(out)]
    assert vals == [0, 1, 2, 18, 19]


def test_filter_not(tmp_path):
    df = make_df(tmp_path)
    filtered = df[~(df["val"] < 17)]
    out = tmp_path / "o.parquet"
    filtered.write_parquet(out)
    vals = [r["val"] for r in read_parquet_vals(out)]
    assert vals == [17, 18, 19]


# ---------- chained filters ----------

def test_slice_then_filter(tmp_path):
    df = make_df(tmp_path)
    result = df[5:15][df["val"] > 10]
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    vals = [r["val"] for r in read_parquet_vals(out)]
    assert vals == [11, 12, 13, 14]


def test_filter_then_slice(tmp_path):
    df = make_df(tmp_path)
    result = df[df["val"] > 10][1:4]
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    vals = [r["val"] for r in read_parquet_vals(out)]
    assert vals == [12, 13, 14]


def test_filter_then_select(tmp_path):
    df = make_df(tmp_path)
    result = df[df["val"] > 15][["val"]]
    assert result.columns == ["val"]
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    t = pq.read_table(out)
    assert t.column("val").to_pylist() == [16, 17, 18, 19]


def test_filter_on_non_selected_column(tmp_path):
    """Фильтр по колонке не из selection работает: filter-колонка добавляется, потом select."""
    df = make_df(tmp_path)
    result = df[["cat"]][df["val"] > 15]
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = read_parquet_vals(out)
    assert len(rows) == 4
    assert set(rows[0].keys()) == {"cat"}


# ---------- head/count/shape ----------

def test_head_after_filter(tmp_path):
    df = make_df(tmp_path)
    h = df[df["val"] > 15].head(2)
    assert h.num_rows == 2
    assert h.column("val").to_pylist() == [16, 17]


def test_count_after_filter(tmp_path):
    df = make_df(tmp_path)
    assert df[df["val"] > 15].count() == 4


def test_shape_is_upper_bound(tmp_path):
    df = make_df(tmp_path)
    assert df[df["val"] > 15].shape[0] == 20  # upper bound


# ---------- filter + groupby/sort ----------

def test_filter_then_groupby(tmp_path):
    df = make_df(tmp_path)
    result = df[df["val"] > 5].groupby("cat").agg("val", "count")
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = {r["cat"]: r["val_count"] for r in read_parquet_vals(out)}
    # значения >5: 6,7,8,9,10,11,12,13,14,15,16,17,18,19
    # cat: g(val%5). 6→g1, 7→g2, 8→g3, 9→g4, 10→g0, 11→g1, 12→g2, ...
    assert sum(rows.values()) == 14


def test_filter_then_sort(tmp_path):
    df = make_df(tmp_path)
    result = df[df["val"] > 15].sort_values("val", ascending=False)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    vals = [r["val"] for r in read_parquet_vals(out)]
    assert vals == [19, 18, 17, 16]


# ---------- facade guards ----------

def test_facade_has_no_as_py_ast():
    import ast
    src = inspect.getsource(df_mod._core)
    tree = ast.parse(src)
    bad = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "as_py"
    ]
    assert not bad, "facade calls .as_py on data"


def test_facade_has_no_row_loop_ast():
    import ast
    src = inspect.getsource(df_mod._core)
    tree = ast.parse(src)
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        it = node.iter
        if (isinstance(it, ast.Call)
                and isinstance(it.func, ast.Name)
                and it.func.id == "range"):
            for arg in it.args:
                if isinstance(arg, ast.Attribute) and arg.attr == "num_rows":
                    offenders.append(ast.unparse(node.target))
    assert not offenders
