"""Тесты CadwDataFrame (патч 03b): честный фасад над Dataset + примитивы."""
import inspect
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import cadw.dataframe as df_mod
from cadw import CadwDataFrame, Dataset, MemoryGuard


def make_table(n: int = 20) -> pa.Table:
    return pa.table({
        "cat": [f"g{i % 5}" for i in range(n)],
        "val": list(range(n)),
    })


def write_parquet(tmp_path: Path, n: int = 20) -> Path:
    t = make_table(n)
    p = tmp_path / "src.parquet"
    pq.write_table(t, p)
    return p


def make_df(tmp_path: Path, n: int = 20, chunk: int = 5) -> CadwDataFrame:
    p = write_parquet(tmp_path, n)
    return CadwDataFrame.from_parquet(
        p, tmp_path / "ds", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=50),
        spill_dir=tmp_path / "spill",
    )


# ---------- construction ----------

def test_from_parquet(tmp_path):
    df = make_df(tmp_path)
    assert df.shape == (20, 2)
    assert set(df.columns) == {"cat", "val"}


def test_from_dataset(tmp_path):
    p = write_parquet(tmp_path)
    ds = Dataset.from_parquet(p, tmp_path / "ds", chunk_size=5)
    df = CadwDataFrame(ds, memory_guard=MemoryGuard(max_memory_mb=50),
                       spill_dir=tmp_path / "spill")
    assert df.shape == (20, 2)


def test_init_requires_dataset():
    with pytest.raises(TypeError):
        CadwDataFrame("not a dataset")


def test_columns_not_found_raises(tmp_path):
    p = write_parquet(tmp_path)
    ds = Dataset.from_parquet(p, tmp_path / "ds", chunk_size=5)
    with pytest.raises(KeyError):
        CadwDataFrame(ds, columns=["nope"])


# ---------- reading ----------

def test_head(tmp_path):
    df = make_df(tmp_path)
    h = df.head(3)
    assert h.num_rows == 3
    assert h.schema.names == ["cat", "val"]


def test_head_zero(tmp_path):
    df = make_df(tmp_path)
    h = df.head(0)
    assert h.num_rows == 0


def test_to_batches(tmp_path):
    df = make_df(tmp_path)
    sizes = [b.num_rows for b in df.to_batches(batch_size=7)]
    assert sum(sizes) == 20
    assert all(b.schema.names == ["cat", "val"] for b in df.to_batches())


# ---------- groupby ----------

def test_groupby_mean(tmp_path):
    df = make_df(tmp_path)
    result = df.groupby("cat").agg("val", "mean")
    assert set(result.columns) == {"cat", "val_mean"}
    # roundtrip через write_parquet + чтение
    out = tmp_path / "agg.parquet"
    result.write_parquet(out)
    t = pq.read_table(out)
    assert t.num_rows == 5
    vals = {r["cat"]: r["val_mean"] for r in t.to_pylist()}
    # для cat g0: i=0,5,10,15 → mean=7.5
    assert vals["g0"] == 7.5


def test_groupby_sum(tmp_path):
    df = make_df(tmp_path)
    result = df.groupby("cat").agg("val", "sum")
    out = tmp_path / "agg.parquet"
    result.write_parquet(out)
    t = pq.read_table(out)
    vals = {r["cat"]: r["val_sum"] for r in t.to_pylist()}
    assert vals["g0"] == 0 + 5 + 10 + 15  # 30
    assert vals["g1"] == 1 + 6 + 11 + 16  # 34


def test_groupby_multiple_aggs(tmp_path):
    df = make_df(tmp_path)
    result = df.groupby("cat").agg(["val", "val"], ["sum", "count"])
    out = tmp_path / "agg.parquet"
    result.write_parquet(out)
    t = pq.read_table(out)
    vals = {r["cat"]: (r["val_sum"], r["val_count"]) for r in t.to_pylist()}
    assert vals["g0"] == (30, 4)


# ---------- unique ----------

def test_unique(tmp_path):
    df = make_df(tmp_path)
    uniq = df.unique("cat")
    assert uniq.columns == ["cat"]
    assert uniq.shape[0] == 5
    out = tmp_path / "u.parquet"
    uniq.write_parquet(out)
    t = pq.read_table(out)
    assert sorted(t.column("cat").to_pylist()) == ["g0", "g1", "g2", "g3", "g4"]


# ---------- sort_values ----------

def test_sort_values_ascending(tmp_path):
    df = make_df(tmp_path)
    sorted_df = df.sort_values("val", ascending=True)
    out = tmp_path / "sorted.parquet"
    sorted_df.write_parquet(out)
    t = pq.read_table(out)
    assert t.column("val").to_pylist() == list(range(20))


def test_sort_values_descending(tmp_path):
    df = make_df(tmp_path)
    sorted_df = df.sort_values("val", ascending=False)
    out = tmp_path / "sorted.parquet"
    sorted_df.write_parquet(out)
    t = pq.read_table(out)
    assert t.column("val").to_pylist() == list(range(19, -1, -1))


def test_sort_values_missing_raises(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(KeyError):
        df.sort_values("nope")


# ---------- integration ----------

def test_groupby_chain_write(tmp_path):
    df = make_df(tmp_path)
    out = tmp_path / "result.parquet"
    (df.groupby("cat").agg("val", "mean")).write_parquet(out)
    t = pq.read_table(out)
    assert t.num_rows == 5
    assert t.column("val_mean").to_pylist() is not None


def test_full_pipeline_parquet_to_parquet(tmp_path):
    """Исходный parquet → sort → groupby → parquet."""
    df = make_df(tmp_path)
    pipeline = df.sort_values("val").groupby("cat").agg("val", "mean")
    out = tmp_path / "final.parquet"
    pipeline.write_parquet(out)
    t = pq.read_table(out)
    assert t.num_rows == 5


# ---------- facade is honest (AST-based) ----------
#
# Guard проверяет реальные вызовы в AST, а не текст: строки и docstring,
# упоминающие ".as_py()" как паттерн-запрет, не должны ломать проверку.

import ast


def _call_attr_names(module: ast.Module) -> list[str]:
    """Имена атрибутов, вызываемых как Method(...). Только реальные Call."""
    out: list[str] = []
    for node in ast.walk(module):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            out.append(node.func.attr)
    return out


def test_facade_has_no_as_py():
    """CadwDataFrame не должен вызывать .as_py() на данных (AST-level)."""
    src = inspect.getsource(df_mod._core)
    tree = ast.parse(src)
    bad = [n for n in _call_attr_names(tree) if n == "as_py"]
    assert not bad, (
        f"CadwDataFrame calls .as_py() {len(bad)} time(s); "
        f"delegate to primitives"
    )


def test_facade_has_no_row_loops():
    """Не должно быть for-range с диапазоном batch.num_rows (AST-level)."""
    src = inspect.getsource(df_mod._core)
    tree = ast.parse(src)
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        it = node.iter
        if not (
            isinstance(it, ast.Call)
            and isinstance(it.func, ast.Name)
            and it.func.id == "range"
        ):
            continue
        for arg in it.args:
            if (
                isinstance(arg, ast.Attribute)
                and arg.attr == "num_rows"
            ):
                offenders.append(ast.unparse(node.target))
    assert not offenders, (
        f"row-level loops over batch.num_rows in facade: {offenders}"
    )

def test_facade_does_not_define_sampling_helpers():
    """`_peek_first`/`_data_chars_for_join` не живут в dataframe.py.

    Регрессия 11-B-2c-3-fix2: helpers были добавлены в dataframe.py и
    использовали as_py, нарушая test_facade_has_no_as_py. Правило:
    helpers для сэмплинга — в cadw/core/sampling.py; facade остаётся
    без as_py.
    """
    src = inspect.getsource(df_mod._core)
    for name in ("_peek_first", "_data_chars_for_join"):
        assert f"def {name}(" not in src, (
            f"{name!r} must live in cadw/core/sampling.py, "
            f"not in facade"
        )

def test_groupby_same_col_as_by(tmp_path):
    """df.groupby('cat').agg('cat', 'count') — agg == by (§13.4/54)."""
    df = make_df(tmp_path)
    result = df.groupby("cat").agg("cat", "count")
    out = tmp_path / "same_col.parquet"
    result.write_parquet(out)
    rows = {r["cat"]: r["cat_count"] for r in pq.read_table(out).to_pylist()}
    # 20 rows, 5 groups → 4 each
    assert all(v == 4 for v in rows.values())
    assert len(rows) == 5

