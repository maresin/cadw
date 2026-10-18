"""Тесты Dataset (патч 03a): базовое хранилище per-column Arrow IPC."""
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import Dataset


def batches_of(table: pa.Table, chunk: int):
    for i in range(0, table.num_rows, chunk):
        yield table.slice(i, chunk).to_batches()[0]


def make_table(n: int = 20) -> pa.Table:
    return pa.table({
        "cat": [f"g{i % 5}" for i in range(n)],
        "val": list(range(n)),
    })


# ---------- from_batches / roundtrip ----------

def test_from_batches_creates_layout(tmp_path):
    ds = Dataset.from_batches(make_table(20).schema, batches_of(make_table(20), 5), tmp_path / "ds")
    assert set(ds.columns) == {"cat", "val"}
    assert ds.total_rows == 20
    assert (tmp_path / "ds" / "columns" / "cat.arrow").exists()
    assert (tmp_path / "ds" / "columns" / "val.arrow").exists()
    assert (tmp_path / "ds" / "dataset_meta.json").exists()


def test_from_batches_roundtrip(tmp_path):
    t = make_table(20)
    ds = Dataset.from_batches(t.schema, batches_of(t, 5), tmp_path / "ds")
    got = pa.Table.from_batches(list(ds.iter_batches()))
    assert got.num_rows == 20
    assert got.column("cat").to_pylist() == t.column("cat").to_pylist()
    assert got.column("val").to_pylist() == t.column("val").to_pylist()


def test_chunk_boundaries(tmp_path):
    t = make_table(20)
    ds = Dataset.from_batches(t.schema, batches_of(t, 7), tmp_path / "ds")
    # 20 rows, chunk=7 → batches of 7,7,6
    assert ds.chunk_boundaries == [7, 7, 6]


# ---------- from_parquet ----------

def test_from_parquet(tmp_path):
    t = make_table(20)
    src = tmp_path / "src.parquet"
    pq.write_table(t, src)
    ds = Dataset.from_parquet(src, tmp_path / "ds", chunk_size=5)
    assert set(ds.columns) == {"cat", "val"}
    assert ds.total_rows == 20
    assert ds.chunk_boundaries == [5, 5, 5, 5]
    got = pa.Table.from_batches(list(ds.iter_batches()))
    assert got.column("val").to_pylist() == t.column("val").to_pylist()


# ---------- iter_batches ----------

def test_iter_batches_default_yields_stored(tmp_path):
    t = make_table(20)
    ds = Dataset.from_batches(t.schema, batches_of(t, 5), tmp_path / "ds")
    sizes = [b.num_rows for b in ds.iter_batches()]
    assert sizes == [5, 5, 5, 5]


def test_iter_batches_rechunk_smaller(tmp_path):
    t = make_table(20)
    ds = Dataset.from_batches(t.schema, batches_of(t, 5), tmp_path / "ds")
    sizes = [b.num_rows for b in ds.iter_batches(batch_size=3)]
    # 20 rows → 3,3,3,3,3,3,2
    assert sizes == [3, 3, 3, 3, 3, 3, 2]
    total = sum(sizes)
    assert total == 20


def test_iter_batches_rechunk_larger(tmp_path):
    t = make_table(20)
    ds = Dataset.from_batches(t.schema, batches_of(t, 5), tmp_path / "ds")
    sizes = [b.num_rows for b in ds.iter_batches(batch_size=12)]
    assert sizes == [12, 8]


def test_iter_batches_columns_subset(tmp_path):
    t = make_table(20)
    ds = Dataset.from_batches(t.schema, batches_of(t, 5), tmp_path / "ds")
    for b in ds.iter_batches(columns=["val"]):
        assert b.schema.names == ["val"]


def test_iter_batches_missing_column_raises(tmp_path):
    t = make_table(10)
    ds = Dataset.from_batches(t.schema, batches_of(t, 5), tmp_path / "ds")
    with pytest.raises(KeyError):
        list(ds.iter_batches(columns=["nope"]))


# ---------- add_column ----------

def test_add_column_ok(tmp_path):
    t = make_table(20)
    ds = Dataset.from_batches(t.schema, batches_of(t, 5), tmp_path / "ds")
    new_vals = pa.table({"score": [i * 10 for i in range(20)]})
    ds.add_column("score", batches_of(new_vals, 5))
    assert "score" in ds.columns
    got = pa.Table.from_batches(list(ds.iter_batches()))
    assert got.column("score").to_pylist() == list(range(0, 200, 10))


def test_add_column_wrong_boundary_raises(tmp_path):
    t = make_table(20)
    ds = Dataset.from_batches(t.schema, batches_of(t, 5), tmp_path / "ds")
    bad = pa.table({"score": list(range(20))})
    with pytest.raises(ValueError, match="chunk boundaries"):
        ds.add_column("score", batches_of(bad, 7))  # 7,7,6 != 5,5,5,5


def test_add_column_too_few_batches_raises(tmp_path):
    t = make_table(20)
    ds = Dataset.from_batches(t.schema, batches_of(t, 5), tmp_path / "ds")
    short = pa.table({"score": list(range(10))})
    with pytest.raises(ValueError):
        ds.add_column("score", batches_of(short, 5))


# ---------- persistence ----------

def test_reload_from_disk(tmp_path):
    t = make_table(20)
    ds = Dataset.from_batches(t.schema, batches_of(t, 5), tmp_path / "ds")
    ds.add_column("score", batches_of(pa.table({"score": list(range(20))}), 5))

    ds2 = Dataset(tmp_path / "ds")
    assert set(ds2.columns) == {"cat", "val", "score"}
    assert ds2.total_rows == 20
    assert ds2.chunk_boundaries == [5, 5, 5, 5]
    got = pa.Table.from_batches(list(ds2.iter_batches()))
    assert got.column("score").to_pylist() == list(range(20))
