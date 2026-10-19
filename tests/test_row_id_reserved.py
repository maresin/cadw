"""v0.8.5: `row_id` reserved guard (§13.4/126).

`row_id` is CADW internal identity column (appended by external_sort /
hash_shuffle). User column named `row_id` → duplicate → cryptic error
downstream. Guard: fail-fast ValueError in external_sort/hash_shuffle
before any batch read.
"""
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.ops.shuffle import hash_shuffle
from cadw.ops.sort import external_sort


def make_iter(table, bs=2):
    for i in range(0, table.num_rows, bs):
        yield table.slice(i, bs).to_batches()[0]


# ---------- external_sort ----------

def test_external_sort_rejects_row_id_key(tmp_path):
    table = pa.table({"row_id": [1, 2, 3], "v": [10, 20, 30]})
    with pytest.raises(ValueError, match="row_id.*reserved"):
        external_sort(
            make_iter(table),
            key_columns=["row_id"],
            value_columns=["v"],
            memory_guard=MemoryGuard(max_memory_mb=10),
            spill_dir=tmp_path / "s1",
            log=False,
        )


def test_external_sort_rejects_row_id_value(tmp_path):
    table = pa.table({"k": [1, 2, 3], "row_id": [10, 20, 30]})
    with pytest.raises(ValueError, match="row_id.*reserved"):
        external_sort(
            make_iter(table),
            key_columns=["k"],
            value_columns=["row_id"],
            memory_guard=MemoryGuard(max_memory_mb=10),
            spill_dir=tmp_path / "s2",
            log=False,
        )


def test_external_sort_without_row_id_works(tmp_path):
    table = pa.table({"k": [3, 1, 2], "v": [30, 10, 20]})
    p = external_sort(
        make_iter(table),
        key_columns=["k"],
        value_columns=["v"],
        memory_guard=MemoryGuard(max_memory_mb=10),
        spill_dir=tmp_path / "s3",
        log=False,
    )
    assert p.exists()


# ---------- hash_shuffle ----------

def test_hash_shuffle_rejects_row_id_key(tmp_path):
    table = pa.table({"row_id": [1, 2, 3], "v": [10, 20, 30]})
    with pytest.raises(ValueError, match="row_id.*reserved"):
        hash_shuffle(
            make_iter(table),
            key_columns=["row_id"],
            value_columns=["v"],
            num_partitions=2,
            memory_guard=MemoryGuard(max_memory_mb=10),
            spill_dir=tmp_path / "h1",
            log=False,
        )


def test_hash_shuffle_rejects_row_id_value(tmp_path):
    table = pa.table({"k": [1, 2, 3], "row_id": [10, 20, 30]})
    with pytest.raises(ValueError, match="row_id.*reserved"):
        hash_shuffle(
            make_iter(table),
            key_columns=["k"],
            value_columns=["row_id"],
            num_partitions=2,
            memory_guard=MemoryGuard(max_memory_mb=10),
            spill_dir=tmp_path / "h2",
            log=False,
        )


def test_hash_shuffle_without_row_id_works(tmp_path):
    table = pa.table({"k": [1, 2, 3, 1], "v": [10, 20, 30, 40]})
    paths = hash_shuffle(
        make_iter(table),
        key_columns=["k"],
        value_columns=["v"],
        num_partitions=2,
        memory_guard=MemoryGuard(max_memory_mb=10),
        spill_dir=tmp_path / "h3",
        log=False,
    )
    assert len(paths) == 2


# ---------- E2E (was cryptic KeyError) ----------

def test_sort_values_on_user_row_id_raises(tmp_path):
    t = pa.table({"row_id": [3, 1, 2], "v": [30, 10, 20]})
    p = tmp_path / "src.parquet"
    pq.write_table(t, p)
    df = CadwDataFrame.from_parquet(
        p, tmp_path / "ds", chunk_size=2,
        memory_guard=MemoryGuard(max_memory_mb=10),
        spill_dir=tmp_path / "spill",
    )
    with pytest.raises(ValueError, match="row_id.*reserved"):
        df.sort_values("row_id")


def test_groupby_on_user_row_id_raises(tmp_path):
    t = pa.table({"row_id": ["a", "b", "a"], "v": [1, 2, 3]})
    p = tmp_path / "src.parquet"
    pq.write_table(t, p)
    df = CadwDataFrame.from_parquet(
        p, tmp_path / "ds", chunk_size=2,
        memory_guard=MemoryGuard(max_memory_mb=10),
        spill_dir=tmp_path / "spill",
    )
    with pytest.raises(ValueError, match="row_id.*reserved"):
        df.groupby("row_id").agg("v", "sum")
