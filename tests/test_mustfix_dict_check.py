"""
Must-fix §11.4: dict check в `_local_groupby` до `group_by`.

Проверяем:
  - AST: dict-check marker (`estimated_dict_bytes`/`dict_fits`/
    `_estimate_distinct_keys`) предшествует `group_by(`.
  - behavior: file fits, dict overflow -> split (hash_shuffle called);
  - behavior: dict fits -> no recursion;
  - error: max_depth исчерпан при dict overflow -> MemoryLimitExceeded.
"""
import inspect
from pathlib import Path

import pyarrow as pa
import pytest

from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.io.batch_writer import ArrowIPCBatchWriter
from cadw.ops import groupby as gb_mod
from cadw.ops.groupby import _local_groupby


def write_partition(path: Path, table: pa.Table, chunk: int = 200) -> Path:
    writer = ArrowIPCBatchWriter(str(path), table.schema)
    try:
        for b in table.to_batches(max_chunksize=chunk):
            writer.write_batch(b)
    finally:
        writer.close()
    return path


# ---------- AST guard ----------

def test_local_groupby_dict_check_before_group_by():
    """AST: dict check предшествует group_by в _local_groupby."""
    src = inspect.getsource(_local_groupby)
    gb_pos = src.find("group_by(")
    assert gb_pos >= 0, "group_by( not found in _local_groupby"
    markers = [
        src.find("_estimate_distinct_keys"),
        src.find("estimated_dict_bytes"),
        src.find("dict_fits"),
    ]
    valid = [p for p in markers if p >= 0]
    assert valid, (
        "no dict-check marker found in _local_groupby "
        "(must-fix §11.4: estimated_dict_bytes/dict_fits/_estimate_distinct_keys)"
    )
    earliest = min(valid)
    assert earliest < gb_pos, (
        f"dict check at pos {earliest} must precede group_by( at pos {gb_pos}. "
        f"Source: {src[:200]}..."
    )


# ---------- behavior: dict overflow triggers split ----------

def test_dict_overflow_triggers_split(tmp_path, monkeypatch):
    """File fits, dict overflow -> hash_shuffle called (recursion)."""
    called = []
    orig_hash_shuffle = gb_mod.hash_shuffle

    def spy(*args, **kwargs):
        called.append(kwargs.get("hash_seed"))
        return orig_hash_shuffle(*args, **kwargs)

    monkeypatch.setattr(gb_mod, "hash_shuffle", spy)

    # N=1000, K=1000 (unique int32 keys). row_size=8 B.
    n = 1000
    table = pa.table({
        "cat": pa.array(list(range(n)), type=pa.int32()),
        "val": pa.array(list(range(n)), type=pa.int32()),
    })
    p = write_partition(tmp_path / "p.arrow", table)

    # budget=100 KB -> safety=50 KB; file ~10 KB fits, dict=80 KB overflow
    guard = MemoryGuard(max_memory_bytes=100_000)
    result = _local_groupby(
        p, ["cat"], ["val"], ["count"],
        memory_guard=guard,
        spill_dir=tmp_path,
        size_of_key_entry_bytes=80,
    )

    assert called, "dict overflow must trigger hash_shuffle (recursion)"
    assert result.num_rows == n, (
        f"expected {n} groups, got {result.num_rows}"
    )


# ---------- behavior: dict fits, no recursion ----------

def test_dict_fits_no_recursion(tmp_path, monkeypatch):
    """K мал, dict fits -> hash_shuffle не вызывается."""
    called = []
    orig_hash_shuffle = gb_mod.hash_shuffle

    def spy(*args, **kwargs):
        called.append(True)
        return orig_hash_shuffle(*args, **kwargs)

    monkeypatch.setattr(gb_mod, "hash_shuffle", spy)

    # N=100, K=10 (10 unique keys). dict = 10*80 = 800 B
    n = 100
    keys = [i % 10 for i in range(n)]
    table = pa.table({
        "cat": pa.array(keys, type=pa.int32()),
        "val": pa.array(list(range(n)), type=pa.int32()),
    })
    p = write_partition(tmp_path / "p.arrow", table)

    guard = MemoryGuard(max_memory_bytes=100_000)  # 100 KB; dict_limit=50 KB
    result = _local_groupby(
        p, ["cat"], ["val"], ["count"],
        memory_guard=guard,
        spill_dir=tmp_path,
        size_of_key_entry_bytes=80,
    )

    assert not called, "dict fits -> no hash_shuffle expected"
    assert result.num_rows == 10


# ---------- error: max_depth exceeded ----------

def test_max_depth_raises_on_dict_overflow(tmp_path):
    """dict overflow + depth >= max_depth -> MemoryLimitExceeded."""
    n = 1000
    table = pa.table({
        "cat": pa.array(list(range(n)), type=pa.int32()),
        "val": pa.array(list(range(n)), type=pa.int32()),
    })
    p = write_partition(tmp_path / "p.arrow", table)

    guard = MemoryGuard(max_memory_bytes=100_000)
    with pytest.raises(MemoryLimitExceeded, match="max_depth"):
        _local_groupby(
            p, ["cat"], ["val"], ["count"],
            memory_guard=guard,
            spill_dir=tmp_path,
            depth=0,
            max_depth=0,
            size_of_key_entry_bytes=80,
        )


# ---------- invariants ----------

def test_dict_check_estimate_upper_bound(tmp_path):
    """_estimate_distinct_keys — верхняя оценка: <= num_rows, >= K_true."""
    from cadw.ops.groupby import _estimate_distinct_keys

    n = 200
    keys = [i % 50 for i in range(n)]  # K_true = 50
    table = pa.table({"cat": pa.array(keys, type=pa.int32())})
    est = _estimate_distinct_keys(table, ["cat"])
    assert est == 50, f"expected 50, got {est}"

    # multi-key
    table2 = pa.table({
        "a": pa.array([i % 5 for i in range(100)], type=pa.int32()),
        "b": pa.array([i % 7 for i in range(100)], type=pa.int32()),
    })
    est2 = _estimate_distinct_keys(table2, ["a", "b"])
    # product(5, 7) = 35; K_true <= 35
    assert est2 == 35, f"expected 35, got {est2}"
    assert est2 <= 100
