"""Тесты must-fix §11.4 п.1 (патч 02b1): рекурсия _local_groupby."""
from pathlib import Path

import pyarrow as pa
import pytest

from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.io.batch_writer import ArrowIPCBatchWriter
from cadw.ops import groupby as gb_mod
from cadw.ops.groupby import _local_groupby, groupby_aggregate


def make_iter(table: pa.Table, batch_size: int):
    for i in range(0, table.num_rows, batch_size):
        yield table.slice(i, batch_size).to_batches()[0]


def write_partition(path: Path, table: pa.Table, chunk: int = 100) -> Path:
    writer = ArrowIPCBatchWriter(str(path), table.schema)
    try:
        for b in table.to_batches(max_chunksize=chunk):
            writer.write_batch(b)
    finally:
        writer.close()
    return path


# ---------- 1. маленькая партиция: рекурсия не вызывается ----------

def test_local_groupby_no_recursion_small(monkeypatch, tmp_path):
    called = []
    orig = gb_mod.hash_shuffle
    def spy(*a, **kw):
        called.append(kw.get("hash_seed"))
        return orig(*a, **kw)
    monkeypatch.setattr(gb_mod, "hash_shuffle", spy)

    table = pa.table({"cat": ["A", "B", "A", "B"], "val": [1, 2, 3, 4]})
    p = write_partition(tmp_path / "small.arrow", table)
    guard = MemoryGuard(max_memory_mb=10)

    _local_groupby(p, ["cat"], ["val"], ["mean"], memory_guard=guard, spill_dir=tmp_path)
    assert not called, f"recursion should not be called; called={called}"


# ---------- 2. большая партиция: рекурсия вызывается ----------

def test_local_groupby_recursion_called(monkeypatch, tmp_path):
    called = []
    def spy(*a, **kw):
        called.append(kw.get("hash_seed"))
        return []
    monkeypatch.setattr(gb_mod, "hash_shuffle", spy)

    table = pa.table({"cat": ["A"] * 1000, "val": list(range(1000))})
    p = write_partition(tmp_path / "big.arrow", table, chunk=100)
    guard = MemoryGuard(max_memory_bytes=10_000)

    _local_groupby(p, ["cat"], ["val"], ["mean"], memory_guard=guard, spill_dir=tmp_path)
    assert called, "recursion expected but hash_shuffle not called"
    assert any(s is not None and s >= 1 for s in called), f"expected seed>=1; called={called}"


# ---------- 3. MAX_DEPTH -> MemoryLimitExceeded ----------

def test_local_groupby_max_depth_raises_direct(tmp_path):
    table = pa.table({"cat": ["A"] * 100, "val": list(range(100))})
    p = write_partition(tmp_path / "p.arrow", table, chunk=10)
    guard = MemoryGuard(max_memory_bytes=1)

    with pytest.raises(MemoryLimitExceeded, match="max_depth"):
        _local_groupby(
            p, ["cat"], ["val"], ["mean"],
            memory_guard=guard, spill_dir=tmp_path, depth=0, max_depth=0,
        )


# ---------- 4. end-to-end: K=5000, рекурсия возможна, результат корректен ----------

def test_groupby_aggregate_recursion_end_to_end(tmp_path):
    n = 5000
    table = pa.table({
        "cat": [f"g{i:05d}" for i in range(n)],
        "val": list(range(n)),
    })
    guard = MemoryGuard(max_memory_bytes=50_000)
    result = groupby_aggregate(
        make_iter(table, 500), ["cat"], ["val"], ["mean"],
        guard, tmp_path, num_partitions=4, log=False, output="memory",
    )
    assert result.num_rows == n
    rows = sorted(result.to_pylist(), key=lambda r: r["cat"])
    for i in [0, 1, n // 2, n - 1]:
        assert rows[i]["val_mean"] == float(i), f"row {i}: {rows[i]}"


# ---------- 5. sample > available*0.5 -> MemoryLimitExceeded ----------

def test_groupby_aggregate_sample_too_big_raises(tmp_path):
    n = 5000
    table = pa.table({
        "cat": [f"g{i:05d}" for i in range(n)],
        "val": list(range(n)),
    })
    guard = MemoryGuard(max_memory_bytes=1_000)
    with pytest.raises(MemoryLimitExceeded, match="sample"):
        groupby_aggregate(
            make_iter(table, 500), ["cat"], ["val"], ["mean"],
            guard, tmp_path, num_partitions=None, log=False, output="memory",
        )
