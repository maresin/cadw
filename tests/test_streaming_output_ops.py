"""S2: output= parameter on 4 ops (v0.9.90-S2)."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.core.memory_guard import MemoryLimitExceeded
from cadw.core.spill import (
    AutoResult, InMemoryResult, OnDiskResult,
)
from cadw.ops.dedup import drop_duplicates as ops_dedup
from cadw.ops.groupby import groupby_aggregate
from cadw.ops.topk import topk_to_disk


def _batches(table, bs=100):
    for i in range(0, table.num_rows, bs):
        yield table.slice(i, bs).to_batches()[0]


def _small_guard():
    return MemoryGuard(max_memory_mb=64)


# ─── groupby_aggregate ─────────────────────────────────────────

def test_groupby_output_memory_default_returns_table(tmp_path):
    table = pa.table({"k": ["a", "b", "a", "b"], "v": [1, 2, 3, 4]})
    result = groupby_aggregate(
        _batches(table), ["k"], ["v"], ["sum"],
        _small_guard(), tmp_path, num_partitions=2, log=False)
    assert isinstance(result, pa.Table)
    assert result.num_rows == 2


def test_groupby_output_auto_small_inmemory(tmp_path):
    table = pa.table({"k": ["a", "b", "a", "b"], "v": [1, 2, 3, 4]})
    result = groupby_aggregate(
        _batches(table), ["k"], ["v"], ["sum"],
        _small_guard(), tmp_path, num_partitions=2, log=False,
        output="auto")
    assert isinstance(result, AutoResult)
    assert isinstance(result, InMemoryResult)
    assert len(result) == 2


def test_groupby_output_auto_equals_memory(tmp_path):
    rng = np.random.default_rng(0)
    n = 5000
    table = pa.table({
        "k": [f"k{i % 50}" for i in range(n)],
        "v": rng.integers(0, 100, n).tolist(),
    })
    m = groupby_aggregate(
        _batches(table, 200), ["k"], ["v"], ["sum"],
        _small_guard(), tmp_path / "m", num_partitions=8, log=False,
        output="memory")
    a = groupby_aggregate(
        _batches(table, 200), ["k"], ["v"], ["sum"],
        _small_guard(), tmp_path / "a", num_partitions=8, log=False,
        output="auto")
    rows_m = sorted(m.to_pylist(), key=lambda r: r["k"])
    rows_a = sorted(
        [r for b in a for r in b.to_pylist()], key=lambda r: r["k"])
    assert rows_m == rows_a


def test_groupby_output_disk_forces_spill(tmp_path):
    table = pa.table({"k": ["a", "b", "a", "b"], "v": [1, 2, 3, 4]})
    result = groupby_aggregate(
        _batches(table), ["k"], ["v"], ["sum"],
        _small_guard(), tmp_path, num_partitions=2, log=False,
        output="disk")
    assert isinstance(result, OnDiskResult)
    assert result.is_spilled
    with pytest.raises(MemoryLimitExceeded):
        result.to_table()
    result.close()


# ─── drop_duplicates ───────────────────────────────────────────

def test_dedup_output_auto(tmp_path):
    table = pa.table({
        "k": ["a", "a", "b", "b", "c"],
        "v": [1, 2, 3, 4, 5],
    })
    result = ops_dedup(
        _batches(table), ["k"], "first",
        _small_guard(), tmp_path, num_partitions=2, output="auto")
    assert isinstance(result, AutoResult)
    assert len(result) == 3


def test_dedup_output_disk_spills(tmp_path):
    table = pa.table({
        "k": ["a", "a", "b", "b", "c"],
        "v": [1, 2, 3, 4, 5],
    })
    result = ops_dedup(
        _batches(table), ["k"], "first",
        _small_guard(), tmp_path, num_partitions=2, output="disk")
    assert isinstance(result, OnDiskResult)
    result.close()


def test_dedup_output_memory_back_compat(tmp_path):
    table = pa.table({"k": ["a", "a", "b"], "v": [1, 2, 3]})
    result = ops_dedup(
        _batches(table), ["k"], "first",
        _small_guard(), tmp_path, num_partitions=2)
    assert isinstance(result, pa.Table)


# ─── topk_to_disk ──────────────────────────────────────────────

def test_topk_output_auto_small(tmp_path):
    vals = list(range(1000, 0, -1))
    table = pa.table({"v": vals})
    result = topk_to_disk(
        _batches(table, 50), column="v", n=5, kind="largest",
        memory_guard=_small_guard(), output="auto")
    assert isinstance(result, AutoResult)
    assert len(result) == 5
    got = [r["v"] for b in result for r in b.to_pylist()]
    assert got == [1000, 999, 998, 997, 996]


def test_topk_output_disk(tmp_path):
    table = pa.table({"v": list(range(100))})
    result = topk_to_disk(
        _batches(table, 10), column="v", n=3, kind="smallest",
        memory_guard=_small_guard(), output="disk")
    assert isinstance(result, OnDiskResult)
    got = [r["v"] for b in result for r in b.to_pylist()]
    assert got == [0, 1, 2]
    result.close()


def test_topk_incremental_merge_ooc_friendly(tmp_path):
    """v0.9.90-S2 fix: per-batch accumulation O(P*k) -> O(k).

    Old: 200 batches x k=10 -> 200 tables accumulated -> >1000 B.
    New: current_top <= k=10 rows, ~80 B. Passes 2 KB guard.
    """
    n_total = 10_000
    vals = list(range(n_total))
    table = pa.table({"v": vals})
    guard = MemoryGuard(max_memory_bytes=2000)
    result = topk_to_disk(
        _batches(table, 50), column="v", n=10, kind="largest",
        memory_guard=guard, output="memory")
    assert isinstance(result, pa.Table)
    assert result.num_rows == 10


def test_topk_output_invalid_n(tmp_path):
    table = pa.table({"v": [1, 2, 3]})
    with pytest.raises(ValueError, match="n must be int"):
        topk_to_disk(
            _batches(table), column="v", n=0, kind="largest",
            memory_guard=_small_guard(), output="auto")


# ─── output= validation ─────────────────────────────────────────

def test_output_invalid_raises(tmp_path):
    """v0.9.90-S2: output must be memory|auto|disk."""
    table = pa.table({"k": ["a"], "v": [1]})
    with pytest.raises(ValueError, match="output must be"):
        groupby_aggregate(
            _batches(table), ["k"], ["v"], ["sum"],
            _small_guard(), tmp_path, output="garbage")


# ─── pivot - regression: internal auto, output unchanged ──────

def test_pivot_internal_uses_auto():
    """AST: pivot_to_disk calls groupby_aggregate with output='auto'."""
    import inspect
    from cadw.ops import pivot as pv
    src = inspect.getsource(pv.pivot_to_disk)
    assert 'output="auto"' in src, (
        "pivot_to_disk must use output='auto' internally "
        "(v0.9.90-S2); otherwise large K_pairs OOMs")
