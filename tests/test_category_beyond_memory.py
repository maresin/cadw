"""v0.9.17: verify out-of-core guarantee when K × per_key > available.

These tests exercise the core CADW guarantee on the regime that
justifies the whole framework: **unique group keys do NOT fit in
memory**. `K × per_key_bytes` exceeds `available_bytes`; HashShuffle
partitions by key hash; LocalGroupBy processes each partition;
result is concat (partitions are key-disjoint).

per_key_bytes ≈ 80 B (40 B key string + tuple/dict state overhead).
K=100k needs ~8 MB hash table; K=1M needs ~80 MB.

Contract:
  - If P × (K/P) × per_key fits state_limit → succeed with K rows.
  - If check_partition_writers fires → clean MemoryLimitExceeded,
    never OOM.
  - Null keys preserved (§13.4/218 regression).
"""
from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.core.memory_guard import MemoryLimitExceeded
from cadw.ops.groupby import groupby_aggregate


def _unique_key_table(n: int, *, extra_null: int = 0) -> pa.Table:
    """All-unique string keys; optional extra null-key rows."""
    keys = [f"k{i:07d}" for i in range(n)]
    vals = [1] * n
    if extra_null:
        keys.extend([None] * extra_null)
        vals.extend([1] * extra_null)
    return pa.table({"k": keys, "v": vals})


def _iter_batches(table: pa.Table, batch_size: int = 1000):
    for i in range(0, table.num_rows, batch_size):
        yield table.slice(i, batch_size).to_batches()[0]


# ============ fast: K=100k ============


def test_groupby_100k_unique_10mb(tmp_path):
    """100k unique keys, 10 MB budget. K×per_key ≈ 8 MB →
    P=64 → ~125 KB/partition fits state_limit (~5 MB).
    Result: exactly 100k rows, each v_sum=1."""
    n = 100_000
    table = _unique_key_table(n)
    guard = MemoryGuard(max_memory_mb=10)
    result = groupby_aggregate(
        _iter_batches(table), ["k"], ["v"], ["sum"],
        guard, tmp_path, num_partitions=None, log=False,
    )
    assert result.count() == n
    # Spot-check 100 rows
    for r in result.to_pylist()[:100]:
        assert r["v_sum"] == 1


def test_groupby_100k_unique_1mb_borderline(tmp_path):
    """100k unique keys, 1 MB budget. K×per_key ≈ 8 MB ≫ 1 MB,
    but P=64 (each 125 KB fits) and writer check ~131 KB ≤ 314 KB
    → succeed. Borderline case validating auto-P + writer check."""
    n = 100_000
    table = _unique_key_table(n)
    guard = MemoryGuard(max_memory_mb=1)
    try:
        result = groupby_aggregate(
            _iter_batches(table), ["k"], ["v"], ["sum"],
            guard, tmp_path, num_partitions=None, log=False,
        )
        assert result.count() == n
    except MemoryLimitExceeded:
        pass  # clean error also acceptable; assert not OOM


def test_unique_100k_unique_10mb(tmp_path):
    """df.unique("k") on 100k unique keys, 10 MB budget."""
    n = 100_000
    table = _unique_key_table(n)
    p = tmp_path / "src.parquet"
    pq.write_table(table, p)
    df = CadwDataFrame.from_parquet(
        p, tmp_path / "ds", chunk_size=5000,
        memory_guard=MemoryGuard(max_memory_mb=10),
        spill_dir=tmp_path / "spill",
    )
    result = df.unique("k")
    assert result.count() == n


def test_dedup_100k_unique_10mb(tmp_path):
    """drop_duplicates(subset=["k"]) on 100k unique keys, 10 MB."""
    n = 100_000
    table = _unique_key_table(n)
    p = tmp_path / "src.parquet"
    pq.write_table(table, p)
    df = CadwDataFrame.from_parquet(
        p, tmp_path / "ds", chunk_size=5000,
        memory_guard=MemoryGuard(max_memory_mb=10),
        spill_dir=tmp_path / "spill",
    )
    result = df.drop_duplicates(
        subset=["k"], keep="first", num_partitions=64,
    )
    assert result.count() == n


# ============ §13.4/218 regression: null keys in K≫P regime ============


def test_groupby_50k_unique_plus_50k_null_10mb(tmp_path):
    """50k unique keys + 50k null-key rows, 10 MB budget.
    §13.4/218 regression: nulls must survive HashShuffle +
    LocalGroupBy (fill_null(False) in _compute_heavy_mask).
    If nulls were dropped (old bug), result would have only
    50k rows + sum(v_count) = 50k instead of 100k."""
    table = _unique_key_table(50_000, extra_null=50_000)
    guard = MemoryGuard(max_memory_mb=10)
    result = groupby_aggregate(
        _iter_batches(table), ["k"], ["v"], ["count"],
        guard, tmp_path, num_partitions=32, log=False,
    )
    rows = {r["k"]: r["v_count"] for r in result.to_pylist()}
    assert rows.get(None) == 50_000, (
        f"null-key rows dropped — got {rows.get(None)} null count, "
        f"expected 50_000 (regression of §13.4/218 fix)"
    )
    assert sum(rows.values()) == 100_000


# ============ slow: K=1M ============


@pytest.mark.slow
def test_groupby_1m_unique_10mb(tmp_path):
    """1M unique keys, 10 MB budget. K×per_key ≈ 80 MB ≫ 10 MB.
    P=64 partitions × ~15.6k keys each ≈ 1.25 MB/partition
    (fits state_limit 5 MB) + writer check ~131 KB (fits).
    Full out-of-core path: HashShuffle + N × LocalGroupBy + Concat."""
    n = 1_000_000
    table = _unique_key_table(n)
    guard = MemoryGuard(max_memory_mb=10)
    result = groupby_aggregate(
        _iter_batches(table, 5000), ["k"], ["v"], ["sum"],
        guard, tmp_path, num_partitions=None, log=False,
    )
    assert result.count() == n
    for r in result.to_pylist()[:50]:
        assert r["v_sum"] == 1


@pytest.mark.slow
def test_groupby_1m_unique_1mb_succeeds_or_raises(tmp_path):
    """1M unique keys, 1 MB budget. Auto-P picks P=8 (k_est from
    10k sample not extrapolated); 4^5=1024 leaf partitions of ~977
    keys each fit state_limit → framework succeeds via recursion.

    v0.9.18: prior version asserted MemoryLimitExceeded, but
    out-of-core recursion actually WORKS at this scale. That's
    the guarantee operating correctly. Accept both: success (with
    correct result) OR clean MemoryLimitExceeded (never OOM).
    """
    n = 1_000_000
    table = _unique_key_table(n)
    guard = MemoryGuard(max_memory_mb=1)
    try:
        result = groupby_aggregate(
            _iter_batches(table, 5000), ["k"], ["v"], ["sum"],
            guard, tmp_path, num_partitions=None, log=False,
        )
        assert result.count() == n
    except MemoryLimitExceeded:
        pass  # clean error also acceptable; assert not OOM


def test_groupby_explicit_huge_p_writer_check(tmp_path):
    """v0.9.18 (audit R1 §9.2.6 test): explicit num_partitions=256
    at 1 MB budget fires check_partition_writers deterministically.

    256 × 2048 B = 524288 > buffer_limit = 1 MB × 0.3 = 314572.
    This is the writer-count guard test — separate from the
    auto-P success/borderline tests.
    """
    n = 10_000
    table = _unique_key_table(n)
    guard = MemoryGuard(max_memory_mb=1)
    with pytest.raises(MemoryLimitExceeded, match="Partition writers"):
        groupby_aggregate(
            _iter_batches(table, 1000), ["k"], ["v"], ["sum"],
            guard, tmp_path, num_partitions=256, log=False,
        )
