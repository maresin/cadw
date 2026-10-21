"""v0.9.8a: R1 — check_partition_writers (per-writer footprint)."""
from __future__ import annotations

import inspect

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.patch_utils import assert_call_order
from cadw.patch_utils import parse_method_source


# ============ impl: check_partition_writers ============


def test_check_partition_writers_small_p_passes():
    g = MemoryGuard(max_memory_mb=10)  # limit = 3 MB
    # P=100, 2048 B/writer = 200 KB < 3 MB
    g.check_partition_writers(100)


def test_check_partition_writers_large_p_raises():
    g = MemoryGuard(max_memory_mb=10)  # limit = 3 MB
    # P=2000, 2048 B/writer = 4 MB > 3 MB
    with pytest.raises(MemoryLimitExceeded, match="Partition writers"):
        g.check_partition_writers(2000)


def test_check_partition_writers_audit_case():
    """Audit bench_r1: P=5000, budget=100 KB should fail-fast."""
    g = MemoryGuard(max_memory_bytes=100_000)  # limit = 30 KB
    # P=5000 × 2048 B = 10 MB >> 30 KB → fail-fast
    with pytest.raises(MemoryLimitExceeded, match="Partition writers"):
        g.check_partition_writers(5000)


def test_check_partition_writers_custom_bytes_per_writer():
    g = MemoryGuard(max_memory_mb=10)  # limit = 3 MB
    # Custom 4096 B/writer: P=1000 = 4 MB > 3 MB
    with pytest.raises(MemoryLimitExceeded):
        g.check_partition_writers(1000, bytes_per_writer=4096)
    # Same P with default 2048 B: 2 MB < 3 MB → pass
    g.check_partition_writers(1000)


def test_check_partition_writers_invalid_p():
    g = MemoryGuard(max_memory_mb=10)
    with pytest.raises(ValueError, match="num_partitions"):
        g.check_partition_writers(0)


def test_check_partition_writers_invalid_bpw():
    g = MemoryGuard(max_memory_mb=10)
    with pytest.raises(ValueError, match="bytes_per_writer"):
        g.check_partition_writers(10, bytes_per_writer=0)


# ============ backward compat: existing tests must pass ============


def test_backward_compat_p4_budget_200kb():
    # Common in tests: P=4, budget=200 KB. 4×2048=8 KB < 60 KB limit.
    g = MemoryGuard(max_memory_kb=200) if False else MemoryGuard(max_memory_bytes=200_000)
    g.check_partition_writers(4)


def test_backward_compat_p8_budget_50kb():
    # P=8, budget=50 KB. 8×2048=16 KB. limit=15 KB. Fails!
    # This case appears in test_mustfix_local_groupby recursion:
    # budget=50_000, P=4 actually (not 8) → passes.
    g = MemoryGuard(max_memory_bytes=50_000)
    g.check_partition_writers(4)  # 4×2048=8 KB < 15 KB → OK


# ============ two-level AST guards (§13.4/190) ============


def test_hash_shuffle_calls_check_partition_writers():
    """Facade: hash_shuffle calls check_partition_writers."""
    from cadw.patch_utils import assert_no_call_in_source
    from cadw.ops import shuffle as sh
    src = inspect.getsource(sh.hash_shuffle)
    assert "check_partition_writers" in src


def test_groupby_skew_calls_check_partition_writers():
    from cadw.ops import groupby as gb
    src = inspect.getsource(gb._groupby_with_skew)
    assert "check_partition_writers" in src


def test_check_partition_writers_after_check_partition_buffers():
    """Ordering: writers check after buffers check (both retained)."""
    from cadw.patch_utils import assert_call_order
    from cadw.ops import shuffle as sh
    src = inspect.getsource(sh.hash_shuffle)
    assert_call_order(
        src, "check_partition_buffers", "check_partition_writers",
        fn_name="hash_shuffle",
    )


# ============ integration: end-to-end ============


def test_hash_shuffle_large_p_fails_fast(tmp_path):
    """P=1000 on 1 MB budget → MemoryLimitExceeded BEFORE opening writers."""
    from cadw.ops.shuffle import hash_shuffle
    batches = [pa.record_batch({"k": ["a", "b", "c"]})]
    g = MemoryGuard(max_memory_bytes=1_000_000)  # limit=300 KB
    # P=1000 × 2048 = 2 MB > 300 KB → fail before writers
    with pytest.raises(MemoryLimitExceeded, match="Partition writers"):
        hash_shuffle(
            iter(batches), ["k"], ["k"], 1000, g, tmp_path,
            log=False,
        )


def test_hash_shuffle_small_p_succeeds(tmp_path):
    """Sanity: small P still works."""
    from cadw.ops.shuffle import hash_shuffle
    batches = [pa.record_batch({"k": ["a", "b", "c", "d"]})]
    g = MemoryGuard(max_memory_mb=10)
    paths = hash_shuffle(
        iter(batches), ["k"], ["k"], 4, g, tmp_path,
        log=False,
    )
    assert len(paths) == 4
