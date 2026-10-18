# tests/test_memory_guard_v2.py
import pyarrow as pa
import pytest

from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded


def make_batch(n: int = 10) -> pa.RecordBatch:
    return pa.RecordBatch.from_arrays(
        [pa.array(range(n), type=pa.int64())], names=["x"]
    )


# ---------- бюджет и расчёт размера батча ----------

def test_budget_conversion():
    g = MemoryGuard(max_memory_mb=10)
    assert g.budget_bytes == 10 * 1024 * 1024
    assert g.available_bytes() == g.budget_bytes


def test_get_batch_rows_uses_safety():
    g = MemoryGuard(max_memory_bytes=1_000_000, safety=0.5, min_batch_rows=1)
    g.estimate_row_size_from_batch(make_batch(10))
    # row_size для int64-колонки = 8 B (data) + overhead; проверим границы
    rows = g.get_batch_rows(row_size_bytes=10, expansion_factor=1.0)
    # usable = 500_000 B, raw = 50_000
    assert rows == 50_000


def test_expansion_factor_reduces_batch():
    g = MemoryGuard(max_memory_bytes=1_000_000, safety=1.0, min_batch_rows=1)
    rows_ef1 = g.get_batch_rows(row_size_bytes=10, expansion_factor=1.0)
    rows_ef2 = g.get_batch_rows(row_size_bytes=10, expansion_factor=2.0)
    assert rows_ef2 == rows_ef1 // 2


def test_min_batch_rows_respected_when_fits():
    g = MemoryGuard(max_memory_bytes=10_000_000, min_batch_rows=1000)
    rows = g.get_batch_rows(row_size_bytes=10, expansion_factor=1.0)
    assert rows >= 1000


def test_min_batch_rows_warns_and_lowers_when_budget_tiny():
    g = MemoryGuard(max_memory_bytes=200, safety=1.0, min_batch_rows=1000)
    rows = g.get_batch_rows(row_size_bytes=10, expansion_factor=1.0)
    # 200 / 10 = 20 < 1000 -> возвращаем raw
    assert rows == 20


def test_get_batch_rows_with_zero_usable_returns_1():
    g = MemoryGuard(max_memory_bytes=10, safety=1.0, min_batch_rows=1)
    # usable=10 B, row=100 B -> raw=0 -> 1
    rows = g.get_batch_rows(row_size_bytes=100, expansion_factor=1.0)
    assert rows == 1


# ---------- оценка размера строки ----------

def test_estimate_row_size_from_batch():
    g = MemoryGuard(max_memory_mb=10)
    batch = pa.RecordBatch.from_arrays(
        [pa.array(range(100), type=pa.int64())], names=["x"]
    )
    size = g.estimate_row_size_from_batch(batch)
    assert size >= 8          # как минимум 8 B на int64
    assert g.row_size_bytes == size


def test_estimate_row_size_from_schema_mixed():
    g = MemoryGuard(max_memory_mb=10)
    schema = pa.schema([
        ("a", pa.int64()),       # 8
        ("b", pa.float32()),     # 4
        ("c", pa.string()),      # 50
        ("d", pa.bool_()),       # 1
    ])
    assert g.estimate_row_size_from_schema(schema) == 63


def test_estimate_row_size_from_empty_batch_raises():
    g = MemoryGuard(max_memory_mb=10)
    empty = pa.RecordBatch.from_arrays([pa.array([], type=pa.int64())], names=["x"])
    with pytest.raises(ValueError):
        g.estimate_row_size_from_batch(empty)


# ---------- проверки (check_*) ----------

def test_check_batch_ok():
    g = MemoryGuard(max_memory_mb=10)
    g.check_batch(make_batch(100))   # не должно бросить


def test_check_batch_raises_on_excess():
    g = MemoryGuard(max_memory_bytes=100)
    with pytest.raises(MemoryLimitExceeded):
        g.check_batch(make_batch(100))   # int64*100 = 800 B > 100 B


def test_check_state_raises_on_excess():
    g = MemoryGuard(max_memory_bytes=1_000_000, state_safety=0.5)
    g.check_state(400_000)           # 400k < 500k
    with pytest.raises(MemoryLimitExceeded):
        g.check_state(600_000)       # 600k > 500k


def test_check_partition_buffers():
    g = MemoryGuard(max_memory_bytes=1_000_000, buffer_safety=0.3)
    g.check_partition_buffers(num_partitions=8, buffer_bytes=10_000)   # 80k < 300k
    with pytest.raises(MemoryLimitExceeded):
        g.check_partition_buffers(num_partitions=8, buffer_bytes=50_000)  # 400k > 300k


def test_recommended_partition_buffer_bytes():
    g = MemoryGuard(max_memory_bytes=1_000_000, buffer_safety=0.3)
    buf = g.recommended_partition_buffer_bytes(10)
    assert buf == 30_000


# ---------- уникальные ключи ----------

def test_estimate_unique_keys_extrapolates():
    g = MemoryGuard(max_memory_mb=10)
    sample = pa.RecordBatch.from_arrays(
        [pa.array([1, 2, 3, 1, 2], type=pa.int64())], names=["k"]
    )
    # 3 уникальных из 5, total=1000 -> K_est = 3 * 200 = 600
    assert g.estimate_unique_keys(sample, total_rows=1000) == 600


def test_estimate_unique_keys_empty():
    g = MemoryGuard(max_memory_mb=10)
    empty = pa.RecordBatch.from_arrays([pa.array([], type=pa.int64())], names=["k"])
    assert g.estimate_unique_keys(empty, total_rows=1000) == 0


# ---------- размер состояния ----------

@pytest.mark.parametrize(
    "op, expected",
    [
        ("count", 16),
        ("sum", 16),
        ("mean", 16),
        ("var", 24),
        ("std", 24),
    ],
)
def test_estimate_state_size_simple(op, expected):
    g = MemoryGuard(max_memory_mb=10)
    assert g.estimate_state_size(op, schema=None) == expected


def test_estimate_state_size_cov():
    g = MemoryGuard(max_memory_mb=10)
    # d=3 -> 8 + 24 + 72 = 104
    assert g.estimate_state_size("cov", d=3) == 104


def test_estimate_state_size_histogram():
    g = MemoryGuard(max_memory_mb=10)
    assert g.estimate_state_size("histogram", bins=255) == 8 * 255 + 16


def test_estimate_state_size_unknown_is_conservative():
    g = MemoryGuard(max_memory_mb=10)
    assert g.estimate_state_size("mystery_op") == 1024 * 1024


# ---------- резервирование ----------

def test_reserve_release():
    g = MemoryGuard(max_memory_bytes=1000)
    g.reserve(400)
    assert g.available_bytes() == 600
    g.release(200)
    assert g.available_bytes() == 800
    with pytest.raises(MemoryLimitExceeded):
        g.reserve(900)          # 200 + 900 = 1100 > 1000 — вот здесь ожидаемо