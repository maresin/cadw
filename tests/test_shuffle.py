import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ops.shuffle import (
    compute_hashes,
    estimate_num_partitions,
    hash_shuffle,
    read_partition_batches,
)


def make_data() -> pa.Table:
    return pa.table({
        "cat": ["A", "B", "C", "D", "A", "B", "C", "D"],
        "val": [10, 20, 30, 40, 11, 21, 31, 41],
    })


def make_iter(table: pa.Table, batch_size: int):
    for i in range(0, table.num_rows, batch_size):
        yield table.slice(i, batch_size).to_batches()[0]


def collect_partitions(paths):
    """Возвращает list[list[dict]] — содержимое каждой партиции."""
    out = []
    for p in paths:
        rows = []
        for b in read_partition_batches(p):
            for i in range(b.num_rows):
                rows.append({name: b.column(name)[i].as_py() for name in b.schema.names})
        out.append(rows)
    return out


def make_guard(mb=10):
    return MemoryGuard(max_memory_mb=mb)


# ---------------- главный инвариант ----------------

def test_partitions_disjoint_by_key(tmp_path):
    """Все строки каждого ключа попадают в ровно одну партицию.

    Разные ключи могут оказаться в одной партиции (хэш-коллизия) —
    это норма, downstream-агрегация их различает по значению ключа.
    """
    table = make_data()
    paths = hash_shuffle(
        make_iter(table, 2), ["cat"], ["val"], 4, make_guard(), tmp_path, log=False,
    )
    assert len(paths) == 4

    partitions = collect_partitions(paths)

    # Для каждого ключа: множество партиций, где он встречается, должно быть {p}
    key_to_partitions: dict[str, set[int]] = {}
    for p_idx, rows in enumerate(partitions):
        for r in rows:
            key_to_partitions.setdefault(r["cat"], set()).add(p_idx)

    for key, parts in key_to_partitions.items():
        assert len(parts) == 1, f"key {key!r} split across partitions {parts}"

    # Все ключи присутствуют
    assert set(key_to_partitions.keys()) == {"A", "B", "C", "D"}


def test_row_id_preserved(tmp_path):
    table = make_data()
    paths = hash_shuffle(
        make_iter(table, 2), ["cat"], ["val"], 4, make_guard(), tmp_path, log=False,
    )
    partitions = collect_partitions(paths)
    all_row_ids = [r["row_id"] for rows in partitions for r in rows]
    assert sorted(all_row_ids) == list(range(8))


def test_concat_equals_original(tmp_path):
    """Конкатенация всех партиций (по row_id) даёт исходные строки."""
    table = make_data()
    paths = hash_shuffle(
        make_iter(table, 3), ["cat"], ["val"], 4, make_guard(), tmp_path, log=False,
    )
    rows = [r for part in collect_partitions(paths) for r in part]
    rows.sort(key=lambda r: r["row_id"])

    expected = [
        {"cat": c, "val": v}
        for c, v in zip(table.column("cat").to_pylist(), table.column("val").to_pylist())
    ]
    got = [{"cat": r["cat"], "val": r["val"]} for r in rows]
    assert got == expected


# ---------------- пограничные ----------------

def test_p_gt_n(tmp_path):
    table = make_data()  # 8 строк, 4 ключа
    paths = hash_shuffle(
        make_iter(table, 8), ["cat"], ["val"], 100, make_guard(), tmp_path, log=False,
    )
    assert len(paths) == 100
    partitions = collect_partitions(paths)
    non_empty = [p for p in partitions if p]
    assert len(non_empty) == 4          # ровно 4 ключа
    assert sum(len(p) for p in partitions) == 8


def test_p_equals_1(tmp_path):
    table = make_data()
    paths = hash_shuffle(
        make_iter(table, 8), ["cat"], ["val"], 1, make_guard(), tmp_path, log=False,
    )
    assert len(paths) == 1
    rows = collect_partitions(paths)[0]
    assert len(rows) == 8


def test_empty_input_raises(tmp_path):
    with pytest.raises(ValueError):
        hash_shuffle(
            iter([]), ["cat"], ["val"], 4, make_guard(), tmp_path, log=False,
        )


def test_stable_hash_across_batches(tmp_path):
    """Одинаковые ключи из разных батчей → одна партиция."""
    # много A в разных батчах + немного B
    table = pa.table({
        "cat": ["A"] * 10 + ["B"] * 5,
        "val": list(range(15)),
    })
    paths = hash_shuffle(
        make_iter(table, 3), ["cat"], ["val"], 4, make_guard(), tmp_path, log=False,
    )
    partitions = collect_partitions(paths)
    a_parts = [i for i, rows in enumerate(partitions) if any(r["cat"] == "A" for r in rows)]
    b_parts = [i for i, rows in enumerate(partitions) if any(r["cat"] == "B" for r in rows)]
    assert len(a_parts) == 1, f"A split across {a_parts}"
    assert len(b_parts) == 1, f"B split across {b_parts}"
    assert a_parts != b_parts


# ---------------- детерминированность хэша ----------------

def test_compute_hashes_deterministic():
    batch = pa.RecordBatch.from_arrays(
        [pa.array(["A", "B", "A"])], names=["cat"],
    )
    h1 = compute_hashes(batch, ["cat"], seed=0).to_pylist()
    h2 = compute_hashes(batch, ["cat"], seed=0).to_pylist()
    assert h1 == h2
    # A,A → одинаковый хэш
    assert h1[0] == h1[2]
    # A,B → разные хэши (с огромной вероятностью)
    assert h1[0] != h1[1]


def test_compute_hashes_seed_changes_hash():
    batch = pa.RecordBatch.from_arrays(
        [pa.array(["A"])], names=["cat"],
    )
    h1 = compute_hashes(batch, ["cat"], seed=0).to_pylist()[0]
    h2 = compute_hashes(batch, ["cat"], seed=1).to_pylist()[0]
    assert h1 != h2


# ---------------- estimate_num_partitions ----------------

def test_estimate_num_partitions_basic():
    # k_est=100, available=10MB, bytes_per_key=80 → groups_per_partition≈78643
    p = estimate_num_partitions(100, available_bytes=10 * 1024 * 1024)
    assert p >= 1
    assert p <= 100  # sanity

def test_estimate_num_partitions_tiny_budget():
    # маленький бюджет → много партиций
    p_small = estimate_num_partitions(100_000, available_bytes=10_000)
    p_big = estimate_num_partitions(100_000, available_bytes=100_000_000)
    assert p_small > p_big
    assert p_small >= 1


def test_estimate_num_partitions_zero_keys():
    assert estimate_num_partitions(0, available_bytes=1_000_000) == 1


# ---------------- check_partition_buffers ----------------

def test_hash_shuffle_respects_partition_buffer_limit(tmp_path):
    # P=1000, buffer_safety=0.3, budget=1KB → 1000 * min(rec, 64KB) > 300B
    guard = MemoryGuard(max_memory_bytes=1000)
    table = make_data()
    with pytest.raises(MemoryLimitExceeded):
        hash_shuffle(
            make_iter(table, 8), ["cat"], ["val"], 1000, guard, tmp_path, log=False,
        )