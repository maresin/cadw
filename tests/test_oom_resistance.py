"""
OOM resistance: 10^6 строк под малым бюджетом.

Проверяем, что operations либо успешно завершаются (за счёт shuffle/скью),
либо поднимают MemoryLimitExceeded ДО OOM. OOM = падение процесса; тут
проверяем, что процесс выживает.
"""
import numpy as np
import pyarrow as pa
import pytest

from cadw import MemoryGuard
from cadw.core.memory_guard import MemoryLimitExceeded
from cadw.ops.groupby import groupby_aggregate
from cadw.ops.reduce import CovarianceOp, reduce


def make_iter(table: pa.Table, batch_size: int):
    for i in range(0, table.num_rows, batch_size):
        yield table.slice(i, batch_size).to_batches()[0]


# ---------- groupby: 10^6 групп под 10 MB ----------

@pytest.mark.slow
def test_groupby_many_groups_10_6_under_10mb(tmp_path):
    """10^6 уникальных групп, 10 MB budget. Успех или MemoryLimitExceeded, не OOM."""
    n = 1_000_000
    table = pa.table({
        "k": [f"g{i:07d}" for i in range(n)],
        "v": [1.0] * n,
    })
    guard = MemoryGuard(max_memory_mb=10)
    try:
        result = groupby_aggregate(
            make_iter(table, 10_000), ["k"], ["v"], ["count"],
            guard, tmp_path, num_partitions=64, log=False,
        )
        # Если успех — 10^6 строк, count=1 каждая
        assert result.num_rows == n
        for r in result.to_pylist()[:10]:
            assert r["v_count"] == 1
    except MemoryLimitExceeded:
        # Допустимо: budget слишком мал даже для P=64. Главное — не OOM.
        pass


# ---------- groupby: skew 90/10 под 10 MB ----------

@pytest.mark.slow
def test_groupby_skew_10_6_under_10mb(tmp_path):
    """90/10 skew, 10^6 строк, 10 MB. Skew isolation должна справиться."""
    n = 1_000_000
    heavy = n * 9 // 10
    table = pa.table({
        "k": ["A"] * heavy + ["B"] * (n - heavy),
        "v": [1.0] * n,
    })
    guard = MemoryGuard(max_memory_mb=10)
    result = groupby_aggregate(
        make_iter(table, 10_000), ["k"], ["v"], ["count"],
        guard, tmp_path, log=False,
    )
    rows = {r["k"]: r["v_count"] for r in result.to_pylist()}
    assert rows == {"A": heavy, "B": n - heavy}


# ---------- covariance: d=5000 под 10 MB ----------

@pytest.mark.slow
def test_covariance_oom_resistance(tmp_path):
    """
    CovarianceOp(d=5000) → d²·8 = 200 MB state. Budget 10 MB →
    MemoryLimitExceeded ДО np.zeros((d,d)). Не OOM.

    До патча 10: np.zeros((5000, 5000)) = 200 MB аллоцируется в op.init()
    до первого check_state. На budget 10 MB — проверка срабатывала бы
    после аллокации (но аллокация 200 MB не OOM сама по себе на машине
    с 8-16 GB; для демонстрации — d=50000 = 20 GB, но 50000 столбцов
    в RecordBatch медленно собирать; используем d=5000 как proxy —
    raise должен произойти ДО init()).
    """
    d = 5000
    cols = [f"c{i}" for i in range(d)]
    # tiny batch, d columns
    batch = pa.RecordBatch.from_arrays(
        [pa.array([1.0]) for _ in cols], names=cols,
    )
    guard = MemoryGuard(max_memory_mb=10)
    op = CovarianceOp(cols)
    with pytest.raises(MemoryLimitExceeded):
        reduce(iter([batch]), op, guard, log=False)
