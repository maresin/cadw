"""Тесты интеграции JoinMemoryModel в merge (11-B-2c-3)."""
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.core.memory_guard import MemoryLimitExceeded


def make_df(tmp_path, name, data, chunk=3, mem_mb=20):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


# ---------- planner используется по умолчанию ----------

def test_merge_uses_planner_num_partitions(tmp_path):
    """num_partitions=None + use_planner=True → модель выбирает P."""
    left = make_df(tmp_path, "L", {"k": [1, 2, 3, 4], "a": ["x", "y", "z", "w"]})
    right = make_df(tmp_path, "R", {"k": [1, 3, 5], "b": ["p", "q", "r"]})
    # Не падаем, merge возвращает результат (plan выбрал P)
    result = left.merge(right, on="k")  # use_planner=True default
    assert result.shape[0] == 2


def test_merge_use_planner_false(tmp_path):
    """use_planner=False → дефолт num_partitions=8, работает."""
    left = make_df(tmp_path, "L", {"k": [1, 2, 3], "a": [10, 20, 30]})
    right = make_df(tmp_path, "R", {"k": [1, 3], "b": [100, 300]})
    result = left.merge(right, on="k", use_planner=False)
    assert result.shape[0] == 2


def test_merge_explicit_num_partitions_overrides_planner(tmp_path):
    """num_partitions явно → planner всё равно считает limits."""
    left = make_df(tmp_path, "L", {"k": [1, 2, 3], "a": [10, 20, 30]})
    right = make_df(tmp_path, "R", {"k": [1, 3], "b": [100, 300]})
    result = left.merge(right, on="k", num_partitions=2)
    assert result.shape[0] == 2


# ---------- early failure когда план не сходится ----------

def test_merge_raises_when_plan_fails(tmp_path):
    """Огромные стороны под 1 MB → MemoryLimitExceeded от mg.plan."""
    left = make_df(
        tmp_path, "L",
        {"k": list(range(10_000)), "a": list(range(10_000))},
        chunk=1000, mem_mb=20,
    )
    right = make_df(
        tmp_path, "R",
        {"k": list(range(10_000)), "b": list(range(10_000))},
        chunk=1000, mem_mb=20,
    )
    # merge использует mg из left (20 MB); для join двух 10k-row сторон
    # с row_size ~16 B это должно влезть... Уменьшим бюджет до 100 KB.
    left._memory_guard = MemoryGuard(max_memory_bytes=100_000)
    with pytest.raises(MemoryLimitExceeded):
        left.merge(right, on="k")


# ---------- planner + skew: runtime защищает ----------

def test_merge_cartesian_skew_raises(tmp_path):
    """Один тяжёлый ключ: 1000×1000 cartesian → MemoryLimitExceeded."""
    # Left: 1000 строк с одним ключом; right: 1000 строк с тем же ключом
    left = make_df(
        tmp_path, "L",
        {"k": ["HOT"] * 1000, "a": list(range(1000))},
        chunk=200, mem_mb=20,
    )
    right = make_df(
        tmp_path, "R",
        {"k": ["HOT"] * 1000, "b": list(range(1000))},
        chunk=200, mem_mb=20,
    )
    # Сообщение: "exact matches between this probe batch ... Likely skew."
    # (с v0.2.5 cartesian check заменён на exact-match; §13.4/47).
    with pytest.raises(MemoryLimitExceeded, match="exact matches|Likely skew"):
        left.merge(right, on="k")


# ---------- planner + left join + PK-FK ----------

def test_merge_left_planner(tmp_path):
    """Left join через planner: PK-FK, ожидаемый output = len(left)."""
    left = make_df(
        tmp_path, "L",
        {"k": [1, 2, 3, 4], "a": ["w", "x", "y", "z"]},
    )
    right = make_df(tmp_path, "R", {"k": [1, 3, 5], "b": ["p", "q", "r"]})
    result = left.merge(right, on="k", how="left")
    assert result.shape[0] == 4


# ---------- empty side ----------

def test_merge_empty_right_with_planner(tmp_path):
    """Пустой right + planner → inner даёт пусто."""
    left = make_df(tmp_path, "L", {"k": [1, 2], "a": [10, 20]})
    right_data = {
        "k": pa.array([], type=pa.int64()),
        "b": pa.array([], type=pa.string()),
    }
    right = make_df(tmp_path, "R", right_data)
    result = left.merge(right, on="k")
    assert result.shape[0] == 0


def test_merge_empty_right_left_with_planner(tmp_path):
    """Пустой right + planner + how='left' → left preserved."""
    left = make_df(tmp_path, "L", {"k": [1, 2], "a": [10, 20]})
    right_data = {
        "k": pa.array([], type=pa.int64()),
        "b": pa.array([], type=pa.string()),
    }
    right = make_df(tmp_path, "R", right_data)
    result = left.merge(right, on="k", how="left")
    assert result.shape[0] == 2
