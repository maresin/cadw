from pathlib import Path

import pyarrow as pa
import pytest

from cadw.core.memory_guard import MemoryGuard
from cadw.ops.groupby import groupby_aggregate


def make_iter(table: pa.Table, batch_size: int):
    for i in range(0, table.num_rows, batch_size):
        yield table.slice(i, batch_size).to_batches()[0]


def make_guard(mb=10):
    return MemoryGuard(max_memory_mb=mb)


def result_to_dicts(t: pa.Table) -> list[dict]:
    return t.to_pylist()


# ---------------- базовый случай ----------------

def test_groupby_mean_simple(tmp_path):
    table = pa.table({
        "cat": ["A", "B", "A", "B", "A", "B"],
        "val": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })
    result = groupby_aggregate(
        make_iter(table, 2), ["cat"], ["val"], ["mean"],
        make_guard(), tmp_path, num_partitions=2, log=False, output="memory",
    )
    rows = sorted(result_to_dicts(result), key=lambda r: r["cat"])
    assert rows == [
        {"cat": "A", "val_mean": 3.0},
        {"cat": "B", "val_mean": 4.0},
    ]


def test_groupby_sum(tmp_path):
    table = pa.table({
        "cat": ["A", "B", "A", "B"],
        "val": [1, 2, 3, 4],
    })
    result = groupby_aggregate(
        make_iter(table, 2), "cat", "val", "sum",
        make_guard(), tmp_path, num_partitions=2, log=False, output="memory",
    )
    rows = sorted(result_to_dicts(result), key=lambda r: r["cat"])
    assert rows == [
        {"cat": "A", "val_sum": 4},
        {"cat": "B", "val_sum": 6},
    ]


def test_groupby_multiple_aggs(tmp_path):
    table = pa.table({
        "cat": ["A", "B", "A", "B"],
        "val": [1.0, 2.0, 3.0, 4.0],
    })
    result = groupby_aggregate(
        make_iter(table, 2), ["cat"], ["val", "val"], ["sum", "mean"],
        make_guard(), tmp_path, num_partitions=2, log=False, output="memory",
    )
    rows = sorted(result_to_dicts(result), key=lambda r: r["cat"])
    assert rows == [
        {"cat": "A", "val_sum": 4.0, "val_mean": 2.0},
        {"cat": "B", "val_sum": 6.0, "val_mean": 3.0},
    ]


# ---------------- K > памяти ----------------

def test_groupby_many_groups(tmp_path):
    """1000 уникальных групп при малом бюджете памяти.

    Старый SpillManager-подход сломался бы: число групп больше лимита
    текущего словаря. HashShuffle + LocalGroupBy корректен.
    """
    n = 1000
    table = pa.table({
        "cat": [f"g{i:04d}" for i in range(n)],
        "val": list(range(n)),
    })
    guard = MemoryGuard(max_memory_bytes=200_000)   # 200 KB
    result = groupby_aggregate(
        make_iter(table, 100), ["cat"], ["val"], ["mean"],
        guard, tmp_path, num_partitions=16, log=False, output="memory",
    )
    assert result.num_rows == n
    # Каждая группа = одна строка, mean = соответствующее значение
    rows = sorted(result_to_dicts(result), key=lambda r: r["cat"])
    for i, r in enumerate(rows):
        assert r["cat"] == f"g{i:04d}"
        assert r["val_mean"] == float(i)


def test_groupby_skewed(tmp_path):
    """Один ключ встречается в 90% строк — все равно корректно."""
    table = pa.table({
        "cat": ["A"] * 90 + ["B"] * 10,
        "val": [1.0] * 90 + [2.0] * 10,
    })
    result = groupby_aggregate(
        make_iter(table, 20), ["cat"], ["val", "val"], ["sum", "count"],
        make_guard(), tmp_path, num_partitions=8, log=False, output="memory",
    )
    rows = sorted(result_to_dicts(result), key=lambda r: r["cat"])
    assert rows[0]["val_sum"] == 90.0
    assert rows[0]["val_count"] == 90
    assert rows[1]["val_sum"] == 20.0
    assert rows[1]["val_count"] == 10


# ---------------- пограничные ----------------

def test_groupby_single_group(tmp_path):
    table = pa.table({"cat": ["A"] * 10, "val": list(range(10))})
    result = groupby_aggregate(
        make_iter(table, 3), ["cat"], ["val"], ["mean"],
        make_guard(), tmp_path, num_partitions=4, log=False, output="memory",
    )
    assert result.num_rows == 1
    assert result_to_dicts(result)[0]["val_mean"] == pytest.approx(4.5)


def test_groupby_each_row_unique(tmp_path):
    n = 50
    table = pa.table({
        "cat": [f"k{i}" for i in range(n)],
        "val": list(range(n)),
    })
    result = groupby_aggregate(
        make_iter(table, 7), ["cat"], ["val"], ["count"],
        make_guard(), tmp_path, num_partitions=8, log=False, output="memory",
    )
    assert result.num_rows == n
    for r in result_to_dicts(result):
        assert r["val_count"] == 1


def test_groupby_empty_input_raises(tmp_path):
    with pytest.raises(ValueError):
        groupby_aggregate(
            iter([]), ["cat"], ["val"], ["mean"],
            make_guard(), tmp_path, num_partitions=4, log=False, output="memory",
        )


# ---------------- корректность распределения ----------------

def test_groupby_partitions_do_not_overlap_keys(tmp_path):
    """Все строки одного ключа попадают в одну партицию (следствие HashShuffle)."""
    table = pa.table({
        "cat": ["A", "B", "C", "D"] * 5,
        "val": list(range(20)),
    })
    result = groupby_aggregate(
        make_iter(table, 3), ["cat"], ["val"], ["count"],
        make_guard(), tmp_path, num_partitions=4, log=False, output="memory",
    )
    rows = {r["cat"]: r["val_count"] for r in result_to_dicts(result)}
    assert rows == {"A": 5, "B": 5, "C": 5, "D": 5}


# ---------------- multi-key ----------------

def test_groupby_multiple_by_columns(tmp_path):
    table = pa.table({
        "c1": ["A", "A", "B", "B", "A", "B"],
        "c2": ["X", "Y", "X", "Y", "X", "Y"],
        "val": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })
    result = groupby_aggregate(
        make_iter(table, 2), ["c1", "c2"], ["val"], ["sum"],
        make_guard(), tmp_path, num_partitions=4, log=False, output="memory",
    )
    rows = sorted(result_to_dicts(result), key=lambda r: (r["c1"], r["c2"]))
    assert [(r["c1"], r["c2"], r["val_sum"]) for r in rows] == [
        ("A", "X", 6.0),
        ("A", "Y", 2.0),
        ("B", "X", 3.0),
        ("B", "Y", 10.0),
    ]

# ---------------- v0.9.49 BUG-1: multi-chunk value_counts -------------

def _make_multichunk_frame(tmp_path, n=100_000, ncat=20, chunk_size=50_000):
    import numpy as np
    import pyarrow.parquet as pq
    from cadw import CadwDataFrame

    rng = np.random.default_rng(0)
    tab = pa.table({
        **{f"c{i}": pa.array(rng.normal(size=n)) for i in range(10)},
        "cat": pa.array([f"g{j % ncat}" for j in rng.integers(0, ncat, size=n)]),
    })
    tmp_path.mkdir(parents=True, exist_ok=True)
    src = str(tmp_path / "src.parquet")
    pq.write_table(tab, src)
    return CadwDataFrame.from_parquet(
        src, str(tmp_path / "ds"), chunk_size=chunk_size,
        memory_guard=MemoryGuard(max_memory_mb=500),
        spill_dir=str(tmp_path / "spill"),
    )


def _vc_as_dict(t: pa.Table) -> dict:
    return {row[0]: row[1] for row in zip(t.column(0).to_pylist(), t.column("count").to_pylist())}


def test_value_counts_multichunk_matches_singlechunk(tmp_path):
    """v0.9.49 BUG-1: multi-chunk value_counts must equal single-chunk."""
    df_multi = _make_multichunk_frame(tmp_path, chunk_size=50_000)
    df_single = _make_multichunk_frame(tmp_path / "single", chunk_size=1_000_000)

    multi = _vc_as_dict(df_multi.value_counts("cat"))
    single = _vc_as_dict(df_single.value_counts("cat"))

    assert multi == single, f"multi-chunk {multi} != single-chunk {single}"
    assert len(multi) == 20
    assert sum(multi.values()) == 100_000


def test_value_counts_multichunk_matches_pandas(tmp_path):
    """M2: cadw multi-chunk value_counts == pandas value_counts."""
    import pandas as pd

    df_multi = _make_multichunk_frame(tmp_path, chunk_size=50_000)
    df_single = _make_multichunk_frame(tmp_path / "single", chunk_size=1_000_000)

    # Reference: pandas over the same source data.
    pdf = pd.read_parquet(str(tmp_path / "src.parquet"), columns=["cat"])
    ref = pdf["cat"].value_counts().to_dict()

    multi = _vc_as_dict(df_multi.value_counts("cat"))
    single = _vc_as_dict(df_single.value_counts("cat"))

    assert multi == ref, f"multi-chunk != pandas: {multi} vs {ref}"
    assert single == ref, f"single-chunk != pandas: {single} vs {ref}"
