"""
drop_duplicates (v0.3.0): HashShuffle + local pandas.drop_duplicates.

Проверяем:
  - keep ∈ {first, last, False};
  - subset=list[str] и subset=None;
  - multi-column subset;
  - pandas-equivalence (по содержимому, порядок — partition-concat);
  - edge cases: empty, missing column, invalid keep, subset=[];
  - после filter;
  - skew → MemoryLimitExceeded (без recursion).
"""
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.core.memory_guard import MemoryLimitExceeded
from cadw.ops.dedup import drop_duplicates as ops_drop_duplicates


def make_df(tmp_path, name, data, chunk=3, mem_mb=20):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def _norm(rows: list[dict], key: str) -> list[dict]:
    return sorted(rows, key=lambda r: (r[key] is None, r[key]))


# ---------- basic semantics ----------

def test_drop_duplicates_first_basic(tmp_path):
    """keep='first': оставить первое (по row_id) вхождение каждого ключа."""
    df = make_df(tmp_path, "L", {
        "k": ["a", "a", "b", "b", "c"],
        "v": [1, 2, 3, 4, 5],
    })
    result = df.drop_duplicates(subset=["k"], keep="first", num_partitions=4)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    # Один ряд на ключ; v из первой встречи (по исходному порядку)
    by_k = {r["k"]: r["v"] for r in rows}
    assert by_k == {"a": 1, "b": 3, "c": 5}


def test_drop_duplicates_last_basic(tmp_path):
    df = make_df(tmp_path, "L", {
        "k": ["a", "a", "b", "b", "c"],
        "v": [1, 2, 3, 4, 5],
    })
    result = df.drop_duplicates(subset=["k"], keep="last", num_partitions=4)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    by_k = {r["k"]: r["v"] for r in rows}
    assert by_k == {"a": 2, "b": 4, "c": 5}


def test_drop_duplicates_false_keeps_only_unique_keys(tmp_path):
    """keep=False: удалить ВСЕ строки с дублирующимися ключами."""
    df = make_df(tmp_path, "L", {
        "k": ["a", "a", "b", "b", "c"],
        "v": [1, 2, 3, 4, 5],
    })
    result = df.drop_duplicates(subset=["k"], keep=False, num_partitions=4)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert len(rows) == 1
    assert rows[0] == {"k": "c", "v": 5}


# ---------- subset=None (all columns) ----------

def test_drop_duplicates_full_row(tmp_path):
    """subset=None: dedup по всем колонкам."""
    df = make_df(tmp_path, "L", {
        "k": ["a", "a", "b"],
        "v": [1, 1, 2],
    })
    result = df.drop_duplicates(keep="first", num_partitions=4)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = _norm(pq.read_table(out).to_pylist(), "k")
    assert rows == [
        {"k": "a", "v": 1},
        {"k": "b", "v": 2},
    ]


def test_drop_duplicates_full_row_keeps_distinct_values(tmp_path):
    """subset=None: (a,1) и (a,2) — разные."""
    df = make_df(tmp_path, "L", {
        "k": ["a", "a", "b"],
        "v": [1, 2, 3],
    })
    result = df.drop_duplicates(keep="first", num_partitions=4)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = _norm(pq.read_table(out).to_pylist(), "k")
    # both a,1 and a,2 survive (distinct full tuples)
    assert len(rows) == 3


# ---------- multi-column subset ----------

def test_drop_duplicates_multi_key(tmp_path):
    df = make_df(tmp_path, "L", {
        "k1": ["a", "a", "a", "b"],
        "k2": [1, 1, 2, 1],
        "v": [10, 20, 30, 40],
    })
    result = df.drop_duplicates(subset=["k1", "k2"], keep="first", num_partitions=4)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = _norm(pq.read_table(out).to_pylist(), "v")
    # (a,1): 10 (first), (a,2): 30, (b,1): 40
    assert {(r["k1"], r["k2"], r["v"]) for r in rows} == {
        ("a", 1, 10),
        ("a", 2, 30),
        ("b", 1, 40),
    }


# ---------- all same key ----------

def test_drop_duplicates_all_same_first(tmp_path):
    df = make_df(tmp_path, "L", {"k": ["X"] * 10, "v": list(range(10))})
    result = df.drop_duplicates(subset=["k"], keep="first", num_partitions=4)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert len(rows) == 1
    assert rows[0]["v"] == 0  # first occurrence


def test_drop_duplicates_all_same_last(tmp_path):
    df = make_df(tmp_path, "L", {"k": ["X"] * 10, "v": list(range(10))})
    result = df.drop_duplicates(subset=["k"], keep="last", num_partitions=4)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert len(rows) == 1
    assert rows[0]["v"] == 9  # last occurrence


def test_drop_duplicates_all_same_false(tmp_path):
    df = make_df(tmp_path, "L", {"k": ["X"] * 10, "v": list(range(10))})
    result = df.drop_duplicates(subset=["k"], keep=False, num_partitions=4)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert len(rows) == 0


# ---------- all unique ----------

def test_drop_duplicates_all_unique(tmp_path):
    n = 50
    df = make_df(tmp_path, "L", {"k": [f"k{i}" for i in range(n)], "v": list(range(n))})
    result = df.drop_duplicates(subset=["k"], keep="first", num_partitions=8)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    assert pq.read_table(out).num_rows == n


# ---------- pandas-equivalence ----------

def test_drop_duplicates_pandas_equivalence(tmp_path):
    """Содержимое (набор строк) совпадает с pandas для keep='first'."""
    rng = pd.np.random.default_rng(42) if hasattr(pd, "np") else None
    # use plain numpy
    import numpy as np
    rng = np.random.default_rng(42)
    n = 200
    data = {
        "k": [f"k{i % 20}" for i in range(n)],
        "v": rng.integers(0, 100, size=n).tolist(),
    }
    df = make_df(tmp_path, "L", data, chunk=20)
    result = df.drop_duplicates(subset=["k"], keep="first", num_partitions=8)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    got = pq.read_table(out).to_pandas()

    pdf = pd.DataFrame(data)
    expected = pdf.drop_duplicates(subset=["k"], keep="first")

    # Сравнение по set-of-tuples (порядок не гарантирован, §11.5)
    got_set = set(map(tuple, got[["k", "v"]].itertuples(index=False, name=None)))
    exp_set = set(map(tuple, expected[["k", "v"]].itertuples(index=False, name=None)))
    assert got_set == exp_set, f"diff: {got_set ^ exp_set}"


# ---------- after filter ----------

def test_drop_duplicates_after_filter(tmp_path):
    df = make_df(tmp_path, "L", {
        "k": ["a", "a", "b", "b", "c"],
        "v": [1, 2, 3, 4, 5],
    })
    filtered = df[df["v"] > 1]  # rows: (a,2),(b,3),(b,4),(c,5)
    result = filtered.drop_duplicates(subset=["k"], keep="first", num_partitions=4)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = _norm(pq.read_table(out).to_pylist(), "k")
    by_k = {r["k"]: r["v"] for r in rows}
    assert by_k == {"a": 2, "b": 3, "c": 5}


# ---------- edge cases ----------

def test_drop_duplicates_missing_subset_column_raises(tmp_path):
    df = make_df(tmp_path, "L", {"k": ["a"], "v": [1]})
    with pytest.raises(KeyError, match="nope"):
        df.drop_duplicates(subset=["nope"])


def test_drop_duplicates_invalid_keep_raises(tmp_path):
    df = make_df(tmp_path, "L", {"k": ["a"], "v": [1]})
    with pytest.raises(ValueError, match="keep"):
        df.drop_duplicates(subset=["k"], keep="middle")


def test_drop_duplicates_empty_subset_raises(tmp_path):
    df = make_df(tmp_path, "L", {"k": ["a"], "v": [1]})
    with pytest.raises(ValueError, match="non-empty"):
        df.drop_duplicates(subset=[])


def test_drop_duplicates_empty_input_raises(tmp_path):
    empty = pa.table({
        "k": pa.array([], type=pa.string()),
        "v": pa.array([], type=pa.int64()),
    })
    empty_batches: list[pa.RecordBatch] = []
    mg = MemoryGuard(max_memory_mb=10)
    with pytest.raises(ValueError, match="empty input"):
        ops_drop_duplicates(
            iter(empty_batches),
            subset=["k"],
            keep="first",
            memory_guard=mg,
            spill_dir=tmp_path / "spill",
            output="memory",
        )


def test_drop_duplicates_p_gt_n(tmp_path):
    """P > rows: small data, many partitions."""
    df = make_df(tmp_path, "L", {"k": ["a", "b", "c"], "v": [1, 2, 3]})
    result = df.drop_duplicates(subset=["k"], keep="first", num_partitions=32)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    assert pq.read_table(out).num_rows == 3


# ---------- skew: oversized partition → MemoryLimitExceeded ----------

def test_drop_duplicates_skew_raises(tmp_path):
    """1000 строк с одним ключом при малом бюджете — raise, не OOM."""
    df = make_df(
        tmp_path, "L",
        {"k": ["HOT"] * 1000, "v": list(range(1000))},
        chunk=200,
        mem_mb=1,  # 1 MB budget
    )
    # Партиция с 1000 строк (row_size ~ string + int) → партиция ~20 KB
    # при P=4 = ~5 KB. При num_partitions=1 — одна партиция ~20 KB.
    # С budget 1 MB и safety 0.5 → limit = 500 KB. Партиция влезает.
    # Хотим oversized: увеличим num_partitions... нет, наоборот уменьшим.
    # Используем num_partitions=1 и уменьшим budget.
    df._memory_guard = MemoryGuard(max_memory_bytes=1000)  # 1 KB

    with pytest.raises(MemoryLimitExceeded):
        df.drop_duplicates(subset=["k"], keep="first", num_partitions=1)


# ---------- chained with sort ----------

def test_drop_duplicates_then_sort(tmp_path):
    df = make_df(tmp_path, "L", {
        "k": ["c", "a", "b", "a", "c"],
        "v": [1, 2, 3, 4, 5],
    })
    result = df.drop_duplicates(subset=["k"], keep="last", num_partitions=4)
    sorted_result = result.sort_values("k", ascending=True)
    out = tmp_path / "o.parquet"
    sorted_result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    # keep='last': a→4, b→3, c→5. sorted by k: a,b,c
    assert rows == [
        {"k": "a", "v": 4},
        {"k": "b", "v": 3},
        {"k": "c", "v": 5},
    ]
