import tempfile
from pathlib import Path

import pyarrow as pa
import pytest

from cadw.core.memory_guard import MemoryGuard
from cadw.ops.sort import external_sort, read_sorted


def make_iter(data: dict, batch_size: int, value_columns, key_columns):
    n = len(next(iter(data.values())))
    table = pa.table({c: data[c] for c in list(value_columns) + list(key_columns)})
    for i in range(0, n, batch_size):
        yield table.slice(i, batch_size).to_batches()[0]


def collect(path: Path, value_columns) -> list[dict]:
    rows = []
    for b in read_sorted(path):
        for i in range(b.num_rows):
            rows.append({c: b.column(c)[i].as_py() for c in list(value_columns) + ["row_id"]})
    return rows


def make_guard(mb=10):
    return MemoryGuard(max_memory_mb=mb)


# ---------- базовый случай ----------

def test_sort_ascending(tmp_path):
    data = {"val": [40, 10, 30, 20, 50, 5, 15, 25], "id": list(range(8))}
    it = make_iter(data, 2, value_columns=["val"], key_columns=["id"])
    path = external_sort(
        it, ["val"], ["val"], make_guard(), tmp_path,
        ascending=True, nulls_first=False, log=False,
    )
    rows = collect(path, ["val"])
    vals = [r["val"] for r in rows]
    assert vals == [5, 10, 15, 20, 25, 30, 40, 50]
    assert [r["row_id"] for r in rows] == [5, 1, 6, 3, 7, 2, 0, 4]


def test_sort_descending(tmp_path):
    data = {"val": [40, 10, 30, 20], "id": list(range(4))}
    it = make_iter(data, 2, value_columns=["val"], key_columns=["id"])
    path = external_sort(
        it, ["val"], ["val"], make_guard(), tmp_path,
        ascending=False, nulls_first=False, log=False,
    )
    rows = collect(path, ["val"])
    assert [r["val"] for r in rows] == [40, 30, 20, 10]


# ---------- стабильность ----------

def test_sort_stable_by_row_id(tmp_path):
    # все ключи равны -> порядок должен остаться по row_id
    data = {"val": [1, 1, 1, 1, 1, 1], "id": list(range(6))}
    it = make_iter(data, 2, value_columns=["val"], key_columns=["id"])
    path = external_sort(
        it, ["val"], ["val"], make_guard(), tmp_path,
        ascending=True, nulls_first=False, log=False,
    )
    rows = collect(path, ["val"])
    assert [r["row_id"] for r in rows] == [0, 1, 2, 3, 4, 5]


def test_sort_multiple_keys(tmp_path):
    data = {
        "a": [1, 2, 1, 2, 1, 2],
        "b": [10, 20, 5, 15, 20, 10],
        "id": list(range(6)),
    }
    it = make_iter(data, 2, value_columns=["a", "b"], key_columns=["id"])
    path = external_sort(
        it, ["a", "b"], ["a", "b"], make_guard(), tmp_path,
        ascending=True, nulls_first=False, log=False,
    )
    rows = collect(path, ["a", "b"])
    expected = [(1, 5), (1, 10), (1, 20), (2, 10), (2, 15), (2, 20)]
    assert [(r["a"], r["b"]) for r in rows] == expected


# ---------- null ----------

def test_sort_nulls_last(tmp_path):
    data = {"val": [3, None, 1, None, 2], "id": list(range(5))}
    it = make_iter(data, 2, value_columns=["val"], key_columns=["id"])
    path = external_sort(
        it, ["val"], ["val"], make_guard(), tmp_path,
        ascending=True, nulls_first=False, log=False,
    )
    rows = collect(path, ["val"])
    vals = [r["val"] for r in rows]
    assert vals == [1, 2, 3, None, None]


def test_sort_nulls_first(tmp_path):
    data = {"val": [3, None, 1, None, 2], "id": list(range(5))}
    it = make_iter(data, 2, value_columns=["val"], key_columns=["id"])
    path = external_sort(
        it, ["val"], ["val"], make_guard(), tmp_path,
        ascending=True, nulls_first=True, log=False,
    )
    rows = collect(path, ["val"])
    vals = [r["val"] for r in rows]
    assert vals == [None, None, 1, 2, 3]


# ---------- многоуровневый merge ----------

def test_sort_multi_level_merge(tmp_path):
    # 100 строк, batch_size=1 -> много run'ов; fan_in=4 -> многоуровневый merge
    n = 100
    data = {"val": list(range(n, 0, -1)), "id": list(range(n))}
    it = make_iter(data, 1, value_columns=["val"], key_columns=["id"])
    path = external_sort(
        it, ["val"], ["val"], make_guard(), tmp_path,
        ascending=True, nulls_first=False, merge_fan_in=4, log=False,
    )
    rows = collect(path, ["val"])
    assert [r["val"] for r in rows] == list(range(1, n + 1))
    assert [r["row_id"] for r in rows] == list(range(n - 1, -1, -1))


# ---------- уже отсортирован ----------

def test_sort_already_sorted(tmp_path):
    data = {"val": list(range(20)), "id": list(range(20))}
    it = make_iter(data, 3, value_columns=["val"], key_columns=["id"])
    path = external_sort(
        it, ["val"], ["val"], make_guard(), tmp_path,
        ascending=True, nulls_first=False, log=False,
    )
    rows = collect(path, ["val"])
    assert [r["val"] for r in rows] == list(range(20))


# ---------- пустой вход ----------

def test_sort_empty_raises(tmp_path):
    with pytest.raises(ValueError):
        external_sort(
            iter([]), ["val"], ["val"], make_guard(), tmp_path, log=False,
        )


# ---------- row_id уникален ----------

def test_sort_row_id_unique(tmp_path):
    data = {"val": [5, 3, 5, 1, 3, 5], "id": list(range(6))}
    it = make_iter(data, 2, value_columns=["val"], key_columns=["id"])
    path = external_sort(
        it, ["val"], ["val"], make_guard(), tmp_path,
        ascending=True, nulls_first=False, log=False,
    )
    rows = collect(path, ["val"])
    row_ids = [r["row_id"] for r in rows]
    assert sorted(row_ids) == list(range(6))