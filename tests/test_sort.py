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

# ---------- v0.5.2a (P5): AST guard against Python-list buffer in merge ----------

def test_merge_keys_uses_arrow_builder_not_python_list():
    """_merge_keys must use pa.Array.Builder, not `buffer[c].append(as_py())`.

    Регрессия P5 (Pingis audit): Python-list buffer даёт ×3-5 peak vs
    Arrow. AST guard: нет `buffer[...].append` в _merge_keys.
    """
    import ast
    import inspect
    from cadw.ops import sort as sort_mod

    src = inspect.getsource(sort_mod._merge_keys)
    tree = ast.parse(src)

    # Никаких `buffer[...].append(...)` вызовов
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr == "append":
            # f.value должно быть Subscript с base Name == 'buffer'
            base = f.value
            if isinstance(base, ast.Subscript) and isinstance(base.value, ast.Name):
                if base.value.id == "buffer":
                    violations.append(f"buffer[...].append() at line {node.lineno}")

    assert not violations, (
        "P5 must use pa.Array.Builder; Python-list buffer detected:\n"
        + "\n".join(violations)
    )


def test_merge_keys_uses_arrow_builder():
    """_merge_keys must call _append_to_builders (P5 fix marker)."""
    import inspect
    from cadw.ops import sort as sort_mod
    src = inspect.getsource(sort_mod._merge_keys)
    assert "_append_to_builders" in src, (
        "P5 fix: _merge_keys must call _append_to_builders (Arrow Builder)"
    )

def test_sort_string_with_builder(tmp_path):
    """P5 fix: string sort must work with pyarrow 26 builder API.

    Regression: v0.5.2a used pa.StringBuilder which doesn't exist in
    pyarrow 26. v0.5.2a-fix2 uses multi-name fallback + BinaryBuilder→cast.
    """
    data = {"s": ["c", "a", "b", "a"], "v": [3.0, 1.0, 2.0, 4.0]}
    it = make_iter(data, 2, value_columns=["s", "v"], key_columns=["s"])
    path = external_sort(
        it, ["s"], ["s", "v"], make_guard(), tmp_path,
        ascending=True, nulls_first=False, log=False,
    )
    rows = collect(path, ["s", "v"])
    assert [r["s"] for r in rows] == ["a", "a", "b", "c"]
    assert [r["v"] for r in rows] == [1.0, 4.0, 2.0, 3.0]


def test_sort_string_large_string(tmp_path):
    """Large string builder fallback."""
    data = {"s": ["y", "x", "z"], "v": [1.0, 2.0, 3.0]}
    it = make_iter(data, 2, value_columns=["s", "v"], key_columns=["s"])
    path = external_sort(
        it, ["s"], ["s", "v"], make_guard(), tmp_path,
        ascending=True, nulls_first=False, log=False,
    )
    rows = collect(path, ["s", "v"])
    assert [r["s"] for r in rows] == ["x", "y", "z"]

def test_sort_int_with_nulls_builder_path(tmp_path):
    """Numeric with nulls — exercises _NumpyBackedBuilder null bitmap
    (np.packbits). Regression for v0.5.2a-fix5/fix6.
    """
    data = {
        "v": pa.array([3, None, 1, None, 2], type=pa.int64()),
        "id": list(range(5)),
    }
    it = make_iter(data, 2, value_columns=["v"], key_columns=["id"])
    path = external_sort(
        it, ["v"], ["v"], make_guard(), tmp_path,
        ascending=True, nulls_first=False, log=False,
    )
    rows = collect(path, ["v"])
    assert [r["v"] for r in rows] == [1, 2, 3, None, None]


def test_sort_float_with_nulls_builder_path(tmp_path):
    """Float64 with nulls — np.packbits null bitmap."""
    data = {
        "v": pa.array([3.5, None, 1.5, None, 2.5], type=pa.float64()),
        "id": list(range(5)),
    }
    it = make_iter(data, 2, value_columns=["v"], key_columns=["id"])
    path = external_sort(
        it, ["v"], ["v"], make_guard(), tmp_path,
        ascending=True, nulls_first=False, log=False,
    )
    rows = collect(path, ["v"])
    assert [r["v"] for r in rows] == [1.5, 2.5, 3.5, None, None]


def test_sort_int_nulls_first_builder_path(tmp_path):
    """Numeric nulls_first — bitmap at start."""
    data = {
        "v": pa.array([3, None, 1, None, 2], type=pa.int64()),
        "id": list(range(5)),
    }
    it = make_iter(data, 2, value_columns=["v"], key_columns=["id"])
    path = external_sort(
        it, ["v"], ["v"], make_guard(), tmp_path,
        ascending=True, nulls_first=True, log=False,
    )
    rows = collect(path, ["v"])
    assert [r["v"] for r in rows] == [None, None, 1, 2, 3]


def test_external_sort_phase1_limit_small():
    """AST guard (P5b): phase-1 buffer must be ≤ 0.25× available.

    Baseline v0.5.2a-fix5 had 0.5× → peak 3× buf. v0.5.2a-fix6 = 0.2×.
    """
    import inspect
    from cadw.ops import sort as sort_mod
    src = inspect.getsource(sort_mod.external_sort)
    assert "0.2" in src or "0.25" in src or "0.3" in src, (
        "phase-1 buffer limit must be small (0.2-0.3× available); "
        "was 0.5× (P5b regression). See DECISIONS §13.4/71."
    )
    assert "memory_guard.available_bytes() // 2" not in src, (
        "phase-1 limit must NOT be `available // 2` (0.5×) — P5b regression"
    )

def test_numpy_backed_builder_null_bitmap_direct():
    """_NumpyBackedBuilder: Arrow bitmap 1=valid (not 1=null).

    Регрессия v0.7.6.1 (B2): до фикса `packbits(mask=True=null)` давал
    1 в null-позициях, Arrow читал 1=valid → null↔valid инвертированы.
    Диагностика: packbits([F,F,T]) → pa.Array.from_buffers → [None,None,30]
    (2 nulls, неверно). После фикса: [10,20,None] (1 null).

    Тест прямой (не через external_sort) — чтобы не маскировалось
    native builder'ом, если он есть в текущем pyarrow.
    """
    from cadw.ops.sort import _NumpyBackedBuilder
    import numpy as np

    b = _NumpyBackedBuilder(pa.int64(), np.int64)
    b.append(10)
    b.append(20)
    b.append_null()
    arr = b.finish()

    assert arr.to_pylist() == [10, 20, None], (
        f"null position wrong: {arr.to_pylist()} (expected [10, 20, None])"
    )
    assert arr.null_count == 1, f"null_count={arr.null_count}, expected 1"


def test_numpy_backed_builder_null_first():
    """Bitmap: null at position 0, then valid."""
    from cadw.ops.sort import _NumpyBackedBuilder
    import numpy as np

    b = _NumpyBackedBuilder(pa.float64(), np.float64)
    b.append_null()
    b.append(1.5)
    b.append(2.5)
    arr = b.finish()

    assert arr.to_pylist() == [None, 1.5, 2.5]
    assert arr.null_count == 1


def test_numpy_backed_builder_all_valid():
    """No nulls — buffer path (no bitmap)."""
    from cadw.ops.sort import _NumpyBackedBuilder
    import numpy as np

    b = _NumpyBackedBuilder(pa.int64(), np.int64)
    b.append(1)
    b.append(2)
    arr = b.finish()

    assert arr.to_pylist() == [1, 2]
    assert arr.null_count == 0

