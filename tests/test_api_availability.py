"""
Pre-flight guard: API PyArrow / xxhash, которые CADW
использует. Если внешняя библиотека изменит сигнатуру — этот файл упадёт
первым с понятным сообщением, а не 30 тестов с AttributeError.

Основание: DECISIONS.md §13.4 — 24 эпизода отладки, ~40% из них —
API-контракты. После патча 09 sklearn удалён из CADW; остаются только
PyArrow и xxhash. Классы: параметры (`type_` → `type`), методы
(`arr.length` → `len(arr)`, `num_data` → `num_data()`), отсутствующие
функции (`pc.hash`, `serialize_schema`), deprecated kwargs
(`IncrementalPCA(random_state=)`), required kwargs (`classes=`, `n_features=`).
"""
import inspect
import pytest


# ---------- PyArrow ----------

def test_pyarrow_version_and_imports():
    import pyarrow as pa
    import pyarrow.compute as pc
    assert pa.__version__ >= "26.0.0", f"pyarrow>=26 required, got {pa.__version__}"
    for name in ("add", "subtract", "multiply", "divide",
                 "shift_left", "shift_right", "bit_wise_and", "bit_wise_xor",
                 "and_", "or_", "invert",
                 "greater", "greater_equal", "less", "less_equal", "equal", "not_equal",
                 "sum", "mean", "min_max",
                 "sort_indices"):
        assert hasattr(pc, name), f"pc.{name} missing in pyarrow {pa.__version__}"


def test_pa_array_type_kwarg():
    """pa.array(values, type=...) — параметр `type`, не `type_`."""
    import pyarrow as pa
    sig = inspect.signature(pa.array)
    assert "type" in sig.parameters, "pa.array must accept type= kwarg"


def test_pa_array_has_len_not_length():
    """len(arr), не arr.length."""
    import pyarrow as pa
    arr = pa.array([1, 2, 3])
    assert len(arr) == 3
    assert not hasattr(arr, "length"), "pa.Array should not have .length; use len()"


def test_schema_serialize():
    """pa.Schema.serialize() — метод, не pa.ipc.serialize_schema."""
    import pyarrow as pa
    schema = pa.schema([("x", pa.int64())])
    assert hasattr(schema, "serialize")
    assert callable(schema.serialize)


def test_ipc_read_schema():
    import pyarrow.ipc as ipc
    assert hasattr(ipc, "read_schema")


def test_table_group_by():
    """pa.Table.group_by — есть; RecordBatch.group_by — нет."""
    import pyarrow as pa
    t = pa.table({"x": [1, 2], "y": [3.0, 4.0]})
    assert hasattr(t, "group_by")
    b = t.to_batches()[0]
    assert not hasattr(b, "group_by"), "RecordBatch.group_by should not exist; use Table"


def test_sort_indices_null_placement_per_key():
    """null_placement должен приниматься в sort_keys (per-key), а не в SortOptions."""
    import pyarrow as pa
    import pyarrow.compute as pc
    t = pa.table({"x": [3, None, 1]})
    # per-key null_placement: (name, order, null_placement)
    order = pc.sort_indices(t, sort_keys=[("x", "ascending", "at_end")])
    assert order.to_pylist() == [2, 0, 1]


# ---------- xxhash ----------

def test_xxhash_accepts_bytes():
    """xxh64 принимает bytes, не str."""
    import xxhash
    h = xxhash.xxh64(b"abc", seed=0).intdigest()
    assert isinstance(h, int)
    with pytest.raises(TypeError):
        xxhash.xxh64("abc", seed=0)
