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

# ---------- v0.7.8-fix3: every name in __all__ actually imports ----------
#
# Catches silent no-op patch bugs (§13.4/44, 111, 114, 115): a patched
# __init__.py could add a name to `__all__` but fail to add the import
# line, leaving ImportError for downstream users while test collection
# passes (until someone does `from cadw import X`).
#
# Three levels of verify:
#   1. AST: is the name in an ImportFrom / Assign?
#   2. __all__: is every name present in module?
#   3. Runtime: can we actually `from pkg import name`?

import ast as _ast
import inspect as _inspect
import importlib as _importlib


def test_cadw_all_exports_importable():
    """Every name in cadw.__all__ must be importable from cadw."""
    import cadw
    missing = []
    for name in cadw.__all__:
        if not hasattr(cadw, name):
            missing.append(name)
    assert not missing, (
        f"cadw.__all__ references names not importable from cadw: "
        f"{missing}. Likely __all__ updated without import line (§13.4/115)."
    )


def test_cadw_ml_all_exports_importable():
    """Every name in cadw.ml.__all__ must be importable from cadw.ml."""
    import cadw.ml
    missing = []
    for name in cadw.ml.__all__:
        if not hasattr(cadw.ml, name):
            missing.append(name)
    assert not missing, (
        f"cadw.ml.__all__ references names not importable from cadw.ml: "
        f"{missing}"
    )


def test_cadw_ast_all_matches_imports():
    """AST: every name in `cadw.__all__` is either imported or assigned
    at module level. Catches the specific bug where __all__ line was
    added but import line was missed.
    """
    import cadw
    src = _inspect.getsource(cadw)
    tree = _ast.parse(src)

    imported: set[str] = set()
    assigned: set[str] = set()
    for node in tree.body:
        if isinstance(node, _ast.ImportFrom):
            for alias in node.names:
                imported.add(alias.asname or alias.name)
        elif isinstance(node, _ast.Import):
            for alias in node.names:
                imported.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, _ast.Assign):
            for target in node.targets:
                if isinstance(target, _ast.Name):
                    assigned.add(target.id)

    all_names: set[str] = set()
    for node in tree.body:
        if isinstance(node, _ast.Assign):
            for target in node.targets:
                if (isinstance(target, _ast.Name)
                        and target.id == "__all__"
                        and isinstance(node.value, _ast.List)):
                    for elt in node.value.elts:
                        if isinstance(elt, _ast.Constant):
                            all_names.add(elt.value)

    missing = all_names - imported - assigned - {"__all__"}
    # Allow names re-exported from other modules (they appear in ImportFrom
    # and thus would be in `imported` already). If anything is missing —
    # the __all__ line was added without the import.
    assert not missing, (
        f"cadw/__init__.py: names in __all__ but not imported/assigned: "
        f"{missing}. Add `from ... import name` line (§13.4/115)."
    )

