"""v0.9.25g: Phase C batch C-1 — isin, between, select_dtypes."""
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard


def _mk(data, mem_mb=20):
    return CadwDataFrame.from_table(
        pa.table(data),
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
    )


# ============ isin ============


def test_isin_basic():
    df = _mk({"x": [1, 2, 3, 4], "v": [10, 20, 30, 40]})
    sub = df[df["x"].isin([2, 3])]
    assert sub.count() == 2
    out = sub.head(10)
    assert sorted(out.column("v").to_pylist()) == [20, 30]


def test_isin_empty_list():
    df = _mk({"x": [1, 2, 3]})
    sub = df[df["x"].isin([])]
    assert sub.count() == 0


def test_isin_strings():
    df = _mk({
        "s": ["a", "b", "c", "a"], "v": [1, 2, 3, 4],
    })
    sub = df[df["s"].isin(["a", "c"])]
    assert sub.count() == 3


def test_isin_null_row_not_matched():
    df = _mk({
        "x": pa.array([1, None, 3], type=pa.int64()),
        "v": [10, 20, 30],
    })
    sub = df[df["x"].isin([1, 3])]
    assert sub.count() == 2


def test_isin_null_in_values_matches_null_row():
    df = _mk({
        "x": pa.array([1, None, 3], type=pa.int64()),
        "v": [10, 20, 30],
    })
    sub = df[df["x"].isin([1, None])]
    assert sub.count() == 2


def test_isin_bad_type_raises():
    df = _mk({"x": [1, 2, 3]})
    # Wrong value type: int column, string values → TypeError at mask
    with pytest.raises(TypeError, match="isin"):
        df[df["x"].isin(["a", "b"])].count()


# ============ between ============


def test_between_basic_both():
    df = _mk({"x": [1, 5, 10, 15, 20], "v": [1, 2, 3, 4, 5]})
    sub = df[df["x"].between(5, 15)]
    assert sub.count() == 3  # 5, 10, 15 (both inclusive)


def test_between_neither():
    df = _mk({"x": [1, 5, 10, 15, 20]})
    sub = df[df["x"].between(5, 15, inclusive="neither")]
    assert sub.count() == 1  # 10 only


def test_between_left():
    df = _mk({"x": [1, 5, 10, 15, 20]})
    sub = df[df["x"].between(5, 15, inclusive="left")]
    assert sub.count() == 2  # 5, 10


def test_between_right():
    df = _mk({"x": [1, 5, 10, 15, 20]})
    sub = df[df["x"].between(5, 15, inclusive="right")]
    assert sub.count() == 2  # 10, 15


def test_between_null_row_not_matched():
    df = _mk({
        "x": pa.array([1, None, 10], type=pa.int64()),
    })
    sub = df[df["x"].between(0, 20)]
    assert sub.count() == 2  # null drops


def test_between_lo_gt_hi_empty():
    df = _mk({"x": [1, 5, 10]})
    sub = df[df["x"].between(10, 5)]
    assert sub.count() == 0  # no reorder


def test_between_invalid_inclusive():
    df = _mk({"x": [1, 2, 3]})
    with pytest.raises(ValueError, match="between: inclusive"):
        df[df["x"].between(1, 2, inclusive="bogus")].count()


def test_between_float():
    df = _mk({"x": [1.5, 2.5, 3.5, 4.5]})
    sub = df[df["x"].between(2.0, 4.0)]
    assert sub.count() == 2  # 2.5, 3.5


# ============ select_dtypes ============


def test_select_dtypes_include_number():
    df = _mk({"i": [1, 2], "f": [1.0, 2.0], "s": ["a", "b"]})
    sub = df.select_dtypes(include="number")
    assert sorted(sub.columns) == ["f", "i"]


def test_select_dtypes_exclude_string():
    df = _mk({"i": [1, 2], "f": [1.0, 2.0], "s": ["a", "b"]})
    sub = df.select_dtypes(exclude="string")
    assert sorted(sub.columns) == ["f", "i"]


def test_select_dtypes_int_only():
    df = _mk({"i": [1, 2], "f": [1.0, 2.0]})
    sub = df.select_dtypes(include="int")
    assert sub.columns == ["i"]


def test_select_dtypes_float_only():
    df = _mk({"i": [1, 2], "f": [1.0, 2.0]})
    sub = df.select_dtypes(include="float")
    assert sub.columns == ["f"]


def test_select_dtypes_include_list():
    df = _mk({"i": [1, 2], "f": [1.0, 2.0], "s": ["a", "b"]})
    sub = df.select_dtypes(include=["int64", "string"])
    assert sorted(sub.columns) == ["i", "s"]


def test_select_dtypes_include_and_exclude():
    df = _mk({
        "i": [1, 2], "f32": pa.array([1.0, 2.0], type=pa.float32()),
        "f64": pa.array([1.0, 2.0], type=pa.float64()),
    })
    sub = df.select_dtypes(include="float", exclude="float32")
    assert sub.columns == ["f64"]


def test_select_dtypes_all():
    df = _mk({"i": [1], "f": [1.0], "s": ["a"]})
    sub = df.select_dtypes(include="all")
    assert sorted(sub.columns) == ["f", "i", "s"]


def test_select_dtypes_no_match_raises():
    df = _mk({"i": [1, 2]})
    with pytest.raises(ValueError, match="no columns matched"):
        df.select_dtypes(include="string")


def test_select_dtypes_no_include_exclude_raises():
    df = _mk({"i": [1, 2]})
    with pytest.raises(ValueError, match="at least one"):
        df.select_dtypes()


def test_select_dtypes_preserves_filter():
    df = _mk({"x": [1, 2, 3, 4], "v": [10, 20, 30, 40]})
    filtered = df[df["x"] > 1]
    sub = filtered.select_dtypes(include="number")
    # 3 filtered rows survive, but only "v" selected (x is int, still number...)
    # both x and v are int → both kept
    assert set(sub.columns) == {"x", "v"}
    assert sub.count() == 3


# ============ integration with __getitem__ ============


def test_isin_and_between_chain():
    df = _mk({"x": [1, 5, 10, 15, 20], "v": [1, 2, 3, 4, 5]})
    sub = df[(df["x"].isin([1, 5, 10, 15])) & (df["x"].between(5, 15))]
    # isin: 1, 5, 10, 15 → 4 rows; AND between 5..15 → 5, 10, 15
    assert sub.count() == 3


def test_select_dtypes_explicit_type_for_alias():
    """v0.9.25g-fix1 (§13.4/262): explicit dtype name via pa.type_for_alias.

    str(pa.float32()) == "float", str(pa.float64()) == "double" —
    string-compare would fail both. type_for_alias handles them.
    """
    df = _mk({
        "i32": pa.array([1, 2], type=pa.int32()),
        "i64": pa.array([1, 2], type=pa.int64()),
        "f32": pa.array([1.0, 2.0], type=pa.float32()),
        "f64": pa.array([1.0, 2.0], type=pa.float64()),
    })
    assert df.select_dtypes(include="int32").columns == ["i32"]
    assert df.select_dtypes(include="int64").columns == ["i64"]
    assert df.select_dtypes(include="float32").columns == ["f32"]
    assert df.select_dtypes(include="float64").columns == ["f64"]
    assert df.select_dtypes(include="double").columns == ["f64"]


def test_select_dtypes_string_category_matches_all_strings():
    """'string' category (CADW vocabulary) matches string + large_string
    + binary — broader-than-exact by design (pandas-parity: pandas
    'string' also covers multiple Arrow string backings).
    """
    df = _mk({
        "s": pa.array(["a", "b"], type=pa.string()),
        "ls": pa.array(["a", "b"], type=pa.large_string()),
    })
    assert sorted(df.select_dtypes(include="string").columns) == ["ls", "s"]


def test_select_dtypes_bracket_syntax_rejected():
    """v0.9.25g-fix3 (§13.4/264): pandas BRACKET syntax
    (`string[pyarrow]`) not resolvable by pa.type_for_alias → CADW
    select_dtypes raises ValueError.

    Only bracket syntax is rejected. Case-insensitive aliases (`Int64`,
    `Float64`) resolve via pa.type_for_alias — see
    test_select_dtypes_alias_case_insensitive.

    Why: `pa.type_for_alias("string[pyarrow]")` raises KeyError
    (bracket syntax isn't in PyArrow's alias table). fix1's
    `except (ValueError, KeyError, TypeError)` catches it, falls
    through to `str(t) == s` → False for all → ValueError from
    select_dtypes "no columns matched".
    """
    df = _mk({"s": pa.array(["a", "b"], type=pa.string())})
    with pytest.raises(ValueError, match="no columns matched"):
        df.select_dtypes(include="string[pyarrow]")


def test_select_dtypes_alias_case_insensitive():
    """v0.9.25g-fix3 (§13.4/264): pa.type_for_alias is case-
    insensitive for most aliases. `Int64` resolves to `pa.int64()`
    — CADW naturally accepts pandas-style capitalized type names
    without any special code.

    Empirically (PyArrow 26):
        pa.type_for_alias("Int64")  → pa.int64()    ← case-insensitive
        pa.type_for_alias("int64")  → pa.int64()
        pa.type_for_alias("Float64") → pa.float64()

    Type-mismatch (different bit-width or int vs float) → no match
    → ValueError.

    (v0.9.25g-fix2 wrongly assumed Int64 would be rejected; this
    test documents actual behavior.)
    """
    df_int64 = _mk({"i": pa.array([1, 2], type=pa.int64())})
    df_int32 = _mk({"i": pa.array([1, 2], type=pa.int32())})
    df_float64 = _mk({"f": pa.array([1.0, 2.0], type=pa.float64())})

    # Case-insensitive: Int64 == int64 (matches)
    assert df_int64.select_dtypes(include="Int64").columns == ["i"]
    assert df_float64.select_dtypes(include="Float64").columns == ["f"]

    # Different bit-width: no match → ValueError
    with pytest.raises(ValueError, match="no columns matched"):
        df_int32.select_dtypes(include="Int64")

    # Float vs int: no match → ValueError
    with pytest.raises(ValueError, match="no columns matched"):
        df_int64.select_dtypes(include="Float64")
