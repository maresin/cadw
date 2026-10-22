"""v0.9.25k: Phase C C-4a — str.slice, str.cat, str.count."""
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard


def _mk(data, mem_mb=20):
    return CadwDataFrame.from_table(
        pa.table(data),
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
    )


def _to_rows(df):
    rows = []
    for b in df._iter_filtered_batches():
        rows.extend(b.to_pylist())
    return rows


# ============ str.slice ============

def test_slice_basic():
    df = _mk({"s": ["abcdef", "xyz"]})
    out = df["s"].str.slice(1, 4)
    rows = _to_rows(out)
    # 'abcdef'[1:4] = 'bcd'; 'xyz'[1:4] = 'yz' (truncated to len)
    assert [r["s_slice"] for r in rows] == ["bcd", "yz"]


def test_slice_negative_start():
    df = _mk({"s": ["abcdef", "xyz"]})
    out = df["s"].str.slice(-2)
    rows = _to_rows(out)
    # 'abcdef'[-2:] = 'ef'; 'xyz'[-2:] = 'yz'
    assert [r["s_slice"] for r in rows] == ["ef", "yz"]


def test_slice_stop_none():
    df = _mk({"s": ["abcdef"]})
    out = df["s"].str.slice(2)
    rows = _to_rows(out)
    assert [r["s_slice"] for r in rows] == ["cdef"]


def test_slice_negative_stop():
    df = _mk({"s": ["abcdef"]})
    out = df["s"].str.slice(0, -1)
    rows = _to_rows(out)
    assert [r["s_slice"] for r in rows] == ["abcde"]


def test_slice_out_of_bounds():
    df = _mk({"s": ["abc", "hi"]})
    out = df["s"].str.slice(10, 20)
    rows = _to_rows(out)
    # OOB → empty string (per pc.utf8_slice_codeunits)
    assert [r["s_slice"] for r in rows] == ["", ""]


def test_slice_null_input():
    df = _mk({"s": pa.array(["abc", None, "def"], type=pa.string())})
    out = df["s"].str.slice(0, 1)
    rows = _to_rows(out)
    assert [r["s_slice"] for r in rows] == ["a", None, "d"]


def test_slice_non_string_raises():
    df = _mk({"v": [1, 2]})
    with pytest.raises(TypeError, match="string"):
        df["v"].str.slice(0, 1)


def test_slice_start_non_int_raises():
    df = _mk({"s": ["abc"]})
    with pytest.raises(TypeError, match="start must be int"):
        df["s"].str.slice("0", 1)


def test_slice_preserves_original():
    df = _mk({"s": ["abcdef"], "v": [42]})
    out = df["s"].str.slice(1, 4)
    rows = _to_rows(out)
    assert rows[0]["s"] == "abcdef"
    assert rows[0]["v"] == 42
    assert rows[0]["s_slice"] == "bcd"


def test_slice_filtered():
    df = _mk({"s": ["abc", "xyz", "def"], "v": [1, 2, 3]})
    filtered = df[df["v"] > 1]
    out = filtered["s"].str.slice(0, 2)
    rows = _to_rows(out)
    assert [r["s_slice"] for r in rows] == ["xy", "de"]


# ============ str.cat ============

def test_cat_basic():
    df = _mk({"a": ["x", "y"], "b": ["p", "q"]})
    out = df["a"].str.cat("b", sep="-")
    rows = _to_rows(out)
    assert [r["a_cat_b"] for r in rows] == ["x-p", "y-q"]


def test_cat_empty_sep():
    df = _mk({"a": ["x", "y"], "b": ["p", "q"]})
    out = df["a"].str.cat("b")  # default sep=''
    rows = _to_rows(out)
    assert [r["a_cat_b"] for r in rows] == ["xp", "yq"]


def test_cat_custom_sep():
    df = _mk({"a": ["x", "y"], "b": ["p", "q"]})
    out = df["a"].str.cat("b", sep="::")
    rows = _to_rows(out)
    assert [r["a_cat_b"] for r in rows] == ["x::p", "y::q"]


def test_cat_null_in_left():
    df = _mk({
        "a": pa.array(["x", None], type=pa.string()),
        "b": ["p", "q"],
    })
    out = df["a"].str.cat("b", sep="-")
    rows = _to_rows(out)
    assert [r["a_cat_b"] for r in rows] == ["x-p", None]


def test_cat_null_in_right():
    df = _mk({
        "a": ["x", "y"],
        "b": pa.array([None, "q"], type=pa.string()),
    })
    out = df["a"].str.cat("b", sep="-")
    rows = _to_rows(out)
    assert [r["a_cat_b"] for r in rows] == [None, "y-q"]


def test_cat_null_in_both():
    df = _mk({
        "a": pa.array([None, "x"], type=pa.string()),
        "b": pa.array([None, "y"], type=pa.string()),
    })
    out = df["a"].str.cat("b")
    rows = _to_rows(out)
    assert [r["a_cat_b"] for r in rows] == [None, "xy"]


def test_cat_missing_column_raises():
    df = _mk({"a": ["x"]})
    with pytest.raises(KeyError, match="nope"):
        df["a"].str.cat("nope")


def test_cat_non_string_raises():
    df = _mk({"a": ["x"], "b": [1]})
    with pytest.raises(TypeError, match="string"):
        df["a"].str.cat("b")


def test_cat_non_string_sep_raises():
    df = _mk({"a": ["x"], "b": ["y"]})
    with pytest.raises(TypeError, match="sep must be str"):
        df["a"].str.cat("b", sep=1)


def test_cat_self_concatenation_raises():
    df = _mk({"a": ["x"]})
    with pytest.raises(ValueError, match="column with itself"):
        df["a"].str.cat("a")


def test_cat_preserves_other_columns():
    df = _mk({"a": ["x"], "b": ["y"], "v": [42]})
    out = df["a"].str.cat("b")
    rows = _to_rows(out)
    assert rows[0]["a"] == "x"
    assert rows[0]["b"] == "y"
    assert rows[0]["v"] == 42


# ============ str.count ============

def test_count_basic():
    df = _mk({"s": ["a-b-c", "x-y", "abc"]})
    out = df["s"].str.count("-")
    rows = _to_rows(out)
    assert [r["s_count"] for r in rows] == [2, 1, 0]


def test_count_not_found():
    df = _mk({"s": ["abc", "def"]})
    out = df["s"].str.count("xyz")
    rows = _to_rows(out)
    assert [r["s_count"] for r in rows] == [0, 0]


def test_count_empty_string_input():
    df = _mk({"s": ["", "abc"]})
    out = df["s"].str.count("a")
    rows = _to_rows(out)
    assert [r["s_count"] for r in rows] == [0, 1]


def test_count_null_input():
    df = _mk({"s": pa.array(["abc", None, "def"], type=pa.string())})
    out = df["s"].str.count("a")
    rows = _to_rows(out)
    assert [r["s_count"] for r in rows] == [1, None, 0]


def test_count_non_string_raises():
    df = _mk({"v": [1, 2]})
    with pytest.raises(TypeError, match="string"):
        df["v"].str.count("a")


def test_count_empty_pat_raises():
    df = _mk({"s": ["abc"]})
    with pytest.raises(ValueError, match="non-empty"):
        df["s"].str.count("")


def test_count_preserves_original():
    df = _mk({"s": ["a-b-c"], "v": [42]})
    out = df["s"].str.count("-")
    rows = _to_rows(out)
    assert rows[0]["s"] == "a-b-c"
    assert rows[0]["v"] == 42
    assert rows[0]["s_count"] == 2


def test_count_after_filter():
    df = _mk({"s": ["a-b", "c-d-e", "f"], "v": [1, 2, 3]})
    filtered = df[df["v"] > 1]
    out = filtered["s"].str.count("-")
    rows = _to_rows(out)
    assert [r["s_count"] for r in rows] == [2, 0]
