"""v0.9.25j: Phase C batch C-3b — cut, qcut, duplicated."""
import math

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


# ============ cut ============

def test_cut_int_bins_basic():
    df = _mk({"x": [0.0, 2.5, 5.0, 7.5, 10.0]})
    result, edges = df.cut("x", bins=2)
    assert len(edges) == 3
    # Edges [0, 5, 10]; labels 0 (0<=x<=5), 1 (5<x<=10)
    rows = _to_rows(result)
    labels = [r["x_bin"] for r in rows]
    assert labels == [0, 0, 0, 1, 1]


def test_cut_list_bins_basic():
    df = _mk({"x": [0.0, 2.5, 5.0, 7.5, 10.0, 15.0]})
    result, edges = df.cut("x", bins=[0, 5, 10])
    assert edges == [0.0, 5.0, 10.0]
    rows = _to_rows(result)
    labels = [r["x_bin"] for r in rows]
    # 0→bin0 (include_lowest), 2.5→bin0, 5→bin0 (<=5),
    # 7.5→bin1, 10→bin1, 15→null (above max)
    assert labels == [0, 0, 0, 1, 1, None]


def test_cut_edges_returned():
    df = _mk({"x": list(range(11))})
    _, edges = df.cut("x", bins=2)
    assert edges == [0.0, 5.0, 10.0]


def test_cut_right_false():
    df = _mk({"x": [0.0, 5.0, 10.0]})
    result, _ = df.cut("x", bins=[0, 5, 10], right=False,
                       include_lowest=False)
    rows = _to_rows(result)
    # left-closed: [0, 5) → bin0, [5, 10) → bin1, 10 → null
    labels = [r["x_bin"] for r in rows]
    assert labels == [0, 1, None]


def test_cut_include_lowest_false():
    df = _mk({"x": [0.0, 2.5, 5.0]})
    result, _ = df.cut("x", bins=[0, 5, 10], include_lowest=False)
    rows = _to_rows(result)
    # 0 == min edge; include_lowest=False → null
    labels = [r["x_bin"] for r in rows]
    assert labels == [None, 0, 0]


def test_cut_null_input():
    df = _mk({"x": pa.array([1.0, None, 9.0], type=pa.float64())})
    result, _ = df.cut("x", bins=2)
    rows = _to_rows(result)
    labels = [r["x_bin"] for r in rows]
    assert labels[1] is None


def test_cut_all_same_value():
    df = _mk({"x": [5.0, 5.0, 5.0]})
    result, edges = df.cut("x", bins=3)
    # All identical → 1 bin
    assert len(edges) == 2
    rows = _to_rows(result)
    labels = [r["x_bin"] for r in rows]
    assert labels == [0, 0, 0]


def test_cut_empty():
    df = _mk({"x": [1.0, 2.0], "v": [10, 20]})
    empty = df[df["v"] > 100]
    result, edges = empty.cut("x", bins=2)
    assert result.count() == 0
    assert edges == []


def test_cut_missing_column_raises():
    df = _mk({"x": [1.0]})
    with pytest.raises(KeyError, match="nope"):
        df.cut("nope", bins=2)


def test_cut_non_numeric_raises():
    df = _mk({"s": ["a", "b"]})
    with pytest.raises(TypeError, match="numeric"):
        df.cut("s", bins=2)


def test_cut_invalid_bins_int():
    df = _mk({"x": [1.0, 2.0]})
    with pytest.raises(ValueError, match="bins"):
        df.cut("x", bins=0)


def test_cut_invalid_bins_list():
    df = _mk({"x": [1.0, 2.0]})
    with pytest.raises(ValueError, match=">= 2"):
        df.cut("x", bins=[5])
    with pytest.raises(ValueError, match="strictly ascending"):
        df.cut("x", bins=[5, 5])
    with pytest.raises(ValueError, match="strictly ascending"):
        df.cut("x", bins=[5, 3])


def test_cut_preserves_other_columns():
    df = _mk({"x": [1.0, 2.0, 3.0], "v": [10, 20, 30]})
    result, _ = df.cut("x", bins=2)
    assert "v" in result.columns
    assert "x" in result.columns
    assert "x_bin" in result.columns
    # Original x unchanged
    rows = _to_rows(result)
    assert [r["x"] for r in rows] == [1.0, 2.0, 3.0]


# ============ qcut ============

def test_qcut_int_bins():
    df = _mk({"x": list(range(10))})
    result, edges = df.qcut("x", q=2)
    # Edges = 3 (min, median, max) → 2 bins
    assert len(edges) == 3
    rows = _to_rows(result)
    labels = [r["x_bin"] for r in rows]
    # 0..4 → bin0, 5..9 → bin1
    assert labels[:5] == [0] * 5
    assert labels[5:] == [1] * 5


def test_qcut_4_quartiles():
    df = _mk({"x": list(range(100))})
    result, edges = df.qcut("x", q=4)
    assert len(edges) == 5  # min, q25, q50, q75, max
    rows = _to_rows(result)
    # Each bin should have ~25 rows
    labels = [r["x_bin"] for r in rows]
    counts = [labels.count(i) for i in range(4)]
    for c in counts:
        assert 20 <= c <= 30, f"uneven qcut distribution: {counts}"


def test_qcut_list_q():
    df = _mk({"x": list(range(100))})
    result, edges = df.qcut("x", q=[0.25, 0.5, 0.75])
    # edges = [min] + quantile(q) + [max] → 5 edges → 4 bins
    assert len(edges) == 5
    rows = _to_rows(result)
    assert len(rows) == 100


def test_qcut_edges_sorted():
    df = _mk({"x": list(range(50))})
    _, edges = df.qcut("x", q=3)
    assert edges == sorted(edges)
    assert len(edges) == 4


def test_qcut_empty():
    df = _mk({"x": [1.0, 2.0], "v": [10, 20]})
    empty = df[df["v"] > 100]
    result, edges = empty.qcut("x", q=2)
    assert result.count() == 0
    assert edges == []


def test_qcut_all_same_raises():
    df = _mk({"x": [5.0, 5.0, 5.0]})
    with pytest.raises(ValueError, match="identical"):
        df.qcut("x", q=2)


def test_qcut_missing_column_raises():
    df = _mk({"x": [1.0]})
    with pytest.raises(KeyError, match="nope"):
        df.qcut("nope", q=2)


def test_qcut_non_numeric_raises():
    df = _mk({"s": ["a", "b"]})
    with pytest.raises(TypeError, match="numeric"):
        df.qcut("s", q=2)


def test_qcut_null_input():
    df = _mk({"x": pa.array([1.0, None, 9.0, 5.0], type=pa.float64())})
    result, _ = df.qcut("x", q=2)
    rows = _to_rows(result)
    labels = [r["x_bin"] for r in rows]
    assert labels[1] is None


# ============ duplicated ============

def test_duplicated_first_basic():
    df = _mk({"k": [1, 2, 1, 3, 1], "v": [10, 20, 30, 40, 50]})
    result = df.duplicated(subset=["k"])
    rows = _to_rows(result)
    mask = [r["duplicated"] for r in rows]
    # First 1 (pos 0) not dup, second 1 (pos 2) is dup, third 1 (pos 4) is dup
    assert mask == [False, False, True, False, True]


def test_duplicated_last_basic():
    df = _mk({"k": [1, 2, 1, 3, 1], "v": [10, 20, 30, 40, 50]})
    result = df.duplicated(subset=["k"], keep="last")
    rows = _to_rows(result)
    mask = [r["duplicated"] for r in rows]
    # Last 1 (pos 4) not dup, earlier 1s are dup
    assert mask == [True, False, True, False, False]


def test_duplicated_false_basic():
    df = _mk({"k": [1, 2, 1, 3, 1], "v": [10, 20, 30, 40, 50]})
    result = df.duplicated(subset=["k"], keep=False)
    rows = _to_rows(result)
    mask = [r["duplicated"] for r in rows]
    # All 1s are dup (three of them), 2 and 3 are unique
    assert mask == [True, False, True, False, True]


def test_duplicated_no_dupes():
    df = _mk({"k": [1, 2, 3, 4, 5], "v": [10, 20, 30, 40, 50]})
    result = df.duplicated(subset=["k"])
    rows = _to_rows(result)
    mask = [r["duplicated"] for r in rows]
    assert mask == [False] * 5


def test_duplicated_all_dupes():
    df = _mk({"k": [1, 1, 1, 1], "v": [10, 20, 30, 40]})
    result = df.duplicated(subset=["k"])
    rows = _to_rows(result)
    mask = [r["duplicated"] for r in rows]
    assert mask == [False, True, True, True]


def test_duplicated_multi_subset():
    df = _mk({
        "a": [1, 1, 2, 2, 1],
        "b": [10, 20, 30, 40, 10],
        "v": [100, 200, 300, 400, 500],
    })
    result = df.duplicated(subset=["a", "b"])
    rows = _to_rows(result)
    mask = [r["duplicated"] for r in rows]
    # (1,10) appears twice: pos 0 (first, not dup), pos 4 (dup)
    assert mask == [False, False, False, False, True]


def test_duplicated_null_in_subset():
    df = _mk({
        "k": pa.array([1, None, 1, None], type=pa.int64()),
        "v": [10, 20, 30, 40],
    })
    result = df.duplicated(subset=["k"])
    rows = _to_rows(result)
    mask = [r["duplicated"] for r in rows]
    # pos 0 (k=1) not dup, pos 1 (None) not dup, pos 2 (k=1) dup, pos 3 (None) dup
    assert mask == [False, False, True, True]


def test_duplicated_preserves_original_columns():
    df = _mk({"k": [1, 1], "v": [10, 20]})
    result = df.duplicated(subset=["k"])
    rows = _to_rows(result)
    assert "k" in rows[0]
    assert "v" in rows[0]
    assert "duplicated" in rows[0]
    # Original values preserved
    assert [r["v"] for r in rows] == [10, 20]


def test_duplicated_filter_usage():
    """Users filter dup rows via df2[df2['duplicated'] == True]."""
    df = _mk({"k": [1, 2, 1, 3], "v": [10, 20, 30, 40]})
    dup_df = df.duplicated(subset=["k"])
    filtered = dup_df[dup_df["duplicated"] == True]
    assert filtered.count() == 1
    rows = _to_rows(filtered)
    assert rows[0]["v"] == 30  # row 2 (pos 2) is the duplicate of k=1


def test_duplicated_empty():
    df = _mk({"k": [1, 2], "v": [10, 20]})
    empty = df[df["v"] > 100]
    result = empty.duplicated(subset=["k"])
    assert result.count() == 0
    assert "duplicated" in result.columns


def test_duplicated_missing_column_raises():
    df = _mk({"k": [1, 2]})
    with pytest.raises(KeyError, match="nope"):
        df.duplicated(subset=["nope"])


def test_duplicated_invalid_keep_raises():
    df = _mk({"k": [1, 2]})
    with pytest.raises(ValueError, match="keep"):
        df.duplicated(subset=["k"], keep="middle")


def test_duplicated_empty_subset_raises():
    df = _mk({"k": [1, 2]})
    with pytest.raises(ValueError, match="non-empty"):
        df.duplicated(subset=[])
