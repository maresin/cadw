"""v0.9.25i: Phase C batch C-3a — ffill, bfill, reindex."""
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


def _mk_multi(tmp_path, data, chunk_size, mem_mb=20):
    src = tmp_path / "src.parquet"
    pq.write_table(pa.table(data), src, row_group_size=chunk_size)
    return CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=chunk_size,
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
        spill_dir=tmp_path / "spill",
    )


# ============ ffill ============


def test_ffill_basic():
    df = _mk({"x": pa.array([1, None, None, 2, None],
                             type=pa.int64())})
    result = df.ffill()
    vals = result.head(10).column("x").to_pylist()
    assert vals == [1, 1, 1, 2, 2]


def test_ffill_leading_null_stays():
    df = _mk({"x": pa.array([None, 1, None, 2], type=pa.int64())})
    result = df.ffill()
    vals = result.head(10).column("x").to_pylist()
    assert vals == [None, 1, 1, 2]


def test_ffill_all_null():
    df = _mk({"x": pa.array([None, None, None], type=pa.int64())})
    result = df.ffill()
    vals = result.head(10).column("x").to_pylist()
    assert vals == [None, None, None]


def test_ffill_no_nulls():
    df = _mk({"x": [1, 2, 3]})
    result = df.ffill()
    vals = result.head(10).column("x").to_pylist()
    assert vals == [1, 2, 3]


def test_ffill_float():
    df = _mk({"x": pa.array([1.5, None, 2.5], type=pa.float64())})
    result = df.ffill()
    vals = result.head(10).column("x").to_pylist()
    assert vals == [1.5, 1.5, 2.5]


def test_ffill_multiple_columns():
    df = _mk({
        "a": pa.array([1, None, 3], type=pa.int64()),
        "b": pa.array([10.0, 20.0, None], type=pa.float64()),
    })
    result = df.ffill()
    out = result.head(10)
    assert out.column("a").to_pylist() == [1, 1, 3]
    assert out.column("b").to_pylist() == [10.0, 20.0, 20.0]


def test_ffill_columns_subset():
    df = _mk({
        "a": pa.array([1, None, 3], type=pa.int64()),
        "b": pa.array([10, None, 30], type=pa.int64()),
    })
    result = df.ffill(columns=["a"])
    out = result.head(10)
    assert out.column("a").to_pylist() == [1, 1, 3]
    assert out.column("b").to_pylist() == [10, None, 30]


def test_ffill_cross_batch(tmp_path):
    df = _mk_multi(
        tmp_path,
        {"x": pa.array([1, None, None, 2, None], type=pa.int64())},
        chunk_size=1,
    )
    result = df.ffill()
    vals = result.head(10).column("x").to_pylist()
    assert vals == [1, 1, 1, 2, 2]


def test_ffill_filtered():
    df = _mk({
        "x": pa.array([1, None, 3, None, 5], type=pa.int64()),
        "v": [10, 20, 30, 40, 50],
    })
    filtered = df[df["v"] > 15]
    result = filtered.ffill()
    vals = result.head(10).column("x").to_pylist()
    assert vals == [None, 3, 3, 5]


def test_ffill_empty():
    df = _mk({"x": [1, 2], "v": [10, 20]})
    empty = df[df["v"] > 100]
    result = empty.ffill()
    out = result.head(10)
    assert out.num_rows == 0


def test_ffill_string():
    df = _mk({
        "s": pa.array(["a", None, "b"], type=pa.string()),
    })
    result = df.ffill()
    vals = result.head(10).column("s").to_pylist()
    assert vals == ["a", "a", "b"]


# ============ bfill ============


def test_bfill_basic():
    df = _mk({"x": pa.array([None, None, 3, None],
                             type=pa.int64())})
    result = df.bfill()
    vals = result.head(10).column("x").to_pylist()
    assert vals == [3, 3, 3, None]


def test_bfill_trailing_null_stays():
    df = _mk({"x": pa.array([1, None, None], type=pa.int64())})
    result = df.bfill()
    vals = result.head(10).column("x").to_pylist()
    assert vals == [1, None, None]


def test_bfill_all_null():
    df = _mk({"x": pa.array([None, None], type=pa.int64())})
    result = df.bfill()
    vals = result.head(10).column("x").to_pylist()
    assert vals == [None, None]


def test_bfill_no_nulls():
    df = _mk({"x": [1, 2, 3]})
    result = df.bfill()
    vals = result.head(10).column("x").to_pylist()
    assert vals == [1, 2, 3]


def test_bfill_float():
    df = _mk({"x": pa.array([None, 1.5, None], type=pa.float64())})
    result = df.bfill()
    vals = result.head(10).column("x").to_pylist()
    assert vals == [1.5, 1.5, None]


def test_bfill_multiple_columns():
    df = _mk({
        "a": pa.array([None, 2, None], type=pa.int64()),
        "b": pa.array([10.0, None, 30.0], type=pa.float64()),
    })
    result = df.bfill()
    out = result.head(10)
    assert out.column("a").to_pylist() == [2, 2, None]
    assert out.column("b").to_pylist() == [10.0, 30.0, 30.0]


def test_bfill_columns_subset():
    df = _mk({
        "a": pa.array([None, 2, None], type=pa.int64()),
        "b": pa.array([None, 20, None], type=pa.int64()),
    })
    result = df.bfill(columns=["a"])
    out = result.head(10)
    assert out.column("a").to_pylist() == [2, 2, None]
    assert out.column("b").to_pylist() == [None, 20, None]


def test_bfill_cross_batch(tmp_path):
    df = _mk_multi(
        tmp_path,
        {"x": pa.array([None, None, 3, None], type=pa.int64())},
        chunk_size=1,
    )
    result = df.bfill()
    vals = result.head(10).column("x").to_pylist()
    assert vals == [3, 3, 3, None]


def test_bfill_filtered():
    """v0.9.25i-fix1 (§13.4/268): bfill fills ALL nulls with the
    next non-null value, regardless of gap length. Prior expectation
    [2, None, 5, 5] wrongly assumed bfill(limit=1)-like behavior.

    Filtered rows (v>15 at raw pos 1-4): x = [2, None, None, 5].
    bfill → [2, 5, 5, 5] (both Nones get 5 from pos 3).
    """
    df = _mk({
        "x": pa.array([None, 2, None, None, 5], type=pa.int64()),
        "v": [10, 20, 30, 40, 50],
    })
    filtered = df[df["v"] > 15]
    result = filtered.bfill()
    vals = result.head(10).column("x").to_pylist()
    assert vals == [2, 5, 5, 5]


def test_bfill_empty():
    df = _mk({"x": [1, 2], "v": [10, 20]})
    empty = df[df["v"] > 100]
    result = empty.bfill()
    out = result.head(10)
    assert out.num_rows == 0


def test_bfill_preserves_original_order():
    df = _mk({"x": pa.array([1, None, 3, None, 5],
                             type=pa.int64())})
    result = df.bfill()
    vals = result.head(10).column("x").to_pylist()
    assert vals == [1, 3, 3, 5, 5]


# ============ reindex ============


def test_reindex_all_present(tmp_path):
    src = tmp_path / "src.parquet"
    pq.write_table(
        pa.table({
            "k": ["a", "b", "c"], "v": [10, 20, 30],
        }),
        src,
    )
    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=2,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=tmp_path / "spill",
    ).set_index("k", drop=False)
    result = df.reindex(["a", "b", "c"])
    out = result.head(10)
    assert out.column("v").to_pylist() == [10, 20, 30]


def test_reindex_missing_label_nulls(tmp_path):
    src = tmp_path / "src.parquet"
    pq.write_table(
        pa.table({
            "k": ["a", "b", "c"], "v": [10, 20, 30],
        }),
        src,
    )
    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=2,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=tmp_path / "spill",
    ).set_index("k", drop=False)
    result = df.reindex(["a", "z", "c"])
    out = result.head(10)
    assert out.column("v").to_pylist() == [10, None, 30]


def test_reindex_order_matters(tmp_path):
    src = tmp_path / "src.parquet"
    pq.write_table(
        pa.table({
            "k": ["a", "b", "c"], "v": [10, 20, 30],
        }),
        src,
    )
    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=2,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=tmp_path / "spill",
    ).set_index("k", drop=False)
    result = df.reindex(["c", "a", "b"])
    out = result.head(10)
    assert out.column("v").to_pylist() == [30, 10, 20]


def test_reindex_duplicate_target_first_wins(tmp_path):
    src = tmp_path / "src.parquet"
    pq.write_table(
        pa.table({
            "k": ["a", "b", "c"], "v": [10, 20, 30],
        }),
        src,
    )
    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=2,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=tmp_path / "spill",
    ).set_index("k", drop=False)
    result = df.reindex(["a", "a"])
    out = result.head(10)
    assert out.column("v").to_pylist() == [10, 10]


def test_reindex_duplicate_source_first_wins(tmp_path):
    src = tmp_path / "src.parquet"
    pq.write_table(
        pa.table({
            "k": ["a", "a", "b"], "v": [10, 20, 30],
        }),
        src,
    )
    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=2,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=tmp_path / "spill",
    ).set_index("k", drop=False)
    result = df.reindex(["a"])
    out = result.head(10)
    assert out.column("v").to_pylist() == [10]


def test_reindex_range_index():
    df = _mk({"v": [10, 20, 30, 40, 50]})
    result = df.reindex([0, 2, 4])
    out = result.head(10)
    assert out.column("v").to_pylist() == [10, 30, 50]


def test_reindex_range_index_oob():
    df = _mk({"v": [10, 20, 30]})
    result = df.reindex([0, 100])
    out = result.head(10)
    assert out.column("v").to_pylist() == [10, None]


def test_reindex_filtered_range_index():
    """v0.9.25i-corrected (§13.4/266): filtered RangeIndex — labels
    are ORIGINAL positional ints, not filtered-view enumeration.

    df = [10, 20, 30, 40, 50]; filtered = df[df > 15] = rows at raw
    positions 1, 2, 3, 4 → filtered index labels [1, 2, 3, 4].
    reindex([2, 4]) → rows at labels 2 and 4 → v=[20, 40].
    """
    df = _mk({"v": [10, 20, 30, 40, 50]})
    filtered = df[df["v"] > 15]  # rows at raw positions 1-4
    result = filtered.reindex([2, 4])
    out = result.head(10)
    # Label 2 → raw pos 2 → v=30; label 4 → raw pos 4 → v=50
    # Wait — filtered rows are at raw [1, 2, 3, 4] with labels [1,2,3,4].
    # Label 2 = raw pos 2 = v=30. Label 4 = raw pos 4 = v=50.
    assert out.column("v").to_pylist() == [30, 50]


def test_reindex_empty_labels_raises():
    df = _mk({"v": [10, 20]})
    with pytest.raises(ValueError, match="non-empty"):
        df.reindex([])


def test_reindex_all_missing(tmp_path):
    src = tmp_path / "src.parquet"
    pq.write_table(
        pa.table({
            "k": ["a", "b"], "v": [10, 20],
        }),
        src,
    )
    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=2,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=tmp_path / "spill",
    ).set_index("k", drop=False)
    result = df.reindex(["x", "y", "z"])
    out = result.head(10)
    assert out.column("v").to_pylist() == [None, None, None]


def test_reindex_preserves_index(tmp_path):
    src = tmp_path / "src.parquet"
    pq.write_table(
        pa.table({
            "k": ["a", "b"], "v": [10, 20],
        }),
        src,
    )
    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=2,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=tmp_path / "spill",
    ).set_index("k", drop=False)
    result = df.reindex(["a", "b"])
    assert result.count() == 2
