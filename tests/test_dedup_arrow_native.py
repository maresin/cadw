"""v0.8.11a: Arrow-native dedup behavior tests.

Prototype-validated (scripts/proto_dedup_arrow.py, Path C adopted):
  peak 0.29× pandas, correctness A==B==C across keep × selectivity.
"""
from pathlib import Path

import numpy as np
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


def _norm(rows: list, key):
    return sorted(rows, key=lambda r: (r[key] is None, r[key]))


# ============ correctness (was in test_drop_duplicates.py, extended) ============

def test_dedup_first_basic(tmp_path):
    df = make_df(tmp_path, "L", {
        "k": ["a", "a", "b", "b", "c"],
        "v": [1, 2, 3, 4, 5],
    })
    result = df.drop_duplicates(subset=["k"], keep="first", num_partitions=4)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    by_k = {r["k"]: r["v"] for r in rows}
    assert by_k == {"a": 1, "b": 3, "c": 5}


def test_dedup_last_basic(tmp_path):
    df = make_df(tmp_path, "L", {
        "k": ["a", "a", "b", "b", "c"],
        "v": [1, 2, 3, 4, 5],
    })
    result = df.drop_duplicates(subset=["k"], keep="last", num_partitions=4)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    by_k = {r["k"]: r["v"] for r in pq.read_table(out).to_pylist()}
    assert by_k == {"a": 2, "b": 4, "c": 5}


def test_dedup_false_basic(tmp_path):
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


def test_dedup_full_row(tmp_path):
    df = make_df(tmp_path, "L", {"k": ["a", "a", "b"], "v": [1, 1, 2]})
    result = df.drop_duplicates(keep="first", num_partitions=4)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = _norm(pq.read_table(out).to_pylist(), "k")
    assert rows == [{"k": "a", "v": 1}, {"k": "b", "v": 2}]


def test_dedup_multi_subset(tmp_path):
    df = make_df(tmp_path, "L", {
        "k1": ["a", "a", "a", "b"],
        "k2": [1, 1, 2, 1],
        "v": [10, 20, 30, 40],
    })
    result = df.drop_duplicates(
        subset=["k1", "k2"], keep="first", num_partitions=4,
    )
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert {(r["k1"], r["k2"], r["v"]) for r in rows} == {
        ("a", 1, 10), ("a", 2, 30), ("b", 1, 40),
    }


def test_dedup_nulls_in_subset(tmp_path):
    """Null в subset — своя группа (несколько null-строк → одна)."""
    df = make_df(tmp_path, "L", {
        "k": ["a", None, "b", None, "a"],
        "v": [1, 2, 3, 4, 5],
    })
    result = df.drop_duplicates(subset=["k"], keep="first", num_partitions=4)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    # Distinct keys in result: a, b, None — 3 rows.
    assert len(rows) == 3
    k_set = {r["k"] for r in rows}
    assert k_set == {"a", "b", None}


def test_dedup_nulls_in_values(tmp_path):
    df = make_df(tmp_path, "L", {
        "k": ["a", "a", "b"],
        "v": [1, None, 2],
    })
    result = df.drop_duplicates(subset=["k"], keep="first", num_partitions=4)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    by_k = {r["k"]: r["v"] for r in pq.read_table(out).to_pylist()}
    assert by_k == {"a": 1, "b": 2}


def test_dedup_all_unique(tmp_path):
    n = 100
    df = make_df(tmp_path, "L", {
        "k": [f"k{i}" for i in range(n)],
        "v": list(range(n)),
    })
    result = df.drop_duplicates(subset=["k"], keep="first", num_partitions=8)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    assert pq.read_table(out).num_rows == n


def test_dedup_all_same_key(tmp_path):
    n = 10
    df = make_df(tmp_path, "L", {"k": ["X"] * n, "v": list(range(n))})
    result = df.drop_duplicates(subset=["k"], keep="first", num_partitions=4)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert len(rows) == 1 and rows[0]["v"] == 0


def test_dedup_int_key(tmp_path):
    n = 50
    df = make_df(tmp_path, "L", {
        "k": [i % 10 for i in range(n)],
        "v": list(range(n)),
    })
    result = df.drop_duplicates(subset=["k"], keep="first", num_partitions=4)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert len(rows) == 10


# ============ pandas-equivalence (set-of-tuples) ============

def test_dedup_pandas_equivalence(tmp_path):
    import pandas as pd
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

    got_set = set(map(tuple, got[["k", "v"]].itertuples(index=False, name=None)))
    exp_set = set(
        map(tuple, expected[["k", "v"]].itertuples(index=False, name=None))
    )
    assert got_set == exp_set


# ============ edge: empty partition ============

def test_dedup_empty_input_raises(tmp_path):
    empty_batches: list = []
    mg = MemoryGuard(max_memory_mb=10)
    with pytest.raises(ValueError, match="empty input"):
        ops_drop_duplicates(
            iter(empty_batches),
            subset=["k"], keep="first",
            memory_guard=mg, spill_dir=tmp_path / "spill",
        )


# ============ memory bounds ============

def test_dedup_arrow_native_bound_fires(tmp_path):
    """Oversize partition → MemoryLimitExceeded before materialize."""
    df = make_df(
        tmp_path, "L",
        {"k": ["HOT"] * 1000, "v": list(range(1000))},
        chunk=200, mem_mb=1,
    )
    df._memory_guard = MemoryGuard(max_memory_bytes=1000)
    with pytest.raises(MemoryLimitExceeded):
        df.drop_duplicates(subset=["k"], keep="first", num_partitions=1)


def test_dedup_removes_stopgap_5x():
    """AST: `_local_dedup` uses Arrow-native guard name (v0.8.11a).

    v0.8.11a-fix1 (§13.4/143): check specific OLD guard NAME
    `dedup.pandas_chain`, NOT the arithmetic pattern `5 * table.nbytes` —
    the latter is a substring of `1.5 * table.nbytes` (the CORRECT new
    code), so `"5 * table.nbytes" not in src` fails even after fix.
    Substring `in` doesn't respect token boundaries.
    """
    import inspect
    from cadw.ops import dedup as dedup_mod
    src = inspect.getsource(dedup_mod._local_dedup)
    assert "dedup.pandas_chain" not in src, (
        "v0.8.11a: old pandas_chain stopgap name must be removed"
    )
    assert "dedup.arrow_native" in src, (
        "v0.8.11a: expect `dedup.arrow_native` guard name"
    )
    assert "1.5 * table.nbytes" in src, (
        "v0.8.11a: expect `1.5 * table.nbytes` Arrow-native bound"
    )


# ============ regression: chained ops ============

def test_dedup_then_sort(tmp_path):
    df = make_df(tmp_path, "L", {
        "k": ["c", "a", "b", "a", "c"],
        "v": [1, 2, 3, 4, 5],
    })
    result = df.drop_duplicates(subset=["k"], keep="last", num_partitions=4)
    sorted_result = result.sort_values("k", ascending=True)
    out = tmp_path / "o.parquet"
    sorted_result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert rows == [
        {"k": "a", "v": 4},
        {"k": "b", "v": 3},
        {"k": "c", "v": 5},
    ]


def test_dedup_after_filter(tmp_path):
    df = make_df(tmp_path, "L", {
        "k": ["a", "a", "b", "b", "c"],
        "v": [1, 2, 3, 4, 5],
    })
    filtered = df[df["v"] > 1]
    result = filtered.drop_duplicates(
        subset=["k"], keep="first", num_partitions=4,
    )
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = _norm(pq.read_table(out).to_pylist(), "k")
    by_k = {r["k"]: r["v"] for r in rows}
    assert by_k == {"a": 2, "b": 3, "c": 5}


# ============ v0.8.11a-fix3: schema-order robustness ============

def test_dedup_schema_order_agnostic(tmp_path):
    """_local_dedup works regardless of row_id position in file schema.

    v0.8.11a-fix3 (§13.4/145): production hash_shuffle always puts
    row_id last, but a partition from any other IPC writer might not.
    _local_dedup builds explicit select_schema (out_fields + [row_id])
    so file column order doesn't matter.

    Test writes partition with row_id FIRST (opposite of hash_shuffle).
    """
    from cadw.io.batch_writer import ArrowIPCBatchWriter
    from cadw.ops.dedup import _local_dedup

    table = pa.table({
        "row_id": pa.array(range(10), type=pa.uint64()),  # FIRST
        "k": ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"],
        "v": list(range(10)),
    })
    p = tmp_path / "partition_rid_first.arrow"
    w = ArrowIPCBatchWriter(str(p), table.schema)
    try:
        for b in table.to_batches(max_chunksize=5):
            w.write_batch(b)
    finally:
        w.close()

    guard = MemoryGuard(max_memory_mb=10)
    result = _local_dedup(p, ["k"], "first", guard)

    assert result.num_rows == 5
    assert "row_id" not in result.schema.names
    assert set(result.column("k").to_pylist()) == {"a", "b", "c", "d", "e"}
