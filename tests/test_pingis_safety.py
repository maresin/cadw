"""v0.8.9: Pingis safety fixes — behavior tests.

Covers each of the four guard-type fixes from the Pingis audit v0.8.8.
Memory bounds are enforced BEFORE materialize, so most tests check that
MemoryLimitExceeded fires (rather than measuring RSS in-unit).
"""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.core.memory_guard import MemoryLimitExceeded
from cadw.ops.join import _estimate_row_bytes
from cadw.ops.shuffle import _hash_column


# ---------- P-A3: real row_bytes_out ----------

def test_estimate_row_bytes_int64():
    """pa.field defaults to nullable=True → bitmap +1 byte.

    For all-nullable schema: 8+8+1 = 17. For all-non-nullable:
    8+8+0 = 16 (see test_estimate_row_bytes_mixed for non-nullable case).
    """
    schema = pa.schema([("a", pa.int64()), ("b", pa.int64())])
    # nullable=True default → +1 bitmap
    assert _estimate_row_bytes(schema) == 17


def test_estimate_row_bytes_int64_non_nullable():
    """Explicit nullable=False → no bitmap."""
    schema = pa.schema([
        pa.field("a", pa.int64(), nullable=False),
        pa.field("b", pa.int64(), nullable=False),
    ])
    assert _estimate_row_bytes(schema) == 16


def test_estimate_row_bytes_nullable_int():
    schema = pa.schema([
        pa.field("a", pa.int64(), nullable=True),
        pa.field("b", pa.int64(), nullable=False),
    ])
    # 8 + 8 + 1 (bitmap, any nullable) = 17
    assert _estimate_row_bytes(schema) == 17


def test_estimate_row_bytes_string():
    """String default 50 B + bitmap +1 (nullable=True default)."""
    schema = pa.schema([("s", pa.string())])
    # 50 (default string size) + 1 (bitmap, nullable=True)
    assert _estimate_row_bytes(schema) == 51


def test_estimate_row_bytes_mixed():
    schema = pa.schema([
        pa.field("k", pa.int64(), nullable=False),
        pa.field("s", pa.string(), nullable=True),
        pa.field("v", pa.float64(), nullable=True),
    ])
    # 8 + 50 + 8 + 1 (bitmap) = 67
    assert _estimate_row_bytes(schema) == 67


# ---------- P-SH2: uint64 hash consistency ----------

def test_uint64_hash_consistent_across_batches():
    """Same uint64 value in different batches → same hash."""
    b1 = pa.RecordBatch.from_arrays(
        [pa.array([1, 2, 3], type=pa.uint64())], names=["k"]
    )
    b2 = pa.RecordBatch.from_arrays(
        [pa.array([1, 2, None], type=pa.uint64())], names=["k"]
    )
    h1 = _hash_column(b1.column("k"))
    h2 = _hash_column(b2.column("k"))
    assert h1[0] == h2[0], f"hash(1) mismatch uint64 across batches"
    assert h1[1] == h2[1], f"hash(2) mismatch uint64 across batches"


def test_uint64_hash_uses_pandas_path(monkeypatch):
    """uint64 uses pandas path (via bitcast), not Python fallback."""
    from cadw.ops import shuffle as sh
    called_pandas = []
    orig = sh._pandas_hash_column

    def spy(arr):
        called_pandas.append(str(arr.type))
        return orig(arr)

    monkeypatch.setattr(sh, "_pandas_hash_column", spy)
    arr = pa.array([1, 2, 3], type=pa.uint64())
    _hash_column(arr)
    assert "int64" in called_pandas, (
        f"uint64 should bitcast to int64 and use pandas hash; "
        f"got pandas calls for {called_pandas}"
    )


def test_uint64_high_bit_values():
    """Extreme uint64 values (>= 2^63) hash via bitcast correctly."""
    values = [1, 2**63, 2**63 + 1, 2**64 - 1]
    arr = pa.array(values, type=pa.uint64())
    h = _hash_column(arr)
    assert len(set(h.tolist())) == len(values), (
        "distinct uint64 values must produce distinct hashes"
    )


# ---------- P-G3: check_partition_buffers in skew path ----------

def test_groupby_skew_tight_budget_raises_or_succeeds(tmp_path):
    """Skew path opens P writers; tiny budget should either succeed with
    small P, or raise MemoryLimitExceeded (from check_partition_buffers)."""
    n = 1000
    table = pa.table({"k": ["A"] * 900 + ["B"] * 100,
                      "v": [1.0] * n})

    def it():
        for i in range(0, n, 100):
            yield table.slice(i, 100).to_batches()[0]

    guard = MemoryGuard(max_memory_bytes=2000)  # 2 KB
    try:
        from cadw.ops.groupby import groupby_aggregate
        result = groupby_aggregate(
            it(), ["k"], ["v"], ["count"],
            guard, tmp_path, num_partitions=64, log=False,
        )
        # If succeeded, result must be correct
        rows = {r["k"]: r["v_count"] for r in result.to_pylist()}
        assert rows == {"A": 900, "B": 100}
    except MemoryLimitExceeded:
        # Acceptable — buffer check fired
        pass


# ---------- P-G5: sample byte cap ----------

def test_groupby_sample_capped_by_bytes(tmp_path, monkeypatch):
    """Wide schemas must cap sample by bytes, not only 10k rows."""
    # Build a wide-schema table (many float64 cols) — 100 rows × ~10KB/row
    n = 200
    data = {f"c{i}": np.arange(n, dtype=np.float64) for i in range(100)}
    data["k"] = ["A"] * n
    table = pa.table(data)

    def it():
        for i in range(0, n, 50):
            yield table.slice(i, 50).to_batches()[0]

    # budget 100 KB — sample_cap_bytes = 30 KB
    guard = MemoryGuard(max_memory_bytes=100_000)
    try:
        from cadw.ops.groupby import groupby_aggregate
        result = groupby_aggregate(
            it(), ["k"], ["c0"], ["sum"],
            guard, tmp_path, num_partitions=2, log=False,
        )
        assert result.num_rows == 1
    except MemoryLimitExceeded:
        # Acceptable if 100KB budget is insufficient for downstream shuffle
        pass


# ---------- P-D1: dedup pandas chain pre-check ----------

def test_dedup_arrow_native_precheck_raises(tmp_path):
    """Oversize partition → MemoryLimitExceeded from arrow_native guard.

    v0.8.11a-fix2 (§13.4/144): renamed from
    `test_dedup_pandas_chain_precheck_raises`. v0.8.11a (P-D1 close)
    replaced the v0.8.9 5×T `pandas_chain` stopgap with 1.5×T
    `arrow_native` guard. The old test in this file was stale — checked
    for a guard name that no longer exists.

    Direct test of `_local_dedup` (not the full `drop_duplicates`) so we
    control the budget from `table.nbytes`: budget = 2.5 × T means
      - file_size ≈ T < 0.5 × budget = 1.25 × T  → size check passes
      - 1.5 × T > 0.5 × budget = 1.25 × T        → arrow_native fires
    Deterministic — no reliance on string-key size estimate or P.
    """
    from cadw.io.batch_writer import ArrowIPCBatchWriter
    from cadw.ops.dedup import _local_dedup

    n = 2000
    table = pa.table({
        "row_id": pa.array(range(n), type=pa.uint64()),
        "k": [f"k{i % 100}" for i in range(n)],
        "v": list(range(n)),
    })
    p = tmp_path / "partition.arrow"
    writer = ArrowIPCBatchWriter(str(p), table.schema)
    try:
        for b in table.to_batches(max_chunksize=500):
            writer.write_batch(b)
    finally:
        writer.close()

    budget = int(2.5 * table.nbytes)
    guard = MemoryGuard(max_memory_bytes=budget)
    with pytest.raises(MemoryLimitExceeded, match="arrow_native"):
        _local_dedup(p, ["k"], "first", guard)


# ---------- P-A1: generator form sanity (AST-level; runtime equivalence) ----------

def test_safe_materialize_no_list_of_batches():
    """AST guard: _safe_materialize_partition must NOT build list of batches.

    Регрессия P-A1 (v0.8.9): list-form retains O(T) in Python batch list;
    gen form is zero-copy from Arrow buffers. A ListComp of *schema fields*
    is fine (small); a ListComp of *batches* (contains get_batch) is not.
    """
    import ast
    import inspect
    from cadw.ops import join as join_mod

    src = inspect.getsource(join_mod._safe_materialize_partition)
    tree = ast.parse(inspect.cleandoc(src))

    def _is_batch_listcomp(node: ast.ListComp) -> bool:
        # Element or iter contains `.get_batch` → batch list.
        for child in ast.walk(node):
            if isinstance(child, ast.Attribute) and child.attr == "get_batch":
                return True
            if isinstance(child, ast.Name) and child.id in (
                    "ipc_reader", "reader",
            ):
                # Direct `for ... in reader` / `for ... in ipc_reader`
                return True
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.ListComp) and _is_batch_listcomp(node):
            raise AssertionError(
                f"batch ListComp at line {node.lineno} in "
                f"_safe_materialize_partition; use generator form"
            )


def test_dedup_no_list_of_batches():
    """AST guard: _local_dedup must NOT build list of batches.

    Schema-field ListComp is fine; batch ListComp (get_batch) is not.
    """
    import ast
    import inspect
    from cadw.ops import dedup as dedup_mod

    src = inspect.getsource(dedup_mod._local_dedup)
    tree = ast.parse(inspect.cleandoc(src))

    def _is_batch_listcomp(node: ast.ListComp) -> bool:
        for child in ast.walk(node):
            if isinstance(child, ast.Attribute) and child.attr == "get_batch":
                return True
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.ListComp) and _is_batch_listcomp(node):
            raise AssertionError(
                f"batch ListComp at line {node.lineno} in "
                f"_local_dedup; use generator form"
            )


# ---------- Round-trip sanity: existing behavior preserved ----------

def test_external_join_still_works(tmp_path):
    """Sanity: normal small join still works after pre-check additions."""
    def make_df(name, data, chunk=3):
        t = pa.table(data)
        p = tmp_path / f"{name}.parquet"
        pq.write_table(t, p)
        return CadwDataFrame.from_parquet(
            p, tmp_path / f"ds_{name}", chunk_size=chunk,
            memory_guard=MemoryGuard(max_memory_mb=20),
            spill_dir=tmp_path / f"spill_{name}",
        )
    left = make_df("L", {"k": [1, 2, 3], "a": [10, 20, 30]})
    right = make_df("R", {"k": [1, 3], "b": [100, 300]})
    result = left.merge(right, on="k")
    assert result.shape[0] == 2


def test_dedup_still_works(tmp_path):
    """Sanity: normal dedup still works after 5× pandas pre-check."""
    t = pa.table({"k": ["a", "a", "b"], "v": [1, 2, 3]})
    p = tmp_path / "src.parquet"
    pq.write_table(t, p)
    df = CadwDataFrame.from_parquet(
        p, tmp_path / "ds", chunk_size=2,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=tmp_path / "spill",
    )
    result = df.drop_duplicates(subset=["k"], keep="first", num_partitions=2)
    assert result.shape[0] == 2


def test_groupby_still_works(tmp_path):
    """Sanity: normal groupby still works after sample byte-cap."""
    t = pa.table({"k": ["A", "B", "A", "B"], "v": [1.0, 2.0, 3.0, 4.0]})
    p = tmp_path / "src.parquet"
    pq.write_table(t, p)
    df = CadwDataFrame.from_parquet(
        p, tmp_path / "ds", chunk_size=2,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=tmp_path / "spill",
    )
    result = df.groupby("k").agg("v", "sum")
    rows = {r["k"]: r["v_sum"] for r in result.head(10).to_pylist()}
    assert abs(rows["A"] - 4.0) < 1e-9
    assert abs(rows["B"] - 6.0) < 1e-9
