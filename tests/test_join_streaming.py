"""P2 fix (v0.5.1): hash_join_streaming writes output to disk.

Verifies:
  - hash_join_streaming returns (schema, path) with valid IPC.
  - Output batches written incrementally (not accumulated).
  - hash_join (compat) still returns pa.Table.
  - CadwDataFrame.merge uses streaming path.
  - _process_partition_pair is a generator (yield).
"""
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.io.batch_writer import ArrowIPCBatchWriter
from cadw.ops import join as join_mod
from cadw.ops.join import hash_join, hash_join_streaming


def make_df(tmp_path, name, data, chunk=3):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


# ---------- AST: _process_partition_pair is generator ----------

def test_process_partition_pair_is_generator():
    """_process_partition_pair must yield (not return list)."""
    import inspect
    src = inspect.getsource(join_mod._process_partition_pair)
    assert "yield " in src or "yield from" in src, (
        "_process_partition_pair must be a generator (P2 fix). "
        "See DECISIONS §13.4/63."
    )


# ---------- hash_join_streaming writes to disk ----------

def test_hash_join_streaming_returns_schema_and_path(tmp_path):
    left = make_df(tmp_path, "L", {"k": [1, 2, 3, 4], "a": [10, 20, 30, 40]})
    right = make_df(tmp_path, "R", {"k": [1, 3], "b": [100, 300]})
    schema, path = hash_join_streaming(
        left._iter_filtered_batches(),
        right._iter_filtered_batches(),
        left._dataset.schema, right._dataset.schema, ["k"],
        MemoryGuard(max_memory_mb=20),
        tmp_path / "join_spill",
        how="inner", num_partitions=2,
    )
    assert isinstance(schema, pa.Schema)
    assert isinstance(path, Path)
    assert path.exists()
    # Read back to verify
    reader = ipc.open_file(str(path))
    total = sum(
        reader.get_batch(i).num_rows
        for i in range(reader.num_record_batches)
    )
    assert total == 2  # k=1, k=3


def test_hash_join_streaming_writes_incrementally(tmp_path, monkeypatch):
    """hash_join_streaming writes output incrementally (not accumulated).

    Note: hash_shuffle also uses ArrowIPCBatchWriter (for partition files).
    Spy filters by target path so we only count writes to join_output.arrow.
    """
    write_calls = []
    orig_init = ArrowIPCBatchWriter.__init__
    orig_write = ArrowIPCBatchWriter.write_batch

    def patched_init(self, path, schema, *args, **kwargs):
        self._test_path = str(path)
        return orig_init(self, path, schema, *args, **kwargs)

    def spy(self, batch):
        # Only count writes to the final join_output.arrow (P2 target).
        if "join_output.arrow" in getattr(self, "_test_path", ""):
            write_calls.append(batch.num_rows)
        return orig_write(self, batch)

    monkeypatch.setattr(ArrowIPCBatchWriter, "__init__", patched_init)
    monkeypatch.setattr(ArrowIPCBatchWriter, "write_batch", spy)

    left = make_df(tmp_path, "L", {"k": list(range(100)), "a": list(range(100))})
    right = make_df(tmp_path, "R", {"k": list(range(100)), "b": list(range(100))})
    schema, path = hash_join_streaming(
        left._iter_filtered_batches(),
        right._iter_filtered_batches(),
        left._dataset.schema, right._dataset.schema, ["k"],
        MemoryGuard(max_memory_mb=20),
        tmp_path / "join_spill", how="inner", num_partitions=4,
    )
    assert write_calls, "hash_join_streaming must write to join_output.arrow"
    assert sum(write_calls) == 100, (
        f"total rows written to join_output.arrow: {sum(write_calls)} (expected 100)"
    )
    assert len(write_calls) > 1, (
        f"must write incrementally (>1 batch); got {len(write_calls)} call(s)"
    )


# ---------- backward compat: hash_join returns Table ----------

def test_hash_join_returns_table(tmp_path):
    left = make_df(tmp_path, "L", {"k": [1, 2, 3], "a": [10, 20, 30]})
    right = make_df(tmp_path, "R", {"k": [1, 3], "b": [100, 300]})
    result = hash_join(
        left._iter_filtered_batches(),
        right._iter_filtered_batches(),
        left._dataset.schema, right._dataset.schema, ["k"],
        MemoryGuard(max_memory_mb=20),
        tmp_path / "join_spill", how="inner", num_partitions=2,
    )
    assert isinstance(result, pa.Table)
    assert result.num_rows == 2


# ---------- merge uses streaming ----------

def test_merge_uses_streaming(tmp_path, monkeypatch):
    """CadwDataFrame.merge must call hash_join_streaming, not hash_join."""
    from cadw import dataframe as df_mod

    called_streaming = []
    called_collect = []
    orig_streaming = df_mod.hash_join_streaming
    orig_collect = df_mod.hash_join

    def spy_streaming(*a, **kw):
        called_streaming.append(True)
        return orig_streaming(*a, **kw)

    def spy_collect(*a, **kw):
        called_collect.append(True)
        return orig_collect(*a, **kw)

    monkeypatch.setattr(df_mod, "hash_join_streaming", spy_streaming)
    monkeypatch.setattr(df_mod, "hash_join", spy_collect)

    left = make_df(tmp_path, "L", {"k": [1, 2, 3], "a": [10, 20, 30]})
    right = make_df(tmp_path, "R", {"k": [1, 3], "b": [100, 300]})
    left.merge(right, on="k")

    assert called_streaming, "merge must use hash_join_streaming (P2)"
    assert not called_collect, "merge must NOT use hash_join (P2)"


def test_merge_still_works(tmp_path):
    left = make_df(tmp_path, "L", {"k": [1, 2, 3, 4], "a": ["x", "y", "z", "w"]})
    right = make_df(tmp_path, "R", {"k": [1, 3, 5], "b": ["p", "q", "r"]})
    result = left.merge(right, on="k")
    head = result.head(20)
    assert head.num_rows == 2


def test_merge_join_output_file_cleaned_up(tmp_path):
    """merge should delete intermediate join_output.arrow after Dataset.from_batches."""
    left = make_df(tmp_path, "L", {"k": [1, 2, 3], "a": [10, 20, 30]})
    right = make_df(tmp_path, "R", {"k": [1, 3], "b": [100, 300]})
    left.merge(right, on="k")
    # Intermediate file should be gone
    join_out = left._spill_dir / "join" / "join_output.arrow"
    assert not join_out.exists(), (
        f"intermediate {join_out} should be cleaned up after merge"
    )


# ---------- left join via streaming ----------

def test_merge_left_streaming(tmp_path):
    left = make_df(tmp_path, "L", {"k": [1, 2, 3, 4], "a": ["w", "x", "y", "z"]})
    right = make_df(tmp_path, "R", {"k": [1, 3, 5], "b": ["p", "q", "r"]})
    result = left.merge(right, on="k", how="left")
    head = result.head(20)
    assert head.num_rows == 4  # all left preserved


# ---------- empty inputs ----------

def test_hash_join_streaming_empty_left(tmp_path):
    left = make_df(tmp_path, "L", {"k": pa.array([], type=pa.int64()), "a": pa.array([], type=pa.int64())})
    right = make_df(tmp_path, "R", {"k": [1, 2], "b": [100, 200]})
    schema, path = hash_join_streaming(
        left._iter_filtered_batches(),
        right._iter_filtered_batches(),
        left._dataset.schema, right._dataset.schema, ["k"],
        MemoryGuard(max_memory_mb=20),
        tmp_path / "join_spill", how="inner", num_partitions=2,
    )
    assert path.exists()
    reader = ipc.open_file(str(path))
    assert reader.num_record_batches == 0  # no output
