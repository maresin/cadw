"""v0.9.34 (B2): encode_categorical in-loop check_state guard.

Already-fixed-at-baseline proof: commit 6fa5e49 (v0.9.25c) added
in-loop `check_state(len(uniques)*180)` inside Pass 1. Before that,
@ 10M unique strings Python set grew to ~600 MB invisible to
MemoryGuard (per-batch check only saw ~130 KB). This test asserts
the guard fires early on high-cardinality input.
"""
import inspect
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard, MemoryLimitExceeded


def _mk(tmp_path, n=100_000, chunk=1000):
    data = {"cat": [f"s{i}" for i in range(n)], "v": list(range(n))}
    t = pa.table(data)
    p = Path(tmp_path) / "hi.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / "ds", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=2),
        spill_dir=Path(tmp_path) / "spill",
    )


def test_encode_uses_inloop_check_state():
    from cadw.dataframe._ml_bindings import _MLBindingsMixin
    src = inspect.getsource(_MLBindingsMixin.encode_categorical)
    assert "check_state" in src
    assert "len(uniques)" in src


def test_encode_high_card_raises_early(tmp_path):
    df = _mk(tmp_path, n=100_000)
    with pytest.raises(MemoryLimitExceeded):
        df.encode_categorical("cat")
