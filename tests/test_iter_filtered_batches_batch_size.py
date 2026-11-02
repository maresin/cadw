"""v0.9.106 (DEC-023): batch_size propagation through _iter_filtered_batches."""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame, MemoryGuard


def _make_df(n: int = 10_000, d: int = 3):
    rng = np.random.RandomState(42)
    cols = {f"x{j}": rng.randn(n) for j in range(d)}
    cols["y"] = rng.randn(n)
    return CadwDataFrame.from_table(
        pa.table(cols),
        memory_guard=MemoryGuard(max_memory_mb=100),
    )


def test_default_unchanged():
    """batch_size=None (default) -> behavior identical to prior."""
    df = _make_df(10_000)
    batches = list(df._iter_filtered_batches())
    assert len(batches) == 1
    assert batches[0].num_rows == 10_000


def test_batch_size_propagates():
    """batch_size=1000 -> 10 batches."""
    df = _make_df(10_000)
    batches = list(df._iter_filtered_batches(batch_size=1000))
    assert len(batches) == 10
    assert all(b.num_rows == 1000 for b in batches)
    assert sum(b.num_rows for b in batches) == 10_000


def test_batch_size_remainder():
    """n=10_500, batch=1000 -> 10 full + 1 remainder (500)."""
    df = _make_df(10_500)
    batches = list(df._iter_filtered_batches(batch_size=1000))
    assert len(batches) == 11
    sizes = [b.num_rows for b in batches]
    assert sizes == [1000] * 10 + [500]
    assert sum(sizes) == 10_500


def test_batch_size_larger_than_n():
    """batch=20000 > n=10000 -> single batch (n rows)."""
    df = _make_df(10_000)
    batches = list(df._iter_filtered_batches(batch_size=20_000))
    assert len(batches) == 1
    assert batches[0].num_rows == 10_000


def test_batch_size_validation():
    """batch_size < 1 raises."""
    df = _make_df(100)
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        list(df._iter_filtered_batches(batch_size=0))
