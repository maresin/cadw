"""v0.8.11b: strict join enforcement (budget >= 160 MB → raise)."""
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.core.memory_guard import MemoryLimitExceeded


def make_df(tmp_path, name, data, chunk=3, mem_mb=200):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def test_strict_budget_threshold_constant():
    """Threshold = 4 × overhead (160 MB)."""
    from cadw.ops.join import (
        _JOIN_FIXED_OVERHEAD_BYTES,
        _JOIN_STRICT_BUDGET_THRESHOLD_BYTES,
    )
    assert _JOIN_STRICT_BUDGET_THRESHOLD_BYTES == 4 * _JOIN_FIXED_OVERHEAD_BYTES
    assert _JOIN_STRICT_BUDGET_THRESHOLD_BYTES == 160 * 1024 * 1024


def test_state_limit_bytes_property():
    """MemoryGuard exposes state_limit_bytes = available × state_safety."""
    g = MemoryGuard(max_memory_mb=200)  # state_safety=0.5 default
    assert g.state_limit_bytes == 100 * 1024 * 1024


def test_strict_mode_raises_with_large_budget(tmp_path):
    """Budget 200 MB (>= 160 threshold), predicted_output=28.8 MB
    → total=184 MB > state_limit=100 MB → MemoryLimitExceeded.

    Data: 30K rows each side, K_true=100 per side (300 dupes/key).
    sample=3000 (chunk) > K_true → K_sample=100, K_est=100 × 30000/3000=1000.
    avg_matches = 30000 × 30000 / 1000 / 30000 = 30.
    predicted = 30000 × 30 × 32 (row_bytes_out) = 28.8 MB.
    """
    n = 30000
    k_true = 100
    left = make_df(
        tmp_path, "L",
        {"k": [i % k_true for i in range(n)], "a": list(range(n))},
        chunk=3000, mem_mb=200,
    )
    right = make_df(
        tmp_path, "R",
        {"k": [i % k_true for i in range(n)], "b": list(range(n))},
        chunk=3000, mem_mb=200,
    )
    with pytest.raises(MemoryLimitExceeded, match="whole_operation_rss"):
        left.merge(right, on="k")


def test_strict_mode_passes_small_join(tmp_path):
    """Budget 200 MB (strict mode), 1000 × 1000 PK join → predicted tiny,
    check passes, no raise."""
    left = make_df(
        tmp_path, "L",
        {"k": list(range(1000)), "a": list(range(1000))},
        chunk=100, mem_mb=200,
    )
    right = make_df(
        tmp_path, "R",
        {"k": list(range(1000)), "b": list(range(1000))},
        chunk=100, mem_mb=200,
    )
    result = left.merge(right, on="k")
    assert result.shape[0] == 1000
