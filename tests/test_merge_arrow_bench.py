"""v0.9.23c: external_sort timing at N=500k / 8 cols under 32 MB.

Pre-fix (AUDIT_FULL_v0922 §6.6): 24.4s (heapq per-row).
Post-fix (v0.9.23c): target <5s (Arrow batched merge).
"""
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard


@pytest.mark.slow
def test_external_sort_n500k_8cols_32mb_under_5s(tmp_path):
    """N=500k, 8 cols, 32 MB budget: wall < 5s (pre-fix was 24s)."""
    n = 500_000
    rng = np.random.default_rng(0)
    data = {f"x{i}": rng.normal(size=n).tolist() for i in range(7)}
    data["k"] = rng.integers(0, 1000, size=n).tolist()
    t = pa.table(data)
    src = tmp_path / "src.parquet"
    pq.write_table(t, src, row_group_size=10_000)

    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=10_000,
        memory_guard=MemoryGuard(max_memory_mb=32),
        spill_dir=tmp_path / "spill",
    )
    t0 = time.time()
    res = df.sort_values("k")
    out = tmp_path / "out.parquet"
    res.write_parquet(out)
    elapsed = time.time() - t0
    assert elapsed < 5.0, (
        f"external_sort N=500k/8cols/32MB: {elapsed:.2f}s "
        f"(pre-fix was 24.4s; target <5s). AUDIT_FULL_v0922 §6.6."
    )
    # Correctness
    got = pq.read_table(out).column("k").to_pylist()
    expected = sorted(t.column("k").to_pylist())
    assert got == expected
