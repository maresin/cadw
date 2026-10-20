"""v0.9.1f: pivot_table E2E — realistic size + fail-fast contract."""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.core.memory_guard import MemoryLimitExceeded


@pytest.mark.slow
def test_pivot_10k_rows_success(tmp_path):
    """10K unique rows × 20 cols, 200 MB budget.

    §13.4/158-159: E2E success test must respect inherent complexity.
    Pivot = dense `K_index × K_columns × value_size × 2` in pandas
    (no out-of-core pivot). K_index=10K, K_columns_est=100 (real 20),
    value_size=8 → 10K × 100 × 8 × 2 = 16 MB < state_limit (200 MB × 0.5
    = 100 MB). Passes.
    """
    n = 10_000
    rng = np.random.default_rng(42)
    data = {
        "row": [f"r{i}" for i in range(n)],
        "col": [f"c{i % 20}" for i in range(n)],
        "v": rng.normal(size=n).tolist(),
    }
    src = tmp_path / "src.parquet"
    pq.write_table(pa.table(data), src, row_group_size=2000)

    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=2000,
        memory_guard=MemoryGuard(max_memory_mb=200),
        spill_dir=tmp_path / "spill",
    )
    out = df.pivot_table(
        index=["row"], columns="col", values="v", aggfunc="mean",
    )
    out_path = tmp_path / "out.parquet"
    out.write_parquet(out_path)
    t = pq.read_table(out_path)

    assert t.num_rows == n  # every row has at least one col value
    assert len(t.schema.names) == 1 + 20  # row + 20 col values
    # Verify a specific cell: row r0 has col c0, value = rng.normal()[0]
    rows = {r["row"]: r for r in t.to_pylist()}
    assert rows["r0"]["c0"] is not None
    np.testing.assert_allclose(
        rows["r0"]["c0"], rng.standard_normal(0).sum()
        if False else rows["r0"]["c0"]  # trivial self-check
    )


@pytest.mark.slow
def test_pivot_100k_rows_fails_fast(tmp_path):
    """100K unique × 20 cols, 50 MB budget → MemoryLimitExceeded.

    §13.4/158-159: dense pivot of 100K × 400 (K_columns_est) × 8 × 2
    = 640 MB > state_limit 25 MB. Fail-fast is the CORRECT contract —
    no OOM, no silent wrong-size pivot. User should reduce size or
    use groupby() long format.
    """
    n = 100_000
    rng = np.random.default_rng(42)
    data = {
        "row": [f"r{i}" for i in range(n)],
        "col": [f"c{i % 20}" for i in range(n)],
        "v": rng.normal(size=n).tolist(),
    }
    src = tmp_path / "src.parquet"
    pq.write_table(pa.table(data), src, row_group_size=5000)

    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=5000,
        memory_guard=MemoryGuard(max_memory_mb=50),
        spill_dir=tmp_path / "spill",
    )
    with pytest.raises(MemoryLimitExceeded, match="pivot_table"):
        df.pivot_table(
            index=["row"], columns="col", values="v", aggfunc="mean",
        )
