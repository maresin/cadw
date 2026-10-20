"""v0.9.1e: per-group apply E2E (slow)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard


@pytest.mark.slow
def test_apply_e2e_10k_rows(tmp_path):
    n = 10_000
    rng = np.random.default_rng(42)
    data = {
        "g": [f"k{i % 20}" for i in range(n)],
        "v": rng.normal(size=n).tolist(),
    }
    pdf = pd.DataFrame(data)
    src = tmp_path / "src.parquet"
    pq.write_table(pa.table(data), src, row_group_size=1000)

    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=1000,
        memory_guard=MemoryGuard(max_memory_mb=50),
        spill_dir=tmp_path / "spill",
    )
    out = df.groupby("g").apply(lambda g: g["v"].mean())
    out_path = tmp_path / "out.parquet"
    out.write_parquet(out_path)
    got = (pq.read_table(out_path).to_pandas()
           .sort_values("g").reset_index(drop=True))

    expected = pdf.groupby("g")["v"].mean().reset_index()
    expected.columns = ["g", "apply_result"]
    expected = expected.sort_values("g").reset_index(drop=True)

    np.testing.assert_allclose(
        got["apply_result"].to_numpy(),
        expected["apply_result"].to_numpy(),
        rtol=1e-9,
    )
    assert got.shape[0] == 20
