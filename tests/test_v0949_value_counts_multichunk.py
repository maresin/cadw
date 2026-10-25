"""v0.9.49 BUG-1 reproducer: value_counts crashes on multi-chunk frames.

Copied verbatim from .pingis/audits/v0.9.48_bugs.md (BUG-1 section).
Fails at HEAD 4e7e614 with ArrowInvalid schema mismatch;
passes after v0.9.49 fix.
"""
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from cadw import CadwDataFrame, MemoryGuard


def test_value_counts_multichunk_reproducer(tmp_path):
    n = 100_000
    rng = np.random.default_rng(0)
    tab = pa.table({
        **{f"c{i}": pa.array(rng.normal(size=n)) for i in range(10)},
        "cat": pa.array([f"g{j % 20}" for j in rng.integers(0, 20, size=n)]),
    })
    p = str(tmp_path / "src.parquet")
    pq.write_table(tab, p)
    df = CadwDataFrame.from_parquet(
        p, str(tmp_path / "ds"), chunk_size=50_000,          # <-- MULTI-CHUNK
        memory_guard=MemoryGuard(max_memory_mb=500),
        spill_dir=str(tmp_path / "spill"),
    )
    out = df.value_counts("cat")                        # raises ArrowInvalid at HEAD
    assert out.num_rows == 20
    assert sum(out.column("count").to_pylist()) == n
