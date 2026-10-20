# v0.9.3c: P-D2 dedup type-based estimator (S2 fix).
# Prior 1.5x bound was 7.5x underestimate for string keys at K=N.

from pathlib import Path

import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from cadw import CadwDataFrame, MemoryGuard
from cadw.core.memory_guard import MemoryLimitExceeded


def make_df(tmp_path, data, chunk=200, name="L", mem_mb=20):
    t = pa.table(data)
    p = Path(tmp_path) / (name + ".parquet")
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / ("ds_" + name), chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
        spill_dir=Path(tmp_path) / ("spill_" + name),
    )


def test_dedup_string_kN_raises_tight_budget(tmp_path):
    # K=N strings. Partition ~20 KB. 12x = 240 KB peak estimate.
    # Budget 200 KB -> state_limit 100 KB. check_state(12x) raises.
    n = 1000
    df = make_df(tmp_path, {
        "k": ["k%04d" % i for i in range(n)],
        "v": list(range(n)),
    }, chunk=200, mem_mb=20)
    df._memory_guard = MemoryGuard(max_memory_bytes=200_000)
    with pytest.raises(MemoryLimitExceeded, match="arrow_native"):
        df.drop_duplicates(subset=["k"], keep="first", num_partitions=1)


def test_dedup_string_kN_ample_budget_passes(tmp_path):
    # Same data, ample budget (10 MB). All rows unique -> all kept.
    n = 1000
    df = make_df(tmp_path, {
        "k": ["k%04d" % i for i in range(n)],
        "v": list(range(n)),
    }, chunk=200, mem_mb=10)
    result = df.drop_duplicates(subset=["k"], keep="first", num_partitions=1)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert len(rows) == n


def test_dedup_int_kN_ample_budget_passes(tmp_path):
    # K=N int64. All unique -> all kept.
    n = 1000
    df = make_df(tmp_path, {
        "k": list(range(n)),
        "v": list(range(n)),
    }, chunk=200, mem_mb=10)
    result = df.drop_duplicates(subset=["k"], keep="first", num_partitions=1)
    assert result.count() == n


def test_dedup_uses_12x_for_string_ast():
    # AST: _dedup_peak_mult set to 12 for string subset branch.
    import inspect
    from cadw.ops import dedup as dedup_mod
    src = inspect.getsource(dedup_mod._local_dedup)
    assert "_dedup_peak_mult = 12 if _has_string_subset" in src
    assert "else 2.5" in src


def test_dedup_uses_2_5x_for_numeric_ast():
    # AST: _dedup_peak_mult * table.nbytes drives check_state.
    import inspect
    from cadw.ops import dedup as dedup_mod
    src = inspect.getsource(dedup_mod._local_dedup)
    assert "_dedup_peak_mult * table.nbytes" in src
    assert "_dedup_peak_mult = 12 if" in src
    assert "else 2.5" in src


def test_dedup_type_check_present():
    # AST: _local_dedup checks pa.types.is_string for subset columns.
    import inspect
    from cadw.ops import dedup as dedup_mod
    src = inspect.getsource(dedup_mod._local_dedup)
    assert "is_string" in src
    assert "subset_cols" in src


def test_dedup_multiplier_variable():
    # AST: _dedup_peak_mult variable drives check_state.
    import inspect
    from cadw.ops import dedup as dedup_mod
    src = inspect.getsource(dedup_mod._local_dedup)
    assert "_dedup_peak_mult" in src

