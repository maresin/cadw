"""v0.9.23a: rank extra_factor fix (AUDIT_FULL_v0922 §3.7, item 7.A.11).

Derived factor per prior-audit 3.5*K pylist (28 B/value vs 8 B/value):
  - 1 pylist column: 3.5  (pass1)
  - 2 pylist columns: 7.0 (streamable, pass2)
"""
import inspect

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard


def make_df(tmp_path, data, chunk=5, name="L"):
    t = pa.table(data)
    p = tmp_path / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, tmp_path / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=tmp_path / f"spill_{name}",
    )


# ============ AST checks: factors ============

def test_streamable_extra_factor_is_7():
    """_compute_streamable: 2 pylist (by + row_id) -> 3.5*2 = 7.0."""
    from cadw.ops.rank import _compute_streamable
    src = inspect.getsource(_compute_streamable)
    assert 'name="rank.streamable.input", extra_factor=7.0' in src, (
        "rank.streamable.input extra_factor must be 7.0 "
        "(AUDIT_FULL_v0922 §3.7)"
    )


def test_pass1_extra_factor_is_3_5():
    """_pass1_groups_to_disk: 1 pylist (by) -> 3.5."""
    from cadw.ops.rank import _pass1_groups_to_disk
    src = inspect.getsource(_pass1_groups_to_disk)
    assert 'name="rank.pass1.input", extra_factor=3.5' in src, (
        "rank.pass1.input extra_factor must be 3.5"
    )


def test_pass2_extra_factor_is_7():
    """_pass2_ranks_to_disk: 2 pylist (by + row_id) -> 3.5*2 = 7.0."""
    from cadw.ops.rank import _pass2_ranks_to_disk
    src = inspect.getsource(_pass2_ranks_to_disk)
    assert 'name="rank.pass2.input", extra_factor=7.0' in src, (
        "rank.pass2.input extra_factor must be 7.0"
    )


# ============ Regression: rank still correct ============

def test_rank_nullfree_still_correct(tmp_path):
    """Null-free rank(by='k', method='first') unchanged by factor bump."""
    df = make_df(tmp_path, {
        "k": [1.0, 3.0, 2.0, 4.0, 5.0],
        "v": [10, 30, 20, 40, 50],
    })
    result = df.rank(by="k", method="first")
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    # sort by k: row_id 0(v=10)->1, 2(v=20)->2, 1(v=30)->3, 3(v=40)->4, 4(v=50)->5
    # original order: [1, 3, 2, 4, 5]
    assert [r["k_rank"] for r in rows] == [1, 3, 2, 4, 5]


def test_rank_with_null_still_correct(tmp_path):
    """Null-path rank(by='k', method='first') still correct."""
    df = make_df(tmp_path, {
        "k": pa.array([1.0, None, 3.0, 2.0, None, 4.0], type=pa.float64()),
        "v": list(range(6)),
    })
    result = df.rank(by="k", method="first")
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    # Nulls sort last. Non-null ranks: 1.0->1, 2.0->2, 3.0->3, 4.0->4;
    # nulls: 5, 6 (by row_id tie-break).
    # Original order (k=[1,None,3,2,None,4]):
    # row 0(k=1)->1, row 1(None)->5, row 2(k=3)->3,
    # row 3(k=2)->2, row 4(None)->6, row 5(k=4)->4
    assert [r["k_rank"] for r in rows] == [1, 5, 3, 2, 6, 4]
