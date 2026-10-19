"""v0.5.2b: P9 (estimate_unique_keys) + P11 (rank 'first' Arrow-native)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard


def make_df(tmp_path, name="L", data=None, chunk=3):
    if data is None:
        data = {"k": ["a", "b", "c", "d", "e"], "v": [1, 2, 3, 4, 5]}
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


# ---------- P9: estimate_unique_keys ----------

def test_estimate_unique_keys_single_col():
    guard = MemoryGuard(max_memory_mb=100)
    sample = pa.RecordBatch.from_arrays(
        [pa.array(["a", "b", "c", "a", "b"])], names=["k"]
    )
    assert guard.estimate_unique_keys(sample, total_rows=1000) == 600


def test_estimate_unique_keys_single_col_all_unique():
    guard = MemoryGuard(max_memory_mb=100)
    sample = pa.RecordBatch.from_arrays(
        [pa.array([1, 2, 3, 4, 5])], names=["k"]
    )
    assert guard.estimate_unique_keys(sample, total_rows=100) == 100


def test_estimate_unique_keys_multi_col():
    guard = MemoryGuard(max_memory_mb=100)
    sample = pa.RecordBatch.from_arrays(
        [pa.array(["a", "a", "b"]), pa.array([1, 2, 1])],
        names=["k1", "k2"],
    )
    assert guard.estimate_unique_keys(sample, total_rows=300) == 300


def test_estimate_unique_keys_empty():
    guard = MemoryGuard(max_memory_mb=100)
    sample = pa.RecordBatch.from_arrays(
        [pa.array([], type=pa.string())], names=["k"]
    )
    assert guard.estimate_unique_keys(sample, total_rows=1000) == 0


def test_estimate_unique_keys_uses_arrow_native():
    """AST guard: fast path uses dictionary_encode, not to_pylist loop."""
    import inspect
    src = inspect.getsource(MemoryGuard.estimate_unique_keys)
    assert "dictionary_encode" in src, (
        "P9 fix: estimate_unique_keys must use pc.dictionary_encode"
    )
    # to_pylist only allowed in except fallback
    assert "count_distinct" in src


# ---------- P11: rank 'first' Arrow-native ----------

def test_rank_first_still_works(tmp_path):
    df = make_df(tmp_path, data={"k": ["b", "a", "c", "a"], "v": [1, 2, 3, 4]})
    result = df.rank(by="k", method="first")
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["k_rank"] for r in rows] == [3, 1, 4, 2]


def test_rank_first_pandas_equivalence(tmp_path):
    n = 100
    rng = np.random.default_rng(7)
    data = {"k": rng.integers(0, 5, size=n).tolist(), "v": list(range(n))}
    df = make_df(tmp_path, data=data, chunk=10)
    result = df.rank(by="k", method="first")
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    got = pq.read_table(out).to_pandas()

    pdf = pd.DataFrame(data)
    expected = pdf["k"].rank(method="first").astype("int64")
    np.testing.assert_array_equal(
        got["k_rank"].to_numpy(), expected.to_numpy()
    )


def test_rank_first_pct(tmp_path):
    df = make_df(tmp_path, data={"k": [10, 20, 30, 40], "v": list(range(4))})
    result = df.rank(by="k", method="first", pct=True)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["k_rank"] for r in rows] == [0.25, 0.5, 0.75, 1.0]


def test_rank_first_uses_native():
    """AST guard: 'first' branch uses Arrow/numpy — no to_pylist.

    P11 goal: no Python loop in 'first' path. Uses np.arange + pc.add,
    equivalent to ordinal position on sorted input.
    """
    import inspect
    from cadw.ops import rank as rank_mod
    src = inspect.getsource(rank_mod._compute_streamable)
    idx_first = src.find('if method == "first"')
    assert idx_first >= 0, "first branch not found"
    idx_continue = src.find('continue', idx_first)
    assert idx_continue > idx_first, "continue not found after first"
    first_branch = src[idx_first:idx_continue]
    assert "to_pylist" not in first_branch, (
        "P11: 'first' branch must not call to_pylist"
    )
    assert ("np.arange" in first_branch
            or "_pc.rank" in first_branch
            or "pc.rank" in first_branch), (
        "P11: 'first' branch must use Arrow/numpy native"
    )


# ---------- Sanity: other methods unchanged ----------

def test_rank_min_unchanged(tmp_path):
    df = make_df(tmp_path, data={"k": ["a", "b", "a", "c", "a"], "v": list(range(5))})
    result = df.rank(by="k", method="min")
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["k_rank"] for r in rows] == [1, 4, 1, 5, 1]


def test_rank_dense_unchanged(tmp_path):
    df = make_df(tmp_path, data={"k": ["b", "a", "c", "a", "b"], "v": list(range(5))})
    result = df.rank(by="k", method="dense")
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["k_rank"] for r in rows] == [2, 1, 3, 1, 2]


def test_rank_max_unchanged(tmp_path):
    df = make_df(tmp_path, data={"k": ["a", "b", "a", "c", "a"], "v": list(range(5))})
    result = df.rank(by="k", method="max")
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["k_rank"] for r in rows] == [3, 4, 3, 5, 3]
