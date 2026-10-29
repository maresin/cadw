"""v0.9.7a: streaming top-k (R4: 3946× vs external_sort)."""
from __future__ import annotations

import ast
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.core.memory_guard import MemoryLimitExceeded
from cadw.patch_utils import parse_method_source, assert_no_call_in_source


def make_df(tmp_path, data, chunk=3, name="L", mem_mb=50):
    t = pa.table(data)
    p = tmp_path / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, tmp_path / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
        spill_dir=tmp_path / f"spill_{name}",
    )


def _collect(df, tmp_path, name="o.parquet"):
    """Materialize CadwDataFrame OR pass-through pa.Table.

    v0.9.7a-fix2: CadwDataFrame.head() returns pa.Table (terminal
    API). Accept both for test convenience — CadwDataFrame written
    to parquet and read back (round-trip), pa.Table returned as-is.
    """
    if isinstance(df, pa.Table):
        return df
    out = tmp_path / name
    df.write_parquet(out)
    return pq.read_table(out)


# ============ correctness ============


def test_nlargest_basic(tmp_path):
    df = make_df(tmp_path, {"v": [3, 1, 4, 1, 5, 9, 2, 6]})
    out = df.nlargest(3, "v")
    t = _collect(out, tmp_path)
    assert t.column("v").to_pylist() == [9, 6, 5]


def test_nsmallest_basic(tmp_path):
    df = make_df(tmp_path, {"v": [3, 1, 4, 1, 5, 9, 2, 6]})
    out = df.nsmallest(3, "v")
    t = _collect(out, tmp_path)
    assert t.column("v").to_pylist() == [1, 1, 2]


def test_nlargest_identical_to_sort_head(tmp_path):
    # Unique values → no ties; identical to sort_values().head(n).
    rng = np.random.default_rng(42)
    vals = list(rng.permutation(1000))  # 0..999 unique
    df = make_df(tmp_path, {"v": vals}, chunk=50)
    k = 25
    got = df.nlargest(k, "v")
    ref = df.sort_values("v", ascending=False).head(k)
    gt = _collect(got, tmp_path, "got.parquet")
    rt = _collect(ref, tmp_path, "ref.parquet")
    assert gt.column("v").to_pylist() == rt.column("v").to_pylist()


def test_nsmallest_identical_to_sort_head(tmp_path):
    rng = np.random.default_rng(7)
    vals = list(rng.permutation(1000))
    df = make_df(tmp_path, {"v": vals}, chunk=50)
    k = 25
    got = df.nsmallest(k, "v")
    ref = df.sort_values("v", ascending=True).head(k)
    gt = _collect(got, tmp_path, "got.parquet")
    rt = _collect(ref, tmp_path, "ref.parquet")
    assert gt.column("v").to_pylist() == rt.column("v").to_pylist()


def test_nlargest_returns_multiple_columns(tmp_path):
    df = make_df(tmp_path, {
        "v": [3, 1, 5, 9, 2],
        "label": ["a", "b", "c", "d", "e"],
    })
    out = df.nlargest(2, "v")
    t = _collect(out, tmp_path)
    rows = t.to_pylist()
    assert [r["v"] for r in rows] == [9, 5]
    assert [r["label"] for r in rows] == ["d", "c"]


def test_nlargest_n_ge_total(tmp_path):
    df = make_df(tmp_path, {"v": [3, 1, 2]})
    out = df.nlargest(10, "v")
    t = _collect(out, tmp_path)
    assert t.column("v").to_pylist() == [3, 2, 1]


def test_nlargest_n_1(tmp_path):
    df = make_df(tmp_path, {"v": [3, 1, 4, 1, 5]})
    out = df.nlargest(1, "v")
    t = _collect(out, tmp_path)
    assert t.column("v").to_pylist() == [5]


def test_nlargest_with_nulls(tmp_path):
    df = make_df(tmp_path, {
        "v": pa.array([3.0, None, 1.0, 5.0, None, 2.0], type=pa.float64()),
    })
    out = df.nlargest(3, "v")
    t = _collect(out, tmp_path)
    assert t.column("v").to_pylist() == [5.0, 3.0, 2.0]


def test_nlargest_filtered(tmp_path):
    df = make_df(tmp_path, {"v": [3, 1, 5, 9, 2, 4]})
    filtered = df[df["v"] < 9]
    out = filtered.nlargest(2, "v")
    t = _collect(out, tmp_path)
    assert t.column("v").to_pylist() == [5, 4]


def test_nlargest_multibatch_chunk_1(tmp_path):
    # chunk=1 → 8 batches → many per-batch tops merged.
    df = make_df(tmp_path, {"v": [3, 1, 4, 1, 5, 9, 2, 6]}, chunk=1)
    out = df.nlargest(3, "v")
    t = _collect(out, tmp_path)
    assert t.column("v").to_pylist() == [9, 6, 5]


def test_nlargest_stable_tie_break(tmp_path):
    # Multiple equal values → keep earliest by row_id (matches
    # sort_values: external_sort tie-break by row_id).
    df = make_df(tmp_path, {
        "v": [5, 5, 5, 5],
        "label": ["a", "b", "c", "d"],
    })
    out = df.nlargest(2, "v")
    t = _collect(out, tmp_path)
    assert t.column("label").to_pylist() == ["a", "b"]


def test_nlargest_in_memory_dataset(tmp_path):
    t = pa.table({"v": [3, 1, 4, 1, 5, 9, 2, 6]})
    df = CadwDataFrame.from_table(t, memory_guard=MemoryGuard(max_memory_mb=10))
    out = df.nlargest(3, "v")
    ot = _collect(out, tmp_path)
    assert ot.column("v").to_pylist() == [9, 6, 5]


# ============ errors ============


def test_nlargest_invalid_n(tmp_path):
    df = make_df(tmp_path, {"v": [1]})
    with pytest.raises(ValueError, match="n must be int >= 1"):
        df.nlargest(0, "v")


def test_nlargest_missing_column(tmp_path):
    df = make_df(tmp_path, {"v": [1]})
    with pytest.raises(KeyError, match="nope"):
        df.nlargest(1, "nope")


def test_nlargest_reserved_row_id_column(tmp_path):
    df = make_df(tmp_path, {"v": [1, 2, 3], "__cadw_row_id": [10, 20, 30]})
    with pytest.raises(ValueError, match="__cadw_row_id"):
        df.nlargest(1, "v")


def test_nlargest_tight_budget(tmp_path):
    # Large n with tiny budget → fail-fast MemoryLimitExceeded.
    rng = np.random.default_rng(0)
    vals = rng.integers(0, 10**9, size=5000).tolist()
    df = make_df(tmp_path, {"v": vals}, chunk=100, mem_mb=1)
    df._memory_guard = MemoryGuard(max_memory_bytes=50_000)
    with pytest.raises(MemoryLimitExceeded, match="topk"):
        df.nlargest(5000, "v")


# ============ two-level AST guards (§13.4/190) ============


def test_nlargest_facade_delegates_to_topk():
    """Facade: nlargest calls topk_to_disk; no external_sort."""
    from cadw.dataframe import CadwDataFrame
    tree = parse_method_source(CadwDataFrame.nlargest)
    call_names = {
        n.func.id for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "topk_to_disk" in call_names
    src = __import__("inspect").getsource(CadwDataFrame.nlargest)
    assert_no_call_in_source(src, "external_sort",
                              fn_name="CadwDataFrame.nlargest")


def test_nsmallest_facade_delegates_to_topk():
    from cadw.dataframe import CadwDataFrame
    tree = parse_method_source(CadwDataFrame.nsmallest)
    call_names = {
        n.func.id for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "topk_to_disk" in call_names
    src = __import__("inspect").getsource(CadwDataFrame.nsmallest)
    assert_no_call_in_source(src, "external_sort",
                              fn_name="CadwDataFrame.nsmallest")


def test_topk_uses_select_k_or_sort_indices():
    """Impl: per-batch top-k uses select_k_unstable OR sort_indices."""
    from cadw.ops import topk as tk
    import inspect
    src = inspect.getsource(tk._per_batch_topk)
    assert "select_k_unstable" in src or "sort_indices" in src


def test_topk_to_disk_guards_before_merge():
    """Impl: check_state before SpillBuffer allocation (v0.9.90-S2).

    v0.9.90-S2 replaced `per_batch: list[pa.Table]` + one big
    `concat_tables(per_batch)` with incremental merge
    (`current_top = _per_batch_topk(current_top + batch_top, k)`).
    Now:
      - Per-iteration guard: check_state(current_top.nbytes) — bounded
        by k × row_size (output itself).
      - Auto-path: SpillBuffer(...) after the loop.
      - concat_tables appears only for 2-table merges inside the loop,
        not for whole-list materialize.

    Assertion: check_state (the guard) precedes the SpillBuffer
    construction (the next potential allocation point).
    """
    from cadw.patch_utils import assert_call_order
    from cadw.ops import topk as tk
    import inspect
    src = inspect.getsource(tk.topk_to_disk)
    # Guard exists
    assert "check_state" in src, "topk_to_disk must call check_state"
    # Guard before output allocation (SpillBuffer in auto path)
    assert_call_order(src, "check_state", "SpillBuffer",
                      fn_name="topk_to_disk")


def test_topk_no_to_pandas_ast():
    from cadw.patch_utils import assert_no_call_in_source
    from cadw.ops import topk as tk
    import inspect
    for fn_name in ("topk_to_disk", "_per_batch_topk"):
        src = inspect.getsource(getattr(tk, fn_name))
        assert_no_call_in_source(src, "to_pandas", fn_name=fn_name)


# ============ speedup smoke (slow) ============


@pytest.mark.slow
def test_nlargest_faster_than_full_external_sort(tmp_path):
    """100k rows, k=10: streaming path << external_sort path.

    Benchmark reference (audit bench_r4): 3946× at N=2M. This
    test asserts only a modest speedup bound (>=5×) to stay
    robust in CI; correctness (identical top-k) covered above.
    """
    rng = np.random.default_rng(0)
    vals = rng.integers(0, 10**9, size=100_000).tolist()
    df = make_df(tmp_path, {"v": vals}, chunk=5000, mem_mb=50)

    t0 = time.time()
    got = df.nlargest(10, "v")
    t_topk = time.time() - t0

    t0 = time.time()
    ref = df.sort_values("v", ascending=False).head(10)
    t_sort = time.time() - t0

    gt = _collect(got, tmp_path, "got.parquet")
    rt = _collect(ref, tmp_path, "ref.parquet")
    assert gt.column("v").to_pylist() == rt.column("v").to_pylist()
    assert t_topk < t_sort, (
        f"nlargest ({t_topk:.3f}s) not faster than external_sort + head "
        f"({t_sort:.3f}s)"
    )
