"""v0.9.23c: Arrow-native batched merge (AUDIT_FULL_v0922 item 7.A.1).

Output identity: byte-for-byte same sort order as v0.5.2a heapq.
Reference: Python sorted with stable tie-break by row_id (input order).
"""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.core.memory_guard import MemoryLimitExceeded
from cadw.io.batch_writer import ArrowIPCBatchWriter
from cadw.ops.sort import _merge_keys, external_sort


def _write_run(path, rows, schema=None):
    """Write a sorted run file with row_id column."""
    if schema is None:
        schema = pa.schema([
            ("k", pa.int64()),
            ("v", pa.int64()),
            ("row_id", pa.uint64()),
        ])
    t = pa.Table.from_pylist(rows, schema=schema)
    w = ArrowIPCBatchWriter(str(path), schema)
    try:
        for b in t.to_batches():
            w.write_batch(b)
    finally:
        w.close()
    return path


def _read_all(path):
    reader = pa.ipc.open_file(str(path))
    rows = []
    for i in range(reader.num_record_batches):
        rows.extend(reader.get_batch(i).to_pylist())
    return rows


def _python_sorted_reference(rows, key_columns, ascending, nulls_first):
    """Stable Python sorted with per-column asc/desc + nulls_first + row_id.

    Matches external_sort's contract exactly. row_id monotonic with
    input order -> stable sort by key equals heap's (key, row_id) compare.

    v0.9.23c-fix1: flat tuple (not per-column sub-tuples). Prior
    version nested row_id inside each per-column tuple, causing
    row_id to decide before second key column on multi-key sorts.
    Current tests pass with buggy version (only 1 key column), but
    flat tuple is correct for any key count.
    """
    def _cmp_key(row):
        parts = []
        for col, asc in zip(key_columns, ascending):
            v = row[col]
            # nulls_first: nulls compare less (0); else greater (2)
            if v is None:
                parts.append(0 if nulls_first else 2)
                parts.append(0)  # placeholder value for null
            else:
                parts.append(1)
                parts.append(v if asc else _Neg(v))
        parts.append(row["row_id"])
        return tuple(parts)
    return sorted(rows, key=_cmp_key)


class _Neg:
    """Wrapper for descending sort in Python sorted (inverts comparison)."""
    __slots__ = ("v",)
    def __init__(self, v): self.v = v
    def __lt__(self, other): return self.v > other.v
    def __eq__(self, other): return self.v == other.v


# ============ Basic correctness ============

def test_merge_two_runs_ascending(tmp_path):
    """Two sorted runs, ascending keys, no ties, no nulls."""
    run1 = _write_run(tmp_path / "r1.arrow", [
        {"k": 1, "v": 10, "row_id": 0},
        {"k": 5, "v": 50, "row_id": 2},
        {"k": 9, "v": 90, "row_id": 4},
    ])
    run2 = _write_run(tmp_path / "r2.arrow", [
        {"k": 2, "v": 20, "row_id": 1},
        {"k": 6, "v": 60, "row_id": 3},
        {"k": 10, "v": 100, "row_id": 5},
    ])
    out = tmp_path / "out.arrow"
    _merge_keys(
        [run1, run2], ["k", "v", "row_id"], ["k"], ["k", "v"],
        [True], False, out, read_batch_size=10,
        memory_guard=MemoryGuard(max_memory_mb=20),
    )
    rows = _read_all(out)
    assert [r["k"] for r in rows] == [1, 2, 5, 6, 9, 10]


def test_merge_three_runs_descending(tmp_path):
    """Descending keys."""
    run1 = _write_run(tmp_path / "r1.arrow", [
        {"k": 9, "v": 1, "row_id": 4},
        {"k": 6, "v": 2, "row_id": 3},
        {"k": 3, "v": 3, "row_id": 2},
    ])
    run2 = _write_run(tmp_path / "r2.arrow", [
        {"k": 10, "v": 4, "row_id": 5},
        {"k": 7, "v": 5, "row_id": 1},
        {"k": 1, "v": 6, "row_id": 0},
    ])
    out = tmp_path / "out.arrow"
    _merge_keys(
        [run1, run2], ["k", "v", "row_id"], ["k"], ["k", "v"],
        [False], False, out, read_batch_size=10,
        memory_guard=MemoryGuard(max_memory_mb=20),
    )
    rows = _read_all(out)
    assert [r["k"] for r in rows] == [10, 9, 7, 6, 3, 1]


def test_merge_tie_break_by_row_id(tmp_path):
    """Equal keys -> tie-break by row_id ascending."""
    run1 = _write_run(tmp_path / "r1.arrow", [
        {"k": 5, "v": 1, "row_id": 0},
        {"k": 5, "v": 3, "row_id": 2},
    ])
    run2 = _write_run(tmp_path / "r2.arrow", [
        {"k": 5, "v": 2, "row_id": 1},
        {"k": 5, "v": 4, "row_id": 3},
    ])
    out = tmp_path / "out.arrow"
    _merge_keys(
        [run1, run2], ["k", "v", "row_id"], ["k"], ["k", "v"],
        [True], False, out, read_batch_size=10,
        memory_guard=MemoryGuard(max_memory_mb=20),
    )
    rows = _read_all(out)
    assert [r["row_id"] for r in rows] == [0, 1, 2, 3]


def test_merge_nulls_last(tmp_path):
    """Nulls in key column sort last (nulls_first=False)."""
    run1 = _write_run(tmp_path / "r1.arrow", [
        {"k": 1, "v": 10, "row_id": 0},
        {"k": None, "v": 20, "row_id": 2},
    ])
    run2 = _write_run(tmp_path / "r2.arrow", [
        {"k": 2, "v": 30, "row_id": 1},
        {"k": None, "v": 40, "row_id": 3},
    ])
    out = tmp_path / "out.arrow"
    _merge_keys(
        [run1, run2], ["k", "v", "row_id"], ["k"], ["k", "v"],
        [True], False, out, read_batch_size=10,
        memory_guard=MemoryGuard(max_memory_mb=20),
    )
    rows = _read_all(out)
    assert [r["k"] for r in rows] == [1, 2, None, None]
    # Null tie-break by row_id
    assert [r["row_id"] for r in rows[2:]] == [2, 3]


def test_merge_nulls_first(tmp_path):
    """Nulls first (nulls_first=True)."""
    run1 = _write_run(tmp_path / "r1.arrow", [
        {"k": None, "v": 10, "row_id": 0},
        {"k": 5, "v": 20, "row_id": 2},
    ])
    run2 = _write_run(tmp_path / "r2.arrow", [
        {"k": 1, "v": 30, "row_id": 1},
        {"k": None, "v": 40, "row_id": 3},
    ])
    out = tmp_path / "out.arrow"
    _merge_keys(
        [run1, run2], ["k", "v", "row_id"], ["k"], ["k", "v"],
        [True], True, out, read_batch_size=10,
        memory_guard=MemoryGuard(max_memory_mb=20),
    )
    rows = _read_all(out)
    assert [r["k"] for r in rows] == [None, None, 1, 5]


def test_merge_multi_key_mixed_order(tmp_path):
    """Two keys, mixed asc/desc: [k1 asc, k2 desc]."""
    # v0.9.23c-fix2: explicit schema (k1, k2, v, row_id). Prior used
    # _write_run default schema (k, v, row_id), which does not contain
    # k1/k2 -> file was written with wrong schema -> _merge_keys KeyError.
    schema = pa.schema([
        ("k1", pa.int64()),
        ("k2", pa.int64()),
        ("v", pa.int64()),
        ("row_id", pa.uint64()),
    ])
    run1 = _write_run(tmp_path / "r1.arrow", [
        {"k1": 1, "k2": 10, "v": 1, "row_id": 0},
        {"k1": 1, "k2": 5, "v": 2, "row_id": 1},
        {"k1": 2, "k2": 100, "v": 3, "row_id": 2},
    ], schema=schema)
    run2 = _write_run(tmp_path / "r2.arrow", [
        {"k1": 1, "k2": 7, "v": 4, "row_id": 3},
        {"k1": 2, "k2": 50, "v": 5, "row_id": 4},
    ], schema=schema)
    out = tmp_path / "out.arrow"
    _merge_keys(
        [run1, run2], ["k1", "k2", "v", "row_id"],
        ["k1", "k2"], ["k1", "k2", "v"],
        [True, False], False, out, read_batch_size=10,
        memory_guard=MemoryGuard(max_memory_mb=20),
    )
    rows = _read_all(out)
    # k1 asc, then k2 desc within each k1
    assert [(r["k1"], r["k2"]) for r in rows] == [
        (1, 10), (1, 7), (1, 5), (2, 100), (2, 50),
    ]


def test_merge_output_identity_vs_python_sorted(tmp_path):
    """Output row order == Python stable sort by (key..., row_id)."""
    rng = np.random.default_rng(42)
    n = 500
    keys = rng.integers(0, 50, size=n).astype(np.int64).tolist()
    vals = list(range(n))
    # Split into 4 sorted runs by key
    indices = sorted(range(n), key=lambda i: keys[i])
    rng.shuffle(indices)
    runs = [[], [], [], []]
    for pos, i in enumerate(indices):
        runs[pos % 4].append({"k": keys[i], "v": vals[i], "row_id": i})
    # Sort each run by (k, row_id)
    paths = []
    for ri, run_rows in enumerate(runs):
        run_rows.sort(key=lambda r: (r["k"], r["row_id"]))
        paths.append(_write_run(tmp_path / f"r{ri}.arrow", run_rows))

    out = tmp_path / "out.arrow"
    _merge_keys(
        paths, ["k", "v", "row_id"], ["k"], ["k", "v"],
        [True], False, out, read_batch_size=10,
        memory_guard=MemoryGuard(max_memory_mb=50),
    )
    got = _read_all(out)
    expected = _python_sorted_reference(
        [{"k": keys[i], "v": vals[i], "row_id": i} for i in range(n)],
        ["k"], [True], False,
    )
    assert [(r["k"], r["v"]) for r in got] == [
        (r["k"], r["v"]) for r in expected
    ]


# ============ Edge cases ============

def test_merge_single_run(tmp_path):
    """K=1 (defensive — external_sort never calls with K<2)."""
    run1 = _write_run(tmp_path / "r1.arrow", [
        {"k": 1, "v": 10, "row_id": 0},
        {"k": 2, "v": 20, "row_id": 1},
        {"k": 3, "v": 30, "row_id": 2},
    ])
    out = tmp_path / "out.arrow"
    _merge_keys(
        [run1], ["k", "v", "row_id"], ["k"], ["k", "v"],
        [True], False, out, read_batch_size=10,
        memory_guard=MemoryGuard(max_memory_mb=20),
    )
    assert [r["k"] for r in _read_all(out)] == [1, 2, 3]


def test_merge_empty_runs(tmp_path):
    """All runs empty -> empty output."""
    schema = pa.schema([
        ("k", pa.int64()), ("v", pa.int64()), ("row_id", pa.uint64()),
    ])
    run1 = _write_run(tmp_path / "r1.arrow", [], schema=schema)
    run2 = _write_run(tmp_path / "r2.arrow", [], schema=schema)
    out = tmp_path / "out.arrow"
    _merge_keys(
        [run1, run2], ["k", "v", "row_id"], ["k"], ["k", "v"],
        [True], False, out, read_batch_size=10,
        memory_guard=MemoryGuard(max_memory_mb=20),
    )
    assert _read_all(out) == []


def test_merge_rbs_gt_total(tmp_path):
    """rbs=1000 but only 6 rows total -> single round."""
    run1 = _write_run(tmp_path / "r1.arrow", [
        {"k": 1, "v": 10, "row_id": 0},
        {"k": 3, "v": 30, "row_id": 2},
        {"k": 5, "v": 50, "row_id": 4},
    ])
    run2 = _write_run(tmp_path / "r2.arrow", [
        {"k": 2, "v": 20, "row_id": 1},
        {"k": 4, "v": 40, "row_id": 3},
        {"k": 6, "v": 60, "row_id": 5},
    ])
    out = tmp_path / "out.arrow"
    _merge_keys(
        [run1, run2], ["k", "v", "row_id"], ["k"], ["k", "v"],
        [True], False, out, read_batch_size=1000,
        memory_guard=MemoryGuard(max_memory_mb=20),
    )
    assert [r["k"] for r in _read_all(out)] == [1, 2, 3, 4, 5, 6]


def test_merge_rbs_1(tmp_path):
    """rbs=1: one row per round — slow but correct."""
    run1 = _write_run(tmp_path / "r1.arrow", [
        {"k": 1, "v": 10, "row_id": 0},
        {"k": 5, "v": 50, "row_id": 2},
    ])
    run2 = _write_run(tmp_path / "r2.arrow", [
        {"k": 3, "v": 30, "row_id": 1},
        {"k": 7, "v": 70, "row_id": 3},
    ])
    out = tmp_path / "out.arrow"
    _merge_keys(
        [run1, run2], ["k", "v", "row_id"], ["k"], ["k", "v"],
        [True], False, out, read_batch_size=1,
        memory_guard=MemoryGuard(max_memory_mb=20),
    )
    assert [r["k"] for r in _read_all(out)] == [1, 3, 5, 7]


# ============ Integration via external_sort ============

def test_external_sort_n5000_ascending(tmp_path):
    """N=5000 rows, chunk=100, 4-col schema, ascending by k."""
    rng = np.random.default_rng(0)
    n = 5000
    table = pa.table({
        "k": rng.integers(0, 100, size=n).astype(np.int64),
        "v": rng.integers(0, 10000, size=n).astype(np.int64),
        "w": rng.integers(0, 10000, size=n).astype(np.int64),
    })
    src = tmp_path / "src.parquet"
    pq.write_table(table, src, row_group_size=100)
    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=100,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=tmp_path / "spill",
    )
    res = df.sort_values("k", ascending=True)
    out = tmp_path / "out.parquet"
    res.write_parquet(out)
    got = pq.read_table(out).column("k").to_pylist()
    # Reference: sorted k
    expected = sorted(table.column("k").to_pylist())
    assert got == expected


def test_external_sort_multi_level_merge(tmp_path):
    """Many runs -> multi-level merge. Final order must equal sorted."""
    n = 500
    keys = list(range(n, 0, -1))  # descending
    table = pa.table({
        "k": pa.array(keys, type=pa.int64()),
        "v": pa.array(list(range(n)), type=pa.int64()),
    })
    src = tmp_path / "src.parquet"
    pq.write_table(table, src, row_group_size=5)  # small chunks
    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=5,
        memory_guard=MemoryGuard(max_memory_mb=5),
        spill_dir=tmp_path / "spill",
    )
    res = df.sort_values("k", ascending=True)
    out = tmp_path / "out.parquet"
    res.write_parquet(out)
    got = pq.read_table(out).column("k").to_pylist()
    assert got == list(range(1, n + 1))


# ============ Memory guard ============

def test_merge_memory_guard_fires_on_tight_budget(tmp_path):
    """Huge rbs * K * row_bytes > state_limit -> MemoryLimitExceeded."""
    run1 = _write_run(tmp_path / "r1.arrow", [
        {"k": 1, "v": 10, "row_id": 0},
    ])
    run2 = _write_run(tmp_path / "r2.arrow", [
        {"k": 2, "v": 20, "row_id": 1},
    ])
    out = tmp_path / "out.arrow"
    guard = MemoryGuard(max_memory_bytes=1000)
    # rbs=10000 * K=2 * row=~40 = 800 KB buffer estimate > 500 B state_limit
    with pytest.raises(MemoryLimitExceeded, match="merge_buffer"):
        _merge_keys(
            [run1, run2], ["k", "v", "row_id"], ["k"], ["k", "v"],
            [True], False, out, read_batch_size=10000,
            memory_guard=guard,
        )


# ============ AST guard ============

def test_merge_keys_no_heapq():
    """v0.9.23c: no heapq.* Call in _merge_keys (AST-based).

    v0.9.23c-fix2: prior used `"heapq" not in src` — docstring says
    "Replaces per-row Python heapq with batched Arrow sort" -> substring
    false positive (§13.4/161). Now AST-based via
    patch_utils.assert_no_call_in_source, which walks Call nodes only
    (docstring/comments invisible).
    """
    import inspect
    from cadw.ops import sort as sort_mod
    from cadw.patch_utils import assert_no_call_in_source
    src = inspect.getsource(sort_mod._merge_keys)
    assert_no_call_in_source(
        src, "heappush", "heappop",
        fn_name="sort._merge_keys",
    )


def test_merge_keys_uses_pc_sort_indices():
    """v0.9.23c: pc.sort_indices present in _merge_keys."""
    import inspect
    from cadw.ops import sort as sort_mod
    src = inspect.getsource(sort_mod._merge_keys)
    assert "pc.sort_indices" in src
    assert "_RunStream" in src
    assert "row_id" in src  # tie-break key


def test_merge_keys_has_guard():
    """check_state Call precedes pc.sort_indices Call (AST guard §3.11).

    v0.9.23c-fix2: prior used src.find(). Docstring mentions
    "pc.sort_indices" (in "# Sort spec for pc.sort_indices" comment,
    line ~605) BEFORE actual check_state Call (line ~1313) ->
    substring order wrong (§13.4/185). Now AST-based via
    patch_utils.assert_call_order, which compares Call node linenos
    (docstring/comments invisible).
    """
    import inspect
    from cadw.ops import sort as sort_mod
    from cadw.patch_utils import assert_call_order
    src = inspect.getsource(sort_mod._merge_keys)
    assert_call_order(
        src, "check_state", "sort_indices",
        fn_name="sort._merge_keys",
    )
