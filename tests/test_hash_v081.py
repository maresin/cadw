"""v0.8.1: compute_hashes — pandas C-hash + cross-batch consistency.

Critical invariants:
    1. Same value → same hash across batches (with/without nulls). This is
       the partition-disjoint precondition. Without types_mapper,
       arr.to_pandas() gives int64 or float64 depending on null presence
       → broken.
    2. Seed changes hash (for recursive resplit).
    3. Null hashed deterministically (all nulls same hash).
    4. Fallback path (Python loop) for uint64 / unsupported.
"""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.core.memory_guard import MemoryGuard as MG
from cadw.ops.shuffle import compute_hashes, hash_shuffle


def batch(*cols, names):
    return pa.RecordBatch.from_arrays(
        [pa.array(c) if not isinstance(c, pa.Array) else c for c in cols],
        names=names,
    )


def make_df(tmp_path, data, chunk=2, name="L"):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


# ---------- CRITICAL: cross-batch consistency with nulls ----------

def test_int64_consistent_hash_across_null_presence():
    """int64 with nulls vs without → same hash for non-null values.

    v0.9.6a: hash is deterministic on value regardless of
    nulls present in the batch (own XXH64 for numeric;
    nulls sentinel).
    """
    b1 = batch([1, 2, 3], names=["k"])
    b2 = batch([1, 2, None], names=["k"])
    h1 = compute_hashes(b1, ["k"]).to_pylist()
    h2 = compute_hashes(b2, ["k"]).to_pylist()
    assert h1[0] == h2[0], f"hash(1) mismatch: {h1[0]} vs {h2[0]}"
    assert h1[1] == h2[1], f"hash(2) mismatch: {h1[1]} vs {h2[1]}"


def test_int32_consistent_hash_across_null_presence():
    """int32 — same invariant."""
    b1 = batch([1, 2], names=["k"])
    b2 = batch([1, None], names=["k"])
    # Force int32
    b1_int32 = pa.RecordBatch.from_arrays(
        [pa.array([1, 2], type=pa.int32())], names=["k"]
    )
    b2_int32 = pa.RecordBatch.from_arrays(
        [pa.array([1, None], type=pa.int32())], names=["k"]
    )
    h1 = compute_hashes(b1_int32, ["k"]).to_pylist()
    h2 = compute_hashes(b2_int32, ["k"]).to_pylist()
    assert h1[0] == h2[0]


def test_string_consistent_hash_across_null_presence():
    """String — same invariant (or consistent through fallback)."""
    b1 = batch(["a", "b"], names=["k"])
    b2 = batch(["a", None], names=["k"])
    h1 = compute_hashes(b1, ["k"]).to_pylist()
    h2 = compute_hashes(b2, ["k"]).to_pylist()
    assert h1[0] == h2[0]


# ---------- invariants ----------

def test_compute_hashes_deterministic():
    """Repeated calls produce identical hashes."""
    b = batch([1, 2, 3, 1], names=["k"])
    h1 = compute_hashes(b, ["k"]).to_pylist()
    h2 = compute_hashes(b, ["k"]).to_pylist()
    assert h1 == h2


def test_same_value_same_hash():
    b = batch([1, 2, 1, 3], names=["k"])
    h = compute_hashes(b, ["k"]).to_pylist()
    assert h[0] == h[2]


def test_different_value_different_hash():
    b = batch([1, 2, 3], names=["k"])
    h = compute_hashes(b, ["k"]).to_pylist()
    assert len(set(h)) == 3


def test_seed_changes_hash():
    """XOR seed → different partition layout for same data."""
    b = batch([1, 2, 3], names=["k"])
    h0 = compute_hashes(b, ["k"], seed=0).to_pylist()
    h1 = compute_hashes(b, ["k"], seed=1).to_pylist()
    h42 = compute_hashes(b, ["k"], seed=42).to_pylist()
    assert h0 != h1
    assert h0 != h42
    assert h1 != h42


def test_nulls_hash_consistently():
    """All nulls in a column hash to the same value (null == null for group)."""
    b = batch([1, None, 2, None, 3], names=["k"])
    h = compute_hashes(b, ["k"]).to_pylist()
    assert h[1] == h[3]


def test_nulls_hash_differently_from_any_number():
    """Null hash differs from all non-null hashes (null is its own group)."""
    b = batch([1, 2, 3, None], names=["k"])
    h = compute_hashes(b, ["k"]).to_pylist()
    null_h = h[3]
    assert null_h not in h[:3]


# ---------- AST / path check ----------


# ---------- partition-disjoint end-to-end ----------

def test_hash_shuffle_partition_disjoint_int(tmp_path):
    """Partition-disjoint invariant preserved after hash change."""
    n = 2000
    rng = np.random.default_rng(0)
    keys = rng.integers(0, 100, size=n).astype(np.int64).tolist()
    vals = list(range(n))
    t = pa.table({"k": pa.array(keys), "v": pa.array(vals)})

    def bi():
        for i in range(0, n, 100):
            yield t.slice(i, 100).to_batches()[0]

    paths = hash_shuffle(
        bi(), ["k"], ["v"], 8,
        MG(max_memory_mb=20), tmp_path, log=False,
    )
    # key->partition map
    key_to_parts: dict[int, set[int]] = {}
    for p_idx, path in enumerate(paths):
        if not path.exists():
            continue
        reader = pa.ipc.open_file(str(path))
        for i in range(reader.num_record_batches):
            b = reader.get_batch(i)
            for r in b.to_pylist():
                key_to_parts.setdefault(r["k"], set()).add(p_idx)
    for k, parts in key_to_parts.items():
        assert len(parts) == 1, f"key {k} split across {parts}"


def test_hash_shuffle_partition_disjoint_string(tmp_path):
    """Same, for string keys — even if falls back to Python."""
    n = 2000
    rng = np.random.default_rng(1)
    keys = [f"g{int(x):03d}" for x in rng.integers(0, 50, size=n)]
    t = pa.table({"k": pa.array(keys), "v": pa.array(list(range(n)))})

    def bi():
        for i in range(0, n, 100):
            yield t.slice(i, 100).to_batches()[0]

    paths = hash_shuffle(
        bi(), ["k"], ["v"], 8,
        MG(max_memory_mb=20), tmp_path, log=False,
    )
    key_to_parts: dict[str, set[int]] = {}
    for p_idx, path in enumerate(paths):
        if not path.exists():
            continue
        reader = pa.ipc.open_file(str(path))
        for i in range(reader.num_record_batches):
            b = reader.get_batch(i)
            for r in b.to_pylist():
                key_to_parts.setdefault(r["k"], set()).add(p_idx)
    for k, parts in key_to_parts.items():
        assert len(parts) == 1


def test_groupby_after_hash_change(tmp_path):
    """End-to-end groupby with new hash — same result."""
    n = 1000
    t = pa.table({
        "cat": [f"g{i % 50:03d}" for i in range(n)],
        "v": list(range(n)),
    })
    p = Path(tmp_path) / "src.parquet"
    pq.write_table(t, p)
    df = CadwDataFrame.from_parquet(
        p, tmp_path / "ds", chunk_size=100,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=tmp_path / "spill",
    )
    result = df.groupby("cat").agg("v", "sum")
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert len(rows) == 50
    for r in rows:
        i = int(r["cat"][1:])
        # sum of v for g{i}: i, i+50, i+100, ... (20 items)
        # = 20*i + 50*(0+1+...+19) = 20*i + 9500
        assert r["v_sum"] == pytest.approx(20 * i + 9500, rel=1e-9)

# ---------- Avalanche: low-bit distribution (v0.8.1-fix2) ----------

def test_avalanche_spreads_low_bits():
    """After avalanche, h & (P-1) is roughly uniform for P in {2, 4, 8, 16}.

    Before avalanche: pandas hash for string is even → h & 1 = 0 always,
    so P=4 uses only partitions {0, 2}. After: all partitions used.
    """
    n = 5000
    values = [f"g{i:05d}" for i in range(n)]
    b = batch(values, names=["k"])
    h = np.array(compute_hashes(b, ["k"]).to_pylist(), dtype=np.uint64)
    for P in (2, 4, 8, 16):
        counts = np.bincount(h & np.uint64(P - 1), minlength=P)
        # Each bucket should get ~n/P (allow 30% deviation, binomial noise)
        expected = n / P
        min_ok = expected * 0.7
        max_ok = expected * 1.3
        assert counts.min() >= min_ok, (
            f"P={P}: min bucket {counts.min()} < {min_ok} "
            f"(counts={counts.tolist()})"
        )
        assert counts.max() <= max_ok, (
            f"P={P}: max bucket {counts.max()} > {max_ok} "
            f"(counts={counts.tolist()})"
        )


def test_avalanche_seed_spreads_low_bits():
    """After avalanche + seed, distribution still uniform for any seed."""
    n = 5000
    values = [f"g{i:05d}" for i in range(n)]
    b = batch(values, names=["k"])
    for seed in (0, 1, 2, 3, 4, 5, 42):
        h = np.array(
            compute_hashes(b, ["k"], seed=seed).to_pylist(),
            dtype=np.uint64,
        )
        counts = np.bincount(h & np.uint64(3), minlength=4)
        expected = n / 4
        assert counts.min() >= expected * 0.7, (
            f"seed={seed}, P=4: min bucket {counts.min()} < {expected * 0.7}"
        )


def test_avalanche_similar_strings_different_hashes():
    """Adjacent strings g00000, g00001, g00002 — must differ."""
    b = batch(["g00000", "g00001", "g00002"], names=["k"])
    h = compute_hashes(b, ["k"]).to_pylist()
    assert len(set(h)) == 3


def test_groupby_recursion_end_to_end_v081(tmp_path):
    """Regression: test_mustfix_local_groupby::test_groupby_aggregate_recursion_end_to_end
    must pass with v0.8.1-fix2 avalanche."""
    from cadw.core.memory_guard import MemoryGuard
    from cadw.ops.groupby import groupby_aggregate

    n = 5000
    table = pa.table({
        "cat": [f"g{i:05d}" for i in range(n)],
        "val": list(range(n)),
    })

    def bi():
        for i in range(0, n, 500):
            yield table.slice(i, 500).to_batches()[0]

    guard = MemoryGuard(max_memory_bytes=50_000)
    result = groupby_aggregate(
        bi(), ["cat"], ["val"], ["mean"],
        guard, tmp_path, num_partitions=4, log=False, output="memory",
    )
    assert result.num_rows == n
    rows = sorted(result.to_pylist(), key=lambda r: r["cat"])
    for i in [0, 1, n // 2, n - 1]:
        assert rows[i]["val_mean"] == float(i)

# ---------- v0.8.1-fix3: string uses Python path ----------


# ============ v0.9.18-fix1 (§13.4/226): dict-encode null path ============


def test_string_hash_dict_encode_path_with_nulls():
    """Dictionary-encode fast path with nulls must not IndexError.

    §13.4/226: DictionaryArray.indices.to_numpy(zero_copy_only=False)
    returns float64 (NaN for nulls), not int64. Prior
    `uniq_hashes[indices[non_null]]` → IndexError: arrays used as
    indices must be integer type. Fix: fill_null(0).astype(int64) +
    separate is_null mask.
    """
    from cadw.ops.shuffle import _string_hash_column, _NULL_SENTINEL
    arr = pa.array(["a", "b", "a", None, "b", None], type=pa.string())
    h = _string_hash_column(arr)
    assert h[0] == h[2]          # "a" == "a"
    assert h[1] == h[4]          # "b" == "b"
    assert h[0] != h[1]          # "a" != "b"
    assert h[3] == h[5] == _NULL_SENTINEL  # nulls coalesce
    assert _NULL_SENTINEL not in h[[0, 1, 2, 4]]


def test_string_hash_all_nulls():
    """All-null array → all _NULL_SENTINEL, no crash."""
    from cadw.ops.shuffle import _string_hash_column, _NULL_SENTINEL
    arr = pa.array([None, None, None], type=pa.string())
    h = _string_hash_column(arr)
    assert all(int(v) == int(_NULL_SENTINEL) for v in h)


def test_string_hash_empty_array():
    """Empty array → empty result."""
    from cadw.ops.shuffle import _string_hash_column
    arr = pa.array([], type=pa.string())
    h = _string_hash_column(arr)
    assert len(h) == 0


def test_string_hash_no_nulls_no_regression():
    """No-null path unchanged (all unique values)."""
    from cadw.ops.shuffle import _string_hash_column
    arr = pa.array(["a", "b", "c"], type=pa.string())
    h = _string_hash_column(arr)
    assert len(set(h.tolist())) == 3



# ============ v0.9.6a: own XXH64 hash, no pandas ============


def test_hash_no_pandas_import_ast():
    # AST: cadw/ops/shuffle.py has no `import pandas`.
    import ast as _ast
    import inspect
    from cadw.ops import shuffle as sh
    src = inspect.getsource(sh)
    tree = _ast.parse(src)
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("pandas"), (
                    "shuffle.py must not import pandas (v0.9.6a)"
                )
        elif isinstance(node, _ast.ImportFrom):
            if node.module:
                assert not node.module.startswith("pandas"), (
                    "shuffle.py must not import pandas (v0.9.6a)"
                )


def test_own_hash_removed_pandas_functions():
    # Belt-and-suspenders: old function names gone from source.
    import inspect
    from cadw.ops import shuffle as sh
    src = inspect.getsource(sh)
    assert "_pandas_hash_column" not in src
    assert "_python_hash_column" not in src
    assert "_vectorized_hash_numeric" in src
    assert "_string_hash_column" in src


def test_own_hash_cross_batch_int64_with_and_without_nulls():
    b1 = batch([1, 2, 3], names=["k"])
    b2 = batch([1, 2, None], names=["k"])
    h1 = compute_hashes(b1, ["k"]).to_pylist()
    h2 = compute_hashes(b2, ["k"]).to_pylist()
    assert h1[0] == h2[0]
    assert h1[1] == h2[1]


def test_own_hash_cross_batch_float64_with_and_without_nulls():
    b1 = batch([1.5, 2.5, 3.5], names=["k"])
    b2 = batch([1.5, 2.5, None], names=["k"])
    h1 = compute_hashes(b1, ["k"]).to_pylist()
    h2 = compute_hashes(b2, ["k"]).to_pylist()
    assert h1[0] == h2[0]
    assert h1[1] == h2[1]


def test_own_hash_cross_batch_string_with_and_without_nulls():
    b1 = batch(["a", "b"], names=["k"])
    b2 = batch(["a", None], names=["k"])
    h1 = compute_hashes(b1, ["k"]).to_pylist()
    h2 = compute_hashes(b2, ["k"]).to_pylist()
    assert h1[0] == h2[0]


def test_own_hash_null_consistency():
    b = batch([None, None, None], names=["k"])
    h = compute_hashes(b, ["k"]).to_pylist()
    assert h[0] == h[1] == h[2]


def test_own_hash_uint64_extreme_values():
    # uint64 values >= 2^63 must hash distinctively (safe=False bitcast).
    values = [1, 2**63, 2**63 + 1, 2**64 - 1]
    b = pa.RecordBatch.from_arrays(
        [pa.array(values, type=pa.uint64())], names=["k"],
    )
    h = compute_hashes(b, ["k"]).to_pylist()
    assert len(set(h)) == len(values), (
        f"distinct uint64 values must hash distinctively; got {h}"
    )


def test_own_hash_timestamp_cross_batch():
    import datetime as dt
    ts = [dt.datetime(2020, 1, 1), dt.datetime(2020, 6, 1)]
    b1 = pa.RecordBatch.from_arrays(
        [pa.array(ts, type=pa.timestamp("s"))], names=["k"],
    )
    b2 = pa.RecordBatch.from_arrays(
        [pa.array([ts[0], None], type=pa.timestamp("s"))], names=["k"],
    )
    h1 = compute_hashes(b1, ["k"]).to_pylist()
    h2 = compute_hashes(b2, ["k"]).to_pylist()
    assert h1[0] == h2[0]


def test_own_hash_avalanche_uniform_low_bits():
    # After splitmix64, low bits uniform for P in {2,4,8,16}.
    import numpy as np_
    n = 5000
    values = [f"g{i:05d}" for i in range(n)]
    b = batch(values, names=["k"])
    h = np_.array(compute_hashes(b, ["k"]).to_pylist(),
                  dtype=np_.uint64)
    for P in (2, 4, 8, 16):
        counts = np_.bincount(h & np_.uint64(P - 1), minlength=P)
        expected = n / P
        assert counts.min() >= expected * 0.7, (
            f'P={P}: min bucket {counts.min()} < {expected*0.7}'
        )
        assert counts.max() <= expected * 1.3


def test_no_test_imports_stale_hash_names():
    # AST: no test file imports _pandas_hash_column or
    # _python_hash_column (removed v0.9.6a). Any import attempt
    # fails collection with cryptic ImportError; AST guard
    # fails earlier with actionable message (§13.4/174).
    import ast as _ast
    from pathlib import Path as _P
    STALE = {'_pandas_hash_column', '_python_hash_column'}
    tests_root = _P(__file__).parent
    violations = []
    for py in tests_root.rglob('test_*.py'):
        if py.name == _P(__file__).name:
            continue
        try:
            tree = _ast.parse(py.read_text())
        except SyntaxError:
            continue
        for node in _ast.walk(tree):
            if not isinstance(node, _ast.ImportFrom):
                continue
            if not node.module or 'shuffle' not in node.module:
                continue
            for alias in node.names:
                if alias.name in STALE:
                    violations.append(
                        f'{py.name}:{node.lineno} '
                        f'imports {alias.name}'
                    )
    assert not violations, (
        'stale hash imports (removed v0.9.6a):\n'
        + '\n'.join(violations)
    )
