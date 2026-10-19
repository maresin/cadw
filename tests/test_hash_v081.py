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
from cadw.ops.shuffle import compute_hashes, hash_shuffle, _pandas_hash_column


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

    Without types_mapper in _pandas_hash_column:
      - batch without nulls: to_pandas → int64
      - batch with nulls:    to_pandas → float64 (NaN)
      → hash(1) ≠ hash(1.0) → partition inconsistency → WRONG results.
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

def test_pandas_path_used_for_int64(monkeypatch):
    """Spy: _pandas_hash_column is called for int64 (fast path)."""
    from cadw.ops import shuffle as sh
    called = []
    orig = sh._pandas_hash_column

    def spy(arr):
        called.append(str(arr.type))
        return orig(arr)

    monkeypatch.setattr(sh, "_pandas_hash_column", spy)
    b = batch([1, 2, 3], names=["k"])
    compute_hashes(b, ["k"])
    assert "int64" in called, f"pandas path not used for int64: {called}"


def test_uint64_falls_back_to_python(monkeypatch):
    """uint64 can't fit Int64Dtype → must skip pandas, use Python."""
    from cadw.ops import shuffle as sh
    called_pandas = []
    called_python = []
    orig_pd = sh._pandas_hash_column
    orig_py = sh._python_hash_column

    def spy_pd(arr):
        called_pandas.append(str(arr.type))
        return orig_pd(arr)

    def spy_py(arr):
        called_python.append(str(arr.type))
        return orig_py(arr)

    monkeypatch.setattr(sh, "_pandas_hash_column", spy_pd)
    monkeypatch.setattr(sh, "_python_hash_column", spy_py)
    b = pa.RecordBatch.from_arrays(
        [pa.array([1, 2, 3], type=pa.uint64())], names=["k"]
    )
    compute_hashes(b, ["k"])
    # pandas called but returned None; Python called as fallback
    assert "uint64" in called_python


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
        guard, tmp_path, num_partitions=4, log=False,
    )
    assert result.num_rows == n
    rows = sorted(result.to_pylist(), key=lambda r: r["cat"])
    for i in [0, 1, n // 2, n - 1]:
        assert rows[i]["val_mean"] == float(i)

# ---------- v0.8.1-fix3: string uses Python path ----------

def test_string_uses_python_path(monkeypatch):
    """Strings must use _python_hash_column (pandas is slower — bench 50 vs 39 ms)."""
    from cadw.ops import shuffle as sh
    called = {"pandas": [], "python": []}

    orig_pd = sh._pandas_hash_column
    orig_py = sh._python_hash_column

    def spy_pd(arr):
        called["pandas"].append(str(arr.type))
        return orig_pd(arr)

    def spy_py(arr):
        called["python"].append(str(arr.type))
        return orig_py(arr)

    monkeypatch.setattr(sh, "_pandas_hash_column", spy_pd)
    monkeypatch.setattr(sh, "_python_hash_column", spy_py)

    b = batch(["a", "b", "c"], names=["k"])
    compute_hashes(b, ["k"])

    assert "string" in called["python"], (
        f"string must use _python_hash_column; got python={called['python']}, "
        f"pandas={called['pandas']}"
    )


def test_numeric_uses_pandas_path(monkeypatch):
    """Numeric keeps pandas C-hash (19-40× speedup)."""
    from cadw.ops import shuffle as sh
    called_pandas = []
    orig_pd = sh._pandas_hash_column

    def spy_pd(arr):
        called_pandas.append(str(arr.type))
        return orig_pd(arr)

    monkeypatch.setattr(sh, "_pandas_hash_column", spy_pd)
    b = batch([1, 2, 3], names=["k"])
    compute_hashes(b, ["k"])
    assert "int64" in called_pandas

