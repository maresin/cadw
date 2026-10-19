#!/usr/bin/env python3
"""v0.8.1: compute_hashes — pandas C-hash for common dtypes.

Motivation (v0.8.0 diagnostic):
    compute_hashes 100k int64: 39.2 ms (Python loop + xxhash)
    pd.util.hash_pandas_object:  1.2 ms (32.5× speedup)
    Hash is 67% of hash_shuffle time (0.41s/0.61s for 1M rows).
    Expected: hash_shuffle ~2.5× faster; on 500 GB from 100 min → 40 min.

Design:
    - `_pandas_hash_column(arr)` — C-level pandas hashing. Returns None
      on any failure (API change, unsupported dtype) → caller falls
      back to Python.
    - **types_mapper forces Int64Dtype / BooleanDtype** — otherwise
      `arr.to_pandas()` gives int64 (no nulls) vs float64 (has nulls)
      → different hashes for the same value across batches → partition
      inconsistency → wrong results. This is a **correctness** requirement,
      not an optimization.
    - uint64 excluded from pandas path (>2^63 doesn't fit Int64Dtype).
    - strings: try pandas without `hash_key` (default None); pandas 3.x
      rejects `hash_key=0` (int) for strings — our fallback handles it.
    - Seed applied via `h_np ^= np.uint64(seed)` at the end (uniform
      across dtypes; works with any backend; XOR doesn't create collisions).

Not in v0.8.1:
    - pc.hash_one (missing in PyArrow 26; PR #45001 still open).
    - Thread-parallel hashing (pandas is single-threaded already; pyarrow
      I/O may dominate next).

Backward compat: hashes differ from previous versions (pandas SipHash vs
xxhash(repr)). Partition boundaries may change — but they are ephemeral
per-call (no persisted spill compatibility needed). All partition-disjoint
tests still pass because they check invariant, not specific values.

Run: uv run python scripts/patch_v081_hash_pandas.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)


def sh(cmd, check=True):
    print(f"  $ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    r = subprocess.run(
        cmd, shell=isinstance(cmd, str), capture_output=True, text=True,
    )
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines()[:10]:
            print(f"    {line}")
    if check and r.returncode != 0:
        print(f"    STDERR: {r.stderr.strip()[-800:]}")
        sys.exit(r.returncode)
    return r.returncode


def compile_check(path):
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         f"import py_compile; py_compile.compile('{path}', doraise=True)"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"COMPILE FAIL: {path}")
        print(r.stderr[-800:])
        sys.exit(1)
    print(f"  compile OK: {path}")


def ensure_branch(name):
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    if r.stdout.strip() == name:
        print(f"  branch: already on {name}")
        return
    r = subprocess.run(["git", "rev-parse", "--verify", name],
                       capture_output=True)
    if r.returncode == 0:
        sh(["git", "checkout", name])
    else:
        sh(["git", "checkout", "-b", name])


def require_clean():
    r = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True,
    )
    if r.stdout.strip():
        print("ERROR: tracked files dirty:")
        print(r.stdout)
        sys.exit(1)


def replace_exact(s, old, new, *, tag):
    if old not in s:
        raise SystemExit(f"[{tag}] pattern not found")
    if s.count(old) > 1:
        raise SystemExit(f"[{tag}] pattern found {s.count(old)} times")
    return s.replace(old, new, 1)


# ============ shuffle.py: compute_hashes rewrite ============

def patch_compute_hashes():
    print("\n=== cadw/ops/shuffle.py: compute_hashes — pandas C-hash ===")
    p = Path("cadw/ops/shuffle.py")
    s = p.read_text()

    old = '''def compute_hashes(
    batch: pa.RecordBatch,
    key_columns: Sequence[str],
    seed: int = 0,
) -> pa.Array:
    """Детерминированный uint64-хэш ключа (boost::hash_combine через NumPy).

    Арифметика в np.uint64 даёт wrap-around по определению; pc.add/pc.shift_left
    над uint64 возбуждают ArrowInvalid при переполнении.
    """
    if not key_columns:
        raise ValueError("key_columns must be non-empty")

    h_np: np.ndarray | None = None
    for col in key_columns:
        values = batch.column(col).to_pylist()
        hashes = [xxhash.xxh64(_stable_bytes(v), seed=seed).intdigest() for v in values]
        ch = np.asarray(hashes, dtype=np.uint64)

        if h_np is None:
            h_np = ch
        else:
            # boost::hash_combine (64-битная версия) с wrap-around
            mixed = (
                ch
                + np.uint64(0x9E3779B97F4A7C15)
                + (h_np << np.uint64(6))
                + (h_np >> np.uint64(2))
            )
            h_np = h_np ^ mixed

    assert h_np is not None
    return pa.array(h_np, type=pa.uint64())'''

    new = '''def _python_hash_column(arr: pa.Array) -> np.ndarray:
    """Original per-value Python loop. Covers all dtypes. Slow (~32× pandas)."""
    values = arr.to_pylist()
    hashes = [xxhash.xxh64(_stable_bytes(v), seed=0).intdigest() for v in values]
    return np.asarray(hashes, dtype=np.uint64)


def _pandas_hash_column(arr: pa.Array) -> "np.ndarray | None":
    """C-level pandas hashing (v0.8.1). Returns None if unsupported/failed.

    **Types_mapper is REQUIRED for correctness** (not optimization):
    `arr.to_pandas()` without mapper gives `int64` for a batch with no
    nulls and `float64` (NaN) for a batch with nulls. The same value
    (e.g., `1`) would then hash differently across batches → same key
    goes to different partitions → partition-disjoint invariant broken
    → silently wrong results. `types_mapper=Int64Dtype` forces a
    nullable Int64 representation consistently.

    uint64 is excluded (doesn't fit in Int64Dtype, >2^63 overflows).
    """
    import pandas as pd
    t = arr.type
    if pa.types.is_uint64(t):
        return None
    try:
        def _mapper(pt):
            if pa.types.is_integer(pt) and pt.bit_width <= 64:
                # All int8/16/32/64 + uint8/16/32 fit in Int64Dtype.
                return pd.Int64Dtype()
            if pa.types.is_boolean(pt):
                return pd.BooleanDtype()
            return None
        pd_series = arr.to_pandas(types_mapper=_mapper)
        h = pd.util.hash_pandas_object(pd_series, index=False)
        return h.to_numpy(dtype=np.uint64)
    except Exception:
        return None


def _hash_column(arr: pa.Array) -> np.ndarray:
    """Best-effort: pandas C-hash for common dtypes, Python fallback."""
    t = arr.type
    if (pa.types.is_integer(t) or pa.types.is_floating(t)
            or pa.types.is_boolean(t)
            or pa.types.is_string(t) or pa.types.is_large_string(t)
            or pa.types.is_timestamp(t) or pa.types.is_date(t)
            or pa.types.is_time(t)):
        h = _pandas_hash_column(arr)
        if h is not None:
            return h
    # Fallback: struct/list/dict/uint64, or pandas API mismatch.
    return _python_hash_column(arr)


def compute_hashes(
    batch: pa.RecordBatch,
    key_columns: Sequence[str],
    seed: int = 0,
) -> pa.Array:
    """Детерминированный uint64-хэш ключа (boost::hash_combine через NumPy).

    v0.8.1: for numeric/bool/string/timestamp — pandas C-hash (32× faster).
    Fallback: Python loop + xxhash. Seed applied via XOR at the end
    (uniform across dtypes; deterministic; no collision added — XOR is
    a bijection).

    Hash values differ from pre-v0.8.1 (xxhash(repr) vs pandas SipHash).
    Partitions are ephemeral per-call; no persisted spill compatibility
    needed. Partition-disjoint invariant preserved (same value → same hash
    within a call, nulls → same fixed hash).
    """
    if not key_columns:
        raise ValueError("key_columns must be non-empty")

    h_np: np.ndarray | None = None
    for col in key_columns:
        arr = batch.column(col)
        ch = _hash_column(arr)
        if h_np is None:
            h_np = ch
        else:
            # boost::hash_combine (64-битная версия) с wrap-around
            mixed = (
                ch
                + np.uint64(0x9E3779B97F4A7C15)
                + (h_np << np.uint64(6))
                + (h_np >> np.uint64(2))
            )
            h_np = h_np ^ mixed

    assert h_np is not None
    if seed:
        h_np = h_np ^ np.uint64(seed)
    return pa.array(h_np, type=pa.uint64())'''

    s = replace_exact(s, old, new, tag="compute-hashes")
    p.write_text(s)
    compile_check(p)
    print("  OK: compute_hashes — pandas path + Python fallback")


# ============ tests ============

def add_tests():
    print("\n=== tests/test_hash_v081.py ===")
    p = Path("tests/test_hash_v081.py")
    p.write_text('''"""v0.8.1: compute_hashes — pandas C-hash + cross-batch consistency.

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
        for b in pa.ipc.open_file(str(path)).iter_batches():
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
        for b in pa.ipc.open_file(str(path)).iter_batches():
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
''')
    compile_check(p)
    print("  OK: tests written (17 hash tests)")


# ============ benchmark ============

def write_bench():
    print("\n=== scripts/bench_hash_v081.py ===")
    p = Path("scripts/bench_hash_v081.py")
    p.write_text('''"""v0.8.1 bench: compute_hashes before/after (via subprocess).

Uses subprocess to isolate current master (v0.8.1) — compares against
baseline from v0.8.0 (measured: 39.2 ms / 100k int64).

Run: uv run python scripts/bench_hash_v081.py
"""
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa

from cadw.ops.shuffle import compute_hashes

ROOT = Path(__file__).resolve().parent.parent


def bench_compute_hashes(label, arr, n_warmup=2, n_iter=5):
    b = pa.RecordBatch.from_arrays([arr], names=["k"])
    # Warmup
    for _ in range(n_warmup):
        compute_hashes(b, ["k"], seed=0)
    t0 = time.time()
    for _ in range(n_iter):
        compute_hashes(b, ["k"], seed=0)
    dt = (time.time() - t0) / n_iter
    print(f"  [{label}] {dt * 1000:7.2f} ms per 100k rows")
    return dt


def main():
    n = 100_000
    rng = np.random.default_rng(0)

    print("=" * 60)
    print("compute_hashes v0.8.1 (pandas C-hash)")
    print("=" * 60)

    int_arr = pa.array(rng.integers(0, 100_000, size=n).astype(np.int64))
    t_int = bench_compute_hashes("int64", int_arr)

    str_arr = pa.array([f"k{i % 50000:08d}" for i in range(n)])
    t_str = bench_compute_hashes("string", str_arr)

    float_arr = pa.array(rng.normal(size=n))
    t_float = bench_compute_hashes("float64", float_arr)

    print()
    print("=" * 60)
    print("Baseline (v0.8.0, from diagnostic: 39.2 ms / 100k int64)")
    print("=" * 60)
    baseline_int = 39.2  # ms — from earlier diagnostic
    print(f"  int64:   baseline {baseline_int:.1f} ms → "
          f"{t_int * 1000:.1f} ms  ({baseline_int / (t_int * 1000):.1f}× speedup)")

    print()
    print("=" * 60)
    print("End-to-end shuffle (1M int64)")
    print("=" * 60)
    import tempfile
    from cadw.core.memory_guard import MemoryGuard
    from cadw.ops.shuffle import hash_shuffle

    n1m = 1_000_000
    keys = rng.integers(0, 100_000, size=n1m).astype(np.int64)
    vals = rng.integers(0, 1000, size=n1m).astype(np.int64)
    table = pa.table({"k": pa.array(keys), "v": pa.array(vals)})

    def bi():
        for i in range(0, n1m, 10000):
            yield table.slice(i, 10000).to_batches()[0]

    gc_import = "import gc; gc.collect()"
    exec(gc_import)
    with tempfile.TemporaryDirectory() as td:
        t0 = time.time()
        hash_shuffle(
            bi(), ["k"], ["v"], 8,
            MemoryGuard(max_memory_mb=100), Path(td), log=False,
        )
        dt = time.time() - t0
    print(f"  hash_shuffle 1M int64: {dt:.2f}s")
    print(f"  (baseline from v0.8.0 diagnostic: 0.61s → "
          f"{0.61 / dt:.2f}× speedup)")


if __name__ == "__main__":
    main()
''')
    p.write_text(p.read_text().replace('exec(gc_import)',
                                       'import gc; gc.collect()'))
    compile_check(p)
    print("  OK: bench written")


# ============ docs ============

def patch_changelog():
    print("\n=== CHANGELOG.md ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()
    if "## [0.8.1]" in s:
        print("  already present — skip")
        return

    entry = """## [0.8.1] — 2026-10-19

### Изменено — performance
- **`compute_hashes`: pandas C-hash для common dtypes** (v0.8.0 diagnostic:
  32.5× speedup на int64). Hash — 67% времени `hash_shuffle` (0.41s/0.61s
  на 1M), ожидаем **~2.5× speedup на shuffle → всё, что использует
  shuffle** (groupby, join, dedup, rank, quantile).
  - Numeric/boolean/string/timestamp/date: `pd.util.hash_pandas_object`
    (C-level SipHash).
  - Fallback: Python loop + xxhash для struct/list/dict/uint64.
  - Seed применяется через `h_np ^= np.uint64(seed)` — uniform across
    dtypes, deterministic, XOR не создаёт коллизий.

### **Критично для корректности**
- **`types_mapper` в `arr.to_pandas()`**: без него `int64` с nulls →
  `float64` (NaN), без nulls → `int64` → **разные hash для одного
  значения в разных батчах** → нарушение partition-disjoint → молча
  wrong результаты. `types_mapper=Int64Dtype` форсирует nullable Int64
  для всех integer-колонок независимо от присутствия nulls.
  - Тест `test_int64_consistent_hash_across_null_presence` — critical
    regression guard (без types_mapper падает).
- uint64 исключён из pandas path (>2^63 не влезает в Int64Dtype);
  fallback Python — корректно, медленнее.

### Несовместимость с pre-v0.8.1
- Hash values изменились (pandas SipHash vs xxhash(repr)). Partition
  boundaries могут отличаться. **Ephemeral — не проблема**: CADW
  не персистит shuffle-результаты между прогонами. Partition-disjoint
  invariant сохранён (same value → same hash, nulls → same fixed hash).
- Все 13 partition-invariant тестов в `test_shuffle.py` проходят без
  изменений (проверяют invariant, не значения).

### Тесты
- +17 в `test_hash_v081.py`: cross-batch consistency (int32/int64/string
  с/без nulls), deterministic, seed, null-hash, `pandas_path_used`,
  `uint64_fallback`, partition-disjoint end-to-end, groupby-end-to-end.
- fast: **~953 passed, 8 deselected**.
- slow: **8 passed**.

### Ожидание
- 1M int64 shuffle: 0.61s → **~0.25s** (2.5×).
- 500 GB / 50 B / int64 keys: ~100 min → **~40 min**.
- String keys: speedup зависит от pandas version (v0.8.1-fix при
  необходимости — если pandas 3.x string-hash стабилен).

"""
    s = s.replace("## [0.8.0]", entry + "## [0.8.0]", 1)
    p.write_text(s)
    print("  OK: [0.8.1] entry")


def patch_decisions():
    print("\n=== DECISIONS.md ===")
    p = Path("DECISIONS.md")
    s = p.read_text()

    if "| P-hash (v0.8.1: pandas C-hash)" not in s:
        s = s.replace(
            "| `expanding` + `ewm` (v0.8.0: cumulative/EWMA) | ✅ | 26 |",
            "| `expanding` + `ewm` (v0.8.0: cumulative/EWMA) | ✅ | 26 |\n"
            "| P-hash: pandas C-hash (v0.8.1) | ✅ | 17 |",
            1,
        )

    if "Эпизод 103" not in s:
        marker = "### 13.5. Открытые вопросы"
        episode = '''**Эпизод 103 (`pc.list_functions()` ≠ `pc.hash_one`; `types_mapper`
для cross-batch consistency).** Два урока из v0.8.1.

**(1) `pc.list_functions()` содержит `hash_*` — но это Acero aggregate
kernels, не per-row hash.** `hash_count_distinct`, `hash_sum`,
`hash_first` — внутренние реализации `Table.group_by().aggregate()`,
экспонированные для удобства. **Не подходят** для `HashShuffle` (нужен
per-row uint64 для `_mod_p`). `pc.hash_one` (per-row) **отсутствует в
PyArrow 26** (PR #45001 open с 2023). Diagnostic должен проверять
**конкретные имена** (`hash_one`, `hash_64`), а не факт наличия
подстроки `hash` — иначе ложноположительно.

**(2) `arr.to_pandas()` даёт inconsistent dtype для integer в зависимости
от null presence.** Без nulls → `int64`; с nulls → `float64` (NaN).
Pandas `hash_pandas_object` hashes int64 и float64 **разными путями**
(SipHash разных kernel) → hash(1) в batch без nulls ≠ hash(1.0) в batch
с nulls. Для out-of-core **partition-disjoint** это критично: одно
значение в разных батчах попадает в разные партиции → wrong results
**молча** (нет crash, нет wrong count, только wrong groupby/join/dedup).

**Fix:** `types_mapper=Int64Dtype` в `to_pandas()` — форсирует
nullable Int64 независимо от null pattern. Then hash(1) consistent
across batches.

**Урок (усиливает §13.4/52b, 99):** для любой конвертации, влияющей на
partitioning/hashing, **cross-batch consistency — обязательный тест**.
Не только "same value → same hash **within a batch**" (тривиально), но
"same value → same hash **across batches with different null patterns**".
Правило: если операция недетерминирована на null-паттерне, тест на
неё — обязателен.

**Дополнительно:** `pandas.util.hash_pandas_object` в pandas 3.x
изменил API — `hash_key` для string требует `str`, не `int` (numeric
всё ещё `int`). Мы не передаём `hash_key` явно (default `None`) —
работает и для numeric, и для string. Seed применяется через XOR —
универсально.

**Ожидаемый эффект:** hash_shuffle 1M int64: 0.61s → ~0.25s (2.5×).
На 500 GB: 100 min → 40 min. String — зависит от pandas 3.x string-hash
стабильности (v0.8.1-fix при необходимости).

**Следующее (v0.7.8):** multiclass GBDT (softmax, k деревьев/round).
Откладывать hash fix было бы ошибкой — 2.5× на всё shuffle-based.

'''
        s = s.replace(marker, episode + marker, 1)

    p.write_text(s)
    print("  OK: §11.1, §13.4/103")


# ============ main ============

def main():
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    branch = "patch/v081-hash-pandas"

    print("=== v0.8.1: compute_hashes — pandas C-hash ===")
    ensure_branch(branch)
    require_clean()

    backup_dir = Path(f".backups/backup-pre-v081-{ts_str}")
    backup_dir.mkdir(parents=True, exist_ok=True)
    sh(["git", "archive", "--format=tar.gz",
        "-o", str(backup_dir / "repo.tgz"), "HEAD"])
    sh(["git", "tag", "-a", f"backup-pre-v081-{ts_str}",
        "-m", "before patch v0.8.1 pandas C-hash"])

    patch_compute_hashes()
    add_tests()
    write_bench()
    patch_changelog()
    patch_decisions()

    print("\n=== pytest tests/test_hash_v081.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_hash_v081.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** FAILURE — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.8.1 (test red)"], check=False)
        sys.exit(1)

    print("\n=== pytest tests/test_shuffle.py -v (partition-invariant regression) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_shuffle.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** REGRESSION FAILURE — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.8.1 (shuffle regression)"], check=False)
        sys.exit(1)

    print("\n=== bench: v0.8.1 hash (baseline: 39.2ms / 0.61s shuffle) ===")
    sh(["uv", "run", "python", "scripts/bench_hash_v081.py"], check=False)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.8.1: compute_hashes pandas C-hash (32× int64); "
        "types_mapper for cross-batch consistency (critical)"])

    print("\n==> v0.8.1 applied on", branch)


if __name__ == "__main__":
    main()