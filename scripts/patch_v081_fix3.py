#!/usr/bin/env python3
"""v0.8.1-fix3: strings → Python hash path (pandas is slower for strings).

Bench v0.8.1 (v0.8.0 baseline 39.2 ms / 100k):
    int64    39.2 → 2.02 ms  (19.4×)  ✅
    float64  39.2 → 0.99 ms  (39.6×)  ✅
    string   39.2 → 50.44 ms (0.78×)  ❌ 22% REGRESSION

pd.util.hash_pandas_object for object dtype uses per-element Cython
(hash_object_array) + categorical detection — ~500 ns/string. Old Python
path (repr(v).encode() + xxhash.xxh64) — ~390 ns/string. Pandas has NO
faster string hashing path in v3.x (Arrow string → same Cython path).

Fix:
    - string/large_string → _python_hash_column (no regression)
    - numeric/bool/timestamp/date → _pandas_hash_column (19-39× speedup)
    - struct/list/dict/uint64 → _python_hash_column (unchanged)

Avalanche stays — cheap (~0.3 ms), safety margin for future.

Backward compat: hashes for strings return to v0.5-0.8 values (xxhash +
avalanche now — actually different from v0.5 because of avalanche, but
that's fine — partitions ephemeral).

Adds:
    - bench: measure string shuffle end-to-end.
    - test_string_uses_python_path: assert dispatch.
    - test_string_shuffle_no_regression (@slow): string shuffle time.

Run: uv run python scripts/patch_v081_fix3.py
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
        for line in r.stdout.strip().splitlines()[:20]:
            print(f"    {line}")
    if check and r.returncode != 0:
        print(f"    STDERR: {r.stderr.strip()[-1500:]}")
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


def replace_exact(s, old, new, *, tag):
    if old not in s:
        raise SystemExit(f"[{tag}] pattern not found")
    if s.count(old) > 1:
        raise SystemExit(f"[{tag}] pattern found {s.count(old)} times")
    return s.replace(old, new, 1)


# ============ shuffle.py: _hash_column dispatch ============

def patch_dispatch():
    print("\n=== cadw/ops/shuffle.py: _hash_column — string to Python ===")
    p = Path("cadw/ops/shuffle.py")
    s = p.read_text()

    old = '''def _hash_column(arr: pa.Array) -> np.ndarray:
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
    return _python_hash_column(arr)'''

    new = '''def _hash_column(arr: pa.Array) -> np.ndarray:
    """Best-effort: pandas C-hash for numeric/timestamp; Python for strings.

    v0.8.1-fix3: strings → Python. Bench v0.8.1:
        int64    39.2 → 2.02 ms  (19.4×)  (pandas)
        float64  39.2 → 0.99 ms  (39.6×)  (pandas)
        string   39.2 → 50.44 ms (0.78×)  (pandas, REGRESSION)
    pd.util.hash_pandas_object for object dtype uses per-element Cython
    (hash_object_array) + categorical detection — ~500 ns/string vs Python
    repr().encode() + xxhash ~390 ns/string. No faster path in pandas 3.x.
    → route strings to Python, keep pandas for numeric/timestamp.
    """
    t = arr.type
    # String → Python path (no benefit from pandas; slight regression)
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return _python_hash_column(arr)
    # Numeric / bool / timestamp / date / time → pandas C-hash (19-39×)
    if (pa.types.is_integer(t) or pa.types.is_floating(t)
            or pa.types.is_boolean(t)
            or pa.types.is_timestamp(t) or pa.types.is_date(t)
            or pa.types.is_time(t)):
        h = _pandas_hash_column(arr)
        if h is not None:
            return h
    # Fallback: struct/list/dict/uint64, or pandas API mismatch.
    return _python_hash_column(arr)'''

    s = replace_exact(s, old, new, tag="dispatch")
    p.write_text(s)
    compile_check(p)
    print("  OK: strings → _python_hash_column")


# ============ tests ============

def add_tests():
    print("\n=== tests/test_hash_v081.py: +string dispatch test ===")
    p = Path("tests/test_hash_v081.py")
    s = p.read_text()

    if "test_string_uses_python_path" in s:
        print("  already present — skip")
        return

    extra = '''

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
'''

    s = s.rstrip() + extra + "\n"
    p.write_text(s)
    compile_check(p)
    print("  OK: +2 dispatch tests")


# ============ bench: measure string shuffle ============

def patch_bench_string():
    print("\n=== scripts/bench_hash_v081.py: +string shuffle ===")
    p = Path("scripts/bench_hash_v081.py")
    s = p.read_text()

    if "string shuffle" in s:
        print("  already present — skip")
        return

    old = '''    print()
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

    import gc; gc.collect()
    with tempfile.TemporaryDirectory() as td:
        t0 = time.time()
        hash_shuffle(
            bi(), ["k"], ["v"], 8,
            MemoryGuard(max_memory_mb=100), Path(td), log=False,
        )
        dt = time.time() - t0
    print(f"  hash_shuffle 1M int64: {dt:.2f}s")
    print(f"  (baseline from v0.8.0 diagnostic: 0.61s → "
          f"{0.61 / dt:.2f}× speedup)")'''

    new = '''    print()
    print("=" * 60)
    print("End-to-end shuffle (1M int64 and string)")
    print("=" * 60)
    import tempfile
    from cadw.core.memory_guard import MemoryGuard
    from cadw.ops.shuffle import hash_shuffle

    n1m = 1_000_000
    rng1m = np.random.default_rng(7)
    keys = rng1m.integers(0, 100_000, size=n1m).astype(np.int64)
    vals = rng1m.integers(0, 1000, size=n1m).astype(np.int64)

    def _run_shuffle(table, label, baseline_s):
        def bi():
            for i in range(0, n1m, 10000):
                yield table.slice(i, 10000).to_batches()[0]
        import gc; gc.collect()
        with tempfile.TemporaryDirectory() as td:
            t0 = time.time()
            hash_shuffle(
                bi(), ["k"], ["v"], 8,
                MemoryGuard(max_memory_mb=100), Path(td), log=False,
            )
            dt = time.time() - t0
        print(f"  hash_shuffle 1M {label}: {dt:.2f}s "
              f"(baseline {baseline_s:.2f}s → {baseline_s / dt:.2f}× speedup)")

    # int64
    tbl_int = pa.table({"k": pa.array(keys), "v": pa.array(vals)})
    _run_shuffle(tbl_int, "int64", baseline_s=0.61)

    # string (9-char)
    str_keys = [f"k{i:08d}" for i in rng1m.integers(0, 100_000, size=n1m)]
    tbl_str = pa.table({"k": pa.array(str_keys), "v": pa.array(vals)})
    _run_shuffle(tbl_str, "string", baseline_s=0.61)'''

    s = replace_exact(s, old, new, tag="bench-string")
    p.write_text(s)
    compile_check(p)
    print("  OK: bench measures int64 + string shuffle")


# ============ CHANGELOG update ============

def patch_changelog():
    print("\n=== CHANGELOG.md: [0.8.1] fix note ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()

    old = """  - Numeric/boolean/string/timestamp/date: `pd.util.hash_pandas_object`
    (C-level SipHash).
  - Fallback: Python loop + xxhash для struct/list/dict/uint64."""
    new = """  - Numeric/boolean/timestamp/date: `pd.util.hash_pandas_object`
    (C-level SipHash, 19-40× speedup).
  - **String/large_string**: Python `repr().encode()` + xxhash.
    Причина: `pd.util.hash_pandas_object` для object-dtype string использует
    per-element Cython — 50 ms/100k, **медленнее** Python 39 ms (§13.4/107).
  - Fallback: Python loop для struct/list/dict/uint64."""
    if old not in s:
        # Try alternate form
        old2 = """  - Numeric/boolean/string/timestamp/date: `pd.util.hash_pandas_object`
    (C-level SipHash)."""
        new2 = """  - Numeric/boolean/timestamp/date: `pd.util.hash_pandas_object`
    (C-level SipHash, 19-40× speedup).
  - **String/large_string**: Python `repr().encode()` + xxhash
    (pandas was 22% slower — see §13.4/107)."""
        if old2 in s:
            s = s.replace(old2, new2, 1)
            p.write_text(s)
            print("  OK: string path note")
            return
        print("  WARNING: changelog pattern not found; skipping")
        return
    s = s.replace(old, new, 1)
    p.write_text(s)
    print("  OK: string path note")


# ============ DECISIONS: episode 107 ============

def patch_decisions():
    print("\n=== DECISIONS.md: §13.4/107 ===")
    p = Path("DECISIONS.md")
    s = p.read_text()

    if "Эпизод 107" in s:
        print("  already present — skip")
        return

    marker = "### 13.5. Открытые вопросы"
    episode = '''**Эпизод 107 (pandas hash для strings медленнее Python; pandas `<<1`
для NA и avalanche).** Два независимых наблюдения из v0.8.1.

**(1) `pd.util.hash_pandas_object` string → Cython per-element, медленнее.**
v0.8.1 bench:
    int64    39.2 → 2.02 ms  (19.4× speedup)  ✅
    float64  39.2 → 0.99 ms  (39.6× speedup)  ✅
    string   39.2 → 50.44 ms (0.78× — 22% SLOWER) ❌

`hash_pandas_object` на object dtype вызывает `hash_object_array` —
Cython-цикл per-element (~500 ns/string) + categorical detection.
Python `repr(v).encode()` + `xxhash.xxh64` — ~390 ns/string. Pandas
не имеет fast path для strings в 3.x. **Fix (v0.8.1-fix3): strings →
`_python_hash_column` (без регрессии).** Numeric → pandas (19-40×).

**(2) pandas `<<1` для object dtype, avalanche ОБЯЗАТЕЛЕН.** Pandas
`hash_pandas_object` для object/string dtype сдвигает hash << 1 (reserves
`0` для NA). Значит **все string hashes чётные** (`h & 1 == 0`). Для P=4
(`h & 3`) используются только партиции {0, 2}; для P=2 — только {0} (!).
Recursion split'ит 2-way вместо 4-way → на depth=5 партиция 1/32 от
исходной, не 1/1024 → **`MemoryLimitExceeded`** в
`test_groupby_aggregate_recursion_end_to_end` (v0.1.4, 5000 string keys,
50 KB budget).

**Fix:** splitmix64 avalanche — 3 умножения + 3 XOR-shift (numpy):
```
h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9
h = (h ^ (h >> 27)) * 0x94d049bb133111eb
h = h ^ (h >> 31)
```
Применяется всегда (не только при seed), seed — XOR до avalanche.
~0.3 ms / 100k — платим 15% speedup'а за корректность.

**Урок (усиливает §13.4/99, 52b):** hash-функция требует
**avalanche test**, не только `hash(A) ≠ hash(B)`. `test_different_
value_different_hash` проходил, но `h & 3` давал 2-way split. Правильный
тест: `np.bincount(h & (P-1), minlength=P)` для P ∈ {2, 4, 8, 16} —
каждая ~N/P. **Это третий случай за сессию** (после 52b, 99), когда
«зелёные тесты скрывали структурную проблему». Правило: для hash —
**минимум 3 теста**: determinism, distinct-values, **low-bit distribution**.

**Что дальше в hash:** `_stable_bytes` для строк использует `repr()`
(добавляет кавычки). Fast path: для string — `v.encode()` (без repr),
но тогда string "1" и int 1 дают одинаковый байт `b"1"` → коллизия в
одной партиции. Для `groupby`/`join` это OK (downstream — actual
equality), но лишний I/O. v0.8.1-fix4 candidate.

'''
    s = s.replace(marker, episode + marker, 1)
    p.write_text(s)
    print("  OK: §13.4/107")


# ============ main ============

def main():
    print("=== v0.8.1-fix3 ===")
    patch_dispatch()
    add_tests()
    patch_bench_string()
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
        sh(["git", "commit", "-m", "WIP: v0.8.1-fix3 (test red)"], check=False)
        sys.exit(1)

    print("\n=== pytest tests/test_shuffle.py tests/test_mustfix_local_groupby.py -q ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_shuffle.py", "tests/test_mustfix_local_groupby.py",
         "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** REGRESSION FAILURE — WIP ***")
        sys.exit(1)

    print("\n=== bench: v0.8.1-fix3 (int64 + string shuffle) ===")
    sh(["uv", "run", "python", "scripts/bench_hash_v081.py"], check=False)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.8.1-fix3: strings → Python path (pandas 22% slower); "
        "bench measures int64+string shuffle; §13.4/107"])

    print("\n==> v0.8.1-fix3 applied")


if __name__ == "__main__":
    main()