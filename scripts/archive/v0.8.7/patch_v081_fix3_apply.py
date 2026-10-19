#!/usr/bin/env python3
"""v0.8.1-fix3-apply: доложить патч после failure на bench pattern.

patch_v081_fix3.py упал на `patch_bench_string()` (pattern mismatch);
до этого dispatch + tests уже применились. Это idempotent recovery:
1. Verify: shuffle.py `_hash_column` содержит dispatch string→python.
2. Full rewrite bench_hash_v081.py (не pattern — whole file).
3. CHANGELOG + DECISIONS (idempotent — skip if already present).
4. Run tests.
5. Commit.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)


def sh(cmd, check=True):
    print(f"  $ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    r = subprocess.run(
        cmd, shell=isinstance(cmd, str), capture_output=True, text=True,
    )
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines()[:25]:
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


# ============ Step 1: verify shuffle.py ============

def verify_shuffle():
    print("\n=== Verify shuffle.py state ===")
    p = Path("cadw/ops/shuffle.py")
    s = p.read_text()

    # Marker: строка / comment про "strings → python"
    if "strings → Python path" not in s and "strings → python" not in s.lower():
        # Check via actual code structure
        if "if pa.types.is_string(t) or pa.types.is_large_string(t):\n        return _python_hash_column(arr)" not in s:
            raise SystemExit(
                "FAIL: shuffle.py не содержит dispatch string→python. "
                "fix3 не применился или был откачен. Нужен полный re-run "
                "patch_v081_fix3.py (только dispatch)."
            )
    print("  OK: strings → _python_hash_column (verified)")
    print("  OK: numeric → _pandas_hash_column (implicit from code)")


# ============ Step 2: full rewrite bench ============

BENCH_SRC = '''"""v0.8.1-fix3 bench: compute_hashes + end-to-end shuffle (int64 + string).

Baselines (v0.8.0 diagnostic, 39.2 ms / 100k compute_hashes; 0.61s shuffle):
    int64      → expect ~2 ms   (pandas C-hash, 19×)
    float64    → expect ~1 ms   (pandas C-hash, 39×)
    string     → expect ~39 ms  (Python path, no change — v0.8.1-fix3)
    shuffle int64 1M → expect ~0.22s (2.76×)
    shuffle string 1M → expect ~0.61s (no change)

Run: uv run python scripts/bench_hash_v081.py
"""
from __future__ import annotations

import gc
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow as pa

# `python scripts/foo.py` puts scripts/ in sys.path[0], not project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cadw.core.memory_guard import MemoryGuard
from cadw.ops.shuffle import compute_hashes, hash_shuffle


def bench_compute_hashes(label, arr, n_warmup=2, n_iter=5):
    b = pa.RecordBatch.from_arrays([arr], names=["k"])
    for _ in range(n_warmup):
        compute_hashes(b, ["k"], seed=0)
    t0 = time.time()
    for _ in range(n_iter):
        compute_hashes(b, ["k"], seed=0)
    dt = (time.time() - t0) / n_iter
    print(f"  [{label}] {dt * 1000:7.2f} ms per 100k rows")
    return dt


def run_shuffle(table, label, num_rows, baseline_s):
    def bi():
        for i in range(0, num_rows, 10000):
            yield table.slice(i, 10000).to_batches()[0]
    gc.collect()
    with tempfile.TemporaryDirectory() as td:
        t0 = time.time()
        hash_shuffle(
            bi(), ["k"], ["v"], 8,
            MemoryGuard(max_memory_mb=100), Path(td), log=False,
        )
        dt = time.time() - t0
    print(f"  hash_shuffle 1M {label:7s}: {dt:.2f}s  "
          f"(baseline {baseline_s:.2f}s → {baseline_s / dt:.2f}× speedup)")
    return dt


def main():
    n = 100_000
    rng = np.random.default_rng(0)

    print("=" * 60)
    print("compute_hashes v0.8.1-fix3 (pandas numeric, python string)")
    print("=" * 60)

    int_arr = pa.array(rng.integers(0, 100_000, size=n).astype(np.int64))
    t_int = bench_compute_hashes("int64", int_arr)

    str_arr = pa.array([f"k{i % 50000:08d}" for i in range(n)])
    t_str = bench_compute_hashes("string", str_arr)

    float_arr = pa.array(rng.normal(size=n))
    t_float = bench_compute_hashes("float64", float_arr)

    print()
    print("=" * 60)
    print("Baseline (v0.8.0 diagnostic: 39.2 ms / 100k)")
    print("=" * 60)
    baseline_ms = 39.2
    for name, t in (("int64", t_int), ("string", t_str), ("float64", t_float)):
        speedup = baseline_ms / (t * 1000)
        mark = "OK" if speedup >= 1.5 else ("~same" if speedup >= 0.9 else "REGRESSION")
        print(f"  {name:7s}: baseline {baseline_ms:.1f} ms → "
              f"{t * 1000:.1f} ms  ({speedup:.2f}×) [{mark}]")

    print()
    print("=" * 60)
    print("End-to-end shuffle (1M rows, 8 partitions)")
    print("=" * 60)
    n1m = 1_000_000
    rng1m = np.random.default_rng(7)
    keys = rng1m.integers(0, 100_000, size=n1m).astype(np.int64)
    vals = rng1m.integers(0, 1000, size=n1m).astype(np.int64)

    tbl_int = pa.table({"k": pa.array(keys), "v": pa.array(vals)})
    run_shuffle(tbl_int, "int64", n1m, baseline_s=0.61)

    str_keys = [f"k{i:08d}" for i in rng1m.integers(0, 100_000, size=n1m)]
    tbl_str = pa.table({"k": pa.array(str_keys), "v": pa.array(vals)})
    run_shuffle(tbl_str, "string", n1m, baseline_s=0.61)


if __name__ == "__main__":
    main()
'''


def rewrite_bench():
    print("\n=== scripts/bench_hash_v081.py: full rewrite ===")
    p = Path("scripts/bench_hash_v081.py")
    p.write_text(BENCH_SRC)
    compile_check(p)
    print("  OK: bench rewritten")


# ============ Step 3: CHANGELOG (idempotent) ============

def patch_changelog():
    print("\n=== CHANGELOG.md ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()
    if "pandas was 22% slower" in s or "pandas is 22% slower" in s:
        print("  already present — skip")
        return

    old = """  - Numeric/boolean/string/timestamp/date: `pd.util.hash_pandas_object`
    (C-level SipHash)."""
    new = """  - Numeric/boolean/timestamp/date: `pd.util.hash_pandas_object`
    (C-level SipHash, 19-40× speedup).
  - **String/large_string**: Python `repr().encode()` + xxhash
    (pandas was 22% slower for object dtype — see §13.4/107)."""
    if old not in s:
        print("  WARNING: changelog base pattern not found — skipping")
        return
    s = replace_exact(s, old, new, tag="changelog-string")
    p.write_text(s)
    print("  OK: string path note")


# ============ Step 4: DECISIONS/107 (idempotent) ============

def patch_decisions():
    print("\n=== DECISIONS.md: §13.4/107 ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "Эпизод 107" in s:
        print("  already present — skip")
        return

    marker = "### 13.5. Открытые вопросы"
    episode = '''**Эпизод 107 (pandas hash для strings медленнее Python; pandas `<<1`
для NA; avalanche).** Два независимых наблюдения из v0.8.1.

**(1) `pd.util.hash_pandas_object` string → Cython per-element, медленнее.**
v0.8.1 bench:
    int64    39.2 → 2.02 ms  (19.4× speedup)
    float64  39.2 → 0.99 ms  (39.6× speedup)
    string   39.2 → 50.44 ms (0.78× — 22% SLOWER)

`hash_pandas_object` на object dtype вызывает `hash_object_array` —
Cython-цикл per-element (~500 ns/string) + categorical detection.
Python `repr(v).encode()` + `xxhash.xxh64` — ~390 ns/string. Pandas
не имеет fast path для strings в 3.x. **Fix (v0.8.1-fix3): strings →
`_python_hash_column`.** Numeric → pandas.

**(2) pandas `<<1` для object dtype; avalanche ОБЯЗАТЕЛЕН.**
`hash_pandas_object` для object/string dtype сдвигает hash << 1
(reserves `0` для NA) → **все string hashes чётные** (`h & 1 == 0`).
Для P=4 (`h & 3`) используются только партиции {0, 2}; для P=2 —
только {0}. Recursion split'ит 2-way вместо 4-way → на depth=5 партиция
1/32 от исходной (не 1/1024) → **`MemoryLimitExceeded`** в
`test_groupby_aggregate_recursion_end_to_end`.

**Fix:** splitmix64 avalanche (3 умножения + 3 XOR-shift, numpy) —
применяется всегда, seed XOR до avalanche, ~0.3 ms/100k.
`(h ^ (h>>30)) * C1; (h ^ (h>>27)) * C2; h ^ (h>>31)` — стандартный
финализатор Java SplittableRandom.

**Урок (усиливает §13.4/52b, 99, 106):** hash-функция требует
**low-bit distribution test**, не только `different value → different
hash`. `test_different_value_different_hash` проходил, но `h & 3` давал
2-way split. Правильный тест: `np.bincount(h & (P-1), minlength=P)` для
P ∈ {2,4,8,16} — каждая ~N/P. **Третий случай за сессию**, когда
«зелёные тесты скрывали структурную проблему».

**Правило hash-tests: минимум 3** — determinism, distinct-values,
**low-bit distribution**.

'''
    s = s.replace(marker, episode + marker, 1)
    p.write_text(s)
    print("  OK: §13.4/107")


# ============ main ============

def main():
    print("=== v0.8.1-fix3-apply ===")
    verify_shuffle()
    rewrite_bench()
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
        sh(["git", "commit", "-m", "WIP: v0.8.1-fix3-apply (test red)"], check=False)
        sys.exit(1)

    print("\n=== regression: shuffle + mustfix_local_groupby ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_shuffle.py",
         "tests/test_mustfix_local_groupby.py",
         "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** REGRESSION FAILURE ***")
        sys.exit(1)

    print("\n=== bench: v0.8.1-fix3 (int64 + string) ===")
    sh(["uv", "run", "python", "scripts/bench_hash_v081.py"], check=False)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.8.1-fix3: strings → Python hash path (no regression); "
        "bench int64+string; §13.4/107"])

    print("\n==> v0.8.1-fix3 complete.")


if __name__ == "__main__":
    main()