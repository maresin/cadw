#!/usr/bin/env python3
"""v0.8.1-fix2: hash avalanche + bench sys.path.

Diagnosis (from failed test):
    test_groupby_aggregate_recursion_end_to_end (5000 unique string keys,
    50 KB budget) fails at depth=5: file=36466 B still > safety=25000 B.
    Hash_shuffle at each depth produces ~1 non-empty partition.

Root cause:
    pd.util.hash_pandas_object for object/string dtype shifts hash << 1
    (reserves 0 for NA). All string hashes are EVEN (h & 1 = 0).
    For P=4 (h & 3), only partitions {0, 2} used → 2-way split instead
    of 4-way → recursion at depth=5 gives (1/2)^5 = 1/32 of original,
    not (1/4)^5 = 1/1024 → not enough.

Fix:
    1. **Avalanche** after final hash (splitmix64-style finalizer):
       spreads low bits, removes structure (parity, alignment).
       Applied ALWAYS (not only with seed) — otherwise seed=0 partitions
       are still 2-way.
    2. **Bench** — `sys.path.insert(0, project_root)`.

Why splitmix64 finalizer:
    - Used by Java's SplittableRandom; known properties.
    - 3 lines, vectorized on numpy.
    - Passes avalanche test: 1-bit input change → ~50% output bit changes.
    - After avalanche: P=4 → all 4 partitions used; P=8 → all 8.
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


def patch_avalanche():
    print("\n=== cadw/ops/shuffle.py: compute_hashes + avalanche ===")
    p = Path("cadw/ops/shuffle.py")
    s = p.read_text()

    old = '''    assert h_np is not None
    if seed:
        h_np = h_np ^ np.uint64(seed)
    return pa.array(h_np, type=pa.uint64())'''

    new = '''    assert h_np is not None
    # v0.8.1-fix2: splitmix64 avalanche (ALWAYS, not only when seed).
    # pd.util.hash_pandas_object for object/string dtype shifts << 1
    # (reserves 0 for NA), so all string hashes are EVEN: h & 1 = 0.
    # For P=4 (h & 3) only partitions {0, 2} would be used → 2-way split
    # (not 4-way) → recursion never converges. Avalanche spreads low bits,
    # removing structure. Seed applied via XOR BEFORE avalanche — diffuses
    # cleanly through the finalizer.
    if seed:
        h_np = h_np ^ np.uint64(seed)
    h_np = (h_np ^ (h_np >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
    h_np = (h_np ^ (h_np >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
    h_np = h_np ^ (h_np >> np.uint64(31))
    return pa.array(h_np, type=pa.uint64())'''

    s = replace_exact(s, old, new, tag="avalanche")
    p.write_text(s)
    compile_check(p)
    print("  OK: splitmix64 avalanche applied")


def patch_bench():
    print("\n=== scripts/bench_hash_v081.py: sys.path.insert ===")
    p = Path("scripts/bench_hash_v081.py")
    s = p.read_text()

    old = '''import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa

from cadw.ops.shuffle import compute_hashes

ROOT = Path(__file__).resolve().parent.parent'''

    new = '''import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa

# v0.8.1-fix2: `python scripts/foo.py` puts scripts/ in sys.path[0], not
# project root. Add project root so `import cadw` works.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cadw.ops.shuffle import compute_hashes

ROOT = _PROJECT_ROOT'''

    s = replace_exact(s, old, new, tag="sys-path")
    p.write_text(s)
    compile_check(p)
    print("  OK: sys.path.insert(0, project_root)")


def add_avalanche_tests():
    print("\n=== tests/test_hash_v081.py: +avalanche distribution test ===")
    p = Path("tests/test_hash_v081.py")
    s = p.read_text()

    if "test_avalanche_spreads_low_bits" in s:
        print("  already present — skip")
        return

    extra = '''

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
'''

    s = s.rstrip() + extra + "\n"
    p.write_text(s)
    compile_check(p)
    print("  OK: +4 avalanche tests")


def main():
    print("=== v0.8.1-fix2 ===")
    patch_avalanche()
    patch_bench()
    add_avalanche_tests()

    print("\n=== pytest tests/test_hash_v081.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_hash_v081.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** FAILURE — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.8.1-fix2 (test red)"], check=False)
        sys.exit(1)

    print("\n=== pytest tests/test_mustfix_local_groupby.py -v (original failure) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_mustfix_local_groupby.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** REGRESSION FAILURE — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.8.1-fix2 (mustfix red)"], check=False)
        sys.exit(1)

    print("\n=== bench: v0.8.1 hash (baseline: 39.2ms / 0.61s shuffle) ===")
    sh(["uv", "run", "python", "scripts/bench_hash_v081.py"], check=False)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.8.1-fix2: splitmix64 avalanche (low-bit spread for "
        "pandas-string hash); bench sys.path"]) 

    print("\n==> v0.8.1-fix2 applied")


if __name__ == "__main__":
    main()