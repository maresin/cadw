#!/usr/bin/env python3
# v0.9.8a-fix3 — test_oversize_triggers_recursion budget vs R1.
#
# Bug: 1 test failure in full fast:
#     test_join_oom_guard.py::test_oversize_triggers_recursion
#     AssertionError: recursive hash_shuffle not called; seeds=[0]
#   Captured log:
#     check_partition_writers FAIL: P=2 × 2048 B/writer = 4096 B > limit=1500 B
#
# Root: test uses MemoryGuard(max_memory_bytes=5000) to force recursion
# via small budget. With new R1 check_partition_writers(P=2) needing
# P×2048 ≤ 0.3×avail → needs budget ≥ 13653. At 5000, top-level
# hash_shuffle fails BEFORE recursion → spy records only seed=0.
# The R1 check is correct; test predates it.
#
# Fix: budget 5000 → 20000; data 100 → 1000 rows per side.
#   - Writers: 2×2048 = 4096 ≤ 0.3×20000 = 6000 ✓ (top-level passes)
#   - Partition size: 1000 rows × 24 B / 2 = ~12 KB > 0.5×20000 = 10 KB
#     → recursion triggers → spy captures seed=1 from recursive call.
#   - Recursive hash_shuffle uses num_partitions_recurse=4 →
#     4×2048 = 8192 > 6000 → raises MemoryLimitExceeded
#     (caught by test try/except; spy already recorded seed=1).
#
# v0.9.8a writes (memory_guard.py + shuffle.py + groupby.py +
# tests/test_partition_writers.py) all on disk. Only this 1 test
# needs fixing before commit.
#
# Run: uv run python scripts/patch_v098a_fix3.py

from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import importlib
import cadw.patch_utils as pu
importlib.reload(pu)
from cadw.patch_utils import (
    assert_compiles, commit_with_guards,
)


# ============ fixes to test_join_oom_guard.py ============

FIXES = [
    # Budget: 5000 → 20000 (R1: P=2 × 2048 = 4096 ≤ 0.3 × 20000 = 6000).
    (
        "    # Маленький бюджет, но max_depth достаточно большой\n"
        "    guard = MemoryGuard(max_memory_bytes=5000)\n",
        "    # v0.9.8a-fix3 (R1): budget must satisfy real per-writer\n"
        "    # check (2×2048 = 4096 ≤ 0.3×budget → budget ≥ 13653).\n"
        "    # Data must still produce oversize partitions for recursion:\n"
        "    # 1000 rows × 24 B / 2 partitions = ~12 KB > 0.5×20000 = 10 KB.\n"
        "    guard = MemoryGuard(max_memory_bytes=20000)\n",
    ),
    # Data: 100 → 1000 rows per side.
    (
        '    left = make_df(tmp_path, "L", {"k": list(range(100)), "a": list(range(100))})\n'
        '    right = make_df(tmp_path, "R", {"k": list(range(100)), "b": list(range(100))})\n',
        '    left = make_df(tmp_path, "L", '
        '{"k": list(range(1000)), "a": list(range(1000))})\n'
        '    right = make_df(tmp_path, "R", '
        '{"k": list(range(1000)), "b": list(range(1000))})\n',
    ),
]


def fix_test():
    print("\n=== test_join_oom_guard.py: R1 budget + data size ===")
    p = Path("tests/test_join_oom_guard.py")
    if not p.exists():
        raise SystemExit(f"{p} missing")
    s = p.read_text()
    original = s
    for old, new in FIXES:
        if old in s:
            s = s.replace(old, new, 1)
            print(f"  fixed: {old.strip()[:70]!r}")
        elif new in s:
            print(f"  already fixed: {new.strip()[:70]!r}")
        else:
            raise SystemExit(
                f"anchor not found:\n{old!r}\nand\n{new!r}"
            )
    if s == original:
        print("  nothing to write — skip")
        return
    p.write_text(s)
    assert_compiles(p)
    print("  written + compiled")


# ============ main ============

def main():
    print("=== v0.9.8a-fix3 ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    fix_test()

    print("\n=== pytest: fixed test + join OOM regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_join_oom_guard.py",
         "tests/test_join_streaming.py",
         "tests/test_partition_writers.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** JOIN OOM / PARTITION WRITERS REGRESSION FAILED ***")
        sys.exit(1)

    print("\n=== pytest fast ===")
    r = subprocess.run(["uv", "run", "pytest", "-q", "-m", "not slow"],
                       timeout=1800)
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)

    print("\n=== pytest slow ===")
    r = subprocess.run(["uv", "run", "pytest", "-q", "-m", "slow"],
                       timeout=900)
    if r.returncode != 0:
        print("\n*** SLOW FAILURE ***")
        sys.exit(1)

    commit_with_guards(
        "v0.9.8a-fix3: test_oversize_triggers_recursion — R1 budget "
        "incompatibility. Test used MemoryGuard(max_memory_bytes=5000) "
        "to force recursion via small budget. New R1 check_partition_"
        "writers(P=2) needs 2×2048 = 4096 ≤ 0.3×avail → budget ≥ 13653. "
        "At 5000, top-level hash_shuffle fails BEFORE recursion → spy "
        "records only seed=0 → assertion any(s >= 1) fails. R1 check "
        "is correct; test predates it. Fix: budget 5000 → 20000; "
        "data 100 → 1000 rows per side. Writers check passes "
        "(2×2048=4096 ≤ 6000); partition ~12 KB > 10 KB (0.5×avail) "
        "→ recursion triggers → spy captures seed=1 → recursive call "
        "itself fails writers (4×2048=8192 > 6000) → caught by test. "
        "v0.9.8a complete.",
        self_path="scripts/patch_v098a_fix3.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.8a complete (fix1+fix2+fix3).")


if __name__ == "__main__":
    main()
