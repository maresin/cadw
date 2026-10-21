#!/usr/bin/env python3
# v0.9.8a-fix2 — §13.4/178 stale module cache in dogfood.
#
# Bug: dogfood failed:
#     AttributeError: 'MemoryGuard' object has no attribute
#     'check_partition_writers'
#   despite method added on disk to cadw/core/memory_guard.py.
#
# Root: §13.4/178 (5th occurrence in session). `MemoryGuard` was
# imported into sys.modules['cadw.core.memory_guard'] before the
# patch modified the file. `from cadw.core.memory_guard import
# MemoryGuard` in dogfood returns the OLD class object (no new method).
# `importlib.reload(pu)` at patch top only reloads patch_utils.
#
# State: all writes on disk (memory_guard.py + shuffle.py +
# groupby.py + tests). dogfood was the only failing step.
# Patch script is idempotent (each step has "already present — skip"),
# so re-run completes dogfood + pytest + commit.
#
# Fix: subprocess-run corrected patch_v098a_partition_writers.py —
# in subprocess, `from cadw.core.memory_guard import MemoryGuard`
# imports the FRESH class (no stale cache). This is the same
# §13.4/184 pattern (subprocess for fresh interpreter) applied to
# the dogfood problem: run the whole patch script as subprocess.
#
# Alternative (in-process reload) is also valid but riskier: dogfood
# would need explicit importlib.reload on memory_guard + shuffle +
# groupby (3 modules just modified). Subprocess is simpler and
# treats the patch script as black box (idempotent).
#
# §13.4/178 recurring (5th: v0.9.5a-fix1, v0.9.6d-fix1,
# v0.9.6d-fix2, v0.9.6e2-fix1, this). Scaffold fix (v0.9.6e2):
# is_patched(fn, *markers). For patch-script own dogfood: use
# subprocess always (v0.9.8d backlog: helper
# `run_patch_in_subprocess(script_path)`).
#
# Run: uv run python scripts/patch_v098a_fix2.py

from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def verify_state():
    print("\n=== verify: writes done, method on disk ===")
    mg = Path("cadw/core/memory_guard.py").read_text()
    if "def check_partition_writers(" not in mg:
        raise SystemExit("check_partition_writers method missing from disk")
    print("  memory_guard.py: check_partition_writers present")

    sh = Path("cadw/ops/shuffle.py").read_text()
    if "check_partition_writers" not in sh:
        raise SystemExit("shuffle.py: call site missing")
    print("  shuffle.py: call site present")

    gb = Path("cadw/ops/groupby.py").read_text()
    if "check_partition_writers" not in gb:
        raise SystemExit("groupby.py: call site missing")
    print("  groupby.py: call site present")

    tp = Path("tests/test_partition_writers.py")
    if not tp.exists():
        raise SystemExit("tests/test_partition_writers.py missing")
    print("  test_partition_writers.py present")


def run_patch_fresh():
    print("\n=== run patch_v098a_partition_writers.py in subprocess ===")
    # Subprocess = fresh interpreter, no stale module cache (§13.4/178).
    # Patch script is idempotent: writes already there → each step
    # logs "already present — skip"; dogfood now sees the fresh
    # MemoryGuard class; pytest + commit run.
    r = subprocess.run(
        ["uv", "run", "python",
         "scripts/patch_v098a_partition_writers.py"],
        cwd=ROOT, timeout=1800,
    )
    if r.returncode != 0:
        print(f"\n*** patch failed (exit {r.returncode}) ***")
        sys.exit(r.returncode)
    print("\n  patch completed successfully")


if __name__ == "__main__":
    print("=== v0.9.8a-fix2 ===")
    verify_state()
    run_patch_fresh()
