#!/usr/bin/env python3
# v0.9.8b-fix1 — §13.4/178 stale module cache in dogfood (6th time).
#
# Bug: dogfood failed:
#     TypeError: MemoryGuard.get_batch_rows() got an unexpected keyword
#     argument 'sample_batch'
#   despite get_batch_rows updated on disk with `*, sample_batch=None`.
#
# Root: §13.4/178 (6th: v0.9.5a-fix1, v0.9.6d-fix1, v0.9.6d-fix2,
# v0.9.6e2-fix1, v0.9.8a-fix2, this). MemoryGuard was imported at
# patch-script's top (via `import cadw.patch_utils as pu`? No — via
# earlier `from cadw.dataframe import ...`? Actually: patch script
# loaded `cadw.core.memory_guard` at the first import chain from
# `cadw.patch_utils`). Then update_get_batch_rows() modified file on
# disk. In-process MemoryGuard is the OLD class object.
#
# State: all writes on disk (memory_guard.py: docstring + 2 methods +
# kwarg; tests/test_row_size_reestimate.py). Only dogfood was failing.
# Patch script is idempotent ("already updated/present — skip").
#
# Fix: subprocess-run corrected patch_v098b_row_size_reestimate.py —
# fresh interpreter sees fresh MemoryGuard class. Same pattern as
# v0.9.8a-fix2 (§13.4/202 subprocess as universal fix).
#
# v0.9.8d backlog reinforced: helper `run_patch_in_subprocess`. Every
# patch-script's own dogfood should run in subprocess, not in-process.
#
# Run: uv run python scripts/patch_v098b_fix1.py

from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def verify_state():
    print("\n=== verify: writes done ===")
    mg = Path("cadw/core/memory_guard.py").read_text()
    for anchor in (
        "WARNING (v0.9.8b, R2)",
        "def reestimate_from_batch(",
        "def estimate_row_size_from_schema_with_sample(",
        "sample_batch: Optional[pa.RecordBatch] = None,",
    ):
        if anchor not in mg:
            raise SystemExit(f"memory_guard.py: missing {anchor!r}")
    print("  memory_guard.py: all anchors present")
    tp = Path("tests/test_row_size_reestimate.py")
    if not tp.exists():
        raise SystemExit("test_row_size_reestimate.py missing")
    print("  test_row_size_reestimate.py present")


def run_patch_fresh():
    print("\n=== run patch_v098b_row_size_reestimate.py in subprocess ===")
    r = subprocess.run(
        ["uv", "run", "python",
         "scripts/patch_v098b_row_size_reestimate.py"],
        cwd=ROOT, timeout=1800,
    )
    if r.returncode != 0:
        print(f"\n*** patch failed (exit {r.returncode}) ***")
        sys.exit(r.returncode)
    print("\n  patch completed successfully")


if __name__ == "__main__":
    print("=== v0.9.8b-fix1 ===")
    verify_state()
    run_patch_fresh()
