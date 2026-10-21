#!/usr/bin/env python3
# v0.9.8c-fix1 — §13.4/178 stale module cache in dogfood (7th time).
#
# Bug: dogfood failed:
#     ValueError: propagate_chars: unsupported op 'hash_shuffle'.
#     Supported: filter, with_column, groupby, sort, reduce, join.
#   despite planner.py updated on disk. Traceback line 184 shows NEW
#   source text (`# N unchanged (partition does not change row count)`)
#   but exception message is OLD (no hash_shuffle in Supported list).
#
# Root: §13.4/178 (7th: v0.9.5a-fix1, v0.9.6d-fix1, v0.9.6d-fix2,
# v0.9.6e2-fix1, v0.9.8a-fix2, v0.9.8b-fix1, this). planner module
# was imported (via patch_utils import chain) before file update.
# Traceback line numbers (co_firstlineno) from OLD bytecode = 184;
# current file line 184 = new comment. Python reads file for display
# (shows new), executes old code (old exception). Diagnostic signature.
#
# State: planner.py writes on disk (5 branches + updated raise);
# tests/test_propagate_chars_ops.py on disk. dogfood was only
# failing step. Patch script idempotent ("already present — skip"
# via is_patched on source contains 'op.name == "hash_shuffle"').
#
# Fix: subprocess re-run.
# §13.4/178 recurring 7× — MUST structurally fix. Two-phase patch
# (write/verify subprocess) is planned v0.9.8d backlog. Immediate
# mitigation: every patch-script dogfood should be subprocess.
#
# Run: uv run python scripts/patch_v098c_fix1.py

from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def verify_state():
    print("\n=== verify: writes done ===")
    pl = Path("cadw/core/planner.py").read_text()
    for anchor in (
        'if op.name == "hash_shuffle":',
        'if op.name == "dedup":',
        'if op.name == "rank":',
        'if op.name == "quantile":',
        'if op.name == "resample":',
        "Supported: filter, with_column, groupby, sort, reduce, join,",
    ):
        if anchor not in pl:
            raise SystemExit(f"planner.py: missing {anchor!r}")
    print("  planner.py: all 5 branches + updated message present")
    tp = Path("tests/test_propagate_chars_ops.py")
    if not tp.exists():
        raise SystemExit("test_propagate_chars_ops.py missing")
    print("  test_propagate_chars_ops.py present")


def run_patch_fresh():
    print("\n=== run patch_v098c_propagate_shuffle.py in subprocess ===")
    r = subprocess.run(
        ["uv", "run", "python",
         "scripts/patch_v098c_propagate_shuffle.py"],
        cwd=ROOT, timeout=1800,
    )
    if r.returncode != 0:
        print(f"\n*** patch failed (exit {r.returncode}) ***")
        sys.exit(r.returncode)
    print("\n  patch completed successfully")


if __name__ == "__main__":
    print("=== v0.9.8c-fix1 ===")
    verify_state()
    run_patch_fresh()
