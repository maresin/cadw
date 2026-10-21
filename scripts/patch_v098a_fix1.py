#!/usr/bin/env python3
# v0.9.8a-fix1 — remove invalid `if False else None` import artifact.
#
# Bug: patch_v098a_partition_writers.py failed py_compile at line 53:
#     write_content_checked, assert_compiles, commit_to_guards if
#     False else None,
#   SyntaxError: invalid syntax. Same artifact in generated test:
#     'from cadw.patch_utils import assert_call_order if False else None'
#
# Root: leftover `if False else None` artifacts from editing — invalid
# Python in import list. State clean (py_compile failed BEFORE run,
# no files touched). Fix: replace both with valid single-name imports;
# py_compile; subprocess-run corrected patch.
#
# §13.4/201 (same class as /193 /194 /198): patch-script pre-flight
# py_compile catches broken artifacts BEFORE mutation. Recovery =
# line-fix + rerun (this fix1).
#
# Run: uv run python scripts/patch_v098a_fix1.py

from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


FIXES = [
    # (broken, fixed)
    (
        "    write_content_checked, assert_compiles, commit_to_guards "
        "if False else None,\n",
        "    write_content_checked, assert_compiles,\n",
    ),
    (
        "    'from cadw.patch_utils import assert_call_order "
        "if False else None',\n",
        "    'from cadw.patch_utils import assert_call_order',\n",
    ),
]


def fix_broken_lines():
    print("\n=== fix `if False else None` artifacts ===")
    p = Path("scripts/patch_v098a_partition_writers.py")
    if not p.exists():
        raise SystemExit(f"{p} missing — nothing to fix")
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
                f"anchor not found:\n  {old!r}\n  and\n  {new!r}"
            )
    if s == original:
        print("  nothing to write — skip")
    else:
        p.write_text(s)
        print("  written")

    # Compile check.
    import py_compile
    try:
        py_compile.compile(str(p), doraise=True)
    except py_compile.PyCompileError as e:
        print("  COMPILE FAIL:", str(e)[:800])
        raise SystemExit(1)
    print("  py_compile OK")


def run_patch():
    print("\n=== run corrected patch_v098a_partition_writers.py ===")
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
    print("=== v0.9.8a-fix1 ===")
    fix_broken_lines()
    run_patch()
