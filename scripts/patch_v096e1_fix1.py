#!/usr/bin/env python3
# v0.9.6e1-fix1 — fix broken string escape in v0.9.6e1 patch.
#
# The user ran `uv run python scripts/patch_v096e1_pandas_optional.py`
# — py_compile failed at line 252, file never even loaded:
#     \' "format=\'numpy\'"\'|\' "format=\'torch\'"\'". "',
#     ^
#     SyntaxError: unexpected character after line continuation character
#
# Root cause (§13.4/174, /183, /192 — same session, 3rd time for /174):
# NEW_TEST_LINES had a multi-line string concat that tried to escape
# quotes for the message "...use format='arrow'|'numpy'|'torch'".
# Written as:
#     '... format=\'arrow\' "\'|'
#     \' "format=\'numpy\'"\'|\' "format=\'torch\'"\'". "',
# — invalid. Result: file is broken on disk (untracked, no commit).
#
# Fix: line-based replace — find the anchor line containing
# "v0.9.6e1: pandas is optional; use format=", remove it + next line,
# insert single clean string (no nested quotes). Then re-run the
# now-valid patch_v096e1_pandas_optional.py in fresh subprocess.
#
# Rule reinforced: any patch script embedding assertion messages or
# data with `'` `"` `\` must build strings line-by-line with
# `write_content()` (no multi-line Python template). `assert_compiles`
# runs BEFORE any file mutation — fail fast.

from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def fix_broken_string():
    print("\n=== fix NEW_TEST_LINES broken escape ===")
    p = Path("scripts/patch_v096e1_pandas_optional.py")
    if not p.exists():
        raise SystemExit(f"{p} missing — nothing to fix")

    lines = p.read_text().split("\n")
    anchor = "v0.9.6e1: pandas is optional; use format="
    idx = None
    for i, line in enumerate(lines):
        if anchor in line:
            idx = i
            break
    if idx is None:
        raise SystemExit(f"anchor {anchor!r} not found")

    # New single-line replacement (no nested quotes):
    #   '        + ". v0.9.6e1: pandas is optional; use format=arrow/numpy/torch. "',
    # (single-quoted Python string; inside it just a double-quoted
    #  literal `"..."` — no escapes needed).
    new_line = (
        "        '        + \". v0.9.6e1: pandas is optional; "
        "use format=arrow/numpy/torch. \"',"
    )

    # Confirm second line (continuation of broken string) exists.
    if idx + 1 >= len(lines):
        raise SystemExit("broken pair incomplete — only one line at anchor")

    print(f"  line {idx + 1}: {lines[idx]!r}")
    print(f"  line {idx + 2}: {lines[idx + 1]!r}")

    lines[idx : idx + 2] = [new_line]
    p.write_text("\n".join(lines))
    print("  replaced (2 lines -> 1)")

    # Verify syntax before running.
    import py_compile
    try:
        py_compile.compile(str(p), doraise=True)
    except py_compile.PyCompileError as e:
        print("  COMPILE FAIL:", str(e)[:600])
        raise SystemExit(1)
    print("  py_compile OK")


def run_patch():
    print("\n=== run fixed patch_v096e1_pandas_optional.py (fresh) ===")
    r = subprocess.run(
        ["uv", "run", "python", "scripts/patch_v096e1_pandas_optional.py"],
        cwd=ROOT, timeout=1200,
    )
    if r.returncode != 0:
        print(f"\n*** patch_v096e1_pandas_optional.py failed "
              f"(exit {r.returncode}) — inspect output above ***")
        sys.exit(r.returncode)
    print("\n  patch completed successfully")


if __name__ == "__main__":
    print("=== v0.9.6e1-fix1 ===")
    fix_broken_string()
    run_patch()