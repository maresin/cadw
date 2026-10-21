#!/usr/bin/env python3
# v0.9.7a-fix1 — patch_v097a_topk.py: unterminated string in
# NEW_TEST_LINES (line 592).
#
# Bug:
#     '    """Facade: nlargest calls topk_to_disk; no external_sort."""
#                                                              ^
#   missing closing "'" + "," at end of line. Content is `'    """..."""'`
#   — outer Python string never closed. py_compile failed BEFORE run
#   → no mutation to cadw/ or tests/. State clean.
#
# Fix: append "'," to that line; py_compile; subprocess-run the
# corrected patch_v097a_topk.py.
#
# §13.4/174 (4th repeat: escape trap in patch-script test-content
# template). write_content_checked protects the WRITTEN file, but the
# patch script itself must compile before main() runs — the pre-flight
# py_compile in `cat > ... << PYEOF\nuv run python -c "import py_compile;
# py_compile.compile(...)"\nuv run python scripts/patch_X.py` catches
# this case. This fix1 is the recovery: broken file → fix line →
# run.
#
# Run: uv run python scripts/patch_v097a_fix1.py

from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def fix_broken_line():
    print("\n=== fix unterminated string in patch_v097a_topk.py ===")
    p = Path("scripts/patch_v097a_topk.py")
    if not p.exists():
        raise SystemExit(f"{p} missing — nothing to fix")

    lines = p.read_text().split("\n")
    anchor = "Facade: nlargest calls topk_to_disk; no external_sort."
    idx = None
    for i, line in enumerate(lines):
        if anchor in line:
            idx = i
            break
    if idx is None:
        raise SystemExit(f"anchor {anchor!r} not found")

    line = lines[idx]
    print(f"  line {idx + 1}: {line!r}")
    stripped = line.rstrip()
    if stripped.endswith("',"):
        print("  already closed — skip")
        return
    # Should end with `"""`; append closing `'` and comma.
    if not stripped.endswith('"""'):
        raise SystemExit(
            f"unexpected line ending: {stripped[-10:]!r} "
            f"(expected `\"\"\"`)"
        )
    lines[idx] = stripped + "',"
    print(f"  fixed line:  {lines[idx]!r}")
    p.write_text("\n".join(lines))
    print("  written")

    # Verify compile.
    import py_compile
    try:
        py_compile.compile(str(p), doraise=True)
    except py_compile.PyCompileError as e:
        print("  COMPILE FAIL:", str(e)[:800])
        raise SystemExit(1)
    print("  py_compile OK")


def run_patch():
    print("\n=== run corrected patch_v097a_topk.py (fresh subprocess) ===")
    r = subprocess.run(
        ["uv", "run", "python", "scripts/patch_v097a_topk.py"],
        cwd=ROOT, timeout=1800,
    )
    if r.returncode != 0:
        print(f"\n*** patch failed (exit {r.returncode}) — inspect above ***")
        sys.exit(r.returncode)
    print("\n  patch completed successfully")


if __name__ == "__main__":
    print("=== v0.9.7a-fix1 ===")
    fix_broken_line()
    run_patch()
