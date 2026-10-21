#!/usr/bin/env python3
# v0.9.6c2-fix4 — test_pivot_to_disk_guarded_before_output: missing import.
#
# fix2 added new contract test using assert_call_order (dogfood
# scaffold), but forgot the import. Project convention: helper
# imports inside test functions (see test_apply.py, test_pivot_
# arrow_native.py) — not module-level, to keep test module import
# fast and avoid circulars.
#
# Fix: add `from cadw.patch_utils import assert_call_order` inside
# the test. One-line change.
#
# Run: uv run python scripts/patch_v096c2_fix4.py

from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import assert_compiles, commit_with_guards


def fix_missing_import():
    print("\n=== test_pivot_table.py: add assert_call_order import ===")
    p = Path("tests/test_pivot_table.py")
    s = p.read_text()

    old = (
        '    import inspect\n'
        '\n'
        '    from cadw.ops.pivot import pivot_to_disk\n'
        '\n'
        '    src = inspect.getsource(pivot_to_disk)\n'
        '    assert_call_order(\n'
    )
    new = (
        '    import inspect\n'
        '\n'
        '    from cadw.patch_utils import assert_call_order\n'
        '    from cadw.ops.pivot import pivot_to_disk\n'
        '\n'
        '    src = inspect.getsource(pivot_to_disk)\n'
        '    assert_call_order(\n'
    )
    if new in s:
        print("  already imported — skip")
        return
    if old not in s:
        raise SystemExit("test anchor not found")
    s = s.replace(old, new, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  added import")


def main():
    print("=== v0.9.6c2-fix4 ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    fix_missing_import()

    print("\n=== pytest: pivot regression + new tests ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_pivot_table.py",
         "tests/test_pivot_table_e2e.py",
         "tests/test_pivot_arrow_native.py", "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** PIVOT REGRESSION FAILED ***")
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
        "v0.9.6c2-fix4: test_pivot_to_disk_guarded_before_output — "
        "missing import of assert_call_order. fix2 added new contract "
        "test using patch_utils.assert_call_order (scaffold dogfood), "
        "but forgot the import. Project convention: helper imports "
        "inside test functions (see test_apply.py, test_pivot_arrow_"
        "native.py). One-line fix: add `from cadw.patch_utils import "
        "assert_call_order`. v0.9.6c2 complete.",
        self_path="scripts/patch_v096c2_fix4.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.6c2 complete (fix1+fix2+fix3+fix4).")


if __name__ == "__main__":
    main()