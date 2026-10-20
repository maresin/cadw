#!/usr/bin/env python3
# v0.9.3a-fix3: repair broken test only. NO triple-quote in this file.
# v0.9.3a-fix1 and -fix2 both had SyntaxError themselves: outer
# triple-single-quote closed early by inner ''' or """. Lesson:
# this file uses NO triple-quote anywhere - # comments only for
# header, single-quote + explicit \n for content strings.
#
# Scope: fix tests/test_cv_keyed_prng.py broken assert only. Commit.
# DECISIONS/CHANGELOG update is a SEPARATE patch (v0.9.3a-docs),
# after green tests. Not mixing broken-fix with docs in one patch.
#
# Run: uv run python scripts/patch_v093a_fix3.py

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def sh(cmd, check=True):
    print("  $ " + (" ".join(cmd) if isinstance(cmd, list) else cmd))
    r = subprocess.run(cmd, shell=isinstance(cmd, str),
                       capture_output=True, text=True)
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines()[:20]:
            print("    " + line)
    if check and r.returncode != 0:
        print("    STDERR: " + r.stderr.strip()[-1500:])
        sys.exit(r.returncode)
    return r.returncode


def compile_check(path):
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         "import py_compile; py_compile.compile('" + str(path) +
         "', doraise=True)"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print("COMPILE FAIL: " + str(path))
        print(r.stderr[-800:])
        sys.exit(1)
    print("  compile OK: " + str(path))


def fix_broken_test():
    print("\n=== tests/test_cv_keyed_prng.py: fix embedded newline ===")
    p = Path("tests/test_cv_keyed_prng.py")
    s = p.read_text()

    # Old block: two physical lines because \n became newline in the
    # original patch template (see v0.9.3a §13.4/174). Match exact.
    old = (
        '    assert "_attach_column_with_fn" in src\n'
        '    assert "_attach_column(df," in src or "_attach_column(\n'
        '" in src\n'
    )

    new = (
        '    assert "_attach_column_with_fn" in src\n'
        '    assert "_attach_column(df," in src\n'
    )

    if new in s and old not in s:
        print("  already fixed - skip")
        return
    if old not in s:
        print("  WARNING: exact block not found; searching variants...")
        # Alternative: the current file may have only partial old line
        # (e.g. if a prior partial run trimmed it). Show what's there.
        idx = s.find("_attach_column")
        while idx >= 0:
            start = max(0, idx - 80)
            end = min(len(s), idx + 120)
            print("  context: ..." + repr(s[start:end]) + "...")
            idx = s.find("_attach_column", idx + 1)
        raise SystemExit(
            "broken block not found in expected form; inspect file "
            "tests/test_cv_keyed_prng.py around 'assert \"_attach_column'"
        )
    s = s.replace(old, new, 1)
    p.write_text(s)
    compile_check(p)
    print("  OK")


def main():
    print("=== v0.9.3a-fix3 (minimal: repair test only) ===")

    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    fix_broken_test()

    print("\n=== pytest: cv keyed prng ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_cv_keyed_prng.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** NEW TESTS FAILED ***")
        sys.exit(1)

    print("\n=== pytest: existing cv tests ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_cv.py",
         "tests/test_cv_e2e.py", "-q"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** EXISTING CV FAILED ***")
        sys.exit(1)

    print("\n=== pytest: guards ===")
    from cadw.patch_utils import assert_guards_green
    assert_guards_green([
        "tests/test_changelog.py",
        "tests/test_arrow_call_memory_guard.py",
        "tests/test_no_pandas_in_ops.py",
        "tests/test_patch_utils.py",
    ])

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

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.9.3a-fix3: repair tests/test_cv_keyed_prng.py broken assert "
        "(embedded newline from patch template); tests green"])

    print("\n==> v0.9.3a-fix3 applied. Tests green.")
    print("NEXT: v0.9.3a-docs (DECISIONS §13.4/173+174, CHANGELOG entry)")


if __name__ == "__main__":
    main()
