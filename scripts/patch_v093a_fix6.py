#!/usr/bin/env python3
# v0.9.3a-fix6: update 2 stale tests in test_cv.py (pre-v0.9.3a).
#
# v0.9.3a changed KFold(shuffle=True) default to keyed PRNG (binomial
# fold sizes). Two tests written for v0.9.2e still assert
# [20,20,20,20,20] exact sizes -> fail [21,16,20,19,24].
#
# Class: §13.4/160 (stale tests after behavior change). Rule: grep ALL
# tests for behavior-change markers (here: "KFold.*shuffle=True") before
# commit. I missed this in v0.9.3a patch.
#
# Fix:
#   - test_kfold_shuffle_true_same_sizes: intent was "exact-balanced
#     shuffle works" -> use materialize=True.
#   - test_kfold_shuffle_different_assignment: intent was "different
#     seeds differ" -> works with keyed PRNG; drop exact-sizes assert,
#     keep sum + union invariant.
#
# Commit: v0.9.3a code+tests. DECISIONS/CHANGELOG -> separate patch
# (v0.9.3a-docs), to avoid any docs-template escape traps again.
#
# Run: uv run python scripts/patch_v093a_fix6.py

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


def replace_line_block(s, old_block, new_block, *, tag):
    old_norm = [l.rstrip() for l in old_block.split("\n")]
    lines = s.split("\n")
    starts = []
    for i in range(len(lines) - len(old_norm) + 1):
        if all(lines[i + k].rstrip() == old_norm[k]
               for k in range(len(old_norm))):
            starts.append(i)
    if not starts:
        print("[" + tag + "] block not found. Context of nearby match:")
        # Show first 5 lines with "def " prefix that contain tag word
        for i, l in enumerate(lines):
            if "def " in l and tag.lower() in l.lower():
                end = min(len(lines), i + 15)
                for j in range(i, end):
                    print("    " + repr(lines[j]))
                break
        raise SystemExit("[" + tag + "] block not found")
    if len(starts) > 1:
        raise SystemExit("[" + tag + "] block found " + str(len(starts)) +
                         " times")
    start = starts[0]
    new_lines = new_block.split("\n")
    lines = lines[:start] + new_lines + lines[start + len(old_norm):]
    return "\n".join(lines)


def fix_test_shuffle_true_same_sizes():
    print("\n=== test_cv.py: test_kfold_shuffle_true_same_sizes ===")
    p = Path("tests/test_cv.py")
    s = p.read_text()

    old = (
        'def test_kfold_shuffle_true_same_sizes(tmp_path):\n'
        '    df = make_xy(tmp_path, n=100)\n'
        '    cv = KFold(n_splits=5, shuffle=True, random_state=42)\n'
        '    sizes = [test.count() for _, _, test in cv.split(df)]\n'
        '    assert sizes == [20, 20, 20, 20, 20]'
    )

    new = (
        'def test_kfold_shuffle_true_same_sizes(tmp_path):\n'
        '    # v0.9.3a: shuffle=True default is keyed PRNG (binomial\n'
        '    # fold sizes, ~mean n/k, sigma ~= sqrt(n/k)). For\n'
        '    # exact-balanced, use materialize=True.\n'
        '    df = make_xy(tmp_path, n=100)\n'
        '    cv = KFold(n_splits=5, shuffle=True, random_state=42,\n'
        '               materialize=True)\n'
        '    sizes = [test.count() for _, _, test in cv.split(df)]\n'
        '    assert sizes == [20, 20, 20, 20, 20]'
    )

    s = replace_line_block(s, old, new, tag="shuffle_true_same_sizes")
    p.write_text(s)
    compile_check(p)
    print("  OK")


def fix_test_shuffle_different_assignment():
    print("\n=== test_cv.py: test_kfold_shuffle_different_assignment ===")
    p = Path("tests/test_cv.py")
    s = p.read_text()

    # Only change the exact-sizes assertion. Rest of test is fine
    # (different seeds -> different sets; union invariant holds).
    old = (
        '    # Sizes equal (same n, k)\n'
        '    assert [len(s) for s in sets1] == [20, 20, 20, 20, 20]\n'
        '    assert [len(s) for s in sets2] == [20, 20, 20, 20, 20]'
    )

    new = (
        '    # v0.9.3a: shuffle=True default = keyed PRNG; sizes are\n'
        '    # ~binomial, not exact. Sum invariant holds (all rows).\n'
        '    assert sum(len(s) for s in sets1) == 100\n'
        '    assert sum(len(s) for s in sets2) == 100'
    )

    s = replace_line_block(s, old, new, tag="shuffle_diff_assignment")
    p.write_text(s)
    compile_check(p)
    print("  OK")


def main():
    print("=== v0.9.3a-fix6 ===")

    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    fix_test_shuffle_true_same_sizes()
    fix_test_shuffle_different_assignment()

    print("\n=== pytest: full cv suite ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_cv.py", "tests/test_cv_keyed_prng.py",
         "tests/test_cv_e2e.py", "-q"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** CV FAILED ***")
        sys.exit(1)

    print("\n=== pytest: pipeline + scalers regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_pipeline.py",
         "tests/test_pipeline_dag.py",
         "tests/test_scalers.py",
         "-q"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** REGRESSION FAILED ***")
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
        "v0.9.3a: KFold keyed PRNG - O(batch) state default; "
        "materialize=True opt-in (guarded check_state(20*N)); "
        "random_state=None->42; _attach_column_with_fn streaming helper; "
        "test_cv.py 2 stale tests updated (shuffle=True now binomial)"])

    print("\n==> v0.9.3a applied (committed).")
    print("NEXT: v0.9.3a-docs (DECISIONS 13.4/173+174, CHANGELOG)")


if __name__ == "__main__":
    main()
