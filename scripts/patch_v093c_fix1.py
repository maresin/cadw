#!/usr/bin/env python3
# v0.9.3c-fix1: 3 stale AST assertions + docstring.
#
# Code correct (uses _dedup_peak_mult var), tests assert literals that
# don't exist. Same class as v092e fix: code does the right thing, test
# pattern wrong.
#
# Run: uv run python scripts/patch_v093c_fix1.py

from __future__ import annotations
import os, subprocess, sys
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


# ============ Fix 1: new test - string branch ============

def fix_test_string_branch():
    print("\n=== test_dedup_estimator_v093c.py: 12x string ===")
    p = Path("tests/test_dedup_estimator_v093c.py")
    s = p.read_text()

    old = (
        'def test_dedup_uses_12x_for_string_ast():\n'
        '    # AST: _local_dedup contains 12 * table.nbytes for string case.\n'
        '    import inspect\n'
        '    from cadw.ops import dedup as dedup_mod\n'
        '    src = inspect.getsource(dedup_mod._local_dedup)\n'
        '    assert "12 * table.nbytes" in src'
    )

    new = (
        'def test_dedup_uses_12x_for_string_ast():\n'
        '    # AST: _dedup_peak_mult set to 12 for string subset branch.\n'
        '    import inspect\n'
        '    from cadw.ops import dedup as dedup_mod\n'
        '    src = inspect.getsource(dedup_mod._local_dedup)\n'
        '    assert "_dedup_peak_mult = 12 if _has_string_subset" in src\n'
        '    assert "else 2.5" in src'
    )

    if new in s and old not in s:
        print("  already fixed - skip")
        return
    if old not in s:
        raise SystemExit("string_branch block not found")
    s = s.replace(old, new, 1)
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ Fix 2: new test - numeric branch ============

def fix_test_numeric_branch():
    print("\n=== test_dedup_estimator_v093c.py: 2.5x numeric ===")
    p = Path("tests/test_dedup_estimator_v093c.py")
    s = p.read_text()

    old = (
        'def test_dedup_uses_2_5x_for_numeric_ast():\n'
        '    # AST: _local_dedup contains 2.5 * table.nbytes for numeric case.\n'
        '    import inspect\n'
        '    from cadw.ops import dedup as dedup_mod\n'
        '    src = inspect.getsource(dedup_mod._local_dedup)\n'
        '    assert "2.5 * table.nbytes" in src'
    )

    new = (
        'def test_dedup_uses_2_5x_for_numeric_ast():\n'
        '    # AST: _dedup_peak_mult * table.nbytes drives check_state.\n'
        '    import inspect\n'
        '    from cadw.ops import dedup as dedup_mod\n'
        '    src = inspect.getsource(dedup_mod._local_dedup)\n'
        '    assert "_dedup_peak_mult * table.nbytes" in src\n'
        '    assert "_dedup_peak_mult = 12 if" in src\n'
        '    assert "else 2.5" in src'
    )

    if new in s and old not in s:
        print("  already fixed - skip")
        return
    if old not in s:
        raise SystemExit("numeric_branch block not found")
    s = s.replace(old, new, 1)
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ Fix 3: existing AST guard in test_dedup_arrow_native ============

def fix_existing_ast_guard():
    print("\n=== test_dedup_arrow_native.py: stopgap 5x -> mult var ===")
    p = Path("tests/test_dedup_arrow_native.py")
    s = p.read_text()

    old = (
        '    # v0.9.3c (P-D2): 1.5x -> 12x (string) / 2.5x (numeric)\n'
        '    # based on measured peak (11.26x strings at K=N, 1.97x int).\n'
        '    assert "12 * table.nbytes" in src, (\n'
        '        "v0.9.3c: expect `12 * table.nbytes` for string subset"\n'
        '    )\n'
        '    assert "2.5 * table.nbytes" in src, (\n'
        '        "v0.9.3c: expect `2.5 * table.nbytes` for numeric subset"\n'
        '    )\n'
        '    assert "_dedup_peak_mult" in src, (\n'
        '        "v0.9.3c: expect _dedup_peak_mult variable"\n'
        '    )'
    )

    new = (
        '    # v0.9.3c (P-D2): 1.5x -> 12x (string) / 2.5x (numeric)\n'
        '    # based on measured peak (11.26x strings at K=N, 1.97x int).\n'
        '    assert "_dedup_peak_mult = 12 if" in src, (\n'
        '        "v0.9.3c: 12x for string subset (measured 11.26x)"\n'
        '    )\n'
        '    assert "else 2.5" in src, (\n'
        '        "v0.9.3c: 2.5x for numeric subset (measured 1.97x)"\n'
        '    )\n'
        '    assert "_dedup_peak_mult * table.nbytes" in src, (\n'
        '        "v0.9.3c: multiplier applied to check_state"\n'
        '    )'
    )

    if new in s and old not in s:
        print("  already fixed - skip")
        return
    if old not in s:
        print("  WARNING: block not found. Context:")
        idx = s.find("12 * table.nbytes")
        if idx >= 0:
            print("    " + repr(s[max(0, idx-100):idx+300]))
        raise SystemExit("existing AST guard anchor not found")
    s = s.replace(old, new, 1)
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ Fix 4: docstring 1.5x stale ============

def fix_docstring():
    print("\n=== dedup.py: _local_dedup docstring 1.5x stale ===")
    p = Path("cadw/ops/dedup.py")
    lines = p.read_text().split("\n")

    # Find docstring lines with "1.5" near "table.nbytes"
    target_start = None
    target_end = None
    for i, line in enumerate(lines):
        if "Raises MemoryLimitExceeded" in line and "available" in line:
            target_start = i
            break

    if target_start is None:
        print("  WARNING: docstring 'Raises' anchor not found - skip")
        return

    # Following 1-2 lines contain "1.5× table.nbytes"
    # Find end: first line that contains "state_limit"
    for j in range(target_start, min(target_start + 4, len(lines))):
        if "state_limit" in lines[j]:
            target_end = j
            break

    if target_end is None:
        print("  WARNING: 'state_limit' anchor not found - skip")
        return

    # Print current
    print("  current docstring lines " + str(target_start + 1) +
          "-" + str(target_end + 1) + ":")
    for j in range(target_start, target_end + 1):
        print("    " + repr(lines[j]))

    # Replace with updated text (no unicode cross - use 'x' or '*')
    new_block = [
        "    Raises MemoryLimitExceeded if partition > available*0.5, or if",
        "    Arrow-native bound exceeds state_limit:",
        "      - 12x table.nbytes for string subset (11.26x measured at K=N,",
        "        +7% safety)",
        "      - 2.5x table.nbytes for numeric subset (1.97x measured at K=N,",
        "        +27% safety)",
        "    See v0.9.3c / P-D2 / 13.4/175.",
    ]
    lines[target_start:target_end + 1] = new_block
    p.write_text("\n".join(lines))
    compile_check(p)
    print("  OK")


def main():
    print("=== v0.9.3c-fix1 ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    fix_test_string_branch()
    fix_test_numeric_branch()
    fix_existing_ast_guard()
    fix_docstring()

    print("\n=== pytest: new estimator tests ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_dedup_estimator_v093c.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** NEW TESTS STILL RED ***")
        sys.exit(1)

    print("\n=== pytest: existing dedup tests ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_dedup_arrow_native.py",
         "tests/test_drop_duplicates.py",
         "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** DEDUP REGRESSION FAILED ***")
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
        "v0.9.3c: P-D2 dedup type-based multiplier S2 fix — string 12x "
        "table.nbytes (11.26x measured at K=N); numeric 2.5x (1.97x); "
        "_dedup_peak_mult drives check_state; AST guards match var-based "
        "pattern; docstring updated; 13.4/175 (memory formula scope "
        "bench-validated)"])

    print("\n==> v0.9.3c applied.")
    print("NEXT: v0.9.3d (C2 StdOp int truncation S2)")


if __name__ == "__main__":
    main()
