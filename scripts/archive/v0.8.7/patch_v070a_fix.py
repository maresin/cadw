#!/usr/bin/env python3
"""v0.7.0a-fix: test_fit_linear_all_null_raises — use explicit float64 type.

`pa.table({"x0": [None, None, None]})` infers `null` type, not float64.
`null` type correctly rejected by fit_linear (can't infer numeric dtype).
Fix test: explicit pa.array([None]*3, type=pa.float64()) → all-null
float64 → n_valid=0 → ValueError "no valid rows".
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)


def sh(cmd, check=True):
    print(f"  $ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    r = subprocess.run(
        cmd, shell=isinstance(cmd, str), capture_output=True, text=True,
    )
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines()[:10]:
            print(f"    {line}")
    if check and r.returncode != 0:
        print(f"    STDERR: {r.stderr.strip()[-800:]}")
        sys.exit(r.returncode)
    return r.returncode


def replace_exact(s, old, new, *, tag):
    if old not in s:
        raise SystemExit(f"[{tag}] pattern not found")
    if s.count(old) > 1:
        raise SystemExit(f"[{tag}] pattern found {s.count(old)} times")
    return s.replace(old, new, 1)


def compile_check(path):
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         f"import py_compile; py_compile.compile('{path}', doraise=True)"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"COMPILE FAIL: {path}")
        print(r.stderr[-800:])
        sys.exit(1)
    print(f"  compile OK: {path}")


def patch_test():
    print("\n=== tests/test_linear.py: test_fit_linear_all_null_raises ===")
    p = Path("tests/test_linear.py")
    s = p.read_text()

    old = '''def test_fit_linear_all_null_raises(tmp_path):
    df = make_df(tmp_path, {
        "x0": [None, None, None],
        "y": [None, None, None],
    })
    # Fit — n_valid = 0
    with pytest.raises(ValueError, match="no valid"):
        fit_linear(df, ["x0"], "y")'''

    new = '''def test_fit_linear_all_null_raises(tmp_path):
    """All-null float64 columns: n_valid=0 → ValueError.

    Explicit type=pa.float64() — `pa.table({"x0": [None, None, None]})`
    infers `null` type (correctly rejected by fit_linear as non-numeric).
    """
    df = make_df(tmp_path, {
        "x0": pa.array([None, None, None], type=pa.float64()),
        "y": pa.array([None, None, None], type=pa.float64()),
    })
    with pytest.raises(ValueError, match="no valid"):
        fit_linear(df, ["x0"], "y")


def test_fit_linear_null_typed_column_raises(tmp_path):
    """Column of `null` type (not float64) — TypeError (non-numeric)."""
    df = make_df(tmp_path, {
        "x0": [None, None, None],   # `null` type
        "y": [1.0, 2.0, 3.0],       # float64
    })
    with pytest.raises(TypeError, match="numeric"):
        fit_linear(df, ["x0"], "y")'''

    s = replace_exact(s, old, new, tag="all-null-test")
    p.write_text(s)
    compile_check(p)
    print("  OK: test fixed + added null-type test")


def main():
    print("=== v0.7.0a-fix ===")
    patch_test()

    print("\n=== pytest tests/test_linear.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_linear.py", "-v"]
    )
    if r.returncode != 0:
        print("\n*** FAILURE ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.7.0a-fix (test red)"], check=False)
        sys.exit(1)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.7.0a-fix: all-null test uses explicit float64; null-type column rejected"])

    print("\n==> v0.7.0a complete.")


if __name__ == "__main__":
    main()