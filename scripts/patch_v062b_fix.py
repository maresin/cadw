#!/usr/bin/env python3
"""v0.6.2b-fix: test_diff_with_nulls — исправить expected индексы.

df.v = [1.0, None, 3.0, 5.0]
diff(1): out[0]=null (fill); out[1]=None-1.0=null; out[2]=3.0-None=null;
out[3]=5.0-3.0=2.0. Тест проверял out[2]=2.0 — ошибка. Fix: out[3]=2.0.
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
    print("\n=== tests/test_shift_diff.py: test_diff_with_nulls ===")
    p = Path("tests/test_shift_diff.py")
    s = p.read_text()

    old = '''def test_diff_with_nulls(tmp_path):
    df = make_df(tmp_path, data={"v": [1.0, None, 3.0, 5.0]})
    res = df.diff(1, columns=["v"])
    # 1, None, 3-1, 5-3 -> None, None, 2.0, 2.0 (pc.subtract null-propagates)
    vals = read_col(res, tmp_path, "v")
    assert vals[0] is None
    assert vals[1] is None
    assert abs(vals[2] - 2.0) < 1e-9
    assert abs(vals[3] - 2.0) < 1e-9'''

    new = '''def test_diff_with_nulls(tmp_path):
    df = make_df(tmp_path, data={"v": [1.0, None, 3.0, 5.0]})
    res = df.diff(1, columns=["v"])
    # diff(1): out[0]=null (fill); out[i]=x[i]-x[i-1]
    # x = [1.0, None, 3.0, 5.0]
    # out[0] = null (fill)
    # out[1] = None - 1.0 = null (null-propagate)
    # out[2] = 3.0 - None = null (null-propagate)
    # out[3] = 5.0 - 3.0 = 2.0
    vals = read_col(res, tmp_path, "v")
    assert vals[0] is None
    assert vals[1] is None
    assert vals[2] is None
    assert abs(vals[3] - 2.0) < 1e-9'''

    s = replace_exact(s, old, new, tag="diff-with-nulls")
    p.write_text(s)
    compile_check(p)
    print("  OK: expected fixed (out[3]=2.0)")


def main():
    print("=== v0.6.2b-fix ===")
    patch_test()

    print("\n=== pytest tests/test_shift_diff.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_shift_diff.py", "-v"]
    )
    if r.returncode != 0:
        print("\n*** FAILURE ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.6.2b-fix (test red)"], check=False)
        sys.exit(1)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.6.2b-fix: test_diff_with_nulls — expected out[3]=2.0 (not out[2])"])

    print("\n==> v0.6.2b complete.")


if __name__ == "__main__":
    main()