#!/usr/bin/env python3
"""v0.6.2c-fix: _RollState.push — _sum только для sum/mean.

Баг: `count` на string-колонке вызывает float('a') → ValueError.
Fix: _sum обновляется только для agg in ('sum','mean'); _count — всегда;
min/max — только deque (сравнения generic).
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


def patch_roll_state():
    print("\n=== cadw/dataframe.py: _RollState.push ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()

    # 1. Add-current block
    old = '''        # 1. Add current
        is_null = v is None
        if not is_null:
            self._sum += float(v)
            self._count += 1
        self._buf.append((i, is_null, v))'''
    new = '''        # 1. Add current
        is_null = v is None
        if not is_null:
            self._count += 1
            if self._agg in ("sum", "mean"):
                self._sum += float(v)
        self._buf.append((i, is_null, v))'''
    s = replace_exact(s, old, new, tag="add-current")

    # 2. Eviction block
    old = '''        while len(self._buf) > window:
            old_i, old_null, old_v = self._buf.popleft()
            if not old_null:
                self._sum -= float(old_v)
                self._count -= 1
            if self._agg in ("min", "max") and self._dq and self._dq[0][0] == old_i:
                self._dq.popleft()'''
    new = '''        while len(self._buf) > window:
            old_i, old_null, old_v = self._buf.popleft()
            if not old_null:
                self._count -= 1
                if self._agg in ("sum", "mean"):
                    self._sum -= float(old_v)
            if self._agg in ("min", "max") and self._dq and self._dq[0][0] == old_i:
                self._dq.popleft()'''
    s = replace_exact(s, old, new, tag="eviction")

    p.write_text(s)
    compile_check(p)
    print("  OK: _RollState.push — _sum только для sum/mean")


def main():
    print("=== v0.6.2c-fix ===")
    patch_roll_state()

    print("\n=== pytest tests/test_rolling.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_rolling.py", "-v"]
    )
    if r.returncode != 0:
        print("\n*** FAILURE ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.6.2c-fix (test red)"], check=False)
        sys.exit(1)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.6.2c-fix: _RollState.push — _sum только для sum/mean (count на string)"])

    print("\n==> v0.6.2c complete.")


if __name__ == "__main__":
    main()