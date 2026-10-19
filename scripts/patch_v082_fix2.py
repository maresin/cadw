#!/usr/bin/env python3
"""v0.8.2-fix2: import ArrowIPCBatchWriter в cadw/dataframe.py.

Причина: _shift_on использует ArrowIPCBatchWriter, но dataframe.py
импортирует только ParquetBatchWriter из cadw.io.batch_writer.
NameError: name 'ArrowIPCBatchWriter' is not defined.

Fix: add ArrowIPCBatchWriter to import line.
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
        for line in r.stdout.strip().splitlines()[:25]:
            print(f"    {line}")
    if check and r.returncode != 0:
        print(f"    STDERR: {r.stderr.strip()[-1500:]}")
        sys.exit(r.returncode)
    return r.returncode


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


def fix_import():
    print("\n=== cadw/dataframe.py: +ArrowIPCBatchWriter import ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()

    old = "from cadw.io.batch_writer import ParquetBatchWriter"
    new = "from cadw.io.batch_writer import ArrowIPCBatchWriter, ParquetBatchWriter"

    if "ArrowIPCBatchWriter" in s.split("\n")[0:80].__str__():
        # Already imported
        if new in s or "from cadw.io.batch_writer import ArrowIPCBatchWriter" in s:
            print("  already imported — skip")
            return

    if old not in s:
        # Try alternate single-import form
        old2 = "from cadw.io.batch_writer import ArrowIPCBatchWriter"
        if old2 in s:
            print("  already imported (single form) — skip")
            return
        raise SystemExit(
            f"pattern {old!r} not found in cadw/dataframe.py. "
            f"Imports at top: {[l for l in s.splitlines()[:80] if 'batch_writer' in l]}"
        )

    if s.count(old) > 1:
        raise SystemExit(f"pattern found {s.count(old)} times")
    s = s.replace(old, new, 1)
    p.write_text(s)
    compile_check(p)
    print("  OK: ArrowIPCBatchWriter + ParquetBatchWriter imported")

    # Verify runtime
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         "from cadw.dataframe import ArrowIPCBatchWriter; print('OK')"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"  IMPORT FAIL:\n{r.stderr[-800:]}")
        sys.exit(1)
    print(f"  runtime verify: {r.stdout.strip()}")


def main():
    print("=== v0.8.2-fix2 ===")
    fix_import()

    print("\n=== pytest tests/test_shift_on_col.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_shift_on_col.py", "-v"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** FAILURE — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.8.2-fix2 (test red)"], check=False)
        sys.exit(1)

    print("\n=== regression: shift_diff + rolling ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_shift_diff.py",
         "tests/test_rolling.py",
         "-q"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** REGRESSION FAILURE ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.8.2-fix2 (regression)"], check=False)
        sys.exit(1)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.8.2-fix2: import ArrowIPCBatchWriter в dataframe.py"])

    print("\n==> v0.8.2-fix2 complete. Готов squash → tag v0.8.2.")


if __name__ == "__main__":
    main()