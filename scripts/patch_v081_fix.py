#!/usr/bin/env python3
"""v0.8.1-fix: RecordBatchFileReader API in partition-invariant tests.

PyArrow 26 `RecordBatchFileReader` не имеет `iter_batches()` (только
`get_batch(i)` и `num_record_batches`). Заменяем вызов в 2 тестах.
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
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


def replace_exact(s, old, new, *, tag):
    if old not in s:
        raise SystemExit(f"[{tag}] pattern not found")
    if s.count(old) > 1:
        raise SystemExit(f"[{tag}] pattern found {s.count(old)} times")
    return s.replace(old, new, 1)


def patch_tests():
    print("\n=== tests/test_hash_v081.py: get_batch loop вместо iter_batches ===")
    p = Path("tests/test_hash_v081.py")
    s = p.read_text()

    # Заменяем обе occurrences (int и string тесты)
    old = '''    key_to_parts: dict[int, set[int]] = {}
    for p_idx, path in enumerate(paths):
        if not path.exists():
            continue
        for b in pa.ipc.open_file(str(path)).iter_batches():
            for r in b.to_pylist():
                key_to_parts.setdefault(r["k"], set()).add(p_idx)'''

    new = '''    key_to_parts: dict[int, set[int]] = {}
    for p_idx, path in enumerate(paths):
        if not path.exists():
            continue
        reader = pa.ipc.open_file(str(path))
        for i in range(reader.num_record_batches):
            b = reader.get_batch(i)
            for r in b.to_pylist():
                key_to_parts.setdefault(r["k"], set()).add(p_idx)'''

    s = replace_exact(s, old, new, tag="int-reader")

    old_str = '''    key_to_parts: dict[str, set[int]] = {}
    for p_idx, path in enumerate(paths):
        if not path.exists():
            continue
        for b in pa.ipc.open_file(str(path)).iter_batches():
            for r in b.to_pylist():
                key_to_parts.setdefault(r["k"], set()).add(p_idx)'''

    new_str = '''    key_to_parts: dict[str, set[int]] = {}
    for p_idx, path in enumerate(paths):
        if not path.exists():
            continue
        reader = pa.ipc.open_file(str(path))
        for i in range(reader.num_record_batches):
            b = reader.get_batch(i)
            for r in b.to_pylist():
                key_to_parts.setdefault(r["k"], set()).add(p_idx)'''

    s = replace_exact(s, old_str, new_str, tag="str-reader")

    p.write_text(s)
    compile_check(p)
    print("  OK: 2 tests fixed")


def main():
    print("=== v0.8.1-fix ===")
    patch_tests()

    print("\n=== pytest tests/test_hash_v081.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_hash_v081.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** FAILURE ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.8.1-fix (test red)"], check=False)
        sys.exit(1)

    print("\n=== pytest tests/test_shuffle.py -v (partition-invariant regression) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_shuffle.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** REGRESSION FAILURE ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.8.1-fix (shuffle regression)"], check=False)
        sys.exit(1)

    print("\n=== bench: v0.8.1 hash (baseline: 39.2ms / 0.61s shuffle) ===")
    sh(["uv", "run", "python", "scripts/bench_hash_v081.py"], check=False)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.8.1-fix: get_batch loop в partition-invariant тестах "
        "(RecordBatchFileReader.iter_batches не существует)"])

    print("\n==> v0.8.1 complete.")


if __name__ == "__main__":
    main()