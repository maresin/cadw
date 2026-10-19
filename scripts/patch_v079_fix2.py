#!/usr/bin/env python3
"""v0.7.9-fix2: spec[:4][:4] → spec[:4] (idempotent).

Причина: patch_v079_fix делал s.replace("j, thr, left, right = spec",
"j, thr, left, right = spec[:4]") — substring replace, сработал на уже-
патченной строке → "spec[:4][:4]". AST guard поймал, но код некрасивый.

Функционально spec[:4][:4] == spec[:4] (double slice no-op), но:
1. Читателю непонятно.
2. AST guard верно запрещает (strict).

Fix — idempotent replace (all occurrences, no count).
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


def main():
    print("=== v0.7.9-fix2 ===")

    p = Path("cadw/ml/gbdt.py")
    s = p.read_text()

    # Idempotent: collapse any number of trailing [:4]
    for _ in range(5):  # max 5 levels of repeated [:4]
        if "spec[:4][:4]" not in s:
            break
        s = s.replace("spec[:4][:4]", "spec[:4]")  # all

    # Verify: no remaining [:4][:4]
    assert "spec[:4][:4]" not in s, "leftover double slice"
    assert "spec[:5][:5]" not in s, "leftover double slice (5)"

    p.write_text(s)
    compile_check(p)
    print("  OK: spec[:4][:4] → spec[:4]")

    print("\n=== pytest tests/test_feature_importance_gain.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_feature_importance_gain.py", "-v"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** FAILURE — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.7.9-fix2 (test red)"], check=False)
        sys.exit(1)

    print("\n=== regression: GBDT + viz ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_gbdt_regression.py",
         "tests/test_gbdt_classification.py",
         "tests/test_gbdt_multiclass.py",
         "tests/test_viz_matplotlib.py",
         "tests/test_viz_plotly.py",
         "-q"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** REGRESSION FAILURE ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.7.9-fix2 (regression)"], check=False)
        sys.exit(1)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.7.9-fix2: spec[:4][:4] → spec[:4] (idempotent; AST guard catches)"])

    print("\n==> v0.7.9-fix2 complete. Готов squash → tag v0.7.9.")


if __name__ == "__main__":
    main()