#!/usr/bin/env python3
"""v0.7.8-fix: cadw/__init__.py + cadw/ml/__init__.py — explicit exports.

Причина: patch_v078 `patch_exports` использовал str.replace с `\n)` в
pattern, но `GBDTClassifier, fit_gbdt_classifier,` не последняя в
многострочном import — replace silent no-op → ImportError в тестах.

Fix: однострочные замены + assert (fail-fast, не silent).
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


def replace_or_fail(s, old, new, *, tag, path):
    """Замена с assert — никаких silent no-op."""
    if old not in s:
        raise SystemExit(
            f"[{path}:{tag}] pattern NOT FOUND:\n---\n{old}\n---"
        )
    if s.count(old) > 1:
        raise SystemExit(
            f"[{path}:{tag}] pattern found {s.count(old)} times"
        )
    return s.replace(old, new, 1)


def patch_ml_init():
    print("\n=== cadw/ml/__init__.py ===")
    p = Path("cadw/ml/__init__.py")
    s = p.read_text()

    # State check
    has_import = "GBDTMulticlassClassifier" in s
    print(f"  state: has_import={has_import}")

    if not has_import:
        # In `from .gbdt import (...)` — append line before closing paren.
        # Safer pattern: anchor on `    GBDTClassifier, fit_gbdt_classifier,`
        # (single-line, known-content).
        old_import = "    GBDTClassifier, fit_gbdt_classifier,\n"
        if old_import in s:
            s = s.replace(
                old_import,
                old_import
                + "    GBDTMulticlassClassifier, fit_gbdt_multiclass,\n",
                1,
            )
            print("  OK: import line appended")
        else:
            # Fallback: find `from .gbdt import (` line and check
            raise SystemExit(
                "  ERROR: `    GBDTClassifier, fit_gbdt_classifier,\\n` "
                "not found in cadw/ml/__init__.py. Inspect manually."
            )

        # __all__: find `    "GBDTClassifier", "fit_gbdt_classifier",\n`
        old_all = '    "GBDTClassifier", "fit_gbdt_classifier",\n'
        if old_all in s:
            s = s.replace(
                old_all,
                old_all
                + '    "GBDTMulticlassClassifier", "fit_gbdt_multiclass",\n',
                1,
            )
            print("  OK: __all__ line appended")
        else:
            raise SystemExit(
                "  ERROR: __all__ line not found in cadw/ml/__init__.py"
            )

        p.write_text(s)
    else:
        print("  already present — skip")

    # Verify
    s2 = p.read_text()
    assert "GBDTMulticlassClassifier" in s2
    assert "fit_gbdt_multiclass" in s2
    compile_check(p)


def patch_cadw_init():
    print("\n=== cadw/__init__.py ===")
    p = Path("cadw/__init__.py")
    s = p.read_text()

    has_import = "GBDTMulticlassClassifier" in s
    print(f"  state: has_import={has_import}")

    if not has_import:
        # Import line
        old_import = "    GBDTClassifier, fit_gbdt_classifier,\n"
        if old_import in s:
            s = s.replace(
                old_import,
                old_import
                + "    GBDTMulticlassClassifier, fit_gbdt_multiclass,\n",
                1,
            )
            print("  OK: import line appended")
        else:
            raise SystemExit(
                "  ERROR: import line not found in cadw/__init__.py"
            )

        # __all__
        old_all = '    "GBDTClassifier", "fit_gbdt_classifier",\n'
        if old_all in s:
            s = s.replace(
                old_all,
                old_all
                + '    "GBDTMulticlassClassifier", "fit_gbdt_multiclass",\n',
                1,
            )
            print("  OK: __all__ line appended")
        else:
            raise SystemExit(
                "  ERROR: __all__ line not found in cadw/__init__.py"
            )

        p.write_text(s)
    else:
        print("  already present — skip")

    s2 = p.read_text()
    assert "GBDTMulticlassClassifier" in s2
    assert "fit_gbdt_multiclass" in s2
    compile_check(p)


def verify_imports():
    print("\n=== Verify imports work ===")
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         "from cadw import fit_gbdt_multiclass, GBDTMulticlassClassifier; "
         "from cadw.ml import fit_gbdt_multiclass as f2; "
         "print('OK')"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"  IMPORT FAIL:\n{r.stderr[-800:]}")
        sys.exit(1)
    print(f"  {r.stdout.strip()}")


def main():
    print("=== v0.7.8-fix ===")
    patch_ml_init()
    patch_cadw_init()
    verify_imports()

    print("\n=== pytest tests/test_gbdt_multiclass.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_gbdt_multiclass.py", "-v"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** FAILURE — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.7.8-fix (test red)"], check=False)
        sys.exit(1)

    print("\n=== regression: binary GBDT ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_gbdt_regression.py",
         "tests/test_gbdt_classification.py",
         "-q"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** REGRESSION FAILURE ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.7.8-fix (regression)"], check=False)
        sys.exit(1)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.7.8-fix: cadw/__init__.py + cadw/ml/__init__.py — "
        "GBDTMulticlassClassifier + fit_gbdt_multiclass exports"])

    print("\n==> v0.7.8-fix applied")


if __name__ == "__main__":
    main()