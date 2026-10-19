#!/usr/bin/env python3
"""v0.7.3-fix2: predict — mask NaN rows before binning (warning cleanup).

Баг: `predict(df)` для NaN-строк делает `NaN.astype(int64)` — RuntimeWarning
'invalid value encountered in cast'. Значение undefined, но результат
корректный (nan_mask в pred_nulls). Fix: маска NaN rows — они не спускаются
по дереву, остаются в leaf=0, nulled в _iter_pred.
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


def patch_gbdt_predict():
    print("\n=== cadw/ml/gbdt.py: predict NaN mask ===")
    p = Path("cadw/ml/gbdt.py")
    s = p.read_text()

    old = '''        def _iter_pred():
            for b in X._iter_filtered_batches():
                Xm = _stack_numeric(b, self.feature_names_)
                n = Xm.shape[0]
                pred = np.full(n, base, dtype=np.float64)
                for tree in trees:
                    leaf = np.zeros(n, dtype=np.int64)
                    for nid, spec in tree["structure"].items():
                        if spec is None:
                            continue
                        j, thr, left, right = spec
                        m = leaf == nid
                        if not m.any():
                            continue
                        b_idx = np.clip(
                            ((Xm[m, j] - mn[j]) / rng[j] * bins)
                            .astype(np.int64),
                            0, bins - 1,
                        )
                        leaf[m] = np.where(b_idx <= thr, left, right)'''

    new = '''        def _iter_pred():
            for b in X._iter_filtered_batches():
                Xm = _stack_numeric(b, self.feature_names_)
                n = Xm.shape[0]
                # v0.7.3-fix2: rows with any NaN feature stay at leaf=0,
                # nulled in output. Avoids NaN.astype(int64) warnings.
                nan_rows = np.isnan(Xm).any(axis=1)
                pred = np.full(n, base, dtype=np.float64)
                for tree in trees:
                    leaf = np.zeros(n, dtype=np.int64)
                    for nid, spec in tree["structure"].items():
                        if spec is None:
                            continue
                        j, thr, left, right = spec
                        m = (leaf == nid) & ~nan_rows
                        if not m.any():
                            continue
                        b_idx = np.clip(
                            ((Xm[m, j] - mn[j]) / rng[j] * bins)
                            .astype(np.int64),
                            0, bins - 1,
                        )
                        leaf[m] = np.where(b_idx <= thr, left, right)'''

    s = replace_exact(s, old, new, tag="predict-nan-mask")
    p.write_text(s)
    compile_check(p)
    print("  OK: predict — NaN mask")


def main():
    print("=== v0.7.3-fix2 ===")
    patch_gbdt_predict()

    print("\n=== pytest tests/test_gbdt_regression.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_gbdt_regression.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** FAILURE ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.7.3-fix2 (test red)"], check=False)
        sys.exit(1)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.7.3-fix2: predict NaN-mask (warnings-free)"])

    print("\n==> v0.7.3 complete (warnings-free).")


if __name__ == "__main__":
    main()