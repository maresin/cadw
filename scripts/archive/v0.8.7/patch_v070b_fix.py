#!/usr/bin/env python3
"""v0.7.0b-fix: test_fit_logistic_noisy — threshold + coef recovery test.

Проблема: y = Bernoulli(sigmoid(z)) — этикетки вероятностные. Даже
идеальный β даёт accuracy = E[max(p, 1-p)] ≈ 0.80 на этом датасете
(seed=42). Test threshold 0.85 — Bayes-limited, невыполним.

Fix:
1. Threshold 0.85 → 0.75 (модель beats chance с запасом).
2. Add test_fit_logistic_coef_recovery: n=5000, noise=0, |coef - true| < 0.2.
   Это прямая проверка correctness — сильнее accuracy.
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
    print("\n=== tests/test_logistic.py: fix noisy test + add recovery ===")
    p = Path("tests/test_logistic.py")
    s = p.read_text()

    # 1. Replace noisy test (relax threshold + explanatory comment)
    old = '''def test_fit_logistic_noisy(tmp_path):
    n = 1000
    data, X, y = _synth_logistic(n, [2.0, -1.5], intercept=0.5, noise=0.1)
    df = make_df(tmp_path, data)
    model = fit_logistic(df, ["x0", "x1"], "y", alpha=1.0)
    assert isinstance(model, LogisticModel)
    # Right sign
    assert model.coef_[0] > 0.5   # true +2
    assert model.coef_[1] < -0.3  # true -1.5
    # Reasonable accuracy
    proba = model.predict_proba(X)
    pred = (proba >= 0.5).astype(int)
    acc = (pred == y).mean()
    assert acc > 0.85'''

    new = '''def test_fit_logistic_noisy(tmp_path):
    """Right-sign + accuracy above chance floor.

    Note: y = Bernoulli(sigmoid(z)) — inherent label noise. Even an
    ideal model has accuracy = E[max(p,1-p)] ≈ 0.80 for this dataset.
    Stronger correctness check: test_fit_logistic_coef_recovery below.
    """
    n = 1000
    data, X, y = _synth_logistic(n, [2.0, -1.5], intercept=0.5, noise=0.1)
    df = make_df(tmp_path, data)
    model = fit_logistic(df, ["x0", "x1"], "y", alpha=1.0)
    assert isinstance(model, LogisticModel)
    # Right sign of coefficients
    assert model.coef_[0] > 0.5   # true +2
    assert model.coef_[1] < -0.3  # true -1.5
    # Accuracy beats chance (Bayes-optimal for this data ~0.80)
    proba = model.predict_proba(X)
    pred = (proba >= 0.5).astype(int)
    acc = (pred == y).mean()
    assert acc > 0.75, f"acc={acc} (Bayes floor ~0.80)"


def test_fit_logistic_coef_recovery(tmp_path):
    """Direct correctness: recovered coefs match true (noise=0, large n).

    n=5000 gives SE ~ 0.05 per coef; tolerance 0.2 is safe.
    """
    n = 5000
    data, X, y = _synth_logistic(
        n, [2.0, -1.5], intercept=0.5, noise=0.0, seed=42,
    )
    df = make_df(tmp_path, data, chunk=100)
    # Very small alpha: ignore L2 penalty for recovery test
    model = fit_logistic(df, ["x0", "x1"], "y", alpha=0.001)
    assert abs(model.coef_[0] - 2.0) < 0.2, (
        f"coef[0]={model.coef_[0]:.3f}, expected 2.0"
    )
    assert abs(model.coef_[1] - (-1.5)) < 0.2, (
        f"coef[1]={model.coef_[1]:.3f}, expected -1.5"
    )
    assert abs(model.intercept_ - 0.5) < 0.2, (
        f"intercept={model.intercept_:.3f}, expected 0.5"
    )'''

    s = replace_exact(s, old, new, tag="noisy-test")
    p.write_text(s)
    compile_check(p)
    print("  OK: noisy relaxed + coef_recovery added")


def main():
    print("=== v0.7.0b-fix ===")
    patch_test()

    print("\n=== pytest tests/test_logistic.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_logistic.py", "-v"]
    )
    if r.returncode != 0:
        print("\n*** FAILURE ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.7.0b-fix (test red)"], check=False)
        sys.exit(1)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.7.0b-fix: noisy threshold 0.75 (Bayes-limited); +coef recovery test"])

    print("\n==> v0.7.0b complete.")


if __name__ == "__main__":
    main()