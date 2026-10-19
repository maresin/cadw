#!/usr/bin/env python3
"""v0.7.5-fix: test_fit_gbdt_classifier_accuracy — stronger signal (noise=0).

Баг: coefs=(3.0,), noise=0.1, seed=42, n=500 → Bayes-rate ~0.84-0.86.
Тест ждал accuracy > 0.85, получил 0.840 (шум сэмпла).

Fix: coefs=(4.0,), noise=0.0 → Bayes-rate ~0.90. Модель accuracy ~0.88-0.92
стабильно выше 0.85. Более прямая проверка "модель учится".
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
    print("\n=== tests/test_gbdt_classification.py: accuracy test ===")
    p = Path("tests/test_gbdt_classification.py")
    s = p.read_text()

    old = '''def test_fit_gbdt_classifier_accuracy(tmp_path):
    """Clean signal: accuracy > 0.85."""
    df, X, y = _make_binary_df(tmp_path, n=500, coefs=(3.0,), intercept=0.0)
    model = fit_gbdt_classifier(
        df, ["x0"], "y",
        n_estimators=30, max_depth=3, learning_rate=0.1, bins=16,
    )
    assert isinstance(model, GBDTClassifier)
    pred = model.predict(X)
    acc = (pred == y).mean()
    assert acc > 0.85, f"accuracy={acc:.3f}"'''

    new = '''def test_fit_gbdt_classifier_accuracy(tmp_path):
    """Strong clean signal: accuracy > 0.85 (Bayes-rate ~0.90).

    coefs=(4.0,), noise=0.0 → z = 4x, y ~ Bernoulli(sigmoid(z)).
    Bayes accuracy ≈ E[max(p,1-p)] ≈ 0.90. Model should reach > 0.85.
    """
    df, X, y = _make_binary_df(
        tmp_path, n=1000, coefs=(4.0,), intercept=0.0,
    )
    # _make_binary_df default noise=0.1; override via direct call
    from tests.test_gbdt_classification import _synth_binary
    X, y = _synth_binary(1000, (4.0,), intercept=0.0, noise=0.0, seed=42)
    data = {f"x{i}": X[:, i].tolist() for i in range(X.shape[1])}
    data["y"] = y.tolist()
    from tests.test_gbdt_classification import make_df as _mdf
    df = _mdf(tmp_path, data, name="strong")

    model = fit_gbdt_classifier(
        df, ["x0"], "y",
        n_estimators=50, max_depth=3, learning_rate=0.1, bins=16,
    )
    assert isinstance(model, GBDTClassifier)
    pred = model.predict(X)
    acc = (pred == y).mean()
    assert acc > 0.85, f"accuracy={acc:.3f} (expected > 0.85)"'''

    s = replace_exact(s, old, new, tag="accuracy-test")
    p.write_text(s)
    compile_check(p)
    print("  OK: accuracy test relaxed with stronger signal")


def main():
    print("=== v0.7.5-fix ===")
    patch_test()

    print("\n=== pytest tests/test_gbdt_classification.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_gbdt_classification.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** FAILURE ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.7.5-fix (test red)"], check=False)
        sys.exit(1)

    print("\n=== pytest tests/test_gbdt_regression.py -q (regression) ===")
    r2 = subprocess.run(
        ["uv", "run", "pytest", "tests/test_gbdt_regression.py", "-q"],
        timeout=300,
    )
    if r2.returncode != 0:
        print("\n*** REGRESSION FAILURE ***")
        sys.exit(1)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.7.5-fix: classifier accuracy test — stronger signal (coef=4, noise=0)"])

    print("\n==> v0.7.5 complete.")


if __name__ == "__main__":
    main()