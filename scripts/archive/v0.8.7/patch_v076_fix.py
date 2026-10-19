#!/usr/bin/env python3
"""v0.7.6-fix: Bernoulli NB test signal + NB ReduceOps state_bytes.

Bugs fixed:
    1. test_bernoulli_nb_accuracy: `_binary_data(p0=0.2, p1=0.8)` has
       Bayes-rate ≈ 0.896 (below 0.9 threshold). Model at 0.878 = noise.
       Fix: p0=0.1, p1=0.9 → Bayes-rate ≈ 0.972 (same idea as
       v0.7.5-fix for GBDT accuracy — stronger signal, not lower bar).
    2. NB ReduceOps (_GaussianStatsOp, _MultinomialStatsOp,
       _BernoulliStatsOp) — no `estimate_state_nbytes()`, so reduce()
       falls back to 1 MiB hardcode + emits warning per fit.
       Fix: conservative override `C_ASSUMED=32` per-class state bound.
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


# ============ 1. Fix _binary_data (stronger signal) ============

def patch_binary_data():
    print("\n=== tests/test_naive_bayes.py: _binary_data (p0=0.1, p1=0.9) ===")
    p = Path("tests/test_naive_bayes.py")
    s = p.read_text()

    old = '''def _binary_data(n_per_class=150, seed=42):
    rng = np.random.default_rng(seed)
    # Class 0: features mostly 0; Class 1: features mostly 1
    X0 = (rng.random((n_per_class, 3)) < 0.2).astype(np.float64)
    X1 = (rng.random((n_per_class, 3)) < 0.8).astype(np.float64)
    X = np.vstack([X0, X1])
    y = np.array([0.0] * n_per_class + [1.0] * n_per_class)
    return {
        "x0": X[:, 0].tolist(), "x1": X[:, 1].tolist(),
        "x2": X[:, 2].tolist(), "y": y.tolist(),
    }, X, y'''

    new = '''def _binary_data(n_per_class=150, seed=42, p0=0.1, p1=0.9):
    """Binary features: P(x_j=1|c=0)=p0, P(x_j=1|c=1)=p1.

    Bayes-rate = E[max(P(c|x), 1-P(c|x))] depends on (p0, p1, d=3):
        p0=0.2, p1=0.8  → Bayes-rate ≈ 0.896 (n=300 acc=0.878 — below 0.9)
        p0=0.1, p1=0.9  → Bayes-rate ≈ 0.972 (safe for acc > 0.9)
    v0.7.6-fix: use p0=0.1, p1=0.9 (same principle as v0.7.5 accuracy
    fix — strengthen signal, not lower threshold).
    """
    rng = np.random.default_rng(seed)
    X0 = (rng.random((n_per_class, 3)) < p0).astype(np.float64)
    X1 = (rng.random((n_per_class, 3)) < p1).astype(np.float64)
    X = np.vstack([X0, X1])
    y = np.array([0.0] * n_per_class + [1.0] * n_per_class)
    return {
        "x0": X[:, 0].tolist(), "x1": X[:, 1].tolist(),
        "x2": X[:, 2].tolist(), "y": y.tolist(),
    }, X, y'''

    s = replace_exact(s, old, new, tag="binary-data")
    p.write_text(s)
    compile_check(p)
    print("  OK: _binary_data(p0=0.1, p1=0.9)")


# ============ 2. NB ops: estimate_state_nbytes ============

def patch_nb_state_bytes():
    print("\n=== cadw/ml/naive_bayes.py: estimate_state_nbytes on 3 ops ===")
    p = Path("cadw/ml/naive_bayes.py")
    s = p.read_text()

    # ---- Gaussian ----
    anchor_g = '''    def __init__(self, feature_cols: Sequence[str], target_col: str):
        self.feature_cols = list(feature_cols)
        self.target_col = target_col
        self.d = len(feature_cols)

    def init(self):
        return {}

    def update(self, state, batch):
        X = _stack_numeric(batch, self.feature_cols)
        y = _column_to_f64(batch.column(self.target_col))
        valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[valid]
        y = y[valid]
        if X.shape[0] == 0:
            return state
        for cls in np.unique(y):
            m = y == cls
            Xc = X[m]
            key = float(cls)
            if key not in state:
                state[key] = [0, np.zeros(self.d), np.zeros(self.d)]
            state[key][0] += int(Xc.shape[0])
            state[key][1] += Xc.sum(axis=0)
            state[key][2] += (Xc * Xc).sum(axis=0)
        return state'''

    new_g = '''    def __init__(self, feature_cols: Sequence[str], target_col: str):
        self.feature_cols = list(feature_cols)
        self.target_col = target_col
        self.d = len(feature_cols)

    def estimate_state_nbytes(self) -> int:
        # v0.7.6-fix: pre-init bound for reduce() check_state.
        # Real state is O(C×d), C discovered dynamically from y.
        # Conservative C_ASSUMED=32 (typical multi-class); actual C > 32
        # → per-update check_state in reduce() catches real overflow.
        # Per class: 8 (n) + 8d (sum_x) + 8d (sum_xx) = 8 + 16d.
        return 32 * (8 + 16 * self.d)

    def init(self):
        return {}

    def update(self, state, batch):
        X = _stack_numeric(batch, self.feature_cols)
        y = _column_to_f64(batch.column(self.target_col))
        valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[valid]
        y = y[valid]
        if X.shape[0] == 0:
            return state
        for cls in np.unique(y):
            m = y == cls
            Xc = X[m]
            key = float(cls)
            if key not in state:
                state[key] = [0, np.zeros(self.d), np.zeros(self.d)]
            state[key][0] += int(Xc.shape[0])
            state[key][1] += Xc.sum(axis=0)
            state[key][2] += (Xc * Xc).sum(axis=0)
        return state'''

    s = replace_exact(s, anchor_g, new_g, tag="gaussian-estimate")

    # ---- Multinomial ----
    anchor_m = '''class _MultinomialStatsOp(ReduceOp):
    """Per-class: (n_c, sum_x[d]). X assumed >= 0 (counts)."""

    def __init__(self, feature_cols: Sequence[str], target_col: str):
        self.feature_cols = list(feature_cols)
        self.target_col = target_col
        self.d = len(feature_cols)

    def init(self):
        return {}'''

    new_m = '''class _MultinomialStatsOp(ReduceOp):
    """Per-class: (n_c, sum_x[d]). X assumed >= 0 (counts)."""

    def __init__(self, feature_cols: Sequence[str], target_col: str):
        self.feature_cols = list(feature_cols)
        self.target_col = target_col
        self.d = len(feature_cols)

    def estimate_state_nbytes(self) -> int:
        # Per class: 8 (n) + 8d (sum_x) = 8 + 8d.
        return 32 * (8 + 8 * self.d)

    def init(self):
        return {}'''

    s = replace_exact(s, anchor_m, new_m, tag="multinomial-estimate")

    # ---- Bernoulli ----
    anchor_b = '''class _BernoulliStatsOp(ReduceOp):
    """Per-class: (n_c, count_ones[d]). X assumed in {0, 1}."""

    def __init__(self, feature_cols: Sequence[str], target_col: str):
        self.feature_cols = list(feature_cols)
        self.target_col = target_col
        self.d = len(feature_cols)

    def init(self):
        return {}'''

    new_b = '''class _BernoulliStatsOp(ReduceOp):
    """Per-class: (n_c, count_ones[d]). X assumed in {0, 1}."""

    def __init__(self, feature_cols: Sequence[str], target_col: str):
        self.feature_cols = list(feature_cols)
        self.target_col = target_col
        self.d = len(feature_cols)

    def estimate_state_nbytes(self) -> int:
        # Per class: 8 (n) + 8d (count_ones) = 8 + 8d.
        return 32 * (8 + 8 * self.d)

    def init(self):
        return {}'''

    s = replace_exact(s, anchor_b, new_b, tag="bernoulli-estimate")

    p.write_text(s)
    compile_check(p)
    print("  OK: estimate_state_nbytes on all 3 NB ops")


# ============ main ============

def main():
    print("=== v0.7.6-fix ===")
    patch_binary_data()
    patch_nb_state_bytes()

    print("\n=== pytest tests/test_naive_bayes.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_naive_bayes.py", "-v"]
    )
    if r.returncode != 0:
        print("\n*** FAILURE ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.7.6-fix (test red)"], check=False)
        sys.exit(1)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.7.6-fix: Bernoulli accuracy (p0=0.1,p1=0.9 → Bayes~0.97); "
        "NB ReduceOps estimate_state_nbytes (silences 1 MiB warning)"])

    print("\n==> v0.7.6 complete.")


if __name__ == "__main__":
    main()