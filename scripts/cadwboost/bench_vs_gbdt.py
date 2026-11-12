"""CADWBoost vs cadw.ml.gbdt — comprehensive benchmark.

7 datasets: smooth, nonlinear, noisy, XOR-like, binary, count, imbalanced.
Metrics: rmse/acc, fit time, state bytes.

API compat: cadw.ml.gbdt takes CadwDataFrame; CADWBoost takes numpy.
Both fit on same data.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_PROJ = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(_PROJ))
sys.path.insert(0, str(_PROJ / ".pingis" / "drafts" / "cadwboost"))

import numpy as np
import pyarrow as pa

from cadw import CadwDataFrame, MemoryGuard
from cadw.ml.gbdt import fit_gbdt_regressor, fit_gbdt_classifier

from cadwboost import CADWBoost


def _make_df(X, y):
    cols = {f"x{j}": X[:, j] for j in range(X.shape[1])}
    cols["y"] = y
    return CadwDataFrame.from_table(
        pa.table(cols), memory_guard=MemoryGuard(max_memory_mb=2000))


def _gbdt_regression(X, y, n_estimators=50, max_depth=4, lr=0.15):
    df = _make_df(X, y)
    Xs = [f"x{j}" for j in range(X.shape[1])]
    t0 = time.perf_counter()
    m = fit_gbdt_regressor(df, Xs, "y",
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            learning_rate=lr,
                            memory_guard=df._memory_guard)
    fit_t = time.perf_counter() - t0
    p = m.predict(X)
    return p, fit_t


def _gbdt_classification(X, y, n_estimators=50, max_depth=4, lr=0.15):
    df = _make_df(X, y)
    Xs = [f"x{j}" for j in range(X.shape[1])]
    t0 = time.perf_counter()
    m = fit_gbdt_classifier(df, Xs, "y",
                             n_estimators=n_estimators,
                             max_depth=max_depth,
                             learning_rate=lr,
                             memory_guard=df._memory_guard)
    fit_t = time.perf_counter() - t0
    proba = m.predict_proba(X)
    return proba, fit_t


def _cb_regression(X, y, n_estimators=50, max_depth=4, lr=0.15, **kw):
    t0 = time.perf_counter()
    m = CADWBoost(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=lr, random_state=42, **kw)
    m.fit(X, y)
    fit_t = time.perf_counter() - t0
    return m.predict(X), fit_t, m


def _cb_classification(X, y, n_estimators=50, max_depth=4, lr=0.15, **kw):
    t0 = time.perf_counter()
    m = CADWBoost(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=lr, loss="logistic", random_state=42, **kw)
    m.fit(X, y)
    fit_t = time.perf_counter() - t0
    return m.predict_proba(X), fit_t, m


def _r2(y, p):
    return 1.0 - np.mean((y - p) ** 2) / np.var(y)


def _rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def _report(name, metric, gbdt_val, cb_val, gbdt_t, cb_t, cb_state):
    delta = (cb_val - gbdt_val) / max(abs(gbdt_val), 1e-9)
    verdict = ("better" if delta < -0.05
               else "comparable" if abs(delta) < 0.10
               else "worse")
    print(f"\n=== {name} ===")
    print(f"  cadw.gbdt: {metric}={gbdt_val:.4f}  fit={gbdt_t:.2f}s")
    print(f"  CADWBoost: {metric}={cb_val:.4f}  fit={cb_t:.2f}s  "
          f"state={cb_state} B")
    print(f"  delta: {delta:+.1%}  speed: {cb_t/max(gbdt_t, 1e-6):.2f}x  "
          f"[{verdict}]")
    return verdict


def bench_smooth():
    rng = np.random.RandomState(42)
    n = 5000
    X = rng.randn(n, 5)
    y = 2.0 * X[:, 0] + 0.5 * X[:, 1] + 0.1 * rng.randn(n)
    p_g, t_g = _gbdt_regression(X, y)
    p_c, t_c, m = _cb_regression(X, y)
    _report("Smooth linear (d=5)", "rmse", _rmse(y, p_g), _rmse(y, p_c),
            t_g, t_c, m.state_nbytes())


def bench_nonlinear():
    rng = np.random.RandomState(43)
    n = 5000
    X = rng.randn(n, 5)
    y = np.sin(2 * X[:, 0]) * np.cos(X[:, 1]) + 0.1 * rng.randn(n)
    p_g, t_g = _gbdt_regression(X, y)
    p_c, t_c, m = _cb_regression(X, y)
    _report("Nonlinear sin·cos (d=5)", "rmse", _rmse(y, p_g),
            _rmse(y, p_c), t_g, t_c, m.state_nbytes())


def bench_noisy():
    rng = np.random.RandomState(44)
    n = 5000
    X = rng.randn(n, 10)
    y = X[:, 0] + 0.5 * rng.randn(n)
    p_g, t_g = _gbdt_regression(X, y)
    p_c, t_c, m = _cb_regression(X, y)
    _report("Noisy (d=10, σ=0.5)", "rmse", _rmse(y, p_g),
            _rmse(y, p_c), t_g, t_c, m.state_nbytes())


def bench_xor():
    rng = np.random.RandomState(45)
    n = 5000
    X = rng.randn(n, 4)
    y = ((X[:, 0] > 0) * (X[:, 1] > 0)
         + (X[:, 2] > 0) * (X[:, 3] > 0)).astype(np.float64)
    y += 0.05 * rng.randn(n)
    p_g, t_g = _gbdt_regression(X, y)
    p_c, t_c, m = _cb_regression(X, y)
    _report("XOR-like interactions (d=4)", "rmse",
            _rmse(y, p_g), _rmse(y, p_c), t_g, t_c, m.state_nbytes())


def bench_binary():
    rng = np.random.RandomState(46)
    n = 5000
    X = rng.randn(n, 5)
    logit = 1.5 * X[:, 0] - 2.0 * X[:, 1] + 0.3 * rng.randn(n)
    y = (logit > 0).astype(np.float64)
    p_g, t_g = _gbdt_classification(X, y)
    p_c, t_c, m = _cb_classification(X, y)
    acc_g = float(((p_g > 0.5) == y).mean())
    acc_c = float(((p_c > 0.5) == y).mean())
    _report("Binary logistic (d=5)", "acc", acc_g, acc_c,
            t_g, t_c, m.state_nbytes())


def bench_imbalanced():
    rng = np.random.RandomState(47)
    n = 5000
    X = rng.randn(n, 5)
    logit = 1.5 * X[:, 0] + 0.5 * rng.randn(n)
    y = (logit > 1.5).astype(np.float64)  # ~5% positive
    p_g, t_g = _gbdt_classification(X, y)
    p_c, t_c, m = _cb_classification(X, y)
    # AUC-like: mean prob for positive > mean prob for negative
    auc_g = p_g[y == 1].mean() - p_g[y == 0].mean()
    auc_c = p_c[y == 1].mean() - p_c[y == 0].mean()
    _report("Imbalanced binary (~5% pos)", "score_gap",
            float(auc_g), float(auc_c), t_g, t_c, m.state_nbytes())


def bench_count():
    rng = np.random.RandomState(48)
    n = 5000
    X = rng.randn(n, 5)
    mu = np.exp(0.5 * X[:, 0] - 0.3 * X[:, 1])
    y = rng.poisson(mu).astype(np.float64)
    # Compare MSE on log(y+1) for both (gbdt doesn't have poisson)
    p_g, t_g = _gbdt_regression(X, y)
    p_c, t_c, m = _cb_regression(X, y, loss="poisson")
    print(f"\n=== Count Poisson (d=5, mean_y={y.mean():.1f}) ===")
    print(f"  cadw.gbdt (mse):        rmse={_rmse(y, p_g):.4f}  "
          f"fit={t_g:.2f}s")
    print(f"  CADWBoost (poisson):    rmse={_rmse(y, p_c):.4f}  "
          f"fit={t_c:.2f}s  state={m.state_nbytes()}B")


def bench_wide():
    rng = np.random.RandomState(49)
    n = 5000
    X = rng.randn(n, 50)
    # Only 5 features matter
    y = (2.0 * X[:, 0] + 1.5 * X[:, 1] - X[:, 2]
         + 0.5 * X[:, 3] + 0.3 * X[:, 4] + 0.1 * rng.randn(n))
    p_g, t_g = _gbdt_regression(X, y)
    p_c, t_c, m = _cb_regression(X, y)
    _report("Wide (d=50, 5 informative)", "rmse",
            _rmse(y, p_g), _rmse(y, p_c), t_g, t_c, m.state_nbytes())


def main():
    print("=" * 70)
    print("CADWBoost vs cadw.ml.gbdt")
    print("  common: n_estimators=50, max_depth=4, lr=0.15, n=5000")
    print("  CADWBoost defaults: symmetric, quantile binning, mse/logistic")
    print("=" * 70)

    for fn in (bench_smooth, bench_nonlinear, bench_noisy, bench_xor,
               bench_binary, bench_imbalanced, bench_count, bench_wide):
        try:
            fn()
        except Exception as e:
            print(f"\n=== {fn.__name__}: ERROR ===")
            print(f"  {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
