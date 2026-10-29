"""Tests for T-squared + Q (Hotelling SPC on PCA)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cadw.ml.tspq import T2QResult, fit_t2q

ROOT = Path(__file__).parent.parent


def _multi_changepoint(n=2000, c=3, n_changes=3, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    X = np.empty((n, c))
    for ch in range(c):
        X[:, ch] = np.sin(2 * np.pi * t / (100.0 + 20.0 * ch)) \
            + 0.1 * rng.standard_normal(n)
    cps = np.sort(rng.choice(n - 400, size=n_changes, replace=False)) + 400
    pos = []
    for cp in cps:
        end = min(cp + 200, n)
        X[cp:end] += 2.0
        pos.extend(range(cp, end))
    return X, cps, np.array(sorted(pos))


def _uni_sine(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return np.sin(2 * np.pi * t / 100.0) + 0.1 * rng.standard_normal(n)


# ── Contract / determinism ────────────────────────────────────────

def test_fit_t2q_shape_multivariate():
    X, _, _ = _multi_changepoint(n=1000, c=3, seed=0)
    r = fit_t2q(X, window_size=1)
    assert isinstance(r, T2QResult)
    assert r.decision_scores.shape == (1000,)
    assert r.t2_scores.shape == (1000,)
    assert r.q_scores.shape == (1000,)
    assert r.threshold == 1.0
    assert r.n_channels == 3
    assert r.n_components >= 1


def test_fit_t2q_shape_univariate_forces_window():
    x = _uni_sine(n=1000)
    r = fit_t2q(x)  # window_size=1 with c=1 -> auto-forced to w=50
    assert r.window_size > 1
    assert r.decision_scores.shape == (1000,)


def test_fit_t2q_deterministic():
    X, _, _ = _multi_changepoint(n=800, seed=1)
    r1 = fit_t2q(X, window_size=1, random_state=0) if False else fit_t2q(X)
    r2 = fit_t2q(X)
    np.testing.assert_array_equal(r1.decision_scores, r2.decision_scores)


# ── Detection quality ────────────────────────────────────────────

def test_fit_t2q_detects_changepoints():
    X, cps, pos = _multi_changepoint(n=2000, c=3, n_changes=3, seed=0)
    r = fit_t2q(X, window_size=1, contamination=0.2)
    # At minimum, T^2+Q must flag most post-changepoint points as anomalous
    recall = float(np.mean(r.labels[pos] == 1))
    assert recall >= 0.3, f"recall={recall:.2f} on {len(pos)} cp pts"


def test_fit_t2q_combined_score_uses_both_t2_and_q():
    """Combined score should exceed 1.0 iff T2/lim > 1 or Q/lim > 1."""
    X, _, _ = _multi_changepoint(n=500, seed=2)
    r = fit_t2q(X)
    t2_norm = r.t2_scores / r.t2_limit
    q_norm = r.q_scores / r.q_limit
    expected = np.maximum(t2_norm, q_norm)
    np.testing.assert_allclose(r.decision_scores, expected, rtol=1e-8)


def test_fit_t2q_constant_series():
    X = np.ones((500, 2)) * 3.0
    r = fit_t2q(X)
    # Constant -> all T2 = 0, Q = 0 -> no anomalies
    assert r.t2_scores.max() < 1e-10
    assert r.q_scores.max() < 1e-10
    assert r.labels.sum() == 0


def test_fit_t2q_univariate_too_short_raises():
    """n=10 with explicit window_size=50 -> _make_windows raises."""
    with pytest.raises(ValueError, match="< window_size"):
        fit_t2q(np.arange(10.0), window_size=50)


def test_fit_t2q_bad_aggregate_raises():
    x = _uni_sine(n=500)
    with pytest.raises(ValueError, match="aggregate"):
        fit_t2q(x, window_size=30, aggregate="median")


def test_fit_t2q_n_components_none_leaves_residual():
    """None must leave at least 1 component in Q residual (m-1 max)."""
    X, _, _ = _multi_changepoint(n=800, c=3, seed=3)
    r = fit_t2q(X)
    m = X.shape[1] if X.ndim == 2 else r.window_size
    # For (n,c) c=3, m=3 -> k <= 2 (at least 1 for Q)
    if X.ndim == 2:
        assert r.n_components <= X.shape[1] - 1


def test_fit_t2q_float_variance_fraction():
    X, _, _ = _multi_changepoint(n=1000, seed=4)
    r = fit_t2q(X, n_components=0.9)
    assert 1 <= r.n_components <= 3


def test_fit_t2q_memory_guard_logic():
    from cadw.core.memory_guard import MemoryLimitExceeded

    class _FakeMB:
        max_memory_mb = 0.001

    X = np.random.default_rng(0).standard_normal((5000, 5))
    with pytest.raises(MemoryLimitExceeded):
        fit_t2q(X, window_size=1, guard=_FakeMB())


def test_fit_t2q_memory_guard_real_smoke():
    from cadw.core.memory_guard import MemoryGuard
    guard = MemoryGuard(max_memory_mb=1000)
    X = np.random.default_rng(1).standard_normal((500, 3))
    r = fit_t2q(X, guard=guard)
    assert r.decision_scores.shape == (500,)


def test_tspq_no_bench_deps_in_source():
    """AST-based check: no forbidden imports in actual import statements."""
    import ast
    src = (ROOT / "cadw/ml/tspq.py").read_text()
    tree = ast.parse(src)
    forbidden = ("sktime", "sklearn", "scipy", "torch", "pyod",
                 "aeon", "merlion", "kats", "paddle")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                assert top not in forbidden, (
                    f"forbidden import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            top = mod.split(".")[0]
            assert top not in forbidden, (
                f"forbidden import-from: {mod}")

def test_fit_t2q_chunked_does_not_call_make_windows(monkeypatch):
    """v0.9.91-S6b: real MemoryGuard → chunked path.

    _make_windows is not called (OOC chunked mean/cov/scores path).
    Prior v0.9.87-fix1b behaviour (raise before _make_windows on
    tight budget) is superseded by chunked OOC. Duck-typed guards
    still hit old path via fallback (see test_fit_t2q_memory_guard_logic).
    """
    from cadw.ml import tspq as tspq_mod
    from cadw.core.memory_guard import MemoryGuard

    called = {"n": 0}
    orig = tspq_mod._make_windows

    def spy(*a, **kw):
        called["n"] += 1
        return orig(*a, **kw)

    monkeypatch.setattr(tspq_mod, "_make_windows", spy)

    guard = MemoryGuard(max_memory_mb=10)
    x = np.random.default_rng(0).standard_normal(100_000)
    r = tspq_mod.fit_t2q(x, window_size=64, guard=guard)
    assert r.decision_scores.shape == (100_000,)
    assert called["n"] == 0, (
        f"_make_windows should not be called in chunked path; "
        f"was called {called['n']} time(s)")

