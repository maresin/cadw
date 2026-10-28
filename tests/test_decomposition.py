"""Tests for STL decomposition."""
from __future__ import annotations

import numpy as np
import pytest

from cadw.ts._decomposition import (
    STLResult, _loess_local_linear, _low_pass_seasonal_cycles, _nextodd,
    _tricube, stl_decompose,
)


# ── Unit tests: helpers ──────────────────────────────────────────

def test_nextodd():
    assert _nextodd(1) == 3
    assert _nextodd(2) == 3
    assert _nextodd(3) == 3
    assert _nextodd(4) == 5
    assert _nextodd(10) == 11
    assert _nextodd(11) == 11


def test_tricube_bounded():
    u = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    w = _tricube(u)
    assert w[0] == 0.0  # outside
    assert w[1] == 0.0  # at boundary
    assert w[2] > 0.0
    assert w[3] == 1.0  # at center
    assert w[5] == 0.0  # at boundary
    assert w[6] == 0.0  # outside


def test_loess_recovers_linear():
    """LOESS on perfectly linear data returns the data (interior)."""
    y = 2.0 + 0.5 * np.arange(50)
    smoothed = _loess_local_linear(y, window=7)
    # Interior points should match linear data closely
    np.testing.assert_allclose(smoothed[10:-10], y[10:-10], atol=1e-6)


def test_loess_recovers_constant():
    y = np.full(40, 3.5)
    smoothed = _loess_local_linear(y, window=9)
    np.testing.assert_allclose(smoothed, 3.5, atol=1e-6)


def test_loess_smooths_noise():
    rng = np.random.RandomState(0)
    y = 5.0 + rng.normal(0, 1.0, 200)
    smoothed = _loess_local_linear(y, window=15)
    # Smoothed series has lower variance than raw
    assert smoothed.std() < y.std()
    # Mean roughly preserved
    assert abs(smoothed.mean() - 5.0) < 0.5


def test_low_pass_seasonal_cycles_mean_zero():
    period = 4
    n = 40
    rng = np.random.RandomState(1)
    y = rng.normal(0, 1, n)
    season = _low_pass_seasonal_cycles(y, period, window_cycles=3)
    assert season.size == n
    assert abs(season.mean()) < 1e-9


def test_low_pass_seasonal_preserves_periodic_signal():
    """If input has clear period-p signal, low-pass preserves it."""
    period = 6
    n = 120
    t = np.arange(n)
    y = 2.0 * np.sin(2 * np.pi * t / period)
    season = _low_pass_seasonal_cycles(y, period, window_cycles=3)
    # Per-phase values should be close to the underlying sine pattern
    np.testing.assert_allclose(
        season[:period],
        np.sin(2 * np.pi * np.arange(period) / period) * 2.0,
        atol=0.2)


# ── stl_decompose: contract ─────────────────────────────────────

def test_stl_basic_shape():
    period = 12
    n = 120
    t = np.arange(n)
    y = 10 + 0.1 * t + 3 * np.sin(2 * np.pi * t / period)
    r = stl_decompose(y, period=period)
    assert isinstance(r, STLResult)
    assert r.trend.shape == (n,)
    assert r.seasonal.shape == (n,)
    assert r.remainder.shape == (n,)
    assert r.period == period
    # Additive decomposition exact: trend + seasonal + remainder == y
    np.testing.assert_allclose(
        r.trend + r.seasonal + r.remainder, y, atol=1e-10)


def test_stl_pure_seasonal():
    """On pure seasonal signal (no trend/noise), trend ~ constant,
    seasonal ~ the signal, remainder ~ 0."""
    period = 12
    n = 120
    t = np.arange(n)
    y = 3 * np.sin(2 * np.pi * t / period)
    r = stl_decompose(y, period=period)
    # Trend should be near 0
    assert np.abs(r.trend).max() < 0.5
    # Remainder should be small
    assert r.remainder.std() < 0.5


def test_stl_pure_trend():
    """On pure linear trend (no season), trend recovers signal,
    seasonal ~ 0."""
    n = 200
    t = np.arange(n)
    y = 5 + 0.5 * t
    r = stl_decompose(y, period=12)
    # Trend should be close to y
    np.testing.assert_allclose(r.trend, y, atol=1.0)
    # Seasonal near 0
    assert np.abs(r.seasonal).max() < 0.5


def test_stl_remainder_zero_on_perfect_decomposition():
    """Synthetic: y = trend + seasonal. Decomposition should split it
    with small remainder."""
    period = 12
    n = 240
    t = np.arange(n)
    trend_true = 10 + 0.2 * t
    seasonal_true = 3 * np.sin(2 * np.pi * t / period)
    y = trend_true + seasonal_true
    r = stl_decompose(y, period=period, n_iter=3)
    # Remainder small relative to signal amplitude
    assert r.remainder.std() < 0.5 * y.std()


def test_stl_deterministic():
    period = 12
    n = 120
    t = np.arange(n)
    rng = np.random.RandomState(42)
    y = 10 + 0.1 * t + 3 * np.sin(2 * np.pi * t / period) \
        + rng.normal(0, 0.2, n)
    r1 = stl_decompose(y, period=period)
    r2 = stl_decompose(y, period=period)
    np.testing.assert_array_equal(r1.trend, r2.trend)
    np.testing.assert_array_equal(r1.seasonal, r2.seasonal)
    np.testing.assert_array_equal(r1.remainder, r2.remainder)


# ── stl_decompose: errors ───────────────────────────────────────

def test_stl_bad_period():
    with pytest.raises(ValueError, match="period"):
        stl_decompose(np.zeros(100), period=1)


def test_stl_too_short():
    with pytest.raises(ValueError, match="2\\*period"):
        stl_decompose(np.zeros(10), period=12)


def test_stl_bad_n_iter():
    with pytest.raises(ValueError, match="n_iter"):
        stl_decompose(np.zeros(100), period=12, n_iter=0)


# ── AST guard ────────────────────────────────────────────────────

def test_decomposition_no_bench_deps_in_source():
    import ast
    from pathlib import Path
    src = Path("cadw/ts/_decomposition.py").read_text()
    tree = ast.parse(src)
    forbidden = ("sklearn", "statsmodels", "torch", "pyod", "aeon",
                 "merlion", "kats", "lifelines", "implicit", "ruptures",
                 "scipy")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                assert top not in forbidden, (
                    f"forbidden import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            assert top not in forbidden, (
                f"forbidden import-from: {node.module}")

def test_stl_decompose_guard_raises_small_limit():
    """v0.9.87-fix4: stl_decompose guard raises before alloc."""
    import numpy as np
    import pytest
    from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
    from cadw.ts._decomposition import stl_decompose

    n = 10000
    y = np.sin(2 * np.pi * np.arange(n) / 12)
    mg = MemoryGuard(max_memory_bytes=1024)
    with pytest.raises(MemoryLimitExceeded):
        stl_decompose(y, period=12, memory_guard=mg)


def test_stl_decompose_no_guard_still_works():
    """v0.9.87-fix4: memory_guard=None -> no check, same result."""
    import numpy as np
    from cadw.ts._decomposition import stl_decompose

    n = 200
    t = np.arange(n)
    y = 5.0 + 2.0 * np.sin(2 * np.pi * t / 12) + 0.05 * t
    r = stl_decompose(y, period=12)
    assert r.trend.shape == (n,)
    assert r.seasonal.shape == (n,)
    assert r.remainder.shape == (n,)

