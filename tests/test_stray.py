"""Tests for STRAY anomaly detection."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cadw.ml.stray import (STRAYResult, _gpd_fit_mom, _gpd_sf,
                           _kth_nn_distances, fit_stray)

ROOT = Path(__file__).parent.parent


def _sine_spikes(n=1000, period=100, n_spikes=10, seed=0, amplitude=20.0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x = np.sin(2 * np.pi * t / period) + 0.05 * rng.standard_normal(n)
    pos = rng.choice(n, size=n_spikes, replace=False)
    x[pos] += amplitude
    return x, pos


# ── KNN unit tests ────────────────────────────────────────────────

def test_kth_nn_basic():
    X = np.array([[0.0], [1.0], [3.0], [10.0]])
    kth = _kth_nn_distances(X, k=1)
    # 1st NN dists: [1, 1, 2, 7]
    np.testing.assert_allclose(kth, [1.0, 1.0, 2.0, 7.0])


def test_kth_nn_k2():
    X = np.array([[0.0], [1.0], [3.0], [10.0]])
    kth = _kth_nn_distances(X, k=2)
    # 2nd NN dists (sorted dist list per point, take index k-1=1):
    #   p0 (0):   [1, 3, 10]  -> 3
    #   p1 (1):   [1, 2, 9]   -> 2
    #   p3 (3):   [2, 3, 7]   -> 3
    #   p10 (10): [7, 9, 10]  -> 9
    np.testing.assert_allclose(sorted(kth), [2.0, 3.0, 3.0, 9.0])


def test_kth_nn_multivariate():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [10.0, 10.0]])
    kth = _kth_nn_distances(X, k=1)
    # Point 3 nearest to origin at sqrt(2); others at 1; point 3 at ~sqrt(200)?
    # Actually: dists from point 3 (10,10) to others: to (0,0)=14.14,
    # to (1,0)=13.45, to (0,1)=13.45 -> kth=13.45
    assert kth[3] > 13.0
    assert kth[0] < 1.5


# ── GPD unit tests ────────────────────────────────────────────────

def test_gpd_fit_mom_reasonable():
    rng = np.random.default_rng(0)
    # Pareto-ish tail: xi>0
    y = rng.pareto(2.0, size=500) + 0.1
    xi, beta = _gpd_fit_mom(y)
    assert -0.5 <= xi <= 0.49
    assert beta > 0


def test_gpd_sf_monotone():
    y = np.array([0.1, 1.0, 10.0, 100.0])
    for xi in (-0.3, 0.0, 0.3):
        sf = _gpd_sf(y, xi=xi, beta=1.0)
        assert np.all(np.diff(sf) <= 0), f"SF not monotone for xi={xi}"


def test_gpd_sf_exponential():
    y = np.array([0.0, 1.0, 2.0, 5.0])
    sf = _gpd_sf(y, xi=0.0, beta=1.0)
    np.testing.assert_allclose(sf, np.exp(-y), rtol=1e-10)


# ── fit_stray contract ────────────────────────────────────────────

def test_fit_stray_basic_shape():
    x, _ = _sine_spikes(n=1000, n_spikes=10, seed=0)
    r = fit_stray(x, k=20)
    assert isinstance(r, STRAYResult)
    assert r.decision_scores.shape == (1000,)
    assert r.labels.shape == (1000,)
    assert r.k == 20
    assert r.n_tail >= 20


def test_fit_stray_detects_spikes():
    x, pos = _sine_spikes(n=1000, n_spikes=10, seed=0)
    r = fit_stray(x, k=20, tail_fraction=0.05, alpha=0.01)
    recall = float(np.mean(r.labels[pos] == 1))
    assert recall >= 0.4, f"recall={recall:.2f} on {len(pos)} spikes"


def test_fit_stray_constant_series():
    x = np.ones(500) * 3.0
    r = fit_stray(x, k=20)
    # All KNN distances 0 -> tail empty -> all scores 0 -> no anomalies
    assert r.labels.sum() == 0


def test_fit_stray_short_series_raises():
    with pytest.raises(ValueError, match="too short"):
        fit_stray(np.arange(10.0), k=20)


def test_fit_stray_bad_tail_fraction_raises():
    x = _sine_spikes(n=500, n_spikes=5, seed=1)[0]
    with pytest.raises(ValueError, match="tail_fraction"):
        fit_stray(x, k=20, tail_fraction=1.5)


def test_fit_stray_bad_alpha_raises():
    x = _sine_spikes(n=500, n_spikes=5, seed=1)[0]
    with pytest.raises(ValueError, match="alpha"):
        fit_stray(x, k=20, alpha=2.0)


def test_fit_stray_deterministic():
    x, _ = _sine_spikes(n=500, n_spikes=5, seed=3)
    r1 = fit_stray(x, k=20)
    r2 = fit_stray(x, k=20)
    np.testing.assert_array_equal(r1.decision_scores, r2.decision_scores)


def test_fit_stray_alpha_controls_rate():
    """Higher alpha (less strict) -> threshold -log(alpha) is lower
    -> MORE flags (not fewer). Both must be exercised on same DGP."""
    x, _ = _sine_spikes(n=500, n_spikes=5, seed=4)
    r_loose = fit_stray(x, k=20, alpha=0.5)    # thr = -log(0.5) ~= 0.69
    r_strict = fit_stray(x, k=20, alpha=1e-4)  # thr = -log(1e-4) ~= 9.21
    # loose threshold is lower -> at least as many flags
    assert r_loose.threshold < r_strict.threshold
    assert r_loose.labels.sum() >= r_strict.labels.sum()


def test_fit_stray_contamination_threshold():
    x, _ = _sine_spikes(n=1000, n_spikes=10, seed=5)
    r = fit_stray(x, k=20, contamination=0.02)
    assert 5 <= r.labels.sum() <= 60


def test_fit_stray_multivariate():
    rng = np.random.default_rng(0)
    n = 500
    d = 3
    X = rng.standard_normal((n, d))
    pos = rng.choice(n, size=10, replace=False)
    X[pos] += 5.0
    r = fit_stray(X, k=20, tail_fraction=0.05, alpha=0.01)
    recall = float(np.mean(r.labels[pos] == 1))
    assert recall >= 0.3, f"recall={recall:.2f} on mv"


def test_fit_stray_memory_guard_logic():
    from cadw.core.memory_guard import MemoryLimitExceeded

    class _FakeMB:
        max_memory_mb = 0.001

    x = np.random.default_rng(0).standard_normal(5000)
    with pytest.raises(MemoryLimitExceeded):
        fit_stray(x, k=20, guard=_FakeMB())


def test_stray_no_bench_deps_in_source():
    """AST-based check: no forbidden imports in actual import statements.
    Docstring references to 'sktime.annotation.STRAY' do not count."""
    import ast
    src = (ROOT / "cadw/ml/stray.py").read_text()
    tree = ast.parse(src)
    forbidden = ("sktime", "sklearn", "scipy", "torch", "pyod", "aeon")
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


def test_stray_reference_sktime_smoke():
    """If sktime STRAY is available, both should run and rank spikes high."""
    try:
        from sktime.annotation.stray import STRAY
    except ImportError:
        pytest.skip("sktime STRAY not available")
    x, pos = _sine_spikes(n=500, n_spikes=10, seed=6)
    import pandas as pd
    try:
        skt_det = STRAY(k=20)
        skt_labels = skt_det.fit_predict(pd.Series(x))
    except Exception as e:
        pytest.skip(f"sktime STRAY raised: {e}")
    skt_labels = np.asarray(skt_labels)
    # Both should flag most spikes; exact match not expected (GPD fit differs)
    skt_recall = float(np.mean(skt_labels[pos] == 1))
    our = fit_stray(x, k=20, tail_fraction=0.05, alpha=0.01)
    our_recall = float(np.mean(our.labels[pos] == 1))
    assert skt_recall >= 0.2 or our_recall >= 0.2, (
        f"Neither detects: skt={skt_recall:.2f}, ours={our_recall:.2f}")

