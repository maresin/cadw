"""Tests for KMeansAD anomaly detection."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cadw.ml.kmeans_ad import (KMeansADResult, _distances_sq,
                               _kmeans_plus_plus_init, _lloyd,
                               fit_kmeans_ad)

ROOT = Path(__file__).parent.parent


def _sine_segments(n=5000, period=100, n_segments=5, seg_len=30,
                   seed=0, amplitude=5.0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x = np.sin(2 * np.pi * t / period) + 0.05 * rng.standard_normal(n)
    starts = rng.choice(n - seg_len, size=n_segments, replace=False)
    pos = []
    for s in starts:
        x[s:s + seg_len] = amplitude
        pos.extend(range(s, s + seg_len))
    return x, np.array(sorted(pos))


def _sine_spikes(n=5000, n_spikes=10, seed=0, amplitude=20.0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x = np.sin(2 * np.pi * t / 100.0) + 0.05 * rng.standard_normal(n)
    pos = rng.choice(n, size=n_spikes, replace=False)
    x[pos] += amplitude
    return x, pos


# ── Unit: k-means++ init ────────────────────────────────────────

def test_kmeans_plus_plus_shape():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 10))
    C = _kmeans_plus_plus_init(X, k=5, rng=rng)
    assert C.shape == (5, 10)
    # All centroids are actual samples
    for i in range(5):
        dists = np.linalg.norm(X - C[i], axis=1)
        assert dists.min() < 1e-12


def test_kmeans_plus_plus_too_few_samples():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((3, 5))
    with pytest.raises(ValueError, match="< n_clusters"):
        _kmeans_plus_plus_init(X, k=5, rng=rng)


def test_kmeans_plus_plus_deterministic():
    X = np.random.default_rng(0).standard_normal((100, 10))
    C1 = _kmeans_plus_plus_init(X, k=5, rng=np.random.default_rng(42))
    C2 = _kmeans_plus_plus_init(X, k=5, rng=np.random.default_rng(42))
    np.testing.assert_array_equal(C1, C2)


# ── Unit: distances_sq formula ──────────────────────────────────

def test_distances_sq_matches_explicit():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 5))
    C = rng.standard_normal((3, 5))
    d_sq = _distances_sq(X, C)
    explicit = np.zeros((20, 3))
    for i in range(20):
        for j in range(3):
            explicit[i, j] = np.sum((X[i] - C[j]) ** 2)
    np.testing.assert_allclose(d_sq, explicit, rtol=1e-10)


# ── Unit: Lloyd converges on blobs ──────────────────────────────

def test_lloyd_converges_blobs():
    rng = np.random.default_rng(0)
    # 3 well-separated blobs, 150 points, variance 1 per dim
    # -> theoretical inertia = 150 points * 2 dims = 300.
    blobs = np.vstack([
        rng.standard_normal((50, 2)) + [0, 0],
        rng.standard_normal((50, 2)) + [10, 0],
        rng.standard_normal((50, 2)) + [0, 10],
    ])
    C = _kmeans_plus_plus_init(blobs, k=3, rng=rng)
    C_final, labels, _, n_iter, inertia = _lloyd(
        blobs, C, max_iter=100, tol=1e-4, rng=rng)
    assert inertia < 400.0, (
        f"inertia={inertia:.2f} (theoretical ~300; >400 means no convergence)")
    assert n_iter <= 100


def test_lloyd_empty_cluster_reinit():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 3))
    # Initialize 5 centroids all at the same point -> 4 will be empty
    C = np.tile(X[0], (5, 1))
    C_new, labels, _, _, _ = _lloyd(X, C, max_iter=20, tol=1e-4, rng=rng)
    # All 5 centroids should be distinct after Lloyd
    unique_count = len({tuple(np.round(c, 6)) for c in C_new})
    assert unique_count == 5, f"only {unique_count}/5 unique centroids"


# ── fit_kmeans_ad contract ──────────────────────────────────────

def test_fit_kmeans_ad_basic_shape():
    x, _ = _sine_spikes(n=2000, n_spikes=5, seed=0)
    r = fit_kmeans_ad(x, window_size=50, n_clusters=8)
    assert isinstance(r, KMeansADResult)
    assert r.decision_scores.shape == (2000,)
    assert r.labels.shape == (2000,)
    assert r.window_scores.shape == (2000 - 50 + 1,)
    assert r.centroids.shape == (8, 50)
    assert r.labels_per_window.shape == (2000 - 50 + 1,)


def test_fit_kmeans_ad_deterministic():
    x, _ = _sine_spikes(n=2000, n_spikes=5, seed=3)
    r1 = fit_kmeans_ad(x, window_size=50, n_clusters=8, random_state=42)
    r2 = fit_kmeans_ad(x, window_size=50, n_clusters=8, random_state=42)
    np.testing.assert_array_equal(r1.decision_scores, r2.decision_scores)
    np.testing.assert_allclose(r1.centroids, r2.centroids)


def test_fit_kmeans_ad_detects_segments():
    x, pos = _sine_segments(n=5000, n_segments=5, seg_len=30, seed=0)
    r = fit_kmeans_ad(x, window_size=50, n_clusters=8,
                      contamination=0.05, random_state=0)
    recall = float(np.mean(r.labels[pos] == 1))
    assert recall >= 0.4, f"segment recall={recall:.2f} on {len(pos)} pts"


def test_fit_kmeans_ad_detects_spikes():
    x, pos = _sine_spikes(n=5000, n_spikes=10, seed=0)
    r = fit_kmeans_ad(x, window_size=50, n_clusters=8,
                      contamination=0.05, random_state=0)
    recall = float(np.mean(r.labels[pos] == 1))
    assert recall >= 0.3, f"spike recall={recall:.2f} on {len(pos)} pts"


def test_fit_kmeans_ad_constant_series():
    x = np.ones(500) * 3.0
    r = fit_kmeans_ad(x, window_size=50, n_clusters=4, random_state=0)
    # All windows identical -> all scores 0 -> labels = 0
    assert r.labels.sum() == 0


def test_fit_kmeans_ad_short_series_raises():
    with pytest.raises(ValueError, match="< window_size"):
        fit_kmeans_ad(np.arange(30.0), window_size=50)


def test_fit_kmeans_ad_too_few_windows_raises():
    x = np.arange(55.0)
    with pytest.raises(ValueError, match="< n_clusters"):
        fit_kmeans_ad(x, window_size=50, n_clusters=10)


def test_fit_kmeans_ad_bad_aggregate_raises():
    x = np.random.default_rng(0).standard_normal(200)
    with pytest.raises(ValueError, match="aggregate"):
        fit_kmeans_ad(x, window_size=30, aggregate="median")


def test_fit_kmeans_ad_contamination_threshold():
    x, _ = _sine_segments(n=2000, n_segments=3, seg_len=30,
                          seed=2, amplitude=5.0)
    r = fit_kmeans_ad(x, window_size=50, n_clusters=4,
                      contamination=0.05, random_state=0)
    n_flagged = int(r.labels.sum())
    assert 30 <= n_flagged <= 200, (
        f"n_flagged={n_flagged} outside [30, 200] "
        f"(n=2000, contamination=0.05 -> ~100 expected)")


def test_fit_kmeans_ad_memory_guard_logic():
    from cadw.core.memory_guard import MemoryLimitExceeded

    class _FakeMB:
        max_memory_mb = 0.001

    x = np.random.default_rng(0).standard_normal(5000)
    with pytest.raises(MemoryLimitExceeded):
        fit_kmeans_ad(x, window_size=100, guard=_FakeMB())


def test_kmeans_ad_no_bench_deps_in_source():
    """AST-based check: no forbidden imports."""
    import ast
    src = (ROOT / "cadw/ml/kmeans_ad.py").read_text()
    tree = ast.parse(src)
    forbidden = ("sktime", "sklearn", "scipy", "torch", "pyod",
                 "aeon", "merlion", "kats")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                assert top not in forbidden, f"forbidden: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            assert top not in forbidden, f"forbidden: {node.module}"

def test_fit_kmeans_ad_chunked_does_not_call_prepare_windows(monkeypatch):
    """v0.9.91-S6c: real MemoryGuard → chunked path.

    _prepare_windows is not called (chunked reservoir + Lloyd +
    score path). Prior v0.9.87-fix1b behaviour (raise before
    _prepare_windows) is superseded by chunked OOC on tight budget.
    Duck-typed guards still hit old path via fallback
    (test_fit_kmeans_ad_memory_guard_logic).
    """
    from cadw.ml import kmeans_ad as km_mod
    from cadw.core.memory_guard import MemoryGuard

    called = {"n": 0}
    orig = km_mod._prepare_windows

    def spy(*a, **kw):
        called["n"] += 1
        return orig(*a, **kw)

    monkeypatch.setattr(km_mod, "_prepare_windows", spy)

    guard = MemoryGuard(max_memory_mb=10)
    x = np.random.default_rng(0).standard_normal(100_000)
    r = km_mod.fit_kmeans_ad(x, window_size=64, guard=guard)
    assert r.decision_scores.shape == (100_000,)
    assert called["n"] == 0, (
        f"_prepare_windows should not be called in chunked path; "
        f"was called {called['n']} time(s)")

