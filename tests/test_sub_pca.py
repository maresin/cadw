"""Tests for Sub-PCA anomaly detection."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cadw.ml.sub_pca import (SubPCAResult, _map_max, _map_mean,
                             fit_sub_pca)

ROOT = Path(__file__).parent.parent


def _sine_spikes(n=1000, period=100, n_spikes=20, seed=0,
                 amplitude=20.0):
    """Sine + isolated spikes.

    Amplitude 20 + mean aggregation needed (see _sine_spikes notes
    in fit_sub_pca tests): max aggregation spreads each spike to ~w
    timestamps, putting threshold inside spike-affected zone.
    Mean aggregation gives spike position ~19, neighbor at d=25 ~9.8,
    neighbor at d=49 ~1 — clear separation from natural ~0.6.

    Amplitude 20 (vs 10) provides additional margin: spike residual
    = A*sqrt(1 - sum_j c_j[i]^2) ~ 19 near sine zeros, ~14 near peaks.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x = np.sin(2 * np.pi * t / period) + 0.1 * rng.standard_normal(n)
    pos = rng.choice(n, size=n_spikes, replace=False)
    x[pos] += amplitude
    return x, pos


# ─── Aggregation unit tests ──────────────────────────────────────

def test_map_max_correctness():
    ws = np.array([1.0, 2.0, 3.0, 4.0])  # n_w=4, w=3 -> n=6
    out = _map_max(ws, w=3, n=6)
    # t=0: max(ws[0]) = 1
    # t=1: max(ws[0:2]) = 2
    # t=2: max(ws[0:3]) = 3
    # t=3: max(ws[1:4]) = 4
    # t=4: max(ws[2:4]) = 4
    # t=5: max(ws[3:4]) = 4
    expected = np.array([1.0, 2.0, 3.0, 4.0, 4.0, 4.0])
    np.testing.assert_allclose(out, expected)


def test_map_mean_correctness():
    ws = np.array([1.0, 2.0, 3.0, 4.0])
    out = _map_mean(ws, w=3, n=6)
    # t=0: mean(ws[0]) = 1
    # t=1: mean(ws[0:2]) = 1.5
    # t=2: mean(ws[0:3]) = 2
    # t=3: mean(ws[1:4]) = 3
    # t=4: mean(ws[2:4]) = 3.5
    # t=5: mean(ws[3:4]) = 4
    expected = np.array([1.0, 1.5, 2.0, 3.0, 3.5, 4.0])
    np.testing.assert_allclose(out, expected)


# ─── fit_sub_pca contract ────────────────────────────────────────

def test_fit_sub_pca_basic_shape():
    x, _ = _sine_spikes(n=500, seed=0)
    r = fit_sub_pca(x, window_size=50, n_components=10)
    assert isinstance(r, SubPCAResult)
    assert r.decision_scores.shape == (500,)
    assert r.labels.shape == (500,)
    assert r.components.shape == (10, 50)
    assert r.mean.shape == (50,)
    assert r.n_components == 10


def test_fit_sub_pca_deterministic():
    x, _ = _sine_spikes(n=500, seed=42)
    r1 = fit_sub_pca(x, window_size=50, n_components=10, contamination=0.05)
    r2 = fit_sub_pca(x, window_size=50, n_components=10, contamination=0.05)
    np.testing.assert_array_equal(r1.decision_scores, r2.decision_scores)
    np.testing.assert_array_equal(r1.labels, r2.labels)


def test_fit_sub_pca_detects_spikes():
    """Point spikes need small k + sufficient amplitude.

    With A=10, k=5, w=50: spike residual 7-9.5 (position-dependent),
    natural max-of-window residual ~0.9. Clear separation.
    Detection fails at k=15+ where spike leaks into subspace (see
    test_k_sensitivity).
    """
    # n=5000, 10 spikes -> 500 affected timestamps (10% of series),
    # contamination=0.01 -> 50 flagged = top-50 near-peak -> recall~1.0.
    # At n=1000 with 20 spikes, all series is "affected" -> recall low.
    x, pos = _sine_spikes(n=5000, n_spikes=10, seed=0)
    r = fit_sub_pca(x, window_size=50, n_components=5,
                    contamination=0.01, aggregate="mean")
    recall = float(np.mean(r.labels[pos] == 1))
    assert recall >= 0.6, f"recall={recall:.2f} on {len(pos)} spikes"


def test_fit_sub_pca_k_sensitivity():
    """Detection degrades with larger k (spike leaks into subspace).

    At k=5, spike residual is high (~19). At k=25 (of 50), subspace
    covers 50% of spike energy -> residual halves -> below threshold.
    Mean aggregation reduces neighbor spillover so k-sensitivity is
    visible.
    """
    x, pos = _sine_spikes(n=5000, n_spikes=10, seed=0)
    rec = {}
    for k in (3, 5, 25):
        r = fit_sub_pca(x, window_size=50, n_components=k,
                        contamination=0.01, aggregate="mean")
        rec[k] = float(np.mean(r.labels[pos] == 1))
    assert rec[3] >= rec[25], (
        f"small-k should beat large-k: {rec}")
    assert rec[5] >= 0.6, (
        f"k=5 recall={rec[5]:.2f} < 0.6 (recommended configuration)")


def test_fit_sub_pca_constant_series():
    x = np.ones(500) * 3.0
    r = fit_sub_pca(x, window_size=50, n_components=10)
    # Constant -> all windows identical -> residual 0 -> scores 0
    assert r.decision_scores.max() < 1e-10
    assert r.labels.sum() == 0


def test_fit_sub_pca_short_series_raises():
    with pytest.raises(ValueError, match="< window_size"):
        fit_sub_pca(np.arange(10.0), window_size=50)


def test_fit_sub_pca_none_uses_half_window():
    x = np.random.default_rng(0).standard_normal(200)
    r = fit_sub_pca(x, window_size=40, n_components=None)
    assert r.n_components == 20  # 40 // 2


def test_fit_sub_pca_float_variance_fraction():
    x, _ = _sine_spikes(n=500, seed=1)
    r = fit_sub_pca(x, window_size=50, n_components=0.9)
    # k determined by variance; should be reasonable (>0 and <=50)
    assert 1 <= r.n_components <= 50


def test_fit_sub_pca_int_clipped():
    x = np.random.default_rng(0).standard_normal(200)
    # ask for k=100 but window_size=50 -> clipped to 50
    r = fit_sub_pca(x, window_size=50, n_components=100)
    assert r.n_components == 50


def test_fit_sub_pca_aggregate_max_vs_mean():
    """Max spreads spike to ~w timestamps (neighbor spillover);
    mean gives spike position max influence, decays with distance."""
    x, pos = _sine_spikes(n=500, n_spikes=10, seed=7)
    r_max = fit_sub_pca(x, window_size=50, n_components=5,
                        contamination=0.05, aggregate="max")
    r_mean = fit_sub_pca(x, window_size=50, n_components=5,
                         contamination=0.05, aggregate="mean")
    # max aggregation has higher or equal scores everywhere
    assert np.all(r_max.decision_scores >= r_mean.decision_scores - 1e-12)
    # Mean aggregation: spike position score significantly exceeds
    # neighbor scores (decay); max spreads score across ~w neighbors
    rec_mean = float(np.mean(r_mean.labels[pos] == 1))
    rec_max = float(np.mean(r_max.labels[pos] == 1))
    assert rec_mean >= rec_max, (
        f"mean should beat max for point anomalies: "
        f"mean={rec_mean:.2f}, max={rec_max:.2f}")


def test_fit_sub_pca_bad_aggregate_raises():
    x = np.random.default_rng(0).standard_normal(200)
    with pytest.raises(ValueError, match="aggregate"):
        fit_sub_pca(x, window_size=50, aggregate="median")


def test_fit_sub_pca_memory_guard_logic():
    from cadw.core.memory_guard import MemoryLimitExceeded

    class _FakeMB:
        max_memory_mb = 0.001

    x = np.random.default_rng(0).standard_normal(5000)
    with pytest.raises(MemoryLimitExceeded):
        fit_sub_pca(x, window_size=50, guard=_FakeMB())


def test_fit_sub_pca_memory_guard_real_smoke():
    from cadw.core.memory_guard import MemoryGuard
    guard = MemoryGuard(max_memory_mb=1000)
    x = np.random.default_rng(1).standard_normal(500)
    r = fit_sub_pca(x, window_size=50, guard=guard)
    assert r.decision_scores.shape == (500,)


def test_sub_pca_no_bench_deps_in_source():
    src = (ROOT / "cadw/ml/sub_pca.py").read_text()
    for f in ("import sklearn", "import pyod", "import torch",
              "import scipy", "from sklearn", "from pyod",
              "from torch", "from scipy"):
        assert f not in src, f"forbidden import: {f}"


def test_sub_pca_reference_sklearn_reconstruction():
    """Window reconstruction errors must match sklearn.decomposition.PCA.

    Projection onto a k-dim subspace is unique regardless of SVD vs
    eigendecomposition; the ||residual|| per window should match.
    """
    from sklearn.decomposition import PCA
    x, _ = _sine_spikes(n=400, seed=3)
    w = 40
    k = 15
    # our
    r = fit_sub_pca(x, window_size=w, n_components=k)
    # reconstruct from our model
    windows = np.lib.stride_tricks.sliding_window_view(x, w)
    centered = windows - r.mean
    proj = centered @ r.components.T
    recon = proj @ r.components
    residual = centered - recon
    our_window_scores = np.linalg.norm(residual, axis=1)
    # sklearn
    sk = PCA(n_components=k)
    sk.fit(windows)
    sk_recon = sk.inverse_transform(sk.transform(windows))
    sk_residual = windows - sk_recon
    sk_window_scores = np.linalg.norm(sk_residual, axis=1)
    # tolerance: eigh vs SVD of same covariance; sign ambiguity doesn't
    # affect projection or residual norm
    np.testing.assert_allclose(
        our_window_scores, sk_window_scores, rtol=1e-5, atol=1e-8)

