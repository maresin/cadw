"""Tests for EIF (Extended Isolation Forest, Hariri et al. 2019)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cadw.ml.eif import (EIFResult, _c, _EIFTree, _make_windows,
                         _random_normal, fit_eif)

ROOT = Path(__file__).parent.parent


def _sine_segments(n=5000, period=100, n_segments=5, seg_len=20,
                   amplitude=20.0, seed=0):
    """Sine + square-wave segments.

    EIF is a tabular outlier detector. With windowing, an isolated
    *point* spike contributes ~1/w = 2% of the window vector and
    is invisible. Segment anomalies (>= w/2 timesteps anomalous)
    produce a substantial fraction of outlier content per window and
    are detectable (this matches EIF's inductive bias: it detects
    *tabular* outliers — a whole window must look anomalous, not one
    coordinate of it).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x = np.sin(2 * np.pi * t / period) + 0.1 * rng.standard_normal(n)
    starts = rng.choice(n - seg_len, size=n_segments, replace=False)
    pos = []
    for s in starts:
        x[s:s + seg_len] = amplitude  # constant high-value segment
        pos.extend(range(s, s + seg_len))
    return x, np.array(sorted(pos))


def _multi_segments(n=3000, c=3, n_segments=5, seg_len=20, seed=0):
    """Multi-channel sine + synchronous segment anomalies."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    X = np.empty((n, c))
    for ch in range(c):
        X[:, ch] = np.sin(2 * np.pi * t / (100.0 + 20.0 * ch) + 0.5 * ch) \
            + 0.1 * rng.standard_normal(n)
    starts = rng.choice(n - seg_len, size=n_segments, replace=False)
    pos = []
    for s in starts:
        X[s:s + seg_len, :] = 5.0  # all channels anomalous for whole segment
        pos.extend(range(s, s + seg_len))
    return X, np.array(sorted(pos))


# ── c(psi) and helpers ───────────────────────────────────────────

def test_c_psi_small_values():
    assert _c(1) == 0.0
    assert _c(2) == 1.0
    assert 1.5 < _c(10) < 4.0
    assert 5.0 < _c(256) < 11.0  # c(256) ~= 10.24


def test_random_normal_extension_level_zero():
    rng = np.random.default_rng(0)
    v = _random_normal(5, extension_level=0, rng=rng)
    assert np.sum(np.abs(v) > 1e-12) == 1
    assert abs(abs(v[np.argmax(np.abs(v))]) - 1.0) < 1e-12


def test_random_normal_extension_full():
    rng = np.random.default_rng(0)
    v = _random_normal(5, extension_level=4, rng=rng)
    assert np.all(np.abs(v) > 1e-12)
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-9


def test_make_windows_univariate():
    x = np.arange(10.0)
    w = _make_windows(x, window_size=3, stride=1)
    assert w.shape == (8, 3)
    np.testing.assert_array_equal(w[0], [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(w[-1], [7.0, 8.0, 9.0])


def test_make_windows_multivariate():
    X = np.arange(20.0).reshape(10, 2)
    w = _make_windows(X, window_size=3, stride=2)
    # stride=2, w=3 -> n_w = (10-3)//2+1 = 4
    assert w.shape == (4, 6)
    np.testing.assert_array_equal(w[0], X[0:3].ravel())
    np.testing.assert_array_equal(w[1], X[2:5].ravel())


# ── fit_eif contract ─────────────────────────────────────────────

def test_fit_eif_basic_shape():
    x, _ = _sine_segments(n=500, n_segments=2, seg_len=15, seed=0)
    r = fit_eif(x, n_estimators=20, window_size=30, max_samples=64,
                random_state=0)
    assert isinstance(r, EIFResult)
    assert r.decision_scores.shape == (500,)
    assert r.labels.shape == (500,)
    assert 0.0 <= r.decision_scores.min() <= r.decision_scores.max() <= 1.0
    assert r.extension_level == 30 - 1


def test_fit_eif_deterministic():
    x, _ = _sine_segments(n=500, n_segments=2, seg_len=15, seed=7)
    r1 = fit_eif(x, n_estimators=20, window_size=30, max_samples=64,
                 random_state=42)
    r2 = fit_eif(x, n_estimators=20, window_size=30, max_samples=64,
                 random_state=42)
    np.testing.assert_array_equal(r1.decision_scores, r2.decision_scores)


def test_fit_eif_detects_segments():
    """Segment anomalies (20 timesteps, window=30 -> ~67% outlier content)."""
    x, pos = _sine_segments(n=5000, n_segments=5, seg_len=20, seed=0)
    r = fit_eif(x, n_estimators=100, window_size=30, contamination=0.02,
                random_state=0)
    recall = float(np.mean(r.labels[pos] == 1))
    assert recall >= 0.5, f"recall={recall:.2f} on {len(pos)} segment pts"


def test_fit_eif_multivariate_segments():
    X, pos = _multi_segments(n=2000, c=3, n_segments=4, seg_len=20, seed=0)
    r = fit_eif(X, n_estimators=50, window_size=20, contamination=0.04,
                random_state=0)
    assert r.decision_scores.shape == (2000,)
    assert r.n_channels == 3
    recall = float(np.mean(r.labels[pos] == 1))
    assert recall >= 0.4, f"recall={recall:.2f} on {len(pos)} seg pts"


def test_fit_eif_constant_series():
    x = np.ones(200) * 3.0
    r = fit_eif(x, n_estimators=10, window_size=30, max_samples=32,
                random_state=0)
    # Constant -> all windows identical -> all scores identical
    assert r.decision_scores.std() < 1e-6
    assert r.labels.sum() == 0


def test_fit_eif_short_series_raises():
    with pytest.raises(ValueError, match="< window_size"):
        fit_eif(np.arange(10.0), window_size=50)


def test_fit_eif_bad_aggregate_raises():
    x = np.random.default_rng(0).standard_normal(200)
    with pytest.raises(ValueError, match="aggregate"):
        fit_eif(x, window_size=30, aggregate="median")


def test_fit_eif_bad_extension_level_raises():
    x = np.random.default_rng(0).standard_normal(200)
    with pytest.raises(ValueError, match="extension_level"):
        fit_eif(x, window_size=30, extension_level=100)


def test_fit_eif_tabular_mode():
    """Paper use case: tabular outliers with window_size=1 (no windowing).

    Contaminated Gaussian blobs: 95% from N(0,1)^d, 5% from N(3,1)^d.
    EIF should separate them well (single-window, tabular).
    """
    from cadw.ml.eif import _EIFForest

    rng = np.random.default_rng(0)
    n = 2000
    n_out = 100
    d = 4
    X = rng.standard_normal((n, d))
    outlier_idx = rng.choice(n, size=n_out, replace=False)
    X[outlier_idx] += 3.0

    forest = _EIFForest(n_estimators=100, max_samples=256,
                        extension_level=d - 1, random_state=0).fit(X)
    scores = forest.score(X)
    # Mean score of outliers should be higher (lower margin — EIF scores
    # are compressed toward 0.5 at this n and dim, so shift is modest)
    mean_out = float(scores[outlier_idx].mean())
    mean_in = float(np.delete(scores, outlier_idx).mean())
    assert mean_out > mean_in + 0.02, (
        f"outlier mean score {mean_out:.3f} vs inlier {mean_in:.3f}")
    # Threshold at 95p (matching 5% contamination) -> top 100 flagged
    thr = np.percentile(scores, 95)
    recall = float((scores[outlier_idx] > thr).mean())
    assert recall >= 0.5, f"tabular recall={recall:.2f} at 95p threshold"


def test_fit_eif_extension_zero_equivalent_to_iforest_direction():
    """extension_level=0 -> single random axis per split (IForest).
    Should still produce sensible scores in [0, 1]."""
    x, _ = _sine_segments(n=1000, n_segments=3, seg_len=15, seed=1)
    r = fit_eif(x, n_estimators=50, window_size=30, extension_level=0,
                max_samples=64, random_state=0)
    assert r.extension_level == 0
    assert 0.0 <= r.decision_scores.min() <= r.decision_scores.max() <= 1.0


def test_fit_eif_stride_changes_window_count():
    x = np.random.default_rng(0).standard_normal(500)
    r1 = fit_eif(x, n_estimators=10, window_size=30, stride=1,
                 max_samples=32, random_state=0)
    r2 = fit_eif(x, n_estimators=10, window_size=30, stride=5,
                 max_samples=32, random_state=0)
    assert r1.window_scores.shape[0] > r2.window_scores.shape[0]


def test_fit_eif_memory_guard_logic():
    from cadw.core.memory_guard import MemoryLimitExceeded

    class _FakeMB:
        max_memory_mb = 0.001

    x = np.random.default_rng(0).standard_normal(5000)
    with pytest.raises(MemoryLimitExceeded):
        fit_eif(x, n_estimators=5, window_size=100, max_samples=32,
                guard=_FakeMB(), random_state=0)


def test_fit_eif_memory_guard_real_smoke():
    from cadw.core.memory_guard import MemoryGuard
    guard = MemoryGuard(max_memory_mb=1000)
    x = np.random.default_rng(1).standard_normal(300)
    r = fit_eif(x, n_estimators=5, window_size=30, max_samples=32,
                guard=guard, random_state=0)
    assert r.decision_scores.shape == (300,)


def test_eif_no_bench_deps_in_source():
    src = (ROOT / "cadw/ml/eif.py").read_text()
    for f in ("import sklearn", "import pyod", "import torch",
              "import scipy", "import aeon",
              "from sklearn", "from pyod", "from torch",
              "from scipy", "from aeon"):
        assert f not in src, f"forbidden import: {f}"


def test_eif_reference_sahandha_smoke():
    """If sahandha/eif is installed, compare decision scores."""
    eif_mod = pytest.importorskip("eif")
    from cadw.ml.eif import _EIFForest, _make_windows
    X = np.random.default_rng(3).standard_normal((500, 5))
    # reference: 5-dim EIF, extension 4, 50 trees, subsample 64
    ref = eif_mod.iForest(X, ntrees=50, sample_size=64, ExtensionLevel=4)
    ref_scores = ref.compute_paths(X)
    # ours: direct forest (no windowing)
    forest = _EIFForest(n_estimators=50, max_samples=64,
                        extension_level=4, random_state=None).fit(X)
    our_scores = forest.score(X)
    # Both in [0, 1]; correlation should be strong (different RNG seeds
    # and no matched init -> exact match impossible).
    corr = float(np.corrcoef(ref_scores, our_scores)[0, 1])
    assert corr > 0.5, f"corr={corr:.3f} too low (ref vs ours)"

def test_fit_eif_guard_raises_before_allocation(monkeypatch):
    """v0.9.87-fix1b: guard fires BEFORE _make_windows allocation.

    Regression for audit HIGH-1: eif.py allocated (n_w, d) window
    array at line 301 BEFORE the guard at line 318. Move guard ahead.
    """
    from cadw.ml import eif as eif_mod
    from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded

    called = {"n": 0}
    orig = eif_mod._make_windows

    def spy(*a, **kw):
        called["n"] += 1
        return orig(*a, **kw)

    monkeypatch.setattr(eif_mod, "_make_windows", spy)

    guard = MemoryGuard(max_memory_mb=10)
    x = np.random.default_rng(0).standard_normal(50_000)
    with pytest.raises(MemoryLimitExceeded):
        eif_mod.fit_eif(x, window_size=64, guard=guard)
    assert called["n"] == 0, (
        f"guard should fire BEFORE _make_windows; "
        f"_make_windows was called {called['n']} time(s)")

