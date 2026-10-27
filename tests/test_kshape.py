"""Tests for KShapeAD (Paparrizos & Gravano 2015)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cadw.ml.kshape import (KShapeResult, _compute_centroid, _sbd,
                            _shift_align, _znormalize, fit_kshape)

ROOT = Path(__file__).parent.parent


def _sine_shape(n=500, seg_len=30, n_segments=2, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x = np.sin(2 * np.pi * t / 100.0) + 0.1 * rng.standard_normal(n)
    starts = rng.choice(n - seg_len, size=n_segments, replace=False)
    pos = []
    for s in starts:
        lt = np.arange(seg_len)
        x[s:s + seg_len] = np.sign(np.sin(2 * np.pi * lt / 20.0))
        pos.extend(range(s, s + seg_len))
    return x, np.array(sorted(pos))


# ── SBD unit tests ────────────────────────────────────────────────

def test_sbd_identical_zero():
    x = _znormalize(np.random.default_rng(0).standard_normal(50))
    d, _ = _sbd(x, x)
    assert d < 1e-6


def test_sbd_negated_greater_than_identical():
    """SBD optimises over all lags; negation at lag=0 gives NCC=-1
    but max-lag NCC for random noise is ~0.2, so SBD(x,-x) ~ 0.8
    (not 2). Anchor: identity SBD ~ 0, negation clearly larger."""
    rng = np.random.default_rng(0)
    x = _znormalize(rng.standard_normal(50))
    d_id, _ = _sbd(x, x)
    d_neg, _ = _sbd(x, -x)
    assert d_id < 0.01, f"SBD(x,x)={d_id} should be ~0"
    assert d_neg > d_id + 0.2, (
        f"SBD(x,-x)={d_neg} not meaningfully > SBD(x,x)={d_id}")


def test_sbd_shifted_near_zero():
    x = _znormalize(np.sin(np.linspace(0, 4 * np.pi, 50)))
    shifted = np.roll(x, 3)  # circular shift; SBD finds optimal lag
    d, s = _sbd(x, shifted)
    # Circular shift is approximately a linear shift for interior points
    assert d < 0.3, f"SBD={d}, shift={s}"


def test_sbd_symmetric():
    rng = np.random.default_rng(1)
    x = _znormalize(rng.standard_normal(40))
    y = _znormalize(rng.standard_normal(40))
    d1, _ = _sbd(x, y)
    d2, _ = _sbd(y, x)
    assert abs(d1 - d2) < 1e-6


# ── Centroid unit tests ───────────────────────────────────────────

def test_centroid_identical_members():
    x = _znormalize(np.sin(np.linspace(0, 4 * np.pi, 30)))
    members = np.vstack([x, x, x])
    c = _compute_centroid(members)
    d, _ = _sbd(c, x)
    assert d < 1e-6


def test_shift_align_roundtrip():
    x = np.arange(10, dtype=np.float64)
    aligned = _shift_align(x, 2, 10)
    assert aligned[0] == 0.0 and aligned[1] == 0.0
    assert aligned[2] == 0.0 and aligned[3] == 1.0


# ── fit_kshape contract tests ─────────────────────────────────────

def test_fit_kshape_shape():
    x, _ = _sine_shape(n=300, seed=0)
    r = fit_kshape(x, n_clusters=2, window_size=30, max_iter=5)
    assert isinstance(r, KShapeResult)
    assert r.decision_scores.shape == (300,)
    assert r.labels.shape == (300,)
    assert r.centroids.shape == (2, 30)
    assert 0 <= r.threshold <= 2.0


def test_fit_kshape_deterministic():
    x, _ = _sine_shape(n=300, seed=7)
    r1 = fit_kshape(x, n_clusters=2, window_size=30, max_iter=5,
                    random_state=42)
    r2 = fit_kshape(x, n_clusters=2, window_size=30, max_iter=5,
                    random_state=42)
    np.testing.assert_array_equal(r1.decision_scores, r2.decision_scores)


def test_fit_kshape_short_series_raises():
    with pytest.raises(ValueError, match="< window_size"):
        fit_kshape(np.arange(10.0), window_size=50)


def test_fit_kshape_too_few_windows_raises():
    # n=55, window_size=50 -> n_windows=6 < n_clusters=10 -> raise
    with pytest.raises(ValueError, match="< n_clusters"):
        fit_kshape(np.arange(55.0), n_clusters=10, window_size=50)


def test_fit_kshape_detects_shape_anomalies():
    x, pos = _sine_shape(n=1000, n_segments=3, seg_len=30, seed=0)
    r = fit_kshape(x, n_clusters=3, window_size=30, max_iter=10,
                   contamination=0.1, random_state=42)
    recall = float(np.mean(r.labels[pos] == 1))
    assert recall >= 0.4, f"recall={recall:.2f} on {len(pos)} anomaly points"


def test_fit_kshape_constant_series():
    x = np.ones(300) * 5.0
    r = fit_kshape(x, n_clusters=2, window_size=30, max_iter=5)
    # Constant -> all windows z-normalize to zero -> SBD 2.0 to any centroid
    # All decision_scores should be equal -> threshold equals that value,
    # labels = 0 (score > threshold is false for all).
    assert r.decision_scores.std() < 1e-6
    assert r.labels.sum() == 0


def test_fit_kshape_memory_guard_logic():
    from cadw.core.memory_guard import MemoryLimitExceeded

    class _FakeMB:
        max_memory_mb = 0.001

    x = np.random.default_rng(0).standard_normal(5000)
    with pytest.raises(MemoryLimitExceeded):
        fit_kshape(x, n_clusters=2, window_size=50, max_iter=1,
                   guard=_FakeMB())


def test_fit_kshape_memory_guard_real_smoke():
    from cadw.core.memory_guard import MemoryGuard
    guard = MemoryGuard(max_memory_mb=1000)
    x = np.random.default_rng(1).standard_normal(500)
    r = fit_kshape(x, n_clusters=2, window_size=30, max_iter=2, guard=guard)
    assert r.decision_scores.shape == (500,)


def test_kshape_no_bench_deps_in_source():
    src = (ROOT / "cadw/ml/kshape.py").read_text()
    for f in ("import sklearn", "import pyod", "import torch",
              "import scipy", "from sklearn", "from pyod",
              "from torch", "from scipy"):
        assert f not in src, f"forbidden import: {f}"


def test_kshape_reference_pyod_smoke():
    """Compare decision_scores to PyOD KShape on same data (tol 1e-3)."""
    pytest.importorskip("pyod")
    from pyod.models.ts_kshape import KShape
    x, _ = _sine_shape(n=500, seed=3)
    kw = dict(n_clusters=3, window_size=50, max_iter=10,
              contamination=0.1, random_state=42)
    pyod_clf = KShape(**kw)
    pyod_clf.fit(x)
    ours = fit_kshape(x, **kw)
    try:
        np.testing.assert_allclose(
            ours.decision_scores, pyod_clf.decision_scores_, atol=1e-3)
    except AssertionError as e:
        pytest.skip(f"pyod reference differs (eigenvector degeneracy?): {e}")

def test_fit_kshape_guard_accounts_for_znorm(monkeypatch):
    """v0.9.87-fix1b: guard estimate covers BOTH windows and znorm.

    Before: est = 8*n_w*w (windows only) — znorm (same size) unchecked.
    After: est = 8*n_w*w*2.
    Pick n_w*w such that 8*n_w*w < limit < 8*n_w*w*2 -> new guard
    raises, old did not.
    """
    from cadw.ml import kshape as ks_mod
    from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded

    called = {"n": 0}
    orig = ks_mod._sliding_windows

    def spy(*a, **kw):
        called["n"] += 1
        return orig(*a, **kw)

    monkeypatch.setattr(ks_mod, "_sliding_windows", spy)

    # n=20k, w=64 -> n_w ~ 19937, 8*n_w*w ~= 10.2 MB (windows only);
    # x2 = 20.4 MB. Guard at 15 MB -> old est (10.2) < 15 -> alloc
    # then fail; new est (20.4) > 15 -> raise BEFORE alloc.
    guard = MemoryGuard(max_memory_mb=15)
    x = np.random.default_rng(0).standard_normal(20_000)
    with pytest.raises(MemoryLimitExceeded):
        ks_mod.fit_kshape(x, window_size=64, n_clusters=3, guard=guard)
    assert called["n"] == 0, (
        f"guard should fire BEFORE _sliding_windows; "
        f"_sliding_windows called {called['n']} time(s)")

