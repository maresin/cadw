"""Tests for DWT-MLEAD (Thill, Konen, Baeck 2017)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cadw.ml.dwt_mlead import DwtMleadResult, fit_dwt_mlead

ROOT = Path(__file__).parent.parent


# ─── Synthetic DGPs ──────────────────────────────────────────────

def _sine_with_spikes(n=500, period=100, n_spikes=10, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x = np.sin(2 * np.pi * t / period) + 0.1 * rng.standard_normal(n)
    spike_pos = rng.choice(n, size=n_spikes, replace=False)
    x[spike_pos] += 5.0
    return x, spike_pos


# ─── Tests ───────────────────────────────────────────────────────

def test_dwt_mlead_basic_shape():
    x = np.sin(np.linspace(0, 10, 100)) + np.random.default_rng(0).standard_normal(100) * 0.1
    result = fit_dwt_mlead(x)
    assert isinstance(result, DwtMleadResult)
    assert result.anomaly_scores.shape == (100,)
    assert result.is_anomaly.shape == (100,)
    assert result.is_anomaly.dtype == np.bool_
    assert result.anomaly_scores.dtype == np.float64


def test_dwt_mlead_detects_spikes():
    x, spike_pos = _sine_with_spikes(n=1000, n_spikes=20, seed=0)
    result = fit_dwt_mlead(x, start_level=3, quantile_epsilon=0.01,
                           min_anomaly_consistency=2)
    detected = int(result.is_anomaly[spike_pos].sum())
    # At least 30% of injected spikes detected
    assert detected >= 6, (
        f"Only {detected}/20 spikes detected at {sorted(spike_pos[:5])}..."
    )


def test_dwt_mlead_constant_no_anomalies():
    x = np.ones(200) * 3.0
    result = fit_dwt_mlead(x)
    assert not result.is_anomaly.any(), (
        "Constant series should have no anomalies"
    )
    assert result.anomaly_scores.max() == 0


def test_dwt_mlead_short_series_raises():
    with pytest.raises(ValueError, match="too short"):
        fit_dwt_mlead(np.array([1.0, 2.0, 3.0]))


def test_dwt_mlead_too_few_levels_returns_zeros():
    # n=8 gives max_level ~= 3; start_level=3 → start_level >= max_level
    x = np.random.default_rng(0).standard_normal(8)
    result = fit_dwt_mlead(x, start_level=3)
    assert not result.is_anomaly.any()
    assert (result.anomaly_scores == 0).all()


def test_dwt_mlead_deterministic():
    x, _ = _sine_with_spikes(n=500, seed=42)
    r1 = fit_dwt_mlead(x)
    r2 = fit_dwt_mlead(x)
    np.testing.assert_array_equal(r1.anomaly_scores, r2.anomaly_scores)
    np.testing.assert_array_equal(r1.is_anomaly, r2.is_anomaly)


def test_dwt_mlead_quantile_controls_rate():
    x, _ = _sine_with_spikes(n=1000, n_spikes=20, seed=7)
    r_low = fit_dwt_mlead(x, quantile_epsilon=0.01, min_anomaly_consistency=1)
    r_high = fit_dwt_mlead(x, quantile_epsilon=0.05, min_anomaly_consistency=1)
    # Higher quantile_epsilon → more flagged windows per level
    assert r_high.is_anomaly.sum() >= r_low.is_anomaly.sum()


def test_dwt_mlead_min_consistency_filters():
    x, _ = _sine_with_spikes(n=500, seed=11)
    r1 = fit_dwt_mlead(x, min_anomaly_consistency=1)
    r3 = fit_dwt_mlead(x, min_anomaly_consistency=3)
    assert r3.is_anomaly.sum() <= r1.is_anomaly.sum()


def test_dwt_mlead_memory_guard_logic():
    """Test OUR guard-check logic with a fake guard exposing max_memory_mb.

    Real MemoryGuard's internal API is not part of our contract; if
    opaque, the production check is best-effort and skips silently.
    This test verifies the check itself (est_bytes > limit -> raise).
    """
    from cadw.core.memory_guard import MemoryLimitExceeded
    from cadw.ml.dwt_mlead import _probe_memory_limit_mb

    class _FakeGuardMB:
        max_memory_mb = 0.001

    class _FakeGuardBytes:
        _max_memory_bytes = int(0.001 * 1024 * 1024)  # 1048 bytes

    assert _probe_memory_limit_mb(_FakeGuardMB()) == 0.001
    assert abs(_probe_memory_limit_mb(_FakeGuardBytes()) - 0.001) < 1e-6

    x = np.random.default_rng(0).standard_normal(10_000)
    # n=10000 -> padded 16384 -> max_level=14 -> est = 8*10000*14 ≈ 1.07 MB
    # limits: 0.001 MB both -> must raise
    with pytest.raises(MemoryLimitExceeded):
        fit_dwt_mlead(x, guard=_FakeGuardMB())
    with pytest.raises(MemoryLimitExceeded):
        fit_dwt_mlead(x, guard=_FakeGuardBytes())


def test_dwt_mlead_memory_guard_real_api_smoke():
    """Real MemoryGuard: must not crash regardless of internal API.

    If MemoryGuard exposes a numeric limit, we either raise (tight) or
    succeed (loose). If opaque, we succeed. Never crash.
    """
    from cadw.core.memory_guard import MemoryGuard
    guard = MemoryGuard(max_memory_mb=1000)  # generous -> no raise
    x = np.random.default_rng(1).standard_normal(2000)
    result = fit_dwt_mlead(x, guard=guard)
    assert result.is_anomaly.shape == (2000,)


def test_dwt_mlead_no_bench_deps_in_source():
    """cadw/ml/dwt_mlead.py must not import sklearn/aeon/torch/scipy."""
    src = (ROOT / "cadw/ml/dwt_mlead.py").read_text()
    forbidden = ["import sklearn", "import aeon", "import torch",
                 "import scipy",
                 "from sklearn", "from aeon", "from torch", "from scipy"]
    for f in forbidden:
        assert f not in src, f"forbidden import in cadw/ml/dwt_mlead.py: {f}"


def test_dwt_mlead_reference_aeon_smoke():
    """If aeon is available, DWT_MLEAD runs on same DGP without error."""
    try:
        from aeon.anomaly_detection import DWT_MLEAD
    except ImportError:
        pytest.skip("aeon not installed")
    x, _ = _sine_with_spikes(n=500, seed=3)
    try:
        detector = DWT_MLEAD(start_level=3, quantile_epsilon=0.01, n_jobs=1)
    except TypeError:
        # aeon API may differ slightly across versions
        detector = DWT_MLEAD(start_level=3, quantile_epsilon=0.01)
    aeon_scores = detector.fit_predict(x)
    assert aeon_scores.shape == (500,)
    # Our implementation must also run on same input
    result = fit_dwt_mlead(x)
    assert result.is_anomaly.shape == (500,)

def test_fit_dwt_mlead_guard_fires_before_streaming(monkeypatch):
    """P4b: guard fires BEFORE per-level streaming begins.

    Was fix1b's test (_sliding_windows); P4b removes _sliding_windows
    and uses _stream_mahalanobis. Spy the latter; tight guard; expect
    raise AND spy untouched.
    """
    from cadw.ml import dwt_mlead as dm
    from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded

    called = {"n": 0}
    orig = dm._stream_mahalanobis

    def spy(*a, **kw):
        called["n"] += 1
        return orig(*a, **kw)

    monkeypatch.setattr(dm, "_stream_mahalanobis", spy)

    guard = MemoryGuard(max_memory_mb=1)  # 1 MB, forces raise
    x = np.random.default_rng(0).standard_normal(100_000)
    with pytest.raises(MemoryLimitExceeded):
        dm.fit_dwt_mlead(x, guard=guard)
    assert called["n"] == 0, (
        f"guard should fire BEFORE streaming; "
        f"_stream_mahalanobis was called {called['n']} time(s)")

def test_streaming_mahalanobis_matches_batch():
    """P4b: _stream_mahalanobis bit-identical to batch reference."""
    from cadw.ml.dwt_mlead import (
        _mahalanobis_distances, _stream_mahalanobis,
    )
    # Build small 2D windows array as reference source
    rng = np.random.default_rng(42)
    detail = rng.standard_normal(2000)
    w = 16
    # Batch reference (windows materialized)
    windows = np.lib.stride_tricks.sliding_window_view(detail, w).copy()
    ref = _mahalanobis_distances(windows)
    # Streaming
    got = _stream_mahalanobis(detail, w, chunk_nw=256)
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-10)


def test_streaming_mahalanobis_no_dense_windows(monkeypatch):
    """P4b: verify per-level chunks bounded, not full (n_w, w)."""
    from cadw.ml import dwt_mlead as dm

    max_chunk = [0]
    orig = dm._iter_detail_window_chunks

    def spy(*args, **kwargs):
        for chunk, start in orig(*args, **kwargs):
            max_chunk[0] = max(max_chunk[0], chunk.shape[0])
            yield chunk, start

    monkeypatch.setattr(dm, "_iter_detail_window_chunks", spy)

    x = np.random.default_rng(0).standard_normal(50_000)
    dm.fit_dwt_mlead(x)
    assert max_chunk[0] <= 10_000, (
        f"chunk size {max_chunk[0]} > 10_000 (streaming broken)")

