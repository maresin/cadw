"""Tests for ZMS rolling z-score anomaly detection."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cadw.ml.zms import ZMSResult, fit_zms

ROOT = Path(__file__).parent.parent


def _sine_spikes(n=1000, period=100, n_spikes=10, seed=0, amplitude=20.0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x = np.sin(2 * np.pi * t / period) + 0.05 * rng.standard_normal(n)
    pos = rng.choice(n, size=n_spikes, replace=False)
    x[pos] += amplitude
    return x, pos


def test_fit_zms_basic_shape():
    x, _ = _sine_spikes(n=500, seed=0)
    r = fit_zms(x, window_size=50)
    assert isinstance(r, ZMSResult)
    assert r.decision_scores.shape == (500,)
    assert r.labels.shape == (500,)
    assert r.window_size == 50


def test_fit_zms_detects_spikes():
    x, pos = _sine_spikes(n=2000, n_spikes=10, seed=0)
    r = fit_zms(x, window_size=100, threshold=3.0)
    recall = float(np.mean(r.labels[pos] == 1))
    assert recall >= 0.6, f"recall={recall:.2f} on {len(pos)} spikes"


def test_fit_zms_constant_no_anomalies():
    x = np.ones(200) * 5.0
    r = fit_zms(x, window_size=50)
    # Constant -> score ~0 (sigma floored to eps)
    assert r.labels.sum() == 0


def test_fit_zms_short_series_raises():
    with pytest.raises(ValueError, match="< window_size"):
        fit_zms(np.arange(10.0), window_size=50)


def test_fit_zms_threshold_controls_rate():
    x, _ = _sine_spikes(n=2000, n_spikes=10, seed=1)
    r_low = fit_zms(x, window_size=100, threshold=2.0)
    r_high = fit_zms(x, window_size=100, threshold=5.0)
    assert r_low.labels.sum() >= r_high.labels.sum()


def test_fit_zms_contamination_threshold():
    x, _ = _sine_spikes(n=2000, n_spikes=10, seed=2)
    r = fit_zms(x, window_size=100, contamination=0.02)
    assert 20 <= r.labels.sum() <= 80


def test_fit_zms_deterministic():
    x, _ = _sine_spikes(n=1000, seed=3)
    r1 = fit_zms(x, window_size=50)
    r2 = fit_zms(x, window_size=50)
    np.testing.assert_array_equal(r1.decision_scores, r2.decision_scores)


def test_fit_zms_memory_guard_logic():
    from cadw.core.memory_guard import MemoryLimitExceeded

    class _FakeMB:
        max_memory_mb = 0.001

    x = np.random.default_rng(0).standard_normal(5000)
    with pytest.raises(MemoryLimitExceeded):
        fit_zms(x, window_size=100, guard=_FakeMB())


def test_zms_no_bench_deps_in_source():
    src = (ROOT / "cadw/ml/zms.py").read_text()
    for f in ("import sklearn", "import scipy", "import torch",
              "import merlion", "from sklearn", "from scipy",
              "from torch", "from merlion"):
        assert f not in src, f"forbidden import: {f}"

