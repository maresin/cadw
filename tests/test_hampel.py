"""Tests for Hampel filter anomaly detection."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cadw.ml.hampel import HampelResult, fit_hampel

ROOT = Path(__file__).parent.parent


def _sine_spikes(n=1000, period=100, n_spikes=10, seed=0, amplitude=20.0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x = np.sin(2 * np.pi * t / period) + 0.05 * rng.standard_normal(n)
    pos = rng.choice(n, size=n_spikes, replace=False)
    x[pos] += amplitude
    return x, pos


def test_fit_hampel_basic_shape():
    x, _ = _sine_spikes(n=500, n_spikes=5, seed=0)
    r = fit_hampel(x, window_size=7)
    assert isinstance(r, HampelResult)
    assert r.decision_scores.shape == (500,)
    assert r.labels.shape == (500,)
    assert r.window_size == 7


def test_fit_hampel_detects_spikes():
    x, pos = _sine_spikes(n=1000, n_spikes=10, seed=0)
    r = fit_hampel(x, window_size=7, n_sigma=3.0)
    recall = float(np.mean(r.labels[pos] == 1))
    assert recall >= 0.8, f"recall={recall:.2f} on {len(pos)} spikes"


def test_fit_hampel_constant_no_anomalies():
    x = np.ones(200) * 5.0
    r = fit_hampel(x)
    assert r.labels.sum() == 0
    assert r.decision_scores.max() < 1.0


def test_fit_hampel_even_window_auto_increment():
    x, _ = _sine_spikes(n=200, n_spikes=2, seed=0)
    r = fit_hampel(x, window_size=6)
    assert r.window_size == 7


def test_fit_hampel_short_series_raises():
    with pytest.raises(ValueError, match="too short"):
        fit_hampel(np.array([1.0, 2.0]))


def test_fit_hampel_bad_window_raises():
    with pytest.raises(ValueError, match="window_size must be >= 3"):
        fit_hampel(np.arange(100.0), window_size=2)


def test_fit_hampel_n_sigma_controls_rate():
    x, _ = _sine_spikes(n=1000, n_spikes=10, seed=1)
    r_low = fit_hampel(x, n_sigma=2.0)
    r_high = fit_hampel(x, n_sigma=5.0)
    assert r_low.labels.sum() >= r_high.labels.sum()


def test_fit_hampel_contamination_threshold():
    x, _ = _sine_spikes(n=1000, n_spikes=10, seed=2)
    r = fit_hampel(x, window_size=7, contamination=0.02)
    # 2% of 1000 -> threshold set at 98th percentile -> about 20 flagged
    assert 10 <= r.labels.sum() <= 40


def test_fit_hampel_deterministic():
    x, _ = _sine_spikes(n=500, seed=3)
    r1 = fit_hampel(x)
    r2 = fit_hampel(x)
    np.testing.assert_array_equal(r1.decision_scores, r2.decision_scores)


def test_fit_hampel_memory_guard_logic():
    from cadw.core.memory_guard import MemoryLimitExceeded

    class _FakeMB:
        max_memory_mb = 0.001

    x = np.random.default_rng(0).standard_normal(5000)
    with pytest.raises(MemoryLimitExceeded):
        fit_hampel(x, window_size=51, guard=_FakeMB())


def test_hampel_no_bench_deps_in_source():
    src = (ROOT / "cadw/ml/hampel.py").read_text()
    for f in ("import sklearn", "import scipy", "import torch",
              "import aeon", "from sklearn", "from scipy",
              "from torch", "from aeon"):
        assert f not in src, f"forbidden import: {f}"


def test_hampel_reference_sktime_smoke():
    skt = pytest.importorskip("sktime")
    # sktime HampelDetector API is version-dependent; just ensure it runs
    try:
        from sktime.transformations.series.outlier_detection import HampelDetector
        det = HampelDetector(window_length=7, n_sigma=3.0)
    except (ImportError, TypeError):
        pytest.skip("sktime HampelDetector API not available")
    x, _ = _sine_spikes(n=500, seed=4)
    import pandas as pd
    try:
        skt_labels = det.fit_transform(pd.Series(x))
    except Exception:
        pytest.skip("sktime HampelDetector fit_transform raised")
    assert len(skt_labels) == len(x)

