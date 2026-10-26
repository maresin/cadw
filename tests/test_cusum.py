"""Tests for CUSUM changepoint detection."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cadw.ml.cusum import CUSUMResult, fit_cusum

ROOT = Path(__file__).parent.parent


def _mean_shift(n=1000, shift_at=500, shift=2.0, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n) * noise
    x[shift_at:] += shift
    return x, shift_at


def test_fit_cusum_basic_shape():
    x, _ = _mean_shift(n=500, seed=0)
    r = fit_cusum(x, k=0.5, h=5.0)
    assert isinstance(r, CUSUMResult)
    assert r.decision_scores.shape == (500,)
    assert r.labels.shape == (500,)
    assert r.k == 0.5
    assert r.h == 5.0


def test_fit_cusum_detects_mean_shift():
    # shift at 800 (20% contamination): sigma not inflated, clear separation.
    # h=2.0 with z_post~2.0, k=0.5 -> s_pos grows 1.5/step, hits h in 2 steps
    # -> ~50% post-shift flag rate.
    x, shift_at = _mean_shift(n=1000, shift_at=800, shift=2.0, noise=0.1, seed=0)
    r = fit_cusum(x, k=0.5, h=2.0)
    post_shift = r.labels[shift_at:]
    pre_shift = r.labels[:shift_at]
    assert post_shift.mean() >= 0.4, (
        f"post-shift flag rate {post_shift.mean():.2f} too low")
    assert pre_shift.mean() <= 0.1, (
        f"pre-shift flag rate {pre_shift.mean():.2f} too high")


def test_fit_cusum_negative_shift():
    """Two-sided CUSUM must detect drops, not only rises."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1000) * 0.1
    x[800:] -= 2.0  # drop (20% contamination)
    r = fit_cusum(x, k=0.5, h=2.0)
    post = r.labels[800:]
    assert post.mean() >= 0.3, f"negative shift not detected: {post.mean():.2f}"


def test_fit_cusum_constant_series():
    x = np.ones(200) * 3.0
    r = fit_cusum(x, k=0.5, h=5.0)
    # Constant -> z = 0 - k -> s_pos/s_neg = 0 after clipping
    assert r.labels.sum() == 0


def test_fit_cusum_short_series_raises():
    with pytest.raises(ValueError, match="too short"):
        fit_cusum(np.array([1.0, 2.0]))


def test_fit_cusum_deterministic():
    x, _ = _mean_shift(n=500, seed=3)
    r1 = fit_cusum(x)
    r2 = fit_cusum(x)
    np.testing.assert_array_equal(r1.decision_scores, r2.decision_scores)


def test_fit_cusum_h_controls_rate():
    x, _ = _mean_shift(n=1000, shift_at=800, shift=2.0, noise=0.1, seed=1)
    r_low = fit_cusum(x, k=0.5, h=2.0)
    r_high = fit_cusum(x, k=0.5, h=10.0)
    assert r_low.labels.sum() >= r_high.labels.sum()


def test_fit_cusum_contamination_threshold():
    x, _ = _mean_shift(n=1000, shift_at=800, shift=2.0, noise=0.1, seed=2)
    r = fit_cusum(x, k=0.5, h=2.0, contamination=0.05)
    # ~5% at percentile threshold; loose sanity bound
    assert 5 <= r.labels.sum() <= 200


def test_fit_cusum_scores_monotone_post_shift():
    """After persistent positive shift, s_pos grows until >h then resets
    -> post-shift scores have peaks (pre-reset max ~ h+step). Pre-shift
    baseline s_pos stays near 0."""
    x, shift_at = _mean_shift(n=1000, shift_at=800, shift=3.0, noise=0.1, seed=4)
    r = fit_cusum(x, k=0.5, h=5.0)
    max_post = float(r.decision_scores[shift_at:].max())
    max_pre = float(r.decision_scores[:shift_at].max())
    assert max_post > max_pre + 3.0, (
        f"post-shift max {max_post:.2f} not clearly > pre-shift max {max_pre:.2f}")


def test_fit_cusum_memory_guard_logic():
    from cadw.core.memory_guard import MemoryLimitExceeded

    class _FakeMB:
        max_memory_mb = 0.001

    x = np.random.default_rng(0).standard_normal(5000)
    with pytest.raises(MemoryLimitExceeded):
        fit_cusum(x, guard=_FakeMB())


def test_cusum_no_bench_deps_in_source():
    src = (ROOT / "cadw/ml/cusum.py").read_text()
    for f in ("import sklearn", "import scipy", "import torch",
              "import kats", "from sklearn", "from scipy",
              "from torch", "from kats"):
        assert f not in src, f"forbidden import: {f}"


def test_cusum_reference_kats_smoke():
    pytest.importorskip("kats")
    from kats.detectors.cusum_detection import CUSUMDetector
    import pandas as pd
    x, _ = _mean_shift(n=500, seed=5)
    df = pd.DataFrame({
        "time": pd.date_range("2020-01-01", periods=500, freq="D"),
        "value": x,
    })
    try:
        det = CUSUMDetector(df)
        changes = det.detector(change_directions=["increase"])
    except Exception:
        pytest.skip("kats CUSUM API unavailable")
    assert len(changes) >= 0

