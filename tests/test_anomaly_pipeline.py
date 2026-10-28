"""Tests for AnomalyPipeline (contextual / structural)."""
from __future__ import annotations

import numpy as np
import pytest

from cadw.ml.anomaly_pipeline import (
    AnomalyPipeline, AnomalyPipelineResult, _extract_scores_and_labels,
)


# ── Synthetic detectors for unit isolation ──────────────────────

class _FakeResult:
    def __init__(self, scores, labels, threshold=0.5):
        self.decision_scores = np.asarray(scores, dtype=np.float64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.threshold = threshold


def _fake_detector_threshold(series, *, threshold=1.0):
    """Flags |series| > threshold."""
    scores = np.abs(series)
    labels = (scores > threshold).astype(np.int64)
    return _FakeResult(scores, labels, threshold)


def _fake_detector_len_check(series):
    """Returns length-marked scores (for pipeline mapping check)."""
    scores = np.arange(len(series), dtype=np.float64)
    labels = (scores > len(series) / 2).astype(np.int64)
    return _FakeResult(scores, labels, threshold=len(series) / 2)


# ── _extract_scores_and_labels ──────────────────────────────────

def test_extract_ok():
    r = _FakeResult([1.0, 2.0, 3.0], [0, 0, 1])
    s, l = _extract_scores_and_labels(r)
    assert s.shape == (3,)
    assert l.shape == (3,)
    assert s.dtype == np.float64
    assert l.dtype == np.int64


def test_extract_missing_raises():
    class _BadResult:
        pass
    with pytest.raises(TypeError, match="lacks decision_scores"):
        _extract_scores_and_labels(_BadResult())


# ── Structural mode ─────────────────────────────────────────────

def test_structural_no_transform():
    """Structural mode: detector sees raw y (scores = |y|)."""
    y = np.array([0.5, -0.6, 5.0, 0.3, -0.4, 0.2, 0.1, 0.3])
    pipe = AnomalyPipeline(_fake_detector_threshold,
                           mode="structural", threshold=1.0)
    r = pipe.fit_predict(y)
    assert isinstance(r, AnomalyPipelineResult)
    assert r.mode == "structural"
    assert r.transforms_applied == []
    np.testing.assert_allclose(r.decision_scores, np.abs(y))
    assert r.labels[2] == 1  # 5.0 > 1.0
    assert r.labels[0] == 0  # 0.5 < 1.0


def test_structural_no_length_change():
    y = np.arange(50, dtype=np.float64)
    pipe = AnomalyPipeline(_fake_detector_len_check,
                           mode="structural")
    r = pipe.fit_predict(y)
    assert r.decision_scores.size == 50


# ── Contextual mode with STL ────────────────────────────────────

def test_contextual_stl_removes_seasonal():
    """On strongly seasonal y, STL residual has low magnitude; detector
    on raw (structural) flags seasonal peaks as anomalies; contextual
    (STL) does not."""
    n = 120
    s = 12
    t = np.arange(n, dtype=np.float64)
    y = 5.0 * np.sin(2 * np.pi * t / s)  # pure seasonal, no anomalies

    pipe = AnomalyPipeline(_fake_detector_threshold,
                           mode="contextual", seasonal_period=s,
                           threshold=1.0)
    r = pipe.fit_predict(y)
    assert r.mode == "contextual"
    assert r.transforms_applied == ["stl"]
    # STL residual removes seasonal peak -> nothing flagged
    assert r.labels.sum() == 0, (
        f"contextual STL flagged {r.labels.sum()} seasonal peaks; "
        f"scores max={r.decision_scores.max():.2f}")


def test_structural_naive_flags_seasonal_peaks():
    """Sanity: structural mode on pure seasonal flags peaks (motivates
    the pipeline)."""
    n = 120
    s = 12
    t = np.arange(n, dtype=np.float64)
    y = 5.0 * np.sin(2 * np.pi * t / s)

    pipe = AnomalyPipeline(_fake_detector_threshold,
                           mode="structural", threshold=1.0)
    r = pipe.fit_predict(y)
    # Threshold 1.0; |5*sin| > 1.0 near peaks -> many flagged
    assert r.labels.sum() >= 10, (
        f"structural should flag seasonal peaks; got "
        f"{r.labels.sum()}")


def test_contextual_stl_preserves_length():
    n = 96
    t = np.arange(n, dtype=np.float64)
    y = 5.0 + 2.0 * np.sin(2 * np.pi * t / 12)
    pipe = AnomalyPipeline(_fake_detector_len_check,
                           mode="contextual", seasonal_period=12)
    r = pipe.fit_predict(y)
    assert r.decision_scores.size == n
    assert r.labels.size == n


# ── Contextual mode with detrending ─────────────────────────────

def test_contextual_detrend_removes_trend():
    """Pure linear trend + small spike. Structural flags many points
    (trend range); contextual with detrend flags only the spike."""
    n = 100
    y = 0.5 * np.arange(n, dtype=np.float64)
    y[50] = 200.0  # big spike above trend

    pipe = AnomalyPipeline(_fake_detector_threshold,
                           mode="contextual", detrend=True,
                           threshold=1.0)
    r = pipe.fit_predict(y)
    assert r.mode == "contextual"
    assert "diff_d1" in r.transforms_applied
    assert r.decision_scores.size == n
    # First point unknown (0), spike in diff at t=50, t=51
    assert r.labels.sum() <= 3, (
        f"detrend should isolate spike; got {r.labels.sum()}")


def test_contextual_detrend_first_timestamp_zero():
    """Differencing loses first timestamp; pipeline sets scores[0]=0."""
    n = 50
    y = np.arange(n, dtype=np.float64)
    pipe = AnomalyPipeline(_fake_detector_threshold,
                           mode="contextual", detrend=True,
                           threshold=0.5)
    r = pipe.fit_predict(y)
    assert r.decision_scores[0] == 0.0
    assert r.labels[0] == 0


def test_contextual_detrend_preserves_length():
    n = 50
    y = np.arange(n, dtype=np.float64)
    pipe = AnomalyPipeline(_fake_detector_len_check,
                           mode="contextual", detrend=True)
    r = pipe.fit_predict(y)
    assert r.decision_scores.size == n


# ── Auto mode resolution ────────────────────────────────────────

def test_auto_mode_structural_when_no_transforms():
    y = np.random.RandomState(0).normal(5.0, 0.5, 60)
    pipe = AnomalyPipeline(_fake_detector_threshold, mode="auto")
    r = pipe.fit_predict(y)
    assert r.mode == "structural"


def test_auto_mode_contextual_when_stl():
    y = np.random.RandomState(1).normal(5.0, 0.5, 60) + np.sin(
        2 * np.pi * np.arange(60) / 12) * 2
    pipe = AnomalyPipeline(_fake_detector_threshold, mode="auto",
                           seasonal_period=12)
    r = pipe.fit_predict(y)
    assert r.mode == "contextual"
    assert "stl" in r.transforms_applied


def test_auto_mode_contextual_when_detrend():
    y = np.random.RandomState(2).normal(5.0, 0.5, 60)
    pipe = AnomalyPipeline(_fake_detector_threshold, mode="auto",
                           detrend=True)
    r = pipe.fit_predict(y)
    assert r.mode == "contextual"


# ── Validation ─────────────────────────────────────────────────

def test_bad_detector_raises():
    with pytest.raises(ValueError, match="detector_fn must be callable"):
        AnomalyPipeline("not_callable")


def test_bad_mode_raises():
    with pytest.raises(ValueError, match="mode="):
        AnomalyPipeline(_fake_detector_threshold, mode="magic")


def test_bad_seasonal_period():
    with pytest.raises(ValueError, match="seasonal_period must be >= 2"):
        AnomalyPipeline(_fake_detector_threshold, seasonal_period=1)


def test_short_series_raises():
    pipe = AnomalyPipeline(_fake_detector_threshold)
    with pytest.raises(ValueError, match="at least 4"):
        pipe.fit_predict(np.array([1.0, 2.0, 3.0]))


def test_nan_series_raises():
    pipe = AnomalyPipeline(_fake_detector_threshold)
    with pytest.raises(ValueError, match="non-finite"):
        pipe.fit_predict(np.array([1.0, np.nan, 3.0, 4.0, 5.0]))


def test_contextual_requested_but_no_transform_falls_back():
    """mode='contextual' without seasonal_period or detrend: silent
    fallback to structural (honest: detector sees raw)."""
    y = np.random.RandomState(3).normal(5.0, 0.5, 60)
    pipe = AnomalyPipeline(_fake_detector_threshold,
                           mode="contextual")
    r = pipe.fit_predict(y)
    assert r.mode == "structural"
    assert r.transforms_applied == []


def test_get_params_before_and_after_fit():
    pipe = AnomalyPipeline(_fake_detector_threshold,
                           mode="contextual", seasonal_period=12)
    p1 = pipe.get_params()
    assert p1["resolved_mode"] is None
    y = np.random.RandomState(4).normal(5.0, 0.5, 60)
    pipe.fit_predict(y)
    p2 = pipe.get_params()
    assert p2["resolved_mode"] == "contextual"
    assert p2["transforms_applied"] == ["stl"]


# ── AST guard ───────────────────────────────────────────────────

def test_anomaly_pipeline_no_bench_deps_in_source():
    import ast
    from pathlib import Path
    src = Path("cadw/ml/anomaly_pipeline.py").read_text()
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

def test_pipeline_guard_stl_raises_small_limit():
    """v0.9.87-fix4: guard raises on tight limit (STL branch)."""
    import numpy as np
    import pytest
    from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded

    n = 10000
    y = 5.0 * np.sin(2 * np.pi * np.arange(n) / 12)
    pipe = AnomalyPipeline(_fake_detector_threshold,
                           mode="contextual", seasonal_period=12)
    mg = MemoryGuard(max_memory_bytes=1024)
    with pytest.raises(MemoryLimitExceeded):
        pipe.fit_predict(y, memory_guard=mg)


def test_pipeline_guard_diff_raises_small_limit():
    """v0.9.87-fix4: guard raises on tight limit (diff branch)."""
    import numpy as np
    import pytest
    from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded

    n = 10000
    y = np.arange(n, dtype=np.float64)
    pipe = AnomalyPipeline(_fake_detector_threshold,
                           mode="contextual", detrend=True)
    mg = MemoryGuard(max_memory_bytes=1024)
    with pytest.raises(MemoryLimitExceeded):
        pipe.fit_predict(y, memory_guard=mg)


def test_pipeline_no_guard_backward_compat():
    """v0.9.87-fix4: memory_guard=None default -> prior behavior."""
    import numpy as np

    y = np.array([0.5, -0.6, 5.0, 0.3, -0.4, 0.2, 0.1, 0.3])
    pipe = AnomalyPipeline(_fake_detector_threshold,
                           mode="structural", threshold=1.0)
    r = pipe.fit_predict(y)
    assert r.labels[2] == 1

