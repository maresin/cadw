"""Tests for BinarySegmentation change-point detection."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cadw.ml.binseg import (BinSegResult, _best_split, _segment_cost,
                            fit_binseg)

ROOT = Path(__file__).parent.parent


def _mean_shift_series(n=2000, changepoints=(500, 1000, 1500),
                       amplitudes=(2.0, -1.5, 1.0), noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n) * noise
    level = 0.0
    for cp, amp in zip(changepoints, amplitudes):
        level += amp
        x[cp:] += amp
    return x, np.array(changepoints)


# ── Unit tests ──────────────────────────────────────────────────

def test_segment_cost_zero_for_constant():
    x = np.ones(50) * 3.0
    assert _segment_cost(x) < 1e-12


def test_segment_cost_matches_manual():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    # mean=2.5, SSE = 2.25+0.25+0.25+2.25 = 5.0
    assert abs(_segment_cost(x) - 5.0) < 1e-10


def test_best_split_finds_obvious_change():
    x = np.concatenate([np.zeros(100), np.ones(100) * 10.0])
    i, g = _best_split(x, min_size=20)
    assert 95 <= i <= 105, f"expected ~100, got {i}"
    assert g > 0


def test_best_split_none_if_too_short():
    x = np.random.default_rng(0).standard_normal(30)
    i, g = _best_split(x, min_size=20)
    assert i == -1  # n=30 < 2*20=40


# ── fit_binseg contract ─────────────────────────────────────────

def test_fit_binseg_shape():
    x, _ = _mean_shift_series(n=2000)
    r = fit_binseg(x, n_bkps=3, min_size=20)
    assert isinstance(r, BinSegResult)
    assert r.decision_scores.shape == (2000,)
    assert r.labels.shape == (2000,)
    assert r.changepoints.shape == (3,)


def test_fit_binseg_detects_mean_shifts():
    x, true_cps = _mean_shift_series(
        n=2000, changepoints=(500, 1000, 1500),
        amplitudes=(3.0, -2.0, 2.0), noise=0.1, seed=0)
    r = fit_binseg(x, n_bkps=3, min_size=20)
    # Each true cp should have a detected cp within ±30 samples
    detected = r.changepoints
    hits = sum(
        1 for cp in true_cps
        if np.min(np.abs(detected - cp)) <= 30)
    assert hits >= 2, f"{hits}/3 detected; cp={detected}, true={true_cps}"


def test_fit_binseg_constant_no_changepoints():
    x = np.ones(500) * 3.0
    r = fit_binseg(x, n_bkps=3, min_size=20)
    # Constant -> no gain at any split -> no changepoints found
    assert len(r.changepoints) == 0
    assert r.labels.sum() == 0


def test_fit_binseg_short_series_raises():
    with pytest.raises(ValueError, match=r"2\*min_size"):
        fit_binseg(np.arange(30.0), n_bkps=3, min_size=20)


def test_fit_binseg_bad_n_bkps_raises():
    x = np.random.default_rng(0).standard_normal(500)
    with pytest.raises(ValueError, match="n_bkps"):
        fit_binseg(x, n_bkps=0)


def test_fit_binseg_bad_min_size_raises():
    x = np.random.default_rng(0).standard_normal(500)
    with pytest.raises(ValueError, match="min_size"):
        fit_binseg(x, n_bkps=3, min_size=0)


def test_fit_binseg_deterministic():
    x, _ = _mean_shift_series(n=1000, seed=3)
    r1 = fit_binseg(x, n_bkps=2, min_size=20)
    r2 = fit_binseg(x, n_bkps=2, min_size=20)
    np.testing.assert_array_equal(r1.decision_scores, r2.decision_scores)
    np.testing.assert_array_equal(r1.changepoints, r2.changepoints)


def test_fit_binseg_labels_binary():
    x, _ = _mean_shift_series(n=1000, seed=5)
    r = fit_binseg(x, n_bkps=2, min_size=20)
    unique = set(r.labels.tolist())
    assert unique.issubset({0, 1}), f"labels not binary: {unique}"


def test_fit_binseg_memory_guard_logic():
    from cadw.core.memory_guard import MemoryLimitExceeded

    class _FakeMB:
        max_memory_mb = 0.001

    x = np.random.default_rng(0).standard_normal(5000)
    with pytest.raises(MemoryLimitExceeded):
        fit_binseg(x, n_bkps=3, min_size=20, guard=_FakeMB())


def test_binseg_no_bench_deps_in_source():
    """AST-based check."""
    import ast
    src = (ROOT / "cadw/ml/binseg.py").read_text()
    tree = ast.parse(src)
    forbidden = ("sktime", "sklearn", "scipy", "torch", "pyod",
                 "aeon", "merlion", "kats", "ruptures")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                assert top not in forbidden, f"forbidden: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            assert top not in forbidden, f"forbidden: {node.module}"

