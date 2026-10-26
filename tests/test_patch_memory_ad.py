"""Tests for PatchMemoryAD (PaAno-inspired)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cadw.ml.patch_memory_ad import (PatchMemoryADResult,
                                     _encode, _encode_backward,
                                     _extract_patches, _sq_dists,
                                     _znormalize_rows,
                                     fit_patch_memory_ad)

ROOT = Path(__file__).parent.parent


def _sine_spikes(n=20000, period=100, n_spikes=15, seed=0, amplitude=20.0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x = np.sin(2 * np.pi * t / period) + 0.1 * rng.standard_normal(n)
    pos = rng.choice(n, size=n_spikes, replace=False)
    x[pos] += amplitude
    return x, pos


def _sine_segments(n=10000, period=100, n_segments=10, seg_len=30,
                   seed=0, amplitude=5.0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x = np.sin(2 * np.pi * t / period) + 0.1 * rng.standard_normal(n)
    starts = rng.choice(n - seg_len, size=n_segments, replace=False)
    pos = []
    for s in starts:
        x[s:s + seg_len] = amplitude
        pos.extend(range(s, s + seg_len))
    return x, np.array(sorted(pos))


# ── Unit: patch extraction ──────────────────────────────────────

def test_extract_patches_shape():
    rng = np.random.default_rng(0)
    windows = rng.standard_normal((100, 50))
    X = _extract_patches(windows, patch_len=10)
    assert X.shape == (100, 50)  # 5 patches * 10 = 50


def test_extract_patches_values():
    windows = np.arange(50.0).reshape(1, 50)
    X = _extract_patches(windows, patch_len=10)
    # Patches are [0..9], [10..19], ..., [40..49] concatenated
    np.testing.assert_array_equal(X[0], np.arange(50.0))


# ── Unit: encoder forward/backward ──────────────────────────────

def test_encode_shapes():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((8, 20))
    W1 = rng.standard_normal((20, 16)) * 0.1
    b1 = np.zeros(16)
    W2 = rng.standard_normal((16, 8)) * 0.1
    b2 = np.zeros(8)
    z, h = _encode(X, W1, b1, W2, b2)
    assert z.shape == (8, 8)
    assert h.shape == (8, 16)


def test_encode_backward_shapes():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((8, 20))
    W1 = rng.standard_normal((20, 16)) * 0.1
    b1 = np.zeros(16)
    W2 = rng.standard_normal((16, 8)) * 0.1
    b2 = np.zeros(8)
    z, h = _encode(X, W1, b1, W2, b2)
    dz = np.ones_like(z)
    dW1, db1, dW2, db2 = _encode_backward(dz, X, h, W2)
    assert dW1.shape == W1.shape
    assert db1.shape == b1.shape
    assert dW2.shape == W2.shape
    assert db2.shape == b2.shape


def test_encode_backward_numerical():
    """Numerical gradient check for one weight in W2."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((4, 6))
    W1 = rng.standard_normal((6, 4)) * 0.2
    b1 = np.zeros(4)
    W2 = rng.standard_normal((4, 3)) * 0.2
    b2 = np.zeros(3)
    z, h = _encode(X, W1, b1, W2, b2)
    dz = rng.standard_normal(z.shape)
    dW1, db1, dW2, db2 = _encode_backward(dz, X, h, W2)
    # Numerical: perturb W2[2, 1] slightly
    eps = 1e-6
    W2_p = W2.copy()
    W2_p[2, 1] += eps
    z_p, _ = _encode(X, W1, b1, W2_p, b2)
    numerical = float(np.sum(dz * (z_p - z))) / eps
    analytical = float(dW2[2, 1])
    assert abs(numerical - analytical) < 1e-4, (
        f"numerical={numerical:.6f}, analytical={analytical:.6f}")


def test_sq_dists_correctness():
    z = np.array([[1.0, 0.0], [0.0, 1.0]])
    mem = np.array([[0.0, 0.0], [1.0, 1.0]])
    d = _sq_dists(z, mem)
    expected = np.array([[1.0, 1.0], [1.0, 1.0]])
    np.testing.assert_allclose(d, expected, atol=1e-10)


def test_znormalize_rows():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 20))
    Z = _znormalize_rows(X)
    np.testing.assert_allclose(Z.mean(axis=1), 0.0, atol=1e-10)
    np.testing.assert_allclose(Z.std(axis=1), 1.0, atol=1e-10)


# ── fit_patch_memory_ad contract ───────────────────────────────

def test_fit_patch_memory_ad_basic_shape():
    x, _ = _sine_spikes(n=3000, n_spikes=5, seed=0)
    r = fit_patch_memory_ad(x, window_size=50, patch_len=10,
                            n_memory=32, n_epochs=3, random_state=0)
    assert isinstance(r, PatchMemoryADResult)
    assert r.decision_scores.shape == (3000,)
    assert r.labels.shape == (3000,)
    assert r.memory_bank.shape == (32, 16)
    assert r.window_scores.shape == (3000 - 50 + 1,)
    assert len(r.training_losses) == 3


def test_fit_patch_memory_ad_deterministic():
    x, _ = _sine_spikes(n=2000, n_spikes=3, seed=3)
    r1 = fit_patch_memory_ad(x, window_size=50, patch_len=10,
                             n_memory=16, n_epochs=2, random_state=42)
    r2 = fit_patch_memory_ad(x, window_size=50, patch_len=10,
                             n_memory=16, n_epochs=2, random_state=42)
    np.testing.assert_array_equal(r1.decision_scores, r2.decision_scores)
    np.testing.assert_allclose(r1.memory_bank, r2.memory_bank)


def test_fit_patch_memory_ad_detects_spikes():
    """Patch-based + memory should detect point spikes at scale."""
    x, pos = _sine_spikes(n=30000, n_spikes=20, seed=0)
    r = fit_patch_memory_ad(x, window_size=50, patch_len=10,
                            n_memory=64, n_epochs=5,
                            contamination=0.05, random_state=0)
    recall = float(np.mean(r.labels[pos] == 1))
    assert recall >= 0.3, f"recall={recall:.2f} on {len(pos)} spikes"


def test_fit_patch_memory_ad_segments_are_not_primary():
    """Patch-based memory is a POINT-anomaly specialist.

    A +20 spike in a 10-point patch makes the patch unique (distance to
    all memory slots large -> high score). A 30-point constant segment
    spans ~3 patches, but each patch is FLAT — indistinguishable from
    normal flat regions of a sine at its peaks/troughs. So segment
    recall is near-zero by design, matching TSB-AD results: PaAno is
    SOTA on point anomalies, not on segment/shape ones.

    For segments prefer KMeansAD, Sub-PCA (mean agg), or CUSUM.
    """
    x, pos = _sine_segments(n=10000, n_segments=10, seg_len=30, seed=0)
    r = fit_patch_memory_ad(x, window_size=50, patch_len=10,
                            n_memory=64, n_epochs=5,
                            contamination=0.05, random_state=0)
    recall = float(np.mean(r.labels[pos] == 1))
    # Honest expectation: near zero; only boundary patches might flag
    assert recall < 0.3, (
        f"unexpected: segments flagged at recall={recall:.2f} "
        f"(documented limitation: patch-based memory is point specialist)")


def test_fit_patch_memory_ad_constant_series():
    x = np.ones(500) * 3.0
    r = fit_patch_memory_ad(x, window_size=50, patch_len=10,
                            n_memory=16, n_epochs=2, random_state=0)
    # All windows identical -> all scores similar -> no anomalies
    # (may not be exactly 0 because memory initialization is random)
    assert r.decision_scores.std() < 10.0  # flat-ish


def test_fit_patch_memory_ad_short_series_raises():
    with pytest.raises(ValueError, match="< window_size"):
        fit_patch_memory_ad(np.arange(30.0), window_size=50)


def test_fit_patch_memory_ad_bad_window_patch_raises():
    x = np.random.default_rng(0).standard_normal(200)
    with pytest.raises(ValueError, match="multiple of"):
        fit_patch_memory_ad(x, window_size=50, patch_len=7)


def test_fit_patch_memory_ad_bad_n_memory_raises():
    x = np.random.default_rng(0).standard_normal(200)
    with pytest.raises(ValueError, match="n_memory"):
        fit_patch_memory_ad(x, window_size=50, patch_len=10, n_memory=1)


def test_fit_patch_memory_ad_bad_aggregate_raises():
    x = np.random.default_rng(0).standard_normal(200)
    with pytest.raises(ValueError, match="aggregate"):
        fit_patch_memory_ad(x, window_size=50, patch_len=10,
                            aggregate="median")


def test_fit_patch_memory_ad_memory_guard_logic():
    from cadw.core.memory_guard import MemoryLimitExceeded

    class _FakeMB:
        max_memory_mb = 0.001

    x = np.random.default_rng(0).standard_normal(5000)
    with pytest.raises(MemoryLimitExceeded):
        fit_patch_memory_ad(x, window_size=100, patch_len=10,
                            n_epochs=1, guard=_FakeMB())


def test_patch_memory_ad_no_bench_deps_in_source():
    import ast
    src = (ROOT / "cadw/ml/patch_memory_ad.py").read_text()
    tree = ast.parse(src)
    forbidden = ("sktime", "sklearn", "scipy", "torch", "pyod",
                 "aeon", "merlion", "kats", "tensorflow")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                assert top not in forbidden, f"forbidden: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            assert top not in forbidden, f"forbidden: {node.module}"

