"""Tests for MLP Autoencoder anomaly detection."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cadw.ml.autoencoder import (AutoencoderResult, _forward, _backward,
                                 _znormalize_rows, fit_autoencoder)

ROOT = Path(__file__).parent.parent


def _sine_spikes(n=10000, period=100, n_spikes=15, seed=0, amplitude=20.0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x = np.sin(2 * np.pi * t / period) + 0.1 * rng.standard_normal(n)
    pos = rng.choice(n, size=n_spikes, replace=False)
    x[pos] += amplitude
    return x, pos


# ── Unit tests: z-norm, forward/backward ─────────────────────────

def test_znormalize_rows_mean_zero_std_one():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 20))
    Z = _znormalize_rows(X)
    np.testing.assert_allclose(Z.mean(axis=1), 0.0, atol=1e-10)
    np.testing.assert_allclose(Z.std(axis=1), 1.0, atol=1e-10)


def test_znormalize_constant_row():
    X = np.ones((5, 10)) * 7.0
    Z = _znormalize_rows(X)
    np.testing.assert_allclose(Z, 0.0)  # std floored to 1


def test_forward_shapes():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((8, 20))
    W1 = _xavier = rng.uniform(-0.1, 0.1, (20, 16))
    b1 = np.zeros(16)
    W2 = rng.uniform(-0.1, 0.1, (16, 4))
    b2 = np.zeros(4)
    W3 = rng.uniform(-0.1, 0.1, (4, 16))
    b3 = np.zeros(16)
    W4 = rng.uniform(-0.1, 0.1, (16, 20))
    b4 = np.zeros(20)
    x_hat, cache = _forward(X, W1, b1, W2, b2, W3, b3, W4, b4)
    assert x_hat.shape == (8, 20)
    h1, z, h2 = cache
    assert h1.shape == (8, 16)
    assert z.shape == (8, 4)
    assert h2.shape == (8, 16)


def test_backward_grad_shapes():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((8, 10))
    W1 = rng.uniform(-0.1, 0.1, (10, 6))
    b1 = np.zeros(6)
    W2 = rng.uniform(-0.1, 0.1, (6, 3))
    b2 = np.zeros(3)
    W3 = rng.uniform(-0.1, 0.1, (3, 6))
    b3 = np.zeros(6)
    W4 = rng.uniform(-0.1, 0.1, (6, 10))
    b4 = np.zeros(10)
    x_hat, cache = _forward(X, W1, b1, W2, b2, W3, b3, W4, b4)
    grads = _backward(X, x_hat, cache, W1, W2, W3, W4)
    dW1, db1, dW2, db2, dW3, db3, dW4, db4 = grads
    assert dW1.shape == (10, 6)
    assert db1.shape == (6,)
    assert dW2.shape == (6, 3)
    assert dW3.shape == (3, 6)
    assert dW4.shape == (6, 10)


# ── Contract / determinism ──────────────────────────────────────

def test_fit_autoencoder_shape():
    x, _ = _sine_spikes(n=2000, n_spikes=5, seed=0)
    r = fit_autoencoder(x, window_size=30, hidden_size=16, latent_size=4,
                        n_epochs=5, random_state=0)
    assert isinstance(r, AutoencoderResult)
    assert r.decision_scores.shape == (2000,)
    assert r.labels.shape == (2000,)
    assert r.window_scores.shape == (2000 - 30 + 1,)
    assert len(r.training_losses) == 5


def test_fit_autoencoder_deterministic():
    x, _ = _sine_spikes(n=1000, n_spikes=3, seed=3)
    r1 = fit_autoencoder(x, window_size=30, n_epochs=5, random_state=42)
    r2 = fit_autoencoder(x, window_size=30, n_epochs=5, random_state=42)
    np.testing.assert_array_equal(r1.decision_scores, r2.decision_scores)
    np.testing.assert_allclose(r1.training_losses, r2.training_losses)


def test_fit_autoencoder_loss_decreases():
    """Training loss should decrease substantially over epochs."""
    x, _ = _sine_spikes(n=3000, n_spikes=5, seed=1)
    r = fit_autoencoder(x, window_size=30, hidden_size=16, latent_size=4,
                        n_epochs=30, lr=0.02, random_state=0)
    first = float(np.mean(r.training_losses[:3]))
    last = float(np.mean(r.training_losses[-3:]))
    assert last < first * 0.7, (
        f"loss did not decrease: first={first:.4f}, last={last:.4f}")


def test_fit_autoencoder_detects_spikes():
    """Spike detection with median reconstruction error.

    Scale matters: MLP memorises 600 affected windows out of 10k easily
    (capacity 1900 params, 4650 updates); 3000 affected windows out of
    50k cannot all be memorised (capacity 4900, 23400 updates, 3000
    diverse windows). Hence bench-scale n=50000 with w=50 required.

    Contamination matches window-spread: 30 spikes × 50 × 2 (spread
    ±w before/after) / 50000 ≈ 6%. Use 8% for margin.
    """
    x, pos = _sine_spikes(n=50000, n_spikes=30, seed=0)
    # v0.9.63-fix2: normalization='none' (default) preserves spike
    # amplitude; z-norm 'window' suppresses amplitude anomalies per
    # HYDRA (SIGMOD 2026).
    r = fit_autoencoder(x, window_size=50, hidden_size=32, latent_size=8,
                        n_epochs=30, lr=0.01, batch_size=64,
                        contamination=0.08, aggregate="max",
                        reconstruction_error="median",
                        normalization="none", random_state=0)
    recall = float(np.mean(r.labels[pos] == 1))
    assert recall >= 0.4, f"recall={recall:.2f} on {len(pos)} spikes"


def test_fit_autoencoder_median_vs_mean():
    """Both modes give valid scores; median is more robust to point
    memorization (documented behaviour, not asserted recall here)."""
    x, _ = _sine_spikes(n=5000, n_spikes=5, seed=3)
    r_mean = fit_autoencoder(x, window_size=30, hidden_size=16,
                             latent_size=4, n_epochs=10,
                             reconstruction_error="mean", random_state=0)
    r_med = fit_autoencoder(x, window_size=30, hidden_size=16,
                            latent_size=4, n_epochs=10,
                            reconstruction_error="median", random_state=0)
    assert r_mean.decision_scores.shape == r_med.decision_scores.shape
    assert r_med.reconstruction_error == "median"
    # Mean score should be >= median score per window (mean >= median of squares)
    assert r_mean.window_scores.mean() >= r_med.window_scores.mean() - 1e-9


def test_fit_autoencoder_bad_reconstruction_error_raises():
    x = np.random.default_rng(0).standard_normal(200)
    with pytest.raises(ValueError, match="reconstruction_error"):
        fit_autoencoder(x, window_size=30, n_epochs=1,
                        reconstruction_error="max")


def test_fit_autoencoder_constant_series():
    x = np.ones(500) * 3.0
    r = fit_autoencoder(x, window_size=30, n_epochs=5, random_state=0)
    # All windows identical -> z-normalized to zero -> reconstruction
    # should be near zero after enough epochs. With 5 epochs, may be
    # non-zero but flat. Test flatness.
    assert r.decision_scores.std() < 0.05


def test_fit_autoencoder_short_series_raises():
    with pytest.raises(ValueError, match="< window_size"):
        fit_autoencoder(np.arange(10.0), window_size=50)


def test_fit_autoencoder_bad_params_raise():
    x = np.random.default_rng(0).standard_normal(200)
    with pytest.raises(ValueError, match="hidden_size"):
        fit_autoencoder(x, window_size=30, hidden_size=0)
    with pytest.raises(ValueError, match="latent_size"):
        fit_autoencoder(x, window_size=30, latent_size=0)
    with pytest.raises(ValueError, match="aggregate"):
        fit_autoencoder(x, window_size=30, aggregate="median")
    with pytest.raises(ValueError, match="n_epochs"):
        fit_autoencoder(x, window_size=30, n_epochs=0)


def test_fit_autoencoder_memory_guard_logic():
    from cadw.core.memory_guard import MemoryLimitExceeded

    class _FakeMB:
        max_memory_mb = 0.001

    x = np.random.default_rng(0).standard_normal(5000)
    with pytest.raises(MemoryLimitExceeded):
        fit_autoencoder(x, window_size=100, n_epochs=1, guard=_FakeMB())


def test_fit_autoencoder_memory_guard_real_smoke():
    from cadw.core.memory_guard import MemoryGuard
    guard = MemoryGuard(max_memory_mb=1000)
    x = np.random.default_rng(1).standard_normal(500)
    r = fit_autoencoder(x, window_size=30, n_epochs=2, guard=guard)
    assert r.decision_scores.shape == (500,)


def test_autoencoder_no_bench_deps_in_source():
    """AST check: no forbidden imports."""
    import ast
    src = (ROOT / "cadw/ml/autoencoder.py").read_text()
    tree = ast.parse(src)
    forbidden = ("sktime", "sklearn", "scipy", "torch", "pyod",
                 "aeon", "merlion", "kats", "tensorflow", "jax")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                assert top not in forbidden, f"forbidden: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            assert top not in forbidden, f"forbidden: {node.module}"

def test_fit_autoencoder_guard_raises_before_allocation(monkeypatch):
    """v0.9.87-fix1b(2/2): guard fires BEFORE _prepare_windows."""
    from cadw.ml import autoencoder as ae_mod
    from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded

    called = {"n": 0}
    orig = ae_mod._prepare_windows

    def spy(*a, **kw):
        called["n"] += 1
        return orig(*a, **kw)

    monkeypatch.setattr(ae_mod, "_prepare_windows", spy)

    guard = MemoryGuard(max_memory_mb=10)
    x = np.random.default_rng(0).standard_normal(100_000)
    with pytest.raises(MemoryLimitExceeded):
        ae_mod.fit_autoencoder(x, window_size=64, guard=guard)
    assert called["n"] == 0, (
        f"guard should fire BEFORE _prepare_windows; "
        f"_prepare_windows was called {called['n']} time(s)")

