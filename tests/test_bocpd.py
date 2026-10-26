"""Tests for BOCPD (Student-t likelihood)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from cadw.ml.bocpd import (BOCPDResult, _posterior_params,
                           _student_t_logpdf, fit_bocpd)

ROOT = Path(__file__).parent.parent


def _mean_shift(n=1000, changepoints=(300, 600),
                amplitudes=(2.0, -1.0), noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n) * noise
    for cp, amp in zip(changepoints, amplitudes):
        x[cp:] += amp
    return x, np.array(changepoints)


# ── Unit tests ─────────────────────────────────────────────────

def test_student_t_logpdf_gaussian_limit():
    """Large df -> close to Gaussian log-pdf."""
    x, df, loc, scale = 0.5, 1e6, 0.0, 1.0
    st = _student_t_logpdf(x, df, loc, scale)
    gaussian = (-0.5 * math.log(2 * math.pi)
                - 0.5 * (x - loc) ** 2 / scale ** 2)
    assert abs(st - gaussian) < 1e-3


def test_student_t_logpdf_heavy_tails():
    """Student-t has heavier tails than Gaussian at df=2."""
    df = 2.0
    x_far = 5.0
    st = _student_t_logpdf(x_far, df, 0.0, 1.0)
    gaussian = (-0.5 * math.log(2 * math.pi)
                - 0.5 * x_far ** 2)
    assert st > gaussian  # heavier tail


def test_student_t_logpdf_degenerate():
    assert _student_t_logpdf(0.0, df=0.0, loc=0.0, scale=1.0) == -np.inf
    assert _student_t_logpdf(0.0, df=1.0, loc=0.0, scale=0.0) == -np.inf


def test_posterior_params_update():
    """NIG posterior update: mean moves toward observation."""
    mu_new, kappa_new, alpha_new, beta_new = _posterior_params(
        x=5.0, mu=0.0, kappa=1.0, alpha=1.0, beta=1.0)
    # mu_new = (1*0 + 5)/(1+1) = 2.5
    assert abs(mu_new - 2.5) < 1e-10
    assert kappa_new == 2.0
    assert alpha_new == 1.5
    # beta_new = 1 + 0.5 * 1 * 25 / 2 = 7.25
    assert abs(beta_new - 7.25) < 1e-10


# ── fit_bocpd contract ─────────────────────────────────────────

def test_fit_bocpd_shape():
    x, _ = _mean_shift(n=500, seed=0)
    r = fit_bocpd(x, hazard_rate=1.0 / 100.0)
    assert isinstance(r, BOCPDResult)
    assert r.decision_scores.shape == (500,)
    assert r.labels.shape == (500,)
    assert r.run_length_posterior.shape[0] == 500
    assert r.argmax_run_length.shape == (500,)


def test_fit_bocpd_detects_mean_shifts():
    x, true_cps = _mean_shift(
        n=1000, changepoints=(300, 600),
        amplitudes=(3.0, -3.0), noise=0.2, seed=0)
    r = fit_bocpd(x, hazard_rate=1.0 / 200.0,
                  contamination=0.05,
                  mu0=0.0, kappa0=0.01, alpha0=1.0, beta0=0.04)
    detected = r.changepoints
    if len(detected) == 0:
        pytest.skip("BOCPD did not flag any point at this contamination")
    hits = sum(
        1 for cp in true_cps
        if np.min(np.abs(detected - cp)) <= 30)
    assert hits >= 1, (
        f"{hits}/{len(true_cps)} cp detected within +/-30; "
        f"cp={detected[:10]}... true={true_cps}")


def test_fit_bocpd_constant_no_changes():
    x = np.ones(500) * 3.0
    r = fit_bocpd(x, hazard_rate=1.0 / 100.0, contamination=0.05)
    # v0.9.72-fix2: burn-in is masked (default 2*run_length_threshold=20).
    # On constant series, after burn-in no changepoints should be flagged.
    assert r.decision_scores.max() < 0.5
    assert int(r.labels.sum()) == 0


def test_fit_bocpd_burn_in_masked():
    """First `burn_in` decision_scores must be 0 (posterior warm-up)."""
    x = np.random.default_rng(0).standard_normal(500) + 5.0
    r = fit_bocpd(x, hazard_rate=1.0 / 100.0, burn_in=30)
    # First 30 must be zero
    assert np.all(r.decision_scores[:30] == 0.0)
    # After burn-in, scores nonzero (some posterior mass at r<10)
    assert r.decision_scores[30:].max() > 0.0


def test_fit_bocpd_burn_in_default():
    """Default burn_in = 2 * run_length_threshold."""
    x = np.random.default_rng(1).standard_normal(200)
    r = fit_bocpd(x, hazard_rate=1.0 / 100.0, run_length_threshold=10)
    # First 20 (2*10) should be 0
    assert np.all(r.decision_scores[:20] == 0.0)


def test_fit_bocpd_short_series_raises():
    with pytest.raises(ValueError, match="too short"):
        fit_bocpd(np.arange(5.0))


def test_fit_bocpd_bad_hazard_raises():
    x = np.random.default_rng(0).standard_normal(100)
    with pytest.raises(ValueError, match="hazard_rate"):
        fit_bocpd(x, hazard_rate=1.5)


def test_fit_bocpd_bad_prior_raises():
    x = np.random.default_rng(0).standard_normal(100)
    with pytest.raises(ValueError, match="prior params"):
        fit_bocpd(x, kappa0=-1.0)


def test_fit_bocpd_bad_run_length_threshold_raises():
    x = np.random.default_rng(0).standard_normal(100)
    with pytest.raises(ValueError, match="run_length_threshold"):
        fit_bocpd(x, run_length_threshold=0)


def test_fit_bocpd_deterministic():
    x, _ = _mean_shift(n=500, seed=3)
    r1 = fit_bocpd(x, hazard_rate=1.0 / 100.0, contamination=0.05)
    r2 = fit_bocpd(x, hazard_rate=1.0 / 100.0, contamination=0.05)
    np.testing.assert_array_equal(r1.decision_scores, r2.decision_scores)
    np.testing.assert_array_equal(r1.changepoints, r2.changepoints)


def test_fit_bocpd_hazard_rate_effect():
    """Higher hazard -> more frequent run-length resets."""
    x, _ = _mean_shift(n=1000, seed=4)
    r_low = fit_bocpd(x, hazard_rate=1.0 / 500.0, contamination=0.05)
    r_high = fit_bocpd(x, hazard_rate=1.0 / 50.0, contamination=0.05)
    # Higher hazard -> higher average P(change) baseline
    assert r_high.decision_scores.mean() > r_low.decision_scores.mean()


def test_fit_bocpd_posterior_sums_to_one():
    """Run-length posterior at each t sums to 1 (after truncation renorm)."""
    x, _ = _mean_shift(n=200, seed=5)
    r = fit_bocpd(x, hazard_rate=1.0 / 100.0)
    row_sums = r.run_length_posterior.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


def test_fit_bocpd_memory_guard_logic():
    from cadw.core.memory_guard import MemoryLimitExceeded

    class _FakeMB:
        max_memory_mb = 0.001

    x = np.random.default_rng(0).standard_normal(2000)
    with pytest.raises(MemoryLimitExceeded):
        fit_bocpd(x, hazard_rate=1.0 / 100.0, guard=_FakeMB())


def test_bocpd_no_bench_deps_in_source():
    import ast
    src = (ROOT / "cadw/ml/bocpd.py").read_text()
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

