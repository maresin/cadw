"""v0.9.152: tests for cadw.ml._causal_common.dr_score (P1 L11)."""
import numpy as np
import pytest

from cadw.ml._causal_common import dr_score, aipw_ate


def test_dr_score_exact_when_both_models_correct():
    """If mu0,mu1 exactly match true outcomes, ATE == mean(mu1-mu0)."""
    rng = np.random.RandomState(0)
    n = 1000
    t = rng.randint(0, 2, size=n).astype(np.float64)
    # constant treatment effect 2.0
    mu0 = np.zeros(n)
    mu1 = np.full(n, 2.0)
    y = np.where(t == 1, mu1, mu0)  # exact
    e = np.full(n, 0.5)
    ate = aipw_ate(y, t, mu0, mu1, e)
    assert abs(ate - 2.0) < 1e-10


def test_dr_score_doubly_robust_against_bad_propensity():
    """If mu0,mu1 correct but e biased, ATE still exact (1st term)."""
    rng = np.random.RandomState(1)
    n = 1000
    t = rng.randint(0, 2, size=n).astype(np.float64)
    mu0 = np.zeros(n); mu1 = np.full(n, 3.0)
    y = np.where(t == 1, mu1, mu0)
    e = np.full(n, 0.9)  # "true" propensity is 0.5; biased here
    ate = aipw_ate(y, t, mu0, mu1, e)
    assert abs(ate - 3.0) < 1e-10


def test_dr_score_doubly_robust_against_bad_outcome_models():
    """If e correct but mu0,mu1 biased, ATE corrected to truth."""
    rng = np.random.RandomState(2)
    n = 5000
    e_true = 0.4
    t = (rng.random(n) < e_true).astype(np.float64)
    # True outcome: y = 1 + 2*t + noise
    y = 1.0 + 2.0 * t + rng.normal(scale=0.1, size=n)
    # Biased mu (both = 0)
    mu0 = np.zeros(n); mu1 = np.zeros(n)
    e = np.full(n, e_true)
    ate = aipw_ate(y, t, mu0, mu1, e)
    # IPW terms correct the bias: E[y|t=1]-E[y|t=0] = 2.0
    assert abs(ate - 2.0) < 0.2  # sampling noise


def test_dr_score_shapes():
    y = np.array([1.0, 2.0, 3.0])
    t = np.array([1.0, 0.0, 1.0])
    mu0 = np.zeros(3); mu1 = np.ones(3); e = np.full(3, 0.5)
    out = dr_score(y, t, mu0, mu1, e)
    assert out.shape == (3,)


def test_dr_score_empty():
    y = np.empty(0); t = np.empty(0)
    mu0 = np.empty(0); mu1 = np.empty(0); e = np.empty(0)
    out = dr_score(y, t, mu0, mu1, e)
    assert out.size == 0


def test_dr_score_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        dr_score(np.zeros(3), np.zeros(4), np.zeros(3),
                 np.zeros(3), np.zeros(3))


def test_dr_score_not_binary_treatment():
    with pytest.raises(ValueError, match="binary"):
        dr_score(np.zeros(3), np.array([0.0, 1.0, 2.0]),
                 np.zeros(3), np.zeros(3), np.zeros(3))


def test_dr_score_clipping_extreme_propensity():
    """e=0 or e=1 -> clipped; result finite."""
    y = np.array([1.0, 2.0])
    t = np.array([1.0, 0.0])
    mu0 = np.zeros(2); mu1 = np.zeros(2)
    e = np.array([0.0, 1.0])  # extreme
    out = dr_score(y, t, mu0, mu1, e)
    assert np.isfinite(out).all()
    # For t=1, e=0: term_1 = (1/eps)*y ~ large but finite
    # For t=0, e=1: term_0 = (1/(1-1+eps))*(-mu0) = -mu0/eps
    # No NaN, no inf


def test_dr_score_invalid_eps():
    with pytest.raises(ValueError, match="eps"):
        dr_score(np.zeros(3), np.zeros(3), np.zeros(3),
                 np.zeros(3), np.zeros(3), eps=0.0)
    with pytest.raises(ValueError, match="eps"):
        dr_score(np.zeros(3), np.zeros(3), np.zeros(3),
                 np.zeros(3), np.zeros(3), eps=0.6)


def test_dr_score_term_decomposition():
    """Manually verify AIPW decomposition term-by-term."""
    y = np.array([2.0])
    t = np.array([1.0])
    mu0 = np.array([1.0]); mu1 = np.array([3.0])
    e = np.array([0.4])
    # AIPW = (mu1 - mu0) + (t/e)*(y - mu1) - ((1-t)/(1-e))*(y - mu0)
    #     = 2 + (1/0.4)*(2-3) - 0
    #     = 2 + 2.5*(-1) = 2 - 2.5 = -0.5
    out = dr_score(y, t, mu0, mu1, e)[0]
    assert abs(out - (-0.5)) < 1e-10
