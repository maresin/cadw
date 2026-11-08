"""v0.9.151: tests for cadw.ml._gbdt_common (P1 group D)."""
import numpy as np
import pytest

from cadw.ml._gbdt_common import (
    pinball_loss, pinball_grad, adaboost_weight, cumulative_logit_prob,
)


# ============================================================================
# pinball_loss / pinball_grad
# ============================================================================

def test_pinball_loss_zero_when_correct():
    y = np.array([1.0, 2.0, 3.0])
    assert (pinball_loss(y, y, 0.5) == 0.0).all()


def test_pinball_loss_median_symmetric():
    """q=0.5: loss(r) == loss(-r) (absolute/2)."""
    r = np.array([1.0, -2.0, 3.0, -4.0])
    y = r.copy()
    yp = np.zeros_like(r)
    np.testing.assert_allclose(
        pinball_loss(y, yp, 0.5), np.abs(r) * 0.5, rtol=1e-12)


def test_pinball_loss_asymmetric():
    """q=0.9 (high quantile): under-prediction penalized MORE than over.

    For q > 0.5, quantile regression pushes prediction up (conservative
    for the high-quantile target); under-prediction gets weight q,
    over-prediction gets weight (1-q).
    """
    y = np.array([1.0, 1.0])
    yp_over = np.array([2.0, 1.0])  # over by 1, r=-1 -> loss=(q-1)*(-1)=0.1
    yp_under = np.array([0.0, 1.0])  # under by 1, r=+1 -> loss=q*1=0.9
    l_over = pinball_loss(y, yp_over, 0.9)[0]
    l_under = pinball_loss(y, yp_under, 0.9)[0]
    assert l_under > l_over
    np.testing.assert_allclose(l_over, 0.1, rtol=1e-12)
    np.testing.assert_allclose(l_under, 0.9, rtol=1e-12)


def test_pinball_grad_matches_loss_derivative():
    """Numeric check of gradient vs loss."""
    y = np.array([1.0, 2.0, 3.0])
    yp = np.array([1.5, 1.5, 3.5])
    q = 0.3
    eps = 1e-6
    grad_num = (pinball_loss(y, yp + eps, q) - pinball_loss(y, yp - eps, q)) / (2 * eps)
    grad_anal = -pinball_grad(y, yp, q)  # grad returns negative gradient
    np.testing.assert_allclose(grad_anal, grad_num, atol=1e-6)


def test_pinball_grad_values():
    y = np.array([1.0, 2.0])
    yp = np.array([0.5, 2.5])  # first under, second over
    q = 0.7
    out = pinball_grad(y, yp, q)
    # under (r>0): q=0.7; over (r<0): q-1=-0.3
    np.testing.assert_allclose(out, [0.7, -0.3], rtol=1e-12)


def test_pinball_invalid_quantile():
    with pytest.raises(ValueError, match="quantile"):
        pinball_loss(np.zeros(3), np.zeros(3), 0.0)
    with pytest.raises(ValueError, match="quantile"):
        pinball_loss(np.zeros(3), np.zeros(3), 1.0)


# ============================================================================
# adaboost_weight
# ============================================================================

def test_adaboost_weight_correct_downweighted():
    """Correct samples -> exp(-alpha) < 1 (weight decreases)."""
    w = adaboost_weight(1.0, np.array([0, 0, 1]))
    assert w[0] < 1.0
    assert w[1] < 1.0
    assert w[2] > 1.0  # misclassified -> upweighted


def test_adaboost_weight_values():
    alpha = 0.5
    mis = np.array([0, 1])
    out = adaboost_weight(alpha, mis)
    np.testing.assert_allclose(out, [np.exp(-alpha), np.exp(alpha)], rtol=1e-12)


def test_adaboost_weight_zero_alpha():
    """alpha=0 -> no change (all 1.0)."""
    w = adaboost_weight(0.0, np.array([0, 1, 1]))
    np.testing.assert_array_equal(w, np.ones(3))


# ============================================================================
# cumulative_logit_prob
# ============================================================================

def test_cumulative_logit_sums_to_one():
    eta = np.array([0.0, 1.0, -1.0, 2.0])
    theta = np.array([-1.0, 0.5, 2.0])  # K = 4
    probs = cumulative_logit_prob(eta, theta)
    assert probs.shape == (4, 4)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-12)
    assert (probs >= 0).all()


def test_cumulative_logit_monotone_in_eta():
    """Higher eta -> higher class probability."""
    theta = np.array([-1.0, 0.5, 2.0])
    eta_low = np.array([-5.0])
    eta_high = np.array([5.0])
    p_low = cumulative_logit_prob(eta_low, theta)
    p_high = cumulative_logit_prob(eta_high, theta)
    # eta low -> mass at low classes; eta high -> mass at high classes
    assert p_low[0, 0] > p_high[0, 0]
    assert p_low[0, -1] < p_high[0, -1]


def test_cumulative_logit_scalar_eta():
    probs = cumulative_logit_prob(0.0, np.array([-1.0, 1.0]))
    assert probs.shape == (3,)  # K=3


def test_cumulative_logit_theta_not_increasing():
    with pytest.raises(ValueError, match="increasing"):
        cumulative_logit_prob(np.zeros(3), np.array([1.0, 0.5]))


def test_cumulative_logit_theta_too_short():
    with pytest.raises(ValueError, match="at least 1"):
        cumulative_logit_prob(np.zeros(3), np.array([]))


def test_cumulative_logit_eta_extreme():
    """Sigmoid numerical stability: extreme eta -> 0 or 1, not NaN."""
    eta = np.array([-1000.0, 1000.0])
    theta = np.array([0.0])
    probs = cumulative_logit_prob(eta, theta)
    assert np.isfinite(probs).all()
    assert probs[0, 0] > 0.999  # eta=-1000 -> P(Y=0) near 1
    assert probs[1, 0] < 0.001  # eta=+1000 -> P(Y=0) near 0
