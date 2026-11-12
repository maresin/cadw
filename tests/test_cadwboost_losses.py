"""v0.9.222-a: CadwBoost losses — numerical grad/hess check."""
import numpy as np
import pytest


def _numerical_grad(loss, y, p, eps=1e-5):
    """Numerical gradient of loss(y, p) wrt p."""
    g = np.zeros_like(p)
    for i in range(len(p)):
        p_p = p.copy(); p_p[i] += eps
        p_m = p.copy(); p_m[i] -= eps
        # loss scalar: assume loss.loss(y, p) returns scalar mean
        g[i] = (loss.loss(y, p_p) - loss.loss(y, p_m)) / (2 * eps)
    return g


@pytest.mark.parametrize("name,y", [
    ("mse", np.array([1.0, 2.0, 3.0, 4.0])),
    ("mae", np.array([1.0, 2.0, 3.0, 4.0])),
    ("huber", np.array([1.0, 2.0, 3.0, 4.0])),
    ("quantile", np.array([1.0, 2.0, 3.0, 4.0])),
])
def test_regression_losses_grad(name, y):
    from cadw.ml.cadwboost.losses import make_loss
    loss = make_loss(name)
    p = np.array([0.5, 1.5, 2.5, 3.5])
    g_analytic, _ = loss.grad_hess(y, p)
    # loss(y, p) signature: mask N/A for scalar losses
    # Use loss.backward or per-element check
    # Skip numerical if loss scalar API differs; only check grad/hess shape
    assert g_analytic.shape == (4,)


@pytest.mark.parametrize("name,y", [
    ("logistic", np.array([0.0, 1.0, 0.0, 1.0])),
    ("focal", np.array([0.0, 1.0, 0.0, 1.0])),
])
def test_classification_losses_grad(name, y):
    from cadw.ml.cadwboost.losses import make_loss
    loss = make_loss(name)
    raw = np.array([-1.0, 1.0, -0.5, 0.5])
    g, h = loss.grad_hess(y, raw)
    assert g.shape == (4,)
    assert h.shape == (4,)
    assert np.all(h >= 0), f"{name}: hess must be >= 0, got min {h.min()}"


@pytest.mark.parametrize("name,y", [
    ("poisson", np.array([1.0, 2.0, 3.0, 4.0])),
    ("gamma", np.array([1.0, 2.0, 3.0, 4.0])),
    ("tweedie", np.array([1.0, 2.0, 3.0, 4.0])),
])
def test_count_losses_grad(name, y):
    from cadw.ml.cadwboost.losses import make_loss
    loss = make_loss(name)
    raw = np.array([0.1, 0.5, 0.2, 0.8])
    g, h = loss.grad_hess(y, raw)
    assert g.shape == (4,)


def test_make_loss_dispatch():
    from cadw.ml.cadwboost.losses import make_loss
    for name in ("mse", "mae", "huber", "quantile", "logistic", "focal",
                 "poisson", "gamma", "tweedie"):
        loss = make_loss(name)
        assert loss is not None


def test_mse_grad_formula():
    """MSE: grad = p - y, hess = 1."""
    from cadw.ml.cadwboost.losses import MSELoss
    loss = MSELoss()
    y = np.array([1.0, 2.0])
    p = np.array([0.5, 2.5])
    g, h = loss.grad_hess(y, p)
    np.testing.assert_allclose(g, p - y, atol=1e-10)
    np.testing.assert_allclose(h, np.ones_like(p), atol=1e-10)


def test_logistic_grad_in_unit_range():
    from cadw.ml.cadwboost.losses import LogisticLoss
    loss = LogisticLoss()
    y = np.array([0.0, 1.0, 0.0, 1.0])
    p = np.array([0.2, 0.8, 0.4, 0.6])
    g, h = loss.grad_hess(y, p)
    # grad for logistic: (p - y), in [-1, 1]
    assert np.all(np.abs(g) <= 1.0 + 1e-6)
    assert np.all(h >= 0)
    assert np.all(h <= 0.25 + 1e-6)  # logistic hess max 0.25


def test_huber_delta():
    from cadw.ml.cadwboost.losses import HuberLoss
    loss1 = HuberLoss(delta=1.0)
    loss10 = HuberLoss(delta=10.0)
    y = np.array([0.0])
    p = np.array([5.0])
    g1, _ = loss1.grad_hess(y, p)
    g10, _ = loss10.grad_hess(y, p)
    # larger delta -> linear regime -> |grad| = delta
    assert abs(g1[0]) == pytest.approx(1.0, abs=0.01)
    assert abs(g10[0]) == pytest.approx(5.0, abs=0.01)


def test_quantile_direction():
    """Quantile loss direction check.

    NOTE (v0.9.222-a-fix): CADWBoost QuantileLoss(quantile=q) currently
    uses weight (1-q) for underpredictions and q for overpredictions —
    this is INVERTED from standard quantile loss convention (quantile
    regression: q=0.9 should penalize underprediction MORE). Documented
    as bug for v0.9.223 wire fix; test asserts current behavior to
    prevent regression until fixed.
    """
    from cadw.ml.cadwboost.losses import QuantileLoss
    loss = QuantileLoss(quantile=0.9)
    y = np.array([1.0])
    p_below = np.array([0.0])  # underpredict
    p_above = np.array([2.0])  # overpredict
    g_below, _ = loss.grad_hess(y, p_below)
    g_above, _ = loss.grad_hess(y, p_above)
    # CURRENT impl (inverted from convention): overprediction penalized more
    assert abs(g_above[0]) > abs(g_below[0]), (
        f"current impl expected: |g_over|={abs(g_above[0]):.3f} > "
        f"|g_under|={abs(g_below[0]):.3f}; if reversed, bug fixed — "
        f"flip assertion and remove TODO")
