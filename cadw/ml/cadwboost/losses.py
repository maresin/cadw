"""Loss functions: grad + hess for GBDT.

Convention: `grad` = dL/d_pred (gradient of loss w.r.t. raw prediction),
`hess` = d²L/d_pred² (second derivative). Tree fits on `grad, hess`;
leaf value = -G / (H + λ) is the Newton step.

For binary classification, `pred` = logit (raw score).
For count/positive regression, `pred` = log(mu) (log link).

Init prediction (`init_pred`) chosen so that initial p minimizes loss
on training data: e.g. MSE → y.mean(), logistic → logit(y.mean()).
"""
from __future__ import annotations

import numpy as np


class Loss:
    name: str = "base"
    task: str = "regression"   # "regression" | "binary" | "count"

    def init_pred(self, y: np.ndarray) -> float:
        raise NotImplementedError

    def grad_hess(
        self, y: np.ndarray, pred: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def predict_from_raw(self, raw: np.ndarray) -> np.ndarray:
        """Convert raw model output to prediction (identity for regression)."""
        return raw


class MSELoss(Loss):
    name = "mse"
    task = "regression"

    def init_pred(self, y):
        return float(np.mean(y))

    def grad_hess(self, y, pred):
        return pred - y, np.ones_like(y)


class MAELoss(Loss):
    """MAE: grad = sign(pred - y), hess = 1 (constant; XGBoost convention).

    True second derivative is Dirac at 0; XGBoost uses hess=1 for
    stability. Reference: Friedman 2001 (gradient boosting with L1 uses
    median for init, sign for grad).
    """
    name = "mae"
    task = "regression"

    def init_pred(self, y):
        return float(np.median(y))

    def grad_hess(self, y, pred):
        diff = pred - y
        grad = np.sign(diff)
        hess = np.ones_like(y)
        return grad, hess


class HuberLoss(Loss):
    """Huber: quadratic for |r| < δ, linear beyond. δ = `delta`."""
    name = "huber"
    task = "regression"

    def __init__(self, delta: float = 1.0):
        self.delta = float(delta)

    def init_pred(self, y):
        return float(np.median(y))

    def grad_hess(self, y, pred):
        r = pred - y
        abs_r = np.abs(r)
        grad = np.where(abs_r <= self.delta, r, self.delta * np.sign(r))
        # v0.9.109-fix: hess=0 for outliers makes leaves diverge (H→0).
        # XGBoost uses constant hess=1 for Huber (surrogate). We do same.
        hess = np.ones_like(y)
        return grad, hess


class QuantileLoss(Loss):
    """Quantile (pinball) loss. q ∈ (0,1)."""
    name = "quantile"
    task = "regression"

    def __init__(self, quantile: float = 0.5):
        if not (0 < quantile < 1):
            raise ValueError("quantile must be in (0, 1)")
        self.q = float(quantile)

    def init_pred(self, y):
        return float(np.quantile(y, self.q))

    def grad_hess(self, y, pred):
        r = pred - y
        # Pinball grad: q if r>0, -(1-q) if r<0, 0 if r==0
        grad = np.where(r > 0, self.q,
                        np.where(r < 0, -(1.0 - self.q), 0.0))
        hess = np.ones_like(y)
        return grad, hess


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class LogisticLoss(Loss):
    """Binary cross-entropy. `pred` = logit."""
    name = "logistic"
    task = "binary"

    def init_pred(self, y):
        pos = float(np.mean(y))
        pos = max(min(pos, 1 - 1e-6), 1e-6)
        return float(np.log(pos / (1 - pos)))

    def grad_hess(self, y, pred):
        p = _sigmoid(pred)
        grad = p - y
        hess = p * (1 - p)
        return grad, hess

    def predict_from_raw(self, raw):
        return _sigmoid(raw)


class FocalLoss(Loss):
    """Focal loss for imbalanced binary. FL = -α(1-p)^γ y log(p) - ...
    Here: applied to binary cross-entropy with focusing parameter γ.

    Reference: Lin et al. 2017. grad/hess include the (1-p)^γ factor.
    """
    name = "focal"
    task = "binary"

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def init_pred(self, y):
        pos = float(np.mean(y))
        pos = max(min(pos, 1 - 1e-6), 1e-6)
        return float(np.log(pos / (1 - pos)))

    def grad_hess(self, y, pred):
        p = _sigmoid(pred)
        eps = 1e-12
        p_c = np.clip(p, eps, 1 - eps)
        g = self.gamma
        a = self.alpha
        # Simplified focal gradient — full derivation not needed for prototype.
        # Use BCE grad/hess scaled by (1 - p_t)^γ
        p_t = np.where(y == 1, p_c, 1 - p_c)
        alpha_t = np.where(y == 1, a, 1 - a)
        scale = alpha_t * (1 - p_t) ** g
        bce_grad = p - y
        bce_hess = p * (1 - p)
        grad = scale * bce_grad
        hess = scale * bce_hess + 1e-12
        return grad, hess

    def predict_from_raw(self, raw):
        return _sigmoid(raw)


class PoissonLoss(Loss):
    """Poisson regression. `pred` = log(μ). Init: log(y.mean())."""
    name = "poisson"
    task = "count"

    def init_pred(self, y):
        m = float(np.mean(y))
        return float(np.log(max(m, 1e-6)))

    def grad_hess(self, y, pred):
        mu = np.exp(np.clip(pred, -30, 30))
        grad = mu - y
        hess = mu + 1e-12
        return grad, hess

    def predict_from_raw(self, raw):
        return np.exp(np.clip(raw, -30, 30))


class GammaLoss(Loss):
    """Gamma regression (log link). `pred` = log(μ)."""
    name = "gamma"
    task = "count"

    def init_pred(self, y):
        m = float(np.mean(y))
        return float(np.log(max(m, 1e-6)))

    def grad_hess(self, y, pred):
        mu = np.exp(np.clip(pred, -30, 30))
        grad = 1.0 - y / np.maximum(mu, 1e-12)
        hess = y / np.maximum(mu * mu, 1e-12)
        return grad, hess + 1e-12

    def predict_from_raw(self, raw):
        return np.exp(np.clip(raw, -30, 30))


class TweedieLoss(Loss):
    """Tweedie with log link. p ∈ [1, 2) — 1=Poisson, 2=Gamma.
    p=1.5 typical for zero-inflated positive (insurance claims).
    """
    name = "tweedie"
    task = "count"

    def __init__(self, power: float = 1.5):
        if not (1.0 < power < 2.0):
            raise ValueError("tweedie power must be in (1, 2)")
        self.p = float(power)

    def init_pred(self, y):
        m = float(np.mean(y))
        return float(np.log(max(m, 1e-6)))

    def grad_hess(self, y, pred):
        mu = np.exp(np.clip(pred, -30, 30))
        # Simplified Tweedie deviance grad/hess.
        # Full: L = -y·μ^(1-p)/(1-p) + μ^(2-p)/(2-p)
        p = self.p
        grad = mu ** (1 - p) * (mu - y)
        hess = mu ** (1 - 2 * p) * (
            (1 - p) * mu + (2 - p) * (mu - y) * 0
        )  # approx; hess ≈ mu^(2-2p) for zero-inflated
        # Use safe form: hess = mu^(2-2p) for stability
        hess = np.maximum(mu ** (2 - 2 * p), 1e-12) + 1e-12
        return grad, hess

    def predict_from_raw(self, raw):
        return np.exp(np.clip(raw, -30, 30))


LOSSES = {
    "mse": MSELoss,
    "mae": MAELoss,
    "huber": HuberLoss,
    "quantile": QuantileLoss,
    "logistic": LogisticLoss,
    "focal": FocalLoss,
    "poisson": PoissonLoss,
    "gamma": GammaLoss,
    "tweedie": TweedieLoss,
}


def make_loss(name: str, **kwargs) -> Loss:
    if name not in LOSSES:
        raise ValueError(
            f"unknown loss {name!r}; available: {sorted(LOSSES)}")
    return LOSSES[name](**kwargs)
