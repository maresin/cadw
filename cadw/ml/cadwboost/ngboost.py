"""NGBoost-style probabilistic prediction (two-stage).

Stage 1: fit mean via MSE GBDT.
Stage 2: fit log(sigma) on log(|y - mu| + eps) via MSE GBDT.

Not true NGBoost (Duan et al 2019, joint natural gradient). Two-stage
gives (mu, sigma) per sample with proper calibration on well-behaved
residuals. For exact joint training: P2 backlog.

Usage:
    m = CADWBoostNGBoost(n_estimators=100)
    m.fit(X, y)
    mu, sigma = m.predict_distribution(X)
    lo, hi = m.predict_interval(X, alpha=0.95)
"""
from __future__ import annotations

import numpy as np

from .model import CADWBoost


def _norm_ppf(p):
    """Inverse normal CDF (Abramowitz & Stegun 26.2.23 approximation)."""
    if p <= 0 or p >= 1:
        raise ValueError(f"p must be in (0, 1), got {p}")
    if p < 0.5:
        return -_norm_ppf(1 - p)
    t = np.sqrt(-2 * np.log(1 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return float(t - (c0 + c1 * t + c2 * t * t)
                 / (1 + d1 * t + d2 * t * t + d3 * t * t * t))


class CADWBoostNGBoost:
    def __init__(
        self, n_estimators: int = 100, max_depth: int = 4,
        learning_rate: float = 0.1, min_sigma: float = 1e-6,
        **boost_kwargs,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_sigma = min_sigma
        self.boost_kwargs = boost_kwargs
        self.mu_model_ = None
        self.sigma_model_ = None
        self.fitted_ = False

    def fit(self, X, y):
        y = np.asarray(y, dtype=np.float64)
        # Stage 1: mean via MSE
        self.mu_model_ = CADWBoost(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            loss="mse", **self.boost_kwargs)
        self.mu_model_.fit(X, y)
        mu = self.mu_model_.predict(X)
        resid = y - mu
        abs_resid = np.abs(resid)

        # Stage 2: sigma via quantile regression on |resid|.
        # q=0.6827: for N(0,sigma), 68.27th percentile of |resid| = sigma.
        self.sigma_model_ = CADWBoost(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            loss="quantile",
            loss_kwargs={"quantile": 0.6827},
            **self.boost_kwargs)
        self.sigma_model_.fit(X, abs_resid)

        # Global calibration: scale so mean(sigma_hat) matches theoretical
        # scale. For N(0,1), E[|Z|] = sqrt(2/pi) ≈ 0.7979.
        raw_sigma = np.maximum(self.sigma_model_.predict(X), self.min_sigma)
        target_mean_sigma = float(np.mean(abs_resid) / 0.7979)
        actual_mean_sigma = float(np.mean(raw_sigma))
        self.sigma_scale_ = (
            target_mean_sigma / actual_mean_sigma
            if actual_mean_sigma > 1e-12 else 1.0)

        self.fitted_ = True
        return self

    def predict(self, X):
        if not self.fitted_:
            raise RuntimeError("not fitted")
        return self.mu_model_.predict(X)

    def predict_distribution(self, X):
        if not self.fitted_:
            raise RuntimeError("not fitted")
        mu = self.mu_model_.predict(X)
        raw_sigma = self.sigma_model_.predict(X)
        sigma = np.maximum(raw_sigma * self.sigma_scale_, self.min_sigma)
        return mu, sigma

    def predict_interval(self, X, alpha: float = 0.95):
        mu, sigma = self.predict_distribution(X)
        z = _norm_ppf((1 + alpha) / 2)
        return mu - z * sigma, mu + z * sigma

    def sample(self, X, n_samples: int = 100, random_state: int = 0):
        mu, sigma = self.predict_distribution(X)
        rng = np.random.RandomState(random_state)
        return rng.normal(mu[:, None], sigma[:, None],
                          size=(mu.shape[0], n_samples))

    def state_nbytes(self):
        return (self.mu_model_.state_nbytes()
                + self.sigma_model_.state_nbytes())
