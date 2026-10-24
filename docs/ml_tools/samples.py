"""Deterministic synthetic sample generators for reproduction (M2, v0.9.46).

Each generator mimics the *task structure* of a real benchmark, not the
actual data. Numbers from CADW vs reference on these samples are honest
for "does the model do what the paper claims", not for "matches the
leaderboard number". Real benchmark runs use `--data=<path>` in
scripts/reproduce_benchmark.py (feature planned v0.9.47).
"""
from __future__ import annotations

import numpy as np


def gen_odds_like(n=1000, d=5, contamination=0.05, seed=0):
    """Tabular anomaly: 95% N(0,1), 5% N(3,1). Mirrors ODDS-style setup."""
    rng = np.random.RandomState(seed)
    n_anom = int(n * contamination)
    n_norm = n - n_anom
    X_norm = rng.normal(size=(n_norm, d))
    X_anom = rng.normal(loc=3.0, size=(n_anom, d))
    X = np.vstack([X_norm, X_anom])
    y = np.concatenate([np.zeros(n_norm), np.ones(n_anom)])
    idx = rng.permutation(n)
    return X[idx], y[idx]


def gen_amlb_binary_like(n=1000, d=5, seed=0):
    """Tabular binary: y depends on linear + interaction. Mirrors AMLB."""
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, d))
    logit = 2.0 * X[:, 0] + 1.5 * X[:, 1] * X[:, 2] - 0.5 * X[:, 3]
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(size=n) < p).astype(np.float64)
    return X, y


def gen_amlb_regression_like(n=1000, d=5, seed=0):
    """Tabular regression: y = 2x0 + 1.5*x1*x2 - 0.5*x3 + noise."""
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, d))
    y = (2.0 * X[:, 0] + 1.5 * X[:, 1] * X[:, 2] - 0.5 * X[:, 3]
         + 0.1 * rng.normal(size=n))
    return X, y


def gen_m4_like(n=200, freq=7, seed=0):
    """Univariate TS: trend + weekly seasonality + noise."""
    rng = np.random.RandomState(seed)
    t = np.arange(n)
    trend = 0.01 * t
    seasonal = 1.0 * np.sin(2 * np.pi * t / freq)
    noise = 0.1 * rng.normal(size=n)
    return (trend + seasonal + noise).astype(np.float64)


def gen_survival_like(n=500, d=3, seed=0):
    """Survival: exponential hazards with covariate-dependent rates.

    Returns X (n,d), durations (n,), events (n,) with ~30% censoring.
    """
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, d))
    rate = np.exp(0.5 * X[:, 0] - 0.3 * X[:, 1])
    duration = rng.exponential(scale=1.0 / rate)
    censor_time = rng.exponential(scale=2.0, size=n)
    event = (duration <= censor_time).astype(np.float64)
    observed = np.where(event == 1, duration, censor_time)
    return X, observed, event
