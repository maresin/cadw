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


def gen_recsys_like(n_users=200, n_items=50, n_factors=5, density=0.10,
                     seed=0):
    """Implicit-feedback recsys sample: low-rank user/item factors.

    Returns (users, items) arrays — interaction pairs at ~`density` fill.
    Positive interaction iff latent dot product > 0.
    """
    rng = np.random.RandomState(seed)
    U = rng.normal(size=(n_users, n_factors))
    V = rng.normal(size=(n_items, n_factors))
    affinity = U @ V.T  # (n_users, n_items)
    # v0.9.46-fix2: add per-user noise to break the systematic top-k bias.
    # Without it, items with small ||V[i]|| are never in any user's top-k,
    # so after 20% per-user holdout they vanish from the train set entirely
    # (CADW item_factors shape=(42, 16) instead of 50). Real recsys data is
    # also skewed, but not this systematically — noise restores coverage.
    # Noise scale 0.7 ~ std of affinity entries (V norm ~sqrt(k)=2.24, U ~1,
    # so single dot-product ~N(0, k) with std sqrt(k)=2.24; 0.7 breaks bias
    # without dominating signal).
    users, items = [], []
    k = max(1, int(n_items * density))
    for u in range(n_users):
        noisy = affinity[u] + 0.7 * rng.normal(size=n_items)
        top = np.argsort(-noisy)[:k]
        users.extend([u] * len(top))
        items.extend(top.tolist())
    return np.array(users, dtype=np.int64), np.array(items, dtype=np.int64)


# ── v0.9.52: TS anomaly DGPs ──────────────────────────────────────

def gen_ts_point_anomaly(n: int = 1000, n_spikes: int = 20,
                         seed: int = 0) -> tuple:
    """Sine + isolated spikes at random positions.

    Returns
    -------
    x : np.ndarray, shape (n,)
    spike_positions : np.ndarray, shape (n_spikes,)
        Indices of injected anomalies.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x = np.sin(2 * np.pi * t / 100.0) + 0.1 * rng.standard_normal(n)
    spike_pos = rng.choice(n, size=n_spikes, replace=False)
    x[spike_pos] += 5.0
    return x, spike_pos


def gen_ts_shape_anomaly(n: int = 1000, n_segments: int = 3,
                         seg_len: int = 30, seed: int = 0) -> tuple:
    """Sine baseline with square-wave segments injected as shape anomalies.

    Returns
    -------
    x : np.ndarray, shape (n,)
    anomaly_positions : np.ndarray
        All indices belonging to injected segments.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x = np.sin(2 * np.pi * t / 100.0) + 0.1 * rng.standard_normal(n)
    starts = rng.choice(n - seg_len, size=n_segments, replace=False)
    anomaly_positions = []
    for s in starts:
        lt = np.arange(seg_len)
        x[s:s + seg_len] = np.sign(np.sin(2 * np.pi * lt / 20.0)) * 1.0
        anomaly_positions.extend(range(s, s + seg_len))
    return x, np.array(sorted(anomaly_positions))


def gen_multivariate_outliers(n: int = 5000, n_channels: int = 3,
                              n_outliers: int = 50, seed: int = 0) -> tuple:
    """Multi-channel Gaussian process + scattered multivariate outliers.

    Baseline: n_channels independent sine waves with phase offsets
    (correlated across channels in time); outliers are injected as
    amplitude/phase changes in *all* channels simultaneously.

    Returns
    -------
    X : np.ndarray, shape (n, n_channels)
    outlier_positions : np.ndarray, shape (n_outliers,)
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    X = np.empty((n, n_channels), dtype=np.float64)
    for ch in range(n_channels):
        period = 100.0 + 20.0 * ch
        phase = 0.5 * ch
        X[:, ch] = np.sin(2 * np.pi * t / period + phase) \
            + 0.1 * rng.standard_normal(n)
    pos = rng.choice(n, size=n_outliers, replace=False)
    for p in pos:
        # amplitude burst in all channels, opposite phase
        X[p] = -3.0 * X[p]
    return X, pos


def gen_ts_changepoint(n: int = 5000, n_channels: int = 3,
                       n_changes: int = 3, seed: int = 0) -> tuple:
    """Multi-channel sine with mean shifts at random changepoint positions.

    Returns
    -------
    X : np.ndarray, shape (n, n_channels)
    changepoints : np.ndarray, shape (n_changes,)
        Indices where the mean level shifts.
    anomaly_positions : np.ndarray
        All timestamps in anomaly zone (after each changepoint until
        the next baseline segment ends). For SKAB-style scoring, the
        post-changepoint region is flagged anomalous.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    X = np.empty((n, n_channels), dtype=np.float64)
    for ch in range(n_channels):
        period = 100.0 + 20.0 * ch
        X[:, ch] = np.sin(2 * np.pi * t / period) + 0.1 * rng.standard_normal(n)
    cps = np.sort(rng.choice(n - 500, size=n_changes, replace=False)) + 500
    pos = []
    shift_per_ch = rng.choice([-2.5, -1.5, 1.5, 2.5], size=n_channels)
    for cp in cps:
        end = min(cp + 300, n)
        X[cp:end] += shift_per_ch  # mean shift in all channels
        pos.extend(range(cp, end))
    return X, cps, np.array(sorted(pos))


def gen_ts_shape_anomaly_multi(n: int = 5000, n_channels: int = 3,
                                n_segments: int = 5, seg_len: int = 50,
                                anomaly_noise: float = 3.0,
                                seed: int = 0) -> tuple:
    """Multi-channel shape anomaly via cross-channel correlation break.

    Normal:
      x1 = sin(2*pi*t/100) + 0.1*N(0,1)
      x2 = cos(2*pi*t/100) + 0.1*N(0,1)
      x3 = x1 - x2 + 0.1*N(0,1)   (x3 lies on the plane x3 = x1 - x2)

    Anomaly segment [s:s+seg_len]:
      x3 = (x1 - x2) + anomaly_noise * N(0,1)   (off-plane; structural break)

    Univariate detector on x3 sees only elevated noise; multivariate/
    cross-channel detector (EIF with windows, STRAY on per-row vectors)
    sees the deviation from the x3 = x1 - x2 plane.

    Returns
    -------
    X : np.ndarray, shape (n, n_channels)
    anomaly_positions : np.ndarray, shape (n_segments * seg_len,)
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x1 = np.sin(2 * np.pi * t / 100.0) + 0.1 * rng.standard_normal(n)
    x2 = np.cos(2 * np.pi * t / 100.0) + 0.1 * rng.standard_normal(n)
    x3 = x1 - x2 + 0.1 * rng.standard_normal(n)

    starts = rng.choice(n - seg_len - 10, size=n_segments,
                        replace=False) + 5
    pos = []
    for s in starts:
        x3[s:s + seg_len] = ((x1[s:s + seg_len] - x2[s:s + seg_len])
                             + anomaly_noise * rng.standard_normal(seg_len))
        pos.extend(range(s, s + seg_len))
    X = np.stack([x1, x2, x3], axis=1)
    return X, np.array(sorted(pos))
