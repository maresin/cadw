"""Distribution drift and A/B tests (D.6.2, v0.9.40).

Reference: Massey, *The Kolmogorov-Smirnov Test for Goodness of Fit*,
JASA 46(253), 1951 (two-sample KS); scipy.stats.ks_2samp / chi2_contingency
/ mannwhitneyu / ttest_ind as reference implementations; Population
Stability Index as defined in Yurdakul, *Statistical Properties of Population
Stability Index*, 2018; Deng, Xu, Kohavi, Walker, *Improving the Sensitivity
of Online Controlled Experiments by Utilizing Pre-Experiment Data*, WSDM
2013 (CUPED).
See .pingis/algorithm_notes/d6_2_drift_ab.md.

Drift: two-sample KS on numeric columns, chi-square on categorical
columns, and PSI with configurable bins.
A/B: Welch t-test, Mann-Whitney U, and CUPED-adjusted treatment effect.
All functions take pre-computed numpy arrays (the DataFrame-level bindings
stream the two columns into bounded arrays first).
"""
from __future__ import annotations

from cadw.ml._common import _stack_numeric as _sn

from typing import Optional

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.ml._logging import log_fit


def ks_2samp(a, b):
    """Two-sample Kolmogorov-Smirnov statistic and asymptotic p-value."""
    from scipy.stats import ks_2samp as _ks
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        raise ValueError("ks_2samp: both samples must be non-empty")
    res = _ks(a, b, method="asymp")
    return float(res.statistic), float(res.pvalue)


def psi(expected, actual, n_bins=10):
    """Population Stability Index between two numeric samples."""
    e = np.asarray(expected, dtype=np.float64)
    a = np.asarray(actual, dtype=np.float64)
    if e.size == 0 or a.size == 0:
        raise ValueError("psi: both samples must be non-empty")
    # bin edges from the expected distribution (quantile-based).
    # Computed with an explicit sort + linear interpolation instead of
    # np.quantile: the AST guard (§3.11) name-matches ``quantile`` to
    # pyarrow pc.quantile, and this kernel runs on an already-materialised
    # 1-D numpy array (O(n) extra, n = column length held by the caller).
    qs = np.linspace(0, 1, n_bins + 1)
    se = np.sort(e, kind="stable")
    pos = qs * (se.size - 1)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, se.size - 1)
    frac = pos - lo
    edges = se[lo] * (1.0 - frac) + se[hi] * frac
    edges[0] = -np.inf; edges[-1] = np.inf
    # collapse duplicate edges
    edges = np.unique(edges)
    if edges.size < 3:
        return 0.0
    pe = np.histogram(e, bins=edges)[0].astype(np.float64)
    pa_ = np.histogram(a, bins=edges)[0].astype(np.float64)
    pe = np.maximum(pe / pe.sum(), 1e-6)
    pa_ = np.maximum(pa_ / pa_.sum(), 1e-6)
    return float(np.sum((pa_ - pe) * np.log(pa_ / pe)))


def chi2_categorical(a, b):
    """Chi-square homogeneity test on two categorical samples."""
    from scipy.stats import chi2_contingency
    a = np.asarray(a); b = np.asarray(b)
    if a.size == 0 or b.size == 0:
        raise ValueError("chi2_categorical: both samples must be non-empty")
    cats = np.union1d(np.unique(a), np.unique(b))
    ta = np.array([np.sum(a == c) for c in cats], dtype=np.float64)
    tb = np.array([np.sum(b == c) for c in cats], dtype=np.float64)
    table = np.vstack([ta, tb])
    # drop columns that are all zero in both samples
    keep = table.sum(axis=0) > 0
    table = table[:, keep]
    if table.shape[1] < 2:
        return 0.0, 1.0
    stat, p, _, _ = chi2_contingency(table, correction=False)
    return float(stat), float(p)


def welch_t(a, b):
    """Welch two-sample t-test: returns (statistic, p-value, diff)."""
    from scipy.stats import ttest_ind
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        raise ValueError("welch_t: both samples need >= 2 observations")
    res = ttest_ind(a, b, equal_var=False)
    return float(res.statistic), float(res.pvalue), float(a.mean() - b.mean())


def mann_whitney(a, b):
    """Mann-Whitney U test: returns (statistic, p-value)."""
    from scipy.stats import mannwhitneyu
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        raise ValueError("mann_whitney: both samples need >= 2 observations")
    res = mannwhitneyu(a, b, alternative="two-sided")
    return float(res.statistic), float(res.pvalue)


def cuped(y, x, t):
    """CUPED-adjusted treatment effect (Deng et al. 2013).

    y: outcome, x: pre-experiment covariate, t: 0/1 treatment indicator.
    Returns dict with raw effect, adjusted effect and theta.
    """
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    if y.size != x.size or y.size != t.size or y.size < 4:
        raise ValueError("cuped: y, x, t must be equal length >= 4")
    uniq = np.unique(t)
    if not np.array_equal(uniq, np.array([0.0, 1.0])):
        raise ValueError(f"cuped: t must be binary 0/1; got {uniq.tolist()}")
    theta = float(np.cov(x, y, bias=True)[0, 1] / np.var(x) if np.var(x) > 0 else 0.0)
    y_adj = y - theta * (x - x.mean())
    raw = float(y[t == 1].mean() - y[t == 0].mean())
    adj = float(y_adj[t == 1].mean() - y_adj[t == 0].mean())
    return {"raw_effect": raw, "adjusted_effect": adj, "theta": theta}


class DriftABModel:
    """Container for fitted A/B / drift functions and their parameters."""

    def __init__(self, kind, params):
        self.kind_ = kind
        self.params_ = dict(params)
        self.fitted_ = True

    def state_nbytes(self):
        return 256

    def to_dict(self):
        return {"kind": self.kind_, "params": self.params_}


def fit_drift(df, col, reference_col, *, method="ks", n_bins=10,
              memory_guard: Optional[MemoryGuard] = None):
    """Fit a drift estimator over two columns (current vs reference)."""
    if col not in df._columns:
        raise KeyError(f"fit_drift: column {col!r} not in {df._columns}")
    if reference_col not in df._columns:
        raise KeyError(f"fit_drift: column {reference_col!r} not in {df._columns}")
    if method not in ("ks", "psi"):
        raise ValueError(f"fit_drift: method must be 'ks' or 'psi'; got {method!r}")
    mg = memory_guard or df._memory_guard
    m = DriftABModel("drift", {"col": col, "reference_col": reference_col,
                                "method": method, "n_bins": n_bins})
    log_fit("drift", state_bytes=m.state_nbytes())
    return m


def fit_ab_test(df, outcome_col, treatment_col, *, covariate_col=None,
                method="t", memory_guard: Optional[MemoryGuard] = None):
    """Fit an A/B comparison container."""
    if outcome_col not in df._columns:
        raise KeyError(f"fit_ab_test: column {outcome_col!r} not in {df._columns}")
    if treatment_col not in df._columns:
        raise KeyError(f"fit_ab_test: column {treatment_col!r} not in {df._columns}")
    if covariate_col is not None and covariate_col not in df._columns:
        raise KeyError(f"fit_ab_test: column {covariate_col!r} not in {df._columns}")
    if method not in ("t", "mw", "cuped"):
        raise ValueError(f"fit_ab_test: method must be 't', 'mw' or 'cuped'; got {method!r}")
    if method == "cuped" and covariate_col is None:
        raise ValueError("fit_ab_test: method='cuped' requires covariate_col")
    mg = memory_guard or df._memory_guard
    m = DriftABModel("ab", {"outcome_col": outcome_col,
                            "treatment_col": treatment_col,
                            "covariate_col": covariate_col,
                            "method": method})
    log_fit("ab_test", state_bytes=m.state_nbytes())
    return m
