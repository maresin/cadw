"""Survival analysis primitives (v0.9.155, P1 L10, DEC-056).

- logrank: k-sample log-rank test (Peto & Peto 1972; Mantel-Haenszel)
Consumers: RandomSurvivalForest (I_survival card).

Companion: Breslow partial log-likelihood lives inline in
cadw.ml.coxph (verified, `_breslow` + `_breslow_streaming`). Not
duplicated here — different signature (X-feature matmul).

No Arrow / no I/O; numpy only.
"""
from __future__ import annotations

import numpy as np


def logrank(
    t: np.ndarray,
    e: np.ndarray,
    groups: np.ndarray,
) -> tuple[float, float, int]:
    """Log-rank test for equality of survival curves across k >= 2 groups.

    Standard Mantel-Haenszel form (Peto & Peto 1972):
      At each event time t_i:
        n_i  = number at risk in group (vector over groups)
        d_i  = number of events (scalar across all groups)
        O_i  = observed events per group
        E_i  = d_i * n_i / N_i (expected per group)
        V_i  = d_i * (n_i/N_i)*(1 - n_i/N_i) * (N_i - d_i)/(N_i - 1)
               (per-group variance; for k>2 full k×k covariance below)

    Aggregate:
      O = sum_i O_i         (k,)
      E = sum_i E_i         (k,)
      V = sum_i V_i_matrix  (k, k) with off-diagonal covariance terms
      chi2 = (O - E)^T @ V^+ @ (O - E)   (V^+ = pseudo-inverse)
      dof  = k - 1

    Args:
      t: (n,) non-negative times.
      e: (n,) event indicator, 1 = event, 0 = censored.
      groups: (n,) integer-like group labels (>= 2 distinct).

    Returns:
      (chi2_stat: float, p_value: float, dof: int)

    Raises:
      ValueError: empty input, non-positive times, fewer than 2 groups.

    Complexity: O(n log n) for sort + O(n_groups^2 * n_unique_times).
    For n ~ 1e5 with ~10^2 groups, ~1e9 ops — subsample if needed.
    """
    t = np.asarray(t, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    groups = np.asarray(groups)
    if t.ndim != 1 or e.ndim != 1 or groups.ndim != 1:
        raise ValueError(
            f"t, e, groups must be 1-D; got {t.shape}, {e.shape}, {groups.shape}")
    if not (t.size == e.size == groups.size):
        raise ValueError(
            f"shape mismatch: t={t.size}, e={e.size}, groups={groups.size}")
    if t.size == 0:
        raise ValueError("logrank: empty input")
    if not np.isfinite(t).all() or (t < 0).any():
        raise ValueError("logrank: times must be finite and non-negative")

    unique_groups, g_idx = np.unique(groups, return_inverse=True)
    k = unique_groups.size
    if k < 2:
        raise ValueError(f"logrank: need >= 2 groups; got {k}")

    # Sort by time; handle ties via np.unique on sorted times
    order = np.argsort(t, kind="stable")
    t = t[order]
    e = e[order]
    g_idx = g_idx[order]

    unique_times = np.unique(t)
    n_total = t.size

    O = np.zeros(k, dtype=np.float64)
    E = np.zeros(k, dtype=np.float64)
    V = np.zeros((k, k), dtype=np.float64)

    # at_risk[g] = number in group g with time >= current time
    at_risk = np.zeros(k, dtype=np.float64)
    # pre-compute counts: at each unique time, how many in each group had
    # t >= that time (i.e., at-risk before that time point)
    # Efficient: iterate time descending, add counts of t == unique_times[i]
    # Note: at-risk for event at time t_i = count of {t_j >= t_i}.
    for ti in unique_times:
        # events and censored AT this exact time
        mask_at = t == ti
        d_total = int(e[mask_at].sum())
        if d_total == 0:
            # no events, still update at_risk for later steps? No —
            # since we iterate ascending, at_risk for FUTURE times already
            # includes these rows. Just add current-at-risk counts.
            # Actually at_risk is n_i at TIME ti (i.e., rows with t >= ti
            # whose events/censoring have NOT happened yet — so it's
            # 1 + count of future times per group).
            # Standard formulation: at_risk = sum of {t_j >= ti}.
            continue

        # at-risk = number of rows (across all groups) with t >= ti
        at_risk_mask = t >= ti
        n_g = np.array(
            [int(((g_idx == gi) & at_risk_mask).sum()) for gi in range(k)],
            dtype=np.float64,
        )
        # N = total at risk at ti
        N_i = n_g.sum()
        if N_i <= 0:
            continue

        # observed events per group at ti
        d_g = np.array(
            [int(((g_idx == gi) & mask_at & (e == 1.0)).sum())
             for gi in range(k)],
            dtype=np.float64,
        )
        O += d_g
        E += d_g.sum() * (n_g / N_i)

        # covariance contribution (only if N_i >= 2)
        if N_i > 1:
            d_i = float(d_total)
            # V[g,h] = d_i*(N_i-d_i)/(N_i-1) * (diag(p) - p p^T)
            # where p = n_g / N_i
            p = n_g / N_i
            factor = d_i * (N_i - d_i) / (N_i - 1.0)
            V += factor * (np.diag(p) - np.outer(p, p))

    # chi2 = (O - E)^T V^+ (O - E); use pseudo-inverse for k>=2
    delta = O - E
    try:
        V_inv = np.linalg.pinv(V)
        chi2 = float(delta @ V_inv @ delta)
    except np.linalg.LinAlgError:
        chi2 = float("nan")

    # p-value from chi2 with k-1 dof
    from scipy.stats import chi2 as _chi2_dist
    dof = k - 1
    p_value = float(1.0 - _chi2_dist.cdf(chi2, dof)) if np.isfinite(chi2) else float("nan")
    return chi2, p_value, dof
