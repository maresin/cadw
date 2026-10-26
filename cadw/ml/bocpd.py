"""BOCPD: Bayesian Online Change-Point Detection (Student-t likelihood).

Adams & MacKay (2007). "Bayesian Online Changepoint Detection."
arXiv:0710.3742.

Uses a Student-t predictive likelihood (from a Normal-Inverse-Gamma
conjugate prior) rather than Gaussian. This is critical for robustness
to outliers in industrial sensor data (SKAB changepoint benchmark:
BOCPD-StudentT gets 32% DetRate at 13% FAR; BOCPD-Gaussian typically
0% DR on the same data).

Algorithm:
  Run-length posterior r_t | x_{1:t}, with r in [0, t].
  Hazard rate H = 1/lambda (constant; lambda = expected run length).
  Growth: p(r_t = r_{t-1}+1) = (1 - H) * predictive(x_t | r_{t-1})
  Change: p(r_t = 0) = H * predictive_from_prior(x_t)
  Predictive (Student-t):
    df      = 2 * alpha
    loc     = mu
    scale^2 = beta * (kappa + 1) / (alpha * kappa)
  Posterior update (growth):
    mu'    = (kappa * mu + x) / (kappa + 1)
    kappa' = kappa + 1
    alpha' = alpha + 0.5
    beta'  = beta + 0.5 * kappa * (x - mu)^2 / (kappa + 1)
  Change reset: back to prior (mu0, kappa0, alpha0, beta0).

Anomaly score at t = R[0] = P(r_t = 0 | x_{1:t}) = P(change at t).
  Threshold = 0.5 (mode is change-point). This is the standard
  Adams & MacKay output used by gwgundersen/bocd. (v0.9.72-fix3:
  earlier used P(r < run_length_threshold), which drifted to zero
  threshold on stationary series and over-flagged.)

Reference: gwgundersen/bocd (bocd/studentt.py).

Pure numpy — no sklearn/scipy/torch in cadw/ (AST-guard
test_no_bench_deps_in_core.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class BOCPDResult:
    decision_scores: np.ndarray
    labels: np.ndarray
    threshold: float
    changepoints: np.ndarray
    run_length_posterior: np.ndarray
    argmax_run_length: np.ndarray
    hazard_rate: float
    mu0: float
    kappa0: float
    alpha0: float
    beta0: float
    run_length_threshold: int
    contamination: float | None


def _probe_memory_limit_mb(guard) -> float | None:
    def _is_pos(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0
    MB = ("max_memory_mb", "_max_memory_mb", "limit_mb", "budget_mb",
          "_budget_mb", "max_mb", "_max_mb", "memory_limit_mb",
          "_memory_limit_mb")
    for a in MB:
        v = getattr(guard, a, None)
        if _is_pos(v):
            return float(v)
    BYTES = ("max_memory_bytes", "_max_memory_bytes", "max_bytes",
             "_max_bytes", "limit_bytes", "_limit_bytes", "budget_bytes",
             "_budget_bytes")
    for a in BYTES:
        v = getattr(guard, a, None)
        if _is_pos(v):
            return float(v) / (1024.0 * 1024.0)
    try:
        d = guard.__dict__
    except AttributeError:
        return None
    for k, v in d.items():
        kl = k.lower()
        if _is_pos(v) and "mb" in kl and any(
                t in kl for t in ("mem", "budget", "limit", "max")):
            return float(v)
    return None


def _student_t_logpdf(x: float, df: float, loc: float,
                      scale: float) -> float:
    """Log-pdf of Student-t (scalar; used for prior predictive only)."""
    if df <= 0 or scale <= 0:
        return -np.inf
    z = (x - loc) / scale
    return (
        math.lgamma((df + 1.0) / 2.0)
        - math.lgamma(df / 2.0)
        - 0.5 * math.log(df * math.pi)
        - math.log(scale)
        - (df + 1.0) / 2.0 * math.log1p(z * z / df)
    )


def _posterior_params(x: float, mu: float, kappa: float,
                      alpha: float, beta: float) -> tuple:
    """NIG posterior update with one observation x (scalar reference)."""
    mu_new = (kappa * mu + x) / (kappa + 1.0)
    kappa_new = kappa + 1.0
    alpha_new = alpha + 0.5
    beta_new = beta + 0.5 * kappa * (x - mu) ** 2 / (kappa + 1.0)
    return mu_new, kappa_new, alpha_new, beta_new


def fit_bocpd(
    series: np.ndarray,
    *,
    hazard_rate: float = 1.0 / 250.0,
    mu0: float = 0.0,
    kappa0: float = 1.0,
    alpha0: float = 1.0,
    beta0: float = 1.0,
    run_length_threshold: int = 10,
    contamination: float | None = None,
    max_run: int | None = None,
    burn_in: int | None = None,
    guard=None,
) -> BOCPDResult:
    """BOCPD with Student-t predictive likelihood.

    Parameters
    ----------
    series : np.ndarray, shape (n,)
        1D univariate series.
    hazard_rate : float, default=1/250
        Constant hazard H in (0, 1). Expected run length = 1/H.
    mu0, kappa0, alpha0, beta0 : float
        Normal-Inverse-Gamma prior on segment mean/variance.
        Defaults: mu0=0, kappa0=1, alpha0=1, beta0=1 (weakly informative).
        For non-normalised data, set mu0=median(series),
        beta0=var(series).
    run_length_threshold : int, default=10
        DEPRECATED (v0.9.72-fix3). Kept for backward compatibility;
        no effect on score (score = R[0], P(change)). Must still be >= 1.
    contamination : float | None
        If set, threshold = percentile(1-contamination) on scores.
    max_run : int | None, default=None
        Cap on run-length truncation. None -> min(n, 200).
        v0.9.72-fix1: cap prevents O(n * max_run) Python loop blowup
        (initial v0.9.72 had per-element Python calls, not vectorized).
        Production should not exceed ~500; higher values don't help
        because posterior mass beyond 500 is negligible for
        hazard_rate >= 1/500.
    burn_in : int | None, default=None
        First `burn_in` timestamps get decision_scores = 0 (no data
        yet; posterior mass concentrated at small run-lengths because
        there are not enough observations to establish a long run).
        None -> 2 * run_length_threshold.
        v0.9.72-fix2: without burn-in masking, BOCPD flags the first
        ~200 timestamps as anomalies on any series (bench: 400 false
        positives in [0..207] on n=20000, contamination=0.02).
    guard : MemoryGuard | None

    Returns
    -------
    BOCPDResult
    """
    x = np.asarray(series, dtype=np.float64).ravel()
    n = len(x)
    if n < 10:
        raise ValueError(f"series too short for BOCPD (min 10), got {n}")
    if not (0.0 < hazard_rate < 1.0):
        raise ValueError(
            f"hazard_rate must be in (0, 1), got {hazard_rate}")
    if kappa0 <= 0 or alpha0 <= 0 or beta0 <= 0:
        raise ValueError(
            f"prior params must be positive: kappa0={kappa0}, "
            f"alpha0={alpha0}, beta0={beta0}")
    if run_length_threshold < 1:
        raise ValueError(
            f"run_length_threshold must be >= 1, "
            f"got {run_length_threshold}")

    if max_run is None:
        max_run = min(n, 200)
    max_run = max(2, min(int(max_run), n))

    if burn_in is None:
        burn_in = 2 * run_length_threshold
    burn_in = max(0, min(int(burn_in), n))

    if guard is not None:
        est_bytes = 8 * n * (max_run + 1) * 4
        limit_mb = _probe_memory_limit_mb(guard)
        if limit_mb is not None and limit_mb > 0:
            if est_bytes > limit_mb * 1024 * 1024:
                from cadw.core.memory_guard import MemoryLimitExceeded
                raise MemoryLimitExceeded(
                    f"BOCPD estimated {est_bytes} bytes > "
                    f"guard {limit_mb * 1024 * 1024:.0f} bytes")

    H = hazard_rate

    # v0.9.72-fix1: analytic alpha/df/kappa arrays (never updated).
    # Invariant: alpha[r] = alpha0 + r/2, kappa[r] = kappa0 + r.
    # Only mu and beta need array tracking per step.
    rs = np.arange(max_run + 1, dtype=np.float64)
    df_arr = 2.0 * alpha0 + rs
    kappa_arr = kappa0 + rs
    log_df_pi_arr = 0.5 * np.log(df_arr * math.pi)
    log_gamma_ratio_arr = np.array([
        math.lgamma(df_arr[i] / 2.0 + 0.5) - math.lgamma(df_arr[i] / 2.0)
        for i in range(max_run + 1)
    ])
    # alpha_arr = df_arr / 2, used in scale denominator
    alpha_arr = df_arr / 2.0
    scale_denom_base = alpha_arr * kappa_arr

    # Prior predictive constant (for cp_prob)
    df0 = 2.0 * alpha0
    scale_prior = math.sqrt(beta0 * (kappa0 + 1.0) / (alpha0 * kappa0))
    log_prior_const = (
        math.lgamma(df0 / 2.0 + 0.5) - math.lgamma(df0 / 2.0)
        - 0.5 * math.log(df0 * math.pi) - math.log(scale_prior)
    )

    R = np.zeros(max_run + 1, dtype=np.float64)
    R[0] = 1.0
    mu = np.full(max_run + 1, mu0, dtype=np.float64)
    beta = np.full(max_run + 1, beta0, dtype=np.float64)

    anomaly_score = np.zeros(n, dtype=np.float64)
    argmax_run = np.zeros(n, dtype=np.int64)
    R_history = np.zeros((n, max_run + 1), dtype=np.float64)
    k_ = min(run_length_threshold, max_run + 1)

    for t in range(n):
        xt = float(x[t])

        # Vectorized Student-t log-pdf over all r
        scale = np.sqrt(beta * (kappa_arr + 1.0) / scale_denom_base)
        z = (xt - mu) / scale
        log_pred = (
            log_gamma_ratio_arr - log_df_pi_arr - np.log(scale)
            - (df_arr + 1.0) / 2.0 * np.log1p(z * z / df_arr)
        )
        pred = np.exp(log_pred)

        growth = R * pred * (1.0 - H)

        # Prior predictive (scalar)
        z_prior = (xt - mu0) / scale_prior
        log_prior_pred = (log_prior_const
                          - (df0 + 1.0) / 2.0 * np.log1p(
                              z_prior * z_prior / df0))
        cp_prob = float(R.sum() * math.exp(log_prior_pred) * H)

        # Assemble new R
        R_new = np.empty(max_run + 1, dtype=np.float64)
        R_new[0] = cp_prob
        R_new[1:] = growth[:-1]
        total = R_new.sum()
        if total < 1e-300:
            R_new[0] = 1.0
            total = 1.0
        R = R_new / total

        # Vectorized posterior update (shift; reset r=0 to prior)
        mu_new = np.empty_like(mu)
        beta_new = np.empty_like(beta)
        mu_new[0] = mu0
        beta_new[0] = beta0
        if max_run >= 1:
            kappa_old = kappa_arr[:-1]
            kappa_new_arr = kappa_arr[1:]
            mu_new[1:] = (kappa_old * mu[:-1] + xt) / kappa_new_arr
            beta_new[1:] = (beta[:-1]
                            + 0.5 * kappa_old * (xt - mu[:-1]) ** 2
                            / kappa_new_arr)
        mu = mu_new
        beta = beta_new

        # v0.9.72-fix3: score = P(change-point at t) = R[0].
        # Prior scoring used R[:k].sum() (mass at small run lengths);
        # this drifts toward the percentile threshold because cp_prob
        # accumulates at every step, so R[0] > 0 even in stationary
        # periods. Adams & MacKay and gwgundersen/bocd use R[0]
        # directly with a fixed 0.5 threshold (mode is change).
        # run_length_threshold is deprecated (kept for backward compat;
        # no effect on score).
        anomaly_score[t] = float(R[0])
        argmax_run[t] = int(np.argmax(R))
        R_history[t] = R

    # v0.9.72-fix2: mask burn-in period (no data yet).
    if burn_in > 0:
        anomaly_score[:burn_in] = 0.0

    # v0.9.72-fix3: threshold = max(percentile, 0.5) when contamination
    # set, else 0.5. R[0] is the actual P(change); requiring > 0.5 means
    # "mode is change-point" (Adams & MacKay convention). The
    # max(percentile, 0.5) floor prevents the 400-false-positive
    # explosion on stationary series (v0.9.72-fix2 bug: percentile on
    # drift scores was ~0.001, so everything got flagged).
    if contamination is not None:
        p_thresh = float(np.percentile(
            anomaly_score, 100.0 * (1.0 - contamination)))
        threshold = max(p_thresh, 0.5)
    else:
        threshold = 0.5

    labels = (anomaly_score > threshold).astype(int)
    changepoints = np.flatnonzero(labels)

    return BOCPDResult(
        decision_scores=anomaly_score,
        labels=labels,
        threshold=threshold,
        changepoints=changepoints,
        run_length_posterior=R_history,
        argmax_run_length=argmax_run,
        hazard_rate=H,
        mu0=mu0,
        kappa0=kappa0,
        alpha0=alpha0,
        beta0=beta0,
        run_length_threshold=run_length_threshold,
        contamination=contamination,
    )

