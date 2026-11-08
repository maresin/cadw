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
import pyarrow as pa


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
    # v0.9.87-fix1c: posterior_keep = effective keep size for
    # run_length_posterior (compact mode); None means full
    # (n rows, opt-in via full_posterior=True).
    posterior_keep: int | None = None

    def state_nbytes(self):
        """Persistent output state.

        v0.9.114: 3 n-sized arrays (decision_scores n*f64 + labels n*i64 +
        argmax_run_length n*i64) + run_length_posterior (compact
        posterior_keep*(max_run+1)*f64, ~103KB at defaults
        posterior_keep=64, max_run=200) + 256 header.

        NOT included: `changepoints` (np.flatnonzero(labels), sparse
        subset of labels — already counted) and the R/mu/beta working
        arrays (O(max_run), hyperparameter-bounded).

        full_posterior=True changes run_length_posterior to
        (n, max_run+1) = O(n*max_run) — quadratic; documented
        limitation, default False. bounded_by_domain n<=1e6 applies
        to default compact mode.
        """
        return int(
            self.decision_scores.nbytes
            + self.labels.nbytes
            + self.argmax_run_length.nbytes
            + self.run_length_posterior.nbytes
            + 256
        )


# v0.9.130 (DEC-054): _probe_memory_limit_mb -> 
# cadw.core.memory_guard.probe_limit_mb (single source of truth).
from cadw.core.memory_guard import probe_limit_mb as _probe_memory_limit_mb


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
    df_or_series,
    value_col=None,
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
    full_posterior: bool = False,
    posterior_keep: int = 64,
    batch_size: "int | None" = None,
) -> BOCPDResult:
    """Dual dispatch (v0.9.116): df + value_col | series (legacy).

    df path: fit_bocpd(df, value_col)
    Legacy:  fit_bocpd(series)  # equivalent to previous behaviour
    """
    is_df = (hasattr(df_or_series, "_iter_filtered_batches")
             and hasattr(df_or_series, "_columns"))
    if is_df:
        if value_col is None:
            raise ValueError(
                "fit_bocpd: value_col required when df is passed")
        if value_col not in df_or_series._columns:
            raise KeyError(
                f"fit_bocpd: column {value_col!r} not in {df_or_series._columns}")
        return _fit_bocpd_df(
            df_or_series, value_col,
            hazard_rate=hazard_rate,
            mu0=mu0,
            kappa0=kappa0,
            alpha0=alpha0,
            beta0=beta0,
            run_length_threshold=run_length_threshold,
            contamination=contamination,
            max_run=max_run,
            burn_in=burn_in,
            full_posterior=full_posterior,
            posterior_keep=posterior_keep,
            guard=guard, batch_size=batch_size)
    if value_col is not None:
        raise ValueError(
            "fit_bocpd: value_col only used with df input; omit for np.ndarray")
    return _fit_bocpd_series(
        df_or_series,
        hazard_rate=hazard_rate,
        mu0=mu0,
        kappa0=kappa0,
        alpha0=alpha0,
        beta0=beta0,
        run_length_threshold=run_length_threshold,
        contamination=contamination,
        max_run=max_run,
        burn_in=burn_in,
        full_posterior=full_posterior,
        posterior_keep=posterior_keep,
        guard=guard)


def _fit_bocpd_series(series, *,
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
    full_posterior: bool = False,
    posterior_keep: int = 64,
):
    return _fit_bocpd_series_impl(
        series,
        hazard_rate=hazard_rate,
        mu0=mu0,
        kappa0=kappa0,
        alpha0=alpha0,
        beta0=beta0,
        run_length_threshold=run_length_threshold,
        contamination=contamination,
        max_run=max_run,
        burn_in=burn_in,
        full_posterior=full_posterior,
        posterior_keep=posterior_keep,
        guard=guard)


def _fit_bocpd_df(df, value_col, *,
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
    full_posterior: bool = False,
    posterior_keep: int = 64,
    batch_size=None):
    mg = guard if guard is not None else getattr(df, "_memory_guard", None)
    parts = []
    for b in df._iter_filtered_batches(batch_size=batch_size):
        if mg is not None and hasattr(mg, "check_batch"):
            mg.check_batch(b, name="fit_bocpd.batch", extra_factor=2.0)
        x_b = b.column(value_col).cast(
            pa.float64(), safe=False).to_numpy(zero_copy_only=False)
        if x_b.size > 0:
            parts.append(x_b)
    if not parts:
        raise ValueError(f"fit_bocpd: column {value_col!r} is empty")
    series = np.concatenate(parts) if len(parts) > 1 else parts[0]
    return _fit_bocpd_series(
        series,
        hazard_rate=hazard_rate,
        mu0=mu0,
        kappa0=kappa0,
        alpha0=alpha0,
        beta0=beta0,
        run_length_threshold=run_length_threshold,
        contamination=contamination,
        max_run=max_run,
        burn_in=burn_in,
        full_posterior=full_posterior,
        posterior_keep=posterior_keep,
        guard=mg)


def _fit_bocpd_series_impl(
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
    full_posterior: bool = False,
    posterior_keep: int = 64,
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

    # v0.9.87-fix1c: compact rolling buffer (default) or full
    # posterior (opt-in via full_posterior=True). Audit HIGH: full
    # R_history is 8*n*(max_run+1) bytes -> 1.6 GB @ n=1M,
    # max_run=200. Compact keeps only last `_keep` posteriors in
    # a ring buffer + full-length argmax_run + anomaly_score arrays.
    if full_posterior:
        _keep = n
    else:
        if (isinstance(posterior_keep, bool)
                or not isinstance(posterior_keep, int)
                or posterior_keep < 1):
            raise ValueError(
                f"posterior_keep must be int >= 1, got {posterior_keep!r}")
        _keep = max(1, min(int(posterior_keep), n))

    if guard is not None:
        # v0.9.87-fix1c: est = R_hist + argmax_run + anomaly_score
        # (R_hist dominates when _keep == n; 16*n for the two n-sized
        # 8-byte arrays). Old est 8*n*(max_run+1)*4 overestimated by
        # ~4x, blocking moderately-tight guards even though real
        # usage was lower. Now matches actual.
        est_bytes = 8 * _keep * (max_run + 1) + 16 * n
        if hasattr(guard, "check_state") and callable(guard.check_state):
            guard.check_state(est_bytes, name="bocpd.history")
        else:
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
    # v0.9.87-fix1c: R_hist is (_keep, max_run+1) compact or
    # (n, max_run+1) full — governed by full_posterior.
    R_hist = np.zeros((_keep, max_run + 1), dtype=np.float64)
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
        # v0.9.87-fix1c: ring-buffer write in compact mode
        R_hist[t if full_posterior else t % _keep] = R

    # v0.9.87-fix1c: reorder ring buffer into chronological order.
    # R_hist[t % _keep] = R leaves rows in ring order; reorder so
    # run_length_posterior[i] = R at time (n - _keep + i).
    #   n=10, keep=3: writes at slots 0,1,2,0,1,2,0,1,2,0.
    #   After loop: R_hist[0]=R[9], R_hist[1]=R[7], R_hist[2]=R[8].
    #   Chronological last 3: R[7], R[8], R[9] = slots 1,2,0.
    #   offset = 10 % 3 = 1; vstack([R_hist[1:], R_hist[:1]]) = correct.
    if not full_posterior and _keep < n:
        _offset = n % _keep
        run_length_posterior = np.vstack(
            [R_hist[_offset:], R_hist[:_offset]])
    else:
        run_length_posterior = R_hist

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
        run_length_posterior=run_length_posterior,
        argmax_run_length=argmax_run,
        hazard_rate=H,
        mu0=mu0,
        kappa0=kappa0,
        alpha0=alpha0,
        beta0=beta0,
        run_length_threshold=run_length_threshold,
        contamination=contamination,
        posterior_keep=(None if full_posterior else _keep),
    )

