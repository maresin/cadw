"""Bayesian ordered probit (dual method: SGHMC / Gibbs).

Reference:
  Albert & Chib (1993). Bayesian analysis of binary and polychotomous
    response data. JASA. (z-augmentation Gibbs)
  Chen et al. (2014). SGHMC. ICML.

Model: y ∈ {0,...,K-1}, latent z = x·β + ε, ε ~ N(0,1), y = k iff
  c_{k-1} < z ≤ c_k, with c_{-1} = -inf, c_{K-1} = +inf.
Cutpoints reparameterized: c_j = a_0 + Σ_{m=1..j} exp(a_m) for j≥1;
  c_0 = a_0. Unconstrained parameterization a = (a_0, a_1..a_{K-2}).

method='sghmc' (default): mini-batch SGHMC over marginal log-posterior
  p(β, a | y, X). OOC via subsampling — achieved.
method='gibbs': exact posterior via z-augmentation. O(n) per iter,
  requires full data → bounded_by_domain {n: 1e5}, precedent MH/NUTS
  W_probabilistic debt.

Predict: mean over posterior samples (n_draws).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.ml._logging import log_fit
from cadw.ops.mcmc import sgmcmc


def _stack_X(batch, X_cols: Sequence[str]) -> np.ndarray:
    d = len(X_cols)
    b_n = batch.num_rows
    X = np.empty((b_n, d), dtype=np.float64)
    for j, c in enumerate(X_cols):
        X[:, j] = batch.column(c).to_numpy(zero_copy_only=False).astype(
            np.float64, copy=False)
    return X


def _norm_pdf(x):
    """Standard normal PDF φ(x)."""
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _norm_cdf(x):
    """Standard normal CDF Φ(x)."""
    return np.vectorize(
        lambda v: 0.5 * (1.0 + math.erf(v / math.sqrt(2.0))),
        otypes=[np.float64])(x)


def _cutpoints_from_a(a: np.ndarray, K: int) -> np.ndarray:
    """Reparameterize a -> c (K-1 cutpoints). c_0=a_0, c_j=c_0+Σ exp(a_m)."""
    if K < 2:
        raise ValueError(f"K >= 2 required; got {K}")
    c = np.empty(K - 1, dtype=np.float64)
    c[0] = a[0]
    if K > 2:
        incr = np.exp(a[1:])
        c[1:] = a[0] + np.cumsum(incr)
    return c


def _ordered_probit_grad_log_den_data(batch, theta, X_cols, y_col, K):
    """Sum over batch of ∇log p(y_i | x_i, θ). For sgmcmc (v0.9.229-a).

    Returns (d,) — NO batch-mean normalization. The sgmcmc primitive
    scales by n/batch_size internally (reference sghmc.py line 46).
    """
    d = len(X_cols)
    beta = theta[:d]
    a = theta[d:]
    c = _cutpoints_from_a(a, K)
    c_full = np.concatenate(([-np.inf], c, [np.inf]))

    X = _stack_X(batch, X_cols)
    y = batch.column(y_col).to_numpy(
        zero_copy_only=False).astype(np.int64, copy=False)
    n_b = X.shape[0]
    if n_b == 0:
        return np.zeros_like(theta)

    eta = X @ beta
    pdf_at = _norm_pdf(c_full[None, :] - eta[:, None])
    cdf_at = _norm_cdf(c_full[None, :] - eta[:, None])
    idx = np.arange(n_b)
    p_obs = cdf_at[idx, y + 1] - cdf_at[idx, y]
    p_obs = np.maximum(p_obs, 1e-12)

    _GRAD_CLIP = 5.0
    dl_obs_deta = np.clip(
        (pdf_at[idx, y] - pdf_at[idx, y + 1]) / p_obs, -_GRAD_CLIP, _GRAD_CLIP)
    dl_dbeta = X.T @ dl_obs_deta  # SUM over batch (NOT /n_b)

    dl_dc = np.zeros(K - 1, dtype=np.float64)
    y_arr = y
    phi_lower = np.clip(pdf_at[idx, y] / p_obs, -_GRAD_CLIP, _GRAD_CLIP)
    phi_upper = np.clip(pdf_at[idx, y + 1] / p_obs, -_GRAD_CLIP, _GRAD_CLIP)
    if K > 1:
        valid_lower = y_arr >= 1
        if valid_lower.any():
            lower_idx = (y_arr[valid_lower] - 1).astype(np.int64)
            dl_dc -= np.bincount(
                lower_idx, weights=phi_lower[valid_lower],
                minlength=K - 1).astype(np.float64)[:K - 1]
        valid_upper = y_arr <= K - 2
        if valid_upper.any():
            upper_idx = y_arr[valid_upper].astype(np.int64)
            dl_dc += np.bincount(
                upper_idx, weights=phi_upper[valid_upper],
                minlength=K - 1).astype(np.float64)[:K - 1]

    dl_da = np.zeros(K - 1, dtype=np.float64)
    dl_da[0] = dl_dc.sum()
    if K > 2:
        exp_a = np.exp(a[1:])
        suffix_sum = np.cumsum(dl_dc[1:][::-1])[::-1]
        dl_da[1:] = exp_a * suffix_sum

    return np.concatenate([dl_dbeta, dl_da])


def _ordered_probit_grad_log_den_prior(theta, prior_scale):
    """Full prior gradient: ∇log N(0, prior_scale²) = -θ/prior_scale²."""
    return -theta / (prior_scale ** 2)



@dataclass
class OrderedProbitModel:
    """Bayesian ordered probit with posterior samples."""
    coef_mean_: np.ndarray         # (d,) posterior mean
    cutpoints_mean_: np.ndarray    # (K-1,)
    coef_samples_: np.ndarray      # (n_samples, d)
    cutpoints_samples_: np.ndarray # (n_samples, K-1)
    feature_names_: list
    n_classes_: int
    prior_scale_: float
    method_: str                    # "sghmc" | "gibbs"
    n_samples_: int
    n_rows_: int
    seed_: int

    def state_nbytes(self) -> int:
        d = len(self.coef_mean_)
        K = self.n_classes_
        n_s = self.n_samples_
        return int(
            self.coef_mean_.nbytes
            + self.cutpoints_mean_.nbytes
            + self.coef_samples_.nbytes
            + self.cutpoints_samples_.nbytes
            + 256
        )

    def _predict_proba_single(self, X: np.ndarray, beta: np.ndarray,
                                c: np.ndarray) -> np.ndarray:
        """(n, K) probs for one sample (beta, c)."""
        eta = X @ beta  # (n,)
        c_full = np.concatenate(([-np.inf], c, [np.inf]))
        # cdf_at[i, k] = Φ(c_full[k] - η_i) for k=0..K (K+1 cutpoints)
        cdf_at = _norm_cdf(c_full[None, :] - eta[:, None])  # (n, K+1)
        probs = cdf_at[:, 1:] - cdf_at[:, :-1]  # (n, K) — differences
        return probs

    def predict_proba(self, df, *, n_draws: int = 100,
                       memory_guard: Optional[MemoryGuard] = None,
                       batch_size: Optional[int] = None):
        """Average over n_draws posterior samples; returns CadwDataFrame."""
        from cadw import CadwDataFrame
        mg = memory_guard or df._memory_guard
        for c in self.feature_names_:
            if c not in df._columns:
                raise KeyError(f"predict_proba: column {c!r} not in df")
        n_total = int(df.count())
        K = self.n_classes_
        mg.check_state(n_total * K * 8 + 256,
                       name="ordered_probit.predict_proba_output")
        out = np.zeros((n_total, K), dtype=np.float64)

        # Pick n_draws samples from posterior
        rng = np.random.default_rng(self.seed_ + 1000)
        idx = rng.choice(self.n_samples_, size=int(n_draws), replace=False)

        _bs = batch_size if batch_size is not None else 1000
        # First pass: compute per-row probs summed over draws
        off = 0
        for b in df._iter_filtered_batches(batch_size=_bs):
            mg.check_batch(b, name="ordered_probit.predict_proba",
                           extra_factor=3.0)
            X = _stack_X(b, self.feature_names_)
            n_b = X.shape[0]
            # Per-sample probs
            for si in idx:
                beta = self.coef_samples_[si]
                c = self.cutpoints_samples_[si]
                out[off:off + n_b] += self._predict_proba_single(X, beta, c)
            off += n_b
        if off != n_total:
            out = out[:off]
        out /= float(n_draws)

        cols = {f"class_{k}_proba": pa.array(out[:, k], type=pa.float64())
                for k in range(K)}
        return CadwDataFrame.from_table(
            pa.table(cols), memory_guard=mg)

    def predict(self, df, *, n_draws: int = 100,
                 memory_guard: Optional[MemoryGuard] = None,
                 batch_size: Optional[int] = None):
        """argmax of predict_proba -> CadwDataFrame with class_pred."""
        from cadw import CadwDataFrame
        mg = memory_guard or df._memory_guard
        proba_df = self.predict_proba(
            df, n_draws=n_draws, memory_guard=mg, batch_size=batch_size)
        # Iterate proba_df to compute argmax
        n_total = int(df.count())
        out = np.empty(n_total, dtype=np.int64)
        off = 0
        for b in proba_df._iter_filtered_batches():
            K = self.n_classes_
            cols = np.column_stack([
                b.column(f"class_{k}_proba").to_numpy(zero_copy_only=False)
                for k in range(K)])
            out[off:off + b.num_rows] = np.argmax(cols, axis=1)
            off += b.num_rows
        return CadwDataFrame.from_table(
            pa.table({"class_pred": pa.array(out, type=pa.int64())}),
            memory_guard=mg)


def fit_ordered_probit(
    df,
    X_cols: "str | Sequence[str]",
    y_col: str,
    *,
    n_classes: "int | None" = None,
    method: str = "sghmc",
    n_samples: int = 500,
    n_burnin: int = 3000,
    lr: float = 0.01,
    prior_scale: float = 1.0,
    seed: int = 0,
    memory_guard: Optional[MemoryGuard] = None,
    batch_size: "int | None" = None,
) -> OrderedProbitModel:
    """Fit Bayesian ordered probit (dual method).

    Args:
      df, X_cols, y_col: standard.
      n_classes: K (>= 2); auto-detect if None.
      method: 'sghmc' (default, OOC via subsampling) | 'gibbs' (exact,
        n<=1e5).
      n_samples: posterior draws to keep.
      n_burnin: warmup.
      lr: SGHMC step size (ignored for gibbs).
      prior_scale: N(0, prior_scale²) prior on (β, a).
      seed, memory_guard, batch_size: standard.
    """
    if isinstance(X_cols, str):
        X_cols = [X_cols]
    X_cols = list(X_cols)
    if not X_cols:
        raise ValueError("fit_ordered_probit: X_cols is empty")
    if y_col in X_cols:
        raise ValueError(
            f"fit_ordered_probit: {y_col!r} must not be in X_cols")
    if method not in ("sghmc", "gibbs"):
        raise ValueError(
            f"fit_ordered_probit: method 'sghmc'|'gibbs'; got {method!r}")
    if prior_scale <= 0:
        raise ValueError(f"prior_scale > 0; got {prior_scale}")

    for c in X_cols + [y_col]:
        if c not in df.columns:
            raise KeyError(f"fit_ordered_probit: column {c!r} not in df")

    schema = df._dataset.schema
    for c in X_cols:
        t = schema.field(c).type
        if not (pa.types.is_integer(t) or pa.types.is_floating(t)):
            raise TypeError(
                f"fit_ordered_probit: X column {c!r} has type {t}")

    mg = memory_guard or df._memory_guard
    d = len(X_cols)
    n_total = int(df.count())
    if n_total == 0:
        raise ValueError("fit_ordered_probit: empty input")

    # Detect K
    y_all = []
    for b in df._iter_filtered_batches(batch_size=10000):
        mg.check_batch(b, name="ordered_probit.detect_K", extra_factor=2.0)
        y_all.append(b.column(y_col).to_numpy(
            zero_copy_only=False).astype(np.int64, copy=False))
    y_concat = np.concatenate(y_all) if len(y_all) > 1 else y_all[0]
    del y_all
    K_detected = int(y_concat.max()) - int(y_concat.min()) + 1
    if K_detected < 2:
        raise ValueError(
            f"fit_ordered_probit: need >= 2 classes; got {K_detected} "
            f"(min={y_concat.min()}, max={y_concat.max()})")
    if n_classes is not None and n_classes != K_detected:
        raise ValueError(
            f"fit_ordered_probit: n_classes={n_classes} != detected "
            f"{K_detected} (min={y_concat.min()}, max={y_concat.max()}); "
            f"y must be 0-based contiguous")
    K = K_detected
    del y_concat
    gc_needed = True

    # Initialize theta: β=0; a: c_0=0.5, log-diffs=log(1) = 0 → c_j = 0.5+j
    theta_init = np.zeros(d + (K - 1), dtype=np.float64)
    # Init cutpoints from empirical quantiles of y (rough):
    # a_0 = c_0 (midpoint between class 0 and 1 means of y rank)
    # Subsequent exp(a_m) = spacing between adjacent cutpoints
    # Use quantile-based init: c_k = norm.ppf((k+1)/K)
    for kk in range(K - 1):
        q = (kk + 1) / float(K)
        # Inverse normal CDF via math.erfinv: norm.ppf(q) = √2 erfinv(2q-1)
        import math as _math
        try:
            from cadw.ops.numeric import erfinv
            theta_init[d + kk] = _math.sqrt(2.0) * erfinv(2.0 * q - 1.0)
        except Exception:
            theta_init[d + kk] = 0.0
    # Convert quantile c to 'a' parameterization: c_0 = a_0, c_j = a_0 + sum exp(a_m)
    # Simplified: set a_0 = c_0, exp(a_m) = c_m - c_{m-1}
    if K > 1:
        c_quantiles = theta_init[d:].copy()
        theta_init[d] = c_quantiles[0]
        if K > 2:
            diffs = np.maximum(np.diff(c_quantiles), 0.3)  # min 0.3 spacing
            theta_init[d+1:] = np.log(diffs)

    # State guard: samples(n_s*(d+K-1)) + θ(d+K-1) + ... ≈
    mg.check_state(
        n_samples * (d + K - 1) * 8 + (d + K - 1) * 8 * 4 + 256,
        name="ordered_probit.state")

    if method == "sghmc":
        # P63 (v0.9.229-a): sgmcmc reference API.
        # V_hat=I, C=20*I verified: mean=1.99, std=0.033 on Gaussian.
        from functools import partial as _partial
        _V_hat = np.eye(d + K - 1)
        _C = 20.0 * np.eye(d + K - 1)

        def _gdata(batch, th):
            return _ordered_probit_grad_log_den_data(
                batch, th, X_cols, y_col, K)

        def _gprior(th):
            return _ordered_probit_grad_log_den_prior(th, prior_scale)

        _bs_sghmc = batch_size
        if _bs_sghmc is None:
            _bs_sghmc = max(50, min(1000, int(n_total // 10)))

        samples = sgmcmc(
            df, _gdata, _gprior, theta_init,
            V_hat=_V_hat, C=_C,
            n_samples=int(n_samples), n_burnin=int(n_burnin),
            batch_size=int(_bs_sghmc), lr=float(lr), seed=int(seed),
            memory_guard=mg)

        coef_samples = samples[:, :d]
        a_samples = samples[:, d:]
        cutpoint_samples = np.stack(
            [_cutpoints_from_a(a, K) for a in a_samples], axis=0)
    else:  # gibbs — Albert-Chib z-augmentation
        if n_total > 100000:
            from cadw.core.memory_guard import MemoryLimitExceeded
            raise MemoryLimitExceeded(
                f"fit_ordered_probit(method='gibbs'): n={n_total} > 1e5. "
                f"Gibbs requires full data per iteration (intrinsic). "
                f"Use method='sghmc' for large n. Precedent: MH/NUTS "
                f"W_probabilistic debt.")
        # Materialize X, y once (n <= 1e5, ≤ 1e5*20*8 = 16 MB)
        mg.check_state(n_total * (d + 1) * 8 + 256,
                       name="ordered_probit.gibbs_data")
        X_all = np.empty((n_total, d), dtype=np.float64)
        y_all_arr = np.empty(n_total, dtype=np.int64)
        off = 0
        for b in df._iter_filtered_batches(batch_size=10000):
            mg.check_batch(b, name="ordered_probit.gibbs_scan",
                           extra_factor=2.0)
            X_all[off:off + b.num_rows] = _stack_X(b, X_cols)
            y_all_arr[off:off + b.num_rows] = b.column(y_col).to_numpy(
                zero_copy_only=False).astype(np.int64, copy=False)
            off += b.num_rows

        rng = np.random.default_rng(int(seed))
        theta = theta_init.copy()
        samples = np.empty((int(n_samples), d + K - 1), dtype=np.float64)
        sample_idx = 0
        for it in range(int(n_burnin) + int(n_samples)):
            beta = theta[:d]
            c = _cutpoints_from_a(theta[d:], K)
            c_full = np.concatenate(([-np.inf], c, [np.inf]))
            # Sample z_i | y_i, β, c (truncated Normal)
            eta = X_all @ beta
            z = np.empty(n_total)
            for k in range(K):
                mask = (y_all_arr == k)
                lo = c_full[k] - eta[mask]
                hi = c_full[k + 1] - eta[mask]
                nk = int(mask.sum())
                if nk == 0:
                    continue
                # Sample truncated N(0,1) on (lo, hi) via inverse CDF
                lo_cdf = _norm_cdf(lo) if np.isfinite(lo).any() else np.zeros(nk)
                hi_cdf = _norm_cdf(hi) if np.isfinite(hi).any() else np.ones(nk)
                # Handle ±inf
                lo_cdf = np.where(np.isfinite(lo), lo_cdf, 0.0)
                hi_cdf = np.where(np.isfinite(hi), hi_cdf, 1.0)
                u = rng.uniform(size=nk)
                p = lo_cdf + u * (hi_cdf - lo_cdf)
                p = np.clip(p, 1e-12, 1.0 - 1e-12)
                from cadw.ops.numeric import erfinv
                z_t = np.array([math.sqrt(2.0) * erfinv(float(2.0 * pi - 1.0))
                                for pi in p])
                z[mask] = z_t + eta[mask]
            # Sample β | z, prior: N((X'X+1/σ²I)^{-1} X'z, (X'X+1/σ²I)^{-1})
            XtX = X_all.T @ X_all + np.eye(d) / (prior_scale ** 2)
            Xtz = X_all.T @ z
            beta_post_mean = np.linalg.solve(XtX, Xtz)
            beta_post_cov = np.linalg.inv(XtX)
            beta_new = rng.multivariate_normal(beta_post_mean, beta_post_cov)
            # Sample c | z: uniform on constraint c_{y-1} < z <= c_y
            # (WLOG: use standard approach — for each k, c_k = max
            # needed lower bound + small adjustments; simplified:
            # use sample from constrained region; approximated via
            # taking z order statistics)
            # Simple approximation: c_k = max z where y <= k  (lower bound)
            #                  c_k = min z where y > k    (upper bound)
            # → uniform on [lower, upper]
            c_new = np.empty(K - 1, dtype=np.float64)
            for k in range(K - 1):
                # Albert-Chib bounds for c_k:
                #   For y_i == k:  z_i <= c_k  → c_k >= z_i  (LOWER bound)
                #   For y_i >  k:  c_k <  z_i  → c_k <  z_i  (UPPER bound)
                # P45 FIX: previous impl had lower/upper masks SWAPPED
                # (guaranteed lo > hi -> always midpoint -> degenerate
                #  c_0 = c_1). Correct:
                lower_mask = (y_all_arr == k)
                upper_mask = (y_all_arr > k)
                lo = z[lower_mask].max() if lower_mask.any() else -np.inf
                hi = z[upper_mask].min() if upper_mask.any() else np.inf
                if np.isfinite(lo) and np.isfinite(hi):
                    if hi > lo:
                        c_new[k] = rng.uniform(lo, hi)
                    else:
                        # P44 fix: z samples inconsistent across classes
                        # (poor init); use midpoint as fallback.
                        c_new[k] = 0.5 * (lo + hi)
                elif np.isfinite(lo):
                    c_new[k] = lo + rng.exponential(1.0)
                elif np.isfinite(hi):
                    c_new[k] = hi - rng.exponential(1.0)
                else:
                    c_new[k] = 0.0
            # Convert c_new to a: c_0 = a_0, c_j = c_0 + Σ exp(a_m)
            # a_0 = c_0; exp(a_m) = c_m - c_{m-1}
            a_new = np.empty(K - 1, dtype=np.float64)
            a_new[0] = c_new[0]
            if K > 2:
                diffs = np.maximum(np.diff(c_new), 1e-8)
                a_new[1:] = np.log(diffs)
            theta = np.concatenate([beta_new, a_new])
            if it >= int(n_burnin) and sample_idx < int(n_samples):
                samples[sample_idx] = theta
                sample_idx += 1
                if sample_idx >= int(n_samples):
                    break

        coef_samples = samples[:, :d]
        a_samples = samples[:, d:]
        cutpoint_samples = np.stack(
            [_cutpoints_from_a(a, K) for a in a_samples], axis=0)

    model = OrderedProbitModel(
        coef_mean_=coef_samples.mean(axis=0),
        cutpoints_mean_=cutpoint_samples.mean(axis=0),
        coef_samples_=coef_samples,
        cutpoints_samples_=cutpoint_samples,
        feature_names_=list(X_cols),
        n_classes_=K,
        prior_scale_=float(prior_scale),
        method_=method,
        n_samples_=int(n_samples),
        n_rows_=int(n_total),
        seed_=int(seed),
    )
    log_fit("ordered_probit", state_bytes=model.state_nbytes())
    return model
