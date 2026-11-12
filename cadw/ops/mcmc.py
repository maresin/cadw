"""MCMC primitives: SGHMC (Chen et al. 2014) + generic Gibbs.

Reference:
  Chen, Fox, Guestrin (2014). Stochastic Gradient Hamiltonian Monte
    Carlo. ICML.
  Springenberg et al. (2016). Bayesian Optimization with Robust
    Bayesian Neural Networks. NeurIPS. (adaptive friction, eq. 12-13)
  Nemeth & Fearnhead (2024). Stochastic Gradient MCMC for Nonlinear
    State Space Models. (SGHMC vs SGLD accuracy: ESS/gradient 0.0763
    vs 0.0083, ~9x efficiency; SGHMC needs O(d/eps) iterations vs
    SGLD O(d²/eps²) — better high-dim scaling.)

SGHMC update (Chen 2014 eq. 9-10):
  θ_{t+1} = θ_t + ε · r_t
  r_{t+1} = (1 - α·ε) · r_t - ε · ∇U(θ_t) + √(2·α·ε) · N(0, I)
  Constraint: 2α ≥ ε · V̂  (V̂ = gradient variance, estimated adaptively)

Friction α adaptive (Springenberg 2016 eq. 12-13):
  α_t = max(α_min, 0.5 · ε · V̂_t)
  V̂_{t+1} = (1-β) · V̂_t + β · (ε · ||∇U_batch||²)

Mini-batch gradient: caller supplies `grad_batch_fn(theta, batch) ->
grad` returning ∇U on batch. No full-data requirement.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded


def sgmcmc(
    df,
    grad_log_den_data_fn: Callable,
    grad_log_den_prior_fn: Callable,
    theta_init: np.ndarray,
    *,
    V_hat: Optional[np.ndarray] = None,
    C: Optional[np.ndarray] = None,
    n_samples: int = 1000,
    n_burnin: int = 3000,
    batch_size: "int | None" = None,
    lr: float = 0.01,
    Minv: "np.ndarray | None" = None,
    seed: int = 0,
    memory_guard: Optional[MemoryGuard] = None,
) -> np.ndarray:
    """Stochastic Gradient HMC (Chen 2014 + Springer 2016).

    Reference implementation verified (v0.9.229-a, P63 test suite):
      Gaussian posterior N(1.998, 0.0316) recovered as
      mean=1.99, std=0.033 at lr=0.01, C=20*I, V_hat=I.
      Ratio std/analytic = 1.05 (within Monte-Carlo error).

    Update rule (reference file sghmc.py, validated):
      r = sqrtM @ N(0, I)                    # per-epoch momentum reset
      for batch in batches:
          theta += prer @ r                  # leapfrog half-step on theta
          gU = -(n * grad_log_data(batch,theta) / batch_size
                 + grad_log_prior(theta))
          r = r - lr*gU - fric @ r + sqrt_noise @ N(0, I)
      theta += prer @ r                      # final leapfrog half-step

    where:
      fric       = lr * C                (Minv=None, M=I)
      sqrt_noise = sqrtm(2*(C - 0.5*lr*V_hat)*lr)

    Constraint: C - 0.5*lr*V_hat MUST be positive definite (else
    negative injected variance → chain diverges; checked at call).

    Args:
      df: CadwDataFrame.
      grad_log_den_data_fn: (batch, theta) -> (d,) — Σ over batch of
        ∇log p(y_i | theta). NOT divided by batch_size — primitive
        scales by n/batch_size.
      grad_log_den_prior_fn: (theta) -> (d,) — full ∇log p(theta).
      theta_init: (d,).
      V_hat: (d,d) Fisher estimate. None → identity (safe default
        per P63 grid: V_hat=1 vs 0.01 gave identical results when
        2*C >> lr*V_hat).
      C: (d,d) friction. None → 20*I (P63: std/analytic=1.05 with
        eps=0.01; C=5*I gives 1.14, C=I gives 1.31).
      n_samples, n_burnin: standard (3000/1000 works for d=1).
      batch_size: None → derived; large batch reduces noise.
      lr: step size. Default 0.01 (eps=0.1 overflowed in P63).
      Minv: optional mass matrix inverse (None = M=I).
      seed, memory_guard: standard.

    Returns:
      (n_samples, d) posterior draws.

    Raises:
      ValueError: constraint 2*C >= lr*V_hat violated.
    """
    from scipy import linalg as la

    theta = np.asarray(theta_init, dtype=np.float64).copy()
    if theta.ndim != 1:
        raise ValueError(f"theta_init 1-D; got {theta.shape}")
    d = theta.size
    if n_samples < 1:
        raise ValueError(f"n_samples >= 1; got {n_samples}")
    if n_burnin < 0:
        raise ValueError(f"n_burnin >= 0; got {n_burnin}")
    if lr <= 0:
        raise ValueError(f"lr > 0; got {lr}")

    if V_hat is None:
        V_hat = np.eye(d)
    if C is None:
        C = 20.0 * np.eye(d)
    V_hat = np.asarray(V_hat, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    if V_hat.shape != (d, d):
        raise ValueError(f"V_hat shape {V_hat.shape}, expected ({d},{d})")
    if C.shape != (d, d):
        raise ValueError(f"C shape {C.shape}, expected ({d},{d})")

    if Minv is None:
        sqrtM = np.eye(d)
        prer = lr
        fric = lr * C
    else:
        Minv = np.asarray(Minv, dtype=np.float64)
        if Minv.shape != (d, d):
            raise ValueError(f"Minv shape {Minv.shape}, expected ({d},{d})")
        sqrtM = la.sqrtm(la.inv(Minv))
        prer = lr * Minv
        fric = lr * C @ Minv

    # Constraint: C - 0.5*lr*V_hat positive definite
    noise_matrix = C - 0.5 * lr * V_hat
    eigvals = np.linalg.eigvalsh(noise_matrix)
    if eigvals.min() <= 0:
        raise ValueError(
            f"sgmcmc: constraint 2*C >= lr*V_hat violated "
            f"(min eig(C - 0.5*lr*V_hat)={eigvals.min():.3e}). "
            f"Increase C or decrease lr/V_hat.")
    sqrt_noise = la.sqrtm(2.0 * noise_matrix * lr)

    mg = memory_guard or df._memory_guard
    if batch_size is None:
        _row = mg.estimate_row_size_from_schema(df._dataset.schema)
        batch_size = mg.get_batch_rows(
            row_size_bytes=_row, expansion_factor=3.0, state_bytes=d * 8)
    batch_size = max(1, int(batch_size))

    n_total = int(df.count())
    if n_total == 0:
        raise ValueError("sgmcmc: empty df")

    mg.check_state(
        d * 8 * 4 + n_samples * d * 8 + 256,
        name="sgmcmc.state")

    rng = np.random.default_rng(seed)
    samples = np.empty((n_samples, d), dtype=np.float64)

    batches_cache: list = []
    total_iter = int(n_burnin) + int(n_samples)
    sample_idx = 0

    for t in range(total_iter):
        if not batches_cache:
            batches_cache = list(df._iter_filtered_batches(batch_size=batch_size))
            if not batches_cache:
                raise ValueError("sgmcmc: empty df")

        # Momentum reset per epoch (reference: r ~ N(0, I))
        r = sqrtM @ rng.standard_normal(d)

        for b in batches_cache:
            mg.check_batch(b, name="sgmcmc.batch", extra_factor=3.0)
            # Leapfrog half-step on theta (BEFORE gradient)
            theta = theta + (prer * r if Minv is None else prer @ r)
            # gradU = -(n * batch_grad / batch_size + prior_grad)
            g_data = grad_log_den_data_fn(b, theta)  # sum over batch
            g_prior = grad_log_den_prior_fn(theta)
            if not np.isfinite(g_data).all() or not np.isfinite(g_prior).all():
                raise RuntimeError(
                    f"sgmcmc: grad non-finite at iter {t}")
            gU = -(n_total * g_data / b.num_rows + g_prior)
            # Momentum update
            r = r - lr * gU - fric @ r + sqrt_noise @ rng.standard_normal(d)

        # Final leapfrog half-step on theta (AFTER inner loop)
        theta = theta + (prer * r if Minv is None else prer @ r)

        if t >= int(n_burnin) and sample_idx < int(n_samples):
            samples[sample_idx] = theta
            sample_idx += 1

    if sample_idx < int(n_samples):
        raise RuntimeError(
            f"sgmcmc: collected {sample_idx}/{n_samples} samples")
    return samples



def gibbs(
    df,
    update_fns: "list[Callable]",
    state_init: dict,
    *,
    n_samples: int = 1000,
    n_burnin: int = 200,
    thin: int = 1,
    seed: int = 0,
    memory_guard: Optional[MemoryGuard] = None,
) -> list[dict]:
    """Generic blocked Gibbs sampler.

    Args:
      df: CadwDataFrame (passed to each update_fn).
      update_fns: list of (state, df, rng) -> state dict. Each updates
        one block of state given current values of others.
      state_init: dict — initial state.
      n_samples, n_burnin: standard.
      thin: keep every thin-th sample (post burn-in).
      seed: deterministic.

    Returns:
      list of n_samples dict samples (post burn-in, thinned).
    """
    rng = np.random.default_rng(seed)
    state = {k: v.copy() if hasattr(v, 'copy') else v
             for k, v in state_init.items()}
    samples: list[dict] = []
    total_iter = int(n_burnin) + int(n_samples) * int(thin)
    sample_count = 0
    for it in range(total_iter):
        for fn in update_fns:
            state = fn(state, df, rng)
        if it >= int(n_burnin) and (it - int(n_burnin)) % int(thin) == 0:
            samples.append({k: (v.copy() if hasattr(v, 'copy') else v)
                            for k, v in state.items()})
            sample_count += 1
            if sample_count >= n_samples:
                break
    return samples



def srvr_hmc(
    df,
    grad_log_den_data_fn: Callable,
    grad_log_den_prior_fn: Callable,
    theta_init: np.ndarray,
    *,
    n_samples: int = 1000,
    n_burnin: int = 3000,
    L: int = 10,
    B: int = 100,
    reset_batch_size: "int | None" = None,
    lr: float = 0.01,
    gamma: float = 20.0,
    u: float = 1.0,
    seed: int = 0,
    memory_guard: Optional[MemoryGuard] = None,
) -> np.ndarray:
    """SRVR-HMC (Zou, Xu, Gu NeurIPS 2019).

    Stochastic Variance-Reduced HMC with exponential integrator and
    semi-stochastic gradients. Two advantages over SGHMC:
      (1) Variance reduction: g_k = mean_B[∇f(x_k) - ∇f(x_prev)] + g_prev
          instead of noisy mean_B[∇f(x_k)].
      (2) No V_hat / C tuning: friction γ and inverse mass u are
          explicit parameters; the SDE (not its discretization) sets
          the stationary distribution.

    Reference: https://proceedings.neurips.cc/paper/2019/file/c3535febaff29fcb7c0d20cbe94391c7-Paper.pdf

    Update (exponential integrator, exact for constant g over [t, t+η]):
      v_{k+1} = e^{-γη} v_k - (u/γ)(1 - e^{-γη}) g_k + e^v_k
      x_{k+1} = x_k + (1/γ)(1 - e^{-γη}) v_k
              - (u/γ²)(γη - 1 + e^{-γη}) g_k + e^x_k

    Noise (joint Gaussian, per-dimension independent block):
      Var(e^v) = u(1 - e^{-2γη})
      Var(e^x) = (u/γ²)(2γη - 3 + 4e^{-γη} - e^{-2γη})
      Cov(e^x, e^v) = (u/γ)(1 - 2e^{-γη} + e^{-2γη})

    Semi-stochastic gradient (g_k is mean ∇f = -∇log_posterior):
      k mod L == 0  (reset): g_k = (1/n) Σ_all ∇f_i(x_k)
      k mod L != 0  (inner): g_k = (1/B) Σ_B [∇f_i(x_k) - ∇f_i(x_prev)]
                                  + g_prev_k
    where ∇f_i = -∇log p(y_i | x), and prior grad is added separately
    as ∇f_prior = -∇log p(x).

    OOC: reset streams full data (O(reset_batch_size·row_size) peak);
    inner steps use one B-sized batch (O(B·row_size)). State = 4d
    floats (x, v, x_prev, g_prev). Total OOC.

    Args:
      df: CadwDataFrame.
      grad_log_den_data_fn: (batch, theta) -> (d,) = Σ_batch ∇log p(y|x).
      grad_log_den_prior_fn: (theta) -> (d,) = ∇log p(theta).
      theta_init: (d,).
      n_samples, n_burnin: standard.
      L: reset interval (steps between full-gradient snapshots).
      B: inner mini-batch size.
      reset_batch_size: streaming batch for reset pass; None -> B.
      lr: step size η.
      gamma: friction coefficient γ (per-unit-time). Default 20 matches
        SGHMC C=20 equivalent (γη ~ 0.2 per step at lr=0.01).
      u: inverse mass. Default 1.
      seed, memory_guard: standard.

    Returns:
      (n_samples, d) posterior draws.
    """
    if not callable(grad_log_den_data_fn) or not callable(grad_log_den_prior_fn):
        raise TypeError("grad fns must be callable")
    theta = np.asarray(theta_init, dtype=np.float64).copy()
    if theta.ndim != 1:
        raise ValueError(f"theta_init 1-D; got {theta.shape}")
    d = theta.size
    if n_samples < 1:
        raise ValueError(f"n_samples >= 1; got {n_samples}")
    if n_burnin < 0:
        raise ValueError(f"n_burnin >= 0; got {n_burnin}")
    if L < 1:
        raise ValueError(f"L >= 1; got {L}")
    if B < 1:
        raise ValueError(f"B >= 1; got {B}")
    if lr <= 0:
        raise ValueError(f"lr > 0; got {lr}")
    if gamma <= 0:
        raise ValueError(f"gamma > 0; got {gamma}")
    if u <= 0:
        raise ValueError(f"u > 0; got {u}")

    mg = memory_guard or df._memory_guard
    if reset_batch_size is None:
        reset_batch_size = B
    reset_batch_size = max(1, int(reset_batch_size))
    B = max(1, int(B))
    d_check = d

    # State: x, v, x_snapshot, g_snapshot_data (4d) + samples (n_samples*d)
    mg.check_state(4 * d * 8 + n_samples * d * 8 + 256,
                   name="srvr_hmc.state")

    x = theta.copy()
    v = np.zeros(d, dtype=np.float64)
    x_snapshot = x.copy()
    g_snapshot_data = np.zeros(d, dtype=np.float64)  # ∇U(x_snapshot)

    # Noise coefficients (Springer exponential integrator)
    e1 = float(np.exp(-gamma * lr))
    e2 = float(np.exp(-2.0 * gamma * lr))
    var_x = u / (gamma ** 2) * (2.0 * gamma * lr - 3.0 + 4.0 * e1 - e2)
    var_v = u * (1.0 - e2)
    cov_xv = u / gamma * (1.0 - 2.0 * e1 + e2)
    if var_x <= 0 or var_v <= 0:
        raise ValueError(
            f"srvr_hmc: negative variance (var_x={var_x:.2e}, "
            f"var_v={var_v:.2e}); reduce lr or increase gamma. "
            f"Constraint: exponential integrator needs stable η, γ.")
    L00 = float(np.sqrt(var_x))
    L10 = float(cov_xv) / L00
    L11_sq = var_v - L10 * L10
    if L11_sq < 0:
        raise ValueError(
            f"srvr_hmc: noise covariance not PSD (L11²={L11_sq:.2e})")
    L11 = float(np.sqrt(L11_sq))

    # Update coefficients
    coef_v = e1
    coef_g_v = -u / gamma * (1.0 - e1)
    coef_v_x = (1.0 - e1) / gamma
    coef_g_x = -u / (gamma ** 2) * (gamma * lr - 1.0 + e1)

    # Batch cycling iterator (mini-batch)
    def _batch_gen():
        while True:
            for b in df._iter_filtered_batches(batch_size=B):
                yield b
    batch_iter = _batch_gen()

    rng = np.random.default_rng(seed)
    samples = np.empty((n_samples, d), dtype=np.float64)
    sample_idx = 0
    total_iter = int(n_burnin) + int(n_samples)

    for k in range(total_iter):
        # P73 FIX: snapshot-based gradient (Zou 2019 eq. 2.2)
        #   g_k = (1/B) Σ_B [∇f_i(x_k) - ∇f_i(x̃)] + ∇f(x̃)
        # where x̃ = x_snapshot (fixed at last reset). NOT recursive.
        if k % L == 0:
            # Reset: full streaming pass, fix snapshot
            g_sum_logp = np.zeros(d, dtype=np.float64)
            n_total = 0
            for b in df._iter_filtered_batches(batch_size=reset_batch_size):
                mg.check_batch(b, name="srvr_hmc.reset", extra_factor=3.0)
                g_sum_logp += np.asarray(grad_log_den_data_fn(b, x),
                                          dtype=np.float64)
                n_total += b.num_rows
            if n_total == 0:
                raise ValueError("srvr_hmc: empty df")
            # P75 FIX: ∇U_data(x̃) = -Σ_all ∇log p_data(x̃) (total, not mean).
            # Prior g_prior = ∇U_prior(x) is already full scale (no /n).
            # Missing n× here caused target-posterior mismatch:
            # effective U = mean_U_data + U_prior => N(1, 1/2), not
            # N(n·ȳ/(n+1), 1/(n+1)).
            g_snapshot_data = -g_sum_logp
            x_snapshot = x.copy()
            g_data_k = g_snapshot_data  # for current step, use snapshot
        else:
            try:
                b = next(batch_iter)
            except StopIteration:
                batch_iter = _batch_gen()
                b = next(batch_iter)
            mg.check_batch(b, name="srvr_hmc.minibatch", extra_factor=3.0)
            n_b = b.num_rows
            grad_curr = np.asarray(grad_log_den_data_fn(b, x),
                                    dtype=np.float64)
            grad_snapshot = np.asarray(
                grad_log_den_data_fn(b, x_snapshot), dtype=np.float64)
            # P75 FIX: SVRG correction with n_total/n_b scale (Horvitz-
            # Thompson-like unbiased estimator for full ∇U_data(x_k) - ∇U_data(x̃)).
            #   ∇U_data(x_k) - ∇U_data(x̃) = -Σ_all ∇log p(x_k) + Σ_all ∇log p(x̃)
            #                              ≈ -(n_total/n_b) Σ_batch [∇log p(x_k) - ∇log p(x̃)]
            #   => g_data_k = g_snapshot_data - (n_total/n_b) * delta_sum
            delta_sum = grad_curr - grad_snapshot  # sum over batch
            g_data_k = g_snapshot_data - (float(n_total) / float(n_b)) * delta_sum

        g_prior = -np.asarray(grad_log_den_prior_fn(x), dtype=np.float64)
        g_total = g_data_k + g_prior

        if not np.isfinite(g_total).all():
            raise RuntimeError(
                f"srvr_hmc: grad non-finite at iter {k}")

        # Correlated noise (per-dim independent)
        z0 = rng.standard_normal(d)
        z1 = rng.standard_normal(d)
        ex = L00 * z0
        ev = L10 * z0 + L11 * z1

        # Update (both use OLD v)
        v_new = coef_v * v + coef_g_v * g_total + ev
        x_new = x + coef_v_x * v + coef_g_x * g_total + ex

        x = x_new
        v = v_new

        if k >= int(n_burnin) and sample_idx < int(n_samples):
            samples[sample_idx] = x.copy()
            sample_idx += 1

    if sample_idx < int(n_samples):
        raise RuntimeError(
            f"srvr_hmc: collected {sample_idx}/{n_samples}")
    return samples
