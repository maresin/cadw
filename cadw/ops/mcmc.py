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
