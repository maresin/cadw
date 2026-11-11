"""Plaid biclustering (Lazzeroni & Owen 2002, deterministic ALS).

Model: A_ij = sum_l theta_ijl + eps_ij
  theta_ijl = mu_l + alpha_il + beta_jl (row i, col j in layer l)
  theta_ijl = 0                         (otherwise)

Reference: Lazzeroni & Owen 2002, "Plaid models for gene expression
data", Statistica Sinica 12:61-86.

CADW implements a DETERMINISTIC ALS variant of the original Bayesian
MCMC approach (which is stochastic and slow). The additive layer
decomposition is faithful; the inference (ALS vs MCMC) differs.
Documented as "deterministic ALS" in every API call.

Algorithm:
  1. Materialize A (r, c) f64.
  2. residual R = A
  3. For layer l = 0..max_layers-1:
     a. seed = argmax |R|. If |R[seed]| < threshold, break.
     b. R_l = {seed_row}, C_l = {seed_col}. Expand until stable:
        - for each row i not in R_l: if mean|R[i, C_l]| > threshold,
          add i
        - for each col j not in C_l: if mean|R[R_l, j]| > threshold,
          add j
     c. ALS fit (mu, alpha, beta) on R[np.ix_(R_l, C_l)] with
        re-centering per iter. max_als_iters, tol.
     d. theta_l = mu + alpha[:,None] + beta[None,:] on R_l x C_l.
     e. R[np.ix_(R_l, C_l)] -= theta_l
     f. Stop if |R_l| < min_rows OR |C_l| < min_cols OR reduction < tol.
  4. Return layers.

OOC (Pattern #3 intrinsic — iterative dense):
  - A + residual (r*c f64 each): bounded_by_domain {r*c: 1e8}
  - Per-layer ALS: dense submatrix — intrinsic
  - refuse_at_guard {min(r,c): 500}
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import (
    MemoryGuard, MemoryLimitExceeded)
from cadw.core.warnings import SlowOperationWarning
from cadw.ml._logging import log_fit


_MAX_RC_INTRINSIC = 100_000_000  # 1e8; A + R = 1.6 GB f64
_MAX_MIN_RC = 500                # per-layer ALS refuses above


def _batch_to_X(b, X_cols):
    arrs = []
    for c in X_cols:
        col = b.column(c)
        if not pa.types.is_floating(col.type) and \
           not pa.types.is_integer(col.type):
            col = col.cast(pa.float64())
        arrs.append(col.to_numpy(zero_copy_only=False).astype(
            np.float64, copy=False))
    return np.column_stack(arrs) if arrs else np.empty((b.num_rows, 0))


@dataclass
class PlaidLayer:
    """One layer in Plaid decomposition.

    rows_: tuple[int] — row indices in layer
    cols_: tuple[int] — col indices in layer
    mu_: float
    alpha_: np.ndarray — per-row offsets, shape (len(rows_),)
    beta_: np.ndarray  — per-col offsets, shape (len(cols_),)
    """
    rows_: tuple = ()
    cols_: tuple = ()
    mu_: float = 0.0
    alpha_: Optional[np.ndarray] = None
    beta_: Optional[np.ndarray] = None

    def nbytes(self) -> int:
        sz = 8 + 8  # mu
        if self.alpha_ is not None:
            sz += int(self.alpha_.nbytes)
        if self.beta_ is not None:
            sz += int(self.beta_.nbytes)
        sz += len(self.rows_) * 8 + len(self.cols_) * 8
        return int(sz)


@dataclass
class PlaidResult:
    """Plaid biclustering fit result.

    layers_: list[PlaidLayer]
    A_shape: (r, c)
    n_layers_: len(layers_)
    residual_norm_: ||A - sum_l theta_l||_F
    """
    layers_: list = field(default_factory=list)
    A_shape: tuple = ()
    n_layers_: int = 0
    residual_norm_: float = 0.0
    threshold_: float = 0.5
    max_als_iters_: int = 10

    def state_nbytes(self) -> int:
        sz = 256
        for layer in self.layers_:
            sz += layer.nbytes()
        return int(sz)

    def to_dict(self) -> dict:
        return {
            "A_shape": tuple(self.A_shape),
            "n_layers": int(self.n_layers_),
            "residual_norm": float(self.residual_norm_),
            "threshold": float(self.threshold_),
            "max_als_iters": int(self.max_als_iters_),
            "layers": [
                {"rows": len(l.rows_), "cols": len(l.cols_),
                 "mu": float(l.mu_)}
                for l in self.layers_
            ],
        }


def _als_fit_submatrix(R_sub, max_iters, tol):
    """Fit mu + alpha_i + beta_j to R_sub via alternating LS.

    Returns (mu, alpha, beta). Model:
      R_sub[i,j] ≈ mu + alpha[i] + beta[j]
    with constraint sum(alpha)=0, sum(beta)=0 (absorbed into mu).
    """
    n_r, n_c = R_sub.shape
    if n_r == 0 or n_c == 0:
        return 0.0, np.zeros(n_r), np.zeros(n_c)
    mu = float(R_sub.mean())
    alpha = np.zeros(n_r, dtype=np.float64)
    beta = np.zeros(n_c, dtype=np.float64)
    for _ in range(int(max_iters)):
        alpha_old = alpha.copy()
        beta_old = beta.copy()
        # Fix beta, solve alpha: alpha[i] = mean_j(R - mu - beta)
        alpha = (R_sub - mu - beta[None, :]).mean(axis=1)
        # Fix alpha, solve beta
        beta = (R_sub - mu - alpha[:, None]).mean(axis=0)
        # Re-center (absorb into mu)
        mu_new = mu + float(alpha.mean()) + float(beta.mean())
        alpha -= alpha.mean()
        beta -= beta.mean()
        mu = mu_new
        if (np.linalg.norm(alpha - alpha_old) < tol and
                np.linalg.norm(beta - beta_old) < tol):
            break
    return mu, alpha, beta


def _expand_layer(R, seed_i, seed_j, threshold, min_rows, min_cols,
                   max_expand_iters=100):
    """Grow layer from seed via mean|R| > threshold criterion."""
    r, c = R.shape
    R_l = {seed_i}
    C_l = {seed_j}
    for _ in range(max_expand_iters):
        size_before = len(R_l) + len(C_l)
        # Add rows
        C_list = sorted(C_l)
        sub = np.abs(R[:, C_list])
        row_means = sub.mean(axis=1)
        for i in range(r):
            if i not in R_l and row_means[i] > threshold:
                R_l.add(i)
        # Add cols
        R_list = sorted(R_l)
        sub = np.abs(R[R_list, :])
        col_means = sub.mean(axis=0)
        for j in range(c):
            if j not in C_l and col_means[j] > threshold:
                C_l.add(j)
        if len(R_l) + len(C_l) == size_before:
            break
    return tuple(sorted(R_l)), tuple(sorted(C_l))


def fit_plaid_model(
    df,
    X_cols,
    *,
    max_layers: int = 10,
    min_rows: int = 2,
    min_cols: int = 2,
    threshold: float = 0.5,
    max_als_iters: int = 10,
    tol: float = 1e-4,
    batch_size: Optional[int] = None,
    memory_guard: Optional[MemoryGuard] = None,
) -> PlaidResult:
    """Plaid biclustering (Lazzeroni & Owen 2002, deterministic ALS).

    Args:
      df: CadwDataFrame with numeric X_cols.
      X_cols: list[str].
      max_layers: maximum layers (default 10).
      min_rows, min_cols: minimum layer size (default 2x2).
      threshold: |mean R| cutoff for layer expansion (default 0.5).
      max_als_iters: ALS iterations per layer (default 10).
      tol: ALS convergence tolerance (default 1e-4).
      batch_size, memory_guard: standard.

    Returns PlaidResult with layers_.

    OOC (Pattern #3 intrinsic):
      - A + residual (r*c f64 each): bounded_by_domain {r*c: 1e8}.
      - Per-layer ALS: dense submatrix — intrinsic.
      - refuse_above {min(r,c): 500}.
      - SlowOperationWarning at est > 60s.

    Raises ValueError/KeyError/MemoryLimitExceeded.
    """
    if not X_cols:
        raise ValueError("fit_plaid_model: X_cols is empty")
    X_cols = list(X_cols)
    if max_layers < 1:
        raise ValueError(
            f"fit_plaid_model: max_layers must be >= 1; got {max_layers}")
    if min_rows < 1 or min_cols < 1:
        raise ValueError(
            f"fit_plaid_model: min_rows/min_cols must be >= 1; got "
            f"({min_rows}, {min_cols})")
    if max_als_iters < 1:
        raise ValueError(
            f"fit_plaid_model: max_als_iters must be >= 1; got "
            f"{max_als_iters}")
    missing = [c for c in X_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"fit_plaid_model: columns missing: {missing}")

    mg = memory_guard or getattr(df, "_memory_guard", None)
    r = int(df.count())
    c = len(X_cols)
    if r < 1 or c < 1:
        raise ValueError(f"need r>=1 and c>=1; got r={r}, c={c}")

    # Intrinsic bound: min(r, c) > 500 -> refuse (per-layer ALS slow)
    if min(r, c) > _MAX_MIN_RC:
        raise MemoryLimitExceeded(
            f"fit_plaid_model: min(r={r}, c={c})={min(r,c)} > "
            f"{_MAX_MIN_RC} (per-layer ALS on dense submatrix; "
            f"O(r*c*max_als_iters) intrinsic). Options: (a) subsample "
            f"rows/cols; (b) reduce max_layers; (c) use Bimax (binary) "
            f"or SpectralBiclustering.",
            model="PlaidModel", params={"r": r, "c": c,
                                        "min_rc": min(r, c)})

    # A + residual: 2 * r * c * 8
    A_bytes = r * c * 8
    state_est = 2 * A_bytes + 256
    if mg is not None and hasattr(mg, "check_state"):
        mg.check_state(state_est, name="plaid.state")

    if batch_size is None:
        batch_size = max(1, min(r, 1000))
    batch_size = max(1, min(int(batch_size), r))

    # AP22 slow-op warning
    est_iters = min(2 ** min(r, c), max_layers * max_als_iters * 10)
    est_time_s = est_iters * r * c / 1e7
    if est_time_s > 60.0:
        import warnings as _w
        _w.warn(
            f"fit_plaid_model: r={r}, c={c}, max_layers={max_layers}, "
            f"max_als_iters={max_als_iters}; estimated ~{est_time_s:.0f}s "
            f"(>{60.0:.0f}s). Options: (a) reduce max_layers; "
            f"(b) reduce max_als_iters; (c) subsample df.",
            SlowOperationWarning,
            stacklevel=2)

    # Pass 1: materialize A
    A_parts = []
    for b in df._iter_filtered_batches(batch_size=batch_size):
        if mg is not None and hasattr(mg, "check_batch"):
            mg.check_batch(b, name="plaid.A", extra_factor=2.0)
        A_parts.append(_batch_to_X(b, X_cols))
    A = (np.vstack(A_parts) if len(A_parts) > 1
         else A_parts[0] if A_parts else np.empty((0, c)))
    if A.shape != (r, c):
        raise RuntimeError(f"A shape mismatch: {A.shape} != ({r},{c})")
    if np.isnan(A).any():
        raise ValueError(
            "fit_plaid_model: A contains NaN; impute first")

    # Pass 2: iterative ALS layers
    R = A.copy()
    layers = []
    for _ in range(int(max_layers)):
        # Find seed = argmax|R|
        abs_R = np.abs(R)
        flat_idx = int(np.argmax(abs_R))
        seed_i, seed_j = np.unravel_index(flat_idx, R.shape)
        if abs_R[seed_i, seed_j] < threshold:
            break  # no significant residual left

        # Expand layer
        R_l, C_l = _expand_layer(
            R, seed_i, seed_j, threshold, min_rows, min_cols)
        if len(R_l) < min_rows or len(C_l) < min_cols:
            break

        # ALS fit
        sub = R[np.ix_(R_l, C_l)]
        mu, alpha, beta = _als_fit_submatrix(sub, max_als_iters, tol)

        # Reduction check
        theta = mu + alpha[:, None] + beta[None, :]
        reduction = float(np.sum(sub * theta))
        if abs(reduction) < tol * len(R_l) * len(C_l):
            break

        # Update residual
        R[np.ix_(R_l, C_l)] -= theta

        layers.append(PlaidLayer(
            rows_=R_l, cols_=C_l,
            mu_=float(mu),
            alpha_=alpha.copy(),
            beta_=beta.copy(),
        ))

    residual_norm = float(np.linalg.norm(R))

    result = PlaidResult(
        layers_=layers,
        A_shape=(int(r), int(c)),
        n_layers_=len(layers),
        residual_norm_=residual_norm,
        threshold_=float(threshold),
        max_als_iters_=int(max_als_iters),
    )

    try:
        log_fit("plaid_model", state_bytes=result.state_nbytes())
    except Exception:
        pass

    return result
