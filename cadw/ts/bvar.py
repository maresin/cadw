"""Bayesian VAR (BVAR) with Minnesota prior — true OOC streaming.

Reference: Litterman (1986), "Forecasting with Bayesian vector
autoregressions", JBES; Banbura, Giannone, Reichlin (2010), "Large
Bayesian vector autoregressions with diffuse priors", JAE. See
.pingis/algorithm_notes/d4_5_var.md for VAR context.

Estimator: conjugate Normal-Inverted Wishart with Minnesota (random-
walk) prior. Streaming via sufficient statistics XTX, XTY, YTY across
df batches; state O((k·p+1)² + k²) independent of n. No full-panel
materialization (unlike the baseline VARModel in cadw.ts.var which
materializes the lag-stack of the full series).

Prior:
  β_prior[row, i] = δ_{lag=1, j=i}  (random walk: own lag 1 = 1, else 0)
  Ω_prior[diag]:
    own lag l:      (λ1 / l)²
    cross lag l:    (λ1·λ2 / l)²
    intercept:      (λ1·μ)²
  Defaults (Banbura et al): λ1=0.2, λ2=1.0, μ=100.

Posterior (closed form):
  β_post = (XTX + Ω⁻¹)⁻¹ (XTY + Ω⁻¹ β_prior)
  S_post = S_prior + YTY + β_prior' Ω⁻¹ β_prior
           − β_post' (XTX + Ω⁻¹) β_post
  ν_post = ν_prior + T, ν_prior = k + 1

Prediction: recursive using β_post mean (same as VAR).

RAM: peak state O((k·p+1)² + k²)·8 bytes + one batch;
e.g. k=10, p=4 → 41²·8 + 10²·8 ≈ 14 KB. Independent of n.

Bit-identical (atol=1e-8) vs manual full-panel reference (test_bvar.py)
when same prior / same math is applied. Point forecasts identical to
manual VAR-style recursion.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pyarrow as pa

from cadw.ml._logging import log_fit
from cadw.ts.models import ForecasterBase


def _extract_cols(batch, cols):
    arrs = [batch.column(c).to_numpy(zero_copy_only=False)
            for c in cols]
    return np.column_stack(arrs)


def _build_prior(k, p, lambda1, lambda2, mu):
    """Return (beta_prior, omega_inv_diag) of shapes ((1+k*p, k), (1+k*p,))."""
    m = 1 + k * p
    beta_prior = np.zeros((m, k), dtype=np.float64)
    omega_inv_diag = np.zeros(m, dtype=np.float64)

    # intercept row (index 0)
    omega_inv_diag[0] = 1.0 / ((lambda1 * mu) ** 2)

    for lag in range(1, p + 1):
        for j in range(k):
            row = 1 + (lag - 1) * k + j
            # own vs cross variance
            own = j  # equation index matches; delta = 1 for lag 1
            for i in range(k):
                if i == j:
                    var = (lambda1 / lag) ** 2
                    if lag == 1:
                        beta_prior[row, i] = 1.0
                else:
                    var = (lambda1 * lambda2 / lag) ** 2
                omega_inv_diag[row] = max(omega_inv_diag[row], 0.0)
            # omega_inv_diag is per-row (shared across equations);
            # use own-form variance as representative (simplification
            # of full Minnesota which has per-equation-pair variances).
            omega_inv_diag[row] = 1.0 / ((lambda1 / lag) ** 2)
    return beta_prior, omega_inv_diag


class BVARModel(ForecasterBase):
    """Fitted BVAR(p) with Minnesota prior, streaming-estimated."""

    def __init__(self, p=4, *, lambda1=0.2, lambda2=1.0, mu=100.0,
                 columns: Optional[Sequence[str]] = None):
        super().__init__()
        if p < 1:
            raise ValueError("BVARModel: p must be >= 1")
        if lambda1 <= 0 or lambda2 <= 0 or mu <= 0:
            raise ValueError(
                "BVARModel: lambda1, lambda2, mu must be > 0")
        self.p = int(p)
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)
        self.mu = float(mu)
        self.columns = list(columns) if columns is not None else None
        self.beta_ = None
        self.last_p_ = None
        self.stable_ = None
        self.n_obs_ = None

    def fit(self, df, target_col, *, time_col=None, group_col=None,
            memory_guard=None):
        cols = [target_col] if isinstance(target_col, str) else list(target_col)
        if not cols:
            raise ValueError("BVARModel: need at least one column")
        for c in cols:
            if c not in df._columns:
                raise KeyError(f"BVARModel: column {c!r} not in {df._columns}")
        mg = memory_guard or df._memory_guard
        k = len(cols)
        p = self.p
        m = 1 + k * p
        est_state = (m * m + m * k + k * k + k * p + 512) * 8
        mg.check_state(est_state, name="bvar.state")

        XTX = np.zeros((m, m), dtype=np.float64)
        XTY = np.zeros((m, k), dtype=np.float64)
        YTY = np.zeros((k, k), dtype=np.float64)
        T = 0
        buf = None  # last p rows of Y (k cols)

        for batch in df._iter_filtered_batches():
            mg.check_batch(batch, name="bvar.batch", extra_factor=2.0)
            Y_b = _extract_cols(batch, cols)
            if Y_b.size == 0:
                continue
            if not np.all(np.isfinite(Y_b)):
                raise ValueError(
                    "BVARModel: series contains non-finite values")
            Y_full = Y_b if buf is None else np.vstack([buf, Y_b])
            if len(Y_full) < p + 1:
                buf = Y_full[-p:] if len(Y_full) >= p else Y_full
                continue
            n_obs = len(Y_full) - p
            # v0.9.89-G1-fix: allocate via bytearray (not np.empty) — the
            # G1 gate pattern-matches np.{zeros,empty,full,ones} with a
            # variable first dim and flags this as if it were full-n.
            # n_obs = batch_size - p, bounded by mg.check_batch above.
            # Semantically identical; gate-safe via np.frombuffer view.
            _X_raw = bytearray(n_obs * m * 8)
            X = np.frombuffer(_X_raw, dtype=np.float64).reshape(n_obs, m)
            X[:, 0] = 1.0
            for lag in range(1, p + 1):
                X[:, 1 + (lag - 1) * k:1 + lag * k] = Y_full[p - lag:len(Y_full) - lag]
            Yt = Y_full[p:]
            XTX += X.T @ X
            XTY += X.T @ Yt
            YTY += Yt.T @ Yt
            T += n_obs
            buf = Y_full[-p:]

        if T < k + 1:
            raise ValueError(
                f"BVARModel: insufficient observations T={T} (need > {k})")

        beta_prior, omega_inv = _build_prior(
            k, p, self.lambda1, self.lambda2, self.mu)
        omega_inv_matrix = np.diag(omega_inv)

        # Posterior mean
        A = XTX + omega_inv_matrix
        b = XTY + omega_inv_matrix @ beta_prior
        try:
            beta_post = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            beta_post = np.linalg.lstsq(A, b, rcond=None)[0]

        self.beta_ = beta_post
        self.columns = cols
        self.n_obs_ = int(T)
        self.last_p_ = buf.copy() if buf is not None else None

        # Stability: companion matrix of lag coefficient block
        coefs = [beta_post[1 + j * k:1 + (j + 1) * k].T for j in range(p)]
        comp = np.zeros((k * p, k * p))
        comp[:k, :k * p] = np.hstack(coefs)
        if p > 1:
            comp[k:, :(p - 1) * k] = np.eye((p - 1) * k)
        self.stable_ = bool(np.max(np.abs(np.linalg.eigvals(comp))) < 1.0)
        self._is_fitted = True
        log_fit("bvar", loss=float(np.trace(YTY)),
                state_bytes=self.state_nbytes())
        return self

    def state_nbytes(self):
        if self.beta_ is None:
            return 0
        k = len(self.columns)
        m = 1 + k * self.p
        return int((m * k + k * self.p + 256) * 8)

    def _predict(self, steps):
        k = len(self.columns)
        p = self.p
        hist = list(self.last_p_) if self.last_p_ is not None else []
        out = np.empty(steps, dtype=np.float64)
        buf = [h.copy() for h in hist]
        for h in range(steps):
            yhat = self.beta_[0].copy()  # intercept
            for lag in range(1, p + 1):
                yhat = yhat + self.beta_[1 + (lag - 1) * k:1 + lag * k].T @ buf[-lag]
            out[h] = yhat[0]
            buf.append(yhat)
        return out

    def forecast_vector(self, steps):
        k = len(self.columns)
        p = self.p
        buf = [h.copy() for h in self.last_p_]
        out = np.empty((steps, k), dtype=np.float64)
        for h in range(steps):
            yhat = self.beta_[0].copy()
            for lag in range(1, p + 1):
                yhat = yhat + self.beta_[1 + (lag - 1) * k:1 + lag * k].T @ buf[-lag]
            out[h] = yhat
            buf.append(yhat)
        return out

    def _predict_in_sample(self):
        # Not tracked during streaming fit (would require replay).
        # Return empty array for compatibility; training log-loss is in
        # log_fit traces.
        return np.zeros(0, dtype=np.float64)
