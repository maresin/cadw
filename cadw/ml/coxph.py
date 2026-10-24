"""Cox proportional hazards (D.3.1, Phase D, v0.9.36).

Reference: Cox, Regression Models and Life-Tables, JRSS-B 34(2), 1972;
Andersen & Gill, Ann. Statist. 10(4), 1982 (counting-process/Breslow
formulation used here); lifelines CoxPHFitter (MIT) as the API reference.
See .pingis/algorithm_notes/d3_1_coxph.md.

RAM shape (stated honestly): CoxPH's Newton step is a sum over ALL events
of risk-set quantities, and the risk set at t depends on every row with
T >= t. Unlike a pure Reduce, the information matrix cannot be built from
batch-local partial sums without knowing the risk-set membership at every
event. We therefore take the guarded-pre-pass approach already accepted
for LambdaMART (D.2.1): one streaming pass collects X/T/E, guarded by
check_state(N*d*8) BEFORE allocating, then Newton-Raphson runs in memory.

Breslow partial log-likelihood:
    L(b) = sum_{events i} [ x_i.b - log( sum_{j: T_j >= T_i} exp(x_j.b)) ]
Gradient:  sum_events [ x_i - S1/S0 ]
Hessian:  -sum_events [ S2/S0 - (S1/S0)(S1/S0)^T ]
with S_k = sum_{risk set} w x^k, w = exp(x.b).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.ml._logging import log_fit


def _stack_num(b, cols):
    arrs = []
    for c in cols:
        a = b.column(c)
        if not (pa.types.is_integer(a.type) or pa.types.is_floating(a.type)):
            raise TypeError(f"coxph: column {c!r} must be numeric; got {a.type}")
        arrs.append(a.cast(pa.float64(), safe=False).to_numpy(zero_copy_only=False))
    return np.column_stack(arrs)


def _breslow(X, T, E, b):
    """Gradient and observed information at coefficients b (Breslow ties)."""
    n, d = X.shape
    eta = np.clip(X @ b, -50.0, 50.0)
    w = np.exp(eta)
    order = np.argsort(-T, kind="stable")
    Xo, To, Eo, wo = X[order], T[order], E[order], w[order]
    S0 = np.cumsum(wo)
    S1 = np.cumsum(wo[:, None] * Xo, axis=0)
    S2 = np.cumsum(wo[:, None, None] * Xo[:, :, None] * Xo[:, None, :], axis=0)
    # last index of each time block (risk set = rows [0..last])
    last = np.empty(n, dtype=np.int64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and To[j + 1] == To[i]:
            j += 1
        last[i:j + 1] = j
        i = j + 1
    ev = np.flatnonzero(Eo.astype(bool))
    if ev.size == 0:
        raise ValueError("coxph: no events (all censored)")
    li = last[ev]
    rs0 = S0[li]
    m1 = S1[li] / rs0[:, None]
    grad = (Xo[ev] - m1).sum(axis=0)
    info = (S2[li] / rs0[:, None, None]
            - m1[:, :, None] * m1[:, None, :]).sum(axis=0)
    ll = float((Xo[ev] @ b).sum() - np.log(rs0).sum())
    return grad, info, ll


class CoxPHModel:
    """Fitted Cox proportional hazards model (Breslow ties)."""

    def __init__(self, coef, cov, baseline_t, baseline_h, feature_names,
                 n_events, n_iter, converged, loglik):
        self.coef_ = np.asarray(coef, dtype=np.float64)
        self.covariance_ = np.asarray(cov, dtype=np.float64)
        self.baseline_t_ = np.asarray(baseline_t, dtype=np.float64)
        self.baseline_h_ = np.asarray(baseline_h, dtype=np.float64)
        self.feature_names_ = list(feature_names)
        self.n_events_ = int(n_events)
        self.n_iter_ = int(n_iter)
        self.converged_ = bool(converged)
        self.log_likelihood_ = float(loglik)
        self.fitted_ = True

    def state_nbytes(self):
        return int(self.coef_.nbytes + self.covariance_.nbytes
                   + self.baseline_t_.nbytes + self.baseline_h_.nbytes)

    def to_dict(self):
        return {"coef": self.coef_.tolist(),
                "feature_names": self.feature_names_,
                "n_events": self.n_events_, "n_iter": self.n_iter_,
                "converged": self.converged_,
                "log_likelihood": self.log_likelihood_}

    def predict_risk(self, df, X_cols=None):
        cols = list(X_cols) if X_cols is not None else self.feature_names_
        out = []
        for b in df._iter_filtered_batches():
            out.append(_stack_num(b, cols) @ self.coef_)
        arr = np.concatenate(out) if out else np.zeros(0)
        return pa.table({"risk": pa.array(arr, type=pa.float64())})

    def concordance_index(self, df, time_col, event_col, X_cols=None):
        """Harrell's c-index (higher risk = shorter survival)."""
        cols = list(X_cols) if X_cols is not None else self.feature_names_
        Ts, Es, Rs = [], [], []
        for b in df._iter_filtered_batches():
            Ts.append(b.column(time_col).to_numpy(zero_copy_only=False))
            Es.append(b.column(event_col).to_numpy(zero_copy_only=False))
            Rs.append(_stack_num(b, cols) @ self.coef_)
        T = np.concatenate(Ts); E = np.concatenate(Es); R = np.concatenate(Rs)
        conc = disc = ties = 0
        n = len(T)
        # Harrell's pairs, vectorized in row chunks: i must have an event and
        # T_j > T_i. Chunking keeps the temporary bool matrix at ~8 MB.
        ev = np.flatnonzero(E.astype(bool))
        if ev.size == 0:
            return float("nan")
        chunk = max(1, int(1_000_000 // max(1, n)))
        for start in range(0, ev.size, chunk):
            idx = ev[start:start + chunk]
            Ti = T[idx][:, None]
            Ri = R[idx][:, None]
            valid = T[None, :] > Ti          # j with longer survival
            conc += int(np.sum(valid & (Ri > R[None, :])))
            disc += int(np.sum(valid & (Ri < R[None, :])))
            ties += int(np.sum(valid & (Ri == R[None, :])))
        denom = conc + disc + 0.5 * ties
        return float(conc / denom) if denom else float("nan")


def _baseline_hazard(X, T, E, b):
    """Breslow baseline cumulative hazard at event times."""
    n = X.shape[0]
    eta = np.clip(X @ b, -50.0, 50.0)
    w = np.exp(eta)
    order = np.argsort(-T, kind="stable")
    To, Eo, wo = T[order], E[order], w[order]
    S0 = np.cumsum(wo)
    times, haz = [], []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and To[j + 1] == To[i]:
            j += 1
        d = Eo[i:j + 1].sum()
        if d > 0:
            times.append(float(To[i]))
            haz.append(float(d / S0[j]))
        i = j + 1
    times = np.asarray(times)
    return times, np.cumsum(np.asarray(haz))


def fit_coxph(df, X_cols, time_col, event_col, *, max_iter=50, tol=1e-7,
              memory_guard: Optional[MemoryGuard] = None) -> CoxPHModel:
    """Fit Cox PH by Newton-Raphson on the Breslow partial likelihood.

    One guarded streaming pre-pass collects X/T/E (check_state before the
    allocation), then Newton iterations run in memory. State = O(d^2).
    """
    cols = [X_cols] if isinstance(X_cols, str) else list(X_cols)
    if not cols:
        raise ValueError("fit_coxph: X_cols must be non-empty")
    for c in cols + [time_col, event_col]:
        if c not in df._columns:
            raise KeyError(f"fit_coxph: column {c!r} not in {df._columns}")
    mg = memory_guard or df._memory_guard
    Xs, Ts, Es = [], [], []
    n_rows = 0
    for b in df._iter_filtered_batches():
        mg.check_batch(b, name="coxph.pass1", extra_factor=2.0)
        Xs.append(_stack_num(b, cols))
        Ts.append(b.column(time_col).cast(pa.float64(), safe=False)
                  .to_numpy(zero_copy_only=False))
        Es.append(b.column(event_col).cast(pa.float64(), safe=False)
                  .to_numpy(zero_copy_only=False))
        n_rows += b.num_rows
        mg.check_state(n_rows * (len(cols) * 8 + 16), name="coxph.state")
    if n_rows == 0:
        raise ValueError("fit_coxph: empty input")
    X = np.vstack(Xs); T = np.concatenate(Ts); E = np.concatenate(Es)
    if np.isnan(X).any() or np.isnan(T).any():
        raise ValueError("fit_coxph: nulls in features or time")
    if (T <= 0).any():
        raise ValueError("fit_coxph: survival times must be > 0")
    n_events = int((E > 0).sum())
    if n_events == 0:
        raise ValueError("fit_coxph: no events (all censored)")
    d = X.shape[1]
    beta = np.zeros(d)
    converged = False
    n_iter = 0
    ll = 0.0
    for it in range(int(max_iter)):
        n_iter = it + 1
        grad, info, ll = _breslow(X, T, E, beta)
        try:
            step = np.linalg.solve(info + 1e-10 * np.eye(d), grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(info, grad, rcond=None)[0]
        beta = beta + step
        log_fit("coxph", iter_i=it + 1, iter_n=int(max_iter), loss=-ll,
                state_bytes=d * d * 8)
        if np.max(np.abs(step)) < float(tol):
            converged = True
            break
    _, info, ll = _breslow(X, T, E, beta)
    cov = np.linalg.pinv(info)
    bt, bh = _baseline_hazard(X, T, E, beta)
    model = CoxPHModel(beta, cov, bt, bh, cols, n_events, n_iter, converged, ll)
    mg.check_state(model.state_nbytes(), name="coxph.model")
    return model
