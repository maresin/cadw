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

import tempfile
import shutil
from typing import Optional, Sequence

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.io.batch_reader import ArrowIPCBatchReader
from cadw.ml._logging import log_fit
from cadw.ops.sort import external_sort


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
                 n_events, n_iter, converged, loglik, solver="closed"):
        self.coef_ = np.asarray(coef, dtype=np.float64)
        # v0.9.93: solver="sgd" returns cov=None (would be O(d^2) to
        # compute; info/pinv impossible at wide d). Documented limitation:
        # use solver="closed" for inference (d <= 1000).
        self.covariance_ = (None if cov is None
                            else np.asarray(cov, dtype=np.float64))
        self.solver_ = str(solver)
        self.baseline_t_ = np.asarray(baseline_t, dtype=np.float64)
        self.baseline_h_ = np.asarray(baseline_h, dtype=np.float64)
        self.feature_names_ = list(feature_names)
        self.n_events_ = int(n_events)
        self.n_iter_ = int(n_iter)
        self.converged_ = bool(converged)
        self.log_likelihood_ = float(loglik)
        self.fitted_ = True

    def state_nbytes(self):
        base = (self.coef_.nbytes + self.baseline_t_.nbytes
                + self.baseline_h_.nbytes + 256)
        if self.covariance_ is not None:
            base += self.covariance_.nbytes
        return int(base)

    def to_dict(self):
        return {"coef": self.coef_.tolist(),
                "feature_names": self.feature_names_,
                "n_events": self.n_events_, "n_iter": self.n_iter_,
                "converged": self.converged_,
                "log_likelihood": self.log_likelihood_,
                "solver": self.solver_}

    def predict_risk(self, df, X_cols=None, *, output: str = "auto"):
        """Predict risk scores (higher = shorter survival).

        v0.9.94: output="auto" default for spill-on-demand.

        output="memory": returns pa.Table with "risk" column
          (back-compat; O(n) RAM).
        output="auto"/"disk": returns AutoArrayResult (1-D numpy-like;
          spill when guard exhausted). Iterate via `for arr in result`.
          to_numpy() on spilled raises MemoryLimitExceeded (§1.3).
        """
        if output not in ("memory", "auto", "disk"):
            raise ValueError(
                f"predict_risk: output must be 'memory', 'auto' or "
                f"'disk'; got {output!r}")
        cols = list(X_cols) if X_cols is not None else self.feature_names_

        if output == "memory":
            out = []
            for b in df._iter_filtered_batches():
                out.append(_stack_num(b, cols) @ self.coef_)
            arr = np.concatenate(out) if out else np.zeros(0)
            return pa.table({"risk": pa.array(arr, type=pa.float64())})

        from cadw.core.spill import SpillArray
        mg = getattr(df, "_memory_guard", None)
        buf = SpillArray(
            guard=mg, width=1, dtype=np.float64,
            name="coxph.predict_risk",
            spill_ratio=0.0 if output == "disk" else 0.5)
        for b in df._iter_filtered_batches():
            buf.write_array(_stack_num(b, cols) @ self.coef_)
        return buf.finalize()

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


def _breslow_streaming(sorted_reader, X_cols, time_col, event_col, b, mg,
                       *, compute_info: bool = True):
    """Streaming Breslow reduce (v0.9.87-P5a-fix-coxph).

    Bit-identical to `_breslow` for realistic data (ties order-invariant
    under associativity). RAM: O(batch*d + d^2 + pending_ties*d). Guard
    refuses if pending ties block exceeds budget (pathological all-ties).
    """
    d = len(X_cols)
    S0 = 0.0
    S1 = np.zeros(d)
    S2 = np.zeros((d, d))
    grad = np.zeros(d)
    # v0.9.93: compute_info=False skips O(d^2) allocation for SGD path
    # (full-batch GD needs only grad; info is used only by Newton step).
    info = np.zeros((d, d)) if compute_info else None
    ll = 0.0
    n_events = 0

    pending_Xs = []
    pending_Es = []
    pending_T = None
    pending_bytes = 0
    pending_budget = mg.state_limit_bytes // 8

    def _flush_block():
        nonlocal S0, S1, S2, grad, info, ll, n_events, pending_bytes
        if not pending_Xs:
            return
        Xb = np.vstack(pending_Xs) if len(pending_Xs) > 1 else pending_Xs[0]
        Eb = (np.concatenate(pending_Es) if len(pending_Es) > 1
              else pending_Es[0])
        eta = np.clip(Xb @ b, -50.0, 50.0)
        w = np.exp(eta)
        # Breslow: risk set = T >= t; add ALL block rows to S first.
        S0 = S0 + float(w.sum())
        S1 = S1 + w @ Xb
        S2 = S2 + Xb.T @ (w[:, None] * Xb)
        ev_mask = Eb.astype(bool)
        n_ev_b = int(ev_mask.sum())
        if n_ev_b > 0:
            Xe = Xb[ev_mask]
            mu = S1 / S0
            grad += Xe.sum(axis=0) - n_ev_b * mu
            if compute_info:
                info += n_ev_b * (S2 / S0 - np.outer(mu, mu))
            ll += float((Xe @ b).sum()) - n_ev_b * float(np.log(S0))
            n_events += n_ev_b
        pending_Xs.clear()
        pending_Es.clear()
        pending_bytes = 0

    for batch in sorted_reader:
        bn = batch.num_rows
        if bn == 0:
            continue
        Xb = _stack_num(batch, X_cols)
        Tb = batch.column(time_col).cast(
            pa.float64(), safe=False).to_numpy(zero_copy_only=False)
        Eb = batch.column(event_col).cast(
            pa.float64(), safe=False).to_numpy(zero_copy_only=False)
        if bn > 1:
            starts = np.concatenate(
                [[0], np.flatnonzero(np.diff(Tb) != 0) + 1])
            ends = np.concatenate([starts[1:], [bn]])
        else:
            starts = np.array([0])
            ends = np.array([bn])
        for s, e in zip(starts, ends):
            t_this = Tb[s]
            if pending_T is not None and t_this != pending_T:
                _flush_block()
                pending_T = None
            if pending_T is None:
                pending_T = t_this
            xs = Xb[s:e]
            es = Eb[s:e]
            pending_Xs.append(xs)
            pending_Es.append(es)
            pending_bytes += xs.nbytes + es.nbytes
            if pending_bytes > pending_budget:
                raise MemoryLimitExceeded(
                    f"coxph: ties block at T={t_this} exceeds budget "
                    f"({pending_bytes} > {pending_budget}); pathological "
                    f"ties distribution, use coarser time resolution.")
    _flush_block()
    return grad, info, ll, n_events


def _baseline_hazard_streaming(sorted_reader, X_cols, time_col, event_col,
                                b, mg):
    """Streaming Breslow baseline hazard (v0.9.87-P5a-fix-coxph).

    Output arrays are O(n_distinct_event_times) <= O(n_events). Caller
    must guard-check 2*n_events*8 bytes before invoking.
    """
    S0 = 0.0
    times = []
    haz = []
    pending_Xs = []
    pending_Es = []
    pending_T = None
    pending_bytes = 0
    pending_budget = mg.state_limit_bytes // 8

    def _flush():
        nonlocal S0, pending_bytes
        if not pending_Xs:
            return
        Xb = np.vstack(pending_Xs) if len(pending_Xs) > 1 else pending_Xs[0]
        Eb = (np.concatenate(pending_Es) if len(pending_Es) > 1
              else pending_Es[0])
        eta = np.clip(Xb @ b, -50.0, 50.0)
        w = np.exp(eta)
        S0 = S0 + float(w.sum())
        d_ev = float(Eb.sum())
        if d_ev > 0.0:
            times.append(float(pending_T))
            haz.append(d_ev / S0)
        pending_Xs.clear()
        pending_Es.clear()
        pending_bytes = 0

    for batch in sorted_reader:
        bn = batch.num_rows
        if bn == 0:
            continue
        Xb = _stack_num(batch, X_cols)
        Tb = batch.column(time_col).cast(
            pa.float64(), safe=False).to_numpy(zero_copy_only=False)
        Eb = batch.column(event_col).cast(
            pa.float64(), safe=False).to_numpy(zero_copy_only=False)
        if bn > 1:
            starts = np.concatenate(
                [[0], np.flatnonzero(np.diff(Tb) != 0) + 1])
            ends = np.concatenate([starts[1:], [bn]])
        else:
            starts = np.array([0])
            ends = np.array([bn])
        for s, e in zip(starts, ends):
            t_this = Tb[s]
            if pending_T is not None and t_this != pending_T:
                _flush()
                pending_T = None
            if pending_T is None:
                pending_T = t_this
            xs = Xb[s:e]
            es = Eb[s:e]
            pending_Xs.append(xs)
            pending_Es.append(es)
            pending_bytes += xs.nbytes + es.nbytes
            if pending_bytes > pending_budget:
                raise MemoryLimitExceeded(
                    f"coxph: ties block at T={t_this} exceeds budget")
    _flush()
    times_arr = np.asarray(times, dtype=np.float64) if times else np.zeros(0)
    haz_arr = (np.cumsum(np.asarray(haz, dtype=np.float64))
               if haz else np.zeros(0))
    return times_arr, haz_arr


def _sgd_coxph(sorted_path, X_cols, time_col, event_col, sgd_lr, sgd_epochs, mg):
    """Full-batch gradient ASCENT on Breslow partial likelihood.

    Cox Breslow gradient at event i requires risk-set sum over ALL
    samples with T >= t_i. Cannot online-SGD by batch — needs one full
    pass per epoch (external sorted reader). State: O(d) for beta.

    Returns (best_beta, best_ll). Tracks best over epochs (partial LL
    is concave, but plain GD can overshoot; best-LL is monotone-safe).

    Note: gradient is dLL/dbeta (ascent direction). Newton uses
    beta += solve(info, grad); we use beta += lr*grad/n_ev.
    Division by n_ev makes lr scale-invariant w.r.t. n_events.
    """
    d = len(X_cols)
    mg.check_state(2 * d * 8 + 64, name="coxph.sgd.state")
    beta = np.zeros(d, dtype=np.float64)
    best_ll = -np.inf
    best_beta = beta.copy()
    for _epoch in range(int(sgd_epochs)):
        reader = ArrowIPCBatchReader(sorted_path, None)
        try:
            grad, _info, ll, n_ev = _breslow_streaming(
                reader, X_cols, time_col, event_col, beta, mg,
                compute_info=False)
        finally:
            reader.close()
        if ll > best_ll:
            best_ll = ll
            best_beta = beta.copy()
        # Ascent: partial LL is maximized at true beta.
        beta = beta + sgd_lr * grad / max(n_ev, 1)
    return best_beta, best_ll


def fit_coxph(df, X_cols, time_col, event_col, *, max_iter=50, tol=1e-7,
              solver: str = "auto",
              sgd_lr: float = 0.1,
              sgd_epochs: int = 100,
              memory_guard: Optional[MemoryGuard] = None,
              spill_dir: Optional[str] = None) -> CoxPHModel:
    """Fit Cox PH by Newton-Raphson on the Breslow partial likelihood.

    v0.9.87-P5a-fix-coxph: true streaming implementation.

    Pre-pass counts n_rows/n_events. External sort by T (desc). Newton
    iterations stream sorted file, accumulating S0/S1/S2 per ties block
    (Breslow: risk set = T >= t, block fully added before event
    contributions). State RAM: O(d^2 + pending_ties*d), independent of n.
    Final baseline arrays are O(n_events); guard-checked before alloc.

    Bit-identical to prior in-RAM implementation (to float-assoc tol).
    """
    cols = [X_cols] if isinstance(X_cols, str) else list(X_cols)
    if not cols:
        raise ValueError("fit_coxph: X_cols must be non-empty")
    for c in cols + [time_col, event_col]:
        if c not in df._columns:
            raise KeyError(f"fit_coxph: column {c!r} not in {df._columns}")
    mg = memory_guard or df._memory_guard
    d = len(cols)

    # State guard: 3 d^2 matrices (S2, info, outer(mu,mu)) + pending ties.
    pending_budget = mg.state_limit_bytes // 8
    est_state = 3 * d * d * 8 + 5 * pending_budget
    mg.check_state(est_state, name="coxph.state")

    if spill_dir is None:
        _spill_root = tempfile.mkdtemp(prefix="coxph_")
        _cleanup = True
    else:
        _spill_root = spill_dir
        _cleanup = False

    try:
        # Pre-pass: n_rows + n_events (streaming, no n-sized alloc).
        n_rows = 0
        n_events = 0
        for b in df._iter_filtered_batches():
            mg.check_batch(b, name="coxph.pass1", extra_factor=2.0)
            e_arr = b.column(event_col).cast(
                pa.float64(), safe=False).to_numpy(zero_copy_only=False)
            n_rows += b.num_rows
            n_events += int((e_arr > 0).sum())
        if n_rows == 0:
            raise ValueError("fit_coxph: empty input")
        if n_events == 0:
            raise ValueError("fit_coxph: no events (all censored)")

        def _checked_iter():
            for b in df._iter_filtered_batches():
                Xb = _stack_num(b, cols)
                Tb = b.column(time_col).cast(
                    pa.float64(), safe=False).to_numpy(zero_copy_only=False)
                if np.isnan(Xb).any() or np.isnan(Tb).any():
                    raise ValueError(
                        "fit_coxph: nulls in features or time")
                if (Tb <= 0).any():
                    raise ValueError(
                        "fit_coxph: survival times must be > 0")
                yield b

        value_cols = list(cols) + [time_col, event_col]
        sorted_path = external_sort(
            _checked_iter(), [time_col], value_cols,
            mg, f"{_spill_root}/sort", ascending=False, log=False)

        # v0.9.93: solver routing.
        if solver not in ("auto", "closed", "sgd"):
            raise ValueError(
                f"fit_coxph: solver must be 'auto'|'closed'|'sgd'; "
                f"got {solver!r}")
        if solver == "auto":
            if mg is None or d <= 1000:
                solver = "closed"
            else:
                est_closed = 3 * d * d * 8
                if hasattr(mg, "available_bytes") and callable(
                        mg.available_bytes):
                    try:
                        avail = mg.available_bytes()
                        solver = ("sgd" if est_closed > int(avail * 0.5)
                                  else "closed")
                    except Exception:
                        solver = "closed"
                else:
                    solver = "closed"

        if solver == "sgd":
            if sgd_lr <= 0:
                raise ValueError(
                    f"fit_coxph: sgd_lr must be > 0; got {sgd_lr}")
            if sgd_epochs < 1:
                raise ValueError(
                    f"fit_coxph: sgd_epochs must be >= 1; got {sgd_epochs}")
            beta, ll = _sgd_coxph(
                sorted_path, cols, time_col, event_col,
                sgd_lr, sgd_epochs, mg)
            # Baseline hazard: O(n_events) state, guard before alloc.
            mg.check_state(2 * n_events * 8 + 256,
                           name="coxph.baseline")
            reader = ArrowIPCBatchReader(sorted_path, None)
            try:
                bt, bh = _baseline_hazard_streaming(
                    reader, cols, time_col, event_col, beta, mg)
            finally:
                reader.close()
            model = CoxPHModel(beta, None, bt, bh, cols, n_events,
                               sgd_epochs, True, ll, solver="sgd")
            mg.check_state(model.state_nbytes(), name="coxph.model")
            return model

        # Newton-Raphson (solver="closed" or resolved "auto").
        beta = np.zeros(d)
        converged = False
        n_iter = 0
        ll = 0.0
        for it in range(int(max_iter)):
            n_iter = it + 1
            reader = ArrowIPCBatchReader(sorted_path, None)
            try:
                grad, info, ll, _n_ev = _breslow_streaming(
                    reader, cols, time_col, event_col, beta, mg)
            finally:
                reader.close()
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

        # Final info at converged beta.
        reader = ArrowIPCBatchReader(sorted_path, None)
        try:
            _, info, ll, _n_ev = _breslow_streaming(
                reader, cols, time_col, event_col, beta, mg)
        finally:
            reader.close()
        cov = np.linalg.pinv(info)

        # Baseline arrays O(n_events); guard before allocation.
        mg.check_state(3 * d * d * 8 + 2 * n_events * 8,
                       name="coxph.baseline")
        reader = ArrowIPCBatchReader(sorted_path, None)
        try:
            bt, bh = _baseline_hazard_streaming(
                reader, cols, time_col, event_col, beta, mg)
        finally:
            reader.close()

        model = CoxPHModel(beta, cov, bt, bh, cols, n_events, n_iter,
                           converged, ll, solver="closed")
        mg.check_state(model.state_nbytes(), name="coxph.model")
        return model
    finally:
        if _cleanup:
            shutil.rmtree(_spill_root, ignore_errors=True)
