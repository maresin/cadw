"""LinUCB disjoint linear bandit (D.6.7, v0.9.41).

Reference: Li, Chu, Langford, Schapire, *A Contextual-Bandit Approach to
Personalized News Article Recommendation*, WWW 2010. The per-arm ridge
update is closed form (A_a += x x^T, b_a += r x; theta_a = A_a^-1 b_a),
so the test reference is the analytic ridge solution, not a library.
See DECISION_LOG for algorithm reference.

State is O(K * d^2 + K * d) in the per-arm sufficient statistics
(A_a, b_a), independent of n. Fit streams over batches and updates the
per-arm statistics; no observation is retained.
"""
from __future__ import annotations

from cadw.ml._common import _stack_numeric as _stack_numeric_safe

from typing import Optional

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.ml._logging import log_fit




class LinUCBModel:
    """Fitted disjoint LinUCB: per-arm A_a, b_a and theta_a."""

    def __init__(self, A, b, feature_cols, params):
        self.A_ = np.asarray(A, dtype=np.float64)
        self.b_ = np.asarray(b, dtype=np.float64)
        self.feature_cols_ = list(feature_cols)
        self.params_ = dict(params)
        self.fitted_ = True

    @property
    def n_arms_(self):
        return int(self.A_.shape[0])

    def state_nbytes(self):
        return int(self.A_.nbytes + self.b_.nbytes + 256)

    def to_dict(self):
        K, d = self.A_.shape[0], self.A_.shape[1]
        return {"n_arms": K, "d": d, "params": self.params_}

    def theta(self):
        """Per-arm ridge coefficients theta_a = A_a^-1 b_a."""
        K = self.A_.shape[0]
        th = np.empty_like(self.b_)
        for a in range(K):
            th[a] = np.linalg.solve(self.A_[a], self.b_[a])
        return th

    def _ucb(self, X, alpha):
        th = self.theta()                      # (K, d)
        K, d = th.shape
        mean = X @ th.T                        # (n, K)
        out = np.empty_like(mean)
        for a in range(K):
            Ainv = np.linalg.inv(self.A_[a])
            # x^T Ainv x for every row, vectorised
            tmp = X @ Ainv
            var = np.einsum("ij,ij->i", tmp, X)
            out[:, a] = mean[:, a] + alpha * np.sqrt(np.maximum(var, 0.0))
        return out

    def predict(self, df, *, alpha=None, memory_guard=None,
                batch_size: "int | None" = None):
        """Per-row UCB scores, shape (n, n_arms)."""
        mg = memory_guard or df._memory_guard
        al = float(self.params_["alpha"] if alpha is None else alpha)
        # v0.9.110b: derive batch_size (OOC; model state loaded once, not
        # per batch -> state_bytes=0). expansion_factor=3.0 matches
        # _stack_numeric (1x) + _ucb transients (mean + out, up to 2x batch*K).
        if batch_size is None:
            row_size = mg.estimate_row_size_from_schema(df._dataset.schema)
            batch_size = mg.get_batch_rows(
                row_size_bytes=row_size,
                expansion_factor=3.0,
                state_bytes=0,
            )
        else:
            batch_size = int(batch_size)
        out = []
        for b in df._iter_filtered_batches(batch_size=batch_size):
            mg.check_batch(b, name="linucb.predict", extra_factor=3.0)
            out.append(self._ucb(_stack_numeric_safe(b, self.feature_cols_), al))
        return np.vstack(out) if out else np.zeros((0, self.n_arms_))

    def choose(self, df, *, alpha=None, memory_guard=None,
               batch_size: "int | None" = None):
        """Greedy UCB arm per row."""
        s = self.predict(df, alpha=alpha, memory_guard=memory_guard,
                         batch_size=batch_size)
        return np.argmax(s, axis=1).astype(np.int64) if s.size else np.zeros(0, np.int64)


def fit_linucb(df, X_cols, *, arm_col, reward_col, alpha=1.0, l2=1.0,
               memory_guard: Optional[MemoryGuard] = None,
               batch_size: "int | None" = None,
               k_max_hint: "int | None" = None):
    """Fit disjoint LinUCB sufficient statistics by streaming the log.

    OOC: batch_size=None -> derived via MemoryGuard.get_batch_rows.
    K is estimated from the first 1000-row sample (or k_max_hint if given).
    """
    cols = [X_cols] if isinstance(X_cols, str) else list(X_cols)
    if not cols:
        raise ValueError("fit_linucb: X_cols must be non-empty")
    if alpha <= 0:
        raise ValueError("fit_linucb: alpha must be > 0")
    if l2 <= 0:
        raise ValueError("fit_linucb: l2 must be > 0")
    if k_max_hint is not None and int(k_max_hint) < 1:
        raise ValueError("fit_linucb: k_max_hint must be >= 1")
    for c in cols + [arm_col, reward_col]:
        if c not in df._columns:
            raise KeyError(f"fit_linucb: column {c!r} not in {df._columns}")
    mg = memory_guard or df._memory_guard
    d = len(cols)

    # v0.9.110b: estimate K for state_est (K * d^2 + K * d).
    # K is a DATA property, not a parameter (unlike iforest S/n_trees).
    # Strategy (chosen v0.9.110b, see DEC-032):
    #   1. k_max_hint provided -> use directly.
    #   2. else sample first 1000 rows, K_est = (max_arm+1) * 2 (2x safety).
    #   3. empty df -> K_est = 2.
    # SAFETY: batch_size=1000 sample is a probe only (one pass, cheap).
    #   Under-estimate of K is safe: growth check_state (below) fires
    #   before newA alloc. Over-estimate -> smaller batch, slower but safe.
    if k_max_hint is not None:
        k_est = int(k_max_hint)
    else:
        try:
            first = next(df._iter_filtered_batches(batch_size=1000))
        except StopIteration:
            first = None
        if first is None or first.num_rows == 0:
            k_est = 2
        else:
            import pyarrow.compute as pc
            m = pc.max(first.column(arm_col))
            m_val = int(m.as_py()) if m.is_valid and m.as_py() is not None else 0
            k_est = max(2, (m_val + 1) * 2)

    if batch_size is None:
        # state_est = persistent (A, b) + per-arm transients.
        # Formula v1 (predicted, verify by experiment):
        #   2 * batch * row_size              (X stack 1x + X[mask] 1x)
        #   K * d * d * f64                    (A state)
        #   K * d * f64                        (b state)
        #   d * d * f64                        (Xa.T @ Xa temp, one arm)
        #   d * f64                            (Xa.T @ r[mask] temp)
        state_est = (
            k_est * d * d * 8
            + k_est * d * 8
            + d * d * 8
            + d * 8
            + 512
        )
        row_size = mg.estimate_row_size_from_schema(df._dataset.schema)
        # v0.9.110c: expansion_factor=4.0 (coef_batch=3.18 measured;
        # ceil(3.18*1.15)=4 per AI_CONSTITUTION AP23 §5.1 "round up to
        # nearest int"). Guard over-estimates peak by ~25% (safe side:
        # refuse-at-guard rather than silent OOM, §2 p.6).
        batch_size = mg.get_batch_rows(
            row_size_bytes=row_size,
            expansion_factor=4.0,
            state_bytes=state_est,
        )
    else:
        batch_size = int(batch_size)

    A = None
    b = None
    K = 0
    n = 0
    for batch_idx, batch in enumerate(df._iter_filtered_batches(batch_size=batch_size)):
        mg.mark(f"b{batch_idx}.start")
        mg.check_batch(batch, name="linucb.fit", extra_factor=3.0)
        X = _stack_numeric_safe(batch, cols)
        mg.mark(f"b{batch_idx}.after_X")
        if np.isnan(X).any():
            raise ValueError("fit_linucb: nulls in features")
        # v0.9.110b: cast+to_numpy avoids to_pylist Python-object overhead.
        arms = batch.column(arm_col).cast(pa.int64(), safe=False).to_numpy(
            zero_copy_only=False)
        mg.mark(f"b{batch_idx}.after_arms")
        r = _stack_numeric_safe(batch, [reward_col]).ravel()
        mg.mark(f"b{batch_idx}.after_r")
        if np.isnan(r).any():
            raise ValueError("fit_linucb: nulls in reward")
        if arms.size and arms.min() < 0:
            raise ValueError("fit_linucb: arm ids must be >= 0")
        k_here = int(arms.max()) + 1 if arms.size else 0
        if k_here > K:
            # AP4 fix: check_state BEFORE newA alloc (was after).
            mg.check_state(
                k_here * d * d * 8 + k_here * d * 8 + 512,
                name="linucb.state_grow",
            )
            newA = np.zeros((k_here, d, d), dtype=np.float64)
            newb = np.zeros((k_here, d), dtype=np.float64)
            if K:
                newA[:K] = A
                newb[:K] = b
            A, b, K = newA, newb, k_here
            mg.mark(f"b{batch_idx}.after_grow")
        for a in range(K):
            mask = arms == a
            if not mask.any():
                continue
            Xa = X[mask]
            A[a] += Xa.T @ Xa
            b[a] += Xa.T @ r[mask]
        mg.mark(f"b{batch_idx}.end")
        n += X.shape[0]
    if A is None or K == 0:
        raise ValueError("fit_linucb: no observations")
    A = A + float(l2) * np.eye(d)[None, :, :]
    model = LinUCBModel(A, b, cols, {"alpha": alpha, "l2": l2,
                                     "n_arms": K, "n_obs": n})
    log_fit("linucb", state_bytes=model.state_nbytes())
    return model
