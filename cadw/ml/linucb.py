"""LinUCB disjoint linear bandit (D.6.7, v0.9.41).

Reference: Li, Chu, Langford, Schapire, *A Contextual-Bandit Approach to
Personalized News Article Recommendation*, WWW 2010. The per-arm ridge
update is closed form (A_a += x x^T, b_a += r x; theta_a = A_a^-1 b_a),
so the test reference is the analytic ridge solution, not a library.
See .pingis/algorithm_notes/d6_7_linucb.md.

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

    def predict(self, df, *, alpha=None, memory_guard=None):
        """Per-row UCB scores, shape (n, n_arms)."""
        mg = memory_guard or df._memory_guard
        al = float(self.params_["alpha"] if alpha is None else alpha)
        out = []
        for b in df._iter_filtered_batches():
            mg.check_batch(b, name="linucb.predict", extra_factor=3.0)
            out.append(self._ucb(_stack_numeric_safe(b, self.feature_cols_), al))
        return np.vstack(out) if out else np.zeros((0, self.n_arms_))

    def choose(self, df, *, alpha=None, memory_guard=None):
        """Greedy UCB arm per row."""
        s = self.predict(df, alpha=alpha, memory_guard=memory_guard)
        return np.argmax(s, axis=1).astype(np.int64) if s.size else np.zeros(0, np.int64)


def fit_linucb(df, X_cols, *, arm_col, reward_col, alpha=1.0, l2=1.0,
               memory_guard: Optional[MemoryGuard] = None):
    """Fit disjoint LinUCB sufficient statistics by streaming the log."""
    cols = [X_cols] if isinstance(X_cols, str) else list(X_cols)
    if not cols:
        raise ValueError("fit_linucb: X_cols must be non-empty")
    if alpha <= 0:
        raise ValueError("fit_linucb: alpha must be > 0")
    if l2 <= 0:
        raise ValueError("fit_linucb: l2 must be > 0")
    for c in cols + [arm_col, reward_col]:
        if c not in df._columns:
            raise KeyError(f"fit_linucb: column {c!r} not in {df._columns}")
    mg = memory_guard or df._memory_guard
    d = len(cols)
    A = None
    b = None
    K = 0
    n = 0
    for batch in df._iter_filtered_batches():
        mg.check_batch(batch, name="linucb.fit", extra_factor=3.0)
        X = _stack_numeric_safe(batch, cols)
        if np.isnan(X).any():
            raise ValueError("fit_linucb: nulls in features")
        arms = np.asarray(batch.column(arm_col).to_pylist(), dtype=np.int64)
        r = _stack_numeric_safe(batch, [reward_col]).ravel()
        if np.isnan(r).any():
            raise ValueError("fit_linucb: nulls in reward")
        if arms.size and arms.min() < 0:
            raise ValueError("fit_linucb: arm ids must be >= 0")
        k_here = int(arms.max()) + 1 if arms.size else 0
        if k_here > K:
            newA = np.zeros((k_here, d, d), dtype=np.float64)
            newb = np.zeros((k_here, d), dtype=np.float64)
            if K:
                newA[:K] = A
                newb[:K] = b
            A, b, K = newA, newb, k_here
        for a in range(K):
            mask = arms == a
            if not mask.any():
                continue
            Xa = X[mask]
            A[a] += Xa.T @ Xa
            b[a] += Xa.T @ r[mask]
        n += X.shape[0]
        mg.check_state(K * d * d * 8 + K * d * 8 + 512, name="linucb.state")
    if A is None or K == 0:
        raise ValueError("fit_linucb: no observations")
    A = A + float(l2) * np.eye(d)[None, :, :]
    model = LinUCBModel(A, b, cols, {"alpha": alpha, "l2": l2,
                                     "n_arms": K, "n_obs": n})
    log_fit("linucb", state_bytes=model.state_nbytes())
    mg.check_state(model.state_nbytes(), name="linucb.model")
    return model
