"""Randomized SVD / truncated PCA (D.6.4, v0.9.40).

Reference: Halko, Martinsson, Tropp, *Finding Structure with Randomness:
Probabilistic Algorithms for Constructing Approximate Matrix
Decompositions*, SIAM Review 53(2), 2011; scikit-learn TruncatedSVD /
PCA as reference implementation.
See .pingis/algorithm_notes/d6_4_randsvd.md.

Streaming randomized range finder: draw an (d x (k+p)) Gaussian test
matrix, accumulate Y = X @ Omega over mini-batches, then do a small QR/SVD
on the d x (k+p) sketch. The full N x d matrix is never materialised;
state is O(d * (k+p)) plus the running Y. One power iteration by default
for accuracy on decaying spectra.
"""
from __future__ import annotations

from cadw.ml._common import _stack_numeric as _stack_numeric_safe

from typing import Optional

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.core.spill import SpillArray, AutoArrayResult
from cadw.ml._logging import log_fit




class RandomizedSVDModel:
    """Fitted randomized SVD (PCA-like projection)."""

    def __init__(self, components, singular_values, mean, feature_cols, params):
        self.components_ = np.asarray(components, dtype=np.float64)
        self.singular_values_ = np.asarray(singular_values, dtype=np.float64)
        self.mean_ = np.asarray(mean, dtype=np.float64)
        self.feature_cols_ = list(feature_cols)
        self.params_ = dict(params)
        self.fitted_ = True

    def state_nbytes(self):
        return int(self.components_.nbytes + self.singular_values_.nbytes
                   + self.mean_.nbytes + 256)

    def to_dict(self):
        return {"n_components": int(self.components_.shape[0]),
                "singular_values": self.singular_values_.tolist(),
                "params": self.params_}

    def explained_variance_ratio(self, df, memory_guard=None):
        mg = memory_guard or df._memory_guard
        total = 0.0
        n = 0
        for b in df._iter_filtered_batches():
            mg.check_batch(b, name="randsvd.explained", extra_factor=3.0)
            X = _stack_numeric_safe(b, self.feature_cols_) - self.mean_
            total += float(np.sum(X * X))
            n += X.shape[0]
        ev = self.singular_values_ ** 2
        return ev / max(total, 1e-300) if n else ev

    def transform(self, df, memory_guard=None, *, output: str = "auto"):
        """v0.9.95 (BREAKING): unified CadwDataFrame return.
        output= selects storage strategy only.
        Columns: ["pc0", ..., "pc{k-1}"].
        """
        if output not in ("memory", "auto", "disk"):
            raise ValueError(
                f"randsvd.transform: output must be 'memory', "
                f"'auto' or 'disk'; got {output!r}")
        mg = memory_guard or df._memory_guard
        k = int(self.components_.shape[0])
        if output == "memory":
            out = []
        else:
            buf = SpillArray(guard=mg, width=k, dtype=np.float64,
                             name="randsvd.transform",
                             spill_ratio=0.0 if output == "disk" else 0.5)
        for b in df._iter_filtered_batches():
            mg.check_batch(b, name="randsvd.transform", extra_factor=3.0)
            X = _stack_numeric_safe(b, self.feature_cols_) - self.mean_
            proj_b = X @ self.components_.T
            if output == "memory":
                out.append(proj_b)
            else:
                buf.write_array(proj_b)
        if output == "memory":
            raw = np.vstack(out) if out else np.zeros((0, k))
        else:
            raw = buf.finalize()
        # v0.9.95 (BREAKING): unified CadwDataFrame return.
        from cadw.dataframe import CadwDataFrame
        return CadwDataFrame.from_producer_result(
            raw, col_names=[f"pc{i}" for i in range(k)],
            memory_guard=mg)


def fit_randomized_svd(df, X_cols, *, n_components=8, n_oversamples=10,
                       n_power_iter=1, seed=0,
                       memory_guard: Optional[MemoryGuard] = None,
                       batch_size: "int | None" = None):
    """Fit a randomized SVD on the column block X_cols."""
    cols = [X_cols] if isinstance(X_cols, str) else list(X_cols)
    if not cols:
        raise ValueError("fit_randomized_svd: X_cols must be non-empty")
    if n_components < 1:
        raise ValueError("fit_randomized_svd: n_components must be >= 1")
    for c in cols:
        if c not in df._columns:
            raise KeyError(f"fit_randomized_svd: column {c!r} not in {df._columns}")
    mg = memory_guard or df._memory_guard
    d = len(cols)
    p = int(n_oversamples)
    L = min(d, n_components + p)
    # v0.9.109c: derive batch_size from state-aware guard.
    # State = mean(d) + omega(d,L) + Y(d,L) + Q(d,L) + B(L,L)
    #       = 5*d*L*8 + L^2*8. Peak = 3*batch*row_size (column_stack 2x
    # + X-mean subtraction 1x; measured coef=3.0 in sweep) + state.
    # See EXPERIMENT_PROCEDURE.md §3.
    if batch_size is None:
        state_est = 5 * d * L * 8 + L * L * 8 + 256
        row_size = mg.estimate_row_size_from_schema(df._dataset.schema)
        batch_size = mg.get_batch_rows(
            row_size_bytes=row_size,
            expansion_factor=3.0,   # matches measured coef=3.0
            state_bytes=state_est,
        )
    else:
        batch_size = int(batch_size)
    rng = np.random.RandomState(seed)
    # streaming mean
    n = 0
    s = np.zeros(d, dtype=np.float64)
    for b in df._iter_filtered_batches(batch_size=batch_size):
        mg.check_batch(b, name="randsvd.mean", extra_factor=3.0)
        X = _stack_numeric_safe(b, cols)
        if np.isnan(X).any():
            raise ValueError("fit_randomized_svd: nulls in features")
        s += X.sum(axis=0)
        n += X.shape[0]
    if n == 0:
        raise ValueError("fit_randomized_svd: empty input")
    mean = s / n
    omega = rng.normal(size=(d, L))
    Y = np.zeros((d, L), dtype=np.float64)
    for b in df._iter_filtered_batches(batch_size=batch_size):
        mg.check_batch(b, name="randsvd.pass1", extra_factor=3.0)
        X = _stack_numeric_safe(b, cols) - mean
        Y += X.T @ (X @ omega)
    # Y = X^T X Omega; one power iteration already folded into Y.
    for _ in range(max(0, int(n_power_iter) - 1)):
        Yp = np.zeros((d, L), dtype=np.float64)
        for b in df._iter_filtered_batches(batch_size=batch_size):
            mg.check_batch(b, name="randsvd.power", extra_factor=3.0)
            X = _stack_numeric_safe(b, cols) - mean
            Yp += X.T @ (X @ Y)
        Y = Yp
    Q, _ = np.linalg.qr(Y)
    B = np.zeros((L, L), dtype=np.float64)
    for b in df._iter_filtered_batches(batch_size=batch_size):
        mg.check_batch(b, name="randsvd.pass2", extra_factor=3.0)
        X = _stack_numeric_safe(b, cols) - mean
        XQ = X @ Q
        B += XQ.T @ XQ
    Uhat, sv, _ = np.linalg.svd(B, full_matrices=False)
    U = Q @ Uhat
    k = min(int(n_components), U.shape[1])
    components = U[:, :k].T
    singular_values = np.sqrt(np.maximum(sv[:k], 0.0))
    model = RandomizedSVDModel(components, singular_values, mean, cols,
                               {"n_components": k,
                                "n_oversamples": p,
                                "n_power_iter": n_power_iter,
                                "seed": seed})
    log_fit("randomized_svd", state_bytes=model.state_nbytes())
    mg.check_state(model.state_nbytes(), name="randsvd.model")
    return model
