"""GMM density estimation (D.6.8, Phase D, v0.9.31).

Reference: Dempster, Laird, Rubin, *Maximum Likelihood from Incomplete
Data via the EM Algorithm*, JRSS-B 39(1), 1977. See sklearn.mixture.
GaussianMixture (BSD) for the standard API and validation.
Design: docs/decisions/d6_8_gmm_density.md.

Bounded-sample EM: reservoir sample of `max_samples` rows (default
10k), standard EM on the sample. State = O(k*d), independent of N;
scoring is per-batch with check_batch. True per-batch streaming EM
is a future D.6.x item (roadmap).
Diagonal covariance only (full cov is a future D.6.x item).
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.core.spill import SpillBuffer, AutoResult


def _log_gauss_diag(X, mu, var):
    """Log N(X | mu, diag(var)) per row for one component."""
    d = X.shape[1]
    # -0.5 * (d log(2pi) + sum log var + sum (x-mu)^2/var)
    diff = X - mu
    return -0.5 * (d * math.log(2 * math.pi)
                   + np.log(var).sum()
                   + np.sum(diff * diff / var, axis=1))


def _responsibilities(X, weights, means, covars):
    """Return (n, k) responsibilities matrix."""
    n = X.shape[0]
    k = weights.shape[0]
    logp = np.empty((n, k), dtype=np.float64)
    for j in range(k):
        logp[:, j] = math.log(max(weights[j], 1e-300)) + _log_gauss_diag(X, means[j], covars[j])
    m = logp.max(axis=1, keepdims=True)
    e = np.exp(logp - m)
    r = e / e.sum(axis=1, keepdims=True)
    log_lik = (m.ravel() + np.log(e.sum(axis=1)))
    return r, log_lik


class GMMModel:
    """Fitted Gaussian Mixture Model (diagonal covariance)."""

    def __init__(self, weights, means, covars, feature_names, n_iter, converged, log_lik):
        self.weights_ = np.asarray(weights, dtype=np.float64)
        self.means_ = np.asarray(means, dtype=np.float64)
        self.covariances_ = np.asarray(covars, dtype=np.float64)
        self.feature_names_ = list(feature_names)
        self.n_components_ = self.weights_.shape[0]
        self.n_iter_ = int(n_iter)
        self.converged_ = bool(converged)
        self.log_likelihood_ = float(log_lik)
        self.fitted_ = True

    def state_nbytes(self) -> int:
        return int((self.weights_.size + self.means_.size + self.covariances_.size) * 8)

    def _dense_batch(self, b, cols, mg):
        mg.check_batch(b, name="gmm.score_batch", extra_factor=3.0)
        arrays = []
        for c in cols:
            arr = b.column(c)
            if not (pa.types.is_integer(arr.type) or pa.types.is_floating(arr.type)):
                raise TypeError(f"gmm: column {c!r} must be numeric; got {arr.type}")
            arrays.append(arr.cast(pa.float64(), safe=False).to_numpy(zero_copy_only=False))
        return np.column_stack(arrays)

    def score_samples(self, df, X_cols=None, memory_guard=None, *,
                      output: str = "memory"):
        """v0.9.90-S4: output= for spill-on-demand (auto/disk).

        "memory" (default) → pa.Table.
        "auto"/"disk"      → AutoResult (iterable pa.RecordBatch);
                              .to_table() raises on spilled.
        """
        if output not in ("memory", "auto", "disk"):
            raise ValueError(
                f"gmm.score_samples: output must be 'memory', 'auto' "
                f"or 'disk'; got {output!r}")
        cols = list(X_cols) if X_cols is not None else self.feature_names_
        mg = memory_guard or MemoryGuard(max_memory_mb=64)
        schema = pa.schema([("log_likelihood", pa.float64())])

        if output == "memory":
            ll_parts: list[np.ndarray] = []
        else:
            buf = SpillBuffer(guard=mg, schema=schema, name="gmm.score",
                              spill_ratio=0.0 if output == "disk" else 0.5)

        for b in df._iter_filtered_batches():
            X = self._dense_batch(b, cols, mg)
            mask = ~np.isnan(X).any(axis=1)
            out = np.full(X.shape[0], np.nan, dtype=np.float64)
            if mask.any():
                _, ll = _responsibilities(X[mask], self.weights_, self.means_, self.covariances_)
                out[mask] = ll
            if output == "memory":
                ll_parts.append(out)
            else:
                rb = pa.RecordBatch.from_arrays(
                    [pa.array(out, type=pa.float64())], schema=schema)
                buf.write_batch(rb)

        if output == "memory":
            arr = np.concatenate(ll_parts) if ll_parts else np.zeros(0)
            return pa.table({"log_likelihood": pa.array(arr, type=pa.float64())})
        return buf.finalize()

    def score(self, df, X_cols=None, memory_guard=None) -> float:
        t = self.score_samples(df, X_cols, memory_guard)
        vals = t.column("log_likelihood").to_numpy(zero_copy_only=False)
        return float(np.nanmean(vals))

    def predict(self, df, X_cols=None, memory_guard=None, *,
                output: str = "memory"):
        """v0.9.90-S4: output= for spill-on-demand (auto/disk).

        "memory" (default) → pa.Table.
        "auto"/"disk"      → AutoResult.
        """
        if output not in ("memory", "auto", "disk"):
            raise ValueError(
                f"gmm.predict: output must be 'memory', 'auto' or "
                f"'disk'; got {output!r}")
        cols = list(X_cols) if X_cols is not None else self.feature_names_
        mg = memory_guard or MemoryGuard(max_memory_mb=64)
        schema = pa.schema([("cluster", pa.int64())])

        if output == "memory":
            lab_parts: list[np.ndarray] = []
        else:
            buf = SpillBuffer(guard=mg, schema=schema, name="gmm.predict",
                              spill_ratio=0.0 if output == "disk" else 0.5)

        for b in df._iter_filtered_batches():
            X = self._dense_batch(b, cols, mg)
            mask = ~np.isnan(X).any(axis=1)
            out = np.full(X.shape[0], -1, dtype=np.int64)
            if mask.any():
                r, _ = _responsibilities(X[mask], self.weights_, self.means_, self.covariances_)
                out[mask] = np.argmax(r, axis=1)
            if output == "memory":
                lab_parts.append(out)
            else:
                rb = pa.RecordBatch.from_arrays(
                    [pa.array(out, type=pa.int64())], schema=schema)
                buf.write_batch(rb)

        if output == "memory":
            arr = np.concatenate(lab_parts) if lab_parts else np.zeros(0, dtype=np.int64)
            return pa.table({"cluster": pa.array(arr, type=pa.int64())})
        return buf.finalize()


def _kmeanspp_init(X, k, rng):
    n, d = X.shape
    idx = rng.randint(0, n)
    centers = [X[idx].copy()]
    d2 = np.sum((X - centers[0])**2, axis=1)
    for _ in range(1, k):
        p = d2 / d2.sum()
        idx = int(rng.choice(n, p=p))
        centers.append(X[idx].copy())
        d2 = np.minimum(d2, np.sum((X - centers[-1])**2, axis=1))
    return np.asarray(centers)


def fit_gmm(df, X_cols, *, n_components=3, max_iter=100, tol=1e-4,
            reg_covar=1e-6, max_samples=10_000,
            random_state=None, memory_guard=None) -> GMMModel:
    """Fit a diagonal-covariance GMM via out-of-core EM.

    EM uses a bounded reservoir sample of `max_samples` rows (default
    10k) to run M-steps; per-iteration E-step would need O(N) counts
    otherwise. State = O(k*d) weights+means+variances — independent of N.
    """
    if not X_cols:
        raise ValueError("fit_gmm: X_cols must be non-empty")
    if n_components < 1:
        raise ValueError("fit_gmm: n_components must be >= 1")
    if max_samples < n_components:
        raise ValueError(f"fit_gmm: max_samples ({max_samples}) < n_components ({n_components})")

    mg = memory_guard or MemoryGuard(max_memory_mb=64)
    cols = list(X_cols)
    rng = np.random.RandomState(0 if random_state is None else int(random_state))

    # ---- Bounded reservoir sample of max_samples rows ----
    S = int(max_samples)
    sample = np.empty((S, len(cols)), dtype=np.float64)
    n_seen = 0; n_filled = 0
    for b in df._iter_filtered_batches():
        mg.check_batch(b, name="gmm.sample", extra_factor=3.0)
        arrays = []
        for c in cols:
            arr = b.column(c)
            if not (pa.types.is_integer(arr.type) or pa.types.is_floating(arr.type)):
                raise TypeError(f"gmm: column {c!r} must be numeric; got {arr.type}")
            arrays.append(arr.cast(pa.float64(), safe=False).to_numpy(zero_copy_only=False))
        X_b = np.column_stack(arrays)
        mask = ~np.isnan(X_b).any(axis=1)
        for row in X_b[mask]:
            if n_filled < S:
                sample[n_filled] = row; n_filled += 1
            else:
                j = int(rng.randint(0, n_seen + 1))
                if j < S:
                    sample[j] = row
            n_seen += 1

    if n_filled < n_components:
        raise ValueError(f"fit_gmm: fewer than n_components={n_components} valid rows")
    X = sample[:n_filled]
    n, d = X.shape

    # ---- Init (k-means++ centers + diagonal variances) ----
    centers = _kmeanspp_init(X, n_components, rng)
    # assign responsibilities by nearest center for initial variances
    d2 = np.stack([np.sum((X - centers[j])**2, axis=1) for j in range(n_components)], axis=1)
    labels = np.argmin(d2, axis=1)
    weights = np.zeros(n_components)
    means = centers.copy()
    covars = np.zeros((n_components, d))
    for j in range(n_components):
        members = X[labels == j]
        weights[j] = max(len(members) / n, 1e-3)
        if len(members) > 0:
            covars[j] = np.var(members, axis=0) + reg_covar
        else:
            covars[j] = np.ones(d) * reg_covar
    weights /= weights.sum()

    prev_ll = -np.inf
    converged = False
    n_iter = 0
    final_ll = 0.0
    for it in range(int(max_iter)):
        n_iter = it + 1
        # ---- E-step on the (bounded) sample ----
        r, ll = _responsibilities(X, weights, means, covars)
        cur_ll = float(ll.sum())
        # ---- M-step ----
        Nk = r.sum(axis=0) + 1e-12
        weights = Nk / n
        for j in range(n_components):
            means[j] = (r[:, j:j+1] * X).sum(axis=0) / Nk[j]
            diff = X - means[j]
            covars[j] = (r[:, j:j+1] * diff * diff).sum(axis=0) / Nk[j] + reg_covar
            covars[j] = np.maximum(covars[j], reg_covar)
        weights /= weights.sum()
        final_ll = cur_ll
        if it > 0 and abs(cur_ll - prev_ll) < tol * max(1.0, abs(prev_ll)):
            converged = True
            break
        prev_ll = cur_ll

    model = GMMModel(weights, means, covars, cols, n_iter, converged, final_ll)
    mg.check_state(model.state_nbytes(), name="gmm.state")
    return model
