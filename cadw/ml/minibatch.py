"""MiniBatch KMeans and streaming EM GMM (D.5.3, v0.9.39).

Reference: Sculley, *Web-Scale K-Means Clustering*, WWW 2010 (MiniBatch
KMeans with per-centroid streaming updates); Dempster, Laird, Rubin,
*Maximum Likelihood from Incomplete Data via the EM Algorithm*, JRSS-B
39(1), 1977 (EM); Pedregosa et al., *scikit-learn*, JMLR 12, 2011
(reference implementations used in the tests).
See DECISION_LOG for algorithm reference.

Two streaming estimators share the same mini-batch contract:

  MiniBatchKMeans  - Sculley per-centroid streaming centroid update with
                     per-centroid counts; one pass over mini-batches.
  MiniBatchGMM     - diagonal-covariance EM with sufficient statistics
                     accumulated over mini-batches (sum r, sum r*x,
                     sum r*x^2), seeded from MiniBatchKMeans centroids.

State is O(K*d) in both cases, independent of n. Every batch passes
through check_batch(extra_factor=3.0); a bounded reservoir is never
materialised -- only the per-component sufficient statistics.
"""
from __future__ import annotations

from cadw.ml._common import _stack_numeric as _stack_numeric_safe

from typing import Optional

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.ml._logging import log_fit




def _kmeanspp(X, k, rng):
    n = X.shape[0]
    idx = [int(rng.randint(n))]
    d2 = np.sum((X - X[idx[0]]) ** 2, axis=1)
    for _ in range(1, k):
        tot = d2.sum()
        if tot <= 0:
            idx.append(int(rng.randint(n)))
            continue
        probs = d2 / tot
        idx.append(int(rng.choice(n, p=probs)))
        new = np.sum((X - X[idx[-1]]) ** 2, axis=1)
        d2 = np.minimum(d2, new)
    return X[np.array(idx)].copy()


def _assign(X, centers):
    """Squared Euclidean distances and argmin labels."""
    d2 = (np.sum(X * X, axis=1)[:, None]
          - 2.0 * X @ centers.T
          + np.sum(centers * centers, axis=1)[None, :])
    d2 = np.maximum(d2, 0.0)
    return d2, np.argmin(d2, axis=1)


def _iter_batches(X, batch_size):
    n = X.shape[0]
    for s in range(0, n, batch_size):
        yield X[s:s + batch_size]


class MiniBatchKMeansModel:
    """Fitted MiniBatch KMeans (Sculley 2010)."""

    def __init__(self, centers, counts, feature_cols, params):
        self.centers_ = np.asarray(centers, dtype=np.float64)
        self.counts_ = np.asarray(counts, dtype=np.float64)
        self.feature_cols_ = list(feature_cols)
        self.params_ = dict(params)
        self.fitted_ = True

    def state_nbytes(self):
        return int(self.centers_.nbytes + self.counts_.nbytes + 256)

    def to_dict(self):
        return {"k": int(self.centers_.shape[0]),
                "d": int(self.centers_.shape[1]),
                "params": self.params_}

    def _predict_array(self, X):
        _, lab = _assign(X, self.centers_)
        return lab.astype(np.int64)

    def predict(self, df, memory_guard=None):
        mg = memory_guard or df._memory_guard
        out = []
        for b in df._iter_filtered_batches(batch_size=io_batch_size):
            mg.check_batch(b, name="minibatch.predict", extra_factor=3.0)
            out.append(self._predict_array(_stack_numeric_safe(b, self.feature_cols_)))
        return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)

    def inertia(self, df, memory_guard=None):
        """Sum of squared distances to the assigned centroid (sklearn units)."""
        mg = memory_guard or df._memory_guard
        total = 0.0
        n = 0
        for b in df._iter_filtered_batches(batch_size=io_batch_size):
            mg.check_batch(b, name="minibatch.inertia", extra_factor=3.0)
            X = _stack_numeric_safe(b, self.feature_cols_)
            d2, lab = _assign(X, self.centers_)
            total += float(np.sum(d2[np.arange(X.shape[0]), lab]))
            n += X.shape[0]
        return total if n else 0.0


def fit_minibatch_kmeans(df, X_cols, *, n_clusters=8, batch_size=1024,
                         max_iter=100, tol=1e-4, seed=0,
                         io_batch_size: "int | None" = None,
                         memory_guard: Optional[MemoryGuard] = None):
    """Fit MiniBatch KMeans with per-centroid streaming updates."""
    cols = [X_cols] if isinstance(X_cols, str) else list(X_cols)
    if not cols:
        raise ValueError("fit_minibatch_kmeans: X_cols must be non-empty")
    if n_clusters < 1:
        raise ValueError("fit_minibatch_kmeans: n_clusters must be >= 1")
    if batch_size < 2:
        raise ValueError("fit_minibatch_kmeans: batch_size must be >= 2")
    for c in cols:
        if c not in df._columns:
            raise KeyError(f"fit_minibatch_kmeans: column {c!r} not in {df._columns}")
    mg = memory_guard or df._memory_guard
    # pass 1: collect per-batch arrays into a reservoir of up to 10*batch rows
    reservoir = []
    reservoir_rows = 0
    cap = 10 * batch_size
    # v0.9.138 (DEC-023 pattern): io_batch_size auto-derive if None.
    # Separation: batch_size = SGD mini-batch (existing); io_batch_size
    # = df iterator (new). Before fix, both _iter_filtered_batches below
    # used default (1 batch = n) → OOC false in verification.
    if io_batch_size is None:
        _row_size = mg.estimate_row_size_from_schema(df._dataset.schema)
        io_batch_size = mg.get_batch_rows(
            row_size_bytes=_row_size, expansion_factor=3.0,
            state_bytes=int(n_clusters * len(X_cols) * 8 + n_clusters * 8))
    rng = np.random.RandomState(seed)
    n_seen = 0
    for b in df._iter_filtered_batches(batch_size=io_batch_size):
        mg.check_batch(b, name="minibatch.fit", extra_factor=3.0)
        X = _stack_numeric_safe(b, cols)
        if np.isnan(X).any():
            raise ValueError("fit_minibatch_kmeans: nulls in features")
        for row in X:
            n_seen += 1
            if reservoir_rows < cap:
                reservoir.append(row)
                reservoir_rows += 1
            else:
                j = int(rng.randint(n_seen))
                if j < cap:
                    reservoir[j] = row
        mg.check_state(reservoir_rows * (len(cols) * 8 + 8),
                       name="minibatch.reservoir")
    if n_seen == 0:
        raise ValueError("fit_minibatch_kmeans: empty input")
    R = np.asarray(reservoir, dtype=np.float64)
    k = min(int(n_clusters), R.shape[0])
    centers = _kmeanspp(R, k, rng)
    counts = np.zeros(k, dtype=np.float64)
    # pass 2: streaming per-centroid updates over the data itself
    prev_inertia = np.inf
    for _ in range(int(max_iter)):
        for b in df._iter_filtered_batches(batch_size=io_batch_size):
            mg.check_batch(b, name="minibatch.sweep", extra_factor=3.0)
            X = _stack_numeric_safe(b, cols)
            for s in range(0, X.shape[0], batch_size):
                Xb = X[s:s + batch_size]
                d2, lab = _assign(Xb, centers)
                for c in np.unique(lab):
                    mask = lab == c
                    counts[c] += mask.sum()
                    eta = mask.sum() / counts[c]
                    centers[c] = (1 - eta) * centers[c] + eta * Xb[mask].mean(axis=0)
        # convergence: relative change of centroids
        shift = 0.0
        total_norm = 0.0
        for b in df._iter_filtered_batches(batch_size=io_batch_size):
            mg.check_batch(b, name="minibatch.conv", extra_factor=3.0)
            X = _stack_numeric_safe(b, cols)
            d2, lab = _assign(X, centers)
            shift += float(np.sum(d2[np.arange(X.shape[0]), lab]))
            total_norm += float(np.sum(X * X))
        if total_norm > 0 and abs(shift - prev_inertia) <= tol * total_norm:
            prev_inertia = shift
            break
        prev_inertia = shift
    model = MiniBatchKMeansModel(centers, counts, cols,
                                 {"n_clusters": k, "batch_size": batch_size,
                                  "max_iter": max_iter, "tol": tol, "seed": seed})
    log_fit("minibatch_kmeans", loss=prev_inertia,
            state_bytes=model.state_nbytes())
    mg.check_state(model.state_nbytes(), name="minibatch.model")
    return model


def _log_gauss_diag(X, mu, var):
    d = X.shape[1]
    return (-0.5 * (d * np.log(2 * np.pi)
                    + float(np.sum(np.log(var)))
                    + np.sum((X - mu) ** 2 / var, axis=1)))


def _responsibilities(X, weights, means, covars):
    k = weights.size
    n = X.shape[0]
    logp = np.empty((n, k))
    for j in range(k):
        logp[:, j] = np.log(weights[j] + 1e-300) + _log_gauss_diag(X, means[j], covars[j])
    m = logp.max(axis=1, keepdims=True)
    p = np.exp(logp - m)
    s = p.sum(axis=1, keepdims=True)
    return p / s, float(np.sum(m[:, 0] + np.log(s[:, 0])))


class MiniBatchGMMModel:
    """Fitted diagonal-covariance GMM, EM fitted over mini-batches."""

    def __init__(self, weights, means, covars, feature_cols, params):
        self.weights_ = np.asarray(weights, dtype=np.float64)
        self.means_ = np.asarray(means, dtype=np.float64)
        self.covars_ = np.asarray(covars, dtype=np.float64)
        self.feature_cols_ = list(feature_cols)
        self.params_ = dict(params)
        self.fitted_ = True

    def state_nbytes(self):
        return int(self.weights_.nbytes + self.means_.nbytes
                   + self.covars_.nbytes + 256)

    def to_dict(self):
        return {"k": int(self.weights_.size),
                "d": int(self.means_.shape[1]),
                "params": self.params_,
                "weights": self.weights_.tolist()}

    def predict_proba(self, df, memory_guard=None):
        mg = memory_guard or df._memory_guard
        out = []
        for b in df._iter_filtered_batches(batch_size=io_batch_size):
            mg.check_batch(b, name="minibatch.gmm_proba", extra_factor=3.0)
            X = _stack_numeric_safe(b, self.feature_cols_)
            r, _ = _responsibilities(X, self.weights_, self.means_, self.covars_)
            out.append(r)
        return np.vstack(out) if out else np.zeros((0, self.weights_.size))

    def predict(self, df, memory_guard=None):
        return np.argmax(self.predict_proba(df, memory_guard), axis=1)

    def score_samples(self, df, memory_guard=None):
        mg = memory_guard or df._memory_guard
        out = []
        for b in df._iter_filtered_batches(batch_size=io_batch_size):
            mg.check_batch(b, name="minibatch.gmm_score", extra_factor=3.0)
            X = _stack_numeric_safe(b, self.feature_cols_)
            k = self.weights_.size
            logp = np.empty((X.shape[0], k))
            for j in range(k):
                logp[:, j] = np.log(self.weights_[j] + 1e-300) + _log_gauss_diag(
                    X, self.means_[j], self.covars_[j])
            m = logp.max(axis=1, keepdims=True)
            out.append(m[:, 0] + np.log(np.exp(logp - m).sum(axis=1)))
        return np.concatenate(out) if out else np.zeros(0)

    def score(self, df, memory_guard=None):
        return float(np.mean(self.score_samples(df, memory_guard)))


def fit_minibatch_gmm(df, X_cols, *, n_components=3, batch_size=1024,
                      max_iter=50, tol=1e-4, seed=0, reg=1e-6,
                      memory_guard: Optional[MemoryGuard] = None):
    """Fit a diagonal GMM by EM with per-batch sufficient statistics."""
    cols = [X_cols] if isinstance(X_cols, str) else list(X_cols)
    if not cols:
        raise ValueError("fit_minibatch_gmm: X_cols must be non-empty")
    if n_components < 1:
        raise ValueError("fit_minibatch_gmm: n_components must be >= 1")
    for c in cols:
        if c not in df._columns:
            raise KeyError(f"fit_minibatch_gmm: column {c!r} not in {df._columns}")
    mg = memory_guard or df._memory_guard
    rng = np.random.RandomState(seed)
    # init: small sample + kmeans++
    sample = []
    cap = 10 * batch_size
    n_seen = 0
    for b in df._iter_filtered_batches():
        mg.check_batch(b, name="minibatch.gmm_init", extra_factor=3.0)
        X = _stack_numeric_safe(b, cols)
        if np.isnan(X).any():
            raise ValueError("fit_minibatch_gmm: nulls in features")
        for row in X:
            n_seen += 1
            if len(sample) < cap:
                sample.append(row)
            else:
                j = int(rng.randint(n_seen))
                if j < cap:
                    sample[j] = row
        mg.check_state(len(sample) * (len(cols) * 8 + 8),
                       name="minibatch.gmm_sample")
    if n_seen == 0:
        raise ValueError("fit_minibatch_gmm: empty input")
    R = np.asarray(sample, dtype=np.float64)
    k = min(int(n_components), R.shape[0])
    means = _kmeanspp(R, k, rng)
    weights = np.full(k, 1.0 / k)
    covars = np.tile(R.var(axis=0) + reg, (k, 1))
    d = R.shape[1]
    prev_ll = -np.inf
    for _ in range(int(max_iter)):
        Nk = np.zeros(k)
        sumx = np.zeros((k, d))
        sumx2 = np.zeros((k, d))
        ll_sum = 0.0
        nb = 0
        for b in df._iter_filtered_batches():
            mg.check_batch(b, name="minibatch.gmm_em", extra_factor=3.0)
            X = _stack_numeric_safe(b, cols)
            r, ll = _responsibilities(X, weights, means, covars)
            ll_sum += ll
            nb += X.shape[0]
            Nk += r.sum(axis=0)
            sumx += r.T @ X
            sumx2 += r.T @ (X * X)
        Nk_safe = np.maximum(Nk, 1e-10)
        weights = Nk_safe / Nk_safe.sum()
        means = sumx / Nk_safe[:, None]
        covars = sumx2 / Nk_safe[:, None] - means * means
        covars = np.maximum(covars, reg)
        avg_ll = ll_sum / max(1, nb)
        if abs(avg_ll - prev_ll) <= tol:
            prev_ll = avg_ll
            break
        prev_ll = avg_ll
    model = MiniBatchGMMModel(weights, means, covars, cols,
                              {"n_components": k, "batch_size": batch_size,
                               "max_iter": max_iter, "tol": tol, "seed": seed,
                               "reg": reg})
    log_fit("minibatch_gmm", loss=prev_ll, state_bytes=model.state_nbytes())
    mg.check_state(model.state_nbytes(), name="minibatch.gmm_model")
    return model
