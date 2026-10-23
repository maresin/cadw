"""Matrix Profile (D.1.2, Phase D, v0.9.35) — streaming z-normalized.

Reference: Yeh et al., Matrix Profile I, ICDM 2016 (arXiv:1701.02795);
Zhu et al., Matrix Profile II, ICDM 2016 (arXiv:1612.00835); STUMPY docs
(BSD) for the API contract.

Bounded-state design (project constraint: <=1 batch in RAM): the full
STAMP algorithm keeps O(N) for the profile. Here we keep a reservoir
sample of S z-normalized subsequences (default 4096) and run exact block
STOMP over that sample. State = O(S*w), independent of N. The profile is
approximate w.r.t. the whole series and sample_fraction_ records that.

Discord score = mP / sqrt(2w): larger = more anomalous (max 2.0).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.ml._logging import log_fit

_SUBSEQ_BYTES = 8


def _znorm_rows(X):
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    out = (X - mu) / sd
    out[np.abs(X - mu).max(axis=1) < 1e-12] = 0.0
    return out


def _sliding_view(arr, w):
    n = arr.shape[0]
    if n < w:
        return np.zeros((0, w), dtype=np.float64)
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    s = arr.strides[0]
    return np.lib.stride_tricks.as_strided(
        arr, shape=(n - w + 1, w), strides=(s, s), writeable=False)


class MatrixProfileModel:
    """Bounded matrix profile over a reservoir sample of subsequences."""

    def __init__(self, profile, index, ref_zn, window, n_seen):
        self.profile_ = np.asarray(profile, dtype=np.float64)
        self.index_ = np.asarray(index, dtype=np.int64)
        self.ref_zn_ = np.asarray(ref_zn, dtype=np.float64)
        self.window_ = int(window)
        self.sample_size_ = int(self.profile_.shape[0])
        self.n_seen_ = int(n_seen)
        self.sample_fraction_ = self.sample_size_ / max(1, int(n_seen))
        self.fitted_ = True

    def state_nbytes(self):
        return int(self.profile_.nbytes + self.index_.nbytes + self.ref_zn_.nbytes)

    def discord_scores(self):
        """Largest profile value = most anomalous (never repeats anywhere)."""
        return self.profile_ / np.sqrt(2.0 * self.window_)

    def top_discords(self, k=5):
        sc = self.discord_scores()
        order = np.argsort(-sc)[: int(k)]
        return [(int(i), float(sc[i])) for i in order]

    def to_dict(self):
        return {"window": self.window_, "sample_size": self.sample_size_,
                "n_seen": self.n_seen_, "sample_fraction": self.sample_fraction_,
                "profile": self.profile_.tolist(), "index": self.index_.tolist()}

    def transform(self, df, value_col, memory_guard=None):
        """Nearest-neighbour distance of each incoming subsequence to the sample."""
        mg = memory_guard or MemoryGuard(max_memory_mb=64)
        ref = self.ref_zn_
        rsq = (ref * ref).sum(axis=1)
        out = []
        for b in df._iter_filtered_batches():
            mg.check_batch(b, name="matrix_profile.transform", extra_factor=3.0)
            x = b.column(value_col).cast(pa.float64(), safe=False).to_numpy(zero_copy_only=False)
            if np.isnan(x).any():
                raise ValueError("matrix_profile: nulls in value column")
            V = _sliding_view(x, self.window_)
            if V.shape[0] == 0:
                out.append(np.zeros(0))
                continue
            Z = _znorm_rows(V)
            d2 = ((Z * Z).sum(axis=1)[:, None] + rsq[None, :] - 2.0 * (Z @ ref.T))
            out.append(np.sqrt(np.maximum(d2.min(axis=1), 0.0)))
        prof = np.concatenate(out) if out else np.zeros(0)
        return pa.table({"matrix_profile": pa.array(prof, type=pa.float64())})


def fit_matrix_profile(df, value_col, *, window=32, max_samples=4096,
                       random_state=0, memory_guard=None):
    """Bounded-state matrix profile (z-normalized Euclidean distance).

    Pass 1: reservoir-sample up to max_samples subsequences of length window
    (deterministic given the seed). Pass 2: exact block STOMP over the sample,
    chunked so peak RAM stays O(block * S).

    State: O(S*w). Raises MemoryLimitExceeded if that exceeds the budget.
    """
    if value_col not in df._columns:
        raise KeyError(f"matrix_profile: column {value_col!r} not in {df._columns}")
    w = int(window)
    if w < 4:
        raise ValueError("matrix_profile: window must be >= 4")
    S = int(max_samples)
    if S < 8:
        raise ValueError("matrix_profile: max_samples must be >= 8")
    mg = memory_guard or df._memory_guard
    mg.check_state(S * w * _SUBSEQ_BYTES, name="matrix_profile.sample")
    rng = np.random.RandomState(0 if random_state is None else int(random_state))
    samp = np.empty((S, w), dtype=np.float64)
    filled = 0
    seen = 0
    for b in df._iter_filtered_batches():
        mg.check_batch(b, name="matrix_profile.pass1", extra_factor=2.0)
        x = b.column(value_col).cast(pa.float64(), safe=False).to_numpy(zero_copy_only=False)
        if np.isnan(x).any():
            raise ValueError("matrix_profile: nulls in value column")
        V = _sliding_view(x, w)
        for j in range(V.shape[0]):
            seen += 1
            if filled < S:
                samp[filled] = V[j]
                filled += 1
            else:
                r = int(rng.randint(0, seen))
                if r < S:
                    samp[r] = V[j]
    if filled < 8:
        raise ValueError(f"matrix_profile: only {filled} subsequences; need >= 8")
    ref = _znorm_rows(samp[:filled])
    n = ref.shape[0]
    prof = np.full(n, np.inf)
    idx = np.full(n, -1, dtype=np.int64)
    sq = (ref * ref).sum(axis=1)
    block = max(16, min(256, int(32_000_000 // max(1, n * 8))))
    for i0 in range(0, n, block):
        i1 = min(n, i0 + block)
        d2 = sq[i0:i1, None] + sq[None, :] - 2.0 * (ref[i0:i1] @ ref.T)
        np.maximum(d2, 0.0, out=d2)
        for r in range(i1 - i0):
            d2[r, i0 + r] = np.inf
        j = np.argmin(d2, axis=1)
        prof[i0:i1] = np.sqrt(d2[np.arange(i1 - i0), j])
        idx[i0:i1] = j
        log_fit("matrix_profile", iter_i=i1, iter_n=n,
                loss=float(np.nanmean(prof[:i1])), state_bytes=n * w * 8)
    return MatrixProfileModel(prof, idx, ref, w, seen)
