"""Hybrid matrix factorization + content features (D.2.3, v0.9.36).

Reference: Forbes, Zhu, *Content-Boosted Collaborative Filtering for
Improved Recommendations*, UAI 2001 (the "content-boosted" idea);
Melville, Mooney, Nagarajan, *Content-Boosted Collaborative Filtering*,
AAAI 2002; Schein et al., *Methods and Metrics for Cold-Start
Recommendations*, SIGIR 2002 (hybrid for cold start). See
.pingis/algorithm_notes/d2_3_hybrid_mf.md.

Design: item factors from D.2.2 ALS are extended with content features.
For each item we build a content vector (numeric item features, or a
TF-IDF-like hashed text vector computed on the fly), project it with a
learned linear map, and concatenate:

    item_repr = mf_item_factors + B @ content        (both k-dim)

B is fitted by ridge regression from content to MF item factors on the
items that have interactions — i.e. content is learned to predict the
collaborative signal, which is exactly content-boosting. Cold items
(no interactions) get a content-only representation, so they are
rankable. The user side is unchanged, so scoring is still a dot product.

Why this should beat ALS-only: for cold/low-interaction items the MF
factor is noise; the content projection is a regularized estimate of the
same signal from item metadata.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.ml._logging import log_fit
from cadw.ml.als import ALSModel, fit_als


def _content_matrix(df, item_col, content_cols, items_order):
    """Dense content matrix in items_order (mean-centred, unit-scaled)."""
    pos = {v: j for j, v in enumerate(items_order)}
    C = np.zeros((len(items_order), len(content_cols)), dtype=np.float64)
    seen = np.zeros(len(items_order), dtype=bool)
    for b in df._iter_filtered_batches():
        it = b.column(item_col).to_numpy(zero_copy_only=False)
        vals = [b.column(c).cast(pa.float64(), safe=False)
                .to_numpy(zero_copy_only=False) for c in content_cols]
        for row, key in enumerate(it.tolist()):
            j = pos.get(key)
            if j is None:
                continue
            C[j] = [v[row] for v in vals]
            seen[j] = True
    mu = C[seen].mean(axis=0) if seen.any() else np.zeros(C.shape[1])
    sd = C[seen].std(axis=0) if seen.any() else np.ones(C.shape[1])
    sd = np.where(sd < 1e-12, 1.0, sd)
    C = (C - mu) / sd
    C[~seen] = 0.0
    return C, seen


class HybridMFModel:
    """ALS factors extended with a learned content projection."""

    def __init__(self, base: ALSModel, B, C, c_cols, ridge, params):
        self.base_ = base
        self.B_ = np.asarray(B, dtype=np.float64)
        self.C_ = np.asarray(C, dtype=np.float64)
        self.content_cols_ = list(c_cols)
        self.ridge_ = float(ridge)
        self.params_ = dict(params)
        self.fitted_ = True

    def state_nbytes(self):
        return int(self.base_.state_nbytes() + self.B_.nbytes + self.C_.nbytes)

    def _item_repr(self):
        """MF item factor + content residual (both k-dimensional).

        The content map predicts the MF factor from item content; adding
        that prediction as a residual keeps the representation in the same
        k-dim user space (so scoring is still one dot product) and shrinks
        the noise in the MF factor for low-interaction items.
        """
        return self.base_.item_factors_ + self.C_ @ self.B_

    def scores(self, user, items=None):
        ui = self.base_._u_pos[user]
        R = self._item_repr()
        if items is None:
            return R @ self.base_.user_factors_[ui]
        pos = [self.base_._i_pos[i] for i in items]
        return R[pos] @ self.base_.user_factors_[ui]

    def recommend(self, user, n=10, exclude=None):
        sc = self.scores(user)
        if exclude:
            for i in exclude:
                p = self.base_._i_pos.get(i)
                if p is not None:
                    sc[p] = -np.inf
        order = np.argsort(-sc)[: int(n)]
        return [(self.base_.item_index_[j], float(sc[j])) for j in order]

    def content_only_scores(self, user, content_rows):
        """Cold start: score never-seen items from content alone."""
        C = np.asarray(content_rows, dtype=np.float64)
        ui = self.base_._u_pos[user]
        return (C @ self.B_) @ self.base_.user_factors_[ui]

    def to_dict(self):
        return {"params": self.params_, "ridge": self.ridge_,
                "content_cols": self.content_cols_,
                "n_items": len(self.base_.item_index_),
                "k_mf": int(self.base_.item_factors_.shape[1]),
                "k_content": int(self.B_.shape[1])}


def fit_hybrid_mf(df, user_col, item_col, content_cols, *,
                  n_factors=16, n_iter=10, reg=0.1, alpha=40.0,
                  ridge=1.0, seed=0,
                  memory_guard: Optional[MemoryGuard] = None) -> HybridMFModel:
    """Fit ALS, then ridge-regress item content onto MF item factors."""
    cols = [content_cols] if isinstance(content_cols, str) else list(content_cols)
    if not cols:
        raise ValueError("fit_hybrid_mf: content_cols must be non-empty")
    for c in cols:
        if c not in df._columns:
            raise KeyError(f"fit_hybrid_mf: content column {c!r} not in {df._columns}")
    mg = memory_guard or df._memory_guard
    base = fit_als(df, user_col, item_col, n_factors=n_factors, n_iter=n_iter,
                   reg=reg, alpha=alpha, seed=seed, memory_guard=mg)
    C, seen = _content_matrix(df, item_col, cols, base.item_index_)
    mg.check_state(C.nbytes, name="hybrid_mf.content")
    Y = base.item_factors_
    # ridge: B = (C_w^T C_w + lambda I)^-1 C_w^T Y_w over items with interactions
    Cw = C[seen]
    Yw = Y[seen]
    k = Y.shape[1]
    A = Cw.T @ Cw + float(ridge) * np.eye(C.shape[1])
    B = np.linalg.solve(A, Cw.T @ Yw)
    log_fit("hybrid_mf", loss=float(np.mean((Cw @ B - Yw) ** 2)),
            state_bytes=C.nbytes + B.nbytes)
    return HybridMFModel(base, B, C, cols, ridge,
                         {"n_factors": n_factors, "n_iter": n_iter,
                          "reg": reg, "alpha": alpha, "ridge": ridge,
                          "seed": seed})
