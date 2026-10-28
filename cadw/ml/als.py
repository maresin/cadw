"""Implicit-feedback ALS (D.2.2, Phase D, v0.9.36).

Reference: Hu, Koren, Volinsky, ICDM 2008; implicit.cpu.als (BSD) API.
See .pingis/algorithm_notes/d2_2_als.md.

Out-of-core: the matrix is never materialized. Each sweep sorts the
interactions by user with external_sort, streams the sorted run, and
solves one k x k system per user. State O((U+I)*k) = model, not data.
"""
from __future__ import annotations

import tempfile
from typing import Optional

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.ml._logging import log_fit
from cadw.ops.sort import external_sort, read_sorted


def _iter_user_groups(path, cols):
    """Yield (user, items_array, weights_array) from a user-sorted run."""
    cur = None
    items, wts = [], []
    for b in read_sorted(path, columns=list(cols)):
        u = b.column("u").to_numpy(zero_copy_only=False)
        it = b.column("i").to_numpy(zero_copy_only=False)
        w = b.column("w").to_numpy(zero_copy_only=False)
        for a, bb, c in zip(u, it, w):
            if cur is None:
                cur = a
            elif a != cur:
                yield cur, np.asarray(items, dtype=np.int64), np.asarray(wts, dtype=np.float64)
                items, wts = [], []
                cur = a
            items.append(bb); wts.append(c)
    if cur is not None:
        yield cur, np.asarray(items, dtype=np.int64), np.asarray(wts, dtype=np.float64)


class ALSModel:
    """Implicit-feedback ALS factors."""

    def __init__(self, user_factors, item_factors, user_index, item_index, params):
        self.user_factors_ = np.asarray(user_factors, dtype=np.float64)
        self.item_factors_ = np.asarray(item_factors, dtype=np.float64)
        self.user_index_ = list(user_index)
        self.item_index_ = list(item_index)
        self._u_pos = {u: i for i, u in enumerate(self.user_index_)}
        self._i_pos = {v: i for i, v in enumerate(self.item_index_)}
        self.params_ = dict(params)
        self.fitted_ = True

    def state_nbytes(self):
        return int(self.user_factors_.nbytes + self.item_factors_.nbytes)

    def to_dict(self):
        return {"params": self.params_, "n_users": len(self.user_index_),
                "n_items": len(self.item_index_),
                "k": int(self.user_factors_.shape[1])}

    def scores(self, user, items=None):
        """Score items for a user.

        `items=None`: scores for all train items, shape `(n_train_items,)`.
        `items=[...]`: scores in caller order, shape `(len(items),)`.
        Unknown user or unseen item -> `-inf` (no KeyError), matching the
        "cold item ranks last" convention. Fixed v0.9.46-fix1 (was KeyError).
        """
        ui = self._u_pos.get(user)
        if ui is None:
            shape = (len(items),) if items is not None else (self.item_factors_.shape[0],)
            return np.full(shape, -np.inf, dtype=np.float64)
        if items is None:
            return self.item_factors_ @ self.user_factors_[ui]
        out = np.full(len(items), -np.inf, dtype=np.float64)
        hit_j, hit_pos = [], []
        for j, item in enumerate(items):
            pos = self._i_pos.get(item)
            if pos is not None:
                hit_j.append(j)
                hit_pos.append(pos)
        if hit_pos:
            vals = self.item_factors_[hit_pos] @ self.user_factors_[ui]
            for j, v in zip(hit_j, vals):
                out[j] = v
        return out

    def recommend(self, user, n=10, exclude=None):
        sc = self.scores(user)
        if exclude:
            for i in exclude:
                p = self._i_pos.get(i)
                if p is not None:
                    sc[p] = -np.inf
        order = np.argsort(-sc)[: int(n)]
        return [(self.item_index_[j], float(sc[j])) for j in order]


def _objective_streaming(df, X, Y, u_pos, i_pos, user_col, item_col,
                         alpha, reg, mg):
    """Implicit-feedback objective, streamed over events (v0.9.87).

    Same formula as pre-fix `_objective`: uniform (1 + alpha) weighting
    (weight_col is only used in _solve_user/iter path, not in this
    loss). Bit-identical to in-RAM version up to float summation order.

    sum_{(u,i)} (1 + alpha) * (1 - x_u.y_i)^2 + reg*(||X||^2+||Y||^2)
    The unobserved mass is a constant w.r.t. the factors, so this is
    the Hu-Koren-Volinsky objective that actually moves.

    RAM: O(batch * k) per batch (X[up], Y[ip] fancy index); independent
    of n. Pre-fix held n-sized u_arr/i_arr in RAM alongside u_pos/i_pos
    dicts — silent OOM at n > ~1e7.
    """
    loss_events = 0.0
    for b in df._iter_filtered_batches():
        mg.check_batch(b, name="als.objective", extra_factor=8.0)
        n = b.num_rows
        if n == 0:
            continue
        u = b.column(user_col).to_pylist()
        i = b.column(item_col).to_pylist()
        up = np.fromiter((u_pos[v] for v in u), dtype=np.int64, count=n)
        ip = np.fromiter((i_pos[v] for v in i), dtype=np.int64, count=n)
        dot = np.einsum("ij,ij->i", X[up], Y[ip])
        loss_events += float(np.sum((1.0 + alpha) * (1.0 - dot) ** 2))
    loss_reg = float(reg * (np.sum(X * X) + np.sum(Y * Y)))
    return loss_events + loss_reg


def _gram(Y):
    return Y.T @ Y


def _solve_user(Yty, Y, items, weights, alpha, reg):
    """One implicit-ALS user update. c = 1 + alpha*w."""
    k = Y.shape[1]
    A = Yty + reg * np.eye(k)
    b = np.zeros(k)
    c = 1.0 + alpha * weights
    for idx, ci in zip(items, c):
        y = Y[idx]
        A += (ci - 1.0) * np.outer(y, y)
        b += ci * y
    return np.linalg.solve(A, b)


def fit_als(df, user_col, item_col, *, weight_col=None, n_factors=16,
            n_iter=10, reg=0.1, alpha=40.0, seed=0,
            memory_guard: Optional[MemoryGuard] = None):
    """Fit implicit-feedback ALS out-of-core (sort by user/item per sweep)."""
    cols = [user_col, item_col]
    for c in cols + ([weight_col] if weight_col else []):
        if c not in df._columns:
            raise KeyError(f"fit_als: column {c!r} not in {df._columns}")
    mg = memory_guard or df._memory_guard
    if n_factors < 1 or n_iter < 1:
        raise ValueError("fit_als: n_factors and n_iter must be >= 1")
    rng = np.random.RandomState(int(seed))
    spill = tempfile.mkdtemp(prefix="cadw_als_")

    # v0.9.87-P5a-fix-als: streaming set for unique user/item values.
    # Pre-fix: users/items list accum (O(n)) + np.concatenate (O(n))
    # + np.unique (O(n log n) sort) — ~3n floats @ n=1e8 = silent OOM,
    # guard was only factor state (U+I)*k. Post-fix: O(U + I) Python
    # set state (bounded by cardinality), no n-sized arrays.
    u_set: set = set()
    i_set: set = set()
    n_seen = 0
    for b in df._iter_filtered_batches():
        mg.check_batch(b, name="als.dict", extra_factor=8.0)
        u_set.update(b.column(user_col).to_pylist())
        i_set.update(b.column(item_col).to_pylist())
        n_seen += b.num_rows
    if n_seen == 0:
        raise ValueError("fit_als: empty input")
    u_vals = np.asarray(sorted(u_set))
    i_vals = np.asarray(sorted(i_set))
    U, I = len(u_vals), len(i_vals)
    mg.check_state((U + I) * n_factors * 8, name="als.state")
    u_pos = {v: j for j, v in enumerate(u_vals.tolist())}
    i_pos = {v: j for j, v in enumerate(i_vals.tolist())}
    # Note: sorted() vs np.unique produces same deterministic order
    # for homogeneous types. u_vals dtype may differ (object vs int)
    # — kept only for reverse lookup completeness; not used downstream
    # except via u_pos/i_pos dicts (keyed by original Python value).

    # Match implicit library exactly (als.py:145-147): uniform [0, 0.01).
    # Earlier normal(scale=0.01) gives mixed signs → b = Σ(1+alpha)·y_i
    # cancels near zero on iter 1 → |X| grows 250x over |Y| over 30 iters
    # → poor NDCG. Uniform-positive matches implicit and converges fast.
    X = rng.uniform(0.0, 0.01, size=(U, n_factors))
    Y = rng.uniform(0.0, 0.01, size=(I, n_factors))

    def _sorted_batches(by_user):
        """Emit u = group key, i = partner (position space)."""
        for b in df._iter_filtered_batches():
            u = b.column(user_col).to_numpy(zero_copy_only=False)
            i = b.column(item_col).to_numpy(zero_copy_only=False)
            w = (b.column(weight_col).cast(pa.float64(), safe=False)
                 .to_numpy(zero_copy_only=False) if weight_col else np.ones(len(u)))
            up = np.array([u_pos[v] for v in u.tolist()], dtype=np.int64)
            ip = np.array([i_pos[v] for v in i.tolist()], dtype=np.int64)
            if by_user:
                yield pa.record_batch({"u": pa.array(up), "i": pa.array(ip),
                                       "w": pa.array(w)})
            else:
                yield pa.record_batch({"u": pa.array(ip), "i": pa.array(up),
                                       "w": pa.array(w)})

    upath = external_sort(_sorted_batches(True), ["u"], ["u", "i", "w"],
                          mg, spill + "/u", log=False)
    ipath = external_sort(_sorted_batches(False), ["u"], ["u", "i", "w"],
                          mg, spill + "/i", log=False)

    prev = None
    for it in range(int(n_iter)):
        Yty = _gram(Y)
        for u, its, ws in _iter_user_groups(upath, ["u", "i", "w"]):
            X[u] = _solve_user(Yty, Y, its, ws, alpha, reg)
        Xtx = _gram(X)
        for i, us, ws in _iter_user_groups(ipath, ["u", "i", "w"]):
            Y[i] = _solve_user(Xtx, X, us, ws, alpha, reg)
        loss = _objective_streaming(
            df, X, Y, u_pos, i_pos, user_col, item_col,
            alpha, reg, mg)
        log_fit("als", iter_i=it + 1, iter_n=int(n_iter), loss=loss,
                state_bytes=(U + I) * n_factors * 8)
    model = ALSModel(X, Y, u_vals.tolist(), i_vals.tolist(),
                     {"n_factors": n_factors, "n_iter": n_iter, "reg": reg,
                      "alpha": alpha, "seed": seed})
    return model
