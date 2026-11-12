"""Multi-label kNN with Bayesian prior (Zhang & Zhou 2007).

Reference: Zhang, M.-L., & Zhou, Z.-H. (2007). ML-KNN: A lazy learning
approach to multi-label learning. Pattern Recognition, 40(7), 2038-2048.

Training:
  1. Reservoir-sample X||Y (Algorithm R); n_train = min(max_samples, n_seen).
  2. For each training x_i, find k nearest neighbours (self-excluded),
     count C_j = number of neighbours with label_j = 1.
  3. Build Laplace-smoothed conditional tables:
       P_pos[j, c] = P(C_j=c | y_j=1) = (s + n_pos_j_c) / (s*(k+1) + n_pos_j)
       P_neg[j, c] = P(C_j=c | y_j=0) = (s + n_neg_j_c) / (s*(k+1) + n_neg_j)
       prior[j]    = (s + n_pos_j)    / (2*s + n_train)

Prediction:
  For test x, kNN -> C_j; y_j=1 iff prior[j]*P_pos[j,C_j] >
                                 (1-prior[j])*P_neg[j,C_j].

OOC: state O(max_samples*(d+L)); bounded by hyperparams, not n.
Fit peak = O(max_samples*(d+L)) state + O(max_samples*phase1_chunk)
  Phase-1 transient from knn_query merged_d + O(batch*row_size)
  reservoir input. All bounded by hyperparams, NOT n.
Predict: batched input (batch_size derived), n-sized output guarded
  by mg.check_state(n_total*L*8) before np.empty (predict_batched:
  false — materializes; documented).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.ml._logging import log_fit
from cadw.ops.linalg import knn_query
from cadw.ops.sampling import reservoir_sample

# Phase 1 knn_query chunk size. Explicit bound so ram_formula is
# honest about merged_d transient (max_samples * phase1_chunk * 8).
_PHASE1_CHUNK_DEFAULT = 500


def _stack_X(batch, X_cols: Sequence[str]) -> np.ndarray:
    """Stack X_cols into (b, d) float64. Row-bounded by batch size."""
    d = len(X_cols)
    b_n = batch.num_rows
    X = np.empty((b_n, d), dtype=np.float64)
    for j, c in enumerate(X_cols):
        X[:, j] = batch.column(c).to_numpy(zero_copy_only=False).astype(
            np.float64, copy=False)
    return X


def _build_conditional_tables(
    train_X: np.ndarray,
    train_Y: np.ndarray,
    k: int,
    smoothing: float,
    phase1_chunk: int,
    mg: Optional[MemoryGuard],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Phase 1: build (prior, P_pos, P_neg) from training data."""
    n_train, d = train_X.shape
    L = train_Y.shape[1]
    if n_train < k + 1:
        raise ValueError(
            f"ML-kNN: need at least k+1={k + 1} train samples, "
            f"got {n_train}")

    # k+1 nearest of each train_X (self at [:,0] with dist 0)
    nbr_idx, _ = knn_query(
        train_X, train_X, k + 1, return_dist=True,
        metric="euclidean", memory_guard=mg,
        chunk_size=int(phase1_chunk))
    nbr_idx = nbr_idx[:, 1:]  # drop self -> (n_train, k)

    # C[j,c] counts: for each sample, count positive labels in k neighbours
    neighbor_Y = train_Y[nbr_idx]  # (n_train, k, L) float64 (from reservoir)
    C = neighbor_Y.sum(axis=1).astype(np.int64)  # (n_train, L) int

    n_pos = train_Y.sum(axis=0).astype(np.float64)  # (L,)
    prior = (smoothing + n_pos) / (2.0 * smoothing + float(n_train))

    P_pos = np.zeros((L, k + 1), dtype=np.float64)
    P_neg = np.zeros((L, k + 1), dtype=np.float64)
    for j in range(L):
        yj = train_Y[:, j]
        Cj = C[:, j]
        n_pos_j = float(n_pos[j])
        n_neg_j = float(n_train) - n_pos_j
        for c in range(k + 1):
            mask_c = (Cj == c)
            n_pos_j_c = int(((yj == 1) & mask_c).sum())
            n_neg_j_c = int(((yj == 0) & mask_c).sum())
            P_pos[j, c] = (
                (smoothing + n_pos_j_c)
                / (smoothing * (k + 1) + n_pos_j))
            P_neg[j, c] = (
                (smoothing + n_neg_j_c)
                / (smoothing * (k + 1) + n_neg_j))
    return prior, P_pos, P_neg


@dataclass
class MLkNNModel:
    """ML-kNN (Zhang & Zhou 2007). State O(max_samples*(d+L))."""
    train_X_: np.ndarray
    train_Y_: np.ndarray
    prior_: np.ndarray
    P_pos_: np.ndarray
    P_neg_: np.ndarray
    k_: int
    smoothing_: float
    feature_names_: list
    label_names_: list
    n_train_: int
    seed_: int

    def state_nbytes(self) -> int:
        """State = reservoir (X||Y) + conditional tables + prior + header."""
        n_train, d = self.train_X_.shape
        L = self.train_Y_.shape[1]
        return int(
            n_train * (d + L) * 8
            + L * (self.k_ + 1) * 8 * 2
            + L * 8
            + 256
        )

    def _batch_posterior(self, X, mg):
        """Compute (b, L) posterior P(y_j=1|x) for one batch.

        Per-batch transient (all guarded):
          nbr_idx:     b * k * i64   (knn_query.state guards within;
                                      return_dist=False -> no dist alloc)
          neighbor_Y:  b * k * L * f64  (fancy index allocates)
          C:           b * L * i64
          out:         b * L * f64
        All bounded by batch_size (predict kwarg, default 1000).
        """
        b = X.shape[0]
        L = len(self.label_names_)
        # Guard batch transient BEFORE fancy-index neighbor_Y alloc.
        mg.check_state(
            b * (self.k_ * 16 + self.k_ * L * 8 + L * 8 * 2) + 256,
            name="mlknn.predict.batch_transient")
        nbr_idx = knn_query(
            self.train_X_, X, self.k_, return_dist=False,
            metric="euclidean", memory_guard=mg)  # (b, k) int64; no dist
        neighbor_Y = self.train_Y_[nbr_idx]  # (b, k, L) fancy-index alloc
        C = neighbor_Y.sum(axis=1).astype(np.int64)  # (b, L)
        out = np.empty((X.shape[0], L), dtype=np.float64)
        for j in range(L):
            p1 = self.prior_[j] * self.P_pos_[j, C[:, j]]
            p0 = (1.0 - self.prior_[j]) * self.P_neg_[j, C[:, j]]
            out[:, j] = p1 / (p1 + p0 + 1e-30)
        return out

    def predict_proba(self, df, *, memory_guard: Optional[MemoryGuard] = None,
                      batch_size: Optional[int] = None):
        """Return CadwDataFrame {label}_proba per label (float64)."""
        from cadw import CadwDataFrame
        mg = memory_guard or df._memory_guard
        for c in self.feature_names_:
            if c not in df._columns:
                raise KeyError(f"predict: column {c!r} not in df")
        n_total = int(df.count())
        L = len(self.label_names_)
        mg.check_state(
            n_total * L * 8 + 256,
            name="mlknn.predict_proba.output")
        out = np.empty((n_total, L), dtype=np.float64)
        off = 0
        _bs = batch_size if batch_size is not None else 1000
        for b in df._iter_filtered_batches(batch_size=_bs):
            mg.check_batch(b, name="mlknn.predict_proba", extra_factor=2.0)
            X = _stack_X(b, self.feature_names_)
            out[off:off + b.num_rows] = self._batch_posterior(X, mg)
            off += b.num_rows
        if off != n_total:
            out = out[:off]
        cols = {f"{self.label_names_[j]}_proba":
                pa.array(out[:, j], type=pa.float64())
                for j in range(L)}
        return CadwDataFrame.from_table(pa.table(cols), memory_guard=mg)

    def predict(self, df, *, memory_guard: Optional[MemoryGuard] = None,
                batch_size: Optional[int] = None):
        """Return CadwDataFrame {label}_pred per label (int64, 0/1)."""
        from cadw import CadwDataFrame
        mg = memory_guard or df._memory_guard
        for c in self.feature_names_:
            if c not in df._columns:
                raise KeyError(f"predict: column {c!r} not in df")
        n_total = int(df.count())
        L = len(self.label_names_)
        mg.check_state(
            n_total * L * 8 + 256,
            name="mlknn.predict.output")
        out = np.zeros((n_total, L), dtype=np.int64)
        off = 0
        _bs = batch_size if batch_size is not None else 1000
        for b in df._iter_filtered_batches(batch_size=_bs):
            mg.check_batch(b, name="mlknn.predict", extra_factor=2.0)
            X = _stack_X(b, self.feature_names_)
            post = self._batch_posterior(X, mg)  # (b, L) float
            out[off:off + b.num_rows] = (post > 0.5).astype(np.int64)
            off += b.num_rows
        if off != n_total:
            out = out[:off]
        cols = {f"{self.label_names_[j]}_pred":
                pa.array(out[:, j], type=pa.int64())
                for j in range(L)}
        return CadwDataFrame.from_table(pa.table(cols), memory_guard=mg)


def fit_knn_multilabel(
    df,
    X_cols: "str | Sequence[str]",
    y_cols: "str | Sequence[str]",
    *,
    k: int = 3,
    max_samples: int = 1000,
    smoothing: float = 1.0,
    seed: int = 0,
    phase1_chunk: int = _PHASE1_CHUNK_DEFAULT,
    memory_guard: Optional[MemoryGuard] = None,
    batch_size: "int | None" = None,
) -> MLkNNModel:
    """Fit multi-label kNN (Zhang & Zhou 2007), OOC in n.

    Args:
      df: CadwDataFrame.
      X_cols: feature columns (numeric).
      y_cols: label columns (0/1, int or float).
      k: neighbours used at both fit (self+ k) and predict (k).
      max_samples: reservoir size; >= k+1.
      smoothing: Laplace smoothing; > 0. Default 1.0 (paper).
      seed: reservoir deterministic.
      memory_guard, batch_size: standard.
    """
    if isinstance(X_cols, str):
        X_cols = [X_cols]
    if isinstance(y_cols, str):
        y_cols = [y_cols]
    X_cols = list(X_cols)
    y_cols = list(y_cols)
    if not X_cols:
        raise ValueError("fit_knn_multilabel: X_cols is empty")
    if not y_cols:
        raise ValueError("fit_knn_multilabel: y_cols is empty")
    overlap = set(X_cols) & set(y_cols)
    if overlap:
        raise ValueError(
            f"fit_knn_multilabel: X/y overlap: {sorted(overlap)}")
    if k < 1:
        raise ValueError(f"fit_knn_multilabel: k >= 1; got {k}")
    if max_samples < k + 1:
        raise ValueError(
            f"fit_knn_multilabel: max_samples >= k+1 ({k + 1}); "
            f"got {max_samples}")
    if smoothing <= 0:
        raise ValueError(
            f"fit_knn_multilabel: smoothing > 0; got {smoothing}")

    for c in X_cols + y_cols:
        if c not in df.columns:
            raise KeyError(f"fit_knn_multilabel: column {c!r} not in df")

    schema = df._dataset.schema
    for c in X_cols:
        t = schema.field(c).type
        if not (pa.types.is_integer(t) or pa.types.is_floating(t)):
            raise TypeError(
                f"fit_knn_multilabel: X column {c!r} has type {t}; "
                f"numeric required")

    mg = memory_guard or df._memory_guard
    d = len(X_cols)
    L = len(y_cols)

    if batch_size is None:
        batch_size = 1000
    batch_size = max(1, int(batch_size))

    # State guard: max_samples * (d+L) f64 bounded by hyperparams
    mg.check_state(
        max_samples * (d + L) * 8 + 4096,
        name="mlknn.state")

    # Reservoir over X||Y (single pass); batch-check via wrapper.
    def _checked_batches():
        for b in df._iter_filtered_batches(batch_size=batch_size):
            mg.check_batch(b, name="mlknn.reservoir_batch",
                           extra_factor=2.5)
            yield b

    feature_cols_all = list(X_cols) + list(y_cols)
    sample, n_seen = reservoir_sample(
        _checked_batches(), feature_cols_all, max_samples,
        seed=seed, memory_guard=mg)

    if n_seen < k + 1:
        raise ValueError(
            f"fit_knn_multilabel: n_seen={n_seen} < k+1={k + 1}")

    train_X = sample[:, :d].copy()
    train_Y = sample[:, d:d + L].copy()

    if not np.all((train_Y == 0) | (train_Y == 1)):
        raise ValueError(
            "fit_knn_multilabel: y_cols must be binary (0/1); "
            f"got range [{train_Y.min()}, {train_Y.max()}]")

    if phase1_chunk < 1:
        raise ValueError(
            f"fit_knn_multilabel: phase1_chunk >= 1; got {phase1_chunk}")

    prior, P_pos, P_neg = _build_conditional_tables(
        train_X, train_Y, k, smoothing, int(phase1_chunk), mg)

    model = MLkNNModel(
        train_X_=train_X,
        train_Y_=train_Y,
        prior_=prior,
        P_pos_=P_pos,
        P_neg_=P_neg,
        k_=int(k),
        smoothing_=float(smoothing),
        feature_names_=list(X_cols),
        label_names_=list(y_cols),
        n_train_=int(train_X.shape[0]),
        seed_=int(seed),
    )
    log_fit("mlknn", state_bytes=model.state_nbytes())
    return model
