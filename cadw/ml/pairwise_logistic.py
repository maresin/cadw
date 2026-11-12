"""Pairwise logistic LTR (RankNet, Burges et al. 2005).

Reference: Burges, Shaked, Renshaw, Lazier, Deeds, Hamilton, Hullender
(2005). Learning to rank using gradient descent. ICML.

Fit: per-query SGD on pairwise logistic loss.
  s_i = coef·x_i, P_ij = sigmoid(sigma*(s_i - s_j))
  dL/dcoef += (1 - P_ij) * (x_i - x_j) for y_i > y_j

  Equivalent to coef -= lr * (X_q.T @ ranknet_grad(s_q, y_q, sigma)) / n_q
  where ranknet_grad returns per-item gradient of pairwise loss.

OOC:
  state  = coef_(d)*f64 + 256B header  (O(d) hyperparam, not n)
  fit    = O(max_query_size * d)  per-query X_q + scores + grad
         + O(batch * row_size)  sorted-stream read transient
  predict = O(n_total * f64) output materialized (predict_batched:
            false, honest; guard.check_state before np.empty)

Query grouping: external_sort by qid (P20 precedent fit_lambdamart).
Each query group iterated as one unit; max_query_size guards against
pathological giant queries (refuses cleanly at guard, honest).
"""
from __future__ import annotations

from dataclasses import dataclass
import shutil
import tempfile
from typing import Optional, Sequence

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.io.batch_reader import ArrowIPCBatchReader
from cadw.ml._logging import log_fit
from cadw.ml._ltr_common import ranknet_grad
from cadw.ops.sort import external_sort


_MAX_QUERY_SIZE_DEFAULT = 1000
# external_sort k-way merge fan-in: bounds merge-phase peak to
# merge_fan_in * sort_read_batch_size * row_size (independent of n_runs).
_MERGE_FAN_IN_DEFAULT = 4
_SORT_READ_BATCH_DEFAULT = 500


def _stack_X(batch, X_cols: Sequence[str]) -> np.ndarray:
    d = len(X_cols)
    b_n = batch.num_rows
    X = np.empty((b_n, d), dtype=np.float64)
    for j, c in enumerate(X_cols):
        X[:, j] = batch.column(c).to_numpy(zero_copy_only=False).astype(
            np.float64, copy=False)
    return X


def _iter_query_groups(sorted_path, qid_col, cols, mg):
    """Yield (qid, X_q, y_q) per query from a sorted stream.

    Precondition: external_sort on qid_col. Groups are contiguous.
    Guarded per-batch; max_query_size enforced by caller.

    cols: [qid_col, y_col] + X_cols order.
    """
    X_cols = cols[2:]
    # Use ArrowIPCBatchReader directly (read_sorted is a generator
    # wrapper; closing the generator does NOT close the underlying
    # ipc.open_file memory-map). Explicit close() releases mmap and
    # makes peak RSS independent of n (P22 diagnosis).
    reader = ArrowIPCBatchReader(str(sorted_path), columns=cols)
    pending_qid = None
    pending_X = []
    pending_y = []

    def _flush(qid, Xs, ys, mg):
        if not Xs:
            return None
        X_q = np.concatenate(Xs, axis=0) if len(Xs) > 1 else Xs[0]
        y_q = np.concatenate(ys, axis=0) if len(ys) > 1 else ys[0]
        return (qid, X_q, y_q)

    try:
        for b in reader:
            if mg is not None:
                mg.check_batch(b, name="pairwise_logistic.group_batch",
                               extra_factor=2.5)
            qids = b.column(qid_col).to_numpy(zero_copy_only=False)
            y = b.column(cols[1]).cast(
                pa.float64(), safe=False).to_numpy(zero_copy_only=False)
            X = _stack_X(b, X_cols)
            if qids.size == 0:
                continue
            # Split contiguous runs
            boundaries = [0]
            diffs = np.flatnonzero(qids[1:] != qids[:-1]) + 1
            boundaries.extend(diffs.tolist())
            boundaries.append(qids.size)
            for i in range(len(boundaries) - 1):
                s, e = boundaries[i], boundaries[i + 1]
                qid = qids[s]
                if pending_qid is not None and qid != pending_qid:
                    out = _flush(pending_qid, pending_X, pending_y, mg)
                    if out is not None:
                        yield out
                    pending_X = []
                    pending_y = []
                pending_qid = qid
                pending_X.append(X[s:e])
                pending_y.append(y[s:e])
        if pending_qid is not None and pending_X:
            out = _flush(pending_qid, pending_X, pending_y, mg)
            if out is not None:
                yield out
    finally:
        # ArrowIPCBatchReader.close() releases the memory-mapped file.
        # Without this, the mmap persists in RSS until GC (P22: 3x
        # reader leak across fit's validation + 2-epoch iterations
        # caused 22x peak ratio at nq=2000).
        reader.close()


@dataclass
class PairwiseLogisticModel:
    """RankNet linear scorer. State = coef_(d)*f64 + 256B."""
    coef_: np.ndarray
    feature_names_: list
    query_col_: str
    relevance_col_: str
    sigma_: float
    lr_: float
    n_epochs_: int
    n_queries_: int
    max_query_size_: int
    final_loss_: float

    def state_nbytes(self) -> int:
        return int(self.coef_.nbytes + 256)

    def predict(self, df, *, memory_guard: Optional[MemoryGuard] = None,
                batch_size: Optional[int] = None):
        """Score each row: returns CadwDataFrame {relevance_col}_score."""
        from cadw import CadwDataFrame
        mg = memory_guard or df._memory_guard
        for c in self.feature_names_:
            if c not in df._columns:
                raise KeyError(f"predict: column {c!r} not in df")
        n_total = int(df.count())
        mg.check_state(n_total * 8 + 256,
                       name="pairwise_logistic.predict_output")
        out = np.empty(n_total, dtype=np.float64)
        off = 0
        _bs = batch_size if batch_size is not None else 1000
        for b in df._iter_filtered_batches(batch_size=_bs):
            mg.check_batch(b, name="pairwise_logistic.predict",
                           extra_factor=2.0)
            X = _stack_X(b, self.feature_names_)
            out[off:off + b.num_rows] = X @ self.coef_
            off += b.num_rows
        if off != n_total:
            out = out[:off]
        return CadwDataFrame.from_table(
            pa.table({f"{self.relevance_col_}_score":
                      pa.array(out, type=pa.float64())}),
            memory_guard=mg)


def fit_pairwise_logistic(
    df,
    X_cols: "str | Sequence[str]",
    qid_col: str,
    relevance_col: str,
    *,
    sigma: float = 1.0,
    n_epochs: int = 20,
    lr: float = 0.01,
    max_query_size: int = _MAX_QUERY_SIZE_DEFAULT,
    merge_fan_in: int = _MERGE_FAN_IN_DEFAULT,
    sort_read_batch_size: int = _SORT_READ_BATCH_DEFAULT,
    phase1_buffer_bytes: "int | None" = None,
    memory_guard: Optional[MemoryGuard] = None,
    batch_size: "int | None" = None,
) -> PairwiseLogisticModel:
    """Fit RankNet (pairwise logistic on preference pairs), OOC in n.

    Args:
      df: CadwDataFrame.
      X_cols: feature columns (numeric).
      qid_col: query/group identifier column.
      relevance_col: relevance labels (numeric, higher = more relevant).
      sigma: sigmoid scale (>0). Default 1.0.
      n_epochs: SGD passes over sorted queries. Default 20.
      lr: learning rate (>0). Default 0.01.
      max_query_size: guard; refuse if any query has more rows.
      memory_guard, batch_size: standard.
    """
    if isinstance(X_cols, str):
        X_cols = [X_cols]
    X_cols = list(X_cols)
    if not X_cols:
        raise ValueError("fit_pairwise_logistic: X_cols is empty")
    if qid_col in X_cols or relevance_col in X_cols:
        raise ValueError(
            f"fit_pairwise_logistic: {qid_col!r} or {relevance_col!r} "
            f"must not be in X_cols")
    if qid_col == relevance_col:
        raise ValueError(
            "fit_pairwise_logistic: qid_col == relevance_col")
    if sigma <= 0:
        raise ValueError(
            f"fit_pairwise_logistic: sigma > 0; got {sigma}")
    if n_epochs < 1:
        raise ValueError(
            f"fit_pairwise_logistic: n_epochs >= 1; got {n_epochs}")
    if lr <= 0:
        raise ValueError(
            f"fit_pairwise_logistic: lr > 0; got {lr}")
    if max_query_size < 2:
        raise ValueError(
            f"fit_pairwise_logistic: max_query_size >= 2; got {max_query_size}")
    if merge_fan_in < 2:
        raise ValueError(
            f"fit_pairwise_logistic: merge_fan_in >= 2; got {merge_fan_in}")
    if sort_read_batch_size < 1:
        raise ValueError(
            f"fit_pairwise_logistic: sort_read_batch_size >= 1; "
            f"got {sort_read_batch_size}")
    if phase1_buffer_bytes is not None and phase1_buffer_bytes < 1:
        raise ValueError(
            f"fit_pairwise_logistic: phase1_buffer_bytes >= 1; "
            f"got {phase1_buffer_bytes}")

    for c in X_cols + [qid_col, relevance_col]:
        if c not in df.columns:
            raise KeyError(f"fit_pairwise_logistic: column {c!r} not in df")

    schema = df._dataset.schema
    for c in X_cols:
        t = schema.field(c).type
        if not (pa.types.is_integer(t) or pa.types.is_floating(t)):
            raise TypeError(
                f"fit_pairwise_logistic: X column {c!r} has type {t}; "
                f"numeric required")
    for c in [qid_col, relevance_col]:
        t = schema.field(c).type
        if not (pa.types.is_integer(t) or pa.types.is_floating(t)):
            raise TypeError(
                f"fit_pairwise_logistic: {c!r} has type {t}; "
                f"numeric required")

    mg = memory_guard or df._memory_guard
    d = len(X_cols)
    n_total = int(df.count())
    if n_total == 0:
        raise ValueError("fit_pairwise_logistic: empty input")
    if batch_size is None:
        batch_size = 1000
    batch_size = max(1, int(batch_size))

    # State guard: O(d), before any np.zeros
    mg.check_state(d * 8 + 256, name="pairwise_logistic.state")
    coef = np.zeros(d, dtype=np.float64)

    sort_dir = tempfile.mkdtemp(prefix="pairwise_logistic_sort_")
    try:
        cols_sorted = [qid_col, relevance_col] + X_cols
        # Bound external_sort Phase 1 buffer to ~2 max-sized queries.
        # Phase 1 buffer grows with n up to 0.2*budget without this;
        # write_run peak ~4x buffer = 8*max_query_size*(d+2)*8.
        # P25c measured 13x ratio at nq=2000 vs 200 without bound.
        _p1_buffer = phase1_buffer_bytes
        if _p1_buffer is None:
            _p1_buffer = 2 * int(max_query_size) * (d + 2) * 8
        sorted_path = external_sort(
            df._iter_filtered_batches(batch_size=batch_size),
            [qid_col], cols_sorted,
            mg, sort_dir, ascending=True, log=False,
            merge_fan_in=int(merge_fan_in),
            read_batch_size=int(sort_read_batch_size),
            max_buffer_bytes=int(_p1_buffer))

        # First pass: validate group sizes + count queries + total pairs
        n_queries = 0
        max_size_seen = 0
        for qid, X_q, y_q in _iter_query_groups(
                sorted_path, qid_col, cols_sorted, mg):
            n_queries += 1
            gs = X_q.shape[0]
            if gs > max_query_size:
                raise MemoryLimitExceeded(
                    f"fit_pairwise_logistic: query {qid!r} has {gs} rows "
                    f"> max_query_size={max_query_size}. RankNet pairs are "
                    f"O(n_q^2); raise max_query_size (with proportional "
                    f"RAM cost) or re-split queries. Honest refuse.")
            if gs > max_size_seen:
                max_size_seen = gs
        if n_queries < 2:
            raise ValueError(
                f"fit_pairwise_logistic: need >= 2 queries; "
                f"got n_queries={n_queries}")

        # SGD epochs
        final_loss = float("nan")
        for epoch in range(int(n_epochs)):
            epoch_loss = 0.0
            epoch_queries = 0
            for qid, X_q, y_q in _iter_query_groups(
                    sorted_path, qid_col, cols_sorted, mg):
                n_q = X_q.shape[0]
                if n_q < 2:
                    continue
                # Per-query transient: s_q + grad (both (n_q,))
                mg.check_state(
                    n_q * 8 * 2 + 256,
                    name="pairwise_logistic.query_transient")
                s = X_q @ coef
                grad = ranknet_grad(s, y_q, sigma=float(sigma))
                # ranknet_grad returns direction-to-move-s (negative
                # gradient of loss; see _ltr_common.ranknet_grad
                # docstring "grad_i += lambda_ij"). Empirically:
                # y=[1,0], s=[0,0] -> grad=[+0.5*sig, -0.5*sig].
                # Move s in that direction: coef += lr * X.T @ grad.
                coef += lr * (X_q.T @ grad) / n_q
                # loss: 0.5 * mean(grad^2) (proxy; ranknet_grad gives dL/ds)
                epoch_loss += 0.5 * float(np.mean(grad ** 2))
                epoch_queries += 1
            final_loss = epoch_loss / max(epoch_queries, 1)
            log_fit("pairwise_logistic", iter_i=epoch + 1,
                    iter_n=int(n_epochs), loss=final_loss,
                    state_bytes=int(d * 8 + 256))

        model = PairwiseLogisticModel(
            coef_=coef,
            feature_names_=list(X_cols),
            query_col_=qid_col,
            relevance_col_=relevance_col,
            sigma_=float(sigma),
            lr_=float(lr),
            n_epochs_=int(n_epochs),
            n_queries_=int(n_queries),
            max_query_size_=int(max_size_seen),
            final_loss_=float(final_loss),
        )
        log_fit("pairwise_logistic", state_bytes=model.state_nbytes())
        return model
    finally:
        shutil.rmtree(sort_dir, ignore_errors=True)
