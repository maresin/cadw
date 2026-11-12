"""ListNet (Cao et al. 2007): listwise probabilistic LTR.

Reference: Cao, Qin, Liu, Tsai, Li (2007). Learning to rank: from
pairwise approach to listwise approach. ICML.

Top-1 listwise loss (softmax cross-entropy):
  P_y = softmax(y)       (over items in one query)
  P_s = softmax(s)       (s_i = coef·x_i)
  L   = -Σ_i P_y(i) · log P_s(i)
  ∂L/∂s_k = P_s(k) - P_y(k)  (positive CE gradient)

`listnet_grad(s, y)` returns **positive CE gradient** (P31: source
is softmax(s) - softmax(y) = ∂L/∂s; docstring claim of 'negative'
was wrong). So SGD step:
  coef -= lr * X_q.T @ listnet_grad(s_q, y_q) / n_q

Query grouping: external_sort by qid_col (PairwiseLogistic precedent).
Per-query transient O(n_q·d); output n-sized, guarded (predict_batched:
false, honest materialization).

OOC: state = coef_(d)*f64 + 256B header = O(d) (hyperparam, not n).
Fit peak = O(max_query_size·d) + O(max_query_size·d) (external_sort
Phase 1 bounded via max_buffer_bytes) + O(batch·row_size) (input read).
Not bounded_by_domain — bounded_state via max_query_size + phase1_buf.
"""
from __future__ import annotations

from dataclasses import dataclass
import shutil
import tempfile
from typing import Optional, Sequence

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ml._logging import log_fit
from cadw.ml._ltr_common import listnet_grad
from cadw.ml.pairwise_logistic import (
    _iter_query_groups, _stack_X,
    _MAX_QUERY_SIZE_DEFAULT, _MERGE_FAN_IN_DEFAULT,
    _SORT_READ_BATCH_DEFAULT,
)
from cadw.ops.sort import external_sort


def _ce_per_query(s: np.ndarray, y: np.ndarray) -> float:
    """Listwise CE(P_y || P_s) with stable softmax; log(0) floored at 1e-30."""
    s_max = float(s.max())
    ss = np.exp(s - s_max)
    ss /= ss.sum()
    y_max = float(y.max())
    sy = np.exp(y - y_max)
    sy /= sy.sum()
    return -float((sy * np.log(ss + 1e-30)).sum())


@dataclass
class ListNetModel:
    """ListNet linear scorer. State = coef_(d)*f64 + 256B."""
    coef_: np.ndarray
    feature_names_: list
    query_col_: str
    relevance_col_: str
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
                       name="listnet.predict_output")
        out = np.empty(n_total, dtype=np.float64)
        off = 0
        _bs = batch_size if batch_size is not None else 1000
        for b in df._iter_filtered_batches(batch_size=_bs):
            mg.check_batch(b, name="listnet.predict", extra_factor=2.0)
            X = _stack_X(b, self.feature_names_)
            out[off:off + b.num_rows] = X @ self.coef_
            off += b.num_rows
        if off != n_total:
            out = out[:off]
        return CadwDataFrame.from_table(
            pa.table({f"{self.relevance_col_}_score":
                      pa.array(out, type=pa.float64())}),
            memory_guard=mg)


def fit_listnet(
    df,
    X_cols: "str | Sequence[str]",
    qid_col: str,
    relevance_col: str,
    *,
    n_epochs: int = 20,
    lr: float = 0.01,
    max_query_size: int = _MAX_QUERY_SIZE_DEFAULT,
    merge_fan_in: int = _MERGE_FAN_IN_DEFAULT,
    sort_read_batch_size: int = _SORT_READ_BATCH_DEFAULT,
    phase1_buffer_bytes: "int | None" = None,
    memory_guard: Optional[MemoryGuard] = None,
    batch_size: "int | None" = None,
) -> ListNetModel:
    """Fit ListNet (listwise softmax CE on per-query items), OOC in n.

    Args:
      df: CadwDataFrame.
      X_cols: feature columns (numeric).
      qid_col: query/group identifier column.
      relevance_col: relevance labels (numeric; softmax applied
        per-query, so absolute scale irrelevant, only relative).
      n_epochs: SGD passes over sorted queries. Default 20.
      lr: learning rate (>0). Default 0.01.
      max_query_size: guard; refuse if any query has more rows.
      merge_fan_in, sort_read_batch_size, phase1_buffer_bytes:
        external_sort controls (mirror PairwiseLogistic).
      memory_guard, batch_size: standard.
    """
    if isinstance(X_cols, str):
        X_cols = [X_cols]
    X_cols = list(X_cols)
    if not X_cols:
        raise ValueError("fit_listnet: X_cols is empty")
    if qid_col in X_cols or relevance_col in X_cols:
        raise ValueError(
            f"fit_listnet: {qid_col!r} or {relevance_col!r} "
            f"must not be in X_cols")
    if qid_col == relevance_col:
        raise ValueError("fit_listnet: qid_col == relevance_col")
    if n_epochs < 1:
        raise ValueError(
            f"fit_listnet: n_epochs >= 1; got {n_epochs}")
    if lr <= 0:
        raise ValueError(
            f"fit_listnet: lr > 0; got {lr}")
    if max_query_size < 2:
        raise ValueError(
            f"fit_listnet: max_query_size >= 2; got {max_query_size}")
    if merge_fan_in < 2:
        raise ValueError(
            f"fit_listnet: merge_fan_in >= 2; got {merge_fan_in}")
    if sort_read_batch_size < 1:
        raise ValueError(
            f"fit_listnet: sort_read_batch_size >= 1; "
            f"got {sort_read_batch_size}")
    if phase1_buffer_bytes is not None and phase1_buffer_bytes < 1:
        raise ValueError(
            f"fit_listnet: phase1_buffer_bytes >= 1; "
            f"got {phase1_buffer_bytes}")

    for c in X_cols + [qid_col, relevance_col]:
        if c not in df.columns:
            raise KeyError(f"fit_listnet: column {c!r} not in df")

    schema = df._dataset.schema
    for c in X_cols:
        t = schema.field(c).type
        if not (pa.types.is_integer(t) or pa.types.is_floating(t)):
            raise TypeError(
                f"fit_listnet: X column {c!r} has type {t}; "
                f"numeric required")
    for c in [qid_col, relevance_col]:
        t = schema.field(c).type
        if not (pa.types.is_integer(t) or pa.types.is_floating(t)):
            raise TypeError(
                f"fit_listnet: {c!r} has type {t}; numeric required")

    mg = memory_guard or df._memory_guard
    d = len(X_cols)
    n_total = int(df.count())
    if n_total == 0:
        raise ValueError("fit_listnet: empty input")
    if batch_size is None:
        batch_size = 1000
    batch_size = max(1, int(batch_size))

    mg.check_state(d * 8 + 256, name="listnet.state")
    coef = np.zeros(d, dtype=np.float64)

    sort_dir = tempfile.mkdtemp(prefix="listnet_sort_")
    try:
        cols_sorted = [qid_col, relevance_col] + X_cols
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

        # Validation pass
        n_queries = 0
        max_size_seen = 0
        for qid, X_q, y_q in _iter_query_groups(
                sorted_path, qid_col, cols_sorted, mg):
            n_queries += 1
            gs = X_q.shape[0]
            if gs > max_query_size:
                raise MemoryLimitExceeded(
                    f"fit_listnet: query {qid!r} has {gs} rows "
                    f"> max_query_size={max_query_size}. ListNet "
                    f"softmax is O(n_q) but per-query transient "
                    f"X_q·d grows; raise max_query_size or re-split "
                    f"queries.")
            if gs > max_size_seen:
                max_size_seen = gs
        if n_queries < 2:
            raise ValueError(
                f"fit_listnet: need >= 2 queries; got {n_queries}")

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
                mg.check_state(
                    n_q * 8 * 2 + 256,
                    name="listnet.query_transient")
                s = X_q @ coef
                # P31: listnet_grad returns POSITIVE CE gradient
                # (softmax(s) - softmax(y) = ∂L/∂s). Different
                # convention from ranknet_grad (which returns
                # negative direction-to-move, per P20). SGD step
                # for positive grad is MINUS lr * X.T @ grad.
                grad = listnet_grad(s, y_q)  # ∂L/∂s
                coef -= lr * (X_q.T @ grad) / n_q
                epoch_loss += _ce_per_query(s, y_q)
                epoch_queries += 1
            final_loss = epoch_loss / max(epoch_queries, 1)
            log_fit("listnet", iter_i=epoch + 1,
                    iter_n=int(n_epochs), loss=final_loss,
                    state_bytes=int(d * 8 + 256))

        model = ListNetModel(
            coef_=coef,
            feature_names_=list(X_cols),
            query_col_=qid_col,
            relevance_col_=relevance_col,
            lr_=float(lr),
            n_epochs_=int(n_epochs),
            n_queries_=int(n_queries),
            max_query_size_=int(max_size_seen),
            final_loss_=float(final_loss),
        )
        log_fit("listnet", state_bytes=model.state_nbytes())
        return model
    finally:
        shutil.rmtree(sort_dir, ignore_errors=True)
