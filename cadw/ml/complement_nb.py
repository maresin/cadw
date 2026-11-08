"""Complement Naive Bayes (Rennie et al. 2003) — out-of-core.

Promoted from .pingis/impl/J/complement_nb.py (DEC-058).
See J_nlp.yaml::ComplementNB card.

State: (C, V) int64 class-term counts + (C,) totals = O(C*V).
Fit: single streaming Reduce pass over COO (doc_id, term_id, count, label).
Predict: streaming argmin of complement log-likelihood.

bounded_state: true (C*V user params). Not OOC in V: refuses at guard
if C*V*8 > state_limit_bytes (honest per AI_CONSTITUTION §2 p.6).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ops.reduce import ReduceOp, reduce


_HEADER_BYTES = 256


class _CNBStatsOp(ReduceOp):
    """Per-class term counts: state = (C, V) int64 dense + (C,) totals."""

    def __init__(
        self, n_classes: int, vocab_size: int, *,
        label_col: str = "label",
        term_col: str = "term_id",
        count_col: str = "count",
        memory_guard=None,
    ) -> None:
        if n_classes < 1 or vocab_size < 1:
            raise ValueError("n_classes and vocab_size must be >= 1")
        self.n_classes = int(n_classes)
        self.vocab_size = int(vocab_size)
        self.label_col = label_col
        self.term_col = term_col
        self.count_col = count_col
        self._mg = memory_guard

    def estimate_state_nbytes(self) -> int:
        return int(8 * self.n_classes * self.vocab_size
                   + 8 * self.n_classes + _HEADER_BYTES)

    def init(self):
        # G5/G2: guard BEFORE np.zeros (AST-gate requires explicit
        # check_state inside function, not just caller).
        if self._mg is not None:
            self._mg.check_state(
                self.estimate_state_nbytes(), name="complement_nb.init")
        return {
            "counts": np.zeros((self.n_classes, self.vocab_size),
                               dtype=np.int64),
            "totals": np.zeros(self.n_classes, dtype=np.int64),
        }

    def update(self, state, batch: pa.RecordBatch):
        if batch.num_rows == 0:
            return state
        labels = np.asarray(
            batch.column(self.label_col).to_numpy(zero_copy_only=False),
            dtype=np.int64)
        terms = np.asarray(
            batch.column(self.term_col).to_numpy(zero_copy_only=False),
            dtype=np.int64)
        cnts = np.asarray(
            batch.column(self.count_col).to_numpy(zero_copy_only=False),
            dtype=np.int64)
        valid = ((labels >= 0) & (labels < self.n_classes)
                 & (terms >= 0) & (terms < self.vocab_size))
        if not valid.all():
            labels = labels[valid]; terms = terms[valid]; cnts = cnts[valid]
        if labels.size == 0:
            return state
        idx = labels * self.vocab_size + terms
        np.add.at(state["counts"].reshape(-1), idx, cnts)
        np.add.at(state["totals"], labels, cnts)
        return state

    def merge(self, a, b):
        return {
            "counts": a["counts"] + b["counts"],
            "totals": a["totals"] + b["totals"],
        }

    def state_nbytes(self, state) -> int:
        return int(state["counts"].nbytes + state["totals"].nbytes
                   + _HEADER_BYTES)

    def finalize(self, state):
        return state


@dataclass
class ComplementNBModel:
    classes_: np.ndarray               # (C,) int64
    feature_log_prob_: np.ndarray      # (C, V) f64
    class_log_prior_: np.ndarray       # (C,) f64
    class_totals_: np.ndarray          # (C,) int64
    vocab_size_: int
    n_classes_: int
    alpha_: float

    def state_nbytes(self) -> int:
        return int(self.feature_log_prob_.nbytes
                   + self.class_log_prior_.nbytes
                   + self.class_totals_.nbytes
                   + self.classes_.nbytes
                   + _HEADER_BYTES)

    def predict_batches(
        self,
        df, *,
        term_col: str = "term_id",
        count_col: str = "count",
        doc_col: str = "doc_id",
        batch_size: int | None = None,
        memory_guard=None,
    ) -> Iterator[pa.RecordBatch]:
        """Stream per-batch predicted labels (argmin complement score).

        Caller controls iteration; RAM O(batch). Null docs (NaN counts)
        propagate as no prediction (skipped).
        """
        mg = memory_guard or df._memory_guard
        V = self.vocab_size_
        C = self.n_classes_
        log_prior = self.class_log_prior_
        log_prob = self.feature_log_prob_
        for batch in df._iter_filtered_batches(batch_size=batch_size):
            if batch.num_rows == 0:
                continue
            n = batch.num_rows
            # guard: per-batch dense X = n_docs * V
            mg.check_state(n * V * 8, name="complement_nb.predict_batch")
            terms = np.asarray(
                batch.column(term_col).to_numpy(zero_copy_only=False),
                dtype=np.int64)
            cnts = np.asarray(
                batch.column(count_col).to_numpy(zero_copy_only=False),
                dtype=np.float64)
            if doc_col in batch.schema.names:
                doc_ids = np.asarray(
                    batch.column(doc_col).to_numpy(zero_copy_only=False),
                    dtype=np.int64)
            else:
                doc_ids = np.zeros(terms.shape[0], dtype=np.int64)
            n_docs = int(doc_ids.max()) + 1 if doc_ids.size else 0
            X = np.zeros((n_docs, V), dtype=np.float64)
            np.add.at(X, (doc_ids, terms), cnts)
            scores = X @ log_prob.T + log_prior
            pred = np.argmin(scores, axis=1).astype(np.int64)
            yield pa.RecordBatch.from_arrays(
                [pa.array(pred, type=pa.int64())],
                names=["predicted_label"],
            )


def _infer_class_vocab(
    df, *, label_col: str, term_col: str,
    n_classes: Optional[int], vocab_size: Optional[int],
    sample_bs: int = 2000,
) -> tuple[int, int]:
    """Sample first batch -> infer missing (n_classes, vocab_size)."""
    try:
        first = next(df._iter_filtered_batches(batch_size=sample_bs))
    except StopIteration:
        raise ValueError("complement_nb: empty dataframe")
    labels = first.column(label_col).to_numpy(zero_copy_only=False)
    terms = first.column(term_col).to_numpy(zero_copy_only=False)
    if n_classes is None:
        if len(labels) == 0:
            raise ValueError("complement_nb: empty label column")
        n_classes = int(np.max(labels)) + 1
        seen = len(np.unique(labels))
        if seen < n_classes * 0.5:
            warnings.warn(
                f"n_classes inferred as {n_classes} from first batch of "
                f"{first.num_rows} rows, but only {seen} distinct labels "
                f"seen. Pass n_classes explicitly if known.",
                stacklevel=4,
            )
    if vocab_size is None:
        if len(terms) == 0:
            raise ValueError("complement_nb: empty term column")
        vocab_size = int(np.max(terms)) + 1
        seen = len(np.unique(terms))
        if seen < vocab_size * 0.5:
            warnings.warn(
                f"vocab_size inferred as {vocab_size} from first batch of "
                f"{first.num_rows} rows, but only {seen} distinct terms "
                f"seen. Pass vocab_size explicitly if known.",
                stacklevel=4,
            )
    return int(n_classes), int(vocab_size)


def _resolve_batch_size(mg, C: int, V: int) -> int:
    """Auto-derive batch_size. Fit transient: 3 cols (i64) + linear index
    (batch * i64) + 2 masked copies. Upper-bound row bytes ~ 5 * 8 = 40.
    """
    state_bytes = 8 * C * V + 8 * C + _HEADER_BYTES
    budget = mg.available_bytes - state_bytes
    if budget < 64:
        raise MemoryLimitExceeded(
            f"complement_nb: state {state_bytes} B (C={C}, V={V}) exceeds "
            f"guard budget {mg.available_bytes} B. Increase memory_guard "
            f"or reduce (n_classes, vocab_size)."
        )
    per_row = 5 * 8  # labels, terms, cnts, idx, transient
    return max(1, int(budget // per_row))


def fit_complement_nb(
    df, *,
    label_col: str = "label",
    term_col: str = "term_id",
    count_col: str = "count",
    n_classes: Optional[int] = None,
    vocab_size: Optional[int] = None,
    alpha: float = 1.0,
    memory_guard=None,
    batch_size: int | None = None,
) -> ComplementNBModel:
    """Fit Complement NB on COO doc-term matrix with per-doc labels.

    n_classes=None, vocab_size=None -> sample-infer from first batch
    (warn if sparse). Refuses at guard if C*V*8 > state_limit_bytes.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    mg = memory_guard or MemoryGuard(max_memory_mb=512)
    n_classes, vocab_size = _infer_class_vocab(
        df, label_col=label_col, term_col=term_col,
        n_classes=n_classes, vocab_size=vocab_size)
    est = 8 * n_classes * vocab_size + 8 * n_classes + _HEADER_BYTES
    if est > mg.state_limit_bytes:
        raise MemoryLimitExceeded(
            f"ComplementNB: dense C*V state ({est} B) > state_limit "
            f"({mg.state_limit_bytes} B); use smaller vocab or spill "
            f"variant (v0.9.144+)."
        )
    mg.check_state(est, name="complement_nb.preflight")

    if batch_size is None:
        batch_size = _resolve_batch_size(mg, n_classes, vocab_size)

    op = _CNBStatsOp(
        n_classes, vocab_size,
        label_col=label_col, term_col=term_col, count_col=count_col,
        memory_guard=mg,
    )
    stats = reduce(
        df._iter_filtered_batches(batch_size=batch_size), op, mg, log=False)
    counts = stats["counts"]
    totals = stats["totals"]
    C = counts.shape[0]; V = counts.shape[1]
    col_total = counts.sum(axis=0)
    grand_total = int(totals.sum())
    comp_counts = col_total[None, :] - counts
    comp_total = grand_total - totals
    # Complement log-prob (Rennie 2003):
    #   w[c, t] = log((alpha + sum_{c'!=c} N_{c't})
    #                / (alpha*V + sum_{c'!=c} N_{c'}))
    w = np.log(
        (alpha + comp_counts)
        / (alpha * V + comp_total[:, None]))
    class_log_prior = np.log(np.maximum(totals, 1) / max(grand_total, 1))
    return ComplementNBModel(
        classes_=np.arange(C, dtype=np.int64),
        feature_log_prob_=w.astype(np.float64),
        class_log_prior_=class_log_prior.astype(np.float64),
        class_totals_=totals.copy(),
        vocab_size_=V,
        n_classes_=C,
        alpha_=float(alpha),
    )
