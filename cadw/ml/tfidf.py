"""Streaming TF-IDF vectorizer (D.6.9, v0.9.41).

Reference: Salton, Buckley, *Term-Weighting Approaches in Automatic Text
Retrieval*, Information Processing & Management 24(5), 1988. Reference
implementation in tests: sklearn TfidfVectorizer (Pedregosa et al., JMLR
12, 2011).
See .pingis/algorithm_notes/d6_9_tfidf.md.

Two passes over the text column: pass 1 builds the document-frequency
vector (and the vocabulary for a bounded vocab_size), pass 2 applies the
sublinear/idf weighting. State is O(V) (idf + vocabulary), independent of
the number of documents. Sparse documents are produced as (indices,
values) pairs per row so a dense D x V matrix is never materialised.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from cadw.core.memory_guard import MemoryGuard
from cadw.ml._logging import log_fit
from cadw.ml.lda import tokenize


class TfidfModel:
    """Fitted TF-IDF: idf vector, vocabulary, transform -> sparse rows."""

    def __init__(self, vocab, idf, params):
        self.vocab_ = list(vocab)
        self._index = {w: i for i, w in enumerate(self.vocab_)}
        self.idf_ = np.asarray(idf, dtype=np.float64)
        self.params_ = dict(params)
        self.fitted_ = True

    def state_nbytes(self):
        return int(self.idf_.nbytes + 8 * len(self.vocab_) + 512)

    def to_dict(self):
        return {"vocab_size": len(self.vocab_), "params": self.params_}

    def _row(self, text, norm):
        tf = {}
        for tok in tokenize(text):
            j = self._index.get(tok)
            if j is not None:
                tf[j] = tf.get(j, 0) + 1
        if not tf:
            return np.zeros(0, np.int64), np.zeros(0)
        idx = np.fromiter(tf.keys(), dtype=np.int64, count=len(tf))
        cnt = np.fromiter(tf.values(), dtype=np.float64, count=len(tf))
        if self.params_["sublinear_tf"]:
            cnt = 1.0 + np.log(cnt)
        val = cnt * self.idf_[idx]
        if norm == "l2":
            nrm = np.sqrt(np.sum(val * val))
            if nrm > 0:
                val = val / nrm
        elif norm == "l1":
            nrm = np.sum(np.abs(val))
            if nrm > 0:
                val = val / nrm
        order = np.argsort(idx)
        return idx[order], val[order]

    def transform(self, df, *, text_col, norm="l2", memory_guard=None):
        """Yield (indices, values) sparse rows per document."""
        mg = memory_guard or df._memory_guard
        rows = []
        for b in df._iter_filtered_batches():
            mg.check_batch(b, name="tfidf.transform", extra_factor=3.0)
            for t in b.column(text_col).to_pylist():
                rows.append(self._row(t, norm))
        return rows

    def transform_dense(self, df, *, text_col, norm="l2", memory_guard=None):
        """Dense D x V matrix (small inputs / tests only)."""
        mg = memory_guard or df._memory_guard
        V = len(self.vocab_)
        out = []
        for b in df._iter_filtered_batches():
            mg.check_batch(b, name="tfidf.transform_dense", extra_factor=3.0)
            for t in b.column(text_col).to_pylist():
                idx, val = self._row(t, norm)
                row = np.zeros(V)
                row[idx] = val
                out.append(row)
        return np.vstack(out) if out else np.zeros((0, V))


def fit_tfidf(df, text_col, *, vocab_size=10000, sublinear_tf=True,
              min_df=1, norm="l2", memory_guard: Optional[MemoryGuard] = None):
    """Fit IDF weights by streaming the document-frequency vector."""
    if vocab_size < 2:
        raise ValueError("fit_tfidf: vocab_size must be >= 2")
    if min_df < 1:
        raise ValueError("fit_tfidf: min_df must be >= 1")
    if norm not in ("l2", "l1", None):
        raise ValueError("fit_tfidf: norm must be 'l2', 'l1' or None")
    if text_col not in df._columns:
        raise KeyError(f"fit_tfidf: column {text_col!r} not in {df._columns}")
    mg = memory_guard or df._memory_guard
    dfc = {}
    n = 0
    for b in df._iter_filtered_batches():
        mg.check_batch(b, name="tfidf.fit", extra_factor=3.0)
        # v0.9.90-S3 (G2): pre-check BEFORE dict grows.
        mg.check_state(
            (len(dfc) + b.num_rows) * 96 + 1024, name="tfidf.state")
        for t in b.column(text_col).to_pylist():
            n += 1
            for tok in set(tokenize(t)):
                dfc[tok] = dfc.get(tok, 0) + 1
    if n == 0:
        raise ValueError("fit_tfidf: empty input")
    items = [(w, c) for w, c in dfc.items() if c >= min_df]
    items.sort(key=lambda kv: (-kv[1], kv[0]))
    items = items[:vocab_size]
    if len(items) < 2:
        raise ValueError("fit_tfidf: vocabulary has fewer than 2 tokens")
    vocab = [w for w, _ in items]
    dfv = np.array([c for _, c in items], dtype=np.float64)
    idf = np.log((1.0 + n) / (1.0 + dfv)) + 1.0
    model = TfidfModel(vocab, idf, {"sublinear_tf": bool(sublinear_tf),
                                    "min_df": min_df, "norm": norm,
                                    "n_docs": n})
    log_fit("tfidf", state_bytes=model.state_nbytes())
    mg.check_state(model.state_nbytes(), name="tfidf.model")
    return model
