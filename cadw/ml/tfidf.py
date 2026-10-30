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
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.core.spill import SpillArray, AutoArrayResult, SpillBuffer
from cadw.ml._logging import log_fit
from cadw.ml.lda import tokenize


# v0.9.98: sparse row schema for transform_sparse (CSR-like).
# int32 for indices (vocab_size default 1e4; max 2^31), float64 for
# values (matches _row() dtype → sklearn reference bit-identical).
SPARSE_SCHEMA = pa.schema([
    pa.field("indices", pa.list_(pa.int32())),
    pa.field("values",  pa.list_(pa.float64())),
])


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

    def transform_sparse(self, df, *, text_col, norm="l2",
                         memory_guard=None, output: str = "auto"):
        """Sparse (CSR-like) rows, spillable via SpillBuffer.

        v0.9.98. Returns one (indices, values) pair per document. Each
        pair is a CSR-style row of the D x V matrix; zero entries are
        omitted. RAM is O(batch * avg_nnz_per_doc) when output is
        "auto" or "disk"; only output="memory" holds O(D) RAM.

        NOT a CadwDataFrame: list-typed columns break the tabular
        contract (sort/groupby/join do not apply to list cells). This
        is an explicit exception to v0.9.95 unified-CadwDataFrame.
        For a tabular return use transform_dense(output=...) instead.

        Parameters
        ----------
        df : CadwDataFrame
        text_col : str
        norm : {"l2", "l1", None}, default "l2"
            Passed to _row().
        memory_guard : MemoryGuard | None
            Default: df._memory_guard.
        output : {"memory", "auto", "disk"}, default "auto"

            "memory" -> list[tuple[np.ndarray int64, np.ndarray float64]]
                        (backward-compat with transform())
            "auto"   -> AutoResult; iterable pa.RecordBatch
                        with SPARSE_SCHEMA (in-memory if fits budget)
            "disk"   -> AutoResult (OnDiskResult; forces spill)

        Returns
        -------
        list[tuple] when output="memory"; AutoResult otherwise.

        Raises
        ------
        ValueError
            output not in {"memory", "auto", "disk"}.
        KeyError
            text_col not in df.columns.
        """
        if output not in ("memory", "auto", "disk"):
            raise ValueError(
                f"tfidf.transform_sparse: output must be 'memory', "
                f"'auto' or 'disk'; got {output!r}")
        if text_col not in df.columns:
            raise KeyError(
                f"tfidf.transform_sparse: {text_col!r} not in {df.columns}")

        mg = memory_guard or df._memory_guard

        if output == "memory":
            rows = []
            for b in df._iter_filtered_batches():
                mg.check_batch(b, name="tfidf.transform_sparse",
                               extra_factor=3.0)
                for t in b.column(text_col).to_pylist():
                    rows.append(self._row(t, norm))
            return rows

        buf = SpillBuffer(
            guard=mg,
            schema=SPARSE_SCHEMA,
            name="tfidf.sparse",
            spill_ratio=0.0 if output == "disk" else 0.5,
        )
        for b in df._iter_filtered_batches():
            mg.check_batch(b, name="tfidf.transform_sparse",
                           extra_factor=3.0)
            texts = b.column(text_col).to_pylist()
            idx_lists = []
            val_lists = []
            for t in texts:
                idx, val = self._row(t, norm)
                idx_lists.append(idx.astype(np.int32).tolist())
                val_lists.append(val.tolist())
            rb = pa.RecordBatch.from_arrays([
                pa.array(idx_lists, type=pa.list_(pa.int32())),
                pa.array(val_lists, type=pa.list_(pa.float64())),
            ], schema=SPARSE_SCHEMA)
            buf.write_batch(rb)
        return buf.finalize()

    def transform_dense(self, df, *, text_col, norm="l2",
                        memory_guard=None, output: str = "auto"):
        """Dense D x V matrix (small inputs / tests only).

        v0.9.95 (BREAKING): unified CadwDataFrame return.
        output= selects storage strategy only.
        Columns: ["c0", ..., "c{V-1}"].
        """
        if output not in ("memory", "auto", "disk"):
            raise ValueError(
                f"tfidf.transform_dense: output must be 'memory', "
                f"'auto' or 'disk'; got {output!r}")
        mg = memory_guard or df._memory_guard
        V = len(self.vocab_)

        if output == "memory":
            out = []
            for b in df._iter_filtered_batches():
                mg.check_batch(b, name="tfidf.transform_dense",
                               extra_factor=3.0)
                for t in b.column(text_col).to_pylist():
                    idx, val = self._row(t, norm)
                    row = np.zeros(V)
                    row[idx] = val
                    out.append(row)
            raw = np.vstack(out) if out else np.zeros((0, V))
        else:
            buf = SpillArray(guard=mg, width=V, dtype=np.float64,
                             name="tfidf.dense",
                             spill_ratio=0.0 if output == "disk" else 0.5)
            for b in df._iter_filtered_batches():
                mg.check_batch(b, name="tfidf.transform_dense",
                               extra_factor=3.0)
                texts = b.column(text_col).to_pylist()
                mat = np.zeros((len(texts), V))
                for i, t in enumerate(texts):
                    idx, val = self._row(t, norm)
                    mat[i, idx] = val
                buf.write_array(mat)
            raw = buf.finalize()

        # v0.9.95 (BREAKING): unified CadwDataFrame return.
        from cadw.dataframe import CadwDataFrame
        return CadwDataFrame.from_producer_result(
            raw, col_names=[f"c{i}" for i in range(V)],
            memory_guard=mg)


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
