"""NMF (Lee & Seung 1999/2001) — out-of-core via mini-batch.

Promoted from .pingis/impl/J/nmf.py (DEC-058). Two YAML consumers
share this function: C_dim_reduction.yaml::NMF (baseline) and
J_nlp.yaml::NMF (sota).

Input: COO columns (doc_col, term_col, count_col). For text input,
build COO via tfidf.fit(...).transform_sparse(...) first.

State = H (k x V), O(k*V), bounded. Fit is multi-pass streaming:
each pass iterates df._iter_filtered_batches() and updates H via
cadw.ops.linalg.nmf_mu. W_b (batch doc-topic) is per-batch only.

bounded_state: true (H). Not OOC in V (V must fit in state_limit;
raises MemoryLimitExceeded otherwise — refuse-at-guard, honest).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ops.linalg import nmf_mu


_HEADER_BYTES = 128


def _coo_to_dense(
    batch: pa.RecordBatch,
    *,
    doc_col: str, term_col: str, count_col: str,
    V: int,
    mg=None,
) -> tuple[np.ndarray, int, int]:
    """Convert COO batch to dense (n_docs_in_batch, V).

    Returns (X, min_doc_id, n_docs_in_batch).
    min_doc_id is returned so the caller can stream W with original ids.
    """
    docs = np.asarray(batch.column(doc_col).to_numpy(zero_copy_only=False),
                      dtype=np.int64)
    terms = np.asarray(batch.column(term_col).to_numpy(zero_copy_only=False),
                       dtype=np.int64)
    cnts = np.asarray(batch.column(count_col).to_numpy(zero_copy_only=False),
                      dtype=np.float64)
    if docs.size == 0:
        if mg is not None:
            mg.check_state(0, name="nmf.coo.empty")
        return np.zeros((0, V), dtype=np.float64), 0, 0
    d_min = int(docs.min())
    d_max = int(docs.max())
    n = d_max - d_min + 1
    if mg is not None:
        mg.check_state(n * V * 8, name="nmf.coo.dense")
    X = np.zeros((n, V), dtype=np.float64)
    np.add.at(X, (docs - d_min, terms), cnts)
    return X, d_min, n


def _infer_vocab_size(df, term_col: str, *, sample_bs: int = 2000) -> int:
    """Sample first batch -> vocab_size = max(term_id)+1. Warn if sparse."""
    try:
        first = next(df._iter_filtered_batches(batch_size=sample_bs))
    except StopIteration:
        raise ValueError("empty dataframe: cannot infer vocab_size")
    terms = first.column(term_col).to_numpy(zero_copy_only=False)
    if len(terms) == 0:
        raise ValueError("empty first batch: cannot infer vocab_size")
    V = int(np.max(terms)) + 1
    seen = len(np.unique(terms))
    if seen < V * 0.5:
        warnings.warn(
            f"vocab_size inferred as {V} from first batch of "
            f"{first.num_rows} rows, but only {seen} distinct terms seen. "
            f"Rare terms may be missing. Pass vocab_size explicitly if known.",
            stacklevel=4,
        )
    return V


def _resolve_batch_size(mg, V: int, k: int) -> int:
    """Auto-derive batch_size from MemoryGuard. Transient per row =
    V*f64 (dense X) + k*f64 (W_b) + ~24 (Arrow 3x i64). State = k*V*f64.
    """
    state_bytes = k * V * 8 + _HEADER_BYTES
    budget = mg.available_bytes - state_bytes
    if budget < V * 8:
        raise MemoryLimitExceeded(
            f"nmf: vocab_size={V} (dense batch row = {V*8} B) does not fit "
            f"with state H (k*V*8 = {state_bytes} B); budget = "
            f"{mg.available_bytes} B. Increase memory_guard or reduce V."
        )
    per_row = V * 8 + k * 8 + 24
    return max(1, int(budget // per_row))


@dataclass
class NMFModel:
    """Fitted NMF: X ≈ W · H, W >= 0, H >= 0."""
    components_: np.ndarray            # H (k, V)
    n_features_: int                   # V
    k_: int                            # k
    reconstruction_error_: float
    n_iter_: int
    eps_: float = 1e-10

    def state_nbytes(self) -> int:
        return int(self.components_.nbytes + _HEADER_BYTES)

    def transform_batches(
        self,
        df,
        *,
        doc_col: str, term_col: str, count_col: str,
        batch_size: int | None = None,
        memory_guard=None,
    ) -> Iterator[pa.RecordBatch]:
        """Stream per-batch W (b x k) given H. Caller controls iteration
        (OOC-safe: caller can consume one batch at a time)."""
        mg = memory_guard or df._memory_guard
        H = self.components_
        k = self.k_
        HtH_inv = np.linalg.inv(H @ H.T + self.eps_ * np.eye(k, dtype=H.dtype))
        for b in df._iter_filtered_batches(batch_size=batch_size):
            mg.check_state(b.num_rows * self.n_features_ * 8 + b.num_rows * k * 8,
                           name="nmf.transform.batch")
            X, _, _ = _coo_to_dense(
                b, doc_col=doc_col, term_col=term_col, count_col=count_col,
                V=self.n_features_, mg=mg)
            if X.shape[0] == 0:
                continue
            W_b = np.maximum(X @ H.T @ HtH_inv, 0.0)
            yield pa.RecordBatch.from_arrays(
                [pa.array(W_b[:, j], type=pa.float64()) for j in range(k)],
                names=[f"w{j}" for j in range(k)],
            )


def fit_nmf(
    df, *,
    doc_col: str, term_col: str, count_col: str,
    n_components: int,
    vocab_size: int | None = None,
    n_epochs: int = 30,
    memory_guard=None,
    batch_size: int | None = None,
    seed: int = 0,
    eps: float = 1e-10,
) -> NMFModel:
    """Fit NMF on a COO doc-term matrix (streaming mini-batch mu).

    vocab_size=None -> sample-infer from first batch (warn if sparse).
    State = H (k x V). n_epochs full passes over df.
    """
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}")
    mg = memory_guard or MemoryGuard(max_memory_mb=512)
    if vocab_size is None:
        V = _infer_vocab_size(df, term_col)
    else:
        V = int(vocab_size)
    if V < 1:
        raise ValueError(f"vocab_size must be >= 1, got {V}")
    k = int(n_components)

    state_bytes = k * V * 8 + _HEADER_BYTES
    mg.check_state(state_bytes, name="nmf.H")

    if batch_size is None:
        batch_size = _resolve_batch_size(mg, V, k)

    rng = np.random.default_rng(seed)
    H = rng.random((k, V)).astype(np.float64) + 0.1

    for _epoch in range(n_epochs):
        for b in df._iter_filtered_batches(batch_size=batch_size):
            if b.num_rows == 0:
                continue
            # guard for the b*V dense alloc (upper bound: num_rows * V)
            mg.check_state(b.num_rows * V * 8, name="nmf.dense_X")
            X, _d_min, _n = _coo_to_dense(
                b, doc_col=doc_col, term_col=term_col, count_col=count_col,
                V=V, mg=mg)
            if X.shape[0] == 0:
                continue
            H = nmf_mu(X, H, eps=eps)

    model = NMFModel(
        components_=H, n_features_=V, k_=k,
        reconstruction_error_=0.0, n_iter_=n_epochs, eps_=eps,
    )
    return model
