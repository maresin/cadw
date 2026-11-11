"""Latent Semantic Analysis (TF-IDF + sparse randomized SVD).

Kluger/Deerwester 1990: TF-IDF doc-term matrix reduced by truncated
SVD to k dense factors (LSA / LSI embedding).

Pipeline:
  1. fit_tfidf(df, text_col, vocab_size) — streaming TF-IDF
  2. transform_iter (yields csr (b, V) batches)
  3. Randomized SVD on sparse via Y = X^T X Omega (V, L) — Halko 2011
  4. components_ (k, V), singular_values_ (k,)

OOC (Pattern #1 batch-layer, axis=n):
  - state = V*L*f64*2 (Y, Q) + L*L*f64 (B) + 256. Independent of n.
  - bounded_by_domain {V: 1e5}: V*L + L^2 at V=1e5, L~110 = 88 MB.
  - transform: X @ components.T per batch -> SpillArray (OOC in n).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.core.spill import SpillArray
from cadw.ml._logging import log_fit
from cadw.ml.tfidf import fit_tfidf, TfidfModel


@dataclass
class LsaModel:
    """LSA fit result.

    components_: (k, V) f64 — top-k right singular vectors of TF-IDF.
    singular_values_: (k,) f64 — sigma values.
    tfidf_model_: fitted TfidfModel (needed for transform of new df).
    transform(df) -> CadwDataFrame (n, k) with columns lsa_0..lsa_{k-1}.
    """
    components_: Optional[np.ndarray] = None
    singular_values_: Optional[np.ndarray] = None
    tfidf_model_: Optional[TfidfModel] = None
    text_col: str = ""
    n_components_: int = 0
    vocab_size_: int = 0
    n_docs_: int = 0

    def state_nbytes(self) -> int:
        sz = 256
        if self.components_ is not None:
            sz += int(self.components_.nbytes)
        if self.singular_values_ is not None:
            sz += int(self.singular_values_.nbytes)
        if self.tfidf_model_ is not None:
            sz += int(self.tfidf_model_.state_nbytes())
        return int(sz)

    def release_train_state(self) -> None:
        """Idempotent. Drops components_/singular_values_; transform raises."""
        self.components_ = None
        self.singular_values_ = None
        # tfidf_model_ kept (would be needed for transform; dropping it too
        # means transform raises cleanly via None check below).
        self.tfidf_model_ = None

    def to_dict(self) -> dict:
        return {
            "n_components": int(self.n_components_),
            "vocab_size": int(self.vocab_size_),
            "n_docs": int(self.n_docs_),
            "singular_values": (self.singular_values_.tolist()
                                if self.singular_values_ is not None else []),
        }

    def transform(self, df, *, text_col=None, batch_size=None,
                  memory_guard=None, output="auto"):
        """Project df -> (n, k) LSA embedding (OOC in n via SpillArray)."""
        if self.components_ is None or self.tfidf_model_ is None:
            raise RuntimeError("LsaModel.transform: model released")
        text_col = text_col or self.text_col
        mg = memory_guard or getattr(df, "_memory_guard", None)
        k = self.n_components_
        spill_ratio = {"memory": 1.0, "auto": 0.5, "disk": 0.0}.get(
            output, 0.5)
        buf = SpillArray(guard=mg, width=k, dtype=np.float64,
                         name="lsa.transform", spill_ratio=spill_ratio)
        comp_T = self.components_.T  # (V, k)
        for M_b in self.tfidf_model_.transform_iter(
                df, text_col=text_col, batch_size=batch_size,
                memory_guard=mg):
            Z_b = M_b @ comp_T
            if hasattr(Z_b, "toarray"):
                Z_b = Z_b.toarray()
            buf.write_array(np.asarray(Z_b, dtype=np.float64))
        result = buf.finalize()

        # Wrap as CadwDataFrame (spill-aware)
        from cadw.dataset.dataset import Dataset
        from cadw import CadwDataFrame
        schema = pa.schema([pa.field(f"lsa_{j}", pa.float64())
                            for j in range(k)])

        # If AutoArrayResult is in-memory, wrap directly; else stream.
        if hasattr(result, "shape"):
            Z = np.asarray(result)
            cols = {f"lsa_{j}": pa.array(Z[:, j], type=pa.float64())
                    for j in range(k)}
            return CadwDataFrame.from_table(
                pa.table(cols), memory_guard=mg,
                spill_dir=getattr(df, "_spill_dir", None))

        def _iter_z():
            for chunk in result:
                arr = np.asarray(chunk)
                if arr.ndim == 1:
                    arr = arr[:, None]
                yield pa.RecordBatch.from_arrays(
                    [pa.array(arr[:, j], pa.float64()) for j in range(k)],
                    schema=schema)

        ds = Dataset.from_batches(schema, _iter_z(), df._new_dataset_dir())
        return CadwDataFrame(ds, memory_guard=mg,
                             spill_dir=getattr(df, "_spill_dir", None))


def _fit_randomized_svd_sparse(M_iter_factory, V, k, n_oversamples,
                                 n_power_iter, seed, mg):
    """Randomized SVD on sparse (n, V) matrix via Y = X^T X Omega.

    Returns (components (k, V), singular_values (k,)).

    OOC: state = V*L*8*2 + L*L*8 + 256. Independent of n.
    M_iter_factory: callable returning fresh iterator of csr (b, V) batches.
    """
    L = min(V, k + n_oversamples)
    state_est = V * L * 8 * 2 + L * L * 8 + 256
    if mg is not None and hasattr(mg, "check_state"):
        mg.check_state(state_est, name="lsa.svd.state")

    rng = np.random.RandomState(seed)
    omega = rng.normal(size=(V, L))

    def _batches():
        for M_b in M_iter_factory():
            b_rows = M_b.shape[0]
            if mg is not None and hasattr(mg, "check_state"):
                mg.check_state(b_rows * L * 8 * 3 + 256,
                               name="lsa.svd.batch")
            yield M_b

    # Init Y = X^T X Omega (V, L)
    Y = np.zeros((V, L), dtype=np.float64)
    for M_b in _batches():
        Xomega = M_b @ omega
        if hasattr(Xomega, "toarray"):
            Xomega = Xomega.toarray()
        Y += M_b.T @ Xomega  # (V, b) @ (b, L) = (V, L)

    # Power iterations (each: Q=qr(Y), Y_next = X^T X Q)
    for _ in range(max(0, int(n_power_iter) - 1)):
        Q_cur, _ = np.linalg.qr(Y)
        Y_next = np.zeros((V, L), dtype=np.float64)
        for M_b in _batches():
            XQ = M_b @ Q_cur
            if hasattr(XQ, "toarray"):
                XQ = XQ.toarray()
            Y_next += M_b.T @ XQ
        Y = Y_next

    Q, _ = np.linalg.qr(Y)  # (V, L)

    # B = XQ^T XQ (L, L)
    B = np.zeros((L, L), dtype=np.float64)
    for M_b in _batches():
        XQ = M_b @ Q
        if hasattr(XQ, "toarray"):
            XQ = XQ.toarray()
        B += XQ.T @ XQ

    Uhat, sv, _ = np.linalg.svd(B, full_matrices=False)
    U = Q @ Uhat  # (V, L)
    k_use = min(int(k), U.shape[1])
    components = U[:, :k_use].T  # (k_use, V)
    singular_values = np.sqrt(np.maximum(sv[:k_use], 0.0))
    return components, singular_values


def fit_lsa(
    df,
    text_col,
    *,
    n_components: int = 100,
    vocab_size: int = 10000,
    sublinear_tf: bool = True,
    n_oversamples: int = 10,
    n_power_iter: int = 4,
    seed: int = 0,
    batch_size: Optional[int] = None,
    memory_guard: Optional[MemoryGuard] = None,
) -> LsaModel:
    """LSA (TF-IDF + randomized SVD) with OOC in n.

    Args:
      df: CadwDataFrame with `text_col` string column.
      text_col: str column name.
      n_components: k (target embedding dim, default 100).
      vocab_size: max TF-IDF vocab (default 10000).
      sublinear_tf: 1+log(tf) (default True; sklearn default False).
      n_oversamples: Halko oversample count (default 10).
      n_power_iter: power iterations (default 4; Halko q=4 for stability).
      seed: RNG seed for Omega (deterministic).
      batch_size: df batch rows (auto if None).
      memory_guard: optional.

    Returns LsaModel.

    Sklearn parity: TruncatedSVD(n_components=k, random_state=seed).fit
    on sklearn TfidfVectorizer with same vocab_size/sublinear_tf/norm.

    OOC (Pattern #1 batch-layer, axis=n):
      state = V*L*f64*2 + L*L*f64 + 256. Independent of n.
      bounded_by_domain {V: 1e5} (88 MB at V=1e5, L~110).

    Raises ValueError / KeyError / MemoryLimitExceeded.
    """
    if not isinstance(text_col, str) or not text_col:
        raise ValueError("fit_lsa: text_col must be non-empty str")
    if n_components < 1:
        raise ValueError(
            f"fit_lsa: n_components must be >= 1; got {n_components}")
    if vocab_size < n_components:
        raise ValueError(
            f"fit_lsa: vocab_size={vocab_size} < n_components="
            f"{n_components}")
    if text_col not in df.columns:
        raise KeyError(
            f"fit_lsa: column {text_col!r} not in {df.columns}")

    mg = memory_guard or getattr(df, "_memory_guard", None)
    n = int(df.count())
    if n < 1:
        raise ValueError("fit_lsa: empty df")

    # AP22: n_power_iter * 2 + initial + B = ~2*n_power_iter+2 passes.
    # Warn if est compute > 60s (rough: 1e6 rows/sec/iter * V_fraction).
    _est_passes = 2 * int(n_power_iter) + 2
    _est_time_s = (n / 1e6) * _est_passes * float(vocab_size) ** 0.3
    if _est_time_s > 60.0:
        import warnings
        from cadw.core.warnings import SlowOperationWarning
        warnings.warn(
            f"fit_lsa: n={n}, {_est_passes} passes over data; "
            f"estimated ~{_est_time_s:.0f}s (>{60.0:.0f}s threshold). "
            f"Options: (a) reduce n_power_iter; (b) reduce vocab_size; "
            f"(c) sample df.",
            SlowOperationWarning,
            stacklevel=2)

    # Pass 1: fit TF-IDF (streams internally)
    tfidf_model = fit_tfidf(
        df, text_col, vocab_size=vocab_size, sublinear_tf=sublinear_tf,
        memory_guard=mg, batch_size=batch_size)
    V = len(tfidf_model.vocab_)
    if V < n_components:
        raise ValueError(
            f"fit_lsa: learned vocab_size={V} < n_components="
            f"{n_components}; increase vocab_size or reduce n_components")

    # Pass 2+: sparse randomized SVD
    def _M_iter():
        return tfidf_model.transform_iter(
            df, text_col=text_col, batch_size=batch_size,
            memory_guard=mg)

    components, singular_values = _fit_randomized_svd_sparse(
        _M_iter, V, n_components, n_oversamples, n_power_iter, seed, mg)

    model = LsaModel(
        components_=components,
        singular_values_=singular_values,
        tfidf_model_=tfidf_model,
        text_col=text_col,
        n_components_=int(n_components),
        vocab_size_=int(V),
        n_docs_=int(n),
    )

    try:
        log_fit("lsa", state_bytes=model.state_nbytes())
    except Exception:
        pass

    return model
