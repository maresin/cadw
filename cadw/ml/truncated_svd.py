"""Truncated SVD (sklearn parity, algorithm='arpack' via svd_topk).

Sklearn TruncatedSVD(algorithm='arpack', tol=0) parity (P1/P3):
  singular_values_: same
  components_: V.T (after svd_flip)
  explained_variance_: np.var(X @ V, axis=0)
  explained_variance_ratio_: explained_variance_ / np.var(X, axis=0).sum()
  transform: X @ components_.T

algorithm='randomized' (sklearn default) deferred v0.9.206 (power
iteration impl; svd_topk is exact arpack-equivalent).

Pattern #1 (ROADMAP §OOC by primitive): batch-layer, axis=n_rows.
svd_topk does chunked Gram; fit+transform stream df via _iter_filtered_batches.
State O(model) = O(d·k + k). bounded_by_domain {d: 1e4} (from svd_topk).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pyarrow as pa

from cadw.core.spill import SpillArray
from cadw.ml._logging import log_fit
from cadw.ops.linalg import svd_topk


def _batch_to_X(b, X_cols):
    arrs = []
    for c in X_cols:
        col = b.column(c)
        if not pa.types.is_floating(col.type):
            col = col.cast(pa.float64())
        arrs.append(col.to_numpy(zero_copy_only=False).astype(
            np.float64, copy=False))
    return np.column_stack(arrs) if arrs else np.empty((b.num_rows, 0))


@dataclass
class TruncatedSVDResult:
    """TruncatedSVD fit result (sklearn parity, algorithm='arpack').

    components_ (k, d) = V.T
    singular_values_ (k,) descending
    explained_variance_ (k,)
    explained_variance_ratio_ (k,)

    transform(df) -> CadwDataFrame with cols svd_0..svd_{k-1}, OOC in n
    via SpillArray + stream.
    """
    components_: Optional[np.ndarray] = None
    singular_values_: Optional[np.ndarray] = None
    explained_variance_: Optional[np.ndarray] = None
    explained_variance_ratio_: Optional[np.ndarray] = None
    X_cols: list = field(default_factory=list)
    n_components: int = 0
    n_train: int = 0
    algorithm: str = "arpack"
    n_features_in_: int = 0

    def state_nbytes(self) -> int:
        if self.components_ is None:
            return 0
        sz = 256
        sz += int(self.components_.nbytes)
        sz += int(self.singular_values_.nbytes)
        sz += int(self.explained_variance_.nbytes)
        sz += int(self.explained_variance_ratio_.nbytes)
        return int(sz)

    def release_train_state(self) -> None:
        """Clear components_ etc. (idempotent). transform raises after."""
        self.components_ = None
        self.singular_values_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def transform(self, df, X_cols=None, *, batch_size=None,
                  memory_guard=None, output="auto"):
        """Project Q (n_q, d) -> (n_q, n_components) via X @ components_.T.

        Streaming: OOC in n_q via SpillArray. Returns CadwDataFrame.
        """
        if self.components_ is None:
            raise RuntimeError(
                "TruncatedSVDResult.transform: model released")
        X_cols = list(X_cols) if X_cols is not None else list(self.X_cols)
        if set(X_cols) != set(self.X_cols):
            raise ValueError(
                f"TruncatedSVDResult.transform: X_cols mismatch "
                f"(fit: {self.X_cols}, got: {X_cols})")
        X_cols = list(self.X_cols)  # reorder to fit order

        k = self.n_components
        n_q = int(df.count())
        if n_q == 0:
            raise ValueError("TruncatedSVDResult.transform: df is empty")

        mg = memory_guard or getattr(df, "_memory_guard", None)
        d = self.components_.shape[1]
        if batch_size is None:
            try:
                avail = mg.available_bytes() if mg is not None else (8 << 20)
            except Exception:
                avail = (8 << 20)
            target = min(8 << 20, avail // 20)
            bsz = max(1, target // max(1, (d + k + 4) * 16))
            bsz = min(int(bsz), n_q)
        else:
            bsz = int(batch_size)
        bsz = max(1, bsz)

        spill_ratio = {"memory": 1.0, "auto": 0.5, "disk": 0.0}.get(
            output, 0.5)
        buf = SpillArray(
            guard=mg, width=k, dtype=np.float64,
            name="truncated_svd.transform", spill_ratio=spill_ratio)

        Xt = self.components_  # (k, d)
        for b in df._iter_filtered_batches(batch_size=bsz):
            if mg is not None and hasattr(mg, "check_batch"):
                mg.check_batch(b, name="truncated_svd.Q", extra_factor=3.0)
            X_b = _batch_to_X(b, X_cols)
            if X_b.shape[0] == 0:
                continue
            Z_b = X_b @ Xt.T  # (b, k)
            buf.write_array(Z_b)
        result = buf.finalize()

        # Wrap as CadwDataFrame via Dataset.from_batches from result iterator
        from cadw.dataset.dataset import Dataset
        from cadw import CadwDataFrame
        schema = pa.schema([pa.field(f"svd_{i}", pa.float64())
                            for i in range(k)])

        # If result is already in-memory ndarray, wrap directly.
        # If spilled (AutoArrayResult iterable), re-stream via Dataset.
        if hasattr(result, "to_numpy"):
            try:
                Z = result.to_numpy()
                cols = {f"svd_{i}": pa.array(Z[:, i], type=pa.float64())
                        for i in range(k)}
                return CadwDataFrame.from_table(
                    pa.table(cols), memory_guard=mg,
                    spill_dir=getattr(df, "_spill_dir", None))
            except Exception:
                pass  # spilled — fall through

        def _iter_z():
            for chunk in result:
                arr = np.asarray(chunk)
                if arr.ndim == 1:
                    arr = arr[:, None]
                cols = [pa.array(arr[:, i], type=pa.float64())
                        for i in range(k)]
                yield pa.RecordBatch.from_arrays(cols, schema=schema)

        ds = Dataset.from_batches(schema, _iter_z(), df._new_dataset_dir())
        return CadwDataFrame(ds, memory_guard=mg,
                             spill_dir=getattr(df, "_spill_dir", None))

    def fit_transform(self, df, X_cols=None, *, batch_size=None,
                      memory_guard=None, output="auto"):
        """Sklearn parity: fit_transform = transform (components_ fixed)."""
        return self.transform(df, X_cols=X_cols,
                                batch_size=batch_size,
                                memory_guard=memory_guard,
                                output=output)


def fit_truncated_svd(
    df,
    X_cols,
    n_components: int = 2,
    *,
    algorithm: str = "arpack",
    random_state: Optional[int] = None,
    batch_size: Optional[int] = None,
    memory_guard=None,
) -> TruncatedSVDResult:
    """Fit TruncatedSVD (sklearn parity, algorithm='arpack' only).

    Args:
      df: CadwDataFrame with X_cols numeric (no NaN).
      X_cols: list[str].
      n_components: k (2 <= k <= min(n, d); sklearn default 2).
      algorithm: "arpack" only (CADW default; exact, bit-identical to
        sklearn algorithm='arpack', tol=0). "randomized" deferred v0.9.206.
      random_state: unused for arpack (eigh_topk is exact); kept for API
        compat.
      batch_size, memory_guard: standard.

    Returns TruncatedSVDResult.

    Sklearn TruncatedSVD(algorithm='arpack', tol=0) parity:
      singular_values_: atol 1e-9
      components_: atol 1e-9
      explained_variance_, explained_variance_ratio_: atol 1e-9

    Pattern #1 (batch-layer, axis=n_rows): svd_topk chunked Gram +
    two streaming stats passes for explained variance; state O(model).
    """
    if algorithm != "arpack":
        raise ValueError(
            f"fit_truncated_svd: algorithm must be 'arpack' "
            f"(CADW: exact parity); 'randomized' deferred v0.9.206; "
            f"got {algorithm!r}")
    if not X_cols:
        raise ValueError("fit_truncated_svd: X_cols is empty")
    if n_components < 1:
        raise ValueError(
            f"fit_truncated_svd: n_components must be >= 1; "
            f"got {n_components}")
    missing = [c for c in X_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"fit_truncated_svd: columns missing: {missing}; "
            f"available: {df.columns}")

    mg = memory_guard or getattr(df, "_memory_guard", None)
    n_total = int(df.count())
    d = len(X_cols)
    k = int(n_components)
    if n_total < 1 or k > min(n_total, d):
        raise ValueError(
            f"fit_truncated_svd: need 1 <= k <= min(n={n_total}, d={d}); "
            f"got k={k}")
    if batch_size is None:
        batch_size = max(1, min(n_total, 1000))
    batch_size = max(1, min(int(batch_size), n_total))

    # Pass 1: svd_topk (chunked Gram + eigh + svd_flip) -> V, sigma
    _U, sigma, V = svd_topk(
        df, X_cols, k, compute_u=False,
        batch_size=batch_size, memory_guard=mg)
    components_ = V.T.copy()  # (k, d)

    # Pass 2: explained_variance_ = var(X @ V, axis=0)  [chunked]
    # Pass 3: full_var = np.var(X, axis=0).sum()        [chunked]
    # Combine into 1 stream: accumulate Z stats + X stats per batch.
    z_s1 = np.zeros(k, dtype=np.float64)
    z_s2 = np.zeros(k, dtype=np.float64)
    x_s1 = np.zeros(d, dtype=np.float64)
    x_s2 = np.zeros(d, dtype=np.float64)
    n_processed = 0
    for b in df._iter_filtered_batches(batch_size=batch_size):
        if mg is not None and hasattr(mg, "check_batch"):
            mg.check_batch(b, name="truncated_svd.fit", extra_factor=3.0)
        X_b = _batch_to_X(b, X_cols)
        if X_b.shape[0] == 0:
            continue
        Z_b = X_b @ V  # (b, k); V (d, k)
        z_s1 += Z_b.sum(axis=0)
        z_s2 += (Z_b * Z_b).sum(axis=0)
        x_s1 += X_b.sum(axis=0)
        x_s2 += (X_b * X_b).sum(axis=0)
        n_processed += X_b.shape[0]
    if n_processed != n_total:
        raise RuntimeError(
            f"fit_truncated_svd: X rows mismatch "
            f"({n_processed} != {n_total})")

    n = float(n_total)
    mean_z = z_s1 / n
    var_z = np.maximum(z_s2 / n - mean_z * mean_z, 0.0)
    mean_x = x_s1 / n
    var_x = np.maximum(x_s2 / n - mean_x * mean_x, 0.0)
    full_var = float(var_x.sum())
    if full_var <= 0.0:
        exp_var_ratio = np.zeros(k, dtype=np.float64)
    else:
        exp_var_ratio = var_z / full_var

    result = TruncatedSVDResult(
        components_=components_,
        singular_values_=sigma.astype(np.float64),
        explained_variance_=var_z.astype(np.float64),
        explained_variance_ratio_=exp_var_ratio.astype(np.float64),
        X_cols=list(X_cols),
        n_components=k,
        n_train=int(n_total),
        algorithm="arpack",
        n_features_in_=int(d),
    )

    try:
        log_fit("truncated_svd", state_bytes=result.state_nbytes())
    except Exception:
        pass

    return result
