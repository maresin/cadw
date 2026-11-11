"""Spectral co-clustering (Dhillon 2001).

Bipartite spectral clustering on a checkerboard matrix A (r x c). Rows
and columns are clustered simultaneously on the spectral embedding.

Reference:
  Dhillon 2001, "Co-clustering documents and words using bipartite
  spectral graph partitioning", KDD 2001.

Sklearn parity: SpectralCoclustering(svd_method='randomized', n_init=10,
init='k-means++'). Note: sklearn svd_method='arpack' has a known issue
(svds returns unsorted sigmas; n_discard=1 drops arbitrary) — CADW uses
exact eigh (always sorted descending) and is bit-correct, potentially
more accurate than sklearn's randomized path on k>=3.

OOC pattern (ROADMAP §OOC by primitive #1 batch-layer):
  - axis = r (rows unbounded): chunked Reduce passes
  - gram_dim = c (bounded): state c^2 * f64 (eigh Gram)
  - honest bound: bounded_by_window {gram_dim: 1e4} = 800 MB Gram
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.core.spill import SpillArray, AutoArrayResult
from cadw.dataset.dataset import Dataset
from cadw.ml._logging import log_fit
from cadw.ops.linalg import eigh_topk, sq_dists


def _batch_to_X(b, X_cols):
    """Extract (b, c) f64 ndarray from pa.RecordBatch."""
    arrs = []
    for c in X_cols:
        col = b.column(c)
        if not pa.types.is_floating(col.type):
            col = col.cast(pa.float64())
        arrs.append(col.to_numpy(zero_copy_only=False).astype(
            np.float64, copy=False))
    return np.column_stack(arrs) if arrs else np.empty((b.num_rows, 0))


def _svd_flip_columns(V):
    """Sklearn svd_flip(u_based_decision=False) column-wise in-place.

    For each column, find argmax(|col|); if negative, flip column sign.
    """
    V = V.copy()
    for j in range(V.shape[1]):
        col = V[:, j]
        idx = int(np.argmax(np.abs(col)))
        if col[idx] < 0:
            V[:, j] *= -1
    return V


@dataclass
class SpectralCoclusteringResult:
    """SpectralCoclustering fit result (Dhillon 2001).

    row_labels_: (r,) int64 — AutoArrayResult (spillable) or ndarray
    column_labels_: (c,) int64 — ndarray (c bounded, RAM)
    row_diag_: (r,) f64 — NOT stored (recomputable)
    col_diag_: (c,) f64 — RAM
    sigma_: (k_used,) f64 — sqrt eigenvalues (after n_discard)
    n_clusters: int
    r, c: int (input dims)
    k_sv: int (svd order = 1 + ceil(log2(k)))
    """
    row_labels_: object = None       # AutoArrayResult | np.ndarray
    column_labels_: Optional[np.ndarray] = None
    col_diag_: Optional[np.ndarray] = None
    sigma_: Optional[np.ndarray] = None
    n_clusters: int = 0
    r: int = 0
    c: int = 0
    k_sv: int = 0

    def state_nbytes(self) -> int:
        """Persistent state: col_labels (c) + col_diag (c) + sigma (k_used)
        + row_labels handle (SpillArray disk O(r) not counted — RAM O(chunk))
        + header 256.
        """
        sz = 256
        if self.column_labels_ is not None:
            sz += int(self.column_labels_.nbytes)
        if self.col_diag_ is not None:
            sz += int(self.col_diag_.nbytes)
        if self.sigma_ is not None:
            sz += int(self.sigma_.nbytes)
        return int(sz)

    def to_dict(self) -> dict:
        return {
            "n_clusters": int(self.n_clusters),
            "r": int(self.r),
            "c": int(self.c),
            "k_sv": int(self.k_sv),
            "sigma": self.sigma_.tolist() if self.sigma_ is not None else [],
            "col_diag_shape": (self.col_diag_.shape
                               if self.col_diag_ is not None else None),
        }


def fit_spectral_coclustering(
    df,
    X_cols,
    n_clusters: int = 3,
    *,
    n_discard: int = 1,
    chunk_size: Optional[int] = None,
    memory_guard: Optional[MemoryGuard] = None,
    random_state: int = 0,
) -> SpectralCoclusteringResult:
    """Spectral co-clustering on bipartite matrix df[X_cols].

    Args:
      df: CadwDataFrame, rows = one side of bipartite (users, docs),
          X_cols = other side (items, terms). Values must be non-negative
          (sklearn `make_nonnegative`; negative -> 0).
      X_cols: list of numeric column names; c = len(X_cols).
      n_clusters: int >= 1; k.
      n_discard: default 1 (sklearn); drop n_discard largest sigmas.
      chunk_size: rows per batch (auto from guard if None).
      memory_guard: optional.
      random_state: for kmeans.

    Returns SpectralCoclusteringResult.

    OOC (pattern #1 batch-layer):
      - state = c^2*f64 (Gram) + c*k_sv*f64 + chunk*(c+k_sv)*f64*2 + 256
      - independent of r (rows unbounded)
      - bounded_by_window {axis: gram_dim = min(r,c), bound: 1e4}
        (c^2*8 = 800 MB at c=1e4 — realistic vocab/items range)

    Raises ValueError / KeyError / MemoryLimitExceeded.
    """
    if not X_cols:
        raise ValueError("fit_spectral_coclustering: X_cols is empty")
    X_cols = list(X_cols)
    if not isinstance(n_clusters, int):
        raise TypeError(
            f"n_clusters must be int; got {type(n_clusters).__name__}")
    if n_clusters < 1:
        raise ValueError(f"n_clusters must be >= 1; got {n_clusters}")
    if n_discard < 0:
        raise ValueError(f"n_discard must be >= 0; got {n_discard}")
    missing = [c for c in X_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"X_cols missing: {missing}; available: {df.columns}")

    mg = memory_guard or getattr(df, "_memory_guard", None)
    r = int(df.count())
    c = len(X_cols)
    if r < 1 or c < 1:
        raise ValueError(f"need r>=1 and c>=1; got r={r}, c={c}")

    k_sv = 1 + int(np.ceil(np.log2(max(1, n_clusters))))
    k_used = k_sv - n_discard
    if k_used < 1:
        raise ValueError(
            f"n_discard={n_discard} >= k_sv={k_sv}; no singular vectors "
            f"left. Reduce n_discard or increase n_clusters.")
    if k_used > min(r, c):
        raise ValueError(
            f"k_used={k_used} > min(r={r}, c={c}); "
            f"reduce n_clusters or n_discard")

    if chunk_size is None:
        try:
            avail = (mg.available_bytes() if mg is not None else 8 << 20)
        except Exception:
            avail = 8 << 20
        target = min(8 << 20, avail // 20)
        bsz = max(1, target // max(1, (c + k_sv + 4) * 16))
        bsz = min(int(bsz), r)
    else:
        bsz = max(1, int(chunk_size))
    bsz = max(1, min(bsz, r))

    # state guard: Gram + V + transients
    state_bytes = (c * c * 8             # G (c,c)
                   + c * k_sv * 8        # V (c,k_sv)
                   + k_sv * 8            # eigvals
                   + bsz * (c + k_sv) * 8 * 2  # per-chunk transients
                   + 256)
    if mg is not None and hasattr(mg, "check_state"):
        mg.check_state(state_bytes, name="coclust.state")

    # ============================================
    # Pass 1: col_sum (c,)
    # ============================================
    col_sum = np.zeros(c, dtype=np.float64)
    for b in df._iter_filtered_batches(batch_size=bsz):
        if mg is not None and hasattr(mg, "check_batch"):
            mg.check_batch(b, name="coclust.col_sum", extra_factor=2.0)
        X_b = _batch_to_X(b, X_cols)
        if X_b.shape[0] == 0:
            continue
        # non-negative (sklearn make_nonnegative)
        X_b = np.maximum(X_b, 0.0)
        col_sum += X_b.sum(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        col_diag = 1.0 / np.sqrt(col_sum)
    col_diag = np.where(np.isnan(col_diag), 0.0, col_diag)
    # also handle inf (col_sum=0): 1/sqrt(0) = inf; make_nonnegative gives 0
    col_diag = np.where(np.isinf(col_diag), 0.0, col_diag)

    # ============================================
    # Pass 2: G = A_norm^T A_norm (c, c)
    # (row_diag recompute per chunk — no O(r) state)
    # ============================================
    G = np.zeros((c, c), dtype=np.float64)
    for b in df._iter_filtered_batches(batch_size=bsz):
        if mg is not None and hasattr(mg, "check_batch"):
            mg.check_batch(b, name="coclust.gram", extra_factor=3.0)
        X_b = _batch_to_X(b, X_cols)
        if X_b.shape[0] == 0:
            continue
        X_b = np.maximum(X_b, 0.0)
        rs_b = X_b.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            rd_b = 1.0 / np.sqrt(rs_b)
        rd_b = np.where(np.isnan(rd_b), 0.0, rd_b)
        rd_b = np.where(np.isinf(rd_b), 0.0, rd_b)
        A_norm_b = X_b * rd_b[:, None] * col_diag[None, :]
        G += A_norm_b.T @ A_norm_b

    # ============================================
    # eigh_topk on G → V, sigma (descending)
    # ============================================
    eigvals, V = eigh_topk(G, k_sv, which="LA", memory_guard=mg)
    eigvals = np.maximum(eigvals, 0.0)
    sigma = np.sqrt(eigvals)

    # n_discard=1: drop first singular vector (sklearn convention)
    V_red = V[:, n_discard:]
    sigma_red = sigma[n_discard:]

    # svd_flip V (sklearn): per-column argmax sign
    V_red = _svd_flip_columns(V_red)

    # ============================================
    # Pass 3: Z_r = row_diag * U (SpillArray, r x k_used)
    # U = A_norm @ V_red / sigma_red
    # ============================================
    sigma_safe = np.maximum(sigma_red, 1e-12)
    zr_sp = SpillArray(
        guard=mg, width=k_used, dtype=np.float64,
        name="coclust.Zr", spill_ratio=0.5)

    for b in df._iter_filtered_batches(batch_size=bsz):
        if mg is not None and hasattr(mg, "check_batch"):
            mg.check_batch(b, name="coclust.U", extra_factor=3.0)
        X_b = _batch_to_X(b, X_cols)
        if X_b.shape[0] == 0:
            continue
        X_b = np.maximum(X_b, 0.0)
        rs_b = X_b.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            rd_b = 1.0 / np.sqrt(rs_b)
        rd_b = np.where(np.isnan(rd_b), 0.0, rd_b)
        rd_b = np.where(np.isinf(rd_b), 0.0, rd_b)
        A_norm_b = X_b * rd_b[:, None] * col_diag[None, :]
        U_b = (A_norm_b @ V_red) / sigma_safe[None, :]
        # Z_r row: row_diag * U (sklearn formula)
        Z_r_b = rd_b[:, None] * U_b
        zr_sp.write_array(Z_r_b)
    Z_r = zr_sp.finalize()

    # ============================================
    # Z_c = col_diag * V_red (c, k_used) — RAM (c bounded)
    # ============================================
    Z_c = col_diag[:, None] * V_red

    # ============================================
    # Build Z df (r+c, k_used): stream Z_r + one Z_c batch
    # ============================================
    z_schema = pa.schema(
        [pa.field(f"z{j}", pa.float64()) for j in range(k_used)])

    def _iter_z_batches():
        for chunk_arr in Z_r:  # re-iterable AutoArrayResult
            arr = np.asarray(chunk_arr)
            if arr.ndim == 1:
                arr = arr[:, None]
            yield pa.RecordBatch.from_arrays(
                [pa.array(arr[:, j], type=pa.float64())
                 for j in range(k_used)],
                schema=z_schema)
        yield pa.RecordBatch.from_arrays(
            [pa.array(Z_c[:, j], type=pa.float64())
             for j in range(k_used)],
            schema=z_schema)

    z_ds = Dataset.from_batches(
        z_schema, _iter_z_batches(), df._new_dataset_dir())
    z_df = type(df)(z_ds, memory_guard=mg,
                    spill_dir=getattr(df, "_spill_dir", None))

    # ============================================
    # fit_kmeans on Z (verified, OOC in (r+c))
    # ============================================
    from cadw.ml import fit_kmeans
    km = fit_kmeans(
        z_df, [f"z{j}" for j in range(k_used)],
        k=n_clusters, n_init=10, seed=random_state,
        memory_guard=mg, batch_size=bsz)
    centers = km.cluster_centers_  # (n_clusters, k_used)

    # ============================================
    # row_labels (r,): chunked sq_dists + argmin → SpillArray int64
    # ============================================
    row_lbl_sp = SpillArray(
        guard=mg, width=1, dtype=np.int64,
        name="coclust.row_labels", spill_ratio=0.5)
    for chunk_arr in Z_r:  # re-iterable
        arr = np.asarray(chunk_arr)
        if arr.ndim == 1:
            arr = arr[:, None]
        # (chunk, n_clusters) sq distances
        diff = arr[:, None, :] - centers[None, :, :]
        d2 = (diff * diff).sum(axis=2)
        row_lbl_sp.write_array(
            d2.argmin(axis=1).astype(np.int64, copy=False))
    row_labels = row_lbl_sp.finalize()

    # ============================================
    # col_labels (c,): RAM (c bounded by gram_dim)
    # ============================================
    diff = Z_c[:, None, :] - centers[None, :, :]
    d2 = (diff * diff).sum(axis=2)
    col_labels = d2.argmin(axis=1).astype(np.int64, copy=False)

    result = SpectralCoclusteringResult(
        row_labels_=row_labels,
        column_labels_=col_labels,
        col_diag_=col_diag,
        sigma_=sigma_red.astype(np.float64),
        n_clusters=int(n_clusters),
        r=int(r),
        c=int(c),
        k_sv=int(k_sv),
    )

    try:
        log_fit("spectral_coclustering", state_bytes=result.state_nbytes())
    except Exception:
        pass

    return result
