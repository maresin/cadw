"""KNNImputer (sklearn parity, OOC in n).

Fit: X (n, d) -> memmap (keep NaN); detect all-NaN columns (dropped, as
sklearn). Transform: per-column kNN among donors via nan_sq_dists
(partial-distance), weighted avg donors' col-j values.

sklearn.metrics.pairwise.nan_euclidean_distances(squared=True) parity
(P8b max diff 0.0). All-NaN column drop (P8a). Distance weights
inf_row → 1.0/0.0 (P8a _get_weights).
"""
from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa

from cadw.core.spill import SpillDir
from cadw.ml._logging import log_fit
from cadw.ops.linalg import nan_sq_dists


@dataclass
class KNNImputerResult:
    """KNNImputer fit result (sklearn parity, OOC in n).

    Fit stores X memmap at `train_spill_path` + per-column donor masks.
    Transform: O(steps), streaming via Dataset.from_batches; per-column
    kNN on donors (nan_euclidean).

    State: O(model) = 256 + fit_nan_counts + X_cols_kept sum + path.
    """
    train_spill_path: Optional[Path]
    X_cols_kept: list
    X_cols_dropped: list
    fit_nan_counts: np.ndarray   # (n_cols_kept,) int64
    k: int
    weights: str
    n_train: int
    is_spilled: bool = True  # results always disk-backed at fit

    def state_nbytes(self) -> int:
        """O(model): path + col metadata; X memmap on disk (not counted)."""
        if self.train_spill_path is None:
            return 0
        sz = 256
        sz += int(self.fit_nan_counts.nbytes)
        for c in self.X_cols_kept:
            sz += 8 + len(c)
        for c in self.X_cols_dropped:
            sz += 8 + len(c)
        return int(sz)

    def release_train_state(self) -> None:
        """Delete training memmap + donor masks from disk. Idempotent."""
        if self.train_spill_path is None:
            return
        d = Path(self.train_spill_path)
        if d.exists():
            try:
                shutil.rmtree(d)
            except Exception:
                pass
        self.train_spill_path = None

    def transform(self, df, X_cols=None, *, batch_size=None,
                  memory_guard=None):
        """Return CadwDataFrame with NaN in X_cols_kept imputed (OOC).

        df must have X_cols_kept (order-insensitive). Output schema =
        X_cols_kept (sklearn behavior: all-NaN cols are dropped).
        """
        if self.train_spill_path is None:
            raise RuntimeError(
                "KNNImputerResult.transform: train_spill_path missing "
                "(call fit_knn_imputer first; do not release until "
                "transform done).")
        d_kept = len(self.X_cols_kept)
        n_train = int(self.n_train)
        k = int(self.k)
        weights = self.weights

        X_cols = list(X_cols) if X_cols is not None else list(self.X_cols_kept)
        if set(X_cols) != set(self.X_cols_kept):
            raise ValueError(
                f"KNNImputerResult.transform: X_cols mismatch. "
                f"fit kept {self.X_cols_kept}; got {X_cols}. "
                f"Dropped at fit: {self.X_cols_dropped}.")

        fit_dir = Path(self.train_spill_path)
        if not fit_dir.exists():
            raise RuntimeError(
                f"KNNImputerResult.transform: train_spill_path "
                f"{fit_dir!r} missing")
        X_mm = np.memmap(str(fit_dir / "X.bin"),
                         dtype=np.float64, mode="r",
                         shape=(n_train, d_kept))

        mg = memory_guard or getattr(df, "_memory_guard", None)
        n_q = int(df.count())
        if n_q == 0:
            raise ValueError("KNNImputerResult.transform: df is empty")
        if batch_size is None:
            # Conservative auto: O(batch * (d + k) * 16) per column batch
            try:
                avail = mg.available_bytes() if mg is not None else 8 << 20
            except Exception:
                avail = 8 << 20
            target = min(8 << 20, avail // 20)
            bsz = max(1, target // max(1, (d_kept + k + 4) * 16))
            bsz = min(int(bsz), n_q)
        else:
            bsz = int(batch_size)
        bsz = max(1, bsz)

        # Preload per-column donor masks (fit_nan_counts computable)
        # Actually we compute donors from X_mm on demand per column chunk.

        # Build streaming output via Dataset.from_batches.
        from cadw.dataset.dataset import Dataset
        from cadw import CadwDataFrame

        schema = pa.schema([pa.field(c, pa.float64())
                            for c in self.X_cols_kept])

        def _iter_imputed_batches():
            for qb in df._iter_filtered_batches(batch_size=bsz):
                if mg is not None and hasattr(mg, "check_batch"):
                    mg.check_batch(qb, name="knn_imputer.Q",
                                   extra_factor=2.0)
                Q = _batch_to_X(qb, X_cols)
                # Q order must map to self.X_cols_kept ordering:
                perm = [X_cols.index(c) for c in self.X_cols_kept]
                Q = Q[:, perm] if perm else Q
                Q_out = Q.copy()  # imputed copy
                for j_kept in range(d_kept):
                    _impute_col_in_Q_batch(
                        Q_out, Q, j_kept, X_mm, k, weights, mg)
                # Yield pa.RecordBatch
                cols = [pa.array(Q_out[:, j], type=pa.float64())
                        for j in range(d_kept)]
                yield pa.RecordBatch.from_arrays(
                    cols, schema=schema)

        ds = Dataset.from_batches(schema, _iter_imputed_batches(),
                                  df._new_dataset_dir())
        return CadwDataFrame(ds, memory_guard=mg,
                             spill_dir=getattr(df, "_spill_dir", None))


# ====================================================================
# Helpers
# ====================================================================

def _batch_to_X(b, X_cols):
    """Extract (b, d) float64 from pa.RecordBatch; NaN preserved."""
    arrs = []
    for c in X_cols:
        col = b.column(c)
        if not pa.types.is_floating(col.type):
            col = col.cast(pa.float64())
        arrs.append(col.to_numpy(zero_copy_only=False).astype(
            np.float64, copy=False))
    return np.column_stack(arrs) if arrs else np.empty((b.num_rows, 0))


def _get_weights(dist: np.ndarray, weights: str) -> Optional[np.ndarray]:
    """sklearn.impute._knn._get_weights semantics (P8a).

    uniform/None -> None (caller uses ones w/ NaN mask).
    distance: 1/dist with inf-row 0/1 hard-lock (duplicates -> exact 1.0
    on 0-distance points, 0.0 on others); NaN dist -> weight 0.
    """
    if weights in (None, "uniform"):
        return None
    if weights == "distance":
        with np.errstate(divide="ignore"):
            w = 1.0 / dist
        inf_mask = np.isinf(w)
        inf_row = np.any(inf_mask, axis=1)
        # sklearn semantics: 0-distance points -> weight 1.0, others 0.0
        w[inf_row] = inf_mask[inf_row].astype(np.float64)
        # NaN dist -> weight 0
        w[np.isnan(dist)] = 0.0
        return w
    raise ValueError(
        f"_get_weights: weights must be 'uniform'|'distance'; got {weights!r}")


def _impute_col_in_Q_batch(Q_out, Q_orig, j, X_mm, k, weights, mg):
    """Impute column j for all rows of Q_out (in-place) where Q_orig[:,j] NaN.

    Q_out: (n_q, d) mutable (pre-copied). Q_orig: (n_q, d) original.
    X_mm: (n_train, d) read-only memmap (NaN preserved).
    """
    n_train = X_mm.shape[0]
    q_j_nan = np.isnan(Q_orig[:, j])
    if not q_j_nan.any():
        return  # nothing to impute in this column for this Q batch

    # Donors: rows in X_mm where col j not NaN
    # Streaming scan of X_mm to get donor mask (avoid full materialize when
    # n_train large; chunked for OOC).
    x_chunk = max(1, min(n_train, 4096))
    donor_indices_chunks = []
    donor_col_j_chunks = []
    for s in range(0, n_train, x_chunk):
        e = min(s + x_chunk, n_train)
        x_block = np.asarray(X_mm[s:e, :])
        if mg is not None and hasattr(mg, "check_state"):
            mg.check_state(x_block.nbytes, name="knn_imputer.X_chunk")
        col_j_block = x_block[:, j]
        valid = ~np.isnan(col_j_block)
        if valid.any():
            donor_indices_chunks.append(
                np.where(valid)[0] + s)  # global idx
            donor_col_j_chunks.append(col_j_block[valid])
    if not donor_indices_chunks:
        # No donors in this column (should have been dropped at fit —
        # defensive). sklearn raises in fit, but we drop, so this
        # unreachable if fit dropped. Fallback: mean of training col?
        # sklearn behavior would have dropped at fit; if all-NaN appeared
        # in training there is no donor → return without filling.
        return

    donor_idx = np.concatenate(donor_indices_chunks)  # (n_donors,)
    donor_col_j = np.concatenate(donor_col_j_chunks)   # (n_donors,)
    n_donors = len(donor_idx)

    # Receivers: Q rows where col j NaN
    receiver_idx = np.where(q_j_nan)[0]
    Q_rec = Q_orig[receiver_idx]  # (n_recv, d) with col j still NaN

    # Compute nan_sq_dists in x-chunks, top-k merge
    k_eff = min(k, n_donors)
    top_d = np.full((len(receiver_idx), k_eff), np.inf, dtype=np.float64)
    top_global_idx = np.full((len(receiver_idx), k_eff), -1, dtype=np.int64)

    # Build global donor index iter (chunks of donor rows from X_mm)
    # — but we already collected donor_idx; iterate in x_chunk slices
    #   of the donor array to keep peak bounded.
    dchunk = max(1, min(n_donors, x_chunk))
    for s in range(0, n_donors, dchunk):
        e = min(s + dchunk, n_donors)
        idx_block = donor_idx[s:e]
        X_donors_block = np.asarray(X_mm[idx_block, :])  # (b, d)
        if mg is not None and hasattr(mg, "check_state"):
            mg.check_state(X_donors_block.nbytes,
                           name="knn_imputer.donor_chunk")
        sq = nan_sq_dists(Q_rec, X_donors_block)  # (n_recv, b), NaN where no common
        # Merge with current top-k (NaN -> treat as +inf)
        sq_inf = np.where(np.isnan(sq), np.inf, sq)
        chunk_global = np.broadcast_to(idx_block, sq_inf.shape)
        merged_d = np.concatenate([top_d, sq_inf], axis=1)
        merged_g = np.concatenate([top_global_idx, chunk_global], axis=1)
        part = np.argpartition(merged_d, k_eff - 1, axis=1)[:, :k_eff]
        part_d = np.take_along_axis(merged_d, part, axis=1)
        part_g = np.take_along_axis(merged_g, part, axis=1)
        order = np.argsort(part_d, axis=1, kind="stable")
        top_d = np.take_along_axis(part_d, order, axis=1)
        top_global_idx = np.take_along_axis(part_g, order, axis=1)

    # Reconstruct distances (sqrt of sq dists -> sklearn uses euclidean dist
    # for weights; nan_euclidean_distances(squared=False))
    distances = np.sqrt(np.maximum(top_d, 0.0))
    # -inf dist mask (no common dims in any donor) -> weight 0
    distances = np.where(np.isinf(top_d), np.nan, distances)

    weights_arr = _get_weights(distances, weights)
    if weights_arr is None:
        weights_arr = np.ones_like(distances)
        weights_arr[np.isnan(distances)] = 0.0

    # Donors' col-j values. v0.9.199-b-fix (OOC): use searchsorted
    # into donor_col_j (O(n_donors)) instead of full (n_train,) lookup.
    # donor_idx is sorted (np.where across chunked X_mm scan).
    valid_mask = (top_global_idx >= 0) & np.isfinite(top_d)
    safe_global = np.where(valid_mask, top_global_idx, donor_idx[0])
    # searchsorted: pos in donor_idx[] of each global idx
    pos_in_donors = np.searchsorted(donor_idx, safe_global)
    donors_mat = donor_col_j[pos_in_donors]  # (n_recv, k_eff)
    donors_mat = np.where(valid_mask, donors_mat, np.nan)

    # Weighted average (skip NaN donors)
    num = np.nansum(weights_arr * donors_mat, axis=1)
    den = np.nansum(weights_arr * ~np.isnan(donors_mat), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        imputed = np.where(den > 0, num / den, np.nan)
    Q_out[receiver_idx, j] = imputed


# ====================================================================
# Public fit
# ====================================================================

def fit_knn_imputer(
    df,
    X_cols,
    *,
    k: int = 5,
    weights: str = "uniform",
    spill_path=None,
    batch_size: Optional[int] = None,
    memory_guard=None,
) -> KNNImputerResult:
    """Fit KNNImputer (sklearn parity, OOC in n).

    Args:
      df: CadwDataFrame with X_cols numeric (may contain NaN/None).
      X_cols: list[str].
      k: number of donors (uniform: mean of k; distance: 1/d weighted).
      weights: "uniform" (mean) | "distance" (1/d; 0-dist -> 1.0 hard-lock,
        other weights 0, sklearn parity).
      spill_path: explicit dir (default: <df._spill_dir>/knn_imputer_<uuid>).
        Note: fit always saves training state (imputer without transform
        has no value; caller must release_train_state() when done).
      batch_size, memory_guard: standard.

    Returns KNNImputerResult. All-NaN training columns are DROPPED
    (sklearn behavior); use result.X_cols_dropped to inspect.
    """
    if not X_cols:
        raise ValueError("fit_knn_imputer: X_cols is empty")
    if k < 1:
        raise ValueError(f"fit_knn_imputer: k must be >= 1; got {k}")
    if weights not in ("uniform", "distance"):
        raise ValueError(
            f"fit_knn_imputer: weights must be 'uniform'|'distance'; "
            f"got {weights!r}")
    missing = [c for c in X_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"fit_knn_imputer: columns missing: {missing}; "
            f"available: {df.columns}")

    mg = memory_guard or getattr(df, "_memory_guard", None)
    n_total = int(df.count())
    if n_total < 2:
        raise ValueError(
            f"fit_knn_imputer: need n >= 2; got n={n_total}")
    d_full = len(X_cols)
    if batch_size is None:
        batch_size = max(1, min(n_total, 1000))
    batch_size = max(1, min(int(batch_size), n_total))

    # Spill dir
    if spill_path is not None:
        fit_dir = Path(spill_path)
    else:
        fit_dir = (Path(df._spill_dir)
                   / f"knn_imputer_{uuid.uuid4().hex[:8]}")
    fit_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: X -> memmap (keep NaN)
    X_path = fit_dir / "X.bin"
    X_mm = np.memmap(str(X_path), dtype=np.float64, mode="w+",
                     shape=(n_total, d_full))
    row = 0
    for b in df._iter_filtered_batches(batch_size=batch_size):
        if mg is not None and hasattr(mg, "check_batch"):
            mg.check_batch(b, name="knn_imputer.X", extra_factor=2.0)
        n_b = b.num_rows
        X_mm[row:row + n_b] = _batch_to_X(b, X_cols)
        row += n_b
    if row != n_total:
        raise RuntimeError(
            f"fit_knn_imputer: X rows mismatch ({row} != {n_total})")
    X_mm.flush()

    # Detect all-NaN columns (fit_nan_counts)
    nan_counts = np.zeros(d_full, dtype=np.int64)
    for s in range(0, n_total, batch_size):
        e = min(s + batch_size, n_total)
        block = np.asarray(X_mm[s:e, :])
        nan_counts += np.isnan(block).sum(axis=0)
    dropped_idx = [j for j in range(d_full) if nan_counts[j] == n_total]
    kept_idx = [j for j in range(d_full) if nan_counts[j] < n_total]

    X_cols_kept = [X_cols[j] for j in kept_idx]
    X_cols_dropped = [X_cols[j] for j in dropped_idx]

    if not X_cols_kept:
        X_mm.flush()
        del X_mm
        shutil.rmtree(fit_dir, ignore_errors=True)
        raise ValueError(
            f"fit_knn_imputer: all X_cols are all-NaN; nothing to impute "
            f"(all dropped).")

    # Rewrite X_mm as (n_train, d_kept) — subset kept columns; drop rest.
    # For simplicity: keep original (n, d_full) file + save kept_idx in meta.
    # Actually transform needs kept-columns only; simpler to re-write:
    if len(kept_idx) < d_full:
        X_kept_path = fit_dir / "X_kept.bin"
        X_kept_mm = np.memmap(str(X_kept_path), dtype=np.float64,
                              mode="w+",
                              shape=(n_total, len(kept_idx)))
        for s in range(0, n_total, batch_size):
            e = min(s + batch_size, n_total)
            X_kept_mm[s:e] = np.asarray(X_mm[s:e, :])[:, kept_idx]
        X_kept_mm.flush()
        del X_kept_mm
        # Replace X.bin with X_kept.bin content
        try:
            X_path.unlink()
        except FileNotFoundError:
            pass
        X_kept_path.rename(X_path)
    X_mm.flush()
    del X_mm

    d_kept = len(kept_idx)
    nan_counts_kept = nan_counts[kept_idx]

    result = KNNImputerResult(
        train_spill_path=fit_dir,
        X_cols_kept=X_cols_kept,
        X_cols_dropped=X_cols_dropped,
        fit_nan_counts=nan_counts_kept,
        k=int(k),
        weights=weights,
        n_train=int(n_total),
        is_spilled=True,
    )

    import warnings as _warnings
    if X_cols_dropped:
        _warnings.warn(
            f"fit_knn_imputer: dropped {len(X_cols_dropped)} all-NaN "
            f"training column(s): {X_cols_dropped}. Output schema = "
            f"{X_cols_kept} (sklearn KNNImputer behavior).",
            stacklevel=2,
        )

    try:
        log_fit("knn_imputer", state_bytes=result.state_nbytes())
    except Exception:
        pass

    return result
