"""Local Outlier Factor (Breunig et al. 2000).

OOC in n via 5-pass SpillDir + memmap:
  Pass 1: X (n, d) → memmap.
  Pass 2: k-NN (k+1, includes self) via knn_query_stream → idx/dist memmap.
  Pass 3: k-dist per point (dist[:, -1], self excluded) → memmap.
  Pass 4: LRD = 1/(mean(max(dist, kdist[neighbor])) + 1e-10) → memmap.
  Pass 5: LOF = mean(LRD[neighbor]) / LRD[self] → SpillArray.

RAM: O(batch * (d + k * 2 * 8)) per pass, persistent state O(model)=256.
Disk: O(n * (d + k+1 + 1 + 1 + 1) * 8) transient (SpillDir), scores O(n).

predict deferred v0.9.198 (X memmap persistence + neighbor LRD lookup
needs lifecycle design — SpillDir cleanup vs predict-on-new-data).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pyarrow as pa

from cadw.core.spill import AutoArrayResult, SpillArray, SpillDir
from cadw.ml._logging import log_fit
from cadw.ops.linalg import knn_query_stream


@dataclass
class LOFResult:
    """LOF fit result (v0.9.197).

    `scores` = LOF values per training point. LOF > 1.0 → above-average
    local-density-inverse (Breunig 2000). `labels` = 1 for outlier,
    0 for inlier. `is_spilled` True → on-disk (SpillArray); iterate or
    .to_numpy() (raises if spilled).

    Threshold semantics:
      * contamination=None (default): threshold=1.0 (classic Breunig).
        Note: LOF is a *relative* score, mean(LOF)=1 by construction;
        ~50% of points will have LOF > 1.0 for balanced data. This is
        NOT "flag half as outliers" — it's the natural LOF=1 reference
        boundary. For "top-N% flagged", use contamination.
      * contamination=c (0<c<0.5): threshold = (1-c) percentile of
        scores; requires output="memory".
    """
    scores: Union[np.ndarray, AutoArrayResult]
    labels: Union[np.ndarray, AutoArrayResult]
    threshold: float
    k: int
    contamination: Optional[float]
    is_spilled: bool = False

    def state_nbytes(self) -> int:
        """Honest O(model) state (DEC-033 CW-1).

        Spilled: 256. In-memory: 256 + scores.nbytes + labels.nbytes.
        """
        sz = 256
        if self.is_spilled:
            return sz
        for arr in (self.scores, self.labels):
            if isinstance(arr, np.ndarray):
                sz += int(arr.nbytes)
            elif hasattr(arr, "to_numpy"):
                try:
                    sz += int(arr.to_numpy().nbytes)
                except Exception:
                    pass
        return int(sz)


def _batch_to_X(b, X_cols):
    """Extract (b.num_rows, d) float64 from pa.RecordBatch columns."""
    arrs = []
    for c in X_cols:
        col = b.column(c)
        if not pa.types.is_floating(col.type):
            col = col.cast(pa.float64())
        arrs.append(col.to_numpy(zero_copy_only=False).astype(np.float64,
                                                              copy=False))
    return np.column_stack(arrs) if arrs else np.empty((b.num_rows, 0))


def _resolve_batch_size(n_total: int, d: int, k: int, mg) -> int:
    """Auto batch_size from guard: want kNN merge (b, k+1 + chunk) to fit.

    Worst transient per pass: b * (d + (k+1)) * 2 * 8 (X_b + merged dist/idx)
    + n_q * chunk * 8 (knn internal, chunk auto). Rough target: 8 MB or 5%
    of guard. Simple heuristic: rows such that b * (d + k) * 16 <= 8 MB.
    """
    if mg is None:
        return max(1, min(n_total, 1000))
    try:
        avail = mg.available_bytes()
    except Exception:
        return max(1, min(n_total, 1000))
    target = min(8 * 1024 * 1024, avail // 20)
    denom = max(1, (d + k + 4) * 16)
    b = max(1, target // denom)
    return int(min(b, n_total))


def fit_lof(
    df,
    X_cols,
    *,
    k: int = 20,
    contamination: Optional[float] = None,
    output: str = "auto",
    batch_size: Optional[int] = None,
    memory_guard=None,
) -> LOFResult:
    """Fit LOF (Breunig 2000) via 5-pass SpillDir+memmap.

    Args:
      df: CadwDataFrame with X_cols numeric.
      X_cols: list[str] of feature columns.
      k: neighbors (k=20 → kNN with 21 incl. self; sklearn convention).
      contamination: if not None, threshold at (1-c)*100 percentile;
        requires output="memory" (percentile needs full array).
      output: "memory" | "auto" | "disk" for scores (SpillArray
        spill_ratio: 0.0 for disk, 0.5 for auto, no-spill for memory).
      batch_size: rows per batch; auto from guard.
      memory_guard: optional MemoryGuard.

    Returns LOFResult. state_nbytes = 256 (spilled) or scores+labels+n.

    predict() deferred to v0.9.198 (X memmap persistence pattern).
    """
    if not X_cols:
        raise ValueError("fit_lof: X_cols is empty")
    if k < 1:
        raise ValueError(f"fit_lof: k must be >= 1; got {k}")
    if output not in ("memory", "auto", "disk"):
        raise ValueError(
            f"fit_lof: output must be 'memory'|'auto'|'disk'; got {output!r}")
    if contamination is not None:
        if not (0.0 < float(contamination) < 1.0):
            raise ValueError(
                f"fit_lof: contamination must be in (0, 1); "
                f"got {contamination!r}")
        if output != "memory":
            raise NotImplementedError(
                f"fit_lof: contamination={contamination!r} requires "
                f"output='memory' (percentile over full scores array). "
                f"Use output='memory' or contamination=None.")

    mg = memory_guard or getattr(df, "_memory_guard", None)

    # Confirm X_cols exist
    missing = [c for c in X_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"fit_lof: columns not in df: {missing}; "
            f"available: {df.columns}")

    n_total = int(df.count())
    d = len(X_cols)
    if n_total < k + 2:
        raise ValueError(
            f"fit_lof: need n >= k+2 (k={k}); got n={n_total}")

    if batch_size is None:
        batch_size = _resolve_batch_size(n_total, d, k, mg)
    batch_size = max(1, min(int(batch_size), n_total))

    result = None
    with SpillDir(guard=mg, name="lof") as sdir:
        # --- Pass 1: X → memmap ---
        X_path = sdir.path("X.bin")
        X_mm = np.memmap(str(X_path), dtype=np.float64, mode="w+",
                         shape=(n_total, d))
        row = 0
        for b in df._iter_filtered_batches(batch_size=batch_size):
            if mg is not None and hasattr(mg, "check_batch"):
                mg.check_batch(b, name="lof.X", extra_factor=2.0)
            n_b = b.num_rows
            X_mm[row:row + n_b] = _batch_to_X(b, X_cols)
            row += n_b
        if row != n_total:
            raise RuntimeError(
                f"fit_lof: X rows mismatch ({row} != {n_total})")
        X_mm.flush()

        # --- Pass 2: k-NN (k+1 incl self) → idx/dist memmap ---
        idx_mm = np.memmap(str(sdir.path("idx.bin")),
                           dtype=np.int64, mode="w+",
                           shape=(n_total, k + 1))
        dist_mm = np.memmap(str(sdir.path("dist.bin")),
                            dtype=np.float64, mode="w+",
                            shape=(n_total, k + 1))

        def _x_iter():
            # Re-iterable; each knn_query_stream call iterates fresh.
            for s in range(0, n_total, batch_size):
                e = min(s + batch_size, n_total)
                yield np.ascontiguousarray(X_mm[s:e])

        for q_start in range(0, n_total, batch_size):
            q_end = min(q_start + batch_size, n_total)
            Q = np.ascontiguousarray(X_mm[q_start:q_end])
            idx_b, dist_b = knn_query_stream(
                _x_iter(), Q, k + 1, return_dist=True,
                memory_guard=mg,
            )
            idx_mm[q_start:q_end] = idx_b
            dist_mm[q_start:q_end] = dist_b
        idx_mm.flush()
        dist_mm.flush()

        # --- Pass 3: k-dist per point (self excluded; sklearn: dist
        # to k-th neighbor where 0-index among n_neighbors-1=last col
        # = k-dist; matches sklearn _distances_fit_X_[:, n_neighbors_-1]) ---
        kdist_mm = np.memmap(str(sdir.path("kdist.bin")),
                             dtype=np.float64, mode="w+",
                             shape=(n_total,))
        for s in range(0, n_total, batch_size):
            e = min(s + batch_size, n_total)
            # dist_b[:, -1] = distance to (k+1)-th neighbor; 0-index 0
            # is self (dist=0). k-dist = dist to k-th (non-self) =
            # dist[:, -1] (last of k+1, since sorted ascending).
            kdist_mm[s:e] = dist_mm[s:e, -1]
        kdist_mm.flush()

        # --- Pass 4: LRD → memmap ---
        # sklearn _local_reachability_density:
        #   dist_k = _distances_fit_X_[neighbors_indices, n_neighbors_-1]
        #   reach = max(dist_X, dist_k)  (neighbors: [:, 1:] to skip self)
        #   lrd = 1 / (mean(reach, axis=1) + 1e-10)
        lrd_mm = np.memmap(str(sdir.path("lrd.bin")),
                           dtype=np.float64, mode="w+",
                           shape=(n_total,))
        for s in range(0, n_total, batch_size):
            e = min(s + batch_size, n_total)
            idx_b = idx_mm[s:e]   # (b, k+1)
            dist_b = dist_mm[s:e]  # (b, k+1)
            kdist_neigh = kdist_mm[idx_b]   # (b, k+1)
            reach = np.maximum(dist_b, kdist_neigh)  # (b, k+1)
            reach = reach[:, 1:]  # drop self (first neighbor)
            lrd_mm[s:e] = 1.0 / (reach.mean(axis=1) + 1e-10)
        lrd_mm.flush()

        # --- Pass 5: LOF → SpillArray ---
        # sklearn: lof = mean(lrd[neighbors]) / lrd[self]
        spill_ratio = {"memory": 0.5, "auto": 0.5, "disk": 0.0}[output]
        if output == "memory":
            spill_ratio = 1.0  # never spill (unless forced by guard)
        scores_sp = SpillArray(
            guard=mg, width=1, dtype=np.float64,
            name="lof.scores", spill_ratio=spill_ratio)
        for s in range(0, n_total, batch_size):
            e = min(s + batch_size, n_total)
            idx_b = idx_mm[s:e][:, 1:]  # (b, k)
            lrd_neigh = lrd_mm[idx_b]  # (b, k)
            lrd_self = lrd_mm[s:e]     # (b,)
            lof_b = lrd_neigh.mean(axis=1) / (lrd_self + 1e-10)
            scores_sp.write_array(lof_b)
        scores = scores_sp.finalize()

        # --- labels (threshold) ---
        if contamination is not None:
            # output="memory" enforced at top
            scores_np = scores.to_numpy()
            thr = float(np.percentile(
                scores_np, 100.0 * (1.0 - float(contamination))))
        else:
            thr = 1.0

        if scores.is_spilled:
            labels_sp = SpillArray(
                guard=mg, width=1, dtype=np.int8,
                name="lof.labels",
                spill_ratio=0.0)
            for chunk in scores:
                labels_sp.write_array(
                    (chunk > thr).astype(np.int8))
            labels = labels_sp.finalize()
        else:
            labels = (scores.to_numpy() > thr).astype(np.int8)

        # Release memmaps before SpillDir cleanup (best-effort)
        try:
            del _x_iter
        except NameError:
            pass
        del X_mm, idx_mm, dist_mm, kdist_mm, lrd_mm

    return LOFResult(
            scores=scores, labels=labels, threshold=float(thr),
            k=int(k), contamination=contamination,
            is_spilled=scores.is_spilled,
        )
