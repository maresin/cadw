"""DBSCAN (Ester et al. 1996).

Density-based clustering: core points have >= min_samples neighbors
within eps; clusters are connected components of core points +
border points.

Algorithm:
  1. Core detection: for each point Q, count neighbors within eps.
     Chunked via sq_dists(Q_chunk, X_batch), state O(batch^2).
  2. Union-Find: for each core Q, union Q with all neighbors P within eps.
     UF parent (n int32) + size (n int32) intrinsic O(n).
  3. Labels: cluster_id = UF root for core/border points; -1 for noise.
     SpillArray (disk O(n)).

OOC:
  - core mask (n bool) + labels (n int64): SpillArray.
  - UF parent/size: O(n) int32 each — intrinsic (union-find needs
    random access). Cannot stream without approximation.
  - SlowOperationWarning (AP22) for O(n^2/eps^d) compute.
  - bounded_by_domain {n: 1e6} (UF 8 MB at 1e6).

Reference: Ester, Kriegel, Sander, Xu 1996, KDD.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import (
    MemoryGuard, MemoryLimitExceeded)
from cadw.core.spill import SpillArray
from cadw.core.warnings import SlowOperationWarning
from cadw.ml._logging import log_fit
from cadw.ops.linalg import sq_dists


_MAX_N_INTRINSIC = 1_000_000  # UF + labels: ~16 MB at 1e6


def _batch_to_X(b, X_cols):
    arrs = []
    for c in X_cols:
        col = b.column(c)
        if not pa.types.is_floating(col.type) and \
           not pa.types.is_integer(col.type):
            col = col.cast(pa.float64())
        arrs.append(col.to_numpy(zero_copy_only=False).astype(
            np.float64, copy=False))
    return np.column_stack(arrs) if arrs else np.empty((b.num_rows, 0))


class _UF:
    """Union-Find with path compression + union by rank."""
    def __init__(self, n):
        self.parent = np.arange(n, dtype=np.int32)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, i):
        # Iterative path compression
        root = i
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[i] != root:
            nxt = self.parent[i]
            self.parent[i] = root
            i = nxt
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


@dataclass
class DBSCANResult:
    """DBSCAN fit result.

    labels_: AutoArrayResult (int64) — -1 noise, 0..k-1 cluster.
    n_clusters_: int (excluding noise).
    n_noise_: int.
    core_sample_indices_: np.ndarray (bool mask, size n).
    X_cols: list.
    eps_: float.
    min_samples_: int.
    """
    labels_: object = None
    n_clusters_: int = 0
    n_noise_: int = 0
    core_sample_indices_: Optional[np.ndarray] = None
    X_cols: list = field(default_factory=list)
    eps_: float = 0.5
    min_samples_: int = 5
    n_train_: int = 0

    def state_nbytes(self) -> int:
        sz = 256
        if self.core_sample_indices_ is not None:
            sz += int(self.core_sample_indices_.nbytes)
        # labels_ may be spilled; count in-memory part only
        if self.labels_ is not None and hasattr(self.labels_, "nbytes"):
            sz += int(self.labels_.nbytes)
        return int(sz)

    def to_dict(self) -> dict:
        return {
            "n_clusters": int(self.n_clusters_),
            "n_noise": int(self.n_noise_),
            "eps": float(self.eps_),
            "min_samples": int(self.min_samples_),
            "n_train": int(self.n_train_),
        }


def _iter_X(df, X_cols, batch_size, mg):
    for b in df._iter_filtered_batches(batch_size=batch_size):
        if mg is not None and hasattr(mg, "check_batch"):
            mg.check_batch(b, name="dbscan.X", extra_factor=2.0)
        yield _batch_to_X(b, X_cols)


def fit_dbscan(
    df,
    X_cols,
    *,
    eps: float = 0.5,
    min_samples: int = 5,
    batch_size: Optional[int] = None,
    output: str = "auto",
    memory_guard: Optional[MemoryGuard] = None,
) -> DBSCANResult:
    """DBSCAN (Ester et al. 1996).

    Args:
      df: CadwDataFrame with numeric X_cols.
      X_cols: list[str].
      eps: neighborhood radius (default 0.5).
      min_samples: core threshold (default 5).
      batch_size: batch rows for streaming (auto from guard).
      output: "memory" | "auto" | "disk" (labels storage).
      memory_guard: standard.

    Returns DBSCANResult with labels_ (AutoArrayResult) + core mask.

    OOC:
      - labels + core via SpillArray (output='auto' default).
      - UF parent/size O(n) intrinsic (union-find random access).
      - bounded_by_domain {n: 1e6}; SlowOperationWarning for O(n^2) compute.
    """
    if not X_cols:
        raise ValueError("fit_dbscan: X_cols is empty")
    X_cols = list(X_cols)
    if eps <= 0:
        raise ValueError(f"fit_dbscan: eps must be > 0; got {eps}")
    if min_samples < 1:
        raise ValueError(
            f"fit_dbscan: min_samples must be >= 1; got {min_samples}")
    missing = [c for c in X_cols if c not in df.columns]
    if missing:
        raise KeyError(f"fit_dbscan: columns missing: {missing}")

    mg = memory_guard or getattr(df, "_memory_guard", None)
    n = int(df.count())
    d = len(X_cols)
    if n < 1:
        raise ValueError("fit_dbscan: empty df")
    if n > _MAX_N_INTRINSIC:
        raise MemoryLimitExceeded(
            f"fit_dbscan: n={n} > {_MAX_N_INTRINSIC} (UF + labels O(n) "
            f"intrinsic). Options: (a) subsample df; (b) reduce via "
            f"sampling + assign; (c) use MiniBatchKMeans "
            f"(approximate).",
            model="DBSCAN", params={"n": n, "min_samples": min_samples})

    if batch_size is None:
        batch_size = max(1, min(n, 1000))
    batch_size = max(1, min(int(batch_size), n))

    # State: UF parent (n i32) + size (n i32) + core (n bool)
    state_bytes = n * 4 * 2 + n * 1 + 256
    if mg is not None and hasattr(mg, "check_state"):
        mg.check_state(state_bytes, name="dbscan.state")

    # AP22: O(n^2) worst; O(n^2/eps^d) for clustered data.
    # Rough: n * (n/eps^d) operations. For eps=0.5, d=10: ~1e6 per query.
    est_ops = n * n
    est_time_s = est_ops / 1e8  # ~100M sq_dists ops/sec
    if est_time_s > 60.0:
        import warnings as _w
        _w.warn(
            f"fit_dbscan: n={n}, eps={eps}, min_samples={min_samples}; "
            f"O(n^2) compute ~{est_time_s:.0f}s (>{60.0:.0f}s). "
            f"Options: (a) increase eps; (b) subsample; "
            f"(c) use knn_query + filter (needs known k).",
            SlowOperationWarning,
            stacklevel=2)

    eps2 = float(eps) * float(eps)

    # ────────────────────────────────────────────────────────────────
    # Pass 1: core detection
    # ────────────────────────────────────────────────────────────────
    core = np.zeros(n, dtype=bool)
    global_offset = 0
    for Q_b in _iter_X(df, X_cols, batch_size, mg):
        if Q_b.shape[0] == 0:
            continue
        qn = Q_b.shape[0]
        counts = np.zeros(qn, dtype=np.int64)
        # Chunk Q by query batch to bound sq_dists (chunk, X_b)
        for X_b in _iter_X(df, X_cols, batch_size, mg):
            if X_b.shape[0] == 0:
                continue
            D = sq_dists(Q_b, X_b)  # (qn, xn)
            counts += (D <= eps2).sum(axis=1)
        core[global_offset:global_offset + qn] = counts >= min_samples
        global_offset += qn

    # ────────────────────────────────────────────────────────────────
    # Pass 2: union-find — for each core Q, union with neighbors
    # ────────────────────────────────────────────────────────────────
    uf = _UF(n)
    global_offset = 0
    for Q_b in _iter_X(df, X_cols, batch_size, mg):
        if Q_b.shape[0] == 0:
            continue
        qn = Q_b.shape[0]
        core_q = core[global_offset:global_offset + qn]
        if not core_q.any():
            global_offset += qn
            continue
        # Only core Q need union
        x_off = 0
        for X_b in _iter_X(df, X_cols, batch_size, mg):
            if X_b.shape[0] == 0:
                continue
            xn = X_b.shape[0]
            D = sq_dists(Q_b, X_b)  # (qn, xn)
            for qi in range(qn):
                if not core_q[qi]:
                    continue
                gi = global_offset + qi
                near = np.flatnonzero(D[qi] <= eps2)
                for xj_local in near:
                    gj = x_off + int(xj_local)
                    uf.union(gi, gj)
            x_off += xn
        global_offset += qn

    # ────────────────────────────────────────────────────────────────
    # Pass 3: labels from UF roots (only core/border; noise = -1)
    # ────────────────────────────────────────────────────────────────
    # Border points: not core, but neighbor to a core point -> assign
    # to that core's cluster (highest priority / first found).
    # Simpler: sklearn-style: assign non-noise = UF root of any
    # connected core; isolated non-core -> -1.
    labels_arr = np.full(n, -1, dtype=np.int64)

    # Assign labels: for each i with core[i] OR neighbor-of-core
    # (quick: only core points get union; border via re-stream)
    global_offset = 0
    for Q_b in _iter_X(df, X_cols, batch_size, mg):
        if Q_b.shape[0] == 0:
            continue
        qn = Q_b.shape[0]
        core_q = core[global_offset:global_offset + qn]
        # Core points -> UF root
        for qi in range(qn):
            gi = global_offset + qi
            if core_q[qi]:
                labels_arr[gi] = uf.find(gi)
        global_offset += qn

    # Border: non-core, but within eps of a core point -> that cluster
    global_offset = 0
    for Q_b in _iter_X(df, X_cols, batch_size, mg):
        if Q_b.shape[0] == 0:
            continue
        qn = Q_b.shape[0]
        core_q = core[global_offset:global_offset + qn]
        x_off = 0
        for X_b in _iter_X(df, X_cols, batch_size, mg):
            if X_b.shape[0] == 0:
                continue
            xn = X_b.shape[0]
            D = sq_dists(Q_b, X_b)  # (qn, xn)
            for qi in range(qn):
                gi = global_offset + qi
                if core_q[qi] or labels_arr[gi] != -1:
                    continue  # already assigned
                near = np.flatnonzero(D[qi] <= eps2)
                for xj_local in near:
                    gj = x_off + int(xj_local)
                    if core[gj]:
                        labels_arr[gi] = uf.find(gj)
                        break
            x_off += xn
        global_offset += qn

    # Compact cluster ids: 0..k-1
    unique_roots = np.unique(labels_arr[labels_arr >= 0])
    root_to_cluster = {int(r): i for i, r in enumerate(unique_roots)}
    n_clusters = len(unique_roots)
    for i in range(n):
        if labels_arr[i] >= 0:
            labels_arr[i] = root_to_cluster[int(labels_arr[i])]

    n_noise = int((labels_arr == -1).sum())

    # Store labels in SpillArray
    spill_ratio = {"memory": 1.0, "auto": 0.5, "disk": 0.0}.get(output, 0.5)
    buf = SpillArray(guard=mg, width=1, dtype=np.int64,
                     name="dbscan.labels", spill_ratio=spill_ratio)
    CHUNK = 4096
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        buf.write_array(labels_arr[s:e])
    labels_res = buf.finalize()

    result = DBSCANResult(
        labels_=labels_res,
        n_clusters_=int(n_clusters),
        n_noise_=n_noise,
        core_sample_indices_=core,
        X_cols=list(X_cols),
        eps_=float(eps),
        min_samples_=int(min_samples),
        n_train_=int(n),
    )

    try:
        log_fit("dbscan", state_bytes=result.state_nbytes())
    except Exception:
        pass

    return result
