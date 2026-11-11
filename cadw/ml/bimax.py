"""Bimax binary biclustering (Prelic 2006).

Recursive divide-and-conquer on binary matrix A (r, c) -> list of
inclusion-maximal all-1 submatrices (>= min_rows x min_cols).

Reference: Prelic et al. 2006, "A systematic comparison and evaluation
of biclustering methods for gene expression data", Bioinformatics
22(9):1122-1131. Bimax section.

Algorithm:
  1. Partition (R, C). If |sub(A[R,C])| all-1 (>= min_r, min_c): emit.
     If all-0: discard.
  2. Else: compute row_scores(i) = min(ones_i, zeros_i) per row i in R
     (over C); col_scores(j) similarly over R. Pick axis with max score.
     - best_row i* wins -> split C by row i*:
         C1 = {j : A[i*, j] == 1}, C2 = rest.
     - best_col j* wins -> split R by col j*:
         R1 = {i : A[i, j*] == 1}, R2 = rest.
  3. Recurse on both halves.
  4. Dedup: keep only inclusion-maximal biclusters.

OOC (Pattern #3 intrinsic):
  - A (r, c) int8 — bounded_by_domain {r*c: 1e8} (100 MB).
  - Exponential in min(r, c) — hard caps (max_iterations, max_biclusters).
  - refuse_at_guard on max_iterations breach (pathological cases).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import (
    MemoryGuard, MemoryLimitExceeded)
from cadw.core.warnings import SlowOperationWarning
from cadw.ml._logging import log_fit


_DEFAULT_MAX_ITER = 100_000
_DEFAULT_MAX_BICLUSTERS = 1000


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


@dataclass
class BimaxResult:
    """Bimax fit result (Prelic 2006).

    biclusters_: list of (row_tuple, col_tuple) — inclusion-maximal
    A_shape: (r, c)
    n_biclusters_: len(biclusters_)
    threshold_: binarize threshold used (A > threshold -> 1)
    n_iterations_: actual iterations used (for diagnostics)
    """
    biclusters_: list = field(default_factory=list)
    A_shape: tuple = ()
    n_biclusters_: int = 0
    threshold_: float = 0.5
    min_rows_: int = 2
    min_cols_: int = 2
    n_iterations_: int = 0

    def state_nbytes(self) -> int:
        sz = 256
        for R, C in self.biclusters_:
            sz += len(R) * 8 + len(C) * 8
        return int(sz)

    def to_dict(self) -> dict:
        return {
            "A_shape": tuple(self.A_shape),
            "n_biclusters": int(self.n_biclusters_),
            "threshold": float(self.threshold_),
            "min_rows": int(self.min_rows_),
            "min_cols": int(self.min_cols_),
            "n_iterations": int(self.n_iterations_),
            "sizes": [(len(R), len(C)) for R, C in self.biclusters_],
        }


def _bimax_core(A, min_r, min_c, max_iterations, max_biclusters):
    """Prelic 2006 BIMAX recursive core. Returns list of (R, C) tuples."""
    r, c = A.shape
    biclusters = []
    stack = [(list(range(r)), list(range(c)))]
    n_iter = 0
    iter_limit_hit = False

    while stack:
        n_iter += 1
        if n_iter > max_iterations:
            iter_limit_hit = True
            break
        if len(biclusters) >= max_biclusters:
            break

        R, C = stack.pop()
        nr, nc = len(R), len(C)
        if nr == 0 or nc == 0:
            continue

        sub = A[np.ix_(R, C)]
        s = int(sub.sum())
        n = nr * nc
        if s == 0:
            continue  # all-zero
        if s == n:
            if nr >= min_r and nc >= min_c:
                biclusters.append((tuple(R), tuple(C)))
            continue

        # score per row/col (max-balance)
        row_sums = sub.sum(axis=1)
        col_sums = sub.sum(axis=0)
        row_scores = np.minimum(row_sums, nc - row_sums)
        col_scores = np.minimum(col_sums, nr - col_sums)
        max_row_score = int(row_scores.max())
        max_col_score = int(col_scores.max())

        if max_row_score == 0 and max_col_score == 0:
            # Cannot split further (layered permutation, no bicluster)
            continue

        if max_row_score >= max_col_score:
            # Split COLS by best_row
            i_star_local = int(np.argmax(row_scores))
            pivot_row = R[i_star_local]
            row_vals = A[pivot_row, C]  # (nc,)
            C1 = [C[j] for j in range(nc) if row_vals[j] == 1]
            C2 = [C[j] for j in range(nc) if row_vals[j] == 0]
            if C1 and C2:
                stack.append((R, C1))
                stack.append((R, C2))
        else:
            # Split ROWS by best_col
            j_star_local = int(np.argmax(col_scores))
            pivot_col = C[j_star_local]
            col_vals = A[R, pivot_col]
            R1 = [R[i] for i in range(nr) if col_vals[i] == 1]
            R2 = [R[i] for i in range(nr) if col_vals[i] == 0]
            if R1 and R2:
                stack.append((R1, C))
                stack.append((R2, C))

    return biclusters, n_iter, iter_limit_hit


def _filter_inclusion_maximal(biclusters):
    """Drop biclusters that are strict subsets of others."""
    n = len(biclusters)
    if n <= 1:
        return biclusters
    sets = [(set(R), set(C)) for R, C in biclusters]
    keep = []
    for i in range(n):
        sRi, sCi = sets[i]
        is_max = True
        for j in range(n):
            if i == j:
                continue
            sRj, sCj = sets[j]
            if sRi <= sRj and sCi <= sCj and \
               (sRi, sCi) != (sRj, sCj):
                is_max = False
                break
        if is_max:
            keep.append(biclusters[i])
    return keep


def fit_bimax(
    df,
    X_cols,
    *,
    min_rows: int = 2,
    min_cols: int = 2,
    threshold: float = 0.5,
    max_iterations: int = _DEFAULT_MAX_ITER,
    max_biclusters: int = _DEFAULT_MAX_BICLUSTERS,
    batch_size: Optional[int] = None,
    memory_guard: Optional[MemoryGuard] = None,
) -> BimaxResult:
    """Bimax binary biclustering (Prelic 2006).

    Args:
      df: CadwDataFrame with numeric X_cols.
      X_cols: list[str].
      min_rows, min_cols: minimum bicluster size (default 2 x 2).
      threshold: binarize X > threshold -> 1 (default 0.5).
      max_iterations: hard cap on while-loop iterations (default 100_000;
        exponential worst case in min(r,c) — cap prevents runaway).
      max_biclusters: early exit if this many found (default 1000).
      batch_size, memory_guard: standard.

    Returns BimaxResult.

    OOC (Pattern #3 intrinsic — exponential):
      - A (r, c) int8: bounded_by_domain {r*c: 1e8} (100 MB at 1e8).
      - Complexity O(r*c*2^min(r,c)) worst case. max_iterations cap.
      - refuse_at_guard on max_iterations breach (pathological).

    Raises ValueError/KeyError/MemoryLimitExceeded.
    """
    if not X_cols:
        raise ValueError("fit_bimax: X_cols is empty")
    X_cols = list(X_cols)
    if min_rows < 1 or min_cols < 1:
        raise ValueError(
            f"fit_bimax: min_rows/min_cols must be >= 1; got "
            f"({min_rows}, {min_cols})")
    if max_iterations < 1:
        raise ValueError(
            f"fit_bimax: max_iterations must be >= 1; got {max_iterations}")
    if max_biclusters < 1:
        raise ValueError(
            f"fit_bimax: max_biclusters must be >= 1; got {max_biclusters}")
    missing = [c for c in X_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"fit_bimax: columns missing: {missing}")

    mg = memory_guard or getattr(df, "_memory_guard", None)
    r = int(df.count())
    c = len(X_cols)
    if r < 1 or c < 1:
        raise ValueError(f"need r>=1 and c>=1; got r={r}, c={c}")

    # A (r, c) int8 materialize
    A_bytes = r * c * 1
    if mg is not None and hasattr(mg, "check_state"):
        mg.check_state(A_bytes + 256, name="bimax.A_state")

    if batch_size is None:
        batch_size = max(1, min(r, 1000))
    batch_size = max(1, min(int(batch_size), r))

    # Pass 1: materialize A (binary int8)
    A_parts = []
    for b in df._iter_filtered_batches(batch_size=batch_size):
        if mg is not None and hasattr(mg, "check_batch"):
            mg.check_batch(b, name="bimax.A", extra_factor=2.0)
        X_b = _batch_to_X(b, X_cols)
        A_parts.append((X_b > threshold).astype(np.int8))
    A = (np.vstack(A_parts) if len(A_parts) > 1
         else A_parts[0] if A_parts else np.empty((0, c), dtype=np.int8))
    if A.shape != (r, c):
        raise RuntimeError(f"A shape mismatch: {A.shape} != ({r},{c})")

    # AP22 slow-op warning
    # Estimated iterations for random binary: log2(r)+log2(c) ~ 7-20.
    # For worst case, exponential. Estimate: iterations ~ min(2^min(r,c),
    # max_iterations). Conservative: if min(r,c) >= 15, could blow up.
    est_iters = min(2 ** min(r, c), max_iterations)
    est_time_s = est_iters * r * c / 1e7  # crude, ~10M ops/sec
    if est_time_s > 60.0:
        import warnings as _w
        _w.warn(
            f"fit_bimax: min(r,c)={min(r,c)}, worst-case "
            f"iterations=2^{min(r,c)}={2**min(r,c)}; capped at "
            f"{max_iterations}. Est ~{est_time_s:.0f}s "
            f"(>{60.0:.0f}s). Options: (a) reduce r or c; "
            f"(b) reduce max_iterations (early cut); "
            f"(c) use MAX-PLAID formulation.",
            SlowOperationWarning,
            stacklevel=2)

    biclusters, n_iter, hit_limit = _bimax_core(
        A, min_rows, min_cols, max_iterations, max_biclusters)

    if hit_limit:
        raise MemoryLimitExceeded(
            f"fit_bimax: max_iterations={max_iterations} exceeded "
            f"(pathological case; exponential in min(r,c)={min(r,c)}). "
            f"Options: (a) reduce r or c; (b) increase max_iterations; "
            f"(c) use MAX-PLAID formulation.",
            model="Bimax", params={"r": r, "c": c,
                                    "max_iterations": max_iterations})

    # Dedup
    biclusters = _filter_inclusion_maximal(biclusters)

    result = BimaxResult(
        biclusters_=biclusters,
        A_shape=(int(r), int(c)),
        n_biclusters_=len(biclusters),
        threshold_=float(threshold),
        min_rows_=int(min_rows),
        min_cols_=int(min_cols),
        n_iterations_=int(n_iter),
    )

    try:
        log_fit("bimax", state_bytes=result.state_nbytes())
    except Exception:
        pass

    return result
