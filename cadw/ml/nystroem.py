"""Nystroem kernel approximation (sklearn parity, OOC in n).

Nystroem (Williams & Seeger 2001): sample k landmarks, K_mm = kernel on
landmarks, SVD-based normalization_ = U / sqrt(S) @ Vh. Feature map
phi(X) = K(X, landmarks) @ normalization_.T.

Bit-identical to sklearn.kernel_approximation.Nystroem for
n <= _LANDMARK_PERMUTATION_LIMIT (permutation allocation limit).
Above limit: reservoir sampling of landmarks (uniform but not sklearn
parity for same seed) + ApproximationInfo.

State O(model): O(k*d + k^2) = O(n_components^2). RAM O(batch) per
stream; k landmarks collected row-by-row from df.

Closes KernelPCA n>1e4 debt: use fit_nystroem_kernel_map + fit_pca
instead of fit_kernel_pca for large n.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import pyarrow as pa

from cadw.core.approximation import ApproximationInfo
from cadw.ml._logging import log_fit
from cadw.ops.linalg import pairwise_kernel


_SUPPORTED_KERNELS = ("linear", "rbf", "poly", "laplacian",
                     "sigmoid", "cosine")
_LANDMARK_PERMUTATION_LIMIT = 10_000_000  # 8*limit = 80 MB permutation


def _batch_to_X(b, X_cols):
    """Extract (b, d) float64 from pa.RecordBatch (NaN preserved)."""
    arrs = []
    for c in X_cols:
        col = b.column(c)
        if not pa.types.is_floating(col.type):
            col = col.cast(pa.float64())
        arrs.append(col.to_numpy(zero_copy_only=False).astype(
            np.float64, copy=False))
    return np.column_stack(arrs) if arrs else np.empty((b.num_rows, 0))


def _sample_landmarks_permutation(n_total, k_eff, random_state, batch_size,
                                    df, X_cols, mg):
    """Bit-identical to sklearn: rng.permutation(n)[:k_eff].

    Peak = n_total * int64 (8 B per n) for permutation.
    Requires n_total <= _LANDMARK_PERMUTATION_LIMIT.
    """
    rng = np.random.RandomState(random_state)
    inds = rng.permutation(n_total)[:k_eff]  # (k_eff,)
    # Sorted indices for streaming selection; positions to restore order
    sorted_positions = np.argsort(inds)
    sorted_inds = inds[sorted_positions]
    basis_sorted = np.empty((k_eff, len(X_cols)), dtype=np.float64)
    row = 0
    sel_pos = 0
    for b in df._iter_filtered_batches(batch_size=batch_size):
        if mg is not None and hasattr(mg, "check_batch"):
            mg.check_batch(b, name="nystroem.X", extra_factor=2.0)
        n_b = b.num_rows
        end = row + n_b
        X_b = _batch_to_X(b, X_cols)
        if np.isnan(X_b).any():
            raise ValueError(
                "fit_nystroem: X contains NaN; impute first (fit_knn_imputer)")
        # Collect rows at sorted_inds in [row, end)
        while sel_pos < k_eff and sorted_inds[sel_pos] < end:
            local = int(sorted_inds[sel_pos] - row)
            basis_sorted[sel_pos] = X_b[local]
            sel_pos += 1
        row = end
        if sel_pos >= k_eff:
            break
    if sel_pos < k_eff:
        raise RuntimeError(
            f"fit_nystroem: only {sel_pos}/{k_eff} landmarks collected")
    # Restore original order
    basis = np.empty_like(basis_sorted)
    basis[sorted_positions] = basis_sorted
    return basis, inds


def _sample_landmarks_reservoir(n_total, k_eff, random_state, batch_size,
                                  df, X_cols, mg):
    """Reservoir sampling (uniform, NOT sklearn parity for same seed).

    For n > _LANDMARK_PERMUTATION_LIMIT: O(k*d) RAM (vs O(n) permutation).
    """
    rng = np.random.RandomState(random_state)
    basis = np.empty((k_eff, len(X_cols)), dtype=np.float64)
    inds = np.empty(k_eff, dtype=np.int64)
    n_seen = 0
    n_filled = 0
    for b in df._iter_filtered_batches(batch_size=batch_size):
        if mg is not None and hasattr(mg, "check_batch"):
            mg.check_batch(b, name="nystroem.X", extra_factor=2.0)
        n_b = b.num_rows
        X_b = _batch_to_X(b, X_cols)
        if np.isnan(X_b).any():
            raise ValueError(
                "fit_nystroem: X contains NaN; impute first (fit_knn_imputer)")
        for i in range(n_b):
            gidx = n_seen + i
            if n_filled < k_eff:
                basis[n_filled] = X_b[i]
                inds[n_filled] = gidx
                n_filled += 1
            else:
                j = rng.randint(0, gidx + 1)
                if j < k_eff:
                    basis[j] = X_b[i]
                    inds[j] = gidx
        n_seen += n_b
    if n_filled < k_eff:
        raise RuntimeError(
            f"fit_nystroem reservoir: only {n_filled}/{k_eff} filled")
    # Sort by original index for determinism-ish (reservoir nondet anyway)
    order = np.argsort(inds)
    return basis[order], inds[order]


@dataclass
class NystroemResult:
    """Nystroem kernel approximation fit result.

    `components_` (k_eff, d) = sampled landmarks.
    `normalization_` (k_eff, k_eff) = U / sqrt(S) @ Vh (SVD of K_mm).
    `component_indices_` (k_eff,) = indices in training set.

    transform(Q): K(Q, components_) @ normalization_.T -> CadwDataFrame.

    State O(model): O(k*d + k*k). RAM O(batch) per stream in transform.
    """
    components_: Optional[np.ndarray] = None
    normalization_: Optional[np.ndarray] = None
    component_indices_: Optional[np.ndarray] = None
    kernel: str = "rbf"
    gamma: float = 1.0
    degree: int = 3
    coef0: float = 1.0
    n_components: int = 0
    n_train: int = 0
    X_cols: list = field(default_factory=list)
    approximation: Optional[list] = None

    def state_nbytes(self) -> int:
        if self.components_ is None:
            return 0
        sz = 256
        sz += int(self.components_.nbytes)
        sz += int(self.normalization_.nbytes)
        sz += int(self.component_indices_.nbytes)
        return int(sz)

    def release_train_state(self) -> None:
        self.components_ = None
        self.normalization_ = None
        self.component_indices_ = None

    def transform(self, df, X_cols=None, *, batch_size=None,
                  memory_guard=None):
        """Feature map for Q (streaming, OOC in n_q). Returns CadwDataFrame."""
        if self.components_ is None:
            raise RuntimeError(
                "NystroemResult.transform: model released")
        X_cols = list(X_cols) if X_cols is not None else list(self.X_cols)
        if set(X_cols) != set(self.X_cols):
            raise ValueError(
                f"NystroemResult.transform: X_cols mismatch "
                f"(fit: {self.X_cols}, got: {X_cols})")
        X_cols = list(self.X_cols)  # reorder to fit order

        k = self.n_components
        n_q = int(df.count())
        if n_q == 0:
            raise ValueError("NystroemResult.transform: df is empty")

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

        from cadw.dataset.dataset import Dataset
        from cadw import CadwDataFrame

        schema = pa.schema([pa.field(f"nys_{i}", pa.float64())
                            for i in range(k)])

        def _iter():
            for qb in df._iter_filtered_batches(batch_size=bsz):
                if mg is not None and hasattr(mg, "check_batch"):
                    mg.check_batch(qb, name="nystroem.Q", extra_factor=3.0)
                Q = _batch_to_X(qb, X_cols)
                if Q.shape[0] == 0:
                    continue
                K_new = pairwise_kernel(
                    Q, self.components_, kernel=self.kernel,
                    gamma=self.gamma, degree=self.degree,
                    coef0=self.coef0, memory_guard=mg,
                )
                Z = K_new @ self.normalization_.T
                cols = [pa.array(Z[:, i], type=pa.float64())
                        for i in range(k)]
                yield pa.RecordBatch.from_arrays(cols, schema=schema)

        ds = Dataset.from_batches(schema, _iter(),
                                   df._new_dataset_dir())
        return CadwDataFrame(ds, memory_guard=mg,
                             spill_dir=getattr(df, "_spill_dir", None))

    def fit_transform(self, df, *, batch_size=None, memory_guard=None):
        """Feature map for training data (sklearn parity fit_transform).

        Note: streams df again; caller must not have released state.
        """
        if self.components_ is None:
            raise RuntimeError(
                "NystroemResult.fit_transform: model released")
        out_df = self.transform(df, X_cols=self.X_cols,
                                 batch_size=batch_size,
                                 memory_guard=memory_guard)
        parts = []
        for b in out_df._iter_filtered_batches():
            parts.append(np.column_stack([
                b.column(f"nys_{i}").to_numpy(zero_copy_only=False)
                for i in range(self.n_components)
            ]))
        return np.vstack(parts) if parts else np.empty((0, self.n_components))


def fit_nystroem_kernel_map(
    df,
    X_cols,
    *,
    n_components: int = 100,
    kernel: str = "rbf",
    gamma: Optional[float] = None,
    degree: int = 3,
    coef0: float = 1.0,
    random_state: Optional[int] = None,
    batch_size: Optional[int] = None,
    memory_guard=None,
    _permutation_limit: Optional[int] = None,
) -> NystroemResult:
    """Fit Nystroem kernel approximation (sklearn parity).

    Args:
      df: CadwDataFrame with X_cols numeric (no NaN allowed).
      X_cols: list[str].
      n_components: number of landmarks k.
      kernel: "linear"|"rbf"|"poly"|"laplacian"|"sigmoid"|"cosine".
      gamma: rbf/poly/laplacian/sigmoid; None -> 1/d (sklearn default).
      degree, coef0: poly/sigmoid.
      random_state: seed for landmark sampling (sklearn parity for
        n <= limit).
      _permutation_limit: test-only override of _LANDMARK_PERMUTATION_LIMIT.

    Returns NystroemResult. For n <= limit: bit-identical to sklearn
    Nystroem (same seed, same n_components). Above limit: reservoir
    sampling (uniform, not sklearn parity) + ApproximationInfo disclosed.

    State O(model): O(k*d + k*k).
    """
    if not X_cols:
        raise ValueError("fit_nystroem: X_cols is empty")
    if n_components < 1:
        raise ValueError(
            f"fit_nystroem: n_components must be >= 1; got {n_components}")
    if kernel not in _SUPPORTED_KERNELS:
        raise ValueError(
            f"fit_nystroem: kernel must be one of {_SUPPORTED_KERNELS}; "
            f"got {kernel!r}")
    if gamma is not None and gamma <= 0:
        raise ValueError(f"fit_nystroem: gamma must be > 0; got {gamma}")
    if degree < 1:
        raise ValueError(f"fit_nystroem: degree must be >= 1; got {degree}")
    missing = [c for c in X_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"fit_nystroem: columns missing: {missing}; "
            f"available: {df.columns}")

    mg = memory_guard or getattr(df, "_memory_guard", None)
    n_total = int(df.count())
    d = len(X_cols)
    if n_total < 1:
        raise ValueError(f"fit_nystroem: n must be >= 1; got {n_total}")
    if batch_size is None:
        batch_size = max(1, min(n_total, 1000))
    batch_size = max(1, min(int(batch_size), n_total))

    k_eff = min(n_components, n_total)
    if k_eff < n_components:
        warnings.warn(
            f"fit_nystroem: n_components={n_components} > n_samples="
            f"{n_total}; using k={k_eff} (sklearn Nystroem behavior).",
            stacklevel=2,
        )

    gamma_eff = (1.0 / d if gamma is None else float(gamma))
    limit = (_permutation_limit if _permutation_limit is not None
             else _LANDMARK_PERMUTATION_LIMIT)

    approximations = None
    if n_total <= limit:
        basis, component_indices = _sample_landmarks_permutation(
            n_total, k_eff, random_state, batch_size, df, X_cols, mg)
    else:
        basis, component_indices = _sample_landmarks_reservoir(
            n_total, k_eff, random_state, batch_size, df, X_cols, mg)
        approximations = [ApproximationInfo(
            type="subsample",
            axis="n",
            requested=n_total,
            effective=k_eff,
            reason=(
                f"n={n_total} > permutation_limit={limit}: landmarks "
                f"sampled via reservoir (O(k*d) RAM, uniform but NOT "
                f"sklearn parity for same seed)"),
            accuracy_impact=(
                f"k={k_eff} landmarks via reservoir; sklearn parity "
                f"requires n <= {limit}"),
        )]


    # K_mm = kernel(basis, basis) — k x k
    if mg is not None and hasattr(mg, "check_state"):
        mg.check_state(k_eff * k_eff * 8 * 2 + k_eff * d * 8,
                       name="nystroem.K_mm")
    K_mm = pairwise_kernel(
        basis, basis, kernel=kernel, gamma=gamma_eff,
        degree=degree, coef0=coef0, memory_guard=mg,
    )
    # SVD (sklearn: U, S, V; numpy returns U, S, Vh)
    U, S, Vh = np.linalg.svd(K_mm)
    S = np.clip(S, 1e-12, None)
    # normalization_ = (U / sqrt(S)) @ Vh — symmetric for PSD K_mm
    normalization_ = (U / np.sqrt(S)[None, :]) @ Vh

    result = NystroemResult(
        components_=basis.astype(np.float64),
        normalization_=normalization_.astype(np.float64),
        component_indices_=component_indices.astype(np.int64),
        kernel=kernel, gamma=float(gamma_eff),
        degree=int(degree), coef0=float(coef0),
        n_components=int(k_eff), n_train=int(n_total),
        X_cols=list(X_cols),
        approximation=approximations,
    )

    try:
        log_fit("nystroem", state_bytes=result.state_nbytes())
    except Exception:
        pass

    return result
