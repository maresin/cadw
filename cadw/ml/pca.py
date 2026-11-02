
"""Principal Component Analysis (v0.7.2).

Composition: Reduce (CovarianceOp) → eigh. Peak RAM = batch + O(d²).

Model:
    Fit: cov (d×d), mean (d), eigh → components (k, d).
    Transform: (X - mean) @ components.T.
    Inverse: Z @ components + mean.

    sklearn.PCA-compatible attributes:
    - components_               (k, d)
    - mean_                     (d,)
    - explained_variance_       (k,)
    - explained_variance_ratio_ (k,)
    - singular_values_          (k,)
"""
from __future__ import annotations

from cadw.ml._common import _column_to_f64, _stack_numeric

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.ops.reduce import reduce
from cadw.ops.reduce import CovarianceMeanOp

if TYPE_CHECKING:  # v0.9.9 (B2): forward-annotation only
    from cadw.dataframe import CadwDataFrame






@dataclass
class PCA:
    """Fitted PCA model (sklearn.PCA-compatible attributes).

    v0.9.9 (B4): PCA is a *fitted-model* class — constructor takes
    all fitted attributes; `n_components` is NOT an __init__ kwarg.
    Use `fit_pca(df, X_cols, n_components=...)` (or
    `CadwDataFrame.fit_pca(...)`) to produce a PCA instance.
    """

    components_: np.ndarray             # (k, d)
    mean_: np.ndarray                   # (d,)
    explained_variance_: np.ndarray     # (k,)
    explained_variance_ratio_: np.ndarray  # (k,)
    singular_values_: np.ndarray        # (k,)
    feature_names_: list[str]
    n_components_: int
    n_samples_: int
    n_features_in_: int
    fitted_: bool = True
    method_: str = "exact"   # v0.9.93: "exact" | "randomized"

    def to_dict(self) -> dict:
        return {
            "components": self.components_.tolist(),
            "mean": self.mean_.tolist(),
            "explained_variance": self.explained_variance_.tolist(),
            "explained_variance_ratio": (
                self.explained_variance_ratio_.tolist()
            ),
            "singular_values": self.singular_values_.tolist(),
            "feature_names": self.feature_names_,
            "n_components": int(self.n_components_),
            "n_samples": int(self.n_samples_),
            "method": getattr(self, "method_", "exact"),
        }

    def _check_features(self, X: np.ndarray) -> None:
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"PCA: X has {X.shape[1]} features, "
                f"expected {self.n_features_in_}"
            )

    def transform(self, X):
        """Project X to principal component space.

        - np.ndarray (n, d) → (n, k)
        - CadwDataFrame → CadwDataFrame with `pc0..pc{k-1}` columns.
          Rows with null feature → null PC values.
        """
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            self._check_features(X)
            return (X - self.mean_) @ self.components_.T

        from cadw.dataframe import CadwDataFrame
        from cadw.dataset import Dataset
        if not isinstance(X, CadwDataFrame):
            raise TypeError(
                f"transform: X must be np.ndarray or CadwDataFrame; "
                f"got {type(X).__name__}"
            )
        for c in self.feature_names_:
            if c not in X.columns:
                raise KeyError(f"transform: column {c!r} not in df")

        orig_schema = X._dataset.schema
        out_fields = [orig_schema.field(c) for c in X._columns]
        for i in range(self.n_components_):
            out_fields.append(pa.field(f"pc{i}", pa.float64(), nullable=True))
        out_schema = pa.schema(out_fields)
        mean = self.mean_
        comp = self.components_

        def _iter_transform():
            for b in X._iter_filtered_batches():
                Xm = _stack_numeric(b, self.feature_names_)
                n = Xm.shape[0]
                # NaN row → all-NaN PCs (converted to Arrow null below)
                proj = np.full((n, self.n_components_), np.nan, dtype=np.float64)
                mask = ~np.isnan(Xm).any(axis=1)
                if mask.any():
                    Xv = Xm[mask]
                    proj[mask] = (Xv - mean) @ comp.T
                arrays = [b.column(c) for c in X._columns]
                for i in range(self.n_components_):
                    col_np = proj[:, i]
                    null_mask = np.isnan(col_np)
                    arr = pa.array(
                        col_np, type=pa.float64(),
                        mask=null_mask if null_mask.any() else None,
                    )
                    arrays.append(arr)
                yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

        new_ds = Dataset.from_batches(
            out_schema, _iter_transform(), X._new_dataset_dir()
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=X._memory_guard,
            spill_dir=X._spill_dir,
        )

    def fit_transform(self, df) -> "CadwDataFrame":
        """Fit on df, then transform. For CadwDataFrame input returns df
        with `pc0..pc{k-1}` columns (original columns preserved)."""
        self._fit_df = df  # noqa (for callers who use both)
        return self.transform(df)

    def inverse_transform(self, Z):
        """Reconstruct feature space from PC space.

        - np.ndarray (n, k) → (n, d)
        - CadwDataFrame with pc0..pc{k-1} → CadwDataFrame with feature
          columns overwritten (reconstructed values, float64).
        """
        if isinstance(Z, np.ndarray):
            if Z.ndim == 1:
                Z = Z.reshape(1, -1)
            if Z.shape[1] != self.n_components_:
                raise ValueError(
                    f"inverse_transform: Z has {Z.shape[1]} components, "
                    f"expected {self.n_components_}"
                )
            return Z @ self.components_ + self.mean_

        from cadw.dataframe import CadwDataFrame
        from cadw.dataset import Dataset
        if not isinstance(Z, CadwDataFrame):
            raise TypeError(
                f"inverse_transform: Z must be np.ndarray or CadwDataFrame; "
                f"got {type(Z).__name__}"
            )
        pc_cols = [f"pc{i}" for i in range(self.n_components_)]
        for c in pc_cols:
            if c not in Z.columns:
                raise KeyError(f"inverse_transform: column {c!r} not in df")

        orig_schema = Z._dataset.schema
        # v0.7.2-fix: replace feature columns (avoid duplicates if Z already
        # has x0..xd from a preceding transform).
        existing_features = set(self.feature_names_) & set(Z._columns)
        passthrough_cols = [
            c for c in Z._columns if c not in existing_features
        ]
        out_fields = [orig_schema.field(c) for c in passthrough_cols]
        for name in self.feature_names_:
            out_fields.append(pa.field(name, pa.float64(), nullable=True))
        out_schema = pa.schema(out_fields)
        comp = self.components_
        mean = self.mean_

        def _iter_inv():
            for b in Z._iter_filtered_batches():
                Zm = np.column_stack(
                    [_column_to_f64(b.column(c)) for c in pc_cols]
                )
                n = Zm.shape[0]
                recon = np.full((n, self.n_features_in_), np.nan,
                                dtype=np.float64)
                mask = ~np.isnan(Zm).any(axis=1)
                if mask.any():
                    recon[mask] = Zm[mask] @ comp + mean
                arrays = [b.column(c) for c in passthrough_cols]
                for i in range(self.n_features_in_):
                    col_np = recon[:, i]
                    null_mask = np.isnan(col_np)
                    arr = pa.array(
                        col_np, type=pa.float64(),
                        mask=null_mask if null_mask.any() else None,
                    )
                    arrays.append(arr)
                yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

        new_ds = Dataset.from_batches(
            out_schema, _iter_inv(), Z._new_dataset_dir()
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=Z._memory_guard,
            spill_dir=Z._spill_dir,
        )


def _fit_pca_randomized(df, X_cols, k, mg, n_oversamples, random_state):
    """Randomized SVD wrapper returning PCA-compatible model.

    Uses cadw.ml.randsvd.fit_randomized_svd (already OOC via Reduce +
    power iterations on (n, k) state). Convert its output to PCA
    convention:
      components_       <- randsvd.components_  (k, d)
      mean_             <- randsvd.mean_        (d,)
      singular_values_  <- randsvd.singular_values_  (k,)
      explained_var_    <- singular_values^2 / (n-1)  [sample var]
      explained_var_ratio_ <- singular_values^2 / sum_all_sv^2
    The last requires one extra pass for total sum of squared centered X
    (trace(cov)·(n-1)); randsvd.explained_variance_ratio() provides it.

    Covariance_ is not exposed on PCA; only components/ev/ratios.
    Deterministic given random_state.
    """
    from cadw.ml.randsvd import fit_randomized_svd

    svd = fit_randomized_svd(
        df, X_cols, n_components=int(k),
        n_oversamples=int(n_oversamples),
        seed=int(random_state),
        memory_guard=mg)

    # n_samples for sample-variance conversion.
    n_samples = 0
    for b in df._iter_filtered_batches():
        n_samples += b.num_rows
    if n_samples < 2:
        raise ValueError(
            f"fit_pca: need at least 2 valid rows; got {n_samples}")

    sv = np.asarray(svd.singular_values_, dtype=np.float64)
    ev = (sv ** 2) / max(n_samples - 1, 1)
    ratio = svd.explained_variance_ratio(df, memory_guard=mg)
    ratio = np.asarray(ratio, dtype=np.float64)
    if ratio.size > 0 and ratio.sum() > 1e-12:
        total_var = float(ev.sum() / ratio.sum())
    else:
        total_var = float(ev.sum())
    if total_var > 0.0:
        ev_ratio = ev / total_var
    else:
        ev_ratio = np.zeros_like(ev)

    from cadw.ml._logging import log_fit
    log_fit("pca",
            state_bytes=int(svd.components_.nbytes + sv.nbytes))

    return PCA(
        components_=svd.components_,
        mean_=svd.mean_,
        explained_variance_=ev,
        explained_variance_ratio_=ev_ratio,
        singular_values_=sv,
        feature_names_=list(X_cols),
        n_components_=int(k),
        n_samples_=int(n_samples),
        n_features_in_=len(X_cols),
        method_="randomized",
    )


def fit_pca(
    df,
    X_cols,
    n_components: Optional[int] = None,
    *,
    method: str = "auto",
    n_oversamples: int = 10,
    random_state: int = 0,
    memory_guard: Optional[MemoryGuard] = None,
) -> PCA:
    """Fit PCA via Reduce(CovarianceOp) + eigh.

    - X_cols: numeric feature columns.
    - n_components: k (default = len(X_cols)).
    - memory_guard: default df._memory_guard.

    Peak RAM = batch + O(d²). Independent of N.

    Raises: ValueError, KeyError, TypeError as in fit_linear.
    """
    if isinstance(X_cols, str):
        X_cols = [X_cols]
    else:
        X_cols = list(X_cols)
    if not X_cols:
        raise ValueError("fit_pca: X_cols must be non-empty")
    for c in X_cols:
        if c not in df.columns:
            raise KeyError(f"fit_pca: column {c!r} not in df")

    schema = df._dataset.schema
    for c in X_cols:
        t = schema.field(c).type
        if not (pa.types.is_integer(t) or pa.types.is_floating(t)):
            raise TypeError(
                f"fit_pca: column {c!r} has type {t}; "
                f"numeric required (use encode_categorical for strings)"
            )

    d = len(X_cols)
    if n_components is None:
        k = d
    else:
        if not isinstance(n_components, int):
            raise TypeError(
                f"fit_pca: n_components must be int or None; "
                f"got {type(n_components).__name__}"
            )
        if not (1 <= n_components <= d):
            raise ValueError(
                f"fit_pca: n_components must be in [1, {d}]; "
                f"got {n_components}"
            )
        k = n_components

    mg = memory_guard or df._memory_guard

    # v0.9.93: method routing.
    if method not in ("auto", "exact", "randomized"):
        raise ValueError(
            f"fit_pca: method must be 'auto'|'exact'|'randomized'; "
            f"got {method!r}")
    if method == "auto":
        if mg is None or d <= 1000:
            method = "exact"
        else:
            est_closed = 3 * d * d * 8
            if hasattr(mg, "available_bytes") and callable(mg.available_bytes):
                try:
                    avail = mg.available_bytes()
                    method = ("randomized"
                              if est_closed > int(avail * 0.5)
                              else "exact")
                except Exception:
                    method = "exact"
            else:
                method = "exact"

    if method == "randomized":
        if n_oversamples < 0:
            raise ValueError(
                f"fit_pca: n_oversamples must be >= 0; got {n_oversamples}")
        return _fit_pca_randomized(
            df, X_cols, k, mg, n_oversamples, random_state)

    # v0.8.0: CovarianceMeanOp returns cov + mean + n in one pass.
    # 2× I/O reduction vs v0.7.2 (which re-scanned for mean).
    op = CovarianceMeanOp(X_cols)
    res = reduce(df._iter_filtered_batches(), op, mg, log=False)

    cov = res["cov"]
    mean = res["mean"]
    n_valid = res["n"]

    if n_valid < 2:
        raise ValueError(
            f"fit_pca: need at least 2 valid rows; got {n_valid}"
        )

    # eigendecomposition: eigh returns ascending eigenvalues
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort descending
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Take top k
    explained_variance = eigvals[:k]
    components = eigvecs[:, :k].T  # (k, d)
    total_var = eigvals.sum()
    if total_var > 0:
        explained_variance_ratio = explained_variance / total_var
    else:
        explained_variance_ratio = np.zeros(k, dtype=np.float64)

    # singular_values_ = sqrt(eigenvalue × (n-1)) — sklearn convention
    singular_values = np.sqrt(np.maximum(explained_variance * (n_valid - 1), 0))

    # v0.9.34 (A9): uniform ML log. PCA is one-shot (eigendecomp).
    from cadw.ml._logging import log_fit
    log_fit("pca", state_bytes=int(components.nbytes + explained_variance.nbytes))
    return PCA(
        components_=components,
        mean_=mean,
        explained_variance_=explained_variance,
        explained_variance_ratio_=explained_variance_ratio,
        singular_values_=singular_values,
        feature_names_=list(X_cols),
        n_components_=int(k),
        n_samples_=int(n_valid),
        n_features_in_=int(d),
        method_="exact",
    )
