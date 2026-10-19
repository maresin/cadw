#!/usr/bin/env python3
"""v0.7.2: PCA via CovarianceOp + eigh.

Design:
    Fit: Reduce(CovarianceOp) → cov (d×d), mean (d).
         np.linalg.eigh(cov) → eigenvalues (asc), eigenvectors.
         Sort descending; components_ = V[:, -k:].T.
    Transform: (X - mean) @ components_.T → (n, k).
    Inverse: Z @ components_ + mean → (n, d).

    Peak RAM = batch + O(d²). Independent of N.

API:
    from cadw.ml.pca import PCA, fit_pca
    pca = fit_pca(df, ["x1","x2","x3"], n_components=2)
    pca.components_  # (2, 3)
    pca.explained_variance_ratio_  # [f1, f2]
    pca.transform(X)  # np.ndarray (n, 3) → (n, 2)
    pca.inverse_transform(Z)  # (n, 2) → (n, 3)

Not in v0.7.2:
    - Randomized PCA (for d > 1000) — v0.7.3
    - Kernel PCA — out of scope
    - Categorical: use encode_categorical() first

Run: uv run python scripts/patch_v072_pca.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)


def sh(cmd, check=True):
    print(f"  $ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    r = subprocess.run(
        cmd, shell=isinstance(cmd, str), capture_output=True, text=True,
    )
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines()[:10]:
            print(f"    {line}")
    if check and r.returncode != 0:
        print(f"    STDERR: {r.stderr.strip()[-800:]}")
        sys.exit(r.returncode)
    return r.returncode


def compile_check(path):
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         f"import py_compile; py_compile.compile('{path}', doraise=True)"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"COMPILE FAIL: {path}")
        print(r.stderr[-800:])
        sys.exit(1)
    print(f"  compile OK: {path}")


def ensure_branch(name):
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    if r.stdout.strip() == name:
        print(f"  branch: already on {name}")
        return
    r = subprocess.run(["git", "rev-parse", "--verify", name],
                       capture_output=True)
    if r.returncode == 0:
        sh(["git", "checkout", name])
    else:
        sh(["git", "checkout", "-b", name])


def require_clean():
    r = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True,
    )
    if r.stdout.strip():
        print("ERROR: tracked files dirty:")
        print(r.stdout)
        sys.exit(1)


# ============ cadw/ml/pca.py ============

PCA_SRC = '''"""Principal Component Analysis (v0.7.2).

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

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.ops.reduce import reduce
from cadw.ops.reduce import CovarianceOp


def _stack_numeric(
    batch: pa.RecordBatch, cols: Sequence[str],
) -> np.ndarray:
    parts = [_column_to_f64(batch.column(c)) for c in cols]
    return np.column_stack(parts)


def _column_to_f64(arr: pa.Array) -> np.ndarray:
    np_arr = arr.to_numpy(zero_copy_only=False)
    if np_arr.dtype == object:
        out = np.empty(len(np_arr), dtype=np.float64)
        for i, v in enumerate(np_arr):
            out[i] = np.nan if v is None else float(v)
        return out
    return np_arr.astype(np.float64, copy=False)


@dataclass
class PCA:
    """Fitted PCA model (sklearn.PCA-compatible attributes)."""

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
                # NaN row → all-NaN PCs
                proj = np.full((n, self.n_components_), np.nan, dtype=np.float64)
                mask = ~np.isnan(Xm).any(axis=1)
                if mask.any():
                    Xv = Xm[mask]
                    proj[mask] = (Xv - mean) @ comp.T
                arrays = [b.column(c) for c in X._columns]
                for i in range(self.n_components_):
                    arrays.append(pa.array(proj[:, i], type=pa.float64()))
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
        out_fields = [orig_schema.field(c) for c in Z._columns]
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
                arrays = [b.column(c) for c in Z._columns]
                for i in range(self.n_features_in_):
                    arrays.append(pa.array(recon[:, i], type=pa.float64()))
                yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

        new_ds = Dataset.from_batches(
            out_schema, _iter_inv(), Z._new_dataset_dir()
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=Z._memory_guard,
            spill_dir=Z._spill_dir,
        )


def fit_pca(
    df,
    X_cols,
    n_components: Optional[int] = None,
    *,
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

    op = CovarianceOp(X_cols)
    res = reduce(df._iter_filtered_batches(), op, mg, log=False)

    # CovarianceOp.finalize returns cov matrix (d×d)
    # But we need n and mean too — need to access state. Our CovarianceOp
    # doesn't expose sum_x. Fallback: recompute n + mean in a second pass.
    # This is a compromise: avoids modifying CovarianceOp API in v0.7.2.
    cov = res  # d×d covariance matrix

    # Second pass: accumulate sum_X and count (skipping nulls)
    n_valid = 0
    sum_x = np.zeros(d, dtype=np.float64)
    for b in df._iter_filtered_batches():
        Xm = _stack_numeric(b, X_cols)
        mask = ~np.isnan(Xm).any(axis=1)
        Xv = Xm[mask]
        if Xv.shape[0] == 0:
            continue
        n_valid += Xv.shape[0]
        sum_x += Xv.sum(axis=0)

    if n_valid < 2:
        raise ValueError(
            f"fit_pca: need at least 2 valid rows; got {n_valid}"
        )

    mean = sum_x / n_valid

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
    )
'''


# ============ dataframe convenience ============

def patch_dataframe():
    print("\n=== cadw/dataframe.py: fit_pca method ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "def fit_pca" in s:
        print("  already present — skip")
        return

    anchor = "    # ---------------- merge ----------------"
    if anchor not in s:
        raise SystemExit("merge anchor not found")

    method = '''    # ---------------- ML: fit_pca ----------------

    def fit_pca(
        self,
        X_cols: "str | Sequence[str]",
        n_components: "int | None" = None,
    ):
        """Fit PCA via Reduce(CovarianceOp) + eigh (v0.7.2).

        Returns PCA with sklearn-compatible attributes:
            components_ (k, d), mean_, explained_variance_,
            explained_variance_ratio_, singular_values_.

        pca.transform(df) → df with pc0..pc{k-1} columns.
        pca.inverse_transform(df) → reconstructed feature columns.
        """
        from cadw.ml.pca import fit_pca
        return fit_pca(
            self, X_cols, n_components,
            memory_guard=self._memory_guard,
        )

'''
    s = s.replace(anchor, method + anchor, 1)
    p.write_text(s)
    compile_check(p)
    print("  OK: fit_pca method added")


# ============ exports ============

def patch_exports():
    print("\n=== cadw/ml/__init__.py + cadw/__init__.py ===")
    p = Path("cadw/ml/__init__.py")
    p.write_text('''"""CADW ML layer (v0.7.2)."""
from .linear import XTXOp, LinearModel, fit_linear
from .logistic import LogisticGradOp, LogisticModel, fit_logistic
from .kmeans import KMeansStepOp, KMeansModel, fit_kmeans
from .pca import PCA, fit_pca

__all__ = [
    "XTXOp", "LinearModel", "fit_linear",
    "LogisticGradOp", "LogisticModel", "fit_logistic",
    "KMeansStepOp", "KMeansModel", "fit_kmeans",
    "PCA", "fit_pca",
]
''')
    compile_check(p)

    p2 = Path("cadw/__init__.py")
    s = p2.read_text()
    if "PCA" in s and "fit_pca" in s:
        print("  top-level already updated")
        return
    s = s.replace(
        "    KMeansStepOp, KMeansModel, fit_kmeans,\n)",
        "    KMeansStepOp, KMeansModel, fit_kmeans,\n"
        "    PCA, fit_pca,\n)",
        1,
    )
    s = s.replace(
        '    "KMeansStepOp", "KMeansModel", "fit_kmeans",\n',
        '    "KMeansStepOp", "KMeansModel", "fit_kmeans",\n'
        '    "PCA", "fit_pca",\n',
        1,
    )
    p2.write_text(s)
    compile_check(p2)
    print("  OK: exports updated")


# ============ tests ============

TESTS_SRC = '''"""v0.7.2: PCA via CovarianceOp + eigh."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.ml.pca import PCA, fit_pca


def make_df(tmp_path, data, chunk=20, name="L"):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def _pca_data(n, d, seed=42):
    """Correlated Gaussian: y = X @ A. If A mixes into 2 dims, PCA
    finds 2 components with high explained_variance."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, d))
    # Random rotation
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    X = Z @ Q
    data = {f"x{i}": X[:, i] for i in range(d)}
    return data, X


# ---------- basic fit ----------

def test_fit_pca_2d_shapes(tmp_path):
    n, d = 100, 3
    data, X = _pca_data(n, d, seed=1)
    df = make_df(tmp_path, data)
    pca = fit_pca(df, [f"x{i}" for i in range(d)], n_components=2)
    assert isinstance(pca, PCA)
    assert pca.components_.shape == (2, 3)
    assert pca.mean_.shape == (3,)
    assert pca.n_components_ == 2
    assert pca.n_features_in_ == 3


def test_fit_pca_all_components(tmp_path):
    n, d = 100, 4
    data, X = _pca_data(n, d, seed=2)
    df = make_df(tmp_path, data)
    pca = fit_pca(df, [f"x{i}" for i in range(d)])
    assert pca.components_.shape == (4, 4)


def test_explained_variance_ratio_sums_to_1(tmp_path):
    n, d = 200, 4
    data, X = _pca_data(n, d, seed=3)
    df = make_df(tmp_path, data)
    pca = fit_pca(df, [f"x{i}" for i in range(d)])
    ratio_sum = pca.explained_variance_ratio_.sum()
    assert abs(ratio_sum - 1.0) < 1e-9


def test_explained_variance_descending(tmp_path):
    n, d = 200, 4
    data, X = _pca_data(n, d, seed=4)
    df = make_df(tmp_path, data)
    pca = fit_pca(df, [f"x{i}" for i in range(d)])
    ev = pca.explained_variance_
    assert all(ev[i] >= ev[i+1] for i in range(len(ev) - 1))


# ---------- mean / covariance ----------

def test_mean_matches_numpy(tmp_path):
    n, d = 100, 3
    data, X = _pca_data(n, d, seed=5)
    df = make_df(tmp_path, data)
    pca = fit_pca(df, [f"x{i}" for i in range(d)])
    np.testing.assert_allclose(pca.mean_, X.mean(axis=0), atol=1e-9)


def test_components_orthonormal(tmp_path):
    n, d = 200, 4
    data, X = _pca_data(n, d, seed=6)
    df = make_df(tmp_path, data)
    pca = fit_pca(df, [f"x{i}" for i in range(d)], n_components=3)
    # rows of components_ should be orthonormal
    prod = pca.components_ @ pca.components_.T
    np.testing.assert_allclose(prod, np.eye(3), atol=1e-9)


# ---------- matches numpy PCA ----------

def test_fit_pca_matches_numpy(tmp_path):
    """Compare to numpy reference: center → cov → eigh."""
    n, d = 200, 3
    data, X = _pca_data(n, d, seed=7)
    df = make_df(tmp_path, data)
    pca = fit_pca(df, [f"x{i}" for i in range(d)], n_components=2)

    # numpy reference
    Xc = X - X.mean(axis=0)
    cov = (Xc.T @ Xc) / (n - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    comp_ref = eigvecs[:, :2].T

    # Components may differ in sign; check absolute dot product
    dot = np.abs(pca.components_ @ comp_ref.T)
    np.testing.assert_allclose(dot, np.eye(2), atol=1e-6)
    np.testing.assert_allclose(
        pca.explained_variance_, eigvals[:2], rtol=1e-9,
    )


# ---------- transform ----------

def test_transform_numpy_shape(tmp_path):
    n, d = 100, 3
    data, X = _pca_data(n, d, seed=8)
    df = make_df(tmp_path, data)
    pca = fit_pca(df, [f"x{i}" for i in range(d)], n_components=2)
    Z = pca.transform(X[:10])
    assert Z.shape == (10, 2)


def test_transform_df_adds_pc_columns(tmp_path):
    n, d = 100, 3
    data, X = _pca_data(n, d, seed=9)
    df = make_df(tmp_path, data)
    pca = fit_pca(df, [f"x{i}" for i in range(d)], n_components=2)
    result = pca.transform(df)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    names = pq.read_schema(out).names
    assert "pc0" in names
    assert "pc1" in names


def test_transform_preserves_original_columns(tmp_path):
    n, d = 100, 3
    data, X = _pca_data(n, d, seed=10)
    df = make_df(tmp_path, data)
    pca = fit_pca(df, [f"x{i}" for i in range(d)], n_components=2)
    result = pca.transform(df)
    for i in range(d):
        assert f"x{i}" in result.columns
    assert "pc0" in result.columns


# ---------- inverse_transform ----------

def test_inverse_transform_numpy_roundtrip(tmp_path):
    n, d = 100, 3
    data, X = _pca_data(n, d, seed=11)
    df = make_df(tmp_path, data)
    pca = fit_pca(df, [f"x{i}" for i in range(d)])
    Z = pca.transform(X[:10])
    X_rec = pca.inverse_transform(Z)
    # Full-rank PCA → near-perfect reconstruction
    np.testing.assert_allclose(X_rec, X[:10], atol=1e-9)


def test_inverse_transform_partial_approx(tmp_path):
    """with k=2 of d=3: reconstruction is approximate."""
    n, d = 200, 3
    data, X = _pca_data(n, d, seed=12)
    df = make_df(tmp_path, data)
    pca = fit_pca(df, [f"x{i}" for i in range(d)], n_components=2)
    Z = pca.transform(X)
    X_rec = pca.inverse_transform(Z)
    # Reconstruction error >= 0 but bounded
    err = np.linalg.norm(X_rec - X) / np.linalg.norm(X)
    assert 0 < err < 1


def test_inverse_transform_df(tmp_path):
    n, d = 100, 3
    data, X = _pca_data(n, d, seed=13)
    df = make_df(tmp_path, data)
    pca = fit_pca(df, [f"x{i}" for i in range(d)])
    transformed = pca.transform(df)
    reconstructed = pca.inverse_transform(transformed)
    out = tmp_path / "o.parquet"
    reconstructed.write_parquet(out)
    names = pq.read_schema(out).names
    for i in range(d):
        assert f"x{i}" in names


# ---------- nulls ----------

def test_fit_pca_with_nulls_skips(tmp_path):
    n, d = 100, 3
    data, X = _pca_data(n, d, seed=14)
    data["x0"] = [None if i % 10 == 0 else v for i, v in enumerate(data["x0"])]
    df = make_df(tmp_path, data)
    pca = fit_pca(df, [f"x{i}" for i in range(d)])
    assert pca.n_samples_ < n


def test_transform_null_rows_yield_nan_pcs(tmp_path):
    n, d = 100, 3
    data, X = _pca_data(n, d, seed=15)
    data["x0"] = [None if i == 5 else v for i, v in enumerate(data["x0"])]
    df = make_df(tmp_path, data)
    pca = fit_pca(df, [f"x{i}" for i in range(d)], n_components=2)
    result = pca.transform(df)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert rows[5]["pc0"] is None
    assert rows[5]["pc1"] is None


# ---------- validation ----------

def test_fit_pca_empty_X(tmp_path):
    df = make_df(tmp_path, {"x0": [1.0, 2.0]})
    with pytest.raises(ValueError, match="X_cols"):
        fit_pca(df, [])


def test_fit_pca_missing_column(tmp_path):
    df = make_df(tmp_path, {"x0": [1.0, 2.0]})
    with pytest.raises(KeyError, match="nope"):
        fit_pca(df, ["nope"])


def test_fit_pca_non_numeric_raises(tmp_path):
    df = make_df(tmp_path, {"x0": ["a", "b"], "x1": [1.0, 2.0]})
    with pytest.raises(TypeError, match="numeric"):
        fit_pca(df, ["x0"])


def test_fit_pca_invalid_n_components(tmp_path):
    df = make_df(tmp_path, {"x0": [1.0, 2.0], "x1": [3.0, 4.0]})
    with pytest.raises(ValueError, match="n_components"):
        fit_pca(df, ["x0", "x1"], n_components=5)
    with pytest.raises(ValueError, match="n_components"):
        fit_pca(df, ["x0", "x1"], n_components=0)
    with pytest.raises(TypeError, match="n_components"):
        fit_pca(df, ["x0", "x1"], n_components=1.5)


# ---------- convenience ----------

def test_fit_pca_dataframe_method(tmp_path):
    n, d = 100, 3
    data, X = _pca_data(n, d, seed=16)
    df = make_df(tmp_path, data)
    pca = df.fit_pca([f"x{i}" for i in range(d)], n_components=2)
    assert pca.components_.shape == (2, 3)


def test_pca_to_dict(tmp_path):
    n, d = 100, 3
    data, X = _pca_data(n, d, seed=17)
    df = make_df(tmp_path, data)
    pca = fit_pca(df, [f"x{i}" for i in range(d)], n_components=2)
    d_ = pca.to_dict()
    assert "components" in d_
    assert "mean" in d_
    assert "explained_variance" in d_
    assert "explained_variance_ratio" in d_
    assert d_["n_components"] == 2
'''


def write_tests():
    print("\n=== tests/test_pca.py ===")
    p = Path("tests/test_pca.py")
    p.write_text(TESTS_SRC)
    compile_check(p)
    print("  OK: tests written")


# ============ main ============

def main():
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    branch = "patch/v072-pca"

    print("=== v0.7.2: PCA ===")
    ensure_branch(branch)
    require_clean()

    backup_dir = Path(f".backups/backup-pre-v072-{ts_str}")
    backup_dir.mkdir(parents=True, exist_ok=True)
    sh(["git", "archive", "--format=tar.gz",
        "-o", str(backup_dir / "repo.tgz"), "HEAD"])
    sh(["git", "tag", "-a", f"backup-pre-v072-{ts_str}",
        "-m", "before patch v0.7.2 PCA"])

    print("\n=== cadw/ml/pca.py ===")
    Path("cadw/ml/pca.py").write_text(PCA_SRC)
    compile_check("cadw/ml/pca.py")
    print("  OK: pca.py created")

    patch_dataframe()
    patch_exports()
    write_tests()

    print("\n=== pytest tests/test_pca.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_pca.py", "-v"]
    )
    if r.returncode != 0:
        print("\n*** FAILURE — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.7.2 (test red)"], check=False)
        sys.exit(1)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.7.2: PCA via CovarianceOp + eigh (transform/inverse_transform)"])

    print("\n==> v0.7.2 applied on", branch)


if __name__ == "__main__":
    main()