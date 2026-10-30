"""v0.7.2: PCA via CovarianceOp + eigh."""
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


# v0.9.93: method="auto"|"exact"|"randomized"


def _make_pca_df(n, d, seed=42, rank=3, noise=0.1):
    """Rank-`rank` data + small Gaussian noise."""
    rng = np.random.RandomState(seed)
    U = rng.randn(n, rank)
    W = rng.randn(rank, d)
    X = U @ W + noise * rng.randn(n, d)
    X_cols = [f"x{i}" for i in range(d)]
    tbl = pa.table({
        **{c: pa.array(X[:, i]) for i, c in enumerate(X_cols)},
    })
    return CadwDataFrame.from_table(tbl), X_cols


def _recon_err_pca(model, df, X_cols):
    """Relative Frobenius reconstruction error."""
    err = 0.0
    norm = 0.0
    for b in df._iter_filtered_batches():
        X = np.column_stack([
            b.column(c).to_numpy(zero_copy_only=False) for c in X_cols])
        Xc = X - model.mean_
        proj = Xc @ model.components_.T
        recon_c = proj @ model.components_
        err += float(np.sum((Xc - recon_c) ** 2))
        norm += float(np.sum(Xc ** 2))
    return float(np.sqrt(err / max(norm, 1e-300)))


def test_pca_method_exact_vs_randomized_reconstruction():
    """method="exact" and method="randomized" give similar
    reconstruction error on well-conditioned data.

    Per §4.2: PCA has no solver (not an optimizer); use reconstruction
    error, not component comparison (components are sign/rotation-
    ambiguous; reconstruction is invariant)."""
    n, d, k = 500, 20, 5
    df, X_cols = _make_pca_df(n, d, seed=42, rank=3)
    m_exact = fit_pca(df, X_cols, n_components=k, method="exact")
    m_rand = fit_pca(df, X_cols, n_components=k, method="randomized",
                     n_oversamples=10, random_state=0)
    assert m_exact.method_ == "exact"
    assert m_rand.method_ == "randomized"
    e_exact = _recon_err_pca(m_exact, df, X_cols)
    e_rand = _recon_err_pca(m_rand, df, X_cols)
    # rank-3 data with k=5: both should capture nearly all variance
    assert e_exact < 0.20, f"exact recon err {e_exact:.4f} > 0.20"
    assert e_rand < 0.25, f"randomized recon err {e_rand:.4f} > 0.25"
    # Also close to each other
    assert abs(e_exact - e_rand) < 0.05, (
        f"recon err difference {abs(e_exact - e_rand):.4f} > 0.05")


def test_pca_method_randomized_deterministic():
    df, X_cols = _make_pca_df(300, 15, seed=7, rank=3)
    m1 = fit_pca(df, X_cols, n_components=3, method="randomized",
                 random_state=42)
    m2 = fit_pca(df, X_cols, n_components=3, method="randomized",
                 random_state=42)
    assert np.allclose(m1.components_, m2.components_)
    assert np.allclose(m1.singular_values_, m2.singular_values_)


def test_pca_method_auto_large_d_picks_randomized():
    """auto at d=2000 + 10 MB guard -> randomized."""
    n, d, k = 50, 2000, 3
    df, X_cols = _make_pca_df(n, d, seed=3, rank=2, noise=0.05)
    mg = MemoryGuard(max_memory_mb=10)
    m = fit_pca(df, X_cols, n_components=k, method="auto",
                memory_guard=mg)
    assert m.method_ == "randomized"


def test_pca_method_invalid_raises():
    df, X_cols = _make_pca_df(100, 5, seed=0)
    with pytest.raises(ValueError, match="method must be"):
        fit_pca(df, X_cols, n_components=2, method="bogus")
