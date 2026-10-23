"""Tests for D.6.8 GMM density (v0.9.31)."""
from __future__ import annotations
import tempfile

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.ml.gmm import fit_gmm, GMMModel


def _df(X):
    return CadwDataFrame.from_table(pa.table({f"x{i}": pa.array(X[:, i]) for i in range(X.shape[1])}))


def test_smoke_returns_model():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(500, 2))
    m = fit_gmm(_df(X), ["x0", "x1"], n_components=3, random_state=0)
    assert isinstance(m, GMMModel)
    assert m.weights_.shape == (3,)
    assert m.means_.shape == (3, 2)
    assert m.covariances_.shape == (3, 2)
    assert abs(m.weights_.sum() - 1.0) < 1e-6


def test_two_components_recover_centers():
    rng = np.random.RandomState(1)
    X = np.vstack([rng.normal(-3, 0.3, (300, 1)), rng.normal(3, 0.3, (300, 1))])
    m = fit_gmm(_df(X), ["x0"], n_components=2, random_state=0, max_iter=200)
    means = np.sort(m.means_.ravel())
    assert abs(means[0] - (-3)) < 0.5
    assert abs(means[1] - 3) < 0.5


def test_score_samples_finite():
    rng = np.random.RandomState(2)
    X = rng.normal(size=(200, 2))
    m = fit_gmm(_df(X), ["x0", "x1"], n_components=2, random_state=0)
    t = m.score_samples(_df(X))
    vals = t.column("log_likelihood").to_numpy(zero_copy_only=False)
    assert np.all(np.isfinite(vals))
    assert vals.shape == (200,)


def test_determinism_same_seed():
    rng = np.random.RandomState(3)
    X = rng.normal(size=(200, 2))
    a = fit_gmm(_df(X), ["x0", "x1"], n_components=2, random_state=42)
    b = fit_gmm(_df(X), ["x0", "x1"], n_components=2, random_state=42)
    assert np.allclose(a.means_, b.means_)
    assert np.allclose(a.covariances_, b.covariances_)
    assert np.allclose(a.weights_, b.weights_)


def test_predict_labels_match_argmax_density():
    rng = np.random.RandomState(4)
    X = np.vstack([rng.normal(-5, 0.1, (50, 1)), rng.normal(5, 0.1, (50, 1))])
    m = fit_gmm(_df(X), ["x0"], n_components=2, random_state=0, max_iter=200)
    labels = m.predict(_df(X)).column("cluster").to_numpy(zero_copy_only=False)
    # All -5 cluster rows must share one label, all +5 the other
    low = set(labels[:50].tolist())
    high = set(labels[50:].tolist())
    assert len(low) == 1 and len(high) == 1
    assert low != high


def test_ref_vs_sklearn_log_likelihood():
    from sklearn.mixture import GaussianMixture
    rng = np.random.RandomState(5)
    X = np.vstack([rng.normal(-3, 0.5, (500, 2)), rng.normal(3, 0.5, (500, 2))])
    cadw = fit_gmm(_df(X), ["x0", "x1"], n_components=2, random_state=0, max_iter=200)
    sk = GaussianMixture(n_components=2, covariance_type="diag", random_state=0, max_iter=200).fit(X)
    cadw_ll = cadw.score(_df(X))
    sk_ll = float(sk.score(X))
    # Different init/iteration → tolerance 1.0 nat per row
    assert abs(cadw_ll - sk_ll) < 1.0, f"{cadw_ll:.3f} vs {sk_ll:.3f}"


def test_too_few_rows_raises():
    X = np.array([[1.0]])
    with pytest.raises(ValueError):
        fit_gmm(_df(X), ["x0"], n_components=3, max_samples=10)


def test_chunk_invariance_mean_loglik():
    import pyarrow.parquet as pq
    rng = np.random.RandomState(6)
    X = rng.normal(size=(500, 2))
    tbl = pa.table({f"x{i}": pa.array(X[:, i]) for i in range(2)})
    path = tempfile.mktemp(suffix=".parquet")
    pq.write_table(tbl, path, row_group_size=1)
    df1 = CadwDataFrame.from_parquet(path, base_dir="/tmp", chunk_size=1)
    df2 = CadwDataFrame.from_parquet(path, base_dir="/tmp", chunk_size=1000)
    m1 = fit_gmm(df1, ["x0", "x1"], n_components=2, random_state=0, max_samples=500)
    m2 = fit_gmm(df2, ["x0", "x1"], n_components=2, random_state=0, max_samples=500)
    ll1 = m1.score(_df(X)); ll2 = m2.score(_df(X))
    assert abs(ll1 - ll2) < 0.5


def test_state_does_not_grow_with_n():
    rng = np.random.RandomState(7)
    X_small = rng.normal(size=(1000, 2))
    X_large = rng.normal(size=(10_000, 2))
    m_small = fit_gmm(_df(X_small), ["x0", "x1"], n_components=3, max_samples=1000, random_state=0)
    m_large = fit_gmm(_df(X_large), ["x0", "x1"], n_components=3, max_samples=1000, random_state=0)
    assert m_small.state_nbytes() == m_large.state_nbytes()


def test_unsupervised_no_y_col():
    rng = np.random.RandomState(8)
    X = rng.normal(size=(200, 2))
    m = fit_gmm(_df(X), ["x0", "x1"], n_components=2, random_state=0)
    assert m.log_likelihood_ < 0  # log-likelihood of continuous data is negative
