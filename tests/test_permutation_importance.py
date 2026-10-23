"""Tests for D.6.3 permutation importance (v0.9.31)."""
from __future__ import annotations
import tempfile

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.ml.importance import permutation_importance


class _LinearModel:
    def __init__(self, weights):
        self.w = np.asarray(weights, dtype=np.float64)
    def predict(self, X):
        return X @ self.w


def _df(X, y=None):
    cols = {f"x{i}": pa.array(X[:, i], type=pa.float64()) for i in range(X.shape[1])}
    if y is not None:
        cols["y"] = pa.array(y, type=pa.float64())
    return CadwDataFrame.from_table(pa.table(cols))


def test_smoke_returns_table():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(200, 3))
    y = 2*X[:,0] - 3*X[:,1] + 0.01*rng.normal(size=200)
    m = _LinearModel([2, -3, 0])
    imp = permutation_importance(m, _df(X, y), ["x0","x1","x2"], y_col="y",
                                 n_repeats=2, max_samples=200, random_state=0)
    assert imp.column_names == ["feature", "importance_mean", "importance_std"]
    assert imp.num_rows == 3


def test_relevant_feature_outranks_noise():
    rng = np.random.RandomState(1)
    X = rng.normal(size=(300, 2))
    y = 3*X[:,0] + 0.01*rng.normal(size=300)
    m = _LinearModel([3, 0])
    imp = permutation_importance(m, _df(X, y), ["x0","x1"], y_col="y",
                                 n_repeats=3, max_samples=300, random_state=0)
    feats = imp.column("feature").to_pylist()
    means = imp.column("importance_mean").to_pylist()
    d = dict(zip(feats, means))
    assert d["x0"] > d["x1"] + 0.1


def test_ranking_stable_across_repeats():
    rng = np.random.RandomState(2)
    X = rng.normal(size=(200, 3))
    y = 5*X[:,0] + X[:,1] + 0.01*rng.normal(size=200)
    m = _LinearModel([5, 1, 0])
    a = permutation_importance(m, _df(X, y), ["x0","x1","x2"], y_col="y",
                               n_repeats=1, max_samples=200, random_state=0)
    b = permutation_importance(m, _df(X, y), ["x0","x1","x2"], y_col="y",
                               n_repeats=5, max_samples=200, random_state=0)
    ra = list(np.argsort(-np.array(a.column("importance_mean").to_pylist())))
    rb = list(np.argsort(-np.array(b.column("importance_mean").to_pylist())))
    assert ra == rb


def test_determinism_same_seed():
    rng = np.random.RandomState(3)
    X = rng.normal(size=(200, 2))
    y = X[:,0] + 0.1*rng.normal(size=200)
    m = _LinearModel([1, 0])
    a = permutation_importance(m, _df(X, y), ["x0","x1"], y_col="y", n_repeats=3, max_samples=100, random_state=11)
    b = permutation_importance(m, _df(X, y), ["x0","x1"], y_col="y", n_repeats=3, max_samples=100, random_state=11)
    assert a.column("importance_mean").to_pylist() == b.column("importance_mean").to_pylist()


def test_too_small_sample_raises():
    X = np.array([[1.0]])
    m = _LinearModel([1])
    with pytest.raises(ValueError):
        permutation_importance(m, _df(X), ["x0"], max_samples=1)


def test_constant_feature_zero_importance():
    rng = np.random.RandomState(4)
    X = np.column_stack([np.zeros(100), rng.normal(size=100)])
    y = X[:,1]
    m = _LinearModel([0, 1])
    imp = permutation_importance(m, _df(X, y), ["x0","x1"], y_col="y",
                                 n_repeats=2, max_samples=100, random_state=0)
    means = dict(zip(imp.column("feature").to_pylist(),
                     imp.column("importance_mean").to_pylist()))
    assert abs(means["x0"]) < 0.01


def test_ref_vs_sklearn_spearman():
    from sklearn.inspection import permutation_importance as sk_imp
    from scipy.stats import spearmanr
    from sklearn.linear_model import LinearRegression
    rng = np.random.RandomState(5)
    X = rng.normal(size=(400, 4))
    y = 3*X[:,0] + 2*X[:,1] + 0.1*rng.normal(size=400)
    sk = LinearRegression().fit(X, y)
    m = _LinearModel(np.concatenate([sk.coef_, [sk.intercept_]]) if False else sk.coef_)
    # sklearn prediction includes intercept; our _LinearModel does not.
    class _WithIntercept:
        def __init__(self, coef, b): self.coef=coef; self.b=b
        def predict(self, X): return X @ self.coef + self.b
    m = _WithIntercept(sk.coef_, sk.intercept_)
    a = permutation_importance(m, _df(X, y), [f"x{i}" for i in range(4)],
                               y_col="y", n_repeats=3, max_samples=400, random_state=0)
    b = sk_imp(sk, X, y, n_repeats=3, random_state=0, scoring="r2")
    rho, _ = spearmanr(a.column("importance_mean").to_pylist(), b.importances_mean)
    assert rho >= 0.7, f"Spearman {rho:.3f} < 0.7"


def test_chunk_invariance_ranking(tmp_path):
    import pyarrow.parquet as pq
    rng = np.random.RandomState(6)
    X = rng.normal(size=(300, 3))
    y = 4*X[:,0] + X[:,1]
    tbl = pa.table({**{f"x{i}": pa.array(X[:,i]) for i in range(3)}, "y": pa.array(y)})
    # v0.9.32 (executor): shared tempfile.mktemp + hardcoded /tmp caused
    # cross-worker collisions under xdist. Isolated per-test via tmp_path.
    path = str(tmp_path / "chunk_inv.parquet")
    pq.write_table(tbl, path, row_group_size=1)
    df1 = CadwDataFrame.from_parquet(path, base_dir=str(tmp_path), chunk_size=1)
    df2 = CadwDataFrame.from_parquet(path, base_dir=str(tmp_path), chunk_size=1000)
    m = _LinearModel([4, 1, 0])
    a = permutation_importance(m, df1, ["x0","x1","x2"], y_col="y",
                               n_repeats=2, max_samples=300, random_state=0)
    b = permutation_importance(m, df2, ["x0","x1","x2"], y_col="y",
                               n_repeats=2, max_samples=300, random_state=0)
    ra = list(np.argsort(-np.array(a.column("importance_mean").to_pylist())))
    rb = list(np.argsort(-np.array(b.column("importance_mean").to_pylist())))
    assert ra == rb


def test_accuracy_metric_classifier():
    class _Clf:
        def predict(self, X):
            return (X[:, 0] > 0).astype(np.float64)
    rng = np.random.RandomState(7)
    X = rng.normal(size=(200, 2))
    y = (X[:,0] > 0).astype(np.float64)
    imp = permutation_importance(_Clf(), _df(X, y), ["x0","x1"], y_col="y",
                                 n_repeats=2, max_samples=200, metric="accuracy",
                                 random_state=0)
    means = dict(zip(imp.column("feature").to_pylist(), imp.column("importance_mean").to_pylist()))
    assert means["x0"] > means["x1"]


def test_unsupervised_shift_mode():
    class _KMeansLike:
        def predict(self, X): return np.linalg.norm(X, axis=1)
    rng = np.random.RandomState(8)
    X = rng.normal(size=(100, 2))
    imp = permutation_importance(_KMeansLike(), _df(X), ["x0","x1"],
                                 n_repeats=2, max_samples=100, random_state=0)
    assert imp.num_rows == 2
    assert all(np.isfinite(v) for v in imp.column("importance_mean").to_pylist())
