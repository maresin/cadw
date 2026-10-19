"""v0.7.1: k-means via iter-Reduce."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.ml.kmeans import KMeansModel, KMeansStepOp, fit_kmeans


def make_df(tmp_path, data, chunk=20, name="L"):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def _synth_kmeans(n, k_true, d=2, spread=0.5, seed=42):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k_true, d)) * 5
    labels = rng.integers(0, k_true, size=n)
    X = centers[labels] + spread * rng.normal(size=(n, d))
    data = {f"x{i}": X[:, i] for i in range(d)}
    return data, X, centers


# ---------- op ----------

def test_kmeans_step_op_estimate_state():
    op = KMeansStepOp(["a", "b", "c"], np.zeros((3, 3)))
    assert op.estimate_state_nbytes() == 8*3*3 + 8*3 + 8


def test_kmeans_step_op_state_nbytes_matches():
    op = KMeansStepOp(["a", "b"], np.zeros((3, 2)))
    assert op.state_nbytes(op.init()) <= op.estimate_state_nbytes()


# ---------- basic fit ----------

def test_fit_kmeans_two_clusters(tmp_path):
    n = 500
    data, X, centers = _synth_kmeans(n, k_true=2, d=2, seed=1)
    df = make_df(tmp_path, data)
    model = fit_kmeans(df, ["x0", "x1"], k=2, n_init=1, seed=42)
    assert isinstance(model, KMeansModel)
    assert model.cluster_centers_.shape == (2, 2)
    assert model.inertia_ > 0
    assert model.converged_


def test_fit_kmeans_three_clusters(tmp_path):
    n = 500
    data, X, centers = _synth_kmeans(n, k_true=3, d=2, seed=2)
    df = make_df(tmp_path, data)
    model = fit_kmeans(df, ["x0", "x1"], k=3, n_init=3, seed=42)
    # Recovered centers should be close to true (up to permutation)
    from itertools import permutations
    best = min(
        float(np.linalg.norm(model.cluster_centers_ - np.array(p)))
        for p in permutations(centers)
    )
    assert best < 2.0, f"permutation error={best:.3f}"


def test_fit_kmeans_high_dimensional(tmp_path):
    n = 300
    data, X, centers = _synth_kmeans(n, k_true=3, d=5, seed=3)
    df = make_df(tmp_path, data)
    model = fit_kmeans(df, [f"x{i}" for i in range(5)], k=3, n_init=2)
    assert model.cluster_centers_.shape == (3, 5)
    assert model.converged_


def test_fit_kmeans_k1(tmp_path):
    n = 100
    data, X, _ = _synth_kmeans(n, k_true=1, d=2, seed=4)
    df = make_df(tmp_path, data)
    model = fit_kmeans(df, ["x0", "x1"], k=1)
    assert model.cluster_centers_.shape == (1, 2)
    # Cluster center = mean of all points
    np.testing.assert_allclose(
        model.cluster_centers_[0], X.mean(axis=0), atol=1e-6,
    )


def test_fit_kmeans_n_init_picks_best(tmp_path):
    n = 300
    data, X, _ = _synth_kmeans(n, k_true=3, d=2, seed=5)
    df = make_df(tmp_path, data)
    m1 = fit_kmeans(df, ["x0", "x1"], k=3, n_init=1, seed=42)
    m5 = fit_kmeans(df, ["x0", "x1"], k=3, n_init=5, seed=42)
    # More inits: inertia <= single init (never worse)
    assert m5.inertia_ <= m1.inertia_ + 1e-6


def test_fit_kmeans_deterministic_with_seed(tmp_path):
    n = 300
    data, X, _ = _synth_kmeans(n, k_true=3, d=2, seed=6)
    df = make_df(tmp_path, data)
    m1 = fit_kmeans(df, ["x0", "x1"], k=3, n_init=2, seed=42)
    m2 = fit_kmeans(df, ["x0", "x1"], k=3, n_init=2, seed=42)
    np.testing.assert_allclose(
        m1.cluster_centers_, m2.cluster_centers_, rtol=1e-9,
    )


# ---------- predict ----------

def test_predict_numpy(tmp_path):
    n = 200
    data, X, _ = _synth_kmeans(n, k_true=2, d=2, seed=7)
    df = make_df(tmp_path, data)
    model = fit_kmeans(df, ["x0", "x1"], k=2, seed=42)
    labels = model.predict(X[:10])
    assert labels.shape == (10,)
    assert set(labels.tolist()) <= {0, 1}


def test_predict_df_adds_cluster_column(tmp_path):
    n = 200
    data, X, _ = _synth_kmeans(n, k_true=3, d=2, seed=8)
    df = make_df(tmp_path, data)
    model = fit_kmeans(df, ["x0", "x1"], k=3, seed=42)
    result = model.predict(df)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    names = pq.read_schema(out).names
    assert "cluster" in names


def test_predict_null_rows_marked(tmp_path):
    n = 100
    data, X, _ = _synth_kmeans(n, k_true=2, d=2, seed=9)
    data["x0"] = [None if i == 5 else v for i, v in enumerate(data["x0"])]
    df = make_df(tmp_path, data)
    model = fit_kmeans(df, ["x0", "x1"], k=2)
    result = model.predict(df)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert rows[5]["cluster"] is None


def test_predict_consistent_with_fit(tmp_path):
    n = 300
    data, X, _ = _synth_kmeans(n, k_true=3, d=2, seed=10)
    df = make_df(tmp_path, data)
    model = fit_kmeans(df, ["x0", "x1"], k=3, seed=42)
    # Predict on same data — all labels should be non-null
    result = model.predict(df)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    t = pq.read_table(out)
    assert t.column("cluster").null_count == 0


# ---------- with nulls in fit ----------

def test_fit_with_nulls_skips_rows(tmp_path):
    n = 200
    data, X, _ = _synth_kmeans(n, k_true=2, d=2, seed=11)
    data["x0"] = [None if i % 10 == 0 else v for i, v in enumerate(data["x0"])]
    df = make_df(tmp_path, data)
    model = fit_kmeans(df, ["x0", "x1"], k=2)
    assert model.n_rows_ < n
    assert model.n_rows_ > 0


# ---------- validation ----------

def test_fit_kmeans_invalid_k(tmp_path):
    df = make_df(tmp_path, {"x0": [1.0, 2.0], "x1": [3.0, 4.0]})
    with pytest.raises(ValueError, match="k must be >= 1"):
        fit_kmeans(df, ["x0", "x1"], k=0)
    with pytest.raises(TypeError, match="k must be int"):
        fit_kmeans(df, ["x0", "x1"], k=1.5)


def test_fit_kmeans_missing_column(tmp_path):
    df = make_df(tmp_path, {"x0": [1.0], "x1": [2.0]})
    with pytest.raises(KeyError, match="nope"):
        fit_kmeans(df, ["nope"], k=1)


def test_fit_kmeans_non_numeric_raises(tmp_path):
    df = make_df(tmp_path, {"x0": ["a", "b"], "x1": [1.0, 2.0]})
    with pytest.raises(TypeError, match="numeric"):
        fit_kmeans(df, ["x0", "x1"], k=1)


def test_fit_kmeans_k_gt_n(tmp_path):
    df = make_df(tmp_path, {"x0": [1.0, 2.0], "x1": [3.0, 4.0]})
    with pytest.raises(ValueError, match="k=.*>.*available"):
        fit_kmeans(df, ["x0", "x1"], k=10)


# ---------- convenience ----------

def test_fit_kmeans_dataframe_method(tmp_path):
    n = 200
    data, X, _ = _synth_kmeans(n, k_true=2, d=2, seed=12)
    df = make_df(tmp_path, data)
    model = df.fit_kmeans(["x0", "x1"], k=2)
    assert hasattr(model, "cluster_centers_")


def test_kmeans_model_to_dict(tmp_path):
    n = 100
    data, X, _ = _synth_kmeans(n, k_true=2, d=2, seed=13)
    df = make_df(tmp_path, data)
    model = fit_kmeans(df, ["x0", "x1"], k=2)
    d = model.to_dict()
    assert "cluster_centers" in d
    assert "inertia" in d
    assert "n_iter" in d
    assert "converged" in d
