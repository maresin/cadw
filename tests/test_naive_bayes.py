"""v0.7.6: Naive Bayes — Gaussian, Multinomial, Bernoulli."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.ml.naive_bayes import (
    GaussianNB, MultinomialNB, BernoulliNB,
    fit_gaussian_nb, fit_multinomial_nb, fit_bernoulli_nb,
)


def make_df(tmp_path, data, chunk=50, name="L"):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


# ============ Gaussian NB ============

def _gauss_data(n_per_class=200, seed=42):
    rng = np.random.default_rng(seed)
    # Two classes with distinct means
    X0 = rng.normal(loc=[-2.0, -2.0], scale=1.0, size=(n_per_class, 2))
    X1 = rng.normal(loc=[2.0, 2.0], scale=1.0, size=(n_per_class, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * n_per_class + [1] * n_per_class, dtype=np.float64)
    return {"x0": X[:, 0].tolist(), "x1": X[:, 1].tolist(), "y": y.tolist()}, X, y


def test_gaussian_nb_basic(tmp_path):
    data, X, y = _gauss_data()
    df = make_df(tmp_path, data)
    model = fit_gaussian_nb(df, ["x0", "x1"], "y")
    assert isinstance(model, GaussianNB)
    assert model.classes_.tolist() == [0.0, 1.0]
    assert model.theta_.shape == (2, 2)


def test_gaussian_nb_predict_accuracy(tmp_path):
    data, X, y = _gauss_data(n_per_class=300)
    df = make_df(tmp_path, data)
    model = fit_gaussian_nb(df, ["x0", "x1"], "y")
    pred = model.predict(X)
    acc = (pred == y).mean()
    assert acc > 0.9, f"accuracy={acc:.3f}"


def test_gaussian_nb_predict_proba_range(tmp_path):
    data, X, y = _gauss_data()
    df = make_df(tmp_path, data)
    model = fit_gaussian_nb(df, ["x0", "x1"], "y")
    proba = model.predict_proba(X[:10])
    assert proba.shape == (10, 2)
    assert (proba >= 0).all() and (proba <= 1).all()
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_gaussian_nb_multiclass(tmp_path):
    rng = np.random.default_rng(0)
    n = 100
    data = {"x0": [], "x1": [], "y": []}
    for cls, center in enumerate([[-3.0, 0.0], [0.0, 0.0], [3.0, 0.0]]):
        X = rng.normal(loc=center, scale=0.5, size=(n, 2))
        data["x0"].extend(X[:, 0].tolist())
        data["x1"].extend(X[:, 1].tolist())
        data["y"].extend([float(cls)] * n)
    df = make_df(tmp_path, data)
    model = fit_gaussian_nb(df, ["x0", "x1"], "y")
    assert len(model.classes_) == 3
    assert model.theta_.shape == (3, 2)


def test_gaussian_nb_predict_df(tmp_path):
    data, X, y = _gauss_data()
    df = make_df(tmp_path, data)
    model = fit_gaussian_nb(df, ["x0", "x1"], "y")
    result = model.predict(df)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    names = pq.read_schema(out).names
    assert "y_pred" in names


def test_gaussian_nb_predict_proba_df_adds_columns(tmp_path):
    data, X, y = _gauss_data()
    df = make_df(tmp_path, data)
    model = fit_gaussian_nb(df, ["x0", "x1"], "y")
    result = model.predict_proba(df)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    names = pq.read_schema(out).names
    assert "y_prob_0" in names
    assert "y_prob_1" in names


def test_gaussian_nb_with_nulls(tmp_path):
    data, X, y = _gauss_data()
    data["x0"][5] = None
    df = make_df(tmp_path, data)
    model = fit_gaussian_nb(df, ["x0", "x1"], "y")
    result = model.predict(df)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert rows[5]["y_pred"] is None


def test_gaussian_nb_var_smoothing(tmp_path):
    """Constant feature (var=0) must not divide by zero."""
    data = {"x0": [1.0, 1.0, 2.0, 2.0], "x1": [3.0, 4.0, 3.0, 4.0], "y": [0.0, 0.0, 1.0, 1.0]}
    df = make_df(tmp_path, data)
    model = fit_gaussian_nb(df, ["x0", "x1"], "y", var_smoothing=1e-6)
    # Predict with constant x0
    pred = model.predict(np.array([[1.0, 3.0], [2.0, 3.0]]))
    assert len(pred) == 2
    assert not np.any(np.isnan(pred))


def test_gaussian_nb_validation_empty_X(tmp_path):
    df = make_df(tmp_path, {"x0": [1.0], "y": [0.0]})
    with pytest.raises(ValueError, match="X_cols"):
        fit_gaussian_nb(df, [], "y")


def test_gaussian_nb_validation_target_in_X(tmp_path):
    df = make_df(tmp_path, {"x0": [1.0], "y": [0.0]})
    with pytest.raises(ValueError, match="must not be in X_cols"):
        fit_gaussian_nb(df, ["x0", "y"], "y")


def test_gaussian_nb_validation_non_numeric(tmp_path):
    df = make_df(tmp_path, {"x0": ["a", "b"], "y": [0.0, 1.0]})
    with pytest.raises(TypeError, match="numeric"):
        fit_gaussian_nb(df, ["x0"], "y")


def test_gaussian_nb_dataframe_method(tmp_path):
    data, X, y = _gauss_data()
    df = make_df(tmp_path, data)
    model = df.fit_gaussian_nb(["x0", "x1"], "y")
    assert isinstance(model, GaussianNB)


def test_gaussian_nb_to_dict(tmp_path):
    data, X, y = _gauss_data()
    df = make_df(tmp_path, data)
    model = fit_gaussian_nb(df, ["x0", "x1"], "y")
    d = model.to_dict()
    assert d["variant"] == "gaussian"
    assert "classes" in d and "theta" in d and "var" in d


# ============ Multinomial NB ============

def _counts_data(n_per_class=100, seed=42):
    rng = np.random.default_rng(seed)
    # Class 0: high x0, low x1; Class 1: low x0, high x1
    X0 = rng.poisson(lam=[10, 1], size=(n_per_class, 2)).astype(np.float64)
    X1 = rng.poisson(lam=[1, 10], size=(n_per_class, 2)).astype(np.float64)
    X = np.vstack([X0, X1])
    y = np.array([0.0] * n_per_class + [1.0] * n_per_class)
    return {"x0": X[:, 0].tolist(), "x1": X[:, 1].tolist(), "y": y.tolist()}, X, y


def test_multinomial_nb_basic(tmp_path):
    data, X, y = _counts_data()
    df = make_df(tmp_path, data)
    model = fit_multinomial_nb(df, ["x0", "x1"], "y")
    assert isinstance(model, MultinomialNB)
    assert model.feature_log_prob_.shape == (2, 2)


def test_multinomial_nb_accuracy(tmp_path):
    data, X, y = _counts_data(n_per_class=150)
    df = make_df(tmp_path, data)
    model = fit_multinomial_nb(df, ["x0", "x1"], "y")
    pred = model.predict(X)
    acc = (pred == y).mean()
    assert acc > 0.9, f"accuracy={acc:.3f}"


def test_multinomial_nb_predict_proba(tmp_path):
    data, X, y = _counts_data()
    df = make_df(tmp_path, data)
    model = fit_multinomial_nb(df, ["x0", "x1"], "y")
    proba = model.predict_proba(X[:5])
    assert proba.shape == (5, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_multinomial_nb_alpha(tmp_path):
    """alpha=0 vs alpha=1: both work (alpha=0 may warn on zero counts)."""
    data, X, y = _counts_data()
    df = make_df(tmp_path, data)
    model = fit_multinomial_nb(df, ["x0", "x1"], "y", alpha=1.0)
    assert model.alpha_ == 1.0


def test_multinomial_nb_negative_alpha_raises(tmp_path):
    df = make_df(tmp_path, {"x0": [1.0], "x1": [0.0], "y": [0.0]})
    with pytest.raises(ValueError, match="alpha"):
        fit_multinomial_nb(df, ["x0", "x1"], "y", alpha=-0.1)


def test_multinomial_nb_predict_df(tmp_path):
    data, X, y = _counts_data()
    df = make_df(tmp_path, data)
    model = fit_multinomial_nb(df, ["x0", "x1"], "y")
    result = model.predict(df)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    names = pq.read_schema(out).names
    assert "y_pred" in names


# ============ Bernoulli NB ============

def _binary_data(n_per_class=150, seed=42, p0=0.1, p1=0.9):
    """Binary features: P(x_j=1|c=0)=p0, P(x_j=1|c=1)=p1.

    Bayes-rate = E[max(P(c|x), 1-P(c|x))] depends on (p0, p1, d=3):
        p0=0.2, p1=0.8  → Bayes-rate ≈ 0.896 (n=300 acc=0.878 — below 0.9)
        p0=0.1, p1=0.9  → Bayes-rate ≈ 0.972 (safe for acc > 0.9)
    v0.7.6-fix: use p0=0.1, p1=0.9 (same principle as v0.7.5 accuracy
    fix — strengthen signal, not lower threshold).
    """
    rng = np.random.default_rng(seed)
    X0 = (rng.random((n_per_class, 3)) < p0).astype(np.float64)
    X1 = (rng.random((n_per_class, 3)) < p1).astype(np.float64)
    X = np.vstack([X0, X1])
    y = np.array([0.0] * n_per_class + [1.0] * n_per_class)
    return {
        "x0": X[:, 0].tolist(), "x1": X[:, 1].tolist(),
        "x2": X[:, 2].tolist(), "y": y.tolist(),
    }, X, y


def test_bernoulli_nb_basic(tmp_path):
    data, X, y = _binary_data()
    df = make_df(tmp_path, data)
    model = fit_bernoulli_nb(df, ["x0", "x1", "x2"], "y")
    assert isinstance(model, BernoulliNB)
    assert model.feature_log_prob_.shape == (2, 3)


def test_bernoulli_nb_accuracy(tmp_path):
    data, X, y = _binary_data(n_per_class=300)
    df = make_df(tmp_path, data)
    model = fit_bernoulli_nb(df, ["x0", "x1", "x2"], "y")
    pred = model.predict(X)
    acc = (pred == y).mean()
    assert acc > 0.9, f"accuracy={acc:.3f}"


def test_bernoulli_nb_predict_proba(tmp_path):
    data, X, y = _binary_data()
    df = make_df(tmp_path, data)
    model = fit_bernoulli_nb(df, ["x0", "x1", "x2"], "y")
    proba = model.predict_proba(X[:5])
    assert proba.shape == (5, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_bernoulli_nb_predict_df(tmp_path):
    data, X, y = _binary_data()
    df = make_df(tmp_path, data)
    model = fit_bernoulli_nb(df, ["x0", "x1", "x2"], "y")
    result = model.predict_proba(df)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    names = pq.read_schema(out).names
    assert "y_prob_0" in names
    assert "y_prob_1" in names


# ============ unified entry point ============

def test_fit_naive_bayes_variant_gaussian(tmp_path):
    data, X, y = _gauss_data()
    df = make_df(tmp_path, data)
    model = df.fit_naive_bayes(["x0", "x1"], "y", variant="gaussian")
    assert isinstance(model, GaussianNB)


def test_fit_naive_bayes_variant_multinomial(tmp_path):
    data, X, y = _counts_data()
    df = make_df(tmp_path, data)
    model = df.fit_naive_bayes(["x0", "x1"], "y", variant="multinomial")
    assert isinstance(model, MultinomialNB)


def test_fit_naive_bayes_variant_bernoulli(tmp_path):
    data, X, y = _binary_data()
    df = make_df(tmp_path, data)
    model = df.fit_naive_bayes(["x0", "x1", "x2"], "y", variant="bernoulli")
    assert isinstance(model, BernoulliNB)


def test_fit_naive_bayes_invalid_variant(tmp_path):
    data, X, y = _gauss_data()
    df = make_df(tmp_path, data)
    with pytest.raises(ValueError, match="variant"):
        df.fit_naive_bayes(["x0", "x1"], "y", variant="unknown")

def test_multinomial_nb_negative_X_raises(tmp_path):
    """Multinomial NB: X < 0 → ValueError (counts must be >= 0).

    Без валидации: feature_sum + alpha может уйти в отрицательное
    → log(negative) = nan → молчаливые nan предсказания.
    """
    data = {"x0": [-1.0, 2.0, 3.0], "x1": [0.0, 1.0, 0.0], "y": [0.0, 0.0, 1.0]}
    df = make_df(tmp_path, data)
    with pytest.raises(ValueError, match="must be >= 0"):
        fit_multinomial_nb(df, ["x0", "x1"], "y")


def test_multinomial_nb_non_negative_X_passes(tmp_path):
    """All-zero X — валидно (sparse counts)."""
    data = {"x0": [0.0, 0.0, 0.0], "x1": [0.0, 0.0, 0.0], "y": [0.0, 0.0, 1.0]}
    df = make_df(tmp_path, data)
    model = fit_multinomial_nb(df, ["x0", "x1"], "y")
    assert model is not None



# ============ v0.9.22 (§13.4/242): NB C_ASSUMED=64 ============


def test_nb_c_assumed_is_64():
    """v0.9.22: C_ASSUMED bumped 32→64 (AUDIT_ML §1.3).

    Drift is bounded (K=100 gives 3.125×, RSS=2MB — not RAM bug);
    bump + doc is honest fix, ValueError too aggressive.
    """
    import inspect
    from cadw.ml import naive_bayes as nb_mod
    for op_name in ("_GaussianStatsOp", "_MultinomialStatsOp",
                    "_BernoulliStatsOp"):
        src = inspect.getsource(getattr(nb_mod, op_name))
        assert "return 64 * (8 +" in src, (
            f"{op_name}: expected C_ASSUMED=64 (return 64 * ...)"
        )
        assert "return 32 * (8 +" not in src, (
            f"{op_name}: old C_ASSUMED=32 must be removed"
        )
