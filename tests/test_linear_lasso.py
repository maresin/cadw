"""v0.7.7: Lasso / ElasticNet via coordinate descent on XTXOp state."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import (
    CadwDataFrame, MemoryGuard, fit_linear,
    fit_lasso, fit_elasticnet,
)
from cadw.ml.linear import LinearModel, _soft_threshold


def make_df(tmp_path, data, chunk=50, name="L"):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def _synth(n, coefs, intercept, noise=0.0, seed=42):
    rng = np.random.default_rng(seed)
    d = len(coefs)
    X = rng.normal(size=(n, d))
    y = X @ np.array(coefs) + intercept + noise * rng.normal(size=n)
    data = {f"x{i}": X[:, i] for i in range(d)}
    data["y"] = y
    return data, X, y


def _ref_lasso_closed_form_d1(x, y, lambda_1):
    """Analytical solution for d=1 Lasso (centered)."""
    x_c = x - x.mean()
    y_c = y - y.mean()
    xtx_c = float(np.dot(x_c, x_c))
    xty_c = float(np.dot(x_c, y_c))
    if xtx_c == 0:
        return 0.0, float(y.mean())
    b = _soft_threshold(xty_c, lambda_1) / xtx_c
    b0 = float(y.mean() - b * x.mean())
    return float(b), b0


# ---------- soft threshold ----------

def test_soft_threshold_basic():
    assert _soft_threshold(5.0, 2.0) == 3.0
    assert _soft_threshold(-5.0, 2.0) == -3.0
    assert _soft_threshold(1.5, 2.0) == 0.0
    assert _soft_threshold(-1.5, 2.0) == 0.0
    assert _soft_threshold(0.0, 2.0) == 0.0
    assert _soft_threshold(5.0, 0.0) == 5.0
    assert _soft_threshold(-5.0, -1.0) == -5.0


# ---------- Lasso: analytical d=1 ----------

def test_lasso_d1_matches_analytical(tmp_path):
    """d=1 Lasso has closed form: b = S(xy, λ) / xx."""
    n = 200
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    y = 2.0 * x + 0.1 * rng.normal(size=n)
    data = {"x": x.tolist(), "y": y.tolist()}
    df = make_df(tmp_path, data)
    alpha = 5.0
    model = fit_lasso(df, ["x"], "y", alpha=alpha, max_iter=5000, tol=1e-10)
    b_exp, b0_exp = _ref_lasso_closed_form_d1(x, y, alpha)
    assert model.coef_[0] == pytest.approx(b_exp, rel=1e-6, abs=1e-7)
    assert model.intercept_ == pytest.approx(b0_exp, rel=1e-6, abs=1e-7)


def test_lasso_d1_zero_coef_when_alpha_large(tmp_path):
    n = 100
    rng = np.random.default_rng(1)
    x = rng.normal(size=n)
    y = 1.0 * x + 0.1 * rng.normal(size=n)
    df = make_df(tmp_path, {"x": x.tolist(), "y": y.tolist()})
    model = fit_lasso(df, ["x"], "y", alpha=1e6)
    assert model.coef_[0] == 0.0
    assert model.intercept_ == pytest.approx(y.mean(), rel=1e-9)


# ---------- Lasso: sparsity ----------

def test_lasso_zeroes_irrelevant_features(tmp_path):
    """y = 2·x0 + 0·x1 + 0·x2 (no noise) — Lasso zeroes x1, x2."""
    n = 500
    data, X, y = _synth(n, [2.0, 0.0, 0.0], intercept=1.0)
    df = make_df(tmp_path, data, chunk=100)
    model = fit_lasso(df, ["x0", "x1", "x2"], "y", alpha=0.1)
    assert abs(model.coef_[0] - 2.0) < 0.05
    assert abs(model.coef_[1]) < 0.05
    assert abs(model.coef_[2]) < 0.05
    assert abs(model.intercept_ - 1.0) < 0.05


def test_lasso_keeps_only_one_of_duplicated(tmp_path):
    """Perfectly correlated features — Lasso picks one."""
    n = 300
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    y = 2.0 * x
    df = make_df(tmp_path, {
        "a": x.tolist(), "b": (x * 1.0).tolist(), "y": y.tolist(),
    })
    model = fit_lasso(df, ["a", "b"], "y", alpha=0.5)
    # Exactly one of coef_[0], coef_[1] is non-zero (up to tolerance)
    nonzero = [abs(c) > 0.01 for c in model.coef_]
    assert sum(nonzero) == 1, f"coef={model.coef_}"


# ---------- Ridge (penalty='l2') backward compat ----------

def test_ridge_penalty_l2_matches_fit_linear(tmp_path):
    """penalty='l2' = existing fit_linear(alpha) path (no API change)."""
    n = 200
    data, X, y = _synth(n, [2.0, -1.0], intercept=0.5, noise=0.1)
    df = make_df(tmp_path, data)
    m1 = fit_linear(df, ["x0", "x1"], "y", alpha=1.0)
    m2 = fit_linear(df, ["x0", "x1"], "y", alpha=1.0, penalty="l2")
    np.testing.assert_array_equal(m1.coef_, m2.coef_)
    assert m1.intercept_ == m2.intercept_
    assert m2.penalty_ == "l2"
    assert m2.l1_ratio_ == 0.0
    assert m2.n_iter_cd_ == 0


# ---------- ElasticNet equivalence ----------

def test_elasticnet_l1_ratio_0_equals_ridge(tmp_path):
    """ElasticNet(l1_ratio=0) == Ridge(penalty='l2')."""
    n = 300
    data, X, y = _synth(n, [2.0, -1.0], intercept=0.5, noise=0.1)
    df = make_df(tmp_path, data)
    ridge = fit_linear(df, ["x0", "x1"], "y", alpha=2.0, penalty="l2")
    enet = fit_linear(
        df, ["x0", "x1"], "y",
        alpha=2.0, penalty="elasticnet", l1_ratio=0.0,
        max_iter=5000, tol=1e-8,
    )
    np.testing.assert_allclose(enet.coef_, ridge.coef_, rtol=1e-4, atol=1e-5)
    assert enet.intercept_ == pytest.approx(ridge.intercept_, rel=1e-4)


def test_elasticnet_l1_ratio_1_equals_lasso(tmp_path):
    """ElasticNet(l1_ratio=1) == Lasso(penalty='l1')."""
    n = 300
    data, X, y = _synth(n, [2.0, 0.0], intercept=0.0, noise=0.1)
    df = make_df(tmp_path, data)
    lasso = fit_lasso(df, ["x0", "x1"], "y", alpha=0.5, max_iter=5000, tol=1e-8)
    enet = fit_elasticnet(df, ["x0", "x1"], "y", alpha=0.5, l1_ratio=1.0,
                          max_iter=5000, tol=1e-8)
    np.testing.assert_allclose(enet.coef_, lasso.coef_, rtol=1e-6, atol=1e-7)


# ---------- large alpha → all zeros ----------

def test_lasso_large_alpha_all_zero(tmp_path):
    n = 200
    data, X, y = _synth(n, [1.0, 2.0, 3.0], intercept=0.5, noise=0.1)
    df = make_df(tmp_path, data)
    model = fit_lasso(df, ["x0", "x1", "x2"], "y", alpha=1e9)
    assert np.allclose(model.coef_, 0.0, atol=1e-6)
    assert model.intercept_ == pytest.approx(y.mean(), rel=1e-6)


# ---------- deterministic / multibatch ----------

def test_lasso_deterministic(tmp_path):
    n = 300
    data, X, y = _synth(n, [1.0, -1.0], intercept=0.5, noise=0.1)
    df = make_df(tmp_path, data)
    m1 = fit_lasso(df, ["x0", "x1"], "y", alpha=0.5)
    m2 = fit_lasso(df, ["x0", "x1"], "y", alpha=0.5)
    np.testing.assert_array_equal(m1.coef_, m2.coef_)
    assert m1.intercept_ == m2.intercept_


def test_lasso_multibatch_consistent(tmp_path):
    n = 300
    data, X, y = _synth(n, [1.0, -1.0], intercept=0.5, noise=0.1)
    df1 = make_df(tmp_path, data, chunk=10, name="a")
    df2 = make_df(tmp_path, data, chunk=100, name="b")
    m1 = fit_lasso(df1, ["x0", "x1"], "y", alpha=0.5)
    m2 = fit_lasso(df2, ["x0", "x1"], "y", alpha=0.5)
    np.testing.assert_allclose(m1.coef_, m2.coef_, rtol=1e-12)
    assert m1.intercept_ == pytest.approx(m2.intercept_, rel=1e-12)


# ---------- nulls ----------

def test_lasso_with_nulls(tmp_path):
    n = 300
    data, X, y = _synth(n, [2.0], intercept=0.5)
    data["x0"] = [None if i % 10 == 0 else v for i, v in enumerate(data["x0"])]
    df = make_df(tmp_path, data)
    model = fit_lasso(df, ["x0"], "y", alpha=0.01, max_iter=5000, tol=1e-10)
    # Small alpha, no noise → coef ≈ 2.0
    assert abs(model.coef_[0] - 2.0) < 0.01


# ---------- KKT property (basic sanity) ----------

def test_lasso_kkt_sign_property(tmp_path):
    """For non-zero coef, sign(coef_j) = sign(x_j^T residual)."""
    n = 400
    data, X, y = _synth(n, [2.0, 0.5], intercept=1.0, noise=0.05)
    df = make_df(tmp_path, data)
    model = fit_lasso(df, ["x0", "x1"], "y", alpha=0.05,
                      max_iter=5000, tol=1e-9)
    # Manual residual
    pred = X @ model.coef_ + model.intercept_
    res = y - pred
    x_c = X - X.mean(axis=0, keepdims=True)
    corr = x_c.T @ res
    for j in range(2):
        if abs(model.coef_[j]) > 0.01:
            assert np.sign(model.coef_[j]) == np.sign(corr[j]), (
                f"KKT sign mismatch at j={j}: coef={model.coef_[j]}, corr={corr[j]}"
            )


# ---------- invalid arguments ----------

def test_lasso_invalid_penalty(tmp_path):
    df = make_df(tmp_path, {"x": [1.0, 2.0], "y": [1.0, 2.0]})
    with pytest.raises(ValueError, match="penalty"):
        fit_linear(df, ["x"], "y", penalty="l3")


def test_lasso_invalid_l1_ratio(tmp_path):
    df = make_df(tmp_path, {"x": [1.0, 2.0], "y": [1.0, 2.0]})
    with pytest.raises(ValueError, match="l1_ratio"):
        fit_linear(df, ["x"], "y", penalty="elasticnet", l1_ratio=1.5)
    with pytest.raises(ValueError, match="l1_ratio"):
        fit_linear(df, ["x"], "y", penalty="elasticnet", l1_ratio=-0.1)


def test_lasso_invalid_max_iter(tmp_path):
    df = make_df(tmp_path, {"x": [1.0, 2.0], "y": [1.0, 2.0]})
    with pytest.raises(ValueError, match="max_iter"):
        fit_linear(df, ["x"], "y", penalty="l1", max_iter=0)


def test_lasso_negative_alpha_raises(tmp_path):
    df = make_df(tmp_path, {"x": [1.0, 2.0], "y": [1.0, 2.0]})
    with pytest.raises(ValueError, match="alpha"):
        fit_lasso(df, ["x"], "y", alpha=-1.0)


# ---------- constant feature ----------

def test_lasso_constant_feature_zero_coef(tmp_path):
    """Constant column: after centering, z_j=0 → coef=0."""
    n = 100
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=n)
    y = 2.0 * x0 + 0.5
    df = make_df(tmp_path, {
        "x0": x0.tolist(),
        "const": [1.0] * n,
        "y": y.tolist(),
    })
    model = fit_lasso(df, ["x0", "const"], "y", alpha=0.01,
                      max_iter=2000, tol=1e-9)
    assert model.coef_[1] == 0.0  # constant feature
    assert abs(model.coef_[0] - 2.0) < 0.01


# ---------- CadwDataFrame convenience ----------

def test_fit_lasso_dataframe_method(tmp_path):
    n = 200
    data, X, y = _synth(n, [2.0, -1.0], intercept=0.5, noise=0.1)
    df = make_df(tmp_path, data)
    model = df.fit_lasso(["x0", "x1"], "y", alpha=0.5)
    assert isinstance(model, LinearModel)
    assert model.penalty_ == "l1"
    assert model.l1_ratio_ == 1.0
    assert model.n_iter_cd_ >= 1
    assert model.converged_


def test_fit_elasticnet_dataframe_method(tmp_path):
    n = 200
    data, X, y = _synth(n, [2.0, -1.0], intercept=0.5, noise=0.1)
    df = make_df(tmp_path, data)
    model = df.fit_elasticnet(["x0", "x1"], "y", alpha=0.5, l1_ratio=0.5)
    assert model.penalty_ == "elasticnet"
    assert model.l1_ratio_ == 0.5
    assert model.converged_


# ---------- to_dict ----------

def test_lasso_to_dict(tmp_path):
    n = 200
    data, X, y = _synth(n, [2.0, -1.0], intercept=0.5, noise=0.1)
    df = make_df(tmp_path, data)
    model = fit_lasso(df, ["x0", "x1"], "y", alpha=0.5)
    d = model.to_dict()
    assert d["penalty"] == "l1"
    assert d["l1_ratio"] == 1.0
    assert d["n_iter_cd"] >= 1
    assert d["converged"] is True
    assert "coefficients" in d
    assert "x0" in d["coefficients"]


# ---------- predict works ----------

def test_lasso_predict_df(tmp_path):
    n = 200
    data, X, y = _synth(n, [2.0, 0.0], intercept=0.5, noise=0.0)
    df = make_df(tmp_path, data)
    model = fit_lasso(df, ["x0", "x1"], "y", alpha=0.01,
                      max_iter=2000, tol=1e-9)
    result = model.predict(df)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    t = pq.read_table(out)
    pred = t.column("y_pred").to_pylist()
    for p, yy in zip(pred, y.tolist()):
        assert abs(p - yy) < 0.05
