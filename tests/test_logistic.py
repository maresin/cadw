"""v0.7.0b: Logistic regression via iter-Reduce Newton IRLS."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.ml.logistic import LogisticGradOp, LogisticModel, fit_logistic


def make_df(tmp_path, data, chunk=20, name="L"):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def _synth_logistic(n, coefs, intercept, noise=0.1, seed=42):
    rng = np.random.default_rng(seed)
    d = len(coefs)
    X = rng.normal(size=(n, d))
    z = X @ np.array(coefs) + intercept + noise * rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-z))
    y = (rng.random(n) < p).astype(np.int64)
    data = {f"x{i}": X[:, i] for i in range(d)}
    data["y"] = y
    return data, X, y


# ---------- op ----------

def test_logistic_grad_op_estimate_state_nbytes():
    op = LogisticGradOp(["a", "b", "c"], "y", np.zeros(4))
    assert op.estimate_state_nbytes() == 8 + 8*4 + 8*4*4 + 8


def test_logistic_grad_op_state_nbytes_matches():
    op = LogisticGradOp(["a", "b"], "y", np.zeros(3))
    assert op.state_nbytes(op.init()) <= op.estimate_state_nbytes()


# ---------- basic fit ----------

def test_fit_logistic_noisy(tmp_path):
    """Right-sign + accuracy above chance floor.

    Note: y = Bernoulli(sigmoid(z)) — inherent label noise. Even an
    ideal model has accuracy = E[max(p,1-p)] ≈ 0.80 for this dataset.
    Stronger correctness check: test_fit_logistic_coef_recovery below.
    """
    n = 1000
    data, X, y = _synth_logistic(n, [2.0, -1.5], intercept=0.5, noise=0.1)
    df = make_df(tmp_path, data)
    model = fit_logistic(df, ["x0", "x1"], "y", alpha=1.0)
    assert isinstance(model, LogisticModel)
    # Right sign of coefficients
    assert model.coef_[0] > 0.5   # true +2
    assert model.coef_[1] < -0.3  # true -1.5
    # Accuracy beats chance (Bayes-optimal for this data ~0.80)
    proba = model.predict_proba(X)
    pred = (proba >= 0.5).astype(int)
    acc = (pred == y).mean()
    assert acc > 0.75, f"acc={acc} (Bayes floor ~0.80)"


def test_fit_logistic_coef_recovery(tmp_path):
    """Direct correctness: recovered coefs match true (noise=0, large n).

    n=5000 gives SE ~ 0.05 per coef; tolerance 0.2 is safe.
    """
    n = 5000
    data, X, y = _synth_logistic(
        n, [2.0, -1.5], intercept=0.5, noise=0.0, seed=42,
    )
    df = make_df(tmp_path, data, chunk=100)
    # Very small alpha: ignore L2 penalty for recovery test
    model = fit_logistic(df, ["x0", "x1"], "y", alpha=0.001)
    assert abs(model.coef_[0] - 2.0) < 0.2, (
        f"coef[0]={model.coef_[0]:.3f}, expected 2.0"
    )
    assert abs(model.coef_[1] - (-1.5)) < 0.2, (
        f"coef[1]={model.coef_[1]:.3f}, expected -1.5"
    )
    assert abs(model.intercept_ - 0.5) < 0.2, (
        f"intercept={model.intercept_:.3f}, expected 0.5"
    )


def test_fit_logistic_converges(tmp_path):
    n = 500
    data, X, y = _synth_logistic(n, [1.0], intercept=0.0, noise=0.1)
    df = make_df(tmp_path, data)
    model = fit_logistic(df, ["x0"], "y", alpha=1.0, max_iter=100, tol=1e-6)
    assert model.converged_
    assert model.n_iter_ < 100
    assert model.n_iter_ >= 1


def test_fit_logistic_multibatch_consistent(tmp_path):
    n = 300
    data, X, y = _synth_logistic(n, [1.0, -0.5], intercept=0.5)
    df1 = make_df(tmp_path, data, chunk=10, name="a")
    df2 = make_df(tmp_path, data, chunk=100, name="b")
    m1 = fit_logistic(df1, ["x0", "x1"], "y", alpha=1.0)
    m2 = fit_logistic(df2, ["x0", "x1"], "y", alpha=1.0)
    np.testing.assert_allclose(m1.coef_, m2.coef_, rtol=1e-6, atol=1e-6)


def test_fit_logistic_alpha_shrinks(tmp_path):
    n = 500
    data, X, y = _synth_logistic(n, [3.0, -3.0], intercept=0.0)
    df = make_df(tmp_path, data)
    m_small = fit_logistic(df, ["x0", "x1"], "y", alpha=0.01)
    m_large = fit_logistic(df, ["x0", "x1"], "y", alpha=100.0)
    assert np.linalg.norm(m_large.coef_) < np.linalg.norm(m_small.coef_)


# ---------- predict ----------

def test_predict_proba_numpy(tmp_path):
    n = 200
    data, X, y = _synth_logistic(n, [1.0], intercept=0.0)
    df = make_df(tmp_path, data)
    model = fit_logistic(df, ["x0"], "y", alpha=1.0)
    proba = model.predict_proba(X[:5])
    assert proba.shape == (5,)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_predict_numpy(tmp_path):
    n = 200
    data, X, y = _synth_logistic(n, [1.0], intercept=0.0)
    df = make_df(tmp_path, data)
    model = fit_logistic(df, ["x0"], "y", alpha=1.0)
    pred = model.predict(X[:5])
    assert pred.shape == (5,)
    assert set(pred.tolist()) <= {0, 1}


def test_predict_df_adds_columns(tmp_path):
    n = 100
    data, X, y = _synth_logistic(n, [1.0], intercept=0.0)
    df = make_df(tmp_path, data)
    model = fit_logistic(df, ["x0"], "y", alpha=1.0)
    result = model.predict(df)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    names = pq.read_schema(out).names
    assert "y_prob" in names
    assert "y_pred" in names


def test_predict_df_proba_only(tmp_path):
    n = 100
    data, X, y = _synth_logistic(n, [1.0], intercept=0.0)
    df = make_df(tmp_path, data)
    model = fit_logistic(df, ["x0"], "y", alpha=1.0)
    result = model.predict_proba(df)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    names = pq.read_schema(out).names
    assert "y_prob" in names
    assert "y_pred" not in names


def test_predict_threshold(tmp_path):
    n = 100
    data, X, y = _synth_logistic(n, [1.0], intercept=0.0)
    df = make_df(tmp_path, data)
    model = fit_logistic(df, ["x0"], "y", alpha=1.0)
    pred_default = model.predict(X, threshold=0.5)
    pred_low = model.predict(X, threshold=0.1)
    # Lower threshold → more positives (or equal)
    assert pred_low.sum() >= pred_default.sum()


def test_predict_invalid_threshold(tmp_path):
    n = 50
    data, X, y = _synth_logistic(n, [1.0], intercept=0.0)
    df = make_df(tmp_path, data)
    model = fit_logistic(df, ["x0"], "y", alpha=1.0)
    with pytest.raises(ValueError, match="threshold"):
        model.predict(X, threshold=1.5)
    with pytest.raises(ValueError, match="threshold"):
        model.predict(X, threshold=-0.5)


# ---------- validation ----------

def test_fit_logistic_target_in_X(tmp_path):
    df = make_df(tmp_path, {"x0": [1.0, 0.0], "y": [1, 0]})
    with pytest.raises(ValueError, match="must not be in X_cols"):
        fit_logistic(df, ["x0", "y"], "y")


def test_fit_logistic_empty_X(tmp_path):
    df = make_df(tmp_path, {"x0": [1.0], "y": [1]})
    with pytest.raises(ValueError, match="X_cols"):
        fit_logistic(df, [], "y")


def test_fit_logistic_negative_alpha(tmp_path):
    df = make_df(tmp_path, {"x0": [1.0, 0.0], "y": [1, 0]})
    with pytest.raises(ValueError, match="alpha must be >= 0"):
        fit_logistic(df, ["x0"], "y", alpha=-1.0)


def test_fit_logistic_missing_column(tmp_path):
    df = make_df(tmp_path, {"x0": [1.0, 0.0], "y": [1, 0]})
    with pytest.raises(KeyError, match="nope"):
        fit_logistic(df, ["nope"], "y")


def test_fit_logistic_non_numeric_raises(tmp_path):
    df = make_df(tmp_path, {"x0": ["a", "b"], "y": [1, 0]})
    with pytest.raises(TypeError, match="numeric"):
        fit_logistic(df, ["x0"], "y")


def test_fit_logistic_non_binary_y_raises(tmp_path):
    df = make_df(tmp_path, {
        "x0": [1.0, 2.0, 3.0, 4.0],
        "y": [2, 3, 2, 3],   # not 0/1
    })
    with pytest.raises(ValueError, match="binary"):
        fit_logistic(df, ["x0"], "y")


def test_fit_logistic_single_class_raises(tmp_path):
    df = make_df(tmp_path, {
        "x0": [1.0, 2.0, 3.0],
        "y": [1, 1, 1],
    })
    with pytest.raises(ValueError, match="both classes"):
        fit_logistic(df, ["x0"], "y")


def test_fit_logistic_all_null_rows_raises(tmp_path):
    df = make_df(tmp_path, {
        "x0": pa.array([None, None], type=pa.float64()),
        "y": pa.array([None, None], type=pa.float64()),
    })
    # Will fail _validate_binary_y first (no values)
    with pytest.raises(ValueError):
        fit_logistic(df, ["x0"], "y")


# ---------- multi / convenience ----------

def test_fit_logistic_dataframe_method(tmp_path):
    n = 200
    data, X, y = _synth_logistic(n, [1.0], intercept=0.0)
    df = make_df(tmp_path, data)
    model = df.fit_logistic("x0", "y")
    assert hasattr(model, "coef_")
    assert hasattr(model, "predict_proba")


def test_logistic_model_to_dict(tmp_path):
    n = 200
    data, X, y = _synth_logistic(n, [1.0, -0.5], intercept=0.5)
    df = make_df(tmp_path, data)
    model = fit_logistic(df, ["x0", "x1"], "y")
    d = model.to_dict()
    assert "coefficients" in d
    assert "x0" in d["coefficients"]
    assert "n_iter" in d
    assert "converged" in d
    assert "log_likelihood" in d


def test_fit_logistic_filtered(tmp_path):
    n = 500
    data, X, y = _synth_logistic(n, [1.0], intercept=0.0, noise=0.3)
    df = make_df(tmp_path, data)
    filtered = df[df["x0"] > -2.0]
    model = fit_logistic(filtered, ["x0"], "y", alpha=1.0)
    assert model.n_rows_ < n


def test_fit_logistic_with_nulls(tmp_path):
    n = 300
    data, X, y = _synth_logistic(n, [1.0], intercept=0.0)
    data["x0"] = [
        None if i % 10 == 0 else v for i, v in enumerate(data["x0"])
    ]
    df = make_df(tmp_path, data)
    model = fit_logistic(df, ["x0"], "y", alpha=1.0)
    # n_rows_ reflects valid count
    assert model.n_rows_ < n
    assert model.n_rows_ > 0
