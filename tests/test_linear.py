"""v0.7.0a: Linear/Ridge via Reduce."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.ml.linear import LinearModel, XTXOp, fit_linear


def make_df(tmp_path, data, chunk=10, name="L"):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def _synth_linear(n, coefs, intercept, noise=0.0, seed=42):
    rng = np.random.default_rng(seed)
    d = len(coefs)
    X = rng.normal(size=(n, d))
    y = X @ np.array(coefs) + intercept + noise * rng.normal(size=n)
    data = {f"x{i}": X[:, i] for i in range(d)}
    data["y"] = y
    return data, X, y


# ---------- XTXOp ----------

def test_xtxop_estimate_state_nbytes():
    """P8 API for custom ReduceOp."""
    op = XTXOp(["a", "b", "c"], "y")
    assert op.estimate_state_nbytes() == 8 + 8*3*3 + 8*3 + 8*3 + 8


def test_xtxop_state_nbytes_matches_actual():
    op = XTXOp(["a", "b"], "y")
    assert op.state_nbytes(op.init()) <= op.estimate_state_nbytes()


# ---------- OLS / Ridge ----------

def test_fit_linear_perfect():
    """Noiseless linear data: coefficients exactly recovered."""
    n = 200
    data, X, y = _synth_linear(n, [2.0, -3.0, 0.5], intercept=1.5)
    # In-memory synthetic: use minimal parquet df via numpy->table
    import pyarrow as pa_  # local
    tbl = pa_.table(data)
    # Temporary: use in-memory wrapper directly
    from cadw.dataset import Dataset
    ds = Dataset.from_batches(
        tbl.schema, iter(tbl.to_batches(max_chunksize=20)),
        Path("/tmp/cadw_v070_test_ds"),
    )
    df = CadwDataFrame(
        ds,
        memory_guard=MemoryGuard(max_memory_mb=50),
        spill_dir=Path("/tmp/cadw_v070_test_spill"),
    )
    model = fit_linear(df, ["x0", "x1", "x2"], "y")
    assert isinstance(model, LinearModel)
    np.testing.assert_allclose(model.coef_, [2.0, -3.0, 0.5], atol=1e-9)
    assert abs(model.intercept_ - 1.5) < 1e-9
    assert model.n_rows_ == n


def test_fit_linear_with_noise_approx():
    n = 500
    data, X, y = _synth_linear(n, [2.0, -3.0], intercept=1.0, noise=0.1)
    import pyarrow as pa_  # local
    tbl = pa_.table(data)
    from cadw.dataset import Dataset
    ds = Dataset.from_batches(
        tbl.schema, iter(tbl.to_batches(max_chunksize=50)),
        Path("/tmp/cadw_v070_noise_ds"),
    )
    df = CadwDataFrame(
        ds,
        memory_guard=MemoryGuard(max_memory_mb=50),
        spill_dir=Path("/tmp/cadw_v070_noise_spill"),
    )
    model = fit_linear(df, ["x0", "x1"], "y")
    assert abs(model.coef_[0] - 2.0) < 0.05
    assert abs(model.coef_[1] + 3.0) < 0.05
    assert abs(model.intercept_ - 1.0) < 0.1


def test_fit_linear_matches_numpy_lstsq(tmp_path):
    """Compare to np.linalg.lstsq."""
    n, d = 100, 3
    data, X, y = _synth_linear(n, [1.0, 2.0, 3.0], intercept=0.5, noise=0.1)
    df = make_df(tmp_path, data)
    model = fit_linear(df, [f"x{i}" for i in range(d)], "y")

    # numpy reference
    X_aug = np.column_stack([np.ones(n), X])
    beta_ref, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
    np.testing.assert_allclose(model.coef_, beta_ref[1:], rtol=1e-9)
    assert abs(model.intercept_ - beta_ref[0]) < 1e-9


def test_fit_ridge_shrinks_coef():
    n = 200
    data, X, y = _synth_linear(n, [5.0, 5.0], intercept=0.0)
    import pyarrow as pa_
    tbl = pa_.table(data)
    from cadw.dataset import Dataset
    ds = Dataset.from_batches(
        tbl.schema, iter(tbl.to_batches(max_chunksize=20)),
        Path("/tmp/cadw_v070_ridge_ds"),
    )
    df = CadwDataFrame(
        ds,
        memory_guard=MemoryGuard(max_memory_mb=50),
        spill_dir=Path("/tmp/cadw_v070_ridge_spill"),
    )
    m_ols = fit_linear(df, ["x0", "x1"], "y", alpha=0.0)
    m_ridge = fit_linear(df, ["x0", "x1"], "y", alpha=100.0)
    # Ridge shrinks coef magnitude
    assert np.linalg.norm(m_ridge.coef_) < np.linalg.norm(m_ols.coef_)


def test_fit_ridge_large_alpha_near_zero():
    n = 200
    data, _, _ = _synth_linear(n, [5.0], intercept=0.0)
    import pyarrow as pa_
    tbl = pa_.table(data)
    from cadw.dataset import Dataset
    ds = Dataset.from_batches(
        tbl.schema, iter(tbl.to_batches(max_chunksize=20)),
        Path("/tmp/cadw_v070_ridge_big_ds"),
    )
    df = CadwDataFrame(
        ds,
        memory_guard=MemoryGuard(max_memory_mb=50),
        spill_dir=Path("/tmp/cadw_v070_ridge_big_spill"),
    )
    m = fit_linear(df, ["x0"], "y", alpha=1e8)
    assert abs(m.coef_[0]) < 0.01


def test_fit_linear_single_feature(tmp_path):
    n = 100
    x = np.arange(n, dtype=np.float64)
    y = 2.0 * x + 3.0
    df = make_df(tmp_path, {"x": x.tolist(), "y": y.tolist()})
    model = fit_linear(df, ["x"], "y")
    assert abs(model.coef_[0] - 2.0) < 1e-9
    assert abs(model.intercept_ - 3.0) < 1e-9


def test_fit_linear_multi_batch(tmp_path):
    """Chunk size doesn't affect result."""
    n = 100
    data, X, y = _synth_linear(n, [1.5, -2.0], intercept=0.5, noise=0.1)
    df1 = make_df(tmp_path, data, chunk=10, name="a")
    df2 = make_df(tmp_path, data, chunk=33, name="b")
    m1 = fit_linear(df1, ["x0", "x1"], "y")
    m2 = fit_linear(df2, ["x0", "x1"], "y")
    np.testing.assert_allclose(m1.coef_, m2.coef_, rtol=1e-12)
    assert abs(m1.intercept_ - m2.intercept_) < 1e-12


# ---------- nulls ----------

def test_fit_linear_with_nulls(tmp_path):
    """Nulls in X or y → row skipped."""
    n = 100
    data, X, y = _synth_linear(n, [2.0], intercept=1.0)
    # Add nulls
    data["x0"] = [None if i % 10 == 0 else v for i, v in enumerate(data["x0"])]
    data["y"] = [None if i % 15 == 0 else v for i, v in enumerate(data["y"])]

    df = make_df(tmp_path, data)
    model = fit_linear(df, ["x0"], "y")
    # Still recovers coef (approx)
    assert abs(model.coef_[0] - 2.0) < 0.01
    assert model.n_rows_ < n


def test_fit_linear_all_null_raises(tmp_path):
    """All-null float64 columns: n_valid=0 → ValueError.

    Explicit type=pa.float64() — `pa.table({"x0": [None, None, None]})`
    infers `null` type (correctly rejected by fit_linear as non-numeric).
    """
    df = make_df(tmp_path, {
        "x0": pa.array([None, None, None], type=pa.float64()),
        "y": pa.array([None, None, None], type=pa.float64()),
    })
    with pytest.raises(ValueError, match="no valid"):
        fit_linear(df, ["x0"], "y")


def test_fit_linear_null_typed_column_raises(tmp_path):
    """Column of `null` type (not float64) — TypeError (non-numeric)."""
    df = make_df(tmp_path, {
        "x0": [None, None, None],   # `null` type
        "y": [1.0, 2.0, 3.0],       # float64
    })
    with pytest.raises(TypeError, match="numeric"):
        fit_linear(df, ["x0"], "y")


# ---------- errors ----------

def test_fit_linear_empty_X_cols(tmp_path):
    df = make_df(tmp_path, {"x0": [1.0, 2.0], "y": [1.0, 2.0]})
    with pytest.raises(ValueError, match="X_cols"):
        fit_linear(df, [], "y")


def test_fit_linear_target_in_X(tmp_path):
    df = make_df(tmp_path, {"x0": [1.0, 2.0], "y": [1.0, 2.0]})
    with pytest.raises(ValueError, match="must not be in X_cols"):
        fit_linear(df, ["x0", "y"], "y")


def test_fit_linear_negative_alpha(tmp_path):
    df = make_df(tmp_path, {"x0": [1.0, 2.0], "y": [1.0, 2.0]})
    with pytest.raises(ValueError, match="alpha must be >= 0"):
        fit_linear(df, ["x0"], "y", alpha=-1.0)


def test_fit_linear_missing_column(tmp_path):
    df = make_df(tmp_path, {"x0": [1.0, 2.0], "y": [1.0, 2.0]})
    with pytest.raises(KeyError, match="nope"):
        fit_linear(df, ["nope"], "y")


def test_fit_linear_non_numeric_raises(tmp_path):
    df = make_df(tmp_path, {
        "x0": ["a", "b", "c"],
        "y": [1.0, 2.0, 3.0],
    })
    with pytest.raises(TypeError, match="numeric"):
        fit_linear(df, ["x0"], "y")


# ---------- predict ----------

def test_predict_numpy():
    n = 50
    data, X, y = _synth_linear(n, [2.0, -1.0], intercept=0.5)
    import pyarrow as pa_
    tbl = pa_.table(data)
    from cadw.dataset import Dataset
    ds = Dataset.from_batches(
        tbl.schema, iter(tbl.to_batches(max_chunksize=20)),
        Path("/tmp/cadw_v070_pred_ds"),
    )
    df = CadwDataFrame(
        ds,
        memory_guard=MemoryGuard(max_memory_mb=50),
        spill_dir=Path("/tmp/cadw_v070_pred_spill"),
    )
    model = fit_linear(df, ["x0", "x1"], "y")
    pred = model.predict(X[:5])
    expected = X[:5] @ model.coef_ + model.intercept_
    np.testing.assert_allclose(pred, expected)


def test_predict_dataframe(tmp_path):
    n = 50
    data, X, y = _synth_linear(n, [2.0], intercept=1.0)
    df = make_df(tmp_path, data)
    model = fit_linear(df, ["x0"], "y")
    result = model.predict(df)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    t = pq.read_table(out)
    assert "y_pred" in t.schema.names
    pred_vals = t.column("y_pred").to_pylist()
    # Should be close to y (noiseless)
    y_vals = t.column("y").to_pylist()
    for p, yy in zip(pred_vals, y_vals):
        assert abs(p - yy) < 1e-6


def test_predict_df_missing_column(tmp_path):
    n = 30
    data, X, y = _synth_linear(n, [2.0], intercept=1.0)
    df = make_df(tmp_path, data)
    model = fit_linear(df, ["x0"], "y")
    # Select only y
    sub = df[["y"]]
    with pytest.raises(KeyError, match="x0"):
        model.predict(sub)


# ---------- to_dict ----------

def test_linear_model_to_dict(tmp_path):
    n = 50
    data, X, y = _synth_linear(n, [2.0, -1.0], intercept=0.5)
    df = make_df(tmp_path, data)
    model = fit_linear(df, ["x0", "x1"], "y")
    d = model.to_dict()
    assert "coefficients" in d
    assert "x0" in d["coefficients"]
    assert "x1" in d["coefficients"]
    assert abs(d["intercept"] - 0.5) < 1e-9
    assert d["n_rows"] == n


# ---------- convenience method ----------

def test_fit_linear_dataframe_method(tmp_path):
    n = 50
    data, X, y = _synth_linear(n, [2.0], intercept=1.0)
    df = make_df(tmp_path, data)
    model = df.fit_linear("x0", "y")
    assert hasattr(model, "coef_")
    assert abs(model.coef_[0] - 2.0) < 1e-9


# ---------- filtered ----------

def test_fit_linear_filtered(tmp_path):
    n = 100
    data, X, y = _synth_linear(n, [2.0], intercept=1.0, noise=0.0)
    df = make_df(tmp_path, data)
    # Filter to subset
    filtered = df[df["x0"] > 0]
    model = fit_linear(filtered, ["x0"], "y")
    # On subset X>0, coef should still be 2.0, intercept 1.0
    assert abs(model.coef_[0] - 2.0) < 1e-9
    assert abs(model.intercept_ - 1.0) < 1e-9
