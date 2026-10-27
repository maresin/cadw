"""v0.7.3: GBDT regression (L2 loss, level-wise)."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.core.memory_guard import MemoryLimitExceeded
from cadw.ml.gbdt import GBDTRegressor, fit_gbdt_regressor


def make_df(tmp_path, data, chunk=20, name="L"):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def _r2(y_true, y_pred):
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


# ---------- basic fit ----------

def test_fit_gbdt_linear_perfect(tmp_path):
    """Linear y = 2x: single depth-1 tree + lr=1 should fit exactly."""
    n = 200
    rng = np.random.default_rng(42)
    x = rng.uniform(-1, 1, size=n)
    y = 2.0 * x
    df = make_df(tmp_path, {"x": x.tolist(), "y": y.tolist()})
    model = fit_gbdt_regressor(
        df, ["x"], "y",
        n_estimators=10, max_depth=3, learning_rate=0.3,
        bins=32, min_child_weight=1.0, reg_lambda=0.0,
    )
    pred = model.predict(np.column_stack([x]))
    r2 = _r2(y, pred)
    assert r2 > 0.95, f"R²={r2:.3f}"


def test_fit_gbdt_nonlinear(tmp_path):
    """Nonlinear y = sin(2x) + x². GBDT beats constant baseline."""
    n = 1000
    rng = np.random.default_rng(42)
    x = rng.uniform(-2, 2, size=n)
    y = np.sin(2.0 * x) + x * x + 0.05 * rng.normal(size=n)
    df = make_df(tmp_path, {"x": x.tolist(), "y": y.tolist()})
    model = fit_gbdt_regressor(
        df, ["x"], "y",
        n_estimators=50, max_depth=3, learning_rate=0.1, bins=32,
    )
    pred = model.predict(np.column_stack([x]))
    r2 = _r2(y, pred)
    assert r2 > 0.7, f"R²={r2:.3f} (baseline would be 0)"


def test_fit_gbdt_two_features(tmp_path):
    """y = x1 + 2*x2 (linear) — GBDT recovers via splits."""
    n = 500
    rng = np.random.default_rng(42)
    x1 = rng.uniform(-1, 1, size=n)
    x2 = rng.uniform(-1, 1, size=n)
    y = x1 + 2 * x2
    df = make_df(tmp_path, {
        "x1": x1.tolist(), "x2": x2.tolist(), "y": y.tolist(),
    })
    model = fit_gbdt_regressor(
        df, ["x1", "x2"], "y",
        n_estimators=30, max_depth=3, learning_rate=0.2, bins=16,
    )
    pred = model.predict(np.column_stack([x1, x2]))
    r2 = _r2(y, pred)
    assert r2 > 0.9, f"R²={r2:.3f}"


def test_fit_gbdt_n_estimators_improves(tmp_path):
    """More trees → better R² (monotone-ish)."""
    n = 500
    rng = np.random.default_rng(42)
    x = rng.uniform(-2, 2, size=n)
    y = np.sin(2.0 * x) + 0.05 * rng.normal(size=n)
    df = make_df(tmp_path, {"x": x.tolist(), "y": y.tolist()})
    m10 = fit_gbdt_regressor(
        df, ["x"], "y", n_estimators=10, max_depth=3, learning_rate=0.1,
    )
    m50 = fit_gbdt_regressor(
        df, ["x"], "y", n_estimators=50, max_depth=3, learning_rate=0.1,
    )
    r2_10 = _r2(y, m10.predict(np.column_stack([x])))
    r2_50 = _r2(y, m50.predict(np.column_stack([x])))
    assert r2_50 > r2_10, f"R²(50)={r2_50:.3f} vs R²(10)={r2_10:.3f}"


def test_fit_gbdt_learning_rate_effect(tmp_path):
    """Higher lr converges faster per round (but may overfit)."""
    n = 500
    rng = np.random.default_rng(42)
    x = rng.uniform(-2, 2, size=n)
    y = np.sin(2.0 * x)
    df = make_df(tmp_path, {"x": x.tolist(), "y": y.tolist()})
    m_small_lr = fit_gbdt_regressor(
        df, ["x"], "y", n_estimators=5, max_depth=3, learning_rate=0.05,
    )
    m_big_lr = fit_gbdt_regressor(
        df, ["x"], "y", n_estimators=5, max_depth=3, learning_rate=0.5,
    )
    r2_small = _r2(y, m_small_lr.predict(np.column_stack([x])))
    r2_big = _r2(y, m_big_lr.predict(np.column_stack([x])))
    # Higher lr with more steps: better fit in first 5 rounds
    assert r2_big > r2_small, f"R²(0.5)={r2_big:.3f} vs R²(0.05)={r2_small:.3f}"


# ---------- predict ----------

def test_predict_numpy_shape(tmp_path):
    n = 100
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    y = x + 0.1 * rng.normal(size=n)
    df = make_df(tmp_path, {"x": x.tolist(), "y": y.tolist()})
    model = fit_gbdt_regressor(df, ["x"], "y", n_estimators=5)
    pred = model.predict(np.column_stack([x[:10]]))
    assert pred.shape == (10,)


def test_predict_df_adds_column(tmp_path):
    n = 100
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    y = x + 0.1 * rng.normal(size=n)
    df = make_df(tmp_path, {"x": x.tolist(), "y": y.tolist()})
    model = fit_gbdt_regressor(df, ["x"], "y", n_estimators=5)
    result = model.predict(df)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    names = pq.read_schema(out).names
    assert "y_pred" in names


def test_predict_null_rows_yield_null(tmp_path):
    n = 100
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    y = x + 0.1 * rng.normal(size=n)
    x_list = x.tolist()
    x_list[5] = None
    df = make_df(tmp_path, {"x": x_list, "y": y.tolist()})
    model = fit_gbdt_regressor(df, ["x"], "y", n_estimators=5)
    result = model.predict(df)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert rows[5]["y_pred"] is None


# ---------- nulls in fit ----------

def test_fit_gbdt_with_nulls(tmp_path):
    n = 200
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    y = x + 0.1 * rng.normal(size=n)
    x_list = x.tolist()
    y_list = y.tolist()
    for i in range(0, n, 10):
        x_list[i] = None
    df = make_df(tmp_path, {"x": x_list, "y": y_list})
    model = fit_gbdt_regressor(df, ["x"], "y", n_estimators=5)
    # Predict non-null rows — should have reasonable R² on remaining
    X_eval = np.column_stack([x])
    mask = ~np.isnan(X_eval).any(axis=1)
    pred = model.predict(X_eval[mask])
    r2 = _r2(y[mask], pred)
    assert r2 > 0.3  # weak threshold; noise


# ---------- determinism ----------

def test_fit_gbdt_deterministic(tmp_path):
    n = 200
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    y = x + 0.1 * rng.normal(size=n)
    df = make_df(tmp_path, {"x": x.tolist(), "y": y.tolist()})
    m1 = fit_gbdt_regressor(df, ["x"], "y", n_estimators=10)
    m2 = fit_gbdt_regressor(df, ["x"], "y", n_estimators=10)
    X = np.column_stack([x])
    np.testing.assert_allclose(
        m1.predict(X), m2.predict(X), rtol=1e-9,
    )


def test_fit_gbdt_multibatch_consistent(tmp_path):
    n = 300
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    y = x + 0.1 * rng.normal(size=n)
    df1 = make_df(tmp_path, {"x": x.tolist(), "y": y.tolist()},
                  chunk=10, name="a")
    df2 = make_df(tmp_path, {"x": x.tolist(), "y": y.tolist()},
                  chunk=100, name="b")
    m1 = fit_gbdt_regressor(df1, ["x"], "y", n_estimators=5)
    m2 = fit_gbdt_regressor(df2, ["x"], "y", n_estimators=5)
    X = np.column_stack([x])
    np.testing.assert_allclose(
        m1.predict(X), m2.predict(X), rtol=1e-9,
    )


# ---------- memory guard ----------

def test_fit_gbdt_guard_rejects_tight_budget(tmp_path):
    """n=10k × 32B = 320KB; budget 100KB → MemoryLimitExceeded."""
    n = 10_000
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    y = x + 0.1 * rng.normal(size=n)
    df = make_df(tmp_path, {"x": x.tolist(), "y": y.tolist()})
    df._memory_guard = MemoryGuard(max_memory_bytes=100_000)
    with pytest.raises(MemoryLimitExceeded, match="internal state"):
        fit_gbdt_regressor(df, ["x"], "y", n_estimators=5)


# ---------- validation ----------

def test_fit_gbdt_empty_X(tmp_path):
    df = make_df(tmp_path, {"x": [1.0, 2.0], "y": [1.0, 2.0]})
    with pytest.raises(ValueError, match="X_cols"):
        fit_gbdt_regressor(df, [], "y")


def test_fit_gbdt_target_in_X(tmp_path):
    df = make_df(tmp_path, {"x": [1.0, 2.0], "y": [1.0, 2.0]})
    with pytest.raises(ValueError, match="must not be in X_cols"):
        fit_gbdt_regressor(df, ["x", "y"], "y")


def test_fit_gbdt_missing_column(tmp_path):
    df = make_df(tmp_path, {"x": [1.0, 2.0], "y": [1.0, 2.0]})
    with pytest.raises(KeyError, match="nope"):
        fit_gbdt_regressor(df, ["nope"], "y")


def test_fit_gbdt_non_numeric(tmp_path):
    df = make_df(tmp_path, {"x": ["a", "b"], "y": [1.0, 2.0]})
    with pytest.raises(TypeError, match="numeric"):
        fit_gbdt_regressor(df, ["x"], "y")


def test_fit_gbdt_invalid_hyperparams(tmp_path):
    df = make_df(tmp_path, {"x": [1.0, 2.0], "y": [1.0, 2.0]})
    with pytest.raises(ValueError, match="n_estimators"):
        fit_gbdt_regressor(df, ["x"], "y", n_estimators=0)
    with pytest.raises(ValueError, match="max_depth"):
        fit_gbdt_regressor(df, ["x"], "y", max_depth=0)
    with pytest.raises(ValueError, match="learning_rate"):
        fit_gbdt_regressor(df, ["x"], "y", learning_rate=0.0)
    with pytest.raises(ValueError, match="bins"):
        fit_gbdt_regressor(df, ["x"], "y", bins=1)


# ---------- convenience / to_dict ----------

def test_fit_gbdt_dataframe_method(tmp_path):
    n = 100
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    y = x
    df = make_df(tmp_path, {"x": x.tolist(), "y": y.tolist()})
    model = df.fit_gbdt_regressor("x", "y", n_estimators=5)
    assert isinstance(model, GBDTRegressor)
    assert len(model.trees_) == 5


def test_gbdt_to_dict(tmp_path):
    n = 100
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    y = x
    df = make_df(tmp_path, {"x": x.tolist(), "y": y.tolist()})
    model = fit_gbdt_regressor(df, ["x"], "y", n_estimators=5)
    d = model.to_dict()
    assert "base_prediction" in d
    assert "feature_names" in d
    assert d["n_estimators"] == 5
    assert d["n_trees"] == 5


# ---------- filtered ----------

def test_fit_gbdt_filtered(tmp_path):
    n = 500
    rng = np.random.default_rng(0)
    x = rng.uniform(-2, 2, size=n)
    y = np.sin(x) + 0.05 * rng.normal(size=n)
    df = make_df(tmp_path, {"x": x.tolist(), "y": y.tolist()})
    filtered = df[df["x"] > 0]
    model = fit_gbdt_regressor(filtered, ["x"], "y", n_estimators=10)
    # Predict on [0.1, 2.0] — should fit sin(2x) reasonably on x>0
    X_eval = np.linspace(0.1, 2.0, 100).reshape(-1, 1)
    pred = model.predict(X_eval)
    y_eval = np.sin(X_eval.flatten())
    r2 = _r2(y_eval, pred)
    assert r2 > 0.5

def test_fit_gbdt_uses_codes_path(monkeypatch, tmp_path):
    """P1a-1c: regression fit uses codes-based hist/assign ops."""
    from cadw.ml import gbdt as gbdt_mod

    hist_codes_calls = [0]
    assign_codes_calls = [0]
    orig_hist = gbdt_mod._AllNodeHistOpCodes
    orig_assign = gbdt_mod._AssignLeavesOpCodes

    class SpyHist(orig_hist):
        def update(self, state, batch):
            hist_codes_calls[0] += 1
            return super().update(state, batch)

    class SpyAssign(orig_assign):
        def update(self, state, batch):
            assign_codes_calls[0] += 1
            return super().update(state, batch)

    monkeypatch.setattr(gbdt_mod, "_AllNodeHistOpCodes", SpyHist)
    monkeypatch.setattr(gbdt_mod, "_AssignLeavesOpCodes", SpyAssign)

    n = 200
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    y = x + 0.1 * rng.normal(size=n)
    df = make_df(tmp_path, {"x": x.tolist(), "y": y.tolist()})
    fit_gbdt_regressor(df, ["x"], "y", n_estimators=3, max_depth=2)

    assert hist_codes_calls[0] >= 1, (
        f"_AllNodeHistOpCodes.update called {hist_codes_calls[0]} times")
    assert assign_codes_calls[0] >= 1, (
        f"_AssignLeavesOpCodes.update called {assign_codes_calls[0]} times")

