"""v0.7.5: GBDT binary classification (logloss)."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard, fit_gbdt_classifier
from cadw.core.memory_guard import MemoryLimitExceeded
from cadw.ml.gbdt import GBDTClassifier, fit_gbdt_regressor


def make_df(tmp_path, data, chunk=50, name="L"):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def _synth_binary(n, coefs, intercept=0.0, noise=0.1, seed=42):
    rng = np.random.default_rng(seed)
    d = len(coefs)
    X = rng.normal(size=(n, d))
    z = X @ np.array(coefs) + intercept + noise * rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-z))
    y = (rng.random(n) < p).astype(np.int64)
    return X, y


def _make_binary_df(tmp_path, n=500, coefs=(2.0, -1.5), intercept=0.5, seed=42):
    X, y = _synth_binary(n, coefs, intercept=intercept, seed=seed)
    data = {f"x{i}": X[:, i] for i in range(X.shape[1])}
    data["y"] = y.tolist()
    return make_df(tmp_path, data), X, y


# ---------- basic ----------

def test_fit_gbdt_classifier_accuracy(tmp_path):
    """Strong clean signal: accuracy > 0.85 (Bayes-rate ~0.90).

    coefs=(4.0,), noise=0.0 → z = 4x, y ~ Bernoulli(sigmoid(z)).
    Bayes accuracy ≈ E[max(p,1-p)] ≈ 0.90. Model should reach > 0.85.
    """
    df, X, y = _make_binary_df(
        tmp_path, n=1000, coefs=(4.0,), intercept=0.0,
    )
    # _make_binary_df default noise=0.1; override via direct call
    from tests.test_gbdt_classification import _synth_binary
    X, y = _synth_binary(1000, (4.0,), intercept=0.0, noise=0.0, seed=42)
    data = {f"x{i}": X[:, i].tolist() for i in range(X.shape[1])}
    data["y"] = y.tolist()
    from tests.test_gbdt_classification import make_df as _mdf
    df = _mdf(tmp_path, data, name="strong")

    model = fit_gbdt_classifier(
        df, ["x0"], "y",
        n_estimators=50, max_depth=3, learning_rate=0.1, bins=16,
    )
    assert isinstance(model, GBDTClassifier)
    pred = model.predict(X)
    acc = (pred == y).mean()
    assert acc > 0.85, f"accuracy={acc:.3f} (expected > 0.85)"


def test_fit_gbdt_classifier_predict_proba_range(tmp_path):
    df, X, y = _make_binary_df(tmp_path)
    model = fit_gbdt_classifier(df, ["x0", "x1"], "y", n_estimators=10)
    proba = model.predict_proba(X[:20])
    assert proba.shape == (20,)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_fit_gbdt_classifier_n_estimators_improves(tmp_path):
    """More trees → higher accuracy (monotone-ish)."""
    df, X, y = _make_binary_df(tmp_path, n=800, coefs=(2.0,), intercept=0.0)
    m5 = fit_gbdt_classifier(df, ["x0"], "y", n_estimators=5, learning_rate=0.1)
    m50 = fit_gbdt_classifier(df, ["x0"], "y", n_estimators=50, learning_rate=0.1)
    acc5 = (m5.predict(X) == y).mean()
    acc50 = (m50.predict(X) == y).mean()
    assert acc50 >= acc5, f"acc(5)={acc5:.3f} vs acc(50)={acc50:.3f}"


def test_fit_gbdt_classifier_deterministic(tmp_path):
    df, X, y = _make_binary_df(tmp_path)
    m1 = fit_gbdt_classifier(df, ["x0", "x1"], "y", n_estimators=10)
    m2 = fit_gbdt_classifier(df, ["x0", "x1"], "y", n_estimators=10)
    np.testing.assert_allclose(
        m1.predict_proba(X), m2.predict_proba(X), rtol=1e-9,
    )


def test_fit_gbdt_classifier_multibatch_consistent(tmp_path):
    X, y = _synth_binary(500, (2.0,), seed=42)
    data = {"x0": X[:, 0].tolist(), "y": y.tolist()}
    df1 = make_df(tmp_path, data, chunk=10, name="a")
    df2 = make_df(tmp_path, data, chunk=100, name="b")
    m1 = fit_gbdt_classifier(df1, ["x0"], "y", n_estimators=10)
    m2 = fit_gbdt_classifier(df2, ["x0"], "y", n_estimators=10)
    np.testing.assert_allclose(
        m1.predict_proba(X), m2.predict_proba(X), rtol=1e-9,
    )


# ---------- predict / threshold ----------

def test_predict_threshold(tmp_path):
    df, X, y = _make_binary_df(tmp_path)
    model = fit_gbdt_classifier(df, ["x0", "x1"], "y", n_estimators=20)
    pred_05 = model.predict(X, threshold=0.5)
    pred_low = model.predict(X, threshold=0.1)
    pred_high = model.predict(X, threshold=0.9)
    assert pred_low.sum() >= pred_05.sum()
    assert pred_high.sum() <= pred_05.sum()


def test_predict_invalid_threshold(tmp_path):
    df, X, y = _make_binary_df(tmp_path)
    model = fit_gbdt_classifier(df, ["x0", "x1"], "y", n_estimators=5)
    with pytest.raises(ValueError, match="threshold"):
        model.predict(X, threshold=1.5)
    with pytest.raises(ValueError, match="threshold"):
        model.predict(X, threshold=-0.1)


def test_predict_df_adds_columns(tmp_path):
    df, X, y = _make_binary_df(tmp_path)
    model = fit_gbdt_classifier(df, ["x0", "x1"], "y", n_estimators=5)
    result = model.predict(df)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    names = pq.read_schema(out).names
    assert "y_prob" in names
    assert "y_pred" in names


def test_predict_proba_df_adds_only_prob(tmp_path):
    df, X, y = _make_binary_df(tmp_path)
    model = fit_gbdt_classifier(df, ["x0", "x1"], "y", n_estimators=5)
    result = model.predict_proba(df)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    names = pq.read_schema(out).names
    assert "y_prob" in names
    assert "y_pred" not in names


def test_predict_null_rows_yield_null(tmp_path):
    df, X, y = _make_binary_df(tmp_path)
    # Introduce null
    df_np = df.head(df.count())  # materialize
    # Easier: rebuild df with one null
    data = {f"x{i}": X[:, i].tolist() for i in range(X.shape[1])}
    data["y"] = y.tolist()
    data["x0"][5] = None
    df2 = make_df(tmp_path, data, name="with_null")
    model = fit_gbdt_classifier(df2, ["x0", "x1"], "y", n_estimators=5)
    result = model.predict(df2)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert rows[5]["y_prob"] is None
    assert rows[5]["y_pred"] is None


# ---------- validation ----------

def test_fit_gbdt_classifier_non_binary_y_raises(tmp_path):
    rng = np.random.default_rng(0)
    n = 100
    df = make_df(tmp_path, {
        "x0": rng.normal(size=n).tolist(),
        "y": [2 if i % 2 else 3 for i in range(n)],
    })
    with pytest.raises(ValueError, match="binary"):
        fit_gbdt_classifier(df, ["x0"], "y", n_estimators=5)


def test_fit_gbdt_classifier_single_class_raises(tmp_path):
    rng = np.random.default_rng(0)
    df = make_df(tmp_path, {
        "x0": rng.normal(size=50).tolist(),
        "y": [1] * 50,
    })
    with pytest.raises(ValueError, match="both classes"):
        fit_gbdt_classifier(df, ["x0"], "y", n_estimators=5)


def test_fit_gbdt_classifier_target_in_X(tmp_path):
    df, X, y = _make_binary_df(tmp_path)
    with pytest.raises(ValueError, match="must not be in X_cols"):
        fit_gbdt_classifier(df, ["x0", "y"], "y", n_estimators=5)


def test_fit_gbdt_classifier_missing_column(tmp_path):
    df, X, y = _make_binary_df(tmp_path)
    with pytest.raises(KeyError, match="nope"):
        fit_gbdt_classifier(df, ["nope"], "y")


def test_fit_gbdt_classifier_non_numeric(tmp_path):
    df = make_df(tmp_path, {
        "x0": ["a", "b", "c", "d"],
        "y": [0, 1, 0, 1],
    })
    with pytest.raises(TypeError, match="numeric"):
        fit_gbdt_classifier(df, ["x0"], "y")


def test_fit_gbdt_classifier_memory_guard(tmp_path):
    rng = np.random.default_rng(0)
    n = 10_000
    X = rng.normal(size=n)
    y = (rng.random(n) < 0.5).astype(np.int64)
    # Ensure both classes
    y[0] = 0; y[1] = 1
    df = make_df(tmp_path, {"x0": X.tolist(), "y": y.tolist()})
    df._memory_guard = MemoryGuard(max_memory_bytes=100_000)
    with pytest.raises(MemoryLimitExceeded, match="internal_state"):
        fit_gbdt_classifier(df, ["x0"], "y", n_estimators=5)


# ---------- convenience / to_dict ----------

def test_fit_gbdt_classifier_dataframe_method(tmp_path):
    df, X, y = _make_binary_df(tmp_path)
    model = df.fit_gbdt_classifier(["x0", "x1"], "y", n_estimators=5)
    assert isinstance(model, GBDTClassifier)


def test_classifier_to_dict(tmp_path):
    df, X, y = _make_binary_df(tmp_path)
    model = fit_gbdt_classifier(df, ["x0", "x1"], "y", n_estimators=5)
    d = model.to_dict()
    assert d["loss"] == "logloss_binary"
    assert d["n_trees"] == 5


# ---------- regression still works ----------

def test_regressor_still_works(tmp_path):
    """Regression API unchanged after refactor."""
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    y = 2.0 * x + 0.1 * rng.normal(size=n)
    df = make_df(tmp_path, {"x": x.tolist(), "y": y.tolist()})
    model = fit_gbdt_regressor(
        df, ["x"], "y", n_estimators=10, max_depth=2, learning_rate=0.2,
    )
    pred = model.predict(np.column_stack([x]))
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot
    assert r2 > 0.9

def test_fit_gbdt_classifier_uses_codes_path(monkeypatch, tmp_path):
    """P1a-1c + v0.9.172: hist via _AllNodeHistOpCodes; assign via _assign_leaf_batch."""
    from cadw.ml import gbdt as gbdt_mod
    hist_codes_calls = [0]
    assign_codes_calls = [0]
    orig_hist = gbdt_mod._AllNodeHistOpCodes
    orig_assign_batch = gbdt_mod._assign_leaf_batch

    class SpyHist(orig_hist):
        def update(self, state, batch):
            hist_codes_calls[0] += 1
            return super().update(state, batch)

    def spy_assign_batch(codes, nan_mask, tree, root_id=0):
        # v0.9.172 (S2c): _fit_boosting calls _assign_leaf_batch directly
        # (no ReduceOp) for memmap-write compatibility.
        assign_codes_calls[0] += 1
        return orig_assign_batch(codes, nan_mask, tree, root_id)

    monkeypatch.setattr(gbdt_mod, "_AllNodeHistOpCodes", SpyHist)
    monkeypatch.setattr(gbdt_mod, "_assign_leaf_batch", spy_assign_batch)

    n = 200
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    y = (x > 0).astype(int).tolist()
    df = make_df(tmp_path, {"x": x.tolist(), "y": y})
    fit_gbdt_classifier(df, ["x"], "y", n_estimators=3, max_depth=2)

    assert hist_codes_calls[0] >= 1
    assert assign_codes_calls[0] >= 1

