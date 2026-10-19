"""v0.7.8: Multiclass GBDT (softmax) via per-class loop."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import (
    CadwDataFrame, MemoryGuard,
    fit_gbdt_classifier, fit_gbdt_multiclass,
)
from cadw.core.memory_guard import MemoryLimitExceeded
from cadw.ml.gbdt import (
    GBDTClassifier, GBDTMulticlassClassifier, _softmax,
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


def _synth_multiclass(n, n_classes, d=2, seed=42, sep=4.0):
    """K gaussian clusters along unit circle; y in {0..K-1}."""
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, n_classes, endpoint=False)
    centers = np.column_stack([np.cos(angles), np.sin(angles)]) * sep
    y = rng.integers(0, n_classes, size=n)
    X = centers[y] + rng.normal(scale=0.5, size=(n, d))
    data = {f"x{i}": X[:, i].tolist() for i in range(d)}
    data["y"] = y.astype(np.int64).tolist()
    return data, X, y


# ---------- _softmax ----------

def test_softmax_rows_sum_to_1():
    z = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    p = _softmax(z)
    np.testing.assert_allclose(p.sum(axis=1), [1.0, 1.0])


def test_softmax_large_values_stable():
    """Numerical stability: no overflow for large z."""
    z = np.array([[1000.0, 999.0], [-1000.0, -999.0]])
    p = _softmax(z)
    assert np.all(np.isfinite(p))
    np.testing.assert_allclose(p.sum(axis=1), [1.0, 1.0])


def test_softmax_wrong_dim_raises():
    with pytest.raises(ValueError, match="2D"):
        _softmax(np.array([1.0, 2.0]))


# ---------- basic fit ----------

def test_fit_multiclass_basic(tmp_path):
    n = 300
    data, X, y = _synth_multiclass(n, n_classes=3, seed=0)
    df = make_df(tmp_path, data)
    model = fit_gbdt_multiclass(
        df, ["x0", "x1"], "y",
        n_estimators=10, max_depth=3, learning_rate=0.2, bins=16,
    )
    assert isinstance(model, GBDTMulticlassClassifier)
    assert model.n_classes_ == 3
    assert len(model.trees_) == 10  # n_estimators
    assert len(model.trees_[0]) == 3  # K trees per round


def test_fit_multiclass_accuracy(tmp_path):
    """Separable clusters: accuracy > 0.9."""
    n = 500
    data, X, y = _synth_multiclass(n, n_classes=3, seed=1, sep=4.0)
    df = make_df(tmp_path, data, chunk=100)
    model = fit_gbdt_multiclass(
        df, ["x0", "x1"], "y",
        n_estimators=30, max_depth=3, learning_rate=0.2, bins=16,
    )
    pred = model.predict(X)
    acc = (pred == y).mean()
    assert acc > 0.9, f"accuracy={acc:.3f}"


def test_fit_multiclass_n_estimators_improves(tmp_path):
    n = 500
    data, X, y = _synth_multiclass(n, n_classes=3, seed=2, sep=3.0)
    df = make_df(tmp_path, data, chunk=100)
    m10 = fit_gbdt_multiclass(
        df, ["x0", "x1"], "y",
        n_estimators=10, max_depth=3, learning_rate=0.2, bins=16,
    )
    m40 = fit_gbdt_multiclass(
        df, ["x0", "x1"], "y",
        n_estimators=40, max_depth=3, learning_rate=0.2, bins=16,
    )
    acc10 = (m10.predict(X) == y).mean()
    acc40 = (m40.predict(X) == y).mean()
    assert acc40 >= acc10, f"acc(10)={acc10:.3f} vs acc(40)={acc40:.3f}"


def test_fit_multiclass_deterministic(tmp_path):
    n = 200
    data, X, y = _synth_multiclass(n, n_classes=3, seed=3)
    df = make_df(tmp_path, data)
    m1 = fit_gbdt_multiclass(
        df, ["x0", "x1"], "y", n_estimators=5, max_depth=2,
    )
    m2 = fit_gbdt_multiclass(
        df, ["x0", "x1"], "y", n_estimators=5, max_depth=2,
    )
    np.testing.assert_allclose(
        m1.predict_proba(X), m2.predict_proba(X), rtol=1e-9,
    )


# ---------- predict ----------

def test_predict_proba_shape(tmp_path):
    n = 300
    data, X, y = _synth_multiclass(n, n_classes=4, seed=4)
    df = make_df(tmp_path, data)
    model = fit_gbdt_multiclass(
        df, ["x0", "x1"], "y", n_estimators=5, max_depth=2,
    )
    proba = model.predict_proba(X[:10])
    assert proba.shape == (10, 4)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(10))


def test_predict_returns_int_labels(tmp_path):
    n = 300
    data, X, y = _synth_multiclass(n, n_classes=4, seed=5)
    df = make_df(tmp_path, data)
    model = fit_gbdt_multiclass(
        df, ["x0", "x1"], "y", n_estimators=5, max_depth=2,
    )
    pred = model.predict(X[:20])
    assert pred.shape == (20,)
    assert pred.dtype == np.int64
    assert set(pred.tolist()) <= {0, 1, 2, 3}


def test_predict_df_adds_columns(tmp_path):
    n = 200
    data, X, y = _synth_multiclass(n, n_classes=3, seed=6)
    df = make_df(tmp_path, data)
    model = fit_gbdt_multiclass(
        df, ["x0", "x1"], "y", n_estimators=5, max_depth=2,
    )
    result = model.predict(df)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    names = pq.read_schema(out).names
    assert "y_pred" in names


def test_predict_proba_df_adds_prob_columns(tmp_path):
    n = 200
    data, X, y = _synth_multiclass(n, n_classes=3, seed=7)
    df = make_df(tmp_path, data)
    model = fit_gbdt_multiclass(
        df, ["x0", "x1"], "y", n_estimators=5, max_depth=2,
    )
    result = model.predict_proba(df)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    names = pq.read_schema(out).names
    assert "y_prob_0" in names
    assert "y_prob_1" in names
    assert "y_prob_2" in names


def test_predict_null_rows_yield_null(tmp_path):
    n = 200
    data, X, y = _synth_multiclass(n, n_classes=3, seed=8)
    data["x0"][5] = None
    df = make_df(tmp_path, data)
    model = fit_gbdt_multiclass(
        df, ["x0", "x1"], "y", n_estimators=5, max_depth=2,
    )
    result = model.predict(df)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert rows[5]["y_pred"] is None


# ---------- fit_gbdt_classifier dispatch ----------

def test_dispatch_binary(tmp_path):
    """K=2 → GBDTClassifier (backward compat)."""
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    y = (x > 0).astype(np.int64)
    df = make_df(tmp_path, {"x0": x.tolist(), "y": y.tolist()})
    model = fit_gbdt_classifier(df, ["x0"], "y", n_estimators=10)
    assert isinstance(model, GBDTClassifier)
    assert not isinstance(model, GBDTMulticlassClassifier)


def test_dispatch_multiclass(tmp_path):
    """K=3 → GBDTMulticlassClassifier (auto-dispatch)."""
    n = 300
    data, X, y = _synth_multiclass(n, n_classes=3, seed=10)
    df = make_df(tmp_path, data)
    model = fit_gbdt_classifier(
        df, ["x0", "x1"], "y", n_estimators=10,
    )
    assert isinstance(model, GBDTMulticlassClassifier)


def test_dispatch_single_class_raises(tmp_path):
    n = 100
    df = make_df(tmp_path, {
        "x0": np.random.default_rng(0).normal(size=n).tolist(),
        "y": [1] * n,
    })
    with pytest.raises(ValueError, match="classes"):
        fit_gbdt_classifier(df, ["x0"], "y")


# ---------- y validation ----------

def test_multiclass_1_indexed_raises(tmp_path):
    """y = {1, 2, 3} — rejected (0-indexed required)."""
    n = 100
    rng = np.random.default_rng(0)
    df = make_df(tmp_path, {
        "x0": rng.normal(size=n).tolist(),
        "y": [(i % 3) + 1 for i in range(n)],  # {1, 2, 3}
    })
    with pytest.raises(ValueError, match="0-indexed"):
        fit_gbdt_multiclass(df, ["x0"], "y")


def test_multiclass_missing_class_raises(tmp_path):
    """y = {0, 2} — not {0, 1, 2}; missing 1 → reject."""
    n = 100
    rng = np.random.default_rng(0)
    df = make_df(tmp_path, {
        "x0": rng.normal(size=n).tolist(),
        "y": [0 if i % 2 else 2 for i in range(n)],  # only {0, 2}
    })
    with pytest.raises(ValueError, match="0-indexed"):
        fit_gbdt_multiclass(df, ["x0"], "y")


def test_multiclass_k2_rejected_by_fit_multiclass(tmp_path):
    """fit_gbdt_multiclass с K=2 — error (use fit_gbdt_classifier)."""
    n = 100
    rng = np.random.default_rng(0)
    df = make_df(tmp_path, {
        "x0": rng.normal(size=n).tolist(),
        "y": [i % 2 for i in range(n)],  # {0, 1}
    })
    with pytest.raises(ValueError, match="binary"):
        fit_gbdt_multiclass(df, ["x0"], "y")


# ---------- memory guard ----------

def test_multiclass_memory_guard(tmp_path):
    n = 10_000
    data, X, y = _synth_multiclass(n, n_classes=3, seed=11)
    df = make_df(tmp_path, data, chunk=1000)
    df._memory_guard = MemoryGuard(max_memory_bytes=100_000)
    with pytest.raises(MemoryLimitExceeded, match="state"):
        fit_gbdt_multiclass(df, ["x0", "x1"], "y", n_estimators=5)


# ---------- convenience / to_dict ----------

def test_fit_gbdt_multiclass_dataframe_method(tmp_path):
    n = 200
    data, X, y = _synth_multiclass(n, n_classes=3, seed=12)
    df = make_df(tmp_path, data)
    model = df.fit_gbdt_multiclass(["x0", "x1"], "y", n_estimators=5)
    assert isinstance(model, GBDTMulticlassClassifier)


def test_multiclass_to_dict(tmp_path):
    n = 200
    data, X, y = _synth_multiclass(n, n_classes=3, seed=13)
    df = make_df(tmp_path, data)
    model = fit_gbdt_multiclass(
        df, ["x0", "x1"], "y", n_estimators=5,
    )
    d = model.to_dict()
    assert d["loss"] == "multiclass_softmax"
    assert d["n_classes"] == 3
    assert d["n_trees_per_round"] == 3


# ---------- binary regression ----------

def test_binary_classifier_still_works(tmp_path):
    """GBDTClassifier (K=2) not broken by v0.7.8."""
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    y = (x > 0).astype(np.int64)
    df = make_df(tmp_path, {"x0": x.tolist(), "y": y.tolist()})
    model = fit_gbdt_classifier(df, ["x0"], "y", n_estimators=20)
    proba = model.predict_proba(np.column_stack([x]))
    assert proba.shape == (n,)  # binary stays (n,)
    pred = (proba >= 0.5).astype(int)
    assert (pred == y).mean() > 0.85
