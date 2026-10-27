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


# ============ v0.9.22 (§13.4/241): multiclass state formula ============


def test_multiclass_est_state_includes_y_all():
    """v0.9.22: est_n_state = n·8·(1 + 4K), not n·K·8·3.

    Prior formula missed y_all (n·8) and K-th round_cur_leaves (K·n·8).
    """
    from cadw.ml.gbdt import _fit_boosting_multiclass
    import inspect
    src = inspect.getsource(_fit_boosting_multiclass)
    assert "n_total * 8 * (1 + 4 * K)" in src, (
        "multiclass state formula must be n·8·(1+4K) "
        "(AUDIT_ML §1.1). Prior n·K·8·3 under-reports."
    )
    assert "n_total * K * 8 * 3" not in src, (
        "old formula n·K·8·3 must be removed"
    )


# v0.9.87-P1a-1a: binning infrastructure tests.
# Verifies _binning_pass + _AllNodeHistOpCodes match _AllNodeHistOp
# behavior, without changing existing fit paths.

def test_binning_pass_correctness(tmp_path):
    """codes from _binning_pass matches reference uniform formula."""
    from cadw.ml.gbdt import _binning_pass
    from cadw.core.memory_guard import MemoryGuard
    n, d, bins = 200, 3, 16
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, d))
    cols = [f"f{j}" for j in range(d)]
    tbl = pa.table({c: pa.array(X[:, j]) for j, c in enumerate(cols)})
    df = CadwDataFrame.from_table(tbl)
    mg = MemoryGuard(max_memory_mb=200)
    mn, mx, range_, codes_mm, nan_mask, spill = _binning_pass(
        df, cols, bins=bins, memory_guard=mg)
    try:
        assert mn.shape == (d,)
        assert mx.shape == (d,)
        assert range_.shape == (d,)
        assert not nan_mask.any()
        expected = np.clip(
            ((X - mn) / range_ * bins).astype(np.int64),
            0, bins - 1).astype(np.uint8)
        np.testing.assert_array_equal(np.asarray(codes_mm), expected)
    finally:
        spill.__exit__(None, None, None)


def test_binning_pass_nan_handling(tmp_path):
    """NaN rows: nan_mask True, codes = 0."""
    from cadw.ml.gbdt import _binning_pass
    from cadw.core.memory_guard import MemoryGuard
    n, d, bins = 100, 2, 8
    rng = np.random.default_rng(1)
    X = rng.standard_normal((n, d))
    X[5, 0] = np.nan
    X[10, 1] = np.nan
    cols = ["f0", "f1"]
    tbl = pa.table({
        "f0": pa.array(X[:, 0], mask=np.isnan(X[:, 0])),
        "f1": pa.array(X[:, 1], mask=np.isnan(X[:, 1])),
    })
    df = CadwDataFrame.from_table(tbl)
    mg = MemoryGuard(max_memory_mb=200)
    mn, mx, range_, codes_mm, nan_mask, spill = _binning_pass(
        df, cols, bins=bins, memory_guard=mg)
    try:
        assert bool(nan_mask[5])
        assert bool(nan_mask[10])
        assert not bool(nan_mask[0])
        codes_arr = np.asarray(codes_mm)
        assert codes_arr[5, 0] == 0
        assert codes_arr[5, 1] == 0
        assert codes_arr[10, 0] == 0
        assert codes_arr[10, 1] == 0
    finally:
        spill.__exit__(None, None, None)


def test_all_node_hist_op_codes_matches_existing():
    """Histogram from codes == histogram from raw X (within 1e-12)."""
    from cadw.ml.gbdt import _AllNodeHistOp, _AllNodeHistOpCodes
    n, d, bins = 300, 3, 16
    rng = np.random.default_rng(2)
    X = rng.standard_normal((n, d))
    g = rng.standard_normal(n)
    h = rng.random(n) + 0.1
    leaf = np.zeros(n, dtype=np.int64)
    cols = [f"f{j}" for j in range(d)]

    batch = pa.RecordBatch.from_arrays(
        [pa.array(X[:, j]) for j in range(d)]
        + [pa.array(g), pa.array(h), pa.array(leaf)],
        names=cols + ["__g", "__h", "__leaf"])
    op_old = _AllNodeHistOp(cols, [0], X.min(axis=0), X.max(axis=0), bins)
    state_old = op_old.init()
    state_old = op_old.update(state_old, batch)
    golden = op_old.finalize(state_old)

    mn = X.min(axis=0)
    mx = X.max(axis=0)
    range_ = np.where(mx - mn > 0, mx - mn, 1.0)
    codes = np.clip(
        ((X - mn) / range_ * bins).astype(np.int64),
        0, bins - 1).astype(np.uint8)
    nan_mask = np.isnan(X).any(axis=1)

    batch_codes = pa.RecordBatch.from_arrays(
        list(batch.columns)
        + [pa.array(codes[:, j], type=pa.uint8()) for j in range(d)]
        + [pa.array(nan_mask)],
        names=cols + ["__g", "__h", "__leaf"]
              + [f"__code_{j}" for j in range(d)] + ["__nan_mask"])
    op_new = _AllNodeHistOpCodes(cols, [0], bins)
    state_new = op_new.init()
    state_new = op_new.update(state_new, batch_codes)
    new = op_new.finalize(state_new)

    np.testing.assert_allclose(new["g_hist"], golden["g_hist"], atol=1e-12)
    np.testing.assert_allclose(new["h_hist"], golden["h_hist"], atol=1e-12)
    np.testing.assert_allclose(new["g_tot"], golden["g_tot"], atol=1e-12)
    np.testing.assert_allclose(new["h_tot"], golden["h_tot"], atol=1e-12)
    np.testing.assert_array_equal(new["n_tot"], golden["n_tot"])


def test_all_node_hist_op_codes_multi_node():
    """Codes-based hist matches existing hist for multiple nodes."""
    from cadw.ml.gbdt import _AllNodeHistOp, _AllNodeHistOpCodes
    n, d, bins = 400, 2, 12
    rng = np.random.default_rng(3)
    X = rng.standard_normal((n, d))
    g = rng.standard_normal(n)
    h = rng.random(n) + 0.1
    leaf = rng.integers(0, 2, size=n, dtype=np.int64)
    cols = ["f0", "f1"]

    batch = pa.RecordBatch.from_arrays(
        [pa.array(X[:, 0]), pa.array(X[:, 1]),
         pa.array(g), pa.array(h), pa.array(leaf)],
        names=cols + ["__g", "__h", "__leaf"])
    op_old = _AllNodeHistOp(
        cols, [0, 1], X.min(axis=0), X.max(axis=0), bins)
    state_old = op_old.init()
    state_old = op_old.update(state_old, batch)
    golden = op_old.finalize(state_old)

    mn = X.min(axis=0)
    mx = X.max(axis=0)
    range_ = np.where(mx - mn > 0, mx - mn, 1.0)
    codes = np.clip(
        ((X - mn) / range_ * bins).astype(np.int64),
        0, bins - 1).astype(np.uint8)
    nan_mask = np.zeros(n, dtype=bool)
    batch_codes = pa.RecordBatch.from_arrays(
        list(batch.columns)
        + [pa.array(codes[:, j], type=pa.uint8()) for j in range(d)]
        + [pa.array(nan_mask)],
        names=cols + ["__g", "__h", "__leaf"]
              + [f"__code_{j}" for j in range(d)] + ["__nan_mask"])
    op_new = _AllNodeHistOpCodes(cols, [0, 1], bins)
    state_new = op_new.init()
    state_new = op_new.update(state_new, batch_codes)
    new = op_new.finalize(state_new)

    np.testing.assert_allclose(new["g_hist"], golden["g_hist"], atol=1e-12)
    np.testing.assert_allclose(new["h_hist"], golden["h_hist"], atol=1e-12)
    np.testing.assert_array_equal(new["n_tot"], golden["n_tot"])

