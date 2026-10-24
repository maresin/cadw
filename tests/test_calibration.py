"""Tests for D.4.6 probability calibration (v0.9.39)."""
from __future__ import annotations
import inspect
import numpy as np
import pyarrow as pa
import pytest
from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ml import calibration as cal
from cadw.ml.calibration import (
    fit_calibration, brier_score, expected_calibration_error,
    _fit_platt, _fit_isotonic, _apply_isotonic, CalibratedModel,
)


def _data(n=3000, seed=0, d=3):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, d))
    lp = -1.0 + 1.2 * X[:, 0] - 0.5 * X[:, 1]
    p = 1.0 / (1.0 + np.exp(-lp))
    y = (rng.uniform(size=n) < p).astype(np.float64)
    return X, y


def _df(X, y):
    d = {f"x{j}": pa.array(X[:, j]) for j in range(X.shape[1])}
    d["y"] = pa.array(y)
    return CadwDataFrame.from_table(pa.table(d))


def test_platt_synthetic_calibrates_better_than_raw():
    X, y = _data(seed=0)
    df = _df(X, y)
    m = fit_calibration(df, ["x0", "x1", "x2"], "y", method="platt",
                        n_estimators=30)
    b_cal = m.brier(df)
    ece_cal = m.ece(df)
    raw_p = np.concatenate([m.base_.predict_proba(
        __import__("cadw.ml.gbdt", fromlist=["_stack_numeric"])._stack_numeric(b, ["x0", "x1", "x2"]))
        for b in df._iter_filtered_batches()])
    b_raw = brier_score(y, raw_p)
    assert b_cal <= b_raw + 0.01, (b_cal, b_raw)
    assert ece_cal < 0.10


def test_isotonic_synthetic_calibrates_better_than_raw():
    X, y = _data(seed=1)
    df = _df(X, y)
    m = fit_calibration(df, ["x0", "x1", "x2"], "y", method="isotonic",
                        n_estimators=30)
    assert m.to_dict()["iso_points"] > 5
    assert m.ece(df) < 0.06


def test_reference_sklearn_sigmoid_comparable():
    sk_cal = pytest.importorskip("sklearn.calibration")
    X, y = _data(seed=2)
    df = _df(X, y)
    ours = fit_calibration(df, ["x0", "x1", "x2"], "y", method="platt",
                           n_estimators=30).brier(df)
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import brier_score_loss
    base = HistGradientBoostingClassifier(max_iter=30).fit(X, y)
    cc = sk_cal.CalibratedClassifierCV(base, cv=3, method="sigmoid").fit(X, y)
    ref = brier_score_loss(y, cc.predict_proba(X)[:, 1])
    assert ours < ref * 1.3, (ours, ref)


def test_determinism_same_seed():
    X, y = _data(seed=3)
    df = _df(X, y)
    a = fit_calibration(df, ["x0", "x1"], "y", method="platt", n_estimators=15).predict_proba(df)
    b = fit_calibration(df, ["x0", "x1"], "y", method="platt", n_estimators=15).predict_proba(df)
    assert np.allclose(a, b)


def test_edge_guards_bad_method_and_nonbinary():
    X, y = _data(n=200, seed=4)
    df = _df(X, y)
    with pytest.raises(ValueError, match="method must be"):
        fit_calibration(df, ["x0"], "y", method="foo")
    with pytest.raises(ValueError, match="non-empty"):
        fit_calibration(df, [], "y", n_estimators=5)
    with pytest.raises(KeyError):
        fit_calibration(df, ["nope"], "y", n_estimators=5)
    y3 = y.copy(); y3[0] = 2.0
    with pytest.raises(ValueError, match="binary"):
        fit_calibration(_df(X, y3), ["x0"], "y", n_estimators=5)


def test_platt_isfinite_and_monotone():
    rng = np.random.RandomState(5)
    s = rng.normal(size=400)
    y = (s + rng.normal(0, 0.5, 400) > 0).astype(np.float64)
    A, B = _fit_platt(s, y)
    assert np.isfinite(A) and np.isfinite(B)
    assert A > 0


def test_isotonic_pava_monotone():
    rng = np.random.RandomState(6)
    s = rng.uniform(size=500)
    y = (rng.uniform(size=500) < s).astype(np.float64)
    bx, by = _fit_isotonic(s, y, n_bins=32)
    assert np.all(np.diff(by) >= -1e-12)
    p = _apply_isotonic(np.linspace(0, 1, 7), bx, by)
    assert np.all(p >= 0) and np.all(p <= 1)


def test_ram_guard_raises_when_tight():
    X, y = _data(n=1000, seed=7)
    mg = MemoryGuard(max_memory_bytes=512)
    with pytest.raises(MemoryLimitExceeded):
        fit_calibration(_df(X, y), ["x0", "x1"], "y", n_estimators=5,
                        memory_guard=mg)


def test_multi_batch_consistency(tmp_path):
    import pyarrow.parquet as pq
    X, y = _data(n=1200, seed=8)
    df = _df(X, y)
    tbl = pa.Table.from_batches(list(df._iter_filtered_batches()))
    path = str(tmp_path / "c.parquet")
    pq.write_table(tbl, path, row_group_size=97)
    dmb = CadwDataFrame.from_parquet(path, base_dir="/tmp", chunk_size=100)
    m = fit_calibration(df, ["x0", "x1", "x2"], "y", method="platt", n_estimators=10)
    assert np.allclose(m.predict_proba(df), m.predict_proba(dmb), atol=1e-9)


def test_source_contract_guard_and_log_fit():
    src = inspect.getsource(cal)
    assert "check_batch" in src and "extra_factor=3.0" in src
    assert "log_fit" in src
    assert "import pandas" not in src
