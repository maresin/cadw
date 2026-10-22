"""Tests for cadw.ts.backtest (v0.9.24)."""
import math
import warnings

import numpy as np
import pyarrow as pa
import pytest

import cadw
from cadw import CadwDataFrame
from cadw.ts import (
    NaiveForecaster, SARIMAForecaster, GBDTForecaster,
    backtest_forecaster,
)


def _mk(vals, ts=None, group=None):
    cols = {"y": pa.array([float(v) for v in vals], type=pa.float64())}
    if ts is not None:
        cols["ts"] = pa.array(ts, type=pa.timestamp("ns"))
    if group is not None:
        cols["g"] = pa.array(group, type=pa.string())
    return CadwDataFrame.from_table(pa.table(cols))


def _ts(n):
    return [1700000000000000000 + i * 3600 * 10**9 for i in range(n)]


def _summ(summ, group, metric):
    for r in summ:
        if r["group"] == group and r["metric"] == metric:
            return r
    return None


def test_expanding_folds_count_and_boundaries():
    y = list(range(1, 41))  # n=40
    df = _mk(y, ts=_ts(40))
    f = NaiveForecaster(strategy="last")
    summ, folds = backtest_forecaster(
        df, f, target_col="y", time_col="ts",
        horizon=2, n_folds=3, strategy="expanding",
        train_size=10, step=2, metrics=("mae",),
    )
    # folds at (te=10,12), (12,14), (14,16)
    assert len(folds) == 3
    assert folds[0]["n_train"] == 10
    assert folds[0]["n_test"] == 2
    assert folds[1]["n_train"] == 12
    assert folds[2]["n_train"] == 14
    r = _summ(summ, None, "mae")
    assert r is not None and r["n_folds"] == 3


def test_sliding_fixed_train_size():
    y = list(range(1, 41))
    df = _mk(y, ts=_ts(40))
    f = NaiveForecaster(strategy="last")
    summ, folds = backtest_forecaster(
        df, f, target_col="y", time_col="ts",
        horizon=1, n_folds=3, strategy="sliding",
        train_size=10, step=1, metrics=("mae",),
    )
    assert len(folds) == 3
    assert all(r["n_train"] == 10 for r in folds)


def test_metrics_hand_computable():
    # Naive last on monotone series: fc = y[-1], actual = y[-1]+1
    # error = 1 always. MAE=1.0. MASE(m=1) uses scale = mean(|dy|) = 1.
    # -> MASE = 1.0.
    y = list(range(1, 31))
    df = _mk(y, ts=_ts(30))
    f = NaiveForecaster(strategy="last")
    summ, _ = backtest_forecaster(
        df, f, target_col="y", time_col="ts",
        horizon=1, n_folds=3, strategy="expanding",
        train_size=10, step=1,
        metrics=("mae", "rmse", "mase", "mape", "smape"),
        seasonal_period=1,
    )
    for m, expect in (("mae", 1.0), ("rmse", 1.0), ("mase", 1.0)):
        r = _summ(summ, None, m)
        assert abs(r["mean"] - expect) < 1e-9, f"{m} = {r['mean']}"


def test_mape_zero_in_test_returns_nan():
    # Test window hits a zero -> MAPE must be NaN, not inf.
    y = [1.0, 2.0, 3.0, 0.0, 5.0, 6.0]
    df = _mk(y, ts=_ts(6))
    f = NaiveForecaster(strategy="last")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summ, _ = backtest_forecaster(
            df, f, target_col="y", time_col="ts",
            horizon=1, n_folds=1, strategy="expanding",
            train_size=3, step=1, metrics=("mape", "smape"),
        )
    r = _summ(summ, None, "mape")
    assert math.isnan(r["mean"])
    r2 = _summ(summ, None, "smape")
    # sMAPE should be finite (denom = |y|+|f| = 3)
    assert not math.isnan(r2["mean"])


def test_mase_seasonal_period_differs():
    # y = [1,2,3,...,10]; m=1 scale=1, m=2 scale=2 -> MASE(m=2) < MASE(m=1)
    y = list(range(1, 31))
    df = _mk(y, ts=_ts(30))
    f = NaiveForecaster(strategy="last")
    summ1, _ = backtest_forecaster(
        df, f, target_col="y", time_col="ts",
        horizon=1, n_folds=3, strategy="expanding", train_size=10,
        step=1, metrics=("mase",), seasonal_period=1,
    )
    summ2, _ = backtest_forecaster(
        df, f, target_col="y", time_col="ts",
        horizon=1, n_folds=3, strategy="expanding", train_size=10,
        step=1, metrics=("mase",), seasonal_period=2,
    )
    m1 = _summ(summ1, None, "mase")["mean"]
    m2 = _summ(summ2, None, "mase")["mean"]
    # scale(m=1)=1, scale(m=2)=2, errors=1
    assert abs(m1 - 1.0) < 1e-9
    assert abs(m2 - 0.5) < 1e-9


def test_multi_series_summary_per_group():
    n = 30
    vals = list(range(1, n + 1)) * 2
    grp = ["a"] * n + ["b"] * n
    ts = _ts(n) + _ts(n)
    df = _mk(vals, ts=ts, group=grp)
    f = NaiveForecaster(strategy="last")
    summ, _ = backtest_forecaster(
        df, f, target_col="y", time_col="ts", group_col="g",
        horizon=1, n_folds=2, strategy="expanding",
        train_size=10, step=1, metrics=("mae",),
    )
    grps = {r["group"] for r in summ}
    assert grps == {"a", "b"}


def test_short_group_skipped():
    n = 30
    vals = list(range(1, n + 1)) + list(range(1, 5))
    grp = ["a"] * n + ["b"] * 4
    ts = _ts(n) + _ts(4)
    df = _mk(vals, ts=ts, group=grp)
    f = NaiveForecaster(strategy="last")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        summ, folds = backtest_forecaster(
            df, f, target_col="y", time_col="ts", group_col="g",
            horizon=1, n_folds=2, strategy="expanding",
            train_size=10, step=1, metrics=("mae",),
        )
    # 'b' skipped -> folds row with metric SKIPPED
    assert any(r["metric"] == "SKIPPED" and r["group"] == "b" for r in folds)
    # summ contains only 'a' rows
    assert all(r["group"] == "a" for r in summ)
    assert any("skipped" in str(wi.message) for wi in w)


def test_clone_independence():
    # Spy forecaster that sets a fitted marker only in fit; each fold
    # must clone fresh -> marker never accumulates.
    from cadw.ts.models import ForecasterBase
    class Spy(ForecasterBase):
        counter = 0
        def fit(self, df, target_col, *, time_col=None, group_col=None):
            Spy.counter += 1
            self._is_fitted = True
            self.my_id = Spy.counter
            return self
        def _predict(self, steps):
            return np.zeros(steps, dtype=np.float64)
        def _predict_in_sample(self):
            return np.array([])
    Spy.counter = 0
    y = list(range(1, 21))
    df = _mk(y, ts=_ts(20))
    summ, _ = backtest_forecaster(
        df, Spy(), target_col="y", time_col="ts",
        horizon=1, n_folds=3, strategy="expanding",
        train_size=8, step=1, metrics=("mae",),
    )
    # Each fold fit its own instance -> counter increments per fold.
    assert Spy.counter == 3


def test_invalid_strategy_raises():
    y = list(range(1, 20))
    df = _mk(y, ts=_ts(19))
    f = NaiveForecaster()
    with pytest.raises(ValueError):
        backtest_forecaster(
            df, f, target_col="y", time_col="ts",
            horizon=1, n_folds=2, strategy="nope",
        )


def test_n_folds_lt_1_raises():
    y = list(range(1, 20))
    df = _mk(y, ts=_ts(19))
    f = NaiveForecaster()
    with pytest.raises(ValueError):
        backtest_forecaster(
            df, f, target_col="y", time_col="ts",
            horizon=1, n_folds=0, strategy="expanding",
        )


def test_memory_guard_smoke():
    n = 60
    y = list(range(1, n + 1))
    df = _mk(y, ts=_ts(n))
    guard = cadw.MemoryGuard(max_memory_mb=8)
    df2 = CadwDataFrame(df._dataset, memory_guard=guard,
                        spill_dir=df._spill_dir)
    f = NaiveForecaster(strategy="last")
    summ, _ = backtest_forecaster(
        df2, f, target_col="y", time_col="ts",
        horizon=1, n_folds=3, strategy="expanding",
        train_size=10, step=1, metrics=("mae",),
    )
    r = _summ(summ, None, "mae")
    assert r is not None
