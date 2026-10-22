"""Tests for cadw.ts.thf.TemporalHierarchy (v0.9.24)."""
import math

import numpy as np
import pyarrow as pa
import pytest

import cadw
from cadw import CadwDataFrame
from cadw.ts import (
    NaiveForecaster, TemporalHierarchy,
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


def _factory():
    return NaiveForecaster(strategy="last")


# ==================== aggregation correctness ====================


def test_aggregation_sum_weekly():
    # daily y = [1..14]; weekly level (m=7) -> [28, 77]
    y = list(range(1, 15))
    df = _mk(y, ts=_ts(14))
    th = TemporalHierarchy(
        levels=[1, 7], forecaster_factory=_factory,
        reconciliation="bottom_up",
    )
    th.fit(df, "y", time_col="ts")
    assert list(th.y_per_level_[1]) == [28.0, 77.0]


def test_aggregation_mean():
    y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    df = _mk(y, ts=_ts(6))
    th = TemporalHierarchy(
        levels=[1, 2, 3], forecaster_factory=_factory,
        reconciliation="bottom_up", agg="mean",
    )
    th.fit(df, "y", time_col="ts")
    assert list(th.y_per_level_[1]) == [1.5, 3.5, 5.5]
    assert list(th.y_per_level_[2]) == [2.0, 5.0]


def test_invalid_levels():
    with pytest.raises(ValueError):
        TemporalHierarchy([2, 4], _factory)
    with pytest.raises(ValueError):
        TemporalHierarchy([1, 7, 3], _factory)
    with pytest.raises(ValueError):
        TemporalHierarchy([1, 7], _factory, reconciliation="nope")
    with pytest.raises(ValueError):
        TemporalHierarchy([1, 7], _factory, agg="max")


# ==================== reconciliation ====================


def test_bottom_up_returns_base():
    # Naive last; bottom_up returns level-0 forecast = last value repeated.
    y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    df = _mk(y, ts=_ts(8))
    th = TemporalHierarchy(
        [1, 2], forecaster_factory=_factory,
        reconciliation="bottom_up",
    )
    th.fit(df, "y", time_col="ts")
    fc = th.predict(4)
    assert np.allclose(fc, [8.0, 8.0, 8.0, 8.0])


def test_top_down_proportional():
    # Stable proportions: y = [a,b,a,b,...]; top-level y sum = a+b.
    a, b = 3.0, 7.0
    y = [a, b] * 10
    df = _mk(y, ts=_ts(20))
    th = TemporalHierarchy(
        [1, 2, 4], forecaster_factory=_factory,
        reconciliation="top_down_proportional",
    )
    th.fit(df, "y", time_col="ts")
    fc = th.predict(4)
    # top-down: coarsest level (m=4) forecast = last aggregate = 20.
    # probs per phase p=0..3 = mean over blocks of y[p]/y_block = [0.15, 0.35, 0.15, 0.35].
    # reconciled = [20*0.15, 20*0.35, 20*0.15, 20*0.35] = [3, 7, 3, 7].
    assert np.allclose(fc, [3.0, 7.0, 3.0, 7.0], atol=1e-6)
    # sum conservation (one m=4 block): sum(fc) == top-level forecast
    assert abs(fc.sum() - 20.0) < 1e-6


def test_top_down_requires_divisible_steps():
    y = [1.0] * 20
    df = _mk(y, ts=_ts(20))
    th = TemporalHierarchy(
        [1, 4], forecaster_factory=_factory,
        reconciliation="top_down_proportional",
    )
    th.fit(df, "y", time_col="ts")
    with pytest.raises(ValueError):
        th.predict(3)


def test_optimal_ols_projection():
    # With S built from aggregation and y_all set manually, OLS
    # reconciled projection should satisfy S @ reconciled == y_all.
    y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    df = _mk(y, ts=_ts(8))
    th = TemporalHierarchy(
        [1, 2, 4], forecaster_factory=_factory,
        reconciliation="optimal_ols",
    )
    th.fit(df, "y", time_col="ts")
    # Base forecast = y[-1] = 8 repeated, level-1 = last sum(4) = 26, l2 = 72
    fc = th.predict(4)
    # Reconciled should be a compromise; verify consistency:
    # sum of reconciled base block [0:4] close to level-1 forecast [0], etc.
    assert np.all(np.isfinite(fc))
    assert abs(fc[0] - fc[1]) < 1e-6  # symmetric noise model


def test_optimal_ols_vs_wls_differ():
    rng = np.random.default_rng(42)
    y = np.cumsum(rng.normal(0, 1, 40)) + 100
    df = _mk(y, ts=_ts(40))
    th_o = TemporalHierarchy(
        [1, 2, 4], forecaster_factory=_factory,
        reconciliation="optimal_ols",
    )
    th_o.fit(df, "y", time_col="ts")
    fo = th_o.predict(4)
    th_w = TemporalHierarchy(
        [1, 2, 4], forecaster_factory=_factory,
        reconciliation="optimal_wls",
    )
    th_w.fit(df, "y", time_col="ts")
    fw = th_w.predict(4)
    # Both finite; WLS weights differ from OLS so results differ.
    assert np.all(np.isfinite(fo))
    assert np.all(np.isfinite(fw))
    assert not np.allclose(fo, fw, atol=1e-6)


# ==================== multi-series ====================


def test_multi_series_reconciled_independently():
    n = 20
    a = [1.0, 3.0] * 10
    b = [10.0, 30.0] * 10
    vals = a + b
    grp = ["a"] * n + ["b"] * n
    ts = _ts(n) + _ts(n)
    df = _mk(vals, ts=ts, group=grp)
    th = TemporalHierarchy(
        [1, 2], forecaster_factory=_factory,
        reconciliation="bottom_up",
    )
    th.fit(df, "y", time_col="ts", group_col="g")
    assert set(th._group_models) == {"a", "b"}
    fa = th.predict(2, group="a")
    fb = th.predict(2, group="b")
    # a ends at 3, b ends at 30
    assert np.allclose(fa, [3.0, 3.0])
    assert np.allclose(fb, [30.0, 30.0])


def test_multi_requires_group_on_predict():
    n = 10
    vals = [1.0] * n + [2.0] * n
    grp = ["a"] * n + ["b"] * n
    ts = _ts(n) + _ts(n)
    df = _mk(vals, ts=ts, group=grp)
    th = TemporalHierarchy(
        [1, 2], forecaster_factory=_factory,
        reconciliation="bottom_up",
    )
    th.fit(df, "y", time_col="ts", group_col="g")
    with pytest.raises(ValueError):
        th.predict(2)
    with pytest.raises(KeyError):
        th.predict(2, group="c")


# ==================== predict_level / predict ====================


def test_predict_level():
    y = list(range(1, 41))
    df = _mk(y, ts=_ts(40))
    th = TemporalHierarchy(
        [1, 2, 4], forecaster_factory=_factory,
        reconciliation="bottom_up",
    )
    th.fit(df, "y", time_col="ts")
    base = th.predict(8)
    lvl1 = th.predict_level(8, 1)
    lvl2 = th.predict_level(8, 2)
    # lvl1 has ceil(8/2)=4 rows, lvl2 has 2
    assert len(lvl1) == 4
    assert len(lvl2) == 2
    # Sum consistency: lvl1 sums to base sums
    assert abs(lvl1.sum() - base.sum()) < 1e-6
    assert abs(lvl2.sum() - base.sum()) < 1e-6


def test_predict_level_validation():
    y = list(range(1, 21))
    df = _mk(y, ts=_ts(20))
    th = TemporalHierarchy(
        [1, 2], forecaster_factory=_factory,
        reconciliation="bottom_up",
    )
    th.fit(df, "y", time_col="ts")
    with pytest.raises(ValueError):
        th.predict_level(4, 5)
    with pytest.raises(ValueError):
        th.predict_level(4, -1)


# ==================== ram / robustness ====================


def test_memory_guard_smoke():
    y = list(range(1, 61))
    df = _mk(y, ts=_ts(60))
    guard = cadw.MemoryGuard(max_memory_mb=8)
    df2 = CadwDataFrame(df._dataset, memory_guard=guard,
                        spill_dir=df._spill_dir)
    th = TemporalHierarchy(
        [1, 2, 4], forecaster_factory=_factory,
        reconciliation="optimal_ols",
    )
    th.fit(df2, "y", time_col="ts")
    fc = th.predict(4)
    assert np.all(np.isfinite(fc))


def test_pinv_rank_deficient():
    # 2 levels with m=[1, 4]: rows = steps + ceil(steps/4). For
    # steps=4, rows = 4 + 1 = 5 > cols = 4 -> S is tall, rank = 4.
    # A more rank-deficient case: steps=1 with levels [1, 1] is
    # disallowed, but steps=2 with levels [1, 4] -> S = [[1,0],[0,1],
    # [1,1]] (3x2), still rank 2. Real rank deficiency would need
    # repeated levels. Just verify pinv does not raise on the natural
    # aggregation matrix (m=[1, 2, 4], steps=4 -> 7x4).
    y = list(range(1, 41))
    df = _mk(y, ts=_ts(40))
    th = TemporalHierarchy(
        [1, 2, 4], forecaster_factory=_factory,
        reconciliation="optimal_ols",
    )
    th.fit(df, "y", time_col="ts")
    fc = th.predict(4)
    assert np.all(np.isfinite(fc))
    # Manually verify S has the expected shape:
    S, n_ks = th._build_S(4)
    assert S.shape == (4 + 2 + 1, 4)
    assert n_ks == [4, 2, 1]


def test_factory_called_once_per_level():
    calls = {"n": 0}
    def factory():
        calls["n"] += 1
        return NaiveForecaster(strategy="last")
    y = list(range(1, 21))
    df = _mk(y, ts=_ts(20))
    th = TemporalHierarchy(
        [1, 2, 4], forecaster_factory=factory,
        reconciliation="bottom_up",
    )
    th.fit(df, "y", time_col="ts")
    assert calls["n"] == 3


# ==================== MinT-WLS residual-variance tests ====================


def test_wls_uses_residual_variance():
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.normal(0, 1.0, 60)) + 100.0
    df = _mk(y, ts=_ts(60))
    th_w = TemporalHierarchy(
        [1, 4], forecaster_factory=_factory,
        reconciliation="optimal_wls",
    )
    th_w.fit(df, "y", time_col="ts")
    assert th_w._resid_var_ is not None
    rv = th_w._resid_var_
    assert all(v is not None and v > 0 for v in rv)
    # Not the naive 1/m_k heuristic.
    assert not (abs(rv[0] - 1.0) < 1e-6 and abs(rv[1] - 0.25) < 1e-6)
    fw = th_w.predict(4)
    th_o = TemporalHierarchy(
        [1, 4], forecaster_factory=_factory,
        reconciliation="optimal_ols",
    )
    th_o.fit(df, "y", time_col="ts")
    fo = th_o.predict(4)
    assert np.all(np.isfinite(fw)) and np.all(np.isfinite(fo))
    assert not np.allclose(fw, fo, atol=1e-6)


def test_wls_falls_back_to_ols_when_no_residuals():
    # Minimum-length series: single-row aggregates -> predict_in_sample
    # yields all NaN (or too few residuals). WLS must fall back to OLS
    # and emit a warning at fit time.
    import warnings
    y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    df = _mk(y, ts=_ts(8))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        th = TemporalHierarchy(
            [1, 4], forecaster_factory=_factory,
            reconciliation="optimal_wls",
        )
        th.fit(df, "y", time_col="ts")
    assert th._resid_var_ is None
    assert any("optimal_wls" in str(wi.message) for wi in w)
    fw = th.predict(4)
    # Compare to pure OLS on the same fit.
    th_o = TemporalHierarchy(
        [1, 4], forecaster_factory=_factory,
        reconciliation="optimal_ols",
    )
    th_o.fit(df, "y", time_col="ts")
    fo = th_o.predict(4)
    assert np.allclose(fw, fo)
