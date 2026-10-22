"""Tests for cadw.ts.stats.rolling_stats (v0.9.24)."""
import math

import numpy as np
import pyarrow as pa
import pytest

import cadw
from cadw import CadwDataFrame
from cadw.ts import rolling_stats


def _mk(vals, group=None):
    cols = {"y": pa.array([float(v) if v is not None else None
                           for v in vals], type=pa.float64())}
    if group is not None:
        cols["g"] = pa.array(group, type=pa.string())
    return CadwDataFrame.from_table(pa.table(cols))


def _get(out, col):
    res = []
    for b in out.to_batches(format="numpy", batch_size=100):
        res.extend(b[col].tolist())
    return res


def test_mean_std_basic():
    df = _mk([1, 2, 3, 4, 5])
    out = rolling_stats(df, "y", windows=(3,), stats=("mean", "std"))
    m = _get(out, "y_roll3_mean")
    s = _get(out, "y_roll3_std")
    assert math.isnan(m[0])
    assert math.isnan(m[1])
    assert abs(m[2] - 2.0) < 1e-9
    assert abs(m[3] - 3.0) < 1e-9
    assert abs(m[4] - 4.0) < 1e-9
    assert math.isnan(s[0])
    assert abs(s[2] - 1.0) < 1e-9


def test_multiple_windows():
    df = _mk([1, 2, 3, 4])
    out = rolling_stats(df, "y", windows=(2, 3), stats=("mean",))
    m2 = _get(out, "y_roll2_mean")
    m3 = _get(out, "y_roll3_mean")
    assert math.isnan(m2[0])
    assert abs(m2[1] - 1.5) < 1e-9
    assert math.isnan(m3[1])
    assert abs(m3[2] - 2.0) < 1e-9


def test_quantiles():
    df = _mk([1, 2, 3, 4, 5])
    out = rolling_stats(
        df, "y", windows=(5,),
        stats=("q10", "q25", "q50", "q75", "q90"),
    )
    q50 = _get(out, "y_roll5_q50")[4]
    assert abs(q50 - 3.0) < 1e-9
    q10 = _get(out, "y_roll5_q10")[4]
    assert q10 < q50


def test_skew_kurt_constant_returns_nan():
    df = _mk([5, 5, 5, 5, 5])
    out = rolling_stats(df, "y", windows=(3,), stats=("skew", "kurt"))
    sk = _get(out, "y_roll3_skew")
    ku = _get(out, "y_roll3_kurt")
    for v in sk[2:] + ku[3:]:
        assert math.isnan(v)


def test_energy_and_crossings():
    df = _mk([1, 2, 1, 2, 1])
    out = rolling_stats(
        df, "y", windows=(3,),
        stats=("energy", "crossings"),
    )
    e = _get(out, "y_roll3_energy")
    c = _get(out, "y_roll3_crossings")
    assert abs(e[2] - 2.0) < 1e-9
    assert c[2] == 2.0


def test_slope():
    df = _mk([1, 2, 3, 4, 5])
    out = rolling_stats(df, "y", windows=(4,), stats=("slope",))
    sl = _get(out, "y_roll4_slope")
    assert abs(sl[3] - 1.0) < 1e-9


def test_autocorr():
    df = _mk([1, 2, 3, 4, 5, 6, 7, 8])
    out = rolling_stats(
        df, "y", windows=(6,), stats=("autocorr_lag1",),
    )
    ac = _get(out, "y_roll6_autocorr_lag1")
    assert abs(ac[5] - 1.0) < 1e-9


def test_group_no_leak():
    df = _mk([1, 2, 3, 10, 20, 30], group=["a", "a", "a", "b", "b", "b"])
    out = rolling_stats(
        df, "y", windows=(3,), stats=("mean",), group_col="g",
    )
    m = _get(out, "y_roll3_mean")
    assert math.isnan(m[0])
    assert math.isnan(m[1])
    assert abs(m[2] - 2.0) < 1e-9
    assert math.isnan(m[3])
    assert math.isnan(m[4])
    assert abs(m[5] - 20.0) < 1e-9


def test_nan_handling():
    df = _mk([1, None, 3, None, 5])
    out = rolling_stats(df, "y", windows=(3,), stats=("mean",))
    m = _get(out, "y_roll3_mean")
    assert math.isnan(m[2])
    assert math.isnan(m[3])


def test_min_periods():
    df = _mk([1, 2, 3, 4])
    out = rolling_stats(
        df, "y", windows=(3,), stats=("mean",), min_periods=1,
    )
    m = _get(out, "y_roll3_mean")
    assert abs(m[0] - 1.0) < 1e-9
    assert abs(m[1] - 1.5) < 1e-9
    assert abs(m[2] - 2.0) < 1e-9


def test_validation():
    df = _mk([1, 2, 3])
    with pytest.raises(ValueError):
        rolling_stats(df, "y", windows=(0,))
    with pytest.raises(ValueError):
        rolling_stats(df, "y", windows=(2,), stats=("nope",))
    with pytest.raises(KeyError):
        rolling_stats(df, "nope", windows=(2,))
    with pytest.raises(ValueError):
        rolling_stats(df, "y", windows=(2,), stats=("autocorr_lag2",))


def test_memory_guard_budget():
    df = _mk(list(range(100)))
    guard = cadw.MemoryGuard(max_memory_mb=8)
    df2 = CadwDataFrame(df._dataset, memory_guard=guard,
                        spill_dir=df._spill_dir)
    out = rolling_stats(
        df2, "y", windows=(5, 10),
        stats=("mean", "std", "q50", "autocorr_lag1"),
    )
    n = sum(b.num_rows for b in out._iter_arrow_batches(batch_size=50))
    assert n == 100


def test_single_row():
    df = _mk([7.0])
    out = rolling_stats(df, "y", windows=(3,), stats=("mean",))
    m = _get(out, "y_roll3_mean")
    assert math.isnan(m[0])


def test_xcorr_identity():
    from cadw.ts import rolling_crosscorr
    df = _mk([1, 2, 3, 4, 5], group=None)
    # add a second numeric column x = y
    import pyarrow as pa
    t = df.to_batches(format="numpy", batch_size=100)
    # simpler: build from table
    arr = [1., 2., 3., 4., 5.]
    table = pa.table({
        "x": pa.array(arr, type=pa.float64()),
        "y": pa.array(arr, type=pa.float64()),
    })
    df2 = CadwDataFrame.from_table(table)
    out = rolling_crosscorr(df2, "x", "y", windows=(5,), lags=(0,))
    vals = _get(out, "x__y_roll5_xcorr_lag0")
    assert abs(vals[4] - 1.0) < 1e-9


def test_xcorr_phase_shift():
    from cadw.ts import rolling_crosscorr
    import pyarrow as pa
    # y[t] = x[t-1] -> corr(x[t], y[t-1]) at lag 1 should be ~1.
    x = [1., 2., 3., 4., 5., 6., 7., 8.]
    y = [0., 1., 2., 3., 4., 5., 6., 7.]
    df = CadwDataFrame.from_table(pa.table({
        "x": pa.array(x, type=pa.float64()),
        "y": pa.array(y, type=pa.float64()),
    }))
    out = rolling_crosscorr(df, "x", "y", windows=(6,), lags=(0, 1))
    l0 = _get(out, "x__y_roll6_xcorr_lag0")
    l1 = _get(out, "x__y_roll6_xcorr_lag1")
    # At lag 1, the pairs are (x[t], y[t-1]) = (k, k-2) -> strong linear.
    assert abs(l1[-1] - 1.0) < 1e-6


def test_xcorr_constant_nan():
    from cadw.ts import rolling_crosscorr
    import pyarrow as pa
    df = CadwDataFrame.from_table(pa.table({
        "x": pa.array([5., 5., 5., 5.], type=pa.float64()),
        "y": pa.array([1., 2., 3., 4.], type=pa.float64()),
    }))
    out = rolling_crosscorr(df, "x", "y", windows=(3,), lags=(0,))
    vals = _get(out, "x__y_roll3_xcorr_lag0")
    for v in vals[2:]:
        assert math.isnan(v)


def test_xcorr_group_no_leak():
    from cadw.ts import rolling_crosscorr
    import pyarrow as pa
    df = CadwDataFrame.from_table(pa.table({
        "x": pa.array([1., 2., 3., 10., 20., 30.], type=pa.float64()),
        "y": pa.array([1., 2., 3., 10., 20., 30.], type=pa.float64()),
        "g": pa.array(["a", "a", "a", "b", "b", "b"], type=pa.string()),
    }))
    out = rolling_crosscorr(
        df, "x", "y", windows=(3,), lags=(0,), group_col="g",
    )
    vals = _get(out, "x__y_roll3_xcorr_lag0")
    assert math.isnan(vals[0]) and math.isnan(vals[1])
    assert abs(vals[2] - 1.0) < 1e-9
    assert math.isnan(vals[3]) and math.isnan(vals[4])
    assert abs(vals[5] - 1.0) < 1e-9


def test_xcorr_memory_guard():
    from cadw.ts import rolling_crosscorr
    import pyarrow as pa
    n = 100
    xs = [float(i) for i in range(n)]
    ys = [float(i) for i in range(n)]
    df = CadwDataFrame.from_table(pa.table({
        "x": pa.array(xs, type=pa.float64()),
        "y": pa.array(ys, type=pa.float64()),
    }))
    guard = cadw.MemoryGuard(max_memory_mb=8)
    df2 = CadwDataFrame(df._dataset, memory_guard=guard,
                        spill_dir=df._spill_dir)
    out = rolling_crosscorr(df2, "x", "y", windows=(5,), lags=(0, 1))
    cnt = sum(b.num_rows for b in out._iter_arrow_batches(batch_size=50))
    assert cnt == n


def test_xcorr_validation():
    from cadw.ts import rolling_crosscorr
    import pyarrow as pa
    df = CadwDataFrame.from_table(pa.table({
        "x": pa.array([1., 2., 3.], type=pa.float64()),
        "y": pa.array([1., 2., 3.], type=pa.float64()),
    }))
    with pytest.raises(KeyError):
        rolling_crosscorr(df, "x", "nope", windows=(3,), lags=(0,))
    with pytest.raises(ValueError):
        rolling_crosscorr(df, "x", "x", windows=(3,), lags=(0,))
    with pytest.raises(ValueError):
        rolling_crosscorr(df, "x", "y", windows=(3,), lags=(3,))
