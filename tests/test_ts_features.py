"""Tests for cadw.ts.features (v0.9.24)."""
import numpy as np
import pytest

import cadw
from cadw import CadwDataFrame
from cadw.ts import (
    lag_features, lead_features, diff_features,
    calendar_features, cyclical_features, fourier_features,
    holiday_features, make_features,
)


def _make_panel(n_groups=3, n_per=10, start_ts=1_700_000_000_000_000_000):
    rows = []
    for g in range(n_groups):
        for t in range(n_per):
            rows.append({
                "g": f"g{g}",
                "ts": start_ts + t * 3600 * 1_000_000_000,
                "y": float(g * 100 + t),
                "x": float(t),
            })
    import pyarrow as pa
    schema = pa.schema([
        ("g", pa.string()),
        ("ts", pa.timestamp("ns")),
        ("y", pa.float64()),
        ("x", pa.float64()),
    ])
    arrays = {c: [r[c] for r in rows] for c in ("g", "ts", "y", "x")}
    table = pa.table({c: pa.array(arrays[c], type=schema.field(c).type)
                       for c in ("g", "ts", "y", "x")})
    return CadwDataFrame.from_table(table)


def test_lag_basic():
    df = _make_panel(n_groups=1, n_per=5)
    out = lag_features(df, "y", lags=(1, 2))
    assert "y_lag1" in out.columns
    assert "y_lag2" in out.columns
    # batch=10k, 5 rows -> single batch, positional
    vals = [b for b in out.to_batches(format="numpy", batch_size=10)]
    ys = []
    for b in vals:
        ys.extend(b["y_lag1"].tolist())
    assert np.isnan(ys[0])
    assert ys[1] == 0.0
    assert ys[4] == 3.0


def test_lag_validation():
    df = _make_panel(n_groups=1, n_per=5)
    with pytest.raises(ValueError):
        lag_features(df, "y", lags=(0,))
    with pytest.raises(KeyError):
        lag_features(df, "nope", lags=(1,))


def test_lag_group_no_leak():
    # Critical: group-aware lag must never cross group boundary.
    df = _make_panel(n_groups=3, n_per=5)
    out = lag_features(df, "y", lags=(1,), group_col="g", time_col="ts")
    # After sort by (g, ts): rows are g0 x5, g1 x5, g2 x5. First row of
    # each group must be NaN, regardless of the other groups.
    ys, gs = [], []
    for b in out.to_batches(format="numpy", batch_size=100):
        ys.extend(b["y_lag1"].tolist())
        gs.extend(b["g"].tolist())
    assert len(ys) == 15
    for g in ("g0", "g1", "g2"):
        gi = [i for i, v in enumerate(gs) if v == g]
        assert np.isnan(ys[gi[0]]), f"first row of {g} should be NaN"


def test_lead_basic():
    df = _make_panel(n_groups=1, n_per=5)
    out = lead_features(df, "y", leads=(1,), time_col="ts")
    assert "y_lead1" in out.columns
    vals = []
    for b in out.to_batches(format="numpy", batch_size=100):
        vals.extend(b["y_lead1"].tolist())
    assert vals[0] == 1.0
    assert np.isnan(vals[4])


def test_diff_basic():
    df = _make_panel(n_groups=1, n_per=5)
    out = diff_features(df, "y", lags=(1,))
    vals = []
    for b in out.to_batches(format="numpy", batch_size=100):
        vals.extend(b["y_diff1"].tolist())
    assert np.isnan(vals[0])
    for i in range(1, 5):
        assert abs(vals[i] - 1.0) < 1e-9


def test_calendar_basic():
    df = _make_panel(n_groups=1, n_per=5)
    out = calendar_features(df, "ts", ("year", "month", "weekday"))
    assert "ts_year" in out.columns
    assert "ts_weekday" in out.columns
    # all values non-null
    for b in out.to_batches(format="numpy", batch_size=100):
        for name in ("ts_year", "ts_month", "ts_weekday"):
            assert all(v is not None for v in b[name].tolist())


def test_calendar_rejects_numeric():
    df = _make_panel(n_groups=1, n_per=5)
    with pytest.raises(TypeError):
        calendar_features(df, "y", ("year",))


def test_cyclical_range():
    df = _make_panel(n_groups=1, n_per=24)
    out = cyclical_features(df, "ts", ("hour",))
    for b in out.to_batches(format="numpy", batch_size=100):
        s = b["ts_hour_sin"].tolist()
        c = b["ts_hour_cos"].tolist()
        for v in s + c:
            assert -1.0 - 1e-9 <= v <= 1.0 + 1e-9


def test_fourier_pure_sin():
    # With period=4 and hourly ts, sin(2*pi*t/4) is known for integer t.
    df = _make_panel(n_groups=1, n_per=8, start_ts=0)
    out = fourier_features(df, "ts", period=4, order=1, prefix="f")
    # ts is timestamp(ns); secs = t*3600. We need period=4*3600 for clean.
    out = fourier_features(df, "ts", period=4 * 3600, order=1, prefix="f2")
    vals = []
    for b in out.to_batches(format="numpy", batch_size=100):
        vals.extend(b["f2_sin1"].tolist())
    # sin(2*pi*t/4): t=0 -> 0; t=1 -> 1; t=2 -> 0; t=3 -> -1
    assert abs(vals[0]) < 1e-9
    assert abs(vals[1] - 1.0) < 1e-9
    assert abs(vals[2]) < 1e-9
    assert abs(vals[3] + 1.0) < 1e-9


def test_holiday_weekend():
    # 2023-11-04 is a Saturday; 11-05 Sunday; 11-06 Monday.
    import pyarrow as pa
    ts = [
        1699056000 * 1_000_000_000,  # 2023-11-04
        1699142400 * 1_000_000_000,  # 2023-11-05
        1699228800 * 1_000_000_000,  # 2023-11-06
    ]
    table = pa.table({
        "ts": pa.array(ts, type=pa.timestamp("ns")),
    })
    df = CadwDataFrame.from_table(table)
    out = holiday_features(df, "ts")
    vals = []
    for b in out.to_batches(format="numpy", batch_size=10):
        vals.extend(b["ts_is_weekend"].tolist())
    assert vals == [True, True, False]


def test_make_features_pipeline():
    df = _make_panel(n_groups=2, n_per=20)
    out = make_features(
        df, target_col="y", ts_col="ts",
        lags=(1, 2), diffs=(1,),
        group_col="g",
        fourier=(24,), fourier_order=2,
    )
    for name in ("ts_month", "ts_weekday", "ts_month_sin",
                 "ts_f24_sin1", "ts_f24_cos2",
                 "ts_is_weekend", "y_lag1", "y_lag2", "y_diff1"):
        assert name in out.columns, f"missing {name}"


def test_memory_guard_used():
    # Small budget to force MemoryGuard checks on every batch.
    df = _make_panel(n_groups=2, n_per=100)
    guard = cadw.MemoryGuard(max_memory_mb=8)
    df2 = CadwDataFrame(df._dataset, memory_guard=guard,
                        spill_dir=df._spill_dir)
    out = lag_features(df2, "y", lags=(1, 2, 3, 4, 5))
    # If no exception and we can read it, MemoryGuard passed.
    n = sum(b.num_rows for b in out._iter_arrow_batches(batch_size=50))
    assert n == 200
