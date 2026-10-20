"""v0.9.0a: datetime accessor tests."""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard


def make_df(tmp_path, data, chunk=3, name="L"):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def read_col(df, tmp_path, col, name="o.parquet"):
    out = tmp_path / name
    df.write_parquet(out)
    return pq.read_table(out).column(col).to_pylist()


@pytest.fixture
def dt_df(tmp_path):
    ts = pd.to_datetime([
        "2020-01-15 10:30:45",   # Wednesday
        "2020-03-20 14:00:00",   # Friday
        "2020-06-01 00:00:00",   # Monday
        "2020-12-31 23:59:59",   # Thursday
    ])
    return make_df(tmp_path, {"ts": ts.tolist(), "v": [1, 2, 3, 4]}), ts


@pytest.mark.parametrize("feature,pd_attr", [
    ("year", "year"),
    ("month", "month"),
    ("day", "day"),
    ("hour", "hour"),
    ("minute", "minute"),
    ("second", "second"),
    ("day_of_year", "day_of_year"),
    ("quarter", "quarter"),
])
def test_dt_feature_matches_pandas(tmp_path, feature, pd_attr):
    ts = pd.to_datetime([
        "2020-01-15 10:30:45",
        "2020-03-20 14:00:00",
        "2020-06-01 00:00:00",
        "2020-12-31 23:59:59",
    ])
    df = make_df(tmp_path, {"ts": ts.tolist(), "v": [1, 2, 3, 4]})
    result = getattr(df["ts"].dt, feature)
    got = read_col(result, tmp_path, f"ts_{feature}")
    expected = getattr(ts, pd_attr).tolist()
    assert got == expected


def test_dt_weekday_matches_pandas(tmp_path):
    """Pandas weekday: Monday=0..Sunday=6. Arrow pc.day_of_week: same."""
    ts = pd.to_datetime([
        "2020-01-13",   # Monday
        "2020-01-14",   # Tuesday
        "2020-01-19",   # Sunday
    ])
    df = make_df(tmp_path, {"ts": ts.tolist()})
    result = df["ts"].dt.weekday
    got = read_col(result, tmp_path, "ts_weekday")
    expected = ts.weekday.tolist()
    assert got == expected


def test_dt_preserves_original_columns(tmp_path, dt_df):
    df, _ = dt_df
    result = df["ts"].dt.year
    assert "v" in result.columns
    assert "ts" in result.columns
    assert "ts_year" in result.columns


def test_dt_on_date_column(tmp_path):
    import datetime as dt
    dates = [dt.date(2020, 1, 1), dt.date(2020, 6, 15)]
    df = make_df(tmp_path, {"d": pa.array(dates, type=pa.date32())})
    result = df["d"].dt.month
    got = read_col(result, tmp_path, "d_month")
    assert got == [1, 6]


def test_dt_non_temporal_raises(tmp_path):
    df = make_df(tmp_path, {"s": ["a", "b"]})
    with pytest.raises(TypeError, match="expected timestamp"):
        df["s"].dt.year


def test_dt_missing_column_raises(tmp_path):
    df = make_df(tmp_path, {"ts": [pd.Timestamp("2020-01-01")]})
    with pytest.raises(KeyError):
        df["nope"]


def test_dt_null_timestamp_propagates(tmp_path):
    """NaT (pandas) → null (Arrow) → null (dt.month output).

    v0.9.0a-fix1: use `pa.array(ts)` not `ts.tolist()`. The latter yields
    [Timestamp, NaT, Timestamp]; `pa.table({"ts": [...]})` calls
    `NaT.utcoffset()` → `NaTType does not support utcoffset`. `pa.array`
    handles NaT → null (Arrow-native conversion).
    """
    ts = pd.to_datetime(["2020-01-15", None, "2020-03-20"])
    df = make_df(tmp_path, {"ts": pa.array(ts)})
    result = df["ts"].dt.month
    got = read_col(result, tmp_path, "ts_month")
    assert got == [1, None, 3]
