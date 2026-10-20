# v0.9.4b-2: transform, describe, dt/str extras.

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard


def make_df(tmp_path, data, chunk=10, name="L", mem_mb=50):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def to_pd(df, tmp_path, name="out.parquet"):
    out = tmp_path / name
    df.write_parquet(out)
    return pq.read_table(out).to_pandas()


# ============ _GroupBy.transform ============

def test_transform_mean_basic(tmp_path):
    df = make_df(tmp_path, {
        "g": ["a", "a", "b", "b"],
        "v": [1.0, 3.0, 10.0, 20.0],
    }, name="a")
    out = df.groupby("g").transform("v", "mean")
    pd_out = to_pd(out, tmp_path)
    assert "v_mean" in pd_out.columns
    assert "v" in pd_out.columns
    assert len(pd_out) == 4
    grouped = pd_out.groupby("g")["v_mean"].first().to_dict()
    assert grouped["a"] == 2.0
    assert grouped["b"] == 15.0


def test_transform_sum_basic(tmp_path):
    df = make_df(tmp_path, {
        "g": ["x", "x", "y"],
        "v": [1, 2, 5],
    }, name="a")
    out = df.groupby("g").transform("v", "sum")
    pd_out = to_pd(out, tmp_path)
    grouped = pd_out.groupby("g")["v_sum"].first().to_dict()
    assert grouped["x"] == 3
    assert grouped["y"] == 5


def test_transform_with_filter(tmp_path):
    df = make_df(tmp_path, {
        "g": ["a", "a", "a", "b", "b"],
        "v": [1.0, 2.0, 3.0, 10.0, 20.0],
    }, name="a")
    f = df[df["v"] > 2]
    out = f.groupby("g").transform("v", "mean")
    pd_out = to_pd(out, tmp_path)
    assert len(pd_out) == 3
    grouped = pd_out.groupby("g")["v_mean"].first().to_dict()
    assert grouped["a"] == 3.0
    assert grouped["b"] == 15.0


def test_transform_length_mismatch(tmp_path):
    df = make_df(tmp_path, {"g": ["a"], "v": [1.0]}, name="a")
    with pytest.raises(ValueError, match="same length"):
        df.groupby("g").transform(["v"], ["mean", "sum"])


# ============ describe ============

def test_describe_basic(tmp_path):
    df = make_df(tmp_path, {"x": [1.0, 2.0, 3.0, 4.0, 5.0]}, name="a")
    desc = df.describe()
    assert "x" in desc
    d = desc["x"]
    assert d["count"] == 5
    assert d["mean"] == pytest.approx(3.0)
    assert d["std"] == pytest.approx(1.5811388300841898, rel=1e-9)
    assert d["min"] == 1.0
    assert d["25%"] == pytest.approx(2.0)
    assert d["50%"] == pytest.approx(3.0)
    assert d["75%"] == pytest.approx(4.0)
    assert d["max"] == 5.0


def test_describe_filtered(tmp_path):
    df = make_df(tmp_path, {"x": [1, 2, 3, 4, 5]}, name="a")
    f = df[df["x"] > 2]
    desc = f.describe()
    assert desc["x"]["count"] == 3
    assert desc["x"]["min"] == 3.0
    assert desc["x"]["max"] == 5.0


def test_describe_subset(tmp_path):
    df = make_df(tmp_path, {
        "a": [1, 2], "b": [10, 20], "s": ["x", "y"],
    }, name="a")
    desc = df.describe(columns=["a"])
    assert "a" in desc
    assert "b" not in desc
    assert "s" not in desc


def test_describe_all_numeric_default(tmp_path):
    df = make_df(tmp_path, {
        "i": [1, 2], "f": [1.0, 2.0], "s": ["a", "b"],
    }, name="a")
    desc = df.describe()
    assert "i" in desc
    assert "f" in desc
    assert "s" not in desc


def test_describe_missing_column(tmp_path):
    df = make_df(tmp_path, {"x": [1]}, name="a")
    with pytest.raises(KeyError, match="nope"):
        df.describe(columns=["nope"])


# ============ dt.floor/ceil/round/week_of_year ============

def test_dt_floor_hour(tmp_path):
    ts = pd.to_datetime([
        "2020-01-01 10:30:00", "2020-01-01 10:45:00",
    ])
    df = make_df(tmp_path, {"ts": pa.array(ts)}, name="a")
    out = df["ts"].dt.floor("1h")
    pd_out = to_pd(out, tmp_path)
    assert [str(t) for t in pd_out["ts_floor"].tolist()] == [
        "2020-01-01 10:00:00", "2020-01-01 10:00:00",
    ]


def test_dt_ceil_hour(tmp_path):
    ts = pd.to_datetime([
        "2020-01-01 10:30:00", "2020-01-01 10:00:00",
    ])
    df = make_df(tmp_path, {"ts": pa.array(ts)}, name="a")
    out = df["ts"].dt.ceil("1h")
    pd_out = to_pd(out, tmp_path)
    assert [str(t) for t in pd_out["ts_ceil"].tolist()] == [
        "2020-01-01 11:00:00", "2020-01-01 10:00:00",
    ]


def test_dt_round_hour(tmp_path):
    ts = pd.to_datetime([
        "2020-01-01 10:20:00", "2020-01-01 10:40:00",
    ])
    df = make_df(tmp_path, {"ts": pa.array(ts)}, name="a")
    out = df["ts"].dt.round("1h")
    pd_out = to_pd(out, tmp_path)
    assert [str(t) for t in pd_out["ts_round"].tolist()] == [
        "2020-01-01 10:00:00", "2020-01-01 11:00:00",
    ]


def test_dt_week_of_year(tmp_path):
    # 2020-01-01 is ISO week 1; 2020-12-31 is ISO week 53.
    ts = pd.to_datetime(["2020-01-01", "2020-12-31"])
    df = make_df(tmp_path, {"ts": pa.array(ts)}, name="a")
    out = df["ts"].dt.week_of_year
    pd_out = to_pd(out, tmp_path)
    assert pd_out["ts_week_of_year"].tolist() == [1, 53]


def test_dt_floor_non_timestamp_raises(tmp_path):
    df = make_df(tmp_path, {"x": [1, 2]}, name="a")
    with pytest.raises(TypeError, match="expected timestamp"):
        df["x"].dt.floor("1h")


# ============ str.strip / starts_with / replace ============

def test_str_strip(tmp_path):
    df = make_df(tmp_path, {"s": ["  hi  ", "a\tb"]}, name="a")
    out = df["s"].str.strip
    pd_out = to_pd(out, tmp_path)
    assert pd_out["s_strip"].tolist() == ["hi", "a\tb"]


def test_str_starts_with_mask(tmp_path):
    df = make_df(tmp_path, {
        "s": ["foo_bar", "baz_qux", "foo_baz", None],
        "v": [1, 2, 3, 4],
    }, name="a")
    f = df[df["s"].str.starts_with("foo")]
    pd_out = to_pd(f, tmp_path)
    assert pd_out["v"].tolist() == [1, 3]


def test_str_ends_with_mask(tmp_path):
    df = make_df(tmp_path, {
        "s": ["foo_bar", "baz_qux", "baz_bar", None],
        "v": [1, 2, 3, 4],
    }, name="a")
    f = df[df["s"].str.ends_with("bar")]
    pd_out = to_pd(f, tmp_path)
    assert pd_out["v"].tolist() == [1, 3]


def test_str_replace_literal(tmp_path):
    df = make_df(tmp_path, {"s": ["a_b_c", "x_y"]}, name="a")
    out = df["s"].str.replace("_", "-")
    pd_out = to_pd(out, tmp_path)
    assert pd_out["s_replaced"].tolist() == ["a-b-c", "x-y"]


def test_str_replace_regex(tmp_path):
    df = make_df(tmp_path, {"s": ["ab12cd", "ef34"]}, name="a")
    out = df["s"].str.replace("[0-9]+", "#", regex=True)
    pd_out = to_pd(out, tmp_path)
    assert pd_out["s_replaced"].tolist() == ["ab#cd", "ef#"]

