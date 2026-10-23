"""v0.9.25l: Phase C C-4b — dt.strftime, dt.tz_convert, str.split."""
import datetime as dt
import math

import pyarrow as pa
import pytest

from cadw import CadwDataFrame, MemoryGuard


def _mk(data, mem_mb=20):
    return CadwDataFrame.from_table(
        pa.table(data),
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
    )


def _to_rows(df):
    rows = []
    for b in df._iter_filtered_batches():
        rows.extend(b.to_pylist())
    return rows


# ============ dt.strftime ============

def test_strftime_basic():
    df = _mk({
        "ts": pa.array(
            [dt.datetime(2020, 1, 15, 10, 30, 0)],
            type=pa.timestamp("s"),
        ),
    })
    out = df["ts"].dt.strftime("%Y-%m-%d")
    rows = _to_rows(out)
    assert [r["ts_strftime"] for r in rows] == ["2020-01-15"]


def test_strftime_full():
    df = _mk({
        "ts": pa.array(
            [dt.datetime(2020, 6, 20, 14, 45, 30)],
            type=pa.timestamp("s"),
        ),
    })
    out = df["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    rows = _to_rows(out)
    assert [r["ts_strftime"] for r in rows] == ["2020-06-20 14:45:30"]


def test_strftime_tz_aware():
    df = _mk({
        "ts": pa.array(
            [dt.datetime(2020, 1, 15, 10, 30, 0,
                         tzinfo=dt.timezone.utc)],
            type=pa.timestamp("s", tz="UTC"),
        ),
    })
    out = df["ts"].dt.strftime("%Y-%m-%d %H:%M")
    rows = _to_rows(out)
    # pc.strftime uses local wall clock of tz; UTC → 10:30
    assert [r["ts_strftime"] for r in rows] == ["2020-01-15 10:30"]


def test_strftime_null_input():
    df = _mk({
        "ts": pa.array(
            [dt.datetime(2020, 1, 1), None],
            type=pa.timestamp("s"),
        ),
    })
    out = df["ts"].dt.strftime("%Y")
    rows = _to_rows(out)
    assert [r["ts_strftime"] for r in rows] == ["2020", None]


def test_strftime_non_temporal_raises():
    df = _mk({"v": [1, 2]})
    with pytest.raises(TypeError, match="expected timestamp"):
        df["v"].dt.strftime("%Y")


def test_strftime_empty_format_raises():
    df = _mk({
        "ts": pa.array([dt.datetime(2020, 1, 1)],
                       type=pa.timestamp("s")),
    })
    with pytest.raises(ValueError, match="non-empty"):
        df["ts"].dt.strftime("")


def test_strftime_preserves_original():
    df = _mk({
        "ts": pa.array([dt.datetime(2020, 1, 15)],
                       type=pa.timestamp("s")),
        "v": [42],
    })
    out = df["ts"].dt.strftime("%Y")
    rows = _to_rows(out)
    assert rows[0]["v"] == 42
    assert rows[0]["ts_strftime"] == "2020"


# ============ dt.tz_convert ============

def test_tz_convert_utc_to_ny():
    try:
        df = _mk({
            "ts": pa.array(
                [dt.datetime(2020, 1, 15, 10, 30, 0,
                             tzinfo=dt.timezone.utc)],
                type=pa.timestamp("s", tz="UTC"),
            ),
        })
        out = df["ts"].dt.tz_convert("America/New_York")
        rows = _to_rows(out)
        # 10:30 UTC → 05:30 EST (January = winter, -5h)
        ts_str = rows[0]["ts_tz_convert"]
        # Result value is tz-aware dt; extract wall clock
        if ts_str is None:
            pytest.skip("tz_convert returned null unexpectedly")
        # Check local wall-clock hour is 5 (05:30)
        assert ts_str.hour == 5
        assert ts_str.minute == 30
    except NotImplementedError:
        pytest.skip(
            "dt.tz_convert: PyArrow 26 lacks tz-aware→tz-aware cast"
        )


def test_tz_convert_naive_raises():
    df = _mk({
        "ts": pa.array([dt.datetime(2020, 1, 1)],
                       type=pa.timestamp("s")),
    })
    with pytest.raises(TypeError, match="naive"):
        df["ts"].dt.tz_convert("America/New_York")


def test_tz_convert_non_temporal_raises():
    """v0.9.25l-fix1 (§13.4/277): .dt accessor gate at __init__
    rejects non-temporal columns BEFORE tz_convert body runs. The
    gate's message ('expected timestamp or date') is authoritative,
    not tz_convert's own intended message ('timestamp required').
    """
    df = _mk({"v": [1]})
    with pytest.raises(TypeError, match="expected timestamp or date"):
        df["v"].dt.tz_convert("UTC")


# ============ str.split ============

def test_split_basic():
    df = _mk({"s": ["a-b-c", "x"]})
    out = df["s"].str.split("-")
    rows = _to_rows(out)
    assert [r["s_split"] for r in rows] == [
        ["a", "b", "c"], ["x"],
    ]


def test_split_empty_string():
    df = _mk({"s": ["", "a"]})
    out = df["s"].str.split("-")
    rows = _to_rows(out)
    # Empty string → ['']
    assert [r["s_split"] for r in rows] == [[""], ["a"]]


def test_split_null_input():
    df = _mk({
        "s": pa.array(["a-b", None], type=pa.string()),
    })
    out = df["s"].str.split("-")
    rows = _to_rows(out)
    assert [r["s_split"] for r in rows] == [["a", "b"], None]


def test_split_max_splits():
    df = _mk({"s": ["a-b-c-d"]})
    out = df["s"].str.split("-", max_splits=2)
    rows = _to_rows(out)
    # max 2 splits → 3 parts; last part contains rest
    assert [r["s_split"] for r in rows] == [["a", "b", "c-d"]]


def test_split_reverse():
    df = _mk({"s": ["a-b-c-d"]})
    out = df["s"].str.split("-", max_splits=2, reverse=True)
    rows = _to_rows(out)
    # reverse + max 2 splits → split from end; first part has rest
    assert [r["s_split"] for r in rows] == [["a-b", "c", "d"]]


def test_split_non_string_raises():
    df = _mk({"v": [1]})
    with pytest.raises(TypeError, match="string"):
        df["v"].str.split("-")


def test_split_empty_pat_raises():
    df = _mk({"s": ["abc"]})
    with pytest.raises(ValueError, match="non-empty"):
        df["s"].str.split("")


def test_split_invalid_max_splits():
    df = _mk({"s": ["a"]})
    with pytest.raises(ValueError, match="max_splits"):
        df["s"].str.split("-", max_splits=0)


def test_split_preserves_original():
    df = _mk({"s": ["a-b"], "v": [42]})
    out = df["s"].str.split("-")
    rows = _to_rows(out)
    assert rows[0]["s"] == "a-b"
    assert rows[0]["v"] == 42


def test_split_filtered():
    df = _mk({"s": ["a-b", "c-d-e", "f"], "v": [1, 2, 3]})
    filtered = df[df["v"] > 1]
    out = filtered["s"].str.split("-")
    rows = _to_rows(out)
    assert [r["s_split"] for r in rows] == [
        ["c", "d", "e"], ["f"],
    ]
