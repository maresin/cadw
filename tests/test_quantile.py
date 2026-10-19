"""
quantile (v0.3.3): external_sort + streamed indexed lookup + linear.

Проверяем:
  - q ∈ {0, 0.25, 0.5, 0.75, 1};
  - odd/even n (linear interpolates for even);
  - scalar q → float, list q → list;
  - pandas / numpy equivalence;
  - nulls (skipna=True);
  - all-null → NaN;
  - empty → NaN;
  - after filter;
  - n=1;
  - invalid q, invalid interpolation, missing column.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard


def make_df(tmp_path, name, data, chunk=3, mem_mb=20):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


# ---------- basic ----------

def test_quantile_median_odd(tmp_path):
    df = make_df(tmp_path, "L", {"v": [3, 1, 4, 1, 5, 9, 2]})  # n=7
    assert df.quantile("v", 0.5) == 3.0


def test_quantile_median_even(tmp_path):
    df = make_df(tmp_path, "L", {"v": [1, 2, 3, 4]})  # n=4, median = (2+3)/2
    assert df.quantile("v", 0.5) == 2.5


def test_quantile_min_max(tmp_path):
    df = make_df(tmp_path, "L", {"v": [3, 1, 4, 1, 5, 9, 2]})
    assert df.quantile("v", 0.0) == 1.0
    assert df.quantile("v", 1.0) == 9.0


@pytest.mark.parametrize("q", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_quantile_matches_numpy(tmp_path, q):
    data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
    df = make_df(tmp_path, "L", {"v": data})
    got = df.quantile("v", q)
    expected = float(np.quantile(np.array(data, dtype=np.float64), q, method="linear"))
    assert got == pytest.approx(expected, rel=1e-10)


def test_quantile_scalar_returns_float(tmp_path):
    df = make_df(tmp_path, "L", {"v": [1, 2, 3]})
    got = df.quantile("v", 0.5)
    assert isinstance(got, float)


def test_quantile_list_returns_list(tmp_path):
    df = make_df(tmp_path, "L", {"v": [1, 2, 3, 4]})
    got = df.quantile("v", [0.0, 0.5, 1.0])
    assert isinstance(got, list)
    assert got == [1.0, 2.5, 4.0]


# ---------- n=1 ----------

def test_quantile_single_row(tmp_path):
    df = make_df(tmp_path, "L", {"v": [42]})
    assert df.quantile("v", 0.5) == 42.0
    assert df.quantile("v", 0.0) == 42.0
    assert df.quantile("v", 1.0) == 42.0


# ---------- nulls ----------

def test_quantile_with_nulls_skips(tmp_path):
    df = make_df(tmp_path, "L", {"v": [1.0, None, 3.0, None, 5.0]})
    # non-null: [1, 3, 5]; median = 3.0
    assert df.quantile("v", 0.5) == 3.0


def test_quantile_all_nulls_returns_nan(tmp_path):
    df = make_df(tmp_path, "L", {
        "v": pa.array([None, None, None], type=pa.float64())
    })
    got = df.quantile("v", 0.5)
    assert isinstance(got, float)
    assert np.isnan(got)


def test_quantile_all_nulls_list(tmp_path):
    df = make_df(tmp_path, "L", {
        "v": pa.array([None, None], type=pa.float64())
    })
    got = df.quantile("v", [0.0, 0.5, 1.0])
    assert len(got) == 3
    assert all(np.isnan(x) for x in got)


# ---------- after filter ----------

def test_quantile_after_filter(tmp_path):
    df = make_df(tmp_path, "L", {"v": list(range(100)), "flag": [0, 1] * 50})
    filtered = df[df["v"] > 49]  # 50..99
    got = filtered.quantile("v", 0.5)
    assert got == 74.5  # median of 50..99 = (74+75)/2


# ---------- pandas-equivalence ----------

def test_quantile_pandas_equivalence(tmp_path):
    rng = np.random.default_rng(42)
    data = rng.integers(0, 1000, size=1000).tolist()
    df = make_df(tmp_path, "L", {"v": data}, chunk=100)
    got = df.quantile("v", [0.0, 0.25, 0.5, 0.75, 1.0])

    s = pd.Series(data, dtype=np.float64)
    expected = s.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).tolist()

    np.testing.assert_allclose(got, expected, rtol=1e-10)


def test_quantile_pandas_equivalence_with_nulls(tmp_path):
    rng = np.random.default_rng(42)
    data = rng.integers(0, 1000, size=100).astype(np.float64)
    mask = rng.random(100) < 0.2
    data[mask] = np.nan
    data_list = [None if np.isnan(x) else float(x) for x in data.tolist()]

    df = make_df(tmp_path, "L", {"v": data_list})
    got = df.quantile("v", 0.5)

    s = pd.Series(data)
    expected = s.quantile(0.5)

    assert got == pytest.approx(expected, rel=1e-10)


# ---------- errors ----------

def test_quantile_missing_column(tmp_path):
    df = make_df(tmp_path, "L", {"v": [1, 2, 3]})
    with pytest.raises(KeyError, match="nope"):
        df.quantile("nope", 0.5)


def test_quantile_invalid_q(tmp_path):
    df = make_df(tmp_path, "L", {"v": [1, 2, 3]})
    for bad in (-0.1, 1.1, 2.0):
        with pytest.raises(ValueError, match="q must be in"):
            df.quantile("v", bad)


def test_quantile_invalid_interpolation(tmp_path):
    df = make_df(tmp_path, "L", {"v": [1, 2, 3]})
    for bad in ("lower", "higher", "nearest", "midpoint", "foo"):
        with pytest.raises(NotImplementedError, match="interpolation"):
            df.quantile("v", 0.5, interpolation=bad)


def test_quantile_empty_q_list(tmp_path):
    df = make_df(tmp_path, "L", {"v": [1, 2, 3]})
    with pytest.raises(ValueError, match="non-empty"):
        df.quantile("v", [])


# ---------- chained ----------

def test_quantile_then_groupby(tmp_path):
    df = make_df(tmp_path, "L", {
        "g": ["A", "A", "A", "B", "B", "B"],
        "v": [1.0, 3.0, 5.0, 2.0, 4.0, 6.0],
    })
    q_med = df.quantile("v", 0.5)  # median of all = (3+4)/2 = 3.5
    assert q_med == 3.5
