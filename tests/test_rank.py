"""
rank (v0.3.1): external_sort + stream-rank + re-sort by row_id.

Проверяем:
  - method='first'/'min'/'dense' на asc и desc;
  - pct=True;
  - сохранение исходного порядка (position);
  - all-unique / all-same / ties;
  - после filter;
  - пустой результат;
  - output_col custom;
  - invalid method → NotImplementedError;
  - pandas-equivalence для first/min/dense (set-based).
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


def read_df(df, tmp_path, name="out.parquet"):
    out = Path(tmp_path) / name
    df.write_parquet(out)
    return pq.read_table(out).to_pandas()


# ---------- method='first' ----------

def test_rank_first_ascending(tmp_path):
    df = make_df(tmp_path, "L", {"k": ["b", "a", "c", "a"], "v": [1, 2, 3, 4]})
    result = df.rank(by="k", method="first")
    got = read_df(result, tmp_path)
    # sort by k asc, ties by row_id asc:
    # row_id 1 (a,2), row_id 3 (a,4), row_id 0 (b,1), row_id 2 (c,3)
    # ranks:           1,              2,              3,              4
    # back to original order: b→3, a→1, c→4, a→2
    assert got["k_rank"].tolist() == [3, 1, 4, 2]


def test_rank_first_descending(tmp_path):
    df = make_df(tmp_path, "L", {"k": ["b", "a", "c", "a"], "v": [1, 2, 3, 4]})
    result = df.rank(by="k", method="first", ascending=False)
    got = read_df(result, tmp_path)
    # sort by k desc, ties by row_id asc:
    # row_id 2 (c,3), row_id 0 (b,1), row_id 1 (a,2), row_id 3 (a,4)
    # ranks:           1,              2,              3,              4
    # original: b→2, a→3, c→1, a→4
    assert got["k_rank"].tolist() == [2, 3, 1, 4]


# ---------- method='min' ----------

def test_rank_min_with_ties(tmp_path):
    df = make_df(tmp_path, "L", {"k": ["a", "b", "a", "c", "a"], "v": list(range(5))})
    result = df.rank(by="k", method="min")
    got = read_df(result, tmp_path)
    # sorted: a(pos1), a(pos2), a(pos3), b(pos4), c(pos5)
    # min: a→1, a→1, a→1, b→4, c→5
    # original (a,b,a,c,a): 1, 4, 1, 5, 1
    assert got["k_rank"].tolist() == [1, 4, 1, 5, 1]


# ---------- method='dense' ----------

def test_rank_dense_with_ties(tmp_path):
    df = make_df(tmp_path, "L", {"k": ["b", "a", "c", "a", "b"], "v": list(range(5))})
    result = df.rank(by="k", method="dense")
    got = read_df(result, tmp_path)
    # distinct sorted values: a→1, b→2, c→3
    # original (b,a,c,a,b): 2, 1, 3, 1, 2
    assert got["k_rank"].tolist() == [2, 1, 3, 1, 2]


# ---------- pct ----------

def test_rank_pct(tmp_path):
    df = make_df(tmp_path, "L", {"k": [10, 20, 30, 40], "v": list(range(4))})
    result = df.rank(by="k", method="first", pct=True)
    got = read_df(result, tmp_path)
    # ranks 1,2,3,4 / 4 = 0.25, 0.5, 0.75, 1.0 (original order 10,20,30,40)
    assert got["k_rank"].tolist() == [0.25, 0.5, 0.75, 1.0]


# ---------- preserve original order ----------

def test_rank_preserves_row_count_and_order(tmp_path):
    n = 50
    rng = np.random.default_rng(42)
    data = {
        "k": rng.integers(0, 5, size=n).tolist(),
        "v": list(range(n)),
    }
    df = make_df(tmp_path, "L", data, chunk=10)
    result = df.rank(by="k", method="first")
    got = read_df(result, tmp_path)
    assert got.shape[0] == n
    # v column unchanged and in original order
    assert got["v"].tolist() == list(range(n))


# ---------- edge cases ----------

def test_rank_all_unique(tmp_path):
    n = 20
    df = make_df(tmp_path, "L", {"k": list(range(n)), "v": list(range(n))})
    result = df.rank(by="k", method="first")
    got = read_df(result, tmp_path)
    # ranks = 1..n (identity, since sorted order = original order)
    assert got["k_rank"].tolist() == list(range(1, n + 1))


def test_rank_all_same_first(tmp_path):
    n = 10
    df = make_df(tmp_path, "L", {"k": ["X"] * n, "v": list(range(n))})
    result = df.rank(by="k", method="first")
    got = read_df(result, tmp_path)
    assert got["k_rank"].tolist() == list(range(1, n + 1))


def test_rank_all_same_min(tmp_path):
    n = 10
    df = make_df(tmp_path, "L", {"k": ["X"] * n, "v": list(range(n))})
    result = df.rank(by="k", method="min")
    got = read_df(result, tmp_path)
    assert got["k_rank"].tolist() == [1] * n


def test_rank_all_same_dense(tmp_path):
    n = 10
    df = make_df(tmp_path, "L", {"k": ["X"] * n, "v": list(range(n))})
    result = df.rank(by="k", method="dense")
    got = read_df(result, tmp_path)
    assert got["k_rank"].tolist() == [1] * n


# ---------- after filter ----------

def test_rank_after_filter(tmp_path):
    df = make_df(tmp_path, "L", {
        "k": ["a", "a", "b", "b", "c"],
        "v": [1, 2, 3, 4, 5],
    })
    filtered = df[df["v"] > 2]  # rows (b,3),(b,4),(c,5)
    result = filtered.rank(by="k", method="first")
    got = read_df(result, tmp_path)
    # sorted: b(orig pos 0), b(orig pos 1), c(orig pos 2) → ranks 1,2,3
    # but keep='first' with ties by row_id: b at filtered-pos-0 gets 1, b at 1 gets 2
    # c gets 3.
    # output in filtered order: [1, 2, 3]
    assert got["k_rank"].tolist() == [1, 2, 3]


# ---------- output_col custom ----------

def test_rank_custom_output_col(tmp_path):
    df = make_df(tmp_path, "L", {"k": [3, 1, 2], "v": list(range(3))})
    result = df.rank(by="k", method="first", output_col="my_rank")
    got = read_df(result, tmp_path)
    assert "my_rank" in got.columns
    assert "k_rank" not in got.columns


# ---------- errors ----------

def test_rank_invalid_method(tmp_path):
    df = make_df(tmp_path, "L", {"k": [1, 2], "v": [1, 2]})
    for bad in ("max", "average", "last", "unknown"):
        with pytest.raises(NotImplementedError, match="method"):
            df.rank(by="k", method=bad)


def test_rank_missing_column(tmp_path):
    df = make_df(tmp_path, "L", {"k": [1, 2], "v": [1, 2]})
    with pytest.raises(KeyError, match="not in"):
        df.rank(by="nope")


def test_rank_output_col_collision(tmp_path):
    df = make_df(tmp_path, "L", {"k": [1, 2], "v": [1, 2]})
    with pytest.raises(KeyError, match="already in"):
        df.rank(by="k", output_col="v")


# ---------- pandas-equivalence ----------

def test_rank_first_pandas_equivalence(tmp_path):
    n = 100
    rng = np.random.default_rng(7)
    data = {
        "k": rng.integers(0, 5, size=n).tolist(),
        "v": list(range(n)),
    }
    df = make_df(tmp_path, "L", data, chunk=10)
    result = df.rank(by="k", method="first")
    got = read_df(result, tmp_path)

    pdf = pd.DataFrame(data)
    expected = pdf["k"].rank(method="first").astype("int64")
    np.testing.assert_array_equal(
        got["k_rank"].to_numpy(), expected.to_numpy()
    )


def test_rank_min_pandas_equivalence(tmp_path):
    n = 100
    rng = np.random.default_rng(7)
    data = {
        "k": rng.integers(0, 5, size=n).tolist(),
        "v": list(range(n)),
    }
    df = make_df(tmp_path, "L", data, chunk=10)
    result = df.rank(by="k", method="min")
    got = read_df(result, tmp_path)

    pdf = pd.DataFrame(data)
    expected = pdf["k"].rank(method="min").astype("int64")
    np.testing.assert_array_equal(
        got["k_rank"].to_numpy(), expected.to_numpy()
    )


def test_rank_dense_pandas_equivalence(tmp_path):
    n = 100
    rng = np.random.default_rng(7)
    data = {
        "k": rng.integers(0, 5, size=n).tolist(),
        "v": list(range(n)),
    }
    df = make_df(tmp_path, "L", data, chunk=10)
    result = df.rank(by="k", method="dense")
    got = read_df(result, tmp_path)

    pdf = pd.DataFrame(data)
    expected = pdf["k"].rank(method="dense").astype("int64")
    np.testing.assert_array_equal(
        got["k_rank"].to_numpy(), expected.to_numpy()
    )


def test_rank_pct_pandas_equivalence(tmp_path):
    n = 100
    rng = np.random.default_rng(11)
    data = {
        "k": rng.integers(0, 10, size=n).tolist(),
        "v": list(range(n)),
    }
    df = make_df(tmp_path, "L", data, chunk=10)
    result = df.rank(by="k", method="first", pct=True)
    got = read_df(result, tmp_path)

    pdf = pd.DataFrame(data)
    expected = pdf["k"].rank(method="first", pct=True)
    np.testing.assert_allclose(
        got["k_rank"].to_numpy(), expected.to_numpy(), rtol=1e-10
    )


# ---------- chained ----------

def test_rank_then_groupby(tmp_path):
    df = make_df(tmp_path, "L", {
        "g": ["A", "A", "B", "B"],
        "k": [3, 1, 4, 2],
    })
    ranked = df.rank(by="k", method="first")
    # k sorted asc: k=1(rank 1), k=2(rank 2), k=3(rank 3), k=4(rank 4)
    # grouped mean of k_rank per group:
    # A: (3, 1) → mean rank (3+1)/2 = 2.0
    # B: (4, 2) → mean rank (4+2)/2 = 3.0
    result = ranked.groupby("g").agg("k_rank", "mean")
    out = tmp_path / "g.parquet"
    result.write_parquet(out)
    rows = {r["g"]: r["k_rank_mean"] for r in pq.read_table(out).to_pylist()}
    assert rows == {"A": 2.0, "B": 3.0}
