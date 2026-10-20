"""v0.9.0a: string accessor tests."""
from pathlib import Path

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


def test_str_len(tmp_path):
    df = make_df(tmp_path, {"s": ["a", "abc", "", None]})
    result = df["s"].str.len
    got = read_col(result, tmp_path, "s_len")
    assert got == [1, 3, 0, None]


def test_str_lower(tmp_path):
    df = make_df(tmp_path, {"s": ["Hello", "WORLD", "MiXeD", None]})
    result = df["s"].str.lower
    got = read_col(result, tmp_path, "s_lower")
    assert got == ["hello", "world", "mixed", None]


def test_str_upper(tmp_path):
    df = make_df(tmp_path, {"s": ["Hello", "world", None]})
    result = df["s"].str.upper
    got = read_col(result, tmp_path, "s_upper")
    assert got == ["HELLO", "WORLD", None]


def test_str_contains_filter(tmp_path):
    df = make_df(tmp_path, {
        "s": ["foo", "bar", "foobar", None],
        "v": [1, 2, 3, 4],
    })
    filtered = df[df["s"].str.contains("foo")]
    assert filtered.count() == 2
    out = tmp_path / "o.parquet"
    filtered.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert {r["v"] for r in rows} == {1, 3}


def test_str_contains_null_does_not_match(tmp_path):
    """Pandas default for .str.contains: na=False → null row excluded."""
    df = make_df(tmp_path, {"s": ["foo", None], "v": [1, 2]})
    filtered = df[df["s"].str.contains("foo")]
    assert filtered.count() == 1


def test_str_non_string_raises(tmp_path):
    df = make_df(tmp_path, {"v": [1, 2]})
    with pytest.raises(TypeError, match="expected string"):
        df["v"].str.lower


def test_str_preserves_original(tmp_path):
    df = make_df(tmp_path, {"s": ["Hello", "World"], "v": [1, 2]})
    result = df["s"].str.lower
    assert "s" in result.columns
    assert "v" in result.columns
    assert "s_lower" in result.columns


def test_str_multibatch(tmp_path):
    df = make_df(tmp_path, {"s": ["a", "b", "c", "d"]}, chunk=1)
    result = df["s"].str.len
    got = read_col(result, tmp_path, "s_len")
    assert got == [1, 1, 1, 1]
