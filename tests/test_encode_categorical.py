"""v0.6.2a: encode_categorical (Arrow-native, O(K) memory)."""
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard


def make_df(tmp_path, name="L", data=None, chunk=3):
    if data is None:
        data = {
            "cat": ["a", "b", "c", "a", "b"],
            "v": [1, 2, 3, 4, 5],
        }
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def read_cols(df, tmp_path, cols):
    out = tmp_path / "o.parquet"
    df.write_parquet(out)
    return pq.read_table(out).select(cols).to_pandas()


def test_encode_basic(tmp_path):
    df = make_df(tmp_path)
    encoded, maps = df.encode_categorical("cat")
    assert maps["cat"] == ["a", "b", "c"]
    got = read_cols(encoded, tmp_path, ["cat", "v"])
    assert list(got["cat"]) == [0, 1, 2, 0, 1]
    assert list(got["v"]) == [1, 2, 3, 4, 5]


def test_encode_preserves_original(tmp_path):
    df = make_df(tmp_path)
    _ = df.encode_categorical("cat")
    # Original unchanged
    got = read_cols(df, tmp_path, ["cat"])
    assert list(got["cat"]) == ["a", "b", "c", "a", "b"]


def test_encode_dtype_int32(tmp_path):
    df = make_df(tmp_path)
    encoded, _ = df.encode_categorical("cat")
    out = tmp_path / "o.parquet"
    encoded.write_parquet(out)
    t = pq.read_table(out)
    assert t.schema.field("cat").type == pa.int32()


def test_encode_deterministic_order(tmp_path):
    """Sorted values → deterministic ids across runs."""
    df = make_df(tmp_path)
    _, maps1 = df.encode_categorical("cat")
    _, maps2 = df.encode_categorical("cat")
    assert maps1["cat"] == maps2["cat"]


def test_encode_multi_columns(tmp_path):
    df = make_df(tmp_path, data={
        "a": ["x", "y", "x"],
        "b": ["p", "q", "p"],
        "v": [1, 2, 3],
    })
    encoded, maps = df.encode_categorical(["a", "b"])
    assert maps["a"] == ["x", "y"] and maps["b"] == ["p", "q"]
    got = read_cols(encoded, tmp_path, ["a", "b", "v"])
    assert list(got["a"]) == [0, 1, 0]
    assert list(got["b"]) == [0, 1, 0]


def test_encode_missing_column(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(KeyError, match="nope"):
        df.encode_categorical("nope")


def test_encode_non_string_raises(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(TypeError, match="only string"):
        df.encode_categorical("v")


def test_encode_empty_columns_raises(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(ValueError, match="non-empty"):
        df.encode_categorical([])


def test_encode_nulls_skip(tmp_path):
    """Nulls: null in source → null id (not 0)."""
    df = make_df(tmp_path, data={
        "cat": ["a", None, "b", None, "a"],
        "v": [1, 2, 3, 4, 5],
    })
    encoded, maps = df.encode_categorical("cat")
    assert maps["cat"] == ["a", "b"]  # no None in map
    got = read_cols(encoded, tmp_path, ["cat"])
    # ids: 0, null, 1, null, 0
    import numpy as np
    vals = got["cat"].tolist()
    assert vals[0] == 0 and vals[2] == 1 and vals[4] == 0
    assert vals[1] is None or (isinstance(vals[1], float) and np.isnan(vals[1]))


def test_encode_multiple_batches(tmp_path):
    """Multi-batch: same value → same id across batches."""
    df = make_df(tmp_path, data={
        "cat": ["a", "b"] * 10,  # 20 rows
        "v": list(range(20)),
    }, chunk=3)
    encoded, maps = df.encode_categorical("cat")
    got = read_cols(encoded, tmp_path, ["cat"])
    assert list(got["cat"]) == [0, 1] * 10


def test_encode_then_roundtrip(tmp_path):
    """encode → save → load → reverse-map."""
    df = make_df(tmp_path)
    encoded, maps = df.encode_categorical("cat")
    out = tmp_path / "o.parquet"
    encoded.write_parquet(out)
    reloaded = CadwDataFrame.from_parquet(
        out, tmp_path / "ds2", chunk_size=3,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=tmp_path / "spill2",
    )
    got = read_cols(reloaded, tmp_path, ["cat"])
    reverse = [maps["cat"][i] for i in got["cat"]]
    assert reverse == ["a", "b", "c", "a", "b"]

# ── v0.9.99: actionable error on K-axis overflow ────────────────────


def test_encode_categorical_k_limit_actionable():
    """K > state_limit → MemoryLimitExceeded with actionable hint."""
    import pyarrow as pa
    from cadw import CadwDataFrame, MemoryGuard
    from cadw.core.memory_guard import MemoryLimitExceeded
    # 1 MB guard, K~10^5 unique strings ~18 MB state → triggers.
    K = 100_000
    df = CadwDataFrame.from_table(
        pa.table({"cat": [f"v{i}" for i in range(K)]}),
        memory_guard=MemoryGuard(max_memory_mb=1),
    )
    with pytest.raises(MemoryLimitExceeded) as ei:
        df.encode_categorical("cat")
    msg = str(ei.value)
    assert "O(K) by API contract" in msg
    assert "reverse_maps" in msg
    assert "top-K" in msg or "v0.9.102" in msg

