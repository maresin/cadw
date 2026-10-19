"""v0.4.0d: filter-compat для index / set_index / reset_index / iloc / loc."""
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.dataset.index import MaterializedIndex, RangeIndex


def make_df(tmp_path, data=None, chunk=3):
    if data is None:
        data = {"k": ["a", "b", "c", "d", "e"], "v": [10, 20, 30, 40, 50]}
    t = pa.table(data)
    p = Path(tmp_path) / "src.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, tmp_path / "ds", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=tmp_path / "spill",
    )


# ---------- filtered.index (RangeIndex parent) ----------

def test_filtered_index_range(tmp_path):
    df = make_df(tmp_path)
    filtered = df[df["v"] > 15]  # b,c,d,e → raw positions 1,2,3,4
    idx = filtered.index
    assert isinstance(idx, MaterializedIndex)
    assert list(idx) == [1, 2, 3, 4]


def test_filtered_index_all_filtered_out(tmp_path):
    df = make_df(tmp_path)
    filtered = df[df["v"] > 1000]
    idx = filtered.index
    assert isinstance(idx, MaterializedIndex)
    assert len(idx) == 0


def test_filtered_index_with_materialized_parent(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    filtered = df[df["v"] > 15]  # b,c,d,e
    idx = filtered.index
    assert isinstance(idx, MaterializedIndex)
    assert list(idx) == ["b", "c", "d", "e"]


def test_filtered_index_slice_filter(tmp_path):
    df = make_df(tmp_path)
    sliced = df[1:4]  # raw positions 1,2,3
    idx = sliced.index
    assert isinstance(idx, MaterializedIndex)
    assert list(idx) == [1, 2, 3]


def test_filtered_index_mask_then_slice(tmp_path):
    df = make_df(tmp_path)
    filtered = df[df["v"] > 15][0:2]  # of [1,2,3,4] take [0:2] → raw [1,2]
    idx = filtered.index
    assert isinstance(idx, MaterializedIndex)
    assert list(idx) == [1, 2]


# ---------- filtered.set_index ----------

def test_set_index_on_filtered(tmp_path):
    df = make_df(tmp_path)
    filtered = df[df["v"] > 15]  # b,c,d,e
    result = filtered.set_index("k", drop=True)
    assert isinstance(result.index, MaterializedIndex)
    assert list(result.index) == ["b", "c", "d", "e"]
    assert "k" not in result.columns


def test_set_index_then_filter(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    filtered = df[df["v"] > 30]  # d, e
    assert list(filtered.index) == ["d", "e"]


# ---------- filtered.reset_index ----------

def test_reset_index_filtered_drop_true(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=True)
    filtered = df[df["v"] > 15]  # b,c,d,e
    restored = filtered.reset_index(drop=True)
    assert isinstance(restored.index, RangeIndex)
    assert restored.index.start == 0
    assert restored.index.stop == 4  # 4 filtered rows


def test_reset_index_filtered_drop_false(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=True)
    filtered = df[df["v"] > 15]
    restored = filtered.reset_index(drop=False)
    assert isinstance(restored.index, RangeIndex)
    assert restored.index.stop == 4
    assert "index" in restored.columns
    # values
    out = tmp_path / "out.parquet"
    restored.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["index"] for r in rows] == ["b", "c", "d", "e"]


# ---------- filtered.iloc[list] preserves parent ----------

def test_iloc_list_filtered_range_parent(tmp_path):
    df = make_df(tmp_path)
    filtered = df[df["v"] > 15]  # b,c,d,e → raw [1,2,3,4]
    sub = filtered.iloc[[0, 2]]  # filtered 0, 2 → raw 1, 3
    assert isinstance(sub.index, MaterializedIndex)
    assert list(sub.index) == [1, 3]


def test_iloc_list_filtered_materialized_parent(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    filtered = df[df["v"] > 15]  # b,c,d,e
    sub = filtered.iloc[[0, 2]]  # raw 1, 3 → k values b, d
    assert isinstance(sub.index, MaterializedIndex)
    assert list(sub.index) == ["b", "d"]


# ---------- filtered.loc works ----------

def test_loc_filtered(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    filtered = df[df["v"] > 15]  # b, c, d, e
    assert filtered.loc["c"] == {"k": "c", "v": 30}
    with pytest.raises(KeyError):
        filtered.loc["a"]


def test_loc_filtered_slice(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    filtered = df[df["v"] > 15]  # b, c, d, e
    sub = filtered.loc["c":"d"]
    out = tmp_path / "out.parquet"
    sub.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["k"] for r in rows] == ["c", "d"]


# ---------- no filter — unchanged ----------

def test_no_filter_index_still_range(tmp_path):
    df = make_df(tmp_path)
    assert isinstance(df.index, RangeIndex)


def test_no_filter_set_index_still_works(tmp_path):
    df = make_df(tmp_path)
    result = df.set_index("k", drop=False)
    assert isinstance(result.index, MaterializedIndex)
    assert list(result.index) == ["a", "b", "c", "d", "e"]

# ---------- AST guard: no undefined module-level helper calls ----------

def test_helper_is_defined_not_just_called():
    """Регрессия v0.4.0d: `if "name" not in s` видел call site и не
    добавлял def. Guard: `def _apply_filter_with_positions(` присутствует.
    """
    import inspect
    from cadw import dataframe as df_mod
    src = inspect.getsource(df_mod)
    assert "def _apply_filter_with_positions(" in src, (
        "helper _apply_filter_with_positions not defined (only called) — "
        "NameError at runtime. §13.4/60."
    )
    assert "_apply_filter_with_positions(" in src

