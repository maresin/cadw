"""Stage 8: pin sort_values ascending-list fix; RAM verification harness."""
import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame


def _df():
    return CadwDataFrame.from_table(pa.table({
        "a": pa.array([2, 1, 2, 1], type=pa.int64()),
        "b": pa.array([10, 20, 5, 15], type=pa.int64()),
    }))


def test_sort_values_ascending_list_mixed_direction():
    # Old bug: ascending=[True, False] treated as truthy -> both asc.
    df = _df()
    out = df.sort_values(["a", "b"], ascending=[True, False])
    rows = []
    for batch in out.to_batches(format="numpy", batch_size=10):
        for i in range(len(batch["a"])):
            rows.append((int(batch["a"][i]), int(batch["b"][i])))
    assert rows == [(1, 20), (1, 15), (2, 10), (2, 5)], rows


def test_sort_values_ascending_list_length_mismatch():
    df = _df()
    with pytest.raises(ValueError):
        df.sort_values(["a", "b"], ascending=[True])


def test_sort_values_scalar_unchanged():
    df = _df()
    out = df.sort_values("b", ascending=False)
    vals = []
    for batch in out.to_batches(format="numpy", batch_size=10):
        vals.extend(int(v) for v in batch["b"])
    assert vals == [20, 15, 10, 5]
