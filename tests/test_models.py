"""
Тесты OpMemoryModel и ReduceMemoryModel (11-A-1).

Инвариант модели: estimate(solve(x)) <= x * safety.
Плюс граничные случаи: state не влезает -> None; string без sample -> ValueError.
"""
import pyarrow as pa
import pytest

from cadw.core.models import (
    DataChars,
    data_chars_from_sample,
    get_model,
    list_models,
    row_size,
    type_size,
)
from cadw.core.models_reduce import ReduceMemoryModel, ReduceParams


# ---------- type_size / row_size ----------

def test_type_size_numeric():
    assert type_size(pa.int64()) == 8
    assert type_size(pa.int32()) == 4
    assert type_size(pa.float64()) == 8
    assert type_size(pa.float32()) == 4
    assert type_size(pa.bool_()) == 1


def test_type_size_timestamp():
    assert type_size(pa.timestamp("ns")) == 8
    assert type_size(pa.date32()) == 4


def test_type_size_string_requires_sample():
    with pytest.raises(ValueError, match="sample required"):
        type_size(pa.string(), sample=None)


def test_type_size_string_with_sample():
    sample = pa.array(["abc", "de", "f"])
    # avg_len = (3+2+1)/3 = 2, + offset 4 = 6
    assert type_size(pa.string(), sample=sample) == 6


def test_type_size_empty_sample_raises():
    with pytest.raises(ValueError, match="empty sample"):
        type_size(pa.string(), sample=pa.array([], type=pa.string()))


def test_row_size_mixed():
    schema = pa.schema([("a", pa.int64()), ("b", pa.float32())])
    assert row_size(schema) == 12  # 8 + 4


def test_row_size_with_string_requires_sample():
    schema = pa.schema([("a", pa.int64()), ("b", pa.string())])
    with pytest.raises(ValueError):
        row_size(schema, sample_batch=None)


def test_data_chars_from_sample():
    batch = pa.RecordBatch.from_arrays(
        [pa.array([1, 2, 3]), pa.array(["a", "bb", "ccc"])],
        names=["n", "s"],
    )
    dc = data_chars_from_sample(batch, n_rows_total=1000, k_est=10)
    # row = 8 (int64) + (avg string (1+2+3)/3=2 + 4 offset = 6) = 14
    assert dc.row_size_bytes == 14
    assert dc.n_rows == 1000
    assert dc.k_est == 10


# ---------- registry ----------

def test_reduce_model_registered():
    m = get_model("reduce")
    assert isinstance(m, ReduceMemoryModel)
    assert "reduce" in list_models()


def test_get_unknown_model_raises():
    with pytest.raises(KeyError, match="not registered"):
        get_model("nonexistent_op")


# ---------- ReduceMemoryModel ----------

def _data_chars(row_size=16, n_rows=10_000) -> DataChars:
    schema = pa.schema([("x", pa.int64()), ("y", pa.int64())])
    return DataChars(schema=schema, n_rows=n_rows, row_size_bytes=row_size)


@pytest.mark.parametrize(
    "state_bytes, available_mb",
    [
        (16, 10),       # CountOp/SumOp/MeanOp
        (24, 10),       # StdOp (Welford)
        (8 + 800 + 80_000, 10),      # CovarianceOp(d=100): 8+8*100+8*10000
        (255 * 8 + 16, 10),          # HistogramOp(bins=255)
    ],
)
def test_reduce_estimate_within_safety(state_bytes, available_mb):
    """estimate(solve(x)) <= x * safety для всех реалистичных state."""
    model = ReduceMemoryModel()
    dc = _data_chars()
    available = available_mb * 1024 * 1024
    safety = 0.7

    params = model.solve(available, dc, safety=safety, op_context={"state_bytes": state_bytes})
    assert params is not None, (
        f"solve returned None for state={state_bytes}, available={available_mb}MB"
    )
    assert isinstance(params, ReduceParams)
    assert params.batch_size > 0
    assert params.state_bytes == state_bytes

    est = model.estimate(params, dc)
    assert est <= int(available * safety), (
        f"estimate {est} > available*safety {int(available * safety)} "
        f"for state={state_bytes}"
    )


def test_reduce_solve_returns_none_when_state_too_big():
    """CovarianceOp(d=5000) ~ 200 MB; available=10 MB -> None."""
    d = 5000
    state_bytes = 8 + 8 * d + 8 * d * d  # ~200 MB
    model = ReduceMemoryModel()
    dc = _data_chars()
    available = 10 * 1024 * 1024
    params = model.solve(available, dc, op_context={"state_bytes": state_bytes})
    assert params is None


def test_reduce_solve_requires_state_bytes():
    model = ReduceMemoryModel()
    dc = _data_chars()
    with pytest.raises(ValueError, match="state_bytes"):
        model.solve(10 * 1024 * 1024, dc, op_context=None)
    with pytest.raises(ValueError, match="state_bytes"):
        model.solve(10 * 1024 * 1024, dc, op_context={})


def test_reduce_solve_negative_state_raises():
    model = ReduceMemoryModel()
    dc = _data_chars()
    with pytest.raises(ValueError, match=">= 0"):
        model.solve(10 * 1024 * 1024, dc, op_context={"state_bytes": -1})


def test_reduce_estimate_type_check():
    model = ReduceMemoryModel()
    dc = _data_chars()
    # Параметр не того типа
    class FakeParams:
        pass
    with pytest.raises(TypeError, match="ReduceParams"):
        model.estimate(FakeParams(), dc)


def test_reduce_describe():
    model = ReduceMemoryModel()
    d = model.describe()
    assert "ReduceMemoryModel" in d
    assert "state_bytes" in d
