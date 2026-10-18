import tempfile

import numpy as np
import pyarrow as pa
import pytest

from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ops.reduce import (
    reduce, CountOp, SumOp, MinMaxOp, MeanOp, StdOp, CovarianceOp, HistogramOp,
)


def make_iter(data, batch_size, type=pa.float64(), name="x"):
    for i in range(0, len(data), batch_size):
        yield pa.RecordBatch.from_arrays(
            [pa.array(data[i:i + batch_size], type=type)], names=[name]
        )


def make_guard(mb=10):
    return MemoryGuard(max_memory_mb=mb)


# ---------------- инвариант batch_size ----------------

@pytest.mark.parametrize("bs", [1, 2, 3, 5, 1000])
def test_mean_invariant(bs):
    data = [10.0, 20.0, 30.0, 40.0, 50.0]
    got = reduce(make_iter(data, bs), MeanOp("x"), make_guard(), log=False)
    assert got == pytest.approx(30.0, rel=1e-12)


@pytest.mark.parametrize("bs", [1, 2, 3, 5, 1000])
def test_sum_invariant(bs):
    data = [1.0, 2.0, 3.0, 4.0]
    got = reduce(make_iter(data, bs), SumOp("x"), make_guard(), log=False)
    assert got == pytest.approx(10.0)


@pytest.mark.parametrize("bs", [1, 2, 3, 5])
def test_std_invariant(bs):
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    got = reduce(make_iter(data, bs), StdOp("x"), make_guard(), log=False)
    assert got == pytest.approx(1.5811388300841898, rel=1e-12)


@pytest.mark.parametrize("bs", [1, 2, 3, 5])
def test_min_max_invariant(bs):
    data = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0]
    got_min = reduce(make_iter(data, bs), MinMaxOp("x", "min"), make_guard(), log=False)
    got_max = reduce(make_iter(data, bs), MinMaxOp("x", "max"), make_guard(), log=False)
    assert got_min == 1.0
    assert got_max == 9.0


@pytest.mark.parametrize("bs", [1, 2, 3, 5])
def test_count_invariant(bs):
    data = [1, 2, 3, 4, 5]
    got = reduce(
        make_iter(data, bs, type=pa.int64()), CountOp("x"), make_guard(), log=False
    )
    assert got == 5


# ---------------- пограничные ----------------

def test_empty_mean_is_nan():
    got = reduce(make_iter([], 10), MeanOp("x"), make_guard(), log=False)
    assert np.isnan(got)


def test_one_row_std_is_nan():
    got = reduce(make_iter([42.0], 10), StdOp("x"), make_guard(), log=False)
    assert np.isnan(got)


def test_last_partial_batch_not_lost():
    got = reduce(
        make_iter([1, 2, 3, 4, 5], 2, type=pa.int64()),
        CountOp("x"),
        make_guard(),
        log=False,
    )
    assert got == 5


def test_all_null_mean_is_nan():
    got = reduce(make_iter([None, None, None], 2), MeanOp("x"), make_guard(), log=False)
    assert np.isnan(got)


def test_partial_null_mean_ignores_nulls():
    got = reduce(make_iter([10.0, None, 30.0], 2), MeanOp("x"), make_guard(), log=False)
    assert got == pytest.approx(20.0)


# ---------------- точность ----------------

def test_std_precision_large_offset():
    """Наивная формула sqrt(sum(x²)/n − mean²) здесь дала бы мусор."""
    data = [1e9 + 1, 1e9 + 2, 1e9 + 3]
    got = reduce(make_iter(data, 2), StdOp("x"), make_guard(), log=False)
    assert got == pytest.approx(1.0, abs=1e-6)


# ---------------- Reduce не пишет на диск ----------------

def test_reduce_does_not_spill(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("reduce() must not spill to disk")

    monkeypatch.setattr(tempfile, "mkstemp", boom)
    monkeypatch.setattr(tempfile, "mkdtemp", boom)
    monkeypatch.setattr(tempfile, "NamedTemporaryFile", boom)

    data = [float(i) for i in range(100)]
    got = reduce(make_iter(data, 10), MeanOp("x"), make_guard(), log=False)
    assert got == pytest.approx(49.5)


# ---------------- cov ----------------

def test_covariance_2d():
    x = [1.0, 2.0, 3.0, 4.0]
    y = [2.0, 4.0, 6.0, 8.0]

    def batches():
        for i in range(0, len(x), 2):
            yield pa.RecordBatch.from_arrays(
                [
                    pa.array(x[i:i + 2], type=pa.float64()),
                    pa.array(y[i:i + 2], type=pa.float64()),
                ],
                names=["x", "y"],
            )

    got = reduce(batches(), CovarianceOp(["x", "y"]), make_guard(), log=False)
    assert got[0, 0] == pytest.approx(5 / 3)
    assert got[1, 1] == pytest.approx(20 / 3)
    assert got[0, 1] == pytest.approx(10 / 3)
    assert got[1, 0] == pytest.approx(10 / 3)


# ---------------- histogram ----------------

def test_histogram():
    data = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    op = HistogramOp("x", bins=2, min_val=0.0, max_val=3.0)
    got = reduce(make_iter(data, 2), op, make_guard(), log=False)
    assert got.sum() == 6
    assert got[0] == 3   # [0, 1.5): 0, 0.5, 1.0
    assert got[1] == 3   # [1.5, 3]: 1.5, 2.0, 2.5


def test_histogram_requires_bounds():
    with pytest.raises(ValueError):
        HistogramOp("x", bins=10)


# ---------------- check_state срабатывает ----------------

def test_state_size_checked():
    # budget=1000 B → state_limit=500 B; CovarianceOp(d=10) = 8+80+800=888 B > 500
    guard = MemoryGuard(max_memory_bytes=1000)
    cols = [f"c{i}" for i in range(10)]
    batch = pa.RecordBatch.from_arrays(
        [pa.array([1.0], type=pa.float64()) for _ in cols], names=cols
    )

    def it():
        yield batch

    with pytest.raises(MemoryLimitExceeded):
        reduce(it(), CovarianceOp(cols), guard, log=False)