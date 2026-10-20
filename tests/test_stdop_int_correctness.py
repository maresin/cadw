# v0.9.3d: C2 (S2) StdOp int-column correctness.
# Prior bug: int-column mean truncated -> wrong std when batch
# mean non-integral. e.g. [1,2,3,4] int64: buggy sqrt(2),
# correct sqrt(5/3).

import numpy as np
import pyarrow as pa
import pytest

from cadw.core.memory_guard import MemoryGuard
from cadw.ops.reduce import reduce, StdOp


def make_iter(vals, batch_size, type_=pa.int64()):
    for i in range(0, len(vals), batch_size):
        yield pa.RecordBatch.from_arrays(
            [pa.array(vals[i:i+batch_size], type=type_)],
            names=["x"],
        )


def test_std_int_column_correct_known():
    # [1,2,3,4] int64: mean 2.5, M2=5, ddof=1 -> sqrt(5/3).
    # Buggy version: mean truncated to 2, M2=6, sqrt(2).
    data = [1, 2, 3, 4]
    got = reduce(
        make_iter(data, 2), StdOp("x"),
        MemoryGuard(max_memory_mb=10), log=False,
    )
    expected = float(np.std(np.array(data, dtype=np.float64), ddof=1))
    assert got == pytest.approx(expected, rel=1e-12)
    # Sanity: must not equal the buggy truncated value.
    assert not (abs(got - 1.4142135623730951) < 1e-9)


def test_std_int_vs_float_same():
    # Same data as int vs float -> identical std.
    int_data = [1, 2, 3, 4]
    float_data = [1.0, 2.0, 3.0, 4.0]
    got_int = reduce(
        make_iter(int_data, 2, type_=pa.int64()),
        StdOp("x"), MemoryGuard(max_memory_mb=10), log=False,
    )
    got_float = reduce(
        make_iter(float_data, 2, type_=pa.float64()),
        StdOp("x"), MemoryGuard(max_memory_mb=10), log=False,
    )
    assert got_int == pytest.approx(got_float, rel=1e-12)


@pytest.mark.parametrize("data,batch_size", [
    ([1, 2, 3, 4], 2),
    ([1, 2], 1),
    ([0, 0, 0, 10], 2),
    ([3, 7], 2),
    ([10, 20, 30, 40, 50], 3),
])
def test_std_int_matches_numpy(data, batch_size):
    got = reduce(
        make_iter(data, batch_size), StdOp("x"),
        MemoryGuard(max_memory_mb=10), log=False,
    )
    expected = float(np.std(np.array(data, dtype=np.float64), ddof=1))
    assert got == pytest.approx(expected, rel=1e-12)


def test_std_float_unchanged():
    # Regression: float path still works (was always correct).
    data = [1.0, 2.0, 3.0, 4.0]
    got = reduce(
        make_iter(data, 2, type_=pa.float64()),
        StdOp("x"), MemoryGuard(max_memory_mb=10), log=False,
    )
    expected = float(np.std(data, ddof=1))
    assert got == pytest.approx(expected, rel=1e-12)


def test_std_int_with_nulls():
    # Int64 with nulls; nulls skipped, correct ddof=1.
    data = [1, None, 3, 4, None, 6]
    got = reduce(
        make_iter(data, 2), StdOp("x"),
        MemoryGuard(max_memory_mb=10), log=False,
    )
    valid = [1.0, 3.0, 4.0, 6.0]
    expected = float(np.std(valid, ddof=1))
    assert got == pytest.approx(expected, rel=1e-12)


def test_std_int32_type():
    # int32 column too.
    data = [1, 2, 3, 4]
    got = reduce(
        make_iter(data, 2, type_=pa.int32()),
        StdOp("x"), MemoryGuard(max_memory_mb=10), log=False,
    )
    expected = float(np.std(data, ddof=1))
    assert got == pytest.approx(expected, rel=1e-12)


def test_std_ast_cast_to_float64():
    # AST: StdOp.update casts arr to float64 before subtract.
    # Note: `from cadw.ops import reduce as reduce_mod` would
    # bind the FUNCTION `reduce` (via __init__.py re-export),
    # not the module `reduce.py`. Import StdOp class directly.
    import inspect
    from cadw.ops import StdOp
    src = inspect.getsource(StdOp.update)
    assert "pc.cast(arr, pa.float64())" in src
    assert "pa.float64()" in src

