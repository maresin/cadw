"""v0.9.189: _collect_last_w streaming buffer (DEC-072 P3)."""
import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard
from cadw.ts.models import _collect_last_w


def _mk_df(y, guard_mb=500):
    return CadwDataFrame.from_table(
        pa.table({"y": pa.array(np.asarray(y, dtype=np.float64))}),
        memory_guard=MemoryGuard(max_memory_mb=guard_mb))


@pytest.mark.parametrize("n", [10, 100, 1000])
def test_collect_last_w_no_truncation(n):
    """n <= W: y_used == full y, lost=None."""
    rng = np.random.RandomState(0)
    y = rng.normal(100, 5, n)
    df = _mk_df(y)
    y_used, n_total, lost = _collect_last_w(df, "y", W=max(2, n + 100))
    assert n_total == n
    assert lost is None
    np.testing.assert_array_equal(y_used, y)


@pytest.mark.parametrize("n,W", [(100, 50), (1000, 100), (1000, 333)])
def test_collect_last_w_truncation(n, W):
    """n > W: y_used == last W, lost=(0, n-W)."""
    rng = np.random.RandomState(0)
    y = rng.normal(100, 5, n)
    df = _mk_df(y)
    y_used, n_total, lost = _collect_last_w(df, "y", W=W)
    assert n_total == n
    assert lost == (0, n - W)
    assert y_used.shape == (W,)
    np.testing.assert_array_equal(y_used, y[-W:])


def test_collect_last_w_exact_w():
    """n == W: no truncation (filled == W, lost=None)."""
    rng = np.random.RandomState(0)
    y = rng.normal(100, 5, 200)
    df = _mk_df(y)
    y_used, n_total, lost = _collect_last_w(df, "y", W=200)
    assert n_total == 200
    assert lost is None
    np.testing.assert_array_equal(y_used, y)


def test_collect_last_w_invalid_w():
    df = _mk_df([1.0, 2.0])
    with pytest.raises(ValueError, match="W must be >= 2"):
        _collect_last_w(df, "y", W=1)
