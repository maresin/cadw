"""v0.9.158: bounded_by_domain verification for SARIMA + VAR.

DEC-011: single/multi-series state O(n), n<=1e6 typical.
"""
import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ts.models import SARIMAForecaster
from cadw.ts.var import VARModel


def _make_df_1d(n, seed=0, *, guard_mb=500):
    rng = np.random.RandomState(seed)
    df = CadwDataFrame.from_table(
        pa.table({"y": pa.array(rng.normal(size=n), type=pa.float64())}),
        memory_guard=MemoryGuard(max_memory_mb=guard_mb),
    )
    return df


def _make_df_kd(n, k, seed=0, *, guard_mb=500):
    rng = np.random.RandomState(seed)
    cols = {f"y{i}": pa.array(rng.normal(size=n), type=pa.float64())
            for i in range(k)}
    df = CadwDataFrame.from_table(
        pa.table(cols), memory_guard=MemoryGuard(max_memory_mb=guard_mb))
    return df


# ============================================================================
# SARIMA
# ============================================================================

def test_sarima_state_formula_exact():
    """SARIMA state = 2*n*f64 + 256 (y_raw_ + _y_centered_)."""
    n = 500
    m = SARIMAForecaster(order=(1, 0, 1)).fit(
        _make_df_1d(n, guard_mb=500), "y")
    assert m.state_nbytes() == 2 * n * 8 + 256


def test_sarima_state_grows_linearly():
    n1, n2 = 500, 2000
    m1 = SARIMAForecaster(order=(1, 0, 1)).fit(_make_df_1d(n1), "y")
    m2 = SARIMAForecaster(order=(1, 0, 1)).fit(_make_df_1d(n2), "y")
    ratio = m2.state_nbytes() / m1.state_nbytes()
    assert 3.0 <= ratio <= 5.5, f"ratio={ratio:.2f}"


def test_sarima_guard_refuses_at_domain_bound():
    """n=500_000 + guard=2 MB -> MemoryLimitExceeded (state > 2 MB)."""
    df = _make_df_1d(500_000, guard_mb=2.0)
    m = SARIMAForecaster(order=(1, 0, 1))
    with pytest.raises(MemoryLimitExceeded):
        m.fit(df, "y")


# ============================================================================
# VAR
# ============================================================================

def test_var_state_formula_exact():
    """VAR state = n*k*f64 + k*k*p*f64 + k*f64 + 256 (from state_nbytes)."""
    n, k, p = 500, 2, 1
    df = _make_df_kd(n, k, guard_mb=500)
    m = VARModel(p=p).fit(df, [f"y{i}" for i in range(k)])
    expected = n * k * 8 + k * k * p * 8 + k * 8 + 256
    assert m.state_nbytes() == expected


def test_var_state_grows_linearly():
    n1, n2, k, p = 500, 2000, 2, 1
    m1 = VARModel(p=p).fit(_make_df_kd(n1, k), [f"y{i}" for i in range(k)])
    m2 = VARModel(p=p).fit(_make_df_kd(n2, k), [f"y{i}" for i in range(k)])
    ratio = m2.state_nbytes() / m1.state_nbytes()
    # VAR state dominated by n*k; ratio close to 4 (n1->n2=4x)
    assert 3.0 <= ratio <= 5.5, f"ratio={ratio:.2f}"


def test_var_guard_refuses_at_domain_bound():
    """n=500_000, k=2 + guard=2 MB -> MemoryLimitExceeded."""
    df = _make_df_kd(500_000, 2, guard_mb=2.0)
    m = VARModel(p=1)
    with pytest.raises(MemoryLimitExceeded):
        m.fit(df, ["y0", "y1"])
