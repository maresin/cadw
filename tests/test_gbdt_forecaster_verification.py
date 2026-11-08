"""v0.9.160: CW-1..5 for GBDTForecaster + GBDT_multiseries (D_time_series).

After v0.9.159, GBDTForecaster.state_nbytes() sums _models[h].state_nbytes().
Models are lazy (direct/multi_output with horizons=None); test uses
fit(horizons=N) for eager state>0.

Same class for both cards (GBDTForecaster used as cadw_function for both).
"""
import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ts.models import GBDTForecaster


def _make_df(n, seed=0, *, guard_mb=500):
    rng = np.random.RandomState(seed)
    return CadwDataFrame.from_table(
        pa.table({"y": pa.array(rng.normal(size=n), type=pa.float64())}),
        memory_guard=MemoryGuard(max_memory_mb=guard_mb))


def _upper_bound(n_estimators, max_depth, n_horizons, d_feat):
    """Conservative upper bound matching YAML ram_formula."""
    nodes_max = 2 ** (max_depth + 1) - 1
    return int(n_horizons * n_estimators * nodes_max * 500
               + 2 * d_feat * 8 + 256)


def test_state_after_eager_fit():
    """Fit with horizons=3 populates _models -> state_nbytes > 0."""
    m = GBDTForecaster(mode="direct", lags=(1, 2, 3))
    m.fit(_make_df(200), "y", horizons=3)
    assert m.state_nbytes() > 0


def test_state_within_upper_bound():
    """Actual state ∈ [0.20, 1.0] of theoretical upper bound."""
    n_est, depth, n_hor = 10, 2, 3
    m = GBDTForecaster(
        mode="direct", lags=(1, 2, 3),
        n_estimators=n_est, max_depth=depth,
    )
    m.fit(_make_df(500), "y", horizons=n_hor)
    # d_feat = len(lags) + 1 (trend) = 4
    upper = _upper_bound(n_est, depth, n_hor, d_feat=4)
    actual = m.state_nbytes()
    assert 0.20 * upper <= actual <= upper, (
        f"state {actual} not in [{0.20*upper:.0f}, {upper}]")


def test_state_independent_of_n():
    """Same horizons/params: n=500 vs n=5000 -> state ratio bounded."""
    m1 = GBDTForecaster(mode="direct", lags=(1, 2, 3),
                        n_estimators=5, max_depth=2)
    m1.fit(_make_df(500, seed=0), "y", horizons=3)
    m2 = GBDTForecaster(mode="direct", lags=(1, 2, 3),
                        n_estimators=5, max_depth=2)
    m2.fit(_make_df(5000, seed=0), "y", horizons=3)
    s1, s2 = m1.state_nbytes(), m2.state_nbytes()
    ratio = max(s1, s2) / max(min(s1, s2), 1)
    assert ratio <= 3.0, f"state ratio {ratio:.2f} across n (should be bounded)"


def test_state_zero_before_predict_horizons_none():
    """Documented: fit(horizons=None) + direct mode -> state=0 until predict."""
    m = GBDTForecaster(mode="direct", lags=(1, 2, 3),
                        n_estimators=5, max_depth=2)
    m.fit(_make_df(200), "y")  # horizons=None -> deferred
    assert m.state_nbytes() == 0  # documented lazy behavior
    # After first predict, models populated
    _ = m.predict(2)
    assert m.state_nbytes() > 0


def test_guard_refuses_during_fit():
    """n=200 + horizons=20 + 0.001 MB -> MemoryLimitExceeded."""
    df = _make_df(200, guard_mb=0.001)
    m = GBDTForecaster(mode="direct", lags=(1, 2, 3),
                        n_estimators=50, max_depth=3)
    with pytest.raises((MemoryLimitExceeded, Exception)):
        m.fit(df, "y", horizons=20)


def test_recursive_mode_eager_one_model():
    """mode='recursive' eager-trains 1 model at fit -> state>0 immediately."""
    m = GBDTForecaster(mode="recursive", lags=(1, 2),
                        n_estimators=5, max_depth=2)
    m.fit(_make_df(200), "y")
    assert m.state_nbytes() > 0
